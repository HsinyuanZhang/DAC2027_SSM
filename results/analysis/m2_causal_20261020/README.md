# FALCON M2，causal-first-B（2026-10-20）

M2 的 5 个 target 都在 source 之后 1–7 天，属于正文的「≤ 30 days」档。此前的数字是 random-draw。这一轮把冻结方法接到 000688 的 causal-first-B 上：校准集是 target session 里按记录顺序的前 B 个 usable trial，query 是其后的 usable trial；prefix 统计；anytime commit；整段 session 一次连续因果解码；query R² 为物理单位。Commit 边界、prefix 守卫、AND gate 与 rebased loop 没有改语义，见 `ssm_decode/causal_protocol.py` 与 `ssm_decode/stream_f7.py`。M2 只加了 adapter（`ssm_decode/m2_causal.py`，`scripts/run_m2_causal.py`）。

Latency 0 是主表。Latency 1 的 target 均值差在 ±0.004 以内（`latency.csv`），下面不单列。

## 间隙

两种算法是同一天，因为每个 pool 最晚的 source session 就是 trunk 的最后训练日。

| target | pool | 距 trunk 最后训练日 | 距最近 source / selection session |
|---|---|---:|---:|
| 2020-10-20 Run1 / Run2 | pool1 | 1（2020-10-19） | 1（2020-10-19 Run2） |
| 2020-10-27 Run1 / Run2 | pool2 | 7（2020-10-20） | 7（2020-10-20 Run2） |
| 2020-10-28 Run1 | pool3 | 1（2020-10-27） | 1（2020-10-27 Run2） |

Selection pool 就是该 target 自己的 pool 的 source session（protocol (a) expanding-window：session 与 trunk 的全部训练日都严格早于 target）。3 个 backbone seed × 2 个 calibrator seed。Source 上的分数打在 trunk 自己的 validation tail（raw trial 的最后 20%），不是前 B 之后的全部 trial：中间那段被 trunk 的梯度用过。

## 主表（latency 0，5 个 target 的均值）

括号外是 target 均值。Target 层面的 95% t 区间（df = 4）大约 ±0.05–0.15，见 `summary.csv` 的 `target_ci95`。Pool 层面的绝对 R² 区间更宽（3 个 pool 的水平差很大，df = 2，B=8 的 P0 是 0.338±0.275），所以下面的比较用配对差，不拿绝对水平的区间当证据。

| 方法 | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior | 0.280 | 0.290 | 0.290 | 0.300 | 0.320 | 0.326 |
| P0 | 0.307 | 0.309 | 0.310 | 0.325 | 0.346 | 0.350 |
| ridge-CCA | −0.047 | 0.034 | 0.069 | 0.161 | 0.227 | 0.302 |
| W_alpha (a) | 0.307 | 0.309 | 0.310 | 0.325 | 0.346 | 0.373 |
| S1（AND） | 0.307 | 0.309 | 0.310 | 0.325 | 0.346 | 0.350 |
| COMBO (a) | 0.307 | 0.309 | 0.310 | 0.325 | 0.346 | 0.373 |
| CCA+S1（AND） | −0.047 | 0.034 | 0.070 | 0.159 | 0.227 | 0.303 |
| R1-stream lr16 | 0.307 | 0.286 | 0.288 | 0.316 | 0.338 | 0.356 |
| R2（参照） | 0.261 | 0.309 | 0.311 | 0.348 | 0.371 | 0.400 |
| R3（参照） | 0.269 | 0.303 | 0.298 | 0.338 | 0.364 | 0.399 |
| P0+ui_reg（参照） | 0.304 | 0.310 | 0.320 | 0.341 | 0.360 | 0.391 |

S1 的 AND 在全部 pool、全部 B 上退回 η = 0，所以 S1 = P0。COMBO 的 AND 同样退回，所以 COMBO = W_alpha。B ≤ 16 时 source 选出的 α = 0，W_alpha = P0。只有 B = 33 三个 pool 都选出 α = 0.25。

相对 P-prior 的 target 配对（正的 target 数）：

| 对比 | 1 | 8 | 16 | 33 |
|---|---|---|---|---|
| P0 − prior | +0.028（5/5） | +0.026（5/5） | +0.025（3/5） | +0.023（3/5） |
| W_alpha − prior | +0.028（5/5） | +0.026（5/5） | +0.025（3/5） | +0.047（5/5） |
| W_alpha − P0 | 0 | 0 | 0 | +0.023（5/5，+0.018…+0.029） |
| R1 lr16 − prior | +0.028（5/5） | +0.017（4/5） | +0.017（4/5） | +0.029（5/5） |
| R1 lr16 − P0 | 0（B=1 规则是 off） | −0.009 | −0.008 | +0.006 |
| ridge-CCA − prior | −0.327（0/5） | −0.139（0/5） | −0.093（0/5） | −0.024（2/5） |

B = 33 的 α = 0.25 在 3 个 pool 上都为正（pool1/2/3：+0.025 / +0.025 / +0.018，相对 P0）。3 个 pool 的均值 +0.023，df = 2 的 t 区间大约 ±0.010，不含 0。这是看过 source 网格之后才打开的那一档，source 模拟又没有跨 session 漂移，不能写成已经确认的增益。B ≤ 16 没有这个项。

## 逐 pool（latency 0）

| pool | 方法 | 1 | 8 | 33 |
|---|---|---:|---:|---:|
| pool1 | P-prior | 0.202 | 0.208 | 0.224 |
| pool1 | P0 / W_alpha（B=8 时 α=0） | 0.207 | 0.210 | 0.220 / 0.245 |
| pool1 | R1 lr16 | 0.207 | 0.204 | 0.247 |
| pool1 | P0+ui_reg | 0.207 | 0.225 | 0.270 |
| pool2 | P-prior | 0.338 | 0.360 | 0.384 |
| pool2 | P0 / W_alpha | 0.362 | 0.402 | 0.428 / 0.453 |
| pool2 | R1 lr16 | 0.362 | 0.391 | 0.414 |
| pool2 | P0+ui_reg | 0.362 | 0.419 | 0.459 |
| pool3 | P-prior | 0.318 | 0.363 | 0.417 |
| pool3 | P0 / W_alpha | 0.398 | 0.403 | 0.454 / 0.472 |
| pool3 | R1 lr16 | 0.398 | 0.392 | 0.456 |
| pool3 | P0+ui_reg | 0.385 | 0.417 | 0.496 |

和 random-draw 那一轮一样：P0 的增益在 pool2/3，pool1 接近 0（B=33 的 P0 − prior = −0.004）。R1 lr16 在 B=33 相对 P0 只在 pool1 为正（+0.027），pool2 为 −0.014，pool3 ≈ 0。完整逐 target 在 `per_target.csv`。

## 门

主门是 AND（增益 ≥ 0.005 且 > 2 SE）。保守门是同一 (η, ρ) 至少在两个 selection session 上增益为正，否则退回基座。

S1 与 COMBO 的 AND 在所有 pool、所有 B 都退回基座（η = 0）。保守门因此没有额外可退的东西，`s1_cons` = S1，`combo_cons` = COMBO，5 个 target 的差都是 0。

CCA+S1 只在 B = 4 部署了 η = 0.3（三个 pool 的正 session 数是 2 / 4 / 4，保守门也放行）。Target 上 CCA+S1 − ridge-CCA = +0.001，5/5 的差在噪声里。其余 B 的 AND 退回 ridge-CCA。两道门部署的图是同一张。

## 与 random-draw 的差（causal − 存储的 draw 均值）

同一 5 个 target。P-prior / P0 / ui_reg 是 3 个 backbone seed；紧凑 R1 的旧行只有 seed 0，而且只到 B ∈ {8, 16, 33}。

| 对比 | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior | −0.006 | −0.012 | −0.025 | −0.018 | −0.004 | −0.003 |
| P0 | +0.011 | −0.014 | −0.026 | −0.016 | −0.005 | −0.006 |
| R1 lr16 | — | — | — | −0.023 | −0.002 | −0.009 |
| ui_reg（prior 初始化，旧超参） | −0.021 | −0.017 | −0.025 | −0.008 | −0.008 | −0.006 |

First-B 在 B = 4 附近比随机抽 trial 低大约 0.02–0.03，B = 33 只低大约 0.005。方向符合「开头的 trial 不如随机的一组代表性好」，量级小于 pool 之间的差。

`ui_reg_legacy` 用的是 2026-10-16 存下来的 (λ, lr, steps)，从 prior 表出发，只是把 trial 换成 first-B。上表这一行才是协议差。`P0+ui_reg` 是 000688 的参照：从该 prefix 的 P0 表出发，惩罚也对着这张 P0 表（`run_traj` 的 centre = init），超参在 causal source episode 上重选。它不是旧 ui_reg 的协议差，B=33 为 0.391，高于 W_alpha 0.373，低于 R2 0.400。

## 适配

**Bias。** 每张安装的表都加上折叠 root `in_proj` bias（source session fold 的均值，与 `best.pt` 的 `in_proj.bias` 差 < 1e-5）。它不乘 α。`shrink_tables` 仍按「W0 无 bias，b_α = α b_cca」使用；cache 里存的是 `b_full − prior_bias`。

**Condition。** Held-in NWB 的 `intervals/trials/tgt_loc` 是 reach target。绝对角几乎落在一个象限里，不能当方向。Condition 是 trial 起点 `finger_pos`（index, mrs）到 `tgt_loc` 的 8-bin 方向，函数仍是 `condition_label`。没有 go cue，所以 align = `trial`，不用 days 那套 `go`。`(phases, ridge)` 在 source validation tail、B ∈ {8, 16, 33} 上选：pool1 = (10, 3)，pool2 = (20, 10)，pool3 = (10, 10)。d = 参考日的 active channel 数（96）。参考日是该 pool 最后一个 source session。Ridge-CCA 在 target 上从 B = 1 的负值升到 B = 33 的 0.302，仍低于 P0，5 个 target 里 B ≤ 16 没有一个超过 prior。

**R1。** 只跑了预算内的 lr16（固定 source 基，r = 16）。(λ, burn) 在 causal first-B 的 source 模拟上重选，off 是候选。B = 1 三个 pool 都选了 off。没有跑 257×257 的完整 Gram。

**R2 / R3。** 前 B 个 trial 的 replay（F7-J），装在 final commit。网格是冻结的 L ∈ {8, 16, 32}、η、ρ、S ∈ {1, 3, 5, 10}。R3 在选中的 R2 表上再选 readout；`full` 超预算，只作为参照出现（例如 pool2 的多数 B）。Jacobian 在 source 上用各 session 自己的 fold（含 bias）估计，selection 时 leave-one-out，target 上用全部 source。

## 没有适配进去的

- **Protocol (b)** 在 pool2 和 pool3 上没有合法 episode：这两个 trunk 的训练集包含已经报告过的 target 日（2020-10-20、2020-10-27）。`w_alpha_b` / `combo_b` 在这 3 个 target 上是 unavailable（行数的 60%），不是用 α = 0 填的。pool1 的 trunk 只训练在 2020-10-19，(a) 和 (b) 选出的 α 相同，所以 pool1 上 (b) = (a)。
- **Go-aligned CCA** 没有 go cue，没有做。Trial-aligned ridge-CCA 做了，不是占位。
- 没有按每张 target 重选 η / ρ。一个 pool 一套规则。
- 没有定点仿真。R1 的字节数沿用 `m2_calib.r1_stream_bytes`（lr16 的 target 侧 working state 1,360 B）。

## 检查

`tests/test_m2_causal.py`，7 个通过。B = 8、J = 4：P-prior、P0、ridge-CCA、W_alpha 的 commit 不读 query，也不读第 J 个 trial 之后的 bin；正对照确认后面的 commit 会动。S1（η = 1）在 GPU 上做了同一件事，并且 η = 0 的最终表等于 P0（含 bias）。Static decode 与存储的 random-draw P0 行对齐：seed 0、pool1、c0、2020-10-20-Run1、B = 8、draw 0（存储 R² 0.1896），容差 1e-3；与 `m2.decode_full` 的差 < 1e-5。Candidate 路径的反传计数在每次构建后核对为 0；ui_reg 在这之后才反传。

## 文件

- 代码：`ssm_decode/m2_causal.py`，`scripts/run_m2_causal.py`，`scripts/analyze_m2_causal.py`，`tests/test_m2_causal.py`
- 行与选择（gitignored）：`results/m2_causal_20261020/rows.jsonl`，`select/`
- 表：`summary.csv`，`per_target.csv`，`per_pool.csv`，`gate_sensitivity.csv`，`protocol_deltas.csv`，`latency.csv`，`selection_rules.csv`
