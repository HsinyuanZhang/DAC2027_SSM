# Causal-first-B 主表（000688，2026-10-19）

部署故事是开机后的校准块，不是随机抽 trial。校准集是 target session 里按记录顺序的前 B 个 usable trial；query 是其后的 usable trial，只在这些 trial 的 valid bin 上、以物理单位算 R²。整段 session 做一次连续因果解码，SSM state 不在 trial 边界重置。trial k 结束时能算出来的表，默认从 usable trial k 的第一个 bin 起生效（latency = 0）；另报 latency = 1 个 usable trial。校准块之前的 bin 用 cold map：electrode prior，mean 0、std 1，不读任何 target bin。Trial 0 允许 one-trial staging：整段 trial 0 只用来做它自己的归一化，表仍是 prior。这是唯一允许读到「当前 bin 之后」的地方，而且不越过 trial 0 的末尾。

B ∈ {1, 2, 4, 8, 16, 33}，2 个 P0 seed。校准集是确定的，方差只来自 seed 和 target。Drift（12 个 target）和 days（2 个 target）的括号是 target-cluster 95% CI；days 只有 2 个 target，这个区间基本就是两个 target 的差别。Month（20140626）和 year（20150511）各 1 个 target，不写 CI。Month 不是另一次解码：它就是 drift 里这一天、同一 month trunk、同一套 month 规则。

原始行在 `results/causal_main_20261019/rows.jsonl`（gitignored）。本目录是汇总：`cells.csv`、`per_target.csv`、`paired.csv`、`latency.csv`、`protocol_delta.csv`、`cost.csv`、`rows_long.csv`、`select/`。

## 1. 每张表什么时候装上

| 方法 | 安装 | trial 0 | commit k（trials ≤ k） | 备注 |
|---|---|---|---|---|
| P-prior | anytime | prior + trial 0 的 norm | 仍是 prior，norm 换成前 k 个 trial | 最终表就是 electrode prior |
| P0 | anytime | prior + trial 0 的 norm | P0(trials ≤ k) | 最终表 = 离线 P0(前 B 个 trial) |
| W_alpha-gap | anytime | 同 P0 | (1−α)P0 + α CCA，α 是计划中的 B 和 gap 箱，不随前缀变 | α=0 时逐位等于 P0 |
| S1-causal | anytime | 同 rebased 的 trial 0（prior） | `rebased_feedback_loop`（e7af2d8）每个 trial 的 snapshot | η=ρ=0 时表就是 P0 |
| combo | anytime | 同 S1 | 基座换成该前缀的 W_alpha，再叠加同一套 Δ | α=0 时就是 S1 |
| R1-stream | anytime | prior | 前缀 P0 + 到此为止的 stream readout ridge | 统计量的前向仍是 P3 的 per-trial 零 state |
| R2、R3、ridge-CCA、P0+ui_reg | 只在块末 | prior + trial 0 norm | 块末一次性装上离线表 | reference；R2/R3/ui_reg 会重放或反传 |

S1 的 F7 step 仍按 e7af2d8 在每个校准 trial 上从零 state 前向（不改方法语义）。报出的 R² 是另一条连续解码，用的是这些已经提交的表。Query 在 latency 0 时全部落在最终表上；中间表只通过进入 query 时的 SSM state 影响分数。

## 2. 选择

- W_alpha 的 α、R1/R2/R3 的 λ/kind/η/ρ/L/S、ui_reg 的 lam/lr/steps：沿用已有 source 文件，协议是 random-draw，不是 first-B。行上标 `selection_protocol=random_draw_legacy`。Days 的 ui_reg 仍是 P2 规则，`like_for_like=false`。
- S1 和 combo 的 (η, ρ) 在 source 上按 causal-first-B 重选（days 10 个 oosc session，month 只有 20140307，year 4 个 oos session，sub-C 各自的 held-out pool session）。预注册门是 OR：配对增益 ≥ 0.005 **或** > 2 SE。灵敏度门是 AND：两个条件都要。并列时 η 靠近该 B 的 R2 规则。Month 每个 B 只有 2 个 seed，没有 draw。
- Days target 的 gap 箱是 ≤14 天，α 全是 0，combo 就是 S1，没有另跑一套 days combo 网格。
- Sub-C 的 α 只在 held-out session 上选。该 session 相对 trunk 是 le14，选出的 α 全是 0，mid/gt180 都退回 le14。因此 sub-C 的 W_alpha-gap 等于 P0，combo 等于 S1。

## 3. 主表（latency 0，query R²）

Reference 行：R2、R3、ridge-CCA、P0+ui_reg。

### days（days trunk，0616/0617）

| 方法 | B=1 | B=8 | B=33 |
|---|---:|---:|---:|
| P-prior | 0.338 [0.259, 0.416] | 0.404 [0.388, 0.419] | 0.382 [0.357, 0.407] |
| P0 | 0.679 [0.641, 0.717] | 0.728 [0.722, 0.735] | 0.743 [0.725, 0.761] |
| W_alpha-gap | 0.679 | 0.728 | 0.743 |
| S1-causal | 0.674 [0.636, 0.712] | 0.723 [0.715, 0.731] | 0.728 [0.703, 0.752] |
| combo | 0.674 | 0.723 | 0.728 |
| R1-stream | 0.672 [0.629, 0.716] | 0.708 [0.705, 0.710] | 0.688 [0.688, 0.689] |
| R2 | 0.684 [0.656, 0.711] | 0.725 [0.724, 0.726] | 0.736 [0.724, 0.748] |
| R3 | 0.681 [0.647, 0.714] | 0.687 [0.670, 0.703] | 0.697 [0.696, 0.698] |
| ridge-CCA | 0.062 [−0.013, 0.137] | 0.651 | 0.734 [0.733, 0.734] |
| P0+ui_reg | 0.698 [0.683, 0.713] | 0.715 [0.711, 0.719] | 0.739 [0.727, 0.751] |

α=0，W_alpha 与 P0 的差是 0。S1 相对 P0：B=1/8/33 为 −0.005 / −0.005 / −0.016（target CI 不含 0）。Days 上这套 causal source 选出的步长没有超过已经很高的 P0。

### month（20140626，单 target）

| 方法 | B=1 | B=8 | B=33 |
|---|---:|---:|---:|
| P-prior | 0.331 | 0.295 | 0.323 |
| P0 | 0.528 | 0.510 | 0.544 |
| W_alpha-gap | 0.479 | 0.551 | 0.604 |
| S1-causal | 0.528 | 0.538 | 0.607 |
| combo | 0.479 | 0.575 | 0.653 |
| R1-stream | 0.531 | 0.514 | 0.563 |
| R2 | 0.531 | 0.542 | 0.622 |
| R3 | 0.520 | 0.540 | 0.620 |
| ridge-CCA | 0.052 | 0.494 | 0.566 |
| P0+ui_reg | 0.515 | 0.534 | 0.629 |

B=1 的 month 规则 η=ρ=0（source 上门没过），S1 就是 P0。W_alpha 在 B=1 低于 P0 0.049（α=0.25）。B=33 时 combo 0.653，高于 W_alpha 0.049、高于 S1 0.046，高于 R2 0.031，高于 ui_reg 0.024。这是一个 target 的前 33 个 trial，不是 12 天的均值。

### year（20150511，year trunk，单 target）

| 方法 | B=1 | B=8 | B=33 |
|---|---:|---:|---:|
| P-prior | 0.147 | 0.118 | 0.124 |
| P0 | 0.148 | 0.243 | 0.287 |
| W_alpha-gap | 0.148 | 0.335 | 0.408 |
| S1-causal | −0.100 | 0.376 | 0.387 |
| combo | −0.100 | 0.389 | 0.423 |
| R1-stream | 0.148 | 0.220 | 0.339 |
| R2 | 0.124 | 0.393 | 0.361 |
| R3 | 0.123 | 0.398 | 0.393 |
| ridge-CCA | −0.019 | 0.318 | 0.375 |
| P0+ui_reg | 0.130 | 0.421 | 0.464 |

B=1 的 gt180 箱 α=0。OR 门把 η 选成 3，两个 seed 的 query R² 都是 −0.10。AND 门在这个 B 上退回基座，S1 和 combo 都回到 0.148（P0）。B=8/33 两道门选了同一个 (η, ρ)。B=33 的 combo 0.423，高于 W_alpha 0.015、高于 S1 0.036，低于 ui_reg 0.041。

### drift（12 个 target，month trunk，month 规则）

| 方法 | B=1 | B=8 | B=33 |
|---|---:|---:|---:|
| P-prior | 0.106 [0.066, 0.156] | 0.104 [0.066, 0.150] | 0.106 [0.066, 0.157] |
| P0 | 0.181 [0.115, 0.258] | 0.209 [0.144, 0.284] | 0.215 [0.143, 0.298] |
| W_alpha-gap | 0.155 [0.088, 0.232] | 0.282 [0.217, 0.355] | 0.342 [0.271, 0.414] |
| S1-causal | 0.181 [0.115, 0.262] | 0.276 [0.211, 0.345] | 0.353 [0.281, 0.420] |
| combo | 0.155 [0.089, 0.233] | 0.336 [0.268, 0.403] | 0.434 [0.359, 0.503] |
| R1-stream | 0.177 [0.111, 0.256] | 0.212 [0.147, 0.285] | 0.252 [0.178, 0.333] |
| R2 | 0.200 [0.127, 0.281] | 0.346 [0.279, 0.408] | 0.406 [0.336, 0.467] |
| R3 | 0.200 [0.129, 0.282] | 0.347 [0.285, 0.406] | 0.411 [0.341, 0.473] |
| ridge-CCA | −0.029 [−0.102, 0.029] | 0.329 [0.277, 0.379] | 0.409 [0.351, 0.463] |
| P0+ui_reg | 0.199 [0.131, 0.280] | 0.308 [0.244, 0.370] | 0.450 [0.381, 0.512] |

配对差（target-cluster CI），latency 0：

- B=33：W_alpha−P0 = +0.127 [0.101, 0.152]；S1−P0 = +0.137 [0.105, 0.169]；combo−W_alpha = +0.092 [0.068, 0.116]；combo−S1 = +0.081 [0.066, 0.095]。Combo 高于两张基座，仍低于 ui_reg（0.434 对 0.450）和大约持平于 R2（0.406）之上、R2 的 CI 与 combo 的 CI 重叠。R1-stream−P0 只有 +0.037 [0.026, 0.048]。
- B=8：combo−W_alpha = +0.053 [0.039, 0.068]；S1−P0 = +0.068 [0.051, 0.084]；S1−W_alpha = −0.006 [−0.021, 0.008]（含 0）。
- B=1：month 规则 η=0，S1=P0；α=0.25 的 W_alpha 低于 P0 0.025 [−0.034, −0.016]，combo 因为退回 W_alpha，和 W_alpha 的差是 0。

### sub-C month（按 backbone，不要把两套 backbone 合成一个窄 CI）

α=0，W_alpha = P0，combo = S1。AND 门在 sc_0629 的 B≥4 全部退回 P0；sc_0708 除 B=1 外也退回。

| backbone | 方法 | B=1 | B=8 | B=33 |
|---|---|---:|---:|---:|
| sc_0629 | P0 | 0.097 | 0.179 | 0.218 |
| sc_0629 | S1 OR | 0.097 | 0.220 | 0.269 |
| sc_0629 | S1 AND | 0.097 | 0.179 | 0.218 |
| sc_0629 | R2 | 0.102 | 0.254 | 0.307 |
| sc_0708 | P0 / S1（B≥4 为 fallback） | 0.307 / 0.296 | 0.428 | 0.421 |
| sc_0708 | R2 | 0.317 | 0.466 | 0.492 |

sc_0629 上 OR 门的 S1 在 B=8/33 比 P0 高 0.041 / 0.052（6 个 session 的 target CI 不含 0），仍低于 R2。AND 门把这层打回 P0。两套 backbone 合成的「sub-C 均值」会把 sc_0708 的 fallback 和 sc_0629 的小增益混在一起，正文不用那个合并数。

## 4. 门和 latency

OR 对 AND（query R² 之差，OR − AND）。Drift 上 month 规则两道门选了同一个 (η, ρ)，差是 0。Days B=1 的 AND 比 OR 高 0.004（AND 的 η 更小）。Year B=1 的 AND 把 −0.100 拉回 P0 的 0.148，差是 0.248；B=8/33 差是 0。Sub-C 的 sc_0629 在 B=8/33 上 AND−OR = −0.041 / −0.052，也就是整段增益被门槛拿掉。

Latency = 1 减 latency = 0：全部 stratum、全部主方法里绝对值最大的是 days ridge-CCA、B=33 的 −0.005。Drift 上 P0 / W_alpha / S1 / R2 的差在 0.002 以内，多数 B 是 0.000。多等一个 trial 再装表，几乎不改变 query R²：latency 0 时 query 已经全在最终表上，SSM 记不住多出来的那一个 trial。

## 5. 和 random-draw 协议比

随机抽 trial 的存档均值减去这里的因果均值，符号是「因果 − 随机」。不是同一批 draw 的配对。R1/R2 的随机行来自 drift curve；W_alpha / P0 / ridge-CCA / ui_reg / R3 来自 P0-Solve 的 `cells.csv`；S1 来自 stream_f7 的 rebased 行（没有 days）。

Drift、B=33：R2 的差是 +0.000，R3 −0.000，ui_reg −0.000，P0 +0.005，W_alpha +0.008。B=8 的 R2 是 +0.001。大 B 上，换不换「最早的 B 个 trial」几乎不动这些最终表的 query R²。B=1 的差大约 +0.02 到 +0.03（P0、R2、S1），最早的那一个 trial 比随机的一个 trial 好一点。

Month 这一个 target 动得多：B=8 的 S1 是 −0.086（因果 0.538，随机 rebased 0.624），W_alpha −0.061，ui_reg −0.089。20140626 的前 8 个 trial 比随机 8 个 trial 差。这和 12 天均值在 B=33 上不动，不矛盾。

Year B=1 的 S1 差是 −0.282（因果 OR 门的 η=3 对上了 −0.100，随机 rebased 是 0.181）。这是门和 first-B source 选择造成的，不是解码写错：同一行的 AND 门等于 P0。

Days 没有随机 S1。P0 的差是 B=1 +0.032、B=8 +0.005、B=33 +0.006。

## 6. 代价（解析，B=8、T=200 的点估计；trunk state 不含在 SRAM 里）

数字来自 `population_calib.mac_counts`、`p0solve.cost_model`、`stream_f7.fir_bin_macs` / `like_for_like_memory`、`refine_cost.refinement_cost`。MAC/bin 不含冻结 trunk 的推理。完整两列 N 在 `cost.csv`。

| 方法 | N=26 MAC/bin | MAC/commit | SRAM | ROM | replay |
|---|---:|---:|---:|---:|---|
| P-prior | 1.8e3 | 9e1 | 6.4 KiB | prior 1.0 KiB | 否 |
| P0 | 1.8e3 | 2.0e5 | 6.4 KiB | calibrator 31 KiB | 否 |
| W_alpha-gap | 2.1e3 | 2.0e5 | 7.7 KiB | calibrator 31 KiB | 否 |
| S1-causal | 1.1e4 | 2.0e5 | 13 KiB | calibrator+J̄ 32 KiB | 否 |
| combo | 1.1e4 | 2.0e5 | 14 KiB | 同 S1 | 否 |
| R1-stream | 9.9e3 | 2.5e5 | 39 KiB | — | 否 |
| R2 | 1.2e3（每遍、不含 trunk） | 8e2 | 25 KiB | J̄ 0.6 KiB | 是 |
| ridge-CCA | 3.0e2 | 6.7e3 | 2.5 KiB | — | 否 |
| P0+ui_reg | — | — | replay 22 KiB | calibrator 31 KiB | 是（反传） |

N=256 的 P0 SRAM（对称 Gram）约 177 KiB，S1 的 3-bit ring 版 SRAM 约 230 KiB，ROM 仍是那一份 calibrator。W_alpha 的 per-bin 里，CCA 矩用了 p0solve 的 raw-moment 计数当替身，没有单独的 CCA 代价模型。R3 是 R2 再加一遍 readout，replay 同样是「是」。

## 7. 没有做

- Sub-C 的 ridge-CCA。`run_method` 把参考 session 的 NWB 拼到了 `sub-M/` 路径下（`sub-C_ses-CO-20150313`），文件打不开。α 网格里 CCA 缺失的格子记成 −1，选出的 α 全是 0。没有改 published baseline 的读文件路径。
- Sub-C 的 P0+ui_reg。Round-1 没有 like-for-like 的 ui_reg 选择，没有发明一套步数。
- Sub-C 的 B=2。Round-1 的 source 规则没有这个 B。
- S1 的 staged 模式。主表只要 e7af2d8 的 rebase。
- 把 legacy 的 R1/R2/R3/α/ui_reg 选择重跑成 first-B。这些行标明了 random-draw。S1 和 combo 重跑了。
- 定点累加器仿真。代价是解析的。
- Days 单独的 combo source 网格。Target 上 α=0，combo 与 S1 相同。

## 8. 代码

`ssm_decode/causal_protocol.py`（调度、因果审计、OR/AND 门、batched rebased 网格），`scripts/run_causal_main.py`，`scripts/analyze_causal_main.py`，`tests/test_causal_main.py`。`rebased_feedback_loop` 加了可选的 `history` 和 `base_bias`，不传时行为和 e7af2d8 一样。
