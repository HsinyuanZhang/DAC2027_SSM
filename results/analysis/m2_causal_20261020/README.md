# FALCON M2，causal-first-B（2026-10-20）

## 审查后修正

独立审查之后，主表的 α 不再只用本 pool 的 source。000688 的 `alpha_episode_allowed` 允许任何「session 与 trunk 都严格早于 target」的 episode，包括别的 trunk 上、该 trunk 没见过的 session（pool1 trunk 在 10-20 上的 first-B，给 10-27 / 10-28 投票；pool2 trunk 在 10-27 上的 first-B，给 10-28 投票）。这是现在的主规则（`expanding`）。旧的 in-pool 规则留在 `w_alpha_inpool`。

CCA 的参考日（每个 trunk 最后一个 source）的 validation tail 进了 `fit_anchor`，所以这一轮它不再给自己的 α、CCA 配置和 CCA+S1 的 (η, ρ) 投票。拿掉它之后，(phases, ridge) 没变：pool1 (10, 3)，pool2 (20, 10)，pool3 (10, 10)。

B=33 的 W_alpha − P0 随选择而变，范围 **+0.010 到 +0.023**：

| 变体 | B=33 的 α | W_alpha − P0 |
|---|---|---:|
| 主规则：跨 trunk，anchor 不投票 | 5 个 target 都是 0.25 | +0.023（5/5，+0.018…+0.029） |
| in-pool，anchor 仍投票（上一轮主表） | 同样全是 0.25 | +0.023 |
| in-pool，anchor 不投票 | 只有 pool1 是 0.25，pool2/3 退回 0 | +0.010 |

审查复算的「pooled、anchor 仍投票」在 B=16 是 0.353（pool1 的 α 仍是 0），与 `w_alpha_inpool` 的 0.346 不是同一列：in-pool 在 B=16 全是 α=0。跨 trunk 且 anchor 仍投票时，pool2/pool3 在 B=16 变成 0.25、pool1 仍是 0，5 个 target 的均值是 0.353。主规则再拿掉 anchor 之后，pool1 在 B=16 也变成 0.25，均值是 **0.361**。B=8 在这些变体里 α 都是 0（pool2 的 source 均值 0.465 对 0.462，0.25 没有翻过去），所以是 0.325，不是 0.329。

Trunk 和 P0 的 checkpoint 都是在 source validation tail 上选的，门也打在同一段 tail 上。这是 checkpoint-selected，不只是「尾部没有被梯度用过」。

S1 的 AND 退回是对的：在 target 上直接扫 η（ρ=0），S1 − P0 的均值 ≤ 0（B=8 最大单格也只有约 +0.007，均值约 0）。同一批 label 上，replay 的 R2 在 B=8 相对 prior +0.048，P0+ui_reg +0.041。单次前向的 refinement 不刷新残差，所以步长帮不上；多步 replay 和梯度参照仍看得到 label。

统计单位是 pool（n=3）。pool1 的两个 target、pool2 的两个 target 是同一天的 Run1/Run2，共用一个 trunk。配对差在 `paired_pool.csv`，区间是 3 个 pool 的 t 区间（df=2）。6 个 budget × 11 个方法没有做多重比较校正。

## 主表（latency 0，5 个 target）

α 用上面的主规则。S1 / COMBO 的 AND 仍是 η=0（B=16 在 α=0.25 上重选过，anchor 不投票，三个 pool 都退回），所以 S1 = P0，COMBO = W_alpha。Latency 1 的差仍在 ±0.004 以内。

| 方法 | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior | 0.280 | 0.290 | 0.290 | 0.300 | 0.320 | 0.326 |
| P0 | 0.307 | 0.309 | 0.310 | 0.325 | 0.346 | 0.350 |
| ridge-CCA | −0.047 | 0.034 | 0.069 | 0.161 | 0.227 | 0.302 |
| W_alpha（跨 trunk，anchor 除外） | 0.307 | 0.309 | 0.310 | 0.325 | 0.361 | 0.373 |
| W_alpha in-pool（敏感度） | 0.307 | 0.309 | 0.310 | 0.325 | 0.346 | 0.373 |
| S1（AND） | 0.307 | 0.309 | 0.310 | 0.325 | 0.346 | 0.350 |
| COMBO | 0.307 | 0.309 | 0.310 | 0.325 | 0.361 | 0.373 |
| CCA+S1（AND，anchor 不投票） | −0.047 | 0.034 | 0.075 | 0.159 | 0.227 | 0.303 |
| R1-stream lr16 | 0.307 | 0.286 | 0.288 | 0.316 | 0.338 | 0.356 |
| R2（参照） | 0.261 | 0.309 | 0.311 | 0.348 | 0.371 | 0.400 |
| R3（参照） | 0.269 | 0.303 | 0.298 | 0.338 | 0.364 | 0.399 |
| P0+ui_reg（参照） | 0.304 | 0.310 | 0.320 | 0.341 | 0.360 | 0.391 |

W_alpha − P0 的 pool 配对（3 个 pool 的均值 ± 95% t 区间）：B=8 是 0（α=0）；B=16 是 +0.015±0.009（3/3 为正）；B=33 是 +0.023±0.010（3/3 为正）。B=33 相对 prior 是 +0.048±0.061，区间含 0：绝对水平被 pool1 拉低。没有对 6×11 做校正。

## 间隙

两种算法仍是同一天：最晚的 source 就是 trunk 的最后训练日。pool1 与 pool3 为 1 天，pool2 为 7 天。

## 逐 pool（latency 0，B=8 / 16 / 33）

| pool | 方法 | 8 | 16 | 33 |
|---|---|---:|---:|---:|
| pool1 | P-prior | 0.208 | 0.232 | 0.224 |
| pool1 | P0 | 0.210 | 0.230 | 0.220 |
| pool1 | W_alpha | 0.210 | 0.249 | 0.245 |
| pool2 | P-prior | 0.360 | 0.379 | 0.384 |
| pool2 | P0 | 0.402 | 0.421 | 0.428 |
| pool2 | W_alpha | 0.402 | 0.434 | 0.453 |
| pool3 | P-prior | 0.363 | 0.380 | 0.417 |
| pool3 | P0 | 0.403 | 0.426 | 0.454 |
| pool3 | W_alpha | 0.403 | 0.439 | 0.472 |

P0 − prior 在 B=8 的三个 pool 是 +0.002 / +0.042 / +0.040。增益在 pool2/3，pool1 ≈ 0。

## 门

η=ρ=0 的分数和基座只差浮点噪声，不再记成正 session，`session_gate_deploy` 在退回时是 false。

S1 与 COMBO 的 AND 在所有 pool、所有 B 都退回。保守门没有额外可退的步，两道门的图相同。

CCA+S1：拿掉 anchor 之后，pool1 的 B=4 从 η=0.3 退回 ridge-CCA；pool2/pool3 的 B=4 仍是 η=0.3，且 ≥2 个 session 为正，保守门也放行。这三天相对 ridge-CCA 约 +0.021 / +0.005 / +0.004。其余 B 退回 ridge-CCA。

## 与 random-draw 的差

同一组 seed。R1 lr16 的旧行只有 seed 0，因果侧也只用 seed 0。Random-draw 的 query 含有 session 开头的 trial；causal 的 query 只有前 B 个 trial 之后的 trial。

| 对比 | seeds | 8 | 33 |
|---|---|---:|---:|
| P-prior | 0,1,2 | −0.018 | −0.003 |
| P0 | 0,1,2 | −0.016 | −0.006 |
| R1 lr16 | 0 | −0.016 | −0.012 |
| ui_reg（prior 初始化，旧超参） | 0,1,2 | −0.008 | −0.006 |

`P0+ui_reg` 从 P0 表出发、惩罚也对着这张表，不是上面的协议差。

## 适配与没放进去的

Bias、8-bin condition（`tgt_loc` 相对 trial 起点；没有 go cue，所以 align=`trial`）、lr16、R2/R3 replay 与上一轮相同。

Protocol (b) 现在可以跨 trunk：pool2/pool3 用 pool1 trunk 在 10-19 上的 episode（那个 trunk 没在已报告的 target 日上训练过）。因此 `w_alpha_b` 不再是 unavailable；在这套数据上它和主规则选出的 α 相同（B=16/33 为 0.25）。

没有做的：按每张 target 重选 η/ρ；把 anchor 只拟合在 train 区（投票时排除它已经是审查点的另一条合法做法，范围见上表）；定点仿真；完整 257×257 Gram。

## 检查

`tests/test_m2_causal.py`，9 个通过。B=8：P-prior / P0 / ridge-CCA / W_alpha 的 prefix 扰动；S1 的 η=0 走 loop 而不是短路，最终表等于 P0（含 bias）；COMBO（α=0.5）和 CCA+S1 的 prefix 扰动。Parity 覆盖存储的 random-draw P0 行（每个 seed×pool 抽 2 条，外加原先那条 B=8），最大差 < 1e-3。

## 文件

- 代码：`ssm_decode/m2_causal.py`，`scripts/run_m2_causal.py`，`scripts/analyze_m2_causal.py`，`tests/test_m2_causal.py`
- 行（gitignored）：`results/m2_causal_20261020/rows.jsonl`，`select/`（含 unseen episode 与 `combo_b16_alpha025.jsonl`）
- 表：`summary.csv`，`per_target.csv`，`per_pool.csv`，`paired_pool.csv`，`gate_sensitivity.csv`，`protocol_deltas.csv`，`latency.csv`，`selection_rules.csv`
