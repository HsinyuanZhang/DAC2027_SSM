# Causal-first-B 主表（000688，2026-10-19）

## 审查后修正

独立审查确认 3cc6fa0 的数字没有泄漏，但有四件事要改。下面的表以这次重跑为准。上一版只展示 B=1/8/33，把 B=16 上旧 COMBO（0.359）低于 R2（0.396）、ridge-CCA（0.387）和 ui_reg（0.410）藏掉了。那个差距主要是 α 没选好，不是反馈步本身。

1. **α 按 causal-first-B 重选**，网格 {0, 0.25, 0.5, 0.75, 1}。每个 trunk 的因果 OOS 很少：month 只有 20140307（le14），days 的 oosc 全是相对 20150615 的 gt180，所以 per-trunk 的 le14/mid 会退回仅有的那个箱子。另外预注册一条 **pooled** 规则，把 days、month、year、sc_0629、sc_0708 的 source episode 合在一起，同时给出 α(B, gap) 和 α(B)。主表的 W_alpha 和 COMBO 用 pooled 的 α(B, gap)；`w_alpha_trunk` 和 `w_alpha_B` 是对照。Pooled 的 gt180 在 B=8/16/33 是 0.75，不再是旧规则退回 mid 之后的 0.25。
2. **ridge-CCA 是可部署的**（不重放、不反传、只用 first-B）。它不在 reference 里。
3. **COMBO 的门相对自己的基座（W_alpha），不是相对 P0**。网格含 η=0。主门从现在起是 AND（增益 ≥ 0.005 且 > 2 SE）。OR 只作灵敏度。Year B=1 的 OR 仍会选出 η=3，query R² = −0.100；AND 退回基座。
4. **CCA+S1**：基座是前缀 ridge-CCA，上面加同一套 S1 步，门相对 CCA 而不是 P0。问的是：固定反馈还能不能加在最强的闭式可部署基线上。
5. **Sub-C 路径**。`trial_meta` 以前把参考 NWB 拼到 `sub-M/`。改成按 session 的 subject 目录打开。上一版 sub-C 的 α=0 是这个 bug，不是 gap 规则。Sub-C 的 ridge-CCA、W_alpha、COMBO、CCA+S1 已重跑。

S1 的 (η, ρ) 仍是上一轮 first-B source 网格（当时还没有把 η=0 放进网格；门不过时 fallback 本来就是 η=ρ=0，所以 AND 的部署点不变）。COMBO 和 CCA+S1 用的是含 η=0 的新网格。Source 选择只给最终表打分，不模拟 commit 时刻表。S1/COMBO/CCA+S1 的 step 是每个校准 trial 一次从零 state 的 trunk 前向，和报 R² 的那条连续解码不是同一次前向；代价表里这条多出来的 trunk pass 单列了。

统计：drift 的 12 个 target 共用一个 month backbone、同一个 selection session。Month 就是其中的 20140626，不是独立重复。Year 是 1 个 target。Days 是 2 个 target，target-cluster 区间没有意义。Seed spread 在 `per_target.md`（B=8 那一列）。CI 仍按 target session 聚类。

扰动测试 `CAUSAL_LEAK=1 pytest tests/test_causal_main.py::test_real_query_and_prefix_perturbation` 通过：query 之后的 bin，以及块内某个前缀之后的 bin，都不会改变还不允许读它们的 commit。覆盖 P0、W_alpha、S1、COMBO（含 η=0 的 bias 与闭式基座一致）、CCA+S1、R1、R2、R3、ui_reg。

## Drift（12 target，month trunk；latency 0，query R²）

| 方法 | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior | 0.106 | 0.098 | 0.104 | 0.104 | 0.106 | 0.106 |
| P0 | 0.181 | 0.182 | 0.201 | 0.209 | 0.218 | 0.215 |
| W_alpha trunk | 0.181 | 0.182 | 0.232 | 0.282 | 0.317 | 0.422 |
| W_alpha pooled | 0.181 | 0.182 | 0.229 | 0.348 | 0.405 | 0.438 |
| W_alpha α(B) | 0.181 | 0.182 | 0.232 | 0.330 | 0.405 | 0.436 |
| ridge-CCA | −0.029 | 0.080 | 0.169 | 0.329 | 0.387 | 0.409 |
| S1（AND） | 0.181 | 0.182 | 0.267 | 0.276 | 0.313 | 0.353 |
| COMBO（AND，pooled α） | 0.181 | 0.182 | 0.281 | 0.380 | 0.403 | 0.471 |
| CCA+S1（AND） | −0.029 | 0.080 | 0.195 | 0.343 | 0.427 | 0.450 |
| R1-stream | 0.177 | 0.179 | 0.202 | 0.212 | 0.234 | 0.252 |
| R2 | 0.200 | 0.227 | 0.273 | 0.346 | 0.396 | 0.406 |
| R3 | 0.200 | 0.227 | 0.274 | 0.347 | 0.396 | 0.411 |
| P0+ui_reg | 0.199 | 0.220 | 0.260 | 0.308 | 0.410 | 0.450 |

B=16 不再被藏：重选 α 之后 COMBO 0.403，R2 0.396，ridge-CCA 0.387，ui_reg 0.410。COMBO 高于 R2 和 ridge-CCA，仍低于 ui_reg。CCA+S1 在 B=16 是 0.427，高于这三行。

配对（COMBO 或 CCA+S1 减对方，target-cluster 95% CI）：

| 对比 | B=8 | B=16 | B=33 |
|---|---:|---:|---:|
| W_alpha pooled − P0 | +0.139 [0.105, 0.172] | +0.188 [0.140, 0.232] | +0.223 [0.172, 0.271] |
| W_alpha pooled − ridge-CCA | +0.018 [0.005, 0.031] | +0.019 [0.009, 0.028] | +0.029 [0.021, 0.038] |
| COMBO − W_alpha pooled | +0.033 [0.021, 0.045] | −0.002 [−0.017, 0.012] | +0.033 [0.023, 0.044] |
| COMBO − ridge-CCA | +0.051 [0.037, 0.065] | +0.016 [0.000, 0.031] | +0.062 [0.049, 0.074] |
| CCA+S1 − ridge-CCA | +0.014 [−0.028, 0.046] | +0.040 [0.023, 0.056] | +0.041 [0.029, 0.053] |

B=16 上 COMBO 相对自己的基座没有增益（区间含 0）。CCA+S1 在 B=16 和 B=33 相对 ridge-CCA 有增益，B=8 没有。旧的 COMBO−ridge-CCA（+0.006 / −0.028 / +0.025）是 α=0.25 那一版，不要和这张表混用。

Drift 上 S1 的 OR 和 AND 选了同一个 (η, ρ)，差是 0。Year B=1 的 OR 是 −0.100，AND 是 0.148（P0）。COMBO 的 OR 在 year B=1 同样是 −0.100，AND 留在 0.148。CCA+S1 的 OR 在 year B=1 是 −0.335，AND 退回 ridge-CCA（−0.019）。

## Days / month / year

Days 的 per-trunk OOS 没有 ≤14 天的 episode，le14 退回 gt180，B=8 的 α=0.75，W_alpha trunk 0.657，低于 P0 0.728。Pooled 的 le14 在 B=8 是 0.25（0.720），B=33 是 0.5（0.690），仍低于 P0 0.743。近间隔上这次 α 比旧的「le14 一律 0」更激进，days 会受伤。

| 层 | 方法 | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---|---:|---:|---:|---:|---:|---:|
| days | P0 | 0.679 | 0.703 | 0.722 | 0.728 | 0.734 | 0.743 |
| days | W_alpha pooled | 0.679 | 0.703 | 0.695 | 0.720 | 0.724 | 0.690 |
| days | COMBO | 0.678 | 0.703 | 0.665 | 0.715 | 0.724 | 0.692 |
| days | CCA+S1 | 0.062 | 0.215 | 0.491 | 0.647 | 0.681 | 0.746 |
| days | ridge-CCA | 0.062 | 0.214 | 0.475 | 0.651 | 0.668 | 0.734 |
| days | R2 | 0.684 | 0.701 | 0.728 | 0.725 | 0.732 | 0.736 |
| month | P0 | 0.528 | 0.465 | 0.497 | 0.510 | 0.530 | 0.544 |
| month | W_alpha pooled | 0.528 | 0.465 | 0.516 | 0.537 | 0.583 | 0.626 |
| month | COMBO | 0.528 | 0.465 | 0.545 | 0.569 | 0.557 | 0.658 |
| month | CCA+S1 | 0.052 | 0.194 | 0.454 | 0.534 | 0.591 | 0.617 |
| month | ui_reg | 0.515 | 0.450 | 0.482 | 0.534 | 0.594 | 0.629 |
| year | P0 | 0.148 | 0.170 | 0.216 | 0.243 | 0.284 | 0.287 |
| year | W_alpha pooled | 0.148 | 0.170 | 0.220 | 0.335 | 0.369 | 0.408 |
| year | COMBO AND | 0.148 | 0.189 | 0.247 | 0.389 | 0.405 | 0.423 |
| year | CCA+S1 AND | −0.019 | 0.043 | 0.226 | 0.364 | 0.387 | 0.408 |
| year | ui_reg | 0.130 | 0.171 | 0.298 | 0.421 | 0.440 | 0.464 |

Month、year 各一个 target，没有 CI。完整格子和 seed spread 在 `per_target.md`、`cells.csv`。

## Sub-C month（路径修好之后，按 backbone）

| backbone | 方法 | 1 | 4 | 8 | 16 | 33 |
|---|---|---:|---:|---:|---:|---:|
| sc_0629 | P0 | 0.097 | 0.158 | 0.179 | 0.200 | 0.218 |
| sc_0629 | W_alpha pooled | 0.097 | 0.151 | 0.212 | 0.281 | 0.357 |
| sc_0629 | ridge-CCA | −0.052 | 0.081 | 0.172 | 0.242 | 0.301 |
| sc_0629 | COMBO | 0.097 | 0.151 | 0.212 | 0.281 | 0.357 |
| sc_0629 | CCA+S1 | −0.024 | 0.149 | 0.268 | 0.292 | 0.383 |
| sc_0629 | R2 | 0.102 | 0.180 | 0.254 | 0.290 | 0.307 |
| sc_0708 | P0 | 0.307 | 0.407 | 0.428 | 0.429 | 0.421 |
| sc_0708 | W_alpha pooled | 0.307 | 0.438 | 0.456 | 0.519 | 0.535 |
| sc_0708 | ridge-CCA | 0.000 | 0.340 | 0.425 | 0.512 | 0.553 |
| sc_0708 | COMBO | 0.296 | 0.438 | 0.456 | 0.519 | 0.535 |
| sc_0708 | CCA+S1 | 0.000 | 0.220 | 0.465 | 0.534 | 0.573 |
| sc_0708 | R2 | 0.317 | 0.439 | 0.466 | 0.472 | 0.492 |

sc_0629 的 COMBO 在 AND 门下几乎全程 fallback，所以等于 W_alpha，不等于 P0（α 不再是 0）。CCA+S1 在 B=33 是 0.383，高于 R2 的 0.307。sc_0708 的 CCA+S1 在 B=8/16/33 高于 ridge-CCA，也高于 R2。没有 sub-C 的 ui_reg（没有 like-for-like 选择），也没有 B=2（round-1 没有这个 B 的规则）。

## 代价

`cost.csv`。S1 / COMBO / CCA+S1 的 MAC/bin 现在含一次校准 bin 上的零 state trunk 前向（和推理那次分开）。N=26 时这条 trunk pass 是大头。SRAM / ROM 的拆分和上一版一样：calibrator 权重算 ROM，3-bit ring 的 S1 在 N=26 大约 13 KiB SRAM + 32 KiB ROM，不重放。ridge-CCA 和 W_alpha 不重放、不反传。R2/R3/ui_reg 重放或反传，仍是 reference。

## 没有做

- 把 R1/R2/R3/ui_reg 的超参重选成 first-B。它们仍是 random-draw legacy。
- Days 上一个只含近间隔 episode 的 α。现有因果 OOS 里没有，pooled 用了别的 trunk 的 le14，B=33 仍伤 P0。
- 定点仿真。
- S1 网格补 η=0 之后重跑。Fallback 已经是 η=0，AND 的部署点不变。

代码：`ssm_decode/causal_protocol.py`（`measured_read_bin`，门的 `base_key`），`scripts/run_baselines_readin.py`（sub-C 路径），`scripts/run_causal_main.py`，`scripts/analyze_causal_main.py`，`tests/test_causal_main.py`。原始行 `results/causal_main_20261019/rows.jsonl`。
