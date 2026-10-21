# H4：S1 live residual（2026-10-21）

问题：S1 的残差现在来自每个 calibration trial 的零状态前向（L0，`rebased_feedback_loop`）。部署推理是整段连续、trunk state 跨 trial 携带。若 live residual 下增益还在，校准不必再跑一次 shadow forward。

这是 fp32 算法检查，不是 rule 8 的逐次取整定点仿真。Trunk 冻结，target 不反传。Selection 只在 source 上，protocol (a) expanding window，主门是 AND；conservative 门（被选中的 (η, ρ) 至少在 2 个 selection session 上增益为正，否则 η=0）一并报告。Latency 0 是主行。

代码：`ssm_decode/live_residual.py`（L0 语义未改）。L1 = state 与 FIR ring 都跨 trial 携带。L2 = state 携带，但每个 calibration trial 开始时 ring 清零。评分仍是 causal-first-B 的整段 contiguous decode（表在 commit 边界切换）。

## 结论

**S1-L1 保持 31–180 d 的增益，不需要 shadow forward。** Round-2 这一层 B4–33，L0 − P0 是 **+0.055 [+0.034, +0.076]（16/17）**，与已发表的 +0.055 [+0.034, +0.076] 一致。L1 − P0 是 **+0.058 [+0.035, +0.082]（14/17）**。L1 − L0 是 +0.003 [−0.002, +0.009]（11/17），区间含 0。L2 − P0 是 +0.057 [+0.033, +0.080]（14/17）：ring 跨 trial 携带不是增益的来源，state 携带也没有把它吃掉。

≤30 d：L1 不伤害 P0。B4–33 L1 − P0 **+0.009 [+0.005, +0.014]（7/8）**；L0 − P0 是 +0.003 [−0.002, +0.009]（5/8），与 round-2 表一致。唯一略负的 session 是 20150716（约 −0.002）。

## 残差怎么取

Latency 0 的安装时刻与 `commit_segments` 相同。Trial 0 仍是文档里的 one-trial staging：自己的 normalisation，表是 prior。Trial k>0 用 P0 of trials < k 加上此前的 Delta。

Trial 之间、以及第一个 usable trial 之前的 bin 会解码（trunk state 往前走），但 residual 强制为 0：它们不是 calibration label，不往 U 里加项。L1 把这些 bin 的 normalised xz（当时 active map 的坐标，换 map 后不重归一）推进 ring，后面的 labeled bin 可以把它们当成 x_{t−l}。L2 在下一个 trial 开始时清 ring，所以 gap 不会变成下一个 trial 的 lag。两条路的 trunk state 都携带。前向停在最后一个 calibration trial 的末尾；更后的 bin 改变不了 residual 或 commit。

η=0 且 ρ=0 时 Delta 保持 0，commit 就是 prefix P0，contiguous decode 与 P0 相同。

## 数字（latency 0，session mean，95% percentile bootstrap，10,000 次，seed 0，聚类 = target session）

B4–33 先对每个 session 的 B ∈ {4, 8, 16, 33} 取均值。完整格子在 `summary.csv`，逐行在 `per_target.csv`，选参变化在 `pick_changes.csv`。

### Round-2，31–180 d（17 sessions）

| | B=1 | B=4 | B=8 | B=16 | B=33 | B4–33 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.213 | 0.295 | 0.322 | 0.332 | 0.336 | 0.321 |
| S1-L0 | 0.213 | 0.321 | 0.369 | 0.399 | 0.414 | 0.376 |
| S1-L1 | 0.213 | 0.323 | 0.369 | 0.401 | 0.422 | 0.379 |
| S1-L2 | 0.213 | 0.323 | 0.368 | 0.401 | 0.419 | 0.378 |
| L1，用 L0 的 (η, ρ) | 0.213 | 0.323 | 0.370 | 0.403 | 0.415 | 0.377 |
| L0 − P0 | 0 | +0.026 [+0.009, +0.046] 14/17 | +0.048 [+0.028, +0.069] 14/17 | +0.068 [+0.039, +0.096] 15/17 | +0.078 [+0.053, +0.104] 16/17 | **+0.055 [+0.034, +0.076] 16/17** |
| L1 − P0 | 0 | +0.028 [+0.014, +0.047] 16/17 | +0.048 [+0.029, +0.069] 15/17 | +0.070 [+0.041, +0.099] 14/17 | +0.086 [+0.047, +0.124] 14/17 | **+0.058 [+0.035, +0.082] 14/17** |
| L1 − L0 | 0 | +0.002 [−0.003, +0.009] 7/17 | +0.000 [−0.007, +0.007] 7/17 | +0.002 [−0.004, +0.008] 9/17 | +0.008 [−0.008, +0.024] 10/17 | +0.003 [−0.002, +0.009] 11/17 |

B4–33 上 L1 − P0 为负的三个 session：20150616（−0.014）、20150617（−0.007，L0 也是 −0.010）、20140627（约 0，L0 是 +0.013）。Conservative 门在这一层没有把 L1 退回 P0（L1-cons = L1）。

L0 与 L1 的 (η, ρ) 不同：B=33 有 17/17 变了，其中 14 个从 (0.3, 0.0) 变成 (1.0, 0.03)；B=4 是 2/17，B=8 是 2/17，B=16 是 1/17，B=1 都是 η=0。用 L0 的 pick 跑 L1（transfer）相对 L0 仍是 +0.002 [−0.002, +0.006]，增益不依赖这次重选。

Latency 1 的 B4–33：L0 − P0 +0.055 [+0.034, +0.076] 16/17，L1 − P0 +0.058 [+0.035, +0.082] 14/17。

这一层里的 sub-C month target（causal-main 的 sc_0629 / sc_0708，round-2 复用了 round-1 trunk）单独看，B4–33 L1 − P0：sc_0629 六天 **+0.100，6/6**（最小 +0.073）；sc_0708 的 20151103–1112 六天 **+0.059，6/6**（最小 +0.026）。

### Round-2，≤30 d（8 sessions；10 个 pair 先按 session 平均）

| | B4–33 |
|---|---:|
| P0 | 0.637 |
| L0 − P0 | +0.003 [−0.002, +0.009] 5/8 |
| L1 − P0 | **+0.009 [+0.005, +0.014] 7/8** |
| L2 − P0 | +0.008 [+0.004, +0.013] 7/8 |
| L1 − L0 | +0.006 [+0.003, +0.009] 8/8 |

L1 相对 P0 唯一为负的是 20150716（约 −0.002）。B=1 三行都等于 P0。

### Round-2，>180 d（压力测试，7 sessions，其中 6 个是未收敛的 sc_0309）

| | 全部 7 | 去掉 sc_0309（只剩 sm_0512，n=1，无 CI） |
|---|---:|---:|
| P0 | 0.148 | 0.532 |
| L0 − P0 | +0.071 [+0.007, +0.137] 4/7 | −0.036 |
| L1 − P0 | +0.083 [+0.017, +0.150] 4/7 | −0.003 |
| L2 − P0 | +0.074 [+0.003, +0.145] 4/7 | −0.008 |

和 round-2 的披露一致：这一层的正增益主要来自共用 trunk 的 sc_0309。去掉之后 L0 伤害 P0，L1 接近 0。不要把 +0.083 读成 7 个独立 trunk。

### 冻结 trunk（causal-main 路径）

L0 用存档的 AND pick（`s1_causal_and`）。L1/L2 按 protocol (a) 在 live episode 上重选。Sub-C month target 已含在上面的 31–180 d（同一批 round-1 trunk）。Year 标签没有单列：20150511 在 drift 的 month trunk 上。

**Days trunk，20150616 / 20150617（n=2，间隔 1–2 d）。** P0 的 B4–33 是 0.732。L0 − P0 = −0.014（两天都负）。L1 − P0 = −0.003（两天都略负，幅度大约是 L0 的 1/5）。n=2，bootstrap 区间没有意义。这里 live residual 去掉了大部分伤害，但没有变成正增益；P0 已经很高。

**Month trunk，drift 12 天。**

| | B=8 | B4–33 |
|---|---:|---:|
| P0 | 0.209 | 0.211 |
| L0 − P0 | +0.068 [+0.051, +0.085] 12/12 | **+0.091 [+0.066, +0.116] 12/12** |
| L1 − P0 | +0.076 [+0.043, +0.114] 10/12 | **+0.101 [+0.063, +0.138] 10/12** |
| L1 − L0 | +0.008 [−0.018, +0.037] 8/12 | +0.009 [−0.014, +0.031] 8/12 |

B=8 的 L0 与 causal-main 主表一致（P0 0.209，S1 0.276，差 +0.067）。L1 保持这份增益。

## Parity（L0 对存档行）

重算出的 L0 query R² 相对存档：

| 集合 | 行数 | max \|diff\| | median |
|---|---:|---:|---:|
| round-2 31–180 d | 340 | 3.0e-4 | 3e-5 |
| round-2 ≤30 d | 200 | 9.9e-5 | 2e-5 |
| round-2 >180 d | 140 | 3.8e-4 | 4e-5 |
| causal-main days + drift | 280 | 0（逐位相同） | 0 |

Selection 上抽查的 L0 单点（每个 trunk 一个 (η, ρ)，含 B=33）绝对差 8e-5 到 3e-3（最大在 sc_0713 的一个大负分上）。η=0 的 live 分数与同 scorer 的 P0 差都 < 1e-3（没有 WARN）。

## 硬件

L1 不再为校准保留第二份 trunk state。架构文档里的 shadow forward 是校准期 trunk MAC ×2，外加约 50 KiB 状态；这条在 L1 下不需要。Ring 仍是 L0 已经有的那一块（深度 L−1；本实验的 kernel 是 lag 0..L，L 来自各 backbone 的 R2 规则，常见 L=32，即 33 tap）。差别只是 trial 边界不清零。本检查里 ring 存的是 fp32 normalised xz，不是架构文档里的 3-bit spike count；没有做逐次取整。

## 没做

COMBO 与 CCA+S1 的 L1 对照没有跑（要另做带 CCA 的 prefix cache 和 α grid）。2016 tier、final session、20140307 作为 target、`results/official_evalai/` 都没有用。
