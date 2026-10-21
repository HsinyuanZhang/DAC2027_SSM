# H4：S1 live residual（2026-10-21）

## Review corrections（2026-10-21）

审查指出两处和设备不一致：原来的 L1 在任何 latency 下都按 latency 0 取残差（trial k 用 commit k）；trial 0 用自己的 normalisation（INITIAL），设备上不能实时做出来，需要把这一 trial 重放一遍。修正后的主行是 **L1-dev**：残差来自 `commit_segments` 在该 latency 上已经装上的那张表，并且 **不用 trial 0 的残差做 S1 step**（P0 的 staging 和 query 计分不变）。原来的 L1 改称 **L1-staged**（需要 one-trial replay）。(η, ρ) 在 L1-dev 的 source episode 上按 protocol (a) 重选，latency 0 和 1 分开选。下面原有的「L1」数字都是 L1-staged、latency 0 的安装。

和 `/tmp/h4_review` 对过：`lat1_mid.jsonl` 里用同一组 L1 pick 重放，`l1_lat0` 相对本仓库存档的 `s1_l1` **max |diff| = 0**（128 行）。本实现在 `sc_0629 / 20150629 / B=4 / seed 0` 上和那份 scratch 逐位相同：`l1_lat0` 0.12704565，`l1live_lat1` 0.11654516，`l1_skip0_lat0` 0.11133229。`lat1_short.jsonl` 的 `l1_lat0` 相对存档最大差 1.9e-4。同一组 pick 下，mid 的 B4–33：真 latency 1（`l1live_lat1` − P0）是 +0.052 [+0.030, +0.077]（14/17），比「latency 0 的表只晚装一 trial」低 0.006；丢掉 trial 0 残差（`l1_skip0_lat0` − P0）是 +0.052 [+0.032, +0.073]（14/17）。重选之后的 L1-dev 与这两个同 pick 数字同向，见下表。

Ring 深度：`jbar_for(L)` / `average_jbar` 返回的是 **L+1 个 tap**（lag 0..L）。R2 规则里常见 L=32，所以是 33 tap，ring 存 **32 行**（`n_lags - 1`），不是 31 行。Cold bin 写进 ring 的 xz 是 mean 0、std 1，也就是 raw count；trial 内是当时那张表的 normalised xz，换表后不重归一。L1 与 L2 的差就是这个坐标混合的上界：31–180 d B4–33，L1 − L2 约 +0.001（≤30 d +0.001，drift +0.005，>180 d +0.009）。

Trial 0 的 staging：L1-dev 在校准前向里仍用 INITIAL map 解码 trial 0（为了 state 与 ring 的连续性），实时设备做不到这一点。审查者把 trial 0 改成 cold map 解码（31–180 d，L1-dev latency 0 的 pick），B4–33 为 +0.0515 [+0.032, +0.073]（14/17），与 L1-dev 差 −0.0008 [−0.002, +0.000]，可忽略但不是严格为 0。另外，各方法在 latency 1 下都在 trial 1 安装 INITIAL，这是协议层面的共同假设。

B=33 的 L1-staged pick 相对 L0 是 **17/17** 都变了：14 个 (0.3, 0)→(1.0, 0.03)，2 个 (1.0, 0.1)→(1.0, 0.03)，1 个退回 η=0。

硬件：L1-dev 在任何 latency 都不需要 shadow forward，也不需要第二份 trunk state。L1-staged 需要把 trial 0 重放一次（约 50 KiB 的 state 快照，加上这一个 trial 的缓冲）才能得到 INITIAL 上的 ŷ。仍是 fp32 算法检查，不是 rule 8 的逐次取整。

数字是 session 均值，再做 95% percentile bootstrap（10,000，seed 0）。(i) 聚类 = target session；(ii) 聚类 = backbone（一个 backbone 先把它的 target 取均值）。B4–33 先对 B ∈ {4, 8, 16, 33} 取均值。完整格子在 `summary_clusters.csv`。

### 31–180 d（17 sessions，4 个 backbone）

| | latency 0，session | latency 0，backbone | latency 1，session | latency 1，backbone |
|---|---:|---:|---:|---:|
| P0 | 0.321 | 0.314 | 0.321 | 0.314 |
| L0 − P0 | +0.055 [+0.034, +0.076] 16/17 | +0.042 [+0.012, +0.073] 4/4 | +0.055 [+0.034, +0.076] 16/17 | +0.042 [+0.012, +0.073] 4/4 |
| L1-staged − P0 | +0.058 [+0.035, +0.082] 14/17 | +0.042 [+0.004, +0.080] 3/4 | +0.058 [+0.034, +0.082] 14/17 | +0.042 [+0.004, +0.080] 3/4 |
| **L1-dev − P0** | **+0.052 [+0.032, +0.073] 14/17** | **+0.038 [+0.004, +0.071] 3/4** | **+0.047 [+0.028, +0.068] 14/17** | **+0.034 [+0.003, +0.066] 3/4** |
| L1-dev − L0 | −0.002 [−0.008, +0.003] 8/17 | −0.005 [−0.010, −0.000] 1/4 | −0.007 [−0.014, +0.000] 5/17 | −0.008 [−0.011, −0.005] 0/4 |
| L1-dev，用 L0 的 (η, ρ) − P0 | +0.051 [+0.033, +0.069] 17/17 | +0.040 [+0.013, +0.066] 4/4 | +0.045 [+0.028, +0.065] 17/17 | +0.036 [+0.013, +0.062] 4/4 |

Latency 0 的逐 B（session 聚类），L1-dev − P0：B=1 为 0（η=0）；B=4 +0.016 [+0.005, +0.028] 12/17；B=8 +0.041 [+0.025, +0.059] 15/17；B=16 +0.067 [+0.040, +0.095] 14/17；B=33 +0.085 [+0.049, +0.123] 14/17。两种聚类下 B4–33 的区间都离开 0。丢掉 trial 0 的 step 大约吃掉 0.006（L1-dev − L1-staged 的 session 区间在 0 以下），但相对 P0 的增益还在。

Subject 拆开（session 聚类，B4–33，latency 0）：

| | n | P0 | L0 − P0 | L1-staged − P0 | L1-dev − P0 |
|---|---:|---:|---:|---:|---:|
| sub-M | 5 | 0.360 | +0.011 [−0.001, +0.025] 4/5 | +0.006 [−0.009, +0.023] 2/5 | +0.007 [−0.007, +0.022] 2/5 |
| sub-C | 12 | 0.305 | +0.073 [+0.051, +0.095] 12/12 | +0.080 [+0.057, +0.103] 12/12 | +0.071 [+0.051, +0.092] 12/12 |

这一层的增益在 sub-C。sub-M 上 L0 本身区间就含 0，L1-dev 没有把它变成伤害（点估计仍是小的正数）。Latency 1 的 sub-C L1-dev − P0 是 +0.066 [+0.046, +0.086] 12/12；sub-M 是 +0.004 [−0.008, +0.019] 2/5。

Per-backbone，B4–33，latency 0，L1-dev − P0：sc_0629 **+0.091（6/6）**，sc_0708 **+0.052（6/6）**，sm_0611 +0.008（2/4），sm_0627 0.000（1 个 session，门退回 P0）。

### ≤30 d（8 sessions，3 个 backbone）

| | latency 0，session | latency 1，session |
|---|---:|---:|
| P0 | 0.637 | 0.637 |
| L0 − P0 | +0.003 [−0.002, +0.009] 5/8 | +0.003 [−0.002, +0.009] 5/8 |
| L1-staged − P0 | +0.009 [+0.005, +0.014] 7/8 | +0.009 [+0.005, +0.014] 7/8 |
| **L1-dev − P0** | **+0.009 [+0.005, +0.014] 7/8** | **+0.008 [+0.004, +0.012] 7/8** |

Backbone 聚类（n=3）latency 0：L1-dev − P0 +0.011 [+0.007, +0.014] 3/3。不伤害 P0。L1-dev 与 L1-staged 在 latency 0 几乎相同（差 +0.000，区间含 0）。

### >180 d（7 sessions；6 个是 sc_0309，2 个 backbone）

| | 7 sessions，latency 0 | 7 sessions，latency 1 | 去掉 sc_0309（sm_0512，n=1，latency 0） |
|---|---:|---:|---:|
| P0 | 0.148 | 0.147 | 0.532 |
| L0 − P0 | +0.071 [+0.008, +0.136] 4/7 | +0.071 [+0.007, +0.136] 4/7 | −0.036 |
| L1-dev − P0 | +0.077 [+0.010, +0.146] 5/7 | +0.079 [+0.020, +0.141] 4/7 | **+0.002** |
| L1-dev，L0 的 pick − P0 | +0.073 [+0.012, +0.139] 4/7 | +0.073 [+0.015, +0.135] 4/7 | −0.005 |

Backbone 聚类只有 2 个点（sc_0309 与 sm_0512），latency 0 的 L1-dev − P0 是 +0.046，两个 backbone 都为正；不要把 session 区间读成 7 个独立 trunk。

### Days / drift（冻结 trunk）

L0 用的是 causal-main 的 **per-trunk** AND 规则（`s1_causal_and`，同一 stratum 的 target 共用一个 (η, ρ)）。L1-dev 用的是 protocol (a) 的 pool，按 target 重选。并排的 like-for-like 行是 **transfer**：L1-dev 的前向，但 (η, ρ) 用 L0 的 pick。

Days trunk，20150616 / 20150617（n=2，间隔 1–2 d；backbone 只有 1 个，没有 backbone CI）：

| | latency 0 | latency 1 |
|---|---:|---:|
| P0 | 0.732 | 0.732 |
| L0 − P0 | −0.014（0/2） | −0.014（0/2） |
| L1-staged − P0 | −0.003（0/2） | −0.003（0/2） |
| L1-dev − P0 | −0.002（1/2） | −0.003（0/2） |
| transfer − P0 | −0.007（0/2） | −0.014（0/2） |

P0 已经是 0.732。L0 的 per-trunk 规则在这里是负的。换成 L1-dev 之后几乎贴着 P0；用 L0 的 pick 做 L1-dev（transfer）仍比 L0 伤害小（latency 0：−0.007 对 −0.014），但没有变成正增益。

Month trunk，drift 12 天（1 个 backbone）：

| | latency 0 | latency 1 |
|---|---:|---:|
| P0 | 0.211 | 0.211 |
| L0 − P0 | +0.091 [+0.066, +0.117] 12/12 | +0.091 [+0.066, +0.116] 12/12 |
| L1-staged − P0（protocol (a)） | +0.101 [+0.062, +0.137] 10/12 | +0.101 [+0.064, +0.139] 10/12 |
| **L1-dev − P0** | **+0.093 [+0.056, +0.130] 10/12** | **+0.084 [+0.051, +0.119] 10/12** |
| transfer − P0 | +0.087 [+0.065, +0.108] 12/12 | +0.081 [+0.060, +0.102] 12/12 |
| transfer − L0 | −0.005 [−0.012, +0.002] 3/12 | −0.011 [−0.017, −0.004] 1/12 |

B=8 的 L0 − P0 仍是 +0.068，与 causal-main 主表（P0 0.209，S1 0.276）一致。Like-for-like 的 transfer 在 latency 0 保持 L0 的增益（12/12，区间离开 0）。Latency 1 上 transfer 比 L0 低 0.011，区间不含 0，但相对 P0 仍是 +0.081 [+0.060, +0.102]。

问题：S1 的残差现在来自每个 calibration trial 的零状态前向（L0，`rebased_feedback_loop`）。部署推理是整段连续、trunk state 跨 trial 携带。若 live residual 下增益还在，校准不必再跑一次 shadow forward。

这是 fp32 算法检查，不是 rule 8 的逐次取整定点仿真。Trunk 冻结，target 不反传。Selection 只在 source 上，protocol (a) expanding window，主门是 AND；conservative 门（被选中的 (η, ρ) 至少在 2 个 selection session 上增益为正，否则 η=0）一并报告。Latency 0 是主行。

代码：`ssm_decode/live_residual.py`（L0 语义未改）。L1 = state 与 FIR ring 都跨 trial 携带。L2 = state 携带，但每个 calibration trial 开始时 ring 清零。评分仍是 causal-first-B 的整段 contiguous decode（表在 commit 边界切换）。

## 结论

**（本节是 L1-staged 的原始结论；它需要 trial 0 的一次重放。设备一致的结论见顶部 Review corrections 的 L1-dev：31–180 d latency 0 +0.052 [+0.032, +0.073]，latency 1 +0.047，无 shadow forward、无第二份 state。）** S1-L1 保持 31–180 d 的增益。 Round-2 这一层 B4–33，L0 − P0 是 **+0.055 [+0.034, +0.076]（16/17）**，与已发表的 +0.055 [+0.034, +0.076] 一致。L1 − P0 是 **+0.058 [+0.035, +0.082]（14/17）**。L1 − L0 是 +0.003 [−0.002, +0.009]（11/17），区间含 0。L2 − P0 是 +0.057 [+0.033, +0.080]（14/17）：ring 跨 trial 携带不是增益的来源，state 携带也没有把它吃掉。

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

L0 与 L1-staged 的 (η, ρ) 不同：B=33 是 17/17，其中 14 个 (0.3, 0)→(1.0, 0.03)，2 个 (1.0, 0.1)→(1.0, 0.03)，1 个退回 η=0；B=4 是 2/17，B=8 是 2/17，B=16 是 1/17，B=1 都是 η=0。用 L0 的 pick 跑 L1（transfer）相对 L0 仍是 +0.002 [−0.002, +0.006]，增益不依赖这次重选。

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

L1-dev 不再为校准保留第二份 trunk state，任何 latency 都不需要 shadow forward。L1-staged 需要把 trial 0 重放一次（约 50 KiB state 快照加这一 trial 的缓冲）。Ring 仍是 L0 已经有的那一块：R2 的 L=32 时 `jbar` 有 33 个 tap（lag 0..32），ring 深度是 32 行。差别只是 trial 边界不清零。写进 ring 的是当时 active map 下的 fp32 xz（cold 段 mean 0、std 1，即 raw count）；不是架构文档里的 3-bit spike count。没有做逐次取整。

## 没做

COMBO 与 CCA+S1 的 L1-dev 对照没有跑（要另做带 CCA 的 prefix cache 和 α grid）。2016 tier、final session、20140307 作为 target、`results/official_evalai/` 都没有用。
