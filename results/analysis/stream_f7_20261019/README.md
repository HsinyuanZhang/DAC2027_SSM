# Streaming fixed-feedback refinement（F7-J 求和重排，2026-10-19）

单遍、不重放的 read-in 更新。冻结 trunk，只使用 B 个有标签 calibration trial。Source 上拟合固定反馈核，target 端只有前向、FIR / eligibility trace 累加和 trial 边界的 shrinkage。没有 target backprop（11,040 行 streaming 记录的 `h2_backward_calls` 全是 0）。没有使用 final session（20150623 / 0625 / 0626），20140307 只作为 month 的 source selection session，不是 target。

代码：`ssm_decode/stream_f7.py`，`scripts/run_stream_f7.py`，`scripts/analyze_stream_f7.py`，`tests/test_stream_f7.py`（8 个测试通过）。原始行在 `results/stream_f7_20261019/`（gitignored）。本目录是汇总。

## 1. 重排是恒等，单遍不是 R2

Batch F7-J（`fwd_refine.temporal_feedback` + `fj_gradient`）先把 residual 做反向时间卷积得到 latent error，再和输入做外积。同一条前向、同一个 \(\bar J_\ell\)、同样的 trial 末尾补零，改写成「当前 residual × 过去输入的 ring」：

\[
U=\sum_t\sum_{\ell}(\bar J_\ell^\mathsf T r_t)\,x_{t-\ell}^\mathsf T.
\]

这只是求和顺序。随机数据和线性 trunk 上与 batch 项一致到 \(10^{-9}\)；trace 在核正好是指数和、且 lag 盖住 trial 时与 FIR 一致到 \(10^{-9}\)。Real-data 检查（source session 20140203，P0 表，B=4，L=32，`exactness_real.json`）：

| 检查 | max abs |
|---|---:|
| 同一条轨迹，float64 online vs batch 卷积 | \(1.24\times 10^{-14}\) |
| 同一条 residual，切到真实 window vs 带 end-padding 的 batch | \(6.2\times 10^{-15}\) |
| 去掉 end-padding 的前向 vs padded 前缀 | 0 |
| float32 `fj_gradient` vs float64 重排 | \(2.9\times 10^{-6}\)（float32 舍入） |
| 单 trial 前向 vs R2 的 batched kernel（诊断，不是代数） | 0.008 |
| 由此造成的一步 table 差 | \(1.7\times 10^{-5}\) |

末尾 padding 不改变因果前缀。单 trial 前向和 R2 的 batch 前向不是 bit-exact，这是 kernel 数值，不是边界代数。

因果性（`stream_f7.CAUSALITY`）：

- **R2 replay**：S 次重放已存的 B 个 trial。Latent error 在 trial 内部读未来 residual（trial 已经录完）。不读 trial 以外的 bin。
- **FIR 重排**：每个 bin 的贡献只用 \(r_t\) 和 \(x_{\le t}\)，不读未来 residual。
- **S1-FIR**：每个 calibration trial 从零 state 前向一次（与 headline R2 的 `pad_trials` 边界相同），trial 内表固定，trial 边界做一次朝 P0 的 shrinkage（anytime commit）。后面的 trial 不改前面的 commit。不读 query。表在 trial 之间变了，所以不等于一步 R2，也不等于 S 步重放。
- **Trace**：同样的 commit 和边界。\(s_{m,t}=\rho_m s_{m,t-1}+x_t\) 在 trial 起点清零。它近似 FIR，不读未来 residual。\(\ell\ge L\) 的尾巴是指数外推，不是未来数据。
- **S1-FIR+R1s**：同一次 live 前向里累加 readout Gram（用的是该 trial commit 之前的表）。最后一个 trial 之后解一次 ridge，不在最终表上再跑一遍。

归一化是这 B 个 trial 的 P0 归一化。Query 是整段连续因果解码，只在 query-trial 的 bin 上、以物理单位算 R²。

## 2. 协议

| 层 | target | 规则 | 复用的 P0 / R2 / R3 |
|---|---|---|---|
| subm_month | 20140626 | P3 因果 month（选择集 {20140307}） | drift curve |
| subm_year | 20150511 | P3 year OOS（4 个早于 target 的 session） | `eval_year_oos.jsonl`，R3 取 source 上最好的 kind |
| subm_drift | drift 的 12 个 target（含上面两个日期；20150511 在这里用的是 **month** 规则） | 同上 month | drift curve |
| subc_month | sc_0629、sc_0708 的 month target（各 6 个，session 不重叠） | round-1 各自的 selection session | round-1 行 |

B：sub-M 为 {1, 2, 4, 8, 16, 33}；sub-C 为 round-1 的 {1, 4, 8, 16, 33}（没有 B=2 的 source 规则，没有补）。8 draws，与已有行同一 `draw_trials`；2 个 P0 seed。Draws 先在 session 内平均，再对 session 做 95% t 区间（df = n−1）。只有一个 target 的层改用 draw cluster bootstrap（一个 draw 的两个 seed 绑在一起）。12 个 drift target 共用一个 trunk / 一套规则，这个 CI 不含 trunk 和规则选择的方差。

新超参只在 source 上选，网格与 P3 相同（η ∈ {1, 3, 10, 30, 100, 300, 1000}；readout kind 与 λ 用 P3 的网格）。L 和 ρ 锁在该 B 的 R2 规则上，不在 target 上重选。S1-FIR 的 η 单独选；S1-FIR+R1s 的 (η, kind, λ) 联合选；trace-M4 / M8 各自选 η。并列时取排序后的第一个 key（较小的 η）。

选出的 η 都在 {1, 3}，没有顶到 1000。Month B=8 的 R2 η 是 3，单遍重选是 1（单遍不能用为 S 步调的步长）。Year B=33 的 S1+R1s 把 λ 选到 0.01（网格偏小的一端）。

## 3. Headline（query R²）

完整格子和 CI 在 `cells.csv`、`paired.csv`、`retained.csv`。下表是均值。S1 是 source 重选 η 的单遍 FIR；S1/R2η 是沿用 R2 的 η；R1s 是单遍 FIR 加 live readout ridge。

| 层 | B | P0 | R2 | R3 | S1/R2η | S1 | S1+R1s | M4 | M8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| subm_month（1 个 target） | 1 | 0.466 | 0.488 | 0.492 | 0.481 | 0.494 | 0.495 | 0.495 | 0.496 |
|  | 8 | 0.565 | 0.626 | 0.630 | 0.620 | 0.606 | 0.610 | 0.607 | 0.607 |
|  | 33 | 0.566 | 0.670 | 0.679 | 0.628 | 0.649 | 0.664 | 0.650 | 0.650 |
| subm_year（year 规则） | 1 | 0.198 | 0.219 | 0.219 | 0.209 | 0.195 | 0.195 | 0.195 | 0.195 |
|  | 8 | 0.256 | 0.336 | 0.334 | 0.299 | 0.313 | 0.310 | 0.301 | 0.313 |
|  | 33 | 0.315 | 0.361 | 0.435 | 0.148 | 0.307 | 0.373 | 0.309 | 0.309 |
| subm_drift（12 target，month 规则） | 1 | 0.153 | 0.170 | 0.165 | 0.166 | 0.153 | 0.152 | 0.153 | 0.152 |
|  | 8 | 0.212 | 0.345 | 0.348 | 0.300 | 0.284 | 0.287 | 0.283 | 0.283 |
|  | 33 | 0.211 | 0.406 | 0.412 | 0.348 | 0.354 | 0.363 | 0.353 | 0.353 |
| subc_month（12 session） | 1 | 0.222 | 0.226 | 0.224 | 0.226 | 0.226 | 0.223 | 0.226 | 0.179 |
|  | 8 | 0.301 | 0.362 | 0.361 | 0.330 | 0.349 | 0.346 | 0.343 | 0.345 |
|  | 33 | 0.322 | 0.415 | 0.416 | 0.396 | 0.389 | 0.391 | 0.388 | 0.388 |

Drift B=8 的 P0 均值 0.212，与 drift-curve README 的 12-target 平均一致，说明复用的行接对了。

### 保留的 R2 增益

分数 = (方法 − P0) / (R2 − P0)，先在 session 内配对再平均。|R2 − P0| < 0.01 的格子留空。Session bootstrap 的 95% 区间见 `retained.csv`。

| 层 | B | R2 − P0 | S1/R2η | S1 | S1+R1s | M4 | M8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| subm_drift | 8 | +0.133 [0.101, 0.164]，12/12 | 0.66 | 0.54 | 0.57 | 0.53 | 0.53 |
| subm_drift | 33 | +0.195 [0.152, 0.238]，12/12 | 0.71 | 0.73 | 0.78 | 0.73 | 0.73 |
| subm_month | 8 | +0.061 | 0.90 | 0.67 | 0.74 | 0.68 | 0.68 |
| subm_month | 33 | +0.104 | 0.60 | 0.80 | 0.94 | 0.81 | 0.81 |
| subm_year | 8 | +0.081 | 0.54 | 0.71 | 0.68 | 0.57 | 0.71 |
| subc_month | 8 | +0.061 [0.041, 0.081]，12/12 | 0.49 | 0.79 [0.60, 1.09] | 0.75 | 0.69 | 0.73 |
| subc_month | 33 | +0.093 [0.070, 0.116]，12/12 | 0.79 | 0.71 [0.49, 1.03] | 0.74 | 0.71 | 0.71 |

Drift 上 S1 − R2 的 session CI 不含 0，而且 12/12 target 上 S1 低于 R2（B=8 差 −0.061 [−0.077, −0.045]；B=33 差 −0.052 [−0.064, −0.040]）。单遍留下大约一半到四分之三的 R2 增益，不是无损替换。重选 η 在 B=8 的 drift 上不如直接用 R2 的 η（0.54 vs 0.66）：source 上选出的更小步长偏保守。B=33 则是重选更好。

S1+R1s 在大多数格子上只比 S1 高约 0.00–0.01。例外是 year B=33：S1 是 0.307（低于 P0 0.315），加上 live ridge 到 0.373，接近 R2 的 0.361，仍低于 R3 的 0.435。同一格子上，沿用 R2 的 η=10 单遍掉到 0.148。这是 P3 里 year B=33 长 trial 不稳定的同一类现象；R2 − P0 的 draw CI 含 0，所以这一格的「保留比例」没有解释价值（表里没有列）。

Trace 与 FIR 在 B≥8 上几乎重合（drift B=33：M4/M8 都是 0.353，S1 是 0.354）。核拟合的相对 Frobenius 误差（source \(\bar J\)，极点限制在 [−0.92, 0.92] 且两两相距 ≥ 0.07，避免抵消型大系数）：

| 核 | L=32, M=4 | L=32, M=8 | L=8, M=4 | L=8, M=8 |
|---|---:|---:|---:|---:|
| sub-M（month = year trunk） | 0.128 | 0.037 | 0.024 | 0.009 |
| sc_0629 | 0.110 | 0.036 | 0.031 | 0.008 |
| sc_0708 | 0.082 | 0.024 | 0.026 | 0.006 |

M=8 把 L=32 的核拟合到约 3–4%，数据项方向与 FIR 的余弦在随机探针上约 0.99，所以 R² 贴着 S1 是预期结果，不是独立的增益。没有约束的极点会挤在一起，系数到 \(10^3\)，float32 trace 不稳定；那一版没有采用。

负面结果：sub-C B=1 的 trace-M8 平均 0.179，低于 P0 0.222（差 −0.043 [−0.079, −0.007]，12 个 session 里 3 个为正）。拆开看，sc_0629 的 B=1 均值从 P0 的 0.136 掉到 0.050，sc_0708 的 0.308 与 P0 持平。短核（L=8）上 8 个极点过拟合 source 核，B=1 没有第二步可以收回。

## 4. 代价（解析，规则 8）

没有定点模拟。MAC 是乘加各一次。Trunk 从 K=10、d_model=128、2 层、d_state=32 的模块读出：fixed-weight 350,720 + scan（activation×activation）32,768 = 383,488 / bin，再加 read-in 的 K·N。下表的 L = 33 taps（规则里的 L=32 含 lag 0）。R2 用 S=10、B=8、T=200 的点估计只为了 replay buffer；每 bin 的 FIR / trace MAC 不依赖 T。数字在 `cost.json`。

每 bin，校准自己的累加（不含 trunk）：

| | N=26 weight×act | N=26 act×act | N=256 weight×act | N=256 act×act |
|---|---:|---:|---:|---:|
| FIR（33 taps） | 660（\(\bar J\) ROM × residual） | 8,580（lag 外积） | 660 | 84,480 |
| trace M=4 | 184（\(C^\mathsf T e\) 与 \(\rho s\)） | 1,040 | 1,104 | 10,240 |
| trace M=8 | 368 | 2,080 | 2,208 | 20,480 |

N=256、M=4 的 trace 合计 11,344 MAC/bin，与设计备忘录的例子一致。FIR 的大头是 activation×activation 的外积，不是 ROM。N=26 时 FIR 额外 MAC 约是一次 trunk bin 的 2%；N=256 时约 22%，因为外积按 N 线性涨而 trunk 不涨。

SRAM（fp32，含当前 trial 的 staging，不含与推理共享的 trunk state）：

| | N=26 | N=256 |
|---|---:|---:|
| S1-FIR（ring L·N + U 的 K×N + 表 + T0 + 一个 trial staging） | 9.2 KB | 83.6 KB |
| trace M=4（M·N 代替 ring） | 6.3 KB | 54.6 KB |
| S1+R1s（再加 full Gram 与 V） | 44.0 KB | 118 KB |
| R2（S=10，含全部 B trial 的 replay buffer） | 57 KB，其中 replay 22 KB | 221 KB，其中 replay 156 KB |

R2 在每个 calibration bin 上把 trunk 再跑 S 次（这里 S=10）。B=8、T=200、N=26 时 R2 总计约 \(6.2\times 10^9\) MAC，几乎全是这 S 次 trunk。S1 的 trunk 前向就是这次 calibration trial 的推理前向，不再重放。

## 5. 没有做

- sub-C 的 B=2。Round-1 没有这个预算的 R2 规则，也没有对应的 draw 行。
- 定点累加。与 P3 一样，本轮是 fp32 评测加解析代价。
- Calibration trial 之间传递 trunk state（P3 的 contig 变体）。这里对齐的是 headline 的零 state `pad_trials`，以便和已存的 R2 比。
- Days 层。
- 在最终表上再跑一遍 readout（那是第二次前向）。R1s 用的是 live 统计。
- 没有改 trunk，没有改已有模块的语义，没有 push，没有动 Overleaf。
