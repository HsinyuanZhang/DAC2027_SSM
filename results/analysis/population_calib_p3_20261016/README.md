# P3：可部署的 forward-only 细化能恢复多少 P0+ui_reg 的增益（readout 闭式岭回归 R1、read-in 局部规则 F7-J R2、二者组合 R3、identity-free + label-free 对照 R4）

日期：2026-10-16。执行者：P3 工作线（GPU0，≤3 个并行 job）。规格见任务说明；背景见 `results/analysis/population_calib_20261015/README.md`（P1，含 §0.1 审查）、`results/analysis/population_calib_p2_20261015/README.md`（P2）、`docs/next_experiments_round2_20261015.md` §2–3（F7-J 设计）。

代码（全部为新增文件，没有改动 P1/P2 的代码语义，也没有改动 `calib_quant.py / calib_cost.py / run_pc2_quant.py / pc2_cost_report.py`）：`ssm_decode/fwd_refine.py`（F7-J 反馈卷积与更新、平均 Jacobian、梯度对齐诊断、readout 统计与岭回归、trunk MAC 计数）、`ssm_decode/refine_cost.py`（P3 解析代价）、`scripts/run_pc3_refine.py`（jacobian / basis / select / select-r3 / eval / ablate）、`scripts/run_pc3_{r4.sh,jac_all.sh,queue.sh,phase2.py,ablate.sh}`（队列）、`scripts/analyze_pc3.py`、`scripts/pc3_cost_report.py`、`tests/test_{fwd_refine,refine_cost,pc3_driver}.py`（20 个新测试，全部通过）。
原始 rows：`results/population_calib_p3_20261016/`（gitignored）；本目录是汇总 csv/json：`tables_p3.md`（全部 markdown 表）、`cells_r2.csv`、`gap_recovery*.csv`、`selection.csv`、`source_sim/`（selection json 与 source simulation 网格）、`diag_gradient_alignment.csv`、`jacobian_representativeness.csv`、`ablation_feedback_kernel.csv`、`cost_p3*.csv/json`、`flags.json`、`all_rows_seedavg.csv.gz`。

## 0. 结论摘要

设置：pool O、aligned prior、P1 的 P0 checkpoint（2 seed）、3 个 stratum（days：0616/0617；month：20140626；year：20150511）、B∈{1,2,4,8,16,33}、每格 8 个 draw、连续因果解码、不在 trial 边界重置状态。参照行（P-prior、P0、ui_reg、P0+ui_reg）直接取 P1/P2 的 rows（draw、seed、target 完全一致，已核对：本轮重算的 P0 与 P1 的 rows 最大差 6e-5，P-prior 最大差 1.5e-5）。gap = P0+ui_reg − P0（配对）。

1. **F7-J（R2）是本轮最有效的可部署细化：在 month/year 上恢复了 P0+ui_reg−P0 增益的大约 2/3，且不反传 trunk。** B≥4 汇总（每个 B 的 delta 与 gap 的均值之比；表中每个 B 的数字见 §2）：month 67%（source-OOS 选择）/ 47%（source-IS 选择），year 72% / 66%。绝对值（month，B=8）：P0 0.565 → R2 0.625，P0+ui_reg 0.653；（month，B=33）0.566 → 0.669，P0+ui_reg 0.731；（year，B=8）0.256 → 0.336，P0+ui_reg 0.309；（year，B=33）0.315 → 0.361 (R2) / 0.435 (R3)，P0+ui_reg 0.488。B=1–2 的 gap 本身只有 +0.02–0.03，R2 恢复 +0.004 到 +0.021。
2. **readout 闭式岭回归（R1）单独只恢复一小部分**：B≥4 汇总 month 30%（OOS）/ 13%（IS），year 27% / 9%；只有 B=33 有明显数字（month +0.072，year +0.073，OOS 选择）。**R3（R2 之后再做 readout 岭回归）略好于 R2**：B≥4 汇总 month 75% / 55%（OOS / IS），year 92% / 72%；R3 比 R2 在 B=33 补回了 R2 的一个不稳定 draw（见 §2.3）。full、diag、低秩（rank 8/16/32）三种 readout 变体中 full 最好；lr16 在 B≤16 取得 full 增益的 65–98%（month OOS 92–98%，year OOS 65–95%），B=33 只有 34–69%（lr32 更接近）；diag 在 B=33 只有 full 的 1/3 到 1/2。另外，**只做 readout 端的 P-prior 对照**（不做任何 read-in 校准）相对 P-prior 有明显提升但仍远低于 P0：B=8/33，month +0.074/+0.159（OOS），year +0.008/+0.088，days +0.017/+0.052（IS）。
3. **days 层没有可恢复的 gap（P0+ui_reg−P0 在 −0.013 到 +0.008，在 CI 内），R1/R2/R3 也不改变 P0**：用 source-IS 选择（headline）R1 +0.001 到 +0.003，R2 +0.001 到 +0.008，R3 +0.001 到 +0.010。用 source-OOS 选择则 R1/R3 在 B≥8 掉 0.03–0.06（规则是在漂移很大的 2014 年 session 上选的，对 1–2 天间隔的 target 过激）：**选择用的 source 模拟的漂移量必须与 target 的间隔相当**，这是本轮最重要的方法学教训（§1.3）。
4. **平均 Jacobian 反馈（J̄）只是一个弱的梯度近似，但它携带了真实的方向信息**：第一步 F7-J 更新与真实梯度的余弦（B 个标注 trial 上的损失，中位数，L=32）为 days 0.30 / month 0.41 / year 0.45，未达到 G3′ 的 0.5 门槛；L 越长越好（L=0：0.14/0.16/0.22）。后续步的余弦迅速下降（第 3 步 0.17–0.20，第 10 步 0.05–0.14）。单个窗口的 Jacobian 相对均值的 RMS 偏差为均值范数的 4–20 倍（§2.2），即平均 Jacobian 并不“代表”各时刻的 Jacobian。控制实验：把 J̄ 换成 −J̄ 灾难性变差（month B=8 −1.25），换成同范数的随机核也变差（−0.05 到 −0.12），e=0 时是严格的 no-op；因此 R2 的增益来自 J̄ 的方向，不是来自“随机扰动 + 收缩”（§2.4）。
5. **R4（identity-free + label-free P0，globalmean prior，补上 P2 缺少的对照）**：days 在 B≥8 只有 0.218–0.226（month 0.054–0.063，year 0.04–0.09），而 identity-free **带标签**的 P0 是 0.687–0.696（month 0.43–0.47，year 0.12–0.25）。也就是说 identity-free 的 P0 在 B≥8 的 days 层拿到的 0.69，**约 2/3 来自标签衍生的 per-unit 描述量（行为互协方差、方向调谐）即“按标签调谐”，只有约 0.23 来自无标签的发放统计**；而在 aligned（有 electrode 身份）条件下去掉标签几乎没有损失（days 0.734 vs 0.723，P1 的 `_nolab`）。因此身份元数据与标签调谐是可互相替代的两条信息来源（§4）。
6. **代价（解析，000688：N=26、K=10、D=2、H=128、平均 252 bin/trial）**：一次 B=8 的 P0 校准约 2.4 MMAC；R1 需要把 B 个 trial 通过冻结 trunk 重放一遍，约 0.78 GMAC（P0 的约 330 倍，其中 trunk 前向占 99.4%，全部是 RRAM 可映射的固定权重 MAC）；R2（S=5，L=16）约 3.9 GMAC，S=10、L=32 约 7.8 GMAC；R3 4.7–8.5 GMAC。B=8 的实时当量（按 20 ms bin 的 trunk 推理）：R1 40 s，R2 S=10 为 403 s。**原始 replay buffer：000688 在 3-bit 计数（所检查的 session 里最大计数 7）+16-bit 标签下，B=8/16/33 为 27.3/54.6/112.7 KB，buffer 本身能放进 128 KB（B=33 刚好）；但总工作集（含 32 KB trunk 状态、readout Gram 等）在 B=16 为 90 KB，B=33 为 148 KB，B=33 放不下。HD-sEMG 规模（N=256、K=16、D=5、1250 bin/trial、8-bit 样本）单 trial 就是 325 KB，B=8 为 2.6 MB，replay 完全不可行。** 便宜的替代方案“在 anytime 校准块里随 trial 到达流式累加 readout 统计，不保存 replay buffer”（R1-stream）已评测：精度与 replay 版相比 month 在 B≤16 差 ≤0.003、B=33 低 0.005；year 在 B≤16 低 0.001–0.008、B=33 低 0.013（OOS，full）；SRAM 与 B 无关，为 38.4 KB（含 32 KB trunk 状态，即额外约 6.4 KB），HD-sEMG 在不做首个 trial 缓冲（running normalisation，未评测）时 52.6 KB。该方案只适用于 R1（readout 端），F7-J 需要多轮 replay，没有无 replay 版本（§5）。

## 1. 设置与协议

### 1.1 target 端的候选路径与约束检查

- **P0 起点**：对 B 个 trial 用 P1 的 P0 calibrator（`calibrate` 不变）得到表 T0、以及 B-trial 的 mean/std。之后所有细化都以 T0 为起点和收缩中心。
- **replay**：B 个标注 trial 各自作为一个窗口，从零状态通过冻结 trunk 前向（与 `ui_reg` 的窗口一致；评测解码仍是整段 session 的连续因果解码，不在 trial 边界重置）。窗口里 `eval_mask` 无效或标签非有限的 bin 只在 loss/残差中被屏蔽，仍然通过 trunk。
- **F7-J 一步**（`fwd_refine.fj_gradient / fj_update`）：残差 r_t = y_t − ŷ_t（source label 归一化）→ e_t = Σ_{ℓ=0..L} J̄_ℓᵀ r_{t+ℓ}（固定时间卷积，窗口末尾之后的残差记为 0；使用 trial 内未来残差，需要 L-bin 延迟线）→ G = (1/n)Σ_t e_t x_tᵀ（对所有输入 bin 累加，包括被屏蔽的 bin，因为它们仍影响 trunk）→ q_i = mean x_i²（有效 bin）→ T ← T + η G diag(1/(q+ε)) − ηλ(T−T0)，ε=1e-3。把收缩写成 ρ=ηλ（每步收缩系数），网格 ρ∈{0, 0.03, 0.1, 0.3}。
- **必做测试**（`tests/test_fwd_refine.py`）：线性 trunk 上 F7-J 的 G 与 −autograd 梯度逐元素相等（1e-9），一步 F7-J 更新逐元素等于一步（带预条件与收缩的）梯度步；L 截断、padding、不反传（对 `autograd.grad/backward` 计数为 0）、readout 统计可加性、岭回归极限（λ→0 为最小二乘，λ→∞ 无变化，满秩正交基与 full 一致）。写测试时发现并修正了一个真实错误：最初把被屏蔽 bin 的 x 也屏蔽了，测试立即报不一致；正确做法是只屏蔽残差。
- **readout 岭回归**（R1）：特征 h 是 trunk 末端 LayerNorm 之后、线性 readout 之前的 128 维向量（加 bias，共 129 维）；V = V0 + R(G+λI)⁻¹，G、R 为 B 个 trial 重放得到的、按有效 bin 数归一化的统计（bias 与权重同样做先验中心化）。变体：`full`、`diag`（对角 G）、`lr8/lr16/lr32`（固定 source PCA 基 U 上的 r×r 求解，基来自 6 个 source session 在 P0 表下的特征二阶矩，`basis_*.pt`；能量占比 r=8/16/32 为 0.85/0.94/0.97 (days)、0.93/0.97/0.99 (month/year)）。
- **R1 on P-prior**：起点换成 P-prior 表（B-trial 归一化 + electrode-aligned prior），不做任何 read-in 校准，只做 readout 岭回归（λ 在 source 上按同样的模拟选择）。
- **R1-stream**（便宜的替代方案）：按时间顺序到达的第 j 个 trial 只前向一次，用“由第 1..j−1 个 trial 得到的 P0 表与归一化”（j=1 用 prior 表与该 trial 自身的 mean/std，即需要一个 trial 大小的暂存区）；readout 统计在线累加，最后用最终（B 个 trial 的）P0 表解码。没有 replay buffer；与 replay 版的差别是特征来自更早的（更差的）表，且归一化是前缀的（`tests/test_pc3_driver.py` 逐项验证了每个 trial 只前向一次、表只依赖前缀）。
- **R3**：先做 R2 的 S 步，再在细化后的表上重放一遍做 readout 岭回归（λ 由 source 模拟另外选择，见下）。
- **约束检查**（`flags.json`，15360 条 eval row）：全部候选 row 的反传调用数为 0（对 `torch.autograd.grad/backward` 打补丁计数），query 与校准 trial 不相交，trunk 哈希不变。梯度对齐诊断（`diag_*.jsonl`）使用 `autograd.grad`，单独记录，不进入任何校准表。未使用 0623/0625/0626（2015 final）；未读 `results/official_evalai/`；只用 B 个标注 trial（H4）；所有选择只在 source 上。与审查意见 (4) 对应：`load_p0` 现在断言 `calibrator.prior_mode == world.prior_mode`；R4 的训练与评测都用 `--prior-mode globalmean`。

### 1.2 平均 Jacobian（离线，仅 source，允许 autograd）

对每个 stratum 的 6 个 source session，用该 session 自己的表和整段 session 的 z-score 输入，随机取 600 个以有效 trial bin 为终点的 320-bin 窗口（零状态启动），对最后一个输出 bin 做 D=2 次反传，得到 J_{ℓ}（ℓ=0..32，D×K）；对窗口与 session 平均得到 J̄_ℓ（`jacobian_*.pt`）。month 与 year 的 trunk 和 source 完全相同（哈希一致），所以两者的 J̄ 相同。`jacobian_representativeness.csv` 记录每个 lag 的“代表性”：

| stratum | ‖J̄_ℓ‖_F ℓ=0 / 8 / 32 | 单窗口 RMS ‖J_ℓ‖_F ℓ=0 / 8 / 32 | rel. dev ℓ=0 / 8 / 32 |
|---|---|---|---|
| days | 0.011 / 0.018 / 0.004 | 0.211 / 0.105 / 0.045 | 20.0 / 5.8 / 11.9 |
| month/year | 0.022 / 0.023 / 0.006 | 0.180 / 0.088 / 0.040 | 8.2 / 3.7 / 6.9 |

rel. dev = RMS‖J−J̄‖_F / ‖J̄‖_F：单个窗口的 Jacobian 远大于均值，意味着 trunk 的局部 Jacobian 随状态强烈变化（正负抵消，均值很小），平均 Jacobian 是一个很弱的“代表”，G3′ 的前提并不成立（见 §2.2 的余弦）。

### 1.3 超参数选择（只用 source）：两种模拟集合

“用 source session 做模拟校准”有一个问题：6 个原始 source session 对 trunk 是 in-sample（trunk 的表与 readout 就是在这些 session 上训练的），P0 calibrator 也是在它们上面训练的。在这些 session 上，readout 已经最优、P0 起点已经很好，任何细化都没有收益，选择会退化为“λ 取最大 / η 取最小”（几乎不改变 P0）。因此做了两套模拟，**两套都完整跑了 R1/R2/R3 的选择、评测与报告**：

- **IS**（in-sample）：该 stratum 的 6 个原始 source session，J̄ 留一（排除被模拟的 session），起点为 P0 在 LOO prior 下的表。字面执行任务说明。
- **OOS**（out-of-sample）：trunk 和 P0 calibrator 都没有见过的 source session（pool-E 额外 session，不含任何 D1 target / dev / final）。days：10 个 2014 年的 session（0203、0217、0218、0303、0304、0306、0307、0627、0929、1203）；month 与 year：0307、0627、0929、1203。**对 month 这四个里有三个晚于 month target（20140626）**，因为 month 层在 target 之前只有 0307 一个 trunk 未见过的 session，不足以选择；这是对“因果”的放宽（它们不是 target / dev / final，也没有用到 target 数据），报告时必须写明。
- **事先固定的 headline**（在看到任何 target 结果之前，依据 source 侧的证据——IS 选择退化为 no-op、OOS 选择有可测收益——以及间隔元数据）：**days 用 IS，month/year 用 OOS**。依据是 days 的间隔只有 1–2 天，与 IS 的“零漂移”设定相当，而 OOS 的 2014 年 session 距离 trunk 一年左右，漂移远大；month/year 的 target 距 source 3.7/15 个月，与 OOS 的漂移相当。另一套总是同时给出（§2 的每张表分别列出 IS 与 OOS），days-OOS 是一个“选择与 target 失配”的负面示例。
- 模拟的具体做法：每个 session × seed(2) × B(6) × 2 个 draw 为一个 episode（IS：6 session×24=144 个 episode；OOS：days 10×24=240，month/year 4×24=96），在该 session 前 50000 bin 内抽 trial 并在同一范围的其余 trial 上算连续解码 R²（物理单位，同 P 行）；P0 起点用该 seed 的 P0 calibrator。按 B 分别选择（与 P2 的 ui_reg 一致）：R2 的 (L∈{8,16,32}, η∈{1,3,10,30,100,300,1000}, ρ∈{0,0.03,0.1,0.3}, S∈{1,3,5,10})；R1 的 λ∈{1e-3,…,100}（9 档，三种 readout 来源 × 五种 kind）；R3 在已选的 R2 规则上选 readout 的 λ。选出的 η 常落在网格下边缘（η=1）、L 常落在上边缘（L=32），没有扩展网格。
- month 与 year 的 trunk、6 个 source、P0 calibrator（权重逐元素相同，已核对）和 OOS session 完全相同，所以 year 的 source 模拟文件与 month 的完全相同，直接复制（把 stratum 标签改写）而没有重算。
- **一个与 days-IS 有关的说明**：days 的 6 个原始 source 里含 20150511，它同时是 year 的 target。这只影响 days 的 IS 选择（只用于 days 的评测），year 的评测从不使用 days 的选择；这是 P1 就有的 D1 设定。

模拟给出的收益是实际收益的上界参考：OOS 选出的 R2 在 source 模拟里相对 P0 的收益是 month/year +0.02/+0.07/+0.10/+0.13/+0.16/+0.17（B=1…33），实际 target 上是 +0.019/+0.004/+0.065/+0.061/+0.093/+0.102（month），在 B≥4 约为模拟的 50–65%（B=1 为 95%，B=2 仅 6%）；IS 模拟的收益 ≤ +0.003（所以 IS 选出的规则几乎是 no-op，其在 target 上的收益来自 target 与 IS session 的差异）。

## 2. 结果

全部数字为 query R²（物理单位），mean ± 95% CI（样本为 target × draw × seed 平均后的配对点：days 16，month/year 8），表中 R1/R3 的 readout 变体（full/diag/lr）按 source 模拟分数每个 B 选一个（`chosen_readout_kind.csv`）。完整表见 `tables_p3.md`（含 IS 与 OOS 两套、每个 readout kind、R1-stream、R1 on P-prior）。

### 2.1 Headline 表（days: IS；month, year: OOS）

days（set=IS）：

| method | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior [ref] | 0.333 | 0.337 | 0.360 | 0.360 | 0.362 | 0.361 |
| P0 | 0.647 | 0.702 | 0.708 | 0.723 | 0.732 | 0.738 |
| ui_reg [ref, backprop] | 0.378 | 0.432 | 0.527 | 0.607 | 0.661 | 0.704 |
| P0+ui_reg [ref, backprop] | 0.654 | 0.689 | 0.701 | 0.720 | 0.730 | 0.746 |
| R1 | 0.649 | 0.704 | 0.711 | 0.724 | 0.733 | 0.739 |
| R1 on P-prior table | 0.335 | 0.344 | 0.368 | 0.377 | 0.403 | 0.414 |
| R1-stream | 0.648 | 0.703 | 0.710 | 0.723 | 0.732 | 0.740 |
| R2 F7-J | 0.650 | 0.704 | 0.713 | 0.728 | 0.737 | 0.746 |
| R3 | 0.651 | 0.704 | 0.714 | 0.728 | 0.737 | 0.747 |

month（set=OOS）：

| method | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior [ref] | 0.315 | 0.348 | 0.319 | 0.332 | 0.328 | 0.333 |
| P0 | 0.466 | 0.518 | 0.535 | 0.565 | 0.558 | 0.566 |
| ui_reg [ref, backprop] | 0.391 | 0.484 | 0.542 | 0.632 | 0.683 | 0.720 |
| P0+ui_reg [ref, backprop] | 0.487 | 0.543 | 0.612 | 0.653 | 0.709 | 0.731 |
| R1 | 0.468 | 0.521 | 0.550 | 0.581 | 0.597 | 0.639 |
| R1 on P-prior table | 0.316 | 0.353 | 0.365 | 0.406 | 0.437 | 0.492 |
| R1-stream | 0.467 | 0.519 | 0.548 | 0.578 | 0.596 | 0.634 |
| R2 F7-J | 0.485 | 0.522 | 0.600 | 0.625 | 0.651 | 0.669 |
| R3 | 0.486 | 0.528 | 0.600 | 0.632 | 0.668 | 0.687 |

year（set=OOS）：

| method | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior [ref] | 0.145 | 0.144 | 0.135 | 0.133 | 0.132 | 0.150 |
| P0 | 0.198 | 0.250 | 0.263 | 0.256 | 0.286 | 0.315 |
| ui_reg [ref, backprop] | 0.191 | 0.217 | 0.267 | 0.290 | 0.367 | 0.478 |
| P0+ui_reg [ref, backprop] | 0.228 | 0.265 | 0.321 | 0.309 | 0.393 | 0.488 |
| R1 | 0.199 | 0.252 | 0.267 | 0.262 | 0.309 | 0.389 |
| R1 on P-prior table | 0.145 | 0.147 | 0.144 | 0.141 | 0.180 | 0.239 |
| R1-stream | 0.199 | 0.250 | 0.261 | 0.254 | 0.304 | 0.376 |
| R2 F7-J | 0.219 | 0.265 | 0.325 | 0.336 | 0.377 | 0.361 |
| R3 | 0.219 | 0.263 | 0.325 | 0.334 | 0.386 | 0.435 |

（CI 见 `tables_p3.md`；month 的 R2/R3 CI 在 ±0.005–0.034，year 的 CI 在 ±0.012–0.093，B=33 的 year 因一个失稳 draw 变宽。）

### 2.2 Gap 恢复（配对 delta vs P0；括号为 delta / (P0+ui_reg − P0)，仅当 |gap| ≥ 0.02）

| stratum (set) | 方法 | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 | 汇总 B≥4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| days (IS) | gap | +0.006 | −0.013 | −0.007 | −0.003 | −0.002 | +0.008 | −0.001 |
| | R1 / R2 / R3 | +.001 / +.003 / +.004 | +.001 / +.001 / +.001 | +.003 / +.005 / +.006 | +.001 / +.004 / +.004 | +.002 / +.005 / +.006 | +.002 / +.008 / +.010 | 无 gap 可恢复 |
| month (OOS) | gap | +0.021 | +0.025 | +0.077 | +0.088 | +0.151 | +0.165 | +0.120 |
| | R1 | +.001 (6%) | +.003 (14%) | +.015 (19%) | +.017 (19%) | +.039 (26%) | +.072 (44%) | 30% |
| | R2 | +.019 (90%) | +.004 (16%) | +.065 (84%) | +.061 (69%) | +.093 (62%) | +.102 (62%) | 67% |
| | R3 | +.020 (95%) | +.010 (39%) | +.065 (85%) | +.067 (76%) | +.110 (73%) | +.120 (73%) | 75% |
| month (IS) | R1 / R2 / R3 | 6/34/40% | 3/23/26% | 9/40/48% | 7/44/49% | 12/46/55% | 20/52/62% | 13/47/55% |
| year (OOS) | gap | +0.029 | +0.015 | +0.058 | +0.054 | +0.107 | +0.172 | +0.098 |
| | R1 | +.001 (4%) | +.003 | +.004 (7%) | +.006 (11%) | +.023 (22%) | +.073 (43%) | 27% |
| | R2 | +.021 (71%) | +.015 | +.062 (107%) | +.081 (151%) | +.091 (85%) | +.046 (27%) | 72% |
| | R3 | +.021 (72%) | +.014 | +.062 (107%) | +.078 (146%) | +.100 (94%) | +.119 (69%) | 92% |
| year (IS) | R1 / R2 / R3 | 4/12/15% | – | 4/65/68% | 5/86/89% | 9/70/76% | 12/58/66% | 9/66/72% |

注意事项：(i) B=2 的 year gap 只有 +0.015（<0.02，不报分数）；(ii) “恢复比例”在 gap 小的格子里不稳定（year B=8 的 151% 是因为 R2 超过了 P0+ui_reg：0.336 vs 0.309）；(iii) 审查意见指出 ui_reg 在 B=33 很可能欠收敛（步数触及 300 的上界，source R² 仍在上升），因此 B=33 的 gap（month +0.165，year +0.172）可能被低估，B=33 的恢复比例可能被高估；(iv) days-OOS 的负面结果：R1 −0.004/−0.007/−0.014/−0.035/−0.042/−0.041，R3 +0.002/−0.004/−0.011/−0.054/−0.038/−0.057，R2 +0.004/−0.003/+0.003/−0.002/−0.002/−0.018（`gap_recovery.csv`）。

**梯度对齐（诊断，非校准）**（`diag_gradient_alignment.csv`；4 个 draw × seed0 × B，P0 表上第一步，中位数；cos_B 为与 B 个 trial 上损失的真实负梯度，cos_query 为与 30 个非校准 trial 上损失的真实负梯度）：

| stratum | L=0 | L=8 | L=16 | L=32 | cos_query (L=32) |
|---|---:|---:|---:|---:|---:|
| days | 0.135 | 0.234 | 0.275 | 0.296 | 0.168 |
| month | 0.156 | 0.324 | 0.361 | 0.413 | 0.313 |
| year | 0.221 | 0.368 | 0.414 | 0.453 | 0.283 |

沿 OOS 选出的轨迹（cos_B，第 1/2/3/5/10 步）：days 0.30/0.21/0.17/0.14/0.14，month 0.41/0.24/0.19/0.16/0.14，year 0.45/0.25/0.20/0.16/0.05。**G3′（≥0.5）未达到**（最好的是 year 的 0.45）；J̄ 的反馈只在第一步有可用的方向，之后依赖收缩与有限步数。尽管余弦不高，下面的控制实验表明这个方向是真实的。

### 2.3 不稳定性

OOS 选出的 B=33 规则（month/year：S=5，η=10，ρ=0.3，L=32）比较激进。year B=33 的 8 个 draw 里有 1 个 R2 塌缩（draw 4：P0 0.258 → R2 0.041），其余 7 个 +0.017 到 +0.106；R3 的 readout 岭回归把它救回到 0.292，这是 R2 在 year B=33 的均值（0.361）远低于 R3（0.435）的原因。month 没有塌缩。fraction of draws worse than comparator by 0.05（`gap_recovery.csv`）：month/year 的 R1/R3 为 0，R2 仅 year B=33 为 12.5%。这是单一失稳样本，不是系统性结论，但说明“按 source 模拟选的 η 在 B=33 缺少安全余量”。

### 2.4 控制实验：反馈核（`ablation_feedback_kernel.csv`，R2 在已选规则下，delta vs P0，B∈{2,8,33}，8 draws × 2 seeds，目标 query 仅用于报告）

| variant | days (IS rule) B=2/8/33 | month (OOS rule) B=2/8/33 | year (OOS rule) B=2/8/33 |
|---|---|---|---|
| J̄（已选 L） | +.001 / +.004 / +.008 | +.004 / +.061 / +.102 | +.015 / +.081 / +.046 |
| −J̄ | −.015 / −.009 / −.041 | −.183 / −1.248 / −1.352 | −.385 / −.908 / −1.132 |
| 同范数随机核 | .000 / .000 / .000 | −.007 / −.047 / −.076 | −.014 / −.060 / −.124 |
| e=0（只有收缩） | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| J̄ 截断 L=0 | 0 / 0 / 0 | +.002 / +.011 / +.018 | +.002 / +.011 / +.016 |
| J̄ 截断 L=8 | +.001 / +.002 / +.005 | +.009 / +.055 / +.089 | +.011 / +.047 / +.062 |
| J̄ 截断 L=16 | +.002 / +.004 / +.007 | −.003 / +.050 / +.090 | +.012 / +.064 / +.030 |

结论：(1) 增益来自 J̄ 的方向（−J̄ 与随机核都变差）；(2) 时间卷积是必要的：L=0 只有已选 L=32 增益的 10–20%，L=8 已有约 60–90%；(3) days 的 IS 规则基本是 no-op（|delta|≤0.008）。

## 3. R4：identity-free + label-free P0

训练 `p0`，`--prior-mode globalmean --label-free`（label-free：描述量中的行为互协方差、方向调谐、T4 系数以及 κ 可靠性特征置零；globalmean 下 prior 列置零，所以 calibrator 看不到 electrode 映射，也看不到任何标签衍生量），pool O，2 seed，3 个 stratum（month 与 year 的 O 池 trunk 与 source 相同，各自独立训练，结果相近），评测协议同 P2（8 draws，同 draw）。训练均满足收敛规则（best step 200–400，≤0.8×预算，不触发翻倍；val R² days 0.057–0.134，month/year 0.38–0.41，val prior −0.02）。注意 val 是 trunk 的训练 session，month/year 的 val 数字（0.38–0.41）远高于 target（0.04–0.09），与 P1 已说明的 in-sample 偏差一致。

query R²（mean±CI，两个 seed 平均）：

| stratum | 方法 | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 |
|---|---|---:|---:|---:|---:|---:|---:|
| days | P-prior (globalmean) | −0.010 | −0.015 | −0.012 | −0.008 | −0.008 | −0.010 |
| | P0 aligned（身份+标签） | 0.647 | 0.702 | 0.708 | 0.723 | 0.732 | 0.738 |
| | P0 aligned，label-free（P1 的 `_nolab`） | 0.684 | 0.726 | 0.723 | 0.734 | 0.742 | 0.746 |
| | P0 identity-free，有标签（P2） | 0.306 | 0.588 | 0.629 | 0.687 | 0.695 | 0.696 |
| | **P0 identity-free + label-free [R4]** | 0.126 | 0.155 | 0.183 | 0.218 | 0.220 | 0.226 |
| month | P0 aligned | 0.466 | 0.518 | 0.535 | 0.565 | 0.558 | 0.566 |
| | P0 aligned，label-free | 0.417 | 0.469 | 0.457 | 0.484 | 0.479 | 0.479 |
| | P0 identity-free，有标签 | 0.054 | 0.231 | 0.327 | 0.433 | 0.444 | 0.466 |
| | **R4** | 0.027 | 0.079 | 0.045 | 0.063 | 0.063 | 0.054 |
| year | P0 aligned | 0.198 | 0.250 | 0.263 | 0.256 | 0.286 | 0.315 |
| | P0 aligned，label-free | 0.211 | 0.237 | 0.238 | 0.229 | 0.245 | 0.263 |
| | P0 identity-free，有标签 | −0.025 | 0.063 | 0.101 | 0.117 | 0.205 | 0.246 |
| | **R4** | −0.024 | 0.058 | 0.067 | 0.039 | 0.088 | 0.087 |

解读：**在没有 electrode 身份时，days 在 B≥8 的 0.69 绝大部分来自标签衍生的 per-unit 统计（tuning），不是无标签的发放统计**：去掉标签后从 0.687 降到 0.218，而无身份且无标签的 P-prior 本身是 −0.01，所以无标签统计贡献约 +0.23，标签贡献约 +0.47。有身份时则相反：去掉标签几乎不影响（days 0.734 vs 0.723，month −0.08，year −0.03）。身份元数据与标签调谐因此是两条可互相替代的信息来源，它们在 days 层的“冗余”程度很高，月/年层标签仍有价值。这回答了 P2 报告里“无法区分调谐与无标签统计”的遗留问题。局限：R4 的 P0 只有 2 个 seed，早停在 200–400 步（val 增益很小）；label-free 掩码只覆盖描述量与可靠性特征，B-trial 归一化仍使用“有标签的有效 bin”（有效 bin 的定义依赖标签是否有限）。

## 4. 代价（解析，含 replay forward 与 replay buffer）

方法：`ssm_decode/refine_cost.py`，由显式张量形状得到，trunk MAC 从实例化的 trunk 读出（`fwd_refine.trunk_macs_per_bin`：Linear 的 in×out + Mamba3 SISO 的状态更新与读出各 1 MAC/状态元素，2·d_inner·d_state；门控、RoPE、LayerNorm、短卷积未计）。000688 与 HD-sEMG 的 trunk 为 K→128 宽、2 层、d_state 32，总 383,488 / 384,640 MAC/bin（线性 350,720 / 351,872 + scan 32,768），状态 16,384 个元素（16-bit 即 32 KB）。**没有沿用 P2 的总量**（审查指出 P2 漏掉了 q_* 每 trial 暂存、lag 延迟线、per-trial staging 副本、P4 的第二遍 φ buffer，并把 activation×activation 项记为固定权重）；这里只对 P3 新增的操作逐项计数，P0 的基线 MAC 只做对照（我重新推导了 MAC 部分，未审计 P0 的存储）。完整分项见 `cost_p3_detail.json`（每个方法、每个 B 的 MAC 与 SRAM 条目）。

计入的条目：
- replay 前向：B·T 个 bin × (trunk + read-in K·N)，R2 为 S 遍，R3 为 S+1 遍；
- readout 累加：full 对称 Gram (H+1)(H+2)/2 + D(H+1) MAC/bin，diag (H+1)+D(H+1)，lr-r 为 (H+1)r + r(r+1)/2 + Dr MAC/bin（先投影到固定基）；求解：full (H+1)³/6 + 2D(H+1)²，lr-r 为 r³/6 + 2Dr² + Dr(H+1)；
- F7-J：反馈卷积 (L+1)·D·K MAC/bin、外积 K·N MAC/bin、每步 3KN 的预条件与更新；
- SRAM：原始 replay buffer（B·T·(N·b_x + D·b_y + 1)/8 字节：计数 b_x bit、标签 b_y=16 bit、有效位 1 bit；e_t 的 x_{t−L} 直接从该 buffer 偏移读取，无副本）、残差延迟线 (L+1)·D·16 bit、G 累加器 KN、T0（收缩中心）KN、q_i、当前表 T、mean/std、readout 统计（fp32 累加）、readout 的新 V、特征向量、trunk 状态（16-bit，与推理共用）；Cholesky 原地进行（无额外 scratch）；J̄ 核与低秩基 U 为只读 ROM（int8），单列。统计累加器的位宽没有评测过低于 fp32 的精度。

000688（T=252 bin/trial，取 0616/0617 两个 target 的平均；N=26 是该数据集最大的 target；3-bit 计数，因为检查的 7 个 session 里最大计数为 5–7，>3 的 bin 占比 ≤5e-4；标签 16-bit）：

| 方法 | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 |
|---|---:|---:|---:|---:|---:|---:|
| P0 校准（MAC，对照） | 0.5 M | 0.8 M | 1.3 M | 2.4 M | 4.7 M | 9.4 M |
| R1（lr16）重放 | 97 M | 195 M | 389 M | 778 M | 1.56 G | 3.21 G |
| R1（full）重放 | 99 M | 198 M | 396 M | 792 M | 1.58 G | 3.26 G |
| R1-stream（lr16） | 97 M | 195 M | 390 M | 780 M | 1.56 G | 3.22 G |
| R2 所选 (S,L)：MAC | 0.29 G (S=3) | 0.97 G (5) | 1.94 G (5) | 7.75 G (10) | 15.5 G (10) | 16.0 G (5) |
| R3（lr16）所选 (S,L) | 0.39 G | 1.16 G | 2.33 G | 8.53 G | 17.1 G | 19.2 G |
| 实时当量（R1，20 ms bin） | 5 s | 10 s | 20 s | 40 s | 81 s | 166 s |
| 原始 replay buffer（3-bit 计数） | 3.4 KB | 6.8 KB | 13.7 KB | 27.3 KB | 54.6 KB | **112.7 KB** |
| 同上，4-bit 计数 | 4.2 | 8.4 | 16.9 | 33.7 | 67.4 | 139.1 |
| 同上，8-bit 计数 | 7.4 | 14.8 | 29.7 | 59.3 | 118.6 | 244.6 |
| R1（lr16，replay）总 SRAM（含 32 KB trunk 状态） | 38.4 | 41.9 | 48.7 | 62.3 | 89.7 | 147.7 KB |
| R1-stream（lr16）总 SRAM | 38.4 | 38.4 | 38.4 | 38.4 | 38.4 | 38.4 KB |
| R2（S=5，L=16）总 SRAM | 38.7 | 42.1 | 49.0 | 62.6 | 89.9 | 148.0 KB |

要点（000688）：(a) 每次细化的 MAC 由 trunk 重放主导：占 R1 的 99.4%、R2 的 98–99.8%，全部是 RRAM 可映射的固定权重 MAC；P0 校准本身只有 ~2 MMAC，所以“细化”的代价是 P0 的 ~330 倍（R1）到 ~3000 倍（R2，S=10）。(b) **原始 replay buffer 本身**在 3-bit 计数 + 16-bit 标签下能放进 128 KB，一直到 B=33（112.7 KB）；4-bit 计数 B=33 放不下（139 KB）；8-bit 计数在 B≥16 放不下。(c) **总工作集**（把 32 KB 的 trunk 状态、readout 统计一起算）：B=16 约 90 KB 可放进 128 KB，B=33 约 148 KB 放不下。fp32 的 full Gram 单独就是 33.5 KB；lr16 把它降到 0.7 KB（加 2 KB 的 int8 基 ROM）。(d) 把 trunk 状态视为与推理共用（校准期间推理暂停时成立；如果 anytime 校准与在线解码并行，需要再加一份 32 KB 的状态）。(e) 所有行都按“重放 = 重新通过 trunk”计，没有利用 trunk 激活的缓存（B 个 trial 的激活随表变化，无法缓存）。

HD-sEMG 规模（N=256、K=16、D=5、1250 bin/trial = 25 s×50 Hz、8-bit 样本、16-bit 标签、H=128、B≤33；与 P2 同样假定沿用 000688 的 AP trunk 配置）：

| 量 | B=1 | B=4 | B=8 | B=33 |
|---|---:|---:|---:|---:|
| P0 校准（MAC，对照） | 96.6 M | 372 M | 739 M | 3.04 G |
| R1（lr16）/ R1-stream | 489 M | 1.96 G | 3.9 G | 16.1 G |
| R2（S=5，L=16） | 2.5 G | 9.9 G | 19.7 G | 81.3 G |
| R3（lr16，S=5，L=16） | 3.0 G | 11.8 G | 23.6 G | 97.4 G |
| 原始 replay buffer（8-bit） | **325 KB** | 1.30 MB | 2.60 MB | 10.7 MB |
| R1-stream（lr16）总 SRAM，含 trial 缓冲 | 377 KB | 377 KB | 377 KB | 377 KB |
| R1-stream（lr16）总 SRAM，running normalisation（**未评测**） | 52.6 KB | 52.6 KB | 52.6 KB | 52.6 KB |

HD-sEMG：replay buffer 在 B=1 就超过 128 KB 的 2.5 倍，**replay 版本（R1、R2、R3）在这个规模不可行**；R1-stream 的 readout 统计只有 0.7 KB（lr16）到 33.5 KB（full），但“首个 trial 用自身统计归一化”的一个 trial 缓冲（333 KB）本身就放不下，只有改成 running（Welford）归一化才可行，这一点**没有评测**，不能声称精度不变。F7-J 没有无 replay 的形式（每步需要对全部 B 个 trial 的前向），在 HD 规模不可部署，除非限制 B 与 trial 长度；这里没有实现 streaming F7-J。

## 5. 便宜的替代方案：流式累加 readout 统计（已评测）

R1-stream 相对 R1（replay）的精度（OOS 选择，month / year，full 变体，B=1…33）：month 0.000/−0.002/−0.002/−0.003/−0.001/−0.005，year −0.001/−0.003/−0.006/−0.008/−0.005/−0.013；lr16 变体：month 0.000 到 −0.010（B=4 −0.008，B=33 −0.010），year −0.001 到 −0.007。IS 选择（days）差 ≤0.002。即：**对 readout 端，流式累加在 month 几乎无损、year 损失 0.001–0.013（B 越大越多），而 SRAM 与 B 无关**（38.4 KB，额外约 6.4 KB），MAC 与 replay 版相同（每个 bin 仍前向一次）。损失来自特征由更早的、更差的表产生以及前缀归一化。它不需要 replay buffer，是 HD-sEMG 规模唯一可行的 R1 形式（在 running normalisation 的前提下）。同时要说明：R1 本身只占 gap 的 10–30%，真正有价值的 R2 无法流式化。

## 6. 局限与未做

- **选择集合**：IS 与 OOS 给出显著不同的规则，headline 的“days=IS、month/year=OOS”是事先固定的，依据是间隔量级；这只是一个粗糙的间隔匹配启发式，没有一般化。OOS 的 month/year 选择集合只有 4 个 session，其中 3 个晚于 month target；days OOS 用的是距离 trunk 一年以上的 session。网格边缘：η 常在下边缘（η=1），L 常在上边缘（L=32）；没有扩展到 η<1 或 L>32。
- **没有做 hindsight 对照**（用 target query 挑最佳 η/λ）；因此无法区分“选择失配”和“方法上限”。
- **target 只有 4 个**（days 2，month 1，year 1；month 与 year 共用 trunk/source/P0），CI 是 draw 层面的，低估了 target 层面的变化；year B=33 有一个失稳 draw。2 个 seed 只改变 P0 calibrator，J̄ 与规则不依赖 seed。
- **P0+ui_reg 与 ui_reg 是 P2 的参照行**：B=33 可能欠收敛（审查意见 (3)），没有在这里重跑，所以 B=33 的恢复比例可能偏高。
- **J̄ 的代表性弱**（§1.2、§2.2），G3′ 的余弦门槛（≥0.5）未达到；F7-Jnet / F7-G / F4 不在本任务范围，未做。
- **未评测**：readout 统计与 replay buffer 的低位宽、running normalisation 版的 R1-stream、streaming F7-J、噪声/器件非理想；代价是解析计数，不是硬件测量，且 trunk 的 gating/RoPE/LayerNorm/短卷积未计。
- R4 的 label-free 只屏蔽描述量与可靠性特征；归一化的有效 bin 仍依赖标签是否有限；R4 只有 2 个 seed。
- 没有推送，没有改动 P1/P2 的 `results/`，没有触碰 DB6 / Hyser / sEMG 相关文件，没有写 Overleaf（本轮未被要求）。

## 7. 复现（GPU0，全部可续跑）

`bash scripts/run_pc3_r4.sh`（R4）；`python scripts/run_pc3_refine.py jacobian|basis --stratum S`；`python scripts/run_pc3_refine.py select --stratum S --set is|oos`，`select-r3`，`eval`，`ablate`（见 `scripts/run_pc3_queue.sh`、`run_pc3_phase2.py`、`run_pc3_ablate.sh`）；`python scripts/analyze_pc3.py`；`python scripts/pc3_cost_report.py`。环境变量 `PC_OUT=results/population_calib_p3_20261016`。
