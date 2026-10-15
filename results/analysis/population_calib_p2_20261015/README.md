# P2 迭代（工作线 A′）：identity-free 校准、公平梯度参照、校准通路量化、架构代价

日期：2026-10-15/16。执行者：工作线 A′（GPU0）。规格见 `docs/feasibility_p1_20261015.md` §5。前序结果见 `results/analysis/population_calib_20261015/README.md`（含 §0.1 独立审查）。
代码：`ssm_decode/{calib_quant,calib_cost,ui_reg}.py`、`ssm_decode/population_calib.py`（新增 `prior_mode`、`identity_free_prior`、`softmax_fn`、`weight_quant`）、`scripts/{run_pc2_queue,run_pc2_uireg,run_pc2_quant,pc2_cost_report,analyze_pc2}.py`、`scripts/run_population_calib.py`（新增 `--prior-mode`、`--init-from/--qat`、`PC_OUT`）、`tests/test_population_calib_p2.py`（30 个新测试，连同 P1 的 21 个共 51 个，全部通过）。
原始 rows：`results/population_calib_p2_20261015/`（gitignored）；本目录是汇总 csv/json（`tables_a1_a2_a3.md` 是全部 markdown 表，`a*_rows.csv.gz` 是逐行 rows）。
约束检查：trunk 冻结；候选行（P*、QAT）target 端 0 次反传；只用 B 个有标签 trial；未使用 0623/0625/0626，未读 `official_evalai/`；全部选择（ui_reg 的 lambda/steps/lr、accumulator/descriptor 的定点 scale、QAT 早停）只用 source。A′1 的 3072 条 eval row 全部满足 H1/H2/H4/H5 与 unit-ratio 检查（`a1_flags.json`）。`ui_*` 行是标注为非部署的梯度参照。P1 结果在改动后逐位复现（96 行 P0 eval row，最大差 9e-16）。

## 0. 结论摘要

1. **A′1（无 electrode 对齐）：population pooling 在没有身份元数据时有弱的正向趋势，但幅度只在 seed 噪声底的量级，且没有补回身份元数据的损失。**
   - 去掉身份后 P0 的 R²（globalmean / zero 两种 prior，B=1 / 8 / 33）：days 0.31/0.69/0.70 与 0.36/0.68/0.70（aligned P0 为 0.65/0.72/0.74）；month 0.05/0.43/0.47 与 0.12/0.43/0.46（aligned 0.47/0.57/0.57）；year −0.03/0.12/0.25 与 0.02/0.12/0.23（aligned 0.20/0.26/0.32）。相对 aligned P0 的差距（三个 stratum 平均，globalmean / zero）：B=1 −0.33 / −0.27，B=8 −0.10 / −0.10，B=33 −0.07 / −0.08。days 层 B≥8 只差 0.03–0.04，month/year 差 0.07–0.14。**也就是说，身份元数据在 B=1–2 时极其重要（+0.3），在 B≥8 的 days 层几乎不需要，在 month/year 层仍值 0.1。** 这修正了 P1 审查表中“电极身份是主要来源”的外推：那张表是用 aligned 训练的 calibrator 在测试时替换 prior，存在训练/测试失配；匹配训练后，B≥8 时仅靠 unit 自身统计（含行为互协方差和方向调谐，即描述量中的标签信息）就能拿到 aligned 的约 94–95%（days，B≥8）。
   - pooling 的配对差（P×−P0，同 mode、target、B、draw、seed；所有 B 汇总；三个 stratum 的均值）：P4 globalmean +0.018 / zero +0.027（aligned 参照 −0.004）；P3 +0.010 / +0.020（aligned −0.004）；P2 −0.011 / +0.010（aligned −0.005）。P4 在两种 prior 下 6 个 (mode×stratum) 单元全部为正（+0.006 到 +0.056），其中 3/6 超过 seed 噪声底（比值 1.02、1.40、1.66，对应噪声底 0.018、0.026、0.034），没有一个超过 1.7 倍。结论：**弱正向（约 +0.02 到 +0.03，最好 +0.056），与 P1 在 aligned 条件下的“pooling 无增益”方向相反，但幅度和噪声相当，只有 2 个 seed、4 个 target，不能说已被证实**。P2 无稳定增益，P3 介于二者之间。pooling 不能替代身份：P4 identity-free 相对 aligned P0 的差距为 −0.30/−0.08/−0.07（B=1/8/33，globalmean）和 −0.24/−0.08/−0.06（zero），只比 P0 的 −0.33/−0.10/−0.07 和 −0.27/−0.10/−0.08 好 0.00–0.03。
   - globalmean 与 zero 两种 prior 没有稳定差别（zero 在 B=1 略好，差距在 CI 内）。P-prior 本身无信息（R² −0.03 到 −0.005）。

2. **A′2（公平梯度参照 ui_reg）：正则化主要在 B=1–2 有帮助；B≥8 时选出的 lambda 很小，ui_reg≈P1 的无正则 ui_elec。P1 关于 amortization gap 的结论成立，且量化为 P0 在 month/year 层 B=33 时落后 0.15–0.16。**
   - ui_reg 的 R²（B=1/8/33）：days 0.378/0.607/0.704，month 0.391/0.632/0.720，year 0.191/0.290/0.478。相对无正则 ui_elec（5 个共享 draw 配对）：B=1 +0.09/+0.07/+0.06，B=8 −0.005/+0.015/−0.028，B=33 0.000（选出的 lambda≤1e-3，近似无正则）。
   - P0 − ui_reg（配对，CI 见 `a2_paired.csv`）：days +0.27/+0.12/+0.03（B=1/8/33），month +0.08/−0.07/−0.15，year +0.01/−0.04/−0.16。days 层 P0 在所有 B 上不低于 ui_reg；month/year 层 B≥8 起 ui_reg 更好（B=4 持平）。
   - **P0+ui_reg（从 P0 的表出发，用同一套 source 选出的正则化细化）：days 层不变（−0.013 到 +0.008，在 CI 内），month 层 +0.02/+0.09/+0.17，year 层 +0.03/+0.05/+0.17（B=1/8/33）；在 month/year 上也不低于单独的 ui_reg（+0.01 到 +0.10）。** 即在 month/year，静态表的上限（约 0.72–0.73 / 0.49）明显高于 P0（0.57 / 0.32），P0 只是一个好的初始点；days 层 P0 已在这个 trunk 的静态表上限。这对硬件的含义：如果存在 target 端可部署的局部细化规则（不反传 trunk），month/year 在 B=8 约有 +0.05–0.09、B=33 约有 +0.17 的空间；days 层没有。

3. **A′3（量化，评测不重训）：对 P0 与 P4 一致。**
   - 权重（calibrator，per-output-channel）：int8 的 ΔR² 绝对值 ≤0.0001（预测 RMS 变化 0.5–0.7%）；int4 ≤0.002（RMS 2–7%）；**ternary（TWN）PTQ 掉 0.03–0.06（RMS 22–35%）**。用 source 上 300–1200 步 STE 的 ternary QAT 微调（已做，不属“重训”，但需说明）后 ΔR² 回到 −0.012 到 +0.001（target）和 −0.010 到 −0.001（source val），回收 PTQ 损失的约 80–100%。
   - descriptor accumulators（原始累加量，每个 key 一个 power-of-two 步长，只由 source 拟合）：**16 bit 无损**（|ΔR²|≤0.0006）；**12 bit 在 B≥8 几乎无损（≥−0.003）但 B=1 掉 0.14（target）/0.06（val）**，原因是步长按 B=33 的量程选，B=1 只用到低位；让 block exponent 随 B 变化（再留 2 bit 余量）后 12 bit 在 B=1 只掉 0.001；**8 bit 不可用**（−0.3 到 −0.8，且大量行出现非有限预测，见 §4.2）。12/8 bit 的分组消融：12 bit 时只有 dir 组（`sdir`/`cdir`/`sphi*`）在 B=1 明显掉（target −0.13），其余 ≥−0.006；8 bit 时 norm（`n/sx/gram`）、dir、cross（`sy/syx`）都崩（−0.1 到 −0.9），ac 约 −0.06 到 −0.07，quad 无影响。
   - 最终描述量的定点（16/12/8 bit，每列一个步长）：ΔR² 绝对值 ≤0.0004，RMS 变化 0.5–1%，即描述量本身 8 bit 就够，瓶颈在累加器，而不在描述量。
   - 输出表 int8（每行一个 scale）：|ΔR²|≤0.0001；int4：0 到 −0.005（RMS 7–9%）。
   - P4 softmax 用 exp2 LUT（256×16 bit）加 reciprocal（精确或 256 项 LUT）：ΔR² 绝对值 <0.00005，RMS 变化 0.4–0.5%，等于数值噪声底（见 §4.2），P3/P4 的 softmax 可以用 LUT 实现。
   - 组合：int8 权重 + 16 bit 累加器与描述量 + int8 表 + LUT：ΔR² 绝对值 ≤0.0006，RMS 0.7–1.1%（P4 B=1 target 为 2.8%）。int4 权重 + 12 bit + int8 表：B≥8 掉 ≤0.004，B=1 掉 0.02–0.07。
   - **source val 与 target 的差异**：val 上的 ΔR² 一般小于或等于 target（target 的 FP R² 低，噪声交叉项更大）；两者的排序一致。RMS 变化列才是量化本身的扰动，二者在 val 与 target 上相近（ternary：val 22–25%，target 29–35%，同一量级）。

4. **A′4（代价，解析）**：详见 §5。000688（N=26, K=10, D=2；平均可用 trial 约 220 bin）一次 B=8 校准：累加 1.94 M MAC（每 bin 1101 MAC，Gram 占 676），P0 的 finalize+前向+写表 0.22 M MAC，P4 为 0.66 M MAC；总 2.2 M（P0）/2.6 M（P4）MAC；SRAM（P4，16 bit 累加器、16 bit 描述量、int8 双缓冲表）约 7.8 KB，表本身 260 B（int8）/1040 B（fp32）。HD-sEMG（N=256, K=16, D=5, 每 trial 1250 bin）：每 bin 73.5 k MAC，一次 B=8 校准 0.74 G MAC，其中固定权重部分只占 0.3%（P0）/1.0%（P4）；**Gram 累加器 131 KB（16 bit）/262 KB（fp32）远超过 32 KB 的表预算**，总 SRAM（同上配置）约 212 KB；表本身 4 KB（int8）/16 KB（fp32，双缓冲刚好 32 KB）。结论：在 HD-sEMG 规模，**瓶颈是 N² Gram 累加，不是 calibrator**；换成 N·K 的 latent 累加（解析，未评测）每 bin 可降到 16.1 k MAC，但会改变特征。

## 1. 实现与设置

**A′1 identity-free。** `PopulationCalibrator(prior_mode=...)`，`prior_mode ∈ {aligned, globalmean, zero}`，其余不变（P0/P2/P3 m=8/P4 m=8，pool O，2 seed，3 stratum，B∈{1,2,4,8,16,33}，每格 8 个共享 draw，训练协议、早停规则、unit dropout、episodic B 采样同 P1，配对 draw 与 P1 完全相同）。
- `globalmean`：所有 source 表的全部 unit 列取平均（每个 source unit 等权），得到一个 K 维向量，赋给每个 unit（`identity_free_prior`，训练时对原始 source 做 leave-one-out）；不使用 `unit_to_electrode`。`zero`：零表。
- 描述量：**去掉 prior 列**（calibrator 里 `d_keep` 置零，因为 globalmean 下它对所有 unit 相同，不含信息）；**与 prior 潜变量的相关**：globalmean 下用非对齐 prior 的潜变量重新计算（K 个潜变量都是 `p_k·Σ_n x̃_n`，即各 unit 与群体总活动的相关，K 个通道完全相同，所以带来的是一个群体和相关特征而不是 K 维信息）；zero 下潜变量恒为 0，相关特征同样置零。因此两种条件下 calibrator 都完全看不到 electrode 映射（测试 `test_noalign_calibrator_ignores_prior_column_and_unit_order`：改变 prior 列特征不改变输出；unit 置换后输出等变）。
- 另外加了 identity-free 的 P-prior 行（`prior_*_na_*.jsonl`）。R² 约 −0.03 到 −0.005：该 prior 不提供预测能力。
- 训练诊断：identity-free 的 best step 更晚（P0 days 约 2200–2600，P4 到 3700），均满足“best step 落在最后 20% 前”的收敛规则（`a1_training.csv`）。month-O 与 year-O 仍是同一个训练结果（P1 已说明）。

**A′2 ui_reg。** `loss = MSE(trunk(x_z Tᵀ), y) + lambda·‖T−T_center‖²_F`（Frobenius 平方和，非均值），Adam，每步随机取一个标注 trial（与 `ui_grad_table` 相同的窗口；lambda=0 时逐位等于 `ui_grad_table`，有测试）。网格：lambda∈{0, 1e-4, 1e-3, 1e-2, 1e-1, 1}，lr∈{1e-3, 3e-3, 1e-2}，steps∈{10, 30, 100, 300}（同一条轨迹取快照）。**选择只在 source 上模拟**：对 6 个原始 source session（各用 leave-one-out 电极均值 prior 作为起点和正则中心），每个 B∈{1,2,4,8,16,33} 抽 2 个 draw，query 为同 session 其余 trial，按 B 分别取 12 个 episode 平均 R² 最大的 (lambda, steps, lr)（`a2_selection.json`，网格见 `select_grid_*.csv`）。year 复用 month 的选择（同一 trunk、同一批 source）。
- days：B=1 lambda=1 (lr 1e-3, 300 步)，B=2/4/8 lambda=0.1，B=16 0.01，B=33 0.001；month：B=1/2 lambda=1 (10 步)，B=4/8 0.1，B=16 0.01，B=33 1e-4。lambda 随 B 单调下降，符合预期。B=1 的最优在网格边缘（lambda=1），没有向上扩展网格。
- ui_reg 从电极对齐 prior 起步（正则中心同为 prior）；P0+ui_reg 从 P0 的表起步，正则中心取 P0 的表，使用同一套按 B 选出的 (lambda, steps, lr)（**没有**为 P0 起点另外选参）。归一化、query bin、物理单位 R² 与 P 行相同（B 个 trial 的有效 bin 统计）。4 个 target × 6 个 B × 8 个 draw；ui_reg 1 个 seed（seed 只影响窗口采样），P0+ui_reg 用 P0 的 2 个 seed，配对分析时先对 seed 取平均。

**A′3 量化。** 对 P1 已训练的 P0 与 P4(m=8) 检查点（aligned，pool O，每个 stratum 2 个 seed）做评测；B∈{1,8,33}，每格 8 个 draw；target 沿用 P1 的 draw，source val 用 `qval-` 种子的 draw（val session：days 0615，month/year 0306，用 LOO prior）。
- 权重：所有 `nn.Linear` 的权重和 P3/P4 的种子矩阵 u、a，按输出通道（行）对称量化；偏置、`mu0`、输入标准化常数保持 FP（数字电路）；激活不量化。ternary 为 TWN（阈值 0.7·mean|w|，scale=保留项的 mean|w|，按行）。
- accumulators：每个 PopStats key（`n, sx, gram, ac_*, sy, syx, yx_n, sdir, cdir, sphi*, q_*`）有符号定点，位宽 16/12/8，一个 power-of-two 步长（取 source 训练 session 上 B=33 的最大 |值|所需的最小步长，5 个 session × 5 个 draw = 25 个 episode 内取最大），每加入一个 trial 后 round-to-nearest 并饱和。`*_bs` 变体：步长再乘 2^(ceil(log2(B/33))+2)（上限 1），即 block exponent 随 B 变化并保留 2 bit 余量；第一版没有余量时，长 trial 会饱和（B=1 的 acc16_bs 反而更差），已改并重跑。“`accK_<group>`”只量化一组 key（norm = n/sx/gram；ac；cross = sy/syx/yx_n；dir = sdir/cdir/sphi*；quad = q_*）。
- “descriptor accumulators” 有两种读法，**两种都评测**：上面的 accumulators（`acc*`）和最终描述量/可靠性特征 d、r（`desc*`，每列一个 power-of-two 步长，同样来自 source）。
- 输出表：每行（每个潜变量 k）一个 scale 的对称 int8/int4（`tab*`）。
- softmax：`Exp2LUT`（256 项，16 bit 码，15 位小数；项取区间中点，相对误差界 0.135%+code 舍入；输出舍入到 16 位小数）和 reciprocal（`lut`=尾数 256 项 LUT，误差界 0.21%；`exact`=精确倒数）。作用于 P4 的两处 softmax（unit 间 pooling 与 route）；是批量形式，不是 online-softmax 逐块形式（后者的块间 rescale 也要过 LUT，未单独评测）。
- 报告 ΔR² = 量化 − 同一子集上的 FP；`rms_delta` = query bin 上预测差的 RMS（物理单位），`rel_rms` = 它除以 FP 预测的标准差。
- ternary QAT（额外项，非“评测不重训”）：从 P1 检查点出发，source 上 STE fake-quant 微调（lr 3e-4，早停同 P1 规则，300–1200 步），仍无 target 数据；`--init-from/--qat` 实现。

## 2. A′1 结果（pool O；mean±CI）

query R²（节选 B=1/8/33；完整 6 个 B、CI 见 `a1_cells_r2.csv`、`tables_a1_a2_a3.md`）：

| mode : variant | days B=1 | days B=8 | days B=33 | month B=1 | month B=8 | month B=33 | year B=1 | year B=8 | year B=33 |
|---|---|---|---|---|---|---|---|---|---|
| aligned : p0 | 0.647±0.017 | 0.723±0.006 | 0.738±0.005 | 0.466±0.021 | 0.565±0.007 | 0.566±0.005 | 0.198±0.016 | 0.256±0.016 | 0.315±0.030 |
| aligned : p4 | 0.663±0.015 | 0.724±0.006 | 0.742±0.004 | 0.434±0.028 | 0.528±0.013 | 0.528±0.013 | 0.232±0.014 | 0.263±0.017 | 0.317±0.030 |
| globalmean : p_prior | −0.010 | −0.008 | −0.010 | −0.019 | −0.019 | −0.020 | −0.004 | −0.005 | −0.006 |
| globalmean : p0 | 0.306±0.088 | 0.687±0.012 | 0.696±0.009 | 0.054±0.059 | 0.433±0.027 | 0.466±0.014 | −0.025±0.045 | 0.117±0.027 | 0.246±0.024 |
| globalmean : p2 | 0.239±0.100 | 0.654±0.014 | 0.674±0.008 | 0.112±0.054 | 0.430±0.031 | 0.419±0.026 | −0.019±0.038 | 0.117±0.027 | 0.218±0.030 |
| globalmean : p3 | 0.288±0.096 | 0.670±0.016 | 0.668±0.017 | 0.087±0.051 | 0.440±0.024 | 0.457±0.014 | 0.028±0.039 | 0.165±0.021 | 0.249±0.027 |
| globalmean : p4 | 0.318±0.102 | 0.695±0.013 | 0.704±0.012 | 0.072±0.049 | 0.437±0.023 | 0.452±0.014 | 0.023±0.040 | 0.164±0.018 | 0.252±0.028 |
| zero : p_prior | −0.029 | −0.030 | −0.030 | −0.030 | −0.030 | −0.031 | −0.023 | −0.023 | −0.026 |
| zero : p0 | 0.360±0.077 | 0.683±0.009 | 0.702±0.008 | 0.122±0.049 | 0.429±0.027 | 0.458±0.015 | 0.024±0.044 | 0.124±0.022 | 0.230±0.020 |
| zero : p2 | 0.312±0.067 | 0.678±0.012 | 0.703±0.007 | 0.116±0.037 | 0.463±0.023 | 0.493±0.010 | 0.049±0.042 | 0.154±0.022 | 0.234±0.028 |
| zero : p3 | 0.383±0.083 | 0.694±0.009 | 0.711±0.006 | 0.158±0.035 | 0.452±0.027 | 0.474±0.017 | 0.037±0.038 | 0.153±0.018 | 0.236±0.029 |
| zero : p4 | 0.394±0.076 | 0.691±0.008 | 0.708±0.006 | 0.166±0.035 | 0.489±0.021 | 0.517±0.009 | 0.034±0.033 | 0.139±0.021 | 0.224±0.034 |

配对差 P×−P0（同 mode；所有 B 汇总；`a1_paired_vs_p0.csv` 含逐 B、两个 seed 的分别数值和比值）。“floor”= 该 stratum×mode 的 seed 噪声底（两个 seed 的 target 平均 R² 之差的绝对值，对所有变体和 B 取平均）：

| mode | variant | days | month | year | 3 层平均 | floor (days/month/year) |
|---|---|---|---|---|---|---|
| globalmean | p2 | −0.044±0.013 | +0.013±0.012 | −0.003±0.013 | −0.011 | 0.040/0.038/0.026 |
| globalmean | p3 | −0.023±0.011 | +0.016±0.009 | +0.038±0.013 | +0.010 | |
| globalmean | p4 | +0.008±0.010 | +0.010±0.009 | +0.036±0.013 | +0.018 | |
| zero | p2 | −0.015±0.010 | +0.029±0.012 | +0.017±0.012 | +0.010 | 0.018/0.034/0.021 |
| zero | p3 | +0.016±0.009 | +0.027±0.008 | +0.016±0.010 | +0.020 | |
| zero | p4 | +0.019±0.008 | +0.056±0.009 | +0.006±0.009 | +0.027 | |
| aligned（P1 参照） | p4 | +0.008±0.004 | −0.033±0.006 | +0.013±0.007 | −0.004 | 0.014/0.029/0.032 |

CI 是 draws×targets 层面的，偏窄；比较时应对照 floor（见 P1 README §2.4 的同样说明）。逐 B：P4−P0 在 zero 下 month 为 +0.044/+0.060/+0.059（B=1/8/33），year globalmean 为 +0.048/+0.046/+0.006，说明增益集中在 B 较小到中等；days 层 P4−P0 约 +0.01 到 +0.03。

相对 aligned P0 的差距（identity-free 变体 − aligned P0，三个 stratum 平均，`a1_gap_to_aligned_p0.csv` 含分 stratum 和全部 B）：

| mode | variant | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---|---|---|---|---|
| globalmean | p0 | −0.325 | −0.196 | −0.150 | −0.102 | −0.077 | −0.071 |
| globalmean | p4 | −0.300 | −0.164 | −0.126 | −0.083 | −0.070 | −0.071 |
| zero | p0 | −0.269 | −0.187 | −0.132 | −0.103 | −0.076 | −0.076 |
| zero | p4 | −0.239 | −0.156 | −0.099 | −0.075 | −0.054 | −0.056 |

分 stratum 的 P0 差距（globalmean）：days −0.34/−0.04/−0.04，month −0.41/−0.13/−0.10，year −0.22/−0.14/−0.07（B=1/8/33）。

**解读。** (i) 回答“没有身份元数据时 pooling 是否有帮助”：有一个小的、方向一致的正向趋势（P4：+0.018/+0.027），但每个单元只有 1–1.7 倍噪声底，样本只有 4 个 target、2 个 seed；P1 在 aligned 条件下 P4−P0 为 −0.004，且 month 层 −0.033。我们不能把它写成已确认的收益，只能写成“与假设方向一致、需要更多 seed/target 检验”。(ii) 身份元数据的作用是 B 依赖的：B=1–2 时最大，B≥8 后在 days 层几乎消失。(iii) 复杂度成本（§5）：P4 比 P0 多约 3 倍的 calibrator MAC。

## 3. A′2 结果

query R²（mean±CI；`ui_elec` 是 P1 的无正则参照，B∈{1,8,33}、5 个 draw、1 个 seed；其余 8 个 draw）：

| stratum | variant | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---|---|---|---|---|
| days | p0 | 0.647±0.017 | 0.702±0.011 | 0.708±0.009 | 0.723±0.006 | 0.732±0.005 | 0.738±0.005 |
| days | ui_elec | 0.294±0.057 | - | - | 0.596±0.030 | - | 0.700±0.014 |
| days | ui_reg | 0.378±0.025 | 0.432±0.025 | 0.527±0.025 | 0.607±0.026 | 0.661±0.022 | 0.704±0.012 |
| days | p0_ui_reg | 0.654±0.016 | 0.689±0.012 | 0.701±0.010 | 0.720±0.008 | 0.730±0.007 | 0.746±0.005 |
| month | p0 | 0.466±0.021 | 0.518±0.021 | 0.535±0.009 | 0.565±0.007 | 0.558±0.005 | 0.566±0.005 |
| month | ui_elec | 0.336±0.067 | - | - | 0.622±0.030 | - | 0.723±0.008 |
| month | ui_reg | 0.391±0.032 | 0.484±0.040 | 0.542±0.032 | 0.632±0.022 | 0.683±0.009 | 0.720±0.008 |
| month | p0_ui_reg | 0.487±0.021 | 0.543±0.021 | 0.612±0.018 | 0.653±0.013 | 0.709±0.003 | 0.731±0.005 |
| year | p0 | 0.198±0.016 | 0.250±0.018 | 0.263±0.017 | 0.256±0.016 | 0.286±0.008 | 0.315±0.030 |
| year | ui_elec | 0.125±0.055 | - | - | 0.335±0.020 | - | 0.418±0.015 |
| year | ui_reg | 0.191±0.017 | 0.217±0.035 | 0.267±0.021 | 0.290±0.038 | 0.367±0.006 | 0.478±0.073 |
| year | p0_ui_reg | 0.228±0.015 | 0.265±0.017 | 0.321±0.018 | 0.309±0.017 | 0.393±0.007 | 0.488±0.049 |

配对差（a − b，同 target/B/draw，P0 对 seed 取平均；n = days 16、month/year 8 个配对点，CI 见 `a2_paired.csv`）：

| a − b | stratum | B=1 | B=2 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---|---|---|---|---|
| ui_reg − P-prior | days | +0.045 | +0.095 | +0.167 | +0.247 | +0.298 | +0.343 |
| ui_reg − P-prior | month | +0.076 | +0.136 | +0.223 | +0.300 | +0.354 | +0.387 |
| ui_reg − P-prior | year | +0.046 | +0.072 | +0.131 | +0.157 | +0.234 | +0.328 |
| P0 − ui_reg | days | +0.270 | +0.270 | +0.181 | +0.116 | +0.071 | +0.034 |
| P0 − ui_reg | month | +0.075 | +0.034 | −0.007 | −0.067 | −0.124 | −0.153 |
| P0 − ui_reg | year | +0.008 | +0.033 | −0.004 | −0.035 | −0.081 | −0.163 |
| P0+ui_reg − P0 | days | +0.006 | −0.013 | −0.007 | −0.003 | −0.002 | +0.008 |
| P0+ui_reg − P0 | month | +0.021 | +0.025 | +0.077 | +0.088 | +0.151 | +0.165 |
| P0+ui_reg − P0 | year | +0.029 | +0.015 | +0.058 | +0.054 | +0.107 | +0.172 |
| P0+ui_reg − ui_reg | month | +0.096 | +0.059 | +0.069 | +0.021 | +0.027 | +0.011 |

**解读与限制。** (i) 公平化并没有推翻 P1 的结论：ui_reg 在 B≥8 与 ui_elec 差别 ≤0.03，B=33 因为选出的 lambda 近 0 而完全等价；它改善的是 B=1–2（+0.06 到 +0.09），但 B=1 的 ui_reg 仍低于 P0 0.27（days）/0.08（month），year 持平。所以“B=1 时 P 变体远高于任何梯度方法”仍然成立；“days 层 P 变体不低于梯度参照”也成立（ui_reg 的 B=33 为 0.704，P0 为 0.738）。(ii) month/year 的 B=33 amortization gap 在 ui_reg 下为 0.153/0.163，与 P1 对 ui_elec 报告的 0.155/0.138 同量级。(iii) 新发现：**从 P0 的表出发的正则化细化在 month/year 显著提升（B=8 +0.09/+0.05，B=33 +0.17），在 days 层无效**；P1 审查（§0.1 第 7 点）只在 days 层试过 P0 起点，结论“从 P0 的表出发再做梯度下降反而下降”因此只对 days 成立。(iv) 限制：source session 对 trunk 是 in-sample（month 的 source 模拟 R² 0.62–0.76，target 的 ui_reg 只有 0.39–0.72），所以选出的 lambda 对 target 可能偏小；P0+ui_reg 没有为 P0 起点单独选参；year 复用 month 的选择；ui_reg 只有 1 个 seed。这些是参照行，不是候选。

## 4. A′3 结果

### 4.1 主要数字（P0 与 P4(m=8)；3 个 stratum、2 个 seed 合并；ΔR² = 量化 − FP，RMS = 预测 RMS 变化相对 FP 预测标准差）

FP R²：P0 val 0.753/0.805/0.812，target 0.490/0.567/0.589（B=1/8/33）；P4 val 0.757/0.814/0.823，target 0.498/0.560/0.582。

pooled 各配置（完整表含 P0/P4 与 val/target 的 CI：`a3_cells.csv`、`tables_a1_a2_a3.md`，分 stratum：`a3_cells_by_stratum.csv`）：

| 配置 | P0 val ΔR² B=1/8/33 | P0 target ΔR² B=1/8/33 | P0 target RMS B=1/8/33 | P4 target ΔR² B=1/8/33 | P4 target RMS B=1/8/33 |
|---|---|---|---|---|---|
| w8 | −0.0000/−0.0000/−0.0001 | −0.0001/+0.0000/+0.0001 | 0.7/0.6/0.6% | −0.0001/−0.0001/−0.0001 | 0.6/0.6/0.5% |
| w4 | −0.0020/+0.0001/−0.0000 | −0.0018/−0.0008/−0.0006 | 7.3/5.7/5.9% | +0.0003/−0.0004/−0.0021 | 5.2/4.0/4.3% |
| tern（PTQ） | −0.0421/−0.0452/−0.0464 | −0.0464/−0.0524/−0.0601 | 34.9/31.1/32.0% | −0.0278/−0.0358/−0.0402 | 30.9/29.3/29.3% |
| acc16 | −0.0000/+0.0000/−0.0000 | +0.0001/+0.0000/−0.0001 | 0.9/0.5/0.6% | −0.0006/−0.0000/−0.0002 | 2.6/0.6/0.6% |
| acc12 | −0.0628/−0.0014/−0.0012 | −0.1396/−0.0027/+0.0001 | 47.3/8.1/6.2% | −0.1207/−0.0003/+0.0014 | 42.6/6.5/5.0% |
| acc12_bs（B 相关 exponent） | −0.0002/−0.0014/−0.0012 | −0.0007/−0.0027/+0.0001 | 2.6/8.1/6.2% | −0.0014/−0.0003/+0.0014 | 4.2/6.5/5.0% |
| acc8 | −0.614/−0.556/−0.492 | −0.812/−0.707/−0.367 | 119/99/85% | −0.829/−0.618/−0.322 | 121/92/79% |
| acc8_bs | −0.145/−0.556/−0.492 | −0.465/−0.707/−0.367 | 85/99/85% | −0.460/−0.618/−0.322 | 80/92/79% |
| desc16 | +0.0000/+0.0000/−0.0000 | +0.0000/−0.0000/−0.0000 | 0.5/0.5/0.5% | −0.0000/+0.0000/−0.0000 | 0.5/0.5/0.5% |
| desc12 | +0.0000/−0.0000/−0.0000 | −0.0000/−0.0000/−0.0000 | 0.5/0.5/0.5% | −0.0000/−0.0000/−0.0000 | 0.5/0.5/0.5% |
| desc8 | +0.0001/+0.0001/+0.0000 | +0.0001/+0.0003/+0.0002 | 1.0/0.9/0.9% | +0.0002/+0.0002/+0.0004 | 0.9/0.8/0.9% |
| tab8 | −0.0000/+0.0000/−0.0000 | −0.0001/−0.0001/+0.0001 | 0.8/0.7/0.7% | +0.0000/−0.0001/−0.0001 | 0.8/0.7/0.7% |
| tab4 | −0.0011/−0.0020/−0.0045 | −0.0003/−0.0029/+0.0001 | 9.2/8.0/7.6% | −0.0039/−0.0030/−0.0039 | 8.9/8.6/8.3% |
| lut（exact 倒数） | - | - | - | −0.0000/+0.0000/−0.0000 | 0.5/0.4/0.4% |
| lut_rlut（LUT 倒数） | - | - | - | −0.0000/+0.0000/−0.0000 | 0.5/0.5/0.5% |
| w8+acc16+desc16+tab8（+LUT） | −0.0002/−0.0001/−0.0001 | −0.0000/−0.0000/+0.0001 | 1.1/0.7/0.8% | −0.0006/−0.0001/−0.0004 | 2.8/0.8/0.8% |
| w4+acc12+desc12+tab8（+LUT） | −0.0395/−0.0013/−0.0013 | −0.0738/−0.0039/−0.0010 | 40/10/9% | −0.0471/−0.0007/−0.0008 | 34/8/7% |
| tern+acc16+desc16+tab4（+LUT） | −0.0420/−0.0468/−0.0465 | −0.0477/−0.0553/−0.0631 | 36/32/33% | −0.0306/−0.0376/−0.0410 | 32/30/30% |

accumulator 的分组消融（12 / 8 bit，P0 target ΔR²，B=1/8/33）：norm（n,sx,gram）−0.006/−0.003/−0.0015 与 −0.80/−0.57/−0.40；ac −0.003/+0.001/+0.002 与 −0.07/−0.07/−0.06；cross −0.0001/0/0 与 −0.64/−0.30/−0.09；dir −0.129/−0.0005/−0.0002 与 −0.79/−0.88/−0.55；quad（P0 不读 r，无影响；P4：−0.0015/+0.0001/0 与 −0.001/0/0）。

### 4.2 ternary 的 QAT 对照与数值噪声底

| 变体 | 评测集 | B | FP（原检查点） | ternary PTQ | ternary QAT | ΔR² PTQ | ΔR² QAT (CI) |
|---|---|---|---|---|---|---|---|
| P0 | target | 1/8/33 | 0.490/0.567/0.589 | 0.444/0.515/0.529 | 0.483/0.556/0.577 | −0.046/−0.052/−0.060 | −0.007±0.008/−0.011±0.006/−0.012±0.006 |
| P0 | val | 1/8/33 | 0.753/0.805/0.812 | 0.710/0.760/0.766 | 0.746/0.804/0.810 | −0.042/−0.045/−0.046 | −0.006±0.005/−0.001±0.003/−0.003±0.001 |
| P4 | target | 1/8/33 | 0.498/0.560/0.582 | 0.470/0.524/0.542 | 0.498/0.561/0.577 | −0.028/−0.036/−0.040 | −0.000±0.008/+0.001±0.006/−0.005±0.007 |
| P4 | val | 1/8/33 | 0.757/0.814/0.823 | 0.717/0.761/0.764 | 0.751/0.809/0.813 | −0.040/−0.054/−0.059 | −0.006±0.005/−0.006±0.004/−0.010±0.005 |

QAT 的 source val 与原检查点相近（days 0.68–0.70，month/year 0.83–0.84，见 `a3_qat_ternary*.csv`）。

**数值噪声底**：trunk 的输出对 table 的极小扰动已有 0.4–0.5% 的预测 RMS 变化（我们实测：对 P0 days target 的表加幅度为表平均绝对值 1e-6 倍的随机扰动，RMS 变化 0.0167，约为预测标准差 4.1 的 0.4%；扰动 1e-3 时 0.025，饱和）。因此 `rel_rms` ≤0.5% 的行（desc16/12、acc16、lut、w8、tab8）是在噪声底，不应当读成量化的影响；同一个 FP 子集重复运行是逐位确定的（RMS 0）。

**非有限预测**：acc8 的预测出现 NaN/Inf（`finite` 标志；P0 target B=1：64 行中 48 行；B=8 与 B=33 分别为 30、44 行，见 `a3_cells.csv` 的 `nonfinite` 列），ΔR² 的均值只在有限行上计算，因此 acc8 的 ΔR² 被低估（真实更差）。原因在于 σ² = Σx²/n − μ² 的相消：8 bit 的 Gram/sx 使方差为负或 0。

### 4.3 解读

- **RRAM（固定权重 calibrator）**：8 bit 与 4 bit 权重都不需要重训（4 bit 的 RMS 扰动 2–7%，ΔR² ≤0.002）。**ternary 不能 PTQ**（−0.03 到 −0.06），但 STE 微调很便宜（约 3–5 分钟/run，source-only）可回收大部分。2T2R 的 ternary 映射因此可行，前提是训练时加 QAT。
- **累加器宽度是校准通路里最敏感的部分**：16 bit 无损，12 bit 需要随 B 变化的 block exponent（或者小 B 时不使用高位），8 bit 不可用。敏感度排序（12 bit、B=1）：dir/sphi > norm(Gram, sx) > ac > cross > quad。最终描述量对位宽几乎不敏感（8 bit 就够），说明可以：累加器 16 bit，finalize 后截到 8–12 bit 送入 calibrator。
- **输出表 int8 无损、int4 掉 0–0.005**；**softmax LUT 无损**。
- target 与 source val：二者的 ΔR² 排序相同，量级相同；val 上的 ΔR² 通常略小（val 的 FP R² 更高，约 0.75–0.82 对 0.49–0.59），所以 target 上的 ΔR² 部分包含了噪声交叉项的放大；`rel_rms` 在两者上是同一量级（例如 ternary：val 22–25%，target 29–35%），是更干净的比较量。注意 val session 是 trunk 的训练 session，且参与过 P1 的早停选择，所以 val 数字偏乐观，只看相对变化。

## 5. A′4：架构级代价（解析；公式来自代码中的实际 tensor 形状，测试中与真实 PopStats tensor 形状和 forward hook 的 MAC 计数逐项核对）

记号：N units，K 个潜变量，D 个行为维度，L 个 bin/trial，B 个标注 trial，H=64，m=8（种子数），dctx=32，Fd = 17+2K+5D（描述量宽度；K=10, D=2 时 47），MAC = 一次乘加。

**累加（流式，每个有效 bin）**：MAC = N²（Gram）+ 3N（3 个 lag 的自相关）+ 5DN（5 个 lag 的行为互协方差）+ 3N（T4 的 φxᵀ）+ 9（φφᵀ）= N² + (6+5D)N + 9；加法 = 8N + 5D + 13。每个 trial 结束时 q_* 一致性累加：N + 2DN + D MAC（及同样数量的除法）。一次校准的累加 = B·(L·[N²+(6+5D)N+9] + N+2DN+D)。
**finalize（一次）**：(2K+1)N² + (K+27)N + 27 MAC（latent 相关两次 N²K、Gram 归一化 N²、N 个 3×3 求解）+ O(N·Fd) 逐元素运算。
**calibrator 前向**：N × (每 unit 的所有 Linear 层 in×out 之和) + pooling。P0：Fd·H + H² + H·K；P4：phi (Fd·H+H²+H²) + val (H·dctx) + rel (6·16+16) + gen ((Fd+dctx+1)·H+H²+H·K) + 打分与路由 2·H·m + μᵢ（m·dctx）；pooling = N·m·dctx（P2：N·dctx）。P4 若第二遍重算 φ（不存 N×H 的激活），φ 的 MAC 要再加一遍。
**写表**：K·N 个字。
**推理 read-in MVM**：每 bin K·N MAC（外加 2N 逐元素的标准化）。
**SRAM（元素数 × 位宽）**：accumulators 共 N² + (22+7D)N + 29 + 6D 项（Gram N²，syx 5DN，sdir 8N，ac 9N，…），描述量 N·(Fd+6)，表 K·N（双缓冲 ×2），归一化参数 2N，P4 的 online-softmax 状态 m·(dctx+2)。
**anytime 单 trial 更新**：流式部分 L·[N²+(6+5D)N+9] + (N+2DN+D)；若每个 trial 后刷新表，再加 finalize + 前向 + 写表。

数值（B=8；`cost_model.csv` 含 B=1/8/33、P0/P2/P4、fp32 与 16 bit/int8 两种位宽）：

| 量 | 000688（N=26, K=10, D=2, L≈220） P0 | P4 | HD-sEMG（N=256, K=16, D=5, L=1250） P0 | P4 |
|---|---|---|---|---|
| 累加 MAC / 有效 bin | 1101（Gram 676；对称 Gram 776） | 同左 | 73,481（Gram 65,536；对称 40,841） | 同左 |
| 累加加法 / bin | 231 | 同左 | 2,086 | 同左 |
| 累加 MAC / trial | 242,352 | 同左 | 91.85 M | 同左 |
| finalize MAC | 15,185 | 同左 | 2.17 M | 同左 |
| calibrator 前向 MAC（每 unit） | 201,344（7,744） | 643,552（24,496） | 2.52 M（9,856） | 7.32 M（28,336） |
| 其中 pooling MAC | 0 | 6,656 | 0 | 65,536 |
| 写表 | 260 字 | 260 | 4,096 | 4,096 |
| 一次校准总 MAC（B=8） | 2.16 M | 2.60 M | 739.5 M | 744.3 M |
| 其中固定权重（RRAM 可映射）比例 | 9.3% | 24.8% | 0.34% | 0.98% |
| anytime：每 trial 流式 MAC | 242 k | 同左 | 91.9 M | 同左 |
| anytime：每次刷新 MAC（finalize+前向+写） | 216 k | 659 k | 4.70 M | 9.49 M |
| anytime：每 trial 总 MAC | 0.46 M | 0.90 M | 96.6 M | 101.3 M |
| 刷新折算成推理 read-in bin 数（÷K·N） | 833 | 2,534 | 1,147 | 2,318 |
| 推理 read-in MVM / bin | 260 MAC（+52 逐元素） | 同左 | 4,096 MAC（+512） | 同左 |
| calibrator 参数量 | 7,882 | 24,666 | 10,000 | 28,539 |

按 B：P4 总 MAC 随 B 线性增长（000688：B=1/8/33 为 0.90/2.60/8.66 M；HD：101/744/3041 M），刷新部分与 B 无关。

SRAM 字节（P4）：

| 量 | 000688 fp32 | 000688 16 bit 累加/描述，int8 表 | HD-sEMG fp32 | HD-sEMG 16 bit / int8 |
|---|---|---|---|---|
| 累加器（总计） | 6,612 | 3,306 | 320,748 | 160,374 |
| 其中 Gram | 2,704 | 1,352 | 262,144 | 131,072 |
| 描述量缓冲 | 5,512 | 2,756 | 81,920 | 40,960 |
| 表（单缓冲 / 双缓冲） | 1,040 / 2,080 | 260 / 520 | 16,384 / 32,768 | 4,096 / 8,192 |
| pooling 状态 | 1,088 | 1,088 | 1,088 | 1,088 |
| 归一化参数 | 104 | 104 | 1,024 | 1,024 |
| 总计（含双缓冲表） | 约 15.4 KB | 约 7.8 KB | 约 437 KB | 约 212 KB |

**要点。** (i) 000688 规模：整个校准是 MFLOP 量级，SRAM <10 KB，表仅 260 B，符合 ≤32 KB 的约束；P4 的 pooling 与 P0 相比约 3 倍 calibrator MAC，对应 P1/P2 没有稳定收益（A′1 只有弱正向），代价不划算。(ii) HD-sEMG 规模：累加是主要代价（每 trial 92 M MAC，约 3.7 MMAC/s，是推理 read-in 4,096 MAC/bin 的 18 倍），固定权重的 calibrator 只占 0.3–1%，所以 RRAM 的受益点在推理 read-in 和 calibrator 前向，而不是校准的主要算术；**Gram 累加器（131 KB@16 bit）远大于表预算 32 KB**。可选缓解（解析，**未实现、未评测**，会改变特征）：用 latent 累加 2KN = 8,192 MAC/bin 替换 N² Gram 项（总计 16.1 k MAC/bin，约 4.6 倍减少）；利用 Gram 对称（每 bin MAC −44%，Gram 存储 −50%）；累加器只需 16 bit（§4）；存储可与推理阶段的缓冲复用（校准期间推理不运行）。(iii) 这些是解析计数，不是硬件测量；000688 的 L 取两个 session（0616、0626）可用 trial 的平均长度 220 bin，HD 取 25 s×50 Hz。

## 6. 局限与未做

- A′1：只有 2 个 seed、4 个 target（days 2、month 1、year 1，month-O 与 year-O 是同一训练结果），pooling 的正向趋势在噪声底量级，未做更多 seed 的重复；没有做 identity-free + 无标签（nolab）的组合，因此无法区分“B≥8 时 days 的 identity-free 性能来自标签调谐统计还是无标签统计”；只试了 m=8（未重选 m）；identity-free 训练的 best step 更晚，预算翻倍规则触发多次，但最终都收敛。
- A′2：lambda 的 B=1 最优在网格边缘；选择依赖 trunk 的 in-sample source session；year 复用 month 的选择；P0+ui_reg 没有单独选参；ui_reg 与 ui_elec 的配对只有 5 个 draw；没有 ui_prior 的正则版本。
- A′3：激活（ADC/DAC 精度）、偏置、`mu0`、输入标准化常数没有量化；softmax 用批量形式，没有评测 online-softmax 形式；accumulator 的步长是全局每 key 一个，不是每元素；8 bit 累加器没有尝试 per-unit scale 或浮点累加（可能挽回）；没有噪声/器件非理想性（RRAM 电导漂移）建模；val session 是 trunk 的训练 session，且参与了 P1 的早停选择；ternary 的 QAT 只对 P0 与 P4 做，没有 int4 QAT。
- A′4：解析计数，非硬件测量；latent 累加替代 Gram 的方案没有实现或评测；HD-sEMG 的描述量沿用 000688 的特征族（包括方向调谐特征），D=5 取 Hyser 的 5 指力，实际 sEMG 特征可能不同；没有计 trunk 本身的代价。
- 没有推送，没有改动 P1 的 `results/` 内容，也没有触碰 DB6 / Hyser / sEMG 相关文件。

## 7. 文件与复现

- 本目录：`a1_*.csv`（cells、paired、gap、gain、floor、training、flags）、`a2_*`（cells、paired、selection、select_grid）、`a3_*`（cells、by_stratum、qat）、`cost_model.csv`、`cost_trial_stats.json`、`tables_a1_a2_a3.md`、`a*_rows.csv.gz`。
- 原始：`results/population_calib_p2_20261015/{runs,reference,uireg,quant,logs}/`（gitignored）。
- 复现（GPU0，全部幂等可续跑）：`python scripts/run_pc2_queue.py --stage noalign --workers 3`（A′1）；`python scripts/run_pc2_uireg.py select --stratum days|month`，再 `eval --stratum days|month|year [--sel-from month]`（A′2）；`python scripts/run_pc2_quant.py --stratum days|month|year`（A′3），`python scripts/run_pc2_queue.py --stage qat --workers 1` 后 `run_pc2_quant.py --stratum S --qat`（QAT）；`python scripts/pc2_cost_report.py`（A′4）；`python scripts/analyze_pc2.py`（汇总）。
