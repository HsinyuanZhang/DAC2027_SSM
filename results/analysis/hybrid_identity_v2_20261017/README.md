# Hybrid identity v2：切断 descriptor 层耦合后，pooling / 逐对 cross-attention 能否补回 uncovered unit 的信息（DANDI 000688 sub-M，2026-10-17）

执行者：hybrid-identity 工作线（本地 `cuda:0`，≤3 个并行 job）。前置：v1（`../hybrid_identity_20261016/README.md`，顶部已加 “审查后修正”）、drift-curve、P1。v2 的目的是修正 v1 的三个缺陷：(a) 对照仍带 descriptor 层群体耦合；(b) PMA 退化成 mean pooling；(c) uncovered 比例只覆盖到 0.5。

代码：`ssm_decode/hybrid_calib.py`（`cut_coupling`、`pma_fix`、`CrossAttention`、`pair_corr`、collapse 监测）、`scripts/run_hybrid_identity_v2.py`（`train / valsrc / eval / oracle / check`）、`scripts/analyze_hybrid_identity_v2.py`、`scripts/xatt_cost_table.py`、`scripts/quantify_descriptor_coupling.py`、`scripts/v1_review_*.py`、`tests/test_hybrid_calib.py`（52 项；连同 `tests/test_drift_curve_guard.py` 共 64 项通过）。原始 rows / checkpoint / 日志在 `results/hybrid_identity_v2_20261017/`（gitignored）。

## 0. 结论

1. **XATT 不优于它的 no-pairwise 对照，在任何 uncovered 比例上都一样**。hx − hx_np（B≥2，12 target，across-target t-CI）：natural total +0.001 [+0.001, +0.002]（11/12，量级 0.001，没有实际意义），`unc` +0.000，in-context attr −0.000；stress 0.5 / 0.7 / 0.9：total +0.001 / −0.000 / +0.000，CI 都含 0。把 pair 输入做 row-standardise（hxs）以后：hxs − hxs_np 在 natural 为 −0.002 [−0.008, +0.003]，在 stress 0.5 / 0.7 / 0.9 为 −0.009 / −0.010 / −0.014（后两者 CI 不含 0，**pair 项反而有害**）。XATT 相对参数量对齐的 wide 对照（h1w137）也是负的（natural total −0.021 [−0.040, −0.003]）。在 coupled 条件下同样（hx − hx_np：total +0.001 / attr +0.000）。原因的直接证据：真实 target 上 uncovered–covered 的 count correlation 很小（平均 |c| ≈ 0.04），训练后 pair 项对 attention logit 的跨 key 标准差只有 q·k 项的 1/6–1/10（0.07 vs 0.45–0.68，在三个 target 上检查，没有看 R2），XATT 的 attention 主要由 q·k（per-unit descriptor 相似度）决定，两个模型几乎一样（source val 差 <0.001）。没有加入 lagged cross-covariance 的 pair 特征（PopStats 不累加 N×N 的 lag Gram），见 §6。
2. **修好的 PMA（pma_fix）不再退化，但没有超过 per-unit 对照**。β entropy 0.89–0.94（v1 init 重训后仍为 0.996–0.999），pooling 权重 entropy 0.73–0.95，max pool weight × N = 3–13，不同 unit 的 context 标准差 0.03–0.09（v1 / 退化版本 0.001）。h2pf − h2pf_loc（B≥2）：natural total −0.003 [−0.010, +0.003]，`unc` +0.004 [−0.000, +0.009]，attr +0.002 [−0.002, +0.007]；stress 0.5 / 0.7 / 0.9 的 total −0.004 / −0.005 / +0.006，CI 都含 0。h2pf − wide153 在 natural 为 total +0.015 [+0.003, +0.028]，但 `unc` / attr 为 −0.001 / 0.000，即来自 hybrid 结构里的 λ/μ0 shrinkage 等，不是 attention（v1 审查 (e)(3) 的不对称：wide 没有 shrinkage 分支）。**唯一稳定的小正信号来自没有修复的、退化为 uniform mean pooling 的 PMA（h2p）**：h2p − h2pf_loc 的 attr +0.011 [+0.006, +0.017]（10/12，三个 seed 都为正）、h2p − h1（flag-only）attr +0.008 [+0.004, +0.012]（12/12），在 stress 0.9 上 h2p − wide153 total +0.020 [+0.011, +0.030]；量级约 +0.01 R2，且 `unc`（swap 到 P0 covered 列背景下）并不显著，所以应当读作 “群体 mean context 对 uncovered 列的 in-context 补充有 ≈ +0.01”，不是 attention 的效果（锐化的 attention h2pf − h2p = −0.009 [−0.014, −0.004]）。
3. **cut 与 coupled 的差很小但稳定**：切断 descriptor 层耦合使 flag-only 的 attr 下降 0.007 [0.004, 0.010]（12/12），per-unit 对照（h2pf_loc）下降 0.015 [0.007, 0.023]，total 下降 0.005–0.011（CI 常含 0）；也就是 v1 对照里的 descriptor 耦合贡献约 +0.005 到 +0.015 R2。XATT / PMA 相对各自对照的结论在 cut 与 coupled 下一致。
4. **headroom 很大，候选捕获得很少**（diagnostic oracle：uncovered 列用 target 全部 trial 做 gradient 拟合，trunk 冻结，H4 违反）。B=8 时 oracle 相对 P0 的 uncovered 列增益为 +0.19（natural）/ +0.39 / +0.51 / +0.59（uncovered 0.5 / 0.7 / 0.9）；各候选（相对 P0）只拿到 +0.01–0.03 / +0.05–0.07 / +0.06–0.08 / +0.06–0.12，即捕获 oracle 增益的 7–17% / 12–17% / 12–16% / 11–20%；其中几乎全部来自 masked 训练 + flag（h0m / h1 已有 7–17%），XATT 与 PMA 在此之外的增量为 0（hx − hx_np、h2pf − h2pf_loc 同 §0.1–0.2）。用 **B 个 trial 做 gradient 拟合** uncovered 列（反传穿过 trunk，不是候选，只是参照）在 B=8 达到 R2 0.29（natural）/ 0.33 / 0.36 / 0.38，B=33 达到 0.35 / 0.43 / 0.47 / 0.51，而 calibrator 只有 0.2 / 0.2 / 0.17 / 0.16；所以 headroom 在 “B 个 trial 的标签” 里是可达的，瓶颈是 calibrator 把统计量映射成列的能力（amortization gap），不是信息不在 B 个 trial 里。
5. **高 uncovered 比例下（stress）masked 训练仍是主角**：P0 在 0.5 / 0.7 / 0.9 为 0.131 / 0.084 / 0.043（B≥2 … 见 §2），masked 训练的 flag-only 为 0.194 / 0.158 / 0.140；masked 训练相对 P0 的 total 为 +0.068 / +0.074 / +0.097（11–12/12 个 target）。XATT / fixed PMA 在 0.9 上相对 flag-only 为 −0.017 [−0.057, +0.023] / +0.007 [−0.007, +0.021]。
6. **一句话**：在 cut-coupling 条件下，非退化的逐对 XATT 和非退化的 PMA 都没有超过各自的容量对照；剩余的小信号（≈ +0.01）只出现在退化为 mean pooling 的 PMA 上，且与 descriptor 耦合（≈ +0.005–0.015）同量级；相对 oracle headroom（+0.2 到 +0.6）候选捕获 ≤ 20%，其中 pooling 的净贡献 ≈ 0。

## 1. 设计

- **trunk、targets、draws、B**：与 v1 / drift-curve 完全相同（冻结 month AP trunk；12 个 target，B∈{1,2,4,8,16,33}，8 个 draw，整段连续因果解码）。P0 为复用的 P1 pool-O（seed 0/1，只有两个 seed；P0 参照值取两个 seed 的均值，swap 指标以 seed 0 的 table 为底）。`eval` 里 assert flag == prior 零列；trunk hash 在进程结束前核对；target 端 autograd 调用数 0（50,400 行）；`h4_query_disjoint` 全为真。
- **Cut coupling**：`cut_coupling=True` 在 `forward` 里把 uncovered unit 的 latent-correlation 描述量（`corr_columns`）置 0（prior 列本来就为 0）；normalizer 的 corr 列只用 covered 行拟合。条件 `cut`（主）与 `coupled`（v1 式 descriptor，作对比）。
- **训练**：episodic，source session，留一 prior，模拟 uncovered 比例 f ~ U(0, **0.9**)（v1 为 0.5），同 v1 协议（lr 1e-3、batch 8、window 256、每 100 步在 source val 上选 checkpoint、patience 8、预算翻倍）。**3 个 seed（0/1/2）**，共 16 个 variant×条件、48 个 calibrator，全部收敛（`train_summary.csv`；预算 1000–8000）。selection 只用 source val（20140306；natural + 模拟 0.3 / 0.6 / 0.9 uncovered × B 网格 × 2 draw）。
- **Variants**（`cut` 条件全有；`coupled` 条件只有 h1、hx、hx_np、h2pf、h2pf_loc）：

| 名称 | 结构 | 参数量 |
|---|---|---:|
| h0m / h1 | P0 + masked 训练（无 / 有 covered flag），无 shrinkage 分支 | 7,882 / 7,946 |
| h1w137 / h1w153 | wide + flag，宽度由 `match_width` 对齐 XATT / PMA | 26,999 / 32,599 |
| h2p | v1 的 hyb PMA（u、a 初始化 0.1），重训作 “未修复” 对照 | 32,740 |
| h2pf | hyb PMA + fix：u、a 初始化 1.0、可学习的 score / routing 温度（初始 2）、u / a / 温度不做 weight decay | 32,742 |
| h2pf_loc | h2pf 的 per-unit 对照（同参数，ctx 换成 val(φ(d~_i))） | 32,742 |
| hx | XATT：uncovered unit 为 query，covered unit 为 key / value；score = q·k/√dk + g_h([c_ij, c_ij√(n/100)])，c_ij 取自流式 Gram 的 count correlation；value = [covered 的 prior 列, descriptor]；softmax over covered unit；输出 = g_unc([d~, flag, ctx, attention 加权的 covered prior 列]) + γ·（attention 加权的 prior 列）；covered unit 走 g_cov | 27,033 |
| hx_np | 容量对照：同网络、同参数量，pair 输入换成 key 的 per-unit 标量（covered unit 与其他 covered unit 的平均 correlation），不含任何 (i, j) pair 项 | 27,033 |
| hxs / hxs_np | hx / hx_np 的 pair 输入做 row-standardise（跨 covered key）。**这是在看到 hx 的 pair 项很小（上述 attention logit 统计，没有看 R2）之后加的第二个设计**，两个版本都报告 | 27,033 |

  P0 与 h0m 的区别是 masked 训练；h1 与 h0m 的区别是 flag；hx / hx_np / h2pf / h2pf_loc 与 flat 的区别是 hybrid 结构 + reliability shrinkage（λ/μ0 只在 pooled / xatt 种类里，wide 与 flag-only 没有，见 v1 审查 (e)(3)）。
- **Stress**：uncovered 比例 f ∈ {0.5, 0.7, 0.9}，在 eval 时按 electrode 整组 mask（`simulate_uncovered`，种子 = crc32("stress-target-f")，对 variant / B / draw / seed 都固定；实际比例 0.50–0.59 / 0.71–0.76 / 0.88–0.96，f=0.9 时 target 只剩 1–2 个 covered unit，`check.json`）。B ∈ {2, 8, 33}，4 个 draw。natural 为 8 个 draw 的完整网格。
- **指标**（同 v1，增加两项）：`total` = R2(v) − R2(P0 的均值)；`unc` = R2(covered=P0, uncovered=v) − R2(P0)；`cov`；**`attr` = R2(v) − R2(covered=v, uncovered=P0)**（in-context 的 uncovered 归因）；`value` = R2(v) − R2(v 的 uncovered 列置零)。
- **统计**：target 为单位（12 个共用一个 trunk），先对 draw、seed 取均值，across-target t-CI（df=11）；`seed_min/seed_max` 是 3 个 seed 各自 across-target 均值的范围（P0 对比时 P0 取两 seed 均值）。CI 不含 trunk 与 calibrator 训练的变异，seed 谱就是这部分的粗略估计；本文称 “显著” 只指 CI 不含 0，量级 <0.005 的 “显著” 不当作效应。
- **Oracle（diagnostic only，H4 违反）**：对每个 (target, f, B, draw)，covered 列固定为 P0 table，uncovered 列从 P0 的取值起步，在 target 全部有标签 trial 上用 Adam（500 步，lr 0.02，trunk 冻结）拟合；B ∈ {2, 8, 33}、2 个 draw、f ∈ {natural, 0.5, 0.7, 0.9}。另有 **in-sample 偏差检查**（B=8：按 trial 奇偶 2-fold，在一半 trial 上拟合、另一半 query 上评分）和 **B-trial 拟合参照**（只在 B 个 calibration trial 上拟合，反传穿过 trunk，不是候选）。

## 2. 结果：绝对 R2（12 target 平均，`cells_r2.csv`）

| 变体 | nat B=1 | nat B≥2 | nat B=8 | uncov 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| P0（复用，无 masked 训练） | 0.153 | 0.206 | 0.212 | 0.131 | 0.084 | 0.043 |
| cut/h0m | 0.154 | 0.237 | 0.243 | 0.199 | 0.158 | 0.140 |
| cut/h1（flag） | 0.137 | 0.226 | 0.234 | 0.194 | 0.158 | 0.140 |
| cut/h1w137（wide） | 0.148 | 0.232 | 0.240 | 0.201 | 0.166 | 0.143 |
| cut/h1w153（wide） | 0.136 | 0.210 | 0.213 | 0.178 | 0.148 | 0.132 |
| cut/hx（XATT） | 0.161 | 0.211 | 0.217 | 0.191 | 0.156 | 0.123 |
| cut/hx_np | 0.161 | 0.209 | 0.215 | 0.190 | 0.156 | 0.123 |
| cut/hxs | 0.155 | 0.206 | 0.209 | 0.179 | 0.145 | 0.115 |
| cut/hxs_np | 0.155 | 0.208 | 0.212 | 0.188 | 0.155 | 0.129 |
| cut/h2p（退化 PMA） | 0.172 | 0.234 | 0.238 | 0.198 | 0.157 | 0.152 |
| cut/h2pf（修复 PMA） | 0.159 | 0.225 | 0.229 | 0.190 | 0.149 | 0.147 |
| cut/h2pf_loc | 0.154 | 0.228 | 0.234 | 0.194 | 0.155 | 0.141 |
| coupled/h1 | 0.144 | 0.231 | 0.239 | 0.196 | 0.160 | 0.144 |
| coupled/hx / hx_np | 0.173 / 0.172 | 0.220 / 0.221 | 0.223 / 0.223 | 0.194 / 0.194 | 0.157 / 0.155 | 0.128 / 0.129 |
| coupled/h2pf / h2pf_loc | 0.164 / 0.168 | 0.232 / 0.234 | 0.236 / 0.241 | 0.207 / 0.209 | 0.163 / 0.166 | 0.157 / 0.161 |

（stress 列是 B∈{2, 8, 33} 的平均。）图：`r2_vs_uncovered_fraction.png`（B=2 / 8 / 33）。随着 uncovered 比例上升，所有 masked 训练的变体下降的幅度比 P0 小得多（0.9 时 0.12–0.15 vs 0.043），曲线之间的差异远小于它们与 P0 的差异。XATT / hxs 的绝对值低于 flag-only 与 wide（natural B≥2：0.206–0.211 vs 0.226–0.237）。

## 3. 配对差（across-target t-CI；`paired_across_targets.csv` 含所有 B、frac、metric、seed 谱）

natural，B≥2，括号内是正号 target 数；s = 三个 seed 的范围：

| a − b | total | `unc` | attr |
|---|---|---|---|
| h0m − P0（masked 训练） | +0.031 [+0.009, +0.053] 8/12 (s +0.025..+0.036) | +0.011 [+0.003, +0.018] | +0.008 [−0.001, +0.018] |
| h1 − P0 | +0.020 [+0.004, +0.036] 9/12 | +0.018 [+0.010, +0.027] 11/12 | +0.011 [+0.005, +0.017] |
| h1 − h0m（flag 本身） | −0.011 [−0.021, −0.000] | +0.008 [+0.003, +0.012] | +0.002 [−0.002, +0.007] |
| **hx − hx_np（XATT vs 容量对照）** | **+0.001 [+0.001, +0.002] 11/12** | +0.000 [−0.000, +0.000] | −0.000 [−0.001, +0.000] |
| hx − h1w137（vs wide） | −0.021 [−0.040, −0.003] 2/12 | −0.007 [−0.012, −0.001] | −0.006 [−0.014, +0.002] |
| hx − h1（vs flag-only） | −0.015 [−0.039, +0.009] | +0.000 [−0.003, +0.004] | +0.002 [−0.007, +0.010] |
| hxs − hxs_np | −0.002 [−0.008, +0.003] 4/12 | −0.000 [−0.004, +0.004] | +0.000 [−0.005, +0.006] |
| hxs − h1w137 | −0.026 [−0.043, −0.010] | −0.008 [−0.012, −0.003] | −0.006 [−0.011, −0.000] |
| **h2pf − h2pf_loc（PMA 修复版 vs 对照）** | −0.003 [−0.010, +0.003] 2/12 | +0.004 [−0.000, +0.009] | +0.002 [−0.002, +0.007] |
| h2pf − h1w153（vs wide） | +0.015 [+0.003, +0.028] | −0.001 [−0.006, +0.004] | +0.000 [−0.004, +0.005] |
| h2pf − h1 | −0.001 [−0.015, +0.012] | −0.002 [−0.005, +0.001] | −0.001 [−0.004, +0.002] |
| h2p − h2pf_loc（退化 PMA vs per-unit） | +0.006 [−0.006, +0.018] | +0.007 [+0.001, +0.013] | +0.011 [+0.006, +0.017] 10/12 |
| h2p − h1 | +0.008 [−0.013, +0.029] | +0.000 [−0.004, +0.004] | +0.008 [+0.004, +0.012] 12/12 |
| h2pf − h2p（修复 vs 未修复） | −0.009 [−0.022, +0.003] | −0.002 [−0.005, +0.001] | −0.009 [−0.014, −0.004] 1/12 |
| coupled/h1 − cut/h1（descriptor 耦合的价值） | +0.005 [−0.000, +0.010] | +0.004 [+0.002, +0.007] 12/12 | +0.007 [+0.004, +0.010] 12/12 |
| coupled/h2pf_loc − cut/h2pf_loc | +0.006 [−0.002, +0.013] | +0.007 [+0.004, +0.011] | +0.015 [+0.007, +0.023] |
| coupled/hx − coupled/hx_np | −0.000 [−0.001, +0.000] | −0.000 | −0.000 |
| coupled/h2pf − coupled/h2pf_loc | −0.001 [−0.012, +0.009] | +0.002 [−0.003, +0.007] | −0.005 [−0.011, +0.002] |

stress（B∈{2, 8, 33} 平均，total；括号为 attr）：

| a − b | 0.5 | 0.7 | 0.9 |
|---|---|---|---|
| h0m − P0 | +0.068 [+0.038, +0.098] | +0.074 [+0.043, +0.105] | +0.097 [+0.051, +0.143] |
| h1 − P0 | +0.063 [+0.033, +0.093] | +0.074 [+0.042, +0.107] | +0.097 [+0.051, +0.143] |
| hx − hx_np | +0.001 [−0.000, +0.002] | −0.000 [−0.001, +0.001] | +0.000 [−0.002, +0.002] |
| hxs − hxs_np | −0.009 [−0.019, +0.002] | −0.010 [−0.020, −0.001] | −0.014 [−0.025, −0.003] |
| hx − h1w137 | −0.010 [−0.021, +0.001] | −0.010 [−0.021, +0.001] | −0.020 [−0.050, +0.010] |
| hx − h1 | −0.003 [−0.020, +0.014] | −0.002 [−0.020, +0.016] | −0.017 [−0.057, +0.023] |
| h2pf − h2pf_loc | −0.004 [−0.014, +0.006] (attr −0.002) | −0.005 [−0.016, +0.005] (−0.003) | +0.006 [−0.002, +0.014] (+0.006) |
| h2pf − h1w153 | +0.012 [−0.002, +0.025] | +0.001 [−0.015, +0.018] | +0.015 [+0.008, +0.023] (attr +0.011 [+0.003, +0.019]) |
| h2pf − h1 | −0.004 [−0.019, +0.011] | −0.009 [−0.020, +0.003] | +0.007 [−0.007, +0.021] |
| h2p − h2pf_loc | +0.004 [−0.012, +0.020] | +0.002 [−0.012, +0.017] | +0.010 [−0.006, +0.026] (attr +0.015 [−0.001, +0.032]) |
| h2p − h1w153 | +0.020 [+0.002, +0.038] | +0.009 [−0.008, +0.027] | +0.020 [+0.011, +0.030] |
| coupled/h2pf_loc − cut/h2pf_loc | +0.015 [+0.002, +0.028] (attr +0.020) | +0.011 [+0.000, +0.022] | +0.020 [+0.008, +0.032] |

图：`paired_vs_controls.png`（XATT − 控制 / wide / flag-only，PMA-fixed − 控制 / wide，三个指标，对 uncovered 比例）。B=1 的 natural 结果（`paired_across_targets.csv`）：hx − hx_np −0.000，h2pf − h2pf_loc total +0.005 [−0.000, +0.010]，h2p − h2pf_loc +0.018 [+0.005, +0.031]（attr +0.010），其余同上。

所有 pooled − 对照的差，XATT 的 +0.001 是唯一在 natural 上 CI 不含 0 的 “正” 结果，但量级是噪声底的 1/10；其余的 CI 含 0，或者符号为负。

## 4. 收缩与退化监测（β entropy 等；`train_summary.csv`、`eval_diagnostics.csv`、`collapse_monitors.png`）

| 模型（cut） | β / attention entropy | pool entropy | max pool weight × N | context std（across units） | per-unit local ctx std |
|---|---:|---:|---:|---:|---:|
| h2p（v1 初始化，重训） | 0.996–0.999 | 0.998 | 1.15–1.33 | ≈ 0.001 | 0.80–1.0 |
| h2pf（修复） | 0.888 / 0.938 / 0.905（seed 0/1/2，best step） | 0.731 / 0.943 / 0.929 | 13.5 / 4.5 / 4.4 | 0.094 / 0.068 / 0.029 | 0.47 / 0.26 / 0.40 |
| hx（XATT，attention 对 covered key） | 0.954–0.971 | - | 2.1–2.4（over covered keys） | - | - |
| hxs | 0.943–0.966 | - | 2.3–2.9 | - | - |

eval 时（12 个 target、各 B、各 f 的均值）：h2pf 的 β entropy 0.89–0.92、context std 0.03–0.04；h2p 0.998 / 0.001（再次退化）；XATT 的 attention entropy 0.90–0.94（f=0.9 时只有 1–2 个 covered key，max×N 趋于 1）。**结论：修复有效（把 context 的跨 unit 方差抬高了约 30–100 倍），退化版本在 v2 训练下复现，而 “有区分度的 pooling” 没有给出比对照更好的 R2**。训练曲线中 β entropy 在前 300 步从 0.985 降到 0.92–0.95。

## 5. Headroom 与 oracle（`headroom.csv`、`headroom_calibfit.csv`、`oracle_summary.csv`、`headroom_captured.png`）

B=8，12 target × 2 draw 的均值（P0 table 的 covered 列 + 不同来源的 uncovered 列）：

| uncovered | P0 列 | uncovered 列置零 | **oracle（全部 trial）** | oracle 的 held-out 版（2-fold，一半 trial 拟合） / 同一组 query 上的 P0 / in-sample | 只在 B=8 个 trial 上 gradient 拟合 | 最好的候选（swap_unc） |
|---|---:|---:|---:|---|---:|---:|
| natural | 0.229 | 0.201 | 0.421 | 0.402 / 0.236 / 0.432 | 0.288 | ≈ 0.26 |
| 0.5 | 0.147 | 0.091 | 0.542 | 0.514 / 0.151 / 0.556 | 0.327 | ≈ 0.21 |
| 0.7 | 0.088 | 0.034 | 0.594 | 0.562 / 0.091 / 0.610 | 0.361 | ≈ 0.17 |
| 0.9 | 0.049 | −0.004 | 0.641 | 0.605 / 0.050 / 0.657 | 0.384 | ≈ 0.17 |

- 每个候选在 B=8 下 “捕获的 oracle 增益” = (R2(P0 covered, v 的 uncovered) − R2(P0)) / (R2(oracle) − R2(P0))：natural 7%（h0m）到 17%（h1w137）；0.5：13–17%；0.7：12–16%；0.9：11–20%（hx 14%、hxs 11%、h2pf 20%、h2p 19%、h1 17%、h0m 17%）。XATT 与其容量对照的捕获比例相同（natural 12.4% vs 12.3%；0.9：14.2% vs 14.5%）。oracle 在 in-sample 上高估约 0.03–0.05（held-out 版相对同一组 query 的 P0 增益 +0.17 / +0.36 / +0.47 / +0.55，in-sample +0.20 / +0.41 / +0.52 / +0.61，所以 headroom 的量级不是过拟合造成的）。
- 参照：只用 B 个 trial gradient 拟合（不是候选）在 B=2 时（0.14–0.17）与 calibrator 相当（natural：0.17 vs 0.18–0.20）或更好（0.9：0.15 vs 约 0.06），在 B=8、33 时远超 calibrator（B=33：0.35 / 0.43 / 0.47 / 0.51）。因此 headroom 的一大部分是 “把 B 个 trial 的标签用到梯度拟合” 上，不是 “群体信息” 上；这是 amortization gap（见 P1 的 ui_prior），XATT / PMA 都没有缩小它。

## 6. 代价（analytic，`xatt_cost.csv`，N=96 个 unit，K=10，heads=4）

| uncovered 比例 | N_unc × N_cov | P0 fixed-weight MACs | PMA-hybrid MACs | XATT MACs（其中 pairwise） | exp 数 | online-softmax 状态（word） |
|---|---:|---:|---:|---:|---:|---:|
| 0.10 | 10 × 86 = 860 | 0.75 M | 2.18 M | 1.44 M（0.20 M） | 3,440 | 800 |
| 0.25 | 24 × 72 = 1,728 | 0.75 M | 2.21 M | 1.65 M（0.40 M） | 6,912 | 1,920 |
| 0.50 | 48 × 48 = 2,304 | 0.75 M | 2.26 M | 1.81 M（0.53 M） | 9,216 | 3,840 |
| 0.75 | 72 × 24 = 1,728 | 0.75 M | 2.31 M | 1.69 M（0.40 M） | 6,912 | 5,760 |
| 0.90 | 86 × 10 = 860 | 0.75 M | 2.34 M | 1.50 M（0.20 M） | 3,440 | 6,880 |

pairwise 部分每个 (i, j) 对 232 MAC（pair MLP 96 + q·k 64 + value 与 prior 加权 72）+ 4 个 exp；峰值在 uncovered ≈ 0.5。参数量：P0 7,946，XATT 27,033，PMA 32,742（int8 约 8 / 27 / 33 KB）。流式 Gram 累加（每个 bin 9,216 MAC，N=96）对所有变体相同，且是 pairwise 项的输入（c_ij 由 gram、sx、n 在线算出，不需要额外存储）。XATT 的 SRAM 额外开销是 online-softmax 状态（≤ 6.9 K word）；全部 N_unc × N_cov 的分数不需要驻留。收益为 0 而代价 > P0 的 1.9–2.4 倍，所以不值得上硬件。

## 7. source validation（`source_val.csv`）

natural / 模拟 0.3 / 0.6 / 0.9 uncovered 的 source-val R2（3 seed 均值，单 session 20140306）：P0 0.837 / 0.799 / 0.581 / 0.406；masked 训练的所有 variant 在 0.82–0.83 / 0.80–0.82 / 0.74–0.78 / 0.71–0.74（均值 0.77–0.79，其中 hx 0.783、hx_np 0.783、h2pf 0.768、h2pf_loc 0.770、h1 0.776）。pooled 与对照在 source 上的差 ≤ 0.015，分不开（与 v1 一致）；hx 与 hx_np 几乎逐位相同（0.783 / 0.783）。

## 8. 局限、没有做的

- **没有加入 lagged cross-covariance 的 pair 特征**：`PopStats` 只累加 lag-0 的 N×N Gram，lag Gram 需要额外的 N×N×lag 累加器和 bank；因此 pair 特征只有 count correlation（及 √n 缩放）。如果 uncovered–covered 之间的时间滞后结构有信息，本文没有检验。
- 只有 12 个 target、一个 trunk、一个 source val session；训练 seed 3 个（P0 只有 2 个）；hxs / hxs_np 是在看过 hx 的 pair 项很小之后加入的第二个设计（没有看 R2）。XATT 只用了 4 个 head、dk=16 的一种容量，没有做超参搜索。
- 高 uncovered 比例用的是模拟 mask（按 electrode 整组），f=0.9 时 target 只剩 1–2 个 covered unit，这与真实的 array 漂移机制不一定一致；stress 只有 B∈{2, 8, 33}、4 个 draw。
- oracle 只做了 B∈{2, 8, 33}、2 个 draw，500 步优化未必完全收敛（800 步在一个测试 case 上再提高 0.015），所以 oracle 是 headroom 的下界；covered 列固定为 P0 table，没有与其他候选的 covered 列联合。headroom 捕获比例 = 相对 P0 的比例，其中绝大部分来自 masked 训练 + flag。
- cost 是 analytic，没有硬件模拟（累加器位宽、exp LUT 等不在本轮）；没有 Overleaf exp-log 更新（留给调用者）。
- 退化的 PMA（h2p）在 cut 下仍给出 +0.01 的 attr，但没有与它自己的 per-unit 对照（h2p_loc）同 seed 配对训练（对照用 h2pf_loc，结构相同、初始化的 u、a 不同、且 u/a 在 loc 里不起作用）；这个 +0.01 的因果解释（mean pooling 本身 vs 训练轨迹）没有进一步隔离。

## 9. 文件与复现

csv：`cells_r2.csv`、`paired_across_targets.csv`、`paired_per_target`（`per_target.csv`）、`headroom.csv`、`headroom_calibfit.csv`、`oracle_summary.csv`、`oracle_rows.csv`、`oracle_calibfit_rows.csv`、`train_summary.csv`、`train_curves.csv`、`eval_diagnostics.csv`、`source_val.csv`、`descriptor_coupling.csv`、`xatt_cost.csv`、`row_counts.csv`、`audit.json`、`v1_pma_collapse_monitors.csv`（在 v1 目录）。图：`r2_vs_uncovered_fraction`、`paired_vs_controls`、`collapse_monitors`、`headroom_captured`（`.png`，前三个还有 `.pdf`）。

```
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_hyb2
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
$PY scripts/run_hybrid_identity_v2.py check
$PY scripts/run_hybrid_identity_v2.py train --conds cut --variants h0m,h1,h1w137,h1w153,hx,hx_np,hxs,hxs_np,h2p,h2pf,h2pf_loc
$PY scripts/run_hybrid_identity_v2.py train --conds coupled --variants h1,hx,hx_np,h2pf,h2pf_loc
$PY scripts/run_hybrid_identity_v2.py eval --targets <12 dates> --fracs nat --conds cut --tag cut_nat_a        # + coupled; hxs: --variants hxs,hxs_np
$PY scripts/run_hybrid_identity_v2.py eval --targets <dates> --fracs 0.5,0.7,0.9 --bs 2,8,33 --draws 4 --conds cut --tag cut_str_a
$PY scripts/run_hybrid_identity_v2.py oracle --targets <dates> --fracs nat,0.5,0.7,0.9 --bs 2,8,33 --draws 2 --steps 500 --lr 0.02 --xfit --tag a
$PY scripts/run_hybrid_identity_v2.py oracle --fit-on calib ... --tag a     # B-trial gradient-fit reference
$PY scripts/run_hybrid_identity_v2.py valsrc
$PY scripts/analyze_hybrid_identity_v2.py; $PY scripts/xatt_cost_table.py
```
