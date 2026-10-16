# Hybrid identity：aligned electrode prior（covered units）+ population pooling（uncovered units），DANDI 000688 sub-M（2026-10-16）

执行者：hybrid-identity 工作线（本地 GPU `cuda:0`；前半段与 baselines 线共享，训练较慢，baselines 结束后每个 calibrator 约 50 s）。背景：drift-curve（`../drift_curve_20261016/`，12 个 target，18–50% 的 unit 落在没有 source unit 的 electrode 上）、P1（`../population_calib_20261015/`）、P2 A′1（`../population_calib_p2_20261015/`）、sEMG 容量对照（`../semg_calib_20261016/` §审查后修正）。

代码：`ssm_decode/hybrid_calib.py`（covered flag、uncovered 模拟、`HybridCalibrator`、`match_width`）、`scripts/run_hybrid_identity.py`（`train` / `valsrc` / `eval` / `check`）、`scripts/analyze_hybrid_identity.py`、`tests/test_hybrid_calib.py`（36 项通过；连同 `tests/test_drift_curve_guard.py` 共 48 项）。原始 rows、checkpoint、日志在 `results/hybrid_identity_20261016/`（gitignored）；本目录是汇总 csv 与图。

## 0. 结论一览

问题：当一部分 target unit 所在的 electrode 没有 source unit 时，hybrid calibrator（covered unit 走 aligned-prior 通路，uncovered unit 的 table 列由 [自身 descriptor + 群体 pooling context] 生成）是否优于 P0（uncovered unit 为零 prior 列 + per-unit encoder residual）？

1. **所有 hybrid 变体的绝对收益都很小**。12 个 target 上 P0 的 query R²（B≥2 平均）为 0.206；各变体相对 P0 的 paired 增益（B≥2 平均，across-target 95% t-CI，df=11）：H1（P0+flag）+0.026 [−0.004, +0.057]；H2-DeepSets +0.019 [−0.000, +0.038]；H2-PMA +0.047 [+0.028, +0.065]（12/12 target 为正）；H3-DeepSets +0.054 [+0.035, +0.074]（12/12）；H3-PMA +0.023 [+0.011, +0.035]（11/12）。
2. **增益的大部分不能归因于 pooling**。同样用 masked-uncovered 训练、但没有任何跨 unit 信息交换的对照也有类似增益：P0-wide（+flag，参数量对齐）+0.020 到 +0.034；P2-local / PMA-local（参数量逐一致）+0.015 到 +0.045；只做 masked 训练、不加 flag 的 P0（h0m）+0.010 [−0.009, +0.029]。对照之间的离散度（0.015–0.045）与 pooling 相对对照的差同量级。
3. **uncovered unit 的列：pooling 没有超出容量对照的证据。** 把 “covered 列取自 P0、uncovered 列取自变体 v” 的 R² 减去 P0（`unc` 指标）：所有 masked 训练的变体都是 +0.013 到 +0.023（CI 均不含 0，包括没有 pooling 的 h0m、H1、wide、local 对照），pooled − local 为 −0.003 到 +0.009，pooled − wide 为 −0.005 到 +0.005。唯一 CI 不含 0 的是 H2-PMA − H2-PMA-local = +0.009 [+0.003, +0.016]（9/12 target），但 H2-PMA − wide 的 CI 含 0（+0.005 [−0.002, +0.011]），H2-DeepSets 和 H3 的两个 pooled-local 差都含 0。**所以 “增益来自 uncovered 列”几乎全部来自 masked 训练 + 显式 flag 这一对 source-only 的训练修改，而不是 pooling。**
4. **pooling 唯一站得住的信号在 covered unit，并且不稳定**：H3-DeepSets（所有 unit 都吃 pooling context）总增益 − 参数对齐 local 对照 = +0.010 [+0.003, +0.017]（8/12），− wide 对照 = +0.027 [+0.013, +0.041]（11/12），其中 covered 列贡献 +0.012 [+0.006, +0.019]；但同一设计的 PMA 版本为 −0.010 [−0.030, +0.011]，H2-DeepSets 为 +0.004 [−0.003, +0.012]，H2-PMA 为 +0.017 [+0.003, +0.031]。四个 pooled-local 差符号不一致，我们不把它当作稳定的 pooling 效应。
5. **uncovered unit 的列本身价值有限**：把 uncovered unit 的 table 列置零，P0 的 R² 只降 0.022（B≥2，[0.013, 0.032]），hybrid 变体降 0.030–0.051。在这个 trunk 上，uncovered 列能贡献的 R² 上限量级就是 0.02–0.05，所以即便生成得更好，空间也很小；20141203（50% uncovered）P0 为 0.056，各变体只到 0.09–0.13（B≥2）。
6. **与 uncovered 比例的关系**：`unc` 效应随 uncovered 比例上升（Spearman 0.43–0.75；h0m 的 ρ=0.75，p=0.005），但对照同样上升，所以这反映 “uncovered unit 越多，masked 训练 + flag 能修的越多”，不是 pooling 随比例增益；总增益与 uncovered 比例无显著关系（|ρ|<0.4，p>0.2；只有 20141203 一个 >0.30 的 target，斜率主要由它决定）。
7. **source validation 选不出 pooling**：source val（20140306；自然 LOO coverage 与人为 mask 25% / 50% 的均值）所有 masked 训练的变体都在 0.811–0.820，P0 为 0.795（P0 在 50% mask 下掉到 0.707 / 0.762，masked 训练的变体为 0.78–0.80）。因此 “masked 训练修复了 P0 在高 uncovered 比例下的退化” 在 source 上就能看到；pooled 与对照在 source 上差 ≤0.01，无法区分。

一句话：**在 DANDI 000688 / month AP trunk 上，hybrid 的 aligned + 显式 covered flag + masked-uncovered 训练给 P0 带来约 +0.01 到 +0.03 的 R²（相对 0.21）；set pooling 在此之上没有可靠的、跨 DeepSets/PMA 一致的额外增益，对 uncovered 列的增益与容量对照无法区分。**

## 1. 设计

### 1.1 covered flag 与 uncovered 模拟（全部只用 source / target 的元数据）

- `covered_flags(source_electrodes, unit_electrodes, exclude)`：unit 的 electrode 是否出现在 trunk 的 6 个 source session 的 unit→electrode 映射里。函数只拿到 electrode id，读不到神经数据、标签或 query。target 端 12 个 target 上 flag 与 P-prior 的零列逐位一致（`eval` 里 assert；`check` 子命令写 `results/hybrid_identity_20261016/coverage_check.json`）。训练时对 source session 用留一（LOO）的覆盖，与 `World.priors` 的留一 prior 完全一致。
- 训练时 uncovered 模拟：每个 episode 抽目标 uncovered 比例 `f ~ U(0, 0.5)`（范围取自 12 个 target 的观测值 0.08–0.50；这是只依赖 source / target 元数据的设计选择，没有看任何 target query；分布未做调参），在自然 LOO uncovered 的基础上，随机按 electrode 整组 mask（electrode 内所有 unit 一起），直到总 uncovered 比例 ≥ f（可能超出一个 electrode 的 unit 数）；被 mask 的 unit 的 prior 列置零（所以 descriptor 里的 prior 列和与 prior latent 的相关特征也随之改变），flag 置 0。unit dropout（原有 P0 训练里的随机 unit 子集）保留。
- 测试：`test_simulate_uncovered_*`（整 electrode、达到比例、不全部 mask）、`test_simulated_fraction_distribution_in_training_range`、`test_covered_flag_*`（只用 source 元数据；等于 electrode-aligned prior 的零列，含 LOO）、`test_calibrate_rejects_nonzero_prior_on_uncovered_units`、`test_permutation_equivariance`（9 种 kind）、`test_cross_unit_dependence_only_for_pooled_nonlocal`（local / wide 对照没有跨 unit 泄漏，pooled 有）、`test_hybrid_covered_path_has_no_pooling_context`、`test_target_calibrate_runs_without_grad_and_params_unchanged`（`autograd.grad/backward` 调用数为 0、参数不变）、`test_table_uses_only_the_supplied_trial_statistics`。

### 1.2 变体（全部 fixed-weight forward；单个 calibrator 只在 trunk 的 source session 上 episodic 训练）

descriptor = P0/P2 同款 47 维（含 aligned prior 列）+ reliability shrinkage（P1 的 λ，pooled 变体用，P0 系列不用）+ covered flag。

| 名称 | 结构 | 参数量 |
|---|---|---:|
| H0 | P0（reuse，P1 的 pool-O seed 0/1，**没有** masked 训练、没有 flag） | 7,882 |
| h0m | P0 + masked 训练（无 flag）：只检验训练增强本身 | 7,882 |
| H1 = h1 | P0 + covered flag（无 pooling） | 7,946 |
| H2d / H2p | hybrid：covered unit 走 g_cov([d~, flag])（aligned-prior residual 通路，无 pooling）；uncovered unit 走 g_unc([d~, flag, ctx])；ctx = DeepSets（mean_i ψ）/ PMA（m=8，log λ 打分，逐 unit 路由，P4 风格） | 27,492 / 32,740 |
| H3d / H3p | 所有 unit 用同一个 g([d~, flag, ctx])（pooling 也作用于 covered unit） | 19,546 / 24,794 |
| H2d_loc 等 4 个 `*_loc` | 容量对照（P2-local）：结构、参数量与对应 pooled 变体完全相同，ctx 换成逐 unit 的 ψ(d~_i)（PMA：val(φ(d~_i))，即对单个 unit 池化；u、a 保留为不起作用的参数以保持参数量一致，有效参数略少 3%）。测试验证无跨 unit 信息流 | 同 pooled |
| h1w113 / w130 / w138 / w153 | 容量对照（P0-wide + flag）：hidden 宽度由 `match_width` 与 H3d / H3p / H2d / H2p 的参数量对齐（误差 <1%，测试覆盖） | 19,559 / 24,710 / 27,334 / 32,599 |

训练协议与 P0 相同（lr 1e-3、batch 8、window 256、每 100 步在 source val 上选 checkpoint、patience 8、至少 600 步、最佳点落在最后 20% 就把预算翻倍至多 4 次），2 个 seed（0/1，对应 P0 的两个 seed）。28 个 calibrator 全部收敛（best_step 600–2500，`train_summary.csv`）。**checkpoint 选择只用 source val**：source val 是 trunk 的最后一个 source session（20140306），评分 = 自然 LOO coverage、模拟 25%、50% uncovered 三种情形 × B∈{1,2,4,8,16,33} × 2 draw 的均值（固定随机数）。注意 source val 只有一个 session，对 pooling 的分辨力很弱（见 §0 第 7 点）。

### 1.3 评测（与 drift-curve 一致）

12 个 target、B∈{1,2,4,8,16,33}、8 个 draw（`draw_trials` 与 drift-curve 完全相同）、2 个 seed，整段 session 的连续因果解码，只对 query bin 计分，物理单位 R²。trunk 冻结（hash 在每个 `eval` 进程结束前核对），target 端 `torch.autograd.grad/backward` 调用数为 0（17,280 行全部），只用 B 个有标签 trial（`h4_query_disjoint` 全为真），不使用 0623/0625/0626 与 20140307（`validate_targets`）。**H0 的 R² 与 drift-curve 的 P0 行逐位一致（max |Δ| = 0.0，0626 全部 B、draw 0、两个 seed）。**

每个 (变体, target, B, draw, seed) 记录 4 个 R²：整表、uncovered 列置零、“covered 列取自 P0 / uncovered 列取自变体”（`swap_unc`）、“covered 列取自变体 / uncovered 列取自 P0”（`swap_cov`）。所有表共享同一个 (mean, std) 归一化。指标：

- `total` = R²(v) − R²(P0)；
- `unc` = R²(swap_unc) − R²(P0)：只换 uncovered 列的效果；
- `cov` = R²(swap_cov) − R²(P0)：只换 covered 列的效果；
- 两个变体 a、b 之间的差同样按这些指标配对相减（`unc` 差 = R²(covered=P0, uncovered=a) − R²(covered=P0, uncovered=b)，即其余列固定在 P0 时的 uncovered 列差）。
- 这是整表之间的交换分解，不是可加的 Shapley 分解（列之间有相互作用，`unc + cov ≠ total`）。

统计：每个 target 先在 draw × seed 上平均；across-target CI 是 t 区间（df=11）。**12 个 target 共用一个 trunk、一组 calibrator，CI 只覆盖 target / session 的变异，不含 calibrator 训练 seed 与 trunk 的变异；训练 seed 只有 2 个**（seed0 / seed1 的 across-target 均值见 `paired_across_targets.csv`，H1 的总增益在两个 seed 上为 −0.003 / +0.056，说明训练 seed 的离散与效应量同级）。per-target CI 是对 draw 的 cluster bootstrap（两个 seed 随 draw 一起重采样）。

## 2. 结果

### 2.1 total：相对 P0 的配对差（`paired_across_targets.csv`，B≥2 = B∈{2,4,8,16,33} 的平均）

P0 的 R²（12 target 平均）：B=1/2/4/8/16/33 = 0.153 / 0.188 / 0.204 / 0.212 / 0.217 / 0.211。

| 变体 − P0 | B=1 | B=8 | B=33 | B≥2 | 正号 target（B≥2） |
|---|---|---|---|---|---:|
| h0m（masked 训练，无 flag） | +0.004 | +0.008 | +0.015 | +0.010 [−0.009, +0.029] | 9/12 |
| H1（flag） | +0.001 [−0.016, +0.018] | +0.029 [−0.004, +0.061] | +0.029 [−0.013, +0.071] | +0.026 [−0.004, +0.057] | 10/12 |
| H2-DeepSets | +0.024 [+0.007, +0.041] | +0.015 [−0.005, +0.035] | +0.017 [−0.012, +0.047] | +0.019 [−0.000, +0.038] | 9/12 |
| H2-PMA | +0.043 [+0.022, +0.064] | +0.045 [+0.027, +0.063] | +0.052 [+0.031, +0.073] | +0.047 [+0.028, +0.065] | 12/12 |
| H3-DeepSets | +0.027 [+0.010, +0.045] | +0.056 [+0.034, +0.079] | +0.066 [+0.039, +0.094] | +0.054 [+0.035, +0.074] | 12/12 |
| H3-PMA | +0.023 [+0.008, +0.039] | +0.020 [+0.006, +0.033] | +0.030 [+0.012, +0.049] | +0.023 [+0.011, +0.035] | 11/12 |
| wide 113 / 130 / 138 / 153（对照） | +0.013 / +0.014 / +0.013 / −0.003 | +0.028 / +0.034 / +0.033 / +0.018 | +0.030 / +0.036 / +0.047 / +0.034 | +0.027 / +0.034 / +0.032 / +0.020 | 9–11/12 |
| local：H2d / H2p / H3d / H3p（对照） | +0.015 / +0.023 / +0.016 / +0.013 | +0.010 / +0.027 / +0.048 / +0.033 | +0.019 / +0.038 / +0.051 / +0.040 | +0.015 / +0.030 / +0.045 / +0.032 | 8–11/12 |

### 2.2 pooling 的归因（pooled − 参数量对齐对照；B≥2；`paired_across_targets.csv` 的 `role` = `pooling vs local` / `pooling vs wide`）

| 比较 | total | `unc`（uncovered 列） | `cov`（covered 列） |
|---|---|---|---|
| H2d − H2d_loc | +0.004 [−0.003, +0.012] | −0.003 [−0.008, +0.002] | +0.007 [+0.000, +0.015] |
| H2d − wide138 | −0.013 [−0.025, −0.002] | −0.005 [−0.011, +0.000] | −0.003 |
| H2p − H2p_loc | +0.017 [+0.003, +0.031] | +0.009 [+0.003, +0.016] | −0.003 [−0.007, +0.001] |
| H2p − wide153 | +0.027 [+0.000, +0.054] | +0.005 [−0.002, +0.011] | +0.014 |
| H3d − H3d_loc | +0.010 [+0.003, +0.017] | −0.000 [−0.006, +0.005] | +0.012 [+0.006, +0.019] |
| H3d − wide113 | +0.027 [+0.013, +0.041] | +0.002 [−0.004, +0.008] | +0.024 |
| H3p − H3p_loc | −0.010 [−0.030, +0.011] | −0.001 [−0.010, +0.008] | −0.004 [−0.022, +0.015] |
| H3p − wide130 | −0.011 [−0.034, +0.012] | −0.000 [−0.009, +0.008] | −0.004 |
| H1 − h0m（flag 本身） | +0.016 [−0.001, +0.033] | +0.006 [−0.001, +0.012] | +0.012 |
| h0m − H0（masked 训练本身） | +0.010 [−0.009, +0.029] | +0.013 [+0.007, +0.020] | +0.001 |
| H3d − H2d（也让 covered unit 吃 pooling） | +0.035 [+0.023, +0.048] | +0.007 [+0.001, +0.013] | +0.026 |
| H2p − H2d（PMA vs DeepSets） | +0.027 [+0.013, +0.042] | +0.007 [+0.002, +0.011] | +0.012 |

读法：uncovered 列上 pooling 的净效应 ≤ +0.009；covered 列上 H3d 有 +0.012，但同一设计的 PMA 版本为负，H2 系列又是另一个符号模式，不一致。含 CI 的完整表（B=1…33、B≥2、all、seed 谱、对 uncovered 比例的 Spearman / 斜率）见 `paired_across_targets.csv`。

### 2.3 uncovered 列的价值（`uncovered_columns_contribution.csv`；R²(整表) − R²(uncovered 列置零)，B≥2）

P0 0.022 [0.013, 0.032]；h0m 0.027；H1 0.034；wide 0.036–0.047；H2d / H2d_loc 0.036 / 0.036；H2p / H2p_loc 0.051 / 0.030；H3d / H3d_loc 0.046 / 0.046；H3p / H3p_loc 0.034 / 0.042。生成的 uncovered 列比 P0 的列有用一些，但量级只有几个百分点，且 pooled 与对照重叠。

### 2.4 每个 target（B≥2 平均，相对 P0 的 total 增益；`paired_per_target.csv` 有 CI）

| target | uncovered 比例 | P0 R²（B=8） | h0m | H1 | H2d | H2p | H3d | H3p | wide138 | H3d_loc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20140626 | 0.08 | 0.565 | +0.008 | +0.021 | −0.016 | +0.006 | +0.028 | +0.014 | +0.025 | +0.028 |
| 20140627 | 0.08 | 0.307 | +0.041 | +0.041 | +0.015 | +0.014 | +0.050 | +0.060 | +0.043 | +0.054 |
| 20140929 | 0.14 | 0.262 | −0.016 | +0.039 | −0.004 | +0.009 | +0.075 | +0.001 | +0.044 | +0.051 |
| 20141203 | 0.50 | 0.056 | +0.030 | +0.073 | +0.042 | +0.039 | +0.071 | +0.038 | +0.048 | +0.055 |
| 20150511 | 0.21 | 0.256 | −0.009 | −0.021 | +0.003 | +0.041 | +0.021 | +0.002 | −0.003 | −0.001 |
| 20150512 | 0.20 | 0.215 | −0.062 | −0.089 | −0.039 | +0.038 | +0.015 | −0.005 | −0.047 | −0.009 |
| 20150610 | 0.22 | 0.166 | +0.026 | +0.087 | +0.052 | +0.065 | +0.088 | +0.040 | +0.081 | +0.090 |
| 20150611 | 0.25 | 0.136 | +0.005 | +0.008 | +0.020 | +0.066 | +0.037 | +0.015 | +0.026 | +0.030 |
| 20150612 | 0.27 | 0.074 | +0.007 | +0.007 | +0.014 | +0.032 | +0.018 | +0.015 | +0.010 | +0.022 |
| 20150615 | 0.29 | 0.155 | +0.014 | +0.032 | +0.029 | +0.078 | +0.057 | +0.029 | +0.037 | +0.044 |
| 20150616 | 0.18 | 0.177 | +0.043 | +0.062 | +0.055 | +0.085 | +0.102 | +0.031 | +0.062 | +0.099 |
| 20150617 | 0.18 | 0.177 | +0.038 | +0.057 | +0.057 | +0.086 | +0.092 | +0.031 | +0.062 | +0.073 |

20150512（5.8 Hz 的低发放 session）上 masked 训练的 P0 系列（h0m、H1、wide）都变差（−0.05 到−0.09），pooled 的 H2p / H3d 反而为正；这是 pooling 变体少数 “有区别” 的地方，但只涉及一个 target。

### 2.5 图

`hybrid_gain_vs_uncovered_fraction.png / .pdf`：(a) 每个 target 的 total 增益（B≥2）对 uncovered 比例，误差棒 = draw cluster-bootstrap 95% CI；(b) 只换 uncovered 列的效应；(c) H2 的 pooled − local、pooled − wide（uncovered 列）；(d) across-target 均值 ± 95% t-CI 随 B（H2d、H2p、H1、H2d-local、wide138）。没有把 target 连线（相邻 target 差异主要来自 session 本身）。

### 2.6 source validation（`source_val.csv`，两个 seed 平均）

自然 / 模拟 25% / 模拟 50% uncovered 的均值：H0 0.795（单独：0.837 / 0.814 / 0.734），h0m 0.812，H1 0.814，wide 0.811–0.814，H2d / H2d_loc 0.816 / 0.815，H2p / H2p_loc 0.819 / 0.820，H3d / H3d_loc 0.813 / 0.812，H3p / H3p_loc 0.817 / 0.811。注意 H0 没有 masked 训练，这一列的差异主要说明 masked 训练的作用。

## 3. 局限、没有做的

- 训练 seed 只有 2 个；across-target CI 不含 calibrator 训练 seed 与 trunk 的变异（所有 target 共用一个 trunk）。对照之间的离散度（不含 pooling 的 9 个对照 h0m、h1w*、*_loc 相对 P0 为 +0.010 到 +0.045）本身就是 “训练噪声 + 架构差异” 的量级，所以 ≤ +0.02 的 pooled − control 差不宜解读为效应。
- uncovered 比例 >0.30 的 target 只有 20141203（N=10）一个，斜率检验的有效样本极少；没有在 >0.5 的比例上评估（训练模拟的上限是 0.5）。uncovered 在目标上是 electrode 聚簇的，模拟也按 electrode 整组 mask，但 “哪些 electrode 在未来 session 里消失” 的真实机制（如 array 漂移）没有建模。
- H0 沿用 P1 的 P0，**训练时没有 masked 增强**；like-for-like 的参照是 h0m（同一协议加 masked 训练）。两者的差见 §2.2（`unc` +0.013）。
- 没有 oracle 列上限（例如用 target 全部 trial 拟合 uncovered 列）；§2.3 的 “置零” 只给出现有 trunk 下 uncovered 列的价值，不是上限。
- 没有与 R1-stream / R2 / R3 规则组合；没有 identity-free 全 pooling（A′1 已做）；没有量化、MAC 与 SRAM 代价核算（pooled 变体的 calibrator 参数约为 P0 的 2.5–4 倍；table 仍是 K×N，H5 不变）；没有更新 Overleaf exp-log（这一步留给调用者）。
- source val 只有一个 session（20140306），选择力很弱；因此 pooled 与对照在 source 上不可分。
- 评测中途一次 eval 进程被我自己的 `pkill` 误杀，从 resumable 的 jsonl 继续（按 (variant, seed, target, B, draw) 去重；分析脚本断言每个变体恰为 1152 行，通过）。

## 4. 文件

`cells_r2.csv`（变体 × target × B：R² 均值、CI、两个 seed、zero-uncovered / swap 的 R²）、`paired_across_targets.csv`（28 种比较 × {total, unc, cov} × {B=1…33, B≥2, all}：均值、t-CI、正号 target 数、seed0/seed1、对 uncovered 比例的 Spearman 与斜率）、`paired_per_target.csv`、`uncovered_columns_contribution.csv`、`source_val.csv`、`train_summary.csv`、`target_coverage.csv`、`flags.json`、图 `hybrid_gain_vs_uncovered_fraction.{png,pdf}`。

## 5. 复现

```
export PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_hybrid
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
$PY scripts/run_hybrid_identity.py check
$PY scripts/run_hybrid_identity.py train --variants h1,h2d,h2d_loc,h3d,h3d_loc,h0m,h1w138,h1w113     # job A
$PY scripts/run_hybrid_identity.py train --variants h2p,h2p_loc,h3p,h3p_loc,h1w153,h1w130           # job B
$PY scripts/run_hybrid_identity.py valsrc
$PY scripts/run_hybrid_identity.py eval --targets 20140626,20140627,20140929,20141203,20150511,20150512 --tag s1
$PY scripts/run_hybrid_identity.py eval --targets 20150610,20150611,20150612,20150615,20150616,20150617 --tag s2
$PY scripts/analyze_hybrid_identity.py
```
