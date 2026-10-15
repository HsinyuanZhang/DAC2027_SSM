# P2 迭代（工作线 A′）：identity-free 校准、公平梯度参照、校准通路量化、架构代价

日期：2026-10-15/16。执行者：工作线 A′（GPU0）。规格见 `docs/feasibility_p1_20261015.md` §5。前序结果见 `results/analysis/population_calib_20261015/README.md`（含 §0.1 独立审查）。
代码：`ssm_decode/{calib_quant,calib_cost,ui_reg}.py`、`ssm_decode/population_calib.py`（新增 `prior_mode`、`identity_free_prior`、`softmax_fn`、`weight_quant`）、`scripts/{run_pc2_queue,run_pc2_uireg,run_pc2_quant,pc2_cost_report,analyze_pc2}.py`、`scripts/run_population_calib.py`（新增 `--prior-mode`、`--init-from/--qat`、`PC_OUT`）、`tests/test_population_calib_p2.py`（45 个测试（审查修正新增 15 个），连同 P1 的 21 个共 66 个，全部通过）。
原始 rows：`results/population_calib_p2_20261015/`（gitignored）；本目录是汇总 csv/json（`tables_a1_a2_a3.md` 是全部 markdown 表，`a*_rows.csv.gz` 是逐行 rows）。
约束检查：trunk 冻结；候选行（P*、QAT）target 端 0 次反传；只用 B 个有标签 trial；未使用 0623/0625/0626，未读 `official_evalai/`；全部选择（ui_reg 的 lambda/steps/lr、accumulator/descriptor 的定点 scale、QAT 早停）只用 source。A′1 的 3072 条 eval row 全部满足 H1/H2/H4/H5 与 unit-ratio 检查（`a1_flags.json`）。`ui_*` 行是标注为非部署的梯度参照。P1 结果在改动后逐位复现（96 行 P0 eval row，最大差 9e-16）。

> **2026-10-16 审查后修正：A′3 的 accumulator 条目与 A′4 的代价表已被取代，见 §R；旧数字（逐 trial 取整、计数器被量化、代价遗漏）只在附录 S 中保留并标注为 superseded。§4.1/§4.2 中未涉及 accumulator 的行（权重、descriptor、输出表、LUT）不受影响。**

## R. 审查后修正（2026-10-16，独立审查之后）

独立审查对 A′3（accumulator 量化）与 A′4（代价模型）提出了两处高优先级和若干低优先级问题。本节给出修正后的模型、重跑的结果和结论；§0 第 3、4 点、§4、§5 已据此改写，**被取代的旧数字只保留在附录 S**。代码：`ssm_decode/calib_quant.py`（`perbin_stats`、`perbin_stats_multi`、`twolevel_stats`、`COUNTER_KEYS`、`DERIVED_KEYS`、`DATA_GROUPS`）、`ssm_decode/calib_cost.py`、`scripts/{run_pc2_quant,pc2_cost_report,analyze_pc2,pc2_hyser_peradd,pc2_uireg_ext}.py`；测试 `tests/test_population_calib_p2.py`（新增 15 个，共 45 个，连同 P1 的 21 个共 66 个，全部通过）。原始 rows：`results/population_calib_p2_20261015/fix/`（gitignored）。

### R.1 发现与修正一览

| # | 审查发现 | 修正 |
|---|---|---|
| 1 | accumulator 每个 trial 才取整一次（精确 float64 的 trial 内求和，只对跨 trial 的 running sum 取整） | 新的**默认**硬件模型 = 逐 add（逐 bin）定点：每个 bin 的增量先 round-to-nearest（ties-to-even）到寄存器步长，再饱和加法，不回绕；饱和次数计入 `sat_events`。旧模型保留为 **two-level design（wide per-trial staging + narrow running sum）**，名称 `acc*_2lvl`，其 staging 缓冲加进代价模型。 |
| 2 | bin 计数器（`n, ac_n, yx_n, cdir`）被当作数据量化 | 计数器是**精确的无符号整数计数器**，不参与数据量化（测试 `test_counters_are_never_quantised`）。位宽见 R.2.4。同时发现 `sphi`、`sphiphi` 是 `cdir` 的精确函数（8 个方向 bin × 常数 cos/sin 表），`sphix` 第 0 行等于 `sx`：这些**不再累加**，由 finalize 推出（`DERIVED_KEYS`），也不参与步长拟合；否则 `sphi[0]=n` 这类元素会把整个 key 的步长拉大，使 cos/sin 分量被舍成 0（第一次重跑中 14/16 bit 的 dir 组因此崩到 −1.5，是建模伪影，已修正）。 |
| 3 | 代价模型遗漏 | 加入 q_* 的 per-trial scratch、lag delay line（5N 的 x 历史 + 2D 的 y 历史）、two-level 的 staging 拷贝、P4 第二遍的 φ buffer（N×H）或重算（+每 unit Fd·H+2H² MAC）、计数器位宽；固定权重比例改为只计 weight×activation。HD-sEMG 连续特征的 per-add 误差已实测（R.2.6）。 |
| 4 | 措辞类 | 见 R.5。 |
| 5 | World 的 prior_mode 与检查点不一致时静默 | `rpc.load_run(..., strict_prior_mode=True)` 现在断言一致（新函数 `check_prior_mode`；旧检查点无该字段视为 `aligned`；传 `strict_prior_mode=False` 可恢复旧行为）。**这是默认行为的变化**。 |
| 6 | ui_reg 网格边缘 | 见 R.6。 |

### R.2 修正后的 A′3 accumulator 结果

**模型。** 每个数据 key 一个有符号定点寄存器（两补码，范围 ±(2^(b−1)−1)·step，step 为 power-of-two，取 source 训练 session 上 B=33 求和的最大 |值| 所需的最小步长，由 source 单独拟合）；每个 bin 的增量（乘积或选通输入）先舍入到 step 再加，加法饱和。lag 对 (x_t, x_{t−k}) 记在较晚到达的 bin；delay line 在 trial 边界清空。q_* 键用 per-trial scratch（sx、syx lag 0、sy lag 0，同样逐 add 取整，步长取这三个 key 的步长），trial 结束时把 sx²/n 等取整后饱和加进 q 寄存器。计数器精确，位宽单独提供。评测集同旧版（P0、P4 m=8；days / month / year；2 seed；B∈{1,8,33}；8 draw；val 与 target 分开）。

**month 与 year 不是两个独立模型**：两个 stratum 使用比特相同的 P0/P4 检查点（已核对，最大权重差 0），只是 target 与 trunk 不同。因此下面所有汇总按 **model group** 报告：`days` 与 `month_year`；不要把 month 与 year 的行当作独立重复。

**R.2.1 全部数据 key 同宽度（逐 add，计数器精确）**（P0/P4 target 的 ΔR² = 量化 − FP，B=1/8/33；完整含 val 的表：`a3_readme_tables.md`、`a3_cells_by_modelgroup.csv`）

| cfg | days p0 tgt dR2 B1/8/33 | days p4 tgt dR2 B1/8/33 | month_year p0 tgt dR2 B1/8/33 | month_year p4 tgt dR2 B1/8/33 | worst cell dR2 | NaN rows | sat rows |
|---|---|---|---|---|---|---|---|
| acc8 | -0.5113/-0.6175/-0.6791 | -0.6121/-0.6951/-0.7227 | -0.3385/-0.4179/-0.4503 | -0.3370/-0.3996/-0.4265 | -0.8838 | 0/672 | 0 |
| acc12 | -0.8215/-0.7269/-0.7295 | -0.8350/-0.6387/-0.6085 | -0.3438/-0.4713/-0.5340 | -0.4170/-1.0369/-1.2074 | -2.5306 | 36/672 | 4 |
| acc14 | +0.0002/-0.0001/-0.0003 | -0.0006/-0.0002/-0.0005 | nan/nan/nan | nan/nan/nan | -0.0013 | 384/672 | 16 |
| acc16 | -0.0000/+0.0000/-0.0002 | -0.0001/+0.0001/-0.0004 | -0.0001/-0.0001/-0.0000 | -0.0005/-0.0000/-0.0000 | -0.0005 | 0/672 | 16 |
| acc20 | +0.0000/+0.0000/-0.0002 | -0.0001/+0.0000/-0.0004 | +0.0000/+0.0000/+0.0000 | -0.0001/+0.0000/+0.0000 | -0.0004 | 0/672 | 16 |
| acc24 | -0.0000/-0.0000/-0.0002 | -0.0000/+0.0000/-0.0004 | +0.0000/+0.0000/+0.0000 | +0.0000/-0.0000/+0.0000 | -0.0004 | 0/672 | 16 |


**读法与结论。** 整数计数特征（spike count）下，逐 add 取整的关键条件是**步长 ≤ 1**：增量 x_i·x_j 的最小非零值是 1，步长为 2 时 round(0.5)=0（ties-to-even）把所有 1 丢掉，Gram 的对角线（方差）系统性偏小。步长 = 2^ceil(log2(max/(2^(b−1)−1)))：days 的 Gram 步长在 12/14/16 bit 为 4/1/0.25，month（year 同）为 8/2/0.5。所以：
- **12 bit 逐 add 完全不可用**：ΔR² −0.34 到 −1.2，B=1/8/33 都崩（与审查给出的 days −0.60 到 −0.73 同一量级，本次 days 为 −0.6 到 −0.84；P4 在 month_year 更差）；
- **14 bit 只对 days 成立**（worst −0.0013），month 与 year 的全部 384 行都是 NaN（Gram 步长 2，方差为负，见 R.2.3）；
- **16 bit 对所有 stratum / variant / B 无损**（worst cell −0.0005，rel-RMS 最大 3.9%（P4 month_year target B=1），val 同组 2.3%，其余 ≤1%），20/24 bit 同样；
- **B-adaptive exponent**（`*_bs`，步长随 B 缩小，**假设最终 B 事先已知**；anytime 流式时 B 会增长，需要在 B 翻倍时重新缩放全部寄存器）只在 B=1 起作用（12 bit 的 B=1：−0.0001，但 B≥8 仍崩；14 bit 只有 days 可用）。结论：不能靠 block exponent 把位宽压到 16 bit 以下；
- **two-level（staging 宽、running sum 窄，逐 trial 取整一次）**：12 bit 在 B=1 掉 −0.007（days）/−0.015（month_year），B≥8 ≥−0.003；16 bit 无损。也就是说，**旧版"12 bit 在 B≥8 几乎无损"只对 two-level 成立，对逐 add 的真实累加器不成立**；代价是 staging 缓冲（见 R.4）。

**R.2.2 每个 accumulator 组的最小安全位宽（逐 add；安全 = 所有 variant × val/target × model group × B∈{1,8,33} 的 cell 均值 ΔR² ≥ −0.002 且无 NaN 行；测试位宽 8/12/14/16，全 key 另测 20/24）**

| group | bits | worst cell ΔR² | worst cell | max rel-RMS | NaN rows | rows with a saturating add | safe |
|---|---|---|---|---|---|---|---|
| all | 8 | -0.8838 | p0/val/month_year/B33 | 102.2% | 0 | 0 | no |
| all | 12 | -2.5306 | p4_m8/val/month_year/B8 | 175.4% | 36 | 4 | no |
| all | 14 | -0.0013 | p4_m8/val/days/B1 | 1.7% | 384 | 16 | no |
| all | 16 | -0.0005 | p4_m8/target/month_year/B1 | 3.9% | 0 | 16 | **yes** |
| all | 20 | -0.0004 | p4_m8/target/days/B33 | 1.8% | 0 | 16 | **yes** |
| all | 24 | -0.0004 | p4_m8/target/days/B33 | 0.9% | 0 | 16 | **yes** |
| norm | 8 | -0.9118 | p4_m8/val/month_year/B33 | 103.2% | 0 | 0 | no |
| norm | 12 | -2.4390 | p4_m8/val/month_year/B8 | 177.4% | 36 | 0 | no |
| norm | 14 | -0.0004 | p4_m8/target/days/B33 | 0.4% | 384 | 16 | no |
| norm | 16 | -0.0004 | p4_m8/target/days/B33 | 0.4% | 0 | 16 | **yes** |
| ac | 8 | -0.1287 | p0/val/days/B1 | 39.5% | 0 | 0 | no |
| ac | 12 | -0.1337 | p0/val/days/B1 | 39.6% | 0 | 0 | no |
| ac | 14 | -0.1557 | p0/val/month_year/B1 | 55.9% | 0 | 0 | no |
| ac | 16 | +0.0000 | p0/target/days/B1 | 0.0% | 0 | 0 | **yes** |
| cross | 8 | -0.0411 | p0/target/month_year/B33 | 29.1% | 0 | 0 | no |
| cross | 12 | -0.0040 | p0/target/month_year/B1 | 6.2% | 0 | 0 | no |
| cross | 14 | -0.0010 | p4_m8/val/days/B1 | 1.9% | 0 | 0 | **yes** |
| cross | 16 | -0.0002 | p4_m8/target/month_year/B1 | 1.1% | 0 | 0 | **yes** |
| dir | 8 | -0.1139 | p0/target/month_year/B33 | 60.4% | 0 | 0 | no |
| dir | 12 | -0.0022 | p4_m8/val/days/B1 | 6.3% | 0 | 4 | no |
| dir | 14 | -0.0003 | p0/val/month_year/B1 | 3.0% | 0 | 0 | **yes** |
| dir | 16 | -0.0001 | p0/target/month_year/B1 | 0.7% | 0 | 0 | **yes** |
| quad | 8 | -0.0040 | p4_m8/val/days/B33 | 6.7% | 0 | 0 | no |
| quad | 12 | -0.0030 | p4_m8/val/days/B8 | 6.0% | 0 | 0 | no |
| quad | 14 | -0.0011 | p4_m8/target/month_year/B1 | 4.8% | 0 | 0 | **yes** |
| quad | 16 | -0.0004 | p4_m8/val/days/B1 | 3.8% | 0 | 0 | **yes** |

**最小安全位宽（测试网格 {8, 12, 14, 16} 内）：全部数据 key 统一 16 bit；分组：norm（sx, gram）16、ac（ac_xx, ac_a, ac_b）16、cross（sy, syx）14、dir（sdir, sphix）14、quad（q_*）14。** 13 bit 未测。组间的敏感度排序（12 bit，B=1 起各 B 的最差 cell）：norm（−0.4 到 −1.4，且产生 NaN）≫ ac（days −0.07 到 −0.11，month_year 约 −0.02）> dir（≤−0.002）≈ cross（≤−0.004）≈ quad（≤−0.003）；8 bit：norm（−0.34 到 −0.75）≫ dir（−0.03 到 −0.11）≈ ac（days −0.06 到 −0.10）> cross（≤−0.04）> quad（≤−0.004）。与旧版的"dir/sphi > norm > ac > cross > quad"不同：旧排序里 dir 的崩溃一部分来自计数器 `cdir` 和 `sphi` 被按数据量化，修正后 norm（Gram）是唯一在 12 bit 就崩溃的组。注意 days 单独看时 norm 在 14 bit 就安全（步长 1）；month_year 需要 16 bit。

**R.2.3 NaN 行。** 旧版 README 把 acc8 的 NaN 归因于"σ² = Σx²/n − μ² 的相消"，**这一解释不准确**：旧实现把 `n` 当作数据量化，n 的步长超过累计 bin 数的一半时（如 B=1）n 被舍成 0，`describe` 把它夹到 1，归一化 Gram 不再半正定，`corr` 列里 `sqrt(xvar·zvar)` 对负数开方得到 NaN（`/tmp/pc2_review/nan8.py` 的复现）。计数器精确之后 8 bit 不再产生 NaN（0/672 行）。NaN 仍然出现在两处，机制相同（量化后的 Gram 使归一化 Gram 非半正定，不是计数器问题）：12 bit 的 36/672 行（month/year B=1 val 与少数 target），以及 **14 bit 的 month 与 year 全部行**（Gram 步长 2）。主表（R.2.1 与 `a3_readme_tables.md`）已加 "NaN rows" 与 "sat rows" 两列；ΔR² 均值只在有限行上计算，"nan/nan/nan"表示该格全部行非有限。

**R.2.4 计数器宽度。** `n, ac_n(3), yx_n(5), cdir(8)` 共 17 个无符号计数器，上界为 B·L_max。000688 的最长 trial 为 2741 bin（0616），B=33 时 n ≤ 90,453，**需要 17 bit**；典型 trial（均值 240 bin）n≈7.9k，13 bit 够用，但按最坏情形配置；L_max ≤ 1985 时 16 bit 够。HD-sEMG（L=1250，B≤33）：41,250，16 bit。17 个计数器共 36 B（17 bit）。

**R.2.5 饱和与量程余量。** 逐 add 饱和行为：钳位在 ±qmax·step，不回绕。步长只由 source 拟合，不留余量：days target 0617 在 B=33 时 Gram 达到 source 量程的 1.08 倍，16 bit 时这 8/32 行出现饱和 add（ΔR² 影响 ≤0.0004）；饱和与位宽无关（20/24 bit 同样饱和，因为步长按量程缩放），需要**量程余量（+1 bit）**而不是更多精度位。有符号累加器（sy、syx）的中间值可能超过最终值，量程应按 running sum 的最大偏移配置（HD 实测见 R.2.6）。

**R.2.6 HD-sEMG 连续特征（未解码，只测 Gram / 交叉和相对 float64 的误差；`pc2_hyser_peradd.py`，`hyser_peradd*.csv`）。** 000688 是整数计数，Hyser 缓存特征是连续包络（80 ms 漏积分，50 Hz，N=256，每 trial 1250 bin，float32）。数据：source = subject 4–6、target = subject 1–3（session 1），15 个 trial；寄存器步长取 source 的 15-trial 量程（"source"），或取 target 自己的 15-trial 量程（"own"，理想自动增益，只隔离取整误差）；B∈{1,8,15}。

| range 模式 | 位宽 | B | Gram 相对误差 | sx | syx(lag 0) | 相关矩阵 relerr | std 相对误差 | 饱和 add 数 | 方差≤0 的通道数 |
|---|---|---|---|---|---|---|---|---|---|
| own | 16 | 1 | 9.37e-01 | 3.9e-01 | 7.8e-02 | 2.1e+00 | 7.8e-01 | 0 | 489 |
| own | 16 | 15 | 8.73e-01 | 3.9e-01 | 1.0e-01 | 2.0e+00 | 6.1e-01 | 261 | 315 |
| own | 20 | 1 | 1.49e-01 | 4.5e-03 | 9.6e-04 | 1.2e+00 | 3.3e-01 | 0 | 146 |
| own | 20 | 15 | 1.66e-01 | 2.8e-03 | 4.1e-03 | 9.8e-01 | 2.4e-01 | 1550 | 68 |
| own | 24 | 1 | 5.09e-03 | 8.1e-05 | 5.6e-05 | 1.7e-01 | 1.8e-02 | 0 | 8 |
| own | 24 | 15 | 2.67e-03 | 2.2e-05 | 3.5e-03 | 1.5e-01 | 1.2e-02 | 1811 | 5 |
| own | 28 | 1 | 5.71e-05 | 5.1e-06 | 3.4e-06 | 7.8e-03 | 6.1e-04 | 0 | 0 |
| own | 28 | 15 | 1.68e-05 | 1.4e-06 | 3.5e-03 | 1.5e-03 | 1.2e-04 | 1823 | 0 |
| own | 32 | 1 | 3.56e-06 | 3.0e-07 | 2.2e-07 | 4.9e-04 | 2.9e-05 | 0 | 0 |
| own | 32 | 15 | 1.00e-06 | 8.4e-08 | 3.4e-03 | 6.5e-05 | 3.9e-06 | 1824 | 0 |
| source | 16 | 1 | 4.17e-01 | 3.9e-01 | 7.8e-02 | 7.4e-01 | 3.1e-01 | 0 | 147 |
| source | 16 | 15 | 4.14e-01 | 3.9e-01 | 1.0e-01 | 2.9e-01 | 3.5e-01 | 15423 | 10 |
| source | 20 | 1 | 2.34e-02 | 4.5e-03 | 9.6e-04 | 3.0e-01 | 7.6e-02 | 0 | 11 |
| source | 20 | 15 | 1.15e-01 | 2.8e-03 | 4.1e-03 | 2.7e-01 | 3.5e-01 | 34843 | 11 |
| source | 24 | 1 | 1.71e-04 | 8.1e-05 | 5.6e-05 | 7.1e-02 | 1.5e-03 | 0 | 1 |
| source | 24 | 15 | 1.09e-01 | 2.2e-05 | 3.5e-03 | 1.3e-01 | 3.5e-01 | 42232 | 2 |
| source | 28 | 1 | 1.03e-05 | 5.1e-06 | 3.4e-06 | 6.4e-04 | 6.6e-05 | 0 | 0 |
| source | 28 | 15 | 1.09e-01 | 1.4e-06 | 3.5e-03 | 7.3e-02 | 3.5e-01 | 42515 | 1 |
| source | 32 | 1 | 6.42e-07 | 3.0e-07 | 2.2e-07 | 4.6e-05 | 3.9e-06 | 0 | 0 |
| source | 32 | 15 | 1.09e-01 | 8.4e-08 | 3.4e-03 | 7.3e-02 | 3.5e-01 | 42581 | 1 |

结论：连续特征下逐 add 取整的相对误差远大于整数计数。**Gram 相对误差在 16 bit 为 0.87–0.94（完全不可用），20 bit 0.15–0.19，24 bit 3–5×10⁻³，28 bit ≤6×10⁻⁵，32 bit ≈0**；用于归一化的相关矩阵在 24 bit 仍有 15–17% 的相对误差（低方差通道方差为负），28 bit 降到 0.2–0.8%，32 bit 7×10⁻⁵–5×10⁻⁴。原因同 000688：通道包络均值 ≈20–30、标准差 ≈12–26，Gram 由均值项主导，方差靠相消得到，所需的有效位数 ≈ log2(量程/最小增量) 远大于 16。source 拟合的量程会被受试者间幅度差（最大值差 10 倍以上）打穿（"source" 行的饱和 add 数以 10⁴ 计，24 bit 以上仍有 9–11% 的 Gram 误差），所以 HD-sEMG 需要每 session 的自动增益或 block exponent。**因此"16 bit 累加器"的结论只适用于整数计数特征（000688），不能外推到 HD-sEMG；HD-sEMG 逐 add 至少需要 28 bit（连续特征）**。替代方案：把输入量化成 int8 码后精确累加，Gram 需要 21 bit（`hyser_int8_exact_width.csv`：15 个 trial 的最大 Gram 9×10⁵ 码²）；其解码质量（int8 输入本身的损失）**未测**。syx 在 B=15 的误差 3.4×10⁻³ 是中间值饱和（有符号累加，量程按最终和拟合）造成的，需要 +1 bit 余量。

**R.2.7 组合（含逐 add 的 accumulator）。** w8 + acc16 + desc16 + tab8（+LUT）：最差 cell −0.0006；w8 + acc20 同；w4 + acc16 + desc12 + tab8：最差 −0.0052（来自 int4 权重，与 accumulator 无关）；tern + acc16 + desc16 + tab4：PTQ 的 −0.14 主导。旧版组合里的 "w4 + acc12" 在逐 add 下不再成立（acc12 崩溃），已替换为 acc16。

### R.3 ternary QAT 的更正

- **按 model group 报告**（`a3_qat_ternary_pooled.csv`；month 与 year 共用同一个 QAT 检查点，不是独立重复）。
- **val 行是乐观的**：QAT 的早停就是在同一个 source val session 上做的（P1 规则），所以 val 的 ΔR² 偏好 QAT；target 行才是外样本。
- **步数**：旧文写"300–1200 步"不对。实际 6 个 best step 为 100–900（budget 为 600 或 1200，翻倍规则触发 3 次），见 `results/.../runs/*_qat_tern/seed*/done.json`。
- **结论较旧版弱**：按 model group 拆开后，QAT 在 days target 把 ΔR² 拉回 ±0.01 内，但 month_year target 在 B≥8 仍掉 0.02–0.025（P0），P4 掉 0.001–0.011；P4 days val 仍掉 0.018–0.030。旧版 pooled 的"−0.012 到 +0.001（target）"是 days 与 month_year 平均的结果，掩盖了 month_year 的残余损失。

| 变体 | 评测集 | model group | FP（B=1/8/33） | ternary PTQ | ternary QAT | ΔR² PTQ | ΔR² QAT (CI) |
|---|---|---|---|---|---|---|---|
| P0 | target | days | 0.647/0.723/0.738 | 0.589/0.671/0.684 | 0.639/0.723/0.738 | -0.059/-0.053/-0.054 | -0.008±0.008/+0.000±0.003/+0.000±0.002 |
| P0 | target | month_year | 0.332/0.410/0.441 | 0.298/0.358/0.375 | 0.326/0.388/0.416 | -0.034/-0.052/-0.066 | -0.006±0.014/-0.022±0.010/-0.025±0.009 |
| P0 | val（**乐观**：QAT 在该 session 上早停） | days | 0.636/0.699/0.719 | 0.556/0.610/0.629 | 0.626/0.707/0.720 | -0.080/-0.089/-0.090 | -0.010±0.011/+0.008±0.005/+0.002±0.003 |
| P0 | val（**乐观**：QAT 在该 session 上早停） | month_year | 0.811/0.858/0.859 | 0.788/0.835/0.835 | 0.806/0.853/0.854 | -0.023/-0.023/-0.025 | -0.005±0.004/-0.005±0.001/-0.005±0.001 |
| P4_M8 | target | days | 0.663/0.724/0.742 | 0.622/0.689/0.701 | 0.662/0.728/0.743 | -0.041/-0.035/-0.040 | -0.001±0.012/+0.004±0.004/+0.001±0.003 |
| P4_M8 | target | month_year | 0.333/0.395/0.422 | 0.318/0.358/0.382 | 0.333/0.394/0.411 | -0.014/-0.037/-0.040 | +0.001±0.010/-0.001±0.011/-0.011±0.013 |
| P4_M8 | val（**乐观**：QAT 在该 session 上早停） | days | 0.644/0.730/0.755 | 0.556/0.604/0.614 | 0.631/0.712/0.725 | -0.088/-0.126/-0.142 | -0.013±0.013/-0.018±0.009/-0.030±0.008 |
| P4_M8 | val（**乐观**：QAT 在该 session 上早停） | month_year | 0.814/0.857/0.857 | 0.798/0.839/0.839 | 0.811/0.857/0.857 | -0.016/-0.017/-0.018 | -0.002±0.004/+0.000±0.001/-0.000±0.001 |

### R.4 修正后的 A′4 代价（解析；`cost_model.csv`、`cost_tables.md`）

遗漏的项已全部加入（`calib_cost.sram_bytes`/`full_report` 的新键；旧键值不变，**唯一改变默认值的是 `fixed_weight_fraction`**，现在只计 weight×activation，旧口径保留为 `forward_share_incl_actact`）：
- **q_* 的 per-trial scratch**（sx、syx lag 0、sy lag 0、n）：N + D·N + D 个字，与 accumulator 同宽；
- **lag delay line**：x 历史 5N（最大的自相关 lag）+ y 历史 2D（负 cross lag）+ 有效位；
- **two-level 的 staging 拷贝**：除 q_* 外全部数据 key 的 32 bit 拷贝（宽度是假设，足够容纳 per-trial 求和）+ 计数器；
- **计数器**单独按其位宽（17 bit，HD 16 bit）；`sphi/sphiphi/sphix[0]` 不再累加（`derived_dir`：每 bin 少 9+N 个 MAC、3 个 add，少 12+N 个寄存器）；
- **P4 第二遍**（pooling 完成之后才能生成每个 unit 的输出）需要 φ(d̃_i)：要么存 N×H 的 buffer，要么重算（每 unit Fd·H+2H² MAC，000688 为 11,200，HD 为 12,928）。两种都给；
- **固定权重比例**排除 activation×activation（softmaxᵀ·values 的 pooling 与 β·z 的 μ_i）。

**MAC / 一次校准 (B=8)**

| quantity | 000688 P0 | 000688 P4 | HD-sEMG P0 | HD-sEMG P4 |
|---|---|---|---|---|
| acc MAC / valid bin (full) | 1,101 | 1,101 | 73,481 | 73,481 |
| acc MAC / trial | 242,352 | 242,352 | 91,854,071 | 91,854,071 |
| finalize MAC | 15,185 | 15,185 | 2,173,723 | 2,173,723 |
| calibrator forward MAC (all units) | 201,344 | 643,552 | 2,523,136 | 7,319,552 |
|   of which activation x activation | 0 | 13,312 | 0 | 131,072 |
|   of which fixed-weight (RRAM-mappable) | 201,344 | 630,240 | 2,523,136 | 7,188,480 |
| total MAC, one calibration (phi buffered) | 2,155,345 | 2,597,553 | 739,529,427 | 744,325,843 |
| total MAC, phi recomputed in pass 2 | 2,155,345 | 2,888,753 | 739,529,427 | 747,635,411 |
| fixed-weight share of all MAC (corrected) | 9.34% | 24.26% | 0.34% | 0.97% |
| old share (forward incl. act x act) | 9.34% | 24.78% | 0.34% | 0.98% |

**SRAM (KiB), P0, B=8; 表 int8 双缓冲, 描述量 16 bit**

| case | option | acc width | accumulators (data+counters) | of which counters (B) | q_* trial scratch | lag delay lines | two-level staging | descriptors | table x2 | pool state | TOTAL | TOTAL + P4 phi buffer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000688 | fp32 | 32 | 6.5 | 68 | 0.3 | 0.1 | 0.0 | 5.4 | 2.0 | 0.0 | 14.4 | - |
| 000688 | perAdd_uniform | 16 | 3.2 | 36 | 0.2 | 0.1 | 0.0 | 2.7 | 0.5 | 0.0 | 6.8 | - |
| 000688 | perAdd_perGroup | {"norm": 16, "ac": 16, "cross": 14, "dir": 14, "quad": 14} | 3.0 | 36 | 0.2 | 0.1 | 0.0 | 2.7 | 0.5 | 0.0 | 6.6 | - |
| 000688 | twoLevel_uniform | 16 | 3.2 | 36 | 0.2 | 0.1 | 5.8 | 2.7 | 0.5 | 0.0 | 12.5 | - |
| HD-sEMG | fp32 | 32 | 313.2 | 68 | 6.0 | 2.5 | 0.0 | 80.0 | 32.0 | 0.0 | 434.8 | - |
| HD-sEMG | perAdd_uniform | 16 | 156.1 | 34 | 3.0 | 2.5 | 0.0 | 40.0 | 8.0 | 0.0 | 210.6 | - |
| HD-sEMG | perAdd_perGroup | {"norm": 16, "ac": 16, "cross": 14, "dir": 14, "quad": 14} | 153.2 | 34 | 3.0 | 2.5 | 0.0 | 40.0 | 8.0 | 0.0 | 207.7 | - |
| HD-sEMG | twoLevel_uniform | 16 | 156.1 | 34 | 3.0 | 2.5 | 301.1 | 40.0 | 8.0 | 0.0 | 511.8 | - |
| HD-sEMG | perAdd_cont28 | 28 | 273.1 | 34 | 5.3 | 2.5 | 0.0 | 40.0 | 8.0 | 0.0 | 329.9 | - |
| HD-sEMG | perAdd_int8code21 | 21 | 204.9 | 34 | 4.0 | 1.3 | 0.0 | 40.0 | 8.0 | 0.0 | 259.1 | - |

**SRAM (KiB), P4, B=8; 表 int8 双缓冲, 描述量 16 bit**

| case | option | acc width | accumulators (data+counters) | of which counters (B) | q_* trial scratch | lag delay lines | two-level staging | descriptors | table x2 | pool state | TOTAL | TOTAL + P4 phi buffer |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000688 | fp32 | 32 | 6.5 | 68 | 0.3 | 0.1 | 0.0 | 5.4 | 2.0 | 1.1 | 15.5 | 18.7 |
| 000688 | perAdd_uniform | 16 | 3.2 | 36 | 0.2 | 0.1 | 0.0 | 2.7 | 0.5 | 1.1 | 7.8 | 11.1 |
| 000688 | perAdd_perGroup | {"norm": 16, "ac": 16, "cross": 14, "dir": 14, "quad": 14} | 3.0 | 36 | 0.2 | 0.1 | 0.0 | 2.7 | 0.5 | 1.1 | 7.7 | 10.9 |
| 000688 | twoLevel_uniform | 16 | 3.2 | 36 | 0.2 | 0.1 | 5.8 | 2.7 | 0.5 | 1.1 | 13.6 | 16.8 |
| HD-sEMG | fp32 | 32 | 313.2 | 68 | 6.0 | 2.5 | 0.0 | 80.0 | 32.0 | 1.1 | 435.8 | 467.8 |
| HD-sEMG | perAdd_uniform | 16 | 156.1 | 34 | 3.0 | 2.5 | 0.0 | 40.0 | 8.0 | 1.1 | 211.7 | 243.7 |
| HD-sEMG | perAdd_perGroup | {"norm": 16, "ac": 16, "cross": 14, "dir": 14, "quad": 14} | 153.2 | 34 | 3.0 | 2.5 | 0.0 | 40.0 | 8.0 | 1.1 | 208.8 | 240.8 |
| HD-sEMG | twoLevel_uniform | 16 | 156.1 | 34 | 3.0 | 2.5 | 301.1 | 40.0 | 8.0 | 1.1 | 512.8 | 544.8 |
| HD-sEMG | perAdd_cont28 | 28 | 273.1 | 34 | 5.3 | 2.5 | 0.0 | 40.0 | 8.0 | 1.1 | 331.0 | 363.0 |
| HD-sEMG | perAdd_int8code21 | 21 | 204.9 | 34 | 4.0 | 1.3 | 0.0 | 40.0 | 8.0 | 1.1 | 260.1 | 292.1 |

**要点（修正后）。** (i) 000688：MAC 约 2.2 M（P0）/2.6 M（P4，φ 重算 2.9 M）；SRAM 约 6.8 KiB（P0）/7.8 KiB（P4）（逐 add、16 bit、int8 表），P4 若缓存 φ 为 11.1 KiB，**two-level 的 staging 使 SRAM 翻倍**（13.6 KiB）；全部低于 32 KB。P4 的 φ 选择：缓存 3.3 KiB 对应重算多 11% 的总 MAC。(ii) HD-sEMG：累加仍是主要算术（73.5 k MAC/bin）；固定权重 calibrator 只占 0.34%（P0）/0.97%（P4），修正 act×act 后几乎不变（旧 0.98%）。**连续特征需要 28 bit 逐 add 累加（Gram 224 KiB），总 SRAM 约 330 KiB（int8 码 + 21 bit 精确累加：约 260 KiB）**，旧表"16 bit、约 212 KB"对连续特征不成立；任何一种都远超 32 KB 的表预算，Gram 是主要对象。two-level 对 HD 不可行（staging 301 KiB）。(iii) 仍是解析计数，不是硬件测量。

### R.5 其他措辞修正（低优先级）

- "FP 重复运行逐位确定"不准确：同一 FP 子集在两次运行（旧 run 与本次 fix run）之间 days 最大相差 2.3×10⁻⁵（均值 3.8×10⁻⁶），month 为 0；同一点内不同批次的 `r2_fp` 最大相差 2.0×10⁻⁵（GPU 非确定性 / 批处理）。因此 |ΔR²| < 1×10⁻⁴ 的差别都在数值噪声内。
- **B-adaptive exponent 假设最终 B 事先已知**；anytime 流式需要随 B 增长重新缩放全部寄存器（见 R.2.1）。
- `Exp2LUT` 的输出范围不是 [0,1]：中点项使最大输出为 2^(0.5/256) = 1.00135（exp(0)→1.00135），输出寄存器需要 1 个整数位或饱和到 1；已修正 docstring，行为不变（softmax 归一化吸收公共偏差）。
- month 与 year 的 A′3 / QAT 共用同一模型（R.2、R.3），不当作独立重复。
- A′2 在 R.6。

### R.6 A′2（ui_reg）措辞修正与 B=33 的网格扩展

**措辞更正（不需要重跑）。**
- 原 A′2 的 ui_reg 选择在网格边缘：month 的 B=1/2 选到 steps=10（下界），days 与 month 的 B=33 选到 steps=300（上界）且 source R² 仍在上升。因此 **B=33 的 ui_reg 可能欠收敛，偏差方向是对 P0 有利**（ui_reg 的 R² 偏低、P0 − ui_reg 偏大）；下面的网格扩展实测显示这一偏差 ≤0.01。B=1/2 的下界只说明最优可能是更少步数/更强的早停，对 ui_reg 偏低的方向同样有利于 P0。
- ui_reg 相对无正则 ui_elec 在 B=1 的增益（+0.06 到 +0.09）应归因于 **(lambda, lr, steps) 的联合选择**（B=1 选出 lambda=1、lr 1e-3 或 1e-2、10–300 步），而不能单独归因于正则项：ui_elec 固定 300 步、lr 3e-3，两者在步数 / 学习率上本来就不同，没有做只改 lambda 的消融。

**B=33 的网格扩展（可做的部分：约 1 GPU 小时量级）。** 只在 B=33、只用 source session 重新选择，steps ∈ {10, 30, 100, 300, 600, 1000}，(lambda, lr) 网格缩小为 lambda ∈ {0, 1e-4, 1e-3, 1e-2} × lr ∈ {3e-3, 1e-2}（原网格里 B=33 从未选到 lr=1e-3，lambda ≥ 0.1 的 B=33 选择为近 0），每个 source session 1 个 draw（原来 2 个，GPU 与其他任务共用，缩减以控制时间）；然后在 target 上重评 ui_reg（起点为电极对齐 prior，8 个 draw，year 复用 month 的选择）。选择结果：days：lambda=0.001, lr=0.003, steps=300, source R²=0.695（6 个 source episode）；month：lambda=0.01, lr=0.003, steps=600, source R²=0.799（6 个 source episode）。

| stratum | B=33 ui_reg 原选择（steps≤300） | 扩展后（steps≤1000） | 差（扩展−原，±CI） | 扩展后选出的 (lambda, lr, steps) |
|---|---|---|---|---|
| days | 0.704 | 0.704 | -0.000±0.000 (n=16) | (0.001, 0.003, 300) |
| month | 0.720 | 0.719 | -0.000±0.007 (n=8) | (0.01, 0.003, 600) |
| year | 0.478 | 0.486 | +0.008±0.005 (n=8) | (0.01, 0.003, 600) |

结论：扩展步数后 B=33 的 ui_reg 变化 ≤0.01（见表）：days 重选出同一规则（steps=300，上界并不起约束作用，结果逐位一致），month 选到 600 步但 R² 不变，year 仅 +0.008。因此 B=33 的 ui_reg 欠收敛的假设**不成立到足以改变结论**，§0.2 不变：P0 在 month/year 的 B=33 amortization gap（0.15–0.17）仍然成立，欠收敛带来的偏差量级 ≤0.01，不是该差距的主要来源。注意这只是缩小网格 + 1 draw 的重选，不是对 A′2 完整网格的重做；B=1/2 的下界（steps<10）未扩展；P0+ui_reg 没有重选。

### R.7 本次没有做的

- 没有为 13 bit（12 与 14 之间）、也没有为其它 rounding mode（round-half-up、stochastic rounding）测试；stochastic rounding 或 dithering 可能让较窄的寄存器可用，但那是不同的硬件设计。
- 没有对 HD-sEMG 做解码（没有 Hyser 的 trunk / 解码器）；R.2.6 只测 Gram / 交叉和的相对误差。int8 码输入的解码质量未测。
- 没有重做 A′1，也没有重训任何检查点（QAT 检查点沿用）。
- 没有测硬件；staging 的 32 bit 宽度是假设。
- Overleaf exp-log 与 GDrive 备份未更新（本次只改 GitHub 本地提交，未 push）。

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
   - 权重（calibrator，per-output-channel）：int8 的 ΔR² 绝对值 ≤0.0001（预测 RMS 变化 0.5–0.7%）；int4 ≤0.002（RMS 2–7%）；**ternary（TWN）PTQ 掉 0.03–0.06（RMS 22–35%）**。用 source 上 STE 的 ternary QAT 微调（best step 100–900，budget 600–1200；已做，不属“重训”，但需说明）：**按 model group 拆开**（§R.3），days target 的 ΔR² 回到 ±0.01 内，但 month_year target 在 B≥8 仍掉 0.02–0.025（P0）、P4 掉 0.001–0.011；val 行偏乐观（QAT 早停就在该 val session 上）；month 与 year 共用同一个检查点，不是独立重复。
   - **accumulator（已按审查修正，见 §R.2，逐 add 取整、计数器精确）：12 bit 不可用（ΔR² −0.34 到 −1.2，B=1/8/33 都崩）；14 bit 只对 days 可用，month/year 全部 NaN；16 bit 对所有 stratum/variant/B 无损（worst −0.0005）。各组最小安全位宽：norm 16、ac 16、cross 14、dir 14、quad 14，统一 16 bit。**这只适用于整数计数特征：HD-sEMG 连续特征在 16 bit 下 Gram 相对误差 0.9，至少要 28 bit（R.2.6）。two-level 设计（逐 trial 取整）在 12 bit 也只在 B≥8 可用，且 staging 使 SRAM 翻倍。
   - 最终描述量的定点（16/12/8 bit，每列一个步长）：ΔR² 绝对值 ≤0.0004，RMS 变化 0.5–1%，即描述量本身 8 bit 就够，瓶颈在累加器，而不在描述量。
   - 输出表 int8（每行一个 scale）：|ΔR²|≤0.0001；int4：0 到 −0.005（RMS 7–9%）。
   - P4 softmax 用 exp2 LUT（256×16 bit）加 reciprocal（精确或 256 项 LUT）：ΔR² 绝对值 <0.00005，RMS 变化 0.4–0.5%，等于数值噪声底（见 §4.2），P3/P4 的 softmax 可以用 LUT 实现。
   - **source val 与 target 的差异**：val 上的 ΔR² 一般小于或等于 target（target 的 FP R² 低，噪声交叉项更大）；两者的排序一致。RMS 变化列才是量化本身的扰动，二者在 val 与 target 上相近（ternary：val 22–25%，target 29–35%，同一量级）。

4. **A′4（代价，解析；已按审查修正，见 §R.4）**：000688（N=26, K=10, D=2）一次 B=8 校准约 2.2 M（P0）/2.6 M（P4；φ 重算 2.9 M）MAC，SRAM 约 6.8/7.8 KiB（逐 add、16 bit 累加器、int8 双缓冲表，已计入 q_* scratch、lag delay line、精确计数器 36 B），P4 缓存 φ 再加 3.3 KiB；two-level 的 staging 使 SRAM 翻倍。固定权重（RRAM 可映射）的 MAC 占比改为只计 weight×activation：P0 9.3%/P4 24.3%（000688），0.34%/0.97%（HD-sEMG）。HD-sEMG（N=256, K=16, D=5）连续特征逐 add 需要约 28 bit 累加器（Gram 224 KiB），总 SRAM 约 330 KiB（int8 码 + 21 bit 精确累加约 260 KiB），远超 32 KB。

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
- accumulators（**已按审查修正，见 §R.2**）：逐 add 定点寄存器 + 精确计数器 + 派生键；步长为 power-of-two，由 source 训练 session 的 B=33 求和量程拟合。两种读法仍然都评测：accumulator（`acc*`）与最终 descriptor（`desc*`）。旧设置（逐 trial 取整、计数器被量化）见附录 S。
- “descriptor accumulators” 有两种读法，**两种都评测**：上面的 accumulators（`acc*`）和最终描述量/可靠性特征 d、r（`desc*`，每列一个 power-of-two 步长，同样来自 source）。
- 输出表：每行（每个潜变量 k）一个 scale 的对称 int8/int4（`tab*`）。
- softmax：`Exp2LUT`（256 项，16 bit 码，15 位小数；项取区间中点，相对误差界 0.135%+code 舍入；输出舍入到 16 位小数）和 reciprocal（`lut`=尾数 256 项 LUT，误差界 0.21%；`exact`=精确倒数）。作用于 P4 的两处 softmax（unit 间 pooling 与 route）；是批量形式，不是 online-softmax 逐块形式（后者的块间 rescale 也要过 LUT，未单独评测）。
- 报告 ΔR² = 量化 − 同一子集上的 FP；`rms_delta` = query bin 上预测差的 RMS（物理单位），`rel_rms` = 它除以 FP 预测的标准差。
- ternary QAT（额外项，非“评测不重训”）：从 P1 检查点出发，source 上 STE fake-quant 微调（lr 3e-4，早停同 P1 规则；实际 best step 100–900，budget 600–1200，见 §R.3），仍无 target 数据；`--init-from/--qat` 实现。

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

**解读与限制（审查后补充见 §R.6：B=33 的 ui_reg 可能欠收敛）。** (i) 公平化并没有推翻 P1 的结论：ui_reg 在 B≥8 与 ui_elec 差别 ≤0.03，B=33 因为选出的 lambda 近 0 而完全等价；它改善的是 B=1–2（+0.06 到 +0.09），但 B=1 的 ui_reg 仍低于 P0 0.27（days）/0.08（month），year 持平。所以“B=1 时 P 变体远高于任何梯度方法”仍然成立；“days 层 P 变体不低于梯度参照”也成立（ui_reg 的 B=33 为 0.704，P0 为 0.738）。(ii) month/year 的 B=33 amortization gap 在 ui_reg 下为 0.153/0.163，与 P1 对 ui_elec 报告的 0.155/0.138 同量级。(iii) 新发现：**从 P0 的表出发的正则化细化在 month/year 显著提升（B=8 +0.09/+0.05，B=33 +0.17），在 days 层无效**；P1 审查（§0.1 第 7 点）只在 days 层试过 P0 起点，结论“从 P0 的表出发再做梯度下降反而下降”因此只对 days 成立。(iv) 限制：source session 对 trunk 是 in-sample（month 的 source 模拟 R² 0.62–0.76，target 的 ui_reg 只有 0.39–0.72），所以选出的 lambda 对 target 可能偏小；P0+ui_reg 没有为 P0 起点单独选参；year 复用 month 的选择；ui_reg 只有 1 个 seed。这些是参照行，不是候选。

## 4. A′3 结果

### 4.1 主要数字（P0 与 P4(m=8)；3 个 stratum、2 个 seed 合并；ΔR² = 量化 − FP，RMS = 预测 RMS 变化相对 FP 预测标准差）

FP R²：P0 val 0.753/0.805/0.812，target 0.490/0.567/0.589（B=1/8/33）；P4 val 0.757/0.814/0.823，target 0.498/0.560/0.582。

**以下 pooled 表只保留与 accumulator 无关的行（权重、descriptor、输出表、LUT；数值未受审查影响）。accumulator 与组合行已被 §R.2 取代（旧行在附录 S）。注意 pooled 把 month 与 year 当作两个 stratum 平均，二者共用同一模型，不是独立重复；按 model group 的拆分在 `a3_cells_by_modelgroup.csv`。**

pooled 各配置（完整表含 P0/P4 与 val/target 的 CI：`a3_cells.csv`、`tables_a1_a2_a3.md`，分 stratum：`a3_cells_by_stratum.csv`）：

| 配置 | P0 val ΔR² B=1/8/33 | P0 target ΔR² B=1/8/33 | P0 target RMS B=1/8/33 | P4 target ΔR² B=1/8/33 | P4 target RMS B=1/8/33 |
|---|---|---|---|---|---|
| w8 | −0.0000/−0.0000/−0.0001 | −0.0001/+0.0000/+0.0001 | 0.7/0.6/0.6% | −0.0001/−0.0001/−0.0001 | 0.6/0.6/0.5% |
| w4 | −0.0020/+0.0001/−0.0000 | −0.0018/−0.0008/−0.0006 | 7.3/5.7/5.9% | +0.0003/−0.0004/−0.0021 | 5.2/4.0/4.3% |
| tern（PTQ） | −0.0421/−0.0452/−0.0464 | −0.0464/−0.0524/−0.0601 | 34.9/31.1/32.0% | −0.0278/−0.0358/−0.0402 | 30.9/29.3/29.3% |
| desc16 | +0.0000/+0.0000/−0.0000 | +0.0000/−0.0000/−0.0000 | 0.5/0.5/0.5% | −0.0000/+0.0000/−0.0000 | 0.5/0.5/0.5% |
| desc12 | +0.0000/−0.0000/−0.0000 | −0.0000/−0.0000/−0.0000 | 0.5/0.5/0.5% | −0.0000/−0.0000/−0.0000 | 0.5/0.5/0.5% |
| desc8 | +0.0001/+0.0001/+0.0000 | +0.0001/+0.0003/+0.0002 | 1.0/0.9/0.9% | +0.0002/+0.0002/+0.0004 | 0.9/0.8/0.9% |
| tab8 | −0.0000/+0.0000/−0.0000 | −0.0001/−0.0001/+0.0001 | 0.8/0.7/0.7% | +0.0000/−0.0001/−0.0001 | 0.8/0.7/0.7% |
| tab4 | −0.0011/−0.0020/−0.0045 | −0.0003/−0.0029/+0.0001 | 9.2/8.0/7.6% | −0.0039/−0.0030/−0.0039 | 8.9/8.6/8.3% |
| lut（exact 倒数） | - | - | - | −0.0000/+0.0000/−0.0000 | 0.5/0.4/0.4% |
| lut_rlut（LUT 倒数） | - | - | - | −0.0000/+0.0000/−0.0000 | 0.5/0.5/0.5% |


### 4.2 ternary 的 QAT 对照与数值噪声底

（**已被 §R.3 取代**：按 model group 报告，标注 val 为乐观，步数更正为 best step 100–900。）

QAT 的 source val 与原检查点相近（val 是早停所用的 session，数字偏乐观）（days 0.68–0.70，month/year 0.83–0.84，见 `a3_qat_ternary*.csv`）。

**数值噪声底**：trunk 的输出对 table 的极小扰动已有 0.4–0.5% 的预测 RMS 变化（我们实测：对 P0 days target 的表加幅度为表平均绝对值 1e-6 倍的随机扰动，RMS 变化 0.0167，约为预测标准差 4.1 的 0.4%；扰动 1e-3 时 0.025，饱和）。因此 `rel_rms` ≤0.5% 的行（desc16/12、acc16、lut、w8、tab8）是在噪声底，不应当读成量化的影响；同一个 FP 子集重复运行并非逐位确定（days 最大相差 2.3×10⁻⁵，见 §R.5）。


**非有限预测**：见 §R.2.3（旧解释有误，已更正；旧数字在附录 S）。

### 4.3 解读

- **RRAM（固定权重 calibrator）**：8 bit 与 4 bit 权重都不需要重训（4 bit 的 RMS 扰动 2–7%，ΔR² ≤0.002，w4 在 P4 month_year B=33 为 −0.005）。**ternary 不能 PTQ**（−0.03 到 −0.06），STE 微调便宜（约 3–5 分钟/run，source-only）但回收并不完全：按 model group 拆开，days target 回到 ±0.01 内，month_year target 在 B≥8 仍掉 0.02–0.025（P0）（§R.3）。2T2R 的 ternary 映射可行，前提是训练时加 QAT，且 month/year 类漂移下仍有残余损失。
- **累加器宽度是校准通路里最敏感的部分**（已按审查修正，见 §R.2）：逐 add 模型下 16 bit 无损，12 bit 不可用（block exponent 只在 B=1 有用，且假设 B 事先已知），14 bit 只对 days 可用；norm（Gram, sx）最敏感。最终描述量对位宽几乎不敏感（8 bit 就够），可以：累加器 16 bit（整数计数特征），finalize 后截到 8–12 bit 送入 calibrator。
- **输出表 int8 无损、int4 掉 0–0.005**；**softmax LUT 无损**。
- target 与 source val：二者的 ΔR² 排序相同，量级相同；val 上的 ΔR² 通常略小（val 的 FP R² 更高，约 0.75–0.82 对 0.49–0.59），所以 target 上的 ΔR² 部分包含了噪声交叉项的放大；`rel_rms` 在两者上是同一量级（例如 ternary：val 22–25%，target 29–35%），是更干净的比较量。注意 val session 是 trunk 的训练 session，且参与过 P1 的早停选择，所以 val 数字偏乐观，只看相对变化。

## 5. A′4：架构级代价（解析；公式来自代码中的实际 tensor 形状，测试中与真实 PopStats tensor 形状和 forward hook 的 MAC 计数逐项核对）

记号：N units，K 个潜变量，D 个行为维度，L 个 bin/trial，B 个标注 trial，H=64，m=8（种子数），dctx=32，Fd = 17+2K+5D（描述量宽度；K=10, D=2 时 47），MAC = 一次乘加。

**累加（流式，每个有效 bin）**：MAC = N²（Gram）+ 3N（3 个 lag 的自相关）+ 5DN（5 个 lag 的行为互协方差）+ 3N（T4 的 φxᵀ）+ 9（φφᵀ）= N² + (6+5D)N + 9；加法 = 8N + 5D + 13。每个 trial 结束时 q_* 一致性累加：N + 2DN + D MAC（及同样数量的除法）。一次校准的累加 = B·(L·[N²+(6+5D)N+9] + N+2DN+D)。
**finalize（一次）**：(2K+1)N² + (K+27)N + 27 MAC（latent 相关两次 N²K、Gram 归一化 N²、N 个 3×3 求解）+ O(N·Fd) 逐元素运算。
**calibrator 前向**：N × (每 unit 的所有 Linear 层 in×out 之和) + pooling。P0：Fd·H + H² + H·K；P4：phi (Fd·H+H²+H²) + val (H·dctx) + rel (6·16+16) + gen ((Fd+dctx+1)·H+H²+H·K) + 打分与路由 2·H·m + μᵢ（m·dctx）；pooling = N·m·dctx（P2：N·dctx）。P4 若第二遍重算 φ（不存 N×H 的激活），φ 的 MAC（每 unit Fd·H+2H²）要再加一遍；两种方案都在 §R.4 的表中。
**写表**：K·N 个字。
**推理 read-in MVM**：每 bin K·N MAC（外加 2N 逐元素的标准化）。
**SRAM（元素数 × 位宽）**：accumulators 共 N² + (22+7D)N + 29 + 6D 项（Gram N²，syx 5DN，sdir 8N，ac 9N，…；其中 17 个是精确计数器；`derived_dir` 时 sphi/sphiphi/sphix[0] 不占寄存器，少 12+N 项），**q_* 的 per-trial scratch（N+DN+D+1）、lag delay line（5N+2D）、two-level 的 staging 拷贝（可选）**，描述量 N·(Fd+6)，表 K·N（双缓冲 ×2），归一化参数 2N，P4 的 online-softmax 状态 m·(dctx+2)，**P4 第二遍的 φ buffer（N×H）或重算**。
**anytime 单 trial 更新**：流式部分 L·[N²+(6+5D)N+9] + (N+2DN+D)；若每个 trial 后刷新表，再加 finalize + 前向 + 写表。

**数值表与要点已改写，见 §R.4（修正后）；旧表在附录 S。**

## 6. 局限与未做

- A′1：只有 2 个 seed、4 个 target（days 2、month 1、year 1，month-O 与 year-O 是同一训练结果），pooling 的正向趋势在噪声底量级，未做更多 seed 的重复；没有做 identity-free + 无标签（nolab）的组合，因此无法区分“B≥8 时 days 的 identity-free 性能来自标签调谐统计还是无标签统计”；只试了 m=8（未重选 m）；identity-free 训练的 best step 更晚，预算翻倍规则触发多次，但最终都收敛。
- A′2：lambda 的 B=1 最优在网格边缘；选择依赖 trunk 的 in-sample source session；year 复用 month 的选择；P0+ui_reg 没有单独选参；ui_reg 与 ui_elec 的配对只有 5 个 draw；没有 ui_prior 的正则版本。
- A′3（旧条目，部分已更新）：激活（ADC/DAC 精度）、偏置、`mu0`、输入标准化常数没有量化；softmax 用批量形式，没有评测 online-softmax 形式；accumulator 的步长是全局每 key 一个，不是每元素；8 bit 累加器没有尝试 per-unit scale、stochastic rounding 或浮点累加（可能挽回）；没有噪声/器件非理想性（RRAM 电导漂移）建模；val session 是 trunk 的训练 session，且参与了 P1 与 QAT 的早停选择（val 行对 QAT 乐观）；ternary 的 QAT 只对 P0 与 P4 做，没有 int4 QAT。
- A′4：解析计数，非硬件测量；latent 累加替代 Gram 的方案没有实现或评测；HD-sEMG 的描述量沿用 000688 的特征族（包括方向调谐特征），D=5 取 Hyser 的 5 指力，实际 sEMG 特征可能不同；HD-sEMG 的 per-add 位宽只测了 Gram/交叉和的误差，没有解码；没有计 trunk 本身的代价。
- 没有推送，没有改动 P1 的 `results/` 内容，也没有触碰 DB6 / sEMG 的代码；审查修正中只读了 Hyser 的缓存特征（R.2.6）。

## 7. 文件与复现

- 本目录：`a3_safe_width.csv`、`a3_cells_by_modelgroup.csv`、`a3_readme_tables.md`、`cost_tables.md`、`hyser_peradd*.csv`、`a2_ext_*`、`superseded_a3_a4_pertrial/`（旧 a3/cost 文件）、`a1_*.csv`（cells、paired、gap、gain、floor、training、flags）、`a2_*`（cells、paired、selection、select_grid）、`a3_*`（cells、by_stratum、qat）、`cost_model.csv`、`cost_trial_stats.json`、`tables_a1_a2_a3.md`、`a*_rows.csv.gz`。
- 原始：`results/population_calib_p2_20261015/{runs,reference,uireg,quant,logs}/`（gitignored）。
- 复现（GPU0，全部幂等可续跑）：`python scripts/run_pc2_queue.py --stage noalign --workers 3`（A′1）；`python scripts/run_pc2_uireg.py select --stratum days|month`，再 `eval --stratum days|month|year [--sel-from month]`（A′2）；`python scripts/run_pc2_quant.py --stage stats --stratum S --workers 8`（CPU，逐 add 统计缓存）再 `python scripts/run_pc2_quant.py --stage eval --stratum S`（GPU；输出在 `fix/quant/`）（A′3），`python scripts/run_pc2_queue.py --stage qat --workers 1` 后 `run_pc2_quant.py --stratum S --qat`（QAT）；`python scripts/pc2_cost_report.py`（A′4，读 `a3_safe_width.csv`）；`python scripts/pc2_hyser_peradd.py`（R.2.6，CPU）；`python scripts/pc2_uireg_ext.py select|eval --stratum S`（R.6）；`python scripts/analyze_pc2.py`（汇总）。

## 附录 S. 被取代（superseded）的旧数字——不要引用

以下数字来自 commit a7ef56a 的 A′3/A′4：accumulator 每个 trial 才取整一次、计数器被当作数据量化、代价模型遗漏若干项。独立审查后已被 §R 取代。原始文件在 `superseded_a3_a4_pertrial/`。

**§0 第 3 点：accumulator 条目（旧）**

   - descriptor accumulators（原始累加量，每个 key 一个 power-of-two 步长，只由 source 拟合）：**16 bit 无损**（|ΔR²|≤0.0006）；**12 bit 在 B≥8 几乎无损（≥−0.003）但 B=1 掉 0.14（target）/0.06（val）**，原因是步长按 B=33 的量程选，B=1 只用到低位；让 block exponent 随 B 变化（再留 2 bit 余量）后 12 bit 在 B=1 只掉 0.001；**8 bit 不可用**（−0.3 到 −0.8，且大量行出现非有限预测，见 §4.2）。12/8 bit 的分组消融：12 bit 时只有 dir 组（`sdir`/`cdir`/`sphi*`）在 B=1 明显掉（target −0.13），其余 ≥−0.006；8 bit 时 norm（`n/sx/gram`）、dir、cross（`sy/syx`）都崩（−0.1 到 −0.9），ac 约 −0.06 到 −0.07，quad 无影响。

**§0 第 3 点：组合条目（旧）**

   - 组合：int8 权重 + 16 bit 累加器与描述量 + int8 表 + LUT：ΔR² 绝对值 ≤0.0006，RMS 0.7–1.1%（P4 B=1 target 为 2.8%）。int4 权重 + 12 bit + int8 表：B≥8 掉 ≤0.004，B=1 掉 0.02–0.07。

**§0 第 4 点：A′4 摘要（旧）**

4. **A′4（代价，解析）**：详见 §5。000688（N=26, K=10, D=2；平均可用 trial 约 220 bin）一次 B=8 校准：累加 1.94 M MAC（每 bin 1101 MAC，Gram 占 676），P0 的 finalize+前向+写表 0.22 M MAC，P4 为 0.66 M MAC；总 2.2 M（P0）/2.6 M（P4）MAC；SRAM（P4，16 bit 累加器、16 bit 描述量、int8 双缓冲表）约 7.8 KB，表本身 260 B（int8）/1040 B（fp32）。HD-sEMG（N=256, K=16, D=5, 每 trial 1250 bin）：每 bin 73.5 k MAC，一次 B=8 校准 0.74 G MAC，其中固定权重部分只占 0.3%（P0）/1.0%（P4）；**Gram 累加器 131 KB（16 bit）/262 KB（fp32）远超过 32 KB 的表预算**，总 SRAM（同上配置）约 212 KB；表本身 4 KB（int8）/16 KB（fp32，双缓冲刚好 32 KB）。结论：在 HD-sEMG 规模，**瓶颈是 N² Gram 累加，不是 calibrator**；换成 N·K 的 latent 累加（解析，未评测）每 bin 可降到 16.1 k MAC，但会改变特征。

**§4.1 表中 accumulator / 组合行（旧，逐 trial 取整模型，计数器被当作数据量化）**

| 配置 | P0 val ΔR² B=1/8/33 | P0 target ΔR² B=1/8/33 | P0 target RMS B=1/8/33 | P4 target ΔR² B=1/8/33 | P4 target RMS B=1/8/33 |
|---|---|---|---|---|---|
| acc16 | −0.0000/+0.0000/−0.0000 | +0.0001/+0.0000/−0.0001 | 0.9/0.5/0.6% | −0.0006/−0.0000/−0.0002 | 2.6/0.6/0.6% |
| acc12 | −0.0628/−0.0014/−0.0012 | −0.1396/−0.0027/+0.0001 | 47.3/8.1/6.2% | −0.1207/−0.0003/+0.0014 | 42.6/6.5/5.0% |
| acc12_bs（B 相关 exponent） | −0.0002/−0.0014/−0.0012 | −0.0007/−0.0027/+0.0001 | 2.6/8.1/6.2% | −0.0014/−0.0003/+0.0014 | 4.2/6.5/5.0% |
| acc8 | −0.614/−0.556/−0.492 | −0.812/−0.707/−0.367 | 119/99/85% | −0.829/−0.618/−0.322 | 121/92/79% |
| acc8_bs | −0.145/−0.556/−0.492 | −0.465/−0.707/−0.367 | 85/99/85% | −0.460/−0.618/−0.322 | 80/92/79% |
| w8+acc16+desc16+tab8（+LUT） | −0.0002/−0.0001/−0.0001 | −0.0000/−0.0000/+0.0001 | 1.1/0.7/0.8% | −0.0006/−0.0001/−0.0004 | 2.8/0.8/0.8% |
| w4+acc12+desc12+tab8（+LUT） | −0.0395/−0.0013/−0.0013 | −0.0738/−0.0039/−0.0010 | 40/10/9% | −0.0471/−0.0007/−0.0008 | 34/8/7% |
| tern+acc16+desc16+tab4（+LUT） | −0.0420/−0.0468/−0.0465 | −0.0477/−0.0553/−0.0631 | 36/32/33% | −0.0306/−0.0376/−0.0410 | 32/30/30% |

**§4.1 分组消融（旧）**

accumulator 的分组消融（12 / 8 bit，P0 target ΔR²，B=1/8/33）：norm（n,sx,gram）−0.006/−0.003/−0.0015 与 −0.80/−0.57/−0.40；ac −0.003/+0.001/+0.002 与 −0.07/−0.07/−0.06；cross −0.0001/0/0 与 −0.64/−0.30/−0.09；dir −0.129/−0.0005/−0.0002 与 −0.79/−0.88/−0.55；quad（P0 不读 r，无影响；P4：−0.0015/+0.0001/0 与 −0.001/0/0）。

**§4.2 非有限预测说明（旧，原因解释有误）**

**非有限预测**：acc8 的预测出现 NaN/Inf（`finite` 标志；P0 target B=1：64 行中 48 行；B=8 与 B=33 分别为 30、44 行，见 `a3_cells.csv` 的 `nonfinite` 列），ΔR² 的均值只在有限行上计算，因此 acc8 的 ΔR² 被低估（真实更差）。原因在于 σ² = Σx²/n − μ² 的相消：8 bit 的 Gram/sx 使方差为负或 0。

**§5 数值表与 SRAM 表（旧）**

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


**§5 要点（旧）**

**要点。** (i) 000688 规模：整个校准是 MFLOP 量级，SRAM <10 KB，表仅 260 B，符合 ≤32 KB 的约束；P4 的 pooling 与 P0 相比约 3 倍 calibrator MAC，对应 P1/P2 没有稳定收益（A′1 只有弱正向），代价不划算。(ii) HD-sEMG 规模：累加是主要代价（每 trial 92 M MAC，约 3.7 MMAC/s，是推理 read-in 4,096 MAC/bin 的 18 倍），固定权重的 calibrator 只占 0.3–1%，所以 RRAM 的受益点在推理 read-in 和 calibrator 前向，而不是校准的主要算术；**Gram 累加器（131 KB@16 bit）远大于表预算 32 KB**。可选缓解（解析，**未实现、未评测**，会改变特征）：用 latent 累加 2KN = 8,192 MAC/bin 替换 N² Gram 项（总计 16.1 k MAC/bin，约 4.6 倍减少）；利用 Gram 对称（每 bin MAC −44%，Gram 存储 −50%）；累加器只需 16 bit（§4）；存储可与推理阶段的缓冲复用（校准期间推理不运行）。(iii) 这些是解析计数，不是硬件测量；000688 的 L 取两个 session（0616、0626）可用 trial 的平均长度 220 bin，HD 取 25 s×50 Hz。

**§1 A′3 accumulators 设置（旧）**

- accumulators：每个 PopStats key（`n, sx, gram, ac_*, sy, syx, yx_n, sdir, cdir, sphi*, q_*`）有符号定点，位宽 16/12/8，一个 power-of-two 步长（取 source 训练 session 上 B=33 的最大 |值|所需的最小步长，5 个 session × 5 个 draw = 25 个 episode 内取最大），每加入一个 trial 后 round-to-nearest 并饱和。`*_bs` 变体：步长再乘 2^(ceil(log2(B/33))+2)（上限 1），即 block exponent 随 B 变化并保留 2 bit 余量；第一版没有余量时，长 trial 会饱和（B=1 的 acc16_bs 反而更差），已改并重跑。“`accK_<group>`”只量化一组 key（norm = n/sx/gram；ac；cross = sy/syx/yx_n；dir = sdir/cdir/sphi*；quad = q_*）。

