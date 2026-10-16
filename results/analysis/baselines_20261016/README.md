# 已发表方法 baseline 与可部署行的同协议对比（DANDI 000688 sub-M，DAC2027）

日期：2026-10-16。执行者：baseline 工作线（单卡 cuda:0，最多 2 个并行 GPU job；NoMAD 全程 CPU）。规格：MZ 的任务说明（`docs/paper_mainline_proposal_20261016.md`、`docs/feasibility_p1_20261015.md` §0、P1/P2/P3 README 的协议）。

代码（全部新增，未改动 P1–P3 的语义）：`ssm_decode/published_baselines.py`（FA/Procrustes/Stabilizer、PCA+CCA）、`scripts/run_baselines_readin.py`（source 选择 / 评测 / 桥自检）、`scripts/run_baselines_apst.py`（APST 适配器）、`scripts/apst_2014_train.py`（APST 在 2014 roster 上按作者冻结配方重训）、`scripts/run_baselines_nomad_tf.py` + `nomad_prep.py` + `run_baselines_nomad_score.py`（NoMAD）、`scripts/run_baselines_native_p.py`（P-prior / P0 在 native-query 子集上重算）、`scripts/analyze_baselines.py`（汇总）、`tests/test_published_baselines.py`（5 个单元测试，通过）。
原始 rows：`results/baselines_20261016/`（gitignored；`eval/*.jsonl`、`select/*.jsonl`、`apst2014/`、`nomad/`、`logs/`）。本目录是汇总：`main_table.md`（主表）、`main_table_native.md`、`cells.csv`、`cells_native.csv`、`paired_vs_p0*.csv`、`rows_long.csv.gz`（全部逐 draw 的行）、`method_registry.csv`（每个方法用了什么 target 数据、是否反传）、`selection_readin.csv`、`bridge_ceiling.csv`、`unavailable_cells.csv`、`nomad_per_target_seed.csv`、`native_p_reproduction_check.csv`。

## 0. 结论摘要

先说明读表口径：**P 行与 baseline 行是同一批 `(target, B, draw)` 子集、同一个冻结 backbone、同一个连续因果解码、同一个物理单位 R²**，逐行配对（§1）。表中“target data used / target backprop”两列逐行写明各方法用了什么（§3）。下面的 CI 只含 draw 方差（8 个 draw），月/年各只有 1 个 target，共享 trunk，不是独立重复。

1. **在 H4 合规（只用 B 个校准 trial）的设定下，没有任何一个已发表 baseline 在我们的 query 口径下追上 P0，更不用说 R1-stream / R3。** B=8 时 P0 / 最好的已发表 H4 合规行：days 0.723 / 0.516（APST-FULL，segment 选择见过 target；source-only 的 last segment 为 0.500），month 0.565 / 0.394（岭正则 CCA，我们加的变体；plain Gallego 0.350，Stabilizer 0.214），year 0.256 / 0.232（岭正则 CCA；plain 0.193，APST 0.150）。配对差（P0 − baseline，B=8）：Stabilizer −0.32 / −0.35 / −0.24（days/month/year），Gallego-style CCA −0.33 / −0.23 / −0.07，APST-FULL −0.22 / −0.29 / −0.11。
2. **小 B 差距最大。** B=1 时 P0 为 0.647 / 0.466 / 0.198，而 Stabilizer 为 0.281 / 0.076 / −0.065，PCA+CCA 为 0.04–0.09 / −0.03–0.07 / ≈0（B=1 只有一个方向的条件平均，CCA 欠定）；APST-FULL 在 B=1 **不可用**（MOVE-T4 需要 ≥3 个方向非共线的有标签 trial；month/year 的 B=4 也有 3/8、1/8 个 draw 不可用，见 `unavailable_cells.csv`），APST ACT-only（不用标签）days B=1 为 0.026。P0 在 B=1 的优势来自 electrode-aligned prior（P-prior 本身已有 0.333 / 0.315 / 0.145），该优势依赖 `unit_to_electrode`（P1 审查 §0.1）。
3. **大 B 下差距缩小，且有两处例外必须如实写出**：
   - **year（20150511，1 个 target，离 source 1 年多）**：label-using 的 PCA+CCA 在 B=33 追平并略超 P0（岭 CCA 0.360 对 P0 0.315，配对差 +0.044 [±0.013]；在 native 子集上 0.498 对 0.415，+0.083 [±0.022]），但仍低于 R1-stream（0.376）和 R3（0.435）。B≤8 不高于 P0（B=8 配对差 −0.023 [±0.041]，持平；B≤4 明显更低）。
   - **days（20150616/17，APST 自己的 source/dev 设定）**：在 APST 的 native query 口径（窗口完全落在 trial 内）下，APST-FULL 在 B=16/33 **高于 P0**（0.789 / 0.814 对 0.737 / 0.744，配对差 +0.052 / +0.070，CI ±0.012；dev 选择的 segment 为 0.799 / 0.822），在 B=8 与 P0 持平（0.700 对 0.728），B=4 仍低很多（0.39 对 0.71）。在 year 的 native 口径下，2014 重训的 APST-FULL（source-OOS 的 0307 上选的 segment）与 P0 持平（B=8/16/33：−0.009 / +0.028 / +0.018，CI ±0.04–0.07）；month 上 APST 在两种口径下都低于 P0。**days 上这是 MZ 自己的方法在稳定植入的大 B 区间占优的证据**，需要在论文里如实写：APST 是 350 万参数的 set-temporal transformer（非固定权重 trunk 方案），其优势不在 B 小的区间。
4. **口径问题对 APST 影响很大（§2.3）**：我们和 P 行对 trial 内全部 bin 打分，APST 的训练/评测只覆盖 50-bin 窗口完全在 trial 内的 bin；同一批预测在 native 子集上 days B=33 从 0.590 升到 0.814。用作者协议（前 M 个 trial + Q50）复现作者 dev 数 0.855956 一致，说明适配器本身没有问题。因此主表给“我们的口径”，native 表给“APST 的口径”，两张表里所有能重算的方法（P-prior、P0、Stabilizer、CCA、NoMAD、APST）都在两个口径下报告；R1-stream / R3 的逐 bin 预测没有保存，不在 native 表里。
5. **Stabilizer 在 000688 上很弱，且弱的原因是协议性的**：它假设固定的通道集合，而 sub-M 每天重新分选 unit，只能靠“同电极 + 发放率排序”配对 13–15/17（days）、19/26（month）、14/24（year）个 unit。对照 `stab_a_noalign`（不做 Procrustes 旋转）是 −0.06…−0.16（days），所以旋转步骤确实在工作，但上限被桥（FA 潜变量→AP 潜变量的 K×K 最小二乘）和 unit 对应关系限制：用整段 target recording 的无标签数据（`stab_b`，论文自己的数据量、违反 H4）也只有 0.348 / 0.257 / 0.029，与 B 个 trial 的 `stab_a` 同量级；**数据量不是它的瓶颈**。
6. **NoMAD（非可部署、违反 H4、target 端有梯度）在 3/4 个 target 上对齐失败，且 3 个 seed 一致**：days 20150616 为 +0.48（native +0.52），20150617 为 −0.70…−0.78，month −0.88…−0.92，year −0.59…−0.68（`nomad_per_target_seed.csv`）。pipeline 自检：把 baseline day 自己作为“target”走一遍对齐网络得到 0.647，day-0 decoder 的训练集内 R² 为 0.71（days）/ 0.89（2014），所以失败是对齐本身（几个 sorted-unit 的 session、unit 集合每天变化，且 alignment network 的训练在 NLL 退火结束后马上早停，约 110–130 epoch），不是数据管线 bug。这个结果只说明 NoMAD 在这种“每天重新分选的少量 sorted unit”数据上的表现，不能推广到论文自己的数据。
7. **微调参照行（已有，复用）**：`ui_reg`（从 prior 起、L2 到 prior、反传穿过 trunk）与 P0+ui_reg 的数字见主表末尾；P0+ui_reg 在 days 与 P0 持平，month/year 在 B≥4 高于 P0（这是 P3 的 gap），R3 在 month/year 恢复其中的大部分。**在我们的 query 口径下，所有已发表 H4 合规 baseline 在每个 (stratum, B) 都低于 ui_reg 类参照（后者反传穿过 trunk，不可部署）。**
8. **没有运行 / 没有做的内容**：APST ACT-only 在 2014 roster（month/year）上**没有训练**（FULL 才训练；ACT 只在 days 上有）；R1-stream / R3 没有 native-query 版本；APST 的 first-M 协议下的 B=33 与我们 B=33 的 draw 不同（所以不能用作者的 0.864 与我们的 0.590/0.814 直接比，只作协议说明）。其余项目（Stabilizer、Gallego、APST days / month / year、NoMAD 4 个 target × 3 seed）全部跑完。
9. **自检（均通过）**：P-prior/P0 在本评测里重新前向的 query R² 与 P1 存档行逐行一致（最大差 5.8e-5，`native_p_reproduction_check.csv`）；APST 适配器的受限预测与作者 `predict_record` 在 6000-bin 前缀上最大差 8.6e-6；作者 first-M 协议复现作者 dev 数；所有 `check_physical_units` 通过；5 个单元测试通过。没有读 final session，没有读 `results/official_evalai/`。

## 1. 协议（与 P1/P3 的 P 行逐项对齐）

| 项 | 内容 |
|---|---|
| 数据 | DANDI 000688 sub-M CO，三个 stratum：days（targets 20150616 / 20150617；backbone `ap_pool_x8`）、month（20140626）、year（20150511）（后两者各一个 target，共用同一批 6 个 2014 source session，见 P1 README）。**没有打开任何 final session（0623/0625/0626），没有读 `results/official_evalai/`。** |
| backbone | 冻结的 K=10 AP backbone（`D1_POOLS[...]['ap']`），解码 = `trunk(z)`，`z = T·((x−mean)/std)+c`；P-prior / P0 / R1-stream / R3 / ui_reg 的 `T` 是 K×N read-in 表（c=0），baseline 的 `T,c` 是它们各自方法给出的仿射 read-in。 |
| 校准子集 | **完全复用** P1/P3 的 `draw_trials(target, B, draw, n_trials)`（`zlib.crc32(f"{target}-{B}-{draw}")` 播种的 `rng.choice`，不放回）。B ∈ {1, 4, 8, 16, 33}（P1/P3 另有 B=2，这里不报），draw = 0…7（8 个）。每个 baseline 的 `(target,B,draw)` 行与 P 行逐行配对。 |
| 解码 | 整段 recording 连续因果前向一次（不在 trial 边界重置 state，规则 7）；query = usable trial 内、不在校准 trial 中的全部 eval-valid bin；R² = 物理单位、方差加权（`check_physical_units` 全部通过）。APST 用它自己的 50-bin 因果窗 + full-clock EMA（α=1/3），query bin 与上面完全相同（仅去掉 recording 最前 49 bin）。 |
| CI | 对每个 (方法, stratum, B)：先对每个 draw 取 target 均值（P 行已按 seed 平均），再对 8 个 draw 做 t 区间 `mean ± t(0.975,7)·sd/√8`。**只含 draw 方差**，不含 target / trunk / source 方差；days 有 2 个 target，month 与 year 各 1 个且共享 trunk，不是独立重复。baseline 的闭式方法没有训练 seed（FA 的 5 次重启固定种子 0），所以没有 seed 维度。`*_b` 行（对齐不依赖 B）在不同 B、draw 之间只有 query mask 不同，CI 因此极窄，不代表不确定性。 |
| 选择规则 | 所有 baseline 超参数只在 **source** 上选：pseudo-target = backbone 的 6 个 source session 中除 baseline day 外的 5 个（B∈{4,16}，每个 2 个 draw，种子 `val-<session>`，与 P1 的 source val 同一套）；打分 = pseudo-target 的 query R²（经冻结 trunk）。pseudo-target 是 trunk 的训练 session，因此这个分数只用于**排序**，不是 target 预期。`results/analysis/baselines_20261016/selection_readin.csv` 给出全部网格。 |
| baseline day | 所有 read-in 对齐方法（Stabilizer、CCA）与 NoMAD 的“第 0 天” = backbone 的**最近一个 source session**（days：20150615；month/year：20140306），按日期的固定规则，不看 target。 |
| 约束 | H1：trunk 全程冻结（baseline 不反传 trunk，也不改 trunk）；H2：除 NoMAD 的 alignment network 与 ui_reg 参照行外，target 端没有任何梯度；H4：标成 `(a)` 的行只用 B 个校准 trial 的数据；`(b)` 行用整段 target recording 的数据，**违反 H4**，只作“论文自己的数据量”参照，不可部署。 |

**两个必须先说明的比较口径问题：**

1. **P 行与 baseline 行使用的 target 数据不同**：P-prior 用 B 个 trial 的神经数据（只估输入 mean/std）；P0 / R1-stream / R3 用 B 个 trial 的神经数据 + 行为标签（cross-covariance、方向调谐等描述量）；Stabilizer 只用神经数据，不用任何标签；CCA 与 APST-FULL 用 B 个 trial 的神经数据 + 每个 trial 的 reach 方向标签（任务元数据，不是行为轨迹）；APST ACT-only 只用神经数据。表中“target data used”列逐行写明。
2. **P 行依赖电极映射 `unit_to_electrode`**（P1 审查 §0.1.5：打乱电极后 P0 B=8 从 0.73 掉到 0.59，随机分配 0.21）；Stabilizer 也用它（匹配 stable channels）；CCA、APST、NoMAD 不用。

## 2. 各 baseline：引用、实现与简化、超参数、target 数据、是否反传

### 2.1 Stabilizer（Degenhart et al. 2020）

- **引用（已核实）**：Degenhart AD, Bishop WE, Oby ER, Tyler-Kabara EC, Chase SM, Batista AP, Yu BM. *Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity.* Nature Biomedical Engineering 4(7):672–685 (2020), doi:10.1038/s41551-020-0542-9（Europe PMC 记录 PMC7822646；Nature 页面）。
- **实现来源**：作者的 MATLAB 配套代码（github.com/alandegenhart/stabilizedbci，`fitBaseStabilizer.m / updateStabilizer.m / alignLoadingMatrices.m / identifyStableLoadingRows.m / learnOptimalOrthonormalTransformation.m / getStabilizatonMatrices.m / stabilizationExample.m`，逐个读过），移植到 numpy（`ssm_decode/published_baselines.py`）：(i) baseline day 与新一天各拟合 FA（EM，5 次随机重启取最优似然，私有方差下限）；(ii) 用 `identifyStableLoadingRows` 的迭代 Procrustes 丢行法在 `n_stable` 个 stable channel 上求正交变换 `W`（行范数阈值 `ALIGN_TH=0.01`），把新一天的载荷旋转到 baseline 的 FA 空间 `C_new ← C_new Wᵀ`；(iii) 潜变量 `l = βy+o`，`β = Cᵀ(CCᵀ+Ψ)⁻¹`。label-free。
- **与原法不同之处（全部是为了接到冻结的 AP trunk 上）**：
  1. 原文的 decoder 是在 baseline 日 FA 潜变量上训练的 Kalman filter。这里 decoder = 冻结 trunk，FA 潜变量经一个 **K×K 最小二乘桥 `A`（+偏置）** 映射到 AP 潜变量；`A` 只在 baseline source session 上拟合（该 session 的 AP 潜变量 `W_ref·x` 是已知的），不看 target。桥的上限（baseline 日自己走一遍）见 §3.3：经 FA 桥 days 0.81 / month 0.865（直接用 source 表为 0.88 / 0.92）。
  2. 原文的通道是固定的电极；000688 的通道是每天重新分选的 unit。这里 stable-channel 的对应关系用 **同一电极上的 unit 按平均发放率排序配对**（label-free；days 目标日 13–15/17 个 unit 能配上，month 19/26，year 14/24），配不上的 unit 视为 unstable。
  3. FA 的输入先按各自拟合数据做 per-unit z-score，于是作者的 `MIN_PRIV_VAR=0.1`（原始计数单位）变成“单位方差的 10%”；EM 迭代上限 400、相对容差 1e-5（作者 1e5 次/1e-5）。
  4. 原文在线持续更新 stabilizer；这里只做一次更新（离线）。FA 维数 K=10（= backbone K，也是作者示例的值）。
- **超参数（只在 source 选）**：`MIN_PRIV_VAR ∈ {0.01, 0.1}`，`ALIGN_N = {0.5, 0.75, 1.0} ×（配对 unit 数）`；选出的组合见 `selection_readin.csv`（days：floor 0.01、frac 1.0；month/year：floor 0.01、frac 0.75）。
- **target 数据**：(a) B 个校准 trial 的神经计数（无标签）；(b) 整段 target recording 全部 usable trial 的神经计数（无标签，包括 query bin 的输入，transductive，违反 H4）。**target 反传：无**（EM + SVD）。
- 对照 `stab_a_noalign`：同样的 target FA 但不做 Procrustes 旋转（说明旋转步骤确实在起作用）。

### 2.2 PCA + CCA 潜动力学对齐（Gallego et al. 2020）

- **引用（已核实）**：Gallego JA et al. *Long-term stability of cortical population dynamics underlying consistent behavior.* Nature Neuroscience 23(2):260–270 (2020), doi:10.1038/s41593-019-0555-4（PubMed 31907438；方法描述以 bioRxiv 10.1101/447441v3 全文为准，已读）。
- **实现**：与论文 Methods 一致的部分——M1 潜变量维数 m=10（= K）；对计数做 √ 变换；PCA 取前 m 个主成分；用 CCA（QR + SVD）找两天潜动力学之间使相关最大的线性变换，把新一天的潜变量映到第 1 天（baseline day）的坐标；再用第 1 天的 decoder。
- **与论文不同之处**：
  1. 论文用整日（按 target 匹配 trial 数、按时间点对齐的）单 trial 潜动力学做 CCA。这里按任务要求只能用 B 个有标签 trial：每个 trial 按时间归一化到 20 个相位 bin（分段平均），**按 reach 方向（NWB trials 表的 `target_dir`，8 个方向）取条件平均**，在两天都出现的方向上拼接成 CCA 样本；B 小时只有 1–几个方向，样本数 = 方向数×20，**CCA 严重欠定**（这是该方法在小 B 下的真实困难，不是 bug）。
  2. 论文用 50 ms 高斯平滑；这里没有非因果平滑（相位分段平均相当于平滑），解码时的 read-in 是逐 bin 的因果线性映射。
  3. decoder = 冻结 trunk，经 baseline 日的 K×K 最小二乘桥 `A`（PCA 潜变量 → AP 潜变量）。
  4. 论文没有正则；这里除了 `plain`（岭参数 0，忠实版）外，另给一个 **岭正则 CCA 变体**（我们加的，在 source 上选岭系数，标成 variant），因为 plain CCA 在 B 小时数值上近乎奇异。
- **超参数（只在 source 选）**：输入变换 ∈ {raw, √}（√ 是论文的预处理，作为主行 `Gallego-style`；raw 作对照）；岭系数 ∈ {0.01,…,100}（相对协方差迹）；选出的岭系数：raw 输入 days 3、month/year 10；√ 输入 days 1、month/year 10（全部网格见 `selection_readin.csv`）。
- **target 数据**：(a) B 个校准 trial 的神经计数 + 每个 trial 的 reach 方向标签（不使用行为轨迹）；(b) 整段 target recording 全部 usable trial 的神经计数 + 方向标签（违反 H4，且 query trial 的方向标签也被用到）。**target 反传：无**（SVD/特征分解）。

### 2.3 APST（MZ 自己的方法）

- **引用**：Zhang X, Mo H, Wen P, Liang S, Yang J, Zeng Y, Wang Z, Wang H. *Association profile conditioning in a set-temporal transformer for cross-session intracortical motor decoding.* arXiv:2609.39080 (2026)。**此条目取自本地 `APST/README.md` 的 BibTeX，没有联网核实**（其余三篇的引用已联网核实）。
- **代码**：`/home/xinyuan/Work_host/APST/src/apst/dandi_subm`（作者代码，**未改动**）+ `APST/docs/dandi_subm.md` 与 `examples/dandi_subm/budget/evaluate.py` 的 calibration-budget 评估。APST 的“校准”是前向、闭式的：activity encoder（E0）对 B 个校准 trial 重采样到 100 bin 的神经活动做池化；APST-FULL 另用 B 个 trial 的 MOVE 窗（go cue 后 0.1–0.6 s）计数和 reach 方向拟合 MOVE-T4 association profile（OLS 余弦调谐，需要 ≥3 个方向非共线的 trial，否则按作者协议 `UNAVAILABLE_RANK_LT3`，B=1 总是不可用）；**target 端没有任何梯度**。ACT-only 变体不用方向标签。
- **本评测相对作者协议的唯一改动**：(1) 校准 trial 不是“前 M 个合法 trial”，而是我们的 `draw_trials` 的随机 B-子集（只保留 APST 合法的 trial：rewarded、≥50 bin、go cue 有限且 MOVE 窗在 trial 内；本评测中所有 draw 的 B 个 trial 全部合法）；(2) query 不是作者的 Q50（第 51 个合法 trial 之后、窗口完全在 trial 内的 bin），而是和 P 行相同的“非校准 usable trial 内全部 eval-valid bin”（仅去掉 recording 最前 49 个 bin）。预测仍是 APST 自己的 50-bin 因果窗 + full-clock EMA(α=1/3)；`run_baselines_apst.py` 里的预测函数只把 full-clock 推理限制在 query bin 及其前 16 个 bin（EMA 记忆 (1/3)^16≈2e-8），与作者的 `predict_record` 在 6000-bin 前缀上逐元素一致（最大差 8.6e-6），并在作者自己的“前 M 个 trial + Q50”协议下复现了作者的 dev 数（0616，M=8：0.855956 对 0.855956）。
- **checkpoint 与选择规则（重点）**：
  - days：作者发布的 sub-M 2015 campaign 的 `sua_dual_site_F0_s42` / `sua_act_only_F0_s42`（6 个 2015 source = 我们 days 的 source，**同一批 session**）。作者的选择规则是“在 dev=0616/0617 上 earliest-max”（= 我们的 days target），**对我们是在 target 上选**，所以主行用 **最后一段（segment_24，不做选择，只靠 source 训练的固定 24×3165 更新计划）**；dev 选出的段（FULL=segment_09，ACT=segment_08）单独标成“选择见过 target”的乐观参照行。
  - month / year：作者没有 2014 的 checkpoint；用作者**冻结的配方**（24×3165 更新、seed 42、F0+FiLM、MOVE-T4 carrier）在我们 month/year 的 6 个 2014 source 上重训（`scripts/apst_2014_train.py` 只覆盖 `protocol.TRAIN_SESSIONS/DEV_SESSIONS` 两个常量；dev = 20140307，介于 source 与两个 target 之间，既不是 target 也不参与评估），同样取最后一段；另给“在 source-OOS 的 20140307 上 earliest-max 选出的 segment”（FULL：segment_08，0307 上 dev R² 0.798，而最后一段 segment_24 为 0.744；这是**只看 source 侧**的选择，因此对 2014 是合规的）。重训 FULL 的 concat 预训练 + dual_site 训练各约 1.5–2.7 GPU 小时（与评测并行，单卡共享）。**ACT-only 在 2014 上没有训练**（不影响 FULL 行；理由：单卡 GPU 时间；它是 label-free 的消融，已在 days 上给出）。
- **超参数**：全部是作者冻结的配方，没有任何我们调的参数，没有在 target query 上调任何东西。
- **target 数据**：APST-FULL：B 个校准 trial 的神经活动 + 每 trial 的 reach 方向；ACT-only：只有神经活动。**target 反传：无。**

#### APST 的 query 口径问题（影响解读）

APST 的训练和评测只看“50-bin 窗口完全落在 trial 内”的 bin（Q50）。我们的协议和 P 行一样对 trial 内**全部** bin 打分，trial 起始的前 49 个 bin 的 50-bin 窗口会跨过 trial 边界（含上一个 trial 的尾部和 ITI），对 APST 是分布外输入。因此 APST 在“我们协议”下的 R² 明显低于它自己的协议。为了公平，我们**同时**报告：(i) 主表（我们的协议，全部 query bin）；(ii) “native-query 子集”表：只在窗口完全落在 trial 内的 query bin 上打分，**所有能重算的方法（P-prior、P0、Stabilizer、CCA、NoMAD、APST）都在这个子集上重算**（R1-stream / R3 的逐 bin 预测没有保存，不在 native 表中）。

### 2.4 NoMAD（Karpowicz et al.）——可选参照，**不可部署 / 违反 H4**

- **引用（已核实）**：Karpowicz BM, Ali YH, Wimalasena LN, Sedler AR, Keshtkaran MR, Bodkin K, Ma X, Rubin DB, Williams ZM, Cash SS, Hochberg LR, Miller LE, Pandarinath C. *Stabilizing brain-computer interfaces through alignment of latent dynamics.* Nature Communications 16:4662 (2025)；前身为 bioRxiv 10.1101/2022.04.06.487388（2022，本地仓库 README 引用的版本）。
- **代码**：`/home/xinyuan/Work_host/nomad`（作者的 `lfads_tf2` + `nomad`，库代码**未改**）。仓库固定 TF 2.0 / Python 3.7 / CUDA 10，RTX 3090 不能用，所以在新环境 `/mnt/data/work_host/envs/nomad_tf26`（Python 3.8、TF 2.6.0、tfp 0.12.2、tfa 0.14.0、pandas 1.0.5、numpy 1.19.5；TF 2.15 的第一次尝试在 `tf.function` 的 input_signature 处失败，放弃）里**在 CPU 上**运行（没有占用 GPU）。TF 2.0→2.6 需要 6 处运行时兼容补丁，全部写在 `scripts/run_baselines_nomad_tf.py` 的 monkeypatch 里：git 仓库桩（本地副本不是 git 仓库）、`ClippedGRUCell` 序列化（仅 TF≥2.10 需要）、autograph 对 `train_df[...]=` 的处理、`AlignLFADS._graph_step` 的 input_signature 多了一个 spec（TF 2.0 容忍）、`np.zeros(tf.shape(...))`→`tf.zeros`、控制器输出先验的 `tf.map_fn` 改成逐步展开的循环（**同一个 AR(1) 对数概率公式**，只是因为 map_fn 在 TF 2.6 的对齐图里报 TensorList 形状错误）。
- **流程（对应作者 demo）**：(1) 第 0 天 = baseline day（与 Stabilizer/CCA 相同的 source session）：在其连续 recording 上切 30-bin 窗口（步长 24，每 5 个窗口取 1 个做验证）训练 `BehaviorLFADS`（脉冲 + 标准化速度的行为头；demo 的网络规模：GEN/CON/IC 100、CO 4、FAC 30、ENC_INPUT 50，早停 patience 10）；(2) decoder = 对第 0 天因果 `gen_states` 的 3 阶滞后岭回归（demo 的 `fit_and_eval_decoder`，α 网格交叉验证）；(3) 每个 target：冻结 LFADS，训练 alignment network（FFN-KL，隐层 50，`gen_states` 上的 KL + Poisson NLL，demo 的 nomad.yaml，patience 10），**训练数据 = 该 target session 的全部无标签脉冲窗口**；(4) 对整段 target recording 做 30-bin 因果滑窗（步长 1）取每个窗口最后一个 bin 的 `gen_states`，过第 0 天 decoder 得速度，转回物理单位；(5) 在与其它方法相同的 `(target,B,draw)` query mask 上打分。
- **与 demo 的差别**：DATA_DIM = baseline day 的 unit 数，target 的 unit 零填充到该维数（unit 数不同，没有 demo 里固定的 100 通道）；per-unit z-score 的 `normalization.h5` 由本数据自行生成；LFADS 训练用 graph 模式（demo 在 notebook 里强制 eager）。
- **超参数**：全部取作者 demo 默认，没有调。每个 target 训练 3 个 alignment seed（seed 0 为未显式设种的第一次运行，seed 1、2 显式设 `tf.random.set_seed`）；第 0 天 LFADS 与 decoder 每个 baseline day 只训练一次。
- **target 数据**：整段 target 的无标签脉冲（包括 query bin 的输入，transductive）；**target 反传：有**（alignment network 与 `lowd_readin / generator / rate_linear / fac_linear` 被训练）。因此是 **非可部署、违反 H4** 的参照行；对齐不依赖 B，行内不同 B/draw 之间只有 query mask 不同。

### 2.5 微调参照行（复用，不重算）

`ui_reg`（从 electrode-aligned prior 起、L2 拉回 prior 的逐 target 梯度拟合，反传穿过冻结 trunk；P2 A′2）、`P0 + ui_reg`（P2 规则，从 P0 的表起）以及 month / year 的 like-for-like 版本（P3 R.12，`fix/uiref`）直接取自 `scripts/analyze_pc3_fix.py::reference_rows()`（seed 按 draw 平均）。它们使用相同的 `(target,B,draw)`、相同的 B-trial 归一化、相同的 query 与 R²。**target 端有梯度（穿过 trunk），不可部署**，只是梯度参照。R1-stream / R3 取自 P3 R.12 的 headline 选择集（`analyze_pc3_fix.candidate_rows`，seed 平均；days：IS；month：因果 OOS；year：OOS），均无 trunk 反传。

## 3. 结果

### 3.1 主表：query R²（物理单位），我们的 query 口径（trial 内全部 eval-valid bin）

读法：每格 `mean ± 95% CI`，对 8 个 draw 的 target 均值做 t 区间；`n/a` = 该格没有可用 draw（APST-FULL 需要 ≥3 个方向非共线的有标签 trial）；`(n=k)` = 只有 k 个 draw 可用。`(a)` = 只用 B 个校准 trial（H4 合规）；`(b)` = 用整段 target recording，**违反 H4，不可部署**，只表示“该方法在论文自己的数据量下”的结果。“published-variant”= 我们加的正则/输入变换变体，不是原文方法。P 行：P-prior / P0 为 P1（pool O，2 个 seed 按 draw 平均）；R1-stream / R3 按 P3 R.12 的 headline 选择集（days：IS；month：因果 OOS `oosc`；year：OOS）；ui_reg / P0+ui_reg 为 P2/P3 的行（`like-for-like` 行仅 month/year 有）。

**NoMAD 行说明**：days 行是两个 target × 3 个 seed 的平均（20150616 约 +0.48，20150617 约 −0.75），逐 target 见 §3.3 (ii)；NoMAD、`(b)` 行在不同 B 上几乎相同，因为对齐不依赖 B。**APST 行说明**：days 的 `dev-selected` 行是作者在我们的 days target 上选的 segment（乐观）；month/year 的 `source-only` 行是 2014 重训后在 source-OOS 的 20140307 上选的 segment；`last segment` 行在三个 stratum 上都是不做选择的固定计划终点。

#### days

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.333±0.025 | 0.360±0.018 | 0.360±0.023 | 0.362±0.011 | 0.361±0.012 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.647±0.021 | 0.708±0.009 | 0.723±0.009 | 0.732±0.005 | 0.738±0.005 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.648±0.021 | 0.710±0.009 | 0.723±0.010 | 0.732±0.005 | 0.740±0.005 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.651±0.020 | 0.714±0.009 | 0.728±0.010 | 0.737±0.006 | 0.747±0.006 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.281±0.089 | 0.341±0.098 | 0.400±0.060 | 0.383±0.056 | 0.435±0.060 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.119±0.101 | -0.150±0.119 | -0.156±0.178 | -0.060±0.090 | -0.085±0.112 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.348±0.001 | 0.348±0.003 | 0.348±0.002 | 0.348±0.004 | 0.346±0.005 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.037±0.059 | 0.223±0.071 | 0.391±0.080 | 0.465±0.040 | 0.525±0.017 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.092±0.045 | 0.284±0.054 | 0.423±0.054 | 0.487±0.035 | 0.559±0.015 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.036±0.077 | 0.242±0.086 | 0.433±0.078 | 0.500±0.039 | 0.558±0.013 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.092±0.052 | 0.317±0.055 | 0.472±0.036 | 0.511±0.033 | 0.583±0.014 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.591±0.001 | 0.593±0.003 | 0.590±0.001 | 0.589±0.004 | 0.592±0.005 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.650±0.001 | 0.651±0.003 | 0.649±0.001 | 0.648±0.004 | 0.652±0.006 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.605±0.001 | 0.606±0.003 | 0.604±0.002 | 0.601±0.003 | 0.606±0.004 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.661±0.001 | 0.662±0.003 | 0.660±0.001 | 0.657±0.003 | 0.663±0.006 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.279±0.118 | 0.500±0.059 | 0.568±0.011 | 0.590±0.009 |
| APST ACT-only (last segment, source-only rule) | B trials: neural only (no labels) | no | 0.026±0.068 | 0.178±0.074 | 0.340±0.070 | 0.393±0.083 | 0.437±0.103 |
| APST-FULL (authors' dev-selected segment_09; dev = our days targets, so optimistic) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.311±0.119 | 0.516±0.043 | 0.567±0.006 | 0.588±0.009 |
| APST ACT-only (authors' dev-selected segment_08; dev = our days targets, so optimistic) | B trials: neural only (no labels) | no | 0.041±0.076 | 0.196±0.086 | 0.338±0.080 | 0.396±0.093 | 0.429±0.110 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.130±0.002 | -0.125±0.008 | -0.131±0.006 | -0.130±0.013 | -0.133±0.013 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.378±0.028 | 0.527±0.039 | 0.607±0.038 | 0.661±0.020 | 0.704±0.015 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.654±0.019 | 0.701±0.015 | 0.720±0.015 | 0.730±0.012 | 0.746±0.008 |

#### month

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.315±0.046 | 0.319±0.021 | 0.332±0.010 | 0.328±0.007 | 0.333±0.004 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.466±0.034 | 0.535±0.014 | 0.565±0.011 | 0.558±0.007 | 0.566±0.006 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.472±0.037 | 0.542±0.014 | 0.576±0.011 | 0.588±0.007 | 0.606±0.008 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.492±0.048 | 0.591±0.028 | 0.630±0.011 | 0.652±0.014 | 0.679±0.007 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.076±0.076 | 0.235±0.037 | 0.214±0.073 | 0.247±0.042 | 0.290±0.022 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.075±0.045 | -0.120±0.089 | -0.149±0.139 | -0.090±0.183 | -0.151±0.154 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.257±0.000 | 0.257±0.001 | 0.258±0.001 | 0.258±0.001 | 0.258±0.001 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.030±0.165 | 0.193±0.101 | 0.333±0.046 | 0.388±0.059 | 0.423±0.051 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.063±0.051 | 0.281±0.075 | 0.384±0.034 | 0.435±0.043 | 0.497±0.036 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.003±0.108 | 0.232±0.114 | 0.350±0.035 | 0.406±0.054 | 0.444±0.050 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.069±0.062 | 0.317±0.075 | 0.394±0.043 | 0.441±0.045 | 0.506±0.036 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.376±0.000 | 0.376±0.001 | 0.376±0.001 | 0.375±0.002 | 0.376±0.001 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.487±0.000 | 0.487±0.001 | 0.487±0.001 | 0.487±0.001 | 0.488±0.002 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.385±0.000 | 0.385±0.001 | 0.385±0.001 | 0.385±0.002 | 0.386±0.001 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.488±0.000 | 0.488±0.001 | 0.488±0.001 | 0.488±0.001 | 0.489±0.002 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.219±0.071 (n=5) | 0.271±0.056 (n=7) | 0.324±0.054 | 0.376±0.027 |
| APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.240±0.040 (n=5) | 0.290±0.062 (n=7) | 0.342±0.064 | 0.387±0.029 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.900±0.001 | -0.902±0.002 | -0.902±0.003 | -0.900±0.005 | -0.900±0.005 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.391±0.038 | 0.542±0.039 | 0.632±0.026 | 0.683±0.011 | 0.720±0.010 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.487±0.033 | 0.612±0.030 | 0.653±0.022 | 0.709±0.003 | 0.731±0.008 |
| P0 + ui_reg (like-for-like: month oosc / year oos) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.489±0.035 | 0.612±0.030 | 0.623±0.016 | 0.699±0.018 | 0.731±0.009 |

#### year

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.145±0.017 | 0.135±0.014 | 0.133±0.011 | 0.132±0.004 | 0.150±0.022 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.198±0.018 | 0.263±0.027 | 0.256±0.024 | 0.286±0.009 | 0.315±0.052 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.199±0.018 | 0.261±0.026 | 0.254±0.026 | 0.304±0.011 | 0.376±0.061 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.219±0.029 | 0.325±0.033 | 0.334±0.017 | 0.386±0.013 | 0.435±0.072 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.065±0.076 | 0.017±0.101 | 0.020±0.051 | 0.098±0.042 | 0.100±0.046 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.135±0.116 | -0.111±0.110 | -0.019±0.077 | -0.110±0.062 | -0.131±0.111 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.029±0.000 | 0.029±0.001 | 0.028±0.001 | 0.028±0.001 | 0.033±0.005 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.010±0.058 | 0.119±0.046 | 0.184±0.040 | 0.247±0.035 | 0.315±0.048 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | -0.004±0.078 | 0.168±0.038 | 0.215±0.029 | 0.277±0.035 | 0.345±0.044 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.004±0.066 | 0.126±0.062 | 0.193±0.047 | 0.265±0.036 | 0.335±0.048 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | -0.008±0.085 | 0.183±0.044 | 0.232±0.036 | 0.295±0.035 | 0.360±0.049 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.219±0.000 | 0.219±0.000 | 0.218±0.001 | 0.217±0.001 | 0.241±0.041 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.334±0.000 | 0.334±0.000 | 0.333±0.001 | 0.331±0.001 | 0.368±0.063 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.225±0.000 | 0.225±0.000 | 0.224±0.001 | 0.223±0.001 | 0.248±0.042 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.344±0.000 | 0.344±0.000 | 0.343±0.001 | 0.340±0.001 | 0.379±0.064 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.105±0.039 (n=7) | 0.150±0.034 | 0.189±0.016 | 0.213±0.047 |
| APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.149±0.042 (n=7) | 0.189±0.025 | 0.234±0.018 | 0.257±0.050 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.625±0.001 | -0.622±0.004 | -0.621±0.007 | -0.621±0.003 | -0.688±0.121 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.191±0.021 | 0.267±0.025 | 0.290±0.046 | 0.367±0.007 | 0.478±0.088 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.228±0.023 | 0.321±0.030 | 0.309±0.029 | 0.393±0.011 | 0.488±0.086 |
| P0 + ui_reg (like-for-like: month oosc / year oos) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.230±0.026 | 0.345±0.040 | 0.348±0.017 | 0.405±0.009 | 0.492±0.087 |


### 3.2 native-query 子集：只在 50-bin 因果窗口完全落在 trial 内的 query bin 上打分（APST 自己的口径）

同一批预测、同一批 `(target,B,draw)`，只是去掉每个 trial 起始的前 49 个 bin。P-prior / P0 在这里是**重新前向**得到的（与 P1 存档的全 query R² 逐行一致，见 `native_p_reproduction_check.csv`）。R1-stream / R3 没有保存逐 bin 预测，不在此表。

#### days

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.250±0.029 | 0.275±0.021 | 0.273±0.026 | 0.273±0.011 | 0.272±0.014 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.643±0.026 | 0.712±0.012 | 0.728±0.011 | 0.737±0.006 | 0.744±0.007 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.265±0.088 | 0.330±0.098 | 0.384±0.055 | 0.371±0.045 | 0.419±0.059 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.151±0.112 | -0.163±0.130 | -0.192±0.202 | -0.080±0.096 | -0.089±0.129 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.331±0.002 | 0.331±0.004 | 0.331±0.003 | 0.332±0.005 | 0.329±0.006 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.012±0.065 | 0.170±0.067 | 0.349±0.089 | 0.426±0.037 | 0.488±0.018 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.072±0.032 | 0.229±0.054 | 0.373±0.063 | 0.442±0.034 | 0.521±0.018 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.014±0.085 | 0.200±0.086 | 0.406±0.092 | 0.480±0.038 | 0.536±0.018 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.078±0.042 | 0.276±0.057 | 0.436±0.045 | 0.481±0.033 | 0.559±0.018 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.554±0.001 | 0.555±0.004 | 0.552±0.001 | 0.550±0.005 | 0.555±0.008 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.614±0.001 | 0.616±0.004 | 0.612±0.001 | 0.611±0.005 | 0.617±0.009 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.582±0.001 | 0.583±0.003 | 0.580±0.002 | 0.577±0.004 | 0.583±0.007 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.637±0.001 | 0.639±0.004 | 0.636±0.002 | 0.633±0.004 | 0.642±0.009 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.393±0.163 | 0.700±0.083 | 0.789±0.012 | 0.814±0.011 |
| APST ACT-only (last segment, source-only rule) | B trials: neural only (no labels) | no | 0.041±0.094 | 0.257±0.096 | 0.476±0.100 | 0.546±0.116 | 0.606±0.138 |
| APST-FULL (authors' dev-selected segment_09; dev = our days targets, so optimistic) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.449±0.164 | 0.731±0.059 | 0.799±0.010 | 0.822±0.010 |
| APST ACT-only (authors' dev-selected segment_08; dev = our days targets, so optimistic) | B trials: neural only (no labels) | no | 0.062±0.105 | 0.290±0.111 | 0.477±0.115 | 0.560±0.128 | 0.607±0.147 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.174±0.002 | -0.170±0.011 | -0.174±0.008 | -0.175±0.016 | -0.177±0.018 |

#### month

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.293±0.051 | 0.297±0.021 | 0.312±0.011 | 0.308±0.011 | 0.311±0.005 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.463±0.043 | 0.538±0.017 | 0.574±0.011 | 0.567±0.009 | 0.571±0.009 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.059±0.103 | 0.244±0.053 | 0.223±0.092 | 0.257±0.052 | 0.295±0.024 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.089±0.035 | -0.144±0.106 | -0.190±0.167 | -0.105±0.201 | -0.191±0.178 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.250±0.000 | 0.251±0.001 | 0.251±0.002 | 0.251±0.001 | 0.252±0.002 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.037±0.167 | 0.206±0.116 | 0.358±0.052 | 0.413±0.065 | 0.447±0.060 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.059±0.059 | 0.294±0.094 | 0.402±0.044 | 0.441±0.053 | 0.513±0.046 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.001±0.109 | 0.253±0.129 | 0.384±0.040 | 0.444±0.059 | 0.479±0.061 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.067±0.070 | 0.336±0.094 | 0.417±0.051 | 0.454±0.056 | 0.528±0.045 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.367±0.000 | 0.367±0.001 | 0.367±0.001 | 0.367±0.002 | 0.367±0.002 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.478±0.000 | 0.478±0.001 | 0.478±0.001 | 0.478±0.001 | 0.479±0.002 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.384±0.000 | 0.384±0.001 | 0.384±0.001 | 0.384±0.002 | 0.385±0.002 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.486±0.000 | 0.486±0.001 | 0.486±0.001 | 0.487±0.002 | 0.487±0.002 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.284±0.080 (n=5) | 0.338±0.075 (n=7) | 0.394±0.072 | 0.466±0.037 |
| APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.306±0.052 (n=5) | 0.357±0.081 (n=7) | 0.417±0.084 | 0.477±0.038 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.877±0.001 | -0.879±0.002 | -0.879±0.004 | -0.876±0.006 | -0.877±0.007 |

#### year

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.187±0.019 | 0.186±0.016 | 0.180±0.014 | 0.183±0.006 | 0.187±0.007 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.273±0.031 | 0.378±0.047 | 0.361±0.044 | 0.416±0.017 | 0.415±0.019 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.122±0.126 | 0.004±0.160 | 0.011±0.083 | 0.148±0.081 | 0.136±0.087 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.245±0.185 | -0.163±0.161 | -0.031±0.128 | -0.159±0.071 | -0.159±0.180 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.033±0.000 | 0.033±0.001 | 0.032±0.003 | 0.032±0.001 | 0.035±0.003 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.017±0.102 | 0.191±0.085 | 0.285±0.067 | 0.381±0.056 | 0.446±0.026 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | -0.017±0.140 | 0.257±0.064 | 0.324±0.044 | 0.419±0.050 | 0.472±0.017 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.001±0.108 | 0.204±0.112 | 0.300±0.073 | 0.412±0.056 | 0.481±0.025 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | -0.027±0.152 | 0.280±0.078 | 0.355±0.049 | 0.451±0.050 | 0.498±0.019 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.366±0.000 | 0.366±0.001 | 0.366±0.002 | 0.365±0.001 | 0.366±0.002 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.500±0.000 | 0.500±0.001 | 0.499±0.002 | 0.498±0.001 | 0.500±0.002 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.383±0.000 | 0.382±0.001 | 0.382±0.002 | 0.381±0.001 | 0.383±0.002 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.519±0.000 | 0.520±0.001 | 0.519±0.002 | 0.518±0.001 | 0.520±0.002 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.200±0.072 (n=7) | 0.283±0.059 | 0.365±0.027 | 0.359±0.032 |
| APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.284±0.081 (n=7) | 0.352±0.048 | 0.444±0.032 | 0.433±0.027 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -1.007±0.001 | -1.004±0.009 | -1.005±0.014 | -1.010±0.006 | -1.006±0.015 |


### 3.3 辅助检查与对照

**(i) 桥的上限（baseline day 自己走一遍）**——Stabilizer / CCA 的“FA 或 PCA 潜变量 → AP 潜变量”K×K 桥本身损失多少（用 baseline day 的全部数据，不涉及任何对齐）：

| stratum | ref_session | source table 直接解码 | 经 FA 桥 | 经 PCA 桥 |
|---|---|---|---|---|
| days | sub-M_ses-CO-20150615 | 0.884 | 0.807 | 0.831 |
| month | sub-M_ses-CO-20140306 | 0.918 | 0.865 | 0.867 |
| year | sub-M_ses-CO-20140306 | 0.918 | 0.865 | 0.867 |

所以两类 read-in 对齐的上限分别约为 0.81/0.83（days）、0.865/0.867（month、year），而 P0 在 days 已达 0.72–0.74；对齐误差（不是桥）是主要损失。

**(ii) NoMAD 逐 target / seed：**

| stratum | target | seed | query R²（全部 B、draw 平均） | native 子集 R² | alignment epochs |
|---|---|---|---|---|---|
| days | sub-M_ses-CO-20150616 | 0 | 0.481 | 0.528 | 122 |
| days | sub-M_ses-CO-20150616 | 1 | 0.474 | 0.518 | 119 |
| days | sub-M_ses-CO-20150616 | 2 | 0.472 | 0.515 | 111 |
| days | sub-M_ses-CO-20150617 | 0 | -0.780 | -0.922 | 122 |
| days | sub-M_ses-CO-20150617 | 1 | -0.724 | -0.854 | 113 |
| days | sub-M_ses-CO-20150617 | 2 | -0.702 | -0.830 | 129 |
| month | sub-M_ses-CO-20140626 | 0 | -0.917 | -0.893 | 113 |
| month | sub-M_ses-CO-20140626 | 1 | -0.882 | -0.885 | 114 |
| month | sub-M_ses-CO-20140626 | 2 | -0.904 | -0.855 | 122 |
| year | sub-M_ses-CO-20150511 | 0 | -0.636 | -1.029 | 121 |
| year | sub-M_ses-CO-20150511 | 1 | -0.588 | -0.936 | 115 |
| year | sub-M_ses-CO-20150511 | 2 | -0.683 | -1.053 | 115 |
| days | sub-M_ses-CO-20150615 (control: baseline day through its own alignment net) | 0 | 0.647 |  | 129 |

**(iii) APST 作者自己的 dev 数（first-M 协议、Q50、dev-selected checkpoint，来自 `…/budget/dev_results/summary.csv`，只读 dev，没有读 final）**：APST-FULL M=4/8/16/32 的两个 dev session 均值为 0.511 / 0.831 / 0.839 / 0.841；ACT-only 为 0.294 / 0.620 / 0.621 / 0.699。这些数用的是 dev 上选的 checkpoint、前 M 个 trial、作者的 Q50 query，与本评测的随机 B-子集 + 我们的 query 口径**不是同一协议**，只用来说明：我们的适配器在作者协议下复现这一数（0616，M=8：0.855956 对 0.855956），而在 native 子集上用随机 B-子集得到的是 0.70–0.73（B=8）/ 0.81–0.82（B=33）。

**(iv) 与 P0 的配对差（P0 − 方法，逐 draw 配对；全表见 `paired_vs_p0.csv` / `paired_vs_p0_native.csv`）**：主口径下，H4 合规的已发表 baseline 在 days 和 month 的每个 B 上都显著低于 P0（配对差的 CI 不跨 0）；year 上 label-using 的 PCA+CCA（plain / 岭，raw / √）在 B=16 与 P0 持平、B=33 持平或略高（岭 CCA +0.044 [±0.013]）。native 口径下另有：days 的 APST-FULL 在 B=8 与 P0 持平（−0.028 [±0.075]）、B=16/33 高于 P0（+0.052 / +0.070，±0.012；dev 选择的 segment +0.062 / +0.077），year 的 APST-FULL（source-OOS 选择的 segment）在 B=8/16/33 与 P0 持平（−0.009 / +0.028 / +0.018，±0.04–0.07），month 的岭 CCA 在 B=33 与 P0 持平（−0.042 [±0.048]）。

## 4. 局限与未做

1. **target 数少**：days 2 个、month 与 year 各 1 个且共享 trunk 与 source，所以 CI 只代表 draw 方差；上面“高于 P0”的两处例外（year 的 CCA、days 的 APST）在这个意义下成立，不能外推到别的 target。
2. **Stabilizer 与 CCA 的实现是“忠实移植 + 为接冻结 AP trunk 加的桥”**，不是原文的闭环实验；桥（K×K 最小二乘）本身有 0.04–0.07 的上限损失（§3.3 i）。Stabilizer 的 unit 对应关系（同电极按发放率排序）是我们的规则，其它规则（例如先做 unit 匹配再 FA）可能更好；Gallego 的条件平均只有 B 个 trial，原文用整日数据。
3. **APST month/year 的 checkpoint 是我们在 2014 roster 上按作者冻结配方重训的**（seed 42，24×3165 更新，取最后一段和 0307 上 earliest-max 的一段；0307 既不是 target 也不参与评估）。ACT-only 没有为 2014 训练。APST 的 source-only 选择规则在 days 上只有“最后一段”可用（可选的 held-out source session 不存在）。
4. **query 口径**：主表用 P 行的口径（trial 内全部 bin）；APST 为 native 口径训练，所以主表对 APST 不利；native 表对不能重算的 R1-stream / R3 缺项。
5. **NoMAD**：TF 2.0→2.6 的移植需要 6 处运行时补丁（§2.4），训练在 CPU 上，每个 target 只有 3 个 alignment seed；alignment 训练在 ≈120 epoch 早停，若放宽早停是否改善没有试（没有 target 上调参，按“不在 target query 上调”的规则）。
6. **所有 CI 只含 draw 方差**；`(b)` 行和 NoMAD 的对齐不依赖 B/draw，同一行内不同 B 之间只有 query mask 不同。
7. 与规则 7–9 的关系：评测是整段连续因果解码；没有把同一过程的长步数版本当“上限”（`*_b` 行是独立的“论文数据量”过程，标了违反 H4）；本评测不包含硬件数字。

## 5. 复现

环境：SSM 代码 `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM /home/xinyuan/miniconda3/envs/spint/bin/python`；APST 另加 `/home/xinyuan/Work_host/APST/src` 到 PYTHONPATH；NoMAD 用 `/mnt/data/work_host/envs/nomad_tf26/bin/python`（CPU，`CUDA_VISIBLE_DEVICES=""`）。

```
python scripts/run_baselines_readin.py select --stratum {days,month,year}   # source-only 选超参
python scripts/run_baselines_readin.py eval   --stratum {days,month,year}   # Stabilizer + CCA 行
python scripts/run_baselines_readin.py sanity --stratum {days,month,year}   # 桥上限
python scripts/run_baselines_native_p.py {days,month,year}                  # P-prior/P0 重算 + native
python scripts/run_baselines_apst.py eval --stratum days  --model {apst_full_last,apst_act_last,apst_full_devsel,apst_act_devsel}
python scripts/apst_2014_train.py prepare; python scripts/apst_2014_train.py train --name sua_concat_pretrain_s42  # 然后 sua_dual_site_F0_s42
python scripts/run_baselines_apst.py eval --stratum {month,year} --model {apst14_full_last,apst14_full_sel0307}
python scripts/nomad_prep.py days month year
python scripts/run_baselines_nomad_tf.py {day0,decoder} --stratum {days,month} --tag {days,y2014}; ... align --target <session> --seed {0,1,2}
python scripts/run_baselines_nomad_score.py {days,month,year}
python scripts/analyze_baselines.py
```
