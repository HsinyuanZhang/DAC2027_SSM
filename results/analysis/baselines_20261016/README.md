# 已发表方法 baseline 与可部署行的同协议对比（DANDI 000688 sub-M，DAC2027）——两轮审查后修订版

日期：初版 2026-10-16，第一轮审查后修订 2026-10-17，第二轮审查后修订 2026-10-17。执行者：baseline 工作线（read-in 对齐与 APST 评测用单卡 cuda:0，NoMAD 全程 CPU）。规格：MZ 的任务说明（`docs/paper_mainline_proposal_20261016.md`、`docs/feasibility_p1_20261015.md` §0、P1/P2/P3 README 的协议）。

代码：`ssm_decode/published_baselines.py`（FA/Procrustes/Stabilizer、PCA+CCA）、`ssm_decode/electrode_align.py`（electrode pooling、电极匹配列重排、NoMAD 列顺序 guard，仅依赖 numpy）、`scripts/run_baselines_readin.py`（source 选择 / 评测 / 桥自检）、`scripts/run_baselines_apst.py`（APST 适配器）、`scripts/apst_2014_train.py`（APST 在 2014 roster 上按作者冻结配方重训）、`scripts/run_baselines_nomad_tf.py` + `nomad_prep.py` + `run_baselines_nomad_score.py` + `run_nomad_jobs.sh`（NoMAD）、`scripts/run_baselines_native_p.py`（P-prior / P0 在 native-query 子集上重算）、`scripts/analyze_baselines.py`（汇总）、`scripts/baselines_selection_sensitivity.py`（第二次审查：重选之后的敏感性）、`tests/test_published_baselines.py` + `tests/test_electrode_align.py`（单元测试）。
原始 rows：`results/baselines_20261016/`（gitignored；`eval/*.jsonl`、`select/*.jsonl`、`apst2014/`、`nomad/`、`logs/`；审查前第一版在 `v1_pre_review/`）。本目录是汇总：`headline_table.md`（B=1/8/33 的 headline 表）、`main_table.md`（全表）、`main_table_native.md`、`main_table_per_target.md` / `headline_per_target.md`（逐 target 表）、`cells*.csv`、`cells_per_target*.csv`、`paired_vs_p0*.csv`（**值 = baseline − P0**）、`beats_p0_counts_per_target.csv`、`rows_long.csv.gz`、`method_registry.csv`、`selection_sensitivity_configs.csv` + `selection_sensitivity_eval.csv`（第二次审查）、`selection_readin.csv` + `selected_configs.csv`、`bridge_ceiling.csv`、`unavailable_cells.csv`、`nomad_per_target_seed.csv`、`nomad_legacy_invalid_column_order.csv`、`cells_v1_pre_review_superseded.csv`、`native_p_reproduction_check.csv`、`stats_notes.json`。

## 审查后修正（2026-10-17）

初版（commit e203a36）经独立审查后发现下列问题；本节说明处理方式和对结论的影响。**初版里“没有任何一个已发表 baseline 在我们的 query 口径下追上 P0”“NoMAD 在 3/4 个 target 上对齐失败”两条结论都不成立，已撤回；下面和 §0 是修正后的结论。**

| # | 严重度 | 审查发现 | 处理 |
|---|---|---|---|
| 1 | CRITICAL | NoMAD 把 target 的 sorted unit 按各自的列序零填充，而 `INIT_READIN_FROM_DAY0=True` 把 target read-in 初始化成 day-0 的 read-in：第 c 列在 day 0 和 target 上是不同的 unit。0617 有 11/17 列的电极与 day-0 同位置列不一致。 | 主变体改为 **electrode-pooled 输入**（每个 session 把同电极 unit 的计数相加，固定 96 个 M1 通道），重训 day-0 LFADS + decoder（DATA_DIM=96）；另给**电极匹配列重排**变体（审查者用的）；加列顺序 guard（不对齐直接报错）与“baseline day 自己打乱列”对照；加 `INIT_READIN_FROM_DAY0=False` 敏感性；4 个 target × 3 seed 在 CPU 上全部重跑。NoMAD 现在用到 `unit_to_electrode`。数字见下面和 §3.3 (ii)。 |
| 2 | HIGH | 初版 §0 的第 1 条与 §0 第 3 条/`paired_vs_p0.csv` 矛盾；Gallego-style 标签前后不一致；配对差的符号（baseline − P0）表述混乱。 | §0 整体重写；“Gallego-style”只指忠实版（√ + plain CCA），岭 / raw 一律标 our variant；全文统一“配对差 = baseline − P0（正 = baseline 更好）”，csv 列改名 `diff_baseline_minus_p0`；整段 session 的（违反 H4）岭 CCA 行高于 P0 在 month / year 的每个 B 上成立，明文写出。 |
| 3 | MEDIUM | FA/PCA 维数固定为 K=10，桥损失 0.05–0.08；桥上限用的是 floor 0.1 + raw 的配置而不是表里的配置。 | d ∈ {3,5,8,10,N} 在 source pseudo-target 上选（K×d 桥，d=N 时精确）；桥上限对**表中实际使用的每个配置**重算（§3.3 i）。 |
| 4 | MEDIUM | Stabilizer 与 CCA 的 unit 配对规则是 ad-hoc 的；Gallego 的相位数/对齐没选；Stabilizer fraction 1.0 = 没有丢行。 | 加 electrode-pooled 变体（Stabilizer 与 CCA，固定电极通道，不需要配对规则；输入是电极上 sorted spike 的和，不是 96 通道阈值穿越）；Gallego 的相位数 ∈ {10,20,40} 与对齐 ∈ {整 trial, go cue 之后 1 s} 在 source 上选（NWB 没有 movement onset，用 go cue 近似）；明确披露 stable-row 丢行在所有最终配置里都没有运行（§2.1）。 |
| 5 | MEDIUM | APST days：20150616 / 20150617 是 APST 自己的 dev session；checkpoint 单 seed（42）。 | days 的 APST 行与 native-query “例外”都加标注（表标签、§2.3、§4.6）。 |
| 6 | MEDIUM | 统计表述。 | 加逐 target 表（`main_table_per_target.md`）；写明 4 个 target session / 2 个 trunk（month 与 year 同一 trunk，`state_dict` 逐项相同）/ 1 只猴子 / 1 个任务；CI 只含 draw 方差，B 无关的行没有真正的 CI；说明约 450 个（主 query 口径；native 口径另有若干）配对格未做多重比较校正；结论写成“在这 4 个 session 上（k/4）”。 |
| 7 | LOW | 不可用格；选择网格在第一次 target 评测之后扩展过（17:09 评测、17:46 的 select2 加了 √ 计数、ridge=100、`cca_a_plain`）；`best_ridge_cfg` 聚合里有 ridge=0 的旧行；`stab_a_noalign` 对照的 floor 与主行不一致；R1/R3 的选择集。 | 全部披露（§4.8、§4.4、§4.9）。**核对：去掉 ridge=0 的旧行后，初版所有选出的岭系数不变**（days raw 3 / √ 1，month、year 10；6 组全部一致）。`*_noalign` 对照现在与主行用完全相同的配置。新一轮的网格在任何 target 行之前完整固定；注意新网格仍是在看过初版 target 结果之后设计的。 |
| 8 | 引用 | APST 标题两处不一致；NoMAD 引用未核实；APST 自引需匿名。 | 见 §4.11（两个标题都列出，**请 MZ 确认**；NoMAD 引用标为离线未核实；APST 自引在双盲投稿时必须匿名化）。 |

## 第二次审查后修正（2026-10-17，second pass；commit 22ed055 之后）

第二轮独立审查对 commit 22ed055 提出下列问题。处理方式与对结论的影响如下；上面第一轮的表格、§0 和后面各节里的数字已经是**第二次修正后重跑的数字**（第一轮的数字在 commit 22ed055 的 README 与下面的前后对比表里）。

| # | 严重度 | 审查发现 | 处理 |
|---|---|---|---|
| 1 | MEDIUM | `resolve_d` 把 d 截断到 target 的有效维数，并把 baseline day 的 FA/PCA 与桥也重拟合到被截断的 d：month / year 的 sorted-unit 行（`stab_a/b`、`cca_*_ridge`）实际在 d=26 / 24 运行，而桥上限表写的是 d_eff=43、“d=N 时精确”。baseline day 的潜变量 R² 经桥在 d=26 / 24 只剩 0.79–0.83（审查者复现：月 Stabilizer 0.7914、CCA 0.8329；年 0.8073 / 0.8039；d=43 时为 1.0000）。days 的 electrode-pooled 行同样受影响（baseline day 13 个有信号的通道，target 只有 12 个，被截成 12）。 | **d 只由 baseline day 决定**（`resolve_d`：N = baseline day 的有效维数，或选出的数值 d，不再取 min(d, target 有效维数)）。target 的有效维数更少时，target 的 FA / PCA 潜变量是秩亏的（`fa_fit` 缺的维度载荷恒为 0；`pca_basis` 对缺的秩返回恰好为 0 的列，**不再返回 SVD 的零空间方向**——零空间方向会在 target 之后出现的活动上产生任意潜变量），由 Procrustes / CCA 映到 d 维的 baseline 参考潜空间，所以 target 不再被降维；桥 `A` 始终是 baseline day 上的 K×d 桥，在 d=N_ref 时精确。评测行的 `info` 记录 `d_eff`（= 参考维数）和 `d_target_active`。**用同样的 source 选出的配置**（没有在 target 上选任何东西）重跑了三个 stratum 的全部 baseline 行（2880 行）和桥上限检查（`logs/sanity_v3_*.log`）。旧的 eval jsonl 保留在 `results/baselines_20261016/eval_pre_dfix/`。 |
| 1b | — | 在 source 上的选择里 pseudo-target 也受同一个截断影响。 | 用修正后的代码**重复了 source 选择**（`BL_OUT=results/baselines_20261016_selsens`，days 与 month；year 与 month 共享 trunk、source 与选择）：16 个配置族中 13 个选出的配置不变；变的 3 个：days 的 `cca/units/sqrt/plain`（相位 20、go → 相位 40、trial，pseudo R² 0.3599 → 0.3623）、month / year 的 `cca/units/sqrt/ridge`（相位 20 → 10，0.4471 → 0.4483）、month / year 的 `stab/pooled`（d=10、floor 0.01、frac 1.0 → d=N、floor 0.1、frac 0.5，0.5775 → 0.5808）。**主表仍用原来 source 选出的配置**（任务要求“同样的 source 选出的配置”），换成重选的配置只作敏感性检查（`selection_sensitivity_configs.csv`、`selection_sensitivity_eval.csv`、`scripts/baselines_selection_sensitivity.py`）：B=8 / 33 的 R²——days Gallego-style (a)（sorted）0.538 / 0.682 → 0.446 / 0.660；month √ 岭 CCA 0.495 / 0.646 → 0.511 / 0.638；year 0.265 / 0.427 → 0.267 / 0.405；month pooled Stabilizer 0.295 / 0.304 → 0.353 / 0.354，year −0.004 / 0.034 → 0.088 / 0.085。所有这些仍低于 P0（month 0.565 / 0.566，year 0.256 / 0.315；month / year 的 √ 岭 CCA 在 B=33 仍高于 P0，与主表相同），结论不变。 |
| 2 | MEDIUM | electrode-pooled 输入被写成 “96-ch 阈值穿越”。 | 该输入是 **sorted spike 在每个电极上相加**（有信号的通道 days 12 个、month 20 个、year 17 个；与 baseline day 配对且两边有信号的 11 / 15 / 11 个），不是论文的 96 通道阈值穿越计数；本数据集只发布 sorted unit，所以这是**数据集强加的近似**。已改 README（§0、§2.1、§2.4、本节）和 registry 标签（`analyze_baselines.py`、`method_registry.csv`），所有 “96-ch” 字样已去掉（NoMAD 的 `DATA_DIM=96` 仍是 96 列的电极布局，说明里写明大部分列无信号）。 |
| 3 | MEDIUM-LOW | NoMAD 在 days 上低于 P0 说不了对齐好不好。 | NoMAD 自己的 baseline-day 上限（in-sample）在 days 只有 **0.534**，低于 P0 的 ~0.72，所以 days 的 NoMAD 比较几乎没有信息；**只有 month（上限 0.758）是有信息的比较**。已写入 §0.6。 |
| 4 | LOW | §0 措辞：漏掉了下列高于 P0 的格。 | §0.2–0.3 已补：raw plain CCA（`cca_a`）在 year B=33 也高于 P0（+0.007 [±0.033]，区间含 0）；`cca_pool_sqrt_b_ridge` 在 year B=1 高于 P0（+0.023 [±0.018]，区间不含 0）；pooled CCA `(b)` 在 days B=1/4 高于 P0；我们的岭 CCA 在 year B=16 高于 P0（**修正后 +0.066 raw / +0.048 √**，第一轮是 +0.057 / +0.033）。 |
| 5 | LOW | 标签不一致：§0.1 “忠实” vs registry 里 sorted-unit Stabilizer 是 `published-variant`；Stabilizer 在 d=N 时偏离原文很远；前后对比的说明只提了 Gallego 下降。 | §0.1 现在只把 registry 里标 `published` 的行叫“最接近已发表做法”，sorted-unit Stabilizer 单列为 variant（用了 ad-hoc unit 配对）；披露 d=N（以及 pooled 的 frac=1.0）时 Stabilizer 是**满秩 FA + 不丢行的 Procrustes**，与原文相差很远（§0.1、§0.5、§2.1）；前后对比说明里补上 Stabilizer 的下降（sorted-unit Stabilizer 在 month B=33：d=K=10 的 0.290 → 第一轮 0.252 → 现在 0.280；year：0.100 → −0.047 → 0.070，仍低于固定 K=10）。 |
| 6 | 措辞 | claim (c)。 | 改为：“在主 query 口径下，**P0+R3 与每个 H4 合规的已发表行相比是持平或更高**”（逐行配对核对：没有任何一行以 CI 排除 0 的方式高于 R3）；并写明 R3 没有 native-query 版本，APST 在它自己的 native 协议、自己的 dev session（days）上高于 P0（§0.2、§0.4）。 |

### 受影响的行：第一轮 → 第二次修正（query R²，draw 均值）

下表只列变化 ≥ 0.004 的行（B=1 / 8 / 33）。P 行（P-prior、P0、R1-stream、R3、ui_reg）、APST、NoMAD、`d` 为数值且 target 有效维数 ≥ d 的 month / year 的 pooled 行（d=5, 10）不受影响。变化来自：month / year 的 sorted-unit d=N 行（原先在 d=26 / 24）、days 的 pooled d=N 行（原先 12，现在 13）、B=1 时个别 draw 的有效维数更少的行。

| 行 | stratum | B=1 旧 → 新 | B=8 旧 → 新 | B=33 旧 → 新 |
|---|---|---|---|---|
| Stabilizer (a), electrode-pooled | days | 0.590 → 0.643 | 0.689 → 0.715 | 0.716 → 0.745 |
| Stabilizer (a), sorted units [variant] | month | 0.150 → 0.263 | 0.249 → 0.267 | 0.252 → 0.280 |
| Stabilizer (a), sorted units [variant] | year | -0.084 → 0.020 | -0.036 → 0.061 | -0.047 → 0.070 |
| Gallego-style (a), electrode-pooled | days | -0.103 → -0.132 | 0.523 → 0.533 | 0.663 → 0.674 |
| Gallego-style (a), sorted units | days | -0.157 → -0.123 | 0.538 → 0.538 | 0.682 → 0.682 |
| CCA (a), raw, plain, sorted units [variant] | days | -0.143 → -0.095 | 0.578 → 0.578 | 0.708 → 0.708 |
| ridge CCA (a), sqrt, electrode-pooled [variant] | days | -0.046 → 0.001 | 0.552 → 0.568 | 0.677 → 0.688 |
| ridge CCA (a), sqrt, electrode-pooled [variant] | month | 0.068 → 0.063 | 0.378 → 0.378 | 0.478 → 0.478 |
| ridge CCA (a), sqrt, electrode-pooled [variant] | year | -0.008 → 0.003 | 0.164 → 0.164 | 0.243 → 0.243 |
| ridge CCA (a), sqrt, sorted units [variant] | days | -0.040 → -0.056 | 0.576 → 0.576 | 0.702 → 0.702 |
| ridge CCA (a), sqrt, sorted units [variant] | month | 0.069 → 0.047 | 0.463 → 0.495 | 0.611 → 0.646 |
| ridge CCA (a), sqrt, sorted units [variant] | year | 0.019 → 0.033 | 0.250 → 0.265 | 0.411 → 0.427 |
| ridge CCA (a), raw, sorted units [variant] | days | -0.061 → -0.026 | 0.609 → 0.609 | 0.727 → 0.727 |
| ridge CCA (a), raw, sorted units [variant] | month | 0.068 → 0.060 | 0.491 → 0.512 | 0.626 → 0.656 |
| ridge CCA (a), raw, sorted units [variant] | year | 0.013 → 0.023 | 0.276 → 0.289 | 0.427 → 0.439 |
| Stabilizer (b), electrode-pooled | days | 0.722 → 0.748 | 0.722 → 0.747 | 0.722 → 0.746 |
| Stabilizer (b), sorted units [variant] | month | 0.233 → 0.290 | 0.233 → 0.290 | 0.234 → 0.291 |
| Stabilizer (b), sorted units [variant] | year | -0.022 → 0.055 | -0.022 → 0.055 | -0.024 → 0.061 |
| ridge CCA (b), sqrt, electrode-pooled | days | 0.709 → 0.719 | 0.709 → 0.719 | 0.709 → 0.718 |
| ridge CCA (b), sqrt, sorted units | month | 0.646 → 0.689 | 0.646 → 0.689 | 0.646 → 0.689 |
| ridge CCA (b), sqrt, sorted units | year | 0.406 → 0.412 | 0.405 → 0.411 | 0.447 → 0.454 |
| ridge CCA (b), raw, sorted units | month | 0.659 → 0.698 | 0.659 → 0.698 | 0.659 → 0.698 |
| Gallego-style (b), electrode-pooled | days | 0.689 → 0.697 | 0.690 → 0.697 | 0.690 → 0.696 |

**修正对结论的影响：**

- 变化最大的是 sorted-unit Stabilizer 在 month / year（B=33：0.252 → 0.280、−0.047 → 0.070）和 sorted-unit 岭 CCA 在 month（√：0.611 → 0.646，raw：0.626 → 0.656）；这些行仍然低于（Stabilizer）或只在 B≥16 高于（岭 CCA）P0，定性结论不变。
- **days 的 electrode-pooled Stabilizer (a) 从“低于 P0 −0.02…−0.03”变成与 P0 持平**（B=33：0.716 → 0.745，配对差 −0.022 → +0.007 [±0.007]）。这是本次修正唯一改变定性结论的地方：§0.1 的 “0/4” 不再成立（见 §0.1）。P0+R3（0.747）仍不低于它。
- **R3 对 ridge-CCA 在 year B=33**：R3 0.435；raw 岭 CCA 0.439（第一轮 0.427），√ 岭 CCA 0.427（第一轮 0.411）；配对差 岭 CCA − R3 = **+0.004 [±0.038]**（raw）、−0.008 [±0.041]（√）：持平，CI 含 0。第一轮写的“R3 0.435 对 0.427，持平”现在是“0.435 对 0.439，持平”，仍不能说 R3 高于岭 CCA。month B=33：R3 0.679，raw 岭 CCA 0.656，岭 CCA − R3 = −0.023 [±0.017]。
- 单元测试新增（`tests/test_published_baselines.py`）：FA / PCA 桥在参考维数 d=N 时精确；d < N 时有损；`resolve_d` 返回参考维数、不被 target 截断；Stabilizer 与 CCA 在 target 有效维数少于参考维数时 shape / 有限性 / 沉默列无权重，且同一天在参考维数下精确还原。


### 修正后的 headline 表（query R²，物理单位，我们的 query 口径；mean ± 95% CI 只含 draw 方差）

`H4 = yes` 的行只用 B 个校准 trial；`NO` 的行用整段 target recording，**违反 H4，不可部署**，只作“论文数据量”参照。完整表见 §3.1 和 `main_table.md`；逐 target 见 §3.4。

#### days

| method | H4 | B=1 | B=8 | B=33 |
|---|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | yes | 0.333±0.025 | 0.360±0.023 | 0.361±0.012 |
| P0 (fixed-weight per-unit encoder, level 1) | yes | 0.647±0.021 | 0.723±0.009 | 0.738±0.005 |
| P0 + R1-stream (readout ridge, level 2) | yes | 0.648±0.021 | 0.723±0.010 | 0.740±0.005 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | yes | 0.651±0.020 | 0.728±0.010 | 0.747±0.006 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | yes | 0.643±0.040 | 0.715±0.014 | 0.745±0.008 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | yes | 0.528±0.035 | 0.633±0.023 | 0.667±0.025 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | yes | -0.132±0.177 | 0.533±0.031 | 0.674±0.017 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | yes | -0.123±0.133 | 0.538±0.037 | 0.682±0.008 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | yes | 0.001±0.083 | 0.568±0.035 | 0.688±0.015 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | yes | -0.056±0.149 | 0.576±0.052 | 0.702±0.008 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | yes | -0.026±0.122 | 0.609±0.058 | 0.727±0.007 |
| APST-FULL (last segment, source-only rule; single training seed 42) | yes | n/a | 0.500±0.059 | 0.590±0.009 |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | yes | n/a | 0.516±0.043 | 0.588±0.009 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | NO (violates H4) | 0.748±0.000 | 0.747±0.001 | 0.746±0.004 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | NO (violates H4) | 0.719±0.000 | 0.719±0.001 | 0.718±0.006 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | NO (violates H4) | 0.753±0.001 | 0.752±0.002 | 0.753±0.004 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | NO (violates H4) | 0.457±0.001 | 0.456±0.002 | 0.458±0.007 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | NO (violates H4) | 0.515±0.001 | 0.514±0.002 | 0.515±0.007 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | yes | 0.378±0.028 | 0.607±0.038 | 0.704±0.015 |
| P0 + ui_reg (P2 rule) | yes | 0.654±0.019 | 0.720±0.015 | 0.746±0.008 |

#### month

| method | H4 | B=1 | B=8 | B=33 |
|---|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | yes | 0.315±0.046 | 0.332±0.010 | 0.333±0.004 |
| P0 (fixed-weight per-unit encoder, level 1) | yes | 0.466±0.034 | 0.565±0.011 | 0.566±0.006 |
| P0 + R1-stream (readout ridge, level 2) | yes | 0.472±0.037 | 0.576±0.011 | 0.606±0.008 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | yes | 0.492±0.048 | 0.630±0.011 | 0.679±0.007 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | yes | 0.154±0.060 | 0.295±0.030 | 0.304±0.067 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | yes | 0.263±0.077 | 0.267±0.044 | 0.280±0.021 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | yes | -0.013±0.169 | 0.305±0.043 | 0.365±0.065 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | yes | 0.032±0.071 | 0.316±0.053 | 0.390±0.039 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | yes | 0.063±0.065 | 0.378±0.025 | 0.478±0.017 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | yes | 0.047±0.054 | 0.495±0.046 | 0.646±0.020 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | yes | 0.060±0.059 | 0.512±0.050 | 0.656±0.017 |
| APST-FULL (last segment, source-only rule; single training seed 42) | yes | n/a | 0.271±0.056 (n=7) | 0.376±0.027 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | yes | n/a | 0.290±0.062 (n=7) | 0.387±0.029 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | NO (violates H4) | 0.376±0.000 | 0.376±0.001 | 0.376±0.001 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | NO (violates H4) | 0.466±0.000 | 0.466±0.001 | 0.466±0.001 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | NO (violates H4) | 0.698±0.000 | 0.698±0.000 | 0.698±0.001 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | NO (violates H4) | 0.230±0.000 | 0.230±0.001 | 0.227±0.004 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | NO (violates H4) | 0.101±0.000 | 0.101±0.002 | 0.099±0.004 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | yes | 0.391±0.038 | 0.632±0.026 | 0.720±0.010 |
| P0 + ui_reg (P2 rule) | yes | 0.487±0.033 | 0.653±0.022 | 0.731±0.008 |

#### year

| method | H4 | B=1 | B=8 | B=33 |
|---|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | yes | 0.145±0.017 | 0.133±0.011 | 0.150±0.022 |
| P0 (fixed-weight per-unit encoder, level 1) | yes | 0.198±0.018 | 0.256±0.024 | 0.315±0.052 |
| P0 + R1-stream (readout ridge, level 2) | yes | 0.199±0.018 | 0.254±0.026 | 0.376±0.061 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | yes | 0.219±0.029 | 0.334±0.017 | 0.435±0.072 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | yes | -0.054±0.055 | -0.004±0.059 | 0.034±0.040 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | yes | 0.020±0.050 | 0.061±0.021 | 0.070±0.019 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | yes | -0.011±0.040 | 0.101±0.047 | 0.181±0.037 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | yes | 0.030±0.072 | 0.155±0.050 | 0.243±0.053 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | yes | 0.003±0.079 | 0.164±0.033 | 0.243±0.046 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | yes | 0.033±0.068 | 0.265±0.025 | 0.427±0.074 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | yes | 0.023±0.062 | 0.289±0.030 | 0.439±0.076 |
| APST-FULL (last segment, source-only rule; single training seed 42) | yes | n/a | 0.150±0.034 | 0.213±0.047 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | yes | n/a | 0.189±0.025 | 0.257±0.050 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | NO (violates H4) | 0.079±0.000 | 0.079±0.001 | 0.087±0.014 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | NO (violates H4) | 0.221±0.000 | 0.220±0.001 | 0.244±0.041 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | NO (violates H4) | 0.412±0.000 | 0.410±0.001 | 0.454±0.077 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | NO (violates H4) | -0.378±0.000 | -0.376±0.003 | -0.414±0.072 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | NO (violates H4) | -0.216±0.000 | -0.215±0.003 | -0.237±0.043 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | yes | 0.191±0.021 | 0.290±0.046 | 0.478±0.088 |
| P0 + ui_reg (P2 rule) | yes | 0.228±0.023 | 0.309±0.029 | 0.488±0.086 |


配对差（baseline − P0，正 = baseline 更好）的全表在 `paired_vs_p0.csv`，各 target 上高于 P0 的个数在 `beats_p0_counts_per_target.csv`。

### NoMAD 修正后的数字

逐 target、跨 3 个 alignment seed 的平均 query R²（所有 B、draw 平均；±= seed 间标准差）。P0 取同一 target 上所有 B 的平均（B=1…33 的范围是 0.639–0.740 / 0.656–0.735 / 0.466–0.566 / 0.198–0.315）。

| target | 初版（**无效**：target 列顺序未对齐） | **pooled（主）** | units_elec（电极重排） | P0（B=8） |
|---|---:|---:|---:|---:|
| days 20150616 | +0.476 | **+0.457 ± 0.003** | +0.532 ± 0.002 | 0.730 |
| days 20150617 | −0.735 | **+0.457 ± 0.010** | +0.497 ± 0.007 | 0.717 |
| month 20140626 | −0.901 | **+0.229 ± 0.038** | +0.101 ± 0.012 | 0.565 |
| year 20150511 | −0.636 | **−0.383 ± 0.044** | −0.219 ± 0.004 | 0.256 |

对照（baseline day 自己走一遍 alignment network，R² 在该 session 全部 usable trial 上；它是 NoMAD 在这套 day-0 模型上的上限，且是 in-sample）：

| 变体 | baseline day 未打乱 | baseline day 活跃列随机打乱（必须失败） |
|---|---:|---:|
| pooled，days（20150615） | +0.534 | **−0.486** |
| pooled，month / year（20140306） | +0.758 | **−0.916** |
| 初版 sorted-unit，days（20150615；审查者复核） | +0.647 | −0.882 |

审查者的电极重排结果（0617 +0.503、0626 +0.088）被本版独立重跑复现（units_elec：0617 +0.492…+0.505，0626 +0.088…+0.113）。

### 审查前后的数字变化（H4 合规的 read-in 行，query R²，draw 均值）

“旧”= 初版（FA/PCA 维数固定 K=10，sorted unit，d 没选）；“新”= 本版（d、相位数、对齐在 source 上选；**第二次审查后 d 不再被 target 截断，下表是重跑之后的数字**；第一轮的数字见 commit 22ed055 和下面“第二次审查后修正”的表）。注意新版 source 选出的配置在 target 上不一定更好：**Gallego-style plain CCA 在 month/year 变差（d=5，见上面）；sorted-unit Stabilizer 在 month 也低于 d=K=10 的旧行（B=33：0.290 → 0.280，第一轮是 0.252），在 year 的 B=33 也从 0.100 降到 0.070（第一轮是 −0.047）——source 上选出的 d=N（满秩 FA + 不丢行的 Procrustes）在 target 上不如固定的 K=10。**

| 行 | stratum | B=8 旧 (d=K=10) | B=8 新 | B=33 旧 | B=33 新 |
|---|---|---:|---:|---:|---:|
| Stabilizer, sorted units | days | 0.400 | 0.633 | 0.435 | 0.667 |
| Stabilizer, sorted units | month | 0.214 | 0.267 | 0.290 | 0.280 |
| Stabilizer, sorted units | year | 0.020 | 0.061 | 0.100 | 0.070 |
| Stabilizer, electrode-pooled (new) | days | — | 0.715 | — | 0.745 |
| Stabilizer, electrode-pooled (new) | month | — | 0.295 | — | 0.304 |
| Stabilizer, electrode-pooled (new) | year | — | -0.004 | — | 0.034 |
| Gallego-style (sqrt, plain CCA), sorted units | days | 0.391 | 0.538 | 0.525 | 0.682 |
| Gallego-style (sqrt, plain CCA), sorted units | month | 0.333 | 0.316 | 0.423 | 0.390 |
| Gallego-style (sqrt, plain CCA), sorted units | year | 0.184 | 0.155 | 0.315 | 0.243 |
| Gallego-style (sqrt, plain CCA), pooled (new) | days | — | 0.533 | — | 0.674 |
| Gallego-style (sqrt, plain CCA), pooled (new) | month | — | 0.305 | — | 0.365 |
| Gallego-style (sqrt, plain CCA), pooled (new) | year | — | 0.101 | — | 0.181 |
| our variant: sqrt, ridge CCA, sorted units | days | 0.423 | 0.576 | 0.559 | 0.702 |
| our variant: sqrt, ridge CCA, sorted units | month | 0.384 | 0.495 | 0.497 | 0.646 |
| our variant: sqrt, ridge CCA, sorted units | year | 0.215 | 0.265 | 0.345 | 0.427 |
| our variant: raw, ridge CCA, sorted units | days | 0.472 | 0.609 | 0.583 | 0.727 |
| our variant: raw, ridge CCA, sorted units | month | 0.394 | 0.512 | 0.506 | 0.656 |
| our variant: raw, ridge CCA, sorted units | year | 0.232 | 0.289 | 0.360 | 0.439 |

### 没有做 / 没有解决的

- 没有做 NoMAD 的“随机初始化 read-in”敏感性：配置键 `INIT_READIN_FROM_DAY0` 在作者代码里是死键（见 §2.4、§3.3 ii）。
- Stabilizer 的 stable-channel 丢行步骤在所有最终配置里没有运行，没有单独检验（§2.1）。
- APST ACT-only 没有在 2014 roster 上训练；APST 只有单个训练 seed（42），没有多 seed。
- R1-stream / R3 没有 native-query 版本；没有为它们重新选配置（用 P3 的选择集）。
- Gallego 的 movement-onset 对齐：NWB 没有 movement-onset 事件，用 go cue 近似；没有从行为轨迹推 onset（那会用到行为标签）。
- 没有在 target 上做 d / 相位数 / 对齐的敏感性扫描（按“选择只在 source”规则）；source 上的坐标下降不是全网格。
- 没有更多 target session：结论受限于 4 个 session / 2 个 trunk；month 与 year 不是独立的 trunk 重复。
- 没有做独立的 code review（AGENTS.md 要求关键代码由另一个 Opus agent 复核），留给协调者。
- 没有 push（由协调者 push）。


## 0. 结论摘要（修正后）

读表口径：**P 行与 baseline 行是同一批 `(target, B, draw)` 子集、同一个冻结 backbone、同一个连续因果解码、同一个物理单位 R²**，逐行配对（§1）。“target data used / target backprop”两列逐行写明各方法用了什么（§3）。**配对差一律是 baseline − P0（正 = baseline 比 P0 好）**。所有 CI 只含 draw 方差（8 个 draw）；数据是 4 个 target session（days 2 个、month 1 个、year 1 个）、2 个 trunk（month 与 year 共用同一个）、1 只猴子、1 个任务，所以下面所有说法都限于“在这 4 个 session 上（k/4）”。

1. **H4 合规的、最接近已发表做法的行（registry 标 `published`）在这 4 个 session 上基本不高于 P0，只有 days 上的 electrode-pooled Stabilizer 与 P0 持平。**“最接近已发表做法”指：electrode-pooled 输入的 Stabilizer、Gallego-style（√ + plain CCA，sorted-unit 与 electrode-pooled 两种输入）、APST-FULL（主 query 口径）。**sorted-unit 输入的 Stabilizer 用了我们加的 ad-hoc unit 配对规则，registry 里是 `published-variant`，这里不叫“忠实”，只作变体列出。**两点必须同时说明：（i）electrode-pooled 输入本身是数据集强加的近似——每个电极上 sorted spike 相加，不是论文的 96 通道阈值穿越计数（实际有信号的通道 days 12 个、month 20 个、year 17 个）；（ii）d=N（days）或 frac=1.0（pooled month/year）时 Stabilizer 是满秩 FA + 不丢行的 Procrustes，与原文的低维 FA + 稳定通道识别相差很远（§2.1），所以它不代表原 Stabilizer。第二次审查后（d 不再被 target 截断，§“第二次审查后修正”）的结果：除 days 的 electrode-pooled Stabilizer 外，`published` 行在 B=1/8/33 的每个 stratum 都低于 P0（0 个 target 高于）。**days 的 electrode-pooled Stabilizer (a) 与 P0 持平**：配对差 B=1/4/8/16/33 为 −0.005 [±0.041] / +0.005 [±0.022] / −0.009 [±0.007] / 0.000 [±0.009] / +0.007 [±0.007]；两个 days target 一正一负（B=33：0616 −0.012，0617 +0.026；B=8：−0.034 / +0.017），所以按“|差| ≤ 0.01、target 间符号不一致”读作持平，不读作更好；它在 B=33 的区间刚好不含 0，这一点如实写出。**初版和第一轮写的“B=1/8/33 都是 0/4”在 d 修正之后对这一行不再成立**（修正前它是 −0.02…−0.06，是被 d 截断压低的）。B=8 时 P0 与这些行中最好的：days 0.723 对 0.715（pooled Stabilizer，−0.009 [±0.007]；Gallego-style 0.538，−0.186 [±0.034]；sorted-unit Stabilizer [variant] 0.633；APST-FULL 0.500）；month 0.565 对 0.316（Gallego-style，−0.249 [±0.054]；pooled Stabilizer 0.295，−0.270；sorted-unit Stabilizer [variant] 0.267）；year 0.256 对 0.155（Gallego-style，−0.101 [±0.035]；pooled Stabilizer −0.004；sorted-unit Stabilizer [variant] 0.061，−0.195 [±0.033]；APST 0.150 / 2014 重训并在 source-OOS 选段的 0.189）。**B=1 差距最大**：P0 为 0.647 / 0.466 / 0.198，`published` 行最好为 0.643（days pooled Stabilizer）/ 0.154（month pooled Stabilizer；sorted-unit Stabilizer [variant] 0.263）/ 0.030（year Gallego-style sorted）（B=1 时 CCA 只有一个方向的条件平均，欠定；APST-FULL 在 B=1 不可用）。
2. **高于 P0 的主要是我们加的岭正则 CCA 变体（用 reach-direction 标签和 go cue 时间），而且只在大 B、month / year**：B=33 时 raw 计数 + 岭 CCA（sorted unit）对 P0 的配对差为 days −0.010 [±0.006]、month +0.090 [±0.020]、year +0.123 [±0.026]（3/4 个 target 高于 P0）；√ + 岭 CCA 为 −0.036 [±0.007] / +0.079 [±0.023] / +0.111 [±0.025]（2/4）。B=16 时 month / year 也高于 P0：raw +0.039 [±0.021] / +0.066 [±0.020]，√ +0.025 [±0.021] / +0.048 [±0.019]（**year B=16 比第一轮报的 +0.057 / +0.033 更大，因为 d 不再被截断**）。**这些变体不是原文方法**（岭系数、d、相位数、对齐都是 source 上选的），在 B≤8 仍不高于 P0（B=8：days −0.115 [±0.051]，month −0.053 [±0.055]，year +0.034 [±0.042]，持平）。另外 raw 计数 + plain CCA（`cca_a`，也是变体）在 year B=33 的点估计也高于 P0（+0.007 [±0.033]，区间含 0，是持平）。与 P 行的更高层级相比：month B=33 的 0.656 介于 P0+R1-stream（0.606）和 P0+R3（0.679）之间，year B=33 的 0.439 与 P0+R3（0.435）持平（配对差 岭 CCA − R3：raw +0.004 [±0.038]，√（0.427）−0.008 [±0.041]；raw 岭 CCA 在 year B=33 的点估计略高于 R3，但 CI ±0.04，是持平）；**逐行配对核对：P0+R3 在每个 (stratum, B) 上与每个 H4 合规的 `published` / `published-variant` 行相比都是“持平或更高”，没有任何一行以 CI 排除 0 的方式高于 R3**（最接近的是 year B=33 的 raw 岭 CCA +0.004 [±0.038] 与 days B=33 的 pooled Stabilizer −0.002 [±0.005]；days B=33 pooled Stabilizer 与 R3 在 0.745 / 0.747 处于同一水平）。**这个比较的限制（claim (c) 的措辞）：在主 query 口径下，P0+R3 与每个 H4 合规的已发表行相比“持平或更高”（comparable to or above），不是“全面高于”；R3 没有 native-query 版本，所以 R3 无法在 APST 自己的 native 口径下比较，而 APST 在它自己的 native 协议、自己的 dev session 上高于 P0（见第 4 条）——这一点不能被主 query 口径下的 R3 优势抵消。**P0+ui_reg（反传穿过 trunk，不可部署）在每个格上也不低于它们。
3. **整段 target recording 的（违反 H4，`(b)`）岭 CCA 行高于 P0**：month 在每个 B 上高（√ +0.12…+0.22，raw +0.13…+0.23），year 在每个 B 上高（√ +0.12…+0.21，raw +0.12…+0.21），days 在多数 B 上略高或持平（raw +0.015…+0.105；√ −0.006…+0.085）。sorted-unit 输入的 plain √ CCA `(b)`（Gallego-style）在 year 上也高于 P0（+0.028…+0.119），在 month 上低于 P0（−0.13…−0.23）。也就是说，在“论文的数据量”下 label-using 的 CCA 变体比 P0 强；P0 在 H4 约束（只有 B 个有标签 trial）下的位置取决于 B，而不是全面领先。electrode-pooled 输入的 `(b)` 行：Stabilizer `(b)` 在 days 的每个 B 上都高于 P0（+0.100 / +0.040 / +0.024 / +0.016 / +0.008，B=1…33），在 month / year 上都低于 P0；pooled 岭 CCA `(b)` 在 days 的 B=1/4 高于 P0（+0.071 / +0.012），pooled plain CCA `(b)` 只在 days B=1 高于 P0（+0.050），pooled 岭 CCA `(b)` 在 year B=1 高于 P0（+0.023 [±0.018]），其余格都不高于 P0（month 在每个 B 上为 −0.001…−0.144）。
4. **APST**：主 query 口径下（全部 bin），APST-FULL 在每个 (stratum, B) 都低于 P0（days −0.15…−0.43，month −0.19…−0.32，year −0.05…−0.16）。在 APST 自己的 native-query 口径下（§2.3）：days 的 B=16/33 高于 P0（+0.052 / +0.070 [±0.012]；dev 选择的 segment +0.062 / +0.077），B=8 持平（−0.028 [±0.075]），B=4 低很多；year（2014 重训 + source-OOS 选段）B=8/16/33 为 −0.009 / +0.028 / +0.018（±0.04–0.07，持平）；month 低于 P0。**但 days 的 20150616 / 20150617 是 APST 自己的 dev session，且 checkpoint 只有单个训练 seed（42）**，所以 days 的“例外”是 APST 在自己 dev 设定下的结果，不是对未见 session 的证据；month/year 的 checkpoint 是我们按作者冻结配方重训的（seed 42）。
5. **输入表示与 FA/PCA 维数的影响很大（审查前被低估）**：把 d 从固定 K=10 改成 source 上选（实际选到 d=N 或 5–10），days 的 Stabilizer B=33 从 0.435 升到 0.667（sorted unit）/ 0.745（electrode-pooled，第二次审查后；第一轮 0.716），岭 CCA 从 0.583 升到 0.727。electrode-pooled 输入对 Stabilizer 有利（days 0.745 对 0.667），对 CCA 不利（plain CCA 0.663 对 0.682）。但在 month / year，Stabilizer 即使有 pooled 输入和整段数据（`stab_pool_b` 0.376 / 0.087）也很弱，而且 pooled 的 `(b)` 行只比 `(a)` 行高 0.05–0.07（month 0.376 对 0.304，year 0.087 对 0.034；sorted-unit Stabilizer 的 `(b)` 与 `(a)` 为 0.291 对 0.280 / 0.055 对 0.070），所以这里的瓶颈不是数据量，也不是桥（pooled d=10 的桥上限 0.855）；评测不能判断原因（unit 集合每天变化、3 个月/1 年的子空间漂移都可能）。`*_noalign` 对照（同配置、不旋转）在所有 stratum 为 −0.19…0，说明 Procrustes 旋转确实在起作用。**Stabilizer 的 stable-channel 丢行步骤在所有最终配置里没有运行（§2.1）；d=N 时 Stabilizer 是满秩 FA + Procrustes，与原文相差很远。**
6. **NoMAD（非可部署、违反 H4、target 端有梯度）**：**初版“3/4 个 target 对齐失败”的结论撤回**——它主要来自 target 列顺序与 day-0 不一致（§2.4）。修正后（列与电极对齐）：electrode-pooled 输入下 NoMAD 的 query R² 为 days +0.457 / +0.457（两个 target，3 个 seed，seed 间 ±0.003–0.010）、month +0.229、year −0.383；电极重排（units_elec）为 days +0.532 / +0.497、month +0.101、year −0.219。**NoMAD 在这 4 个 target 上都低于 P0**（B=8 的 P0 为 0.730 / 0.717 / 0.565 / 0.256），而且它用的是整段 target 的无标签数据并反传（违反 H4）。baseline day 自己的对照：未打乱 +0.534（days）/ +0.758（month、year），打乱列后 −0.486 / −0.916，说明它对列顺序极其敏感，初版的结果是输入错位。两种输入表示在不同 stratum 上排序不一致（days、year 上 units_elec 更好，month 上 pooled 更好），所以 NoMAD 的数字依赖输入表示，只能作粗略参照；year 上两种表示都是负 R²，原因本评测没有隔离。`INIT_READIN_FROM_DAY0=False` 的敏感性无法做：该配置键在作者代码里从未被读取（day-k 模型由 day-0 模型目录的拷贝构造），True/False 的 run 逐位相同（实测 days 两个 target、month、year 的 seed-0 结果与 epoch 数完全一致），要做“随机初始化 read-in”需要改作者库代码，没有做。NoMAD 现在用到 `unit_to_electrode`。**NoMAD 自己的 baseline-day 上限（in-sample）在 days 只有 0.534（pooled；对照见 §3.3 (ii)），低于 P0 的 ~0.72（baseline-day 对照见 §3.3 (ii) 与上面的 NoMAD 对照表），所以 days 上 NoMAD 低于 P0（pooled 0.457，units_elec 0.497–0.532）说明不了对齐好不好——它在没有任何跨日偏移的 baseline day 上本来就到不了 P0 的水平；只有 month（baseline-day 上限 0.758）的比较有信息量（pooled 0.229，低于该上限与 P0 的 0.565）。**这些结果只说明 NoMAD 在这种“每天重新分选的少量 sorted unit”数据上的表现，不能推广到论文自己的数据。
7. **微调参照行**：`ui_reg`、`P0 + ui_reg`（反传穿过 trunk，不可部署）的数字见 §3.1；P0+ui_reg 在 days 与 P0 持平，month / year 在 B≥4 高于 P0（P3 的 gap），R3 在 month / year 恢复其中的大部分。
8. **口径问题对 APST 影响很大（§2.3）**：我们和 P 行对 trial 内全部 bin 打分，APST 只覆盖 50-bin 窗口完全在 trial 内的 bin；同一批预测在 native 子集上 days B=33 从 0.590 升到 0.814。用作者协议（前 M 个 trial + Q50）复现作者 dev 数 0.855956 一致，说明适配器本身没有问题。主表给“我们的口径”，native 表给“APST 的口径”，两张表里所有能重算的方法（P-prior、P0、Stabilizer、CCA、NoMAD、APST）都在两个口径下报告；R1-stream / R3 没有 native-query 版本。
9. **没有运行 / 没有做的内容**：APST ACT-only 在 2014 roster（month/year）上没有训练；R1-stream / R3 没有 native-query 版本；APST 的 first-M 协议下的 B=33 与我们 B=33 的 draw 不同，只作协议说明；Stabilizer 的 stable-channel 丢行没有被检验（§2.1）；NoMAD 没有放宽早停试验；没有对 d、相位数、对齐做 target 上的敏感性扫描（按规则不在 target 上调）。
10. **自检（均通过）**：P-prior/P0 在本评测里重新前向的 query R² 与 P1 存档行逐行一致（最大差 5.8e-5，`native_p_reproduction_check.csv`）；APST 适配器的受限预测与作者 `predict_record` 在 6000-bin 前缀上最大差 8.6e-6；作者 first-M 协议复现作者 dev 数；所有 `check_physical_units` 通过；桥对 baseline day 自己做 sanity（d=N 的 sorted-unit 行精确还原 source table 的 0.884 / 0.918）；单元测试（electrode pooling 的 unit-order 不变性、列顺序 guard 在不对齐/打乱时报错、K×d 桥在参考维数 d=N 时精确、`resolve_d` 不被 target 截断、target 有效维数更少时的 Stabilizer / CCA 行为等）通过；target 行与桥上限使用同一个参考维数 d（第二次审查后）。没有读 final session，没有读 `results/official_evalai/`。


## 1. 协议（与 P1/P3 的 P 行逐项对齐）

| 项 | 内容 |
|---|---|
| 数据 | DANDI 000688 sub-M CO，三个 stratum：days（targets 20150616 / 20150617；backbone `ap_pool_x8`）、month（20140626）、year（20150511）（后两者各一个 target，共用同一批 6 个 2014 source session，见 P1 README）。**没有打开任何 final session（0623/0625/0626），没有读 `results/official_evalai/`。** |
| backbone | 冻结的 K=10 AP backbone（`D1_POOLS[...]['ap']`），解码 = `trunk(z)`，`z = T·((x−mean)/std)+c`；P-prior / P0 / R1-stream / R3 / ui_reg 的 `T` 是 K×N read-in 表（c=0），baseline 的 `T,c` 是它们各自方法给出的仿射 read-in。 |
| 校准子集 | **完全复用** P1/P3 的 `draw_trials(target, B, draw, n_trials)`（`zlib.crc32(f"{target}-{B}-{draw}")` 播种的 `rng.choice`，不放回）。B ∈ {1, 4, 8, 16, 33}（P1/P3 另有 B=2，这里不报），draw = 0…7（8 个）。每个 baseline 的 `(target,B,draw)` 行与 P 行逐行配对。 |
| 解码 | 整段 recording 连续因果前向一次（不在 trial 边界重置 state，规则 7）；query = usable trial 内、不在校准 trial 中的全部 eval-valid bin；R² = 物理单位、方差加权（`check_physical_units` 全部通过）。APST 用它自己的 50-bin 因果窗 + full-clock EMA（α=1/3），query bin 与上面完全相同（仅去掉 recording 最前 49 bin）。 |
| CI | 对每个 (方法, stratum, B)：先对每个 draw 取 target 均值（P 行已按 seed 平均；NoMAD 同时按 seed 平均），再对 8 个 draw 做 t 区间 `mean ± t(0.975,7)·sd/√8`。**只含 draw 方差**，不含 target / trunk / source 方差；days 有 2 个 target，month 与 year 各 1 个且共享 trunk（两个 `best.pt` 的 `state_dict` 逐项相同），不是独立重复；整个评测是 4 个 target session / 2 个 trunk / 1 只猴子 / 1 个任务。baseline 的闭式方法没有训练 seed（FA 的 5 次重启固定种子 0），所以没有 seed 维度。`*_b` 行与 NoMAD（对齐不依赖 B）在不同 B、draw 之间只有 query mask 不同，CI 因此极窄，**不是真正的不确定性**。配对差 = baseline − P0（正 = baseline 更好）。 |
| 选择规则 | 所有 baseline 超参数只在 **source** 上选：pseudo-target = backbone 的 6 个 source session 中除 baseline day 外的 5 个（B∈{4,16}，每个 2 个 draw，种子 `val-<session>`，与 P1 的 source val 同一套）；打分 = pseudo-target 的 query R²（经冻结 trunk）。pseudo-target 是 trunk 的训练 session，因此这个分数只用于**排序**，不是 target 预期。待选项：FA/PCA 维数 d ∈ {3,5,8,10,N}、Stabilizer 的 floor / 稳定通道比例、CCA 的岭系数 / 输入变换 / 相位数 / 对齐（§2.1、§2.2）；输入类型（sorted unit / electrode-pooled）不选，两种都报。整个网格在任何 target 行评测之前固定。month 与 year 共用同一个 trunk、source 与 pseudo-target，所以选出的配置相同（`selected_configs.csv`）。R1-stream / R3 行用 P3 为各 stratum 固定的选择集（days：IS；month：因果 OOS `oosc`；year：OOS），不是本评测重新选的。`selection_readin.csv` 给出全部网格。 |
| baseline day | 所有 read-in 对齐方法（Stabilizer、CCA）与 NoMAD 的“第 0 天” = backbone 的**最近一个 source session**（days：20150615；month/year：20140306），按日期的固定规则，不看 target。 |
| 约束 | H1：trunk 全程冻结（baseline 不反传 trunk，也不改 trunk）；H2：除 NoMAD 的 alignment network 与 ui_reg 参照行外，target 端没有任何梯度；H4：标成 `(a)` 的行只用 B 个校准 trial 的数据；`(b)` 行用整段 target recording 的数据，**违反 H4**，只作“论文自己的数据量”参照，不可部署。 |

**两个必须先说明的比较口径问题：**

1. **P 行与 baseline 行使用的 target 数据不同**：P-prior 用 B 个 trial 的神经数据（只估输入 mean/std）；P0 / R1-stream / R3 用 B 个 trial 的神经数据 + 行为标签（cross-covariance、方向调谐等描述量）；Stabilizer 只用神经数据，不用任何标签；CCA 与 APST-FULL 用 B 个 trial 的神经数据 + 每个 trial 的 reach 方向标签（任务元数据，不是行为轨迹；CCA 另用 go cue 时间）；APST ACT-only 只用神经数据。表中“target data used”列逐行写明。
2. **P 行依赖电极映射 `unit_to_electrode`**（P1 审查 §0.1.5：打乱电极后 P0 B=8 从 0.73 掉到 0.59，随机分配 0.21）；**审查后**，Stabilizer（sorted-unit 的配对规则与 pooled 输入）、Gallego-style 的 pooled 变体和 NoMAD（pooled / units_elec）也用它；sorted-unit 输入的 CCA 与 APST 不用。


## 2. 各 baseline：引用、实现与简化、超参数、target 数据、是否反传

### 2.1 Stabilizer（Degenhart et al. 2020）

- **引用（已核实）**：Degenhart AD, Bishop WE, Oby ER, Tyler-Kabara EC, Chase SM, Batista AP, Yu BM. *Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity.* Nature Biomedical Engineering 4(7):672–685 (2020), doi:10.1038/s41551-020-0542-9（Europe PMC 记录 PMC7822646；Nature 页面）。
- **实现来源**：作者的 MATLAB 配套代码（github.com/alandegenhart/stabilizedbci，`fitBaseStabilizer.m / updateStabilizer.m / alignLoadingMatrices.m / identifyStableLoadingRows.m / learnOptimalOrthonormalTransformation.m / getStabilizatonMatrices.m / stabilizationExample.m`，逐个读过），移植到 numpy（`ssm_decode/published_baselines.py`）：(i) baseline day 与新一天各拟合 FA（EM，5 次随机重启取最优似然，私有方差下限）；(ii) 用 `identifyStableLoadingRows` 的迭代 Procrustes 丢行法在 `n_stable` 个 stable channel 上求正交变换 `W`（行范数阈值 `ALIGN_TH=0.01`），把新一天的载荷旋转到 baseline 的 FA 空间 `C_new ← C_new Wᵀ`；(iii) 潜变量 `l = βy+o`，`β = Cᵀ(CCᵀ+Ψ)⁻¹`。label-free。
- **两种输入（审查后新增）**：
  1. **electrode-pooled（忠实输入，主行）**：把同一电极上的 unit 计数相加，输入的列是固定的电极通道（`ssm_decode/electrode_align.py::pool_counts`；通道 `e` 在每个 session 里都是电极 `e`；96 个 M1 电极的布局，但实际有信号的通道很少：target 上有信号的通道：days 12 个（baseline day 13 个）、month 20 个、year 17 个；与 baseline day 配对且两边都有信号的通道 11 / 15 / 11 个），不需要任何 unit 配对规则。**输入不是论文用的 96 通道阈值穿越（threshold-crossing）计数：本数据集只发布 sorted unit 的 spike，所以这里是“每个电极上的 sorted spike 相加”，是数据集强加的近似；与论文的固定通道输入同构（固定的、跨 session 一致的通道），但不是同一种信号。**baseline day 与 target 上无发放的通道（方差为 0）不参与 FA、不构成 stable channel 配对。**这个输入用到了 `unit_to_electrode`（与 P 行相同的元数据）。**
  2. **sorted-unit（原版本，降为变体）**：stable-channel 对应关系用 **同一电极上的 unit 按平均发放率排序配对**（ad-hoc 规则，label-free；days 目标日 13–15/17 个 unit 能配上，month 19/26，year 14/24），配不上的 unit 视为 unstable。该规则是我们加的，不是原文方法。
- **与原法不同之处（全部是为了接到冻结的 AP trunk 上）**：
  1. 原文的 decoder 是在 baseline 日 FA 潜变量上训练的 Kalman filter。这里 decoder = 冻结 trunk，FA 潜变量经一个 **K×d 最小二乘桥 `A`（+偏置）** 映射到 AP 潜变量（K=10 是 trunk 的潜变量维数，d 是 FA 维数）；`A` 只在 baseline source session 上拟合（该 session 的 AP 潜变量 `W_ref·x` 是已知的），不看 target。**审查前 d 固定为 K=10；审查后 d ∈ {3,5,8,10,N} 在 source pseudo-target 上选，N = 输入的全部有效维数（d=N 时桥是精确的）；**第二次审查后：d 只由 baseline day 决定（N = baseline day 的有效维数，或选出的数值 d），不再被 target 的有效维数截断**；target 的有效维数少于 d（sorted unit 变少、B 很小）时，target 的 FA / PCA 潜变量是秩亏的（缺的维度载荷恒为 0），由 Procrustes / CCA 映到 d 维的 baseline 参考潜空间，桥 `A` 始终是 baseline day 上拟合的 K×d 桥。（审查前的代码把 baseline day 的 FA/PCA 与桥也重拟合到被截断的 d：month 的 d=N 行实际在 d=26 运行、year 在 d=24，baseline day 的潜变量 R² 经桥只剩 0.79–0.83，而表里写的是 d_eff=43、“精确”。）**桥的上限见 §3.3 (i)。
  2. FA 的输入先按各自拟合数据做 per-column z-score，于是作者的 `MIN_PRIV_VAR=0.1`（原始计数单位）变成“单位方差的 10%”；EM 迭代上限 400、相对容差 1e-5（作者 1e5 次/1e-5）。
  3. 原文在线持续更新 stabilizer；这里只做一次更新（离线）。
- **超参数（只在 source 选，整个网格在任何 target 行评测之前固定）**：`MIN_PRIV_VAR ∈ {0.01, 0.1}`，`ALIGN_N = {0.5, 0.75, 1.0} ×（配对通道数）`，`d ∈ {3,5,8,10,N}`（每种输入 30 个配置）；选出的组合见 `selected_configs.csv`（§3.3）。
- **必须说明：stable-channel 丢行步骤在所有最终配置里都没有真正运行。**代码沿用作者的下限 `n_stable = max(round(frac·m), min(m, d))`（m = 配对通道数）：当 d ≥ m 时（days 与 sorted-unit 的 month/year 选到 d=N）`n_stable = m`，即不丢任何一行；pooled 的 month/year 选到 frac=1.0，同样不丢行。**因此 d=N 时这里的 Stabilizer 是满秩 FA（d = 输入有效维数）+ 全部配对通道上的 Procrustes 旋转，没有丢行，与原文（低维 FA、识别并丢掉不稳定通道）相差很远；它只能说明“对齐 FA 载荷”这一部分，不能代表原 Stabilizer。**核对全部 `stab_a` / `stab_pool_a` 评测行：`n_stable == n_matched` 的比例是 100%（每个 stratum 都是）。所以这里的 Stabilizer 实际是“全部配对通道上的 Procrustes 旋转”，原文的 stable-channel 识别部分没有被检验；其重要性无法由本评测判断。
- **target 数据**：(a) B 个校准 trial 的神经计数（无标签）；(b) 整段 target recording 全部 usable trial 的神经计数（无标签，包括 query bin 的输入，transductive，违反 H4）。**target 反传：无**（EM + SVD）。
- 对照 `*_noalign`：与 (a) 行**完全相同的配置**（floor、d；审查前该对照用了与主行不同的 floor，已改），只是不做 Procrustes 旋转。

### 2.2 PCA + CCA 潜动力学对齐（Gallego et al. 2020）

- **引用（已核实）**：Gallego JA et al. *Long-term stability of cortical population dynamics underlying consistent behavior.* Nature Neuroscience 23(2):260–270 (2020), doi:10.1038/s41593-019-0555-4（PubMed 31907438；方法描述以 bioRxiv 10.1101/447441v3 全文为准，已读）。
- **实现**：与论文 Methods 一致的部分——对计数做 √ 变换；PCA 取前 d 个主成分；用 CCA（QR + SVD）找两天潜动力学之间使相关最大的线性变换，把新一天的潜变量映到第 1 天（baseline day）的坐标；再用第 1 天的 decoder。
- **命名约定（审查后统一）**：“**Gallego-style**”只指忠实版本 = **√ 变换 + plain CCA（岭系数 0）**，有 sorted-unit 与 electrode-pooled 两种输入；其余行（raw 计数、岭 CCA）一律标成“our variant”。审查前正文在不同地方把岭 CCA 也叫 Gallego-style，已更正。
- **与论文不同之处**：
  1. 论文用整日（按 target 匹配 trial 数、按时间点对齐的）单 trial 潜动力学做 CCA。这里按任务要求只能用 B 个有标签 trial：每个 trial 的轨迹按 **reach 方向（NWB trials 表的 `target_dir`，8 个方向）取条件平均**，在两天都出现的方向上拼接成 CCA 样本；B 小时只有 1–几个方向，样本数 = 方向数×相位数，**CCA 严重欠定**（这是该方法在小 B 下的真实困难，不是 bug）。
  2. **轨迹的相位数与对齐方式（审查后在 source 上选）**：相位数 ∈ {10, 20, 40}；对齐 ∈ {整个 trial 时间归一化, go cue 之后 50 个 bin（1 s）窗口}。NWB trials 表有 `go_cue_time` 但没有 movement-onset 事件（movement onset 只能从行为轨迹推出，那等于用了行为标签），所以“对齐到运动起点”用 go cue 近似；go cue 时间是任务事件元数据，与 reach 方向标签同属“任务元数据”，不是行为轨迹。
  3. 论文用 50 ms 高斯平滑；这里没有非因果平滑（相位分段平均相当于平滑），解码时的 read-in 是逐 bin 的因果线性映射。
  4. decoder = 冻结 trunk，经 baseline 日的 K×d 最小二乘桥 `A`（PCA 潜变量 → AP 潜变量）。
  5. 论文没有正则；除 `plain`（岭参数 0，忠实版）外，另给一个 **岭正则 CCA 变体**（我们加的，在 source 上选岭系数），因为 plain CCA 在 B 小时数值上近乎奇异。
- **超参数（只在 source 选，网格在任何 target 行评测之前固定）**：输入变换 ∈ {raw, √}（√ 是论文的预处理；raw 是变体；electrode-pooled 只做 √）；d ∈ {3,5,8,10,N}；岭系数 ∈ {0.01, 0.1, 1, 3, 10, 30, 100}（相对协方差迹；plain 为 0）；第一阶段在（20 相位，整 trial）上选 (d, 岭)，第二阶段在该最优处选（相位数，对齐）——坐标下降，不是全网格。plain 与岭各自独立选；全部网格与选择结果见 `selection_readin.csv` / `selected_configs.csv`。
- **target 数据**：(a) B 个校准 trial 的神经计数 + 每个 trial 的 reach 方向标签和 go cue 时间（不使用行为轨迹）；(b) 整段 target recording 全部 usable trial 的神经计数 + 方向标签（违反 H4，且 query trial 的方向标签也被用到）。**target 反传：无**（SVD/特征分解）。


### 2.3 APST（MZ 自己的方法）

- **引用**：Zhang X, Mo H, Wen P, Liang S, Yang J, Zeng Y, Wang Z, Wang H. *Association profile conditioning in a set-temporal transformer for cross-session intracortical motor decoding.* arXiv:2609.39080 (2026)。**此条目取自本地 `APST/README.md` 的 BibTeX，没有联网核实**（其余三篇的引用已联网核实）。
- **代码**：`/home/xinyuan/Work_host/APST/src/apst/dandi_subm`（作者代码，**未改动**）+ `APST/docs/dandi_subm.md` 与 `examples/dandi_subm/budget/evaluate.py` 的 calibration-budget 评估。APST 的“校准”是前向、闭式的：activity encoder（E0）对 B 个校准 trial 重采样到 100 bin 的神经活动做池化；APST-FULL 另用 B 个 trial 的 MOVE 窗（go cue 后 0.1–0.6 s）计数和 reach 方向拟合 MOVE-T4 association profile（OLS 余弦调谐，需要 ≥3 个方向非共线的 trial，否则按作者协议 `UNAVAILABLE_RANK_LT3`，B=1 总是不可用）；**target 端没有任何梯度**。ACT-only 变体不用方向标签。
- **本评测相对作者协议的唯一改动**：(1) 校准 trial 不是“前 M 个合法 trial”，而是我们的 `draw_trials` 的随机 B-子集（只保留 APST 合法的 trial：rewarded、≥50 bin、go cue 有限且 MOVE 窗在 trial 内；本评测中所有 draw 的 B 个 trial 全部合法）；(2) query 不是作者的 Q50（第 51 个合法 trial 之后、窗口完全在 trial 内的 bin），而是和 P 行相同的“非校准 usable trial 内全部 eval-valid bin”（仅去掉 recording 最前 49 个 bin）。预测仍是 APST 自己的 50-bin 因果窗 + full-clock EMA(α=1/3)；`run_baselines_apst.py` 里的预测函数只把 full-clock 推理限制在 query bin 及其前 16 个 bin（EMA 记忆 (1/3)^16≈2e-8），与作者的 `predict_record` 在 6000-bin 前缀上逐元素一致（最大差 8.6e-6），并在作者自己的“前 M 个 trial + Q50”协议下复现了作者的 dev 数（0616，M=8：0.855956 对 0.855956）。
- **checkpoint 与选择规则（重点）**：**所有 APST checkpoint 都是单个训练 seed（42）；days 的 20150616 / 20150617 是 APST 自己的 dev session（作者的 checkpoint 选择与配方设计都见过它们）；下面 days 的 APST 行和 native-query 表里 days 的“高于 P0”都必须按这个意思读。**
  - days：作者发布的 sub-M 2015 campaign 的 `sua_dual_site_F0_s42` / `sua_act_only_F0_s42`（6 个 2015 source = 我们 days 的 source，**同一批 session**）。作者的选择规则是“在 dev=0616/0617 上 earliest-max”（= 我们的 days target），**对我们是在 target 上选**，所以主行用 **最后一段（segment_24，不做选择，只靠 source 训练的固定 24×3165 更新计划；但配方本身的设计见过这两个 dev session）**；dev 选出的段（FULL=segment_09，ACT=segment_08）单独标成“选择见过 target”的乐观参照行。
  - month / year：作者没有 2014 的 checkpoint；用作者**冻结的配方**（24×3165 更新、seed 42、F0+FiLM、MOVE-T4 carrier）在我们 month/year 的 6 个 2014 source 上重训（`scripts/apst_2014_train.py` 只覆盖 `protocol.TRAIN_SESSIONS/DEV_SESSIONS` 两个常量；dev = 20140307，介于 source 与两个 target 之间，既不是 target 也不参与评估），同样取最后一段；另给“在 source-OOS 的 20140307 上 earliest-max 选出的 segment”（FULL：segment_08，0307 上 dev R² 0.798，而最后一段 segment_24 为 0.744；这是**只看 source 侧**的选择，因此对 2014 是合规的）。重训 FULL 的 concat 预训练 + dual_site 训练各约 1.5–2.7 GPU 小时（与评测并行，单卡共享）。**ACT-only 在 2014 上没有训练**（不影响 FULL 行；理由：单卡 GPU 时间；它是 label-free 的消融，已在 days 上给出）。
- **超参数**：全部是作者冻结的配方，没有任何我们调的参数，没有在 target query 上调任何东西。
- **target 数据**：APST-FULL：B 个校准 trial 的神经活动 + 每 trial 的 reach 方向；ACT-only：只有神经活动。**target 反传：无。**

#### APST 的 query 口径问题（影响解读）

APST 的训练和评测只看“50-bin 窗口完全落在 trial 内”的 bin（Q50）。我们的协议和 P 行一样对 trial 内**全部** bin 打分，trial 起始的前 49 个 bin 的 50-bin 窗口会跨过 trial 边界（含上一个 trial 的尾部和 ITI），对 APST 是分布外输入。因此 APST 在“我们协议”下的 R² 明显低于它自己的协议。为了公平，我们**同时**报告：(i) 主表（我们的协议，全部 query bin）；(ii) “native-query 子集”表：只在窗口完全落在 trial 内的 query bin 上打分，**所有能重算的方法（P-prior、P0、Stabilizer、CCA、NoMAD、APST）都在这个子集上重算**（R1-stream / R3 的逐 bin 预测没有保存，不在 native 表中）。

### 2.4 NoMAD（Karpowicz et al.）——可选参照，**不可部署 / 违反 H4**

- **引用（离线未核实）**：Karpowicz BM, Ali YH, Wimalasena LN, Sedler AR, Keshtkaran MR, Bodkin K, Ma X, Rubin DB, Williams ZM, Cash SS, Hochberg LR, Miller LE, Pandarinath C. *Stabilizing brain-computer interfaces through alignment of latent dynamics.* Nature Communications 16:4662 (2025)。**卷/页码/年份没有联网核对（初版曾标“已核实”，已撤回）**；前身为 bioRxiv 10.1101/2022.04.06.487388（2022，本地仓库 README 引用的版本）。
- **代码**：`/home/xinyuan/Work_host/nomad`（作者的 `lfads_tf2` + `nomad`，库代码**未改**）。仓库固定 TF 2.0 / Python 3.7 / CUDA 10，RTX 3090 不能用，所以在新环境 `/mnt/data/work_host/envs/nomad_tf26`（Python 3.8、TF 2.6.0、tfp 0.12.2、tfa 0.14.0、pandas 1.0.5、numpy 1.19.5）里**在 CPU 上**运行（没有占用 GPU）。TF 2.0→2.6 需要 6 处运行时兼容补丁，全部写在 `scripts/run_baselines_nomad_tf.py` 的 monkeypatch 里：git 仓库桩、`ClippedGRUCell` 序列化（仅 TF≥2.10 需要）、autograph 对 `train_df[...]=` 的处理、`AlignLFADS._graph_step` 的 input_signature 多了一个 spec、`np.zeros(tf.shape(...))`→`tf.zeros`、控制器输出先验的 `tf.map_fn` 改成逐步展开的循环（**同一个 AR(1) 对数概率公式**）。
- **审查发现的错误（CRITICAL，已修）**：NoMAD 默认 `INIT_READIN_FROM_DAY0: True`，即把 target 的 `lowd_readin` 初始化成 day-0 的 read-in，这要求 target 的第 c 列和 day 0 的第 c 列是**同一个通道**。000688 sub-M 每天重新分选 unit，初版把 target 的 unit 按它们自己的列序零填充到 day-0 的宽度（`pad_units`），于是 target 的第 c 列和 day-0 的第 c 列是不同电极上的不同 unit（0617：17 列里 11 列对不上），alignment network 从一个错误的初始化出发，结果（+0.48 / −0.75 / −0.90 / −0.63）主要反映的是这个输入错位，而不是 NoMAD 本身的对齐能力。审查者把 target 列按电极重排后 0617 → +0.503、0626 → +0.088（`/tmp/bl_review/`），并且把 baseline day 自己的列打乱后 −0.88，证明模型对列顺序极其敏感。
- **修正（本版）**：
  1. **主变体 `pooled`（electrode-pooled 输入）**：每个 session 把同一电极上的 unit 计数相加，列是固定的电极通道（96 个 M1 电极的布局，`electrode_align.pool_counts`；输入是每个电极上 sorted spike 相加，不是论文的阈值穿越计数，是数据集强加的近似）；第 c 列在每个 session 里都是电极 c，day-0 的 read-in 初始化因此有意义。这是这类论文的忠实输入（固定电极通道的多单元活动）。day-0 LFADS 与 decoder 在 pooled 输入上**重新训练**（DATA_DIM=96；tag `days_pool` / `y2014_pool`）。
  2. **变体 `units_elec`**：保留 sorted-unit 的 day-0 模型（DATA_DIM=17/43），把 target 的 unit 按同电极、发放率排序配进 day-0 的列（`electrode_matched_order`；没配上的 unit 任意填进空列，多出的 unit 丢弃），即审查者用的重排。
  3. **列顺序 guard**：`electrode_align.assert_columns_electrode_aligned` 在 `INIT_READIN_FROM_DAY0=True` 且 target 列的电极与 day-0 同位置列不一致时直接报错（对初版的 `units` 布局会报错：0617 为 11/17 列不一致，已实测）；`--legacy-unsafe` 才能绕过。单元测试覆盖：pooled 输入对 unit 顺序不变、guard 在不对齐/打乱列时报错、对重排后的布局通过。
  4. **对照**：把 baseline day 自己的活跃列随机打乱（模拟重新分选）后走一遍 alignment network（`--shuffle-control`，guard 绕过），**必须失败**，用来展示对列顺序的敏感性；同时跑未打乱的 baseline day（pipeline 的上限）。
  5. **敏感性**：`INIT_READIN_FROM_DAY0=False`（pooled，4 个 target 各 1 个 seed）。
  6. **NoMAD 现在用到 `unit_to_electrode`**（和 P 行、Stabilizer 的 pooled 变体一样的电极元数据），这一点在比较时必须说明：它不是“纯无元数据”的方法。
- **流程（对应作者 demo）**：(1) 第 0 天 = baseline day（与 Stabilizer/CCA 相同的 source session）：在其连续 recording 上切 30-bin 窗口（步长 24，每 5 个窗口取 1 个做验证）训练 `BehaviorLFADS`（脉冲 + 标准化速度的行为头；demo 的网络规模：GEN/CON/IC 100、CO 4、FAC 30、ENC_INPUT 50，早停 patience 10）；(2) decoder = 对第 0 天因果 `gen_states` 的 3 阶滞后岭回归（demo 的 `fit_and_eval_decoder`，α 网格交叉验证）；(3) 每个 target：冻结 LFADS，训练 alignment network（FFN-KL，隐层 50，`gen_states` 上的 KL + Poisson NLL，demo 的 nomad.yaml，patience 10），**训练数据 = 该 target session 的全部无标签脉冲窗口**；(4) 对整段 target recording 做 30-bin 因果滑窗（步长 1）取每个窗口最后一个 bin 的 `gen_states`，过第 0 天 decoder 得速度，转回物理单位；(5) 在与其它方法相同的 `(target,B,draw)` query mask 上打分。
- **与 demo 的差别**：DATA_DIM = baseline day 的列数（pooled：96 列的电极布局，其中大部分通道在这些 session 里没有 sorted unit）；per-column z-score 的 `normalization.h5` 由本数据自行生成；LFADS 训练用 graph 模式（demo 在 notebook 里强制 eager）。
- **超参数**：全部取作者 demo 默认，没有调。每个 target 训练 3 个 alignment seed（本版所有 run 都显式设 `tf.random.set_seed`；初版的 seed 0 是未设种的）；第 0 天 LFADS 与 decoder 每个 baseline day 只训练一次，所以 3 个 seed 共享同一个 day-0 模型。
- **target 数据**：整段 target 的无标签脉冲（包括 query bin 的输入，transductive）；**target 反传：有**（alignment network 与 `lowd_readin / generator / rate_linear / fac_linear` 被训练）。因此是 **非可部署、违反 H4** 的参照行；对齐不依赖 B，行内不同 B/draw 之间只有 query mask 不同，没有真正的 CI。


### 2.5 微调参照行（复用，不重算）

`ui_reg`（从 electrode-aligned prior 起、L2 拉回 prior 的逐 target 梯度拟合，反传穿过冻结 trunk；P2 A′2）、`P0 + ui_reg`（P2 规则，从 P0 的表起）以及 month / year 的 like-for-like 版本（P3 R.12，`fix/uiref`）直接取自 `scripts/analyze_pc3_fix.py::reference_rows()`（seed 按 draw 平均）。它们使用相同的 `(target,B,draw)`、相同的 B-trial 归一化、相同的 query 与 R²。**target 端有梯度（穿过 trunk），不可部署**，只是梯度参照。R1-stream / R3 取自 P3 R.12 的 headline 选择集（`analyze_pc3_fix.candidate_rows`，seed 平均；days：IS；month：因果 OOS；year：OOS），均无 trunk 反传。


## 3. 结果

### 3.1 主表：query R²（物理单位），我们的 query 口径（trial 内全部 eval-valid bin）

读法：每格 `mean ± 95% CI`，对 8 个 draw 的 target 均值做 t 区间（**只含 draw 方差**）；`n/a` = 该格没有可用 draw（APST-FULL 需要 ≥3 个方向非共线的有标签 trial）；`(n=k)` = 只有 k 个 draw 可用。`(a)` = 只用 B 个校准 trial（H4 合规）；`(b)` = 用整段 target recording，**违反 H4，不可部署**，只表示“该方法在论文自己的数据量下”的结果，且 `(b)` 与 NoMAD 的行在 B 之间只有 query mask 不同，CI 不代表不确定性。“published”= 原文方法的忠实版本；“published-variant”= 我们加的正则/输入变换/输入表示变体，不是原文方法。**Stabilizer 与 Gallego-style 各有两种输入：electrode-pooled（96 个固定通道，忠实输入）与 sorted-unit（需要 ad-hoc 配对规则的原版本）。**P 行：P-prior / P0 为 P1（pool O，2 个 seed 按 draw 平均）；R1-stream / R3 按 P3 R.12 的 headline 选择集（days：IS；month：因果 OOS `oosc`；year：OOS）；ui_reg / P0+ui_reg 为 P2/P3 的行（`like-for-like` 行仅 month/year 有）。

**NoMAD 行说明**：初版的 NoMAD 行因 target 列顺序与 day-0 错位而无效，已撤回；下面是修正后的行：`pooled`（主）与 `units_elec`（电极重排），各为 3 个 seed、所有 target 的平均。逐 target、逐 seed 见 §3.3 (ii)。NoMAD 的对齐不依赖 B，不同 B 的差别只来自 query mask，行内 CI 不代表不确定性；NoMAD 现在用到 `unit_to_electrode`。 **APST 行说明**：days 的 `dev-selected` 行是作者在我们的 days target 上选的 segment（乐观）；days 的所有 APST 行都是 APST 自己的 dev session + 单个训练 seed 42；month/year 的 `source-only` 行是 2014 重训后在 source-OOS 的 20140307 上选的 segment；`last segment` 行在三个 stratum 上都是不做选择的固定计划终点。

#### days

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.333±0.025 | 0.360±0.018 | 0.360±0.023 | 0.362±0.011 | 0.361±0.012 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.647±0.021 | 0.708±0.009 | 0.723±0.009 | 0.732±0.005 | 0.738±0.005 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.648±0.021 | 0.710±0.009 | 0.723±0.010 | 0.732±0.005 | 0.740±0.005 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.651±0.020 | 0.714±0.009 | 0.728±0.010 | 0.737±0.006 | 0.747±0.006 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.643±0.040 | 0.713±0.022 | 0.715±0.014 | 0.731±0.007 | 0.745±0.008 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.056±0.050 | -0.046±0.042 | -0.085±0.039 | -0.072±0.041 | -0.068±0.028 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.748±0.000 | 0.748±0.003 | 0.747±0.001 | 0.747±0.003 | 0.746±0.004 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.528±0.035 | 0.634±0.013 | 0.633±0.023 | 0.652±0.011 | 0.667±0.025 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.095±0.061 | -0.191±0.091 | -0.086±0.136 | -0.021±0.152 | 0.000±0.065 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.652±0.001 | 0.653±0.002 | 0.652±0.002 | 0.652±0.004 | 0.653±0.006 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.132±0.177 | 0.358±0.042 | 0.533±0.031 | 0.612±0.014 | 0.674±0.017 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.001±0.083 | 0.418±0.050 | 0.568±0.035 | 0.632±0.017 | 0.688±0.015 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | -0.123±0.133 | 0.341±0.066 | 0.538±0.037 | 0.615±0.026 | 0.682±0.008 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.056±0.149 | 0.415±0.039 | 0.576±0.052 | 0.645±0.018 | 0.702±0.008 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | -0.095±0.169 | 0.383±0.075 | 0.578±0.038 | 0.644±0.025 | 0.708±0.012 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.026±0.122 | 0.448±0.045 | 0.609±0.058 | 0.666±0.017 | 0.727±0.007 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.697±0.000 | 0.698±0.003 | 0.697±0.001 | 0.697±0.003 | 0.696±0.006 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.719±0.000 | 0.720±0.003 | 0.719±0.001 | 0.718±0.003 | 0.718±0.006 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.697±0.000 | 0.699±0.003 | 0.696±0.002 | 0.696±0.002 | 0.697±0.005 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.733±0.000 | 0.734±0.003 | 0.732±0.001 | 0.731±0.002 | 0.732±0.004 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.722±0.000 | 0.723±0.003 | 0.721±0.002 | 0.720±0.001 | 0.723±0.005 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.753±0.001 | 0.754±0.003 | 0.752±0.002 | 0.750±0.002 | 0.753±0.004 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.279±0.118 | 0.500±0.059 | 0.568±0.011 | 0.590±0.009 |
| APST ACT-only (last segment, source-only rule; seed 42) | B trials: neural only (no labels) | no | 0.026±0.068 | 0.178±0.074 | 0.340±0.070 | 0.393±0.083 | 0.437±0.103 |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.311±0.119 | 0.516±0.043 | 0.567±0.006 | 0.588±0.009 |
| APST ACT-only (authors' dev-selected segment_08; days targets are APST dev sessions, optimistic) | B trials: neural only (no labels) | no | 0.041±0.076 | 0.196±0.086 | 0.338±0.080 | 0.396±0.093 | 0.429±0.110 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.457±0.001 | 0.459±0.003 | 0.456±0.002 | 0.455±0.004 | 0.458±0.007 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.515±0.001 | 0.516±0.004 | 0.514±0.002 | 0.514±0.005 | 0.515±0.007 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.378±0.028 | 0.527±0.039 | 0.607±0.038 | 0.661±0.020 | 0.704±0.015 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.654±0.019 | 0.701±0.015 | 0.720±0.015 | 0.730±0.012 | 0.746±0.008 |

#### month

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.315±0.046 | 0.319±0.021 | 0.332±0.010 | 0.328±0.007 | 0.333±0.004 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.466±0.034 | 0.535±0.014 | 0.565±0.011 | 0.558±0.007 | 0.566±0.006 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.472±0.037 | 0.542±0.014 | 0.576±0.011 | 0.588±0.007 | 0.606±0.008 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.492±0.048 | 0.591±0.028 | 0.630±0.011 | 0.652±0.014 | 0.679±0.007 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.154±0.060 | 0.237±0.041 | 0.295±0.030 | 0.334±0.026 | 0.304±0.067 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.108±0.052 | -0.083±0.090 | -0.112±0.086 | -0.088±0.083 | -0.038±0.047 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.376±0.000 | 0.376±0.000 | 0.376±0.001 | 0.376±0.001 | 0.376±0.001 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.263±0.077 | 0.251±0.055 | 0.267±0.044 | 0.283±0.035 | 0.280±0.021 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.064±0.045 | -0.104±0.099 | -0.123±0.103 | -0.115±0.198 | -0.132±0.137 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.290±0.000 | 0.290±0.001 | 0.290±0.001 | 0.291±0.001 | 0.291±0.002 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.013±0.169 | 0.196±0.046 | 0.305±0.043 | 0.374±0.025 | 0.365±0.065 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.063±0.065 | 0.278±0.056 | 0.378±0.025 | 0.430±0.020 | 0.478±0.017 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | 0.032±0.071 | 0.230±0.042 | 0.316±0.053 | 0.356±0.030 | 0.390±0.039 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.047±0.054 | 0.349±0.089 | 0.495±0.046 | 0.583±0.022 | 0.646±0.020 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | 0.038±0.083 | 0.239±0.077 | 0.350±0.038 | 0.435±0.039 | 0.458±0.026 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.060±0.059 | 0.376±0.089 | 0.512±0.050 | 0.597±0.020 | 0.656±0.017 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.422±0.000 | 0.422±0.001 | 0.422±0.001 | 0.422±0.001 | 0.422±0.001 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.466±0.000 | 0.466±0.001 | 0.466±0.001 | 0.466±0.001 | 0.466±0.001 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.335±0.000 | 0.335±0.000 | 0.335±0.001 | 0.335±0.001 | 0.336±0.002 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.689±0.000 | 0.689±0.000 | 0.689±0.001 | 0.690±0.001 | 0.689±0.001 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.363±0.000 | 0.364±0.001 | 0.364±0.001 | 0.363±0.001 | 0.364±0.001 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.698±0.000 | 0.698±0.000 | 0.698±0.000 | 0.699±0.001 | 0.698±0.001 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.219±0.071 (n=5) | 0.271±0.056 (n=7) | 0.324±0.054 | 0.376±0.027 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.240±0.040 (n=5) | 0.290±0.062 (n=7) | 0.342±0.064 | 0.387±0.029 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.230±0.000 | 0.229±0.001 | 0.230±0.001 | 0.230±0.001 | 0.227±0.004 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.101±0.000 | 0.101±0.001 | 0.101±0.002 | 0.101±0.002 | 0.099±0.004 |
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
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.054±0.055 | -0.023±0.057 | -0.004±0.059 | 0.004±0.067 | 0.034±0.040 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.063±0.074 | -0.028±0.027 | -0.079±0.078 | -0.086±0.068 | -0.051±0.168 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.079±0.000 | 0.079±0.000 | 0.079±0.001 | 0.078±0.001 | 0.087±0.014 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.020±0.050 | 0.060±0.028 | 0.061±0.021 | 0.055±0.016 | 0.070±0.019 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.073±0.081 | -0.089±0.082 | -0.012±0.051 | -0.083±0.062 | -0.077±0.109 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.055±0.000 | 0.055±0.001 | 0.055±0.001 | 0.054±0.001 | 0.061±0.011 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.011±0.040 | 0.094±0.024 | 0.101±0.047 | 0.172±0.018 | 0.181±0.037 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.003±0.079 | 0.119±0.028 | 0.164±0.033 | 0.198±0.022 | 0.243±0.046 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | 0.030±0.072 | 0.113±0.030 | 0.155±0.050 | 0.209±0.028 | 0.243±0.053 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.033±0.068 | 0.203±0.054 | 0.265±0.025 | 0.334±0.018 | 0.427±0.074 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | -0.017±0.068 | 0.136±0.044 | 0.195±0.045 | 0.277±0.025 | 0.323±0.051 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.023±0.062 | 0.227±0.055 | 0.289±0.030 | 0.352±0.017 | 0.439±0.076 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.175±0.000 | 0.175±0.001 | 0.174±0.001 | 0.173±0.001 | 0.193±0.033 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.221±0.000 | 0.221±0.000 | 0.220±0.001 | 0.219±0.001 | 0.244±0.041 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.317±0.000 | 0.317±0.001 | 0.316±0.001 | 0.314±0.001 | 0.349±0.059 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.412±0.000 | 0.412±0.000 | 0.411±0.001 | 0.408±0.000 | 0.454±0.077 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.270±0.000 | 0.270±0.000 | 0.269±0.001 | 0.267±0.001 | 0.298±0.052 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.412±0.000 | 0.411±0.000 | 0.410±0.001 | 0.408±0.000 | 0.454±0.077 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.105±0.039 (n=7) | 0.150±0.034 | 0.189±0.016 | 0.213±0.047 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.149±0.042 (n=7) | 0.189±0.025 | 0.234±0.018 | 0.257±0.050 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.378±0.000 | -0.375±0.001 | -0.376±0.003 | -0.374±0.002 | -0.414±0.072 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.216±0.000 | -0.215±0.001 | -0.215±0.003 | -0.214±0.002 | -0.237±0.043 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.191±0.021 | 0.267±0.025 | 0.290±0.046 | 0.367±0.007 | 0.478±0.088 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.228±0.023 | 0.321±0.030 | 0.309±0.029 | 0.393±0.011 | 0.488±0.086 |
| P0 + ui_reg (like-for-like: month oosc / year oos) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.230±0.026 | 0.345±0.040 | 0.348±0.017 | 0.405±0.009 | 0.492±0.087 |



### 3.2 native-query 子集：只在 50-bin 因果窗口完全落在 trial 内的 query bin 上打分（APST 自己的口径）

同一批预测、同一批 `(target,B,draw)`，只是去掉每个 trial 起始的前 49 个 bin。P-prior / P0 在这里是**重新前向**得到的（与 P1 存档的全 query R² 逐行一致，见 `native_p_reproduction_check.csv`）。R1-stream / R3 没有保存逐 bin 预测，不在此表。**days 的 APST 行是 APST 自己的 dev session（单个训练 seed 42）。**

#### days

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.250±0.029 | 0.275±0.021 | 0.273±0.026 | 0.273±0.011 | 0.272±0.014 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.643±0.026 | 0.712±0.012 | 0.728±0.011 | 0.737±0.006 | 0.744±0.007 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.617±0.045 | 0.695±0.022 | 0.693±0.018 | 0.711±0.007 | 0.728±0.010 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.083±0.057 | -0.072±0.046 | -0.113±0.044 | -0.092±0.052 | -0.090±0.039 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.727±0.001 | 0.727±0.004 | 0.726±0.002 | 0.726±0.003 | 0.725±0.006 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.498±0.030 | 0.607±0.016 | 0.611±0.026 | 0.628±0.010 | 0.646±0.028 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.097±0.063 | -0.211±0.108 | -0.086±0.126 | -0.009±0.132 | -0.007±0.064 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.621±0.001 | 0.622±0.003 | 0.620±0.003 | 0.621±0.005 | 0.624±0.008 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.182±0.209 | 0.285±0.045 | 0.481±0.038 | 0.564±0.018 | 0.637±0.021 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.044±0.082 | 0.354±0.059 | 0.516±0.041 | 0.588±0.020 | 0.653±0.018 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | -0.167±0.170 | 0.266±0.070 | 0.487±0.043 | 0.569±0.034 | 0.649±0.009 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.104±0.173 | 0.349±0.039 | 0.529±0.061 | 0.607±0.023 | 0.672±0.011 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | -0.135±0.196 | 0.321±0.082 | 0.543±0.045 | 0.610±0.032 | 0.689±0.013 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.061±0.150 | 0.395±0.048 | 0.572±0.069 | 0.641±0.023 | 0.710±0.011 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.664±0.000 | 0.665±0.004 | 0.665±0.002 | 0.664±0.003 | 0.664±0.008 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.686±0.000 | 0.688±0.004 | 0.686±0.002 | 0.686±0.004 | 0.685±0.007 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.676±0.001 | 0.678±0.004 | 0.675±0.003 | 0.674±0.002 | 0.677±0.007 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.710±0.000 | 0.712±0.005 | 0.709±0.002 | 0.708±0.002 | 0.710±0.005 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.714±0.000 | 0.716±0.004 | 0.713±0.002 | 0.711±0.002 | 0.716±0.007 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.739±0.001 | 0.741±0.005 | 0.738±0.002 | 0.735±0.002 | 0.740±0.006 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.393±0.163 | 0.700±0.083 | 0.789±0.012 | 0.814±0.011 |
| APST ACT-only (last segment, source-only rule; seed 42) | B trials: neural only (no labels) | no | 0.041±0.094 | 0.257±0.096 | 0.476±0.100 | 0.546±0.116 | 0.606±0.138 |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.449±0.164 | 0.731±0.059 | 0.799±0.010 | 0.822±0.010 |
| APST ACT-only (authors' dev-selected segment_08; days targets are APST dev sessions, optimistic) | B trials: neural only (no labels) | no | 0.062±0.105 | 0.290±0.111 | 0.477±0.115 | 0.560±0.128 | 0.607±0.147 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.468±0.001 | 0.469±0.004 | 0.466±0.002 | 0.466±0.004 | 0.469±0.009 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.544±0.001 | 0.545±0.005 | 0.543±0.002 | 0.543±0.005 | 0.546±0.011 |

#### month

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.293±0.051 | 0.297±0.021 | 0.312±0.011 | 0.308±0.011 | 0.311±0.005 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.463±0.043 | 0.538±0.017 | 0.574±0.011 | 0.567±0.009 | 0.571±0.009 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.146±0.073 | 0.234±0.041 | 0.303±0.041 | 0.353±0.032 | 0.310±0.079 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.132±0.059 | -0.119±0.113 | -0.150±0.102 | -0.122±0.097 | -0.049±0.047 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.410±0.000 | 0.410±0.001 | 0.410±0.001 | 0.411±0.001 | 0.410±0.002 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.246±0.099 | 0.248±0.066 | 0.264±0.055 | 0.284±0.043 | 0.279±0.027 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.081±0.055 | -0.131±0.119 | -0.151±0.110 | -0.147±0.224 | -0.149±0.129 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.299±0.000 | 0.299±0.001 | 0.299±0.002 | 0.300±0.002 | 0.301±0.003 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.027±0.186 | 0.209±0.055 | 0.332±0.045 | 0.408±0.022 | 0.390±0.076 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.055±0.075 | 0.287±0.067 | 0.399±0.034 | 0.442±0.025 | 0.488±0.023 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | 0.021±0.073 | 0.245±0.048 | 0.335±0.065 | 0.384±0.027 | 0.421±0.037 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.046±0.056 | 0.376±0.101 | 0.537±0.043 | 0.623±0.028 | 0.691±0.023 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | 0.027±0.099 | 0.256±0.097 | 0.373±0.052 | 0.473±0.042 | 0.492±0.031 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.058±0.069 | 0.410±0.099 | 0.560±0.048 | 0.644±0.025 | 0.707±0.019 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.464±0.000 | 0.465±0.001 | 0.465±0.001 | 0.464±0.001 | 0.465±0.002 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.461±0.000 | 0.461±0.001 | 0.462±0.001 | 0.462±0.001 | 0.461±0.002 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.375±0.000 | 0.375±0.001 | 0.375±0.001 | 0.375±0.001 | 0.375±0.002 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.718±0.000 | 0.719±0.000 | 0.718±0.001 | 0.719±0.001 | 0.718±0.002 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.362±0.000 | 0.362±0.001 | 0.362±0.001 | 0.362±0.002 | 0.363±0.002 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.733±0.000 | 0.733±0.000 | 0.733±0.001 | 0.734±0.001 | 0.732±0.002 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.284±0.080 (n=5) | 0.338±0.075 (n=7) | 0.394±0.072 | 0.466±0.037 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.306±0.052 (n=5) | 0.357±0.081 (n=7) | 0.417±0.084 | 0.477±0.038 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.234±0.000 | 0.234±0.001 | 0.234±0.001 | 0.234±0.002 | 0.231±0.004 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.067±0.000 | 0.067±0.001 | 0.067±0.002 | 0.068±0.003 | 0.065±0.005 |

#### year

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.187±0.019 | 0.186±0.016 | 0.180±0.014 | 0.183±0.006 | 0.187±0.007 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.273±0.031 | 0.378±0.047 | 0.361±0.044 | 0.416±0.017 | 0.415±0.019 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.097±0.083 | -0.064±0.093 | -0.020±0.085 | -0.010±0.107 | 0.023±0.068 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.085±0.082 | -0.040±0.042 | -0.137±0.121 | -0.129±0.101 | -0.071±0.226 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.109±0.000 | 0.109±0.001 | 0.108±0.002 | 0.109±0.002 | 0.109±0.003 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.002±0.083 | 0.072±0.050 | 0.072±0.039 | 0.065±0.026 | 0.074±0.021 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.137±0.125 | -0.124±0.098 | -0.017±0.082 | -0.110±0.065 | -0.085±0.164 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.040±0.001 | 0.040±0.001 | 0.040±0.002 | 0.039±0.002 | 0.041±0.003 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.024±0.065 | 0.155±0.036 | 0.164±0.074 | 0.272±0.027 | 0.265±0.022 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.000±0.136 | 0.193±0.034 | 0.253±0.051 | 0.302±0.030 | 0.335±0.011 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | 0.047±0.118 | 0.191±0.050 | 0.249±0.071 | 0.337±0.040 | 0.363±0.042 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.055±0.100 | 0.303±0.085 | 0.392±0.038 | 0.499±0.028 | 0.577±0.010 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | -0.044±0.121 | 0.215±0.072 | 0.304±0.065 | 0.432±0.039 | 0.455±0.035 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.035±0.107 | 0.341±0.089 | 0.434±0.044 | 0.533±0.023 | 0.600±0.009 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.292±0.000 | 0.292±0.001 | 0.292±0.001 | 0.291±0.001 | 0.293±0.002 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.335±0.000 | 0.335±0.001 | 0.335±0.001 | 0.334±0.002 | 0.335±0.002 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.474±0.000 | 0.474±0.001 | 0.475±0.001 | 0.472±0.001 | 0.474±0.003 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.621±0.000 | 0.621±0.001 | 0.621±0.002 | 0.620±0.001 | 0.622±0.002 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.440±0.000 | 0.441±0.001 | 0.441±0.002 | 0.438±0.002 | 0.441±0.002 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.628±0.000 | 0.629±0.001 | 0.629±0.002 | 0.628±0.001 | 0.629±0.002 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.200±0.072 (n=7) | 0.283±0.059 | 0.365±0.027 | 0.359±0.032 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.284±0.081 (n=7) | 0.352±0.048 | 0.444±0.032 | 0.433±0.027 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.701±0.001 | -0.699±0.003 | -0.702±0.007 | -0.702±0.003 | -0.697±0.006 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.391±0.001 | -0.391±0.002 | -0.392±0.006 | -0.393±0.003 | -0.390±0.010 |


### 3.3 辅助检查与对照

**(i) 桥的上限（baseline day 自己走一遍）**——Stabilizer / CCA 的“FA 或 PCA 潜变量 → AP 潜变量”K×d 桥本身损失多少（用 baseline day 的全部数据，不涉及任何对齐）。**这里对表中实际使用的每个配置重算**（初版用的是 floor 0.1 + raw 计数 + d=10 的配置，与表中配置不一致，其差距 0.077 / 0.053 / 0.051–0.053 不能代表表里的行）。`d_eff` = 实际 d；d=N 的 sorted-unit 行桥是精确的（差 0）。**第二次审查后 target 侧也用同一个 d（见上），所以这张表的 d_eff 与 target 行实际使用的 d 一致**（审查前 month / year 的 d=N 行在 target 上实际是 d=26 / 24，表里却写 43）。桥是 baseline day 上的量，不含 target 侧信息损失：target 的 sorted unit 少于 d 时，目标日自己的上限更低，本评测没有单独计算。electrode-pooled 输入在 pooled 之后丢失同电极 unit 间的区分，上限低于 sorted-unit；month/year 选到 d=5–10 的行桥有 0.06–0.13 的损失。

| stratum | 方法行 | d_eff | source table 直接解码 | 经桥后的上限 | 差 |
|---|---|---:|---:|---:|---:|
| days | stab_b | 17 | 0.884 | 0.884 | 0.000 |
| days | cca_b | 17 | 0.884 | 0.884 | 0.000 |
| days | cca_b_ridge | 17 | 0.884 | 0.884 | 0.000 |
| days | cca_sqrt_b | 17 | 0.884 | 0.878 | 0.007 |
| days | cca_sqrt_b_ridge | 17 | 0.884 | 0.878 | 0.007 |
| days | stab_pool_b | 13 | 0.884 | 0.864 | 0.020 |
| days | cca_pool_sqrt_b | 13 | 0.884 | 0.853 | 0.031 |
| days | cca_pool_sqrt_b_ridge | 13 | 0.884 | 0.853 | 0.031 |
| month | stab_b | 43 | 0.918 | 0.918 | 0.000 |
| month | cca_b | 8 | 0.918 | 0.844 | 0.074 |
| month | cca_b_ridge | 43 | 0.918 | 0.918 | 0.000 |
| month | cca_sqrt_b | 5 | 0.918 | 0.804 | 0.114 |
| month | cca_sqrt_b_ridge | 43 | 0.918 | 0.915 | 0.003 |
| month | stab_pool_b | 10 | 0.918 | 0.855 | 0.063 |
| month | cca_pool_sqrt_b | 5 | 0.918 | 0.793 | 0.126 |
| month | cca_pool_sqrt_b_ridge | 10 | 0.918 | 0.834 | 0.085 |
| year | stab_b | 43 | 0.918 | 0.918 | 0.000 |
| year | cca_b | 8 | 0.918 | 0.844 | 0.074 |
| year | cca_b_ridge | 43 | 0.918 | 0.918 | 0.000 |
| year | cca_sqrt_b | 5 | 0.918 | 0.804 | 0.114 |
| year | cca_sqrt_b_ridge | 43 | 0.918 | 0.915 | 0.003 |
| year | stab_pool_b | 10 | 0.918 | 0.855 | 0.063 |
| year | cca_pool_sqrt_b | 5 | 0.918 | 0.793 | 0.126 |
| year | cca_pool_sqrt_b_ridge | 10 | 0.918 | 0.834 | 0.085 |

选出的配置（source pseudo-target 上平均 R²）：

| stratum | 配置族 | 选出的配置 | 平均 pseudo-target R² |
|---|---|---|---:|
| days | stab/units | `{"d": "N", "floor": 0.1, "frac": 0.5, "inp": "units"}` | 0.461 |
| days | cca/units/raw/plain | `{"align": "go", "d": "N", "inp": "units", "phases": 20, "ridge": 0.0, "transform": "raw"}` | 0.395 |
| days | cca/units/raw/ridge | `{"align": "go", "d": "N", "inp": "units", "phases": 10, "ridge": 3.0, "transform": "raw"}` | 0.438 |
| days | cca/units/sqrt/plain | `{"align": "go", "d": "N", "inp": "units", "phases": 20, "ridge": 0.0, "transform": "sqrt"}` | 0.360 |
| days | cca/units/sqrt/ridge | `{"align": "go", "d": "N", "inp": "units", "phases": 10, "ridge": 1.0, "transform": "sqrt"}` | 0.400 |
| days | stab/pooled | `{"d": "N", "floor": 0.1, "frac": 0.5, "inp": "pooled"}` | 0.496 |
| days | cca/pooled/sqrt/plain | `{"align": "go", "d": "N", "inp": "pooled", "phases": 20, "ridge": 0.0, "transform": "sqrt"}` | 0.351 |
| days | cca/pooled/sqrt/ridge | `{"align": "go", "d": "N", "inp": "pooled", "phases": 10, "ridge": 1.0, "transform": "sqrt"}` | 0.382 |
| month | stab/units | `{"d": "N", "floor": 0.1, "frac": 0.5, "inp": "units"}` | 0.558 |
| month | cca/units/raw/plain | `{"align": "trial", "d": 8, "inp": "units", "phases": 40, "ridge": 0.0, "transform": "raw"}` | 0.354 |
| month | cca/units/raw/ridge | `{"align": "trial", "d": "N", "inp": "units", "phases": 20, "ridge": 10.0, "transform": "raw"}` | 0.472 |
| month | cca/units/sqrt/plain | `{"align": "trial", "d": 5, "inp": "units", "phases": 20, "ridge": 0.0, "transform": "sqrt"}` | 0.325 |
| month | cca/units/sqrt/ridge | `{"align": "trial", "d": "N", "inp": "units", "phases": 20, "ridge": 10.0, "transform": "sqrt"}` | 0.447 |
| month | stab/pooled | `{"d": 10, "floor": 0.01, "frac": 1.0, "inp": "pooled"}` | 0.578 |
| month | cca/pooled/sqrt/plain | `{"align": "trial", "d": 5, "inp": "pooled", "phases": 20, "ridge": 0.0, "transform": "sqrt"}` | 0.330 |
| month | cca/pooled/sqrt/ridge | `{"align": "trial", "d": 10, "inp": "pooled", "phases": 10, "ridge": 10.0, "transform": "sqrt"}` | 0.423 |
| year | stab/units | `{"d": "N", "floor": 0.1, "frac": 0.5, "inp": "units"}` | 0.558 |
| year | cca/units/raw/plain | `{"align": "trial", "d": 8, "inp": "units", "phases": 40, "ridge": 0.0, "transform": "raw"}` | 0.354 |
| year | cca/units/raw/ridge | `{"align": "trial", "d": "N", "inp": "units", "phases": 20, "ridge": 10.0, "transform": "raw"}` | 0.472 |
| year | cca/units/sqrt/plain | `{"align": "trial", "d": 5, "inp": "units", "phases": 20, "ridge": 0.0, "transform": "sqrt"}` | 0.325 |
| year | cca/units/sqrt/ridge | `{"align": "trial", "d": "N", "inp": "units", "phases": 20, "ridge": 10.0, "transform": "sqrt"}` | 0.447 |
| year | stab/pooled | `{"d": 10, "floor": 0.01, "frac": 1.0, "inp": "pooled"}` | 0.578 |
| year | cca/pooled/sqrt/plain | `{"align": "trial", "d": 5, "inp": "pooled", "phases": 20, "ridge": 0.0, "transform": "sqrt"}` | 0.330 |
| year | cca/pooled/sqrt/ridge | `{"align": "trial", "d": 10, "inp": "pooled", "phases": 10, "ridge": 10.0, "transform": "sqrt"}` | 0.423 |

**(ii) NoMAD 逐 target / seed**（修正后；初版的无效结果见 `nomad_legacy_invalid_column_order.csv`：days 0616 +0.47…+0.48，0617 −0.70…−0.78，month −0.88…−0.92，year −0.59…−0.68，原因是 target 列顺序没有与 day-0 对齐）：

| 变体 | stratum | target | seed | query R²（全部 B、draw 平均） | native 子集 R² | alignment epochs |
|---|---|---|---:|---:|---:|---:|
| pooled (主) | days | sub-M_ses-CO-20150616 | 0 | 0.454 | 0.452 | 115 |
| pooled (主) | days | sub-M_ses-CO-20150616 | 1 | 0.458 | 0.463 | 114 |
| pooled (主) | days | sub-M_ses-CO-20150616 | 2 | 0.459 | 0.457 | 119 |
| pooled (主) | days | sub-M_ses-CO-20150617 | 0 | 0.448 | 0.460 | 129 |
| pooled (主) | days | sub-M_ses-CO-20150617 | 1 | 0.468 | 0.497 | 111 |
| pooled (主) | days | sub-M_ses-CO-20150617 | 2 | 0.456 | 0.475 | 117 |
| pooled (主) | month | sub-M_ses-CO-20140626 | 0 | 0.249 | 0.251 | 113 |
| pooled (主) | month | sub-M_ses-CO-20140626 | 1 | 0.252 | 0.270 | 118 |
| pooled (主) | month | sub-M_ses-CO-20140626 | 2 | 0.186 | 0.180 | 113 |
| pooled (主) | year | sub-M_ses-CO-20150511 | 0 | -0.333 | -0.618 | 128 |
| pooled (主) | year | sub-M_ses-CO-20150511 | 1 | -0.404 | -0.740 | 111 |
| pooled (主) | year | sub-M_ses-CO-20150511 | 2 | -0.414 | -0.743 | 120 |
| units_elec | days | sub-M_ses-CO-20150616 | 0 | 0.530 | 0.540 | 114 |
| units_elec | days | sub-M_ses-CO-20150616 | 1 | 0.534 | 0.546 | 124 |
| units_elec | days | sub-M_ses-CO-20150616 | 2 | 0.533 | 0.545 | 126 |
| units_elec | days | sub-M_ses-CO-20150617 | 0 | 0.492 | 0.538 | 120 |
| units_elec | days | sub-M_ses-CO-20150617 | 1 | 0.505 | 0.557 | 111 |
| units_elec | days | sub-M_ses-CO-20150617 | 2 | 0.495 | 0.539 | 120 |
| units_elec | month | sub-M_ses-CO-20140626 | 0 | 0.113 | 0.078 | 115 |
| units_elec | month | sub-M_ses-CO-20140626 | 1 | 0.088 | 0.049 | 114 |
| units_elec | month | sub-M_ses-CO-20140626 | 2 | 0.101 | 0.073 | 115 |
| units_elec | year | sub-M_ses-CO-20150511 | 0 | -0.224 | -0.394 | 127 |
| units_elec | year | sub-M_ses-CO-20150511 | 1 | -0.218 | -0.395 | 111 |
| units_elec | year | sub-M_ses-CO-20150511 | 2 | -0.216 | -0.385 | 115 |

对照（baseline day 自己走一遍 alignment network；R² 在该 session 的全部 usable trial 上，无 B/draw）：

| 变体 | stratum | session | seed | R² | alignment epochs |
|---|---|---|---:|---:|---:|
| pooled (主) | days | sub-M_ses-CO-20150615 (CONTROL: baseline day, active columns shuffled) | 0 | -0.486 | 123 |
| pooled (主) | days | sub-M_ses-CO-20150615 (control: baseline day, unshuffled) | 0 | 0.534 | 115 |
| pooled (主) | month | sub-M_ses-CO-20140306 (CONTROL: baseline day, active columns shuffled) | 0 | -0.916 | 114 |
| pooled (主) | month | sub-M_ses-CO-20140306 (control: baseline day, unshuffled) | 0 | 0.758 | 115 |
| pooled (主) | year | sub-M_ses-CO-20140306 (CONTROL: baseline day, active columns shuffled) | 0 | -0.916 | 114 |
| pooled (主) | year | sub-M_ses-CO-20140306 (control: baseline day, unshuffled) | 0 | 0.758 | 115 |

读法：`pooled` 为主变体；`units_elec` 为审查者的电极重排变体；两者共用同一个 baseline day 和各自的 day-0 LFADS。month 与 year 的两行 baseline-day 对照是同一次运行（两个 stratum 共用 baseline day 20140306 与 day-0 模型）。全部 run 都显式设了 seed，且同一 seed 可逐位复现（`INIT_READIN_FROM_DAY0` 的 True/False 两次 run 结果完全相同）。alignment 训练在 ≈110–130 epoch 早停（patience 10）。guard 在每个 run 的日志里都打印（`column guard {...}`：pooled 为 96 通道对 96 通道；units_elec 的 0616 / 0617 / 0626 / 0511 分别为 15 / 13 / 19 / 14 个标注列、0 个不一致）。 day-0 decoder 在 baseline day 上的训练集内 R²（标准化速度单位）：pooled 输入 days 0.605、2014 为 0.849；初版 sorted-unit 输入为 0.711 / 0.891（pooled 丢失同电极 unit 间的区分，上限更低）。

**(iii) APST 作者自己的 dev 数**（first-M 协议、Q50、dev-selected checkpoint，来自 `…/budget/dev_results/summary.csv`，只读 dev，没有读 final）：APST-FULL M=4/8/16/32 的两个 dev session 均值为 0.511 / 0.831 / 0.839 / 0.841；ACT-only 为 0.294 / 0.620 / 0.621 / 0.699。这些数用的是 dev 上选的 checkpoint、前 M 个 trial、作者的 Q50 query，与本评测的随机 B-子集 + 我们的 query 口径**不是同一协议**，只用来说明：我们的适配器在作者协议下复现这一数（0616，M=8：0.855956 对 0.855956），而在 native 子集上用随机 B-子集得到的是 0.70–0.73（B=8）/ 0.81–0.82（B=33）。

**(iv) 与 P0 的配对差（baseline − P0，逐 draw 配对；正 = baseline 更好；全表见 `paired_vs_p0.csv` / `paired_vs_p0_native.csv`）**：见 §0 第 1–4 条；每个 (方法, stratum, B) 上有几个 target 高于 P0 见 `beats_p0_counts_per_target.csv`。

### 3.4 逐 target 表（B=1/8/33；R² 与 [baseline − P0]）

`[…]` 是该 target 上 8 个 draw 的平均配对差（baseline − P0）。完整行见 `main_table_per_target.md`，数值见 `cells_per_target.csv`（含 draw 间的 CI）。

#### days / sub-M_ses-CO-20150616

| method | B=1: R2 [baseline-P0] | B=8: R2 [baseline-P0] | B=33: R2 [baseline-P0] |
|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | 0.335 [-0.304] | 0.375 [-0.355] | 0.375 [-0.366] |
| P0 (fixed-weight per-unit encoder, level 1) | 0.639 | 0.730 | 0.740 |
| P0 + R1-stream (readout ridge, level 2) | 0.639 [+0.000] | 0.728 [-0.001] | 0.741 [+0.001] |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | 0.642 [+0.003] | 0.731 [+0.001] | 0.744 [+0.004] |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | 0.645 [+0.006] | 0.696 [-0.034] | 0.728 [-0.012] |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | 0.592 [-0.047] | 0.669 [-0.061] | 0.690 [-0.050] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | -0.166 [-0.805] | 0.510 [-0.220] | 0.657 [-0.083] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | -0.161 [-0.800] | 0.510 [-0.220] | 0.670 [-0.070] |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | -0.062 [-0.701] | 0.524 [-0.206] | 0.671 [-0.069] |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | -0.019 [-0.658] | 0.551 [-0.179] | 0.688 [-0.053] |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | -0.009 [-0.648] | 0.588 [-0.142] | 0.718 [-0.022] |
| APST-FULL (last segment, source-only rule; single training seed 42) | n/a | 0.465 [-0.264] | 0.593 [-0.147] |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | n/a | 0.495 [-0.234] | 0.597 [-0.143] |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | 0.731 [+0.092] | 0.731 [+0.002] | 0.728 [-0.012] |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | 0.706 [+0.067] | 0.707 [-0.023] | 0.704 [-0.036] |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | 0.750 [+0.111] | 0.750 [+0.021] | 0.749 [+0.009] |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | 0.458 [-0.181] | 0.457 [-0.272] | 0.454 [-0.286] |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | 0.533 [-0.106] | 0.532 [-0.198] | 0.531 [-0.209] |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | 0.375 [-0.264] | 0.605 [-0.125] | 0.695 [-0.045] |
| P0 + ui_reg (P2 rule) | 0.644 [+0.005] | 0.719 [-0.011] | 0.741 [+0.001] |

#### days / sub-M_ses-CO-20150617

| method | B=1: R2 [baseline-P0] | B=8: R2 [baseline-P0] | B=33: R2 [baseline-P0] |
|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | 0.330 [-0.326] | 0.345 [-0.372] | 0.348 [-0.387] |
| P0 (fixed-weight per-unit encoder, level 1) | 0.656 | 0.717 | 0.735 |
| P0 + R1-stream (readout ridge, level 2) | 0.656 [+0.000] | 0.718 [+0.001] | 0.738 [+0.003] |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | 0.661 [+0.005] | 0.725 [+0.007] | 0.750 [+0.015] |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | 0.640 [-0.016] | 0.734 [+0.017] | 0.761 [+0.026] |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | 0.464 [-0.191] | 0.597 [-0.120] | 0.643 [-0.092] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | -0.099 [-0.755] | 0.556 [-0.161] | 0.691 [-0.044] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | -0.086 [-0.742] | 0.565 [-0.152] | 0.694 [-0.041] |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | 0.063 [-0.592] | 0.613 [-0.104] | 0.705 [-0.030] |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | -0.092 [-0.748] | 0.601 [-0.116] | 0.716 [-0.020] |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | -0.043 [-0.699] | 0.630 [-0.088] | 0.737 [+0.002] |
| APST-FULL (last segment, source-only rule; single training seed 42) | n/a | 0.536 [-0.182] | 0.587 [-0.149] |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | n/a | 0.537 [-0.180] | 0.578 [-0.157] |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | 0.765 [+0.109] | 0.763 [+0.046] | 0.764 [+0.029] |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | 0.731 [+0.076] | 0.731 [+0.013] | 0.732 [-0.003] |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | 0.755 [+0.100] | 0.754 [+0.037] | 0.757 [+0.022] |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | 0.457 [-0.199] | 0.455 [-0.263] | 0.462 [-0.274] |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | 0.497 [-0.159] | 0.495 [-0.222] | 0.500 [-0.235] |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | 0.380 [-0.276] | 0.610 [-0.107] | 0.713 [-0.022] |
| P0 + ui_reg (P2 rule) | 0.664 [+0.008] | 0.721 [+0.004] | 0.750 [+0.015] |

#### month / sub-M_ses-CO-20140626

| method | B=1: R2 [baseline-P0] | B=8: R2 [baseline-P0] | B=33: R2 [baseline-P0] |
|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | 0.315 [-0.151] | 0.332 [-0.233] | 0.333 [-0.233] |
| P0 (fixed-weight per-unit encoder, level 1) | 0.466 | 0.565 | 0.566 |
| P0 + R1-stream (readout ridge, level 2) | 0.472 [+0.005] | 0.576 [+0.012] | 0.606 [+0.040] |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | 0.492 [+0.026] | 0.630 [+0.065] | 0.679 [+0.113] |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | 0.154 [-0.312] | 0.295 [-0.270] | 0.304 [-0.263] |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | 0.263 [-0.204] | 0.267 [-0.298] | 0.280 [-0.286] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | -0.013 [-0.480] | 0.305 [-0.259] | 0.365 [-0.202] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | 0.032 [-0.435] | 0.316 [-0.249] | 0.390 [-0.177] |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | 0.063 [-0.403] | 0.378 [-0.187] | 0.478 [-0.089] |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | 0.047 [-0.419] | 0.495 [-0.070] | 0.646 [+0.079] |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | 0.060 [-0.407] | 0.512 [-0.053] | 0.656 [+0.090] |
| APST-FULL (last segment, source-only rule; single training seed 42) | n/a | 0.271 [-0.294] | 0.376 [-0.191] |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | n/a | 0.290 [-0.275] | 0.387 [-0.180] |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | 0.376 [-0.090] | 0.376 [-0.188] | 0.376 [-0.190] |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | 0.466 [-0.001] | 0.466 [-0.099] | 0.466 [-0.101] |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | 0.698 [+0.232] | 0.698 [+0.133] | 0.698 [+0.131] |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | 0.230 [-0.237] | 0.230 [-0.335] | 0.227 [-0.339] |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | 0.101 [-0.365] | 0.101 [-0.464] | 0.099 [-0.467] |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | 0.391 [-0.075] | 0.632 [+0.067] | 0.720 [+0.153] |
| P0 + ui_reg (P2 rule) | 0.487 [+0.021] | 0.653 [+0.088] | 0.731 [+0.165] |

#### year / sub-M_ses-CO-20150511

| method | B=1: R2 [baseline-P0] | B=8: R2 [baseline-P0] | B=33: R2 [baseline-P0] |
|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | 0.145 [-0.054] | 0.133 [-0.122] | 0.150 [-0.165] |
| P0 (fixed-weight per-unit encoder, level 1) | 0.198 | 0.256 | 0.315 |
| P0 + R1-stream (readout ridge, level 2) | 0.199 [+0.000] | 0.254 [-0.002] | 0.376 [+0.060] |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | 0.219 [+0.021] | 0.334 [+0.078] | 0.435 [+0.119] |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | -0.054 [-0.252] | -0.004 [-0.260] | 0.034 [-0.282] |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | 0.020 [-0.179] | 0.061 [-0.195] | 0.070 [-0.246] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | -0.011 [-0.210] | 0.101 [-0.154] | 0.181 [-0.134] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | 0.030 [-0.168] | 0.155 [-0.101] | 0.243 [-0.073] |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | 0.003 [-0.196] | 0.164 [-0.092] | 0.243 [-0.073] |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | 0.033 [-0.166] | 0.265 [+0.010] | 0.427 [+0.111] |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | 0.023 [-0.175] | 0.289 [+0.034] | 0.439 [+0.123] |
| APST-FULL (last segment, source-only rule; single training seed 42) | n/a | 0.150 [-0.106] | 0.213 [-0.103] |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | n/a | 0.189 [-0.066] | 0.257 [-0.058] |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | 0.079 [-0.119] | 0.079 [-0.177] | 0.087 [-0.229] |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | 0.221 [+0.023] | 0.220 [-0.035] | 0.244 [-0.072] |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | 0.412 [+0.213] | 0.410 [+0.154] | 0.454 [+0.138] |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | -0.378 [-0.576] | -0.376 [-0.632] | -0.414 [-0.729] |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | -0.216 [-0.414] | -0.215 [-0.471] | -0.237 [-0.553] |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | 0.191 [-0.008] | 0.290 [+0.035] | 0.478 [+0.163] |
| P0 + ui_reg (P2 rule) | 0.228 [+0.029] | 0.309 [+0.054] | 0.488 [+0.172] |



## 4. 局限与未做

1. **数据量**：4 个 target session = days 的 20150616 / 20150617（共用 `ap_pool_x8` trunk）+ month 的 20140626 + year 的 20150511；month 与 year 共用同一个 trunk（两个 `best.pt` 的 `state_dict` 逐项相同，已核对）、同一批 6 个 source、同一个 baseline day，所以实际只有 **2 个 trunk**；1 只猴子（sub-M）、1 个任务（center-out）。所有“高于/低于 P0”的说法都只限于“这 4 个 session（k/4）”，不能外推；`beats_p0_counts_per_target.csv` 给出每个 (方法, stratum, B) 上高于 P0 的 target 数。
2. **CI 只含 draw 方差**（8 个 draw 的 t 区间）。不含 target / trunk / source 方差；`(b)` 行与 NoMAD 的对齐不依赖 B/draw（同一行内不同 B 之间只有 query mask 不同），没有真正意义上的 CI；baseline 的闭式方法没有训练 seed；APST 只有一个训练 seed（42）；NoMAD 有 3 个 alignment seed 但 day-0 LFADS 只训练一次。
3. **多重比较**：`paired_vs_p0.csv` 里约 450 个（主 query 口径；native 口径另有若干） (方法, stratum, B) 配对格都做了 95% 区间，没有做多重比较校正；单个格“CI 不跨 0”不是显著性证明，主要看跨 B、跨 session 的一致性。
4. **选择规则的限制**：pseudo-target 是 trunk 的训练 session，只用于排序，source 上的最优配置在 target 上不一定最优（例如 Gallego-style plain CCA 在 month/year 选到 d=5，B=33 的 R² 比审查前固定 d=10 的数还低：0.390 对 0.423、0.243 对 0.315，见 §“审查后修正”的前后对比表）。审查后的网格（d、相位数、对齐、electrode-pooled 变体）是在**看过第一版 target 结果之后**设计的，所以不是预注册；但网格在新一轮任何 target 行评测之前已完整固定，选择只用 source。CCA 的第二阶段（相位数/对齐）是在第一阶段最优点附近的坐标下降，不是全网格。
5. **Stabilizer 与 CCA 的实现是“忠实移植 + 为接冻结 AP trunk 加的桥”**，不是原文的闭环实验；桥的上限见 §3.3 (i)。Stabilizer 的 stable-channel 丢行步骤在所有最终配置里没有运行（§2.1）；electrode-pooled 输入会损失 unit 级信息（桥上限低于 sorted-unit 输入），所以 pooled 对 Stabilizer 有利、对 CCA 不利。
6. **APST**：days 的 20150616 / 20150617 是 APST 作者自己的 dev session，checkpoint 的 dev 选择、配方设计都见过它们；所有 APST checkpoint 都是单个训练 seed（42）；month/year 的 checkpoint 是我们按作者冻结配方重训的；APST ACT-only 在 2014 roster 上没有训练。native-query 表里 “APST 在 days 的 B=16/33 高于 P0” 是 APST 自己 dev 设定下的结果，不是对未见 session 的证据。
7. **query 口径**：主表用 P 行的口径（trial 内全部 bin）；APST 为 native 口径训练，所以主表对 APST 不利；native 表对不能重算的 R1-stream / R3 缺项。
8. **不可用格**：APST-FULL 需要 ≥3 个方向非共线的有标签 trial：B=1 总是不可用；B=4 在 days 有 1/16 个 (target, draw) 格不可用（即 days B=4 有一个 draw 只由单个 target 贡献），month 3/8、year 1/8 个 draw 不可用；**month B=8 有 1/8 个 draw 不可用**（n=7）。完整列表见 `unavailable_cells.csv`；主表里相应格标 `(n=k)`。
9. **R1-stream / R3 行**用 P3 为各 stratum 固定的选择集：days 用 IS，month 用因果 OOS（`oosc`），year 用 OOS；它们不是本评测重新选的，与 baseline 行的选择规则不同。
10. **NoMAD**：见 §2.4 / §3.3 (ii)：TF 2.0→2.6 的移植需要 6 处运行时补丁，训练在 CPU 上，每个 target 3 个 alignment seed，alignment 训练在 ≈110–130 epoch 早停，若放宽早停是否改善没有试（没有 target 上调参）。NoMAD 现在用到 `unit_to_electrode`。
11. **引用**：NoMAD 的 Nat. Commun. 条目（16:4662, 2025）**未能离线核实**（本评测环境无法联网核对卷/页码，审查前的“已核实”标记已撤回；bioRxiv 10.1101/2022.04.06.487388 是本地 nomad 仓库 README 引用的版本）。**APST 标题有两个版本**：`APST/README.md` 的 BibTeX 是 “Association profile conditioning in a set-temporal transformer for cross-session intracortical motor decoding”（arXiv:2609.39080），`APST-public/README.md` 的第一段是 “Association Profile Conditioning for Set-Temporal Transformers in Neural Decoding Across Sessions”；**请 MZ 确认哪个是最终标题**，论文里的引用要与之一致。另外 APST 是 MZ 自己的工作，**投稿 DAC 双盲时这条自引必须匿名化**（改成 “anonymous et al.” / 第三人称并去掉仓库与 arXiv 链接）。
12. 与规则 7–9 的关系：评测是整段连续因果解码；没有把同一过程的长步数版本当“上限”（`*_b` 行是独立的“论文数据量”过程，标了违反 H4）；本评测不包含硬件数字。

## 5. 复现

环境：SSM 代码 `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM /home/xinyuan/miniconda3/envs/spint/bin/python`；APST 另加 `/home/xinyuan/Work_host/APST/src` 到 PYTHONPATH；NoMAD 用 `/mnt/data/work_host/envs/nomad_tf26/bin/python`（CPU，`CUDA_VISIBLE_DEVICES=""`）。审查前的第一版原始 rows / 选择文件 / NoMAD 预测在 `results/baselines_20261016/v1_pre_review/`（已被取代，保留备查）。

```
python scripts/run_baselines_readin.py select --stratum {days,month,year}   # source-only 选超参（整个网格固定后才评测）
python scripts/run_baselines_readin.py eval   --stratum {days,month,year}   # Stabilizer + CCA 行（sorted-unit 与 electrode-pooled）
python scripts/run_baselines_readin.py sanity --stratum {days,month,year}   # 桥上限（针对表中实际使用的配置）
# 第二次审查：source 选择敏感性（修正 d 之后重选，结果写到另一个目录；只作报告）
BL_OUT=results/baselines_20261016_selsens python scripts/run_baselines_readin.py select --stratum {days,month}
python scripts/baselines_selection_sensitivity.py cuda:0
python scripts/run_baselines_native_p.py {days,month,year}                  # P-prior/P0 重算 + native
python scripts/run_baselines_apst.py eval --stratum days  --model {apst_full_last,apst_act_last,apst_full_devsel,apst_act_devsel}
python scripts/apst_2014_train.py prepare; python scripts/apst_2014_train.py train --name sua_concat_pretrain_s42  # 然后 sua_dual_site_F0_s42
python scripts/run_baselines_apst.py eval --stratum {month,year} --model {apst14_full_last,apst14_full_sel0307}
python scripts/nomad_prep.py days month year                                # 同时导出 unit_to_electrode
# NoMAD（TF 2.6 环境，CPU）：pooled 主变体需要重训 day-0 LFADS 与 decoder（DATA_DIM=96）
python scripts/run_baselines_nomad_tf.py {day0,decoder} --stratum {days,month} --tag {days_pool,y2014_pool} --input pooled
python scripts/run_baselines_nomad_tf.py align --stratum S --tag T --input {pooled,units_elec} --init-readin {1,0} --target <session> --seed {0,1,2} [--shuffle-control]
scripts/run_nomad_jobs.sh <jobs-file> <parallel>                            # 批量 align
python scripts/run_baselines_nomad_score.py {days,month,year} {pooled,units_elec,pooled_noinit}
python scripts/analyze_baselines.py
python -m pytest tests/test_published_baselines.py tests/test_electrode_align.py
```
