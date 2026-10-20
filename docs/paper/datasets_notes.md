# Datasets 小节事实核对笔记（2026-10-20）

对应文件：`docs/paper/datasets_section.tex`、`docs/paper/tab_datasets.tex`、`docs/paper/datasets_refs.bib`。
所有路径相对 `SSM/`，除非写成绝对路径。只读核对，没有改动任何代码或结果。

## 1. DANDI 000688 通用事实

| 事实 | 值 | 来源 |
|---|---|---|
| bin 宽 | 20 ms | `ssm_decode/dandi688_data.py:62-63`（`BIN_SECONDS = 0.020`） |
| 输入 | M1 sorted single units，每个 session 的 unit 集合不同；映射到 96-electrode canonical M1 map | `ssm_decode/dandi688_data.py:9-11, 64, 176-177` |
| 电极 | "96-electrode Utah array in M1"（NWB `electrode_group_M1` 的 description） | sub-C NWB `general/extracellular_ephys/electrode_group_M1`（h5py 读取 `sub-C_ses-CO-20150629`） |
| 输出 | 2-D cursor velocity（bin 中心插值） | `ssm_decode/dandi688_data.py:280-283, 297-300, 316-317` |
| eval mask | rewarded（`result == "R"`）trial 内、且在 `cursor_vel` 时间支撑内的 bin | `ssm_decode/dandi688_data.py:18-25` |
| usable trial | ≥ 50 bins 且至少一个 eval-valid bin | `ssm_decode/track_a_experiment.py:60, 63-71` |
| 任务 | "Monkey M/C performing center-out reaching task."；lab Miller，Northwestern | NWB `session_description` / `general/lab`（`sub-M_ses-CO-20140203`、`sub-C_ses-CO-20150629`） |
| dandiset 版本 | 0.250122.1735 | `/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/manifest.json` |
| related_publications | 10.1016/j.neuron.2018.09.030；10.48550/arXiv.2310.16046；10.1007/s00221-017-4997-1；10.1038/s41593-019-0555-4；10.1038/s41467-018-04062-6 | NWB `general/related_publications`（sub-M 与 sub-C 相同） |

## 2. sub-M（monkey M）

| 事实 | 值 | 来源 |
|---|---|---|
| 使用的 CO session | 19 个，20140203–20150617 | `results/analysis/expansion_20261016/session_inventory.json`（用脚本统计：sub-M 19 行） |
| 每 session unit 数 | 10–52（20141203 只有 10） | 同上；`results/analysis/drift_curve_20261016/README.md:21-30` |
| 磁盘上 CO session 总数 | 22（19 + 3 final） | `ls SPINT/sua_exploration/data/dandi_000688/sub-M/` |
| 从未使用的 3 个 session | 20150623 / 0625 / 0626（最后三个），loader 抛 PermissionError | `ssm_decode/dandi688_data.py:77, 139-144`；`docs/feasibility_p1_20261015.md:12`（§0 第 4 条） |
| month backbone 训练 session | 20140203/0217/0218/0303/0304/0306（最新 20140306） | `scripts/run_labeled_calibration.py:43-45`；`results/analysis/drift_curve_20261016/README.md:9` |
| 12 个 drift target 与 gap | 20140626 … 20150617，112–468 d | `results/analysis/drift_curve_20261016/README.md:12, 21-28`；`scripts/run_causal_main.py:1441-1442`（`_DRIFT`） |
| 规则选择 session | 20140307（source，非 target） | `results/analysis/drift_curve_20261016/README.md:11`；`results/analysis/causal_main_20261019/README.md:15` |
| days backbone | sources 20150511–20150615，targets 20150616/0617（gap 1–2 d） | `scripts/run_labeled_calibration.py:40-42`；`scripts/run_causal_main.py:73, 1459`；`results/analysis/causal_main_20261019/README.md:85-97` |
| strata | days 1–7 d、month 90–130 d、year ≥ 365 d（协议定义）；P1 中 month target 20140626 = 112 d，year target 20150511 = 431 d | `ssm_decode/expansion_protocol.py:49`；`scripts/run_labeled_calibration.py:43-48`；`results/analysis/drift_curve_20261016/README.md:23, 27` |
| year 不是独立 trunk | year 与 month trunk 权重逐张相同，主表不再单列 year | `results/analysis/causal_main_20261019/README.md:5, 19`；`results/analysis/drift_curve_20261016/README.md:9` |

正文只写 "12 later sessions 112–468 days out" 和 "two targets 1–2 days later"，没有单列 year stratum（理由见上一行）。

## 3. sub-C（monkey C）

| 事实 | 值 | 来源 |
|---|---|---|
| 磁盘上 CO session | 53 | `SPINT/sua_exploration/data/dandi_000688/manifest.json`（sub-C 文件中 `_ses-CO-` 计数） |
| inventory 中允许的 session | 47（53 − 6 个 held-out），其中 14 个是 2016 年 | `session_inventory.json`（脚本统计） |
| 被保留为 held-out、从不使用的 6 个 | 20151113/1116/1117/1119/1120/1201 | `ssm_decode/dandi688_data.py:86-89, 97-101` |
| 2016 session 不使用 | year targets（2016）是 optional，没有运行；README 明确不使用 2016 | `ssm_decode/expansion_protocol.py:243-247`；`results/analysis/expansion_20261016/README.md:10` |
| round 1 实际用到的 session | 25 个（3 个 backbone 的 pool + primary target 并集），20131220–20151112，units 38–91 | `ssm_decode/expansion_protocol.py:240-254` + inventory（脚本统计） |
| round 1 内容 | sc_0629（6 month）、sc_0708（5 days + 6 month）、sc_0713（4 days）；21 个 (backbone, target) 对 = 16 个不同 session；days 7 session，month 12 session | `results/analysis/expansion_20261016/round1_README.md:1, 14-23` |
| round 1 协议 | 随机 draw（8 draws × 2 seeds），B ∈ {1,4,8,16,33}，contiguous 解码 | `round1_README.md:6, 22`；`ssm_decode/expansion_protocol.py:46-48` |
| gap（相对 pool[-1]） | days 1–7 d；month 102–128 d（sc_0629: 102–119，sc_0708: 119–128） | `ssm_decode/expansion_protocol.py:49, 123-124`（脚本计算 `gap_days`） |
| causal 主表 | sc_0629、sc_0708 两个 month backbone，各 6 个 target；B ∈ {1,4,8,16,33}（B=2 没有 source 规则，跳过） | `results/analysis/causal_main_20261019/README.md:99-120`；`scripts/run_causal_main.py:57, 1020, 1443-1446` |
| causal 主表涉及的 session | 21 个（两个 pool 加 12 个 target，去重） | 脚本统计 |

## 4. FALCON M2

| 事实 | 值 | 来源 |
|---|---|---|
| 数据 | DANDI 000953 MonkeyN 的 held-in（calib）session，7 个：2020-10-19 R1/R2、10-20 R1/R2、10-27 R1/R2、10-28 R1 | `results/data_cache/m2/held_in/*.npz` 的 `path` 字段；`ssm_decode/m2_calib.py:84-112`；`results/analysis/m2_calib_20261016/README.md:79` |
| 通道 | 96 个固定通道（unit_to_electrode 为 identity） | `ssm_decode/m2_calib.py:41, 88-89`；`m2_calib README:69, 92` |
| 输出 | 2-D finger velocity（index、MRS 两个 finger group），20 ms bins | NWB `acquisition/finger_vel/{index,mrs}`、`experiment_description`（"Two finger group movement in NHP. Behavior provided in 20ms bins"）；FALCON 官方 loader `falcon_challenge/dataloaders.py:146-164` |
| 5 个 target（D2 pools） | pool1→10-20 R1/R2（gap 1 d）；pool2→10-27 R1/R2（7 d）；pool3→10-28 R1（1 d） | `scripts/run_labeled_calibration.py:50-60`；`m2_calib README:92` |
| 协议 | 随机 draw（8 draws），B ∈ {1,2,4,8,16,33}，整段连续因果解码，query = 其余 usable trial，物理单位方差加权 R² | `m2_calib README:56-60, 96` |
| 非官方指标 | `data.py` 声明是本地时间协议，不是 EvalAI/FALCON 评测；没有读 held-out/minival/`official_evalai` | `ssm_decode/data.py:1-4`；`m2_calib README:79, 192` |
| 注意 | 第一轮 D2 结果（`run_labeled_calibration.py` d2 分支）丢了 in_proj bias，无效；有效结果是 `m2_calib_20261016` | `scripts/run_labeled_calibration.py:12`；`m2_calib README:31` |

## 5. Hyser HD-sEMG

| 事实 | 值 | 来源 |
|---|---|---|
| 来源 | PhysioNet hd-semg 2.0.0 | `ssm_decode/hyser_data.py:1`；`/mnt/data/datasets/hd-semg/readme.txt` |
| 受试者 / session | 20 × 2（不同天），共 40 | `results/analysis/hyser_20261015/README.md:11`；readme.txt（20 subjects） |
| 通道 | 256（4 块 8×8 阵列 ED/EP/FD/FP），2048 Hz | `ssm_decode/hyser_data.py:4, 36, 39`；`hyser README:13` |
| 输出 | 5 通道手指力（thumb..little），1-DoF 数据集 | `ssm_decode/hyser_data.py:5`；`hyser README:3, 14` |
| trial | 每 session 5 指 × 3 sample = 15 个 trial，每个 25 s（1250 bins） | `ssm_decode/hyser_data.py:3-5, 40-42`；`hyser README:12` |
| 前端 | 整流 + one-pole leaky integrator（τ = 80 ms）+ 50 Hz（20 ms bin 末端取样） | `ssm_decode/hyser_data.py:14-22, 38, 43`；`hyser README:25-27` |
| 前端的上游非因果 | 输入是 PhysioNet 的 `preprocess` 文件，审查确认其带通为零相位（非因果）；没有从 raw 重建因果缓存 | `hyser README:18, 105`；`semg_calib README:420`。正文写成 "applied to its released band-passed signals" |
| 协议 | target = 每人 session2；B ∈ {1,2,5,10}，按手指均衡随机抽取，8 draws；query = 其余 15−B 个 trial；每个 trial 独立序列、从零状态解码 | `hyser README:34, 46`；`semg_calib README:244` |
| R² 口径 | 5 个力通道各自 R² 的平均（逐通道、尺度无关），不是 pooled 物理单位 R² | `hyser README:17, 34` |
| 角色 | 边界场景 | `docs/paper_mainline_proposal_20261016.md:33`；`docs/software_results_analysis_20261020.md:9` |

## 6. NinaPro DB6

| 事实 | 值 | 来源 |
|---|---|---|
| 结构 | 10 受试者 × 5 天 × 2 session/天（T1 上午、T2 下午），100 个文件 | `ssm_decode/db6_data.py:3, 61`；`results/analysis/db6_v2_20261016/README.md:34` |
| 通道 | 16 列中 0 起第 8、9 列恒为 0，14 个 Delsys 电极 | `ssm_decode/db6_data.py:4-6, 65-68`；`db6_v2 README:36` |
| 类别 | 7 个抓取（restimulus 码 1,3,4,6,9,10,11）+ 休息 = 8 类；per-bin balanced accuracy（chance 0.125） | `ssm_decode/db6_data.py:11-13, 62-63`；`db6_v2 README:130`；`results/analysis/db6_20261015/README.md` §1 |
| 每 session trial | 7 抓取 × 12 次重复 = 84 trial | `ssm_decode/db6_data.py:12-13`；`db6_v2 README:47` |
| source / target | source = day 1–2（39 个 session，剔除截断的 S2_D2_T2）；target = day 3–5（60 个） | `ssm_decode/db6_backbone.py:534-543`；`db6_v2 README:9, 37` |
| 截断文件 | S2_D2_T2（103 s，只有一个抓取，12 个 trial），从 source 剔除 | `ssm_decode/db6_backbone.py:534-535`；`db6_v2 README:34` |
| 通道 15 | S1–S4 早期活、后来死（S1 从 D5、S2 从 D4、S3 从 D3T2、S4 从 D3 起）；S5–S10 几乎全程死；60 个 target 中 53 个死 | `db6_v2 README:18, 37` |
| 前端 | 整流 + one-pole leaky integrator，状态跨 trial 连续；v2 主 backbone 用 τ = 160 ms（loader 默认 80 ms，drift 统计用 80） | `ssm_decode/db6_data.py:32-41, 60`；`db6_v2 README:9, 43` |
| v2 解码 | 整个 session 一条连续因果流，不在 trial 边界重置；backbone 在 trial 顺序打乱的流上训练（每 1000 步重排），评测按记录顺序 | `db6_v2 README:15, 45, 60`；`semg_calib README:230, 349` |
| 协议 | B ∈ {1,2,4,8,16}（诊断 42），class-balanced 随机抽取，5 draws；query = 其余 84−B 个 trial；校准 trial 可能晚于部分 query trial（不是严格在线） | `semg_calib README:349`；`db6_v2 README:47-50` |
| 电极重新放置 | v2 README 明确写 "电极重新放置没有在数据中验证"；内部文档写 "每次重新贴电极" | `db6_v2 README:321`；`docs/paper_mainline_proposal_20261016.md:33`。正文带 `% VERIFY` |

## 7. causal-first-B 协议（000688 主表）

| 事实 | 来源 |
|---|---|
| 校准集 = 记录顺序中的前 B 个 usable trial，不是 draw | `ssm_decode/causal_protocol.py:3-4, 70-77` |
| query = 之后的全部 usable trial | `ssm_decode/causal_protocol.py:4, 85-87` |
| anytime commit：前 k 个 trial 算出的表在 usable trial k 开始时才装载（latency 0，默认） | `ssm_decode/causal_protocol.py:5-9`；`causal_main README:33` |
| 前缀统计；trial 0 可用单 trial staging | `ssm_decode/causal_protocol.py:11-15`；`causal_main README:27` |
| 整段连续因果解码，不重置状态 | `docs/feasibility_p1_20261015.md:13`（§0 第 7 条）；`drift_curve README:14` |
| R² = 1 − SS_res/SS_tot，物理单位，在 query bin 上对两维合并计算 | `scripts/run_causal_main.py:566-578` |
| B 列表：sub-M (1,2,4,8,16,33)；sub-C (1,4,8,16,33) | `scripts/run_causal_main.py:56-57` |

注意：`drift_curve_20261016/README.md` 的 12-target 数字是随机 draw 协议（8 draws × 2 seeds，第 13 行）；正文说的 "main results" 指 `causal_main_20261019`（causal-first-B）。

## 8. 引用核对状态

| key | 状态 |
|---|---|
| `dandi000688` | id、版本号来自本地 manifest；DOI 是按 DANDI 惯例拼出来的（`10.48324/dandi.000688/0.250122.1735`），title 与作者没有离线来源 → `% VERIFY` |
| `gallego2020stability` | DOI 来自 NWB `related_publications`；title、作者、卷页凭记忆 → `% VERIFY` |
| `karpowicz2024falcon` | title、venue、前两位作者来自任务说明；其余作者凭记忆，用 "others" → `% VERIFY` |
| `jiang2021hyser` | 期刊、年份、DOI 来自任务说明；title、合作者、页码凭记忆 → `% VERIFY`（PhysioNet readme 只给出数据集标题 "Open Access Dataset and Toolbox of High-Density Surface Electromyogram Recordings"，与论文标题不同） |
| `palermo2017repeatability` | 本地没有任何元数据（DB6 的 .mat 与下载脚本不含引用）；整条凭记忆 → `% VERIFY` |

## 9. 给 MZ 的开放问题

1. **"year ≈ 15 months"**：year stratum 的 target 20150511 距最新 source 431 d（约 14.2 个月）；drift curve 的最远 target 468 d（约 15.4 个月）。正文没有单列 year（主表也已不单列，因为 year 与 month 是同一个 trunk），只写 112–468 d。是否需要恢复三档命名？
2. **Hyser 前端的上游是零相位滤波**：PhysioNet `preprocess` 文件的带通被审查确认为非因果；我们的 envelope 是因果的，但整条链不是。正文用了 "applied to its released band-passed signals"。要不要从 raw 重建因果前端，或者在正文更明确地披露？
3. **B 列表不统一**：Hyser {1,2,5,10}，DB6 {1,2,4,8,16}，sub-C causal 主表没有 B=2。正文和表都按实际写了，请确认表述方式。
4. **Hyser 不是连续解码**：每个 25 s trial 是独立文件，从零状态解码（与 §0 第 7 条的 "整段连续" 不同，但该数据本身没有连续流）。Hyser 的 R² 是 5 个力通道 R² 的平均，与 000688/M2 的 pooled 物理单位 R² 口径不同。是否需要在正文点明？
5. **电极重新放置**：DB6 v2 README 说 "电极重新放置没有在数据中验证"；正文的 "electrodes are re-applied every session" 带 `% VERIFY`，需要对照 Hyser 与 DB6 原论文的采集协议确认。
6. **Hyser 两个 session 的间隔天数**：本地文件里没有，表中写 "2nd day"，需要查 Hyser 原论文。
7. **sub-C session 数**：表写 25（round 1 的 pool + target 并集）；causal 主表只涉及 21 个。报哪一个？
8. **DB6 随机 draw 的时序**：校准 trial 可能晚于部分 query trial（`db6_v2 README:48`），Hyser 同理。正文只说 "random balanced draws"，没有展开。
9. **M2 结果的状态**：有效数字来自 `m2_calib_20261016`（修复 bias 之后，random-draw）；B=33 的增益是看过 6 个 budget 之后才突出的，README 标为需要预注册复验。这个属于结果小节，这里只提醒。
10. 正文约 360 词（按空格计，数字和日期各算一个词），比半栏略长；如需压缩，可以先删 DB6 的 "two sessions per day" 括号或 M2 的 "1–7 days" 细节（表里都有）。
