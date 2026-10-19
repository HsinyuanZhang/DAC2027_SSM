# 证据扩展：sub-C 第二受试者 + sub-M days 层 rolling backbones（2026-10-16，v2 修订）

状态：**代码、协议、队列已准备并通过 CPU 测试；没有任何 GPU 训练，也没有评测 rows。** 本机 GPU 被其他任务占用（`nvidia-smi` 显示约 19 GB 已用，高于 2 GB 门槛），所以没有做 GPU smoke test；GPU 时间由已有 P1 / P3 日志推算（§8）。

v2 相对 commit `634c08d` 的修订，来自独立审查和 MZ 的新决定：

* **MZ 决定 (a)**：R3（F7-J + readout ridge）是主方法之一。每个新 backbone 的评测行除 P-prior、P0、R1-stream 外，增加 **R2（F7-J）和 R3**；平均 Jacobian J̄ 在该 backbone 的 trunk SOURCE session 上离线计算，F7-J 与 R3 的规则由 source simulation 选择。
* **MZ 决定 (b)**：sub-C 的 month target 用 APST dev session（2015-11）；**不用 2016 年 session**。optional tier 默认关闭，`--tier all` 需要额外的 `--include-2016-optional`，并记录 warning。
* **协议变更（out-of-sample 选择）**：见 §1。
* **审查修正**：队列退出码（§6）、他处协议的 exam face 披露（§2.3）、重复 target 与按 session 聚类（§4.4）、forbidden 检查提前与 sub-M final 也被 loader 拒绝（§2.2）、`_scan_session` 保护、逐行审计字段（§5.3）、原子写入（§5.4）、成本重估（§8）。

代码与数据文件：

| 文件 | 作用 |
|---|---|
| `ssm_decode/dandi688_data.py` | loader：sub-C CO；**6 个 APST final 与 3 个 sub-M final 在 loader / `_scan_session` / inventory 中 `PermissionError`**，没有 override |
| `ssm_decode/expansion_protocol.py` | 协议（pool、trunk sources、held-out、target、strata、K、规则校验、重复 target 报告）；纯 Python |
| `ssm_decode/expansion_rules.py` | R1-stream / R2 / R3 的 source-only 规则选择（纯 Python / numpy） |
| `ssm_decode/atomic_jsonl.py` | 原子追加（temp + `os.replace`）与截断末行容错 |
| `scripts/build_expansion_inventory.py` | 一次性扫描 sub-M / sub-C 的所有**允许**的 CO session，生成 `session_inventory.json` |
| `scripts/run_expansion.py` | 分阶段 driver：`check / bundle / status / backbone / p0 / jacobian / select / eval` |
| `scripts/run_expansion_queue.sh` | 可恢复队列（每个 backbone 一条链，`--jobs N` 并发；退出码见 §6） |
| `scripts/analyze_expansion.py` | 按 target session 聚类的聚合（§4.4） |
| `scripts/expansion_cost.py` | 成本模型（§8，无 GPU） |
| `scripts/smoke_expansion_cpu.py` | CPU 端到端 smoke（GRU trunk 代替官方 Mamba-3） |
| `scripts/run_population_calib.py` | 仅一处改动：`sessions_for` 在 `D1_POOLS` 条目有 `heldout` 时把它附在最后（P1 条目没有该键，行为不变） |
| `tests/test_{dandi688_subc,expansion_protocol,run_expansion,atomic_jsonl,expansion_rules,analyze_expansion,expansion_cost}.py` | 测试（§10） |
| `results/analysis/expansion_20261016/` | `README.md`（分析规则）、`session_inventory.json`、`protocol.json`、`target_check.json`、`colab_bundle.json`、`colab_bundle_subC_primary.txt`、`colab_bundle_code_files.txt`、`cost_estimate.json` |

## 0. 摘要

* 8 个 backbone（4 个 sub-M days，其中 1 个复用现有 month trunk；4 个 sub-C）；**新训练 7 个 AP backbone**，复用 1 个。
* 35 个 primary target（sub-C 27 个：days 9、month 12、year 6；sub-M days 8），**不含任何 2016 年 session**。
* 评测行：`p_prior`、`p0`、`r1s`、`r2`、`r3`；B ∈ {1,4,8,16,33}，每个 (target, B) 8 个 draw，2 个 P0 seed；连续因果解码；物理单位 R²。每个 target 的行数 = 5 × 8 × (1 + 4 × 2) = 360，primary 共 12,600 行。
* 选择协议：新 backbone 的 trunk 只训练前 5 个 pool session，pool[-1] 做 P0 early stopping 和 R1/R2/R3 规则选择（OOS、因果）；sm_0307 是 IS，不与其他 rows 合并。
* 27 个 sub-C (backbone, target) 对 = **19 个不同 session**；统计推断以 session 为单位。
* **成本（§8）**：L4（1.54 CU/h，假设 L4 比 3090 慢 2×），顺序执行的 wall 期望 **9.3 h ≈ 14 CU**，上界 **14.5 h ≈ 22 CU**（L4 因子 1.5–2.5 时上界 11–18 h，17–27 CU）。总 CU 低于 30 CU，但上界超过一个 ~12 h session，所以**建议拆成两轮**：round 1（sc_0708、sc_0713、sc_0629）期望 4.1 h / 6.3 CU、上界 6.6 h / 10.2 CU；round 2（sc_0309 + 4 个 sub-M）期望 5.2 h / 8.0 CU、上界 7.9 h / 12.2 CU。两轮合计期望 14.3 CU、上界 22.4 CU（上限 40 CU）。
* Colab 数据 bundle：sub-C primary 30 个 NWB，**3,568,531,938 B（3.57 GB，3.32 GiB）**；代码 bundle 清单 22 个文件，约 326 KB。sub-M 所需 18 个文件已在现有 `data_000688_subM.tar.zst`。

## 1. 协议变更：trunk 不训练 pool[-1]（out-of-sample 选择）

**问题。** P3 §R.12 显示：规则（readout λ、F7-J 的 L/η/ρ/S、R3 的 readout）如果在 trunk 训练过的 session 上选（IS），选择会塌向“不改动”（小步、大 λ），因为在 trunk 已拟合的 session 上校准几乎没有可恢复的 gap；OOS 且因果的选择集才给出有意义的规则。v1 扩展里 pool 的最后一个 session 既被 trunk 训练又被用来选规则，所以是 IS。

**v2 规则。**

* 每个**新** backbone：AP trunk 在 `pool[:-1]`（前 5 个，`Backbone.trunk_sources`）上训练；`pool[-1]`（最晚的 pool session，`Backbone.heldout` = `val_session`）**不进 trunk**。
* `pool[-1]` 用于：P0 early stopping（P0 训练 episode 来自 5 个 trunk session，validation 在 `pool[-1]` 上；它的 prior 是 5 个 trunk table 的 electrode 平均，没有 leave-one-out，因为它不在 trunk 里）；source simulation（R1-stream、R2、R3 的规则选择）。
* 它是 trunk-unseen（OOS），也早于该 backbone 的所有 target（因果；`validate_backbone` 与测试 `test_new_backbones_hold_pool_last_out_of_trunk_training` 检查）。
* J̄（平均 Jacobian）和 readout PCA basis 只用 `trunk_sources`（5 个 session）计算，不用 held-out session。
* 复用的 `sm_0307`（现有 month trunk，看过全部 6 个 pool session，P0 calibrator 是 P1 的 month_O）**保持原样**，其选择 session 20140306 是 IS，标记 `selection_oos=false`。**它的 rows 不与 OOS rows 合并，也不与 P3 / drift-curve 的 R1–R3 rows 合并**（那些规则在 20140307 上选，而 20140307 是 sm_0307 的 target，所以旧规则不能复用）。聚合脚本的所有分组键含 `selection_oos`。
* 实现：`D1_POOLS[bid]` 的 `sources` = trunk sources，新键 `heldout` = `[pool[-1]]`；`sessions_for(bid, "O")` 把它附在最后，所以 `World.train_ids` = 5 个 trunk session，`World.val_ids` = `[pool[-1]]`。AP 命令的 `--dandi688-sources` 是前 5 个；`check_backbone_matches` 要求 checkpoint 的 `source_sessions` 恰为这 5 个，并拒绝包含 held-out session 的 checkpoint。

**后果（需要读者注意）。**

1. **days 标签是相对 pool[-1]，不是相对 trunk。** 新 trunk 的最后训练 session 比 pool[-1] 更早，所以 trunk gap ≥ stratum gap。最极端的是 sm_0627：target 20140627 到 pool[-1]=20140626 为 1 天，到 trunk 最后训练 session 20140307 为 112 天；sm_0512 为 1 天 vs 225 天。stratum 仍按“到最晚已见 session（pool[-1]）”标注（它决定 P0 / 规则选择所处的时间位置），rows 同时记录 `gap_days` 与 `trunk_gap_days`，聚合脚本另给按 trunk gap 分层的表。sc_* 的差只有 1 天（连续的 2015-06/07），sm_0611 为 30 天。
2. trunk 的训练数据少一个 session（5 而非 6），pool 的“最近 6 个 session”不变；Ledermann 上界按 6 个 session 的最小 N 校验（更保守）。
3. 成本：R2 / R3 的规则选择是最重的一段（§8）。

## 2. 数据、禁止 session、披露

### 2.1 sub-C 数据

数据路径：`/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C`，68 个 NWB（9.1 GB）：CO 53 个、RT 15 个。只用 CO（loader 对 RT 抛 `ValueError`）。

| 项 | sub-C | sub-M（既有） |
|---|---|---|
| electrode 表 | 2013、2015 年只有 `electrode_group_M1`（96 个）；2016-09 起另有 `electrode_group_PMd`，loader 只取 M1 | 同 schema |
| units | 每个 unit 恰好映射 1 个 electrode；M1 sorted units 数 38–95 | 10–52 |
| `unit_to_electrode` | 96 个 canonical M1 electrode 中的索引；多个 unit 可共用一个 electrode，prior 按 electrode 平均后取列 | 同 |
| behavior / trials | `cursor_vel` 2 维，20 ms bin 中心插值；`result == R` 的 trial 内且在 `cursor_vel` 支持区内的 bin 为 eval bin | 同 |
| usable trials | 152–1038（2015-03 的 5 个 session 有 605–1038 个） | 173–733 |
| session 长度 | 30–190 k bins；2015-03 文件 255–389 MB，加载约 20 s/个，其余约 7 s/个 | 48–319 k bins |

sub-M 行为逐位不变：`test_subm_loader_is_bit_identical_to_pre_extension` 对 `sub-M_ses-CO-20150612`、`sub-M_ses-CO-20140306` 的 sha256 与扩展前 loader 比较（本次修订后仍通过）。根目录可由 `SSM_DANDI688_ROOT` 重定向。

### 2.2 禁止的 session 与保护层

**FORBIDDEN：** sub-C 的 6 个 APST final / test：`20151113`、`1116`、`1117`、`1119`、`1120`、`1201`；sub-M 的 3 个 final：`20150623`、`0625`、`0626`。证据（本机文件，三处一致）：`APST/src/apst/legacy_dandi/protocol.json`（与 `APST-public` 同 sha256）的 `final_sessions`；`SPINT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json` 的 `test`；`SPINT/btransform_unified_v2/dandi688_bench_v2/protocol.py` 的 `FINAL_SESSIONS`。2015-11 的 6 个 development session（1103、1104、1106、1109、1110、1112）是 APST 的 selection 集，不是 held-out，按 MZ 决定作为 month target。2016-09…10 的 13 个 session **不使用**（optional tier 关闭）。

保护层（v2 修订后，全部有测试）：

1. `load_dandi688_recording`：对 6 + 3 个 final 在打开任何文件之前 `PermissionError`（sub-M final 是 v2 新增；测试把 `_scan_session` 换成会报错的桩，验证未被调用）。
2. **`_scan_session` 本身**也在 `import pynwb` 之前检查 session id（v2 新增；测试把 `pynwb` 置空，验证 PermissionError 先于 ImportError）。
3. inventory：显式请求 final 时 `PermissionError`，整个 subject 扫描时自动跳过（sub-M 的 final 现在也被跳过；`dandi688_plan(include_final=True)` 仍只规划路径，加 `with_metadata=True` 会 `PermissionError`）。
4. 协议：`validate_backbone` 拒绝 pool 或 target 中出现 forbidden；`bundle_files` 拒绝把 forbidden 放进 bundle。
5. driver：每个 stage 的第一件事是 `guard_backbone(bb)`；`eval` 里**在 `world.target()` 之前**还有逐 target 的检查（v2：之前检查在 `world.target()` 之后；测试 `test_eval_checks_forbidden_before_loading_the_target` 绕过上游保护后验证 `world.target` 未被调用）。
6. 没有读取 `results/official_evalai/`。

### 2.3 披露：这些 sub-C session 在其他协议里是 exam / test face（不禁止，必须披露）

`SPINT/btransform_unified_v2/dandi688_bench_v1/src/dandi688_bench_v1/plan.py`：

| session | 其他协议里的角色 | 本扩展里的用途 |
|---|---|---|
| 20150713、0714、0715、0716 | `EXP1_NARROW_EXAM_SESSIONS`（`plan.py:682`）：exp1_narrow 与 exp1_poverty 的 exam face（“days-after” 考卷）；同时是 exp2_full / exp2015_full 的 train 成员 | sc_0713 的全部 4 个 target；sc_0708 的 days target 0713 / 0714；sc_0629 的 month target 0713 / 0716；sc_0309 的 year target 0716 |
| 20151103、1104、1106、1109、1110、1112 | exp2_full / exp2015_full 的 exam face（manifest 的 val = APST development session） | sc_0708 的 6 个 month target（`apst_dev=true`） |
| 20160909 … 20161021（14 个） | `EXP2016_EXAM_SESSIONS`（`plan.py:727`）：exp2016 的 exam face | **不使用** |
| 20150629…0710（9 个） | exp1_narrow 的 train 成员 | sc_0708 / sc_0713 的 pool 与 days target（0708、0709、0710） |

各协议的 face 是独立的测量；本扩展的 rows 不能与那些协议的 face 数字直接比较，也不能把这些 session 描述成“从未被任何实验用过”。另一类披露：sm_0611 的 target 20150616 / 0617 是 P1 的设计 target（旧 days pool 的 dev），这里用另一个 pool；sm_0627 / sm_0512 / sm_0611 的 pool[-1]（20140626、20150511、20150610）曾是 P1 / P3 的 target 或 source，现在作为选择 session，不是它们自己的 target。

## 3. sub-C 协议

### 3.1 规则

* **pool = 6 个比所有 target 都早的 session**（严格按日期）；前 5 个是 trunk sources，第 6 个 held-out（§1）。一个 backbone（AP trunk + P0 calibrator）服务该 pool 之后的多个 target。
* **strata**：由 target 与 `pool[-1]` 的间隔决定：days 1–7 天，month 90–130 天，year ≥ 365 天（`validate_backbone` 检查标签与间隔一致）。
* **K = 10**（与 sub-M P1 一致；Ledermann 上限 `⌊(2N+1−√(8N+1))/2⌋`，sub-C 各 pool 最小 N 为 41–67，上限 32–55；sub-M 新 pool 最小 N 为 18–34，上限 12–26）。pool 里 N<15 的 session 不入 pool（唯一受影响的是 sub-M 的 20141203，N=10）。
* **选择只用 source**：在 `pool[-1]` 上（§1）。
* target 至少 53 个 usable trial（B=33 + 20 个 query trial）；评测与 P1 一致：`draw_trials` 抽 B 个校准 trial，query = 其余 usable trial 的所有有效 bin，整段 recording 一次连续因果解码，物理单位 R²。

### 3.2 backbone 与 target（含 trunk gap 与 electrode 覆盖率）

“gap” = 到 pool[-1] 的天数（决定 stratum）；“trunk gap” = 到 trunk 最后训练 session 的天数；覆盖率 = target 的 unit 的 electrode 出现在 **trunk 5 个 session** 的 electrode 并集里的比例（prior 是 trunk table 的平均）。

**sc_0708**（OOS）trunk：20150629(49)、0630(44)、0701(49)、0703(52)、0706(46)；held-out：20150707(42)；最小 N=42，上限 33

| target | stratum | gap | trunk gap | N | trials | 覆盖率 | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| 20150708 | days | 1 | 2 | 64 | 190 | 0.77 | |
| 20150709 | days | 2 | 3 | 61 | 196 | 1.00 | |
| 20150710 | days | 3 | 4 | 55 | 203 | 1.00 | |
| 20150713 | days | 6 | 7 | 58 | 185 | 0.97 | |
| 20150714 | days | 7 | 8 | 66 | 199 | 0.88 | |
| 20151103 | month | 119 | 120 | 38 | 250 | 0.66 | APST dev |
| 20151104 | month | 120 | 121 | 59 | 319 | 0.59 | APST dev |
| 20151106 | month | 122 | 123 | 60 | 258 | 0.63 | APST dev |
| 20151109 | month | 125 | 126 | 65 | 173 | 0.68 | APST dev |
| 20151110 | month | 126 | 127 | 61 | 198 | 0.66 | APST dev |
| 20151112 | month | 128 | 129 | 42 | 188 | 0.59 | APST dev |

**sc_0713**（OOS）trunk：20150703(52)、0706(46)、0707(42)、0708(64)、0709(61)；held-out：20150710(55)；最小 N=42

| target | stratum | gap | trunk gap | N | trials | 覆盖率 |
|---|---|---:|---:|---:|---:|---:|
| 20150713 | days | 3 | 4 | 58 | 185 | 0.97 |
| 20150714 | days | 4 | 5 | 66 | 199 | 0.98 |
| 20150715 | days | 5 | 6 | 60 | 192 | 0.98 |
| 20150716 | days | 6 | 7 | 61 | 191 | 0.97 |

**sc_0629**（OOS）trunk：20131220(67)、20150309(74)、0311(88)、0312(91)、0313(86)；held-out：20150319(72)；最小 N=67，上限 55。2015-03 只有 5 个 session，用 20131220 补满 6 个，跨度 15 个月（混龄 pool，§9）

| target | stratum | gap | trunk gap | N | trials | 覆盖率 |
|---|---|---:|---:|---:|---:|---:|
| 20150629 | month | 102 | 108 | 49 | 179 | 0.96 |
| 20150701 | month | 104 | 110 | 49 | 196 | 0.98 |
| 20150706 | month | 109 | 115 | 46 | 184 | 1.00 |
| 20150709 | month | 112 | 118 | 61 | 196 | 0.98 |
| 20150713 | month | 116 | 122 | 58 | 185 | 0.97 |
| 20150716 | month | 119 | 125 | 61 | 191 | 0.97 |

**sc_0309**（OOS）trunk：20131031(49)、1101(41)、1203(48)、1204(52)、1219(65)；held-out：20131220(67)；最小 N=41，上限 32

| target | stratum | gap | trunk gap | N | trials | 覆盖率 |
|---|---|---:|---:|---:|---:|---:|
| 20150309 | year | 444 | 445 | 74 | 605 | 0.81 |
| 20150312 | year | 447 | 448 | 91 | 905 | 0.74 |
| 20150319 | year | 454 | 455 | 72 | 1026 | 0.86 |
| 20150629 | year | 556 | 557 | 49 | 179 | 0.78 |
| 20150706 | year | 563 | 564 | 46 | 184 | 0.85 |
| 20150716 | year | 573 | 574 | 61 | 191 | 0.74 |

month 层的两个独立来源：sc_0629（2015-03 pool → 2015-07，不含任何 APST session）和 sc_0708 的 dev target（pool 更紧凑，但这 6 个是 APST dev session，MZ 已决定使用）。2016 年 year target（原 sc_0708 的 6 个 optional target）已**关闭**，不在 bundle 中。

## 4. sub-M days 层 rolling backbones 与聚合

### 4.1 backbone 列表

规则：pool = 目标日期之前**最晚的 6 个**合格 session（N ≥ 15、非 final，`ep.rolling_pool`）；测试证明 3 个新 sub-M pool 等于该规则的输出。

| backbone | scope | trunk sources（N） | held-out / 选择 session | target（gap / trunk gap） |
|---|---|---|---|---|
| sm_0307 | **IS**（复用） | 0203(34)、0217(44)、0218(38)、0303(52)、0304(39)、0306(43)（现有 month trunk，P1 month_O 的 P0） | 20140306(43)，trunk 见过 | 20140307（1 / 1） |
| sm_0627 | OOS | 20140218(38)、0303(52)、0304(39)、0306(43)、0307(26) | 20140626(26) | 20140627（1 / 112） |
| sm_0512 | OOS | 20140306(43)、0307(26)、0626(26)、0627(25)、0929(21) | 20150511(24) | 20150512（1 / 225） |
| sm_0611 | OOS | 20140626(26)、0627(25)、0929(21)、20150511(24)、0512(30) | 20150610(18) | 0611（1/30）、0612（2/31）、0615（5/34）、0616（6/35）、0617（7/36） |

要点：20140307 是旧 month 规则的 causal selection session；这里它只是 sm_0307 的 target，**不参与任何选择**（选择只在每个 backbone 的 pool[-1] 上做；它在 sm_0627 / sm_0512 里是 trunk source，与旧规则无关）。20141203（N=10）被跳过。final（20150623/25/26）不在任何 pool 或 target 里。

### 4.2 评测管线（每个 backbone 的五个阶段）

1. **backbone**：`python -m ssm_decode.ap_pretraining --task dandi688sm --k 10 --width 128 --layers 2 --state-size 32`，`--dandi688-sources` 为**前 5 个**；起始 2000 步，`best_step > 0.8 × steps` 则预算翻倍重训（2000 → 4000 → 8000 → 16000，最多 4 次，每次从头；到顶仍不收敛时 `converged=false` 如实记录）。
2. **p0**：2 个 seed 的 P0（independent per-unit encoder，residual 输出，episodic，unit dropout，leave-one-out 的 electrode-aligned prior）；5 个 trunk session 训练，**pool[-1] 做 validation 与 early stopping**；起始 1000 步，best step 落在最后 20% 就翻倍（最多 4 次）。直接调用 P1 的 `run_population_calib.cmd_train`。
3. **jacobian**（v2 新增）：在 trunk 的 5 个 source session 上用 autograd 离线计算平均 Jacobian J̄（lmax=32，每个 session 600 个 width-320 窗口，`default_rng(0)`，与 P3 `cmd_jacobian` 相同），保存 `jacobian/<bid>.pt`。窗口宽度固定，所以 Triton backward 只对一个序列长度 autotune。
4. **select**：在 `pool[-1]` 上做 source simulation，每个 (seed∈2, B∈{1,4,8,16,33}, draw∈4) 一个模拟 B-trial 校准 episode（40 个 episode）：
   * R1-stream：P0 表 + 流式累加 readout 统计，5 种 readout kind（full、diag、lr8、lr16、lr32）× 9 个 λ，每个 B 选 source 分数最高的 (kind, λ)；
   * R2（F7-J）：L ∈ {8,16,32} × η ∈ {1,3,10,30,100,300,1000} × ρ ∈ {0,0.03,0.1,0.3} × S ∈ {1,3,5,10}（P3 的网格），每个 B 选 source 分数最高的 (L, η, ρ, S)；
   * R3：用同一 B 选出的 R2 规则得到 F7-J 表，再对该表做 readout ridge，选 (kind, λ)。
   规则存入 `select/<bid>.json`（含 `selection_oos`、`selection_scope`、网格）；episode 级日志 `select/<bid>_episodes.jsonl`、`select/<bid>_r3_episodes.jsonl`。PCA basis 只用 trunk sources。
5. **eval**：每个 target 的行 `rows/<bid>/<date>.jsonl`：`p_prior`（seed 无关，只记 seed 0）、`p0`、`r1s`、`r2`、`r3`。

资源规则：trunk 冻结、target 端无反传（`h2_backward_calls` 必须为 0）、只用 B 个有标签 trial、选择只用 source（OOS）、连续因果解码、无 final。

### 4.3 R2 / R3 的 target 端路径

`ep.refine(J̄_L, η, ρ, [S])` 是对 B 个校准 trial 的 S 步 forward-only F7-J（forward → residual → 固定时间卷积 → 外积累加 → 缩向 P0 表）；R3 在 F7-J 表上用 `ep.stats(tab)` 的 replay 统计做 readout ridge。所有 target 端代码在 `no_grad` 下执行，autograd 调用数被计数并写入每一行。

### 4.4 聚合：按 target session 聚类（审查修正 3）

* 27 个 sub-C (backbone, target) 对只覆盖 **19 个不同 session**：0713、0716 各 3 个 backbone；0629、0706、0709、0714 各 2 个（审查清单里漏了 0714：它同时是 sc_0708 与 sc_0713 的 target）。`ep.duplicate_target_report` 和 `protocol.json["duplicate_targets"]` 给出完整列表。sub-M 的 8 个 target 是 8 个不同 session，其中 sm_0611 的 0616 / 0617 是 P1 设计 target。
* `scripts/analyze_expansion.py` 的推断单位是 **session**：先对每个 (backbone, target, B, 对比) 在 draw × seed 上取均值，再在同一 session 的各 backbone 间取均值，最后对 session 做 10,000 次 bootstrap 与 t 区间；同时报告 `n_sessions`、`n_backbone_target_pairs`、有多个 backbone 的 session 数；**backbone 层面的离散**（各 backbone 均值的 min / max / sd、同一 session 跨 backbone 的最大差）在 `backbone_spread.csv`。分层：subject × stratum × 选择范围（OOS / IS）× B（含 B4–33 汇总）× 对比（p0−p_prior、r1s−p0、r2−p0、r3−p0、r3−r2）；另有按 trunk gap 分层的表。
* IS（sm_0307）单独成组；只有 1 个 session 的组不报告 CI；缺 `rule_version == expansion_v2` 的 rows（P1 / P3 / drift-curve）被拒绝。
* 结果 README：`results/analysis/expansion_20261016/README.md` 把上述规则和 §2.3 的披露写在读者最先看到的位置。

## 5. 实现细节（审查修正 5–8）

### 5.1 forbidden 检查
见 §2.2。

### 5.2 `--tier all` 与 2016
`--tier all` 现在需要 `--include-2016-optional`，否则 `SystemExit`；带上它时在 stderr 和 warnings 中记录 `WARNING: ... 2016 sub-C sessions (optional year tier) are included`。`run_expansion_queue.sh` 透传该选项。默认 `--tier primary` 不含 2016。

### 5.3 逐行审计字段
每行：`h1_trunk_hash`（trunk SHA-256 前 16 位）、`h1_trunk_unchanged`（每个 (B, draw, seed) 组之后重新计算）、`h2_backward_calls`、`h4_budget`、`h4_query_disjoint`、`h5_bytes`、`h5_within_32kb`（read-in table，P1 口径；`r1s` / `r3` 另有 `h5_readout_bytes`，因为 readout 状态在 table 之外）、`unit_check_passed` / `unit_ratio`（物理单位检查，`rpc.r2_physical` 的 `check_physical_units`，与 P1 同源）、`selection_oos`、`selection_session`、`rule_version`、`gap_days`、`trunk_gap_days`、`apst_dev`、`tier`。`score()` 复用 `Ep.r2` 的同一次解码并取回单位检查。

### 5.4 原子写入
所有 rows / selection episode 日志用 `atomic_jsonl.append_jsonl`（把已有的有效行加上新行写到 `.tmp`，`fsync` 后 `os.replace`）。读取时容忍被截断的末行（旧版本 driver 在中途被杀可能留下），`status` 与每个 stage 开始前 `repair_jsonl`；非末行损坏则报错。rows 文件每个 target 约 0.2 MB，重写的代价可以忽略。JSON 已经是 temp + replace。

## 6. 运行命令与退出码

环境（host）：`PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM /home/xinyuan/miniconda3/envs/spint/bin/python`。

```bash
cd /home/xinyuan/Work_host/SSM
python scripts/run_expansion.py check            # 协议审计（CPU，秒级）；--write 重写 protocol.json / target_check.json
python scripts/run_expansion.py status
scripts/run_expansion_queue.sh --jobs 1           # 本机（先看 nvidia-smi / tmux ls）
scripts/run_expansion_queue.sh --bids sm_0627,sm_0512,sm_0611 --jobs 2
scripts/run_expansion_queue.sh --exclude-apst-dev
python scripts/smoke_expansion_cpu.py             # CPU 端到端 smoke（GRU trunk）
python scripts/expansion_cost.py                  # 成本模型
python scripts/analyze_expansion.py --rows <out>/rows --out results/analysis/expansion_20261016/summary
```

**退出码（审查修正 1）：** `run_expansion_queue.sh` 退出 0 = 每条链都完成；**3** = 至少一条链在 `--deadline-hours` 检查处停止，且没有失败（重跑同一条命令即可恢复）；其他 = 第一条失败 stage 的返回码（预检失败为 1，xargs 失败但没有记录到链返回码时为 1）。失败的链不会停止其他链。链的返回码写在 `<out>/logs/.rc_<bid>`，脚本末尾汇总（之前 xargs 的失败被吞掉，退出码总是 grep 的）。测试用 fake stage 覆盖：全部成功、stage 失败、失败优先于 deadline、deadline rc 3、预检失败。

### Colab（两轮）

环境按 `docs/colab_runbook.md` §1 逐步搭建：zstd；bundle 拷到本地并校验 SHA256；repo 与数据放在和 host 相同的绝对路径；APST 源码（`apst_src.tgz`）；`pip install pynwb==4.0.0 hdmf==6.1.0`；`pip install --target …/.tools/mamba_deps triton==3.5.0`；`git config --global --add safe.directory …/.tools/mamba_official`；然后做 runbook 的 **Mamba-3 check**——`tests/test_mamba3_official.py` 会空转通过（接受 `unsupported_or_unavailable`），必须看 `compatibility_attempt('cuda:0')` 打印 `"status": "supported"`、`triton_version 3.5.0`、`kernel_triton_matches_runtime true`、`causal_future_perturb_max_abs 0.0`，否则 trunk 训练会静默失败。解压 `data_000688_subC.tar.zst`（路径 `dandi_000688/sub-C/…`）到 `/home/xinyuan/Work_host/SPINT/sua_exploration/data`。另需新代码 bundle（本提交之后重新打包）和复用的 sm_0307 小文件（§7）。

```bash
cd /home/xinyuan/Work_host/SSM
# round 1: sub-C days + month（21 个 target）
PY=python scripts/run_expansion_queue.sh --bids sc_0708,sc_0713,sc_0629 \
    --out /content/drive/MyDrive/DAC27/colab/expansion_out --work /content/expansion_work --jobs 3 --deadline-hours 10
# round 2: sub-C year + sub-M days（14 个 target）
PY=python scripts/run_expansion_queue.sh --bids sc_0309,sm_0307,sm_0627,sm_0512,sm_0611 \
    --out /content/drive/MyDrive/DAC27/colab/expansion_out --work /content/expansion_work --jobs 3 --deadline-hours 10
echo "queue exit status: $?"        # 0 = done, 3 = deadline (re-run), other = a stage failed (see <out>/logs/<bid>.log)
```

* `--out` 持久根目录（Drive），`--work` 本地 scratch；每完成一个工作单元（AP 尝试、P0 seed、Jacobian、selection episode 块、(target, B) 块）就原子地复制到 `--out`。断线后用同一条命令重跑恢复。
* 并发：GPU 是瓶颈（runbook：3 个 job 共享 L4 时 backward 33 → 130 ms/step），并发的主要收益是让 CPU 数据加载（每个阶段重读 pool）与别的 job 的 GPU 计算重叠；因此成本按顺序执行估计，`--jobs 3` 只会更快不会更慢。`--jobs 4` 在 runbook 里也跑过。
* 两轮的 `--out` 可以相同，因为 backbone id 不同；`status` 汇总全部。
* 完成后把 `rows/`、`select/*.json`、`backbones/*/backbone.json`、`jacobian/`、`runs/*/*/seed*/done.json` 同步回 host，再跑 `analyze_expansion.py`。
* 风险：第一个 target 的耗时要看一眼——如果 forward 在新序列长度上被 Triton 重新 autotune（runbook 对 backward 记录过），第一轮会明显变慢；监控 `eval` 日志里 `t=…s`，必要时调整 `--deadline-hours`。

## 7. Colab 数据与代码 bundle（**未上传**，由 coordinator 执行 rclone）

**数据** `data_000688_subC.tar.zst`（NWB 已压缩，预计接近原始大小）：

* 成员列表：`results/analysis/expansion_20261016/colab_bundle_subC_primary.txt`（30 个，**3,568,531,938 B ≈ 3.57 GB / 3.32 GiB**）。v2 删除了含 2016 的 `colab_bundle_subC_all.txt`（2016 的 6 个文件 1.54 GB 不再出现在任何成员列表里）。
* 打包（路径与 sub-M bundle 一致）：

```bash
cd /home/xinyuan/Work_host/SPINT/sua_exploration/data
tar -I zstd -cf /tmp/data_000688_subC.tar.zst \
    --transform 's,^sub-C/,dandi_000688/sub-C/,' \
    -C dandi_000688 -T /home/xinyuan/Work_host/SSM/results/analysis/expansion_20261016/colab_bundle_subC_primary.txt
tar -tf /tmp/data_000688_subC.tar.zst | grep -E '2015(1113|1116|1117|1119|1120|1201)|CO-2016' && echo "FORBIDDEN PRESENT" || echo clean
```

| 组 | 文件数 | 大小 |
|---|---:|---:|
| 2013-10…12（sc_0309 pool：1031、1101、1203、1204、1219、1220） | 6 | 254 MiB |
| 2015-03（0309、0311、0312、0313、0319） | 5 | 1,734 MiB |
| 2015-06/07（13 个） | 13 | 925 MiB |
| 2015-11 dev（6 个） | 6 | 489 MiB |
| **primary 合计** | **30** | **3,403 MiB（3.32 GiB）** |

sub-M 需要的 18 个文件已在现有 `data_000688_subM.tar.zst` 里（该 bundle 同时含 sub-M final，别读——现在 loader 也会拒绝）；名单在 `colab_bundle.json::subM_needed_in_existing_bundle`。

**代码**：在含本修订的提交之后重新打包 `code.tar.zst`；`results/analysis/expansion_20261016/colab_bundle_code_files.txt` 列出扩展需要的 22 个已跟踪文件（共约 326 KB；含 `session_inventory.json` 和新模块）。**复用的小文件**（必须随 `ckpts.tar.zst` 或另行带上，共约 1.7 MB）：`results/stage2_intervals_20261006/month_pool_k10_x8/best.pt`（1.47 MB）、`results/population_calib_20261015/runs/month_O/p0/seed{0,1}`（各约 110 KB）。sm_0307 的 Jacobian / basis / 规则在 Colab 上重新计算（不复用 P3 的文件）。

## 8. GPU 小时与 Colab CU 估计（v2：含 R2 / R3、Jacobian、2015-03 加载）

没有 GPU smoke（本机 GPU 被占用）。`scripts/expansion_cost.py` 的模型由已有日志推算（每个常数在脚本注释里有出处）：

| 项 | 3090 数值 | 来源 |
|---|---|---|
| AP 链 | 期望 155 s（2 次尝试），上界 391 s（2000/4000/8000/16000 步 = 99+56+87+148 s） | `stage2_intervals_20261006/*/train_log.jsonl` |
| P0（每 seed） | sub-M 120–172 s；sub-C ×1.5 = 180–260 s | P1 `done.json`（81–172 s） |
| Jacobian | 60 s / backbone | P3 `jac_year.log`（6 个 session，lmax 32：57 s） |
| select（R1-stream + 84 条 R2 规则 / episode） | 20–28 s / episode；R3 阶段 2–3 s / episode | P3 select 日志（14 s 无 stream 的 contig 版 … 28 s，3 个 job 并发时测得）；`select-r3_month_oosc.log`（46 s / 24 episode） |
| eval | 1.2–2.4 s / (draw, seed) 组 | P3 `eval_month_oosc.log`：228 s / 96 组 = 2.4 s，其中包含我们不算的所有 readout kind；本队列只算 p_prior、p0、r1s、r2、r3 |
| 进程开销 | 30 s / stage 进程 | 估计 |
| 数据加载（CPU） | 约 7 s / 文件；≥150 MB（2015-03）约 20 s / 文件 | MZ；P1 / 先前扫描 |

每阶段重读 pool：AP 链每次尝试读 5 个 trunk session，P0 每个 seed 读 6 个，jacobian / select 各读 6 个，eval 读 5 个 + 每个 target 一个。**L4 比 3090 慢多少没有实测，取 ×2（范围 1.5–2.5）。**

按 L4 ×2、`--sel-draws 4`（每个 (seed, B) 4 个 draw，即每个 backbone 40 个 selection episode）：

| backbone | target 数 | 期望 wall (h) | 上界 wall (h) |
|---|---:|---:|---:|
| sm_0307（复用，只有 jacobian/select/eval） | 1 | 0.64 | 0.89 |
| sm_0627 | 1 | 0.95 | 1.43 |
| sm_0512 | 1 | 1.02 | 1.52 |
| sm_0611 | 5 | 1.28 | 2.01 |
| sc_0708 | 11 | 1.55 | 2.59 |
| sc_0713 | 4 | 1.16 | 1.83 |
| sc_0629 | 6 | 1.39 | 2.19 |
| sc_0309 | 6 | 1.28 | 2.06 |
| **合计（顺序）** | 35 | **9.3 h ≈ 14.3 CU** | **14.5 h ≈ 22.4 CU** |

构成（L4 小时，期望 / 上界）：AP 0.6 / 1.5；P0 1.2 / 1.7；Jacobian 0.3 / 0.3；**selection（R2 网格）3.9 / 5.5**；eval 1.9 / 3.7；CPU 加载 1.0 / 1.2。selection 是最大的一项；`--sel-draws 2`（P3 的 4 episode / B）可把总量降到期望 7.3 h（11 CU）/ 上界 11.8 h（18 CU），代价是规则噪声更大。

* **一轮的 CU 上限 40**：整个队列上界 22 CU（L4 因子 2.5 时 27 CU）< 30 CU，**CU 不是约束**；但上界 14.5 h（因子 2.5 时 17.7 h）超过一个约 12 h 的 session（`--deadline-hours 10.5`）。因此**建议拆成两轮**：

| 轮 | backbone | target | 期望 | 上界 |
|---|---|---:|---|---|
| 1 | sc_0708、sc_0713、sc_0629（sub-C days + month） | 21 | 4.1 h / 6.3 CU | 6.6 h / 10.2 CU |
| 2 | sc_0309、sm_0307、sm_0627、sm_0512、sm_0611（sub-C year + sub-M days） | 14 | 5.2 h / 8.0 CU | 7.9 h / 12.2 CU |

两轮合计期望 14.3 CU、上界 22.4 CU；每轮加上 session 启动、拷贝数据（round 1 约 3.35 GB sub-C 文件；round 2 约 1.56 GB sub-C + 2.08 GB sub-M）和 Mamba-3 check 约 0.3 h（0.5 CU）。每轮的上界都低于 `--deadline-hours 10`，L4 因子 2.5 时 round 2 上界 9.6 h，仍在限内。优先级建议：如果只能跑一轮，先跑 round 1。
* 如果第一轮实测 L4 速度明显不同，用 `--jobs`、`--deadline-hours`、`--sel-draws` 调整即可，不需要改代码。

## 9. 开放问题（请 MZ 决定）

1. **混龄 pool。** sc_0629（含 20131220）与 sm_0512 / sm_0611 / sm_0627 的 pool 跨度 1–15 个月；这是“最晚 6 个合格 session”的直接后果，更紧凑的 pool 在数据里不存在。OOS 改动之后，held-out session 与 trunk 最后训练 session 之间也有间隔（sm_0627：约 3.6 个月，sm_0512：约 7.4 个月），所以这三个 sm_* backbone 的“days”标签对 trunk 来说是 month / year 量级——解读时请用 trunk-gap 分层表。
2. **K=10。** sub-C 的 Ledermann 余量大（上限 32–55），为了与 sub-M 可比保持 10。
3. **sel-draws。** 默认 4 draw / (seed, B)；降到 2 可省 2–2.7 L4 小时（约 4 CU）。
4. **sm_0307 的 IS 选择。** 若想要一个 OOS 的 sm_0307 对照，需要另训一个 5-session trunk（现有 month trunk 看过 20140306），不在本任务内。

## 10. 测试

CPU，全部通过（见最终汇总）：

* `tests/test_dandi688_subc.py`：sub-M 逐位回归、sub-C 合同、6 个 sub-C final **和 3 个 sub-M final** 的 loader / inventory / `_scan_session`（在 pynwb import 之前）保护、整个 subject 扫描跳过 final、RT 与其他 subject 被拒；
* `tests/test_expansion_protocol.py`：pool 构造、forbidden、tier、**trunk_sources / heldout / OOS / IS 标记、trunk gap、`d1_spec`、重复 target 报告（27 对 = 19 session）、primary 不含 2016**；
* `tests/test_run_expansion.py`：Store、`--tier all` 需要额外 flag 并告警、**每个新 backbone 的 AP 命令不含 pool[-1]**、含 pool[-1] 的 checkpoint 被拒、`World` 的 train/val 划分（pool[-1] 只做 validation）、trunk-only Jacobian、forbidden 检查先于 `world.target()`、每个 stage 拒绝含 final 的 backbone、被截断的 rows 文件的 status 修复、**队列退出码**（fake stage：成功 / 失败 / 失败优先 / deadline rc 3 / 预检失败）、bundle 清单（30 个，无 forbidden / 2016）；
* `tests/test_atomic_jsonl.py`：原子追加（`os.replace` 前被杀不留半行）、截断末行恢复、中间行损坏报错；
* `tests/test_expansion_rules.py`：R1 / R2 / R3 规则选择、tie 确定性；
* `tests/test_analyze_expansion.py`：按 session 聚类、CI 不比朴素 pair CI 窄、backbone 离散、IS / OOS 不合并、拒绝其他协议的 rows、截断末行容忍、审计 flag；
* `tests/test_expansion_cost.py`：成本模型的单调性、整体 < 40 CU、两轮计划覆盖全部 backbone 且每轮 < 10.5 h。

## 11. CPU smoke

`python scripts/smoke_expansion_cpu.py`（真实 sub-C 数据：backbone sc_0713，target 20150714；官方 Mamba-3 换成同接口的小 GRU trunk；极小预算与极小 R2 网格）。**SMOKE OK，约 12 分钟（CPU）。**

* 五个 stage 全部跑通：AP 预算扩展规则触发（20 → 40 步到 `--ap-attempts 2` 的顶后 `converged=false` 被如实记录）；P0 两个 seed 在 20 步起按同一规则翻倍到 320 步（占了大部分时间）；`jacobian` 只写 5 个 trunk session（断言通过）；`select` 写出 `select/sc_0713.json`（`selection_oos=true`、每个 B 的 R1/R2/R3 规则）及两个 episode 日志；`eval` 写出 18 行 = 2 个 B × 1 draw × 9（`p_prior` 一行 + 4 个方法 × 2 个 seed）。
* 断言：trunk checkpoint 的 `source_sessions` 恰为前 5 个 pool session，不含 pool[-1]；P0 的 `train_ids` 不含 pool[-1]、`val_ids == [pool[-1]]`；所有行 `h2_backward_calls=0`、`h4_query_disjoint=True`，并含 `h1_trunk_hash / h1_trunk_unchanged / h5_bytes / h5_within_32kb / unit_check_passed`。
* **可恢复性：** 同一进程里第二遍调用所有 stage，全部跳过（eval 日志里 `t=0s`），行数不变。
* 数字本身没有意义（玩具 GRU trunk、20 步、6 个 Jacobian 窗口、η=10 的单点 R2 网格；R² 约 p_prior 0.07、p0 0.16、r1s 0.16、r3 0.00，玩具 R2 为负），只说明链路是通的。

## 12. Round 2（frozen methods，causal-first-B）

不改 `backbone / p0 / jacobian / select / eval` 的语义。默认队列阶段仍是这五个。Round 2 在它们后面加两个 stage，行写到 `causal_rows/`，不写进原来的 `rows/`（原来的行是 random draw）。

### 12.1 Stage

* **`causal-source`**（每个 OOS backbone 一条）：在 held-out `pool[-1]` 上按 causal-first-B 记 alpha、S1、COMBO（每个 α 一张 grid）、CCA+S1。`sm_0307` 是 IS，写 `skipped: IS`，不进池。已有的 legacy episode（`results/causal_main_20261019/select/`，days / month / year / sc_0629 / sc_0708）按 `(tag, session, seed, B, alpha)` 合并；本地行优先。month 与 year 是同一套 trunk 权重，同一 session 只留 month。
* **`causal-eval`**（等所有 chain 的 `causal-source` 结束之后再跑，队列脚本把这一阶段从 chain 里拿出来做第二遍）：每个 target 用 frozen method 打分。校准集是前 B 个 usable trial，不是 draw。Latency 0 是主行，1 是 sensitivity。

方法：`p_prior`、`p0`、`ridge_cca`（sub-C 走修好的 `nwb_path_for_session`，CCA 配置沿用 sub-M month 的 `CCA_CFG`，不在 sub-C 上重选）、`w_alpha_a` / `w_alpha_b`、`s1_causal`（协议 (a) 的 AND）和 `s1_causal_b`、`combo_a` / `combo_b`、`cca_s1` / `cca_s1_b`，以及它们的 `*_cons`（≥2 个 selection session 增益为正才部署，否则退回基座）、`r2` / `r3`（replay：用该 backbone 已有的 source 规则，在前 B 个 trial 上做 final-only commit）。Candidate 的 `h2_backward_calls` 必须为 0。`20140307` 只作为 `sm_0307` 的 target，不当作新的 selection session。2016 tier 仍关闭。Final session 仍被 loader 拒绝。

选择池：legacy episode，加上每个 OOS backbone 的 held-out `pool[-1]`，但只保留协议允许的。**(a) expanding-window**：episode 的 session 和 trunk 最后一个训练日都严格早于 target。**(b) strict**：再去掉已报告的 target（causal-main 的 drift / sub-C month，加上本扩展 primary target）以及在这些 target 上训练过的 trunk。每个 target 的组成写在 `causal_select/composition/<bid>_<date>.json`（`n_sessions` 是不同的日期，`n_trunk_sessions` 把同一天的不同 trunk 分开计）。池不齐时 `causal-eval` 直接退出；smoke 才用 `--allow-incomplete-pool`。

`score_tables` 里的 batched trunk forward 若和逐张 forward 不一致（3090 smoke 上 max abs 约 9.9e-3），会退回逐张 forward。分数仍然对，source 阶段会更慢。上界按这条慢路径留了余量。

### 12.2 Colab 命令

`--out` 放 VM 本地盘。`--sync-tar` 每 30 min 把 `--out` 打成 tar，先写 `.partial` 再 `os.replace` 到 Drive 路径（和 round 1 一样）。先把 `expansion_out_round1.tgz` 解进同一个 `--out`，这样 sc_0708 / sc_0713 / sc_0629 不用重训。Legacy select 目录放到 `--legacy-select`。

```bash
cd /home/xinyuan/Work_host/SSM
# session 2a: 新 backbone 的旧 stage + 全部 OOS 的 causal-source（含 round-1 的三个）
PY=python scripts/run_expansion_queue.sh \
    --bids sc_0309,sm_0307,sm_0627,sm_0512,sm_0611,sc_0708,sc_0713,sc_0629 \
    --stages backbone,p0,jacobian,select,eval,causal-source \
    --out /content/expansion_out --work /content/expansion_work \
    --artifacts /content/expansion_out \
    --legacy-select /home/xinyuan/Work_host/SSM/results/causal_main_20261019/select \
    --jobs 2 --deadline-hours 10 \
    --sync-tar /content/drive/MyDrive/DAC27/colab/expansion_out_round2.tgz \
    --sync-every-min 30
# session 2b: 等 2a 的 causal-source 都在 --out 里之后，打 frozen-method 行（含 round-1 重打分）
PY=python scripts/run_expansion_queue.sh \
    --bids sc_0309,sm_0307,sm_0627,sm_0512,sm_0611,sc_0708,sc_0713,sc_0629 \
    --stages causal-eval \
    --out /content/expansion_out --work /content/expansion_work \
    --artifacts /content/expansion_out \
    --legacy-select /home/xinyuan/Work_host/SSM/results/causal_main_20261019/select \
    --jobs 1 --deadline-hours 10 \
    --sync-tar /content/drive/MyDrive/DAC27/colab/expansion_out_round2.tgz \
    --sync-every-min 30
```

`causal-eval` 不在 chain 里并行：它要等每个更早的 held-out 都已打分。`--jobs 1` 是因为单个 target 的 prefix cache 已经占满一张卡上的 trunk；2a 的 `--jobs 2` 只重叠数据加载。若 2b 的第一个 target 超过 15 min，改成下面的 fallback，不要把 deadline 拉过 10 h。

```bash
# fallback：2b-subc 然后 2b-subm（命令同上，只改 --bids）
# 2b-subc: sc_0309,sc_0708,sc_0713,sc_0629
# 2b-subm: sm_0307,sm_0627,sm_0512,sm_0611
```

### 12.3 CU / 时间（L4，1.54 CU/h）

CU 按 session 的 wall clock 计，不按 GPU job 相加。Round 1 实测 3 个 backbone、1 h 51 min、3.1 CU。下表是按这个速度外推的，上界假设 AP budget 翻倍，并且每个 causal target 大约是 round-1 eval 的 4 倍。数字在 `ssm_decode.expansion_causal.round2_cost_plan`。

| session | 内容 | 期望 | 上界 |
|---|---|---|---|
| 2a | 5 个新 backbone 的旧 stage + 全部 OOS `causal-source` | 4.5 h / 6.9 CU | 8.5 h / 13.1 CU |
| 2b | 全部 primary target 的 `causal-eval`（含 round-1 重打分） | 2.5 h / 3.9 CU | 8.0 h / 12.3 CU |
| **合计** | | **10.8 CU** | **25.4 CU** |
| fallback 2b-subc | 只打 sub-C | 2.0 h / 3.1 CU | 6.5 h / 10.0 CU |
| fallback 2b-subm | 只打 sub-M | 0.8 h / 1.2 CU | 2.5 h / 3.9 CU |

每一段的上界都低于 12 h，合计上界低于 40 CU。不需要把 2a 和 2b 放进同一个 session。

### 12.4 Bundle（不上传）

清单：`results/analysis/expansion_20261016/round2_bundle.json`。

* 已在 `gdrive:DAC27/colab/`：`data_000688_subM.tar.zst`、`data_000688_subC.tar.zst`（30 个 primary sub-C NWB）、`ckpts.tar.zst`、`expansion_out_round1.tgz`。没有新的 NWB。
* 需要重新打的 code tar：本 commit 的 repo（外加 runbook 里的 `.tools/mamba_official`）。扩展专用文件列表 `colab_bundle_code_files.txt`，合计 **543,077 B**。
* 还不在 Drive 上的小数据：`results/causal_main_20261019/select/`，**1,947,673 B**，42 个文件。这是 legacy selection episode，round 2 的池要读它。

### 12.5 测试与 smoke

* `tests/test_expansion_causal.py`：协议 (a)/(b) 丢掉不允许的 session / trunk（含 month==year 同一 trunk 的重复、days trunk 在 (b) 下整支退出、sc_0713 的 held-out 只在 trunk 早于 target 时投票、2016 与 final 不进池）；≥2 session 的门在只有一个日期时退回；新 stage 的 unit 可恢复；`--sync-tar` 原子替换；`causal-eval` 排在所有 chain 之后；B=8、α=0.75、η=1 的前缀扰动（query 和块内后缀都不改还不能读它们的 commit）。
* CPU smoke：`scripts/smoke_expansion_causal_cpu.py`，GRU 代替 Mamba-3，sc_0713 / 20150714，B=8。第二遍 `causal-source` 跳过，`causal-eval` 不追加行。`h2_backward_calls=0`。R² 无意义。
* GPU smoke（`cuda:0`，round-1 的真实 sc_0713 trunk，B=8，seed 0，latency 0，`--allow-incomplete-pool`，空的 legacy 目录）：`causal-source` 与 `causal-eval` 都跑完，19 行，`h2=0`，R2/R3 `replay=true`。再跑一遍 `causal-eval` 打印 `already complete`。这个 smoke 的 COMBO 行是 unavailable：只打了 α=0.5，而协议 (a) 选出的 α 不是 0.5。Colab 用完整 α grid，不会这样。
