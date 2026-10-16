# 证据扩展：sub-C 第二受试者 + sub-M days 层 rolling backbones（2026-10-16）

状态：**代码、协议、队列已准备并通过 CPU 测试；没有任何 GPU 训练。** 本机 GPU 在准备期间被其他任务占用（`nvidia-smi`：11 GB / 100% util），所以没有做 GPU smoke test；GPU 时间由已有训练日志推算（§7）。

代码与数据文件：

| 文件 | 作用 |
|---|---|
| `ssm_decode/dandi688_data.py` | loader 扩展：sub-C CO；sub-C 的 6 个 APST final session 在 loader / inventory 中 `PermissionError`，没有 override |
| `ssm_decode/expansion_protocol.py` | 协议（pool、target、strata、K、规则校验）；纯 Python，不依赖 NWB / torch |
| `scripts/build_expansion_inventory.py` | 一次性扫描 sub-M / sub-C 的所有**允许**的 CO session，生成 `results/analysis/expansion_20261016/session_inventory.json` |
| `scripts/run_expansion.py` | 分阶段 driver：`check / bundle / status / backbone / p0 / select / eval` |
| `scripts/run_expansion_queue.sh` | 可恢复队列（每个 backbone 一条链，`--jobs N` 并发） |
| `scripts/smoke_expansion_cpu.py` | CPU 端到端 smoke（官方 Mamba-3 需要 CUDA，所以换成同接口的 GRU trunk） |
| `tests/test_dandi688_subc.py`、`tests/test_expansion_protocol.py`、`tests/test_run_expansion.py` | loader / forbidden guard / pool 构造 / driver 测试 |
| `results/analysis/expansion_20261016/` | `session_inventory.json`、`protocol.json`、`target_check.json`、`colab_bundle*.{json,txt}` |

## 0. 摘要

* 8 个 backbone（4 个 sub-M days，其中 1 个复用现有 month trunk；4 个 sub-C）；**新训练 7 个 AP backbone**，另外复用 1 个。
* 35 个 primary target（sub-C 27 个：days 9 个、month 12 个、year 6 个；sub-M days 8 个）。另有 6 个 optional 的 sub-C year target（2016 年 session，见 §3.4 和 §9 的开放问题），默认不跑。
* 评测行：P-prior、P0、P0+R1-stream（来源选择的 readout kind 和 λ）；B ∈ {1,4,8,16,33}；每个 (target, B) 8 个 draw；P0 两个 seed；连续因果解码；每 target 200 行，primary 共 7000 行。
* 估计：约 3.3 GPU 小时（RTX 3090 口径），L4 约 6.7 小时（顺序，假设 L4 比 3090 慢 2×）；`--jobs 4` 并发的 wall 约 2–3 小时，**约 4–10 CU**（1.54 CU/h），预留 8 CU 即可（上限 40 CU）。
* Colab 数据 bundle：sub-C primary 30 个 NWB，**3.57 GB（3.32 GiB）**；含 optional 共 36 个，5.11 GB。sub-M 所需 18 个文件已在现有 `data_000688_subM.tar.zst` 里。

## 1. 数据：sub-C 的 loader 与检查

数据路径：`/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-C`，68 个 NWB（9.1 GB）：**CO 53 个、RT 15 个**。本扩展只用 CO（与 sub-M 一致；loader 对 RT 抛 `ValueError`）。

| 项 | sub-C | sub-M（既有） |
|---|---|---|
| electrode 表 | 2013、2015 年只有 `electrode_group_M1`（96 个，bank A/B/C × pin）；**2016-09 起另有 `electrode_group_PMd`（96 个）**，loader 只取 M1 | 同 schema |
| units | 每个 unit 恰好映射 1 个 electrode；M1 sorted units 数 38–95（2016 年的总 unit 数 246–295 含 PMd，M1 部分 55–95） | 10–52 |
| `unit_to_electrode` | 96 个 canonical M1 electrode 中的索引；**多个 unit 可以共用一个 electrode**（例如 74 units / 49 electrodes），electrode-aligned prior 按 electrode 平均后取列，这点与 sub-M 相同 | 同 |
| behavior | `processing/behavior/Velocity/cursor_vel`，2 维，按 20 ms bin 中心插值 | 同 |
| trials | `trials` 表，`result` ∈ {R, A, I, F}；eval mask = R trial 内且在 `cursor_vel` 支持区内 | 同 |
| usable trials | 152–1038（2015-03 的 5 个 session 有 605–1038 个，其余 152–319） | 173–733 |
| session 长度 | 30–190 k bins（2015-03 的 session 最长，173–187 k bins，文件 255–389 MB） | 48–319 k bins |

实现要点（sub-M 行为逐位不变）：

* `load_dandi688_recording(session)` 按前缀分派 subject；sub-M 路径的代码和数值不变。回归测试 `test_subm_loader_is_bit_identical_to_pre_extension` 对 `sub-M_ses-CO-20150612`、`sub-M_ses-CO-20140306` 把 neural / behavior / trial_change / eval_mask / trial_bounds / unit_to_electrode / observed_mask / bin_edges 的 sha256 与**扩展之前**的 loader 产物比较（相同）。
* 新函数：`dandi688_subject_root(subject)`、`list_dandi688_co_sessions(subject)`（原 `list_dandi688_subm_co_sessions` 变成其包装）、`assert_session_allowed`、`session_subject`。
* APST 源码的 import 在 Colab 上缺失时退化为本地 glob（行为相同）；不再强制要求 APST 仓库。
* 根目录可由 `SSM_DANDI688_ROOT` 重定向（`<root>/sub-C`、`<root>/sub-M`）。

## 2. 禁止使用的 sub-C session（证据）

**FORBIDDEN（6 个，APST final / test）：** `sub-C_ses-CO-20151113`、`-20151116`、`-20151117`、`-20151119`、`-20151120`、`-20151201`。

证据（均为本机文件，三处一致）：

1. `APST/src/apst/legacy_dandi/protocol.json` 与 `APST-public/src/apst/dandi/protocol.json`（两者 sha256 相同 `b593c9d1…`）：`source_sessions` 18 个（2015-03 与 2015-06/07），`development_sessions` 6 个（20151103、1104、1106、1109、1110、1112），`final_sessions` 6 个（上面的 6 个）。
2. `SPINT/sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json`（sha256 `4607e979…`）：train 27 / val 6 / test 6；`test` 与 `final_sessions` 相同；`val` 与 `development_sessions` 相同；train = 2013-10 到 2015-07 的 27 个 session。
3. `SPINT/btransform_unified_v2/dandi688_bench_v2/protocol.py`：`FINAL_SESSIONS = test`，`assert_authorized` 对 final 抛 `PermissionError`。
4. `SPINT/sua_exploration/docs/ELECTRODE_ANCHOR_DESIGNS.md` §6：6 个 held-out test session 只允许碰名字和 unit 数。
5. `SPINT/apst_final_week_20260916/dandi688_lstm_20260922/{finetune_v1,budget_v1}/sub_C/{final_receipt,budget_final_binding}.json`：同样的 6 个 final session（APST 已做过 final 评测）。

其他 sub-C session 的角色：

| 组 | session | APST 角色 | 本扩展中的处理 |
|---|---|---|---|
| 2013-10…2013-12 | 9 个 | train（strict manifest） | 允许 |
| 2015-03 | 5 个（0309、0311、0312、0313、0319） | train / source | 允许 |
| 2015-06/07 | 13 个（0629…0716） | train / source | 允许 |
| 2015-11（dev） | 1103、1104、1106、1109、1110、1112 | **development**（APST 的 selection 集，不是 held-out） | 允许，作为 month target，并标记 `apst_dev`；`--exclude-apst-dev` 可整体去掉（开放问题 1） |
| 2015-11…12（final） | 6 个 | **final / test** | **禁止** |
| 2016-09…10 | 13 个 | 不在 strict manifest（该 manifest 用 `max_units_exclusive=100` 的总 unit 数筛选，2016 年因含 PMd 总 unit 数 ≥ 100；这是推断，manifest 本身未写原因）。2026-07-24 的旧 P3 分析曾把 2016-09 之后的 8 个 session 当过探索性的 chronological "test"（该划分已被 strict 27/6/6 取代，没有 formal receipt） | 不是 APST final，但**默认不用**（optional tier，开放问题 2） |

执行层面的保护（全部有测试）：

* loader：`load_dandi688_recording` 在打开任何文件之前 `PermissionError`（测试里把 `_scan_session` 换成会报错的桩，验证未被调用）。
* inventory：显式请求 forbidden session 时 `PermissionError`；整个 subject 扫描时自动跳过（`build_expansion_inventory.py` 的 `session_inventory.json` 里只有 47 个 sub-C session，不含 6 个 final）。
* 协议：`validate_backbone` 拒绝 pool 或 target 中出现 forbidden；`bundle_files` 拒绝把 forbidden 放进 bundle；driver 在加载 target 前再检查一次。
* 同时保留 sub-M final（0623/0625/0626）的所有原有保护，且不在 inventory 中。
* `tests/test_dandi688_subc.py::test_forbidden_set_equals_apst_final_split` 从 APST 的 protocol.json 与 strict manifest 读取 final 列表，断言与代码里的 6 个相同。
* 没有读取 `results/official_evalai/`。

## 3. sub-C 协议

### 3.1 规则

* **pool = 6 个比所有 target 都早的 session**（严格按日期；pool 内部按日期排序）。一个 backbone（AP trunk + P0 calibrator）服务该 pool 之后的多个 target。
* **strata 由 target 与 pool 最晚 session 的间隔决定：** days 1–7 天，month 90–130 天（约 3–4.3 个月），year ≥ 365 天（`validate_backbone` 检查标签与间隔一致）。
* **K 的选择：K = min(10, Ledermann 上限(pool 内最小 unit 数))。** 与 sub-M 的 P1 保持 K=10（便于比较）。Ledermann 上限 `⌊(2N+1−√(8N+1))/2⌋` 只在 pool 内的 source session 上起作用（`init_from_fa` 对每个 source session 做 FA；target 侧没有 FA）。sub-C 各 pool 的最小 N 为 41–67，上限 32–55，所以 K=10 远在界内（有余量，没有为 sub-C 单独调大 K，以免引入新变量）。sub-M 新 pool 的最小 N 为 18–34（上限 12–26）。pool 里 N<15（`ledermann_limit(15)=10`）的 session 一律不入 pool（唯一受影响的是 sub-M 的 20141203，N=10，上限 6）。
* **选择只用 source：** P0 early stopping 与 R1-stream 的（readout kind, λ）选择都在 pool 的**最后一个 session**（P0 validation session）上完成；它是 pool 成员，所以永远不是本 backbone 的 target。该 session 被 trunk 训练见过，但没被 P0 的训练 episode 用过（和 P1 的 pool O 约定一致）。
* target 至少 53 个 usable trial（B=33 + 20 个 query trial）。所有 target 满足。
* 评测与 P1 一致：每个 (target, B, draw) 用 `draw_trials` 抽 B 个校准 trial，query = 其余 usable trial 的所有有效 bin，整段 recording 一次连续因果解码，物理单位 R²。

### 3.2 backbone 与 target 列表（包含 N、间隔、electrode 覆盖率）

electrode 覆盖率 = target 的 unit 中其 electrode 出现在 pool 的 electrode 并集里的比例（覆盖率低意味着 electrode-aligned prior 在这些 unit 上为零列）。

**sc_0708（days + month-dev，optional year）** pool：20150629(49)、20150630(44)、20150701(49)、20150703(52)、20150706(46)、20150707(42)；validation 20150707；最小 N=42（Ledermann 上限 33）

| target | stratum | 间隔(天) | N | usable trials | 覆盖率 | tier |
|---|---|---:|---:|---:|---:|---|
| 20150708 | days | 1 | 64 | 190 | 0.77 | primary |
| 20150709 | days | 2 | 61 | 196 | 1.00 | primary |
| 20150710 | days | 3 | 55 | 203 | 1.00 | primary |
| 20150713 | days | 6 | 58 | 185 | 0.97 | primary |
| 20150714 | days | 7 | 66 | 199 | 0.88 | primary |
| 20151103 | month | 119 | 38 | 250 | 0.66 | primary, APST-dev |
| 20151104 | month | 120 | 59 | 319 | 0.59 | primary, APST-dev |
| 20151106 | month | 122 | 60 | 258 | 0.63 | primary, APST-dev |
| 20151109 | month | 125 | 65 | 173 | 0.68 | primary, APST-dev |
| 20151110 | month | 126 | 61 | 198 | 0.66 | primary, APST-dev |
| 20151112 | month | 128 | 42 | 188 | 0.59 | primary, APST-dev |
| 20160909 | year | 430 | 55 | 154 | 0.38 | optional |
| 20160914 | year | 435 | 95 | 195 | 0.47 | optional |
| 20160919 | year | 440 | 60 | 186 | 0.67 | optional |
| 20160923 | year | 444 | 76 | 203 | 0.55 | optional |
| 20160929 | year | 450 | 74 | 209 | 0.51 | optional |
| 20161006 | year | 457 | 63 | 209 | 0.67 | optional |

**sc_0713（days）** pool：20150703(52)、20150706(46)、20150707(42)、20150708(64)、20150709(61)、20150710(55)；validation 20150710；最小 N=42

| target | stratum | 间隔 | N | usable trials | 覆盖率 | tier |
|---|---|---:|---:|---:|---:|---|
| 20150713 | days | 3 | 58 | 185 | 0.97 | primary |
| 20150714 | days | 4 | 66 | 199 | 0.98 | primary |
| 20150715 | days | 5 | 60 | 192 | 0.98 | primary |
| 20150716 | days | 6 | 61 | 191 | 0.97 | primary |

（20150713 / 20150714 同时是 sc_0708 的 target：同一 target 在两个 rolling pool 下的比较，用来检查 days 间隔内的稳定性。）

**sc_0629（month）** pool：20131220(67)、20150309(74)、20150311(88)、20150312(91)、20150313(86)、20150319(72)；validation 20150319；最小 N=67（上限 55）。2015-03 只有 5 个 session，用 20131220 补满 6 个；这个 pool 的跨度因此是 15 个月（与 sub-M 的 month pool 是 1 个月内的 6 个 session 不同，开放问题 4）。

| target | stratum | 间隔 | N | usable trials | 覆盖率 | tier |
|---|---|---:|---:|---:|---:|---|
| 20150629 | month | 102 | 49 | 179 | 0.98 | primary |
| 20150701 | month | 104 | 49 | 196 | 0.98 | primary |
| 20150706 | month | 109 | 46 | 184 | 1.00 | primary |
| 20150709 | month | 112 | 61 | 196 | 0.98 | primary |
| 20150713 | month | 116 | 58 | 185 | 0.97 | primary |
| 20150716 | month | 119 | 61 | 191 | 0.98 | primary |

**sc_0309（year）** pool：20131031(49)、20131101(41)、20131203(48)、20131204(52)、20131219(65)、20131220(67)；validation 20131220；最小 N=41（上限 32）

| target | stratum | 间隔 | N | usable trials | 覆盖率 | tier |
|---|---|---:|---:|---:|---:|---|
| 20150309 | year | 444 | 74 | 605 | 0.82 | primary |
| 20150312 | year | 447 | 91 | 905 | 0.77 | primary |
| 20150319 | year | 454 | 72 | 1026 | 0.88 | primary |
| 20150629 | year | 556 | 49 | 179 | 0.80 | primary |
| 20150706 | year | 563 | 46 | 184 | 0.85 | primary |
| 20150716 | year | 573 | 61 | 191 | 0.74 | primary |

### 3.3 为什么这样挑 pool

只有三个"簇"可用作紧凑 pool：2013-10…12（9 个）、2015-03（5 个）、2015-06/07（13 个）；之后是 2015-11 的 dev（6 个）和 final（6 个，禁止）。用 4 个 backbone 覆盖 days / month / year 三层、共 27 个 target，而不是每个 target 一个 backbone（与 drift-curve 工作里"一个 month trunk 服务 12 个 target"的做法一致）。month 层有两个独立来源：sc_0629（2015-03 pool → 2015-07，约 3.4–3.9 个月，不含任何 APST session）和 sc_0708 的 dev target（约 3.9–4.2 个月，pool 更紧凑，但这 6 个是 APST dev session）。

### 3.4 optional tier

sc_0708 的 6 个 2016 年 year target（间隔 430–457 天，electrode 覆盖率仅 0.38–0.67，是更苛刻的 year 测试）默认不跑：`--tier all` 才加入。2016 年 session 需要额外 1.54 GB 的数据（含 PMd 的文件更大，但 loader 只读 M1）。

## 4. sub-M days 层 rolling backbones（任务 2）

规则：pool = 目标日期之前**最晚的 6 个**合格 session（N ≥ 15、非 final），`ep.rolling_pool` 实现，测试证明 4 个 sub-M pool 都等于该规则的输出。

| backbone | pool（N） | 最小 N / 上限 | target（间隔） | 是否新 backbone |
|---|---|---|---|---|
| sm_0307 | 0203(34)、0217(44)、0218(38)、0303(52)、0304(39)、0306(43) | 34 / 26 | **20140307（1 天）** | **否**：复用现有 month/year trunk（`stage2_intervals_20261006/month_pool_k10_x8`）和 P1 的 month_O P0 calibrators |
| sm_0627 | 20140218(38)、0303(52)、0304(39)、0306(43)、0307(26)、20140626(26) | 26 / 19 | **20140627（1 天）** | 是 |
| sm_0512 | 20140306(43)、0307(26)、20140626(26)、20140627(25)、20140929(21)、20150511(24) | 21 / 15 | **20150512（1 天）** | 是 |
| sm_0611 | 20140626(26)、20140627(25)、20140929(21)、20150511(24)、20150512(30)、20150610(18) | 18 / 12 | **20150611（1）、20150612（2）、20150615（5）、20150616（6）、20150617（7）** | 是 |

要点：

* **20140307 是旧 month 规则的 causal selection session。** 这里它只作为 sm_0307 的 target（间隔 1 天）出现，**不参与任何 selection**：本扩展的 selection 只在每个 pool 的最后一个 session 上做（sm_0307 的是 20140306）。它在 sm_0627 / sm_0512 的 pool 里是 source（trunk 训练 session），这与旧 month 规则的 selection 无关（那是旧 rows，不重算）。测试 `test_0307_is_target_only_never_selection` 保证这一点。
* 20141203（N=10）被跳过：低于 K=10 的 Ledermann 最小 unit 数（`ledermann_k(10)=6`）。
* sm_0611 里的 0611、0612、0615 是旧 days pool 的训练 session，但这里是另一个 backbone 的 target，不冲突；0616 / 0617 与旧 days backbone 的 target 相同（不同 pool，可以直接比较）。
* final（20150623/25/26）不在任何 pool 或 target 里。
* sm_0512 的 pool 跨度约 14 个月，没有更紧凑的合格 pool 可用（开放问题 4）。

## 5. 评测管线（每个 backbone 的四个阶段）

1. **backbone**：`python -m ssm_decode.ap_pretraining --task dandi688sm --k 10 --width 128 --layers 2 --state-size 32`，起始 2000 步；若 `best_step > 0.8 × steps` 则预算翻倍重训（2000 → 4000 → 8000 → 16000，最多 4 次；与 `run_stage2_688.sh` / `run_interval_pools.sh` 相同的规则，每次从头训练），到顶仍不收敛时在 `backbone.json` 里 `converged=false`，不隐藏。pool 顺序即 `--dandi688-sources` 顺序；driver 在后续阶段检查 checkpoint 的 `source_sessions` 与协议完全一致。
2. **p0**：2 个 seed 的 P0（independent per-unit encoder，residual 输出，episodic，B 对数均匀抽样 1…33，unit dropout，leave-one-out 的 electrode-aligned prior），pool O 类比：前 5 个 session 训练，最后 1 个做 validation；起始 1000 步，best step 落在最后 20% 就翻倍（最多 4 次）。直接调用 P1 的 `run_population_calib.cmd_train`，没有改动。
3. **select**：对 pool 的 validation session，用每个 (seed, B∈{1,4,8,16,33}, draw∈4) 的模拟 B-trial 校准（P0 表 + 流式累加 readout 统计，即 R1-stream），在 5 种 readout kind（full、diag、lr8、lr16、lr32）×9 个 λ 上打分，**每个 B 选 source 分数最高的 (kind, λ)**，与 P3 的 `best_kind` 规则相同。readout 的低秩基底是 pool session 上的特征二阶矩 PCA（只用 source）。
4. **eval**：每个 target 的行 `rows/<bid>/<date>.jsonl`：`p_prior`（electrode-aligned 均值 prior，B-trial 归一化，不校准；seed 无关，只记 seed 0）、`p0`、`r1s`（P0 表 + R1-stream readout，规则来自 select）。每行记录 `h2_backward_calls`（必须为 0）、`h4_query_disjoint`、`n_units`、`stratum`、`gap_days`、`apst_dev`、`tier`、`selection` 范围。trunk 哈希在 target 循环结束后检查。

资源规则对应：trunk 冻结、target 端无反传、只用 B 个有标签 trial、selection 只用 source、连续因果解码（规则 7）、无 final。

## 6. 运行命令

环境（host）：`PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM /home/xinyuan/miniconda3/envs/spint/bin/python`。

```bash
cd /home/xinyuan/Work_host/SSM
# 协议审计和状态（CPU，秒级）
python scripts/run_expansion.py check --tier all
python scripts/run_expansion.py status
# 本机（一块 GPU；启动前看 nvidia-smi / tmux ls）
scripts/run_expansion_queue.sh --jobs 1
# 只跑 sub-M 的新 days backbone，或只跑某几个 backbone
scripts/run_expansion_queue.sh --bids sm_0627,sm_0512,sm_0611 --jobs 2
# 加入 optional 的 2016 year target，或去掉 APST dev 的 month target
scripts/run_expansion_queue.sh --tier all
scripts/run_expansion_queue.sh --exclude-apst-dev
# CPU 端到端 smoke（GRU trunk；约 13 分钟）
python scripts/smoke_expansion_cpu.py
```

Colab（`docs/colab_runbook.md` 的环境，repo 与数据放在和 host 相同的绝对路径）：

```bash
cd /home/xinyuan/Work_host/SSM
PY=python scripts/run_expansion_queue.sh \
    --out /content/drive/MyDrive/DAC27/colab/expansion_out \
    --work /content/expansion_work --jobs 4 --deadline-hours 10
```

* `--out` 是持久根目录（可指向 Drive），`--work` 是本地 scratch；每完成一个工作单元（一个 AP 尝试、一个 P0 seed、一个 (target, B) 块）就原子地复制到 `--out`。Colab 上断线后用同一条命令重跑即可恢复；rows 以 `(method, seed, B, draw)` 去重。
* `--deadline-hours` 在每个阶段开始前检查，超时则退出（返回 3），不会在 12 小时上限附近开新任务。
* 日志：`<out>/logs/<bid>.log`；汇总：`python scripts/run_expansion.py status --out <out>`。
* 结果布局：`backbones/<bid>/{ap_pool*,final,backbone.json}`、`runs/<bid>_O/p0/seed{0,1}`、`basis/<bid>.pt`、`select/<bid>{.json,_episodes.jsonl}`、`rows/<bid>/<date>.jsonl`。
* 评测结束后把 `rows/`、`select/*.json`、`backbones/*/backbone.json`、`runs/*/*/seed*/done.json` 同步到 host 做分析（`best.pt` 体积小，可一并拷）。
* 前置：新 backbone 的 trunk 用官方 Mamba-3 kernel，Colab 上先做 runbook 的 "Mamba-3 check"。Colab 上的复用文件（见 §8）必须存在。

## 7. GPU 小时与 Colab CU 估计

**没有做 GPU smoke test**（本机 GPU 被其他任务占用）。以下由同一代码的已有日志推算：

* AP 训练：`results/stage2_intervals_20261006/*/train_log.jsonl`，2000 / 4000 / 8000 / 16000 步分别 99 / 57 / 87 / 148 s（含每 100 步的 source validation；稳态约 0.009–0.014 s/步），整条扩展链 391 s ≈ 6.5 min；峰值显存 485–513 MB。
* P0 训练：P1 的 `done.json`/`train_log.jsonl`，每个 seed 81–172 s（0.07–0.15 s/步，1000–2000 步）。sub-C 的 N 约 2 倍于 sub-M，按 1.5× 估。
* 数据加载（CPU）：sub-C 单文件约 7 s（3 分钟 / 47 个文件，8 进程；2015-03 的大文件更慢）。每个阶段重新读 pool：AP 链 4 次、P0 2 次、select 1 次、每个 target 1 次。
* eval：drift-curve 的整套 rows（含 R2/R3/ui_reg，远比本队列重）平均约 2.4 s 每点；本队列每点只有 p_prior / p0 / r1s，估 0.3–0.5 s；每 target 80 个点（5 B × 8 draw × 2 seed）加数据加载，约 45–120 s。

| 阶段 | 每单位（3090） | 单位数 | 3090 小时 | L4 小时（×2 假设） |
|---|---|---:|---:|---:|
| backbone（含数据加载） | 10 min | 7 | 1.2 | 2.3 |
| P0（2 seed） | 8 min | 7 | 0.9 | 1.9 |
| select | 4 min | 8 | 0.5 | 1.1 |
| eval | 1.2 min | 35 target | 0.7 | 1.4 |
| **合计** | | | **3.3** | **6.7** |

* L4 比 3090 慢多少没有实测（L4 的 FP32 峰值接近但显存带宽约为 1/3）；取 ×2 作为保守假设，范围 ×1.5–2.5 对应 5–8 小时。
* 顺序执行：6.7 h × 1.54 = **10.3 CU**（上限）。`--jobs 4`（runbook 记录 4 个并发进程在 L4 上正常；12 vCPU）：最长链是 sc_0708（11 个 primary target，约 1.2–1.5 L4 小时），wall 约 2–3 小时，**3.1–4.6 CU**。加上 session 启动、拷贝数据（约 3.6 GB）和 Mamba-3 check 约 0.3 小时。
* **建议预算 8 CU（上限 40 CU 的 20%）。** optional tier 再加约 0.5 L4 小时（0.8 CU）。

## 8. Colab 数据 bundle（**未上传**，由 coordinator 执行 rclone）

新 bundle `data_000688_subC.tar.zst`（NWB 已压缩，zstd 收益很小，预计接近原始大小）：

* 成员列表：`results/analysis/expansion_20261016/colab_bundle_subC_primary.txt`（30 个，**3,568,531,938 B ≈ 3.57 GB**）；含 optional 的 `colab_bundle_subC_all.txt`（36 个，5,105,070,092 B ≈ 5.11 GB）。
* 打包建议（路径与 runbook 的 sub-M bundle 一致，解压到 `/home/xinyuan/Work_host/SPINT/sua_exploration/data`）：

```bash
cd /home/xinyuan/Work_host/SPINT/sua_exploration/data
tar -I zstd -cf /tmp/data_000688_subC.tar.zst \
    --transform 's,^sub-C/,dandi_000688/sub-C/,' \
    -C dandi_000688 -T /home/xinyuan/Work_host/SSM/results/analysis/expansion_20261016/colab_bundle_subC_primary.txt
```

（列表里的路径形如 `sub-C/sub-C_ses-CO-….nwb`；清单**不含** 6 个 forbidden session，打包前请再确认 `tar -tf` 输出里没有 `20151113/16/17/19/20`、`20151201`。）

| 组 | 文件数 | 大小 |
|---|---:|---:|
| 2013-10…12（sc_0309 pool：1031、1101、1203、1204、1219、1220） | 6 | 254 MiB |
| 2015-03（0309、0311、0312、0313、0319） | 5 | 1,734 MiB |
| 2015-06/07（13 个） | 13 | 925 MiB |
| 2015-11 dev（6 个） | 6 | 489 MiB |
| **primary 合计** | **30** | **3,403 MiB（3.32 GiB）** |
| optional 2016-09…10（0909、0914、0919、0923、0929、1006） | 6 | 1,466 MiB |

sub-M 需要的 18 个文件（0203、0217、0218、0303、0304、0306、0307、0626、0627、0929、20150511、0512、0610、0611、0612、0615、0616、0617）已在现有 `data_000688_subM.tar.zst` 里（该 bundle 同时含 sub-M final，别读）；名单在 `colab_bundle.json::subM_needed_in_existing_bundle`。

复用的小文件（必须随 `ckpts.tar.zst` 或另行带上，共约 1.7 MB）：`results/stage2_intervals_20261006/month_pool_k10_x8/best.pt`（1.47 MB）、`results/population_calib_20261015/runs/month_O/p0/seed{0,1}`（各约 110 KB）。代码 bundle 需要在本次提交之后重新打包（包含 `results/analysis/expansion_20261016/*.json`）。

## 9. 开放问题（请 MZ 决定）

1. **APST dev session（20151103、1104、1106、1109、1110、1112）能否作为 month target？** 它们不是 held-out，但 APST 用它们做过 checkpoint 选择；默认包含（sc_0708 的 6 个 month target，pool 最紧凑），不想用就加 `--exclude-apst-dev`（month 层仍有 sc_0629 的 6 个 target）。
2. **2016 年 session（optional year tier）要不要用？** 它们不是 APST final，也不在 strict manifest 里，但在 2026-07 的旧 P3 探索里当过 test，且 2016-09 起带 PMd 阵列（这里只读 M1）。electrode 覆盖率低（0.38–0.67）。默认关闭。
3. **R1-stream 的 selection session。** 这里统一用 pool 的 validation session（trunk 见过、P0 没用过）。P3 里 month / year 用过 OOS（trunk 没见过的）session；只有 sm_0307 的 pool 后面恰好有 20140307 可用，但它是 target，按要求不能用。如果要 OOS selection，需要另外留出一个 pool 之后、target 之前的 session（本数据多数 pool 后面没有）。
4. **混龄 pool。** sc_0629（含 20131220）和 sm_0512 / sm_0611 / sm_0627 的 pool 跨度 1–15 个月；这是"最晚 6 个合格 session"的直接后果。更紧凑的 pool 在数据里不存在（2015-03 只有 5 个 session）。
5. **K=10。** sub-C 的 Ledermann 余量大（上限 32–55），为了与 sub-M 可比保持 10；是否另做一个 K 更大的 sub-C 对照留给后续。
6. **Colab 并发与时间。** 估计按 L4 慢 2×；如果第一轮实测的 L4 速度明显不同，用 `--jobs` 和 `--deadline-hours` 调整即可，不需要改代码。

## 10. 测试与 CPU smoke

* `tests/test_dandi688_subc.py`（含 sub-M 逐位回归、sub-C 合同、6 个 forbidden 的 loader / inventory 保护、RT 与其他 subject 被拒）、`tests/test_expansion_protocol.py`（pool 构造：无 target 或 target 之后 session 进入 pool，forbidden 在 pool 与 target 中均被拒，错误尺寸/未排序/重复/缺失/subject 不符/间隔不符/低 unit/低 trial/超 Ledermann 都被拒；sub-M pool 等于 rolling 规则；20140307 不是任何 selection session；tier 与 dev 过滤；`D1_POOLS` 注册不破坏 P1 条目）、`tests/test_run_expansion.py`（Store 的 Drive/scratch 往返、backbone 选择、复用路径、队列脚本语法与 dry-run）。
* 上述加上既有的 `test_dandi688_data.py`、`test_population_calib*.py`、`test_drift_curve_guard.py`、`test_ap_backbone.py`、`test_pc3_driver.py` 全部通过。
* CPU 端到端 smoke 的结果见 §11。

## 11. CPU smoke 结果

`python scripts/smoke_expansion_cpu.py`（真实 sub-C 数据：backbone sc_0713 的 pool 20150703…0710 和 target 20150714；官方 Mamba-3 换成同接口的小 GRU trunk；ap 20→40 步、P0 20 步起、`B∈{1,4}`、1 个 draw、1 个 selection draw）：约 13 分钟，**SMOKE OK**。

* 四个阶段都跑通：AP 预算扩展规则触发（best_step 落在最后 20% 就翻倍，到 `--ap-attempts 2` 的顶后 `converged=false` 被如实记录）；P0 两个 seed 训练并按同一规则扩展预算；select 写出 `select/sc_0713.json`（每个 B 选出 kind 与 λ）和 `basis/`；eval 写出 10 行（2 个 B × 1 draw × [p_prior(seed 0)、p0×2 seed、r1s×2 seed]），所有行 `h2_backward_calls=0`、`h4_query_disjoint=True`。
* **可恢复性：** 同一进程里第二遍调用所有阶段，全部跳过（eval 对已有 `(method, seed, B, draw)` 不重算，日志里 `t=0s`），行数不变。
* 数字本身没有意义（玩具 trunk、20 步）；R² 约 p_prior 0.05、p0 0.20、r1s 0.21 只说明链路是通的。
* CPU 端各阶段的粗略耗时（单线程，仅作量级参考，**不能**换算成 GPU 时间）：sub-C 一次 6 个 session 的 pool 加载约 40–60 s；select 每个 episode 23 s（45 次整段解码）；P0 的 GRU 版约 0.65 s/步。
