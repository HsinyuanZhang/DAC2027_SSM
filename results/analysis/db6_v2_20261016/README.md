# NinaPro DB6 v2：contiguous streaming 解码、block-free backbone、重跑参照行（2026-10-16）

**状态：本报告取代 v1（`results/analysis/db6_20261015/README.md`，已在其顶部标注 superseded）。v1 的结果文件保留在 `results/db6_20261015/`、`results/analysis/db6_20261015/`，仅用于对比。**

**新 backbone checkpoint（2 个 seed，`results/db6_v2_20261016/` 为 gitignore）：**
- `/home/xinyuan/Work_host/SSM/results/db6_v2_20261016/final/seed0/best.pt`
- `/home/xinyuan/Work_host/SSM/results/db6_v2_20261016/final/seed1/best.pt`

（配置 K=8、width 128、2 层、state 32、dropout 0.1、tau 160 ms、训练窗口 512-1024 bin、trial 顺序随机重排；39 个 source session；trunk 354,072 参数，39 张表共 4,368 参数，一张表 8 x 14 fp16 = 224 B。）
全部 GPU 计算在 cuda:1（`CUDA_VISIBLE_DEVICES=1`），最多 2 个并行任务。所有选择（K/size/tau/窗口/std floor/ridge）只用 source validation 或 source 内的校准模拟，没有用 target query。

## 0. 摘要（先看这里）

1. **v1 的 trial 边界重置确实泄漏且不可流式**（review 第 1 条成立）。同一个 v1 checkpoint 改成整个 session 一次因果解码后：source val 0.874 -> 0.676（seed 1：0.872 -> 0.600）；target 的 `reuse_latest_calnorm` B=16 从 0.574 掉到 0.402，`mean_table_calnorm` 从 0.548 掉到 0.290（表 8.1）。
2. **按 review 的配方（连续流、随机偏移窗口、burn-in）重训后，出现一个 review 没有提到的新问题：DB6 每个抓取是连续 12 次重复的一个 block，连续解码的模型会学会 “上一个 trial 是什么抓取，这一个就是什么”。** 原顺序训练的 backbone（v2a）source val 0.937，但把 trial 顺序打乱后只有 0.569（K=14：0.935 -> 0.525）；原顺序训练的 v2a 在 target 上也高出 block-free 模型约 +0.07（`reuse_latest_calnorm` B=16：0.626 对 0.548）。这是一个不依赖 EMG 信号的捷径（真实使用时抓取序列不会按 12 个一组出现）。所以 **v2 主结果的 backbone 训练时每 1000 步把每个 session 的 trial 顺序重新随机排列（窗口仍是随机偏移、不对齐 trial 边界、跨 trial 携带状态），早停用 “打乱顺序后的 source val”**；评测仍按 review 要求在**记录顺序**下整段因果解码，同时做了打乱顺序的评测（差别 <= 0.01，见 9.3）。
3. **v2 主结果（contiguous、block-free；2 seeds；balanced accuracy，n=10 受试者，all gaps）**：source val（打乱顺序/原顺序）0.859/0.845；零 target 数据复用 0.407；B=16 时 `reuse_latest_calnorm` 0.548、`rownorm_ridge` 0.578、`ui_prior_latest_calnorm` 0.724；B=42 的 `ui_prior` 0.787。和 v1 相比（重置解码）各行低 0.00-0.04（见 8），**排序没变**：B>=8 时 ui_prior（梯度参照）明显高于 reuse 与 closed-form；B<=2 时 population/mean 先验最强。
4. **ridge 的 EMA 时间常数必须 block-free 地选择。** 在原顺序 source 流上选出的 EMA 0.998（时间常数约 10 s，约一个 trial）在 target 上 B=16 达到 0.683，比 block-free 选择的 0.578 高 0.106；但把 target trial 顺序打乱后它掉到 0.366（block-free 的是 0.579）。高出的部分是 block 结构，不是信号。主 ridge 行用 block-free 选择；`rownorm_ridge_origsel` 作为反例保留。
5. **漂移统计**：0 起第 15 通道不是 “噪声底通道”，它对 S1-S4 在早期是活的（raw std 1.2-56 uV）后来才死（~0.1 uV）；S5-S10 全程是死的。14 通道版本的 log 增益比 std 随 gap 的上升（0.54 -> 0.90）主要由它造成；去掉它后是 0.45 -> 0.64，12 个始终好的通道是 0.41 -> 0.54（见 10）。
6. 此前 v1 中几个说法被收回或降级：“table-only 上限 0.80” 改为 “42 个 trial 与该优化器下观察到的最好 table 拟合”；“own-subject 先验不比群体均值好” 在 n=10 下功效不足，且 `mean_table` 比去掉自己 4 张表的 `mean_loso` 高 0.015-0.022（显著），说明包含自己的表有一点帮助；gap 3 上 mean > latest 不显著；“同日漂移与跨天同量级” 依赖通道集合和 S9（见 10、11）。

## 1. 对独立 review 的逐条处理

| # | review 条目 | 处理 | 状态 |
|---|---|---|---|
| 1 | CRITICAL：在 trial 边界重置 SSM 状态 / ridge EMA | 全部改为整个 session 一次因果解码（状态跨 trial 携带），只在 query trial 的 bin 上打分，校准 trial 的 bin 不计分；backbone 在连续流的随机偏移窗口（512-1024 bin，前 128 bin burn-in 不计 loss，不对齐 trial 边界）上重训；source 选择用留出重复的整段连续解码；ridge 的 EMA 对整个 session 流；ui_prior 与 within 只在 B 个校准 trial 上拟合，相邻校准 trial 合成连续片段（从零状态开始），片段之外的神经数据一律不用（H4）。K/size/tau 在 source 上重选（小搜索，6 + 2 次运行），2 个 seed。**额外发现并处理 block 结构捷径**（见 0.2、5）。 | 完成 |
| 2 | HIGH：“table-only 上限 0.80” | within-session 行与 ui_prior 是同一流程（只差步数 1000 对 300），B=42 时配对差 -0.000 +- 0.006。改写为 “42 个 trial 与该优化器下观察到的最好 table 拟合（0.787）”，不称上限（见 7.4）。 | 完成 |
| 3 | MEDIUM：通道事实错误；B-trial 归一化对近零 std 不稳；漂移统计 | README 通道事实已更正（第 2 节）；B-trial std 加下限 `max(std, floor x median_c std)`，floor 在 source 上选（0.01，见 5.3）；漂移统计在 14 通道、13 通道、12 个始终好的通道上都重算并报告（第 10 节）。 | 完成 |
| 4 | MEDIUM-LOW：ridge 网格边界 | 网格 lam 1e-6..1e4，ema 0..0.9995（含 > 0.995）；block-free 选择的最优点在内部（除 B=42 的 lam，见 5.4）。 | 完成 |
| 5 | LOW：披露早期 probe 评测 | 见第 11 节。 | 完成 |
| 6 | LOW：软化结论 | 见 7、10。 | 完成 |

## 2. 数据事实（更正后）

数据来源、文件结构、标签与 trial 定义同 v1（10 名受试者 x 5 天 x 2 session，14 个电极通道，8 类：休息 + 7 个抓取；100 个 session，其中 `S2_D2_T2` 是截断文件，已从 source 剔除；标签用 `restimulus/rerepetition`；一个 trial = 一段 `rerepetition` 恒定的 `[之前的休息 | 动作]`）。更正如下。

- 16 列中 0 起第 8、9 列在所有文件里恒为 0，剩余 14 列是 Delsys 电极。
- **0 起第 15 通道**（`drift_stats.json` 的 `channel_facts.per_file` 有全部 100 个文件的值）：S1-S4 早期是活的（raw std 5.35-56 uV）然后死（~0.1 uV）：**S1 从 D5 起、S2 从 D4 起、S3 从 D3T2 起、S4 从 D3 起**；S5-S10 几乎全程是死的（只有 `S5_D1_T2` 是活的，1.15 uV；阈值取 raw std < 1 uV）。因此它在 60 个 target session 中有 53 个是死的，在 39 个 source session 中有 23 个是死的（全是 S5-S10 的 source）。
- **0 起第 7 通道**：raw std 大部分 0.2-0.6 uV（接近噪声底），但有 8 个文件 > 1 uV（`S3_D1_T1, S6_D1_T1, S8_D2_T2, S8_D5_T2, S9_D3_T1, S9_D4_T2, S9_D5_T1, S9_D5_T2`），所以它是 “噪声底附近、偶尔有信号” 的通道，不是恒定的噪声底。v1 “第 8、16 列在所有 session 里 std < 0.9 uV” 是错的。
- 两个通道都保留在输入里（14 通道）。S5-S10 的死通道 15 在训练里就是被 z-score 放大到单位方差的噪声，表可以学会忽略它；S1-S4 的 target 里它从活变死，是 B-trial 归一化需要 std floor 的原因（5.3）。

## 3. 因果流式前端与解码协议（v2）

前端同 v1（整流 -> 一阶泄漏积分器，tau 160 ms，整个 session 流上状态连续 -> 50 Hz 取 bin 末端样本）。**解码协议变化：**

- **一个 session = 一条连续流**。模型、ridge EMA 都对整条流（约 35.7k bin）运行一次，状态跨 trial 携带。trial 边界只用来决定哪些 bin 参与训练 / 打分，从不重置状态。
- source validation：每个抓取的第 4、8、12 次重复留出，其标签从不用于训练（loss mask）；其输入作为无标签上下文出现在连续窗口里。对整条 session 流一次解码，只在留出 trial 的 bin 上打分。
- target：query = 其余 84-B 个 trial 的 bin；校准 trial 的 bin 参与解码（作为状态上下文）但不计分；最后一个动作之后不属于任何 trial 的约 1 s 休息不计分。
- B-trial 的统计量（逐通道均值、std，std 下限见 5.3）只来自 B 个校准 trial 的 bin，对整条流做常数 z-score（因果）。注意：校准 trial 在记录中可能晚于某些 query trial，和 v1、Hyser 一样是 “开机校准块” 的评测等价化，不是严格在线。
- 拟合（ui_prior、within、ridge）只用校准 trial。相邻的校准 trial 合成连续片段，每个片段从零状态开始解码 / 做 EMA，片段外的神经数据（包括紧邻的未校准 trial）一律不用。
- 校准抽样、预算 B in {1,2,4,8,16}（诊断 42）、聚合方式同 v1（抽样平衡，覆盖见 `calibration_coverage.csv`：B=1/2/4/>=8 分别见到 2/3/5/8 类）。

测试（`tests/test_db6_backbone.py` 13 项在 CPU 上通过，另有 1 项 Mamba GPU 测试用 `DB6_TEST_CUDA=cuda:0` 开启，在 cuda:1 上通过；`tests/test_db6_data.py` 9 项）：包括 “第 k 个 trial 首个 bin 的输出依赖第 k-1 个 trial 的输入”（有状态玩具 trunk 与真实 Mamba trunk 各一项；若实现在 trial 边界重置状态则失败），“整条流解码与逐 trial 解码在第一个 trial 之后不同”，“query 打分不含校准 bin 且校准 bin 的输出被篡改后分数不变”，“校准片段只合并相邻 trial”，“ridge 的 EMA 对整条流因果且片段内不看片段外”，“std 下限”，“reorder_stream 保持 trial 内容”。

## 4. Backbone 与训练

结构同 v1（`APBackbone`：每个 source session 一张 K x 14 read-in table + 共享 Mamba-3 trunk + 线性 head；损失 = trunk CE + 0.5 x head CE + 0.1 x whitening；AdamW lr 1e-3 wd 1e-2，batch 16，`drift_augment`）。训练改为：

- 每步取 16 个窗口，窗口长度每步在 512-1024 bin 内随机，起点在 session 流内均匀随机（不对齐 trial 边界）；窗口前 128 bin 只提供状态上下文，不计 loss；5% 的窗口从流起点开始且无 burn-in（对齐推理时的零状态启动）。
- loss 只在训练 trial 的 bin 上（留出重复和不属于任何 trial 的 bin 为 IGNORE）。
- **trial 顺序重排**：每 1000 步把每个 session 的 trial 顺序重新随机排列后拼成新的流，窗口从这些流里切（排列只改变 trial 拼接顺序，不改变 trial 内部的连续性）。拼接点处包络有一次跳变（原数据里是按 tau 衰减），这是该做法的人工痕迹。
- 早停：每 250 步在 “打乱顺序的 source val”（固定排列，整段连续解码）上评分，patience 8 次（2000 步），上限 15000 步。同时记录原顺序 source val。

## 5. 选择（全部只用 source；`backbone_selection.csv`）

### 5.1 先验证 block 捷径（阶段 0，原顺序训练，seed 0）

| 运行 | 训练顺序 | source val 原顺序 | source val 打乱顺序（`order_shuffle_diag.json`） |
|---|---|---|---|
| K8 w128 tau160 | 原顺序 | 0.937 | 0.569 |
| K14 w128 tau160 | 原顺序 | 0.935 | 0.525 |
| v1 checkpoint seed 0（trial 重置训练） | - | 0.676（v1 重置解码时 0.874） | 0.261 |
| v1 checkpoint seed 1 | - | 0.600（v1 重置解码时 0.872） | 0.222 |
| **v2 final seed 0（打乱训练）** | 打乱 | 0.846 | 0.857 |
| **v2 final seed 1（打乱训练）** | 打乱 | 0.842 | 0.856 |

原顺序训练得到最高的 source val（0.937）但依赖 block 顺序；它不是我们要的 backbone，保留为 v2a 对照。

### 5.2 架构、tau、窗口（阶段 1，打乱训练，seed 0，早停于打乱顺序 source val）

| 运行 | 打乱顺序 val | 原顺序 val | best step / 预算 |
|---|---|---|---|
| **K8 w128 l2 s32 tau160（v1 赢家，选用）** | 0.8605 | 0.8464 | 10250 / 15000 |
| K8 w128 tau160，窗口 1024-2048 | 0.8647 | 0.8241 | 10000 |
| K8 w128 tau320 | 0.8624 | 0.8409 | 11500 |
| K8 w256 s64 tau160 | 0.8567 | 0.8338 | 13750（>80% 预算） |
| K14 w128 tau160 | 0.8480 | 0.8275 | 10000 |
| K8 w128 tau80 | 0.8475 | 0.8360 | 8500 |

K8 w128 tau160、tau320、窗口 1024-2048 的差距 <= 0.004，小于 seed 间波动（v2 final 两个 seed 差 0.003，v1 中见到最多 0.02），不能区分；按 “参数最少、沿用 v1 赢家、窗口取 review 给的默认 512-1024” 选 K8 w128 tau160。K=14 与 tau80 低 0.012-0.013（与 seed 间波动同量级，不显著）。搜索是小搜索（6 个配置，seed 0），没有穷举。

最终 2 个 seed：seed 0 即该阶段 1 运行；seed 1 重新训练：打乱 val 0.8572（best step 14250，**超过预算的 80%，仍在缓慢上升**，所以 seed 1 略欠训练，`budget_rule_ok = false`）。

### 5.3 B-trial std floor（`std_floor_selection.json`）

B-trial 的逐通道 std 取 `max(std, floor x median_c std)`。选择只用 source：同一个 backbone（训练时 floor = 1e-3），换 test floor；打分用 source session 内的 B=8 校准模拟（natural：39 个 session；stress：S1-S4 的 source session，把通道 15 整条流换成真实死通道的包络（S5_D1_T1 的通道 15，约 0.02 uV；只缩小幅度会被 z-score 抵消，不是死通道））。规则（事先写在脚本里）：natural 不低于最小 floor 的 natural - 0.005 的候选中取 stress 最高。

| test floor | natural（39 session） | stress（S1-S4，死通道） |
|---|---|---|
| 0.001 | 0.8295 | 0.7790 |
| **0.01（选用）** | 0.8258 | 0.7816 |
| 0.02 | 0.8203 | 0.7928 |
| 0.05 | 0.8063 | 0.8078 |
| 0.1 | 0.7968 | 0.8072 |
| 0.2 | 0.7906 | 0.8031 |
| 参照：训练时也用 floor 0.05 重训 | 0.8206 | 0.8019 |
| 参照：训练时也用 floor 0.2 重训 | 0.8122 | 0.7932 |

死通道使 stress 掉 0.05（0.829 -> 0.779）；更大的 floor 能回收一部分（0.05 时到 0.808）但 natural 掉 0.02（S5-S10 的 source 里死通道本来就是被学会忽略的单位方差噪声，floor 改变了它的尺度），规则下选 0.01。**这个 floor 的实际防护很弱**（stress 只 +0.003）。事后敏感性（不用于选择）：用 floor 0.05 重训的 checkpoint 在 target 上，`reuse_latest_calnorm` B=16 seed 0 为 0.555（主 checkpoint 0.551），差别 +0.004，说明在这个协议下 floor 对平均结果几乎没有影响（`extra_floor0.05_ckpt/`）。

### 5.4 ridge（lam, ema）（`ridge_*_floor0.01_tau160.json`）

在每个 source session 里模拟 B-trial 校准（3 次 draw、训练 trial），在留出重复上整条流因果解码并打分；网格 lam 1e-6..1e4（11 档）、ema 0..0.9995（10 档）。

| B | block-free 选择（打乱顺序的 source 流；主行）lam / ema / source 模拟分数 | 原顺序选择（`origsel`）lam / ema / 分数 |
|---|---|---|
| 1, 2 | 100 / 0 / 0.196, 0.267 | 100 / 0 / 0.191, 0.265 |
| 4 | 100 / 0.95 / 0.379 | 10 / 0.998 / 0.402 |
| 8 | 10 / 0.98 / 0.491 | 10 / 0.998 / 0.652 |
| 16 | 0.1 / 0.98 / 0.535 | 10 / 0.998 / 0.734 |
| 42 | 1e-6 / 0.98 / 0.587 | 1 / 0.998 / 0.808 |

最优点都在网格内部，唯一例外是 block-free 的 B=42 的 lam 落在下边缘 1e-6，但该处目标面是平的（lam 1e-6 到 0.01 都是 0.5874，lam=1 是 0.5866），所以不构成问题。原顺序选择里 source 模拟分数高得多（B=16：0.734 对 0.535），差值就是 block 结构。

## 6. 参照行定义

同 v1：`reuse_latest_srcnorm`（最新 source 表 + source 统计量，零 target 数据）、`reuse_latest_calnorm`（同表 + B-trial 统计量）、`mean_table_calnorm`、`mean_loso_calnorm`、部署型 `rownorm_ridge`（14 -> 8 的闭式 ridge + 整条流因果 EMA；`rownorm_ridge_origsel` 为原顺序选择的反例）、`ui_prior_*`（梯度穿过 frozen trunk，只更新 read-in table；300 步、lr 3e-3，固定值，沿用 v1，没有重新调）、B=42 诊断的 `within_table_*`（1000 步）。ui_prior 与 within 是**非部署参照**，H2：reuse/mean/ridge 0 次反传。H1（trunk 哈希不变）在全部 28,980 行成立，H5（表 224 B <= 32 KB）成立；H3（乘加比例）没有计算。2 个 seed 的 ridge 行完全相同（ridge 不依赖 backbone），所以 “2 seeds” 对 ridge 行没有意义。

## 7. 结果（v2，contiguous 解码）

source validation：打乱顺序 0.859、原顺序 0.845（2 seed 均值）；chance = 0.125。以下为 balanced accuracy，均值 +- 95% CI（t 分布，n=10 受试者），2 seeds 平均，B=42 只有 seed 0、每个 target 3 次 draw。

#### T1 balanced accuracy, all gaps

| method | B=1 | B=2 | B=4 | B=8 | B=16 | B=42（诊断） |
|---|---|---|---|---|---|---|
| `reuse_latest_srcnorm`（零 target 数据） | 0.407 ± 0.091 | 0.407 ± 0.091 | 0.407 ± 0.092 | 0.407 ± 0.092 | 0.407 ± 0.092 | 0.404 ± 0.093 |
| `reuse_latest_calnorm` | 0.396 ± 0.062 | 0.457 ± 0.076 | 0.508 ± 0.086 | 0.541 ± 0.092 | 0.548 ± 0.093 | 0.556 ± 0.097 |
| `mean_table_calnorm` | 0.428 ± 0.059 | 0.477 ± 0.056 | 0.511 ± 0.051 | 0.533 ± 0.053 | 0.532 ± 0.053 | 0.516 ± 0.056 |
| `mean_loso_calnorm` | 0.413 ± 0.058 | 0.459 ± 0.055 | 0.490 ± 0.050 | 0.511 ± 0.051 | 0.510 ± 0.052 | 0.495 ± 0.056 |
| `rownorm_ridge`（部署型 closed-form，block-free 选择） | 0.201 ± 0.014 | 0.277 ± 0.020 | 0.381 ± 0.029 | 0.514 ± 0.049 | 0.578 ± 0.047 | 0.630 ± 0.055 |
| `rownorm_ridge_origsel`（原顺序 source 上选择，可吃 block 结构） | 0.201 ± 0.014 | 0.277 ± 0.020 | 0.382 ± 0.023 | 0.598 ± 0.052 | 0.683 ± 0.041 | 0.754 ± 0.031 |
| `ui_prior_latest_srcnorm`（参照） | 0.361 ± 0.057 | 0.414 ± 0.062 | 0.509 ± 0.073 | 0.619 ± 0.086 | 0.684 ± 0.085 | 0.745 ± 0.081 |
| `ui_prior_latest_calnorm`（参照） | 0.355 ± 0.047 | 0.445 ± 0.065 | 0.553 ± 0.067 | 0.666 ± 0.073 | 0.724 ± 0.073 | 0.787 ± 0.069 |
| `ui_prior_mean_calnorm`（参照） | 0.346 ± 0.037 | 0.437 ± 0.045 | 0.549 ± 0.054 | 0.673 ± 0.068 | 0.730 ± 0.068 | 0.790 ± 0.064 |
| `within_table_latestinit_long`（诊断） | - | - | - | - | - | 0.787 ± 0.068 |
| `within_table_supinit`（诊断） | - | - | - | - | - | 0.782 ± 0.062 |

#### T2 balanced accuracy per gap (B=4 / B=16)

| method | gap 1 (day 3) | gap 2 (day 4) | gap 3 (day 5) |
|---|---|---|---|
| `reuse_latest_srcnorm`（零 target 数据） | 0.412 / 0.412 | 0.442 / 0.442 | 0.367 / 0.366 |
| `reuse_latest_calnorm` | 0.537 / 0.582 | 0.525 / 0.568 | 0.463 / 0.494 |
| `mean_table_calnorm` | 0.516 / 0.537 | 0.511 / 0.536 | 0.504 / 0.523 |
| `mean_loso_calnorm` | 0.494 / 0.515 | 0.491 / 0.514 | 0.484 / 0.501 |
| `rownorm_ridge`（部署型 closed-form，block-free 选择） | 0.382 / 0.568 | 0.386 / 0.583 | 0.375 / 0.582 |
| `rownorm_ridge_origsel`（原顺序 source 上选择，可吃 block 结构） | 0.387 / 0.682 | 0.395 / 0.693 | 0.365 / 0.675 |
| `ui_prior_latest_srcnorm`（参照） | 0.512 / 0.685 | 0.541 / 0.706 | 0.475 / 0.659 |
| `ui_prior_latest_calnorm`（参照） | 0.569 / 0.729 | 0.571 / 0.737 | 0.517 / 0.706 |
| `ui_prior_mean_calnorm`（参照） | 0.555 / 0.729 | 0.560 / 0.740 | 0.532 / 0.723 |

#### T3 macro-F1 and vote accuracy (all gaps; B = 1 / 4 / 16)

| method | macro-F1 B=1 / 4 / 16 | vote acc B=1 / 4 / 16 |
|---|---|---|
| `reuse_latest_srcnorm`（零 target 数据） | 0.366 / 0.365 / 0.365 | 0.410 / 0.410 / 0.411 |
| `reuse_latest_calnorm` | 0.365 / 0.496 / 0.540 | 0.408 / 0.556 / 0.607 |
| `mean_table_calnorm` | 0.399 / 0.491 / 0.513 | 0.445 / 0.559 / 0.587 |
| `mean_loso_calnorm` | 0.384 / 0.470 / 0.491 | 0.425 / 0.531 / 0.558 |
| `rownorm_ridge`（部署型 closed-form，block-free 选择） | 0.109 / 0.307 / 0.578 | 0.132 / 0.419 / 0.760 |
| `rownorm_ridge_origsel`（原顺序 source 上选择，可吃 block 结构） | 0.109 / 0.263 / 0.583 | 0.132 / 0.423 / 0.797 |
| `ui_prior_latest_srcnorm`（参照） | 0.311 / 0.470 / 0.675 | 0.349 / 0.540 / 0.778 |
| `ui_prior_latest_calnorm`（参照） | 0.323 / 0.532 / 0.719 | 0.366 / 0.606 / 0.822 |
| `ui_prior_mean_calnorm`（参照） | 0.311 / 0.522 / 0.722 | 0.343 / 0.593 / 0.825 |

#### T4 paired differences (a - b), balanced accuracy, all gaps, 95% CI over 10 subjects

| a - b | B=1 | B=2 | B=4 | B=8 | B=16 | B=42 |
|---|---|---|---|---|---|---|
| `reuse_latest_calnorm` - `reuse_latest_srcnorm` | -0.011 ± 0.037 | +0.051 ± 0.039 | +0.102 ± 0.043 | +0.133 ± 0.049 | +0.142 ± 0.051 | +0.152 ± 0.054 |
| `ui_prior_latest_calnorm` - `reuse_latest_calnorm` | -0.041 ± 0.024 | -0.012 ± 0.022 | +0.044 ± 0.037 | +0.125 ± 0.050 | +0.176 ± 0.054 | +0.231 ± 0.059 |
| `ui_prior_latest_calnorm` - `rownorm_ridge` | +0.155 ± 0.035 | +0.168 ± 0.050 | +0.172 ± 0.045 | +0.151 ± 0.038 | +0.147 ± 0.032 | +0.157 ± 0.021 |
| `rownorm_ridge` - `reuse_latest_calnorm` | -0.195 ± 0.052 | -0.181 ± 0.064 | -0.128 ± 0.072 | -0.026 ± 0.077 | +0.030 ± 0.073 | +0.074 ± 0.069 |
| `rownorm_ridge_origsel` - `rownorm_ridge` | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.001 ± 0.015 | +0.083 ± 0.015 | +0.106 ± 0.015 | +0.124 ± 0.027 |
| `reuse_latest_calnorm` - `mean_table_calnorm` | -0.032 ± 0.052 | -0.020 ± 0.060 | -0.002 ± 0.072 | +0.008 ± 0.076 | +0.016 ± 0.077 | +0.040 ± 0.082 |
| `mean_table_calnorm` - `mean_loso_calnorm` | +0.015 ± 0.006 | +0.018 ± 0.009 | +0.021 ± 0.009 | +0.022 ± 0.011 | +0.022 ± 0.010 | +0.021 ± 0.011 |
| `ui_prior_latest_calnorm` - `ui_prior_mean_calnorm` | +0.009 ± 0.033 | +0.008 ± 0.036 | +0.004 ± 0.032 | -0.008 ± 0.021 | -0.006 ± 0.015 | -0.004 ± 0.015 |
| `within_table_latestinit_long` - `ui_prior_latest_calnorm` | - | - | - | - | - | -0.000 ± 0.006 |
| `within_table_supinit` - `ui_prior_latest_calnorm` | - | - | - | - | - | -0.005 ± 0.013 |

#### T5 own-subject latest vs population mean table, per gap (a = `mean_table_calnorm`, b = `reuse_latest_calnorm`)

| gap | B=4 | B=8 | B=16 |
|---|---|---|---|
| 1 | -0.021 ± 0.097 | -0.038 ± 0.099 | -0.045 ± 0.099 |
| 2 | -0.014 ± 0.090 | -0.029 ± 0.094 | -0.032 ± 0.095 |
| 3 | +0.041 ± 0.088 | +0.043 ± 0.097 | +0.028 ± 0.103 |

### 7.1 事实

- 零 target 数据复用 0.407；用 B-trial 无标签统计量重新 z-score：B=4 +0.102 +- 0.043，B=16 +0.142 +- 0.051；统计量估计在 B>=8 基本饱和（B=8、16、42 为 0.541、0.548、0.556）。B=1 时 calnorm 不比 srcnorm 好（-0.011 +- 0.037）。
- 有标签的 table 更新（ui_prior，梯度参照）相对 reuse：B=1 -0.041 +- 0.024（显著更差）、B=2 -0.012 +- 0.022、B=4 +0.044 +- 0.037（边缘）、B=8 +0.125、B=16 +0.176、B=42 +0.231。B<=2 时最强的是零标签先验（`mean_table_calnorm` B=1 0.428、B=2 0.477）。
- 部署型 closed-form（block-free）：B=8 与 reuse 无显著差（-0.026 +- 0.077）、B=16 +0.030 +- 0.073（不显著）、B=42 +0.074 +- 0.069（刚好显著）；B<=4 受制于没见过的类（B=1 最多预测 2 类，balanced accuracy 上界 0.25；B=2、4 上界 0.375、0.625），所以很差。ui_prior 比它高 0.147-0.172（B=1 到 42 全部显著）。
- 原顺序选择的 ridge（`origsel`）比 block-free 的高 +0.083/+0.106/+0.124（B=8/16/42），但在打乱顺序的评测里它掉到 0.332/0.366（B=8/16），主 ridge 不变（0.515/0.579）；因此这部分增益不是信号。
- 受试者差异大：B=16 的 `ui_prior_latest_calnorm` 从 0.549（S9）到 0.844（S5），`reuse_latest_calnorm` 从 0.358（S8）到 0.763（S1）。没有排除任何受试者。两个 seed 的 target 均值互差：ui_prior 行 <= 0.006，reuse 行 <= 0.011，`mean_table_calnorm`/`mean_loso_calnorm` 最大 0.032（`mean_table_calnorm` B=16：0.516 对 0.548）；seed 1 略欠训练（5.2）。

### 7.2 own-subject 先验 vs 群体均值（降级后的表述）

`reuse_latest_calnorm - mean_table_calnorm` 在 B>=4 为 -0.002 到 +0.040，CI 约 +-0.07-0.08，全部不显著；按 gap 分：gap 1/2 上 latest 稍好（B=16：mean - latest = -0.045、-0.032，CI +-0.10），gap 3 上 mean 稍好（+0.028 +- 0.103），**都不显著**，所以 v1 中 “gap 3 上 mean 表高于自己最新的表” 不能作为结论。n=10 且 CI 约 +-0.08，这个比较**功效不足**，不能说 own-subject 先验 “不比群体均值好”，只能说差别小于约 0.08，方向无法确定。`mean_table_calnorm` 比去掉自己 4 张表的 `mean_loso_calnorm` 高 0.015-0.022（显著），说明包含自己的表有小幅帮助。`ui_prior` 从 latest 表还是 mean 表出发无差别（|差| <= 0.009，CI 含 0）。

### 7.3 小 B 的瓶颈

B<=4 时 8 类中只见到 2/3/5 类；只用 B 个 trial 的方法对没见过的类没有信息，所以 B<=2 时先验行最强，B>=8（8 类覆盖）后才出现 ui_prior 的优势。这来自协议（B 为总 trial 数），不是实现问题。

### 7.4 “table 拟合上限” 的表述

B=42：`ui_prior_latest_calnorm`（300 步）0.787，`within_table_latestinit_long`（1000 步）0.787（配对差 -0.000 +- 0.006），`within_table_supinit` 0.782（-0.005 +- 0.013）。三者用同一流程（相同的校准数据、优化器和 loss），只差初始化与步数，所以它们的一致性只说明**该优化器在 42 个 trial 上收敛到的位置**，不是独立的上限证据；0.787 是 “42 个 trial 与此优化器下观察到的最好 table 拟合”。它低于 source val（0.859）约 0.07，原因可能是 table 表达能力不足、需要更新 trunk、42 个 trial 数据量或优化设定，本报告没有区分。

## 8. v1（trial 重置）对 v2（contiguous）：连续解码的影响

### 8.1 source validation

| | v1 重置解码 | 同一 v1 checkpoint 连续解码 | v2 final（打乱训练，连续解码） |
|---|---|---|---|
| seed 0 | 0.874 | 0.676 | 打乱 0.8605 / 原顺序 0.8464 |
| seed 1 | 0.872 | 0.600 | 打乱 0.8572 / 原顺序 0.8425 |

### 8.2 target（balanced accuracy，all gaps，n=10 受试者）

列：v1（重置解码，v1 报告的数字，B=42 为 seed 0）；v1 ckpt contiguous（同一 v1 checkpoint 只改成连续解码，2 seeds，`extra_v1ckpt_contiguous/`）；v2a（原顺序训练的连续 backbone，仅 seed 0，会利用 block 结构）；v2（本报告主结果）。v1 的 closed-form 行对应本报告的 `rownorm_ridge`（v1 报告里叫 `rownorm_ridge_v2`；两者的 ridge 选择协议不同：v1 在 trial 内 EMA 且重置）。v1 ckpt contiguous 与 v2a 只跑了不需要反传的 reuse/mean 行。

**B = 1**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.310 ± 0.056 | 0.438 ± 0.093 | 0.407 ± 0.091 |
| `reuse_latest_calnorm` | 0.403 ± 0.043 | 0.308 ± 0.037 | 0.419 ± 0.063 | 0.396 ± 0.062 |
| `mean_table_calnorm` | 0.440 ± 0.056 | 0.291 ± 0.035 | 0.446 ± 0.066 | 0.428 ± 0.059 |
| `rownorm_ridge` | 0.199 ± 0.015 | - | - | 0.201 ± 0.014 |
| `ui_prior_latest_calnorm` | 0.374 ± 0.032 | - | - | 0.355 ± 0.047 |

**B = 2**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.310 ± 0.056 | 0.438 ± 0.093 | 0.407 ± 0.091 |
| `reuse_latest_calnorm` | 0.475 ± 0.058 | 0.353 ± 0.056 | 0.504 ± 0.089 | 0.457 ± 0.076 |
| `mean_table_calnorm` | 0.494 ± 0.051 | 0.291 ± 0.032 | 0.519 ± 0.069 | 0.477 ± 0.056 |
| `rownorm_ridge` | 0.275 ± 0.020 | - | - | 0.277 ± 0.020 |
| `ui_prior_latest_calnorm` | 0.470 ± 0.047 | - | - | 0.445 ± 0.065 |

**B = 4**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.311 ± 0.056 | 0.438 ± 0.093 | 0.407 ± 0.092 |
| `reuse_latest_calnorm` | 0.533 ± 0.069 | 0.388 ± 0.059 | 0.575 ± 0.103 | 0.508 ± 0.086 |
| `mean_table_calnorm` | 0.527 ± 0.049 | 0.286 ± 0.025 | 0.569 ± 0.065 | 0.511 ± 0.051 |
| `rownorm_ridge` | 0.395 ± 0.027 | - | - | 0.381 ± 0.029 |
| `ui_prior_latest_calnorm` | 0.559 ± 0.050 | - | - | 0.553 ± 0.067 |

**B = 8**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.310 ± 0.056 | 0.438 ± 0.093 | 0.407 ± 0.092 |
| `reuse_latest_calnorm` | 0.565 ± 0.074 | 0.400 ± 0.065 | 0.613 ± 0.112 | 0.541 ± 0.092 |
| `mean_table_calnorm` | 0.550 ± 0.053 | 0.294 ± 0.022 | 0.593 ± 0.071 | 0.533 ± 0.053 |
| `rownorm_ridge` | 0.537 ± 0.045 | - | - | 0.514 ± 0.049 |
| `ui_prior_latest_calnorm` | 0.663 ± 0.056 | - | - | 0.666 ± 0.073 |

**B = 16**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.310 ± 0.056 | 0.437 ± 0.093 | 0.407 ± 0.092 |
| `reuse_latest_calnorm` | 0.574 ± 0.075 | 0.402 ± 0.067 | 0.626 ± 0.114 | 0.548 ± 0.093 |
| `mean_table_calnorm` | 0.548 ± 0.053 | 0.290 ± 0.023 | 0.596 ± 0.070 | 0.532 ± 0.053 |
| `rownorm_ridge` | 0.608 ± 0.043 | - | - | 0.578 ± 0.047 |
| `ui_prior_latest_calnorm` | 0.726 ± 0.057 | - | - | 0.724 ± 0.073 |

**B = 42**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.434 ± 0.075 | - | - | 0.404 ± 0.093 |
| `reuse_latest_calnorm` | 0.583 ± 0.081 | - | - | 0.556 ± 0.097 |
| `mean_table_calnorm` | 0.557 ± 0.057 | - | - | 0.516 ± 0.056 |
| `rownorm_ridge` | 0.672 ± 0.043 | - | - | 0.630 ± 0.055 |
| `ui_prior_latest_calnorm` | 0.802 ± 0.053 | - | - | 0.787 ± 0.069 |
| `within_table_latestinit_long` | 0.803 ± 0.051 | - | - | 0.787 ± 0.068 |

要点：（i）连续解码对 v1 checkpoint 是破坏性的：reuse_latest_calnorm B=16 -0.172，mean_table_calnorm -0.258（与 review 给出的 -0.15 到 -0.20 一致）；（ii）重训后的 v2 与 v1 的重置数字接近但略低（reuse_latest_calnorm B=16：0.548 对 0.574；ui_prior_latest_calnorm B=16 0.724 对 0.726；ridge 0.578 对 0.608），排序与结论不变；（iii）v2a 比 v2 高 +0.06 到 +0.08（B=4-16 的 reuse/mean 行；srcnorm +0.03），差值来自 block 顺序，不应被当作真实性能。

## 9. 额外检查

### 9.1 block 结构检查的细节
`scripts/db6_order_shuffle_diag.py`：对每个 source session 在记录顺序和 3 种随机 trial 顺序下整段解码，打分留出重复（见 5.1 的表）。

### 9.2 std floor 的 target 敏感性
见 5.3（`extra_floor0.05_ckpt/`，只 seed 0、reuse 行）。

### 9.3 target 评测用打乱的 trial 顺序（`extra_shuffled_order_eval/`，seed 0，reuse + ridge 行）
`reuse_latest_srcnorm` 0.404（原顺序 0.403）、`reuse_latest_calnorm` B=16 0.558（0.551）、`mean_table_calnorm` 0.516（0.516）、`rownorm_ridge` 0.579（0.578）；`rownorm_ridge_origsel` B=16 0.366（原顺序 0.683）。block-free backbone 和 block-free ridge 对 trial 顺序不敏感（+0.00-0.01），而 `origsel` 依赖顺序。ui_prior / within 行没有做打乱顺序的评测。

## 10. 跨天漂移统计（`drift_stats.json`，tau 80 缓存，剔除 S2_D2_T2）

pattern = 每通道对数平均包络；先在 session 对上取平均，再对 10 名受试者取平均。三个通道集合：12 个始终好的通道（去掉 0 起第 7、15 通道）、13 通道（只去掉通道 15）、14 通道。

| session 对 | 12 通道：corr / log 增益比 std | 13 通道（去掉 ch15） | 14 通道 |
|---|---|---|---|
| 同 session 内（奇偶重复） | 0.995 / 0.075 | 0.998 / 0.077 | 0.998 / 0.082 |
| 同一天（T1 vs T2） | 0.829 / 0.417 | 0.937 / 0.469 | 0.955 / 0.471 |
| 跨天 gap 1 | 0.833 / 0.411 | 0.938 / 0.453 | 0.931 / 0.537 |
| gap 2 | 0.805 / 0.443 | 0.926 / 0.495 | 0.888 / 0.672 |
| gap 3 | 0.806 / 0.449 | 0.930 / 0.511 | 0.855 / 0.769 |
| gap 4 | 0.755 / 0.535 | 0.910 / 0.643 | 0.832 / 0.897 |

- **14 通道的增益比 std 随 gap 的上升（0.54 -> 0.90）主要由通道 15 在 S1-S4 的 target 里变死造成**：去掉它后是 0.45 -> 0.64，12 个始终好的通道是 0.41 -> 0.54（随 gap 仍有上升，幅度小得多）。14 通道的 pattern 相关随 gap 的下降（0.955 -> 0.832）同样如此，13 通道里基本是平的（0.94 -> 0.91）。
- **同日（T1 vs T2）与 gap 1 的比较依赖通道集合和 S9**：12 通道与 13 通道上 gap 1 - 同日的 log 增益比 std 差 = -0.005/-0.016（CI +-0.08/+-0.11，含 0），去掉 S9 后 +0.023/+0.028（CI +-0.05，含 0），即无法区分同日与 gap 1；14 通道上 gap 1 比同日大 0.066 +- 0.135（去掉 S9 为 0.117 +- 0.08，显著），因为通道 15 在 gap 1 的 target 里开始死。S9 是离群受试者（12 通道上同日 std 1.15、gap 1 0.88，其余 9 名受试者的同日 std 为 0.15-0.63、gap 1 为 0.15-0.71）。
- 事实不变：漂移是逐通道相对增益 / 空间模式的变化，不是整体幅度（log 增益比均值接近 0）；同 session 内两半几乎一致（0.995），所以同 session 内留出重复的 source val 几乎看不到漂移，这是 source val 与跨 session 之间差距的来源。“同日漂移已与跨天同量级” 在 12/13 通道上成立，在 14 通道上不成立；“主要来自 session 切换而非时间间隔” 只是推测（电极重新放置没有在数据中验证）。

## 11. 披露与局限

- **早期 probe 评测（review 第 5 条）**：`results/db6_20261015/probe_eval/rows.jsonl`（600 行）是 v1 开发期的冒烟测试：一个只训练 400 步（source val 0.628）、tau 80 的 K=8 probe checkpoint，在 S1 的 6 个 target session（D3-D5 的 T1/T2）上跑了所有方法和 B in {1,2,4,8,16,42}，写于最终模型选择之前（22:10-22:26，最终 checkpoint 22:38）。S1 是 n=10 之一，所以 S1 的 target 结果不是盲的；我没有依据该评测改任何选择，但 v1 的固定值（ui_prior 300 步 / lr 3e-3、tau 起点等）是在看过它之后确定的这一点不能排除，v2 沿用了这些固定值而没有重新调。
- **v2 中我看过的 target 数据**：全量评测前用 `S1_D3_T1`（2 次 draw）做了一次运行时间与冒烟检查并看了该 session 各方法的 balanced accuracy；此后没有用 target 做过任何选择。全量评测期间看过 seed 1 的汇总（seed 0 尚未完成）；该汇总没有改变任何设定。
- ridge 的 “block-free 选择” 与 “原顺序选择” 两个版本都是在 `S1_D3_T1` 冒烟运行之前，仅依据 source 模拟（两种顺序）定义的；冒烟运行里 origsel 明显更高，但主行取 block-free 的决定依据是 5.1 / 5.4 的 source 证据（block 结构捷径），不是 target 分数。
- 评测协议里 B-trial 校准 trial 在记录中可能晚于 query trial（见第 3 节）。
- 窗口 / burn-in 参数（512-1024、128、5% 起点窗口）沿用 review 建议，没有搜索，只试了一个 1024-2048 的变体。v2a 只有 1 个 seed；`within_*` 与 B=42 只有 seed 0、3 次 draw；seed 1 略欠训练（5.2）。block-free 是用 “trial 顺序重排” 实现的，拼接点处的包络跳变是人工痕迹，未单独评估其影响；也没有验证 Mamba 的整段并行扫描与逐步递推推理在数值上一致（测试只检查了因果性与状态携带）。
- 未做：Overleaf exp-log、GDrive 备份、push（只提交到本地 git）；per-class（见过 / 没见过的类）的拆分；ui_prior / within 行的打乱顺序评测；target 端从头训练整模型（trunk + 表）的上限；reliability-shrinkage / population-conditioned 生成本身；H3。
- 另一个 agent 的 `ssm_decode/semg_calib_db6.py`（未提交）已 import `db6_backbone`（`load_backbone`、`SourceSet`、`decode_z`、`score_stream`、`TRAIN_STD_FLOOR` 等），我保持了这些名字和签名稳定；其中 `fit_tables`/`ridge_fit` 的签名已改变（现在接收片段列表），若它调用了旧签名需要同步。

## 12. 文件与复现

代码：`ssm_decode/db6_data.py`（仅文档字符串更新）、`ssm_decode/db6_backbone.py`（重写）；测试：`tests/test_db6_data.py`、`tests/test_db6_backbone.py`；脚本：`scripts/run_db6_train.py`、`run_db6v2_select.sh`（阶段 1）、`run_db6v2_floor.sh`（阶段 2）、`run_db6v2_final.sh`、`select_db6_std_floor.py`、`select_db6_ridge.py`、`run_db6_eval.py`、`db6_order_shuffle_diag.py`、`summarize_db6.py`、`summarize_db6_selection.py`、`compare_db6_v1_v2.py`、`format_db6_tables.py`、`db6_drift_stats.py`。旧脚本 `run_db6_grid*.sh`、`run_db6_final*.sh` 是 v1 的，保留（它们调用的训练接口已改为 v2，所以不能原样复现 v1；v1 在提交 261fe35）。

本目录：`reference_rows_main.{csv,json}`、`reference_rows_seed{0,1}.csv`、`per_subject_main.csv`、`paired_diffs_main.csv`、`calibration_coverage.csv`、`h_flags_main.json`、`backbone_selection.csv`、`drift_stats.json`、`tables.md`（第 7 节的表）、`v1_vs_v2.md`、`extra_*/`（v1 checkpoint 连续解码、原顺序训练的 v2a、打乱顺序评测、floor 0.05 重训的敏感性）、`raw/`（逐行 rows 的 gz、ridge 选择、std floor 选择、order-shuffle 诊断、final 的 summary/history）。checkpoint 与完整 rows 在 `results/db6_v2_20261016/`（gitignore）。

复现（顺序）：`run_db6v2_select.sh {0,1}` -> `run_db6v2_floor.sh {0.05,0.2}` -> `select_db6_std_floor.py` -> `select_db6_ridge.py --floor 0.01 --order {shuffled,original}` -> final seed 0（阶段 1 运行的拷贝）与 `run_db6v2_final.sh 1` -> `run_db6_eval.py --ckpt final/seed$S/best.pt --floor 0.01 --ridge-cfg ... --ridge-cfg-alt ...`（`--no-diag` 对 seed 1）-> `summarize_db6.py`。环境：`CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=SSM/.tools/mamba_deps:SSM python ...`。
