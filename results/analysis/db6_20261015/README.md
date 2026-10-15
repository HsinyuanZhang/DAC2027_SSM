# NinaPro DB6 管线、backbone 与跨天参照行（协议 D-A，2026-10-15/16）

对应 `docs/feasibility_p1_20261015.md` 第 3 节的 Hyser 模式，换成 DB6 抓取分类：10 名受试者 x 5 天 x 每天 2 个 session（上午 T1、下午 T2），14 通道 sEMG，7 种抓取 + 休息。
全部 GPU 计算在 GPU1（`CUDA_VISIBLE_DEVICES=1`，cuda:1），GPU0 未使用。所有超参数（K、模型大小、dropout、tau、ridge 超参数）只在 source validation 上选，target query 没有参与任何选择。

## 1. 数据事实

| 项 | 值 |
|---|---|
| 来源 | `/mnt/data/datasets/ninapro_db6/DB6_s{1..10}_{a,b}.zip`（保留不动），解压到 `.../extracted/`（21 GB，HDD） |
| 文件 | `S{s}_D{d}_T{t}.mat`，MATLAB v5（`scipy.io.loadmat` 可读，无需 h5py），s=1..10，d=1..5，t=1（上午）/2（下午），共 100 个 session |
| 字段 | `emg [T,16] float32`（伏特，2000 Hz）、`stimulus/restimulus`、`repetition/rerepetition`、`object/reobject`、`acc [T,48]`、`daytesting`、`time`、`subj`。读取时核对文件名与 `subj/daytesting/time` 一致 |
| 通道 | 16 列中第 9、10 列（0 起第 8、9 列）在所有 100 个文件里恒为 0；其余 14 列是 Delsys 电极。其中第 8、16 列（0 起 7、15）在所有 session 里 std 只有约 0.9 uV 以下（多数 0.1-0.6 uV；其他通道约 2.6-1500 uV），是噪声底通道；管线仍保留这 14 列，没有剔除 |
| 标签 | 抓取码（`restimulus`）{1,3,4,6,9,10,11} 映射为类 1..7，休息为 0，共 8 类。使用重标注的 `restimulus/rerepetition`（对齐实测动作起点），不用 `stimulus/repetition`（对齐屏幕提示） |
| session 结构 | 99 个 session 是 7 个抓取各连续 12 次重复，共 84 个 trial，约 11.9 min（35.7k 个 50 Hz bin）。**S2_D2_T2 是截断文件**（103 s，只有抓取 1，12 个 trial；`tests/test_db6_data.py` 里有检测），已从 source 剔除；它不是 target，所以 target 无损失 |
| 类别分布（bin） | 休息 28.8%，每个抓取约 9.9%-10.4%；chance 的 balanced accuracy = 0.125 |

**Trial 的定义**：`rerepetition` 把数据流切成 `[该次重复之前的休息段 | 该次动作]` 的连续段（计数器在上一个动作结束时递增）。一个 trial = 一段 `rerepetition` 恒定且 >0 的最大连续 bin 段，且必须恰含一个非零 `restimulus` 码，否则记入 `meta['bad_runs']` 并排除（实际没有任何 session 出现 bad run）。最后一次动作之后约 1 s 的休息（`rerepetition == 0`）不属于任何 trial。每个 trial 有 1 个抓取 + 它前面的休息段；84 个 (抓取, 重复) 对各出现一次，trial 之间互不重叠。平均 424 bin（8.5 s，其中动作 302 bin、休息 122 bin），最短 113，最长 700。每个 trial 当作独立序列从零状态解码（右侧补零到 704 bin，用 mask）。

## 2. 因果流式前端（`ssm_decode/db6_data.py`）

1. V -> mV，逐通道整流 |x|；2. 一阶泄漏积分器 `y[n] = y[n-1] + a(|x[n]| - y[n-1])`，`a = 1 - exp(-1/(fs·tau))`，初态 0，在整个 session 流上运行（状态跨 trial 保持，和实际流式一致）；3. 降到 50 Hz：2000/50 = 40 整除，取每个 20 ms bin 最后一个样本（bin k 取下标 `(k+1)*40-1`）；4. bin 的类别标签和 trial 归属都取该 bin-end 样本处的 `restimulus` / `rerepetition`。bin k 的输出只依赖到 bin k 末端为止的输入。缓存：`results/data_cache/db6/s{SS}_d{D}_t{T}_tau{TTT}.npz`（float32，tau 20/40/80/160/320 各一套，共 1.0 GB，gitignore；根分区剩余 216 GB）。

测试 `tests/test_db6_data.py`（9 项）：真实文件解析（形状、零通道、码集合、文件名与头字段一致）、envelope 的前缀不变性与阶跃响应、bin 取样下标、label/bin 对齐（合成数据）、trial 切分（合成数据 + 真实文件：84 个 (抓取, 重复) 恰好各一次、trial 互不重叠且恰好覆盖 `rerepetition>0` 的全部 bin、每个 trial 只含休息和它自己的抓取）、cache 往返（两套 tau）、已知截断 session 的检测。`tests/test_db6_backbone.py`（5 项）：校准抽样平衡且不重复、协议 key、padding 不影响统计、指标（balanced accuracy / macro-F1 / vote）、ridge 闭式解与因果性。共 14 项通过。

**tau 的选择**：在 source validation 上选，tau in {20,40,80,160,320} ms（同一配置 K=8 w128 l2 s32 dropout 0.1），source val balanced accuracy（15000 步上限，2 个 seed 的平均）：tau 80 为 0.8615（0.8657/0.8573），tau 160 为 **0.8728**（0.8737/0.8720），tau 320 为 0.8688（0.8780/0.8595）；tau 20、40 只跑了 seed 0（0.8615、0.8697）。最终取 **tau = 160 ms**（任务给的起点是 80 ms）。差距 0.01，与 seed 间波动（最多 0.02）同量级，不算强结论；tau 20-320 的整体范围 0.86-0.88。drift 统计（第 5 节）用的是 tau 80 ms 的缓存。

## 3. Backbone 与选择

结构：`APBackbone`（未修改）= 每个 source session 一张 K x 14 的 read-in table + 共享 Mamba-3 trunk（官方实现）+ 线性 head，输出 8 类 logit，损失 = trunk 交叉熵 + 0.5 x head 交叉熵 + 0.1 x whitening（沿用 AP 约定）。`ssm_decode/db6_backbone.py` 是薄封装（训练循环、指标、ridge、table 拟合），没有改任何共享模块。

- source：全部 10 名受试者的 day 1-2 共 39 个 session（4 x 10 - 截断的 S2_D2_T2），每个 session 一张自己的表。每个抓取的第 4、8、12 次重复留出作 source validation（每个 session 63 个训练 trial、21 个验证 trial）。输入：每个 session 用自己训练 trial 的逐通道 z-score；训练用 `drift_augment`（gain 0.3、offset 0.1、channel dropout 0.05）；表用 `init_from_supervised`（ridge 加 PCA 补全）初始化。
- 选择指标：source validation 的 per-bin balanced accuracy（均值跨 39 个 session）。
- 阶段 1（seed 0，tau 80，6000 步，18 次运行，`backbone_selection.csv`）：K in {4,8,14} x (width,layers,state) in {(64,1,32),(128,2,32),(256,2,64)} x dropout {0.1,0.3}。K=4 明显更差（0.72-0.80）；K=8/14 的 w128/w256 基本持平在 0.847-0.859（K=14 w256 d0.1 0.859 最高，K=8 w128 d0.1 0.8507；唯一的离群是 K=8 w128 d0.3 的 0.826）；w64 l1 容量不足（0.81-0.82）。阶段 1 的最优步都在末尾，不满足"best step 不在最后 20%"。
- 阶段 2（15000 步上限，patience 15）：K=8 w128 d0.1 0.8657、K=8 w256 s64 0.8679、K=14 w128 d0.3 0.8628，三者持平（差 < 0.005），取参数最少、表最小的 **K=8、width 128、2 层、state 32、dropout 0.1**（trunk 354,072 参数，39 张表共 4,368 参数；一张表 8 x 14 fp16 = 224 B，远小于 32 KB）。然后在该配置上扫 tau（上节）。
- 阶段 3：tau 160 配置上的 Hyser 式扰动 weight decay 0.1（0.8692）、channel dropout 0.2（0.8547）都没有超过基线，保持默认。
- 最终（2 个 seed，tau 160，best step 都在 15000 步预算的 70% 左右）：seed 0 source val balanced accuracy 0.8737（macro-F1 0.8733，best step 10300），seed 1 0.8720（0.8720，best step 10200）；各 session 的验证分数范围 0.71-0.98。

## 4. 评测协议与各行

### 4.1 协议 D-A

- source = 10 名受试者的 day 1-2 全部 session（39 个），target = 每名受试者的 day 3、4、5 的 session（每天 T1、T2，共 60 个 target session），时间向前，不用 day 3/4 作为 day 4/5 的 source。day gap = target 天 - 2：day 3 / 4 / 5 对应 gap 1 / 2 / 3。
- 校准预算 B in {1, 2, 4, 8, 16}（总 trial 数，不是每类），query = 该 session 其余的 84-B 个 trial。抽样：随机排列 7 个抓取，轮转取，每次从该抓取没用过的重复里随机取一个（休息出现在每个 trial 中）。因此各抓取的计数相差至多 1。覆盖（`calibration_coverage.csv`）：B=1 -> 休息 + 1 个抓取（8 类中 2 类）；B=2 -> 3 类；B=4 -> 5 类；B>=8 -> 8 类全覆盖。每个 (target, B) 抽 5 次，随机数只依赖 (target, B, draw)，所以所有方法与 2 个 seed 使用相同的划分。
- 任务：per-bin 8 类分类。指标：per-bin balanced accuracy（主，每类 recall 的均值）、macro-F1（8 类）、以及每个 trial 的 majority-vote 抓取准确率（次要：对该 trial 内预测为非休息的 bin 取多数类，全为休息则判休息；真值 = 该 trial 的抓取）。
- 聚合：5 次 draw 取平均 -> 同一天的 2 个 session 取平均 -> 2 个 seed 取平均 -> 对 10 名受试者取均值；"±" 是 t 分布 95% CI（n=10 受试者）。"all gaps" = 每名受试者三个 gap 的平均。
- 诊断预算 B=42（半个 session 训练、半个 session query）只用于 within-session 对比：只跑 seed 0，每个 target 3 次 draw（为节省 GPU 时间；seed 1 的 B=42 没有完整做，已从汇总里排除）。

### 4.2 参照行

每个 target 和 B：
- `reuse_latest_srcnorm`：同一受试者最新 source session（day 2 T2；S2 用 day 2 T1，因为 D2_T2 被剔除）的表，输入用该 source session 的统计量（零 target 数据，所以不随 B 变化）。`reuse_latest_calnorm`：同一张表，输入用 B 个 trial 的无标签逐通道统计量重新 z-score（自然的不校准基线，只用神经数据）。
- `mean_table_calnorm`：39 张 source 表的均值 + calnorm；`mean_loso_calnorm`：去掉该受试者自己的 4 张表后的均值（更干净的 source mean）。
- 部署型闭式解 `rownorm_ridge_v2`：B 个 trial 的逐通道 z-score（行归一化）+ 因果 EMA + 14 -> 8 的 ridge 回归到 one-hot（LDA 风格的闭式解，不反传 trunk，不用 trunk）。(lam, ema) **只在 source 上、对每个 B 单独选**：在每个 source session 里用其训练 trial 模拟 B-trial 校准（3 次 draw），在它的留出重复上评分。选到的值：B=1,2 -> lam 100、ema 0；B=4 -> 100、0.98；B=8 -> 10、0.99；B=16 -> 1、0.99；B=42 -> 0.1、0.995（网格 lam 1e-4..1e3，ema 0..0.995）。这是第二遍：第一遍（`rownorm_ridge_v1grid`，lam 0.01..1000，ema 0..0.95）的 source 选择停在了网格边界（B=16、42 选到 lam 下界与 ema 上界），在看 target 结果的同时我注意到这点，于是只依据 source 模拟分数把网格扩大后重跑 ridge 行（source 模拟分数 B=16：0.518 -> 0.574，B=42：0.553 -> 0.627）。两个版本都保留在表里；用 v2 作为主要的 closed-form 基线，其选择过程没有用到 target 分数。
- `ui_prior_*`（**非部署参照**，梯度反传穿过 frozen trunk，只更新 read-in table）：Adam，300 步，lr 3e-3，交叉熵，全 batch 使用 B 个 trial，固定值，事先给定，没有在 target 上调。三个变体：从最新 source 表出发 + source 统计量（`ui_prior_latest_srcnorm`）、从最新 source 表出发 + calnorm（`ui_prior_latest_calnorm`）、从 mean 表出发 + calnorm（`ui_prior_mean_calnorm`）。
- within-session 诊断（**仅诊断**，B=42）：`within_table_latestinit_long` = 从最新 source 表出发、1000 步；`within_table_supinit` = 从 target 数据的 supervised 初始化出发、1000 步。没有做"在 target 上从头训练 trunk"的 within-session 模型（未执行）。

H 项：H1（trunk 哈希前后一致）所有 28,980 行都成立；H5（表 <= 32 KB：224 B）成立；H2：reuse/mean/ridge 行 0 次反传，ui_prior 300 步、within 1000 步（所以仅为参照）；H4：校准只用 B 个有标签 trial（ui_prior 与 within 的标签用于梯度拟合，已标注为非部署）。H3（固定权重乘加比例）没有计算，这些行只是基线，不是候选。

### 4.3 结果（balanced accuracy，均值 ± 95% CI，n = 10 受试者，all gaps，2 seeds）

chance = 0.125；source validation（同一天、留出重复）= 0.873，它是这个 backbone 在没有跨 session 漂移时的水平。

| method | B=1 | B=2 | B=4 | B=8 | B=16 | B=42（诊断，seed 0） |
|---|---|---|---|---|---|---|
| `reuse_latest_srcnorm`（零 target 数据） | 0.429 ± 0.076 | 0.429 ± 0.076 | 0.429 ± 0.076 | 0.429 ± 0.076 | 0.429 ± 0.076 | 0.434 ± 0.075 |
| `reuse_latest_calnorm` | 0.403 ± 0.043 | 0.475 ± 0.058 | 0.533 ± 0.069 | 0.565 ± 0.074 | 0.574 ± 0.075 | 0.583 ± 0.081 |
| `mean_table_calnorm` | 0.440 ± 0.056 | 0.494 ± 0.051 | 0.527 ± 0.049 | 0.550 ± 0.053 | 0.548 ± 0.053 | 0.557 ± 0.057 |
| `mean_loso_calnorm` | 0.427 ± 0.055 | 0.479 ± 0.049 | 0.509 ± 0.048 | 0.531 ± 0.053 | 0.529 ± 0.053 | 0.536 ± 0.056 |
| `rownorm_ridge_v2`（部署型 closed-form） | 0.199 ± 0.015 | 0.275 ± 0.020 | 0.395 ± 0.027 | 0.537 ± 0.045 | 0.608 ± 0.043 | 0.672 ± 0.043 |
| `rownorm_ridge_v1grid`（第一遍，网格受限） | 0.199 ± 0.015 | 0.275 ± 0.020 | 0.388 ± 0.029 | 0.507 ± 0.047 | 0.564 ± 0.052 | 0.604 ± 0.058 |
| `ui_prior_latest_srcnorm`（参照） | 0.387 ± 0.044 | 0.436 ± 0.056 | 0.523 ± 0.060 | 0.621 ± 0.069 | 0.688 ± 0.070 | 0.763 ± 0.058 |
| `ui_prior_latest_calnorm`（参照） | 0.374 ± 0.032 | 0.470 ± 0.047 | 0.559 ± 0.050 | 0.663 ± 0.056 | 0.726 ± 0.057 | 0.802 ± 0.053 |
| `ui_prior_mean_calnorm`（参照） | 0.384 ± 0.031 | 0.463 ± 0.035 | 0.557 ± 0.042 | 0.661 ± 0.059 | 0.719 ± 0.063 | 0.800 ± 0.056 |
| `within_table_latestinit_long`（诊断） | - | - | - | - | - | 0.803 ± 0.051 |
| `within_table_supinit`（诊断） | - | - | - | - | - | 0.793 ± 0.055 |

按 day gap（balanced accuracy，B=4 / B=16；完整数据在 `reference_rows_main.csv`）：

| method | gap 1 (day 3) | gap 2 (day 4) | gap 3 (day 5) |
|---|---|---|---|
| `reuse_latest_srcnorm` | 0.441 / 0.441 | 0.460 / 0.461 | 0.387 / 0.386 |
| `reuse_latest_calnorm` | 0.566 / 0.613 | 0.540 / 0.591 | 0.492 / 0.519 |
| `mean_table_calnorm` | 0.530 / 0.548 | 0.528 / 0.555 | 0.523 / 0.541 |
| `rownorm_ridge_v2` | 0.397 / 0.599 | 0.402 / 0.613 | 0.386 / 0.611 |
| `ui_prior_latest_calnorm` | 0.578 / 0.737 | 0.574 / 0.740 | 0.526 / 0.699 |

macro-F1 与 per-trial majority-vote 准确率（all gaps，同样的 B 列）：

| method | macro-F1 B=1 / 4 / 16 | vote acc B=1 / 4 / 16 |
|---|---|---|
| `reuse_latest_srcnorm` | 0.382 / 0.382 / 0.382 | 0.395 / 0.395 / 0.395 |
| `reuse_latest_calnorm` | 0.367 / 0.516 / 0.562 | 0.368 / 0.536 / 0.588 |
| `mean_table_calnorm` | 0.407 / 0.508 / 0.533 | 0.411 / 0.522 / 0.548 |
| `rownorm_ridge_v2` | 0.108 / 0.322 / 0.622 | 0.132 / 0.412 / 0.742 |
| `ui_prior_latest_calnorm` | 0.338 / 0.537 / 0.721 | 0.335 / 0.560 / 0.768 |
| `within_table_latestinit_long`（B=42） | 0.802 | 0.859 |

macro-F1 与 balanced accuracy 基本同序，vote 准确率在 B>=8 时比 per-bin 高：reuse 约 +0.01-0.02，ui_prior 约 +0.03-0.06，ridge 约 +0.1-0.13（trial 内多数投票平滑掉了 bin 级错误；ridge 的 bin 级错误最碎）；B<=4 时 vote 准确率受制于未见过的抓取（B=1 的 ridge 只可能预测休息和一个抓取，vote = 1/7）。

配对差（同一受试者上相减，all gaps，95% CI，n=10；全表见 `paired_diffs_main.csv`）：
- `reuse_latest_calnorm - reuse_latest_srcnorm`：B=1 -0.026 ± 0.044，B=2 +0.046 ± 0.041，B=4 +0.103 ± 0.045，B=8 +0.136 ± 0.049，B=16 +0.145 ± 0.051。
- `ui_prior_latest_calnorm - reuse_latest_calnorm`：B=1 -0.029 ± 0.018，B=2 -0.005 ± 0.019，B=4 +0.027 ± 0.029（不显著），B=8 +0.098 ± 0.040，B=16 +0.151 ± 0.048，B=42 +0.219 ± 0.059。
- `ui_prior_latest_calnorm - rownorm_ridge_v2`：B=1 +0.175 ± 0.021，B=4 +0.164 ± 0.032，B=8 +0.126 ± 0.029，B=16 +0.118 ± 0.026，B=42 +0.130 ± 0.024。
- `rownorm_ridge_v2 - reuse_latest_calnorm`：B<=4 为 -0.14 到 -0.20（显著更差），B=8 -0.028 ± 0.065，B=16 +0.033 ± 0.062，B=42 +0.090 ± 0.066（B>=8 不显著）。
- `reuse_latest_calnorm - mean_table_calnorm`：B=1 -0.037 ± 0.049，B=4 +0.006 ± 0.066，B=16 +0.026 ± 0.072（都不显著）。
- `ui_prior_latest_calnorm - ui_prior_mean_calnorm`：|差| <= 0.009，CI 全部包含 0（初始化方式无关）。
- 两个 seed 的 target 均值互差不超过 0.02（例如 B=16：`ui_prior_latest_calnorm` 0.734 / 0.717，`reuse_latest_calnorm` 0.583 / 0.566），见 `reference_rows_seed{0,1}.csv`。
- 受试者间差异大：B=16 的 `ui_prior_latest_calnorm` 在各受试者上从 0.61（S9）到 0.83（S1、S5）；`reuse_latest_calnorm` 从 0.44（S9、S10）到 0.75（S1）。没有排除任何受试者。

## 5. 跨天漂移有多大（`drift_stats.json`，tau 80 缓存，剔除 S2_D2_T2）

pattern = 每通道对数平均包络（`overall`，所有 trial 的 bin）；`per_grasp` = 每个抓取的 pattern 展平后的空间相关（先去掉各自均值）。下表是 12 个非噪声底通道，对每个受试者先在 session 对上取平均，再对 10 名受试者取平均。

| session 对 | pattern 相关（overall） | pattern 相关（per grasp） | 逐通道 log 增益比的标准差 | log 增益比均值 |
|---|---|---|---|---|
| 同 session 内（奇数 vs 偶数重复） | 0.995 | 0.939 | 0.075 | 0.025 |
| 同一天（T1 vs T2，上午 vs 下午） | 0.829 | 0.819 | 0.417 | 0.101 |
| 跨天 gap 1 | 0.833 | 0.821 | 0.411 | 0.011 |
| 跨天 gap 2 | 0.805 | 0.796 | 0.443 | 0.002 |
| 跨天 gap 3 | 0.806 | 0.792 | 0.449 | 0.002 |
| 跨天 gap 4 | 0.755 | 0.744 | 0.535 | 0.046 |

含两个噪声底通道的 14 通道版本相关更高（同日 0.955，gap 4 0.832），log 增益比的标准差同日 0.47，gap 1/2/3/4 为 0.54/0.67/0.77/0.90，见 JSON 的 `all14`。

事实：(1) 漂移是逐通道相对增益 / 空间模式的变化，不是整体幅度（log 增益比均值几乎为 0）；(2) **同一天内上午与下午之间的漂移已经和跨天同量级**（相关 0.83、增益比 std 0.42，与 gap 1 的 0.83、0.41 几乎相同），所以 DB6 的 session 间漂移主要来自 session 切换本身（推测是每个 session 重新放置电极，这一点我没有在数据里验证），而不是时间间隔本身；(3) gap 越大还会继续变差，但幅度较小（gap 1 -> 4：相关 0.83 -> 0.76，增益 std 0.41 -> 0.54）；(4) 同 session 内两半几乎一致（0.995，0.075），所以 source validation（同一 session 内留出重复）几乎看不到漂移，这是 source val 0.873 与跨 session 0.43-0.58 之间巨大差距的来源。

## 6. 对校准方法的含义

1. **跨 session 的缺口很大，且不是靠重新归一化能补回的。** 零 target 数据复用 0.429（chance 0.125），source val 0.873；只用 B=16 个 trial 的无标签统计量重新 z-score 到 0.574（比 srcnorm +0.145 ± 0.051），B=42 也只有 0.583。统计量估计在 B=4 之后已经饱和（B=8、16、42 差异小于 0.02），剩下的是表结构 / 空间模式的变化，需要标签才能修。小 B 时 calnorm 反而不好：B=1 比 srcnorm 低 0.026（不显著），统计量来自一个抓取 + 休息，估计偏；这是 reliability shrinkage 方向的理由，但本轮没有测试。
2. **有标签的表更新在 B>=8 才有意义。** `ui_prior`（梯度参照）在 B=4 与 reuse 持平（+0.027 ± 0.029），B=8 +0.098，B=16 +0.151，B=42 +0.219（0.802）。B<=2 时三种 ui_prior 变体都低于零标签的先验（B=1：0.374-0.387 对 `mean_table_calnorm` 0.440，B=2 同样落后）：1-2 个 trial 只含 2-3 类，梯度拟合在少数类上过拟合并丢掉先验对其他类的知识。小 B 的最优方法仍是"先验 + calnorm"（B=1 的 `mean_table_calnorm` 0.440 > `reuse` 0.403 > ui_prior），没有标签方法在 B<=2 里胜过它们。
3. **table-only 的上限约 0.80，低于 source val 0.873。** B=42 的 `ui_prior`（300 步）0.802、`within_table_latestinit_long`（1000 步）0.803、`within_table_supinit` 0.793 三者持平，说明在这个 backbone 下仅靠改 K x 14 read-in 表能回收到约 0.80，剩下约 0.07 是 table 表达不了的漂移（或需要更新 trunk；也可能受优化设定和 42 个 trial 的数据量限制，没有进一步检验）。所以对表生成方法而言，天花板大约是 0.80，B=16 的 ui_prior（0.726）已回收天花板与 reuse（0.574）之间约 2/3 的差距。
4. **部署型闭式解的位置。** v2 ridge 在 B=8 与 reuse 持平（-0.028 ± 0.065），B=16 略高（+0.033 ± 0.062，不显著），B=42 到 0.672（+0.090 ± 0.066）；B<=4 受制于没见过的类（B=1 最多预测 2 类，balanced accuracy 上界 0.25；B=2 上界 0.375，B=4 上界 0.625），因此很差（0.20、0.28、0.40）。ui_prior 比它高 0.12-0.20（B=1 到 B=42 全部显著），所以 B>=8 时"梯度穿过 trunk 的表更新"比"不用 trunk 的闭式解"有稳定的约 0.12-0.13 的余量；任何想用闭式 / local rule 逼近 ui_prior 的校准方法，在 B=8-16 的潜在收益是 0.12-0.15，比较对象应是 reuse（0.565-0.574）与 ridge v2（0.537-0.608），而不是 mean 表。第一遍网格受限的 ridge（`v1grid`）在 B=16/42 低 0.04/0.07，说明闭式基线的 EMA 平滑强度（时间常数约 2 s，匹配 8.5 s 的 trial 与块状标签）对结果很重要。
5. **own-subject 先验并不比群体均值好。** `reuse_latest_calnorm` 与 `mean_table_calnorm` 在 B>=4 无显著差（+0.006 到 +0.026，CI ±0.07），去掉自己 4 张表的 `mean_loso_calnorm` 低约 0.02；`ui_prior` 从 latest 表还是 mean 表出发无差别（<= 0.009）。随 gap 变大 reuse 变差（B=16：0.613 -> 0.591 -> 0.519），而 mean 表基本不变（0.548 -> 0.555 -> 0.541），到 gap 3 mean 表（0.541）反而高于自己最新的表（0.519）。这与 Hyser 的结论相反（那里同一受试者的 session1 表远好于 mean 表）：DB6 的 session 间漂移（同日上午/下午已与跨天同量级，见第 5 节）使 session 间的表更接近"随机漂移的样本"，而不是稳定的受试者专属映射。这意味着群体信息（population-conditioned 生成）的价值不依赖"同一受试者的先验"，反过来"最新 session 的表"这个强先验在 DB6 里并不强，群体生成的比较对象应是 reuse 与 mean 两者。
6. **小 B 的瓶颈是类覆盖，不是统计量估计。** B<=4 时 8 类中只有 2/3/5 类被看到；任何只用 B 个 trial 的方法对没见过的类都没有信息，只有来自 source 的先验（reuse、mean 表）能预测它们，这也是 B<=2 时先验行最强、B>=8（8 类全覆盖）时才出现 ui_prior 的优势的原因。这个现象来自协议（"总 trial 数，不是每类"），不是实现问题。

## 7. 没做的事 / 局限

- 未做：Overleaf exp-log 更新、GDrive 备份、push（按要求只提交到本地 git）；per-class（见过 / 没见过的类）的拆分（行里没有存 per-class recall，补做需要重跑）；target 端从头训练整模型（trunk + 表）的 within-session 上限；reliability-shrinkage / population-conditioned 生成本身（这里只做参照行）；H3（乘加比例）。
- B=42 的诊断行只有 seed 0、3 次 draw；seed 1 的 B=42 行不完整，已从汇总里排除。
- ui_prior 超参数（300 步、lr 3e-3）固定，未调。within 诊断没有在 target 之外选超参数。
- tau 的选择差距（160 对 80 是 +0.011）与 seed 间波动同量级；阶段 1 没有满足"best step 不在最后 20%"（6000 步预算），阶段 2 起放宽到 15000 步，之后满足。
- ridge 的网格先后跑了两遍，v1grid 与 v2 都已报告；v2 的网格扩展是在看到 source 选择停在网格边界之后做的，但同时 v1 的 target 结果我也已经看到，所以 v2 并非严格盲选；其选择只用 source 模拟分数，不涉及 target query。
- 两个噪声底通道保留在输入里，未专门处理；drift 统计用 12 通道作主结果，14 通道版本在 JSON 里。
- S2_D2_T2 截断，source 少 1 个 session；S2 的"最新 source 表"因此是 day 2 T1。
- 每个 (target, B) 只有 5 次 draw；CI 以受试者为单位（n=10），没有把 backbone 训练 seed 的方差包含进去（2 个 seed 的均值相差 <= 0.02）。

## 8. 文件与复现

代码：`ssm_decode/db6_data.py`、`ssm_decode/db6_backbone.py`；测试：`tests/test_db6_data.py`、`tests/test_db6_backbone.py`；脚本：`scripts/run_db6_train.py`、`run_db6_grid.sh`（阶段 1）、`run_db6_grid2.sh`（阶段 2）、`run_db6_final.sh` / `run_db6_final2.sh`（阶段 3）、`run_db6_eval.py`（评测，`--ridge-only` 为 v2 ridge，`--no-diag` 跳过 B=42）、`summarize_db6.py`、`summarize_db6_selection.py`、`db6_drift_stats.py`。

本目录：`reference_rows_main.{csv,json}`（含各 gap、各 B 的 balanced accuracy / macro-F1 / vote / acc 及 CI）、`reference_rows_seed{0,1}.csv`、`per_subject_main.csv`、`paired_diffs_main.csv`、`calibration_coverage.csv`、`backbone_selection.csv`（所有选择运行）、`drift_stats.json`、`h_flags_main.json`、`raw/`（逐行 rows 的 gz、ridge 选择 v1/v2、最终 backbone 的 summary/history）。checkpoint 和完整 rows 在 `results/db6_20261015/`（gitignore）。

运行环境：`CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=SSM/.tools/mamba_deps:SSM python SSM/scripts/run_db6_eval.py --ckpt results/db6_20261015/final/seed0/best.pt --out results/db6_20261015/eval_seed0 --seed-tag 0 --draws 5`（评测拆成两个受试者区间并行跑）。
