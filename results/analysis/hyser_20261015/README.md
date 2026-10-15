# Hyser HD-sEMG 工作线 B：管线、backbone 与跨天参照行（2026-10-15）

对应 `docs/feasibility_p1_20261015.md` 第 3 节。范围：1dof 力回归，协议 H-A（同一受试者、session1 -> session2）。
全部 GPU 计算在 GPU1（`CUDA_VISIBLE_DEVICES=1`），GPU0 未使用。target 的 session2 没有参与任何选择；所有超参数只在 source validation（session1 留出 trial）上选。

## 1. 数据事实

| 项 | 值 |
|---|---|
| 来源 | PhysioNet hd-semg 2.0.0，`/mnt/data/datasets/hd-semg/`（只读，未移动或复制原始数据） |
| 受试者 / session | 20 名受试者 x 2 个 session（不同天） |
| 1dof 结构 | 每 session 5 根手指 x 3 个 sample = 15 个 trial，每个 25 s；1dof_dataset 目录 30 GB |
| EMG | 256 通道（4 块 8x8 阵列 ED/EP/FD/FP），2048 Hz，51200 样本/trial，WFDB format 16（int16 小端，多路复用） |
| 力 | 5 通道（thumb..little），100 Hz，2500 样本/trial |
| 文件 | `1dof_{raw,preprocess,force}_fingerF_sampleS.{hea,dat}`，共 40 session x 15 trial，600 个 preprocess 文件全部完整 |
| .hea 格式 | `file 16 gain(baseline)/V adc_res adc_zero init_value checksum blocksize name`；物理量 = (digital - baseline)/gain。numpy 解析并校验了样本数、`.dat` 字节数、init value 与 16 位 checksum（全部 600 个 preprocess 文件和 force 文件读入时通过） |
| 物理单位 | 单位是 V，但是放大器输出尺度（EMG 通道标准差约 3e-2 V），不是电极处的 uV；缓存里 EMG 包络以 mV 存。力是传感器放大器输出的 V（不同手指的增益不同，所以 R2 用逐通道、与尺度无关的定义） |
| `preprocess` 内容 | 由功率谱推断（没有找到 toolbox 的文字说明）：相对 `raw`，<5 Hz 能量下降约 300 倍，>500 Hz 约 40 倍，50 Hz 工频约 50 倍；即约 10-500 Hz 带通加 50 Hz 陷波。**审查确认该滤波为零相位（非因果）**：从 `raw` 到 `preprocess` 拟合的最小二乘 FIR 冲激响应近似对称（相关 0.94–0.98），峰值在 0 延迟，约 26% 能量落在未来的 tap 上，但超出 5 ms 的只有约 2.5%。经过 80 ms 包络和 20 ms 分 bin 后，对 R2 的影响应可忽略；但严格来说，前端不是可流式实现的。后续改为从 `raw` 出发，用因果带通 + 陷波重建缓存。从包络起的所有步骤严格因果 |
| 缓存 | `results/data_cache/hyser_1dof/sXX_sesY.npz`，float32，`x [15,1250,256]`、`y [15,1250,5]`，共 747 MB（gitignore）；根分区剩余 221 GB |

ndof 结构（只记录，未实现）：`ndof_dataset/subjectXX_sessionY/ndof_{raw,preprocess,force}_combinationC_sampleS`，C = 1..15（手指组合），S = 1..2，共 30 个 trial/session，格式同上（256 通道 EMG，2048 Hz，25 s；力 5 通道 100 Hz）。`random_dataset` 为 `random_*_sampleS`，S = 1..5。`ssm_decode/hyser_data.py` 里有 `ndof_trial_names()`。

## 2. 因果流式前端（`ssm_decode/hyser_data.py`）

1. 物理量转换；2. 逐通道整流 |x|；3. 一阶泄漏积分器 `y[n] = y[n-1] + a(|x[n]| - y[n-1])`，`a = 1 - exp(-1/(fs·tau))`，初态 0（这就是一阶 SSM，可在硬件上流式实现）；4. 降采样到 50 Hz：2048/50 不是整数，取每个 20 ms bin 末端的积分器样本（bin k 取下标 `(k+1)*2048//50 - 1`），bin 数 1250 = 25 s x 50 Hz；5. 力：取以 bin 末端结束的 2 个 100 Hz 样本的平均（因果）。

tau 取 80 ms（任务要求 50-100 ms）。选择依据与限制：`scripts/hyser_select_tau.py` 只用 session1 source validation，对 tau in {20,40,80,160,320} ms 做 z-score + ridge（无时间记忆）。分数随 tau 单调上升（80 ms：0.728；160 ms：0.772；320 ms：0.791，lambda 取网格下界），因为无记忆的 ridge 需要更长平滑。我没有按这个结果改 tau：SSM trunk 自带时间记忆，这个测试对 SSM 不适用，也没有为其他 tau 重建缓存并训练 SSM（没做）。closed-form 基线则通过额外的因果 EMA（在 source validation 上选，见 4.2）获得更长平滑，因此对基线是公平的。

## 3. Backbone 与选择

结构：`APBackbone`（`ssm_decode/ap_backbone.py`，未修改）= 每个 source session 一张 K x 256 的 read-in table + 共享 Mamba-3 trunk（官方实现）+ 线性 head（head 与 whitening 辅助损失沿用 AP 约定，lam_head = 0.5、lam_white = 0.1）。`ap_pretraining.py` 与 000688 的 NWB record 绑定，所以用薄封装 `ssm_decode/hyser_backbone.py` 写了训练循环，复用了 `APBackbone`、`drift_augment` 和同样的损失，没有改任何共享模块。

- source：20 名受试者的 session1，各自一张表；每个 session 每根手指留出 sample3 作 source validation（10 个训练 trial，5 个验证 trial）。
- 输入：source 用各 session 训练 trial 的逐通道 z-score；标签用 source 训练集合并统计量归一化；R2 在物理单位上算，取 5 个力通道 R2 的平均（另存 pooled R2）。每个 trial 是独立序列，从零状态解码；Mamba-3 在 T = 1250 上的因果性已用前缀不变性检查过。
- 表初始化：`init_from_supervised`；训练：AdamW、batch 16 个整 trial、`drift_augment`（增益 0.3、偏置 0.1、通道丢弃 0.05）、每 50 步验证，patience 16 次验证，上限 6000 步。
- 选择（只看 source validation，seed 0，`backbone_selection.csv`，26 次运行）：
  - 阶段 1：K in {8,16,32} x (width,layers,state) in {(64,1,32),(128,2,32),(256,2,64)} x dropout in {0.1,0.3}。最好：K=16 w128 l2 s32 d0.1，0.8428；K=32 同尺寸 0.8425（持平，取表更小的 K=16，表 8 KB fp16）；w256 不优于 w128；dropout 0.3 一致更差。
  - 阶段 2：围绕最优改 dropout {0,0.05}、weight decay 0.1、lr 5e-4、channel dropout 0.2：全部在 0.835-0.841，没有超过，保持阶段 1 的配置。
  - 最终：K=16、width 128、2 层、state 32、dropout 0.1；trunk 354,709 参数，20 张表共 81,920 参数。2 个 seed：seed0 source val R2 = 0.8428（best step 4600/6000），seed1 = 0.8356（3600/6000），都满足"best step 不在最后 20%"；**更正（审查）**：两个 seed 的 target 平均值并非都相差不超过 0.03。`reuse_s1_srcnorm` 为 −0.10 对 −0.34（各个 B 都差约 0.24）；`reuse_s1_calnorm` 在 B=1 时为 0.207 对 0.126，B=2 时相差 0.06；`ui_prior_*` 在 B=1/2 时相差 0.05–0.07。所以“零 target 数据复用 = −0.22”依赖 seed，跨 target 的 CI 不包含 backbone 的方差（`reference_rows_seed{0,1}.csv`）。
- 参照：source 上 ridge（无 trunk）的 val R2 约 0.79，所以 SSM trunk 在同日验证上高约 0.05。

## 4. 评测协议与各行

### 4.1 协议

target = 每名受试者的 session2（20 个 target）。B in {1,2,5,10}，按手指均衡抽取（随机手指顺序轮转，每次抽一个未用的 sample；B=5 每指 1 个，B=10 每指 2 个，B<5 时有手指没见过）。query = 该 session 其余 trial（15-B 个）。每个 (target, B) 抽 8 次，抽样的随机数只依赖 (target,B,draw)，所以所有方法和两个 seed 使用相同的 calibration/query 划分。表中数值：先对每个 target 在 8 次抽样 x 2 个 seed 上取平均，再对 20 个 target 取平均；± 是跨 target 的 t 分布 95% CI（n=20）。

target 端的 input z-score 有两种：`srcnorm` = 用 session1 的统计量（完全不用 target 数据）；`calnorm` = 用 B 个 calibration trial 的神经数据（无标签）算逐通道均值/标准差。

### 4.2 参照行

- (a) `reuse_s1_srcnorm` / `reuse_s1_calnorm`：同一受试者 session1 的表原样使用（前者是严格零 target 数据的自然基线，后者用 B 个 trial 的无标签统计量重新归一化）。
- (b) `mean_table_calnorm`：20 张 source 表的均值；`mean_loso_calnorm`：留出该受试者自己的 19 张均值（更干净的版本）。
- (c) `ui_prior_*`（**非部署参照**，反传穿过 frozen trunk，只更新 read-in table）：Adam，300 步，lr 3e-3，全 batch 使用 B 个 trial，固定值，没有在 target 上调；三个变体：从 s1 表初始化 + s1 统计量、从 s1 表初始化 + calnorm、从 mean 表初始化 + calnorm。
- (d) `within_table_s1init_long` / `within_table_supinit`（**仅诊断**，B=10：10 个 trial 训练 table，5 个 trial query；前者从 s1 表出发 1000 步，后者从 target 数据的 supervised 初始化出发 1000 步）。没有做"在 target 上从头训练整个 trunk"的 within-session 模型（未执行）。
- 部署型 closed-form 基线：`rownorm_ridge` = calnorm 的 z-score + 因果 EMA + 256->5 ridge（trunk 不参与）。lambda 与 EMA 系数在 source validation 上选一次（lam=1.0、ema=0.9，网格 5x5）；`rownorm_ridge_perB` 对每个 B 在 source session 内模拟 B-trial calibration 选超参数（审查后把 lambda 网格扩到 1e8 重跑；B=1 选 1e6、B=2 选 1e5，即接近常数预测；B=5,10 与全局相同）。
- 诊断（用到整个 target session 的无标签神经数据，不是候选）：`*_oracle_norm` = 用 target 全部 15 个 trial 的统计量归一化。

H 项：H1（trunk 哈希前后一致）所有 14080 行都满足；H5（表 <= 32 KB：K=16 fp16 为 8 KB）都满足；H2：reuse、mean、ridge 行 0 次反传，ui_prior 行 300 步反传、within 行 1000 步，所以这几类只是参照；H4：只用 B 个有标签 trial（oracle 行除外，已标注）。H3（固定权重乘加比例）这里没有计算，这些行只是基线，不是候选。

### 4.3 结果表（query R2，物理单位，均值 ± 95% CI，n=20 target）

| method | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|
| `reuse_s1_srcnorm`（零 target 数据） | -0.223 ± 0.342 | -0.223 ± 0.342 | -0.210 ± 0.336 | -0.231 ± 0.347 |
| `reuse_s1_calnorm` | 0.167 ± 0.167 | 0.416 ± 0.143 | 0.578 ± 0.115 | 0.583 ± 0.116 |
| `mean_table_calnorm` | -0.073 ± 0.249 | 0.208 ± 0.166 | 0.336 ± 0.113 | 0.347 ± 0.110 |
| `mean_loso_calnorm` | -0.132 ± 0.258 | 0.152 ± 0.175 | 0.286 ± 0.118 | 0.297 ± 0.115 |
| `rownorm_ridge`（部署型 closed-form） | -2.874 ± 1.686 | -3.275 ± 4.034 | 0.680 ± 0.135 | 0.780 ± 0.079 |
| `rownorm_ridge_perB`（部署型 closed-form） | -0.043 ± 0.013 | 0.013 ± 0.018 | 0.680 ± 0.135 | 0.780 ± 0.079 |
| `ui_prior_s1_srcnorm`（参照） | -0.197 ± 0.133 | 0.046 ± 0.153 | 0.701 ± 0.105 | 0.782 ± 0.085 |
| `ui_prior_s1_calnorm`（参照） | -0.175 ± 0.274 | 0.168 ± 0.196 | 0.729 ± 0.093 | 0.801 ± 0.072 |
| `ui_prior_mean_calnorm`（参照） | -0.374 ± 0.202 | -0.043 ± 0.186 | 0.731 ± 0.087 | 0.813 ± 0.063 |
| `within_table_s1init_long`（诊断） | - | - | - | 0.792 ± 0.074 |
| `within_table_supinit`（诊断） | - | - | - | 0.756 ± 0.082 |
| `reuse_s1_oracle_norm`（oracle 诊断） | 0.595 ± 0.113 | 0.596 ± 0.110 | 0.597 ± 0.110 | 0.592 ± 0.113 |
| `mean_table_oracle_norm`（oracle 诊断） | 0.344 ± 0.118 | 0.342 ± 0.118 | 0.347 ± 0.114 | 0.352 ± 0.108 |

配对差（同一 target 上相减，95% CI；全表见 `paired_diffs_main.csv`）：
- ui_prior_s1_calnorm - reuse_s1_calnorm：B=1 -0.34 ± 0.18，B=2 -0.25 ± 0.09，B=5 +0.15 ± 0.06，B=10 +0.22 ± 0.06。
- ui_prior_mean_calnorm - mean_table_calnorm：B=5 +0.39 ± 0.09，B=10 +0.47 ± 0.08。
- ui_prior_s1_calnorm - rownorm_ridge：B=5 +0.05 ± 0.05，B=10 +0.02 ± 0.02（不显著）。
- reuse_s1_calnorm - mean_table_calnorm：B=1 +0.24 ± 0.22，B=2 +0.21 ± 0.15，B=5 +0.24 ± 0.12，B=10 +0.24 ± 0.11。
- ui_prior 的初始化（s1 表 vs mean 表）：B=1 +0.20 ± 0.17、B=2 +0.21 ± 0.07 有利于 s1 表；B>=5 差异小于 0.012。
- 单位比值（预测/真值标准差）：reuse/mean 行 0.6-0.96，ridge 与 ui_prior 行 0.74-1.30，都在 [0.1, 10] 带内。

## 5. 跨天漂移有多大

1. 输入层（`drift_stats.json`，20 名受试者平均）：逐通道 log 平均包络的空间模式相关，跨天 0.60，同日内（session1 的 sample1+2 vs sample3）0.99；逐通道 log 增益比的标准差，跨天 0.25，同日内 0.04；整体增益几乎不变（log 比均值 -0.04）。所以漂移主要是逐通道的相对增益/空间模式变化，不是整体幅度变化。
2. 输出层：同一张 s1 表，不重新归一化，R2 = -0.22（完全失效）；用 B>=5 个 trial 的无标签统计量重新归一化后 0.58；整个 session2 的统计量（oracle）也只有 0.595。同一 trunk 下，用 10 个 trial 在 target 上拟合 table 最好约 0.80–0.81（B=10），source 同日验证为 0.84。即：跨天复用（重新归一化后）与 table 拟合之间有约 0.2 的缺口；**从 calnorm 出发**，逐通道统计量只能再吸收一小部分（oracle 统计量在 B=5 时 +0.019，B=10 时 +0.009）；**从零 target 数据复用（srcnorm）出发**，重新归一化是最大的一项，约 +0.8。
3. 受试者间差异很大：B=10 时 reuse_s1_calnorm 在各 target 上从 0.03 到 0.89，ui_prior_s1_calnorm 从 0.37 到 0.95（受试者 19、20 最差，约 0.4-0.6）。

## 6. 对校准方法的含义

- 空间/表结构漂移才是 B>=5 时的瓶颈：重新归一化后，reuse 的 0.58-0.60 与 ui_prior 的 0.73-0.80 之间有 0.15（B=5）到 0.22（B=10）的真实余量（配对 CI 不含 0）。这是 label 驱动的表更新方法能争取的区间。
- ui_prior 本身不比简单 ridge 好多少：B=5 +0.05、B=10 +0.02（均不显著）；**更正（审查）**：`within_table_s1init_long` 与 `ui_prior_s1_calnorm`（B=10）是同一过程（相同的 10 个 trial、s1 初始化、calnorm、学习率），只是步数从 300 换成 1000，不是独立的上限，二者一致是必然的。所以不能说“表-only 的上限约 0.80”，只能说“10 个 trial 下见到的最好 table 拟合约 0.80”（`ui_prior_mean_calnorm` 为 0.813，source 同日验证 0.84）。B=10 的 closed-form ridge 已经 0.78；有意义的比较对象是 reuse 与 mean 表。
- B=1、2：ui_prior 反而差于 reuse_s1_calnorm（-0.34、-0.25），梯度拟合在 1-2 个 trial 上过拟合，并且没见过的手指无法学到；ridge 更差（全局 lambda 为 -2.9、-3.3，这是真实的过拟合：B=1 时 z-score 只来自一个 trial，没发力的通道 std 很小，query 上 |z| 可达 40–80；per-B 调参并扩展网格后为 -0.04、0.01，接近常数预测，仍低于 reuse）。这里没有来自梯度拟合的余量；可争取的是统计量：同样的 reuse，用 oracle 统计量是 0.595，而 B=1 的 calnorm 只有 0.167、B=2 为 0.416。1-2 个 trial 只有一两根手指在发力，统计量偏，所以小 B 的瓶颈是统计量估计（需要朝 session1 统计量或群体统计量做可靠性加权的收缩），这与 P1 的 reliability shrinkage 设计一致。B>=5 时 calnorm 已与 oracle 统计量接近（0.578 vs 0.597）。
- mean 表显著差于同一受试者自己的 s1 表（0.35 vs 0.58，差 0.24 ± 0.11），留一版本更差（0.30）。受试者之间电极到 latent 的映射不共享，简单平均不是好先验。若 Q-D 要说明 population-conditioned 生成优于 independent encoder + shrinkage，对比对象要用 reuse 而不是 mean 表；H-A 里有同一受试者的 s1 表这个强先验，群体信息的价值需要靠跨受试者协议（本轮没有做）才能检验。
- 对比 B=5 与 B=10：reuse、mean 行几乎不变（0.578 vs 0.583），ui_prior 与 ridge 继续上升（0.73 -> 0.80，0.68 -> 0.78），说明增加 calibration trial 对基于数据的拟合有益，对复用表没有帮助。

## 7. 没做的事 / 局限

- 未做：ndof/random 数据集的管线与评测；target 端整模型（trunk+表）从头训练的 within-session 上限；对 SSM 选择 tau（只固定 80 ms）；H3 的乘加统计；GDrive 备份（结果很小，原始数据未移动）；未 push。
- ui_prior 超参数（300 步、lr 3e-3）固定，未调；within_table_s1init_long（1000 步）不优于 300 步的 ui_prior，说明不是步数不足。
- `preprocess` 的滤波器经审查确认为零相位（见第 1 节），待改为因果前端。
- 其他小问题（审查）：`ui_prior_mean_calnorm` 和 `mean_table_calnorm` 的初值 `mean_all` 含该受试者自己的 session1 表（权重 1/20），没有留一受试者的 ui_prior 变体；B=1 只有 15 种可能的 calibration 集合，8 次抽样平均约 6.3 种不同集合（只影响方差）；`h4_uses_labels` 恒为 False（对这些行恰好正确）；每行记录的 `fit_loss` 是最后一步更新之前的 loss。
- 抽样每 (target,B) 8 次、2 个 seed；CI 以 target 为单位，不包含 backbone 训练 seed 之间的方差之外的因素。
- 受试者 19、20 的结果拉低均值，没有排除任何 target。
- B<5 时 calibration 只含部分手指，所有方法在没见过的手指上都受限；这是协议使然，不是实现问题。`seen_unseen_main.csv` 给出已见/未见手指的 R2 拆分。

## 8. 文件与复现

代码：`ssm_decode/hyser_data.py`、`ssm_decode/hyser_backbone.py`；测试：`tests/test_hyser_data.py`（9 项：header 解析、样本数/checksum、物理单位、包络因果性与阶跃响应、力降采样、cache 往返）和 `tests/test_hyser_backbone.py`（5 项）；脚本 `scripts/run_hyser_{train,eval,oracle_norm,ridge_perB}.py`、`run_hyser_grid{,2}.sh`、`hyser_select_tau.py`、`hyser_drift_stats.py`、`summarize_hyser{,_selection}.py`。

本目录：`reference_rows_main.{csv,json}`（含 per-target 数值）、`reference_rows_seed{0,1}.*`、`paired_diffs_*.csv`、`seen_unseen_*.csv`、`backbone_selection.csv`、`drift_stats.json`、`h_flags_*.json`、`raw/`（逐行 rows 的 gz、tau/ridge 选择、训练历史）。checkpoint 与完整 rows 在 `results/hyser_20261015/`（gitignore）。

运行环境：`CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=SSM/.tools/mamba_deps:SSM python SSM/scripts/run_hyser_eval.py --ckpt results/hyser_20261015/final/seed0/best.pt --out results/hyser_20261015/eval_seed0 --seed-tag 0 --draws 8`。
