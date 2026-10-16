# sEMG 上的 fixed-weight forward calibrator（Q-D，Hyser H-A + NinaPro DB6 v2）

日期：2026-10-16。GPU 计算全部在 cuda:1。代码：`ssm_decode/semg_calib.py`（描述量、流式累加器、calibrator、cost model）、`ssm_decode/semg_calib_env.py`（Hyser episode 环境）、`ssm_decode/semg_calib_db6.py`（DB6 v2 contiguous-stream episode 环境）、`scripts/run_semg_calib.py`（train / eval）、`scripts/summarize_semg_calib.py`、`scripts/summarize_semg_calib_meta.py`、`scripts/semg_cost_report.py`、`tests/test_semg_calib.py`（21 项）。原始 rows 与 checkpoint 在 `results/semg_calib_20261016/`（gitignore）；本目录是压缩后的 csv/json。

**状态：** Hyser（2 个 backbone seed x 3 种 prior x 4 个变体 x 5 个 subject fold，每个 seed 7680 条 candidate row）和 DB6 v2（2 seed x 3 prior x 4 变体，记录顺序整段解码，每个 seed 18000 条 candidate row；打乱顺序的诊断只做了 seed 0、B in {4,16}）都已完成。**两处与计划不同，必须知道：（1）GPU 在 2026-10-16 06:00 左右失效（cuda:1 的 `torch.cuda.is_available()` 变为 False，`nvidia-smi` 报 GPU0 Unknown Error），所以全部 DB6 评测和 DB6 的 lr 扫描是在 CPU 上用 Mamba-3 的 CPU 端口（`mamba3_cpu.py`，解除 128 bin 窗口限制）完成的；CPU 解码已用 6000 个 (target,B,draw,方法) 对照存档的 GPU 参照行：平均偏差 −0.0007，平均绝对偏差 0.0022，最大 0.069，12% 的行偏差 > 0.005（集中在含近死通道的受试者 S4–S6、S10，bf16 舍入翻转个别 bin），不影响均值层面的比较。DB6 的 calibrator 训练（24 个）和 Hyser 全部在 GPU 上完成。（2）DB6 参照行（reuse/mean/ridge/ui_prior）直接取自 v2 报告的存档行（GPU 解码，同一抽样），没有重算。**

## 0. 结论摘要（Hyser，H-A；query R²，物理单位，macro 5 根手指；n=20 target，t-CI；2 个 backbone seed 平均）

最强的可部署零/低标签基线（按每个 B 在 target 均值上事后挑选，对基线有利）：B=1/2 为 `reuse_s1_calnorm`（0.167/0.416），B=5/10 为每个 B 在 source 上选超参数的闭式 ridge（0.680/0.780）。ui_prior（梯度穿过 frozen trunk，非部署）：0.729/0.801（s1 表起点）。

1. **用同一受试者 session1 表作 prior（`own`）的 calibrator 只在小 B 超过最好的可部署基线，且幅度有限。** P4（PMA m=8 + reliability）相对 `reuse_s1_calnorm`：B=1 +0.096 ± 0.074，B=2 +0.063 ± 0.040；P0 为 +0.058 ± 0.075 / +0.049 ± 0.030；B=5/10 只比 reuse 高 +0.02（0.604/0.602 对 0.578/0.583），但比闭式 ridge 低 0.076 ± 0.069 / 0.178 ± 0.054（0.680/0.780）。
2. **离 ui_prior 的距离：** B=1/2 反而高于 ui_prior_s1_calnorm（+0.44 ± 0.17 / +0.31 ± 0.09，那里梯度拟合因 1–2 个 trial 过拟合而失败，不是 calibrator 强）；B=5/10 低 0.125 ± 0.047 / 0.200 ± 0.059（0.604 对 0.729，0.602 对 0.801）。即 B>=5 时固定权重 calibrator 只拿到 ui_prior 相对 reuse 增益（+0.15/+0.22）的约 17%/9%。
3. **population mean prior（`mean`，跨受试者的 channel-aligned 均值表）：** 相对 `mean_loso_calnorm`（同一 prior 不校准），P4 +0.165 ± 0.118 / +0.084 ± 0.108 / +0.065 ± 0.097 / +0.048 ± 0.095（B=1/2/5/10）；但绝对值（0.034/0.236/0.351/0.345）远低于 reuse 与 ridge。Hyser 里 population 表不是好 prior（与第一轮结论一致）。
4. **identity-free 对照（零 prior，descriptor 里没有 prior 列和 prior-latent 相关）下 set pooling 有明显作用，已知身份时没有：**
   - identity-free：P2/P4 − P0 = +0.07/+0.10/+0.21/+0.23（P2，B=1/2/5/10）、+0.08/+0.12/+0.20/+0.20（P4），B=1–10 的 CI 均不含 0；P1 − P0 ≈ 0（−0.05 到 +0.04）；P4 ≈ P2（|差| <= 0.03）。P0 在 B>=5 时 R² 只有 0.02–0.05，P2/P4 到 0.21–0.26。
   - 已知身份（`own`）：|P4 − P0| <= 0.037（B=1 为 +0.037 ± 0.039，B>=5 时 <= 0.015），P2、P1 同样；pooling 没有增益。
   - `mean` prior：P4 − P0 = +0.10 ± 0.08（B=1,5,10），边界显著。
   - 但 identity-free 的绝对水平仍很低（B=10：P2 0.247、P4 0.215，对 reuse 0.583、ridge 0.780），所以 “pooling 在身份缺失时有用” 不等于 “身份缺失时可用”。
5. **两个 backbone seed 的 spread：** `own` 变体 <= 0.035（B=1 的 P1 为 0.135）；`mean` 与 identity-free 的 P0/P1 高达 0.07–0.18，P2/P4 只有 0.003–0.037。pooling 变体在 seed 间更稳定。
6. 以 source validation 选变体（各 prior 下 source-val 均值最高的变体，Hyser 全部为 P2）：`own`/P2 为 0.212/0.446/0.599/0.600，与 P4 相差 <= 0.05（B=1）。source validation 在 `own`/`mean` 下对变体几乎没有区分度（P2/P4 比 P0/P1 高 0.01–0.03），在 identity-free 下则明确区分（P2 0.270、P4 0.245 对 P0 0.072、P1 0.103），与 target 上的排序一致。

约束检查：Hyser 全部 15360 条 candidate row 满足 H1（trunk 哈希不变）、H2（校准在 `no_grad` 下，0 次反传）、H4（描述量只用 B 个校准 trial：一次测试把非校准 trial 换成噪声后 stats 逐位相同）、H5（静态表 K x N fp16 = 8 KB <= 32 KB）。

### 0.1 DB6 v2（D-A，记录顺序整段 contiguous 解码；per-bin balanced accuracy；n=10 受试者，t-CI；2 个 backbone seed 平均）

最强的可部署零/低标签基线（事后按 B 在受试者均值上挑选）：B=1/2/4 为 `mean_table_calnorm`（0.428/0.477/0.511），B=8 为 `reuse_latest_calnorm`（0.541），B=16 为闭式 `rownorm_ridge`（0.578）。ui_prior_latest_calnorm（非部署）：0.355/0.445/0.553/0.666/0.724（B=1/2/4/8/16）。

1. **没有任何 calibrator 在任何 B 上以 CI 排除 0 的幅度超过最强可部署基线。** 最接近的是 `mean` prior 的 P0：B=4 +0.022 ± 0.027，B=8 +0.035 ± 0.067，B=16 +0.003 ± 0.022；B=1/2 为 −0.020 ± 0.021 / −0.002 ± 0.024。identity-free 在 B=1/2 显著更差（−0.08 到 −0.12，CI 约 ±0.03），B=16 持平（P2/P4：+0.002 / −0.000 ± 0.025）。
2. **相对 “同一个 prior 不校准”，只有 population prior 的 calibrator 有增益：** `mean` prior 的 P0 − `mean_loso_calnorm` = −0.004 ± 0.024 / +0.016 ± 0.030 / +0.043 ± 0.034 / +0.065 ± 0.034 / +0.071 ± 0.034（B=1/2/4/8/16）。**`own`（最新 source session 的表）prior 的 calibrator 基本等于 prior 本身：** P0 − `reuse_latest_calnorm` = −0.009 ± 0.011 … +0.011 ± 0.022（全部 B）。这与 source-side 事实一致：`own` calibrator 的 source validation 只从 prior-only 的 0.503 提到 0.508–0.514（best step 250–900），lr 降到 3e-4、1e-4（seed 0，P0/P4，CPU，只看 source val）仍是 0.505–0.506，所以不是学习率问题；在 19 个训练 session（早期 session 作 prior）上，固定权重 calibrator 学不到跨 session 的表修正。
3. **离 ui_prior 的距离：** B=1 时 `mean`/`own` calibrator 高于 ui_prior（+0.03 到 +0.055，那里 ui_prior 因类覆盖不足而失败），identity-free 低 0.03–0.04；B=4 低 0.02–0.04（identity-free 低 0.07–0.10），B=8 低 0.09–0.13，B=16 低 0.14–0.17（0.58 对 0.724）。B=16 时最好的 calibrator（`mean`/P0 0.581、identity-free P2 0.579）只拿到 ui_prior 相对 reuse 的增益（+0.176）的约 19%/18%。B>=8 的 ui_prior 增益需要标签驱动的逐 session 梯度拟合；固定权重前向目前拿不到。
4. **identity-free（零 prior，无 prior 列）下 pooling 有小而可靠的作用，已知身份时没有：** P2/P4 − P0 = +0.015 / +0.019 / +0.020 / +0.024 / +0.021（P2，B=1/2/4/8/16，CI 约 ±0.008–0.010，B>=1 均不含 0）；P4 ≈ P2（|差| <= 0.004）；P1 − P0 在 ±0.01 以内。`mean` 或 `own` prior 下 |P2/P4 − P0| <= 0.007（不显著），`mean` 下 P1 − P0 为 −0.008 到 −0.014（略负）。方向与 Hyser 一致，幅度小一个量级。identity-free 在小 B（1–4）显著差于带身份的变体（B=4：0.48 对 0.53/0.51），在 B>=8 不低于 ridge 与 reuse（B=8：0.556–0.557 对 0.514/0.541；B=16：0.578–0.579 对 0.578/0.548）。
5. **backbone-seed spread：** calibrator 行 0.000–0.046（多数 <= 0.03；`mean`/P0 在 B=1 最大 0.046），`reuse_latest_calnorm` 0.005–0.010，`ui_prior` <= 0.006。pooling 变体（P2/P4）的 seed spread 一般小于 P0/P1。
6. **打乱顺序诊断（seed 0，B in {4,16}，每格 3 个 draw，10 名受试者；`db6_shuffled_*`）与记录顺序一致：** `own` 0.517/0.557（reuse 0.518/0.558），`mean`/P0 0.530/0.586（mean_table 0.504/0.518），identity-free P2/P4 0.480/0.590 与 0.491/0.590（ridge 0.379/0.578）；与记录顺序的差别 <= 0.015（样本不同：seed 0、3 个 draw），没有 block 顺序捷径的迹象。
7. 约束检查：DB6 全部 36000 + 5064 条 candidate row 满足 H1/H2/H4/H5（描述量与归一化只用 B 个校准 trial，224 B 表）。

## 1. 方法（与第一轮 000688 的关系、偏离处）

**协议：** 与现有 sEMG 参照行完全相同：Hyser H-A（20 个 target = 每名受试者的 session2，B in {1,2,5,10}，每个 (target,B) 8 次平衡抽样——随机数种子 `[7919,t,B,draw]` 与参照一致，已验证 `cal_trials` 逐条相同；query = 其余 trial）。backbone 是已有的冻结 K=16 w128 模型，2 个 seed。**流水线等价检查：** 用本模块的流式 B-trial 统计量 + `reuse_s1` 表重新算出的 R² 与已存的 `reuse_s1_calnorm` 行最大绝对差为 0（`--check`，20 target x 4 B x 8 draw x 2 seed）。

**前端注意：** Hyser `preprocess` 的 EMG 经审查是零相位（非因果）滤波；本轮为与已有参照行可比，仍使用现有 cache（80 ms 泄漏积分器的包络，从 `preprocess` 文件算出），因此 “前端因果” 对 Hyser 不成立，结论里的数字不应当作严格流式结果。

**描述量（每个通道一个 unit；全部由 B 个校准 trial 的可加累加器得到，`SemgStats`）：** log 平均包络、log std、lag 1/3/10 自相关、与 prior 潜变量（K=16）的相关（用 N x N Gram）、与全体通道平均活动的相关、与 5 根手指力在 lag 0/2/4/6 的归一化互协方差（20 维）、标签均值、prior 列（K 维）、log n、log B；共 65 维（DB6：40 维，标签块为 8 类 one-hot 的 lag-0 互协方差）。可靠性特征 5 维：log n、log B、log 水平、trial 间包络 CV²、标签互协方差跨 trial 一致性。输入归一化的均值/std 也来自同一批累加器（std 下限 1e-3 x mean std，与 `hyser_backbone.input_stats` 相同）。

**变体：** P0 独立 per-channel 编码器；P1 = P0 + `d~ = lam d + (1-lam) mu0`，lam = sigmoid(MLP(可靠性特征))；P2 = P1 + DeepSets 上下文 `mean_i psi(d~_i)`；P4 = PMA（m=8 个固定 seed，可 online-softmax）+ 打分里加 log lam。（`semg_calib.py` 是 `population_calib.py` 的独立副本，P3 与 P4 在 000688 上相同，故未单独做 P3。）生成器 2 层宽 64 MLP，输出层零初始化，表 = prior + out_scale x 输出，out_scale = source 表元素的 RMS（固定常数），所以训练起点就是 “prior + B-trial 归一化”。

**prior 的三种构造（训练与评测用同一函数）：**
- `mean`：除目标受试者外其余受试者 session1 表的 channel-aligned 均值（训练时再去掉验证和 held-out fold 的受试者）。
- `own`：目标受试者自己的 session1 表。**Hyser 每个受试者只有一个 source session，所以训练里无法得到 “同一受试者前一天的表”。** 为避免 calibrator 在 “自己的表就是自己的 target” 的 episode 上学成恒等映射，训练使用 stale 替身：prior = (1−rho) T_own + rho T_mean(其他训练受试者) + sigma x out_scale x 噪声，rho~U(0,0.7)，sigma~U(0,0.6)（事先固定，没有调）；并且采用 **subject-fold 训练**：5 个 fold（subject i 属于 fold (i−1) mod 5），fold f 的 calibrator 训练时完全看不到 fold f 的 4 个受试者（数据、表、prior 都不用），再留 3 个受试者做 validation；评测 target t 用 fold(t) 的 calibrator。target 的 session2 从未被训练代码加载。这个 stale 替身是对真实跨天漂移的假设性代理，**没有验证它的分布与真实 session1→session2 漂移相近**（见局限）。
- `free`（identity-free 对照）：prior = 0，descriptor 中 prior 列和 prior-latent 相关置 0（实现为固定 mask；单元测试确认生成器输出与它们无关），其余描述量不变。`mean` 和 `free` 同样用 fold 训练，保证三种 prior 的训练数据量一致（每个 fold 13 个训练受试者 + 3 个 validation 受试者）。

**episode 训练（只用 source session1）：** 随机训练受试者；B 取 log-uniform 整数 1..10；校准 trial 在该受试者的训练 trial（sample 1,2）里按手指均衡抽取；query = 该 session 的 4 个留出 trial（sample 3，backbone 从未见过其标签）；loss = 冻结 trunk 输出的 MSE（梯度只回到 calibrator）。AdamW lr 1e-3，batch 8 episode，每 100 步在 3 个 validation 受试者上评 source-val（B in {1,2,5,10} x 3 draws，query = sample 3 的 5 个 trial，own 模式的 validation prior 取三档固定 stale 强度），patience 6 次、至少 400 步、上限 4000 步。选择、早停全部只看 source validation。全部 120 个 run 中 119 个满足 “best step <= 80% 预算”（1 个 own/P4 未满足，最优步靠近末端；没有再加预算）。calibrator 参数量 P0 9,280 / P1 9,441 / P2 17,697 / P4 22,881。

**目标端：** 校准 = 流式累加 B 个 trial 的可加和 -> 固定权重前向（`torch.no_grad`，`calibrate()` 在 grad 开启时也会强制进入 no_grad）-> 静态 16 x 256 表；没有任何目标端梯度。单元测试覆盖：通道置换等变性（P0/P1/P2/P4 x aligned/free）、流式分块累加 == 批量、trial 顺序/合并精确、no-grad 守卫、prior 构造不含目标/held-out 数据、描述量只依赖请求的 trial（把其他 trial 换成噪声后 stats 逐位相同）、`input_stats` 一致、cal 抽样与参照行一致、cost model 自洽。

## 2. Hyser 结果

### 2.1 Query R²（均值 ± 95% CI，n=20 target，2 seed 平均；参照行复用已有行：协议逐条相同）

| method | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|
| `reuse_s1_srcnorm`（零 target 数据） | -0.223 ± 0.342 | -0.223 ± 0.342 | -0.210 ± 0.336 | -0.231 ± 0.347 |
| `reuse_s1_calnorm`（reuse + B-trial 统计量） | 0.167 ± 0.167 | 0.416 ± 0.143 | 0.578 ± 0.115 | 0.583 ± 0.116 |
| `mean_table_calnorm` | -0.073 ± 0.249 | 0.208 ± 0.166 | 0.336 ± 0.113 | 0.347 ± 0.110 |
| `mean_loso_calnorm`（去掉自己的 mean 表） | -0.132 ± 0.258 | 0.152 ± 0.175 | 0.286 ± 0.118 | 0.297 ± 0.115 |
| `rownorm_ridge_perB`（闭式 ridge，每个 B 在 source 上选） | -0.043 ± 0.013 | 0.013 ± 0.018 | 0.680 ± 0.135 | 0.780 ± 0.079 |
| `ui_prior_s1_calnorm`（非部署） | -0.175 ± 0.274 | 0.168 ± 0.196 | 0.729 ± 0.093 | 0.801 ± 0.072 |
| `ui_prior_mean_calnorm`（非部署） | -0.374 ± 0.202 | -0.043 ± 0.186 | 0.731 ± 0.087 | 0.813 ± 0.063 |
| **prior = own（session1 表）** | | | | |
| `cal_p0_own` | 0.225 ± 0.189 | 0.464 ± 0.145 | 0.602 ± 0.111 | 0.587 ± 0.127 |
| `cal_p1_own` | 0.160 ± 0.261 | 0.425 ± 0.168 | 0.594 ± 0.120 | 0.584 ± 0.135 |
| `cal_p2_own` | 0.212 ± 0.211 | 0.446 ± 0.153 | 0.599 ± 0.113 | 0.600 ± 0.118 |
| `cal_p4_own` | 0.262 ± 0.168 | 0.479 ± 0.134 | 0.604 ± 0.112 | 0.602 ± 0.120 |
| **prior = mean（population 表）** | | | | |
| `cal_p0_mean` | -0.069 ± 0.201 | 0.156 ± 0.134 | 0.249 ± 0.140 | 0.242 ± 0.140 |
| `cal_p1_mean` | -0.075 ± 0.198 | 0.123 ± 0.148 | 0.223 ± 0.165 | 0.220 ± 0.166 |
| `cal_p2_mean` | 0.010 ± 0.180 | 0.211 ± 0.132 | 0.328 ± 0.123 | 0.316 ± 0.122 |
| `cal_p4_mean` | 0.034 ± 0.170 | 0.236 ± 0.125 | 0.351 ± 0.114 | 0.345 ± 0.108 |
| **identity-free（零 prior，无 prior 列）** | | | | |
| `cal_p0_free` | -0.122 ± 0.070 | -0.063 ± 0.068 | 0.050 ± 0.106 | 0.016 ± 0.140 |
| `cal_p1_free` | -0.083 ± 0.055 | -0.075 ± 0.096 | 0.027 ± 0.126 | -0.029 ± 0.182 |
| `cal_p2_free` | -0.050 ± 0.073 | 0.041 ± 0.093 | 0.257 ± 0.107 | 0.247 ± 0.116 |
| `cal_p4_free` | -0.040 ± 0.061 | 0.054 ± 0.075 | 0.248 ± 0.089 | 0.215 ± 0.103 |

（CI 以 target 为单位；同一 target 内 8 draw x 2 seed 先取平均，所以 CI 不含 backbone 训练 seed 的方差——seed spread 见 2.4。）

### 2.2 相对基线与 ui_prior 的配对差（a − b，n=20 target）

| a | b | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|---|
| `cal_p4_own` | best deployable（B=1,2: reuse；B=5,10: ridge） | +0.096 ± 0.074 | +0.063 ± 0.040 | -0.076 ± 0.069 | -0.178 ± 0.054 |
| `cal_p0_own` | best deployable | +0.058 ± 0.075 | +0.049 ± 0.030 | -0.078 ± 0.076 | -0.193 ± 0.063 |
| `cal_p2_own` | best deployable | +0.045 ± 0.117 | +0.031 ± 0.042 | -0.081 ± 0.074 | -0.180 ± 0.052 |
| `cal_p4_own` | `reuse_s1_calnorm` | +0.096 ± 0.074 | +0.063 ± 0.040 | +0.026 ± 0.041 | +0.019 ± 0.047 |
| `cal_p4_own` | `ui_prior_s1_calnorm` | +0.437 ± 0.167 | +0.311 ± 0.092 | -0.125 ± 0.047 | -0.200 ± 0.059 |
| `cal_p4_mean` | `mean_loso_calnorm` | +0.165 ± 0.118 | +0.084 ± 0.108 | +0.065 ± 0.097 | +0.048 ± 0.095 |
| `cal_p4_mean` | best deployable | -0.133 ± 0.164 | -0.180 ± 0.125 | -0.329 ± 0.116 | -0.435 ± 0.082 |
| `cal_p4_free` | best deployable | -0.206 ± 0.126 | -0.362 ± 0.107 | -0.432 ± 0.101 | -0.565 ± 0.078 |

“best deployable” 是事后在 target 均值上挑选的（对基线有利，是保守比较）。全部配对差见 `hyser_paired.csv`。按 target 看：`cal_p4_own` 在 B=1/2/5/10 比 reuse 好的 target 占 90%/75%/70%/65%，最差的 target 分别 −0.455/−0.112/−0.183/−0.200。

### 2.3 变体之间（pooling 问题；配对差 a − b，n=20 target，完整见 `hyser_variant_pairs.csv`）

| prior | a − b | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|---|
| free | P1 − P0 | +0.040 ± 0.035 | -0.012 ± 0.051 | -0.023 ± 0.034 | -0.045 ± 0.061 |
| free | P2 − P0 | +0.073 ± 0.057 | +0.104 ± 0.067 | +0.207 ± 0.077 | +0.230 ± 0.104 |
| free | P4 − P0 | +0.083 ± 0.037 | +0.117 ± 0.054 | +0.199 ± 0.079 | +0.198 ± 0.112 |
| free | P4 − P2 | +0.010 ± 0.030 | +0.013 ± 0.035 | -0.009 ± 0.053 | -0.032 ± 0.059 |
| mean | P2 − P0 | +0.079 ± 0.075 | +0.055 ± 0.051 | +0.079 ± 0.058 | +0.074 ± 0.060 |
| mean | P4 − P0 | +0.103 ± 0.083 | +0.080 ± 0.067 | +0.103 ± 0.081 | +0.102 ± 0.080 |
| mean | P4 − P2 | +0.024 ± 0.059 | +0.025 ± 0.044 | +0.023 ± 0.056 | +0.029 ± 0.051 |
| own | P1 − P0 | -0.064 ± 0.120 | -0.040 ± 0.048 | -0.008 ± 0.024 | -0.003 ± 0.030 |
| own | P2 − P0 | -0.013 ± 0.046 | -0.018 ± 0.023 | -0.003 ± 0.011 | +0.013 ± 0.021 |
| own | P4 − P0 | +0.037 ± 0.039 | +0.015 ± 0.023 | +0.002 ± 0.015 | +0.015 ± 0.022 |
| own | P4 − P2 | +0.051 ± 0.063 | +0.033 ± 0.030 | +0.005 ± 0.012 | +0.002 ± 0.013 |

解读：set pooling（P2/P4）的价值随 “每个通道自带的身份信息” 单调下降：identity-free 时 +0.2（B>=5），population prior 时 +0.1，own prior 时 ≈ 0。P1（reliability shrinkage）在三种设置下都没有可靠增益。P4 相对 P2 没有超出噪声的增益（reliability-weighted PMA 与 DeepSets 等价）。这与 000688 上 “已知 unit-to-electrode 映射时 pooling 无增益” 一致，也支持 “身份被隐藏时 pooling 才起作用”。

### 2.4 backbone seed 的 spread（两个 seed 的 target 均值之差的绝对值；`hyser_seed_table.csv`）

`own`：P0 0.004–0.025、P2 0.022–0.035、P4 0.006–0.033、P1 0.024–0.135；`mean`：P0 0.07–0.115、P1 0.14–0.18、P2 0.003–0.033、P4 0.021–0.037；`free`：P0 0.017–0.12、P1 0.018–0.092、P2 0.008–0.036、P4 0.003–0.020。对照：`reuse_s1_calnorm` 在 B=1/2 的 seed spread 为 0.081/0.056（B>=5 <= 0.004），`ui_prior_s1_calnorm` <= 0.047。B=1 的 P4−reuse 增益（+0.096）与 reuse 自己的 seed spread（0.081）同量级，所以 B=1 的结论对 backbone seed 敏感。

## 3. 成本：校准通路的 MAC / SRAM（解析，按实际 tensor 统计，`cost_model.{json,csv}`）

| 情形（B 个 trial） | 累加 MAC（B 个 trial 流式） | 累加 MAC/bin | 终结 MAC | 固定权重 MAC（P0 / P4） | 固定权重占比（P0 / P4） | calibrator 权重 | 累加器 SRAM（fp32，全 Gram / 上三角） | 输出表 |
|---|---|---|---|---|---|---|---|---|
| Hyser N=256, K=16, 1250 bin/trial, B=1 | 9.06e7 | 72,448 | 1.31e6 | 2.38e6 / 5.91e6 | 2.5% / 6.0% | 9,280 / 22,881 | 391 KB / 261 KB | 8 KB（fp16） |
| Hyser B=10 | 9.06e8 | 72,448 | 1.31e6 | 同上 | 0.26% / 0.65% | 同上 | 同上 | 8 KB |
| DB6 N=14, K=8, 424 bin/trial, B=1 | 1.72e5 | 406 | 3.1e3 | 1.00e5 / 2.71e5 | 36% / 61% | 7,168 / 19,144 | 6.2 KB / 5.8 KB | 224 B |
| DB6 B=16 | 2.75e6 | 406 | 3.1e3 | 同上 | 3.5% / 9.0% | 同上 | 同上 | 224 B |

要点：（i）累加通路由 N x N Gram（用于 prior-latent 相关和全局活动相关）主导：N=256 时 65.5k MAC/bin 占 90%，持续 3.6 MMAC/s（50 Hz），固定权重 calibrator 只占 0.3–6%；（ii）**N=256 的累加器 SRAM 约 391 KB（fp32，全 Gram；上三角 261 KB）远超 32 KB**，H5 的 32 KB 约束目前只对输出表成立，对描述量累加器不成立——这是一个要诚实写出的硬件问题；（iii）解析估计的 Gram-free 替代（用固定参考归一化后的 prior-latent 投影累加 N x K，而不是 N x N Gram）约 16.1 kMAC/bin、61.6 KB 累加器（DB6 为 490 MAC/bin、2.6 KB），**但没有实现也没有评测，不知道精度是否保持**；（iv）DB6 N=14 的 SRAM 只有 6 KB，累加不是瓶颈，固定权重前向占 3.5–61%。

## 4. DB6（D-A，v2 contiguous 解码）

### 4.1 实现与协议

- 整段 session 是一条连续因果流，状态跨 trial 携带；query 只在非校准 trial 的 bin 上打分；协议、抽样（随机数 `[7919, subject, day*10+tod, B, draw]`，5 个 draw，B in {1,2,4,8,16}，60 个 target session）、聚合（draw -> tod -> seed -> 受试者 -> 3 个 gap 平均，t-CI n=10）与 v2 参照行逐条相同；backbone 为 v2 的 `final/seed{0,1}`（K=8 w128，在 trial 顺序打乱的流上训练）。
- B-trial 归一化：均值/std 来自 B 个校准 trial，std 下限 `0.01 x median_c std`（v2 在 source 上选出的值，近零 std 通道不会被放大成单位方差噪声；本模块的 `norm_stats(..., floor_ref="median", floor=0.01)` 与 `db6_backbone.input_stats` 同规则，由 `--check-ref` 行验证）。
- calibrator 训练（`semg_calib_db6.py`，24 个：2 seed x 3 prior x 4 变体）：episode = 某个 source session 的 trial 顺序打乱后拼成的流（每个 session 6 个固定排列副本）里随机偏移的 512–1024 bin 窗口，前 128 bin burn-in；校准 trial 是训练重复（class-balanced，B log-uniform 1..20），loss 是窗口内留出重复（backbone 从未用过其标签）的 bin 上的 CE；梯度只到 calibrator。source validation = v2 的固定打乱顺序留出流，10 个受试者各取最新的 source session 作为验证 session（其数据只出现在验证里；`own` 验证用它的前一个 session 表），B in {1,2,4,8,16} x 2 个 draw。AdamW lr 1e-3（`own` 另扫 3e-4、1e-4），patience 6、至少 400 步、上限 4000 步。没有 subject fold：DB6 每个受试者有多个 source session，`own` prior 在训练里就是同一受试者**更早的**一个 source session 的表（不用目标 session 自己的表，测试时用最新 source session 的表），所以不需要 Hyser 里的 stale 替身；`mean` = 其余受试者 source 表的均值（训练时再去掉验证 session 的表，测试与 `mean_loso_calnorm` 相同）。
- source-side 结果（`training_summary.csv`）：source validation（打乱顺序 balanced accuracy）prior-only -> 训练后：`mean` 0.440 -> 0.545–0.553；`free` 0.127 -> 0.460–0.476；`own` 0.503 -> 0.508–0.514；lr 扫描（seed 0）`own` 0.504 -> 0.505–0.506。

### 4.2 结果（记录顺序；balanced accuracy，均值 ± 95% CI，n=10 受试者，all gaps，2 seed 平均；参照行取自 v2 存档）

| method | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| `reuse_latest_srcnorm`（零 target 数据） | 0.407 ± 0.091 | 0.407 ± 0.091 | 0.407 ± 0.092 | 0.407 ± 0.092 | 0.407 ± 0.092 |
| `reuse_latest_calnorm`（reuse + B-trial 统计量） | 0.396 ± 0.062 | 0.457 ± 0.076 | 0.508 ± 0.086 | 0.541 ± 0.092 | 0.548 ± 0.093 |
| `mean_table_calnorm` | 0.428 ± 0.059 | 0.477 ± 0.056 | 0.511 ± 0.051 | 0.533 ± 0.053 | 0.532 ± 0.053 |
| `mean_loso_calnorm` | 0.413 ± 0.058 | 0.459 ± 0.055 | 0.490 ± 0.050 | 0.511 ± 0.051 | 0.510 ± 0.052 |
| `rownorm_ridge`（闭式，block-free 选择） | 0.201 ± 0.014 | 0.277 ± 0.020 | 0.381 ± 0.029 | 0.514 ± 0.049 | 0.578 ± 0.047 |
| `ui_prior_latest_calnorm`（非部署） | 0.355 ± 0.047 | 0.445 ± 0.065 | 0.553 ± 0.067 | 0.666 ± 0.073 | 0.724 ± 0.073 |
| `ui_prior_mean_calnorm`（非部署） | 0.346 ± 0.037 | 0.437 ± 0.045 | 0.549 ± 0.054 | 0.673 ± 0.068 | 0.730 ± 0.068 |
| **prior = own（最新 source session 表）** | | | | | |
| `cal_p0_own` | 0.386 ± 0.056 | 0.455 ± 0.071 | 0.512 ± 0.075 | 0.551 ± 0.080 | 0.557 ± 0.081 |
| `cal_p1_own` | 0.390 ± 0.063 | 0.456 ± 0.075 | 0.510 ± 0.081 | 0.546 ± 0.087 | 0.552 ± 0.089 |
| `cal_p2_own` | 0.387 ± 0.060 | 0.452 ± 0.074 | 0.509 ± 0.078 | 0.546 ± 0.083 | 0.552 ± 0.084 |
| `cal_p4_own` | 0.387 ± 0.060 | 0.454 ± 0.074 | 0.510 ± 0.079 | 0.548 ± 0.084 | 0.553 ± 0.087 |
| **prior = mean（其余受试者的均值表）** | | | | | |
| `cal_p0_mean` | 0.409 ± 0.050 | 0.475 ± 0.050 | 0.533 ± 0.052 | 0.575 ± 0.057 | 0.581 ± 0.060 |
| `cal_p1_mean` | 0.401 ± 0.051 | 0.462 ± 0.046 | 0.519 ± 0.046 | 0.563 ± 0.053 | 0.568 ± 0.053 |
| `cal_p2_mean` | 0.405 ± 0.053 | 0.469 ± 0.049 | 0.526 ± 0.049 | 0.570 ± 0.054 | 0.576 ± 0.057 |
| `cal_p4_mean` | 0.410 ± 0.051 | 0.471 ± 0.048 | 0.526 ± 0.052 | 0.569 ± 0.057 | 0.574 ± 0.060 |
| **identity-free（零 prior，无 prior 列）** | | | | | |
| `cal_p0_free` | 0.311 ± 0.040 | 0.373 ± 0.047 | 0.457 ± 0.054 | 0.532 ± 0.059 | 0.558 ± 0.057 |
| `cal_p1_free` | 0.308 ± 0.042 | 0.376 ± 0.046 | 0.462 ± 0.052 | 0.542 ± 0.057 | 0.568 ± 0.057 |
| `cal_p2_free` | 0.326 ± 0.042 | 0.393 ± 0.046 | 0.477 ± 0.053 | 0.556 ± 0.057 | 0.579 ± 0.057 |
| `cal_p4_free` | 0.323 ± 0.043 | 0.394 ± 0.046 | 0.480 ± 0.050 | 0.557 ± 0.054 | 0.578 ± 0.052 |

macro-F1 与 balanced accuracy 同序（`db6_original_table_macro_f1.csv`：例如 B=16 `cal_p0_mean` 0.569、`cal_p2_free` 0.565、ridge 0.578、ui_prior_latest 0.719、reuse 0.540）。按 gap（B=16，`db6_original_gap_table_bal_acc.csv`）：`cal_p4_own` 0.579/0.574/0.507（gap 1/2/3，与 reuse 0.582/0.568/0.494 同形，随 gap 下降），`cal_p0_mean` 0.593/0.594/0.557，`cal_p2_free` 0.580/0.601/0.557，ridge 0.568/0.583/0.582，ui_prior_latest 0.729/0.737/0.706。

### 4.3 配对差（a − b，n=10 受试者，95% CI；全表见 `db6_original_paired.csv`）

| a | b | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|---|
| `cal_p0_mean` | best deployable | -0.020 ± 0.021 | -0.002 ± 0.024 | +0.022 ± 0.027 | +0.035 ± 0.067 | +0.003 ± 0.022 |
| `cal_p4_mean` | best deployable | -0.018 ± 0.019 | -0.006 ± 0.023 | +0.015 ± 0.028 | +0.028 ± 0.069 | -0.004 ± 0.018 |
| `cal_p4_own` | best deployable | -0.041 ± 0.049 | -0.023 ± 0.056 | -0.000 ± 0.062 | +0.007 ± 0.024 | -0.024 ± 0.065 |
| `cal_p2_free` | best deployable | -0.102 ± 0.030 | -0.085 ± 0.026 | -0.034 ± 0.023 | +0.016 ± 0.072 | +0.002 ± 0.024 |
| `cal_p4_free` | best deployable | -0.106 ± 0.029 | -0.083 ± 0.028 | -0.031 ± 0.027 | +0.016 ± 0.075 | -0.000 ± 0.025 |
| `cal_p0_mean` | `mean_loso_calnorm` | -0.004 ± 0.024 | +0.016 ± 0.030 | +0.043 ± 0.034 | +0.065 ± 0.034 | +0.071 ± 0.034 |
| `cal_p4_mean` | `mean_loso_calnorm` | -0.002 ± 0.022 | +0.012 ± 0.028 | +0.036 ± 0.035 | +0.058 ± 0.035 | +0.064 ± 0.036 |
| `cal_p4_own` | `reuse_latest_calnorm` | -0.009 ± 0.010 | -0.003 ± 0.012 | +0.002 ± 0.017 | +0.007 ± 0.024 | +0.005 ± 0.024 |
| `cal_p0_mean` | `ui_prior_latest_calnorm` | +0.053 ± 0.026 | +0.030 ± 0.033 | -0.020 ± 0.034 | -0.090 ± 0.029 | -0.143 ± 0.029 |
| `cal_p4_own` | `ui_prior_latest_calnorm` | +0.031 ± 0.021 | +0.009 ± 0.024 | -0.042 ± 0.033 | -0.118 ± 0.047 | -0.171 ± 0.050 |
| `cal_p4_free` | `ui_prior_latest_calnorm` | -0.033 ± 0.034 | -0.051 ± 0.048 | -0.073 ± 0.050 | -0.109 ± 0.041 | -0.147 ± 0.040 |

best deployable（事后按 B 在受试者均值上挑）：B=1/2/4 `mean_table_calnorm`，B=8 `reuse_latest_calnorm`，B=16 `rownorm_ridge`。

变体之间（`db6_original_variant_pairs.csv`）：identity-free 下 P2 − P0 = +0.015 ± 0.008 / +0.019 ± 0.008 / +0.020 ± 0.009 / +0.024 ± 0.010 / +0.021 ± 0.010，P4 − P0 = +0.012 ± 0.012 … +0.025 ± 0.012，P4 − P2 在 ±0.004 内，P1 − P0 在 −0.002 … +0.010 内；`mean`：P2 − P0 = −0.003 … −0.007、P4 − P0 = +0.002 … −0.007（均不显著），P1 − P0 = −0.008 … −0.014；`own`：|P_x − P0| <= 0.005。

### 4.4 解读

- DB6 上固定权重 calibrator 的实际价值很低：`own` prior（最新 source session 表）下 calibrator 退化为 prior 本身（source validation 无提升，lr 扫描也无提升）；`mean` prior 下 calibrator 比 “未校准的 mean 表” 好 0.04–0.07（B>=4），但仍只是追平 reuse/ridge，不超过；identity-free 下追平 ridge/reuse（B>=8）但小 B 很差。三者在 B=16 都约在 0.56–0.58，离 ui_prior 的 0.72 还有 0.14–0.17。
- 这与 Hyser 的结论方向一致：已知身份且 prior 强（`own`）时固定权重 calibrator 只在最小 B 有一点点收益（Hyser）或没有收益（DB6），prior 弱/缺失时（`mean`、identity-free）calibrator 能显著提升，但提升到的水平仍低于最强的可部署基线或持平。DB6 的 `own` prior 弱于 Hyser（同日 T1/T2 的漂移已与跨天同量级，见 v2 报告），而 calibrator 训练样本只有 19 个 session 级对。
- pooling：identity-free 下稳定正向但很小（+0.02），已知身份时为 0；这与 000688 的 “已知映射时 pooling 无增益” 一致，也支持 “pooling 的价值来自缺失的身份信息”。

## 5. 局限与未做

- **GPU 失效后的 CPU 回退（DB6 评测与 lr 扫描）：** CPU 端口与 Triton kernel 在个别行上差 0.02–0.07（bf16 舍入翻转，集中在含近死通道的受试者），均值偏差 −0.0007；参照行（reuse/mean/ridge/ui_prior）取自 GPU 存档，calibrator 行取自 CPU，所以 “calibrator − 参照” 的差里含这个 CPU/GPU 噪声（量级 0.002 的均值绝对偏差，对 0.02–0.07 的效应量可忽略）。已用 `chk_*` 行把同一流水线重算的 `reuse_latest_calnorm`/`mean_table_calnorm` 与存档对照。
- 打乱顺序的 DB6 诊断只做了 seed 0、B in {4,16}、3 个 draw（CPU 时间）；`ui_prior` 的打乱顺序参照行不存在。
- Hyser `own` 模式的训练 prior 是 stale 替身（table 空间的混合 + 噪声），不是真实 session1->session2 漂移；真实漂移主要是输入空间的空间模式变化。这是 Hyser `own` 在 B>=5 时低于 ridge 的可能原因之一，没有验证（Hyser 只有一个 source session/受试者，无法得到真实的跨 session 训练对）。
- Hyser source 数据：每个 fold 只有 13 个训练受试者；train/test 的 mean prior 张数略不同（训练 12–15 张，测试 19 张）。DB6 只有 19–29 个训练 session，calibrator 容易在 session 层过拟合（best step 250–3150，`own` 在 source val 上无提升）。
- 只做了 P0/P1/P2/P4；P3、P-oracle（Q-B 诊断）、行增益/偏移、`_nolab` 标签对照没有做。calibrator 没有 unit/channel dropout。DB6 的 lr 扫描只做了 `own`/P0,P4/seed 0（CPU）；`mean`、`free` 没有扫。
- 超参数（lr 1e-3、patience、hidden=64、Hyser stale 强度、B 分布）在 Hyser 上是事先固定的，没有调。
- CI 以 target/subject 为单位，不含 backbone seed 与 calibrator 训练 seed 的方差（每个 backbone seed 只训练一个 calibrator，Hyser 是每个 fold 一个）；seed spread 见 2.4 和 0.1 第 5 条。
- Gram 累加器 SRAM 远大于 32 KB（第 3 节）；Gram-free 替代没有评测。
- Hyser 前端不是因果的（见第 1 节）。DB6 v2 的 trial 打乱拼接点保留了前一个 trial 的包络衰减尾（v2 review：约 0.006 的弱 “同一抓取” 线索）；本模块沿用 `reorder_stream` 的原始拼接，没有做 review 建议的连续性修正（backbone 训练本身也带这个线索，修正需要重训 backbone）。
- 事后挑选的 “best deployable” 基线对基线有利；它是按 target/受试者均值选的，需与配对 CI 一起读。
- 没有更新 Overleaf exp-log、没有 GDrive 备份、没有 push。

## 6. 异常

2026-10-16 约 06:00，cuda:1 的 CUDA 驱动失效（`torch.cuda.is_available()` False，`nvidia-smi` 报 GPU0 Unknown Error，需要 root 重置）。当时 DB6 calibrator 训练已完成、Hyser 全部完成，只剩 DB6 评测。回退方案：`mamba3_cpu.CPUDecoder`（已有的 CPU 端口，SSM 状态在 64-bin chunk 间携带）去掉 128 bin 窗口限制后整段解码，并与存档 GPU 行对照（见第 5 节首条）。DB6 lr 扫描用了同一 CPU 端口的可微版本（输入可微、权重为常数）训练。

## 7. 文件

本目录：`hyser_table.csv`（所有方法 x B 的均值与 CI）、`hyser_paired.csv`、`hyser_variant_pairs.csv`、`hyser_seed_table.csv`、`hyser_tables.md`；`db6_original_{table_bal_acc,table_macro_f1,gap_table_bal_acc,seed_table_bal_acc,paired,variant_pairs}.csv`、`db6_original_tables.md`、`db6_shuffled_{table_bal_acc,paired}.csv`、`db6_shuffled_tables.md`、`db6_cpu_vs_gpu_check_original.json`；`training_summary.csv`（每个 calibrator 的 source-val 与 best step，含 DB6 lr 扫描）、`h_flags.json`、`cost_model.{json,csv}`。

复现：`python scripts/run_semg_calib.py train hyser --seed S`（120 个 run/2 seed，GPU 拥挤时数小时）-> `eval hyser --seed S --check`；`train db6 --seed S --db6-ckpt results/db6_v2_20261016/final/seed{seed}/best.pt --out-tag db6v2 --std-floor 0.01` -> `eval db6 --seed S ... --order original --check-ref [--device cpu --threads N]`（`--targets 1-10`；默认 `1-20` 是 Hyser 的范围，DB6 跑完 10 名受试者后会在 11 报错退出，不影响已写出的行）；`python scripts/summarize_semg_calib.py {hyser,db6}`、`scripts/summarize_semg_calib_meta.py`、`scripts/semg_cost_report.py`。测试：`pytest tests/test_semg_calib.py`（21 项）。
