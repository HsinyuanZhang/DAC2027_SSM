# sEMG 上的 fixed-weight forward calibrator（Q-D，Hyser H-A + NinaPro DB6 v2）

日期：2026-10-16。Hyser 与 DB6 calibrator 训练在 cuda:1；DB6 评测与 DB6 lr 扫描在 CPU（GPU 中途失效，见状态与第 6 节）。代码：`ssm_decode/semg_calib.py`（描述量、流式累加器、calibrator、cost model）、`ssm_decode/semg_calib_env.py`（Hyser episode 环境）、`ssm_decode/semg_calib_db6.py`（DB6 v2 contiguous-stream episode 环境）、`scripts/run_semg_calib.py`（train / eval）、`scripts/summarize_semg_calib.py`、`scripts/summarize_semg_calib_meta.py`、`scripts/semg_cost_report.py`、`tests/test_semg_calib.py`（60 项）。原始 rows 与 checkpoint 在 `results/semg_calib_20261016/`（gitignore）；本目录是压缩后的 csv/json。

**状态：** Hyser（2 个 backbone seed x 3 种 prior x 4 个变体 x 5 个 subject fold，每个 seed 7680 条 candidate row）和 DB6 v2（2 seed x 3 prior x 4 变体，记录顺序整段解码，每个 seed 18000 条 candidate row；打乱顺序的诊断只做了 seed 0、B in {4,16}）都已完成。**两处与计划不同，必须知道：（1）GPU 在 2026-10-16 06:00 左右失效（cuda:1 的 `torch.cuda.is_available()` 变为 False，`nvidia-smi` 报 GPU0 Unknown Error），所以全部 DB6 评测和 DB6 的 lr 扫描是在 CPU 上用 Mamba-3 的 CPU 端口（`mamba3_cpu.py`，解除 128 bin 窗口限制）完成的；CPU 解码已用 6000 个 (target,B,draw,方法) 对照存档的 GPU 参照行：平均偏差 −0.0007，平均绝对偏差 0.0022，最大 0.069，12% 的行偏差 > 0.005（集中在含近死通道的受试者 S4–S6、S10，bf16 舍入翻转个别 bin），不影响均值层面的比较。DB6 的 calibrator 训练（24 个）和 Hyser 全部在 GPU 上完成。（2）DB6 参照行（reuse/mean/ridge/ui_prior）直接取自 v2 报告的存档行（GPU 解码，同一抽样），没有重算。**

## 审查后修正（2026-10-16，独立审查之后；本节优先于下文所有与之冲突的数字和措辞）

独立审查（复现脚本在 `/tmp/semg_review/`）指出了若干问题。本节逐项给出修正和一个补做的对照实验。下文第 0–7 节保留审查前的叙述，凡被本节改动的数字或结论，已在原处加 “[审查后]” 标注或直接替换。选择规则没有改动：全部 calibrator 仍按原始规则（source validation、patience 6、至少 400 步、上限 4000 步）选 checkpoint；P0-wide 等新对照也用同一规则、同一 5 个 subject fold、同一 2 个 backbone seed、同一随机数种子序列；训练只用 source，候选行无 target 反传，只用 B 个校准 trial。新对照的原始 rows 和 checkpoint 在 `results/semg_calib_20261016/fix/`（gitignore）。

### 审查后 0. 结果一览

1. **容量对照（最重要）：identity-free 的 pooling 增益不是容量造成的。** 把 P0 的生成器 MLP 加宽到与 P2 同参数量（P0-wide，hidden 113，22,164 参数，P2 为 22,114）后，P0-wide 与 P0 几乎相同（B=5/10 的 R² 为 0.059/0.043，对 P0 的 0.050/0.016；P0-wide − P0 = +0.009/+0.027，CI 含 0）。P2 − P0-wide = +0.081 / +0.123 / +0.198 / +0.203（B=1/2/5/10；seed×fold 聚类 CI ±0.089 / ±0.083 / ±0.070 / ±0.103，B>=2 不含 0）；P4 − P0-wide = +0.092 / +0.136 / +0.190 / +0.171（±0.067 / ±0.054 / ±0.064 / ±0.097，四个 B 都不含 0）；与参数量对齐 P4 的 P0-wide4（27,364 参数）比，P4 − P0-wide4 = +0.094 / +0.118 / +0.213 / +0.251（四个 B 都不含 0，fold-only 聚类也不含 0）。**即：+0.2（B>=5）的增益在容量对齐后仍在，约 +0.20；它来自跨通道的 set pooling，不是参数更多。**
2. 去掉 pooling 但保留 P2 全部参数（P2-local：上下文换成逐通道的 psi(d~_i)，22,114 参数）只拿到一部分：B=5/10 为 0.114/0.103，P2 − P2-local = +0.143 / +0.143（B>=5 不含 0）；P2-local − P0-wide = +0.055 / +0.060（B>=5，CI 含 0）。上下文换成一个学出来的常数向量（P2-const，11,682 参数，无任何群体信息）则回到 P0 水平（0.026 / −0.074），所以增益需要真实的群体上下文。
3. **零 table 坍塌的披露**（第 2 小节）：identity-free 的 P0 有 3 个 cell（seed0 fold4、seed1 fold2、seed1 fold4）、P1 有 2 个 cell 的 best_step 为 0，table 始终是初始的全零表。这些 cell 去掉后 P2/P4 − P0 在 B=1/2 变大（+0.12 到 +0.16），B=5/10 与全样本同量级（+0.17 到 +0.21），所以增益不是坍塌造成的。但是：**所有不含 pooling 的变体（P0、P1、P0-wide、P0-wide4、P2-local、P2-const）在 seed0 fold4 和 seed1 fold4 都坍塌，P2/P4 在这两个 fold 不坍塌**，因此 “pooling 的增益” 里有一部分是 “pooling 避免了优化失败”，现有数据无法把两者分开。
4. DB6 identity-free 容量对照（第 1 小节末尾）：P0-wide 只比 P0 高 +0.001 到 +0.009，P2 − P0-wide 仍有 +0.006 到 +0.019，方向一致但幅度小一个量级。
5. DB6 `own`：seed 0 的四个 calibrator 都是 best_step 0，即 calibrator 就是 prior 本身；seed 1 才有 500–1800 步的最优点（第 3 小节）。
6. DB6 identity-free 的 pooling 增益降级为 “依赖 backbone seed”：P2 − P0 在 seed 0 为 +0.020 到 +0.037，seed 1 为 +0.005 到 +0.014（第 4 小节）。
7. **Hyser 的 CI 改成 seed×fold 聚类，作为主要 CI**（第 5 小节）：同一 subject fold 的 4 个 target 共用一个 calibrator，原来的 target-level CI 反保守。`mean` prior 下 P4 − P0 在 B>=2 不再显著；`own` 的 P4 − reuse 在 B=1/2 仍显著。
8. **成本模型修正**（第 6 小节）：P2/P4 漏了一个 hidden×hidden 层，P2 固定权重 MAC 21,728/channel（原来 17,632）、P4 27,424（原来 23,072）；实际参数量 P2 22,114、P4 27,362；还补算了 StreamAcc 逐 trial 复制的累加器（约 2 倍 Gram）和只需 6.2 KB 的最小方案。累加器 SRAM 仍远超 32 KB，**H5 只对输出表成立，不能写成整体满足**（第 7 小节）。

### 审查后 1. 容量对照（Hyser H-A，identity-free）

**设计。** 新增四个对照变体（`semg_calib.py`，`CONTROL_VARIANTS`），全部是 identity-free（`free`：零 prior、无 prior 列），全部不含任何跨通道的信息交换（单元测试 `test_control_variants_have_no_cross_channel_dependence` 验证：扰动第 j 个通道的描述量，其他通道的输出逐元素不变；P2/P4 作为阳性对照确实会变）：

- `p0w`：P0（独立逐通道编码器，无 reliability）的 hidden 从 64 加宽到 113，22,164 参数，对 P2 的 22,114（差 0.2%）。
- `p0w4`：同上，hidden 129，27,364 参数，对 P4 的 27,362。
- `p2l`（P2-local）：结构与 P2 完全相同（含 reliability shrinkage 和 psi），只把上下文 `mean_i psi(d~_i)` 换成逐通道的 `psi(d~_i)`；参数量 22,114，与 P2 一致，没有任何池化。
- `p2c`（P2-const）：上下文换成一个学出来的常数向量（dctx=32），去掉 psi，11,682 参数（比 P2 少，这是任务里要求的 “常数向量” 版本；参数量对齐的版本是 `p2l`）。

训练与评测完全沿用原协议：5 个 subject fold x 2 个 backbone seed，`torch.manual_seed` 与 episode 随机数种子同原始变体（`seed*100+fold`），所以同一 (seed, fold) 的 episode 序列与 P0/P2 相同；source validation 选 checkpoint；Hyser H-A 20 个 target、B in {1,2,5,10}、每格 8 个 draw（`[7919,t,B,draw]`，与参照一致）。每个新变体 10 个 calibrator，共 40 个，每个 calibrator 40–250 s（GPU，与 P3 队列共享）。CI：均值差用 seed×fold 聚类（10 个 cluster，t 分布 9 df；同一 subject fold 的 4 个 target 共用一个 calibrator）；`[ ]` 里是 two-way（seed×fold 与 target 同时聚类，Cameron-Gelbach-Miller），因为两个 backbone seed 看到同样的 target；`*` 表示 CI 不含 0。csv：`hyser_variant_pairs.csv`（列 `diff_ci95` 为 seed×fold，`diff_ci95_fold`、`diff_ci95_2way`、`diff_ci95_target` 为其他口径）。


**结果。** T1 是 identity-free 各变体的 R²（均值 ± 95% CI，seed×fold 聚类；`best_step=0 cells` 为该变体 10 个 calibrator 中保留零表的个数，见第 2 小节）；T2 是配对差（seed×fold 聚类 CI，`*` = 不含 0，`[ ]` 内是 two-way CI 的半宽）。

**T1 水平**

| variant | params | src-val mean | best_step=0 cells | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|---|---|---|
| `cal_p0_free` | 9,424 | 0.072 | 3/10 | -0.122 ± 0.057 | -0.063 ± 0.048 | 0.050 ± 0.082 | 0.016 ± 0.107 |
| `cal_p0w_free` | 22,164 | 0.069 | 2/10 | -0.131 ± 0.065 | -0.082 ± 0.049 | 0.059 ± 0.076 | 0.043 ± 0.094 |
| `cal_p0w4_free` | 27,364 | 0.080 | 2/10 | -0.133 ± 0.048 | -0.064 ± 0.039 | 0.036 ± 0.070 | -0.036 ± 0.126 |
| `cal_p1_free` | 9,602 | 0.103 | 2/10 | -0.083 ± 0.037 | -0.075 ± 0.074 | 0.027 ± 0.094 | -0.029 ± 0.134 |
| `cal_p2l_free` | 22,114 | 0.103 | 2/10 | -0.057 ± 0.017 | -0.004 ± 0.034 | 0.114 ± 0.085 | 0.103 ± 0.086 |
| `cal_p2c_free` | 11,682 | 0.084 | 2/10 | -0.073 ± 0.033 | -0.085 ± 0.067 | 0.026 ± 0.115 | -0.074 ± 0.300 |
| `cal_p2_free` | 22,114 | 0.270 | 0/10 | -0.050 ± 0.046 | 0.041 ± 0.061 | 0.257 ± 0.033 | 0.247 ± 0.053 |
| `cal_p4_free` | 27,362 | 0.245 | 0/10 | -0.040 ± 0.033 | 0.054 ± 0.046 | 0.248 ± 0.051 | 0.215 ± 0.072 |


**T2 配对差**

| a - b | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|
| `p2-p0` | +0.073 ± 0.070* [0.076] | +0.104 ± 0.078* [0.086*] | +0.207 ± 0.080* [0.088*] | +0.230 ± 0.113* [0.112*] |
| `p4-p0` | +0.083 ± 0.072* [0.066*] | +0.117 ± 0.059* [0.061*] | +0.199 ± 0.072* [0.080*] | +0.198 ± 0.109* [0.116*] |
| `p2-p0w` | +0.081 ± 0.089 [0.104] | +0.123 ± 0.083* [0.098*] | +0.198 ± 0.070* [0.090*] | +0.203 ± 0.103* [0.120*] |
| `p4-p0w` | +0.092 ± 0.067* [0.077*] | +0.136 ± 0.054* [0.067*] | +0.190 ± 0.064* [0.078*] | +0.171 ± 0.097* [0.110*] |
| `p4-p0w4` | +0.094 ± 0.058* [0.057*] | +0.118 ± 0.051* [0.057*] | +0.213 ± 0.050* [0.053*] | +0.251 ± 0.135* [0.138*] |
| `p0w-p0` | -0.009 ± 0.089 [0.085] | -0.019 ± 0.048 [0.039] | +0.009 ± 0.046 [0.037] | +0.027 ± 0.079 [0.074] |
| `p0w4-p0w` | -0.002 ± 0.073 [0.074] | +0.018 ± 0.049 [0.044] | -0.023 ± 0.035 [0.034] | -0.080 ± 0.104 [0.094] |
| `p2-p2l` | +0.008 ± 0.046 [0.057] | +0.045 ± 0.045 [0.064] | +0.143 ± 0.064* [0.081*] | +0.143 ± 0.079* [0.094*] |
| `p2-p2c` | +0.024 ± 0.047 [0.058] | +0.125 ± 0.101* [0.114*] | +0.231 ± 0.109* [0.114*] | +0.320 ± 0.303* [0.300*] |
| `p4-p2l` | +0.018 ± 0.034 [0.045] | +0.058 ± 0.033* [0.044*] | +0.134 ± 0.043* [0.054*] | +0.111 ± 0.036* [0.047*] |
| `p2l-p0w` | +0.074 ± 0.073* [0.081] | +0.078 ± 0.048* [0.058*] | +0.055 ± 0.057 [0.065] | +0.060 ± 0.096 [0.106] |
| `p2c-p0w` | +0.058 ± 0.060 [0.063] | -0.002 ± 0.041 [0.046] | -0.033 ± 0.086 [0.088] | -0.117 ± 0.279 [0.282] |
| `p4-p2` | +0.010 ± 0.034 [0.035] | +0.013 ± 0.035 [0.038] | -0.009 ± 0.037 [0.053] | -0.032 ± 0.056 [0.068] |


要点：

- 容量不是解释。P0 → P0-wide → P0-wide4（9.4k → 22.2k → 27.4k 参数）在 B>=5 都停在 R² −0.04 到 0.06，source-val 均值 0.072 / 0.069 / 0.080，而 P2/P4 为 0.270 / 0.245；source validation 本身就能看出差别，选择没有被 target 影响。
- P2 的参数中，“逐通道更深的网络” 贡献有限（P2-local 0.114 / 0.103，高于 P0-wide 但在 B>=5 不显著），“跨通道的上下文” 贡献大头（P2 − P2-local = +0.143，B>=5 显著）。P2-const 比 P0 还略差（B=10 的 CI 极宽，±0.30，有个别 target 的表严重偏离），说明上下文不是靠多出来的 bias 向量起作用。
- P2-local 相对 P0-wide 在 B=1/2 显著为正（+0.074 / +0.078），在 B>=5 不显著（+0.055 / +0.060）。
- P4 相对 P2 仍无增益（P4 − P2 的 CI 均含 0），与审查前一致。
- 不能说的：（a）样本是 5 个 fold x 2 个 backbone seed、共 10 个 calibrator 的单次训练，没有重复训练 seed；（b）所有 identity-free 变体的绝对水平都远低于 reuse 与 ridge（B=10：P2 0.247，ridge 0.780），这个对照只回答 “pooling 的增益是不是参数更多”，不回答 “identity-free 可用”；（c）不含 pooling 的变体有 2–3 个 cell 坍塌（见下节），pooling 变体不坍塌，pooling 的作用里含 “更好优化” 的成分，现有实验无法拆开；（d）P0 族的表在 B>=5 可能还存在尺度错配（审查脚本 `diag_hyser.py` 里把表整体缩放 0.5 倍后个别 target 的 R² 反而更高），本轮没有追这条线。
- 同一规则下 P0-wide 没有被 “救” 过：它的 best_step=0 cell 是 seed0 fold4、seed1 fold4（2 个），seed1 fold2 虽然离开了初始点（best_step 1900，source-val 0.035）但很弱。

**DB6 v2 identity-free 容量对照（可选项，已做）。** 同样的两个对照在 DB6 上训练（2 个 backbone seed，各 1 个 calibrator；`free`，source validation 选 checkpoint，规则同原文）：`p0w`（hidden 113，18,427 参数，对 P2 的 18,369）和 `p2l`（P2 去掉池化，18,369 参数）。**注意：这四个 calibrator 是在 CPU 上训练和评测的（GPU 队列被占用，单 GPU），原来的 P0/P1/P2/P4 的训练在 GPU 上、评测在 CPU 上；设备差异只是浮点实现不同，没有做专门核对。**source-val：p0w 0.457 / 0.477（seed 0/1），p0 0.452 / 0.469，p2 0.470 / 0.481，p2l 0.447 / 0.454。评测协议同 §4（记录顺序整段解码，5 个 draw，B in {1,2,4,8,16}，n=10 受试者，t-CI；2 个 seed 平均，CI 不含 calibrator 训练方差）：

| 变体 | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| `cal_p0_free` | 0.311 | 0.373 | 0.457 | 0.532 | 0.558 |
| `cal_p0w_free` | 0.320 | 0.377 | 0.457 | 0.538 | 0.564 |
| `cal_p2l_free` | 0.312 | 0.373 | 0.448 | 0.515 | 0.533 |
| `cal_p2_free` | 0.326 | 0.393 | 0.477 | 0.556 | 0.579 |
| `cal_p4_free` | 0.323 | 0.394 | 0.480 | 0.557 | 0.578 |

| 配对差 | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|
| P0-wide − P0 | +0.009 ± 0.006 | +0.003 ± 0.007 | +0.001 ± 0.006 | +0.006 ± 0.008 | +0.006 ± 0.008 |
| P2 − P0-wide | +0.006 ± 0.004 | +0.016 ± 0.002 | +0.019 ± 0.006 | +0.018 ± 0.010 | +0.015 ± 0.008 |
| P4 − P0-wide | +0.003 ± 0.008 | +0.017 ± 0.004 | +0.023 ± 0.010 | +0.018 ± 0.013 | +0.014 ± 0.011 |
| P2 − P2-local | +0.014 ± 0.006 | +0.019 ± 0.007 | +0.029 ± 0.009 | +0.042 ± 0.014 | +0.047 ± 0.017 |
| P2-local − P0 | +0.001 ± 0.005 | −0.000 ± 0.007 | −0.009 ± 0.008 | −0.018 ± 0.008 | −0.025 ± 0.011 |

分 seed 的 P2 − P0-wide：seed 0 为 +0.010 / +0.015 / +0.021 / +0.021 / +0.025，seed 1 为 +0.003 / +0.017 / +0.018 / +0.015 / +0.005（`db6_original_variant_pairs_by_seed.csv`）。结论：DB6 上加宽 P0 只带来 +0.001 到 +0.009（仅 B=1 的 CI 不含 0），所以 P2 − P0 的 +0.015 到 +0.024 里约 0.00–0.01 可能是容量，P2 − P0-wide 仍有 +0.015 到 +0.019（B>=2），去掉池化但保留参数的 P2-local 反而比 P0 差（B=16 −0.025），与 Hyser 方向一致；但幅度小一个量级，且依赖 seed（见 “审查后 4”），只有 2 个 backbone seed，不能称 “可靠”。



### 审查后 2. 零 table 坍塌（best_step=0）的披露

按原始选择规则（source-val 最高的 checkpoint，step 0 也是候选），identity-free 的下列 calibrator 保留了第 0 步的全零表（生成器输出层零初始化，所以 table = 0，解码输入为 0，R² 约 −0.04）：

| 变体 | best_step=0 的 (seed, fold) | 另有 “几乎没动”（source-val 最优 < 0.05，best_step>0） |
|---|---|---|
| P0 | (0,4)、(1,2)、(1,4) | (1,3)：0.036 |
| P1 | (0,4)、(1,4) | (1,2)：0.023 |
| P0-wide / P0-wide4 | (0,4)、(1,4) | P0-wide：(1,1) 0.031、(1,2) 0.035、(1,3) 0.018；P0-wide4：(1,2)、(1,3) |
| P2-local / P2-const | (0,4)、(1,4) | P2-local：(1,2) 0.026；P2-const：(1,2) −0.024 |
| P2 / P4 | 无 | 无 |

（这些 cell 的 source-val 在 600 步内从未超过第 0 步的约 −0.04，所以早停；`training_summary.csv` 新增列 `collapsed_step0` 和 `weak_val`，阈值 0.05 只是描述用，不参与任何选择。）这些 cell 的 target 行就是 “零表 + B-trial 归一化” 解码的结果，已包含在审查前报告的 P0/P1 数字里。下面给出 “包含” 与 “去掉坍塌 cluster” 两种口径（`drop_collapsed` = 去掉比较双方任一方 best_step=0 的 (seed, fold) cluster；`drop_collapsed_or_weak` = 再去掉 source-val 最优 < 0.05 的；括号内是剩下的 cluster 数；数字的主口径仍是原始规则下的 `all`，`*` = CI 不含 0）：

**T3 坍塌敏感性（配对差，seed×fold 聚类 CI）**

| pair | subset (clusters kept) | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|---|
| `p2-p0` | all (10) | +0.073 ± 0.070* | +0.104 ± 0.078* | +0.207 ± 0.080* | +0.230 ± 0.113* |
| `p2-p0` | drop_collapsed (7) | +0.123 ± 0.053* | +0.135 ± 0.098* | +0.167 ± 0.102* | +0.211 ± 0.166* |
| `p2-p0` | drop_collapsed_or_weak (6) | +0.131 ± 0.061* | +0.142 ± 0.120* | +0.161 ± 0.125* | +0.210 ± 0.206* |
| `p4-p0` | all (10) | +0.083 ± 0.072* | +0.117 ± 0.059* | +0.199 ± 0.072* | +0.198 ± 0.109* |
| `p4-p0` | drop_collapsed (7) | +0.130 ± 0.074* | +0.150 ± 0.069* | +0.183 ± 0.110* | +0.212 ± 0.166* |
| `p4-p0` | drop_collapsed_or_weak (6) | +0.144 ± 0.081* | +0.161 ± 0.078* | +0.178 ± 0.136* | +0.216 ± 0.206* |
| `p1-p0` | all (10) | +0.040 ± 0.052 | -0.012 ± 0.063 | -0.023 ± 0.043 | -0.045 ± 0.046 |
| `p1-p0` | drop_collapsed (7) | +0.063 ± 0.070 | -0.012 ± 0.099 | -0.035 ± 0.065 | -0.065 ± 0.064* |
| `p1-p0` | drop_collapsed_or_weak (6) | +0.068 ± 0.086 | -0.017 ± 0.122 | -0.035 ± 0.081 | -0.068 ± 0.079 |
| `p2-p0w` | all (10) | +0.081 ± 0.089 | +0.123 ± 0.083* | +0.198 ± 0.070* | +0.203 ± 0.103* |
| `p2-p0w` | drop_collapsed (8) | +0.120 ± 0.087* | +0.155 ± 0.084* | +0.178 ± 0.083* | +0.196 ± 0.130* |
| `p2-p0w` | drop_collapsed_or_weak (5) | +0.097 ± 0.073* | +0.141 ± 0.147 | +0.142 ± 0.143 | +0.151 ± 0.232 |
| `p4-p0w` | all (10) | +0.092 ± 0.067* | +0.136 ± 0.054* | +0.190 ± 0.064* | +0.171 ± 0.097* |
| `p4-p0w` | drop_collapsed (8) | +0.119 ± 0.070* | +0.159 ± 0.051* | +0.178 ± 0.082* | +0.182 ± 0.126* |
| `p4-p0w` | drop_collapsed_or_weak (5) | +0.113 ± 0.062* | +0.153 ± 0.093* | +0.152 ± 0.131* | +0.154 ± 0.220 |
| `p4-p0w4` | all (10) | +0.094 ± 0.058* | +0.118 ± 0.051* | +0.213 ± 0.050* | +0.251 ± 0.135* |
| `p4-p0w4` | drop_collapsed (8) | +0.122 ± 0.053* | +0.136 ± 0.053* | +0.207 ± 0.066* | +0.282 ± 0.167* |
| `p4-p0w4` | drop_collapsed_or_weak (6) | +0.132 ± 0.066* | +0.156 ± 0.060* | +0.211 ± 0.097* | +0.313 ± 0.238* |


解读：（i）去掉坍塌 cluster 后 P2/P4 − P0 在 B=1/2 变大（+0.12 到 +0.16），B=5/10 为 +0.16 到 +0.22，与全样本的 +0.20 到 +0.23 同量级（P2 − P0 在 B=5/10 略小，P4 − P0 在 B=10 略大，差都远小于 CI），所以 identity-free 的 pooling 增益不是零表 cell 造成的；（ii）P0 在去掉坍塌 cell 后 B=5 的 R² 也只有 0.098，仍然很低；（iii）P1 − P0 在去掉 cluster 后 B=10 变为 −0.065 ± 0.064（显著为负），原来的 “P1 ≈ P0” 在小样本口径下不稳；（iv）去掉 cluster 后剩余 cluster 只有 5–8 个，CI 变宽，B=5/10 的 “弱 cell 也去掉” 行已有 CI 变得含 0 的情形（如 P2 − P0-wide 在 5 个 cluster 的口径）。

**为什么会坍塌：** 现有证据只说明它发生在 fold4（两个 seed）和 seed1 fold2，且全部发生在没有 pooling 的变体上；没有做机制诊断。可能是优化问题（zero-init 的输出层，lr 1e-3，clip 1.0）或者该 fold 的 source-val 很难（3 个 validation subject 的 R² 噪声），但这只是假设。

**建议的预先注册规则（只用于以后的新运行；本报告的数字不改，P0-wide 等新对照也沿用原规则）：** 一个 calibrator 的 source-val 最优值如果没有比第 0 步（prior-only）高出至少 δ（建议 δ=0.02 R²）（δ 是建议值，没有校准过）就标记为 “untrained”，用预先声明的下一个初始化种子最多重训 2 次；仍然 untrained 的 cell 保留为 untrained 并在所有汇总里同时给出 “含/不含” 两种数字。这样既不依赖 target，也不允许挑选。

### 审查后 3. DB6 `own`：seed 0 的 calibrator 就是 prior

`training_summary.csv`：DB6 `own` 在 seed 0 的四个变体（P0/P1/P2/P4）全部 best_step=0、source-val 0.5037 = prior-only，即选出来的 checkpoint 是零初始化的第 0 步，table = prior（最新 source session 的表）。评测行证实了这一点：`cal_*_own` 与同一流水线重算的 `chk_reuse_latest_calnorm` 在 1500 个 (target, B, draw) 上的差，均值 ≈ −2e-7，最大绝对差 1.4e-4（只有 0.07%–0.13% 的行 >1e-4，是浮点噪声）（`db6_original_own_vs_reuse_check.json`）。seed 1 才学到东西：best_step 为 P0 1500、P1 500、P2 1800、P4 1400（即 500–1800；原文写成 “250–900” 是两个 seed 平均后的值，应当删除），source-val 0.511–0.524，但该验证曲线在相邻 checkpoint 间的波动约 ±0.01（seed 1 P0：0.479 到 0.519），所以 +0.008 到 +0.021 的提升与 “在 20 多个 checkpoint 里取最大” 的选择偏差同量级，不能当作显著提升。lr 扫描（seed 0，lr 3e-4/1e-4，CPU，P0/P4）把 best 推到 0.5048–0.5055，同样只比 prior-only 的 0.5037 高 0.001–0.002。

因此 §4.2 里 `cal_*_own` 的 2-seed 平均实际上是 “seed 0 的 prior + seed 1 的略有偏离”；“`own` 的 calibrator 基本等于 prior” 的结论成立，且比原文更强（seed 0 是严格等于）。

**可能的原因（未验证，也没有重训）：** 训练目标是窗口内留出重复 bin 上的无加权交叉熵，rest 类占多数；验证指标是逐 bin balanced accuracy。以 prior 为起点（prior 已经给出很好的 rest/非 rest 划分）时，降低无加权 CE 的方向不一定提高 balanced accuracy；seed 0 的验证曲线随训练还从 0.5037 掉到 0.486–0.501，与此一致（目标错配）。`mean` 和 `free` 的起点远低于最优，无加权 CE 的方向与 balanced accuracy 大体一致，所以它们能学到东西。检验方法是改成类平衡 CE 重训 `own`，本轮没有做。

### 审查后 4. DB6 identity-free 的 pooling：降级为 “依赖 backbone seed”

2-seed 平均的 P2 − P0 = +0.015 到 +0.024 来自两个 backbone seed 的很不一致的结果（`db6_original_variant_pairs_by_seed.csv`，每个 seed 内部 n=10 受试者，t-CI）：

| 差 | seed | B=1 | B=2 | B=4 | B=8 | B=16 |
|---|---|---|---|---|---|---|
| P2 − P0（free） | 0 | +0.020 ± 0.013 | +0.028 ± 0.014 | +0.026 ± 0.013 | +0.035 ± 0.013 | +0.037 ± 0.015 |
| P2 − P0（free） | 1 | +0.010 ± 0.008 | +0.010 ± 0.005 | +0.014 ± 0.007 | +0.013 ± 0.011 | +0.005 ± 0.011 |
| P4 − P0（free） | 0 | +0.022 ± 0.020 | +0.040 ± 0.016 | +0.035 ± 0.017 | +0.039 ± 0.017 | +0.036 ± 0.018 |
| P4 − P0（free） | 1 | +0.001 ± 0.007 | +0.001 ± 0.007 | +0.011 ± 0.009 | +0.010 ± 0.012 | +0.003 ± 0.010 |

seed 0 为 +0.02 到 +0.04，seed 1 只有 +0.005 到 +0.014，P4 − P0 在 seed 1 的多数 B 含 0。每个 backbone seed 只训练了一个 calibrator，一个 seed 内的 CI 只反映受试者的方差，不含 calibrator 训练方差，所以 “小而可靠” 改为 **“正向、但幅度依赖 seed（+0.005 到 +0.04），只有 2 个 seed，不能说可靠”**。`mean` prior 下 P2 − P0 在 seed 0 约 0（P4 − P0 在 B=1 为 +0.018），在 seed 1 为 −0.011 到 −0.015（CI 基本不含 0），也说明方向在 seed 间不一致。DB6 的各个汇总行里 pooling 的结论以此为准。


### 审查后 5. Hyser CI：seed×fold 聚类作为主要口径

同一 subject fold 的 4 个 target 共用一个 calibrator（每个 backbone seed 每个 fold 一个），所以 target 不是独立单位。`summarize_semg_calib.py` 现在以 (target, backbone seed) 为观测（draw 先平均），**主要 CI 用 seed×fold 聚类（10 个 cluster、cluster-robust 方差、t 分布 9 df）**；同时输出 fold-only 聚类（5 个 cluster，把两个 seed 合并，更保守，4 df）、two-way（seed×fold 与 target 同时聚类，因为两个 seed 看同样的 target；9 df）和原来的 target-level CI（`*_ci95_target`，反保守，仅保留作对照）。csv：`hyser_table.csv`、`hyser_paired.csv`、`hyser_variant_pairs.csv`（`diff_ci95` 主口径；`diff_ci95_fold`、`diff_ci95_2way`、`diff_ci95_target`；`excludes_zero*`）、`hyser_variant_pairs_by_seed.csv`（每个 backbone seed 单独的差，n=20 target）。聚类并不总是更宽：`own` 的 P4 − reuse 在 seed×fold 口径下 CI 比 target-level 窄（B=1：±0.057 对 ±0.074），因为 10 个 cluster 只有 9 df 而配对差在 cluster 内高度一致；所以单个 CI 不要孤立地读，two-way 与 fold-only 列用来检验稳健性。对基线（reuse 等，没有 calibrator）的水平值 seed×fold 口径明显过窄（reuse B=5：±0.061，target-level ±0.115），所以下面第 2 节的水平表用 two-way CI。

对结论的影响（全表见 §2）：

- `mean` prior 的 P4 − P0：+0.103 ± 0.083*（B=1）、+0.080 ± 0.095、+0.103 ± 0.113、+0.102 ± 0.115，**B>=2 不再显著**；P2 − P0 四个 B 都含 0；P4 − `mean_loso_calnorm`：B=1 显著（+0.165 ± 0.077），B>=2 含 0。原来的 “`mean` 下 pooling 有 +0.10 的边界显著增益” 撤回。
- `own` 的 P4 − `reuse_s1_calnorm`：+0.096 ± 0.057*（B=1）、+0.063 ± 0.055*（B=2）仍显著，B=5/10 为 +0.026 / +0.019（含 0）；two-way 口径同样 B=1/2 显著，fold-only（5 个 cluster）只有 B=1 显著（±0.073），B=2 不显著（±0.086）。P0 `own` − reuse 只有 B=2 显著（+0.049 ± 0.036），P2、P1 全部含 0。B=1 的收益仍与 reuse 自己的 seed spread（0.081）同量级。
- identity-free 的 P2/P4 − P0：seed×fold 口径四个 B 都不含 0（P2 − P0 +0.073 / +0.104 / +0.207 / +0.230，P4 − P0 +0.083 / +0.117 / +0.199 / +0.198）；fold-only 下 P2 − P0 只有 B>=5 显著，P4 − P0 四个 B 都显著。
- P4 `own` 相对 best deployable：B=5/10 显著为负（−0.076 ± 0.063、−0.178 ± 0.028），不变。

### 审查后 6. 成本模型（rule 8）

- **漏项：** `semg_calib.py` 的 P2 和 P4 的固定权重 MAC 与权重数漏了 psi/phi 里的 hidden×hidden 层。现在 `fixed_macs_per_channel` 按层逐项统计（单元测试用 forward hook 逐 Linear 数 MAC，与解析值一致）：每通道固定权重 MAC P0 9,280 / P1 9,376 / **P2 21,728（原 17,632）** / **P4 27,424（原 23,072）**；P4 比审查给的 “约 27,168” 多 256，是因为 PMA 的两次池化矩阵乘（softmax 加权的 values 和 beta·zj，各 m×dctx = 256）都计入了。实际参数量（含 bias 和 mu0，与 `training_summary.csv` 一致）：Hyser P0 9,424 / P1 9,602 / P2 22,114 / P4 27,362；DB6 7,304 / 7,457 / 18,369 / 23,617（原 README 的 9,280 / 9,441 / 17,697 / 22,881 是不含 bias 的权重数且 P2/P4 缺层，已作废）。`cost_model.csv` 新增列 `fixed_macs_per_channel`、`calibrator_params_total`。
- **重算的总量与占比**（`cost_model.{json,csv}`；累加通路不变）：

| 情形 | 累加 MAC | 终结 MAC | 固定权重 MAC（P0 / P2 / P4） | 固定权重占比（P0 / P2 / P4） |
|---|---|---|---|---|
| Hyser N=256, K=16, 1250 bin/trial, B=1 | 9.06e7 | 1.31e6 | 2.38e6 / 5.56e6 / 7.02e6 | 2.5% / 5.7% / 7.1% |
| Hyser B=10 | 9.06e8 | 1.31e6 | 同上 | 0.26% / 0.61% / 0.77% |
| DB6 N=14, K=8, 424 bin/trial, B=1 | 1.72e5 | 3.05e3 | 1.00e5 / 2.52e5 / 3.32e5 | 36% / 59% / 65% |
| DB6 B=16 | 2.75e6 | 3.05e3 | 同上 | 3.5% / 8.4% / 10.7% |

  （原 README 的 P4 占比：Hyser 6.0% / 0.65%，DB6 61% / 9.0%；P2 未列。）结论 “Hyser 上累加通路主导（90% 是 N×N Gram），固定权重只占 0.3–7%”，“DB6 上固定权重占 3.5–65%” 不变。
- **逐 trial 累加器拷贝（rule 8 要求的缓冲）：** 审查前的 `StreamAcc.update` 为每个 trial 保留了一份完整的 per-trial 累加器（Gram、sx、自相关、标签互协方差），`end_trial` 再加到 session 总量上，等于约 2 倍 Gram；而 cost model 没有计入这份拷贝。按 fp32 全 Gram 计，这份拷贝 Hyser 为 292,872 B（Gram 本身 262,144 B，另 +30.7 KB 为 sx/自相关/标签互协方差），上三角存储为 162,312 B；DB6 1,800 B。**最小方案**：每个 chunk 直接加进 session 累加器，只为可靠性特征（`_q_terms`）保留 N + D·N + D + 1 个值的 per-trial 状态：Hyser 6,168 B（约 6.2 KB），DB6 540 B。`StreamAcc` 已改成最小方案（单元测试：与批量求和逐项一致，`trial_state_bytes` 为 6,168 B 级别）。重算的累加器 SRAM 总量（fp32，含 descriptor、lag 延迟线、归一化统计、输出表）：Hyser **397.5 KB（全 Gram）/ 267.0 KB（上三角）**，若采用旧的逐 trial 拷贝则 684.2 KB / 423.1 KB；DB6 6.7 KB / 6.4 KB（旧方案 8.0 / 7.3 KB）。原来写的 391 KB / 261 KB 没有计 per-trial 状态。**两种方案在 Hyser 上都远超 32 KB。**
- **精度模拟的范围：** 实验里累加器在 float64 里计算（用 `TrialBank` 的 per-trial 和，数值上与 StreamAcc 相同）。审查者检查了 fp32 逐 bin 流式累加（1 bin 一个 chunk）：累加器相对误差约 2e-7，R² 变化 |ΔR²| <= 1e-4（`/tmp/semg_review/diag_fp32.py`，本轮没有重跑）。**定点累加器每次累加取整和饱和的模拟没有做（rule 8 的未完成项）**，所以 “fp32 SRAM 字节数” 是代价模型的假设位宽，不是已验证的精度。Gram-free 替代方案（用固定参考归一化后对 prior latent 的投影代替 N×N Gram）依旧只有解析估计（16.1 kMAC/bin、61.6 KB），没有实现，没有评测。

### 审查后 7. H-flag 的实际含义

`h_flags.json` 的 “通过数” 比名字显示的弱，不能当成约束已被验证的证据（`h_flags.json` 里新增 `_notes` 字段写明）：

- **H1：** 比较的是评测进程里 trunk 的哈希前后是否一致。DB6 的 CPU 评测行里被哈希的是 CPU 端口持有的权重拷贝，不是 GPU 模型，所以这些行的 H1 只说明 CPU 拷贝没有被修改。
- **H2：** `h2_backprop_steps` 在评测脚本里是写死的常数 0，再加上 `calibrate()` 的 no_grad 守卫；它是代码路径的属性，不是测量。
- **H4：** `h4_trials_used` 取的是 `len(cal)`，检查的只是 “校准 trial 个数等于 B”。“描述量只依赖请求的 trial” 由单元测试（把非校准 trial 换成噪声，stats 逐位相同）保证，不由这个 flag 保证。
- **H5：** 只检查输出表（K×N fp16：Hyser 8 KB、DB6 224 B）<= 32 KB。描述量累加器不在其中：Hyser 的累加器 SRAM 是 267–684 KB，**所以 H5 对整条校准通路在 Hyser 上不成立**（DB6 的 6.4–8.0 KB 成立）。§0 和 §0.1 中 “H5 满足” 的说法已改。

### 审查后 8. 本轮没有做的

- 没有重训 DB6 `own`（类平衡 CE 的检验）；没有给 Hyser 的零 table 坍塌做机制诊断，也没有应用 “最少步数/重训” 规则（只写了提议）。
- 容量对照在 Hyser（4 个变体 x 10 个 calibrator）和 DB6（2 个变体 x 2 个 calibrator，CPU 训练）上做了 identity-free；没有对 `mean`、`own` prior 做容量对照；DB6 没有 P0-wide4 和 P2-const。
- 没有做 fp32/定点累加器的完整模拟（审查者的 fp32 结果只是引用）。
- 没有更新 Overleaf exp-log、没有 GDrive 备份、没有 push。

---

## 0. 结论摘要（Hyser，H-A；query R²，物理单位，macro 5 根手指；2 个 backbone seed）

> **[审查后]** 本节和第 2 节里原来的 “±” 是 target-level t-CI（n=20，反保守）；已重算为 seed×fold 聚类 CI（见顶部 “审查后 5”），第 2 节三张表已替换，本节被改动的结论在条目里加了 “[审查后]”。

最强的可部署零/低标签基线（按每个 B 在 target 均值上事后挑选，对基线有利）：B=1/2 为 `reuse_s1_calnorm`（0.167/0.416），B=5/10 为每个 B 在 source 上选超参数的闭式 ridge（0.680/0.780）。ui_prior（梯度穿过 frozen trunk，非部署）：0.729/0.801（s1 表起点）。

1. **用同一受试者 session1 表作 prior（`own`）的 calibrator 只在小 B 超过最好的可部署基线，且幅度有限。** [审查后，seed×fold 聚类 CI] P4（PMA m=8 + reliability）相对 `reuse_s1_calnorm`：B=1 +0.096 ± 0.057，B=2 +0.063 ± 0.055（two-way 口径同样显著；fold-only 口径只有 B=1 显著）；P0 为 +0.058 ± 0.070（含 0）/ +0.049 ± 0.036；B=5/10 只比 reuse 高 +0.02（0.604/0.602 对 0.578/0.583，CI 含 0），但比闭式 ridge 低 0.076 ± 0.063 / 0.178 ± 0.028（0.680/0.780）。
2. **离 ui_prior 的距离：** B=1/2 反而高于 ui_prior_s1_calnorm（+0.437 ± 0.145 / +0.311 ± 0.077，那里梯度拟合因 1–2 个 trial 过拟合而失败，不是 calibrator 强）；B=5/10 低 0.125 ± 0.026 / 0.200 ± 0.023（0.604 对 0.729，0.602 对 0.801）。即 B>=5 时固定权重 calibrator 只拿到 ui_prior 相对 reuse 增益（+0.15/+0.22）的约 17%/9%。
3. **population mean prior（`mean`，跨受试者的 channel-aligned 均值表）：** [审查后] 相对 `mean_loso_calnorm`（同一 prior 不校准），P4 +0.165 ± 0.077 / +0.084 ± 0.091 / +0.065 ± 0.081 / +0.048 ± 0.077（B=1/2/5/10），只有 B=1 的 CI 不含 0；但绝对值（0.034/0.236/0.351/0.345）远低于 reuse 与 ridge。Hyser 里 population 表不是好 prior（与第一轮结论一致）。
4. **identity-free 对照（零 prior，descriptor 里没有 prior 列和 prior-latent 相关）下 set pooling 有明显作用，已知身份时没有：**
   - identity-free：P2/P4 − P0 = +0.07/+0.10/+0.21/+0.23（P2，B=1/2/5/10）、+0.08/+0.12/+0.20/+0.20（P4），seed×fold 聚类 CI 四个 B 均不含 0；P1 − P0 ≈ 0（−0.05 到 +0.04）；P4 ≈ P2（|差| <= 0.03）。P0 在 B>=5 时 R² 只有 0.02–0.05，P2/P4 到 0.21–0.26。**[审查后] 容量对齐的对照（P0-wide，22.2k 参数）没有改变这个结论：P2 − P0-wide = +0.081/+0.123/+0.198/+0.203，P4 − P0-wide = +0.092/+0.136/+0.190/+0.171（见顶部 “审查后 1”）。** 但 P0/P1 及所有无 pooling 的变体有 2–3 个 cell 保留零表（“审查后 2”），pooling 里含 “避免坍塌” 的成分。
   - 已知身份（`own`）：|P4 − P0| <= 0.037（B=1 为 +0.037 ± 0.038，B>=5 时 <= 0.015），P2、P1 同样；pooling 没有增益。
   - `mean` prior：[审查后] P4 − P0 = +0.103 ± 0.083（B=1，显著），B=2/5/10 为 +0.080 ± 0.095 / +0.103 ± 0.113 / +0.102 ± 0.115（含 0）；P2 − P0 四个 B 都含 0。原来的 “边界显著” 撤回。
   - 但 identity-free 的绝对水平仍很低（B=10：P2 0.247、P4 0.215，对 reuse 0.583、ridge 0.780），所以 “pooling 在身份缺失时有用” 不等于 “身份缺失时可用”。
5. **两个 backbone seed 的 spread：** `own` 变体 <= 0.035（B=1 的 P1 为 0.135）；`mean` 与 identity-free 的 P0/P1 高达 0.07–0.18，P2/P4 只有 0.003–0.037。pooling 变体在 seed 间更稳定。
6. 以 source validation 选变体（各 prior 下 source-val 均值最高的变体，Hyser 全部为 P2）：`own`/P2 为 0.212/0.446/0.599/0.600，与 P4 相差 <= 0.05（B=1）。source validation 在 `own`/`mean` 下对变体几乎没有区分度（P2/P4 比 P0/P1 高 0.01–0.03），在 identity-free 下则明确区分（P2 0.270、P4 0.245 对 P0 0.072、P1 0.103），与 target 上的排序一致。

约束检查（[审查后] 措辞更正，见顶部 “审查后 7”）：Hyser 全部 15360 条 candidate row 的 H1（评测进程里 trunk 哈希不变）、H2（`h2_backprop_steps` 恒为 0 且校准在 `no_grad` 下，是代码路径属性）、H4（校准 trial 个数 == B；“描述量只依赖请求的 trial” 由单元测试保证：把非校准 trial 换成噪声后 stats 逐位相同）通过；H5 **只对输出表成立**（K x N fp16 = 8 KB <= 32 KB），**描述量累加器 267–684 KB 超过 32 KB，H5 对整条校准通路不成立**。

### 0.1 DB6 v2（D-A，记录顺序整段 contiguous 解码；per-bin balanced accuracy；n=10 受试者，t-CI；2 个 backbone seed 平均）

最强的可部署零/低标签基线（事后按 B 在受试者均值上挑选）：B=1/2/4 为 `mean_table_calnorm`（0.428/0.477/0.511），B=8 为 `reuse_latest_calnorm`（0.541），B=16 为闭式 `rownorm_ridge`（0.578）。ui_prior_latest_calnorm（非部署）：0.355/0.445/0.553/0.666/0.724（B=1/2/4/8/16）。

1. **没有任何 calibrator 在任何 B 上以 CI 排除 0 的幅度超过最强可部署基线。** 最接近的是 `mean` prior 的 P0：B=4 +0.022 ± 0.027，B=8 +0.035 ± 0.067，B=16 +0.003 ± 0.022；B=1/2 为 −0.020 ± 0.021 / −0.002 ± 0.024。identity-free 在 B=1/2 显著更差（−0.08 到 −0.12，CI 约 ±0.03），B=16 持平（P2/P4：+0.002 / −0.000 ± 0.025）。
2. **相对 “同一个 prior 不校准”，只有 population prior 的 calibrator 有增益：** `mean` prior 的 P0 − `mean_loso_calnorm` = −0.004 ± 0.024 / +0.016 ± 0.030 / +0.043 ± 0.034 / +0.065 ± 0.034 / +0.071 ± 0.034（B=1/2/4/8/16）。**`own`（最新 source session 的表）prior 的 calibrator 基本等于 prior 本身：** P0 − `reuse_latest_calnorm` = −0.009 ± 0.011 … +0.011 ± 0.022（全部 B）。这与 source-side 事实一致（[审查后更正]：seed 0 的四个变体 best step 全为 0，calibrator 严格等于 prior，评测行与 reuse 行的差 <= 1.4e-4；seed 1 的 best step 为 500–1800，source validation 从 0.503 提到 0.511–0.524，与验证曲线的波动同量级，见 “审查后 3”；原文 “0.508–0.514（best step 250–900）” 是两个 seed 平均，已作废），lr 降到 3e-4、1e-4（seed 0，P0/P4，CPU，只看 source val）仍是 0.505–0.506，所以不是学习率问题；在 19 个训练 session（早期 session 作 prior）上，固定权重 calibrator 学不到跨 session 的表修正。
3. **离 ui_prior 的距离：** B=1 时 `mean`/`own` calibrator 高于 ui_prior（+0.03 到 +0.055，那里 ui_prior 因类覆盖不足而失败），identity-free 低 0.03–0.04；B=4 低 0.02–0.04（identity-free 低 0.07–0.10），B=8 低 0.09–0.13，B=16 低 0.14–0.17（0.58 对 0.724）。B=16 时最好的 calibrator（`mean`/P0 0.581、identity-free P2 0.579）只拿到 ui_prior 相对 reuse 的增益（+0.176）的约 19%/18%。B>=8 的 ui_prior 增益需要标签驱动的逐 session 梯度拟合；固定权重前向目前拿不到。
4. **identity-free（零 prior，无 prior 列）下 pooling 的作用 [审查后：依赖 backbone seed，不能称 “可靠”]，已知身份时没有：** 2-seed 平均 P2/P4 − P0 = +0.015 / +0.019 / +0.020 / +0.024 / +0.021（P2，B=1/2/4/8/16，CI 约 ±0.008–0.010，**该 CI 不含 calibrator 训练方差**）；分 seed 看 P2 − P0 在 seed 0 为 +0.020 到 +0.037，在 seed 1 只有 +0.005 到 +0.014（“审查后 4”）；P4 ≈ P2（|差| <= 0.004）；P1 − P0 在 ±0.01 以内。`mean` 或 `own` prior 下 |P2/P4 − P0| <= 0.007（不显著），`mean` 下 P1 − P0 为 −0.008 到 −0.014（略负）。方向与 Hyser 一致，幅度小一个量级。identity-free 在小 B（1–4）显著差于带身份的变体（B=4：0.48 对 0.53/0.51），在 B>=8 不低于 ridge 与 reuse（B=8：0.556–0.557 对 0.514/0.541；B=16：0.578–0.579 对 0.578/0.548）。
5. **backbone-seed spread：** calibrator 行 0.000–0.046（多数 <= 0.03；`mean`/P0 在 B=1 最大 0.046），`reuse_latest_calnorm` 0.005–0.010，`ui_prior` <= 0.006。pooling 变体（P2/P4）的 seed spread 一般小于 P0/P1。
6. **打乱顺序诊断（seed 0，B in {4,16}，每格 3 个 draw，10 名受试者；`db6_shuffled_*`）与记录顺序一致：** `own` 0.517/0.557（reuse 0.518/0.558），`mean`/P0 0.530/0.586（mean_table 0.504/0.518），identity-free P2/P4 0.480/0.590 与 0.491/0.590（ridge 0.379/0.578）；与记录顺序的差别 <= 0.015（样本不同：seed 0、3 个 draw），没有 block 顺序捷径的迹象。
7. 约束检查（[审查后] 措辞更正，见 “审查后 7”）：DB6 全部 36000 + 5064 条 candidate row 的 H1（CPU 行哈希的是 CPU 端口持有的权重拷贝）/H2（恒为 0 的计数加 no_grad 守卫）/H4（只检查个数 == B）通过；H5 只覆盖 224 B 的输出表，DB6 的累加器 SRAM 6.4–8.0 KB 也 <= 32 KB，但这是代价模型的数字，不是 flag 检查的。

## 1. 方法（与第一轮 000688 的关系、偏离处）

**协议：** 与现有 sEMG 参照行完全相同：Hyser H-A（20 个 target = 每名受试者的 session2，B in {1,2,5,10}，每个 (target,B) 8 次平衡抽样——随机数种子 `[7919,t,B,draw]` 与参照一致，已验证 `cal_trials` 逐条相同；query = 其余 trial）。backbone 是已有的冻结 K=16 w128 模型，2 个 seed。**流水线等价检查：** 用本模块的流式 B-trial 统计量 + `reuse_s1` 表重新算出的 R² 与已存的 `reuse_s1_calnorm` 行最大绝对差为 0（`--check`，20 target x 4 B x 8 draw x 2 seed）。

**前端注意：** Hyser `preprocess` 的 EMG 经审查是零相位（非因果）滤波；本轮为与已有参照行可比，仍使用现有 cache（80 ms 泄漏积分器的包络，从 `preprocess` 文件算出），因此 “前端因果” 对 Hyser 不成立，结论里的数字不应当作严格流式结果。

**描述量（每个通道一个 unit；全部由 B 个校准 trial 的可加累加器得到，`SemgStats`）：** log 平均包络、log std、lag 1/3/10 自相关、与 prior 潜变量（K=16）的相关（用 N x N Gram）、与全体通道平均活动的相关、与 5 根手指力在 lag 0/2/4/6 的归一化互协方差（20 维）、标签均值、prior 列（K 维）、log n、log B；共 65 维（DB6：40 维，标签块为 8 类 one-hot 的 lag-0 互协方差）。可靠性特征 5 维：log n、log B、log 水平、trial 间包络 CV²、标签互协方差跨 trial 一致性。输入归一化的均值/std 也来自同一批累加器（std 下限 1e-3 x mean std，与 `hyser_backbone.input_stats` 相同）。

**变体：** P0 独立 per-channel 编码器；P1 = P0 + `d~ = lam d + (1-lam) mu0`，lam = sigmoid(MLP(可靠性特征))；P2 = P1 + DeepSets 上下文 `mean_i psi(d~_i)`；P4 = PMA（m=8 个固定 seed，可 online-softmax）+ 打分里加 log lam。（`semg_calib.py` 是 `population_calib.py` 的独立副本，P3 与 P4 在 000688 上相同，故未单独做 P3。）生成器 2 层宽 64 MLP，输出层零初始化，表 = prior + out_scale x 输出，out_scale = source 表元素的 RMS（固定常数），所以训练起点就是 “prior + B-trial 归一化”。

**prior 的三种构造（训练与评测用同一函数）：**
- `mean`：除目标受试者外其余受试者 session1 表的 channel-aligned 均值（训练时再去掉验证和 held-out fold 的受试者）。
- `own`：目标受试者自己的 session1 表。**Hyser 每个受试者只有一个 source session，所以训练里无法得到 “同一受试者前一天的表”。** 为避免 calibrator 在 “自己的表就是自己的 target” 的 episode 上学成恒等映射，训练使用 stale 替身：prior = (1−rho) T_own + rho T_mean(其他训练受试者) + sigma x out_scale x 噪声，rho~U(0,0.7)，sigma~U(0,0.6)（事先固定，没有调）；并且采用 **subject-fold 训练**：5 个 fold（subject i 属于 fold (i−1) mod 5），fold f 的 calibrator 训练时完全看不到 fold f 的 4 个受试者（数据、表、prior 都不用），再留 3 个受试者做 validation；评测 target t 用 fold(t) 的 calibrator。target 的 session2 从未被训练代码加载。这个 stale 替身是对真实跨天漂移的假设性代理，**没有验证它的分布与真实 session1→session2 漂移相近**（见局限）。
- `free`（identity-free 对照）：prior = 0，descriptor 中 prior 列和 prior-latent 相关置 0（实现为固定 mask；单元测试确认生成器输出与它们无关），其余描述量不变。`mean` 和 `free` 同样用 fold 训练，保证三种 prior 的训练数据量一致（每个 fold 13 个训练受试者 + 3 个 validation 受试者）。

**episode 训练（只用 source session1）：** 随机训练受试者；B 取 log-uniform 整数 1..10；校准 trial 在该受试者的训练 trial（sample 1,2）里按手指均衡抽取；query = 该 session 的 4 个留出 trial（sample 3，backbone 从未见过其标签）；loss = 冻结 trunk 输出的 MSE（梯度只回到 calibrator）。AdamW lr 1e-3，batch 8 episode，每 100 步在 3 个 validation 受试者上评 source-val（B in {1,2,5,10} x 3 draws，query = sample 3 的 5 个 trial，own 模式的 validation prior 取三档固定 stale 强度），patience 6 次、至少 400 步、上限 4000 步。选择、早停全部只看 source validation。全部 120 个 run 中 119 个满足 “best step <= 80% 预算”（1 个 own/P4 未满足，最优步靠近末端；没有再加预算）。calibrator 实际参数量（`training_summary.csv`，含 bias）Hyser P0 9,424 / P1 9,602 / P2 22,114 / P4 27,362，DB6 7,304 / 7,457 / 18,369 / 23,617（[审查后] 原文的 9,280 / 9,441 / 17,697 / 22,881 不含 bias 且 P2/P4 缺 hidden×hidden 层）。identity-free 的 40 个 Hyser run 中 best_step=0 的 cell 见 “审查后 2”（这些 cell 满足 “best step <= 80% 预算” 的形式规则，但没有训练起来）。

**目标端：** 校准 = 流式累加 B 个 trial 的可加和 -> 固定权重前向（`torch.no_grad`，`calibrate()` 在 grad 开启时也会强制进入 no_grad）-> 静态 16 x 256 表；没有任何目标端梯度。单元测试覆盖：通道置换等变性（P0/P1/P2/P4 x aligned/free）、流式分块累加 == 批量、trial 顺序/合并精确、no-grad 守卫、prior 构造不含目标/held-out 数据、描述量只依赖请求的 trial（把其他 trial 换成噪声后 stats 逐位相同）、`input_stats` 一致、cal 抽样与参照行一致、cost model 自洽。

## 2. Hyser 结果

### 2.1 Query R²（均值 ± 95% CI；[审查后] CI 为 two-way（seed×fold 与 target 同时聚类）口径，因为基线没有 calibrator，seed×fold 单独聚类会过窄；2 seed 平均；参照行复用已有行：协议逐条相同；末尾四行是审查后新增的容量对照）

| method | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|
| `reuse_s1_srcnorm`（零 target 数据） | -0.223 ± 0.403 | -0.223 ± 0.406 | -0.210 ± 0.398 | -0.231 ± 0.408 |
| `reuse_s1_calnorm`（reuse + B-trial 统计量） | 0.167 ± 0.148 | 0.416 ± 0.129 | 0.578 ± 0.106 | 0.583 ± 0.111 |
| `mean_table_calnorm` | -0.073 ± 0.262 | 0.208 ± 0.193 | 0.336 ± 0.142 | 0.347 ± 0.142 |
| `mean_loso_calnorm`（去掉自己的 mean 表） | -0.132 ± 0.275 | 0.152 ± 0.203 | 0.286 ± 0.147 | 0.297 ± 0.146 |
| `rownorm_ridge_perB`（闭式 ridge，每个 B 在 source 上选） | -0.043 ± 0.013 | 0.013 ± 0.017 | 0.680 ± 0.131 | 0.780 ± 0.075 |
| `ui_prior_s1_calnorm`（非部署） | -0.175 ± 0.269 | 0.168 ± 0.180 | 0.729 ± 0.083 | 0.801 ± 0.064 |
| `ui_prior_mean_calnorm`（非部署） | -0.374 ± 0.202 | -0.043 ± 0.173 | 0.731 ± 0.081 | 0.813 ± 0.058 |
| **prior = own（session1 表）** | | | | |
| `cal_p0_own` | 0.225 ± 0.159 | 0.464 ± 0.118 | 0.602 ± 0.090 | 0.587 ± 0.105 |
| `cal_p1_own` | 0.160 ± 0.234 | 0.425 ± 0.139 | 0.594 ± 0.098 | 0.584 ± 0.115 |
| `cal_p2_own` | 0.212 ± 0.180 | 0.446 ± 0.124 | 0.599 ± 0.090 | 0.600 ± 0.096 |
| `cal_p4_own` | 0.262 ± 0.146 | 0.479 ± 0.108 | 0.604 ± 0.089 | 0.602 ± 0.098 |
| **prior = mean（population 表）** | | | | |
| `cal_p0_mean` | -0.069 ± 0.211 | 0.156 ± 0.144 | 0.249 ± 0.145 | 0.242 ± 0.158 |
| `cal_p1_mean` | -0.075 ± 0.239 | 0.123 ± 0.164 | 0.223 ± 0.175 | 0.220 ± 0.180 |
| `cal_p2_mean` | 0.010 ± 0.181 | 0.211 ± 0.126 | 0.328 ± 0.107 | 0.316 ± 0.113 |
| `cal_p4_mean` | 0.034 ± 0.186 | 0.236 ± 0.121 | 0.351 ± 0.097 | 0.345 ± 0.102 |
| **identity-free（零 prior，无 prior 列）** | | | | |
| `cal_p0_free` | -0.122 ± 0.070 | -0.063 ± 0.061 | 0.050 ± 0.102 | 0.016 ± 0.126 |
| `cal_p1_free` | -0.083 ± 0.053 | -0.075 ± 0.090 | 0.027 ± 0.120 | -0.029 ± 0.160 |
| `cal_p2_free` | -0.050 ± 0.072 | 0.041 ± 0.092 | 0.257 ± 0.085 | 0.247 ± 0.098 |
| `cal_p4_free` | -0.040 ± 0.055 | 0.054 ± 0.072 | 0.248 ± 0.082 | 0.215 ± 0.100 |
| `cal_p0w_free` | -0.131 ± 0.082 | -0.082 ± 0.073 | 0.059 ± 0.100 | 0.043 ± 0.124 |
| `cal_p2l_free` | -0.057 ± 0.051 | -0.004 ± 0.064 | 0.114 ± 0.113 | 0.103 ± 0.118 |
| `cal_p2c_free` | -0.073 ± 0.049 | -0.085 ± 0.095 | 0.026 ± 0.141 | -0.074 ± 0.324 |
| `cal_p0w4_free` | -0.133 ± 0.065 | -0.064 ± 0.058 | 0.036 ± 0.096 | -0.036 ± 0.144 |

（观测单位为 (target, backbone seed)，8 个 draw 先取平均；10 个 (seed, fold) cluster 与 20 个 target 同时聚类；cluster 数少，CI 本身只是近似；calibrator 训练种子的方差仍未包含，seed spread 见 2.4。）

### 2.2 相对基线与 ui_prior 的配对差（a − b；[审查后] seed×fold 聚类 CI，`*` 表示 CI 不含 0）

| a | b | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|---|
| `cal_p4_own` | best deployable（B=1,2: reuse；B=5,10: ridge） | +0.096 ± 0.057* | +0.063 ± 0.055* | -0.076 ± 0.063* | -0.178 ± 0.028* |
| `cal_p0_own` | best deployable（B=1,2: reuse；B=5,10: ridge） | +0.058 ± 0.070 | +0.049 ± 0.036* | -0.078 ± 0.066* | -0.193 ± 0.034* |
| `cal_p2_own` | best deployable（B=1,2: reuse；B=5,10: ridge） | +0.045 ± 0.108 | +0.031 ± 0.048 | -0.081 ± 0.065* | -0.180 ± 0.027* |
| `cal_p4_own` | `reuse_s1_calnorm` | +0.096 ± 0.057* | +0.063 ± 0.055* | +0.026 ± 0.050 | +0.019 ± 0.052 |
| `cal_p4_own` | `ui_prior_s1_calnorm` | +0.437 ± 0.145* | +0.311 ± 0.077* | -0.125 ± 0.026* | -0.200 ± 0.023* |
| `cal_p4_mean` | `mean_loso_calnorm` | +0.165 ± 0.077* | +0.084 ± 0.091 | +0.065 ± 0.081 | +0.048 ± 0.077 |
| `cal_p4_mean` | best deployable（B=1,2: reuse；B=5,10: ridge） | -0.133 ± 0.128* | -0.180 ± 0.085* | -0.329 ± 0.051* | -0.435 ± 0.050* |
| `cal_p4_free` | best deployable（B=1,2: reuse；B=5,10: ridge） | -0.206 ± 0.078* | -0.362 ± 0.090* | -0.432 ± 0.057* | -0.565 ± 0.050* |

“best deployable” 是事后在 target 均值上挑选的（对基线有利，是保守比较）。全部配对差见 `hyser_paired.csv`。按 target 看：`cal_p4_own` 在 B=1/2/5/10 比 reuse 好的 target 占 90%/75%/70%/65%，最差的 target 分别 −0.455/−0.112/−0.183/−0.200。

### 2.3 变体之间（pooling 问题；配对差 a − b；[审查后] seed×fold 聚类 CI，`*` 表示 CI 不含 0；完整见 `hyser_variant_pairs.csv`，容量对照见顶部）

| prior | a − b | B=1 | B=2 | B=5 | B=10 |
|---|---|---|---|---|---|
| free | P1 − P0 | +0.040 ± 0.052 | -0.012 ± 0.063 | -0.023 ± 0.043 | -0.045 ± 0.046 |
| free | P2 − P0 | +0.073 ± 0.070* | +0.104 ± 0.078* | +0.207 ± 0.080* | +0.230 ± 0.113* |
| free | P4 − P0 | +0.083 ± 0.072* | +0.117 ± 0.059* | +0.199 ± 0.072* | +0.198 ± 0.109* |
| free | P4 − P2 | +0.010 ± 0.034 | +0.013 ± 0.035 | -0.009 ± 0.037 | -0.032 ± 0.056 |
| mean | P1 − P0 | -0.005 ± 0.074 | -0.033 ± 0.041 | -0.026 ± 0.062 | -0.022 ± 0.058 |
| mean | P2 − P0 | +0.079 ± 0.087 | +0.055 ± 0.079 | +0.079 ± 0.084 | +0.074 ± 0.089 |
| mean | P4 − P0 | +0.103 ± 0.083* | +0.080 ± 0.095 | +0.103 ± 0.113 | +0.102 ± 0.115 |
| mean | P4 − P2 | +0.024 ± 0.067 | +0.025 ± 0.048 | +0.023 ± 0.057 | +0.029 ± 0.049 |
| own | P1 − P0 | -0.064 ± 0.131 | -0.040 ± 0.049 | -0.008 ± 0.026 | -0.003 ± 0.035 |
| own | P2 − P0 | -0.013 ± 0.050 | -0.018 ± 0.022 | -0.003 ± 0.014 | +0.013 ± 0.023 |
| own | P4 − P0 | +0.037 ± 0.038 | +0.015 ± 0.025 | +0.002 ± 0.018 | +0.015 ± 0.026 |
| own | P4 − P2 | +0.051 ± 0.069 | +0.033 ± 0.026* | +0.005 ± 0.009 | +0.002 ± 0.008 |

解读：set pooling（P2/P4）的价值随 “每个通道自带的身份信息” 下降：identity-free 时 +0.2（B>=5，容量对齐的对照后仍在），population prior 时 +0.1（[审查后] 聚类后基本不显著，只有 P4 − P0 的 B=1），own prior 时 ≈ 0。P1（reliability shrinkage）在三种设置下都没有可靠增益。P4 相对 P2 没有超出噪声的增益（reliability-weighted PMA 与 DeepSets 等价）。这与 000688 上 “已知 unit-to-electrode 映射时 pooling 无增益” 一致，也支持 “身份被隐藏时 pooling 才起作用”。

### 2.4 backbone seed 的 spread（两个 seed 的 target 均值之差的绝对值；`hyser_seed_table.csv`）

`own`：P0 0.004–0.025、P2 0.022–0.035、P4 0.006–0.033、P1 0.024–0.135；`mean`：P0 0.07–0.115、P1 0.14–0.18、P2 0.003–0.033、P4 0.021–0.037；`free`：P0 0.017–0.12、P1 0.018–0.092、P2 0.008–0.036、P4 0.003–0.020。对照：`reuse_s1_calnorm` 在 B=1/2 的 seed spread 为 0.081/0.056（B>=5 <= 0.004），`ui_prior_s1_calnorm` <= 0.047。B=1 的 P4−reuse 增益（+0.096）与 reuse 自己的 seed spread（0.081）同量级，所以 B=1 的结论对 backbone seed 敏感。

## 3. 成本：校准通路的 MAC / SRAM（解析，按实际 tensor 统计，`cost_model.{json,csv}`；[审查后] 已按 “审查后 6” 重算）

| 情形（B 个 trial） | 累加 MAC（B 个 trial 流式） | 累加 MAC/bin | 终结 MAC | 固定权重 MAC（P0 / P2 / P4） | 固定权重占比（P0 / P2 / P4） | calibrator 参数（P0 / P2 / P4） | 累加器 SRAM（fp32，全 Gram / 上三角；含最小 per-trial 状态） | 若保留逐 trial 完整拷贝 | 输出表 |
|---|---|---|---|---|---|---|---|---|---|
| Hyser N=256, K=16, 1250 bin/trial, B=1 | 9.06e7 | 72,448 | 1.31e6 | 2.38e6 / 5.56e6 / 7.02e6 | 2.5% / 5.7% / 7.1% | 9,424 / 22,114 / 27,362 | 397.5 KB / 267.0 KB | 684.2 KB / 423.1 KB | 8 KB（fp16） |
| Hyser B=10 | 9.06e8 | 72,448 | 1.31e6 | 同上 | 0.26% / 0.61% / 0.77% | 同上 | 同上 | 同上 | 8 KB |
| DB6 N=14, K=8, 424 bin/trial, B=1 | 1.72e5 | 406 | 3.05e3 | 1.00e5 / 2.52e5 / 3.32e5 | 36.4% / 59.0% / 65.5% | 7,304 / 18,369 / 23,617 | 6.7 KB / 6.4 KB | 8.0 KB / 7.3 KB | 224 B |
| DB6 B=16 | 2.75e6 | 406 | 3.05e3 | 同上 | 3.5% / 8.4% / 10.7% | 同上 | 同上 | 同上 | 224 B |

要点：（i）累加通路由 N x N Gram（用于 prior-latent 相关和全局活动相关）主导：N=256 时 65.5k MAC/bin 占 90%，持续 3.6 MMAC/s（50 Hz），固定权重 calibrator 只占 0.3–7%；（ii）**N=256 的累加器 SRAM 约 398 KB（fp32，全 Gram；上三角 267 KB）远超 32 KB**；若像审查前的 `StreamAcc` 那样为每个 trial 再保留一份完整累加器，则为 684 KB（上三角 423 KB）；H5 的 32 KB 约束只对输出表成立，对描述量累加器不成立；（iii）解析估计的 Gram-free 替代（用固定参考归一化后的 prior-latent 投影累加 N x K，而不是 N x N Gram）约 16.1 kMAC/bin、61.6 KB 累加器（DB6 为 490 MAC/bin、2.6 KB），**但没有实现也没有评测，不知道精度是否保持**；（iv）DB6 N=14 的 SRAM 只有 6–8 KB，累加不是瓶颈，固定权重前向占 3.5–65%。（v）累加器在实验里用 float64 模拟；fp32 逐 bin 的结果引自审查（相对误差约 2e-7，|ΔR²| <= 1e-4）；定点逐次取整/饱和没有模拟（rule 8 未完成项）。

## 4. DB6（D-A，v2 contiguous 解码）

### 4.1 实现与协议

- 整段 session 是一条连续因果流，状态跨 trial 携带；query 只在非校准 trial 的 bin 上打分；协议、抽样（随机数 `[7919, subject, day*10+tod, B, draw]`，5 个 draw，B in {1,2,4,8,16}，60 个 target session）、聚合（draw -> tod -> seed -> 受试者 -> 3 个 gap 平均，t-CI n=10）与 v2 参照行逐条相同；backbone 为 v2 的 `final/seed{0,1}`（K=8 w128，在 trial 顺序打乱的流上训练）。
- B-trial 归一化：均值/std 来自 B 个校准 trial，std 下限 `0.01 x median_c std`（v2 在 source 上选出的值，近零 std 通道不会被放大成单位方差噪声；本模块的 `norm_stats(..., floor_ref="median", floor=0.01)` 与 `db6_backbone.input_stats` 同规则，由 `--check-ref` 行验证）。
- calibrator 训练（`semg_calib_db6.py`，24 个：2 seed x 3 prior x 4 变体）：episode = 某个 source session 的 trial 顺序打乱后拼成的流（每个 session 6 个固定排列副本）里随机偏移的 512–1024 bin 窗口，前 128 bin burn-in；校准 trial 是训练重复（class-balanced，B log-uniform 1..20），loss 是窗口内留出重复（backbone 从未用过其标签）的 bin 上的 CE；梯度只到 calibrator。source validation = v2 的固定打乱顺序留出流，10 个受试者各取最新的 source session 作为验证 session（其数据只出现在验证里；`own` 验证用它的前一个 session 表），B in {1,2,4,8,16} x 2 个 draw。AdamW lr 1e-3（`own` 另扫 3e-4、1e-4），patience 6、至少 400 步、上限 4000 步。没有 subject fold：DB6 每个受试者有多个 source session，`own` prior 在训练里就是同一受试者**更早的**一个 source session 的表（不用目标 session 自己的表，测试时用最新 source session 的表），所以不需要 Hyser 里的 stale 替身；`mean` = 其余受试者 source 表的均值（训练时再去掉验证 session 的表，测试与 `mean_loso_calnorm` 相同）。
- source-side 结果（`training_summary.csv`，各变体、2 seed 平均）：source validation（打乱顺序 balanced accuracy）prior-only -> 训练后：`mean` 0.440 -> 0.545–0.553；`free` 0.127 -> 0.460–0.476；`own` 0.503 -> 0.508–0.514（[审查后更正] 分 seed：seed 0 四个变体全部 best_step=0、0.5037 -> 0.5037，即 calibrator = prior；seed 1 best_step 500–1800（P0 1500 / P1 500 / P2 1800 / P4 1400），0.503 -> 0.511–0.524；原文的 “best step 250–900” 是平均值）；lr 扫描（seed 0）`own` 0.504 -> 0.505–0.506（best step 200–500）。

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

变体之间（`db6_original_variant_pairs.csv`，2-seed 平均；[审查后] 分 seed 的结果见 `db6_original_variant_pairs_by_seed.csv` 和顶部 “审查后 4”，P2 − P0 在 seed 0 为 +0.020 到 +0.037，seed 1 为 +0.005 到 +0.014）：identity-free 下 P2 − P0 = +0.015 ± 0.008 / +0.019 ± 0.008 / +0.020 ± 0.009 / +0.024 ± 0.010 / +0.021 ± 0.010，P4 − P0 = +0.012 ± 0.012 … +0.025 ± 0.012，P4 − P2 在 ±0.004 内，P1 − P0 在 −0.002 … +0.010 内；`mean`：P2 − P0 = −0.003 … −0.007、P4 − P0 = +0.002 … −0.007（均不显著），P1 − P0 = −0.008 … −0.014；`own`：|P_x − P0| <= 0.005。

### 4.4 解读

- DB6 上固定权重 calibrator 的实际价值很低：`own` prior（最新 source session 表）下 calibrator 退化为 prior 本身（seed 0 严格等于 prior，seed 1 的 source validation 提升在噪声内，lr 扫描也无提升；可能的原因是训练用无加权 CE、验证用 balanced accuracy，见 “审查后 3”，未验证）；`mean` prior 下 calibrator 比 “未校准的 mean 表” 好 0.04–0.07（B>=4），但仍只是追平 reuse/ridge，不超过；identity-free 下追平 ridge/reuse（B>=8）但小 B 很差。三者在 B=16 都约在 0.56–0.58，离 ui_prior 的 0.72 还有 0.14–0.17。
- 这与 Hyser 的结论方向一致：已知身份且 prior 强（`own`）时固定权重 calibrator 只在最小 B 有一点点收益（Hyser）或没有收益（DB6），prior 弱/缺失时（`mean`、identity-free）calibrator 能显著提升，但提升到的水平仍低于最强的可部署基线或持平。DB6 的 `own` prior 弱于 Hyser（同日 T1/T2 的漂移已与跨天同量级，见 v2 报告），而 calibrator 训练样本只有 19 个 session 级对。
- pooling：[审查后] identity-free 下为正向但很小，且依赖 backbone seed（seed 0 +0.02 到 +0.04，seed 1 +0.005 到 +0.014，只有 2 个 seed）；已知身份时为 0；这与 000688 的 “已知映射时 pooling 无增益” 一致，方向上支持 “pooling 的价值来自缺失的身份信息”，但 DB6 上证据弱。

## 5. 局限与未做

- **GPU 失效后的 CPU 回退（DB6 评测与 lr 扫描）：** CPU 端口与 Triton kernel 在个别行上差 0.02–0.07（bf16 舍入翻转，集中在含近死通道的受试者），均值偏差 −0.0007；参照行（reuse/mean/ridge/ui_prior）取自 GPU 存档，calibrator 行取自 CPU，所以 “calibrator − 参照” 的差里含这个 CPU/GPU 噪声（量级 0.002 的均值绝对偏差，对 0.02–0.07 的效应量可忽略）。已用 `chk_*` 行把同一流水线重算的 `reuse_latest_calnorm`/`mean_table_calnorm` 与存档对照。
- 打乱顺序的 DB6 诊断只做了 seed 0、B in {4,16}、3 个 draw（CPU 时间）；`ui_prior` 的打乱顺序参照行不存在。
- Hyser `own` 模式的训练 prior 是 stale 替身（table 空间的混合 + 噪声），不是真实 session1->session2 漂移；真实漂移主要是输入空间的空间模式变化。这是 Hyser `own` 在 B>=5 时低于 ridge 的可能原因之一，没有验证（Hyser 只有一个 source session/受试者，无法得到真实的跨 session 训练对）。
- Hyser source 数据：每个 fold 只有 13 个训练受试者；train/test 的 mean prior 张数略不同（训练 12–15 张，测试 19 张）。DB6 只有 19–29 个训练 session，calibrator 容易在 session 层过拟合（best step 0–3800；`own` 在 seed 0 为 0，在 seed 1 的 source val 提升在噪声内）。
- 只做了 P0/P1/P2/P4；P3、P-oracle（Q-B 诊断）、行增益/偏移、`_nolab` 标签对照没有做。calibrator 没有 unit/channel dropout。DB6 的 lr 扫描只做了 `own`/P0,P4/seed 0（CPU）；`mean`、`free` 没有扫。
- 超参数（lr 1e-3、patience、hidden=64、Hyser stale 强度、B 分布）在 Hyser 上是事先固定的，没有调。
- [审查后] Hyser 的 CI 已改为 seed×fold 聚类（共用 calibrator 的 target 作为一个 cluster）；DB6 的 CI 仍以受试者为单位（每个 backbone seed 一个 calibrator，被全部受试者共用），不含 calibrator 训练方差，所以 DB6 的 2-seed 平均 CI 不可用于 “跨训练种子是否可靠” 的判断，以分 seed 的差为准；两种数据集都没有重复训练 seed；seed spread 见 2.4 和 0.1 第 5 条。
- Gram 累加器 SRAM 远大于 32 KB（第 3 节，Hyser 267–684 KB）；Gram-free 替代没有评测；定点累加器的逐次取整/饱和没有模拟。
- [审查后] 零 table 坍塌（顶部 “审查后 2”）、Hyser CI 口径（“审查后 5”）、H-flag 的弱含义（“审查后 7”）。
- Hyser 前端不是因果的（见第 1 节）。DB6 v2 的 trial 打乱拼接点保留了前一个 trial 的包络衰减尾（v2 review：约 0.006 的弱 “同一抓取” 线索）；本模块沿用 `reorder_stream` 的原始拼接，没有做 review 建议的连续性修正（backbone 训练本身也带这个线索，修正需要重训 backbone）。
- 事后挑选的 “best deployable” 基线对基线有利；它是按 target/受试者均值选的，需与配对 CI 一起读。
- 没有更新 Overleaf exp-log、没有 GDrive 备份、没有 push。

## 6. 异常

2026-10-16 约 06:00，cuda:1 的 CUDA 驱动失效（`torch.cuda.is_available()` False，`nvidia-smi` 报 GPU0 Unknown Error，需要 root 重置）。当时 DB6 calibrator 训练已完成、Hyser 全部完成，只剩 DB6 评测。回退方案：`mamba3_cpu.CPUDecoder`（已有的 CPU 端口，SSM 状态在 64-bin chunk 间携带）去掉 128 bin 窗口限制后整段解码，并与存档 GPU 行对照（见第 5 节首条）。DB6 lr 扫描用了同一 CPU 端口的可微版本（输入可微、权重为常数）训练。

## 7. 文件

本目录：`hyser_table.csv`（所有方法 x B 的均值与 CI）、`hyser_paired.csv`、`hyser_variant_pairs.csv`、`hyser_seed_table.csv`、`hyser_tables.md`；`db6_original_{table_bal_acc,table_macro_f1,gap_table_bal_acc,seed_table_bal_acc,paired,variant_pairs}.csv`、`db6_original_tables.md`、`db6_shuffled_{table_bal_acc,paired}.csv`、`db6_shuffled_tables.md`、`db6_cpu_vs_gpu_check_original.json`；`training_summary.csv`（每个 calibrator 的 source-val 与 best step，含 DB6 lr 扫描和审查后的容量对照 run；列 `run`=orig/fix、`collapsed_step0`、`weak_val`）、`h_flags.json`（含 `_notes`）、`cost_model.{json,csv}`；审查后新增：`hyser_variant_pairs_by_seed.csv`、`hyser_free_collapse_{sensitivity,levels}.csv`、`hyser_free_cells.csv`、`db6_original_variant_pairs_by_seed.csv`、`db6_original_own_vs_reuse_check.json`；DB6 容量对照的行（`cal_p0w_free`、`cal_p2l_free`）并入 `db6_original_*` 文件。

复现（审查后的容量对照：`... train hyser --seed S --sources free --variants p0w,p0w4,p2l,p2c --out-subdir fix` -> `eval hyser ... --out-subdir fix`；DB6：`train/eval db6 ... --sources free --variants p0w,p2l --out-subdir fix --device cpu`）：`python scripts/run_semg_calib.py train hyser --seed S`（120 个 run/2 seed，GPU 拥挤时数小时）-> `eval hyser --seed S --check`；`train db6 --seed S --db6-ckpt results/db6_v2_20261016/final/seed{seed}/best.pt --out-tag db6v2 --std-floor 0.01` -> `eval db6 --seed S ... --order original --check-ref [--device cpu --threads N]`（`--targets` 默认 Hyser 为 1-20、DB6 为 1-10；本次 DB6 评测运行时默认值还是 1-20，所以跑完 10 名受试者后在 11 报错退出，不影响已写出的行，现已修正）；`python scripts/summarize_semg_calib.py {hyser,db6}`、`scripts/summarize_semg_calib_meta.py`、`scripts/semg_cost_report.py`。测试：`pytest tests/test_semg_calib.py`（60 项）。
