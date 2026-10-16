# FALCON M2（固定通道）population calibration：Colab 回合（2026-10-16）

日期：2026-10-16。执行：Colab session `dac27`（NVIDIA L4 24 GB，12 vCPU），由本机用 `colab` CLI 驱动。规格：任务说明、`docs/feasibility_p1_20261015.md`（§0 约束，含 rule 7–9）、P1/P2/P3 的 README。
代码（全部为新增文件，没有改动已有代码）：`ssm_decode/m2_calib.py`、`scripts/run_m2_calib.py`、`scripts/analyze_m2_calib.py`、`scripts/colab_orch.py`、`scripts/colab_exec_retry.sh`、`tests/test_m2_calib.py`（17 个测试；连同 P1 的 `test_population_calib.py` 21 个和 `test_mamba3_official.py` 1 个，在 VM 上共 39 个全部通过，35 s）。环境配方：`docs/colab_runbook.md`。
原始 rows：`results/m2_calib_20261016/`（gitignored；Drive 备份 `DAC27/colab/out/m2_calib_out.tgz`）。本目录：`tables_m2.md`（全部 markdown 表）、`all_rows.csv.gz`（逐行 rows）、`cells_*.csv`、`paired_*.csv`、`train_summary.csv`、`r1_selection.csv`、`uireg_selection.csv`、`noise_floor.json`、`flags.json`、`m2_r2_vs_B.png`、`parity_p0.csv`。

## 0. 结论摘要

1. **Step 1：环境 parity 通过。** VM（Python 3.13.15 / torch 2.11.0+cu130 / 隔离的 triton 3.5.0）上用 `scripts/run_population_calib.py evaluate` 重算 20 条已存的 P1 P0 row（000688 days stratum，pool O，seed 0 与 seed 1，targets 0616/0617，B∈{1,8,33}，2 draws），**与主机 row 的最大 |ΔR²| = 8.8e-5**（seed 0 的 8 条 B∈{1,8} 为 6.2e-5），远低于 1e-3 的验收线。官方 Mamba-3 在 VM 上 `status=supported`（causal future-perturb 误差 0.0，5.1 ms/forward）。**未发现需要改动仓库代码的 blocker**（triton 3.5.0 `pip install --target SSM/.tools/mamba_deps` 可直接装；`_require_isolated_triton` 的检查完全没有被削弱）。需要注意：`tests/test_mamba3_official.py` 会在 `unsupported_or_unavailable` 时也“通过”，因此它不是有效的检查（见 §1）。
2. **M2 headline（backbone seed 0，5 个 target，query R²，物理单位，均值 ± 95% CI over targets）。**

| B | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior（均值表 + B-trial 输入归一化） | 0.273±0.146 | 0.285±0.146 | 0.302±0.142 | 0.307±0.135 | 0.313±0.134 | 0.319±0.133 |
| P0（可部署，fixed-weight） | 0.292±0.163 | 0.317±0.170 | 0.334±0.165 | 0.339±0.157 | 0.345±0.163 | 0.351±0.164 |
| P0 + R1-stream（可部署） | 0.292±0.163 | 0.316±0.171 | 0.335±0.162 | 0.339±0.155 | 0.343±0.159 | 0.370±0.131 |
| ui_reg（非部署，梯度） | 0.283±0.145 | 0.273±0.129 | 0.311±0.130 | 0.334±0.135 | 0.357±0.118 | 0.379±0.135 |
| 第一轮 F3 / F7（存档，不同协议，见 §3.5） | 0.231 / 0.201 | 0.237 / 0.244 | 0.244 / 0.277 | 0.261 / 0.285 | 0.288 / 0.307 | 0.291 / 0.297 |

   P0 − P-prior（配对，per-target，seed 0）：+0.019 / +0.032 / +0.033 / +0.032 / +0.032 / +0.032（B=1…33），CI ±0.023–0.040；B≥2 时 5 个 target 全部为正。三个 backbone seed 平均后降为 +0.011 / +0.021 / +0.022 / +0.024 / +0.026 / +0.026（CI ±0.022–0.033），4/5、4/5、3/5、4/5、4/5、3/5 个 target 为正。
3. **回答问题：在固定通道的 M2 上，没有任何可部署的 fixed-weight 校准“在噪声之上”稳定地胜过 prior。** 最好的可部署行 P0 比 prior 高约 +0.02 到 +0.03（相对 +7%–10%），符号在 (backbone seed × target) 的 15 个单元中有 10–13 个为正，但是：(i) target 层面的 95% CI 包含 0（seed 0 的 B=2…33 约 ±0.03–0.04；n=5 个 target、df=4，且 pool1 的两个 target 共享同一个 trunk，pool1 上三个 seed 平均几乎为 0：−0.004…−0.001；增益主要来自 pool2 +0.03–0.05 与 pool3 +0.01–0.04）；(ii) 量级与 backbone-seed 噪声底同阶（P0 的 seed-to-seed SD 0.019，prior 0.024；calibrator-seed SD 只有 0.007）；(iii) 增益几乎与 B 无关（B=2 已经 +0.03，之后不增长），B≥8 时约 2/3 可以由“用另一个 source session 的 B-trial 统计量生成的表”复现（p0_swap，§3.3），也就是说主要是一个静态的 prior 修正，而不是对 target 的校准；(iv) 去掉标签（p0_nolab）损失 −0.02（5/5 个 target 为负），所以标签相关的部分（≈ +0.02）才是 target 特异的那部分，量级与噪声底相当。**R1-stream 在 P0 之上没有增益**：B≤16 的差在 −0.005…0.000（三个 seed 平均时 B=8、16 的小幅负值经 sign-flip 检验有统计显著，但只有 0.004–0.005），只有 B=33 是 +0.019（seed 0，4/5）/ +0.011（三 seed，3/5），且几乎全部来自 pool1。**ui_reg（梯度参照）在 B≤8 并不比 P0 好**（seed 0，B=8：−0.005；三 seed：−0.010），在 B=16/33 才领先 P0（seed 0：+0.012 / +0.028；三 seed：+0.005 / +0.015，CI 包含 0），相对 prior 的增益到 B=33 为 +0.060（seed 0，5/5）/ +0.042（三 seed，5/5）。即 M2 上不存在 000688 days 层那种“amortization gap 已闭合”的局面，也不存在 month/year 层那种大的 gap：整个“校准空间”本身就只有 +0.06 左右（梯度参照），trunk 与 prior 已经占据了大部分可得的 R²。
4. **表大小（K×N = 256×96 = 24,576 个参数）：fp32 98,304 B = 96 KB，超过 32 KB 预算；int8 24,576 B（含逐行 scale 25,600 B）= 25 KB，放得下。** P0 的 post-training 逐行 int8 量化对 R² 无影响（逐 target 的 |p0_int8 − p0| ≤ 5e-4，均值 ≤ 1e-4）。第一轮 D2 行的 `h5_bytes` 也是 98,304 B，即它们同样没有满足 32 KB（fp32）。
5. **预算**：见 §5。

## 1. Step 1：环境与 parity

环境配方（精确版本和命令）在 `docs/colab_runbook.md`，要点：

* 仓库与数据放在与主机相同的绝对路径（`/home/xinyuan/Work_host/SSM`、`/home/xinyuan/Work_host/SPINT/sua_exploration/data/dandi_000688/sub-M`），代码路径零改动；`zstd` 需要 `apt-get install`；先把 bundle 从 Drive FUSE 拷到本地盘再解压并核对 SHA256；
* **补充的包/文件**：`pip install pynwb==4.0.0 hdmf==6.1.0`（`dandi688_data.py` 需要 `pynwb`）；`ssm_decode/data.py` 与 `dandi688_data.py` 在 import 时需要 `APST/src`（125 KB，不在 `code.tar.zst` 里，单独 `colab upload`）；`git config --global --add safe.directory .../.tools/mamba_official`（解压后属主为 uid 1002，`compatibility_attempt` 里的 `git rev-parse` 否则返回 128 并被静默报告为 `unsupported_or_unavailable`）；
* triton：`pip install --target SSM/.tools/mamba_deps triton==3.5.0`。pip 会警告 `torch 2.11.0+cu130 requires triton==3.6.0`，这是预期的、无害的（VM 的系统 triton 仍是 3.6.0，隔离的 3.5.0 通过 `PYTHONPATH` 载入；`kernel_triton_matches_runtime=true`）；
* **数据 bundle 的一个问题**：任务说明里的 `data_falcon_m2.tar.zst` 实际装的是 DANDI 000954（HumanPitt，98 MB），**不是 M2**。M2 是 DANDI 000953（MonkeyN，15 GB NWB）。本轮没有上传 NWB，而是用 `ckpts.tar.zst` 里已有的 npz 缓存（`results/data_cache/m2/held_in/*.npz`，7 个 held-in session）通过新增的 `m2_calib.load_m2_recording` 直接读取；因此 `multi_target.cached_recording`（要求 NWB 存在并核对 sha256）没有被使用。只读取了 `held_in` split，没有读 minival / held-out / `official_evalai/`。

Parity 结果（`parity_p0.csv`；VM 上 `PC_OUT=/content/parity` 重算，主机行取自 `results/population_calib_20261015/runs/days_O/p0/seed{0,1}/eval_rows.jsonl`）：

| 对比 | 行数 | 最大 \|ΔR²\| | 备注 |
|---|---:|---:|---|
| seed 0，B∈{1,8}，2 draws，2 targets | 8 | 6.2e-5 | 规定的检查 |
| seed 1，B∈{1,8,33}，2 draws，2 targets | 12 | 8.8e-5 | 额外检查（包含 B=33） |

差异来自 GPU/torch/triton 的数值差别（RTX 3090 / torch 2.5.1 vs L4 / torch 2.11），符合预期。单元测试：VM 上 `test_mamba3_official.py`、`test_population_calib.py`、`test_m2_calib.py` 共 39 个通过（35 s）。**警告**：`test_mamba3_official.py` 在 `status == unsupported_or_unavailable` 时也通过，它不能作为 GPU 可用性的证明；真正的检查是 `compatibility_attempt('cuda:0')`（见 runbook）。

## 2. 设置

**数据与 backbone。** `D2_POOLS`（`scripts/run_labeled_calibration.py`）：pool1 = source {10-19 R1, R2}，targets {10-20 R1, R2}；pool2 = source {10-19 ×2, 10-20 ×2}，targets {10-27 R1, R2}；pool3 = source {10-19, 10-20, 10-27 各 ×2}，target {10-28 R1}。冻结的 wide_n16 backbone（Mamba-3 SISO，width 256，4 层，state 16；K = 256，N = 96 通道）：seed 0 = `forward_eval_m2_wide_n16_20261005`，seed 1/2 = `…_s1/_s2_20261006`；按“best step 落在预算最后 20% 就换成两倍预算的 `_x2` 重训”的规则，seed 1 的 pool3 与 seed 2 的 pool2 用 `_x2`（与当时的 forward-eval 一致，`m2_calib.resolve_pool_dir`，有测试）。三个 seed 都用于 prior/P0/R1 行；ui_reg 也覆盖了三个 seed（seed 1/2 每个 cell 4 个 draw，见 §3.4）。

**prior 与 bias（本轮的一个发现）。** 每个 pool 的 `session_bank_best.pt` 给出每个 source session 的折叠表 `T_s = W0·diag(g_s)`、`b_s = W0·o_s + b0 + e_s`（`m2_calib.fold_bank`）；它们的均值与 `best.pt` 里已折叠的 root `in_proj` 逐元素一致（最大差 3.1e-8，有测试）。**root `in_proj` 带 K 维 bias**（来自 mean offset 与 latent embedding）。第一轮的 `M2BackboneAdapter.make_decode` 与 `cmd_evaluate` 做的是 `z = x_z·Wᵀ`，**丢掉了 bias**。本轮把这个 bias 当作冻结 trunk 的常数加回（不参与校准、不计入 table）。这使 prior 提高约 +0.05（`p_prior` vs `p_prior_nobias`，B=8：0.307 vs 0.255，5/5 个 target 为正，seed 0）。所以：(a) 主表的 prior 是带 bias 的版本；(b) `p_prior_nobias` 保留为与第一轮解码一致的诊断行；(c) 第一轮 D2 数字（F1/F2/F3/F7，约 0.26–0.29）是在“丢 bias 的 prior”之上相对比较的，不能直接与本轮的绝对值比较（§3.5）。`p_prior_srcnorm`（用 source 的全局 x_mean/x_std 而不是 B-trial 统计量归一化）只有 0.174–0.178，且与 B 无关：B-trial 归一化本身对这个 trunk 是必需的（+0.13）。

**评测协议（所有行相同）。** target 的 B 个有标签 trial 为 `draw_trials(target, B, draw)` 随机抽取（与 P1 的 crc32 规则相同，所有变体共享同一批 draw；每个 (target, B) 8 个 draw），统计量/归一化只来自这 B 个 trial 的有效 bin；整个 session 做一次连续因果解码（状态不重置，rule 7），query = 其余 usable trial 的全部有效 bin，R² 为物理单位、方差加权，`check_physical_units` 全部通过。候选行在 `torch.no_grad` 下运行，H2 反传次数逐行计数（全部为 0）；H1 trunk 哈希逐行核对（不变）；H4 query 与校准 trial 不相交（逐行核对，全部为真）；`flags.json` 汇总。选择（训练早停、R1 的 λ/kind/burn、ui_reg 的 λ/lr/steps、scale 基线）全部只用 source session；targets 只来自 `D2_POOLS`（held-in），没有使用 held-out/final。

**source 模拟。** M2 的 source session 对 trunk 是 in-sample，且 pool1 只有 2 个 source。因此模拟 episode 只用 source session 内部的 trial 划分：校准 trial 取自前 80% usable trial（trunk 自己的 train 部分），计分的是最后 20% trial（trunk 自己的 validation trial，从未被梯度使用，只用于 trunk 的 checkpoint 选择）；source 的 prior 是 leave-one-out 的均值表（均值取自**其他** source session 的表与 bias）；解码仍是整段连续因果。这是 within-session 的时间泛化，**没有真正的跨 session 漂移**。后果见 §3.1（scale 基线）与 §6。

**P0（可部署）。** `M2Calibrator`（`PopulationCalibrator` 的子类）：独立的 per-channel 编码器，所有通道共享权重，输入是 P1 的 `describe` 描述量（log rate/std、3 个 autocorr、与 prior latent 的相关、behavior 互协方差 5 个 lag × D=2、8 方向分箱、T4 系数、prior 列、log n、log B），但 prior latent 先投影到 K=256 维 prior 的 **前 16 个左奇异向量**（固定的 source 基，描述量 59 维），输出每个通道的 K 维残差加一个标量 log-gain：`T_i = (1 + a_i)·prior_i + res_i`（最后一层零初始化 ⇒ 未训练时严格等于 prior，有测试）。与 P1 的 P0 的偏离（均为 M2 适配）：(1) 输出 K+1 维而不是 K 维，因为 K=256 时 16 维 prior 坐标不足以重构 `α·prior_i`；(2) prior latent 降到 16 维；(3) unit dropout 率 U(0, 0.1)（trunk 训练时 channel dropout 0.05）；(4) 训练 episode 取自上述 source 模拟。其余同 P1：episodic 训练，B 从 log-uniform{1…33} 抽取，leave-one-out prior，256 bin 窗口前 64 bin burn-in，MSE 经冻结 trunk，AdamW lr 1e-3，每 100 步做 source-val（B∈{1,2,4,8,16,33}×2 draws），patience 8、至少 600 步，best step 落在最后 20% 就翻倍预算（最多 4 次）。参数量 24,705（+ 固定的 256×16 投影 4,096）。**calibrator seed：每个 (backbone seed, pool) 训练 2 个**（c0、c1），行内取平均；所有 18 个 run 都满足收敛规则（`train_summary.csv`）。消融（仅 backbone seed 0，c0）：`p0_nogain`（无 log-gain 输出）、`p0_nolab`（描述量中去掉所有标签相关的列，与 P1 的 `_nolab` 相同）。

**R1-stream（可部署）。** 与 P3 相同的 readout 先验中心化岭回归 `V = V0 + R(G+λI)⁻¹`（特征 h 是 `final_norm` 之后、线性 readout 之前的 256 维加 bias，257 维），统计量 G、R 按到达顺序逐个 trial 在线累加：第 j 个 trial 只前向一次，用由前 j−1 个 trial 得到的 P0 表与归一化（j=1 用 prior 表与该 trial 自身统计量，需要一个 trial 大小的暂存区），**最后用 B 个 trial 的 P0 表做连续因果解码**。规则 (λ, kind, burn) 按 B 在 source 模拟上选择：λ∈{1e-3,…,100}（6 档）∪ {关闭}，kind∈{full, diag}，burn∈{0,10,20}（每个 replay trial 前几个 bin 不计入统计量，用来抵消零初始状态瞬态）。“关闭”是候选之一，所以如果模拟里岭回归没有帮助，R1 就退化为 P0。**已知偏差（P3 的 §R 中记录过）：**统计量来自“每个 trial 从零状态前向”，而评测是整段连续解码，两种状态机制不同；M2 的 trial 很短（50–110 bin），burn-in 占比更大。没有做 contig 变体。另有 `p_prior_r1s`（在 prior 表上做同样的 R1，规则单独在 source 上选）。λ 的敏感度行 `p0_r1s_lam*`（固定 λ，full，burn 取所选值）**是在 target 上评测的诊断行，从未用于选择**。

**ui_reg（非部署参照）。** `loss = MSE(trunk(x_z Tᵀ + b), y) + λ‖T − T_prior‖²_F`，Adam，每步随机取一个标注 trial，只更新 read-in 表，bias 冻结（与 `ui_reg.ui_reg_trajectory` 的配方相同）。Mamba-3 的 Triton 反向对每个新序列长度重新 autotune，所以把每个 trial 窗口在**末尾**补零到 64 的倍数（trunk 是因果的，补零不影响窗口自身的输出；有测试证明轨迹不变）。(λ, lr, steps) 按 B 在 source 模拟上选择：λ∈{0, 0.01, 0.1}，lr∈{1e-3, 3e-3, 1e-2}，steps∈{10, 30, 100, 300}（同一条轨迹取快照），episode = **每个 pool 最近的 2 个 source session × 每个 B 1 个 draw（即每个 B 只有 2 个 episode）**，prior 为 leave-one-out。这个规模比 P2（12 个 episode/B）小得多，因此选择噪声很大（例如 seed 0 的 B=2 的 ui_reg 反而低于 B=1，很可能是选择噪声，见 `uireg_selection.csv`）；而且 Mamba-3 反向在 24,576 个参数上约 33 ms/步，网格不能更大。第一次试跑（更大的 λ/lr 网格、全部 source）因 Triton 对变长序列的 autotune（没有 padding）和 GPU 争用而放弃，其部分结果存放于 `results/m2_calib_20261016/uireg_v1_discarded/`，**没有用于任何表**。

**其他对照行。** `p_prior_scale`：prior 表乘一个标量 c(B)（c 从 {0.8,…,4.0} 中按 B 在 source 模拟上选）——用来检验 P0 是否只是学到了一个整体缩放；`p0_int8`：P0 表做逐行对称 int8 量化后解码；`p0_swap`：用“最近一个 source session 的 B 个 trial”（target 无关的信息）生成 P0 表，再用 target 自己的 B-trial 归一化解码，是“增益是否 target 特异”的对照（诊断行）。

## 3. 结果

统计单位 = **target**（5 个，分属 3 个 pool）；一个 target 内先对 8 个 draw、calibrator seed（和 backbone seed）取平均，再对 target 做 t 区间（df = 4）。draw 之间并不独立，所以不拿 draw 层面的 CI 当证据；`paired_draw_level.csv` 仅给出尺度。另给 (backbone seed × target) 单元的符号翻转检验（精确，2^15）——这些单元共享 target，不独立，只是灵敏度视图。完整表见 `tables_m2.md`（含逐 target、逐 pool、噪声底、训练/选择摘要）。

### 3.1 主表（backbone seed 0；5 个 target；mean ± 95% CI over targets）

见 §0 第 2 点的表；补充行（seed 0）：

| B | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| p_prior_nobias（第一轮式解码） | 0.212±0.109 | 0.222±0.111 | 0.244±0.106 | 0.255±0.092 | 0.263±0.099 | 0.273±0.098 |
| p_prior_srcnorm | 0.174±0.163 | 0.173±0.162 | 0.174±0.164 | 0.174±0.164 | 0.173±0.160 | 0.174±0.166 |
| p_prior_scale（source 选 c） | 0.259±0.171 | 0.262±0.190 | 0.291±0.157 | 0.297±0.149 | 0.293±0.159 | 0.304±0.159 |
| p_prior + R1-stream | 0.273±0.146 | 0.286±0.144 | 0.299±0.137 | 0.311±0.131 | 0.314±0.130 | 0.341±0.102 |
| p0_int8 | 0.293±0.163 | 0.317±0.170 | 0.334±0.165 | 0.339±0.157 | 0.345±0.164 | 0.351±0.164 |

* **p_prior_scale 比 prior 低**（seed 0：−0.010…−0.024，只有 1–2/5 个 target 为正；三 seed 平均 −0.007…−0.016）：source 模拟选出的 c 在各 (seed, pool) 之间差别很大（pool1 的 2-source 模拟给出 c = 1.2–4.0，source 增益 0.00–0.05；其余 pool 的 c 在 0.8–2.0，source 增益 ≤ 0.013），在 target 上并不迁移，pool1 最差（−0.02…−0.06）。这是“source 模拟（within-session，LOO prior）与跨 session target 之间有差异”的一个直接证据，也说明 P0 的增益不是一个整体缩放（P0 − p_prior_scale：seed 0 为 +0.034…+0.055，5/5 个 target；三 seed 平均 +0.023…+0.039，12–15/15 个单元为正）。P0 学到的 log-gain 均值在 +0.13…+0.49（B 越小越大，通道间 SD 0.05–0.23；一个 pool×B 的 seed 0 检查，非全量统计）、残差的 Frobenius 范数约为 prior 的 29%–49%，所以它不是一个小修正，但修正的方向主要是在 source 上学到的。
* **p0_int8 ≡ p0**（|差| ≤ 1e-4）。

### 3.2 三个 backbone seed 的平均

| B | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior | 0.286±0.141 | 0.302±0.133 | 0.315±0.129 | 0.317±0.123 | 0.324±0.125 | 0.330±0.121 |
| P0 | 0.296±0.156 | 0.324±0.159 | 0.337±0.153 | 0.341±0.150 | 0.350±0.154 | 0.356±0.152 |
| P0 + R1-stream | 0.296±0.157 | 0.322±0.161 | 0.333±0.148 | 0.337±0.146 | 0.345±0.150 | 0.366±0.129 |
| p_prior + R1-stream | 0.285±0.141 | 0.300±0.134 | 0.306±0.118 | 0.315±0.120 | 0.309±0.134 | 0.338±0.100 |

配对差（3 seed）：

| 对比 | B=1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P0 − P-prior | +0.011±0.022 | +0.021±0.027 | +0.022±0.026 | +0.024±0.030 | +0.026±0.032 | +0.026±0.033 |
| 为正的 target 数 | 4/5 | 4/5 | 3/5 | 4/5 | 4/5 | 3/5 |
| 为正的 (seed, target) 单元 | 10/15 | 12/15 | 11/15 | 12/15 | 12/15 | 13/15 |
| 符号翻转 p（单元，非独立） | 0.080 | 0.006 | 0.005 | 0.005 | 0.004 | 0.005 |
| per-pool（pool1/2/3） | −0.004/+0.026/+0.010 | −0.002/+0.034/+0.043 | −0.001/+0.038/+0.035 | −0.002/+0.040/+0.043 | −0.002/+0.045/+0.043 | −0.003/+0.047/+0.042 |
| P0+R1s − P0 | −0.000±0.001 | −0.001±0.005 | −0.003±0.009 | −0.004±0.005 | −0.005±0.007 | +0.011±0.024 |

噪声底（`noise_floor.json`，per-target cell 的 seed 间 SD）：calibrator seed 0.0073（P0）、0.0076（P0+R1）；backbone seed 0.0185（P0）、0.0186（P0+R1）、0.0239（P-prior）。

### 3.3 P0 的增益来自哪里（seed 0 的消融与对照）

| 对比（seed 0） | B=1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| p0_nolab − p0 | −0.023±0.022 | −0.014±0.013 | −0.018±0.007 | −0.020±0.013 | −0.021±0.019 | −0.020±0.026 |
| p0_nogain − p0 | −0.002±0.027 | −0.003±0.027 | +0.003±0.022 | +0.003±0.025 | +0.004±0.021 | +0.003±0.019 |
| p0_swap − p_prior | −0.016±0.040 | +0.004±0.023 | +0.013±0.018 | +0.021±0.016 | +0.022±0.019 | +0.021±0.022 |
| p0 − p0_swap | +0.035±0.044 | +0.027±0.051 | +0.019±0.044 | +0.011±0.037 | +0.010±0.035 | +0.011±0.034 |

* **标签有用**：去掉标签相关的描述量，P0 损失约 0.02（B=2…16 时 5/5 个 target 为负）；去掉标签后 P0 比 prior 只高 −0.003…+0.018（B≥2 为 +0.011…+0.018），不显著。
* **log-gain 输出不起作用**（nogain ≈ p0）。
* **p0_swap**：用另一个 source session 的统计量生成的表在 B≥8 时已经比 prior 高 +0.021（5/5…4/5），且 p0 − p0_swap 只有 +0.01，CI 包含 0；B≤2 时 swap 的表被噪声破坏（B=1：−0.016），而 p0 用 target 自己的统计量则没有这个问题（+0.035）。所以：B≥8 的增益大部分是 P0 把 source 上学到的“prior 校正”静态地加进来了；target 特异的信息主要体现在小 B 时避免噪声（和标签相关的 ≈ 0.02）。

### 3.4 ui_reg 与梯度参照

seed 0（5 个 target，每个 (target, B) 8 个 draw）见 §0。配对（seed 0）：ui_reg − P-prior = +0.009 / −0.012 / +0.010 / +0.027 / +0.044 / +0.060（B=1…33；B≥8 为 5/5 个 target 为正）；ui_reg − P0 = −0.010 / −0.044 / −0.023 / −0.005 / +0.012 / +0.028（B=33 为 4/5 个 target 为正，CI ±0.03）。所选规则见 `uireg_selection.csv`：lr 多数选在 3e-3，steps 在 30–300，λ∈{0, 0.01, 0.1}；每个 B 只有 2 个 source episode，选择有噪声（见 §2 与 §6）。**三个 backbone seed 都跑了 ui_reg**（seed 1/2 每个 (target, B) 4 个 draw，选择规则各自在该 backbone 的 source 上重选；配对差只在两行共有的 draw 上计算）。三 seed 平均：ui_reg 的 R² 为 0.297 / 0.299 / 0.319 / 0.330 / 0.353 / 0.367（B=1…33；`ui_reg` 单元格的 draw 数与其他行不同，仅作参考），ui_reg − P-prior = +0.006 / −0.005 / +0.006 / +0.017 / +0.032 / +0.042（B≥8 为 5/5 个 target 为正；(seed, target) 单元 12–14/15），ui_reg − P0 = −0.004 / −0.029 / −0.018 / −0.010 / +0.005 / +0.015（CI ±0.02–0.05，全部包含 0，B=2 的符号翻转 p=0.013 是唯一的“P0 好于 ui_reg”）。也就是说梯度参照只在 B≥16 才超过 prior 一个可见的量（+0.03–0.04），与 P0 在统计上不可分；amortization gap（ui_reg − P0）在 M2 上 B≤8 为 0 或为负，B=33 约 +0.015（三 seed）/ +0.028（seed 0）。

### 3.5 与第一轮 M2/D2 结果的关系

第一轮（`docs/labeled_calibration_results_20261007.md`，存档行，**未重新评测**）：F1（label-free）0.077…0.265，F2 0.064…0.268，F3（residual）0.231…0.291，F7（error-driven）0.201…0.297（B=1…33，3 个 pool 共 5 个 target 的均值）；第一轮自己的 source-only prior 为 0.265（b=0）/ 0.285（B=8）/ 0.293（B=33），结论是 F7 与 F3“在 prior 水平，不在噪声底之上”。协议差别：(a) 第一轮的校准 trial 是 session 开头的前 B 个 trial，本轮是随机 draw；(b) 第一轮**丢掉了 in_proj 的 bias**（§2），其 prior 与本轮的 `p_prior_nobias`（B=8：0.255，随机 draw）同属“丢 bias”水平，而不是本轮主表的 0.27–0.32；(c) 本轮没有重新评测第一轮的方法，因此 F3/F7 与本轮 P0 的数字不能当作同协议的头对头比较。可以说的只有：第一轮“没有任何东西稳健地超过 source-mean prior”的结论，在本轮更强的（带 bias 的）prior 之上依然成立——最好的可部署行 P0 只有 +0.02–0.03，且不稳健。

## 4. 对问题的回答与对 DAC 主线的含义

* 固定通道的 M2 上，prior（source 均值表 + B-trial 输入归一化 + 冻结 bias）已经占据了几乎全部可得的 R²：梯度参照 ui_reg 相对 prior 在 B=8 只有 +0.027（三 seed +0.017）、B=33 +0.060（三 seed +0.042）；可部署的 P0 +0.03；R1-stream 为 0。**这是与 000688 的 days/month/year（那里 prior 只有 0.13–0.36，P0 把它提到 0.26–0.72）本质不同的体制**：身份对齐 + 固定通道使 prior 本身就很强，剩下的是 trunk 本身的上限。
* “在噪声之上”的严格回答：没有。P0 的 +0.02–0.03 与 backbone-seed 噪声底同量级，target 层面 CI 包含 0，且 pool1 上为 0。它不是 B 的函数（B=2 起不再增长），一半以上的部分对标签依赖不大（nolab 去掉 0.02，剩下的 +0.01 不显著）。这里的“校准”主要是一个在 source 上学到的静态 prior 修正，而不是对 target 的适应。
* R1-stream 在 M2 上没有价值；它的 SRAM 代价（full：257×257 fp32 Gram = 264 KB，加 R 的 514 个数；diag：约 1 KB；还要一个 trial 大小的暂存区）也远大于它带来的收益。
* 表大小：fp32 的 K×N = 96 KB 不满足 32 KB；int8（25 KB）满足。如果最终的硬件把 M2 类通道固定的数据集放在 32 KB 的 SRAM 里，int8 表是必要的，P0 的 int8 量化是无损的（post-training，逐行 scale）。

## 5. 预算与资源

* 起始余额 199.89 CU（`colab usage`，16:00 JST 前后），本报告写作时的余额与最终值见 §5.1。session `dac27` 从约 14:56 JST 开始；速率 1.54 CU/h。
* GPU 作业：P0 训练（18 个 + 6 个消融）、R1 选择、评测、scale 选择、ui_reg 选择与评测；同时最多 3–4 个进程共享 L4。Mamba-3 反向 33 ms/步（单进程），三个 ui_reg 进程并行时约 130 ms/步（GPU 饱和）。
* **5.1 最终 CU**：起始余额 199.89 CU → 结束余额 **195.53 CU，本轮实际消耗约 4.4 CU**（`colab usage` 的余额更新可能略有滞后），远低于 40 CU 的上限；session `dac27` 已 `colab stop`（速率回到 0.00/h，active assignments 0）。session 总时长约 4 h（14:56–19:00 JST），没有触及 12 h 限制。

## 6. 局限与未做的事

1. **source 模拟不含跨 session 漂移**（within-session 的时间划分 + LOO prior）。p_prior_scale 在 source 上选出的 c 在 target 上不迁移就是一个症状；P0 的 early-stopping、R1 的 λ、ui_reg 的规则都是在这种模拟下选的。pool1 只有 2 个 source，LOO prior 只是另一个 session 的表。真正的“漂移匹配”的模拟（P3 的教训）在 M2 上没有可用的 OOS source session（所有 held-in session 要么是 source 要么是 target）。
2. **target 只有 5 个、来自 3 个 pool**（pool1 的两个共享 trunk；pool3 只有一个）。t 区间 df=4，非常保守，但结论“增益 ≈ 噪声”对 pool 的选择敏感（pool1 为 0，pool2/3 为 +0.03–0.05）。
3. **ui_reg 的选择只有 2 个 source episode/B**、lr 常选到网格上限附近；该参照行的绝对水平有噪声（不是上界）。
4. **R1-stream 的零初始状态 replay** 与连续评测状态机制不一致（P3 的已知偏差），M2 的短 trial 使它更严重；没有做 contig、cap、低秩（lr16）readout 变体，也没有评测量化的 R1 统计量。
5. **P0 的 bias 不校准**（冻结的 K 维常数）；R2（F7-J）没有在 M2 上运行（任务未要求）。
6. **没有硬件代价建模**（只报告了张量字节数），也没有逐步定点模拟（rule 8）。表字节数是静态张量大小，不是代价结论。
7. 没有请独立 Opus 评审；自查项：候选路径 no_grad + 反传计数、query 与校准 trial 不相交、trunk 哈希、因果性（解码对未来扰动不变，有测试）、选择只用 source。
8. 本轮没有使用 M2 的 minival / held-out、也没有读 `results/official_evalai/`；000688 bundle 中的 final session（0623/0625/0626）NWB 随 bundle 解压到了 VM 上但没有被读取（parity 只用 days 的 0616/0617）。

## 附录 A：自动生成的全部表（`scripts/analyze_m2_calib.py`）

见同目录 `tables_m2.md`。
