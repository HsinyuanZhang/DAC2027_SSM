# 架构级分析：冻结算法的算子、带宽、CiM 与存储（2026-10-19）

对象：MZ 2026-10-19 冻结的算法 —— **P0**（electrode-aligned prior + per-unit 固定权重 encoder，统计量按前缀流式累加）+ **W_α**（向 ridge-CCA 表收缩，α 由 expanding-window 协议选）+ **S1-causal re-based 固定反馈精修**（合称 COMBO；备选 CCA+S1），协议为 causal-first-B + anytime commit + 整段连续解码（`docs/paper_mainline_proposal_20261016.md` §6）。本文回答贡献 2 的四个问题：有哪些算子、流式输入下各处带宽多大、CiM（MVM）怎么用且占多少、特殊算子怎么优化；并更新已暂停的 `docs/hw_opportunities_qa_20261016.md`（该文已标注为被本文取代）。

**边界声明（先读）**
- 流水线上所有精度结果（`results/analysis/causal_main_20261019/README.md`）都是 **fp32 评估，没有定点或器件仿真**【F】。本文的 MAC、字节、SRAM 都是解析计数或 CPU 上的小测量，**不是硬件测量，也不是 RTL**。
- 只有 N≈17–26 的 000688 sorted-unit 数据评估过。N=96（Utah array 电极级计数）、N=256（HD-sEMG 规模）是把同一结构（K=10、D=2）外推，**N 相关的结论是缩放推演，不是实验**。
- 不改任何代码。本文中的 CPU 测量脚本留在 `/tmp/arch_scratch`（易失；如 MZ 同意，建议整理后放进 `scripts/`），见附录。

**来源标签**：【C】代码（file:line）；【R】已审查的 README 数字；【D】本文用【C】/【R】推导的算术（公式在文中）；【M】本次在 CPU 上测得；【F】只在 float 下验证过。

---

## 0. 十条结论（数字详见后文）

1. **推理 per bin（N=26）= 418,052 MAC**：RRAM 可映射的 trunk block 层 344,064（**82.3%**）、其余数字端 weight×act 6,656（1.6%）、SRAM 表 read-in 260（0.06%）、activation×activation 67,072（16.0%，几乎全是 SSM 状态递推 65,536）。N 从 26 到 256，CiM 占比只从 82.3% 降到 81.9%——**read-in 不是主要算术**。【C】【D】
2. **非线性算子 1,209 次/bin（60.5 k/s）**，其中 SiLU 1,024（85%）、sin/cos 128、tanh 16、softplus/sigmoid/exp2/倒数各 8、rsqrt 9。没有 softmax，没有 conv。【C】【D】
3. **trunk 完整状态 17,216 个值**（ssm 16,384 + K carry 256 + V carry 512 + angle 64），INT24 为 50.4 KiB（`ssm_decode/state_budget.py` 同数）。旧成本模型的 `state_elems=16,384` 漏掉 832 个 carry 值；`scan_macs=32,768` 低估了真实递推（trapezoid 有两个外积、外加衰减乘法，实际 65,536）。【C】
4. **带宽瓶颈是 SSM 状态读写**：INT24 下 103 kB/bin = 5.2 MB/s，占 N=26 总 SRAM 流量的 ~79%；其余（trunk 激活 1.0 MB/s、校准累加与 FIR 合计 0.3 MB/s、read-in <0.02 MB/s）都小一到两个数量级。输入侧反而最便宜：96 通道原始 5.76 MB/s → 计数 36 B/bin（1.8 kB/s）。N=256 的连续特征下，Gram 的 read-modify-write（15.5 MB/s，28 bit）反超状态。【D】
5. **算力不是约束，能耗和 SRAM 才是**：每 bin 只需 ~0.42 M MAC（≈21 MMAC/s），量级上一个 25 MHz 的单 MAC 引擎已接近满载，实际 2–4 路并行即可有余量；校准的 per-trial 提交（~5 s 一次）更宽松。
6. **CiM 的账要按 ADC 记**：混合精度（z/x、out_proj 8 bit，B/C 与 FFN ternary）需 **1.45 Mi 个 2T2R 对 / 24 个 256×256 阵列（致密打包）**；每 bin **78,848 次 ADC 转换**（bit-serial 8 周期），平均每次转换只摊到 4.4 个 MAC。**CiM 能耗的主项是 ADC，不是 MAC**。PTQ 研究没有建模 ADC。【D】
7. **P0 encoder 放不放 RRAM 对功耗几乎无差**：7,744 个权重，每次提交复用 N 次（201 k MAC/commit，≈0.04 M MAC/s），而 trunk 每秒 20.9 M MAC。encoder 进 CiM 是面积/统一性选择，不是功耗收益。校准的 CiM 占比：累加阶段 0%，提交阶段（不含 CCA）91.6% / 78.0% / 58.3%（N=26/96/256）。【D】
8. **真正的“重”算子不在 P0，而在 W_α 的 ridge-CCA 提交**：d_ref=43 时约 4.7 M flop 级的 eig/SVD/求逆（对 P0 提交的 ~0.22 M MAC 约 21 倍，flop 对 MAC，只作数量级比较），不规则、需要 sqrt/除法、不是 CiM 友好；α=0（B=1）时可整条旁路。此前的成本表（P2、P3、stream_f7、p0solve）都没有计入它。【D】
9. **峰值 SRAM（N=26，INT24 状态，fp32 求解工作区，按代码 d=43）= 158.5 KiB，超过 128 KiB**：常驻 81.0 KiB（推理 60.4 + 校准 20.6），CCA 提交再加 77.5 KiB 工作区。旧表里的 44.7–57.6 KB（S1-causal）只含 P0 前缀统计，不含 CCA 的 cond/phase 累加器（12.2 KiB）和求解工作区。若 CCA 的目标侧维数取 min(N, d_ref)=26（**未评估**），峰值降到 ~121 KiB。N=96 的 CCA 分支（工作区 356 KiB）和 N=256 的 N×N Gram（28 bit 下 139 KiB）都放不下。【D】
10. **算法–硬件有一处不一致必须先定**：S1 的评估前向对每个校准 trial 从零状态单独跑（`rebased_feedback_loop`），而部署的推理是整段连续、状态跨 trial 携带。要让 S1 的残差“免费”来自推理，需要用 live 残差（未评估）；否则要加一次 shadow forward（校准期间 trunk MAC ×2，外加第二份 ~50 KiB 状态）。见 §7。

---

## 1. 算子清单

### 1.0 冻结流水线与三个阶段

```
 spikes ─▶ bin(20 ms, 3-bit count/unit) ─┬─▶ [read-in MVM: z = T_int8·x + b] ─▶ trunk ─▶ ŷ_t      (i) 推理，每 bin
                                         │        ▲ 双缓冲表（SRAM）
                                         └─▶ (ii) 校准，只在前 B 个有标签 trial、每 bin
                                              P0 前缀统计 · CCA staging · 活残差 r_t=y_t−ŷ_t · S1 FIR(Jbar, ring)
        trial 边界 ─▶ (iii) 提交 k=1..B:
              describe() → P0 encoder(N×) → [CCA: eig/SVD](α>0) → W_α 混合 → S1 更新 → 折叠 μ,σ → int8 → 切表
```

约定：K=10，D=2（光标速度），trunk width 128、2 层、d_state=32、nheads=4、headdim=64、d_inner=256、FFN 256（从两个 checkpoint 读出，二者结构相同，353,554 参数：`results/stage2_intervals_20261006/month_pool_k10_x8/best.pt`、`results/stage2_688_20261005/ap_pool_x8/best.pt`【M】；`in_proj.weight` 为 (596,128)=2·256+2·32+3·4+8；`mamba3_official.py:51-57` 的 `_PreNormMamba3Block`）。**Mamba-3 没有 short conv**；无 softmax。MAC=一次乘加；calibration 的 trial 平均 252 bin（5.04 s）【R：population_calib_p3 README】；最长观测 trial：0616 为 2,741 bin【R】，0617 为 5,144 bin【M】。

### 1.1 (i) 推理，每 bin（50 Hz）

trunk 的 per-bin 递推取自官方 step 核（`.tools/mamba_official/.../mamba3_siso_step.py:150-219`）与仓库内的纯 PyTorch 单步参考（`ssm_decode/mamba3_streaming.py:219-283`），两者一致；块结构见 `mamba3_official.py:51-57`。下表数量为**两层合计**。

| 算子 | 形状 | 次数/bin | 类别 | MAC / ops |
|---|---|---:|---|---:|
| read-in MVM（int8 表 × 3-bit 计数）| 10×N | 1 | SRAM-table MVM | 10N = 260 / 960 / 2,560（事件驱动见 §1.5）|
| lift（root）| 10→128 +bias | 1 | weight×act（PTQ 研究里 root 层放 SRAM INT8）| 1,280 |
| LayerNorm（norm1、norm2 各 2 层 + final）| 128 | 5 | reduction + elementwise + rsqrt | ≈512 ops 每个（均值、方差、归一化、仿射）|
| in_proj z、x 行 | 128→512 | 2 | weight×act（**CiM，8 bit**）| 131,072 |
| in_proj B、C 行 | 128→64 | 2 | weight×act（**CiM，ternary**）| 16,384 |
| in_proj dt、A、trap、angle 行 | 128→20 | 2 | weight×act（数字端：PTQ 研究把这 20 行留在数字，因其喂 exp/softplus）| 5,120 |
| B_norm、C_norm（RMSNorm，32 维，一个 group 广播到 4 头）| 32 | 4 | reduction + rsqrt | 4 rsqrt，~96 ops 每个 |
| Q/K bias + RoPE 旋转（8 对 × 4 头 × {B,C}）| 8 pairs | 2 | act×act | 512 MAC（每对 4 乘 2 加）|
| dt = softplus(·+dt_bias)；A = −heavy_tail(·)；ADT=A·dt；α=exp2(ADT·log2e)；β=α·dt·(1−σ(trap))；γ=dt·σ(trap) | 4 头 | 2 | 非线性 + elementwise | softplus 8、倒数 8、exp2 8、sigmoid 8 |
| angle 链：tanh(8 个原始角，4 头共享)·π·dt，累加，mod 2π，sin/cos | 4×8 | 2 | 非线性 | tanh 16、sin 64、cos 64、mod 64 |
| **SSM 状态更新** h←α·h + β·(v_prev⊗k_prev) + γ·(v⊗k) | 4×64×32 / 层 | 2 | **act×act** | **49,152**（3 MAC/元素）|
| **SSM 读出** y=C·h | 同上 | 2 | act×act | 16,384 |
| D skip、门控 y·SiLU(z) | 256 | 2 | act×act + 非线性 | 512 + 512 MAC，SiLU 512 |
| out_proj | 256→128 | 2 | weight×act（**CiM，8 bit**）| 65,536 |
| FFN0 | 128→256 | 2 | weight×act（**CiM，ternary**）+ SiLU 512 | 65,536 |
| FFN2 | 256→128 | 2 | weight×act（**CiM，ternary**）| 65,536 |
| 残差加 | 128 | 4 | elementwise | 512 |
| final out_proj（readout）| 128→2 | 1 | weight×act（数字）| 256 |

【C】【D】。汇总（N=26 / 96 / 256）：

| 类别 | MAC/bin | 占比 |
|---|---:|---|
| CiM 映射层（PTQ 研究的 RRAM 放置：z/x、B/C、out_proj、FFN）| 344,064 | 82.3% / 82.2% / 81.9% |
| 其他 weight×act（root、dt/A/trap/angle 行、readout）| 6,656 | 1.6% |
| SRAM-table MVM（read-in）| 260 / 960 / 2,560 | 0.06% / 0.23% / 0.61% |
| act×act：SSM 递推 65,536 + D/门控/RoPE 1,536 | 67,072 | 16.0% |
| 合计 | 418,052 / 418,752 / 420,352 | |

若把所有 weight×act（含 root）都算作“可固定权重”：83.9% / 83.8% / 83.4%。elementwise/reduction（LN、RMSNorm、残差、bias、angle 算术）约 4 k ops/bin（~1%）。**非线性共 1,209 次/bin**：SiLU 1,024、sin 64、cos 64、tanh 16、softplus 8、sigmoid 8、exp2 8、倒数 8、rsqrt 9（5 个 LN + 4 个 RMSNorm）。50 Hz 下 ≈ 60 k 次/s，LUT 或小型迭代单元即可；SiLU 占 85%，是优化的第一目标。

**与旧成本模型的差异**：`fwd_refine.trunk_macs_per_bin`（`ssm_decode/fwd_refine.py:301-326`）的 scan 项 = 2·d_inner·d_state·层数 = 32,768（更新、读出各 1 MAC/元素）。官方递推实际每元素 3 个乘加（α·h，β·v_prev⊗k_prev，γ·v⊗k）+ 读出 1 个 = 4，共 65,536（`mamba3_siso_step.py:186-200`、`mamba3_streaming.py:255-258`）。此外旧模型把 `state_elems` 记为 16,384，漏了 K/V/angle 三个 carry（832 个值；`state_budget.py:25-35` 已含）。旧模型的 fixed-weight 350,720 与本文一致。

### 1.2 (ii) 校准，每个有标签 bin（只在前 B 个 trial，B≤33；N=26 dense | 事件驱动实测）

事件驱动数字来自 5 个 sub-M session 的实测稀疏度【M】：20140626（N=26）零计数占 83.5%，每 bin 平均 4.45 个非零 unit（0.171·N）；其余 4 个 session 的非零占比 0.175–0.207；最大计数 5–7，>3 的 bin 占比 ≤1.6×10⁻⁴，>7 为 0（所以 3-bit 饱和计数足够，与【R】P3 一致）。**N=96/256 没有对应的稀疏数据**，下表只给 dense。

| 算子 | 形状 | 类别 | N=26 dense | N=26 事件驱动 | N=96 dense | N=256 dense |
|---|---|---|---:|---:|---:|---:|
| Gram xxᵀ（对称）| N(N+1)/2 | act×act | 351 | 14.0 | 4,656 | 32,896 |
| 自相关 x_t·x_{t−l}，l∈{1,2,5} | 3N | act×act | 78 | 5.0 | 288 | 768 |
| 互协方差 y_{t+l}·x_t，5 lag × D=2 | 10N | act×act（16-bit 标签 × 3-bit 计数）| 260 | 44.5 | 960 | 2,560 |
| sphix（cos/sin 两行）| 2N | act×act | 52 | 8.9 | 192 | 512 |
| **P0 累加合计（MAC）**| | | **741** | **72** | **6,096** | **36,736** |
| 加法型累加（sx、ac_a+ac_b、sdir、sy）与 17 个计数器 | ~8N+27 | adds | ~235 | | | |
| 活残差 r_t=y_t−ŷ_t（含 valid 掩码）| D=2 | elementwise | 2 | | | |
| S1：Jbarₗᵀ·r（ROM 核）| L=33 lag × D×K | weight×act | 660 | 660 | 660 | 660 |
| S1：外积 v_l⊗x_{t−l} → U（**direct 形式**）| 33·K·N | act×act | 8,580 | 1,469 | 31,680 | 84,480 |
| S1：（**reorder 形式**）corr_l += r·x_{t−l}，lag 各一份 D×N | 33·D·N | act×act | 1,716 | 294 | 6,336 | 16,896 |
| 　　reorder 的 trial 末尾：u=Σ_l Jbar_lᵀ·corr_l | 33·K·D·N（每 trial 一次）| weight×act（ROM）| 17,160 | | 63,360 | 168,960 |
| 每 trial 的 Σxz²（q）| N | act×act | 26 | | 96 | 256 |

【C】（形状来自 `population_calib.py:58-104`、`calib_cost.py:45-63`、`stream_f7.py:89-111,645-652`）【D】【M】。

要点：
- **P0 统计里 4 个 q_* 键、可靠性特征（fano、kappa）对冻结的 P0 是死代码**：`PopulationCalibrator.forward` 在 `variant=="p0"` 时只用描述向量 `d`，`r` 不进网络（`population_calib.py:411-416,434`）。硬件可以不累加 q_*（省 N+2DN+D 个字和每 trial 的 scratch）。`sphi/sphiphi` 与 `sphix[0]` 已在 `calib_quant.py:46,176-185` 里作为 derived 处理。`ac_a` 与 `ac_b` 在 `describe` 里只以和的形式出现（`population_calib.py:232`），可以合并成一个累加器（【D】精确恒等，未实现）。按此，P0 最小统计量为 N(N+1)/2+30N+10 个数据字+17 个计数器：N=26 为 1,141 字。
- **S1 的 FIR 有两种精确等价的求和顺序**，`fir_data_term`（`stream_f7.py:89-111`）实现的是 reorder 形式（先对每个 lag 做 D×N 的相关，最后乘 Jbar）。`stream_f7` README 与成本模型记的 8,580 act×act/bin 是 direct 形式。reorder 把 per-bin act×act 降为 1/5（K/D），代价是累加器从 K×N 变为 33·D·N（N=26 时 1,716 字 vs 260 字，6.6×）和 SRAM 流量上升（§2）。这是一个 **MAC–SRAM 的真实取舍**。
- 稀疏性：Gram 的 outer product 事件驱动后是 14.0 次，dense 的 351 次的 1/25；P0 累加整体 72 vs 741（10.3×）。

### 1.3 (iii) trial 边界提交，每次（k=1..B，每个 session ≤B 次；N=26 | 96 | 256）

| 算子 | 形状 / 公式 | 类别 | N=26 | N=96 | N=256 |
|---|---|---|---:|---:|---:|
| **describe()** 矩阵部分（gn·Pᵀ 及归一化）| N×N · N×K 等 | SRAM-table / act×act | 15,185 | 197,115 | 1,385,755（MAC；`calib_cost.finalize_macs`，`calib_cost.py:71-77`）|
| 　【D】最小重写 | zvar 由 cz 得到（K·N²→K·N）；不形成 N×N 的 gn；1/σ 预算 | | ≈8.6 k | ≈99 k | ≈674 k（主项 N²K）|
| describe 的非线性 | log(mean+ε)、log(std+ε)：2N+2；sqrt：2N；corr 的 rsqrt：N·K；除法 ~25N+N·K | log / sqrt / rsqrt / div | log 54；sqrt/rsqrt ≈312；div ≈900 | 按 N 缩放 | 按 N 缩放 |
| **P0 encoder MLP**（每 unit：47→64→GELU→64→GELU→10）| 7,744 权重 / 7,882 参数 | **weight×act（CiM 候选，复用 N 次）** | 201,344 | 743,424 | 1,982,464 |
| 　GELU | 128 / unit | 非线性 | 3,328 | 12,288 | 32,768 |
| 　d_mu/d_sd 归一化、d_keep 掩码 | 可精确折进第一层权重 | 融合 | 0 | 0 | 0 |
| 表 = prior + encoder 输出 | K×N | elementwise | 260 | 960 | 2,560 |
| **ridge-CCA**（α>0 才需要，flop）| 见下，d=43 / N / N | 数字端 eig/SVD/求逆 | ≈4.70 M | ≈55.0 M | ≈989 M |
| W_α 混合 + CCA→W0 坐标变换 | W=W0+α(W_cca−W0)，b=α·b_cca | elementwise/MAC | 520 | 1,920 | 5,120 |
| **S1 提交** | step=(U/n)/(q+ε)；shift_correction（μ,σ 重标）；Δ←(1−ρ)Δ+η·step；表=P0+Δ+bias | elementwise（≈8KN）| 2,080 | 7,680 | 20,480 |
| 　（reorder 形式额外）u=Σ_l Jbar_lᵀ·corr_l | 33·K·D·N | weight×act（ROM）| 17,160 | 63,360 | 168,960 |
| 折叠 (x−μ)/σ 进表 + 偏置 | W'=W/σ，b'=b−W'μ | MAC（≈2KN）| 520 | 1,920 | 5,120 |
| int8 量化 | 每行 max-abs、倒数、乘、取整 | elementwise（≈KN）+ K 个倒数 | 260 | 960 | 2,560 |
| 写 inactive 表 + 翻转 bank | K·N 字节（+K×(scale,bias)）| 存储写 | 260 B | 960 B | 2,560 B |

不含 CCA 的提交：N=26 为 219,909 MAC，其中 encoder 201,344（91.6%）；N=96 为 953,019（encoder 78.0%）；N=256 为 3,401,499（58.3%）。【D】

**ridge-CCA 提交的算子**（`published_baselines.cca_readin`，`published_baselines.py:322-355`；配置 `run_p0solve.py:56-61`：month/year 为 trial 对齐、20 phase、ridge 10、**d="N"**，即 d=baseline day 的有效维数 d_ref，month trunk 为 43，days trunk 为 17；`run_baselines_readin.py:183-191`）：
1. **cond/phase 累加器**：按 8 个 reach 方向 × 20 个 phase 累加原始计数（160·N 个和，26→4,160 字），**phase 划分要知道 trial 长度**（`phase_average` 对整个 trial 做 `array_split`，`published_baselines.py:202-205`），所以只能在 trial 末尾回放 staging 缓冲一遍：T·N 的 3-bit 读 + 约 T·nz 次累加；每段需要一个按段长的倒数（20 个/trial）。
2. 目标侧 PCA：由 P0 的 Gram+sx 得到相关矩阵 R（N²），对称特征分解 ≈9N³ flop。
3. 投影 160×N×min(N,d) MAC，协方差 ctt、crt 各 160·d² MAC。
4. `cca()`（`published_baselines.py:295-319`）：两次 d×d 对称特征分解（crr、ctt，各 ≈9d³）、一次 d×d SVD（≈22d³）、若干 d×d 乘、一次 d×d 求逆（≈2d³）。crr 只依赖 reference day 与共同条件集（20140626 上 B≥8 为 6 个），共同集稳定时原则上可缓存。
5. K·d²+K·d·N 的最终映射，再变到 P0 坐标。

d=43、N=26 的合计：特征分解 N 0.158 M + 投影 0.108 M + 协方差 0.592 M + 2×eig_d 1.431 M + SVD 1.749 M + 乘 0.477 M + 求逆 0.159 M + 最终 0.030 M = **4.70 M flop**（系数 9n³/22n³ 是教科书估计【D】，不是实现测量）。N=96、256 按 d=N 外推为 55 M、989 M flop。**含 sqrt、除法、迭代，需要一个小的数值核（Jacobi/Householder），不是 MAC 阵列能做的**。B=1 时 `W_alpha (a)` 与 P0 一致到三位（0.181，`causal_main README` 主表），说明 α=0，CCA 可旁路；B≥2 起可能调用。

**CPU 精度探针**【M】（20140626，month trunk，B∈{1,…,33}，在每个线性代数原语的输出上截断尾数，**不是定点误差传播，也不是 R²**）：该 CCA 问题条件数很好（B=8/33 时 cond(ctt+J)=2.2/2.6，因为 ridge=10 使 J≈10×平均特征值），尾数截到 11 bit 时 latent 输出相对误差 ≈0.1%，8 bit ≈1%，6 bit ≈3–4.4%；**B=1（只有 1 个共同条件、秩亏）连 fp32 都与 fp64 差 63%**，解本身不稳定，但 α=0 不会用到。结论：求解的数值精度要求看起来不苛刻，但没有 R² 证据（H2）。

### 1.4 阶段汇总（每 trial，T=252 bin，N=26，dense，S1 用 direct 形式）

| 项 | MAC |
|---|---:|
| 推理 252×418,052 | 105.35 M |
| 校准累加 + FIR：252×(741+9,240) | 2.52 M（推理的 2.4%）|
| 提交（不含 CCA）| 0.22 M |
| CCA（4.70 M flop ≈ 2.35 M MAC 当量）| 2.35 M |
| 校准合计 / 推理 | 2.6%（无 CCA），4.8%（含 CCA）|

N=96：10.1% / 36.2%；N=256：32.2% / 499%（CCA 989 M flop ≈ 495 M MAC 当量，单次就是整个 trial 推理的 4.7 倍）。**校准在 N=26 时是舍入误差级的算术，但在 N≥96 时先被 Gram/FIR 的 N² 或 N·K·L 项、再被 CCA 的 N³ 项吃掉**。

### 1.5 事件驱动（零跳过）

稀疏度实测（上）。在 N=26 上：read-in 260→~44 MAC/bin；P0 累加 741→72；FIR direct 9,240→2,129，reorder 1,716→294。输入 3-bit 计数天然适合 AER 事件表示。它同时降低 SRAM 的 read-modify-write 次数（§2），是代价最低的一条优化。**未测**：N=96 的 Utah array 阈值穿越计数（速率更高）、HD-sEMG（连续特征，无稀疏性）。

---

## 2. 50 Hz 流式数据带宽

假设：20 ms bin，计数 3 bit/unit，标签 y 为 2×16 bit，trunk 激活 16 bit 缓冲（PTQ 研究的 INT16 激活缓冲）、状态 INT24（`state_budget.py` 的口径）或 INT16，P0 数据累加器 16 bit（N=26/96，整数计数）/ 28 bit（N=256 连续特征，【R】P2 §R.2.6），S1 的 U/corr 累加器 32 bit。**“SRAM 流量” = 一次读+一次写的字数 × 字宽，不含寄存器级复用**。

### 2.1 前端 → 芯片

| 接口 | 每 bin | 速率 |
|---|---:|---:|
| 96 通道原始，30 kHz × 16 bit | 115,200 B | **5.76 MB/s** |
| 分箱计数，N=26 / 96 / 256（N=256 若也是 3-bit 计数；sEMG 特征见下行），3 bit | 10 / 36 / 96 B | 0.5 / 1.8 / 4.8 kB/s |
| HD-sEMG 原始 256 ch × 2048 Hz × 16 bit【R：hyser README】| ≈21.0 kB | 1.05 MB/s |
| HD-sEMG 50 Hz 特征（8 bit）| 256 B | 12.8 kB/s |
| 标签 y（2×16 bit，校准期间）、预测 ŷ | 4 B + 4 B | 0.4 kB/s |

原始 → 计数 的压缩：96 通道为 3,200×（sorted N=26 时 11,520×）。**输入侧带宽不是问题，前端把数据压缩到 kB/s 级**；如果片上做 spike 检测/分箱，这一步在芯片外也无关紧要。

### 2.2 片内每 bin SRAM 流量（N=26 | 96 | 256；dense）

| 流 | 估算 | N=26 | N=96 | N=256 |
|---|---|---:|---:|---:|
| read-in 表读（int8，dense）| K·N | 260 B | 960 B | 2,560 B |
| read-in 表读（事件驱动）| K·nz | 44 B（实测）| | |
| trunk 激活写+读（5,034 个值/bin，16 bit）| 5,034×2×2 | 20.1 kB | 同 | 同 |
| **SSM 状态读+写（17,216 值）** | 17,216×2×字宽 | **INT24 103.3 kB / INT16 68.9 kB** | 同 | 同 |
| P0 累加 RMW（触及字数×字宽×2）| N(N+1)/2+23N+27 字 | 976 字→3.9 kB | 6,891→27.6 kB | 38,811 字×4 B→310 kB |
| S1 FIR，direct 形式（U 按 unit 主序累加，K·N 字 RMW，32 bit）+ ring 读 | 2·4·K·N + L·N·3/8 | 2.4 kB | 8.9 kB | 23.6 kB |
| S1 FIR，reorder 形式（33·D·N 字 RMW）| 8·33·D·N | 13.7 kB | 50.7 kB | 135 kB |
| 提交时表写 | K·N + 4K | 0.3 kB/commit | 1.0 kB | 2.6 kB |

【D】。**总流量（INT24 状态，P0+FIR direct）**：N=26 为 **130 kB/bin = 6.5 MB/s**（状态 79%、trunk 激活 15%、P0 累加 3%、FIR 2%）；N=96 为 161 kB/bin（状态 64%，P0 累加 17%）；N=256（连续特征 dense）为 ~460 kB/bin = 23 MB/s，**Gram 的 RMW 占 67%，状态降到 22%**。没有校准时（推理期）N=26 为 124 kB/bin = 6.2 MB/s。事件驱动后 N=26 的校准流量下降约一个数量级（P0 累加 3.9→~0.5 kB），状态占比上升到 >80%。

### 2.3 瓶颈与含义

1. **N≤96：SSM 状态读写是第一瓶颈**（3.4–5.2 MB/s，状态 17,216 值每 bin 各读 1 次、写 1 次，每个状态元素做 4 次乘加（ssm 部分））。绝对带宽对 SRAM 毫无压力；它的意义在**能耗**（每 bin ~34 k 次 24-bit SRAM 访问）和**位宽选择**（INT16→INT24 差 50%）。优化方向：状态就地更新（near-memory / in-SRAM 乘加），避免经过总线；位宽见 §5。
2. **trunk 激活 20 kB/bin** 次之。CiM 下，这些值是 DAC 输入/ADC 输出的搬运，不是额外的内存压力。
3. **N≥96：校准累加器的 RMW 反超**，N=256 为状态的 3×（28 bit Gram）。事件驱动对离散计数特征有效；对连续 sEMG 特征无效，要靠换成 N×K 的潜变量累加器（§4）。
4. **S1 FIR 的 direct/reorder 取舍**：direct 流量小（2.4 kB）但 MAC 多（8,580）；reorder MAC 少（1,716）但流量 5.7×（13.7 kB）。在以 SRAM 访问为主要能耗时，direct+事件驱动（2,129 MAC、~0.4 kB）是 N=26 的更优点。
5. **提交的流量不在 per-bin 预算里，但不能忽略**：CCA 提交对 d×d 工作区做多遍扫描。上界（每个 flop 一次 4 B 字访问、无寄存器复用）约 19 MB/commit，与一个 trial 的 per-bin 流量（252×130 kB≈33 MB）同量级；做寄存器分块可低一个数量级。因此数值核需要局部寄存器或小 SRAM bank，而不是让它直接打主 SRAM。【D】

---

## 3. 存内计算（CiM）使用

### 3.1 什么放 RRAM、什么放 SRAM

| 对象 | 权重数 | 位置 | 理由 |
|---|---:|---|---|
| trunk block 层：in_proj z/x（65,536）+ B/C（8,192）+ out_proj（32,768）+ FFN0/FFN2（65,536）每 block，共 344,064 | 344,064 | **RRAM（2T2R）** | 固定、source 训练；每 bin 复用 1 次；占推理 MAC 的 82% |
| root lift/readout、dt/A/trap/angle 行 | 6,656 | 数字 ROM/SRAM（INT8）| 小、喂指数/softplus，对量化敏感 |
| P0 encoder | 7,744（+138 bias）| RRAM 候选或数字 ROM（7.7 KiB int8）| 固定权重，但调用稀（见 §3.4）|
| Jbar 核（L=33×D×K）| 660 | ROM（int8，**成本假设，未做定点**）| 太小，不值得上 CiM |
| prior 表（K×96 电极）| 960 | ROM | 每个植入固定 |
| **read-in 表 T（K×N）** | 260–2,560 | **SRAM，双缓冲** | 每次提交都变（≤B 次/session），每 bin 读 → 读写比 ~10⁴ |
| Gram、CCA 累加器、状态 | | SRAM | act×act，动态 |

不要把 HᵀDH、Gram·Pᵀ、CCA 的 eig/SVD 映射成需要重写 RRAM 的操作（研究文档 B1 的原则成立，本文量化了它们的规模）。

### 3.2 阵列与位切片（256×256 阵列；2T2R 差分对=1 个有符号单元）

按 PTQ 研究的编码（`rram_quant_20261003/README.md` §设定：k bit 权重用符号–幅值位切片，**k−1 个差分对**；ternary 1 对）和“FFN + B/C ternary，其余 8 bit”配置：

| 层（每 block）| 权重 | 位宽 | 切片 | 差分对 |
|---|---:|---|---:|---:|
| z、x | 65,536 | 8 bit | 7 | 458,752 |
| B、C | 8,192 | ternary | 1 | 8,192 |
| out_proj | 32,768 | 8 bit | 7 | 229,376 |
| FFN0+FFN2 | 65,536 | ternary | 1 | 65,536 |
| 合计 | | | | 761,856 |

两个 block 共 **1,523,712 对 = 1.453 Mi**（PTQ 研究在 d_state=16 的 M1/M2 上的数字 1.45 Mi 同量级；本 trunk 的 B/C 行加倍，差 <1%）。阵列数：致密打包 ⌈1,523,712/65,536⌉ = **24**；按“一层一套 tile”的朴素映射 **42 个，利用率 55%**（z/x 的输入只有 128 行，半个阵列；可通过时分复用行组提高）。吞吐不是问题（50 Hz），所以高度时分复用、牺牲并行换面积是合理的。P0 encoder 若放 RRAM：8-bit 切片 54 k 对，致密打包 1 个阵列（利用率 83%），朴素 5 个。**这些是容量/映射推演；PTQ 研究只做权重扰动，没有 IR drop、弛豫、写入验证，也没有 ADC。**

### 3.3 ADC 转换（CiM 的主要成本）

每 bin 的 ADC 转换数 = Σ 层 输出列数 × 权重切片数 × 输入 bit-serial 周期数（rows≤256，无行 tile 拆分）：

| 配置 | 转换/bin（两层）| 转换/s | MAC/转换 |
|---|---:|---:|---:|
| 混合（z/x、out 8 bit；B/C、FFN ternary），输入 8 周期 bit-serial | **78,848** | 3.94 M | 4.36 |
| 同上，8-bit DAC 一周期 | 9,856 | 0.49 M | 34.9 |
| 全部 8 bit，8 周期 | 121,856 | 6.1 M | 2.8 |
| 全部 ternary，8 周期 | 17,408 | 0.87 M | 19.8 |

【D】。**bit-serial + 多切片下，每次 ADC 转换只摊到 ~4 个 MAC**；算力对 CiM 不是瓶颈，转换才是。因此“CiM 占 82% MAC”不等于“82% 的能耗便宜”（研究文档 B2 的提醒），必须在 ADC 位宽、切片数、输入编码上做联合设计。实测的激活动态范围加剧了这一点（§5.1 P2）：L1 out_proj 的输出元素 p50=12、p99=2,900，通道最大值的中位 ≈2×10²、最大 ≈6.8×10³（约 30×），全局 ADC 量程会被少数列拖垮。

### 3.4 各阶段的 CiM 占比与权重复用

| 阶段 | CiM 映射 MAC | 其他 | CiM 占比 |
|---|---:|---|---:|
| 推理/bin（N=26 / 96 / 256）| 344,064 | 74,000 / 74,700 / 76,300 | 82.3% / 82.2% / 81.9% |
| 校准累加/bin（P0 + FIR direct）| 0 | 741 + 9,240（N=26）| **0%** |
| 提交，不含 CCA（N=26 / 96 / 256）| 201,344 / 743,424 / 1,982,464（encoder）| 18,565 / 209,595 / 1,419,035 | 91.6% / 78.0% / 58.3% |
| 提交，CCA 分支 | 0 | 4.7 M / 55 M / 989 M flop | 0% |
| 一个校准 trial 总计（N=26 / 96 / 256；CCA 按 flop/2 计）| | | 80.4% / 75.3% / 63.3%（不含 CCA）；78.7% / 60.9% / 14.0%（含 CCA）|

权重复用：trunk 权重每 bin 用 1 次（50 次/s；一个 5×10⁴–2.5×10⁵ bin 的 session 用同数量级次数）；**P0 encoder 每次提交用 N 次，每个 session ≤B 次提交，N=26、B=33 时 858 次**，比 trunk 少约 2–2.5 个数量级。encoder 的 MAC 功率：201 k MAC / ~5 s ≈ 0.04 M MAC/s，是 trunk（20.9 M MAC/s）的 0.2%。**所以 encoder 上不上 RRAM 对功耗不起作用，只影响面积和“校准路径也用 CiM”的故事**；若不上，数字 MAC 引擎（本来就要有）以 int8 权重 ROM 顺序执行即可，20 ms 内能做完（201 k MAC@25 MHz≈8 ms）。校准路径的 CiM 收益应主张在“同一套固定权重技术覆盖 trunk 与 encoder”而不是能耗。

校准硬件的占空比：session 里可用 trial 数 174–733【M】，B=33 时校准活跃 4.5–19%（B=8 时 1.1–4.6%）；此外累加器、staging、CCA 工作区在 B 个 trial 之后可以断电。

---

## 4. 存储：峰值同时驻留

宽度假设：状态 INT24（`state_budget.py`；INT16 见下）、trunk 激活缓冲 9.4 KiB（PTQ 研究 INT16 双缓冲；d_state 16→32 差异可忽略）、P0 数据累加器 16 bit（N=26/96）/28 bit（N=256）、计数器 17 bit×17 个、S1 的 U/Δ 32 bit、ring 3 bit、CCA cond/phase 和 24 bit（**假设**：按段求平均需要小数位，exact 复现需要段长倒数；16 bit 未验证）、staging 取典型 T=252、CCA 工作区 fp32 无就地复用。**每个“工作区”数字都是上界式估计**。

| 项（KiB）| N=26 | N=96 | N=256 | 备注 |
|---|---:|---:|---:|---|
| SSM 完整状态 17,216 值（INT24）| 50.4 | 50.4 | 50.4 | INT16：33.6 |
| trunk 激活缓冲 | 9.4 | 9.4 | 9.4 | |
| read-in 表 ×2（int8 + 每行 scale/bias）| 0.6 | 2.0 | 5.1 | |
| **推理常驻小计** | **60.4** | 61.8 | 64.9 | |
| P0 前缀统计（最小集，对称 Gram）| 2.3 | 14.8 | 138.7 | N=256 为 28 bit |
| S1：ring + U + Δ + 每 trial Σx、Σx² | 2.6 | 9.5 | 25.1 | reorder 形式的 corr 为 6.7 / 24.8 / 66.0 |
| CCA cond/phase 累加器 160·N（24 bit）| 12.2 | 45.0 | 120.0 | **旧表没有** |
| 单 trial staging（T=252）| 3.4 | 9.9 | 24.6 | 最长 trial 2,741 bin：37 KiB；5,144 bin：70 KiB |
| mean/std（16 bit）| 0.1 | 0.4 | 1.0 | |
| **校准常驻小计** | 20.6 | 79.5 | 309.5 | |
| **常驻合计** | **81.0** | **141.3** | **374.4** | |
| 提交工作区：K×N fp32 表 ×2 | 2.0 | 7.5 | 20.0 | |
| 　CCA：lt（160×d fp32，d=43 / N / N）| 26.9 | 60.0 | 160.0 | |
| 　CCA：目标侧特征分解 R,V（2N² fp32）| 5.3 | 72.0 | 512.0 | |
| 　CCA：6 个 d×d fp32 | 43.3 | 216.0 | 1,536 | N=96/256 取 d=N（“d=N”规则）|
| **CCA 提交峰值** | **158.5** | **496.8** | **2,602** | |
| 　无 CCA 分支（常驻−CCA 累加器）| 68.8 | 96.3 | 254.4 | |
| 　乐观（状态 16 bit、CCA 累加器 16 bit）| 60.1 | 109.5 | 317.6 | 常驻口径 |

ROM/非易失（不占上表）：RRAM 1.453 Mi 对（24 阵列）；数字 ROM ≈ 37 KiB = root/行权重 6.5 + 小向量 6.5 + P0 encoder int8 7.7 + prior 0.9 + Jbar 0.6 + CCA reference-day 常量（8×20×43 条件平均 + A_pca）14.4 + α/η/ρ 时间表 0.2【D】。

**读法**
1. **N=26：推理 60.4 KiB，加上校准常驻 81.0 KiB，放得进 128 KiB；CCA 提交按代码的 d=d_ref=43、fp32 就是 158.5 KiB，超了**。主要是 d×d 的 6 个矩阵（43 KiB）和 160×43 的投影（27 KiB）。若目标侧维数取 min(N,d_ref)=26 且工作区 fp32：lt 16.6 + d² 15.9 + R,V 5.3 + 表 2.0 = 39.8 KiB，峰值 ≈121 KiB（**这是一个未评估的算法改动**：代码里 target 侧 `pca_basis` 对秩亏补零列，等价性需验证）。算法侧另两条出路：只在 B≥某值时算 CCA、或把 CCA 的 crr/ir 缓存进 ROM。
2. **与旧表对账**：`stream_f7` README 的 S1-causal（N=26）“不含 state 44.7 KB / 含 76.7 KB”里 31.4 KB 是 fp32 的 P0 权重（本文放 ROM，int8 仅 7.7 KiB），统计量用 32 bit 且含 q_*，且**没有 CCA 的 cond/phase 累加器与求解工作区**；两个口径不能直接比，关键差别是 CCA 分支。
3. **N=96**：不含 CCA 常驻 96.3 KiB（假设 16 bit 累加器对电极级计数仍足够——**未验证**：电极级速率更高，Gram 步长要 ≤1，可能要 18–20 bit），能放下；CCA 分支（工作区 356 KiB）不能。
4. **N=256**：P0 Gram 28 bit 就 139 KiB，加状态已超；**N×N Gram 的缩放问题不止是 Gram 本身**：CCA 的 R、V（512 KiB）与 6 个 d×d（1.5 MiB）更严重。替代方案：N×K 潜变量累加器（P2 估算 ~62 KB，未评估）、整数码（int8 码 21 bit 精确累加，259 KiB，未评估解码）、CCA 取小 d / subspace iteration（baselines README：d=K=10 的 bridge 有 0.05–0.08 的上限损失，d=N 才精确；所以降维有精度代价）。
5. **staging 的最坏情形**：trial 长度分布 p95≈255–293 bin，但最长 2,741–5,144 bin（长尾）；staging 必须设上限并定义溢出行为（截断 trial / 跳过该 trial 的 CCA 与 trial-0 归一化），**该行为未评估**。

---

## 5. 精度 / 融合 / 近似 / 存储：机会、证据、缺口（更新自 `hw_opportunities_qa_20261016.md`）

标记：✅ 已有证据且适用于冻结流水线（本节暂无）；🟡 证据存在但对象不是冻结流水线（旧协议、旧模型、或 P0 only）；⬜ 未测。

### 5.1 精度

| # | 对象 | 现有证据 | 对冻结流水线的状态 | 缺口 / 本次新增测量 |
|---|---|---|---|---|
| P1 | trunk 权重混合精度（FFN+B/C ternary、z/x/out 8 bit）| 【R】PTQ：M1/M2（d_state=16）source val 上损失 −0.012/−0.010；逐层：B/C −0.001–+0.008，FFN −0.001–−0.009，in_zx −0.016/−0.020，out_proj −0.024 | 🟡 **000688 的 K=10、d_state=32 trunk 未测**；QAT 未做 | H1 |
| P2 | 激活位宽与 ADC | 【R】全部 8 bit 权重 + RRAM 层输入 INT8（support 窗口标定）时 source val 损失 ≤0.002；ADC/部分和**未建模** | ⬜ | 【M】12k bin 实测（month trunk，2 个 session）：LN 输出有界（p99.9≤4.7，最大≤7.2）→ RRAM 层输入 INT8 有余量；**但残差流有离群通道**：中位 21、p99 3,300、最大 7.6×10³（通道 41/66/123 最大 ~6×10³，通道最大值的中位 2.4×10²）；L1 out_proj 的输出 p50=12、p99=2,900、最大 6,085；FFN 输出最大 240–300。含义：数字残差流（最大 7.6×10³）需要 13 位整数位，INT16 只剩 2 位小数（步长 0.25，约为中位值 21 的 1%）；out_proj/FFN2 的 ADC 要逐列量程或数字缩放，**全局 ADC 量程会被少数列拖垮** |
| P3 | SSM 状态位宽 | 【R】hardware_contract：通用 real-2×2 引擎 int16 饱和 4,656 次，int24/int32 为 0；`state_budget` 按 INT24 | 🟡 端到端解码精度随状态位宽的变化**未测** | 【M】本 trunk 状态实测峰值：ssm 元素绝对值最大 25.2（源 session）/ 19.6（target+prior 表），k carry ≤6.2，v carry ≤17.3，α_min=0.27，dt∈[4×10⁻⁵, 0.36]：整数位 5 位（含余量 6），INT16 只有 Q5.10、对 25.2 只有 1.27× 余量，**漂移更大的 target 上不安全**；INT24 有富余。样本小（12k bin ×2），需要全 session |
| P4 | P0 encoder 权重 | 【R】P2：int8 的 ΔR² 绝对值 ≤1×10⁻⁴，int4 ≤2×10⁻³，ternary PTQ −0.03–−0.06，source QAT 后 days 基本恢复、month/year 仍 ≈−0.02 | 🟡 P0 only，旧协议；冻结流水线（W_α/COMBO）下未重测 | encoder 只占 0.2% 功耗 → **int8 即可，没必要追 ternary** |
| P5 | P0 数据累加器 / 计数器 | 【R】P2 §R.2：逐 add 取整，16 bit 全 key 安全（最差 −0.0005）；14 bit 在 month/year 出 NaN（Gram 步长 2）；12 bit 崩；计数器精确 17 bit（B=33、最长 trial 2,741 bin：90,453）| 🟡 旧随机抽取协议；**未在 causal-first-B 前缀提交下重跑**（累加器跨提交持续、每个前缀都 finalize）| H3；注意步长需 ≤1 与量程余量 +1 bit（Gram 在某 target 上达 source 量程 1.08×）|
| P6 | 新累加器：CCA cond/phase 和、S1 的 U/corr/Δ/ring/q | 无 | ⬜ **全部 fp32**【F】 | H3/H4；ring 3 bit 有实测支撑（最大计数 5–7）|
| P7 | read-in 表 int8 | 【R】P2：P0 表 int8 无损、int4 ≤0.005 | 🟡 P0 表；**合成表（P0→W_α→+Δ→折叠 μ,σ）未测** | 【M】CPU 代理（z 空间相对 RMS，**不是 R²**）：先 int8 量化再归一化 0.5–0.7%，**折叠后再量化 0.9%**（1.3–2×，约损失 1 bit：1/σ∈[1.3, 6.1]，行量程被小 σ 的 unit 放大 3.2–3.9×）；int4 折叠后 15–18%，不能用。→ H5 |
| P8 | CCA 求解精度 | 无 | ⬜ | 【M】见 §1.3：条件数 2.2–2.6，11 bit 尾数→0.1%，8 bit→1%（z 空间代理）。→ H2 |
| P9 | S1 FIR 的 Jbar ROM 位宽 | 【C】`refine_cost`/`stream_f7` 把 Jbar、readout 基记为 int8 ROM，但**所有路径 fp32，无定点** | ⬜ | H4。冻结流水线用的是 FIR（`run_causal_main.py:452` `kind="fir"`），不是 trace，所以“大的相消系数在 float32 下不稳”（`stream_f7` README）**不在冻结路径上**；只有为省 ring 而改用 trace 时才会遇到 |
| P10 | read-out ridge（R1-stream）| 旧计划 Q1.7 | 不在 COMBO 里（无 readout 分支）| 删除该项 |

### 5.2 融合

| # | 融合 | 状态 | 估计收益 |
|---|---|---|---|
| F1 | (x−μ)/σ 折进 read-in 表（W'=W/σ，b'=b−W'μ）| 数学精确。评估器是先归一化再乘表（`run_causal_main.py:528-535`）；折叠后每次提交都要重算+重量化 | 省 2N 次/bin 的减/乘；代价见 P7（≈1 bit）|
| F2 | 不要把 read-in 与 in_proj/lift 合并 | 本 trunk：K·N+H·K = 260+1,280=1,540（N=26）vs 合并 H·N=3,328（2.2×）；N=256 为 3,840 vs 32,768（8.5×）；合并矩阵每次提交都变，得放 SRAM | 维持分离 |
| F3 | prior + encoder 残差 + Δ + CCA 混合 → 提交时合成一张 int8 表 | 冻结流水线已是此结构 | 推理期只有一张表、一个 bias |
| F4 | 推理与校准共用同一次输入读取（事件列表）| 可行 | 省一遍 N·3 bit 读；实际收益在累加器的零跳过 |
| F5 | P0 统计去冗余：q_* 与 r 特征（P0 不用）、sphi/sphiphi（派生）、sphix[0]=sx、ac_a+ac_b 合并 | 精确恒等；sphi/sphiphi/sphix[0] 在 `calib_quant` 里已按派生建模，q_* 去除与 ac 合并**未实现** | N=26：最小集 1,141 字 + 17 计数器，对比 `calib_cost.acc_elements` 全集 1,653 字（N×N Gram）/1,328 字（对称 Gram）；每 bin 省 3N 次加 |
| F6 | describe：zvar=Σp·cz（K·N²→K·N），不形成 N×N 的 gn，1/σ 预算 | 【D】精确/近似（1e-12 项除外），未实现 | 15.2 k→~8.6 k MAC（N=26）；N=256：1.39 M→0.67 M |
| F7 | S1 FIR：corr 先于 Jbar（reorder）| 精确，`fir_data_term` 即如此 | MAC 5×↓、累加器 6.6×↑、流量 5.7×↑，见 §1.2、§2 |
| F8 | CCA 与 P0 共享 Gram/mean/std，CCA 结果直接变到 W0 坐标（`p0solve.map_cca_into_w0_coordinates`）| 已是代码结构 | 避免第二份 N×N |
| F9 | encoder 逐 unit 流式：不存 N×47 描述矩阵；d_mu/d_sd 折进第一层 | 精确，未实现 | 省 N×94 B（24 KiB@N=256）|
| F10 | encoder 的 N 个 unit 作为 weight-stationary 批 | 结构上天然 | 权重复用 N 次，见 §3.4 |
| ~~F11~~ | ~~readout 解折进 head~~ | COMBO 无 readout | 删除 |

### 5.3 近似 / 特殊算子

| # | 算子 | 次数 | 现有证据 | 冻结流水线状态 | 实测输入域（【M】，月 trunk，2×12k bin）|
|---|---|---:|---|---|---|
| A1 | softmax（旧 P3/P4 pooling）| — | 【R】exp2 LUT+倒数无损 | **冻结算法不含 pooling，不需要**；不要再当贡献 | — |
| A2 | SiLU（门控 256 + FFN 256，每层）| 1,024/bin | 无 | ⬜ | FFN 前激活绝对值最大 35.8（L1）/ 18.4（L0）；门控 z ∈ [−12.3, 14.7]。两侧饱和，LUT/PWL 域约 [−13, 36] |
| A3 | softplus（dt）| 8/bin | 无 | ⬜ | 输入(含 dt_bias) ∈ [−10.0, −1.28] → 输出 ∈ [4×10⁻⁵, 0.36]，在 softplus 的指数区，近似 e^x |
| A4 | exp2(ADT·log2e)（衰减 α）| 8/bin | 无 | ⬜ | α ∈ [0.27, 1.0]，自变量 ≤0 |
| A5 | sigmoid（trap）、heavy-tail 倒数（A）| 16/bin | 无 | ⬜ | trap ∈ [−7.6, 4.8]；dd_A ∈ [−8.8, 10.1] |
| A6 | tanh、sin/cos、mod 2π（RoPE）| 16 + 128 + 64 | 无 | ⬜ | tanh 输入 ∈ [−12.3, 9.8]（饱和）；相位累加器每 bin 做 mod 2π 累加，**舍入误差随时间累积**（`mamba3_streaming.py` 文档记录：官方 wrapper 的 bf16 角状态每个边界损失 ~2⁻⁸ 相对精度；官方 step 核用 fp32 `cos.approx/sin.approx`），定点相位字宽需单独验证 |
| A7 | rsqrt（LN×5、RMSNorm×4）| 9/bin | 无 | ⬜ | LN 方差入口；共享一个迭代 rsqrt 单元即可 |
| A8 | describe 的 log/sqrt/rsqrt/除法 | N=26：~54+~312+~900 次/commit | 无 | ⬜ | 每 ~5 s 一次，**共享一个迭代（Newton）单元，不需要 LUT**；log 的 ε（1e-2、1e-3）使域有界 |
| A9 | encoder 的 GELU | 128N/commit | 无 | ⬜ | N=26：3,328 次/commit |
| A10 | CCA 的 eig/SVD/求逆/sqrt | §1.3 | 无 | ⬜ | 需要 Jacobi/Householder 小核；见 H2 |
| A11 | 事件驱动零跳过 | — | 【M】实测稀疏 83% 零，每 bin ~4.45 非零 | 精确（零乘零）| 收益：P0 累加 10×、FIR direct 4×、流量 ~1 数量级 |

### 5.4 存储

| # | 问题 | 结论（冻结流水线）|
|---|---|---|
| M1 | 推理时最大 SRAM 项 | SSM 状态：50.4 KiB（INT24）/33.6 KiB（INT16）；加激活缓冲 9.4 KiB 与表：60.4 KiB。旧文的“000688 状态约 32 KB”漏了 carry 值（+1.6 KiB）|
| M2 | 校准常驻 | N=26：20.6 KiB，其中 **CCA cond/phase 累加器 12.2 KiB 是新项**（旧表的 P0 6.8 KiB / P4 7.8 KiB 不含它）|
| M3 | 提交峰值 | N=26：158.5 KiB（按代码 d=43，fp32 工作区）> 128 KiB；是 CCA 求解工作区，不是 P0 |
| M4 | N 缩放 | Gram N(N+1)/2、CCA 的 R/V（2N²）、d×d（6d²，d=N）：N=96 的 CCA 峰值 497 KiB；N=256 的 Gram 本身 139 KiB。需要 N×K 潜变量累加器 / 低秩 CCA / 小 d |
| M5 | 数据带宽瓶颈 | §2：N≤96 是状态读写；N=256 连续特征是 Gram RMW |
| M6 | 重放缓存 | COMBO 不重放：只有 1 个 trial 的 staging（典型 3.4 KiB，最坏 37–70 KiB）；R2/R3 的重放缓存（114 KB 级）不在可部署路径上 |

### 5.5 需要补的硬件忠实仿真实验（按优先级）

| 序 | 编号 | 内容 | 为什么 | 工作量 |
|---|---|---|---|---|
| 1 | **H1** | **trunk-in-the-loop 定点/混合精度**：在 000688 K=10、d_state=32 trunk 上用纯 PyTorch 流式参考（`mamba3_streaming.decoder_step`，CPU 可跑；两个 session 共 24k bin 含数据加载约 57 s）注入：状态 Q5.x（16/20/24 bit）、残差流 INT16、RRAM 层输入 INT8（LN 输出）、PTQ 混合权重、逐列 ADC（6–8 bit，含离群列）、器件 σ；评估 P0 / W_α / COMBO 在 12 个 drift target 上的 R² | 最大证据缺口：PTQ 只做了 M1/M2（d_state 16）的 source val、没有 ADC；trunk 占 82% MAC 且是唯一进 CiM 的块；§5.1 P2/P3 的离群/动态范围已提示风险 | M–L（3–5 天）。纯 torch 逐 bin 的 CPU 版对整段 session 偏慢（250k bin 约 10 min/配置/target，12 个 target × 多配置）→ 需分块批处理版或上 Colab |
| 2 | **H4** | **S1 live 残差一致性**：把 S1 的校准前向改为用推理的 live ŷ（状态跨 trial 携带）而不是每 trial 零状态重跑，重算 COMBO；顺带比较 direct/reorder/事件驱动 FIR 的定点（Jbar int8、r 16 bit、U/corr 24–32 bit） | 决定 S1 是否真的“免费”（§7 G1）；不通过则校准期要 shadow forward（trunk MAC×2 + 50 KiB 状态）| M（改 `rebased_feedback_loop` 的前向，GPU 评估）|
| 3 | **H2** | **CCA 提交的定点/低精度**：Jacobi 特征分解 + SVD 的低字宽实现，量化 cond/phase 累加器和段长倒数；评估 R² 与 α 调度；同时评估 target 侧维数 min(N,d_ref)、crr 缓存、旁路规则（α=0 / B 小）| CCA 是提交中最重、最不规则、也是 SRAM 峰值的来源；CPU 代理（§1.3）说明条件数好但无 R² | M（2–3 天；R² 需 GPU/Colab，其余 CPU）|
| 4 | **H3** | **causal-first-B 下重跑逐 add 累加器**：用 `calib_quant.perbin_stats` 在前缀提交 k=1..B 上跑 P0 统计（16 bit/17 bit 计数器），加入 CCA cond/phase 累加器（16/20/24 bit）；W_α、COMBO 全链路 | P2 的无损结论来自旧随机抽取协议且只有 P0；冻结流水线要用同协议证明 | S–M（复用现有函数；GPU 评估）|
| 5 | **H5** | 合成表量化：W_α/COMBO 的表 → 折叠 μ,σ → int8 每行 scale（含 bias 通路、双缓冲、切表）| CPU 代理已显示折叠损 ~1 bit；R² 影响未知 | S（CPU 代理已写；R² 需 GPU）|
| 6 | **H6** | 非线性近似：SiLU/softplus/exp2/sigmoid/tanh/sin-cos（含相位 mod 2π 定点）/rsqrt 的 LUT 或 PWL，逐个替换再全替换；describe 的 log/sqrt/rsqrt/除法；GELU | 1,209 次/bin，SiLU 占 85%，输入域已测 | M（与 H1 共用纯 torch 参考）|
| 7 | **H8** | **commit 延迟标定**：用 §1.3 的周期数模型（引擎并行度、频率）换算成 bin/trial 级延迟，扫 `latency` 参数（协议已支持）| 把“anytime”落到硬件延迟；CCA 提交最长 | S–M |
| 8 | **H7** | 事件驱动统计覆盖更多 session 与（若有）Utah array 数据；N=96 的计数位宽与累加器位宽 | 稀疏度/最大计数/累加量程直接决定 N=96 的位宽与流量 | S |
| 9 | **H9** | N≥96 的替代结构：N×K 潜变量累加器取代 N×N Gram、低秩/小 d 的 CCA、subspace iteration | 缩放结论，sEMG 为边界场景，优先级低（MZ 已定）| L |
| — | H10 | 第一阶能耗/面积模型（MAC、SRAM 访问、ADC 转换、RRAM 读）| 论文需要，但依赖工艺假设，应在 H1 之后用 H1 的位宽配置做 | S（需要引用的参数）|

---

## 6. 论文用汇总表与图

### 6.1 汇总表（N=26 为主；括号内 N=96 / N=256）

| 阶段 | CiM MAC | 数字 MAC | 非线性 | 峰值 SRAM（KiB）| ROM / RRAM | 带宽（每 bin）| 主导成本 |
|---|---|---|---|---|---|---|---|
| (i) 推理，每 bin | 344,064（82.3%）| act×act 67,072；其余 weight×act 6,656；read-in 260（960 / 2,560）| 1,209 次（SiLU 85%）| 60.4（61.8 / 64.9）| 1.45 Mi 2T2R 对，24 阵列；数字 ROM ≈37 KiB | 130 kB 中的 103 kB 是状态读写；**78,848 次 ADC 转换** | ADC 转换 + SSM 状态读写 |
| (ii) 校准，每 bin（前 B 个 trial）| 0（Jbar 660 可选）| 741 + 8,580（6,096 + 31,680；36,736 + 84,480）；事件驱动 72 + 1,469 | 0 | +20.6 常驻（79.5 / 309.5）| — | 3.9 + 2.4 kB（27.6 + 8.9；310 + 23.6）| Gram 的 RMW 与 FIR 外积 |
| (iii) 提交，P0 分支，每次 | 201,344（743,424 / 1,982,464）| 18,565（209,595 / 1,419,035）| GELU 3,328；log ~54；sqrt/rsqrt ~312；div ~900 | 工作区 2.0（7.5 / 20）| encoder int8 7.7 KiB | 表写 0.3 kB | encoder MLP（N 小时）→ describe 的 N²K（N 大时）|
| (iii) 提交，CCA 分支（α>0）| 0 | 4.70 M flop（55 M / 989 M）| sqrt ~86、除法 | 工作区 77.5（355.5 / 2,228）+ 累加器 12.2（45 / 120）| ref-day 常量 14.4 KiB | 提交内部上界 ~5×10⁶ 字访问（无分块）| eig/SVD；SRAM 峰值 |
| 切表 | 0 | ≈6 KN | — | 表×2 已计 | — | K·N B | 可忽略 |

峰值总计：N=26 常驻 81.0；CCA 提交 158.5（d 缩到 26：~121）；N=96 无 CCA 96.3、有 CCA 497；N=256 常驻 374（无 CCA 254）。

### 6.2 图建议

1. **Fig. A 系统框图 + 带宽标注**：前端（5.76 MB/s）→ 分箱/事件化（36 B/bin）→ read-in SRAM 表（双缓冲）→ CiM trunk 宏（24 阵列，78.8 k ADC/bin）↔ 数字 SSM 引擎 + 状态 SRAM（5.2 MB/s）→ readout；校准引擎（P0 统计、CCA 累加器、S1 FIR、encoder、数值核）与 commit/量化/切表。边上标 bytes/s，色块区分 RRAM / SRAM / 数字。
2. **Fig. B 按类别堆叠的 MAC 条形图（对数轴）**：横轴 N=26/96/256，每组三栏——推理（CiM / 数字 weight×act / act×act）、校准累加（dense 与事件驱动）、提交（encoder CiM / describe / CCA 数字）。要点：推理 CiM 占 82% 不随 N 变，校准在 N≥96 才显著。
3. **Fig. C 峰值 SRAM 堆叠 + 128 KiB 线**：N=26/96/256 各一栏，层次为状态、激活、表、P0 统计、S1、CCA 累加器、staging、CCA 工作区；同时画“无 CCA 分支”“d 缩到 26”两条参考。直接展示 N² 项与 CCA 工作区。
4. （可选）**Fig. D 精度–代价 Pareto**：P-prior / P0 / W_α / COMBO / CCA+S1 在 B=8/16/33 的 drift R²（`causal_main` 主表：COMBO (a) 0.381 / 0.406 / 0.471；W_α 0.348 / 0.404 / 0.437；ridge-CCA 0.329 / 0.387 / 0.409；P0 0.209 / 0.218 / 0.215；(b) 保守协议 COMBO 0.336 / 0.383 / 0.465）对提交 MAC 与峰值 SRAM。注意：主表里 S1/COMBO 的增量来自 AND 选参门（drift 上只有 1 个 selection session）；README 里更保守的门（被选中的 (η,ρ) 至少在 2 个 selection session 上增益为正）在 drift 上会把 S1/COMBO/CCA+S1 全部退回基座，即 S1 的增量依赖这一门；(b) 协议下 COMBO 相对 ridge-CCA 与 ui_reg 只在 B=33 稳定为正。图上要如实标注，不要把 S1 的硬件代价画成无条件的收益。

---

## 7. 算法–硬件不一致与待 MZ 决定

**G1（必须先定）S1 的残差来源。** `rebased_feedback_loop`（`stream_f7.py:491-570`）对每个校准 trial 调 `_forward_biased(base, xz.unsqueeze(0), table, …)`，即**从零状态**单独前向；而评估用的推理是整段连续（`run_causal_main.py:538-548` `_forward_batch(base, z)`，状态跨 trial，表按 commit 段切换）。`stream_f7` README §5 也记着“trial 之间传递 trunk state 没有做”。部署中 ŷ_t 来自 live 状态，与评估里的零状态残差不同（差别集中在 trial 起始的状态预热阶段，幅度未测）。两条路：(a) 用 live ŷ 作残差，**需要重跑 COMBO 的 S1 评估**（H4）；(b) 校准 trial 里跑 shadow forward（零状态），则校准期间 trunk MAC×2、再多一份 ~50 KiB 状态。`stream_f7` README 里“S1 的 trunk 前向就是这次推理前向，不再重放”仅在 (a) 成立。

**G2 whole-trial staging。** trial-0 的归一化（`initial` map）与 CCA 的 phase 划分都要看完整个 trial。staging 峰值取决于 trial 最长值（252 平均，2,741–5,144 最长）；溢出行为未评估。可选：running normalisation（P3 记为未评估）、go-aligned 的 CCA（days 配置，50 bin 固定窗，不需 trial 长度）。

**G3 CCA 分支的去留与形态。** 它是 W_α、COMBO 的收益来源之一（drift B=8：W_α 0.348 vs P0 0.209），也是提交里最贵、SRAM 峰值最高、最不规则的部分。可选：d=min(N,d_ref)、缓存 crr/ir、小 d（精度代价见 baselines README）、只在若干 commit 上算。需要 MZ 在“保留完整 CCA 并接受 ~159 KiB 峰值”与“先改算法再冻结硬件”之间选。

**G4 128 KiB 预算的口径。** 现有预算（`state_budget.BUDGET_BYTES`）把整块 SRAM 视为 128 KiB；本文显示 N=26 的 trunk 状态（INT24）已占 39%、推理常驻 47%。是否把状态降到 INT16 或让 CCA 工作区借用其他缓冲，需要一个明确的内存规划。

**G5 本文覆盖不到的。** (1) 动态功耗与能量（需工艺参数）；(2) RRAM 的写入/漂移/IR drop；(3) trunk 在 final 之后的漂移（只用到 20150616/17 等已允许 session）；(4) N=96/256 上冻结算法的精度（只有结构外推）。

---

## 附录：本次 CPU 测量（`/tmp/arch_scratch`，未提交）

| 脚本 | 做了什么 | 关键输出 |
|---|---|---|
| `measure_input.py`、`measure_sparsity.py` | 5 个 sub-M session（20140626/0929、20150511/0616/0617）的计数稀疏度、最大计数、trial 长度分布 | 零计数 79.8–84.7%；每 bin 非零 3.0–4.5；N=26 session 的 E[nz(nz+1)/2]=14.0；最大计数 5–7；trial 长度 p95 247–293、最大 800–5,144 |
| `arch_counts.py`、`mem_bw.py`、`commit_tot.py`、`bw_tot.py` | 由 checkpoint 形状与上文公式生成全部 MAC/字节/SRAM/ADC 表 | 本文所有【D】数字；检验与 `calib_cost.finalize_macs`（15,185）、`fwd_refine`（350,720）、`state_budget`（17,216 值）一致 |
| `range_probe.py`、`outlier_ch.py` | 用 `mamba3_streaming.decoder_step`（CPU，fp32）跑 20140306（源，自带表）与 20140626（target，electrode prior 表）各 12k bin，记录状态/激活/非线性输入域 | §5.1 P2/P3、§5.3 的范围；残差流离群通道 |
| `cca_precision.py` | 在 20140626 上复现 month 配置的 ridge-CCA（`published_baselines`），在原语输出上截尾数，量 latent 误差与条件数 | §1.3 |
| `fold_quant.py` | 折叠 (μ,σ) 后再量化 vs 先量化的 z 空间误差 | §5.1 P7 |

局限：所有测量是 fp32 CPU 探针；样本是 1–2 个 session；目标侧用的是 electrode prior 表而不是 P0 表（动态范围可能不同）；CCA 探针是尾数截断而非定点；没有读 final session。
