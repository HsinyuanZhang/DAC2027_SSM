# 给 ChatGPT Pro 的提问（2026-10-15）

> 用法：把下面分隔线以下的全部内容粘贴给 ChatGPT Pro。它的回答存到 `docs/chatgpt_pro_answer_20261015.md`，之后我们一起筛选。

---

## 你的角色

你是一位同时熟悉 compute-in-memory（CiM）硬件、低功耗 ASIC 和 neural decoding 的审稿人兼合作者。我们要投 **DAC 2027**（软硬件协同设计的会议，不是 ICML）：abstract 截止 2026-11-11，6 页 full paper 截止 2026-11-18。离截止大约 4.5 周，RTL 和商业综合由我们另一位成员负责。

请帮我们找到**有创造性、而且 4 周内做得完**的方向。重点放在 quantization（逐层或混合精度）、approximate computing、weight/parameter update 的方式，以及 set-attention 的硬件化。不要把重心放在“提出一个全新的学习算法”上。

## 系统设定（硬约束）

1. **任务**：跨 session（跨天到跨年）的生物信号解码，统一用 supervised calibration。新 session 只给 B 个有标签 trial，B ∈ {1, 2, 4, 8, 16, 33}。
2. **模型**：SSM backbone（Mamba-3 风格，宽度 64–128，2–4 层，state 10–16；时间轴用 SSM 扫描）。前端有一个 per-session 的 read-in table，把 N 个 channel 或 unit 映射到 K 维潜变量（K×N），后端接 readout。
3. **硬件预算**：
   - trunk 权重放在 Mb 级 2T2R RRAM（一个差分对存 {−1, 0, +1}，k bit 用 bit-slicing），部署后**不写 RRAM**；
   - SRAM 总量 < 128 KB；calibration 的结果是静态参数，≤ 32 KB；
   - 推理时，相比无校准模型，额外代价最多是一次线性 read-in。
4. **target 端 adaptation 的规则**：**不允许反传穿过 trunk**。允许的操作有：
   - 冻结 trunk 的前向，也就是在 RRAM 上做固定权重 MAC；
   - 流式累加；
   - read-in 和 readout 两端的闭式求解（如小矩阵 ridge）或 local learning rule；
   - 在 source 上训练好、之后冻结的固定权重网络（例如 encoder）的前向。
5. **数据集**（四个都要在论文里出现）：
   - DANDI 000688 sub-M：intracortical，unit 数随 session 变化，2D kinematics 回归；按间隔分 days（约 1 周）、month（约 3.7 个月）、year（约 15 个月）三层；
   - FALCON M2：intracortical，通道固定，手指 kinematics 回归，公开 benchmark；
   - Hyser HD-sEMG：20 名受试者，每人 2 次采集、分在不同天，每次 256 通道高密度电极；1dof/ndof 是手指力回归，pr 是 34 类手势；
   - NinaPro DB6：10 名受试者，采集 5 天、每天 2 个 session，14 通道，7 类抓握，专门研究多天重复性。

## 已有结果（有正面也有负面，请据此判断）

1. **Gradient 方法与 forward-only 方法的差距**：B=8 时，在 000688 上按 days / month / year 三层：
   - ui_prior（gradient 只更新 read-in table，但需要反传穿过 trunk，**现在不允许**）：0.536 / 0.500 / 0.397；
   - 最好的 forward-only fixed-weight encoder：0.347 / 0.105 / 0.131（R²）。
2. **主要原因：统计量太噪**。per-unit 统计量只来自 B 个 trial，噪声太大；encoder 又是逐个 unit 独立映射的。佐证：同一个 encoder 换用整段 recording（约 5 万个 bin，不用标签）算统计量，达到 0.673；只用 B 个 trial 时为 0.347。
3. **Set-attention 的证据**（APST 在 000688 sub-C 上的消融）：只用 activity 为 0.402；加入 behavior-association profile（per-unit 的 tuning 描述）为 0.778；把 profile 打乱后约为 0。说明“每个 unit 是谁”的描述，是跨 session 对齐的关键。
4. **2T2R PTQ 敏感度**（在 source validation 上测）：
   - 全部 8 bit 几乎无损；全部 4 bit 损失约 0.003–0.017；全部 ternary 损失约 0.04–0.05；
   - B/C 行和 FFN 用 ternary 几乎不受影响；in_proj 的 z/x 行和 out_proj 最敏感；
   - “FFN + B/C ternary，其余 8 bit”约 1.45 Mi 个差分对，损失约 0.01。
5. **一个评估上的陷阱**：在 target session 上，跨 session 的残差很大，量化引起的 R² 变化被一个符号随机的交叉项主导，会出现“加噪声反而更好”的假象。量化必须在 source validation 上评估。
6. **在 M2 上**，到目前为止没有任何 calibration 超过 source 平均的 prior。

## 我们想重点做的方向

1. **Set-attention 作为核心**：把神经活动看成 unit 组成的集合（population coding），用 set-attention 总结群体特征，并据此给每个 unit 定位。sEMG 的 channel 也可以看成集合，HD-sEMG 还带有 grid 的空间坐标。
   - 我们的初步判断：用 learned seeds 做 query 的 PMA 或 inducing-point 式注意力，打分可以改写成 (W_kᵀq)ᵀ·desc_i，这是固定权重 MVM，可以放进 RRAM；softmax 和加权求和放在数字端，用 online softmax 流式计算。
   - 注意力只在 calibration 时运行一次，结果折叠成 SRAM 里的 K×N read-in table，推理时零额外代价。
   - unit 之间的 full self-attention 是 N×N 的动态矩阵乘，对 CiM 不友好。
2. 面向 DAC 的切入点：逐层或混合精度 quantization、approximate computing、SRAM 中参数的更新方式、calibration 数据通路与推理数据通路的复用。

## 请回答以下问题

**A. Set-attention 的硬件化**
- A1. 在“只在 calibration 时运行、折叠成静态表”的前提下，哪种 set-attention 变体最适合 RRAM 加数字后端？候选包括 PMA、ISAB、linear attention、slot attention、Perceiver 风格的 latent array，也欢迎你提别的。请给出每种变体的运算分解：哪些是固定权重 MVM，哪些是动态 × 动态，以及大致的 SRAM 占用。
- A2. 怎样让 set-attention 在只有 B 个 trial 的噪声统计量下仍然可靠？比如让 unit 之间的注意力去“借力”，用群体信息给单个 unit 的噪声描述去噪。有没有可以借鉴的硬件友好做法，例如 shrinkage、empirical Bayes、hierarchical pooling？
- A3. softmax 的近似：base-2 exp 加 LUT、ReLU 或 sigmoid attention、top-k、低比特 attention score。在“只算一次、结果折叠成表”的场景下，近似误差会怎样传播到 read-in table 和最终解码？哪些近似几乎不会造成损失？
- A4. HD-sEMG 有已知的电极空间坐标，跨天的主要变化是电极平移、旋转和阻抗变化。set-attention 应该怎样加入 positional encoding，才能对这些变化稳健？硬件上有没有便宜的做法？

**B. Quantization 与 approximate computing**
- B1. 对 SSM trunk，在 2T2R 的 {−1, 0, +1} 单元上，有没有比“逐层组选 ternary 或 8 bit”更有意思的做法？例如 per-mode 精度、和 calibration 联动的精度分配（让 SRAM 里的 read-in 吸收 RRAM 的量化误差）、量化感知的 source 预训练。
- B2. 一个可能的创新点：**冻结、低精度的 RRAM trunk + 高精度、可写的 SRAM 边界参数**。read-in 和 readout 在 calibration 时，同时补偿跨 session 漂移和 RRAM 器件误差（每颗芯片不同的非理想性）。这个想法有没有先例？怎样设计实验才能把两种误差分开？
- B3. SSM recurrence 的 approximate computing：state 位宽、decay 系数的量化、selective dt 的近似（LUT 或分段线性）、skip 或 event-driven 更新（spike 数据很稀疏）。哪些方法在 DAC 的语境下有新意？

**C. 参数更新方式（不反传 trunk）**
- C1. 在 read-in 和 readout 两端，有哪些硬件友好的更新方式？我们想到的有：闭式 ridge 或 RLS（d² 存储）、对角或低秩近似的 RLS、sign-based 或量化的 local update、用固定反馈矩阵近似 Jᵀ（feedback alignment）、交替优化。请比较精度潜力和硬件代价（MAC、SRAM、除法或开方）。
- C2. 用 forward-only 的方法估计梯度，比如 zeroth-order 或 SPSA、forward gradient、在 read-in 上做扰动加前向，B 个 trial 的数据量够不够？在 RRAM trunk 上，前向很便宜而反传不允许，这种不对称能不能变成卖点？
- C3. 有没有办法让 calibration 时的数据通路（累加、outer-product、小矩阵求解）和推理数据通路（SSM 的 outer-product 状态更新、read-in MVM）共用同一套硬件单元？

**D. 论文层面**
- D1. 给出 3 条可以在 4 周内完成的 DAC paper storyline。每条要说明：核心贡献（排在第一的是“硬件友好的 adaptation 设计”，第二是“它的硬件实现”，第三是“借助 CiM 的硬件友好 SSM”）、必做实验、预期的主图或主表、最大的风险和退路。
- D2. 你认为审稿人最可能质疑的三点是什么？我们应该提前准备哪些对照实验？

## 回答要求

- 每个 idea 都要写清：机制（一两句话）；为什么对硬件友好（运算类型、存储、精度）；如何用我们的数据验证；4 周内是否可行；相关文献。
- **文献请只列你确定存在的**，给出作者、标题、会议和年份；不确定的请明确标注“不确定”。
- 对新意的风险要坦诚：如果某个想法已经有人做过，请直接指出是谁。
- 优先给出可以直接做 ablation 的具体设计，而不是笼统的方向。
