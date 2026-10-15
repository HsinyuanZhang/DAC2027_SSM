# DAC 2027：冻结 RRAM 递归 SSM 的低预算跨 session 校准

日期：2026-10-15  
依据：用户提供的 `chatgpt_pro_prompt_20261015.md`（以下称 [U]）及文末列出的原始论文／官方资料。  
性质：选题与实验设计备忘录。没有运行用户代码、复现实验、完成器件测量或综合；以下资源数字为注明尺寸的估算，收益均为待验证假设。

## 0. 结论与必须先锁定的边界

**主线推荐：可靠性加权的群体统计 → 一次性生成静态 read-in → 冻结的低精度递归主干。**

把 set-attention 当作“校准时使用的参数生成器”，而不是新增的在线 attention 层；把量化的创新候选定义为“合法校准之后仍不可补偿的误差决定精度”，而不是再做一次逐层 ternary/8-bit 选择。第一优先级是改善受 B 个 trial 限制的 adaptation，第二优先级是证明静态表与共享数字数据通路的资源收益，第三才是 SSM 的 CiM 部署。

### 0.1 已有结果支持什么、不支持什么

[U: L34–L44] 报告的结果是本建议的出发点，并非本次独立复现：

- B=8 时，合法 forward-only encoder 在 DANDI days/month/year 的 R² 为 0.347/0.105/0.131，而需要穿过 trunk 反传的 `ui_prior` 为 0.536/0.500/0.397。**当前主要矛盾是适配质量，不是几个千分点的 PTQ 损失。** `ui_prior` 应保留为非部署性能参考，不能混入符合约束的方法。
- 用整段 recording 计算统计量达到 0.673，是信息预算改变后的诊断参考，不能作为 B-trial 合法方案。它提示统计量质量的重要性，但变化也可能来自行为覆盖、分布覆盖或统计估计方式，不足以单独证明“全部差距都来自方差”。
- APST 的 sub-C 消融支持正确 association profile 的重要性；它既不证明 attention 必不可少，也不能直接预测另一套 sub-M/SSM 实现的收益。
- M2 的最佳方案目前是 source-average prior。方法必须允许保留 prior，而不是规定所有 session 必须产生非零更新。保留 prior 是一个可选动作，不是对未知 target 性能不下降的数学保证。

### 0.2 四个硬边界

**静态生成不等于任意折叠。** 若原 APST/POSSM attention 依赖每个时刻的 activity，则通常不存在一张固定表与其精确等价。需要直接训练满足

\[
W_s=G_\phi(D_s),\quad u_t=W_s x_t+b_s,\quad \hat y_t=V_s f_{\theta,Q}(u_{\leq t})+c_s
\]

的架构。目标端只有统计量累计、固定网络前向和被允许的局部求解；source 训练可以反传。推理时可以说“无在线 attention”，不能笼统说“零成本”：read-in 的 SRAM 读取、MVM 和 readout 实现均须计入。

**PMA 只产生群体摘要，不自动产生每个 unit 的列向量。** 从摘要到 K×N 表还需要逐 unit 的条件化生成器；必须计入其固定权重、前向计算、激活和数字非线性。

**最终参数预算和工作 SRAM 预算必须分别核算。** K=64、N=256 的 16-bit 表已经占 32 KiB，尚未包含 readout、偏置和 scales。初版建议在此示例尺寸用 8-bit read-in（16 KiB）。N=256 仅为估算例子，不能代替 DANDI 的实际最大 unit 数。

**生成 read-in 之后再拟合 readout，需要数据重放。** 表在读完 B 个 trial 后才确定，非线性递归 trunk 在新表下的特征不能由旧表的 Gram 统计量一般性恢复。基础版本采用一次读取校准数据、固定 readout；增强版本可以使用明确计费的数据重放，但不能默认 B 个 trial 免费保存在 128 KB 内。例如 N=256、1024 bins、每点 8 bit 已为 256 KiB。原始高采样率信号通常还会更大。

## A. Set-attention 的硬件化

### A1. 首选固定种子 pooling + 逐 unit 条件化生成，而不是完整 ISAB

令 N 为 unit 数，p 为 descriptor 维度，m 为种子数，d、v 为 key/value 宽度。以下 SRAM 为工作区的数量级估算，不含最终 read-in 表、源模型固定权重以及时间序列存储。

| 方案 | 可放固定权重 RRAM 的部分 | 数字端动态×动态部分 | 时间/存储性质与建议 |
|---|---|---|---|
| DeepSets mean/variance pooling | 逐 unit encoder、输出 MLP | 累加、方差及可选可靠性加权 | 最低成本基线；没有复杂注意力，但可能已经解决大量噪声问题 [R02] |
| PMA + 条件化表生成 | 固定 seed 的打分可合并为一次固定投影；value 投影、生成器 MLP | softmax；attention 权重×value；群体摘要与各 unit 的条件化 | 优先；注意 PMA 本身不输出每个 unit 的定位 [R01] |
| ISAB | 第一阶段种子 query 的线性投影；两阶段 Q/K/V 投影 | 第二阶段 unit query×由本次集合生成的 latent key；两阶段加权和 | 两次 O(Nmd) attention；一个 block 值得作为性能对照，不应先堆多层 [R01] |
| Perceiver latent array | 初始固定 query 的第一轮投影；其余固定线性层 | 更新后的 latent query×data key；latent self-attention；输出解码 | 一轮、少量 latents 可以接受；多轮会增加动态运算和重读 [R05] |
| Linear attention | Q/K/V 固定投影 | φ(k_i)v_iᵀ 的 outer-product 累计；query 与累计矩阵收缩 | 工作 SRAM O(dφv)；线性于 N 不代表比少量 PMA seeds 更便宜 [R03] |
| Slot attention | Q/K/V、GRU/MLP 的固定权重投影 | 随迭代变化的 slot query×key；竞争归一化；加权更新和门控 | 约 O(TNmd)，收敛与量化更难；本轮不优先 [R04] |

以 N=256、p=16、m=8、d=v=32 为例：FP16 descriptor 为 8 KiB，8-bit K=64 表为 16 KiB。固定种子 PMA 的 online softmax 若每个种子保存 maximum、normalizer 和 32 维 FP32 加权和，核心累计区仅为

\[
m(v+2)\times4=1088\ \mathrm{bytes}.
\]

考虑 value、MLP 和输出缓存，先给 2–6 KiB 工作区。ISAB/Perceiver 采用逐 unit 流式处理时也不必保存 N×m attention 矩阵；若缓存 N×32 的 FP16 K、V，额外占 32 KiB，若不缓存则重复投影。Slot attention 同样存在“缓存约 32 KiB 或多次重算”的选择。Linear attention 以 dφ=v=32、FP32 累计计算，矩阵与 normalizer 为 4224 bytes，另有投影工作区。

这些是本备忘录的实现估算，不是上述论文报告的 ASIC SRAM。**完整 self-attention 也可以流式/分块，不能把‘必须保存 N² 权重’当成对照前提；它仍有 O(N²d) 动态计算量。**

建议实现的具体形式：

\[
s_{ji}=(W_k^Tq_j)^T\phi(d_i)/\sqrt d+\log(r_i+\epsilon),\quad
z_j=\sum_i\operatorname{softmax}_i(s_{ji})W_v\phi(d_i).
\]

q_j、W_k 均在部署前固定，因而打分投影可合并。若中间存在非线性，不得跨非线性合并。加权求和仍是动态权重乘动态 value，在数字端执行。

随后逐 unit 生成：

\[
\beta_{ij}=\operatorname{softmax}_j(a_j^T\phi(d_i)),\quad
\mu_i=\sum_j\beta_{ij}z_j,\quad
W_s[:,i]=g_\phi(d_i,\mu_i,r_i).
\]

这里 a_j 和小型 g 都固定。这个“固定路由的群体反馈”是候选简化结构，不等同于完整 ISAB。它避免第二阶段的动态 query×动态 key，但仍保留动态混合。descriptor 可以在 SRAM 里重读两次，不需要重放 B 个 trial。

**新意边界：**集合 pooling 已有 Set Transformer/DeepSets，网络生成另一网络参数已有 HyperNetworks [R01,R02,R06]。不能把 PMA 或生成静态权重本身写成首创。值得验证的是：受噪声与硬件约束的校准方式，能否以小型静态边界代替在线集合处理，并在精度、校准读写和推理成本上形成收益。

**四周可行性：高。** 先实现一个 pooling block、m∈{4,8,16} 和两层以内的小生成器；完整 ISAB 只做对照。

### A2. 用可靠性控制“向谁借信息、借多少”，而非直接平均所有 unit

第一项消融应是无 attention 的 shrinkage：

\[
\tilde d_i=\lambda_i d_i+(1-\lambda_i)\mu_i,\qquad
\lambda_i=\frac{\tau_{g(i)}^2}{\tau_{g(i)}^2+\hat v_i}.
\]

\(\hat v_i\) 是 descriptor 估计噪声的估计，\(\tau_g^2\) 是 source 上估计的组内真实变异。\(\mu_i\) 可以来自 source prior、简单分组或群体 pooling。此处借鉴经验贝叶斯收缩，不主张上述简化估计器是新统计理论 [R07]。

| 具体设计 | 硬件与精度 | 最小验证 | 可行性 |
|---|---|---|---|
| Source-prior shrinkage | 每 unit 少量乘加；λ 可由计数/coverage 查 LUT | 独立 encoder vs 加收缩；B=1/8/33 | 高，优先 |
| Reliability-weighted pooling | 将可靠性加入 score，或乘到未归一化权重；只增加少量累计/LUT | 相同 PMA 有/无可靠性，检查低发放 units | 高，优先 |
| 多组层次 pooling | 先按 source 学得的响应组或 HD-sEMG 局部邻域估计 prior，再反馈 | 单全局均值 vs 多组；防止相反 tuning 被平均 | 高到中 |
| Source episodic training | source 阶段模拟同样 B、缺失 unit、校准噪声；target 不增加运算 | 相同 encoder 的普通训练 vs 匹配 B 的训练 | 高，优先 |

可靠性不应只由“当前幅度大”决定。优先考虑：trial 间一致性、有效 spike/样本数、行为设计矩阵的覆盖、有效通道标志。B=1 时不能估计跨 trial 方差，应使用 source 学到的计数—不确定性关系或强 prior；连续 bins 的相关性也不能忽略。

建议先用 source 选好的 λ 档位（例如 0、1/4、1/2、3/4、1）验证量化收缩系数是否足够。这是可做消融的移位加法方案，不是“几乎无损”的预先结论。

**训练改动比加大 attention 更优先：**从 source session 抽 B 个 calibration trials 计算 descriptor，用另一些 source trials 计算解码损失；生成器与主干在 source 上可以联合训练。不能用 target query 标签，也不能把整段 target recording 的统计量混入合法 B-budget 结果。

建议固定的最小消融链：independent encoder → +source shrinkage → +DeepSets context → PMA → PMA+reliability。若 DeepSets/shrinkage 已达到同样性能，应选更便宜的版本，而不是为了标题保留 attention。

### A3. Softmax 近似：优先保留定义，只近似数值实现

推荐顺序是：max subtraction + online normalizer → exp2 LUT/PWL → score/value 量化 → 最后才考虑改变 attention 形式。

\[
e^x=2^{x\log_2e}
\]

在实数算术下只是变量换算；直接把 e^x 改成 2^x 则改变温度。Online normalizer 也不是近似算法，有限位宽和 LUT 才引入数值误差。相关先例见 [R08,R09]。

| 候选 | 判断与消融 |
|---|---|
| exp2 LUT + reciprocal | 首选。可从 256×16-bit=512-byte LUT 开始；检查范围、分段和 saturation，不能仅报告 LUT 点数 |
| score 8/10/12 bit；value 8/16 bit | 高可行性；先保留 24/32-bit 累加器，分别量化打分、权重、value 和最终表 |
| 很小尾部截断 | 比固定 top-k 更可解释；报告被丢弃的概率质量，而非只报告稀疏率 |
| 固定 top-k | 可能丢掉很多弱但互补的 donor；还增加比较/选择逻辑。本轮低优先级 |
| ReLU / sigmoid attention | 改变模型定义，不是 softmax 的无条件替换；需要 source 重训，归一化及零分母处理仍需设计 |

可以用简单的工程界来选截断阈值：若 value 范数不超过 Vmax，删除 softmax 质量 δ 后重归一化，有

\[
\|z-z_{\rm truncated}\|\le 2V_{\max}\delta.
\]

这只是 pooling 层的界，不是最终 R² 保证。静态表的误差会在每个推理时刻持续注入：

\[
\epsilon_t^{u}=(\hat W_s-W_s)x_t.
\]

因此，“只在校准时算一次”降低的是执行次数，不是误差影响次数。至少记录三项：表/注入特征失真、固定同一模型下的预测差异、最终 target R² 或分类性能。初版不要承诺任何具体位宽几乎无损。

### A4. HD-sEMG 坐标：提供拓扑，但不能把电极坐标当成跨天的肌肉坐标

建议从 descriptor 附加归一化 (x,y)、必要的 array ID、邻域有效性开始；只在 calibration 使用。然后比较坐标辅助的局部 pooling 和单纯群体 pooling。

**关键边界：**电极网格内部的相对距离不会随整片平移而变化。因此，相对位置编码不能单独识别网格相对肌肉的重新放置；绝对网格坐标也不等于解剖坐标。坐标是 prior，behavior-association 描述仍提供任务关联。

最低成本的 source 增强是小范围空间重采样、通道增益/偏置、缺失电极与噪声增强。平移/旋转应作用于模拟的信号或 descriptor 空间场，不能只更改坐标标签而保持信号不变后声称模拟了真实移位。实际允许的变换范围从 source 验证，不预设所有 HD-sEMG 差异都由刚性变换解释。

仿射输入校正可以折入最终 W 和 bias，不增加在线层数。非线性归一化不能随意折叠。最小消融：无坐标、仅坐标、坐标+association、再加局部 pooling/增强；真实跨天结果是主证据，合成移位只是压力测试。Hyser 任务与采集说明以官方资料 [R25] 为准。四周可行性高，不建议临时加入复杂等变网络。

## B. Quantization 与 approximate computing

### B1. 值得做的是“校准后不可补偿的敏感度”，不是再发明逐层混合精度

HAWQ 已研究基于敏感度的混合精度；Quamba、SSDi8 已针对 selective SSM/SSD 的低比特部署；QS4D 已包含 S4D 的量化感知训练和硬件部署。[R10–R13] 因此，单独做 per-layer、per-channel、per-mode 精度不宜成为第一贡献。

建议定义 **calibratability-aware precision allocation**：对某个候选量化配置 Q，在 source 的模拟跨 session 任务上，执行真正允许的 B-trial 校准 C_B，然后评估独立 query trials：

\[
R(Q)=\mathbb E_{s,B,c}\left[
L_{s,\rm query}(f_Q,\mathcal C_B(f_Q))-
L_{s,\rm query}(f_{\rm FP},\mathcal C_B(f_{\rm FP}))\right].
\]

这里 c 表示芯片非理想性实例；开始时只做理想量化，等候选收敛后再加入芯片扰动。所有精度选择与超参数选择都在 source validation 上完成。

**假设：某些投影量化误差容易被允许的边界校准吸收，另一些时序/状态误差不能；前者可以更激进地量化。** 这是待验证的设计准则，不是已证明的结论。

四周内的实现范围建议如下：

| 设计 | 机制及硬件收益 | 必做实验 | 风险/可行性 |
|---|---|---|---|
| 功能块/可实现行组精度 | 从现有 FFN+B/C ternary 出发，仅细分敏感 z/x/out 块；选实际支持的 ternary/4/6/8-bit | 相同差分对数或能耗下，对比人工分组、未校准敏感度和校准后敏感度 | 高；先做少量功能块，不做大规模自动搜索 |
| Source QAT + noisy calibration episodes | source 前向注入真实 2T2R 编码、ADC/activation/state 量化与噪声；保持目标端固定 | PTQ vs QAT；正常训练 vs B-budget episodic QAT | 高到中，已有训练管线可复用时优先 |
| Head/state-block 精度 | 按实际递归状态块的敏感度分配存储位宽/scale | 8/12/16-bit state 与 24/32-bit accumulator；连续运行而非每窗清零 | 中；不要把 Mamba 的状态块误称为固定独立衰减 mode |
| 校准生成器与最终表的精度 | 保护廉价但决定整段推理输入的校准部分，单独量化 read-in 表 | encoder、descriptor、score、table 分别量化 | 高；不能只分析 trunk 权重 |

[U: L39–L42] 中“B/C 行可以 ternary”是**生成 B_t/C_t 的权重行**的结论，并不意味着 B_t/C_t 的动态数值或递归 state 也能直接 ternary。

**物理资源口径：**1.45 Mi 个差分对对应约 2.90 Mi 个 RRAM 器件，不能记成 1.45 Mbit。实际 bit-slicing 对数必须从编码和布局统计，包含零权重占用、tile 空洞、scales、ADC 精度/次数、数字重组与额外生成器权重。在线停用已有 bit planes 可以省读能量，但不减少已经制造的阵列面积；离线配置并实际减少物理切片，才能声称面积下降。

**处理 [U: L43] 的评估陷阱：**令 r=y−ŷ_FP，δ=ŷ_Q−ŷ_FP，则

\[
\mathrm{SSE}_Q-\mathrm{SSE}_{FP}=-2r^T\delta+\|\delta\|^2.
\]

target 残差大时，交叉项可主导。但“只在 source 选精度”不等于最终不报告 target 量化结果。建议分两套评价：

1. 固定同一份边界参数，比较 FP/Q 的预测失真及 source validation 损失，隔离量化扰动；
2. 各自在相同 B 预算下重新执行合法校准，再比较完整系统的 target 表现。

第二套回答系统是否有效，第一套防止误把随机抵消当成量化本身的优势。source 上同样存在交叉项，需要重复 episode 和配对统计。

### B2. 两端同时补偿 session 与器件误差：有先例，但可做更严格的受约束版本

固定物理计算核心、可训练边界不是新概念：Du 等的动态 memristor reservoir 使用训练 readout；HaLoRA 已研究固定 RRAM 主干与可写 SRAM 低秩适配的混合结构及硬件误差。[R14,R15] 你们的差异可以是：**无 trunk 反传、无部署后 RRAM 写入、校准后只有原本需要的线性边界、严格少样本预算。** 是否足以构成新意仍取决于具体机制及结果。

必须区分：固定芯片误差（电导偏差、固定增益/offset）、缓慢变化的误差，以及每次读取重新采样的噪声。静态参数可能补偿前两类的一部分，不能预测并消除每次随机 read noise，也不能一般性恢复饱和、秩丢失或错误时序记忆造成的信息损失。

**最容易实现的两种版本：**

- **版本一：read-in 处理 session，实际芯片特征上的 readout ridge 补偿剩余误差。** 需重放或明确划分校准数据。若表生成器没有芯片响应输入，不应宣称 read-in 自身学到了每颗芯片的误差。
- **版本二：已知 probe 序列 → 小型 chip signature → 条件化静态表。** 预存少量 source 计算的 reference response，用相同初始状态在实际 trunk 上运行固定合成 probe，将响应差压成 8–16 维摘要，与群体 descriptor 一起输入固定生成器。它不需要额外 target 标签或 RRAM 写入，也不需要重放 target 原始数据；但必须验证这些 probe 是否能够识别有用的误差。source 需要按芯片扰动 episode 训练，因此风险高于版本一。不能假定几个 probe 足够识别任意内部误差。

版本二是本备忘录提出的候选实现，不声称无先例。四周内只适合在主线稳定后尝试小原型。

**必做因子分解实验：**

| 因素 | 水平 |
|---|---|
| 数据分布 | source held-out session / target session |
| 硬件 | FP 理想 / 理想量化 / 量化+固定 chip error / 再加入动态 read noise |
| 校准 | 无；read-in only；readout only；两端；有 probe 时增加正确/错误 chip signature |
| 迁移检验 | chip c 上校准，在 c 与另一芯片 c′ 上分别测试 |

同一次 episode 的固定 chip error 在 calibration 和 query 中必须一致；动态噪声每次读取重新采样。source 噪声训练覆盖的芯片实例与测试实例分开。最终候选可用例如 20 个独立 chip seeds 评估分布，但这个数字是实验建议，不是统计充分性的保证。

这样才能区分：仅适应 session、仅有噪声鲁棒预训练、真正的 chip-specific 补偿，以及两者交互。不能用一个同时发生 domain shift 和器件扰动的 R² 增量，直接声称分别解决了两种误差。

### B3. Recurrence 近似的优先级

**先以实际代码的递推式为准。** 用户写的是 Mamba-3 风格，不能直接替换成恒定 S4D。原始 Mamba-3 涉及输入依赖系数、exponential-trapezoidal 更新、旋转/复数状态等价实现及可选 MIMO；有些实现还需保存前一输入项与相位缓存。[R23] 不必研究其理论创新，但需要读清一条 time-step 的真实算子图。

| 候选 | 机制及注意点 | 验证 | 建议 |
|---|---|---|---|
| state 低比特存储、较宽累加 | SRAM 8/12/16 bit，PE 累加 24/32 bit；每 head/block 独立 scale | 长序列、切换瞬态、饱和率、各 head 输出失真 | 第一优先；单独不是新意 |
| 精细表示接近 1 的 decay | 量化 −log α 或泄漏量 1−α，避免粗量化使 α 直接卡到 1；必要时 h−δh 更新 | 比较真实状态保留时间、稳定性与量化误差 | 第二优先；必须保持实际更新式其它项一致 |
| dt/softplus/exp 的 LUT/PWL | 与 attention 共享函数近似单元，使用不同输入范围和 scale | 分别近似与联合近似；不要只测单个函数误差 | 高可行性；属于支撑模块 |
| 输入稀疏时跳过 read-in 的零项 | 只省确定为零的输入 MAC/字典累积 | 真正活动率、地址/控制开销、等价性 | 可以作为基础优化 |
| 无 spike 时跳过整个 SSM | 原始输入为零不保证内部投影、偏置、上一输入项和各层输入为零 | 需证明等价或报告近似误差；还需覆盖输出时钟 | 本轮不推荐作为主创新 |
| state 未变化时抑制写入 | 有可能省 SRAM 写，但判定本身可能仍需读和算 | 分别统计 read/compute/write，加入比较与 metadata 开销 | 可小消融，不能宣称同步省全部计算 |

即使近似 decay 的幅值稳定，旋转实现的数值误差也可能改变状态；如代码包含 sin/cos/RoPE，需一起审计。持续运行的评价必须保留自然 trial/session 的 reset 规则，不能为了定点好看而每几十个 bin 清空状态。

## C. 不反传 trunk 的参数更新

### C1. Readout 有真正的闭式解；read-in 必须说明局部目标

设冻结 trunk 的输出特征为 z_t∈R^d，输出为 y_t∈R^q，source readout prior 为 V0。优先使用 prior-centered ridge：

\[
G=\lambda I+\sum_tz_tz_t^T,\qquad
R=\sum_t(y_t-V_0z_t)z_t^T,\qquad
V=V_0+RG^{-1}.
\]

这对应平方误差加 λ||V−V0||² 的目标；无有效样本或强正则时回到 prior。不要把公式中的逆矩阵理解为硬件必须显式求逆：解线性方程即可。source 可以训练特征和正则超参数，target 仅累计与求解；使用闭式求解的 few-shot/meta-learning 本身已有先例 [R16]。

| 更新方式 | 计算/存储数量级 | 数值部件 | 精度潜力与本轮判断 |
|---|---|---|---|
| Batch ridge | T(d²+dq)+O(d³)，Gram FP32 为 4d² bytes | Cholesky 有开方；LDLᵀ 可避免开方，仍有除法/倒数 | 首选；相同定义下的闭式最优只针对线性 readout |
| Full RLS | 每 bin O(d²+dq)，存逆协方差 4d² bytes | 通常每步一次主要 reciprocal；需保持对称/正定数值性质 | 适合持续更新；一次性 calibration 未必优于最后求一次 ridge |
| Diagonal RLS/ridge | O(d) 协方差状态 | 逐维 reciprocal/LUT | 很便宜；丢弃相关性，不能称为一般 ridge 的精确等价 |
| Block-diagonal | block 宽 b，约 O(db) 协方差状态和每步工作 | 小块求解 | 本轮值得做 b=4/8/16；source 预分组/条件化后更合理 |
| Low-rank RLS | O(dr+r²) 状态，另有子空间维护成本 | 正交化/PSD 维护可能复杂 | source 固定低维投影比在线更新低秩子空间更稳妥 |
| LMS/NLMS/sign update | O(dq)，无需 d² 协方差 | NLMS 需要归一化倒数；sign 降乘法但不保证样本效率 | 低成本对照；少 trial 时需严控学习率/更新次数 |
| Feedback alignment | 固定反馈矩阵+误差×输入 outer-product | MAC/符号运算；还涉及时间 credit assignment | 对冻结主干不能照搬原始 FA 的学习结论；本轮低优先 [R17] |
| 交替优化两端 | 每次改 read-in 后重新前向/统计 readout 特征 | 继承局部求解或 ZO 的开销 | 不是两个小 ridge 就能精确解原任务；需重放，慎用 |

FP32 完整 Gram：d=32/64/128 时分别为 4/16/64 KiB。对称打包可约减半，但访问更复杂。若 d=128 超预算，可在 source 固定一个低维投影 P，以 d=32/64 做校准，再把解出的 readout 与 P 合并成最终 head；P 在 calibration 中的计算和固定权重必须计费，不能说免费。

**对 read-in，以下命题不成立：**因为它是线性的，所以在 y=fθ(Wx) 中能够对 W 直接 ridge。fθ 是非线性递归映射，最终任务损失对 W 一般不是二次函数。

一个合法、便宜的局部目标是先估计 tuning/profile：令 φ(y_t)∈R^{p_y}，

\[
G_y=\lambda_yI+\sum_t\phi(y_t)\phi(y_t)^T,\qquad
C_y=\sum_t r_t\phi(y_t)^T,\qquad
\Beta=C_yG_y^{-1}.
\]

共享的求解矩阵是 p_y×p_y，不是 N×N；每个 unit 的 β_i 再经过固定群体生成器得到 W[:,i]。这解决的是一个局部描述估计问题，不能宣称它是下游解码损失的闭式解。B 很小时加入 shrinkage、coverage mask 和 source prior。

对分类任务，B 是总 trials 而非每类 B。没有出现的类别不能凭空得到 target 统计。DB6 可保留 source head 对未观察类别的参数，并在 source 上决定是否允许其它类别更新；这仍不构成对 target 全类别准确率不下降的保证。Hyser 主实验建议先固定 1dof 或 ndof 回归，而不是同时扩展 pr 分类的全部配置。[U: L15,L29–L30]

### C2. ZO/SPSA 可以尝试，但应限于低维、可折叠参数

SPSA 与前向微调已有成熟先例 [R18,R19]；NoiseZO 更已在 DAC 2025 研究 RRAM 上的 forward-only 训练，QZO 在 ICLR 2026 研究量化模型的 ZO。[R21,R22] 因此“RRAM 前向便宜、反向贵”不再是独立的新颖点。NoiseZO 的写噪声扰动不符合你们部署后不写 RRAM 的约束，不能原样照搬。

**不建议直接扰动 K×N 表的全部自由度。** N=256、K=64 时是 16,384 个参数。两次评估估计一个随机方向，不等于两次评估就能可靠识别全部维度。

建议只试 8–16 个 source 预先组织好的自由度，例如 latent group 的 gain、有限组的 shrinkage 强度或很小的残差坐标：

\[
W_s(\vartheta)=D(g(\vartheta))W_s^{(0)},\qquad
\hat g=\frac{L(\vartheta+c\Delta)-L(\vartheta-c\Delta)}{2c}\Delta.
\]

这里 Δ 取 ±1，校准结束后乘法合并到 W_s，推理不保留适配分支。不要默认存储一个 KN×r 稠密残差基底很便宜；若需要这种基底，应计入固定 RRAM 容量或实时生成成本。

**数据是否够，和前向是否便宜，是两件事。** 很长的一个 trial 可以有很多 bins，但相互相关且行为覆盖可能不足；多跑前向不会补上没有观察到的方向或类别。B=1 时尤其依赖 source prior。建议主线先不用 ZO，之后仅在 B=8/16/33 上做小规模补充，再决定是否扩展全部 B。

最低要求：

- 每个方向两次完整校准序列前向；I 次更新、P 个方向需要约 2IP 次序列评估，另计基线/验证。把重放字节数、SRAM 表更新和 PRNG 开销一起统计。
- 正负扰动使用相同输入、相同自然 reset 规则，并重置所有 state/上一输入项/相位缓存。不能使用一边已经演化过的 state 作为另一边初值。
- 固定 chip error 保持一致，但真实硬件上的随机 read noise 不能假设可通过设置随机种子严格配对消除。扰动要高于量化死区，又不能大到损坏局部近似。
- target 上不能利用 query/test 标签选择迭代数、扰动尺度或最终 checkpoint；这些在 source validation 预定。

**Forward gradient 与纯前向估计不同。** Baydin 等的方法用 forward-mode automatic differentiation/JVP [R20]，仍需沿 trunk 传播导数信息和相应 tangent state。它不需要反向遍历，却也不是你们允许的普通固定权重前向 MAC。除非明确扩展硬件/操作约束，本轮将其排除。

### C3. 值得实现的是共享数字 rank-one/MAC 引擎，RRAM 只做固定投影

| 阶段 | 可共用的基本操作 |
|---|---|
| profile 与可靠性统计 | r_tφ(y_t)ᵀ、平方/计数累计 |
| pooling | scalar×value 的加权累计，normalizer 更新 |
| readout 校准 | z_tz_tᵀ、残差×z_tᵀ |
| 小矩阵求解 | rank-one 更新、三角/对角处理、倒数 |
| SSM recurrence | state 缩放，加一个或多个输入 outer-product；按实际 Mamba-3 代码处理前一项/旋转 |
| read-in/readout | SRAM 权重的 dot-product/MVM |

Mamba-3 的可选 MIMO 可能表现为多个 rank-one 项或小矩阵乘法；不要直接将 MIMO rank 乘到状态 SRAM 公式中，应统计实际缓存 tensor。[R23]

建议原型为少量 SIMD MAC lanes、较宽 accumulator、banked SRAM，加共享 reciprocal/exp2/PWL 单元。校准和正常推理分阶段调度，不要求每个算子都有一块独立阵列。LDLᵀ 的动态矩阵操作在数字/SRAM 端完成，不能为了复用 RRAM 而在校准时重写其权重。

**共享算子本身不是充分新意。** 要比较：分离单元与统一单元的面积、bank 冲突、模式切换、校准总延迟、实时推理最坏延迟，以及 SRAM 读写。不要仅画共用模块框图就声称完成架构贡献。

### 一个可核查的 SRAM 草案

以下只说明预算可设计，不是已经审计你们的真实模型。取 N≤256、K=64、descriptor p=16、局部 profile 基函数 p_y≤8、readout 求解 d=64、输出 q≤7（Hyser 在此选择 force regression，非 34 类 pr）。

| 项目 | 预留 KiB |
|---|---:|
| 最终 8-bit read-in | 16 |
| readout、bias、scales、mask | 4 |
| 递归状态及必要的上一输入/相位缓存 | 32 |
| descriptor、profile 累计、可靠性统计 | 16 |
| pooling/生成器工作区 | 4 |
| readout Gram（FP32 完整 64×64） | 16 |
| cross-statistics 与小求解工作区 | 4 |
| I/O 和其它临时缓存 | 8 |
| **保守相加** | **100 KiB** |

100 KiB=102,400 bytes，低于 128 KB 即使按十进制解释的上限；最终静态参数约 20 KiB。不同阶段还可覆盖复用，但这张表没有利用全部复用收益。额外双缓冲 read-in 需要另加 16 KiB。

**未包含：B 个 trial 的完整数据重放缓存。** 状态 32 KiB 也只是预留，实际状态若超过必须重新分配。要从 `state.numel()`、内部 expansion、head、存储位宽及所有缓存逐项计算，不可仅用论文的 model width。固定生成器的权重放 RRAM/ROM 时，虽不占此 SRAM 表，也占面积和总存储容量。

## D. 论文故事、实验和四周排期

### D1. 三条可选择的 storyline

### 路线 1：Population-conditioned static calibration（首选）

**核心贡献顺序：**①用可靠性与群体 prior，从少量 supervised trials 生成静态 read-in，而不反传 trunk；②固定种子 pooling、表生成和共享数字累计的实现，明确 phase 与峰值 SRAM；③冻结 RRAM 上的混合精度递归推理。

**必须跑：**independent encoder、+shrinkage、DeepSets、PMA、PMA+reliability；在四个数据集报告完整 B 曲线。DANDI 保留 days/month/year 三层；Hyser 先锁定一个 force regression 任务；DB6 报 balanced accuracy 或 macro-F1 及类别覆盖；M2 保留 prior 强基线。source 的训练轮数、encoder 容量和校准信息预算尽量匹配。

**主图/表：**图 1 为 calibration → SRAM table → recursive inference 两阶段数据通路；图 2 为四数据集的 B-performance 曲线；表 1 为精度、校准 passes、MAC、SRAM、参数写入和推理能耗；另以紧凑消融说明 pooling 与可靠性的独立增益。

**最大风险：**pooling 只是参数更多，或与简单 shrinkage 相同；calibration-only 表无法替代动态 APST/POSSM 的表达能力。**停止条件：**第一周若 PMA+reliability 不能稳定优于加强后的 independent/shrinkage 基线，不再堆多层 attention。**退路：**采用更便宜的 shrinkage/DeepSets，把实证贡献放在低成本静态适配与实际硬件收益；不要宣称 attention 是关键。

**四周可行性：最高，但以现有训练和四数据集预处理可运行为前提。**

### 路线 2：Calibratability-aware precision（第二选择）

**核心贡献顺序：**①定义并实现“受限边界校准能补偿什么”的 adaptation 机制，可选实际芯片 readout 校准或小型 chip signature；②依据校准后残差设计物理可实现的 2T2R 精度分配和边界 SRAM；③以递归 SSM 验证持续推理的成本与稳定性。

**必须跑：**手工分组、未校准敏感度、校准后敏感度三种 allocator；相同物理差分对数/面积或相同系统能耗下比较；read-in only/readout only/both；domain×chip-error 因子实验；最终方案覆盖四数据集，完整器件因子矩阵可先集中于代表性任务并明确适用范围。

**主图/表：**精度—RRAM 对数/能耗 Pareto；各功能块“校准前后敏感度”热图；芯片内与跨芯片校准效果；校准参数占用和写入为零的 RRAM 数据通路。

**最大风险：**边界无法补偿重要递归误差，或者先校准再测精度改变不了现有 allocator 排名。**退路：**回到 source QAT 与已有敏感度分组，作为路线 1 的支撑模块。若没有实际芯片响应参与，就将表述限制为硬件噪声鲁棒训练，不能叫 per-chip calibration。

**四周可行性：中。** 先验证“校准确实改变可用精度”这一现象，再建设自动 allocator；不要一开始搜索所有层、所有位宽、所有 chip seeds。

### 路线 3：Prior-centered boundary updates on a shared recurrent engine（架构导向备选）

**核心贡献顺序：**①从有监督统计和 prior 出发，选择低维/block 更新，校准信息不足时抑制更新；②共享 outer-product、状态访问、小矩阵求解和 SRAM bank，实现校准/推理复用；③在冻结 RRAM 递归 SSM 上验证四种生物信号。

**必须跑：**full ridge、block ridge、diagonal、NLMS、prior only，必要时加小维度 SPSA；固定同一模型比较独立单元/共享单元；计入重放和函数近似；展示 block 大小、B、总能耗和目标精度的关系。与已有局部/无主干反传工作做定位比较，不把“不反传”本身当新意。[R16–R24]

**主图/表：**复用时序与 bank map；校准总能耗—最终任务误差曲线；分离/共享硬件的面积与最坏延迟表。

**最大风险：**最后只是“已有 ridge + 普通 MAC 复用”，独立新意不足；数据重放占掉所有收益。**退路：**把引擎作为路线 1 的实现部分，而非独立故事。没有重放来源时，不承诺两个边界都用完整 B 数据重拟合；采用单遍表生成+prior head，或明确降低输出拟合所用样本量。

**四周可行性：算法和 RTL 基础具备时为中到高；独立论文新意风险高于路线 1。**

### D2. 三个最可能的审稿质疑

| 质疑 | 必须准备的证据 |
|---|---|
| “这是 Set Transformer/HyperNetwork 加现成 CiM，没有必要的新机制。” | 同信息预算、相近参数量的 independent/DeepSets/shrinkage/ISAB；证明静态表的合法性；分别展示可靠性、群体信息与数据通路的增益；明确已有 attention/混合精度/ZO 先例 |
| “Few-shot 协议或适配收益不公平，M2 负结果被隐藏。” | source/target 与 calibration/query 严格划分；所有归一化/滤波统计来源；B 总 trials、标签秒数和类别/运动覆盖；full-recording stats 只作为 oracle-like 诊断；source-only 选超参；M2 prior；非部署 ui_prior 明确标识 |
| “能耗与存储估算漏掉实际系统，128 KB 根本装不下。” | 所有状态/cache/生成器/求解/双缓冲/重放的容量与生命周期；完整 ADC/DAC、bit slicing、bank、函数单元；SRAM+数字 MAC 强基线；均摊校准和持续推理成本；模拟/综合/实测结果严格区分 |

DRAFT 已出现在 DAC 2025 官方程序中，题目直接涉及绕开预训练 backbone 的反传 [R24]。本次只核实了存在、作者和会议信息，未取得其完整技术正文，因此不在此断言它采用何种替代梯度机制或给出与它的定量优势。

### 统一实验协议

四个数据集用相同硬件算子与校准原则，不要求同一组模型权重跨模态通用。

- **DANDI sub-M：**三种间隔分别报告；同一 B 内所有方法用同一组 trials；variable-N 的 prior 由合法的 unit descriptor/生成规则定义，不能用未知 target unit 的跨日匹配。
- **FALCON M2：**自定义 B-trial 曲线与官方 benchmark protocol 分别标明；不能把自定义 B 的结果直接写成官方榜单指标 [R26]。prior 是必须击败或合理保留的 baseline。
- **Hyser：**预先锁定 1dof 或 ndof；说明 trial/连续力片段如何定义、总标签秒数、跨日划分和因果预处理。target 全 session normalization 或额外 MVC 数据不能不计入预算 [R25]。
- **NinaPro DB6：**按日期/session 而非随机相邻窗口划分；B 指总 trials，报告未出现的类。保持 source 分类知识，不能默认 B=1 已覆盖七类 [R27]。

所有六个 B 应出现在最终主结果中。开发初期先跑 B=1/8/33 做筛选；只对最终候选展开完整 B、芯片噪声和硬件结果，避免计算量乘法爆炸。置信区间应按 session/trial 与配对校准抽样计算，不能将相邻 bins 当独立样本。芯片随机 seeds 也不能充当生物受试者或独立 session 的数量。

建议主结果同时报告：任务指标、标注量、校准数据 passes、校准总能量、最终 SRAM 参数量、峰值工作 SRAM、RRAM 对数、每步/每秒推理能耗和最坏延迟。部署总成本写作

\[
E_{\rm deployment}=E_{\rm stats}+E_{\rm table}+E_{\rm replay/solve}+T_{\rm run}P_{\rm inference},
\]

并分别给出校准频率。一次性 attention 的功耗可以被长运行摊薄，但其固定阵列面积与最终表误差不能被摊薄。

### 排期：四周内只保留一个主故事

| 时间 | 输出物 | 停止条件 |
|---|---|---|
| 10/15–10/21 | 审计真实 time-step/state；四数据集协议可运行；independent/shrinkage/DeepSets/PMA+reliability 比较；记录数据重放需求 | attention 无独立增益则停止堆层；冻结合法数据预算 |
| 10/22–10/28 | 选定一个主机制；int8 静态表；source PTQ/QAT 与少量功能块精度实验；导出真实 tensor/访存 trace | 校准不改变精度选择就不把 allocator 升为主创新 |
| 10/29–11/04 | 关键 RTL/综合与 SRAM map；代表性芯片误差因子实验；完成四数据集主结果 | 容量超标或重放收益为负，立即简化而非新增复杂 solver |
| 11/05–11/11 | 主图主表、负结果、强基线、摘要与初稿 | 不再新增主算法 |
| 11/12–11/18 | 校验、补必要对照、复现和全文收尾 | 只修复影响结论的问题，不扩展故事 |

这里沿用附件中的 11/11 摘要、11/18 全文日期安排项目工作；正式提交须以投稿系统的截止时间及时区为准。

## 最终建议

**优先提交路线 1，把路线 2 的“校准后敏感度”作为最有价值的硬件协同子贡献，把路线 3 的共享引擎作为实现，而不是同时写三套算法。**

当前最值得用第一周证明的是：

> 在完全相同的 B 个 supervised trials、没有 trunk 反传和没有额外 target recording 的条件下，可靠性加权的群体信息是否真正优于“独立 encoder + 强 shrinkage”，并且可生成足够准确的静态 read-in？

若答案是肯定的，核心工作就有了可检验的算法—硬件接口；若答案是否定的，应据实保留更简单的校准，停止把 set-attention 当成预设答案。上述新意判断是基于已核实的相关工作，不是穷尽检索后的首创性证明。

## 参考文献与来源

以下条目均核实了存在。标为 arXiv 的条目不冒称已正式录用；只核实元数据而未取得完整正文的条目明确说明。文中候选方案与资源预算是本备忘录的建议，不是这些论文的实测结果。

[R01] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, Yee Whye Teh. **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.** ICML, 2019. https://proceedings.mlr.press/v97/lee19d.html

[R02] Manzil Zaheer et al. **Deep Sets.** NeurIPS, 2017. https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html

[R03] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret. **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.** ICML, 2020. https://proceedings.mlr.press/v119/katharopoulos20a.html

[R04] Francesco Locatello et al. **Object-Centric Learning with Slot Attention.** NeurIPS, 2020. https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html

[R05] Andrew Jaegle et al. **Perceiver IO: A General Architecture for Structured Inputs & Outputs.** ICLR, 2022; arXiv version 2021. https://arxiv.org/abs/2107.14795

[R06] David Ha, Andrew Dai, Quoc V. Le. **HyperNetworks.** ICLR, 2017. https://research.google/pubs/hypernetworks-2/

[R07] Bradley Efron, Carl Morris. **Stein’s Estimation Rule and its Competitors—An Empirical Bayes Approach.** Journal of the American Statistical Association, 68(341):117–130, 1973. https://doi.org/10.1080/01621459.1973.10481350

[R08] Maxim Milakov, Natalia Gimelshein. **Online normalizer calculation for softmax.** arXiv:1805.02867, 2018. https://arxiv.org/abs/1805.02867

[R09] Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer. **I-BERT: Integer-only BERT Quantization.** ICML, 2021. https://proceedings.mlr.press/v139/kim21d.html

[R10] Zhen Dong, Zhewei Yao, Amir Gholami, Michael W. Mahoney, Kurt Keutzer. **HAWQ: Hessian AWare Quantization of Neural Networks With Mixed-Precision.** ICCV, 2019. https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html

[R11] Hung-Yueh Chiang, Chi-Chih Chang, Natalia Frumkin, Kai-Chiang Wu, Diana Marculescu. **Quamba: A Post-Training Quantization Recipe for Selective State Space Models.** ICLR, 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/fb4b2fb2434f7cce5cb5ab50271296ee-Abstract-Conference.html

[R12] Hyunwoo Kim et al. **SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality.** ICLR, 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/9d3852f2b1fc883fc41011cb0257f0d0-Abstract-Conference.html

[R13] Sebastian Siegel, Ming-Jay Yang, Younes Bouhadjar, Maxime Fabre, Emre Neftci, John Paul Strachan. **QS4D: Quantization-Aware Training for Efficient Hardware Deployment of Structured State-Space Sequential Models.** Advanced Intelligent Systems, 2026, e202501019. https://doi.org/10.1002/aisy.202501019

[R14] Taiqiang Wu et al. **HaLoRA: Hardware-aware Low-Rank Adaptation for Large Language Models Based on Hybrid Compute-in-Memory Architecture.** arXiv:2502.19747, 2025. https://arxiv.org/abs/2502.19747

[R15] Chao Du, Fuxi Cai, Mohammed A. Zidan, Wen Ma, Seung Hwan Lee, Wei D. Lu. **Reservoir computing using dynamic memristors for temporal information processing.** Nature Communications, 8:2204, 2017. https://www.nature.com/articles/s41467-017-02337-y

[R16] Luca Bertinetto, João F. Henriques, Philip H. S. Torr, Andrea Vedaldi. **Meta-learning with differentiable closed-form solvers.** ICLR, 2019. https://www.robots.ox.ac.uk/~vgg/publications/2019/bertinetto19/

[R17] Timothy P. Lillicrap, Daniel Cownden, Douglas B. Tweed, Colin J. Akerman. **Random synaptic feedback weights support error backpropagation for deep learning.** Nature Communications, 7:13276, 2016. https://www.nature.com/articles/ncomms13276

[R18] James C. Spall. **Multivariate stochastic approximation using a simultaneous perturbation gradient approximation.** IEEE Transactions on Automatic Control, 37(3):332–341, 1992. https://doi.org/10.1109/9.119632

[R19] Sadhika Malladi et al. **Fine-Tuning Language Models with Just Forward Passes.** NeurIPS, 2023. https://arxiv.org/abs/2305.17333

[R20] Atılım Güneş Baydin, Barak A. Pearlmutter, Don Syme, Frank Wood, Philip Torr. **Gradients without Backpropagation.** arXiv:2202.08587, 2022. https://arxiv.org/abs/2202.08587

[R21] Shuqi Wang, Zhengwu Liu, Chenchen Ding, Chen Zhang, Taiqiang Wu, Jiajun Zhou, Ngai Wong. **NoiseZO: RRAM Noise-Driven Zeroth-Order Optimization for Efficient Forward-Only Training.** DAC, 2025. https://hub.hku.hk/handle/10722/359699 ; official program: https://62dac.conference-program.com/ . Note: the program renders “Zero-Order”; the institutional bibliographic record renders “Zeroth-Order”. The repository’s conference-series metadata contains a stale 2023 date, so the 2025 venue/year is cross-checked against the official DAC program.

[R22] Sifeng Shang, Jiayi Zhou, Chenyu Lin, Minxian Li, Kaiyang Zhou. **Fine-tuning Quantized Neural Networks with Zeroth-order Optimization.** ICLR, 2026; arXiv:2505.13430, initially 2025. https://arxiv.org/abs/2505.13430

[R23] Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu. **Mamba-3: Improved Sequence Modeling using State Space Principles.** ICLR, 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html ; technical HTML: https://arxiv.org/html/2603.15569v1

[R24] Zhirui Huang, Shiwei Liu, Haozhe Zhu, Qi Liu, Chixiao Chen. **DRAFT: Decoupling Backpropagation from Pre-trained Backbone for Efficient Transformer Fine-Tuning on Edge.** DAC, 2025. DOI: 10.1109/DAC63849.2025.11132748. Existence and venue checked against https://62dac.conference-program.com/ ; full technical content not retrieved in this review.

[R25] X. Jiang et al. **Open Access Dataset, Toolbox and Benchmark Processing Results of High-Density Surface Electromyogram Recordings.** IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2021. DOI: 10.1109/TNSRE.2021.3082551. Hyser official dataset: https://physionet.org/content/hd-semg/2.0.0/

[R26] Brianna M. Karpowicz, Joel Ye, et al. **Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark.** NeurIPS Datasets and Benchmarks, 2024. https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html

[R27] NinaPro project. **Database 6 (DB6), official dataset instructions.** Dataset documentation, not a new research-paper citation. https://ninapro.hevs.ch/instructions/DB6.html
