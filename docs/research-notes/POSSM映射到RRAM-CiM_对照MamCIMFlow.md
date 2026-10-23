# 对照 MamCIMFlow，把 POSSM 映射到 RRAM 存内计算

> 参考论文：
> - Li, Wang（王中锐）, Ye, Pan, Wang, *MamCIMFlow: An Integrated Co-Design of RRAM-Based CIM and Selective State-Space Streaming for Efficient Mamba Model Acceleration*，ICCD 2025（4 页短文）。
> - Ryoo 等，POSSM，NeurIPS 2025（项目文件 `⑩POSSM - Ryoo 2025.pdf`）。
>
> 标注约定同前：**【原文】** 论文写明，**【推断】** 我的推理，**【估算】** 数量级估计，需用你们自己的参数复算。
> 已有结论不再重复，需要时直接引用：
> - `POSSM_ASIC_cross-session_分析.md`（下称《ASIC 分析》）：输入交叉注意力改写成查表、算力/存储总账、APST 式无梯度适配。
> - `SSM硬件算子与存内计算_前沿梳理.md`（下称《前沿梳理》）：Wei Lu 组 WOₓ 衰减态、IMSSA、HPD 等。

---

## 0. 一页结论

1. **MamCIMFlow 的方法可以概括成一句话：静态投影进 RRAM，动态递推进专用数字流水线，两者之间不留缓冲。** 具体是三条原则：按下游消费顺序组织 RRAM 权重；RRAM 出口自带 ADC 后的激活函数，直接产出"可用"的向量；用握手让各组按时供数，把中间 SRAM 删掉（SRAM 减 66%，延迟降 62.9%）【原文 §III, §V, §VII】。
2. **这个"静态 / 动态"二分法可以直接搬到 POSSM，但 POSSM 需要第三类：每个 session 改写一次的"半静态"参数**（unit embedding 生成的 K/V 表、打分表、session query）。MamCIMFlow 完全没有考虑权重更新。我的建议是：冻结主干放 RRAM，半静态表放 SRAM（或 SRAM-CIM），运行状态放寄存器【推断】。
3. **优化目标要翻过来。** MamCIMFlow 追的是吞吐（tokens/s，GHz 级、1.5 GSps 10 bit ADC、1024 行阵列），而 POSSM 每秒只有 20 个 token（50 ms chunk）。BCI 里 RRAM 真正的价值不是破"内存墙"，而是**非易失、零泄漏地存下几百 KB 权重**，让整个计算阵列在 chunk 之间断电。MamCIMFlow 消融里最重要的一条反而是负面结论：并行度从 8× 到 64×，面积涨了 144×，主要是 ADC【原文 §VII】。所以 POSSM 应该选最低并行度、低精度、多列共享的 ADC【推断】。
4. **主干选 S4D 最合适。** S4D 是 LTI，没有 MamCIMFlow 里的 Dt/B/C 三组投影，状态递推是对角逐元素运算，放在数字端，RRAM 只做混合层，模拟噪声不会沿着递推反复累积。GRU 的 `U·h` 要每步过一次模拟阵列，噪声会在时间上累积；POSSM-Mamba 用的是 Mamba-1，比 MamCIMFlow 针对的 Mamba-2 更难做【推断】。
5. **输入端有一个 MamCIMFlow 里没有、但很适合 crossbar 的结构**：一个 chunk 里所有 spike 的注意力加权和，可以写成"每通道一个标量权重的向量 × V 表矩阵"，也就是一次 (通道数 × 64) 的 MVM（§3.1）。通道数少时用《ASIC 分析》的数字查表即可，通道数上千时这个改写的优势才出来【推断】。
6. **MamCIMFlow 没有报告任何精度结果**，只有吞吐、能效和面积。BCI 方向的审稿人一定会问"器件非理想性下 R² 掉多少"、"跨 session 后掉多少"。这既是它的缺口，也是你们的切入点（§6）。

---

## 1. MamCIMFlow 到底做了什么

### 1.1 问题设定【原文 §I–II】
Mamba-2 的计算分两类：
- **大的静态投影**（X/Z/B/C/Δ 投影、输出投影）：访存受限，适合 RRAM-CIM。
- **动态、依赖输入的状态更新**：`h_t = A_t ⊙ h_{t-1} + B_t ⊙ X_t`，`y_t = C_t h_t`，`A_t = exp(Δ_t ⊙ A)`。逐元素运算多、控制复杂，CIM 阵列做不好。

他们的判断是：只优化其中一侧，瓶颈会转移到两者的接口上。

### 1.2 架构（Fig. 1）

**RRAM 侧：6 个权重组【原文 §V】**

| 组 | 做什么 | 出口处理 | 送给谁 |
|---|---|---|---|
| X | 投影 `U·W_X` → 卷积 PE（ConvX），两级直连 | SiLU | IEPE |
| B | 投影 → 卷积，两级直连 | SiLU | IEPE |
| C | 投影 → 卷积 | SiLU | OCPE |
| Dt | 投影 + Softplus → Δ_t | Softplus | HSPE（算 A_t） |
| Z | 投影 | — | 门控 |
| Out | 输出投影 | — | 下一层 |

每组都在 ADC 后面集成激活函数和数据格式转换，直接输出"SSS 可用"的向量，不经过单独的数字后处理和缓冲。

**数字侧：State Space Streaming（SSS）单元，三级流水【原文 §IV】**

| PE | 计算 | 关键技巧 |
|---|---|---|
| IEPE | `B_t ⊙ X_t` | B_t 驻留在本地 SRAM，X_t 分块流过 N 个乘法器；利用 Mamba-2 多个 head 共享一个 B 的特点 |
| HSPE | `h_t = A_t ⊙ h_{t-1} + B_t X_t` | **前瞻**：算出 h_t 后马上预算 `A_{t+1} ⊙ h_t`，下一个 token 来时只剩一次加法 |
| OCPE | `y_t = C_t · h_t` | C_t 驻留 SRAM，h_t 流过 N 个 MAC + 加法树；多个 head 复用 C_t |

### 1.3 配置和结果【原文 Table I, §VI–VII】
- RRAM：每个 PE 1024 行，8 个 cell 表示 1 个 8 bit 权重（按位切片），读延迟 3.16 ns；ADC 10 bit、1.5 GSps。
- 数字：1 GHz，并行度 = D_inner；TSMC 28 nm 综合；RRAM 用 MNSIM 2.0 仿真。
- 对比 RTX 4090：吞吐最高 14.1×，能效最高 319.3×；对比"存了再取"的 CIM 基线：吞吐 2.29×，能效 3.43×。
- 消融：SSS 让 SRAM 减 66%，token 延迟降 62.9%；并行度 8×→64× 时吞吐近线性、功耗只涨 3.8×，但**面积涨 144×，主要是 ADC**；作者推荐 16×–32×。

### 1.4 局限（从 BCI 角度看）【推断】
- 全文是仿真，没有流片、没有器件数据、**没有精度评估**，也没有考虑 RRAM 电导漂移、读噪声、IR drop。
- 只考虑推理，权重写一次永不改。
- 面向 LLM 规模（D_inner 上千）的高吞吐场景，参数和 ADC 选择对植入端都过大。
- 通讯作者之一是王中锐老师，和你们组（APST 合作方）是同一个组。**他们的仿真框架和 RRAM 模型你们大概率能直接拿到**，这比从头搭 MNSIM/NeuroSim 流程省很多事。

---

## 2. 把 MamCIMFlow 的"划分规则"套到 POSSM 上

### 2.1 三类参数，而不是两类

| 类别 | POSSM 中的内容 | 更新频率 | 介质【推断】 | MamCIMFlow 对应 |
|---|---|---|---|---|
| 静态 | 编码器 W_k/W_v/W_o、FFN、主干（S4D 混合层 / GRU 的 W、U）、输出注意力投影、读出 | 出厂一次 | **RRAM** | 6 个 RRAM 组 |
| **半静态（每 session）** | unit embedding → K/V 表、打分表；session embedding → 输出 query | 每天或每次漂移 | **SRAM**（或 SRAM-CIM） | 无 |
| 动态 | 在线 softmax 累加器、h(t)、最近 k=3 个隐状态 | 每 chunk | 寄存器 / 小 SRAM | SSS 内部缓冲 |

半静态表理论上也可以写进 RRAM：每天写一次，10 年约 3,650 次，远低于 RRAM 常见的 10⁶ 级耐久度。但每次要做 write-verify，写后还有电导弛豫，表的精度不好保证。放 SRAM 更稳妥，几十 KB 的 SRAM 代价也不大【推断】。这和 MamCIMFlow 引用的 RRAM-SRAM 混合 CIM（Wen 等 Science 2024）是同一个思路，只是划分依据从"是否随 token 变化"变成了"是否随 session 变化"。

### 2.2 按算子逐个决定放哪里

MamCIMFlow 隐含的规则是"大 MVM 进 RRAM，逐元素和非线性进数字"。在 POSSM 的规模下需要再加一条：**矩阵太小或输出太窄时，ADC 开销摊不平，直接用数字更划算**【推断】。

以 POSSM-SS（输入 64，隐状态 256，1 层）为例：

| POSSM 算子 | 形状（SS 规模） | 运行频率 | 放哪里【推断】 | 理由 |
|---|---|---|---|---|
| 输入注意力 K/V/打分 | 已折叠成查表（《ASIC 分析》§2.2） | 每个 spike | 数字事件引擎 + SRAM 表 | 运行时没有 MVM |
| 生成 K/V 表 `LN(e_u)·W_k`、`·W_v` | 每 unit 64×64 ×2 | **仅校准时** | RRAM（"表生成组"） | 192 个 unit 批量 MVM，正好是 RRAM 擅长的；运行时阵列断电 |
| 编码器 W_o + FFN | 64×64、64→256→64 | 每 chunk | RRAM | 中等大小，静态 |
| S4D 混合层（GLU / 线性） | 约 H×2H | 每 chunk | RRAM | 静态大 MVM |
| S4D 对角递推 | H × N/2 个复数乘 | 每 chunk | 数字 | 逐元素，Ā 静态可预存 |
| S4D 的 C 读出 | 块对角 | 每 chunk | 数字 | 块对角放 crossbar 浪费阵列 |
| GRU 的 `W·x + U·h` | 64×768 + 256×768 | 每 chunk | RRAM（但要注意噪声累积，见 §4） | 稠密 MVM |
| 输出注意力 K/V 投影 | 256 维 → ？（论文未给出头维度） | 每 chunk 只投影最新的 h | RRAM | 只需对新 h 做一次，旧的 2 个缓存在寄存器 |
| 输出 query | session emb + 行为时间戳 | 每 session 预算 | SRAM | 半静态 |
| 读出 | 256→2 | 每个输出点 | 数字 | 只有 2 列，ADC 不划算 |

### 2.3 对应 MamCIMFlow 的结构图【推断】

```
                ┌──────────── RRAM（冻结，chunk 之间断电）────────────┐
 校准时 ──────► │ [表生成组] W_k, W_v                                    │──► 写 SRAM 表
                │ [编码组]   W_o, FFN                                    │
                │ [主干组]   S4D 混合层（或 GRU 的 [W;U]）               │
                │ [输出组]   输出注意力 K/V 投影                         │
                └────────────────────────────────────────────────────────┘
                     ▲ 低精度 DAC          │ 共享 SAR ADC + 出口激活（GELU/σ/tanh 查表）
                     │                     ▼  就地产出，不进缓冲
 spike 事件 ─► [事件累加引擎 EAE] ─► z(t) ─► [递推单元 RU] ─► h(t) ─► [输出注意力单元 OAU] ─► 速度
              查 SRAM 打分表/exp/累加        S4D：Ā⊙h 前瞻          3 元 softmax + 读出
```

三个数字单元和 SSS 的对应关系：
- **EAE ↔ IEPE**：IEPE 把 B 驻留、让 X 流过；EAE 把打分表和 V 表驻留在 SRAM，让 spike 事件流过。两者都是"小的参数驻留，数据流过"。
- **RU ↔ HSPE**：HSPE 的前瞻 `A_{t+1}⊙h_t` 在 S4D 里更简单，因为 Ā 是常数。chunk 一结束就可以预算 `Ā⊙h_t`，下个 chunk 结束时只剩加上 `B̄·u_{t+1}`。
- **OAU ↔ OCPE**：OCPE 让 C 驻留、h 流过；OAU 让每个 session 的 query 驻留、最新的 h 流过。

---

## 3. 输入端：MamCIMFlow 没有、POSSM 独有的部分

### 3.1 把一个 chunk 的注意力写成一次 MVM【推断】
单 query 交叉注意力在一个 chunk 内的输出是

`z ∝ Σ_j exp(s(i_j, Δt_j)) · v_{i_j}`

其中 i_j 是第 j 个 spike 的通道，v_i 是该通道的 value 向量。把同一通道的 spike 合并：

`z ∝ Σ_i a_i · v_i = Vᵀ a`，其中 `a_i = Σ_{j: i_j = i} exp(s(i, Δt_j))`

所以运行时只需要：
1. 每个通道一个标量累加器 a_i（spike 来时查打分表、查 exp 表、累加）。
2. chunk 结束时做一次 `Vᵀ a`（通道数 × 64 的 MVM），再除以 Σa_i。

`Vᵀ a` 正好是 crossbar 的原生运算：a 作为字线输入，V 表存在阵列里。几点说明：
- **成立前提**：value 上不加 RoPE。POYO/POSSM 的实现里 value 是否旋转，《ASIC 分析》§6 已经列为待查项，这里同样需要先确认。如果 value 也旋转，就要按 Δt 分组，阵列输入变成 (通道 × 时间桶)，规模会乘以时间桶数。
- **不需要在线求最大值**：打分表是离线算好的，最大值已知，可以直接减一个固定偏移，省掉 FlashAttention 式的 running max。
- **V 表是半静态的**。如果放 RRAM，每个 session 要重写 通道数 × 64 个权重；如果用 SRAM-CIM，就没有写入代价。
- **什么时候值得做**：192 通道时，数字逐 spike 累加每个 chunk 只有约 190×64 ≈ 12 K 次乘加，数字实现已经很便宜。到 1,000 通道以上（Neuralink 式高通道数），数字累加随 spike 数线性增长，而 crossbar 每个 chunk 只做一次 MVM，这个改写的优势才明显。

### 3.2 前端接口
MamCIMFlow 的输入是已经 embed 好的 token，不涉及前端。POSSM 要求 spike 时间戳（R² 0.835 vs bin 级 0.557【POSSM 原文 Table 9】），前端必须输出（通道 ID，时间戳）事件，这一点《ASIC 分析》§2.1 已经讨论过，映射到 CIM 不改变这个结论。

---

## 4. 主干选择：从 CIM 角度再看一遍

| | S4D | GRU | Mamba（POSSM 用的是 Mamba-1） |
|---|---|---|---|
| 需要的 RRAM 组 | 只有混合层 | [W;U] 堆叠成一个阵列，输入 [x; h]（64+256 = 320 行） | in_proj（X、Z）、x_proj（B、C、Δ）、dt_proj、out_proj |
| 数字侧 | 复数对角递推，Ā 常数 | σ / tanh 查表、逐元素门控 | 完整的 SSS：exp、softplus、卷积、门控 |
| 递推经过模拟阵列吗 | **不经过**，h 只在数字端递推 | **经过**，每步 h 都要 DAC→阵列→ADC | 部分经过（B、C、Δ 由输入生成，h 在数字端） |
| 时间依赖的流水线 | 前瞻 `Ā⊙h_t` 完全可行 | 候选态需要先算 r 再算 `U_h(r⊙h)`，每步两相 | Mamba-1 的 B/C/Δ 由卷积后的 X 生成（x_proj），比 Mamba-2 的并行投影多一级依赖 |
| 和 MamCIMFlow 的关系 | 是 MamCIMFlow 的大幅简化版：没有 Dt、B、C 组 | 不适用 SSS，更像传统 RNN-CIM | 最接近，但 Mamba-1 的 A 是 D×N 矩阵，每 token 要算 D×N 个 exp，而 Mamba-2 每个 head 只要 1 个 |

**推荐 S4D**【推断】，理由：
1. RRAM 组最少，数字侧最简单。
2. 模拟误差只在每个 chunk 的前向路径里出现一次，不会沿着递推被反复放大。GRU 每步都要把 h 经过 DAC/ADC 送进阵列，量化误差和读噪声会在几百个 chunk 上累积，连续运行几小时的稳定性风险更高（《ASIC 分析》§6 已经提到长时间递推的稳定性问题，CIM 会让它更突出）。
3. POSSM 原文里 S4D 参数最少（0.41 M），FT 跨动物也最好。

如果要更进一步，可以借鉴 Wei Lu 组的做法，把 S4D 改成实数对角、按 block 共享衰减，用器件物理实现衰减（见《前沿梳理》§4.1）。这是比 MamCIMFlow 更激进的路线，可以作为第二阶段。

---

## 5. 尺寸和外围电路：把 MamCIMFlow 的参数"缩下来"

### 5.1 要改的参数【推断】

| 参数 | MamCIMFlow | POSSM / 植入端建议 | 理由 |
|---|---|---|---|
| 阵列行数 | 1024 | 64–256（对齐 POSSM 的 64/256/320 输入维） | 1024 行在 POSSM 上利用率很低 |
| 权重表示 | 8 cell/权重（1 bit/cell 位切片） | INT4–INT8，每 cell 2 bit 多值 + 差分对 | 减少 cell 数和 ADC 转换次数 |
| ADC | 10 bit、1.5 GSps，每列一个 | 6–8 bit SAR，8–32 列共享一个 | 时间预算是 50 ms 一步，不需要速度；ADC 是面积大头 |
| 数字时钟 | 1 GHz | 1–10 MHz 级，或者事件驱动 + 时钟门控 | 每 chunk 只有约 0.3 M MAC |
| 并行度 | 8×–64×，推荐 16×–32× | 取能满足"chunk 结束后几十 µs 内出结果"的最小值 | 吞吐不是目标 |

### 5.2 粗略账【估算】
- POSSM-GRU-SS 约 0.47 M 权重。若 INT4、每 cell 2 bit、差分对，则每个权重 4 个 cell，共约 1.9 M cell，大约 29 个 256×256 阵列。S4D-SS（0.41 M）略少。
- 每 chunk 的 RRAM 活跃时间：几个 MVM，每个几十到几百 ns（含共享 ADC 的串行转换，可能到 µs 级）。占空比在 10⁻⁴ 量级，阵列和 ADC 在其余时间可以完全断电。
- 对比：同样的权重放 SRAM 要约 235 KB（INT4），泄漏功耗在 28 nm 下大约是几十 µW 量级（强依赖工艺和阈值电压，这个数字需要用你们的工艺库确认）。MiBMI 整片预算是 883 µW，所以**省掉这部分泄漏，是 RRAM 在 BCI 里最实在的收益**，比 MamCIMFlow 强调的吞吐收益更重要。

---

## 6. 跨 session 适配：MamCIMFlow 完全没覆盖的部分

1. **为什么只能做无梯度适配**：FT 要改写全部权重。RRAM 每次写要 write-verify，速度慢、能耗高、精度有限，而且只读的工作负载正是 RRAM 的舒适区。所以 CIM 架构反过来强化了《ASIC 分析》§3.3 的结论：主干冻结，只用 APST 式闭式方法重新生成半静态表。
2. **RRAM 在校准时也有用**：APST 式校准里"身份网络前向"和"表生成"都是冻结权重的批量 MVM，可以复用同一套 RRAM 阵列。校准时阵列打开，平时断电。
3. **顺便补偿芯片自身的漂移**：RRAM 电导会随时间弛豫和漂移。可以给每个阵列列存一组数字端的增益/偏置校正（几 KB），在每次 session 校准时用同一批校准 trial 一起重估。这是《前沿梳理》§8.3"一次校准同时补偿两种扰动"的一个具体、便宜的实现方式【推断，需要实验验证】。
4. **评价时要报告的新指标**：每次适配写入多少字节、写入的是 SRAM 还是 RRAM、校准要多少 MAC 和多少 trial。

---

## 7. 建议的研究路线

**第一步：在软件里做"CIM 感知"的 POSSM**
- 复现 POSSM-S4D-SS（代码现状见项目记忆：官方代码未公开，基于 torch_brain POYO 组件搭）。
- 按 §2.2 的划分做量化（RRAM 层 INT4/INT8 + 共享 ADC 位宽，数字层定点），加入 RRAM 噪声模型（读噪声、电导漂移、器件间差异）。如果王老师组有现成的器件模型和噪声感知训练代码，直接用。
- 在 FALCON M1/M2/H1 上报告：理想精度、加噪精度、跨 session（APST 式适配）后的精度。**这一组精度结果正是 MamCIMFlow 缺的。**

**第二步：架构仿真**
- 复用 MamCIMFlow 的方法学：RRAM PE 用 MNSIM 2.0（或王组的框架），数字单元（EAE、RU、OAU、校准引擎）用 Design Compiler 综合。
- 基线：① 纯数字 ASIC（SRAM 存权重，MiBMI 式）；② "存了再取"的 RRAM-CIM（MamCIMFlow 的基线）；③ 你们的流式 + 半静态表方案。
- 指标：每次解码能耗、chunk 结束后的尾延迟、面积、待机泄漏、每次适配的写入量和能耗。

**第三步（可选）：器件级**
- S4D 衰减用器件物理实现（Wei Lu 路线），或者在 RRAM 测试芯片上跑冻结主干。

**可能的论文定位**【推断】
- 和 MamCIMFlow 的差异要讲清楚：它是"高吞吐 Mamba-2 推理"，你们是"事件驱动、低占空比、可跨 session 适配的神经解码"。三个新东西：事件驱动输入注意力（§3.1）、静态/半静态/动态三级存储划分（§2.1）、在 CIM 上的无梯度跨 session 适配（§6）。
- 投稿方向：结合《投稿会议日历》，DAC/ICCAD/TBioCAS 类的软硬件协同方向比较合适。

---

## 8. 需要先确认的事
- POSSM 的 value 是否加 RoPE（决定 §3.1 的 crossbar 改写是否严格成立）。
- POSSM 输出注意力的头数和维度（论文 Table 5 没给）。
- 王老师组的 RRAM 工艺参数：多值 cell 位数、读噪声、漂移速率、write-verify 代价，以及能否共享 MamCIMFlow 的仿真流程。
- S4D 在 POSSM 里的确切结构（状态维度、是否用 GLU 混合），以便算准 RRAM 阵列数。
