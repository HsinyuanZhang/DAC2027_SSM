# 给 ChatGPT Pro 的第二次提问（2026-10-17）

> 用法：把分隔线以下的全部内容粘贴给 ChatGPT Pro。回答存为 `chatgpt_pro_answer2_20261017.md`，放到 `gdrive:DAC27/prompts/`。

---

## 你的角色

你是熟悉 neural decoding、compute-in-memory（CiM）硬件和低功耗 ASIC 的审稿人兼合作者。我们要投 **DAC 2027**（abstract 11-11，6 页 full paper 11-18，剩约 3.5 周）。上一次你给的建议是“population-conditioned static calibration”。过去两天我们按这个方向做了大量实验，每个结果都经过独立的代码审查和修正。下面是**当前的实现逻辑和证据**（正面与负面都有）。请据此指出：

1. 一个**可能更好**的算法方向；
2. 与之配套的 **hardware-aware 设计**；
3. 面向 DAC 的**创新点定位**。

请直接、具体，并指出我们的证据哪里不够。

## 1. 系统与约束（不变）

- **任务**：植入式 intracortical BCI 的跨 session（跨天到跨年）运动解码；sEMG 作为边界场景。有监督校准：新 session 只给 B 个有标签 trial，B ∈ {1, 2, 4, 8, 16, 33}。
- **模型**：冻结的 Mamba-3 风格 SSM trunk（存于 2T2R RRAM，部署后不写 RRAM），前端是 per-session 的 read-in table（K×N，K=10，N 为 unit 数），后端是 readout。
- **target 端允许**：冻结 trunk 的前向；流式累加；read-in 和 readout 两端的闭式求解或 local learning rule；在 source 上训练后冻结的固定权重网络的前向。**不允许**反传穿过 trunk。
- **硬件预算**：SRAM < 128 KB；校准结果是静态参数（≤ 32 KB）；推理时只多一次线性 read-in。
- **流式**：开机校准块 + anytime 切换。推理全程不停，每个 trial 结束时生成新表并用 double buffer 切换；评测是整段连续的因果解码，不在 trial 边界重置 state。
- **MZ 的贡献排序**：(1) 硬件友好的 adaptation 算法；(2) 它的硬件架构分析（算子、流式带宽、CiM MVM 的使用、特殊算子优化；RTL 不是重点）；(3) 借助 CiM 的硬件友好 SSM。

## 2. 当前的方法（按代价分级）

- **P-prior**：electrode-aligned prior（按 unit 所在电极，把 source 的表对齐到 target）+ 用 B 个 trial 做输入归一化。
- **P0**：固定权重的 per-unit encoder。每个 unit 用 B 个 trial 流式累加出一个 47 维描述向量（发放率、自相关、与 prior 潜变量的相关、与行为的滞后互协方差、方向调谐等），经过一个 2 层宽 64 的 MLP 生成该 unit 那一列的残差。encoder 在 source 上做 episodic 训练（模拟 B-trial 校准，在同一 session 的其他 trial 上计算 decode loss）。
- **R1-stream**：readout 端的 prior-centered ridge，统计量在后续 trial 中流式累加，不需要重放。
- **R2 = F7-J**：在 source 上离线算一次冻结 trunk 的**平均雅可比** J̄_ℓ，固定成时间卷积核；target 端重复 S 步：前向 → 残差 → 用 J̄ 做时间卷积得到潜变量误差 → 与输入做外积累加 → 更新表（向 P0 的表收缩）。这相当于用固定反馈近似梯度，需要重放 B 个 trial。
- **R3 = R2 + R1**。
- 超参数一律在 source 上选：对 trunk 来说是样本外的 session，并且日期早于 target。

## 3. 证据（全部经过独立审查；R² 均为 query R²，物理单位）

### 3.1 DANDI 000688 sub-M（4 个核心 target：days 2 个、month 1 个、year 1 个）

| B=8（B=1） | days | month | year |
|---|---|---|---|
| P-prior | 0.36 | 0.33 | 0.13 |
| **P0** | **0.723（0.647）** | **0.565（0.466）** | **0.256（0.198）** |
| R3 | 0.728 | 0.630 | 0.334 |
| R3，B=33 | 0.747 | 0.679 | 0.435 |

- 独立审查确认没有泄漏。提升主要来自 electrode-aligned prior：打乱 prior 的列后，B=8 时从 0.73 降到 0.59。
- days 层的 P0 已接近这个 trunk 的静态表上限（约 0.75）；从 P0 出发再做梯度下降反而变差。
- **口径一致的 P3**：在 month/year 层，R3 追回“P0 + 带正则的梯度精修”相对 P0 差距的 75% / 77%（R2 为 71% / 60%，R1 只有 22% / 23%）。

### 3.2 漂移曲线（sub-M，12 个 target，间隔 112–468 天，共用一个 trunk）

| 12 个 target 的平均 | prior | P0 | P0+R1-stream | R3 | 梯度参照 ui_reg |
|---|---|---|---|---|---|
| B=8 | 0.106 | 0.212 | 0.219 | 0.348 | 0.310 |
| B=33 | 0.105 | 0.211 | 0.246 | 0.412 | 0.451 |

- P0 的优势随间隔缩小。session 质量和间隔同样重要：0626 和 0627 只差 1 天，P0 却从 0.565 降到 0.307。
- 长间隔时有 18–50% 的 unit 落在 source 中没出现过的电极上。

### 3.3 sub-C（第二只动物，16 个 target session，7 个新 backbone 中的 3 个）

- P0 − prior：days +0.20（7/7），month +0.31（12/12）。
- R2 − P0：days +0.006，month +0.065（12/12；B=33 时 +0.093）。R1-stream 与 R3 中的 readout 部分都没有额外贡献。

### 3.4 FALCON M2（通道固定、漂移小，5 个 target）

- prior 已经很强（B=8 时 0.307）；P0 只高约 +0.03，CI 包含 0。
- 低秩 readout ridge（lr16，工作状态约 1.4 KB）在 B=33 时比 prior 高 +0.046 ± 0.027（5/5 为正）；完整 ridge 的 Gram 需要 264 KB。

### 3.5 sEMG（边界场景）

- **Hyser**（HD-sEMG，256 通道，20 名受试者，跨天）：跨天漂移以输入归一化为主。B ≥ 5 时，闭式 ridge 已经接近梯度参照（0.68/0.78 对 0.73/0.80）；B ≤ 2 时，瓶颈在统计量估计。fixed-weight calibrator 在 B ≥ 5 时不如 ridge。
- **DB6**（14 通道，每个 session 重新贴电极）：梯度参照在 B=8/16 时为 0.67/0.72，最好的可部署方法只有 0.54/0.58；calibrator 没有显著增益。

### 3.6 与已发表方法对比（sub-M，4 个 target，相同的抽样与 query）

- 忠实复现的 Stabilizer（Degenhart 2020）、Gallego 2020 的 CCA 对齐、APST、NoMAD，除下面两点外都低于 P0：
  - **按电极汇总输入的 Stabilizer 在 days 层与 P0 打平**；
  - **我们加了正则的 ridge-CCA（闭式，用 B 个 trial 的方向标签）在 B=33 时比 P0 高 +0.09 至 +0.12，与 R3 打平**（year：0.439 对 0.435）。
- B=1 时 CCA 类方法基本不可用，P0 仍有 0.47 / 0.20。

### 3.7 set-attention（我们知道它是 POSSM / SPINT / APST 的核心组件）

- 电极身份已知时：P2（DeepSets）、P3/P4（PMA）相对 P0 没有超出噪声的增益。
- 部分 unit 落在新电极上（混合实验 v1、v2，包括 pairwise cross-attention，key 用两个 unit 之间的噪声相关）：在参数量匹配的对照面前没有增益，即使把新电极 unit 的比例提高到 90% 也一样。
- 通道身份完全未知时（Hyser）：跨通道 pooling 比参数量匹配的对照高 +0.20，但绝对水平低（B=10 时 0.25，ridge 是 0.78）。
- **关键发现**：P0 的描述量本身已经含有群体耦合（与 prior 潜变量的相关）。oracle 显示新电极 unit 的列还有 +0.19 至 +0.59 的提升空间，而所有 calibrator 只捕获了 7–20%。只用 B 个 trial 做梯度拟合，B=8 时能达到 0.29–0.38，calibrator 只有 0.16–0.23。**瓶颈在“统计量 → 表格列”的映射，而不是信息量。**
- 模拟电极缺失的 masking 训练，在高缺失率下稳定提高 +0.07 至 +0.10。

### 3.8 硬件相关的证据

- calibrator 权重 int8/int4 无损；ternary PTQ 损失 0.03–0.06，source QAT 后部分恢复。
- 校准累加器按逐次累加取整模拟：spike count 输入时 16 bit 无损，计数器需要 17 bit；HD-sEMG 的连续值特征需要约 28 bit；N=256 时 N×N Gram 需要 397 KB。
- 输出表 int8 无损；exp2 LUT softmax 无损。
- trunk 的 2T2R PTQ（M1/M2 上的 128×2 模型）：FFN 和 B/C 行用 ternary、其余 8 bit，损失约 0.01。
- 代价：P0 约 2.4 M MAC、约 7 KiB SRAM。F7-J 为 3.9–7.8 G MAC，约 90% 是可以映射到 RRAM 的固定权重 MAC，但需要重放缓存：最坏情况 141–319 KB，把 trial 截断到 256 bin 后 ≤ 114.5 KB。F7-J 与真实梯度的余弦只有 0.26–0.45。

### 3.9 证据的局限

sub-M 的核心对比只有 4 个 target session（month 和 year 共用一个 trunk）；sub-C 的结果里每个间隔层只有 2 个 backbone；baseline 对比只在 sub-M 上做过。

## 4. 请回答

**A. 更好的算法方向**
- A1. 从“瓶颈在统计量 → 表格列的映射”和“ridge-CCA 在大 B 时与 R3 打平”这两个事实出发，你认为最有希望的方向是什么？请给出具体设计、为什么可能更好、以及一周内能做完的验证方案。可以考虑的方向包括：把闭式的监督对齐（CCA 或 latent-space least squares）与 P0 统一起来；更好的 amortized mapping（例如 unrolled 的固定步更新、learned optimizer、meta-learned preconditioner）；把 F7-J 换成无需重放的形式；在潜变量空间里做 target propagation。也可以提出别的。
- A2. F7-J 和 ridge-CCA 之间有没有统一的视角（例如都是潜变量空间里的线性化最小二乘）？能否设计一个方法：小 B 时达到 P0 的水平，大 B 时达到 R3 / ridge-CCA 的水平，并且比 F7-J 便宜得多、不需要重放缓存？
- A3. 在我们的约束下，set-attention 应该怎么定位？有没有一种在我们的场景里确实会起作用的硬件友好形式，比如需要更大的训练规模，或者用在推理时？我们是否应该补一个“推理时动态 attention 的精度上限”来回答审稿人？

**B. hardware-aware 的设计**
- B1. 按你推荐的算法，给出数据通路：哪些运算放在 CiM（RRAM MVM）上，哪些放在数字端；流式输入时各处的数据带宽；memory 瓶颈（SSM state、重放缓存、Gram）怎么解决；特殊算子（log、sqrt、除法、exp、softplus、rsqrt 等）怎么优化。
- B2. 哪些地方适合不同精度的量化、算子融合和近似计算？请指出能直接由我们已有证据支撑的部分，以及还需要补的实验。

**C. 创新点定位**
- C1. 给出题目方向和 3 个贡献点的写法，并说明相对 NoMAD、Stabilizer、Gallego CCA、APST、POSSM、SPINT、HaLoRA、NoiseZO、DRAFT 的差异。
- C2. 论文的主图和主表应该放什么？必须补的实验有哪些？审稿人最可能质疑的三点是什么，怎么提前准备？
- C3. 坦诚评估：按目前的证据，这篇论文最大的风险是什么？如果只能再做一周的实验，你会做哪三件事？

## 回答要求

- 每个建议都要写清：机制、为什么对硬件友好、如何用我们的数据验证、一周内是否可行。
- 文献只列你确定存在的，给出作者、标题、会议和年份；不确定的请标注“不确定”。
- 对新意的风险要坦诚，已经有人做过的请直接指出是谁。
