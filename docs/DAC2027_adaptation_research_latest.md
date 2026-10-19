# DAC 2027：先验约束的流式校准与递归 CiM 软硬件协同研究建议

**我建议把主线从“更强的统计量编码器”改成：P0 提供先验，监督统计量负责显式纠正先验，而不是继续让 MLP 猜整张表。**

具体而言，优先验证 **“以 P0 为中心、在低维任务空间求解的监督对齐”**；把 **“F7-J 的流式 eligibility-trace 实现”**作为第二条候选。前者更可能以很低成本连接小 B 和大 B，后者更直接利用冻结 trunk 的实际预测误差。

同时，需要撤回三个过强的判断：

- 现在没有证据支持继续把 **set-attention 作为主要贡献**。
- “梯度方法优于 calibrator”还不能证明 **47 维描述量已经保留了全部有用信息**。
- **小 B 用 P0、大 B 用正则化 CCA** 已经是非常强的竞争方案。新方法必须超过这个经过认真优化的基线，不能只强调比 F7-J 便宜。

以下实验判断依据你们提供的结果；新增方法是我推导的候选，尚未在你们的数据上验证。

---

# 一、你们现在的证据支持什么，不支持什么？

| 当前证据 | 我认为可以支持的结论 | 还不能支持的结论 |
|---|---|---|
| P0 在 days 接近当前最佳静态表结果，month/year 仍有差距 | 需要针对剩余误差做校正，而不是统一增加 encoder 容量 | 当前约 0.75 是严格的静态表性能上限 |
| ridge-CCA 在 B=33 与 R3 打平 | 大 B 时，显式监督对齐可能已经足够，不必反复运行 trunk | CCA 与 F7-J 在数学上完全等价 |
| 同样 B 个 trial，梯度拟合优于 calibrator | 原始校准序列中存在尚未利用的信息 | 这些信息全部存在于当前 47 维描述量中 |
| P0 已含 prior latent correlation，额外 pooling 没有收益 | 当前描述量已经带有群体耦合，额外 attention 的边际价值很小 | 群体信息不重要，或者所有动态 attention 都无效 |
| sub-C 的 R2 增益稳定，readout 没有额外贡献 | 对这些 session，主要剩余瓶颈更接近输入侧 | 所有数据集都应只更新 read-in |
| M2、Hyser 与 DB6 的表现明显不同 | 适配瓶颈具有任务和观测结构依赖性 | 一种固定 calibrator 能普遍优于闭式方法 |

**最重要的证据缺口是第三行。**梯度方法使用原始时间序列以及 trunk 的时序敏感性；P0 使用压缩后的统计量。二者差距可能来自“映射不好”，也可能来自“时间信息在压缩时丢失”。这需要一个针对性诊断，而不是继续扩大 MLP。

另外，你们的 `ui_reg` 应称为**梯度参照**，不应称为“上限”。B=8 时 R3 高于它，并不矛盾，可能反映了不同的正则化和优化行为。

---

# A. 更好的算法方向

## A1. 首选：P0 先验 + 显式监督残差对齐

我建议分三步推进，先做最便宜、最可能推翻复杂方案的对照。

### 第一步：先建立一个必须击败的简单基线

设 $W_0$ 是 P0 的表，$W_{\mathrm{CCA}}$ 是你们现有正则化 ridge-CCA 生成的表，先测试：

$$
W_{\alpha}
=
W_0+\alpha_B\left(W_{\mathrm{CCA}}-W_0\right).
$$

其中 $\alpha_B$ 只在 source validation 上选择；CCA 条件不足时允许 $\alpha_B=0$。两张表必须处于**相同的归一化约定和 trunk 输入坐标系**。

这是一个很强的对照，因为它已经具有你们需要的形式：

> 小 B 保留 P0，大 B 逐渐相信监督对齐；最终仍是一张静态表。

**不要把这种加权本身当成创新。**如果它已经达到所需精度和成本，新方法就必须解释为什么还需要更复杂的机制。

---

### 第二步：不直接混合两张表，而是求解“P0 尚未满足的监督约束”

我更推荐的候选可以暂称 **P0-Solve**。

#### 1. 在实际 trunk 输入坐标系中建立监督锚点

令：

$$
x_t\in\mathbb R^N,\qquad
u_t=W x_t,\qquad
W\in\mathbb R^{K\times N},\quad K=10.
$$

构造低维行为基函数：

$$
\phi_t=\phi(y_t)\in\mathbb R^p.
$$

**第一版直接复用成功的 ridge-CCA 中的方向标签或行为基函数**，从 $p=8$ 或 16 开始，不要同时发明新的特征网络。

在 source 上，用实际送入该冻结 trunk 的潜变量 $u_t^{\mathrm{src}}$，拟合一个固定映射：

$$
u_t^{\mathrm{src}}\approx F\phi_t,
\qquad F\in\mathbb R^{K\times p}.
$$

这里的 $F$ 是任务相关潜变量的预测器，**不是 $y\to u$ 的精确逆映射**。它必须位于该 trunk 已经接受的输入坐标系，不能直接拿任意旋转后的 CCA canonical variables 当作输入。

#### 2. Target 只累计两个低维统计量

$$
H=\sum_t x_t\phi_t^\mathsf T
\in\mathbb R^{N\times p},
$$

$$
G=\sum_t\phi_t\phi_t^\mathsf T
\in\mathbb R^{p\times p}.
$$

希望新表满足：

$$
WH\approx FG.
$$

直观地说：

> 新 session 中，输入表生成的潜变量与行为之间的关系，应接近 source 中固定 trunk 熟悉的关系。

这一步使用标签，但标签只用于校准，**推理时不需要行为输入**。

#### 3. 只校正 P0 未满足的部分

令 $D$ 为对角的“允许更新程度”，先从 $D=I$ 开始。考虑：

$$
\min_W
\left\|
(WH-FG)G_\epsilon^{-1/2}
\right\|_F^2
+
\lambda
\left\|
(W-W_0)D^{-1/2}
\right\|_F^2,
$$

其中：

$$
G_\epsilon=G+\epsilon I.
$$

这个**代理目标**有闭式解：

$$
\boxed{
W^*
=
W_0+
(FG-W_0H)
\left(H^\mathsf TDH+\lambda G_\epsilon\right)^{-1}
H^\mathsf TD
}
$$

关键是：**需要求解的是 $p\times p$，不是 $N\times N$。**

实际实现解线性方程即可，不需要显式计算逆矩阵或矩阵平方根。

### 为什么它可能比 P0 或纯 CCA 更好？

**第一，它让模型先验与监督证据在同一个目标里竞争。**  
当监督约束不可靠时，较大的 $\lambda$ 保留 P0；数据覆盖充分后，监督残差推动更新。它不会要求一个固定 MLP 在所有 B 下都准确预测最优表。

**第二，它允许所有 unit 的列共同满足任务约束。**  
$H^\mathsf TDH$ 汇总了所有 unit 的信息。这里的群体耦合是一个小矩阵求解，而不是额外的 attention。

**第三，它可以引入 P0 没有利用的输入方向。**  
不要把校正限定为一个 $10\times10$ 的左乘：

$$
W=QW_0.
$$

因为如果 $W_0v=0$，那么任何 $QW_0v$ 仍然是零。这种校正无法找回被 P0 丢掉的神经输入方向。上面的更新则可以改变 read-in 的输入子空间，更有机会利用 source 中未出现过的电极。

**第四，它适合你们的 anytime 流程。**  
$H,G$ 不依赖当前 read-in。每个 trial 结束后，即使 P0 生成了一个更好的 $W_0$，仍然可以用已有统计量重新求解，无须重读过去的 spikes。

### 它的局限必须写清

这是**任务统计关系的最小二乘代理目标**，不是冻结非线性递归 trunk 的真实解码损失的闭式最优解。

它可能遗漏：

- 同一行为条件内的神经活动差异；
- 描述量没有保留的时序信息；
- source 与 target 不一致的条件动力学；
- 单靠输入对齐无法补偿的 trunk 内部误差。

因此，不能预先保证它达到 R3 或 ridge-CCA。

---

### 第三步：基础求解有效后，再加入很小的 source 端学习

我不会先做完整 learned optimizer。更合适的是只学习：

$$
F,\quad \lambda,\quad \text{少量分组正则化参数}.
$$

例如 $K=10,p=16$ 时，$F$ 只有 160 个参数。可以在 source episodic training 中，通过求解器和冻结 trunk 反传优化这些参数；target 仍然只执行统计量累计和小矩阵求解。

这可以改善“行为统计关系如何映射到 trunk 输入坐标”，又不会增加 target 反传或推理分支。

但**通过闭式求解器做 meta-learning 已有明确先例**：Bertinetto、Henriques、Torr、Vedaldi，*Meta-learning with differentiable closed-form solvers*，ICLR 2019。因此，你们的创新应在约束、统计量组织和硬件执行，而不是“首次把 ridge 放入 meta-learning”。([ora.ox.ac.uk](https://ora.ox.ac.uk/objects/uuid%3A23adfe83-f098-4e7f-bb05-075e5878a291))

对于 $D$，应区分两种不确定性：

> **电极没见过，意味着 prior 不可靠；unit 几乎不放电，意味着 target 证据不可靠。二者不能都简单解释为“应该大幅更新”。**

先比较 $D=I$ 和少量 source 固定分组；复杂的逐 unit 置信度网络暂不优先。

---

### 一周内最关键的验证：做一个“更新方向是否已经在统计量里”的诊断

这是我认为最能避免继续走弯路的实验。

取同样 B 个 trial 得到的梯度参照：

$$
\Delta W_g=W_{\mathrm{ui\_reg}}-W_0.
$$

P0-Solve 的更新行空间受 $H^\mathsf TD$ 限制。离线把 $\Delta W_g$ 投影到这个空间，再评估：

$$
W_{\mathrm{diag}}
=
W_0+
\operatorname{Proj}_{\operatorname{row}(H^\mathsf TD)}
(\Delta W_g).
$$

这个投影只用于诊断，不部署，也不作为方法成绩。

| 结果 | 应作出的判断 |
|---|---|
| 投影后仍能保留大部分梯度增益 | 当前监督统计量包含有用的更新方向，值得优化锚点和求解目标 |
| 投影后增益基本消失 | 当前任务基函数或统计量不足，继续改善同一类映射很可能无效 |
| 投影有效，但 P0-Solve 无效 | 主要问题在 source 锚点、代理损失或正则化，而不一定在信息量 |

设计选择应在 source validation 上完成；target 的这个诊断用于解释最终结果，不再据此反复选择超参数。

**一周可行性：高。**主要复用现有 P0、ridge-CCA、梯度参照和推理代码，不需要训练新主干。

---

## A2. F7-J 与 ridge-CCA 的统一视角，以及无需重放的 F7-J

### 可以统一研究目的，不能说它们是同一个算法

| 方法 | 用什么关系约束表？ | 是否显式使用 trunk 的时间敏感性？ |
|---|---|---|
| ridge-CCA | 跨 session 的监督相关结构 | 通常没有 |
| P0-Solve | P0 未满足的任务相关矩约束 | 没有显式 Jacobian |
| F7-J | 输出残差对输入潜变量的近似敏感性 | 有，使用固定时间核 |

统一的视角是：

> **在冻结解码器前，寻找一个受到先验约束的线性输入映射。不同方法选择了不同的误差空间和度量。**

CCA 的归一化相关目标、任务矩匹配，以及输出空间的线性化最小二乘，并不完全等价。

---

### F7-J 有一个可以直接利用的流式改写

令输出维度为 $q$，固定反馈核为：

$$
\bar J_\ell\in\mathbb R^{q\times K}.
$$

在固定表附近：

$$
\delta\hat y_t
\approx
\sum_{\ell=0}^{L-1}
\bar J_\ell\,\Delta W\,x_{t-\ell}.
$$

设 $e_t=y_t-\hat y_t$。F7-J 的数据项更新方向可以写为：

$$
U=
\sum_t\sum_{\ell=0}^{L-1}
\bar J_\ell^\mathsf T e_t
x_{t-\ell}^\mathsf T.
$$

这意味着：

**不必先存完整 residual 序列、做反向时间卷积，再与输入相乘。可以在当前误差到达时，把它与过去输入的滤波结果相乘。**

第一版直接保留 $L$ 个输入的环形缓冲，就能验证这个重排。

对于**同一条前向轨迹、同一个固定反馈核、同样的边界条件**，这只是求和顺序变化，不是近似。我对这个代数关系做了小规模随机数值检查，但没有验证你们的实现。

---

### 进一步把反馈核压缩为少量递归滤波器

在 source 上拟合：

$$
\bar J_\ell
\approx
\sum_{m=1}^{M}\rho_m^\ell C_m,
\qquad |\rho_m|<1.
$$

定义：

$$
r_{m,t}
=
\rho_m r_{m,t-1}+x_t.
$$

于是更新可以流式累计：

$$
U_t\mathrel{+}=
\sum_m
(C_m^\mathsf Te_t)r_{m,t}^\mathsf T.
$$

它需要保存的是 **eligibility traces**，而不是原始 trial。

以 $N=256,K=10,q=2,M=4$ 为例：

| 项目 | 示例规模 |
|---|---:|
| FP32 traces：$M\times N$ | 4 KiB |
| FP32 更新累计：$K\times N$ | 10 KiB |
| 每 bin 的额外运算 | 约 11.3k MAC |
| trunk 前向 | 使用正在运行的实时前向，不要求重复 S 次 |

这只是所写实现的算术估算，不包含原有 trunk、P0 统计和缓冲管理。

### 这里必须保留三个边界

**第一，压缩反馈核是近似。**  
少量实指数未必能很好表达符号变化、延迟峰值或振荡核。先测试有限 FIR，再比较 $M=4/8$ 的稳定递归拟合，不要先假定可压缩。

**第二，一遍流式更新不等于原来的 S 步 F7-J。**  
多轮方法会重新计算残差，在线方法没有重新观察过去的数据。表在不同 trial 间改变后，trace 对当前参数的信用分配也成为近似，需要较保守的更新。

**第三，新的 P0 表不能追溯性地改变过去的预测。**  
过去误差是在当时的表和 state 下产生的，不能在 P0 更新后，把累计量宣称为“新 P0 上的精确梯度”。

因此，我会把它定义为一个独立方法，例如 **streaming fixed-feedback refinement**，而不是“无损移除 F7-J 重放”。

### 新意风险：这个思想有很强的既有基础

使用固定系统模型过滤输入，再由当前误差更新前端参数，与经典 **filtered-x LMS** 很接近。Kuo 与 Morgan 的 *Active Noise Control: A Tutorial Review*，*Proceedings of the IEEE*，1999，系统讨论了这类自适应处理及 DSP 实现。([nokia.com](https://www.nokia.com/bell-labs/publications-and-media/publications/active-noise-control-a-tutorial-review/))

递归网络中使用局部 eligibility traces 与学习信号结合，也有 Bellec 等的 *A solution to the learning dilemma for recurrent networks of spiking neurons*，*Nature Communications*，2020。它不是你们方法的直接等价实现，但足以说明“eligibility trace”本身不能作为新概念。([nature.com](https://www.nature.com/articles/s41467-020-17236-y))

**你们可以研究的差异，是平均时序反馈核如何压缩、如何在固定 RRAM 与少样本校准下保持效果，以及它与 P0 先验和连续提交机制的协同。**

**一周可行性：**代数重排与单遍 FIR 原型高；少量递归滤波器中等。它应是备选，不建议与完整 learned optimizer 同时展开。

---

## A3. Set-attention 应该怎样定位？

**现在应降为已充分测试过的设计选项，而不是论文核心。**

你们已经有比“没调好”更有说服力的证据：参数量匹配、较高新电极比例、不同群体结构，都没有稳定收益；P0 的描述量本身也不是独立的单通道统计。

我不建议本周继续赌“扩大训练规模后会有效”。更应保留已经有一致增益的 **masking 训练**。

至于推理时动态 attention：

- 已有 APST/POSSM 实现能直接跑时，可以增加一个**放宽硬件约束的性能参照**。
- 必须匹配因果性、校准信息和评价区间，并单独报告逐 bin 的计算与状态成本。
- 不应称为“精度上限”：模型容量、训练和优化不同，它不构成数学上限。
- 没有必要为了回答审稿人，再训练一个全新的动态 attention 系统。

你们的主问题已经是：**在一次静态 read-in 的约束下，怎样充分利用监督校准。**动态 attention 可以说明限制的代价，但不必成为主线成立的前提。

---

# B. 配套的 hardware-aware 设计

## B1. 推荐的数据通路

以 P0-Solve 为主，实时与校准两条路径并行：

| 阶段 | 运算 | 建议位置 |
|---|---|---|
| 实时输入统计 | 计数、均值、相关统计、$x_t\phi_t^\mathsf T$、$\phi_t\phi_t^\mathsf T$ | 数字累加器／SRAM |
| P0 表生成 | 固定权重 MLP | RRAM-CiM；低调用频率下也比较数字实现 |
| 监督残差构造 | $FG-W_0H$ | 小型数字 MAC，固定 $F$ 可选择性复用 CiM |
| 小矩阵构造 | $H^\mathsf TDH+\lambda G_\epsilon$ | 数字端，动态×动态 |
| 求解与表修正 | LDLᵀ／三角求解、矩阵乘加 | 数字端 |
| 连续推理 | SRAM read-in → 固定 trunk 投影 → 动态递归状态 → readout | SRAM/数字 + RRAM-CiM |
| 提交 | 归一化折叠、量化、写 inactive table、原子切换 | 数字控制与双缓冲 |

**不要把 $H^\mathsf TDH$ 映射成需要重写 RRAM 的操作。**这正是数字端应该承担的动态计算。

这里 CiM 最主要的角色仍然是持续运行的固定 trunk，以及可复用的固定 encoder。无需让每一个校准算子都使用 CiM。

---

### 1. 先解决两个容易隐藏重放需求的问题

#### 标签何时可用？

无重放成立的前提是：行为标签可以同步或在有界延迟后到达。

若只有整个 trial 结束后才能拿到逐 bin 标签，$H=\sum x_t\phi(y_t)^\mathsf T$ 仍然需要保留相应输入。固定延迟可以用短环形缓冲解决；整段延迟不能凭空消除。

#### 输入归一化不断更新时，统计量不能混用坐标

如果不同 trial 的 $x_t$ 用了不同均值和尺度，直接累计它们得到的 $H$ 不再对应任何单一输入变换。

建议累计原始坐标的：

$$
\sum r_t\phi_t^\mathsf T,\quad
\sum r_t,\quad
\sum\phi_t,\quad
\sum r_t^2,
$$

在每次提交时转换到当前归一化坐标。

最终可以折叠：

$$
u_t
=
W\operatorname{diag}(\sigma)^{-1}r_t
-
W\operatorname{diag}(\sigma)^{-1}\mu.
$$

**需要同时保存等效权重和偏置。**只折叠 scale、遗漏 mean，会改变模型。

P0 中与 prior 潜变量相关的描述量，也最好始终参照一个固定 prior；若参照正在变化的 active table，就需要重新定义统计量的含义。

---

### 2. 算术和带宽预算：给出可审计的例子

以下采用：

$$
N=256,\quad K=10,\quad p=16,
$$

并仅为举例假设 spike bin 频率为 50 Hz；这不是 Hyser 原始采样率。

| 项目 | 示例预算 |
|---|---:|
| FP32 $H$ | 16 KiB |
| FP32 $G$ | 1 KiB |
| 单张 int8 read-in | 2.5 KiB |
| 双 read-in buffer | 5 KiB |
| FP32 候选表 | 10 KiB |
| 每 bin 更新 $H$ | 4096 MAC |
| 每 bin 更新完整 $G$ | 256 MAC |
| 每次提交的小矩阵构造、求解和表修正 | 约 $0.1$–$0.2$ M MAC |

如果 $H$ 每个 bin 从 SRAM 完整读改写：

$$
2\times256\times16\times4\times50
=
1.5625\ \mathrm{MiB/s}.
$$

其余几个接口的示例数据率为：

| 接口 | 条件 | 数据率 |
|---|---|---:|
| 输入 bin | 每通道 8 bit | 12.5 KiB/s |
| read-in 权重读取 | 每 bin 完整读一张表 | 125 KiB/s |
| read-in 输出 | 10 维、16 bit | 约 1 kB/s |

这些数字**不包含 trunk 内部通信**。若 SSM 持续状态共 $M_h$ bytes，每步完整读写的基础流量为：

$$
2M_h f_{\mathrm{bin}}.
$$

应按实际代码中的 state、历史输入、相位及中间缓存逐项统计，不能只使用一个简化 S4D state 公式。

另外，P0 的 2.4 M MAC 若是**每次提交**的成本，那么 B=33、每 trial 提交一次时，累计就是约 79.2 M MAC，而不是只计最后一次。

---

### 3. 128 KB 的关键是峰值同时驻留，不是各模块单独能放下

连续推理期间，求解器读取统计量快照，而新 trial 仍需继续更新统计量。一个保守组织是：

| 同时驻留项目 | 示例规模 |
|---|---:|
| 双 $H$ 缓冲／快照 | 32 KiB |
| 双 $G$ | 2 KiB |
| 双 int8 read-in | 5 KiB |
| FP32 候选表 | 10 KiB |
| 求解工作区 | 约 2 KiB |
| 你们的 lr16 readout 工作状态 | 约 1.4 KiB |
| I/O 与控制余量 | 暂留 4 KiB |
| **小计，不含 P0 与 SSM** | **约 56.4 KiB** |

再加：

$$
M_{\mathrm{P0}}+M_{\mathrm{SSM}}+\text{其他不能复用的缓存}.
$$

这里你们的 **P0≈7 KiB** 需要注明适用的 N、统计量布局和复用方式。例如，若真的同时保存 $256\times47$ 个独立 16-bit 统计量，仅这一数组就需要 23.5 KiB；如果 47 个特征大量由更少统计量派生，7 KiB 可能成立，但必须给出布局。

同样，F7-J 截断后的 114.5 KB 缓冲，不能单独作为满足 128 KB 的证据——还要同时容纳实时 state、active table 和工作区。

**不要把你们完整 $N\times N$ Gram 实现的 397 KB，直接与新方法单个 $H$ 数组的 16 KiB 比较。**主表必须比较完整峰值；也应给 ridge-CCA 一个同样认真优化过的低秩／分块实现。

---

### 4. Anytime 切换必须在实际完成时生效

我建议评测严格采用：

> trial 结束 → 获取统计量快照 → 后台于当前响应内模拟求解时延 → 新表写好 → 在真实可用时间点提交；其间旧表继续推理，state 不重置。

这里“后台”指硬件并行调度，不是离线提前获得候选表。

软件评测不能在 trial 边界零延迟切换一个现实中尚未算完的表。主结果除了 query R²，还应报告提交延迟，以及提交后短时间预测误差。

若需要控制切换冲击，可以在 source 上选择一个表更新幅度限制，最终仍折叠成单张表；不要直接声称 double buffer 已经解决了 state 一致性问题。

---

## B2. 精度、融合和特殊算子

### 已有证据可以支持什么？

| 模块 | 当前证据可以支持 | 还需要补什么 |
|---|---|---|
| P0 权重 | 已测配置中 int8/int4 基本无损 | 与新求解器联动后的完整结果 |
| 最终表 | 已测 int8 基本无损 | 连续提交、scale/bias 一起量化 |
| 原有 spike 统计 | 已测 16-bit 累加、17-bit 计数可用 | 新增 $H,G$ 的符号、动态范围和累计长度 |
| sEMG 统计 | 连续值累计需要更宽精度 | 不能套用 spike 的 16-bit 结论 |
| trunk 分组量化 | 已测 128×2 配置约损失 0.01 | 其他 backbone、连续 state、器件误差组合 |
| softmax LUT | 你们已验证的 attention 中可用 | 新主方案不需要 softmax，无须继续作为贡献 |

**新求解器第一版用 FP32 或足够宽的定点实现建立参照，不要立即全压到 16 bit。**矩阵求解会放大某些统计误差；先检查残差、条件数、对角加载和失败率，再降低精度。

建议保留高精度候选表或累计更新，只在提交时转 int8。否则小更新可能反复落在量化死区，尤其影响流式 F7。

### 哪些融合最有价值？

**归一化与表融合：**将 scale、mean 合并到 SRAM 表和偏置，避免逐通道重复除法。

**统计与校准复用：**P0 已有的行为互协方差可能覆盖 $H$ 的一部分，应按具体公式复用，避免重复累计。

**外积引擎复用：**$x\phi^\mathsf T$、readout ridge、流式 F7 和部分状态更新可以共享数字 MAC 与累加通路；需要报告实际 bank 冲突，不能只画一个“共享 MAC”方框。

### 特殊算子的处理顺序

| 算子 | 建议 |
|---|---|
| 求解中的 sqrt | 使用 LDLᵀ 可避免显式开方；实际精度仍需验证 |
| 除法／reciprocal | 小 LUT 加迭代，或共享数字单元；提交频率低，不必极端压缩 |
| descriptor 的 log、rsqrt | 尽可能放在 trial 提交时，而不是每 bin 计算 |
| trunk 的 exp、softplus、rsqrt | 查实际调用频率，再做 LUT/PWL；以连续序列误差验证 |
| 其他状态相关非线性 | 依实际 Mamba-3 风格实现处理，不能因为算子贵就直接省略 |

**P0-Solve 本身没有使用每颗实际芯片的输出误差，因此不能自动声称它补偿了芯片特异的 RRAM 非理想性。**Source 噪声训练提供的是鲁棒性；只有 R1 或流式 F7 这类使用实际芯片响应的反馈，才有机会针对具体芯片做补偿。

最后，F7-J 中“90% MAC 可放入 RRAM”不等于“90% 能量便宜”。ADC/DAC、位切片、阵列利用率、唤醒与空闲能耗，都应进入成本模型。

---

# C. 面向 DAC 的创新点定位

## C1. 推荐题目与三个贡献点

### 题目方向

首选：

**Replay-Free Calibration of Frozen Recurrent CiM Decoders via Prior-Anchored Moment Alignment**

或者更少限定算法形式：

**Prior-Anchored Streaming Adaptation for Write-Frozen CiM Neural Decoders**

暂时不要把 set-attention、LoRA、全新 SSM 或“统一所有生物信号”写进标题。

### 三个贡献点的候选写法

以下是**实验成立后**可以采用的贡献结构。

**贡献一：先验约束的监督校准。**

> 提出一种将电极对齐先验与低维监督统计量结合的 read-in 校准方法，通过显式求解先验尚未满足的任务约束，在不反传冻结主干的条件下生成静态输入映射。

这里要证明的不只是平均提升，还包括小 B 的先验保留、长间隔／新电极条件下的改进，以及相对 P0＋收缩 CCA 的优势。

**贡献二：不保留校准序列的连续执行架构。**

> 将校准组织为 $O(Np+p^2)$ 的流式统计量与小矩阵求解，联合设计高精度校准、低精度静态表和不中断递归推理的提交机制，消除整段 trial 重放与全通道 Gram 的驻留需求。

必须用完整峰值 SRAM、累计校准开销和实际提交延迟支撑，而不是只给渐近复杂度。

**贡献三：固定 RRAM 主干下的系统验证。**

> 在多 backbone、跨 session intracortical 解码及边界任务中，验证冻结低精度 CiM 主干与可写 SRAM 边界的精度—存储—能量折中，并界定哪些观测漂移适合输入校准、哪些由简单 readout 或归一化已能处理。

**这里的价值是界定适用范围，不是强行让 Hyser、M2、DB6 都成为正面故事。**

---

## 与相关工作的准确差异

| 工作：作者、标题、发表信息 | 已经覆盖什么；你们应怎样区分 |
|---|---|
| **Degenhart 等，*Stabilization of a brain–computer interface via the alignment of low-dimensional spaces of neural activity*，Nature Biomedical Engineering，2020** | 已有低维空间对齐和无监督稳定化。不能以“使用稳定低维空间”为新意；你们需要强调监督少样本、固定主干的执行约束，并保留电极汇总 Stabilizer 这个强基线。([nature.com](https://www.nature.com/articles/s41551-020-0542-9)) |
| **Gallego 等，*Long-term stability of cortical population dynamics underlying consistent behavior*，Nature Neuroscience，2020** | 已用对齐后的群体动力学研究跨日稳定性。你们加入正则与标签预算的 ridge-CCA 是直接竞争者；应明确区分原方法和自己的增强实现。([nature.com](https://www.nature.com/articles/s41593-019-0555-4)) |
| **Karpowicz 等，*Stabilizing brain-computer interfaces through alignment of latent dynamics*，Nature Communications，2025；NoMAD** | 已利用稳定 latent dynamics 做无监督对齐。你们使用标签，不能把同样 B 下的分数差直接解释为算法全面优越；重点是 supervised 校准与硬件限制下的性能成本。([nature.com](https://www.nature.com/articles/s41467-025-59652-y)) |
| **APST：按你们提供的实现描述比较** | 已有 behavior-association profile 与 unit 条件化。新意不能再放在“profile 表示 unit 身份”；差异应是显式监督求解及最终静态表。这里不替你们补一个未经核实的发表身份。 |
| **Ryoo 等，*Generalizable, real-time neural decoding with hybrid state-space models*，NeurIPS，2025；POSSM** | 已有 spike-token 输入、attention 编码与递归 SSM。你们不是首次将 SSM 用于流式神经解码；区别在部署后更新约束和校准的静态化。([arxiv.org](https://arxiv.org/abs/2506.05320)) |
| **Le 等，*SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding*，NeurIPS，2025** | 已有集合表示、上下文条件化、gradient-free few-shot adaptation 和通道 masking。不能宣称首次无梯度适配或首次用 masking；要比较运行时结构、标签使用和校准结果的存储形式。([papers.nips.cc](https://papers.nips.cc/paper_files/paper/2025/hash/24c8d2fe8520746f0084174b75b93ebe-Abstract-Conference.html)) |
| **Wu 等，*HaLoRA: Hardware-aware Low-Rank Adaptation for Large Language Models Based on Hybrid Compute-in-Memory Architecture*，2025，已核实 arXiv 版本** | 固定 RRAM 主干与 SRAM 适配分支已有直接先例。你们的区别是 target 不穿过 trunk 反传、只更新边界、最终不增加逐层适配分支。([arxiv.org](https://arxiv.org/abs/2502.19747)) |
| **Wang 等，*NoiseZO: RRAM Noise-Driven Zeroth-Order Optimization for Efficient Forward-Only Training*，DAC，2025** | 已有 RRAM forward-only 优化，使用写噪声作为扰动。你们不能以“前向替代反传”为唯一新意；固定不写 RRAM、流式统计／局部反馈和无重复序列评估才是具体区别。([hub.hku.hk](https://hub.hku.hk/handle/10722/359699)) |
| **DRAFT：未确认具体论文** | 仅凭这个缩写，本轮没有核实到与你们问题对应的标题与作者，因此不能负责任地比较，也不能据此声称新颖性。 |

---

## C2. 主图、主表与必须补的实验

### 我建议四张主图、两张紧凑主表

| 图表 | 应回答的问题 |
|---|---|
| **图 1：连续校准时间线＋数据通路** | 标签何时到达？统计量何时冻结？旧表何时继续运行？新表何时真正提交？ |
| **图 2：B–R² 曲线与逐 session 配对增益** | 小 B 是否保住 P0，大 B 是否达到强闭式基线；days/month/year 和 sub-C 是否一致？ |
| **图 3：精度—峰值 SRAM—校准能量 Pareto** | 相比 P0、收缩 ridge-CCA、R3、流式 F7，实际改善在哪里？ |
| **图 4：机制与边界** | 新电极比例、行为覆盖、更新方向诊断、切换瞬态；哪些条件使方法失效？ |
| **表 1：方法与信息预算** | 标签、prior、电极身份、额外前向、重放、RRAM 写入、推理增量 |
| **表 2：完整硬件预算** | 固定／动态 MAC、数组与数据类型、峰值 SRAM、带宽、提交延迟、能量估计口径 |

sEMG 用于界定边界即可。Hyser 的结论可以是“归一化与 ridge 已经足够强”；DB6 可以诚实展示剩余差距。不要为了四数据集齐全而掩盖这一点。

### 必须补的三类检查

**第一，最强同类基线。**  
至少加入 P0＋source 选权重的 ridge-CCA、P-prior＋同一求解器。否则审稿人无法区分收益来自 P0 先验、监督闭式更新，还是新架构。

**第二，统计独立性与外推。**  
优先把最强基线补到 sub-C 和更多 backbone。置信区间不能把 bins 当独立样本；session、backbone、动物之间的层级要写清。只有两只动物，不应泛化为广泛生物个体结论。

**第三，真正的连续协议。**  
尤其检查：

- R1-stream 的“后续 trial”是否仍在同一 B 个有标签 trial 预算内；若使用 query 标签，应另列在线协议。
- read-in 不断变化时，R1 累计的是不同特征映射下的数据，不能称为最终固定表对应的精确 batch ridge。
- F7 重放使用的 state 是否与 live state 分离；不能为了重放而重置或覆盖真实推理状态。
- 最终表不能追溯应用到其生成前的评价区间。

---

## 审稿人最可能提出的三个质疑

### 质疑一：“这只是 P0 加正则化 CCA。”

**这是目前最大的风险。**

应对不是换名称，而是给出：相同信息、相同 source 调参预算、相同存储优化条件下，新方法在精度或成本上的不可替代收益。若没有，就不要把求解公式包装成独立算法突破。

### 质疑二：“主要收益来自电极先验，证据局限于少数日期和主干。”

你们已经有 prior shuffle、长间隔和第二只动物的证据，这是优势。但要把最强竞争者扩展到这些设置，并分开分析电极可见性、数据质量和日期间隔。不能把日期间隔单独解释为漂移强度。

### 质疑三：“硬件收益建立在遗漏重放、快照、state 或真实提交时延上。”

需要完整内存布局、真实运算 trace，以及普通 SRAM＋数字 MAC 的强基线。RTL 可以不是论文主角，但**算术计数不能直接替代可信的能量与带宽分析**。

---

# C3. 如果只再做一周，我会只做这三件事

| 优先级 | 实验 | 为什么必须做 | 停止条件 |
|---|---|---|---|
| **1** | **P0＋收缩 CCA、P0-Solve，以及更新方向投影诊断** | 判断你们是否真的需要更强的映射，还是已有闭式方法已经解决问题 | 若收缩 CCA 在精度与成本上都不差，停止增加新 solver／encoder 复杂度 |
| **2** | **F7-J 求和重排验证 → 单遍 FIR → 小 M 递归 trace** | 利用已经有效的反馈信号，直接检验能否去掉重复前向和整段重放 | 若单遍损失大且小型 trace 无法恢复，不再追求“无损替代 R3” |
| **3** | **最强基线的跨 backbone/sub-C 扩展＋完整连续硬件审计** | 解决论文可信度，而不只是提高一个平均分 | 若峰值内存、提交延迟或标签预算不成立，立即缩小部署声明 |

第一项中，建议在 source validation 上预先确定一个低 B 的可接受退化界，以及高 B 的最低收益要求；这些是你们的工程判据，不是事后根据 target 分数挑选。

**我的最终判断：你们现在最缺的不是新的网络组件，而是把“强先验”和“逐渐可靠的监督证据”连接成一个可解释、可流式执行的更新规则。**

先验证 **P0-Solve 能否超过认真收缩后的 ridge-CCA**；同时测试 **F7-J 能否以单遍固定反馈形式保留主要增益**。这两件事中有一件成立，就有比较明确的算法—硬件协同主线。两件都不成立，也应保留已有的强结果，把论文诚实收敛到 P0／闭式校准的部署优势，而不是重新回到没有实证收益的 set-attention。
