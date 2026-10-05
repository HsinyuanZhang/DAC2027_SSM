# 最新结果综合分析与探索路线修订（2026-10-05）

本文汇总 2026-10-03 至 10-04 三条工作线的结果：跨 session 迭代 A/B、unit/session 前端迭代、AP 骨架与 Track A 第一阶段（含 C2 流式实现），再结合代码审查与本次新增的两个只评估对照，修订 [探索路线图](exploration_roadmap_20261004.md)。私有官方评测已经结束；按仓库约定，分数只记在本地被 git 忽略的目录里，本文只引用它对路线修订的影响。

## 0. 结论先行

1. **主效应来自输入合同和预训练配方，适配方法本身是次要因素。**
   - M1：用旧骨架时，support z-score 之后不做监督适配（none，0.674）已经最好。换成新的预训练配方后，LoRA 达到 0.723（small）/ 0.733（wide）。
   - M2：容量影响很大，small 最好 0.399，wide 最好 0.491。
2. **在固定通道的数据上，从零构建读入表（FA / AP）打不过平均 source 表。**
   - M2 的 7 个 rotation 上，平均 source 表约为 0.35；所有用标签的闭式方法都 ≤ 0.14，连拿 query 标签拟合坐标系的 oracle 也只有 0.10。
   - 本次对照排除了“FA 均值偏移 bug”这个解释：修正后变化在 ±0.04 以内。真正的原因是坐标系假设不成立，而且解本身不稳定。
3. **state-offset 用原始坐标系确实比旋转坐标系好**（M2 上 0.333 对 0.315），在 M2 wide 上是最好的方法（0.491）。但在 M1 上它始终不如 LoRA，说明收益依赖任务。
4. **M2 需要的容量和 128 KB SRAM 约束相冲突。** wide（256×4，state 32）的递推状态在 INT24 下约 202 KB。路线必须改成“容量放在无状态的层里，状态保持小”。
5. **本地单 target 的开发分数不可靠。** 证据有三：IO 三个 seed 分别为 0.583、0.703、0.704；本地分数与私有官方评测之间有明显差距；候选的选择会因 seed 改变。因此多 target、时间前向的评测要升为最高优先级，所有方法选择都以它为准。
6. **代码层面**：最严重的路由 bug 已被修复，checkpoint 也已重训；还有 12 项问题未处理（见 §2），其中 4 项会影响 Track A 的比较，需要在第二阶段之前修掉。
7. **新增 Track E：集合注意力（set attention）读入，放在模型最前面**（§4.7）。以静态 key 或慢变 key 为主：session 内的注意力权重不变，可以折叠成 SRAM 小表，部署形态和 AP 读入完全相同。动态 key 的版本只作为精度上限。主要在 unit 数可变的 000688 上验证。依据是仓库内 APST 在 000688 sub-C 上的消融：只用活动 0.402，加入 profile 后 0.778，打乱 profile 后约为 0。

## 1. 最新结果梳理

### 1.1 跨 session 迭代 A/B（单 target：M1 20120928，M2 2020-10-28-Run1）

**A：旧骨架（256×4），修正归一化之后**，direct R²：

| 方法 | M1 | M2 |
|---|---:|---:|
| none（只做 support z-score） | **0.674** | 0.234 |
| projection LoRA | 0.651 | 0.319 |
| affine + LoRA | 0.649 | 0.324 |
| 旋转坐标 offset | 0.644 | 0.315 |
| **原始坐标 offset** | 0.642 | 0.333 |
| 逐步解冻 full | 0.641 | **0.346** |

**B：新预训练配方**：逐 session 统计、session gain/bias/embedding、跨 trial 窗口、漂移增强、更强正则；连续 128-bin 窗口。

- M1：small LoRA 0.723，wide LoRA 0.733。
- M2：small 最好 0.399（affine+LoRA），wide 最好 0.491（原始坐标 offset），wide LoRA 为 0.486。
- 新配方同时改了多项，目前无法把收益归因到某一项。

**合法校准预算下的本地探针：**
- M1 只用 10 个原始 trial：small/none 为 0.688；small IO 三个 seed 为 0.583 / 0.703 / 0.704，很不稳定。
- M2 用 33 个原始 trial：wide LoRA 为 0.482 ± 0.020，source-only 为 0.431。

**幅度收缩仍未解决**：M2 每个 trial 的第 [10, 25) 个 bin 占约 82% 的目标方差；在这一段，wide LoRA 的预测方差只有真值的 0.48，原始坐标 offset 为 0.59。

### 1.2 unit/session 前端迭代

- M1：affine 加 UI 为 0.704；加上完整方向残差（unit）后，UI 降到 0.580。
- M2：unit_readout 加 UI 为 0.421；affine 加 UI 为 0.243 ± 0.072，很不稳定。
- 结论：完整的方向残差自由度太大，在 M1 上明显退化；M2 上的最好结果也低于 wide LoRA。

### 1.3 C2 流式实现

- 已实现带状态携带的 kernel 调用，以及纯 PyTorch 的 step 参考。
- 分块携带与一次性计算的最大差约 4.4e-3 到 7.6e-3，与 BF16 kernel 本身约 4e-3 的复现噪声底相当。
- step 参考与 kernel 的差约 2e-5（核心权重缩小后测量）。
- 遗留问题：流式路径默认用 fp32 的 angle，而训练路径用 bf16，两者计算的不是同一个函数（见 §2）。

### 1.4 AP 骨架与 Track A 第一阶段（M2，7 个 leave-one-run-out rotation）

- 路由 bug 修复并重训之后，trunk 的 source 验证 R² 为 0.51–0.555，每个 source session 都在 0.43–0.67。
- Track A 各方法在 7 个 target 上的均值：

| 方法 | b=0 | b=8 | b=33 |
|---|---:|---:|---:|
| mean_table（平均 source 表） | 0.363 | 0.326 | 0.347 |
| ui_grad（梯度精修） | — | 0.034 | 0.187 |
| am2（向 source 先验收缩） | — | 0.062 | 0.140 |
| oracle_frame（诊断用） | — | 0.008 | 0.097 |
| am1（主方法） | — | −0.008 | 0.049 |

- 决策门槛 3.7 没有通过：am1 和 am3 在 7 个 target 上对 A-M0 是 0 胜。
- 000688 sub-M 的 AP 预训练：source 验证 0.611。但两个 dev target 的 run 完全相同（共用同一个 source 池和种子），而且 best_step 都等于预算上限 2000，还没收敛。Track A 尚未在 000688 上评测。

### 1.5 私有官方评测

M1、M2 各提交了一个候选，评测已结束。分数记录在本地被忽略的目录 `results/official_evalai/`，本文不公开。它对路线的影响是：本地单个次日 target 的开发分数，不能用来预测 held-out 的表现，也不能用来给候选排序（见 §4.1）。

## 2. 代码问题现状

依据 10-04 的代码审查，以及本次对当前代码的复核：

| # | 问题 | 现状 | 对已有结果的影响 | 处理 |
|---|---|---|---|---|
| 1 | `latent_batch` 在通道数相同时都用 `tables[0]` | **已修复并重训**（20:49–20:54） | 当前 checkpoint 有效 | — |
| 2 | 通道数不同时 `torch.stack` 报错 | 部分修复：单 session batch 已绕过；混合 batch 仍会报错 | 000688 预训练用单 session batch，暂不受影响 | 混合 batch 按 session 分组计算 |
| 3 | `core_step` 的 B/C bias 广播 | **已修复** | — | 补一条 batch ≠ nheads 的测试 |
| 4 | AP 预训练的 FA 初始化用到验证 trial | **已修复**（只用训练 trial） | — | — |
| 5 | A-M1 坐标系：D=2、K=16 时有 14 维由 SVD 零空间任意补全 | 未修复 | **大**：同一 seed 重跑，am1/oracle 最多相差约 0.1（本次复现） | 见 §4.2：加动力学约束，或减小 K、加入更多行为特征 |
| 6 | 坐标系在去均值的潜变量上拟合，解码时不去均值 | 未修复 | **小**：偏移约为潜变量标准差的 0.10–0.20，修正后 R² 变化在 ±0.04 以内（本次测量） | 修正，结论不变 |
| 7 | target 端的滞后特征取自拼接后的 bin | 未修复 | 中：A-M0/A-M2 的 profile 被污染，与 source profile 不可比 | 先在连续 recording 上构造特征，再按 bin 选取 |
| 8 | `decode_r2_record` 按 8192 bin 分块，状态重置 | 未修复 | **很小**：mean_table 一次整段解码与分块的差 ≤ 0.001（本次测量） | 改为一次连续解码 |
| 9 | `ui_grad` 在拼接序列上训练、在连续序列上评测；base 没有冻结 | 未修复 | 中：UI 基线可能被低估 | 在连续窗口上训练；冻结 base |
| 10 | `--target-session` 被忽略 | 未修复 | 有贴错标签的风险 | 校验两者一致 |
| 11 | 缓存非原子写入；缓存键不含 data-root | 未修复 | 并行队列可能读到半个文件 | 临时文件加 `os.replace`；缓存键加入 root |
| 12 | 队列脚本不会重跑失败的 target；不创建 `$OUT` | 未修复 | 运维问题 | 失败时清理目录，或改用 `exist_ok` 加完成标记 |
| 13 | 流式路径 angle 用 fp32，训练路径用 bf16 | 未修复 | 部署与训练不一致 | 训练和部署统一用同一种精度 |
| 14 | `initial_states` 传入的是 `device.type` 而不是 device | 未修复 | 多卡时可能出错 | 改为 `x.device` |
| 15 | 000688 plan 实际读了 final session，却记录为未访问 | 未修复 | 协议记录不准确 | 只扫描用到的 session，并如实记录 |
| 16 | 000688 预训练对两个 target 重复跑了同一个训练 | 新发现 | 浪费算力；没有收敛 | source 池相同时只训练一次；延长预算 |

另外两个小问题在我自己的脚本里（`sanity_checks.py` 的激活尺度标定、`ema_eval.py` 的注释），都不影响已报告的结论。EMA 的方差更新已经在跨 session 迭代中修正过。

本次新增的两个对照：`results/analysis/analysis_20261005/fa_centering_check.py` 及其输出 JSON。

## 3. 关键判断

1. **对 FALCON 这种固定通道的数据，跨 session 的剩余漂移主要是逐通道的 gain/offset，再加上少量低秩变化。** 平均 source 表加归一化就能吸收大部分。从零估计整张读入表，是在用少量标签和不稳定的 FA 去重建一个先验早已给出的东西。Track A 在 M2 上失败，说明的是这个场景不需要它，并不能否定 Track A。
2. **FA 坐标系失败的真正原因有两个**：第一，FA 的各维尺度由它自己按方差排序决定，与 source 潜变量之间不只差一个正交变换；source 端的白化也没有达到（对角方差 0.3–0.6）。第二，D=2 的行为标签只能确定 16 维坐标系中的 2 维，其余维度是任意的，所以结果不稳定。两者都是方法设计问题，修补实现解决不了。
3. **state-offset 有一定价值，但不是通用答案。** 原始坐标版本在 M2 上稳定优于旋转版本，在 wide 上也是最好的。但在 M1 上，所有模型都不如 LoRA。目前 CPU 部署也还不支持 offset。
4. **M2 的容量需求是当前最大的系统级矛盾。** 从 small 到 wide，M2 提升约 0.09，可 wide 的递推状态超出了 SRAM 预算。这正好说明硬件约束应该进入模型设计阶段，而不是事后压缩。
5. **单 target 的评估噪声，已经和我们在比较的方法差异一样大。** 方法之间的差异常在 0.01–0.03，而 seed 之间、target 之间的波动可达 0.1。没有多 target 的评测，后续所有排序都不可靠。

## 4. 探索路线修订

### 4.1 新增最高优先级：多 target、时间前向的评测协议

1. **FALCON M1、M2**：在 held-in session 之间按时间前向轮换，用较早的 session 作 source、较晚的作 target，同时保留 leave-one-out 作对照。方法选择以多 target 的配对差值为准。
2. **000688 sub-M**：按时间间隔分档报告（几天、几个月、一年），沿用 APST 的切分，final session 保持不碰。
3. **每个结论都要同时报告**：多 target 均值、按 target 的配对差值、seed 方差。单个 target 的结果只用于调试。
4. **量化相关的比较**继续以 source 验证 R² 为准（这条不变）。

### 4.2 Track A 修订

1. **在固定通道数据上，不再从零估计读入表。** 改为“先验表加修正”：
   - table = mean_table（或由 profile 映射得到的表）+ 逐通道对角修正 + 低秩修正；
   - 修正量用少量标签，以表空间的 ridge 闭式求解；
   - 这推广了路线图 A-M2 的收缩思路。A-M2 是这一轮最好的闭式标签方法，说明先验确实带有信息。
2. **把 Track A 的主战场移到 000688**（unit 数每个 session 都不同，平均表不存在）。前提有两个：
   - 预训练只做一次，并延长预算直到满足收敛规则；
   - 先修 §2 的第 5、7、9 项问题。
3. **修改 A-M1 的坐标系估计：**
   - 两侧都显式白化：target 的 FA 潜变量按维 z-score；source 端改为显式标准化，而不是只靠罚项；
   - 用动力学约束确定零空间（路线图 A-M1 第 5 步），或者把 K 降到能被行为标签约束住的维度，或者加入位置、速度和多个滞后作为更丰富的行为特征；
   - FA 改为确定性初始化并多次重启，同时报告解的方差。
4. **profile→表的映射 g 改为 episodic 训练**（路线图 C8.4）。目前 g 是在 6 个 source session 上事后最小二乘拟合的，样本太少。
5. **新的决策门槛**：在 000688 的多个 target 上，“先验表加修正”或修正后的 A-M1，在 8 个以内的标签 trial 下，要达到 UI 梯度基线用 33 个 trial 的水平。

### 4.3 Track B 修订

1. **默认使用原始坐标系**，旋转坐标版本只作历史对照。
2. **继续做路线图的 B2（source offset 基）和 B3（闭式拟合）**，但要和 LoRA 在相同的寄存器与参数预算下比较。在存内系统里，LoRA 也不能合并进 RRAM，只能作为数字旁路，两者的硬件代价相近，应按同口径比较。
3. **定位**：offset 作为 M2 这类漂移较大任务的候选，不作为通用默认。如果要进入候选，需要先在 CPU 部署路径中实现它。

### 4.4 Track C 修订

1. **新增 C10：容量放在无状态层，状态保持小。** 比较以下几种做法：
   - wide 宽度配小状态（例如 256 宽、N=8，状态约为现在 wide 的四分之一）；
   - 加宽 FFN（放在 RRAM 里，没有状态）；
   - 增加层数但每层状态小。

   每个变体都要按 128 KB 的 SRAM 预算核算后再训练。目标是在预算内拿回 M2 从 small 到 wide 的那约 0.09。
2. **新预训练配方的消融**：B 阶段同时改了五六项，需要逐项回退，看清哪几项真正起作用，再决定 Track A 的 AP 骨架保留哪些。
3. **AP 骨架的瓶颈 K**：K=16 在 2020-10-28-Run1 上的 mean_table 为 0.342，低于同一 target 上直接 96 通道 small 模型的 none（0.370）。需要扫 K（16/32/64），确认瓶颈的代价。
4. **幅度收缩单列为研究问题（C11）**：先判断它是信息不足还是目标函数造成的。可以比较不同损失函数（例如加入方差匹配项）以及更长的上下文，并按 trial 内位置分段报告。
5. **流式**：统一训练和部署的 angle 精度；补上 batch ≠ nheads 的测试。
6. **收敛规则推广到 AP 预训练**：best_step 落在预算后 20% 时，自动延长预算。

### 4.5 Track D 修订

1. **状态大小正式成为模型设计约束**（与 C10 联动）。在硬件成本模型里，状态 SRAM 是硬约束，而不是事后核算的指标。
2. **边界算子的优先级按证据重排**：输入行增益（归一化）> 平均表加对角修正 > 输出 ridge > state offset > 低秩旁路。
3. **其余部分不变**：ADC 部分和仿真、联合漂移实验、存内原生原型匹配仍按原计划进行，但都排在 4.1 的评测协议之后。

### 4.6 工程流程修订

1. **两条工作线并行修改同一批文件，已经出现了冲突。** 例如审查期间 `mamba3_streaming.py` 被改动，000688 预训练被重复执行。今后按模块划分归属，每条工作线在自己的分支上开发。
2. **结果入库前先跑完整测试，并补上审查暴露的测试缺口**：batch ≠ nheads、混合通道数、坐标系的稳定性。
3. **长任务的队列**：完成标记、失败重跑、原子缓存这三件事先修好（§2 第 11、12 项）。

### 4.7 新增 Track E：集合注意力读入

#### 4.7.1 动机与证据

神经群体本身是一个集合：不同 session 的 unit 数不同，没有天然顺序，个别 unit 会丢失或新出现。对置换等变、能接受任意 unit 数 N 的编码器，正好对应“群体活动 → 行为”这个映射。

仓库内已有直接证据（`APST/results/dandi688_reference.json`，000688 sub-C，6 个 session 的等权平均 R²）：

| 条件 | R² |
|---|---:|
| 只用活动的集合编码（act_only） | 0.402 |
| 活动加 profile 双路输入（dual_site） | **0.778** |
| profile 只进入 identity 路径（identity_only） | 0.755 |
| profile 只进入 token 路径（token_only） | 0.772 |
| dual_site 模型，把 profile 在 unit 间打乱 | −0.0005 |
| 对照：source 预训练的 LSTM 在 target 上用 32 个 trial 梯度微调 | 0.772 |

在 unit 数可变的数据上，“以 unit 描述作条件的集合编码”是跨 session 起作用的关键；它和 LSTM 微调的精度相当，但 target 端不需要梯度。

需要谨慎的三点：
1. 在 FALCON 的固定通道数据上，unit 级前端（unit/session 迭代）没有赢过“affine + LoRA”。归一化加平均表已经吸收了大部分漂移。
2. APST 与我们的 SSM 不只差在集合编码器，还有时间模型和校准 bank，不能把它的优势全部归因于 set attention。
3. FALCON 只有 3–6 个 source session，集合模型容易过拟合。它的价值需要在 000688（sub-C 68 个 session、sub-M 22 个）这种规模上体现。

#### 4.7.2 位置：放在最前面

1. 集合在输入端：unit → 潜变量。之后潜变量维度固定，由 SSM 沿时间轴处理（set-temporal 分解）。
2. 输出端不加 set attention。只有两种情况例外：输出本身是集合（例如效应器数目可变）；或者需要在任意时刻查询输出（如 POSSM 的输出 cross-attention）。FALCON 每 20 ms 输出固定维度，用不上。
3. 不用 SSM 扫描 unit 轴。SSM 依赖顺序，而 unit 没有顺序。集合轴用注意力或池化，时间轴用 SSM。

#### 4.7.3 变体谱系（从便宜到昂贵）

记 descᵢ 为 unit i 的 session 级描述（AP profile、活动统计、部分标签下的行为关联），x_{t,i} 为它在时刻 t 的归一化计数。

| 变体 | 形式 | 每步代价 | 部署形态 |
|---|---|---|---|
| **E0：Deep Sets 线性池化** | z_t = Σᵢ x_{t,i}·g(descᵢ) | 一次线性读入 | 就是当前的 AP 折叠读入：SRAM 小表加 RRAM 大矩阵 |
| **E1：静态 key 注意力池化**（Set Transformer 的 PMA） | query 为学习到的 slot；key 由 descᵢ 计算；value 对 x 线性：z_{t,s} = Σᵢ α_{s,i}·x_{t,i}·v(descᵢ)，α = softmax(q_sᵀk(descᵢ)) | session 内 α 不变，折叠成 K×N 表 | 与 E0 相同。**主推变体** |
| **E2：慢变 key** | descᵢ 用 EMA 持续更新，每隔几秒重新计算 α 并重新折叠表 | 平时为零，定期重折叠 | 与 E0 相同，加一个定期更新表的流程。可追踪 session 内漂移 |
| **E3：线性注意力集合池化** | M_t = Σᵢ φ(k_{t,i})·v_{t,i}ᵀ，slot 读取 φ(q_s)ᵀM_t / (φ(q_s)ᵀΣᵢφ(k_{t,i}))，不用 softmax | O(N) 次外积累加 | 与 SSM 状态更新、AP 累加是同一种运算，可共用外积单元（路线图 D4） |
| **E4：动态 key 的 softmax 注意力**（APST 的做法） | key 由每个 unit 当前的活动窗口算出 | 每步 N×S 个打分加 softmax | 打分可改写为静态矩阵乘：q_sᵀW_k·tokenᵢ = (W_kᵀq_s)ᵀ·tokenᵢ，可放 RRAM；softmax 和加权求和在数字端。**只作为精度上限** |

两个与 SSM 特别契合的组件：

1. **unit 级共享 SSM**：一个很小的 SSM，参数在所有 unit 间共享，替代 APST 的逐 unit 因果卷积，为每个 unit 提供时间特征。它保持置换等变；权重被 N 次复用，适合放 RRAM；每个 unit 只需要很小的状态（需要计入 SRAM 预算）。整体结构变为“unit 级 SSM → 集合池化 → 群体级 SSM”。
2. **Slot Attention 式竞争分配**：softmax 沿 slot 维而不是 unit 维，unit 之间竞争 slot，相当于软聚类。它是路线图 A-M5“联想原型匹配”的可学习版本。

#### 4.7.4 与现有路线的关系

1. Track E 是 Track A 的可学习版本：AP profile 作为 E1 的 key；A-M5 原型匹配是 E1 的特例；B4“由 session 统计生成调节量”对应以 session 为条件的 query。
2. 它绕开了 Track A 第一阶段暴露的问题：不再从零闭式估计读入表和 FA 坐标系，而是在预训练中用 episodic 方式学习“描述 → 注意力权重”的映射（路线图 C8.4）。
3. 部分标签的用法不变：标签进入 descᵢ（每个 unit 的行为关联），无标签数据提供活动统计。target 端只需要计算描述、重新折叠表，不需要梯度，符合路线图原则 3。

#### 4.7.5 实验设计

1. 实现 E0、E1、E2；E4 作为精度上限；E3 作为硬件变体，放到第 3′ 阶段。
2. 预训练数据：000688 sub-C 与 sub-M、O'Doherty、FALCON held-in。训练时每步随机子采样 unit、打乱顺序、做通道 dropout，这对集合模型是必须的。
3. 评测使用 4.1 的多 target、时间前向协议，另加三项集合性质检查：
   - 打乱 unit 顺序后输出不变（数值误差以内）；
   - 随机丢掉 10%、30%、50% 的 unit 后的性能曲线；
   - 在 target 中出现、source 中从未见过的 unit 上的表现。
4. 标签预算与 Track A 相同：0、2、4、8、16、33 个 trial。

#### 4.7.6 决策门槛

1. 在 000688 的多个 target 上，相同标签预算下，E1 或 E2 不低于“先验表 + 修正”（4.2）和 UI 梯度基线。
2. 在 FALCON 上，至少在通道 dropout 条件下优于“affine + 平均表”。
3. E4 比 E1 高出噪声底以上，才值得付出每步的动态计算代价；否则部署方案只保留可折叠的 E1/E2。

#### 4.7.7 风险

1. 数据量不足时过拟合。对策是 unit 子采样、更多 session、较小的 slot 数和描述维度。
2. 动态版本（E4）每步多出 N×S 个打分和 softmax，并引入两个动态量相乘的运算，不适合存内计算。
3. APST 的 000688 结果来自 transformer 时间模型，不能直接迁移，需要在同一个 SSM 骨架上做对照。

#### 4.7.8 参考文献

1. Zaheer et al. "Deep Sets." NeurIPS 2017.
2. Lee et al. "Set Transformer." ICML 2019（ISAB、PMA）。
3. Locatello et al. "Object-Centric Learning with Slot Attention." NeurIPS 2020.
4. Jaegle et al. "Perceiver." ICML 2021；"Perceiver IO." ICLR 2022.
5. Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention." ICML 2020.
6. Azabou et al. POYO. NeurIPS 2023（unit token 加 Perceiver 式 cross-attention）。
7. POSSM. arXiv:2506.05320（逐 spike 的输入 cross-attention 加 SSM）。
8. Zhang et al. APST. arXiv:2609.39080（slot 注意力的集合编码加 profile 条件）。
9. SPINT：用于皮层内运动解码的空间置换不变 transformer，2025。[需核对]

## 5. 修订后的执行顺序

| 阶段 | 内容 | 退出条件 |
|---|---|---|
| 0′ | 多 target、时间前向的评测框架（4.1）；修复 §2 第 2、5–14 项 | 在 M2 和 M1 的多个 target 上，能复现已有候选的结果，并给出 seed 和 target 的方差 |
| 1′ | C10 容量与状态权衡；B 配方逐项消融；AP 骨架 K 扫描 | 找到一个状态在 SRAM 预算内、M2 多 target 均值接近 wide 的配置 |
| 2′ | 000688 上的 Track A（先验表加修正、修正后的 A-M1、episodic g）；Track E 的 E0、E1、E2，E4 作为上限（与 Track A 共用预训练数据和评测）；Track B 的 B2、B3 | 4.2 第 5 条和 4.7.6 的门槛 |
| 3′ | 量化感知训练、ADC 模型、幅度收缩研究；Track E 的 E3（线性注意力，硬件变体）和 unit 级共享 SSM | 混合精度加 ADC 模型后，source 验证损失不超过 0.02；E3 与 E1 的差距在噪声底以内 |
| 4′ | 联合漂移、存内原生算子 | 用数据接受或否定联合补偿的主张 |

## 6. 本次分析产物

- `results/analysis/analysis_20261005/fa_centering_check.py` 和 `fa_centering_check.json`：在两个 M2 target、两个预算下，分别对 am1 和 oracle_frame 比较均值修正前后的结果，并用一次整段解码检查 mean_table。
- 私有附注（本地，被 git 忽略）：`results/official_evalai/20261005_private_addendum.md`。
