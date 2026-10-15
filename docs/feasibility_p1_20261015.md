# 可行性验证 P1：population-conditioned static calibration（2026-10-15）

目的：从算法角度验证 DAC2027 主线是否成立。背景见 `chatgpt_pro_answer_20261015.md` 的路线 1，以及会话中确定的设计：开机校准块 + anytime 切换。

## 0. 约束（不变）

1. backbone（trunk）冻结，target 端不写 RRAM，**不反传 trunk**。
2. 只使用 target session 的 B 个有标签 trial。target 端的操作只有四类：流式累加、固定权重前向、read-in/readout 两端的闭式求解或 local rule、写 SRAM。
3. 校准结果是静态 read-in table（K×N）加行参数，≤ 32 KB。
4. 不使用 final session（000688 的 0623/0625/0626），不读取 `results/official_evalai/`。
5. 所有超参数和配置选择只在 source validation 上进行，不看 target query。
6. ui_prior（反传穿过 trunk）只作为非部署的精度参照行。

## 1. 核心问题

- **Q-A**：在相同的 B 个 trial 下，可靠性加权的群体信息（set pooling）生成的静态表，是否显著优于 “independent encoder + shrinkage”？
- **Q-B**：瓶颈是统计量噪声还是生成器容量？方法：用 oracle descriptor 做诊断。oracle descriptor 指用 target 全部有标签 trial 计算的描述，仅用于诊断，不作为候选。
- **Q-C**：扩大 source pool（把 target 之前的全部 session 都用于 episodic 训练）能带来多少提升？第一轮每个池只有 6 个 source session，第 200–400 步就过拟合。
- **Q-D**：同一套方法迁移到 sEMG（Hyser）上是否成立？

## 2. 工作线 A：000688 消融链（GPU0）

**数据**：沿用第一轮的 D1 三层与 AP K=10 backbone（`scripts/run_labeled_calibration.py` 中的 `D1_POOLS`）。
**source pool 两档**：
- O：原来的 6 个 session；
- E：target 之前的全部 sub-M CO session，不含 dev 和 final 的角色冲突项。按日期列出：days 层约 17 个，month 层 7 个，year 层 11 个。

**变体**（全部用 episodic 训练，B 从 {1…33} 中按 log-uniform 抽取，使用 unit dropout 和留一 prior；输出形式取 residual 或 absolute 中在 source val 上更好的一种）：

| 编号 | 内容 |
|---|---|
| P-prior | 只用 prior 表，不做校准。**必须有** |
| P0 | independent per-unit encoder（第一轮的 F2/F3，作为基线） |
| P1 | P0 + reliability shrinkage：d̃ᵢ = λᵢdᵢ + (1−λᵢ)μ。λᵢ 由有效 bin 数、发放率、trial 间一致性决定；B=1 时用 source 上学到的计数–不确定性关系 |
| P2 | P1 + DeepSets context：μ = meanᵢ ψ(d̃ᵢ)，生成器 g(d̃ᵢ, μ) |
| P3 | PMA：固定 seed（m ∈ {4, 8, 16}）pooling，再用固定路由 β 得到每个 unit 的条件 μᵢ，生成器 g(d̃ᵢ, μᵢ, rᵢ) |
| P4 | P3 的打分加入 log rᵢ（reliability-weighted pooling） |
| P-oracle | P2 或 P4 的生成器，输入换成 oracle descriptor，仅用于诊断 |
| ui_prior | 梯度参照（沿用已有数字，或重算） |

**评测**：先用 B ∈ {1, 8, 33} 筛选，入选的再跑完整的 {1, 2, 4, 8, 16, 33}。每个 B 抽 ≥ 5 次不同的 trial 组合，报告均值和 95% CI。指标为 query R²（物理单位）、相对 prior 的提升、与 ui_prior 的差距（amortization gap）。每一行记录 H1–H5。

## 3. 工作线 B：Hyser HD-sEMG 管线与基线（GPU1）

**任务**：先做 1dof 力回归。每个 session 5 根手指 × 3 个 trial，每个 trial 25 s；输出 5 维力。
**预处理（因果、流式）**：先对 256 通道整流，再用一阶低通得到包络（这是一个 leaky integrator，本身就是一阶 SSM），降采样到 50 Hz；力也降采样到 50 Hz。读 WFDB 时直接用 numpy 解析 `.hea` 和 `.dat`。
**协议 H-A（跨天、同一受试者）**：
- source：全部 20 名受试者的 session1，每个 session 有自己的 read-in table；
- target：每名受试者的 session2；
- calibration：B ∈ {1, 2, 5, 10} 个 trial，跨手指均衡抽取；query 为其余 trial。

**backbone**：沿用 000688 的 AP 风格结构，即 per-session read-in table（K×256）+ 共享 SSM trunk + readout。
**参照行**：
- 同一受试者 session1 的表直接用（自然的不校准基线）；
- source mean table；
- ui_prior（梯度参照）；
- within-session 上限：在 target session 内训练和测试，只作诊断。

## 4. 停止与迭代规则

1. 每一项完成后，把结果追加到对应的 README，然后继续下一项。
2. P3/P4 没有在噪声底以上胜过 P1/P2，就停止加深 attention，记录结论，转向更便宜的版本。
3. 报错时，最多花约 1 小时修复；修不好的写明阻塞原因，跳过并继续。
4. 不允许用占位实现冒充已完成的项。
