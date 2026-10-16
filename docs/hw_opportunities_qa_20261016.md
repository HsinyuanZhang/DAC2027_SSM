# 硬件优化机会：问题—证据整理（2026-10-16，v0）

范围：植入式主场景（000688 K=10 AP backbone；M2 wide_n16），sEMG 作为边界。每条按 **问题 → 证据 → 结论 → 缺口** 组织。证据只引用已经审查过的结果；没有证据的明确标为缺口（对应 §5 的补充实验）。算子与带宽的定量数字来自 `docs/arch_analysis_operators_dataflow_20261016.md`（进行中，完成后回填 §2、§4）。

证据来源缩写：
- [PTQ] `results/analysis/rram_quant_20261003/README.md`（Mamba-3 128×2，M1/M2，source val）
- [P2] `results/analysis/population_calib_p2_20261015/README.md`（审查后修正版：逐次累加取整、计数器精确）
- [P3] `results/analysis/population_calib_p3_20261016/README.md`（§R.12）
- [SB] `docs/stage1_capacity_k_ablation_20261005.md` §1（state budget）
- [sEMG] `results/analysis/semg_calib_20261016/README.md`

---

## 1. 不同精度的量化可以用在哪里

| 问题 | 证据 | 结论 | 缺口 |
|---|---|---|---|
| Q1.1 trunk 的哪些权重可以压到很低的位宽？ | [PTQ] source val 上逐层组改成 ternary：B/C 行 −0.001 至 +0.008，FFN 单层 −0.001 至 −0.009；in_proj 的 z/x 行 −0.016/−0.020，out_proj −0.024。“FFN + B/C ternary、其余 8 bit” 约 1.45 Mi 差分对，损失约 0.01 | **B/C 行和 FFN 用 ternary（2T2R 单对）；in_proj z/x 与 out_proj 保持 4–8 bit**——典型的逐层混合精度 | 只在 M1/M2 的 128×2 模型上测过；**000688 AP backbone 未测**；未测“量化 trunk + 校准”联合效果（E-Q1） |
| Q1.2 RRAM 上的激活（MVM 输入）用多少位？ | [PTQ] RRAM 层输入 INT8、静态尺度在 support 窗口上标定，全部 8 bit 权重时 source val 损失 ≤ 0.002 | INT8 激活可行 | ADC 位宽与部分和量化未建模 |
| Q1.3 SSM state 用多少位？ | [SB] 按 INT24 计 state 预算；`hardware_contract` 的 real-2×2 引擎在 int16 下 256 条随机驱动出现 4,656 次饱和，int24/int32 为 0 | int16 state 有饱和风险，int24 安全 | **端到端解码精度随 state 位宽的变化未测**（E-Q2） |
| Q1.4 校准用的固定权重 encoder（存于 RRAM）用多少位？ | [P2] int8 \|ΔR²\| ≤ 0.0001；int4 ≤ 0.002；ternary PTQ −0.03 至 −0.06，source QAT 后 days 基本恢复、month/year 仍约 −0.02 | **encoder int4 无损；ternary 需要 QAT 且不完全无损** | — |
| Q1.5 校准累加器用多少位？ | [P2] 逐次取整：数据累加器 16 bit 无损（最差 −0.0005），14 bit 在 month/year 出现 NaN，12 bit 不可用；计数器必须精确，B=33 时需 17 bit。HD-sEMG 连续值特征：16 bit 时 Gram 相对误差 0.9，24 bit 4e-3，28 bit ≤ 6e-5 | **spike count 输入：16 bit 数据 + 17 bit 计数器；连续值 sEMG 特征：约 28 bit**——位宽取决于输入的数值类型 | sEMG 的定点结论只到误差层面，未做端到端解码 |
| Q1.6 read-in table（SRAM）用多少位？ | [P2] int8 无损，int4 损失 ≤ 0.005 | **int8（或 int4）表** | — |
| Q1.7 readout ridge（R1-stream）的 Gram 与求解用多少位？ | 无（[P3] 全部在 fp32 下运行） | — | **缺口**（E-Q3） |

## 2. 算子融合可以用在哪里

| 问题 | 证据 | 结论 | 缺口 |
|---|---|---|---|
| F2.1 输入归一化能否并入 read-in table？ | 归一化是逐 unit 的仿射 (x−μ)/σ，read-in 是线性的：T·diag(1/σ)·x − T·(μ/σ)，数学上精确等价 | **归一化折叠进表和一个 K 维 bias**，推理时省掉 N 次减法和除法 | 需要确认 int8 量化在折叠之后做（折叠会改变每列的动态范围） |
| F2.2 read-in table 能否和 trunk 的 in_proj（W_lift）合并成一次 MVM？ | 结构：x(N) → T(K×N, SRAM) → W_lift(H×K, RRAM)。K=10 是瓶颈维 | **不合并更好**：两级 MVM 的代价是 K·N + H·K；合并后是 H·N，而且合并后的矩阵会随 session 变化，必须放在 SRAM 里，失去 CiM 的优势。K 瓶颈正好让“小 SRAM 表 + 固定 RRAM”成立 | 用 arch 文档中的实际 H、N 给出数字 |
| F2.3 prior + 残差能否融合？ | 表 = 先验列 + encoder 输出的残差，在切表时一次写入 | 校准时融合，推理时只有一张表 | — |
| F2.4 校准累加能否与推理共用同一次输入读取？ | 两者在每个 bin 都读同一个 x；累加只多做外积和加法 | **共用输入读取**（单次读取、两路使用） | 带宽数字见 arch 文档 |
| F2.5 R1 readout 的解能否折叠进 head？ | V = V₀ + R·G⁻¹ 仍是线性 head | 求解后直接覆盖 head 参数，推理时没有额外代价 | — |
| F2.6 P0 encoder 能否按 unit 批量执行？ | 同一组权重作用于 N 个 unit 的描述向量 | **weight-stationary 的批量 MVM**，权重复用 N 次 | 映射与利用率见 arch 文档 |

## 3. 近似计算可以用在哪里

| 问题 | 证据 | 结论 | 缺口 |
|---|---|---|---|
| A3.1 softmax（P3/P4 pooling） | [P2] exp2 LUT（256 项 × 16 bit）+ reciprocal：\|ΔR²\| < 5e-5 | **LUT softmax 无损** | 在线（流式）形式未测 |
| A3.2 readout ridge 的低秩与对角近似 | [P3] lr16 在 B ≤ 16 时保留 65–98% 的增益，B=33 时只保留 34–69%；对角近似在 B=33 时很差 | **小 B 时可用低秩 ridge**，大 B 时需要完整的 Gram | 未做定点 |
| A3.3 不重放的流式 readout（R1-stream） | [P3] 相对需要重放的 R1，精度损失 ≤ 0.013 | **用流式累加替代重放缓存**，SRAM 从约 100 KB 降到约 6 KB | year 层 B=4/8 时抵消了 R1 的全部增益 |
| A3.4 用平均雅可比近似梯度（F7-J） | [P3] 与真实梯度的余弦为 0.26–0.45，但追回 60–71% 的差距；与 readout 结合后追回 75–77% | 近似梯度足以完成大部分精修，不需要反传 trunk | 只做架构级建模（按 MZ 的决定） |
| A3.5 截断重放的 trial 长度 | [P3] cap256 下，R2/R3 的变化在 CI 之内 | 可以把最坏情况下的缓存限制在 114.5 KB 以内 | — |
| A3.6 describe() 中的非线性（log、sqrt、除法） | 无 | — | **缺口**（E-A2） |
| A3.7 trunk 中的非线性（dt 的 softplus、衰减的 exp、旋转的 sin/cos、norm 的 rsqrt、SiLU 门控） | 无 | — | **缺口**（E-A1） |
| A3.8 事件驱动：跳过零 spike count | 数学上精确（零值的 MAC 结果为零） | 节省多少取决于输入稀疏度 | 稀疏度在 arch 文档中实测 |

## 4. Memory 的瓶颈在哪里

| 问题 | 证据 | 结论 | 缺口 |
|---|---|---|---|
| M4.1 推理时最大的 SRAM 占用是什么？ | [SB] state 预算（INT24）：small 128×2×32 为 50.4 KB；wide_n16 256×4×16 为 103.9 KB；wide 256×4×32 为 201.8 KB（超出 128 KB）。[P3] 000688 trunk state 约 32 KB | **SSM state 是推理时的主要 SRAM 占用**，而且随宽度、层数、state size 线性增长；M2 的 wide_n16 已接近上限 | 000688 backbone 的精确数字见 arch 文档 |
| M4.2 校准时的 SRAM 占用 | [P2] 000688：P0 6.8 KiB，P4 7.8 KiB（16 bit 累加器、int8 表）；表 double buffer 每份几百字节 | 植入式规模下，**校准的 SRAM 很小** | — |
| M4.3 通道数增大时，校准的 SRAM 如何变化？ | [P2]/[sEMG] HD-sEMG（N=256）：N×N Gram 在 28 bit 时 224 KiB，总计约 330–397 KB | **N×N Gram 是扩展的瓶颈**，必须换成 N×K 潜变量累加器（估算约 62 KB，未评估） | **缺口**（E-M1） |
| M4.4 精修需要的重放缓存 | [P3] 000688、B=33：p95 为 138–143 KB，最坏 206–319 KB；cap256 后 ≤ 114.5 KB，但加上 state 后只有 days 层放得进 128 KB；HD-sEMG 规模 B=8 时 2.6 MB | **重放缓存是 F7-J 的硬伤**，只适合小 N 且 B 有限的场景 | — |
| M4.5 数据带宽的瓶颈在哪里？ | 待 arch 文档 | — | 待补 |

---

## 5. 补充实验（填补上面的缺口）

| 编号 | 内容 | 资源 |
|---|---|---|
| E-Q1 | 000688 AP backbone 按 [PTQ] 的方案做混合精度 PTQ（B/C + FFN ternary、其余 8 bit、INT8 激活），在量化 trunk 上重跑 P-prior、P0 和 P0+R1-stream：校准是否仍有效？能否吸收一部分量化误差？ | 只做评测，GPU 约 1–2 小时 |
| E-Q2 | trunk state 位宽 12/16/20/24 bit（逐步取整并饱和），测端到端 R² | 只做评测 |
| E-Q3 | R1-stream 的 Gram 与求解在 16/20/24 bit 下的定点模拟 | CPU 或 GPU，很小 |
| E-A1 | trunk 非线性用 LUT/PWL 近似（softplus、exp、sin/cos、rsqrt、SiLU），逐个替换与全部替换，测 R² | 只做评测 |
| E-A2 | describe() 中的 log、sqrt、除法用 LUT/PWL，测 R² | 只做评测 |
| E-M1 | 为 HD-sEMG 实现并评估 N×K 潜变量累加器（替代 N×N Gram） | 需要重训 calibrator（sEMG 是边界场景，优先级最低） |
