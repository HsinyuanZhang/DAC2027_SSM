# DAC2027 论文主线提案（2026-10-16）

依据：P1–P3（000688）、Hyser、DB6 v2、sEMG calibrator 的结果，全部经过独立审查和修正（见 `results/analysis/*/README.md` 与 Overleaf `exp_log.tex`）。M2 正在 Colab 上运行，结果待补。截止：abstract 11-11，full paper 11-18。RTL 需要在 **10-28 前冻结算法**。

## 1. 证据汇总

| 结论 | 证据 | 强度 |
|---|---|---|
| **固定权重的 per-unit encoder（P0）+ electrode-aligned prior** 在稳定植入（000688）上很强 | B=8：days 0.723 / month 0.565 / year 0.256（prior 只有 0.36 / 0.33 / 0.13）；B=1 时 days 已达 0.65；days 层接近这个 trunk 的静态表上限（约 0.75）；在 days 层每个 B 下都超过带正则的梯度方法 | 强（已审查：无泄漏）；但只有 4 个 target |
| **不反传 trunk 的精修**能追回大漂移时剩余差距的约 3/4 | R3（F7-J + readout ridge）：month 75% [71, 78]，year 77% [69, 84]（口径一致、因果选参） | 中（month 和 year 共用 trunk、各 1 个 target） |
| 校准通路**对量化很友好** | calibrator 权重 int8/int4 无损；ternary 需 source QAT；累加器 16 bit（spike count，逐次取整）；输出表 int8 无损；exp2 LUT softmax 无损 | 强（逐步模拟硬件行为） |
| 校准**代价很小**（P0），精修代价可选 | P0：约 2.4 M MAC、7 KiB SRAM；R1-stream：额外约 6 KB；F7-J：3.9–7.8 G MAC（约 90% 为固定权重 MAC），需要重放缓存（截断到 256 bin 后，小 N 时可放进 128 KB） | 中（解析模型，尚无 RTL） |
| **set pooling 只在通道身份未知时有用** | Hyser 无身份条件下，P2 比容量匹配的 P0-wide 高 +0.20（B ≥ 5）；身份已知时增益 ≤ 0.03（000688 与 sEMG 都如此） | 中（不 pooling 的变体存在优化失败的混杂；绝对水平低） |
| **sEMG 上 fixed-weight calibrator 不占优** | Hyser B ≥ 5 时比闭式 ridge 低 0.08–0.18；DB6 没有显著超过最好的可部署基线，B=16 时离 ui_prior 还差 0.14–0.17；HD-sEMG 的 N×N Gram 需要约 397 KB SRAM | 强（负面） |

## 2. 建议的论文故事

**题目方向**：Calibrate-by-Forward：面向植入式 BCI 的冻结 CiM SSM 与 SRAM 驻留的固定权重校准引擎。

**贡献（按 MZ 的排序）**
1. **硬件友好的 adaptation 算法**：一个分层的、只用前向计算的校准体系，不反传 trunk，不写 RRAM：
   - 第 0 级：electrode-aligned prior + B-trial 归一化；
   - 第 1 级：P0 固定权重 per-unit encoder（流式累加 → 固定权重 MLP → int8 read-in table）；
   - 第 2 级（可选）：R1-stream readout ridge；
   - 第 3 级（可选，大漂移时使用）：F7-J 固定反馈的局部更新 + readout ridge。
   
   每一级都给出“精度—MAC—SRAM”的 Pareto 点。
2. **它的硬件实现**：共享的 rank-one/MAC 数字引擎（累加、MLP、ridge 求解、F7-J 外积），计数器与累加器的位宽规则（16 bit 数据、17 bit 计数器），double buffer anytime 切表。由 MZ 负责 RTL 和综合，我负责架构级模型和 RTL 规格。
3. **借助 CiM 的硬件友好 SSM**：trunk 放在 2T2R RRAM 上，混合精度（已有 PTQ：FFN + B/C ternary、其余 8 bit，损失约 0.01）；校准对 trunk 量化误差的补偿是一个可选的增量实验。

**数据集的定位**
- **主结果**：000688（三个漂移层）+ M2（固定通道，公开 benchmark）——植入式场景。
- **泛化与边界**：Hyser / DB6（sEMG，每次重新贴电极，身份不稳定）。作为“通道身份未知”的场景展示 set pooling 的作用，并如实报告 fixed-weight calibrator 在这类数据上的局限。

## 3. 投稿前必须补的实验（按优先级）

1. **000688 的统计效力**：目前只有 4 个 target。用滚动的时间前向方案增加 target（或加入 sub-C 的 68 个 session），在 Colab 上并行训练 backbone 和 calibrator。
2. **M2 结果**（Colab 上运行中）。
3. **硬件代价表**：按 RTL 规格给出每一级的 MAC、SRAM、延迟和能耗估算（架构级），交给 MZ 的 RTL 对照。
4. **trunk 量化与校准的联合实验**：混合精度 trunk 下重跑 P0 / R3，确认校准不受 trunk 量化影响，或者能部分补偿。
5. （可选）sEMG 上的 N×K 潜变量累加器，替代 N×N Gram。

## 4. 需要 MZ 决定的事

1. 数据集定位：是否同意“000688 + M2 为主，sEMG 作为身份未知场景与边界”？
2. set-attention 的角色：是否接受它作为“身份未知时”的可选模块，而不是主要贡献？
3. F7-J 是否进入 RTL 范围？它的代价比 P0 高三个数量级，需要重放缓存。
4. 10-28 冻结给 RTL 的范围：建议第 0–2 级必做，第 3 级可选。

## 5. MZ 的决定（2026-10-16）

1. 数据定位：**植入式（000688 + M2）为主，sEMG（Hyser / DB6）作为“通道身份未知”的场景与边界。**
2. RTL 范围：**第 0–2 级必做**（prior + B-trial 归一化、P0 固定权重 encoder、R1-stream readout ridge）；**F7-J 只做架构级建模**，作为可选精修级的 Pareto 点。
3. **（同日更正）RTL 不是重点。** 贡献 2 改为架构级分析：系统中有哪些算子；流式输入时各处的数据带宽；存内计算（MVM）的使用方式与占比；特殊算子（非线性、norm、softmax、describe 中的 log/sqrt/除法、SSM 递推中的 exp/softplus/旋转等）的优化。对应文档：`docs/arch_analysis_operators_dataflow_20261016.md`（进行中）。

## 6. 冻结算法（MZ 决定，2026-10-19）

- **主方法**：P0（electrode-aligned prior + 固定权重的 per-unit encoder，统计量流式按前缀累加）+ W_α（向 ridge-CCA 表收缩，α 按扩展窗口协议选）+ S1-causal re-based 固定反馈精修（即“组合方法”）；备选为 CCA + S1。
- **主协议**：causal-first-B（前 B 个 trial 校准、之后的 trial 作为 query、统计量按前缀累加、anytime 提交、整段连续解码）。
- **可部署的对照方法**：P-prior、P0、ridge-CCA、W_α、S1-causal；参照：R2/R3（重放）、P0 + ui_reg（不可部署）。
- 证据：`results/analysis/causal_main_20261019/README.md`（两轮审查）。
- 后续三条线并行：第 2 轮 Colab（增加 backbone 和选参 session）、硬件架构分析（`docs/arch_analysis_20261019.md`）、论文初稿与 exp-log。

## 7. 主方法改为 P0 + S1（MZ 决定，2026-10-21；取代 §6 的主方法）

- **主方法**：P0 + S1（S1-causal re-based 固定反馈精修，η/ρ 按 B 在 source 上选，AND 门；门不过时 η=0，退回 P0）。**COMBO（W_α + S1）和 CCA+S1 降为对照**；ridge-CCA、W_α 仍是可部署对照，R2/R3、P0+ui_reg 仍是参照。
- **原因**：
  1. 证据：round 2（审查后，34 pair / 26 session，`results/analysis/expansion_20261016/round2_README.md`）B4–33 相对 P0，S1 (a) 在 ≤30 d +0.003（不伤 P0）、31–180 d **+0.055 [+0.034, +0.076]，16/17**（COMBO +0.039，R2 replay 参照 +0.050）、>180 d +0.071（压力测试）。
  2. 硬件统一（MZ 要求：不按日期/条件换硬件）：P0 + S1 在所有间隔和 B 下是同一条数据通路；不同条件只改寄存器里的 (η, ρ)。S1 的 (η, ρ) 只按 B 选，不按 gap 分箱（`ssm_decode/expansion_causal.py` `target_rules`：S1 pool 不分 gap bin；只有 α 分箱），设备不需要知道距上次校准的天数。
  3. 去掉 CCA 求解单元（eig/SVD）和 77.5 KiB 工作区；峰值 SRAM 从 158.5 KiB 降到约 81 KiB 常驻（`docs/arch_analysis_20261019.md` §3），进入 128 KiB 预算。
- **不把“可重构”作为创新点**；硬件叙事是“校准复用推理硬件”：S1 的前向就是 CiM trunk 的推理前向，读入表在 SRAM，校准只多流式累加器与小的更新单元。
- **前提（H4，必须验证）**：S1 评估目前对每个校准 trial 从零状态单独前向，部署推理是整段连续、状态跨 trial 携带（`docs/arch_analysis_20261019.md` G1）。若 live 残差下 S1 的增益保持，上述叙事成立；否则校准期需 shadow forward（trunk MAC ×2，约 50 KiB 额外状态）。
