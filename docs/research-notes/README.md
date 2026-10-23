# 研究讨论汇总（2026-10-01 至 2026-10-23）

这个目录收集了项目讨论中形成的分析笔记，作为写 DAC 2027 及后续论文的背景材料。本文件是总索引和结论汇总，细节请看各分文件。

**阅读须知**
- 分文件里的标注：【原文】= 论文写明；【推断】= 讨论中的判断，未经实验验证；【估算】= 数量级估计，假设条件写在旁边，需要用自己的参数复算；【文献】= 外部论文结论（多数只读了摘要）。
- 本汇总只是转述，数字以分文件和原论文为准。截止日期、期刊分级等外部信息有时效，使用前请重新核实。
- `docs/` 根目录下已有 `SSM硬件算子与存内计算_前沿梳理.md` 和 `从SSM演进看POSSM优化方向.md`，与本目录同名文件内容相同。

## 文件索引

| 文件 | 主题 | 日期 |
|---|---|---|
| [POSSM_ASIC_cross-session_分析.md](POSSM_ASIC_cross-session_分析.md) | 从嵌入式 / ASIC 角度拆解 POSSM，跨 session 适配的硬件代价 | 2026-10-01 |
| [SSM硬件算子与存内计算_前沿梳理.md](SSM硬件算子与存内计算_前沿梳理.md) | SSM 硬件实现的五条技术线（数字加速、PIM、RRAM CIM、神经形态、算法让步） | 2026-10-01 |
| [从SSM演进看POSSM优化方向.md](从SSM演进看POSSM优化方向.md) | 从 S4 → Mamba → DeltaNet 的演进看 POSSM 的算法改进点 | 2026-10-02 |
| [投稿会议日历_2026-10至2027-04.md](投稿会议日历_2026-10至2027-04.md) | 半年内可投会议及推荐组合 | 2026-10-02 |
| [投稿期刊梳理_无芯片与流片阶段.md](投稿期刊梳理_无芯片与流片阶段.md) | 无芯片 / 有 RRAM 实测 / 流片后各阶段的期刊选择 | 2026-10-04 |
| [POSSM映射到RRAM-CiM_对照MamCIMFlow.md](POSSM映射到RRAM-CiM_对照MamCIMFlow.md) | 参照 MamCIMFlow（ICCD 2025）把 POSSM 映射到 RRAM 存内计算 | 2026-10-22 |

背景论文：POSSM（Ryoo et al., NeurIPS 2025, arXiv 2506.05320）、MamCIMFlow（Li, Wang et al., ICCD 2025）、APST（本组 set-temporal transformer，arXiv 2609.39080）、SPINT、POYO、NDT2、FALCON、MiBMI（JSSC 2024）、CEBRA、UniBCI。

---

## 1. POSSM 的硬件画像

来源：`POSSM_ASIC_cross-session_分析.md`

**结论**
- 瓶颈是权重存储，不是算力。POSSM-GRU-SS 每 50 ms chunk 约 0.3 M MAC，约 7 MMAC/s，动态功耗约 7–35 µW【估算，1–5 pJ/MAC】。参数 0.47 M【原文 Table 5】，INT8 约 470 KB（INT4 约 235 KB），约为 MiBMI 整片 16.3 kB 的 30 倍。o-POSSM（4.5–9 M 参数）不适合植入端。
- 输入交叉注意力可以改写成"事件驱动 + 查表"：query 固定，每个 spike token 只由 unit ID 和时间戳决定，所以 K/V 和注意力打分都能按 unit 预先算好。每个 chunk 约 25 K 次操作，比字面实现少 50 倍以上【估算】。计算量与 spike 数成正比，与通道数无关。
- 参数天然分三类：冻结主干（数百 KB，只读，适合 ROM / eFlash / RRAM）、session 表（unit embedding 生成的 K/V 表约 24 KB、打分表约 10 KB、embedding 约 12 KB，每次校准改写，放 SRAM）、运行状态（约 2 KB，寄存器）。
- POSSM 的 unit identification 仍要反向传播几百个 epoch（10¹⁰–10¹¹ MAC 量级），不适合植入端。推荐路线 "APST-SSM"：用 APST 的闭式 association profile（或 SPINT 式前向 IDEncoder）替换梯度式 UI，校准只剩"累加统计量 + 小规模闭式求解 + 一次前向"，表重建约 2.2 M MAC、µJ 量级【估算】。
- 延迟由 chunk 长度决定。20 ms chunk 性能基本不掉【原文 Table 6】，输出 query 预测 +50 ms 只掉 R² 0.005–0.02【原文 Table 8】。
- 芯片上建议以通道为 unit（threshold crossing，不做 spike sorting），表大小出厂固定。
- 事件流码率约 57 kb/s（192 通道、20 Hz）；若编码器放植入端，只传 64 维 z(t)，约 10 kb/s【估算】。

## 2. SSM 硬件前沿

来源：`SSM硬件算子与存内计算_前沿梳理.md`

**结论**
- 五条线：数字加速器（面向 Mamba LLM）、PIM（数据中心，发现状态更新是访存受限的逐元素运算）、模拟 RRAM CIM（投影进交叉阵列，状态衰减用器件物理实现，代表是 Michigan Wei Lu 组 Nat Commun 2026 的 WOₓ 衰减态）、神经形态（Loihi 2 上的 S4D、IMSSA、ABR TSP1）、算法向硬件让步（实数对角、共享衰减常数、Mamba-3）。
- 对 BCI（batch=1、流式、小模型）最相关的是 RRAM CIM 和神经形态两条。
- **空白**：没有人在侵入式运动解码上做 SSM 存内计算，更没有和跨 session 适配结合。
- 王中锐组相关积累：RRAM 储备池 LSM（Nat Comput Sci 2025，含 BMI 零样本）、忆阻神经 ODE、MamCIMFlow。
- 想法："一次校准，同时补偿神经漂移和器件漂移"【推断，待验证】。需要确认器件衰减时间常数能否调到 100 ms–1 s。

## 3. 算法改进方向

来源：`从SSM演进看POSSM优化方向.md`

**按优先级**
1. **Delta-rule 校准记忆**（Gated DeltaNet 式）：校准 trial 通过前向递推写入一块"校准记忆"状态，替代梯度式 UI。数学上等价于 APST 闭式岭回归的递推版，可以和推理共用硬件。被认为最可能成为算法论文主贡献。
2. 复数 / 振荡型对角主干（S4D 复数、LinOSS、Mamba-3 复数状态），匹配运动皮层旋转动力学。
3. SSM 专用参数高效适配（SDLoRA、state-offset、Δ 缩放），每 session 几十到几百个参数，硬件上是一组寄存器。
4. 连续时间 / 事件级输入（S5 可变 Δ、Event-SSM），突破 50 ms chunk 下限。
5. 长时间运行稳定性（特征值约束、带状态训练、周期复位）；POSSM 只用 1 s 序列训练。

建议实验顺序：复数主干 → SSM 专用适配 → delta-rule 校准记忆（对比 APST 闭式解）→ Δ 选择性 / 20 vs 50 ms → 事件输入 → 长时稳定性。

## 4. POSSM → RRAM CiM 映射（对照 MamCIMFlow）

来源：`POSSM映射到RRAM-CiM_对照MamCIMFlow.md`，以及 2026-10-22/23 的后续讨论

**MamCIMFlow 做了什么**：Mamba-2 的静态投影放 RRAM（6 个权重组，ADC 后集成激活），动态递推放数字 SSS 流水线（IEPE/HSPE/OCPE），握手直连去掉中间 SRAM（SRAM −66%，延迟 −62.9%）。对 RTX 4090 吞吐 14.1×、能效 319.3×。仿真（MNSIM 2.0 + 28 nm 综合），**没有报告任何精度结果**。

**映射结论**
- 存储三级划分：冻结主干 → RRAM；每 session 半静态表（K/V 表、打分表、session query）→ SRAM 或 SRAM-CIM；运行状态 → 寄存器。MamCIMFlow 没有考虑权重更新这一类。
- 优化目标反过来：POSSM 每秒只有 20 个 backbone step（每 50 ms 一步；spike 约 3.8k/s 是输入 token）。RRAM 在 BCI 里的价值是**非易失、零泄漏、高密度地存权重**，让阵列在 chunk 之间断电（占空比约 10⁻⁴），不是破内存墙。数字实现的绝对算力也只有约 6 µW。
- MamCIMFlow 消融：并行度 8× → 64×，面积涨 144×，主要是 ADC。所以用最低并行度、低精度、多列共享 SAR ADC、小阵列。
- 主干选 S4D：LTI，去掉 Dt/B/C 投影组、卷积 PE、exp/softplus；对角递推留在数字端（DRU，常系数），模拟噪声不沿时间累积。GRU 的 U·h 每步过模拟阵列，噪声累积；POSSM-Mamba 是 Mamba-1，比 Mamba-2 更难做。
- 选择性对比：Mamba-2 在递推内部做选择（Δ、B、C 随 token 变化）；POSSM 在递推之前做选择（set-attention ≈ 部分 B，输出 cross-attn ≈ 部分 C），**Δ（时间选择 / 遗忘）没有替代**。建议消融：LTI vs 每 chunk 全局标量 Δ_t vs 完整 Mamba。
- 极点精度：8 bit 对 τ 0.1–1 s 够用，多秒级太粗；量化后需保证 |λ| < 1。
- 输入端 crossbar 改写：一个 chunk 的注意力加权和 = Vᵀa（a 为每通道标量累加器），即一次（通道数 × 64）的 MVM。字面实现约 1.7 M MAC/chunk，查表改写约 60 K；通道数上千时 crossbar 改写才划算。前提是 value 不加 RoPE（未核实）。
- 规模：0.47 M 权重，INT4、2 bit/cell、差分对 → 约 1.9 M cell，约 29 个 256×256 阵列。同样权重放 SRAM（INT4 约 235 KB）在 28 nm 下泄漏约几十 µW，对比 MiBMI 整片 883 µW【估算】。
- 跨 session 适配必须无梯度：RRAM 写入需要 write-verify，慢且贵。APST 式校准的前向和表生成可以复用同一套 RRAM 阵列；同一批校准 trial 还可以顺带重估每列的数字增益 / 偏置，补偿电导漂移（每列几 KB）【推断】。
- 应报告的新指标：每次适配写入字节数、写 SRAM 还是 RRAM、校准 MAC 数和 trial 数；以及器件噪声下的 R²、跨 session 后的 R²。

**论文新颖性定位（排序）**
1. 主线：CIM 上的无梯度跨 session 适配（冻结 RRAM 主干 + APST 闭式重写 SRAM 表 + 同一次校准补偿 RRAM 列漂移）。
2. 事件驱动、与通道顺序无关的 spike-token 输入（查表 / Vᵀa crossbar）。
3. 面向占空比和泄漏的三级存储，必须打赢"数字 MAC + ROM / eFlash / MRAM 存权重"这个基线（审稿人最可能提）。
4. 辅助分析：递推环路放在哪里（数字 S4D vs 模拟 GRU vs 器件衰减）。

单独讲"第一个用于皮层解码的 SSM-CIM"被认为不够。

**架构仿真基线**：① 纯数字 ASIC（SRAM 存权重，MiBMI 式）；② 数字 + ROM/MRAM；③ "存了再取"的 RRAM-CIM（MamCIMFlow 基线）；④ 本方案。指标：每次解码能耗、尾延迟、面积、待机泄漏、每次适配写入量和能耗。

## 5. 数据集与实验约束

来源：2026-10-23 讨论

- 当前选用 4 个数据集：DANDI 000688（Perich/Miller 两只猴子长期 M1/PMd 单元，2D 光标速度，主证据）、FALCON M2（固定 96 通道，2D 手指速度）、Hyser HD-sEMG（256 通道，手指力，跨天）、NinaPro DB6（多天抓握分类）。EMG 用整流平滑包络，不做 MU 分解。
- 讨论建议：现阶段优先补 spike 数据而不是 EMG。可加 FALCON H1（人，7 维速度）、O'Doherty–Sabes Indy/Loco（连续伸够，session 多）。FALCON M1 输出是 EMG，不适合本模型的运动学输出。EMG 留给单独的"统一传感器"论文；若投 TBioCAS 类可只保留 Hyser，先去掉 DB6。
- EMG 包络路径是稠密的：每通道每 chunk 一个 token，若包络以加性 value embedding 进入，编码器 = Vᵀa + (Σaᵢxᵢ)v_u，可与神经路径共用 Vᵀa crossbar（神经端 a 稀疏，EMG 端 a 稠密）。Hyser 输入约 16 K MAC/chunk，表 256 行约 45 KB【估算】。统一架构的卖点是"减少重校准负担"，不是精度；前端表示不同要承认。
- 约束：不做大规模多动物预训练（不认为 >10 Mb RRAM 集成可行；GPU 约 RTX 3090 级，实验显存 ≤12 GB）。主干 ≤0.5 M 参数（INT4 约 1.6 Mb）作为显式设计约束。主场是同一被试跨 session，不是跨被试。
- 与大模型对比：NDT3 难以直接比较。用 POSSM Table 1 中来自 DANDI 000688 的 session（C-CO、T-CO、T-RT；含 NDT-2、POYO、o-POSSM UI/FT 结果）和 FALCON leaderboard 作主对比，NLB MC_RTT/MC_Maze 只做单 session sanity check。表里加参数量和适配代价两列。
- 代码：POSSM 官方代码截至 2026-10-01 未公开。可基于 torch_brain 的 POYO 组件 + 自写 S4D/GRU 搭建；评测用 APST 仓库的 FALCON / DANDI688 harness；基线 SPINT 官方仓库。
- 算力：单卡开发 / 消融可用 Colab（L4 优先，12 h session 上限，需 checkpoint）或 Kaggle（45 h/周）；大规模训练用 UTokyo HPC（Miyabi / Wisteria）。

## 6. 投稿规划

来源：`投稿会议日历_2026-10至2027-04.md`、`投稿期刊梳理_无芯片与流片阶段.md`

- 主推：**DAC 2027**（11/11 摘要、11/18 全文）投 RRAM-CIM SSM 软硬件协同；**ICML 2027**（约 1 月下旬，推测）投算法（SSM + 前向式跨 session 校准），被拒顺延 NeurIPS 2027。
- 低成本占位：Cosyne 2027（10/18，非存档摘要）、BCI Meeting 2027（1/15 摘要）、EMBC 2027（约 2–3 月）、NICE 2027（11/11，偏神经形态的小众会议，IEEE 论文集）。VLSI 2027（1/25，京都）只在有可测 RRAM 芯片时才值得冲。
- 期刊：无芯片阶段算法投 JNE / TNSRE / TMLR，软硬件仿真投 TBioCAS / TCAS-I / TCAD；有 RRAM 实测（与王组合作）冲 Nat Commun / Nat Comput Sci；流片后 VLSI → JSSC、TBioCAS，闭环演示可冲 Nature Electronics。期刊分级未重新核实。

---

## 7. 待办与待确认

**需要核实**
- [ ] POSSM 的 value 是否加 RoPE（决定 Vᵀa crossbar 改写是否严格成立）。
- [ ] POSSM 输出注意力头数和维度；S4D 在 POSSM 里的状态维度（讨论中假设 N=64）和是否用 GLU。
- [ ] 王组 RRAM 工艺参数：多值 cell 位数、读噪声、漂移速率、write-verify 代价；能否共享 MamCIMFlow 仿真流程。
- [ ] 器件衰减时间常数能否调到 100 ms–1 s。
- [ ] SRAM 泄漏数字用实际工艺库复算。
- [ ] 包络在模型中如何进入（每通道 token + value embedding，还是固定顺序向量）。
- [ ] NinaPro DB6 规格；ICML / EMBC 等推测日期在 CFP 公布后复核。

**实验**
- [ ] 复现 POSSM-S4D-SS（基于 torch_brain POYO 组件）。
- [ ] 用 APST 闭式 profile / delta-rule 校准记忆替换梯度式 UI，在 FALCON M2 + DANDI 000688 上对比 UI / FT / LoRA。
- [ ] CIM 感知训练与评测：RRAM 层 INT4/INT8 + 共享 ADC 位宽、数字层定点、读噪声 / 漂移 / 器件差异噪声模型；报告理想、加噪、跨 session 后的 R²（MamCIMFlow 缺的那组结果）。
- [ ] 验证"同一次校准补偿 RRAM 列漂移"。
- [ ] 消融：LTI vs 每 chunk 标量 Δ_t vs 完整 Mamba；复数 vs 实数主干；20 vs 50 ms chunk；时间戳分辨率 1/2/5 ms。
- [ ] 长时稳定性：拼接多 trial 的长序列测试。
- [ ] 在 Hyser 上测试闭式手势 / 手指 tuning profile 是否成立（EMG 路线的早期风险）。

**架构仿真**
- [ ] RRAM PE 用 MNSIM 2.0（或王组框架），数字单元（输入累加、DRU、输出注意力、校准引擎）用 DC 综合。
- [ ] 对比 4 个基线（纯数字 SRAM、数字 + ROM/MRAM、存了再取 RRAM-CIM、本方案），报告能耗、尾延迟、面积、待机泄漏、适配写入量和能耗。
