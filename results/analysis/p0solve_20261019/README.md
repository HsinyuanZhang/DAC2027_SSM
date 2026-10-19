# P0-Solve 与 W_alpha shrinkage（000688 sub-M，2026-10-19）

目的：在冻结 trunk 上比较两件事先于任何更复杂的校准器必须面对的东西。一是 shrinkage 基线 \(W_\alpha=W_0+\alpha_B(W_{\mathrm{CCA}}-W_0)\)，\(\alpha_B\) 只在 source simulation 上选。二是 P0-Solve：target 只做原始坐标的流式累加，再解一个 \(p\times p\) 的线性方程，不反传。投影诊断 \(W_{\mathrm{diag}}\) 只解释 ui_reg 的更新方向有多少已经落在 \(H^\mathsf T D\) 的行空间里，不是候选。

原始 rows 在 `results/p0solve_20261019/`（gitignored）。本目录是汇总：`headline.csv`、`cells.csv`、`paired.csv`、`projection.csv`、`projection_controls.csv`、`projection_controls_summary.csv`、`selection.csv`、`cost.csv`、`rows_long.csv.gz`、`p0_vs_drift_summary.json`。

## 审查后修正（2026-10-19，独立审查）

下面 §3 的 P0-Solve 列和 §4 的第一稿结论是第一轮网格、且把投影增益当成了「行为统计量本身」的证据。审查后的结论以本节为准。

**投影不是行为监督特有的。** Drift target 的 N 只有 10–30。B=33、p=16 时 rank(H) 的均值是 14.2，N≤17 的 rank/N 约 0.93–1.00（20141203 上 rank=N=10，能量分数 1，W_diag 与 W_ui 相同）。同一秩的对照（Opus 审查脚本 `proj_control.py`，数字导入 `projection_controls.csv`，H 与存档的 W_diag 一致）：

| B | p | H | PCA | random | 打乱 trial 标签 | 能量 H / random / PCA |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 16 | 0.863 | 0.822 | 0.613 | 0.840 | 0.711 / 0.547 / 0.712 |
| 33 | 16 | 0.945 | 0.925 | 0.841 | 0.930 | 0.832 / 0.776 / 0.860 |

表中前四列是保留比例（增益之比）。标签相对打乱只多 0.002–0.004 的 query R²（B=33 p=16 的 H−shuf = +0.004；B=8 p=16 为 +0.002）。H 相对随机子空间仍高一点（B=33 p=16 为 +0.025，target-cluster CI [0.014, 0.037]），但相对校准 bin 上的 top-r PCA 没有稳定优势（H−PCA = +0.005，区间含 0）。更新方向落在输入的主子空间里。这个诊断不能说明行为监督的矩统计有什么特别；P0-Solve 贴近 W0，和锚点很弱或矩匹配代理很弱是一致的，两者分不开。原先的「Frobenius 保留」是范数比 ‖PΔ‖/‖Δ‖，随机子空间的基线大约是 sqrt(r/N)；现在报告能量分数 ‖PΔ‖²/‖Δ‖²。

**λ / ε。** 第一轮网格没有 10³，λ 落在 10² 或网格上沿 10⁴；绝对 ε 相对 trace(G)/p 可以忽略。加宽后的网格是 λ∈{10⁻³…10⁶}（含 10³、10⁵、10⁶）和 ε = ε_rel·trace(G)/p，ε_rel∈{10⁻⁴,10⁻²,10⁻¹,1,10}。重选仍只在原来的 source episode 上。month 每个 B 只有 4 个 episode（1 个 session × 2 seeds × 2 draws）。重选后 B=33 的 p=16：days 0.736→0.695，month 0.582→0.613，year 0.303→0.371，drift 0.244→0.307。方向不定，没有改变「solve 远小于 W_alpha」这个结论。当前 `cells.csv` 用的是重选后的配置。

**协议。** 校准 trial 是随机抽取再按记录顺序排好的子集，不是最早的 B 个 trial。归一化和 P0 用这 B 个 trial 全部，query 可以包含校准 trial 之前的 trial。这和 P1/P3/drift 的旧行一样，对 trial 顺序不是因果的。

**跨 session 的锚点。** month/year 的 leave-one-session-out R² 是 0.011（p=8）和 0.024（p=16），不是 0。days 的 LOO 是 0.000 / −0.001。session 内 R² 仍只有 0.012–0.043。

## 1. 方法

\(W_0\) 是 P0 表（P-prior+Solve 一行则是 electrode-aligned prior）。它作用在 B 个校准 trial 的归一化上：\(u=W_0(r-\mu)/\sigma\)，没有 bias。ridge-CCA 用的是修正后的实现（潜变量维数和 baseline-day 桥都取参考日的维数，不按 target 有效维数截断），并且只用 source 已经选出的 **raw、sorted-unit、ridge** 配置（`cca/units/raw/ridge`：days 为 go 对齐、10 个 phase、ridge 3、\(d=N\)；month/year 为 trial 对齐、20 个 phase、ridge 10、\(d=N\)）。sqrt 计数下的 CCA 不是这套 z-score 坐标里的线性表，不能和 \(W_0\) 相减，因此不进入 shrinkage。

把 CCA 的仿射映射 \((T,c,\mu_c,\sigma_c)\) 改写到同一套 \((\mu,\sigma)\) 之后，

\[
u=W_{\mathrm{CCA}}\frac{r-\mu}{\sigma}+b_{\mathrm{CCA}}
\]

对每一个 \(r\) 都等于原来的 CCA 映射。\(W_\alpha\) 对权重和 bias 一起插值（\(W_0\) 的 bias 为 0）。每个 episode 都做了 float64 恒等检查（校准 bin 上最大绝对误差 \(<10^{-6}\)）；第一次 target 解码还和原来的 `decode_affine` 比过整段输出，最大绝对误差为 0。

行为基 \(\phi\) 复用 ridge-CCA 的 8 个 reach-direction 标签（trial 的 `target_dir`，`condition_label`）。\(p=8\) 是这个 one-hot。\(p=16\) 再接上同一 one-hot 乘以 trial 内相位 \(\tau\in(-1,1)\)，也就是 CCA 条件平均轨迹的最低阶时间展开。无方向标签或非 eval-valid 的 bin 不进入 \(H,G\)，但仍进入 \((\mu,\sigma)\)。

Source 锚点 \(F\in\mathbb R^{K\times p}\)：在 6 个 trunk source session 上，用真正送进冻结 trunk 的潜变量（各 session 自己的表和 `source_statistics`）做 ridge，\(u\approx F\phi\)。ridge 在 source session 的 leave-one-session-out 上选，然后在全部 source bin 上重拟合。session 内 \(u\) 的 \(R^2\) 只有 days 0.012 / 0.028（\(p=8/16\)）、month 与 year（同一 trunk）0.017 / 0.043。跨 session 的 LOO \(R^2\)：month/year 为 0.011 / 0.024，days 为 0.000 / −0.001。方向标签几乎解释不了 trunk 潜变量的方差，后面 solve 选到很大的 \(\lambda\) 就是这个原因。

Target 累加的是原始坐标 \(\sum r\phi^\mathsf T,\sum r,\sum r^2,\sum\phi,\sum\phi\phi^\mathsf T\)。提交时用当前 B-trial 的 \((\mu,\sigma)\)（与 `norm_stats` 逐元素一致，容差 \(10^{-5}\)）换成 \(H,G\)，再解

\[
W^\star=W_0+(FG-W_0 H)(H^\mathsf T D H+\lambda G_\epsilon)^{-1}H^\mathsf T D,
\]

\(G_\epsilon=G+\epsilon I\)，\(D=I\)。解的是线性方程，不显式求逆。\(\lambda\to\infty\) 回到 \(W_0\)（有测试）。

## 2. 协议

- 候选行 target 端无反传。ui_reg 只在投影诊断里重算表，并标明 reference / diagnostic。
- 超参只在 source simulation 上选，episode 与 P3 相同：`draw_sim` 的 tag `p3sel-s{seed}`、每个 seed 2 个 draw、只使用前 50,000 bin 内的 trial。集合是 trunk 没见过、且早于该 stratum 最早 target 的 session：days = 10 个 2014 session（`oosc`）；month = 只有 20140307；year = 0307/0627/0929/1203（都早于 20150511）。Drift 的 12 个 target **不重新选择**，沿用 month 的规则。
- \(\alpha\in\{0,0.25,0.5,0.75,1\}\)，\(\lambda\in\{10^{-3},\ldots,10^{4}\}\)（7 档），\(\epsilon\in\{10^{-8},10^{-4},10^{-2},1\}\)。选出的 \(\alpha\)：days 从 B=1 到 33 为 0.25/0.25/0.5/0.75/0.75/0.75；month 为 0/0/0.25/0.25/0.5/0.5；year 为 0/0.25/0.5/0.5/0.75/0.75。P0-Solve 多数 B 选到 \(\lambda=10^{2}\) 或 \(10^{4}\)，也就是贴近 \(W_0\) 的那一角。
- Target draw 用 `draw_trials`（与 P1/P3/drift 同一个 RNG）。B∈{1,2,4,8,16,33}，8 draws，P0 的 2 个 seed。整段连续因果解码，query \(R^2\) 是物理单位、只在非校准 trial 的 valid bin 上。这些子集是随机抽的再按时间排序，不是最早的 B 个 trial；归一化和 P0 用全部 B 个 trial，query 可以落在校准 trial 之前。与 P1/P3/drift 相同，对 trial 顺序非因果。8,640 行候选的 `h4_query_disjoint` 全部为真，`backprop` 全部为假。
- month 的 λ/ε（含审查后加宽的那一格）每个 B 只有 4 个 source episode：20140307 × 2 seeds × 2 draws。
- 不用 final session（0623/0625/0626），20140307 只做 month/year 的 source，不做 target。
- 重算的 P0 与 drift-curve 存档的 1,152 个格子逐格相同（最大绝对差 0）。重算的 P0+ui_reg 与存档的 \(R^2\) 也逐格相同（drift 1,152 格、year 96 格，最大绝对差 0）。ridge-CCA 这一行与已发表 baseline 表里的 raw sorted-unit ridge CCA 一致（例如 month B=1/8/33 为 0.060/0.512/0.656，days 为 −0.026/0.609/0.727）。
- R3 与 P0+ui_reg 的 \(R^2\) 复用协议相同的存档，不重训。days 的 R3 是 P3 days-IS（days 的 headline 规则）。month 与 drift 的 R3 / ui_reg 是 drift curve 的 month-`oosc` 行。year 的 ui_reg 是 `rule_from=month`、tag `oos` 的 like-for-like 行（四 session）；year 的 R3 是 P3 year-OOS。days 没有 like-for-like 的 P0 起步 ui_reg（P3 明确没做），days 的参照是 P2 的 `p0_ui_reg`，投影诊断也标明 `like_for_like=false`。
- month 核心 target 20140626 不是另一次 GPU 评测：它就是 drift 评测里的这一天（同一 month trunk、同一 month 规则）。year 的 20150511 在 drift 里用的是 month 规则，在 year stratum 里用的是 year 规则，两套都保留。
- CI：drift（12 个 target）和 days（2 个 target）是对 target 均值做 cluster bootstrap。days 只有 2 个 target，这个区间主要反映两个 target 之间的差别，不代替 draw 方差；draw-cluster 区间在 `paired.csv` 的 `draw_ci_*`。month 和 year 各只有 1 个 target，没有 target-cluster 区间，下面引用的是 draw-cluster 95% CI。12 个 drift target 共享一个 trunk 和一套规则，跨 target 的 CI 不是独立重复。

## 3. Headline（query \(R^2\)，B=1/8/33；第一轮网格）

均值。drift / days 括号里是 target-cluster 95% CI。month / year 是单 target，不写 target CI。P0-Solve 列是第一轮 λ/ε 网格；加宽网格之后的 B=33 变化见文首。自适应 \(W_\alpha\) 在 §5，不在这张表里。

### days（days trunk，0616/0617；\(\alpha\) 在 2014 OOS 上选）

| 方法 | B=1 | B=8 | B=33 |
|---|---:|---:|---:|
| P0 | 0.647 [0.639, 0.656] | 0.723 [0.717, 0.730] | 0.738 [0.735, 0.740] |
| W_alpha | 0.628 [0.618, 0.638] | 0.639 [0.631, 0.647] | 0.716 [0.715, 0.718] |
| P0-Solve p=8 | 0.647 | 0.724 | 0.738 |
| P0-Solve p=16 | 0.633 [0.613, 0.653] | 0.725 [0.720, 0.730] | 0.736 [0.735, 0.737] |
| ridge-CCA | −0.026 | 0.609 | 0.727 |
| R3 | 0.651 | 0.728 | 0.747 |
| P0+ui_reg（P2 规则，非 like-for-like） | 0.654 | 0.720 | 0.746 |

\(W_\alpha-\mathrm{P0}\)：B=8 为 −0.085，draw CI [−0.105, −0.068]，不含 0；B=33 为 −0.021，draw CI [−0.028, −0.015]。B=1 的 target CI 不含 0，但 draw CI [−0.038, 0.001] 含 0。P0-Solve p=8 与 P0 的差在 \(10^{-3}\) 量级（\(\lambda=10^{4}\)）。在 1–2 天的间隔上，用一年尺度的 OOS session 选出的 \(\alpha\) 过大，shrinkage 伤到了已经很好的 P0。这和 P3 已记录的「选择用的漂移量必须和 target 间隔相当」是同一件事。

### month（20140626，month trunk，规则来自 20140307）

| 方法 | B=1 | B=8 | B=33 |
|---|---:|---:|---:|
| P0 | 0.466 | 0.565 | 0.566 |
| W_alpha | 0.466 | 0.613 | 0.684 |
| P0-Solve p=8 | 0.466 | 0.567 | 0.604 |
| P0-Solve p=16 | 0.468 | 0.568 | 0.582 |
| ridge-CCA | 0.060 | 0.512 | 0.656 |
| R3 | 0.492 | 0.630 | 0.679 |
| P0+ui_reg（like-for-like） | 0.489 | 0.623 | 0.731 |

\(W_\alpha-\mathrm{P0}\) 的 draw CI：B=8 +0.048 [0.039, 0.057]，B=33 +0.117 [0.105, 0.129]，都不含 0。B=33 时 \(W_\alpha\) 高于 ridge-CCA（CCA\(-\)W_alpha = −0.028，draw CI [−0.033, −0.022]），与 R3 持平（R3\(-\)W_alpha = −0.004，draw CI [−0.015, 0.006]），仍低于 ui_reg 0.048（draw CI [0.035, 0.059]）。P0-Solve p=8 在 B=33 只比 P0 高 0.037 [0.030, 0.046]，低于 \(W_\alpha\)。

### year（20150511，与 month 同一 trunk；规则来自四个 OOS session）

| 方法 | B=1 | B=8 | B=33 |
|---|---:|---:|---:|
| P0 | 0.198 | 0.256 | 0.315 |
| W_alpha | 0.198 | 0.321 | 0.465 |
| P0-Solve p=8 | 0.198 | 0.257 | 0.308 |
| P0-Solve p=16 | 0.198 | 0.253 | 0.303 |
| ridge-CCA | 0.023 | 0.289 | 0.439 |
| R3 | 0.219 | 0.334 | 0.435 |
| P0+ui_reg（like-for-like） | 0.230 | 0.348 | 0.492 |

\(W_\alpha-\mathrm{P0}\)：B=8 +0.065 [0.046, 0.084]，B=33 +0.150 [0.129, 0.172]。B=33 的 \(W_\alpha\) 高于 ridge-CCA 0.027 和 R3 0.030（draw CI 都不含 0），低于 ui_reg 0.027 [0.020, 0.036]。P0-Solve 没有超过 P0（p=16 相对 P0 为 −0.013，draw CI [−0.034, 0.005]，含 0）。

### drift（12 个 target，month trunk，month 规则；均值 [target-cluster 95% CI]）

| 方法 | B=1 | B=8 | B=33 |
|---|---:|---:|---:|
| P0 | 0.153 [0.102, 0.221] | 0.212 [0.149, 0.294] | 0.211 [0.144, 0.296] |
| W_alpha | 0.153 [0.104, 0.223] | 0.285 [0.219, 0.363] | 0.420 [0.354, 0.486] |
| P0-Solve p=8 | 0.153 | 0.217 [0.152, 0.295] | 0.276 [0.217, 0.347] |
| P0-Solve p=16 | 0.158 [0.108, 0.225] | 0.220 [0.156, 0.298] | 0.244 [0.175, 0.323] |
| ridge-CCA | −0.008 [−0.032, 0.018] | 0.319 [0.263, 0.376] | 0.432 [0.368, 0.494] |
| R3 | 0.165 [0.113, 0.236] | 0.348 [0.288, 0.416] | 0.412 [0.347, 0.478] |
| P0+ui_reg | 0.174 [0.121, 0.242] | 0.310 [0.250, 0.382] | 0.451 [0.381, 0.522] |

配对差（target-cluster CI）：B=33 时 \(W_\alpha-\mathrm{P0}=+0.209\) [0.171, 0.244]；P0-Solve p=8 只有 +0.065 [0.037, 0.094]，p=16 为 +0.033 [0.023, 0.046]。同一 B 上 ridge-CCA\(-\)W_alpha = +0.012 [−0.002, 0.026]（含 0），R3\(-\)W_alpha = −0.008 [−0.023, 0.004]（含 0），ui_reg 仍高 0.030 [0.021, 0.039]。B=8 时 \(W_\alpha\) 高于 P0 0.073 [0.059, 0.087]，但低于 ridge-CCA 0.034（target CI [−0.006, 0.074]，含 0；draw CI [0.023, 0.044] 不含 0，所以「低于 CCA」只在 draw 层面稳）。B=1 的 \(\alpha=0\)，\(W_\alpha\) 就是 P0；此时 ridge-CCA 比 P0 低 0.161 [−0.215, −0.118]。

Seed spread（两个 P0 seed 的均值之差的绝对值）：P0 在 days 约 0.015–0.039，month 约 0.008–0.028，year 约 0.018–0.028，drift 约 0.021–0.023。P-prior 与 prior+Solve 是 seed-free，spread 为 0。

P-prior+Solve 把起点从 P0 换成 prior，用来分开「P0 表」和「求解器」。month B=33：prior 0.333，prior+Solve p=16 0.497，仍低于 P0 的 0.566。year B=33：0.150 → 0.254，P0 是 0.315。days 上 prior+Solve 没有超过 prior（B=33：0.361 → 0.305）。求解器在漂移大的 stratum 上能把一张弱 prior 往上推，但推不到 P0；days 上连 prior 都推不过。P0 的贡献不是求解器可以替代的。

## 4. 投影诊断（不是候选；结论以文首为准）

\(\Delta W_g=W_{\mathrm{P0+ui\_reg}}-W_0\) 投影到 \(H^\mathsf T\) 的行空间（\(D=I\)）。保留比例 = 投影后的 query 增益 / ui_reg 增益。能量分数是 \(\|P\Delta\|^2/\|\Delta\|^2\)，不是范数比。

Drift、B=33、p=16（审查对照，同一秩）：H 的保留比例 0.945，但随机子空间已经有 0.841，校准数据的 top-r PCA 有 0.925，打乱 trial 方向标签有 0.930。能量分数分别是 0.83 / 0.78 / 0.86 / 0.83。N≤17 时 rank/N≈0.96，投影几乎是恒等映射；20141203（N=10）上 W_diag 等于 W_ui。B=8、p=16：H 0.863，random 0.613，PCA 0.822，打乱标签 0.840。标签带来的 query 增益只有 0.002–0.004。

所以：更新方向在输入的主子空间里。不能据此说行为监督的 \(H\) 提供了 PCA 或标签打乱所没有的方向。P0-Solve 贴近 \(W_0\) 与「锚点弱」或「矩匹配代理弱」都相容，这个诊断分不开二者。逐 target 的 N、rank(H) 和四个对照在 `projection_controls.csv`。

## 5. 按间隔和可靠性调整的 \(W_\alpha\)

旧的 \(\alpha_B\) 在 days 上是用 2014 年、离 days trunk 很远的 OOS 选的，B=8 的 \(\alpha=0.75\)，query 比 P0 低 0.085。部署时知道的是「距离上一次 trunk/校准 session 过了多少天」，以及 B 个校准 trial 上 CCA 自己的可靠性。三行都只在 source 上拟合，target query 不参与。

- **w_alpha_gap**：按 \(|\)session 日期 − 该 stratum 最晚 trunk source\(|\) 分箱，≤14 天、15–180 天、>180 天，每个 (B, 箱) 单独选 \(\alpha\)。箱子没有 episode 就退到最近的有数据的箱子，并记在 `gap_rules_*.json` 的 `fallbacks`。days 三个箱子都有数据（近端是 trunk 里相距 ≤14 天的 source session，远端是 2014 OOS）。month 没有 >180 天的因果 source，退到 15–180 天。Drift 沿用 month 的表，所以 20140929 及更晚的 target（>180 天）也退到 15–180。
- **w_alpha_rel**：\(\alpha=\mathrm{logistic}(a+bc)\)，\(c\) 是这 B 个 trial 上 CCA 奇异值最大的 3 个的平均，\(b\ge 0\)。\((a,b)\) 在 source 上网格搜索，分数用最近的网格 \(\alpha\) 的已有 \(R^2\)。
- **w_alpha_both**：\(\alpha=\mathrm{clip}(\alpha_{\mathrm{bin}}\cdot\mathrm{logistic}(c),0,1)\)。箱子给出上限，可靠性差就把 \(\alpha\) 往下拉。

days 的 ≤14 天箱子在 B=1/8/33 选出的 \(\alpha\) 都是 0。year 的 >180 天箱子在 B=8/33 是 0.75，和旧的 year \(\alpha_{33}\) 相同。days 上 logistic 的斜率被选成 0（不比常数 \(\alpha\) 更好）；both 因为上限是 0，在 days 上等于 P0。

query \(R^2\)，B=1/8/33。days / drift 的括号是 target-cluster 95% CI。

| stratum | P0 | 旧 W_alpha | gap | rel | both | ridge-CCA | R3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| days | 0.647 / 0.723 / 0.738 | 0.628 / 0.639 / 0.716 | 0.647 / 0.723 / 0.738 | 0.622 / 0.708 / 0.721 | 0.647 / 0.723 / 0.738 | −0.026 / 0.609 / 0.727 | 0.651 / 0.728 / 0.747 |
| month | 0.466 / 0.565 / 0.566 | 0.466 / 0.613 / 0.684 | 0.442 / 0.613 / 0.647 | 0.466 / 0.597 / 0.628 | 0.461 / 0.603 / 0.634 | 0.060 / 0.512 / 0.656 | 0.492 / 0.630 / 0.679 |
| year | 0.198 / 0.256 / 0.315 | 0.198 / 0.321 / 0.465 | 0.198 / 0.320 / 0.465 | 0.186 / 0.299 / 0.419 | 0.198 / 0.323 / 0.466 | 0.023 / 0.289 / 0.439 | 0.219 / 0.334 / 0.435 |
| drift | 0.153 / 0.212 / 0.211 | 0.153 / 0.285 / 0.420 | 0.151 / 0.285 / 0.335 | 0.159 / 0.251 / 0.290 | 0.158 / 0.261 / 0.304 | −0.008 / 0.319 / 0.432 | 0.165 / 0.348 / 0.412 |

gap 相对 P0：days 三个 B 都是 +0.000（draw CI 退化成 0，因为 \(\alpha=0\)，表就是 P0）。旧 W_alpha 在 days B=8 是 −0.085，draw CI [−0.104, −0.068]。month B=8 的 gap 与旧 W_alpha 同为 +0.048 [0.039, 0.057]；B=33 的 gap 是 +0.081 [0.073, 0.088]，旧 W_alpha 是 +0.117，gap 少了 0.036（draw CI [−0.042, −0.031]，不含 0）。year B=8/33 的 gap 与旧 W_alpha 之差是 −0.001 / +0.000，区间含 0 或为 0。drift B=8 的 gap 与旧 W_alpha 相同（+0.073）；B=33 的 gap 是 +0.124 [0.103, 0.144]，仍高于 P0，但比旧 W_alpha 的 +0.209 少 0.085，因为 >180 天的 target 退回到了 15–180 天的 \(\alpha=0.25\)，而旧规则在 B=33 用的是 0.5。

rel 保不住 days：B=8 比 P0 低 0.016，draw CI [−0.023, −0.008]。它在 year/drift 的 B=33 上也低于 gap。both 在 days 上等于 P0，在大漂移上介于 gap 和 rel 之间，没有同时赢过二者。按间隔分箱的 \(\alpha\) 是这三行里唯一满足「days 保持 P0（损失 ≤ 0.01），month/year 的 B=8 和 year 的 B=33 保持旧增益」的一行；drift 的 B=33 只保持了一部分，缺的是 month 因果 source 里没有 >180 天的 episode。

## 6. 代价（解析，不是定点仿真）

MAC 只计乘法。\(G\) 按满的 \(p^2\) 计；对称实现可以只做 \(p(p+1)/2\)。提交代价不含 \(p^3/3\) 量级的求解 flop（另列）。累加器是原始坐标那一组，FP32，并给出双缓冲。

| N | p | 单缓冲 | 双缓冲 | 每 bin MAC（H / 满 G / \(r^2\)） | 每次提交 MAC | 求解 flop | 求解规模 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 26 | 8 | 1.3 KiB | 2.6 KiB | 298（208 / 64 / 26） | 6.7e3 | 1.8e3 | 8×8 |
| 26 | 16 | 2.9 KiB | 5.8 KiB | 698（416 / 256 / 26） | 1.8e4 | 8.0e3 | 16×16 |
| 256 | 8 | 10.3 KiB | 20.6 KiB | 2368（2048 / 64 / 256） | 6.0e4 | 1.7e4 | 8×8 |
| 256 | 16 | 19.1 KiB | 38.1 KiB | 4608（4096 / 256 / 256） | 1.5e5 | 6.7e4 | 16×16 |

N=256、p=16 时每 bin 更新 \(H\) 是 4096 MAC，与设计文档里的例子一致；一次提交约 0.15 M MAC，落在文档举例的 0.1–0.2 M MAC 里。这不含 trunk，也不含 P0 encoder。

## 7. 没有做的

- \(D\) 的 2–3 组 source 固定尺度（覆盖 / 未覆盖电极）。求解器接受对角 \(D\)，有测试，但没有进选择网格，也没有 target 行。报告的求解全部是 \(D=I\)。
- 设计文档的第三步（把 \(F,\lambda\) 放进 source 端的 episodic 训练）。\(F\) 是一次闭式 ridge，不是学出来的。
- sqrt 计数的 ridge-CCA。它不在 \(W_0\) 的坐标里。报告的 ridge-CCA 行是 raw sorted-unit 的修正版。
- 累加器的定点、饱和仿真。这里只给解析代价。
- days 的 like-for-like P0 起步 ui_reg 重选。沿用 P2 规则，并在诊断行里标明。
- 按电极分组之外的任何逐 unit 置信度网络。

## 8. 代码

`ssm_decode/p0solve.py`，`scripts/run_p0solve.py`，`scripts/run_p0solve_adapt.py`，`scripts/analyze_p0solve.py`，`tests/test_p0solve.py`（闭式解、\(\lambda\to\infty\)、原始坐标累加、CCA 坐标系、投影、target 端 `no_grad`，以及分箱回退、单调 logistic、相对 \(\epsilon\) 的尺度）。投影对照的原始 jsonl 是审查脚本 `/tmp/p0s_review/proj_control.py` 的输出，汇总在 `projection_controls.csv`。
