# Causal-first-B 主表（000688，2026-10-19）

## 审查后修正（第二轮）

上一轮的 pooled α 不能进正文。Year trunk 和 month trunk 的权重逐张相同。Year 标签的 episode 跑在 20140627 / 20140929 / 20141203 上，而这些日期本身就是 drift target：mid 箱只有 20140627 的两个 seed，等于自己选自己；gt180 的 24 条里有 10 条在 drift target 上，另外 14 条来自 days trunk，而那个 trunk 的训练集包含 20150511–20150615。le14 还混进了 sub-C 的 20150707，晚于每一个 sub-M target。旧的 `w_alpha_pooled` / `combo` 只留在 `rows.jsonl` 里，当作标明作废的附录，不进主表。

正文只用两套事先写明的协议。Target 自己的 gap 箱不变（相对它的 trunk 最后一天）。变的是哪些 episode 可以投票。

**(a) expanding-window，主表。** 部署时，更早的、已经标过的 session 可以当 source。一条 episode 能投票，当且仅当：episode 的 session 严格早于这个 target，并且跑它的 trunk 的全部训练 session 也严格早于这个 target。更早的 target 因此可以当 source。

**(b) strict target-disjoint，保守对照。** 在 (a) 上再去掉：任何一张已报告的 target session（两个 subject 都算），以及任何在已报告 target 上训练过的 trunk。

Drift 上和审查复算对齐（B=8/16/33）：(a) W_alpha 0.348 / 0.404 / 0.437，COMBO 0.381 / 0.406 / 0.471；(b) W_alpha 0.282 / 0.364 / 0.422，COMBO 0.336 / 0.383 / 0.465。逐格最大偏差在 B=4 的个别 seed（约 0.005），B=8/16/33 与审查的重解码一致。

η/ρ 没有按每张 target 重选。Drift 的 COMBO 用的是 month 规则，selection session 只有 20140307，早于全部 drift target，也不是已报告的 target，(a) 和 (b) 都能用。Days 的 η/ρ 是在 days trunk 的 2014 OOS 上选的；那个 trunk 见过 drift target，所以 (b) 下这组 η/ρ 本身并不 target-disjoint。Sub-C 的 η/ρ 来自各自的 held-out pool session。

**门。** AND 的「> 2 SE」不是 session 级检验。Selection session 只有一个时，SE 是两个 P0 seed 的差；CCA+S1 和 ridge-CCA 不依赖 seed，两条 episode 相同，SE = 0，AND 退化成「增益 ≥ 0.005」。这句话以前写重了。保守的另一道门：被选中的 (η, ρ) 至少在 **两个** selection session 上增益为正，否则退回基座。Drift、month、sub-C 的 S1 / COMBO / CCA+S1 都只有一个 selection session，这道门全部退回基座（S1→P0，COMBO→该协议的 W_alpha，CCA+S1→ridge-CCA）。Days 有 10 个 session，S1 和 COMBO 的 AND 选择在这道门下仍然部署。

**Year 不是独立 trunk。** 20150511 就是 drift 里的那一天、month trunk、同一套 (a)/(b) α。下面不再单列 year。

**Days 的间隙是倒着的。** Days trunk 的 OOS episode 全是 2014 年的 session，比 trunk（20150615）早 194–497 天。`gap_bin_name` 按绝对值分箱，所以它们进了 gt180。没有把「间隔 ≤ 14 天就 α=0」写成预注册规则：近间隔的 source 并不支持它。可以事后看，但不进主表。

**Sub-C 的 CCA 配置**就是 sub-M month 的 `CCA_CFG`（trial 对齐、20 phase、ridge 10），没有在 sub-C 上重选。只有一个 selection session，重选一整张 align/phase/ridge 网格仍然是单 session 的选择，这次没有做。

**Sub-C 的 α。** (a) 的 mid 箱是 sub-M 20140627（year 标签，两个 seed），跨 subject。它早于这些 sub-C target，trunk 也早于它们，所以 expanding-window 允许。 (b) 把它拿掉（它是已报告的 drift target），mid 退回 le14（20140307 和 sc_0629 的 20150319）。

`read_bin` 对 S1/COMBO/CCA+S1 和 R2/R3/ui_reg 是前缀缓存上的标签：记录这些函数被要求读哪些 trial。真正的守卫是扰动测试。测试改到 drift 20140929、B=8、J=4（α=0.75，S1/COMBO/CCA+S1 的 η 都 > 0）。Query 之后的 bin，以及前 4 个 trial 之后的 bin，都不改变还不能读它们的 commit；正对照确认后面的 commit 会动。`CAUSAL_LEAK=1` 下该测试通过。

CCA+S1 和 ridge-CCA 的 seed spread 按构造是 0。

## (a) expanding-window，drift（12 个 target，一个 month backbone）

Latency 0，query R²。括号里的对比是 target-cluster 区间，见 `paired.csv` 的旧列；下面 (a)(b) 的差用同一套 12 天直接相减。

| 方法 | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P-prior | 0.106 | 0.098 | 0.104 | 0.104 | 0.106 | 0.106 |
| P0 | 0.181 | 0.182 | 0.201 | 0.209 | 0.218 | 0.215 |
| ridge-CCA | −0.029 | 0.080 | 0.169 | 0.329 | 0.387 | 0.409 |
| W_alpha (a) | 0.181 | 0.186 | 0.224 | 0.348 | 0.404 | 0.437 |
| S1（AND） | 0.181 | 0.182 | 0.267 | 0.276 | 0.313 | 0.353 |
| COMBO (a) | 0.181 | 0.186 | 0.271 | 0.381 | 0.406 | 0.471 |
| CCA+S1（AND） | −0.029 | 0.080 | 0.195 | 0.343 | 0.427 | 0.450 |
| R2 | 0.200 | 0.227 | 0.273 | 0.346 | 0.396 | 0.406 |
| R3 | 0.200 | 0.227 | 0.274 | 0.347 | 0.396 | 0.411 |
| P0+ui_reg | 0.199 | 0.220 | 0.260 | 0.308 | 0.410 | 0.450 |

(a) 相对 ridge-CCA，B=8/16/33：W_alpha +0.019 / +0.017 / +0.028；COMBO +0.051 / +0.017 / +0.062；CCA+S1 +0.014 / +0.040 / +0.041。COMBO 相对 ui_reg：+0.073 / −0.004 / +0.021。B=16 的 COMBO 0.406 高于 R2 0.396，低于 ui_reg 0.410。

## (b) strict target-disjoint，同一 12 天

| 方法 | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| W_alpha (b) | 0.181 | 0.182 | 0.232 | 0.282 | 0.364 | 0.422 |
| COMBO (b) | 0.181 | 0.182 | 0.284 | 0.336 | 0.383 | 0.465 |

B=8/16/33：W_alpha − ridge-CCA = −0.047 / −0.023 / +0.013；COMBO − ridge-CCA = +0.006 / −0.004 / +0.056；COMBO − ui_reg = +0.028 / −0.028 / +0.015。保守池里 W_alpha 在 B=8 和 B=16 低于 ridge-CCA。COMBO 只在 B=33 稳定高于 ridge-CCA 和 ui_reg；B=16 略低于两者。

α 箱的组成（完整列表在 `select/alpha_protocols.json`）：

| target | 协议 | B=8 的 α | 用到的 session |
|---|---|---:|---|
| 20140626（mid） | (a) 和 (b) | 0.25 | 20140307（month/year 标签） |
| 20150511（gt180） | (a) | 0.75 | 20140929、20141203 |
| 20150511 | (b) | 0.25 | gt180 被拿掉，退回 le14：20140307、20150319 |
| 20150616，days trunk（le14） | (a)=(b) | 0.25 | 别的 trunk 上早于它的 le14 episode；days 自己的 OOS 全是倒向的 gt180，不进这个箱子 |

## 门的灵敏度，以及逐 target 的最差一格

保守门（≥2 个 session 增益为正）在 drift 上把 S1、COMBO、CCA+S1 全部退回基座。退回之后，20150512 B=8 的 CCA+S1 −0.180 不会被部署。

AND 门实际部署时，12 天里增益的最小值（负的个数）：

| B | S1 − P0 | COMBO(a) − W_alpha(a) | CCA+S1 − ridge-CCA |
|---|---|---|---|
| 1 | +0.000（0/12 负） | +0.000（0/12） | +0.000（0/12） |
| 2 | +0.000（0/12） | +0.000（0/12） | −0.001 @20150617（8/12 负，幅度约 0.001） |
| 4 | −0.063 @20150512（1/12） | −0.058 @20150512（2/12） | −0.213 @20150512（5/12） |
| 8 | +0.024 @20150612（0/12） | −0.002 @20150512（1/12） | −0.180 @20150512（3/12） |
| 16 | +0.016 @20150612（0/12） | −0.051 @20150612（4/12） | −0.007 @20141203（2/12） |
| 33 | +0.059 @20150612（0/12） | +0.012 @20150612（0/12） | +0.007 @20141203（0/12） |

S1 从 B=8 起 12 天都非负。COMBO 在 B=16 有 4 天低于自己的 W_alpha。CCA+S1 在 B=4 和 B=8 有明显的单日失败，B=33 才 12 天都非负。

## Days（2 个 target，自己的 trunk）

(a) 和 (b) 在这两天上选出的 α 相同。B=33 的 α=0.5，W_alpha 0.690，低于 P0 0.743。没有预注册「近间隔 α=0」。

| 方法 | 1 | 2 | 4 | 8 | 16 | 33 |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.679 | 0.703 | 0.722 | 0.728 | 0.734 | 0.743 |
| W_alpha (a) | 0.679 | 0.703 | 0.695 | 0.720 | 0.669 | 0.690 |
| S1 AND | 0.678 | 0.703 | 0.694 | 0.723 | 0.726 | 0.728 |
| COMBO (a) | 0.678 | 0.703 | 0.665 | 0.715 | 0.673 | 0.692 |
| ridge-CCA | 0.062 | 0.214 | 0.475 | 0.651 | 0.668 | 0.734 |

Days 的 S1/COMBO 有 10 个 selection session，保守门不退回。

## Sub-C month（各 6 个 target，各 1 个 selection session）

CCA 配置是 sub-M month 的，没有重选。α 在 (a) 上跨 subject，见上。没有 session 级 CI：一个 backbone 6 天，选择只有一个 session。下面是 6 天的均值。

| backbone | 方法 | 1 | 4 | 8 | 16 | 33 |
|---|---|---:|---:|---:|---:|---:|
| sc_0629 | P0 | 0.097 | 0.158 | 0.179 | 0.200 | 0.218 |
| sc_0629 | ridge-CCA | −0.052 | 0.081 | 0.172 | 0.242 | 0.301 |
| sc_0629 | W_alpha (a) | 0.097 | 0.151 | 0.212 | 0.281 | 0.357 |
| sc_0629 | COMBO (a) | 0.097 | 0.151 | 0.212 | 0.281 | 0.357 |
| sc_0629 | W_alpha (b) | 0.097 | 0.169 | 0.221 | 0.292 | 0.357 |
| sc_0629 | CCA+S1 − ridge | +0.028 | +0.068 | +0.096 | +0.049 | +0.082 |
| sc_0629 | 其中 ≤0 的天数 | 2/6 | 1/6 | 0/6 | 0/6 | 0/6 |
| sc_0708 | P0 | 0.307 | 0.407 | 0.428 | 0.429 | 0.421 |
| sc_0708 | ridge-CCA | 0.000 | 0.340 | 0.425 | 0.512 | 0.553 |
| sc_0708 | W_alpha (a) | 0.307 | 0.438 | 0.456 | 0.519 | 0.535 |
| sc_0708 | COMBO (a) | 0.296 | 0.438 | 0.456 | 0.519 | 0.535 |
| sc_0708 | W_alpha (b) | 0.307 | 0.441 | 0.464 | 0.509 | 0.535 |
| sc_0708 | CCA+S1 − ridge | +0.000 | −0.120 | +0.040 | +0.022 | +0.020 |
| sc_0708 | 其中 ≤0 的天数 | 6/6 | 4/6 | 0/6 | 0/6 | 2/6 |

sc_0629 在 B=8/16/33 上 6/6 为正。sc_0708 的 B=4 是 η=3 的失败（均值 −0.120，最差约 −0.30），B=33 均值只 +0.020，6 天里有 2 天 ≤ 0。保守门会把这两套 backbone 的 CCA+S1 全部退回 ridge-CCA，因为都只有一个 selection session。

sc_0629 的 COMBO (a) 与 W_alpha (a) 相同：AND 门退回了基座。(b) 拿掉 20140627 之后 α 更小，B=8 的 W_alpha 从 0.212 变成 0.221。sc_0708 的 B=1 COMBO 是 0.296，低于 W_alpha 0.307，因为那一格 AND 部署了 η=1。

## 没有做

- 按每张 target 重选 COMBO / CCA+S1 的 η/ρ。Drift 的 η/ρ 来自 20140307，对 (a)(b) 都合法；days 的 η/ρ 在 (b) 下不 target-disjoint，已标明。
- 在 sub-C 上重选 CCA 的 align / phase / ridge。配置沿用 sub-M month。
- 把「≤14 天 α=0」写成规则。
- 重跑不使用 α 的行（P0、S1、R1、R2、R3、ui_reg、ridge-CCA）。它们和上一轮相同。

代码：`ssm_decode/causal_protocol.py`（两套 α 过滤、`sessions_with_positive_gain`），`scripts/run_causal_main.py`（`alpha-protocols`，`w_alpha_a` / `combo_a` / `w_alpha_b` / `combo_b`，AND 行的 `cfg.gate` 不再抄成 `or`），`tests/test_causal_main.py`。协议文件 `results/causal_main_20261019/select/alpha_protocols.json`。
