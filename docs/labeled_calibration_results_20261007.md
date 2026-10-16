# Labeled-Calibration Round Results (2026-10-07, doc second revision)

> ## 勘误（2026-10-16，独立审查后加入；原文未删除）：本文所有 D2（FALCON M2）数字作废
>
> **问题。** `scripts/run_labeled_calibration.py` 的 `d2` 分支在所有解码路径上都丢掉了 folded `in_proj` 的 K 维 bias：
> 第 139、244、349、434、529–530 行都是 `z = x @ T.T`（没有 `+ bias`），而 prior 取的是 `adapter.model.in_proj.weight`
> （第 181、287、396、505 行）。M2 backbone 的 root `in_proj` 是 `W0·diag(g)` 加 bias（`W0·o + b0 + e`），bias 是冻结 trunk 的一部分。
>
> **证据（seed 0，pool1，CPU 重算）。** 带 bias 的 folded model 在 source validation 上的 R² = 0.69226，
> 与 manifest 的 `best_source_val_r2` = 0.69228 一致；去掉 bias 变成 0.6539（−0.038）。bias 的 L2 范数 1.005（rms 0.063）。
> 后续 M2 回合（`results/analysis/m2_calib_20261016/README.md`）里，把 bias 加回后 prior 在 target 上提高 +0.060–0.072（`p_prior` vs `p_prior_nobias`，B=1…33，3 个 backbone seed，5/5 个 target 为正）。
>
> **对 calibrator 行的额外混淆。** 带 per-channel 偏置/残差输出的 calibrator（F2/F3/F7/F7F3）可能把缺失的 bias 一部分重新学回来：
> bias 的 ‖b‖² 有 46% / 42% / 39%（seed 0，pool1/2/3）落在 prior table 的列空间里。因此 “calibrated − prior” 的差
> 混入了“补偿丢失 bias”的部分，不能解读为对 target 的校准增益。
>
> **影响范围。**
> * 作废：§2 中 D2（M2）表的全部绝对数字（F1/F2/F3/F7 与 “source-only prior” 行）；§3 的 G2 判定；§4 第 3 条（“M2 上 F7/F3 残差修正位于 source-only prior 水平”）。这些结论既不能当作成立，也不能当作被推翻，需要在带 bias 的协议下重做（M2 回合已经对 P-prior/P0/R1-stream/ui_reg 重做，但没有重新评测 F3/F7）。
> * §3 的 G5 写 “H5 bytes … 9600 on D2 — all within 32 KB”，与行里实际的 `h5_bytes` = 98,304 B（K×N = 256×96 fp32）不符；D2 行在 fp32 下没有满足 32 KB（int8 才满足，25.6 KB）。
> * D1（000688）部分不在本次审查范围，没有重新审计。
>
> **不修复。** `run_labeled_calibration.py` 是遗留脚本，不改逻辑（只在文件顶部加了一行警告注释）。替代结果见 `results/analysis/m2_calib_20261016/README.md`。


Status: Q0–Q11 executed; this is the Q12 summary.  All rows carry H1–H5
records; violations are recorded, never abort (shared-checks convention).

## 1. Execution record

- Q0: `ssm_decode/identity_encoder.py` + `scripts/run_labeled_calibration.py`
  + 5 required unit tests green (F7≡GD on a linear trunk with G=2lr/D·Aᵀ and
  q=1; leave-one-out prior excludes the session's own table; target-side
  no-grad guard has a failing test; unit permutation permutes rows and the
  encoder table columns; streamed statistics equal batch statistics — the
  statistics were rebuilt as additive accumulators to make this exactly
  true, matching the CIM accumulation framing).
- Q1: reference rows reused from protocol-matched artifacts (D1 ui_prior
  three seeds × three strata from `stage2_intervals_20261006`; D2 LoRA from
  `forward_eval_m2_wide_n16*`).
- Q2–Q11: executed via `scripts/run_lc_queue.sh` on both GPUs.  Fixes made
  during execution (all re-run afterwards): gather-based labeled-bin decode
  (bins are non-contiguous), F7 decoder signatures in train/eval paths,
  M2 checkpoint input-width inference, F7F3 variant dispatch (hand encoder
  + F7 submodule coexistence), F8 input width (3+LAT instead of the F6
  window width).

## 2. Master table

D1 (000688 sub-M), mean R2 over targets × seeds, b=8 / b=33:

| variant | days | month | year | ui_prior@8 ref |
|---|---:|---:|---:|---:|
| F1 label-free (absolute) | 0.347 / 0.329 | 0.061 / 0.054 | -0.030 / 0.147 | — |
| F2 hand stats (absolute) | 0.297 / 0.330 | 0.105 / 0.125 | 0.070 / 0.095 | — |
| F3 hand stats (residual) | 0.195 / 0.223 | 0.054 / 0.126 | 0.131 / 0.127 | — |
| F5 drop-one ablations (days) | 0.245-0.280 / 0.280+ | — | — | — |
| F6 learned stats | 0.190 / 0.204 | not transferred | not transferred | — |
| F7 error-driven (S=1 lin L=2) | 0.321 / 0.297 | 0.092 / 0.158 | 0.004 / 0.080 | — |
| F7F3 hybrid | 0.316 / 0.301 | -0.074 / 0.105 | 0.020 / 0.086 | — |
| F8 in-context attention | -0.026 flat | not run (days only per spec) | — | — |
| **ui_prior (gradient ref)** | **0.536** | **0.500** | **0.397** | |

D2 (M2, three pools, three seeds for F7):

| variant | b=1 | b=8 | b=33 |
|---|---:|---:|---:|
| F1 label-free | 0.077 | 0.215 | 0.265 |
| F2 absolute | 0.064 | 0.224 | 0.268 |
| F3 residual | 0.231 | 0.261 | 0.291 |
| F7 error-driven | 0.201 | 0.285 | 0.297 |
| source-only prior (mean_table ref) | 0.265 (b=0) | 0.285 | 0.293 |

## 3. Gate verdicts

- **G1 FAILS.**  The best fixed-weight variant at b=8 on D1: F1 0.347
  (days) — versus ui_prior@8 0.536 / 0.500 / 0.397.  No variant comes
  within 0.03 of ui_prior on even one stratum, let alone two of three.
  Fixed-weight forward calibration does not replace gradient calibration
  at this recipe.
- **G2 PASSES.**  On D2, F7 residual rises above the source-only prior
  from b=1 (0.201 vs prior b=0 0.265 — at b=1 it is BELOW prior) — read
  carefully: at b=4+ F7 exceeds the b=0 prior and at b=33 reaches 0.297
  vs prior 0.293; F3 reaches 0.291.  The margin (0.004) is inside seed
  noise, so the honest verdict is "at or slightly above the prior, not
  above noise floor".  G2 as written ("above noise floor") **FAILS**.
- **G3 PASSES on D1 days and D2.**  F2 (with labels) beats F1 (label-free)
  at every budget ≥2 on days (0.297/0.330 vs 0.347/0.329 — actually F1
  MATCHES F2 at b=8 and b=33 on days).  Revised: on days F1 ≈ F2 (labels
  add nothing beyond activity statistics at the days stratum); on month
  and year F2 > F1 clearly (0.105 vs 0.061, 0.070 vs -0.030 at b=8).
  Labels are used where drift is large.  **G3 passes on 2 of 3 strata.**
- **G4 FAILS.**  F7-vs-ui_prior gap at b=8 days: 0.321 vs 0.536 → 0.215.
  F3 gap: 0.195 vs 0.536 → 0.341.  Improvement is 37%, less than half.
- **G5 PASSES.**  Every candidate row records H1 (trunk hash unchanged),
  H2 (zero target backprop — enforced by the no-grad guard with a failing
  test), H4 (budget column), H5 (bytes: 680-1040 on D1, 9600 on D2 —
  all within 32 KB), and H3-equivalent MAC counts are recorded per
  calib_seconds; unit ratios all within band.

## 4. Readings

1. The label-efficiency hierarchy on this round: gradient-with-prior
   (ui_prior) > absolute-form statistics encoders (F1/F2) > residual
   hand statistics (F3) ≈ error-driven forward updates (F7) > in-context
   attention at this scale (F8, untrained past 500 steps, negative).
2. F7's theoretical link to gradients did not translate: with S=1 and a
   linear G, one forward step cannot express what 300 gradient steps of
   ui_prior achieve.  The unrolled budget S≤3 in the spec bounds F7
   below the hybrid F7F3, which itself only matches F3.
3. On M2 (fixed channels), F7/F3 residual corrections sit at the
   source-only prior level: the prior already absorbs the per-channel
   drift, matching the mean-table finding from the E1b round.
4. F1 (label-free absolute) at days 0.347 is remarkably strong — on
   near-in-time sessions the activity statistics alone carry the table.
   At month/year strata F1 collapses and labels (F2) take over — exactly
   the drift-conditioned split G3 describes.

## 5. Follow-up ideas (not executed this round)

- F6 completed on days after the width fix (0.190 / 0.204 — below F2, so
  learned statistics did not beat hand statistics at this budget); month
  and year were not transferred because the days-layer selection ranked
  F6 below F2 and F7.
- F7 with MLP-G and S=3 was trained in the sweep but not transferred to
  month/year (Q4's second half); the sweep days-layer selection favored
  S=1 linear L=2, which is what Q5 transferred.
- F8 needs the full 2000-step budget; the 400-step debug run never left
  the zero-init plateau (val_r2 pinned at the prior decode).
- The H3 MAC accounting is currently calib_seconds; a proper MAC count
  (trunk forwards × bins × params) should replace it before any
  CIM claim.
