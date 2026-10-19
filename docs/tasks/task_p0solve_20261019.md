# Task: P0-Solve and shrinkage baseline (DAC2027, 2026-10-19)

You are a worker agent. The coordinator (Claude) will review your code and results. Work in `/home/xinyuan/Work_host/SSM` (git repo, branch `main`). Read `/home/xinyuan/Work_host/AGENTS.md` (project rules) and `SSM/docs/feasibility_p1_20261015.md` §0 (constraints and protocol rules 7–9) first.

## Background (read these files)
- Design source: `SSM/docs/DAC2027_adaptation_research_latest.md` §A1 (steps 1–3 and the projection diagnostic).
- P0 (fixed-weight per-unit encoder + electrode-aligned prior): `SSM/ssm_decode/population_calib.py`, `SSM/scripts/run_population_calib.py`, results `SSM/results/analysis/population_calib_20261015/README.md` (incl. §0.1 audit).
- Gradient reference ui_reg and P0+ui_reg: `SSM/ssm_decode/ui_reg.py`, `SSM/scripts/run_pc2_uireg.py`, `SSM/scripts/run_pc3_uiref.py`, `SSM/results/analysis/population_calib_p3_20261016/README.md` §R.12.
- ridge-CCA and other published baselines: `SSM/ssm_decode/published_baselines.py`, `SSM/scripts/run_baselines_readin.py`, `SSM/results/analysis/baselines_20261016/README.md` (two review passes; use the corrected code: latent dimension and bridge at the reference dimension).
- Drift curve (12 targets, month trunk): `SSM/scripts/run_drift_curve.py`, `SSM/results/analysis/drift_curve_20261016/README.md`.
- APST is MZ's unpublished work: do not use or mention it.

## What to build (new files; do not change the semantics of existing modules)
New module `SSM/ssm_decode/p0solve.py`, driver `SSM/scripts/run_p0solve.py`, analysis `SSM/scripts/analyze_p0solve.py`, tests `SSM/tests/test_p0solve.py`.

1. **W_alpha shrinkage baseline**: W_alpha = W0 + alpha_B (W_CCA − W0), where W0 = P0 table and W_CCA = the corrected ridge-CCA table mapped into the SAME trunk input coordinates and the SAME normalization convention as W0 (verify this explicitly with a test). alpha_B in {0, 0.25, 0.5, 0.75, 1} selected per B on SOURCE simulation only (the same source-simulation episodes used for P3 rules; trunk-unseen and earlier than the target where available).
2. **P0-Solve** (closed form; target side = streaming accumulation + one small solve, no backprop):
   - phi_t = phi(y_t) in R^p: reuse the behavior basis that ridge-CCA uses (reach-direction labels / behavior features); start with p = 8 and 16.
   - Fit F (K x p) on SOURCE: u_t^src ≈ F phi_t, where u_t^src are the actual latents fed to the frozen trunk (source tables), ridge regression.
   - Target accumulates H = sum_t x_t phi_t^T (N x p) and G = sum_t phi_t phi_t^T (p x p) over the B calibration trials only. Accumulate in RAW input coordinates (sum r_t phi_t^T, sum r_t, sum phi_t, sum r_t^2) and convert to the current normalization at commit time (see the design doc §B1.1).
   - Solve W* = W0 + (F G − W0 H)(H^T D H + lambda G_eps)^{-1} H^T D with G_eps = G + eps I, D = I first. Solve a linear system (no explicit inverse). lambda and eps per B selected on source only.
   - Variant: P-prior + the same solver (W0 = prior table) to separate the P0 contribution.
   - Optional only if time permits: D from 2–3 source-fixed groups (covered vs uncovered electrodes).
3. **Projection diagnostic** (diagnostic only, never a candidate): DeltaW_g = W_{P0+ui_reg} − W0 (use the like-for-like P0+ui_reg reference); W_diag = W0 + Proj_{row(H^T D)}(DeltaW_g). Report how much of the ui_reg gain survives the projection.

## Protocol
- Targets: sub-M core 4 (days 20150616/20150617 with the days AP backbone; month 20140626 and year 20150511 with the month/year AP backbone) AND the 12 drift-curve targets (month/year trunk). Never final sessions 20150623/0625/0626; never 20140307 as a target.
- B in {1, 2, 4, 8, 16, 33}, 8 draws, the SAME calibration draws as P1/P3/drift-curve rows (reuse their draw RNG), 2 P0 seeds.
- Contiguous causal decoding over the whole session; query R² in physical units on query-trial bins only; normalization from the B calibration trials.
- Rows: P-prior, P0, W_alpha, P0-Solve (p=8, 16), P-prior+Solve, ridge-CCA (corrected), R3 (reuse stored rows where protocol-identical), P0+ui_reg (reference), W_diag (diagnostic).
- Report per stratum and per B: means, paired differences vs P0 and vs W_alpha with 95% CIs clustered by target session; seed spread; and the projection-diagnostic retained fraction.
- Cost (analytic): accumulator sizes for H, G (raw-coordinate form), solve size, MACs per bin and per commit, for N = 26 (000688) and N = 256.

## Rules
- Frozen trunk; no backprop on target for candidate rows (ui_reg rows are labeled references); only the B labeled trials (H4); all selection on source only; poor results are results; no placeholders.
- Tests: closed-form solution equals a direct least-squares solve on small random data; lambda → ∞ returns W0; raw-coordinate accumulation + commit-time conversion equals batch computation in normalized coordinates; W_CCA and W0 share coordinates; target-side no-grad guard.
- GPU: local `cuda:0` only (one RTX 3090; another worker may use it — at most 2 of your jobs at a time; check `nvidia-smi`). Do NOT use Colab. If the GPU disappears, stop and report.
- Env: `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_p0solve /home/xinyuan/miniconda3/envs/spint/bin/python`.
- Outputs: `SSM/results/p0solve_20261019/` (gitignored). Report: `SSM/results/analysis/p0solve_20261019/README.md` in Simplified Chinese prose (technical terms in English) + compact csv. Code and comments in English.
- Git: commit ONLY your own files with explicit `git add <paths>`; English commit messages; NEVER push; never touch `SSM/overleaf/`.

## Final message
Headline table (per stratum: P0, W_alpha, P0-Solve, ridge-CCA, R3, P0+ui_reg at B = 1/8/33), projection-diagnostic retained fraction, what was not done, file paths.
