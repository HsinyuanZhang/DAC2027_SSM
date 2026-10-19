# Task: replay-free streaming fixed-feedback refinement (F7-J rewrite) (DAC2027, 2026-10-19)

You are a worker agent. The coordinator (Claude) will review your code and results. Work in `/home/xinyuan/Work_host/SSM` (git repo, branch `main`). Read `/home/xinyuan/Work_host/AGENTS.md` and `SSM/docs/feasibility_p1_20261015.md` §0 (constraints; protocol rules 7–9) first.

## Background (read these files)
- Design source: `SSM/docs/DAC2027_adaptation_research_latest.md` §A2 (summation reordering; FIR single pass; recursive eligibility traces; the three boundaries; novelty risk: filtered-x LMS, eligibility traces).
- Current F7-J (R2) and R3: `SSM/ssm_decode/fwd_refine.py` (`fj_refine`, averaged Jacobian J̄, residual normalization, prior shrinkage), `SSM/scripts/run_pc3_refine.py` (selection, eval), `SSM/ssm_decode/refine_cost.py`; results `SSM/results/analysis/population_calib_p3_20261016/README.md` §R.12 (like-for-like causal rules; month causal selection set {20140307}).
- Drift curve (12 targets): `SSM/scripts/run_drift_curve.py`, `SSM/results/analysis/drift_curve_20261016/README.md`.
- sub-C round 1 (R2 gains on month): `SSM/results/analysis/expansion_20261016/round1_README.md`; queue code `SSM/scripts/run_expansion.py` (sub-C data is local; round-1 backbones/results are under `SSM/results/expansion_round1/x/`).
- Do not use or mention APST (MZ's unpublished work).

## What to build (new files; do not change semantics of existing modules)
`SSM/ssm_decode/stream_f7.py`, `SSM/scripts/run_stream_f7.py`, `SSM/scripts/analyze_stream_f7.py`, `SSM/tests/test_stream_f7.py`.

1. **Exactness check of the reordering**: for one fixed forward trajectory, fixed kernel J̄_ℓ (ℓ = 0..L−1) and identical boundary handling, show that U = Σ_t Σ_ℓ J̄_ℓᵀ e_t x_{t−ℓ}ᵀ computed "online" (current error × ring buffer of past inputs) equals the existing F7-J data-term update computed with stored residuals and the backward temporal convolution (match the sign/direction conventions of `fj_refine`, including residual normalization and trial-boundary padding). Unit test to ~1e-9 on random data and a numeric check on real data.
2. **S1-FIR (single pass)**: one streaming pass over the B calibration trials using the LIVE forward (no replay): at each bin, error e_t from the current table's prediction, update accumulator with the FIR ring buffer (L from the existing rules), apply the shrinkage-toward-P0 update at trial boundaries (anytime commit). Compare to R2 (S-step replay) with the same η/ρ rules, and with η re-selected on source for the single-pass form.
3. **Recursive traces (M = 4, 8)**: fit J̄_ℓ ≈ Σ_m ρ_m^ℓ C_m on SOURCE (stable |ρ_m| < 1; report fit error), traces r_{m,t} = ρ_m r_{m,t−1} + x_t, update U += Σ_m (C_mᵀ e_t) r_{m,t}ᵀ. Compare FIR vs M = 4/8.
4. Optional: R3-style readout ridge on top (R1-stream) for the single-pass variant.

## Protocol
- Strata/targets: sub-M month (20140626) and year (20150511) core targets; the 12 drift-curve targets (month/year trunk); sub-C month targets from round 1 (backbones sc_0629 and sc_0708; reuse their J̄ if saved, else compute on trunk source sessions only). Never final sessions 20150623/0625/0626; never 20140307 as a target.
- B in {1, 2, 4, 8, 16, 33}, 8 draws (same draws as the existing rows), 2 P0 seeds; contiguous causal decoding; query R² in physical units on query-trial bins; normalization from the B calibration trials.
- Rows: P0, R2 (replay, reuse stored where protocol-identical), R3, S1-FIR, S1-FIR+R1s, trace-M4, trace-M8. Selection of any new hyperparameter on SOURCE simulation only (same causal selection sets as P3).
- Report: per stratum/B means; paired diffs vs P0 and vs R2 with CIs clustered by target session; fraction of the R2 gain retained by S1-FIR and traces; kernel-fit errors.
- Cost (analytic, rule 8: count every buffer): per bin MACs and SRAM for FIR (L·N ring buffer) and traces (M·N), the K×N update accumulator, vs R2 replay buffers; for N = 26 and N = 256; note which MACs are weight×activation vs activation×activation.

## Rules
- Frozen trunk; no target backprop for candidate rows; only the B labeled trials (H4); selection on source only; poor results are results; no placeholders.
- Tests: reordering exactness; trace recursion equals the FIR it approximates when the kernel is exactly a sum of exponentials; streaming update never uses future errors beyond what the design allows (state the causality of each variant); target-side no-grad guard.
- GPU: local `cuda:0` only (one RTX 3090, shared with another worker — at most 2 of your jobs at a time; check `nvidia-smi`). Do NOT use Colab. If the GPU disappears, stop and report.
- Env: `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_streamf7 /home/xinyuan/miniconda3/envs/spint/bin/python`.
- Outputs: `SSM/results/stream_f7_20261019/` (gitignored). Report: `SSM/results/analysis/stream_f7_20261019/README.md` in Simplified Chinese prose (technical terms in English) + compact csv. Code and comments in English.
- Git: commit ONLY your own files with explicit `git add <paths>`; English commit messages; NEVER push; never touch `SSM/overleaf/`.

## Final message
Exactness result; headline table (P0, R2, R3, S1-FIR, traces at B = 1/8/33 per stratum); retained fraction of the R2 gain; cost table; what was not done; file paths.
