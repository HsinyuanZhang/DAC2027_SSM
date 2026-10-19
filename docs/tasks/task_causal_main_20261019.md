# Task: deployment-faithful causal main table (DAC2027, 2026-10-19)

You are a worker agent; the coordinator (Claude) reviews your code and results. Work in `/home/xinyuan/Work_host/SSM` (git, `main`). Read `/home/xinyuan/Work_host/AGENTS.md` and `SSM/docs/feasibility_p1_20261015.md` §0 (constraints; rules 7–9) first. APST is MZ's unpublished work: do not use or mention it.

## Why
All existing rows (P1, P3, drift curve, P0-Solve, W_alpha, stream_f7) use random calibration subsets with normalization and P0 computed from all B trials, and the query includes trials that occur BEFORE the calibration trials. The deployment story is a power-on calibration block followed by use: calibrate on the FIRST B usable trials of the session, then decode the rest, with statistics accumulated as prefixes and tables committed only when they are actually computed. The paper's main table must use this protocol.

## Protocol ("causal-first-B", primary)
- Calibration trials = the first B usable trials of the target session in recorded order; query = all later usable trials (scored on their bins only). Decode the WHOLE session contiguously and causally (no state resets), using at each time the table that is committed at that time:
  - before the end of calibration trial 1: the method's initial table (prior, or P0 from the trial-0 staging where the method allows it — document per method);
  - after each calibration trial k ends: the table the method can compute from trials ≤ k (anytime commit), effective from the next bin after a configurable commit latency (default: the start of trial k+1; also report latency = 1 trial);
  - after the calibration block: the final table, fixed.
- Report query R² (physical units) per target and B ∈ {1, 2, 4, 8, 16, 33}; since the calibration set is deterministic, variability comes from training seeds (2 P0 seeds) and targets — CIs clustered by target session; also report per-backbone results.
- Secondary (for continuity): the existing random-draw protocol rows (reuse stored rows), clearly labelled.

## Methods (reuse existing code paths; do not change their semantics)
1. P-prior (electrode-aligned prior + prefix normalization).
2. P0 (prefix statistics; final = P0 of trials 1..B).
3. W_alpha-gap (from `ssm_decode/p0solve.py` / `scripts/run_p0solve.py`: gap-binned alpha selected on source; ridge-CCA table computed from the prefix of calibration trials).
4. S1-causal re-based (from `ssm_decode/stream_f7.py` / `scripts/run_stream_f7.py`, commit e7af2d8 semantics; pre-registered OR gate as primary; ALSO report the strict gate "margin ≥ 0.005 AND > 2 SE" as a sensitivity row).
5. Combination: W_alpha-gap as the base table, then S1-causal steps on top (define clearly; select η/ρ for the combination on source only with the same gate).
6. R1-stream (P3 causal precedent `stream_stats`) on top of P0.
7. References (non-deployable or replay-based, labelled): R2 and R3 replaying the first B trials; ridge-CCA alone; P0+ui_reg (like-for-like rule). These use the same first-B calibration set and the same commit timing (final table after the block).

## Targets
- sub-M: days (20150616, 20150617; days backbone), month 20140626 and year 20150511 (month/year backbone), the 12 drift-curve targets (month/year backbone).
- sub-C month: round-1 backbones sc_0629 and sc_0708 and their targets (`SSM/results/expansion_round1/x/`; sub-C NWB is local; reuse round-1 J̄ and selection files where protocol-compatible, else recompute on trunk source sessions only).
- Never final sessions 20150623/0625/0626; never 20140307 as a target; respect the sub-C forbidden-session guard.

## Deliverables
- Code: `SSM/ssm_decode/causal_protocol.py`, `SSM/scripts/run_causal_main.py`, `SSM/scripts/analyze_causal_main.py`, tests `SSM/tests/test_causal_main.py` (tests: the calibration set is the first B usable trials; no statistic or table used at time t depends on any bin after t except the documented one-trial staging; committed tables switch at the documented times; the final table equals the method's offline output on the first B trials; latency option).
- Outputs: `SSM/results/causal_main_20261019/` (gitignored). Report: `SSM/results/analysis/causal_main_20261019/README.md` in Simplified Chinese prose (technical terms in English) with the main table (rows = methods, columns = stratum × B), per-target table, per-backbone table, paired differences vs P0 and vs W_alpha-gap with CIs clustered by target session, the gate sensitivity, commit-latency sensitivity, and a short comparison with the random-draw protocol (how much the protocol change moves each method).
- Also a compact cost column per method (MAC per bin, MAC per commit, SRAM and ROM separately, replay yes/no) taken from the reviewed cost models (p0solve, stream_f7, refine_cost), with N = 26 and N = 256.

## Rules
- Frozen trunk; no target backprop in candidate rows; only the B labeled calibration trials (H4); all selection on source only (reuse existing source selections where the protocol is compatible; if a selection used non-causal source simulation, either re-run it causally on source or mark it).
- Local GPU `cuda:0` only (one RTX 3090; ≤3 jobs; check `nvidia-smi`); no Colab. If the GPU disappears, stop and report.
- Env: `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_causal /home/xinyuan/miniconda3/envs/spint/bin/python`.
- Git: commit ONLY your own files with explicit `git add <paths>`; English commit messages; NEVER push; never touch `SSM/overleaf/`.

## Final message
Main table (P-prior, P0, W_alpha-gap, S1-causal, combination, R1-stream; references R2/R3/ridge-CCA/P0+ui_reg) at B = 1/8/33 per stratum; gate and latency sensitivity; protocol-change deltas; what was not done; file paths.
