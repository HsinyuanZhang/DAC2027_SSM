# Task: FALCON M2 under the causal-first-B protocol with the frozen methods (2026-10-20)

You are a worker agent; the coordinator (Claude) reviews your work. Work in `/home/xinyuan/Work_host/SSM` (git, `main`). Read `/home/xinyuan/Work_host/AGENTS.md` and `SSM/docs/feasibility_p1_20261015.md` §0 first. APST is MZ's unpublished work: do not use or mention it.

## Goal
M2's 5 targets (all 1–7 days after their sources) belong to the paper's primary "≤ 30 days" tier, but their results so far use the random-draw protocol. Re-run M2 with the frozen methods under causal-first-B, exactly as implemented for 000688 in `SSM/ssm_decode/causal_protocol.py`, `SSM/scripts/run_causal_main.py` (commit 8f8bfef; README `SSM/results/analysis/causal_main_20261019/README.md` incl. both review sections) — do not change their semantics; add an M2 adapter path.

## Inputs
- M2 code and results: `SSM/ssm_decode/m2_calib.py`, `SSM/scripts/run_m2_calib.py`, `SSM/scripts/run_m2_r1lr.py`, `SSM/results/analysis/m2_calib_20261016/README.md` (review corrections at the top: folded root in_proj bias must be included; full R1 Gram is non-deployable; lr16/lr32 compact readout variants; pool-level clustering; P0 has no prior fallback).
- Data: M2 npz cache in `SSM/results/data_cache/m2` via `load_m2_recording` (with the integrity manifest); NWB root `/mnt/data/work_host/SPINT/SPINT-main/data/000953`. Backbones: wide_n16 pools in `SSM/results/forward_eval_m2_wide_n16*_2026100{5,6}/pools/` (3 seeds), D2_POOLS in `SSM/scripts/run_labeled_calibration.py`.

## Methods (causal-first-B; latency 0 primary, 1 sensitivity)
P-prior (mean-table prior incl. bias), P0, ridge-CCA (adapted to M2's fixed channels and finger-velocity labels: condition = reach target/direction if available, otherwise document the condition definition used), W_alpha (alpha by protocol (a) expanding-window; M2's source sessions are the selection pool — document the per-target pool), S1-causal re-based, COMBO, CCA+S1, R1-stream lr16 (compact, deployable) on top of P0, and references R2/R3 (replay of the first B trials) and P0+ui_reg. Gate: AND (primary) and the conservative ≥2-session gate. If a method cannot be adapted to M2 (e.g. no condition labels for CCA), say so and skip it — no placeholders.

## Protocol details
- Calibration = first B usable trials of the target session in recorded order; query = later usable trials; prefix statistics; anytime commits; contiguous causal decoding of the whole session; query R² in physical units.
- B ∈ {1, 2, 4, 8, 16, 33}; 3 backbone seeds (source seeds 0/1/2 as in the M2 README); CIs clustered by pool (3 pools) and by target (5); report per-target and per-pool tables.
- Report the gap for each target in two ways: days from the latest trunk-training session, and days from the latest source/selection session.
- Compare with the random-draw M2 rows (protocol-change deltas).

## Rules
- Frozen trunk; no target backprop in candidates; H4; all selection on source only; never FALCON held-out/minival/official evaluation data; never `results/official_evalai/`.
- Local GPU `cuda:0` only (one RTX 3090; ≤3 jobs; check `nvidia-smi`); no Colab (another agent runs Colab). If the GPU disappears, stop and report.
- Env: `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_m2causal /home/xinyuan/miniconda3/envs/spint/bin/python`.
- Tests: the M2 causal path's first-B/prefix causality (perturbation test at B ≥ 8), bias folding included, adapter parity with the stored random-draw P0 rows when the same trials are used.
- Outputs: `SSM/results/m2_causal_20261020/` (gitignored). Report: `SSM/results/analysis/m2_causal_20261020/README.md` (Simplified Chinese prose, technical terms English) + compact csv.
- Git: commit ONLY your own files with explicit `git add <paths>`; English messages; NEVER push; never touch `SSM/overleaf/`.

## Final message
Main M2 table for all B (methods as above), per-pool results, gate sensitivity, protocol-change deltas, what could not be adapted, file paths.
