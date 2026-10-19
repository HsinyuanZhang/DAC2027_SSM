# Task: prepare expansion round 2 with the frozen methods under the causal-first-B protocol (2026-10-19)

You are a worker agent; the coordinator (Claude) reviews your work. Work in `/home/xinyuan/Work_host/SSM` (git, `main`). Read `/home/xinyuan/Work_host/AGENTS.md` (incl. the Compute/Colab section) and `SSM/docs/feasibility_p1_20261015.md` §0 first. APST is MZ's unpublished work: do not use or mention it.

## Frozen algorithm (MZ decision 2026-10-19)
Main method = P0 + W_alpha (alpha selected with the EXPANDING-WINDOW protocol (a)) + S1-causal re-based refinement ("COMBO"); alternative = CCA+S1; deployable comparators = P-prior, P0, ridge-CCA, W_alpha, S1-causal; references = R2, R3 (replay), P0+ui_reg (non-deployable). Protocol = causal-first-B (first B usable trials calibrate, later trials are the query, prefix statistics, anytime commits, contiguous decoding). Primary gate = AND as implemented; also report the conservative ≥2-session gate. All of this is implemented in `SSM/ssm_decode/causal_protocol.py`, `SSM/scripts/run_causal_main.py`, `SSM/scripts/analyze_causal_main.py` (commit 8f8bfef; README `SSM/results/analysis/causal_main_20261019/README.md`, incl. both review sections). Do not change their semantics.

## Goal of round 2
Add statistical power and, above all, MORE CAUSAL SELECTION SESSIONS (the main weakness: one selection session per trunk). Backbones (from `SSM/docs/expansion_subc_days_20261016.md`, v2 OOS protocol: trunk trained on pool[:5], pool[-1] held out): sc_0309 (sub-C year), sm_0307 (reused month trunk; its selection is IS — keep it separate), sm_0627, sm_0512, sm_0611 (sub-M days with rolling backbones). Queue code: `SSM/scripts/run_expansion.py`, `SSM/scripts/run_expansion_queue.sh` (round 1 ran on Colab; outputs under `SSM/results/expansion_round1/`).

## What to build
1. Extend the expansion queue (new stage(s), no change to existing stage semantics) so that, per backbone, after the existing stages it also produces the frozen-method rows under causal-first-B: P-prior, P0, ridge-CCA (sub-C path fixed version), W_alpha with protocol (a) AND (b), S1-causal, COMBO, CCA+S1, R2, R3 (replay of the first B trials), with the same commit/latency options (latency 0 primary, 1 as sensitivity). Selection must use only sessions/trunks allowed by protocol (a)/(b) for each target; the selection-session pool now includes every held-out pool[-1] session of every backbone that is strictly before the target (and its trunk strictly before the target) — log the per-target selection composition (number of distinct selection sessions).
2. Also re-score round-1 backbones (sc_0708, sc_0713, sc_0629) with the new stage, so that sub-C gets the frozen-method rows with the enlarged selection pools (round-1 artifacts are in `SSM/results/expansion_round1/x/`).
3. Tests: protocol (a)/(b) selection pools exclude every disallowed session/trunk (incl. same-trunk duplicates such as month==year); first-B causality (perturbation test at B ≥ 8 with α>0, η>0); resumability of the new stage; Colab path handling (`--out` local VM disk, periodic tar sync to Drive as in the round-1 run — see `SSM/results/analysis/expansion_20261016/round1_README.md` and `SSM/docs/colab_runbook.md`).
4. A CPU smoke run end-to-end on one small backbone/target (GRU stand-in is fine) and, if the local GPU is idle (`nvidia-smi` < 2 GB used), a short GPU smoke on one real target.
5. A cost/time estimate for the Colab round on an L4 (1.54 CU/h; round 1 took 1 h 51 min / 3.1 CU for 3 backbones) with a hard cap of 40 CU and the ~12 h session limit; propose a split if needed.
6. A bundle list (code at your final commit + any data not already on Drive in `gdrive:DAC27/colab/`: currently there are data_000688_subM.tar.zst, data_000688_subC.tar.zst (30 primary sub-C NWB), ckpts.tar.zst, code bundles). Do NOT upload (the coordinator runs rclone).

## Rules
- Frozen trunk; no target backprop in candidates; H4; source-only selection; never final sessions 20150623/0625/0626; never 20140307 as a target; sub-C forbidden-session guard; 2016 sub-C tier off.
- Local GPU `cuda:0` only for smoke tests (≤1 job; another agent may analyze code but not use the GPU). Do NOT use Colab yourself.
- Env: `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM /home/xinyuan/miniconda3/envs/spint/bin/python`.
- Git: commit ONLY your own files with explicit `git add <paths>`; English messages; NEVER push; never touch `SSM/overleaf/`.
- Update `SSM/docs/expansion_subc_days_20261016.md` with a "round 2 (frozen methods)" section (Simplified Chinese prose, technical terms in English): stages, run commands for Colab, CU/time estimate, bundle list.

## Final message
What was added, test results, smoke results, CU/time estimate and split, exact Colab run commands, bundle list path and size.
