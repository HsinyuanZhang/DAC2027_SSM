# Task H4: S1 with live residuals (deployment-consistent calibration forward) (2026-10-21)

You are a worker agent; the coordinator (Claude) reviews your work. Work in `/home/xinyuan/Work_host/SSM` (git, `main`). Read `/home/xinyuan/Work_host/AGENTS.md`, `SSM/docs/feasibility_p1_20261015.md` §0 (rules 7-9), `SSM/docs/paper_mainline_proposal_20261016.md` §7 (NEW main method: P0 + S1), and `SSM/docs/arch_analysis_20261019.md` (§7 G1 and the H4 row) first. APST is MZ's unpublished work: do not use or mention it.

## Why
The main method is now P0 + S1 (MZ, 2026-10-21). The hardware story is "calibration reuses the inference hardware": the S1 residual r_t = y_t - yhat_t should come from the live inference forward. But the current S1 (`rebased_feedback_loop`, `ssm_decode/stream_f7.py:491-570`) runs each calibration trial from a ZERO trunk state in a separate forward, while deployed inference (and our evaluation, `scripts/run_causal_main.py:538-548`) decodes the whole session contiguously with state carried across trials and tables switched at commits. If S1 keeps its gain with live residuals, no shadow forward is needed; otherwise calibration needs a second forward (trunk MAC x2, ~50 KiB extra state). This task measures that.

## Variants (same S1 math, same commit timing, causal-first-B, latency 0 primary / 1 sensitivity)
- **L0 (current)**: zero-state per-trial forward for the S1 residual. Must reproduce the stored S1 rows (parity check, report max |diff|).
- **L1 (live, main candidate)**: one contiguous causal forward of the session; during calibration trial k the residual comes from the live forward under the table committed at that time (P0_{<k} + Delta_{<k}, with the documented staging for trial 1); trunk state carried across trial boundaries; the FIR input history (ring of x_{t-l}, L=33) also carried across trial boundaries (no zero padding at trial starts). Bins between usable trials (if any) are decoded but contribute no residual (no label) - document exactly what happens to the ring and to the FIR terms there.
- **L2 (ablation)**: live state as in L1 but the ring/FIR history reset at each trial start - separates the state effect from the history effect.
- For each variant, (eta, rho) must be selected with the SAME variant on the selection episodes (source only, protocol (a), AND gate, conservative gate also reported). Also report "L1 with the stored L0 picks" (transfer) as a cheap secondary row.

## Targets (priority order; stop and report if time runs out)
1. Round-2 31-180 d tier (17 sessions; `results/analysis/expansion_20261016/round2_README.md`, review corrections at top; artifacts in `results/expansion_round2/x/` and `results/expansion_round1/x/`; if any checkpoint/selection artifact is missing locally, say which — do not use Colab).
2. Round-2 <=30 d tier (8 sessions) - S1 must not hurt P0 here.
3. 000688 causal-main targets (`results/analysis/causal_main_20261019/README.md`): sub-M days, month/112 d, drift (>180 d stress test), sub-C sc_0629/sc_0708.
4. Round-2 >180 d (stress test; report with and without sc_0309).
Exclusions: 20140307 never as a target; never final sessions 20150623/0625/0626; sub-C forbidden sessions; no 2016 tier; never `results/official_evalai/`.
Controls (only after 1-3 are done, and only if each costs < 1 h GPU): COMBO and CCA+S1 with L1.

## Report
- Per tier: P0, S1-L0, S1-L1, S1-L2 (session means for B in {1,4,8,16,33}); paired S1-L1 - S1-L0 and S1-L1 - P0 with target-session-clustered CIs and k/n positive; B4-33 averages; pick changes between L0 and L1.
- A clear verdict: does S1-L1 keep the 31-180 d gain (vs +0.055 [+0.034,+0.076] for L0)?
- Hardware note: buffers that L1 needs vs L0 (ring carried, no shadow state), per rule 8 (per-add simulation not required here — this is an fp32 algorithm check; say so).
- Output dir `results/h4_live_20261021/` (gitignored); README `results/analysis/h4_live_20261021/README.md` (Simplified Chinese prose, technical terms English) + compact csv.

## Rules
- Implement L1/L2 as new code paths (a flag or new function); do not change L0 semantics. Tests: L0 parity with stored rows; L1 causality (perturbing bins after time t changes nothing used at <= t, at B >= 8 with eta > 0); L1 with eta = 0 equals the stored P0 contiguous decode exactly; state actually carried (a test that fails if the state is reset at trial starts).
- Local GPU `cuda:0` only (one RTX 3090; check `nvidia-smi`; <= 2 jobs). No Colab. If the GPU disappears, stop and report. Make long runs resumable.
- Env: `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_h4 /home/xinyuan/miniconda3/envs/spint/bin/python`.
- Frozen trunk; no target backprop; H4 label rule (only the first B labeled trials); source-only selection.
- Git: commit ONLY your own files with explicit `git add <paths>`; English messages; NEVER push; never touch `SSM/overleaf/`.
- Before the long runs, print a GPU-time estimate per target group in your log.

## Final message
Code changes (file:line), tests, parity numbers, the per-tier table and verdict, what was not done, commit hash.
