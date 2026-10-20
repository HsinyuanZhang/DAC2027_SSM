# Task: fix the round-2 COMBO selection bug, re-score COMBO, correct the round-2 README (2026-10-21)

You are a worker agent; the coordinator (Claude) reviews your work. Work in `/home/xinyuan/Work_host/SSM` (git, `main`, HEAD 18cb407). Read `/home/xinyuan/Work_host/AGENTS.md` and `SSM/docs/feasibility_p1_20261015.md` §0 first. APST is MZ's unpublished work: do not use or mention it. Do not change method semantics (frozen algorithm); this is a bug fix + re-score.

## Review findings to address (Opus review of 18cb407; scratch artifacts in `/tmp/r2_review/`: combo_fix.json, rescore.jsonl, fixed_tables.py, alpha_groups.py, b_pool.py, orphan.sh — read them, reuse what is correct)
- F1 CRITICAL: `dedupe_same_trunk` (`ssm_decode/expansion_causal.py:125-138`) keys on (trunk, date, seed, B) without alpha; `selection_pool` (:141-146) applies it to COMBO, so each held-out unit keeps only its first-loaded alpha (530 -> 238 COMBO episodes; an alpha-aware key keeps 518). `combo_at_alpha` (:262) then finds nothing at the selected alpha -> spurious "no rule" (COMBO(a) B=33 for 17 targets, COMBO(b) B>=4 everywhere) and picks from 1-3 sessions. Fix: include alpha in the dedupe key for alpha-indexed families (COMBO; check every family that is alpha-indexed). Add a unit test with a multi-alpha COMBO pool (the existing smoke only scored alpha=0.5). Re-run the COMBO causal-eval for every affected cell (the reviewer measured ~15 s/target on the 3090; reviewer's rescore reproduced stored rows to ~1e-4 — use it as a cross-check, your numbers must match it). Expected corrected COMBO(a) B4-33 vs P0: <=30 d +0.006 [+0.001,+0.012]; 31-180 d +0.039 [+0.028,+0.051] 16/17; >180 d +0.206. COMBO(b) should equal W_alpha(b) (S1(b) always falls back).
- F2 HIGH: >180 d tier = 6/7 sessions from sc_0309 (non-converged, step cap; its P0 collapses to 0.084 B4-33). Report >180 d with and without sc_0309, report sm_0512 (also non-converged) separately, and note that session clustering ignores 6 sessions sharing one trunk/selection. Also flag that sc_0309's held-out 20131220 votes in the (a) le14 bin.
- F3 HIGH: protocol (b) pool is a single session (sc_0309:20131220, 2 episodes) for all 34 targets; describe (b) as a single-session ablation; fix README line 25 wording.
- F4 MEDIUM: target 20140307 (sm_0307) must be excluded EVERYWHERE incl. T7, T9 (`scripts/expansion_round2_tables.py:155` uses cells_all), audit counts and README; reportable counts = 34 pairs / 26 distinct sessions. Its use as a selection/source session in legacy episodes stays allowed.
- F5 MEDIUM: selection-session counts: report the number of sessions that actually vote in the target's gap bin (alpha) and per S1 pool, not the whole allowed pool; note the days-trunk imbalance in the S1 pool.
- F6: ridge-CCA short-gap weakness is genuine; disclose that month `CCA_CFG` was used for every backbone (causal-main days used `CCA_CFG["days"]`); do NOT re-select now, just disclose.
- F8 LOW: `TRUNK_TRAIN["sc_0629"]` is empty (`ssm_decode/causal_protocol.py:407`) though that trunk trained on 20150309/20150312; fill it in correctly and confirm no numeric change.
- F9 OPERATIONAL: `scripts/run_expansion_queue.sh:122-124,154` orphaned `sleep` keeps the flock; fix with the reviewer's pattern (trap + `sleep & wait`; kill then wait before the final sync) and make `sync_out` use a unique partial name (`dest.partial.<pid>`) in `ssm_decode/expansion_causal.py:375-382`. Test with a short script (no Colab).

## Outputs
- Re-scored rows into the round-2 output tree (`results/expansion_round2/x/expansion_out/`, gitignored; keep the old rows recoverable, e.g. move them to a `superseded_20261021/` subdir rather than deleting).
- Regenerate `results/analysis/expansion_20261016/round2_summary/`, `round2_tables.md`, and correct `round2_README.md`: add a "Review corrections (2026-10-21)" section at the top listing F1-F9 and what changed; update all headline numbers; Simplified Chinese prose, technical terms English.

## Rules
- Local GPU `cuda:0` only (one RTX 3090, idle now; <=2 jobs); no Colab. If the GPU disappears, stop and report.
- Env: `PYTHONNOUSERSITE=1 PYTHONPATH=/home/xinyuan/Work_host/SSM/.tools/mamba_deps:/home/xinyuan/Work_host/SSM TRITON_CACHE_DIR=/tmp/triton_r2fix /home/xinyuan/miniconda3/envs/spint/bin/python`.
- Frozen trunk; no target backprop; H4; never final sessions 20150623/0625/0626; never 20140307 as a target; sub-C forbidden sessions; no 2016 tier; never `results/official_evalai/`.
- Run the existing tests (`tests/`) plus the new ones; all must pass.
- Git: commit ONLY your own changed files with explicit `git add <paths>`; English messages; NEVER push; never touch `SSM/overleaf/`.

## Final message
Code changes (file:line), tests, number of re-scored cells and cross-check vs /tmp/r2_review/rescore.jsonl, corrected tier tables (all methods, B4-33 vs P0 and vs ridge-CCA with CIs and k/n), >180 d with/without sc_0309, commit hash.
