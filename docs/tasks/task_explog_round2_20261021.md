# Task: add expansion round 2 to the Overleaf experiment log (2026-10-21)

You are a worker agent; the coordinator (Claude) will polish your draft afterwards. File: `/home/xinyuan/Work_host/SSM/overleaf/exp_log.tex` (separate git repo; remote = Overleaf; credentials via git credential store — never print or commit tokens). Read `/home/xinyuan/Work_host/AGENTS.md` first. Language: Simplified Chinese prose; technical terms, method names and metric names in English. Standalone pdfLaTeX + CJKutf8 rules as before (escape underscores; symbols only in math mode; no non-CJK unicode symbols in text). APST is MZ's unpublished work: never mention it.

## Inputs
- Round-2 report: `SSM/results/analysis/expansion_20261016/round2_README.md`, `round2_tables.md`, `round2_summary/` — INCLUDING the review section at the top of the README (apply every correction listed there; reviewed numbers override the original ones).
- The current log already uses the unified vocabulary (§1). Keep it: P-prior, P0, ridge-CCA, W$_\alpha$, S1, COMBO, CCA+S1, R2, R3; causal-first-B; protocol (a)/(b); AND / ≥2-session gates; trunk-gap tiers ≤30 d / 31–180 d / >180 d.

## What to change
1. §2 (main results by tier): add a "round 2 (expanded)" table per tier — methods × B (1, 4, 8, 16, 33 as available), paired differences vs P0 and vs ridge-CCA with target-session-clustered CIs and k/n positive, the exact list of target sessions (and backbones) in each tier, the number of distinct selection sessions per target (range). Keep the round-1 / 000688 tables; state clearly which rows are new.
2. Flag explicitly: non-converged backbones (sc_0309, sm_0512 hit the step cap) and which tier rows depend on them; the sm_0307 row with target 20140307 is excluded (protocol rule); COMBO coverage gaps (B values without a rule) so averages over different B sets are not compared as if equal.
3. Update §0 (一页摘要) tier table and takeaways ONLY where round 2 changes a conclusion or adds sessions; keep each claim tagged with tier, protocol and n (sessions).
4. Update §6 (局限与待办): selection-session scarcity status after round 2.
5. Appendix A timeline: one row for round 2 (Colab, ~5.5 CU, commit hash).

## Rules
- Do not run any experiment or GPU job. Small CPU aggregation scripts may extend `SSM/scripts/explog_tables.py` (commit only that file in the SSM repo with explicit git add; NEVER push the SSM repo).
- Overleaf repo: `git pull --rebase` first; compile-check with tectonic (syntax/structure) plus a brace/environment/ref check; commit with a clear English message; do NOT push.
- Every number must come from the reviewed README / tables or your script (cite the path).
- Final message: what changed per section, recomputed numbers (with script), inconsistencies found, compile-check result, overleaf commit hash.
