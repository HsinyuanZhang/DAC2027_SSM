# Task: reorganize the Overleaf experiment log with ONE consistent vocabulary (2026-10-20)

You are a worker agent; the coordinator (Claude) will polish your draft afterwards. File: `/home/xinyuan/Work_host/SSM/overleaf/exp_log.tex` (separate git repo; remote = Overleaf; credentials via git credential store — never print or commit tokens). Read `/home/xinyuan/Work_host/AGENTS.md` first. Language: Simplified Chinese prose; technical terms, method names and metric names in English (never translated). Standalone pdfLaTeX + CJKutf8 (escape underscores; symbols only in math mode: $\to$, $\approx$, $\pm$, $\ge$, $\le$; no non-CJK unicode symbols in text). APST is MZ's unpublished work: never mention it; held-out sessions are only "reserved as a held-out set".

## Goal
MZ wants the log to read clearly with a single, consistent vocabulary. The current log grew chronologically and mixes protocols, names and superseded numbers. Rewrite it into a clean, reader-first document. Every number must come from a reviewed results README (cite the path), and every table must state its protocol.

## Unified vocabulary (define once in §1 and use everywhere; rename old names)
- Methods (deployable): **P-prior** (electrode-aligned prior + prefix normalization), **P0** (fixed-weight per-unit encoder on streaming statistics), **ridge-CCA** (regularized CCA alignment, our variant of Gallego 2020), **W$_\alpha$** (P0 shrunk toward ridge-CCA with α from the expanding-window rule), **S1** (single-pass replay-free fixed-feedback refinement, re-based; formerly "S1-causal re-based"), **COMBO** = W$_\alpha$ + S1, **CCA+S1**, **R1-stream** / **R1 lr16** (streaming readout ridge).
- References (not deployable or replay-based): **R2** (F7-J with replay), **R3** (R2 + readout ridge), **P0+ui\_reg** (gradient through the trunk; "gradient reference", never "upper bound").
- Published baselines: Stabilizer, Gallego CCA, NoMAD.
- Protocols: **causal-first-B** (main) vs **random-draw** (legacy); α selection **protocol (a) expanding-window** (main) and **(b) strict target-disjoint**; gates **AND** (main) and **≥2-session** (conservative).
- Gap tiers: **≤30 d**, **31–180 d**, **>180 d**, using the gap from the latest TRUNK-TRAINING session to the target; mention the second gap definition (from the latest selection/source session) where it differs.
- Metric: query R² in physical units (intracortical, Hyser); balanced accuracy (DB6).

## Structure
0. 一页摘要: the 5–8 main conclusions with numbers, each tagged with tier and protocol; frozen algorithm; current status (Colab round 2 running — results pending; do not invent them).
1. 定义与协议: the vocabulary above in one table; data sources table (reuse SSM/docs/paper/tab_datasets.tex facts); evaluation protocol; constraints (frozen trunk, no target backprop, only B labeled trials, static SRAM table).
2. 主结果（causal-first-B）by gap tier:
   - ≤30 d: 000688 sub-M days (results/analysis/causal_main_20261019/README.md days table), FALCON M2 (results/analysis/m2_causal_20261020/README.md, corrected version incl. the W$_\alpha$ range and pool-level paired diffs), sub-C days if available under causal-first-B (otherwise state it is random-draw and pending round 2).
   - 31–180 d: sub-M targets with trunk gap 31–180 (20140626/20140627 at 112 d), sub-C month (sc_0629, sc_0708; per backbone).
   - >180 d: the remaining drift targets (stress test; one backbone).
   For each tier: one compact table (methods × B), paired differences vs P0 and vs ridge-CCA with CIs and k/n positive, and 2–4 bullet takeaways. Recompute tier aggregates from per_target.csv where needed and state exactly which targets are in each tier.
3. 与已发表方法对比 (results/analysis/baselines_20261016/README.md, twice-reviewed; protocol stated; limits).
4. 方法分析与负面结果: S1 (stream_f7 three reviews: exact reordering, retains ~0.82 of R2 gain on drift; single-pass limitation on M2), P0-Solve/projection diagnostic (dimension effect), set pooling (hybrid v1/v2; Hyser identity-free +0.20 vs capacity control), sEMG (Hyser, DB6 v2) as boundary.
5. 硬件相关证据: calibration-path quantization (P2 corrected: per-add 16-bit accumulators, 17-bit counters, int8/int4 encoder, int8 table), trunk 2T2R PTQ, architecture analysis (docs/arch_analysis_20261019.md: op mix, state-traffic bottleneck, ADC-dominated CiM, CCA-commit SRAM peak 158.5 KiB > 128 KiB, S1 zero-state forward gap) — clearly marked analytic/fp32.
6. 局限与待办 (selection-session scarcity, single backbone for drift, round 2 pending, Hyser zero-phase front end, etc.).
Appendix A: timeline (one table). Appendix B: superseded results (e.g. DB6 v1, early random-draw headlines, prior-anchored S1, pooled-α rule) — one line each with why superseded. Remove all stale "running"/"next steps" text from the body.

## Rules
- Do not run any experiment or GPU job. Read-only on the SSM repo except reading result files; small CPU scripts for tier aggregation are fine (put them in SSM/scripts/explog_tables.py and commit that one file in the SSM repo with explicit git add; NEVER push the SSM repo).
- In the overleaf repo: `git pull --rebase` first; you may restructure exp_log.tex freely; keep existing figures if still valid (fig8 regenerated without APST is valid; check others against the reviewed numbers — drop or mark any figure that shows superseded numbers). Compile-check with tectonic if available (XeTeX; CJK glyphs won't render; syntax/structure only) plus a brace/environment/ref check. Commit in the overleaf repo with a clear English message but do NOT push — the coordinator will polish and push.
- Final message: section list, which numbers you recomputed (with script), any inconsistencies you found between READMEs, compile-check result, overleaf commit hash.
