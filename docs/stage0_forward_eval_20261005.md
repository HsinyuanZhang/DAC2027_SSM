# Stage 0' Execution: Fixes and the Time-Forward Evaluation Framework

Date: 2026-10-05. Status: fixes complete, framework running, M1 baseline reproduced. M2 wide baseline numbers pending at the time of writing; this file is updated when they land.

## 1. Revision Fixes (roadmap revision section 2)

All items except the two 000688-specific ones are fixed, tested, and committed.

| Item | Fix | Test |
|---|---|---|
| 2 | `latent_batch` routes per sample through its own session table; unequal channel counts fail loudly instead of stacking | `test_latent_batch_routes_mixed_sessions` |
| 3 | `core_step` batch-not-equal-heads check added | `test_core_step_batch_not_equal_heads` |
| 5 | Whitened A-M1: explicit per-dim standardization on both latent sides, identity-biased canonical null-space completion, FA restarts with spread reporting | `test_am1_whitened_*` (determinism, cross-seed stability, offset identity) |
| 6 | Deployment carries an explicit latent offset | `test_am1_whitened_offset_identity` |
| 7 | Behavior features built on the continuous recording, then rows selected | `test_behavior_features_selected_from_continuous_recording` |
| 8 | Decode is one contiguous forward per recording | `test_decode_r2_single_contiguous_pass` |
| 9 | `ui_grad` trains on whole labeled trials with the base frozen | `test_ui_grad_freezes_base_and_trains_table` |
| 10 | Checkpoint target or source-pool must match the evaluation arguments | `test_checkpoint_target_mismatch_raises` |
| 11 | Cache writes atomic via `os.replace`; entries keyed by data root | `test_cache_root_key_and_atomic_write` |
| 12 | Queue scripts create the output root and clean failed directories so reruns retrain | verified by rerun behavior |
| 13 | Streaming angle precision defaults to the public-wrapper (bf16) semantics, the same function the training forward computes; fp32 remains an option | updated `test_gpu_chunked_carry_angle_continuity` (dual gates) |
| 14 | `initial_states` receives the full device object | covered by streaming tests |

Full suite: 216 passed, 1 skipped (integration-gated), 7 deselected.

Item 15 (000688 plan scanning final sessions) and item 16 (single pretraining per source pool with extended budget) are 000688-side and land with the next 000688 round.

Cleanup: 18 Triton caches and the unreferenced smoke and width-probe artifacts are deleted, about 3868 MB. Record: `results/analysis/cleanup_20261005.md`.

## 2. Time-Forward Framework (revision 4.1)

1. `ssm_decode/forward_eval.py` plus `scripts/run_forward_eval.py`: day-grouped forward rotation. Targets on the same day share one source pool, and the pool is pretrained once. Leave-one-out mode exists as the control.
2. Budget-extension rule: a pool whose best step lands in the last 20 percent retrains at double budget, at most twice.
3. Methods with parameters run an LR candidate set; the LR is selected per (target, method) by the mean prefix-validation R2 over seeds. Query evaluation runs only for the selected LR.
4. Aggregation reports the multi-target mean, per-target values, seed SD, and paired differences against a reference method.

A mid-run incident is recorded for provenance: a parallel work line edited two guarded source files while a smoke fit was executing. The implementation-hash guard aborted that fit correctly. No reported number is affected.

## 3. Baseline Reproduction

Protocol: forward rotation, support z-score inputs, support budgets M1 = 10 and M2 = 33 trials, R2 on all-valid query bins, physical units.

### M1 small (128x2), method none

| target | R2 |
|---|---:|
| 20120926 | 0.5246 |
| 20120927 | 0.7262 |
| 20120928 | 0.6914 |
| **multi-target mean** | **0.6474** |

Reading: the historical single-target development numbers (0.674 to 0.688) sit inside the target spread (0.52 to 0.73). One target alone cannot rank methods on M1.

### M2 wide (256x4), methods none and lora

Three seeds for lora, two LR candidates (3e-4 and 1e-3) selected per target on prefix validation.  Multi-target means over five forward targets:

| method | mean | per-target range | seed SD range |
|---|---:|---:|---:|
| none | 0.3236 | 0.164 to 0.424 | deterministic |
| lora | **0.3720** | 0.273 to 0.450 | 0.003 to 0.039 |

Paired difference lora minus none: +0.048 mean, 4 of 5 targets, with the two earliest targets gaining +0.11 to +0.12 and the 10-27-R1 target slightly negative.

Readings:

1. The historical single-target number for this candidate (about 0.486 on 2020-10-28-Run1) reappears as 0.450 on the same target under the forward protocol.  The forward mean is lower because the two earliest targets have only a two-session source pool.
2. Source-pool size is a first-order factor on M2: every method loses about 0.1 to 0.2 on the two-source pool relative to the six-source pool.
3. Seed SD under 0.04 and the paired-difference pattern give the protocol enough resolution to separate methods; single-target ranking would not have.

