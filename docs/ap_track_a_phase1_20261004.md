# AP Backbone and Track A Phase 1 (M2 Rotation)

Date: 2026-10-04. Status: first full iteration complete. The decision gate 3.7 fails on M2. The failure mode is diagnosed and the next design is fixed.

## 1. Scope

This note records the first execution round of the exploration roadmap (`docs/exploration_roadmap_20261004.md`).

1. Phase 0 item C2: state carry for the official Mamba-3 kernel and a streaming step reference. Done by a separate work stream. See `results/analysis/c2_streaming_20261004/README.md`.
2. Phase 0 data item: DANDI 000688 subject-M adapter. Done. See `results/analysis/dandi688_subm_adapter_20261004/README.md`.
3. Phase 1 Track A: closed-form association-profile tables on a frozen AP backbone, evaluated on a 7-target leave-one-run-out rotation of FALCON M2. This note covers items 3 and the supporting build.

The user set the data priority for this round: M2 primary and 000688 subject-M auxiliary. The 000688 pretraining runs launched at the end of this round.

## 2. What Was Built

1. `ssm_decode/association.py`. Closed-form Track A math: FA by EM with a Heywood guard, posterior read-in, orthogonal Procrustes frame, behavior-feature profiles with lags, source-population shrinkage, shared profile-to-table map, and the A-M1 assembly. Ten synthetic tests in `tests/test_association.py` verify subspace recovery, frame recovery, and end-to-end decode through the source head.
2. `ssm_decode/ap_backbone.py`. The C8 backbone: per-session `K x N` read-in tables, shared lift `W_lift` (the base `in_proj`), the frozen Mamba-3 trunk, and the linear head `B_src` on the latent. Includes a whitening penalty and FA or supervised table initialization.
3. `ssm_decode/ap_pretraining.py`. Source pretraining for one rotation. Per-session input z-scores (C1.1), head loss (C8.2), whitening penalty (C8.3), channel gain augmentation and dropout (C8.5). Provenance follows the repository convention: code snapshot, data hashes, manifest, train log.
4. `ssm_decode/multi_target.py`. Leave-one-run-out rotation plans for M2 and M1 plus an SSD npz cache keyed by the NWB sha256. The cache removes repeated HDD reads.
5. `ssm_decode/track_a_experiment.py`. Target-session evaluation at budgets 0, 2, 4, 8, 16, 33 labeled trials. Methods: `fa_identity`, `am1`, `am3` (FA statistics from non-support bins), `am0`, `am2`, `ui_grad` (gradient refinement from the FA init), `mean_table` (fixed-channel reference), and `oracle_frame` (diagnostic only, frame fit on query labels, never compared).
6. `scripts/analyze_track_a.py`. Aggregation and the decision gate 3.7 check.

## 3. Defect Found and Fixed During the Round

The first full rotation trained to trunk source-val R2 near 0.05. The direct 96-channel reference (`session_pretraining`, same trunk size and budget) reached 0.55 on the same data. The cause was a routing bug in `latent_batch`: batches that mix sessions with equal channel counts were routed through `tables[0]` only. Tables 1 to 5 received no gradient and stayed at initialization. Validation used each session's own table, so the macro score matched one trained session averaged with five untrained ones.

The probe chain that isolated it:

1. Train and validation R2 agreed, so no overfitting and no evaluation defect.
2. Instantaneous linear probes: 16-d table latent 0.02, full 96-d 0.09. The system sat at the instantaneous linear ceiling of its input.
3. Ablations without augmentation and without the head and whitening losses failed equally, which excluded those terms.
4. k = 96 with supervised init also failed, which excluded the bottleneck hypothesis.
5. A synthetic single-session overfit passed, which localized the defect to session mixing.

After the fix (`latent_batch` always routes per sample), k = 16 reaches trunk source-val R2 0.45 to 0.49 at 800 steps and 0.51 to 0.55 at 2000 steps across all seven rotations. The K = 16 bottleneck costs about 0.00 to 0.04 source R2 against the 96-channel reference.

## 4. Track A Results on M2

Seven leave-one-run-out rotations. Budgets are labeled support trials. R2 is physical-unit, variance-weighted, query-only. Mean over the seven targets:

| method | b=0 | b=2 | b=4 | b=8 | b=16 | b=33 |
|---|---:|---:|---:|---:|---:|---:|
| fa_identity | -0.035 | -0.094 | -0.061 | -0.070 | -0.085 | -0.040 |
| am1 | -- | -0.042 | -0.029 | -0.008 | 0.032 | 0.049 |
| am2 | -- | 0.005 | 0.029 | 0.062 | 0.101 | 0.140 |
| am3 | -- | -0.032 | -0.028 | 0.003 | 0.061 | 0.066 |
| am0 | -- | -0.011 | -0.005 | 0.009 | 0.034 | 0.068 |
| ui_grad | -- | -0.035 | 0.007 | 0.034 | 0.098 | 0.187 |
| mean_table | 0.363 | 0.316 | 0.322 | 0.326 | 0.341 | 0.347 |
| oracle_frame (diag.) | -- | 0.040 | 0.035 | 0.008 | 0.042 | 0.097 |

Decision gate 3.7: A-M1 and A-M3 at budgets 2, 4, 8 win 0 of 7 against A-M0 at 33 trials. The gate fails.

## 5. Diagnosis

1. `mean_table` dominates every label-using closed-form method at every budget, 7 of 7 sessions. On M2 the 96 electrode-level sua channels are stable across sessions. The cross-session shift that remains is close to a per-channel gain and offset, which the mean source table absorbs. This repeats the audit finding that normalization and channel affine carry most of the M2 gain.
2. `oracle_frame` reaches only 0.097 mean at 33 trials. Even a frame fit on query labels cannot make the target FA latent usable for the trunk. The defect is not the frame fit. The target FA latent and the source latent differ by more than an orthogonal map: FA fixes per-dimension scales by its own variance ranking, and those scales differ per session.
3. The whitening penalty at `lam_white = 0.1` left source latent diagonal covariance at 0.3 to 0.6, not 1.0. Orthogonal Procrustes assumes white latents on both sides (roadmap C8.3). The precondition does not hold yet.
4. `am2` is the best closed-form label method. It beats `am0` by +0.072 at 33 trials and wins 6 of 7. The source-population prior carries real information. This supports the shrinkage family.
5. `ui_grad` improves monotonically with budget and reaches 0.187 at 33 trials. Gradient refinement of the FA table works, but starts from the same weak FA frame.

## 6. Reading Against the Track A Hypothesis

The roadmap hypothesis says: labels fix the frame, unlabeled data fix the per-unit table, and labels then estimate about `K^2` numbers. On M2 this hypothesis has little room to work, because the fixed channel grid already supplies a strong table prior (`mean_table`) and the FA frame cannot reach the source latent by an orthogonal map.

The hypothesis targets datasets with variable unit sets. There the mean table does not exist: sessions have different `N` and different units. DANDI 000688 subject-M is that setting (10 to 52 units per session). The 000688 evaluation is the primary test of Track A, not M2.

## 7. Next Steps

1. Finish the 000688 subject-M pretrainings (two dev targets, k = 12, running at the time of writing) and run the same budget sweep there. `mean_table` is undefined on 000688, so the comparison isolates the closed-form table methods.
2. Whiten both latent sides before the frame fit. Post-hoc per-dimension z-scoring of the target FA latent, and stronger whitening or explicit standardization at source, make the orthogonal Procrustes precondition hold. This is a small change to `am1_table`.
3. Add a table-space shrinkage method: ridge the target table toward the mean source table with the labeled residual. On fixed-channel data this generalizes `mean_table`; on variable-unit data it needs the profile-space prior (`am2`) or the prototype read-in (A-M5).
4. Track the null-space completion of `T` with the source latent dynamics (roadmap A-M1 step 5) once the whitened frame is in place.

## 8. Artifacts

1. Pretraining: `results/ap_pretraining_m2_20261004/<target>/` with manifest and train log per rotation. 000688: `results/ap_pretraining_dandi688sm_20261004/`.
2. Evaluation: `results/track_a_m2_20261004/<target>/{results.jsonl,summary.json}`.
3. Aggregation: `scripts/analyze_track_a.py results/track_a_m2_20261004`.
4. Tests: `tests/test_association.py`, `tests/test_ap_backbone.py`, `tests/test_mamba3_streaming.py`, `tests/test_dandi688_data.py`. Full suite 205 passed, 1 skipped (integration-gated), 7 deselected.
