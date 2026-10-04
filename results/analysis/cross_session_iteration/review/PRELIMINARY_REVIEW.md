# Independent Stage A review

Date: 2026-10-03

Status: **superseded by [FINAL_REVIEW.md](FINAL_REVIEW.md)**.

This file preserves the preliminary findings from the in-progress Stage A review. The final review contains the completed Stage A and Stage B decisions, legal-budget probes, and deployment-candidate checks.

This review covers the new cross-session iteration only. The review did not change model code, runner code, prior results, or the user audit.

The review includes CPU checks and coordinated GPU replay. The review released each GPU after its assigned check.

## Findings that require changes

### P0: the saved exploratory EMA result does not implement its stated protocol

`results/analysis/audit_20261003/ema_eval.py` initializes moments from all support trials. It then starts EMA updates at recording bin zero.

This process changes support inputs and counts support observations twice. The ridge support predictions therefore do not use the fixed support z-score.

Each output uses moments from earlier bins. Thus, the per-bin update is strict-past. The complete process still violates the support-frozen protocol.

The old variance update also uses the updated mean in its squared residual. The required update uses the pre-update residual:

```text
delta = x_t - mean
z_t = delta / sqrt(variance)
variance = (1 - alpha) * (variance + alpha * delta^2)
mean = mean + alpha * delta
```

The old EMA scores can remain exploratory evidence. They must not serve as Stage A acceptance values.

### P0: the audit document gives the opposite state-offset coordinate sign

The official kernel computes `q_rot = R(+Phi) q` with the matrix used in the source code. Therefore:

```text
q_rot^T h_offset = q^T R(-Phi) h_offset
```

The current rotated-readout offset maps to `R(-Phi) h_offset` in physical coordinates. It does not map to `R(+Phi) h_offset`.

The main design conclusion remains valid. The unrotated `C + C_bias` readout gives a constant physical-coordinate state offset.

The final report must state the sign with the implemented rotation convention. A two-dimensional `Phi = pi/2` oracle must enforce it.

### P1: the coordinate controls use different final quantization

The first reviewed `input_adaptation.py` returned the original readout as a float32 sum. The rotated readout quantized its final sum to BF16.

This difference can change the two controls when `Phi = 0`. A CPU counterexample produced a maximum difference of `0.00390625`.

Both coordinate branches must apply the same final quantization. A fractional `C + C_bias` test must require bitwise equality at zero phase.

Resolution: `original_readout` now applies the same final BF16 round trip. The fractional zero-phase test passes bitwise.

### P1: the EMA receipt does not fully describe its initial numerical state

The first reviewed `session_normalization.py` normalizes the prefix with float32 moments. It initializes query EMA from float64 moments.

The receipt stores only the float32 moments. Exact replay from the receipt alone is therefore unavailable.

The implementation must quantize the initial moments once, or record the exact float64 moments and their hashes.

Resolution: the implementation now quantizes the moments once. Query EMA starts from the moments recorded in the receipt.

### P1: the old evaluation scripts do not provide sufficient provenance

`renorm_eval.py` and `ema_eval.py` write only scalar JSON results. They do not record input cohorts, predictions, data hashes, checkpoint hashes, or code hashes.

The scripts also overwrite their output paths. The saved numbers cannot establish their own execution provenance.

Stage A must save immutable run manifests and prediction arrays. The independent replay must use those artifacts.

## Findings that are technically supported

The static support z-score transformation has the intended algebra. After the source normalizer runs, the decoder receives `(x - support_mean) / support_std`.

The support statistics use only neural observations from the first 33 usable trials. The computation does not require support behavior labels.

The observed input mapping is not rank four. The source root projection has rank 64 for M1 and rank 96 for M2.

For the observed support gain mapping, `W diag(g - 1)` has rank 64 for M1 and rank 96 for M2. A rank-four delta cannot represent it exactly.

The new affine export equation is valid. It uses the effective LoRA weight before it folds the affine transform:

```text
W_effective = W_base + delta_W
W_export = W_effective diag(g)
b_export = b_effective + W_effective b
```

The reviewed root-bias scope is valid. Root LoRA can train only the root input bias when the runner enables that option.

The offset controls contain channel affine and state-offset parameters. They do not contain root LoRA parameters.

## Current CPU evidence

Six normalization tests passed. They cover support-only statistics, fixed support normalization, strict-past behavior, and the central variance update.

Seventeen targeted adapter tests passed after the quantization change. They cover freeze scopes, affine merge, replay, causality, and coordinate geometry.

The saved source-normalization scores agree only at the precision used in the audit table:

| Task | `renorm_eval.json` | original source run | formal round-1 `none` |
|---|---:|---:|---:|
| M1 | 0.386209428 | 0.386209428 | 0.386209726 |
| M2 | -0.005921721 | -0.005924225 | -0.005921721 |

The checkpoints are tensor-identical between the source run and formal `none` run. Their saved predictions are not bitwise identical.

M1 differs in 15.1 percent of saved prediction elements. Its maximum physical-value difference is about `0.00513`.

M2 differs in 8.1 percent of saved prediction elements. Its maximum physical-value difference is about `0.0000582`.

GPU replay must use the fixed evaluation batch and pinned runtime. Acceptance should use stored arrays and a declared numerical tolerance.

## Independent normalization replay

The reviewer replayed both tasks on separate RTX 3090 GPUs. The replay used fixed source checkpoints and no training.

| Task | Protocol | Direct R-squared | Ridge R-squared | Direct change from old audit |
|---|---|---:|---:|---:|
| M1 | source normalizer | 0.3862094283 | 0.4829829931 | 0 |
| M1 | support z-score | 0.6735682487 | 0.6727391481 | +0.0000005960 |
| M1 | fixed EMA, half-life 3,000 | 0.6992542744 | 0.7045345306 | +0.0000176430 |
| M2 | source normalizer | -0.0059217215 | 0.1209173203 | 0 |
| M2 | support z-score | 0.2337931991 | 0.2500762343 | -0.0000146627 |
| M2 | fixed EMA, half-life 3,000 | 0.2669103742 | 0.2751362324 | +0.0010291338 |

The fixed EMA freezes the full support prefix. It starts updates at the first query bin and uses the pre-update residual.

The float64 physical R-squared oracle agrees with the runner within `6.75e-8`. All formal cohort index and truth hashes match.

The M1 prediction archive SHA-256 is `cc39d97d69e88a488c9b55d3b2e02cdcd062b1395b062c4e2dd2c05d1b11c9fb`.

The M2 prediction archive SHA-256 is `2da3c749351aa2d0e4dbcf363a6a027b1012f361fd37e969b51b5f11a02da20e`.

These values replace the old exploratory EMA values. Static support z-score conclusions remain unchanged at the reported precision.

## Independent method smoke test

The reviewer tested all eight methods on M1 with the official GPU kernel. The test used no saved training result.

All methods gave the exact source output before an optimizer step. All methods also passed the future-prefix causality check.

Each trainable method had finite gradients and at least one changed permitted parameter. No frozen parameter changed.

The trainable parameter counts were:

| Method | Trainable parameters |
|---|---:|
| `none` | 0 |
| `io` | 21,264 |
| `full` at initial stage | 21,264 |
| `lora` | 36,928 |
| `affine` | 128 |
| `affine_lora` | 37,056 |
| `offset_rotated` | 8,832 |
| `offset_original` | 8,832 |

The report SHA-256 is `b6a00742730271510ef399660090a7db25de489d816d1fc85b5ab7f6cff81bd7`.

## Runner and launcher readiness

The runner masks query labels before it creates training or validation windows. It selects a checkpoint from step zero and prefix validation only.

A query-label poison test produced the exact same selected checkpoint. Query scoring changed only after the query labels became available for final scoring.

A three-block full-tuning test verified the step-200 and step-400 unfreeze boundaries. The unfreeze process preserved AdamW state and unique parameter identities.

The 28 combined runner, normalization, and adapter tests pass. The launcher and runner also pass Python compilation.

The frozen matrix SHA-256 is `a44a4b280e7d36b6a1d8a2728c40debbd57860cd5e70d0b68f6991f3abbf08b9`.

The launcher SHA-256 is `e89e4d2b2da9180081d39356c78386c8b8b463c09bc7db7cf5c3b3760e86d949`.

The launcher freezes the matrix, launcher, transitive code, source checkpoints, and source normalizers. It rejects later byte changes.

Each fit records source, normalizer, target, code, and checkpoint hashes. Selection requires one target-data hash for all fits of a task.

Existing query results must match the selected checkpoint, current code, target data, and required prediction archive.

The final resume gate compares task, method, seed, and learning rate with the deterministic fit path and matrix.

The evaluate phase independently rebuilds prefix-only selections. It rejects a saved selection file that differs from the rebuilt value.

The launcher applies the complete evaluation gate after each new query evaluation.

An independent artifact probe accepted a valid fit and evaluation. It rejected a wrong fit method and a wrong evaluation seed.

## Known receipt limitation

The `root_input_bias_trainable` and `root_bias_behavior` fields describe the LoRA base-bias option correctly.

Those two fields incorrectly report a frozen root input bias for the `io` and `full` methods. The root input bias is trainable for both methods.

The authoritative `trainable_paths`, optimizer groups, parameter counts, and parameter hashes record the actual scope. This wording error does not change training, selection, evaluation, or performance values.

The frozen Stage A and Stage B code remains unchanged. A later receipt-only fix can make these two descriptive fields method-independent.

## Pending formal artifact verification

The formal matrix must finish before the CPU summarizer can give final acceptance.

An independent CPU verifier checks each completed fit as it arrives. It does not read a fit that has no completed `fit_result.json`.

Before final reporting, the review must compare raw truth with current target labels. All new fits of one task must have identical truth hashes.

The formal review must also check every fit identity, source hash, target hash, code hash, checkpoint replay, and parameter allowlist.

## Required Stage A acceptance checks

### 1. Support-only normalization

- Save the exact support bounds, count, mean, standard deviation, and hashes.
- Require `max(support_end) <= min(query_start)` for the scored query cohort.
- Preserve source `y_mean` and `y_std` without modification.
- Replace all query labels and repeat selection. The selected run and checkpoint must remain identical.
- Perturb query neural values. Support statistics and all fit artifacts must remain identical.
- Keep static support z-score and query EMA as separate named protocols.
- Fix the EMA half-life before query scoring.

### 2. Prefix-only learning-rate and checkpoint selection

- Enumerate every method and learning-rate candidate before execution.
- Use only the 26 support-train trials for optimizer updates.
- Use only the 7 support-validation trials for learning-rate and checkpoint selection.
- Include step zero as a checkpoint candidate.
- Define a deterministic tie rule for equal validation scores.
- Extend a run only from prefix-validation evidence.
- Save every validation score, selected step, selected learning rate, and selection reason.
- Repeat selection after query-label poisoning and require the same selected artifacts.

### 3. Valid-mask and frozen-state checks

- Apply `eval_mask` and finite-label checks to every train and validation loss.
- Require finite predictions, losses, and gradients.
- Hash every parameter before and after fitting.
- Permit changes only in the method-specific allowlist.
- Audit every full-finetuning unfreeze boundary before the optimizer step.
- Check optimizer groups, learning rates, weight decay, and parameter identities at each stage.
- Require the zero-initialized adapter output to match the base output for the same batch.

### 4. Direct physical R-squared

- Save direct predictions and truth in physical units.
- Compute the primary score without ridge or residual calibration.
- Recompute `1 - SSE / SST` independently in float64.
- Save `SSE`, `SST`, bin count, and the final scalar.
- Require agreement between the runner and independent oracle within a declared tolerance.

### 5. Cohort identity

- Require one index hash and one truth hash for every task and cohort.
- Require identical hashes across method, learning rate, seed, and normalization protocol.
- Save raw physical target labels as the truth arrays.
- Compare each truth array with the current target data at the saved indices.
- Do not require equality with the round-one truth SHA-256 values.
- Report the numerical difference from the old normalization round trip as a diagnostic.
- Preserve the formal all-valid counts of 50,591 for M1 and 14,115 for M2.
- Preserve the support counts of 4,192 for M1 and 2,053 for M2.
- Report legacy cohorts only as diagnostics.

### 6. Checkpoint replay

- Save source checkpoint, normalizer, data, code, runtime, and argument hashes.
- Rebuild each selected method from its replay receipt.
- Load the selected checkpoint and regenerate every prediction array.
- Use the recorded evaluation batch size and CPU thread count.
- Require exact array replay when the pinned kernel permits it.
- Otherwise, record maximum error and require the declared tolerance.

### 7. Affine and LoRA merge

- Test identity initialization for `affine` and `affine_lora`.
- Test ranks 1, 4, and 16.
- Compare live and folded outputs on random and real support inputs.
- Include nonzero affine bias, nonunit gain, nonzero LoRA delta, and trainable root bias.
- Fold the affine transform with the effective LoRA weight.
- Require the exported state dictionary to contain only source model keys.
- Load the export into a clean source model and repeat the output comparison.

### 8. State-offset coordinates

- Use the same BF16 operations in original and rotated coordinate controls.
- Require bitwise equality when `Phi = 0` with fractional `C + C_bias` values.
- Use a `Phi = pi/2` basis test that verifies the `R(-Phi)` physical mapping.
- Check that RoPE leaves the unrotated tail dimensions unchanged.
- Require original-coordinate output to stay independent of `Phi`.
- Require rotated-coordinate output to change with `Phi` for a nonzero offset.
- Apply the offset gate exactly once.
- Reject streaming APIs until chunked state and angle replay match full forward.

## Interpretation limit

Static support normalization is a large measured factor for M1. It is also a material factor for M2.

The current evidence does not prove that normalization is the largest error source for every target. M2 static normalization remains below the best round-1 adaptation.

The EMA half-life used the query score during exploration. EMA results require a fixed half-life and new replay before comparative claims.
