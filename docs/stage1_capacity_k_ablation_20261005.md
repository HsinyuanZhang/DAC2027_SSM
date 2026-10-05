# Stage 1': Capacity, State Budget, and the AP Bottleneck K

Date: 2026-10-05. Status: K sweep complete; C10 variants and the recipe ablation running at the time of writing. Numbers for those two are appended as they land.

## 1. State-Budget Accounting (C10)

`ssm_decode/state_budget.py` counts carried values from the verified Mamba-3 carry layout (angle cumsum, SSM state, K and V carry) at INT24.  Anchor: 256x4 at state 32 gives 201.8 KB, matching the roadmap figure.

Variants inside the 128 KB budget:

| variant | width x layers x state | state KB |
|---|---|---:|
| small (reference) | 128 x 2 x 32 | 50.4 |
| wide (reference, over budget) | 256 x 4 x 32 | 201.8 |
| wide_n16 | 256 x 4 x 16 | 103.9 |
| deep_n16 | 128 x 6 x 16 | 77.9 |
| wide_n8 | 256 x 4 x 8 | 54.9 |

Each variant runs the full forward protocol (none and lora, three seeds, two LRs) on M2.

## 2. AP Bottleneck K Sweep (revision 4.4.3)

One backbone per (K, forward pool), Track A evaluation on every pool target.  Five targets total.  All revision fixes (whitened frame, contiguous decode, frozen-base UI, continuous lags) are active, so these numbers supersede the 2026-10-04 rotation table.

Multi-target means:

| K | mean_table b=0 | mean_table b=33 | am1 b=33 | ui_grad b=33 | oracle b=33 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.268 | 0.299 | 0.058 | 0.223 | 0.077 |
| 32 | 0.282 | **0.312** | **0.086** | **0.237** | 0.132 |
| 64 | 0.265 | 0.290 | 0.077 | 0.211 | 0.106 |

Findings:

1. K = 32 is the optimum on every method, but the K curve is flat (0.29 to 0.31 for the mean table).  The bottleneck is not what costs Track A its gap on M2.
2. The whitened-frame fixes moved am1 at 33 trials from about 0.02 to 0.05 to 0.086, and the oracle bound from 0.097 to 0.132.  The frame is no longer the dominant defect.  The remaining gap to the mean table (about 0.22) is the from-scratch table itself, consistent with revision 3.1: on fixed channels the prior table already carries the information.
3. am2 is unstable on the two-source pool: one target collapses to -0.97 at K = 32.  Profile shrinkage needs more source sessions than pool 1 has.  The 000688 pools (six sources) are the right arena.
4. Source-pool size shows a clear gradient: closed-form methods gain about 0.1 to 0.2 from the two-source pool to the six-source pool.

Decision: K = 32 for the AP backbone from here on.

## 3. Recipe Ablation (revision 4.4.2, arg-only item)

M2 small capacity on the forward protocol, full recipe versus no drift augmentation (gain 0, offset 0, channel dropout 0).  Multi-target means:

| config | lora | none |
|---|---:|---:|
| small full recipe | 0.3430 | 0.3066 |
| small no augmentation | 0.2423 | 0.2141 |
| **augmentation contribution** | **+0.101** | **+0.093** |

Drift augmentation is the largest single recipe item measured so far: it adds about 0.1 multi-target R2 on M2 at both adaptation levels.  The gap is largest on the two earliest targets (0.10 to 0.18).  Remaining recipe items (per-session statistics, session bank, window policy, regularization) need code-level toggles and are deferred to the next round.

## 4. C10 Variant Results

Capacity references on the same protocol: small lora 0.3430, wide lora 0.3720 (state 201.8 KB, over budget).

(pending; wide_n16 on GPU 1, deep_n16 on GPU 0, wide_n8 chained)
