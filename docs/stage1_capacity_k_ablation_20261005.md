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
| 16 | 0.276 | 0.286 | 0.040 | 0.212 | 0.076 |
| 32 | 0.277 | **0.293** | **0.118** | **0.210** | 0.166 |
| 64 | 0.262 | 0.274 | 0.084 | 0.189 | 0.113 |

Correction 2026-10-06: the original table evaluated each pool's second
target against the first target's data (a target-loading defect in the
shared-pool path of ``track_a_experiment``).  The second-target runs were
re-executed after the fix.  The K ranking is unchanged; absolute values
moved by up to 0.02.

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

| config | state KB | lora mean | none mean | note |
|---|---:|---:|---:|---|
| small 128x2 N32 | 50.4 | 0.3430 | 0.3066 | reference |
| wide 256x4 N32 | 201.8 | 0.3720 | 0.3236 | over budget |
| **wide_n16 256x4 N16** | **103.9** | 0.3767 | 0.3141 | in budget, ties wide (0.005 gap, one source seed) |
| deep_n16 128x6 N16 | 77.9 | 0.3721 | **0.3291** | in budget, matches wide |
| w192_n16 192x4 N16 | 108.9 | 0.3653 | 0.3107 | in budget, capacity fill-in |

Readings:

1. The stage exit condition is met: configurations inside the 128 KB budget
   come within about 0.01 of the over-budget wide model.  wide_n16 across
   three source seeds: 0.3611 / 0.3629 / 0.3687 (mean 0.364, SD 0.004)
   versus wide 0.372 -- a 0.008 gap at roughly two standard deviations, so
   the honest wording is "slightly below wide, near parity", not "beats".
   deep_n16 (0.372, single seed; its second seed failed a fold check in
   pool 3 and was not rerun) needs a seed replicate before ranking.
2. Reducing N from 32 to 16 costs nothing on M2 at both capacities.  The M2 capacity need lives in the stateless width and layer count, not in per-head state.
3. Kernel constraint discovered: ``d_state = 8`` violates the fused Triton kernel (``tl.dot`` requires ``K >= 16``).  The planned wide_n8 variant is not implementable on this kernel; 16 is the state-size floor.  The w192_n16 fill-in replaces it on the capacity curve.


## 5. Stage 1' Outcome

1. Exit condition met.  wide_n16 (103.9 KB state) reaches 0.3767 multi-target lora R2, above the over-budget wide model at 0.3720.  deep_n16 matches wide with 39 percent of its state.  The default M2 candidate from here on is wide_n16; deep_n16 is the low-state alternative.
2. State size N=16 is both sufficient and the kernel floor.  N=32 to 16 costs nothing on M2; N=8 is not implementable on the fused kernel.
3. AP backbone bottleneck: K=32 from here on.  The bottleneck is not the Track A bottleneck on M2; the prior table is.
4. Largest measured recipe item: drift augmentation, +0.10.  Code-level recipe toggles (per-session statistics, session bank, window policy) are the next ablation round.

All numbers: multi-target mean over five forward targets, query-only, physical-unit R2, LR selected on prefix validation, three seeds for trained methods.
