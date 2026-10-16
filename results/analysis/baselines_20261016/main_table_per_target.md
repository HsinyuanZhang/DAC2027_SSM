### days / sub-M_ses-CO-20150616

| method | B=1: R2 [baseline-P0] | B=8: R2 [baseline-P0] | B=33: R2 [baseline-P0] |
|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | 0.335 [-0.304] | 0.375 [-0.355] | 0.375 [-0.366] |
| P0 (fixed-weight per-unit encoder, level 1) | 0.639 | 0.730 | 0.740 |
| P0 + R1-stream (readout ridge, level 2) | 0.639 [+0.000] | 0.728 [-0.001] | 0.741 [+0.001] |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | 0.642 [+0.003] | 0.731 [+0.001] | 0.744 [+0.004] |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | 0.564 [-0.075] | 0.673 [-0.057] | 0.704 [-0.037] |
| Stabilizer control (pooled): target FA, no rotation | -0.051 [-0.690] | -0.073 [-0.802] | -0.047 [-0.787] |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | 0.715 [+0.076] | 0.716 [-0.014] | 0.713 [-0.027] |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | 0.580 [-0.059] | 0.669 [-0.061] | 0.690 [-0.050] |
| Stabilizer control (sorted units): target FA, no rotation | -0.050 [-0.689] | -0.071 [-0.800] | 0.080 [-0.661] |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | 0.696 [+0.057] | 0.697 [-0.033] | 0.695 [-0.045] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | -0.178 [-0.817] | 0.502 [-0.228] | 0.647 [-0.093] |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | -0.062 [-0.701] | 0.514 [-0.216] | 0.663 [-0.077] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | -0.199 [-0.838] | 0.510 [-0.220] | 0.670 [-0.070] |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | -0.030 [-0.669] | 0.551 [-0.179] | 0.688 [-0.053] |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | -0.232 [-0.871] | 0.567 [-0.163] | 0.694 [-0.046] |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | -0.049 [-0.688] | 0.588 [-0.142] | 0.718 [-0.022] |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | 0.684 [+0.045] | 0.685 [-0.045] | 0.683 [-0.057] |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | 0.695 [+0.056] | 0.696 [-0.034] | 0.693 [-0.048] |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | 0.689 [+0.050] | 0.689 [-0.040] | 0.686 [-0.054] |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | 0.729 [+0.090] | 0.729 [-0.001] | 0.727 [-0.013] |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | 0.715 [+0.076] | 0.716 [-0.014] | 0.713 [-0.027] |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | 0.750 [+0.111] | 0.750 [+0.021] | 0.749 [+0.009] |
| APST-FULL (last segment, source-only rule; single training seed 42) | n/a | 0.465 [-0.264] | 0.593 [-0.147] |
| APST ACT-only (last segment, source-only rule; seed 42) | -0.027 [-0.666] | 0.357 [-0.372] | 0.435 [-0.305] |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | n/a | 0.495 [-0.234] | 0.597 [-0.143] |
| APST ACT-only (authors' dev-selected segment_08; days targets are APST dev sessions, optimistic) | -0.017 [-0.656] | 0.347 [-0.382] | 0.438 [-0.302] |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | 0.458 [-0.181] | 0.457 [-0.272] | 0.454 [-0.286] |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | 0.533 [-0.106] | 0.532 [-0.198] | 0.531 [-0.209] |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | 0.375 [-0.264] | 0.605 [-0.125] | 0.695 [-0.045] |
| P0 + ui_reg (P2 rule) | 0.644 [+0.005] | 0.719 [-0.011] | 0.741 [+0.001] |

### days / sub-M_ses-CO-20150617

| method | B=1: R2 [baseline-P0] | B=8: R2 [baseline-P0] | B=33: R2 [baseline-P0] |
|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | 0.330 [-0.326] | 0.345 [-0.372] | 0.348 [-0.387] |
| P0 (fixed-weight per-unit encoder, level 1) | 0.656 | 0.717 | 0.735 |
| P0 + R1-stream (readout ridge, level 2) | 0.656 [+0.000] | 0.718 [+0.001] | 0.738 [+0.003] |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | 0.661 [+0.005] | 0.725 [+0.007] | 0.750 [+0.015] |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | 0.616 [-0.040] | 0.705 [-0.012] | 0.728 [-0.007] |
| Stabilizer control (pooled): target FA, no rotation | -0.074 [-0.730] | -0.106 [-0.823] | -0.065 [-0.800] |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | 0.730 [+0.074] | 0.729 [+0.012] | 0.731 [-0.004] |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | 0.484 [-0.172] | 0.597 [-0.120] | 0.643 [-0.092] |
| Stabilizer control (sorted units): target FA, no rotation | -0.128 [-0.784] | -0.102 [-0.819] | -0.079 [-0.814] |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | 0.608 [-0.048] | 0.606 [-0.111] | 0.611 [-0.124] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | -0.028 [-0.684] | 0.544 [-0.173] | 0.679 [-0.056] |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | -0.029 [-0.685] | 0.591 [-0.127] | 0.691 [-0.044] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | -0.114 [-0.770] | 0.565 [-0.152] | 0.694 [-0.041] |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | -0.049 [-0.705] | 0.601 [-0.116] | 0.716 [-0.020] |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | -0.055 [-0.711] | 0.590 [-0.127] | 0.722 [-0.013] |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | -0.072 [-0.728] | 0.630 [-0.088] | 0.737 [+0.002] |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | 0.695 [+0.039] | 0.695 [-0.023] | 0.697 [-0.039] |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | 0.724 [+0.068] | 0.723 [+0.005] | 0.726 [-0.009] |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | 0.705 [+0.049] | 0.703 [-0.014] | 0.708 [-0.027] |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | 0.737 [+0.081] | 0.736 [+0.018] | 0.737 [+0.002] |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | 0.728 [+0.073] | 0.727 [+0.010] | 0.733 [-0.003] |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | 0.755 [+0.100] | 0.754 [+0.037] | 0.757 [+0.022] |
| APST-FULL (last segment, source-only rule; single training seed 42) | n/a | 0.536 [-0.182] | 0.587 [-0.149] |
| APST ACT-only (last segment, source-only rule; seed 42) | 0.078 [-0.578] | 0.323 [-0.395] | 0.438 [-0.297] |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | n/a | 0.537 [-0.180] | 0.578 [-0.157] |
| APST ACT-only (authors' dev-selected segment_08; days targets are APST dev sessions, optimistic) | 0.098 [-0.558] | 0.330 [-0.388] | 0.420 [-0.315] |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | 0.457 [-0.199] | 0.455 [-0.263] | 0.462 [-0.274] |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | 0.497 [-0.159] | 0.495 [-0.222] | 0.500 [-0.235] |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | 0.380 [-0.276] | 0.610 [-0.107] | 0.713 [-0.022] |
| P0 + ui_reg (P2 rule) | 0.664 [+0.008] | 0.721 [+0.004] | 0.750 [+0.015] |

### month / sub-M_ses-CO-20140626

| method | B=1: R2 [baseline-P0] | B=8: R2 [baseline-P0] | B=33: R2 [baseline-P0] |
|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | 0.315 [-0.151] | 0.332 [-0.233] | 0.333 [-0.233] |
| P0 (fixed-weight per-unit encoder, level 1) | 0.466 | 0.565 | 0.566 |
| P0 + R1-stream (readout ridge, level 2) | 0.472 [+0.005] | 0.576 [+0.012] | 0.606 [+0.040] |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | 0.492 [+0.026] | 0.630 [+0.065] | 0.679 [+0.113] |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | 0.154 [-0.312] | 0.295 [-0.270] | 0.304 [-0.263] |
| Stabilizer control (pooled): target FA, no rotation | -0.108 [-0.575] | -0.112 [-0.677] | -0.038 [-0.605] |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | 0.376 [-0.090] | 0.376 [-0.188] | 0.376 [-0.190] |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | 0.150 [-0.316] | 0.249 [-0.315] | 0.252 [-0.315] |
| Stabilizer control (sorted units): target FA, no rotation | -0.121 [-0.587] | -0.206 [-0.771] | -0.187 [-0.754] |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | 0.233 [-0.233] | 0.233 [-0.332] | 0.234 [-0.332] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | -0.013 [-0.480] | 0.305 [-0.259] | 0.365 [-0.202] |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | 0.068 [-0.398] | 0.378 [-0.187] | 0.478 [-0.089] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | 0.032 [-0.435] | 0.316 [-0.249] | 0.390 [-0.177] |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | 0.069 [-0.397] | 0.463 [-0.101] | 0.611 [+0.044] |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | 0.038 [-0.429] | 0.350 [-0.215] | 0.458 [-0.108] |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | 0.068 [-0.398] | 0.491 [-0.074] | 0.626 [+0.059] |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | 0.422 [-0.044] | 0.422 [-0.143] | 0.422 [-0.144] |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | 0.466 [-0.001] | 0.466 [-0.099] | 0.466 [-0.101] |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | 0.335 [-0.131] | 0.335 [-0.230] | 0.336 [-0.231] |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | 0.646 [+0.180] | 0.646 [+0.081] | 0.646 [+0.079] |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | 0.363 [-0.103] | 0.364 [-0.201] | 0.364 [-0.202] |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | 0.659 [+0.193] | 0.659 [+0.094] | 0.659 [+0.093] |
| APST-FULL (last segment, source-only rule; single training seed 42) | n/a | 0.271 [-0.294] | 0.376 [-0.191] |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | n/a | 0.290 [-0.275] | 0.387 [-0.180] |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | 0.230 [-0.237] | 0.230 [-0.335] | 0.227 [-0.339] |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | 0.101 [-0.365] | 0.101 [-0.464] | 0.099 [-0.467] |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | 0.391 [-0.075] | 0.632 [+0.067] | 0.720 [+0.153] |
| P0 + ui_reg (P2 rule) | 0.487 [+0.021] | 0.653 [+0.088] | 0.731 [+0.165] |
| P0 + ui_reg (like-for-like: month oosc / year oos) | 0.489 [+0.023] | 0.623 [+0.058] | 0.731 [+0.165] |

### year / sub-M_ses-CO-20150511

| method | B=1: R2 [baseline-P0] | B=8: R2 [baseline-P0] | B=33: R2 [baseline-P0] |
|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | 0.145 [-0.054] | 0.133 [-0.122] | 0.150 [-0.165] |
| P0 (fixed-weight per-unit encoder, level 1) | 0.198 | 0.256 | 0.315 |
| P0 + R1-stream (readout ridge, level 2) | 0.199 [+0.000] | 0.254 [-0.002] | 0.376 [+0.060] |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | 0.219 [+0.021] | 0.334 [+0.078] | 0.435 [+0.119] |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | -0.054 [-0.252] | -0.004 [-0.260] | 0.034 [-0.282] |
| Stabilizer control (pooled): target FA, no rotation | -0.063 [-0.262] | -0.079 [-0.334] | -0.051 [-0.367] |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | 0.079 [-0.119] | 0.079 [-0.177] | 0.087 [-0.229] |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | -0.084 [-0.283] | -0.036 [-0.292] | -0.047 [-0.362] |
| Stabilizer control (sorted units): target FA, no rotation | -0.138 [-0.336] | -0.066 [-0.321] | -0.091 [-0.407] |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | -0.022 [-0.220] | -0.022 [-0.277] | -0.024 [-0.339] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | -0.011 [-0.210] | 0.101 [-0.154] | 0.181 [-0.134] |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | -0.008 [-0.206] | 0.164 [-0.092] | 0.243 [-0.073] |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | 0.030 [-0.168] | 0.155 [-0.101] | 0.243 [-0.073] |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | 0.019 [-0.179] | 0.250 [-0.005] | 0.411 [+0.096] |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | -0.017 [-0.215] | 0.195 [-0.061] | 0.323 [+0.007] |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | 0.013 [-0.185] | 0.276 [+0.020] | 0.427 [+0.111] |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | 0.175 [-0.023] | 0.174 [-0.081] | 0.193 [-0.122] |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | 0.221 [+0.023] | 0.220 [-0.035] | 0.244 [-0.072] |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | 0.317 [+0.119] | 0.316 [+0.061] | 0.349 [+0.034] |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | 0.406 [+0.208] | 0.405 [+0.149] | 0.447 [+0.132] |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | 0.270 [+0.072] | 0.269 [+0.014] | 0.298 [-0.017] |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | 0.412 [+0.214] | 0.411 [+0.155] | 0.454 [+0.138] |
| APST-FULL (last segment, source-only rule; single training seed 42) | n/a | 0.150 [-0.106] | 0.213 [-0.103] |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | n/a | 0.189 [-0.066] | 0.257 [-0.058] |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | -0.378 [-0.576] | -0.376 [-0.632] | -0.414 [-0.729] |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | -0.216 [-0.414] | -0.215 [-0.471] | -0.237 [-0.553] |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | 0.191 [-0.008] | 0.290 [+0.035] | 0.478 [+0.163] |
| P0 + ui_reg (P2 rule) | 0.228 [+0.029] | 0.309 [+0.054] | 0.488 [+0.172] |
| P0 + ui_reg (like-for-like: month oosc / year oos) | 0.230 [+0.031] | 0.348 [+0.092] | 0.492 [+0.177] |
