Headline: query R2 (physical units, our query protocol), mean ± 95% CI over 8 draws (draw variance only).

### days

| method | H4 | B=1 | B=8 | B=33 |
|---|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | yes | 0.333±0.025 | 0.360±0.023 | 0.361±0.012 |
| P0 (fixed-weight per-unit encoder, level 1) | yes | 0.647±0.021 | 0.723±0.009 | 0.738±0.005 |
| P0 + R1-stream (readout ridge, level 2) | yes | 0.648±0.021 | 0.723±0.010 | 0.740±0.005 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | yes | 0.651±0.020 | 0.728±0.010 | 0.747±0.006 |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | yes | 0.590±0.048 | 0.689±0.014 | 0.716±0.009 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | yes | 0.532±0.040 | 0.633±0.023 | 0.667±0.025 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | yes | -0.103±0.126 | 0.523±0.029 | 0.663±0.018 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | yes | -0.157±0.168 | 0.538±0.037 | 0.682±0.008 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | yes | -0.046±0.090 | 0.552±0.042 | 0.677±0.014 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | yes | -0.040±0.159 | 0.576±0.052 | 0.702±0.008 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | yes | -0.061±0.144 | 0.609±0.058 | 0.727±0.007 |
| APST-FULL (last segment, source-only rule; single training seed 42) | yes | n/a | 0.500±0.059 | 0.590±0.009 |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | yes | n/a | 0.516±0.043 | 0.588±0.009 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | NO (violates H4) | 0.722±0.000 | 0.722±0.001 | 0.722±0.004 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | NO (violates H4) | 0.709±0.000 | 0.709±0.001 | 0.709±0.005 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | NO (violates H4) | 0.753±0.001 | 0.752±0.002 | 0.753±0.004 |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | NO (violates H4) | 0.457±0.001 | 0.456±0.002 | 0.458±0.007 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | NO (violates H4) | 0.515±0.001 | 0.514±0.002 | 0.515±0.007 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | yes | 0.378±0.028 | 0.607±0.038 | 0.704±0.015 |
| P0 + ui_reg (P2 rule) | yes | 0.654±0.019 | 0.720±0.015 | 0.746±0.008 |

### month

| method | H4 | B=1 | B=8 | B=33 |
|---|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | yes | 0.315±0.046 | 0.332±0.010 | 0.333±0.004 |
| P0 (fixed-weight per-unit encoder, level 1) | yes | 0.466±0.034 | 0.565±0.011 | 0.566±0.006 |
| P0 + R1-stream (readout ridge, level 2) | yes | 0.472±0.037 | 0.576±0.011 | 0.606±0.008 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | yes | 0.492±0.048 | 0.630±0.011 | 0.679±0.007 |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | yes | 0.154±0.060 | 0.295±0.030 | 0.304±0.067 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | yes | 0.150±0.057 | 0.249±0.019 | 0.252±0.011 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | yes | -0.013±0.169 | 0.305±0.043 | 0.365±0.065 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | yes | 0.032±0.071 | 0.316±0.053 | 0.390±0.039 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | yes | 0.068±0.061 | 0.378±0.025 | 0.478±0.017 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | yes | 0.069±0.042 | 0.463±0.043 | 0.611±0.021 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | yes | 0.068±0.086 | 0.491±0.047 | 0.626±0.018 |
| APST-FULL (last segment, source-only rule; single training seed 42) | yes | n/a | 0.271±0.056 (n=7) | 0.376±0.027 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | yes | n/a | 0.290±0.062 (n=7) | 0.387±0.029 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | NO (violates H4) | 0.376±0.000 | 0.376±0.001 | 0.376±0.001 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | NO (violates H4) | 0.466±0.000 | 0.466±0.001 | 0.466±0.001 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | NO (violates H4) | 0.659±0.000 | 0.659±0.001 | 0.659±0.002 |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | NO (violates H4) | 0.230±0.000 | 0.230±0.001 | 0.227±0.004 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | NO (violates H4) | 0.101±0.000 | 0.101±0.002 | 0.099±0.004 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | yes | 0.391±0.038 | 0.632±0.026 | 0.720±0.010 |
| P0 + ui_reg (P2 rule) | yes | 0.487±0.033 | 0.653±0.022 | 0.731±0.008 |

### year

| method | H4 | B=1 | B=8 | B=33 |
|---|---|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | yes | 0.145±0.017 | 0.133±0.011 | 0.150±0.022 |
| P0 (fixed-weight per-unit encoder, level 1) | yes | 0.198±0.018 | 0.256±0.024 | 0.315±0.052 |
| P0 + R1-stream (readout ridge, level 2) | yes | 0.199±0.018 | 0.254±0.026 | 0.376±0.061 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | yes | 0.219±0.029 | 0.334±0.017 | 0.435±0.072 |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | yes | -0.054±0.055 | -0.004±0.059 | 0.034±0.040 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | yes | -0.084±0.036 | -0.036±0.039 | -0.047±0.024 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | yes | -0.011±0.040 | 0.101±0.047 | 0.181±0.037 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | yes | 0.030±0.072 | 0.155±0.050 | 0.243±0.053 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | yes | -0.008±0.092 | 0.164±0.033 | 0.243±0.046 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | yes | 0.019±0.080 | 0.250±0.023 | 0.411±0.069 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | yes | 0.013±0.076 | 0.276±0.031 | 0.427±0.072 |
| APST-FULL (last segment, source-only rule; single training seed 42) | yes | n/a | 0.150±0.034 | 0.213±0.047 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | yes | n/a | 0.189±0.025 | 0.257±0.050 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | NO (violates H4) | 0.079±0.000 | 0.079±0.001 | 0.087±0.014 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | NO (violates H4) | 0.221±0.000 | 0.220±0.001 | 0.244±0.041 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | NO (violates H4) | 0.412±0.000 | 0.411±0.001 | 0.454±0.077 |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | NO (violates H4) | -0.378±0.000 | -0.376±0.003 | -0.414±0.072 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | NO (violates H4) | -0.216±0.000 | -0.215±0.003 | -0.237±0.043 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | yes | 0.191±0.021 | 0.290±0.046 | 0.478±0.088 |
| P0 + ui_reg (P2 rule) | yes | 0.228±0.023 | 0.309±0.029 | 0.488±0.086 |
