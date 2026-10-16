### days

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.333±0.025 | 0.360±0.018 | 0.360±0.023 | 0.362±0.011 | 0.361±0.012 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.647±0.021 | 0.708±0.009 | 0.723±0.009 | 0.732±0.005 | 0.738±0.005 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.648±0.021 | 0.710±0.009 | 0.723±0.010 | 0.732±0.005 | 0.740±0.005 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.651±0.020 | 0.714±0.009 | 0.728±0.010 | 0.737±0.006 | 0.747±0.006 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.643±0.040 | 0.713±0.022 | 0.715±0.014 | 0.731±0.007 | 0.745±0.008 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.056±0.050 | -0.046±0.042 | -0.085±0.039 | -0.072±0.041 | -0.068±0.028 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.748±0.000 | 0.748±0.003 | 0.747±0.001 | 0.747±0.003 | 0.746±0.004 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.528±0.035 | 0.634±0.013 | 0.633±0.023 | 0.652±0.011 | 0.667±0.025 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.095±0.061 | -0.191±0.091 | -0.086±0.136 | -0.021±0.152 | 0.000±0.065 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.652±0.001 | 0.653±0.002 | 0.652±0.002 | 0.652±0.004 | 0.653±0.006 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.132±0.177 | 0.358±0.042 | 0.533±0.031 | 0.612±0.014 | 0.674±0.017 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.001±0.083 | 0.418±0.050 | 0.568±0.035 | 0.632±0.017 | 0.688±0.015 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | -0.123±0.133 | 0.341±0.066 | 0.538±0.037 | 0.615±0.026 | 0.682±0.008 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.056±0.149 | 0.415±0.039 | 0.576±0.052 | 0.645±0.018 | 0.702±0.008 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | -0.095±0.169 | 0.383±0.075 | 0.578±0.038 | 0.644±0.025 | 0.708±0.012 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.026±0.122 | 0.448±0.045 | 0.609±0.058 | 0.666±0.017 | 0.727±0.007 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.697±0.000 | 0.698±0.003 | 0.697±0.001 | 0.697±0.003 | 0.696±0.006 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.719±0.000 | 0.720±0.003 | 0.719±0.001 | 0.718±0.003 | 0.718±0.006 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.697±0.000 | 0.699±0.003 | 0.696±0.002 | 0.696±0.002 | 0.697±0.005 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.733±0.000 | 0.734±0.003 | 0.732±0.001 | 0.731±0.002 | 0.732±0.004 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.722±0.000 | 0.723±0.003 | 0.721±0.002 | 0.720±0.001 | 0.723±0.005 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.753±0.001 | 0.754±0.003 | 0.752±0.002 | 0.750±0.002 | 0.753±0.004 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.279±0.118 | 0.500±0.059 | 0.568±0.011 | 0.590±0.009 |
| APST ACT-only (last segment, source-only rule; seed 42) | B trials: neural only (no labels) | no | 0.026±0.068 | 0.178±0.074 | 0.340±0.070 | 0.393±0.083 | 0.437±0.103 |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.311±0.119 | 0.516±0.043 | 0.567±0.006 | 0.588±0.009 |
| APST ACT-only (authors' dev-selected segment_08; days targets are APST dev sessions, optimistic) | B trials: neural only (no labels) | no | 0.041±0.076 | 0.196±0.086 | 0.338±0.080 | 0.396±0.093 | 0.429±0.110 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.457±0.001 | 0.459±0.003 | 0.456±0.002 | 0.455±0.004 | 0.458±0.007 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.515±0.001 | 0.516±0.004 | 0.514±0.002 | 0.514±0.005 | 0.515±0.007 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.378±0.028 | 0.527±0.039 | 0.607±0.038 | 0.661±0.020 | 0.704±0.015 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.654±0.019 | 0.701±0.015 | 0.720±0.015 | 0.730±0.012 | 0.746±0.008 |

### month

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.315±0.046 | 0.319±0.021 | 0.332±0.010 | 0.328±0.007 | 0.333±0.004 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.466±0.034 | 0.535±0.014 | 0.565±0.011 | 0.558±0.007 | 0.566±0.006 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.472±0.037 | 0.542±0.014 | 0.576±0.011 | 0.588±0.007 | 0.606±0.008 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.492±0.048 | 0.591±0.028 | 0.630±0.011 | 0.652±0.014 | 0.679±0.007 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.154±0.060 | 0.237±0.041 | 0.295±0.030 | 0.334±0.026 | 0.304±0.067 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.108±0.052 | -0.083±0.090 | -0.112±0.086 | -0.088±0.083 | -0.038±0.047 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.376±0.000 | 0.376±0.000 | 0.376±0.001 | 0.376±0.001 | 0.376±0.001 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.263±0.077 | 0.251±0.055 | 0.267±0.044 | 0.283±0.035 | 0.280±0.021 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.064±0.045 | -0.104±0.099 | -0.123±0.103 | -0.115±0.198 | -0.132±0.137 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.290±0.000 | 0.290±0.001 | 0.290±0.001 | 0.291±0.001 | 0.291±0.002 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.013±0.169 | 0.196±0.046 | 0.305±0.043 | 0.374±0.025 | 0.365±0.065 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.063±0.065 | 0.278±0.056 | 0.378±0.025 | 0.430±0.020 | 0.478±0.017 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | 0.032±0.071 | 0.230±0.042 | 0.316±0.053 | 0.356±0.030 | 0.390±0.039 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.047±0.054 | 0.349±0.089 | 0.495±0.046 | 0.583±0.022 | 0.646±0.020 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | 0.038±0.083 | 0.239±0.077 | 0.350±0.038 | 0.435±0.039 | 0.458±0.026 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.060±0.059 | 0.376±0.089 | 0.512±0.050 | 0.597±0.020 | 0.656±0.017 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.422±0.000 | 0.422±0.001 | 0.422±0.001 | 0.422±0.001 | 0.422±0.001 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.466±0.000 | 0.466±0.001 | 0.466±0.001 | 0.466±0.001 | 0.466±0.001 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.335±0.000 | 0.335±0.000 | 0.335±0.001 | 0.335±0.001 | 0.336±0.002 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.689±0.000 | 0.689±0.000 | 0.689±0.001 | 0.690±0.001 | 0.689±0.001 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.363±0.000 | 0.364±0.001 | 0.364±0.001 | 0.363±0.001 | 0.364±0.001 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.698±0.000 | 0.698±0.000 | 0.698±0.000 | 0.699±0.001 | 0.698±0.001 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.219±0.071 (n=5) | 0.271±0.056 (n=7) | 0.324±0.054 | 0.376±0.027 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.240±0.040 (n=5) | 0.290±0.062 (n=7) | 0.342±0.064 | 0.387±0.029 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.230±0.000 | 0.229±0.001 | 0.230±0.001 | 0.230±0.001 | 0.227±0.004 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.101±0.000 | 0.101±0.001 | 0.101±0.002 | 0.101±0.002 | 0.099±0.004 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.391±0.038 | 0.542±0.039 | 0.632±0.026 | 0.683±0.011 | 0.720±0.010 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.487±0.033 | 0.612±0.030 | 0.653±0.022 | 0.709±0.003 | 0.731±0.008 |
| P0 + ui_reg (like-for-like: month oosc / year oos) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.489±0.035 | 0.612±0.030 | 0.623±0.016 | 0.699±0.018 | 0.731±0.009 |

### year

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.145±0.017 | 0.135±0.014 | 0.133±0.011 | 0.132±0.004 | 0.150±0.022 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.198±0.018 | 0.263±0.027 | 0.256±0.024 | 0.286±0.009 | 0.315±0.052 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.199±0.018 | 0.261±0.026 | 0.254±0.026 | 0.304±0.011 | 0.376±0.061 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.219±0.029 | 0.325±0.033 | 0.334±0.017 | 0.386±0.013 | 0.435±0.072 |
| Stabilizer (a), electrode-pooled input (sorted spikes summed per electrode): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.054±0.055 | -0.023±0.057 | -0.004±0.059 | 0.004±0.067 | 0.034±0.040 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.063±0.074 | -0.028±0.027 | -0.079±0.078 | -0.086±0.068 | -0.051±0.168 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.079±0.000 | 0.079±0.000 | 0.079±0.001 | 0.078±0.001 | 0.087±0.014 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.020±0.050 | 0.060±0.028 | 0.061±0.021 | 0.055±0.016 | 0.070±0.019 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.073±0.081 | -0.089±0.082 | -0.012±0.051 | -0.083±0.062 | -0.077±0.109 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.055±0.000 | 0.055±0.001 | 0.055±0.001 | 0.054±0.001 | 0.061±0.011 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.011±0.040 | 0.094±0.024 | 0.101±0.047 | 0.172±0.018 | 0.181±0.037 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.003±0.079 | 0.119±0.028 | 0.164±0.033 | 0.198±0.022 | 0.243±0.046 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | 0.030±0.072 | 0.113±0.030 | 0.155±0.050 | 0.209±0.028 | 0.243±0.053 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.033±0.068 | 0.203±0.054 | 0.265±0.025 | 0.334±0.018 | 0.427±0.074 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | -0.017±0.068 | 0.136±0.044 | 0.195±0.045 | 0.277±0.025 | 0.323±0.051 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.023±0.062 | 0.227±0.055 | 0.289±0.030 | 0.352±0.017 | 0.439±0.076 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.175±0.000 | 0.175±0.001 | 0.174±0.001 | 0.173±0.001 | 0.193±0.033 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.221±0.000 | 0.221±0.000 | 0.220±0.001 | 0.219±0.001 | 0.244±0.041 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.317±0.000 | 0.317±0.001 | 0.316±0.001 | 0.314±0.001 | 0.349±0.059 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.412±0.000 | 0.412±0.000 | 0.411±0.001 | 0.408±0.000 | 0.454±0.077 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.270±0.000 | 0.270±0.000 | 0.269±0.001 | 0.267±0.001 | 0.298±0.052 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.412±0.000 | 0.411±0.000 | 0.410±0.001 | 0.408±0.000 | 0.454±0.077 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.105±0.039 (n=7) | 0.150±0.034 | 0.189±0.016 | 0.213±0.047 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.149±0.042 (n=7) | 0.189±0.025 | 0.234±0.018 | 0.257±0.050 |
| NoMAD, electrode-pooled input (sorted spikes summed per electrode; mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.378±0.000 | -0.375±0.001 | -0.376±0.003 | -0.374±0.002 | -0.414±0.072 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.216±0.000 | -0.215±0.001 | -0.215±0.003 | -0.214±0.002 | -0.237±0.043 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.191±0.021 | 0.267±0.025 | 0.290±0.046 | 0.367±0.007 | 0.478±0.088 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.228±0.023 | 0.321±0.030 | 0.309±0.029 | 0.393±0.011 | 0.488±0.086 |
| P0 + ui_reg (like-for-like: month oosc / year oos) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.230±0.026 | 0.345±0.040 | 0.348±0.017 | 0.405±0.009 | 0.492±0.087 |
