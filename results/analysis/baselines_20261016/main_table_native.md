### days

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.250±0.029 | 0.275±0.021 | 0.273±0.026 | 0.273±0.011 | 0.272±0.014 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.643±0.026 | 0.712±0.012 | 0.728±0.011 | 0.737±0.006 | 0.744±0.007 |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.565±0.053 | 0.655±0.025 | 0.671±0.017 | 0.691±0.008 | 0.701±0.013 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.083±0.085 | -0.068±0.070 | -0.113±0.118 | -0.049±0.102 | -0.069±0.084 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.708±0.001 | 0.708±0.004 | 0.708±0.002 | 0.707±0.003 | 0.707±0.006 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.501±0.037 | 0.607±0.016 | 0.611±0.026 | 0.628±0.010 | 0.646±0.028 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.088±0.067 | -0.211±0.108 | -0.086±0.126 | -0.009±0.132 | -0.007±0.064 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.621±0.001 | 0.622±0.003 | 0.620±0.003 | 0.621±0.005 | 0.624±0.008 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.147±0.158 | 0.279±0.043 | 0.471±0.036 | 0.554±0.012 | 0.626±0.022 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.092±0.099 | 0.347±0.060 | 0.499±0.048 | 0.571±0.025 | 0.643±0.018 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | -0.194±0.202 | 0.266±0.070 | 0.487±0.043 | 0.569±0.034 | 0.649±0.009 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.089±0.186 | 0.349±0.039 | 0.529±0.061 | 0.607±0.023 | 0.672±0.011 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | -0.183±0.151 | 0.321±0.082 | 0.543±0.045 | 0.610±0.032 | 0.689±0.013 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.102±0.161 | 0.395±0.048 | 0.572±0.069 | 0.641±0.023 | 0.710±0.011 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.662±0.000 | 0.664±0.004 | 0.663±0.001 | 0.663±0.004 | 0.663±0.007 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.678±0.000 | 0.680±0.004 | 0.678±0.001 | 0.677±0.005 | 0.678±0.007 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.676±0.001 | 0.678±0.004 | 0.675±0.003 | 0.674±0.002 | 0.677±0.007 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.710±0.000 | 0.712±0.005 | 0.709±0.002 | 0.708±0.002 | 0.710±0.005 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.714±0.000 | 0.716±0.004 | 0.713±0.002 | 0.711±0.002 | 0.716±0.007 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.739±0.001 | 0.741±0.005 | 0.738±0.002 | 0.735±0.002 | 0.740±0.006 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.393±0.163 | 0.700±0.083 | 0.789±0.012 | 0.814±0.011 |
| APST ACT-only (last segment, source-only rule; seed 42) | B trials: neural only (no labels) | no | 0.041±0.094 | 0.257±0.096 | 0.476±0.100 | 0.546±0.116 | 0.606±0.138 |
| APST-FULL (authors' dev-selected segment_09; days targets ARE APST's dev sessions, so optimistic) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.449±0.164 | 0.731±0.059 | 0.799±0.010 | 0.822±0.010 |
| APST ACT-only (authors' dev-selected segment_08; days targets are APST dev sessions, optimistic) | B trials: neural only (no labels) | no | 0.062±0.105 | 0.290±0.111 | 0.477±0.115 | 0.560±0.128 | 0.607±0.147 |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.468±0.001 | 0.469±0.004 | 0.466±0.002 | 0.466±0.004 | 0.469±0.009 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.544±0.001 | 0.545±0.005 | 0.543±0.002 | 0.543±0.005 | 0.546±0.011 |

### month

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.293±0.051 | 0.297±0.021 | 0.312±0.011 | 0.308±0.011 | 0.311±0.005 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.463±0.043 | 0.538±0.017 | 0.574±0.011 | 0.567±0.009 | 0.571±0.009 |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.146±0.073 | 0.234±0.041 | 0.303±0.041 | 0.353±0.032 | 0.310±0.079 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.132±0.059 | -0.119±0.113 | -0.150±0.102 | -0.122±0.097 | -0.049±0.047 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.410±0.000 | 0.410±0.001 | 0.410±0.001 | 0.411±0.001 | 0.410±0.002 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.129±0.083 | 0.217±0.056 | 0.250±0.024 | 0.257±0.017 | 0.248±0.014 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.147±0.077 | -0.204±0.100 | -0.252±0.099 | -0.179±0.116 | -0.225±0.124 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.231±0.000 | 0.231±0.001 | 0.231±0.002 | 0.231±0.002 | 0.232±0.002 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.027±0.186 | 0.209±0.055 | 0.332±0.045 | 0.408±0.022 | 0.390±0.076 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.061±0.070 | 0.287±0.067 | 0.399±0.034 | 0.442±0.025 | 0.488±0.023 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | 0.021±0.073 | 0.245±0.048 | 0.335±0.065 | 0.384±0.027 | 0.421±0.037 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.065±0.056 | 0.345±0.098 | 0.501±0.041 | 0.566±0.028 | 0.650±0.026 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | 0.027±0.099 | 0.256±0.097 | 0.373±0.052 | 0.473±0.042 | 0.492±0.031 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.063±0.098 | 0.375±0.096 | 0.537±0.044 | 0.592±0.031 | 0.671±0.020 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.464±0.000 | 0.465±0.001 | 0.465±0.001 | 0.464±0.001 | 0.465±0.002 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.461±0.000 | 0.461±0.001 | 0.462±0.001 | 0.462±0.001 | 0.461±0.002 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.375±0.000 | 0.375±0.001 | 0.375±0.001 | 0.375±0.001 | 0.375±0.002 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.671±0.000 | 0.671±0.001 | 0.670±0.001 | 0.671±0.001 | 0.671±0.002 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.362±0.000 | 0.362±0.001 | 0.362±0.001 | 0.362±0.002 | 0.363±0.002 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.690±0.000 | 0.690±0.001 | 0.690±0.001 | 0.691±0.001 | 0.690±0.002 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.284±0.080 (n=5) | 0.338±0.075 (n=7) | 0.394±0.072 | 0.466±0.037 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.306±0.052 (n=5) | 0.357±0.081 (n=7) | 0.417±0.084 | 0.477±0.038 |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.234±0.000 | 0.234±0.001 | 0.234±0.001 | 0.234±0.002 | 0.231±0.004 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | 0.067±0.000 | 0.067±0.001 | 0.067±0.002 | 0.068±0.003 | 0.065±0.005 |

### year

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.187±0.019 | 0.186±0.016 | 0.180±0.014 | 0.183±0.006 | 0.187±0.007 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.273±0.031 | 0.378±0.047 | 0.361±0.044 | 0.416±0.017 | 0.415±0.019 |
| Stabilizer (a), electrode-pooled 96-ch input: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.097±0.083 | -0.064±0.093 | -0.020±0.085 | -0.010±0.107 | 0.023±0.068 |
| Stabilizer control (pooled): target FA, no rotation | B trials: neural only (no labels) | no | -0.085±0.082 | -0.040±0.042 | -0.137±0.121 | -0.129±0.101 | -0.071±0.226 |
| Stabilizer (b), pooled: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.109±0.000 | 0.109±0.001 | 0.108±0.002 | 0.109±0.002 | 0.109±0.003 |
| Stabilizer (a), sorted units + ad-hoc same-electrode unit matching: FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.168±0.053 | -0.087±0.056 | -0.086±0.061 | -0.078±0.028 | -0.103±0.040 |
| Stabilizer control (sorted units): target FA, no rotation | B trials: neural only (no labels) | no | -0.231±0.153 | -0.127±0.059 | -0.100±0.038 | -0.157±0.092 | -0.116±0.100 |
| Stabilizer (b), sorted units: FA + Procrustes on stable channels, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | -0.054±0.001 | -0.054±0.001 | -0.053±0.002 | -0.055±0.003 | -0.053±0.002 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, electrode-pooled input | B trials: neural + reach-direction labels | no | -0.024±0.065 | 0.155±0.036 | 0.164±0.074 | 0.272±0.027 | 0.265±0.022 |
| PCA+CCA (a), pooled input: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | -0.015±0.149 | 0.193±0.034 | 0.253±0.051 | 0.302±0.030 | 0.335±0.011 |
| Gallego-style PCA+CCA (a): sqrt, plain CCA, sorted-unit input | B trials: neural + reach-direction labels | no | 0.047±0.118 | 0.191±0.050 | 0.249±0.071 | 0.337±0.040 | 0.363±0.042 |
| PCA+CCA (a), sorted units: sqrt, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.026±0.128 | 0.280±0.079 | 0.377±0.038 | 0.485±0.030 | 0.564±0.013 |
| PCA+CCA (a), sorted units: raw counts, plain CCA [our variant] | B trials: neural + reach-direction labels | no | -0.044±0.121 | 0.215±0.072 | 0.304±0.065 | 0.432±0.039 | 0.455±0.035 |
| PCA+CCA (a), sorted units: raw counts, ridge CCA [our variant, source-selected] | B trials: neural + reach-direction labels | no | 0.012±0.127 | 0.316±0.088 | 0.417±0.046 | 0.523±0.028 | 0.585±0.008 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, pooled input, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.292±0.000 | 0.292±0.001 | 0.292±0.001 | 0.291±0.001 | 0.293±0.002 |
| PCA+CCA (b), pooled: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.335±0.000 | 0.335±0.001 | 0.335±0.001 | 0.334±0.002 | 0.335±0.002 |
| Gallego-style PCA+CCA (b): sqrt, plain CCA, sorted units, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.474±0.000 | 0.474±0.001 | 0.475±0.001 | 0.472±0.001 | 0.474±0.003 |
| PCA+CCA (b), sorted units: sqrt, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.610±0.000 | 0.611±0.001 | 0.610±0.002 | 0.610±0.001 | 0.611±0.002 |
| PCA+CCA (b), sorted units: raw counts, plain CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.440±0.000 | 0.441±0.001 | 0.441±0.002 | 0.438±0.002 | 0.441±0.002 |
| PCA+CCA (b), sorted units: raw counts, ridge CCA [our variant], all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.622±0.000 | 0.622±0.001 | 0.622±0.002 | 0.621±0.001 | 0.622±0.001 |
| APST-FULL (last segment, source-only rule; single training seed 42) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.200±0.072 (n=7) | 0.283±0.059 | 0.365±0.027 | 0.359±0.032 |
| APST-FULL (2014 retrain, seed 42; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.284±0.081 (n=7) | 0.352±0.048 | 0.444±0.032 | 0.433±0.027 |
| NoMAD, electrode-pooled 96-ch input (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.701±0.001 | -0.699±0.003 | -0.702±0.007 | -0.702±0.003 | -0.697±0.006 |
| NoMAD, sorted units re-ordered to day-0 columns by electrode (mean over 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.391±0.001 | -0.391±0.002 | -0.392±0.006 | -0.393±0.003 | -0.390±0.010 |
