### days

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.250±0.029 | 0.275±0.021 | 0.273±0.026 | 0.273±0.011 | 0.272±0.014 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.643±0.026 | 0.712±0.012 | 0.728±0.011 | 0.737±0.006 | 0.744±0.007 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.265±0.088 | 0.330±0.098 | 0.384±0.055 | 0.371±0.045 | 0.419±0.059 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.151±0.112 | -0.163±0.130 | -0.192±0.202 | -0.080±0.096 | -0.089±0.129 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.331±0.002 | 0.331±0.004 | 0.331±0.003 | 0.332±0.005 | 0.329±0.006 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.012±0.065 | 0.170±0.067 | 0.349±0.089 | 0.426±0.037 | 0.488±0.018 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.072±0.032 | 0.229±0.054 | 0.373±0.063 | 0.442±0.034 | 0.521±0.018 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.014±0.085 | 0.200±0.086 | 0.406±0.092 | 0.480±0.038 | 0.536±0.018 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.078±0.042 | 0.276±0.057 | 0.436±0.045 | 0.481±0.033 | 0.559±0.018 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.554±0.001 | 0.555±0.004 | 0.552±0.001 | 0.550±0.005 | 0.555±0.008 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.614±0.001 | 0.616±0.004 | 0.612±0.001 | 0.611±0.005 | 0.617±0.009 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.582±0.001 | 0.583±0.003 | 0.580±0.002 | 0.577±0.004 | 0.583±0.007 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.637±0.001 | 0.639±0.004 | 0.636±0.002 | 0.633±0.004 | 0.642±0.009 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.393±0.163 | 0.700±0.083 | 0.789±0.012 | 0.814±0.011 |
| APST ACT-only (last segment, source-only rule) | B trials: neural only (no labels) | no | 0.041±0.094 | 0.257±0.096 | 0.476±0.100 | 0.546±0.116 | 0.606±0.138 |
| APST-FULL (authors' dev-selected segment_09; dev = our days targets, so optimistic) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.449±0.164 | 0.731±0.059 | 0.799±0.010 | 0.822±0.010 |
| APST ACT-only (authors' dev-selected segment_08; dev = our days targets, so optimistic) | B trials: neural only (no labels) | no | 0.062±0.105 | 0.290±0.111 | 0.477±0.115 | 0.560±0.128 | 0.607±0.147 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.174±0.002 | -0.170±0.011 | -0.174±0.008 | -0.175±0.016 | -0.177±0.018 |

### month

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.293±0.051 | 0.297±0.021 | 0.312±0.011 | 0.308±0.011 | 0.311±0.005 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.463±0.043 | 0.538±0.017 | 0.574±0.011 | 0.567±0.009 | 0.571±0.009 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.059±0.103 | 0.244±0.053 | 0.223±0.092 | 0.257±0.052 | 0.295±0.024 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.089±0.035 | -0.144±0.106 | -0.190±0.167 | -0.105±0.201 | -0.191±0.178 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.250±0.000 | 0.251±0.001 | 0.251±0.002 | 0.251±0.001 | 0.252±0.002 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.037±0.167 | 0.206±0.116 | 0.358±0.052 | 0.413±0.065 | 0.447±0.060 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.059±0.059 | 0.294±0.094 | 0.402±0.044 | 0.441±0.053 | 0.513±0.046 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.001±0.109 | 0.253±0.129 | 0.384±0.040 | 0.444±0.059 | 0.479±0.061 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.067±0.070 | 0.336±0.094 | 0.417±0.051 | 0.454±0.056 | 0.528±0.045 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.367±0.000 | 0.367±0.001 | 0.367±0.001 | 0.367±0.002 | 0.367±0.002 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.478±0.000 | 0.478±0.001 | 0.478±0.001 | 0.478±0.001 | 0.479±0.002 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.384±0.000 | 0.384±0.001 | 0.384±0.001 | 0.384±0.002 | 0.385±0.002 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.486±0.000 | 0.486±0.001 | 0.486±0.001 | 0.487±0.002 | 0.487±0.002 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.284±0.080 (n=5) | 0.338±0.075 (n=7) | 0.394±0.072 | 0.466±0.037 |
| APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.306±0.052 (n=5) | 0.357±0.081 (n=7) | 0.417±0.084 | 0.477±0.038 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.877±0.001 | -0.879±0.002 | -0.879±0.004 | -0.876±0.006 | -0.877±0.007 |

### year

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (re-evaluated, native-query subset) | B trials: neural only (input mean/std) | no | 0.187±0.019 | 0.186±0.016 | 0.180±0.014 | 0.183±0.006 | 0.187±0.007 |
| P0 (re-evaluated, native-query subset) | B trials: neural + behaviour labels | no | 0.273±0.031 | 0.378±0.047 | 0.361±0.044 | 0.416±0.017 | 0.415±0.019 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.122±0.126 | 0.004±0.160 | 0.011±0.083 | 0.148±0.081 | 0.136±0.087 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.245±0.185 | -0.163±0.161 | -0.031±0.128 | -0.159±0.071 | -0.159±0.180 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.033±0.000 | 0.033±0.001 | 0.032±0.003 | 0.032±0.001 | 0.035±0.003 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.017±0.102 | 0.191±0.085 | 0.285±0.067 | 0.381±0.056 | 0.446±0.026 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | -0.017±0.140 | 0.257±0.064 | 0.324±0.044 | 0.419±0.050 | 0.472±0.017 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.001±0.108 | 0.204±0.112 | 0.300±0.073 | 0.412±0.056 | 0.481±0.025 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | -0.027±0.152 | 0.280±0.078 | 0.355±0.049 | 0.451±0.050 | 0.498±0.019 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.366±0.000 | 0.366±0.001 | 0.366±0.002 | 0.365±0.001 | 0.366±0.002 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.500±0.000 | 0.500±0.001 | 0.499±0.002 | 0.498±0.001 | 0.500±0.002 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.383±0.000 | 0.382±0.001 | 0.382±0.002 | 0.381±0.001 | 0.383±0.002 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.519±0.000 | 0.520±0.001 | 0.519±0.002 | 0.518±0.001 | 0.520±0.002 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.200±0.072 (n=7) | 0.283±0.059 | 0.365±0.027 | 0.359±0.032 |
| APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.284±0.081 (n=7) | 0.352±0.048 | 0.444±0.032 | 0.433±0.027 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -1.007±0.001 | -1.004±0.009 | -1.005±0.014 | -1.010±0.006 | -1.006±0.015 |
