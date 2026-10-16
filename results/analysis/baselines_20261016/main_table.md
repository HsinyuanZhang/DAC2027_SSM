### days

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.333±0.025 | 0.360±0.018 | 0.360±0.023 | 0.362±0.011 | 0.361±0.012 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.647±0.021 | 0.708±0.009 | 0.723±0.009 | 0.732±0.005 | 0.738±0.005 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.648±0.021 | 0.710±0.009 | 0.723±0.010 | 0.732±0.005 | 0.740±0.005 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.651±0.020 | 0.714±0.009 | 0.728±0.010 | 0.737±0.006 | 0.747±0.006 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.281±0.089 | 0.341±0.098 | 0.400±0.060 | 0.383±0.056 | 0.435±0.060 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.119±0.101 | -0.150±0.119 | -0.156±0.178 | -0.060±0.090 | -0.085±0.112 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.348±0.001 | 0.348±0.003 | 0.348±0.002 | 0.348±0.004 | 0.346±0.005 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.037±0.059 | 0.223±0.071 | 0.391±0.080 | 0.465±0.040 | 0.525±0.017 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.092±0.045 | 0.284±0.054 | 0.423±0.054 | 0.487±0.035 | 0.559±0.015 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.036±0.077 | 0.242±0.086 | 0.433±0.078 | 0.500±0.039 | 0.558±0.013 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.092±0.052 | 0.317±0.055 | 0.472±0.036 | 0.511±0.033 | 0.583±0.014 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.591±0.001 | 0.593±0.003 | 0.590±0.001 | 0.589±0.004 | 0.592±0.005 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.650±0.001 | 0.651±0.003 | 0.649±0.001 | 0.648±0.004 | 0.652±0.006 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.605±0.001 | 0.606±0.003 | 0.604±0.002 | 0.601±0.003 | 0.606±0.004 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.661±0.001 | 0.662±0.003 | 0.660±0.001 | 0.657±0.003 | 0.663±0.006 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.279±0.118 | 0.500±0.059 | 0.568±0.011 | 0.590±0.009 |
| APST ACT-only (last segment, source-only rule) | B trials: neural only (no labels) | no | 0.026±0.068 | 0.178±0.074 | 0.340±0.070 | 0.393±0.083 | 0.437±0.103 |
| APST-FULL (authors' dev-selected segment_09; dev = our days targets, so optimistic) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.311±0.119 | 0.516±0.043 | 0.567±0.006 | 0.588±0.009 |
| APST ACT-only (authors' dev-selected segment_08; dev = our days targets, so optimistic) | B trials: neural only (no labels) | no | 0.041±0.076 | 0.196±0.086 | 0.338±0.080 | 0.396±0.093 | 0.429±0.110 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.130±0.002 | -0.125±0.008 | -0.131±0.006 | -0.130±0.013 | -0.133±0.013 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.378±0.028 | 0.527±0.039 | 0.607±0.038 | 0.661±0.020 | 0.704±0.015 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.654±0.019 | 0.701±0.015 | 0.720±0.015 | 0.730±0.012 | 0.746±0.008 |

### month

| method | target data used | target backprop | B=1 | B=4 | B=8 | B=16 | B=33 |
|---|---|---|---:|---:|---:|---:|---:|
| P-prior (electrode-aligned source prior + B-trial input z-score) | B trials: neural only (input mean/std) | no | 0.315±0.046 | 0.319±0.021 | 0.332±0.010 | 0.328±0.007 | 0.333±0.004 |
| P0 (fixed-weight per-unit encoder, level 1) | B trials: neural + behaviour labels | no | 0.466±0.034 | 0.535±0.014 | 0.565±0.011 | 0.558±0.007 | 0.566±0.006 |
| P0 + R1-stream (readout ridge, level 2) | B trials: neural + behaviour labels | no | 0.472±0.037 | 0.542±0.014 | 0.576±0.011 | 0.588±0.007 | 0.606±0.008 |
| P0 + R3 (F7-J local update + readout ridge, optional level 3) | B trials: neural + behaviour labels (forward replay) | no (forward-only replay) | 0.492±0.048 | 0.591±0.028 | 0.630±0.011 | 0.652±0.014 | 0.679±0.007 |
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | 0.076±0.076 | 0.235±0.037 | 0.214±0.073 | 0.247±0.042 | 0.290±0.022 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.075±0.045 | -0.120±0.089 | -0.149±0.139 | -0.090±0.183 | -0.151±0.154 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.257±0.000 | 0.257±0.001 | 0.258±0.001 | 0.258±0.001 | 0.258±0.001 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.030±0.165 | 0.193±0.101 | 0.333±0.046 | 0.388±0.059 | 0.423±0.051 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.063±0.051 | 0.281±0.075 | 0.384±0.034 | 0.435±0.043 | 0.497±0.036 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | 0.003±0.108 | 0.232±0.114 | 0.350±0.035 | 0.406±0.054 | 0.444±0.050 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | 0.069±0.062 | 0.317±0.075 | 0.394±0.043 | 0.441±0.045 | 0.506±0.036 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.376±0.000 | 0.376±0.001 | 0.376±0.001 | 0.375±0.002 | 0.376±0.001 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.487±0.000 | 0.487±0.001 | 0.487±0.001 | 0.487±0.001 | 0.488±0.002 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.385±0.000 | 0.385±0.001 | 0.385±0.001 | 0.385±0.002 | 0.386±0.001 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.488±0.000 | 0.488±0.001 | 0.488±0.001 | 0.488±0.001 | 0.489±0.002 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.219±0.071 (n=5) | 0.271±0.056 (n=7) | 0.324±0.054 | 0.376±0.027 |
| APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.240±0.040 (n=5) | 0.290±0.062 (n=7) | 0.342±0.064 | 0.387±0.029 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.900±0.001 | -0.902±0.002 | -0.902±0.003 | -0.900±0.005 | -0.900±0.005 |
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
| Stabilizer (a): FA + Procrustes on stable channels, B-trial data | B trials: neural only (no labels) | no | -0.065±0.076 | 0.017±0.101 | 0.020±0.051 | 0.098±0.042 | 0.100±0.046 |
| Stabilizer control: target FA, no rotation | B trials: neural only (no labels) | no | -0.135±0.116 | -0.111±0.110 | -0.019±0.077 | -0.110±0.062 | -0.131±0.111 |
| Stabilizer (b): FA + Procrustes, all target-session trials | ALL target usable trials: neural only (incl. query bins) | no | 0.029±0.000 | 0.029±0.001 | 0.028±0.001 | 0.028±0.001 | 0.033±0.005 |
| Gallego-style PCA+CCA (a): sqrt counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.010±0.058 | 0.119±0.046 | 0.184±0.040 | 0.247±0.035 | 0.315±0.048 |
| PCA+CCA (a), sqrt counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | -0.004±0.078 | 0.168±0.038 | 0.215±0.029 | 0.277±0.035 | 0.345±0.044 |
| PCA+CCA (a), raw counts, plain CCA, B-trial data | B trials: neural + reach-direction labels | no | -0.004±0.066 | 0.126±0.062 | 0.193±0.047 | 0.265±0.036 | 0.335±0.048 |
| PCA+CCA (a), raw counts, ridge CCA (source-selected) | B trials: neural + reach-direction labels | no | -0.008±0.085 | 0.183±0.044 | 0.232±0.036 | 0.295±0.035 | 0.360±0.049 |
| Gallego-style PCA+CCA (b): sqrt counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.219±0.000 | 0.219±0.000 | 0.218±0.001 | 0.217±0.001 | 0.241±0.041 |
| PCA+CCA (b), sqrt counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.334±0.000 | 0.334±0.000 | 0.333±0.001 | 0.331±0.001 | 0.368±0.063 |
| PCA+CCA (b), raw counts, plain CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.225±0.000 | 0.225±0.000 | 0.224±0.001 | 0.223±0.001 | 0.248±0.042 |
| PCA+CCA (b), raw counts, ridge CCA, all target trials | ALL target usable trials: neural + reach-direction labels (incl. query trials) | no | 0.344±0.000 | 0.344±0.000 | 0.343±0.001 | 0.340±0.001 | 0.379±0.064 |
| APST-FULL (last segment, source-only rule) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.105±0.039 (n=7) | 0.150±0.034 | 0.189±0.016 | 0.213±0.047 |
| APST-FULL (2014 retrain; segment_08 chosen on source-OOS session 20140307, source-only) | B trials: neural + reach-direction labels (MOVE-T4 profile) | no | n/a | 0.149±0.042 (n=7) | 0.189±0.025 | 0.234±0.018 | 0.257±0.050 |
| NoMAD (LFADS + alignment network; mean over targets and 3 seeds) | ALL target-session neural data, unlabeled | YES (alignment net, target gradients) | -0.625±0.001 | -0.622±0.004 | -0.621±0.007 | -0.621±0.003 | -0.688±0.121 |
| ui_reg: L2-to-prior table fine-tune through trunk (from prior) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.191±0.021 | 0.267±0.025 | 0.290±0.046 | 0.367±0.007 | 0.478±0.088 |
| P0 + ui_reg (P2 rule) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.228±0.023 | 0.321±0.030 | 0.309±0.029 | 0.393±0.011 | 0.488±0.086 |
| P0 + ui_reg (like-for-like: month oosc / year oos) | B trials: neural + behaviour labels | YES (through frozen trunk) | 0.230±0.026 | 0.345±0.040 | 0.348±0.017 | 0.405±0.009 | 0.492±0.087 |
