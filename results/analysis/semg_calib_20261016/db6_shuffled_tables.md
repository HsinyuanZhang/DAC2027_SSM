### DB6 v2 (shuffled order) per-bin balanced accuracy, mean ± 95% CI over 10 subjects (all gaps)

| method | B=4 | B=16 |
|---|---|---|
| `reuse_latest_srcnorm` | 0.404 ± 0.091 | 0.404 ± 0.091 |
| `reuse_latest_calnorm` | 0.518 ± 0.082 | 0.558 ± 0.093 |
| `mean_table_calnorm` | 0.504 ± 0.056 | 0.518 ± 0.061 |
| `mean_loso_calnorm` | 0.485 ± 0.057 | 0.499 ± 0.062 |
| `rownorm_ridge` | 0.379 ± 0.030 | 0.578 ± 0.046 |
| `ui_prior_latest_calnorm` | - | - |
| `ui_prior_mean_calnorm` | - | - |
| `cal_p0_free` | 0.456 ± 0.057 | 0.556 ± 0.064 |
| `cal_p0_mean` | 0.530 ± 0.047 | 0.586 ± 0.061 |
| `cal_p0_own` | 0.517 ± 0.082 | 0.557 ± 0.093 |
| `cal_p1_free` | 0.462 ± 0.053 | 0.564 ± 0.063 |
| `cal_p1_mean` | 0.523 ± 0.040 | 0.579 ± 0.055 |
| `cal_p1_own` | 0.517 ± 0.082 | 0.557 ± 0.093 |
| `cal_p2_free` | 0.480 ± 0.053 | 0.590 ± 0.057 |
| `cal_p2_mean` | 0.530 ± 0.044 | 0.584 ± 0.057 |
| `cal_p2_own` | 0.517 ± 0.082 | 0.557 ± 0.093 |
| `cal_p4_free` | 0.491 ± 0.049 | 0.590 ± 0.052 |
| `cal_p4_mean` | 0.534 ± 0.051 | 0.582 ± 0.065 |
| `cal_p4_own` | 0.517 ± 0.082 | 0.557 ± 0.093 |

### DB6 v2 (shuffled order) macro-F1

| method | B=4 | B=16 |
|---|---|---|
| `reuse_latest_srcnorm` | 0.360 ± 0.098 | 0.360 ± 0.099 |
| `reuse_latest_calnorm` | 0.505 ± 0.083 | 0.549 ± 0.097 |
| `mean_table_calnorm` | 0.479 ± 0.062 | 0.495 ± 0.068 |
| `mean_loso_calnorm` | 0.460 ± 0.062 | 0.476 ± 0.067 |
| `rownorm_ridge` | 0.306 ± 0.027 | 0.581 ± 0.051 |
| `ui_prior_latest_calnorm` | - | - |
| `ui_prior_mean_calnorm` | - | - |
| `cal_p0_free` | 0.435 ± 0.061 | 0.542 ± 0.068 |
| `cal_p0_mean` | 0.513 ± 0.049 | 0.572 ± 0.065 |
| `cal_p0_own` | 0.504 ± 0.083 | 0.549 ± 0.097 |
| `cal_p1_free` | 0.442 ± 0.057 | 0.551 ± 0.067 |
| `cal_p1_mean` | 0.505 ± 0.043 | 0.566 ± 0.058 |
| `cal_p1_own` | 0.504 ± 0.083 | 0.549 ± 0.097 |
| `cal_p2_free` | 0.460 ± 0.055 | 0.577 ± 0.061 |
| `cal_p2_mean` | 0.513 ± 0.047 | 0.571 ± 0.059 |
| `cal_p2_own` | 0.504 ± 0.083 | 0.549 ± 0.097 |
| `cal_p4_free` | 0.470 ± 0.052 | 0.577 ± 0.054 |
| `cal_p4_mean` | 0.518 ± 0.051 | 0.573 ± 0.066 |
| `cal_p4_own` | 0.504 ± 0.083 | 0.549 ± 0.097 |

### Backbone-seed spread of the subject-mean balanced accuracy

| method | 4 | 16 |
|---|---|---|
| `cal_p0_free` | 0.0 | 0.0 |
| `cal_p0_mean` | 0.0 | 0.0 |
| `cal_p0_own` | 0.0 | 0.0 |
| `cal_p1_free` | 0.0 | 0.0 |
| `cal_p1_mean` | 0.0 | 0.0 |
| `cal_p1_own` | 0.0 | 0.0 |
| `cal_p2_free` | 0.0 | 0.0 |
| `cal_p2_mean` | 0.0 | 0.0 |
| `cal_p2_own` | 0.0 | 0.0 |
| `cal_p4_free` | 0.0 | 0.0 |
| `cal_p4_mean` | 0.0 | 0.0 |
| `cal_p4_own` | 0.0 | 0.0 |
| `mean_loso_calnorm` | 0.0 | 0.0 |
| `mean_table_calnorm` | 0.0 | 0.0 |
| `reuse_latest_calnorm` | 0.0 | 0.0 |
| `reuse_latest_srcnorm` | 0.0 | 0.0 |
| `rownorm_ridge` | 0.0 | 0.0 |


### Paired difference to the best deployable baseline per B (post hoc on subject means)

| a | 4 | 16 |
|---|---|---|
| `cal_p0_free` | -0.062 ± 0.063 | -0.022 ± 0.035 |
| `cal_p0_mean` | +0.013 ± 0.067 | +0.007 ± 0.029 |
| `cal_p0_own` | -0.000 ± 0.001 | -0.021 ± 0.072 |
| `cal_p1_free` | -0.055 ± 0.064 | -0.014 ± 0.030 |
| `cal_p1_mean` | +0.005 ± 0.067 | +0.001 ± 0.024 |
| `cal_p1_own` | -0.000 ± 0.001 | -0.021 ± 0.072 |
| `cal_p2_free` | -0.038 ± 0.073 | +0.012 ± 0.026 |
| `cal_p2_mean` | +0.012 ± 0.072 | +0.006 ± 0.025 |
| `cal_p2_own` | -0.000 ± 0.001 | -0.021 ± 0.072 |
| `cal_p4_free` | -0.027 ± 0.075 | +0.012 ± 0.027 |
| `cal_p4_mean` | +0.016 ± 0.068 | +0.004 ± 0.025 |
| `cal_p4_own` | -0.000 ± 0.001 | -0.021 ± 0.072 |
