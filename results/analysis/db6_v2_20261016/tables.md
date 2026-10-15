### T1 balanced accuracy, all gaps

| method | B=1 | B=2 | B=4 | B=8 | B=16 | B=42（诊断） |
|---|---|---|---|---|---|---|
| `reuse_latest_srcnorm`（零 target 数据） | 0.407 ± 0.091 | 0.407 ± 0.091 | 0.407 ± 0.092 | 0.407 ± 0.092 | 0.407 ± 0.092 | 0.404 ± 0.093 |
| `reuse_latest_calnorm` | 0.396 ± 0.062 | 0.457 ± 0.076 | 0.508 ± 0.086 | 0.541 ± 0.092 | 0.548 ± 0.093 | 0.556 ± 0.097 |
| `mean_table_calnorm` | 0.428 ± 0.059 | 0.477 ± 0.056 | 0.511 ± 0.051 | 0.533 ± 0.053 | 0.532 ± 0.053 | 0.516 ± 0.056 |
| `mean_loso_calnorm` | 0.413 ± 0.058 | 0.459 ± 0.055 | 0.490 ± 0.050 | 0.511 ± 0.051 | 0.510 ± 0.052 | 0.495 ± 0.056 |
| `rownorm_ridge`（部署型 closed-form，block-free 选择） | 0.201 ± 0.014 | 0.277 ± 0.020 | 0.381 ± 0.029 | 0.514 ± 0.049 | 0.578 ± 0.047 | 0.630 ± 0.055 |
| `rownorm_ridge_origsel`（原顺序 source 上选择，可吃 block 结构） | 0.201 ± 0.014 | 0.277 ± 0.020 | 0.382 ± 0.023 | 0.598 ± 0.052 | 0.683 ± 0.041 | 0.754 ± 0.031 |
| `ui_prior_latest_srcnorm`（参照） | 0.361 ± 0.057 | 0.414 ± 0.062 | 0.509 ± 0.073 | 0.619 ± 0.086 | 0.684 ± 0.085 | 0.745 ± 0.081 |
| `ui_prior_latest_calnorm`（参照） | 0.355 ± 0.047 | 0.445 ± 0.065 | 0.553 ± 0.067 | 0.666 ± 0.073 | 0.724 ± 0.073 | 0.787 ± 0.069 |
| `ui_prior_mean_calnorm`（参照） | 0.346 ± 0.037 | 0.437 ± 0.045 | 0.549 ± 0.054 | 0.673 ± 0.068 | 0.730 ± 0.068 | 0.790 ± 0.064 |
| `within_table_latestinit_long`（诊断） | - | - | - | - | - | 0.787 ± 0.068 |
| `within_table_supinit`（诊断） | - | - | - | - | - | 0.782 ± 0.062 |

### T2 balanced accuracy per gap (B=4 / B=16)

| method | gap 1 (day 3) | gap 2 (day 4) | gap 3 (day 5) |
|---|---|---|---|
| `reuse_latest_srcnorm`（零 target 数据） | 0.412 / 0.412 | 0.442 / 0.442 | 0.367 / 0.366 |
| `reuse_latest_calnorm` | 0.537 / 0.582 | 0.525 / 0.568 | 0.463 / 0.494 |
| `mean_table_calnorm` | 0.516 / 0.537 | 0.511 / 0.536 | 0.504 / 0.523 |
| `mean_loso_calnorm` | 0.494 / 0.515 | 0.491 / 0.514 | 0.484 / 0.501 |
| `rownorm_ridge`（部署型 closed-form，block-free 选择） | 0.382 / 0.568 | 0.386 / 0.583 | 0.375 / 0.582 |
| `rownorm_ridge_origsel`（原顺序 source 上选择，可吃 block 结构） | 0.387 / 0.682 | 0.395 / 0.693 | 0.365 / 0.675 |
| `ui_prior_latest_srcnorm`（参照） | 0.512 / 0.685 | 0.541 / 0.706 | 0.475 / 0.659 |
| `ui_prior_latest_calnorm`（参照） | 0.569 / 0.729 | 0.571 / 0.737 | 0.517 / 0.706 |
| `ui_prior_mean_calnorm`（参照） | 0.555 / 0.729 | 0.560 / 0.740 | 0.532 / 0.723 |

### T3 macro-F1 and vote accuracy (all gaps; B = 1 / 4 / 16)

| method | macro-F1 B=1 / 4 / 16 | vote acc B=1 / 4 / 16 |
|---|---|---|
| `reuse_latest_srcnorm`（零 target 数据） | 0.366 / 0.365 / 0.365 | 0.410 / 0.410 / 0.411 |
| `reuse_latest_calnorm` | 0.365 / 0.496 / 0.540 | 0.408 / 0.556 / 0.607 |
| `mean_table_calnorm` | 0.399 / 0.491 / 0.513 | 0.445 / 0.559 / 0.587 |
| `mean_loso_calnorm` | 0.384 / 0.470 / 0.491 | 0.425 / 0.531 / 0.558 |
| `rownorm_ridge`（部署型 closed-form，block-free 选择） | 0.109 / 0.307 / 0.578 | 0.132 / 0.419 / 0.760 |
| `rownorm_ridge_origsel`（原顺序 source 上选择，可吃 block 结构） | 0.109 / 0.263 / 0.583 | 0.132 / 0.423 / 0.797 |
| `ui_prior_latest_srcnorm`（参照） | 0.311 / 0.470 / 0.675 | 0.349 / 0.540 / 0.778 |
| `ui_prior_latest_calnorm`（参照） | 0.323 / 0.532 / 0.719 | 0.366 / 0.606 / 0.822 |
| `ui_prior_mean_calnorm`（参照） | 0.311 / 0.522 / 0.722 | 0.343 / 0.593 / 0.825 |

### T4 paired differences (a - b), balanced accuracy, all gaps, 95% CI over 10 subjects

| a - b | B=1 | B=2 | B=4 | B=8 | B=16 | B=42 |
|---|---|---|---|---|---|---|
| `reuse_latest_calnorm` - `reuse_latest_srcnorm` | -0.011 ± 0.037 | +0.051 ± 0.039 | +0.102 ± 0.043 | +0.133 ± 0.049 | +0.142 ± 0.051 | +0.152 ± 0.054 |
| `ui_prior_latest_calnorm` - `reuse_latest_calnorm` | -0.041 ± 0.024 | -0.012 ± 0.022 | +0.044 ± 0.037 | +0.125 ± 0.050 | +0.176 ± 0.054 | +0.231 ± 0.059 |
| `ui_prior_latest_calnorm` - `rownorm_ridge` | +0.155 ± 0.035 | +0.168 ± 0.050 | +0.172 ± 0.045 | +0.151 ± 0.038 | +0.147 ± 0.032 | +0.157 ± 0.021 |
| `rownorm_ridge` - `reuse_latest_calnorm` | -0.195 ± 0.052 | -0.181 ± 0.064 | -0.128 ± 0.072 | -0.026 ± 0.077 | +0.030 ± 0.073 | +0.074 ± 0.069 |
| `rownorm_ridge_origsel` - `rownorm_ridge` | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.001 ± 0.015 | +0.083 ± 0.015 | +0.106 ± 0.015 | +0.124 ± 0.027 |
| `reuse_latest_calnorm` - `mean_table_calnorm` | -0.032 ± 0.052 | -0.020 ± 0.060 | -0.002 ± 0.072 | +0.008 ± 0.076 | +0.016 ± 0.077 | +0.040 ± 0.082 |
| `mean_table_calnorm` - `mean_loso_calnorm` | +0.015 ± 0.006 | +0.018 ± 0.009 | +0.021 ± 0.009 | +0.022 ± 0.011 | +0.022 ± 0.010 | +0.021 ± 0.011 |
| `ui_prior_latest_calnorm` - `ui_prior_mean_calnorm` | +0.009 ± 0.033 | +0.008 ± 0.036 | +0.004 ± 0.032 | -0.008 ± 0.021 | -0.006 ± 0.015 | -0.004 ± 0.015 |
| `within_table_latestinit_long` - `ui_prior_latest_calnorm` | - | - | - | - | - | -0.000 ± 0.006 |
| `within_table_supinit` - `ui_prior_latest_calnorm` | - | - | - | - | - | -0.005 ± 0.013 |

### T5 own-subject latest vs population mean table, per gap (a = `mean_table_calnorm`, b = `reuse_latest_calnorm`)

| gap | B=4 | B=8 | B=16 |
|---|---|---|---|
| 1 | -0.021 ± 0.097 | -0.038 ± 0.099 | -0.045 ± 0.099 |
| 2 | -0.014 ± 0.090 | -0.029 ± 0.094 | -0.032 ± 0.095 |
| 3 | +0.041 ± 0.088 | +0.043 ± 0.097 | +0.028 ± 0.103 |
