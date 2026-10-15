
**B = 1**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.310 ± 0.056 | 0.438 ± 0.093 | 0.407 ± 0.091 |
| `reuse_latest_calnorm` | 0.403 ± 0.043 | 0.308 ± 0.037 | 0.419 ± 0.063 | 0.396 ± 0.062 |
| `mean_table_calnorm` | 0.440 ± 0.056 | 0.291 ± 0.035 | 0.446 ± 0.066 | 0.428 ± 0.059 |
| `rownorm_ridge` | 0.199 ± 0.015 | - | - | 0.201 ± 0.014 |
| `ui_prior_latest_calnorm` | 0.374 ± 0.032 | - | - | 0.355 ± 0.047 |

**B = 2**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.310 ± 0.056 | 0.438 ± 0.093 | 0.407 ± 0.091 |
| `reuse_latest_calnorm` | 0.475 ± 0.058 | 0.353 ± 0.056 | 0.504 ± 0.089 | 0.457 ± 0.076 |
| `mean_table_calnorm` | 0.494 ± 0.051 | 0.291 ± 0.032 | 0.519 ± 0.069 | 0.477 ± 0.056 |
| `rownorm_ridge` | 0.275 ± 0.020 | - | - | 0.277 ± 0.020 |
| `ui_prior_latest_calnorm` | 0.470 ± 0.047 | - | - | 0.445 ± 0.065 |

**B = 4**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.311 ± 0.056 | 0.438 ± 0.093 | 0.407 ± 0.092 |
| `reuse_latest_calnorm` | 0.533 ± 0.069 | 0.388 ± 0.059 | 0.575 ± 0.103 | 0.508 ± 0.086 |
| `mean_table_calnorm` | 0.527 ± 0.049 | 0.286 ± 0.025 | 0.569 ± 0.065 | 0.511 ± 0.051 |
| `rownorm_ridge` | 0.395 ± 0.027 | - | - | 0.381 ± 0.029 |
| `ui_prior_latest_calnorm` | 0.559 ± 0.050 | - | - | 0.553 ± 0.067 |

**B = 8**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.310 ± 0.056 | 0.438 ± 0.093 | 0.407 ± 0.092 |
| `reuse_latest_calnorm` | 0.565 ± 0.074 | 0.400 ± 0.065 | 0.613 ± 0.112 | 0.541 ± 0.092 |
| `mean_table_calnorm` | 0.550 ± 0.053 | 0.294 ± 0.022 | 0.593 ± 0.071 | 0.533 ± 0.053 |
| `rownorm_ridge` | 0.537 ± 0.045 | - | - | 0.514 ± 0.049 |
| `ui_prior_latest_calnorm` | 0.663 ± 0.056 | - | - | 0.666 ± 0.073 |

**B = 16**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.429 ± 0.076 | 0.310 ± 0.056 | 0.437 ± 0.093 | 0.407 ± 0.092 |
| `reuse_latest_calnorm` | 0.574 ± 0.075 | 0.402 ± 0.067 | 0.626 ± 0.114 | 0.548 ± 0.093 |
| `mean_table_calnorm` | 0.548 ± 0.053 | 0.290 ± 0.023 | 0.596 ± 0.070 | 0.532 ± 0.053 |
| `rownorm_ridge` | 0.608 ± 0.043 | - | - | 0.578 ± 0.047 |
| `ui_prior_latest_calnorm` | 0.726 ± 0.057 | - | - | 0.724 ± 0.073 |

**B = 42**

| method | v1 (reset) | v1 ckpt contiguous | v2a block-order-trained contiguous (seed0) | v2 (contiguous) |
|---|---|---|---|---|
| `reuse_latest_srcnorm` | 0.434 ± 0.075 | - | - | 0.404 ± 0.093 |
| `reuse_latest_calnorm` | 0.583 ± 0.081 | - | - | 0.556 ± 0.097 |
| `mean_table_calnorm` | 0.557 ± 0.057 | - | - | 0.516 ± 0.056 |
| `rownorm_ridge` | 0.672 ± 0.043 | - | - | 0.630 ± 0.055 |
| `ui_prior_latest_calnorm` | 0.802 ± 0.053 | - | - | 0.787 ± 0.069 |
| `within_table_latestinit_long` | 0.803 ± 0.051 | - | - | 0.787 ± 0.068 |
