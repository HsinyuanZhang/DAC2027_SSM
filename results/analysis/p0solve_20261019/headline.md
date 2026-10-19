# headline (mean [target-cluster 95% CI]; single-target strata have no target CI)

## days
B=1: p0 0.647 [0.639, 0.656] | w_alpha 0.628 [0.618, 0.638] | p0solve_p8 0.647 [0.639, 0.656] | p0solve_p16 0.633 [0.613, 0.653] | ridge_cca -0.026 [-0.043, -0.009] | r3 0.651 [0.642, 0.661] | p0_ui_reg 0.654 [0.644, 0.664]
B=8: p0 0.723 [0.717, 0.730] | w_alpha 0.639 [0.631, 0.647] | p0solve_p8 0.724 [0.718, 0.730] | p0solve_p16 0.725 [0.720, 0.730] | ridge_cca 0.609 [0.588, 0.630] | r3 0.728 [0.725, 0.731] | p0_ui_reg 0.720 [0.719, 0.721]
B=33: p0 0.738 [0.735, 0.740] | w_alpha 0.716 [0.715, 0.718] | p0solve_p8 0.738 [0.736, 0.740] | p0solve_p16 0.736 [0.735, 0.737] | ridge_cca 0.727 [0.718, 0.737] | r3 0.747 [0.744, 0.750] | p0_ui_reg 0.746 [0.741, 0.750]

## month
B=1: p0 0.466 | w_alpha 0.466 | p0solve_p8 0.466 | p0solve_p16 0.468 | ridge_cca 0.060 | r3 0.492 | p0_ui_reg 0.489
B=8: p0 0.565 | w_alpha 0.613 | p0solve_p8 0.567 | p0solve_p16 0.568 | ridge_cca 0.512 | r3 0.630 | p0_ui_reg 0.623
B=33: p0 0.566 | w_alpha 0.684 | p0solve_p8 0.604 | p0solve_p16 0.582 | ridge_cca 0.656 | r3 0.679 | p0_ui_reg 0.731

## year
B=1: p0 0.198 | w_alpha 0.198 | p0solve_p8 0.198 | p0solve_p16 0.198 | ridge_cca 0.023 | r3 0.219 | p0_ui_reg 0.230
B=8: p0 0.256 | w_alpha 0.321 | p0solve_p8 0.257 | p0solve_p16 0.253 | ridge_cca 0.289 | r3 0.334 | p0_ui_reg 0.348
B=33: p0 0.315 | w_alpha 0.465 | p0solve_p8 0.308 | p0solve_p16 0.303 | ridge_cca 0.439 | r3 0.435 | p0_ui_reg 0.492

## drift
B=1: p0 0.153 [0.102, 0.221] | w_alpha 0.153 [0.104, 0.223] | p0solve_p8 0.153 [0.104, 0.224] | p0solve_p16 0.158 [0.108, 0.225] | ridge_cca -0.008 [-0.032, 0.018] | r3 0.165 [0.113, 0.236] | p0_ui_reg 0.174 [0.121, 0.242]
B=8: p0 0.212 [0.149, 0.294] | w_alpha 0.285 [0.219, 0.363] | p0solve_p8 0.217 [0.152, 0.295] | p0solve_p16 0.220 [0.156, 0.298] | ridge_cca 0.319 [0.263, 0.376] | r3 0.348 [0.288, 0.416] | p0_ui_reg 0.310 [0.250, 0.382]
B=33: p0 0.211 [0.144, 0.296] | w_alpha 0.420 [0.354, 0.486] | p0solve_p8 0.276 [0.217, 0.347] | p0solve_p16 0.244 [0.175, 0.323] | ridge_cca 0.432 [0.368, 0.494] | r3 0.412 [0.347, 0.478] | p0_ui_reg 0.451 [0.381, 0.522]

## projection retained fraction (mean diag gain / mean ui gain)
days B=1 p=8: -0.034 (ui gain +0.006, diag gain -0.000, fro 0.255, like_for_like=False)
days B=1 p=16: 0.563 (ui gain +0.006, diag gain +0.004, fro 0.486, like_for_like=False)
days B=8 p=8: -0.044 (ui gain -0.003, diag gain +0.000, fro 0.641, like_for_like=False)
days B=8 p=16: 0.192 (ui gain -0.003, diag gain -0.001, fro 0.841, like_for_like=False)
days B=33 p=8: 1.035 (ui gain +0.008, diag gain +0.008, fro 0.711, like_for_like=False)
days B=33 p=16: 1.036 (ui gain +0.008, diag gain +0.008, fro 0.973, like_for_like=False)
drift B=1 p=8: 0.091 (ui gain +0.021, diag gain +0.002, fro 0.234, like_for_like=True)
drift B=1 p=16: 0.443 (ui gain +0.021, diag gain +0.009, fro 0.497, like_for_like=True)
drift B=8 p=8: 0.637 (ui gain +0.098, diag gain +0.062, fro 0.651, like_for_like=True)
drift B=8 p=16: 0.863 (ui gain +0.098, diag gain +0.085, fro 0.836, like_for_like=True)
drift B=33 p=8: 0.790 (ui gain +0.240, diag gain +0.189, fro 0.730, like_for_like=True)
drift B=33 p=16: 0.945 (ui gain +0.240, diag gain +0.226, fro 0.905, like_for_like=True)
month B=1 p=8: 0.004 (ui gain +0.023, diag gain +0.000, fro 0.200, like_for_like=True)
month B=1 p=16: 0.277 (ui gain +0.023, diag gain +0.006, fro 0.483, like_for_like=True)
month B=8 p=8: 0.338 (ui gain +0.058, diag gain +0.020, fro 0.467, like_for_like=True)
month B=8 p=16: 0.572 (ui gain +0.058, diag gain +0.033, fro 0.646, like_for_like=True)
month B=33 p=8: 0.593 (ui gain +0.165, diag gain +0.098, fro 0.532, like_for_like=True)
month B=33 p=16: 0.790 (ui gain +0.165, diag gain +0.130, fro 0.706, like_for_like=True)
year B=1 p=8: 0.070 (ui gain +0.031, diag gain +0.002, fro 0.223, like_for_like=True)
year B=1 p=16: 0.206 (ui gain +0.031, diag gain +0.006, fro 0.447, like_for_like=True)
year B=8 p=8: 0.552 (ui gain +0.092, diag gain +0.051, fro 0.531, like_for_like=True)
year B=8 p=16: 0.760 (ui gain +0.092, diag gain +0.070, fro 0.702, like_for_like=True)
year B=33 p=8: 0.652 (ui gain +0.177, diag gain +0.115, fro 0.640, like_for_like=True)
year B=33 p=16: 0.882 (ui gain +0.177, diag gain +0.156, fro 0.854, like_for_like=True)
