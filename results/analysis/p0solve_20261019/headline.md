# headline (mean [target-cluster 95% CI]; single-target strata have no target CI)

## days
B=1: p0 0.647 [0.639, 0.656] | w_alpha 0.628 [0.618, 0.638] | w_alpha_gap 0.647 [0.639, 0.656] | w_alpha_rel 0.622 [0.614, 0.631] | w_alpha_both 0.647 [0.639, 0.656] | p0solve_p8 0.647 [0.639, 0.656] | p0solve_p16 0.638 [0.619, 0.657] | ridge_cca -0.026 [-0.043, -0.009] | r3 0.651 [0.642, 0.661] | p0_ui_reg 0.654 [0.644, 0.664]
B=8: p0 0.723 [0.717, 0.730] | w_alpha 0.639 [0.631, 0.647] | w_alpha_gap 0.723 [0.717, 0.730] | w_alpha_rel 0.708 [0.705, 0.710] | w_alpha_both 0.723 [0.717, 0.730] | p0solve_p8 0.722 [0.715, 0.729] | p0solve_p16 0.716 [0.712, 0.720] | ridge_cca 0.609 [0.588, 0.630] | r3 0.728 [0.725, 0.731] | p0_ui_reg 0.720 [0.719, 0.721]
B=33: p0 0.738 [0.735, 0.740] | w_alpha 0.716 [0.715, 0.718] | w_alpha_gap 0.738 [0.735, 0.740] | w_alpha_rel 0.721 [0.717, 0.726] | w_alpha_both 0.738 [0.735, 0.740] | p0solve_p8 0.737 [0.733, 0.740] | p0solve_p16 0.695 [0.688, 0.702] | ridge_cca 0.727 [0.718, 0.737] | r3 0.747 [0.744, 0.750] | p0_ui_reg 0.746 [0.741, 0.750]

## month
B=1: p0 0.466 | w_alpha 0.466 | w_alpha_gap 0.442 | w_alpha_rel 0.466 | w_alpha_both 0.461 | p0solve_p8 0.466 | p0solve_p16 0.468 | ridge_cca 0.060 | r3 0.492 | p0_ui_reg 0.489
B=8: p0 0.565 | w_alpha 0.613 | w_alpha_gap 0.613 | w_alpha_rel 0.597 | w_alpha_both 0.603 | p0solve_p8 0.577 | p0solve_p16 0.584 | ridge_cca 0.512 | r3 0.630 | p0_ui_reg 0.623
B=33: p0 0.566 | w_alpha 0.684 | w_alpha_gap 0.647 | w_alpha_rel 0.628 | w_alpha_both 0.634 | p0solve_p8 0.608 | p0solve_p16 0.613 | ridge_cca 0.656 | r3 0.679 | p0_ui_reg 0.731

## year
B=1: p0 0.198 | w_alpha 0.198 | w_alpha_gap 0.198 | w_alpha_rel 0.186 | w_alpha_both 0.198 | p0solve_p8 0.198 | p0solve_p16 0.198 | ridge_cca 0.023 | r3 0.219 | p0_ui_reg 0.230
B=8: p0 0.256 | w_alpha 0.321 | w_alpha_gap 0.320 | w_alpha_rel 0.299 | w_alpha_both 0.323 | p0solve_p8 0.256 | p0solve_p16 0.257 | ridge_cca 0.289 | r3 0.334 | p0_ui_reg 0.348
B=33: p0 0.315 | w_alpha 0.465 | w_alpha_gap 0.465 | w_alpha_rel 0.419 | w_alpha_both 0.466 | p0solve_p8 0.328 | p0solve_p16 0.371 | ridge_cca 0.439 | r3 0.435 | p0_ui_reg 0.492

## drift
B=1: p0 0.153 [0.104, 0.220] | w_alpha 0.153 [0.104, 0.221] | w_alpha_gap 0.151 [0.104, 0.218] | w_alpha_rel 0.159 [0.111, 0.224] | w_alpha_both 0.158 [0.109, 0.221] | p0solve_p8 0.153 [0.102, 0.222] | p0solve_p16 0.158 [0.106, 0.225] | ridge_cca -0.008 [-0.033, 0.017] | r3 0.165 [0.112, 0.236] | p0_ui_reg 0.174 [0.121, 0.246]
B=8: p0 0.212 [0.149, 0.290] | w_alpha 0.285 [0.220, 0.358] | w_alpha_gap 0.285 [0.221, 0.363] | w_alpha_rel 0.251 [0.187, 0.331] | w_alpha_both 0.261 [0.196, 0.339] | p0solve_p8 0.234 [0.171, 0.311] | p0solve_p16 0.251 [0.191, 0.327] | ridge_cca 0.319 [0.263, 0.374] | r3 0.348 [0.286, 0.414] | p0_ui_reg 0.310 [0.248, 0.383]
B=33: p0 0.211 [0.143, 0.294] | w_alpha 0.420 [0.355, 0.486] | w_alpha_gap 0.335 [0.271, 0.412] | w_alpha_rel 0.290 [0.223, 0.367] | w_alpha_both 0.304 [0.237, 0.381] | p0solve_p8 0.277 [0.219, 0.350] | p0solve_p16 0.307 [0.243, 0.381] | ridge_cca 0.432 [0.367, 0.495] | r3 0.412 [0.347, 0.478] | p0_ui_reg 0.451 [0.382, 0.520]

## projection retained fraction (mean diag gain / mean ui gain)
days B=1 p=8: -0.034 (ui gain +0.006, diag gain -0.000, energy 0.072, like_for_like=False)
days B=1 p=16: 0.563 (ui gain +0.006, diag gain +0.004, energy 0.248, like_for_like=False)
days B=8 p=8: -0.044 (ui gain -0.003, diag gain +0.000, energy 0.415, like_for_like=False)
days B=8 p=16: 0.192 (ui gain -0.003, diag gain -0.001, energy 0.711, like_for_like=False)
days B=33 p=8: 1.035 (ui gain +0.008, diag gain +0.008, energy 0.506, like_for_like=False)
days B=33 p=16: 1.036 (ui gain +0.008, diag gain +0.008, energy 0.946, like_for_like=False)
drift B=1 p=8: 0.091 (ui gain +0.021, diag gain +0.002, energy 0.063, like_for_like=True)
drift B=1 p=16: 0.443 (ui gain +0.021, diag gain +0.009, energy 0.259, like_for_like=True)
drift B=8 p=8: 0.637 (ui gain +0.098, diag gain +0.062, energy 0.439, like_for_like=True)
drift B=8 p=16: 0.863 (ui gain +0.098, diag gain +0.085, energy 0.711, like_for_like=True)
drift B=33 p=8: 0.790 (ui gain +0.240, diag gain +0.189, energy 0.548, like_for_like=True)
drift B=33 p=16: 0.945 (ui gain +0.240, diag gain +0.226, energy 0.832, like_for_like=True)
month B=1 p=8: 0.004 (ui gain +0.023, diag gain +0.000, energy 0.042, like_for_like=True)
month B=1 p=16: 0.277 (ui gain +0.023, diag gain +0.006, energy 0.238, like_for_like=True)
month B=8 p=8: 0.338 (ui gain +0.058, diag gain +0.020, energy 0.224, like_for_like=True)
month B=8 p=16: 0.572 (ui gain +0.058, diag gain +0.033, energy 0.427, like_for_like=True)
month B=33 p=8: 0.593 (ui gain +0.165, diag gain +0.098, energy 0.285, like_for_like=True)
month B=33 p=16: 0.790 (ui gain +0.165, diag gain +0.130, energy 0.500, like_for_like=True)
year B=1 p=8: 0.070 (ui gain +0.031, diag gain +0.002, energy 0.059, like_for_like=True)
year B=1 p=16: 0.206 (ui gain +0.031, diag gain +0.006, energy 0.207, like_for_like=True)
year B=8 p=8: 0.552 (ui gain +0.092, diag gain +0.051, energy 0.287, like_for_like=True)
year B=8 p=16: 0.760 (ui gain +0.092, diag gain +0.070, energy 0.498, like_for_like=True)
year B=33 p=8: 0.652 (ui gain +0.177, diag gain +0.115, energy 0.411, like_for_like=True)
year B=33 p=16: 0.882 (ui gain +0.177, diag gain +0.156, energy 0.729, like_for_like=True)
