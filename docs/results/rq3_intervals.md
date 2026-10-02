<!-- RQ2 winner lgbm_te, data csv, run full, session 20261002-221320 -->
| method | source | calibration | primary | calib_coverage | coverage | cp_low | cp_high | below | above | mean_width | median_width | interval_score | min_cell_coverage | cells_below_nominal | n_cells | const_width_same_cov | width_ratio |
|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_quantile | quantile | raw q10-q90 | False | 0.821 | 0.810 | 0.790 | 0.829 | 0.085 | 0.105 | 857 | 723 | 1354 | 0.755 | 0 | 12 | 723 | 1.185 |
| global_quantile | quantile | CQR, global | False | 0.801 | 0.786 | 0.766 | 0.806 | 0.103 | 0.111 | 819 | 691 | 1352 | 0.736 | 1 | 12 | 694 | 1.180 |
| mondrian_quantile | quantile | CQR, Mondrian | True | 0.809 | 0.802 | 0.782 | 0.822 | 0.099 | 0.098 | 851 | 701 | 1357 | 0.769 | 0 | 12 | 718 | 1.185 |
| raw_gpboost | gpboost | raw Gaussian mu +/- z sigma | False | 0.881 | 0.882 | 0.866 | 0.898 | 0.059 | 0.058 | 975 | 898 | 1353 | 0.769 | 0 | 12 | 966 | 1.009 |
| global_gpboost | gpboost | sigma-normalised, global | False | 0.801 | 0.804 | 0.784 | 0.823 | 0.104 | 0.092 | 786 | 724 | 1321 | 0.692 | 1 | 12 | 720 | 1.092 |
| mondrian_gpboost | gpboost | sigma-normalised, Mondrian | False | 0.812 | 0.803 | 0.783 | 0.822 | 0.108 | 0.089 | 815 | 722 | 1315 | 0.692 | 1 | 12 | 718 | 1.135 |
| constant_conformal | lgbm_te | constant +/- h (conformal) | False | 0.801 | 0.822 | 0.802 | 0.840 | 0.076 | 0.103 | 746 | 746 | 1417 | 0.462 | 3 | 12 | 745 | 1.001 |

