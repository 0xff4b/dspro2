<!-- session 20260927-193408, data csv, run full; one run per stage chosen by cv_mae; CI: cluster bootstrap (2000 draws); p_vs_prev unadjusted (Holm-adjusted RQ1 tests: section 17) -->
| stage | run_name | cv_mae | test_mae | test_rmse | test_mape | test_r2 | test_mae_lt5 | runs_in_stage | mae_ci_low | mae_ci_high | gain_vs_prev | p_vs_prev |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | naive | 352 | 355 | 580 | 19.8 | 0.618 | 376 | 1 | 333 | 379 | – | – |
| dspro1 | dspro1_knn_oof | 277 | 273 | 505 | 14.8 | 0.711 | 279 | 1 | 254 | 294 | +82.1 | 0.0005 |
| tabular | lgbm_tab | 304 | 302 | 549 | 16.3 | 0.658 | 346 | 1 | 282 | 326 | -29.8 | 0.0005 |
| tabular+geo | lgbm_te | 265 | 261 | 483 | 14.2 | 0.735 | 260 | 1 | 242 | 281 | +41.8 | 0.0005 |
| fused | fused_nn | 296 | 288 | 527 | 15.0 | 0.685 | 295 | 1 | 267 | 310 | -27.2 | 0.0005 |
