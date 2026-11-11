<!-- session 20261108-203600, data csv, run full; one run per stage chosen by cv_mae; CI: bootstrap (2000 draws); p_vs_prev unadjusted (Holm-adjusted RQ1 tests: section 17) -->
| stage       | run_name       |   cv_mae |   test_mae |   test_rmse |   test_mape |   test_r2 |   test_mae_lt5 |   runs_in_stage |   mae_ci_low |   mae_ci_high |   gain_vs_prev |   p_vs_prev |
|:------------|:---------------|---------:|-----------:|------------:|------------:|----------:|---------------:|----------------:|-------------:|--------------:|---------------:|------------:|
| baseline    | naive          |      346 |        377 |         589 |        21.5 |     0.558 |            389 |               1 |          356 |           401 |            –   |      –      |
| dspro1      | dspro1_knn_oof |      277 |        278 |         486 |        15.2 |     0.698 |            273 |               1 |          260 |           299 |          +99.9 |      0.0005 |
| tabular     | lgbm_tab       |      304 |        305 |         528 |        16.7 |     0.644 |            338 |               1 |          286 |           327 |          -27.0 |      0.0005 |
| tabular+geo | lgbm_te        |      262 |        270 |         474 |        14.9 |     0.713 |            266 |               1 |          253 |           291 |          +34.8 |      0.0005 |
| fused       | fused_nn       |      290 |        294 |         550 |        15.9 |     0.613 |            292 |               1 |          274 |           319 |          -24.5 |      0.0005 |
