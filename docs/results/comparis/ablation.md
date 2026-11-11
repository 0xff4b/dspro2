<!-- session 20261108-193155, data comparis, run full; one run per stage chosen by cv_mae; CI: bootstrap (2000 draws); p_vs_prev unadjusted (Holm-adjusted RQ1 tests: section 17) -->
| stage       | run_name       |   cv_mae |   test_mae |   test_rmse |   test_mape |   test_r2 |   test_mae_lt5 |   runs_in_stage |   mae_ci_low |   mae_ci_high |   gain_vs_prev |   p_vs_prev |
|:------------|:---------------|---------:|-----------:|------------:|------------:|----------:|---------------:|----------------:|-------------:|--------------:|---------------:|------------:|
| baseline    | naive          |      441 |        429 |         729 |        20.7 |     0.703 |            439 |               1 |          411 |           447 |            –   |      –      |
| dspro1      | dspro1_knn_oof |      337 |        322 |         640 |        14.5 |     0.771 |            301 |               1 |          304 |           340 |         +107.3 |      0.0005 |
| tabular     | lgbm_tab       |      369 |        355 |         681 |        16.1 |     0.741 |            367 |               1 |          337 |           374 |          -33.3 |      0.0005 |
| tabular+geo | lgbm_te        |      321 |        308 |         608 |        14.0 |     0.794 |            292 |               1 |          291 |           325 |          +46.9 |      0.0005 |
| text        | lgbm_te+s1     |      288 |        282 |         568 |        12.7 |     0.819 |            273 |               2 |          266 |           298 |          +26.5 |      0.0005 |
| attributes  | lgbm_te+s3     |      287 |        282 |         577 |        12.7 |     0.814 |            271 |               1 |          266 |           299 |           +0.1 |      0.973  |
| fused       | fused_nn       |      334 |        322 |         617 |        14.6 |     0.787 |            333 |               1 |          306 |           339 |          -40.4 |      0.0005 |
| portal      | lgbm_te+portal |      285 |        280 |         576 |        12.6 |     0.814 |            267 |               1 |          264 |           297 |          +42.0 |      0.0005 |
