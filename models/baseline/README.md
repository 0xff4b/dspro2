# DSPRO1 baseline

`dspro1_gradient_boosting_all_geo.joblib` is the model selected in DSPRO1 (notebook archived in
`docs/archive/dspro1/notebook/dspro1_model_v3_clean.ipynb`, section 20).

| Field | Value |
|---|---|
| Model | `GradientBoostingRegressor(random_state=42)`, target: cold rent in CHF |
| Features | east, north, elevation, area, rooms, year_built, apartments, land_area, population, oev, solar + `geo_cluster` (KMeans k = 8 on scaled LV95) |
| DSPRO1 eval (80/20 split) | R² 0.712, MAE 281 CHF, RMSE 423 CHF, MedAE 191 CHF, train/eval gap 57 CHF |
| Why selected | smallest train/eval gap among the geo candidates |
| Training data | ~4.5k fully enriched rentumo.ch listings (snapshot 2026-04-13) |
| Pickled with | scikit-learn 1.6.1 (loads with 1.7.x; version warning is logged) |

Use: `rentml.baseline.load_dspro1_baseline(ProjectPaths.discover().baseline_model).predict(test_df)`.
Predictions are NaN for objects without full GWR/swisstopo enrichment. Note that the DSPRO1 split
was random and not grouped by object, so its R² is not directly comparable with DSPRO2 test
metrics; compare both models on the same DSPRO2 test set instead.
