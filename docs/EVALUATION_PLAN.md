# DSPRO2 Evaluation Plan (pre-registration draft)

Status: **draft `eval-plan-v1`** — to be committed and tagged (`git tag eval-plan-v1`) before the first
DSPRO2 training run on frozen data. Every later change is recorded in the changelog at the end.
The constants referenced here live in `src/rentml/config.py`, so the notebook cannot drift from
this plan silently.

## 1. Data, target and population

- Unit of analysis: one unique rental object (`object_id`) after the duplicate audit. Repeated listings of
  the same object are grouped, never split across train/calibration/test.
- Target: monthly net rent `price` in CHF; models are trained on `log(price)` and all metrics are reported in
  CHF after back-transformation.
- Population: market-rent apartment listings. Cooperative, cost-based/subsidised rents and shared-flat rooms
  are excluded from the target (rent-regime label) and reported separately. Domain filter: 10–500 m²,
  300–15'000 CHF, 1–15 rooms; price-per-m² outliers per municipality (robust z > 3.5, hierarchical fallback).
- Data freeze: end of week 8. Listings observed afterwards form the temporal drift set (not used for decisions).

## 2. Validation protocol

| Set | Share | Use |
|---|---|---|
| Train | 60 % | fitting, grouped 5-fold CV for model selection and tuning (Optuna) |
| Calibration | 20 % | conformal calibration only |
| Test (hold-out) | 20 % | final comparison, touched once per research question |

- Split: `StratifiedGroupKFold(5)` folds, grouped by `object_id`, stratified by canton × price quartile;
  seed `RANDOM_STATE = 42`.
- Robustness checks: spatially grouped CV by municipality (transfer to unseen regions) and the temporal
  drift set.
- A naive baseline (median CHF/m² per municipality with district → canton → national fallback) is always reported.

## 3. Metrics

- **Primary:** MAE in CHF on the frozen hold-out test set.
- Secondary: RMSE, MedAE, MAPE, R², MAE per segment (language region, predicted/realised price band,
  listings-per-municipality bucket `<5`, `5–20`, `>20`, urban/rural), pinball loss per quantile level,
  interval coverage and mean width (overall, per calibration group, per realised price band), Moran's I of test
  residuals.

## 4. Hypotheses and tests (α = 0.05)

Primary evidence: paired bootstrap 95 % CI of the MAE difference on the test set (2'000 resamples of listings,
`N_BOOTSTRAP`). Secondary, rank-based: Wilcoxon signed-rank test on the paired absolute errors.

| RQ | Hypothesis (direction) | Test |
|---|---|---|
| RQ2 | GPBoost (boosting + nested random effects canton > district > municipality + Vecchia GP on LV95) has a lower test MAE than LightGBM with coordinates + out-of-fold target-encoded municipality, overall and in the `<5` bucket; and lower residual Moran's I | paired bootstrap CI (overall and per bucket), Wilcoxon; Moran's I with kNN weights (k = 8), 999 permutations |
| RQ1 | Each text stage lowers test MAE vs. the previous stage | paired bootstrap per stage, Holm correction across the ablation stages |
| RQ3 | Mondrian-CQR 80 % intervals reach ≥ 80 % coverage (minus a binomial tolerance) in every language region × predicted-rent band cell and are narrower on average than a constant band with the same empirical coverage | per-cell coverage with Clopper–Pearson CI; mean width comparison |
| RQ4 | QoLI shows convergent validity with independent municipal indicators and stable ranks under weight perturbation | Spearman ρ; Dirichlet weight perturbation (500 draws), median Spearman to baseline ranks |

Minimum detectable effect: computed with the bootstrap power simulation (`rentml.evaluation.power_mde`) on the
per-listing errors of the DSPRO1-equivalent baseline for test sets of 2'000–6'000 listings, power 0.8.
The result is filled in below before tagging.

| Test-set size | MDE (relative MAE reduction) |
|---|---|
| to be filled by notebook section "Evaluation plan" | |

## 5. Order of decisions

1. RQ2 is decided first on the tabular + geo stage (point model).
2. The text ablation (RQ1) runs only on the RQ2 winner.
3. Intervals (RQ3) are built only for the resulting point model; if GPBoost wins, the σ-normalised conformal
   score is used, otherwise CQR on the monotone LightGBM quantile models.
4. The what-if view is always served by the monotone-constrained LightGBM (area).

## 6. Ablation grid

`tabular` → `+geo` (coordinates, TE / random effects) → `+text features` (TF-IDF-SVD, keyword flags,
sentence embeddings) → `+structured attributes` (LLM extraction) → optional `+fused transformer` / `+image`.
Every stage is logged in MLflow (stage tag) and the ablation table is rendered automatically from the runs.

## 7. Tuning budget

- LightGBM: Optuna TPE, 60 trials, grouped 5-fold CV on the training set, objective CV MAE (CHF).
- GPBoost: two-stage fit. Covariance parameters (canton/district/municipality variances, Matérn GP
  variance and range, error variance) are estimated once on the training set in a linear mixed model with
  data-driven starting values; the boosting then runs with these parameters fixed. The number of boosting
  rounds is chosen by early stopping on a grouped validation fold inside train (learning rate 0.01, no line
  search). Reason: joint re-estimation in every round took ~12 s per round and, with gpboost's default
  starting values, collapsed the error variance to ~0 because several listings share building coordinates.
- Fused neural network: fixed architecture, early stopping on a grouped validation split inside train.

## Changelog

- v1 (draft): initial plan derived from the DSPRO2 proposal.
- v1 (draft), before tagging: GPBoost switched from joint to two-stage covariance estimation after the
  degenerate joint fit was found on the DSPRO1 data (see section 7).
- v1 (draft), before tagging: the monotone constraint covers `area` **and** `area_per_room`
  (derived from the area). With `area` alone, the what-if was not monotone (94 of 1'625 test flats
  got a lower estimate after enlarging them in the CSV development run). `rooms` has no constraint
  of its own; through `area_per_room`, more rooms at equal area can only lower that feature's
  contribution (section 5, point 4).
- v1 (draft), before tagging: the RQ2 decision rule (section 5, point 1) reads the test set, so the
  RQ1 and RQ3 test results are conditional on this pre-registered rule (the S0 test MAE of RQ1 is
  slightly optimistic when the two RQ2 models are close). Selection within RQ1 and RQ3 uses only
  CV and calibration data.
- v1 (draft), before tagging: H2c is reported descriptively. The permutation test checks each
  model's residual Moran's I against 0; the difference between the two models is not tested.
- v1 (draft), before tagging: the structured attributes (`+structured attributes` stage, S3) and
  the rent-regime filter use the LLM labels only when the LLM cache covers every description (batch
  run); otherwise they use the rule-based extractor, and the live LLM sample is an agreement check.
- v1 (draft), before tagging: RQ3 and the app use the S0 feature set (no text) in this version,
  even if the CV selects a text stage in RQ1. Serving text features needs the fitted TF-IDF,
  embedding and extraction steps in the app bundle; they follow with the database version.
- v1 (draft), before tagging: schema fixes. Half rooms and missing room counts are recovered from
  the description (text value within 0.5 of the stored count). A missing living area is not
  recovered, because the domain filter removes those objects before any text is read. Listings
  with a gross rent cannot be flagged: the DSPRO1 schema has no gross/net indicator.
- v1 (draft), before tagging: monotone constraints use LightGBM's `"intermediate"` method
  (`rentml.config.MONOTONE_CONSTRAINTS_METHOD`). The `"advanced"` method of LightGBM 4.7 let the
  estimate drop for 15–29 of 200 test flats on a fine area grid; `"intermediate"` and `"basic"` were
  exact with the same calibration MAE (261.6 / 261.1 vs. 261.8 CHF).
