"""Tests for rentml.models."""

import pickle

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold

from rentml.config import MONOTONE_CONSTRAINTS_METHOD
from rentml.models import (
    DEFAULT_LGBM_PARAMS,
    LGBM_SEARCH_SPACE,
    NaiveMunicipalityBaseline,
    ParamRange,
    fit_predict_cv,
    make_lgbm,
    make_te_lgbm,
    tune_lgbm_optuna,
)


@pytest.fixture
def hierarchy_frame() -> pd.DataFrame:
    """Municipality 1 has 6 rows (own rate), municipality 2 has 2 (district fallback)."""
    rates = [20.0] * 6 + [40.0, 40.0] + [30.0] * 5 + [10.0] * 5
    return pd.DataFrame(
        {
            "area": np.full(len(rates), 50.0),
            "price": np.array(rates) * 50.0,
            "municipality_id": [1] * 6 + [2] * 2 + [3] * 5 + [4] * 5,
            "district_id": [10] * 8 + [11] * 5 + [12] * 5,
            "canton": ["ZH"] * 13 + ["BE"] * 5,
        }
    )


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 400
    X = pd.DataFrame(
        {
            "area": rng.uniform(25, 160, n),
            "rooms": rng.integers(1, 6, n).astype(float),
            "east": rng.uniform(0, 1, n),
            "noise": rng.normal(size=n),
        }
    )
    y_log = pd.Series(
        np.log(18.0 * X["area"]) + 0.3 * X["east"] + rng.normal(0, 0.1, n), name="log_price"
    )
    return X, y_log


def test_baseline_uses_fallback_chain(hierarchy_frame: pd.DataFrame) -> None:
    model = NaiveMunicipalityBaseline(min_count=5).fit(hierarchy_frame)
    district_10 = np.median([20.0] * 6 + [40.0, 40.0])
    new = pd.DataFrame(
        {
            "area": [100.0, 100.0, 100.0, 100.0],
            "municipality_id": [1, 2, 99, 99],
            "district_id": [10, 10, 99, 99],
            "canton": ["ZH", "ZH", "BE", "TI"],
        }
    )
    rates = model.predict_rate(new)
    assert rates[0] == pytest.approx(20.0)
    assert rates[1] == pytest.approx(district_10)
    assert rates[2] == pytest.approx(10.0)  # canton BE
    assert rates[3] == pytest.approx(model.global_rate_)
    np.testing.assert_allclose(model.predict(new), rates * 100.0)
    assert model.rate_source(new).tolist() == ["municipality_id", "district_id", "canton", "global"]


def test_baseline_accepts_explicit_y_and_skips_missing_levels(
    hierarchy_frame: pd.DataFrame,
) -> None:
    X = hierarchy_frame.drop(columns=["price", "canton"])
    model = NaiveMunicipalityBaseline().fit(X, hierarchy_frame["price"].to_numpy())
    assert "canton" not in model.rates_
    pred = model.predict(X.assign(area=[np.nan] + [50.0] * (len(X) - 1)))
    assert np.isnan(pred[0])
    assert pred[1] == pytest.approx(20.0 * 50.0)
    assert model.score(X, hierarchy_frame["price"]) > 0.0


def test_baseline_log_target_mode_works_with_fit_predict_cv(
    hierarchy_frame: pd.DataFrame,
) -> None:
    y_log = np.log(hierarchy_frame["price"])
    model = NaiveMunicipalityBaseline(log_target=True).fit(hierarchy_frame, y_log)
    chf = NaiveMunicipalityBaseline().fit(hierarchy_frame)
    np.testing.assert_allclose(np.exp(model.predict(hierarchy_frame)), chf.predict(hierarchy_frame))
    folds = list(KFold(3, shuffle=True, random_state=0).split(hierarchy_frame))
    oof = fit_predict_cv(
        lambda: NaiveMunicipalityBaseline(log_target=True), hierarchy_frame, y_log, folds
    )
    assert np.isfinite(oof).all()


def test_baseline_rejects_log_target_and_missing_area(hierarchy_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="log"):
        NaiveMunicipalityBaseline().fit(hierarchy_frame, np.log(hierarchy_frame["price"]))
    with pytest.raises(KeyError):
        NaiveMunicipalityBaseline().fit(hierarchy_frame.drop(columns="area"))
    with pytest.raises(ValueError, match="rows"):
        NaiveMunicipalityBaseline().fit(hierarchy_frame, hierarchy_frame["price"].iloc[:-1])


def test_make_lgbm_defaults_and_overrides() -> None:
    model = make_lgbm()
    params = model.get_params()
    for key in ("objective", "n_estimators", "learning_rate", "subsample_freq"):
        assert params[key] == DEFAULT_LGBM_PARAMS[key]
    assert params["random_state"] == 42
    assert "monotone_constraints" not in params or params["monotone_constraints"] is None
    tuned = make_lgbm({"num_leaves": 63, "random_state": 7}, seed=1).get_params()
    assert tuned["num_leaves"] == 63
    assert tuned["random_state"] == 7
    mono = make_lgbm(monotone=[1, 0, -1]).get_params()
    assert mono["monotone_constraints"] == [1, 0, -1]
    assert mono["monotone_constraints_method"] == "intermediate"


def test_make_lgbm_monotone_constraint_holds(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y_log = regression_data
    model = make_lgbm({"n_estimators": 100}, monotone=[1, 0, 0, 0]).fit(X, y_log)
    grid = pd.concat([X.iloc[:20]] * 30, ignore_index=True)
    grid["area"] = np.repeat(np.linspace(20, 200, 30), 20)
    pred = model.predict(grid).reshape(30, 20)
    assert (np.diff(pred, axis=0) >= -1e-12).all()


def test_make_lgbm_monotone_holds_for_derived_feature_with_deep_trees() -> None:
    # The what-if changes area and the derived area_per_room together. LightGBM 4.7's "advanced"
    # method violated the constraints on the DSPRO1 data (not reproducible on synthetic data; the
    # notebook's sanity check in section 20 guards the real case), so this pins the joint check
    # and the configured method.
    rng = np.random.default_rng(42)
    n = 3000
    rooms = rng.integers(1, 7, n).astype(float)
    area = rng.uniform(20, 220, n)
    X = pd.DataFrame(
        {"area": area, "area_per_room": area / rooms, "rooms": rooms, "x": rng.normal(size=n)}
    )
    y = np.log(area) * 0.8 + 0.3 * np.sin(X["x"] * 3) * np.log(area) + rng.normal(0, 0.15, n)
    params = {"n_estimators": 300, "num_leaves": 58, "min_child_samples": 5, "learning_rate": 0.05}
    model = make_lgbm(params, monotone=[1, 1, 0, 0]).fit(X, y)
    assert model.get_params()["monotone_constraints_method"] == MONOTONE_CONSTRAINTS_METHOD
    rows = X.sample(100, random_state=0)
    grid_area = np.arange(20.0, 221.0, 2.0)
    grid = rows.loc[rows.index.repeat(len(grid_area))].copy()
    grid["area"] = np.tile(grid_area, len(rows))
    grid["area_per_room"] = grid["area"] / grid["rooms"]
    pred = model.predict(grid).reshape(len(rows), len(grid_area))
    assert (np.diff(pred, axis=1) >= -1e-12).all()


def test_make_lgbm_rejects_invalid_monotone_values() -> None:
    with pytest.raises(ValueError, match="monotone"):
        make_lgbm(monotone=[2, 0])


def test_fit_predict_cv_returns_out_of_fold_predictions(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y_log = regression_data
    folds = list(KFold(4, shuffle=True, random_state=0).split(X))
    oof = fit_predict_cv(DummyRegressor, X, y_log, folds)
    assert oof.shape == (len(X),)
    for train_idx, val_idx in folds:
        np.testing.assert_allclose(oof[val_idx], y_log.iloc[train_idx].mean())
    lgbm_oof = fit_predict_cv(lambda: make_lgbm({"n_estimators": 50}), X, y_log, folds)
    assert np.isfinite(lgbm_oof).all()
    assert np.corrcoef(lgbm_oof, y_log)[0, 1] > 0.8


def test_fit_predict_cv_passes_fold_specific_fit_kwargs(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y_log = regression_data
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))
    weights = np.where(np.arange(len(X)) % 2 == 0, 1.0, 0.0)
    oof = fit_predict_cv(
        DummyRegressor,
        X,
        y_log,
        folds,
        fit_kwargs=lambda tr, _va: {"sample_weight": weights[tr]},
    )
    train_idx, val_idx = folds[0]
    expected = np.average(y_log.iloc[train_idx], weights=weights[train_idx])
    np.testing.assert_allclose(oof[val_idx], expected)


def test_fit_predict_cv_edge_cases(regression_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y_log = regression_data
    with pytest.raises(ValueError, match="folds"):
        fit_predict_cv(DummyRegressor, X, y_log, [])
    with pytest.raises(ValueError, match="rows"):
        fit_predict_cv(DummyRegressor, X, y_log.iloc[:-1], [(np.arange(10), np.arange(10, 20))])
    partial = fit_predict_cv(DummyRegressor, X, y_log, [(np.arange(100), np.arange(100, 150))])
    assert np.isnan(partial[:100]).all()
    assert np.isfinite(partial[100:150]).all()


def test_tune_lgbm_optuna_runs_small_study(regression_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y_log = regression_data
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))
    small = {"n_estimators": ParamRange(20, 60, integer=True, step=20)}
    original = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        best, study = tune_lgbm_optuna(
            X, y_log, folds, n_trials=3, monotone=[1, 0, 0, 0], search_space=small
        )
        assert optuna.logging.get_verbosity() == optuna.logging.INFO  # restored
    finally:
        optuna.logging.set_verbosity(original)
    assert len(study.trials) == 3
    assert set(LGBM_SEARCH_SPACE) <= set(best)
    assert 20 <= best["n_estimators"] <= 60
    assert np.isfinite(study.best_value) and study.best_value > 0
    assert len(study.best_trial.user_attrs["fold_mae"]) == 3
    refit = make_lgbm(best).fit(X, y_log)
    assert np.isfinite(refit.predict(X)).all()
    _, same = tune_lgbm_optuna(X, y_log, folds, n_trials=3, search_space=small, prune=False)
    _, again = tune_lgbm_optuna(X, y_log, folds, n_trials=3, search_space=small, prune=False)
    assert same.best_params == again.best_params  # seeded sampler -> reproducible


def test_tune_lgbm_optuna_rejects_invalid_budget(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y_log = regression_data
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))
    with pytest.raises(ValueError, match="n_trials"):
        tune_lgbm_optuna(X, y_log, folds, n_trials=0)
    with pytest.raises(ValueError, match="folds"):
        tune_lgbm_optuna(X, y_log, [], n_trials=1)
    with pytest.raises(RuntimeError, match="completed"):
        tune_lgbm_optuna(X, y_log, folds, n_trials=1, timeout=1e-9)


@pytest.fixture
def te_frame(regression_data: tuple[pd.DataFrame, pd.Series]) -> tuple[pd.DataFrame, pd.Series]:
    X, y_log = regression_data
    rng = np.random.default_rng(7)
    canton = rng.choice(["ZH", "BE"], len(X))
    muni = rng.integers(1, 15, len(X))
    frame = X.assign(
        re_canton=canton,
        re_municipality=[f"{c}/{m}" for c, m in zip(canton, muni, strict=True)],
        object_id=[f"obj_{i // 2}" for i in range(len(X))],
    )
    return frame, y_log + np.where(canton == "ZH", 0.3, 0.0)


def test_make_te_lgbm_encodes_out_of_fold_and_applies_monotone(
    te_frame: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y_log = te_frame
    te_cols = ["re_canton", "re_municipality"]
    pipe = make_te_lgbm(
        ["area", "rooms", "east"],
        te_cols,
        params={"n_estimators": 60},
        monotone_increasing=["area"],
    )
    assert pipe.named_steps["lgbm"].get_params()["monotone_constraints"] == [0, 0, 1, 0, 0]
    prep = pipe.named_steps["prep"]
    train_view = prep.fit_transform(X, y_log)
    assert list(train_view.columns) == [
        "te_re_canton",
        "te_re_municipality",
        "area",
        "rooms",
        "east",
    ]
    assert not np.allclose(
        train_view["te_re_municipality"], prep.transform(X)["te_re_municipality"]
    )
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))
    oof = fit_predict_cv(
        lambda: make_te_lgbm(["area"], te_cols, params={"n_estimators": 60}), X, y_log, folds
    )
    assert np.isfinite(oof).all()
    assert np.corrcoef(oof, y_log)[0, 1] > 0.8


def test_make_te_lgbm_requires_group_column(te_frame: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y_log = te_frame
    with pytest.raises(ValueError, match="column"):  # raised by ColumnTransformer
        make_te_lgbm(["area"], ["re_canton"]).fit(X.drop(columns="object_id"), y_log)
    ungrouped = make_te_lgbm(["area"], ["re_canton"], group_col=None, params={"n_estimators": 20})
    assert np.isfinite(ungrouped.fit(X.drop(columns="object_id"), y_log).predict(X)).all()


def test_tune_lgbm_optuna_accepts_model_factory(te_frame: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y_log = te_frame
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))
    small = {"n_estimators": ParamRange(20, 40, integer=True, step=20)}
    best, study = tune_lgbm_optuna(
        X,
        y_log,
        folds,
        n_trials=2,
        search_space=small,
        model_factory=lambda p: make_te_lgbm(["area", "rooms"], ["re_canton"], params=p),
    )
    assert len(study.trials) == 2
    assert best["n_estimators"] in (20, 40)


def test_tune_lgbm_optuna_keeps_base_params_fixed(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y_log = regression_data
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))
    explicit = {"n_estimators": ParamRange(40, 80, integer=True, step=40)}
    best, study = tune_lgbm_optuna(
        X, y_log, folds, n_trials=2, base_params={"n_estimators": 30}, search_space=explicit
    )
    assert all("n_estimators" not in t.params for t in study.trials)
    assert best["n_estimators"] == 30
    assert "learning_rate" in study.best_params  # the other keys are still tuned


def test_tune_lgbm_optuna_objective_is_cv_mae_in_chf(
    regression_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y_log = regression_data
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))
    small = {"n_estimators": ParamRange(20, 60, integer=True, step=20)}
    best, study = tune_lgbm_optuna(
        X, y_log, folds, n_trials=2, search_space=small, prune=False, base_params={"n_jobs": 1}
    )
    oof = fit_predict_cv(lambda: make_lgbm(best), X, y_log, folds)
    y_chf = np.exp(y_log.to_numpy())
    expected = np.mean([np.mean(np.abs(y_chf[va] - np.exp(oof[va]))) for _, va in folds])
    assert study.best_value == pytest.approx(expected, rel=1e-6)
    assert study.best_value > 10.0  # CHF scale, not log scale (~0.1)


def test_make_te_lgbm_predicts_frames_without_group_column(
    te_frame: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y_log = te_frame
    pipe = make_te_lgbm(
        ["area", "rooms"], ["re_canton", "re_municipality"], params={"n_estimators": 30}
    )
    pipe.fit(X, y_log)
    new = X.drop(columns="object_id").iloc[:20]
    pred = pipe.predict(new)
    np.testing.assert_allclose(pred, pipe.predict(X.iloc[:20]))
    assert "object_id" not in new.columns  # the caller's frame is not modified
    restored = pickle.loads(pickle.dumps(pipe))  # bundles are saved with joblib/pickle
    np.testing.assert_allclose(restored.predict(new), pred)
