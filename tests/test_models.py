"""Tests for rentml.models."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from rentml.models import (
    GPBoostRegressor,
    NaiveMunicipalityBaseline,
    fit_predict_cv,
    global_importance,
    make_lgbm,
    make_te_lgbm,
    monotone_vector,
    top_drivers,
    tree_shap,
    tune_lgbm_optuna,
)

GROUP_COLS = ["re_canton", "re_district", "re_municipality"]


def _listings(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic objects: canton + spatial surface + area effect, 15 % shared coordinates."""
    rng = np.random.default_rng(seed)
    canton, district, muni = rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(0, 4, n)
    east = 2_600_000 + canton * 30_000 + rng.uniform(0, 20_000, n)
    north = 1_200_000 + district * 10_000 + rng.uniform(0, 10_000, n)
    dup = np.flatnonzero(rng.random(n) < 0.15)
    east[dup], north[dup] = east[dup - 1], north[dup - 1]
    area = rng.uniform(30, 150, n)
    rooms = np.round(area / 30)
    rooms[::10] = np.nan
    log_price = (
        7.0
        + 0.8 * np.log(area / 80)
        + 0.25 * (canton - 1)
        + 0.15 * np.sin(east / 6000)
        + rng.normal(0, 0.08, n)
    )
    return pd.DataFrame(
        {
            "re_canton": [f"K{c}" for c in canton],
            "re_district": [f"K{c}/{d}" for c, d in zip(canton, district, strict=True)],
            "re_municipality": [
                f"K{c}/{d}/{m}" for c, d, m in zip(canton, district, muni, strict=True)
            ],
            "east": east,
            "north": north,
            "area": area,
            "rooms": rooms,
            "log_price": log_price,
        },
        index=pd.Index(np.arange(1000, 1000 + n), name="listing_id"),
    )


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return _listings()


@pytest.fixture(scope="module")
def folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(KFold(3, shuffle=True, random_state=0).split(df))


def test_naive_baseline_falls_back_below_min_count() -> None:
    X = pd.DataFrame(
        {
            "municipality_id": [1] * 5 + [2, 2],
            "district_id": [10] * 7,
            "canton": ["ZH"] * 7,
            "area": [50.0] * 7,
        }
    )
    y = np.array([1000.0] * 5 + [2000.0, 2000.0])  # 20 CHF/m² in muni 1, 40 in muni 2
    model = NaiveMunicipalityBaseline(min_count=5).fit(X, y)
    new = pd.DataFrame(
        {
            "municipality_id": [1, 2, 3],
            "district_id": [10, 10, 99],
            "canton": ["ZH", "ZH", "BE"],
            "area": [100.0] * 3,
        }
    )
    np.testing.assert_allclose(model.predict(new), [2000.0, 2000.0, 2000.0])
    assert model.rate_source(new).tolist() == ["municipality_id", "district_id", "national"]
    log_model = NaiveMunicipalityBaseline(min_count=5, log_target=True).fit(X, np.log(y))
    np.testing.assert_allclose(log_model.predict(new), np.log(model.predict(new)))


def test_monotone_vector_and_make_lgbm() -> None:
    assert monotone_vector(["te", "area", "rooms"], ["area"]) == [0, 1, 0]
    with pytest.raises(KeyError, match="living_area"):
        monotone_vector(["area"], ["living_area"])
    model = make_lgbm({"n_estimators": 5}, monotone=[0, 1])
    params = model.get_params()
    assert params["n_estimators"] == 5
    assert params["monotone_constraints"] == [0, 1]
    assert params["monotone_constraints_method"] == "intermediate"


def test_te_lgbm_builds_encoded_matrix_and_is_monotone(df: pd.DataFrame) -> None:
    pipe = make_te_lgbm(
        ["area", "rooms"], GROUP_COLS, params={"n_estimators": 30}, monotone_increasing=["area"]
    )
    pipe.fit(df[[*GROUP_COLS, "area", "rooms"]], df["log_price"])
    encoded = pipe[:-1].transform(df.head(3))
    assert list(encoded.columns) == [*GROUP_COLS, "area", "rooms"]
    assert encoded["re_canton"].dtype == float
    assert pipe.named_steps["model"].get_params()["monotone_constraints"] == [0, 0, 0, 1, 0]
    grid = pd.concat([df.head(1)] * 20, ignore_index=True).assign(area=np.linspace(30, 150, 20))
    assert np.all(np.diff(pipe.predict(grid)) >= 0)


def test_fit_predict_cv_returns_oof_and_nan_outside_folds(
    df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    oof = fit_predict_cv(
        lambda: make_lgbm({"n_estimators": 20}), df[["area"]], df["log_price"], folds[:2]
    )
    in_val = np.zeros(len(df), dtype=bool)
    for _, val in folds[:2]:
        in_val[val] = True
    assert np.isfinite(oof[in_val]).all()
    assert np.isnan(oof[~in_val]).all()


def test_tune_lgbm_optuna_returns_best_trial(
    df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    def factory(params: dict) -> lgb.LGBMRegressor:
        return make_lgbm(params | {"n_estimators": 10})

    best, study = tune_lgbm_optuna(
        df[["area", "rooms"]], df["log_price"], folds, n_trials=3, model_factory=factory
    )
    assert best == study.best_params
    assert len(study.trials) == 3
    assert 0 < study.best_value < 2000  # CHF scale, not log scale


@pytest.fixture(scope="module")
def gpb_model(df: pd.DataFrame) -> GPBoostRegressor:
    train, valid = df.iloc[:220], df.iloc[220:]
    return GPBoostRegressor(num_boost_round=60, eval_every=10, num_threads=1).fit(
        train[["area", "rooms"]],
        train["log_price"],
        train[GROUP_COLS],
        train[["east", "north"]],
        eval_set=(
            valid[["area", "rooms"]],
            valid["log_price"],
            valid[GROUP_COLS],
            valid[["east", "north"]],
        ),
        early_stopping_rounds=30,
    )


def test_gpboost_two_stage_keeps_error_variance(gpb_model: GPBoostRegressor) -> None:
    # Shared coordinates collapsed the error variance with gpboost's defaults; true value 0.0064.
    table = gpb_model.variance_components().set_index("component")
    assert table.loc["residual", "variance"] > 1e-3
    assert {"canton", "district", "municipality", "gp_variance"} <= set(table.index)
    assert table["share"].sum() == pytest.approx(1.0)
    assert np.isnan(table.loc["gp_practical_range", "share"])
    assert gpb_model.best_iteration_ in gpb_model.eval_iterations_
    assert len(gpb_model.evals_result_["valid"]["l1"]) == len(gpb_model.eval_iterations_)


def test_gpboost_predicts_variance_in_chunks_and_location_effect(
    gpb_model: GPBoostRegressor, df: pd.DataFrame
) -> None:
    # Without chunking, gpboost 1.7.4 aborts: 300 rows > 3 + 6 + 24 training levels.
    assert len(df) > gpb_model.n_levels_
    mean, var = gpb_model.predict(df, df[GROUP_COLS], df[["east", "north"]], return_var=True)
    np.testing.assert_allclose(mean, gpb_model.predict(df, df[GROUP_COLS], df[["east", "north"]]))
    assert (var > 0).all()
    location = pd.Series(mean - gpb_model.predict_fixed_effects(df), index=df.index)
    by_canton = location.groupby(df["re_canton"]).mean()
    assert by_canton["K2"] > by_canton["K0"] + 0.2  # simulated +0.5 log rent


def test_gpboost_reuses_cov_pars(gpb_model: GPBoostRegressor, df: pd.DataFrame) -> None:
    part = df.iloc[:150]
    model = GPBoostRegressor(num_boost_round=20, num_threads=1).fit(
        part[["area", "rooms"]],
        part["log_price"],
        part[GROUP_COLS],
        part[["east", "north"]],
        cov_pars=gpb_model.cov_,
    )
    assert model.cov_ is gpb_model.cov_
    assert model.best_iteration_ is None
    assert model.booster_.gp_model.get_cov_pars(format_pandas=True).iloc[0].to_numpy() == (
        pytest.approx(gpb_model.cov_.drop("error_variance").to_numpy())
    )


def test_shap_importance_and_drivers(df: pd.DataFrame) -> None:
    X = df[["area", "rooms"]]
    model = make_lgbm({"n_estimators": 30}).fit(X, df["log_price"])
    expl = tree_shap(model, X, max_rows=50, seed=0)
    assert expl.values.shape == (50, 2)
    assert list(expl.instance_names) == sorted(expl.instance_names)
    np.testing.assert_allclose(
        expl.base_values + expl.values.sum(axis=1),
        model.predict(X.loc[expl.instance_names]),
        rtol=1e-6,
    )
    importance = global_importance(expl)
    assert importance["feature"].iloc[0] == "area"
    assert importance["share"].sum() == pytest.approx(1.0)
    drivers = top_drivers(tree_shap(model, X.iloc[[0]]), 0, k=5)
    assert len(drivers) == 2  # capped at the number of features
    assert drivers["approx_pct"].iloc[0] == pytest.approx(
        100 * np.expm1(drivers["shap_log"].iloc[0])
    )
