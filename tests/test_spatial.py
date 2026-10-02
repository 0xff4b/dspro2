"""Tests for rentml.spatial (Moran's I, k-NN weights, GPBoost wrapper)."""

import logging
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rentml.spatial import (
    GPBoostConfig,
    GPBoostRegressor,
    MoranResult,
    knn_weights,
    morans_i,
)

SEED = 42
GROUP_COLS = ["re_canton", "re_district", "re_municipality"]
# Small and fast; 2 threads keep CI load low (multi-threaded fits are not bit-reproducible).
FAST_CONFIG = GPBoostConfig(num_boost_round=10, learning_rate=0.2, num_threads=2)
RentData = tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]


def _grid_coords(side: int = 20) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(side, dtype=float), np.arange(side, dtype=float))
    return np.column_stack([xs.ravel(), ys.ravel()]) * 100.0


def test_knn_weights_row_standardised_without_self() -> None:
    coords = _grid_coords(6)
    w = knn_weights(coords, k=4)
    assert sparse.issparse(w)
    assert w.shape == (36, 36)
    np.testing.assert_allclose(np.asarray(w.sum(axis=1)).ravel(), 1.0)
    assert np.all(w.diagonal() == 0.0)
    assert np.all(np.diff(w.indptr) == 4)


def test_knn_weights_duplicate_coordinates_exclude_self() -> None:
    coords = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    w = knn_weights(coords, k=2)
    assert np.all(w.diagonal() == 0.0)
    # The two other listings at the same address are the nearest neighbours of listing 0.
    assert set(w[0].indices) == {1, 2}


@pytest.mark.parametrize("k", [0, 5])
def test_knn_weights_invalid_k_raises(k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        knn_weights(_grid_coords(2)[:5], k=k)


def test_knn_weights_nan_coordinates_raise() -> None:
    coords = _grid_coords(3)
    coords[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        knn_weights(coords, k=2)


def test_morans_i_detects_spatial_clustering() -> None:
    rng = np.random.default_rng(SEED)
    coords = _grid_coords(20)
    smooth = np.sin(coords[:, 0] / 400.0) + np.cos(coords[:, 1] / 500.0)
    values = smooth + rng.normal(0.0, 0.2, len(coords))
    result = morans_i(values, coords, k=8, permutations=199, seed=SEED)
    assert isinstance(result, MoranResult)
    assert result.I > 0.5
    assert result.p_value == pytest.approx(1.0 / 200.0)
    assert result.z > 3.0
    assert result.expected == pytest.approx(-1.0 / (len(values) - 1))
    assert result.n == len(values)


def test_morans_i_random_values_not_significant() -> None:
    rng = np.random.default_rng(SEED)
    coords = _grid_coords(20)
    values = rng.normal(size=len(coords))
    result = morans_i(values, coords, k=8, permutations=499, seed=SEED)
    assert abs(result.I - result.expected) < 0.1
    assert result.p_value > 0.05


def test_morans_i_is_deterministic_for_fixed_seed() -> None:
    rng = np.random.default_rng(SEED)
    coords = rng.uniform(0, 1000, (150, 2))
    values = rng.normal(size=150)
    first = morans_i(values, coords, permutations=99, seed=7)
    second = morans_i(values, coords, permutations=99, seed=7)
    assert first == second


def test_morans_i_matches_dense_formula_without_permutations() -> None:
    rng = np.random.default_rng(SEED)
    coords = rng.uniform(0, 1000, (80, 2))
    values = rng.normal(size=80)
    w = knn_weights(coords, k=5)
    dense = w.toarray()
    z = values - values.mean()
    expected_i = len(z) / dense.sum() * (z @ dense @ z) / (z @ z)
    result = morans_i(values, coords, k=5, permutations=0)
    observed = result.I
    assert observed == pytest.approx(expected_i)
    assert np.isnan(result.p_value)
    assert np.isnan(result.z)


def test_morans_i_negative_autocorrelation_two_sided() -> None:
    coords = _grid_coords(12)
    checkerboard = ((coords[:, 0] + coords[:, 1]) / 100.0) % 2
    result = morans_i(checkerboard, coords, k=4, permutations=199, seed=SEED)
    # Border cells get diagonal neighbours (same colour), so I is not exactly -1.
    assert result.I < -0.5
    assert result.p_value == pytest.approx(1.0 / 200.0)


def test_morans_i_constant_values_raise() -> None:
    with pytest.raises(ValueError, match="constant"):
        morans_i(np.ones(25), _grid_coords(5), k=4, permutations=9)


def test_morans_i_length_mismatch_and_nan_raise() -> None:
    coords = _grid_coords(5)
    with pytest.raises(ValueError, match="differ in length"):
        morans_i(np.arange(10.0), coords)
    values = np.arange(25.0)
    values[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        morans_i(values, coords, k=4)


def _rent_data(n: int = 300, seed: int = SEED) -> RentData:
    """Synthetic listings: canton effect + smooth spatial surface + area effect, 15 % duplicates."""
    rng = np.random.default_rng(seed)
    canton = rng.integers(0, 3, n)
    district = rng.integers(0, 2, n)
    muni = rng.integers(0, 4, n)
    index = pd.Index(np.arange(1000, 1000 + n), name="listing_id")
    groups = pd.DataFrame(
        {
            "re_canton": [f"K{c}" for c in canton],
            "re_district": [f"K{c}/{d}" for c, d in zip(canton, district, strict=True)],
            "re_municipality": [
                f"K{c}/{d}/{m}" for c, d, m in zip(canton, district, muni, strict=True)
            ],
        },
        index=index,
    )
    east = 2_600_000 + canton * 30_000 + rng.uniform(0, 20_000, n)
    north = 1_200_000 + district * 10_000 + rng.uniform(0, 10_000, n)
    dup = np.flatnonzero(rng.random(n) < 0.15)
    east[dup], north[dup] = east[dup - 1], north[dup - 1]
    coords = pd.DataFrame({"east": east, "north": north}, index=index)
    area = rng.uniform(30, 150, n)
    X = pd.DataFrame({"area": area, "rooms": np.round(area / 30)}, index=index)
    X.loc[X.index[::10], "rooms"] = np.nan
    y = (
        7.0
        + 0.8 * np.log(area / 80)
        + 0.25 * (canton - 1)
        + 0.15 * np.sin(east / 6000)
        + rng.normal(0, 0.08, n)
    )
    return X, pd.Series(y, index=index, name="log_price"), groups, coords


@pytest.fixture(scope="module")
def rent_data() -> RentData:
    return _rent_data()


@pytest.fixture(scope="module")
def fitted(
    rent_data: RentData,
) -> GPBoostRegressor:
    X, y, groups, coords = rent_data
    return GPBoostRegressor(FAST_CONFIG).fit(X, y, groups, coords)


def test_gpboost_config_booster_params() -> None:
    params = GPBoostConfig().booster_params()
    assert params["learning_rate"] == 0.05
    assert params["seed"] == 42
    assert params["verbose"] == -1
    assert params["line_search_step_length"] is True
    assert "group_cols" not in params


def test_gpboost_predict_mean_and_variance(fitted: GPBoostRegressor, rent_data: RentData) -> None:
    X, y, groups, coords = rent_data
    mean, var = fitted.predict(X, groups, coords, return_var=True)
    assert mean.shape == var.shape == (len(X),)
    assert np.all(np.isfinite(mean)) and np.all(var > 0)
    assert np.corrcoef(mean, y)[0, 1] > 0.9
    residual = fitted.variance_components().set_index("component").loc["residual", "variance"]
    assert np.all(var >= residual * 0.999)
    np.testing.assert_allclose(fitted.predict(X, groups, coords), mean)
    assert fitted.fit_time_s_ > 0


def test_gpboost_predicts_more_rows_than_levels_in_chunks(
    fitted: GPBoostRegressor, rent_data: RentData
) -> None:
    # Unchunked, gpboost 1.7.4 aborts here (C++ assertion): 300 rows > 3 + 6 + 24 levels.
    X, _, groups, coords = rent_data
    assert fitted._chunk_size() == sum(fitted.n_levels_.values()) < len(X)
    _, var_all = fitted.predict(X, groups, coords, return_var=True)
    _, var_one = fitted.predict(
        X.iloc[[250]], groups.iloc[[250]], coords.iloc[[250]], return_var=True
    )
    assert var_one[0] == pytest.approx(var_all[250])


def test_gpboost_unseen_and_missing_levels_get_prior(
    fitted: GPBoostRegressor, rent_data: RentData, caplog: pytest.LogCaptureFixture
) -> None:
    X, _, groups, coords = rent_data
    rows = X.index[:3]
    new_groups = groups.loc[rows].copy()
    new_groups.iloc[0] = ["NEW", "NEW/1", "NEW/1/1"]
    new_groups.iloc[1, 2] = pd.NA
    _, var_seen = fitted.predict(X.loc[rows], groups.loc[rows], coords.loc[rows], return_var=True)
    with caplog.at_level(logging.WARNING, logger="rentml.spatial"):
        _, var_new = fitted.predict(X.loc[rows], new_groups, coords.loc[rows], return_var=True)
    assert "missing group keys" in caplog.text
    canton_var = fitted.variance_components().set_index("component").loc["canton", "variance"]
    assert var_new[0] > var_seen[0] + 0.5 * canton_var
    assert var_new[1] >= var_seen[1] * 0.999
    assert var_new[2] == pytest.approx(var_seen[2])


def test_gpboost_variance_components_table(fitted: GPBoostRegressor) -> None:
    table = fitted.variance_components().set_index("component")
    expected = {"residual", "canton", "district", "municipality", "gp_variance", "gp_range"}
    assert expected | {"gp_practical_range"} == set(table.index)
    assert table["share"].sum() == pytest.approx(1.0)
    assert table.loc[["gp_range", "gp_practical_range"], "share"].isna().all()
    assert table.loc["gp_practical_range", "variance"] == pytest.approx(
        2.7389 * table.loc["gp_range", "variance"]
    )
    assert table.loc["canton", "share"] > table.loc["municipality", "share"]


def test_gpboost_pickle_roundtrip_keeps_gp_model(
    fitted: GPBoostRegressor, rent_data: RentData
) -> None:
    X, _, groups, coords = rent_data
    restored = pickle.loads(pickle.dumps(fitted))
    mean, var = fitted.predict(X, groups, coords, return_var=True)
    mean_r, var_r = restored.predict(X, groups, coords, return_var=True)
    np.testing.assert_allclose(mean_r, mean)
    np.testing.assert_allclose(var_r, var)


def test_gpboost_single_thread_fit_is_deterministic() -> None:
    X, y, groups, coords = _rent_data(n=150, seed=3)
    cfg = replace(FAST_CONFIG, num_boost_round=5, num_threads=1)
    first = GPBoostRegressor(cfg).fit(X, y, groups, coords).predict(X, groups, coords)
    second = GPBoostRegressor(cfg).fit(X, y, groups, coords).predict(X, groups, coords)
    np.testing.assert_array_equal(first, second)


def test_gpboost_groups_only_model_with_early_stopping(rent_data: RentData) -> None:
    X, y, groups, coords = rent_data
    train, valid = X.index[:220], X.index[220:]
    cfg = replace(FAST_CONFIG, use_gp=False, num_boost_round=300, learning_rate=0.3)
    model = GPBoostRegressor(cfg).fit(
        X.loc[train],
        y.loc[train],
        groups.loc[train],
        None,
        eval_set=(X.loc[valid], y.loc[valid], groups.loc[valid], None),
        early_stopping_rounds=5,
    )
    assert model.best_iteration_ is not None and model.best_iteration_ < 300
    assert "l1" in model.evals_result_["valid"]
    assert set(model.variance_components()["component"]) == {
        "residual",
        "canton",
        "district",
        "municipality",
    }
    assert model.predict(X.loc[valid], groups.loc[valid], None).shape == (len(valid),)


def test_gpboost_gp_only_model(rent_data: RentData) -> None:
    X, y, _, coords = rent_data
    model = GPBoostRegressor(replace(FAST_CONFIG, group_cols=())).fit(X, y, None, coords.to_numpy())
    table = model.variance_components()
    assert set(table["component"]) == {"residual", "gp_variance", "gp_range", "gp_practical_range"}
    _, var = model.predict(X.iloc[:5], None, coords.to_numpy()[:5], return_var=True)
    assert np.all(var > 0)


def test_gpboost_predict_before_fit_raises(rent_data: RentData) -> None:
    X, _, groups, coords = rent_data
    with pytest.raises(RuntimeError, match="not fitted"):
        GPBoostRegressor().predict(X, groups, coords)
    with pytest.raises(RuntimeError, match="not fitted"):
        GPBoostRegressor().variance_components()


def test_gpboost_invalid_inputs_raise(rent_data: RentData) -> None:
    X, y, groups, coords = rent_data
    model = GPBoostRegressor(FAST_CONFIG)
    with pytest.raises(ValueError, match="not aligned"):
        model.fit(X, y, groups.iloc[::-1], coords)
    with pytest.raises(KeyError, match="re_municipality"):
        model.fit(X, y, groups.drop(columns="re_municipality"), coords)
    with pytest.raises(TypeError, match="non-numeric"):
        model.fit(X.assign(kind="flat"), y, groups, coords)
    bad_coords = coords.copy()
    bad_coords.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X, y, groups, bad_coords)
    with pytest.raises(ValueError, match="group_cols and/or use_gp"):
        GPBoostRegressor(replace(FAST_CONFIG, group_cols=(), use_gp=False)).fit(X, y, None, None)
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.fit(X, y.where(y.index != y.index[0]), groups, coords)


def test_two_stage_fit_keeps_error_variance_with_shared_coordinates(
    fitted: GPBoostRegressor,
) -> None:
    # 15 % of the fixture rows share coordinates; gpboost's default start collapsed the error
    # variance to ~5e-7 on such data. True noise variance is 0.08**2 = 0.0064.
    assert fitted.cov_ is not None
    assert fitted.cov_.names == ("Group_1", "Group_2", "Group_3", "GP_var", "GP_range")
    residual = fitted.variance_components().set_index("component").loc["residual", "variance"]
    assert residual > 1e-3


def test_cov_pars_reuse_skips_stage_one_and_rejects_mismatch(
    fitted: GPBoostRegressor, rent_data: RentData
) -> None:
    X, y, groups, coords = rent_data
    reused = GPBoostRegressor(FAST_CONFIG).fit(X, y, groups, coords, cov_pars=fitted.cov_)
    assert reused.cov_ is fitted.cov_
    np.testing.assert_allclose(
        reused.variance_components()["variance"], fitted.variance_components()["variance"]
    )
    wrong = replace(FAST_CONFIG, group_cols=("re_canton", "re_district"))
    with pytest.raises(ValueError, match="do not match"):
        GPBoostRegressor(wrong).fit(X, y, groups, coords, cov_pars=fitted.cov_)


def test_gp_early_stopping_with_more_validation_rows_than_levels(rent_data: RentData) -> None:
    # gpboost's own valid_sets predicts in one call and aborts here (C++ assertion).
    X, y, groups, coords = rent_data
    train, valid = X.index[:120], X.index[120:]
    cfg = replace(FAST_CONFIG, num_boost_round=40, eval_every=5)
    model = GPBoostRegressor(cfg).fit(
        X.loc[train],
        y.loc[train],
        groups.loc[train],
        coords.loc[train],
        eval_set=(X.loc[valid], y.loc[valid], groups.loc[valid], coords.loc[valid]),
        early_stopping_rounds=10,
    )
    assert len(valid) > sum(model.n_levels_.values())
    assert model.best_iteration_ in model.eval_iterations_
    assert len(model.evals_result_["valid"]["l1"]) == len(model.eval_iterations_)
    best = model.evals_result_["valid"]["l1"][model.eval_iterations_.index(model.best_iteration_)]
    assert best == min(model.evals_result_["valid"]["l1"])
    np.testing.assert_allclose(
        model.predict(X.loc[valid], groups.loc[valid], coords.loc[valid]),
        model.predict(
            X.loc[valid],
            groups.loc[valid],
            coords.loc[valid],
            num_iteration=model.best_iteration_,
        ),
    )


def test_predict_num_iteration_matches_shorter_model(rent_data: RentData) -> None:
    X, y, groups, coords = rent_data
    cfg = replace(FAST_CONFIG, num_boost_round=12, num_threads=1)
    long_model = GPBoostRegressor(cfg).fit(X, y, groups, coords)
    short = replace(cfg, num_boost_round=6)
    short_model = GPBoostRegressor(short).fit(X, y, groups, coords, cov_pars=long_model.cov_)
    np.testing.assert_allclose(
        long_model.predict(X, groups, coords, num_iteration=6),
        short_model.predict(X, groups, coords),
        rtol=1e-6,
    )


def test_predict_fixed_effects_separates_location_effect(
    fitted: GPBoostRegressor, rent_data: RentData
) -> None:
    X, _, groups, coords = rent_data
    fixed = fitted.predict_fixed_effects(X)
    total = fitted.predict(X, groups, coords)
    assert fixed.shape == total.shape == (len(X),)
    location = pd.Series(total - fixed, index=X.index)
    # Canton K2 has a +0.25 log effect vs K0 in the simulation; the random effects carry it.
    by_canton = location.groupby(groups["re_canton"]).mean()
    assert by_canton["K2"] > by_canton["K0"] + 0.2
    with pytest.raises(KeyError, match="lacks training features"):
        fitted.predict_fixed_effects(X.drop(columns="area"))
