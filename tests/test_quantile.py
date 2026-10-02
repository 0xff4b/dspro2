"""Tests for rentml.quantile."""

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.exceptions import NotFittedError

from rentml import quantile as quantile_mod
from rentml.config import QUANTILE_LEVELS, RANDOM_STATE
from rentml.quantile import (
    MonotoneQuantileLGBM,
    PinballObjective,
    crossing_rate,
    make_pinball_objective,
    market_percentile,
    market_percentiles,
    pinball_loss,
    quantile_calibration_table,
    quantile_columns,
    rearrange,
)

LEVELS = np.asarray(QUANTILE_LEVELS)


def _mu_sigma(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """True conditional mean and noise scale of :func:`_heteroscedastic`."""
    area, x1, x2 = (X[col].to_numpy() for col in ("area", "x1", "x2"))
    mu = 6.5 + 0.9 * np.log(area / 30) / np.log(5) + 0.3 * np.sin(3 * x1) + 0.1 * x2
    return mu, 0.05 + 0.25 * x1


def _heteroscedastic(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Log rent rising in area; noise scale driven by ``x1`` (heteroscedastic)."""
    X = pd.DataFrame(
        {"area": rng.uniform(30, 150, n), "x1": rng.uniform(0, 1, n), "x2": rng.normal(0, 1, n)}
    )
    mu, sigma = _mu_sigma(X)
    return X, mu + sigma * rng.standard_normal(n)


def _oracle_quantiles(X: pd.DataFrame) -> np.ndarray:
    mu, sigma = _mu_sigma(X)
    return mu[:, None] + sigma[:, None] * norm.ppf(LEVELS)[None, :]


def _train_target() -> np.ndarray:
    """Training target of the ``fitted`` fixture (same seed, same draws)."""
    return _heteroscedastic(3000, np.random.default_rng(RANDOM_STATE))[1]


@pytest.fixture(scope="module")
def fitted() -> tuple[MonotoneQuantileLGBM, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _heteroscedastic(3000, rng)
    X_test, y_test = _heteroscedastic(10_000, rng)
    model = MonotoneQuantileLGBM(monotone=[1, 0, 0]).fit(X, y)
    return model, X_test, y_test


def test_pinball_loss_matches_manual_computation() -> None:
    y = np.array([1.0, 2.0, 3.0])
    q = np.array([2.0, 2.0, 2.0])
    # residuals -1, 0, 1 -> 0.25 * 1 (below) + 0 + 0.75 * 1 (above) at alpha=0.75
    assert pinball_loss(y, q, 0.75) == pytest.approx((0.25 + 0.0 + 0.75) / 3)
    assert pinball_loss(y, 2.0, 0.5) == pytest.approx(1.0 / 3)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_pinball_loss_rejects_invalid_level(alpha: float) -> None:
    with pytest.raises(ValueError):
        pinball_loss(np.ones(3), np.ones(3), alpha)


def test_pinball_loss_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        pinball_loss(np.ones(3), np.ones(2), 0.5)


def test_pinball_objective_gradient_and_constant_hessian() -> None:
    objective = make_pinball_objective(0.9, hess_const=2.0)
    grad, hess = objective(np.array([1.0, 0.0, 0.5]), np.array([0.0, 1.0, 0.5]))
    np.testing.assert_allclose(grad, [-0.9, 0.1, 0.0])
    np.testing.assert_allclose(hess, [2.0, 2.0, 2.0])


def test_pinball_objective_is_picklable() -> None:
    objective = make_pinball_objective(0.25)
    restored = pickle.loads(pickle.dumps(objective))
    assert isinstance(restored, PinballObjective)
    assert restored == objective


@pytest.mark.parametrize(("alpha", "hess"), [(1.2, 1.0), (0.5, 0.0), (0.5, -1.0)])
def test_make_pinball_objective_rejects_invalid_arguments(alpha: float, hess: float) -> None:
    with pytest.raises(ValueError):
        make_pinball_objective(alpha, hess_const=hess)


def test_quantile_model_heldout_coverage_within_tolerance(fitted) -> None:
    model, X_test, y_test = fitted
    table = quantile_calibration_table(y_test, model.predict(X_test), model.levels_)
    assert np.all(np.abs(table["deviation"]) <= 0.04), table


def test_quantile_model_is_monotone_in_constrained_feature(fitted) -> None:
    model, X_test, _ = fitted
    base = X_test.iloc[:200].reset_index(drop=True)
    grid = np.linspace(20, 180, 41)
    preds = np.stack([model.predict(base.assign(area=a)) for a in grid])  # (grid, rows, L)
    assert np.all(np.diff(preds, axis=0) >= -1e-12)
    assert np.all(np.diff(preds, axis=2) >= 0)  # rearranged: no crossing


def test_tail_levels_beat_constant_quantiles_and_approach_oracle(fitted) -> None:
    model, X_test, y_test = fitted
    q, oracle, y_train = model.predict(X_test), _oracle_quantiles(X_test), _train_target()
    for j, level in ((0, 0.05), (6, 0.95)):
        loss = pinball_loss(y_test, q[:, j], level)
        constant = pinball_loss(y_test, np.quantile(y_train, level), level)
        assert loss < 0.6 * constant, (level, loss / constant)
        assert loss < 1.10 * pinball_loss(y_test, oracle[:, j], level), level


def test_tails_are_balanced_in_cheapest_and_most_expensive_quintile(fitted) -> None:
    model, X_test, y_test = fitted
    q = model.predict(X_test)
    mu, _ = _mu_sigma(X_test)
    quintile = pd.qcut(mu, 5, labels=False)
    for k in (0, 4):
        rows = quintile == k
        below, above = np.mean(y_test[rows] < q[rows, 1]), np.mean(y_test[rows] > q[rows, 5])
        assert abs(below - 0.10) <= 0.05 and abs(above - 0.10) <= 0.05, (k, below, above)


def test_constant_start_pulls_tails_towards_marginal_quantile() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _heteroscedastic(3000, rng)
    X_test, y_test = _heteroscedastic(5000, rng)
    model = MonotoneQuantileLGBM(levels=(0.1, 0.5, 0.9), monotone=[1, 0, 0], start="constant")
    q = model.fit(X, y).predict(X_test)
    cheapest = _mu_sigma(X_test)[0] < np.quantile(_mu_sigma(X_test)[0], 0.2)
    assert np.mean(y_test[cheapest] < q[cheapest, 0]) > 0.15  # the documented failure mode
    assert not model.shifted_.any() and model.anchor_model_ is None


def test_shifted_start_is_anchor_plus_offset_plus_trees() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _heteroscedastic(800, rng)
    model = MonotoneQuantileLGBM(levels=(0.1, 0.5, 0.9), params={"n_estimators": 20}).fit(X, y)
    anchor = model.anchor_model_.predict(X) + model.anchor_init_
    trees = np.column_stack([m.predict(X) for m in model.models_])
    raw = model.predict_raw(X)
    assert model.models_[1] is model.anchor_model_
    np.testing.assert_allclose(model.shifted_, [True, False, True])
    np.testing.assert_allclose(raw[:, 1], anchor)
    np.testing.assert_allclose(
        raw[:, [0, 2]], anchor[:, None] + model.init_scores_[[0, 2]] + trees[:, [0, 2]]
    )
    assert model.init_scores_[0] < 0 < model.init_scores_[2]  # residual quantiles around the median
    extra = MonotoneQuantileLGBM(levels=(0.1, 0.9), params={"n_estimators": 5}).fit(X, y)
    assert extra.anchor_model_ is not None and extra.shifted_.all()  # 0.5 fitted as hidden anchor


def test_predict_reorders_dataframe_columns_and_validates_input(fitted) -> None:
    model, X_test, _ = fitted
    X = X_test.iloc[:100]
    np.testing.assert_array_equal(model.predict(X[["x2", "area", "x1"]]), model.predict(X))
    np.testing.assert_array_equal(model.predict(X.assign(extra=1.0)), model.predict(X))
    with pytest.warns(UserWarning, match="feature names"):  # positional input is allowed
        np.testing.assert_allclose(model.predict(X.to_numpy()), model.predict(X))
    assert model.feature_names_in_ == ["area", "x1", "x2"]
    with pytest.raises(ValueError, match="x1"):
        model.predict(X[["area", "x2"]])
    with pytest.raises(ValueError, match="columns"):
        model.predict(X.to_numpy()[:, :2])


def test_leaf_and_tree_aliases_do_not_bypass_tail_floor() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _heteroscedastic(2000, rng)
    params = {"min_data_in_leaf": 5, "num_iterations": 7}
    model = MonotoneQuantileLGBM(levels=(0.05, 0.5), params=params, anchor_folds=0).fit(X, y)
    tail = model.models_[0]
    lgb_params = tail.get_params()
    assert lgb_params["min_child_samples"] == 200  # ceil(10 / 0.05)
    assert "min_data_in_leaf" not in lgb_params and "num_iterations" not in lgb_params
    assert tail.booster_.num_trees() == 7
    leaves = tail.booster_.trees_to_dataframe().query("left_child.isna()")
    assert leaves["count"].min() >= 200
    assert model.models_[1].get_params()["min_child_samples"] == 20  # alias still overrides default


def test_validation_fraction_selects_trees_by_pinball_and_refits() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _heteroscedastic(1500, rng)
    groups = np.repeat(np.arange(750), 2)  # duplicate pairs must stay on one side
    model = MonotoneQuantileLGBM(
        levels=(0.1, 0.5, 0.9),
        params={"n_estimators": 400},
        early_stopping_rounds=20,
        validation_fraction=0.25,
    ).fit(X, y, groups=groups)
    selected = model.selected_n_estimators_
    assert set(selected) == {0.1, 0.5, 0.9}
    assert all(1 <= trees < 400 for trees in selected.values())
    assert model.best_iterations_ == [selected[0.1], selected[0.5], selected[0.9]]
    assert [m.booster_.num_trees() for m in model.models_] == model.best_iterations_
    with pytest.raises(ValueError, match="either"):
        MonotoneQuantileLGBM(validation_fraction=0.2).fit(X, y, eval_set=(X, y))
    with pytest.raises(ValueError, match="groups"):
        MonotoneQuantileLGBM(validation_fraction=0.2).fit(X, y, groups=groups[:10])
    with pytest.raises(ValueError, match="validation_fraction"):
        MonotoneQuantileLGBM(validation_fraction=1.5).fit(X, y)


@pytest.mark.filterwarnings("ignore:The argument 'eval_set' is deprecated")  # 4.7 on old path
def test_eval_set_falls_back_for_lightgbm_without_eval_x(monkeypatch) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _heteroscedastic(800, rng)
    X_val, y_val = _heteroscedastic(300, rng)

    def fit_score() -> float:
        model = MonotoneQuantileLGBM(levels=(0.9,), params={"n_estimators": 30})
        model.fit(X, y, eval_set=(X_val, y_val))
        return model.models_[0].best_score_["valid_0"]["pinball"]

    with_eval_x = fit_score()
    monkeypatch.setattr(quantile_mod, "_supports_eval_x", lambda: False)  # LightGBM 4.5 / 4.6
    assert fit_score() == pytest.approx(with_eval_x)


def test_predict_adds_init_score_back() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _heteroscedastic(500, rng)
    model = MonotoneQuantileLGBM(
        levels=(0.1, 0.9), params={"n_estimators": 1}, start="constant"
    ).fit(X, y)
    raw_trees = np.column_stack([m.predict(X) for m in model.models_])
    np.testing.assert_allclose(model.predict_raw(X), raw_trees + model.init_scores_)
    np.testing.assert_allclose(model.init_scores_, np.quantile(y, [0.1, 0.9]))
    # a single shrunken tree barely moves away from the empirical quantile
    assert np.max(np.abs(model.predict_raw(X) - model.init_scores_)) < 0.05


def test_eval_set_uses_pinball_metric_with_init_score() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = _heteroscedastic(1500, rng)
    X_val, y_val = _heteroscedastic(500, rng)
    model = MonotoneQuantileLGBM(
        levels=(0.5,), monotone=[1, 0, 0], params={"n_estimators": 400}, early_stopping_rounds=10
    ).fit(X, y, eval_set=(X_val, y_val))
    lgbm = model.models_[0]
    assert 0 < model.best_iterations_[0] < 400
    reported = lgbm.best_score_["valid_0"]["pinball"]
    assert reported == pytest.approx(pinball_loss(y_val, model.predict_raw(X_val)[:, 0], 0.5))


def test_fitted_model_survives_pickle(fitted) -> None:
    model, X_test, _ = fitted
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(X_test.iloc[:50]), model.predict(X_test.iloc[:50]))


def test_predict_before_fit_raises() -> None:
    with pytest.raises(NotFittedError):
        MonotoneQuantileLGBM().predict(np.zeros((2, 3)))


def test_fit_rejects_wrong_monotone_length_and_objective_param() -> None:
    X = np.random.default_rng(0).normal(size=(50, 3))
    y = X[:, 0]
    with pytest.raises(ValueError, match="monotone"):
        MonotoneQuantileLGBM(monotone=[1, 0]).fit(X, y)
    with pytest.raises(ValueError, match="objective"):
        MonotoneQuantileLGBM(params={"objective": "quantile"}).fit(X, y)
    with pytest.raises(ValueError, match="increasing"):
        MonotoneQuantileLGBM(levels=(0.9, 0.1)).fit(X, y)


def test_fit_rejects_non_finite_target() -> None:
    X = np.zeros((10, 2))
    with pytest.raises(ValueError, match="non-finite"):
        MonotoneQuantileLGBM().fit(X, np.r_[np.nan, np.ones(9)])


def test_fit_rejects_unknown_start() -> None:
    X = np.random.default_rng(0).normal(size=(50, 2))
    with pytest.raises(ValueError, match="start"):
        MonotoneQuantileLGBM(start="median").fit(X, X[:, 0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="anchor_folds"):
        MonotoneQuantileLGBM().fit(X, X[:, 0], groups=np.repeat([0, 1, 2], [20, 20, 10]))


def test_rearrange_sorts_rows_and_crossing_rate_counts_rows() -> None:
    q = np.array([[1.0, 3.0, 2.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [0.0, 0.0, 0.0]])
    assert crossing_rate(q) == pytest.approx(0.5)
    sorted_q = rearrange(q)
    assert crossing_rate(sorted_q) == 0.0
    np.testing.assert_allclose(sorted_q[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rearrange(np.array([2.0, 1.0])), [1.0, 2.0])


def test_crossing_rate_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        crossing_rate(np.empty((0, 3)))
    assert crossing_rate(np.ones((4, 1))) == 0.0


def test_quantile_columns_selects_levels_and_rejects_missing() -> None:
    q = np.tile(np.arange(len(LEVELS), dtype=float), (3, 1))
    np.testing.assert_allclose(quantile_columns(q, LEVELS, (0.1, 0.9))[0], [1.0, 5.0])
    with pytest.raises(ValueError, match="0.33"):
        quantile_columns(q, LEVELS, (0.33,))


def test_market_percentile_interpolates_on_grid() -> None:
    q = np.log([1500, 1600, 1800, 2000, 2200, 2400, 2600])
    assert market_percentile(q, LEVELS, np.log(2000)) == pytest.approx(0.5)
    assert market_percentile(q, LEVELS, np.log(2600)) == pytest.approx(0.95)
    mid = (np.log(2000) + np.log(2200)) / 2
    assert market_percentile(q, LEVELS, mid) == pytest.approx(0.625)


def test_market_percentile_tails_are_clamped() -> None:
    q = np.log([1500, 1600, 1800, 2000, 2200, 2400, 2600])
    assert market_percentile(q, LEVELS, np.log(100)) == pytest.approx(0.01)
    assert market_percentile(q, LEVELS, np.log(90_000)) == pytest.approx(0.99)
    slightly_above = market_percentile(q, LEVELS, np.log(2610))
    assert 0.95 < slightly_above < 0.99


def test_market_percentile_handles_ties_and_nan() -> None:
    flat = np.full(len(LEVELS), np.log(2000))
    assert market_percentile(flat, LEVELS, np.log(1999)) == pytest.approx(0.01)
    assert market_percentile(flat, LEVELS, np.log(2001)) == pytest.approx(0.99)
    assert market_percentile(flat, LEVELS, np.log(2000)) == pytest.approx(0.95)
    assert np.isnan(market_percentile(flat, LEVELS, np.nan))


def test_market_percentiles_matches_rowwise_and_validates_shapes() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    q = np.sort(rng.normal(7.5, 0.3, size=(30, len(LEVELS))), axis=1)
    values = rng.normal(7.5, 0.4, size=30)
    vectorised = market_percentiles(q, LEVELS, values)
    rowwise = [market_percentile(q[i], LEVELS, values[i]) for i in range(30)]
    np.testing.assert_allclose(vectorised, rowwise)
    assert np.all((vectorised >= 0.01) & (vectorised <= 0.99))
    with pytest.raises(ValueError):
        market_percentiles(q, LEVELS, values[:5])
    with pytest.raises(ValueError):
        market_percentiles(q[:, :3], LEVELS, values)


def test_quantile_calibration_table_on_true_quantiles() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    y = rng.standard_normal(20_000)
    true_q = np.tile(np.quantile(rng.standard_normal(200_000), LEVELS), (y.size, 1))
    table = quantile_calibration_table(y, true_q, LEVELS)
    assert list(table.columns) == ["level", "empirical", "deviation", "pinball", "n"]
    assert np.all(np.abs(table["deviation"]) < 0.01)
    with pytest.raises(ValueError):
        quantile_calibration_table(y[:10], true_q, LEVELS)
