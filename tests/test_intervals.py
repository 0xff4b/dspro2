import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer

from rentml import data
from rentml.intervals import (
    CQR,
    MonotoneQuantileLGBM,
    RentModelBundle,
    _Pinball,
    assign_band,
    conformal_quantile,
    constant_band,
    constant_band_for_coverage,
    coverage_report,
    crossing_rate,
    fair_rent_check,
    fit_band_cutpoints,
    interval_score,
    market_percentiles,
    predict_frame,
    quantile_calibration,
    quantile_columns,
    what_if,
)

FEATURES = ["area", "rooms", "area_per_room", "noise"]
LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
FAST = {"n_estimators": 80, "learning_rate": 0.1, "min_child_samples": 20}


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(area_per_room=df["area"] / df["rooms"])


def _flats(n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        {
            "area": rng.uniform(30, 150, n),
            "rooms": rng.integers(1, 6, n).astype(float),
            "noise": rng.normal(size=n),
            "lang_region": rng.choice(["de", "fr", "it"], n, p=[0.6, 0.3, 0.1]),
        }
    )
    raw = _engineer(raw)
    # heteroscedastic: the spread grows with the area
    y_log = 6.0 + 0.01 * raw["area"] + rng.normal(0, 0.05 + raw["area"] / 1000, n)
    return raw, y_log.to_numpy()


@pytest.fixture(scope="module")
def bundle() -> RentModelBundle:
    raw, y = _flats(2000, 0)
    raw_cal, y_cal = _flats(1500, 1)
    encoder = ColumnTransformer(
        [("keep", "passthrough", FEATURES)], verbose_feature_names_out=False
    )
    encoder.set_output(transform="pandas").fit(raw)
    X = encoder.transform(raw)
    monotone = [1, 0, 1, 0]
    qmodel = MonotoneQuantileLGBM(LEVELS, FAST, monotone, patience=10).fit(X, y)
    point = qmodel.anchor_  # any log-rent regressor will do for the contract
    q_cal = qmodel.predict(encoder.transform(raw_cal))
    q_lo, q50, q_hi = quantile_columns(q_cal, qmodel.levels_, (0.1, 0.5, 0.9)).T
    cuts = fit_band_cutpoints(q50)
    groups = pd.DataFrame({"lang_region": raw_cal["lang_region"], "band": assign_band(q50, cuts)})
    calibrator = CQR(min_group_size=100).fit(q_lo, q_hi, y_cal, groups)
    return RentModelBundle(encoder, point, qmodel, calibrator, cuts)


def test_pinball_objective_gradient_and_tie() -> None:
    grad, hess = _Pinball(0.9)(np.array([2.0, 0.0, 1.0]), np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(grad, [-0.9, 0.1, 0.0])
    np.testing.assert_allclose(hess, 1.0)


def test_quantile_model_is_sorted_monotone_and_calibrated(bundle: RentModelBundle) -> None:
    qmodel = bundle.quantile_model
    raw, y = _flats(2000, 2)
    X = bundle.encoder.transform(raw)
    q = qmodel.predict(X)
    assert q.shape == (2000, len(LEVELS)) and crossing_rate(q) == 0.0
    assert set(qmodel.n_trees_) == set(LEVELS) and all(n >= 1 for n in qmodel.n_trees_.values())
    table = quantile_calibration(y, q, qmodel.levels_)
    np.testing.assert_allclose(table["empirical"], LEVELS, atol=0.06)
    grid = _engineer(raw.iloc[[0] * 25].assign(area=np.linspace(30, 150, 25)))
    for pred in (qmodel.predict_raw(grid[FEATURES[::-1]]), qmodel.predict(grid)):
        assert (np.diff(pred, axis=0) >= -1e-12).all()


def test_crossing_rate_and_quantile_columns() -> None:
    q = np.array([[1.0, 2.0, 3.0], [1.0, 0.5, 3.0]])
    assert crossing_rate(q) == 0.5
    np.testing.assert_array_equal(quantile_columns(q, (0.1, 0.5, 0.9), (0.9, 0.1)), q[:, [2, 0]])
    with pytest.raises(ValueError):
        quantile_columns(q, (0.1, 0.5, 0.9), (0.25,))


def test_market_percentiles_interpolate_and_clip() -> None:
    q = np.array([[1.0, 2.0, 3.0]] * 4)
    pct = market_percentiles(q, (0.25, 0.5, 0.75), np.array([2.5, 2.0, 0.0, 9.0]))
    np.testing.assert_allclose(pct, [0.625, 0.5, 0.01, 0.99])
    np.testing.assert_allclose(market_percentiles(q[:1], (0.25, 0.5, 0.75), [0.5]), [0.125])


def test_conformal_quantile_is_exact_order_statistic() -> None:
    scores = np.arange(1.0, 100.0)  # n = 99: rank ceil(100 * 0.8) = 80 despite float noise
    assert conformal_quantile(scores, 0.2) == 80.0
    assert conformal_quantile(scores[:3], 0.1) == np.inf


def test_bands_put_cutpoint_values_in_upper_band() -> None:
    cuts = fit_band_cutpoints(np.arange(1.0, 9.0), n_bands=4)
    np.testing.assert_allclose(cuts, [2.75, 4.5, 6.25])
    assert assign_band(np.array([1.0, 4.5, 99.0]), cuts).tolist() == ["B1", "B3", "B4"]


def test_cqr_controls_both_tails_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)
    y_cal, y_test = rng.normal(size=4000), rng.normal(size=4000)
    cqr = CQR(alpha=0.2).fit(np.full(4000, -0.5), np.full(4000, 0.5), y_cal)
    assert cqr.levels == (0.1, 0.9)
    lo, hi, cell = cqr.predict(np.full(4000, -0.5), np.full(4000, 0.5))
    report = coverage_report(y_test, lo, hi).loc["all"]
    assert abs(report["below"] - 0.1) < 0.02 and abs(report["above"] - 0.1) < 0.02
    assert set(cell) == {"all"}


def test_cqr_negative_corrections_never_invert_interval() -> None:
    y = np.zeros(200)
    cqr = CQR(alpha=0.2).fit(np.full(200, -1.0), np.full(200, 1.0), y)
    lo, hi, _ = cqr.predict(np.array([-0.1]), np.array([0.1]))
    assert lo[0] <= hi[0]


def test_mondrian_falls_back_from_small_cells() -> None:
    groups = pd.DataFrame(
        {
            "lang_region": ["de"] * 150 + ["fr"] * 120 + ["it"] * 30,
            "band": ["B1"] * 100 + ["B2"] * 50 + ["B1"] * 120 + ["B1"] * 30,
        }
    )
    rng = np.random.default_rng(1)
    y = rng.normal(size=300) * np.where(groups["lang_region"].eq("fr"), 3.0, 1.0)
    cqr = CQR(alpha=0.2, min_group_size=100).fit(np.full(300, -1.0), np.full(300, 1.0), y, groups)
    _, hi, cell = cqr.predict(np.full(4, -1.0), np.full(4, 1.0), groups.iloc[[0, 100, 150, 299]])
    assert cell.tolist() == ["de × B1", "de", "fr × B1", "all"]
    assert hi[2] > hi[0]  # the wider French cell gets the larger correction
    table = cqr.cells_
    assert not table.loc["de × B2", "calibrated"] and table.loc["all", "level"] == "global"


def test_constant_bands() -> None:
    y, pred = np.arange(10.0), np.zeros(10)
    assert constant_band_for_coverage(y, pred, 0.8) == 7.0
    assert constant_band_for_coverage(y, pred, 0.3) == 2.0  # 10 * 0.3 = 3.0000000000000004
    assert constant_band(y, pred, alpha=0.2) == 8.0  # rank ceil(11 * 0.8) = 9


def test_coverage_report_with_groups_and_exact_ci() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    report = coverage_report(y, np.full(4, 1.5), np.full(4, 3.5), pd.Series(list("aabb"), name="g"))
    assert report.index.tolist() == ["a", "b", "all"] and report.index.name == "g"
    assert report.loc["all", "coverage"] == 0.5 and report.loc["a", "below"] == 0.5
    ci = binomtest(2, 4).proportion_ci(method="exact")
    assert report.loc["all", "cp_low"] == pytest.approx(ci.low)
    assert report.loc["all", "mean_width"] == 2.0


def test_interval_score_penalises_misses() -> None:
    score = interval_score(np.array([0.0, 5.0]), np.array([1.0, 1.0]), np.array([3.0, 3.0]), 0.2)
    assert score == pytest.approx(2.0 + 10.0 * (1.0 + 2.0) / 2)


def test_predict_frame_and_bundle_round_trip(bundle: RentModelBundle, tmp_path) -> None:
    raw, _ = _flats(50, 3)
    pred = predict_frame(bundle, raw)
    assert pred.columns.tolist()[:3] == ["expected", "lo", "hi"]
    assert {"q10", "q90", "band", "cell"} <= set(pred.columns)
    assert (pred["lo"] < pred["hi"]).all() and pred.index.equals(raw.index)
    loaded = RentModelBundle.load(bundle.save(tmp_path / "bundle.joblib"))
    pd.testing.assert_frame_equal(predict_frame(loaded, raw), pred)


def test_fair_rent_check_verdicts(bundle: RentModelBundle) -> None:
    raw_row = _flats(1, 4)[0]
    pred = predict_frame(bundle, raw_row).iloc[0]
    checks = [
        fair_rent_check(bundle, raw_row, rent)
        for rent in (0.5 * pred["lo"], pred["q50"], 2 * pred["hi"])
    ]
    assert [c.verdict for c in checks] == ["below", "within", "above"]
    assert checks[1].percentile == pytest.approx(0.5)
    assert checks[0].percentile < 0.1 and checks[1].coverage == pytest.approx(0.8)


def test_what_if_recomputes_features_and_is_monotone(bundle, monkeypatch) -> None:
    monkeypatch.setattr(data, "add_engineered_features", _engineer, raising=False)
    raw_row = _flats(1, 5)[0].assign(area=60.0, rooms=3.0)
    table = what_if(bundle, raw_row, [70.0, 80.0, 90.0], rooms_grid=[4.0])
    assert table["feature"].tolist() == ["baseline", "area", "area", "area", "rooms"]
    assert table.loc[0, "delta"] == 0.0
    assert (np.diff(table.loc[:3, "expected"]) >= 0).all()
    expected_rooms = predict_frame(bundle, _engineer(raw_row.assign(rooms=4.0)))["expected"]
    assert table.loc[4, "expected"] == pytest.approx(expected_rooms.iloc[0])
