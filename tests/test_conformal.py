"""Tests for rentml.conformal."""

import logging
import math

import numpy as np
import pandas as pd
import pytest

from rentml.config import INTERVAL_COVERAGE, RANDOM_STATE
from rentml.conformal import (
    CQRCalibrator,
    MondrianCQR,
    NormalizedConformal,
    assign_band,
    conformal_quantile,
    constant_band_for_coverage,
    coverage_report,
    cqr_scores,
    fit_band_cutpoints,
    interval_score,
)

ALPHA = 1.0 - INTERVAL_COVERAGE
Z90 = 1.2815515655446004  # standard normal 0.9 quantile


def _region_data(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Log prices whose noise differs by language region; raw quantiles assume one scale."""
    region = rng.choice(["de", "fr", "it"], size=n, p=[0.6, 0.3, 0.1])
    scale = np.select([region == "de", region == "fr"], [0.10, 0.25], 0.40)
    mu = rng.normal(7.6, 0.3, n)
    y = mu + scale * rng.standard_normal(n)
    q_lo, q_hi = mu - Z90 * 0.2, mu + Z90 * 0.2
    return np.column_stack([q_lo, q_hi]), y, pd.DataFrame({"lang_region": region, "mu": mu})


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    scores = np.arange(1.0, 10.0)  # n = 9 -> k = ceil(10 * 0.8) = 8
    assert conformal_quantile(scores, 0.2) == 8.0
    assert conformal_quantile(scores[::-1], 0.2) == 8.0


def test_conformal_quantile_is_robust_to_float_noise_in_alpha() -> None:
    # (99 + 1) * (1 - 0.19999999999999996) = 80.00000000000001 must still give rank 80
    assert conformal_quantile(np.arange(1.0, 100.0), ALPHA) == 80.0


def test_conformal_quantile_returns_inf_for_too_few_scores() -> None:
    assert conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.2) == math.inf
    assert conformal_quantile(np.array([]), 0.2) == math.inf


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_conformal_quantile_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        conformal_quantile(np.ones(5), alpha)


def test_conformal_quantile_rejects_nan_scores() -> None:
    with pytest.raises(ValueError, match="NaN"):
        conformal_quantile(np.array([1.0, np.nan]), 0.2)


def test_cqr_scores_sign_convention() -> None:
    scores = cqr_scores([1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.5, 0.5, 3.0])
    np.testing.assert_allclose(scores, [-0.5, 0.5, 1.0])
    with pytest.raises(ValueError, match="Length"):
        cqr_scores([1.0], [2.0, 3.0], [1.0])


def test_cqr_reaches_nominal_marginal_coverage() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    q_cal, y_cal, _ = _region_data(4000, rng)
    q_test, y_test, _ = _region_data(20_000, rng)
    cal = CQRCalibrator().fit(q_cal[:, 0], q_cal[:, 1], y_cal)
    lo, hi = cal.predict(q_test[:, 0], q_test[:, 1])
    coverage = np.mean((y_test >= lo) & (y_test <= hi))
    assert cal.n_calib_ == 4000
    assert coverage == pytest.approx(INTERVAL_COVERAGE, abs=0.015)


def test_cqr_negative_qhat_never_inverts_interval() -> None:
    q_lo, q_hi = np.zeros(200), np.full(200, 10.0)
    y = np.full(200, 5.0)  # every score is -5 -> qhat = -5, interval collapses to the midpoint
    lo, hi = CQRCalibrator().fit(q_lo, q_hi, y).predict(np.array([0.0, 0.0]), np.array([10.0, 4.0]))
    assert np.all(lo <= hi)
    np.testing.assert_allclose([lo[0], hi[0]], [5.0, 5.0])


def test_cqr_predict_before_fit_raises() -> None:
    with pytest.raises(AttributeError):
        CQRCalibrator().predict([1.0], [2.0])


def test_normalized_conformal_coverage_with_heteroscedastic_sigma() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n_cal, n_test = 3000, 20_000
    sigma = rng.uniform(0.05, 0.5, n_cal + n_test)
    mu = rng.normal(7.5, 0.3, n_cal + n_test)
    y = mu + sigma * rng.standard_normal(n_cal + n_test)
    cal = NormalizedConformal().fit(mu[:n_cal], sigma[:n_cal], y[:n_cal])
    lo, hi = cal.predict(mu[n_cal:], sigma[n_cal:])
    coverage = np.mean((y[n_cal:] >= lo) & (y[n_cal:] <= hi))
    assert coverage == pytest.approx(INTERVAL_COVERAGE, abs=0.015)
    assert cal.qhat_ == pytest.approx(Z90, abs=0.08)


def test_normalized_conformal_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma"):
        NormalizedConformal().fit([1.0, 2.0], [0.1, 0.0], [1.0, 2.0])


def test_band_cutpoints_and_labels_from_predictions() -> None:
    preds = np.arange(1.0, 101.0)
    cuts = fit_band_cutpoints(preds, n_bands=4)
    assert cuts.shape == (3,)
    labels = assign_band(preds, cuts)
    counts = pd.Series(labels).value_counts().sort_index()
    assert list(counts.index) == ["B1", "B2", "B3", "B4"]
    assert counts.min() >= 24
    assert list(assign_band([-5.0, 1e6], cuts)) == ["B1", "B4"]


def test_band_functions_reject_invalid_input() -> None:
    with pytest.raises(ValueError):
        fit_band_cutpoints(np.array([]))
    with pytest.raises(ValueError):
        fit_band_cutpoints(np.ones(5), n_bands=0)
    with pytest.raises(ValueError):
        assign_band([np.nan], [1.0])
    assert list(assign_band([3.0], fit_band_cutpoints([1.0, 2.0], n_bands=1))) == ["B1"]


def test_mondrian_reaches_coverage_in_every_region() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    q_cal, y_cal, g_cal = _region_data(6000, rng)
    q_test, y_test, g_test = _region_data(30_000, rng)
    cuts = fit_band_cutpoints(g_cal["mu"])
    g_cal["band"], g_test["band"] = assign_band(g_cal["mu"], cuts), assign_band(g_test["mu"], cuts)

    mondrian = MondrianCQR(ALPHA, min_group_size=100).fit(q_cal[:, 0], q_cal[:, 1], y_cal, g_cal)
    lo, hi, cell = mondrian.predict(q_test[:, 0], q_test[:, 1], g_test)
    report = coverage_report(np.exp(y_test), np.exp(lo), np.exp(hi), g_test["lang_region"])
    assert np.all(np.abs(report["coverage"] - INTERVAL_COVERAGE) < 0.03), report
    assert cell.str.contains("band=").all()

    lo_m, hi_m = (
        CQRCalibrator().fit(q_cal[:, 0], q_cal[:, 1], y_cal).predict(q_test[:, 0], q_test[:, 1])
    )
    marginal = coverage_report(np.exp(y_test), np.exp(lo_m), np.exp(hi_m), g_test["lang_region"])
    assert marginal.loc["it", "coverage"] < 0.7  # marginal CQR under-covers the noisy region


def test_mondrian_falls_back_to_parent_cells() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    n = 450
    groups = pd.DataFrame(
        {"lang_region": ["de"] * 300 + ["fr"] * 120 + ["it"] * 30, "band": ["B1", "B2"] * 225}
    )
    q_lo, q_hi = np.zeros(n), np.ones(n)
    y = rng.uniform(-0.5, 1.5, n)
    mondrian = MondrianCQR(0.2, min_group_size=100).fit(q_lo, q_hi, y, groups)
    table = mondrian.cell_table()
    calibrated = set(table.loc[table["calibrated"], "cell"])
    assert calibrated == {
        "lang_region=de|band=B1",
        "lang_region=de|band=B2",
        "lang_region=de",
        "lang_region=fr",
        "all",
    }
    new = pd.DataFrame(
        {"lang_region": ["de", "fr", "it", "rm"], "band": ["B1", "B2", "B1", "B3"]},
        index=[10, 11, 12, 12],
    )
    lo, hi, cell = mondrian.predict(np.zeros(4), np.ones(4), new)
    assert list(cell) == ["lang_region=de|band=B1", "lang_region=fr", "all", "all"]
    assert list(cell.index) == [10, 11, 12, 12]
    assert np.isnan(table.loc[~table["calibrated"], "qhat"]).all()
    assert np.all(hi - lo > 1.0 - 1e-12)


def test_mondrian_appends_global_level_and_validates_groups() -> None:
    groups = pd.DataFrame({"lang_region": ["de"] * 20})
    mondrian = MondrianCQR(0.2, min_group_size=100, hierarchy=(("lang_region",),))
    mondrian.fit(np.zeros(20), np.ones(20), np.full(20, 0.5), groups)
    assert mondrian.levels_[-1] == ()
    assert "all" in mondrian.qhat_  # global cell calibrated despite n < min_group_size
    with pytest.raises(ValueError, match="missing columns"):
        MondrianCQR(0.2).fit(np.zeros(20), np.ones(20), np.zeros(20), groups)
    with pytest.raises(ValueError, match="rows"):
        mondrian.predict(np.zeros(3), np.ones(3), groups)


def test_coverage_report_overall_and_grouped() -> None:
    y = np.array([100.0, 200.0, 300.0, 400.0])
    lo = np.array([90.0, 210.0, 250.0, 300.0])
    hi = np.array([110.0, 260.0, 280.0, 500.0])
    overall = coverage_report(y, lo, hi)
    assert overall.loc["all", "n"] == 4
    assert overall.loc["all", "coverage"] == pytest.approx(0.5)
    assert overall.loc["all", "below"] == pytest.approx(0.25)
    assert overall.loc["all", "above"] == pytest.approx(0.25)
    assert overall.loc["all", "mean_width"] == pytest.approx((20 + 50 + 30 + 200) / 4)
    grouped = coverage_report(y, lo, hi, pd.Series(["a", "a", "b", "b"], name="seg"))
    assert list(grouped.index) == ["a", "b", "all"]
    assert grouped.index.name == "seg"
    assert grouped.loc["b", "coverage"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        coverage_report(y, lo, hi, pd.Series(["a"]))


def test_constant_band_for_coverage_hits_target() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    yhat = rng.uniform(1000, 3000, 1000)
    y = yhat + rng.normal(0, 200, 1000)
    h = constant_band_for_coverage(y, yhat, 0.8)
    covered = np.mean(np.abs(y - yhat) <= h)
    assert covered >= 0.8
    assert np.mean(np.abs(y - yhat) <= h - 1e-6) < 0.8
    assert constant_band_for_coverage([1.0, 2.0], [1.0, 1.0], 1.0) == pytest.approx(1.0)


def test_constant_band_for_coverage_rejects_invalid_target() -> None:
    with pytest.raises(ValueError):
        constant_band_for_coverage([1.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        constant_band_for_coverage([], [], 0.8)


def test_interval_score_penalises_misses() -> None:
    assert interval_score([5.0], [4.0], [6.0], 0.2) == pytest.approx(2.0)
    # y below by 1: width 2 + 2 / 0.2 * 1 = 12
    assert interval_score([3.0], [4.0], [6.0], 0.2) == pytest.approx(12.0)
    assert interval_score([3.0, 5.0], [4.0, 4.0], [6.0, 6.0], 0.2) == pytest.approx(7.0)
    with pytest.raises(ValueError):
        interval_score([1.0], [0.0], [2.0], 0.0)


def test_cqr_with_too_few_rows_gives_infinite_interval() -> None:
    cal = CQRCalibrator(0.2).fit([0.0, 0.0], [1.0, 1.0], [0.5, 2.0])
    lo, hi = cal.predict([0.0], [1.0])
    assert cal.qhat_ == math.inf
    assert lo[0] == -math.inf and hi[0] == math.inf


def _shifted_quantiles(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw 10/90 % quantiles shifted up by 0.1 log units: too few rows above, too many below."""
    mu = rng.normal(7.6, 0.3, n)
    y = mu + 0.2 * rng.standard_normal(n)
    return mu - Z90 * 0.2 + 0.1, mu + Z90 * 0.2 + 0.1, y


def test_asymmetric_cqr_controls_below_and_above_separately() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    lo_cal, hi_cal, y_cal = _shifted_quantiles(5000, rng)
    lo_test, hi_test, y_test = _shifted_quantiles(40_000, rng)
    rates = {}
    for symmetric in (True, False):
        cal = CQRCalibrator(ALPHA, symmetric=symmetric).fit(lo_cal, hi_cal, y_cal)
        lo, hi = cal.predict(lo_test, hi_test)
        rates[symmetric] = (np.mean(y_test < lo), np.mean(y_test > hi))
    below, above = rates[False]
    assert below == pytest.approx(ALPHA / 2, abs=0.015) and above == pytest.approx(
        ALPHA / 2, abs=0.015
    )
    assert rates[True][0] > 0.15 and rates[True][1] < 0.05  # symmetric keeps the lopsidedness
    cal = CQRCalibrator(ALPHA, symmetric=False).fit(lo_cal, hi_cal, y_cal)
    assert math.isnan(cal.qhat_) and cal.qhat_lo_ > 0 > cal.qhat_hi_


def test_symmetric_cqr_exposes_equal_side_corrections_and_levels() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    lo, hi, y = _shifted_quantiles(500, rng)
    cal = CQRCalibrator().fit(lo, hi, y, interval_levels=(0.1, 0.9))
    assert cal.qhat_ == cal.qhat_lo_ == cal.qhat_hi_
    assert cal.interval_levels_ == (0.1, 0.9)
    assert CQRCalibrator().fit(lo, hi, y).interval_levels_ is None
    with pytest.raises(ValueError, match="interval_levels"):
        CQRCalibrator().fit(lo, hi, y, interval_levels=(0.9, 0.1))


def test_asymmetric_mondrian_balances_each_region() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    q_cal, y_cal, g_cal = _region_data(6000, rng)
    q_test, y_test, g_test = _region_data(30_000, rng)
    y_cal = y_cal + np.where(g_cal["lang_region"] == "fr", 0.15, 0.0)  # fr rents sit higher
    y_test = y_test + np.where(g_test["lang_region"] == "fr", 0.15, 0.0)
    mondrian = MondrianCQR(ALPHA, hierarchy=(("lang_region",),), symmetric=False)
    mondrian.fit(q_cal[:, 0], q_cal[:, 1], y_cal, g_cal, interval_levels=(0.1, 0.9))
    lo, hi, _ = mondrian.predict(q_test[:, 0], q_test[:, 1], g_test)
    report = coverage_report(y_test, lo, hi, g_test["lang_region"])
    assert np.all(np.abs(report["below"] - ALPHA / 2) < 0.025), report
    assert np.all(np.abs(report["above"] - ALPHA / 2) < 0.025), report
    table = mondrian.cell_table()
    assert {"qhat_lo", "qhat_hi"} <= set(table.columns) and table["qhat"].isna().all()
    fr = table.set_index("cell").loc["lang_region=fr"]
    assert fr["qhat_hi"] > fr["qhat_lo"]  # the upper side needs the larger correction
    assert mondrian.interval_levels_ == (0.1, 0.9)


def test_mondrian_normalises_numeric_and_missing_keys() -> None:
    n = 300
    groups = pd.DataFrame({"canton_id": np.repeat([1, 2, 3], 100)})
    groups.loc[200:, "canton_id"] = pd.NA
    groups["canton_id"] = groups["canton_id"].astype("Int64")
    y = np.r_[np.zeros(100), np.full(100, 0.4), np.zeros(100)]
    mondrian = MondrianCQR(0.2, min_group_size=50, hierarchy=(("canton_id",),)).fit(
        np.zeros(n), np.zeros(n), y, groups
    )
    assert {"canton_id=1", "canton_id=2", "canton_id=<NA>"} <= set(mondrian.qhat_)
    new = pd.DataFrame({"canton_id": [1.0, 2.0, np.nan, 7.0]})  # float after a merge
    _, _, cell = mondrian.predict(np.zeros(4), np.zeros(4), new)
    assert list(cell) == ["canton_id=1", "canton_id=2", "canton_id=<NA>", "all"]
    as_text = pd.DataFrame({"canton_id": ["1", "2", None, "x"]})
    _, _, cell_text = mondrian.predict(np.zeros(4), np.zeros(4), as_text)
    assert list(cell_text) == ["canton_id=1", "canton_id=2", "canton_id=<NA>", "all"]


def test_mondrian_rejects_misplaced_global_level_and_empty_data() -> None:
    groups = pd.DataFrame({"lang_region": ["de"] * 20})
    with pytest.raises(ValueError, match="last"):
        MondrianCQR(0.2, hierarchy=((), ("lang_region",))).fit(
            np.zeros(20), np.ones(20), np.zeros(20), groups
        )
    with pytest.raises(ValueError, match="at least one"):
        MondrianCQR(0.2).fit([], [], [], pd.DataFrame({"lang_region": [], "band": []}))


def test_mondrian_logs_share_per_hierarchy_level(caplog: pytest.LogCaptureFixture) -> None:
    groups = pd.DataFrame({"lang_region": ["de"] * 150 + ["fr"] * 50})
    mondrian = MondrianCQR(0.2, min_group_size=100, hierarchy=(("lang_region",),))
    mondrian.fit(np.zeros(200), np.ones(200), np.full(200, 0.5), groups)
    with caplog.at_level(logging.INFO, logger="rentml.conformal"):
        mondrian.predict(
            np.zeros(4), np.ones(4), pd.DataFrame({"lang_region": ["de"] * 3 + ["fr"]})
        )
    assert "'lang_region': 0.75" in caplog.text and "'global': 0.25" in caplog.text


def test_coverage_report_accepts_arrays_and_rejects_all_label() -> None:
    y = np.array([100.0, 200.0, 300.0, 400.0])
    lo, hi = y - 10, y + 10
    bands = assign_band(np.log(y), fit_band_cutpoints(np.log(y), n_bands=2))
    report = coverage_report(y, lo, hi, bands)
    assert list(report.index) == ["B1", "B2", "all"] and report.index.name == "group"
    assert list(coverage_report(y, lo, hi, ["x", "x", "y", "y"]).index) == ["x", "y", "all"]
    with pytest.raises(ValueError, match="all"):
        coverage_report(y, lo, hi, pd.Series(["all", "x", "x", "x"]))
    no_overall = coverage_report(y, lo, hi, ["all", "x", "x", "x"], include_overall=False)
    assert list(no_overall.index) == ["all", "x"]
