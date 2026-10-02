"""Tests for rentml.qoli (synthetic data, no network)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rentml import amenities, qoli
from rentml.qoli import (
    Indicator,
    aggregate,
    convergent_validity,
    lsv_exceedance,
    minmax_normalize,
    normalization_bounds,
    value_score,
    weight_sensitivity,
)

INDICATORS = [
    Indicator("oev", "transport", True, transform="log1p"),
    Indicator("noise_day", "noise", False),
    Indicator("acc_total", "amenities", True),
    Indicator("green", "amenities", True),
]


@pytest.fixture
def indicator_table() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "oev": rng.lognormal(8.0, 1.0, n),
            "noise_day": rng.normal(55.0, 6.0, n),
            "acc_total": rng.uniform(0.0, 100.0, n),
            "green": rng.uniform(0.0, 1.0, n),
        },
        index=pd.Index(np.arange(1000, 1000 + n), name="listing_id"),
    )


# --- Indicator / minmax_normalize ------------------------------------------------------------


def test_indicator_rejects_unknown_transform() -> None:
    with pytest.raises(ValueError, match="transform"):
        Indicator("x", "d", True, transform="sqrt")


def test_minmax_normalize_scales_to_0_100_and_orients(indicator_table: pd.DataFrame) -> None:
    norm = minmax_normalize(indicator_table, INDICATORS)
    assert list(norm.columns) == [ind.name for ind in INDICATORS]
    assert norm.index.equals(indicator_table.index)
    assert norm.min().min() == pytest.approx(0.0)
    assert norm.max().max() == pytest.approx(100.0)
    # Lower noise is better: the quietest listing gets the top score.
    assert norm.loc[indicator_table["noise_day"].idxmin(), "noise_day"] == pytest.approx(100.0)
    assert norm.loc[indicator_table["noise_day"].idxmax(), "noise_day"] == pytest.approx(0.0)
    bounds = normalization_bounds(indicator_table, INDICATORS)
    assert norm.attrs["bounds"] == {name: (lo, hi) for name, lo, hi in bounds.itertuples()}


def test_minmax_normalize_winsorizing_resists_outlier() -> None:
    df = pd.DataFrame({"x": np.append(np.arange(100.0), 1e6)})
    ind = [Indicator("x", "d", True)]
    winsorized = minmax_normalize(df, ind)["x"]
    raw = minmax_normalize(df, ind, winsor=None)["x"]
    assert winsorized.iloc[-1] == pytest.approx(100.0)
    assert winsorized.iloc[50] == pytest.approx(50.0, abs=2.0)
    assert raw.iloc[50] < 0.1  # without winsorising the outlier compresses everything else


def test_minmax_normalize_log1p_transform_spreads_skewed_values() -> None:
    df = pd.DataFrame({"x": [0.0, 9.0, 99.0, 999.0]})
    norm = minmax_normalize(df, [Indicator("x", "d", True, transform="log1p")], winsor=None)
    np.testing.assert_allclose(norm["x"], [0.0, 100 / 3, 200 / 3, 100.0])


def test_minmax_normalize_constant_column_scores_50_and_keeps_nan() -> None:
    df = pd.DataFrame({"c": [3.0, 3.0, np.nan]})
    norm = minmax_normalize(df, [Indicator("c", "d", False)])
    np.testing.assert_allclose(norm["c"], [50.0, 50.0, np.nan])


def test_minmax_normalize_reuses_bounds_for_new_locations() -> None:
    ref = pd.DataFrame({"x": np.linspace(0.0, 10.0, 101)})
    ind = [Indicator("x", "d", True)]
    bounds = minmax_normalize(ref, ind, winsor=None).attrs["bounds"]
    new = minmax_normalize(pd.DataFrame({"x": [-5.0, 5.0, 50.0]}), ind, bounds=bounds)
    np.testing.assert_allclose(new["x"], [0.0, 50.0, 100.0])


def test_minmax_normalize_accepts_bounds_mapping() -> None:
    new = minmax_normalize(
        pd.DataFrame({"x": [0.0, 99.0]}),
        [Indicator("x", "d", True, transform="log1p")],
        bounds={"x": (0.0, np.log1p(99.0))},
    )
    np.testing.assert_allclose(new["x"], [0.0, 100.0])
    assert new.attrs["bounds"] == {"x": (0.0, float(np.log1p(99.0)))}


def test_normalization_bounds_on_transformed_scale(indicator_table: pd.DataFrame) -> None:
    bounds = normalization_bounds(indicator_table, INDICATORS)
    assert list(bounds.columns) == ["lo", "hi"]
    assert list(bounds.index) == [i.name for i in INDICATORS]
    log_oev = np.log1p(indicator_table["oev"])
    assert bounds.loc["oev", "lo"] == pytest.approx(log_oev.quantile(0.01))
    assert bounds.loc["oev", "hi"] == pytest.approx(log_oev.quantile(0.99))
    reused = minmax_normalize(indicator_table, INDICATORS, bounds=bounds)
    pd.testing.assert_frame_equal(reused, minmax_normalize(indicator_table, INDICATORS))
    with pytest.raises(ValueError, match="winsor"):
        normalization_bounds(indicator_table, INDICATORS, winsor=(0.5, 0.5))


def test_normalization_bounds_zero_inflated_falls_back_to_raw_range() -> None:
    df = pd.DataFrame(
        {"x": np.r_[np.zeros(995), np.arange(1.0, 6.0)], "flag": np.r_[np.zeros(996), np.ones(4)]}
    )
    inds = [Indicator("x", "d", True), Indicator("flag", "d", False)]
    bounds = normalization_bounds(df, inds)  # the 1 %/99 % quantiles are both 0
    assert bounds.loc["x"].tolist() == [0.0, 5.0] and bounds.loc["flag"].tolist() == [0.0, 1.0]
    norm = minmax_normalize(df, inds)
    assert (norm["x"].iloc[:995] == 0.0).all()
    np.testing.assert_allclose(norm["x"].iloc[995:], [20.0, 40.0, 60.0, 80.0, 100.0])
    assert norm["flag"].iloc[0] == 100.0 and norm["flag"].iloc[-1] == 0.0  # lower is better


def test_normalised_frames_can_be_combined(indicator_table: pd.DataFrame) -> None:
    # Listing-level and municipality-level indicators are normalised separately, then combined.
    listing = minmax_normalize(indicator_table, INDICATORS[:2])
    muni = minmax_normalize(indicator_table, INDICATORS[2:], winsor=None)
    combined = pd.concat([listing, muni], axis=1)
    pd.testing.assert_frame_equal(listing.join(muni), combined)
    merged = listing.merge(muni.copy(), left_index=True, right_index=True)
    pd.testing.assert_frame_equal(merged, combined)
    score, _ = aggregate(combined, INDICATORS)
    assert score.notna().all()
    stacked = pd.concat([listing, listing.copy()])  # equal attrs on different objects
    assert len(stacked) == 2 * len(listing)


def test_minmax_normalize_errors(indicator_table: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        minmax_normalize(indicator_table, [Indicator("missing", "d", True)])
    with pytest.raises(ValueError, match="winsor"):
        minmax_normalize(indicator_table, INDICATORS, winsor=(0.9, 0.1))
    with pytest.raises(ValueError, match="unique"):
        minmax_normalize(indicator_table, [INDICATORS[0], INDICATORS[0]])
    with pytest.raises(ValueError, match="log1p"):
        minmax_normalize(pd.DataFrame({"oev": [-1.0, 2.0]}), [INDICATORS[0]])
    with pytest.raises(ValueError, match="finite"):
        minmax_normalize(pd.DataFrame({"x": [np.nan, np.nan]}), [Indicator("x", "d", True)])
    bounds = pd.DataFrame({"lo": [0.0], "hi": [1.0]}, index=["other"])
    with pytest.raises(KeyError, match="bounds"):
        minmax_normalize(indicator_table, [INDICATORS[2]], bounds=bounds)


# --- aggregate ---------------------------------------------------------------------------------

AGG_INDICATORS = [Indicator("a1", "A", True), Indicator("a2", "A", True), Indicator("b", "B", True)]


def test_aggregate_equal_weights_within_and_user_weights_across() -> None:
    norm = pd.DataFrame({"a1": [0.0, 100.0], "a2": [100.0, 100.0], "b": [50.0, 0.0]})
    score, dims = aggregate(norm, AGG_INDICATORS)
    np.testing.assert_allclose(dims["A"], [50.0, 100.0])
    np.testing.assert_allclose(dims["B"], [50.0, 0.0])
    np.testing.assert_allclose(score, [50.0, 50.0])
    weighted, _ = aggregate(norm, AGG_INDICATORS, {"A": 3.0, "B": 1.0})
    rescaled, _ = aggregate(norm, AGG_INDICATORS, {"A": 6.0, "B": 2.0})
    np.testing.assert_allclose(weighted, [50.0, 75.0])
    pd.testing.assert_series_equal(weighted, rescaled)
    assert weighted.name == "qoli"


def test_aggregate_handles_missing_values() -> None:
    norm = pd.DataFrame(
        {"a1": [np.nan, 20.0, 20.0], "a2": [80.0, 40.0, 40.0], "b": [20.0, np.nan, np.nan]}
    )
    score, dims = aggregate(norm, AGG_INDICATORS, {"A": 1.0, "B": 1.0})
    assert dims.loc[0, "A"] == pytest.approx(80.0)  # missing indicator skipped within dimension
    assert score.iloc[0] == pytest.approx(50.0)
    assert np.isnan(score.iloc[1])  # a weighted dimension is missing
    only_a, _ = aggregate(norm, AGG_INDICATORS, {"A": 1.0, "B": 0.0})
    assert only_a.iloc[1] == pytest.approx(30.0)  # zero-weight dimension may be missing


def test_aggregate_rejects_invalid_weights_and_columns() -> None:
    norm = pd.DataFrame({"a1": [1.0], "a2": [1.0], "b": [1.0]})
    for bad in ({"A": 1.0}, {"A": 1.0, "B": -1.0}, {"A": 0.0, "B": 0.0}, {"A": 1, "B": 1, "C": 1}):
        with pytest.raises(ValueError):
            aggregate(norm, AGG_INDICATORS, bad)
    with pytest.raises(KeyError):
        aggregate(norm.drop(columns="b"), AGG_INDICATORS)


# --- noise -------------------------------------------------------------------------------------


def test_lsv_exceedance_boundaries_and_missing_values() -> None:
    idx = pd.Index([11, 12, 13, 14], name="listing_id")
    day = pd.Series([60.0, 60.5, 70.0, np.nan], index=idx)
    night = np.array([50.0, 49.0, 55.5, 40.0])
    out = lsv_exceedance(day, night)
    assert out.index.equals(idx)
    np.testing.assert_allclose(out["day_excess_db"], [0.0, 0.5, 10.0, np.nan])
    np.testing.assert_allclose(out["night_excess_db"], [0.0, 0.0, 5.5, 0.0])
    assert out["exceeds_day"].tolist()[:3] == [False, True, True]
    assert out["exceeds_day"].isna().tolist() == [False, False, False, True]
    assert out["exceeds_night"].tolist() == [False, False, True, False]
    assert out["exceeds_any"].tolist()[:3] == [False, True, True]
    assert pd.isna(out["exceeds_any"].iloc[3])  # unknown day level, compliant night


def test_lsv_exceedance_custom_thresholds_and_errors() -> None:
    out = lsv_exceedance(np.array([62.0]), np.array([52.0]), {"day": 65.0, "night": 55.0})
    assert not out["exceeds_any"].iloc[0]
    with pytest.raises(ValueError):
        lsv_exceedance(np.array([60.0, 61.0]), np.array([50.0]))
    with pytest.raises(KeyError):
        lsv_exceedance(np.array([60.0]), np.array([50.0]), {"day": 60.0})


def test_lsv_exceedance_aligns_series_by_label() -> None:
    day = pd.Series([70.0, 40.0], index=pd.Index([1, 2], name="listing_id"))
    night = pd.Series([40.0, 70.0], index=pd.Index([2, 1], name="listing_id"))  # other order
    out = lsv_exceedance(day, night)
    assert out.index.equals(day.index)
    np.testing.assert_allclose(out["night_excess_db"], [20.0, 0.0])
    assert out["exceeds_night"].tolist() == [True, False]
    with pytest.raises(ValueError, match="labels"):
        lsv_exceedance(day, pd.Series([40.0, 70.0], index=[2, 3]))


# --- sensitivity / validity / value score ------------------------------------------------------


def _dim_scores(n: int = 300, k: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(0, 100, (n, k)), columns=[f"d{i}" for i in range(k)])


def test_weight_sensitivity_identical_dimensions_are_stable() -> None:
    base = np.random.default_rng(1).uniform(0, 100, 100)
    dims = pd.DataFrame({"a": base, "b": base, "c": base})
    out = weight_sensitivity(dims, n_draws=50)
    np.testing.assert_allclose(out["spearman"], 1.0)
    np.testing.assert_allclose(out["mean_abs_rank_shift_pct"], 0.0, atol=1e-9)
    np.testing.assert_allclose(out["top_decile_retention"], 1.0)


def test_weight_sensitivity_statistics_and_determinism() -> None:
    dims = _dim_scores()
    out = weight_sensitivity(dims, n_draws=200, seed=7)
    cols = ["w_d0", "w_d1", "w_d2", "spearman", "mean_abs_rank_shift_pct"]
    assert cols + ["p90_abs_rank_shift_pct", "top_decile_retention"] == list(out.columns)
    assert len(out) == 200
    np.testing.assert_allclose(out[["w_d0", "w_d1", "w_d2"]].sum(axis=1), 1.0)
    assert 0.5 < out["spearman"].mean() < 1.0
    assert out["mean_abs_rank_shift_pct"].mean() > 0.0
    assert out["top_decile_retention"].between(0.0, 1.0).all()
    pd.testing.assert_frame_equal(out, weight_sensitivity(dims, n_draws=200, seed=7))
    tight = weight_sensitivity(dims, n_draws=200, concentration=1000.0, seed=7)
    assert tight["spearman"].mean() > out["spearman"].mean()


def test_weight_sensitivity_zero_weight_nan_rows_and_errors() -> None:
    dims = _dim_scores(n=50)
    dims.iloc[0, 0] = np.nan
    out = weight_sensitivity(dims, n_draws=20, base_weights={"d0": 1.0, "d1": 1.0, "d2": 0.0})
    assert (out["w_d2"] == 0.0).all()
    with pytest.raises(ValueError):
        weight_sensitivity(dims[["d0"]])
    with pytest.raises(ValueError):
        weight_sensitivity(dims.iloc[1:3])
    with pytest.raises(ValueError):
        weight_sensitivity(dims, concentration=0.0)


def test_weight_sensitivity_ignores_nan_in_zero_weight_dimension() -> None:
    dims = _dim_scores(n=10)
    dims.loc[2:, "d2"] = np.nan  # optional dimension missing for 8 of 10 units
    out = weight_sensitivity(dims, n_draws=30, base_weights={"d0": 1.0, "d1": 1.0, "d2": 0.0})
    ref = weight_sensitivity(dims[["d0", "d1"]], n_draws=30)
    cols = ["spearman", "mean_abs_rank_shift_pct", "p90_abs_rank_shift_pct", "top_decile_retention"]
    pd.testing.assert_frame_equal(out[cols], ref[cols])


def test_weight_sensitivity_draws_centre_on_base_weights() -> None:
    base = {"d0": 0.6, "d1": 0.3, "d2": 0.1}
    out = weight_sensitivity(_dim_scores(n=50), n_draws=2000, base_weights=base)
    w = out[["w_d0", "w_d1", "w_d2"]]
    np.testing.assert_allclose(w.mean(), [0.6, 0.3, 0.1], atol=0.01)
    assert w["w_d0"].var() == pytest.approx(0.6 * 0.4 / 11.0, rel=0.15)  # concentration 10


def test_convergent_validity_monotone_relation_and_alignment() -> None:
    x = pd.Series(np.arange(50.0), index=[f"m{i}" for i in range(50)])
    external = np.exp(x / 10).sample(frac=1.0, random_state=42)  # shuffled order
    external.loc["m0"] = np.nan
    result = convergent_validity(x, pd.concat([external, pd.Series({"extra": 1.0})]))
    assert result["n"] == 49
    assert result["rho"] == pytest.approx(1.0)
    assert result["p_value"] < 1e-6
    assert result["ci_low"] <= result["rho"] + 1e-12 and result["ci_high"] == pytest.approx(1.0)
    noisy = convergent_validity(x, x + np.random.default_rng(42).normal(0, 15, 50))
    assert -1.0 < noisy["ci_low"] < noisy["rho"] < noisy["ci_high"] < 1.0
    anti = convergent_validity(x, -x)
    assert anti["rho"] == pytest.approx(-1.0)


def test_convergent_validity_too_few_or_constant_returns_nan() -> None:
    few = convergent_validity(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 3.0, 2.0]))
    assert few["n"] == 3 and np.isnan(few["rho"]) and np.isnan(few["p_value"])
    const = convergent_validity(pd.Series(np.arange(10.0)), pd.Series(np.ones(10)))
    assert np.isnan(const["rho"])


def test_convergent_validity_rejects_unit_mismatch_and_duplicates() -> None:
    listing = pd.Series(np.arange(100.0), index=pd.Index(range(100), name="listing_id"))
    muni = pd.Series(np.arange(50.0), index=pd.Index(range(50), name="municipality_id"))
    with pytest.raises(ValueError, match="differ"):
        convergent_validity(listing, muni)
    with pytest.raises(ValueError, match="unique"):
        convergent_validity(pd.Series([1.0, 2.0], index=[1, 1]), pd.Series([1.0], index=[1]))
    unnamed = convergent_validity(muni, muni.rename_axis(None) ** 2)
    assert unnamed["rho"] == pytest.approx(1.0) and unnamed["p"] == unnamed["p_value"]


def test_convergent_validity_ci_uses_bonett_wright_variance() -> None:
    x = pd.Series(np.arange(40.0))
    y = x + np.random.default_rng(3).normal(0.0, 10.0, 40)
    res = convergent_validity(x, y, ci=0.9)
    rho = stats.spearmanr(x, y).statistic
    half = stats.norm.ppf(0.95) * np.sqrt((1.0 + rho**2 / 2.0) / 37.0)
    assert res["rho"] == pytest.approx(rho) and res["n"] == 40
    assert res["ci_low"] == pytest.approx(np.tanh(np.arctanh(rho) - half))
    assert res["ci_high"] == pytest.approx(np.tanh(np.arctanh(rho) + half))


def test_value_score_ranks_quality_against_price() -> None:
    idx = pd.Index([1, 2, 3, 4], name="listing_id")
    q = pd.Series([90.0, 50.0, 10.0, 70.0], index=idx)
    p = pd.Series([10.0, 20.0, 30.0, np.nan], index=idx)
    score = value_score(q, p)
    assert score.name == "value_score"
    np.testing.assert_allclose(score, [100.0, 50.0, 0.0, np.nan])
    np.testing.assert_allclose(value_score(q, p, qoli_weight=1.0).iloc[:3], [100.0, 50.0, 0.0])
    tie = value_score(pd.Series([50.0, 50.0]), pd.Series([10.0, 20.0]))
    np.testing.assert_allclose(tie, [75.0, 25.0])
    assert value_score(pd.Series([10.0]), pd.Series([5.0])).iloc[0] == pytest.approx(50.0)


def test_value_score_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        value_score(pd.Series([1.0]), pd.Series([10.0]), qoli_weight=1.5)
    with pytest.raises(ValueError):
        value_score(pd.Series([1.0, 2.0]), pd.Series([10.0, 0.0]))


def test_pipeline_end_to_end(indicator_table: pd.DataFrame) -> None:
    norm = minmax_normalize(indicator_table, INDICATORS)
    score, dims = aggregate(norm, INDICATORS, {"transport": 2, "noise": 1, "amenities": 1})
    assert score.between(0.0, 100.0).all() and list(dims.columns) == [
        "transport",
        "noise",
        "amenities",
    ]
    value = value_score(score, pd.Series(25.0, index=score.index) + indicator_table["green"])
    assert value.index.equals(indicator_table.index) and value.between(0.0, 100.0).all()


def test_qoli_reexports_amenity_api() -> None:
    for name in ("OSM_CATEGORIES", "accessibility", "amenity_arrays", "fetch_osm_amenities"):
        assert getattr(qoli, name) is getattr(amenities, name)
        assert name in qoli.__all__
