"""Tests for rentml.qoli (index construction, robustness and validity)."""

import numpy as np
import pandas as pd
import pytest

from rentml.qoli import (
    Indicator,
    aggregate,
    apply_quiet_noise,
    convergent_validity,
    cronbach_alpha,
    dimension_consistency,
    geometric_aggregate,
    lden_from_lr,
    leave_one_out,
    lsv_exceedance,
    minmax_normalize,
    normalization_bounds,
    percentile_normalize,
    rank_comparison,
    value_score,
    weight_sensitivity,
    weighted_unit_scores,
)

NOISE = Indicator("noise", "quiet", False)
GREEN = Indicator("green", "green", True)
STOP = Indicator("stop_m", "transport", False, transform="log1p")


def _dims(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 100, n)
    return pd.DataFrame({d: base + rng.normal(0, 10, n) for d in ("a", "b", "c")})


def test_minmax_normalize_flips_clips_and_uses_reference_bounds() -> None:
    ref = pd.DataFrame({"noise": np.linspace(40, 80, 101), "green": np.linspace(0, 1, 101)})
    bounds = normalization_bounds(ref, [NOISE, GREEN], winsor=None)
    new = pd.DataFrame({"noise": [40.0, 80.0, 100.0, np.nan], "green": [0.5, 2.0, 0.0, 1.0]})
    norm = minmax_normalize(new, [NOISE, GREEN], bounds=bounds)
    assert norm["noise"].tolist()[:3] == [100.0, 0.0, 0.0]
    assert np.isnan(norm["noise"].iloc[3])
    assert norm["green"].tolist() == [50.0, 100.0, 0.0, 100.0]


def test_bounds_log_transform_and_zero_inflated_fallback() -> None:
    df = pd.DataFrame({"stop_m": np.r_[np.zeros(299), 1000.0], "green": np.r_[np.zeros(299), 1.0]})
    assert normalization_bounds(df, [STOP], winsor=None).loc["stop_m", "hi"] == np.log1p(1000.0)
    bounds = normalization_bounds(df, [GREEN])
    assert bounds.loc["green"].tolist() == [0.0, 1.0]  # winsor quantiles coincide -> min/max


def test_constant_indicator_scores_fifty() -> None:
    norm = minmax_normalize(pd.DataFrame({"green": [0.3, 0.3, np.nan]}), [GREEN])
    assert norm["green"].iloc[:2].tolist() == [50.0, 50.0] and np.isnan(norm["green"].iloc[2])


def test_percentile_normalize_mid_ranks_against_reference() -> None:
    ref = pd.DataFrame({"noise": [1.0, 2.0, 3.0, 4.0]})
    pct = percentile_normalize(pd.DataFrame({"noise": [2.0, 10.0, np.nan]}), [NOISE], ref)
    assert pct["noise"].iloc[:2].tolist() == [100 - 37.5, 0.0] and np.isnan(pct["noise"].iloc[2])


def test_apply_quiet_noise_sets_best_score() -> None:
    norm = pd.DataFrame({"noise_day_db": [10.0, np.nan], "green": [1.0, 2.0]})
    raw = pd.DataFrame({"noise_quiet_day": [False, True]})
    assert apply_quiet_noise(norm, raw)["noise_day_db"].tolist() == [10.0, 100.0]


def test_aggregate_weights_dimensions_and_requires_weighted_ones() -> None:
    inds = [Indicator("x1", "x", True), Indicator("x2", "x", True), Indicator("y1", "y", True)]
    norm = pd.DataFrame({"x1": [0.0, 50.0, 10.0], "x2": [100.0, np.nan, 10.0],
                         "y1": [100.0, 0.0, np.nan]})  # fmt: skip
    qoli, dims = aggregate(norm, inds, {"x": 3.0, "y": 1.0})
    assert dims["x"].tolist() == [50.0, 50.0, 10.0]
    assert qoli.iloc[:2].tolist() == [62.5, 37.5] and np.isnan(qoli.iloc[2])
    assert aggregate(norm, inds, {"x": 1.0, "y": 0.0})[0].tolist() == [50.0, 50.0, 10.0]
    with pytest.raises(KeyError):
        aggregate(norm, inds, {"x": 1.0})


def test_geometric_and_leave_one_out_variants() -> None:
    dims = pd.DataFrame({"a": [100.0, 0.0], "b": [25.0, 100.0], "c": [np.nan, 100.0]})
    geo = geometric_aggregate(dims[["a", "b"]])
    assert geo.iloc[0] == pytest.approx(50.0) and geo.iloc[1] == pytest.approx(10.0)
    loo = leave_one_out(dims)
    assert loo["without c"].tolist() == [62.5, 50.0] and np.isnan(loo["without a"].iloc[0])


def test_weight_sensitivity_is_reproducible_and_rank_stable() -> None:
    dims = _dims()
    first = weight_sensitivity(dims, n_draws=50, seed=1)
    pd.testing.assert_frame_equal(first, weight_sensitivity(dims, n_draws=50, seed=1))
    assert np.allclose(first.filter(like="w_").sum(axis=1), 1.0)
    assert first["spearman"].median() > 0.9 and first["top_decile_retention"].between(0, 1).all()
    tight = weight_sensitivity(dims, n_draws=50, seed=1, concentration=1000.0)
    assert tight["spearman"].min() > first["spearman"].min()


def test_rank_comparison_identical_and_reversed() -> None:
    base = pd.Series(np.arange(20.0))
    table = rank_comparison(base, {"same": base * 2, "reversed": -base, "half": base.iloc[:10]})
    assert table.loc["same", "spearman"] == pytest.approx(1.0)
    assert table.loc["same", ["mean_abs_rank_shift_pct"]].item() == 0.0
    assert table.loc["reversed", "spearman"] == pytest.approx(-1.0)
    assert table.loc["reversed", "top_decile_retention"] == 0.0
    assert table.loc["half", "n"] == 10


def test_cronbach_alpha_and_dimension_consistency() -> None:
    rng = np.random.default_rng(3)
    latent = rng.normal(size=500)
    items = pd.DataFrame({f"i{k}": latent + rng.normal(0, 0.3, 500) for k in range(3)})
    assert cronbach_alpha(items) > 0.9
    noise = items.assign(i1=rng.normal(size=500), i2=rng.normal(size=500))
    assert cronbach_alpha(noise) < 0.3
    inds = [Indicator("i0", "d", True), Indicator("i1", "d", True), Indicator("i2", "e", True)]
    table = dimension_consistency(items, inds)
    assert table.loc["d", "n_indicators"] == 2 and table.loc["d", "cronbach_alpha"] > 0.9
    assert np.isnan(table.loc["e", "cronbach_alpha"])


def test_weighted_unit_scores_skips_missing_values() -> None:
    values = pd.DataFrame({"s": [10.0, 20.0, np.nan, 40.0]})
    out = weighted_unit_scores(values, pd.Series([1.0, 3.0, 5.0, 1.0]), pd.Series([1, 1, 2, 2]))
    assert out["s"].tolist() == [17.5, 40.0] and out["weight"].tolist() == [4.0, 6.0]


def test_lsv_exceedance_per_period_with_unknown_levels() -> None:
    out = lsv_exceedance(pd.Series([55.0, 65.0, np.nan]), pd.Series([51.0, 40.0, 45.0]))
    assert out["day_excess_db"].iloc[1] == 5.0 and out["night_excess_db"].iloc[0] == 1.0
    assert out["exceeds_any"].iloc[:2].all() and pd.isna(out["exceeds_any"].iloc[2])


def test_lden_from_lr_known_value() -> None:
    lden = lden_from_lr(pd.Series([60.0]), pd.Series([50.0]))
    assert lden.iloc[0] == pytest.approx(60.0 + 10 * np.log10((12 + 4 * 10**0.5 + 8) / 24))


def test_convergent_validity_ci_and_too_few_pairs() -> None:
    rng = np.random.default_rng(5)
    score = pd.Series(rng.normal(size=200))
    res = convergent_validity(score, score + rng.normal(0, 0.5, 200))
    assert res["n"] == 200 and res["ci_low"] < res["rho"] < res["ci_high"] and res["rho"] > 0.8
    assert np.isnan(convergent_validity(score.iloc[:3], score.iloc[:3])["rho"])


def test_value_score_prefers_good_and_cheap() -> None:
    qoli = pd.Series([90.0, 90.0, 10.0, np.nan], index=list("abcd"))
    price = pd.Series([20.0, 40.0, 40.0, 10.0], index=list("abcd"))
    score = value_score(qoli, price)
    assert score["a"] == 87.5 and score["a"] > score["b"] > score["c"] and np.isnan(score["d"])
