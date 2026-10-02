"""Tests for rentml.qoli_robustness (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from rentml.qoli import Indicator
from rentml.qoli_robustness import (
    cronbach_alpha,
    dimension_consistency,
    geometric_aggregate,
    leave_one_out,
    percentile_normalize,
    rank_comparison,
    weighted_quantile,
    weighted_unit_scores,
)


def test_cronbach_alpha_perfect_and_degenerate() -> None:
    x = pd.Series(np.arange(10.0))
    assert cronbach_alpha(pd.DataFrame({"a": x, "b": x, "c": x})) == pytest.approx(1.0)
    rng = np.random.default_rng(42)
    noise = pd.DataFrame(rng.normal(size=(500, 3)))
    assert abs(cronbach_alpha(noise)) < 0.2
    assert np.isnan(cronbach_alpha(pd.DataFrame({"a": x})))
    assert np.isnan(cronbach_alpha(pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [2.0, 2.0, 2.0]})))


def test_dimension_consistency_per_dimension() -> None:
    norm = pd.DataFrame({"a1": [1.0, 2, 3, 4], "a2": [2.0, 4, 6, 8], "b": [4.0, 1, 3, 2]})
    inds = [Indicator("a1", "A", True), Indicator("a2", "A", True), Indicator("b", "B", True)]
    out = dimension_consistency(norm, inds)
    assert out.loc["A", "n_indicators"] == 2 and out.loc["A", "mean_spearman"] == 1.0
    assert out.loc["A", "cronbach_alpha"] > 0.8
    assert np.isnan(out.loc["B", "cronbach_alpha"]) and np.isnan(out.loc["B", "mean_spearman"])


def test_percentile_normalize_orients_and_keeps_nan() -> None:
    df = pd.DataFrame({"good": [1.0, 2.0, 3.0, 4.0], "bad": [1.0, 2.0, 3.0, np.nan]})
    out = percentile_normalize(df, [Indicator("good", "d", True), Indicator("bad", "d", False)])
    np.testing.assert_allclose(out["good"], [12.5, 37.5, 62.5, 87.5])
    np.testing.assert_allclose(out["bad"].iloc[:3], [100 - 100 / 6, 50.0, 100 / 6])
    assert np.isnan(out["bad"].iloc[3])
    with pytest.raises(KeyError):
        percentile_normalize(df, [Indicator("x", "d", True)])


def test_percentile_normalize_against_reference() -> None:
    ref = pd.DataFrame({"x": np.arange(100.0)})
    new = pd.DataFrame({"x": [-5.0, 49.0, 500.0]}, index=["a", "b", "c"])
    out = percentile_normalize(new, [Indicator("x", "d", True)], reference=ref)
    np.testing.assert_allclose(out["x"], [0.0, 49.5, 100.0])
    assert list(out.index) == ["a", "b", "c"]
    with pytest.raises(ValueError, match="finite"):
        percentile_normalize(new, [Indicator("x", "d", True)], reference=ref * np.nan)


def test_weighted_quantile_midpoint_cdf() -> None:
    values = np.array([1.0, 2.0, 3.0, np.nan])
    weights = np.array([1.0, 1.0, 2.0, 5.0])
    # midpoint CDF: 1 -> 0.125, 2 -> 0.375, 3 -> 0.75; the median interpolates 2 -> 3
    assert weighted_quantile(values, weights, 0.5) == pytest.approx(2 + 1 / 3)
    assert weighted_quantile(values, np.ones(4), 0.5) == pytest.approx(2.0)
    assert np.isnan(weighted_quantile(values, np.zeros(4), 0.5))
    with pytest.raises(ValueError):
        weighted_quantile(values, weights, 1.5)


def test_geometric_aggregate_penalises_imbalance() -> None:
    dims = pd.DataFrame({"A": [50.0, 90.0, 0.0, np.nan], "B": [50.0, 10.0, 100.0, 50.0]})
    out = geometric_aggregate(dims)
    assert out.iloc[0] == pytest.approx(50.0)
    assert out.iloc[1] == pytest.approx(30.0)  # linear mean would be 50
    assert out.iloc[2] == pytest.approx(10.0)  # floored at 1, not zero
    assert np.isnan(out.iloc[3])
    assert geometric_aggregate(dims, {"A": 0.0, "B": 1.0}).iloc[3] == pytest.approx(50.0)
    with pytest.raises(ValueError):
        geometric_aggregate(dims, floor=0.0)
    with pytest.raises(ValueError):
        geometric_aggregate(dims, {"A": 1.0})


def test_leave_one_out_renormalises_weights() -> None:
    dims = pd.DataFrame({"A": [100.0, 0.0], "B": [0.0, 100.0], "C": [50.0, 50.0]})
    out = leave_one_out(dims, {"A": 2.0, "B": 1.0, "C": 1.0})
    assert set(out) == {"without A", "without B", "without C"}
    np.testing.assert_allclose(out["without C"], [200 / 3, 100 / 3])
    with pytest.raises(ValueError):
        leave_one_out(dims[["A"]])


def test_rank_comparison_identity_and_reversal() -> None:
    base = pd.Series(np.arange(20.0))
    out = rank_comparison(base, {"same": base * 2, "reversed": -base, "tiny": base.iloc[:2]})
    assert out.loc["same", "spearman"] == pytest.approx(1.0)
    assert out.loc["same", "mean_abs_rank_shift_pct"] == 0.0
    assert out.loc["same", "top_decile_retention"] == 1.0
    assert out.loc["reversed", "spearman"] == pytest.approx(-1.0)
    assert out.loc["reversed", "top_decile_retention"] == 0.0
    assert out.loc["tiny", "n"] == 2 and np.isnan(out.loc["tiny", "spearman"])


def test_weighted_unit_scores_skips_missing_per_column() -> None:
    values = pd.DataFrame({"q": [10.0, 30.0, np.nan, 50.0], "r": [1.0, np.nan, 3.0, 4.0]})
    weights = pd.Series([1.0, 3.0, 2.0, 1.0])
    groups = pd.Series([1, 1, 1, 2])
    out = weighted_unit_scores(values, weights, groups)
    assert out.loc[1, "q"] == pytest.approx(25.0)  # (10 + 90) / 4
    assert out.loc[1, "r"] == pytest.approx(7 / 3)  # (1 + 6) / 3
    assert out.loc[1, "weight"] == 6.0 and out.loc[1, "units"] == 3
    assert out.loc[2, "q"] == 50.0
    with pytest.raises(ValueError):
        weighted_unit_scores(values, -weights, groups)
