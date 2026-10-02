"""Robustness of the Quality-of-Life Index (OECD/JRC Handbook, multivariate and uncertainty steps).

Internal consistency of the dimensions (Cronbach's alpha), alternative normalisation (percentile
ranks) and aggregation (weighted geometric mean), leave-one-dimension-out variants, the rank
comparison of any alternative specification with the baseline, and population-weighted unit
scores (hectares -> municipalities). The baseline index itself is built in ``rentml.qoli``.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from rentml.qoli import Indicator


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha of the columns of ``items`` (complete rows only).

    alpha = k / (k - 1) · (1 - Σ var(item) / var(Σ items)). Values above about 0.7 indicate
    that the indicators measure one latent dimension; for two indicators alpha is a monotone
    function of their correlation.

    Args:
        items: Units x indicators (already oriented so that higher is better).

    Returns:
        Alpha, or NaN for fewer than two indicators, fewer than three complete rows or zero
        total variance.
    """
    data = items.dropna()
    k = data.shape[1]
    if k < 2 or len(data) < 3:
        return float("nan")
    total_var = data.sum(axis=1).var(ddof=1)
    if not total_var > 0:
        return float("nan")
    return float(k / (k - 1) * (1.0 - data.var(ddof=1).sum() / total_var))


def dimension_consistency(norm: pd.DataFrame, indicators: Sequence[Indicator]) -> pd.DataFrame:
    """Per dimension: number of indicators, Cronbach's alpha and mean inter-indicator Spearman.

    Args:
        norm: Normalised indicators (0-100, higher is better), e.g. from ``minmax_normalize``.
        indicators: Indicator definitions.

    Returns:
        DataFrame indexed by dimension with ``n_indicators``, ``cronbach_alpha`` and
        ``mean_spearman`` (NaN for single-indicator dimensions).
    """
    rows = {}
    for dim in dict.fromkeys(i.dimension for i in indicators):
        cols = [i.name for i in indicators if i.dimension == dim]
        rho = np.nan
        if len(cols) > 1:
            corr = norm[cols].corr(method="spearman").to_numpy()
            rho = float(corr[np.triu_indices(len(cols), k=1)].mean())
        rows[dim] = {
            "n_indicators": len(cols),
            "cronbach_alpha": cronbach_alpha(norm[cols]),
            "mean_spearman": rho,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("dimension")


def percentile_normalize(
    df: pd.DataFrame,
    indicators: Sequence[Indicator],
    *,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Percentile normalisation (0-100, higher is better): an alternative to min-max.

    The score is the mid-rank percentile of a value in the reference distribution,
    ``(#ref < v + 0.5 · #ref == v) / #ref · 100``, flipped for indicators where less is better.
    Ranks ignore distances between values, so a few extreme locations cannot compress the rest.

    Args:
        df: Raw indicator table to score.
        indicators: Indicators to normalise (``transform`` is irrelevant for ranks).
        reference: Table that defines the distribution (e.g. all inhabited hectares);
            ``None`` = ``df`` itself.

    Returns:
        DataFrame (index of ``df``) with one column per indicator; NaN stays NaN.

    Raises:
        KeyError: If an indicator column is missing.
        ValueError: If the reference has no finite value for an indicator.
    """
    ref_table = df if reference is None else reference
    out = {}
    for ind in indicators:
        if ind.name not in df.columns or ind.name not in ref_table.columns:
            raise KeyError(f"Indicator column {ind.name!r} missing from the table")
        ref = pd.to_numeric(ref_table[ind.name], errors="coerce").to_numpy(dtype=float)
        ref = np.sort(ref[np.isfinite(ref)])
        if ref.size == 0:
            raise ValueError(f"Indicator {ind.name!r} has no finite reference values")
        v = pd.to_numeric(df[ind.name], errors="coerce").to_numpy(dtype=float)
        below = np.searchsorted(ref, v, side="left")
        ties = np.searchsorted(ref, v, side="right") - below
        pct = np.where(np.isnan(v), np.nan, (below + 0.5 * ties) / ref.size * 100.0)
        out[ind.name] = pct if ind.higher_is_better else 100.0 - pct
    return pd.DataFrame(out, index=df.index)


def weighted_quantile(
    values: pd.Series | np.ndarray, weights: pd.Series | np.ndarray, q: float
) -> float:
    """Quantile of ``values`` with non-negative ``weights`` (NaN values are skipped).

    Uses the weighted empirical CDF at the midpoints of the weights, so equal weights give
    the usual linear-interpolation quantile up to the end points.

    Args:
        values: Values (e.g. an indicator per hectare).
        weights: Weights of equal length (e.g. residents per hectare).
        q: Quantile in [0, 1].

    Returns:
        The weighted quantile, NaN without positive weight.

    Raises:
        ValueError: For q outside [0, 1], unequal lengths or negative weights.
    """
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if not 0.0 <= q <= 1.0 or v.shape != w.shape or np.any(w < 0):
        raise ValueError("Need q in [0, 1], equal lengths and non-negative weights")
    keep = np.isfinite(v) & (w > 0)
    if not keep.any():
        return float("nan")
    order = np.argsort(v[keep])
    v, w = v[keep][order], w[keep][order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cdf, v))


def geometric_aggregate(
    dim_scores: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    *,
    floor: float = 1.0,
) -> pd.Series:
    """Weighted geometric mean of the dimension scores (limited compensability).

    A low score in one dimension can be offset less by high scores elsewhere than in the linear
    mean. Scores are floored at ``floor`` so that a single 0 does not zero the index.

    Args:
        dim_scores: Dimension scores (units x dimensions, 0-100) from ``aggregate``.
        weights: Weight per dimension (normalised to sum 1); ``None`` = equal.
        floor: Lower bound applied before taking logarithms (> 0).

    Returns:
        Series ``qoli_geometric``; NaN where a positively weighted dimension is missing.

    Raises:
        ValueError: For a non-positive floor or weights that do not match the dimensions.
    """
    if floor <= 0:
        raise ValueError(f"floor must be > 0, got {floor}")
    w = _weights(list(dim_scores.columns), weights)
    values = np.log(np.clip(dim_scores.to_numpy(dtype=float), floor, None))
    score = np.exp(np.nansum(values * w, axis=1))
    score[np.isnan(values[:, w > 0]).any(axis=1)] = np.nan
    return pd.Series(score, index=dim_scores.index, name="qoli_geometric")


def leave_one_out(
    dim_scores: pd.DataFrame, weights: Mapping[str, float] | None = None
) -> dict[str, pd.Series]:
    """Linear index without each dimension in turn (remaining weights renormalised).

    Args:
        dim_scores: Dimension scores (units x dimensions).
        weights: Baseline dimension weights; ``None`` = equal.

    Returns:
        ``"without <dimension>"`` -> index Series.

    Raises:
        ValueError: For fewer than two dimensions.
    """
    dims = list(dim_scores.columns)
    if len(dims) < 2:
        raise ValueError("Need at least two dimensions")
    w = dict(zip(dims, _weights(dims, weights), strict=True))
    out = {}
    for drop in dims:
        keep = [d for d in dims if d != drop and w[d] > 0]
        if not keep:
            continue
        sub_w = np.array([w[d] for d in keep]) / sum(w[d] for d in keep)
        out[f"without {drop}"] = (dim_scores[keep] * sub_w).sum(axis=1, min_count=len(keep))
    return out


def rank_comparison(baseline: pd.Series, alternatives: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Compare the ranking of alternative index specifications with the baseline.

    Args:
        baseline: Baseline index per unit.
        alternatives: Name -> alternative index on (a subset of) the same units.

    Returns:
        One row per alternative: ``n`` (common units), ``spearman``, ``mean_abs_rank_shift_pct``
        and ``p90_abs_rank_shift_pct`` (percentile points) and ``top_decile_retention``.
    """
    rows = {}
    for name, alt in alternatives.items():
        pair = pd.concat({"b": baseline, "a": alt}, axis=1, join="inner").dropna()
        n = len(pair)
        if n < 3:
            rows[name] = {"n": n}
            continue
        rb, ra = stats.rankdata(pair["b"]), stats.rankdata(pair["a"])
        shift = np.abs(rb - ra) / (n - 1) * 100.0
        n_top = max(1, int(np.ceil(0.1 * n)))
        top_b = stats.rankdata(pair["b"], method="ordinal") > n - n_top
        top_a = stats.rankdata(pair["a"], method="ordinal") > n - n_top
        rows[name] = {
            "n": n,
            "spearman": float(stats.spearmanr(pair["b"], pair["a"]).statistic),
            "mean_abs_rank_shift_pct": float(shift.mean()),
            "p90_abs_rank_shift_pct": float(np.quantile(shift, 0.9)),
            "top_decile_retention": float((top_a & top_b).sum() / n_top),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("specification")


def weighted_unit_scores(
    values: pd.DataFrame, weights: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """Weighted mean of each column per group (e.g. population-weighted hectares per municipality).

    NaN values are skipped per column, so each column uses the weight of its non-missing rows.

    Args:
        values: Units x scores.
        weights: Non-negative weight per unit (e.g. inhabitants), aligned with ``values``.
        groups: Group label per unit (e.g. ``municipality_id``); NaN labels are dropped.

    Returns:
        DataFrame indexed by group with the weighted means plus ``weight`` (total weight) and
        ``units`` (number of units).

    Raises:
        ValueError: For negative weights.
    """
    w = weights.reindex(values.index).astype(float)
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    g = groups.reindex(values.index)
    present = values.notna()
    weighted = values.fillna(0.0).mul(w, axis=0).groupby(g).sum()
    denom = present.mul(w, axis=0).groupby(g).sum()
    out = weighted / denom.where(denom > 0)
    out["weight"] = w.groupby(g).sum()
    out["units"] = w.groupby(g).size()
    return out


def _weights(dims: list[str], weights: Mapping[str, float] | None) -> np.ndarray:
    if weights is None:
        return np.full(len(dims), 1.0 / len(dims))
    if set(weights) != set(dims):
        raise ValueError(f"Dimension weights must cover exactly {dims}, got {sorted(weights)}")
    raw = np.array([float(weights[d]) for d in dims])
    if not np.isfinite(raw).all() or np.any(raw < 0) or raw.sum() <= 0:
        raise ValueError(f"Dimension weights must be finite, >= 0 and not all 0: {dict(weights)}")
    return raw / raw.sum()
