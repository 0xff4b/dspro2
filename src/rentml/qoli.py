"""Quality-of-Life Index (QoLI) after the OECD/JRC Handbook on Constructing Composite Indicators.

Directed indicators, winsorising and min-max normalisation (0-100, 100 = best), equal weights within
and normalised user weights across dimensions, linear aggregation, Dirichlet weight perturbation
(robustness), convergent validity (no ground truth), noise-limit exceedance and a value score.
Accessibility and the OSM download live in ``rentml.amenities`` and are re-exported here.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rentml.amenities import (
    DECAY_FULL_UNTIL_M,
    DECAY_ZERO_AT_M,
    DEFAULT_ACCESS_WEIGHTS,
    OSM_CATEGORIES,
    OSM_COLUMNS,
    OVERPASS_URL,
    SWISS_BBOX_WGS84,
    accessibility,
    amenity_arrays,
    distance_decay,
    fetch_osm_amenities,
)
from rentml.config import LSV_THRESHOLDS_DBA, RANDOM_STATE

__all__ = [
    "DECAY_FULL_UNTIL_M",
    "DECAY_ZERO_AT_M",
    "DEFAULT_ACCESS_WEIGHTS",
    "OSM_CATEGORIES",
    "OSM_COLUMNS",
    "OVERPASS_URL",
    "SWISS_BBOX_WGS84",
    "Indicator",
    "accessibility",
    "aggregate",
    "amenity_arrays",
    "convergent_validity",
    "distance_decay",
    "fetch_osm_amenities",
    "lsv_exceedance",
    "minmax_normalize",
    "normalization_bounds",
    "value_score",
    "weight_sensitivity",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Indicator:
    """One QoLI indicator.

    Attributes:
        name: Column name in the indicator table.
        dimension: Dimension the indicator belongs to (e.g. ``"transport"``, ``"noise"``).
        higher_is_better: Direction; ``False`` flips the normalised score (e.g. noise in dB).
        description: Source, unit and reference year for the report.
        transform: ``"none"`` or ``"log1p"`` (applied before winsorising, for skewed indicators).
    """

    name: str
    dimension: str
    higher_is_better: bool
    description: str = ""
    transform: str = "none"

    def __post_init__(self) -> None:
        if self.transform not in ("none", "log1p"):
            raise ValueError(f"Unknown transform {self.transform!r}; use 'none' or 'log1p'")


def normalization_bounds(
    df: pd.DataFrame,
    indicators: Sequence[Indicator],
    *,
    winsor: tuple[float, float] | None = (0.01, 0.99),
) -> pd.DataFrame:
    """Reference bounds for ``minmax_normalize``: winsor quantiles on the transformed scale.

    If the quantiles coincide although the values vary (zero-inflated or rare binary indicator),
    the raw min/max are used instead so that the rare values keep their signal.

    Args:
        df: Reference indicator table (e.g. all listings, or the training split).
        indicators: Indicators to compute bounds for.
        winsor: Lower/upper quantiles (``None`` = raw min/max).

    Returns:
        DataFrame (index = indicator name) with ``lo``/``hi`` after the indicator's transform
        (log1p units for ``transform="log1p"``); ``lo == hi`` only for constant indicators.

    Raises:
        KeyError: If an indicator column is missing.
        ValueError: For duplicate indicators, invalid winsor quantiles or no finite values.
    """
    _check_indicators(indicators)
    q = (0.0, 1.0) if winsor is None else winsor
    if not 0.0 <= q[0] < q[1] <= 1.0:
        raise ValueError(f"winsor must satisfy 0 <= lower < upper <= 1, got {winsor}")
    rows: dict[str, tuple[float, float]] = {}
    for ind in indicators:
        v = _transformed(df, ind)
        if (v := v[np.isfinite(v)]).size == 0:
            raise ValueError(f"Indicator {ind.name!r} has no finite values")
        lo, hi = (float(b) for b in np.quantile(v, q))
        if hi <= lo and v.min() < v.max():
            logger.warning("%s: winsor quantiles coincide; using the raw min/max", ind.name)
            lo, hi = float(v.min()), float(v.max())
        rows[ind.name] = (lo, hi)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["lo", "hi"])


def minmax_normalize(
    df: pd.DataFrame,
    indicators: Sequence[Indicator],
    *,
    winsor: tuple[float, float] | None = (0.01, 0.99),
    bounds: pd.DataFrame | Mapping[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Winsorise and min-max normalise indicators to 0-100, oriented so that higher is better.

    Values beyond the bounds are clipped, NaN stays NaN, an indicator without spread (lo == hi)
    scores 50. To score new locations on the reference scale, pass ``bounds``.

    Args:
        df: Indicator table (one column per indicator).
        indicators: Indicators to normalise.
        winsor: Lower/upper winsor quantiles (``None`` = raw min/max); ignored with ``bounds``.
        bounds: Bounds on the transformed scale, from ``normalization_bounds`` (DataFrame with
            ``lo``/``hi``) or a mapping name -> (lo, hi); ``None`` = computed from ``df``.

    Returns:
        DataFrame (index of ``df``) with one 0-100 column per indicator; ``attrs["bounds"]``
        holds the bounds used as a plain dict name -> (lo, hi).

    Raises:
        KeyError: If an indicator column or its bounds are missing.
        ValueError: For duplicate indicators, invalid winsor quantiles or no finite values.
    """
    _check_indicators(indicators)
    if bounds is None:
        bounds = normalization_bounds(df, indicators, winsor=winsor)
    elif isinstance(bounds, Mapping):
        bounds = pd.DataFrame.from_dict(dict(bounds), orient="index", columns=["lo", "hi"])
    out: dict[str, np.ndarray] = {}
    used: dict[str, tuple[float, float]] = {}
    for ind in indicators:
        if ind.name not in bounds.index:
            raise KeyError(f"No normalisation bounds for indicator {ind.name!r}")
        lo, hi = (float(v) for v in bounds.loc[ind.name, ["lo", "hi"]])
        used[ind.name] = (lo, hi)
        values = _transformed(df, ind)
        if hi <= lo:
            logger.warning("Indicator %s has no spread (lo >= hi); scored 50", ind.name)
            score = np.where(np.isnan(values), np.nan, 50.0)
        else:
            score = (np.clip(values, lo, hi) - lo) / (hi - lo) * 100.0
        out[ind.name] = score if ind.higher_is_better else 100.0 - score
    result = pd.DataFrame(out, index=df.index)
    result.attrs["bounds"] = used  # plain tuples: pandas compares attrs when combining frames
    return result


def aggregate(
    norm: pd.DataFrame,
    indicators: Sequence[Indicator],
    dimension_weights: Mapping[str, float] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Aggregate normalised indicators into dimension scores and the QoLI (linear, 0-100).

    Equal weights within a dimension (a missing indicator is skipped, i.e. imputed by the unit's
    dimension mean). A unit without a score in a positively weighted dimension gets QoLI NaN.

    Args:
        norm: Output of ``minmax_normalize``.
        indicators: Indicator definitions (dimension membership).
        dimension_weights: Weight per dimension (normalised to sum 1); ``None`` = equal weights.

    Returns:
        Tuple of the QoLI (Series ``"qoli"``) and the dimension scores (one column each).

    Raises:
        KeyError: If an indicator column is missing from ``norm``.
        ValueError: For duplicate indicators or invalid weights.
    """
    _check_indicators(indicators)
    if missing := [ind.name for ind in indicators if ind.name not in norm.columns]:
        raise KeyError(f"Normalised indicators missing: {missing}")
    dims = list(dict.fromkeys(ind.dimension for ind in indicators))
    weights = np.array(list(_dimension_weights(dims, dimension_weights).values()))
    members = {d: [i.name for i in indicators if i.dimension == d] for d in dims}
    dim_scores = pd.DataFrame({d: norm[c].mean(axis=1) for d, c in members.items()})
    values = dim_scores.to_numpy(dtype=float)
    qoli = np.nansum(values * weights, axis=1)
    qoli[np.isnan(values[:, weights > 0]).any(axis=1)] = np.nan
    return pd.Series(qoli, index=norm.index, name="qoli"), dim_scores


def lsv_exceedance(
    day_db: np.ndarray | pd.Series,
    night_db: np.ndarray | pd.Series,
    thresholds: Mapping[str, float] = LSV_THRESHOLDS_DBA,
) -> pd.DataFrame:
    """Compare modelled noise levels (sonBASE L_r,Tag/L_r,Nacht) with the LSV impact thresholds.

    Call it once per noise type (road: LSV Annex 3, railway: Annex 4); the limits are assessed per
    type (LSV Art. 40), so do not sum road and rail levels energetically before calling it.

    Args:
        day_db: Day rating level (06-22 h) per unit in dB(A); modelled, not measured.
        night_db: Night rating level (22-06 h) per unit; aligned by label if both are Series,
            else by position.
        thresholds: Limits with keys ``"day"``/``"night"`` (default: level II, 60/50 dB(A)).

    Returns:
        DataFrame with ``day_excess_db``, ``night_excess_db`` (dB strictly above the limit, else
        0) and nullable booleans ``exceeds_day``, ``exceeds_night``, ``exceeds_any`` (<NA> if
        unknown); index of ``day_db`` if it is a Series.

    Raises:
        ValueError: For inputs that are not 1-D of equal length or Series with other labels.
        KeyError: If a threshold is missing.
    """
    both_series = isinstance(day_db, pd.Series) and isinstance(night_db, pd.Series)
    if both_series and not day_db.index.equals(night_db.index):
        if set(day_db.index) != set(night_db.index):
            raise ValueError("day_db and night_db Series must carry the same index labels")
        night_db = night_db.reindex(day_db.index)  # by label, not by position
    levels = {"day": np.asarray(day_db, dtype=float), "night": np.asarray(night_db, dtype=float)}
    if levels["day"].shape != levels["night"].shape or levels["day"].ndim != 1:
        raise ValueError("day_db and night_db must be 1-D of equal length")
    if missing := {"day", "night"} - set(thresholds):
        raise KeyError(f"Missing noise thresholds: {sorted(missing)}")
    index = day_db.index if isinstance(day_db, pd.Series) else None
    out = pd.DataFrame(index=index if index is not None else pd.RangeIndex(len(levels["day"])))
    for period, level in levels.items():
        excess = level - float(thresholds[period])  # NaN stays NaN in both columns
        out[f"{period}_excess_db"] = np.clip(excess, 0.0, None)
        flag = np.where(np.isnan(excess), None, excess > 0)
        out[f"exceeds_{period}"] = pd.array(flag, dtype="boolean")
    out["exceeds_any"] = out["exceeds_day"] | out["exceeds_night"]
    return out


def weight_sensitivity(
    dim_scores: pd.DataFrame,
    *,
    n_draws: int = 500,
    concentration: float = 10.0,
    seed: int = RANDOM_STATE,
    base_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Rank stability of the index under random dimension weights (OECD/JRC uncertainty analysis).

    Weights ~ Dirichlet(concentration · w0): mean w0 (baseline), Var(w_i) = w0_i (1 - w0_i) /
    (concentration + 1); zero baseline weights stay 0. As in ``aggregate``, a unit is dropped
    only if a positively weighted dimension is missing.

    Args:
        dim_scores: Dimension scores (units x dimensions) from ``aggregate``.
        n_draws: Number of weight draws.
        concentration: Dirichlet precision; larger = smaller perturbations.
        seed: Random seed.
        base_weights: Baseline dimension weights; ``None`` = equal weights.

    Returns:
        One row per draw: ``w_<dim>``, ``spearman`` (vs. baseline ranks),
        ``mean_abs_rank_shift_pct`` (mean |percentile-rank change| in percentile points),
        ``p90_abs_rank_shift_pct`` and ``top_decile_retention`` (share of the top 10 % staying).

    Raises:
        ValueError: For < 3 complete units, < 2 dimensions or invalid settings.
    """
    dims = [str(c) for c in dim_scores.columns]
    if len(dims) < 2 or concentration <= 0 or n_draws < 1:
        raise ValueError(f"Need >= 2 dimensions and valid settings, got {dims}, {concentration}")
    w0 = np.array(list(_dimension_weights(dims, base_weights).values()))
    data = dim_scores.loc[dim_scores.loc[:, w0 > 0].notna().all(axis=1)].fillna(0.0)
    if (n := len(data)) < 3:
        raise ValueError(f"Need >= 3 units with all weighted dimensions, got {n}")
    draws = np.zeros((n_draws, len(dims)))
    draws[:, w0 > 0] = np.random.default_rng(seed).dirichlet(concentration * w0[w0 > 0], n_draws)
    base, perturbed = data.to_numpy(float) @ w0, data.to_numpy(float) @ draws.T
    base_rank, ranks = stats.rankdata(base), stats.rankdata(perturbed, axis=0)
    shift = np.abs(ranks - base_rank[:, None]) / (n - 1) * 100.0
    with np.errstate(invalid="ignore", divide="ignore"):
        spearman = np.corrcoef(np.column_stack([base_rank, ranks]), rowvar=False)[0, 1:]
    n_top = max(1, int(np.ceil(0.1 * n)))
    base_top = stats.rankdata(base, method="ordinal") > n - n_top
    draw_top = stats.rankdata(perturbed, method="ordinal", axis=0) > n - n_top
    out = pd.DataFrame(draws, columns=[f"w_{d}" for d in dims])
    out["spearman"] = spearman
    out["mean_abs_rank_shift_pct"] = shift.mean(axis=0)
    out["p90_abs_rank_shift_pct"] = np.quantile(shift, 0.9, axis=0)
    out["top_decile_retention"] = (draw_top & base_top[:, None]).sum(axis=0) / n_top
    return out


def convergent_validity(
    score: pd.Series, external: pd.Series, *, ci: float = 0.95
) -> dict[str, float]:
    """Spearman correlation of the index with an independent external indicator.

    Aggregate listings to the external unit first (e.g. median QoLI per municipality) and name
    both indexes (e.g. ``municipality_id``). The CI uses the Fisher z-transform with the
    Bonett & Wright (2000) variance (1 + rho²/2) / (n - 3) for Spearman's rho.

    Args:
        score: Index values per unit (unique index; aligned with ``external``, NaN pairs dropped).
        external: External indicator on the same units.
        ci: Confidence level of the interval.

    Returns:
        Dict with ``rho``, ``p_value`` (alias ``p``), ``ci_low``, ``ci_high``, ``n``;
        statistics are NaN for fewer than 4 pairs or a constant series.

    Raises:
        ValueError: If the index names differ (e.g. listing_id vs. municipality_id) or an
            index has duplicates.
    """
    names = (score.index.name, external.index.name)
    if None not in names and names[0] != names[1]:
        raise ValueError(f"Index names differ (different units?): {names[0]!r} vs {names[1]!r}")
    if not (score.index.is_unique and external.index.is_unique):
        raise ValueError("score and external need unique indexes (aggregate per unit first)")
    pair = pd.concat({"a": score, "b": external}, axis=1, join="inner").dropna()
    n = len(pair)
    if n < 0.5 * score.notna().sum():
        logger.warning("convergent_validity: only %d of %d units matched", n, score.notna().sum())
    result = dict.fromkeys(["rho", "p_value", "p", "ci_low", "ci_high"], np.nan) | {"n": float(n)}
    if n < 4 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        logger.warning("convergent_validity: %d usable pairs or constant input; returning NaN", n)
        return result
    res = stats.spearmanr(pair["a"], pair["b"])
    rho = float(res.statistic)
    with np.errstate(divide="ignore"):  # |rho| = 1 gives z = +-inf, i.e. a degenerate CI
        z = np.arctanh(rho)
    half = float(stats.norm.ppf(0.5 + ci / 2)) * np.sqrt((1.0 + rho**2 / 2.0) / (n - 3))
    result.update(rho=rho, p_value=float(res.pvalue), p=float(res.pvalue))
    result.update(ci_low=float(np.tanh(z - half)), ci_high=float(np.tanh(z + half)))
    return result


def value_score(
    qoli: pd.Series, price_per_sqm: pd.Series, *, qoli_weight: float = 0.5
) -> pd.Series:
    """Descriptive value-for-money score: quality of life relative to price (0-100).

    ``qoli_weight · pct(QoLI) + (1 - qoli_weight) · (100 - pct(CHF/m²))``, pct = percentile rank
    (0-100, ties averaged) among units with both values. The QoLI is location-based and partly
    priced in (hedonic capitalisation), so both parts correlate and a high score can also flag
    unobserved negatives: a descriptive ranking, not a causal statement or a rent verdict. For
    like-with-like comparisons call it per peer group (e.g. size band, language region).

    Args:
        qoli: QoLI per unit.
        price_per_sqm: Asking (or expected) rent in CHF/m² per unit.
        qoli_weight: Weight of the QoLI percentile (price gets ``1 - qoli_weight``).

    Returns:
        Series ``value_score`` on the inner-joined index (NaN where an input is missing).

    Raises:
        ValueError: If ``qoli_weight`` is outside [0, 1] or a price is not positive.
    """
    if not 0.0 <= qoli_weight <= 1.0:
        raise ValueError(f"qoli_weight must be in [0, 1], got {qoli_weight}")
    both = pd.concat({"q": qoli, "p": price_per_sqm}, axis=1, join="inner")
    if (both["p"] <= 0).any():
        raise ValueError("price_per_sqm must be positive")
    valid = both.dropna()
    pct = (valid.rank(method="average") - 1.0) / max(len(valid) - 1, 1) * 100.0
    score = qoli_weight * pct["q"] + (1.0 - qoli_weight) * (100.0 - pct["p"])
    return score.reindex(both.index).rename("value_score")


def _check_indicators(indicators: Sequence[Indicator]) -> None:
    names = [ind.name for ind in indicators]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Need at least one indicator and unique names, got {names}")


def _transformed(df: pd.DataFrame, ind: Indicator) -> np.ndarray:
    if ind.name not in df.columns:
        raise KeyError(f"Indicator column {ind.name!r} missing from the table")
    values = pd.to_numeric(df[ind.name], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if ind.transform == "log1p" and np.any(values < 0):
        raise ValueError(f"log1p transform needs non-negative values for {ind.name!r}")
    return np.log1p(values) if ind.transform == "log1p" else values


def _dimension_weights(dims: list[str], weights: Mapping[str, float] | None) -> dict[str, float]:
    if weights is None:
        return {d: 1.0 / len(dims) for d in dims}
    if set(weights) != set(dims):
        raise ValueError(f"Dimension weights must cover exactly {dims}, got {sorted(weights)}")
    raw = np.array([float(weights[d]) for d in dims])
    if not np.isfinite(raw).all() or np.any(raw < 0) or raw.sum() <= 0:
        raise ValueError(f"Dimension weights must be finite, >= 0 and not all 0: {dict(weights)}")
    return {d: float(v) for d, v in zip(dims, raw / raw.sum(), strict=True)}
