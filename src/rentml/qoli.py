"""Quality-of-Life Index (QoLI) after the OECD/JRC Handbook on Constructing Composite Indicators.

Winsorised min-max normalisation (0-100, 100 = best), equal weights within and user weights
across dimensions, linear aggregation, robustness checks (weights, normalisation, aggregation,
indicator set), internal consistency, convergent validity, noise thresholds and a value score.
The raw indicators come from ``rentml.qoli_data``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rentml.config import LSV_THRESHOLDS_DBA, RANDOM_STATE

# WHO Environmental Noise Guidelines for the European Region (2018), L_den in dB.
WHO_LDEN_DB = {"road": 53.0, "rail": 54.0}


@dataclass(frozen=True)
class Indicator:
    """One QoLI indicator; ``transform`` is ``"none"`` or ``"log1p"`` (for skewed indicators)."""

    name: str
    dimension: str
    higher_is_better: bool
    description: str = ""
    transform: str = "none"


def _transformed(df: pd.DataFrame, ind: Indicator) -> pd.Series:
    values = df[ind.name].astype(float)
    return np.log1p(values) if ind.transform == "log1p" else values


def _dimensions(indicators: list[Indicator]) -> dict[str, list[str]]:
    dims: dict[str, list[str]] = {}
    for ind in indicators:
        dims.setdefault(ind.dimension, []).append(ind.name)
    return dims


def _weights(dims: list[str], weights: dict[str, float] | None) -> pd.Series:
    w = pd.Series(1.0, index=dims) if weights is None else pd.Series({d: weights[d] for d in dims})
    return w / w.sum()


def normalization_bounds(
    df: pd.DataFrame,
    indicators: list[Indicator],
    winsor: tuple[float, float] | None = (0.01, 0.99),
) -> pd.DataFrame:
    """Winsor quantiles (``lo``/``hi``, after the transform) per indicator; ``None`` = min/max."""
    rows = {}
    for ind in indicators:
        values = _transformed(df, ind)
        lo, hi = values.quantile(winsor or (0.0, 1.0))
        if hi <= lo:  # zero-inflated or rare binary indicator: keep the rare values' signal
            lo, hi = values.min(), values.max()
        rows[ind.name] = (float(lo), float(hi))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["lo", "hi"])


def minmax_normalize(
    df: pd.DataFrame,
    indicators: list[Indicator],
    bounds: pd.DataFrame | None = None,
    winsor: tuple[float, float] | None = (0.01, 0.99),
) -> pd.DataFrame:
    """Clip to the bounds and scale to 0-100 so that higher is better (no spread: 50).

    Pass the ``bounds`` of a reference population (``normalization_bounds``) to score new
    locations on its scale; without them they are computed from ``df``.
    """
    if bounds is None:
        bounds = normalization_bounds(df, indicators, winsor)
    out = {}
    for ind in indicators:
        lo, hi = bounds.loc[ind.name, ["lo", "hi"]]
        values = _transformed(df, ind)
        if hi > lo:
            score = (values.clip(lo, hi) - lo) / (hi - lo) * 100.0
        else:
            score = values.where(values.isna(), 50.0)
        out[ind.name] = score if ind.higher_is_better else 100.0 - score
    return pd.DataFrame(out, index=df.index)


def percentile_normalize(
    df: pd.DataFrame, indicators: list[Indicator], reference: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Mid-rank percentile (0-100, higher is better) in ``reference`` (default: ``df``)."""
    ref_table = df if reference is None else reference
    out = {}
    for ind in indicators:
        ref = np.sort(ref_table[ind.name].dropna().to_numpy(float))
        values = df[ind.name].to_numpy(float)
        below = np.searchsorted(ref, values, side="left")
        ties = np.searchsorted(ref, values, side="right") - below
        pct = np.where(np.isnan(values), np.nan, (below + 0.5 * ties) / len(ref) * 100.0)
        out[ind.name] = pct if ind.higher_is_better else 100.0 - pct
    return pd.DataFrame(out, index=df.index)


def apply_quiet_noise(norm: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Give places without any modelled road or rail level nearby the best noise score (100)."""
    out = norm.copy()
    for period in ("day", "night"):
        col = f"noise_{period}_db"
        if col in out.columns:
            quiet = raw[f"noise_quiet_{period}"].reindex(out.index).fillna(False).astype(bool)
            out.loc[quiet, col] = 100.0
    return out


def aggregate(
    norm: pd.DataFrame,
    indicators: list[Indicator],
    dimension_weights: dict[str, float] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Dimension scores (mean of their indicators) and the weighted QoLI (linear, 0-100).

    A missing indicator is skipped within its dimension; a unit without a score in a positively
    weighted dimension gets no QoLI. Weights are normalised to sum 1 (``None`` = equal).

    Returns:
        The QoLI (Series ``"qoli"``) and the dimension scores (one column each).
    """
    dims = pd.DataFrame({d: norm[cols].mean(axis=1) for d, cols in _dimensions(indicators).items()})
    w = _weights(list(dims.columns), dimension_weights)
    complete = dims.loc[:, w > 0].notna().all(axis=1)
    return dims.mul(w).sum(axis=1).where(complete).rename("qoli"), dims


def geometric_aggregate(dims: pd.DataFrame, floor: float = 1.0) -> pd.Series:
    """Equally weighted geometric mean of the dimension scores, floored (limited compensation)."""
    score = stats.gmean(dims.clip(lower=floor).to_numpy(float), axis=1)
    return pd.Series(score, index=dims.index, name="qoli_geometric")


def leave_one_out(dims: pd.DataFrame) -> dict[str, pd.Series]:
    """Equally weighted index without each dimension in turn (``"without <dimension>"``)."""
    return {f"without {d}": dims.drop(columns=d).mean(axis=1, skipna=False) for d in dims}


def _rank_agreement(base: np.ndarray, alt: np.ndarray) -> pd.DataFrame:
    """Spearman rho, mean percentile-rank shift and top-decile retention per column of ``alt``."""
    n = len(base)
    base_rank, alt_rank = stats.rankdata(base), stats.rankdata(alt, axis=0)
    n_top = int(np.ceil(0.1 * n))
    base_top = stats.rankdata(base, method="ordinal") > n - n_top
    alt_top = stats.rankdata(alt, method="ordinal", axis=0) > n - n_top
    shift = np.abs(alt_rank - base_rank[:, None]) / (n - 1) * 100.0
    return pd.DataFrame({
        "spearman": [stats.spearmanr(base, a).statistic for a in alt.T],
        "mean_abs_rank_shift_pct": shift.mean(axis=0),
        "top_decile_retention": (alt_top & base_top[:, None]).sum(axis=0) / n_top,
    })  # fmt: skip


def rank_comparison(baseline: pd.Series, alternatives: dict[str, pd.Series]) -> pd.DataFrame:
    """Rank agreement of each alternative specification with the baseline on their common units."""
    rows = {}
    for name, alt in alternatives.items():
        pair = pd.concat({"b": baseline, "a": alt}, axis=1, join="inner").dropna()
        agreement = _rank_agreement(pair["b"].to_numpy(), pair[["a"]].to_numpy())
        rows[name] = {"n": len(pair), **agreement.iloc[0]}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("specification")


def weight_sensitivity(
    dims: pd.DataFrame,
    n_draws: int = 500,
    concentration: float = 10.0,
    seed: int = RANDOM_STATE,
    base_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rank stability under random dimension weights w ~ Dirichlet(concentration · w0).

    The draws have mean w0 (the baseline weights); a larger concentration gives smaller
    perturbations. Units with a missing dimension are dropped.

    Returns:
        One row per draw: ``w_<dimension>``, ``spearman``, ``mean_abs_rank_shift_pct`` and
        ``top_decile_retention`` relative to the baseline ranking.
    """
    data = dims.dropna().to_numpy(float)
    w0 = _weights(list(dims.columns), base_weights).to_numpy()
    draws = np.random.default_rng(seed).dirichlet(concentration * w0, n_draws)
    agreement = _rank_agreement(data @ w0, data @ draws.T)
    return pd.DataFrame(draws, columns=[f"w_{d}" for d in dims.columns]).join(agreement)


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha of the columns (complete rows); about 0.7 or more = one construct."""
    items = items.dropna()
    k = items.shape[1]
    return k / (k - 1) * (1.0 - items.var().sum() / items.sum(axis=1).var())


def dimension_consistency(norm: pd.DataFrame, indicators: list[Indicator]) -> pd.DataFrame:
    """Per dimension: number of indicators, Cronbach's alpha and mean pairwise Spearman rho."""
    rows = {}
    for dim, cols in _dimensions(indicators).items():
        alpha = rho = np.nan
        if len(cols) > 1:
            alpha = cronbach_alpha(norm[cols])
            corr = norm[cols].corr(method="spearman").to_numpy()
            rho = corr[np.triu_indices(len(cols), k=1)].mean()
        rows[dim] = {"n_indicators": len(cols), "cronbach_alpha": alpha, "mean_spearman": rho}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("dimension")


def weighted_unit_scores(
    values: pd.DataFrame, weights: pd.Series, groups: pd.Series
) -> pd.DataFrame:
    """Weighted mean of each column per group (NaN skipped) plus the total ``weight``.

    Used for population-weighted municipal scores from inhabited hectares.
    """
    w = weights.reindex(values.index).astype(float)
    by = groups.reindex(values.index)
    num = values.mul(w, axis=0).groupby(by, observed=True).sum()
    den = values.notna().mul(w, axis=0).groupby(by, observed=True).sum()
    out = num / den.where(den > 0)
    out["weight"] = w.groupby(by, observed=True).sum()
    return out


def lsv_exceedance(
    day_db: pd.Series, night_db: pd.Series, thresholds: dict[str, float] = LSV_THRESHOLDS_DBA
) -> pd.DataFrame:
    """Modelled L_r,Tag/L_r,Nacht against the LSV impact thresholds, once per source type.

    The LSV assesses road and rail separately, so never pass their energetic sum.

    Returns:
        ``day_excess_db``/``night_excess_db`` (dB above the limit, else 0) and nullable booleans
        ``exceeds_day``, ``exceeds_night``, ``exceeds_any`` (<NA> where the level is unknown).
    """
    out = pd.DataFrame(index=day_db.index)
    for period, level in (("day", day_db), ("night", night_db.reindex(day_db.index))):
        excess = level.astype(float) - thresholds[period]
        out[f"{period}_excess_db"] = excess.clip(lower=0.0)
        out[f"exceeds_{period}"] = excess.gt(0).astype("boolean").mask(excess.isna())
    out["exceeds_any"] = out["exceeds_day"] | out["exceeds_night"]
    return out


def lden_from_lr(day_db: pd.Series, night_db: pd.Series) -> pd.Series:
    """Approximate L_den from L_r,Tag and L_r,Nacht, assuming L_evening = L_r,Tag.

    The Swiss rating levels contain level corrections and other periods than L_den, so the result
    is only good for a descriptive comparison with the WHO guideline.
    """
    energy = 12 * 10 ** (day_db / 10) + 4 * 10 ** ((day_db + 5) / 10)
    return 10.0 * np.log10((energy + 8 * 10 ** ((night_db + 10) / 10)) / 24.0)


def convergent_validity(
    score: pd.Series, external: pd.Series, ci: float = 0.95
) -> dict[str, float]:
    """Spearman correlation of the index with an independent indicator on the same units.

    The confidence interval uses the Fisher z-transform with the Bonett & Wright (2000) variance
    (1 + rho²/2) / (n - 3). Statistics are NaN for fewer than four pairs.

    Returns:
        ``rho``, ``p_value``, ``ci_low``, ``ci_high`` and ``n`` (matched units).
    """
    pair = pd.concat({"a": score, "b": external}, axis=1, join="inner").dropna()
    n = len(pair)
    if n < 4:
        return {"rho": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": n}
    rho, p_value = stats.spearmanr(pair["a"], pair["b"])
    half = stats.norm.ppf(0.5 + ci / 2) * np.sqrt((1.0 + rho**2 / 2.0) / (n - 3))
    z = np.arctanh(rho)
    return {"rho": float(rho), "p_value": float(p_value), "ci_low": float(np.tanh(z - half)),
            "ci_high": float(np.tanh(z + half)), "n": n}  # fmt: skip


def value_score(qoli: pd.Series, price_per_sqm: pd.Series, qoli_weight: float = 0.5) -> pd.Series:
    """Descriptive quality-for-price score (0-100) from percentile ranks among complete units.

    ``qoli_weight · pct(QoLI) + (1 - qoli_weight) · (100 - pct(CHF/m²))``. Location quality is
    partly priced in, so a high score can also flag unobserved negatives; call it per peer group.
    """
    both = pd.concat({"q": qoli, "p": price_per_sqm}, axis=1, join="inner")
    valid = both.dropna()
    pct = (valid.rank() - 1.0) / max(len(valid) - 1, 1) * 100.0
    score = qoli_weight * pct["q"] + (1.0 - qoli_weight) * (100.0 - pct["p"])
    return score.reindex(both.index).rename("value_score")
