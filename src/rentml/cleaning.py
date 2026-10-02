"""Target cleaning: domain filters, CHF/m² anomaly detection and rent-regime separation.

The steps follow the DSPRO2 proposal: the DSPRO1 domain rules (area >= 10 m², rent >= CHF 300)
plus upper caps, robust price-per-m² outliers per municipality with a hierarchical fallback
(parking spaces, weekly prices, placeholders), and the separation of cooperative / cost-based /
subsidised rents and shared-flat rooms from the market-rent target.
"""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MAD_TO_Z = 0.6745  # Phi^-1(0.75): makes MAD-based z-scores comparable to normal z-scores.
MEANAD_TO_SIGMA = 1.253314  # sqrt(pi/2): mean absolute deviation -> sigma (normal data).

MARKET_REGIMES: frozenset[str] = frozenset({"market"})
NON_MARKET_REGIMES: frozenset[str] = frozenset(
    {
        "cooperative",
        "cost_based_or_subsidised",
        "cost_based",
        "subsidised",
        "subsidized",
        "shared_flat",
    }
)
_MISSING_REGIMES: frozenset[str] = frozenset({"", "unknown", "none", "nan", "na", "null"})

DOMAIN_REASONS: tuple[str, ...] = (
    "area_missing",
    "area_below_min",
    "area_above_max",
    "price_missing",
    "price_below_min",
    "price_above_max",
)


def _as_float(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("float64")


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Columns not in frame: {missing}")


def domain_filter(
    df: pd.DataFrame,
    *,
    min_area: float = 10.0,
    min_price: float = 300.0,
    max_area: float = 500.0,
    max_price: float = 15000.0,
    area_col: str = "area",
    price_col: str = "price",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the plausibility rules for living area and monthly net rent.

    Bounds are inclusive (DSPRO1: ``area >= 10`` and ``price >= 300``). Rows with a missing
    area or price are removed as well. Rules are checked in the order of
    :data:`DOMAIN_REASONS`; the first failing rule becomes the ``reason``.

    Args:
        df: Listings.
        min_area: Smallest plausible living area in m².
        min_price: Smallest plausible monthly net rent in CHF.
        max_area: Largest plausible living area in m².
        max_price: Largest plausible monthly net rent in CHF.
        area_col: Living-area column.
        price_col: Rent column.

    Returns:
        ``(kept, removed)``; ``removed`` has an extra column ``reason``. Both are copies with
        the original index and column order.

    Raises:
        KeyError: If the area or price column is missing.
        ValueError: If a lower bound exceeds its upper bound.
    """
    _require_columns(df, [area_col, price_col])
    if min_area > max_area or min_price > max_price:
        raise ValueError("Lower bounds must not exceed upper bounds")
    area, price = _as_float(df[area_col]), _as_float(df[price_col])
    conditions = [
        area.isna(),
        area < min_area,
        area > max_area,
        price.isna(),
        price < min_price,
        price > max_price,
    ]
    reason = np.select([c.to_numpy(dtype=bool) for c in conditions], DOMAIN_REASONS, default="")
    removed_mask = reason != ""
    kept = df.loc[~removed_mask].copy()
    removed = df.loc[removed_mask].copy()
    removed["reason"] = reason[removed_mask]
    logger.info(
        "domain_filter: kept %d of %d rows; removed %s",
        len(kept),
        len(df),
        removed["reason"].value_counts().to_dict(),
    )
    return kept, removed


def _object_codes(df: pd.DataFrame, object_ids: pd.Series | None) -> np.ndarray:
    """Positional integer object codes; every row is its own object if ``object_ids`` is None."""
    if object_ids is None:
        return np.arange(len(df), dtype=np.int64)
    if len(object_ids) != len(df) or not object_ids.index.equals(df.index):
        raise ValueError("object_ids must be aligned with df.index")
    if object_ids.isna().any():
        raise ValueError("object_ids contains missing values")
    return pd.factorize(object_ids)[0].astype(np.int64)


def _level_stats(values: pd.Series, keys: pd.Series, objects: np.ndarray) -> pd.DataFrame:
    """Per-row median, MAD and object count of the row's group (NA keys -> NaN).

    Each object contributes one value per group (the median of its listings in that group),
    so duplicate rows can neither shrink the MAD nor satisfy ``min_group``.
    """
    codes = pd.factorize(keys, use_na_sentinel=True)[0]
    has_key = codes >= 0
    units = (
        pd.Series(values.to_numpy()[has_key]).groupby([codes[has_key], objects[has_key]]).median()
    )
    unit_group = units.index.get_level_values(0).to_numpy()
    center = units.groupby(unit_group).median()
    deviation = (units - center.reindex(unit_group).to_numpy()).abs()
    stats = pd.DataFrame(
        {
            "center": center,
            "mad": deviation.groupby(unit_group).median(),
            "group_n": units.groupby(unit_group).size().astype(float),
        }
    )
    return stats.reindex(codes).set_axis(values.index)


def _global_stats(values: pd.Series, objects: np.ndarray) -> tuple[float, float, float]:
    """Global median, sigma and object count (one value per object).

    MAD = 0 falls back to the scaled mean absolute deviation (Iglewicz & Hoaglin); if that is
    0 as well, sigma is 0 and all z-scores are 0.
    """
    per_object = values.groupby(objects).median()
    center = float(per_object.median())
    mad = float((per_object - center).abs().median())
    if mad > 0:
        return center, mad / MAD_TO_Z, float(len(per_object))
    mean_ad = float((per_object - center).abs().mean()) if len(per_object) else 0.0
    return center, MEANAD_TO_SIGMA * mean_ad, float(len(per_object))


def price_per_sqm_zscores(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("municipality_id", "district_id", "canton"),
    min_group: int = 10,
    object_ids: pd.Series | None = None,
    min_scale_ratio: float = 0.5,
    price_col: str = "price",
    area_col: str = "area",
) -> pd.DataFrame:
    """Robust z-scores of log CHF/m² with a hierarchical group fallback.

    ``z = (x - median) / scale`` with ``x = log(price / area)`` and ``scale = MAD / 0.6745``.
    Each row uses the finest level of ``group_cols`` (finest first) whose group has at least
    ``min_group`` distinct objects and a positive MAD; otherwise the parent level. The global
    distribution is the last resort (no size requirement; MAD = 0 falls back to the mean
    absolute deviation). With ``object_ids``, medians, MADs and group sizes are computed on
    one value per object (median of its listings), so re-posted duplicates cannot make a
    small group look artificially tight. Small groups still estimate the MAD noisily, so each
    level's scale is floored at ``min_scale_ratio`` times the (floored) scale of the next
    coarser usable level of the row (global for the coarsest level).

    Args:
        df: Listings.
        group_cols: Grouping columns from finest to coarsest; ``()`` for global only.
        min_group: Minimum number of distinct objects (rows if ``object_ids`` is None) for a
            group to be used.
        object_ids: Optional ``object_id`` per row (aligned with ``df``); recommended.
        min_scale_ratio: Floor of a level's scale relative to its parent scale, in [0, 1];
            ``0`` disables the floor.
        price_col: Rent column (CHF).
        area_col: Living-area column (m²).

    Returns:
        DataFrame aligned with ``df.index`` with columns ``log_ppsqm``, ``center``, ``scale``
        (floored sigma estimate), ``robust_z``, ``level`` (group column name or ``"global"``)
        and ``group_n`` (distinct objects in the group). Rows without a positive price and
        area get NaN / NA.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If ``min_group`` < 1, ``min_scale_ratio`` is outside [0, 1] or
            ``object_ids`` is not aligned with ``df`` or has missing values.
    """
    group_cols = list(group_cols)
    _require_columns(df, [price_col, area_col, *group_cols])
    if min_group < 1:
        raise ValueError("min_group must be >= 1")
    if not 0.0 <= min_scale_ratio <= 1.0:
        raise ValueError("min_scale_ratio must lie in [0, 1]")
    objects = _object_codes(df, object_ids)
    # Work on a positional index so that duplicate index labels cannot misalign groups.
    price = _as_float(df[price_col]).reset_index(drop=True)
    area = _as_float(df[area_col]).reset_index(drop=True)
    log_ppsqm = np.log(price.where(price > 0) / area.where(area > 0))
    valid = log_ppsqm.notna().to_numpy()
    values = log_ppsqm[valid]
    levels = {
        col: _level_stats(values, df[col].reset_index(drop=True)[valid], objects[valid])
        for col in group_cols
    }
    chosen = _choose_levels(values, levels, objects[valid], min_group, min_scale_ratio)
    out = chosen.reindex(log_ppsqm.index)
    out.insert(0, "log_ppsqm", log_ppsqm)
    out["level"] = out["level"].astype("string")
    scale = out["scale"].where(out["scale"] > 0)
    out["robust_z"] = ((out["log_ppsqm"] - out["center"]) / scale).where(scale.notna(), 0.0)
    out["robust_z"] = out["robust_z"].where(pd.Series(valid, index=out.index))
    logger.info("price_per_sqm_zscores: levels used %s", out["level"].value_counts().to_dict())
    out.index = df.index
    return out[["log_ppsqm", "center", "scale", "robust_z", "level", "group_n"]]


def _choose_levels(
    values: pd.Series,
    levels: dict[str, pd.DataFrame],
    objects: np.ndarray,
    min_group: int,
    min_scale_ratio: float,
) -> pd.DataFrame:
    """Finest usable level per row with its center, floored scale and group size."""
    center_g, scale_g, n_g = _global_stats(values, objects)
    chosen = pd.DataFrame(
        {"center": center_g, "scale": scale_g, "group_n": n_g, "level": "global"},
        index=values.index,
    )
    parent_scale = pd.Series(scale_g, index=values.index)
    floored: dict[str, pd.Series] = {}
    usable: dict[str, pd.Series] = {}
    for col in reversed(list(levels)):  # coarsest first: floors cascade down the hierarchy
        stats = levels[col]
        usable[col] = (stats["group_n"] >= min_group) & (stats["mad"] > 0)
        floored[col] = np.maximum(stats["mad"] / MAD_TO_Z, min_scale_ratio * parent_scale)
        parent_scale = floored[col].where(usable[col], parent_scale)
    pending = pd.Series(True, index=values.index)
    for col, stats in levels.items():  # finest first
        rows = pending & usable[col]
        chosen.loc[rows, ["center", "group_n"]] = stats.loc[rows, ["center", "group_n"]]
        chosen.loc[rows, "scale"] = floored[col][rows]
        chosen.loc[rows, "level"] = col
        pending &= ~rows
    return chosen


def price_per_sqm_outliers(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("municipality_id", "district_id", "canton"),
    z_thresh: float = 3.5,
    min_group: int = 10,
    object_ids: pd.Series | None = None,
    min_scale_ratio: float = 0.5,
    price_col: str = "price",
    area_col: str = "area",
) -> pd.Series:
    """Flag implausible CHF/m² values (both tails) with a hierarchical robust z-score.

    See :func:`price_per_sqm_zscores` for the grouping and fallback logic. The default
    threshold 3.5 is the Iglewicz & Hoaglin recommendation for modified z-scores.

    Args:
        df: Listings.
        group_cols: Grouping columns from finest to coarsest; ``()`` for global only.
        z_thresh: Flag rows with ``|z| > z_thresh``.
        min_group: Minimum number of distinct objects (rows if ``object_ids`` is None) for a
            group to be used.
        object_ids: Optional ``object_id`` per row (aligned with ``df``); pass it so that
            duplicates count once in the group statistics (recommended).
        min_scale_ratio: Floor of a level's scale relative to its parent level, in [0, 1].
        price_col: Rent column (CHF).
        area_col: Living-area column (m²).

    Returns:
        Boolean Series ``ppsqm_outlier`` aligned with ``df.index``; rows without a valid
        CHF/m² are False (they are handled by :func:`domain_filter`).

    Raises:
        KeyError: If a required column is missing.
        ValueError: If ``z_thresh`` <= 0, ``min_group`` < 1, ``min_scale_ratio`` is outside
            [0, 1] or ``object_ids`` is misaligned.
    """
    if z_thresh <= 0:
        raise ValueError("z_thresh must be positive")
    scores = price_per_sqm_zscores(
        df,
        group_cols=group_cols,
        min_group=min_group,
        object_ids=object_ids,
        min_scale_ratio=min_scale_ratio,
        price_col=price_col,
        area_col=area_col,
    )
    mask = (scores["robust_z"].abs() > z_thresh).fillna(False).astype(bool)
    logger.info("price_per_sqm_outliers: %d of %d rows flagged", int(mask.sum()), len(df))
    return mask.rename("ppsqm_outlier")


def regime_split(
    df: pd.DataFrame, regime_col: str = "rent_regime"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate market rents from non-market regimes.

    Non-market regimes (cooperative, cost-based or subsidised, shared-flat rooms) sit well
    below market level and are excluded from the market-rent target (reported separately).
    Missing or ``"unknown"`` regimes count as market; if the column is absent, all rows are
    market. Labels are compared case-insensitively with spaces/hyphens mapped to ``_``.

    Args:
        df: Listings.
        regime_col: Column with the rent-regime label (e.g. from LLM extraction).

    Returns:
        ``(market, non_market)`` copies with the original index.

    Raises:
        ValueError: If the column contains labels that are neither market nor non-market.
    """
    if regime_col not in df.columns:
        logger.info("regime_split: column %r missing, all %d rows are market", regime_col, len(df))
        return df.copy(), df.iloc[0:0].copy()
    labels = (
        df[regime_col]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-/]+", "_", regex=True)
    )
    missing = labels.isna() | labels.isin(_MISSING_REGIMES)
    non_market = labels.isin(NON_MARKET_REGIMES).fillna(False).astype(bool)
    unknown = ~missing & ~non_market & ~labels.isin(MARKET_REGIMES).fillna(False)
    if unknown.any():
        raise ValueError(f"Unknown rent regime labels: {sorted(labels[unknown].unique())}")
    logger.info(
        "regime_split: %d market (%d missing regime), %d non-market %s",
        int((~non_market).sum()),
        int(missing.sum()),
        int(non_market.sum()),
        labels[non_market].value_counts().to_dict(),
    )
    return df.loc[~non_market].copy(), df.loc[non_market].copy()


def recover_rooms_from_text(
    rooms: pd.Series,
    text_rooms: pd.Series,
    *,
    max_diff: float = 0.5,
    rooms_min: float = 1.0,
    rooms_max: float = 15.0,
) -> tuple[pd.Series, pd.Series]:
    """Recover half rooms and missing room counts from the room count stated in the text.

    The DSPRO1 scraper stored whole rooms (3.5 rounded to 4). If the description states a
    half-room count (x.5) within ``max_diff`` of the stored count, the text value replaces it;
    a missing count is filled from the text. Text values outside ``[rooms_min, rooms_max]`` or
    not a multiple of 0.5 are ignored, and larger disagreements keep the stored count.

    Args:
        rooms: Stored room counts (NaN = missing).
        text_rooms: Room counts extracted from the description, aligned to ``rooms`` by index.
        max_diff: Largest accepted distance between a stored count and a text half room.
        rooms_min: Smallest plausible room count.
        rooms_max: Largest plausible room count.

    Returns:
        ``(recovered, source)`` indexed like ``rooms``: float room counts and the source per
        row (``"table"``, ``"text_half_room"``, ``"text_missing"`` or ``"missing"``).
    """
    base = pd.to_numeric(rooms, errors="coerce").astype(float)
    text = pd.to_numeric(text_rooms.reindex(base.index), errors="coerce").astype(float)
    valid = text.between(rooms_min, rooms_max) & ((2 * text) % 1 == 0)
    half = valid & (text % 1 == 0.5) & base.notna() & (text - base).abs().between(0, max_diff)
    half &= text.ne(base)
    fill = valid & base.isna()
    recovered = base.mask(half | fill, text)
    source = pd.Series("table", index=base.index, dtype=object)
    source = source.mask(half, "text_half_room").mask(fill, "text_missing")
    source = source.mask(recovered.isna(), "missing")
    logger.info("recover_rooms_from_text: %s", source.value_counts().to_dict())
    return recovered, source


def cleaning_log(steps: list[tuple[str, int]]) -> pd.DataFrame:
    """Tabulate the row count after each cleaning step.

    Args:
        steps: ``(step name, rows after the step)`` in pipeline order; the first entry is the
            starting count.

    Returns:
        DataFrame with columns ``step``, ``n_rows``, ``removed`` (vs. previous step),
        ``removed_pct`` (of the previous step) and ``retained_pct`` (of the first step).

    Raises:
        ValueError: If a count is negative or not an integer.
    """
    columns = ["step", "n_rows", "removed", "removed_pct", "retained_pct"]
    if not steps:
        return pd.DataFrame(columns=columns)
    names = [str(name) for name, _ in steps]
    raw_counts = np.asarray([count for _, count in steps], dtype=float)
    if (raw_counts < 0).any() or not np.all(np.mod(raw_counts, 1) == 0):
        raise ValueError("Row counts must be non-negative integers")
    counts = raw_counts.astype(np.int64)
    previous = np.r_[counts[0], counts[:-1]]
    removed = previous - counts
    if (removed < 0).any():
        logger.warning("cleaning_log: row count increases at some step: %s", steps)
    with np.errstate(divide="ignore", invalid="ignore"):
        removed_pct = np.where(previous > 0, 100.0 * removed / previous, np.nan)
        retained_pct = np.where(counts[0] > 0, 100.0 * counts / counts[0], np.nan)
    return pd.DataFrame(
        {
            "step": names,
            "n_rows": counts,
            "removed": removed,
            "removed_pct": removed_pct,
            "retained_pct": retained_pct,
        },
        columns=columns,
    )
