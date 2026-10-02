"""Pinball loss and quantile-grid utilities for :mod:`rentml.quantile` (private helpers).

Everything works on log-price quantile matrices of shape ``(n, L)`` whose columns belong to strictly
increasing levels. Import the public functions from :mod:`rentml.quantile`.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

PERCENTILE_CLAMP: tuple[float, float] = (0.01, 0.99)
"""Bounds for the market percentile outside the modelled quantile grid."""


def check_level(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"Quantile level must lie in (0, 1), got {alpha}")


def pinball_loss(
    y: np.ndarray | pd.Series, q: np.ndarray | pd.Series | float, alpha: float
) -> float:
    """Mean pinball (quantile) loss of predictions ``q`` at level ``alpha``.

    Args:
        y: Observed values.
        q: Predicted quantiles (same length as ``y`` or a scalar).
        alpha: Quantile level in (0, 1).

    Returns:
        Mean of ``max(alpha * r, (alpha - 1) * r)`` with ``r = y - q``.

    Raises:
        ValueError: If ``alpha`` is outside (0, 1), ``y`` is empty or shapes do not match.
    """
    check_level(alpha)
    y_arr = np.asarray(y, dtype=float).ravel()
    q_arr = np.asarray(q, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("pinball_loss needs at least one observation")
    if q_arr.size not in (1, y_arr.size):
        raise ValueError(f"Shape mismatch: y has {y_arr.size} values, q has {q_arr.size}")
    residual = y_arr - q_arr
    return float(np.mean(np.maximum(alpha * residual, (alpha - 1.0) * residual)))


def validate_levels(levels: Sequence[float]) -> np.ndarray:
    arr = np.asarray(levels, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("At least one quantile level is required")
    if np.any((arr <= 0) | (arr >= 1)):
        raise ValueError(f"Quantile levels must lie in (0, 1), got {arr.tolist()}")
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"Quantile levels must be strictly increasing, got {arr.tolist()}")
    return arr


def rearrange(q: np.ndarray) -> np.ndarray:
    """Sort quantile predictions along the last axis (Chernozhukov et al. rearrangement).

    Args:
        q: Quantile predictions, shape ``(n, L)`` or ``(L,)``.

    Returns:
        A sorted copy (NaN values end up last).
    """
    return np.sort(np.asarray(q, dtype=float), axis=-1)


def crossing_rate(q: np.ndarray) -> float:
    """Share of rows in which at least one pair of adjacent quantiles crosses.

    Args:
        q: Raw (unsorted) quantile predictions of shape ``(n, L)``.

    Returns:
        Fraction of rows with ``q[:, j+1] < q[:, j]`` for some ``j``.

    Raises:
        ValueError: If ``q`` is not 2-D or has no rows.
    """
    arr = np.asarray(q, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError(f"q must be a non-empty 2-D array, got shape {arr.shape}")
    if arr.shape[1] < 2:
        return 0.0
    return float(np.mean(np.any(np.diff(arr, axis=1) < 0, axis=1)))


def quantile_columns(q: np.ndarray, levels: Sequence[float], wanted: Sequence[float]) -> np.ndarray:
    """Select the columns of a quantile matrix that belong to given levels.

    Args:
        q: Quantile predictions of shape ``(n, L)``.
        levels: The ``L`` levels of ``q`` (e.g. ``model.levels_``).
        wanted: Levels to select (matched with ``np.isclose``).

    Returns:
        Array of shape ``(n, len(wanted))``.

    Raises:
        ValueError: If a wanted level is not in ``levels``.
    """
    grid = np.asarray(levels, dtype=float)
    idx = []
    for level in wanted:
        hits = np.flatnonzero(np.isclose(grid, level))
        if hits.size == 0:
            raise ValueError(f"Level {level} not in the quantile grid {grid.tolist()}")
        idx.append(int(hits[0]))
    return np.asarray(q, dtype=float)[:, idx]


def market_percentiles(
    q_log: np.ndarray,
    levels: Sequence[float],
    values_log: np.ndarray | pd.Series | Sequence[float],
    *,
    clamp: tuple[float, float] = PERCENTILE_CLAMP,
) -> np.ndarray:
    """Approximate CDF values of observed rents on each row's quantile grid (vectorised).

    Inside the grid the CDF is interpolated linearly between adjacent quantiles; outside it the
    first/last segment is extended linearly and the result is clamped to ``clamp``. Flat segments
    (tied quantiles) are handled as point masses.

    Args:
        q_log: Log-scale quantiles, shape ``(n, L)`` (rearranged internally), ``L >= 2``.
        levels: Strictly increasing levels of the columns of ``q_log``.
        values_log: One log-scale value per row (e.g. the log asking rent).
        clamp: Lower and upper bound for the returned fraction.

    Returns:
        Fractions in ``[clamp[0], clamp[1]]`` (0.5 = median of the market); NaN where the value or
        a quantile is NaN.

    Raises:
        ValueError: On mismatched shapes or fewer than two levels.
    """
    a = validate_levels(levels)
    q = rearrange(np.atleast_2d(np.asarray(q_log, dtype=float)))
    v = np.asarray(values_log, dtype=float).ravel()
    n_rows, n_levels = q.shape
    if n_levels != a.size or n_levels < 2:
        raise ValueError(f"q_log has {n_levels} columns for {a.size} levels (need >= 2)")
    if v.size != n_rows:
        raise ValueError(f"{v.size} values for {n_rows} quantile rows")
    rows = np.arange(n_rows)
    n_below = np.sum(q <= v[:, None], axis=1)
    seg = np.clip(n_below - 1, 0, n_levels - 2)
    q0, q1 = q[rows, seg], q[rows, seg + 1]
    a0, a1 = a[seg], a[seg + 1]
    dq = q1 - q0
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = a0 + (a1 - a0) * (v - q0) / dq
    flat = ~(dq > 0)
    pct = np.where(flat & (n_below == 0), 0.0, pct)
    upper = np.where(v > q[:, -1], 1.0, a[-1])
    pct = np.where(flat & (n_below == n_levels), upper, pct)
    invalid = np.isnan(v) | np.any(np.isnan(q), axis=1)
    pct = np.where(invalid, np.nan, pct)
    return np.clip(pct, clamp[0], clamp[1])


def market_percentile(q_row_log: np.ndarray, levels: Sequence[float], value_log: float) -> float:
    """Approximate market percentile of one rent on one row's quantile grid.

    Args:
        q_row_log: Log-scale quantiles of one listing, shape ``(L,)``.
        levels: Strictly increasing levels of ``q_row_log``.
        value_log: Log-scale value, e.g. ``np.log(asking_rent)``.

    Returns:
        Fraction in [0.01, 0.99] (multiply by 100 for a percentile).

    Raises:
        ValueError: On mismatched shapes.
    """
    row = np.asarray(q_row_log, dtype=float).reshape(1, -1)
    return float(market_percentiles(row, levels, np.array([value_log], dtype=float))[0])


def quantile_calibration_table(
    y_log: np.ndarray | pd.Series, q_log: np.ndarray, levels: Sequence[float]
) -> pd.DataFrame:
    """Empirical fraction of observations at or below each predicted quantile.

    Args:
        y_log: Observed log prices, shape ``(n,)``.
        q_log: Predicted log quantiles, shape ``(n, L)``.
        levels: The ``L`` quantile levels.

    Returns:
        DataFrame with columns ``level``, ``empirical`` (share of ``y <= q``), ``deviation``
        (empirical - level), ``pinball`` (mean pinball loss) and ``n``.

    Raises:
        ValueError: On mismatched shapes.
    """
    a = validate_levels(levels)
    y = np.asarray(y_log, dtype=float).ravel()
    q = np.asarray(q_log, dtype=float)
    if q.ndim != 2 or q.shape != (y.size, a.size):
        raise ValueError(f"q_log shape {q.shape} does not match ({y.size}, {a.size})")
    empirical = np.mean(y[:, None] <= q, axis=0)
    pinball = [pinball_loss(y, q[:, j], float(level)) for j, level in enumerate(a)]
    return pd.DataFrame(
        {
            "level": a,
            "empirical": empirical,
            "deviation": empirical - a,
            "pinball": pinball,
            "n": y.size,
        }
    )
