"""Input validation and interval metrics for :mod:`rentml.conformal` (private helpers).

Import the public functions (:func:`coverage_report`, :func:`constant_band_for_coverage`,
:func:`interval_score`) from :mod:`rentml.conformal`.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

GLOBAL_CELL = "all"
RANK_TOL = 1e-9  # guards ceil() against float noise, e.g. (99 + 1) * (1 - 0.2) = 80.00000000000001

ArrayLike = np.ndarray | pd.Series | Sequence[float]


def as_array(values: ArrayLike, name: str) -> np.ndarray:
    """Flatten ``values`` to a float array and reject NaN (``name`` is used in the message)."""
    arr = np.asarray(values, dtype=float).ravel()
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    return arr


def check_alpha(alpha: float) -> None:
    """Raise ``ValueError`` unless ``0 < alpha < 1``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")


def same_length(**arrays: np.ndarray) -> int:
    """Return the common length of ``arrays`` or raise ``ValueError`` naming each length."""
    lengths = {name: arr.size for name, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Length mismatch: {lengths}")
    return next(iter(lengths.values()))


def coverage_report(
    y_chf: ArrayLike,
    lo_chf: ArrayLike,
    hi_chf: ArrayLike,
    groups: pd.Series | np.ndarray | Sequence[object] | None = None,
    *,
    include_overall: bool = True,
) -> pd.DataFrame:
    """Empirical coverage and width of intervals, overall and per group (CHF scale).

    Args:
        y_chf: Observed rents in CHF.
        lo_chf: Lower interval bounds in CHF.
        hi_chf: Upper interval bounds in CHF.
        groups: Optional group labels aligned row by row (Series, array or list, e.g. the
            output of ``assign_band``); a Series name becomes the index name.
        include_overall: Append an ``"all"`` row when ``groups`` is given.

    Returns:
        DataFrame indexed by group with ``n``, ``coverage``, ``below`` (share ``y < lo``),
        ``above`` (share ``y > hi``), ``mean_width`` and ``median_width``.

    Raises:
        ValueError: On NaN values, mismatched lengths or a group labelled ``"all"`` together
            with ``include_overall``.
    """
    y, lo, hi = as_array(y_chf, "y_chf"), as_array(lo_chf, "lo_chf"), as_array(hi_chf, "hi_chf")
    n = same_length(y=y, lo=lo, hi=hi)
    frame = pd.DataFrame(
        {"covered": (y >= lo) & (y <= hi), "below": y < lo, "above": y > hi, "width": hi - lo}
    )
    aggregations = {
        "n": ("covered", "size"),
        "coverage": ("covered", "mean"),
        "below": ("below", "mean"),
        "above": ("above", "mean"),
        "mean_width": ("width", "mean"),
        "median_width": ("width", "median"),
    }
    frame["group"] = GLOBAL_CELL
    overall = frame.groupby("group").agg(**aggregations)
    if groups is None:
        return overall
    labels = np.asarray(groups, dtype=object).ravel()
    if labels.size != n:
        raise ValueError(f"groups has {labels.size} rows, expected {n}")
    if include_overall and np.any(labels == GLOBAL_CELL):
        raise ValueError(f"A group is labelled {GLOBAL_CELL!r}, which is the overall row's label")
    frame["group"] = labels
    per_group = frame.groupby("group", sort=True, dropna=False).agg(**aggregations)
    per_group.index.name = getattr(groups, "name", None) or "group"
    if include_overall:
        per_group = pd.concat([per_group, overall.rename_axis(per_group.index.name)])
    return per_group


def constant_band_for_coverage(
    y_chf: ArrayLike, yhat_chf: ArrayLike, target_coverage: float
) -> float:
    """Half-width of the constant CHF band ``yhat +- h`` with at least the target coverage.

    Used as the RQ3 comparison: a DSPRO1-style constant band tuned to the same empirical coverage.

    Args:
        y_chf: Observed rents in CHF.
        yhat_chf: Point predictions in CHF.
        target_coverage: Desired empirical coverage in (0, 1].

    Returns:
        Smallest ``h`` such that the share of ``|y - yhat| <= h`` is at least ``target_coverage``.

    Raises:
        ValueError: On an invalid target, empty input, NaN values or mismatched lengths.
    """
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must lie in (0, 1], got {target_coverage}")
    y, yhat = as_array(y_chf, "y_chf"), as_array(yhat_chf, "yhat_chf")
    n = same_length(y=y, yhat=yhat)
    if n == 0:
        raise ValueError("Empty input")
    k = max(math.ceil(n * target_coverage - RANK_TOL), 1)
    return float(np.sort(np.abs(y - yhat))[k - 1])


def interval_score(y: ArrayLike, lo: ArrayLike, hi: ArrayLike, alpha: float) -> float:
    """Mean interval (Winkler) score of central ``(1 - alpha)`` intervals; lower is better.

    ``S = (hi - lo) + 2/alpha * (lo - y) * 1[y < lo] + 2/alpha * (y - hi) * 1[y > hi]``.

    Args:
        y: Observed values.
        lo: Lower bounds.
        hi: Upper bounds.
        alpha: Nominal miscoverage of the interval.

    Returns:
        Mean interval score (same unit as ``y``).

    Raises:
        ValueError: On invalid ``alpha``, NaN values or mismatched lengths.
    """
    check_alpha(alpha)
    obs, low, high = as_array(y, "y"), as_array(lo, "lo"), as_array(hi, "hi")
    same_length(y=obs, lo=low, hi=high)
    penalty_low = (2.0 / alpha) * np.clip(low - obs, 0.0, None)
    penalty_high = (2.0 / alpha) * np.clip(obs - high, 0.0, None)
    return float(np.mean(high - low + penalty_low + penalty_high))
