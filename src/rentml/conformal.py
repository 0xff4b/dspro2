"""Split-conformal calibration of prediction intervals (CQR, normalised, Mondrian).

Scores and intervals are on the log-price scale unless a function says CHF; fit on the calib split.

* :class:`CQRCalibrator` - conformalized quantile regression (Romano et al., 2019). Symmetric:
  score ``E = max(q_lo - y, y - q_hi)`` and interval ``[q_lo - qhat, q_hi + qhat]``; this controls
  the *total* miscoverage only. Asymmetric (``symmetric=False``): ``qhat_lo`` from ``q_lo - y`` and
  ``qhat_hi`` from ``y - q_hi``, each at level ``alpha / 2``, so the shares below and above the
  interval are controlled separately (needed when "below" and "above" mean different things, as in
  the Fair-Rent verdict). It needs about ``2 / alpha`` calibration rows per cell.
* :class:`NormalizedConformal` - ``|y - mu| / sigma`` score for a Gaussian predictive distribution
  (e.g. GPBoost), interval ``mu +- qhat * sigma``.
* :class:`MondrianCQR` - one correction per group cell (language region x predicted-rent band) with
  a hierarchical fallback to the parent cell when a cell has fewer than ``min_group_size`` rows.
  Bands come from :func:`fit_band_cutpoints` on *predicted* medians, never on the target.
"""

import logging
import math
from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

# Validation helpers and interval metrics live in a private module; metrics re-exported as API.
from rentml._interval_metrics import (
    GLOBAL_CELL,
    RANK_TOL,
    ArrayLike,
    as_array,
    check_alpha,
    same_length,
)
from rentml._interval_metrics import constant_band_for_coverage as constant_band_for_coverage
from rentml._interval_metrics import coverage_report as coverage_report
from rentml._interval_metrics import interval_score as interval_score
from rentml.config import INTERVAL_COVERAGE, MIN_MONDRIAN_GROUP

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 1.0 - INTERVAL_COVERAGE
DEFAULT_HIERARCHY: tuple[tuple[str, ...], ...] = (("lang_region", "band"), ("lang_region",), ())
NA_KEY = "<NA>"
_CELL_COLUMNS = ["level", "depth", "cell", "n", "qhat", "qhat_lo", "qhat_hi", "calibrated"]
_CellRow = tuple[str, int, str, int, float, float, float, bool]
_QHAT_NAN = (math.nan, math.nan, math.nan)


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Finite-sample conformal quantile of calibration scores.

    Returns the ``k``-th smallest score with ``k = ceil((n + 1) * (1 - alpha))`` (1-based), which
    guarantees marginal coverage ``>= 1 - alpha`` for exchangeable data.

    Args:
        scores: Conformity scores of the calibration rows.
        alpha: Miscoverage level in (0, 1).

    Returns:
        The conformal quantile, or ``+inf`` if ``k > n`` (too few calibration rows).

    Raises:
        ValueError: If ``alpha`` is outside (0, 1) or the scores contain NaN.
    """
    check_alpha(alpha)
    arr = as_array(scores, "scores")
    k = math.ceil((arr.size + 1) * (1.0 - alpha) - RANK_TOL)
    if k > arr.size:
        logger.warning("Only %d calibration scores for alpha=%.3f: qhat = +inf", arr.size, alpha)
        return math.inf
    return float(np.partition(arr, k - 1)[k - 1])


def cqr_scores(q_lo: ArrayLike, q_hi: ArrayLike, y: ArrayLike) -> np.ndarray:
    """CQR conformity scores ``max(q_lo - y, y - q_hi)`` (negative inside the interval).

    Args:
        q_lo: Lower raw quantile predictions.
        q_hi: Upper raw quantile predictions.
        y: Observed values (same scale as the quantiles).

    Returns:
        Score per row.

    Raises:
        ValueError: On NaN values or mismatched lengths.
    """
    lo, hi, obs = as_array(q_lo, "q_lo"), as_array(q_hi, "q_hi"), as_array(y, "y")
    same_length(q_lo=lo, q_hi=hi, y=obs)
    return np.maximum(lo - obs, obs - hi)


def _side_qhats(
    lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float, symmetric: bool
) -> tuple[float, float, float]:
    """Return ``(qhat, qhat_lo, qhat_hi)``; ``qhat`` is NaN in the asymmetric case."""
    if symmetric:
        qhat = conformal_quantile(np.maximum(lo - y, y - hi), alpha)
        return qhat, qhat, qhat
    return math.nan, conformal_quantile(lo - y, alpha / 2), conformal_quantile(y - hi, alpha / 2)


def _widen(
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    qhat_lo: np.ndarray | float,
    qhat_hi: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = q_lo - qhat_lo, q_hi + qhat_hi
    # A negative qhat shrinks intervals and can invert narrow ones; collapse those to the midpoint.
    with np.errstate(invalid="ignore"):  # inf - inf; fmin/fmax below ignore the NaN midpoint
        mid = (lo + hi) / 2
    return np.fmin(lo, mid), np.fmax(hi, mid)


def _check_levels_arg(interval_levels: tuple[float, float] | None) -> tuple[float, float] | None:
    if interval_levels is None:
        return None
    lo, hi = (float(level) for level in interval_levels)
    if not 0.0 < lo < hi < 1.0:
        raise ValueError(f"interval_levels must satisfy 0 < lo < hi < 1, got {interval_levels}")
    return lo, hi


class CQRCalibrator:
    """Marginal conformalized quantile regression (symmetric or asymmetric, see module docs).

    Args:
        alpha: Miscoverage level (default ``1 - INTERVAL_COVERAGE`` = 0.2 for an 80 % interval).
        symmetric: One shared ``qhat`` (``True``) or separate ``qhat_lo`` / ``qhat_hi`` at level
            ``alpha / 2`` each (``False``; use it for the Fair-Rent verdict).

    Attributes:
        qhat_: Symmetric additive correction on the log scale (NaN if ``symmetric=False``).
        qhat_lo_, qhat_hi_: Corrections subtracted from ``q_lo`` / added to ``q_hi`` (may be < 0).
        n_calib_: Number of calibration rows.
        interval_levels_: Raw quantile levels ``(lo, hi)`` given to :meth:`fit`, or ``None``.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA, *, symmetric: bool = True) -> None:
        self.alpha = alpha
        self.symmetric = symmetric

    def fit(
        self,
        q_lo: ArrayLike,
        q_hi: ArrayLike,
        y: ArrayLike,
        *,
        interval_levels: tuple[float, float] | None = None,
    ) -> Self:
        """Compute the corrections on the calibration set.

        Args:
            q_lo: Lower raw quantile predictions (e.g. level ``alpha / 2``).
            q_hi: Upper raw quantile predictions (e.g. level ``1 - alpha / 2``).
            y: Observed log prices.
            interval_levels: Quantile levels of ``q_lo`` / ``q_hi``, recorded so that
                ``rentml.rentcheck`` selects the same columns at prediction time.

        Returns:
            The fitted calibrator.

        Raises:
            ValueError: On invalid ``alpha`` or levels, NaN values or mismatched lengths.
        """
        lo, hi, obs = as_array(q_lo, "q_lo"), as_array(q_hi, "q_hi"), as_array(y, "y")
        self.n_calib_ = same_length(q_lo=lo, q_hi=hi, y=obs)
        self.interval_levels_ = _check_levels_arg(interval_levels)
        self.qhat_, self.qhat_lo_, self.qhat_hi_ = _side_qhats(
            lo, hi, obs, self.alpha, self.symmetric
        )
        logger.info(
            "CQR: qhat_lo=%.4f, qhat_hi=%.4f from %d rows", self.qhat_lo_, self.qhat_hi_, obs.size
        )
        return self

    def predict(self, q_lo: ArrayLike, q_hi: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the calibrated interval ``[q_lo - qhat_lo, q_hi + qhat_hi]``.

        Args:
            q_lo: Lower raw quantile predictions.
            q_hi: Upper raw quantile predictions.

        Returns:
            Tuple ``(lo, hi)``.
        """
        lo, hi = as_array(q_lo, "q_lo"), as_array(q_hi, "q_hi")
        same_length(q_lo=lo, q_hi=hi)
        return _widen(lo, hi, self.qhat_lo_, self.qhat_hi_)


class NormalizedConformal:
    """Split conformal with the sigma-normalised score ``|y - mu| / sigma``.

    Args:
        alpha: Miscoverage level (default 0.2).

    Attributes:
        qhat_: Multiplier for ``sigma`` (conformal analogue of the normal ``z``).
        n_calib_: Number of calibration rows.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha

    @staticmethod
    def _check_sigma(sigma: np.ndarray) -> None:
        if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
            raise ValueError("sigma must be finite and strictly positive")

    def fit(self, mu: ArrayLike, sigma: ArrayLike, y: ArrayLike) -> Self:
        """Compute ``qhat`` of the normalised residuals.

        Args:
            mu: Predictive means (log scale).
            sigma: Predictive standard deviations (> 0).
            y: Observed log prices.

        Returns:
            The fitted calibrator.

        Raises:
            ValueError: On non-positive sigma, NaN values or mismatched lengths.
        """
        m, s, obs = as_array(mu, "mu"), as_array(sigma, "sigma"), as_array(y, "y")
        same_length(mu=m, sigma=s, y=obs)
        self._check_sigma(s)
        self.qhat_ = conformal_quantile(np.abs(obs - m) / s, self.alpha)
        self.n_calib_ = obs.size
        return self

    def predict(self, mu: ArrayLike, sigma: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return the interval ``mu +- qhat * sigma``.

        Args:
            mu: Predictive means.
            sigma: Predictive standard deviations (> 0).

        Returns:
            Tuple ``(lo, hi)``.

        Raises:
            ValueError: On non-positive sigma or mismatched lengths.
        """
        m, s = as_array(mu, "mu"), as_array(sigma, "sigma")
        same_length(mu=m, sigma=s)
        self._check_sigma(s)
        return m - self.qhat_ * s, m + self.qhat_ * s


def fit_band_cutpoints(median_pred: ArrayLike, n_bands: int = 4) -> np.ndarray:
    """Quantile cutpoints of *predicted* medians that define the predicted-rent bands.

    Fit on the calibration set's median predictions; the target is never used, so bands can be
    assigned at prediction time.

    Args:
        median_pred: Predicted median (log or CHF; use the same scale in :func:`assign_band`).
        n_bands: Number of bands (4 = quartiles).

    Returns:
        ``n_bands - 1`` increasing cutpoints.

    Raises:
        ValueError: If ``n_bands < 1`` or the input is empty or contains NaN.
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    arr = as_array(median_pred, "median_pred")
    if arr.size == 0:
        raise ValueError("median_pred is empty")
    return np.quantile(arr, np.arange(1, n_bands) / n_bands)


def assign_band(median_pred: ArrayLike, cutpoints: ArrayLike) -> np.ndarray:
    """Label predictions with bands ``"B1"`` (cheapest) to ``"B{len(cutpoints)+1}"``.

    Args:
        median_pred: Predicted medians on the scale of the cutpoints.
        cutpoints: Output of :func:`fit_band_cutpoints`.

    Returns:
        Object array of band labels (a value equal to a cutpoint goes to the upper band).

    Raises:
        ValueError: If the predictions contain NaN.
    """
    arr = as_array(median_pred, "median_pred")
    idx = np.searchsorted(np.asarray(cutpoints, dtype=float).ravel(), arr, side="right")
    return np.array([f"B{i + 1}" for i in idx], dtype=object)


def _key_values(values: pd.Series) -> pd.Series:
    """Stringify cell values so that 1, 1.0 and "1" share a key and every NA maps to NA_KEY."""
    obj = values.astype(object)
    out = obj.map(str)
    if is_bool_dtype(values):
        return out.where(values.notna(), NA_KEY)
    num = pd.to_numeric(obj, errors="coerce").astype(float)  # text labels become NaN
    numeric, integral = num.notna(), num.notna() & np.isfinite(num) & (num % 1 == 0)
    out[numeric] = num[numeric].astype(str)
    out[integral] = num[integral].astype(np.int64).astype(str)
    return out.where(values.notna(), NA_KEY)


def _cell_keys(groups: pd.DataFrame, cols: tuple[str, ...]) -> pd.Series:
    if not cols:
        return pd.Series(GLOBAL_CELL, index=groups.index, dtype=object)
    key = f"{cols[0]}=" + _key_values(groups[cols[0]])
    for col in cols[1:]:
        key = key + f"|{col}=" + _key_values(groups[col])
    return key.astype(object)


class MondrianCQR:
    """Group-conditional (Mondrian) CQR with hierarchical fallback.

    Each hierarchy level (finest first) defines cells by a tuple of group columns. A cell gets
    its own correction when it has at least ``min_group_size`` calibration rows; prediction uses
    the finest level at which the row's cell is calibrated. The global level ``()`` must be last;
    it is appended if missing and always calibrated (with a warning below ``min_group_size``).
    Cell values are normalised (``1``, ``1.0`` and ``"1"`` share a key; NA becomes ``"<NA>"``).

    Args:
        alpha: Miscoverage level per cell (default 0.2).
        min_group_size: Minimum calibration rows for a cell to be calibrated.
        hierarchy: Tuples of column names, finest first.
        symmetric: Shared ``qhat`` or separate lower/upper corrections (see :class:`CQRCalibrator`;
            ``False`` is recommended for the Fair-Rent verdict).

    Attributes:
        qhat_: Cell label -> symmetric ``qhat`` for calibrated cells (NaN if asymmetric).
        qhat_lo_, qhat_hi_: Cell label -> lower / upper correction for calibrated cells.
        cells_: DataFrame of all cells (see :meth:`cell_table`).
        interval_levels_: Raw quantile levels ``(lo, hi)`` given to :meth:`fit`, or ``None``.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        min_group_size: int = MIN_MONDRIAN_GROUP,
        hierarchy: Sequence[Sequence[str]] = DEFAULT_HIERARCHY,
        *,
        symmetric: bool = True,
    ) -> None:
        self.alpha = alpha
        self.min_group_size = min_group_size
        self.hierarchy = hierarchy
        self.symmetric = symmetric

    def _levels(self) -> list[tuple[str, ...]]:
        levels = [tuple(level) for level in self.hierarchy]
        if len(set(levels)) != len(levels):
            raise ValueError(f"Duplicate hierarchy levels: {levels}")
        if () in levels[:-1]:
            raise ValueError(f"The global level () must be the last hierarchy level: {levels}")
        if not levels or levels[-1] != ():
            levels.append(())
        return levels

    @staticmethod
    def _check_groups(groups: pd.DataFrame, levels: list[tuple[str, ...]], n: int) -> None:
        needed = sorted({col for level in levels for col in level})
        missing = [col for col in needed if col not in groups.columns]
        if missing:
            raise ValueError(f"groups is missing columns {missing}")
        if len(groups) != n:
            raise ValueError(f"groups has {len(groups)} rows, expected {n}")

    def fit(
        self,
        q_lo: ArrayLike,
        q_hi: ArrayLike,
        y: ArrayLike,
        groups: pd.DataFrame,
        *,
        interval_levels: tuple[float, float] | None = None,
    ) -> Self:
        """Compute the corrections per sufficiently large cell at every hierarchy level.

        Args:
            q_lo: Lower raw quantile predictions of the calibration rows.
            q_hi: Upper raw quantile predictions.
            y: Observed log prices.
            groups: Group columns, aligned row by row with ``y`` (e.g. ``lang_region``, ``band``).
            interval_levels: Quantile levels of ``q_lo`` / ``q_hi`` (see :class:`CQRCalibrator`).

        Returns:
            The fitted calibrator.

        Raises:
            ValueError: On invalid ``alpha``/hierarchy/levels, missing columns, NaN values,
                mismatched lengths or empty calibration data.
        """
        check_alpha(self.alpha)
        lo, hi, obs = as_array(q_lo, "q_lo"), as_array(q_hi, "q_hi"), as_array(y, "y")
        n = same_length(q_lo=lo, q_hi=hi, y=obs)
        if n == 0:
            raise ValueError("MondrianCQR.fit needs at least one calibration row")
        levels = self._levels()
        self._check_groups(groups, levels, n)
        rows: list[_CellRow] = []
        for depth, cols in enumerate(levels):
            keys = _cell_keys(groups, cols).to_numpy()
            is_root = depth == len(levels) - 1
            rows.extend(self._fit_level((lo, hi, obs), keys, cols, depth, is_root))
        self.levels_ = levels
        self.interval_levels_ = _check_levels_arg(interval_levels)
        self.cells_ = pd.DataFrame(rows, columns=_CELL_COLUMNS)
        calibrated = self.cells_[self.cells_["calibrated"]]
        self.qhat_, self.qhat_lo_, self.qhat_hi_ = (
            dict(zip(calibrated["cell"], calibrated[col].astype(float), strict=True))
            for col in ("qhat", "qhat_lo", "qhat_hi")
        )
        logger.info("Mondrian CQR: %d calibrated cells", len(self.qhat_lo_))
        return self

    def _fit_level(
        self,
        data: tuple[np.ndarray, np.ndarray, np.ndarray],
        keys: np.ndarray,
        cols: tuple[str, ...],
        depth: int,
        is_root: bool,
    ) -> list[_CellRow]:
        level = " x ".join(cols) or "global"
        rows: list[_CellRow] = []
        for cell, positions in pd.Series(np.arange(keys.size)).groupby(keys):
            idx = positions.to_numpy()
            calibrated = idx.size >= self.min_group_size or is_root
            if idx.size < self.min_group_size and is_root:
                logger.warning("Global cell has only %d rows (< %d)", idx.size, self.min_group_size)
            lo, hi, obs = (arr[idx] for arr in data)
            qhats = (
                _side_qhats(lo, hi, obs, self.alpha, self.symmetric) if calibrated else _QHAT_NAN
            )
            rows.append((level, depth, str(cell), idx.size, *qhats, calibrated))
        return rows

    def predict(
        self, q_lo: ArrayLike, q_hi: ArrayLike, groups: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        """Calibrated interval using the finest calibrated cell of each row.

        Args:
            q_lo: Lower raw quantile predictions.
            q_hi: Upper raw quantile predictions.
            groups: Group columns aligned row by row with the predictions.

        Returns:
            Tuple ``(lo, hi, used_cell)``; ``used_cell`` is a string Series indexed like ``groups``.

        Raises:
            ValueError: On missing columns or mismatched lengths.
        """
        lo_raw, hi_raw = as_array(q_lo, "q_lo"), as_array(q_hi, "q_hi")
        n = same_length(q_lo=lo_raw, q_hi=hi_raw)
        self._check_groups(groups, self.levels_, n)
        qhat_lo, qhat_hi = np.full(n, np.nan), np.full(n, np.nan)
        used = np.full(n, None, dtype=object)
        shares: dict[str, float] = {}
        for cols in self.levels_:
            keys = _cell_keys(groups, cols)
            found_lo = keys.map(self.qhat_lo_).to_numpy(dtype=float)
            take = np.isnan(qhat_lo) & ~np.isnan(found_lo)
            qhat_lo[take] = found_lo[take]
            qhat_hi[take] = keys.map(self.qhat_hi_).to_numpy(dtype=float)[take]
            used[take] = keys.to_numpy()[take]
            shares[" x ".join(cols) or "global"] = float(take.mean()) if n else 0.0
        logger.info("Mondrian CQR rows served per hierarchy level: %s", shares)
        lo, hi = _widen(lo_raw, hi_raw, qhat_lo, qhat_hi)
        return lo, hi, pd.Series(used, index=groups.index, name="cell")

    def cell_table(self) -> pd.DataFrame:
        """All cells seen in calibration with size, corrections and calibration status.

        Returns:
            DataFrame with columns ``level``, ``depth``, ``cell``, ``n``, ``qhat``, ``qhat_lo``,
            ``qhat_hi`` and ``calibrated`` (corrections are NaN below ``min_group_size``;
            ``qhat`` is NaN for asymmetric calibration).
        """
        return self.cells_.copy()
