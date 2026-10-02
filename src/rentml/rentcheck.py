"""Fair-Rent Check and landlord what-if on top of a fitted model bundle.

A :class:`RentModelBundle` holds the models the app needs (all duck-typed, log-price scale):

* ``point_model.predict(X) -> (n,)`` log price (expected rent = ``exp`` of it),
* ``quantile_model.predict(X) -> (n, L)`` log quantiles with ``quantile_model.levels_``
  (a :class:`rentml.quantile.MonotoneQuantileLGBM`),
* ``calibrator``: :class:`rentml.conformal.MondrianCQR` (``predict(q_lo, q_hi, groups)``),
  :class:`rentml.conformal.CQRCalibrator` (``predict(q_lo, q_hi)``) or ``None`` (raw quantiles).
  Fit it with ``symmetric=False`` so that "below" and "above" each occur at rate ``alpha / 2``.

The calibrated interval uses the raw quantile levels recorded at calibration time
(``calibrator.fit(..., interval_levels=(0.1, 0.9))`` or ``metadata["interval_levels"]``); only if
neither exists are they derived from the nominal coverage ``c = 1 - calibrator.alpha`` as
``(1 - c) / 2`` and ``(1 + c) / 2`` (with a warning). The listing band is the *uncalibrated*
25th-75th percentile. Predicted-rent bands for Mondrian cells are assigned from
``band_cutpoints`` on the log median prediction (quantile level 0.5 by default, the point model if
``metadata["band_source"] == "point"``), exactly as in calibration - the target is never used.
"""

import inspect
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Self

import joblib
import numpy as np
import pandas as pd

from rentml.config import INTERVAL_COVERAGE
from rentml.conformal import MondrianCQR, assign_band
from rentml.quantile import market_percentile, quantile_columns, rearrange

logger = logging.getLogger(__name__)

BAND_LEVELS: tuple[float, float] = (0.25, 0.75)
BASELINE_LABEL = "baseline"
DERIVED_COLS: tuple[str, ...] = ("area_per_room", "rooms_is_integer")
"""Columns :func:`what_if` drops before its ``transform`` so that they are recomputed."""


class PointModel(Protocol):
    """Anything with ``predict(X) -> log price``."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict log prices."""
        ...


class QuantileModel(Protocol):
    """Anything with ``levels_`` and ``predict(X) -> (n, L)`` log quantiles."""

    @property
    def levels_(self) -> Sequence[float] | np.ndarray:
        """Quantile levels of the prediction columns."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict log quantiles."""
        ...


class GroupCalibrator(Protocol):
    """Mondrian-style calibrator (``MondrianCQR``)."""

    def predict(
        self, q_lo: np.ndarray, q_hi: np.ndarray, groups: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        """Return calibrated bounds and the used cell."""
        ...


class MarginalCalibrator(Protocol):
    """Marginal calibrator (``CQRCalibrator``)."""

    def predict(self, q_lo: np.ndarray, q_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated bounds."""
        ...


def verdict(asking: float, lo: float, hi: float) -> str:
    """Classify an asking rent against a calibrated interval.

    Args:
        asking: Asking rent in CHF.
        lo: Lower interval bound in CHF.
        hi: Upper interval bound in CHF.

    Returns:
        ``"below"`` if ``asking < lo``, ``"above"`` if ``asking > hi``, else ``"within"``.

    Raises:
        ValueError: If a value is NaN or ``lo > hi``.
    """
    if any(math.isnan(v) for v in (asking, lo, hi)):
        raise ValueError("verdict needs non-NaN asking, lo and hi")
    if lo > hi:
        raise ValueError(f"Invalid interval: lo={lo} > hi={hi}")
    if asking < lo:
        return "below"
    if asking > hi:
        return "above"
    return "within"


def _chf(value: float) -> str:
    return f"{value:,.0f}".replace(",", "'")


@dataclass
class RentCheckResult:
    """Outcome of the Fair-Rent Check for one apartment (CHF scale).

    Attributes:
        expected_chf: Point estimate ``exp(point_model)``.
        lo_chf: Lower bound of the calibrated interval.
        hi_chf: Upper bound of the calibrated interval.
        band25_chf: 25th percentile (uncalibrated listing band).
        band75_chf: 75th percentile (uncalibrated listing band).
        market_percentile: Approximate CDF value of the asking rent in [0.01, 0.99].
        verdict: ``"below"``, ``"within"`` or ``"above"`` the calibrated interval.
        coverage: Nominal coverage of the interval (e.g. 0.8).
        drivers: Optional SHAP drivers (e.g. ``explain.top_drivers``).
        asking_chf: The asking rent that was checked.
        cell: Calibration cell used (Mondrian) or ``None``.
    """

    expected_chf: float
    lo_chf: float
    hi_chf: float
    band25_chf: float
    band75_chf: float
    market_percentile: float
    verdict: str
    coverage: float
    drivers: pd.DataFrame | None
    asking_chf: float = math.nan
    cell: str | None = None

    def summary(self) -> str:
        """One-line, human-readable result (for notebooks and the app)."""
        return (
            f"Asking CHF {_chf(self.asking_chf)} is {self.verdict} the expected range: "
            f"likely between CHF {_chf(self.lo_chf)} and {_chf(self.hi_chf)} "
            f"({self.coverage:.0%} interval), estimate CHF {_chf(self.expected_chf)}, "
            f"approx. market percentile {100 * self.market_percentile:.0f}."
        )


@dataclass
class RentModelBundle:
    """Everything needed to serve estimates, intervals and the rent check.

    Attributes:
        feature_cols: Model input columns (order used for prediction).
        point_model: Log-price point model.
        quantile_model: Log-scale quantile model with ``levels_``.
        calibrator: ``MondrianCQR``, ``CQRCalibrator`` or ``None`` (uncalibrated quantiles).
        band_cutpoints: Log-scale cutpoints for the predicted-rent band (``None`` = no band).
        metadata: Free-form info (training date, metrics, ``band_source``, ``coverage``).
        group_cols: Columns of the groups frame passed to the calibrator (e.g. ``["lang_region"]``).
    """

    feature_cols: list[str]
    point_model: PointModel
    quantile_model: QuantileModel
    calibrator: GroupCalibrator | MarginalCalibrator | None
    band_cutpoints: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    group_cols: list[str] = field(default_factory=list)

    def save(self, path: Path) -> Path:
        """Persist the bundle with joblib (parent directories are created).

        Args:
            path: Target file, e.g. ``models/rent_bundle.joblib``.

        Returns:
            The written path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        logger.info("Saved rent model bundle to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a bundle written by :meth:`save` (pickle: only load trusted files).

        Args:
            path: Bundle file.

        Returns:
            The bundle.

        Raises:
            TypeError: If the file does not contain a ``RentModelBundle``.
        """
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} contains {type(obj).__name__}, not {cls.__name__}")
        return obj


@dataclass
class _LogPrediction:
    mu: np.ndarray
    q: np.ndarray
    levels: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    band_q: np.ndarray
    q50: np.ndarray
    coverage: float
    band: np.ndarray | None
    cell: pd.Series | None


def _nominal_coverage(bundle: RentModelBundle) -> float:
    alpha = getattr(bundle.calibrator, "alpha", None)
    if isinstance(alpha, float | int):
        return 1.0 - float(alpha)
    coverage = bundle.metadata.get("coverage", INTERVAL_COVERAGE)
    return float(coverage) if isinstance(coverage, float | int) else INTERVAL_COVERAGE


def _is_default_index(index: pd.Index) -> bool:
    return isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1


def _align_groups(groups: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    if len(groups) != len(index):
        raise ValueError(f"groups has {len(groups)} rows, X has {len(index)}")
    if groups.index.equals(index):
        return groups
    unique = index.is_unique and groups.index.is_unique
    if unique and set(groups.index) == set(index):
        return groups.reindex(index)
    if _is_default_index(groups.index):  # a freshly built frame: align row by row
        return groups.set_axis(index)
    shared = len(set(groups.index) & set(index))
    raise ValueError(
        f"groups and X index labels differ ({shared} of {len(index)} shared); pass groups with "
        "the same labels as X or reset its index (groups.reset_index(drop=True)) to align by row"
    )


def _calibrator_groups(
    bundle: RentModelBundle, groups: pd.DataFrame | None, index: pd.Index, median: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray | None]:
    frame = pd.DataFrame(index=index)
    if bundle.group_cols:
        if groups is None:
            raise ValueError(f"groups with columns {bundle.group_cols} is required")
        missing = [col for col in bundle.group_cols if col not in groups.columns]
        if missing:
            raise ValueError(f"groups is missing columns {missing}")
        frame = _align_groups(groups, index).loc[:, bundle.group_cols].copy()
    band = None
    if bundle.band_cutpoints is not None:
        band = assign_band(median, bundle.band_cutpoints)
        frame["band"] = band
    return frame, band


def _calibrate(
    bundle: RentModelBundle, q_lo: np.ndarray, q_hi: np.ndarray, groups: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.Series | None]:
    calibrator = bundle.calibrator
    if calibrator is None:
        logger.debug("No calibrator in bundle: using raw quantile interval")
        return q_lo, q_hi, None
    takes_groups = "groups" in inspect.signature(calibrator.predict).parameters
    if isinstance(calibrator, MondrianCQR) or takes_groups:
        out = calibrator.predict(q_lo, q_hi, groups)
    else:
        out = calibrator.predict(q_lo, q_hi)
    cell = pd.Series(np.asarray(out[2], dtype=object), index=groups.index) if len(out) > 2 else None
    return np.asarray(out[0], dtype=float), np.asarray(out[1], dtype=float), cell


def _interval_levels(bundle: RentModelBundle, coverage: float) -> tuple[float, float]:
    recorded = getattr(bundle.calibrator, "interval_levels_", None)
    meta = bundle.metadata.get("interval_levels")
    levels = [tuple(float(v) for v in lv) for lv in (recorded, meta) if lv is not None]
    if len(levels) == 2 and not np.allclose(levels[0], levels[1]):
        raise ValueError(f"calibrator.interval_levels_ {recorded} != metadata {meta}")
    if levels:
        return levels[0][0], levels[0][1]
    tail = (1.0 - coverage) / 2.0
    if bundle.calibrator is not None:  # fit calibrators with interval_levels=... to avoid this
        logger.warning("No recorded interval levels: using %.3f/%.3f from alpha", tail, 1 - tail)
    return tail, 1.0 - tail


def _model_input(model: object, features: pd.DataFrame) -> pd.DataFrame:
    names = getattr(model, "feature_names_in_", None)  # LightGBM ignores column order itself
    if names is None or list(names) == list(features.columns):
        return features
    names = list(names)
    return features.loc[:, names] if set(names) <= set(features.columns) else features


def _predict_log(
    bundle: RentModelBundle, X: pd.DataFrame, groups: pd.DataFrame | None
) -> _LogPrediction:
    missing = [col for col in bundle.feature_cols if col not in X.columns]
    if missing:
        raise ValueError(f"X is missing feature columns {missing}")
    features = X.loc[:, bundle.feature_cols]
    mu = bundle.point_model.predict(_model_input(bundle.point_model, features))
    mu = np.asarray(mu, dtype=float).ravel()
    q = rearrange(np.asarray(bundle.quantile_model.predict(features), dtype=float))
    levels = np.asarray(bundle.quantile_model.levels_, dtype=float)
    if q.shape != (len(X), levels.size) or mu.size != len(X):
        raise ValueError(f"Model outputs {mu.shape}, {q.shape} do not match {len(X)} rows")
    coverage = _nominal_coverage(bundle)
    q_lo, q_hi = quantile_columns(q, levels, _interval_levels(bundle, coverage)).T
    band_q = quantile_columns(q, levels, BAND_LEVELS)
    q50 = quantile_columns(q, levels, (0.5,))[:, 0]
    median = mu if bundle.metadata.get("band_source") == "point" else q50
    cal_groups, band = _calibrator_groups(bundle, groups, X.index, median)
    lo, hi, cell = _calibrate(bundle, q_lo, q_hi, cal_groups)
    return _LogPrediction(mu, q, levels, lo, hi, band_q, q50, coverage, band, cell)


def predict_frame(
    bundle: RentModelBundle, X: pd.DataFrame, groups: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Expected rent, calibrated interval and listing band for many apartments.

    Args:
        bundle: Fitted model bundle.
        X: Features (at least ``bundle.feature_cols``).
        groups: Group columns ``bundle.group_cols`` aligned with ``X`` by index labels (same
            labels in any order) or, if ``groups`` has a default ``RangeIndex``, row by row; may
            be ``None`` if the bundle has no group columns.

    Returns:
        DataFrame indexed like ``X`` with ``expected``, ``lo``, ``hi``, ``q25``, ``q75`` and
        ``q50`` in CHF, plus ``band`` and ``cell`` when available.

    Raises:
        ValueError: On missing columns, missing or conflicting interval levels, misaligned
            ``groups`` or wrong model output shapes.
    """
    pred = _predict_log(bundle, X, groups)
    out = pd.DataFrame(
        {
            "expected": np.exp(pred.mu),
            "lo": np.exp(pred.lo),
            "hi": np.exp(pred.hi),
            "q25": np.exp(pred.band_q[:, 0]),
            "q75": np.exp(pred.band_q[:, 1]),
            "q50": np.exp(pred.q50),
        },
        index=X.index,
    )
    if pred.band is not None:
        out["band"] = pred.band
    if pred.cell is not None:
        out["cell"] = pred.cell.to_numpy()
    return out


def _check_single_row(frame: pd.DataFrame, name: str) -> None:
    if len(frame) != 1:
        raise ValueError(f"{name} must contain exactly one row, got {len(frame)}")


def fair_rent_check(
    bundle: RentModelBundle,
    x_row: pd.DataFrame,
    groups_row: pd.DataFrame | None,
    asking_rent: float,
    *,
    drivers: pd.DataFrame | None = None,
) -> RentCheckResult:
    """Check one asking rent against the calibrated interval and the market quantiles.

    Args:
        bundle: Fitted model bundle.
        x_row: One-row feature frame.
        groups_row: One-row group frame (``None`` if the bundle has no group columns).
        asking_rent: Asking rent in CHF (> 0).
        drivers: Optional precomputed SHAP drivers to attach.

    Returns:
        The rent-check result (percentile as a fraction, e.g. 0.62).

    Raises:
        ValueError: If ``x_row`` has not exactly one row or ``asking_rent`` is not positive.
    """
    _check_single_row(x_row, "x_row")
    if not (math.isfinite(asking_rent) and asking_rent > 0):
        raise ValueError(f"asking_rent must be a positive number, got {asking_rent}")
    pred = _predict_log(bundle, x_row, groups_row)
    lo_chf, hi_chf = float(np.exp(pred.lo[0])), float(np.exp(pred.hi[0]))
    cell = None if pred.cell is None else str(pred.cell.iloc[0])
    return RentCheckResult(
        expected_chf=float(np.exp(pred.mu[0])),
        lo_chf=lo_chf,
        hi_chf=hi_chf,
        band25_chf=float(np.exp(pred.band_q[0, 0])),
        band75_chf=float(np.exp(pred.band_q[0, 1])),
        market_percentile=market_percentile(pred.q[0], pred.levels, math.log(asking_rent)),
        verdict=verdict(float(asking_rent), lo_chf, hi_chf),
        coverage=pred.coverage,
        drivers=drivers,
        asking_chf=float(asking_rent),
        cell=cell,
    )


def _variant_rows(
    x_row: pd.DataFrame, changes: dict[str, list[float]]
) -> tuple[pd.DataFrame, list[tuple[str, float, float]]]:
    specs: list[tuple[str, float, float]] = [(BASELINE_LABEL, math.nan, math.nan)]
    for feature, values in changes.items():
        if len(values) == 0:
            raise ValueError(f"No values given for '{feature}'")
        original = float(pd.to_numeric(x_row[feature], errors="coerce").iloc[0])
        specs.extend((feature, original, float(value)) for value in values)
    variants = x_row.loc[x_row.index.repeat(len(specs))].reset_index(drop=True)
    for feature in changes:
        column = variants[feature].to_numpy(dtype=float, copy=True)
        for pos, (name, _, value) in enumerate(specs):
            if name == feature:
                column[pos] = value
        variants[feature] = column
    return variants, specs


def what_if(
    bundle: RentModelBundle,
    x_row: pd.DataFrame,
    groups_row: pd.DataFrame | None,
    changes: dict[str, list[float]],
    *,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    derived_cols: Sequence[str] = DERIVED_COLS,
) -> pd.DataFrame:
    """Effect of single-feature changes on the expected rent and the interval.

    Each value is applied to a copy of ``x_row`` (one change at a time, all else equal). To keep
    derived features consistent, pass ``transform=rentml.features.add_engineered_features``: the
    ``derived_cols`` are dropped first, because that function keeps an existing
    ``rooms_is_integer`` (e.g. from ``data.fix_schema``) instead of recomputing it.

    Args:
        bundle: Fitted model bundle.
        x_row: One-row frame; must contain every changed column.
        groups_row: One-row group frame (``None`` if the bundle has no group columns).
        changes: Column -> values to try, e.g. ``{"area": [60, 80, 100], "has_lift": [0, 1]}``.
        transform: Optional function applied to the modified rows before prediction, e.g. to
            recompute derived features such as ``area_per_room`` (must keep the row count).
        derived_cols: Columns dropped before ``transform`` unless changed explicitly (ignored
            without a transform); the transform must recreate those the models need.

    Returns:
        DataFrame with the baseline first, then one row per change: ``feature``, ``original``,
        ``value``, ``expected_chf``, ``lo_chf``, ``hi_chf``, ``q25_chf``, ``q75_chf``,
        ``delta_chf`` and ``delta_pct`` (relative to the baseline expected rent).

    Raises:
        ValueError: On an empty ``changes``, unknown columns, empty value lists or a ``transform``
            that changes the row count.
    """
    _check_single_row(x_row, "x_row")
    if not changes:
        raise ValueError("changes must name at least one column")
    unknown = [col for col in changes if col not in x_row.columns]
    if unknown:
        raise ValueError(f"Columns not in x_row: {unknown}")
    variants, specs = _variant_rows(x_row, changes)
    groups = None
    if groups_row is not None:
        _check_single_row(groups_row, "groups_row")
        groups = groups_row.loc[groups_row.index.repeat(len(specs))].reset_index(drop=True)
    if transform is not None:
        stale = [col for col in derived_cols if col not in changes]  # keep explicit changes
        variants = transform(variants.drop(columns=stale, errors="ignore"))
        if len(variants) != len(specs):
            raise ValueError("transform must not change the number of rows")
    frame = predict_frame(bundle, variants, groups).reset_index(drop=True)
    out = pd.DataFrame(specs, columns=["feature", "original", "value"])
    for col in ("expected", "lo", "hi", "q25", "q75"):
        out[f"{col}_chf"] = frame[col].to_numpy()
    base = out.loc[0, "expected_chf"]
    out["delta_chf"] = out["expected_chf"] - base
    out["delta_pct"] = 100.0 * (out["expected_chf"] / base - 1.0)
    return out
