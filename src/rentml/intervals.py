"""Monotone quantile models, conformal intervals and the app contract (log-rent scale).

LightGBM rejects monotone constraints with ``objective="quantile"``, so every quantile level is a
LightGBM model with a custom pinball objective. Intervals are calibrated with asymmetric
conformalized quantile regression (Romano et al., 2019), optionally per Mondrian cell
(language region x predicted-rent band).
"""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Self

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from statsmodels.stats.proportion import proportion_confint

from rentml import data
from rentml.config import INTERVAL_COVERAGE, MIN_MONDRIAN_GROUP, QUANTILE_LEVELS, RANDOM_STATE
from rentml.config import MONOTONE_CONSTRAINTS_METHOD as MONOTONE_METHOD

logger = logging.getLogger(__name__)

ALPHA = round(1.0 - INTERVAL_COVERAGE, 6)
MEDIAN = 0.5
# Shallow, strongly regularised trees: a tail level has only 5-10 % of the rows on one side.
QUANTILE_PARAMS: dict[str, float | int] = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 7,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 20.0,
    "path_smooth": 200.0,
}
MIN_TAIL_COUNT = 10  # leaf floor: about this many rows beyond the quantile per leaf
HIERARCHY = (("lang_region", "band"), ("lang_region",), ())  # Mondrian cells, finest first
GLOBAL_CELL = "all"
_COVERAGE_STATS = {"n": ("covered", "size"), "hits": ("covered", "sum"),
                   "coverage": ("covered", "mean"), "below": ("below", "mean"),
                   "above": ("above", "mean"), "mean_width": ("width", "mean"),
                   "median_width": ("width", "median")}  # fmt: skip
CELL_SEP = " × "


@dataclass(frozen=True)
class _Pinball:
    """Pinball loss as LightGBM custom objective and eval metric (picklable, unlike a closure)."""

    alpha: float

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The true Hessian is 0 almost everywhere; a constant one makes Newton steps plain steps.
        residual = y_true - y_pred
        grad = np.select([residual > 0, residual < 0], [-self.alpha, 1.0 - self.alpha], 0.0)
        return grad, np.ones_like(grad)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        return "pinball", mean_pinball_loss(y_true, y_pred, alpha=self.alpha), False


@dataclass
class MonotoneQuantileLGBM:
    """One LightGBM model per quantile level on the log rent, monotone in the given features.

    Shifted start: a median model is fitted first; every other level starts at that median plus
    the level's quantile of the cross-fitted median residuals, so its trees only learn how the
    spread changes. Started from the global quantile instead, the tails stay pulled towards it.
    With ``validation_fraction``, the trees per level are chosen by early stopping on a grouped
    inner split, then all levels are refitted on all rows.

    Attributes:
        levels: Increasing quantile levels.
        params: LightGBM parameters overriding :data:`QUANTILE_PARAMS`.
        monotone: Constraint per feature column (1, 0, -1), e.g. from ``models.monotone_vector``.
        validation_fraction: Share of groups held out for early stopping (``None``: no stopping).
        patience: Early-stopping rounds.
        anchor_folds: Folds for the cross-fitted median residuals.
        random_state: Seed for LightGBM and the splits.
    """

    levels: tuple[float, ...] = QUANTILE_LEVELS
    params: dict[str, float | int] | None = None
    monotone: list[int] | None = None
    validation_fraction: float | None = 0.2
    patience: int = 50
    anchor_folds: int = 5
    random_state: int = RANDOM_STATE

    def fit(self, X: pd.DataFrame, y_log: np.ndarray, groups: np.ndarray | None = None) -> Self:
        """Fit the median anchor and one model per level (groups keep objects on one side)."""
        y = np.asarray(y_log, dtype=float)
        groups = np.arange(y.size) if groups is None else np.asarray(groups)
        self.levels_ = np.asarray(self.levels, dtype=float)
        self.feature_names_in_ = list(X.columns)
        n_trees: dict[float, int] = {}
        if self.validation_fraction:
            split = GroupShuffleSplit(
                1, test_size=self.validation_fraction, random_state=self.random_state
            )
            inner, val = next(split.split(X, y, groups))
            self._fit_levels(X.iloc[inner], y[inner], groups[inner], {}, (X.iloc[val], y[val]))
            n_trees = {lv: max(m.best_iteration_, 1) for lv, m in self._all_models().items()}
        self._fit_levels(X, y, groups, n_trees, None)
        self.n_trees_ = {lv: m.n_estimators for lv, m in self._all_models().items()}
        logger.info("Fitted %d quantile levels, trees %s", self.levels_.size, self.n_trees_)
        return self

    def _all_models(self) -> dict[float, lgb.LGBMRegressor]:
        return {MEDIAN: self.anchor_, **self.models_}

    def _fit_one(
        self,
        level: float,
        X: pd.DataFrame,
        target: np.ndarray,
        n_trees: int | None,
        val: tuple[pd.DataFrame, np.ndarray] | None = None,
    ) -> lgb.LGBMRegressor:
        # A custom objective starts every model at 0, so the start is subtracted from the target.
        params = QUANTILE_PARAMS | {"random_state": self.random_state} | (self.params or {})
        params |= {"verbose": -1, "metric": "None"}
        tail_floor = math.ceil(MIN_TAIL_COUNT / min(level, 1.0 - level))
        params["min_child_samples"] = max(params["min_child_samples"], tail_floor)
        params["n_estimators"] = n_trees or params["n_estimators"]
        if self.monotone and any(self.monotone):
            params["monotone_constraints"] = self.monotone
            params["monotone_constraints_method"] = MONOTONE_METHOD
        model = lgb.LGBMRegressor(objective=_Pinball(level), **params)
        if val is None:
            return model.fit(X, target)
        stop = lgb.early_stopping(self.patience, verbose=False)
        metric = _Pinball(level).metric
        return model.fit(
            X, target, eval_X=(val[0],), eval_y=(val[1],), eval_metric=metric, callbacks=[stop]
        )

    def _fit_levels(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        n_trees: dict[float, int],
        val: tuple[pd.DataFrame, np.ndarray] | None,
    ) -> None:
        self.anchor_init_ = float(np.median(y))
        anchor_val = None if val is None else (val[0], val[1] - self.anchor_init_)
        target = y - self.anchor_init_
        self.anchor_ = self._fit_one(MEDIAN, X, target, n_trees.get(MEDIAN), anchor_val)
        base = self._oof_anchor(X, y, groups)
        val_base = None if val is None else self.anchor_.predict(val[0]) + self.anchor_init_
        self.models_: dict[float, lgb.LGBMRegressor] = {}
        self.offsets_: dict[float, float] = {}
        for level in self.levels_[~np.isclose(self.levels_, MEDIAN)].tolist():
            offset = self.offsets_[level] = float(np.quantile(y - base, level))
            level_val = None if val is None else (val[0], val[1] - val_base - offset)
            target = y - base - offset
            self.models_[level] = self._fit_one(level, X, target, n_trees.get(level), level_val)

    def _oof_anchor(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        # In-sample median residuals are too small and would give too narrow tails.
        n_trees = self.anchor_.best_iteration_ or self.anchor_.n_estimators
        folds = GroupKFold(self.anchor_folds, shuffle=True, random_state=self.random_state)
        oof = np.empty(y.size)
        for train, test in folds.split(X, y, groups):
            model = self._fit_one(MEDIAN, X.iloc[train], y[train] - self.anchor_init_, n_trees)
            oof[test] = model.predict(X.iloc[test]) + self.anchor_init_
        return oof

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Log quantiles of shape (n, levels) before rearrangement (columns may cross)."""
        X = X[self.feature_names_in_]  # LightGBM would silently use the positional order
        base = self.anchor_.predict(X) + self.anchor_init_
        columns = [
            base + self.models_[lv].predict(X) + self.offsets_[lv] if lv in self.models_ else base
            for lv in self.levels_.tolist()
        ]
        return np.column_stack(columns)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Log quantiles, sorted per row so that they never cross (rearrangement)."""
        return np.sort(self.predict_raw(X), axis=1)


def crossing_rate(q: np.ndarray) -> float:
    """Share of rows in which at least one pair of adjacent quantiles crosses."""
    return float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))


def quantile_calibration(y_log: np.ndarray, q_log: np.ndarray, levels: np.ndarray) -> pd.DataFrame:
    """Pinball loss and empirical share of ``y <= q`` per level (ideally equal to the level)."""
    y = np.asarray(y_log, dtype=float)
    return pd.DataFrame(
        {
            "level": levels,
            "pinball": [mean_pinball_loss(y, q_log[:, j], alpha=a) for j, a in enumerate(levels)],
            "empirical": np.mean(y[:, None] <= q_log, axis=0),
        }
    )


def quantile_columns(
    q: np.ndarray, levels: np.ndarray | tuple[float, ...], wanted: tuple[float, ...]
) -> np.ndarray:
    """Columns of a quantile matrix for the wanted levels (ValueError if one is missing)."""
    grid = list(np.round(levels, 6))
    return q[:, [grid.index(round(level, 6)) for level in wanted]]


def market_percentiles(
    q_log: np.ndarray, levels: np.ndarray | tuple[float, ...], values_log: np.ndarray
) -> np.ndarray:
    """Position of each value in its row's predicted quantiles, as a fraction in [0.01, 0.99].

    Linear interpolation between the quantiles; beyond the outermost ones the first and last
    segment are extended to the levels 0 and 1.
    """
    q = np.sort(np.atleast_2d(q_log), axis=1)
    a = np.asarray(levels, dtype=float)
    with np.errstate(invalid="ignore"):
        lower = q[:, 0] - (q[:, 1] - q[:, 0]) * a[0] / (a[1] - a[0])
        upper = q[:, -1] + (q[:, -1] - q[:, -2]) * (1 - a[-1]) / (a[-1] - a[-2])
    grid = np.column_stack([lower, q, upper])
    ext = np.r_[0.0, a, 1.0]
    pct = [np.interp(v, row, ext) for row, v in zip(grid, np.ravel(values_log), strict=True)]
    return np.clip(pct, 0.01, 0.99)


def fit_band_cutpoints(median_pred: np.ndarray, n_bands: int = 4) -> np.ndarray:
    """Cutpoints of the predicted-rent bands: quantiles of the calibration *predicted* medians."""
    return np.quantile(median_pred, np.arange(1, n_bands) / n_bands)


def assign_band(median_pred: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """Band labels ``"B1"`` (cheapest) to ``"B{len(cutpoints) + 1}"``."""
    idx = np.searchsorted(cutpoints, median_pred, side="right")
    return np.char.add("B", (idx + 1).astype(str)).astype(object)


def _kth_smallest(values: np.ndarray, k: float) -> float:
    # Exact order statistic; np.quantile(k / n, method="higher") picks the next one, and the
    # tolerance keeps float noise such as 100 * 0.8 = 80.00000000000001 from adding a rank.
    rank = max(math.ceil(k - 1e-9), 1)
    if rank > values.size:
        return math.inf
    return float(np.partition(values, rank - 1)[rank - 1])


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample conformal quantile: the ceil((n + 1)(1 - alpha))-th smallest score."""
    scores = np.asarray(scores, dtype=float)
    return _kth_smallest(scores, (scores.size + 1) * (1.0 - alpha))


def _cell_keys(groups: pd.DataFrame | None, cols: tuple[str, ...], n: int) -> np.ndarray:
    if not cols:
        return np.full(n, GLOBAL_CELL, dtype=object)
    keys = reduce(lambda a, b: a + CELL_SEP + b, (groups[col].astype(str) for col in cols))
    return keys.to_numpy(dtype=object)


class CQR:
    """Asymmetric conformalized quantile regression, optionally Mondrian (log scale).

    The lower and upper corrections are separate conformal quantiles at ``alpha / 2`` each, so
    "below" and "above" the interval are controlled separately (the fair-rent verdict needs that).
    With ``groups``, each cell of :data:`HIERARCHY` gets its own corrections if it has at least
    ``min_group_size`` calibration rows; otherwise the row falls back to its language region and
    then to the global cell. The same class calibrates the sigma-normalised GPBoost interval when
    fitted on ``(mu / sigma, mu / sigma, y / sigma)``.
    """

    def __init__(self, alpha: float = ALPHA, min_group_size: int = MIN_MONDRIAN_GROUP) -> None:
        self.alpha = alpha
        self.min_group_size = min_group_size

    @property
    def levels(self) -> tuple[float, float]:
        """Raw quantile levels that this interval calibrates, e.g. (0.1, 0.9)."""
        return round(self.alpha / 2, 6), round(1 - self.alpha / 2, 6)

    def fit(
        self, q_lo: np.ndarray, q_hi: np.ndarray, y: np.ndarray, groups: pd.DataFrame | None = None
    ) -> Self:
        """Compute the corrections per cell (only the global one without ``groups``)."""
        self.hierarchy_ = HIERARCHY if groups is not None else ((),)
        frame = pd.DataFrame({"lo_score": np.asarray(q_lo) - y, "hi_score": y - np.asarray(q_hi)})
        rows = []
        for cols in self.hierarchy_:
            for cell, part in frame.groupby(_cell_keys(groups, cols, len(frame))):
                qhat_lo = conformal_quantile(part["lo_score"], self.alpha / 2)
                qhat_hi = conformal_quantile(part["hi_score"], self.alpha / 2)
                calibrated = len(part) >= self.min_group_size or not cols
                level = CELL_SEP.join(cols) or "global"
                rows.append((cell, level, len(part), qhat_lo, qhat_hi, calibrated))
        columns = ["cell", "level", "n", "qhat_lo", "qhat_hi", "calibrated"]
        self.cells_ = pd.DataFrame(rows, columns=columns).set_index("cell")
        logger.info("CQR: %d calibrated cells", int(self.cells_["calibrated"].sum()))
        return self

    def predict(
        self, q_lo: np.ndarray, q_hi: np.ndarray, groups: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibrated ``(lo, hi, cell)``; ``cell`` names the cell that served each row."""
        calibrated = self.cells_.index[self.cells_["calibrated"]]
        served = np.full(len(q_lo), None, dtype=object)
        for cols in self.hierarchy_:
            keys = _cell_keys(groups, cols, served.size)
            take = pd.isna(served) & np.isin(keys, calibrated)
            served[take] = keys[take]
        qhat = self.cells_.loc[served]
        lo = np.asarray(q_lo) - qhat["qhat_lo"].to_numpy()
        hi = np.asarray(q_hi) + qhat["qhat_hi"].to_numpy()
        # Negative corrections can invert a narrow interval; collapse it to its midpoint.
        mid = (lo + hi) / 2
        return np.minimum(lo, mid), np.maximum(hi, mid), served


def constant_band(y: np.ndarray, pred: np.ndarray, alpha: float = ALPHA) -> float:
    """Half-width h of the symmetric conformal band ``pred ± h`` (score ``|y - pred|``)."""
    return conformal_quantile(np.abs(np.asarray(y) - pred), alpha)


def constant_band_for_coverage(y: np.ndarray, pred: np.ndarray, coverage: float) -> float:
    """Smallest h such that ``pred ± h`` covers at least ``coverage`` of the rows."""
    errors = np.abs(np.asarray(y, dtype=float) - pred)
    return _kth_smallest(errors, errors.size * coverage)


def coverage_report(
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray, groups: pd.Series | np.ndarray | None = None
) -> pd.DataFrame:
    """Coverage, misses below/above, widths and the exact 95 % Clopper-Pearson CI of coverage.

    Returns one row per group (if given) plus the overall row ``"all"``.
    """
    y = np.asarray(y, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    frame = pd.DataFrame({"covered": (y >= lo) & (y <= hi), "below": y < lo, "above": y > hi})
    frame["width"] = hi - lo
    keys = [np.full(y.size, GLOBAL_CELL)] + ([] if groups is None else [np.asarray(groups)])
    table = pd.concat([frame.groupby(key).agg(**_COVERAGE_STATS) for key in reversed(keys)])
    table["cp_low"], table["cp_high"] = proportion_confint(table["hits"], table["n"], method="beta")
    return table.drop(columns="hits").rename_axis(getattr(groups, "name", None) or "group")


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = ALPHA) -> float:
    """Mean interval (Winkler) score: width plus 2 / alpha per unit of miss; lower is better."""
    y = np.asarray(y, dtype=float)
    miss = np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None)
    return float(np.mean(np.asarray(hi) - lo + 2.0 / alpha * miss))


@dataclass
class RentModelBundle:
    """Everything the app needs to serve estimates, intervals, the rent check and the what-if.

    Attributes:
        encoder: Fitted transformer from raw rows (incl. ``re_*`` keys) to the model matrix.
        point_model: Log-rent point model (monotone LightGBM).
        quantile_model: Fitted :class:`MonotoneQuantileLGBM`.
        calibrator: :class:`CQR` fitted on the quantile model's calibration predictions.
        band_cutpoints: Log-scale cutpoints of the predicted-rent band (q50 of the quantiles).
        metadata: Free-form info (version, data source, metrics, disclaimer).
    """

    encoder: TransformerMixin
    point_model: lgb.LGBMRegressor
    quantile_model: MonotoneQuantileLGBM
    calibrator: CQR
    band_cutpoints: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        """Write the bundle with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a bundle written by :meth:`save` (pickle: trusted files only)."""
        return joblib.load(path)


def predict_frame(bundle: RentModelBundle, raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Expected rent, calibrated interval, quantiles (CHF), band and Mondrian cell per row.

    ``raw_frame`` holds the model's raw columns, the ``re_*`` keys and ``lang_region``.
    """
    X = bundle.encoder.transform(raw_frame)
    model, calibrator = bundle.quantile_model, bundle.calibrator
    q = model.predict(X)
    q_lo, q50, q_hi = quantile_columns(
        q, model.levels_, (calibrator.levels[0], MEDIAN, calibrator.levels[1])
    ).T
    band = assign_band(q50, bundle.band_cutpoints)
    groups = pd.DataFrame({"lang_region": raw_frame["lang_region"].to_numpy(), "band": band})
    lo, hi, cell = calibrator.predict(q_lo, q_hi, groups)
    out = pd.DataFrame(
        {"expected": np.exp(bundle.point_model.predict(X)), "lo": np.exp(lo), "hi": np.exp(hi)},
        index=raw_frame.index,
    )
    for j, level in enumerate(model.levels_):
        out[f"q{round(100 * level):02d}"] = np.exp(q[:, j])
    return out.assign(band=band, cell=cell)


@dataclass(frozen=True)
class RentCheck:
    """Fair-rent verdict for one asking rent (CHF); ``percentile`` is a fraction.

    ``prediction`` is the :func:`predict_frame` row the verdict is based on.
    """

    asking: float
    expected: float
    lo: float
    hi: float
    coverage: float
    verdict: str
    percentile: float
    cell: str
    prediction: pd.Series = field(compare=False, repr=False)


def fair_rent_check(
    bundle: RentModelBundle, raw_row: pd.DataFrame, asking_rent: float
) -> RentCheck:
    """Compare an asking rent with the calibrated interval and the market quantiles of one flat."""
    pred = predict_frame(bundle, raw_row).iloc[0]
    levels = bundle.quantile_model.levels_
    q_log = np.log(pred[[f"q{round(100 * a):02d}" for a in levels]].to_numpy(dtype=float))
    verdict = "within"
    if asking_rent < pred["lo"]:
        verdict = "below"
    elif asking_rent > pred["hi"]:
        verdict = "above"
    percentile = float(market_percentiles(q_log, levels, np.log([asking_rent]))[0])
    coverage = 1.0 - bundle.calibrator.alpha
    lo, hi, expected = (float(pred[col]) for col in ("lo", "hi", "expected"))
    return RentCheck(
        asking_rent, expected, lo, hi, coverage, verdict, percentile, str(pred["cell"]), pred
    )


def what_if(
    bundle: RentModelBundle,
    raw_row: pd.DataFrame,
    area_grid: Sequence[float],
    rooms_grid: Sequence[float] = (),
) -> pd.DataFrame:
    """Predictions for the flat itself (``baseline``) and one row per changed area or room count.

    Engineered features (``area_per_room``, ...) are recomputed with
    ``rentml.data.add_engineered_features``; ``delta`` and ``delta_pct`` are relative to the
    baseline expected rent.
    """
    spec = pd.DataFrame(
        [("baseline", math.nan)]
        + [("area", v) for v in area_grid]
        + [("rooms", v) for v in rooms_grid],
        columns=["feature", "value"],
    )
    variants = raw_row.loc[raw_row.index.repeat(len(spec))].reset_index(drop=True)
    for feature in ("area", "rooms"):
        changed = spec["feature"].eq(feature)
        variants[feature] = np.where(changed, spec["value"], variants[feature].astype(float))
    pred = predict_frame(bundle, data.add_engineered_features(variants))
    out = pd.concat([spec, pred], axis=1)
    base = out.loc[0, "expected"]
    return out.assign(delta=out["expected"] - base, delta_pct=100 * (out["expected"] / base - 1))
