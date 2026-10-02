"""Monotone LightGBM quantile regression and quantile-grid utilities (log-price scale).

LightGBM's built-in ``objective="quantile"`` rejects monotone constraints, so the pinball loss is a
custom objective with a constant Hessian and every level carries the point model's monotone
constraints (e.g. +1 on living area).

Boosting start (``start="shifted"``, default): a median (anchor) model is fitted first. Every other
level ``a`` starts at ``q50_hat(x) + offset_a`` with ``offset_a`` = empirical ``a``-quantile of the
cross-fitted residuals ``y - q50_hat_oof(x)`` (``anchor_folds``, grouped by ``groups``); its trees
only learn the remaining (heteroscedastic) correction. Anchor and correction trees are monotone,
so their sum is too. With a constant Hessian a tail tree moves the prediction by at most
``learning_rate * min(a, 1 - a)`` per round in the slow direction, so tails started at the *global*
quantile (``start="constant"``, kept for ablations) stay pulled towards it: cheap flats then fall
below q10 and expensive flats exceed q90 far more often than 10 % while marginal coverage still
looks fine. Always check calibration conditionally (e.g. per predicted-rent band).

LightGBM details (verified with 4.7): custom objectives ignore ``boost_from_average`` and
``LGBMRegressor.predict`` omits the ``init_score``; the starts are stored and added back in
:meth:`MonotoneQuantileLGBM.predict_raw`. LightGBM does not check the column order at predict
time, so DataFrame input is reordered to the training columns here.

:data:`DEFAULT_QUANTILE_PARAMS` (shallow, strongly regularised trees and a level-dependent leaf
floor :data:`MIN_TAIL_COUNT`) were compared by held-out pinball loss on synthetic heteroscedastic
rent-like data (n = 3'000-6'000). On real data select ``n_estimators`` by pinball loss with
``validation_fraction`` (inner grouped split, then refit) or ``eval_set`` + early stopping; the
conformal step (``rentml.conformal``) corrects the remaining miscalibration of the interval.
"""

import inspect
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Self

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

# Pinball loss and quantile-grid utilities live in a private module; re-exported here as public API.
from rentml._quantile_grid import PERCENTILE_CLAMP as PERCENTILE_CLAMP
from rentml._quantile_grid import check_level, pinball_loss, rearrange, validate_levels
from rentml._quantile_grid import crossing_rate as crossing_rate
from rentml._quantile_grid import market_percentile as market_percentile
from rentml._quantile_grid import market_percentiles as market_percentiles
from rentml._quantile_grid import quantile_calibration_table as quantile_calibration_table
from rentml._quantile_grid import quantile_columns as quantile_columns
from rentml.config import MONOTONE_CONSTRAINTS_METHOD, QUANTILE_LEVELS, RANDOM_STATE

logger = logging.getLogger(__name__)

DEFAULT_QUANTILE_PARAMS: dict[str, float | int] = {
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
"""Defaults for every per-level model; user ``params`` override them key by key."""

MIN_TAIL_COUNT = 10
"""Expected number of training rows beyond the quantile per leaf (sets the tail leaf size)."""

ANCHOR_LEVEL = 0.5
DEFAULT_PATIENCE = 50
"""Early-stopping patience for ``validation_fraction`` when ``early_stopping_rounds`` is unset."""

_LEAF_ALIASES = (
    "min_child_samples",
    "min_data_in_leaf",
    "min_data",
    "min_data_per_leaf",
    "min_samples_leaf",
)
_TREE_ALIASES = (
    "n_estimators",
    "num_iterations",
    "num_iteration",
    "n_iter",
    "num_tree",
    "num_trees",
    "num_round",
    "num_rounds",
    "nrounds",
    "num_boost_round",
    "max_iter",
)
ObjectiveFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
FeatureMatrix = pd.DataFrame | np.ndarray
EvalSet = tuple[FeatureMatrix, np.ndarray | pd.Series]
Groups = np.ndarray | pd.Series | None


@dataclass(frozen=True)
class PinballObjective:
    """Picklable LightGBM custom objective for the pinball loss (module-level, not a closure).

    For ``r = y - f`` the gradient w.r.t. ``f`` is ``-alpha`` if ``r > 0``, ``1 - alpha`` if
    ``r < 0`` and ``0`` at a tie. The true Hessian is 0 almost everywhere, so ``hess_const`` is
    returned; with ``learning_rate`` and ``reg_lambda`` it sets the Newton step size.

    Attributes:
        alpha: Quantile level in (0, 1).
        hess_const: Constant Hessian value (> 0).
    """

    alpha: float
    hess_const: float = 1.0

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return gradient and Hessian (LightGBM sklearn signature ``(y_true, y_pred)``)."""
        residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        grad = np.where(residual > 0, -self.alpha, 1.0 - self.alpha)
        grad[residual == 0] = 0.0
        return grad, np.full_like(grad, self.hess_const)


@dataclass(frozen=True)
class PinballMetric:
    """Picklable LightGBM eval metric: mean pinball loss (lower is better)."""

    alpha: float

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[str, float, bool]:
        """Return ``(name, value, is_higher_better)`` as LightGBM expects."""
        return "pinball", pinball_loss(y_true, y_pred, self.alpha), False


@dataclass
class _LevelFit:
    anchor: lgb.LGBMRegressor | None
    anchor_init: float
    models: list[lgb.LGBMRegressor]
    inits: np.ndarray
    shifted: np.ndarray


def make_pinball_objective(alpha: float, *, hess_const: float = 1.0) -> ObjectiveFn:
    """Build a LightGBM custom objective for the pinball loss.

    Args:
        alpha: Quantile level in (0, 1).
        hess_const: Constant Hessian (> 0); smaller values mean larger boosting steps.

    Returns:
        A picklable callable ``(y_true, y_pred) -> (grad, hess)``.

    Raises:
        ValueError: If ``alpha`` is outside (0, 1) or ``hess_const`` is not positive.
    """
    check_level(alpha)
    if not hess_const > 0:
        raise ValueError(f"hess_const must be positive, got {hess_const}")
    return PinballObjective(alpha=float(alpha), hess_const=float(hess_const))


def _supports_eval_x() -> bool:
    """``eval_X`` / ``eval_y`` exist from LightGBM 4.7; older versions only take ``eval_set``."""
    return "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters


def _pop_aliases(params: dict[str, float | int | str], aliases: tuple[str, ...]) -> list[int]:
    return [int(params.pop(name)) for name in aliases if name in params]


def _take(X: FeatureMatrix, rows: np.ndarray) -> FeatureMatrix:
    return X.iloc[rows] if isinstance(X, pd.DataFrame) else np.asarray(X)[rows]


def _check_groups(groups: Groups, n: int) -> np.ndarray | None:
    if groups is None:
        return None
    if len(groups) != n:
        raise ValueError(f"groups has {len(groups)} rows, expected {n}")
    return np.asarray(groups)


def _inner_split(
    n: int, fraction: float, groups: np.ndarray | None, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"validation_fraction must lie in (0, 1), got {fraction}")
    if groups is None:
        return next(ShuffleSplit(1, test_size=fraction, random_state=seed).split(np.zeros(n)))
    splitter = GroupShuffleSplit(1, test_size=fraction, random_state=seed)
    return next(splitter.split(np.zeros(n), groups=groups))


def _anchor_folds(n: int, k: int, groups: np.ndarray | None, seed: int) -> list[np.ndarray]:
    # Shuffled groups dealt round-robin to k folds (GroupKFold(shuffle=...) needs sklearn >= 1.6).
    labels = np.arange(n) if groups is None else np.unique(groups, return_inverse=True)[1]
    n_labels = int(labels.max()) + 1 if n else 0
    if n_labels < k:  # also reachable via the inner split of validation_fraction
        raise ValueError(f"anchor_folds={k} exceeds the {n_labels} rows/groups")
    fold = np.random.default_rng(seed).permutation(n_labels)[labels] % k
    return [np.flatnonzero(fold == i) for i in range(k)]


class MonotoneQuantileLGBM:
    """One LightGBM model per quantile level with a custom pinball objective (log-price scale).

    Args:
        levels: Strictly increasing quantile levels in (0, 1).
        params: LightGBM parameters overriding :data:`DEFAULT_QUANTILE_PARAMS` (no ``objective``;
            aliases of ``n_estimators`` / ``min_child_samples`` are normalised).
        monotone: Monotone constraint per feature column (1, 0, -1) or ``None``.
        hess_const: Constant Hessian passed to :func:`make_pinball_objective`.
        min_tail_count: Leaf size floor ``ceil(min_tail_count / min(level, 1 - level))`` so that
            each leaf holds about this many rows beyond the quantile (0 disables it).
        early_stopping_rounds: Patience on the pinball loss of ``eval_set`` / the inner split.
        start: ``"shifted"`` (median anchor + residual offset, see module docstring) or
            ``"constant"`` (empirical level quantile; ablation only, underfits the tails).
        anchor_folds: Folds for the cross-fitted anchor residuals (``< 2``: in-sample residuals,
            which are too small and give too narrow tails).
        validation_fraction: If set, hold out this share of the rows (grouped by ``groups``),
            pick ``n_estimators`` per level by early stopping on its pinball loss (patience
            ``early_stopping_rounds`` or :data:`DEFAULT_PATIENCE`, cap ``params["n_estimators"]``)
            and refit on all rows with those tree counts.
        random_state: Seed for LightGBM, the anchor folds and the inner split.

    Attributes:
        levels_: Fitted levels. models_: One ``LGBMRegressor`` per level (the anchor at 0.5).
        init_scores_: Per level the constant start (anchor, ``"constant"``) or the offset added
            to the anchor prediction (shifted levels, see ``shifted_``).
        anchor_model_, anchor_init_: Median anchor and its constant start (``None`` / NaN for
            ``start="constant"``).
        feature_names_in_: Training columns (``None`` for array input); ``X`` is reordered to them.
        selected_n_estimators_: Trees per level chosen on the inner split (``None`` without).
    """

    def __init__(
        self,
        levels: Sequence[float] = QUANTILE_LEVELS,
        params: dict[str, float | int | str] | None = None,
        monotone: list[int] | None = None,
        *,
        hess_const: float = 1.0,
        min_tail_count: int = MIN_TAIL_COUNT,
        early_stopping_rounds: int | None = None,
        start: Literal["shifted", "constant"] = "shifted",
        anchor_folds: int = 5,
        validation_fraction: float | None = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.levels = tuple(float(level) for level in levels)
        self.params = dict(params or {})
        self.monotone = list(monotone) if monotone is not None else None
        self.hess_const = hess_const
        self.min_tail_count = min_tail_count
        self.early_stopping_rounds = early_stopping_rounds
        self.start = start
        self.anchor_folds = anchor_folds
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _make_model(self, level: float, n_estimators: int | None) -> lgb.LGBMRegressor:
        user = dict(self.params)
        if "objective" in user:
            raise ValueError(
                "'objective' is set internally (custom pinball); remove it from params"
            )
        # LightGBM lets an alias win over the sklearn name, which would bypass the leaf floor.
        leaf, trees = _pop_aliases(user, _LEAF_ALIASES), _pop_aliases(user, _TREE_ALIASES)
        params: dict[str, float | int | str] = {**DEFAULT_QUANTILE_PARAMS, **user}
        params["min_child_samples"] = max(leaf) if leaf else params["min_child_samples"]
        params["n_estimators"] = n_estimators or (trees[0] if trees else params["n_estimators"])
        if self.min_tail_count > 0:
            tail_floor = math.ceil(self.min_tail_count / min(level, 1.0 - level))
            params["min_child_samples"] = max(int(params["min_child_samples"]), tail_floor)
        if self.monotone is not None and any(self.monotone):
            params["monotone_constraints"] = self.monotone
            params.setdefault("monotone_constraints_method", MONOTONE_CONSTRAINTS_METHOD)
        params.setdefault("metric", "None")  # only the custom pinball metric on eval sets
        params.setdefault("verbose", -1)
        params.setdefault("random_state", self.random_state)
        objective = make_pinball_objective(level, hess_const=self.hess_const)
        return lgb.LGBMRegressor(objective=objective, **params)

    def _check_input(self, X: FeatureMatrix, y: np.ndarray) -> int:
        n_features = np.shape(X)[1]
        if len(y) != np.shape(X)[0]:
            raise ValueError(f"X has {np.shape(X)[0]} rows but y has {len(y)}")
        if not np.all(np.isfinite(y)):
            raise ValueError("y_log contains non-finite values")
        if self.monotone is not None and len(self.monotone) != n_features:
            raise ValueError(f"monotone has {len(self.monotone)} entries for {n_features} features")
        if self.start not in ("shifted", "constant"):
            raise ValueError(f"start must be 'shifted' or 'constant', got {self.start!r}")
        return n_features

    def fit(
        self,
        X: FeatureMatrix,
        y_log: np.ndarray | pd.Series,
        *,
        eval_set: EvalSet | list[EvalSet] | None = None,
        groups: Groups = None,
    ) -> Self:
        """Fit the anchor and one model per level.

        Args:
            X: Training features (DataFrame recommended: its column order is enforced later).
            y_log: Log-price target.
            eval_set: ``(X_val, y_val)`` or a list of them, monitored with the pinball loss
                (validation starts are set automatically); early stopping with
                ``early_stopping_rounds``. Not combinable with ``validation_fraction``.
            groups: Group labels aligned with ``y`` (e.g. ``object_id``) for the anchor folds
                and the inner split, so that duplicate listings stay on one side.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: On invalid levels/``start``, non-finite targets, mismatched shapes or
                ``eval_set`` together with ``validation_fraction``.
        """
        levels = validate_levels(self.levels)
        y = np.asarray(y_log, dtype=float).ravel()
        n_features = self._check_input(X, y)
        group_arr = _check_groups(groups, y.size)
        n_units = y.size if group_arr is None else np.unique(group_arr).size
        if self.start == "shifted" and n_units < self.anchor_folds:
            raise ValueError(f"anchor_folds={self.anchor_folds} exceeds the {n_units} rows/groups")
        evals = [eval_set] if isinstance(eval_set, tuple) else list(eval_set or [])
        self.selected_n_estimators_: dict[float, int] | None = None
        if self.validation_fraction is not None:
            if evals:
                raise ValueError("Pass either eval_set or validation_fraction, not both")
            self.selected_n_estimators_ = self._select_n_estimators(X, y, levels, group_arr)
        fitted = self._fit_levels(X, y, levels, group_arr, evals, self.selected_n_estimators_)
        self.levels_, self.models_, self.init_scores_ = levels, fitted.models, fitted.inits
        self.shifted_, self.anchor_model_ = fitted.shifted, fitted.anchor
        self.anchor_init_, self.n_features_in_ = fitted.anchor_init, n_features
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        logger.info("Fitted %d quantile levels on %d rows (%s)", levels.size, y.size, self.start)
        return self

    def _select_n_estimators(
        self, X: FeatureMatrix, y: np.ndarray, levels: np.ndarray, groups: np.ndarray | None
    ) -> dict[float, int]:
        fraction = float(self.validation_fraction or 0.0)
        inner, val = _inner_split(y.size, fraction, groups, self.random_state)
        inner_groups = None if groups is None else groups[inner]
        patience = self.early_stopping_rounds or DEFAULT_PATIENCE
        evals = [(_take(X, val), y[val])]
        fitted = self._fit_levels(
            _take(X, inner), y[inner], levels, inner_groups, evals, None, patience
        )
        pairs = [(ANCHOR_LEVEL, fitted.anchor), *zip(levels.tolist(), fitted.models, strict=True)]
        selected = {lv: max(int(m.best_iteration_ or 0), 1) for lv, m in pairs if m is not None}
        logger.info("n_estimators chosen on %d inner validation rows: %s", val.size, selected)
        return selected

    def _oof_anchor(
        self, X: FeatureMatrix, y: np.ndarray, groups: np.ndarray | None, n_trees: int | None
    ) -> np.ndarray:
        init = float(np.quantile(y, ANCHOR_LEVEL))
        oof = np.empty(y.size)
        for val in _anchor_folds(y.size, self.anchor_folds, groups, self.random_state):
            train = np.setdiff1d(np.arange(y.size), val)
            model = self._fit_one(ANCHOR_LEVEL, _take(X, train), y[train], init, [], n_trees, None)
            oof[val] = model.predict(_take(X, val)) + init
        return oof

    def _fit_levels(
        self,
        X: FeatureMatrix,
        y: np.ndarray,
        levels: np.ndarray,
        groups: np.ndarray | None,
        evals: list[EvalSet],
        n_trees: dict[float, int] | None,
        patience: int | None = None,
    ) -> _LevelFit:
        # Fitting y - base with a constant start equals init_score = base + offset, because the
        # pinball gradient and metric depend on y - prediction only.
        evals = [(x_val, np.asarray(y_val, dtype=float).ravel()) for x_val, y_val in evals]
        n_trees, patience = n_trees or {}, patience or self.early_stopping_rounds
        anchor, anchor_init = None, math.nan
        base, eval_base = np.zeros(y.size), [np.zeros(y_val.size) for _, y_val in evals]
        if self.start == "shifted":
            anchor_init = float(np.quantile(y, ANCHOR_LEVEL))
            trees = n_trees.get(ANCHOR_LEVEL)
            anchor = self._fit_one(ANCHOR_LEVEL, X, y, anchor_init, evals, trees, patience)
            eval_base = [anchor.predict(x_val) + anchor_init for x_val, _ in evals]
            if self.anchor_folds >= 2:  # fold models use the anchor's (early-stopped) size
                base = self._oof_anchor(X, y, groups, trees or anchor.best_iteration_ or None)
            else:
                base = anchor.predict(X) + anchor_init
        shifted_evals = [(x, y_val - b) for (x, y_val), b in zip(evals, eval_base, strict=True)]
        models, inits, shifted = [], [], []
        for level in levels.tolist():
            is_anchor = anchor is not None and math.isclose(level, ANCHOR_LEVEL)
            offset = anchor_init if is_anchor else float(np.quantile(y - base, level))
            model = anchor if is_anchor else None
            if model is None:
                target, trees = y - base, n_trees.get(level)
                model = self._fit_one(level, X, target, offset, shifted_evals, trees, patience)
            models.append(model)
            inits.append(offset)
            shifted.append(anchor is not None and not is_anchor)
        return _LevelFit(anchor, anchor_init, models, np.asarray(inits), np.asarray(shifted))

    def _fit_one(
        self,
        level: float,
        X: FeatureMatrix,
        target: np.ndarray,
        init: float,
        evals: list[tuple[FeatureMatrix, np.ndarray]],
        n_trees: int | None,
        patience: int | None,
    ) -> lgb.LGBMRegressor:
        model = self._make_model(level, n_trees)
        kwargs: dict[str, object] = {}
        if evals:
            eval_x, eval_y = [x for x, _ in evals], [y_val for _, y_val in evals]
            if _supports_eval_x():
                kwargs.update(eval_X=tuple(eval_x), eval_y=tuple(eval_y))
            else:
                kwargs["eval_set"] = list(zip(eval_x, eval_y, strict=True))
            kwargs["eval_init_score"] = [np.full(y_val.size, init) for y_val in eval_y]
            kwargs["eval_metric"] = PinballMetric(level)
            if patience:
                kwargs["callbacks"] = [lgb.early_stopping(patience, verbose=False)]
        model.fit(X, target, init_score=np.full(target.size, init), **kwargs)
        logger.debug("Fitted quantile %.2f (start %.4f)", level, init)
        return model

    def _check_fitted(self) -> None:
        if not hasattr(self, "models_"):
            raise NotFittedError("MonotoneQuantileLGBM is not fitted; call fit() first")

    def _align_columns(self, X: FeatureMatrix) -> FeatureMatrix:
        names = getattr(self, "feature_names_in_", None)
        if isinstance(X, pd.DataFrame) and names is not None:
            missing = [col for col in names if col not in X.columns]
            if missing:
                raise ValueError(f"X is missing feature columns {missing}")
            return X.loc[:, names]  # LightGBM would silently use the positional order
        if np.ndim(X) != 2 or np.shape(X)[1] != self.n_features_in_:
            raise ValueError(f"X has shape {np.shape(X)}, expected {self.n_features_in_} columns")
        return X

    def predict_raw(self, X: FeatureMatrix) -> np.ndarray:
        """Predict all levels without rearrangement (quantiles may cross).

        Args:
            X: Features; a DataFrame is reordered to the training columns (extra ones ignored).

        Returns:
            Array of shape ``(n, L)`` on the log scale (tree sums + stored starts).

        Raises:
            NotFittedError: If the model has not been fitted.
            ValueError: If training columns are missing or an array has the wrong width.
        """
        self._check_fitted()
        X = self._align_columns(X)
        base = 0.0
        if self.anchor_model_ is not None:
            base = self.anchor_model_.predict(X) + self.anchor_init_
        starts = zip(self.models_, self.init_scores_, self.shifted_, strict=True)
        preds = [
            model.predict(X) + init + (base if shift else 0.0) for model, init, shift in starts
        ]
        return np.column_stack(preds)

    def predict(self, X: FeatureMatrix) -> np.ndarray:
        """Predict all levels, rearranged so that quantiles never cross.

        Args:
            X: Features (see :meth:`predict_raw`).

        Returns:
            Array of shape ``(n, L)`` on the log scale, sorted along the level axis.

        Raises:
            NotFittedError: If the model has not been fitted.
            ValueError: If training columns are missing or an array has the wrong width.
        """
        return rearrange(self.predict_raw(X))

    @property
    def best_iterations_(self) -> list[int]:
        """Trees per level chosen on the inner split, else best eval iteration (0 = none)."""
        self._check_fitted()
        if self.selected_n_estimators_:
            return [self.selected_n_estimators_[float(lv)] for lv in self.levels_]
        return [int(model.best_iteration_ or 0) for model in self.models_]
