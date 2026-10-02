"""Point models: naive CHF/m2 baseline, LightGBM factories, grouped CV and Optuna tuning.

All learners except the naive baseline are trained on ``log_price``; errors are always reported
in CHF after ``np.exp`` back-transform (see ``evaluation.regression_metrics``).
"""

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from rentml.config import MONOTONE_CONSTRAINTS_METHOD, RANDOM_STATE
from rentml.features import HierarchicalTargetEncoder, monotone_vector

logger = logging.getLogger(__name__)

Fold = tuple[np.ndarray, np.ndarray]
FitKwargs = dict[str, object] | Callable[[np.ndarray, np.ndarray], dict[str, object]]

DEFAULT_LGBM_PARAMS: dict[str, object] = {
    "objective": "regression",  # L2 on the log target
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,  # bagging is only active with subsample_freq > 0
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "n_jobs": min(8, os.cpu_count() or 1),
    "verbose": -1,
}


@dataclass(frozen=True)
class ParamRange:
    """One dimension of the Optuna search space.

    Attributes:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        log: Sample on a log scale.
        integer: Sample integers instead of floats.
        step: Optional step for integer parameters.
    """

    low: float
    high: float
    log: bool = False
    integer: bool = False
    step: int | None = None


LGBM_SEARCH_SPACE: dict[str, ParamRange] = {
    "learning_rate": ParamRange(0.01, 0.1, log=True),
    "n_estimators": ParamRange(300, 2000, integer=True, step=50),
    "num_leaves": ParamRange(15, 255, log=True, integer=True),
    "min_child_samples": ParamRange(5, 100, log=True, integer=True),
    "subsample": ParamRange(0.5, 1.0),
    "colsample_bytree": ParamRange(0.4, 1.0),
    "reg_lambda": ParamRange(1e-3, 30.0, log=True),
    "reg_alpha": ParamRange(1e-3, 10.0, log=True),
}


class Regressor(Protocol):
    """Structural type of the models accepted by :func:`fit_predict_cv`."""

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: object) -> object: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class NaiveMunicipalityBaseline(RegressorMixin, BaseEstimator):
    """Median CHF/m2 per municipality with a district -> canton -> global fallback chain.

    The rate of a row comes from the finest level whose group has at least ``min_count``
    training listings; the prediction is ``rate * area`` in CHF (not log) unless ``log_target``.

    Args:
        levels: Hierarchy columns from finest to coarsest.
        min_count: Minimum number of training listings for a group rate to be used.
        area_col: Living-area column (m2).
        price_col: Rent column used when ``fit`` is called without ``y``.
        log_target: If True, ``fit`` expects ``log_price`` and ``predict`` returns log CHF, so the
            baseline plugs into :func:`fit_predict_cv` like the log-scale models.

    Attributes:
        global_rate_: Median CHF/m2 over all training rows.
        rates_: Per level a DataFrame with ``rate`` and ``n`` indexed by group code.
    """

    def __init__(
        self,
        levels: Sequence[str] = ("municipality_id", "district_id", "canton"),
        min_count: int = 5,
        area_col: str = "area",
        price_col: str = "price",
        log_target: bool = False,
    ) -> None:
        self.levels = levels
        self.min_count = min_count
        self.area_col = area_col
        self.price_col = price_col
        self.log_target = log_target

    def _area(self, X: pd.DataFrame) -> pd.Series:
        if self.area_col not in X.columns:
            raise KeyError(f"NaiveMunicipalityBaseline: column {self.area_col!r} missing")
        return pd.to_numeric(X[self.area_col], errors="coerce").astype(float)

    def _target(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None) -> pd.Series:
        if y is None:
            if self.price_col not in X.columns:
                raise KeyError(f"y is None and column {self.price_col!r} is missing")
            return pd.to_numeric(X[self.price_col], errors="coerce").astype(float)
        values = np.asarray(y, dtype=float).ravel()
        if len(values) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(values)}")
        return pd.Series(np.exp(values) if self.log_target else values, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> Self:
        """Compute median CHF/m2 and counts per group at every level.

        Args:
            X: Frame with the area column and the hierarchy columns.
            y: Rent in CHF (log price if ``log_target``); defaults to ``X[price_col]`` (CHF).

        Returns:
            The fitted baseline.

        Raises:
            KeyError: If the area or price column is missing.
            ValueError: If ``y`` looks log-transformed or no row has a valid rate.
        """
        price = self._target(X, y)
        if price.median() < 50:
            raise ValueError("y looks like log price; the naive baseline needs rent in CHF")
        area = self._area(X).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = price.to_numpy() / np.where(area > 0, area, np.nan)
        valid = np.isfinite(rate)
        if not valid.any():
            raise ValueError("No row with valid price and area > 0")
        self.global_rate_ = float(np.median(rate[valid]))
        self.rates_: dict[str, pd.DataFrame] = {}
        for level in self.levels:
            if level not in X.columns:
                logger.warning("Baseline level %r not in X; skipped", level)
                continue
            keys = X[level].to_numpy()[valid]
            grouped = pd.Series(rate[valid]).groupby(keys, dropna=True)
            self.rates_[level] = pd.DataFrame({"rate": grouped.median(), "n": grouped.size()})
        self.n_features_in_ = X.shape[1]
        return self

    def _rate_and_source(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        check_is_fitted(self, "rates_")
        rate = np.full(len(X), self.global_rate_)
        source = np.full(len(X), "global", dtype=object)
        for level in reversed(list(self.levels)):  # coarse -> fine, finer levels overwrite
            table = self.rates_.get(level)
            if table is None or level not in X.columns:
                continue
            usable = table.loc[table["n"] >= self.min_count, "rate"]
            mapped = X[level].map(usable).to_numpy(dtype=float)
            hit = ~np.isnan(mapped)
            rate[hit] = mapped[hit]
            source[hit] = level
        return rate, source

    def predict_rate(self, X: pd.DataFrame) -> np.ndarray:
        """Return the CHF/m2 rate used for each row.

        Args:
            X: Frame with the hierarchy columns.

        Returns:
            Array of CHF/m2 rates.
        """
        return self._rate_and_source(X)[0]

    def rate_source(self, X: pd.DataFrame) -> pd.Series:
        """Return the hierarchy level that supplied each row's rate (or ``"global"``).

        Args:
            X: Frame with the hierarchy columns.

        Returns:
            Series of level names, index aligned with ``X``.
        """
        return pd.Series(self._rate_and_source(X)[1], index=X.index, name="rate_source")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict monthly rent as ``rate * area``.

        Args:
            X: Frame with the area and hierarchy columns.

        Returns:
            Predicted rent in CHF, or its log if ``log_target`` (NaN where the area is missing).
        """
        area = self._area(X).to_numpy()
        if np.isnan(area).any():
            logger.warning("Baseline: %d rows without area -> NaN prediction", np.isnan(area).sum())
        pred = self.predict_rate(X) * area
        if not self.log_target:
            return pred
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(pred)


class _FillMissingColumn(TransformerMixin, BaseEstimator):
    """Add ``col = fill`` at predict time if a frame lacks it; ``fit_transform`` never fills.

    The group column only drives the encoder's inner folds during ``fit``, so new listings (rent
    check, app) can be scored without ``object_id`` while fitting without it still fails loudly.
    """

    def __init__(self, col: str = "object_id", fill: str = "__predict__") -> None:
        self.col = col
        self.fill = fill

    def fit(self, X: pd.DataFrame, y: object = None) -> Self:
        return self

    def fit_transform(
        self, X: pd.DataFrame, y: object = None, **fit_params: object
    ) -> pd.DataFrame:
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X if self.col in X.columns else X.assign(**{self.col: self.fill})


def make_lgbm(
    params: dict[str, object] | None = None,
    *,
    monotone: list[int] | None = None,
    seed: int = RANDOM_STATE,
) -> lgb.LGBMRegressor:
    """Create a LightGBM regressor for the log target with sensible defaults.

    Args:
        params: Overrides for :data:`DEFAULT_LGBM_PARAMS` (e.g. tuned Optuna parameters).
        monotone: Optional ``monotone_constraints`` vector (+1/0/-1 per feature, see
            ``features.monotone_vector``); sets ``config.MONOTONE_CONSTRAINTS_METHOD``.
        seed: Random seed (``random_state``) unless ``params`` sets one explicitly.

    Returns:
        An unfitted ``LGBMRegressor``.

    Raises:
        ValueError: If ``monotone`` contains values other than -1, 0 or 1.
    """
    merged: dict[str, object] = {**DEFAULT_LGBM_PARAMS, "random_state": seed, **(params or {})}
    if monotone is not None:
        bad = sorted({int(v) for v in monotone} - {-1, 0, 1})
        if bad:
            raise ValueError(f"monotone constraints must be -1, 0 or 1, got {bad}")
        merged["monotone_constraints"] = [int(v) for v in monotone]
        merged.setdefault("monotone_constraints_method", MONOTONE_CONSTRAINTS_METHOD)
    return lgb.LGBMRegressor(**merged)


def make_te_lgbm(
    feature_cols: list[str],
    te_cols: Sequence[str] = ("re_canton", "re_district", "re_municipality"),
    *,
    group_col: str | None = "object_id",
    params: dict[str, object] | None = None,
    monotone_increasing: Sequence[str] = (),
    smoothing: float = 20.0,
    seed: int = RANDOM_STATE,
) -> Pipeline:
    """LightGBM with out-of-fold hierarchical target encoding (RQ2 comparison model).

    ``fit`` calls the encoder's ``fit_transform`` (out-of-fold encodings, grouped by
    ``group_col``) and ``predict`` its ``transform`` (full-fit statistics), so the pipeline can be
    used with :func:`fit_predict_cv` without leaking targets across outer folds. The model sees
    ``te_<col>`` for every ``te_cols`` entry followed by ``feature_cols`` (in that order; the
    monotone vector is built for that order). Training frames must contain ``group_col``;
    prediction frames may omit it (a ``fill_group`` step adds a placeholder).

    Args:
        feature_cols: Numeric features passed through unchanged.
        te_cols: Hierarchy columns to encode (coarsest first).
        group_col: Object-id column for the inner folds (required when fitting), or None.
        params: LightGBM parameter overrides for :func:`make_lgbm`.
        monotone_increasing: Features with a non-decreasing constraint (e.g. ``("area",)``).
        smoothing: Encoder pseudo-count ``m``.
        seed: Random seed for the encoder folds and LightGBM.

    Returns:
        An unfitted ``Pipeline`` (steps ``"fill_group"`` if ``group_col``, ``"prep"``, ``"lgbm"``)
        expecting a DataFrame with ``feature_cols``, ``te_cols`` (and ``group_col`` for ``fit``).
    """
    encoder = HierarchicalTargetEncoder(
        cols=tuple(te_cols), smoothing=smoothing, random_state=seed, group_col=group_col
    )
    te_inputs = [*te_cols, group_col] if group_col is not None else list(te_cols)
    prep = ColumnTransformer(
        [("te", encoder, te_inputs), ("num", "passthrough", list(feature_cols))],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    model_cols = [f"te_{c}" for c in te_cols] + list(feature_cols)
    monotone = monotone_vector(model_cols, monotone_increasing) if monotone_increasing else None
    steps = [("prep", prep), ("lgbm", make_lgbm(params, monotone=monotone, seed=seed))]
    if group_col is not None:
        steps.insert(0, ("fill_group", _FillMissingColumn(group_col)))
    return Pipeline(steps)


def _take(
    data: pd.DataFrame | pd.Series | np.ndarray, idx: np.ndarray
) -> pd.DataFrame | pd.Series | np.ndarray:
    return data.iloc[idx] if isinstance(data, pd.DataFrame | pd.Series) else data[idx]


def fit_predict_cv(
    make_model: Callable[[], Regressor],
    X: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
    folds: Sequence[Fold],
    *,
    fit_kwargs: FitKwargs | None = None,
) -> np.ndarray:
    """Fit a fresh model per fold and return out-of-fold predictions (log scale).

    Args:
        make_model: Factory returning an unfitted model with ``fit``/``predict``.
        X: Feature frame (rows in the positional order the folds refer to).
        y_log: Log target aligned with ``X``.
        folds: Positional ``(train_idx, val_idx)`` pairs, e.g. from ``splits.grouped_cv_folds``.
        fit_kwargs: Extra ``fit`` arguments, either a dict used for every fold or a callable
            ``(train_idx, val_idx) -> dict`` for fold-specific arguments (sample weights, eval
            sets). An early-stopping ``eval_set`` must be carved out of ``train_idx`` (inner
            grouped split), never built from ``val_idx``: stopping on the scored fold biases the
            OOF predictions optimistically. ``val_idx`` is passed for bookkeeping only.

    Returns:
        OOF log predictions of length ``len(X)``; rows in no validation fold are NaN.

    Raises:
        ValueError: If ``folds`` is empty or ``X`` and ``y_log`` differ in length.
    """
    if len(folds) == 0:
        raise ValueError("folds must contain at least one (train_idx, val_idx) pair")
    if len(X) != len(y_log):
        raise ValueError(f"X has {len(X)} rows but y_log has {len(y_log)}")
    oof = np.full(len(X), np.nan)
    for fold_no, (train_idx, val_idx) in enumerate(folds):
        train_idx, val_idx = np.asarray(train_idx), np.asarray(val_idx)
        kwargs = fit_kwargs(train_idx, val_idx) if callable(fit_kwargs) else (fit_kwargs or {})
        model = make_model()
        model.fit(_take(X, train_idx), _take(y_log, train_idx), **kwargs)
        oof[val_idx] = np.asarray(model.predict(_take(X, val_idx)), dtype=float).ravel()
        logger.debug("fit_predict_cv: fold %d done (%d val rows)", fold_no, len(val_idx))
    n_missing = int(np.isnan(oof).sum())
    if n_missing:
        logger.warning("fit_predict_cv: %d rows are in no validation fold (NaN)", n_missing)
    return oof


def _suggest(trial: optuna.Trial, space: dict[str, ParamRange]) -> dict[str, object]:
    params: dict[str, object] = {}
    for name, rng in space.items():
        if rng.integer:
            step = rng.step or 1
            params[name] = trial.suggest_int(
                name, int(rng.low), int(rng.high), step=step, log=rng.log and step == 1
            )
        else:
            params[name] = trial.suggest_float(name, rng.low, rng.high, log=rng.log)
    return params


def _cv_mae_objective(
    X: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
    folds: Sequence[Fold],
    space: dict[str, ParamRange],
    make: Callable[[dict[str, object]], Regressor],
) -> Callable[[optuna.Trial], float]:
    y_chf = np.exp(np.asarray(y_log, dtype=float))

    def objective(trial: optuna.Trial) -> float:
        model_params = _suggest(trial, space)
        fold_maes: list[float] = []
        for step, (train_idx, val_idx) in enumerate(folds):
            model = make(model_params)
            model.fit(_take(X, train_idx), _take(y_log, train_idx))
            pred = np.exp(model.predict(_take(X, val_idx)))
            fold_maes.append(float(np.mean(np.abs(y_chf[val_idx] - pred))))
            trial.report(float(np.mean(fold_maes)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_mae", fold_maes)
        trial.set_user_attr("mae_std", float(np.std(fold_maes)))
        return float(np.mean(fold_maes))

    return objective


def tune_lgbm_optuna(
    X: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
    folds: Sequence[Fold],
    *,
    n_trials: int = 40,
    timeout: float | None = None,
    monotone: list[int] | None = None,
    seed: int = RANDOM_STATE,
    base_params: dict[str, object] | None = None,
    search_space: dict[str, ParamRange] | None = None,
    prune: bool = True,
    model_factory: Callable[[dict[str, object]], Regressor] | None = None,
) -> tuple[dict[str, object], optuna.Study]:
    """Tune LightGBM with Optuna (TPE, seeded) on the mean grouped-CV MAE in CHF.

    Search space (:data:`LGBM_SEARCH_SPACE`, overridable per key via ``search_space``; keys in
    ``base_params`` are removed from it and stay fixed):
    ``learning_rate`` 0.01-0.1 (log), ``n_estimators`` 300-2000 (step 50), ``num_leaves`` 15-255
    (log), ``min_child_samples`` 5-100 (log), ``subsample`` 0.5-1.0, ``colsample_bytree`` 0.4-1.0,
    ``reg_lambda`` 1e-3-30 (log), ``reg_alpha`` 1e-3-10 (log). Each trial fits one model per fold
    on the log target and scores ``MAE(exp(y_log), exp(pred))``. With ``prune=True`` a
    ``MedianPruner`` stops trials whose running mean fold MAE is worse than the median of earlier
    trials after the same number of folds. Optuna logging is set to WARNING during the run.

    Args:
        X: Training features (positional order matching ``folds``).
        y_log: Log target.
        folds: Positional ``(train_idx, val_idx)`` pairs (grouped by object).
        n_trials: Number of trials.
        timeout: Optional time budget in seconds.
        monotone: Optional monotone constraint vector passed to :func:`make_lgbm`.
        seed: Seed for the TPE sampler and the models.
        base_params: Fixed parameters applied to every trial; they win over the search space
            (e.g. ``{"n_estimators": 300}`` pins the tree count for a faster study).
        search_space: Per-parameter overrides/additions to :data:`LGBM_SEARCH_SPACE`.
        prune: Use a ``MedianPruner``.
        model_factory: Optional ``params -> model`` factory used instead of :func:`make_lgbm`,
            e.g. ``lambda p: make_te_lgbm(cols, params=p)`` to tune the target-encoded pipeline
            (``monotone`` and ``seed`` are then the factory's responsibility).

    Returns:
        ``(best_params, study)``; ``best_params`` combines ``base_params`` with the best trial's
        values and can be passed to :func:`make_lgbm` directly.

    Raises:
        ValueError: If ``n_trials < 1`` or ``folds`` is empty.
        RuntimeError: If no trial completed (e.g. ``timeout`` shorter than one trial).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if len(folds) == 0:
        raise ValueError("folds must contain at least one (train_idx, val_idx) pair")
    fixed = dict(base_params or {})
    merged_space = {**LGBM_SEARCH_SPACE, **(search_space or {})}
    space = {name: rng for name, rng in merged_space.items() if name not in fixed}
    folds = [(np.asarray(tr), np.asarray(va)) for tr, va in folds]

    def make(tuned: dict[str, object]) -> Regressor:
        if model_factory is not None:
            return model_factory({**fixed, **tuned})
        return make_lgbm({**fixed, **tuned}, monotone=monotone, seed=seed)

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        if prune
        else optuna.pruners.NopPruner()
    )
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed), pruner=pruner
        )
        study.optimize(
            _cv_mae_objective(X, y_log, folds, space, make), n_trials=n_trials, timeout=timeout
        )
    finally:
        optuna.logging.set_verbosity(previous)
    if not study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)):
        raise RuntimeError("No Optuna trial completed (timeout too short or all trials failed)")
    logger.info("Optuna best CV MAE %.1f CHF after %d trials", study.best_value, len(study.trials))
    return {**fixed, **study.best_params}, study
