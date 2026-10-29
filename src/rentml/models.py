"""Point models for log rent: naive baseline, LightGBM (+ target encoding), GPBoost and SHAP.

All learners except the naive baseline are trained on ``log_price``; errors are reported in CHF
after ``np.exp``. The PyTorch network lives in ``rentml.nn`` (optional ``dl`` extra).
"""

import logging
import os
import time
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, Self

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.linalg import qr
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from rentml.config import MONOTONE_CONSTRAINTS_METHOD, RANDOM_STATE

if TYPE_CHECKING:
    import shap

logger = logging.getLogger(__name__)

Folds = Sequence[tuple[np.ndarray, np.ndarray]]

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
    "n_jobs": min(8, os.cpu_count() or 1),
    "random_state": RANDOM_STATE,
    "verbose": -1,
}


class Regressor(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> object: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


class NaiveMunicipalityBaseline(BaseEstimator):
    """Median CHF/m² of the municipality times the living area (landlord's rule of thumb).

    A level's rate is used only if its group has at least ``min_count`` training objects;
    otherwise the next coarser level applies, and finally the national median.

    Args:
        levels: Hierarchy columns from finest to coarsest.
        min_count: Training objects a group needs for its own rate.
        area_col: Living-area column (m²).
        log_target: ``fit`` takes and ``predict`` returns log rent (plugs into ``fit_predict_cv``).
    """

    def __init__(
        self,
        levels: Sequence[str] = ("municipality_id", "district_id", "canton"),
        min_count: int = 5,
        area_col: str = "area",
        log_target: bool = False,
    ) -> None:
        self.levels = levels
        self.min_count = min_count
        self.area_col = area_col
        self.log_target = log_target

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> Self:
        price = np.exp(np.asarray(y, float)) if self.log_target else np.asarray(y, float)
        rate = pd.Series(price / X[self.area_col].to_numpy(float), index=X.index)
        self.national_rate_ = float(rate.median())
        self.rates_: dict[str, pd.Series] = {}
        for level in self.levels:
            stats = rate.groupby(X[level]).agg(["median", "size"])
            self.rates_[level] = stats.loc[stats["size"] >= self.min_count, "median"]
        return self

    def _rates(self, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        rate = pd.Series(self.national_rate_, index=X.index)
        source = pd.Series("national", index=X.index, name="rate_source")
        for level in reversed(self.levels):  # coarse to fine: finer levels overwrite
            mapped = X[level].map(self.rates_[level])
            hit = mapped.notna()
            rate[hit], source[hit] = mapped[hit], level
        return rate, source

    def rate_source(self, X: pd.DataFrame) -> pd.Series:
        """Hierarchy level that supplied each row's CHF/m² rate (or ``"national"``)."""
        return self._rates(X)[1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = self._rates(X)[0].to_numpy() * X[self.area_col].to_numpy(float)
        return np.log(pred) if self.log_target else pred


def monotone_vector(columns: Sequence[str], increasing: Sequence[str]) -> list[int]:
    """LightGBM ``monotone_constraints``: +1 for the ``increasing`` features, else 0."""
    if missing := set(increasing) - set(columns):
        raise KeyError(f"Monotone features not among the model columns: {sorted(missing)}")
    return [int(col in increasing) for col in columns]


def make_lgbm(
    params: dict[str, object] | None = None, monotone: list[int] | None = None
) -> lgb.LGBMRegressor:
    """LightGBM regressor for the log target: ``DEFAULT_LGBM_PARAMS`` updated by ``params``."""
    merged = DEFAULT_LGBM_PARAMS | (params or {})
    if monotone is not None:
        merged |= {"monotone_constraints": monotone}
        merged |= {"monotone_constraints_method": MONOTONE_CONSTRAINTS_METHOD}
    return lgb.LGBMRegressor(**merged)


def make_te_lgbm(
    features: Sequence[str],
    te_cols: Sequence[str],
    params: dict[str, object] | None = None,
    monotone_increasing: Sequence[str] = (),
) -> Pipeline:
    """LightGBM on ``features`` plus target-encoded location keys (RQ2 comparator).

    ``Pipeline.fit`` calls ``TargetEncoder.fit_transform``, which cross-fits: training rows get
    out-of-fold encodings, new rows the full-data ones. The model matrix has the encoded
    ``te_cols`` (same names) followed by ``features``; ``pipe[:-1]`` builds it for new rows.
    """
    encoder = TargetEncoder(target_type="continuous", random_state=RANDOM_STATE)
    encode = ColumnTransformer(
        [("te", encoder, list(te_cols)), ("num", "passthrough", list(features))],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    columns = [*te_cols, *features]
    monotone = monotone_vector(columns, monotone_increasing) if monotone_increasing else None
    return Pipeline([("encode", encode), ("model", make_lgbm(params, monotone))])


def fit_predict_cv(
    factory: Callable[[], Regressor],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    folds: Folds,
) -> np.ndarray:
    """Out-of-fold predictions: a fresh ``factory()`` model per positional ``(train, val)`` fold.

    Rows in no validation fold stay NaN.
    """
    y = np.asarray(y, float)
    oof = np.full(len(X), np.nan)
    for train_idx, val_idx in folds:
        model = factory()
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[val_idx] = model.predict(X.iloc[val_idx])
    return oof


def _suggest_lgbm_params(trial: optuna.Trial) -> dict[str, object]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def tune_lgbm_optuna(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    folds: Folds,
    n_trials: int = 60,
    model_factory: Callable[[dict[str, object]], Regressor] = make_lgbm,
    seed: int = RANDOM_STATE,
) -> tuple[dict[str, object], optuna.Study]:
    """Tune LightGBM with seeded TPE on the mean CV MAE in CHF (``exp`` of log predictions).

    ``model_factory`` maps the trial's parameters to an unfitted model, e.g.
    ``lambda p: make_te_lgbm(..., params=p)``; a median pruner stops weak trials after a fold.
    """
    y = np.asarray(y, float)
    y_chf = np.exp(y)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_lgbm_params(trial)
        maes: list[float] = []
        for step, (train_idx, val_idx) in enumerate(folds):
            model = model_factory(params)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = np.exp(model.predict(X.iloc[val_idx]))
            maes.append(float(np.mean(np.abs(y_chf[val_idx] - pred))))
            trial.report(float(np.mean(maes)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(maes))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials)
    logger.info("Optuna best CV MAE %.1f CHF after %d trials", study.best_value, n_trials)
    return study.best_params, study


GPB_TREE_PARAMS: dict[str, object] = {
    "objective": "regression_l2",
    "max_depth": 5,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "lambda_l2": 1.0,
    # Line search gave the same accuracy at 20x the cost per round (DSPRO1 data).
    "line_search_step_length": False,
    "verbose": -1,
}
# Matérn 1.5 in gpboost's parameterisation: correlation 5 % at 2.7389 * range.
MATERN15_PRACTICAL_RANGE = 2.7389


def _linear_design(x: np.ndarray) -> np.ndarray:
    """Intercept plus standardised, median-imputed features without collinear columns."""
    with warnings.catch_warnings():  # all-NaN columns warn; they become constant and drop out
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nan_to_num(np.nanmedian(x, axis=0))
        scale = np.nanstd(x, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    design = np.column_stack([np.ones(len(x)), (np.where(np.isnan(x), center, x) - center) / scale])
    # Pivoted QR keeps a maximal independent set of columns (e.g. year_built vs building_age).
    _, r, pivot = qr(design, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    return design[:, np.sort(pivot[: int((diag > 1e-8 * diag[0]).sum())])]


def _keys(groups: pd.DataFrame) -> np.ndarray:
    return groups.astype(str).to_numpy(dtype=object)


def _km(coords: pd.DataFrame | np.ndarray) -> np.ndarray:
    return np.asarray(coords, dtype=float) / 1000.0  # LV95 metres -> km: GP range in km


class GPBoostRegressor:
    """Tree boosting + nested random intercepts + Matérn GP on LV95 coordinates (GPBoost).

    Two-stage fit: covariance parameters once in a linear mixed model, then boosting with them
    fixed; the reasons and the gpboost 1.7.4 workarounds are in docs/EVALUATION_PLAN.md, sec. 7.

    Attributes:
        cov_: Covariance parameters (gpboost names, ``error_variance`` last); ``fit(cov_pars=)``
            reuses them, e.g. in CV folds.
        best_iteration_: Round chosen by early stopping (None: all rounds).
        evals_result_, eval_iterations_: Validation L1 (log scale) at the checkpoint rounds.
    """

    def __init__(
        self,
        num_boost_round: int = 1000,
        learning_rate: float = 0.01,
        eval_every: int = 10,
        num_threads: int = 0,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.num_boost_round = num_boost_round
        self.learning_rate = learning_rate
        self.eval_every = eval_every
        self.num_threads = num_threads
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        groups: pd.DataFrame,
        coords: pd.DataFrame | np.ndarray,
        eval_set: tuple | None = None,
        early_stopping_rounds: int | None = None,
        cov_pars: pd.Series | None = None,
    ) -> Self:
        """Fit both stages; ``cov_pars`` (e.g. ``cov_`` of a full-train fit) skips stage 1.

        Args:
            X: Numeric fixed-effect features (NaN allowed).
            y: Log rent.
            groups: Nested location keys, coarse to fine (``re_canton``, ``re_district``, ...).
            coords: LV95 coordinates in metres, shape ``(n, 2)``.
            eval_set: ``(X, y, groups, coords)`` of a validation split for early stopping.
            early_stopping_rounds: Patience in rounds (None: scan all checkpoints).
            cov_pars: Covariance parameters from an earlier fit.
        """
        import gpboost as gpb  # 1.7 s import, not needed by the app

        start = time.perf_counter()
        y = np.asarray(y, float)
        x = X.to_numpy(float, na_value=np.nan)
        self.feature_names_ = list(X.columns)
        self.group_cols_ = list(groups.columns)
        self.n_levels_ = int(groups.nunique().sum())
        gp_kwargs = {
            "group_data": _keys(groups),
            "gp_coords": _km(coords),
            "likelihood": "gaussian",
            "cov_function": "matern",
            "cov_fct_shape": 1.5,
            "gp_approx": "vecchia",
            "num_neighbors": 20,
            "matrix_inversion_method": "cholesky",
            "num_parallel_threads": self.num_threads or None,
            "seed": self.seed,
        }
        self.cov_ = cov_pars if cov_pars is not None else self._estimate_cov(gp_kwargs, x, y)
        gp_model = gpb.GPModel(**gp_kwargs)
        fixed = {"init_cov_pars": self.cov_.drop("error_variance").to_numpy()}
        gp_model.set_optim_params(
            params=fixed | {"init_aux_pars": self.cov_[["error_variance"]].to_numpy()}
        )
        params = {"learning_rate": self.learning_rate, "num_threads": self.num_threads}
        self.booster_ = gpb.train(
            GPB_TREE_PARAMS | params | {"seed": self.seed},
            gpb.Dataset(x, label=y, feature_name=self.feature_names_),
            num_boost_round=self.num_boost_round,
            gp_model=gp_model,
            train_gp_model_cov_pars=False,
        )
        self.fit_time_s_ = time.perf_counter() - start
        self.best_iteration_: int | None = None
        self.evals_result_: dict[str, dict[str, list[float]]] = {}
        self.eval_iterations_: list[int] = []
        if eval_set is not None:
            self._early_stopping(eval_set, early_stopping_rounds)
        logger.info("GPBoost fitted on %d rows in %.1f s", len(x), self.fit_time_s_)
        return self

    def _estimate_cov(
        self, gp_kwargs: dict[str, object], x: np.ndarray, y: np.ndarray
    ) -> pd.Series:
        """Stage 1: covariance parameters of a linear mixed model with data-driven start values."""
        import gpboost as gpb

        start = time.perf_counter()
        design = _linear_design(x)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        s2 = float(np.var(y - design @ beta))
        extent = float(np.ptp(gp_kwargs["gp_coords"], axis=0).max())
        n_groups = gp_kwargs["group_data"].shape[1]
        model = gpb.GPModel(**gp_kwargs)
        model.set_optim_params(
            params={
                "init_cov_pars": np.array([0.1 * s2] * (n_groups + 1) + [0.03 * extent]),
                "init_aux_pars": np.array([0.5 * s2]),
                "delta_rel_conv": 1e-6,  # 1e-4 stopped near the start values (DSPRO1 data)
                "maxit": 1000,
            }
        )
        model.fit(y=y, X=design)
        cov = model.get_cov_pars(format_pandas=True).iloc[0].astype(float)
        cov["error_variance"] = float(model.get_aux_pars(format_pandas=True).iloc[0, 0])
        logger.info("GPBoost stage 1 in %.1f s", time.perf_counter() - start)
        return cov

    def _early_stopping(self, eval_set: tuple, patience: int | None) -> None:
        X_val, y_val, groups_val, coords_val = eval_set
        y_val = np.asarray(y_val, float)
        checkpoints = sorted(
            {*range(self.eval_every, self.num_boost_round + 1, self.eval_every)}
            | {self.num_boost_round}
        )
        scores: list[float] = []
        for rounds in checkpoints:
            pred = self.predict(X_val, groups_val, coords_val, num_iteration=rounds)
            scores.append(float(np.mean(np.abs(y_val - pred))))
            best = checkpoints[int(np.argmin(scores))]
            if patience is not None and rounds - best >= patience:
                break
        self.best_iteration_ = best
        self.evals_result_ = {"valid": {"l1": scores}}
        self.eval_iterations_ = checkpoints[: len(scores)]

    def predict(
        self,
        X: pd.DataFrame,
        groups: pd.DataFrame,
        coords: pd.DataFrame | np.ndarray,
        return_var: bool = False,
        num_iteration: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Log-rent mean (incl. random effects and GP), and its predictive variance if asked."""
        x = X[self.feature_names_].to_numpy(float, na_value=np.nan)
        keys, xy = _keys(groups[self.group_cols_]), _km(coords)
        size = max(1, self.n_levels_) if return_var else max(1, len(x))
        outputs = [
            self.booster_.predict(
                x[i : i + size],
                num_iteration=num_iteration or self.best_iteration_,
                group_data_pred=keys[i : i + size],
                gp_coords_pred=xy[i : i + size],
                predict_var=return_var,
                pred_latent=False,
            )
            for i in range(0, len(x), size)
        ]
        mean = np.concatenate([out["response_mean"] for out in outputs])
        if not return_var:
            return mean
        return mean, np.concatenate([out["response_var"] for out in outputs])

    def predict_fixed_effects(self, X: pd.DataFrame) -> np.ndarray:
        """Tree part ``F(X)`` only; ``predict - predict_fixed_effects`` is the location effect."""
        x = X[self.feature_names_].to_numpy(float, na_value=np.nan)
        return np.asarray(
            self.booster_.predict(x, num_iteration=self.best_iteration_, ignore_gp_model=True)
        )

    def variance_components(self) -> pd.DataFrame:
        """Variance components (log scale) with their share of the summed variances.

        ``gp_range`` (Matérn range) and ``gp_practical_range`` (5 % correlation distance) are
        distances in km with ``share`` NaN.
        """
        names = {f"Group_{i}": col.removeprefix("re_") for i, col in enumerate(self.group_cols_, 1)}
        names |= {"GP_var": "gp_variance", "GP_range": "gp_range", "error_variance": "residual"}
        values = self.cov_.rename(names)
        values["gp_practical_range"] = values["gp_range"] * MATERN15_PRACTICAL_RANGE
        table = values.rename_axis("component").rename("variance").reset_index()
        is_variance = ~table["component"].str.contains("range")
        total = table.loc[is_variance, "variance"].sum()
        table["share"] = table["variance"].where(is_variance) / total
        return table


def tree_shap(
    model: object, X: pd.DataFrame, max_rows: int = 2000, seed: int = RANDOM_STATE
) -> "shap.Explanation":
    """TreeSHAP (log scale; +0.10 = +10.5 % rent) for at most ``max_rows`` rows in order.

    ``instance_names`` are the index labels of the explained rows.
    """
    import shap  # 2 s import, only needed for explanations

    if len(X) > max_rows:
        rng = np.random.default_rng(seed)
        X = X.iloc[np.sort(rng.choice(len(X), size=max_rows, replace=False))]
    raw = shap.TreeExplainer(model)(X)
    return shap.Explanation(
        values=raw.values,
        base_values=raw.base_values,
        data=raw.data,
        feature_names=list(X.columns),
        instance_names=list(X.index),
    )


def global_importance(explanation: "shap.Explanation") -> pd.DataFrame:
    """Features ranked by mean |SHAP|, with share and approximate % effect on the rent."""
    values = np.asarray(explanation.values)
    mean_abs = np.abs(values).mean(axis=0)
    table = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "mean_abs_shap": mean_abs,
            "mean_shap": values.mean(axis=0),
            "share": mean_abs / mean_abs.sum(),
            "approx_pct": 100 * np.expm1(mean_abs),
        }
    )
    return table.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def top_drivers(explanation: "shap.Explanation", row: int, k: int = 3) -> pd.DataFrame:
    """The ``k`` largest |SHAP| features of one explained row (position), as % rent effects."""
    shap_row = np.asarray(explanation.values)[row]
    order = np.argsort(-np.abs(shap_row), kind="stable")[:k]
    return pd.DataFrame(
        {
            "feature": [explanation.feature_names[j] for j in order],
            "value": np.asarray(explanation.data, dtype=object)[row, order],
            "shap_log": shap_row[order],
            "approx_pct": 100 * np.expm1(shap_row[order]),
        }
    )
