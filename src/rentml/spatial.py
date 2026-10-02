"""Spatial mixed-effects boosting (GPBoost) and spatial autocorrelation diagnostics.

GPBoost (Sigrist, JMLR 2022) models ``log_price = F(X) + b_canton + b_district +
b_municipality + g(s) + eps``: boosting for ``F``, nested random intercepts and a Matern GP ``g``
(Vecchia approximation). The fit is two-stage by default (see ``rentml._gpboost_cov``): the
covariance parameters are estimated once in a linear mixed model with data-driven starting values
and kept fixed while boosting, because joint re-estimation took ~12 s per round and collapsed the
error variance on listings that share coordinates (verified on the DSPRO1 data).
gpboost 1.7.4 pitfalls handled here (verified): "cholesky" is used as the default "iterative"
solver was >40x slower; groups + Vecchia GP use a Laplace approximation (residual variance
becomes an auxiliary parameter); for that model predictive variances abort the process when a
call holds more points than training random-effect levels - hence chunked, chunk-invariant
predictions, and early stopping by chunked validation predictions instead of gpboost's
``valid_sets`` (which predicts in one call and aborts); multi-threaded fits are not
bit-reproducible (``num_threads=1`` is); pickling a ``Booster`` drops its ``GPModel`` (own pickle
support here). Moran's I lives in ``rentml.autocorrelation`` and is re-exported.
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Self

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from rentml._gpboost_cov import CovEstimate, estimate_cov_pars, expected_names, uses_aux_error
from rentml.autocorrelation import Coords, MoranResult, as_coord_array, knn_weights, morans_i
from rentml.config import RANDOM_STATE

if TYPE_CHECKING:
    import gpboost as gpb

__all__ = [
    "CovEstimate",
    "GPBoostConfig",
    "GPBoostRegressor",
    "MoranResult",
    "knn_weights",
    "morans_i",
]

logger = logging.getLogger(__name__)

EvalSet = tuple[pd.DataFrame, np.ndarray | pd.Series, pd.DataFrame | None, Coords | None]
_TRAIN_NA_LEVEL = "__na__"
_UNSEEN_LEVEL = "__unseen__"
# Distance (in range units) where the correlation is 5 %; gpboost's "matern" follows Rasmussen &
# Williams (checked against a dense likelihood), "exponential" is exp(-d / range).
_PRACTICAL_RANGE = {("matern", 0.5): 2.9957, ("matern", 1.5): 2.7389, ("matern", 2.5): 2.6469}
_PRACTICAL_RANGE[("exponential", None)] = 2.9957  # shape is ignored by gpboost here
_COMPONENT_NAMES = {"Error_var": "residual", "GP_var": "gp_variance", "GP_range": "gp_range"}


def _import_gpboost() -> ModuleType:
    """Import gpboost lazily: compiled optional dependency, not needed for Moran's I."""
    import gpboost

    return gpboost


@dataclass
class GPBoostConfig:
    """Hyperparameters of the GPBoost mixed-effects model.

    Boosting fields (``learning_rate`` to ``lambda_l2``) use LightGBM names; the GP fields
    (``cov_function`` to ``num_neighbors``) are gpboost ``GPModel`` arguments.

    Attributes:
        group_cols: Nested random-intercept key columns, coarse to fine (may be empty).
        use_gp: Add a Gaussian process on the coordinates.
        num_boost_round: Boosting rounds (upper bound with early stopping).
        line_search_step_length: Line search per boosting step. Without it, lr=0.05 overfit
            after a few rounds (GPBoost gradients are scaled by the inverse covariance); with
            it the validation error stayed flat over hundreds of rounds (verified).
        coord_scale_m: Coordinates are divided by this (1000 -> km, so the GP range is in km).
        two_stage: For groups + Vecchia GP (the default structure), estimate the covariance
            parameters once (linear mixed model) and keep them fixed while boosting; False
            re-estimates them in every round (slow, see module notes). Other structures are
            always estimated jointly (gpboost 1.7.4 cannot fix their parameters).
        cov_delta_rel_conv: Convergence tolerance of the covariance estimation (1e-4 stopped
            near the starting values on DSPRO1 data, so keep the strict default).
        cov_maxit: Maximum optimiser iterations of the covariance estimation.
        eval_every: Boosting rounds between two validation checkpoints (early stopping).
        matrix_inversion_method: gpboost linear-algebra backend (see module notes).
        num_threads: Threads for boosting and GP (0 = all cores, 1 = bit-reproducible).
        predict_batch_size: Maximum rows per gpboost prediction call.
        seed: Seed of the Vecchia ordering and the boosting subsampling.
    """

    group_cols: tuple[str, ...] = ("re_canton", "re_district", "re_municipality")
    use_gp: bool = True
    cov_function: str = "matern"
    cov_fct_shape: float = 1.5
    gp_approx: str = "vecchia"
    num_neighbors: int = 20
    num_boost_round: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    lambda_l2: float = 1.0
    line_search_step_length: bool = True
    coord_scale_m: float = 1000.0
    two_stage: bool = True
    cov_delta_rel_conv: float = 1e-6
    cov_maxit: int = 1000
    eval_every: int = 10
    matrix_inversion_method: str = "cholesky"
    num_threads: int = 0
    predict_batch_size: int = 1000
    seed: int = RANDOM_STATE

    def booster_params(self) -> dict[str, float | int | str]:
        """Return the tree-boosting parameters in gpboost/LightGBM naming."""
        keys = ("learning_rate", "max_depth", "num_leaves", "min_data_in_leaf")
        keys += ("feature_fraction", "lambda_l2", "line_search_step_length", "num_threads", "seed")
        return {"objective": "regression_l2", "verbose": -1} | {k: getattr(self, k) for k in keys}


class GPBoostRegressor:
    """Tree boosting with nested random intercepts and a spatial GP (GPBoost) on ``log_price``.

    Unseen (or missing) group keys at prediction time get the prior of their level: a new
    municipality in a known district borrows the district effect and a larger variance.

    Args:
        config: Hyperparameters; defaults to :class:`GPBoostConfig`.

    Attributes:
        booster_: Fitted ``gpboost.Booster`` (its ``gp_model`` holds the random effects).
        feature_names_: Fixed-effect feature columns in training order.
        n_levels_: Number of training levels per group column.
        fit_time_s_: Wall-clock training time in seconds.
        best_iteration_: Best boosting round when early stopping was used, else None.
        evals_result_: Validation L1 curve (``{"valid": {"l1": [...]}}``) at the rounds in
            ``eval_iterations_`` when early stopping was used.
        cov_: Covariance parameters of the two-stage fit (reusable via ``fit(cov_pars=...)``).
    """

    def __init__(self, config: GPBoostConfig | None = None) -> None:
        self.config = config if config is not None else GPBoostConfig()
        self.booster_: gpb.Booster | None = None
        self.feature_names_: list[str] = []
        self.n_levels_: dict[str, int] = {}
        self.coord_origin_: np.ndarray | None = None
        self.fit_time_s_ = float("nan")
        self.best_iteration_: int | None = None
        self.evals_result_: dict[str, dict[str, list[float]]] = {}
        self.eval_iterations_: list[int] = []
        self.cov_: CovEstimate | None = None
        self._train_groups: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y_log: np.ndarray | pd.Series,
        group_data: pd.DataFrame | None,
        coords: Coords | None,
        *,
        eval_set: EvalSet | None = None,
        early_stopping_rounds: int = 20,
        cov_pars: CovEstimate | None = None,
    ) -> Self:
        """Fit the fixed effects, random-effect variances and GP parameters.

        Args:
            X: Numeric fixed-effect features (NaN allowed).
            y_log: Log rent; pandas inputs must share ``X.index``.
            group_data: Frame with ``config.group_cols`` (None if there are none).
            coords: ``(n, 2)`` LV95 coordinates in metres (None if ``use_gp`` is False).
            eval_set: Optional ``(X, y_log, group_data, coords)``: the best round is chosen by
                the validation L1 (log scale) of the full model including random effects,
                evaluated every ``config.eval_every`` rounds; ``predict`` then uses it.
            early_stopping_rounds: Stop scanning checkpoints after this many rounds without
                improvement.
            cov_pars: Covariance parameters from an earlier fit (``cov_``) to skip stage 1,
                e.g. inside cross-validation. Only used for two-stage fits (see ``two_stage``).

        Returns:
            The fitted model.

        Raises:
            ValueError: On misaligned/invalid inputs, a model without random effects or
                ``cov_pars`` that do not match the random-effect structure.
            KeyError: If feature or group columns are missing (TypeError: non-numeric ``X``).
        """
        gpboost = _import_gpboost()
        cfg = self.config
        if not cfg.group_cols and not cfg.use_gp:
            raise ValueError("GPBoostConfig needs group_cols and/or use_gp=True")
        if early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be >= 1")
        x, groups, xy = self._prepare(X, group_data, coords, training=True)
        y = _prepare_target(y_log, X)
        model_kwargs = self._gp_model_kwargs(groups, xy)
        start = time.perf_counter()
        gp_model = gpboost.GPModel(**model_kwargs)
        # Fixed covariance parameters only work in gpboost 1.7.4's latent formulation (groups +
        # Vecchia GP); for groups-only or GP-only models training aborts with an Eigen index
        # assertion (verified), so those structures are estimated jointly as before.
        staged = cfg.two_stage and uses_aux_error(len(cfg.group_cols), cfg.use_gp, cfg.gp_approx)
        if staged:
            self.cov_ = self._stage_one(gpboost, model_kwargs, x, y, xy, cov_pars)
            fixed: dict[str, object] = {"init_cov_pars": self.cov_.cov_pars}
            if self.cov_.aux_pars is not None:
                fixed["init_aux_pars"] = self.cov_.aux_pars
            gp_model.set_optim_params(params=fixed)
        train_set = gpboost.Dataset(x, label=y, feature_name=[str(c) for c in self.feature_names_])
        self.booster_ = gpboost.train(
            cfg.booster_params(),
            train_set,
            num_boost_round=cfg.num_boost_round,
            gp_model=gp_model,
            train_gp_model_cov_pars=not staged,
        )
        self.fit_time_s_ = time.perf_counter() - start
        self.best_iteration_, self.evals_result_, self.eval_iterations_ = None, {}, []
        if eval_set is not None:
            self._select_iteration(eval_set, early_stopping_rounds)
        logger.info("GPBoost fitted on %d rows in %.1f s", len(x), self.fit_time_s_)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        group_data: pd.DataFrame | None,
        coords: Coords | None,
        *,
        return_var: bool = False,
        num_iteration: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict the log rent and optionally its predictive variance.

        Args:
            X: Features with the training columns.
            group_data: Group keys (unseen or missing levels get the prior).
            coords: Coordinates in metres.
            return_var: Also return the predictive variance (log scale).
            num_iteration: Boosting rounds to use; defaults to ``best_iteration_`` (all rounds
                when no early stopping was used).

        Returns:
            Mean predictions, or ``(mean, variance)`` if ``return_var``; the variance adds the
            posterior variance of all random effects and the GP to the residual variance.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        booster = self._check_fitted()
        x, groups, xy = self._prepare(X, group_data, coords, training=False)
        rounds = num_iteration if num_iteration is not None else self.best_iteration_
        means, variances = [np.empty(0)], [np.empty(0)]
        size = self._chunk_size()
        for rows in (slice(i, i + size) for i in range(0, len(x), size)):
            out = booster.predict(
                np.ascontiguousarray(x[rows]),
                num_iteration=rounds,
                group_data_pred=None if groups is None else groups[rows].copy(),
                gp_coords_pred=None if xy is None else np.ascontiguousarray(xy[rows]),
                predict_var=return_var,
                pred_latent=False,
            )
            means.append(np.asarray(out["response_mean"], dtype=float))
            variances.append(np.asarray(out["response_var"] if return_var else [], dtype=float))
        mean = np.concatenate(means)
        return (mean, np.concatenate(variances)) if return_var else mean

    def predict_fixed_effects(
        self, X: pd.DataFrame, *, num_iteration: int | None = None
    ) -> np.ndarray:
        """Predict only the tree-ensemble part ``F(X)`` (log scale), ignoring all random effects.

        ``predict(...) - predict_fixed_effects(...)`` is the location effect (random intercepts
        plus GP), e.g. to show how strongly small municipalities are shrunk.

        Args:
            X: Features with the training columns.
            num_iteration: Boosting rounds to use; defaults to ``best_iteration_``.

        Returns:
            Fixed-effect predictions on the log scale.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        booster = self._check_fitted()
        if missing := [c for c in self.feature_names_ if c not in X.columns]:
            raise KeyError(f"X lacks training features: {missing}")
        x = X.loc[:, self.feature_names_].to_numpy(float, na_value=np.nan)
        rounds = num_iteration if num_iteration is not None else self.best_iteration_
        out = booster.predict(np.ascontiguousarray(x), num_iteration=rounds, ignore_gp_model=True)
        return np.asarray(out, dtype=float).ravel()

    def _stage_one(
        self,
        gpboost: ModuleType,
        model_kwargs: dict[str, object],
        x: np.ndarray,
        y: np.ndarray,
        xy: np.ndarray | None,
        cov_pars: CovEstimate | None,
    ) -> CovEstimate:
        cfg = self.config
        names = expected_names(len(cfg.group_cols), cfg.use_gp, cfg.gp_approx)
        if cov_pars is not None:
            if tuple(cov_pars.names) != names:
                raise ValueError(f"cov_pars {cov_pars.names} do not match the model {names}")
            return cov_pars
        estimate = estimate_cov_pars(
            gpboost, model_kwargs, x, y, xy, n_groups=len(cfg.group_cols),
            gp_approx=cfg.gp_approx, delta_rel_conv=cfg.cov_delta_rel_conv, maxit=cfg.cov_maxit,
        )  # fmt: skip
        if estimate.names != names:
            raise ValueError(f"gpboost returned {estimate.names}, expected {names}")
        return estimate

    def _select_iteration(self, eval_set: EvalSet, patience: int) -> None:
        """Pick the best round by chunked validation predictions (gpboost's own aborts)."""
        x_val, y_val, g_val, c_val = eval_set
        y = _prepare_target(y_val, x_val)
        step, total = max(1, self.config.eval_every), self.config.num_boost_round
        checkpoints = sorted({*range(step, total + 1, step), total})
        best, best_score, scores = checkpoints[0], np.inf, []
        for rounds in checkpoints:
            pred = self.predict(x_val, g_val, c_val, num_iteration=rounds)
            scores.append(float(np.mean(np.abs(y - pred))))
            if scores[-1] < best_score - 1e-12:
                best, best_score = rounds, scores[-1]
            elif rounds - best >= patience:
                break
        self.best_iteration_ = best
        self.evals_result_ = {"valid": {"l1": scores}}
        self.eval_iterations_ = checkpoints[: len(scores)]

    def variance_components(self) -> pd.DataFrame:
        """Estimated variance components on the log scale.

        Returns:
            Frame with ``component`` (residual, canton, district, municipality, gp_variance,
            gp_range, gp_practical_range), ``parameter`` (gpboost name), ``variance`` and
            ``share`` of the summed variances. The range rows are distances in km
            (``coord_scale_m``), not variances (share NaN): ``gp_range`` is the Matern range
            parameter, ``gp_practical_range`` the distance where the GP correlation is 5 %.

        Raises:
            RuntimeError: If the model is not fitted.
        """
        gp_model = self._check_fitted().gp_model
        params = gp_model.get_cov_pars(format_pandas=True).iloc[0].astype(float)
        rows: list[tuple[str, str, float]] = []
        if "Error_var" not in params.index:
            aux = gp_model.get_aux_pars(format_pandas=True).iloc[0].astype(float)
            rows.append(("residual", "error_variance", float(aux["error_variance"])))
        rows += [(self._component_name(str(k)), str(k), float(v)) for k, v in params.items()]
        if "GP_range" in params.index:
            cfg = self.config
            shape = None if cfg.cov_function == "exponential" else cfg.cov_fct_shape
            unit = _PRACTICAL_RANGE.get((cfg.cov_function, shape))
            rows.append(("gp_practical_range", "derived", params["GP_range"] * (unit or np.nan)))
        table = pd.DataFrame(rows, columns=["component", "parameter", "variance"])
        is_var = ~table["component"].str.contains("range")
        total = table.loc[is_var, "variance"].sum()
        table["share"] = np.where(is_var, table["variance"] / total, np.nan)
        return table

    def _gp_model_kwargs(
        self, groups: np.ndarray | None, xy: np.ndarray | None
    ) -> dict[str, object]:
        cfg = self.config
        kwargs: dict[str, object] = {"likelihood": "gaussian", "group_data": groups}
        kwargs |= {"matrix_inversion_method": cfg.matrix_inversion_method, "seed": cfg.seed}
        kwargs["num_parallel_threads"] = cfg.num_threads or None
        if xy is not None:
            kwargs |= {"gp_coords": xy, "cov_function": cfg.cov_function}
            kwargs |= {"cov_fct_shape": cfg.cov_fct_shape, "gp_approx": cfg.gp_approx}
            kwargs["num_neighbors"] = cfg.num_neighbors if cfg.gp_approx != "none" else None
        return kwargs

    def _prepare(
        self, X: pd.DataFrame, group_data: pd.DataFrame | None, coords: Coords | None, *,
        training: bool,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:  # fmt: skip
        """Validate inputs; return features, group keys (object array) and scaled coordinates."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X).__name__}")
        if training:
            self.feature_names_ = list(X.columns)
        if missing := [c for c in self.feature_names_ if c not in X.columns]:
            raise KeyError(f"X lacks training features: {missing}")
        frame = X.loc[:, self.feature_names_]
        if bad := [c for c in frame if not (is_numeric_dtype(frame[c]) or is_bool_dtype(frame[c]))]:
            raise TypeError(f"non-numeric feature columns (encode them first): {bad}")
        groups = self._prepare_groups(group_data, X, training)
        xy = self._prepare_coords(coords, X, training)
        return frame.to_numpy(float, na_value=np.nan), groups, xy

    def _prepare_groups(
        self, group_data: pd.DataFrame | None, X: pd.DataFrame, training: bool
    ) -> np.ndarray | None:
        cols = list(self.config.group_cols)
        if not cols:
            return None
        if group_data is None:
            raise ValueError(f"group_data with columns {cols} is required")
        if missing := [c for c in cols if c not in group_data.columns]:
            raise KeyError(f"group_data lacks columns: {missing}")
        _check_alignment(group_data, X, "group_data")
        frame = group_data[cols].astype("string")
        if n_missing := int(frame.isna().to_numpy().sum()):
            # Training: one shared "missing" level; prediction: a never-seen level -> prior.
            fill = _TRAIN_NA_LEVEL if training else _UNSEEN_LEVEL
            logger.warning("%d missing group keys mapped to level %r", n_missing, fill)
            frame = frame.fillna(fill)
        values = frame.to_numpy(dtype=object)
        if training:
            self.n_levels_ = {c: int(frame[c].nunique()) for c in cols}
            self._train_groups = values
        return values

    def _prepare_coords(
        self, coords: Coords | None, X: pd.DataFrame, training: bool
    ) -> np.ndarray | None:
        if not self.config.use_gp:
            return None
        if coords is None:
            raise ValueError("coords are required when use_gp=True")
        if isinstance(coords, pd.DataFrame):
            _check_alignment(coords, X, "coords")
        xy = as_coord_array(coords, min_rows=2 if training else 0)
        if len(xy) != len(X):
            raise ValueError(f"coords ({len(xy)}) and X ({len(X)}) differ in length")
        if training:
            # Centring keeps the numbers small; distances (and thus the GP) are unaffected.
            self.coord_origin_ = np.floor(xy.mean(axis=0))
        return (xy - self.coord_origin_) / self.config.coord_scale_m

    def _chunk_size(self) -> int:
        size = max(1, self.config.predict_batch_size)
        if self.config.use_gp and self.n_levels_:
            # gpboost 1.7.4 aborts for more prediction points than training RE levels.
            size = min(size, sum(self.n_levels_.values()))
        return size

    def _component_name(self, name: str) -> str:
        if name.startswith("Group_"):
            return self.config.group_cols[int(name.split("_")[1]) - 1].removeprefix("re_")
        return _COMPONENT_NAMES.get(name, name)

    def _check_fitted(self) -> "gpb.Booster":
        if self.booster_ is None:
            raise RuntimeError("GPBoostRegressor is not fitted; call fit() first")
        return self.booster_

    def __getstate__(self) -> dict[str, object]:
        """Pickle the booster as gpboost's JSON model string (plain pickling drops the GPModel)."""
        booster = self.booster_
        return self.__dict__ | {"booster_": None if booster is None else booster.model_to_string()}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore the booster and its GPModel from the JSON model string."""
        self.__dict__.update(state)
        if isinstance(model_str := state.get("booster_"), str):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "gpboost_model.json"
                path.write_text(model_str, encoding="utf-8")
                self.booster_ = _import_gpboost().Booster(model_file=str(path))


def _prepare_target(y_log: np.ndarray | pd.Series, X: pd.DataFrame) -> np.ndarray:
    if isinstance(y_log, pd.Series):
        _check_alignment(y_log, X, "y_log")
    y = np.asarray(y_log, dtype=float).ravel()
    if len(y) != len(X):
        raise ValueError(f"y_log ({len(y)}) and X ({len(X)}) differ in length")
    if not np.isfinite(y).all():
        raise ValueError("y_log contains NaN or infinite values")
    return y


def _check_alignment(obj: pd.DataFrame | pd.Series, X: pd.DataFrame, name: str) -> None:
    if len(obj) != len(X):
        raise ValueError(f"{name} ({len(obj)}) and X ({len(X)}) differ in length")
    if not obj.index.equals(X.index):
        raise ValueError(f"{name} index is not aligned with X.index")
