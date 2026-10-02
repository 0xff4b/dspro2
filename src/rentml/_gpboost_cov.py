"""Stage 1 of the two-stage GPBoost fit: covariance parameters from a linear mixed model.

Estimating the covariance parameters in every boosting round (gpboost's default) took about
12 s per round on 5k DSPRO1 listings, and with gpboost's default starting values the error
variance collapsed to ~5e-7 because many listings share building coordinates (train MAE 0,
80 % intervals covering 62 %). Both were verified on the DSPRO1 data. The fix is to estimate the
covariance parameters once, with data-driven starting values, in a linear mixed model that has
the same random-effect structure, and to keep them fixed while boosting.
"""

import logging
import time
import warnings
from dataclasses import dataclass
from types import ModuleType

import numpy as np
from scipy.linalg import qr

logger = logging.getLogger(__name__)

_INIT_ERROR_SHARE = 0.5
_INIT_EFFECT_SHARE = 0.1
_INIT_RANGE_SHARE = 0.03


@dataclass(frozen=True)
class CovEstimate:
    """Covariance parameters of the random-effect/GP structure (log scale).

    Attributes:
        cov_pars: Values in gpboost order (see ``names``).
        names: gpboost names, e.g. ``("Group_1", "Group_2", "Group_3", "GP_var", "GP_range")``.
        aux_pars: Error variance when gpboost treats it as an auxiliary parameter, else None.
        fit_time_s: Wall-clock time of the estimation in seconds.
    """

    cov_pars: np.ndarray
    names: tuple[str, ...]
    aux_pars: np.ndarray | None
    fit_time_s: float


def uses_aux_error(n_groups: int, use_gp: bool, gp_approx: str) -> bool:
    """Whether gpboost 1.7.4 models the error variance as an auxiliary parameter.

    This is the case for grouped random effects combined with a Vecchia GP (Laplace/latent
    formulation); otherwise ``Error_var`` is the first covariance parameter (verified).

    Args:
        n_groups: Number of grouped random-effect columns.
        use_gp: Whether a GP on the coordinates is part of the model.
        gp_approx: gpboost GP approximation.

    Returns:
        True if the error variance is an auxiliary parameter.
    """
    return n_groups > 0 and use_gp and gp_approx == "vecchia"


def expected_names(n_groups: int, use_gp: bool, gp_approx: str) -> tuple[str, ...]:
    """Covariance parameter names in gpboost order for a model structure.

    Args:
        n_groups: Number of grouped random-effect columns.
        use_gp: Whether a GP on the coordinates is part of the model.
        gp_approx: gpboost GP approximation.

    Returns:
        The parameter names.
    """
    names = [] if uses_aux_error(n_groups, use_gp, gp_approx) else ["Error_var"]
    names += [f"Group_{i + 1}" for i in range(n_groups)]
    return tuple(names + (["GP_var", "GP_range"] if use_gp else []))


def linear_design(x: np.ndarray) -> np.ndarray:
    """Intercept plus standardised, median-imputed features without collinear columns.

    Args:
        x: Feature matrix (NaN allowed).

    Returns:
        Full-rank design matrix whose first column is the intercept.
    """
    with warnings.catch_warnings():
        # All-NaN columns are expected (dropped below as constant); their warnings are noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        center = np.nan_to_num(np.nanmedian(x, axis=0)) if x.size else np.zeros(x.shape[1])
        scale = np.nanstd(x, axis=0) if x.size else np.ones(x.shape[1])
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    z = (np.where(np.isnan(x), center, x) - center) / scale
    design = np.column_stack([np.ones(len(x)), z])
    # Pivoted QR keeps a maximal set of linearly independent columns (e.g. year_built vs age).
    _, r, piv = qr(design, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    rank = int((diag > 1e-8 * diag[0]).sum()) if diag.size else 0
    keep = np.sort(piv[:rank])
    if 0 not in keep:
        keep = np.sort(np.r_[0, keep[: max(rank - 1, 0)]])
    return design[:, keep]


def initial_values(
    y: np.ndarray, design: np.ndarray, xy: np.ndarray | None, n_groups: int, aux: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    """Data-driven starting values scaled by the OLS residual variance.

    Args:
        y: Target (log scale).
        design: Design matrix of the linear fixed effects.
        xy: Scaled GP coordinates, or None without a GP.
        n_groups: Number of grouped random-effect columns.
        aux: Whether the error variance is an auxiliary parameter.

    Returns:
        ``(init_cov_pars, init_aux_pars)``; the second is None when not needed.
    """
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    s2 = float(np.var(y - design @ beta)) or float(np.var(y)) or 1.0
    pars = [_INIT_EFFECT_SHARE * s2] * n_groups
    if xy is not None:
        extent = float(np.ptp(xy, axis=0).max()) if len(xy) else 1.0
        pars += [_INIT_EFFECT_SHARE * s2, max(_INIT_RANGE_SHARE * extent, 1e-3)]
    error = [_INIT_ERROR_SHARE * s2]
    if aux:
        return np.array(pars), np.array(error)
    return np.array(error + pars), None


def estimate_cov_pars(
    gpboost: ModuleType,
    model_kwargs: dict[str, object],
    x: np.ndarray,
    y: np.ndarray,
    xy: np.ndarray | None,
    *,
    n_groups: int,
    gp_approx: str,
    delta_rel_conv: float,
    maxit: int,
) -> CovEstimate:
    """Estimate covariance parameters with a linear mixed model (fixed effects linear in ``x``).

    Args:
        gpboost: The imported gpboost module.
        model_kwargs: ``GPModel`` keyword arguments (random-effect/GP structure).
        x: Fixed-effect features (NaN allowed).
        y: Target (log scale).
        xy: Scaled GP coordinates, or None without a GP.
        n_groups: Number of grouped random-effect columns.
        gp_approx: gpboost GP approximation.
        delta_rel_conv: Convergence tolerance of the optimiser.
        maxit: Maximum optimiser iterations.

    Returns:
        The estimated covariance (and auxiliary) parameters.
    """
    aux = uses_aux_error(n_groups, xy is not None, gp_approx)
    design = linear_design(x)
    init_cov, init_aux = initial_values(y, design, xy, n_groups, aux)
    params: dict[str, object] = {"init_cov_pars": init_cov, "delta_rel_conv": delta_rel_conv}
    params["maxit"] = maxit
    if init_aux is not None:
        params["init_aux_pars"] = init_aux
    model = gpboost.GPModel(**model_kwargs)
    model.set_optim_params(params=params)
    start = time.perf_counter()
    model.fit(y=y, X=design)
    elapsed = time.perf_counter() - start
    table = model.get_cov_pars(format_pandas=True)
    aux_pars = model.get_aux_pars(format_pandas=True).iloc[0].to_numpy(float) if aux else None
    logger.info("Covariance parameters estimated in %.1f s", elapsed)
    return CovEstimate(
        table.iloc[0].to_numpy(float), tuple(map(str, table.columns)), aux_pars, elapsed
    )
