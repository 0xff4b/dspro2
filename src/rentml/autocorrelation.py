"""Spatial autocorrelation diagnostics: k-NN weights and Moran's I with a permutation test."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from rentml.config import RANDOM_STATE

Coords = pd.DataFrame | np.ndarray
_PERMUTATION_BATCH = 256


def as_coord_array(coords: Coords, min_rows: int = 2) -> np.ndarray:
    """Validate coordinates and return them as a float ``(n, 2)`` array.

    Args:
        coords: Array-like of shape ``(n, 2)``.
        min_rows: Minimum number of rows required.

    Returns:
        The coordinates as a float array.

    Raises:
        ValueError: If the shape is wrong, too few rows are given or values are not finite.
    """
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {xy.shape}")
    if xy.shape[0] < min_rows:
        raise ValueError(f"at least {min_rows} observations are required")
    if not np.isfinite(xy).all():
        raise ValueError("coords contain NaN or infinite entries")
    return xy


def knn_weights(coords: np.ndarray, k: int = 8) -> sparse.csr_matrix:
    """Build a row-standardised k-nearest-neighbour spatial weights matrix.

    Each point gets weight ``1 / k`` for its ``k`` nearest neighbours; a point is never its own
    neighbour, even when others share its exact coordinates (listings in the same building).

    Args:
        coords: Array of shape ``(n, 2)`` with projected coordinates (e.g. LV95 metres).
        k: Number of neighbours per point.

    Returns:
        Sparse ``(n, n)`` CSR matrix whose rows sum to 1.

    Raises:
        ValueError: If ``coords`` is not a finite ``(n, 2)`` array or ``k`` not in [1, n - 1].
    """
    xy = as_coord_array(coords)
    n = xy.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be between 1 and n - 1 = {n - 1}, got {k}")
    # kneighbors() without X excludes each query point by index, so duplicates are fine.
    neighbours = NearestNeighbors(n_neighbors=k).fit(xy).kneighbors(return_distance=False)
    data = np.full(n * k, 1.0 / k)
    return sparse.csr_matrix((data, (np.repeat(np.arange(n), k), neighbours.ravel())), (n, n))


@dataclass
class MoranResult:
    """Result of a Moran's I permutation test.

    Attributes:
        I: Observed Moran's I.
        expected: Expectation under no autocorrelation, ``-1 / (n - 1)``.
        z: ``(I - mean(I_perm)) / sd(I_perm)`` (NaN without permutations).
        p_value: Two-sided permutation p-value (NaN without permutations).
        k: Neighbours per point in the weights matrix.
        permutations: Number of random permutations.
        n: Number of observations.
    """

    I: float  # noqa: E741 -- conventional symbol of the statistic
    expected: float
    z: float
    p_value: float
    k: int
    permutations: int
    n: int = 0


def morans_i(
    values: np.ndarray,
    coords: np.ndarray,
    *,
    k: int = 8,
    permutations: int = 999,
    seed: int = RANDOM_STATE,
) -> MoranResult:
    """Moran's I with row-standardised k-NN weights and a two-sided permutation test.

    ``I = (n / S0) z'Wz / z'z`` with ``z`` the centred values and ``S0`` the weight sum. The
    p-value is ``(1 + #{|I_perm - E| >= |I - E|}) / (1 + permutations)``, ``E = -1 / (n - 1)``.

    Args:
        values: 1-D values, e.g. test residuals on the log scale.
        coords: ``(n, 2)`` projected coordinates.
        k: Number of nearest neighbours.
        permutations: Number of random permutations (0 skips the test).
        seed: Seed of the permutation generator.

    Returns:
        Statistic, expectation, permutation z-score and p-value.

    Raises:
        ValueError: On length mismatch, non-finite or constant values, or invalid arguments.
    """
    x = np.asarray(values, dtype=float).ravel()
    xy = as_coord_array(coords)
    n = x.shape[0]
    if xy.shape[0] != n:
        raise ValueError(f"values ({n}) and coords ({xy.shape[0]}) differ in length")
    if not np.isfinite(x).all():
        raise ValueError("values contain NaN or infinite entries")
    if permutations < 0:
        raise ValueError(f"permutations must be >= 0, got {permutations}")
    w = knn_weights(xy, k=k)
    z = x - x.mean()
    denom = float(z @ z)
    if denom <= 0.0:
        raise ValueError("values are constant; Moran's I is undefined")
    scale = n / float(w.sum())
    observed, expected = scale * float(z @ (w @ z)) / denom, -1.0 / (n - 1)
    if permutations == 0:
        return MoranResult(observed, expected, float("nan"), float("nan"), k, 0, n)
    rng = np.random.default_rng(seed)
    simulated = np.empty(permutations)
    for start in range(0, permutations, _PERMUTATION_BATCH):
        size = min(_PERMUTATION_BATCH, permutations - start)
        batch = rng.permuted(np.repeat(z[:, None], size, axis=1), axis=0)  # one column per draw
        simulated[start : start + size] = np.einsum("ij,ij->j", batch, w @ batch) * scale / denom
    extreme = np.abs(simulated - expected) >= abs(observed - expected)
    p_value = (1.0 + float(extreme.sum())) / (1.0 + permutations)
    sd = float(simulated.std(ddof=1)) if permutations > 1 else float("nan")
    z_score = (observed - float(simulated.mean())) / sd if sd > 0 else float("nan")
    return MoranResult(observed, expected, z_score, p_value, k, permutations, n)
