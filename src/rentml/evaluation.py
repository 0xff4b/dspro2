"""Metrics, bootstrap inference, paired model comparison, power analysis and learning curves.

All metrics are on the CHF scale (back-transform log predictions with ``np.exp`` first). The
paired bootstrap CI of the MAE difference is the primary evidence of the pre-registered plan,
the Wilcoxon signed-rank test on the paired absolute errors the rank-based secondary check;
p-values across ablation stages are Holm-corrected. ``groups=object_id`` resamples whole objects
(Wilcoxon then uses per-object mean differences). ``y_true``/``y_pred`` align by position.
"""

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

from rentml.config import ALPHA_TEST, N_BOOTSTRAP, RANDOM_STATE
from rentml.splits import group_codes

logger = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray], float]

DEFAULT_REL_EFFECTS: tuple[float, ...] = tuple(round(0.01 * k, 2) for k in range(1, 11))
DEFAULT_ERROR_CORR = 0.8


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _medae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.median(np.abs(y - p)))


def _mape(y: np.ndarray, p: np.ndarray) -> float:
    mask = y != 0
    return float(100.0 * np.mean(np.abs(y - p)[mask] / np.abs(y[mask]))) if mask.any() else np.nan


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot if ss_tot > 0 else np.nan


METRICS: Mapping[str, MetricFn] = MappingProxyType(
    {"MAE": _mae, "RMSE": _rmse, "MedAE": _medae, "MAPE": _mape, "R2": _r2}
)


@dataclass
class PairedComparison:
    """Paired comparison of model A (reference) and model B (challenger) on the same listings.

    ``diff = mae_a - mae_b`` in CHF (> 0: B is better) with bootstrap CI ``[ci_low, ci_high]``;
    ``p_bootstrap``: two-sided centred-bootstrap p-value (NaN if uninformative); ``p_wilcoxon``:
    Wilcoxon signed-rank p-value on ``|e_a| - |e_b|`` (per-object means if clustered); ``n``.
    """

    name_a: str
    name_b: str
    mae_a: float
    mae_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_bootstrap: float
    p_wilcoxon: float
    n: int

    @property
    def rel_diff(self) -> float:
        """Relative MAE reduction of B vs A (``diff / mae_a``)."""
        return self.diff / self.mae_a if self.mae_a else np.nan


def regression_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """Point metrics in CHF: MAE, RMSE, MedAE, MAPE (%) and R2.

    Args:
        y_true: Observed rents in CHF.
        y_pred: Predicted rents in CHF.

    Returns:
        Dict ``MAE``, ``RMSE``, ``MedAE``, ``MAPE`` (zero targets ignored), ``R2`` (NaN if y const).

    Raises:
        ValueError: If the inputs are empty, differ in length or are not finite.
    """
    y, p = _as_pair(y_true, y_pred)
    return {name: fn(y, p) for name, fn in METRICS.items()}


def segment_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, segments: pd.Series | npt.ArrayLike
) -> pd.DataFrame:
    """Error metrics per segment (region, price band, listings-per-municipality bucket, ...).

    Args:
        y_true: Observed rents in CHF.
        y_pred: Predicted rents in CHF.
        segments: Segment label per listing; missing labels form their own segment and the
            categorical order is kept. Aligned by label if ``y_true`` and ``segments`` are
            both Series (same labels required), else by position (e.g. an array).

    Returns:
        Frame indexed by segment with ``n``, ``MAE``, ``RMSE``, ``MAPE`` (%) and ``bias``
        (mean of prediction minus truth; negative means under-prediction).

    Raises:
        ValueError: If the lengths or labels differ or the inputs are not finite.
    """
    y, p = _as_pair(y_true, y_pred)
    seg = _aligned_segments(y_true, segments, len(y))
    err = p - y
    ape = np.abs(err) / np.where(y != 0, np.abs(y), np.nan)  # zero targets excluded
    work = pd.DataFrame({"abs": np.abs(err), "sq": err**2, "ape": ape, "err": err})
    grouped = work.groupby(seg, observed=True, dropna=False)
    out = grouped.mean().rename(columns={"abs": "MAE", "sq": "RMSE", "ape": "MAPE", "err": "bias"})
    out["RMSE"], out["MAPE"] = np.sqrt(out["RMSE"]), 100.0 * out["MAPE"]
    out.insert(0, "n", grouped.size())
    out.index.name = seg.name if seg.name is not None else "segment"
    return out.astype({"n": int})


def bootstrap_ci(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    metric: MetricFn | str,
    *,
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = RANDOM_STATE,
    groups: npt.ArrayLike | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of any metric.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.
        metric: Callable ``metric(y_true, y_pred) -> float`` or a key of :data:`METRICS`.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level.
        seed: Random seed.
        groups: Optional cluster labels (e.g. ``object_id``); whole clusters are resampled.

    Returns:
        ``(point_estimate, ci_low, ci_high)``.

    Raises:
        ValueError: For invalid ``ci``, ``n_boot`` or inputs, or fewer than 2 listings/clusters.
        KeyError: If ``metric`` is an unknown metric name.
    """
    y, p = _as_pair(y_true, y_pred)
    _check_boot(n_boot, ci)
    fn = METRICS[metric] if isinstance(metric, str) else metric
    draws = _cluster_draws(_cluster_codes(groups, len(y)), np.random.default_rng(seed), n_boot)
    boot = np.array([fn(y[rows], p[rows]) for rows in draws])
    lo, hi = np.nanquantile(boot, [(1 - ci) / 2, (1 + ci) / 2])
    return float(fn(y, p)), float(lo), float(hi)


def paired_bootstrap_mae(
    y_true: npt.ArrayLike,
    pred_a: npt.ArrayLike,
    pred_b: npt.ArrayLike,
    *,
    name_a: str = "A",
    name_b: str = "B",
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
    ci: float = 0.95,
    groups: npt.ArrayLike | None = None,
) -> PairedComparison:
    """Paired bootstrap of the MAE difference plus a Wilcoxon signed-rank check.

    Listings (or ``groups`` clusters) are resampled jointly for both models (paired CI). The
    two-sided centred-bootstrap p-value is ``(1 + #{|d*_b - d| >= |d|}) / (n_boot + 1)``, NaN if
    all (cluster) differences are equal and non-zero. Wilcoxon uses per-cluster mean differences.

    Args:
        y_true: Observed rents in CHF.
        pred_a: Predictions of model A (reference) in CHF.
        pred_b: Predictions of model B (challenger) in CHF.
        name_a: Name of model A.
        name_b: Name of model B.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.
        ci: Confidence level of the CI.
        groups: Optional cluster labels (e.g. ``object_id``), see above.

    Returns:
        The :class:`PairedComparison` (``diff > 0`` means B is better).

    Raises:
        ValueError: If the inputs are invalid or there are fewer than 2 listings/clusters.
    """
    y, pa = _as_pair(y_true, pred_a)
    _, pb = _as_pair(y_true, pred_b)
    _check_boot(n_boot, ci)
    codes = _cluster_codes(groups, len(y))
    d = np.abs(y - pa) - np.abs(y - pb)
    diff = float(d.mean())
    d_cluster = np.bincount(codes, weights=d) / np.bincount(codes)
    draws = _cluster_draws(codes, np.random.default_rng(seed), n_boot)
    boot = np.array([d[rows].mean() for rows in draws])
    lo, hi = (float(v) for v in np.quantile(boot, [(1 - ci) / 2, (1 + ci) / 2]))
    p_boot = (1 + int(np.sum(np.abs(boot - diff) >= abs(diff) - 1e-12))) / (n_boot + 1)
    if diff != 0 and np.ptp(d_cluster) <= 1e-9 * max(1.0, float(np.abs(d_cluster).max())):
        p_boot = np.nan  # every resample equals diff: the bootstrap has no information
    p_w, mae_a, mae_b = _wilcoxon_p(d_cluster), _mae(y, pa), _mae(y, pb)
    return PairedComparison(name_a, name_b, mae_a, mae_b, diff, lo, hi, p_boot, p_w, len(y))


def holm_correction(pvalues: dict[str, float], alpha: float = ALPHA_TEST) -> pd.DataFrame:
    """Holm step-down correction of a family of p-values.

    Args:
        pvalues: Mapping hypothesis name -> raw p-value (NaN is excluded from the family).
        alpha: Family-wise error rate.

    Returns:
        Frame indexed by hypothesis (input order) with ``p``, ``p_holm`` (monotone in the
        ordered p-values, capped at 1) and ``reject`` (``p_holm <= alpha``).

    Raises:
        ValueError: If a p-value lies outside [0, 1].
    """
    names, p = list(pvalues), np.array(list(pvalues.values()), dtype=float)
    adjusted = _holm_adjust(p)
    reject = np.nan_to_num(adjusted, nan=1.0) <= alpha
    index = pd.Index(names, name="hypothesis")
    return pd.DataFrame({"p": p, "p_holm": adjusted, "reject": reject}, index=index)


def comparison_table(results: list[PairedComparison], *, alpha: float = ALPHA_TEST) -> pd.DataFrame:
    """Tabulate paired comparisons, Holm-corrected on the (primary) bootstrap p-values.

    Args:
        results: Paired comparisons (one per ablation stage or model pair).
        alpha: Family-wise error rate for ``reject``.

    Returns:
        One row per comparison: ``comparison``, the dataclass fields, ``rel_diff_pct``,
        ``p_holm`` and ``reject``.
    """
    rows = [{**asdict(r), "rel_diff_pct": 100 * r.rel_diff} for r in results]
    cols = [*(f.name for f in fields(PairedComparison)), "rel_diff_pct"]
    table = pd.DataFrame(rows, columns=cols)
    table.insert(0, "comparison", table["name_a"] + " vs " + table["name_b"])
    table["p_holm"] = _holm_adjust(table["p_bootstrap"].to_numpy(dtype=float))
    table["reject"] = table["p_holm"].fillna(1.0) <= alpha
    return table


def power_mde(
    abs_err_a: npt.ArrayLike,
    *,
    test_sizes: Sequence[int] = (2000, 3000, 4000, 5000, 6000),
    rel_effects: Sequence[float] = DEFAULT_REL_EFFECTS,
    n_sim: int = 500,
    alpha: float = ALPHA_TEST,
    seed: int = RANDOM_STATE,
    error_corr: float | None = None,
    abs_err_b: npt.ArrayLike | None = None,
    target_power: float = 0.8,
) -> pd.DataFrame:
    """Simulated power of the paired MAE comparison and minimum detectable effect (MDE).

    Simulation for test size ``n``, relative effect ``delta`` and each of ``n_sim`` runs:

    1. ``e_A``: ``n`` absolute errors resampled with replacement from ``abs_err_a``.
    2. ``e_B,i = (1 - delta) * (w * e_A,i + (1 - w) * e'_i)``, ``e'_i`` an independent draw
       from the pool, so ``E[e_B] = (1 - delta) * E[e_A]``. ``w = 1 - r / sqrt(2)`` reproduces
       ``r = sd(e_A - e_B) / sd(e_A)`` (what drives power): empirical if ``abs_err_b`` is given
       (recommended), else ``sqrt(2 (1 - rho))`` with ``rho = error_corr`` or 0.8 (WARNING).
    3. Normal approximation of the centred paired bootstrap, ``z = mean(d) / (sd(d) / sqrt(n))``
       with ``d = e_A - e_B``; detection if ``p < alpha`` and ``mean(d) > 0``.

    Args:
        abs_err_a: Per-listing absolute errors of the reference model in CHF (e.g. DSPRO1).
        test_sizes: Test-set sizes to simulate.
        rel_effects: Relative MAE reductions of model B (0.05 = 5 %).
        n_sim: Simulation runs per cell.
        alpha: Two-sided significance level.
        seed: Random seed.
        error_corr: Correlation of the two models' absolute errors (overrides ``abs_err_b``).
        abs_err_b: Absolute errors of a second model on the same listings (recommended).
        target_power: Power that defines the MDE.

    Returns:
        Long frame with ``test_size``, ``rel_effect``, ``effect_chf``, ``power``, ``mcse``
        (Monte-Carlo SE), ``mde_rel``/``mde_chf`` (smallest grid effect with power >=
        ``target_power`` per test size, NaN if none: use a fine grid), ``rho`` (error
        correlation used; negative clipped to 0) and ``pair_sd_ratio`` (simulated ``r``).

    Raises:
        ValueError: For invalid error pools (non-finite, negative, constant ``abs_err_b``),
            test sizes, effects, ``n_sim`` or correlation.
    """
    pool = _abs_errors(abs_err_a, "abs_err_a")
    effects = np.asarray(rel_effects, dtype=float)
    if min(test_sizes, default=0) < 2 or n_sim < 1 or np.any((effects < 0) | (effects >= 1)):
        raise ValueError("need test_sizes >= 2, n_sim >= 1 and rel_effects in [0, 1)")
    w, rho = _pairing_weight(pool, error_corr, abs_err_b)
    rng = np.random.default_rng(seed)
    power = []
    for n in test_sizes:
        e_a = pool[rng.integers(0, pool.size, size=(n_sim, n))]
        mix = w * e_a + (1.0 - w) * pool[rng.integers(0, pool.size, size=(n_sim, n))]
        power.append(_detection_rate(e_a, mix, effects, float(stats.norm.ppf(1 - alpha / 2))))
    grid = pd.MultiIndex.from_product([test_sizes, effects], names=["test_size", "rel_effect"])
    out = pd.DataFrame({"power": np.concatenate(power)}, index=grid).reset_index()
    out.insert(2, "effect_chf", out["rel_effect"] * pool.mean())
    out["mcse"] = np.sqrt(out["power"] * (1 - out["power"]) / n_sim)
    mde = out[out["power"] >= target_power].groupby("test_size")["rel_effect"].min()
    out["mde_rel"] = out["test_size"].map(mde)
    out["mde_chf"] = out["mde_rel"] * pool.mean()
    out["rho"], out["pair_sd_ratio"] = rho, (1.0 - w) * np.sqrt(2.0)
    logger.info("Power simulation (rho=%.2f, w=%.3f): relative MDE %s", rho, w, mde.to_dict())
    return out


def learning_curve(
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_rows: int,
    *,
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    y: npt.ArrayLike,
    seed: int = RANDOM_STATE,
    groups: npt.ArrayLike | None = None,
) -> pd.DataFrame:
    """Validation error as a function of the training-set size.

    Per fold the training units (rows, or whole groups such as objects if ``groups`` is given)
    are permuted once (seeded) and the first ``fraction`` of them is used, so subsets are nested;
    validation folds stay complete.

    Args:
        fit_predict: ``fit_predict(train_positions, val_positions)`` fits a fresh model on the
            training rows and returns validation predictions on the scale of ``y`` (CHF).
        n_rows: Number of rows the positions refer to.
        fractions: Training fractions in (0, 1] (of rows, or of groups).
        folds: ``(train_positions, val_positions)`` tuples, e.g. from ``grouped_cv_folds``.
        y: Targets for all ``n_rows`` rows.
        seed: Random seed for the subsampling.
        groups: Optional group label per row (e.g. ``object_id``); whole groups are sampled.

    Returns:
        One row per fraction: ``n_train_mean``, ``n_groups_mean`` (training units), ``mae_mean``,
        ``mae_std``, ``rmse_mean``, ``rmse_std`` (across folds) and ``n_folds``.

    Raises:
        ValueError: For wrong lengths of ``y``/``groups``, empty ``folds`` or training folds,
            positions outside ``[0, n_rows)`` or fractions outside (0, 1].
    """
    target = np.asarray(y, dtype=float).ravel()
    if target.size != n_rows or not folds or any(not 0 < f <= 1 for f in fractions):
        raise ValueError(f"need len(y) == n_rows={n_rows}, folds and fractions in (0, 1]")
    codes = np.arange(n_rows) if groups is None else _cluster_codes(groups, n_rows)
    rng = np.random.default_rng(seed)
    records = []
    for k, (train_pos, val_pos) in enumerate(folds):
        train_pos, val_pos = np.asarray(train_pos).ravel(), np.asarray(val_pos).ravel()
        both = np.concatenate([train_pos, val_pos])
        if train_pos.size == 0 or both.min() < 0 or both.max() >= n_rows:
            raise ValueError(f"fold {k}: empty training part or positions outside [0, {n_rows})")
        units, inverse = np.unique(codes[train_pos], return_inverse=True)
        rank = rng.permutation(len(units))[inverse]  # random rank of each row's unit
        for frac in fractions:
            n_units = max(1, round(frac * len(units)))
            sub = np.sort(train_pos[rank < n_units])
            y_val, pred = _as_pair(target[val_pos], fit_predict(sub, val_pos))
            records.append((float(frac), len(sub), n_units, _mae(y_val, pred), _rmse(y_val, pred)))
    table = pd.DataFrame(records, columns=["fraction", "n_train", "n_groups", "mae", "rmse"])
    out = table.groupby("fraction").agg(["mean", "std"])
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    out = out.drop(columns=["n_train_std", "n_groups_std"])
    return out.assign(n_folds=len(folds)).reset_index()


def _cluster_draws(codes: np.ndarray, rng: np.random.Generator, n: int) -> Iterator[np.ndarray]:
    """Yield ``n`` bootstrap row-position arrays that resample whole clusters (codes 0..k-1)."""
    order = np.argsort(codes, kind="stable")
    sizes = np.bincount(codes)
    starts = np.cumsum(sizes) - sizes
    for _ in range(n):
        clusters = rng.integers(0, len(sizes), len(sizes))
        lengths = sizes[clusters]
        within = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        yield order[np.repeat(starts[clusters], lengths) + within]


def _cluster_codes(groups: npt.ArrayLike | None, n: int) -> np.ndarray:
    """Cluster codes 0..k-1 (one per row without ``groups``); needs at least two clusters."""
    codes = np.arange(n) if groups is None else group_codes(groups)
    if len(codes) != n:
        raise ValueError(f"groups has {len(codes)} values, expected {n}")
    if n == 0 or codes.max() < 1:
        raise ValueError("need at least 2 listings (clusters) for a bootstrap")
    return codes


def _detection_rate(e_a: np.ndarray, mix: np.ndarray, effects: np.ndarray, z: float) -> np.ndarray:
    """Share of simulation rows (axis 0) where the paired z-test detects B < A, per effect."""
    power = np.empty(len(effects))
    for j, delta in enumerate(effects):
        d = e_a - (1.0 - delta) * mix
        mean, se = d.mean(axis=1), d.std(axis=1, ddof=1) / np.sqrt(d.shape[1])
        stat = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
        power[j] = np.mean((np.abs(stat) > z) & (mean > 0))
    return power


def _wilcoxon_p(d: np.ndarray) -> float:
    if not np.any(d != 0):
        return 1.0  # identical errors: no evidence against H0 (scipy would raise)
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError as exc:
        logger.warning("Wilcoxon test failed: %s", exc)
        return np.nan


def _holm_adjust(p: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(p)
    if np.any((p[valid] < 0) | (p[valid] > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    adjusted, m = np.full(p.shape, np.nan), int(valid.sum())
    order = np.flatnonzero(valid)[np.argsort(p[valid], kind="stable")]
    adjusted[order] = np.minimum(np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0)
    return adjusted


def _abs_errors(values: npt.ArrayLike, name: str) -> np.ndarray:
    errors = np.asarray(values, dtype=float).ravel()
    if errors.size < 2 or not np.all(np.isfinite(errors)) or np.any(errors < 0):
        raise ValueError(f"{name} needs >= 2 finite, non-negative errors")
    return errors


def _pairing_weight(
    pool: np.ndarray, corr: float | None, other: npt.ArrayLike | None
) -> tuple[float, float]:
    """Mixture weight ``w`` reproducing ``sd(e_A - e_B) / sd(e_A)``, and the error correlation."""
    if corr is None and other is None:
        corr = DEFAULT_ERROR_CORR
        logger.warning("power_mde: no abs_err_b/error_corr given, assuming rho=%.2f", corr)
    if corr is not None:
        if not 0 <= corr <= 1:
            raise ValueError(f"error correlation must lie in [0, 1], got {corr}")
        rho, ratio_sq = float(corr), 2.0 * (1.0 - corr)  # equally dispersed pair of models
    else:
        errors_b = _abs_errors(other, "abs_err_b")
        if errors_b.shape != pool.shape or np.var(pool) == 0 or np.var(errors_b) == 0:
            raise ValueError("abs_err_b must match abs_err_a in length; both need variance > 0")
        rho = float(np.corrcoef(pool, errors_b)[0, 1])
        ratio_sq = float(np.var(pool - errors_b) / np.var(pool))
    w = 1.0 - np.sqrt(ratio_sq / 2.0)
    if w < 0 or rho < 0:
        logger.warning("power_mde: rho=%.2f; clipped to independent pairing (w=0)", rho)
        w, rho = 0.0, max(rho, 0.0)
    return float(w), rho


def _as_pair(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_pred, dtype=float).ravel()
    if y.shape != p.shape or y.size == 0:
        raise ValueError(f"y_true/y_pred must be non-empty, equal length ({y.size}, {p.size})")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        raise ValueError("y_true and y_pred must be finite (no NaN/inf)")
    return y, p


def _aligned_segments(y_true: npt.ArrayLike, segments: npt.ArrayLike, n: int) -> pd.Series:
    """Segments as a positional Series; label-aligned to ``y_true`` if both are Series."""
    seg = segments if isinstance(segments, pd.Series) else pd.Series(segments)
    if len(seg) != n:
        raise ValueError(f"segments has {len(seg)} rows, expected {n}")
    labelled = isinstance(y_true, pd.Series) and isinstance(segments, pd.Series)
    if labelled and not y_true.index.equals(seg.index):
        idx = y_true.index
        if not (idx.is_unique and seg.index.is_unique and idx.isin(seg.index).all()):
            raise ValueError("segments.index must match y_true.index (or pass y_true as array)")
        seg = seg.reindex(idx)
    return seg.reset_index(drop=True)


def _check_boot(n_boot: int, ci: float) -> None:
    if n_boot < 1 or not 0 < ci < 1:
        raise ValueError(f"need n_boot >= 1 and 0 < ci < 1, got {n_boot=}, {ci=}")
