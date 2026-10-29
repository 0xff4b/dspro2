"""Metrics, paired model comparison, power analysis, Moran's I, plot style and run tracking.

All metrics are on the CHF scale (back-transform log predictions with ``np.exp`` first). The
paired bootstrap CI of the MAE difference is the primary evidence of the pre-registered plan,
the Wilcoxon signed-rank test on the paired absolute errors the secondary check; p-values across
ablation stages are Holm-corrected. After deduplication one row is one object, so plain
row-level resampling is the object-level bootstrap.
"""

import logging
import os
import re
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

import esda
import libpysal
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from scipy import stats
from sklearn import metrics as skm
from sqlalchemy.exc import SADeprecationWarning
from statsmodels.stats.multitest import multipletests

from rentml.config import ALPHA_TEST, FIGURES, MLRUNS, N_BOOTSTRAP, RANDOM_STATE

logger = logging.getLogger(__name__)

EXPERIMENT = "dspro2-rent"
TRACKING_URI = f"sqlite:///{(MLRUNS / 'mlflow.db').as_posix()}"

METRICS: dict[str, Callable[[npt.ArrayLike, npt.ArrayLike], float]] = {
    "MAE": skm.mean_absolute_error,
    "RMSE": skm.root_mean_squared_error,
    "MedAE": skm.median_absolute_error,
    "MAPE": lambda y, p: 100.0 * skm.mean_absolute_percentage_error(y, p),
    "R2": skm.r2_score,
}

# Palette and rcParams copied from the DSPRO1 plot style so both projects' figures match; the
# light tints are the app's interval bands and the grey of map units without data.
COLORS: dict[str, str] = {
    "primary": "#0077C8", "secondary": "#66B5E9", "reference": "#D0384B", "dark": "#153652",
    "teal": "#16856F", "orange": "#C88126", "purple": "#6861B4", "gray": "#718196",
    "neutral": "#192B3A", "soft": "#DCEBF8", "mid": "#8EC3EC", "no_data": "#E3E1DC",
}  # fmt: skip
PALETTE: list[str] = [
    COLORS[k]
    for k in ("primary", "secondary", "teal", "orange", "purple", "gray", "dark", "neutral")
]
_RC_PARAMS: dict[str, object] = {
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": "#111A28",
    "figure.figsize": (8, 4.8), "figure.dpi": 110, "figure.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF", "savefig.dpi": 160, "savefig.bbox": "tight",
    "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#C8DCEC", "axes.labelcolor": "#4C6280",
    "axes.titlecolor": "#111A28", "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 16, "axes.labelsize": 10, "axes.labelpad": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "grid.color": "#D6E7F5", "grid.alpha": 0.65, "grid.linewidth": 0.6,
    "xtick.color": "#4C6280", "ytick.color": "#4C6280", "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "patch.edgecolor": "#FFFFFF", "pdf.fonttype": 42,
    "svg.fonttype": "none",  # text stays editable in the poster
    "svg.hashsalt": "rentml",  # stable SVG element ids: re-runs create no git noise
}  # fmt: skip


@dataclass(frozen=True)
class PairedComparison:
    """MAE of reference A and challenger B on the same rows; ``diff > 0`` means B is better."""

    mae_a: float
    mae_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_bootstrap: float
    p_wilcoxon: float


@dataclass(frozen=True)
class MoranResult:
    """Moran's I with its expectation under randomness, permutation z-score and p-value."""

    I: float  # noqa: E741 -- conventional symbol of the statistic
    expected: float
    z: float
    p_value: float


def regression_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> dict[str, float]:
    """MAE, RMSE, MedAE, MAPE (%) and R2 in CHF."""
    return {name: float(fn(y_true, y_pred)) for name, fn in METRICS.items()}


def segment_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, segments: pd.Series | npt.ArrayLike
) -> pd.DataFrame:
    """Per-segment ``n``, MAE, RMSE, MAPE (%) and bias (mean of prediction minus truth).

    Inputs align by position; a categorical ``segments`` keeps its category order.
    """
    y = np.asarray(y_true, dtype=float)
    err = np.asarray(y_pred, dtype=float) - y
    seg = pd.Series(segments).reset_index(drop=True)
    work = pd.DataFrame(
        {"MAE": np.abs(err), "RMSE": err**2, "MAPE": 100 * np.abs(err) / y, "bias": err}
    )
    grouped = work.groupby(seg.rename(seg.name or "segment"), observed=True, dropna=False)
    out = grouped.mean()
    out["RMSE"] = np.sqrt(out["RMSE"])
    out.insert(0, "n", grouped.size())
    return out


def bootstrap_ci(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    metric: str = "MAE",
    *,
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of a metric from :data:`METRICS`.

    Returns:
        ``(point_estimate, ci_low, ci_high)``.
    """
    fn = METRICS[metric]
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    res = stats.bootstrap(
        (y, p), fn, paired=True, vectorized=False, n_resamples=n_boot, confidence_level=ci,
        method="percentile", rng=np.random.default_rng(seed),
    )  # fmt: skip
    return float(fn(y, p)), float(res.confidence_interval.low), float(res.confidence_interval.high)


def paired_bootstrap_mae(
    y_true: npt.ArrayLike,
    pred_a: npt.ArrayLike,
    pred_b: npt.ArrayLike,
    *,
    n_boot: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = RANDOM_STATE,
) -> PairedComparison:
    """Paired bootstrap of MAE(A) - MAE(B) plus a Wilcoxon signed-rank test.

    The two-sided bootstrap p-value centres the bootstrap distribution on the observed
    difference: ``(1 + #{|d* - d| >= |d|}) / (n_boot + 1)``.
    """
    y = np.asarray(y_true, dtype=float)
    err_a = np.abs(y - np.asarray(pred_a, dtype=float))
    err_b = np.abs(y - np.asarray(pred_b, dtype=float))
    d = err_a - err_b
    res = stats.bootstrap(
        (d,), np.mean, n_resamples=n_boot, confidence_level=ci, method="percentile",
        rng=np.random.default_rng(seed),
    )  # fmt: skip
    diff = float(d.mean())
    extreme = np.abs(res.bootstrap_distribution - diff) >= abs(diff)
    return PairedComparison(
        mae_a=float(err_a.mean()),
        mae_b=float(err_b.mean()),
        diff=diff,
        ci_low=float(res.confidence_interval.low),
        ci_high=float(res.confidence_interval.high),
        p_bootstrap=(1 + int(extreme.sum())) / (n_boot + 1),
        p_wilcoxon=float(stats.wilcoxon(d).pvalue),
    )


def holm_correction(pvalues: dict[str, float], alpha: float = ALPHA_TEST) -> pd.DataFrame:
    """Holm step-down correction: ``p``, ``p_holm`` and ``reject`` per hypothesis."""
    p = np.array(list(pvalues.values()), dtype=float)
    reject, p_holm, _, _ = multipletests(p, alpha=alpha, method="holm")
    index = pd.Index(list(pvalues), name="hypothesis")
    return pd.DataFrame({"p": p, "p_holm": p_holm, "reject": reject}, index=index)


def comparison_table(
    results: dict[str, PairedComparison], alpha: float = ALPHA_TEST
) -> pd.DataFrame:
    """One row per named comparison, Holm-corrected on the bootstrap p-values."""
    table = pd.DataFrame({name: asdict(r) for name, r in results.items()}).T
    table["rel_diff_pct"] = 100 * table["diff"] / table["mae_a"]
    holm = holm_correction(table["p_bootstrap"].to_dict(), alpha)
    return table.join(holm[["p_holm", "reject"]])


def power_mde(
    abs_err_a: npt.ArrayLike,
    abs_err_b: npt.ArrayLike,
    *,
    test_sizes: Sequence[int] = (2000, 3000, 4000, 5000, 6000),
    rel_effects: Sequence[float] = tuple(np.arange(0.01, 0.105, 0.01).round(2)),
    n_sim: int = 500,
    alpha: float = ALPHA_TEST,
    target_power: float = 0.8,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Simulated power of the paired MAE comparison and the minimum detectable effect (MDE).

    Each run draws ``n`` rows of paired absolute errors (A, B) with replacement and rescales
    B's errors so that MAE(B) = (1 - effect) * MAE(A); the pairing structure of the two models
    (error correlation) is thereby taken from the data. Detection: paired z-test of the mean
    difference, two-sided level ``alpha``, B better.

    Args:
        abs_err_a: Per-row absolute errors (CHF) of the reference model, e.g. from CV.
        abs_err_b: Absolute errors of a second model on the same rows (pairing structure).

    Returns:
        Long frame ``test_size``, ``rel_effect``, ``effect_chf``, ``power``, ``mcse``
        (Monte-Carlo SE) and ``mde_rel``/``mde_chf`` (smallest grid effect with power >=
        ``target_power`` per test size, NaN if none).
    """
    err_a, err_b = np.asarray(abs_err_a, dtype=float), np.asarray(abs_err_b, dtype=float)
    err_b = err_b * err_a.mean() / err_b.mean()
    z_crit = stats.norm.ppf(1 - alpha / 2)
    rng = np.random.default_rng(seed)
    rows = []
    for n in test_sizes:
        idx = rng.integers(0, err_a.size, size=(n_sim, n))
        a, b = err_a[idx], err_b[idx]
        for effect in rel_effects:
            d = a - (1 - effect) * b
            z = d.mean(axis=1) / (d.std(axis=1, ddof=1) / np.sqrt(n))
            rows.append((n, effect, float(np.mean(z > z_crit))))
    out = pd.DataFrame(rows, columns=["test_size", "rel_effect", "power"])
    out.insert(2, "effect_chf", out["rel_effect"] * err_a.mean())
    out["mcse"] = np.sqrt(out["power"] * (1 - out["power"]) / n_sim)
    mde = out[out["power"] >= target_power].groupby("test_size")["rel_effect"].min()
    out["mde_rel"] = out["test_size"].map(mde)
    out["mde_chf"] = out["mde_rel"] * err_a.mean()
    logger.info("Power simulation: relative MDE per test size %s", mde.to_dict())
    return out


def learning_curve(
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    y: npt.ArrayLike,
    *,
    fractions: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Validation MAE/RMSE over nested random subsets of each fold's training rows.

    Args:
        fit_predict: ``fit_predict(train_positions, val_positions)`` fits a fresh model and
            returns validation predictions on the scale of ``y``.
        folds: ``(train_positions, val_positions)`` tuples.
        y: Targets of all rows the positions refer to.

    Returns:
        One row per fraction: ``n_train_mean``, ``mae_mean``, ``mae_std``, ``rmse_mean``,
        ``rmse_std`` (across folds).
    """
    target = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    records = []
    for train_pos, val_pos in folds:
        order = rng.permutation(np.asarray(train_pos))  # fixed per fold: subsets are nested
        for frac in fractions:
            sub = np.sort(order[: max(1, round(frac * len(order)))])
            pred = fit_predict(sub, np.asarray(val_pos))
            m = regression_metrics(target[val_pos], pred)
            records.append((frac, len(sub), m["MAE"], m["RMSE"]))
    table = pd.DataFrame(records, columns=["fraction", "n_train", "mae", "rmse"])
    return (
        table.groupby("fraction")
        .agg(
            n_train_mean=("n_train", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
        )
        .reset_index()
    )


def morans_i(
    values: npt.ArrayLike,
    coords: npt.ArrayLike,
    *,
    k: int = 8,
    permutations: int = 999,
    seed: int = RANDOM_STATE,
) -> MoranResult:
    """Moran's I with row-standardised k-NN weights and a two-sided permutation test."""
    weights = libpysal.weights.KNN.from_array(np.asarray(coords, dtype=float), k=k)
    np.random.seed(seed)  # esda draws its permutations from NumPy's global RNG
    moran = esda.Moran(np.asarray(values, dtype=float), weights, permutations=permutations)
    extreme = np.abs(moran.sim - moran.EI) >= abs(moran.I - moran.EI)
    p_value = (1 + int(extreme.sum())) / (1 + permutations)
    return MoranResult(float(moran.I), float(moran.EI), float(moran.z_sim), p_value)


def apply_plot_style() -> None:
    """Apply the DSPRO1/HSLU presentation style (seaborn whitegrid + project rcParams)."""
    sns.set_theme(context="notebook", style="whitegrid", palette=PALETTE)
    mpl.rcParams.update(_RC_PARAMS)


def save_fig(fig: Figure, name: str, dpi: int = 200) -> tuple[Path, Path]:
    """Save ``fig`` as ``FIGURES/<name>.png`` (report) and ``.svg`` (poster)."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    png, svg = FIGURES / f"{name}.png", FIGURES / f"{name}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    with mpl.rc_context({"svg.fonttype": "none", "svg.hashsalt": "rentml"}):
        fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    logger.info("Saved figure %s (+ .svg)", png)
    return png, svg


def chf(value: float, decimals: int = 0, prefix: str = "CHF ") -> str:
    """CHF amount with the Swiss thousands separator (``CHF 1'850``)."""
    return f"{prefix}{value:,.{decimals}f}".replace(",", "'")


def chf_formatter(decimals: int = 0) -> FuncFormatter:
    """Axis formatter for CHF amounts without the currency prefix (``1'850``)."""
    return FuncFormatter(lambda value, _pos: chf(value, decimals, prefix=""))


def _mlflow() -> ModuleType:
    # Imported lazily: MLflow takes about a second to import and only the tracking needs it.
    # The agent hint and SQLAlchemy deprecations from MLflow's own ORM are noise in notebooks.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    warnings.filterwarnings("ignore", category=SADeprecationWarning, module=r"mlflow(\.|$)")
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow


def _mlflow_key(key: str) -> str:
    # MLflow accepts only ASCII alphanumerics and _ - . / in keys ("R²", "≥ 20", "share %").
    return re.sub(r"[^\w\-./]", "_", key, flags=re.ASCII)


def log_run(
    name: str,
    stage: str,
    *,
    params: dict[str, object] | None = None,
    metrics: dict[str, float] | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    artifacts: Sequence[Path] = (),
    tags: dict[str, str] | None = None,
    experiment: str = EXPERIMENT,
) -> str:
    """Log one finished MLflow run (SQLite backend under ``config.MLRUNS``).

    Args:
        stage: Ablation stage, stored as tag ``stage``.
        tables: Frames stored as ``tables/<key>.csv`` artifacts.
        artifacts: Files stored with the run (figures, model bundles, report tables).

    Returns:
        The run id.
    """
    mlflow = _mlflow()
    exp = mlflow.get_experiment_by_name(experiment)
    exp_id = (
        exp.experiment_id
        if exp
        else mlflow.create_experiment(experiment, (MLRUNS / "artifacts").as_uri())
    )
    with mlflow.start_run(
        experiment_id=exp_id, run_name=name, tags={**(tags or {}), "stage": stage}
    ) as run:
        mlflow.log_params({_mlflow_key(k): v for k, v in (params or {}).items()})
        mlflow.log_metrics({_mlflow_key(k): v for k, v in (metrics or {}).items()})
        for key, df in (tables or {}).items():
            mlflow.log_text(df.to_csv(), f"tables/{key}.csv")
        for path in artifacts:
            mlflow.log_artifact(str(path))
    return run.info.run_id


def load_runs(experiment: str = EXPERIMENT) -> pd.DataFrame:
    """All runs of an experiment as returned by ``mlflow.search_runs``."""
    return _mlflow().search_runs(experiment_names=[experiment])


def ablation_table(
    runs: pd.DataFrame, stages: Sequence[str], metrics: Sequence[str], select_by: str = "cv_mae"
) -> pd.DataFrame:
    """Pick one finished run per stage by the validation metric ``select_by``.

    Selecting on a test metric would be test-set model selection, hence a separate argument.

    Returns:
        Table indexed by ``stage`` (order of ``stages``, missing stages skipped) with
        ``run_name`` and one column per metric.
    """
    done = runs[runs["status"].eq("FINISHED") & runs["tags.stage"].isin(stages)]
    best = done.sort_values(f"metrics.{select_by}").drop_duplicates("tags.stage")
    table = best.set_index("tags.stage").reindex(
        [s for s in stages if s in set(best["tags.stage"])]
    )
    table = table[["tags.mlflow.runName", *(f"metrics.{m}" for m in metrics)]]
    table.columns = ["run_name", *metrics]
    return table.rename_axis("stage")
