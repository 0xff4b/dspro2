"""SHAP explanations for tree models trained on log rent.

All models in ``rentml`` predict ``log(price)``, so a SHAP value is an additive contribution on the
log scale. Exponentiating turns it into a multiplicative effect on the rent: a SHAP value of +0.10
means "about 100 * (exp(0.10) - 1) = +10.5 % relative to the model's base rent, with all other
features at their observed values". This is the ``approx_pct`` column that the Fair-Rent Check
uses to show the three largest drivers of a single estimate.

``shap`` is imported lazily inside the functions because importing it takes about two seconds and
only notebooks and the app need it.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rentml.config import RANDOM_STATE

if TYPE_CHECKING:
    import shap

logger = logging.getLogger(__name__)

DRIVER_COLUMNS: tuple[str, ...] = ("feature", "value", "shap_log", "approx_pct")


def tree_shap(
    model: object,
    X: pd.DataFrame,
    *,
    max_rows: int = 2000,
    seed: int = RANDOM_STATE,
    check_additivity: bool = True,
) -> shap.Explanation:
    """Compute exact TreeSHAP values for a tree model on the log scale.

    Uses ``shap.TreeExplainer`` with the path-dependent algorithm (no background data needed), so
    the base value is the model's average training prediction. Works with ``lgb.LGBMRegressor``,
    ``lgb.Booster`` and every other model supported by ``shap.TreeExplainer``.

    Args:
        model: Fitted tree model that predicts ``log(price)``. Typed as ``object`` because shap
            accepts many untyped model classes.
        X: Feature matrix with the same columns (and dtypes) the model was trained on. If the
            model knows its feature names (``feature_names_in_``, LightGBM ``feature_name_`` or
            ``Booster.feature_name()``), the columns are reordered to the training order, so a
            frame built from a dict is explained with the right labels.
        max_rows: Maximum number of rows to explain. Larger frames are subsampled without
            replacement; the sampled rows keep their original order.
        seed: Seed for the row subsample.
        check_additivity: Let shap verify that SHAP values sum to the model output.

    Returns:
        A ``shap.Explanation`` with ``values`` of shape (n_rows, n_features), ``base_values``,
        ``data``, ``feature_names`` and ``instance_names`` (the index labels of the explained rows,
        e.g. ``listing_id``).

    Raises:
        ValueError: If ``X`` is empty, ``max_rows`` is smaller than 1 or the columns of ``X`` do
            not match the model's feature names.
    """
    # Lazy import: shap takes ~2 s to import and is only needed for explanations.
    import shap

    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    if X.empty:
        raise ValueError("X is empty; nothing to explain")

    sample = _subsample(_align_columns(model, X), max_rows=max_rows, seed=seed)
    explainer = shap.TreeExplainer(model)
    raw = explainer(sample, check_additivity=check_additivity)
    values = np.asarray(raw.values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"Expected single-output SHAP values of shape (n, p), got {values.shape}; "
            "multi-output models are not supported"
        )
    base = np.broadcast_to(np.asarray(raw.base_values, dtype=np.float64), (len(sample),)).copy()
    logger.info(
        "TreeSHAP computed for %d of %d rows (%d features)", len(sample), len(X), X.shape[1]
    )
    return shap.Explanation(
        values=values,
        base_values=base,
        data=sample.to_numpy(),
        feature_names=[str(c) for c in sample.columns],
        instance_names=list(sample.index),
    )


def _model_feature_names(model: object) -> list[str] | None:
    names = getattr(model, "feature_names_in_", None)  # sklearn API: exact training columns
    if names is None:
        names = getattr(model, "feature_name_", None)  # LGBMRegressor
    if names is None and callable(getattr(model, "feature_name", None)):
        names = model.feature_name()  # lgb.Booster
    if names is None:
        return None
    names = [str(name) for name in names]
    if names == [f"Column_{j}" for j in range(len(names))]:  # LightGBM fitted without names
        return None
    return names


def _name_key(name: object) -> str:
    # LightGBM stores names with whitespace replaced by "_" ("living area" -> "living_area").
    return re.sub(r"\s", "_", str(name))


def _align_columns(model: object, X: pd.DataFrame) -> pd.DataFrame:
    """Reorder ``X`` to the model's training column order; SHAP maps features by position."""
    names = _model_feature_names(model)
    if names is None:
        return X
    lookup = {_name_key(col): col for col in X.columns}
    if len(lookup) != X.shape[1]:
        raise ValueError("X has duplicate column names (whitespace counts as '_')")
    keys = [_name_key(name) for name in names]
    missing = [name for name, key in zip(names, keys, strict=True) if key not in lookup]
    unexpected = [str(col) for key, col in lookup.items() if key not in set(keys)]
    if missing or unexpected:
        raise ValueError(
            f"Columns of X do not match the model's features: missing {missing}, "
            f"unexpected {unexpected}"
        )
    if keys != list(lookup):
        logger.info("Reordering X columns to the model's training order for SHAP")
    return X[[lookup[key] for key in keys]]


def _subsample(X: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if len(X) <= max_rows:
        return X
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    return X.iloc[positions]


def top_drivers(explanation: shap.Explanation, row: int, k: int = 3) -> pd.DataFrame:
    """Return the ``k`` features with the largest absolute SHAP value for one explained row.

    Args:
        explanation: Result of :func:`tree_shap` (or any single-output ``shap.Explanation``).
        row: Position of the row inside the explanation (0-based, negative values allowed).
        k: Number of drivers to return; capped at the number of features.

    Returns:
        DataFrame with columns ``feature``, ``value`` (observed feature value), ``shap_log``
        (contribution on the log scale) and ``approx_pct`` (``100 * (exp(shap_log) - 1)``, the
        approximate percentage effect on the rent), sorted by ``|shap_log|`` descending.

    Raises:
        ValueError: If ``k`` is smaller than 1 or the explanation is not two-dimensional.
        IndexError: If ``row`` is out of range.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    values = _values_2d(explanation)
    n_rows, n_features = values.shape
    if not -n_rows <= row < n_rows:
        raise IndexError(f"row {row} out of range for an explanation with {n_rows} rows")

    shap_row = values[row]
    order = np.argsort(-np.abs(shap_row), kind="stable")[: min(k, n_features)]
    names = _feature_names(explanation, n_features)
    data = np.asarray(explanation.data, dtype=object) if explanation.data is not None else None
    observed = [data[row, j] if data is not None else np.nan for j in order]
    shap_top = shap_row[order]
    return pd.DataFrame(
        {
            "feature": [names[j] for j in order],
            "value": observed,
            "shap_log": shap_top,
            "approx_pct": 100.0 * np.expm1(shap_top),
        },
        columns=list(DRIVER_COLUMNS),
    )


def global_importance(explanation: shap.Explanation) -> pd.DataFrame:
    """Rank features by their mean absolute SHAP value.

    Args:
        explanation: Result of :func:`tree_shap`.

    Returns:
        DataFrame with columns ``feature``, ``mean_abs_shap`` (log scale), ``mean_shap`` (signed
        mean, log scale), ``share`` (fraction of the total mean |SHAP|) and ``approx_pct``
        (``100 * (exp(mean_abs_shap) - 1)``), sorted by ``mean_abs_shap`` descending.

    Raises:
        ValueError: If the explanation is not two-dimensional or has no rows.
    """
    values = _values_2d(explanation)
    if values.shape[0] == 0:
        raise ValueError("explanation has no rows")
    mean_abs = np.abs(values).mean(axis=0)
    total = mean_abs.sum()
    table = pd.DataFrame(
        {
            "feature": _feature_names(explanation, values.shape[1]),
            "mean_abs_shap": mean_abs,
            "mean_shap": values.mean(axis=0),
            "share": mean_abs / total if total > 0 else np.zeros_like(mean_abs),
            "approx_pct": 100.0 * np.expm1(mean_abs),
        }
    )
    return table.sort_values("mean_abs_shap", ascending=False, kind="stable").reset_index(drop=True)


def shap_frame(explanation: shap.Explanation) -> pd.DataFrame:
    """Convert SHAP values into a DataFrame (rows = explained instances, columns = features).

    Handy for group-wise summaries, e.g. mean SHAP per canton or per price band.

    Args:
        explanation: Result of :func:`tree_shap`.

    Returns:
        DataFrame of log-scale SHAP values indexed by the explanation's instance names (or a
        RangeIndex if none are set).

    Raises:
        ValueError: If the explanation is not two-dimensional.
    """
    values = _values_2d(explanation)
    names = explanation.instance_names
    index = pd.Index(list(names)) if names is not None else None
    return pd.DataFrame(values, columns=_feature_names(explanation, values.shape[1]), index=index)


def _values_2d(explanation: shap.Explanation) -> np.ndarray:
    values = np.asarray(explanation.values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"Expected a 2-D explanation (rows x features), got shape {values.shape}; "
            "pass the full explanation, not a single row"
        )
    return values


def _feature_names(explanation: shap.Explanation, n_features: int) -> list[str]:
    names = explanation.feature_names
    if names is None:
        return [f"f{j}" for j in range(n_features)]
    return [str(name) for name in names]
