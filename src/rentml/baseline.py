"""DSPRO1 baseline (GradientBoosting, ``ALL+geo``) for comparisons with the DSPRO2 models.

The DSPRO1 notebook selected this model for its small train/eval gap (eval R² 0.712, MAE 281 CHF,
RMSE 423 CHF on the DSPRO1 80/20 split). It predicts the cold rent in CHF directly (no log
target) from 11 listing/geo features plus a KMeans cluster of the LV95 coordinates. It was
trained only on fully enriched listings, so it is applied only where all features are present.
The artefact was pickled with scikit-learn 1.6.1; loading it with a newer version emits an
``InconsistentVersionWarning``, which is logged instead of shown.
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_COORDS = ["east", "north"]
_CLUSTER = "geo_cluster"


@dataclass(frozen=True)
class Dspro1Baseline:
    """The selected DSPRO1 model with its geo-cluster pipeline.

    Attributes:
        model: Fitted ``GradientBoostingRegressor`` (target: rent in CHF).
        geo_pipe: Fitted scaler + KMeans pipeline on ``east``/``north`` (LV95 metres).
        features: Model input columns in training order (last one is ``geo_cluster``).
        metrics: DSPRO1 evaluation metrics stored with the artefact.
    """

    model: GradientBoostingRegressor
    geo_pipe: Pipeline
    features: tuple[str, ...]
    metrics: dict[str, float]

    @property
    def input_columns(self) -> list[str]:
        """Columns the caller must provide (canonical DSPRO2 names)."""
        return [c for c in self.features if c != _CLUSTER]

    def applicable(self, df: pd.DataFrame) -> pd.Series:
        """Mark rows with all baseline inputs present.

        Args:
            df: Frame with the canonical DSPRO2 columns.

        Returns:
            Boolean Series aligned with ``df.index``.

        Raises:
            KeyError: If an input column is missing.
        """
        if missing := [c for c in self.input_columns if c not in df.columns]:
            raise KeyError(f"baseline inputs missing: {missing}")
        return df[self.input_columns].notna().all(axis=1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the monthly cold rent in CHF.

        Args:
            df: Frame with the canonical DSPRO2 columns (e.g. ``test_df``).

        Returns:
            Predictions in CHF aligned with ``df``; NaN where :meth:`applicable` is False.
        """
        mask = self.applicable(df).to_numpy()
        out = np.full(len(df), np.nan)
        if not mask.any():
            return out
        X = df.loc[mask, self.input_columns].astype(float)
        X[_CLUSTER] = self.geo_pipe.predict(X[_COORDS])
        out[mask] = self.model.predict(X[list(self.features)])
        return out


def load_dspro1_baseline(path: Path) -> Dspro1Baseline:
    """Load the DSPRO1 baseline artefact.

    Args:
        path: The joblib file (``ProjectPaths.baseline_model``).

    Returns:
        The loaded baseline.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the artefact lacks the expected keys.
    """
    if not path.is_file():
        raise FileNotFoundError(f"DSPRO1 baseline not found: {path}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        artefact = joblib.load(path)
    for warning in caught:
        if issubclass(warning.category, InconsistentVersionWarning):
            logger.info("DSPRO1 baseline: %s", str(warning.message).split(".")[0])
    if not isinstance(artefact, dict) or not {"model", "geo_pipe", "features"} <= artefact.keys():
        raise ValueError(f"unexpected DSPRO1 baseline artefact in {path}")
    metrics = {k: float(v) for k, v in artefact.get("metrics", {}).items() if _is_number(v)}
    return Dspro1Baseline(
        artefact["model"], artefact["geo_pipe"], tuple(artefact["features"]), metrics
    )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float | np.floating | np.integer)
