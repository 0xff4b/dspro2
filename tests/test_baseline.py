"""Tests for rentml.baseline (the versioned DSPRO1 GradientBoosting ALL+geo artefact)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from rentml.baseline import Dspro1Baseline, load_dspro1_baseline
from rentml.config import ProjectPaths

PATHS = ProjectPaths.discover(Path(__file__).parent)


@pytest.fixture(scope="module")
def baseline() -> Dspro1Baseline:
    return load_dspro1_baseline(PATHS.baseline_model)


def _listing(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "east": [2683000.0, 2600000.0, 2717000.0][:n],
            "north": [1248000.0, 1200000.0, 1096000.0][:n],
            "elevation": [410.0, 540.0, 280.0][:n],
            "area": [80.0, 60.0, 100.0][:n],
            "rooms": [3.0, 2.0, 4.0][:n],
            "year_built": [1990.0, 1975.0, 2010.0][:n],
            "apartments": [10.0, 4.0, 20.0][:n],
            "land_area": [800.0, 500.0, 1500.0][:n],
            "population": [120.0, 40.0, 200.0][:n],
            "oev": [5000.0, 2000.0, 8000.0][:n],
            "solar": [3.0, 2.0, 4.0][:n],
        }
    )


def test_baseline_artefact_is_the_selected_dspro1_model(baseline: Dspro1Baseline) -> None:
    assert type(baseline.model).__name__ == "GradientBoostingRegressor"
    assert baseline.features[-1] == "geo_cluster"
    assert len(baseline.input_columns) == 11
    assert baseline.metrics["R² Eval"] == pytest.approx(0.7116, abs=1e-3)


def test_baseline_predicts_plausible_rents(baseline: Dspro1Baseline) -> None:
    pred = baseline.predict(_listing())
    assert pred.shape == (3,)
    assert np.all((pred > 300) & (pred < 10_000))


def test_baseline_larger_flat_costs_more(baseline: Dspro1Baseline) -> None:
    small, large = _listing(1), _listing(1).assign(area=140.0, rooms=5.0)
    assert baseline.predict(large)[0] > baseline.predict(small)[0]


def test_baseline_returns_nan_for_incomplete_rows(baseline: Dspro1Baseline) -> None:
    df = _listing()
    df.loc[1, "year_built"] = np.nan
    pred = baseline.predict(df)
    assert np.isnan(pred[1]) and np.isfinite(pred[[0, 2]]).all()
    assert baseline.applicable(df).tolist() == [True, False, True]


def test_baseline_missing_column_raises(baseline: Dspro1Baseline) -> None:
    with pytest.raises(KeyError, match="solar"):
        baseline.predict(_listing().drop(columns="solar"))


def test_load_missing_or_invalid_artefact_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dspro1_baseline(tmp_path / "missing.joblib")
    bad = tmp_path / "bad.joblib"
    joblib.dump({"model": None}, bad)
    with pytest.raises(ValueError, match="unexpected"):
        load_dspro1_baseline(bad)
