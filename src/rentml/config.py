"""Paths, the constants of the pre-registered evaluation plan (docs/EVALUATION_PLAN.md), downloads.

Changing a constant after the first training run is a deviation that belongs in the plan's
changelog.
"""

import logging
from pathlib import Path

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SNAPSHOT_DIR = DATA / "dspro1_snapshot"  # versioned DSPRO1 CSVs, offline fallback
EXTERNAL = DATA / "external"  # downloaded open data
INTERIM = DATA / "interim"  # frozen exports, splits
CACHE = DATA / "cache"  # embeddings, LLM extractions, OSM
APP_DATA = DATA / "app"  # map layers and QoLI table for the Dash app
MODELS = ROOT / "models"
FIGURES = ROOT / "docs" / "fig"
RESULTS = ROOT / "docs" / "results"
MLRUNS = ROOT / "mlruns"
BUNDLE_PATH = MODELS / "rent_bundle_v1.joblib"

RANDOM_STATE = 42
REFERENCE_YEAR = 2026
SNAPSHOT_DATE = "2026-04-13"
SB3D_VERSION = "2026-01"

QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
INTERVAL_COVERAGE = 0.80
ALPHA_TEST = 0.05
N_BOOTSTRAP = 2000
MIN_MONDRIAN_GROUP = 100  # calibration objects per Mondrian cell before falling back
MIN_MAP_CELL = 20  # objects per published map cell (data licence)

# Plausible apartments (data cleaning, text extraction and the app's input check).
ROOMS_RANGE = (1.0, 15.0)
AREA_RANGE = (10.0, 500.0)
PRICE_RANGE = (300.0, 15_000.0)

# Swiss noise abatement ordinance (LSV), residential zones, sensitivity level II, dB(A).
LSV_THRESHOLDS_DBA = {"day": 60.0, "night": 50.0}

# LightGBM 4.7's "advanced" method let a larger flat get a lower estimate for 15-29 of 200
# test flats; "intermediate" was exact with the same calibration MAE.
MONOTONE_CONSTRAINTS_METHOD = "intermediate"

# BFS canton numbers as used in swissBOUNDARIES3D.
CANTON_ABBR = {
    1: "ZH", 2: "BE", 3: "LU", 4: "UR", 5: "SZ", 6: "OW", 7: "NW", 8: "GL", 9: "ZG",
    10: "FR", 11: "SO", 12: "BS", 13: "BL", 14: "SH", 15: "AR", 16: "AI", 17: "SG",
    18: "GR", 19: "AG", 20: "TG", 21: "TI", 22: "VD", 23: "VS", 24: "NE", 25: "GE", 26: "JU",
}  # fmt: skip


def ensure_dirs() -> None:
    """Create the writable data, model and report directories."""
    for path in (EXTERNAL, INTERIM, CACHE, APP_DATA, MODELS, FIGURES, RESULTS, MLRUNS):
        path.mkdir(parents=True, exist_ok=True)


def load_env() -> bool:
    """Load ``.env`` from the repository root without overriding set variables.

    Returns:
        True if the file existed.
    """
    return load_dotenv(ROOT / ".env", override=False)


def download(
    url: str, target: Path, params: dict[str, str] | None = None, headers: dict | None = None
) -> Path:
    """Stream ``url`` to ``target`` unless it already exists (``.part`` until complete)."""
    if target.is_file():
        return target
    logger.info("Downloading %s", url)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f"{target.name}.part")
    with requests.get(url, params=params, headers=headers, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with part.open("wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    part.replace(target)
    return target
