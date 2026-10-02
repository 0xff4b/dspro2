"""Project-wide constants, pre-registered evaluation settings and path discovery.

The constants mirror the pre-registered evaluation plan (``docs/EVALUATION_PLAN.md``):
changing one of them after the first training run is a documented deviation.
"""

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
REFERENCE_YEAR = 2026
SNAPSHOT_DATE = "2026-04-13"

QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
INTERVAL_COVERAGE = 0.80
ALPHA_TEST = 0.05
N_BOOTSTRAP = 2000
MIN_MONDRIAN_GROUP = 100
MIN_MAP_CELL = 20

# Swiss noise abatement ordinance (LSV), residential zones, sensitivity level II.
LSV_THRESHOLDS_DBA: dict[str, float] = {"day": 60.0, "night": 50.0}

# BFS canton numbers as used in swissBOUNDARIES3D (``kantonsnummer``).
CANTON_ABBR: dict[int, str] = {
    1: "ZH",
    2: "BE",
    3: "LU",
    4: "UR",
    5: "SZ",
    6: "OW",
    7: "NW",
    8: "GL",
    9: "ZG",
    10: "FR",
    11: "SO",
    12: "BS",
    13: "BL",
    14: "SH",
    15: "AR",
    16: "AI",
    17: "SG",
    18: "GR",
    19: "AG",
    20: "TG",
    21: "TI",
    22: "VD",
    23: "VS",
    24: "NE",
    25: "GE",
    26: "JU",
}

SB3D_VERSION = "2026-01"

# LightGBM 4.7's "advanced" method violated monotone constraints on the DSPRO1 data (a larger
# flat got a lower estimate for 15-29 of 200 test flats on a fine grid; verified), while
# "intermediate" and "basic" were exact with the same calibration MAE.
MONOTONE_CONSTRAINTS_METHOD = "intermediate"

_PACKAGE_NAME = "rentml"
_BASELINE_FILE = "dspro1_gradient_boosting_all_geo.joblib"


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout of the DSPRO2 project.

    Attributes:
        project_root: The repository root that contains ``pyproject.toml``.
        repo_root: Same as ``project_root`` (kept for callers of the former nested layout).
        data_raw: Raw exports (database dumps), never committed.
        data_external: Downloaded open data (swissBOUNDARIES3D, OSM), never committed.
        data_interim: Intermediate tables (cleaned listings, splits).
        cache: Caches for embeddings and LLM extractions.
        models: Trained model artefacts.
        figures: Report figures (``docs/fig``).
        results: Report tables (``docs/results``).
        mlruns: MLflow tracking directory.
        dspro1_csv_dir: DSPRO1 CSV snapshot used as offline fallback (``data/dspro1_snapshot``).
        pipelines: Scraper and enrichment pipelines (``pipelines``).
        baseline_model: The DSPRO1 GradientBoosting ``ALL+geo`` baseline (``models/baseline``).
    """

    project_root: Path
    repo_root: Path
    data_raw: Path
    data_external: Path
    data_interim: Path
    cache: Path
    models: Path
    figures: Path
    results: Path
    mlruns: Path
    dspro1_csv_dir: Path
    pipelines: Path
    baseline_model: Path

    @classmethod
    def from_root(cls, project_root: Path) -> "ProjectPaths":
        """Build the layout for a given project root.

        Args:
            project_root: Directory containing the ``rentml`` ``pyproject.toml``.

        Returns:
            The resolved project paths.
        """
        root = project_root.resolve()
        return cls(
            project_root=root,
            repo_root=root,
            data_raw=root / "data" / "raw",
            data_external=root / "data" / "external",
            data_interim=root / "data" / "interim",
            cache=root / "data" / "cache",
            models=root / "models",
            figures=root / "docs" / "fig",
            results=root / "docs" / "results",
            mlruns=root / "mlruns",
            dspro1_csv_dir=root / "data" / "dspro1_snapshot",
            pipelines=root / "pipelines",
            baseline_model=root / "models" / "baseline" / _BASELINE_FILE,
        )

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        """Find the project root by walking up from ``start``.

        Args:
            start: Directory to start from; defaults to the current working directory.

        Returns:
            The resolved project paths.

        Raises:
            FileNotFoundError: If no ``pyproject.toml`` of the ``rentml`` package is found.
        """
        current = (start or Path.cwd()).resolve()
        for candidate in [current, *current.parents]:
            if _is_project_root(candidate):
                return cls.from_root(candidate)
        raise FileNotFoundError(f"No '{_PACKAGE_NAME}' pyproject.toml found above {current}")

    def ensure(self) -> None:
        """Create all writable directories if they do not exist yet."""
        for path in (
            self.data_raw,
            self.data_external,
            self.data_interim,
            self.cache,
            self.models,
            self.figures,
            self.results,
            self.mlruns,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _is_project_root(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    if not pyproject.is_file():
        return False
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return data.get("project", {}).get("name") == _PACKAGE_NAME


def load_env(paths: ProjectPaths) -> list[Path]:
    """Load ``.env`` files without overriding variables that are already set.

    Reads ``.env`` in the repository root.

    Args:
        paths: Project layout.

    Returns:
        The ``.env`` files that were found and loaded.
    """
    loaded: list[Path] = []
    for env_file in (paths.project_root / ".env",):
        if env_file.is_file():
            load_dotenv(env_file, override=False)
            loaded.append(env_file)
            logger.info("Loaded environment file %s", env_file)
    return loaded
