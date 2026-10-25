"""Quality-of-life profile of a listing's municipality (QoLI) for the marketplace.

Reads the municipal table written by the QoLI notebook (``data/app/qoli_municipalities.parquet``:
population-weighted hectare scores per municipality, 0-100, 100 = best). The hectare lookup for
the exact address is not served by the app yet, so a profile describes the municipality, not the
building. Without the table the marketplace works and simply shows no quality-of-life block.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rentml.config import ProjectPaths

logger = logging.getLogger(__name__)

QOLI_ENV = "RENTML_QOLI_TABLE"
QOLI_FILE = "qoli_municipalities.parquet"
SPEC_FILE = "qoli_spec.json"
ID_COLUMN = "municipality_id"
INDEX_COLUMN = "qoli"
CORE_DIMENSIONS: tuple[str, ...] = ("transport", "noise", "amenities", "green")
EXTENDED_DIMENSIONS: tuple[str, ...] = ("air", "sunshine", "tax", "housing")
DIMENSION_LABELS: dict[str, str] = {
    "transport": "ÖV-Anbindung",
    "noise": "Ruhe (Strassen- und Bahnlärm)",
    "amenities": "Versorgung und Freizeit",
    "green": "Grünraum",
    "air": "Luftqualität",
    "sunshine": "Sonnenschein",
    "tax": "Steuerbelastung",
    "housing": "Wohnungsangebot (Leerstand)",
}


@dataclass(frozen=True)
class QualityProfile:
    """QoLI of one municipality.

    Attributes:
        municipality_id: BFS municipality number.
        qoli: Core index (0-100, 100 = best).
        percentile: Share of Swiss municipalities with a lower core index (0-100).
        dimensions: Core dimension scores (0-100) keyed by dimension name.
        extended: Scores of the extended dimensions that are available.
    """

    municipality_id: int
    qoli: float
    percentile: float
    dimensions: dict[str, float]
    extended: dict[str, float]


def default_table(paths: ProjectPaths) -> Path:
    """Location the QoLI notebook writes the municipal table to."""
    return paths.project_root / "data" / "app" / QOLI_FILE


def resolve_table(path: Path | None = None) -> Path:
    """Table path: argument, then ``$RENTML_QOLI_TABLE``, then the project default."""
    if path is not None:
        return Path(path)
    if env := os.environ.get(QOLI_ENV):
        return Path(env)
    return default_table(ProjectPaths.discover())


def _scores(row: pd.Series, names: tuple[str, ...]) -> dict[str, float]:
    return {name: float(row[name]) for name in names if name in row.index and pd.notna(row[name])}


class QualityLookup:
    """Municipal QoLI scores with their rank among all Swiss municipalities."""

    def __init__(self, table: pd.DataFrame, version: str = "") -> None:
        """Index the table.

        Args:
            table: One row per municipality with ``municipality_id`` (column or index), ``qoli``
                and the dimension columns.
            version: Version of the QoLI specification, shown as provenance.

        Raises:
            KeyError: If the table has no ``qoli`` column or no municipality id.
        """
        if ID_COLUMN in table.columns:
            table = table.set_index(ID_COLUMN)
        if table.index.name != ID_COLUMN or INDEX_COLUMN not in table.columns:
            raise KeyError(f"QoLI table needs '{ID_COLUMN}' and '{INDEX_COLUMN}'")
        self.table = table
        self.version = version
        self._percentile = table[INDEX_COLUMN].rank(pct=True, method="average") * 100.0

    @classmethod
    def load(cls, path: Path, spec_path: Path | None = None) -> "QualityLookup | None":
        """Read the table; ``None`` (with a warning) if the file does not exist.

        Args:
            path: Parquet file written by the QoLI notebook.
            spec_path: Optional ``qoli_spec.json`` whose ``version`` is shown as provenance.

        Returns:
            The lookup, or ``None`` without a table.
        """
        path = Path(path)
        if not path.is_file():
            logger.warning("No QoLI table at %s: listings show no quality-of-life block", path)
            return None
        version = ""
        if spec_path is not None and Path(spec_path).is_file():
            spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
            version = str(spec.get("version", ""))
        return cls(pd.read_parquet(path), version)

    def profile(self, municipality_id: int) -> QualityProfile | None:
        """Profile of a municipality (``None`` if it is unknown or has no core index)."""
        if municipality_id not in self.table.index:
            return None
        row = self.table.loc[municipality_id]
        if pd.isna(row[INDEX_COLUMN]):
            return None
        return QualityProfile(
            municipality_id=int(municipality_id),
            qoli=float(row[INDEX_COLUMN]),
            percentile=float(self._percentile.loc[municipality_id]),
            dimensions=_scores(row, CORE_DIMENSIONS),
            extended=_scores(row, EXTENDED_DIMENSIONS),
        )
