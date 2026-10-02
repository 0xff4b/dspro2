"""Datasheet registry and cached downloads of the Quality-of-Life Index (QoLI) sources.

Every dataset of the QoLI (dimension or validation) is registered with provider, URL, licence,
reference year and resolution, so the datasheet table is generated rather than typed. File-based
sources are downloaded once into ``data/external/qoli``. Sampling the sources at LV95 points is
in ``rentml.qoli_layers``, municipal statistics in ``rentml.qoli_municipal``.
"""

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

STAC_BASE = "https://data.geo.admin.ch"
COMMUNES_API = "https://www.agvchapp.bfs.admin.ch/api/communes/correspondances"
ESTV_TAX_2018_URL = (
    "https://www.estv.admin.ch/dam/de/sd-web/u8ENqfdNX1ci/statistik-belastung-gden-2018-de.xlsx"
)
HTTP_HEADERS = {"User-Agent": "rentml/0.1 (HSLU DSPRO2 student project)"}
_CHUNK = 1 << 20


@dataclass(frozen=True)
class SourceSpec:
    """One dataset of the QoLI with everything the datasheet needs.

    Attributes:
        key: Short identifier used in code and tables.
        title: Dataset name.
        provider: Publishing office.
        url: Download URL (``""`` if fetched through an API helper).
        licence: Terms of use.
        reference: Reference year or period of the data (not the download date).
        resolution: Spatial resolution or unit.
        unit: Unit of the indicator derived from it.
        use: Role in the QoLI (dimension or validation).
    """

    key: str
    title: str
    provider: str
    url: str
    licence: str
    reference: str
    resolution: str
    unit: str
    use: str

    @property
    def filename(self) -> str:
        """File name of the cached download (last URL segment)."""
        return self.url.rsplit("/", 1)[-1]


def _stac(collection: str, item: str, asset: str) -> str:
    return f"{STAC_BASE}/{collection}/{item}/{asset}"


_OGD = "OGD (opendata.swiss 'open use, must provide the source')"
SOURCES: dict[str, SourceSpec] = {
    s.key: s
    for s in [
        SourceSpec(
            "oev",
            "Public-transport quality classes (ÖV-Güteklassen) and classified stops",
            "ARE",
            _stac(
                "ch.are.gueteklassen_oev",
                "gueteklassen_oev_2026",
                "gueteklassen_oev_2026_2056.gpkg.zip",
            ),
            _OGD,
            "timetable 2026",
            "polygons / stop points",
            "class 0-4; m",
            "dimension transport",
        ),
        SourceSpec(
            "road_day",
            "sonBASE road traffic noise, day (L_r,Tag 06-22 h)",
            "BAFU",
            _stac(
                "ch.bafu.laerm-strassenlaerm_tag",
                "laerm-strassenlaerm_tag",
                "laerm-strassenlaerm_tag_2056.tif",
            ),
            f"{_OGD}; source 'BAFU 2025, sonBASE'",
            "2021 (computed 2023)",
            "10 m raster",
            "dB(A)",
            "dimension noise; LSV/WHO thresholds",
        ),
        SourceSpec(
            "road_night",
            "sonBASE road traffic noise, night (L_r,Nacht 22-06 h)",
            "BAFU",
            _stac(
                "ch.bafu.laerm-strassenlaerm_nacht",
                "laerm-strassenlaerm_nacht",
                "laerm-strassenlaerm_nacht_2056.tif",
            ),
            f"{_OGD}; source 'BAFU 2025, sonBASE'",
            "2021 (computed 2023)",
            "10 m raster",
            "dB(A)",
            "dimension noise; LSV/WHO thresholds",
        ),
        SourceSpec(
            "rail_day",
            "sonBASE railway noise, day (L_r,Tag 06-22 h)",
            "BAFU",
            _stac(
                "ch.bafu.laerm-bahnlaerm_tag", "laerm-bahnlaerm_tag", "laerm-bahnlaerm_tag_2056.tif"
            ),
            f"{_OGD}; source 'BAFU 2025, sonBASE'",
            "2021 (computed 2023)",
            "10 m raster",
            "dB(A)",
            "dimension noise; LSV/WHO thresholds",
        ),
        SourceSpec(
            "rail_night",
            "sonBASE railway noise, night (L_r,Nacht 22-06 h)",
            "BAFU",
            _stac(
                "ch.bafu.laerm-bahnlaerm_nacht",
                "laerm-bahnlaerm_nacht",
                "laerm-bahnlaerm_nacht_2056.tif",
            ),
            f"{_OGD}; source 'BAFU 2025, sonBASE'",
            "2021 (computed 2023)",
            "10 m raster",
            "dB(A)",
            "dimension noise; LSV/WHO thresholds",
        ),
        SourceSpec(
            "landuse",
            "Land-use statistics (Arealstatistik), NOAS04 hectare points",
            "BFS",
            _stac("ch.bfs.arealstatistik", "arealstatistik", "arealstatistik_2056.csv.zip"),
            _OGD,
            "survey flights 2013-2020 (release 2024)",
            "100 m hectare points",
            "share 0-1",
            "dimension green",
        ),
        SourceSpec(
            "hectares",
            "Service accessibility per inhabited hectare (STATPOP population, distances)",
            "BFS",
            _stac("ch.bfs.erreichbarkeit", "erreichbarkeit", "erreichbarkeit_2056.csv"),
            _OGD,
            "2021",
            "100 m hectares",
            "inhabitants; m",
            "reference grid (normalisation, municipal aggregates); validation of OSM access",
        ),
        SourceSpec(
            "no2",
            "PolluMap nitrogen dioxide (NO2), annual mean",
            "BAFU",
            _stac(
                "ch.bafu.luftreinhaltung-stickstoffdioxid",
                "luftreinhaltung-stickstoffdioxid_2025",
                "luftreinhaltung-stickstoffdioxid_2025_2056.tif",
            ),
            _OGD,
            "2025",
            "20 m raster",
            "µg/m³",
            "dimension air (extended index)",
        ),
        SourceSpec(
            "pm25",
            "PolluMap fine particulate matter (PM2.5), annual mean",
            "BAFU",
            _stac(
                "ch.bafu.luftreinhaltung-feinstaub_pm2_5",
                "luftreinhaltung-feinstaub_pm2_5_2025",
                "luftreinhaltung-feinstaub_pm2_5_2025_2056.tif",
            ),
            _OGD,
            "2025",
            "100 m raster",
            "µg/m³",
            "dimension air (extended index)",
        ),
        SourceSpec(
            "sunshine",
            "Relative sunshine duration, climate normal 1991-2020",
            "MeteoSwiss",
            _stac(
                "ch.meteoschweiz.ogd-climate-normals-grid",
                "ch",
                "ogd-climate-normals-grid.snormy9120_ch01r.swiss.lv95_"
                "19910101000000_19910101000000.tif",
            ),
            "MeteoSwiss OGD (open use, source required)",
            "1991-2020",
            "1 km grid",
            "% of possible",
            "dimension sunshine (extended index)",
        ),
        SourceSpec(
            "tax",
            "Tax burden in the municipalities (cantonal, municipal and church tax)",
            "ESTV",
            ESTV_TAX_2018_URL,
            "public statistics (no explicit licence; source required)",
            "2018 (latest file)",
            "municipality (remapped to 2026)",
            "% of gross income",
            "dimension tax (extended index)",
        ),
        SourceSpec(
            "vacancy",
            "Vacant dwellings and vacancy rate on 1 June (DF_LWZ_1)",
            "BFS",
            "",
            _OGD,
            "2026",
            "municipality",
            "% of dwellings",
            "dimension housing availability (extended index); validation of the core index",
        ),
        SourceSpec(
            "city_statistics",
            "City Statistics (Urban Audit): selected variables (DF_CITYSTAT_1)",
            "BFS",
            "",
            _OGD,
            "latest year per variable (2018-2025)",
            "10 core cities",
            "various",
            "validation (descriptive, n = 10)",
        ),
        SourceSpec(
            "communes",
            "Historicised commune register: correspondence of municipality numbers",
            "BFS",
            COMMUNES_API,
            _OGD,
            "2018 -> 2026",
            "municipality",
            "-",
            "remap ESTV 2018 to 2026 boundaries",
        ),
    ]
}


def source_table(keys: Iterable[str] | None = None) -> pd.DataFrame:
    """Datasheet table of the QoLI sources (one row per dataset).

    Args:
        keys: Source keys to include; ``None`` = all.

    Returns:
        DataFrame indexed by ``key`` with the ``SourceSpec`` fields.

    Raises:
        KeyError: For an unknown key.
    """
    selected = list(SOURCES) if keys is None else list(keys)
    if unknown := [k for k in selected if k not in SOURCES]:
        raise KeyError(f"Unknown QoLI sources: {unknown}")
    return pd.DataFrame([asdict(SOURCES[k]) for k in selected]).set_index("key")


def download(url: str, dest_dir: Path, *, overwrite: bool = False, timeout: float = 600.0) -> Path:
    """Download a file once into ``dest_dir`` (atomic write, reused on later calls).

    Args:
        url: File URL; the last path segment becomes the file name.
        dest_dir: Target directory (created if needed).
        overwrite: Download again even if the file exists.
        timeout: Connect/read timeout in seconds.

    Returns:
        Path to the local file.

    Raises:
        RuntimeError: If the download fails.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / url.rsplit("/", 1)[-1]
    if target.is_file() and not overwrite:
        return target
    logger.info("Downloading %s", url)
    part = target.with_name(f"{target.name}.part")
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=HTTP_HEADERS) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_CHUNK):
                    handle.write(chunk)
        part.replace(target)
    except requests.RequestException as exc:
        raise RuntimeError(f"Download of {url} failed: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)
    return target


def fetch_sources(
    dest_dir: Path, keys: Iterable[str], *, overwrite: bool = False
) -> dict[str, Path]:
    """Download the file-based sources (those with a URL) into ``dest_dir``.

    Args:
        dest_dir: Cache directory, e.g. ``data/external/qoli``.
        keys: Source keys; API sources without a file URL are skipped.
        overwrite: Download again even if cached.

    Returns:
        Source key -> local path.
    """
    paths = {}
    for key in keys:
        spec = SOURCES[key]
        if spec.url and key != "communes":
            paths[key] = download(spec.url, dest_dir, overwrite=overwrite)
    return paths
