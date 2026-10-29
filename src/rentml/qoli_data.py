"""Data of the Quality-of-Life Index (QoLI): source registry, cached downloads, raw indicators.

Point indicators in LV95 (EPSG:2056 metres) from official geodata and OpenStreetMap, and municipal
statistics (BFS SDMX, ESTV tax burden, BFS commune correspondence).
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import Window
from scipy.spatial import cKDTree

from rentml.config import download

logger = logging.getLogger(__name__)

HTTP_HEADERS = {"User-Agent": "rentml/0.1 (HSLU DSPRO2 student project)"}
STAC_BASE = "https://data.geo.admin.ch"
SDMX_BASE = "https://disseminate.stats.swiss/rest/data"
COMMUNES_API = "https://www.agvchapp.bfs.admin.ch/api/communes/correspondances"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_SDMX_CSV = "application/vnd.sdmx.data+csv;version=1.0.0;labels=both"
_OGD = "OGD (opendata.swiss 'open use, must provide the source')"
_SONBASE = f"{_OGD}; source 'BAFU 2025, sonBASE'"


@dataclass(frozen=True)
class Source:
    """One QoLI dataset with what the datasheet needs (``url`` empty for API sources)."""

    key: str
    title: str
    provider: str
    licence: str
    reference: str
    resolution: str
    unit: str
    use: str
    url: str = ""

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def _stac(collection: str, item: str, asset: str) -> str:
    return f"{STAC_BASE}/{collection}/{item}/{asset}"


def _noise_source(key: str, kind: str, period: str, label: str) -> Source:
    name = f"laerm-{kind}_{period}"
    return Source(
        key, f"sonBASE {label}", "BAFU", _SONBASE, "2021 (computed 2023)", "10 m raster", "dB(A)",
        "dimension noise; LSV/WHO thresholds", _stac(f"ch.bafu.{name}", name, f"{name}_2056.tif"),
    )  # fmt: skip


SOURCES: dict[str, Source] = {s.key: s for s in [
    Source("oev", "Public-transport quality classes (ÖV-Güteklassen) and classified stops", "ARE",
           _OGD, "timetable 2026", "polygons / stop points", "class 0-4; m", "dimension transport",
           _stac("ch.are.gueteklassen_oev", "gueteklassen_oev_2026",
                 "gueteklassen_oev_2026_2056.gpkg.zip")),
    _noise_source("road_day", "strassenlaerm", "tag", "road traffic noise, day (L_r,Tag 06-22 h)"),
    _noise_source("road_night", "strassenlaerm", "nacht",
                  "road traffic noise, night (L_r,Nacht 22-06 h)"),
    _noise_source("rail_day", "bahnlaerm", "tag", "railway noise, day (L_r,Tag 06-22 h)"),
    _noise_source("rail_night", "bahnlaerm", "nacht", "railway noise, night (L_r,Nacht 22-06 h)"),
    Source("landuse", "Land-use statistics (Arealstatistik), NOAS04 hectare points", "BFS", _OGD,
           "survey flights 2013-2020 (release 2024)", "100 m hectare points", "share 0-1",
           "dimension green",
           _stac("ch.bfs.arealstatistik", "arealstatistik", "arealstatistik_2056.csv.zip")),
    Source("hectares", "Service accessibility per inhabited hectare (STATPOP, distances)", "BFS",
           _OGD, "2021", "100 m hectares", "inhabitants; m",
           "reference grid (normalisation, municipal aggregates); validation of OSM access",
           _stac("ch.bfs.erreichbarkeit", "erreichbarkeit", "erreichbarkeit_2056.csv")),
    Source("no2", "PolluMap nitrogen dioxide (NO2), annual mean", "BAFU", _OGD, "2025",
           "20 m raster", "µg/m³", "dimension air (extended index)",
           _stac("ch.bafu.luftreinhaltung-stickstoffdioxid",
                 "luftreinhaltung-stickstoffdioxid_2025",
                 "luftreinhaltung-stickstoffdioxid_2025_2056.tif")),
    Source("pm25", "PolluMap fine particulate matter (PM2.5), annual mean", "BAFU", _OGD, "2025",
           "100 m raster", "µg/m³", "dimension air (extended index)",
           _stac("ch.bafu.luftreinhaltung-feinstaub_pm2_5",
                 "luftreinhaltung-feinstaub_pm2_5_2025",
                 "luftreinhaltung-feinstaub_pm2_5_2025_2056.tif")),
    Source("sunshine", "Relative sunshine duration, climate normal 1991-2020", "MeteoSwiss",
           "MeteoSwiss OGD (open use, source required)", "1991-2020", "1 km grid",
           "% of possible", "dimension sunshine (extended index)",
           _stac("ch.meteoschweiz.ogd-climate-normals-grid", "ch",
                 "ogd-climate-normals-grid.snormy9120_ch01r.swiss.lv95_"
                 "19910101000000_19910101000000.tif")),
    Source("tax", "Tax burden in the municipalities (cantonal, municipal and church tax)", "ESTV",
           "public statistics (no explicit licence; source required)", "2018 (latest file)",
           "municipality (remapped to 2026)", "% of gross income", "dimension tax (extended index)",
           "https://www.estv.admin.ch/dam/de/sd-web/u8ENqfdNX1ci/"
           "statistik-belastung-gden-2018-de.xlsx"),
    Source("vacancy", "Vacant dwellings and vacancy rate on 1 June (DF_LWZ_1)", "BFS", _OGD,
           "2026", "municipality", "% of dwellings",
           "dimension housing availability (extended index); validation of the core index"),
    Source("city_statistics", "City Statistics (Urban Audit): selected variables (DF_CITYSTAT_1)",
           "BFS", _OGD, "latest year per variable (2018-2025)", "10 core cities", "various",
           "validation (descriptive, n = 10)"),
    Source("communes", "Historicised commune register: correspondence of municipality numbers",
           "BFS", _OGD, "2018 -> 2026", "municipality", "-", "remap ESTV 2018 to 2026 boundaries"),
]}  # fmt: skip


def source_table() -> pd.DataFrame:
    """Datasheet of all QoLI sources, one row per dataset, indexed by ``key``."""
    return pd.DataFrame([asdict(s) for s in SOURCES.values()]).set_index("key")


def _download(
    url: str, target: Path, params: dict[str, str] | None = None, accept: str = "*/*"
) -> Path:
    return download(url, target, params, HTTP_HEADERS | {"Accept": accept})


def fetch_sources(dest_dir: Path) -> dict[str, Path]:
    """Download every file source once into ``dest_dir`` (about 1.8 GB); key -> local path."""
    return {k: _download(s.url, dest_dir / s.filename) for k, s in SOURCES.items() if s.url}


# Hectare grids of the BFS give the south-west corner of each 100 m cell.
HECTARE_OFFSET_M = 50.0
HECTARE_SERVICES = ["d_grocery", "d_pharma", "d_medic", "d_school_o", "d_resto", "d_stop_tot"]


def load_hectares(csv_path: Path, year: int = 2021) -> pd.DataFrame:
    """Inhabited hectares (index ``reli``, cell centres) with population and service distances."""
    cols = ["RELI", "E_COORD", "N_COORD", "YEAR", "POP_TOTAL", *map(str.upper, HECTARE_SERVICES)]
    raw = pd.read_csv(csv_path, sep=";", usecols=cols)
    raw = raw.loc[raw["YEAR"].eq(year)].rename(columns=str.lower)
    out = raw[["e_coord", "n_coord", "pop_total", *HECTARE_SERVICES]].astype(float)
    out = out.rename(columns={"e_coord": "east", "n_coord": "north", "pop_total": "population"})
    out[["east", "north"]] += HECTARE_OFFSET_M
    return out.set_index(pd.Index(raw["reli"].astype("int64"), name="reli"))


# OpenStreetMap amenities (ODbL): Overpass tag filter per category.
OSM_CATEGORIES = {
    "supermarket": '["shop"="supermarket"]',
    "pharmacy": '["amenity"="pharmacy"]',
    "doctors": '["amenity"~"^(doctors|clinic)$"]',
    "school": '["amenity"="school"]',
    "kindergarten": '["amenity"~"^(kindergarten|childcare)$"]',
    "gym": '["leisure"="fitness_centre"]',
    "restaurant": '["amenity"~"^(restaurant|cafe)$"]',
    "park": '["leisure"="park"]',
}
SWISS_BBOX_WGS84 = (45.77, 5.90, 47.85, 10.55)  # south, west, north, east; ~4 km border margin
# Walk-Score weights of the n nearest amenities; other categories count their nearest one.
ACCESS_WEIGHTS = {
    "supermarket": [3.0],
    "restaurant": [0.75, 0.45, 0.25, 0.25, 0.225, 0.225, 0.225, 0.225, 0.2, 0.2],
}
DECAY_FULL_UNTIL_M = 400.0  # ~5 min walk
DECAY_ZERO_AT_M = 2400.0  # ~30 min walk


def fetch_osm_amenities(cache_path: Path) -> pd.DataFrame:
    """OSM amenities of ``OSM_CATEGORIES`` (LV95 ``east``/``north``) via Overpass, cached.

    Raises:
        RuntimeError: If Overpass returns an incomplete result.
    """
    if cache_path.is_file():
        return pd.read_parquet(cache_path)
    box = ",".join(str(v) for v in SWISS_BBOX_WGS84)
    parts = []
    for cat, tags in OSM_CATEGORIES.items():
        elements = _overpass(f"[out:json][timeout:300];nwr{tags}({box});out tags center;")
        parts.append(_osm_frame(cat, elements))
        time.sleep(2.0)  # Overpass fair use
    osm = pd.concat(parts, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    osm.to_parquet(cache_path, index=False)
    return osm


# Overpass JSON elements are untyped (library boundary), hence dict[str, Any].
def _overpass(query: str, retries: int = 3) -> list[dict[str, Any]]:
    for attempt in range(retries + 1):
        resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HTTP_HEADERS, timeout=360)
        if resp.status_code not in (429, 502, 503, 504) or attempt == retries:
            break
        logger.warning("Overpass HTTP %d; retry %d/%d", resp.status_code, attempt + 1, retries)
        time.sleep(30.0 * 2**attempt)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in str(payload.get("remark", "")).lower():
        raise RuntimeError(f"Overpass returned an incomplete result: {payload['remark']}")
    return payload.get("elements", [])


def _osm_frame(cat: str, elements: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for el in elements:  # nodes carry lat/lon, ways and relations a "center"
        pos = el.get("center", el)
        name = el.get("tags", {}).get("name")
        rows.append((cat, el.get("type"), el.get("id"), name, pos.get("lon"), pos.get("lat")))
    columns = ["category", "osm_type", "osm_id", "name", "lon", "lat"]
    osm = pd.DataFrame(rows, columns=columns).dropna(subset=["osm_id", "lon", "lat"])
    osm = osm.drop_duplicates(["osm_type", "osm_id"]).reset_index(drop=True)
    osm = osm.astype({"osm_id": "int64", "lon": float, "lat": float, "name": "string"})
    lv95 = gpd.GeoSeries(gpd.points_from_xy(osm["lon"], osm["lat"]), crs=4326).to_crs(2056)
    return osm.assign(east=lv95.x.to_numpy(), north=lv95.y.to_numpy())


def amenity_arrays(osm: pd.DataFrame) -> dict[str, np.ndarray]:
    """LV95 coordinates (m, 2) per category of ``OSM_CATEGORIES`` (empty if none were found)."""
    xy = osm[["east", "north"]].to_numpy(float)
    return {cat: xy[osm["category"].eq(cat).to_numpy()] for cat in OSM_CATEGORIES}


def accessibility(points: pd.DataFrame, amenities: dict[str, np.ndarray]) -> pd.DataFrame:
    """Walk-Score-style accessibility (0-100) at ``points[["east", "north"]]``.

    The i-th nearest amenity of a category adds ``ACCESS_WEIGHTS[i]`` times a cosine distance
    decay (1 up to 400 m, 0 from 2400 m). Returns ``acc_<category>`` and ``acc_total``, which
    weights the categories by their weight sum; rows without coordinates get NaN.
    """
    xy = points[["east", "north"]].to_numpy(float, na_value=np.nan)
    ok = np.isfinite(xy).all(axis=1)
    out, total, weight_sum = {}, 0.0, 0.0
    for cat, coords in amenities.items():
        w = np.asarray(ACCESS_WEIGHTS.get(cat, [1.0]))
        raw = np.zeros(ok.sum())
        if len(coords):
            dist, _ = cKDTree(coords).query(xy[ok], k=len(w), distance_upper_bound=DECAY_ZERO_AT_M)
            span = DECAY_ZERO_AT_M - DECAY_FULL_UNTIL_M
            t = (dist.reshape(len(raw), len(w)) - DECAY_FULL_UNTIL_M) / span
            raw = 0.5 * (1.0 + np.cos(np.pi * np.clip(t, 0.0, 1.0))) @ w
        score = np.full(len(xy), np.nan)
        score[ok] = raw
        out[f"acc_{cat}"] = 100.0 * score / w.sum()
        total, weight_sum = total + score, weight_sum + w.sum()
    out["acc_total"] = 100.0 * total / weight_sum
    return pd.DataFrame(out, index=points.index)


def raster_values(path: Path, xy: np.ndarray) -> np.ndarray:
    """Value of the raster cell containing each LV95 point (NaN for nodata or outside)."""
    with rasterio.open(path) as src:
        values = np.fromiter((v[0] for v in src.sample(xy)), float, len(xy))
        nodata = src.nodata
    values[values == nodata] = np.nan
    return values


# sonBASE carries no level inside buildings, while modelled cells go down to about 0 dB. Without a
# value in its own cell a point takes the loudest cell of the smallest square around it that has
# values (the most exposed façade), else it is "quiet". A fixed 20 m square often picked the road
# surface and tripled the share above the LSV limits compared with the BAFU statistics.
NOISE_SEARCH_RADII_M = (10.0, 20.0, 50.0, 100.0)
NOISE_FACADE_MAX_M = 20.0  # larger radii count as "fallback"


def noise_levels(path: Path, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Modelled noise level of a sonBASE raster at LV95 points, with the façade search.

    Returns:
        Levels in dB(A) (NaN if quiet or outside) and how each was found: "point", "facade",
        "fallback", "quiet" or "outside".
    """
    level = raster_values(path, xy)
    with rasterio.open(path) as src:
        rows, cols = rasterio.transform.rowcol(src.transform, xy[:, 0], xy[:, 1])
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        how = np.where(inside, "point", "outside").astype(object)
        cells = [int(np.ceil(r / src.res[0])) for r in NOISE_SEARCH_RADII_M]
        for i in np.flatnonzero(inside & np.isnan(level)):
            maxima = _square_maxima(src, rows[i], cols[i], cells)
            found = np.flatnonzero(np.isfinite(maxima))
            if found.size == 0:
                how[i] = "quiet"
                continue
            level[i] = maxima[found[0]]
            facade = NOISE_SEARCH_RADII_M[found[0]] <= NOISE_FACADE_MAX_M
            how[i] = "facade" if facade else "fallback"
    return level, how.astype(str)


def _square_maxima(src: rasterio.DatasetReader, row: int, col: int, cells: list[int]) -> np.ndarray:
    k = max(cells)
    window = Window(col - k, row - k, 2 * k + 1, 2 * k + 1).intersection(
        Window(0, 0, src.width, src.height)
    )
    block = src.read(1, window=window, out_dtype="float64")
    block[block == src.nodata] = np.nan
    r, c = row - int(window.row_off), col - int(window.col_off)
    maxima = np.full(len(cells), np.nan)
    for j, n in enumerate(cells):
        square = block[max(r - n, 0) : r + n + 1, max(c - n, 0) : c + n + 1]
        if np.isfinite(square).any():
            maxima[j] = np.nanmax(square)
    return maxima


def _energetic_sum(a: pd.Series, b: pd.Series) -> pd.Series:
    """10·log10(10^(a/10) + 10^(b/10)); a missing level counts as no sound."""
    energy = (10.0 ** (a / 10.0)).add(10.0 ** (b / 10.0), fill_value=0.0)
    return 10.0 * np.log10(energy)


# BFS land-use AS_17 categories counted as green: 5 recreation and parks, 10 forest, 11 brush
# forest, 12 woods. Water, farmland and alpine pasture are not counted.
GREEN_AS17 = (5, 10, 11, 12)
GREEN_RADIUS_M = 300.0  # WHO Europe: green space within 300 m (~5 min walk)


def load_landuse(csv_zip: Path) -> pd.DataFrame:
    """Hectare centres (``east``/``north``) of the land-use statistics with their ``as17`` class."""
    raw = pd.read_csv(csv_zip, sep=";", usecols=["E_COORD", "N_COORD", "AS_17"])
    out = raw.rename(columns={"E_COORD": "east", "N_COORD": "north", "AS_17": "as17"})
    out[["east", "north"]] = out[["east", "north"]].astype(float) + HECTARE_OFFSET_M
    return out


def green_share(
    xy: np.ndarray, landuse: pd.DataFrame, radius_m: float = GREEN_RADIUS_M
) -> np.ndarray:
    """Share of the land-use hectares within ``radius_m`` that are green or wooded."""
    coords = landuse[["east", "north"]].to_numpy(float)
    green = landuse["as17"].isin(GREEN_AS17).to_numpy()
    n_all = cKDTree(coords).query_ball_point(xy, radius_m, return_length=True)
    n_green = cKDTree(coords[green]).query_ball_point(xy, radius_m, return_length=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_all > 0, n_green / n_all, np.nan)


OEV_CLASS_SCORE = {"A": 4, "B": 3, "C": 2, "D": 1}  # ARE classes; outside all classes = 0


def load_oev(gpkg_zip: Path) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """ARE quality-class polygons (``score`` 1-4) and the coordinates of classified stops."""
    classes = gpd.read_file(gpkg_zip, layer="OeV_Gueteklassen_ARE", columns=["KLASSE"])
    classes["score"] = classes["KLASSE"].map(OEV_CLASS_SCORE)
    stops = gpd.read_file(gpkg_zip, layer="OeV_Haltestellen_ARE", columns=["Hst_Kat"])
    stops = stops.loc[stops["Hst_Kat"].between(1, 5)]
    return classes[["score", "geometry"]], stops.get_coordinates().to_numpy()


def oev_class(xy: np.ndarray, classes: gpd.GeoDataFrame) -> np.ndarray:
    """Best public-transport quality class at each point (A = 4 ... D = 1, none = 0)."""
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=2056)
    hits = gpd.sjoin(points, classes, predicate="within")
    return hits.groupby(level=0)["score"].max().reindex(points.index, fill_value=0).to_numpy(float)


NOISE_KEYS = ("road_day", "road_night", "rail_day", "rail_night")
RASTER_KEYS = ("no2", "pm25", "sunshine")
LAYER_KEYS = ("oev", *NOISE_KEYS, "landuse", *RASTER_KEYS)


@dataclass
class QoliLayers:
    """Loaded QoLI sources that turn LV95 points into raw indicators (rasters are read lazily)."""

    paths: dict[str, Path]
    oev_classes: gpd.GeoDataFrame
    oev_stops: np.ndarray
    landuse: pd.DataFrame
    amenities: dict[str, np.ndarray]

    @classmethod
    def load(cls, paths: dict[str, Path], osm: pd.DataFrame | None = None) -> "QoliLayers":
        """Read the sources of ``LAYER_KEYS`` and the OSM table of ``fetch_osm_amenities``."""
        classes, stops = load_oev(paths["oev"])
        amenities = amenity_arrays(osm) if osm is not None else {}
        return cls(dict(paths), classes, stops, load_landuse(paths["landuse"]), amenities)

    def indicators(self, points: pd.DataFrame) -> pd.DataFrame:
        """Raw indicators at the LV95 ``east``/``north`` of ``points`` (index kept).

        Noise: per source and period, the energetic road + rail sum ``noise_<period>_db`` and
        ``noise_quiet_<period>``; ``acc_total_all`` with and ``acc_total`` without parks.
        """
        xy = points[["east", "north"]].to_numpy(float)
        out = pd.DataFrame(index=points.index)
        out["oev_class"] = oev_class(xy, self.oev_classes)
        out["oev_stop_m"] = cKDTree(self.oev_stops).query(xy)[0]
        how = {}
        for key in NOISE_KEYS:
            out[f"{key}_db"], how[key] = noise_levels(self.paths[key], xy)
        out["noise_how"] = how["road_day"]
        for period in ("day", "night"):
            road, rail = out[f"road_{period}_db"], out[f"rail_{period}_db"]
            out[f"noise_{period}_db"] = _energetic_sum(road, rail)
            # Outside the rail raster counts as "no rail noise", not as unknown.
            out[f"noise_quiet_{period}"] = (how[f"road_{period}"] == "quiet") & rail.isna()
        out["green_share"] = green_share(xy, self.landuse)
        if self.amenities:
            access = accessibility(points, self.amenities)  # all categories, incl. parks
            out = out.join(access.rename(columns={"acc_total": "acc_total_all"}))
            no_park = {k: v for k, v in self.amenities.items() if k != "park"}
            out["acc_total"] = accessibility(points, no_park)["acc_total"]
        for key in RASTER_KEYS:
            out[key] = raster_values(self.paths[key], xy)
        return out


def fetch_sdmx_csv(
    flow: str, cache_path: Path, params: dict[str, str] | None = None
) -> pd.DataFrame:
    """BFS SDMX dataflow such as ``"CH1.LWZ,DF_LWZ_1,1.1.0"`` as CSV (codes and labels), cached."""
    path = _download(f"{SDMX_BASE}/{flow}/all", cache_path, params, accept=_SDMX_CSV)
    return pd.read_csv(path, dtype=str)


def _sdmx_codes(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """``columns`` of a "<ID>: <label>" SDMX table with the code part of each value."""
    codes = table.rename(columns=lambda c: c.split(":")[0])[columns]
    return codes.apply(lambda col: col.str.split(":", n=1).str[0].str.strip())


def vacancy_rates(table: pd.DataFrame) -> pd.Series:
    """Municipal vacancy rate (%, all dwellings) of the latest period in the DF_LWZ_1 table."""
    dims = ["GR_KT_GDE", "WOHN_ANZAHL", "LEERWOHN_TYP", "MEASURE_DIMENSION", "TIME_PERIOD"]
    t = _sdmx_codes(table, [*dims, "OBS_VALUE"])
    keep = (
        t["WOHN_ANZAHL"].eq("_T")
        & t["LEERWOHN_TYP"].eq("_T")
        & t["MEASURE_DIMENSION"].eq("RATE")
        & t["GR_KT_GDE"].str.fullmatch(r"\d+")  # municipalities only
    )
    t = t[keep & t["TIME_PERIOD"].eq(t.loc[keep, "TIME_PERIOD"].max())]
    rate = pd.to_numeric(t["OBS_VALUE"], errors="coerce").rename("vacancy_rate")
    index = pd.Index(t["GR_KT_GDE"].astype("int64"), name="municipality_id")
    return rate.set_axis(index).dropna().sort_index()


def city_statistics(table: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Latest observed value per core city (``municipality_id``) and City Statistics variable."""
    dims = ["ID_RAUM", "ID_INDIC_DISPR", "STATISTICAL_OPERATION", "TIME_PERIOD"]
    t = _sdmx_codes(table, [*dims, "OBS_VALUE"])
    t["OBS_VALUE"] = pd.to_numeric(t["OBS_VALUE"], errors="coerce")
    t = t[
        t["ID_RAUM"].str.endswith("901")  # core city "<BFS number>901"
        & t["ID_INDIC_DISPR"].isin(variables)
        & t["STATISTICAL_OPERATION"].eq("OBS")
        & t["OBS_VALUE"].notna()
    ]
    t["municipality_id"] = t["ID_RAUM"].str[:-3].astype("int64")
    latest = t.sort_values("TIME_PERIOD").groupby(["municipality_id", "ID_INDIC_DISPR"])
    return latest["OBS_VALUE"].last().unstack().reindex(columns=variables)


def load_tax_burden(xlsx: Path, profile: str = "Ledig", gross_income: int = 80_000) -> pd.Series:
    """ESTV cantonal, municipal and church tax in % of gross income per municipality number.

    ``profile`` is the sheet of the household type (``"Ledig"``, ``"VOK"``, ...), ``gross_income``
    a column of that sheet.
    """
    raw = pd.read_excel(xlsx, sheet_name=profile, header=None)
    header = raw.index[raw.iloc[:, 1].astype(str).str.contains("Gemeindenummer", na=False)][0]
    column = list(pd.to_numeric(raw.iloc[header - 1], errors="coerce")).index(gross_income)
    body = raw.iloc[header + 1 :]
    ids = pd.to_numeric(body.iloc[:, 1], errors="coerce")
    values = pd.to_numeric(body.iloc[:, column], errors="coerce")
    ok = ids.notna() & values.notna()
    index = pd.Index(ids[ok].astype("int64"), name="municipality_id")
    return pd.Series(values[ok].to_numpy(float), index=index, name="tax_burden_pct")


def fetch_commune_correspondence(start: str, end: str, cache_path: Path) -> pd.DataFrame:
    """BFS municipality numbers between two dates (``DD-MM-YYYY``): initial and terminal id/name."""
    params = {"includeUnmodified": "true", "includeTerritoryExchange": "false",
              "startPeriod": start, "endPeriod": end}  # fmt: skip
    raw = pd.read_csv(_download(COMMUNES_API, cache_path, params))
    names = {"InitialCode": "initial_id", "InitialName": "initial_name",
             "TerminalCode": "terminal_id", "TerminalName": "terminal_name"}  # fmt: skip
    return raw[list(names)].rename(columns=names).drop_duplicates(["initial_id", "terminal_id"])


def remap_municipalities(values: pd.Series, correspondence: pd.DataFrame) -> pd.Series:
    """Carry municipal values to the terminal municipality numbers (a merger gets the mean)."""
    mapped = correspondence.merge(values.rename("value"), left_on="initial_id", right_index=True)
    out = mapped.groupby("terminal_id")["value"].mean().rename(values.name)
    return out.rename_axis("municipality_id")
