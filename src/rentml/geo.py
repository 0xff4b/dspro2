"""Swiss geography: admin units and live geo.admin and GWR lookups.

Admin units come from swissBOUNDARIES3D (swisstopo). Live lookups for a single address use
api3.geo.admin.ch and the federal building and dwelling register (GWR) via the BFS MADD
service (eCH-0206). The app's map layers live in :mod:`rentml.maps`.
"""

import logging
import shutil
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from rentml.config import CANTON_ABBR, EXTERNAL, SB3D_VERSION, download

logger = logging.getLogger(__name__)

LV95 = 2056
WGS84 = 4326
SB3D_URL = (
    "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_{v}/"
    "swissboundaries3d_{v}_2056_5728.gpkg.zip"
)
SB3D_FILE = "swissBOUNDARIES3D_1_5_LV95_LN02.gpkg"
MUNICIPALITY_LAYER = "tlm_hoheitsgebiet"

UNIT_COLUMNS = [
    "municipality_id", "district_id", "canton_id", "municipality_name", "district_name",
    "canton", "canton_name", "muni_population", "muni_area_km2", "muni_density", "lang_region",
]  # fmt: skip
ADMIN_COLUMNS = [*UNIT_COLUMNS, "admin_match"]
COUNT_BUCKET_ORDER = ("<5", "5-20", ">20")

# Majority language per canton, corrected per district and municipality in bilingual cantons.
# Romansh areas count as "de" (the project distinguishes de/fr/it only); Biel/Bienne and the
# Upper Engadine (Maloja) stay "de" as their population is German-majority.
CANTON_LANGUAGE = dict.fromkeys(["GE", "VD", "NE", "JU", "FR", "VS"], "fr") | {"TI": "it"}
DISTRICT_LANGUAGE = {
    ("BE", "Jura bernois"): "fr",
    **{("FR", name): "de" for name in ("Sense", "See")},
    **{("VS", name): "de" for name in ("Goms", "Brig", "Visp", "Raron", "Leuk")},
    **{("GR", name): "it" for name in ("Moesa", "Bernina")},
}
MUNICIPALITY_LANGUAGE = {
    ("GR", "Bregaglia"): "it",
    **{("FR", name): "fr" for name in ("Courtepin", "Misery-Courtion", "Mont-Vully")},
}

API_BASE = "https://api3.geo.admin.ch/rest/services"
SEARCH_URL = f"{API_BASE}/api/SearchServer"
IDENTIFY_URL = f"{API_BASE}/api/MapServer/identify"
HEIGHT_URL = f"{API_BASE}/height"
MADD_URL = "https://madd.bfs.admin.ch/eCH-0206"
MADD_NS = {"m": "http://www.ech.ch/xmlns/eCH-0206/2"}
OEV_LAYER = "ch.are.erreichbarkeit-oev"
SOLAR_LAYER = "ch.bfe.solarenergie-eignung-daecher"
POPULATION_LAYER = "ch.bfs.volkszaehlung-bevoelkerungsstatistik_einwohner"
TIMEOUT_S = 12.0
DWELLING_EXISTING = "3004"

# GWR floor codes as mapped in the DSPRO1 app: (sort order, label).
FLOOR_LABELS = {
    0: (0, "EG"), 3100: (0, "EG"), 3413: (0, "EG"), 3500: (0, "EG"), 3300: (-1, "UG"),
    3401: (-1, "1. UG"), 3402: (-2, "2. UG"), 3403: (-3, "3. UG"),
    3601: (90, "1. DG"), 3602: (91, "2. DG"),
}  # fmt: skip
FLOOR_RANGES = ((3101, 3199, "Stock"), (3501, 3508, "OG"))
FLOOR_ATTIC_ORDER = 90  # attic floors sort last but have no floor number
# Midpoints of the GWR construction periods (GBAUP); "before 1919" is set to 1900.
PERIOD_MIDPOINT = {
    8011: 1900, 8012: 1932, 8013: 1953, 8014: 1966, 8015: 1976, 8016: 1983, 8017: 1988,
    8018: 1993, 8019: 1998, 8020: 2003, 8021: 2008, 8022: 2013, 8023: 2019,
}  # fmt: skip


def download_swissboundaries(dest_dir: Path = EXTERNAL, version: str = SB3D_VERSION) -> Path:
    """Download and unzip the swissBOUNDARIES3D GeoPackage unless it is already cached.

    The cached file is reused regardless of its release; delete it to fetch a new ``version``.
    """
    target = dest_dir / SB3D_FILE
    if target.is_file():
        return target
    url = SB3D_URL.format(v=version)
    archive = download(url, dest_dir / url.rsplit("/", 1)[-1])
    part = target.with_suffix(".part")
    with zipfile.ZipFile(archive) as zf:
        member = next(name for name in zf.namelist() if name.endswith(".gpkg"))
        with zf.open(member) as src, part.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    part.replace(target)
    archive.unlink()
    return target


def load_admin_units(gpkg: Path) -> gpd.GeoDataFrame:
    """Load Swiss municipalities with district, canton, population, land area and language.

    Cantonal lakes and foreign municipalities are dropped. Cantons without districts get
    ``district_id = canton_id * 100`` (as the BFS does) and the canton as district name. The
    area excludes lakes, so lakeside towns are not understated in density.
    """
    fields = ["objektart", "bfs_nummer", "bezirksnummer", "kantonsnummer", "name"]
    munis = gpd.read_file(
        gpkg,
        layer=MUNICIPALITY_LAYER,
        columns=[*fields, "einwohnerzahl", "gem_flaeche", "see_flaeche"],
    )
    munis = munis[munis["objektart"].eq("Gemeindegebiet") & munis["kantonsnummer"].notna()]
    canton_id = munis["kantonsnummer"].astype(int)
    district_id = munis["bezirksnummer"].fillna(canton_id * 100).astype(int)
    canton_names = _names(gpkg, "tlm_kantonsgebiet", "kantonsnummer")
    area_km2 = (munis["gem_flaeche"] - munis["see_flaeche"].fillna(0.0)) / 100.0
    area_km2 = area_km2.where(area_km2 > 0)
    population = munis["einwohnerzahl"].astype(float)
    units = gpd.GeoDataFrame(
        {
            "municipality_id": munis["bfs_nummer"].astype(int),
            "district_id": district_id,
            "canton_id": canton_id,
            "municipality_name": munis["name"],
            "district_name": district_id.map(
                _names(gpkg, "tlm_bezirksgebiet", "bezirksnummer")
            ).fillna(canton_id.map(canton_names)),
            "canton": canton_id.map(CANTON_ABBR),
            "canton_name": canton_id.map(canton_names),
            "muni_population": population,
            "muni_area_km2": area_km2,
            "muni_density": population / area_km2,
        },
        geometry=munis.geometry.force_2d(),
        crs=munis.crs,
    )
    units["lang_region"] = _language_region(units)
    return units.sort_values("municipality_id").reset_index(drop=True)


def load_foreign_mask(gpkg: Path) -> gpd.GeoSeries:
    """Polygons of Liechtenstein, Büsingen and Campione for ``assign_admin_units(exclude=...)``."""
    munis = gpd.read_file(gpkg, layer=MUNICIPALITY_LAYER, columns=["objektart", "kantonsnummer"])
    foreign = munis["objektart"].eq("Gemeindegebiet") & munis["kantonsnummer"].isna()
    return munis.loc[foreign].geometry.force_2d().reset_index(drop=True)


def _names(gpkg: Path, layer: str, key: str) -> dict[int, str]:
    table = gpd.read_file(gpkg, layer=layer, columns=[key, "name"], ignore_geometry=True)
    return dict(zip(table[key].astype(int), table["name"], strict=True))


def _language_region(units: pd.DataFrame) -> list[str]:
    rows = zip(units["canton"], units["district_name"], units["municipality_name"], strict=True)
    return [
        MUNICIPALITY_LANGUAGE.get((c, m))
        or DISTRICT_LANGUAGE.get((c, d))
        or CANTON_LANGUAGE.get(c, "de")
        for c, d, m in rows
    ]


def assign_admin_units(
    df: pd.DataFrame,
    units: gpd.GeoDataFrame,
    exclude: gpd.GeoSeries | None = None,
    max_snap_m: float = 2000.0,
) -> pd.DataFrame:
    """Attach the ``UNIT_COLUMNS`` of the municipality around each LV95 point (``east``, ``north``).

    ``admin_match`` is "within" for points inside a municipality, "nearest" for points snapped to
    the nearest one within ``max_snap_m`` (lakes, border lines) and "none" for points without
    coordinates or inside ``exclude`` (normally :func:`load_foreign_mask`).
    """
    units = units if units.crs.to_epsg() == LV95 else units.to_crs(LV95)
    units = units[[*UNIT_COLUMNS, "geometry"]]
    xy = df[["east", "north"]].apply(pd.to_numeric, errors="coerce")
    ok = xy.notna().all(axis=1).to_numpy()
    points = gpd.GeoDataFrame(
        {"_pos": np.flatnonzero(ok)},
        geometry=gpd.points_from_xy(xy.loc[ok, "east"], xy.loc[ok, "north"]),
        crs=LV95,
    )
    if exclude is not None and not exclude.empty:
        masked = gpd.sjoin(points, gpd.GeoDataFrame(geometry=exclude, crs=LV95), predicate="within")
        points = points.drop(index=masked.index)
    within = gpd.sjoin(points, units, predicate="within").assign(_d=0.0, admin_match="within")
    rest = points[~points["_pos"].isin(within["_pos"])]
    nearest = gpd.sjoin_nearest(rest, units, max_distance=max_snap_m, distance_col="_d")
    attrs = (
        pd.concat([within, nearest.assign(admin_match="nearest")])
        .sort_values(["_pos", "_d", "municipality_id"])
        .drop_duplicates("_pos")
        .set_index("_pos")
        .reindex(range(len(df)))
    )
    attrs["admin_match"] = attrs["admin_match"].fillna("none")
    out = df.drop(columns=ADMIN_COLUMNS, errors="ignore")
    for col in ADMIN_COLUMNS:
        values = attrs[col].astype("Int64") if col.endswith("_id") else attrs[col]
        out[col] = values.array  # positional, so duplicate index labels are safe
    logger.info("Admin assignment: %s", out["admin_match"].value_counts().to_dict())
    return out


def nested_group_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Random-effect keys ``re_canton`` "ZH", ``re_district`` "ZH/112", ``re_municipality``
    "ZH/112/261"; NaN where a level is missing."""
    canton = df["canton"].astype("string")
    district = canton + "/" + pd.Series(df["district_id"], dtype="Int64").astype("string")
    muni = district + "/" + pd.Series(df["municipality_id"], dtype="Int64").astype("string")
    keys = pd.DataFrame({"re_canton": canton, "re_district": district, "re_municipality": muni})
    return keys.astype(object).where(keys.notna(), np.nan)


def count_bucket(counts: pd.Series) -> pd.Series:
    """Bucket listings per unit into the ordered categories ``COUNT_BUCKET_ORDER`` (NaN = 0)."""
    bins = [-np.inf, 5, 21, np.inf]
    return pd.cut(counts.fillna(0), bins, right=False, labels=list(COUNT_BUCKET_ORDER))


@dataclass(frozen=True)
class Address:
    """A geocoded Swiss address with LV95 coordinates and the GWR building id (EGID)."""

    label: str
    east: float
    north: float
    egid: int | None = None


@dataclass(frozen=True)
class Dwelling:
    """One dwelling of a building in the GWR."""

    ewid: int
    label: str
    floor_label: str
    rooms: float | None
    area: float | None
    floor: float | None = None  # 0 = ground floor, -1 = basement; None if unknown or attic

    @property
    def display(self) -> str:
        """Text such as "EWID 2 · 101 · 1. Stock · 2 Zi · 46 m²"."""
        rooms = f"{self.rooms:g} Zi" if self.rooms else ""
        area = f"{self.area:g} m²" if self.area else ""
        parts = [f"EWID {self.ewid}", self.label, self.floor_label, rooms, area]
        return " · ".join(part for part in parts if part)


@dataclass
class LocationData:
    """Building and location inputs of the rent model for one address (``None`` = unknown)."""

    year_built: float | None = None
    apartments: float | None = None
    land_area: float | None = None
    oev: float | None = None
    solar: float | None = None
    population: float | None = None
    elevation: float | None = None
    missing: list[str] = field(default_factory=list)
    dwellings: list[Dwelling] = field(default_factory=list)

    def features(self) -> dict[str, float | None]:
        """The model inputs (BUILDING and LOCATION columns)."""
        names = ("year_built", "apartments", "land_area", "oev", "solar", "population", "elevation")
        return {name: getattr(self, name) for name in names}


def _get(url: str, params: dict, session: requests.Session | None) -> requests.Response | None:
    try:
        response = (session or requests).get(url, params=params, timeout=TIMEOUT_S)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Request to %s failed: %s", url, exc)
        return None
    return response


def _get_json(url: str, params: dict, session: requests.Session | None) -> dict:
    response = _get(url, params, session)
    try:
        return response.json() if response is not None else {}
    except requests.JSONDecodeError:  # a RequestException subclass, but raised outside _get
        logger.warning("Invalid JSON from %s", url)
        return {}


def _number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def search_addresses(
    query: str, limit: int = 8, session: requests.Session | None = None
) -> list[Address]:
    """Address suggestions from the geo.admin SearchServer (empty for queries under 3 chars)."""
    text = " ".join(query.split())
    if len(text) < 3:
        return []
    params = {"searchText": text, "type": "locations", "origins": "address", "sr": LV95}
    data = _get_json(SEARCH_URL, params | {"limit": limit}, session)
    found: dict[str, Address] = {}
    for item in data.get("results", []):
        attrs = item.get("attrs", {})
        label = attrs.get("label", "").replace("<b>", "").replace("</b>", "").strip()
        east, north = _number(attrs.get("y")), _number(attrs.get("x"))  # y = east in LV95
        if not label or east is None or north is None or label in found:
            continue
        egid = str(attrs.get("featureId", "")).split("_")[0]  # "<EGID>_<EDID>" (entrance id)
        found[label] = Address(label, east, north, int(egid) if egid.isdigit() else None)
    return list(found.values())


def _madd_building(egid: int, session: requests.Session | None) -> ET.Element | None:
    response = _get(MADD_URL, {"egid": egid}, session)
    if response is None:
        return None
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        logger.warning("Invalid XML from MADD for EGID %s", egid)
        return None
    return root.find(".//m:buildingItem", MADD_NS)


def _floor(code: float | None) -> tuple[float, str]:
    if code is None:
        return 99, ""
    code = int(code)
    if code in FLOOR_LABELS:
        return FLOOR_LABELS[code]
    for start, stop, suffix in FLOOR_RANGES:
        if start <= code <= stop:
            return code - start + 1, f"{code - start + 1}. {suffix}"
    return 99, str(code)


def _parse_dwellings(building: ET.Element) -> list[Dwelling]:
    rows = []
    for item in building.iterfind(".//m:dwellingItem", MADD_NS):
        values = {child.tag.split("}")[1]: child.text for child in item.find("m:dwelling", MADD_NS)}
        if values.get("dwellingStatus") not in (None, DWELLING_EXISTING):
            continue
        order, floor_label = _floor(_number(values.get("floor")))
        label = values.get("administrativeDwellingNo") or ""
        dwelling = Dwelling(
            ewid=int(item.findtext("m:EWID", namespaces=MADD_NS)),
            label=label,
            floor_label=floor_label,
            rooms=_number(values.get("noOfHabitableRooms")),
            area=_number(values.get("surfaceAreaOfDwelling")),
            floor=float(order) if floor_label and order < FLOOR_ATTIC_ORDER else None,
        )
        rows.append((order, label, dwelling))
    return [row[2] for row in sorted(rows, key=lambda row: row[:2])]


def building_dwellings(egid: int, session: requests.Session | None = None) -> list[Dwelling]:
    """Existing dwellings of a GWR building, sorted by floor and dwelling number."""
    building = _madd_building(egid, session)
    return [] if building is None else _parse_dwellings(building)


def _building_attributes(
    building: ET.Element | None, dwellings: list[Dwelling]
) -> dict[str, float | None]:
    if building is None:
        return {"year_built": None, "apartments": None, "land_area": None}
    date = building.findtext(".//m:dateOfConstruction/m:dateOfConstruction", namespaces=MADD_NS)
    period = building.findtext(".//m:dateOfConstruction/m:periodOfConstruction", namespaces=MADD_NS)
    area = building.findtext(".//m:building/m:surfaceAreaOfBuilding", namespaces=MADD_NS)
    # Many buildings only carry a construction period: take its midpoint. The dwellings' year is
    # no fallback, because the GWR often fills it with the placeholder 1999.
    midpoint = PERIOD_MIDPOINT.get(int(period)) if period and period.isdigit() else None
    return {
        "year_built": _number((date or "")[:4]) or _number(midpoint),
        "apartments": float(len(dwellings)) if dwellings else None,  # GWR: no count when 0
        "land_area": _number(area),
    }


def _location_attributes(
    east: float, north: float, session: requests.Session | None
) -> dict[str, float | None]:
    identify = _get_json(IDENTIFY_URL, location_params(east, north), session)
    height = _get_json(HEIGHT_URL, {"easting": east, "northing": north}, session)
    return parse_location(identify, height)


def location_params(east: float, north: float) -> dict[str, str | int]:
    """Query parameters of the geo.admin identify call for the location layers at a point.

    Args:
        east: LV95 east (m).
        north: LV95 north (m).

    Returns:
        Parameters for :data:`IDENTIFY_URL` (ÖV, solar and population layers, 1 px tolerance).
    """
    return {
        "geometry": f"{east},{north}",
        "geometryType": "esriGeometryPoint",
        "layers": f"all:{OEV_LAYER},{SOLAR_LAYER},{POPULATION_LAYER}",
        "tolerance": 1,
        "returnGeometry": "false",
        "sr": LV95,
        "imageDisplay": "100,100,96",
        "mapExtent": f"{east - 10},{north - 10},{east + 10},{north + 10}",
    }


def parse_location(identify: dict, height: dict) -> dict[str, float | None]:
    """Location inputs from the identify and height responses of geo.admin.

    Args:
        identify: JSON of the identify call with :func:`location_params`.
        height: JSON of the height service.

    Returns:
        ``oev``, ``solar``, ``population`` (latest STATPOP year) and ``elevation``.
    """
    by_layer: dict[str, list[dict]] = {}
    for item in identify.get("results", []):
        by_layer.setdefault(item.get("layerBodId"), []).append(item.get("attributes", {}))
    oev = by_layer.get(OEV_LAYER, [{}])[0].get("oev_erreichb_ewap")
    solar = by_layer.get(SOLAR_LAYER, [{}])[0].get("klasse")
    census = max(by_layer.get(POPULATION_LAYER, [{}]), key=lambda a: a.get("i_year", 0))
    return {
        "oev": _number(oev),
        "solar": _number(solar),
        "population": _number(census.get("number")),  # latest STATPOP year
        "elevation": _number(height.get("height")),
    }


def lookup_location(address: Address, session: requests.Session | None = None) -> LocationData:
    """All building and location inputs for one address, plus the building's dwellings."""
    building = _madd_building(address.egid, session) if address.egid else None
    dwellings = [] if building is None else _parse_dwellings(building)
    values = _building_attributes(building, dwellings)
    values |= _location_attributes(address.east, address.north, session)
    missing = [name for name, value in values.items() if value is None]
    return LocationData(**values, missing=missing, dwellings=dwellings)
