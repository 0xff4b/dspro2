"""Map layers for the Dash app: exact LV95 geometry and listing aggregates per admin level.

Geometry: the swissBOUNDARIES3D municipalities and the non-municipal cantonal areas (lakes, the
Galm state forest, Kommunanz) are simplified together with ``shapely.coverage_simplify``, so
neighbours keep identical shared borders, and dissolved into districts and cantons with
``shapely.coverage_union_all``. The canton outlines are therefore the union of their
municipalities, without gaps or overlaps.

Projection: the map is drawn in LV95 (EPSG:2056, the official Swiss oblique Mercator), without
reprojecting to Web Mercator. Plotly's geo subplot only takes lon/lat, so LV95 metres are mapped
affinely to "plot degrees" (1° = ``PLOT_SCALE_M`` around the LV95 origin in Bern) and drawn with
the equirectangular projection, which is the identity (x = λ, y = φ). The rendered map is LV95 up
to a uniform scale. Because Switzerland spans only about ±2 plot degrees around the equator, the
great-circle edges that d3 draws between two vertices stay within about a metre of the straight
LV95 segment.

Aggregates: one row per object (after deduplication) per canton, district and municipality.
Medians are published only where a unit has at least ``MIN_MAP_CELL`` objects (proposal: smaller
cells are hidden, so no near-individual rents are exposed); counts are always shown.

The app bundle is assembled and written by :mod:`rentml.mapbundle`.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from rentml import geo
from rentml.config import MIN_MAP_CELL

logger = logging.getLogger(__name__)

LEVELS: tuple[str, ...] = ("canton", "district", "municipality")
LEVEL_KEYS: dict[str, str] = {
    "canton": "canton_id",
    "district": "district_id",
    "municipality": "municipality_id",
}
PLOT_SCALE_M = 100_000.0
LV95_ORIGIN: tuple[float, float] = (2_600_000.0, 1_200_000.0)
PLOT_DECIMALS = 5  # 1e-5 plot degrees = 1 m
DEFAULT_TOLERANCE_M = 25.0
MEDIAN_COLUMNS: tuple[str, ...] = ("chf_per_m2", "chf_per_m2_p25", "chf_per_m2_p75", "rent")
STAT_COLUMNS: tuple[str, ...] = ("n_objects", *MEDIAN_COLUMNS)
UNIT_COLUMNS: tuple[str, ...] = (
    ("level", "unit_id", "name", "canton_id", "canton", "district_id", "district_name")
    + ("lang_region", "population", "area_km2", "density", "n_children")
    + ("minx", "miny", "maxx", "maxy")
)
LAKE_SHARE = 0.5  # a non-municipal area counts as lake if water covers at least half of it
_CANTON_FIELDS = ["kantonsnummer", "name"]
_OTHER_FIELDS = ["objektart", "kantonsnummer", "name", "gem_flaeche", "see_flaeche"]


def load_other_areas(gpkg: Path) -> gpd.GeoDataFrame:
    """Load Swiss territory outside municipalities (cantonal lakes, state forest, Kommunanz).

    Args:
        gpkg: Path to the swissBOUNDARIES3D GeoPackage.

    Returns:
        Columns ``name``, ``canton_id``, ``kind`` (``"lake"`` if water covers at least
        ``LAKE_SHARE`` of the area, else ``"other"``) and 2D geometry in EPSG:2056.

    Raises:
        FileNotFoundError: If ``gpkg`` does not exist.
    """
    if not gpkg.is_file():
        raise FileNotFoundError(f"swissBOUNDARIES3D GeoPackage not found: {gpkg}")
    raw = gpd.read_file(gpkg, layer=geo.MUNICIPALITY_LAYER, columns=_OTHER_FIELDS)
    keep = raw["objektart"].ne(geo.MUNICIPALITY_OBJECT_TYPE) & raw["kantonsnummer"].notna()
    raw = raw.loc[keep]
    lake_share = raw["see_flaeche"].fillna(0.0) / raw["gem_flaeche"].where(raw["gem_flaeche"] > 0)
    frame = gpd.GeoDataFrame(
        {
            "name": raw["name"].astype(str).to_numpy(),
            "canton_id": raw["kantonsnummer"].astype(int).to_numpy(),
            "kind": np.where(lake_share.fillna(0.0) >= LAKE_SHARE, "lake", "other"),
        },
        geometry=raw.geometry.force_2d().to_numpy(),
        crs=raw.crs,
    )
    return frame.to_crs(geo.LV95_EPSG) if frame.crs.to_epsg() != geo.LV95_EPSG else frame


def load_canton_names(gpkg: Path) -> dict[int, str]:
    """Official canton names from swissBOUNDARIES3D, keyed by BFS canton number."""
    table = gpd.read_file(
        gpkg, layer=geo.CANTON_LAYER, columns=_CANTON_FIELDS, ignore_geometry=True
    )
    return {int(k): str(v) for k, v in zip(table["kantonsnummer"], table["name"], strict=True)}


def build_levels(
    units: gpd.GeoDataFrame,
    other: gpd.GeoDataFrame | None = None,
    *,
    canton_names: dict[int, str] | None = None,
    tolerance_m: float = DEFAULT_TOLERANCE_M,
) -> tuple[dict[str, gpd.GeoDataFrame], gpd.GeoDataFrame]:
    """Simplify the boundary coverage and dissolve it into the three admin levels.

    Args:
        units: Municipalities from :func:`rentml.geo.load_admin_units` (EPSG:2056).
        other: Non-municipal areas from :func:`load_other_areas`; they belong to their canton.
        canton_names: Full canton names; ``None`` = the abbreviation.
        tolerance_m: ``coverage_simplify`` tolerance in metres (0 = exact source vertices).

    Returns:
        ``(levels, other)``: one GeoDataFrame per level (columns ``UNIT_COLUMNS`` without
        ``level``, plus geometry) and the simplified ``other`` areas.

    Raises:
        ValueError: If ``tolerance_m`` is negative or the input is not a valid coverage.
    """
    if tolerance_m < 0:
        raise ValueError(f"tolerance_m must be >= 0, got {tolerance_m}")
    other = (
        other
        if other is not None
        else gpd.GeoDataFrame({"name": [], "canton_id": [], "kind": []}, geometry=[], crs=units.crs)
    )
    geoms = np.concatenate([units.geometry.to_numpy(), other.geometry.to_numpy()])
    if not shapely.coverage_is_valid(geoms):
        raise ValueError("Boundaries overlap or have gaps along shared edges (invalid coverage)")
    if tolerance_m > 0:
        geoms = shapely.coverage_simplify(geoms, tolerance_m)
    n = len(units)
    munis = _municipality_frame(units, geoms[:n])
    other = other.set_geometry(gpd.GeoSeries(geoms[n:], index=other.index, crs=units.crs))
    districts = _dissolve(munis, "district_id", ["district_name", "canton_id", "canton"])
    districts = districts.rename(columns={"district_name": "name"})
    districts["district_id"], districts["district_name"] = districts["unit_id"], districts["name"]
    cantons = _dissolve(munis, "canton_id", ["canton"], extra=other)
    names = canton_names or {}
    cantons["name"] = [
        names.get(int(c), a) for c, a in zip(cantons["unit_id"], cantons["canton"], strict=True)
    ]
    cantons["canton_id"] = cantons["unit_id"]
    cantons["n_children"] = cantons["unit_id"].map(districts.groupby("canton_id").size())
    districts["n_children"] = districts["unit_id"].map(munis.groupby("district_id").size())
    levels = {"canton": cantons, "district": districts, "municipality": munis}
    for level, frame in levels.items():
        levels[level] = _finish_level(frame)
    return levels, other


def _municipality_frame(units: gpd.GeoDataFrame, geoms: np.ndarray) -> gpd.GeoDataFrame:
    frame = pd.DataFrame(
        {
            "unit_id": units["municipality_id"].astype(int).to_numpy(),
            "name": units["municipality_name"].astype(str).to_numpy(),
            "canton_id": units["canton_id"].astype(int).to_numpy(),
            "canton": units["canton"].astype(str).to_numpy(),
            "district_id": units["district_id"].astype(int).to_numpy(),
            "district_name": units["district_name"].astype(str).to_numpy(),
            "lang_region": units["lang_region"].to_numpy(),
            "population": units["muni_population"].astype(float).to_numpy(),
            "area_km2": units["muni_area_km2"].astype(float).to_numpy(),
            "n_children": 0,
        }
    )
    return gpd.GeoDataFrame(frame, geometry=geoms, crs=units.crs)


def _dissolve(
    munis: gpd.GeoDataFrame, key: str, first_cols: list[str], *, extra: pd.DataFrame | None = None
) -> gpd.GeoDataFrame:
    """Dissolve municipalities (plus ``extra`` areas with the same key) into parent units."""
    parts = munis[[key, "geometry"]]
    if extra is not None and not extra.empty:
        parts = pd.concat([parts, extra[[key, "geometry"]]], ignore_index=True)
    geometry = parts.groupby(key)["geometry"].agg(lambda g: shapely.coverage_union_all(g.values))
    grouped = munis.groupby(key)
    table = grouped[first_cols].first().join(grouped[["population", "area_km2"]].sum(min_count=1))
    weights = munis.groupby([key, "lang_region"])["population"].sum()
    table["lang_region"] = weights.groupby(level=0).idxmax().map(lambda k: k[1])
    table = table.reset_index().rename(columns={key: "unit_id"})
    return gpd.GeoDataFrame(
        table, geometry=table["unit_id"].map(geometry).to_numpy(), crs=munis.crs
    )


def _finish_level(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = frame.copy()
    for col in ("canton_id", "district_id", "district_name"):
        if col not in out.columns:
            out[col] = pd.NA
    out["density"] = out["population"] / out["area_km2"].where(out["area_km2"] > 0)
    bounds = out.geometry.bounds
    out[["minx", "miny", "maxx", "maxy"]] = bounds[["minx", "miny", "maxx", "maxy"]].to_numpy()
    out["canton_id"] = out["canton_id"].astype("Int64")
    out["district_id"] = out["district_id"].astype("Int64")
    columns = [c for c in UNIT_COLUMNS if c != "level"]
    return out[[*columns, "geometry"]].sort_values("unit_id").reset_index(drop=True)


def aggregate_listings(
    listings: pd.DataFrame, *, min_count: int = MIN_MAP_CELL, extra_cols: Sequence[str] = ()
) -> dict[str, pd.DataFrame]:
    """Count objects and compute CHF/m² and rent medians per canton, district and municipality.

    Args:
        listings: One row per object with ``price``, ``area`` and the admin id columns.
        min_count: Units with fewer objects keep their count but get NaN for every median.
        extra_cols: Further numeric per-object columns (e.g. ``"qoli"``, ``"value_score"``),
            aggregated by the median under the same rule.

    Returns:
        Per level a frame indexed by ``unit_id`` with ``STAT_COLUMNS`` and ``extra_cols``.

    Raises:
        KeyError: If a required column is missing.
        ValueError: If ``min_count`` < 1.
    """
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    required = [*LEVEL_KEYS.values(), "price", "area", *extra_cols]
    if missing := [c for c in required if c not in listings.columns]:
        raise KeyError(f"aggregate_listings needs the columns {missing}")
    df = listings.assign(
        _ppsqm=pd.to_numeric(listings["price"], errors="coerce")
        / pd.to_numeric(listings["area"], errors="coerce").where(listings["area"] > 0)
    )
    out: dict[str, pd.DataFrame] = {}
    for level, key in LEVEL_KEYS.items():
        part = df.loc[df[key].notna()]
        grouped = part.groupby(part[key].astype(int))
        stats = pd.DataFrame({"n_objects": grouped.size()})
        stats["chf_per_m2"] = grouped["_ppsqm"].median()
        stats["chf_per_m2_p25"] = grouped["_ppsqm"].quantile(0.25)
        stats["chf_per_m2_p75"] = grouped["_ppsqm"].quantile(0.75)
        stats["rent"] = grouped["price"].median()
        for col in extra_cols:
            stats[col] = grouped[col].median()
        hidden = stats["n_objects"] < min_count
        stats.loc[hidden, [*MEDIAN_COLUMNS, *extra_cols]] = np.nan
        out[level] = stats.rename_axis("unit_id")
    return out


def to_plot_xy(east: np.ndarray, north: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map LV95 metres to plot degrees (see the module docstring)."""
    x = (np.asarray(east, dtype=float) - LV95_ORIGIN[0]) / PLOT_SCALE_M
    y = (np.asarray(north, dtype=float) - LV95_ORIGIN[1]) / PLOT_SCALE_M
    return x, y


def to_plot_geojson(
    gdf: gpd.GeoDataFrame, *, id_col: str | None = "unit_id", props: Sequence[str] = ()
) -> dict:
    """Convert (multi)polygons to a GeoJSON FeatureCollection in plot degrees.

    Exterior rings are clockwise and holes counter-clockwise, the orientation d3-geo (and so
    Plotly) expects on the sphere; RFC 7946 order would fill the whole globe instead.

    Args:
        gdf: Polygons in EPSG:2056.
        id_col: Column used as feature ``id`` (as string); ``None`` = no id.
        props: Columns copied into the feature properties.

    Returns:
        A GeoJSON FeatureCollection with coordinates rounded to 1 m.

    Raises:
        ValueError: If a geometry is not a (multi)polygon.
    """
    geoms = shapely.orient_polygons(gdf.geometry.to_numpy(), exterior_cw=True)
    features = []
    for i, geom in enumerate(geoms):
        feature: dict[str, object] = {"type": "Feature", "geometry": _geometry_json(geom)}
        if id_col is not None:
            feature["id"] = str(gdf[id_col].iloc[i])
        feature["properties"] = {p: _json_value(gdf[p].iloc[i]) for p in props}
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def _geometry_json(geom: BaseGeometry) -> dict:
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"Expected a (multi)polygon, got {geom.geom_type}")
    polygons = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    coords = [[_ring_coords(r) for r in (p.exterior, *p.interiors)] for p in polygons]
    return {"type": "MultiPolygon", "coordinates": coords}


def _ring_coords(ring: BaseGeometry) -> list[list[float]]:
    xy = shapely.get_coordinates(ring)
    x, y = to_plot_xy(xy[:, 0], xy[:, 1])
    return np.round(np.column_stack([x, y]), PLOT_DECIMALS).tolist()


def _json_value(value: object) -> object:
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value
