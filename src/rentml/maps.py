"""Map layers of the RentLens app: admin units with listing aggregates and the lakes.

The unit layers are simplified swissBOUNDARIES3D polygons in WGS84 with median CHF/m² and rent.
swissBOUNDARIES3D assigns most lakes to the surrounding municipalities (Lake Lucerne belongs to
Lucerne, Weggis, Vitznau, ...), so a choropleth would paint them as land. The lakes therefore
come as a separate layer from the BFS generalised boundaries (lakes of about 5 km² and more),
drawn on top of the units. The unit polygons stay whole, because the app also uses them to find
the municipality of an address.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from rentml.config import APP_DATA, EXTERNAL, MIN_MAP_CELL, SB3D_VERSION, download
from rentml.geo import LV95, WGS84

logger = logging.getLogger(__name__)

MAP_LEVELS = {"canton": "canton_id", "district": "district_id", "municipality": "municipality_id"}
LAKES_VERSION = "2026-01-01"  # BFS state matching swissBOUNDARIES3D 2026-01
LAKES_URL = (
    "https://data.geo.admin.ch/ch.bfs.historisierte-administrative_grenzen_g1/"
    "historisierte-administrative_grenzen_g1_{v}/historisierte-administrative_grenzen_g1_{v}"
    "_see_2056.geojson"
)
LAKES_FILE = "map_lakes.parquet"
LAKE_COLOR = "#B5D0E6"  # water
LAKE_EDGE = "#6E9AB8"  # outline, so lakes do not read as a unit with a low value
LAKE_ZORDER = 2.5  # above polygons and border lines; point layers need a higher zorder


def download_lakes(dest_dir: Path = EXTERNAL, version: str = LAKES_VERSION) -> Path:
    """Download the BFS lake polygons (generalised boundaries G1, OGD) once.

    Args:
        dest_dir: Target folder.
        version: Reference date of the BFS boundaries (``YYYY-MM-DD``).

    Returns:
        Path of the GeoJSON in LV95.
    """
    target = dest_dir / f"bfs_lakes_g1_{version}.geojson"
    return download(LAKES_URL.format(v=version), target)


def load_lakes(path: Path) -> gpd.GeoDataFrame:
    """Read the BFS lakes as ``name`` and ``area_km2`` in LV95.

    Args:
        path: GeoJSON from :func:`download_lakes`.

    Returns:
        One row per lake.
    """
    lakes = gpd.read_file(path).to_crs(LV95)
    return gpd.GeoDataFrame(
        {"name": lakes["SEENAME"], "area_km2": lakes.geometry.area / 1e6},
        geometry=lakes.geometry,
        crs=LV95,
    )


def write_lakes_layer(
    lakes: gpd.GeoDataFrame, out_dir: Path = APP_DATA, tolerance_m: float = 50.0
) -> Path:
    """Write the lakes as simplified WGS84 GeoParquet next to the unit layers.

    Args:
        lakes: Output of :func:`load_lakes`.
        out_dir: Folder of the map layers.
        tolerance_m: Simplification in metres.

    Returns:
        Path of :data:`LAKES_FILE`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    layer = lakes.set_geometry(lakes.geometry.simplify(tolerance_m)).to_crs(WGS84)
    layer = layer.assign(unit_id=[f"lake_{i}" for i in range(len(layer))])
    layer = layer.set_geometry(layer.geometry.set_precision(1e-5))
    path = out_dir / LAKES_FILE
    layer.to_parquet(path)
    return path


def plot_lakes(ax: Axes, lakes: gpd.GeoDataFrame | None) -> None:
    """Draw the lakes on a matplotlib map in the CRS of ``lakes``.

    The BFS polygons cover whole lakes, so Lake Geneva, Lake Constance and Lago Maggiore extend
    beyond the Swiss border. Point layers on the same axes need ``zorder`` above
    :data:`LAKE_ZORDER`.

    Args:
        ax: Target axes.
        lakes: Output of :func:`load_lakes`; ``None`` or empty draws nothing.
    """
    if lakes is None or lakes.empty:
        return
    lakes.plot(ax=ax, color=LAKE_COLOR, edgecolor=LAKE_EDGE, linewidth=0.4, zorder=LAKE_ZORDER)


def write_map_layers(
    objects: pd.DataFrame,
    units: gpd.GeoDataFrame,
    out_dir: Path = APP_DATA,
    tolerance_m: float = 50.0,
    min_count: int = MIN_MAP_CELL,
    lakes: gpd.GeoDataFrame | None = None,
) -> list[Path]:
    """Write ``map_<level>.parquet`` (WGS84 GeoParquet) and ``map_meta.json`` for the map page.

    Each layer has ``unit_id``, ``name``, ``canton``, ``lang_region``, ``population``,
    ``area_km2``, ``density``, ``n_objects``, ``chf_per_m2`` and ``rent``; medians only for
    units with at least ``min_count`` objects. The municipality layer keeps the
    ``UNIT_COLUMNS``, so :func:`rentml.geo.assign_admin_units` also works on it.

    Args:
        objects: One row per object with ``price``, ``area`` and the admin id columns.
        units: Output of :func:`rentml.geo.load_admin_units`.
        out_dir: Folder of the map layers.
        tolerance_m: Coverage simplification in metres (neighbours keep shared borders).
        min_count: Objects a unit needs for published medians.
        lakes: Output of :func:`load_lakes`; written as :data:`LAKES_FILE` if given.

    Returns:
        Paths of the written files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    munis = units.set_geometry(units.geometry.simplify_coverage(tolerance_m))
    munis = munis.assign(population=munis["muni_population"], area_km2=munis["muni_area_km2"])
    objects = objects.assign(chf_per_m2=objects["price"] / objects["area"])
    paths = []
    for level, key in MAP_LEVELS.items():
        layer = munis if level == "municipality" else _dissolve(munis, key, f"{level}_name")
        stats = objects.groupby(key).agg(
            n_objects=("price", "size"),
            chf_per_m2=("chf_per_m2", "median"),
            rent=("price", "median"),
        )
        stats.loc[stats["n_objects"] < min_count, ["chf_per_m2", "rent"]] = np.nan
        layer = layer.assign(unit_id=layer[key], name=layer[f"{level}_name"])
        layer = layer.join(stats, on="unit_id")
        layer["n_objects"] = layer["n_objects"].fillna(0).astype(int)
        layer["density"] = layer["population"] / layer["area_km2"]
        layer = layer.to_crs(WGS84)
        layer = layer.set_geometry(layer.geometry.set_precision(1e-5))  # ~1 m, smaller JSON
        paths.append(out_dir / f"map_{level}.parquet")
        layer.to_parquet(paths[-1])
    meta = {
        "boundaries": f"swissBOUNDARIES3D {SB3D_VERSION}",
        "min_count": min_count,
        "n_objects": len(objects),
        "chf_per_m2": float(objects["chf_per_m2"].median()),
        "rent": float(objects["price"].median()),
    }
    if lakes is not None:
        paths.append(write_lakes_layer(lakes, out_dir, tolerance_m))
        meta["lakes"] = f"BFS generalised boundaries G1 {LAKES_VERSION}"
    paths.append(out_dir / "map_meta.json")
    paths[-1].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return paths


def _dissolve(munis: gpd.GeoDataFrame, key: str, name_col: str) -> gpd.GeoDataFrame:
    first = [c for c in ("canton_id", "canton", "canton_name", name_col) if c != key]
    aggfunc = dict.fromkeys(first, "first") | {"population": "sum", "area_km2": "sum"}
    out = munis.dissolve(key, aggfunc=aggfunc, method="coverage").reset_index()
    weights = munis.groupby([key, "lang_region"])["population"].sum()
    out["lang_region"] = out[key].map(weights.groupby(level=0).idxmax().str[1])
    return out


def load_map_layers(map_dir: Path = APP_DATA) -> tuple[dict[str, gpd.GeoDataFrame], dict]:
    """Read the layers (keyed by level) and the metadata written by :func:`write_map_layers`."""
    layers = {level: gpd.read_parquet(map_dir / f"map_{level}.parquet") for level in MAP_LEVELS}
    meta = json.loads((map_dir / "map_meta.json").read_text(encoding="utf-8"))
    return layers, meta


def load_lakes_layer(map_dir: Path = APP_DATA) -> gpd.GeoDataFrame | None:
    """Read :data:`LAKES_FILE` from ``map_dir``; ``None`` if it was never written."""
    path = map_dir / LAKES_FILE
    if not path.is_file():
        logger.info("No lake layer in %s: lakes are drawn as land", map_dir)
        return None
    return gpd.read_parquet(path)
