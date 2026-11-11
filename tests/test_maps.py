"""Tests for rentml.maps (synthetic polygons; no network)."""

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from matplotlib.figure import Figure
from shapely.geometry import box

from rentml import geo, maps

E0, N0 = 2_600_000.0, 1_200_000.0


def _units() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "municipality_id": [1061, 1062], "district_id": [316, 316], "canton_id": [3, 3],
            "municipality_name": ["Luzern", "Weggis"], "district_name": ["Luzern", "Luzern"],
            "canton": ["LU", "LU"], "canton_name": ["Luzern", "Luzern"],
            "muni_population": [80000.0, 4500.0], "muni_area_km2": [29.0, 12.0],
            "lang_region": ["de", "de"],
        },
        geometry=[box(E0, N0, E0 + 5000, N0 + 5000), box(E0 + 5000, N0, E0 + 10000, N0 + 5000)],
        crs=geo.LV95,
    )  # fmt: skip


def _lakes_geojson(tmp_path: Path) -> Path:
    raw = gpd.GeoDataFrame(
        {"SEENR": [9179], "SEENAME": ["Vierwaldstättersee"], "AREA_HA": [400]},
        geometry=[box(E0 + 3000, N0 + 1000, E0 + 7000, N0 + 2000)],
        crs=geo.LV95,
    )
    path = tmp_path / "see.geojson"
    raw.to_file(path, driver="GeoJSON")
    return path


def test_load_lakes_names_and_area(tmp_path: Path) -> None:
    lakes = maps.load_lakes(_lakes_geojson(tmp_path))
    assert lakes.crs.to_epsg() == geo.LV95
    assert lakes["name"].tolist() == ["Vierwaldstättersee"]
    assert round(lakes["area_km2"].iloc[0], 3) == 4.0


def test_write_and_load_lakes_layer(tmp_path: Path) -> None:
    lakes = maps.load_lakes(_lakes_geojson(tmp_path))
    path = maps.write_lakes_layer(lakes, tmp_path / "app")
    layer = maps.load_lakes_layer(tmp_path / "app")
    assert path.name == maps.LAKES_FILE and layer.crs.to_epsg() == geo.WGS84
    assert layer["unit_id"].tolist() == ["lake_0"]


def test_load_lakes_layer_missing_is_none(tmp_path: Path) -> None:
    assert maps.load_lakes_layer(tmp_path) is None


def test_write_map_layers_with_lakes_keeps_units_whole(tmp_path: Path) -> None:
    objects = pd.DataFrame({
        "municipality_id": [1061, 1061], "district_id": [316, 316], "canton_id": [3, 3],
        "price": [2000.0, 2400.0], "area": [80.0, 80.0],
    })  # fmt: skip
    lakes = maps.load_lakes(_lakes_geojson(tmp_path))
    paths = maps.write_map_layers(objects, _units(), tmp_path / "app", min_count=2, lakes=lakes)
    assert tmp_path / "app" / maps.LAKES_FILE in paths
    meta = json.loads((tmp_path / "app" / "map_meta.json").read_text())
    assert meta["lakes"].endswith(maps.LAKES_VERSION)
    layers, _ = maps.load_map_layers(tmp_path / "app")
    munis = layers["municipality"].to_crs(geo.LV95).set_index("unit_id")
    assert round(munis.loc[1061].geometry.area / 1e6) == 25  # the lake is not cut out
    assert munis.loc[1061, "chf_per_m2"] == 27.5


def test_plot_lakes_draws_above_borders(tmp_path: Path) -> None:
    lakes = maps.load_lakes(_lakes_geojson(tmp_path))
    ax = Figure().add_subplot()
    maps.plot_lakes(ax, lakes)
    assert len(ax.collections) == 1 and ax.collections[0].get_zorder() == maps.LAKE_ZORDER


def test_plot_lakes_without_lakes_draws_nothing() -> None:
    ax = Figure().add_subplot()
    maps.plot_lakes(ax, None)
    assert not ax.collections
