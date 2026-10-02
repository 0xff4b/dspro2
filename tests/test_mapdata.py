"""Tests for rentml.mapdata (synthetic boundaries from conftest; no network)."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

from rentml import mapdata
from rentml.mapdata import aggregate_listings, build_levels, to_plot_geojson, to_plot_xy


def test_build_levels_dissolves_coverage(map_units, map_other) -> None:
    levels, other = build_levels(map_units, map_other, canton_names={1: "Zürich"}, tolerance_m=10.0)
    cantons, districts, munis = levels["canton"], levels["district"], levels["municipality"]
    assert list(munis["unit_id"]) == [1, 2, 3, 4]
    assert list(districts["unit_id"]) == [101, 102, 2500]
    zh = cantons.set_index("unit_id").loc[1]
    assert zh["name"] == "Zürich" and zh["n_children"] == 2
    assert cantons.set_index("unit_id").loc[25, "name"] == "GE"  # no name given -> abbreviation
    # The canton includes its lake, the population and land area only the municipalities.
    assert zh.geometry.area == pytest.approx(3e6 + 1.5e6)
    assert zh["population"] == 4500 and zh["area_km2"] == 3.0
    nord = districts.set_index("unit_id").loc[101]
    assert nord["lang_region"] == "fr"  # population-weighted majority (Bex 3000 > Aach 1000)
    assert nord["n_children"] == 2 and nord["district_name"] == "Nord"
    assert len(other) == 1


def test_build_levels_keeps_shared_borders(map_units) -> None:
    levels, _ = build_levels(map_units, tolerance_m=50.0)
    assert shapely.coverage_is_valid(levels["municipality"].geometry.to_numpy())
    assert levels["municipality"][["minx", "miny", "maxx", "maxy"]].notna().all().all()


def test_build_levels_rejects_invalid_input(map_units) -> None:
    with pytest.raises(ValueError, match="tolerance_m"):
        build_levels(map_units, tolerance_m=-1.0)
    overlapping = map_units.copy()
    overlapping.loc[1, "geometry"] = box(2_600_500, 1_200_000, 2_602_000, 1_201_000)
    with pytest.raises(ValueError, match="coverage"):
        build_levels(overlapping)


def test_aggregate_listings_hides_small_cells(map_listings) -> None:
    stats = aggregate_listings(map_listings, min_count=20)
    munis = stats["municipality"]
    assert munis.loc[1, "n_objects"] == 25
    assert munis.loc[1, "chf_per_m2"] == pytest.approx(20.0)
    assert munis.loc[1, "chf_per_m2_p25"] <= 20.0 <= munis.loc[1, "chf_per_m2_p75"]
    assert munis.loc[3, "n_objects"] == 5 and np.isnan(munis.loc[3, "chf_per_m2"])
    assert np.isnan(munis.loc[3, "rent"])
    canton = stats["canton"].loc[1]
    assert canton["n_objects"] == 30 and canton["chf_per_m2"] == pytest.approx(20.0)


def test_aggregate_listings_extra_columns_and_errors(map_listings) -> None:
    listings = map_listings.assign(qoli=50.0)
    stats = aggregate_listings(listings, min_count=1, extra_cols=["qoli"])
    assert stats["district"].loc[102, "qoli"] == 50.0
    with pytest.raises(KeyError, match="price"):
        aggregate_listings(map_listings.drop(columns="price"))
    with pytest.raises(ValueError, match="min_count"):
        aggregate_listings(map_listings, min_count=0)


def test_to_plot_xy_is_affine_around_bern() -> None:
    x, y = to_plot_xy(np.array([2_600_000.0, 2_700_000.0]), np.array([1_200_000.0, 1_150_000.0]))
    assert x.tolist() == [0.0, 1.0] and y.tolist() == [0.0, -0.5]


def test_to_plot_geojson_orients_rings_for_d3() -> None:
    outer = box(2_600_000, 1_200_000, 2_604_000, 1_204_000)
    hole = box(2_601_000, 1_201_000, 2_602_000, 1_202_000)
    gdf = gpd.GeoDataFrame(
        {"unit_id": [7], "name": ["X"]}, geometry=[outer.difference(hole)], crs="EPSG:2056"
    )
    collection = to_plot_geojson(gdf, props=["name"])
    feature = collection["features"][0]
    assert feature["id"] == "7" and feature["properties"] == {"name": "X"}
    exterior, interior = feature["geometry"]["coordinates"][0]
    assert shapely.Polygon(exterior).exterior.is_ccw is False  # clockwise exterior
    assert shapely.Polygon(interior).exterior.is_ccw is True
    assert max(c[0] for c in exterior) == pytest.approx(0.04)
    assert mapdata.PLOT_DECIMALS == 5


def test_to_plot_geojson_rejects_lines() -> None:
    gdf = gpd.GeoDataFrame({"unit_id": [1]}, geometry=[shapely.LineString([(0, 0), (1, 1)])])
    with pytest.raises(ValueError, match="polygon"):
        to_plot_geojson(gdf)


def test_load_other_areas_classifies_lakes(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "objektart": ["Kantonsgebiet", "Kantonsgebiet", "Gemeindegebiet"],
            "kantonsnummer": [2.0, 10.0, 2.0],
            "name": ["Thunersee", "Staatswald Galm", "Bern"],
            "gem_flaeche": [100.0, 100.0, 100.0],
            "see_flaeche": [100.0, 0.0, 0.0],
        }
    )
    geoms = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    path = tmp_path / "sb3d.gpkg"
    gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:2056").to_file(path, layer="tlm_hoheitsgebiet")
    other = mapdata.load_other_areas(path)
    assert other["name"].tolist() == ["Thunersee", "Staatswald Galm"]
    assert other["kind"].tolist() == ["lake", "other"]
    with pytest.raises(FileNotFoundError):
        mapdata.load_other_areas(tmp_path / "missing.gpkg")
