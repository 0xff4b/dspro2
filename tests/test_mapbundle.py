"""Tests for rentml.mapbundle (synthetic boundaries and listings; no network)."""

import pandas as pd
import pytest

from rentml.mapbundle import MapBundle, assemble_bundle, prepare_listings
from rentml.mapdata import LEVELS, UNIT_COLUMNS, aggregate_listings, build_levels


def test_assemble_bundle_joins_stats(map_units, map_other, map_listings) -> None:
    levels, other = build_levels(map_units, map_other, tolerance_m=0.0)
    stats = aggregate_listings(map_listings, min_count=20)
    bundle = assemble_bundle(levels, other, stats, meta={"min_count": 20})
    units = bundle.units
    assert list(units.columns[: len(UNIT_COLUMNS)]) == list(UNIT_COLUMNS)
    assert units.groupby("level").size().to_dict() == {
        "canton": 2,
        "district": 3,
        "municipality": 4,
    }
    geneva = units.loc[units["level"].eq("municipality") & units["unit_id"].eq(4)].iloc[0]
    assert geneva["n_objects"] == 0  # units without listings get a count of 0
    assert set(bundle.geojson) == set(LEVELS)
    assert len(bundle.geojson["municipality"]["features"]) == 4
    assert bundle.other_areas["features"][0]["properties"]["kind"] == "lake"


def test_bundle_roundtrip(tmp_path, map_units, map_listings) -> None:
    levels, other = build_levels(map_units)
    bundle = assemble_bundle(levels, other, aggregate_listings(map_listings), meta={"a": 1})
    loaded = MapBundle.load(bundle.save(tmp_path / "map"))
    assert loaded.meta == {"a": 1}
    assert loaded.units.shape == bundle.units.shape
    assert loaded.geojson["canton"] == bundle.geojson["canton"]


def test_bundle_load_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="rentml.mapbundle"):
        MapBundle.load(tmp_path)


def test_prepare_listings_cleans_and_locates(map_units) -> None:
    rows = [
        (2_600_050.0 + 70 * i, 1_200_500.0, 40.0 + 8 * i, 3.0, 900 + 160 * i) for i in range(12)
    ]
    rows += [(2_600_600.0, 1_200_600.0, 5.0, 1.0, 900.0)]  # 5 m²: domain rule
    rows += [(2_700_000.0, 1_300_000.0, 70.0, 3.0, 1500.0)]  # far outside every municipality
    listings = pd.DataFrame(rows, columns=["east", "north", "area", "rooms", "price"])
    listings.index = pd.Index(range(100, 100 + len(rows)), name="listing_id")
    objects = prepare_listings(listings, map_units)
    assert len(objects) == 12
    assert set(objects["municipality_id"]) == {1} and set(objects["canton_id"]) == {1}
