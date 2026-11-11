"""Tests for rentml.geo (synthetic GeoPackage and saved API responses; no network)."""

import io
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
from shapely.geometry import box

from rentml import geo, maps

E0, N0 = 2_600_000.0, 1_200_000.0

# (objektart, bfs, district, canton, name, population, area_ha, lake_ha, x0, x1) in metres
MUNICIPALITIES = [
    ("Gemeindegebiet", 700, 241.0, 2.0, "Moutier", 5000, 100.0, 0.0, 0, 1000),
    ("Gemeindegebiet", 351, 246.0, 2.0, "Bern", 10000, 200.0, 100.0, 1000, 2000),
    ("Kantonsgebiet", 9073, np.nan, 2.0, "Thunersee", 0, 100.0, 100.0, 2000, 3000),
    ("Gemeindegebiet", 6621, np.nan, 25.0, "Genève", 20000, 100.0, 0.0, 5000, 6000),
    ("Gemeindegebiet", 7001, np.nan, np.nan, "Vaduz", 6000, 100.0, 0.0, 6000, 7000),
]

MADD_XML = """<?xml version="1.0" encoding="utf-8"?>
<maddResponse xmlns="http://www.ech.ch/xmlns/eCH-0206/2"><status><code>100</code></status>
<buildingList><buildingItem><EGID>1230343</EGID>
<building><dateOfConstruction><dateOfConstruction>1974-01</dateOfConstruction>
<periodOfConstruction>8015</periodOfConstruction></dateOfConstruction>
<surfaceAreaOfBuilding>69</surfaceAreaOfBuilding><numberOfFloors>5</numberOfFloors></building>
<buildingEntranceList><buildingEntranceItem><dwellingList>
<dwellingItem><EWID>3</EWID><dwelling><administrativeDwellingNo>201</administrativeDwellingNo>
<yearOfConstruction>1999</yearOfConstruction><noOfHabitableRooms>2</noOfHabitableRooms>
<floor>3102</floor><surfaceAreaOfDwelling>44</surfaceAreaOfDwelling>
<dwellingStatus>3004</dwellingStatus></dwelling></dwellingItem>
<dwellingItem><EWID>2</EWID><dwelling><administrativeDwellingNo>101</administrativeDwellingNo>
<noOfHabitableRooms>2</noOfHabitableRooms><floor>3101</floor>
<surfaceAreaOfDwelling>46</surfaceAreaOfDwelling><dwellingStatus>3004</dwellingStatus>
</dwelling></dwellingItem>
<dwellingItem><EWID>7</EWID><dwelling><floor>3401</floor><noOfHabitableRooms>1</noOfHabitableRooms>
<dwellingStatus>3004</dwellingStatus></dwelling></dwellingItem>
<dwellingItem><EWID>9</EWID><dwelling><floor>3100</floor>
<dwellingStatus>3007</dwellingStatus></dwelling></dwellingItem>
</dwellingList></buildingEntranceItem></buildingEntranceList>
</buildingItem></buildingList></maddResponse>"""

IDENTIFY_JSON = {
    "results": [
        {"layerBodId": geo.OEV_LAYER, "attributes": {"oev_erreichb_ewap": 37865}},
        {"layerBodId": geo.SOLAR_LAYER, "attributes": {"klasse": 4}},
        {"layerBodId": geo.POPULATION_LAYER, "attributes": {"number": 119, "i_year": 1990}},
        {"layerBodId": geo.POPULATION_LAYER, "attributes": {"number": 130, "i_year": 2024}},
    ]
}


def _box(x0: float, x1: float) -> box:
    return box(E0 + x0, N0, E0 + x1, N0 + 1000)


@pytest.fixture
def gpkg(tmp_path: Path) -> Path:
    path = tmp_path / geo.SB3D_FILE
    cols = ["objektart", "bfs_nummer", "bezirksnummer", "kantonsnummer", "name"]
    cols += ["einwohnerzahl", "gem_flaeche", "see_flaeche"]
    rows = [r[:8] for r in MUNICIPALITIES]
    geoms = [_box(r[8], r[9]) for r in MUNICIPALITIES]
    munis = gpd.GeoDataFrame(rows, columns=cols, geometry=geoms, crs=geo.LV95)
    munis.to_file(path, layer=geo.MUNICIPALITY_LAYER)
    districts = {"bezirksnummer": [241, 246], "name": ["Jura bernois", "Bern-Mittelland"]}
    gpd.GeoDataFrame(districts, geometry=[_box(0, 1000), _box(1000, 3000)], crs=geo.LV95).to_file(
        path, layer="tlm_bezirksgebiet"
    )
    cantons = {"kantonsnummer": [2, 25], "name": ["Bern", "Genève"]}
    gpd.GeoDataFrame(cantons, geometry=[_box(0, 3000), _box(5000, 6000)], crs=geo.LV95).to_file(
        path, layer="tlm_kantonsgebiet"
    )
    return path


class FakeResponse:
    def __init__(self, payload: dict | str) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload

    @property
    def content(self) -> bytes:
        return self.payload.encode()


class FakeSession:
    """Routes requests by URL prefix to saved payloads; records the calls."""

    def __init__(self, routes: dict[str, dict | str]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, params: dict, timeout: float) -> FakeResponse:
        self.calls.append(url)
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                return FakeResponse(payload)
        raise requests.ConnectionError(url)


def test_load_admin_units(gpkg: Path) -> None:
    units = geo.load_admin_units(gpkg).set_index("municipality_name")
    assert list(units.index) == ["Bern", "Moutier", "Genève"]  # sorted by BFS number
    assert units.loc["Genève", "district_id"] == 2500  # canton without districts
    assert units.loc["Genève", "district_name"] == "Genève"
    assert units.loc["Bern", "muni_area_km2"] == 1.0  # lake excluded
    assert units.loc["Bern", "muni_density"] == 10000.0
    assert units["lang_region"].to_dict() == {"Bern": "de", "Moutier": "fr", "Genève": "fr"}
    assert units.loc["Bern", "canton_name"] == "Bern"
    assert not units.geometry.has_z.any()


def test_assign_admin_units(gpkg: Path) -> None:
    units, mask = geo.load_admin_units(gpkg), geo.load_foreign_mask(gpkg)
    points = pd.DataFrame(
        {
            "east": [E0 + 1500, E0 + 2500, E0 + 6500, np.nan, E0 + 20000],
            "north": [N0 + 500] * 5,
            "admin_match": ["stale"] * 5,
        },
        index=[7, 7, 3, 1, 0],  # duplicate labels: assignment must be positional
    )
    out = geo.assign_admin_units(points, units, exclude=mask)
    assert out["admin_match"].tolist() == ["within", "nearest", "none", "none", "none"]
    assert out["municipality_id"].tolist()[:2] == [351, 351]
    assert out["municipality_id"].dtype == "Int64"
    unmasked = geo.assign_admin_units(points, units)
    assert unmasked["municipality_name"].iloc[2] == "Genève"  # Vaduz point snaps without mask


def test_assign_admin_units_accepts_wgs84_map_layer(gpkg: Path) -> None:
    layer = geo.load_admin_units(gpkg).to_crs(geo.WGS84)
    out = geo.assign_admin_units(pd.DataFrame({"east": [E0 + 500], "north": [N0 + 500]}), layer)
    assert out["municipality_name"].iloc[0] == "Moutier"


def test_nested_group_keys() -> None:
    df = pd.DataFrame(
        {"canton": ["ZH", None], "district_id": [112, np.nan], "municipality_id": [261, np.nan]}
    )
    keys = geo.nested_group_keys(df)
    assert keys.iloc[0].tolist() == ["ZH", "ZH/112", "ZH/112/261"]
    assert keys.iloc[1].isna().all()


def test_count_bucket() -> None:
    buckets = geo.count_bucket(pd.Series([0, 4, 5, 20, 21, np.nan]))
    assert buckets.tolist() == ["<5", "<5", "5-20", "5-20", ">20", "<5"]
    assert list(buckets.cat.categories) == list(geo.COUNT_BUCKET_ORDER)


def test_download_swissboundaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("data/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg", b"gpkg-bytes")

    class Streamed(FakeResponse):
        def __enter__(self) -> "Streamed":
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [payload.getvalue()]

    monkeypatch.setattr(requests, "get", lambda url, **kwargs: Streamed(url))
    path = geo.download_swissboundaries(tmp_path, "2026-01")
    assert path.read_bytes() == b"gpkg-bytes"
    assert [p.name for p in tmp_path.iterdir()] == [geo.SB3D_FILE]  # archive removed

    monkeypatch.setattr(requests, "get", lambda url, **kwargs: pytest.fail("no download"))
    assert geo.download_swissboundaries(tmp_path, "2026-01") == path


def test_search_addresses() -> None:
    results = [
        {"attrs": {"label": "Herrengasse 10 <b>3011 Bern</b>", "featureId": "1230343_0",
                   "x": 1199557.125, "y": 2600872.0}},
        {"attrs": {"label": "Herrengasse 10 <b>3011 Bern</b>", "featureId": "1230343_0",
                   "x": 1199557.125, "y": 2600872.0}},
        {"attrs": {"label": "Seeweg 1 <b>6000 Luzern</b>", "x": 1211000, "y": 2666000}},
    ]  # fmt: skip
    session = FakeSession({geo.SEARCH_URL: {"results": results}})
    found = geo.search_addresses("Herrengasse 10", session=session)
    assert found == [
        geo.Address("Herrengasse 10 3011 Bern", 2600872.0, 1199557.125, 1230343),
        geo.Address("Seeweg 1 6000 Luzern", 2666000.0, 1211000.0, None),
    ]
    assert geo.search_addresses("He", session=session) == []
    assert len(session.calls) == 1


def test_building_dwellings() -> None:
    dwellings = geo.building_dwellings(1230343, session=FakeSession({geo.MADD_URL: MADD_XML}))
    assert [d.ewid for d in dwellings] == [7, 2, 3]  # by floor; demolished EWID 9 dropped
    assert dwellings[0].floor_label == "1. UG"
    assert dwellings[1].display == "EWID 2 · 101 · 1. Stock · 2 Zi · 46 m²"
    assert dwellings[0].display == "EWID 7 · 1. UG · 1 Zi"
    assert geo.building_dwellings(1, session=FakeSession({})) == []  # network error


def test_lookup_location() -> None:
    session = FakeSession(
        {
            geo.MADD_URL: MADD_XML,
            geo.IDENTIFY_URL: IDENTIFY_JSON,
            geo.HEIGHT_URL: {"height": "537.6"},
        }
    )
    address = geo.Address("Herrengasse 10 3011 Bern", 2600872.0, 1199557.0, 1230343)
    loc = geo.lookup_location(address, session=session)
    assert loc.features() == {
        "year_built": 1974.0,
        "apartments": 3.0,
        "land_area": 69.0,
        "oev": 37865.0,
        "solar": 4.0,
        "population": 130.0,  # latest census year
        "elevation": 537.6,
    }
    assert loc.missing == []
    assert len(loc.dwellings) == 3
    period_only = MADD_XML.replace("<dateOfConstruction>1974-01</dateOfConstruction>", "")
    session.routes[geo.MADD_URL] = period_only
    assert geo.lookup_location(address, session=session).year_built == 1976.0  # 8015 midpoint
    session.routes[geo.MADD_URL] = period_only.replace(
        "<periodOfConstruction>8015</periodOfConstruction>", ""
    )
    assert geo.lookup_location(address, session=session).year_built is None  # not 1999


def test_lookup_location_without_egid_or_network() -> None:
    loc = geo.lookup_location(geo.Address("Feld", 2600000.0, 1200000.0), session=FakeSession({}))
    assert set(loc.features().values()) == {None}
    assert loc.missing == list(loc.features())
    assert loc.dwellings == []


def test_write_and_load_map_layers(gpkg: Path, tmp_path: Path) -> None:
    units = geo.load_admin_units(gpkg)
    objects = pd.DataFrame(
        {
            "municipality_id": [351, 351, 351, 700],
            "district_id": [246, 246, 246, 241],
            "canton_id": [2, 2, 2, 2],
            "price": [1000.0, 2000.0, 3000.0, 9000.0],
            "area": [50.0, 100.0, 100.0, 90.0],
        }
    )
    maps.write_map_layers(objects, units, out_dir=tmp_path / "app", min_count=3)
    layers, meta = maps.load_map_layers(tmp_path / "app")
    assert meta["min_count"] == 3 and meta["n_objects"] == 4
    munis = layers["municipality"].set_index("unit_id")
    assert munis.crs.to_epsg() == geo.WGS84
    assert munis.loc[351, "chf_per_m2"] == 20.0 and munis.loc[351, "rent"] == 2000.0
    assert munis.loc[700, "n_objects"] == 1 and np.isnan(munis.loc[700, "chf_per_m2"])
    assert munis.loc[6621, "n_objects"] == 0
    cantons = layers["canton"].set_index("unit_id")
    assert cantons.loc[2, "name"] == "Bern" and cantons.loc[2, "n_objects"] == 4
    assert cantons.loc[2, "lang_region"] == "de"  # Bern outweighs French-speaking Moutier
    assert cantons.loc[2, "population"] == 15000.0
    assert layers["district"].set_index("unit_id").loc[241, "name"] == "Jura bernois"
