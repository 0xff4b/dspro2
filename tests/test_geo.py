"""Tests for rentml.geo (synthetic GeoPackage; no network)."""

import io
import logging
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import requests
from shapely.geometry import MultiPolygon, Polygon

from rentml import address, geo

E0, N0 = 2_600_000.0, 1_200_000.0
REAL_GPKG = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "external"
    / "swissBOUNDARIES3D_1_5_LV95_LN02.gpkg"
)

GPKG_NAME = "swissBOUNDARIES3D_1_5_LV95_LN02.gpkg"

# (objektart, bfs, district, canton, name, population, area_ha, lake_ha, x0, x1)
MUNICIPALITY_ROWS = [
    ("Gemeindegebiet", 700, 241.0, 2.0, "Moutierville", 5000, 100.0, 0.0, 0, 1000),
    ("Gemeindegebiet", 351, 246.0, 2.0, "Bern", 10000, 200.0, 100.0, 1000, 2000),
    ("Gemeindegebiet", 6621, np.nan, 25.0, "Genève", 20000, 100.0, 0.0, 5000, 6000),
    ("Kantonsgebiet", 9073, np.nan, 2.0, "Thunersee", 0, 100.0, 100.0, 2000, 3000),
    ("Gemeindegebiet", 7001, np.nan, np.nan, "Vaduz", 6000, 100.0, 0.0, 8500, 9500),
]


def _box3d(x0: float, x1: float, y0: float = 0.0, y1: float = 1000.0) -> MultiPolygon:
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return MultiPolygon([Polygon([(E0 + x, N0 + y, 500.0) for x, y in ring])])


def _write_gpkg(path: Path, rows: list[tuple]) -> Path:
    cols = ["objektart", "bfs_nummer", "bezirksnummer", "kantonsnummer", "name", "einwohnerzahl"]
    munis = gpd.GeoDataFrame(
        [r[:8] for r in rows],
        columns=[*cols, "gem_flaeche", "see_flaeche"],
        geometry=[_box3d(r[8], r[9]) for r in rows],
        crs="EPSG:2056",
    )
    districts = gpd.GeoDataFrame(
        {"bezirksnummer": [241, 246], "name": ["Jura bernois", "Bern-Mittelland"]},
        geometry=[_box3d(0, 1000), _box3d(1000, 3000)],
        crs="EPSG:2056",
    )
    cantons = gpd.GeoDataFrame(
        {"kantonsnummer": [2, 25], "name": ["Bern", "Genève"]},
        geometry=[_box3d(0, 3000), _box3d(5000, 6000)],
        crs="EPSG:2056",
    )
    munis.to_file(path, layer=geo.MUNICIPALITY_LAYER, driver="GPKG")
    districts.to_file(path, layer=geo.DISTRICT_LAYER, driver="GPKG")
    cantons.to_file(path, layer=geo.CANTON_LAYER, driver="GPKG")
    return path


@pytest.fixture(scope="module")
def synthetic_gpkg(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_gpkg(tmp_path_factory.mktemp("sb3d") / "sb3d.gpkg", MUNICIPALITY_ROWS)


@pytest.fixture(scope="module")
def units(synthetic_gpkg: Path) -> gpd.GeoDataFrame:
    return geo.load_admin_units(synthetic_gpkg)


def _points(xs: list[float], index: list[int]) -> pd.DataFrame:
    east = [E0 + x if np.isfinite(x) else np.nan for x in xs]
    return pd.DataFrame({"east": east, "north": N0 + 500.0, "price": 2000.0}, index=index)


class _FakeResponse:
    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.fail:
            raise requests.HTTPError("404 Client Error")

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _patch_get(monkeypatch: pytest.MonkeyPatch, payload: bytes, *, fail: bool = False) -> list[str]:
    calls: list[str] = []

    def fake_get(url: str, *, stream: bool, timeout: float) -> _FakeResponse:
        assert stream and timeout > 0
        calls.append(url)
        return _FakeResponse(payload, fail=fail)

    monkeypatch.setattr(geo.requests, "get", fake_get)
    return calls


def test_sb3d_url_follows_swisstopo_pattern() -> None:
    assert geo.sb3d_url("2026-01") == (
        "https://data.geo.admin.ch/ch.swisstopo.swissboundaries3d/swissboundaries3d_2026-01/"
        "swissboundaries3d_2026-01_2056_5728.gpkg.zip"
    )


@pytest.mark.parametrize("version", ["2026/01", "latest", "26-01", ""])
def test_sb3d_url_rejects_malformed_version(version: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM"):
        geo.sb3d_url(version)


def test_download_swissboundaries_extracts_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _zip_bytes({"data/swissBOUNDARIES3D_1_5_LV95_LN02.gpkg": b"gpkg-bytes"})
    calls = _patch_get(monkeypatch, payload)
    gpkg = geo.download_swissboundaries(tmp_path / "ext", "2026-01")
    assert gpkg == tmp_path / "ext" / "swissBOUNDARIES3D_1_5_LV95_LN02.gpkg"
    assert gpkg.read_bytes() == b"gpkg-bytes"
    assert calls == [geo.sb3d_url("2026-01")]
    assert geo.download_swissboundaries(tmp_path / "ext", "2026-01") == gpkg
    assert len(calls) == 1  # cached, no second request
    geo.download_swissboundaries(tmp_path / "ext", "2027-01")
    assert calls[-1] == geo.sb3d_url("2027-01")  # other release -> fresh download
    assert not list((tmp_path / "ext").glob("*.part"))


def test_download_swissboundaries_http_error_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get(monkeypatch, b"", fail=True)
    with pytest.raises(RuntimeError, match="failed"):
        geo.download_swissboundaries(tmp_path, "2026-01")
    assert list(tmp_path.iterdir()) == []


def test_download_swissboundaries_rejects_archive_without_gpkg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get(monkeypatch, _zip_bytes({"readme.txt": b"no data"}))
    with pytest.raises(FileNotFoundError, match="gpkg"):
        geo.download_swissboundaries(tmp_path, "2026-01")


@pytest.mark.parametrize("corrupt", ["member_crc", "html_body"])
def test_download_swissboundaries_bad_archive_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corrupt: str
) -> None:
    good = _zip_bytes({GPKG_NAME: b"gpkg-bytes"})
    bad = good.replace(b"gpkg-bytes", b"gpkg-bytez") if corrupt == "member_crc" else b"<html>"
    first = _patch_get(monkeypatch, bad)
    with pytest.raises(RuntimeError, match="not a valid zip"):
        geo.download_swissboundaries(tmp_path, "2026-01")
    assert list(tmp_path.iterdir()) == []  # no partial gpkg, no stale zip, no marker
    second = _patch_get(monkeypatch, good)
    assert geo.download_swissboundaries(tmp_path, "2026-01").read_bytes() == b"gpkg-bytes"
    assert len(first) == len(second) == 1  # the next call downloads again, no overwrite needed


def test_download_swissboundaries_overwrite_redownloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_get(monkeypatch, _zip_bytes({GPKG_NAME: b"old"}))
    gpkg = geo.download_swissboundaries(tmp_path, "2026-01")
    (tmp_path / "swissBOUNDARIES3D_zz_other.gpkg").write_bytes(b"other")  # sorts last
    calls = _patch_get(monkeypatch, _zip_bytes({GPKG_NAME: b"new"}))
    assert geo.download_swissboundaries(tmp_path, "2026-01") == gpkg  # the marker's file
    assert gpkg.read_bytes() == b"old" and calls == []
    assert geo.download_swissboundaries(tmp_path, "2026-01", overwrite=True).read_bytes() == b"new"
    assert len(calls) == 1


@pytest.mark.parametrize("marker", [None, "2026-01"])  # manual copy / legacy marker
def test_download_swissboundaries_accepts_only_readable_unmarked_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_gpkg: Path, marker: str | None
) -> None:
    if marker is not None:
        (tmp_path / ".sb3d_version").write_text(marker, encoding="utf-8")
    valid = tmp_path / "swissBOUNDARIES3D_a.gpkg"
    shutil.copyfile(synthetic_gpkg, valid)
    (tmp_path / "swissBOUNDARIES3D_z.gpkg").write_bytes(b"")  # broken leftover, sorts last
    calls = _patch_get(monkeypatch, _zip_bytes({GPKG_NAME: b"gpkg-bytes"}))
    assert geo.download_swissboundaries(tmp_path, "2026-01") == valid and calls == []
    valid.unlink()
    assert geo.download_swissboundaries(tmp_path, "2026-01").name == GPKG_NAME
    assert len(calls) == 1


def test_load_admin_units_builds_hierarchy(units: gpd.GeoDataFrame) -> None:
    assert list(units["municipality_id"]) == [351, 700, 6621]  # lake and Vaduz dropped
    assert set(geo.UNIT_COLUMNS) | {"lang_region", "geometry"} == set(units.columns)
    assert units.crs.to_epsg() == 2056
    assert not units.geometry.has_z.any()
    rows = units.set_index("municipality_name")
    assert rows.loc["Genève", "district_id"] == 2500  # canton without districts
    assert rows.loc["Genève", "district_name"] == "Genève"
    assert rows.loc["Moutierville", "district_name"] == "Jura bernois"
    assert rows.loc["Moutierville", "lang_region"] == "fr"
    assert rows.loc["Bern", "canton"] == "BE" and rows.loc["Bern", "canton_id"] == 2
    assert rows.loc["Bern", "muni_area_km2"] == pytest.approx(1.0)  # lake excluded
    assert rows.loc["Bern", "muni_density"] == pytest.approx(10000.0)


def test_load_admin_units_total_area_option(synthetic_gpkg: Path) -> None:
    rows = geo.load_admin_units(synthetic_gpkg, exclude_lakes=False).set_index("municipality_id")
    assert rows.loc[351, "muni_area_km2"] == pytest.approx(2.0)
    assert rows.loc[351, "muni_density"] == pytest.approx(5000.0)


def test_load_admin_units_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        geo.load_admin_units(tmp_path / "missing.gpkg")


def test_load_admin_units_rejects_unknown_canton(tmp_path: Path) -> None:
    rows = [("Gemeindegebiet", 1, np.nan, 99.0, "Nowhere", 10, 100.0, 0.0, 0, 1000)]
    with pytest.raises(ValueError, match="canton"):
        geo.load_admin_units(_write_gpkg(tmp_path / "bad.gpkg", rows))


def test_load_admin_units_without_lake_field_uses_total_area(
    synthetic_gpkg: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "no_lake.gpkg"
    for layer in (geo.MUNICIPALITY_LAYER, geo.DISTRICT_LAYER, geo.CANTON_LAYER):
        table = gpd.read_file(synthetic_gpkg, layer=layer).drop(
            columns="see_flaeche", errors="ignore"
        )
        table.to_file(path, layer=layer, driver="GPKG")
    with caplog.at_level(logging.WARNING, logger="rentml.geo"):
        rows = geo.load_admin_units(path).set_index("municipality_id")
    assert rows.loc[351, "muni_area_km2"] == pytest.approx(2.0)
    assert "see_flaeche" in caplog.text


def test_load_foreign_mask_returns_non_swiss_municipalities(
    synthetic_gpkg: Path, units: gpd.GeoDataFrame, tmp_path: Path
) -> None:
    mask = geo.load_foreign_mask(synthetic_gpkg)
    assert len(mask) == 1 and mask.crs.to_epsg() == 2056 and not mask.has_z.any()
    assert mask.iloc[0].bounds == (E0 + 8500, N0, E0 + 9500, N0 + 1000)  # Vaduz
    vaduz = _points([9000.0], [1])
    assert geo.assign_admin_units(vaduz, units, max_snap_m=5000)["admin_match"].item() == "nearest"
    out = geo.assign_admin_units(vaduz, units, max_snap_m=5000, exclude=mask)
    assert out["admin_match"].item() == "none" and pd.isna(out["municipality_id"].item())
    assert geo.load_foreign_mask(_write_gpkg(tmp_path / "ch.gpkg", MUNICIPALITY_ROWS[:3])).empty
    with pytest.raises(FileNotFoundError):
        geo.load_foreign_mask(tmp_path / "missing.gpkg")


def test_assign_admin_units_within_nearest_none(units: gpd.GeoDataFrame) -> None:
    index = [101, 55, 7, 300, 42, 9]
    df = _points([500.0, 5500.0, 2100.0, 9000.0, np.nan, 50_000.0], index)
    out = geo.assign_admin_units(df, units)
    assert list(out.index) == index and len(out) == len(df)
    assert list(out["admin_match"]) == ["within", "within", "nearest", "none", "none", "none"]
    assert out["municipality_id"].dtype == "Int64"
    assert out["municipality_id"].tolist()[:3] == [700, 6621, 351]
    assert out["municipality_id"].iloc[3:].isna().all()
    assert out["lang_region"].tolist()[:3] == ["fr", "fr", "de"]
    assert out["canton"].tolist()[:3] == ["BE", "GE", "BE"]
    assert out["price"].tolist() == df["price"].tolist()
    assert "municipality_id" not in df.columns  # input untouched


def test_assign_admin_units_zero_snap_disables_nearest(units: gpd.GeoDataFrame) -> None:
    out = geo.assign_admin_units(_points([2100.0], [1]), units, max_snap_m=0)
    assert out["admin_match"].tolist() == ["none"]


def test_assign_admin_units_resolves_overlaps_deterministically(
    units: gpd.GeoDataFrame,
) -> None:
    duplicate = units[units["municipality_id"] == 700].assign(municipality_id=1)
    overlapping = pd.concat([units, duplicate], ignore_index=True)
    out = geo.assign_admin_units(_points([500.0, 500.0], [3, 3]), overlapping)
    assert len(out) == 2
    assert out["municipality_id"].tolist() == [1, 1]  # lowest id wins, no row duplication


def test_assign_admin_units_rerun_replaces_columns(units: gpd.GeoDataFrame) -> None:
    once = geo.assign_admin_units(_points([500.0], [1]), units)
    twice = geo.assign_admin_units(once, units)
    assert list(twice.columns) == list(once.columns)
    assert twice["municipality_id"].tolist() == [700]


def test_assign_admin_units_input_errors(units: gpd.GeoDataFrame) -> None:
    with pytest.raises(KeyError):
        geo.assign_admin_units(pd.DataFrame({"x": [1.0]}), units)
    with pytest.raises(KeyError):
        geo.assign_admin_units(_points([500.0], [1]), units.drop(columns="lang_region"))
    with pytest.raises(ValueError, match="max_snap_m"):
        geo.assign_admin_units(_points([500.0], [1]), units, max_snap_m=-1.0)


def test_assign_admin_units_does_not_snap_points_abroad(units: gpd.GeoDataFrame) -> None:
    foreign = gpd.GeoSeries([_box3d(6000, 7000)], crs="EPSG:2056")  # touches Genève (5000-6000)
    df = _points([6500.0, 6000.0, 7500.0], [1, 2, 3])  # abroad, on the border, beyond the mask
    assert geo.assign_admin_units(df, units)["municipality_id"].tolist() == [6621] * 3
    out = geo.assign_admin_units(df, units, exclude=foreign)
    assert out["admin_match"].tolist() == ["none", "nearest", "nearest"]
    assert out["municipality_id"].tolist()[1:] == [6621, 6621]
    assert pd.isna(out["municipality_id"].iloc[0])


def test_assign_admin_units_border_tie_picks_lowest_id(units: gpd.GeoDataFrame) -> None:
    for ordered in (units, units.iloc[::-1]):
        out = geo.assign_admin_units(_points([1000.0], [1]), ordered)  # border of 700 and 351
        assert out["municipality_id"].tolist() == [351]
        assert out["admin_match"].tolist() == ["nearest"]


def test_assign_admin_units_empty_and_all_nan(units: gpd.GeoDataFrame) -> None:
    empty = geo.assign_admin_units(_points([], []), units)
    assert empty.empty and set(geo.ADMIN_COLUMNS) <= set(empty.columns)
    assert empty["municipality_id"].dtype == "Int64"
    nan = geo.assign_admin_units(_points([np.nan, np.nan], [8, 9]), units)
    assert list(nan.index) == [8, 9] and nan["admin_match"].tolist() == ["none", "none"]
    assert nan["municipality_id"].dtype == "Int64" and nan["municipality_id"].isna().all()
    assert nan["muni_density"].dtype == "float64"


def test_language_region_applies_canton_district_and_municipality_rules() -> None:
    canton = pd.Series(["ZH", "GE", "TI", "BE", "BE", "FR", "FR", "VS", "VS", "GR", "GR", "GR"])
    district = pd.Series(
        ["Zürich", "Genève", "Lugano", "Jura bernois", "Biel/Bienne", "Sense", "La Sarine"]
        + ["Visp", "Sion", "Moesa", "Maloja", "Maloja"]
    )
    municipality = pd.Series([""] * 10 + ["St. Moritz", "Bregaglia"])
    result = geo.language_region(canton, district, municipality_name=municipality)
    expected = ["de", "fr", "it", "fr", "de", "de", "fr", "de", "fr", "it", "de", "it"]
    assert result.tolist() == expected
    assert result.name == "lang_region"


def test_language_region_french_municipalities_in_see_lac() -> None:
    muni = pd.Series(["Courtepin", "Misery-Courtion", "Mont-Vully", "Murten", "Courgevaux"])
    canton, district = pd.Series(["FR"] * 5), pd.Series(["See"] * 5)
    result = geo.language_region(canton, district, municipality_name=muni)
    assert result.tolist() == ["fr", "fr", "fr", "de", "de"]


def test_language_region_normalises_names_and_keeps_missing() -> None:
    canton = pd.Series([" vs", None, "GR"], index=[5, 6, 7])
    district = pd.Series(["  VISP ", "Anything", "Maloja"], index=[5, 6, 7])
    result = geo.language_region(canton, district)
    assert list(result.index) == [5, 6, 7]
    assert result.iloc[0] == "de" and pd.isna(result.iloc[1]) and result.iloc[2] == "de"


def test_language_region_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="Unknown canton"):
        geo.language_region(pd.Series(["XX"]), pd.Series(["Foo"]))
    with pytest.raises(ValueError, match="equal length"):
        geo.language_region(pd.Series(["ZH", "BE"]), pd.Series(["Zürich"]))


def test_parse_address_is_reexported_from_address_module() -> None:
    assert geo.parse_address is address.parse_address  # rentml.geo.parse_address stays valid


def test_nested_group_keys_builds_globally_unique_keys() -> None:
    df = pd.DataFrame(
        {
            "canton": ["ZH", "BE", None],
            "district_id": [112.0, 246.0, np.nan],
            "municipality_id": pd.array([261, 351, None], dtype="Int64"),
        },
        index=[3, 1, 2],
    )
    keys = geo.nested_group_keys(df)
    assert list(keys.index) == [3, 1, 2]
    assert keys.loc[3].tolist() == ["ZH", "ZH/112", "ZH/112/261"]
    assert keys.loc[1, "re_municipality"] == "BE/246/351"
    assert keys.loc[2].isna().all()


def test_nested_group_keys_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        geo.nested_group_keys(pd.DataFrame({"canton": ["ZH"], "district_id": [112]}))


def test_listings_per_unit_counts_training_rows() -> None:
    train = pd.DataFrame({"municipality_id": [261, 261, 351, None]})
    counts = geo.listings_per_unit(train)
    assert counts.to_dict() == {261: 2, 351: 1}
    assert counts.name == "n_listings"


def test_listings_per_unit_missing_column_raises() -> None:
    with pytest.raises(KeyError):
        geo.listings_per_unit(pd.DataFrame({"x": [1]}), col="district_id")


def test_count_bucket_edges() -> None:
    counts = pd.Series([0, 4, 5, 20, 21, np.nan], index=list("abcdef"))
    buckets = geo.count_bucket(counts)
    assert buckets.tolist() == ["<5", "<5", "5-20", "5-20", ">20", "<5"]
    assert list(buckets.index) == list("abcdef")
    assert set(buckets) <= set(geo.COUNT_BUCKET_ORDER)


def test_count_bucket_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        geo.count_bucket(pd.Series([3, -1]))
    with pytest.raises(ValueError, match="low"):
        geo.count_bucket(pd.Series([3]), low=10, high=5)


@pytest.mark.skipif(not REAL_GPKG.is_file(), reason="swissBOUNDARIES3D GeoPackage not downloaded")
def test_real_gpkg_assigns_known_cities() -> None:
    units = geo.load_admin_units(REAL_GPKG)
    assert len(units) > 2000 and units["municipality_id"].is_unique
    df = pd.DataFrame(
        {"east": [2683000.0, 2538000.0, 2717000.0], "north": [1248000.0, 1152000.0, 1096000.0]},
        index=[1, 2, 3],
    )
    out = geo.assign_admin_units(df, units)
    assert out["municipality_name"].tolist() == ["Zürich", "Lausanne", "Lugano"]
    assert out["canton"].tolist() == ["ZH", "VD", "TI"]
    assert out["lang_region"].tolist() == ["de", "fr", "it"]
    assert (out["admin_match"] == "within").all()
    keys = geo.nested_group_keys(out)
    assert keys["re_municipality"].iloc[0] == f"ZH/{out['district_id'].iloc[0]}/261"
