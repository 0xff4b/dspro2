"""Tests for rentml.qoli_data (synthetic rasters and tables; HTTP is mocked)."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from rentml import qoli_data
from rentml.qoli_data import (
    SOURCES,
    accessibility,
    amenity_arrays,
    city_statistics,
    fetch_osm_amenities,
    fetch_sdmx_csv,
    green_share,
    load_tax_burden,
    noise_levels,
    oev_class,
    raster_values,
    remap_municipalities,
    source_table,
    vacancy_rates,
)

E0, N0 = 2_600_000.0, 1_200_000.0


class _Response:
    def __init__(self, content: bytes = b"", payload: dict | None = None) -> None:
        self.content, self.payload, self.status_code = content, payload, 200

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.content]

    def json(self) -> dict:
        return self.payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _raster(path: Path, values: np.ndarray) -> Path:
    """10 m LV95 raster with its top-left corner at (E0, N0); NaN cells become nodata."""
    data = np.where(np.isnan(values), -9999.0, values).astype("float32")
    profile = {"driver": "GTiff", "height": data.shape[0], "width": data.shape[1], "count": 1,
               "dtype": "float32", "crs": "EPSG:2056", "nodata": -9999.0,
               "transform": from_origin(E0, N0, 10.0, 10.0)}  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def _cell(row: int, col: int) -> list[float]:
    return [E0 + 10 * col + 5, N0 - 10 * row - 5]


def test_source_registry_has_licences_and_cached_file_names() -> None:
    table = source_table()
    assert set(table.index) == set(SOURCES)
    assert table["licence"].str.len().gt(0).all() and table["reference"].str.len().gt(0).all()
    assert SOURCES["road_day"].filename == "laerm-strassenlaerm_tag_2056.tif"


def test_fetch_sdmx_csv_downloads_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_get(url: str, **kwargs: object) -> _Response:
        calls.append(url)
        return _Response(b"A: label,OBS_VALUE\nx: X,1\n")

    monkeypatch.setattr(qoli_data.requests, "get", fake_get)
    first = fetch_sdmx_csv("CH1.LWZ,DF_LWZ_1,1.1.0", tmp_path / "lwz.csv")
    second = fetch_sdmx_csv("CH1.LWZ,DF_LWZ_1,1.1.0", tmp_path / "lwz.csv")
    assert len(calls) == 1 and first.equals(second) and first["OBS_VALUE"].iloc[0] == "1"


def test_raster_values_nodata_and_outside(tmp_path: Path) -> None:
    path = _raster(tmp_path / "r.tif", np.array([[1.0, np.nan], [3.0, 4.0]]))
    xy = np.array([_cell(0, 0), _cell(0, 1), _cell(1, 1), [E0 - 100, N0]])
    values = raster_values(path, xy)
    assert values[[0, 2]].tolist() == [1.0, 4.0] and np.isnan(values[[1, 3]]).all()


def test_noise_levels_point_facade_fallback_quiet_outside(tmp_path: Path) -> None:
    grid = np.full((30, 30), np.nan)
    grid[0, 0] = 70.0  # far corner: only reachable with the 100 m fallback
    grid[15, 17], grid[15, 18] = 55.0, 60.0  # façade 20 m east of (15, 15)
    grid[29, 29] = 40.0
    path = _raster(tmp_path / "noise.tif", grid)
    xy = np.array([_cell(29, 29), _cell(15, 15), _cell(5, 5), _cell(25, 15), [E0 - 50, N0]])
    level, how = noise_levels(path, xy)
    assert how.tolist() == ["point", "facade", "fallback", "fallback", "outside"]
    assert level[:4].tolist() == [40.0, 55.0, 70.0, 60.0] and np.isnan(level[4])
    grid_quiet = _raster(tmp_path / "quiet.tif", np.full((30, 30), np.nan))
    assert noise_levels(grid_quiet, xy[:1])[1].tolist() == ["quiet"]


def test_accessibility_decay_and_missing_coordinates() -> None:
    points = pd.DataFrame({"east": [0.0, 1400.0, 5000.0, np.nan], "north": 0.0})
    amenities = {"supermarket": np.array([[0.0, 0.0]]), "pharmacy": np.empty((0, 2))}
    acc = accessibility(points, amenities)
    assert acc["acc_supermarket"].iloc[:3].tolist() == pytest.approx([100.0, 50.0, 0.0])
    assert acc["acc_pharmacy"].iloc[0] == 0.0 and acc["acc_total"].iloc[0] == 75.0
    assert acc.iloc[3].isna().all()


def test_amenity_arrays_keeps_empty_categories() -> None:
    osm = pd.DataFrame({"category": ["park"], "east": [1.0], "north": [2.0]})
    arrays = amenity_arrays(osm)
    assert arrays["park"].tolist() == [[1.0, 2.0]] and arrays["gym"].shape == (0, 2)


def test_fetch_osm_amenities_parses_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    elements = [{"type": "node", "id": 1, "lat": 47.0, "lon": 8.0, "tags": {"name": "A"}},
                {"type": "way", "id": 2, "center": {"lat": 46.5, "lon": 7.5}}]  # fmt: skip
    monkeypatch.setattr(qoli_data.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        qoli_data.requests, "post", lambda *a, **k: _Response(payload={"elements": elements})
    )
    osm = fetch_osm_amenities(tmp_path / "osm.parquet")
    assert len(osm) == 2 * len(qoli_data.OSM_CATEGORIES)
    assert osm["east"].between(2_400_000, 2_900_000).all()
    monkeypatch.setattr(qoli_data.requests, "post", None)  # second call reads the cache
    assert len(fetch_osm_amenities(tmp_path / "osm.parquet")) == len(osm)


def test_green_share_counts_green_hectares_in_radius() -> None:
    landuse = pd.DataFrame({"east": [0.0, 100.0, 200.0, 5000.0], "north": 0.0,
                            "as17": [10, 1, 5, 10]})  # fmt: skip
    share = green_share(np.array([[0.0, 0.0], [9000.0, 0.0]]), landuse, radius_m=250.0)
    assert share[0] == pytest.approx(2 / 3) and np.isnan(share[1])


def test_oev_class_takes_best_overlapping_class() -> None:
    classes = gpd.GeoDataFrame({"score": [2, 4]}, geometry=[box(0, 0, 10, 10), box(5, 5, 10, 10)],
                               crs=2056)  # fmt: skip
    xy = np.array([[1.0, 1.0], [6.0, 6.0], [50.0, 50.0]])
    assert oev_class(xy, classes).tolist() == [2.0, 4.0, 0.0]


def test_vacancy_rates_latest_municipal_totals() -> None:
    cols = ["GR_KT_GDE: Units", "WOHN_ANZAHL: Rooms", "LEERWOHN_TYP: Type",
            "MEASURE_DIMENSION: Measure", "TIME_PERIOD: Period", "OBS_VALUE"]  # fmt: skip
    table = pd.DataFrame([
        ("261: Zürich", "_T: Total", "_T: All", "RATE: Rate", "2026", "0.1"),
        ("261: Zürich", "_T: Total", "_T: All", "RATE: Rate", "2025", "0.2"),
        ("261: Zürich", "2: 2 rooms", "_T: All", "RATE: Rate", "2026", "9.9"),
        ("ZH: Canton ZH", "_T: Total", "_T: All", "RATE: Rate", "2026", "0.5"),
        ("B_112: Bezirk Zürich", "_T: Total", "_T: All", "RATE: Rate", "2026", "0.3"),
        ("351: Bern", "_T: Total", "_T: All", "RATE: Rate", "2026", "0.4"),
    ], columns=cols)  # fmt: skip
    assert vacancy_rates(table).to_dict() == {261: 0.1, 351: 0.4}


def test_city_statistics_latest_observed_year() -> None:
    cols = ["ID_RAUM: Area", "ID_INDIC_DISPR: Variable", "STATISTICAL_OPERATION: Op",
            "TIME_PERIOD: Period", "OBS_VALUE"]  # fmt: skip
    table = pd.DataFrame([
        ("261901: Zürich", "SA1025I: Vacancy", "OBS: Observed", "2020", "1.0"),
        ("261901: Zürich", "SA1025I: Vacancy", "OBS: Observed", "2022", "2.0"),
        ("261901: Zürich", "SA1025I: Vacancy", "OBS: Observed", "2024", ""),
        ("261902: Zürich agglo", "SA1025I: Vacancy", "OBS: Observed", "2024", "9.0"),
    ], columns=cols)  # fmt: skip
    out = city_statistics(table, ["SA1025I", "EC3063I"])
    assert out.loc[261, "SA1025I"] == 2.0 and np.isnan(out.loc[261, "EC3063I"])


def test_load_tax_burden_and_remap_mergers(tmp_path: Path) -> None:
    rows = [[None, None, 50_000, 80_000], [None, "Gemeindenummer", None, None],
            ["A", 1, 5.0, 8.0], ["B", 2, 6.0, 10.0], ["C", 3, 7.0, 12.0]]  # fmt: skip
    sheet = pd.DataFrame(rows)
    xlsx = tmp_path / "tax.xlsx"
    sheet.to_excel(xlsx, sheet_name="Ledig", header=False, index=False)
    tax = load_tax_burden(xlsx)
    assert tax.to_dict() == {1: 8.0, 2: 10.0, 3: 12.0}
    with pytest.raises(ValueError):
        load_tax_burden(xlsx, gross_income=99)
    corr = pd.DataFrame({"initial_id": [1, 2, 3], "terminal_id": [10, 10, 3]})
    assert remap_municipalities(tax, corr).to_dict() == {3: 12.0, 10: 9.0}
