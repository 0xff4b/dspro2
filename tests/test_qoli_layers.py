"""Tests for rentml.qoli_layers (synthetic rasters and vectors, no network)."""

import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from rentml.amenities import DEFAULT_ACCESS_WEIGHTS, OSM_CATEGORIES
from rentml.qoli_layers import (
    LAYER_KEYS,
    NOISE_KEYS,
    QoliLayers,
    apply_quiet_noise,
    energetic_sum,
    focal_max,
    green_share,
    lden_from_lr,
    load_hectares,
    load_landuse,
    load_oev,
    noise_levels,
    oev_indicators,
    raster_values,
)

E0, N0 = 2_600_000.0, 1_200_100.0  # upper-left corner of the synthetic 10 m rasters
NODATA = -9999.0


def _write_tif(path: Path, data: np.ndarray, res: float = 10.0) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float64",
        crs="EPSG:2056",
        transform=from_origin(E0, N0, res, res),
        nodata=NODATA,
    ) as dst:
        dst.write(data, 1)
    return path


def _cell(row: int, col: int, res: float = 10.0) -> tuple[float, float]:
    return E0 + (col + 0.5) * res, N0 - (row + 0.5) * res


@pytest.fixture
def noise_tif(tmp_path: Path) -> Path:
    data = np.full((10, 10), NODATA)
    data[2, 2] = 50.0  # isolated loud cell
    data[2, 4] = 40.0
    data[8, 8] = 30.0
    return _write_tif(tmp_path / "noise.tif", data)


def test_focal_max_radius_zero_reads_the_cell(noise_tif: Path) -> None:
    xy = np.array([_cell(2, 2), _cell(5, 5), [np.nan, np.nan], [E0 - 500.0, N0]])
    values = raster_values(noise_tif, xy)
    assert values[0] == 50.0
    assert np.isnan(values[1:]).all()  # nodata, NaN coordinates, outside the raster


def test_focal_max_takes_the_maximum_within_the_square(noise_tif: Path) -> None:
    xy = np.array([_cell(2, 3), _cell(3, 3), _cell(5, 5)])
    out = focal_max(noise_tif, xy, radii_m=(10.0, 30.0), strip_rows=3)  # strips + halo
    np.testing.assert_allclose(out[:, 0], [50.0, 50.0, np.nan])
    np.testing.assert_allclose(out[:, 1], [50.0, 50.0, 50.0])
    with pytest.raises(ValueError, match="radii"):
        focal_max(noise_tif, xy, radii_m=(-1.0,))


def test_noise_levels_point_facade_fallback_quiet_and_outside(noise_tif: Path) -> None:
    # (2, 2) own cell; (2, 3) in a "building" next to 50 and 40 dB cells; (5, 5) only 30 m
    # rings reach a value; (0, 9) nothing within 30 m; and a point outside the raster.
    xy = np.array([_cell(2, 2), _cell(2, 3), _cell(5, 5), _cell(0, 9), [E0 - 500.0, N0]])
    out = noise_levels(noise_tif, xy, radii_m=(0.0, 10.0, 30.0))
    assert out["how"].tolist() == ["point", "facade", "fallback", "quiet", "outside"]
    np.testing.assert_allclose(out["level_db"], [50.0, 50.0, 50.0, np.nan, np.nan])
    np.testing.assert_allclose(out["radius_m"], [0.0, 10.0, 30.0, np.nan, np.nan])


def test_noise_levels_uses_smallest_ring_not_the_loudest_cell(noise_tif: Path) -> None:
    # (2, 5): the 40 dB cell is adjacent; the louder 50 dB cell is only in the 30 m ring.
    out = noise_levels(noise_tif, np.array([_cell(2, 5)]), radii_m=(0.0, 10.0, 30.0))
    assert out["level_db"].iloc[0] == 40.0 and out["how"].iloc[0] == "facade"
    for bad in ((), (10.0, 10.0), (20.0, 10.0)):
        with pytest.raises(ValueError, match="radii"):
            noise_levels(noise_tif, np.array([_cell(2, 5)]), radii_m=bad)


def test_energetic_sum_and_lden() -> None:
    total = energetic_sum(np.array([60.0, 50.0, np.nan]), np.array([60.0, np.nan, np.nan]))
    np.testing.assert_allclose(total[:2], [60.0 + 10 * np.log10(2), 50.0])
    assert np.isnan(total[2])
    # Equal day and night levels: Lden = L + 10 log10((12 + 4·10^0.5 + 8·10) / 24).
    expected = 50.0 + 10 * np.log10((12 + 4 * 10**0.5 + 8 * 10) / 24)
    assert lden_from_lr(np.array([50.0]), np.array([50.0]))[0] == pytest.approx(expected)
    with pytest.raises(ValueError):
        energetic_sum()
    with pytest.raises(ValueError, match="shape"):
        lden_from_lr(np.zeros(2), np.zeros(3))


def test_apply_quiet_noise_sets_best_score() -> None:
    norm = pd.DataFrame({"noise_day_db": [np.nan, 40.0], "noise_night_db": [np.nan, np.nan]})
    raw = pd.DataFrame({"noise_quiet_day": [True, False], "noise_quiet_night": [True, False]})
    out = apply_quiet_noise(norm, raw)
    assert out["noise_day_db"].tolist() == [100.0, 40.0]
    assert out["noise_night_db"].iloc[0] == 100.0 and np.isnan(out["noise_night_db"].iloc[1])
    assert np.isnan(norm.iloc[0, 0])  # input untouched


def _oev_zip(tmp_path: Path) -> Path:
    gpkg = tmp_path / "OeV_Gueteklassen_ARE.gpkg"
    classes = gpd.GeoDataFrame(
        {"KLASSE": ["A", "C"]},
        geometry=[box(0, 0, 100, 100), box(0, 0, 300, 300)],  # A overlaps C
        crs=2056,
    )
    classes.to_file(gpkg, layer="OeV_Gueteklassen_ARE")
    stops = gpd.GeoDataFrame(
        {"Hst_Kat": [1, 0]}, geometry=[Point(50, 50), Point(1000, 1000)], crs=2056
    )
    stops.to_file(gpkg, layer="OeV_Haltestellen_ARE")
    archive = tmp_path / "oev.gpkg.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(gpkg, gpkg.name)
    gpkg.unlink()
    return archive


def test_load_oev_and_indicators(tmp_path: Path) -> None:
    classes, stops = load_oev(_oev_zip(tmp_path))
    assert len(stops) == 1  # category 0 (unclassified) dropped
    xy = np.array([[50.0, 50.0], [200.0, 200.0], [500.0, 500.0], [np.nan, np.nan]])
    out = oev_indicators(xy, classes, stops)
    np.testing.assert_allclose(out["oev_class"], [4.0, 2.0, 0.0, np.nan])  # best class wins
    assert out["oev_stop_m"].iloc[0] == pytest.approx(0.0)
    assert out["oev_stop_m"].iloc[2] == pytest.approx(np.hypot(450, 450))


def test_green_share_counts_green_hectares_within_radius() -> None:
    landuse = pd.DataFrame(
        {"east": [0.0, 100.0, 200.0, 5000.0], "north": [0.0] * 4, "as17": [10, 7, 5, 10]}
    )
    out = green_share(np.array([[0.0, 0.0], [9000.0, 0.0]]), landuse, radius_m=250.0)
    assert out[0] == pytest.approx(2 / 3)
    assert np.isnan(out[1])  # no hectare nearby


def test_load_landuse_and_hectares_shift_to_centres(tmp_path: Path) -> None:
    landuse_csv = tmp_path / "landuse.csv"
    pd.DataFrame({"RELI": [1], "E_COORD": [2600000], "N_COORD": [1200000], "AS_17": [5]}).to_csv(
        landuse_csv, sep=";", index=False
    )
    landuse = load_landuse(landuse_csv)
    assert landuse.iloc[0].tolist() == [2600050.0, 1200050.0, 5]
    services = {
        c: [100.0, 200.0]
        for c in ["D_GROCERY", "D_PHARMA", "D_MEDIC", "D_SCHOOL_O", "D_RESTO", "D_STOP_TOT"]
    }
    hect_csv = tmp_path / "hect.csv"
    pd.DataFrame(
        {
            "RELI": [7, 7],
            "E_COORD": [2600000] * 2,
            "N_COORD": [1200000] * 2,
            "YEAR": [2018, 2021],
            "POP_TOTAL": [3, 5],
        }
        | services
    ).to_csv(hect_csv, sep=";", index=False)
    hect = load_hectares(hect_csv, year=2021)
    assert hect.index.tolist() == [7] and hect["population"].iloc[0] == 5.0
    assert {"d_grocery", "d_stop_tot"} <= set(hect.columns)
    with pytest.raises(ValueError, match="year"):
        load_hectares(hect_csv, year=1990)


def test_qoli_layers_indicators_end_to_end(tmp_path: Path) -> None:
    noise = np.full((30, 30), NODATA)
    noise[1, 1] = 60.0
    paths = {k: _write_tif(tmp_path / f"{k}.tif", noise) for k in NOISE_KEYS}
    flat = np.full((30, 30), 12.0)
    paths |= {k: _write_tif(tmp_path / f"{k}.tif", flat) for k in ("no2", "pm25", "sunshine")}
    landuse_csv = tmp_path / "landuse.csv"
    pd.DataFrame({"E_COORD": [2600000], "N_COORD": [1200000], "AS_17": [10]}).to_csv(
        landuse_csv, sep=";", index=False
    )
    paths |= {"landuse": landuse_csv, "oev": _oev_zip(tmp_path)}
    assert set(paths) == set(LAYER_KEYS)
    osm = pd.DataFrame(
        {
            "category": ["supermarket", "park"],
            "east": [E0 + 15, E0 + 15],
            "north": [N0 - 15, N0 - 15],
        }
    )
    layers = QoliLayers.load(paths, osm)
    points = pd.DataFrame(
        {"east": [E0 + 15, E0 + 285], "north": [N0 - 15, N0 - 285]},
        index=pd.Index([11, 12], name="listing_id"),
    )
    out = layers.indicators(points)
    assert out.index.equals(points.index)
    assert out["road_day_db"].iloc[0] == 60.0
    assert out["noise_day_db"].iloc[0] == pytest.approx(60.0 + 10 * np.log10(2))
    assert out["noise_quiet_day"].tolist() == [False, True]
    assert out["acc_supermarket"].iloc[0] == pytest.approx(100.0)
    # acc_total averages the seven non-park categories by weight sum, acc_total_all all eight.
    weight_sums = {c: sum(DEFAULT_ACCESS_WEIGHTS.get(c, [1.0])) for c in OSM_CATEGORIES}
    total = sum(weight_sums.values())
    expected = 100.0 * weight_sums["supermarket"] / (total - 1.0)
    assert out["acc_total"].iloc[0] == pytest.approx(expected)
    expected_all = 100.0 * (weight_sums["supermarket"] + 1.0) / total
    assert out["acc_total_all"].iloc[0] == pytest.approx(expected_all)
    assert out["acc_park"].iloc[0] == pytest.approx(100.0)
    assert (out["no2"] == 12.0).all()
    with pytest.raises(KeyError, match="missing"):
        QoliLayers.load({"oev": paths["oev"]})
