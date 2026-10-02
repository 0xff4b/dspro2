"""Shared fixtures: a tiny synthetic Switzerland for the map modules (no network, no files)."""

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon, box

from rentml.features import HierarchicalTargetEncoder, add_engineered_features
from rentml.geo import nested_group_keys
from rentml.mapbundle import MapBundle, assemble_bundle
from rentml.mapdata import aggregate_listings, build_levels
from rentml.rentcheck import RentModelBundle

E0, N0 = 2_600_000.0, 1_200_000.0

# (bfs, district_id, canton_id, name, district, canton, population, lang, x0, x1) in km
MAP_MUNICIPALITIES = [
    (1, 101, 1, "Aach", "Nord", "ZH", 1000.0, "de", 0, 1),
    (2, 101, 1, "Bex", "Nord", "ZH", 3000.0, "fr", 1, 2),
    (3, 102, 1, "Cham", "Süd", "ZH", 500.0, "de", 2, 3),
    (4, 2500, 25, "Genève", "Genève", "GE", 2000.0, "fr", 4, 5),
]


def _km_box(x0: float, x1: float, y0: float = 0.0, y1: float = 1.0) -> box:
    return box(E0 + 1000 * x0, N0 + 1000 * y0, E0 + 1000 * x1, N0 + 1000 * y1)


@pytest.fixture
def map_units() -> gpd.GeoDataFrame:
    """Four municipalities in three districts and two cantons (output of load_admin_units)."""
    rows = MAP_MUNICIPALITIES
    frame = pd.DataFrame(
        {
            "municipality_id": [r[0] for r in rows],
            "district_id": [r[1] for r in rows],
            "canton_id": [r[2] for r in rows],
            "municipality_name": [r[3] for r in rows],
            "district_name": [r[4] for r in rows],
            "canton": [r[5] for r in rows],
            "muni_population": [r[6] for r in rows],
            "muni_area_km2": [1.0] * len(rows),
            "lang_region": [r[7] for r in rows],
        }
    )
    frame["muni_density"] = frame["muni_population"] / frame["muni_area_km2"]
    geometry = [_km_box(r[8], r[9]) for r in rows]
    return gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:2056")


@pytest.fixture
def map_other() -> gpd.GeoDataFrame:
    """A cantonal lake north of the three ZH municipalities (shares their vertices)."""
    ring = [(0, 1), (0, 1.5), (3, 1.5), (3, 1), (2, 1), (1, 1)]
    lake = Polygon([(E0 + 1000 * x, N0 + 1000 * y) for x, y in ring])
    return gpd.GeoDataFrame(
        {"name": ["Testsee"], "canton_id": [1], "kind": ["lake"]},
        geometry=[lake],
        crs="EPSG:2056",
    )


@pytest.fixture
def map_listings() -> pd.DataFrame:
    """25 objects in Aach (CHF 20/m²), 5 in Cham (CHF 40/m²), none elsewhere."""
    rng = np.random.default_rng(42)
    area = rng.integers(40, 120, 30).astype(float)
    rate = np.r_[np.full(25, 20.0), np.full(5, 40.0)]
    return pd.DataFrame(
        {
            "municipality_id": [1] * 25 + [3] * 5,
            "district_id": [101] * 25 + [102] * 5,
            "canton_id": [1] * 30,
            "area": area,
            "price": area * rate,
        }
    )


@pytest.fixture
def map_bundle(map_units, map_other, map_listings) -> MapBundle:
    """Assembled map bundle of the synthetic Switzerland."""
    levels, other = build_levels(map_units, map_other, canton_names={1: "Zürich"})
    meta = {
        "min_count": 20,
        "n_objects": 30,
        "created_at": "2026-10-02T12:00:00+00:00",
        "swiss": {"chf_per_m2": 20.0, "rent": 1600.0},
    }
    return assemble_bundle(levels, other, aggregate_listings(map_listings), meta=meta)


class _QuantileStub:
    """Quantiles = point prediction ± fixed log offsets (enough to exercise the pipeline)."""

    levels_ = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    _offsets = np.array([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30])

    def __init__(self, point: lgb.LGBMRegressor) -> None:
        self.point = point

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.point.predict(X)[:, None] + self._offsets[None, :]


@pytest.fixture
def rent_model() -> RentModelBundle:
    """Small but real RentModelBundle: LightGBM on synthetic rows, fitted target encoder."""
    rng = np.random.default_rng(42)
    n = 400
    muni = rng.choice([1, 2, 3, 4], n)
    district = np.select([muni <= 2, muni == 3], [101, 102], 2500)
    raw = pd.DataFrame(
        {
            "area": rng.uniform(30, 150, n),
            "rooms": rng.choice([1.5, 2.5, 3.5, 4.5], n),
            "east": E0 + rng.uniform(0, 5000, n),
            "north": N0 + rng.uniform(0, 1000, n),
            "canton": np.where(muni == 4, "GE", "ZH"),
            "district_id": district,
            "municipality_id": muni,
            "muni_density": 1000.0,
            "muni_population": 1000.0,
            "year_built": rng.uniform(1900, 2020, n),
            "apartments": rng.integers(1, 30, n),
            "land_area": rng.uniform(100, 900, n),
            "oev": rng.uniform(0, 1e5, n),
            "solar": rng.integers(1, 6, n),
            "population": rng.integers(0, 300, n),
            "elevation": rng.uniform(300, 900, n),
        }
    )
    y = np.log(raw["area"] * np.where(raw["canton"].eq("GE"), 35.0, 22.0))
    raw = raw.join(nested_group_keys(raw))
    encoder = HierarchicalTargetEncoder().fit(raw, y)
    X = add_engineered_features(pd.concat([raw, encoder.transform(raw)], axis=1))
    cols = [
        *[f"te_{c}" for c in encoder.cols],
        "area",
        "rooms",
        "area_per_room",
        "year_built",
        "building_age",
        "apartments",
        "land_area",
        "land_area_per_apartment",
        "population",
        "oev",
        "solar",
        "elevation",
        "muni_density",
        "muni_population",
        "east",
        "north",
        "rot45_x",
        "rot45_y",
    ]
    point = lgb.LGBMRegressor(n_estimators=60, random_state=42, verbose=-1).fit(X[cols], y)
    return RentModelBundle(
        feature_cols=cols,
        point_model=point,
        quantile_model=_QuantileStub(point),
        calibrator=None,
        group_cols=["lang_region"],
        metadata={"te_encoder": encoder, "coverage": 0.8, "interval_levels": (0.1, 0.9)},
    )
