"""Tests for rentml.estimate (synthetic map and a small real LightGBM bundle; no network)."""

import numpy as np
import pytest

from rentml.estimate import Apartment, EstimateError, MunicipalityLocator, RentEstimator
from rentml.geoadmin import Address, LocationData

E0, N0 = 2_600_000.0, 1_200_000.0
LOCATION = LocationData(
    year_built=1980.0,
    apartments=8.0,
    land_area=300.0,
    oev=40000.0,
    solar=3.0,
    population=120.0,
    elevation=450.0,
)


def _apartment(east: float = E0 + 500, area: float = 80.0, text: str | None = None) -> Apartment:
    return Apartment(Address("Teststrasse 1", east, N0 + 500, 1), area, 3.5, text)


def test_locator_finds_municipality(map_bundle) -> None:
    locator = MunicipalityLocator(map_bundle)
    assert locator.locate(E0 + 500, N0 + 500)["name"] == "Aach"
    assert locator.locate(E0 + 4500, N0 + 500)["canton"] == "GE"
    assert locator.locate(E0 + 3200, N0 + 500)["name"] == "Cham"  # gap: snapped to nearest
    assert locator.locate(E0 + 50_000, N0) is None


def test_features_match_the_model_columns(rent_model, map_bundle) -> None:
    estimator = RentEstimator(rent_model, map_bundle)
    x_row, groups, unit = estimator.features(_apartment(), LOCATION)
    assert list(groups.columns) == ["lang_region"] and groups.iloc[0, 0] == "de"
    assert unit["name"] == "Aach"
    assert set(rent_model.feature_cols) <= set(x_row.columns)
    assert x_row["area_per_room"].iloc[0] == pytest.approx(80 / 3.5)
    assert x_row["building_age"].iloc[0] > 0
    assert np.isfinite(x_row[rent_model.feature_cols].to_numpy(dtype=float)).all()


def test_estimate_with_check_drivers_and_text(rent_model, map_bundle) -> None:
    estimator = RentEstimator(rent_model, map_bundle)
    text = "Helle Wohnung mit Balkon und Lift."
    result = estimator.estimate(_apartment(text=text), LOCATION, asking_rent=5000.0)
    p = result.prediction
    assert p["lo"] < p["q25"] <= p["q50"] <= p["q75"] < p["hi"]
    assert result.check.verdict == "above"
    assert len(result.drivers) == 3 and result.drivers["label"].notna().all()
    assert result.attributes.has_lift and result.attributes.has_balcony_or_terrace
    assert estimator.estimate(_apartment(), LOCATION).check is None


def test_geneva_is_more_expensive(rent_model, map_bundle) -> None:
    estimator = RentEstimator(rent_model, map_bundle)
    zurich = estimator.estimate(_apartment(), LOCATION).prediction["expected"]
    geneva = estimator.estimate(_apartment(east=E0 + 4500), LOCATION).prediction["expected"]
    assert geneva > zurich


def test_estimate_rejects_invalid_input(rent_model, map_bundle) -> None:
    estimator = RentEstimator(rent_model, map_bundle)
    with pytest.raises(EstimateError, match="Wohnfläche"):
        estimator.estimate(_apartment(area=5.0), LOCATION)
    with pytest.raises(EstimateError, match="Gemeinde"):
        estimator.estimate(_apartment(east=E0 + 80_000), LOCATION)
    with pytest.raises(EstimateError, match="positive"):
        estimator.estimate(_apartment(), LOCATION, asking_rent=-1.0)


def test_what_if_varies_area_and_rooms(rent_model, map_bundle) -> None:
    table = RentEstimator(rent_model, map_bundle).what_if(_apartment(area=495.0), LOCATION)
    assert table["feature"].iloc[0] == "baseline"
    assert table.loc[table["feature"].eq("area"), "value"].tolist() == [485.0]  # 505+ skipped
    assert table.loc[table["feature"].eq("rooms"), "value"].tolist() == [2.5, 4.5]


def test_estimator_requires_encoder(rent_model, map_bundle) -> None:
    rent_model.metadata.pop("te_encoder")
    with pytest.raises(ValueError, match="te_encoder"):
        RentEstimator(rent_model, map_bundle)
