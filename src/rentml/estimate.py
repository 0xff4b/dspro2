"""Single-apartment estimates for the app: features, Fair-Rent Check, price band and what-if.

:class:`RentEstimator` turns one geocoded address plus area and rooms into the feature row of the
deployed :class:`rentml.rentcheck.RentModelBundle`, exactly as the training pipeline built it:

* municipality, district, canton and language region from the map bundle's boundaries
  (point in polygon on the LV95 geometry), municipal population and land density;
* nested random-effect keys and their out-of-fold target encodings (``metadata["te_encoder"]``);
* building and location inputs from geo.admin (:mod:`rentml.geoadmin`);
* the engineered features of :func:`rentml.features.add_engineered_features`.

Inputs are never stored or logged (proposal: "inputs are not stored").
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from rentml import geo
from rentml.explain import top_drivers, tree_shap
from rentml.extraction import ExtractedAttributes, rule_based_extract
from rentml.features import add_engineered_features
from rentml.geoadmin import Address, LocationData
from rentml.mapbundle import MapBundle
from rentml.mapdata import LV95_ORIGIN, PLOT_SCALE_M
from rentml.rentcheck import RentCheckResult, RentModelBundle, fair_rent_check, predict_frame
from rentml.rentcheck import what_if as bundle_what_if

logger = logging.getLogger(__name__)

AREA_RANGE: tuple[float, float] = (10.0, 500.0)  # domain rules of the training data
ROOMS_RANGE: tuple[float, float] = (1.0, 15.0)
AREA_STEPS: tuple[float, ...] = (-10.0, 10.0, 20.0)
ROOM_STEPS: tuple[float, ...] = (-1.0, 1.0)
N_DRIVERS = 3


class EstimateError(ValueError):
    """Raised when an apartment cannot be estimated (invalid input or outside Switzerland)."""


@dataclass(frozen=True)
class Apartment:
    """User input for one apartment.

    Attributes:
        address: Geocoded address.
        area: Living area in m².
        rooms: Number of rooms (half rooms allowed).
        description: Optional listing text (attributes are extracted, the text is not kept).
    """

    address: Address
    area: float
    rooms: float
    description: str | None = None

    def validate(self) -> None:
        """Check the input against the training domain.

        Raises:
            EstimateError: If area or rooms are outside the trained range.
        """
        if not AREA_RANGE[0] <= self.area <= AREA_RANGE[1]:
            raise EstimateError(
                f"Wohnfläche muss zwischen {AREA_RANGE[0]:.0f} und {AREA_RANGE[1]:.0f} m² liegen."
            )
        if not ROOMS_RANGE[0] <= self.rooms <= ROOMS_RANGE[1]:
            raise EstimateError(
                f"Zimmerzahl muss zwischen {ROOMS_RANGE[0]:.0f} und {ROOMS_RANGE[1]:.0f} liegen."
            )


@dataclass
class Estimate:
    """Everything the tenant and landlord pages show for one apartment.

    Attributes:
        unit: Municipality row of the map bundle (name, district, canton, language region).
        location: Building and location inputs, including the missing ones.
        prediction: One row of :func:`rentml.rentcheck.predict_frame` (CHF).
        drivers: Top SHAP drivers with German labels.
        attributes: Attributes recognised in the description (rule-based).
        check: Fair-Rent Check result, if an asking rent was given.
        coverage: Nominal coverage of the calibrated interval.
    """

    unit: pd.Series
    location: LocationData
    prediction: pd.Series
    drivers: pd.DataFrame
    attributes: ExtractedAttributes
    check: RentCheckResult | None = None
    coverage: float = 0.8


DRIVER_LABELS: dict[str, str] = {
    "te_re_canton": "Kanton (Mietniveau)",
    "te_re_district": "Bezirk (Mietniveau)",
    "te_re_municipality": "Gemeinde (Mietniveau)",
    "area": "Wohnfläche",
    "rooms": "Zimmerzahl",
    "area_per_room": "Fläche pro Zimmer",
    "year_built": "Baujahr",
    "building_age": "Gebäudealter",
    "apartments": "Wohnungen im Gebäude",
    "land_area": "Gebäudegrundfläche",
    "land_area_per_apartment": "Grundfläche pro Wohnung",
    "population": "Einwohner in der Hektare",
    "oev": "ÖV-Erreichbarkeit",
    "solar": "Solareignung Dach",
    "elevation": "Höhe ü. M.",
    "muni_density": "Bevölkerungsdichte Gemeinde",
    "muni_population": "Einwohner Gemeinde",
    "east": "Lage Ost-West",
    "north": "Lage Nord-Süd",
    "rot45_x": "Lage (diagonal)",
    "rot45_y": "Lage (diagonal)",
}


class MunicipalityLocator:
    """Point-in-polygon lookup of the municipality on the map bundle's LV95 geometry."""

    def __init__(self, bundle: MapBundle) -> None:
        features = bundle.geojson["municipality"]["features"]
        self._ids = np.array([int(f["id"]) for f in features])
        self._geoms = np.array([shapely.geometry.shape(f["geometry"]) for f in features])
        self._tree = shapely.STRtree(self._geoms)
        units = bundle.units
        self._units = units.loc[units["level"].eq("municipality")].set_index("unit_id")

    def locate(self, east: float, north: float, *, max_snap_m: float = 500.0) -> pd.Series | None:
        """Municipality row for an LV95 point (nearest within ``max_snap_m`` on borders/lakes)."""
        point = shapely.Point(
            (east - LV95_ORIGIN[0]) / PLOT_SCALE_M, (north - LV95_ORIGIN[1]) / PLOT_SCALE_M
        )
        hits = self._tree.query(point, predicate="within")
        if not len(hits):
            nearest = self._tree.query_nearest(point, max_distance=max_snap_m / PLOT_SCALE_M)
            hits = nearest[:1]
        if not len(hits):
            return None
        return self._units.loc[self._ids[int(hits[0])]]


class RentEstimator:
    """Serve estimates for single apartments from a model bundle and the map bundle."""

    def __init__(self, model: RentModelBundle, map_bundle: MapBundle) -> None:
        self.model = model
        self.locator = MunicipalityLocator(map_bundle)
        self.encoder = model.metadata.get("te_encoder")
        if self.encoder is None:
            raise ValueError("The model bundle has no 'te_encoder' in its metadata")

    @classmethod
    def load(cls, model_path: Path, map_bundle: MapBundle) -> "RentEstimator":
        """Load the model bundle (pickle: trusted files only) and build the estimator."""
        return cls(RentModelBundle.load(Path(model_path)), map_bundle)

    def features(
        self, apartment: Apartment, location: LocationData
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Feature row, calibration group row and municipality for one apartment.

        Raises:
            EstimateError: For invalid inputs or an address outside Swiss municipalities.
        """
        apartment.validate()
        address = apartment.address
        unit = self.locator.locate(address.east, address.north)
        if unit is None:
            raise EstimateError("Die Adresse liegt in keiner Schweizer Gemeinde.")
        raw = pd.DataFrame(
            [
                {
                    "area": float(apartment.area),
                    "rooms": float(apartment.rooms),
                    "east": address.east,
                    "north": address.north,
                    "canton": unit["canton"],
                    "district_id": int(unit["district_id"]),
                    "municipality_id": int(unit.name),
                    "muni_density": unit["density"],
                    "muni_population": unit["population"],
                    **location.to_features(),
                }
            ]
        )
        raw = raw.join(geo.nested_group_keys(raw))
        encoded = self.encoder.transform(raw[list(self.encoder.cols)])
        x_row = add_engineered_features(pd.concat([raw, encoded], axis=1)).astype(
            dict.fromkeys(self.model.feature_cols, float)
        )
        groups = pd.DataFrame({"lang_region": [unit["lang_region"]]})
        return x_row, groups[self.model.group_cols], unit

    def drivers(self, x_row: pd.DataFrame, k: int = N_DRIVERS) -> pd.DataFrame:
        """Top SHAP drivers of the point estimate with German labels."""
        explanation = tree_shap(self.model.point_model, x_row[self.model.feature_cols])
        table = top_drivers(explanation, 0, k=k)
        table["label"] = table["feature"].map(DRIVER_LABELS).fillna(table["feature"])
        return table

    def estimate(
        self, apartment: Apartment, location: LocationData, asking_rent: float | None = None
    ) -> Estimate:
        """Estimate, interval, listing band, drivers, text attributes and optional verdict.

        Args:
            apartment: User input.
            location: Inputs from :func:`rentml.geoadmin.lookup_location`.
            asking_rent: Asking rent in CHF for the Fair-Rent Check (``None`` = no verdict).

        Returns:
            The :class:`Estimate`.

        Raises:
            EstimateError: For invalid inputs.
        """
        x_row, groups, unit = self.features(apartment, location)
        prediction = predict_frame(self.model, x_row, groups).iloc[0]
        drivers = self.drivers(x_row)
        check = None
        if asking_rent is not None:
            if not (math.isfinite(asking_rent) and asking_rent > 0):
                raise EstimateError("Die Miete muss eine positive Zahl sein.")
            check = fair_rent_check(self.model, x_row, groups, float(asking_rent), drivers=drivers)
        coverage = check.coverage if check else float(self.model.metadata.get("coverage", 0.8))
        return Estimate(
            unit=unit,
            location=location,
            prediction=prediction,
            drivers=drivers,
            attributes=rule_based_extract(apartment.description),
            check=check,
            coverage=coverage,
        )

    def what_if(self, apartment: Apartment, location: LocationData) -> pd.DataFrame:
        """Effect of a larger/smaller area and one room more/less (one change at a time).

        Area steps outside the trained range are skipped. The models are monotone in the living
        area, so a larger flat never gets a lower estimate.
        """
        x_row, groups, _ = self.features(apartment, location)
        changes: dict[str, list[float]] = {}
        areas = [apartment.area + s for s in AREA_STEPS]
        changes["area"] = [a for a in areas if AREA_RANGE[0] <= a <= AREA_RANGE[1]]
        rooms = [apartment.rooms + s for s in ROOM_STEPS]
        changes["rooms"] = [r for r in rooms if ROOMS_RANGE[0] <= r <= ROOMS_RANGE[1]]
        changes = {k: v for k, v in changes.items() if v}
        return bundle_what_if(self.model, x_row, groups, changes, transform=add_engineered_features)
