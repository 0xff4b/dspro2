"""Synthetic demo listings for the marketplace (``uv run python -m rentml.market.seed``).

No scraped listing is used or redistributed: addresses, texts and building attributes are
invented, placed at a random point inside large municipalities of the map bundle, and every
listing is flagged as a demo. The asking rent is the model estimate times a random factor, so the
demo shows all three verdicts. Runs offline (no geo.admin lookup).
"""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import shapely

from rentml.config import RANDOM_STATE, ProjectPaths
from rentml.estimate import RentEstimator
from rentml.geoadmin import Address, LocationData
from rentml.mapbundle import MapBundle, default_bundle_dir
from rentml.mapdata import LV95_ORIGIN, PLOT_SCALE_M
from rentml.market.models import Listing, ListingDraft
from rentml.market.service import Market, default_market
from rentml.market.store import MarketStore

logger = logging.getLogger(__name__)

MODEL_FILE = "rent_bundle_v1.joblib"
N_MUNICIPALITIES = 24
RENT_NOISE_SIGMA = 0.13
STREETS = ("Musterweg", "Beispielstrasse", "Demogasse", "Probeweg", "Testplatz")
ROOMS = (1.5, 2.5, 3.5, 3.5, 4.5, 4.5, 5.5)
ADJECTIVES = ("Helle", "Ruhige", "Moderne", "Charmante", "Zentrale", "Sonnige")
FEATURES = (
    "mit Balkon",
    "mit Lift im Haus",
    "mit Seesicht",
    "mit eigener Waschmaschine",
    "mit Garten",
    "mit Einstellhallenplatz",
    "im Minergie-Standard",
    "Haustiere erlaubt",
)
DEMO_CONTACT = ("RentLens Demo", "demo@example.org")


def _random_point(geometry: shapely.Geometry, rng: np.random.Generator) -> tuple[float, float]:
    minx, miny, maxx, maxy = geometry.bounds
    for _ in range(50):
        point = shapely.Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if geometry.contains(point):
            break
    else:
        point = geometry.representative_point()
    return (
        LV95_ORIGIN[0] + point.x * PLOT_SCALE_M,
        LV95_ORIGIN[1] + point.y * PLOT_SCALE_M,
    )


def _location(rng: np.random.Generator) -> LocationData:
    return LocationData(
        year_built=float(rng.integers(1930, 2023)),
        apartments=float(rng.integers(3, 30)),
        land_area=float(rng.integers(150, 900)),
        missing=["oev", "solar", "population", "elevation"],
    )


def _draft(name: str, point: tuple[float, float], rng: np.random.Generator) -> ListingDraft:
    rooms = float(rng.choice(ROOMS))
    area = float(round(rooms * rng.uniform(20, 27) + rng.uniform(-4, 8)))
    features = rng.choice(FEATURES, size=int(rng.integers(1, 4)), replace=False)
    renovated = f" Renoviert {int(rng.integers(2012, 2026))}." if rng.random() < 0.4 else ""
    street = f"{rng.choice(STREETS)} {int(rng.integers(1, 60))}"
    return ListingDraft(
        title=f"{rng.choice(ADJECTIVES)} {rooms:g}-Zimmer-Wohnung in {name}"[:120],
        description=f"Demo-Inserat: Wohnung {', '.join(features)}.{renovated}",
        address=Address(f"{street}, {name} (Demo-Adresse)", point[0], point[1], None),
        area=min(max(area, 20.0), 220.0),
        rooms=rooms,
        rent_net=1000.0,  # replaced by the model-based rent below
        aux_costs=float(10 * rng.integers(12, 34)),
        floor=int(rng.integers(0, 7)),
        available_from=date.today() + timedelta(days=int(rng.integers(10, 120))),
        contact_name=DEMO_CONTACT[0],
        contact_email=DEMO_CONTACT[1],
    )


def seed_demo(
    market: Market, bundle: MapBundle, n: int = 36, *, seed: int = RANDOM_STATE
) -> list[Listing]:
    """Publish ``n`` synthetic demo listings in the municipalities with the most training data.

    Args:
        market: Marketplace with a rent model (the asking rents derive from its estimates).
        bundle: Map bundle (municipality polygons and object counts).
        n: Number of listings.
        seed: Random seed.

    Returns:
        The stored listings.

    Raises:
        ValueError: If the marketplace has no rent model.
    """
    if market.estimator is None:
        raise ValueError("Demo listings need the rent model (models/rent_bundle_v1.joblib)")
    rng = np.random.default_rng(seed)
    units = bundle.units.loc[bundle.units["level"].eq("municipality")]
    top = units.nlargest(N_MUNICIPALITIES, "n_objects").set_index("unit_id")
    shapes = {
        int(f["id"]): shapely.geometry.shape(f["geometry"])
        for f in bundle.geojson["municipality"]["features"]
        if int(f["id"]) in top.index
    }
    listings = []
    for _ in range(n):
        unit_id = int(rng.choice(top.index))
        location = _location(rng)
        draft = _draft(str(top.at[unit_id, "name"]), _random_point(shapes[unit_id], rng), rng)
        expected = market.assess_rent(draft, location)
        rent = float(10 * round(expected * np.exp(rng.normal(0.0, RENT_NOISE_SIGMA)) / 10))
        priced = replace(draft, rent_net=max(rent, 400.0))
        listing, _ = market.publish(priced, location=location, is_demo=True)
        listings.append(listing)
    logger.info("Seeded %d demo listings", len(listings))
    return listings


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: fill the marketplace database with demo listings."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=36, help="number of demo listings")
    parser.add_argument("--db", default=None, help="SQLAlchemy URL (default: data/app)")
    parser.add_argument("--model", type=Path, default=None, help="model bundle (.joblib)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    paths = ProjectPaths.discover()
    bundle = MapBundle.load(default_bundle_dir(paths))
    estimator = RentEstimator.load(args.model or paths.models / MODEL_FILE, bundle)
    store = MarketStore(args.db) if args.db else None
    market = default_market(bundle, estimator, lambda address: LocationData(), store=store)
    seed_demo(market, bundle, args.n)
    logger.info("Marketplace now has %d active listings", market.store.stats()["active"])


if __name__ == "__main__":
    main()
