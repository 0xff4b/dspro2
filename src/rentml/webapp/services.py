"""External services of the app, bundled so tests can replace the network and the model."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from rentml import geoadmin
from rentml.estimate import RentEstimator
from rentml.geoadmin import Address, LocationData
from rentml.market.service import Market

LOCATION_CACHE_SIZE = 256


@dataclass(frozen=True)
class Services:
    """What the pages need.

    Attributes:
        estimator: Model wrapper; ``None`` if no model bundle is deployed.
        search: Address suggestions for a free-text query.
        lookup: Building and location inputs for an address (public geodata only).
        market: Marketplace service; ``None`` = the app builds the default one.
    """

    estimator: RentEstimator | None
    search: Callable[[str], list[Address]]
    lookup: Callable[[Address], LocationData]
    market: Market | None = None


def live_services(estimator: RentEstimator | None) -> Services:
    """Services backed by geo.admin; location lookups of public geodata are cached per address."""
    cached = lru_cache(maxsize=LOCATION_CACHE_SIZE)(geoadmin.lookup_location)
    return Services(estimator=estimator, search=geoadmin.search_addresses, lookup=cached)
