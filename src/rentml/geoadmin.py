"""Live lookups on the federal geodata API (api3.geo.admin.ch) for a single address.

Ported from the DSPRO1 Streamlit app (``src/app.py``) so that the app collects the same inputs
the models were trained on: address search (SearchServer), GWR building attributes (MapServer
find by EGID), public-transport quality and roof solar class (MapServer identify), hectare
population (STATPOP layer) and elevation (height service).

Every function takes an optional ``requests.Session`` so tests can replace the network. Failed
calls never raise for a single missing value: the model handles NaN, and the app shows which
inputs were not available.
"""

import logging
import re
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api3.geo.admin.ch"
SEARCH_URL = f"{API_BASE}/rest/services/api/SearchServer"
FIND_URL = f"{API_BASE}/rest/services/api/MapServer/find"
IDENTIFY_URL = f"{API_BASE}/rest/services/api/MapServer/identify"
HEIGHT_URL = f"{API_BASE}/rest/services/height"
GWR_LAYER = "ch.bfs.gebaeude_wohnungs_register"
OEV_LAYER = "ch.are.erreichbarkeit-oev"
SOLAR_LAYER = "ch.bfe.solarenergie-eignung-daecher"
POPULATION_LAYER = "ch.bfs.volkszaehlung-bevoelkerungsstatistik_einwohner"
TIMEOUT_S = 12.0
MIN_QUERY_CHARS = 3
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Address:
    """A geocoded Swiss address.

    Attributes:
        label: Display label without HTML tags, e.g. ``"Bahnhofstrasse 1 8001 Zürich"``.
        east: LV95 east coordinate (m).
        north: LV95 north coordinate (m).
        egid: Federal building identifier, if the address belongs to a GWR building.
    """

    label: str
    east: float
    north: float
    egid: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Plain dict for a ``dcc.Store``."""
        return {"label": self.label, "east": self.east, "north": self.north, "egid": self.egid}

    @classmethod
    def from_dict(cls, data: dict) -> "Address":
        """Inverse of :meth:`to_dict`."""
        egid = data.get("egid")
        return cls(
            str(data["label"]),
            float(data["east"]),
            float(data["north"]),
            int(egid) if egid is not None else None,
        )


@dataclass
class LocationData:
    """Building and location inputs of the rent model for one address (``None`` = unknown).

    Attributes:
        year_built: GWR construction year (``gbauj``).
        apartments: Dwellings in the building (``ganzwhg``).
        land_area: GWR building footprint area in m² (``garea``).
        oev: ARE public-transport accessibility score.
        solar: BFE roof suitability class.
        population: Inhabitants of the hectare cell.
        elevation: Height above sea level (m).
        missing: Names of the inputs that could not be retrieved.
    """

    year_built: float | None = None
    apartments: float | None = None
    land_area: float | None = None
    oev: float | None = None
    solar: float | None = None
    population: float | None = None
    elevation: float | None = None
    missing: list[str] = field(default_factory=list)

    def to_features(self) -> dict[str, float | None]:
        """Model columns (as in :data:`rentml.features.BUILDING` / ``LOCATION``)."""
        return {
            "year_built": self.year_built,
            "apartments": self.apartments,
            "land_area": self.land_area,
            "oev": self.oev,
            "solar": self.solar,
            "population": self.population,
            "elevation": self.elevation,
        }


def _get_json(url: str, params: dict[str, object], session: requests.Session | None) -> dict | None:
    getter = session or requests
    try:
        response = getter.get(url, params=params, timeout=TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as exc:  # ValueError: invalid JSON
        logger.warning("geo.admin request to %s failed: %s", url, exc)
        return None


def _number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]  # API values are str/int/float/None
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _egid(feature_id: object) -> int | None:
    head = str(feature_id or "").split("_")[0]
    return int(head) if head.isdigit() else None


def search_addresses(
    query: str, *, limit: int = 8, session: requests.Session | None = None
) -> list[Address]:
    """Address suggestions for a free-text query (empty below ``MIN_QUERY_CHARS``).

    Args:
        query: Free text, e.g. ``"Bahnhofstr 1 Luzern"``.
        limit: Maximum number of suggestions.
        session: Optional HTTP session.

    Returns:
        Unique addresses with LV95 coordinates (the SearchServer returns y = east, x = north).
    """
    text = " ".join(str(query or "").split())
    if len(text) < MIN_QUERY_CHARS:
        return []
    params = {
        "searchText": text,
        "type": "locations",
        "origins": "address",
        "sr": 2056,
        "limit": int(limit),
    }
    data = _get_json(SEARCH_URL, params, session) or {}
    found: dict[str, Address] = {}
    for item in data.get("results") or []:
        attrs = item.get("attrs") or {}
        label = _TAG_RE.sub("", str(attrs.get("label") or "")).strip()
        east, north = _number(attrs.get("y")), _number(attrs.get("x"))
        if label and east is not None and north is not None and label not in found:
            egid = _egid(attrs.get("featureId") or attrs.get("feature_id"))
            found[label] = Address(label, east, north, egid)
    return list(found.values())


def building_attributes(
    egid: int | None, *, session: requests.Session | None = None
) -> dict[str, float | None]:
    """GWR construction year, number of dwellings and footprint area of a building."""
    empty: dict[str, float | None] = {"year_built": None, "apartments": None, "land_area": None}
    if egid is None:
        return empty
    # contains=false: the default substring search returns other buildings (213022 -> 502130226)
    params = {
        "layer": GWR_LAYER,
        "searchText": str(egid),
        "searchField": "egid",
        "returnGeometry": "false",
        "contains": "false",
    }
    results = (_get_json(FIND_URL, params, session) or {}).get("results") or []
    matches = [r.get("attributes") or {} for r in results]
    matches = [a for a in matches if str(a.get("egid")) == str(egid)]
    if not matches:
        return empty
    attrs = {str(k).lower(): v for k, v in matches[0].items()}
    return {
        "year_built": _number(attrs.get("gbauj")),
        "apartments": _number(attrs.get("ganzwhg")),
        "land_area": _number(attrs.get("garea")),
    }


def _identify(
    east: float, north: float, layers: str, session: requests.Session | None
) -> list[dict]:
    params = {
        "geometry": f"{east},{north}",
        "geometryType": "esriGeometryPoint",
        "layers": f"all:{layers}",
        "tolerance": 1,
        "returnGeometry": "false",
        "sr": 2056,
        "imageDisplay": "100,100,96",
        "mapExtent": f"{east - 10},{north - 10},{east + 10},{north + 10}",
    }
    return (_get_json(IDENTIFY_URL, params, session) or {}).get("results") or []


def location_attributes(
    east: float, north: float, *, session: requests.Session | None = None
) -> dict[str, float | None]:
    """Public-transport score, solar class, hectare population and elevation at a point."""
    out: dict[str, float | None] = {
        "oev": None,
        "solar": None,
        "population": None,
        "elevation": None,
    }
    for item in _identify(east, north, f"{OEV_LAYER},{SOLAR_LAYER}", session):
        attrs = item.get("attributes") or {}
        if out["oev"] is None and item.get("layerBodId") == OEV_LAYER:
            out["oev"] = _number(attrs.get("oev_erreichb_ewap"))
        elif out["solar"] is None and item.get("layerBodId") == SOLAR_LAYER:
            out["solar"] = _number(attrs.get("klasse"))
    for item in _identify(east, north, POPULATION_LAYER, session):
        attrs = item.get("attributes") or {}
        if (count := _number(attrs.get("number"))) is not None:
            out["population"] = count
            break
    height = _get_json(HEIGHT_URL, {"easting": east, "northing": north}, session) or {}
    out["elevation"] = _number(height.get("height"))
    return out


def lookup_location(address: Address, *, session: requests.Session | None = None) -> LocationData:
    """All building and location inputs for one address.

    Args:
        address: Geocoded address (from :func:`search_addresses`).
        session: Optional HTTP session.

    Returns:
        The inputs; ``missing`` lists those that stayed unknown.
    """
    values = building_attributes(address.egid, session=session)
    values |= location_attributes(address.east, address.north, session=session)
    data = LocationData(**values)
    data.missing = [name for name, value in values.items() if value is None]
    return data
