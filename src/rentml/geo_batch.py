"""HTTP layer of the batch enrichment: JSONL response cache, retries and parallel fetchers.

Each ``fetch_*`` function returns a JSON-serialisable value, an empty value when the service
has no data, or ``None`` when the request failed; only failures are retried on the next run.
"""

import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from rentml import geo
from rentml.config import CACHE

logger = logging.getLogger(__name__)

CACHE_FILE = CACHE / "geo_enrichment.jsonl"
GWR_LAYER = "ch.bfs.gebaeude_wohnungs_register"
BUILDING_RADIUS_M = 25  # nearest GWR building for a precise point without a house number
WORKERS = 6  # parallel requests; geo.admin and MADD are public services, keep the load moderate
RETRIES = 3
RETRY_STATUS = {429, 500, 502, 503, 504}
_CONSTRUCTION = ".//m:building/m:dateOfConstruction"
NOT_FOUND_STATUS = {400, 404}  # MADD answers an unknown EGID with 400/404: cached as "no data"

Fetcher = Callable[[requests.Session, tuple], object | None]


class ResponseCache:
    """Thread-safe append-only JSONL cache of API responses keyed by ``(kind, key)``."""

    def __init__(self, path: Path = CACHE_FILE) -> None:
        """Load the cached responses of ``path`` (created on the first write).

        Args:
            path: JSONL file with one ``{"kind", "key", "value"}`` object per line.
        """
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], object] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self._data[(row["kind"], row["key"])] = row["value"]

    def __contains__(self, item: tuple[str, str]) -> bool:
        return item in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, kind: str, key: str) -> object | None:
        """Return the cached value or ``None`` if the key was never fetched."""
        return self._data.get((kind, key))

    def put(self, kind: str, key: str, value: object) -> None:
        """Store a value in memory and append it to the file."""
        line = json.dumps({"kind": kind, "key": key, "value": value}, ensure_ascii=False)
        with self._lock:
            self._data[(kind, key)] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _request(session: requests.Session, url: str, params: dict) -> requests.Response | None:
    """GET with retries; ``None`` after repeated failures (never cached)."""
    for attempt in range(RETRIES):
        try:
            response = session.get(url, params=params, timeout=geo.TIMEOUT_S)
        except requests.RequestException as exc:
            logger.debug("Request to %s failed: %s", url, exc)
        else:
            if response.status_code not in RETRY_STATUS:
                return response
        time.sleep(2**attempt)
    logger.warning("Giving up on %s %s", url, params)
    return None


def _json(response: requests.Response | None) -> dict | None:
    if response is None or not response.ok:
        return None
    try:
        return response.json()
    except requests.JSONDecodeError:
        return None


def fetch_address(session: requests.Session, args: tuple) -> dict | None:
    """First hit of the geo.admin address search (``{}`` if none, ``None`` on failure).

    Args:
        session: HTTP session.
        args: ``(query,)``, e.g. ``("Bahnhofstrasse 1, 8001 Zürich",)``.

    Returns:
        ``label``, ``east``, ``north`` and ``egid`` of the first hit.
    """
    params = {"searchText": args[0], "type": "locations", "origins": "address", "sr": geo.LV95}
    data = _json(_request(session, geo.SEARCH_URL, params | {"limit": 1}))
    if data is None:
        return None
    for item in data.get("results", []):
        attrs = item.get("attrs", {})
        egid = str(attrs.get("featureId", "")).split("_")[0]
        return {
            "label": re.sub(r"</?b>", "", attrs.get("label", "")).strip(),
            "east": geo._number(attrs.get("y")),  # y = east in LV95
            "north": geo._number(attrs.get("x")),
            "egid": int(egid) if egid.isdigit() else None,
        }
    return {}


def fetch_buildings(session: requests.Session, args: tuple) -> list | None:
    """GWR buildings within :data:`BUILDING_RADIUS_M` of a point.

    Args:
        session: HTTP session.
        args: ``(east, north)`` in LV95.

    Returns:
        ``[egid, east, north, street]`` per building, ``None`` on failure.
    """
    east, north = args
    half = 2 * BUILDING_RADIUS_M
    params = {
        "geometry": f"{east},{north}", "geometryType": "esriGeometryPoint",
        "layers": f"all:{GWR_LAYER}", "tolerance": BUILDING_RADIUS_M, "returnGeometry": "true",
        "sr": geo.LV95, "imageDisplay": f"{2 * half},{2 * half},96",
        "mapExtent": f"{east - half},{north - half},{east + half},{north + half}",
    }  # fmt: skip  # 1 px = 1 m, so the pixel tolerance equals the radius
    data = _json(_request(session, geo.IDENTIFY_URL, params))
    if data is None:
        return None
    rows = []
    for item in data.get("results", []):
        attrs, point = item.get("attributes", {}), item.get("geometry", {})
        egid = str(attrs.get("egid", ""))
        if egid.isdigit() and "x" in point:
            rows.append([int(egid), point["x"], point["y"], attrs.get("strname_deinr") or ""])
    return rows


def fetch_madd(session: requests.Session, args: tuple) -> dict | None:
    """Building attributes and dwellings of one EGID from MADD.

    Args:
        session: HTTP session.
        args: ``(egid,)``.

    Returns:
        ``year_built`` (exact year only), ``period`` (GWR construction period code),
        ``apartments``, ``land_area`` and ``dwellings`` (``[rooms, area, floor]``); ``{}`` for
        an unknown EGID, ``None`` on failure.
    """
    response = _request(session, geo.MADD_URL, {"egid": args[0]})
    if response is None:
        return None
    if response.status_code in NOT_FOUND_STATUS:
        return {}
    if not response.ok:
        return None
    try:
        building = ET.fromstring(response.content).find(".//m:buildingItem", geo.MADD_NS)
    except ET.ParseError:
        return None
    if building is None:
        return {}
    dwellings = geo._parse_dwellings(building)
    values = geo._building_attributes(building, dwellings)
    # Store the exact year and the period separately, so the enrichment can flag period midpoints.
    date = building.findtext(f"{_CONSTRUCTION}/m:dateOfConstruction", namespaces=geo.MADD_NS)
    period = building.findtext(f"{_CONSTRUCTION}/m:periodOfConstruction", namespaces=geo.MADD_NS)
    values["year_built"] = geo._number((date or "")[:4])
    values["period"] = int(period) if period and period.isdigit() else None
    values["dwellings"] = [[d.rooms, d.area, d.floor] for d in dwellings]
    return values


def fetch_location(session: requests.Session, args: tuple) -> dict | None:
    """ÖV, solar class, hectare population and elevation at a point (``None`` on failure).

    Args:
        session: HTTP session.
        args: ``(east, north)`` in LV95.

    Returns:
        The values of :func:`rentml.geo.parse_location`.
    """
    east, north = args
    identify = _json(_request(session, geo.IDENTIFY_URL, geo.location_params(east, north)))
    height = _json(_request(session, geo.HEIGHT_URL, {"easting": east, "northing": north}))
    if identify is None or height is None:
        return None
    return geo.parse_location(identify, height)


def fetch_all(
    cache: ResponseCache,
    kind: str,
    jobs: dict[str, tuple],
    fetch: Fetcher,
    workers: int = WORKERS,
) -> int:
    """Fetch every job whose key is not cached yet, in parallel.

    Args:
        cache: Response cache.
        kind: Cache namespace (``address``, ``buildings``, ``madd``, ``location``).
        jobs: Cache key -> arguments of ``fetch``.
        fetch: One of the ``fetch_*`` functions.
        workers: Parallel threads, each with its own session.

    Returns:
        Number of failed jobs (not cached, retried on the next call).
    """
    todo = {key: args for key, args in jobs.items() if (kind, key) not in cache}
    if not todo:
        return 0
    logger.info("%s: fetching %d of %d keys", kind, len(todo), len(jobs))
    local = threading.local()

    def run(key: str, args: tuple) -> bool:
        if not hasattr(local, "session"):
            local.session = requests.Session()
        value = fetch(local.session, args)
        if value is None:
            return False
        cache.put(kind, key, value)
        return True

    failed, start = 0, time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, key, args) for key, args in todo.items()]
        for done, future in enumerate(as_completed(futures), start=1):
            failed += not future.result()
            if done % 2000 == 0 or done == len(futures):
                rate = done / (time.monotonic() - start)
                logger.info("%s: %d/%d (%.1f/s, %d failed)", kind, done, len(todo), rate, failed)
    return failed
