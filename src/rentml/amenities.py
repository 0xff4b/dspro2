"""Amenity data and Walk-Score-style accessibility for the QoLI.

OpenStreetMap amenities via the Overpass API (per-category parquet cache, LV95 coordinates), a
smooth distance decay and KD-tree accessibility scores (0-100) per category and in total.
Re-exported by ``rentml.qoli``.
"""

import hashlib
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# (south, west, north, east) in WGS84 (Overpass order): Switzerland plus a ~4 km border margin.
SWISS_BBOX_WGS84: tuple[float, float, float, float] = (45.77, 5.90, 47.85, 10.55)
# Overpass tag filters per amenity category (OSM data, ODbL licence).
OSM_CATEGORIES: dict[str, str] = {
    "supermarket": '["shop"="supermarket"]',
    "pharmacy": '["amenity"="pharmacy"]',
    "doctors": '["amenity"~"^(doctors|clinic)$"]',
    "school": '["amenity"="school"]',
    "kindergarten": '["amenity"~"^(kindergarten|childcare)$"]',
    "gym": '["leisure"="fitness_centre"]',
    "restaurant": '["amenity"~"^(restaurant|cafe)$"]',
    "park": '["leisure"="park"]',
}
# Walk-Score-inspired weights of the n nearest amenities; categories not listed here get [1.0].
DEFAULT_ACCESS_WEIGHTS: dict[str, list[float]] = {
    "supermarket": [3.0],
    "restaurant": [0.75, 0.45, 0.25, 0.25, 0.225, 0.225, 0.225, 0.225, 0.2, 0.2],
}
DECAY_FULL_UNTIL_M = 400.0  # ~5 min walk: full points
DECAY_ZERO_AT_M = 2400.0  # ~30 min walk: no points
OSM_COLUMNS = ["category", "osm_type", "osm_id", "name", "lon", "lat", "east", "north"]
_HEADERS = {"User-Agent": "rentml/0.1 (HSLU DSPRO2 student project)"}
_RETRY_STATUS = frozenset({429, 502, 503, 504})


def distance_decay(
    dist_m: np.ndarray, *, full_until: float = DECAY_FULL_UNTIL_M, zero_at: float = DECAY_ZERO_AT_M
) -> np.ndarray:
    """Walk-Score-like smooth distance decay: 1 up to ``full_until``, cosine taper to 0.

    Args:
        dist_m: Distances in metres (any shape); ``inf`` gives 0, NaN stays NaN.
        full_until: Distance up to which an amenity counts fully (default ~5 min walk).
        zero_at: Distance from which an amenity no longer counts (default ~30 min walk).

    Returns:
        Monotone non-increasing, continuously differentiable factors in [0, 1].

    Raises:
        ValueError: For negative distances or unless ``0 <= full_until < zero_at``.
    """
    if not 0.0 <= full_until < zero_at:
        raise ValueError(f"Need 0 <= full_until < zero_at, got {full_until}, {zero_at}")
    d = np.asarray(dist_m, dtype=float)
    if np.any(d < 0):
        raise ValueError("Distances must be non-negative")
    t = np.clip((d - full_until) / (zero_at - full_until), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * t))


def accessibility(
    points: np.ndarray | pd.DataFrame,
    amenities: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, Sequence[float]] | None = None,
    decay_kwargs: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Walk-Score-style accessibility per amenity category and in total (0-100, LV95 metres).

    Per category, the i-th of the n = len(weights) nearest amenities (KD-tree) adds ``weights[i] ·
    decay(distance)``, scaled by ``sum(weights)``; ``acc_total`` weights categories by weight sum.

    Args:
        points: Array or DataFrame (n, 2) of east/north; rows with NaN/<NA> get NaN scores.
        amenities: Category -> array (m, 2) of east/north, e.g. ``amenity_arrays(osm)`` (keeps
            categories without amenities, so they still count in ``acc_total``).
        weights: Category -> weights; ``None`` = ``DEFAULT_ACCESS_WEIGHTS`` (else ``[1.0]``).
        decay_kwargs: Keyword arguments for ``distance_decay``.

    Returns:
        DataFrame with ``acc_<category>`` columns and ``acc_total``; index of ``points`` if it
        is a DataFrame (e.g. listing_id), else a RangeIndex.

    Raises:
        KeyError: If ``weights`` is given but lacks a category of ``amenities``.
        ValueError: For wrong array shapes, invalid weights or no categories.
    """
    xy = _as_xy(points)
    decay = dict(decay_kwargs or {})
    if weights is not None and (missing := set(amenities) - set(weights)):
        raise KeyError(f"No accessibility weights for categories {sorted(missing)}")
    source = weights if weights is not None else DEFAULT_ACCESS_WEIGHTS
    wmap = {cat: np.asarray(source.get(cat, [1.0]), dtype=float) for cat in amenities}
    if not wmap or any(w.ndim != 1 or np.any(w < 0) or w.sum() <= 0 for w in wmap.values()):
        raise ValueError(f"Need >= 1 category and non-negative 1-D weights, got {wmap}")
    valid = np.isfinite(xy).all(axis=1)
    out: dict[str, np.ndarray] = {}
    for cat, w in wmap.items():
        raw = np.full(len(xy), np.nan)
        raw[valid] = _raw_score(xy[valid], _as_xy(amenities[cat]), w, decay)
        out[f"acc_{cat}"] = 100.0 * raw / w.sum()
    total = sum(out[f"acc_{cat}"] * w.sum() for cat, w in wmap.items())
    out["acc_total"] = total / sum(w.sum() for w in wmap.values())
    return pd.DataFrame(out, index=points.index if isinstance(points, pd.DataFrame) else None)


def amenity_arrays(
    osm: pd.DataFrame, categories: Sequence[str] | Mapping[str, str] = tuple(OSM_CATEGORIES)
) -> dict[str, np.ndarray]:
    """Split an amenity table into per-category coordinate arrays for ``accessibility``.

    Args:
        osm: Table with ``category``, ``east`` and ``north`` (e.g. ``fetch_osm_amenities``).
        categories: Categories to include (keys of a mapping); a category without rows gets an
            empty (0, 2) array instead of silently disappearing from ``acc_total``.

    Returns:
        Category -> float array (m, 2) of LV95 east/north.

    Raises:
        KeyError: If a required column is missing.
    """
    if missing := {"category", "east", "north"} - set(osm.columns):
        raise KeyError(f"Amenity table lacks columns {sorted(missing)}")
    xy = osm[["east", "north"]].to_numpy(dtype=float, na_value=np.nan)
    return {c: xy[(osm["category"] == c).to_numpy()] for c in categories}


def fetch_osm_amenities(
    cache_path: Path,
    *,
    bbox: tuple[float, float, float, float] = SWISS_BBOX_WGS84,
    timeout: int = 300,
    overwrite: bool = False,
    categories: Mapping[str, str] | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download OSM amenities via the Overpass API (one POST per category), cached as parquet.

    Nodes, ways and relations (``out tags center``); 2 s pause between queries; HTTP 429/5xx,
    connection errors and timeouts are retried with backoff (30 s, 60 s, ...); a partial result
    (HTTP 200 with a runtime-error ``remark``) raises. The cache is kept per category in
    ``<stem>_parts/<category>_<hash of tag filter and bbox>.parquet``: an interrupted download
    resumes and another bbox or filter never reuses it. Coordinates: EPSG:4326 -> EPSG:2056.

    Args:
        cache_path: Parquet file that receives the combined table (rebuilt from the parts).
        bbox: (south, west, north, east) in WGS84 degrees.
        timeout: Overpass server timeout in seconds (the HTTP timeout adds 60 s).
        overwrite: Query again even if cached category files exist.
        categories: Category -> Overpass tag filter; default ``OSM_CATEGORIES``.
        max_retries: Retries per request for transient errors.

    Returns:
        DataFrame with ``OSM_COLUMNS`` (category, osm_type, osm_id, name, lon, lat, east, north).

    Raises:
        ValueError: For an invalid bounding box.
        RuntimeError: If Overpass reports a runtime error (e.g. timeout, out of memory).
        requests.RequestException: For HTTP errors and connection failures (after retries).
    """
    if not (-90 <= bbox[0] < bbox[2] <= 90 and -180 <= bbox[1] < bbox[3] <= 180):
        raise ValueError(f"Invalid bbox (south, west, north, east): {bbox}")
    box, parts, queried = ",".join(str(v) for v in bbox), [], False
    for cat, tags in (categories or OSM_CATEGORIES).items():
        selector = f"nwr{tags}({box})"
        digest = hashlib.sha256(selector.encode()).hexdigest()[:12]  # other query -> other file
        part_path = cache_path.with_name(f"{cache_path.stem}_parts") / f"{cat}_{digest}.parquet"
        if part_path.is_file() and not overwrite:
            logger.info("OSM %s: loading cached %s", cat, part_path)
            parts.append(pd.read_parquet(part_path))
            continue
        if queried:
            time.sleep(2.0)  # Overpass fair use: no back-to-back heavy queries
        query = f"[out:json][timeout:{int(timeout)}];{selector};out tags center;"
        part = _osm_frame(cat, _overpass_elements(query, http_s=timeout + 60, retries=max_retries))
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(part_path, index=False)
        parts.append(part)
        queried = True
    osm = pd.concat(parts, ignore_index=True)
    osm.to_parquet(cache_path, index=False)
    return osm


def _as_xy(arr: np.ndarray | pd.DataFrame) -> np.ndarray:
    try:
        xy = np.asarray(arr, dtype=float)
    except TypeError:  # nullable pandas dtypes carry pd.NA, which float() rejects
        raw = np.asarray(arr, dtype=object)
        xy = np.where(pd.isna(raw), np.nan, raw).astype(float)
    if xy.size and (xy.ndim != 2 or xy.shape[1] != 2):
        raise ValueError(f"Coordinates must have shape (n, 2) (LV95 east/north), got {xy.shape}")
    return xy.reshape(-1, 2)


def _raw_score(
    xy: np.ndarray, amen: np.ndarray, w: np.ndarray, decay: dict[str, float]
) -> np.ndarray:
    amen = amen[np.isfinite(amen).all(axis=1)]
    if len(amen) == 0 or len(xy) == 0:
        return np.zeros(len(xy))
    # Neighbours beyond the decay radius come back with distance inf, i.e. weight 0.
    upper = float(decay.get("zero_at", DECAY_ZERO_AT_M))
    dist, _ = cKDTree(amen).query(xy, k=len(w), distance_upper_bound=upper)
    return distance_decay(np.reshape(dist, (len(xy), len(w))), **decay) @ w


# Overpass JSON elements are untyped (library boundary), hence dict[str, Any].
def _osm_frame(cat: str, elements: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for el in elements:  # nodes carry lat/lon, ways and relations a "center"
        pos, name = el.get("center", el), el.get("tags", {}).get("name")
        rows.append((cat, el.get("type"), el.get("id"), name, pos.get("lon"), pos.get("lat")))
    osm = pd.DataFrame(rows, columns=OSM_COLUMNS[:6]).dropna(subset=["osm_id", "lon", "lat"])
    osm = osm.drop_duplicates(["osm_type", "osm_id"]).reset_index(drop=True)
    osm = osm.astype({"osm_id": "int64", "lon": float, "lat": float, "name": "string"})
    lv95 = gpd.GeoSeries(gpd.points_from_xy(osm["lon"], osm["lat"]), crs=4326).to_crs(2056)
    osm["east"], osm["north"] = lv95.x.to_numpy(), lv95.y.to_numpy()
    logger.info("OSM %s: %d amenities", cat, len(osm))
    return osm


def _overpass_elements(query: str, *, http_s: int, retries: int) -> list[dict[str, Any]]:
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                OVERPASS_URL, data={"data": query}, headers=_HEADERS, timeout=http_s
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == retries:
                raise
            logger.warning("Overpass %s; retry %d/%d", type(exc).__name__, attempt + 1, retries)
        else:
            if resp.status_code not in _RETRY_STATUS or attempt == retries:
                break
            logger.warning("Overpass HTTP %d; retry %d/%d", resp.status_code, attempt + 1, retries)
        time.sleep(30.0 * 2**attempt)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in str(payload.get("remark", "")).lower():
        raise RuntimeError(f"Overpass returned an incomplete result: {payload['remark']}")
    return payload.get("elements", [])
