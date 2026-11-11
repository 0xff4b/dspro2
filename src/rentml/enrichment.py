"""Batch enrichment of listings with GWR building data and geo.admin location layers.

The same sources as the DSPRO1 enrichment and the app (:func:`rentml.geo.lookup_location`):
the building (EGID) comes from the geo.admin address search or, without a house number, from the
nearest GWR building at a precise coordinate; year built, number of dwellings, building area and
the dwellings themselves from the BFS service MADD (eCH-0206); public-transport quality, roof
solar class, hectare population and elevation from the geo.admin identify and height services.

Every response is stored in a JSONL cache (:data:`CACHE_FILE`), so an interrupted run resumes
where it stopped and the cleaning notebook reruns without network. Failed requests are not
cached. Run ``uv run python -m rentml.enrichment`` to fill the cache for the Comparis listings.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from rentml import comparis
from rentml.geo import PERIOD_MIDPOINT
from rentml.geo_batch import (
    BUILDING_RADIUS_M,
    CACHE_FILE,
    WORKERS,
    ResponseCache,
    fetch_address,
    fetch_all,
    fetch_buildings,
    fetch_location,
    fetch_madd,
)

logger = logging.getLogger(__name__)

MAX_ADDRESS_SHIFT_M = 500.0  # matched address vs. precise listing coordinate
AREA_MAX_REL_SPREAD = 0.05  # GWR candidates must agree within 5 % to fill a missing area
ENRICHED_COLUMNS = [
    "egid", "egid_source", "bldg_east", "bldg_north", "year_built_gwr", "year_built_period",
    "apartments",
    "land_area", "oev", "solar", "population", "elevation", "area_gwr", "area_gwr_candidates",
]  # fmt: skip


def address_queries(df: pd.DataFrame) -> pd.Series:
    """Search text ``"<street> <number>, <zip> <city>"``; NA without street, number or zip."""
    query = df["street"] + " " + df["street_number"] + ", " + df["zip"] + " " + df["city"]
    return query.str.split().str.join(" ").astype("string")


def _house_number(text: pd.Series) -> pd.Series:
    return text.astype("string").str.casefold().str.replace(r"\s+", "", regex=True)


def _street_key(text: pd.Series) -> pd.Series:
    words = text.astype("string").str.casefold().str.replace(r"\d.*$", "", regex=True)
    return words.str.replace(r"\W+", "", regex=True).str.replace("str$", "strasse", regex=True)


def _point_key(east: float, north: float) -> str:
    return f"{round(east)}_{round(north)}"


def match_addresses(df: pd.DataFrame, hits: pd.Series) -> pd.DataFrame:
    """Accept address hits with the same postcode and house number (and nearby, if precise).

    Args:
        df: Listings with ``street_number``, ``zip``, ``east``, ``north``, ``coord_precise``.
        hits: :func:`fetch_address` result per listing (dict or missing).

    Returns:
        ``egid``, ``bldg_east``, ``bldg_north`` for accepted hits (NaN otherwise).
    """
    hit = pd.DataFrame(
        hits.map(lambda h: h if isinstance(h, dict) else {}).tolist(), index=df.index
    )
    hit = hit.reindex(columns=["label", "east", "north", "egid"])
    label = hit["label"].astype("string")
    first = label.str.extract(r"^(.*?)\s+(\d{4})\s", expand=True)
    number = first[0].str.extract(r"(\d+\s*\w?)$", expand=False)
    same = first[1].eq(df["zip"]) & _house_number(number).eq(_house_number(df["street_number"]))
    shift = np.hypot(hit["east"] - df["east"], hit["north"] - df["north"])
    near = ~df["coord_precise"].astype(bool) | (shift <= MAX_ADDRESS_SHIFT_M)
    ok = (same & near).fillna(False) & hit["egid"].notna()
    out = hit[["egid", "east", "north"]].where(ok, axis=0).astype(float)
    return out.set_axis(["egid", "bldg_east", "bldg_north"], axis=1)


def match_buildings(df: pd.DataFrame, candidates: pd.Series) -> pd.DataFrame:
    """Nearest GWR building within the radius, in the listing's street if one is stated.

    Args:
        df: Listings with ``street``, ``east``, ``north``.
        candidates: :func:`fetch_buildings` result per listing (list or missing).

    Returns:
        ``egid``, ``bldg_east``, ``bldg_north`` (NaN without a match).
    """
    rows = []
    street = _street_key(df["street"])
    for pos, cands in enumerate(candidates):
        best = (np.nan, np.nan, np.nan)
        if isinstance(cands, list) and cands:
            c = pd.DataFrame(cands, columns=["egid", "east", "north", "street"])
            c["dist"] = np.hypot(c["east"] - df["east"].iat[pos], c["north"] - df["north"].iat[pos])
            c = c[c["dist"] <= BUILDING_RADIUS_M]
            if pd.notna(street.iat[pos]) and street.iat[pos]:
                c = c[_street_key(c["street"]).eq(street.iat[pos]).to_numpy()]
            if len(c):
                top = c.nsmallest(1, "dist").iloc[0]
                best = (float(top["egid"]), float(top["east"]), float(top["north"]))
        rows.append(best)
    return pd.DataFrame(rows, index=df.index, columns=["egid", "bldg_east", "bldg_north"])


def recover_area(
    rooms: float, floor: float, category: str, dwellings: list | None
) -> tuple[float, int]:
    """Living area of the listed dwelling from the GWR dwellings of its building.

    GWR counts whole rooms without kitchen, so a 3.5-room flat is a 3-room GWR dwelling. A
    single-dwelling house takes its only dwelling. The candidates (same rooms, and the same
    floor if that leaves any) must agree within :data:`AREA_MAX_REL_SPREAD`.

    Args:
        rooms: Listed rooms (NaN if unknown).
        floor: Listed floor (NaN if unknown).
        category: ``apartment``, ``house`` or another category.
        dwellings: ``[rooms, area, floor]`` per GWR dwelling.

    Returns:
        ``(area, n_candidates)``; area is NaN if no unique value exists.
    """
    known = [d for d in dwellings or [] if d[1]]
    if category == "house" and len(dwellings or []) == 1 and known:
        return float(known[0][1]), 1
    if np.isnan(rooms):
        return np.nan, 0
    cands = [d for d in known if d[0] is not None and d[0] == np.floor(rooms)]
    same_floor = [d for d in cands if not np.isnan(floor) and d[2] == floor]
    cands = same_floor or cands
    if not cands:
        return np.nan, 0
    areas = np.array([d[1] for d in cands], dtype=float)
    spread = (areas.max() - areas.min()) / np.median(areas)
    return (float(np.median(areas)) if spread <= AREA_MAX_REL_SPREAD else np.nan), len(cands)


def _cached(cache: ResponseCache, kind: str, keys: pd.Series) -> pd.Series:
    return keys.map(lambda k: cache.get(kind, k) if isinstance(k, str) else None)


def resolve_buildings(
    df: pd.DataFrame, cache: ResponseCache, fetch: bool = True, workers: int = WORKERS
) -> pd.DataFrame:
    """EGID and building coordinates per listing (address first, then nearest building).

    Args:
        df: Listings from :func:`rentml.comparis.normalise` (with ``coord_precise``).
        cache: Response cache.
        fetch: Request missing keys; ``False`` uses the cache only.
        workers: Parallel requests.

    Returns:
        ``egid``, ``egid_source`` (``address``/``coordinates``/NA), ``bldg_east``, ``bldg_north``.
    """
    query = address_queries(df)
    if fetch:
        fetch_all(cache, "address", {q: (q,) for q in query.dropna().unique()}, fetch_address,
                  workers)  # fmt: skip
    by_address = match_addresses(df, _cached(cache, "address", query))
    need = by_address["egid"].isna() & df["coord_precise"].astype(bool)
    points = df.loc[need, ["east", "north"]].round()
    keys = pd.Series([_point_key(e, n) for e, n in points.to_numpy()], index=points.index)
    if fetch:
        jobs = {k: (float(e), float(n)) for k, (e, n) in zip(keys, points.to_numpy(), strict=True)}
        fetch_all(cache, "buildings", jobs, fetch_buildings, workers)
    by_point = match_buildings(df.loc[need], _cached(cache, "buildings", keys))
    out = by_address.copy()
    out.loc[by_point.index] = by_point.where(by_point["egid"].notna(), out.loc[by_point.index])
    source = np.where(by_address["egid"].notna(), "address", "")
    source = np.where(out["egid"].notna() & (source == ""), "coordinates", source)
    out["egid_source"] = pd.Series(source, index=df.index, dtype="string").replace("", pd.NA)
    out["egid"] = out["egid"].round().astype("Int64")
    return out[["egid", "egid_source", "bldg_east", "bldg_north"]]


def enrich(
    df: pd.DataFrame,
    cache_path: Path = CACHE_FILE,
    fetch: bool = True,
    workers: int = WORKERS,
) -> pd.DataFrame:
    """GWR and location enrichment for listings (DSPRO1 features plus the GWR area).

    Location layers are read at the building, else at a precise listing coordinate; listings
    with a municipality centroid and no building stay without location values.

    Args:
        df: Listings from :func:`rentml.comparis.normalise`.
        cache_path: JSONL cache of the responses.
        fetch: Request missing keys; ``False`` uses the cache only (offline rerun).
        workers: Parallel requests.

    Returns:
        :data:`ENRICHED_COLUMNS`, indexed like ``df``.
    """
    cache = ResponseCache(cache_path)
    out = resolve_buildings(df, cache, fetch, workers)
    egid_key = out["egid"].astype("string")
    if fetch:
        fetch_all(cache, "madd", {k: (int(k),) for k in egid_key.dropna().unique()}, fetch_madd,
                  workers)  # fmt: skip
    madd = _cached(cache, "madd", egid_key).map(lambda v: v if isinstance(v, dict) else {})
    for col in ("year_built", "period", "apartments", "land_area"):
        out[col] = pd.to_numeric(madd.map(lambda v, c=col: v.get(c)), errors="coerce")
    from_period = out["period"].map(PERIOD_MIDPOINT)
    out["year_built_period"] = out["year_built"].isna() & from_period.notna()
    out["year_built_gwr"] = out.pop("year_built").fillna(from_period)
    precise = df["coord_precise"].astype(bool)
    east = out["bldg_east"].fillna(df["east"].where(precise)).round()
    north = out["bldg_north"].fillna(df["north"].where(precise)).round()
    pts = pd.DataFrame({"east": east, "north": north}).dropna()
    point = pd.Series(pd.NA, index=df.index, dtype="string")
    point[pts.index] = [_point_key(e, n) for e, n in pts.to_numpy()]
    if fetch:
        jobs = {_point_key(e, n): (float(e), float(n)) for e, n in pts.to_numpy()}
        fetch_all(cache, "location", jobs, fetch_location, workers)
    location = _cached(cache, "location", point).map(lambda v: v if isinstance(v, dict) else {})
    for col in ("oev", "solar", "population", "elevation"):
        out[col] = pd.to_numeric(location.map(lambda v, c=col: v.get(c)), errors="coerce")
    area = [
        recover_area(r, f, c, m.get("dwellings"))
        for r, f, c, m in zip(df["rooms"], df["floor"], df["property_category"], madd, strict=True)
    ]
    out["area_gwr"] = [a for a, _ in area]
    out["area_gwr_candidates"] = [n for _, n in area]
    return out[ENRICHED_COLUMNS]


def combine(listings: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    """Join the enrichment onto the listings in the canonical schema of the ML pipeline.

    The portal's plot area becomes ``plot_area`` (``land_area`` is the GWR building area as in
    DSPRO1); ``year_built`` is the GWR year, else the listed one; a missing living area is
    filled from the GWR (``area_source``); a centroid coordinate is replaced by the building
    coordinate (``coord_source``).

    Args:
        listings: Listings from :func:`rentml.comparis.normalise`.
        enriched: Result of :func:`enrich` for (a subset of) the listings.

    Returns:
        A copy of ``listings`` with the enrichment columns and the source flags.
    """
    out = listings.rename(columns={"land_area": "plot_area", "year_built": "year_built_listed"})
    out = out.join(enriched.reindex(out.index))
    out["year_built"] = out["year_built_gwr"].fillna(out["year_built_listed"])
    out["year_built_source"] = np.select(
        [out["year_built_period"].eq(True), out["year_built_gwr"].notna(),
         out["year_built_listed"].notna()],
        ["gwr_period", "gwr", "listing"],
        "",
    )  # fmt: skip
    filled = out["area"].isna() & out["area_gwr"].notna()
    out["area_source"] = np.select([out["area"].notna(), filled], ["listing", "gwr"], "")
    out["area"] = out["area"].fillna(out["area_gwr"])
    move = ~out["coord_precise"].astype(bool) & out["bldg_east"].notna()
    out.loc[move, ["east", "north"]] = out.loc[move, ["bldg_east", "bldg_north"]].to_numpy()
    out["coord_source"] = np.select(
        [move, out["coord_precise"].astype(bool)], ["building", "listing"], "centroid"
    )
    out["coord_precise"] = out["coord_precise"].astype(bool) | move
    for col in ("year_built_source", "area_source"):
        out[col] = out[col].replace("", pd.NA).astype("string")
    return out


def main() -> None:
    """Fill the cache for every Comparis listing in the rental or purchase scope."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    listings = comparis.normalise(comparis.load_raw())
    scope = comparis.rent_scope(listings).isna() | comparis.buy_scope(listings).isna()
    enriched = enrich(listings[scope])
    logger.info("Coverage:\n%s", enriched.notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
