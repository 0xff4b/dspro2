"""Listings: loading, schema, duplicate audit, cleaning, splits and engineered features."""

import logging
import os
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from rentml.config import (
    AREA_RANGE,
    INTERIM,
    PRICE_RANGE,
    RANDOM_STATE,
    REFERENCE_YEAR,
    ROOMS_RANGE,
    SNAPSHOT_DATE,
    SNAPSHOT_DIR,
)
from rentml.text import anonymize_series

logger = logging.getLogger(__name__)

DB_EXPORT = INTERIM / "listings_db.parquet"  # frozen, anonymised database export

COLUMN_MAP = {
    "area_sqm": "area", "price_cold": "price", "oev_score": "oev", "solar_class": "solar",
    "elevation_m": "elevation", "lv95_east": "east", "lv95_north": "north", "gbauj": "year_built",
    "ganzwhg": "apartments", "garea": "land_area", "auxiliary_costs": "aux_costs",
    "kaution": "deposit",
}  # fmt: skip
TEXT_COLUMNS = ["source", "slug", "address", "description"]
NUMERIC_COLUMNS = [
    "area", "rooms", "price", "aux_costs", "deposit", "population", "oev", "solar", "elevation",
    "east", "north", "year_built", "apartments", "land_area",
]  # fmt: skip
CANONICAL_COLUMNS = ["listing_id", *TEXT_COLUMNS, *NUMERIC_COLUMNS, "egid", "available_from"]
NON_MARKET_REGIMES = ("cooperative", "cost_based_or_subsidised", "shared_flat")

# Columns of final_listings.csv joined onto model_wide.csv.
_ENRICHMENT = [
    "address", "oev_score", "solar_class", "elevation_m", "egid", "gbauj", "ganzwhg", "garea",
]  # fmt: skip
# DSPRO1 completeness rule (final_listings.csv): detail page, swisstopo and GWR all present.
_DSPRO1_REQUIRED = ["area_sqm", "rooms", "price_cold", "population", *_ENRICHMENT]
# Detail-page columns of the database and their search-page fallbacks.
_FALLBACKS = {"area_sqm": "area_listing", "rooms": "rooms_listing", "price_cold": "price_listing"}
# Read-only, one row per listing; with several buildings the most complete GWR row wins.
LISTINGS_SQL = """
SELECT DISTINCT ON (l.id)
    l.id AS listing_id, l.slug, ld.address,
    ld.area_sqm, l.m_sqrd AS area_listing,
    ld.rooms, l.rooms AS rooms_listing,
    ld.price_cold, l.price_cold AS price_listing,
    ld.auxiliary_costs, ld.kaution, ld.available_from, ld.description,
    sd.population, sd.oev_score, sd.solar_class, sd.elevation_m, sd.lv95_east, sd.lv95_north,
    lb.egid, gb.gbauj, gb.ganzwhg, gb.garea
FROM listings AS l
LEFT JOIN listing_details   AS ld ON ld.listing_id = l.id
LEFT JOIN swisstopo_details AS sd ON sd.listing_id = l.id
LEFT JOIN listing_buildings AS lb ON lb.listing_id = l.id
LEFT JOIN gwr_buildings     AS gb ON gb.egid = lb.egid
ORDER BY l.id, num_nulls(gb.egid, gb.gbauj, gb.ganzwhg, gb.garea), lb.egid
"""

DESC_MIN_CHARS = 30  # shorter descriptions are placeholders and never link listings
DESC_MAX_GROUP = 10  # a text shared by more listings is property-manager boilerplate
EXACT_COLUMNS = ["east", "north", "area", "rooms", "price"]
PPSQM_LEVELS = ("municipality_id", "district_id", "canton")  # finest first, then CH
PPSQM_MIN_GROUP = 10
PPSQM_Z = 3.5  # Iglewicz & Hoaglin threshold for modified z-scores
MAD_TO_SIGMA = 1 / 0.6745

BASE = ["area", "rooms", "area_per_room", "rooms_is_integer"]
BUILDING = ["year_built", "building_age", "apartments", "land_area", "land_area_per_apartment"]
LOCATION = ["population", "oev", "solar", "elevation", "muni_density", "muni_population"]
COORDS = ["east", "north"]
ROT_COORDS = ["rot45_x", "rot45_y"]
_LV95_ORIGIN = (2_600_000.0, 1_200_000.0)

Folds = list[tuple[np.ndarray, np.ndarray]]


def load_listings(
    source: Literal["auto", "db", "csv"] = "auto", database_url: str | None = None
) -> tuple[pd.DataFrame, str]:
    """Load the listings from the database (frozen export) or the DSPRO1 CSV snapshot.

    The first database query is anonymised and frozen at :data:`DB_EXPORT`; later runs read
    that file. ``"auto"`` falls back to the CSV snapshot (no descriptions) when no URL is set
    or the database is unreachable.

    ``database_url`` defaults to ``DATABASE_URL``. Returns the canonical listings indexed by
    ``listing_id`` and the source actually used (``"db"`` or ``"csv"``).
    """
    if source not in ("auto", "db", "csv"):
        raise ValueError(f"Unknown source {source!r}")
    if source == "csv":
        return _load_csv(), "csv"
    if DB_EXPORT.is_file():
        logger.info("Reading the frozen database export %s (delete it to re-query)", DB_EXPORT)
        return pd.read_parquet(DB_EXPORT), "db"
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        if source == "db":
            raise ValueError("source='db' needs DATABASE_URL or a frozen export")
        return _load_csv(), "csv"
    try:
        listings = _query_db(url)
    except OperationalError as exc:
        if source == "db":
            raise
        logger.warning("Database unreachable (%s); using the CSV snapshot", str(exc).split("\n")[0])
        return _load_csv(), "csv"
    listings["description"] = anonymize_series(listings["description"])
    DB_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    listings.to_parquet(DB_EXPORT)
    logger.info("Froze %d listings at %s", len(listings), DB_EXPORT)
    return listings, "db"


def _query_db(url: str) -> pd.DataFrame:
    engine = create_engine(url.replace("postgres://", "postgresql://", 1))
    try:
        with engine.connect() as conn:
            raw = pd.read_sql(text(LISTINGS_SQL), conn.execution_options(postgresql_readonly=True))
    finally:
        engine.dispose()
    raw["dspro1_enriched"] = raw.reindex(columns=_DSPRO1_REQUIRED).notna().all(axis=1)
    for preferred, fallback in _FALLBACKS.items():
        raw[preferred] = _to_float(raw[preferred]).fillna(_to_float(raw.pop(fallback)))
    return _canonical(raw)


def _load_csv() -> pd.DataFrame:
    base = pd.read_csv(SNAPSHOT_DIR / "model_wide.csv")
    enrichment = pd.read_csv(
        SNAPSHOT_DIR / "final_listings.csv", usecols=["listing_id", *_ENRICHMENT]
    )
    merged = base.merge(enrichment, on="listing_id", how="left", validate="1:1", indicator=True)
    merged["dspro1_enriched"] = merged.pop("_merge").eq("both")
    logger.info("Loaded %d listings from the CSV snapshot %s", len(merged), SNAPSHOT_DIR)
    return _canonical(merged)


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=COLUMN_MAP)
    out["source"] = out.get("source", "rentumo")
    out["observed_at"] = pd.Timestamp(SNAPSHOT_DATE).as_unit("ns")  # DSPRO1 has no timestamps
    extras = [c for c in out.columns if c not in CANONICAL_COLUMNS]
    out = out.reindex(columns=CANONICAL_COLUMNS + extras)
    return out.astype({"listing_id": "int64"}).set_index("listing_id", verify_integrity=True)


def _to_float(values: pd.Series) -> pd.Series:
    """Parse numbers, also with Swiss formatting (``1'850.–``, ``3½``, ``CHF``)."""
    cleaned = values.astype("string").str.replace("½", ".5")
    cleaned = cleaned.str.replace(r"CHF|Fr\.|\.[-–]+$|[\s'’]", "", regex=True).str.replace(",", ".")
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def fix_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, set implausible room counts to NaN and add ``log_price``.

    Adds ``rooms_invalid`` (outside :data:`ROOMS_RANGE`, set to NaN).
    """
    out = df.copy()
    out[NUMERIC_COLUMNS] = out[NUMERIC_COLUMNS].apply(_to_float)
    out["egid"] = pd.to_numeric(out["egid"], errors="coerce").round().astype("Int64")
    for col in TEXT_COLUMNS:
        out[col] = out[col].map(lambda v: (v.strip() or None) if isinstance(v, str) else None)
    dates = out["available_from"].astype("string")  # ISO or Swiss dd.mm.yyyy
    out["available_from"] = pd.to_datetime(dates, format="mixed", dayfirst=True, errors="coerce")
    rooms = out["rooms"]
    out["rooms_invalid"] = rooms.notna() & ~rooms.between(*ROOMS_RANGE)
    out["rooms"] = rooms.mask(out["rooms_invalid"])
    out["log_price"] = np.log(out["price"].where(out["price"] > 0))
    logger.info("fix_schema: %d implausible room counts set to NaN", out["rooms_invalid"].sum())
    return out


def text_coverage(df: pd.DataFrame, min_chars: int = 50) -> dict[str, float]:
    """Share of listings with a non-blank description and with at least ``min_chars``."""
    lengths = df["description"].fillna("").astype(str).str.strip().str.len()
    has_text = lengths > 0
    return {
        "n_total": float(len(df)),
        "n_with_text": float(has_text.sum()),
        "coverage": float(has_text.mean()),
        "coverage_min_chars": float((lengths >= min_chars).mean()),
        "median_chars": float(lengths[has_text].median()),
    }


def _key_pairs(key: pd.Series) -> pd.DataFrame:
    """Link every row with a non-missing key to the first row of its key group (positions)."""
    codes = pd.factorize(key)[0]
    pos = np.flatnonzero(codes >= 0)
    first = pd.Series(pos).groupby(codes[pos]).transform("min").to_numpy()
    keep = first != pos
    return pd.DataFrame({"pos_a": first[keep], "pos_b": pos[keep]})


def _building_pairs(
    df: pd.DataFrame, block: pd.Series, area_tol: float, price_rel_tol: float
) -> pd.DataFrame:
    """Pairs in the same building block with equal rooms and close area and price."""
    rows = df[["rooms", "area", "price"]].assign(block=block, pos=np.arange(len(df))).dropna()
    pairs = rows.merge(rows, on=["block", "rooms"], suffixes=("_a", "_b"))
    pairs = pairs[pairs["pos_a"] < pairs["pos_b"]]
    close_area = (pairs["area_a"] - pairs["area_b"]).abs() <= area_tol
    max_price = np.maximum(pairs["price_a"], pairs["price_b"])
    close_price = (pairs["price_a"] - pairs["price_b"]).abs() <= price_rel_tol * max_price
    return pairs.loc[close_area & close_price, ["pos_a", "pos_b"]]


def link_duplicates(
    df: pd.DataFrame, area_tol: float = 3.0, price_rel_tol: float = 0.10, coord_round_m: float = 5.0
) -> pd.DataFrame:
    """Find listing pairs that describe the same rental object.

    Rules: same ``slug``; same anonymised description (case and punctuation ignored, see
    :data:`DESC_MIN_CHARS` / :data:`DESC_MAX_GROUP`); exact duplicates on
    :data:`EXACT_COLUMNS`; same building (``egid``, coordinates rounded to ``coord_round_m``,
    or normalised address) with equal rooms, ``|Δarea| <= area_tol`` and
    ``|Δprice| <= price_rel_tol * max(price)``. Returns one row per link: listing ids ``a``,
    ``b`` and the ``rule``.
    """
    desc = df["description"].str.casefold().str.replace(r"\W+", " ", regex=True).str.strip()
    desc = desc.where(desc.str.len() >= DESC_MIN_CHARS)
    desc = desc.where(desc.map(desc.value_counts()) <= DESC_MAX_GROUP)
    grid = (df[COORDS] / coord_round_m).round().astype("Int64").astype("string")
    address = df["address"].str.casefold().str.replace(r"\W+", "", regex=True)
    keys = {"slug": df["slug"], "description": desc}
    keys["exact"] = df.groupby(EXACT_COLUMNS).ngroup().where(df[EXACT_COLUMNS].notna().all(axis=1))
    blocks = {"egid": df["egid"], "coords": grid["east"] + "_" + grid["north"]}
    blocks["address"] = address.where(address != "")
    parts = {rule: _key_pairs(key) for rule, key in keys.items()}
    parts |= {rule: _building_pairs(df, b, area_tol, price_rel_tol) for rule, b in blocks.items()}
    links = pd.concat(parts, names=["rule"]).reset_index(level="rule", drop=False)
    ids = df.index.to_numpy()
    links = links.assign(a=ids[links["pos_a"].astype(int)], b=ids[links["pos_b"].astype(int)])
    return links[["a", "b", "rule"]].reset_index(drop=True)


def assign_object_ids(df: pd.DataFrame, links: pd.DataFrame) -> pd.Series:
    """Object id per listing: connected components of the duplicate links.

    Links are transitive (A~B and B~C put A and C into one object). Ids ``obj_000001, ...``
    are numbered by the smallest ``listing_id`` of each object, independent of row order.
    """
    n = len(df)
    a, b = df.index.get_indexer(links["a"]), df.index.get_indexer(links["b"])
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    smallest = pd.Series(df.index, index=df.index).groupby(labels).transform("min")
    rank = smallest.rank(method="dense").astype(int)
    logger.info("%d listings -> %d objects", n, rank.max() if n else 0)
    return ("obj_" + rank.astype(str).str.zfill(6)).rename("object_id")


def dedup_report(df: pd.DataFrame, object_ids: pd.Series, links: pd.DataFrame) -> pd.Series:
    """Object counts, group sizes, price spread within objects and links per rule."""
    sizes = object_ids.value_counts()
    price = df["price"].groupby(object_ids)
    spread = ((price.max() - price.min()) / price.max())[sizes.index[sizes > 1]]
    n_rows, n_objects = len(df), len(sizes)
    report = {"rows": n_rows, "unique_objects": n_objects, "duplicate_rows": n_rows - n_objects}
    report["duplicate_share_pct"] = 100 * (n_rows - n_objects) / n_rows
    report["objects_with_duplicates"] = (sizes > 1).sum()
    by_size = sizes.clip(upper=4).value_counts().drop(1, errors="ignore")
    report |= by_size.rename(lambda n: f"objects_size_{n}{'plus' if n == 4 else ''}").to_dict()
    report["max_group_size"] = sizes.max()
    report["median_rel_price_range_dup_objects"] = spread.median()
    report |= links["rule"].value_counts().add_prefix("links_").to_dict()
    return pd.Series(report, dtype=float, name="value")


def collapse_objects(df: pd.DataFrame, object_ids: pd.Series) -> pd.DataFrame:
    """One row per object: the listing with the smallest ``listing_id``, median price.

    Adds ``object_id`` and ``n_listings``; ``log_price`` is recomputed from the median price.
    """
    out = df.assign(object_id=object_ids).sort_index()
    median_price = out.groupby("object_id")["price"].median()
    out = out.drop_duplicates("object_id").copy()
    out["price"] = out["object_id"].map(median_price)
    out["n_listings"] = out["object_id"].map(object_ids.value_counts())
    if "log_price" in out:
        out["log_price"] = np.log(out["price"].where(out["price"] > 0))
    return out


def domain_filter(
    df: pd.DataFrame,
    area_range: tuple[float, float] = AREA_RANGE,
    price_range: tuple[float, float] = PRICE_RANGE,
    rooms_range: tuple[float, float] = ROOMS_RANGE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep plausible objects; the defaults are the apartment rules of the evaluation plan.

    Bounds are inclusive; missing area or price is removed, missing rooms are kept. Returns
    ``(kept, removed)``; ``removed`` has a ``reason`` column (first failing rule).
    """
    area, price, rooms = df["area"], df["price"], df["rooms"]
    rules = {
        "area_missing": area.isna(),
        "area_below_min": area < area_range[0],
        "area_above_max": area > area_range[1],
        "price_missing": price.isna(),
        "price_below_min": price < price_range[0],
        "price_above_max": price > price_range[1],
        "rooms_out_of_range": rooms.notna() & ~rooms.between(*rooms_range),
    }
    reason = pd.Series(np.select(list(rules.values()), list(rules), default=""), index=df.index)
    removed = reason != ""
    logger.info("domain_filter: removed %s", reason[removed].value_counts().to_dict())
    return df[~removed].copy(), df[removed].assign(reason=reason[removed])


def price_per_sqm_outliers(df: pd.DataFrame) -> pd.Series:
    """Flag ``|z| > PPSQM_Z`` for the robust z-score of log CHF/m² (one row per object).

    ``z = (x - median) / (MAD / 0.6745)`` within the finest level of :data:`PPSQM_LEVELS` whose
    group has at least :data:`PPSQM_MIN_GROUP` rows and a positive MAD; else the next coarser
    level, finally all of Switzerland.
    """
    x = np.log(df["price"] / df["area"])
    center = pd.Series(x.median(), index=df.index)
    scale = pd.Series((x - x.median()).abs().median() * MAD_TO_SIGMA, index=df.index)
    for col in reversed(PPSQM_LEVELS):  # coarse to fine: a usable finer group overrides its parent
        grouped = x.groupby(df[col])
        median = grouped.transform("median")
        mad = (x - median).abs().groupby(df[col]).transform("median")
        usable = (grouped.transform("count") >= PPSQM_MIN_GROUP) & (mad > 0)
        center = center.mask(usable, median)
        scale = scale.mask(usable, mad * MAD_TO_SIGMA)
    flag = ((x - center) / scale).abs() > PPSQM_Z
    logger.info("price_per_sqm_outliers: %d of %d flagged", flag.sum(), len(df))
    return flag.rename("ppsqm_outlier")


def regime_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by ``rent_regime`` into ``(market, non_market)``; unknown labels count as market."""
    non_market = df["rent_regime"].isin(NON_MARKET_REGIMES)
    return df[~non_market].copy(), df[non_market].copy()


def recover_rooms_from_text(
    rooms: pd.Series, text_rooms: pd.Series, max_diff: float = 0.5
) -> tuple[pd.Series, pd.Series]:
    """Take half rooms (x.5 within ``max_diff``) and missing room counts from the text.

    DSPRO1 stored whole rooms (3.5 became 4). Text values outside :data:`ROOMS_RANGE` or not a
    multiple of 0.5 are ignored. Returns ``(rooms, source)`` with ``source`` one of ``table``,
    ``text_half_room``, ``text_missing``, ``missing``.
    """
    text_rooms = text_rooms.reindex(rooms.index).astype(float)
    valid = text_rooms.between(*ROOMS_RANGE) & (text_rooms * 2 % 1 == 0)
    diff = (text_rooms - rooms).abs()
    half = valid & (text_rooms % 1 == 0.5) & (diff > 0) & (diff <= max_diff)
    fill = valid & rooms.isna()
    recovered = rooms.mask(half | fill, text_rooms)
    source = pd.Series("table", index=rooms.index)
    source = source.mask(half, "text_half_room").mask(fill, "text_missing")
    return recovered, source.mask(recovered.isna(), "missing")


def make_strata(df: pd.DataFrame, min_stratum: int = 20) -> pd.Series:
    """Canton x price quartile strata with at least ``min_stratum`` rows.

    Rare strata merge into ``ALL|<quartile>``; if that is still rare, its rows join the largest
    stratum of their quartile.
    """
    bins = pd.qcut(df["price"], 4, labels=False, duplicates="drop")
    quartile = "P" + (bins + 1).astype("Int64").astype(str)
    strata = df["canton"].astype(str) + "|" + quartile
    strata = strata.mask(strata.map(strata.value_counts()) < min_stratum, "ALL|" + quartile)
    rare = strata.map(strata.value_counts()) < min_stratum
    largest = strata[~rare].groupby(quartile[~rare]).agg(lambda s: s.mode().iloc[0])
    return strata.mask(rare, quartile.map(largest)).rename("stratum")


def _stratified_group_folds(groups: pd.Series, strata: pd.Series, n_splits: int) -> Folds:
    codes = pd.factorize(groups, sort=True)[0]
    # sklearn 1.7's StratifiedGroupKFold(shuffle=True) shuffles the class counts but not the
    # group ids they belong to, which breaks the stratification: permute the groups instead.
    permuted = np.random.default_rng(RANDOM_STATE).permutation(codes.max() + 1)[codes]
    splitter = StratifiedGroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(codes)), strata.to_numpy(), permuted))


def train_calib_test_split(df: pd.DataFrame, strata: pd.Series) -> pd.Series:
    """60/20/20 train/calib/test labels, grouped by ``object_id`` and stratified.

    Five stratified group folds: fold 0 is the test set, fold 1 the calibration set.
    ``strata`` is aligned to ``df`` by index label.
    """
    folds = _stratified_group_folds(df["object_id"], strata.reindex(df.index), 5)
    split = np.full(len(df), "train", dtype=object)
    split[folds[0][1]] = "test"
    split[folds[1][1]] = "calib"
    return pd.Series(split, index=df.index, name="split")


def grouped_cv_folds(df: pd.DataFrame, strata: pd.Series | None = None, n_splits: int = 5) -> Folds:
    """CV folds grouped by ``object_id`` (stratified with ``strata``), positions into ``df``."""
    if strata is None:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        return list(splitter.split(df, groups=df["object_id"]))
    return _stratified_group_folds(df["object_id"], strata.reindex(df.index), n_splits)


def spatial_cv_folds(df: pd.DataFrame, n_splits: int = 5) -> Folds:
    """CV folds that hold out whole municipalities (positions into ``df``)."""
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(df, groups=df["municipality_id"]))


def add_engineered_features(df: pd.DataFrame, reference_year: int = REFERENCE_YEAR) -> pd.DataFrame:
    """Add area per room, building age, land area per apartment and 45° rotated LV95 coordinates.

    Ratios with a non-positive denominator and features of missing input columns are NaN.
    Works on one row, so training and the app derive the features the same way.
    """
    out = df.copy()
    inputs = ["area", "rooms", "year_built", "land_area", "apartments", *COORDS]
    num = df.reindex(columns=inputs).apply(pd.to_numeric, errors="coerce").astype(float)
    rooms, apartments = num["rooms"], num["apartments"]
    out["area_per_room"] = num["area"] / rooms.where(rooms > 0)
    out["rooms_is_integer"] = (rooms % 1 == 0).astype(float).where(rooms.notna())
    out["building_age"] = (reference_year - num["year_built"]).clip(lower=0)
    out["land_area_per_apartment"] = num["land_area"] / apartments.where(apartments > 0)
    east, north = num["east"] - _LV95_ORIGIN[0], num["north"] - _LV95_ORIGIN[1]
    cos45 = np.cos(np.pi / 4)
    out["rot45_x"] = (east - north) * cos45
    out["rot45_y"] = (east + north) * cos45
    return out


def _neighbour_stats(
    pool_xy: np.ndarray,
    pool_y: np.ndarray,
    query_xy: np.ndarray,
    k: int,
    self_pos: np.ndarray | None = None,
) -> np.ndarray:
    """Mean and median target of the k nearest pool rows, skipping ``self_pos`` (LOO)."""
    out = np.full((len(query_xy), 2), np.nan)
    ok = ~np.isnan(query_xy).any(axis=1)
    if not ok.any():
        return out
    n_neighbours = min(k + (self_pos is not None), len(pool_xy))
    nn = NearestNeighbors(n_neighbors=n_neighbours).fit(pool_xy)
    idx = nn.kneighbors(query_xy[ok], return_distance=False)
    keep = np.ones(idx.shape, dtype=bool) if self_pos is None else idx != self_pos[ok, None]
    keep &= keep.cumsum(axis=1) <= k
    values = np.where(keep, pool_y[idx], np.nan)
    out[ok] = np.column_stack([np.nanmean(values, axis=1), np.nanmedian(values, axis=1)])
    return out


def knn_price_features(
    train_df: pd.DataFrame,
    apply_df: pd.DataFrame | None = None,
    k: int = 10,
    mode: Literal["loo", "oof"] = "loo",
    folds: Folds | None = None,
) -> pd.DataFrame:
    """DSPRO1 location feature: mean and median rent of the k nearest training listings.

    Coordinates are standardised on the training rows. With ``apply_df`` its rows use all
    training rows. Otherwise ``mode="loo"`` (DSPRO1) uses the k nearest *other* training rows,
    so relistings of the same object leak their rent; ``mode="oof"`` uses only the training
    part of the fold in which a row is validated (``folds`` as positions into ``train_df``).
    Returns ``knn_price_mean`` and ``knn_price_median``, indexed like the featurised rows.
    """
    scaler = StandardScaler().fit(train_df[COORDS])  # NaN rows are ignored in fit
    xy, y = scaler.transform(train_df[COORDS]), train_df["price"].to_numpy(dtype=float)
    pool = ~np.isnan(xy).any(axis=1) & ~np.isnan(y)
    columns = ["knn_price_mean", "knn_price_median"]
    if apply_df is not None:
        stats = _neighbour_stats(xy[pool], y[pool], scaler.transform(apply_df[COORDS]), k)
        return pd.DataFrame(stats, index=apply_df.index, columns=columns)
    if mode == "loo":
        self_pos = np.where(pool, pool.cumsum() - 1, -1)  # position of each row in the pool
        stats = _neighbour_stats(xy[pool], y[pool], xy, k, self_pos)
    else:
        stats = np.full((len(xy), 2), np.nan)
        for train_pos, val_pos in folds:
            train_pos = train_pos[pool[train_pos]]
            stats[val_pos] = _neighbour_stats(xy[train_pos], y[train_pos], xy[val_pos], k)
    return pd.DataFrame(stats, index=train_df.index, columns=columns)
