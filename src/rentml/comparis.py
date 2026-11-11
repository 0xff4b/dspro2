"""Comparis listings (Apify export): normalising, plausibility flags, rental scope and cleaned DB.

The raw database :data:`RAW_DB` is opened read-only. The cleaning notebook
(``notebooks/comparis_cleaning.ipynb``) writes :data:`CLEANED_DB` with :func:`write_db`. Titles
and descriptions are anonymised (:func:`rentml.text.anonymize_series`) before they reach disk.
Column names follow :data:`rentml.data.CANONICAL_COLUMNS` (``price`` is the net rent, ``area``
the living area, ``east``/``north`` LV95), so the rental objects fit the ML pipeline.
"""

import hashlib
import json
import logging
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from rentml.config import DATA, INTERIM, REFERENCE_YEAR, ROOMS_RANGE
from rentml.data import DESC_MIN_CHARS, domain_filter
from rentml.geo import LV95
from rentml.text import anonymize_series

logger = logging.getLogger(__name__)

RAW_DB = DATA / "raw" / "comparis.db"
CLEANED_DB = INTERIM / "comparis_cleaned.db"

# Raw columns read from apify_listings; raw_json, images_json, key_data and main_data_json only
# repeat them (price_type is taken from raw_json, it has no column of its own).
RAW_SQL = """
SELECT id, url, partner, deal_type, property_type,
       json_extract(raw_json, '$.priceType') AS price_type,
       title, description, price, price_net, side_costs, rooms, living_area, usable_area,
       plot_area, floor, number_of_floors, year_built, renovation_year, balcony_count,
       pets_allowed, image_count, available_from, street, street_number, zip, city,
       latitude, longitude, published_at, scraped_at, first_seen_at, last_seen_at,
       features_json, attributes_json
FROM apify_listings
"""

PROPERTY_CATEGORY = {
    **dict.fromkeys(
        ["Wohnung", "Dachwohnung", "Maisonette", "Attika", "Loft", "Terrassenwohnung", "Studio"],
        "apartment",
    ),
    **dict.fromkeys(
        ["Haus", "Villa", "Doppeleinfamilienhaus", "Reihenhaus", "Bauernhaus", "Terrassenhaus",
         "Stadthaus"],
        "house",
    ),
    "Einzelzimmer": "room",
}  # fmt: skip
MONTHLY_RENT = "Mietpreis pro Monat"  # weekly, daily and yearly rents are holiday or commercial
TOTAL_PRICE = "Kaufpreis"  # "Kaufpreis pro m²" is a rate, not a price
# Purchase objects: below CHF 50k are placeholders, symbolic prices or parking spaces; the upper
# bound removes "987654336"-style placeholders. Houses have larger areas than flats.
BUY_PRICE_RANGE = (50_000.0, 30_000_000.0)
BUY_AREA_RANGE = (15.0, 2_000.0)
BUY_CATEGORIES = ("apartment", "house")
CH_BBOX = (5.95, 45.81, 10.50, 47.81)  # lon/lat bounds of Switzerland (WGS84)
YEAR_RANGE = (1000, REFERENCE_YEAR + 5)  # 0 and 9999 are placeholders
MIN_MEASURE_M2 = 2.0  # "1 m²" is a placeholder for an unknown area
MAX_BALCONIES = 10
# A point shared by listings in several streets (or by many without a street) is a municipality
# or postcode centroid, not the building.
CENTROID_MIN_STREETS = 3
CENTROID_MIN_LISTINGS = 5
LINK_AREA_BUCKET_M2 = 5.0  # a shared text links listings only with equal rooms and area bucket
FLOOR_LABELS = {"eg": 0.0, "erdgeschoss": 0.0, "untergeschoss": -1.0, "ug": -1.0}
# Numeric features and attributes, in metres (Distance*), minutes (DistanceTime*) or m².
NUMERIC_EXTRAS = {
    "GardenMeters": "garden_area", "BalconyMeters": "balcony_area",
    "NumApartments": "apartments_listed",
    "DistanceBus": "dist_bus_m", "DistanceTrainStation": "dist_train_m",
    "DistanceShoppingFacilities": "dist_shopping_m", "DistanceKindergarden": "dist_kindergarten_m",
    "DistancePrimarySchool": "dist_primary_school_m",
    "DistanceSecondarySchool": "dist_secondary_school_m", "DistanceHighway": "dist_highway_m",
    "DistanceUniversity": "dist_university_m", "DistancePlayground": "dist_playground_m",
    "DistanceTimeBus": "time_bus_min", "DistanceTimeShoppingFacilities": "time_shopping_min",
}  # fmt: skip
_TO_LV95 = Transformer.from_crs(4326, LV95, always_xy=True)


def load_raw(path: Path = RAW_DB) -> pd.DataFrame:
    """Read the listings of the raw Comparis database read-only.

    Args:
        path: SQLite file with the ``apify_listings`` table.

    Returns:
        One row per raw listing with the columns of :data:`RAW_SQL`, including ``price_type``.

    Raises:
        FileNotFoundError: If ``path`` does not exist (SQLite would create an empty file).
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        raw = pd.read_sql(RAW_SQL, con)
    finally:
        con.close()
    logger.info("Read %d raw listings from %s", len(raw), path)
    return raw


def parse_floor(values: pd.Series) -> pd.Series:
    """Parse Comparis floor labels.

    Args:
        values: Labels such as ``"EG"``, ``"3. Stock"``, ``"3"`` or ``"nicht verfügbar"``.

    Returns:
        Floor numbers (ground floor 0, basement -1); unknown labels are NaN.
    """
    text = values.astype("string").str.strip().str.casefold()
    number = pd.to_numeric(text.str.extract(r"^(-?\d{1,2})\b", expand=False), errors="coerce")
    return number.astype(float).fillna(text.map(FLOOR_LABELS).astype(float))


def _measure(values: pd.Series) -> pd.Series:
    """Leading number of ``"500 m"``, ``"3 min"`` or ``"1000"``; booleans and text are NaN."""
    text = values.where(values.map(lambda v: not isinstance(v, bool))).astype("string")
    return pd.to_numeric(text.str.extract(r"^\s*(\d+(?:\.\d+)?)", expand=False), errors="coerce")


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name).lower()


def expand_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Expand the ``features_json`` and ``attributes_json`` columns.

    Args:
        raw: Raw listings from :func:`load_raw`.

    Returns:
        One row per listing (index ``listing_id``): ``Has*``/``Is*`` flags as ``has_*``/``is_*``
        (1 = stated, 0 = not stated) and the numeric entries of :data:`NUMERIC_EXTRAS`.
    """
    index = pd.Index(raw["id"].astype("int64"), name="listing_id")
    parsed = [
        {**json.loads(f or "{}"), **json.loads(a or "{}")}
        for f, a in zip(raw["features_json"], raw["attributes_json"], strict=True)
    ]
    entries = pd.DataFrame(parsed, index=index)
    out = pd.DataFrame(index=index)
    for key in sorted(c for c in entries if c.startswith(("Has", "Is"))):
        out[_snake(key)] = entries[key].eq(True).astype("int8")
    for key, name in NUMERIC_EXTRAS.items():
        out[name] = _measure(entries[key]) if key in entries else np.nan
    return out


def _positive(values: pd.Series, minimum: float = 0.0) -> pd.Series:
    num = pd.to_numeric(values, errors="coerce").astype(float)
    return num.where(num > minimum)


def _datetime(values: pd.Series) -> pd.Series:
    """ISO timestamps as naive UTC; values without an offset are taken as given."""
    parsed = pd.to_datetime(values, errors="coerce", utc=True, format="ISO8601")
    return parsed.dt.tz_localize(None)


def _available_from(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = values.astype("string").str.strip().str.casefold()
    date = pd.to_datetime(text, format="%d.%m.%Y", errors="coerce")
    label = pd.Series("other", index=values.index, dtype="string")
    label = label.mask(text.isna(), pd.NA).mask(date.notna(), "date")
    label = label.mask(text.eq("sofort"), "immediately").mask(
        text.eq("nach vereinbarung"), "by_arrangement"
    )
    return date, label


def _years(built: pd.Series, renovated: pd.Series) -> tuple[pd.Series, pd.Series]:
    built = _positive(built).where(lambda s: s.between(*YEAR_RANGE))
    renovated = _positive(renovated).where(lambda s: s.between(*YEAR_RANGE))
    return built, renovated.mask(renovated < built)


def _coordinates(lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
    lat, lon = pd.to_numeric(lat, errors="coerce"), pd.to_numeric(lon, errors="coerce")
    inside = lon.between(CH_BBOX[0], CH_BBOX[2]) & lat.between(CH_BBOX[1], CH_BBOX[3])
    lat, lon = lat.where(inside), lon.where(inside)
    east, north = _TO_LV95.transform(lon.to_numpy(), lat.to_numpy())
    return pd.DataFrame(
        {"latitude": lat, "longitude": lon, "east": east, "north": north}, index=lat.index
    )


def coordinate_precision(df: pd.DataFrame) -> pd.DataFrame:
    """Detect municipality or postcode centroids among the listing coordinates.

    A point is a centroid if listings in at least :data:`CENTROID_MIN_STREETS` streets share it,
    or if it has no street and at least :data:`CENTROID_MIN_LISTINGS` listings.

    Args:
        df: Listings with ``east``, ``north`` and ``street``.

    Returns:
        ``coord_listings`` and ``coord_streets`` per identical point and ``coord_precise``.
    """
    point = df["east"].round(1).astype("string") + "_" + df["north"].round(1).astype("string")
    street = df["street"].astype("string").str.casefold().str.replace(r"\W+", "", regex=True)
    n_listings = point.map(point.value_counts()).astype("Int64")
    n_streets = street.groupby(point).transform("nunique").astype("Int64")
    centroid = (n_streets >= CENTROID_MIN_STREETS) | (
        street.isna() & (n_listings >= CENTROID_MIN_LISTINGS)
    )
    precise = point.notna() & ~centroid.fillna(False)
    return pd.DataFrame(
        {"coord_listings": n_listings, "coord_streets": n_streets, "coord_precise": precise},
        index=df.index,
    )


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Type, anonymise and rename the raw listings; every raw row is kept.

    ``price`` is the net rent for rentals and the purchase price for sales. ``aux_costs_known``
    is true only if the listing states separate auxiliary costs; otherwise net and gross rent
    cannot be told apart. Placeholders (year 0, area "1 m²", coordinates outside Switzerland,
    implausible rooms) become NaN; ``rooms_invalid`` marks the dropped room counts.

    Args:
        raw: Raw listings from :func:`load_raw`.

    Returns:
        Normalised listings indexed by ``listing_id``.
    """
    r = raw.set_index(pd.Index(raw["id"].astype("int64"), name="listing_id"))
    rent = r["deal_type"].eq("rent")
    out = pd.DataFrame(index=r.index)
    out["source"] = "comparis"
    out["partner"] = r["partner"].astype("string").str.strip()
    out["url"] = r["url"]
    out["deal_type"] = r["deal_type"]
    out["price_type"] = r["price_type"]
    out["property_type"] = r["property_type"]
    out["property_category"] = r["property_type"].map(PROPERTY_CATEGORY).fillna("other")
    out["title"] = anonymize_series(r["title"])
    out["description"] = anonymize_series(r["description"])

    gross, net, aux = _positive(r["price"]), _positive(r["price_net"]), _positive(r["side_costs"])
    out["price"] = net.where(rent, gross)
    out["gross_rent"] = gross.where(rent)
    out["aux_costs"] = aux.where(rent)
    out["aux_costs_known"] = rent & aux.notna()
    rooms = _positive(r["rooms"])
    out["rooms_invalid"] = rooms.notna() & ~rooms.between(*ROOMS_RANGE)
    out["rooms"] = rooms.mask(out["rooms_invalid"])
    out["area"] = _positive(r["living_area"], MIN_MEASURE_M2)
    out["usable_area"] = _positive(r["usable_area"], MIN_MEASURE_M2)
    out["land_area"] = _positive(r["plot_area"], MIN_MEASURE_M2)
    out["floor"] = parse_floor(r["floor"])
    out["number_of_floors"] = _positive(r["number_of_floors"])
    out["year_built"], out["renovation_year"] = _years(r["year_built"], r["renovation_year"])
    balconies = pd.to_numeric(r["balcony_count"], errors="coerce")
    out["balcony_count"] = balconies.where(balconies.between(0, MAX_BALCONIES))
    out["pets_allowed"] = pd.to_numeric(r["pets_allowed"], errors="coerce").eq(1).astype("Int8")
    out["pets_allowed"] = out["pets_allowed"].mask(r["pets_allowed"].isna())
    out["image_count"] = pd.to_numeric(r["image_count"], errors="coerce").astype("Int64")
    out["available_from"], out["available_label"] = _available_from(r["available_from"])

    for col in ["street", "street_number", "zip", "city"]:
        out[col] = r[col].astype("string").str.strip().replace("", pd.NA)
    street = (out["street"] + " " + out["street_number"].fillna("")).str.strip()
    out["address"] = (street.fillna("") + ", " + out["zip"] + " " + out["city"]).str.strip(", ")
    out = out.join(_coordinates(r["latitude"], r["longitude"]))
    out = out.join(coordinate_precision(out))

    out["published_at"] = _datetime(r["published_at"])
    out["observed_at"] = _datetime(r["scraped_at"])
    out["first_seen_at"] = _datetime(r["first_seen_at"])
    out["last_seen_at"] = _datetime(r["last_seen_at"])
    logger.info(
        "normalise: %d listings, %d invalid room counts", len(out), out["rooms_invalid"].sum()
    )
    return out


def rent_scope(df: pd.DataFrame) -> pd.Series:
    """Find the listings outside the rental analysis.

    The domain rules (area, rent, rooms) follow later with :func:`rentml.data.domain_filter`.

    Args:
        df: Listings from :func:`normalise`.

    Returns:
        The first failing rule (``deal_type_buy``, ``price_not_monthly``,
        ``coordinates_missing``) per listing, ``<NA>`` for monthly rentals in scope.
    """
    rules = {
        "deal_type_buy": df["deal_type"].ne("rent"),
        "price_not_monthly": df["price_type"].ne(MONTHLY_RENT),
        "coordinates_missing": df[["east", "north"]].isna().any(axis=1),
    }
    return _first_reason(rules, df.index)


def buy_scope(df: pd.DataFrame) -> pd.Series:
    """Find the listings outside the purchase analysis.

    Args:
        df: Listings from :func:`normalise`.

    Returns:
        The first failing rule (``deal_type_rent``, ``price_not_total``, ``category_not_buy``,
        ``coordinates_missing``) per listing, ``<NA>`` for flats and houses for sale in scope.
    """
    rules = {
        "deal_type_rent": df["deal_type"].ne("buy"),
        "price_not_total": df["price_type"].ne(TOTAL_PRICE),
        "category_not_buy": ~df["property_category"].isin(BUY_CATEGORIES),
        "coordinates_missing": df[["east", "north"]].isna().any(axis=1),
    }
    return _first_reason(rules, df.index)


def buy_domain_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Domain rules for purchase objects (:data:`BUY_PRICE_RANGE`, :data:`BUY_AREA_RANGE`).

    Args:
        df: Purchase objects with ``area``, ``price`` and ``rooms``.

    Returns:
        ``(kept, removed)`` as in :func:`rentml.data.domain_filter`; prices below CHF 50'000
        are removed as ``price_below_min``.
    """
    return domain_filter(df, area_range=BUY_AREA_RANGE, price_range=BUY_PRICE_RANGE)


def _first_reason(rules: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    reason = np.select(list(rules.values()), list(rules), default="")
    return pd.Series(reason, index=index, dtype="string").replace("", pd.NA)


def linking_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare listings for :func:`rentml.data.link_duplicates` without over-merging.

    Centroid coordinates and addresses without a house number describe many buildings, so they
    are blanked. Managers reuse one project text for every flat of a new building, so the text
    only links listings with equal rooms and area bucket (:data:`LINK_AREA_BUCKET_M2`).
    Comparis has no slug; the EGID exists only after :func:`rentml.enrichment.combine`.

    Args:
        df: Listings from :func:`normalise`.

    Returns:
        A copy with the masked columns, an empty ``slug`` and ``egid`` as nullable integer.
    """
    out = df.copy()
    text = out["description"].where(out["description"].str.len() >= DESC_MIN_CHARS)
    rooms = out["rooms"].astype("string").fillna("na")
    bucket = (out["area"] / LINK_AREA_BUCKET_M2).round().astype("Int64").astype("string")
    out["description"] = text + " rooms " + rooms + " area " + bucket.fillna("na")
    out[["east", "north"]] = out[["east", "north"]].where(out["coord_precise"], axis=0)
    out["address"] = out["address"].where(out["street_number"].notna())
    out["slug"] = pd.Series(pd.NA, index=out.index, dtype="string")
    egid = out["egid"] if "egid" in out else pd.Series(pd.NA, index=out.index)
    out["egid"] = pd.to_numeric(egid, errors="coerce").astype("Int64")
    return out


BOOL_COLUMNS = (
    "aux_costs_known", "rooms_invalid", "coord_precise", "ppsqm_outlier", "year_built_period",
)  # fmt: skip
DATE_COLUMNS = ("available_from", "published_at", "observed_at", "first_seen_at", "last_seen_at")


def load_cleaned(table: str, path: Path = CLEANED_DB) -> pd.DataFrame:
    """Read a table of the cleaned database read-only, with the types SQLite cannot store.

    Args:
        table: ``listings``, ``rent``, ``buy`` or another table of :func:`write_db`.
        path: The cleaned database.

    Returns:
        The table, indexed by ``listing_id`` if it has one; flags as booleans, timestamps as
        datetimes.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        df = pd.read_sql(f'SELECT * FROM "{table}"', con)
    finally:
        con.close()
    for col in BOOL_COLUMNS:
        if col in df:
            df[col] = df[col].astype("boolean").fillna(False).astype(bool)
    for col in DATE_COLUMNS:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "egid" in df:
        df["egid"] = pd.to_numeric(df["egid"], errors="coerce").astype("Int64")
    return df.set_index("listing_id") if "listing_id" in df else df


def file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sql_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Categoricals as text and nullable integers/booleans as floats, which SQLite stores."""
    out = df.copy()
    for col in out:
        dtype = out[col].dtype
        if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dtype):
            out[col] = out[col].astype(object).where(out[col].notna(), None)
        elif pd.api.types.is_extension_array_dtype(dtype) and out[col].hasnans:
            out[col] = out[col].astype(float)
    return out


def write_db(tables: dict[str, pd.DataFrame], path: Path = CLEANED_DB) -> Path:
    """Write tables to a new SQLite file that replaces ``path`` atomically.

    A named index (``listing_id``) becomes a column with a unique SQL index.

    Args:
        tables: Table name -> frame.
        path: Target file.

    Returns:
        ``path``.

    Raises:
        ValueError: If ``path`` is the raw database.
    """
    if path.resolve() == RAW_DB.resolve():
        raise ValueError("Refusing to overwrite the raw Comparis database")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    try:
        for name, table in tables.items():
            keep_index = table.index.name is not None
            _sql_ready(table).to_sql(name, con, index=keep_index, chunksize=5000)
            if keep_index and table.index.is_unique:
                key = table.index.name
                con.execute(f'CREATE UNIQUE INDEX "idx_{name}_{key}" ON "{name}" ("{key}")')
        con.commit()
    finally:
        con.close()
    tmp.replace(path)
    logger.info("Wrote %s: %s", path, {name: len(t) for name, t in tables.items()})
    return path


def meta_table(raw_path: Path, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Describe the provenance of the cleaned database.

    Args:
        raw_path: The raw database the tables come from.
        tables: The tables that will be written.

    Returns:
        ``key``/``value`` rows: raw file, SHA-256, size, mtime, creation time and row counts.
    """
    stat = raw_path.stat()
    rows = {
        "raw_db": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "raw_bytes": str(stat.st_size),
        "raw_modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    rows |= {f"rows_{name}": str(len(t)) for name, t in tables.items()}
    return pd.DataFrame({"key": list(rows), "value": list(rows.values())})
