"""Loading and schema standardisation of the DSPRO1 rentumo.ch listings.

:func:`load_from_db` (read-only :data:`LISTINGS_SQL`, URL passed in by the caller) and
:func:`load_from_csv` (offline fallback, no descriptions) return the same canonical frame;
:func:`load_listings` picks the source, :func:`fix_schema` coerces types and flags values.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy import exc as sa_exc
from sqlalchemy.engine import URL, make_url

from rentml._anonymize import anonymize_descriptions
from rentml.config import SNAPSHOT_DATE

logger = logging.getLogger(__name__)

COLUMN_MAP: dict[str, str] = {
    "area_sqm": "area",
    "price_cold": "price",
    "oev_score": "oev",
    "solar_class": "solar",
    "elevation_m": "elevation",
    "lv95_east": "east",
    "lv95_north": "north",
    "gbauj": "year_built",
    "ganzwhg": "apartments",
    "garea": "land_area",
    "auxiliary_costs": "aux_costs",
    "kaution": "deposit",
}

CANONICAL_COLUMNS: tuple[str, ...] = (
    ("listing_id", "source", "slug", "address", "area", "rooms", "price", "aux_costs", "deposit")
    + ("available_from", "description", "population", "oev", "solar", "elevation", "east", "north")
    + ("egid", "year_built", "apartments", "land_area", "observed_at")
)
TEXT_COLUMNS: tuple[str, ...] = ("source", "slug", "address", "description")
DATE_COLUMNS: tuple[str, ...] = ("available_from", "observed_at")
# Every other canonical column except the ids is numeric (egid is cast to Int64 separately).
NUMERIC_COLUMNS: tuple[str, ...] = tuple(
    c for c in CANONICAL_COLUMNS if c not in {"listing_id", "egid", *TEXT_COLUMNS, *DATE_COLUMNS}
)
ROOMS_MIN = 1.0
ROOMS_MAX = 15.0
DEFAULT_SOURCE = "rentumo"
WIDE_CSV = "model_wide.csv"
ENRICHED_CSV = "final_listings.csv"

_WIDE_REQUIRED = ("listing_id", "area_sqm", "rooms", "price_cold", "lv95_east", "lv95_north")
_SWISSTOPO_COLUMNS = ("oev_score", "solar_class", "elevation_m")
_ENRICHMENT_COLUMNS = ("address", *_SWISSTOPO_COLUMNS, "egid", "gbauj", "ganzwhg", "garea")
# DSPRO1 completeness rule (final_records.ipynb, FINAL_DATASET_QUERY -> final_listings.csv,
# 4'536 rows): detail-page, swisstopo and GWR values all present, checked before gap filling.
_DSPRO1_REQUIRED = ("address", "area_sqm", "rooms", "price_cold", "population", "egid")
_DSPRO1_REQUIRED += ("gbauj", "ganzwhg", "garea", *_SWISSTOPO_COLUMNS)
# Pairs (preferred detail-page column, fallback column) that are coalesced after the DB export.
_DB_COALESCE = (("area_sqm", "area_listing"), ("rooms", "rooms_listing"))
_DB_COALESCE += (("price_cold", "price_listing"),)
# "Database not usable now" (outage, auth, timeout, no driver); query/schema errors such as
# ProgrammingError are absent on purpose (SQLite would report them as OperationalError).
_UNAVAILABLE_ERRORS = (sa_exc.OperationalError, sa_exc.InterfaceError, sa_exc.TimeoutError)
_UNAVAILABLE_ERRORS += (sa_exc.ArgumentError, ImportError)
_VALID_SOURCES = ("auto", "db", "csv")

# Read-only, one row per listing (PostgreSQL). Only columns of the scraper migrations / DSPRO1
# (no listing_details.year_built: GWR gbauj is the only year); search-page values fill gaps in
# Python. With several buildings the most complete GWR row wins, then the smallest EGID.
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


class DatabaseUnavailableError(RuntimeError):
    """Raised when the database is not usable now (bad URL, no driver, connection error).

    Only this subclass triggers the ``"auto"`` CSV fallback; a failing query is RuntimeError.
    """


def load_from_db(database_url: str, *, connect_timeout_s: int = 15) -> pd.DataFrame:
    """Export all listings from the DSPRO1 database with :data:`LISTINGS_SQL`.

    The connection is opened read-only (PostgreSQL) and disposed afterwards. The DSPRO1 schema
    has no observation timestamps, so ``observed_at`` is set to the snapshot date.

    Args:
        database_url: SQLAlchemy URL, e.g. ``postgresql://user:pw@host/db?sslmode=require``
            (``postgres://`` is rewritten to ``postgresql://``).
        connect_timeout_s: Connection timeout in seconds (PostgreSQL only).

    Returns:
        Standardised listings indexed by ``listing_id`` (same columns as :func:`load_from_csv`).

    Raises:
        ValueError: If ``database_url`` is empty.
        DatabaseUnavailableError: If the URL is unparsable, the driver is missing or the
            connection fails (password redacted in the message).
        RuntimeError: If the query fails, e.g. ``ProgrammingError`` for an unknown column.
    """
    if not database_url or not database_url.strip():
        raise ValueError("database_url must not be empty")
    url = database_url.strip()
    url = "postgresql://" + url[len("postgres://") :] if url.startswith("postgres://") else url
    try:
        parsed = make_url(url)
    except (sa_exc.ArgumentError, ValueError, TypeError):  # e.g. a non-numeric port
        raise DatabaseUnavailableError("Could not load listings: unparsable database URL") from None
    is_postgres = parsed.get_backend_name() == "postgresql"
    engine = None
    try:
        connect_args = {"connect_timeout": connect_timeout_s} if is_postgres else {}
        engine = create_engine(url, connect_args=connect_args, pool_pre_ping=True)
        with engine.connect() as conn:
            if is_postgres:
                conn = conn.execution_options(postgresql_readonly=True)
            raw = pd.read_sql(text(LISTINGS_SQL), conn)
    except (sa_exc.SQLAlchemyError, ImportError, pd.errors.DatabaseError) as exc:
        raise _database_error(exc, url, parsed) from None
    finally:
        if engine is not None:
            engine.dispose()
    logger.info("Loaded %d listings from the database", len(raw))
    return _prepare_db_frame(raw)


def load_from_csv(csv_dir: Path) -> pd.DataFrame:
    """Load the DSPRO1 CSV exports (offline fallback without descriptions).

    Base ``model_wide.csv``; the address/swisstopo/GWR columns of ``final_listings.csv`` are
    left-joined on ``listing_id`` (orphans dropped, logged). ``dspro1_enriched`` marks the
    DSPRO1 modelling set (4'522 of the 4'536 final_listings rows have a base row; DB: same rule).

    Args:
        csv_dir: Directory with ``model_wide.csv`` and ``final_listings.csv``.

    Returns:
        Standardised listings indexed by ``listing_id``.

    Raises:
        FileNotFoundError: If ``model_wide.csv`` is missing.
        KeyError: If ``model_wide.csv`` lacks a required column.
        pandas.errors.MergeError: If ``listing_id`` is not unique in one of the files.
    """
    wide_path = Path(csv_dir) / WIDE_CSV
    if not wide_path.is_file():
        raise FileNotFoundError(f"DSPRO1 CSV export not found: {wide_path}")
    base = pd.read_csv(wide_path)
    missing = [col for col in _WIDE_REQUIRED if col not in base.columns]
    if missing:
        raise KeyError(f"{wide_path.name} lacks required columns {missing}")
    merged = _join_enrichment(base, Path(csv_dir) / ENRICHED_CSV)
    merged = merged.assign(description=None, source=DEFAULT_SOURCE, observed_at=SNAPSHOT_DATE)
    n_enriched = int(merged["dspro1_enriched"].sum())
    logger.info("Loaded %d listings from %s (%d enriched)", len(merged), csv_dir, n_enriched)
    return _finalise(merged)


def load_listings(
    source: str = "auto",
    *,
    csv_dir: Path,
    database_url: str | None = None,
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load the listings from the database or the CSV exports.

    The DB returns every scraped listing (~10k, with texts), the CSV the 9'405 pre-filtered
    DSPRO1 rows without texts: set ``source`` or ``cache_path`` so re-runs see the same data.

    Args:
        source: ``"db"``, ``"csv"`` or ``"auto"`` (database if a URL is given and reachable,
            else CSV; only :class:`DatabaseUnavailableError` triggers the fallback).
        csv_dir: Directory with the DSPRO1 CSV exports.
        database_url: SQLAlchemy URL of the DSPRO1 database, e.g. from ``DATABASE_URL``.
        cache_path: Optional parquet file freezing the database export: if it exists it is
            returned for ``"db"``/``"auto"`` without querying, otherwise the first successful
            export is written to it. Delete the file to re-query.

    Returns:
        ``(listings, used_source)`` with ``used_source`` in ``{"db", "csv"}``; standardised,
        indexed by ``listing_id``, descriptions anonymised before they are returned or cached.

    Raises:
        ValueError: If ``source`` is unknown or ``"db"`` is requested without URL or cache.
        DatabaseUnavailableError: If ``source="db"`` and the database cannot be reached.
        RuntimeError: If the database query fails (also in ``"auto"``: no silent fallback).
    """
    mode = source.strip().lower()
    if mode not in _VALID_SOURCES:
        raise ValueError(f"source must be one of {_VALID_SOURCES}, got {source!r}")
    if mode == "csv":
        return load_from_csv(csv_dir), "csv"
    if cache_path is not None and Path(cache_path).is_file():
        logger.info("Using the cached database export %s (delete it to re-query)", cache_path)
        return anonymize_descriptions(pd.read_parquet(cache_path)), "db"  # also covers older caches
    if not database_url:
        if mode == "db":
            raise ValueError("source='db' requires a database_url or an existing cache_path")
        logger.info("No database URL given; loading the CSV exports from %s", csv_dir)
        return load_from_csv(csv_dir), "csv"
    try:
        listings = anonymize_descriptions(load_from_db(database_url))  # no raw contact data on disk
    except DatabaseUnavailableError as exc:
        if mode == "db":
            raise
        logger.warning("%s -- falling back to the CSV exports in %s", exc, csv_dir)
        return load_from_csv(csv_dir), "csv"
    if cache_path is not None:
        _write_cache(listings, Path(cache_path))
    return listings, "db"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DSPRO1/DB columns to canonical names and add missing canonical columns.

    Canonical columns come first (:data:`CANONICAL_COLUMNS` order), extras follow; missing
    ones are added as ``None``/``NaT``/``NaN``. ``available_from`` (ISO or ``dd.mm.yyyy``,
    other text -> ``NaT``) and ``observed_at`` become naive UTC datetimes in both loaders.
    ``listing_id`` is not added if it is missing or already the index.

    Args:
        df: Raw listings.

    Returns:
        A new frame with canonical column names.

    Raises:
        ValueError: If renaming produces duplicate column names (e.g. ``area`` and ``area_sqm``).
    """
    out = df.rename(columns=COLUMN_MAP)
    duplicated = sorted(set(out.columns[out.columns.duplicated()]))
    if duplicated:
        raise ValueError(f"Duplicate columns after renaming: {duplicated}")
    absent = [c for c in CANONICAL_COLUMNS if c not in out.columns and c != "listing_id"]
    out = out.assign(**{col: _empty_column(col, out.index) for col in absent})
    for col in DATE_COLUMNS:
        out[col] = _parse_dates(out[col])
    ordered = [col for col in CANONICAL_COLUMNS if col in out.columns]
    extras = [col for col in out.columns if col not in CANONICAL_COLUMNS]
    return out[ordered + extras]


def fix_schema(
    df: pd.DataFrame, *, rooms_min: float = ROOMS_MIN, rooms_max: float = ROOMS_MAX
) -> pd.DataFrame:
    """Coerce types, flag implausible room counts and add the log target.

    * numeric columns: ``pd.to_numeric`` after removing Swiss formatting (``1'850``,
      ``3,5``, ``3½``, ``CHF``, ``.–``); unparsable values become ``NaN`` (count logged);
    * ``egid`` as nullable ``Int64``; text columns stripped, blanks and NA as ``None``;
    * ``available_from`` (ISO or ``dd.mm.yyyy``) and ``observed_at`` as datetimes;
    * ``rooms`` as float; values outside ``[rooms_min, rooms_max]`` become ``NaN`` and are
      flagged in ``rooms_invalid`` (kept if already set, so the function is idempotent);
    * ``rooms_is_integer``: whole room count (``False`` if missing). DSPRO1 holds integers only
      (scraper parses integers; migration 20260413141619 cast rooms to INTEGER, rounding 3.5 ->
      4), so on DSPRO1 data it equals ``rooms.notna()``: no half-room signal;
    * ``log_price = log(price)`` for positive prices, otherwise ``NaN``.

    Args:
        df: Standardised listings (see :func:`standardize_columns`).
        rooms_min: Smallest plausible room count.
        rooms_max: Largest plausible room count.

    Returns:
        A fixed copy of ``df``.

    Raises:
        KeyError: If ``rooms`` or ``price`` is missing.
    """
    missing = [col for col in ("rooms", "price") if col not in df.columns]
    if missing:
        raise KeyError(f"fix_schema requires the columns {missing}")
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = _coerce_numeric(out[col], name=col)
    if "egid" in out.columns:
        out["egid"] = _coerce_numeric(out["egid"], name="egid").round().astype("Int64")
    for col in (*TEXT_COLUMNS, *DATE_COLUMNS):
        if col in out.columns:
            out[col] = _parse_dates(out[col]) if col in DATE_COLUMNS else _clean_text(out[col])
    invalid = out["rooms"].notna() & ~out["rooms"].between(rooms_min, rooms_max)
    previous = out["rooms_invalid"].eq(True) if "rooms_invalid" in out.columns else False
    out["rooms_invalid"] = invalid | previous
    out["rooms"] = out["rooms"].mask(invalid)
    out["rooms_is_integer"] = out["rooms"].notna() & (out["rooms"] % 1 == 0)
    price = out["price"]
    out["log_price"] = np.log(price.where(price > 0))
    n_no_price = int(out["log_price"].isna().sum())
    logger.info(
        "fix_schema: %d implausible rooms -> NaN, %d without price", invalid.sum(), n_no_price
    )
    return out


def audit_table(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise dtype, missingness and cardinality per column.

    Args:
        df: Any frame.

    Returns:
        One row per column (index ``column``) with ``dtype``, ``n_missing``, ``pct_missing``
        (0-100, ``NaN`` for an empty frame) and ``n_unique`` (non-missing distinct values).
    """
    n_rows = len(df)
    rows = []
    for position, col in enumerate(df.columns):
        series = df.iloc[:, position]
        n_missing = int(series.isna().sum())
        pct = 100.0 * n_missing / n_rows if n_rows else np.nan
        rows.append((str(col), str(series.dtype), n_missing, pct, _n_unique(series)))
    columns = ["column", "dtype", "n_missing", "pct_missing", "n_unique"]
    return pd.DataFrame(rows, columns=columns).set_index("column")


def text_coverage(
    df: pd.DataFrame, *, col: str = "description", min_chars: int = 50
) -> dict[str, float]:
    """Measure how many listings have a usable description.

    Args:
        df: Listings.
        col: Text column to audit.
        min_chars: Length threshold for ``coverage_min_chars``.

    Returns:
        ``n_total``, ``n_with_text``, ``coverage`` (share with non-blank text),
        ``coverage_min_chars`` (share with at least ``min_chars`` characters) and
        ``median_chars`` (over non-blank texts). If a ``text_lang`` column exists, also
        ``share_lang_<lang>`` among the non-blank texts. Shares are ``NaN`` for an empty frame.

    Raises:
        KeyError: If ``col`` is missing.
    """
    if col not in df.columns:
        raise KeyError(f"Column {col!r} not found")
    lengths = df[col].astype(object).where(df[col].notna(), "").astype(str).str.strip().str.len()
    has_text = lengths > 0
    n_total = len(df)
    result: dict[str, float] = {
        "n_total": float(n_total),
        "n_with_text": float(has_text.sum()),
        "coverage": float(has_text.mean()) if n_total else float("nan"),
        "coverage_min_chars": float((lengths >= min_chars).mean()) if n_total else float("nan"),
        "median_chars": float(lengths[has_text].median()) if has_text.any() else float("nan"),
    }
    if "text_lang" in df.columns and has_text.any():
        shares = df.loc[has_text, "text_lang"].fillna("unknown").value_counts(normalize=True)
        result.update({f"share_lang_{lang}": float(share) for lang, share in shares.items()})
    return result


def _join_enrichment(base: pd.DataFrame, enrich_path: Path) -> pd.DataFrame:
    """Left-join the enrichment columns of ``final_listings.csv`` onto the base table."""
    if not enrich_path.is_file():
        logger.warning("%s not found; enrichment columns stay empty", enrich_path)
        empty = dict.fromkeys(_ENRICHMENT_COLUMNS, np.nan)
        return base.assign(**empty, dspro1_enriched=False)
    wanted = {"listing_id", *_ENRICHMENT_COLUMNS}
    enrich = pd.read_csv(enrich_path, usecols=lambda col: col in wanted)
    absent = [col for col in _ENRICHMENT_COLUMNS if col not in enrich.columns]
    if absent:
        logger.warning("%s lacks enrichment columns %s", enrich_path.name, absent)
    n_orphans = int((~enrich["listing_id"].isin(base["listing_id"])).sum())
    if n_orphans:
        logger.info("%d rows of %s have no base row and are dropped", n_orphans, enrich_path.name)
    merged = base.merge(
        enrich, on="listing_id", how="left", validate="one_to_one", indicator="_merge"
    )
    merged["dspro1_enriched"] = merged["_merge"].eq("both")
    return merged.drop(columns="_merge")


def _prepare_db_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Flag the DSPRO1 set, coalesce the fallback columns and standardise the DB export."""
    out = raw.copy()
    # Before gap filling: DSPRO1 required the detail-page values themselves.
    out["dspro1_enriched"] = out.reindex(columns=list(_DSPRO1_REQUIRED)).notna().all(axis=1)
    for preferred, fallback in _DB_COALESCE:
        if fallback not in out.columns:
            continue
        backup = _coerce_numeric(out.pop(fallback), name=fallback)
        current = out[preferred] if preferred in out.columns else pd.Series(np.nan, out.index)
        out[preferred] = _coerce_numeric(current, name=preferred).fillna(backup)
    if "source" not in out.columns:
        out["source"] = DEFAULT_SOURCE
    if "observed_at" not in out.columns:
        out["observed_at"] = SNAPSHOT_DATE
    return _finalise(out)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns and index the frame by a unique integer ``listing_id``."""
    out = standardize_columns(df)
    if "listing_id" not in out.columns:
        raise KeyError("Listings have no 'listing_id' column")
    ids = pd.to_numeric(out["listing_id"], errors="coerce")
    if ids.isna().any():
        raise ValueError(f"{int(ids.isna().sum())} listings have a missing/non-numeric id")
    out["listing_id"] = ids.astype("int64")
    if out["listing_id"].duplicated().any():
        raise ValueError("listing_id is not unique")
    return out.set_index("listing_id")


def _empty_column(col: str, index: pd.Index) -> pd.Series:
    """Create an all-missing column with the dtype that fits the canonical column."""
    if col in TEXT_COLUMNS:
        return pd.Series([None] * len(index), index=index, dtype=object)
    if col in DATE_COLUMNS:
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    return pd.Series(np.nan, index=index, dtype=float)


def _coerce_numeric(series: pd.Series, *, name: str) -> pd.Series:
    """Convert a column to float, understanding Swiss number formatting."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"(?<=\d)\s*½", ".5", regex=True)
        .str.replace(r"^(?:CHF|Fr\.)\s*|\s*(?:CHF|Fr\.)$", "", regex=True)
        .str.replace(r"\.[-–—]+$", "", regex=True)
        .str.replace(r"[\s'’`]", "", regex=True)
        .str.replace(r",(?=\d{3}(?:\D|$))", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    result = pd.to_numeric(cleaned, errors="coerce").astype(float)
    n_lost = int((series.notna() & result.isna()).sum())
    if n_lost:
        logger.info("Column %r: %d unparsable values set to NaN", name, n_lost)
    return result


def _clean_text(series: pd.Series) -> pd.Series:
    """Strip text values; represent blanks and NA as ``None`` in an object column."""
    as_object = series.astype(object)
    present = as_object.notna()
    stripped = as_object.where(present, "").astype(str).str.strip()
    return stripped.astype(object).where(present & stripped.ne(""), None)


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse ISO dates and Swiss ``dd.mm.yyyy`` dates; everything else becomes ``NaT``."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
    as_text = series.astype("string").str.strip()
    iso = pd.to_datetime(as_text, format="ISO8601", errors="coerce", utc=True)
    swiss = pd.to_datetime(as_text, format="%d.%m.%Y", errors="coerce", utc=True)
    return iso.fillna(swiss).dt.tz_localize(None)


def _n_unique(series: pd.Series) -> int:
    """Count distinct non-missing values, also for unhashable cell values."""
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.dropna().astype(str).nunique())


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """Store the DB export as parquet; a failed write only costs the cache, not the data."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info("Cached the database export (%d rows) at %s", len(df), path)
    except (OSError, ValueError, TypeError) as exc:  # pyarrow errors subclass ValueError/TypeError
        logger.warning("Could not cache the database export at %s: %s", path, exc)


def _database_error(exc: BaseException, url: str, parsed: URL) -> RuntimeError:
    """Wrap a driver error (password redacted): outages as DatabaseUnavailableError."""
    safe_url = parsed.render_as_string(hide_password=True)
    message = f"{type(exc).__name__}: {exc}".splitlines()[0].replace(url, safe_url)
    if parsed.password:
        message = message.replace(str(parsed.password), "***")
    dropped = isinstance(exc, sa_exc.DBAPIError) and bool(exc.connection_invalidated)
    unavailable = dropped or isinstance(exc, _UNAVAILABLE_ERRORS)
    error_type = DatabaseUnavailableError if unavailable else RuntimeError
    return error_type(f"Could not load listings from database {safe_url}: {message}")
