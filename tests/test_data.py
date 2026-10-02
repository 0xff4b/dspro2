"""Tests for rentml.data (synthetic CSVs and a faked database; never the real DB)."""

import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError, ProgrammingError
from sqlalchemy.sql.elements import TextClause

from rentml import data
from rentml.config import SNAPSHOT_DATE, ProjectPaths
from rentml.data import (
    CANONICAL_COLUMNS,
    COLUMN_MAP,
    LISTINGS_SQL,
    DatabaseUnavailableError,
    audit_table,
    fix_schema,
    load_from_csv,
    load_from_db,
    load_listings,
    standardize_columns,
    text_coverage,
)

FAKE_URL = "postgresql://tester:s3cret-pw@db.invalid/rent?sslmode=require"


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    """Write small DSPRO1-style exports (same schema as the real files)."""
    wide = pd.DataFrame(
        {
            "listing_id": [1, 2, 3, 4, 5],
            "area_sqm": [34, 108, 81, 196, 60],
            "rooms": [2.0, 4.0, np.nan, 196.0, 3.0],
            "price_cold": [850, 1990, 2200, 2500, 1500],
            "lv95_east": [2687446.25, 2642156.0, 2534980.0, 2500269.75, 2600000.0],
            "lv95_north": [1248599.0, 1130267.625, 1153490.0, 1118953.375, 1200000.0],
            "population": [44.0, 107.0, 252.0, np.nan, 90.0],
        }
    )
    final = pd.DataFrame(
        {
            "listing_id": [1, 2, 99],
            "address": [
                "8044, Gockhausen-Zürich, Rütistrasse, 1",
                "3904, Naters, Bahnhofstr., 8",
                "x",
            ],
            "area_sqm": [34, 108, 50],
            "rooms": [2, 4, 2],
            "price_cold": [850, 1990, 1000],
            "population": [44, 107, 1],
            "oev_score": [2043.0, 5656.0, 1.0],
            "solar_class": [2.0, 3.0, 1.0],
            "elevation_m": [572.9, 672.3, 400.0],
            "lv95_east": [2687446.25, 2642156.0, 2600000.0],
            "lv95_north": [1248599.0, 1130267.625, 1200000.0],
            "egid": [93458, 892059, 1],
            "gbauj": [1971, 1974, 2000],
            "ganzwhg": [1, 34, 2],
            "garea": [192.0, 2000.0, 100.0],
        }
    )
    wide.to_csv(tmp_path / "model_wide.csv", index=False)
    final.to_csv(tmp_path / "final_listings.csv", index=False)
    return tmp_path


def _db_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "listing_id": [10, 11],
            "slug": ["a-10", "b-11"],
            "address": ["8000, Zürich, Seestrasse, 1", None],
            "area_sqm": [70.0, None],
            "area_listing": [71.0, 55.0],
            "rooms": ["3,5", None],
            "rooms_listing": [3, 2],
            "price_cold": [2100, None],
            "price_listing": [Decimal("2000.00"), Decimal("1450.00")],  # psycopg2 NUMERIC
            "auxiliary_costs": [200.0, None],
            "kaution": [6000.0, None],
            "available_from": ["2026-05-01", "sofort"],
            "description": ["Schöne 3.5-Zimmer-Wohnung mit Balkon", ""],
            "population": [300, 20],
            "oev_score": [9000.0, 100.0],
            "solar_class": [3.0, 2.0],
            "elevation_m": [410.0, 800.0],
            "lv95_east": [2683000.0, 2600000.0],
            "lv95_north": [1247000.0, 1200000.0],
            "egid": [123, None],
            "gbauj": [None, 1955],
            "ganzwhg": [12, 3],
            "garea": [800.0, 300.0],
        }
    )


class _FakeConnection:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def __enter__(self) -> "_FakeConnection":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execution_options(self, **options: object) -> "_FakeConnection":
        self.options.update(options)
        return self


class _FakeEngine:
    def __init__(self) -> None:
        self.connection = _FakeConnection()
        self.disposed = False

    def connect(self) -> _FakeConnection:
        return self.connection

    def dispose(self) -> None:
        self.disposed = True


def _patch_db(monkeypatch: pytest.MonkeyPatch, outcome: pd.DataFrame | Exception) -> None:
    """Fake the engine; ``pd.read_sql`` returns ``outcome`` or raises it."""

    def fake_read_sql(sql: TextClause, con: _FakeConnection) -> pd.DataFrame:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(data, "create_engine", lambda url, **kwargs: _FakeEngine())
    monkeypatch.setattr(data.pd, "read_sql", fake_read_sql)


def test_column_map_covers_dspro1_names() -> None:
    assert COLUMN_MAP["price_cold"] == "price"
    assert COLUMN_MAP["lv95_east"] == "east"
    assert set(COLUMN_MAP.values()) <= set(CANONICAL_COLUMNS)


def test_listings_sql_is_read_only_single_row_query() -> None:
    sql = LISTINGS_SQL.upper()
    assert "DISTINCT ON (L.ID)" in sql
    for forbidden in ("INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER "):
        assert forbidden not in sql
    assert "YEAR_BUILT" not in sql  # listing_details has no such column


def test_listings_sql_matches_scraper_migrations() -> None:
    pipelines = ProjectPaths.discover(Path(__file__).parent).pipelines
    files = sorted((pipelines / "rentables-scraper" / "migrations").glob("*.sql"))
    if not files:
        pytest.skip("Scraper migrations not available")
    ddl = "\n".join(file.read_text(encoding="utf-8") for file in files)
    tables = {
        name: set(re.findall(r"^\s+(\w+)\s+[A-Z]", body, flags=re.MULTILINE))
        for name, body in re.findall(r"CREATE TABLE (\w+) \((.*?)\);", ddl, flags=re.DOTALL)
    }
    for alias, table in (("l", "listings"), ("ld", "listing_details")):
        used = set(re.findall(rf"\b{alias}\.(\w+)", LISTINGS_SQL))
        assert used and used <= tables[table], used - tables[table]


def test_load_from_csv_joins_enrichment(csv_dir: Path) -> None:
    df = load_from_csv(csv_dir)
    assert df.index.name == "listing_id"
    assert list(df.index) == [1, 2, 3, 4, 5]  # orphan enrichment row 99 dropped
    assert list(df.columns[: len(CANONICAL_COLUMNS) - 1]) == list(CANONICAL_COLUMNS[1:])
    assert df.loc[1, "year_built"] == 1971
    assert np.isnan(df.loc[3, "oev"])
    assert df["dspro1_enriched"].tolist() == [True, True, False, False, False]
    assert (df["source"] == "rentumo").all()
    assert df["description"].isna().all()
    assert (df["observed_at"] == pd.Timestamp(SNAPSHOT_DATE)).all()


def test_load_from_csv_without_enrichment_file(csv_dir: Path) -> None:
    (csv_dir / "final_listings.csv").unlink()
    df = load_from_csv(csv_dir)
    assert len(df) == 5
    assert df["egid"].isna().all()
    assert not df["dspro1_enriched"].any()


def test_load_from_csv_missing_base_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_from_csv(tmp_path)


def test_load_from_csv_duplicate_ids_raise(csv_dir: Path) -> None:
    wide = pd.read_csv(csv_dir / "model_wide.csv")
    pd.concat([wide, wide.head(1)]).to_csv(csv_dir / "model_wide.csv", index=False)
    with pytest.raises(pd.errors.MergeError):
        load_from_csv(csv_dir)


def test_load_from_db_uses_readonly_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _FakeEngine()
    seen: dict[str, object] = {}

    def fake_create_engine(url: str, **kwargs: object) -> _FakeEngine:
        seen["url"], seen["kwargs"] = url, kwargs
        return engine

    def fake_read_sql(sql: TextClause, con: _FakeConnection) -> pd.DataFrame:
        assert isinstance(sql, TextClause) and "FROM listings" in str(sql)
        return _db_frame()

    monkeypatch.setattr(data, "create_engine", fake_create_engine)
    monkeypatch.setattr(data.pd, "read_sql", fake_read_sql)
    df = load_from_db(FAKE_URL.replace("postgresql://", "postgres://"))
    assert seen["url"] == FAKE_URL
    assert seen["kwargs"]["connect_args"] == {"connect_timeout": 15}
    assert engine.connection.options == {"postgresql_readonly": True}
    assert engine.disposed
    assert df.loc[10, "area"] == 70.0 and df.loc[11, "area"] == 55.0
    assert df.loc[11, "price"] == 1450 and df.loc[11, "year_built"] == 1955
    assert df.loc[10, "aux_costs"] == 200.0 and df.loc[10, "deposit"] == 6000.0
    assert "area_listing" not in df.columns and "rooms_listing" not in df.columns
    assert fix_schema(df).loc[10, "rooms"] == 3.5


def test_load_from_db_error_is_runtime_error_without_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_create_engine(url: str, **kwargs: object) -> _FakeEngine:
        raise OperationalError("SELECT 1", {}, Exception(f"could not connect to {url}"))

    monkeypatch.setattr(data, "create_engine", failing_create_engine)
    with pytest.raises(DatabaseUnavailableError) as info:
        load_from_db(FAKE_URL)
    assert "s3cret-pw" not in str(info.value)
    assert "db.invalid" in str(info.value)
    assert info.value.__cause__ is None


@pytest.mark.parametrize(
    ("error", "unavailable"),
    [
        (OperationalError("SELECT", {}, Exception("server closed the connection")), True),
        (InterfaceError("SELECT", {}, Exception("connection already closed")), True),
        (DBAPIError("SELECT", {}, Exception("reset"), connection_invalidated=True), True),
        (ProgrammingError("SELECT", {}, Exception("column ld.year_built does not exist")), False),
        (DBAPIError("SELECT", {}, Exception("division by zero")), False),
    ],
    ids=["operational", "interface", "invalidated", "programming", "other-dbapi"],
)
def test_load_from_db_separates_outages_from_query_errors(
    monkeypatch: pytest.MonkeyPatch, error: Exception, unavailable: bool
) -> None:
    _patch_db(monkeypatch, error)
    with pytest.raises(RuntimeError) as info:
        load_from_db(FAKE_URL)
    assert isinstance(info.value, DatabaseUnavailableError) is unavailable
    assert "s3cret-pw" not in str(info.value) and "db.invalid" in str(info.value)


def test_load_from_db_missing_driver_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_driver(url: str, **kwargs: object) -> _FakeEngine:
        raise ModuleNotFoundError("No module named 'psycopg2'")

    monkeypatch.setattr(data, "create_engine", no_driver)
    with pytest.raises(DatabaseUnavailableError, match="psycopg2"):
        load_from_db(FAKE_URL)


def test_load_from_db_rejects_bad_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    def must_not_connect(url: str, **kwargs: object) -> _FakeEngine:
        raise AssertionError("create_engine must not be reached")

    monkeypatch.setattr(data, "create_engine", must_not_connect)
    with pytest.raises(ValueError):
        load_from_db("  ")
    with pytest.raises(DatabaseUnavailableError, match="unparsable") as info:
        load_from_db("not a url with s3cret")
    assert "s3cret" not in str(info.value)
    with pytest.raises(DatabaseUnavailableError, match="unparsable") as info:
        load_from_db("postgresql://u:s3cret@host:abc/db")  # make_url raises ValueError
    assert "s3cret" not in str(info.value)


def test_load_listings_auto_falls_back_to_csv(
    csv_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def unreachable(url: str) -> pd.DataFrame:
        raise DatabaseUnavailableError("Could not load listings from database db.invalid")

    monkeypatch.setattr(data, "load_from_db", unreachable)
    df, used = load_listings("auto", csv_dir=csv_dir, database_url=FAKE_URL)
    assert used == "csv" and len(df) == 5
    assert "falling back" in caplog.text
    with pytest.raises(DatabaseUnavailableError):
        load_listings("db", csv_dir=csv_dir, database_url=FAKE_URL)


@pytest.mark.parametrize(
    "database_url", [FAKE_URL, "postgresql://u:s3cret@host:abc/db"], ids=["outage", "bad-port"]
)
def test_load_listings_auto_falls_back_end_to_end(
    csv_dir: Path, monkeypatch: pytest.MonkeyPatch, database_url: str
) -> None:
    _patch_db(monkeypatch, OperationalError("SELECT", {}, Exception("timeout expired")))
    df, used = load_listings("auto", csv_dir=csv_dir, database_url=database_url)
    assert used == "csv" and list(df.index) == [1, 2, 3, 4, 5]


def test_load_listings_auto_raises_query_errors(
    csv_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    error = ProgrammingError("SELECT", {}, Exception("column ld.year_built does not exist"))
    _patch_db(monkeypatch, error)
    with pytest.raises(RuntimeError, match="year_built") as info:
        load_listings("auto", csv_dir=csv_dir, database_url=FAKE_URL)
    assert not isinstance(info.value, DatabaseUnavailableError)
    assert "falling back" not in caplog.text


def test_load_listings_cache_freezes_db_export(
    csv_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "interim" / "listings_db.parquet"
    _patch_db(monkeypatch, _db_frame())
    first, used = load_listings("auto", csv_dir=csv_dir, database_url=FAKE_URL, cache_path=cache)
    assert used == "db" and cache.is_file()
    _patch_db(monkeypatch, AssertionError("the cache must be used"))
    second, used_again = load_listings("db", csv_dir=csv_dir, cache_path=cache)
    assert used_again == "db"
    pd.testing.assert_frame_equal(first, second)


def test_load_listings_anonymises_descriptions_before_caching(
    csv_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _db_frame()
    raw.loc[0, "description"] = "Mietzins CHF 2'100.- Kontakt: Herr Hans Muster, 079 123 45 67"
    _patch_db(monkeypatch, raw)
    cache = tmp_path / "listings_db.parquet"
    df, _ = load_listings("db", csv_dir=csv_dir, database_url=FAKE_URL, cache_path=cache)
    stored = pd.read_parquet(cache)["description"]
    for text in (df.loc[10, "description"], stored.loc[10]):
        assert "Muster" not in text and "079" not in text and "2'100" not in text
        assert "[NAME]" in text and "[PHONE]" in text and "[PRICE]" in text
    assert df.loc[11, "description"] is None  # blank text stays missing


def test_load_listings_anonymises_an_older_raw_cache(csv_dir: Path, tmp_path: Path) -> None:
    cache = tmp_path / "listings_db.parquet"
    raw = data._prepare_db_frame(_db_frame())
    raw.loc[10, "description"] = "Kontakt: anna.muster@example.ch"
    raw.to_parquet(cache)  # written before descriptions were anonymised at ingestion
    df, used = load_listings("db", csv_dir=csv_dir, cache_path=cache)
    assert used == "db" and df.loc[10, "description"] == "Kontakt: [EMAIL]"


def test_load_listings_cache_write_failure_keeps_data(
    csv_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_db(monkeypatch, _db_frame())
    df, used = load_listings(csv_dir=csv_dir, database_url=FAKE_URL, cache_path=tmp_path)
    assert used == "db" and len(df) == 2
    assert "Could not cache" in caplog.text


def test_loaders_return_same_columns_and_dtypes(csv_dir: Path) -> None:
    from_csv, from_db = load_from_csv(csv_dir), data._prepare_db_frame(_db_frame())
    assert list(from_csv.columns) == list(from_db.columns)
    assert from_csv["available_from"].dtype == from_db["available_from"].dtype
    assert from_db.loc[10, "available_from"] == pd.Timestamp("2026-05-01")
    assert pd.isna(from_db.loc[11, "available_from"])  # "sofort"
    pd.testing.assert_series_equal(fix_schema(from_csv).dtypes, fix_schema(from_db).dtypes)


def test_db_frame_flags_dspro1_set_before_gap_filling() -> None:
    raw = _db_frame()
    raw.loc[0, "gbauj"] = 1990  # listing 10: every DSPRO1 field present
    raw.loc[1, ["address", "rooms", "price_cold", "egid"]] = ["x", "2", 1450, 5]
    df = data._prepare_db_frame(raw)  # listing 11 lacks only the detail-page area
    assert df["dspro1_enriched"].tolist() == [True, False]
    assert df.loc[11, "area"] == 55.0  # still filled from the search page
    assert not data._prepare_db_frame(_db_frame())["dspro1_enriched"].any()


def test_load_listings_modes(csv_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data, "load_from_db", lambda url: data._prepare_db_frame(_db_frame()))
    df_db, used_db = load_listings("DB", csv_dir=csv_dir, database_url=FAKE_URL)
    assert used_db == "db" and list(df_db.index) == [10, 11]
    _, used_auto = load_listings(csv_dir=csv_dir, database_url=None)
    assert used_auto == "csv"
    with pytest.raises(ValueError):
        load_listings("db", csv_dir=csv_dir)
    with pytest.raises(ValueError):
        load_listings("parquet", csv_dir=csv_dir)


def test_standardize_columns_renames_orders_and_fills() -> None:
    raw = pd.DataFrame(
        {"extra": [1], "price_cold": [1000], "listing_id": [7], "observed_at": ["2026-04-13"]}
    )
    out = standardize_columns(raw)
    assert list(out.columns) == [*CANONICAL_COLUMNS, "extra"]
    assert out["description"].iloc[0] is None
    assert pd.api.types.is_datetime64_dtype(out["observed_at"])
    assert out["price"].iloc[0] == 1000


def test_standardize_columns_converts_timezones_and_rejects_duplicates() -> None:
    aware = pd.DataFrame({"observed_at": pd.to_datetime(["2026-04-13T02:00:00+02:00"])})
    assert standardize_columns(aware)["observed_at"].iloc[0] == pd.Timestamp("2026-04-13")
    with pytest.raises(ValueError, match="Duplicate"):
        standardize_columns(pd.DataFrame({"area": [1], "area_sqm": [2]}))


def test_fix_schema_flags_rooms_and_adds_log_price() -> None:
    df = standardize_columns(
        pd.DataFrame(
            {
                "rooms": ["3,5", "196", None, "0.5", "4"],
                "price_cold": ["1'850.–", "2100", "0", "CHF 900", "abc"],
                "egid": [1.0, None, 3.0, 4.0, 5.0],
                "description": ["  Balkon ", "", None, "x", np.nan],
                "available_from": ["2026-05-01", "01.06.2026", "sofort", None, None],
            }
        )
    )
    out = fix_schema(df)
    assert out["rooms"].tolist()[0] == 3.5
    assert out["rooms_invalid"].tolist() == [False, True, False, True, False]
    assert out["rooms_is_integer"].tolist() == [False, False, False, False, True]
    assert out["price"].tolist()[:4] == [1850.0, 2100.0, 0.0, 900.0]
    assert np.isnan(out["log_price"].iloc[2]) and np.isnan(out["price"].iloc[4])
    assert out["log_price"].iloc[1] == pytest.approx(np.log(2100))
    assert str(out["egid"].dtype) == "Int64"
    assert out["description"].tolist() == ["Balkon", None, None, "x", None]
    assert out["available_from"].iloc[1] == pd.Timestamp("2026-06-01")
    assert pd.isna(out["available_from"].iloc[2])
    assert fix_schema(out)["rooms_invalid"].tolist() == out["rooms_invalid"].tolist()


def test_fix_schema_requires_rooms_and_price() -> None:
    with pytest.raises(KeyError):
        fix_schema(pd.DataFrame({"rooms": [1.0]}))


def test_audit_table_reports_missingness() -> None:
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 3.0], "b": [None, None, "x", "y"]})
    audit = audit_table(df)
    assert audit.loc["a", "n_missing"] == 1 and audit.loc["a", "n_unique"] == 2
    assert audit.loc["b", "pct_missing"] == pytest.approx(50.0)
    assert list(audit.columns) == ["dtype", "n_missing", "pct_missing", "n_unique"]


def test_audit_table_empty_frame() -> None:
    audit = audit_table(pd.DataFrame({"a": pd.Series(dtype=float)}))
    assert audit.loc["a", "n_missing"] == 0 and np.isnan(audit.loc["a", "pct_missing"])


def test_text_coverage_counts_non_blank_texts() -> None:
    df = pd.DataFrame(
        {
            "description": ["x" * 60, "short", "   ", None],
            "text_lang": ["de", "fr", "unknown", None],
        }
    )
    cov = text_coverage(df)
    assert cov["n_with_text"] == 2 and cov["coverage"] == pytest.approx(0.5)
    assert cov["coverage_min_chars"] == pytest.approx(0.25)
    assert cov["share_lang_de"] == pytest.approx(0.5)


def test_text_coverage_edge_cases() -> None:
    empty = text_coverage(pd.DataFrame({"description": pd.Series(dtype=object)}))
    assert empty["n_total"] == 0 and np.isnan(empty["coverage"])
    with pytest.raises(KeyError):
        text_coverage(pd.DataFrame({"other": [1]}))


def test_real_dspro1_csv_smoke() -> None:
    csv_dir = ProjectPaths.discover(Path(__file__).parent).dspro1_csv_dir
    if not (csv_dir / "model_wide.csv").is_file():
        pytest.skip("DSPRO1 CSV exports not available")
    df = fix_schema(load_from_csv(csv_dir))
    assert df.index.is_unique and len(df) > 9000
    assert int(df["rooms_invalid"].sum()) >= 1  # e.g. listing with 196 rooms
    assert df["dspro1_enriched"].sum() > 4000
    assert df["log_price"].notna().all()
