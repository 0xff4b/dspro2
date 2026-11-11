"""Tests for rentml.comparis (a synthetic SQLite file, never the real Comparis database)."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rentml import comparis, data

TEXT = "Helle Wohnung mit Balkon und Seesicht, ruhige Lage. Auskunft unter 079 123 45 67."
BASE = {
    "url": "https://example.org/1", "partner": "homegate.ch", "deal_type": "rent",
    "property_type": "Wohnung", "title": "Wohnung", "description": TEXT, "price": 2100,
    "price_net": 1900, "side_costs": 200, "rooms": 3.5, "living_area": 80.0, "usable_area": None,
    "plot_area": 1.0, "floor": "2. Stock", "number_of_floors": 0, "year_built": 1985,
    "renovation_year": 0, "balcony_count": 1, "pets_allowed": 1, "image_count": 5,
    "available_from": "01.12.2026", "street": "Bahnhofstrasse", "street_number": "1",
    "zip": "8001", "city": "Zürich", "latitude": 47.3769, "longitude": 8.5417,
    "published_at": "2026-10-04T09:28:43", "scraped_at": "2026-10-26T20:51:11+00:00",
    "first_seen_at": "2026-10-26T20:54:12+00:00", "last_seen_at": "2026-10-29T15:18:37+00:00",
    "features_json": json.dumps({"HasLift": True, "HasCableTV": True}),
    "attributes_json": json.dumps({"DistanceBus": "200 m", "IsQuiet": True}),
}  # fmt: skip
ROWS = [
    {"id": "1"},
    {"id": "2", "price_type": "Mietpreis pro Woche"},
    {"id": "3", "deal_type": "buy", "price_type": "Kaufpreis", "price": 950_000,
     "price_net": 950_000, "side_costs": None},
    {"id": "4", "latitude": -23.3, "longitude": 168.86, "rooms": 999.0, "living_area": 1.0,
     "year_built": 9999, "floor": "EG", "available_from": "sofort", "side_costs": None,
     "price": 1900, "pets_allowed": None, "features_json": "{}", "attributes_json": "{}"},
]  # fmt: skip


@pytest.fixture
def raw_db(tmp_path: Path) -> Path:
    """A raw database with the apify_listings columns that load_raw reads."""
    rows = []
    for row in ROWS:
        record = BASE | {"price_type": "Mietpreis pro Monat"} | row
        record["raw_json"] = json.dumps({"priceType": record.pop("price_type")})
        rows.append(record)
    path = tmp_path / "comparis.db"
    with sqlite3.connect(path) as con:
        pd.DataFrame(rows).to_sql("apify_listings", con, index=False)
    return path


@pytest.fixture
def listings(raw_db: Path) -> pd.DataFrame:
    return comparis.normalise(comparis.load_raw(raw_db))


def test_load_raw_extracts_price_type(raw_db: Path) -> None:
    raw = comparis.load_raw(raw_db)
    assert raw["price_type"].tolist()[:3] == [
        "Mietpreis pro Monat",
        "Mietpreis pro Woche",
        "Kaufpreis",
    ]


def test_load_raw_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        comparis.load_raw(tmp_path / "missing.db")
    assert not (tmp_path / "missing.db").exists()


def test_normalise_uses_net_rent_and_flags_aux_costs(listings: pd.DataFrame) -> None:
    assert listings.loc[1, ["price", "gross_rent", "aux_costs"]].tolist() == [1900, 2100, 200]
    assert listings.loc[1, "aux_costs_known"]
    assert not listings.loc[4, "aux_costs_known"]
    assert listings.loc[3, "price"] == 950_000
    assert np.isnan(listings.loc[3, "gross_rent"])


def test_normalise_replaces_placeholders(listings: pd.DataFrame) -> None:
    row = listings.loc[4]
    assert row["rooms_invalid"]
    assert np.isnan(row["rooms"]) and np.isnan(row["area"]) and np.isnan(row["year_built"])
    assert np.isnan(row["east"]) and np.isnan(row["latitude"])
    assert row["floor"] == 0 and row["available_label"] == "immediately"
    assert pd.isna(row["pets_allowed"])
    assert np.isnan(listings.loc[1, "land_area"])  # "1 m²"
    assert np.isnan(listings.loc[1, "renovation_year"])


def test_normalise_converts_and_anonymises(listings: pd.DataFrame) -> None:
    row = listings.loc[1]
    assert row["east"] == pytest.approx(2_683_000, abs=1_000)
    assert row["north"] == pytest.approx(1_248_000, abs=1_000)
    assert "[PHONE]" in row["description"] and "079" not in row["description"]
    assert row["address"] == "Bahnhofstrasse 1, 8001 Zürich"
    assert row["available_from"] == pd.Timestamp("2026-12-01")
    assert row["floor"] == 2 and row["pets_allowed"] == 1
    assert row["observed_at"] == pd.Timestamp("2026-10-26 20:51:11")


def test_parse_floor_labels() -> None:
    labels = pd.Series(
        ["EG", "Untergeschoss", "3. Stock", "4", "30. Stock oder höher", "n/a", None]
    )
    expected = [0.0, -1.0, 3.0, 4.0, 30.0, np.nan, np.nan]
    np.testing.assert_array_equal(comparis.parse_floor(labels).to_numpy(), expected)


def test_expand_features(raw_db: Path) -> None:
    features = comparis.expand_features(comparis.load_raw(raw_db))
    assert features.loc[1, ["has_lift", "has_cable_tv", "is_quiet"]].tolist() == [1, 1, 1]
    assert features.loc[4, "has_lift"] == 0
    assert features.loc[1, "dist_bus_m"] == 200
    assert np.isnan(features.loc[1, "dist_train_m"])


def test_coordinate_precision_detects_centroids() -> None:
    df = pd.DataFrame({
        "east": [1.0, 1.0, 1.0, 2.0, 2.0, np.nan],
        "north": [1.0, 1.0, 1.0, 2.0, 2.0, np.nan],
        "street": ["A-Weg", "B-Weg", "C-Weg", "D-Weg", "D-Weg", "E-Weg"],
    })  # fmt: skip
    out = comparis.coordinate_precision(df)
    assert out["coord_precise"].tolist() == [False, False, False, True, True, False]
    assert out.loc[0, "coord_streets"] == 3 and out.loc[3, "coord_listings"] == 2


def test_rent_scope_reasons(listings: pd.DataFrame) -> None:
    reason = comparis.rent_scope(listings)
    assert pd.isna(reason[1])
    assert reason[[2, 3, 4]].tolist() == [
        "price_not_monthly",
        "deal_type_buy",
        "coordinates_missing",
    ]


def test_linking_frame_text_needs_equal_rooms() -> None:
    df = pd.DataFrame(
        {
            "description": [TEXT] * 3, "rooms": [2.5, 3.5, 3.5], "area": [60.0, 80.0, 81.0],
            "price": [1500.0, 1900.0, 1950.0], "east": [1.0, 2.0, 3.0], "north": [1.0, 2.0, 3.0],
            "address": ["X 1", "X 2", "X 3"], "street_number": [pd.NA, "2", "3"],
            "coord_precise": [False, True, True],
        },
        index=pd.Index([10, 11, 12], name="listing_id"),
    )  # fmt: skip
    frame = comparis.linking_frame(df)
    assert frame.loc[10, ["east", "north"]].isna().all() and pd.isna(frame.loc[10, "address"])
    links = data.link_duplicates(frame)
    assert set(links["rule"]) == {"description"}
    assert links[["a", "b"]].values.tolist() == [[11, 12]]


def test_write_db_roundtrip_and_meta(tmp_path: Path, raw_db: Path) -> None:
    table = pd.DataFrame(
        {"x": pd.array([1, None], dtype="Int64"), "s": pd.array(["a", None], dtype="string")},
        index=pd.Index([5, 6], name="listing_id"),
    )
    tables = {"listings": table}
    tables["meta"] = comparis.meta_table(raw_db, tables)
    path = comparis.write_db(tables, tmp_path / "out" / "clean.db")
    with sqlite3.connect(path) as con:
        back = pd.read_sql("SELECT * FROM listings", con)
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
    assert back["listing_id"].tolist() == [5, 6] and back["s"].tolist() == ["a", None]
    assert meta["raw_sha256"] == comparis.file_sha256(raw_db) and meta["rows_listings"] == "2"
    assert not (tmp_path / "out" / "clean.db.tmp").exists()


def test_write_db_refuses_raw_path(monkeypatch: pytest.MonkeyPatch, raw_db: Path) -> None:
    monkeypatch.setattr(comparis, "RAW_DB", raw_db)
    with pytest.raises(ValueError, match="raw"):
        comparis.write_db({"t": pd.DataFrame({"a": [1]})}, raw_db)


def test_buy_scope_reasons(listings: pd.DataFrame) -> None:
    reason = comparis.buy_scope(listings)
    assert pd.isna(reason[3])
    assert reason[[1, 2]].tolist() == ["deal_type_rent", "deal_type_rent"]
    flat = listings.loc[[3]].assign(price_type="Kaufpreis pro m²")
    room = listings.loc[[3]].assign(property_category="room")
    assert comparis.buy_scope(flat).iloc[0] == "price_not_total"
    assert comparis.buy_scope(room).iloc[0] == "category_not_buy"


def test_buy_domain_filter_drops_symbolic_prices() -> None:
    frame = pd.DataFrame({
        "area": [120.0, 120.0, 120.0, 3000.0],
        "price": [49_999.0, 50_000.0, 987_654_336.0, 900_000.0],
        "rooms": [4.5, 4.5, 4.5, 4.5],
    })  # fmt: skip
    kept, removed = comparis.buy_domain_filter(frame)
    assert kept.index.tolist() == [1]
    assert removed["reason"].tolist() == ["price_below_min", "price_above_max", "area_above_max"]


def test_linking_frame_keeps_enriched_egid() -> None:
    df = pd.DataFrame(
        {
            "description": [None, None], "rooms": [3.5, 3.5], "area": [80.0, 81.0],
            "price": [1900.0, 1950.0], "east": [1.0, 500.0], "north": [1.0, 500.0],
            "address": [None, None], "street_number": [pd.NA, pd.NA],
            "coord_precise": [False, False], "egid": [11, 11],
        },
        index=pd.Index([20, 21], name="listing_id"),
    )  # fmt: skip
    links = data.link_duplicates(comparis.linking_frame(df))
    assert links[["a", "b", "rule"]].values.tolist() == [[20, 21, "egid"]]


def test_load_cleaned_restores_types(tmp_path: Path, listings: pd.DataFrame) -> None:
    path = tmp_path / "cleaned.db"
    rent = listings.assign(ppsqm_outlier=[True, False, False, False], egid=[11, None, None, None])
    comparis.write_db({"rent": rent}, path)
    loaded = comparis.load_cleaned("rent", path)
    assert loaded.index.name == "listing_id" and len(loaded) == len(rent)
    assert loaded["coord_precise"].dtype == bool and loaded["ppsqm_outlier"].tolist()[0]
    assert pd.api.types.is_datetime64_any_dtype(loaded["observed_at"])
    assert loaded["egid"].dtype == "Int64" and loaded.loc[1, "egid"] == 11


def test_load_cleaned_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        comparis.load_cleaned("rent", tmp_path / "missing.db")
