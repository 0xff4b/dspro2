"""Tests for rentml.data (synthetic CSVs and a faked database query, never the real DB)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.exc import OperationalError

from rentml import data

LONG_TEXT = "Helle Wohnung mit Balkon und Seesicht, ruhige Lage, nahe Bahnhof und Schulen."


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """DSPRO1-style CSV exports in a temporary snapshot dir; frozen export path redirected."""
    wide = pd.DataFrame({
        "listing_id": [1, 2, 3], "area_sqm": [34, 108, 81], "rooms": [2.0, 4.0, 196.0],
        "price_cold": [850, 1990, 2200], "lv95_east": [2687446.0, 2642156.0, 2534980.0],
        "lv95_north": [1248599.0, 1130267.0, 1153490.0], "population": [44.0, 107.0, 252.0],
    })  # fmt: skip
    final = pd.DataFrame({col: [1.0, 2.0] for col in data._ENRICHMENT}).assign(
        listing_id=[1, 99], address=["8044, Zürich, Rütistrasse, 1", "x"], area_sqm=[34, 50]
    )
    wide.to_csv(tmp_path / "model_wide.csv", index=False)
    final.to_csv(tmp_path / "final_listings.csv", index=False)
    monkeypatch.setattr(data, "SNAPSHOT_DIR", tmp_path)
    monkeypatch.setattr(data, "DB_EXPORT", tmp_path / "interim" / "listings_db.parquet")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    return tmp_path


def test_load_listings_csv_has_canonical_schema(snapshot: Path) -> None:
    listings, used = data.load_listings("csv")
    assert used == "csv"
    assert listings.index.name == "listing_id"
    assert list(listings.columns[: len(data.CANONICAL_COLUMNS) - 1]) == data.CANONICAL_COLUMNS[1:]
    assert listings["dspro1_enriched"].tolist() == [True, False, False]
    assert listings.loc[1, "address"].startswith("8044")
    assert listings["description"].isna().all()


def test_load_listings_auto_falls_back_to_csv(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert data.load_listings("auto")[1] == "csv"  # no URL

    def unreachable(url: str) -> pd.DataFrame:
        raise OperationalError("SELECT", {}, ConnectionError("timeout"))

    monkeypatch.setattr(data, "_query_db", unreachable)
    assert data.load_listings("auto", database_url="postgresql://u@h/db")[1] == "csv"
    with pytest.raises(OperationalError):
        data.load_listings("db", database_url="postgresql://u@h/db")
    with pytest.raises(ValueError, match="DATABASE_URL"):
        data.load_listings("db")


def test_load_listings_db_freezes_anonymised_export(
    snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = pd.DataFrame(
        {
            "listing_id": [10, 11],
            "slug": ["a", "b"],
            "area_sqm": [None, 70.0],
            "area_listing": [55.0, 71.0],
            "rooms": [3, 4],
            "rooms_listing": [3, 4],
            "price_cold": ["1'850.–", None],
            "price_listing": [1900, 2100],
            "description": ["Call 079 123 45 67", None],
        }
    )
    calls = []

    def fake_read_sql(query: object, conn: object) -> pd.DataFrame:
        calls.append(query)
        return raw.copy()

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    monkeypatch.setattr(data, "anonymize_series", lambda s: s.str.replace(r"\d", "#", regex=True))
    listings, used = data.load_listings("db", database_url="sqlite://")
    assert used == "db" and len(calls) == 1
    assert listings["area"].tolist() == [55.0, 70.0]  # detail page first, search page as gap fill
    assert listings["price"].tolist() == [1850.0, 2100.0]
    assert listings.loc[10, "description"] == "Call ### ### ## ##"
    assert data.DB_EXPORT.is_file()
    frozen, used = data.load_listings("auto", database_url="sqlite://")
    assert used == "db" and len(calls) == 1  # read from the frozen export, no second query
    pd.testing.assert_frame_equal(frozen, listings)


def test_fix_schema_coerces_and_flags() -> None:
    raw = pd.DataFrame(
        {col: [np.nan] * 3 for col in data.CANONICAL_COLUMNS if col != "listing_id"},
        index=pd.Index([1, 2, 3], name="listing_id"),
    )
    raw["rooms"] = ["3½", "40", None]
    raw["price"] = ["CHF 1'850.–", "0", "2100"]
    raw["slug"] = ["  a ", "", None]
    raw["available_from"] = ["2026-05-01", "01.06.2026", "sofort"]
    out = data.fix_schema(raw)
    assert out["rooms"].tolist()[0] == 3.5 and out["rooms"].isna().tolist()[1:] == [True, True]
    assert out["rooms_invalid"].tolist() == [False, True, False]
    assert out["log_price"].iloc[0] == pytest.approx(np.log(1850))
    assert out["log_price"].isna().tolist() == [False, True, False]
    assert out["slug"].tolist() == ["a", None, None]
    assert out["available_from"].dt.month.tolist()[:2] == [5, 6]
    assert pd.isna(out["available_from"].iloc[2])
    cov = data.text_coverage(pd.DataFrame({"description": ["x" * 60, "  ", None, "short"]}))
    assert cov["n_with_text"] == 2 and cov["coverage"] == 0.5 and cov["coverage_min_chars"] == 0.25


def _listings() -> pd.DataFrame:
    """Rows 1-2 share a slug, 3-4 a description, 5-6 a building, 7 is too expensive for 5."""
    return pd.DataFrame(
        {
            "slug": ["s1", "s1", None, None, None, None, None, None],
            "description": [None, None, LONG_TEXT, LONG_TEXT.upper() + "!", None, None, None, "x"],
            "egid": pd.array([None, None, None, None, 7, 7, 7, None], dtype="Int64"),
            "address": [None] * 8,
            "east": [2.6e6, 2.61e6, 2.62e6, 2.63e6, 2.64e6, 2.65e6, 2.66e6, 2.67e6],
            "north": [1.2e6] * 8,
            "rooms": [3.0, 3.0, 2.0, 2.0, 4.0, 4.0, 4.0, 1.0],
            "area": [70.0, 70.0, 50.0, 52.0, 90.0, 92.5, 90.0, 30.0],
            "price": [1500.0, 1600.0, 1200.0, 1250.0, 2000.0, 2150.0, 2500.0, 900.0],
        },
        index=pd.Index([20, 21, 30, 31, 40, 41, 42, 50], name="listing_id"),
    )


def test_link_duplicates_rules() -> None:
    links = data.link_duplicates(_listings())
    pairs = {(a, b, rule) for a, b, rule in links.itertuples(index=False)}
    assert pairs == {(20, 21, "slug"), (30, 31, "description"), (40, 41, "egid")}


def test_link_duplicates_ignores_boilerplate_and_coords_need_equal_rooms() -> None:
    frame = pd.concat([_listings().iloc[[2]]] * (data.DESC_MAX_GROUP + 1))
    frame = frame.set_axis(pd.RangeIndex(len(frame), name="listing_id")).assign(egid=pd.NA)
    frame["east"] = 2.6e6 + 1000 * np.arange(len(frame))
    assert data.link_duplicates(frame).empty
    twins = _listings().iloc[[0, 1]].assign(slug=None, east=2.6e6, north=1.2e6)
    assert data.link_duplicates(twins)["rule"].tolist() == ["coords"]
    assert data.link_duplicates(twins.assign(rooms=[3.0, 4.0])).empty
    no_coords = twins.assign(east=np.nan, north=np.nan, price=1500.0)
    assert data.link_duplicates(no_coords).empty


def test_assign_object_ids_is_transitive_and_order_independent() -> None:
    frame = _listings()
    links = pd.DataFrame({"a": [20, 21, 40], "b": [21, 50, 42], "rule": ["x"] * 3})
    ids = data.assign_object_ids(frame, links)
    assert ids[20] == ids[21] == ids[50] == "obj_000001"
    assert ids[30] == "obj_000002" and ids[40] == ids[42] != ids[41]
    shuffled = frame.sample(frac=1, random_state=0)
    pd.testing.assert_series_equal(data.assign_object_ids(shuffled, links).loc[ids.index], ids)


def test_collapse_objects_and_report() -> None:
    frame = _listings().assign(log_price=0.0)
    links = data.link_duplicates(frame)
    ids = data.assign_object_ids(frame, links)
    objects = data.collapse_objects(frame, ids)
    assert len(objects) == 5 and objects.index.tolist() == [20, 30, 40, 42, 50]
    assert objects.loc[20, "price"] == 1550.0 and objects.loc[20, "n_listings"] == 2
    assert objects.loc[20, "log_price"] == pytest.approx(np.log(1550.0))
    report = data.dedup_report(frame, ids, links)
    assert report["unique_objects"] == 5 and report["objects_size_2"] == 3
    assert report["links_slug"] == 1 and report["duplicate_share_pct"] == pytest.approx(37.5)


def test_domain_filter_reasons_and_inclusive_bounds() -> None:
    frame = pd.DataFrame({
        "area": [10.0, 9.9, np.nan, 600.0, 50.0, 50.0, 50.0],
        "price": [300.0, 1000.0, 1000.0, 1000.0, 15_001.0, 1000.0, 1000.0],
        "rooms": [1.0, 2.0, 2.0, 2.0, 2.0, np.nan, 20.0],
    })  # fmt: skip
    kept, removed = data.domain_filter(frame)
    assert kept.index.tolist() == [0, 5]
    assert removed["reason"].tolist() == [
        "area_below_min", "area_missing", "area_above_max", "price_above_max",
        "rooms_out_of_range",
    ]  # fmt: skip


def test_price_per_sqm_outliers_with_fallback() -> None:
    rng = np.random.default_rng(0)
    n = 41
    frame = pd.DataFrame(
        {
            "area": np.full(n, 80.0),
            "price": 80 * 25 * np.exp(rng.normal(0, 0.1, n)),
            "municipality_id": [1] * 30 + [2] * 10 + [3],  # 3 is too small: district level
            "district_id": 1,
            "canton": "ZH",
        }
    )
    frame.loc[0, "price"] = 80 * 25 * 4  # 4x the local CHF/m²
    frame.loc[30:39, "price"] *= 2  # expensive municipality, judged against itself
    flags = data.price_per_sqm_outliers(frame)
    assert flags[0]
    assert not flags[30:].any()
    assert flags.sum() <= 2


def test_regime_split() -> None:
    frame = pd.DataFrame({"rent_regime": ["market", "cooperative", "unknown", "shared_flat"]})
    market, non_market = data.regime_split(frame)
    assert market.index.tolist() == [0, 2] and non_market.index.tolist() == [1, 3]


def test_recover_rooms_from_text() -> None:
    rooms = pd.Series([4.0, 4.0, np.nan, 3.0, 3.5])
    text = pd.Series([3.5, 2.5, 2.0, 3.3, 3.5])
    recovered, source = data.recover_rooms_from_text(rooms, text)
    assert recovered.tolist() == [3.5, 4.0, 2.0, 3.0, 3.5]
    assert source.tolist() == ["text_half_room", "table", "text_missing", "table", "table"]


@pytest.fixture
def objects() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 600
    return pd.DataFrame(
        {
            "object_id": [f"obj_{i // 2:06d}" for i in range(n)],  # two rows per group
            "canton": rng.choice(["ZH", "BE", "GE", "UR"], n, p=[0.5, 0.3, 0.19, 0.01]),
            "price": rng.lognormal(7.5, 0.4, n),
            "municipality_id": rng.integers(0, 40, n),
        },
        index=pd.Index(np.arange(1000, 1000 + n), name="listing_id"),
    )


def test_make_strata_merges_rare(objects: pd.DataFrame) -> None:
    strata = data.make_strata(objects, min_stratum=20)
    assert strata.value_counts().min() >= 20
    assert not strata.str.startswith("UR").any()


def test_train_calib_test_split_grouped_stratified_reproducible(objects: pd.DataFrame) -> None:
    strata = data.make_strata(objects)
    split = data.train_calib_test_split(objects, strata)
    shares = split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.6, abs=0.02)
    assert shares["test"] == pytest.approx(0.2, abs=0.02)
    assert split.groupby(objects["object_id"]).nunique().eq(1).all()
    shuffled = objects.sample(frac=1, random_state=3)
    pd.testing.assert_series_equal(
        data.train_calib_test_split(shuffled, strata)[split.index], split
    )


def test_cv_folds_are_group_disjoint(objects: pd.DataFrame) -> None:
    strata = data.make_strata(objects)
    for folds, col in [
        (data.grouped_cv_folds(objects, strata, n_splits=4), "object_id"),
        (data.grouped_cv_folds(objects, n_splits=4), "object_id"),
        (data.spatial_cv_folds(objects, n_splits=4), "municipality_id"),
    ]:
        assert len(folds) == 4
        groups = objects[col].to_numpy()
        assert all(not set(groups[tr]) & set(groups[va]) for tr, va in folds)
        assert sorted(np.concatenate([va for _, va in folds])) == list(range(len(objects)))


def test_add_engineered_features_on_one_row() -> None:
    row = pd.DataFrame([{"area": 70.0, "rooms": 3.5, "year_built": 2030, "apartments": 0}])
    out = data.add_engineered_features(row.assign(east=2_601_000.0, north=1_200_000.0))
    assert out.loc[0, "area_per_room"] == 20.0 and out.loc[0, "building_age"] == 0
    assert out.loc[0, "rooms_is_integer"] == 0.0
    assert np.isnan(out.loc[0, "land_area_per_apartment"])
    assert out.loc[0, ["rot45_x", "rot45_y"]].tolist() == pytest.approx([707.107, 707.107])
    assert np.isnan(data.add_engineered_features(row).loc[0, "rot45_x"])  # missing coordinates


def test_knn_price_features_modes() -> None:
    train = pd.DataFrame(
        {
            "east": [0.0, 0.0, 10.0, 20.0, 30.0, np.nan],
            "north": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "price": [1000.0, 1100.0, 2000.0, 3000.0, 4000.0, 5000.0],
        }
    )
    loo = data.knn_price_features(train, k=1)
    assert loo["knn_price_mean"].tolist()[:2] == [1100.0, 1000.0]  # the twin leaks its rent
    assert np.isnan(loo.loc[5, "knn_price_mean"])
    folds = [(np.array([2, 3, 4, 5]), np.array([0, 1])), (np.array([0, 1, 5]), np.array([2, 3, 4]))]
    oof = data.knn_price_features(train, k=1, mode="oof", folds=folds)
    assert oof["knn_price_mean"].tolist()[:2] == [2000.0, 2000.0]
    new = pd.DataFrame({"east": [29.0], "north": [0.0]}, index=["new"])
    applied = data.knn_price_features(train, new, k=2)
    assert applied.loc["new"].tolist() == [3500.0, 3500.0]
