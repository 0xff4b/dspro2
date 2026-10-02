"""Tests for rentml.cleaning."""

import logging

import numpy as np
import pandas as pd
import pytest

from rentml.cleaning import (
    cleaning_log,
    domain_filter,
    price_per_sqm_outliers,
    price_per_sqm_zscores,
    recover_rooms_from_text,
    regime_split,
)
from rentml.config import RANDOM_STATE

GROUP_COLS = ("municipality_id", "district_id", "canton")


def _block(
    n: int, rate: float, muni: float, district: float, canton: str, seed: int
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    area = rng.integers(40, 120, n).astype(float)
    price = area * rate * np.exp(rng.normal(0.0, 0.05, n))
    return [
        {"area": a, "price": p, "municipality_id": muni, "district_id": district, "canton": canton}
        for a, p in zip(area, price, strict=True)
    ]


def _geo_frame() -> pd.DataFrame:
    """Two municipalities with different CHF/m² levels plus small / degenerate groups."""
    rows = [
        *_block(30, 25.0, 1, 10, "ZH", RANDOM_STATE),  # muni 1: ~25 CHF/m²
        *_block(30, 50.0, 2, 20, "GE", RANDOM_STATE + 1),  # muni 2: ~50 CHF/m²
        *_block(4, 25.0, 3, 10, "ZH", RANDOM_STATE + 2),  # muni 3: too small -> district
        *[
            {"area": 50.0, "price": 2500.0, "municipality_id": 4, "district_id": 20}
            for _ in range(11)
        ],
        {"area": 50.0, "price": 2600.0, "municipality_id": 4, "district_id": 20},  # MAD = 0
        {"area": 80.0, "price": 4000.0, "municipality_id": 1, "district_id": 10},  # 50 in muni 1
        {"area": 80.0, "price": 4000.0, "municipality_id": 2, "district_id": 20},  # 50 in muni 2
        {"area": 80.0, "price": 2000.0},  # no geography -> global
        {"area": 80.0, "price": np.nan, "municipality_id": 1},  # invalid CHF/m²
    ]
    frame = pd.DataFrame(rows)
    frame["canton"] = frame["canton"].where(frame["district_id"].notna())
    frame.loc[frame["district_id"] == 20, "canton"] = "GE"
    frame.loc[frame["district_id"] == 10, "canton"] = "ZH"
    frame.index = pd.RangeIndex(1000, 1000 + len(frame), name="listing_id")
    return frame


def test_domain_filter_reports_first_failing_rule() -> None:
    df = pd.DataFrame(
        {
            "area": [50.0, np.nan, 5.0, 600.0, 50.0, 50.0, 50.0, 5.0],
            "price": [1500.0, 100.0, 1500.0, 1500.0, np.nan, 250.0, 20000.0, 100.0],
        },
        index=pd.Index([11, 12, 13, 14, 15, 16, 17, 18], name="listing_id"),
    )
    kept, removed = domain_filter(df)
    assert kept.index.tolist() == [11]
    assert removed["reason"].to_dict() == {
        12: "area_missing",
        13: "area_below_min",
        14: "area_above_max",
        15: "price_missing",
        16: "price_below_min",
        17: "price_above_max",
        18: "area_below_min",
    }
    assert list(kept.columns) == ["area", "price"]


def test_domain_filter_bounds_are_inclusive_and_nullable_dtypes_work() -> None:
    df = pd.DataFrame(
        {
            "area": pd.array([10, 500, 9, None], dtype="Int64"),
            "price": pd.array([300.0, 15000.0, 1000.0, 1000.0], dtype="Float64"),
        }
    )
    kept, removed = domain_filter(df)
    assert kept.index.tolist() == [0, 1]
    assert removed["reason"].tolist() == ["area_below_min", "area_missing"]


def test_domain_filter_rejects_invalid_bounds_and_missing_columns() -> None:
    df = pd.DataFrame({"area": [50.0], "price": [1500.0]})
    with pytest.raises(ValueError):
        domain_filter(df, min_area=100.0, max_area=50.0)
    with pytest.raises(KeyError):
        domain_filter(df.drop(columns="price"))


def test_price_per_sqm_outliers_are_relative_to_the_municipality() -> None:
    df = _geo_frame()
    mask = price_per_sqm_outliers(df, group_cols=GROUP_COLS)
    odd_in_muni1, normal_in_muni2 = df.index[-4], df.index[-3]
    assert mask[odd_in_muni1]
    assert not mask[normal_in_muni2]
    assert mask.sum() <= 3  # the regular synthetic listings are not flagged
    # Without geography the CHF/m² mixture hides the anomaly.
    assert not price_per_sqm_outliers(df, group_cols=())[odd_in_muni1]


def test_price_per_sqm_zscores_use_hierarchical_fallback() -> None:
    df = _geo_frame()
    scores = price_per_sqm_zscores(df, group_cols=GROUP_COLS, min_group=10)
    levels = scores["level"]
    assert (levels[df["municipality_id"] == 1].dropna() == "municipality_id").all()
    assert (levels[df["municipality_id"] == 3] == "district_id").all()  # only 4 rows
    assert (levels[df["municipality_id"] == 4] == "district_id").all()  # MAD = 0
    assert levels.iloc[-2] == "global"  # no geography at all
    assert pd.isna(levels.iloc[-1]) and np.isnan(scores["robust_z"].iloc[-1])
    assert scores.index.equals(df.index)
    muni1 = scores[df["municipality_id"] == 1].iloc[:30]
    assert (muni1["group_n"] == 31).all()  # 30 regular rows + the odd row (NaN price excluded)


def test_price_per_sqm_outliers_returns_aligned_boolean_series() -> None:
    df = _geo_frame()
    mask = price_per_sqm_outliers(df, group_cols=GROUP_COLS)
    assert mask.dtype == bool
    assert mask.name == "ppsqm_outlier"
    assert mask.index.equals(df.index)
    assert not mask.iloc[-1]  # invalid CHF/m² is never flagged


def test_price_per_sqm_zscores_global_mad_zero_uses_mean_absolute_deviation() -> None:
    df = pd.DataFrame({"area": [50.0] * 11, "price": [2500.0] * 10 + [5000.0]})
    scores = price_per_sqm_zscores(df, group_cols=())
    assert (scores["robust_z"].iloc[:10] == 0.0).all()
    assert np.isfinite(scores["robust_z"].iloc[10]) and scores["robust_z"].iloc[10] > 3.5
    constant = price_per_sqm_zscores(df.iloc[:10], group_cols=())
    assert (constant["robust_z"] == 0.0).all()


def test_price_per_sqm_zscores_handle_duplicate_index_and_empty_frames() -> None:
    df = _geo_frame()
    duplicated = df.set_axis(np.zeros(len(df), dtype=int))
    expected = price_per_sqm_zscores(df, group_cols=GROUP_COLS)["robust_z"].to_numpy()
    got = price_per_sqm_zscores(duplicated, group_cols=GROUP_COLS)["robust_z"].to_numpy()
    np.testing.assert_allclose(got, expected)
    assert price_per_sqm_outliers(df.iloc[0:0], group_cols=GROUP_COLS).empty


def _rows(
    rates: np.ndarray, muni: int, district: int, canton: str = "BE"
) -> list[dict[str, object]]:
    return [
        {
            "area": 80.0,
            "price": 80.0 * float(rate),
            "municipality_id": muni,
            "district_id": district,
            "canton": canton,
        }
        for rate in rates
    ]


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.index = pd.RangeIndex(1, len(frame) + 1, name="listing_id")
    return frame


def test_price_per_sqm_outliers_ignore_duplicate_inflated_groups() -> None:
    # Municipality 7 (12 rows, 7 objects): re-posts of two objects sit on the median and make
    # the row-level MAD tiny (real case: Lotzwil); four ordinary listings are +-15-20 % away.
    tight = [20.0] * 4 + [20.1] * 3 + [19.9]
    ordinary = [16.0, 17.0, 23.5, 24.0]
    district = 20.0 * np.exp(np.linspace(-0.3, 0.3, 40))
    df = _frame(_rows(np.array(tight + ordinary), 7, 70) + _rows(district, 8, 70))
    object_ids = pd.Series([f"obj_{k}" for k in range(len(df))], index=df.index)
    object_ids.iloc[1:4] = "obj_0"  # rows 1-4: one object
    object_ids.iloc[5:7] = "obj_4"  # rows 5-7: one object
    muni7 = df["municipality_id"] == 7
    naive = price_per_sqm_outliers(df, group_cols=GROUP_COLS, min_scale_ratio=0.0)
    assert naive[muni7].sum() == 4  # precondition: row counting flags the ordinary listings
    mask = price_per_sqm_outliers(df, group_cols=GROUP_COLS, object_ids=object_ids)
    assert not mask.any()
    scores = price_per_sqm_zscores(df, group_cols=GROUP_COLS, object_ids=object_ids)
    assert (scores.loc[muni7, "level"] == "district_id").all()  # 7 objects < min_group
    assert (scores.loc[muni7, "group_n"] == 47).all()  # 7 + 40 distinct objects


def test_price_per_sqm_zscores_floor_the_scale_of_tight_small_groups() -> None:
    tight = 20.0 * np.exp(np.linspace(-0.015, 0.015, 12))
    district = 20.0 * np.exp(np.linspace(-0.3, 0.3, 40))
    df = _frame(_rows(np.r_[tight, 25.0], 9, 90) + _rows(district, 10, 90))
    plus_25pct = df.index[12]
    unfloored = price_per_sqm_outliers(df, group_cols=GROUP_COLS, min_scale_ratio=0.0)
    assert unfloored[plus_25pct]
    assert not price_per_sqm_outliers(df, group_cols=GROUP_COLS)[plus_25pct]
    scores = price_per_sqm_zscores(df, group_cols=GROUP_COLS)
    parent = price_per_sqm_zscores(df, group_cols=("district_id", "canton"))
    muni9 = df["municipality_id"] == 9
    assert (scores.loc[muni9, "level"] == "municipality_id").all()
    np.testing.assert_allclose(scores.loc[muni9, "scale"], 0.5 * parent.loc[muni9, "scale"])


def test_price_per_sqm_outliers_flag_each_tail() -> None:
    rates = 25.0 * np.exp(np.linspace(-0.1, 0.1, 30))
    df = _frame(_rows(np.r_[rates, 8.0, 75.0], 1, 10))
    scores = price_per_sqm_zscores(df, group_cols=GROUP_COLS)
    mask = price_per_sqm_outliers(df, group_cols=GROUP_COLS)
    low, high = df.index[30], df.index[31]
    assert mask[low] and scores.loc[low, "robust_z"] < -3.5
    assert mask[high] and scores.loc[high, "robust_z"] > 3.5
    assert mask.sum() == 2


def test_price_per_sqm_zscores_reject_bad_object_ids_and_ratio() -> None:
    df = _geo_frame()
    ids = pd.Series("obj", index=df.index)
    with pytest.raises(ValueError):
        price_per_sqm_zscores(df, object_ids=ids.iloc[1:])
    with pytest.raises(ValueError):
        price_per_sqm_zscores(df, object_ids=ids.set_axis(ids.index + 1))
    with pytest.raises(ValueError):
        price_per_sqm_zscores(df, object_ids=ids.where(ids.index > df.index[0]))
    with pytest.raises(ValueError):
        price_per_sqm_zscores(df, min_scale_ratio=1.5)
    with pytest.raises(ValueError):
        price_per_sqm_outliers(df, min_scale_ratio=-0.1)


def test_price_per_sqm_outliers_rejects_bad_arguments() -> None:
    df = _geo_frame()
    with pytest.raises(ValueError):
        price_per_sqm_outliers(df, z_thresh=0.0)
    with pytest.raises(ValueError):
        price_per_sqm_outliers(df, min_group=0)
    with pytest.raises(KeyError):
        price_per_sqm_outliers(df, group_cols=("bfs_nr",))


def test_regime_split_separates_non_market_regimes() -> None:
    df = pd.DataFrame(
        {
            "rent_regime": [
                "market",
                None,
                "Cooperative",
                "shared flat",
                "unknown",
                "cost-based or subsidised",
            ],
            "price": [2000, 1900, 1200, 700, 1800, 1100],
        }
    )
    market, non_market = regime_split(df)
    assert market.index.tolist() == [0, 1, 4]
    assert non_market.index.tolist() == [2, 3, 5]
    assert non_market["rent_regime"].tolist() == df["rent_regime"].iloc[[2, 3, 5]].tolist()


def test_regime_split_without_column_keeps_everything_as_market() -> None:
    df = pd.DataFrame({"price": [1000, 2000]})
    market, non_market = regime_split(df)
    assert len(market) == 2
    assert non_market.empty and list(non_market.columns) == ["price"]


def test_regime_split_rejects_unknown_labels() -> None:
    df = pd.DataFrame({"rent_regime": ["market", "luxury"]})
    with pytest.raises(ValueError, match="luxury"):
        regime_split(df)


def test_cleaning_log_computes_removed_and_retained_shares() -> None:
    log = cleaning_log([("raw", 200), ("domain_filter", 180), ("dedup", 150)])
    assert log["removed"].tolist() == [0, 20, 30]
    assert log["removed_pct"].tolist() == pytest.approx([0.0, 10.0, 100 * 30 / 180])
    assert log["retained_pct"].tolist() == pytest.approx([100.0, 90.0, 75.0])
    assert log["n_rows"].dtype == np.int64


def test_cleaning_log_edge_cases(caplog: pytest.LogCaptureFixture) -> None:
    assert cleaning_log([]).empty
    with pytest.raises(ValueError):
        cleaning_log([("raw", 10), ("bad", -1)])
    with pytest.raises(ValueError):
        cleaning_log([("raw", 10.5)])  # type: ignore[list-item]
    with caplog.at_level(logging.WARNING, logger="rentml.cleaning"):
        log = cleaning_log([("raw", 0), ("joined", 5)])
    assert np.isnan(log["retained_pct"].iloc[1])
    assert "increases" in caplog.text


def test_recover_rooms_from_text_half_rooms_and_missing_counts() -> None:
    idx = pd.Index([10, 11, 12, 13, 14, 15, 16, 17], name="listing_id")
    stored = pd.Series([4.0, 3.0, 4.0, np.nan, np.nan, 3.0, 3.0, 2.0], index=idx)
    text = pd.Series([3.5, 3.5, 2.5, 2.5, 20.0, np.nan, 3.3, 2.0], index=idx)
    rooms, source = recover_rooms_from_text(stored, text.iloc[::-1])  # aligned by label
    assert rooms.tolist()[:4] == [3.5, 3.5, 4.0, 2.5]
    assert np.isnan(rooms.loc[14]) and rooms.loc[15:].tolist() == [3.0, 3.0, 2.0]
    expected = ["text_half_room"] * 2 + ["table", "text_missing", "missing"] + ["table"] * 3
    assert source.tolist() == expected
    assert stored.loc[10] == 4.0  # input unchanged
