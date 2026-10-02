"""Tests for rentml.splits (synthetic data, fixed seeds)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from rentml.splits import (
    assert_no_group_leakage,
    group_codes,
    grouped_cv_folds,
    make_strata,
    spatial_cv_folds,
    split_summary,
    temporal_split,
    train_calib_test_split,
)

CANTONS = np.array(["ZH", "BE", "VD", "GE", "AG", "SG", "LU", "TI", "BS", "UR"])


@pytest.fixture
def listings() -> pd.DataFrame:
    """About 5'000 listings of 2'500 objects; objects have 1-4 listings (duplicates)."""
    rng = np.random.default_rng(0)
    n_obj = 2500
    sizes = rng.choice([1, 1, 2, 3, 4], size=n_obj)
    obj = np.repeat(np.arange(n_obj), sizes)
    probs = np.array([0.3, 0.2, 0.15, 0.1, 0.1, 0.06, 0.05, 0.02, 0.019, 0.001])
    canton = rng.choice(CANTONS, size=n_obj, p=probs)[obj]
    muni = rng.integers(0, 150, size=n_obj)[obj]
    price = np.exp(rng.normal(7.6, 0.35, n_obj))[obj] * rng.uniform(0.97, 1.03, obj.size)
    days = rng.integers(0, 60, size=obj.size)
    return pd.DataFrame(
        {
            "object_id": [f"obj_{i:06d}" for i in obj],
            "canton": canton,
            "municipality_id": muni,
            "price": price,
            "area": rng.uniform(30, 150, obj.size),
            "observed_at": pd.Timestamp("2026-04-01") + pd.to_timedelta(days, unit="D"),
        },
        index=pd.Index(np.arange(obj.size) + 10_000, name="listing_id"),
    )


def _assert_disjoint_cover(splits: dict[str, pd.Index], index: pd.Index) -> None:
    union = np.concatenate([idx.to_numpy() for idx in splits.values()])
    assert len(union) == len(index)
    assert set(union) == set(index)


def _fold_of_label(folds: list[tuple[np.ndarray, np.ndarray]], index: pd.Index) -> pd.Series:
    """Map each index label to its validation fold (positions refer to ``index``)."""
    fold = pd.Series(-1, index=index)
    for k, (_, val) in enumerate(folds):
        fold.iloc[val] = k
    return fold.sort_index()


def test_make_strata_merges_rare_cantons(listings: pd.DataFrame) -> None:
    strata = make_strata(listings, min_stratum=30)
    assert strata.index.equals(listings.index)
    assert strata.value_counts().min() >= 30
    assert not strata[listings["canton"] == "UR"].str.startswith("UR|").any()
    assert strata.str.startswith("ALL|").any()
    assert set(strata[listings["canton"] == "ZH"].unique()) == {"ZH|P1", "ZH|P2", "ZH|P3", "ZH|P4"}


def test_make_strata_absorbs_tiny_pools_into_same_band() -> None:
    df = pd.DataFrame({"canton": ["ZH"] * 40 + ["UR", "JU"], "price": np.arange(42.0)})
    strata = make_strata(df, n_price_bins=2, min_stratum=10)
    assert strata.value_counts().min() >= 10
    assert strata.iloc[-2:].tolist() == ["ZH|P2", "ZH|P2"]
    tiny = make_strata(df.iloc[:3], min_stratum=10)
    assert (tiny == "ALL|ALL").all()


def test_make_strata_missing_price_and_errors(listings: pd.DataFrame) -> None:
    df = listings.copy()
    df.loc[df.index[:50], "price"] = np.nan
    strata = make_strata(df, min_stratum=5)
    assert strata.iloc[:50].str.endswith("PNA").all()
    with pytest.raises(KeyError):
        make_strata(df.drop(columns="canton"))
    with pytest.raises(ValueError):
        make_strata(df, n_price_bins=0)


def test_train_calib_test_split_fractions_and_no_leakage(listings: pd.DataFrame) -> None:
    splits = train_calib_test_split(listings, strata=make_strata(listings))
    assert list(splits) == ["train", "calib", "test"]
    _assert_disjoint_cover(splits, listings.index)
    for name, target in zip(splits, (0.6, 0.2, 0.2), strict=True):
        assert abs(len(splits[name]) / len(listings) - target) <= 0.03
    assert_no_group_leakage(splits, listings["object_id"])


def test_train_calib_test_split_is_stratified(listings: pd.DataFrame) -> None:
    strata = make_strata(listings)
    splits = train_calib_test_split(listings, strata=strata)
    test_share = strata.index.isin(splits["test"])
    per_stratum = pd.Series(test_share).groupby(strata.to_numpy()).mean()
    big = strata.value_counts()[lambda c: c >= 200].index
    assert (per_stratum[big] - 0.2).abs().max() < 0.05


def test_train_calib_test_split_deterministic(listings: pd.DataFrame) -> None:
    first = train_calib_test_split(listings, seed=7)
    second = train_calib_test_split(listings, seed=7)
    other = train_calib_test_split(listings, seed=8)
    assert all(first[k].equals(second[k]) for k in first)
    assert not first["test"].equals(other["test"])


def test_train_calib_test_split_invalid_fractions(listings: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="multiples of 0.2"):
        train_calib_test_split(listings, fractions=(0.5, 0.25, 0.25))
    with pytest.raises(ValueError, match="sum to 1"):
        train_calib_test_split(listings, fractions=(0.6, 0.2, 0.4))
    with pytest.raises(ValueError, match="n_groups"):
        train_calib_test_split(listings.iloc[:3])


def test_train_calib_test_split_without_calib(listings: pd.DataFrame) -> None:
    splits = train_calib_test_split(listings, fractions=(0.8, 0.0, 0.2))
    assert len(splits["calib"]) == 0
    assert abs(len(splits["test"]) / len(listings) - 0.2) <= 0.03


def test_missing_group_keys_become_singletons(listings: pd.DataFrame) -> None:
    df = listings.copy()
    df.loc[df.index[:20], "object_id"] = None
    splits = train_calib_test_split(df)
    _assert_disjoint_cover(splits, df.index)
    codes = group_codes(pd.Series(["a", None, "a", np.nan, "b"]))
    assert codes[0] == codes[2]
    assert len(set(codes.tolist())) == 4


def test_grouped_cv_folds_partition_and_groups(listings: pd.DataFrame) -> None:
    folds = grouped_cv_folds(listings, strata=make_strata(listings))
    assert len(folds) == 5
    val_all = np.concatenate([val for _, val in folds])
    assert np.array_equal(np.sort(val_all), np.arange(len(listings)))
    groups = listings["object_id"].to_numpy()
    for train, val in folds:
        assert not set(groups[train]) & set(groups[val])
        assert abs(len(val) / len(listings) - 0.2) <= 0.03


def test_grouped_cv_folds_unstratified_balanced_and_seeded(listings: pd.DataFrame) -> None:
    folds = grouped_cv_folds(listings, n_splits=4, seed=1)
    sizes = [len(val) for _, val in folds]
    assert max(sizes) - min(sizes) <= 4
    again = grouped_cv_folds(listings, n_splits=4, seed=1)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, again, strict=True))


def test_grouped_cv_folds_invalid_n_splits(listings: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        grouped_cv_folds(listings, n_splits=1)
    with pytest.raises(ValueError):
        grouped_cv_folds(listings.iloc[:4], n_splits=5)


def test_spatial_cv_folds_hold_out_whole_municipalities(listings: pd.DataFrame) -> None:
    folds = spatial_cv_folds(listings)
    muni = listings["municipality_id"].to_numpy()
    objects = listings["object_id"].to_numpy()
    for train, val in folds:
        assert not set(muni[train]) & set(muni[val])
        assert not set(objects[train]) & set(objects[val])
    assert sum(len(val) for _, val in folds) == len(listings)


def test_spatial_cv_folds_merges_units_linked_by_objects(listings: pd.DataFrame) -> None:
    df = listings.copy()
    first_obj = df["object_id"].iloc[0]
    rows = df.index[df["object_id"] == first_obj]
    df.loc[rows[0], "municipality_id"] = 998
    df.loc[df.index[-5:], "municipality_id"] = np.nan
    folds = spatial_cv_folds(df, n_splits=3)
    objects = df["object_id"].to_numpy()
    for train, val in folds:
        assert not set(objects[train]) & set(objects[val])
    with pytest.raises(ValueError):
        spatial_cv_folds(df.iloc[:10].assign(municipality_id=1, object_id="x"), n_splits=2)


def test_temporal_split_inclusive_and_unseen_objects() -> None:
    df = pd.DataFrame(
        {
            "object_id": ["a", "b", "a", "c", "d"],
            "observed_at": ["2026-05-01", "2026-05-31 18:00", "2026-06-02", "2026-06-03", None],
        },
        index=pd.Index([1, 2, 3, 4, 5], name="listing_id"),
    )
    pre, post = temporal_split(df, freeze_date="2026-05-31")
    assert list(pre) == [1, 2]
    assert list(post) == [4]
    _, post_all = temporal_split(df, freeze_date="2026-05-31", group_col=None)
    assert list(post_all) == [3, 4]


def test_temporal_split_timezone_and_invalid_date() -> None:
    df = pd.DataFrame(
        {"observed_at": pd.to_datetime(["2026-05-01", "2026-07-01"]).tz_localize("UTC")}
    )
    pre, post = temporal_split(df, freeze_date="2026-06-01", group_col=None)
    assert list(pre) == [0] and list(post) == [1]
    with pytest.raises(ValueError):
        temporal_split(df, freeze_date="not a date")


def test_assert_no_group_leakage_detects_overlap() -> None:
    groups = pd.Series(["a", "a", "b", "c", None, None], index=[1, 2, 3, 4, 5, 6])
    assert_no_group_leakage({"train": pd.Index([1, 2, 5]), "test": pd.Index([3, 6])}, groups)
    with pytest.raises(AssertionError, match="groups in both"):
        assert_no_group_leakage({"train": pd.Index([1, 3]), "test": pd.Index([2])}, groups)
    with pytest.raises(AssertionError, match="rows in both"):
        assert_no_group_leakage({"train": pd.Index([1, 3]), "test": pd.Index([3])}, groups)
    with pytest.raises(ValueError):
        assert_no_group_leakage({"train": pd.Index([99])}, groups)


def test_split_summary_columns_and_counts(listings: pd.DataFrame) -> None:
    splits = train_calib_test_split(listings)
    summary = split_summary(listings, splits, ["price", "canton"])
    assert list(summary.index) == ["train", "calib", "test", "all"]
    assert summary.loc["all", "n"] == len(listings)
    assert summary.loc[["train", "calib", "test"], "n"].sum() == len(listings)
    assert {"price_mean", "price_median", "canton_top", "canton_top_share", "n_groups"} <= set(
        summary.columns
    )
    assert summary.loc["all", "canton_top"] == "ZH"


def test_split_summary_empty_split(listings: pd.DataFrame) -> None:
    splits = {"train": listings.index[:10], "calib": pd.Index([], dtype=int)}
    summary = split_summary(listings, splits, ["canton"], group_col=None)
    assert summary.loc["calib", "n"] == 0
    assert summary.loc["calib", "canton_top"] == ""
    assert "n_groups" not in summary.columns


def test_split_invariant_to_row_order(listings: pd.DataFrame) -> None:
    df = listings.copy()
    df.loc[df.index[:15], "object_id"] = None  # missing keys: singletons ordered by index label
    df.loc[df.index[-10:], "municipality_id"] = np.nan
    shuffled = df.sample(frac=1.0, random_state=3)
    strata = make_strata(df)
    assert make_strata(shuffled).reindex(df.index).equals(strata)
    for st in (None, strata):
        first = train_calib_test_split(df, strata=st)
        second = train_calib_test_split(shuffled, strata=st)
        assert all(set(first[name]) == set(second[name]) for name in first)
        folds_a = _fold_of_label(grouped_cv_folds(df, strata=st), df.index)
        folds_b = _fold_of_label(grouped_cv_folds(shuffled, strata=st), shuffled.index)
        assert folds_a.equals(folds_b)
    spatial_a = _fold_of_label(spatial_cv_folds(df), df.index)
    spatial_b = _fold_of_label(spatial_cv_folds(shuffled), shuffled.index)
    assert spatial_a.equals(spatial_b)


def test_group_codes_sorted_and_missing_ordered_by_index() -> None:
    values = pd.Series(["b", None, "a", "b", None], index=[5, 9, 1, 2, 3])
    codes = group_codes(values)
    assert codes.tolist() == [1, 3, 0, 1, 2]
    reordered = values.iloc[[4, 3, 2, 1, 0]]
    mapping = dict(zip(reordered.index, group_codes(reordered).tolist(), strict=True))
    assert mapping == dict(zip(values.index, codes.tolist(), strict=True))


def test_grouped_cv_folds_on_train_subset_with_full_strata(listings: pd.DataFrame) -> None:
    strata = make_strata(listings)
    train_df = listings.loc[train_calib_test_split(listings, strata=strata)["train"]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # e.g. StratifiedGroupKFold's too-small-class warning
        folds = grouped_cv_folds(train_df, strata=strata)
    groups = train_df["object_id"]
    assert sorted(np.concatenate([val for _, val in folds]).tolist()) == list(range(len(train_df)))
    for train, val in folds:
        assert max(train.max(), val.max()) < len(train_df)
        assert not set(groups.iloc[train]) & set(groups.iloc[val])
    fold = _fold_of_label(folds, train_df.index)
    sub_strata = strata.reindex(fold.index)
    big = sub_strata.value_counts()[lambda c: c >= 150].index
    share = pd.crosstab(sub_strata, fold, normalize="index").loc[big]
    assert (share - 0.2).abs().max().max() < 0.06


def test_split_rejects_duplicate_index_and_bad_strata(listings: pd.DataFrame) -> None:
    duplicated = listings.iloc[:40].set_axis(np.arange(40) % 20)
    with pytest.raises(ValueError, match="unique"):
        train_calib_test_split(duplicated)
    strata = make_strata(listings)
    by_label = grouped_cv_folds(listings, strata=strata)
    by_position = grouped_cv_folds(listings, strata=strata.to_numpy())
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(by_label, by_position, strict=True))
    with pytest.raises(ValueError, match="strata has"):
        grouped_cv_folds(listings, strata=strata.to_numpy()[:-1])
    with pytest.raises(ValueError, match="non-missing"):
        grouped_cv_folds(listings, strata=strata.iloc[:-1])


def test_temporal_split_mixed_offsets_compared_in_utc() -> None:
    df = pd.DataFrame(
        {
            "observed_at": [
                "2026-03-28T23:30:00+01:00",  # 22:30 UTC
                "2026-03-29T00:30:00+01:00",  # 23:30 UTC on the freeze day
                "2026-03-30T10:00:00+02:00",  # after the DST switch
                "garbage",
            ]
        }
    )
    pre, post = temporal_split(df, freeze_date="2026-03-28", group_col=None)
    assert list(pre) == [0, 1]
    assert list(post) == [2]


def test_temporal_split_aware_freeze_on_naive_data() -> None:
    naive = pd.DataFrame({"observed_at": pd.to_datetime(["2026-05-31 23:30", "2026-06-01 00:30"])})
    pre, post = temporal_split(naive, freeze_date="2026-06-01T02:00:00+02:00", group_col=None)
    assert list(pre) == [0]
    assert list(post) == [1]
    zurich = naive.assign(observed_at=naive["observed_at"].dt.tz_localize("Europe/Zurich"))
    pre_z, post_z = temporal_split(zurich, freeze_date="2026-05-31", group_col=None)
    assert list(pre_z) == [0] and list(post_z) == [1]
