"""Tests for rentml.features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GroupKFold, KFold

from rentml.features import (
    BASE,
    BUILDING,
    COORDS,
    ROT_COORDS,
    HierarchicalTargetEncoder,
    add_engineered_features,
    knn_price_features,
    monotone_vector,
)


@pytest.fixture
def listings() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 240
    canton = rng.choice(["ZH", "BE", "VD"], n)
    district = rng.choice(["1", "2", "3"], n)
    muni = rng.choice(["a", "b", "c", "d"], n)
    df = pd.DataFrame(
        {
            "area": rng.uniform(25, 160, n),
            "rooms": rng.choice([1.0, 2.5, 3.0, 3.5, 4.5], n),
            "east": rng.uniform(2_480_000, 2_830_000, n),
            "north": rng.uniform(1_080_000, 1_290_000, n),
            "re_canton": canton,
            "re_district": [f"{c}/{d}" for c, d in zip(canton, district, strict=True)],
            "re_municipality": [
                f"{c}/{d}/{m}" for c, d, m in zip(canton, district, muni, strict=True)
            ],
            "object_id": [f"obj_{i // 2:06d}" for i in range(n)],
        },
        index=pd.Index(np.arange(1000, 1000 + n), name="listing_id"),
    )
    df["price"] = 22.0 * df["area"] * rng.lognormal(0.0, 0.2, n)
    df["log_price"] = np.log(df["price"])
    return df


def test_add_engineered_features_computes_dspro1_features() -> None:
    df = pd.DataFrame(
        {
            "area": [80.0, 50.0, 60.0],
            "rooms": [4.0, 0.0, 2.5],
            "year_built": [1980.0, np.nan, 2028.0],
            "land_area": [600.0, 300.0, 100.0],
            "apartments": [6.0, 0.0, np.nan],
            "east": [2_600_000.0, 2_601_000.0, 2_600_000.0],
            "north": [1_200_000.0, 1_200_000.0, 1_201_000.0],
        }
    )
    out = add_engineered_features(df, reference_year=2026)
    assert out["area_per_room"].tolist()[0] == pytest.approx(20.0)
    assert np.isnan(out["area_per_room"].iloc[1])  # rooms == 0
    assert out["rooms_is_integer"].tolist()[0] == 1.0
    assert out["rooms_is_integer"].tolist()[2] == 0.0
    assert out["building_age"].tolist()[0] == 46.0
    assert np.isnan(out["building_age"].iloc[1])
    assert out["building_age"].iloc[2] == 0.0  # under construction -> clipped
    assert out["land_area_per_apartment"].iloc[0] == pytest.approx(100.0)
    assert out["land_area_per_apartment"].iloc[1:].isna().all()
    assert set(BASE + BUILDING[1:] + COORDS + ROT_COORDS) - {"apartments"} <= set(out.columns)


def test_add_engineered_features_rotation_preserves_distances(listings: pd.DataFrame) -> None:
    out = add_engineered_features(listings)
    orig = listings[COORDS].to_numpy()
    rot = out[ROT_COORDS].to_numpy()
    d_orig = np.linalg.norm(orig[:, None] - orig[None, :], axis=2)
    d_rot = np.linalg.norm(rot[:, None] - rot[None, :], axis=2)
    np.testing.assert_allclose(d_orig, d_rot, rtol=1e-9, atol=1e-6)
    # 45 degrees: a point on the diagonal north-east of the origin lies on the rot45_y axis.
    diag = add_engineered_features(pd.DataFrame({"east": [2_601_000.0], "north": [1_201_000.0]}))
    assert diag["rot45_x"].iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_add_engineered_features_keeps_input_and_tolerates_missing_columns() -> None:
    df = pd.DataFrame({"area": [70.0], "rooms": [3.0], "rooms_is_integer": [True]})
    before = df.copy()
    out = add_engineered_features(df, rotation_angles=(30.0,))
    pd.testing.assert_frame_equal(df, before)
    assert out["rooms_is_integer"].iloc[0]  # existing flag from data.fix_schema is kept
    assert np.isnan(out["building_age"].iloc[0])
    assert {"rot30_x", "rot30_y"} <= set(out.columns)
    assert np.isnan(out["rot30_x"].iloc[0])


def test_hte_transform_shrinks_each_level_towards_parent() -> None:
    X = pd.DataFrame({"canton": ["A"] * 4 + ["B"] * 2, "muni": ["x", "x", "x", "y", "z", "z"]})
    y = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 12.0])
    m = 2.0
    enc = HierarchicalTargetEncoder(cols=["canton", "muni"], smoothing=m).fit(X, y)
    out = enc.transform(X)
    mu = y.mean()
    canton_a = (4 * 2.5 + m * mu) / (4 + m)
    canton_b = (2 * 11.0 + m * mu) / (2 + m)
    muni_x = (3 * 2.0 + m * canton_a) / (3 + m)
    assert list(out.columns) == ["te_canton", "te_muni"]
    assert out["te_canton"].iloc[0] == pytest.approx(canton_a)
    assert out["te_canton"].iloc[4] == pytest.approx(canton_b)
    assert out["te_muni"].iloc[0] == pytest.approx(muni_x)


def test_hte_unseen_and_missing_levels_fall_back_to_parent() -> None:
    X = pd.DataFrame({"canton": ["A", "A", "B", "B"], "muni": [1, 1, 2, 2]})
    y = np.array([1.0, 3.0, 5.0, 7.0])
    enc = HierarchicalTargetEncoder(cols=["canton", "muni"], smoothing=1.0).fit(X, y)
    new = pd.DataFrame({"canton": ["A", "C", None], "muni": [99.0, 1.0, np.nan]})
    out = enc.transform(new)
    fitted = enc.transform(X)
    assert out["te_muni"].iloc[0] == pytest.approx(fitted["te_canton"].iloc[0])  # unseen muni
    assert out["te_canton"].iloc[1] == pytest.approx(enc.global_mean_)  # unseen canton
    assert out["te_muni"].iloc[1] == pytest.approx(enc.global_mean_)  # path C/1 is unseen
    assert out.iloc[2].tolist() == pytest.approx([enc.global_mean_] * 2)
    # int codes in training and float codes with NaN at prediction map to the same key
    same = enc.transform(pd.DataFrame({"canton": ["A"], "muni": [1.0]}))
    assert same["te_muni"].iloc[0] == pytest.approx(fitted["te_muni"].iloc[0])


def test_hte_oof_encoding_ignores_own_target(listings: pd.DataFrame) -> None:
    cols = ["re_canton", "re_district", "re_municipality"]
    y = listings["log_price"].to_numpy().copy()
    enc = HierarchicalTargetEncoder(cols=cols)
    base = enc.fit_transform(listings, y)
    full_base = enc.transform(listings)
    y_changed = y.copy()
    y_changed[7] += 5.0
    changed = clone(enc).fit_transform(listings, y_changed)
    np.testing.assert_allclose(changed.iloc[7], base.iloc[7])
    assert not np.allclose(
        clone(enc).fit(listings, y_changed).transform(listings).iloc[7], full_base.iloc[7]
    )  # the full fit (for new data) does see it
    assert base.index.equals(listings.index)


def test_hte_grouped_oof_ignores_duplicates_of_same_object(listings: pd.DataFrame) -> None:
    cols = ["re_canton", "re_municipality"]
    y = listings["log_price"].to_numpy().copy()
    groups = listings["object_id"]
    base = HierarchicalTargetEncoder(cols=cols).fit_transform(listings, y, groups=groups)
    rows = np.flatnonzero(groups.to_numpy() == groups.iloc[10])
    assert len(rows) == 2
    y_changed = y.copy()
    y_changed[rows] += 3.0
    changed = HierarchicalTargetEncoder(cols=cols).fit_transform(listings, y_changed, groups=groups)
    np.testing.assert_allclose(changed.iloc[rows], base.iloc[rows])


def test_hte_group_col_matches_explicit_groups(listings: pd.DataFrame) -> None:
    cols = ["re_canton", "re_municipality"]
    y = listings["log_price"]
    explicit = HierarchicalTargetEncoder(cols=cols).fit_transform(
        listings, y, groups=listings["object_id"]
    )
    via_col = HierarchicalTargetEncoder(cols=cols, group_col="object_id").fit_transform(listings, y)
    pd.testing.assert_frame_equal(explicit, via_col)
    with pytest.raises(KeyError, match="group_col"):
        HierarchicalTargetEncoder(cols=cols, group_col="nope").fit_transform(listings, y)


def test_hte_sklearn_compatibility(listings: pd.DataFrame) -> None:
    enc = HierarchicalTargetEncoder(cols=["re_canton"], smoothing=5.0, n_inner=3)
    assert clone(enc).get_params()["smoothing"] == 5.0
    with pytest.raises(NotFittedError):
        enc.transform(listings)
    enc.set_output(transform="pandas")
    out = enc.fit_transform(listings, listings["log_price"])
    assert isinstance(out, pd.DataFrame)
    assert enc.get_feature_names_out().tolist() == ["te_re_canton"]
    arr_out = HierarchicalTargetEncoder(cols=["re_canton"]).fit(
        listings[["re_canton"]].to_numpy(), listings["log_price"]
    )
    assert arr_out.transform(listings[["re_canton"]].to_numpy()).shape == (len(listings), 1)


def test_hte_rejects_invalid_input(listings: pd.DataFrame) -> None:
    enc = HierarchicalTargetEncoder(cols=["re_canton", "missing_col"])
    with pytest.raises(KeyError):
        enc.fit(listings, listings["log_price"])
    enc = HierarchicalTargetEncoder(cols=["re_canton"])
    with pytest.raises(ValueError, match="rows"):
        enc.fit(listings, listings["log_price"].iloc[:-1])
    with pytest.raises(ValueError, match="requires y"):
        enc.fit_transform(listings)
    y_nan = listings["log_price"].copy()
    y_nan.iloc[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        enc.fit(listings, y_nan)


def _brute_force_knn(
    train: pd.DataFrame,
    query: pd.DataFrame,
    k: int,
    exclude: bool,
    scale_ref: pd.DataFrame | None = None,
) -> np.ndarray:
    ref = train if scale_ref is None else scale_ref
    mu = ref[COORDS].mean().to_numpy()
    sd = ref[COORDS].std(ddof=0).to_numpy()
    a = (train[COORDS].to_numpy() - mu) / sd
    b = (query[COORDS].to_numpy() - mu) / sd
    dist = np.linalg.norm(b[:, None] - a[None, :], axis=2)
    if exclude:
        np.fill_diagonal(dist, np.inf)
    idx = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return train["price"].to_numpy()[idx].mean(axis=1)


def test_knn_loo_excludes_self(listings: pd.DataFrame) -> None:
    out = knn_price_features(listings, k=5, mode="loo")
    assert list(out.columns) == ["knn_price_mean", "knn_price_median"]
    assert out.index.equals(listings.index)
    np.testing.assert_allclose(out["knn_price_mean"], _brute_force_knn(listings, listings, 5, True))


def test_knn_loo_leaks_exact_duplicates_but_oof_does_not(listings: pd.DataFrame) -> None:
    df = listings.copy()
    dup = df.iloc[[0]].copy()
    dup.index = pd.Index([1], name="listing_id")
    dup["price"] = 99_999.0  # the same object listed twice with an extreme rent
    df.loc[df.index[0], "price"] = 99_999.0
    df = pd.concat([df, dup])
    loo = knn_price_features(df, k=1, mode="loo")
    assert loo["knn_price_mean"].iloc[0] == 99_999.0  # documented leak of the DSPRO1 feature
    groups = np.r_[listings["object_id"].to_numpy(), [listings["object_id"].iloc[0]]]
    folds = list(GroupKFold(5).split(df, groups=groups))
    oof = knn_price_features(df, k=1, mode="oof", folds=folds)
    assert oof["knn_price_mean"].iloc[0] != 99_999.0


def test_knn_oof_uses_only_other_folds(listings: pd.DataFrame) -> None:
    folds = list(KFold(4, shuffle=True, random_state=0).split(listings))
    out = knn_price_features(listings, k=3, mode="oof", folds=folds)
    for train_idx, val_idx in folds:
        expected = _brute_force_knn(
            listings.iloc[train_idx], listings.iloc[val_idx], 3, False, scale_ref=listings
        )
        np.testing.assert_allclose(out["knn_price_mean"].iloc[val_idx], expected)
    assert out.notna().all().all()


def test_knn_apply_df_uses_all_training_rows(listings: pd.DataFrame) -> None:
    train, test = listings.iloc[:200], listings.iloc[200:].copy()
    test.loc[test.index[0], "east"] = np.nan
    out = knn_price_features(train, test, k=4)
    assert out.index.equals(test.index)
    assert out.iloc[0].isna().all()
    expected = _brute_force_knn(train, test.iloc[1:], 4, False)
    np.testing.assert_allclose(out["knn_price_mean"].iloc[1:], expected)


def test_knn_rejects_invalid_arguments(listings: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="folds"):
        knn_price_features(listings, mode="oof")
    with pytest.raises(ValueError, match="mode"):
        knn_price_features(listings, mode="kfold")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k must"):
        knn_price_features(listings, k=0)
    with pytest.raises(KeyError):
        knn_price_features(listings.drop(columns="north"))
    with pytest.raises(KeyError):
        knn_price_features(listings, target="rent")


def test_monotone_vector_marks_constrained_features() -> None:
    cols = ["rooms", "area", "age", "east"]
    assert monotone_vector(cols) == [0, 1, 0, 0]
    assert monotone_vector(cols, decreasing=("age",)) == [0, 1, -1, 0]
    assert monotone_vector([], increasing=("area",)) == []


def test_monotone_vector_rejects_conflicting_constraints() -> None:
    with pytest.raises(ValueError, match="both"):
        monotone_vector(["area"], increasing=("area",), decreasing=("area",))


def test_hte_numeric_codes_in_object_columns_match_numeric_keys() -> None:
    X = pd.DataFrame({"canton": ["A", "A", "B", "B"], "muni": [1, 1, 2, 2]})
    enc = HierarchicalTargetEncoder(cols=["canton", "muni"], smoothing=1.0).fit(
        X, np.array([1.0, 3.0, 5.0, 7.0])
    )
    fitted = enc.transform(X)["te_muni"]
    # object dtype holding 1.0 (e.g. after a concat with None) and a numeric string
    new = pd.DataFrame(
        {"canton": ["A", "A", "B"], "muni": pd.Series([1.0, "1", None], dtype=object)}
    )
    out = enc.transform(new)
    assert out["te_muni"].iloc[0] == pytest.approx(fitted.iloc[0])
    assert out["te_muni"].iloc[1] == pytest.approx(fitted.iloc[0])
    assert out["te_muni"].iloc[2] == pytest.approx(out["te_canton"].iloc[2])  # missing -> parent
    # non-numeric string keys are left untouched
    words = HierarchicalTargetEncoder(cols=["canton"], smoothing=0.0).fit(
        pd.DataFrame({"canton": ["ZH/1", "ZH/1.0"]}), np.array([1.0, 3.0])
    )
    assert words.transform(pd.DataFrame({"canton": ["ZH/1"]}))["te_canton"].iloc[0] == 1.0


def test_hte_inner_folds_are_seeded_and_validated(listings: pd.DataFrame) -> None:
    cols = ["re_canton", "re_municipality"]
    y = listings["log_price"]
    groups = listings["object_id"]
    first = HierarchicalTargetEncoder(cols=cols).fit_transform(listings, y, groups=groups)
    again = HierarchicalTargetEncoder(cols=cols).fit_transform(listings, y, groups=groups)
    other = HierarchicalTargetEncoder(cols=cols, random_state=7).fit_transform(
        listings, y, groups=groups
    )
    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(first, other)
    few_groups = np.where(np.arange(len(listings)) < 100, "g1", "g2")
    with pytest.raises(ValueError, match="n_inner"):
        HierarchicalTargetEncoder(cols=cols).fit_transform(listings, y, groups=few_groups)
    with pytest.raises(ValueError, match="n_inner"):
        HierarchicalTargetEncoder(cols=cols, n_inner=1).fit_transform(listings, y)
    with pytest.raises(ValueError, match="groups"):
        HierarchicalTargetEncoder(cols=cols).fit_transform(listings, y, groups=groups.iloc[:-1])


def test_knn_loo_excludes_self_by_identity_among_duplicates(listings: pd.DataFrame) -> None:
    df = listings.copy()
    df.loc[df.index[0], "price"] = 11_111.0
    dup = df.iloc[[0]].copy()
    dup.index = pd.Index([1], name="listing_id")
    dup["price"] = 99_999.0  # same object and coordinates, different rent
    df = pd.concat([df, dup])
    loo = knn_price_features(df, k=1, mode="loo")["knn_price_mean"]
    # never a row's own price: each copy sees the other one (the documented duplicate leak)
    assert loo.loc[df.index[0]] == 99_999.0
    assert loo.loc[1] == 11_111.0
    groups = np.r_[listings["object_id"].to_numpy(), [listings["object_id"].iloc[0]]]
    folds = list(GroupKFold(5).split(df, groups=groups))
    oof = knn_price_features(df, k=1, mode="oof", folds=folds)["knn_price_mean"]
    assert not {oof.loc[df.index[0]], oof.loc[1]} & {11_111.0, 99_999.0}


def test_knn_loo_featurises_rows_without_target(listings: pd.DataFrame) -> None:
    df = listings.copy()
    no_target = df.index[:3]
    df.loc[no_target, "price"] = np.nan
    loo = knn_price_features(df, k=4, mode="loo")
    assert loo.notna().all().all()
    pool = df.drop(index=no_target)
    expected = _brute_force_knn(pool, df.loc[no_target], 4, False, scale_ref=df)
    np.testing.assert_allclose(loo.loc[no_target, "knn_price_mean"], expected)
    # pool rows are unaffected by the extra query rows
    np.testing.assert_allclose(
        loo.loc[pool.index, "knn_price_mean"], _brute_force_knn(pool, pool, 4, True, scale_ref=df)
    )
