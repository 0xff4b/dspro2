"""Tests for rentml.explain (TreeSHAP on LightGBM log-price models)."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from rentml.explain import global_importance, shap_frame, top_drivers, tree_shap

shap = pytest.importorskip("shap")


@pytest.fixture(scope="module")
def model_and_data() -> tuple[lgb.LGBMRegressor, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 400
    X = pd.DataFrame(
        {
            "area": rng.uniform(30, 150, n),
            "rooms": rng.integers(1, 6, n).astype(float),
            "noise": rng.normal(size=n),
            "year_built": rng.uniform(1900, 2025, n),
        },
        index=pd.Index(np.arange(1000, 1000 + n), name="listing_id"),
    )
    X.loc[X.index[::11], "year_built"] = np.nan
    y_log = 5.0 + 0.9 * np.log(X["area"]) + 0.02 * X["rooms"] + rng.normal(0, 0.05, n)
    model = lgb.LGBMRegressor(n_estimators=60, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X, y_log)
    return model, X


def test_tree_shap_is_additive_on_log_scale(model_and_data):
    model, X = model_and_data
    expl = tree_shap(model, X, max_rows=100, seed=1)
    assert expl.values.shape == (100, X.shape[1])
    reconstructed = expl.values.sum(axis=1) + expl.base_values
    sampled = X.loc[list(expl.instance_names)]
    np.testing.assert_allclose(reconstructed, model.predict(sampled), atol=1e-6)
    assert list(expl.feature_names) == list(X.columns)


def test_tree_shap_subsample_is_seeded_and_ordered(model_and_data):
    model, X = model_and_data
    first = tree_shap(model, X, max_rows=50, seed=7)
    second = tree_shap(model, X, max_rows=50, seed=7)
    other = tree_shap(model, X, max_rows=50, seed=8)
    assert list(first.instance_names) == list(second.instance_names)
    assert list(first.instance_names) != list(other.instance_names)
    positions = X.index.get_indexer(list(first.instance_names))
    assert np.all(np.diff(positions) > 0)


def test_tree_shap_small_frame_explains_all_rows(model_and_data):
    model, X = model_and_data
    expl = tree_shap(model, X.head(5))
    assert list(expl.instance_names) == list(X.index[:5])


def test_tree_shap_works_with_booster(model_and_data):
    model, X = model_and_data
    expl = tree_shap(model.booster_, X.head(10))
    assert expl.values.shape == (10, X.shape[1])


def test_tree_shap_rejects_invalid_input(model_and_data):
    model, X = model_and_data
    with pytest.raises(ValueError, match="empty"):
        tree_shap(model, X.iloc[:0])
    with pytest.raises(ValueError, match="max_rows"):
        tree_shap(model, X, max_rows=0)


def test_tree_shap_reorders_columns_to_training_order(model_and_data):
    model, X = model_and_data
    expected = tree_shap(model, X.head(40))
    shuffled = X.head(40)[["year_built", "noise", "rooms", "area"]]  # e.g. built from a dict
    for fitted in (model, model.booster_):  # feature_names_in_ and Booster.feature_name()
        expl = tree_shap(fitted, shuffled)
        assert list(expl.feature_names) == list(X.columns)
        np.testing.assert_allclose(expl.values, expected.values)
        np.testing.assert_allclose(expl.data.astype(float), expected.data.astype(float))
        assert global_importance(expl).loc[0, "feature"] == "area"


def test_tree_shap_rejects_columns_that_do_not_match_the_model(model_and_data):
    model, X = model_and_data
    with pytest.raises(ValueError, match="missing \\['noise'\\]"):
        tree_shap(model, X.drop(columns="noise"))
    with pytest.raises(ValueError, match="unexpected \\['extra'\\]"):
        tree_shap(model, X.assign(extra=1.0))


def test_tree_shap_with_pandas_categorical_feature():
    rng = np.random.default_rng(3)
    n = 300
    canton = pd.Categorical(rng.choice(["ZH", "GE", "BE"], n))
    X = pd.DataFrame({"area": rng.uniform(30, 150, n), "canton": canton})
    effect = pd.Series(canton).map({"ZH": 0.4, "GE": 0.3, "BE": 0.0}).astype(float)
    y_log = 5.0 + 0.9 * np.log(X["area"]) + effect.to_numpy()
    model = lgb.LGBMRegressor(n_estimators=40, min_child_samples=5, random_state=42, verbose=-1)
    model.fit(X, y_log)
    expl = tree_shap(model, X.head(30))
    reconstructed = expl.values.sum(axis=1) + expl.base_values
    np.testing.assert_allclose(reconstructed, model.predict(X.head(30)), atol=1e-6)
    drivers = top_drivers(expl, row=0, k=2).set_index("feature")
    assert drivers.loc["canton", "value"] == X["canton"].iloc[0]
    assert isinstance(drivers.loc["canton", "value"], str)


def test_tree_shap_matches_lightgbm_names_with_whitespace(model_and_data):
    model, X = model_and_data
    renamed = X.rename(columns={"year_built": "year built"})
    spaced = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
    spaced.fit(renamed, model.predict(X))
    assert "year_built" in spaced.booster_.feature_name()  # LightGBM replaced the blank
    expl = tree_shap(spaced.booster_, renamed[list(reversed(renamed.columns))].head(5))
    assert list(expl.feature_names) == list(renamed.columns)


def test_top_drivers_sorted_by_absolute_shap(model_and_data):
    model, X = model_and_data
    expl = tree_shap(model, X, max_rows=30)
    drivers = top_drivers(expl, row=3, k=3)
    assert list(drivers.columns) == ["feature", "value", "shap_log", "approx_pct"]
    assert len(drivers) == 3
    abs_shap = drivers["shap_log"].abs().to_numpy()
    assert np.all(np.diff(abs_shap) <= 0)
    expected_top = X.columns[np.argmax(np.abs(expl.values[3]))]
    assert drivers.loc[0, "feature"] == expected_top
    np.testing.assert_allclose(drivers["approx_pct"], 100 * (np.exp(drivers["shap_log"]) - 1))
    row_label = expl.instance_names[3]
    assert drivers.loc[0, "value"] == pytest.approx(X.loc[row_label, expected_top], nan_ok=True)


def test_top_drivers_caps_k_and_validates_arguments(model_and_data):
    model, X = model_and_data
    expl = tree_shap(model, X.head(10))
    assert len(top_drivers(expl, row=-1, k=99)) == X.shape[1]
    with pytest.raises(ValueError, match="k must be"):
        top_drivers(expl, row=0, k=0)
    with pytest.raises(IndexError):
        top_drivers(expl, row=10)
    with pytest.raises(ValueError, match="2-D"):
        top_drivers(expl[0], row=0)


def test_global_importance_ranks_area_first(model_and_data):
    model, X = model_and_data
    table = global_importance(tree_shap(model, X, max_rows=200))
    assert table.loc[0, "feature"] == "area"
    assert set(table["feature"]) == set(X.columns)
    assert np.all(np.diff(table["mean_abs_shap"].to_numpy()) <= 0)
    assert table["share"].sum() == pytest.approx(1.0)


def test_global_importance_rejects_empty_explanation():
    empty = shap.Explanation(values=np.zeros((0, 2)), feature_names=["a", "b"])
    with pytest.raises(ValueError, match="no rows"):
        global_importance(empty)


def test_global_importance_all_zero_shap_gives_zero_share():
    flat = shap.Explanation(values=np.zeros((4, 2)), base_values=np.zeros(4))
    table = global_importance(flat)
    assert list(table["feature"]) == ["f0", "f1"]
    assert (table["share"] == 0).all()


def test_shap_frame_uses_instance_names(model_and_data):
    model, X = model_and_data
    expl = tree_shap(model, X.head(8))
    frame = shap_frame(expl)
    assert frame.shape == (8, X.shape[1])
    assert list(frame.index) == list(X.index[:8])
    no_names = shap_frame(shap.Explanation(values=np.ones((2, 3))))
    assert list(no_names.columns) == ["f0", "f1", "f2"]
    assert list(no_names.index) == [0, 1]
