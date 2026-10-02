"""Tests for rentml.fusion (entity-embedding + text fusion network)."""

import importlib
import logging
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import rentml
from rentml.fusion import (
    UNKNOWN_LABEL,
    FusedRentNet,
    FusedRentRegressor,
    FusionConfig,
    embedding_dim,
    is_torch_available,
)

torch = pytest.importorskip("torch")

N_TRAIN, N_EVAL = 450, 525  # rows [0, 450) train, [450, 525) early stopping, [525, 600) test
FAST = {"hidden": (64, 32), "epochs": 30, "batch_size": 64, "patience": 10, "device": "cpu"}


def _synthetic(n: int = 600, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    cantons = np.array(["ZH", "GE", "BE", "TI", "VS"])
    canton_effect = dict(zip(cantons, [0.35, 0.40, 0.0, -0.15, -0.20], strict=True))
    canton = rng.choice(cantons, n)
    muni_no = rng.integers(0, 8, n)
    municipality = pd.Series(canton).str.cat(pd.Series(muni_no).astype(str), sep="/")
    muni_effect = rng.normal(0, 0.1, 40)[pd.factorize(municipality, sort=True)[0]]
    area = rng.uniform(25, 160, n)
    rooms = np.clip(np.round(area / 28 * 2) / 2 + rng.normal(0, 0.3, n), 1, 8)
    elevation = rng.uniform(250, 1500, n)
    y = (
        3.1
        + 0.85 * np.log(area)
        + 0.02 * rooms
        - 0.0002 * elevation
        + pd.Series(canton).map(canton_effect).to_numpy()
        + muni_effect
        + rng.normal(0, 0.07, n)
    )
    index = pd.Index(np.arange(5000, 5000 + n), name="listing_id")
    X_num = pd.DataFrame({"area": area, "rooms": rooms, "elevation": elevation}, index=index)
    X_num.loc[X_num.index[::9], "elevation"] = np.nan
    X_cat = pd.DataFrame({"re_canton": canton, "re_municipality": municipality}, index=index)
    return X_num, X_cat, pd.Series(y, index=index, name="log_price")


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    return _synthetic()


@pytest.fixture(scope="module")
def fitted(data) -> FusedRentRegressor:
    X_num, X_cat, y = data
    tr, va = slice(0, N_TRAIN), slice(N_TRAIN, N_EVAL)
    reg = FusedRentRegressor(FusionConfig(**FAST), log1p_cols=("area",))
    return reg.fit(
        X_num.iloc[tr], X_cat.iloc[tr], y.iloc[tr], eval_set=(X_num[va], X_cat[va], y[va])
    )


def _mae_chf(y_log: np.ndarray, pred_log: np.ndarray) -> float:
    return float(np.mean(np.abs(np.exp(y_log) - np.exp(pred_log))))


def test_is_torch_available():
    assert is_torch_available()


def test_embedding_dim_rule():
    assert embedding_dim(26) == min(16, round(1.6 * 26**0.56))
    assert embedding_dim(2000) == 16
    assert embedding_dim(2000, max_dim=8) == 8
    assert embedding_dim(0) == 2


def test_fit_beats_mean_predictor_on_held_out_rows(data, fitted):
    X_num, X_cat, y = data
    y_test = y.iloc[N_EVAL:].to_numpy()  # never seen by training or early stopping
    pred = fitted.predict(X_num.iloc[N_EVAL:], X_cat.iloc[N_EVAL:])
    baseline = np.full_like(y_test, y.iloc[:N_TRAIN].mean())
    assert pred.shape == y_test.shape
    assert _mae_chf(y_test, pred) < 0.6 * _mae_chf(y_test, baseline)


def test_history_and_best_weights_are_restored(data, fitted):
    X_num, X_cat, y = data
    history = fitted.history_
    assert list(history.columns[:3]) == ["epoch", "train_loss", "val_mae"]
    assert 1 <= len(history) <= FAST["epochs"]
    assert fitted.best_val_mae_ == pytest.approx(history["val_mae"].min())
    assert history.loc[history["val_mae"].idxmin(), "epoch"] == fitted.best_epoch_
    pred = fitted.predict(X_num.iloc[N_TRAIN:N_EVAL], X_cat.iloc[N_TRAIN:N_EVAL])
    assert _mae_chf(y.iloc[N_TRAIN:N_EVAL].to_numpy(), pred) == pytest.approx(
        fitted.best_val_mae_, rel=1e-4
    )


def test_onecycle_with_eval_set_warms_up_and_restores_best_weights(data):
    X_num, X_cat, y = data
    tr, ev = slice(0, N_TRAIN), slice(N_TRAIN, N_EVAL)
    reg = FusedRentRegressor(**{**FAST, "epochs": 12, "patience": 3, "scheduler": "onecycle"})
    reg.fit(X_num[tr], X_cat[tr], y[tr], eval_set=(X_num[ev], X_cat[ev], y[ev]))
    lr = reg.history_["lr"].to_numpy()
    assert lr[0] < 0.1 * reg.config.lr  # OneCycle starts at lr / div_factor
    assert lr.max() > 5 * lr[0]
    assert reg.history_.loc[reg.history_["val_mae"].idxmin(), "epoch"] == reg.best_epoch_
    pred = reg.predict(X_num[ev], X_cat[ev])
    assert _mae_chf(y[ev].to_numpy(), pred) == pytest.approx(reg.best_val_mae_, rel=1e-4)


def test_unseen_categories_and_nan_numerics(data, fitted):
    X_num, X_cat, _ = data
    X_new = X_num.iloc[:4].copy()
    X_new.loc[:, "elevation"] = np.nan
    X_new.iloc[0, 0] = np.nan
    cat_new = X_cat.iloc[:4].copy()
    cat_new.loc[:, "re_municipality"] = ["XX/1", None, "ZH/0", "ZZ/9"]
    cat_new.iloc[3, 0] = "ZZ"
    pred = fitted.predict(X_new, cat_new)
    assert np.isfinite(pred).all()
    assert pred.min() > 4.0 and pred.max() < 10.0


def test_predict_rejects_missing_columns_and_misaligned_rows(data, fitted):
    X_num, X_cat, _ = data
    with pytest.raises(KeyError, match="area"):
        fitted.predict(X_num.drop(columns="area"), X_cat)
    with pytest.raises(ValueError, match="same length"):
        fitted.predict(X_num.iloc[:5], X_cat.iloc[:6])
    with pytest.raises(ValueError, match="X_text"):
        fitted.predict(X_num.iloc[:5], X_cat.iloc[:5], X_text=np.zeros((5, 4)))


def test_predictions_are_deterministic_for_same_seed(data):
    X_num, X_cat, y = data
    cfg = FusionConfig(**{**FAST, "epochs": 8})
    first = FusedRentRegressor(cfg).fit(X_num, X_cat, y).predict(X_num, X_cat)
    state_before = torch.get_rng_state()
    second = FusedRentRegressor(cfg).fit(X_num, X_cat, y).predict(X_num, X_cat)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)
    assert torch.equal(state_before, torch.get_rng_state())
    other = FusedRentRegressor(cfg, seed=7).fit(X_num, X_cat, y).predict(X_num, X_cat)
    assert not np.allclose(first, other)


def test_fit_leaves_global_cpu_and_cuda_rng_state_unchanged(data):
    X_num, X_cat, y = data
    gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=gpus):  # keep this test's own reseeding local
        torch.manual_seed(123)  # differs from config.seed, so a leaked reseed would show
        for device in ["cpu", "cuda"] if gpus else ["cpu"]:
            cpu_before = torch.get_rng_state()
            cuda_before = [torch.cuda.get_rng_state(i) for i in gpus]
            FusedRentRegressor(**{**FAST, "epochs": 2, "device": device}).fit(X_num, X_cat, y)
            assert torch.equal(cpu_before, torch.get_rng_state()), device
            for i, state in zip(gpus, cuda_before, strict=True):
                assert torch.equal(state, torch.cuda.get_rng_state(i)), device


def test_text_branch_with_random_embeddings(data):
    X_num, X_cat, y = data
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(y), 12))
    emb[::10] = np.nan  # listings without description
    reg = FusedRentRegressor(**{**FAST, "epochs": 5, "text_proj_dim": 8, "loss": "l1"})
    reg.fit(
        X_num.iloc[:N_TRAIN],
        X_cat.iloc[:N_TRAIN],
        y.iloc[:N_TRAIN],
        X_text=emb[:N_TRAIN],
        eval_set=(X_num.iloc[N_TRAIN:], X_cat.iloc[N_TRAIN:], y.iloc[N_TRAIN:], emb[N_TRAIN:]),
    )
    pred = reg.predict(X_num.iloc[N_TRAIN:], X_cat.iloc[N_TRAIN:], emb[N_TRAIN:])
    assert np.isfinite(pred).all()
    assert reg.model_.text is not None
    with pytest.raises(ValueError, match="X_text"):
        reg.predict(X_num.iloc[N_TRAIN:], X_cat.iloc[N_TRAIN:])
    with pytest.raises(ValueError, match="columns"):
        reg.predict(X_num.iloc[:3], X_cat.iloc[:3], np.zeros((3, 5)))


def test_text_inf_counts_as_missing_and_dataframe_index_is_checked(data):
    X_num, X_cat, y = data
    emb = pd.DataFrame(np.random.default_rng(1).normal(size=(len(y), 4)), index=y.index)
    reg = FusedRentRegressor(**{**FAST, "epochs": 2}).fit(X_num, X_cat, y, X_text=emb)
    with_inf, with_nan = emb.iloc[:3].copy(), emb.iloc[:3].copy()
    with_inf.iloc[0, 1], with_nan.iloc[0, 1] = np.inf, np.nan
    pred_inf = reg.predict(X_num.iloc[:3], X_cat.iloc[:3], with_inf)
    assert np.isfinite(pred_inf).all()
    np.testing.assert_allclose(pred_inf, reg.predict(X_num.iloc[:3], X_cat.iloc[:3], with_nan))
    np.testing.assert_allclose(
        reg.predict(X_num, X_cat, emb), reg.predict(X_num, X_cat, emb.to_numpy())
    )
    with pytest.raises(ValueError, match="X_text index"):
        reg.predict(X_num, X_cat, emb.sample(frac=1, random_state=0))
    with pytest.raises(ValueError, match="X_text index"):
        FusedRentRegressor(**FAST).fit(X_num, X_cat, y, X_text=emb.reset_index(drop=True))


def test_predict_on_empty_frame_returns_empty_array(data, fitted):
    X_num, X_cat, _ = data
    assert fitted.predict(X_num.iloc[:0], X_cat.iloc[:0]).shape == (0,)


def test_pickle_roundtrip_keeps_predictions(data, fitted, tmp_path):
    X_num, X_cat, _ = data
    path = tmp_path / "fusion.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    assert loaded.device_.type == "cpu"
    expected = fitted.predict(X_num.iloc[:5], X_cat.iloc[:5])
    np.testing.assert_allclose(loaded.predict(X_num.iloc[:5], X_cat.iloc[:5]), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_model_pickles_to_cpu(data, tmp_path):
    X_num, X_cat, y = data
    reg = FusedRentRegressor(**{**FAST, "epochs": 3, "device": "cuda"}).fit(X_num, X_cat, y)
    assert reg.device_.type == "cuda"
    joblib.dump(reg, tmp_path / "gpu.joblib")
    loaded = joblib.load(tmp_path / "gpu.joblib")
    assert loaded.device_.type == "cpu" and reg.device_.type == "cuda"
    np.testing.assert_allclose(
        loaded.predict(X_num, X_cat), reg.predict(X_num, X_cat), rtol=0, atol=1e-4
    )


def test_embedding_table_contains_unknown_row(data, fitted):
    _, X_cat, _ = data
    table = fitted.embedding_table("re_canton")
    n_known = X_cat.iloc[:N_TRAIN]["re_canton"].nunique()
    assert table.index[0] == UNKNOWN_LABEL
    assert len(table) == n_known + 1
    assert table.filter(like="emb_").shape[1] == embedding_dim(n_known, 16)
    assert table["n_train"].sum() == N_TRAIN
    with pytest.raises(KeyError):
        fitted.embedding_table("area")


def test_embed_returns_last_hidden_layer_consistent_with_predict(data, fitted):
    X_num, X_cat, _ = data
    rows = slice(N_EVAL, None)
    emb = fitted.embed(X_num[rows], X_cat[rows])
    assert emb.shape == (len(X_num[rows]), FAST["hidden"][-1])
    np.testing.assert_array_equal(emb, fitted.embed(X_num[rows], X_cat[rows]))  # no dropout
    with torch.no_grad():
        std_log = fitted.model_.out(torch.from_numpy(emb).float()).squeeze(-1).numpy()
    np.testing.assert_allclose(
        std_log * fitted.target_std_ + fitted.target_mean_,
        fitted.predict(X_num[rows], X_cat[rows]),
        atol=1e-5,
    )
    assert fitted.embed(X_num.iloc[:0], X_cat.iloc[:0]).shape == (0, FAST["hidden"][-1])


def test_rare_categories_map_to_unknown():
    X_num = pd.DataFrame({"x": np.linspace(0, 1, 40)})
    X_cat = pd.DataFrame({"muni": ["a"] * 20 + ["b"] * 19 + ["rare"]})
    y = 7 + X_num["x"] + (X_cat["muni"] == "a") * 0.2
    reg = FusedRentRegressor(epochs=3, batch_size=16, device="cpu", min_category_count=2)
    table = reg.fit(X_num, X_cat, y).embedding_table("muni")
    assert list(table.index) == [UNKNOWN_LABEL, "a", "b"]
    assert table.loc[UNKNOWN_LABEL, "n_train"] == 1


def test_float_upcast_ids_match_integer_training_ids():
    X_num = pd.DataFrame({"x": np.arange(30, dtype=float)})
    ids_int = pd.DataFrame({"muni": np.repeat([261, 262, 263], 10)})
    y = 7 + 0.01 * X_num["x"] + np.repeat([0.0, 0.3, 0.6], 10)
    reg = FusedRentRegressor(epochs=2, batch_size=8, device="cpu").fit(X_num, ids_int, y)
    assert list(reg.embedding_table("muni").index) == [UNKNOWN_LABEL, "261", "262", "263"]
    x_new = X_num.iloc[:3]
    as_int = pd.DataFrame({"muni": [261, 263, 999]})  # unseen 999 -> unknown, like NaN
    expected = reg.predict(x_new, as_int)
    variants = {
        "float": [261.0, 263.0, np.nan],
        "category(float)": pd.Categorical([261.0, 263.0, np.nan]),
        "object(float)": pd.Series([261.0, 263.0, None], dtype=object),
        "str": ["261", "263", None],
    }
    for name, values in variants.items():
        pred = reg.predict(x_new, pd.DataFrame({"muni": values}))
        np.testing.assert_allclose(pred, expected, err_msg=name)
    unknown = reg.predict(x_new, pd.DataFrame({"muni": [999, 999, 999]}))
    assert not np.allclose(unknown[:2], expected[:2])  # known ids really hit their embeddings
    as_category = ids_int.astype(float).astype("category")  # e.g. a notebook's category column
    reg_cat = FusedRentRegressor(epochs=2, batch_size=8, device="cpu").fit(X_num, as_category, y)
    assert list(reg_cat.embedding_table("muni").index) == [UNKNOWN_LABEL, "261", "262", "263"]
    ids_261_263 = as_category.iloc[[0, 20]].reset_index(drop=True)
    np.testing.assert_allclose(
        reg_cat.predict(x_new.iloc[:2], ids_261_263), reg_cat.predict(x_new.iloc[:2], as_int[:2])
    )


def test_training_without_eval_set_uses_all_epochs(data):
    X_num, X_cat, y = data
    reg = FusedRentRegressor(**{**FAST, "epochs": 4, "scheduler": "onecycle"})
    reg.fit(X_num, X_cat, y)
    assert len(reg.history_) == 4
    assert reg.history_["val_mae"].isna().all()
    assert np.isnan(reg.best_val_mae_)
    assert reg.best_epoch_ == 4


def test_categorical_only_and_numeric_only_models(data):
    X_num, X_cat, y = data
    cat_only = FusedRentRegressor(**{**FAST, "epochs": 3}).fit(X_num[[]], X_cat, y)
    assert np.isfinite(cat_only.predict(X_num[[]], X_cat)).all()
    num_only = FusedRentRegressor(**{**FAST, "epochs": 3}).fit(X_num, None, y)
    assert np.isfinite(num_only.predict(X_num, None)).all()


def test_fit_rejects_invalid_inputs(data):
    X_num, X_cat, y = data
    reg = FusedRentRegressor(**FAST)
    with pytest.raises(ValueError, match="index"):
        reg.fit(X_num, X_cat, y.reset_index(drop=True))
    bad_y = y.copy()
    bad_y.iloc[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        reg.fit(X_num, X_cat, bad_y)
    with pytest.raises(ValueError, match="No input features"):
        reg.fit(X_num[[]], None, y)
    with pytest.raises(ValueError, match="eval_set"):
        reg.fit(X_num, X_cat, y, eval_set=(X_num, X_cat))


def test_all_nan_column_and_unknown_log1p_column(data, caplog):
    X_num, X_cat, y = data
    X_bad = X_num.assign(empty=np.nan)
    reg = FusedRentRegressor(**{**FAST, "epochs": 2}, log1p_cols=("area", "nope"))
    with caplog.at_level(logging.WARNING, logger="rentml.fusion"), pytest.warns(RuntimeWarning):
        reg.fit(X_bad, X_cat, y)
    assert "nope" in caplog.text
    assert np.isfinite(reg.predict(X_bad, X_cat)).all()


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        FusedRentRegressor().predict(pd.DataFrame({"x": [1.0]}), None)
    with pytest.raises(NotFittedError):
        FusedRentRegressor().embed(pd.DataFrame({"x": [1.0]}), None)


def test_config_validation():
    with pytest.raises(ValueError, match="loss"):
        FusionConfig(loss="quantile")
    with pytest.raises(ValueError, match="scheduler"):
        FusionConfig(scheduler="cosine")
    with pytest.raises(ValueError, match="dropout"):
        FusionConfig(dropout=1.0)
    assert FusionConfig(hidden=[32]).hidden == (32,)
    with pytest.raises(ValueError, match="hidden"):
        FusionConfig(hidden=())
    assert FusionConfig(log1p_cols="area").log1p_cols == ("area",)


def test_cuda_request_without_cuda_raises(data, monkeypatch):
    X_num, X_cat, y = data
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        FusedRentRegressor(**{**FAST, "device": "cuda"}).fit(X_num, X_cat, y)


def test_net_forward_shapes():
    net = FusedRentNet(3, [5, 9], text_dim=6)
    out = net(torch.zeros(4, 3), torch.zeros(4, 2, dtype=torch.long), torch.zeros(4, 6))
    assert out.shape == (4,)
    with pytest.raises(ValueError, match="x_text"):
        net(torch.zeros(4, 3), torch.zeros(4, 2, dtype=torch.long))


def test_module_imports_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    monkeypatch.delitem(sys.modules, "rentml.fusion")
    monkeypatch.delattr(rentml, "fusion", raising=False)
    fusion = importlib.import_module("rentml.fusion")
    assert not fusion.is_torch_available()
    assert fusion.FusionConfig(epochs=3).epochs == 3
    with pytest.raises(ImportError, match="PyTorch"):
        fusion.FusedRentRegressor().fit(pd.DataFrame({"x": [1.0, 2.0]}), None, np.array([7.0, 7.5]))
    with pytest.raises(ImportError, match="PyTorch"):
        fusion.FusedRentNet(1, [])
