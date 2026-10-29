"""Tests for rentml.nn (skipped without the optional torch extra)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from rentml.nn import UNKNOWN_LABEL, FusedRentRegressor  # noqa: E402 -- after importorskip


def _data(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    canton = rng.choice(["ZH", "BE", "GE"], n)
    canton[0] = "JU"  # single training object -> unknown index
    area = rng.uniform(30, 150, n)
    X_num = pd.DataFrame({"area": area, "oev": rng.uniform(0, 4, n)})
    X_num.loc[::7, "oev"] = np.nan
    X_cat = pd.DataFrame({"re_canton": canton})
    y = 7 + 0.8 * np.log(area / 80) + pd.Series(canton).map({"ZH": 0.3, "GE": 0.4}).fillna(0)
    return X_num, X_cat, (y + rng.normal(0, 0.05, n)).to_numpy()


def test_fused_net_learns_and_stops_early() -> None:
    X_num, X_cat, y = _data()
    tr, va = slice(0, 300), slice(300, None)
    model = FusedRentRegressor(epochs=30, patience=5, log1p_cols=["area"], device="cpu")
    model.fit(X_num[tr], X_cat[tr], y[tr], eval_set=(X_num[va], X_cat[va], y[va]))
    hist = model.history_
    assert list(hist.columns) == ["epoch", "train_loss", "val_mae"]
    assert model.best_val_mae_ == hist["val_mae"].min()
    assert model.best_val_mae_ < np.mean(np.abs(np.exp(y[va]) - np.exp(y[tr].mean())))
    pred = model.predict(X_num[va], X_cat[va])
    assert np.mean(np.abs(np.exp(pred) - np.exp(y[va]))) == pytest.approx(model.best_val_mae_)
    table = model.embedding_table("re_canton")
    assert table.index[0] == UNKNOWN_LABEL
    assert table.loc[UNKNOWN_LABEL, "n_train"] == 1  # JU
    assert table["n_train"].sum() == 300


def test_fused_net_text_branch_and_unseen_units() -> None:
    X_num, X_cat, y = _data(n=120)
    text = np.random.default_rng(1).normal(size=(120, 4))
    text[::5] = np.nan  # no description
    model = FusedRentRegressor(epochs=2, device="cpu").fit(X_num, X_cat, y, X_text=text)
    assert model.best_epoch_ == 2  # no eval_set: last epoch kept
    new_cat = X_cat.head(3).assign(re_canton=["XX", None, "ZH"])
    pred = model.predict(X_num.head(3), new_cat, text[:3])
    assert np.isfinite(pred).all()
