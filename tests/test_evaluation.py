"""Tests for rentml.evaluation (synthetic data, fixed seeds)."""

import logging
import time

import numpy as np
import pandas as pd
import pytest

from rentml.evaluation import (
    PairedComparison,
    bootstrap_ci,
    comparison_table,
    holm_correction,
    learning_curve,
    paired_bootstrap_mae,
    power_mde,
    regression_metrics,
    segment_metrics,
)


@pytest.fixture
def rents() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True rents and two models; B's errors are 10 % smaller than A's on every listing."""
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(7.6, 0.35, 3000))
    noise = rng.normal(0.0, 300.0, y.size)
    return y, y + noise, y + 0.9 * noise


def test_regression_metrics_known_values() -> None:
    m = regression_metrics([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])
    assert m["MAE"] == pytest.approx(50 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(1100 / 3))
    assert m["MedAE"] == pytest.approx(10.0)
    assert m["MAPE"] == pytest.approx(100 * (0.1 + 0.05 + 0.1) / 3)
    assert m["R2"] == pytest.approx(1 - 1100 / 20000)


def test_regression_metrics_edge_cases() -> None:
    m = regression_metrics(pd.Series([0.0, 100.0]), np.array([10.0, 90.0]))
    assert m["MAPE"] == pytest.approx(10.0)
    assert np.isnan(regression_metrics([5.0, 5.0], [4.0, 6.0])["R2"])
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        regression_metrics([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        regression_metrics([], [])


def test_segment_metrics_per_segment_and_bias() -> None:
    y = np.array([100.0, 100.0, 200.0, 200.0])
    pred = np.array([90.0, 110.0, 180.0, 180.0])
    seg = pd.Series(pd.Categorical(["b", "b", "a", "a"], categories=["b", "a"]), name="band")
    out = segment_metrics(y, pred, seg)
    assert list(out.index) == ["b", "a"]
    assert out.index.name == "band"
    assert out.loc["b", "n"] == 2
    assert out.loc["b", "MAE"] == pytest.approx(10.0)
    assert out.loc["b", "bias"] == pytest.approx(0.0)
    assert out.loc["a", "bias"] == pytest.approx(-20.0)
    assert out.loc["a", "MAPE"] == pytest.approx(10.0)


def test_segment_metrics_missing_labels_and_length_check() -> None:
    seg = pd.Series(["x", None, "x"], index=[10, 20, 30])
    out = segment_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0], seg)
    assert out["n"].sum() == 3
    assert len(out) == 2
    with pytest.raises(ValueError):
        segment_metrics([1.0, 2.0], [1.0, 2.0], pd.Series(["a"]))


def test_bootstrap_ci_contains_true_mae() -> None:
    rng = np.random.default_rng(2)
    y = rng.uniform(1000, 3000, 2000)
    pred = y + rng.normal(0, 100, y.size)
    point, lo, hi = bootstrap_ci(y, pred, "MAE", n_boot=500)
    true_mae = 100 * np.sqrt(2 / np.pi)
    assert lo < true_mae < hi
    assert lo < point < hi
    assert point == pytest.approx(np.mean(np.abs(y - pred)))


def test_bootstrap_ci_callable_and_clusters() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0, 100, 300)
    err = np.repeat(base, 5)  # five identical duplicates per object
    y = np.full(err.size, 2000.0)
    groups = np.repeat(np.arange(300), 5)
    _, lo_iid, hi_iid = bootstrap_ci(y, y + err, lambda a, b: float(np.mean(np.abs(a - b))))
    _, lo_cl, hi_cl = bootstrap_ci(y, y + err, "MAE", groups=groups)
    assert (hi_cl - lo_cl) > 1.5 * (hi_iid - lo_iid)


def test_bootstrap_ci_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], [1.0, 2.0], "MAE", ci=1.5)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], [1.0, 2.0], "MAE", n_boot=0)
    with pytest.raises(KeyError):
        bootstrap_ci([1.0, 2.0], [1.0, 2.0], "SMAPE")
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], [1.0, 2.0], "MAE", groups=[1])


def test_paired_bootstrap_identical_models_p_one(rents: tuple) -> None:
    y, pred_a, _ = rents
    res = paired_bootstrap_mae(y, pred_a, pred_a.copy(), n_boot=300)
    assert res.diff == 0.0
    assert res.ci_low == 0.0 and res.ci_high == 0.0
    assert res.p_bootstrap == pytest.approx(1.0)
    assert res.p_wilcoxon == pytest.approx(1.0)


def test_paired_bootstrap_detects_better_model(rents: tuple) -> None:
    y, pred_a, pred_b = rents
    res = paired_bootstrap_mae(y, pred_a, pred_b, name_a="lgbm", name_b="gpboost", n_boot=1000)
    assert isinstance(res, PairedComparison)
    assert res.diff == pytest.approx(res.mae_a - res.mae_b)
    assert res.diff > 0 and res.ci_low > 0
    assert res.p_bootstrap < 0.01 and res.p_wilcoxon < 0.01
    assert res.rel_diff == pytest.approx(0.1, abs=1e-9)
    assert res.n == y.size


def test_paired_bootstrap_resamples_jointly(rents: tuple) -> None:
    y, pred_a, pred_b = rents
    paired = paired_bootstrap_mae(y, pred_a, pred_b, n_boot=500)
    _, lo_a, hi_a = bootstrap_ci(y, pred_a, "MAE", n_boot=500)
    # Pairing removes the listing-level variance shared by both models.
    assert (paired.ci_high - paired.ci_low) < 0.5 * (hi_a - lo_a)


def test_paired_bootstrap_null_p_values_not_small() -> None:
    rng = np.random.default_rng(4)
    y = rng.uniform(1000, 3000, 1500)
    res = paired_bootstrap_mae(
        y, y + rng.normal(0, 200, y.size), y + rng.normal(0, 200, y.size), n_boot=500
    )
    assert res.ci_low < 0 < res.ci_high
    assert res.p_bootstrap > 0.05


def test_paired_bootstrap_with_groups_and_errors(rents: tuple) -> None:
    y, pred_a, pred_b = rents
    groups = np.arange(y.size) // 3
    res = paired_bootstrap_mae(y, pred_a, pred_b, n_boot=300, groups=groups)
    assert 0 < res.p_bootstrap <= 1
    with pytest.raises(ValueError):
        paired_bootstrap_mae(y, pred_a, pred_b[:-1])


def test_holm_known_example() -> None:
    out = holm_correction({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert list(out.index) == ["a", "b", "c", "d"]
    expected = {"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02}
    for name, value in expected.items():
        assert out.loc[name, "p_holm"] == pytest.approx(value)
    assert out["reject"].tolist() == [True, False, False, True]


def test_holm_caps_monotone_and_nan() -> None:
    out = holm_correction({"x": 0.6, "y": 0.5, "z": np.nan})
    assert out.loc["x", "p_holm"] == pytest.approx(1.0)
    assert out.loc["y", "p_holm"] == pytest.approx(1.0)
    assert np.isnan(out.loc["z", "p_holm"]) and not out.loc["z", "reject"]
    assert holm_correction({}).empty
    with pytest.raises(ValueError):
        holm_correction({"bad": 1.5})


def test_comparison_table_adds_holm(rents: tuple) -> None:
    y, pred_a, pred_b = rents
    results = [
        paired_bootstrap_mae(y, pred_a, pred_b, name_a="A", name_b="B", n_boot=200),
        paired_bootstrap_mae(y, pred_a, pred_a, name_a="A", name_b="A2", n_boot=200),
    ]
    table = comparison_table(results)
    assert table["comparison"].tolist() == ["A vs B", "A vs A2"]
    assert (table["p_holm"] >= table["p_bootstrap"]).all()
    assert table["reject"].tolist() == [True, False]
    assert table.loc[0, "rel_diff_pct"] == pytest.approx(10.0)
    empty = comparison_table([])
    assert empty.empty and {"p_holm", "reject", "diff"} <= set(empty.columns)


def test_power_mde_shape_monotone_and_fast() -> None:
    rng = np.random.default_rng(5)
    pool = np.abs(rng.standard_t(3, 4000)) * 300
    start = time.perf_counter()
    out = power_mde(pool, n_sim=500)
    assert time.perf_counter() - start < 20
    assert len(out) == 5 * 10
    assert {"test_size", "rel_effect", "effect_chf", "power", "mcse", "mde_rel", "mde_chf"} <= set(
        out.columns
    )
    wide = out.pivot(index="rel_effect", columns="test_size", values="power")
    assert (wide.diff().dropna() >= -0.05).all().all()
    assert (wide.T.diff().dropna() >= -0.05).all().all()
    mde = out.groupby("test_size")["mde_rel"].first()
    assert mde.is_monotonic_decreasing
    assert out.loc[out["rel_effect"] == 0.10, "power"].min() > 0.9


def test_power_mde_null_size_and_perfect_correlation() -> None:
    rng = np.random.default_rng(6)
    pool = rng.exponential(300, 2000)
    null = power_mde(pool, test_sizes=(3000,), rel_effects=(0.0,), n_sim=400)
    assert null["power"].iloc[0] < 0.08
    assert np.isnan(null["mde_rel"].iloc[0])
    perfect = power_mde(pool, test_sizes=(500,), rel_effects=(0.01,), n_sim=50, error_corr=1.0)
    assert perfect["power"].iloc[0] == 1.0


def test_power_mde_uses_second_model_and_validates() -> None:
    rng = np.random.default_rng(7)
    pool = rng.exponential(300, 1000)
    other = 0.5 * pool + 0.5 * rng.exponential(300, 1000)
    weak = power_mde(pool, test_sizes=(2000,), rel_effects=(0.03,), n_sim=300, abs_err_b=other)
    strong = power_mde(pool, test_sizes=(2000,), rel_effects=(0.03,), n_sim=300, error_corr=0.95)
    assert strong["power"].iloc[0] > weak["power"].iloc[0]
    with pytest.raises(ValueError):
        power_mde(-pool)
    with pytest.raises(ValueError):
        power_mde(pool, error_corr=1.2)
    with pytest.raises(ValueError):
        power_mde(pool, rel_effects=(1.0,))
    with pytest.raises(ValueError):
        power_mde(pool, abs_err_b=other[:10])


def _linear_problem() -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(1000, 20))
    y = 2000 + x @ rng.normal(0, 100, 20) + rng.normal(0, 50, 1000)
    positions = rng.permutation(1000)
    folds = [(np.setdiff1d(positions, positions[k::4]), np.sort(positions[k::4])) for k in range(4)]
    return x, y, folds


def test_learning_curve_error_decreases_with_data() -> None:
    x, y, folds = _linear_problem()

    def fit_predict(train: np.ndarray, val: np.ndarray) -> np.ndarray:
        design = np.c_[np.ones(len(train)), x[train]]
        coef, *_ = np.linalg.lstsq(design, y[train], rcond=None)
        return np.c_[np.ones(len(val)), x[val]] @ coef

    out = learning_curve(fit_predict, len(y), fractions=(0.04, 0.2, 1.0), folds=folds, y=y)
    assert out["fraction"].tolist() == [0.04, 0.2, 1.0]
    assert out["n_train_mean"].is_monotonic_increasing
    assert out["mae_mean"].is_monotonic_decreasing
    assert (out["n_folds"] == 4).all() and (out["mae_std"] >= 0).all()


def test_learning_curve_nested_subsets_and_validation() -> None:
    x, y, folds = _linear_problem()
    seen: list[tuple[np.ndarray, np.ndarray]] = []

    def record(train: np.ndarray, val: np.ndarray) -> np.ndarray:
        seen.append((train, val))
        return np.full(len(val), y[train].mean())

    learning_curve(record, len(y), fractions=(0.5, 1.0), folds=folds[:1], y=y)
    (small, val_small), (full, val_full) = seen
    assert set(small) <= set(full) and len(full) == len(folds[0][0])
    assert np.array_equal(val_small, val_full)
    with pytest.raises(ValueError):
        learning_curve(record, len(y) + 1, folds=folds, y=y)
    with pytest.raises(ValueError):
        learning_curve(record, len(y), fractions=(0.0,), folds=folds, y=y)
    with pytest.raises(ValueError):
        learning_curve(record, 10, folds=folds, y=y[:10])


def test_paired_bootstrap_groups_wilcoxon_not_anticonservative() -> None:
    rng = np.random.default_rng(9)
    y = np.exp(rng.normal(7.6, 0.35, 300))
    pred_a, pred_b = y + rng.normal(0, 200, y.size), y + rng.normal(0, 200, y.size)
    single = paired_bootstrap_mae(y, pred_a, pred_b, n_boot=50)
    dup = paired_bootstrap_mae(
        *(np.repeat(v, 3) for v in (y, pred_a, pred_b)),
        n_boot=50,
        groups=np.repeat(np.arange(y.size), 3),
    )
    assert dup.p_wilcoxon == pytest.approx(single.p_wilcoxon)  # one pair per object
    assert dup.diff == pytest.approx(single.diff) and dup.n == 3 * single.n
    rejections = {"listing": 0, "object": 0}
    n_rep = 200
    for _ in range(n_rep):  # H0 with duplicated objects of size 1-5
        sizes = rng.choice([1, 1, 2, 3, 5], 150)
        obj = np.repeat(np.arange(150), sizes)
        truth = np.full(obj.size, 2000.0)
        err_a = np.repeat(rng.normal(0, 200, 150), sizes) + rng.normal(0, 10, obj.size)
        err_b = np.repeat(rng.normal(0, 200, 150), sizes) + rng.normal(0, 10, obj.size)
        naive = paired_bootstrap_mae(truth, truth + err_a, truth + err_b, n_boot=1)
        grouped = paired_bootstrap_mae(truth, truth + err_a, truth + err_b, n_boot=1, groups=obj)
        rejections["listing"] += naive.p_wilcoxon < 0.05
        rejections["object"] += grouped.p_wilcoxon < 0.05
    assert rejections["object"] / n_rep < 0.10
    assert rejections["listing"] / n_rep > 0.15  # why grouping is needed


def test_paired_bootstrap_rejects_single_listing_and_flags_degenerate() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        paired_bootstrap_mae([1000.0], [900.0], [950.0], n_boot=50)
    with pytest.raises(ValueError, match="at least 2"):
        paired_bootstrap_mae([1000.0, 1200.0], [900.0, 1100.0], [950.0, 1150.0], groups=["o", "o"])
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci([1000.0], [900.0], "MAE")
    y = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    const = paired_bootstrap_mae(y, y - 100.0, y - 50.0, n_boot=50)
    assert const.diff == pytest.approx(50.0)
    assert np.isnan(const.p_bootstrap)
    assert const.ci_low == pytest.approx(50.0) and const.ci_high == pytest.approx(50.0)
    cluster_const = paired_bootstrap_mae(
        y, y - np.array([140.0, 60.0, 100.0, 100.0]), y - 50.0, n_boot=50, groups=[1, 1, 2, 2]
    )
    assert np.isnan(cluster_const.p_bootstrap)
    varying = paired_bootstrap_mae(y, y - np.array([140.0, 60.0, 100.0, 100.0]), y - 50.0)
    assert 0 < varying.p_bootstrap <= 1


def test_segment_metrics_label_alignment() -> None:
    y = pd.Series([100.0, 200.0, 300.0], index=[3, 1, 2])
    pred = y.to_numpy() + np.array([1.0, 2.0, 3.0])
    seg = pd.Series(["a", "b", "c"], index=[1, 2, 3], name="seg")
    out = segment_metrics(y, pred, seg)
    assert out["MAE"].to_dict() == {"a": 2.0, "b": 3.0, "c": 1.0}
    assert out.index.name == "seg"
    positional = segment_metrics(y, pred, np.array(["a", "b", "c"]))
    assert positional["MAE"].to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}
    with pytest.raises(ValueError, match="index"):
        segment_metrics(y, pred, pd.Series(["a", "b", "c"], index=[7, 8, 9]))


def test_power_mde_reports_rho_and_validates_second_model(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(10)
    pool = rng.exponential(300, 1000)
    kwargs = {"test_sizes": (500,), "rel_effects": (0.05,), "n_sim": 20}
    with caplog.at_level(logging.WARNING, logger="rentml.evaluation"):
        default = power_mde(pool, **kwargs)
    assert "assuming rho=0.80" in caplog.text
    assert default["rho"].iloc[0] == pytest.approx(0.8)
    assert default["pair_sd_ratio"].iloc[0] == pytest.approx(np.sqrt(0.4))
    other = 0.5 * pool + 0.5 * rng.exponential(300, 1000)
    emp = power_mde(pool, abs_err_b=other, **kwargs)
    assert emp["rho"].iloc[0] == pytest.approx(np.corrcoef(pool, other)[0, 1])
    assert emp["pair_sd_ratio"].iloc[0] == pytest.approx(np.std(pool - other) / np.std(pool))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rentml.evaluation"):
        neg = power_mde(pool, abs_err_b=pool.max() - pool, **kwargs)
    assert neg["rho"].iloc[0] == 0.0 and "independent pairing" in caplog.text
    assert neg["pair_sd_ratio"].iloc[0] == pytest.approx(np.sqrt(2.0))
    for bad in (np.full_like(pool, 5.0), np.r_[other[:-1], np.nan], -other):
        with pytest.raises(ValueError):
            power_mde(pool, abs_err_b=bad, **kwargs)


def test_power_mde_matches_brute_force_paired_bootstrap() -> None:
    rng = np.random.default_rng(11)
    pool = np.abs(rng.standard_t(3, 3000)) * 300
    other = np.abs(0.7 * pool + 0.3 * np.abs(rng.standard_t(3, 3000)) * 300)
    table = power_mde(pool, abs_err_b=other, test_sizes=(1500,), rel_effects=(0.02,), n_sim=400)
    w = 1.0 - table["pair_sd_ratio"].iloc[0] / np.sqrt(2.0)
    hits, n_rep = 0, 120
    for rep in range(n_rep):
        e_a = pool[rng.integers(0, pool.size, 1500)]
        e_b = 0.98 * (w * e_a + (1 - w) * pool[rng.integers(0, pool.size, 1500)])
        truth = np.full(1500, 2000.0)
        res = paired_bootstrap_mae(truth, truth + e_a, truth + e_b, n_boot=200, seed=rep)
        hits += res.p_bootstrap < 0.05 and res.diff > 0
    assert 0.15 < table["power"].iloc[0] < 0.6  # mid-range power: informative check
    assert abs(hits / n_rep - table["power"].iloc[0]) < 0.12


def test_learning_curve_subsamples_whole_groups() -> None:
    _, y, _ = _linear_problem()
    groups = np.arange(y.size) // 4  # 250 objects with 4 listings each
    folds = [(np.flatnonzero(groups % 4 != k), np.flatnonzero(groups % 4 == k)) for k in range(4)]
    seen: list[np.ndarray] = []

    def record(train: np.ndarray, val: np.ndarray) -> np.ndarray:
        seen.append(train)
        return np.full(len(val), y[train].mean())

    out = learning_curve(record, y.size, fractions=(0.5, 1.0), folds=folds[:1], y=y, groups=groups)
    small, full = seen
    assert set(small) <= set(full)
    per_object = np.bincount(groups[small])
    assert set(per_object[per_object > 0].tolist()) == {4}  # whole objects only
    assert out["n_groups_mean"].tolist() == [94.0, 187.0]
    assert out["n_train_mean"].tolist() == [376.0, 748.0]
    rows = learning_curve(record, y.size, fractions=(1.0,), folds=folds[:1], y=y)
    assert rows["n_groups_mean"].tolist() == rows["n_train_mean"].tolist()
    with pytest.raises(ValueError, match="outside"):
        learning_curve(record, y.size, folds=[(np.array([-1, 2]), np.array([3]))], y=y)
    with pytest.raises(ValueError, match="groups"):
        learning_curve(record, y.size, folds=folds, y=y, groups=groups[:10])
