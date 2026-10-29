"""Tests for rentml.evaluation (synthetic data, fixed seeds, temporary MLflow store)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from rentml import evaluation as ev

Rents = tuple[np.ndarray, np.ndarray, np.ndarray]


@pytest.fixture
def rents() -> Rents:
    """True rents and two models; B's errors are 10 % smaller than A's on every row."""
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(7.6, 0.35, 2000))
    noise = rng.normal(0.0, 300.0, y.size)
    return y, y + noise, y + 0.9 * noise


def test_regression_metrics_known_values() -> None:
    m = ev.regression_metrics([100.0, 200.0, 300.0], [110.0, 190.0, 330.0])
    assert m["MAE"] == pytest.approx(50 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(1100 / 3))
    assert m["MedAE"] == pytest.approx(10.0)
    assert m["MAPE"] == pytest.approx(100 * (0.1 + 0.05 + 0.1) / 3)
    assert m["R2"] == pytest.approx(1 - 1100 / 20000)


def test_segment_metrics_bias_and_category_order() -> None:
    seg = pd.Series(pd.Categorical(["b", "a", "b", "a"], categories=["b", "a"]), index=[9, 8, 7, 6])
    out = ev.segment_metrics([100.0, 200.0, 100.0, 200.0], [110.0, 180.0, 130.0, 200.0], seg)
    assert list(out.index) == ["b", "a"]
    assert out.loc["b", "n"] == 2
    assert out.loc["b", "MAE"] == pytest.approx(20.0)
    assert out.loc["b", "bias"] == pytest.approx(20.0)
    assert out.loc["a", "bias"] == pytest.approx(-10.0)
    assert out.loc["a", "MAPE"] == pytest.approx(5.0)


def test_bootstrap_ci_contains_point_and_is_seeded(rents: Rents) -> None:
    y, pred_a, _ = rents
    point, lo, hi = ev.bootstrap_ci(y, pred_a, "MAE", n_boot=300)
    assert lo < point < hi
    assert ev.bootstrap_ci(y, pred_a, "MAE", n_boot=300) == (point, lo, hi)


def test_paired_bootstrap_detects_better_model(rents: Rents) -> None:
    y, pred_a, pred_b = rents
    res = ev.paired_bootstrap_mae(y, pred_a, pred_b, n_boot=500)
    assert res.diff == pytest.approx(res.mae_a - res.mae_b)
    assert res.diff / res.mae_a == pytest.approx(0.1)
    assert 0 < res.ci_low < res.diff < res.ci_high
    assert res.p_bootstrap == pytest.approx(1 / 501)
    assert res.p_wilcoxon < 1e-10


def test_paired_bootstrap_no_difference_for_swapped_noise() -> None:
    rng = np.random.default_rng(3)
    y = np.full(400, 2000.0)
    noise = rng.normal(0, 200, 400)
    res = ev.paired_bootstrap_mae(y, y + noise, y - noise[::-1], n_boot=500)
    assert res.ci_low < 0 < res.ci_high or res.diff == 0
    assert res.p_bootstrap > 0.05


def test_holm_correction_matches_hand_computation() -> None:
    out = ev.holm_correction({"S1": 0.01, "S2": 0.04, "S3": 0.03}, alpha=0.05)
    assert out["p_holm"].tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert out["reject"].tolist() == [True, False, False]


def test_comparison_table_rows(rents: Rents) -> None:
    y, pred_a, pred_b = rents
    results = {
        "B vs A": ev.paired_bootstrap_mae(y, pred_a, pred_b, n_boot=200),
        "A vs A": ev.paired_bootstrap_mae(y, pred_a, pred_a + 1e-9, n_boot=200),
    }
    table = ev.comparison_table(results)
    assert list(table.index) == ["B vs A", "A vs A"]
    assert table.loc["B vs A", "rel_diff_pct"] == pytest.approx(10.0)
    assert table["reject"].tolist() == [True, False]


@pytest.fixture
def abs_errors() -> tuple[np.ndarray, np.ndarray]:
    """Absolute errors of two models whose raw errors correlate with rho = 0.8."""
    rng = np.random.default_rng(2)
    noise_a, other = rng.normal(0.0, 300.0, (2, 3000))
    return np.abs(noise_a), np.abs(0.8 * noise_a + 0.6 * other)


def test_power_mde_increases_with_effect_and_size(
    abs_errors: tuple[np.ndarray, np.ndarray],
) -> None:
    out = ev.power_mde(*abs_errors, test_sizes=(200, 2000), rel_effects=(0.0, 0.05, 0.2), n_sim=200)
    small, large = (out[out["test_size"] == n].set_index("rel_effect") for n in (200, 2000))
    assert small.loc[0.0, "power"] < 0.1
    assert small.loc[0.2, "power"] > small.loc[0.05, "power"]
    assert large.loc[0.05, "power"] > small.loc[0.05, "power"]
    assert large["mde_rel"].iloc[0] <= small["mde_rel"].iloc[0]


def test_power_mde_unreachable_effect_gives_nan(abs_errors: tuple[np.ndarray, np.ndarray]) -> None:
    out = ev.power_mde(*abs_errors, test_sizes=(50,), rel_effects=(0.001,), n_sim=50)
    assert out["mde_rel"].isna().all()


def test_learning_curve_uses_nested_subsets() -> None:
    y = np.arange(100, dtype=float)
    folds = [(np.arange(0, 80), np.arange(80, 100)), (np.arange(20, 100), np.arange(0, 20))]
    seen: list[np.ndarray] = []

    def fit_predict(train: np.ndarray, val: np.ndarray) -> np.ndarray:
        seen.append(train)
        return np.full(len(val), y[train].mean())

    lc = ev.learning_curve(fit_predict, folds, y, fractions=(0.5, 1.0))
    assert lc["n_train_mean"].tolist() == [40, 80]
    assert set(seen[0]) <= set(seen[1])
    assert {"mae_mean", "mae_std", "rmse_mean", "rmse_std"} <= set(lc.columns)


def test_morans_i_clustered_vs_random() -> None:
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1000, size=(300, 2))
    clustered = coords[:, 0] / 1000 + rng.normal(0, 0.05, 300)
    res = ev.morans_i(clustered, coords, permutations=199)
    assert res.I > 0.5
    assert res.p_value == pytest.approx(1 / 200)
    assert res.expected == pytest.approx(-1 / 299)
    noise = ev.morans_i(rng.normal(size=300), coords, permutations=199)
    assert noise.p_value > 0.01
    assert ev.morans_i(clustered, coords, permutations=199) == res


def test_save_fig_writes_png_and_svg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "FIGURES", tmp_path / "fig")
    ev.apply_plot_style()
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    png, svg = ev.save_fig(fig, "demo")
    plt.close(fig)
    assert png.stat().st_size > 0
    assert "<svg" in svg.read_text()


def test_chf_formatter_swiss_separator() -> None:
    assert ev.chf_formatter()(1850.4, None) == "1'850"
    assert ev.chf_formatter(1)(1234567.89, None) == "1'234'567.9"
    assert ev.chf(1850.4) == "CHF 1'850"


def test_log_run_and_ablation_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "MLRUNS", tmp_path)
    monkeypatch.setattr(ev, "TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    table = pd.DataFrame({"x": [1, 2]})
    for name, stage, cv, test in [
        ("tab_a", "tabular", 300.0, 290.0),
        ("tab_b", "tabular", 280.0, 295.0),
        ("geo", "tabular+geo", 250.0, 240.0),
    ]:
        ev.log_run(
            name, stage, params={"k": 8, "min n ≥ 20": 1},
            metrics={"cv_mae": cv, "test_mae": test, "R² test": 0.5},
            tables={"demo": table}, tags={"session": "t"}, experiment="test",
        )  # fmt: skip
    runs = ev.load_runs("test")
    assert len(runs) == 3
    assert (runs["tags.session"] == "t").all()
    assert "metrics.R__test" in runs and "params.min_n___20" in runs
    abl = ev.ablation_table(runs, ["tabular", "text", "tabular+geo"], ["test_mae"])
    assert list(abl.index) == ["tabular", "tabular+geo"]
    assert abl.loc["tabular", "run_name"] == "tab_b"  # lowest CV MAE, not lowest test MAE
    assert abl.loc["tabular", "test_mae"] == 295.0
