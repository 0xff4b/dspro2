"""Tests for rentml.plotting."""

from collections.abc import Iterator
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from rentml.plotting import (
    COLORS,
    PALETTE,
    ablation_table,
    apply_plot_style,
    chf_formatter,
    save_fig,
    to_latex,
    to_markdown,
)


@pytest.fixture(autouse=True)
def _restore_rcparams() -> Iterator[None]:
    with mpl.rc_context():
        yield


def _simple_figure() -> Figure:
    fig = Figure(figsize=(3, 2))
    ax = fig.add_subplot()
    ax.plot([0, 1, 2], [1, 3, 2], label="rent")
    ax.set_title("Coverage by band")
    return fig


def test_palette_matches_dspro1_colors() -> None:
    assert PALETTE[0] == COLORS["primary"] == "#0077C8"
    assert len(PALETTE) == 8
    assert all(color.startswith("#") and len(color) == 7 for color in PALETTE)


def test_apply_plot_style_sets_rcparams() -> None:
    apply_plot_style()
    assert mpl.rcParams["svg.fonttype"] == "none"
    assert mpl.rcParams["axes.titlelocation"] == "left"
    cycle_colors = [c.upper() for c in mpl.rcParams["axes.prop_cycle"].by_key()["color"]]
    assert cycle_colors == PALETTE


def test_apply_plot_style_without_seaborn() -> None:
    apply_plot_style(use_seaborn=False)
    assert mpl.rcParams["axes.spines.top"] is False


def test_save_fig_writes_png_and_editable_svg(tmp_path: Path) -> None:
    fig_dir = tmp_path / "nested" / "fig"
    png, svg = save_fig(_simple_figure(), "coverage_by_band", fig_dir, dpi=50)
    assert png == fig_dir / "coverage_by_band.png"
    assert svg == fig_dir / "coverage_by_band.svg"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    svg_text = svg.read_text(encoding="utf-8")
    assert "Coverage by band" in svg_text  # text kept as text, not as glyph paths
    assert "<dc:date>" not in svg_text


def test_save_fig_is_deterministic_and_strips_suffix(tmp_path: Path) -> None:
    fig = _simple_figure()
    _, svg_a = save_fig(fig, "map.svg", tmp_path / "a", dpi=50)
    _, svg_b = save_fig(fig, "map", tmp_path / "b", dpi=50)
    assert svg_a.name == "map.svg"
    assert svg_a.read_bytes() == svg_b.read_bytes()


def test_save_fig_keeps_inner_dots(tmp_path: Path) -> None:
    png, _ = save_fig(_simple_figure(), "fig_v2.1", tmp_path, dpi=50)
    assert png.name == "fig_v2.1.png"


@pytest.mark.parametrize("bad_name", ["", "   ", "sub/fig", "..\\fig", ".png"])
def test_save_fig_rejects_bad_names(tmp_path: Path, bad_name: str) -> None:
    with pytest.raises(ValueError):
        save_fig(_simple_figure(), bad_name, tmp_path)


def test_save_fig_rejects_non_positive_dpi(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_fig(_simple_figure(), "fig", tmp_path, dpi=0)


def test_chf_formatter_uses_swiss_separator() -> None:
    assert chf_formatter()(1850.4, 0) == "1'850"
    assert chf_formatter(1)(12345.67, 0) == "12'345.7"


def test_chf_formatter_rejects_negative_decimals() -> None:
    with pytest.raises(ValueError):
        chf_formatter(-1)


def _calibration_table() -> pd.DataFrame:
    return pd.DataFrame(
        {"coverage": [0.048, 0.101, 0.951], "p": [0.0003, 0.2, 1.0], "MAE": [250.46, 1234.0, None]},
        index=pd.Index([0.05, 0.10, 0.95], name="level"),
    )


def test_to_markdown_keeps_float_index_labels_and_small_values() -> None:
    lines = to_markdown(_calibration_table()).splitlines()
    assert lines[0] == "| level | coverage | p | MAE |"
    assert lines[2] == "| 0.05 | 0.0480 | 0.000300 | 250.5 |"
    assert lines[3] == "| 0.1 | 0.101 | 0.200 | 1234.0 |"
    assert lines[4] == "| 0.95 | 0.951 | 1.0 | – |"


def test_table_formats_index_by_name_and_explicit_string() -> None:
    table = _calibration_table()
    tex = to_latex(table, caption="Calibration", label="tab:q", floatfmt={"level": ".2f"})
    assert "0.10 & 0.101 & 0.200 & 1234.0 \\\\" in tex
    fixed = to_markdown(table, ".1f").splitlines()  # explicit spec: all value cells .1f
    assert fixed[2] == "| 0.05 | 0.0 | 0.0 | 250.5 |"
    effects = pd.DataFrame({"power": np.linspace(0.1, 1.0, 10)}, index=np.arange(1, 11) / 100)
    labels = [line.split(" | ")[0] for line in to_markdown(effects).splitlines()[2:]]
    assert len(set(labels)) == 10 and labels[0] == "| 0.01"


def _runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_name": ["a1", "a2", "b1"],
            "stage": ["tabular", "tabular", "+geo"],
            "status": ["FINISHED"] * 3,
            "start_time": pd.to_datetime([1, 2, 3], unit="s", utc=True),
            "metrics.MAE": [300.0, 305.0, 280.0],  # test metric
            "metrics.cv_MAE": [320.0, 310.0, np.nan],  # validation metric
        }
    )


def test_ablation_table_selects_on_validation_metric(caplog: pytest.LogCaptureFixture) -> None:
    table = ablation_table(
        _runs(), stage_order=["tabular", "+geo"], metrics=["MAE"], pick="best", select_by="cv_MAE"
    )
    assert table["run_name"].tolist() == ["a2", "b1"]  # b1: all-NaN selector -> last run
    assert table.loc["tabular", "MAE"] == 305.0 and list(table.columns) == ["run_name", "MAE"]
    assert "test-set selection" not in caplog.text
    naive = ablation_table(_runs(), stage_order=["tabular"], metrics=["MAE"], pick="best")
    assert naive.loc["tabular", "run_name"] == "a1"
    assert "test-set selection" in caplog.text


def test_ablation_table_rejects_bad_selector() -> None:
    with pytest.raises(ValueError, match="select_by"):
        ablation_table(_runs(), stage_order=["tabular"], metrics=["MAE"], select_by="cv_MAE")
    with pytest.raises(KeyError, match="calib_MAE"):
        ablation_table(
            _runs(), stage_order=["tabular"], metrics=["MAE"], pick="best", select_by="calib_MAE"
        )
