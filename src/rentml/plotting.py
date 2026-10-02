"""Report output: plot style, figure export and report tables (ablation, Markdown, LaTeX).

The palette and ``rcParams`` are copied from the DSPRO1 module ``src/plot_style.py`` so that
DSPRO1 and DSPRO2 figures look identical in the report and on the poster.
"""

import logging
import math
import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

logger = logging.getLogger(__name__)

COLORS: dict[str, str] = {
    "primary": "#0077C8",
    "secondary": "#66B5E9",
    "reference": "#D0384B",
    "dark": "#153652",
    "teal": "#16856F",
    "orange": "#C88126",
    "purple": "#6861B4",
    "gray": "#718196",
    "neutral": "#192B3A",
}

PALETTE: list[str] = [
    COLORS[key]
    for key in ("primary", "secondary", "teal", "orange", "purple", "gray", "dark", "neutral")
]

_RC_PARAMS: dict[str, object] = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.figsize": (8, 4.8),
    "figure.dpi": 110,
    "figure.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#C8DCEC",
    "axes.labelcolor": "#4C6280",
    "axes.titlecolor": "#111A28",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 16,
    "axes.labelsize": 10,
    "axes.labelpad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "grid.color": "#D6E7F5",
    "grid.alpha": 0.65,
    "grid.linewidth": 0.6,
    "xtick.color": "#4C6280",
    "ytick.color": "#4C6280",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "text.color": "#111A28",
    "lines.linewidth": 2,
    "patch.edgecolor": "#FFFFFF",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}

# Export settings for reproducible, editable SVGs: text stays text (poster editing) and the
# element ids are derived from a fixed salt, so re-running a notebook does not create git noise.
_SVG_RC: dict[str, object] = {"svg.fonttype": "none", "svg.hashsalt": "rentml"}
_KNOWN_SUFFIXES = (".png", ".svg", ".pdf")
_LATEX_WORDS = {"\\": r"\textbackslash{}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def apply_plot_style(*, use_seaborn: bool = True) -> None:
    """Apply the DSPRO1/HSLU presentation style globally.

    Only presentation settings are changed; no data, estimator or random state is touched.

    Args:
        use_seaborn: Also call ``seaborn.set_theme`` (whitegrid, project palette) first, as the
            DSPRO1 notebook did. The project ``rcParams`` are applied afterwards and win.
    """
    if use_seaborn:
        sns.set_theme(context="notebook", style="whitegrid", palette=PALETTE)
    mpl.rcParams.update(_RC_PARAMS)
    logger.debug("Applied rentml plot style (seaborn=%s)", use_seaborn)


def save_fig(fig: Figure, name: str, fig_dir: Path, dpi: int = 200) -> tuple[Path, Path]:
    """Save a figure as PNG (report) and SVG with editable text (poster).

    Args:
        fig: The Matplotlib figure to export.
        name: File stem, e.g. ``"coverage_by_band"``. A trailing ``.png``/``.svg``/``.pdf``
            suffix is stripped; other dots are kept.
        fig_dir: Output directory; created if missing.
        dpi: Resolution of the PNG export.

    Returns:
        Paths of the written PNG and SVG files.

    Raises:
        ValueError: If ``name`` is empty or contains a path separator, or ``dpi`` is not positive.
    """
    stem = _figure_stem(name)
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    out_dir = Path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    svg_path = out_dir / f"{stem}.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", format="png")
    with mpl.rc_context(_SVG_RC):
        fig.savefig(svg_path, bbox_inches="tight", format="svg", metadata={"Date": None})
    logger.info("Saved figure %s (+ .svg)", png_path)
    return png_path, svg_path


def chf_formatter(decimals: int = 0) -> FuncFormatter:
    """Axis formatter for CHF amounts with the Swiss thousands separator (``1'850``).

    Args:
        decimals: Number of decimals to show.

    Returns:
        A formatter for ``ax.xaxis.set_major_formatter`` / ``ax.yaxis.set_major_formatter``.

    Raises:
        ValueError: If ``decimals`` is negative.
    """
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0, got {decimals}")

    def _format(value: float, _pos: int | None = None) -> str:
        return f"{value:,.{decimals}f}".replace(",", "'")

    return FuncFormatter(_format)


def ablation_table(
    runs: pd.DataFrame,
    *,
    stage_order: list[str],
    metrics: list[str],
    pick: str = "last",
    select_by: str | None = None,
    lower_is_better: bool = True,
    only_finished: bool = True,
) -> pd.DataFrame:
    """Build the stage x metric ablation table from :meth:`rentml.tracking.Tracker.runs_frame`.

    Choosing the best of several runs on the reported *test* metric is test-set model
    selection and flatters the ablation. With ``pick="best"`` pass ``select_by`` = a
    validation metric (e.g. ``"cv_MAE"`` or ``"calib_MAE"``); it is only used for choosing and
    need not be displayed. Without it ``metrics[0]`` is used and a warning is logged.

    Args:
        runs: Output of :meth:`rentml.tracking.Tracker.runs_frame`.
        stage_order: Row order; stages without a usable run are skipped with a warning.
        metrics: Metric names to display, with or without the ``metrics.`` prefix.
        pick: ``"last"`` (most recent run per stage) or ``"best"`` (by ``select_by``).
        select_by: Selection metric for ``pick="best"``; never a test metric.
        lower_is_better: Direction of the selection metric.
        only_finished: Ignore runs whose status is not ``FINISHED``.

    Returns:
        Table indexed by ``stage`` with ``run_name`` and one float column per metric.

    Raises:
        ValueError: If ``metrics`` is empty, ``stage_order`` has duplicates, ``pick`` is
            invalid or ``select_by`` is given with ``pick="last"``.
        KeyError: If ``stage``, a requested metric or ``select_by`` is missing from ``runs``.
    """
    if not metrics:
        raise ValueError("metrics must not be empty")
    if len(set(stage_order)) != len(stage_order):
        raise ValueError("stage_order contains duplicates")
    if pick not in ("last", "best"):
        raise ValueError(f"pick must be 'last' or 'best', got {pick!r}")
    if select_by is not None and pick != "best":
        raise ValueError("select_by is only used with pick='best'")
    short = [name.removeprefix("metrics.") for name in metrics]
    columns = [f"metrics.{name}" for name in short]
    selector = f"metrics.{select_by.removeprefix('metrics.')}" if select_by else columns[0]
    if pick == "best" and select_by is None:
        logger.warning(
            "pick='best' ranks runs by the displayed %r; if that is a test metric this is "
            "test-set selection -- pass select_by=<validation metric>",
            short[0],
        )
    missing = [col for col in dict.fromkeys(["stage", *columns, selector]) if col not in runs]
    if missing:
        available = [c.removeprefix("metrics.") for c in runs.columns if c.startswith("metrics.")]
        raise KeyError(f"Columns {missing} not in runs; available metrics: {available}")
    frame = runs[runs["status"] == "FINISHED"] if only_finished and "status" in runs else runs
    rows = []
    for stage in stage_order:
        group = frame[frame["stage"] == stage]
        if group.empty:
            logger.warning("No usable run for stage %r; row skipped", stage)
            continue
        chosen = _pick_run(group, pick, selector, lower_is_better)
        rows.append(
            {
                "stage": stage,
                "run_name": chosen["run_name"],
                **dict(zip(short, chosen[columns], strict=True)),
            }
        )
    table = pd.DataFrame(rows, columns=["stage", "run_name", *short]).set_index("stage")
    table[short] = table[short].astype(float)
    return table


def to_markdown(
    df: pd.DataFrame,
    floatfmt: str | Mapping[str, str] | None = None,
    *,
    index: bool | None = None,
    na_rep: str = "–",
) -> str:
    """Render a DataFrame as a Markdown pipe table (no ``tabulate`` needed).

    Float formatting: ``None`` (default) is automatic -- ``.1f`` for ``|x| >= 1`` (CHF,
    percent) and 3 significant digits below 1, so p-values, coverage and R² keep their
    information. A string applies to every float cell; a mapping ``column -> spec`` applies
    per column (the index name may be a key), the others stay automatic. Float index labels
    use ``g`` unless mapped, so quantile levels 0.05/0.10 stay distinct.

    Args:
        df: Table to render.
        floatfmt: ``None``, one format spec, or a mapping column -> spec (see above).
        index: Include the index; default: only if it is not a plain ``RangeIndex``.
        na_rep: Text for missing values.

    Returns:
        The table, ending with a newline.
    """
    header, numeric, body = _table_cells(df, floatfmt, index, na_rep)
    lines = [_md_row(header), "|" + "|".join("---:" if num else ":---" for num in numeric) + "|"]
    lines.extend(_md_row(row) for row in body)
    return "\n".join(lines) + "\n"


def to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    *,
    floatfmt: str | Mapping[str, str] | None = None,
    index: bool | None = None,
    na_rep: str = "--",
    position: str = "htbp",
) -> str:
    """Render a DataFrame as a booktabs table (preamble needs ``\\usepackage{booktabs}``).

    Headers and cells are escaped. The caption stays LaTeX (``$R^2$`` works); only unescaped
    ``%``, ``&`` and ``#`` in it are escaped. Numbers are formatted as in :func:`to_markdown`.

    Args:
        df: Table to render.
        caption: Table caption.
        label: LaTeX label, e.g. ``"tab:ablation"``.
        floatfmt: ``None`` (automatic), one format spec, or a mapping column -> spec.
        index: Include the index; default: only if it is not a plain ``RangeIndex``.
        na_rep: Text for missing values (inserted as LaTeX).
        position: Float placement specifier.

    Returns:
        The ``table`` environment, ending with a newline.
    """
    header, numeric, body = _table_cells(df, floatfmt, index, na_rep)
    spec = "".join("r" if num else "l" for num in numeric)
    safe_caption = re.sub(r"(?<!\\)([%&#])", r"\\\1", caption)
    lines = [
        f"\\begin{{table}}[{position}]",
        "\\centering",
        f"\\caption{{{safe_caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(_latex_escape(cell) for cell in header) + r" \\",
        "\\midrule",
    ]
    for row in body:
        cells = [cell if cell == na_rep else _latex_escape(cell) for cell in row]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"


def _figure_stem(name: str) -> str:
    """Validate a figure name and strip a known image suffix."""
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Figure name must not be empty")
    if "/" in cleaned or "\\" in cleaned:
        raise ValueError(f"Figure name must not contain a path separator: {name!r}")
    for suffix in _KNOWN_SUFFIXES:
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    if not cleaned:
        raise ValueError(f"Figure name has no stem: {name!r}")
    return cleaned


def _table_cells(
    df: pd.DataFrame, floatfmt: str | Mapping[str, str] | None, index: bool | None, na_rep: str
) -> tuple[list[str], list[bool], list[list[str]]]:
    default_index = isinstance(df.index, pd.RangeIndex) and df.index.start == 0
    show_index = index if index is not None else not default_index
    mapping = floatfmt if isinstance(floatfmt, Mapping) else {}
    header = [str(col) for col in df.columns]
    numeric = [_is_number_dtype(dtype) for dtype in df.dtypes]
    formats = [floatfmt if isinstance(floatfmt, str) else mapping.get(h) for h in header]
    body = [
        [_format_cell(value, fmt, na_rep) for value, fmt in zip(row, formats, strict=True)]
        for row in df.itertuples(index=False, name=None)
    ]
    if show_index:
        index_name = "" if df.index.name is None else str(df.index.name)
        index_fmt = mapping.get(index_name, "g")  # labels: keep 0.05 and 0.10 distinct
        header = [index_name, *header]
        numeric = [_is_number_dtype(df.index.dtype), *numeric]
        body = [
            [_format_cell(lab, index_fmt, na_rep), *row]
            for lab, row in zip(df.index, body, strict=True)
        ]
    return header, numeric, body


def _is_number_dtype(dtype: object) -> bool:
    return pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)


def _format_cell(value: object, fmt: str | None, na_rep: str) -> str:
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return na_rep
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        return format(float(value), fmt if fmt is not None else _auto_float_format(float(value)))
    return str(value)


def _auto_float_format(value: float) -> str:
    # One decimal suits CHF and percentages; below 1 keep 3 significant digits instead, so a
    # p-value of 0.0003 or a coverage of 0.812 is not printed as 0.0 / 0.8.
    small = math.isfinite(value) and 0 < abs(value) < 1
    return "#.3g" if small else ".1f"


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |"


def _latex_escape(text: str) -> str:
    return "".join(_LATEX_WORDS.get(ch, f"\\{ch}" if ch in "&%$#_{}" else ch) for ch in text)


def _pick_run(group: pd.DataFrame, pick: str, metric: str, lower_is_better: bool) -> pd.Series:
    ordered = group.sort_values("start_time", kind="stable") if "start_time" in group else group
    values = ordered[metric].to_numpy(dtype=float)
    if pick == "best" and not np.isnan(values).all():
        best = np.nanargmin(values) if lower_is_better else np.nanargmax(values)
        return ordered.iloc[int(best)]
    return ordered.iloc[-1]
