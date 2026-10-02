"""Plotly map of Switzerland in LV95 and the metrics it can show.

The polygons are not embedded in the figure: every choropleth references the level's GeoJSON
by URL (served once, gzip-compressed and cached by plotly.js), so a figure update only carries
values. See :mod:`rentml.mapdata` for why the equirectangular projection of "plot degrees" is
an undistorted LV95 map.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rentml.mapdata import LEVELS, to_plot_xy
from rentml.plotting import COLORS
from rentml.webapp.state import LEVEL_LABELS, UnitIndex

COLORSCALE: list[list[object]] = [
    [0.0, "#DCEBF8"],
    [0.35, "#8EC3EC"],
    [0.7, COLORS["primary"]],
    [1.0, COLORS["dark"]],
]
MISSING_COLOR = "#E3E1DC"
LAKE_COLOR = "#C9DFF0"
OTHER_AREA_COLOR = "#EEECE7"
HIGHLIGHT_COLOR = COLORS["reference"]
BORDER_WIDTH = {"canton": 1.0, "district": 0.6, "municipality": 0.25}
VIEW_PADDING = 0.06
_TRANSPARENT = [[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(0,0,0,0)"]]
_LOG_TICKS = (1, 2, 5)


@dataclass(frozen=True)
class Metric:
    """A value that can colour the map.

    Attributes:
        column: Column of the units table.
        label: Name in the dropdown and the panel.
        unit: Unit for the colour bar and the hover label ("" = none).
        decimals: Decimals in labels.
        log: Colour on a log10 scale (skewed counts and densities).
        missing: Text for units without a value; ``{min_count}`` is filled in.
    """

    column: str
    label: str
    unit: str
    decimals: int
    log: bool = False
    missing: str = "weniger als {min_count} Inserate"

    def format(self, value: object) -> str:
        """Swiss number format with apostrophes, e.g. ``1'234.5 CHF/m²``."""
        if value is None or pd.isna(value):
            return "–"
        text = fmt_number(float(value), self.decimals)
        return f"{text} {self.unit}".strip()


METRICS: tuple[Metric, ...] = (
    Metric("chf_per_m2", "Median Nettomiete pro m²", "CHF/m²", 1),
    Metric("rent", "Median Nettomiete pro Monat", "CHF", 0),
    Metric("n_objects", "Anzahl Inserate", "", 0, log=True, missing="keine Inserate"),
    Metric("density", "Einwohner pro km²", "Einw./km²", 0, log=True, missing="keine Angabe"),
)


def fmt_number(value: float, decimals: int = 0) -> str:
    """Format with Swiss thousands separators (``1'234``)."""
    return f"{value:,.{decimals}f}".replace(",", "'")


def available_metrics(units: pd.DataFrame, meta: dict | None = None) -> list[Metric]:
    """Built-in metrics present in ``units`` plus extras declared in ``meta["extra_metrics"]``.

    An extra is declared as ``{"qoli": {"label": "Quality-of-Life-Index", "unit": "",
    "decimals": 0}}`` (e.g. once the QoLI or the value score are aggregated per unit).
    """
    metrics = [m for m in METRICS if m.column in units.columns]
    for column, spec in ((meta or {}).get("extra_metrics") or {}).items():
        if column in units.columns and column not in {m.column for m in metrics}:
            metrics.append(
                Metric(
                    column,
                    spec.get("label", column),
                    spec.get("unit", ""),
                    int(spec.get("decimals", 1)),
                )
            )
    return metrics


def metric_values(table: pd.DataFrame, metric: Metric) -> pd.Series:
    """Values to colour by; NaN where the unit is shown as missing."""
    values = pd.to_numeric(table[metric.column], errors="coerce").astype(float)
    return values.where(values > 0) if metric.log else values


def view_ranges(
    bounds: tuple[float, float, float, float], padding: float = VIEW_PADDING
) -> tuple[list[float], list[float]]:
    """Longitude and latitude ranges (plot degrees) that show an LV95 box with padding."""
    minx, miny, maxx, maxy = bounds
    pad = padding * max(maxx - minx, maxy - miny, 2_000.0)
    x, y = to_plot_xy(np.array([minx - pad, maxx + pad]), np.array([miny - pad, maxy + pad]))
    return [float(x[0]), float(x[1])], [float(y[0]), float(y[1])]


def geo_view(
    swiss_bounds: tuple[float, float, float, float], focus_bounds: tuple[float, float, float, float]
) -> dict[str, object]:
    """Geo axes, centre and zoom that show ``focus_bounds`` without clipping the frame.

    The axis ranges always span Switzerland, because plotly clips the map to them: ranges fitted
    to a small unit would cut the map to that unit's box. Zooming uses ``projection.scale``
    (1 = the axis ranges fit the frame) and ``center``; both only scale and translate the
    projected plane, so the map stays undistorted LV95.
    """
    lon_range, lat_range = view_ranges(swiss_bounds)
    lon_focus, lat_focus = view_ranges(focus_bounds)
    scale = min(
        (lon_range[1] - lon_range[0]) / (lon_focus[1] - lon_focus[0]),
        (lat_range[1] - lat_range[0]) / (lat_focus[1] - lat_focus[0]),
    )
    return {
        "lonaxis": {"range": lon_range},
        "lataxis": {"range": lat_range},
        "center": {"lon": sum(lon_focus) / 2, "lat": sum(lat_focus) / 2},
        "projection": {
            "type": "equirectangular",
            "rotation": {"lon": 0, "lat": 0, "roll": 0},
            "scale": max(1.0, scale),
        },
    }


def map_figure(
    index: UnitIndex,
    view: dict,
    metric: Metric,
    *,
    geojson_url: Callable[[str], str],
    other_areas: dict | None = None,
    min_count: int = 20,
) -> go.Figure:
    """Build the choropleth map for the current view.

    Traces (bottom to top): lakes and other non-municipal areas, units without a value (grey),
    units with a value (colour scale), canton borders (below the canton level) and the
    outline of the selected unit. Only the two unit traces react to hover and click; their
    ``customdata[0]`` is the unit key.

    Args:
        index: Unit lookup.
        view: View state (see :mod:`rentml.webapp.state`).
        metric: Metric to colour by.
        geojson_url: Level -> URL of its GeoJSON.
        other_areas: Inline GeoJSON of lakes and other non-municipal areas.
        min_count: Minimum objects per published median (for the missing label).

    Returns:
        The figure.
    """
    level = view["level"]
    if level not in LEVELS:
        raise ValueError(f"Unknown level {level!r}")
    table = index.level_table(level)
    values = metric_values(table, metric)
    fig = go.Figure()
    if other_areas and other_areas.get("features"):
        fig.add_trace(_other_areas_trace(other_areas))
    missing = values.isna()
    missing_text = metric.missing.format(min_count=min_count)
    fig.add_trace(
        _units_trace(table.loc[missing], None, metric, level, missing_text, geojson_url(level))
    )
    fig.add_trace(
        _units_trace(
            table.loc[~missing],
            values.loc[~missing],
            metric,
            level,
            missing_text,
            geojson_url(level),
        )
    )
    if level != "canton":
        cantons = index.level_table("canton")
        fig.add_trace(
            _outline_trace(
                cantons["unit_id"], geojson_url("canton"), COLORS["dark"], 1.1, "Kantonsgrenzen"
            )
        )
    selected = view.get("selected")
    if selected in index:
        row = index.row(selected)
        fig.add_trace(
            _outline_trace(
                [row["unit_id"]], geojson_url(row["level"]), HIGHLIGHT_COLOR, 3.0, "Auswahl"
            )
        )
    fig.update_layout(_layout(index, view))
    return fig


def _layout(index: UnitIndex, view: dict) -> dict:
    return {
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "separators": ".'",
        "dragmode": "pan",
        "showlegend": False,
        "uirevision": f"view-{view.get('rev', 0)}",
        "hoverlabel": {
            "bgcolor": "#FFFFFF",
            "bordercolor": COLORS["dark"],
            "font": {"color": "#111A28", "size": 13},
        },
        "geo": {
            "visible": False,
            "showframe": False,
            "framewidth": 0,
            "bgcolor": "rgba(0,0,0,0)",
            **geo_view(index.swiss_bounds, index.bounds(view.get("focus"))),
        },
    }


def _units_trace(
    table: pd.DataFrame,
    values: pd.Series | None,
    metric: Metric,
    level: str,
    missing_text: str,
    url: str,
) -> go.Choropleth:
    context = _context(level, table)
    shown = (
        [metric.format(v) for v in table[metric.column]]
        if values is not None
        else [missing_text] * len(table)
    )
    counts = [fmt_number(n) for n in table["n_objects"]]
    customdata = (
        np.column_stack([table.index.to_numpy(), table["name"].to_numpy(), context, shown, counts])
        if len(table)
        else None
    )
    hover = (
        "<b>%{customdata[1]}</b><br><span style='color:#4C6280'>%{customdata[2]}</span><br>"
        f"{metric.label}: <b>%{{customdata[3]}}</b><br>Inserate: %{{customdata[4]}}<extra></extra>"
    )
    common = {
        "geojson": url,
        "locations": table["unit_id"].astype(str).tolist(),
        "customdata": customdata,
        "hovertemplate": hover,
        "marker": {"line": {"color": "#FFFFFF", "width": BORDER_WIDTH[level]}},
        "name": LEVEL_LABELS[level],
    }
    if values is None:
        return go.Choropleth(
            z=[0] * len(table),
            colorscale=[[0, MISSING_COLOR], [1, MISSING_COLOR]],
            showscale=False,
            **common,
        )
    z, colorbar = _color_values(values, metric)
    return go.Choropleth(z=z, colorscale=COLORSCALE, colorbar=colorbar, **common)


def _context(level: str, table: pd.DataFrame) -> list[str]:
    if level == "municipality":
        text = "Gemeinde · Bezirk " + table["district_name"].astype(str) + " · " + table["canton"]
        return text.tolist()
    if level == "district":
        return ("Bezirk · Kanton " + table["canton"].astype(str)).tolist()
    return ["Kanton"] * len(table)


def _color_values(values: pd.Series, metric: Metric) -> tuple[list[float], dict]:
    colorbar: dict[str, object] = {
        "title": {"text": metric.unit or metric.label, "side": "top", "font": {"size": 12}},
        "orientation": "h",
        "x": 0.01,
        "xanchor": "left",
        "y": 0.02,
        "yanchor": "bottom",
        "len": 0.32,
        "thickness": 10,
        "outlinewidth": 0,
        "tickfont": {"size": 11},
        "bgcolor": "rgba(255,255,255,0.75)",
    }
    if not metric.log:
        colorbar["tickformat"] = ",.0f"
        return values.tolist(), colorbar
    z = np.log10(values.to_numpy(dtype=float))
    ticks = [t * 10**e for e in range(0, 7) for t in _LOG_TICKS]
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    shown = [t for t in ticks if lo - 1e-9 <= math.log10(t) <= hi + 1e-9] or [10**lo]
    if len(shown) > 6:  # keep powers of ten only
        shown = [t for t in shown if math.log10(t).is_integer()] or shown[:: len(shown) // 5]
    colorbar["tickvals"] = [math.log10(t) for t in shown]
    colorbar["ticktext"] = [fmt_number(t) for t in shown]
    return z.tolist(), colorbar


def _outline_trace(ids: object, url: str, color: str, width: float, name: str) -> go.Choropleth:
    locations = [str(int(i)) for i in ids]
    return go.Choropleth(
        geojson=url,
        locations=locations,
        z=[0] * len(locations),
        colorscale=_TRANSPARENT,
        showscale=False,
        hoverinfo="skip",
        marker={"line": {"color": color, "width": width}},
        name=name,
    )


def _other_areas_trace(collection: dict) -> go.Choropleth:
    features = [{**f, "id": str(i)} for i, f in enumerate(collection["features"])]
    kinds = [f["properties"].get("kind") for f in features]
    return go.Choropleth(
        geojson={"type": "FeatureCollection", "features": features},
        locations=[f["id"] for f in features],
        z=[1 if k == "lake" else 0 for k in kinds],
        zmin=0,
        zmax=1,
        colorscale=[[0, OTHER_AREA_COLOR], [1, LAKE_COLOR]],
        showscale=False,
        hoverinfo="skip",
        marker={"line": {"color": "#FFFFFF", "width": 0.3}},
        name="Seen und gemeindefreie Gebiete",
    )
