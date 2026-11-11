"""Plotly figures of the RentLens app: interval bar, what-if curve and rent map."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from rentml import intervals
from rentml.config import AREA_RANGE, MIN_MAP_CELL
from rentml.evaluation import COLORS, chf
from rentml.intervals import RentModelBundle
from rentml.maps import LAKE_COLOR, LAKE_EDGE

FIG_LAYOUT = {"margin": {"l": 8, "r": 8, "t": 8, "b": 8}, "paper_bgcolor": "rgba(0,0,0,0)",
              "plot_bgcolor": "rgba(0,0,0,0)", "separators": ".'"}  # fmt: skip
CHF_AXIS = {"tickformat": ",.0f", "ticksuffix": " CHF"}


def interval_figure(pred: pd.Series, coverage: float, asking: float | None = None) -> go.Figure:
    """Horizontal bar: calibrated interval, listing band (q25-q75), estimate and asking rent."""
    fig = go.Figure()
    bands = [("lo", "hi", "soft", f"{coverage:.0%}-Intervall"), ("q25", "q75", "mid", "q25–q75")]
    for lo, hi, color, name in bands:
        fig.add_vrect(pred[lo], pred[hi], fillcolor=COLORS[color], line_width=0, layer="below",
                      y0=0.3, y1=0.7, name=name, showlegend=True)  # fmt: skip
    marks = [(pred["expected"], COLORS["dark"], "Schätzung", "diamond")]
    if asking is not None:
        marks.append((asking, COLORS["reference"], "Ihre Miete", "line-ns-open"))
    for x, color, name, symbol in marks:
        line = {"width": 3, "color": color}
        marker = {"color": color, "size": 18, "symbol": symbol, "line": line}
        fig.add_scatter(x=[x], y=[0], mode="markers", name=name, marker=marker)
    xs = [pred["lo"], pred["hi"], *([asking] if asking is not None else [])]
    pad = 0.12 * (max(xs) - min(xs))
    xaxis = {"range": [min(xs) - pad, max(xs) + pad], "showgrid": False, **CHF_AXIS}
    yaxis = {"visible": False, "range": [-1, 1]}
    legend = {"orientation": "h", "y": -0.45}
    return fig.update_layout(FIG_LAYOUT, height=150, xaxis=xaxis, yaxis=yaxis, legend=legend)


def what_if_figure(bundle: RentModelBundle, row: pd.DataFrame, area: float) -> go.Figure:
    """Expected rent and calibrated interval over the living area, all else unchanged."""
    grid = np.unique(np.clip(np.round(np.linspace(0.6 * area, 1.4 * area, 17)), *AREA_RANGE))
    table = intervals.what_if(bundle, row, grid)
    curve, base = table.query("feature == 'area'"), table.iloc[0]
    x, coverage = curve["value"], 1.0 - bundle.calibrator.alpha
    fig = go.Figure()
    fig.add_scatter(x=x, y=curve["hi"], line_width=0, showlegend=False, hoverinfo="skip")
    fig.add_scatter(
        x=x, y=curve["lo"], line_width=0, fill="tonexty", name=f"{coverage:.0%}-Intervall"
    )
    fig.add_scatter(x=x, y=curve["expected"], name="Schätzung", line_color=COLORS["primary"])
    fig.add_scatter(x=[area], y=[base["expected"]], mode="markers", name="Ihre Angaben")
    fig.update_traces(fillcolor=COLORS["soft"], selector={"fill": "tonexty"})
    fig.update_traces(
        marker={"color": COLORS["reference"], "size": 11}, selector={"mode": "markers"}
    )
    legend = {"orientation": "h", "y": -0.25}
    xaxis = {"title": {"text": "Wohnfläche (m²)"}}
    return fig.update_layout(FIG_LAYOUT, height=280, xaxis=xaxis, yaxis=CHF_AXIS, legend=legend)


def map_figure(
    layer: gpd.GeoDataFrame, min_count: int = MIN_MAP_CELL, lakes: gpd.GeoDataFrame | None = None
) -> go.Figure:
    """Choropleth of the median CHF/m²; units below ``min_count`` objects are grey.

    ``lakes`` (from :func:`rentml.maps.load_lakes_layer`) are drawn on top, because most lakes
    belong to the area of the surrounding municipalities.
    """
    has = layer["chf_per_m2"].notna().to_numpy()
    hover = [
        f"<b>{r.name}</b><br>{r.chf_per_m2:.1f} CHF/m² · Median {chf(r.rent)} · {r.n_objects} "
        "Inserate" if ok else f"<b>{r.name}</b><br>weniger als {min_count} Inserate"
        for r, ok in zip(layer.itertuples(), has, strict=True)
    ]  # fmt: skip
    common = {"geojson": json.loads(layer[["unit_id", "geometry"]].to_json()),
              "featureidkey": "properties.unit_id", "hoverinfo": "text",
              "marker": {"line": {"width": 0.4, "color": "white"}}}  # fmt: skip
    grey = [[0, COLORS["no_data"]], [1, COLORS["no_data"]]]
    fig = go.Figure()
    fig.add_choroplethmap(locations=layer["unit_id"], z=np.zeros(len(layer)), text=hover,
                          colorscale=grey, showscale=False, **common)  # fmt: skip
    fig.add_choroplethmap(locations=layer.loc[has, "unit_id"], z=layer.loc[has, "chf_per_m2"],
                          text=np.asarray(hover, dtype=object)[has], colorscale="Blues",
                          colorbar={"title": {"text": "CHF/m²"}}, **common)  # fmt: skip
    if lakes is not None and len(lakes):
        water = [[0, LAKE_COLOR], [1, LAKE_COLOR]]
        fig.add_choroplethmap(
            geojson=json.loads(lakes[["unit_id", "geometry"]].to_json()),
            featureidkey="properties.unit_id", locations=lakes["unit_id"],
            z=np.zeros(len(lakes)), text=lakes["name"], hoverinfo="text", colorscale=water,
            showscale=False, marker={"line": {"width": 0.8, "color": LAKE_EDGE}},
        )  # fmt: skip
    view = {"style": "carto-positron", "center": {"lat": 46.8, "lon": 8.23}, "zoom": 6.4}
    return fig.update_layout(map=view, margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=620)
