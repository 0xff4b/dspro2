"""Start page: interactive map of Switzerland by canton, district or municipality."""

from collections.abc import Callable

import dash
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

from rentml.mapbundle import MapBundle
from rentml.webapp.figures import Metric, available_metrics, map_figure
from rentml.webapp.panel import SWITZERLAND_KEY, panel_children
from rentml.webapp.state import (
    DEFAULT_VIEW,
    LEVEL_PLURALS,
    UnitIndex,
    drill_down,
    reset_view,
    select_on_map,
    select_unit,
    set_level,
)

DROPDOWN_LABELS: dict[str, str] = {
    "search": "Suchen",
    "clear_search": "Suche löschen",
    "clear_selection": "Auswahl aufheben",
    "no_options_found": "Keine Treffer",
}
GRAPH_CONFIG: dict[str, object] = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "toImage"],
}


def layout(index: UnitIndex, metrics: list[Metric], meta: dict) -> html.Div:
    """Page layout: toolbar, map, detail panel and data sources.

    Args:
        index: Unit lookup (search options).
        metrics: Metrics offered in the dropdown (the first is the default).
        meta: Bundle metadata for the source note.

    Returns:
        The page component.
    """
    min_count = int(meta.get("min_count", 20))
    toolbar = html.Div(
        [
            html.Div(
                [
                    html.Label("Ebene", className="control-label"),
                    dcc.RadioItems(
                        id="level",
                        options=[{"label": v, "value": k} for k, v in LEVEL_PLURALS.items()],
                        value=DEFAULT_VIEW["level"],
                        className="segmented",
                        inline=True,
                    ),
                ],
                className="control",
            ),
            html.Div(
                [
                    html.Label("Kennzahl", className="control-label"),
                    dcc.Dropdown(
                        id="metric",
                        options=[{"label": m.label, "value": m.column} for m in metrics],
                        value=metrics[0].column,
                        clearable=False,
                        searchable=False,
                        labels=DROPDOWN_LABELS,
                    ),
                ],
                className="control control-metric",
            ),
            html.Div(
                [
                    html.Label("Suche", className="control-label"),
                    dcc.Dropdown(
                        id="search",
                        options=index.search_options(),
                        placeholder="Gemeinde, Bezirk oder Kanton …",
                        clearable=True,
                        labels=DROPDOWN_LABELS,
                    ),
                ],
                className="control control-search",
            ),
            html.Button("Ganze Schweiz", id="reset", className="ghost-button", n_clicks=0),
        ],
        className="toolbar",
    )
    legend = html.Div(
        [
            html.Span(className="swatch swatch-missing"),
            html.Span(f"Grau: weniger als {min_count} Inserate, Median ausgeblendet"),
            html.Span(className="swatch swatch-lake"),
            html.Span("Seen ausserhalb der Gemeindegebiete"),
            html.Span("LV95 (EPSG:2056), massstabsgetreu", className="legend-right"),
        ],
        className="map-legend",
    )
    return html.Div(
        [
            dcc.Store(id="view", data=dict(DEFAULT_VIEW)),
            toolbar,
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Graph(id="map", config=GRAPH_CONFIG, className="map-graph"),
                            legend,
                        ],
                        className="map-card",
                    ),
                    html.Aside(id="panel", className="panel"),
                ],
                className="map-layout",
            ),
            html.P(
                [
                    f"Grenzen: {meta.get('boundaries', 'swissBOUNDARIES3D')}. ",
                    f"Mieten: {meta.get('listings', 'Inserate')}, Angebotsmieten (netto), "
                    "eine Zeile pro Objekt nach Dublettenprüfung. ",
                    "Die Karte zeigt nur Aggregate; Medianwerte erst ab "
                    f"{min_count} Inseraten pro Einheit.",
                ],
                className="sources",
            ),
        ],
        className="page page-map",
    )


def register(app: Dash, bundle: MapBundle, geojson_url: Callable[[str], str]) -> None:
    """Register the page (path ``/``) and its callbacks on ``app``.

    Args:
        app: Dash app created with ``use_pages=True``.
        bundle: Loaded map bundle.
        geojson_url: Level -> URL of its GeoJSON.
    """
    index = UnitIndex(bundle.units)
    metrics = available_metrics(bundle.units, bundle.meta)
    by_column = {m.column: m for m in metrics}
    min_count = int(bundle.meta.get("min_count", 20))
    dash.register_page(
        "home",
        path="/",
        name="Karte",
        title="RentLens · Karte",
        order=0,
        layout=layout(index, metrics, bundle.meta),
    )

    @app.callback(
        Output("view", "data"),
        Output("level", "value"),
        Output("search", "value"),
        Input("level", "value"),
        Input("map", "clickData"),
        Input("search", "value"),
        Input("reset", "n_clicks"),
        Input({"type": "drill", "key": ALL}, "n_clicks"),
        Input({"type": "unit-link", "key": ALL}, "n_clicks"),
        State("view", "data"),
        prevent_initial_call=True,
    )
    def update_view(
        level: str,
        click: dict | None,
        search: str | None,
        _reset: int,
        _drill: list[int | None],
        _links: list[int | None],
        view: dict,
    ) -> tuple[dict, str, str | None]:
        clicked = bool(ctx.triggered and ctx.triggered[0].get("value"))
        trigger = ctx.triggered_id
        new = next_view(index, view or DEFAULT_VIEW, trigger, clicked, level, click, search)
        if new is None:
            return no_update, no_update, no_update
        return new, new["level"], new["selected"]

    @app.callback(Output("map", "figure"), Input("view", "data"), Input("metric", "value"))
    def render_map(view: dict, metric: str) -> go.Figure:
        return map_figure(
            index,
            view or DEFAULT_VIEW,
            by_column.get(metric, metrics[0]),
            geojson_url=geojson_url,
            other_areas=bundle.other_areas,
            min_count=min_count,
        )

    @app.callback(Output("panel", "children"), Input("view", "data"), Input("metric", "value"))
    def render_panel(view: dict, metric: str) -> list:
        return panel_children(index, bundle.meta, view or DEFAULT_VIEW, by_column[metric])


def next_view(
    index: UnitIndex,
    view: dict,
    trigger: str | dict | None,
    clicked: bool,
    level: str,
    click: dict | None,
    search: str | None,
) -> dict | None:
    """Translate the triggering input into the next view state (``None`` = no change).

    Args:
        index: Unit lookup.
        view: Current view state.
        trigger: ``ctx.triggered_id`` (component id or pattern-matching id dict).
        clicked: Whether the trigger value is truthy. Pattern-matching buttons are re-created
            with every panel update and then fire with ``n_clicks`` 0 or ``None``; ignored.
        level: Value of the level control.
        click: ``clickData`` of the map.
        search: Value of the search dropdown.

    Returns:
        The next view, or ``None`` if nothing changes.
    """
    if trigger == "level":
        return set_level(view, level)
    if trigger == "map":
        points = (click or {}).get("points") or [{}]
        customdata = points[0].get("customdata")
        return select_on_map(view, index, customdata[0]) if customdata else None
    if trigger == "search":
        if not search:
            return {**view, "selected": None} if view.get("selected") else None
        return None if search == view.get("selected") else select_unit(view, index, search)
    if trigger == "reset":
        return reset_view(view)
    if not clicked or not isinstance(trigger, dict):
        return None
    if trigger.get("type") == "drill":
        return drill_down(view, index)
    if trigger.get("type") == "unit-link":
        key = trigger.get("key")
        return reset_view(view) if key == SWITZERLAND_KEY else select_unit(view, index, key)
    return None
