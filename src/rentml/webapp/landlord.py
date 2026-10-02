"""Landlord entry point (``/vermieter``): price band and what-if before publishing a listing.

The recommended listing band is the uncalibrated 25th-75th percentile of the quantile models,
the 80 % interval the calibrated one. The what-if view varies one input at a time; the models are
monotone in the living area, so a larger flat never gets a lower estimate.
"""

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from rentml.estimate import Apartment, EstimateError
from rentml.plotting import COLORS
from rentml.webapp.components import (
    address_field,
    attributes_block,
    chf,
    decode_address,
    drivers_block,
    field,
    interval_figure,
    location_note,
    model_missing_notice,
    register_address,
)
from rentml.webapp.figures import fmt_number
from rentml.webapp.services import Services
from rentml.webapp.tenant import error_card

FEATURE_LABELS: dict[str, str] = {"area": "Wohnfläche", "rooms": "Zimmer"}
NOT_IN_MODEL = "Renovation, Balkon und Lift"


def layout(services: Services) -> html.Div:
    """Form on the left, result on the right."""
    if services.estimator is None:
        return html.Div(model_missing_notice(), className="page")
    form = html.Div(
        [
            html.Div("Für Vermietende", className="eyebrow"),
            html.H2("Preisband & What-if", className="panel-title"),
            html.P(
                "Welche Nettomiete ist für die Wohnung marktüblich, bevor Sie sie ausschreiben?",
                className="lead",
            ),
            address_field("landlord-address"),
            html.Div(
                [
                    field(
                        "Wohnfläche (m²)",
                        dcc.Input(
                            id="landlord-area", type="number", min=10, max=500, step=1, value=75
                        ),
                    ),
                    field(
                        "Zimmer",
                        dcc.Input(
                            id="landlord-rooms", type="number", min=1, max=15, step=0.5, value=3.5
                        ),
                    ),
                ],
                className="field-row",
            ),
            field(
                "Beschrieb (optional)",
                dcc.Textarea(
                    id="landlord-text",
                    className="textarea",
                    placeholder="z. B. «2021 renoviert, Balkon, Lift, Seesicht»",
                ),
                "Erkannte Merkmale werden angezeigt, nicht gespeichert.",
            ),
            html.Button(
                "Preisband berechnen", id="landlord-submit", className="primary-button", n_clicks=0
            ),
        ],
        className="panel form-card",
    )
    result = dcc.Loading(
        html.Div(id="landlord-result", children=_placeholder()), type="dot", color="#0077c8"
    )
    return html.Div(
        [html.Div([form, html.Div(result, className="result-col")], className="tool-layout")],
        className="page",
    )


def _placeholder() -> html.Div:
    return html.Div(
        [
            html.H3("Was Sie erhalten", className="section-title"),
            html.Ul(
                [
                    html.Li("Geschätzte Miete mit kalibriertem 80 %-Intervall"),
                    html.Li("Empfohlenes Angebotsband (25.–75. Perzentil)"),
                    html.Li("What-if: Wirkung von mehr oder weniger Fläche und Zimmern"),
                    html.Li("Die wichtigsten Einflussfaktoren der Lage und des Gebäudes"),
                ]
            ),
        ],
        className="panel result-card muted",
    )


def register(app: Dash, services: Services) -> None:
    """Register the page and its callbacks."""
    dash.register_page(
        "landlord",
        path="/vermieter",
        name="Vermieter",
        title="RentLens · Preisband",
        layout=layout(services),
        order=2,
    )
    if services.estimator is None:
        return
    register_address(app, "landlord-address", services.search)

    @app.callback(
        Output("landlord-result", "children"),
        Input("landlord-submit", "n_clicks"),
        State("landlord-address", "value"),
        State("landlord-area", "value"),
        State("landlord-rooms", "value"),
        State("landlord-text", "value"),
        prevent_initial_call=True,
    )
    def band(
        _n: int, address: str | None, area: float | None, rooms: float | None, text: str | None
    ) -> html.Div:
        return run_band(services, address, area, rooms, text)


def run_band(
    services: Services,
    address: str | None,
    area: float | None,
    rooms: float | None,
    text: str | None,
) -> html.Div:
    """Validate the form, estimate and render price band and what-if."""
    if not address or area is None or rooms is None:
        return error_card("Bitte Adresse, Wohnfläche und Zimmer angeben.")
    apartment = Apartment(decode_address(address), float(area), float(rooms), text)
    try:
        location = services.lookup(apartment.address)
        estimate = services.estimator.estimate(apartment, location)
        table = services.estimator.what_if(apartment, location)
    except EstimateError as exc:
        return error_card(str(exc))
    p = estimate.prediction
    blocks = [
        html.Div(
            [
                html.Div("Empfohlenes Angebotsband", className="eyebrow"),
                html.H2(f"{chf(p['q25'])} – {chf(p['q75'])}", className="verdict"),
                html.P(
                    f"Geschätzte Nettomiete {chf(p['expected'])}; mit "
                    f"{estimate.coverage:.0%} Wahrscheinlichkeit zwischen {chf(p['lo'])} "
                    f"und {chf(p['hi'])}.",
                    className="lead",
                ),
            ]
        ),
        dcc.Graph(figure=interval_figure(estimate), config={"displayModeBar": False}),
        location_note(estimate),
        html.Div(
            [
                html.H3("What-if", className="section-title"),
                dcc.Graph(figure=what_if_figure(table), config={"displayModeBar": False}),
                html.Div(
                    "Jeweils eine Angabe geändert, alles andere gleich. Weniger Zimmer bei "
                    "gleicher Fläche heisst grössere Zimmer; das kann die Schätzung erhöhen.",
                    className="stat-note",
                ),
                html.Div(
                    f"{NOT_IN_MODEL} sind im Modell v1 noch nicht enthalten (Textmerkmale "
                    "folgen mit RQ1); ihre Wirkung wird daher nicht geschätzt.",
                    className="stat-note",
                ),
            ],
            className="result-block",
        ),
        drivers_block(estimate.drivers),
        attributes_block(estimate.attributes),
    ]
    return html.Div([b for b in blocks if b is not None], className="panel result-card")


def what_if_labels(table: pd.DataFrame) -> list[str]:
    """Readable labels such as ``"Wohnfläche 95 m² (+10)"``."""
    labels = []
    for row in table.itertuples():
        if row.feature == "baseline":
            labels.append("Ihre Angaben")
            continue
        unit = " m²" if row.feature == "area" else ""
        decimals = 0 if row.feature == "area" else 1
        step = row.value - row.original
        labels.append(
            f"{FEATURE_LABELS.get(row.feature, row.feature)} "
            f"{fmt_number(row.value, decimals)}{unit} ({step:+g})"
        )
    return labels


def what_if_figure(table: pd.DataFrame) -> go.Figure:
    """Horizontal bars: change of the estimate per variant in CHF."""
    variants = table.iloc[1:]
    labels = what_if_labels(table)[1:]
    colors = [COLORS["primary"] if d >= 0 else COLORS["reference"] for d in variants["delta_chf"]]
    text = [
        f"{d:+,.0f} CHF → {e:,.0f}".replace(",", "'")
        for d, e in zip(variants["delta_chf"], variants["expected_chf"], strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=variants["delta_chf"],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="auto",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=60 + 38 * len(variants),
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "zeroline": True,
            "zerolinecolor": "#8595aa",
            "ticksuffix": " CHF",
            "tickformat": "+,.0f",
            "showgrid": False,
        },
        yaxis={"autorange": "reversed"},
        separators=".'",
        font={"size": 12},
    )
    return fig
