"""Tenant entry point (``/mietcheck``): Fair-Rent Check of an asking rent.

The verdict compares the asking rent with the calibrated 80 % interval of comparable asking
rents. "Fair" means in line with the market, not the legal notion of a non-abusive rent
(Art. 269 ff. OR); the page links to official information instead of giving legal advice.
Inputs are only used for the response and never stored.
"""

import dash
from dash import Dash, Input, Output, State, dcc, html

from rentml.estimate import Apartment, EstimateError
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
from rentml.webapp.services import Services

VERDICTS: dict[str, tuple[str, str, str]] = {
    "below": (
        "Unter der erwarteten Spanne",
        "verdict-below",
        "Günstiger als die meisten vergleichbaren Angebote.",
    ),
    "within": (
        "Im erwarteten Bereich",
        "verdict-within",
        "Entspricht vergleichbaren Angebotsmieten.",
    ),
    "above": (
        "Über der erwarteten Spanne",
        "verdict-above",
        "Teurer als die meisten vergleichbaren Angebote.",
    ),
}
LEGAL_LINKS: tuple[tuple[str, str], ...] = (
    ("Bundesamt für Wohnungswesen (BWO): Mietrecht", "https://www.bwo.admin.ch"),
    ("Schweizerischer Mieterinnen- und Mieterverband", "https://www.mieterverband.ch"),
)


def layout(services: Services) -> html.Div:
    """Form on the left, result on the right."""
    if services.estimator is None:
        return html.Div(model_missing_notice(), className="page")
    form = html.Div(
        [
            html.Div("Für Mietende", className="eyebrow"),
            html.H2("Fair-Rent Check", className="panel-title"),
            html.P(
                "Ist die verlangte Nettomiete marktüblich? Vergleich mit aktuellen "
                "Angebotsmieten ähnlicher Wohnungen.",
                className="lead",
            ),
            address_field("tenant-address"),
            html.Div(
                [
                    field(
                        "Wohnfläche (m²)",
                        dcc.Input(
                            id="tenant-area", type="number", min=10, max=500, step=1, value=75
                        ),
                    ),
                    field(
                        "Zimmer",
                        dcc.Input(
                            id="tenant-rooms", type="number", min=1, max=15, step=0.5, value=3.5
                        ),
                    ),
                ],
                className="field-row",
            ),
            field(
                "Nettomiete pro Monat (CHF)",
                dcc.Input(id="tenant-rent", type="number", min=1, step=1, value=1800),
            ),
            field(
                "Inseratetext (optional)",
                dcc.Textarea(
                    id="tenant-text",
                    className="textarea",
                    placeholder="z. B. «Renovierte Wohnung mit Balkon und Lift …»",
                ),
                "Wird nur ausgewertet, nicht gespeichert.",
            ),
            html.Button("Miete prüfen", id="tenant-submit", className="primary-button", n_clicks=0),
        ],
        className="panel form-card",
    )
    result = dcc.Loading(
        html.Div(id="tenant-result", children=_placeholder()), type="dot", color="#0077c8"
    )
    return html.Div(
        [html.Div([form, html.Div(result, className="result-col")], className="tool-layout")],
        className="page",
    )


def _placeholder() -> html.Div:
    return html.Div(
        [
            html.H3("So funktioniert's", className="section-title"),
            html.Ol(
                [
                    html.Li(
                        "Adresse wählen: Gebäude- und Lagedaten kommen aus den amtlichen "
                        "Registern (GWR, swisstopo, ARE, BFS)."
                    ),
                    html.Li("Fläche, Zimmer und Miete eingeben."),
                    html.Li(
                        "RentLens schätzt die marktübliche Miete mit einem kalibrierten "
                        "80 %-Intervall und zeigt, wo Ihre Miete liegt."
                    ),
                ]
            ),
        ],
        className="panel result-card muted",
    )


def register(app: Dash, services: Services) -> None:
    """Register the page and its callbacks."""
    dash.register_page(
        "tenant",
        path="/mietcheck",
        name="Mietcheck",
        title="RentLens · Fair-Rent Check",
        layout=layout(services),
        order=1,
    )
    if services.estimator is None:
        return
    register_address(app, "tenant-address", services.search)

    @app.callback(
        Output("tenant-result", "children"),
        Input("tenant-submit", "n_clicks"),
        State("tenant-address", "value"),
        State("tenant-area", "value"),
        State("tenant-rooms", "value"),
        State("tenant-rent", "value"),
        State("tenant-text", "value"),
        prevent_initial_call=True,
    )
    def check(
        _n: int,
        address: str | None,
        area: float | None,
        rooms: float | None,
        rent: float | None,
        text: str | None,
    ) -> html.Div:
        return run_check(services, address, area, rooms, rent, text)


def run_check(
    services: Services,
    address: str | None,
    area: float | None,
    rooms: float | None,
    rent: float | None,
    text: str | None,
) -> html.Div:
    """Validate the form, estimate and render the result (errors as a message)."""
    if not address or area is None or rooms is None or rent is None:
        return error_card("Bitte Adresse, Wohnfläche, Zimmer und Miete angeben.")
    apartment = Apartment(decode_address(address), float(area), float(rooms), text)
    try:
        estimate = services.estimator.estimate(
            apartment, services.lookup(apartment.address), asking_rent=float(rent)
        )
    except EstimateError as exc:
        return error_card(str(exc))
    result = estimate.check
    title, css, explanation = VERDICTS[result.verdict]
    blocks = [
        html.Div(
            [
                html.Div("Ergebnis", className="eyebrow"),
                html.H2(title, className=f"verdict {css}"),
                html.P(
                    f"{chf(rent)} – wahrscheinlich zwischen {chf(result.lo_chf)} und "
                    f"{chf(result.hi_chf)} ({result.coverage:.0%}-Intervall). "
                    f"{explanation}",
                    className="lead",
                ),
            ]
        ),
        dcc.Graph(figure=interval_figure(estimate, float(rent)), config={"displayModeBar": False}),
        html.Div(
            [
                _kpi("Geschätzte Miete", chf(result.expected_chf)),
                _kpi(
                    "Marktperzentil",
                    f"ca. {100 * result.market_percentile:.0f}.",
                    "Anteil vergleichbarer Angebote, die günstiger sind",
                ),
                _kpi(
                    "Angebotsband",
                    f"{chf(result.band25_chf)}–{chf(result.band75_chf)}",
                    "mittlere 50 %",
                ),
            ],
            className="stats-grid three",
        ),
        location_note(estimate),
        drivers_block(estimate.drivers),
        attributes_block(estimate.attributes),
        legal_block(estimate.unit["canton"]),
    ]
    return html.Div([b for b in blocks if b is not None], className="panel result-card")


def _kpi(label: str, value: str, note: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="stat-label"),
            html.Div(value, className="stat-value"),
            html.Div(note, className="stat-note"),
        ],
        className="stat",
    )


def error_card(message: str) -> html.Div:
    """Inline error message in place of a result."""
    return html.Div(
        [
            html.H3("Keine Schätzung möglich", className="section-title"),
            html.P(message, className="note"),
        ],
        className="panel result-card",
    )


def legal_block(canton: str) -> html.Div:
    """Disclaimer and official links (no legal assessment)."""
    links = [
        html.Li(html.A(text, href=url, target="_blank", rel="noopener"))
        for text, url in LEGAL_LINKS
    ]
    return html.Div(
        [
            html.H3("Rechtliches", className="section-title"),
            html.P(
                "RentLens vergleicht mit aktuellen Angebotsmieten und macht keine rechtliche "
                "Beurteilung (Art. 269 ff. OR). Den Anfangsmietzins können Sie in allen Kantonen "
                "innert 30 Tagen nach Übernahme der Wohnung bei der Schlichtungsbehörde "
                f"anfechten (Art. 270 OR); zuständig ist die Schlichtungsbehörde im Kanton "
                f"{canton}.",
                className="stat-note",
            ),
            html.Ul(links, className="link-list"),
        ],
        className="result-block legal",
    )
