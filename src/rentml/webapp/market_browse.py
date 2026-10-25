"""Start page (``/``) and listing search (``/inserate``) of the marketplace.

Every card carries the model's verdict on the asking rent, the QoLI of the municipality and the
value score; the search can filter for fairly priced listings and sort by these scores.
"""

import dash
from dash import Dash, Input, Output, dcc, html

from rentml.market.insight import TAG_LABELS
from rentml.market.models import ListingQuery
from rentml.market.service import Market
from rentml.webapp.components import field
from rentml.webapp.home import DROPDOWN_LABELS
from rentml.webapp.market_ui import card_grid, planned, planned_grid

PAGE_SIZE = 60
N_START_CARDS = 6
ROOM_OPTIONS: tuple[float, ...] = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6)
SORT_LABELS: dict[str, str] = {
    "newest": "Neueste zuerst",
    "rent_asc": "Miete aufsteigend",
    "rent_desc": "Miete absteigend",
    "deal": "Preis-Check: günstigste relativ zur Schätzung",
    "value": "Beste Preis-Leistung (Value Score)",
    "qoli": "Höchste Lebensqualität (QoLI)",
}
FILTER_IDS: tuple[str, ...] = (
    "br-text",
    "br-canton",
    "br-rooms",
    "br-rent",
    "br-area",
    "br-tags",
    "br-fair",
    "br-sort",
)
EMPTY_HINT = "uv run python -m rentml.market.seed"
PLANNED_SEARCH: tuple[tuple[str, str], ...] = (
    ("Suchabo", "Neue passende Inserate per E-Mail erhalten."),
    ("Merkliste", "Inserate speichern und vergleichen (braucht ein Konto)."),
    ("Kartenansicht der Treffer", "Inserate als Punkte auf der LV95-Karte."),
    ("Persönliche QoLI-Gewichte", "Lebensqualität nach eigenen Prioritäten gewichten."),
)
PILLARS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Preis-Check in jedem Inserat",
        "Das Mietmodell vergleicht die Angebotsmiete mit einem kalibrierten 80-%-Intervall "
        "vergleichbarer Wohnungen: günstig, marktüblich oder über Markt.",
        "Miete selbst prüfen →",
        "/mietcheck",
    ),
    (
        "Lebensqualität am Standort",
        "Der Quality-of-Life-Index bewertet ÖV, Lärm, Versorgung und Grünraum der Gemeinde; "
        "der Value Score setzt ihn ins Verhältnis zum Preis.",
        "Mietpreiskarte ansehen →",
        "/karte",
    ),
    (
        "Preisassistent für Vermietende",
        "Beim Inserieren schlägt RentLens ein Angebotsband vor und zeigt, wie der gewählte "
        "Preis im Markt liegt, bevor das Inserat online geht.",
        "Wohnung inserieren →",
        "/inserieren",
    ),
)


def query_from_filters(values: dict[str, object]) -> ListingQuery:
    """Search query from the filter widgets (keys: text, canton, rooms, rent, area, tags, ...)."""
    return ListingQuery.from_form(
        {
            "text": values.get("text"),
            "canton": values.get("canton"),
            "rooms_min": values.get("rooms"),
            "rent_max": values.get("rent"),
            "area_min": values.get("area"),
            "tags": values.get("tags"),
            "fair_only": values.get("fair"),
            "sort": values.get("sort"),
        }
    )


def results(market: Market, query: ListingQuery) -> html.Div:
    """Result count and cards for a query."""
    views = market.browse(query)
    if not views:
        empty = market.store.stats()["active"] == 0
        text = (
            f"Noch keine Inserate. Erstes Inserat erfassen oder Demo-Daten laden: {EMPTY_HINT}"
            if empty
            else "Keine Inserate für diese Filter. Bitte Filter lockern."
        )
        return html.Div(html.P(text, className="note"), className="results")
    shown = views[:PAGE_SIZE]
    count = f"{len(views)} Inserate" if len(views) != 1 else "1 Inserat"
    more = f" · die ersten {PAGE_SIZE} werden angezeigt" if len(views) > PAGE_SIZE else ""
    return html.Div(
        [html.Div(f"{count}{more}", className="result-count"), card_grid(shown)],
        className="results",
    )


def _filters(market: Market, query: ListingQuery) -> html.Div:
    cantons = [{"label": c, "value": c} for c in market.store.cantons()]
    rooms = [{"label": f"ab {r:g}", "value": r} for r in ROOM_OPTIONS]
    tags = [{"label": label, "value": tag} for tag, label in TAG_LABELS.items()]
    return html.Div(
        [
            html.Div("Wohnung finden", className="eyebrow"),
            html.H2("Inserate", className="panel-title"),
            field(
                "Ort, PLZ oder Stichwort",
                dcc.Input(id="br-text", type="text", value=query.text, debounce=True),
            ),
            html.Div(
                [
                    field(
                        "Kanton",
                        dcc.Dropdown(
                            id="br-canton",
                            options=cantons,
                            value=query.canton,
                            placeholder="alle",
                            labels=DROPDOWN_LABELS,
                        ),
                    ),
                    field(
                        "Zimmer",
                        dcc.Dropdown(
                            id="br-rooms",
                            options=rooms,
                            value=query.rooms_min,
                            placeholder="alle",
                            searchable=False,
                            labels=DROPDOWN_LABELS,
                        ),
                    ),
                ],
                className="field-row",
            ),
            html.Div(
                [
                    field(
                        "Nettomiete bis (CHF)",
                        dcc.Input(
                            id="br-rent", type="number", min=0, value=query.rent_max, debounce=True
                        ),
                    ),
                    field(
                        "Fläche ab (m²)",
                        dcc.Input(
                            id="br-area", type="number", min=0, value=query.area_min, debounce=True
                        ),
                    ),
                ],
                className="field-row",
            ),
            field(
                "Preis-Check",
                dcc.Checklist(
                    id="br-fair",
                    options=[{"label": "Nur günstig oder marktüblich bewertete", "value": "fair"}],
                    value=["fair"] if query.fair_only else [],
                    className="checks",
                ),
                "Urteil des Mietmodells zum Angebotspreis.",
            ),
            field(
                "Ausstattung",
                dcc.Checklist(
                    id="br-tags", options=tags, value=list(query.tags), className="checks"
                ),
                "Aus dem Beschrieb erkannt oder von Vermietenden angegeben.",
            ),
            field(
                "Sortierung",
                dcc.Dropdown(
                    id="br-sort",
                    options=[{"label": v, "value": k} for k, v in SORT_LABELS.items()],
                    value=query.sort,
                    clearable=False,
                    searchable=False,
                    labels=DROPDOWN_LABELS,
                ),
            ),
            planned_grid(PLANNED_SEARCH),
        ],
        className="panel form-card",
    )


def browse_layout(market: Market, query: ListingQuery) -> html.Div:
    """Filters on the left, result cards on the right."""
    listing_results = html.Div(id="br-results", children=results(market, query))
    return html.Div(
        html.Div(
            [_filters(market, query), html.Div(listing_results, className="result-col")],
            className="tool-layout",
        ),
        className="page",
    )


def _hero() -> html.Div:
    rooms = [html.Option("Zimmer: alle", value="")]
    rooms += [html.Option(f"ab {r:g} Zimmer", value=f"{r:g}") for r in ROOM_OPTIONS]
    form = html.Form(
        [
            dcc.Input(
                name="ort", type="text", placeholder="Ort, PLZ oder Kanton", className="hero-input"
            ),
            html.Select(rooms, name="zimmer", className="hero-input"),
            dcc.Input(
                name="miete", type="number", placeholder="Miete bis CHF", className="hero-input"
            ),
            html.Button("Suchen", type="submit", className="primary-button hero-button"),
        ],
        action="/inserate",
        method="get",
        className="hero-form",
    )
    return html.Div(
        [
            html.H1("Mieten mit eingebautem Preis-Check", className="hero-title"),
            html.P(
                "Wohnungen in der ganzen Schweiz. Jedes Inserat wird vom RentLens-Modell bewertet: "
                "Ist die Miete marktüblich, und wie lebt es sich am Standort?",
                className="hero-lead",
            ),
            form,
        ],
        className="hero",
    )


def start_layout(market: Market) -> html.Div:
    """Start page: search hero, key figures, newest listings and what the models add."""
    stats = market.store.stats()
    figures = [
        (str(stats["active"]), "aktive Inserate"),
        (str(stats["fair"]), "günstig oder marktüblich bewertet"),
        (str(stats["municipalities"]), "Gemeinden"),
    ]
    pillars = [
        html.Div(
            [
                html.H3(title, className="pillar-title"),
                html.P(text),
                dcc.Link(link, href=href, className="cta-link left"),
            ],
            className="panel pillar",
        )
        for title, text, link, href in PILLARS
    ]
    newest = market.browse(ListingQuery())[:N_START_CARDS]
    listings = (
        card_grid(newest)
        if newest
        else html.P(f"Noch keine Inserate. Demo-Daten laden: {EMPTY_HINT}", className="note")
    )
    return html.Div(
        [
            _hero(),
            html.Div(
                [
                    html.Div([html.B(value), html.Span(label)], className="hero-stat")
                    for value, label in figures
                ],
                className="hero-stats",
            ),
            html.Div(
                [
                    html.H2("Neueste Inserate", className="panel-title"),
                    dcc.Link("Alle Inserate →", href="/inserate", className="cta-link left"),
                ],
                className="section-head",
            ),
            listings,
            html.Div(pillars, className="pillar-grid"),
            planned(
                "Konto, Suchabo und Merkliste",
                "Anmelden, Suchen speichern und über neue Inserate benachrichtigt werden.",
            ),
        ],
        className="page page-start",
    )


def register(app: Dash, market: Market) -> None:
    """Register the start page, the search page and the filter callback."""
    dash.register_page(
        "start",
        path="/",
        name="Start",
        title="RentLens · Mieten mit Preis-Check",
        layout=lambda **_: start_layout(market),
        nav=False,
    )

    def page(
        ort: str | None = None, zimmer: str | None = None, miete: str | None = None, **_: object
    ) -> html.Div:
        query = query_from_filters({"text": ort, "rooms": zimmer, "rent": miete})
        return browse_layout(market, query)

    dash.register_page(
        "browse",
        path="/inserate",
        name="Inserate",
        title="RentLens · Inserate",
        layout=page,
        order=0,
    )

    @app.callback(
        Output("br-results", "children"),
        [Input(component_id, "value") for component_id in FILTER_IDS],
        prevent_initial_call=True,
    )
    def update(*values: object) -> html.Div:
        keys = [component_id.removeprefix("br-") for component_id in FILTER_IDS]
        return results(market, query_from_filters(dict(zip(keys, values, strict=True))))
