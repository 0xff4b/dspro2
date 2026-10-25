"""Publish a listing (``/inserieren``) with the live price assistant.

The landlord enters the object once. "Preis prüfen" rates the asking rent with the rent model
(recommended band, calibrated interval, verdict, what-if) and checks for duplicates without
storing anything; "Veröffentlichen" stores the listing with its model snapshot and shows the
secret management link once.
"""

import dash
import flask
from dash import Dash, Input, Output, State, dcc, html

from rentml.geoadmin import Address
from rentml.market import photos
from rentml.market.insight import TAG_LABELS
from rentml.market.models import ListingDraft, MarketError
from rentml.market.service import Assessment, Market
from rentml.webapp.components import (
    address_field,
    attributes_block,
    chf,
    decode_address,
    field,
    register_address,
)
from rentml.webapp.landlord import what_if_figure
from rentml.webapp.market_ui import (
    error_note,
    insight_block,
    listing_url,
    planned,
    planned_grid,
    verdict_badge,
)
from rentml.webapp.services import Services

FORM_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("title", "pub-title", "value"),
    ("area", "pub-area", "value"),
    ("rooms", "pub-rooms", "value"),
    ("floor", "pub-floor", "value"),
    ("rent_net", "pub-rent", "value"),
    ("aux_costs", "pub-aux", "value"),
    ("available_from", "pub-available", "date"),
    ("description", "pub-text", "value"),
    ("contact_name", "pub-name", "value"),
    ("contact_email", "pub-email", "value"),
    ("contact_phone", "pub-phone", "value"),
)
PREVIEW_DEFAULTS: dict[str, str] = {
    "title": "Entwurf",
    "contact_name": "Entwurf",
    "contact_email": "entwurf@example.invalid",
}
PLANNED_PUBLISH: tuple[tuple[str, str], ...] = (
    ("GWR-Wohnungsauswahl", "Fläche, Zimmer und Stockwerk aus dem Wohnungsregister vorbelegen."),
    ("What-if Renovation, Balkon, Lift", "Braucht Textmerkmale im Mietmodell (RQ1)."),
    ("Eigenes Attributmodell", "Merkmale im Beschrieb erkennt aktuell ein Regelwerk."),
    ("Grundriss und Video", "Weitere Medien neben Fotos."),
    ("Import (IDX/SwissRETS)", "Inserate aus der Bewirtschaftungssoftware übernehmen."),
    ("Konto für Vermietende", "Mehrere Inserate unter einem Login verwalten."),
)
CONFIRM_TEXT = "Ich bin berechtigt, dieses Objekt zu vermieten, und die Angaben stimmen."
PREVIEW_JS = """
function(contents) {
    if (!contents) { return []; }
    return contents.map(function(src) {
        var props = {src: src, className: 'thumb'};
        return {type: 'Img', namespace: 'dash_html_components', props: props};
    });
}
"""


def manage_url(token: str) -> str:
    """Secret management URL of a listing."""
    return f"/verwalten/{token}"


def _address(value: str | None) -> Address | None:
    return decode_address(value) if value else None


def _object_fields() -> list:
    return [
        field("Titel", dcc.Input(id="pub-title", type="text", maxLength=120)),
        address_field("pub-address"),
        html.Div(
            [
                field("Wohnfläche (m²)", dcc.Input(id="pub-area", type="number", min=10, max=500)),
                field("Zimmer", dcc.Input(id="pub-rooms", type="number", min=1, max=15, step=0.5)),
                field(
                    "Stockwerk", dcc.Input(id="pub-floor", type="number", min=-1, max=60, step=1)
                ),
            ],
            className="field-row three",
        ),
        html.Div(
            [
                field("Nettomiete (CHF)", dcc.Input(id="pub-rent", type="number", min=1, step=1)),
                field("Nebenkosten (CHF)", dcc.Input(id="pub-aux", type="number", min=0, step=1)),
                field(
                    "Verfügbar ab",
                    dcc.DatePickerSingle(
                        id="pub-available",
                        display_format="DD.MM.YYYY",
                        first_day_of_week=1,
                        placeholder="n. V.",
                        clearable=True,
                    ),
                ),
            ],
            className="field-row three",
        ),
        field(
            "Beschrieb",
            dcc.Textarea(id="pub-text", className="textarea tall", maxLength=4000),
            "Telefonnummern, E-Mail-Adressen und Links werden entfernt; der Kontakt läuft über "
            "die Bewerbung.",
        ),
        field(
            "Ausstattung",
            dcc.Checklist(
                id="pub-tags",
                options=[{"label": label, "value": tag} for tag, label in TAG_LABELS.items()],
                value=[],
                className="checks two",
            ),
            "Zusätzlich erkennt RentLens Merkmale automatisch im Beschrieb.",
        ),
    ]


def _media_and_contact() -> list:
    limit_mb = photos.MAX_UPLOAD_BYTES // (1024 * 1024)
    return [
        field(
            "Fotos",
            dcc.Upload(
                html.Div("Fotos hierher ziehen oder klicken"),
                id="pub-photos",
                multiple=True,
                accept="image/jpeg,image/png,image/webp",
                max_size=photos.MAX_UPLOAD_BYTES,
                className="upload",
            ),
            f"Bis {photos.MAX_PHOTOS} Fotos, je max. {limit_mb} MB. Metadaten (z. B. GPS) werden "
            "entfernt.",
        ),
        html.Div(id="pub-photo-preview", className="thumb-row"),
        html.H3("Kontakt (nicht öffentlich)", className="section-title"),
        html.Div(
            [
                field("Name", dcc.Input(id="pub-name", type="text", maxLength=120)),
                field("E-Mail", dcc.Input(id="pub-email", type="email", maxLength=254)),
            ],
            className="field-row",
        ),
        field("Telefon (optional)", dcc.Input(id="pub-phone", type="tel", maxLength=24)),
        dcc.Checklist(
            id="pub-confirm",
            options=[{"label": CONFIRM_TEXT, "value": "ok"}],
            value=[],
            className="checks",
        ),
    ]


def layout(services: Services) -> html.Div:
    """Form on the left, price assistant on the right."""
    model_note = (
        None
        if services.estimator is not None
        else html.P("Kein Modell geladen: Inserate erscheinen ohne Preis-Check.", className="note")
    )
    form = html.Div(
        [
            html.Div("Für Vermietende", className="eyebrow"),
            html.H2("Wohnung inserieren", className="panel-title"),
            html.P("Kostenlos, ohne Konto. Der Preisassistent prüft Ihre Miete.", className="lead"),
            model_note,
            *_object_fields(),
            html.Button("Preis prüfen", id="pub-check", className="ghost-button wide", n_clicks=0),
            *_media_and_contact(),
            html.Div(id="pub-error"),
            html.Button(
                "Inserat veröffentlichen", id="pub-submit", className="primary-button", n_clicks=0
            ),
        ],
        className="panel form-card",
    )
    assistant = dcc.Loading(
        html.Div(id="pub-assistant", children=_placeholder()), type="dot", color="#0077c8"
    )
    side = html.Div([assistant, planned_grid(PLANNED_PUBLISH)], className="result-col stack")
    return html.Div(
        html.Div([form, side], className="tool-layout wide-form"), id="pub-page", className="page"
    )


def _placeholder() -> html.Div:
    return html.Div(
        [
            html.H3("Preisassistent", className="section-title"),
            html.P(
                "Adresse, Fläche, Zimmer und Miete eingeben und «Preis prüfen» wählen: RentLens "
                "zeigt das empfohlene Angebotsband, das kalibrierte 80-%-Intervall und wie "
                "Ihr Preis im Markt liegt."
            ),
        ],
        className="panel result-card muted",
    )


def assistant_view(assessment: Assessment) -> html.Div:
    """Price assistant for an assessed draft."""
    blocks: list = []
    if assessment.duplicates:
        first = assessment.duplicates[0]
        blocks.append(
            html.P(
                [
                    "Hinweis: Ein sehr ähnliches Objekt ist bereits inseriert (",
                    dcc.Link(first.title, href=listing_url(first.id)),
                    "). Bitte kein Objekt doppelt veröffentlichen.",
                ],
                className="note",
            )
        )
    if assessment.insight is None:
        blocks.append(planned("Preis-Check", "Kein Modell geladen."))
        return html.Div(blocks, className="panel result-card")
    insight = assessment.insight
    blocks.append(
        html.P(
            f"Empfohlenes Angebotsband: {chf(insight.q25)} – {chf(insight.q75)} netto.",
            className="headline headline-value",
        )
    )
    blocks.append(insight_block(insight, asking_label="Ihr Preis"))
    if assessment.what_if is not None:
        graph = dcc.Graph(
            figure=what_if_figure(assessment.what_if), config={"displayModeBar": False}
        )
        title = html.H3("What-if", className="section-title")
        blocks.append(html.Div([title, graph], className="panel result-card"))
    if assessment.estimate is not None:
        blocks.append(attributes_block(assessment.estimate.attributes))
    return html.Div([b for b in blocks if b is not None], className="stack")


def _draft(form: dict[str, object], address: str | None, *, preview: bool) -> ListingDraft:
    values = dict(form)
    if preview:
        values |= {key: values.get(key) or default for key, default in PREVIEW_DEFAULTS.items()}
    return ListingDraft.from_form(values, _address(address))


def run_assistant(
    market: Market, form: dict[str, object], address: str | None, tags: list[str] | None
) -> html.Div:
    """Validate what the assessment needs and render the assistant (errors as a message)."""
    try:
        return assistant_view(market.assess(_draft(form, address, preview=True), tags or ()))
    except MarketError as exc:
        return html.Div(error_note(str(exc)), className="panel result-card")


def success_view(listing_id: int, title: str, token: str, base_url: str, badge: object) -> html.Div:
    """Confirmation with the public link and the secret management link."""
    secret = f"{base_url.rstrip('/')}{manage_url(token)}"
    return html.Div(
        [
            html.Div("Veröffentlicht", className="eyebrow"),
            html.H2("Ihr Inserat ist online", className="panel-title"),
            html.P([f"«{title}» ", badge], className="lead"),
            dcc.Link("Inserat ansehen →", href=listing_url(listing_id), className="cta-link left"),
            html.H3("Ihr Verwaltungs-Link", className="section-title"),
            html.P(
                "Mit diesem geheimen Link sehen Sie Bewerbungen, ändern den Preis oder löschen "
                "das Inserat. Er wird nur jetzt angezeigt: bitte speichern.",
                className="note",
            ),
            html.Div(
                [
                    dcc.Link(
                        secret, href=manage_url(token), id="pub-secret", className="secret-link"
                    ),
                    dcc.Clipboard(content=secret, title="Link kopieren", className="copy"),
                ],
                className="secret-row",
            ),
            planned(
                "Link per E-Mail", "Verwaltungs-Link und neue Bewerbungen per E-Mail erhalten."
            ),
        ],
        className="panel notice-card",
    )


def publish(
    market: Market,
    form: dict[str, object],
    address: str | None,
    tags: list[str] | None,
    uploads: list[str] | None,
    confirmed: bool,
    base_url: str = "",
) -> tuple[html.Div | None, html.Div | None]:
    """Validate and publish; returns ``(success view or None, error or None)``."""
    try:
        if not confirmed:
            raise MarketError("Bitte bestätigen, dass Sie das Objekt vermieten dürfen.")
        draft = _draft(form, address, preview=False)
        images = photos.prepare(uploads)
        listing, token = market.publish(draft, photos=images, selected_tags=tags or ())
    except MarketError as exc:
        return None, error_note(str(exc))
    badge = verdict_badge(listing.insight)
    return success_view(listing.id, listing.title, token, base_url, badge), None


def register(app: Dash, services: Services) -> None:
    """Register the page and its callbacks."""
    market = services.market
    if market is None:
        raise ValueError("services.market is required for the publish page")
    dash.register_page(
        "publish",
        path="/inserieren",
        name="Inserieren",
        title="RentLens · Wohnung inserieren",
        layout=lambda **_: layout(services),
        order=1,
    )
    register_address(app, "pub-address", services.search)
    states = [State(component_id, prop) for _, component_id, prop in FORM_FIELDS]
    keys = [key for key, _, _ in FORM_FIELDS]
    app.clientside_callback(
        PREVIEW_JS, Output("pub-photo-preview", "children"), Input("pub-photos", "contents")
    )

    @app.callback(
        Output("pub-assistant", "children"),
        Input("pub-check", "n_clicks"),
        State("pub-address", "value"),
        State("pub-tags", "value"),
        *states,
        prevent_initial_call=True,
    )
    def check(_n: int, address: str | None, tags: list[str] | None, *values: object) -> html.Div:
        return run_assistant(market, dict(zip(keys, values, strict=True)), address, tags)

    @app.callback(
        Output("pub-page", "children"),
        Output("pub-error", "children"),
        Input("pub-submit", "n_clicks"),
        State("pub-address", "value"),
        State("pub-tags", "value"),
        State("pub-photos", "contents"),
        State("pub-confirm", "value"),
        *states,
        prevent_initial_call=True,
    )
    def submit(
        _n: int,
        address: str | None,
        tags: list[str] | None,
        uploads: list[str] | None,
        confirm: list[str] | None,
        *values: object,
    ) -> tuple:
        form = dict(zip(keys, values, strict=True))
        view, error = publish(
            market, form, address, tags, uploads, bool(confirm), flask.request.host_url
        )
        return (dash.no_update, error) if view is None else (view, dash.no_update)
