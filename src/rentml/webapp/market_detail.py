"""Listing page (``/inserat/<id>``) with the application form, and the applicant's status page.

The listing page shows the stored model snapshot (price check, drivers) and the QoLI of the
municipality; nothing is recomputed. An application is stored for the landlord and can be
withdrawn (deleted) by the applicant through the secret link shown after submitting.
Applicants are never scored or ranked.
"""

import dash
from dash import Dash, Input, Output, State, dcc, html

from rentml.market.insight import TAG_LABELS
from rentml.market.models import Application, ApplicationDraft, Listing, MarketError
from rentml.market.service import ListingView, Market
from rentml.webapp.components import chf, field
from rentml.webapp.market_ui import (
    available_text,
    card_grid,
    error_note,
    facts_text,
    floor_text,
    fmt_date,
    insight_block,
    listing_url,
    notice,
    photo_box,
    photo_url,
    planned,
    planned_grid,
    quality_block,
)
from rentml.webapp.tenant import legal_block

APPLICATION_STATUS: dict[str, tuple[str, str]] = {
    "new": ("Eingegangen", "Die Vermieterschaft hat Ihre Bewerbung erhalten."),
    "invited": ("Zur Besichtigung eingeladen", "Die Vermieterschaft meldet sich für einen Termin."),
    "accepted": ("Zusage", "Die Vermieterschaft möchte Ihnen die Wohnung vermieten."),
    "declined": ("Absage", "Die Wohnung wurde anderweitig vergeben."),
}
PLANNED_APPLY: tuple[tuple[str, str], ...] = (
    ("Besichtigungstermin buchen", "Freie Termine der Vermieterschaft direkt auswählen."),
    ("Dokumente hochladen", "Betreibungsauszug und Lohnnachweis verschlüsselt beilegen."),
    ("Nachrichten", "Rückfragen im Chat statt per E-Mail."),
    ("Bewerbungsdossier", "Angaben einmal erfassen und wiederverwenden (braucht ein Konto)."),
)
PRIVACY_NOTE = (
    "Ihre Angaben werden gespeichert und nur der Vermieterschaft dieses Inserats angezeigt. "
    "Sie können die Bewerbung jederzeit über Ihren persönlichen Link löschen. RentLens bewertet "
    "oder sortiert Bewerbende nicht."
)


def application_url(token: str) -> str:
    """Secret status URL of an application."""
    return f"/bewerbung/{token}"


def _gallery(listing: Listing) -> html.Div:
    if not listing.photo_ids:
        return photo_box(listing, "gallery-main")
    thumbs = [html.Img(src=photo_url(pid), alt=listing.title) for pid in listing.photo_ids[1:]]
    return html.Div(
        [photo_box(listing, "gallery-main"), html.Div(thumbs, className="gallery-thumbs")],
        className="gallery",
    )


def _facts(listing: Listing) -> html.Div:
    gross = "–" if listing.rent_gross is None else chf(listing.rent_gross)
    rows = [
        ("Nettomiete", chf(listing.rent_net)),
        ("Nebenkosten", "–" if listing.aux_costs is None else chf(listing.aux_costs)),
        ("Bruttomiete", gross),
        ("Zimmer", f"{listing.rooms:g}"),
        ("Wohnfläche", f"{listing.area:.0f} m²"),
        ("Stockwerk", floor_text(listing.floor) or "–"),
        ("Verfügbar", available_text(listing)),
        ("Inseriert am", fmt_date(listing.created_at.date())),
    ]
    cells = [
        html.Div(
            [html.Div(k, className="stat-label"), html.Div(v, className="stat-value")],
            className="stat",
        )
        for k, v in rows
    ]
    return html.Div(cells, className="stats-grid four")


def _summary(listing: Listing) -> html.Div:
    tags = [html.Span(TAG_LABELS.get(tag, tag), className="chip") for tag in listing.tags]
    demo = (
        html.P("Demo-Inserat mit erfundenen Angaben (synthetische Daten).", className="note")
        if listing.is_demo
        else None
    )
    return html.Div(
        [
            dcc.Link("← Alle Inserate", href="/inserate", className="cta-link left"),
            html.H1(listing.title, className="listing-title"),
            html.P(f"{listing.address.label} · {facts_text(listing)}", className="lead"),
            demo,
            _gallery(listing),
            _facts(listing),
            html.H3("Beschrieb", className="section-title"),
            html.P(listing.description or "Kein Beschrieb.", className="description"),
            html.Div(tags),
            html.P(f"Anbieter: {listing.contact_name}", className="stat-note"),
        ],
        className="panel result-card",
    )


def application_form(listing: Listing) -> html.Div:
    """Application form of an active listing."""
    return html.Div(
        [
            html.Div("Für Mietende", className="eyebrow"),
            html.H2("Jetzt bewerben", className="panel-title"),
            field("Vor- und Nachname", dcc.Input(id="apply-name", type="text", maxLength=120)),
            field("E-Mail", dcc.Input(id="apply-email", type="email", maxLength=254)),
            field("Telefon (optional)", dcc.Input(id="apply-phone", type="tel", maxLength=24)),
            html.Div(
                [
                    field(
                        "Einzug ab",
                        dcc.DatePickerSingle(
                            id="apply-move-in",
                            display_format="DD.MM.YYYY",
                            first_day_of_week=1,
                            placeholder="Datum",
                            clearable=True,
                        ),
                    ),
                    field(
                        "Personen",
                        dcc.Input(id="apply-household", type="number", min=1, max=12, step=1),
                    ),
                ],
                className="field-row",
            ),
            field(
                "Nachricht",
                dcc.Textarea(
                    id="apply-message",
                    className="textarea",
                    maxLength=2000,
                    placeholder="Stellen Sie sich kurz vor …",
                ),
            ),
            dcc.Checklist(
                id="apply-consent",
                options=[{"label": "Ich bin mit der Weitergabe einverstanden.", "value": "ok"}],
                value=[],
                className="checks",
            ),
            html.P(PRIVACY_NOTE, className="field-hint"),
            html.Div(id="apply-error"),
            html.Button(
                "Bewerbung senden", id="apply-submit", className="primary-button", n_clicks=0
            ),
            dcc.Store(id="apply-listing", data=listing.id),
            planned_grid(PLANNED_APPLY),
        ],
    )


def detail_layout(market: Market, listing_id: object) -> html.Div:
    """The listing page; a notice for unknown or inactive listings."""
    listing = market.store.listing(int(listing_id)) if str(listing_id).isdigit() else None
    if listing is None or listing.status != "active":
        text = "Dieses Inserat existiert nicht oder ist nicht mehr online."
        return html.Div(
            notice("Inserat nicht verfügbar", text, link=("Zu den Inseraten →", "/inserate")),
            className="page",
        )
    view: ListingView = market.view(listing)
    insight = (
        html.Div(
            [insight_block(listing.insight), legal_block(listing.place.canton)],
            className="panel result-card flat-children",
        )
        if listing.insight is not None
        else html.Div(
            planned("Preis-Check", "Für dieses Inserat liegt keine Modellbewertung vor."),
            className="panel result-card",
        )
    )
    similar = market.similar(listing)
    more = (
        [html.H2("Ähnliche Inserate", className="panel-title"), card_grid(similar)]
        if similar
        else []
    )
    main = html.Div([_summary(listing), insight, quality_block(view), *more], className="stack")
    side = html.Div(
        [
            html.Div(application_form(listing), id="apply-box", className="panel form-card"),
            planned("Inserat melden", "Verdächtige Inserate der Moderation melden."),
        ],
        className="stack sticky",
    )
    return html.Div(html.Div([main, side], className="detail-layout"), className="page")


def submit_application(market: Market, listing_id: object, form: dict[str, object]) -> tuple:
    """Store an application; returns ``(confirmation or None, error or None)``."""
    try:
        draft = ApplicationDraft.from_form(form)
        _, token = market.store.add_application(int(str(listing_id)), draft)
    except (MarketError, ValueError) as exc:
        message = str(exc) if isinstance(exc, MarketError) else "Inserat unbekannt."
        return None, error_note(message)
    url = application_url(token)
    confirmation = [
        html.Div("Gesendet", className="eyebrow"),
        html.H2("Bewerbung eingereicht", className="panel-title"),
        html.P(
            "Die Vermieterschaft sieht Ihre Bewerbung jetzt. Über Ihren persönlichen Link sehen "
            "Sie den Status und können die Bewerbung löschen. Der Link wird nur jetzt angezeigt:",
            className="lead",
        ),
        dcc.Link("Status meiner Bewerbung →", href=url, className="secret-link"),
        planned("Bestätigung per E-Mail", "Link und Statusänderungen per E-Mail erhalten."),
    ]
    return confirmation, None


def _application_card(application: Application, listing: Listing | None, token: str) -> html.Div:
    label, text = APPLICATION_STATUS.get(application.status, (application.status, ""))
    title = listing.title if listing else "Inserat gelöscht"
    link = (
        dcc.Link("Inserat ansehen →", href=listing_url(listing.id), className="cta-link left")
        if listing and listing.status == "active"
        else None
    )
    return html.Div(
        [
            html.Div("Meine Bewerbung", className="eyebrow"),
            html.H2(label, className="panel-title"),
            html.P(text, className="lead"),
            html.P(f"{title} · eingereicht am {fmt_date(application.created_at.date())}"),
            link,
            html.P(f"{application.name} · {application.email}", className="stat-note"),
            html.P(application.message, className="description"),
            dcc.ConfirmDialogProvider(
                html.Button("Bewerbung zurückziehen und löschen", className="ghost-button danger"),
                id="applicant-withdraw",
                message="Bewerbung wirklich löschen? Das kann nicht rückgängig gemacht werden.",
            ),
            dcc.Store(id="applicant-token", data=token),
        ],
    )


def applicant_layout(market: Market, token: object) -> html.Div:
    """Status page of one application (secret link)."""
    application = market.store.application_by_token(str(token or ""))
    if application is None:
        text = "Der Link ist ungültig oder die Bewerbung wurde gelöscht."
        return html.Div(notice("Bewerbung nicht gefunden", text), className="page")
    listing = market.store.listing(application.listing_id)
    card = _application_card(application, listing, str(token))
    return html.Div(
        html.Div(card, id="applicant-box", className="panel notice-card"), className="page"
    )


def withdraw(market: Market, token: object) -> html.Div:
    """Delete an application and confirm it."""
    done = market.store.withdraw_application(str(token or ""))
    title = "Bewerbung gelöscht" if done else "Bewerbung nicht gefunden"
    return html.Div(
        [
            html.H2(title, className="panel-title"),
            html.P("Ihre Angaben wurden entfernt." if done else "", className="lead"),
            dcc.Link("Zu den Inseraten →", href="/inserate", className="cta-link left"),
        ]
    )


def register(app: Dash, market: Market) -> None:
    """Register the listing page, the applicant page and their callbacks."""
    dash.register_page(
        "listing",
        path_template="/inserat/<listing_id>",
        name="Inserat",
        title="RentLens · Inserat",
        layout=lambda listing_id=None, **_: detail_layout(market, listing_id),
        nav=False,
    )
    dash.register_page(
        "applicant",
        path_template="/bewerbung/<token>",
        name="Bewerbung",
        title="RentLens · Meine Bewerbung",
        layout=lambda token=None, **_: applicant_layout(market, token),
        nav=False,
    )

    @app.callback(
        Output("apply-box", "children"),
        Output("apply-error", "children"),
        Input("apply-submit", "n_clicks"),
        State("apply-listing", "data"),
        State("apply-name", "value"),
        State("apply-email", "value"),
        State("apply-phone", "value"),
        State("apply-move-in", "date"),
        State("apply-household", "value"),
        State("apply-message", "value"),
        State("apply-consent", "value"),
        prevent_initial_call=True,
    )
    def apply(_n: int, listing_id: int, *values: object) -> tuple:
        keys = ("name", "email", "phone", "move_in", "household_size", "message", "consent")
        confirmation, error = submit_application(
            market, listing_id, dict(zip(keys, values, strict=True))
        )
        return (dash.no_update, error) if confirmation is None else (confirmation, dash.no_update)

    @app.callback(
        Output("applicant-box", "children"),
        Input("applicant-withdraw", "submit_n_clicks"),
        State("applicant-token", "data"),
        prevent_initial_call=True,
    )
    def withdraw_application(_n: int, token: str) -> html.Div:
        return withdraw(market, token)
