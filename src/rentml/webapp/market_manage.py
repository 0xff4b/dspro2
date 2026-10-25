"""Landlord dashboard of one listing (``/verwalten/<token>``).

The secret link from publishing gives access: applications in order of arrival with their
contact data, status changes, a new asking rent with a fresh price check, pausing and deleting.
RentLens does not score or rank applicants; the decision stays with the landlord.
"""

import dash
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html

from rentml.market.models import Application, Listing, MarketError
from rentml.market.service import Market
from rentml.webapp.components import chf, field
from rentml.webapp.market_detail import APPLICATION_STATUS
from rentml.webapp.market_ui import (
    STATUS_LABELS,
    error_note,
    facts_text,
    fmt_date,
    insight_block,
    listing_url,
    notice,
    planned_grid,
)

LISTING_ACTIONS: dict[str, tuple[str, str]] = {
    "manage-pause": ("Pausieren", "paused"),
    "manage-activate": ("Online stellen", "active"),
    "manage-rented": ("Als vermietet markieren", "rented"),
}
APPLICATION_ACTIONS: tuple[tuple[str, str], ...] = (
    ("invited", "Einladen"),
    ("accepted", "Zusage"),
    ("declined", "Absage"),
)
PLANNED_MANAGE: tuple[tuple[str, str], ...] = (
    ("E-Mail-Benachrichtigung", "Neue Bewerbungen und Statusänderungen per E-Mail."),
    ("Besichtigungstermine", "Termine anbieten und Zusagen verwalten."),
    ("Nachrichten", "Mit Bewerbenden direkt schreiben."),
    ("Inserat bearbeiten", "Text und Fotos nachträglich ändern."),
    ("Digitaler Mietvertrag", "Vertrag und amtliches Formular vorbereiten."),
    ("Statistik", "Aufrufe und Bewerbungen im Zeitverlauf."),
)
FAIRNESS_NOTE = (
    "Reihenfolge nach Eingang. RentLens bewertet, sortiert oder filtert Bewerbende nicht; "
    "die Auswahl treffen Sie."
)


def _application(application: Application) -> html.Div:
    label, _ = APPLICATION_STATUS.get(application.status, (application.status, ""))
    details = [f"eingegangen am {fmt_date(application.created_at.date())}"]
    if application.move_in:
        details.append(f"Einzug ab {fmt_date(application.move_in)}")
    if application.household_size:
        details.append(f"{application.household_size} Person(en)")
    buttons = [
        html.Button(
            text,
            id={"type": "manage-app", "app": application.id, "status": status},
            className="ghost-button small",
            n_clicks=0,
            disabled=application.status == status,
        )
        for status, text in APPLICATION_ACTIONS
    ]
    contact: list = [html.A(application.email, href=f"mailto:{application.email}")]
    if application.phone:
        contact += [" · ", application.phone]
    return html.Div(
        [
            html.Div(
                [
                    html.Span(application.name, className="applicant-name"),
                    html.Span(label, className=f"badge status-{application.status}"),
                ],
                className="applicant-head",
            ),
            html.Div(contact, className="applicant-contact"),
            html.Div(" · ".join(details), className="stat-note"),
            html.P(application.message, className="description"),
            html.Div(buttons, className="button-row"),
        ],
        className="applicant",
    )


def _price_panel(listing: Listing) -> html.Div:
    return html.Div(
        [
            html.H3("Preis ändern und neu prüfen", className="section-title"),
            html.Div(
                [
                    field(
                        "Nettomiete (CHF)",
                        dcc.Input(id="manage-rent", type="number", min=1, value=listing.rent_net),
                    ),
                    field(
                        "Nebenkosten (CHF)",
                        dcc.Input(id="manage-aux", type="number", min=0, value=listing.aux_costs),
                    ),
                ],
                className="field-row",
            ),
            html.Button(
                "Preis speichern", id="manage-reprice", className="ghost-button", n_clicks=0
            ),
        ],
        className="result-block",
    )


def _status_panel(listing: Listing) -> html.Div:
    # All buttons stay in the layout (disabled for the current status): the action callback
    # lists every button as an Input, and Dash rejects inputs that are missing from the page.
    buttons = [
        html.Button(
            text,
            id=button_id,
            className="ghost-button small",
            n_clicks=0,
            disabled=status == listing.status,
        )
        for button_id, (text, status) in LISTING_ACTIONS.items()
    ]
    delete = dcc.ConfirmDialogProvider(
        html.Button("Inserat löschen", className="ghost-button small danger"),
        id="manage-delete",
        message="Inserat mit allen Fotos und Bewerbungen endgültig löschen?",
    )
    return html.Div([*buttons, delete], className="button-row")


def body(market: Market, listing: Listing, error: str = "") -> list:
    """Dashboard content for a listing (re-rendered after every action)."""
    applications = market.store.applications(listing.id)
    cards = [_application(a) for a in applications] or [
        html.P("Noch keine Bewerbungen.", className="lead")
    ]
    public = (
        dcc.Link("Öffentliches Inserat →", href=listing_url(listing.id), className="cta-link left")
        if listing.status == "active"
        else html.P("Das Inserat ist nicht öffentlich sichtbar.", className="note")
    )
    summary = html.Div(
        [
            html.Div("Inserat verwalten", className="eyebrow"),
            html.H2(listing.title, className="panel-title"),
            html.P(
                [
                    html.Span(
                        STATUS_LABELS[listing.status], className=f"badge status-{listing.status}"
                    ),
                    f" {listing.address.label} · {facts_text(listing)} · ",
                    f"{chf(listing.rent_net)} netto",
                ],
                className="lead",
            ),
            public,
            _status_panel(listing),
            error_note(error) if error else None,
            _price_panel(listing),
        ],
        className="panel result-card",
    )
    inbox = html.Div(
        [
            html.H2(f"Bewerbungen ({len(applications)})", className="panel-title"),
            html.P(FAIRNESS_NOTE, className="stat-note"),
            *cards,
        ],
        className="panel result-card",
    )
    insight = insight_block(listing.insight, asking_label="Ihr Preis") if listing.insight else None
    main = html.Div([summary, inbox], className="stack")
    side = html.Div([insight, planned_grid(PLANNED_MANAGE)], className="stack")
    return [html.Div([main, side], className="detail-layout even")]


def manage_layout(market: Market, token: object) -> html.Div:
    """The dashboard, or a notice for an unknown link."""
    listing = market.store.listing_by_token(str(token or ""))
    if listing is None:
        text = "Der Verwaltungs-Link ist ungültig oder das Inserat wurde gelöscht."
        return html.Div(notice("Inserat nicht gefunden", text), className="page")
    return html.Div(
        [
            dcc.Store(id="manage-token", data=str(token)),
            html.Div(body(market, listing), id="manage-body"),
        ],
        className="page",
    )


def run_action(
    market: Market, token: object, trigger: object, rent: object = None, aux: object = None
) -> list:
    """Apply the triggering action and return the fresh dashboard content.

    Args:
        market: Marketplace service.
        token: Management token from the page's store.
        trigger: ``ctx.triggered_id``: a button id or a pattern id of an application button.
        rent: Value of the net-rent input (for ``manage-reprice``).
        aux: Value of the auxiliary-costs input.

    Returns:
        Children for ``manage-body``.
    """
    listing = market.store.listing_by_token(str(token or ""))
    if listing is None:
        return [notice("Inserat nicht gefunden", "Der Link ist nicht mehr gültig.")]
    error = ""
    if trigger == "manage-delete":
        market.store.delete_listing(listing.id)
        text = "Inserat, Fotos und alle Bewerbungen wurden entfernt."
        return [notice("Inserat gelöscht", text, link=("Zur Startseite →", "/"))]
    if isinstance(trigger, dict):
        market.store.set_application_status(listing.id, int(trigger["app"]), str(trigger["status"]))
    elif trigger in LISTING_ACTIONS:
        market.store.set_status(listing.id, LISTING_ACTIONS[str(trigger)][1])
    elif trigger == "manage-reprice":
        try:
            if rent is None:
                raise MarketError("Bitte die Nettomiete angeben.")
            market.reprice(listing, float(str(rent)), None if aux is None else float(str(aux)))
        except (MarketError, ValueError) as exc:
            error = str(exc) if isinstance(exc, MarketError) else "Bitte Zahlen angeben."
    fresh = market.store.listing(listing.id)
    return body(market, fresh or listing, error)


def register(app: Dash, market: Market) -> None:
    """Register the page and its single action callback."""
    dash.register_page(
        "manage",
        path_template="/verwalten/<token>",
        name="Verwalten",
        title="RentLens · Inserat verwalten",
        layout=lambda token=None, **_: manage_layout(market, token),
        nav=False,
    )

    @app.callback(
        Output("manage-body", "children"),
        [Input(button_id, "n_clicks") for button_id in LISTING_ACTIONS],
        Input("manage-reprice", "n_clicks"),
        Input("manage-delete", "submit_n_clicks"),
        Input({"type": "manage-app", "app": ALL, "status": ALL}, "n_clicks"),
        State("manage-token", "data"),
        State("manage-rent", "value"),
        State("manage-aux", "value"),
        prevent_initial_call=True,
    )
    def act(*args: object) -> list:
        token, rent, aux = args[-3:]
        # Buttons re-created by a re-render fire with n_clicks 0/None: not a click.
        if not (ctx.triggered and ctx.triggered[0].get("value")):
            return dash.no_update
        return run_action(market, token, ctx.triggered_id, rent, aux)
