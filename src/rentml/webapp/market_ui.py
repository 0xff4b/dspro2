"""Building blocks of the marketplace pages: cards, badges, model blocks and placeholders.

Features that are not built yet are shown as grey "Geplant" boxes (:func:`planned`) at the place
where they will live, so the prototype is honest about its gaps.
"""

from collections.abc import Sequence
from datetime import date

import pandas as pd
from dash import dcc, html

from rentml.market.insight import TAG_LABELS
from rentml.market.models import Listing, RentInsight
from rentml.market.quality import DIMENSION_LABELS, QualityProfile
from rentml.market.service import ListingView
from rentml.webapp.components import MISSING_LABELS, band_figure, chf, drivers_block
from rentml.webapp.figures import fmt_number

PHOTO_ROUTE = "/media/foto/<int:photo_id>.jpg"
N_PLACEHOLDER_STYLES = 5
MAX_CARD_TAGS = 3
VERDICT_BADGES: dict[str, tuple[str, str, str]] = {
    "below": ("Günstig", "badge-below", "Unter der erwarteten Spanne vergleichbarer Angebote"),
    "within": ("Marktüblich", "badge-within", "Im erwarteten Bereich vergleichbarer Angebote"),
    "above": ("Über Markt", "badge-above", "Über der erwarteten Spanne vergleichbarer Angebote"),
}
VERDICT_TITLES: dict[str, str] = {
    "below": "Unter der erwarteten Spanne",
    "within": "Im erwarteten Bereich",
    "above": "Über der erwarteten Spanne",
}
STATUS_LABELS: dict[str, str] = {"active": "Online", "paused": "Pausiert", "rented": "Vermietet"}


def photo_url(photo_id: int) -> str:
    """URL of a stored photo (see :data:`PHOTO_ROUTE`)."""
    return f"/media/foto/{int(photo_id)}.jpg"


def listing_url(listing_id: int) -> str:
    """Public URL of a listing."""
    return f"/inserat/{int(listing_id)}"


def fmt_date(day: date | None) -> str:
    """``01.12.2026`` (empty for ``None``)."""
    return day.strftime("%d.%m.%Y") if day else ""


def available_text(listing: Listing) -> str:
    """``"ab 01.12.2026"`` or ``"nach Vereinbarung"``."""
    return (
        f"ab {fmt_date(listing.available_from)}" if listing.available_from else "nach Vereinbarung"
    )


def floor_text(floor: int | None) -> str:
    """German floor label (empty if unknown)."""
    if floor is None:
        return ""
    if floor < 0:
        return "Untergeschoss"
    return "Erdgeschoss" if floor == 0 else f"{floor}. Stock"


def facts_text(listing: Listing) -> str:
    """``"3.5 Zimmer · 82 m² · 2. Stock"``."""
    parts = [f"{listing.rooms:g} Zimmer", f"{listing.area:.0f} m²", floor_text(listing.floor)]
    return " · ".join(part for part in parts if part)


def planned(title: str, text: str) -> html.Div:
    """Grey box for a feature that is not built yet."""
    return html.Div(
        [
            html.Span("Geplant", className="planned-tag"),
            html.Div(title, className="planned-title"),
            html.Div(text, className="planned-text"),
        ],
        className="planned",
        title="Noch nicht umgesetzt",
    )


def planned_grid(
    items: Sequence[tuple[str, str]], heading: str = "Noch nicht umgesetzt"
) -> html.Div:
    """Several grey boxes under a small heading."""
    return html.Div(
        [
            html.H3(heading, className="section-title planned-heading"),
            html.Div([planned(title, text) for title, text in items], className="planned-grid"),
        ],
        className="planned-section",
    )


def notice(title: str, text: str, *, link: tuple[str, str] | None = None) -> html.Div:
    """Card with a message (errors, unknown links, confirmations)."""
    parts: list = [html.H2(title, className="panel-title"), html.P(text, className="lead")]
    if link:
        parts.append(dcc.Link(link[0], href=link[1], className="cta-link left"))
    return html.Div(parts, className="panel notice-card")


def error_note(message: str) -> html.Div:
    """Inline validation message."""
    return html.Div(message, className="note form-error")


def verdict_badge(insight: RentInsight | None) -> html.Span | None:
    """Coloured pill with the model's verdict on the asking rent (``None`` without a model)."""
    if insight is None or insight.verdict not in VERDICT_BADGES:
        return None
    label, css, hint = VERDICT_BADGES[insight.verdict]
    return html.Span(f"Preis-Check: {label}", className=f"badge {css}", title=hint)


def photo_box(listing: Listing, class_name: str = "card-photo") -> html.Div:
    """First photo of a listing or a neutral placeholder."""
    ribbon = html.Span("Demo", className="demo-ribbon") if listing.is_demo else None
    if listing.photo_ids:
        image = html.Img(src=photo_url(listing.photo_ids[0]), alt=listing.title)
        return html.Div([image, ribbon], className=class_name)
    style = f"placeholder-{listing.id % N_PLACEHOLDER_STYLES}"
    label = html.Span(f"{listing.rooms:g} Zi. · {listing.area:.0f} m²")
    return html.Div([label, ribbon], className=f"{class_name} photo-placeholder {style}")


def listing_card(view: ListingView) -> dcc.Link:
    """Result card linking to the listing."""
    listing = view.listing
    badges = [verdict_badge(listing.insight)]
    if view.quality is not None:
        badges.append(
            html.Span(
                f"Lebensqualität {view.quality.qoli:.0f}",
                className="badge badge-neutral",
                title="Quality-of-Life-Index der Gemeinde (0–100)",
            )
        )
    if view.value is not None:
        badges.append(
            html.Span(
                f"Preis-Leistung {view.value:.0f}",
                className="badge badge-neutral",
                title="Lebensqualität relativ zum Preis unter den aktuellen Inseraten (0–100)",
            )
        )
    tags = [html.Span(TAG_LABELS.get(tag, tag), className="tag") for tag in listing.tags]
    aux = f" + NK {chf(listing.aux_costs)}" if listing.aux_costs else " netto"
    body = html.Div(
        [
            html.Div(
                [html.Span(chf(listing.rent_net), className="card-price"), html.Span(aux)],
                className="card-price-row",
            ),
            html.Div(listing.title, className="card-title"),
            html.Div(facts_text(listing), className="card-facts"),
            html.Div(
                f"{listing.place.municipality} {listing.place.canton} · {available_text(listing)}",
                className="card-place",
            ),
            html.Div([b for b in badges if b is not None], className="badge-row"),
            html.Div(tags[:MAX_CARD_TAGS], className="tag-row"),
        ],
        className="card-body",
    )
    return dcc.Link([photo_box(listing), body], href=listing_url(listing.id), className="card")


def card_grid(views: Sequence[ListingView]) -> html.Div:
    """Responsive grid of listing cards."""
    return html.Div([listing_card(view) for view in views], className="card-grid")


def _kpi(label: str, value: str, note: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="stat-label"),
            html.Div(value, className="stat-value"),
            html.Div(note, className="stat-note"),
        ],
        className="stat",
    )


def insight_block(insight: RentInsight, *, asking_label: str = "Angebotsmiete") -> html.Div:
    """The rent model's view of an asking rent: verdict, interval bar, key figures, drivers."""
    css = f"verdict verdict-{insight.verdict}"
    figure = band_figure(
        lo=insight.lo,
        hi=insight.hi,
        q25=insight.q25,
        q75=insight.q75,
        expected=insight.expected,
        coverage=insight.coverage,
        asking=insight.asking,
        asking_label=asking_label,
    )
    parts: list = [
        html.Div("RentLens Preis-Check", className="eyebrow"),
        html.H2(VERDICT_TITLES.get(insight.verdict, ""), className=css),
        html.P(
            f"{chf(insight.asking)} netto – vergleichbare Angebote liegen wahrscheinlich zwischen "
            f"{chf(insight.lo)} und {chf(insight.hi)} ({insight.coverage:.0%}-Intervall).",
            className="lead",
        ),
        dcc.Graph(figure=figure, config={"displayModeBar": False}),
        html.Div(
            [
                _kpi(
                    "Geschätzte Miete", chf(insight.expected), f"Angebot {insight.deviation:+.0%}"
                ),
                _kpi(
                    "Marktperzentil",
                    f"ca. {100 * insight.percentile:.0f}.",
                    "Anteil vergleichbarer Angebote, die günstiger sind",
                ),
                _kpi("Angebotsband", f"{chf(insight.q25)}–{chf(insight.q75)}", "mittlere 50 %"),
            ],
            className="stats-grid three",
        ),
    ]
    if insight.drivers:
        parts.append(drivers_block(pd.DataFrame(insight.drivers, columns=["label", "approx_pct"])))
    notes = [
        "Statistische Schätzung aus Angebotsmieten, keine rechtliche Beurteilung (Art. 269 ff. OR)."
    ]
    if insight.missing:
        missing = ", ".join(MISSING_LABELS.get(name, name) for name in insight.missing)
        notes.append(f"Ohne diese Angaben geschätzt: {missing}.")
    if insight.model_version:
        notes.append(f"Modell {insight.model_version} ({insight.model_created[:10]}).")
    parts.append(html.P(" ".join(notes), className="stat-note"))
    return html.Div(parts, className="panel result-card")


def _bar(label: str, score: float, class_name: str = "") -> html.Div:
    return html.Div(
        [
            html.Span(label, className="quality-name"),
            html.Span(fmt_number(score), className="rank-value"),
            html.Div(
                html.Div(className="rank-fill", style={"width": f"{max(0.0, min(score, 100.0))}%"}),
                className="rank-bar",
            ),
        ],
        className=f"rank-row {class_name}".strip(),
    )


def quality_block(view: ListingView) -> html.Div:
    """QoLI of the municipality with its dimensions and the value score of the listing."""
    profile: QualityProfile | None = view.quality
    place = view.listing.place
    if profile is None:
        return html.Div(
            planned(
                "Lebensqualität am Standort",
                "Die QoLI-Tabelle ist nicht geladen (data/app/qoli_municipalities.parquet).",
            ),
            className="panel result-card",
        )
    value = "–" if view.value is None else f"{view.value:.0f} / 100"
    core = [_bar(DIMENSION_LABELS[name], score) for name, score in profile.dimensions.items()]
    extended = [
        _bar(DIMENSION_LABELS[name], score, "extended") for name, score in profile.extended.items()
    ]
    return html.Div(
        [
            html.Div("Lebensqualität am Standort", className="eyebrow"),
            html.H2(f"QoLI {profile.qoli:.0f} / 100", className="verdict"),
            html.P(
                f"Gemeinde {place.municipality}: besser als {profile.percentile:.0f} % der "
                "Schweizer Gemeinden (Kernindex aus ÖV, Lärm, Versorgung und Grünraum).",
                className="lead",
            ),
            html.Div(
                [
                    _kpi("Preis-Leistung", value, "Lebensqualität relativ zum Preis pro m²"),
                    _kpi("Miete pro m²", f"CHF {view.listing.chf_per_m2:.1f}", "netto pro Monat"),
                ],
                className="stats-grid",
            ),
            html.Div(core, className="ranking"),
            html.H3("Erweiterte Dimensionen", className="section-title") if extended else None,
            html.Div(extended, className="ranking"),
            html.P(
                "Gemeindemittel (bevölkerungsgewichtete Hektaren), nicht die einzelne Fassade. "
                "Quellen: ARE, BAFU, BFS, MeteoSchweiz, ESTV (OGD); © OpenStreetMap-Mitwirkende "
                "(ODbL).",
                className="stat-note",
            ),
            planned(
                "Hektargenaue Lebensqualität und eigene Gewichte",
                "QoLI für die exakte Adresse statt Gemeindemittel, mit persönlicher Gewichtung "
                "der Dimensionen (z. B. Familie, Pendeln).",
            ),
        ],
        className="panel result-card",
    )
