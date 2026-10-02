"""Building blocks shared by the tenant and landlord pages: form fields and result widgets."""

import json
from collections.abc import Callable

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

from rentml.estimate import Estimate
from rentml.extraction import ExtractedAttributes
from rentml.geoadmin import Address
from rentml.plotting import COLORS
from rentml.webapp.figures import fmt_number
from rentml.webapp.home import DROPDOWN_LABELS

MISSING_LABELS: dict[str, str] = {
    "year_built": "Baujahr",
    "apartments": "Anzahl Wohnungen",
    "land_area": "Gebäudegrundfläche",
    "oev": "ÖV-Erreichbarkeit",
    "solar": "Solareignung",
    "population": "Einwohner in der Hektare",
    "elevation": "Höhe",
}
VIEW_LABELS: dict[str, str] = {
    "lake": "Seesicht",
    "mountain": "Bergsicht",
    "city": "Stadtsicht",
    "other": "Aussicht",
}
PARKING_LABELS: dict[str, str] = {"garage": "Garage/Einstellplatz", "outdoor": "Parkplatz"}


def chf(value: float) -> str:
    """``CHF 1'234``."""
    return f"CHF {fmt_number(float(value))}"


def field(label: str, control: object, hint: str = "") -> html.Div:
    """Labelled form field."""
    parts = [html.Label(label, className="control-label"), control]
    if hint:
        parts.append(html.Div(hint, className="field-hint"))
    return html.Div(parts, className="field")


def address_field(component_id: str) -> html.Div:
    """Address search with live suggestions from geo.admin (see :func:`register_address`)."""
    dropdown = dcc.Dropdown(
        id=component_id,
        options=[],
        placeholder="Strasse, Hausnummer, Ort …",
        labels={**DROPDOWN_LABELS, "no_options_found": "Mindestens 3 Zeichen eingeben"},
        search_order="original",
    )
    return field("Adresse", dropdown, "Amtliche Adressen der Schweiz (geo.admin.ch)")


def register_address(app: Dash, component_id: str, search: Callable[[str], list[Address]]) -> None:
    """Fill the address dropdown with suggestions while the user types."""

    @app.callback(
        Output(component_id, "options"),
        Input(component_id, "search_value"),
        State(component_id, "value"),
        prevent_initial_call=True,
    )
    def suggest(query: str | None, value: str | None) -> list[dict[str, str]]:
        options = [{"label": a.label, "value": encode_address(a)} for a in search(query or "")]
        if value and value not in {o["value"] for o in options}:
            options.insert(0, {"label": decode_address(value).label, "value": value})
        return options if options or value else no_update


def encode_address(address: Address) -> str:
    """Dropdown value of an address (JSON)."""
    return json.dumps(address.to_dict(), ensure_ascii=False)


def decode_address(value: str) -> Address:
    """Inverse of :func:`encode_address`."""
    return Address.from_dict(json.loads(value))


def interval_figure(estimate: Estimate, asking: float | None = None) -> go.Figure:
    """Horizontal bar: calibrated interval, listing band (25-75 %), estimate and asking rent."""
    p = estimate.prediction
    lo, hi = float(p["lo"]), float(p["hi"])
    pad = 0.12 * (hi - lo)
    fig = go.Figure()
    _band(fig, lo, hi, "#DCEBF8", f"{estimate.coverage:.0%}-Intervall")
    _band(fig, float(p["q25"]), float(p["q75"]), "#8EC3EC", "mittlere 50 % (Angebotsband)")
    _marker(fig, float(p["expected"]), COLORS["dark"], "Schätzung", "diamond")
    if asking is not None:
        _marker(fig, asking, COLORS["reference"], "Ihre Miete", "line-ns-open")
    low = min(lo, asking or lo) - pad
    high = max(hi, asking or hi) + pad
    fig.update_layout(
        height=150,
        margin={"l": 8, "r": 8, "t": 8, "b": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        separators=".'",
        showlegend=True,
        legend={"orientation": "h", "y": -0.45, "x": 0, "font": {"size": 11}},
        xaxis={
            "range": [low, high],
            "tickformat": ",.0f",
            "ticksuffix": " CHF",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={"visible": False, "range": [-1, 1]},
        hovermode=False,
    )
    return fig


def _band(fig: go.Figure, x0: float, x1: float, color: str, name: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[-0.45, -0.45, 0.45, 0.45, -0.45],
            fill="toself",
            fillcolor=color,
            line={"width": 0},
            mode="lines",
            name=name,
        )
    )


def _marker(fig: go.Figure, x: float, color: str, name: str, symbol: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[0],
            mode="markers",
            name=name,
            marker={
                "color": color,
                "size": 18,
                "symbol": symbol,
                "line": {"width": 3, "color": color},
            },
        )
    )


def drivers_block(drivers: pd.DataFrame) -> html.Div:
    """The three largest SHAP drivers as ± percent effects on the estimate."""
    rows = [
        html.Div(
            [
                html.Span(row.label, className="driver-name"),
                html.Span(
                    f"{row.approx_pct:+.0f} %",
                    className="driver-up" if row.approx_pct >= 0 else "driver-down",
                ),
            ],
            className="driver-row",
        )
        for row in drivers.itertuples()
    ]
    return html.Div(
        [
            html.H3("Was die Schätzung am stärksten beeinflusst", className="section-title"),
            *rows,
            html.Div(
                "SHAP-Beiträge relativ zum Durchschnitt aller Trainingswohnungen.",
                className="stat-note",
            ),
        ],
        className="result-block",
    )


def attributes_block(attributes: ExtractedAttributes) -> html.Div | None:
    """Premium-relevant attributes recognised in the description, with their text evidence."""
    found = attribute_labels(attributes)
    if not found:
        return None
    chips = [html.Span(label, title=f"«{evidence}»", className="chip") for label, evidence in found]
    return html.Div(
        [
            html.H3("Im Beschrieb erkannt", className="section-title"),
            html.Div(chips),
            html.Div(
                "Können einen Aufpreis begründen; das Modell v1 nutzt den Text noch nicht.",
                className="stat-note",
            ),
        ],
        className="result-block",
    )


def attribute_labels(attributes: ExtractedAttributes) -> list[tuple[str, str]]:
    """(German label, evidence) for every positive attribute."""
    ev = attributes.evidence
    flags = {
        "has_lift": "Lift",
        "has_balcony_or_terrace": "Balkon/Terrasse",
        "is_minergie": "Minergie",
        "has_garden": "Garten",
        "has_own_washer": "Eigene Waschmaschine",
        "is_new_build": "Neubau",
        "is_attic": "Dachwohnung",
        "is_furnished": "Möbliert",
        "is_temporary": "Befristet",
        "pets_allowed": "Haustiere erlaubt",
    }
    out = [
        (label, ev.get(name, ""))
        for name, label in flags.items()
        if getattr(attributes, name) is True
    ]
    if attributes.view in VIEW_LABELS:
        out.append((VIEW_LABELS[attributes.view], ev.get("view", "")))
    if attributes.parking in PARKING_LABELS:
        out.append((PARKING_LABELS[attributes.parking], ev.get("parking", "")))
    if attributes.renovation_year:
        out.append((f"Renoviert {attributes.renovation_year}", ev.get("renovation_year", "")))
    if attributes.floor is not None:
        out.append((f"{attributes.floor}. Stock", ev.get("floor", "")))
    return out


def location_note(estimate: Estimate) -> html.Div:
    """Municipality context and inputs that geo.admin could not provide."""
    unit = estimate.unit
    text = f"Gemeinde {unit['name']}, Bezirk {unit['district_name']}, Kanton {unit['canton']}."
    parts = [html.Span(text)]
    if estimate.location.missing:
        missing = ", ".join(MISSING_LABELS.get(m, m) for m in estimate.location.missing)
        parts.append(
            html.Span(
                f" Nicht verfügbar (Schätzung ohne diese Angaben): {missing}.",
                className="warn-text",
            )
        )
    return html.P(parts, className="stat-note location-note")


def model_missing_notice() -> html.Div:
    """Shown when no model bundle is deployed."""
    return html.Div(
        [
            html.H2("Modell nicht geladen", className="panel-title"),
            html.P(
                "Für Schätzungen braucht RentLens das Modell-Bundle aus dem Notebook "
                "(models/rent_bundle_v1.joblib oder $RENTML_MODEL_BUNDLE).",
                className="lead",
            ),
        ],
        className="panel form-card",
    )
