"""RentLens: Dash app with the Fair-Rent Check, the landlord price band and the rent map.

Start with ``uv run --extra app python -m rentml.app`` (``--help`` for options). The model bundle
is read from ``$RENTML_MODEL_BUNDLE`` or ``config.BUNDLE_PATH``, the map layers
(:func:`rentml.geo.write_map_layers`) and the QoLI table from ``config.APP_DATA``. Without a
bundle the map still works and the two tools show a notice. Inputs of the tools are only used to
answer the request and are never stored or logged.
"""

import argparse
import json
import logging
import os
import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from dash import Dash, Input, Output, State, dcc, html, no_update

from rentml import data, geo, intervals, models
from rentml.app_figures import interval_figure, map_figure, what_if_figure
from rentml.config import APP_DATA, AREA_RANGE, BUNDLE_PATH, MIN_MAP_CELL, ROOMS_RANGE
from rentml.evaluation import chf
from rentml.intervals import RentModelBundle

logger = logging.getLogger(__name__)

MODEL_ENV = "RENTML_MODEL_BUNDLE"
QOLI_FILE = "qoli_municipalities.parquet"
ASSETS = Path(__file__).resolve().parent / "assets"
PAGES = {"/mietcheck": "Mietcheck", "/vermieter": "Preisband", "/karte": "Karte"}
LEVELS = {"canton": "Kantone", "district": "Bezirke", "municipality": "Gemeinden"}
GRAPH = {"displayModeBar": False}
DROPDOWN_LABELS = {"search": "Suchen", "clear_search": "Suche löschen",
                   "clear_selection": "Auswahl aufheben",
                   "no_options_found": "Keine Treffer"}  # fmt: skip
VERDICTS = {
    "below": ("Unter der erwarteten Spanne", "Günstiger als die meisten vergleichbaren Angebote."),
    "within": ("Im erwarteten Bereich", "Entspricht vergleichbaren Angebotsmieten."),
    "above": ("Über der erwarteten Spanne", "Teurer als die meisten vergleichbaren Angebote."),
}
LEGAL_LINKS = (
    ("Bundesamt für Wohnungswesen (BWO): Mietrecht", "https://www.bwo.admin.ch"),
    ("Schweizerischer Mieterinnen- und Mieterverband", "https://www.mieterverband.ch"),
)
DRIVER_LABELS = {
    "re_canton": "Kanton (Mietniveau)", "re_district": "Bezirk (Mietniveau)",
    "re_municipality": "Gemeinde (Mietniveau)", "area": "Wohnfläche", "rooms": "Zimmerzahl",
    "area_per_room": "Fläche pro Zimmer", "year_built": "Baujahr", "building_age": "Gebäudealter",
    "apartments": "Wohnungen im Gebäude", "land_area": "Gebäudegrundfläche",
    "land_area_per_apartment": "Grundfläche pro Wohnung", "population": "Einwohner in der Hektare",
    "oev": "ÖV-Erreichbarkeit", "solar": "Solareignung Dach", "elevation": "Höhe ü. M.",
    "muni_density": "Bevölkerungsdichte Gemeinde", "muni_population": "Einwohner Gemeinde",
    "east": "Lage Ost-West", "north": "Lage Nord-Süd", "rot45_x": "Lage (diagonal)",
    "rot45_y": "Lage (diagonal)",
}  # fmt: skip
FLAG_LABELS = {"has_lift": "Lift", "has_balcony_or_terrace": "Balkon/Terrasse",
               "is_minergie": "Minergie", "has_garden": "Garten",
               "has_own_washer": "Eigene Waschmaschine", "is_new_build": "Neubau",
               "is_attic": "Dachwohnung", "is_furnished": "Möbliert", "is_temporary": "Befristet",
               "pets_allowed": "Haustiere erlaubt"}  # fmt: skip
ENUM_LABELS = {"view": {"lake": "Seesicht", "mountain": "Bergsicht", "city": "Stadtsicht",
                        "other": "Aussicht"},
               "parking": {"garage": "Garage/Einstellplatz", "outdoor": "Parkplatz"}}  # fmt: skip
QOLI_LABELS = {"transport": "ÖV-Anbindung", "noise": "Ruhe (Strassen- und Bahnlärm)",
               "amenities": "Versorgung und Freizeit", "green": "Grünraum"}  # fmt: skip
FOOTER = ("Prototyp DSPRO2, HSLU. Eingaben werden nicht gespeichert. Modellurteile sind "
          "statistische Schätzungen, keine Rechtsberatung.")  # fmt: skip

TOOL_TEXT = {  # eyebrow, title, lead, button
    "tenant": ("Für Mietende", "Fair-Rent Check", "Ist die verlangte Nettomiete marktüblich? "
               "Vergleich mit aktuellen Angebotsmieten ähnlicher Wohnungen.", "Miete prüfen"),
    "landlord": ("Für Vermietende", "Preisband & What-if", "Welche Nettomiete ist für die "
                 "Wohnung marktüblich, bevor Sie sie ausschreiben?", "Preisband berechnen"),
}  # fmt: skip
LEGAL_TEXT = (
    "RentLens vergleicht mit aktuellen Angebotsmieten und macht keine rechtliche Beurteilung "
    "(Art. 269 ff. OR). Den Anfangsmietzins können Sie in allen Kantonen innert 30 Tagen nach "
    "Übernahme der Wohnung bei der Schlichtungsbehörde anfechten (Art. 270 OR); zuständig ist "
    "die Schlichtungsbehörde im Kanton {canton}."
)
NOTES = {
    "drivers": "SHAP-Beiträge relativ zum Durchschnitt aller Trainingswohnungen.",
    "text": "Können einen Aufpreis begründen; die Schätzung nutzt den Text nicht.",
    "what_if": "Nur die Fläche geändert, alles andere gleich. Das Modell ist monoton in der "
    "Fläche: mehr Fläche ergibt nie eine tiefere Schätzung.",
    "no_model": "Für Schätzungen braucht RentLens das Modell-Bundle (models/rent_bundle_v1.joblib "
    "oder $RENTML_MODEL_BUNDLE).",
    "no_layers": "Es fehlen die Kartenebenen data/app/map_*.parquet und map_meta.json "
    "(geo.write_map_layers); sie werden für Karte und Schätzung gebraucht.",
}


class InputError(ValueError):
    """Invalid input or an address outside the model's domain (the message is shown)."""


@dataclass
class Resources:
    """Everything the pages need; ``None`` parts disable the pages that depend on them."""

    bundle: RentModelBundle | None = None
    layers: dict[str, gpd.GeoDataFrame] | None = None
    meta: dict = field(default_factory=dict)
    qoli: pd.DataFrame | None = None
    session: requests.Session | None = None
    units: gpd.GeoDataFrame | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        # The point-in-polygon lookup needs LV95; reproject once instead of on every request.
        if self.layers is not None:
            self.units = self.layers["municipality"].to_crs(geo.LV95)


def _load_bundle(path: Path) -> RentModelBundle | None:
    errors = (OSError, ImportError, AttributeError, EOFError, TypeError, pickle.UnpicklingError)
    try:
        bundle = RentModelBundle.load(path)
        if not isinstance(bundle, RentModelBundle):
            raise TypeError(f"{type(bundle).__name__} is not an intervals.RentModelBundle")
    except errors as exc:
        logger.warning("No usable model bundle at %s (%s): tools disabled", path, exc)
        return None
    return bundle


def load_resources(bundle_path: Path | None = None, app_data: Path = APP_DATA) -> Resources:
    """Load model bundle, map layers and QoLI table; missing files only disable features."""
    bundle = _load_bundle(Path(bundle_path or os.environ.get(MODEL_ENV) or BUNDLE_PATH))
    layers, meta = None, {}
    try:
        layers, meta = geo.load_map_layers(app_data)
    except FileNotFoundError as exc:
        logger.warning("No map layers in %s (%s)", app_data, exc)
    qoli = pd.read_parquet(app_data / QOLI_FILE) if (app_data / QOLI_FILE).is_file() else None
    return Resources(bundle, layers, meta, qoli, requests.Session())


def encode_address(address: geo.Address) -> str:
    """Dropdown value of an address (JSON)."""
    return json.dumps(asdict(address), ensure_ascii=False)


def decode_address(value: str) -> geo.Address:
    """Inverse of :func:`encode_address`."""
    return geo.Address(**json.loads(value))


def build_row(
    address: geo.Address, location: geo.LocationData, area: float, rooms: float,
    municipalities: gpd.GeoDataFrame,
) -> pd.DataFrame:  # fmt: skip
    """One raw model row: location inputs, area, rooms, admin units, ``re_*`` keys, features."""
    (a_min, a_max), (r_min, r_max) = AREA_RANGE, ROOMS_RANGE
    if not a_min <= area <= a_max:
        raise InputError(f"Wohnfläche muss zwischen {a_min:.0f} und {a_max:.0f} m² liegen.")
    if not r_min <= rooms <= r_max:
        raise InputError(f"Zimmerzahl muss zwischen {r_min:.0f} und {r_max:.0f} liegen.")
    values = {"east": address.east, "north": address.north, "area": area, "rooms": rooms}
    row = pd.DataFrame([values | location.features()]).astype(float)
    row = geo.assign_admin_units(row, municipalities)
    if row["admin_match"].iloc[0] == "none":
        raise InputError("Die Adresse liegt in keiner Schweizer Gemeinde.")
    row = pd.concat([row, geo.nested_group_keys(row)], axis=1)
    return data.add_engineered_features(row)


def prepare(
    res: Resources, address: str | None, payload: dict | None, area: float | None,
    rooms: float | None,
) -> tuple[geo.LocationData, pd.DataFrame]:  # fmt: skip
    """Validate the form and build the model row (looks the address up if not done yet).

    The stored ``payload`` is only used if it belongs to ``address``: a submit can arrive
    before the lookup of a newly selected address has replaced it.
    """
    if not address or area is None or rooms is None:
        raise InputError("Bitte Adresse, Wohnfläche und Zimmer angeben.")
    addr = decode_address(address)
    if payload and payload.get("address") == address:
        values = {k: v for k, v in payload.items() if k != "address"}
        values["dwellings"] = [geo.Dwelling(**d) for d in values.get("dwellings", [])]
        location = geo.LocationData(**values)
    else:
        location = geo.lookup_location(addr, res.session)
    row = build_row(addr, location, float(area), float(rooms), res.units)
    return location, row


def address_options(
    res: Resources, query: str | None, value: str | None
) -> list[dict[str, str]] | object:
    """Suggestions for the address dropdown; keeps the selected address among the options."""
    found = geo.search_addresses(query or "", session=res.session)
    options = [{"label": a.label, "value": encode_address(a)} for a in found]
    if value and value not in {o["value"] for o in options}:
        options.insert(0, {"label": decode_address(value).label, "value": value})
    return options if options or value else no_update


def dwelling_choice(payload: dict | None) -> tuple[list[dict], str]:
    """Options of the GWR dwelling picker and the hint below it."""
    if payload is None:
        return [], ""
    dwellings = [geo.Dwelling(**d) for d in payload.get("dwellings", [])]
    if not dwellings:
        return [], "Im GWR sind für dieses Gebäude keine Wohnungen erfasst: bitte selbst eingeben."
    options = [{"label": d.display, "value": i} for i, d in enumerate(dwellings)]
    return options, f"{len(dwellings)} Wohnungen im GWR. Die Auswahl füllt Fläche und Zimmer aus."


def select_address(res: Resources, value: str | None) -> tuple[dict | None, list[dict], str]:
    """Look up building, location and dwellings of the chosen address (stored with it)."""
    if not value:
        return None, [], ""
    payload = asdict(geo.lookup_location(decode_address(value), res.session)) | {"address": value}
    return (payload, *dwelling_choice(payload))


def prefill(index: int | None, payload: dict | None) -> tuple[object, object]:
    """Area and rooms of the selected GWR dwelling (``no_update`` where unknown)."""
    if index is None or not payload or index >= len(payload.get("dwellings", [])):
        return no_update, no_update
    dwelling = payload["dwellings"][index]
    return dwelling["area"] or no_update, dwelling["rooms"] or no_update


def _block(title: str, *children: object, note: str = "") -> html.Div:
    parts = [html.H3(title), *children] + ([html.P(note, className="note")] if note else [])
    return html.Div(parts, className="block")


def _kpi(label: str, value: str, note: str = "") -> html.Div:
    parts = [html.Div(label, className="kpi-label"), html.Div(value, className="kpi")]
    return html.Div([*parts, html.Div(note, className="note")])


def drivers_block(bundle: RentModelBundle, row: pd.DataFrame) -> html.Div:
    """The three largest SHAP drivers of the point estimate as ± % effects."""
    explanation = models.tree_shap(bundle.point_model, bundle.encoder.transform(row))
    rows = []
    for d in models.top_drivers(explanation, 0, k=3).itertuples():
        css = "up" if d.approx_pct >= 0 else "down"
        effect = html.Span(f"{d.approx_pct:+.0f} %", className=css)
        label = html.Span(DRIVER_LABELS.get(d.feature, d.feature))
        rows.append(html.Div([label, effect], className="driver"))
    return _block("Was die Schätzung am stärksten beeinflusst", *rows, note=NOTES["drivers"])


def attribute_labels(found: dict) -> list[tuple[str, str]]:
    """(German label, evidence) for every positive attribute of ``text.rule_based_extract``."""
    ev = found.get("evidence", {})
    out = [(label, ev.get(key, "")) for key, label in FLAG_LABELS.items() if found.get(key) is True]
    for key, labels in ENUM_LABELS.items():
        if found.get(key) in labels:
            out.append((labels[found[key]], ev.get(key, "")))
    if found.get("renovation_year"):
        out.append((f"Renoviert {found['renovation_year']}", ev.get("renovation_year", "")))
    if found.get("floor") is not None:
        out.append((f"{found['floor']}. Stock", ev.get("floor", "")))
    return out


def attributes_block(description: str | None) -> html.Div | None:
    """Attributes recognised in the description, with the text evidence as tooltip."""
    if not description or not description.strip():
        return None
    from rentml import text  # lazy: the regex tables are only needed with a description

    found = attribute_labels(text.rule_based_extract(description))
    chips = [html.Span(label, title=f"«{ev}»", className="chip") for label, ev in found]
    chips = html.Div(chips or "Keine Merkmale erkannt.", className="chips")
    return _block("Im Beschrieb erkannt", chips, note=NOTES["text"])


def qoli_block(qoli: pd.DataFrame | None, municipality_id: int) -> html.Div | None:
    """Quality-of-life index (0-100, 100 = best) of the municipality, if the table has it."""
    if qoli is None or municipality_id not in qoli.index:
        return None
    row = qoli.loc[municipality_id]
    if pd.isna(row["qoli"]):
        return None
    rank = 100 * qoli["qoli"].rank(pct=True).loc[municipality_id]
    dims = [html.Li(f"{label}: {row[key]:.0f}") for key, label in QOLI_LABELS.items()
            if pd.notna(row.get(key))]  # fmt: skip
    note = f"Besser als {rank:.0f} % der Schweizer Gemeinden (Gemeindewert, nicht das Gebäude)."
    return _block(f"Lebensqualität der Gemeinde: {row['qoli']:.0f} / 100", html.Ul(dims), note=note)


def _location_note(row: pd.DataFrame, location: geo.LocationData) -> html.P:
    r = row.iloc[0]
    text = f"Gemeinde {r['municipality_name']}, Bezirk {r['district_name']}, Kanton {r['canton']}."
    if location.missing:
        missing = ", ".join(DRIVER_LABELS.get(m, m) for m in location.missing)
        text += f" Nicht verfügbar (Schätzung ohne diese Angaben): {missing}."
    return html.P(text, className="note")


def _legal_block(canton: str) -> html.Div:
    links = [html.Li(html.A(t, href=u, target="_blank", rel="noopener")) for t, u in LEGAL_LINKS]
    legal = html.P(LEGAL_TEXT.format(canton=canton), className="note")
    return _block("Rechtliches", legal, html.Ul(links))


def tenant_result(
    res: Resources, address: str | None, payload: dict | None, area: float | None,
    rooms: float | None, rent: float | None, description: str | None,
) -> html.Div:  # fmt: skip
    """Fair-Rent Check: verdict, interval, percentile, drivers, attributes, QoLI, legal hint."""
    try:
        if rent is None or not rent > 0:
            raise InputError("Bitte eine positive Nettomiete angeben.")
        location, row = prepare(res, address, payload, area, rooms)
    except InputError as exc:
        return _notice("Keine Schätzung möglich", str(exc))
    check = intervals.fair_rent_check(res.bundle, row, float(rent))
    pred = check.prediction
    title, explanation = VERDICTS[check.verdict]
    summary = (f"{chf(rent)} – wahrscheinlich zwischen {chf(check.lo)} und {chf(check.hi)} "
               f"({check.coverage:.0%}-Intervall). {explanation}")  # fmt: skip
    kpis = [
        _kpi("Geschätzte Miete", chf(check.expected)),
        _kpi("Marktperzentil", f"ca. {100 * check.percentile:.0f}.", "Anteil günstigerer Angebote"),
        _kpi("Angebotsband", f"{chf(pred['q25'])}–{chf(pred['q75'])}", "mittlere 50 %"),
    ]
    blocks = [
        html.H2(title, className=f"verdict {check.verdict}"),
        html.P(summary),
        dcc.Graph(figure=interval_figure(pred, check.coverage, float(rent)), config=GRAPH),
        html.Div(kpis, className="kpis"),
        _location_note(row, location),
        drivers_block(res.bundle, row),
        attributes_block(description),
        qoli_block(res.qoli, int(row["municipality_id"].iloc[0])),
        _legal_block(str(row["canton"].iloc[0])),
    ]
    return html.Div([b for b in blocks if b is not None], className="card")


def landlord_result(
    res: Resources, address: str | None, payload: dict | None, area: float | None,
    rooms: float | None,
) -> html.Div:  # fmt: skip
    """Listing band (q25-q75), calibrated interval and the what-if over the area."""
    try:
        location, row = prepare(res, address, payload, area, rooms)
    except InputError as exc:
        return _notice("Keine Schätzung möglich", str(exc))
    pred = intervals.predict_frame(res.bundle, row).iloc[0]
    coverage = 1.0 - res.bundle.calibrator.alpha
    summary = (f"Geschätzte Nettomiete {chf(pred['expected'])}; mit {coverage:.0%} "
               f"Wahrscheinlichkeit zwischen {chf(pred['lo'])} und {chf(pred['hi'])}.")  # fmt: skip
    what_if = dcc.Graph(figure=what_if_figure(res.bundle, row, float(area)), config=GRAPH)
    blocks = [
        html.Div("Empfohlenes Angebotsband", className="eyebrow"),
        html.H2(f"{chf(pred['q25'])} – {chf(pred['q75'])}", className="verdict"),
        html.P(summary),
        dcc.Graph(figure=interval_figure(pred, coverage), config=GRAPH),
        _location_note(row, location),
        _block("What-if: Wohnfläche", what_if, note=NOTES["what_if"]),
        drivers_block(res.bundle, row),
    ]
    return html.Div(blocks, className="card")


def _notice(title: str, message: str) -> html.Div:
    return html.Div([html.H2(title), html.P(message)], className="card notice")


def _field(label: str, control: object, hint: object = "") -> html.Div:
    parts = [html.Label(label), control, html.Div(hint, className="note")]
    return html.Div(parts, className="field")


def _number(id_: str, value: float, **limits: float) -> dcc.Input:
    return dcc.Input(id=id_, type="number", value=value, debounce=True, **limits)


def tool_page(prefix: str, res: Resources) -> html.Div:
    """Form (address, GWR dwelling picker, area, rooms, ...) and the result column."""
    if res.bundle is None:
        return _notice("Modell nicht geladen", NOTES["no_model"])
    if res.layers is None:
        return _notice("Kartenebenen fehlen", NOTES["no_layers"])
    search_labels = {**DROPDOWN_LABELS, "no_options_found": "Mindestens 3 Zeichen eingeben"}
    address = dcc.Dropdown(id=f"{prefix}-address", placeholder="Strasse, Hausnummer, Ort …",
                           labels=search_labels, search_order="original")  # fmt: skip
    dwelling = dcc.Dropdown(id=f"{prefix}-dwelling", placeholder="Wohnung aus dem GWR …",
                            labels=DROPDOWN_LABELS)  # fmt: skip
    fields = [
        _field("Adresse", address, "Amtliche Adressen der Schweiz (geo.admin.ch)"),
        dcc.Loading(_field("Wohnung auswählen", dwelling, html.Span(id=f"{prefix}-dwelling-hint"))),
        html.Div([_field("Wohnfläche (m²)", _number(f"{prefix}-area", 75, min=10, max=500, step=1)),
                  _field("Zimmer", _number(f"{prefix}-rooms", 3.5, min=1, max=15, step=0.5))],
                 className="row"),
    ]  # fmt: skip
    if prefix == "tenant":
        text = dcc.Textarea(id="tenant-text", placeholder="z. B. «Renoviert, Balkon und Lift …»")
        fields += [
            _field("Nettomiete pro Monat (CHF)", _number("tenant-rent", 1800, min=1, step=1)),
            _field("Inseratetext (optional)", text, "Wird nur ausgewertet, nicht gespeichert."),
        ]
    eyebrow, title, lead, button = TOOL_TEXT[prefix]
    form = html.Div([html.Div(eyebrow, className="eyebrow"), html.H2(title), html.P(lead), *fields,
                     html.Button(button, id=f"{prefix}-submit", n_clicks=0, className="primary"),
                     dcc.Store(id=f"{prefix}-location")], className="card form")  # fmt: skip
    result = dcc.Loading(html.Div(id=f"{prefix}-result"), type="dot")
    return html.Div([form, html.Div(result, className="result")], className="tool")


def map_page(res: Resources) -> html.Div:
    """Level switch, choropleth and source note (or a notice without map layers)."""
    if res.layers is None:
        return _notice("Kartendaten fehlen", NOTES["no_layers"])
    options = [{"label": label, "value": level} for level, label in LEVELS.items()]
    level = dcc.RadioItems(options, "canton", id="map-level", inline=True, className="segmented")
    graph = dcc.Graph(id="map", config={"scrollZoom": True, "displaylogo": False})
    note = (f"Median der Angebotsmieten (netto) pro m². Grau: weniger als "
            f"{res.meta.get('min_count', MIN_MAP_CELL)} Inserate. Grenzen: "
            f"{res.meta.get('boundaries', 'swissBOUNDARIES3D')}.")  # fmt: skip
    return html.Div([level, graph, html.P(note, className="note")], className="card")


def render_page(pathname: str | None, res: Resources) -> html.Div:
    """Page for a URL path; ``/`` and unknown paths show the Fair-Rent Check."""
    if pathname == "/karte":
        return map_page(res)
    return tool_page("landlord" if pathname == "/vermieter" else "tenant", res)


def _shell() -> html.Div:
    nav = [dcc.Link(name, href=path, className="nav-link") for path, name in PAGES.items()]
    brand = [html.B("RentLens"), html.Span(" Mietpreise der Schweiz, transparent erklärt")]
    header = html.Header([dcc.Link(brand, href="/", className="brand"), html.Nav(nav)])
    footer = html.Footer(FOOTER, className="note")
    return html.Div([dcc.Location(id="url"), header, html.Main(id="page"), footer])


def create_app(res: Resources | None = None) -> Dash:
    """Build the app and wire the callbacks to the functions above.

    Args:
        res: Loaded resources; ``None`` = :func:`load_resources` with the defaults.

    Returns:
        The app (``app.server`` is the Flask server for WSGI hosting).
    """
    res = res or load_resources()
    app = Dash(__name__, assets_folder=str(ASSETS), title="RentLens", update_title=None,
               suppress_callback_exceptions=True)  # fmt: skip
    app.layout = _shell()
    # fmt: off
    cb, once = app.callback, {"prevent_initial_call": True}
    cb(Output("page", "children"), Input("url", "pathname"))(lambda path: render_page(path, res))
    for p in ("tenant", "landlord"):
        cb(Output(f"{p}-address", "options"), Input(f"{p}-address", "search_value"),
           State(f"{p}-address", "value"), **once)(lambda q, v: address_options(res, q, v))
        cb(Output(f"{p}-location", "data"), Output(f"{p}-dwelling", "options"),
           Output(f"{p}-dwelling-hint", "children"), Input(f"{p}-address", "value"),
           **once)(lambda value: select_address(res, value))
        cb(Output(f"{p}-area", "value"), Output(f"{p}-rooms", "value"),
           Input(f"{p}-dwelling", "value"), State(f"{p}-location", "data"), **once)(prefill)
    states = {p: [State(f"{p}-address", "value"), State(f"{p}-location", "data"),
                  State(f"{p}-area", "value"), State(f"{p}-rooms", "value")]
              for p in ("tenant", "landlord")}
    cb(Output("tenant-result", "children"), Input("tenant-submit", "n_clicks"), *states["tenant"],
       State("tenant-rent", "value"), State("tenant-text", "value"),
       **once)(lambda _n, *args: tenant_result(res, *args))
    cb(Output("landlord-result", "children"), Input("landlord-submit", "n_clicks"),
       *states["landlord"], **once)(lambda _n, *args: landlord_result(res, *args))
    if res.layers is not None:
        min_count = int(res.meta.get("min_count", MIN_MAP_CELL))
        cb(Output("map", "figure"), Input("map-level", "value"))(
            lambda level: map_figure(res.layers[level], min_count))
    # fmt: on
    return app


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: start the development server."""
    parser = argparse.ArgumentParser(description="RentLens (DSPRO2 Dash app)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--model", type=Path, default=None, help="model bundle (.joblib)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    create_app(load_resources(args.model)).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
