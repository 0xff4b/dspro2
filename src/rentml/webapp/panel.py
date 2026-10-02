"""Detail panel next to the map: the selected unit (or Switzerland) and its subdivisions."""

import pandas as pd
from dash import dcc, html

from rentml.webapp.figures import Metric, fmt_number, metric_values
from rentml.webapp.state import CHILD_LEVEL, LEVEL_LABELS, LEVEL_PLURALS, UnitIndex, parse_key

LANGUAGES: dict[str, str] = {"de": "Deutsch", "fr": "Französisch", "it": "Italienisch"}
SWITZERLAND_KEY = "switzerland"
MAX_RANKING = 12
DRILL_LABELS: dict[str, str] = {
    "canton": "Bezirke anzeigen",
    "district": "Gemeinden anzeigen",
    "municipality": "Heranzoomen",
}


def unit_link(key: str, label: str, class_name: str = "link-button") -> html.Button:
    """Button that selects a unit (pattern-matching id ``{"type": "unit-link", "key": ...}``)."""
    return html.Button(
        label, id={"type": "unit-link", "key": key}, className=class_name, n_clicks=0
    )


def panel_children(
    index: UnitIndex, meta: dict, view: dict, metric: Metric
) -> list[html.Div | html.Button]:
    """Build the panel for the current view.

    Args:
        index: Unit lookup.
        meta: Bundle metadata (Swiss reference values, ``min_count``).
        view: View state.
        metric: Metric shown on the map.

    Returns:
        Dash components for the panel.
    """
    key = view.get("selected")
    if key not in index:
        return _switzerland(index, meta, view["level"], metric)
    level, _ = parse_key(key)
    row = index.row(key)
    min_count = int(meta.get("min_count", 20))
    parts: list[html.Div | html.Button] = [
        html.Div(LEVEL_LABELS[level], className="eyebrow"),
        html.H2(str(row["name"]), className="panel-title"),
        _breadcrumb(index, key),
        _headline(row, metric, meta),
        _stats_grid(row, min_count),
    ]
    if int(row["n_objects"]) < min_count:
        parts.append(
            html.P(
                f"Weniger als {min_count} Inserate: Medianwerte werden hier nicht "
                "veröffentlicht, damit keine Einzelmieten erkennbar sind.",
                className="note",
            )
        )
    drill_id = {"type": "drill", "key": key}
    parts.append(html.Button(DRILL_LABELS[level], id=drill_id, className="primary-button"))
    parts.append(dcc.Link("Miete prüfen (Mietende) →", href="/mietcheck", className="cta-link"))
    parts.append(dcc.Link("Preisband für Vermietende →", href="/vermieter", className="cta-link"))
    children = index.children(key)
    if len(children):
        title = f"{LEVEL_PLURALS[CHILD_LEVEL[level]]} nach {metric.label}"
        parts.append(_ranking(children, metric, title, min_count))
    return parts


def _switzerland(
    index: UnitIndex, meta: dict, level: str, metric: Metric
) -> list[html.Div | html.Button]:
    swiss = meta.get("swiss", {})
    min_count = int(meta.get("min_count", 20))
    counts = {lv: len(index.level_table(lv)) for lv in LEVEL_LABELS}
    parents = f"{counts['canton']} Kantone · {counts['district']} Bezirke"
    grid = [
        _stat("Miete pro m²", _fmt(swiss.get("chf_per_m2"), 1, "CHF"), _iqr(swiss)),
        _stat("Median Nettomiete", _fmt(swiss.get("rent"), 0, "CHF"), "pro Monat"),
        _stat("Inserate (Objekte)", _fmt(meta.get("n_objects"), 0), "nach Dublettenprüfung"),
        _stat("Gemeinden", fmt_number(counts["municipality"]), parents),
    ]
    table = index.level_table(level)
    return [
        html.Div("Übersicht", className="eyebrow"),
        html.H2("Schweiz", className="panel-title"),
        html.P(
            "Klicke auf die Karte oder suche eine Gemeinde, einen Bezirk oder einen Kanton.",
            className="lead",
        ),
        html.Div(grid, className="stats-grid"),
        _ranking(table, metric, f"{LEVEL_PLURALS[level]} nach {metric.label}", min_count),
    ]


def _breadcrumb(index: UnitIndex, key: str) -> html.Nav:
    items: list[html.Button | html.Span] = [unit_link(SWITZERLAND_KEY, "Schweiz")]
    for ancestor in index.ancestors(key):
        items += [
            html.Span("›", className="crumb-sep"),
            unit_link(ancestor, _crumb(index, ancestor)),
        ]
    return html.Nav(items, className="breadcrumb")


def _crumb(index: UnitIndex, key: str) -> str:
    level, _ = parse_key(key)
    return f"{LEVEL_LABELS[level]} {index.row(key)['name']}"


def _headline(row: pd.Series, metric: Metric, meta: dict) -> html.Div:
    value = row.get(metric.column)
    reference = meta.get("swiss", {}).get(metric.column)
    delta = None
    if reference and value is not None and pd.notna(value):
        change = (float(value) / float(reference) - 1.0) * 100.0
        sign = "+" if change >= 0 else "−"
        delta = html.Span(
            f"{sign}{abs(change):.0f} % vs. Schweiz",
            className="delta up" if change >= 0 else "delta down",
        )
    return html.Div(
        [
            html.Div(metric.label, className="headline-label"),
            html.Div([html.Span(metric.format(value), className="headline-value"), delta]),
        ],
        className="headline",
    )


def _stats_grid(row: pd.Series, min_count: int) -> html.Div:
    enough = int(row["n_objects"]) >= min_count
    lang = LANGUAGES.get(str(row.get("lang_region")), "–")
    stats = [
        _stat(
            "Miete pro m²",
            _fmt(row.get("chf_per_m2"), 1, "CHF"),
            _iqr(row) if enough else "",
        ),
        _stat("Median Nettomiete", _fmt(row.get("rent"), 0, "CHF"), "pro Monat"),
        _stat("Inserate (Objekte)", fmt_number(int(row["n_objects"])), "nach Dublettenprüfung"),
        _stat("Einwohner", _fmt(row.get("population"), 0), "swissBOUNDARIES3D"),
        _stat("Landfläche", _fmt(row.get("area_km2"), 1, "km²"), "ohne Seen"),
        _stat("Dichte", _fmt(row.get("density"), 0, "/km²"), f"Sprachregion: {lang}"),
    ]
    return html.Div(stats, className="stats-grid")


def _stat(label: str, value: str, note: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="stat-label"),
            html.Div(value, className="stat-value"),
            html.Div(note, className="stat-note"),
        ],
        className="stat",
    )


def _iqr(values: pd.Series | dict) -> str:
    lo, hi = values.get("chf_per_m2_p25"), values.get("chf_per_m2_p75")
    if lo is None or hi is None or pd.isna(lo) or pd.isna(hi):
        return ""
    return f"mittlere 50 %: {fmt_number(float(lo), 1)}–{fmt_number(float(hi), 1)}"


def _fmt(value: object, decimals: int, unit: str = "") -> str:
    if value is None or pd.isna(value):
        return "–"
    return f"{fmt_number(float(value), decimals)} {unit}".strip()


def _ranking(table: pd.DataFrame, metric: Metric, title: str, min_count: int) -> html.Div:
    values = metric_values(table, metric)
    ranked = values.dropna().sort_values(ascending=False)
    hidden = int(values.isna().sum())
    top = ranked.head(MAX_RANKING)
    vmax = float(top.max()) if len(top) else 1.0
    rows = [
        html.Div(
            [
                unit_link(str(key), str(table.at[key, "name"]), "rank-name"),
                html.Span(metric.format(table.at[key, metric.column]), className="rank-value"),
                html.Div(
                    html.Div(className="rank-fill", style={"width": f"{100 * v / vmax:.1f}%"}),
                    className="rank-bar",
                ),
            ],
            className="rank-row",
        )
        for key, v in top.items()
    ]
    notes = []
    if len(ranked) > MAX_RANKING:
        notes.append(f"Top {MAX_RANKING} von {len(ranked)}")
    if hidden:
        notes.append(f"{hidden} ohne Wert ({metric.missing.format(min_count=min_count)})")
    return html.Div(
        [html.H3(title, className="section-title"), *rows]
        + ([html.Div(" · ".join(notes), className="stat-note")] if notes else []),
        className="ranking",
    )
