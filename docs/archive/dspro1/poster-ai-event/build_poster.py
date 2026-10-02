"""Build the A0 AI-event poster as a vector SVG (841 x 1189 mm).

All coordinates are millimetres (viewBox 0 0 841 1189). Charts are drawn natively
so that every label stays large enough for print. Numbers come from the final
report and from src/external-sources/output_csv/.

Usage:  .venv/bin/python docs/dspro1/poster-ai-event/build_poster.py
"""
from __future__ import annotations

import base64
import csv
import re
from html import escape
from pathlib import Path

from PIL import Image, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASSETS = ROOT / "src" / "assets"
DATA = ROOT / "src" / "external-sources" / "output_csv"
OUT = HERE / "DISPRO1_PosterAIEvent_Team8_A0.svg"
# Variants differ only in the prototype screenshot (section 06).
VARIANTS = {
    "": {"shot": "prototype_demo.jpg", "caption": None},
    "_V2_Adresse": {"shot": "prototype_address.jpg",
                    "caption": "Live example: Baselstrasse 30, 6003 Luzern · 2nd floor, 50 m², 2 rooms"},
}
BLEED = 3                   # mm per side for the print-shop version
AUTHORS = [  # photo: optional square image in assets/ (jpg/png); initials are shown until it exists
    ("Elias Martinelli", "https://www.linkedin.com/in/elias-martinelli-b05ba1194/", "photo_elias"),
    ("Timo Schlumpf", "https://www.linkedin.com/in/tschlumpf/", "photo_timo"),
    ("Dr. Elena Nazarenko", "https://www.linkedin.com/in/lena-nazarenko/", "photo_elena", "Supervisor"),
]
APP_URL = "https://dspro1-streamlit.thankfulpond-a9641a83.switzerlandnorth.azurecontainerapps.io"

W, H = 841, 1189
M = 30                      # outer margin
GAP = 20                    # column gutter
COLW = (W - 2 * M - 2 * GAP) / 3
COLX = [M + i * (COLW + GAP) for i in range(3)]

# Website palette (src/plot_style.py, src/assets/app.css)
BLUE, BLUE_L, BLUE_XL = "#0077C8", "#66B5E9", "#D6E7F5"
INK, NAVY, MUTED = "#111A28", "#192B3A", "#4C6280"
BORDER, PANEL, HIGHLIGHT = "#C8DCEC", "#F3F8FC", "#E7F4EF"
GREEN, ORANGE, RED = "#07856D", "#C88126", "#D0384B"

SANS = "Arial, Helvetica, 'Liberation Sans', sans-serif"
HEAD = "'Barlow Condensed', 'Arial Narrow', sans-serif"

_FONTS = {
    ("sans", 400): "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ("sans", 700): "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ("head", 700): str(ASSETS / "BarlowCondensed-Bold.ttf"),
}
_font_cache: dict = {}


def text_width(s: str, size: float, family: str = "sans", weight: int = 400, ls: float = 0) -> float:
    """Width in mm, measured with metric-compatible TTFs (Liberation = Arial)."""
    key = (family, 700 if family == "head" else weight)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(_FONTS[key], 1000)
    return _font_cache[key].getlength(s) / 1000 * size + ls * size * max(len(s) - 1, 0)


def wrap(s: str, width: float, size: float, weight: int = 400) -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        trial = f"{cur} {word}".strip()
        if cur and text_width(trial, size, weight=weight) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


out: list[str] = []


def add(s: str) -> None:
    out.append(s)


def text(x, y, s, size, fill=INK, weight=400, family=SANS, anchor="start", ls=0, extra=""):
    ls_attr = f' letter-spacing="{ls * size:.3f}"' if ls else ""
    add(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{ls_attr} {extra}>{escape(s)}</text>')


def head(x, y, s, size, fill=INK, anchor="start", ls=0.005):
    text(x, y, s, size, fill, 700, HEAD, anchor, ls)


def para(x, y, s, width, size=8, fill=MUTED, lh=1.42, weight=400) -> float:
    """Wrapped paragraph; y is the first baseline. Returns the baseline after the last line."""
    lines = wrap(s, width, size, weight)
    for i, line in enumerate(lines):
        text(x, y + i * size * lh, line, size, fill, weight)
    return y + len(lines) * size * lh


def rect(x, y, w, h, fill="none", stroke="none", sw=0.5, r=2.5, extra=""):
    add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def line(x1, y1, x2, y2, stroke=BORDER, sw=0.5, extra=""):
    add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def card(x, y, w, h, accent=BLUE, fill=PANEL):
    rect(x, y, w, h, fill, BORDER, 0.5, 2)
    add(f'<path d="M{x + 2:.2f},{y:.2f} h{w - 4:.2f} a2,2 0 0 1 2,2 v0.9 h{-w:.2f} v-0.9 a2,2 0 0 1 2,-2 z" fill="{accent}"/>')


def section(x, y, w, num, title, sub=None) -> float:
    """Numbered heading in the website style. Returns y below the block."""
    head(x, y, num, 17, BLUE)
    tx = x + text_width(num, 17, "head") + 5
    head(tx, y, title.upper(), 17)
    lx = tx + text_width(title.upper(), 17, "head", ls=0.005) + 6
    if lx < x + w:
        line(lx, y - 5.5, x + w, y - 5.5, BORDER, 0.6)
    if sub:
        return para(x, y + 12, sub, w, 8.2, MUTED) + 2
    return y + 10


# ---------------------------------------------------------------- icons (24x24 line icons)
ICONS = {
    "pin": '<path d="M12 21s-6.5-6.1-6.5-11.2a6.5 6.5 0 0 1 13 0C18.5 14.9 12 21 12 21z"/><circle cx="12" cy="9.8" r="2.4"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "chart": '<path d="M4 20h16M7 20v-6M12 20V8M17 20v-10"/>',
    "calendar": '<rect x="4" y="5" width="16" height="15" rx="2"/><path d="M4 10h16M8 3v4M16 3v4"/>',
    "city": '<path d="M3 20h18M5 20V9l5-3v14M10 20V4l7 4v12M13 11h1M13 14h1M7 12h1M7 15h1"/>',
    "text": '<path d="M5 4h14v16H5zM8 8h8M8 12h8M8 16h5"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/>',
    "image": '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="M21 16l-5-5-8 8"/>',
    "layers": '<path d="M12 4l9 5-9 5-9-5z"/><path d="M3 14l9 5 9-5"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "rocket": '<path d="M14 4c3 0 6 3 6 6l-8 8-6-6z"/><path d="M6 12l-3 1 2-4 3-1M12 18l-1 3 4-2 1-3"/><circle cx="15" cy="9" r="1.5"/>',
}


def icon(name, x, y, size=12, color=INK, bg=None):
    if bg:
        rect(x - size * 0.35, y - size * 0.35, size * 1.7, size * 1.7, bg, "none", 0, 2.5)
    s = size / 24
    add(f'<g transform="translate({x:.2f},{y:.2f}) scale({s:.4f})" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</g>')


def qr_path(data: str, x: float, y: float, size: float, quiet: int = 2) -> str:
    """QR code as one vector path (quiet zone included in size)."""
    import segno
    matrix = [list(r) for r in segno.make(data, error="m").matrix]
    n = len(matrix)
    cell = size / (n + 2 * quiet)
    return "".join(f"M{x + (quiet + c) * cell:.3f},{y + (quiet + r) * cell:.3f}h{cell:.3f}v{cell:.3f}h{-cell:.3f}z"
                   for r in range(n) for c in range(n) if matrix[r][c])


def author_card(x, y, w, h, name, url, photo_stem, role=None):
    rect(x, y, w, h, PANEL, BORDER, 0.4, 2.5)
    r = h / 2 - 4
    cx, cy = x + 4 + r, y + h / 2
    photo = next((HERE / "assets" / f"{photo_stem}{ext}" for ext in (".jpg", ".jpeg", ".png")
                  if (HERE / "assets" / f"{photo_stem}{ext}").exists()), None)
    cid = f"clip_{photo_stem}"
    if photo:
        mime = "image/png" if photo.suffix == ".png" else "image/jpeg"
        b64 = base64.b64encode(photo.read_bytes()).decode()
        add(f'<clipPath id="{cid}"><circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/></clipPath>')
        add(f'<image x="{cx - r:.2f}" y="{cy - r:.2f}" width="{2 * r:.2f}" height="{2 * r:.2f}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})" xlink:href="data:{mime};base64,{b64}"/>')
    else:
        add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{BLUE_XL}"/>')
        initials = "".join(part[0] for part in name.replace("Dr. ", "").split())
        head(cx, cy + 3.6, initials, 10, BLUE, "middle")
    add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="none" stroke="{BORDER}" stroke-width="0.6"/>')
    qs = h - 6
    qx = x + w - qs - 3
    rect(qx, y + 3, qs, qs, "#FFFFFF", r=1)
    add(f'<path d="{qr_path(url, qx, y + 3, qs)}" fill="{INK}" shape-rendering="crispEdges"/>')
    tx = cx + r + 4
    first, last = name.rsplit(" ", 1)
    text(tx, y + 7.6, (role or "Author").upper(), 4.8, BLUE, 700, ls=0.1)
    text(tx, y + 14.6, first, 7.2, INK, 700)
    text(tx, y + 21.4, last, 7.2, INK, 700)
    rect(tx, y + 23.4, 4.6, 4.6, BLUE, r=0.9)
    text(tx + 2.3, y + 26.9, "in", 3.8, "#FFFFFF", 700, anchor="middle")
    text(tx + 6.2, y + 27, "LinkedIn", 5, MUTED, 700)
    add(f'<a xlink:href="{escape(url)}" href="{escape(url)}"><rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" '
        f'height="{h:.2f}" fill="transparent"/></a>')


def fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", "'")


# ================================================================= build
def build(bleed: float = 0, variant: str = "") -> str:
    """bleed > 0 widens the page by that margin; backgrounds run into it, content stays put."""
    out.clear()
    bl = bleed
    font_b64 = base64.b64encode((ASSETS / "BarlowCondensed-Bold.ttf").read_bytes()).decode()
    logo = (ASSETS / "hslu-logo.svg").read_text(encoding="utf-8")
    logo_paths = "".join(re.findall(r"<path[\s\S]*?/>", logo))

    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W + 2 * bl}mm" height="{H + 2 * bl}mm" viewBox="{-bl} {-bl} {W + 2 * bl} {H + 2 * bl}">')
    add("<title>Predicting Apartment Rental Prices in Switzerland – DSPRO1 Team 8 poster (A0)</title>")
    add(f'<defs><style>@font-face{{font-family:"Barlow Condensed";font-weight:700;'
        f'src:url(data:font/ttf;base64,{font_b64}) format("truetype");}}</style></defs>')
    ob = bl + 2  # overshoot: backgrounds run past the page edge, no anti-aliased seam
    rect(-ob, -ob, W + 2 * ob, H + 2 * ob, "#FFFFFF", r=0)

    # ------------------------------------------------------------ header
    add(f'<svg x="{M}" y="{M}" width="175" height="{175 * 32.2 / 220:.2f}" viewBox="0 0 220 32.2">'
        f'<g fill="#000">{logo_paths}</g></svg>')
    rx = W - M
    text(rx, M + 6, "AI EVENT · LIVE DEMO", 7, BLUE, 700, anchor="end", ls=0.12)
    cw, chh, cg = 108, 30, 6
    for i, person in enumerate(AUTHORS):
        author_card(rx - (len(AUTHORS) - i) * cw - (len(AUTHORS) - 1 - i) * cg, M + 11, cw, chh, *person)
    text(rx, M + chh + 21, "DSPRO1 · Team 8 · Spring 2026", 7, MUTED, anchor="end")

    y = 108
    kicker = ["DATA", "MACHINE LEARNING", "REAL ESTATE", "SWISS OPEN DATA"]
    kx = M
    for i, k in enumerate(kicker):
        text(kx, y, k, 7.2, BLUE, 700, ls=0.12)
        kx += text_width(k, 7.2, weight=700, ls=0.12) + 5
        if i < len(kicker) - 1:
            text(kx, y, "/", 7.2, BLUE_L, 700)
            kx += 8
    t1, t2 = "PREDICTING APARTMENT", "RENTAL PRICES IN SWITZERLAND"
    tsize = (W - 2 * M) / (text_width(t2, 1, "head", ls=-0.015))
    tsize = min(tsize, 80)
    head(M - 1.5, y + 14 + tsize * 0.72, t1, round(tsize, 2), ls=-0.015)
    head(M - 1.5, y + 14 + tsize * 1.62, t2, round(tsize, 2), ls=-0.015)
    y = y + 14 + tsize * 1.62 + 22
    text(M, y, "A supervised machine-learning approach for the Swiss cold-rent market", 13.5, MUTED)
    y += 13
    add(f'<rect x="{M}" y="{y:.2f}" width="{W - 2 * M}" height="1.2" fill="{BLUE}"/>')
    y += 17
    abstract = ("We built a reproducible pipeline that scrapes Swiss rental listings, enriches them with "
                "official building (GWR) and swisstopo location data, and predicts the monthly cold rent "
                "of a single apartment. From 9'994 scraped listings, 4'536 fully enriched apartments were "
                "used for modelling. GradientBoosting with the ALL+geo feature set is the final model: "
                "not the lowest error, but the best balance of accuracy and generalisation.")
    y = para(M, y, abstract, W - 2 * M - 10, 9.6, INK, 1.42)
    y += 10

    # ------------------------------------------------------------ row 1
    r1 = y + 14
    x, w = COLX[0], COLW
    cy = section(x, r1, w, "01", "The problem", "Can a single apartment's rent be judged fairly?")
    rect(x, cy, w, 25, NAVY, r=2.5)
    text(x + w / 2, cy + 16, "2-room flat · City of Lucerne", 11.5, "#FFFFFF", 700, anchor="middle")
    cy += 30
    rect(x, cy, w, 34, "#FFF5F6", RED, 0.8, 2.5)
    head(x + w / 2, cy + 18, "WHAT SHOULD IT COST?", 16, RED, "middle")
    text(x + w / 2, cy + 28, "Asking rents vary widely for similar flats", 7.4, RED, anchor="middle")
    cy += 44
    items = [
        ("pin", "Location dependency", "The same flat costs far more in the city of Lucerne than in rural regions."),
        ("sliders", "Many interacting features", "Area, rooms, building age and local market interact non-linearly."),
        ("chart", "No apartment-level benchmark", "Official market indicators do not price one specific flat."),
    ]
    for ic, t, d in items:
        icon(ic, x + 4, cy + 1, 13, INK, PANEL)
        text(x + 30, cy + 5, t, 9, INK, 700)
        yy = para(x + 30, cy + 14.5, d, w - 32, 7.8, MUTED, 1.35)
        cy = max(yy, cy + 24) + 7
    end1 = cy

    x = COLX[1]
    cy = section(x, r1, w, "02", "From data to model", "From 25'002 listing URLs to 4'536 enriched apartments.")
    funnel = [
        (25002, "URLs identified", "rentumo.ch, one scrape on 13 Apr 2026", BLUE_XL, INK),
        (9994, "listings scraped", "detail pages successfully extracted", "#A9D3F0", INK),
        (9582, "with core fields", "address · area · rooms · cold rent", BLUE_L, INK),
        (4536, "fully enriched", "GWR building + swisstopo location data", BLUE, "#FFFFFF"),
    ]
    for i, (n, lab, note, fill, fg) in enumerate(funnel):
        bw = w * n / 25002
        rect(x, cy, w, 33, PANEL, BORDER, 0.4, 2)
        head(x + 5, cy + 17.5, fmt(n), 18)
        lx = x + 5 + text_width("25'002", 18, "head") + 6
        text(lx, cy + 10.5, lab, 8.6, INK, 700)
        text(lx, cy + 19, note, 7, MUTED)
        rect(x + 5, cy + 25, w - 10, 3.2, "#FFFFFF", BORDER, 0.3, 1.2)
        rect(x + 5, cy + 25, (w - 10) * n / 25002, 3.2, BLUE if i == 3 else BLUE_L, r=1.2)
        cy += 37
    cy += 4
    text(x, cy + 3, "ENRICHMENT STACK", 7, INK, 700, ls=0.08)
    cy += 8
    chips = [("rentumo.ch", "#FDECEE", RED), ("GeoAdmin", "#E6F1FA", BLUE),
             ("GWR / BFS", HIGHLIGHT, GREEN), ("swisstopo", "#FBF1E3", ORANGE)]
    chx = x
    for lab, bg, fg in chips:
        cw = text_width(lab, 7.6, weight=700) + 10
        rect(chx, cy, cw, 12, bg, r=2)
        text(chx + cw / 2, cy + 8.3, lab, 7.6, fg, 700, anchor="middle")
        chx += cw + 4
    cy += 22
    text(x, cy, "MODEL INPUT · 11 PREDICTORS + GEO CLUSTER", 7, INK, 700, ls=0.08)
    cy = para(x, cy + 9, "Apartment: area, rooms  ·  Building: year built, dwellings, land area  ·  "
              "Location: coordinates, elevation, population, public-transport score, solar class",
              w, 7.4, MUTED, 1.38)
    end2 = cy

    x = COLX[2]
    cy = section(x, r1, w, "03", "Results", "Every tree model clearly beats the linear benchmark.")
    rect(x, cy, w, 30, HIGHLIGHT, "#BFE3D5", 0.5, 2)
    add(f'<rect x="{x}" y="{cy}" width="1.6" height="30" fill="{GREEN}"/>')
    text(x + 7, cy + 9.5, "FINAL MODEL", 6.8, GREEN, 700, ls=0.1)
    head(x + 7, cy + 24.5, "GRADIENTBOOSTING · ALL+GEO", 16)
    cy += 35
    kw = (w - 8) / 3
    kpis = [("EVAL RMSE", "CHF 425", BLUE), ("EVAL R²", "0.710", BLUE), ("TRAIN–EVAL GAP", "CHF 60", GREEN)]
    for i, (lab, val, acc) in enumerate(kpis):
        kx = x + i * (kw + 4)
        card(kx, cy, kw, 30, acc)
        text(kx + 5, cy + 10, lab, 6.2, MUTED, 700, ls=0.06)
        head(kx + 5, cy + 24.5, val, 16)
    cy += 40
    text(x, cy, "RMSE train vs. eval (CHF, engineered features)", 7.6, INK, 700)
    cy += 5
    lg = [("Train", BLUE_L), ("Eval", BLUE)]
    lx = x
    for lab, col in lg:
        rect(lx, cy + 1.5, 5, 5, col, r=1)
        text(lx + 7, cy + 6, lab, 6.8, MUTED)
        lx += 7 + text_width(lab, 6.8) + 8
    cy += 11
    models = [("LightGBM", 150, 399), ("XGBoost", 122, 421), ("GradientBoosting", 365, 425),
              ("RandomForest", 199, 425), ("Ridge (scaled)", 539, 560), ("Dummy (median)", 737, 799)]
    labw, maxv = 54, 850
    plotw = w - labw - 16
    rowh = 15.5
    for i, (name, tr, ev) in enumerate(models):
        ry = cy + i * rowh
        final = name == "GradientBoosting"
        if final:
            rect(x - 1, ry - 1, w + 1, rowh, HIGHLIGHT, r=1.5)
        text(x + labw - 2, ry + 8.6, name, 7.4, INK if final else MUTED, 700 if final else 400, anchor="end")
        bx = x + labw + 1
        rect(bx, ry + 1.5, plotw * tr / maxv, 5.4, BLUE_L, r=1)
        rect(bx, ry + 7.7, plotw * ev / maxv, 5.4, BLUE, r=1)
        text(bx + plotw * ev / maxv + 2, ry + 12.4, fmt(ev), 6.8, INK, 700)
    ax = x + labw + 1
    line(ax, cy - 1, ax, cy + len(models) * rowh - 1, MUTED, 0.4)
    cy += len(models) * rowh + 5
    cy = para(x, cy + 2, "LightGBM has the lowest eval RMSE (399), but memorises the training data "
              "(gap 249). Bootstrap intervals of the top tree models overlap, so the smaller gap decides.",
              w, 7, MUTED, 1.38)
    end3 = cy

    # ------------------------------------------------------------ row 2
    r2 = max(end1, end2, end3) + 44
    line(M, r2 - 28, W - M, r2 - 28, BORDER, 0.4)

    x = COLX[0]
    cy = section(x, r2, w, "04", "What drives rent?", "Living area is the strongest single predictor; location comes next.")
    feats = [("Living area", .41, .54), ("East (LV95)", .17, .24), ("Public transport", .12, .07),
             ("North (LV95)", .07, .07), ("Elevation", .04, .01), ("Area per room", .03, .01),
             ("Land area", .03, .00), ("Year built", .02, .02)]
    lg = [("RF impurity", BLUE_L), ("Permutation", BLUE)]
    lx = x
    for lab, col in lg:
        rect(lx, cy + 1.5, 5, 5, col, r=1)
        text(lx + 7, cy + 6, lab, 6.8, MUTED)
        lx += 7 + text_width(lab, 6.8) + 8
    text(x + w, cy + 6, "share of importance", 6.4, MUTED, anchor="end")
    cy += 12
    labw, maxv = 48, .6
    plotw = w - labw - 14
    rowh = 16
    for i, (name, rf, pm) in enumerate(feats):
        ry = cy + i * rowh
        text(x + labw - 2, ry + 8.8, name, 7.4, INK if i < 2 else MUTED, 700 if i < 2 else 400, anchor="end")
        bx = x + labw + 1
        rect(bx, ry + 1.5, max(plotw * rf / maxv, 0.8), 5.6, BLUE_L, r=1)
        rect(bx, ry + 8, max(plotw * pm / maxv, 0.8), 5.6, BLUE, r=1)
        if i < 2:
            text(bx + plotw * pm / maxv + 2, ry + 12.8, f"{pm:.2f}", 6.8, INK, 700)
    line(x + labw + 1, cy - 1, x + labw + 1, cy + len(feats) * rowh - 1, MUTED, 0.4)
    cy += len(feats) * rowh + 6
    rect(x, cy, w, 29, PANEL, BORDER, 0.4, 2)
    add(f'<rect x="{x}" y="{cy}" width="1.6" height="29" fill="{BLUE}"/>')
    para(x + 7, cy + 9.5, "Size and coordinates dominate every method. Public-transport access "
         "and elevation add local-market context.", w - 12, 7.4, INK, 1.4)
    end1 = cy + 29

    # map --------------------------------------------------------
    x = COLX[1]
    cy = section(x, r2, w, "05", "Where is rent high?", "4'536 modelling apartments, coloured by monthly cold rent.")
    rows = list(csv.DictReader((DATA / "model.csv").open(encoding="utf-8")))
    pts = sorted(((float(r["lv95_east"]), float(r["lv95_north"]), float(r["price_cold"])) for r in rows),
                 key=lambda p: p[2])
    e0, e1, n0, n1 = 2485000, 2834000, 1075000, 1296000
    mw = w
    mh = mw * (n1 - n0) / (e1 - e0)
    rect(x, cy, mw, mh, PANEL, BORDER, 0.4, 2)
    bins = [(0, 1250, "#B9DDF4"), (1250, 1550, "#7FC0EA"), (1550, 1850, "#3A98D8"),
            (1850, 2300, "#0B67AE"), (2300, 1e9, ORANGE)]  # top band = "expensive" colour of section 07

    def proj(e, n):
        return x + (e - e0) / (e1 - e0) * mw, cy + (n1 - n) / (n1 - n0) * mh

    for lo, hi, col in bins:
        dots = [proj(e, n) for e, n, p in pts if lo <= p < hi]
        add(f'<g fill="{col}" fill-opacity="0.9" stroke="#FFFFFF" stroke-width="0.12">' +
            "".join(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.15"/>' for px, py in dots) + "</g>")
    cities = [("Zürich", 2683000, 1248000, "end"), ("Genève", 2500000, 1118000, "start"),
              ("Basel", 2611500, 1267000, "end"), ("Bern", 2600000, 1199700, "end"),
              ("Lausanne", 2538000, 1152500, "start"), ("Luzern", 2666000, 1211500, "start"),
              ("Lugano", 2717500, 1096000, "start"), ("St. Gallen", 2746000, 1254500, "start")]
    for name, e, n, anc in cities:
        px, py = proj(e, n)
        add(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.3" fill="none" stroke="{INK}" stroke-width="0.5"/>')
        dx = -3 if anc == "end" else 3
        text(px + dx, py - 2.2, name, 6.6, INK, 700, anchor=anc,
             extra='stroke="#FFFFFF" stroke-width="1.6" stroke-linejoin="round" paint-order="stroke"')
    cy += mh + 6
    text(x, cy + 3, "Cold rent, CHF / month", 6.8, MUTED, 700)
    cy += 8
    labels = ["< 1'250", "1'250–1'550", "1'550–1'850", "1'850–2'300", "≥ 2'300"]
    lx0 = x
    sw_ = w / 5
    for i, (lo, hi, col) in enumerate(bins):
        rect(lx0 + i * sw_, cy - 1, sw_ - 1.2, 5.5, col, r=1)
        text(lx0 + i * sw_ + (sw_ - 1.2) / 2, cy + 12, labels[i], 6.2, MUTED, anchor="middle")
    cy += 22
    cy = para(x, cy + 2, "Zürich, Geneva, Basel and Lausanne form dense, high-rent clusters; "
              "urban markets are over-represented in the data.", w, 7.4, MUTED, 1.38)
    end2 = cy

    # prototype --------------------------------------------------
    x = COLX[2]
    cy = section(x, r2, w, "06", "Try the prototype", "Swiss address → official GWR dwelling → live rent estimate.")
    shot = HERE / "assets" / VARIANTS[variant]["shot"]
    with Image.open(shot) as im:
        iw, ih = im.size
    fh = 9
    sh = (w - 2) * ih / iw
    rect(x, cy, w, fh + sh + 1, "#FFFFFF", BORDER, 0.5, 2.5)
    add(f'<path d="M{x + 2.5},{cy} h{w - 5} a2.5,2.5 0 0 1 2.5,2.5 v{fh - 2.5} h{-w} v{-(fh - 2.5)} a2.5,2.5 0 0 1 2.5,-2.5z" fill="{PANEL}"/>')
    for i, c in enumerate(["#E5A1A8", "#EBCB8F", "#9FD1B8"]):
        add(f'<circle cx="{x + 6 + i * 5}" cy="{cy + fh / 2}" r="1.5" fill="{c}"/>')
    rect(x + 24, cy + 2, w - 30, fh - 4, "#FFFFFF", BORDER, 0.3, 2.2)
    text(x + 27, cy + 6.2, "dspro1-streamlit · switzerlandnorth.azurecontainerapps.io", 3.8, MUTED)
    b64 = base64.b64encode(shot.read_bytes()).decode()
    add(f'<image x="{x + 1}" y="{cy + fh}" width="{w - 2:.2f}" height="{sh:.2f}" preserveAspectRatio="xMidYMid meet" '
        f'xlink:href="data:image/jpeg;base64,{b64}"/>')
    cy += fh + sh + 1 + 6
    caption = VARIANTS[variant]["caption"]
    steps = [] if caption else [("1", "Search a Swiss address", "GeoAdmin autocomplete"),
             ("2", "Pick the dwelling", "GWR pre-fills building data"),
             ("3", "Get the estimate", "CHF / month, CHF / m², error band")]
    for i, (n, t, d) in enumerate(steps):
        sy = cy + i * 17
        add(f'<circle cx="{x + 5}" cy="{sy + 5}" r="4.6" fill="{BLUE}"/>')
        text(x + 5, sy + 7.6, n, 7, "#FFFFFF", 700, anchor="middle")
        text(x + 14, sy + 4.6, t, 8, INK, 700)
        text(x + 14, sy + 12.2, d, 6.8, MUTED)
    if caption:
        cy = para(x, cy + 4, caption, w, 7, MUTED, 1.38) + 1
    else:
        cy += 3 * 17 + 4
    # QR

    qs = 54
    rect(x, cy, w, qs + 10, NAVY, r=2.5)
    qx, qy = x + 5, cy + 5
    rect(qx, qy, qs, qs, "#FFFFFF", r=1.5)
    add(f'<path d="{qr_path(APP_URL, qx, qy, qs, quiet=3)}" fill="{INK}" shape-rendering="crispEdges"/>')
    tx = qx + qs + 8
    head(tx, cy + 19, "SCAN TO EXPLORE", 16, "#FFFFFF")
    para(tx, cy + 30, "Estimate a monthly cold rent for any Swiss address.", w - (tx - x) - 5, 7.4, "#D0DCE7", 1.38)
    text(tx, cy + qs + 3, "Demo model: LightGBM + KNN, eval RMSE 393", 5.6, "#9FB3C8")
    end3 = cy + qs + 10

    # ------------------------------------------------------------ row 3
    r3 = max(end1, end2, end3) + 44
    line(M, r3 - 28, W - M, r3 - 28, BORDER, 0.4)

    x = COLX[0]
    cy = section(x, r3, w, "07", "Where does it struggle?", "The most expensive quartile has about twice the error.")
    bands = [("Cheap", 216, 150), ("Medium low", 176, 143), ("Medium high", 221, 158), ("Expensive", 445, 312)]
    lg = [("Mean abs. error", BLUE), ("Median", BLUE_L)]
    lx = x
    for lab, col in lg:
        rect(lx, cy + 1.5, 5, 5, col, r=1)
        text(lx + 7, cy + 6, lab, 6.8, MUTED)
        lx += 7 + text_width(lab, 6.8) + 8
    text(x + w, cy + 6, "CHF, eval set", 6.4, MUTED, anchor="end")
    cy += 14
    ch = 62
    base = cy + ch
    gw = w / 4
    for i, (b, mean, med) in enumerate(bands):
        gx = x + i * gw + gw * 0.16
        bw_ = gw * 0.32
        exp = b == "Expensive"
        for j, (v, col) in enumerate([(mean, ORANGE if exp else BLUE), (med, "#E8B97A" if exp else BLUE_L)]):
            bh = ch * v / 480
            bx = gx + j * (bw_ + 1)
            add(f'<path d="M{bx:.2f},{base} v{-(bh - 1.2):.2f} a1.2,1.2 0 0 1 1.2,-1.2 h{bw_ - 2.4:.2f} '
                f'a1.2,1.2 0 0 1 1.2,1.2 v{bh - 1.2:.2f}z" fill="{col}"/>')
        text(gx + bw_ / 2, base - ch * mean / 480 - 2.5, str(mean), 6.6, INK, 700, anchor="middle")
        text(x + i * gw + gw / 2, base + 9, b, 7.2, INK if exp else MUTED, 700 if exp else 400, anchor="middle")
    line(x, base, x + w, base, MUTED, 0.4)
    cy = base + 22
    cy = para(x, cy, "Missing signals such as view, floor, renovation state and finish matter most "
              "for premium flats, mainly around Zürich and Lake Geneva.", w, 7.4, MUTED, 1.38)
    end1 = cy

    x = COLX[1]
    cy = section(x, r3, w, "08", "Limitations")
    lims = [("calendar", "Snapshot dataset", "one scrape on 13 April 2026, no market trend"),
            ("city", "Coverage bias", "urban listings over-represented, rural areas sparse"),
            ("text", "Missing qualitative signals", "view, floor, renovation, balcony, furnishing"),
            ("alert", "High-price uncertainty", "largest errors for rare luxury apartments"),
            ("chart", "Overlapping model CIs", "the final choice is a robustness decision")]
    for ic, t, d in lims:
        icon(ic, x + 3, cy - 1, 11, MUTED, PANEL)
        text(x + 26, cy + 3.8, t, 8.4, INK, 700)
        text(x + 26, cy + 12.5, d, 7.2, MUTED)
        cy += 25
    end2 = cy

    x = COLX[2]
    cy = section(x, r3, w, "09", "Next steps")
    nexts = [("image", "Features from text + images", "extract floor, view and condition"),
             ("layers", "More listing platforms", "e.g. Homegate, ImmoScout24"),
             ("clock", "Repeated snapshots", "monthly scrapes instead of one"),
             ("trend", "Time-aware validation", "retrain and test on newer data"),
             ("rocket", "Production deployment", "model versioning and drift monitoring")]
    for ic, t, d in nexts:
        icon(ic, x + 3, cy - 1, 11, BLUE, "#E6F1FA")
        text(x + 26, cy + 3.8, t, 8.4, INK, 700)
        text(x + 26, cy + 12.5, d, 7.2, MUTED)
        cy += 25
    end3 = cy

    # ------------------------------------------------------------ footer
    fy = H - 62
    content_end = max(end1, end2, end3)
    if content_end > fy - 8:
        print(f"WARNING: content ends at {content_end:.1f} mm, footer starts at {fy} mm")
    rect(-ob, fy, W + 2 * ob, H - fy + ob, NAVY, r=0)
    head(M, fy + 25, "ACCURACY IS NOT EVERYTHING.", 22, "#FFFFFF")
    text(M, fy + 38, "The final model was chosen because it balances accuracy, stability and explainability.", 8.6, "#D0DCE7")
    text(W - M, fy + 20, "HSLU · DSPRO1 · TEAM 8 · SPRING 2026", 8, "#FFFFFF", 700, anchor="end", ls=0.08)
    text(W - M, fy + 31, "Code: github.com/wadafacc/dspro1", 7.4, "#D0DCE7", anchor="end")
    text(W - M, fy + 40.5, "Data: rentumo.ch · GWR (BFS) · swisstopo / GeoAdmin", 7.4, "#D0DCE7", anchor="end")
    add("</svg>")
    print(f"content ends at {content_end:.1f} mm (footer at {fy} mm)")
    return "\n".join(out)


if __name__ == "__main__":
    for variant in VARIANTS:
        for suffix, bleed in (("", 0), ("_Beschnitt3mm", BLEED)):
            path = OUT.with_name(f"{OUT.stem}{variant}{suffix}.svg")
            path.write_text(build(bleed, variant), encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
