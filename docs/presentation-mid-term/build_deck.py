"""Build the DSPRO2 mid-term deck in the HSLU layout used for the DSPRO1 mid-term.

Run: uv run --with python-pptx python docs/presentation-mid-term/build_deck.py
"""

import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
REPO = HERE.parents[1]
FIG = REPO / "docs/fig"
OUT = HERE / "DSPRO2_PresentationMidTerm_Team6_RentLens.pptx"

FONT = "Tahoma"
FOOTER = "RentLens: Explainable Rent Prediction and Quality of Life"
DATE = "22.10.2026"

BLACK = RGBColor(0, 0, 0)
WHITE = RGBColor(255, 255, 255)
GREY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
# Box palette taken from the DSPRO1 diagrams: (fill, border, text)
LAVENDER = (RGBColor(0xEE, 0xEF, 0xFE), RGBColor(0x8C, 0x8F, 0xE8), RGBColor(0x2B, 0x2D, 0x7A))
MINT = (RGBColor(0xE6, 0xF6, 0xEF), RGBColor(0x3C, 0xB3, 0x8A), RGBColor(0x0E, 0x5C, 0x43))
CREAM = (RGBColor(0xFD, 0xF3, 0xE1), RGBColor(0xE0, 0xA0, 0x40), RGBColor(0x6B, 0x45, 0x00))
ROSE = (RGBColor(0xFD, 0xEE, 0xEF), RGBColor(0xC8, 0x10, 0x2E), RGBColor(0xC8, 0x10, 0x2E))
NEUTRAL = (LIGHT, RGBColor(0xBF, 0xBF, 0xBF), RGBColor(0x40, 0x40, 0x40))
INK = (BLACK, BLACK, WHITE)
LIGHT_CONTAINER = (RGBColor(0xF3, 0xF0, 0xEA), RGBColor(0xB0, 0xA8, 0x9C), BLACK)
GREEN = RGBColor(0x1E, 0xB4, 0x5A)
ORANGE = RGBColor(0xF5, 0xA6, 0x23)
RED = RGBColor(0xC8, 0x10, 0x2E)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- primitives
def text(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    lines,
    size: float = 14,
    bold: bool = False,
    color: RGBColor = BLACK,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    italic: bool = False,
    spacing: float = 0,
):
    """Add a text box. `lines` is a str or a list of str / (str, dict) runs per paragraph."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Emu(0))
    fill_paragraphs(tf, lines, size, bold, color, align, italic, spacing)
    return box


def fill_paragraphs(tf, lines, size, bold, color, align, italic=False, spacing=0) -> None:
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        runs = line if isinstance(line, list) else [line]
        for run in runs:
            content, fmt = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            # keep numbers glued to their units so they never wrap apart
            glued = re.sub(r"(\d) (\d{3})\b", "\\1\u00a0\\2", content)
            glued = re.sub(r"\bCHF (\d)", "CHF\u00a0\\1", glued)
            r.text = glued.replace(" %", "\u00a0%").replace("α = ", "α\u00a0=\u00a0")
            f = r.font
            f.name = FONT
            f.size = Pt(fmt.get("size", size))
            f.bold = fmt.get("bold", bold)
            f.italic = fmt.get("italic", italic)
            f.color.rgb = fmt.get("color", color)


def box(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    lines,
    style=NEUTRAL,
    size: float = 13,
    bold: bool = False,
    align=PP_ALIGN.CENTER,
    anchor=MSO_ANCHOR.MIDDLE,
    border: float = 1.5,
    rounded: bool = True,
    margin: float = 0.12,
    dashed: bool = False,
):
    fill, line, ink = style
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(w),
        Inches(h),
    )
    if rounded:
        shape.adjustments[0] = min(0.12, 0.08 / max(h, 0.1) * 1.5)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(border)
    if dashed:
        shape.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.05)
    fill_paragraphs(tf, lines, size, bold, ink, align)
    return shape


def rect(slide, x: float, y: float, w: float, h: float, color: RGBColor):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def circle(slide, cx: float, cy: float, d: float, fill: RGBColor | None, line: RGBColor):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2), Inches(d), Inches(d)
    )
    if fill is None:
        shape.fill.solid()
        shape.fill.fore_color.rgb = WHITE
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(2)
    shape.shadow.inherit = False
    return shape


def arrow(slide, x1: float, y1: float, x2: float, y2: float, color=BLACK, width=1.75):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    ln = conn.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return conn


def line(slide, x1: float, y1: float, x2: float, y2: float, color=BLACK, width=3.0):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    return conn


def picture(slide, path: Path, x: float, y: float, w: float | None = None, h: float | None = None):
    return slide.shapes.add_picture(
        str(path),
        Inches(x),
        Inches(y),
        Inches(w) if w else None,
        Inches(h) if h else None,
    )


def fit_picture(slide, path: Path, x: float, y: float, w: float, h: float):
    """Place a picture centred inside the box (x, y, w, h), keeping its aspect ratio."""
    pic = picture(slide, path, x, y, w=w)
    if pic.height > Inches(h):
        pic.height, pic.width = Inches(h), int(pic.width * Inches(h) / pic.height)
    pic.left = Inches(x) + (Inches(w) - pic.width) // 2
    pic.top = Inches(y) + (Inches(h) - pic.height) // 2
    return pic


def notes(slide, body: str) -> None:
    slide.notes_slide.notes_text_frame.text = body


# ---------------------------------------------------------------- slide types
def footer(slide, number: int, dark: bool = False) -> None:
    ink = WHITE if dark else BLACK
    if dark:
        picture(slide, ASSETS / "logo_white_t.png", 0.42, 6.98, h=0.30)
    else:
        picture(slide, ASSETS / "logo_black.png", 0.524, 6.894, h=0.435)
    text(slide, 2.83, 6.98, 8.66, 0.2, FOOTER, size=7, color=ink, align=PP_ALIGN.RIGHT)
    text(slide, 11.69, 6.98, 0.79, 0.2, DATE, size=7, color=ink, align=PP_ALIGN.RIGHT)
    text(slide, 12.47, 6.98, 0.6, 0.2, str(number), size=7, color=BLACK, align=PP_ALIGN.CENTER)


def content(title: str, subtitle: str | None = None):
    slide = prs.slides.add_slide(BLANK)
    text(slide, 0.858, 0.334, 11.6, 0.4, title.upper(), size=18, bold=True)
    if subtitle:
        text(slide, 0.858, 0.76, 11.6, 0.4, subtitle, size=15, color=BLACK)
    footer(slide, len(prs.slides))
    return slide


def section(number: int, title: str):
    slide = prs.slides.add_slide(BLANK)
    rect(slide, 0, 0, 12.672, 7.5, BLACK)
    text(
        slide,
        2.826,
        1.6,
        6.694,
        1.3,
        str(number),
        size=60,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.BOTTOM,
    )
    text(
        slide,
        2.826,
        3.05,
        6.694,
        1.6,
        title.upper(),
        size=30,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    footer(slide, len(prs.slides), dark=True)
    return slide


def bullets(
    slide,
    x: float,
    y: float,
    w: float,
    h: float,
    items,
    size: float = 14,
    marker: str = "•",
    marker_color: RGBColor = ORANGE,
    spacing: float = 8,
):
    lines = [
        [
            (f"{marker}  ", {"color": marker_color, "bold": True}),
            *(item if isinstance(item, list) else [item]),
        ]
        for item in items
    ]
    return text(slide, x, y, w, h, lines, size=size, spacing=spacing)


# ================================================================ 1 title
s = prs.slides.add_slide(BLANK)
picture(s, ASSETS / "logo_black.png", 0.76, 0.48, h=0.55)
text(
    s,
    1.241,
    1.51,
    9.5,
    1.12,
    [
        "RENTLENS: AN AI-SUPPORTED RENTAL PLATFORM",
        "FOR TENANTS AND LANDLORDS",
        [
            (
                "Explainable rent prediction combined with a quality-of-life index",
                {"size": 14, "bold": False},
            )
        ],
    ],
    size=24,
    bold=True,
    anchor=MSO_ANCHOR.TOP,
)
picture(s, ASSETS / "campus.jpg", 3.279, 3.583, w=10.054, h=3.917)
rect(s, 1.25, 2.93, 5.5, 1.62, BLACK)
text(
    s,
    1.5,
    3.1,
    5.1,
    1.35,
    [
        "DSPRO2 Project · Mid-Term Presentation",
        "",
        [("Team:\t\t", {}), ("Elias Martinelli, Timo Schlumpf", {})],
        [("Supervisor:\t", {}), ("Dr. Elena Nazarenko", {})],
    ],
    size=12,
    color=WHITE,
)
picture(s, ASSETS / "stripes.png", 0, 4.87, w=1.65, h=1.65)
picture(s, ASSETS / "logo_black.png", 0.524, 6.894, h=0.435)
notes(
    s,
    "Welcome. We are Team 6, Elias and Timo. Last semester we predicted rents; this "
    "semester RentLens answers two questions per flat: is the rent fair, and is it a "
    "good place to live?",
)

# ================================================================ 2 agenda
s = content("Agenda")
agenda = [
    "The Problem",
    "Goals & Approach",
    "Current Status",
    "First Results (Pilot)",
    "Demo: Web App Prototype",
    "Next Steps & Open Questions",
    "Questions & Feedback",
]
for i, label in enumerate(agenda):
    col, row = divmod(i, 4)
    x0, y0 = (0.858, 5.781)[col], 1.505 + row * 1.12
    text(
        s,
        x0,
        y0,
        0.98,
        1.12,
        str(i + 1),
        size=30,
        bold=True,
        anchor=MSO_ANCHOR.MIDDLE,
        align=PP_ALIGN.CENTER,
    )
    text(s, x0 + 0.984, y0, 3.74, 1.12, label, size=14, anchor=MSO_ANCHOR.MIDDLE)
picture(s, ASSETS / "stripes.png", 11.684, 4.875, w=1.65, h=1.65)

# ================================================================ 3 section 1
section(1, "The Problem")

# ================================================================ 4 problem
s = content(
    "The Problem",
    "Swiss rents are opaque, and a price alone does not say whether a flat is a good place to live",
)
box(
    s,
    0.86,
    1.65,
    3.3,
    0.9,
    [[("You find a listing", {"bold": True})], "3.5 rooms, 80 m², Zürich"],
    style=INK,
    size=14,
)
arrow(s, 2.51, 2.55, 2.51, 3.05, color=RED, width=2.5)
box(
    s,
    0.86,
    3.05,
    3.3,
    1.75,
    [
        [("CHF 2 400 / month", {"bold": True, "size": 20})],
        "",
        "Is this rent fair?",
        "Is it a good place to live?",
    ],
    style=(WHITE, RED, RGBColor(0x40, 0x40, 0x40)),
    size=13,
    border=6,
)
text(s, 4.9, 1.6, 7.5, 0.35, "What our DSPRO1 model could not answer", size=14, bold=True)
problems = [
    ("One global uncertainty", "A constant ± band for every flat", LAVENDER),
    ("Expensive flats underestimated", "Systematic error in the top price segment", MINT),
    ("One snapshot, half usable", "Only ≈ 4 500 of 9 994 listings fully enriched", CREAM),
    ("Listing text unused", "Stored in the database, never used as a feature", NEUTRAL),
]
for i, (head, sub, style) in enumerate(problems):
    box(s, 4.9, 2.05 + i * 0.72, 7.5, 0.62, [[(head, {"bold": True})], sub], style=style, size=12)
box(
    s,
    0.86,
    5.15,
    11.54,
    0.6,
    [
        [
            ("→  DSPRO2: ", {"bold": True}),
            (
                "rent with calibrated interval, rent-check verdict, "
                "quality-of-life and value score",
                {},
            ),
        ]
    ],
    style=INK,
    size=14,
    rounded=False,
)
notes(
    s,
    "Example: a 3.5-room flat in Zurich for CHF 2400. Two questions: is the price fair, "
    "and is the location good? DSPRO1 only gave a point estimate with one global error "
    "band, underestimated expensive flats and ignored location quality and listing text.",
)

# ================================================================ 5 relevance
s = content("Why is it relevant?")
groups = [
    (
        "icon_tenant_c.png",
        "Tenants",
        "Explainable verdict: is the asking rent in line with "
        "the market? With an interval, not one number",
    ),
    (
        "icon_landlord.png",
        "Landlords",
        "A defensible price band and a what-if view before publishing a listing",
    ),
    (
        "icon_society.png",
        "Society",
        "More transparency in an opaque rental market; "
        "no legal assessment, links to official advice",
    ),
]
for i, (icon, head, sub) in enumerate(groups):
    x0 = 1.15 + i * 3.85
    fit_picture(s, ASSETS / icon, x0, 1.6, 3.2, 2.0)
    text(
        s,
        x0,
        3.85,
        3.2,
        0.4,
        [[("•  ", {"color": ORANGE}), head]],
        size=16,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    text(s, x0 + 0.1, 4.35, 3.0, 1.2, sub, size=13, align=PP_ALIGN.CENTER, color=GREY)
notes(
    s,
    "Tenants get a verdict with an interval instead of a single number; landlords get a "
    "price band they can justify; for society, more transparency in a rental market.",
)

# ================================================================ 6 section 2
section(2, "Goals & Approach")

# ================================================================ 7 research questions
s = content("Research Questions & Goals", "What we want to find out")
rqs = [
    (
        "P1 · RQ2",
        "Spatial model",
        "Does boosting with spatial mixed effects (GPBoost) reduce the error in sparsely covered "
        "municipalities vs. LightGBM with an encoded municipality, and leave less spatial "
        "pattern in the errors?",
        INK,
    ),
    (
        "P2 · RQ4",
        "Quality of life",
        "Can we build a valid quality-of-life index (QoLI) without ground truth? Checked "
        "against independent municipal statistics and for stability under other weights",
        INK,
    ),
    (
        "P3 · RQ3",
        "Price intervals",
        "Do calibrated 80 % intervals really contain 80 % of rents, and are they narrower "
        "than one constant ± band?",
        INK,
    ),
    (
        "Secondary · RQ1",
        "Listing text",
        "Does the listing description significantly improve the prediction over the tabular model?",
        NEUTRAL,
    ),
    (
        "Optional · RQ5",
        "Usability",
        "Do tenants and landlords find the verdict, the price band and the value score useful?",
        NEUTRAL,
    ),
]
for i, (tag, head, question, style) in enumerate(rqs):
    y0 = 1.45 + i * 0.84
    box(s, 0.86, y0, 2.1, 0.7, [[(tag, {"bold": True})], head], style=style, size=12)
    text(s, 3.2, y0, 9.2, 0.7, question, size=13, anchor=MSO_ANCHOR.MIDDLE)
box(
    s,
    0.86,
    5.75,
    11.54,
    0.55,
    [
        [
            ("Goal by 17.12.2026: ", {"bold": True}),
            (
                "tested rent model + intervals + QoLI, delivered in a web app "
                "(tenant check, landlord band, map)",
                {},
            ),
        ]
    ],
    style=ROSE,
    size=13,
)
notes(
    s,
    "Three core priorities: P1 spatial model, P2 quality-of-life index, P3 per-flat "
    "intervals. Text features are secondary, a user test is optional.",
)

# ================================================================ 8 approach
s = content("Planned Approach", "From listing to verdict")
steps = [
    ("Data", "Comparis listings (Apify) + building register (GWR) + geodata, OSM", LAVENDER),
    ("Features", "Size, rooms, location, municipality encoding, listing text", LAVENDER),
    ("Rent model", "LightGBM vs. GPBoost (spatial random effects)", MINT),
    ("Uncertainty", "Calibrated 80 % price interval per flat (CQR)", CREAM),
    ("QoLI + App", "Quality-of-life index (OECD/JRC), value score, Dash web app", ROSE),
]
w, gap = 2.06, 0.31
for i, (head, sub, style) in enumerate(steps):
    x0 = 0.86 + i * (w + gap)
    box(s, x0, 1.5, w, 1.75, [[(head, {"bold": True, "size": 15})], sub], style=style, size=12.5)
    if i < len(steps) - 1:
        arrow(s, x0 + w + 0.03, 2.375, x0 + w + gap - 0.03, 2.375)
line(s, 0.86, 3.55, 12.4, 3.55)
text(s, 0.86, 3.7, 6, 0.35, "Why this approach makes sense", size=15, bold=True)
reasons = [
    (
        "Gradient boosting",
        "Best tabular model in DSPRO1; captures non-linear interactions (area × location)",
    ),
    (
        "Spatial mixed effects",
        "Canton > district > municipality: sparse municipalities "
        "are shrunk towards their district, plus a variance per flat",
    ),
    (
        "Conformal calibration",
        "Guarantees the promised coverage without assuming a price distribution",
    ),
]
for i, (head, sub) in enumerate(reasons):
    box(
        s,
        0.86 + i * 3.9,
        4.15,
        3.74,
        1.45,
        [[(head, {"bold": True, "size": 14})], sub],
        style=NEUTRAL,
        size=12.5,
    )
text(
    s,
    0.86,
    5.85,
    11.54,
    0.6,
    [
        "GWR = federal building and dwelling register · OSM = OpenStreetMap · "
        "CQR = conformalized quantile regression · QoLI = quality-of-life index"
    ],
    size=11,
    color=GREY,
)
notes(
    s,
    "Pipeline: enrich listings, engineer features, compare two boosting models, wrap "
    "the winner in calibrated intervals, add a quality-of-life index and show all of it "
    "in a web app.",
)

# ================================================================ 9 evaluation design
s = content("Evaluation Design", "How we make the results trustworthy")
segments = [
    ("Training 60 %", 0.6, INK),
    ("Calibration 20 %", 0.2, LAVENDER),
    ("Test 20 % (hold-out)", 0.2, MINT),
]
x0 = 0.86
for label, frac, style in segments:
    width = 11.54 * frac
    box(s, x0, 1.5, width - 0.08, 0.55, label, style=style, size=13, rounded=False)
    x0 += width
text(
    s,
    0.86,
    2.15,
    11.54,
    0.35,
    "Split grouped by object (no flat appears in two sets), stratified by canton × price",
    size=12,
    color=GREY,
)
metrics = [
    ("MAE", "Mean absolute error in CHF: main metric", CREAM),
    ("Grouped 5-fold CV", "Model selection and Optuna tuning on training data only", LAVENDER),
    ("Paired tests", "Bootstrap + Wilcoxon, Holm-corrected, α = 0.05", MINT),
    ("Moran's I", "Are the errors still clustered in space?", NEUTRAL),
    ("Coverage & width", "Do intervals hold 80 %, and how wide are they?", ROSE),
]
w = 2.17
for i, (head, sub, style) in enumerate(metrics):
    box(
        s,
        0.86 + i * (w + 0.173),
        2.75,
        w,
        1.85,
        [[(head, {"bold": True, "size": 14})], sub],
        style=style,
        size=12.5,
    )
line(s, 0.86, 4.95, 12.4, 4.95)
box(
    s,
    0.86,
    5.2,
    11.54,
    0.65,
    [
        [
            ("Pre-registration: ", {"bold": True}),
            (
                "we freeze the evaluation plan before the final run, so the test set is used "
                "only once",
                {},
            ),
        ]
    ],
    style=INK,
    size=13,
    rounded=False,
)
notes(
    s,
    "MAE is the average absolute error in francs. All comparisons are paired on the same "
    "test flats. Moran's I measures whether errors cluster geographically.",
)

# ================================================================ 10 scope
s = content("Scope")
text(
    s,
    0.86,
    1.45,
    5.6,
    0.35,
    [[("•  ", {"color": GREEN}), ("In Scope", {"color": GREEN})]],
    size=14,
    bold=True,
)
text(s, 6.9, 1.45, 5.5, 0.35, "Out of Scope", size=14, bold=True, color=RED)
bullets(
    s,
    0.86,
    1.95,
    5.8,
    4.6,
    [
        "New Comparis listings via Apify + data freeze",
        "Enrichment with official geodata and OSM",
        "Spatial model comparison: LightGBM vs. GPBoost",
        "Calibrated price interval per flat",
        "Quality-of-life index and value score",
        "Text features from listing descriptions",
        "Dash web app: tenant check, landlord band, map",
        "Public deployment on Azure Container Apps",
        "Experiment tracking (MLflow), Kanban on GitHub",
    ],
    size=13,
    marker="✓",
    marker_color=GREEN,
    spacing=7,
)
bullets(
    s,
    6.9,
    1.95,
    5.5,
    4.6,
    [
        "Rent forecasting over time (time series)",
        "Cooperative / subsidised rents, shared-flat rooms",
        "Street-level map (privacy)",
        "Real-time price feed",
        "Legally binding rent assessment",
        "Aerial images, fused transformer (optional only)",
    ],
    size=13,
    marker="▬",
    marker_color=RED,
    spacing=7,
)

# ================================================================ 11 section 3
section(3, "Current Status")

# ================================================================ 12 progression
s = content("Status", "Project progression")
stages = [
    ("Proposal & research plan", "done", ""),
    ("Pilot pipeline on DSPRO1 data", "done", ""),
    ("Web app prototype", "progress", "11.12."),
    ("Apify data & freeze", "next", "06.11."),
    ("Features & text", "next", "20.11."),
    ("Final models & evaluation", "next", "06.12."),
]
w = 1.86
line(s, 0.86, 2.0, 12.4, 2.0, color=BLACK, width=1.5)
for i, (label, state, due) in enumerate(stages):
    x0 = 0.86 + i * (w + 0.076)
    cx = x0 + w / 2
    if state == "done":
        circle(s, cx, 2.0, 0.34, GREEN, GREEN)
        style = MINT
    elif state == "progress":
        circle(s, cx, 2.0, 0.34, ORANGE, ORANGE)
        style = CREAM
    else:
        circle(s, cx, 2.0, 0.34, None, BLACK)
        style = NEUTRAL
    box(s, x0, 2.55, w, 1.25, label, style=style, size=13, bold=state != "next")
    text(
        s,
        x0,
        3.9,
        w,
        0.3,
        f"due {due}" if due else "done",
        size=11,
        color=GREY,
        align=PP_ALIGN.CENTER,
    )
for i, (label, color, fill) in enumerate(
    [("Done", GREEN, GREEN), ("In progress", ORANGE, ORANGE), ("Upcoming", BLACK, None)]
):
    cx = 3.9 + i * 2.2
    circle(s, cx, 4.85, 0.22, fill, color)
    text(s, cx + 0.22, 4.72, 1.7, 0.3, label, size=12, color=GREY)
text(
    s,
    0.86,
    5.5,
    11.54,
    0.6,
    "Unusual order on purpose: we ran the full pipeline on DSPRO1 data first, so every "
    "research question has a tested code path before the new data arrives.",
    size=12,
    color=GREY,
    italic=True,
)
notes(
    s,
    "Green = done, orange = in progress, white = upcoming. Dates are the milestone "
    "deadlines on our GitHub board.",
)

# ================================================================ 13 done vs planned
s = content("Done vs. Planned")
cols = [
    (
        "Done",
        GREEN,
        "✓",
        [
            "Project proposal, research questions and priorities",
            "Data sourcing analysis (official data, OSM, scraping limits)",
            "Cleaning pipeline: 9 405 listings → 8 125 objects",
            "End-to-end pilot for RQ2, RQ3 and RQ4, tracked in MLflow",
            "Quality-of-life index for 2 110 municipalities",
            "600+ automated tests (pytest)",
        ],
    ),
    (
        "In progress",
        ORANGE,
        "●",
        [
            "Dash web app: tenant check, landlord price band, map",
            "Evaluation plan (draft, to be frozen)",
        ],
    ),
    (
        "Planned",
        BLACK,
        "○",
        [
            "Comparis data via Apify (with descriptions) + data freeze",
            "Text features (RQ1)",
            "Tune GPBoost before the final comparison",
            "Confirmatory run on frozen data",
            "Deployment on Azure Container Apps, CI; user test (optional)",
        ],
    ),
]
widths = [4.6, 3.0, 3.94]
x0 = 0.86
for (head, color, mark, items), w in zip(cols, widths, strict=True):
    text(s, x0, 1.4, w, 0.35, head, size=15, bold=True, color=color)
    rect(s, x0, 1.82, w - 0.25, 0.04, color)
    bullets(s, x0, 2.05, w - 0.3, 4.5, items, size=14, marker=mark, marker_color=color, spacing=10)
    x0 += w
notes(
    s,
    "Everything in the left column runs today, but only on the DSPRO1 snapshot. "
    "The right column is what turns the pilot into confirmed results.",
)

# ================================================================ 14 data so far
s = content("Data So Far", "Pilot data: DSPRO1 snapshot")
funnel = [
    ("Raw listings", "9 405", INK),
    ("One row per object (−1 107 duplicates)", "8 298", NEUTRAL),
    ("Swiss address, plausible area and rent", "8 280", NEUTRAL),
    ("Without price-per-m² outliers", "8 125", MINT),
]
for i, (label, n, style) in enumerate(funnel):
    inset = i * 0.18
    box(
        s,
        0.86 + inset,
        1.5 + i * 0.78,
        5.6 - 2 * inset,
        0.62,
        [[(f"{n}   ", {"bold": True, "size": 16}), (label, {})]],
        style=style,
        size=12.5,
    )
box(
    s,
    0.86,
    4.75,
    5.6,
    0.5,
    "Split: 4 875 train · 1 625 calibration · 1 625 test",
    style=LAVENDER,
    size=12,
)
text(
    s,
    0.86,
    5.4,
    5.6,
    1.0,
    [
        [
            ("Gaps: ", {"bold": True}),
            "no listing text in the snapshot; 51.5 % of objects lack building-register data.",
        ]
    ],
    size=11.5,
    color=GREY,
)
fit_picture(s, FIG / "objects_language_map.png", 6.8, 1.35, 5.6, 5.0)
notes(
    s,
    "We clean in four transparent steps; most losses are duplicates. The map shows "
    "coverage: mostly German-speaking Switzerland, fewer Italian-speaking listings.",
)

# ================================================================ 14b data source
s = content("Data Source", "From DSPRO1 to DSPRO2")
sides = [
    (
        0.86,
        "DSPRO1",
        BLACK,
        ("Done", MINT),
        "logo_rentumo.png",
        "Rentumo",
        "logo_rust.png",
        "Own Rust scraper",
        [
            ("9 994", "listings scraped"),
            ("1", "snapshot, 13.04.2026"),
            ("No", "listing text in the training data"),
        ],
    ),
    (
        7.4,
        "DSPRO2",
        RED,
        ("Planned", NEUTRAL),
        "logo_comparis.png",
        "Comparis",
        "logo_apify.png",
        "Collected via Apify",
        [
            ("≈ 20 000", "objects (target)"),
            ("06.11.", "data freeze"),
            ("Yes", "listing text included → RQ1"),
        ],
    ),
]
for x0, head, color, (badge, badge_style), logo, name, tool, tool_label, facts in sides:
    text(s, x0, 1.35, 2.0, 0.45, head, size=20, bold=True, color=color)
    box(s, x0 + 1.75, 1.42, 1.2, 0.34, badge, style=badge_style, size=11, bold=True)
    picture(s, ASSETS / logo, x0, 2.0, w=1.3, h=1.3)
    text(s, x0 + 1.55, 2.05, 3.4, 0.45, name, size=20, bold=True)
    fit_picture(s, ASSETS / tool, x0 + 1.55, 2.7, 0.45, 0.45)
    text(s, x0 + 2.1, 2.7, 2.9, 0.45, tool_label, size=13, color=GREY, anchor=MSO_ANCHOR.MIDDLE)
    for i, (value, label) in enumerate(facts):
        y0 = 3.6 + i * 0.55
        text(s, x0, y0, 1.55, 0.5, value, size=18, bold=True, color=color, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x0 + 1.55, y0, 3.45, 0.5, label, size=13, anchor=MSO_ANCHOR.MIDDLE)
chevron = s.shapes.add_shape(
    MSO_SHAPE.RIGHT_ARROW, Inches(6.05), Inches(3.05), Inches(1.1), Inches(0.75)
)
chevron.fill.solid()
chevron.fill.fore_color.rgb = BLACK
chevron.line.fill.background()
chevron.shadow.inherit = False
line(s, 0.86, 5.45, 12.4, 5.45, width=1.5)
text(s, 0.86, 5.62, 2.3, 0.5, "Enrichment for both", size=13, bold=True, anchor=MSO_ANCHOR.MIDDLE)
fit_picture(s, ASSETS / "logo_openstreetmap.png", 3.1, 5.62, 0.5, 0.5)
chips = ["OpenStreetMap", "GWR building register", "geo.admin / swisstopo", "BFS", "BAFU"]
widths = [1.6, 2.3, 2.2, 0.8, 0.95]
x0 = 3.7
for chip, w in zip(chips, widths, strict=True):
    box(s, x0, 5.65, w, 0.44, chip, style=NEUTRAL, size=11.5)
    x0 += w + 0.12
text(
    s,
    7.4,
    1.0,
    5.0,
    0.3,
    "Change since proposal: own incremental scraper → Apify",
    size=10.5,
    color=GREY,
    italic=True,
    align=PP_ALIGN.RIGHT,
)
notes(
    s,
    "DSPRO1: Comparis declined (only about 300 records with approval), Immoscout was "
    "not interested, Rentumo did not answer, so we scraped rentumo.ch with our own Rust "
    "scraper: one snapshot, no listing text. DSPRO2: Comparis listings collected via "
    "Apify, including descriptions, until the data freeze on 06.11. Collection has not "
    "started yet. Enrichment stays the same, now from downloads instead of slow APIs.",
)

# ================================================================ 14c pipeline design
PEACH = (RGBColor(0xFB, 0xEC, 0xE6), RGBColor(0xC0, 0x6A, 0x4E), RGBColor(0x7A, 0x2E, 0x18))
PLANNED = (WHITE, RGBColor(0x80, 0x80, 0x80), RGBColor(0x40, 0x40, 0x40))
s = content("Data Pipeline High-Level Design", "DSPRO2: from the sources to the app")
col_w, col_gap = 2.1, 0.26
cols_x = [0.86 + i * (col_w + col_gap) for i in range(5)]
heads = ["Sources", "Ingestion (rentml)", "Storage", "Modelling", "Serving"]
for x0, head in zip(cols_x, heads, strict=True):
    text(s, x0, 1.3, col_w, 0.3, head, size=12, bold=True, color=GREY, align=PP_ALIGN.CENTER)

# Listing source is new in DSPRO2: Comparis via Apify replaces the DSPRO1 rentumo scraper
box(
    s,
    cols_x[0],
    1.75,
    col_w,
    1.55,
    ["", "", [("Comparis via Apify", {"bold": True})], "listings + descriptions"],
    style=INK,
    size=11,
    anchor=MSO_ANCHOR.BOTTOM,
)
picture(s, ASSETS / "logo_comparis.png", cols_x[0] + 0.5, 1.85, w=0.5, h=0.5)
apify_bg = box(s, cols_x[0] + 1.1, 1.85, 0.5, 0.5, "", style=(WHITE, WHITE, BLACK))
apify_bg.adjustments[0] = 0.5
fit_picture(s, ASSETS / "logo_apify.png", cols_x[0] + 1.15, 1.9, 0.4, 0.4)
box(s, cols_x[0] + 1.55, 1.62, 0.62, 0.26, "NEW", style=(RED, RED, WHITE), size=9, bold=True)
sources = [
    ([("Official geodata", {"bold": True})], "GWR, swisstopo, ARE, BAFU, BFS"),
    ([("OpenStreetMap", {"bold": True})], "amenities (Geofabrik)"),
]
for i, (head, sub) in enumerate(sources):
    box(s, cols_x[0], 3.55 + i * 1.05, col_w, 0.85, [head, sub], style=INK, size=11)

container = box(
    s, cols_x[1] - 0.08, 1.65, col_w + 0.16, 4.0, "", style=LIGHT_CONTAINER, dashed=True, border=1
)
ingest = [
    ("Clean & deduplicate", "one object_id per flat, domain rules, outliers"),
    ("Anonymise text", "remove contacts and prices from descriptions"),
    ("Enrich", "spatial joins via coordinates / EGID, QoLI layers"),
]
for i, (head, sub) in enumerate(ingest):
    box(
        s,
        cols_x[1],
        1.85 + i * 1.25,
        col_w,
        1.05,
        [[(head, {"bold": True})], sub],
        style=PEACH,
        size=11,
    )

box(
    s,
    cols_x[2],
    2.45,
    col_w,
    1.4,
    [[("SQL DB", {"bold": True})], "neon.com", "(PostgreSQL serverless)"],
    style=MINT,
    size=11,
)
box(
    s,
    cols_x[2],
    4.25,
    col_w,
    1.0,
    [[("Data freeze 06.11.", {"bold": True})], "frozen snapshot for the final run"],
    style=PLANNED,
    size=11,
    dashed=True,
)

models = [
    ("Features + split", "grouped 60/20/20, target encoding"),
    ("Rent model", "LightGBM vs. GPBoost"),
    ("Intervals", "quantile models + CQR"),
    ("QoLI", "index + value score"),
]
for i, (head, sub) in enumerate(models):
    box(
        s,
        cols_x[3],
        1.75 + i * 1.0,
        col_w,
        0.82,
        [[(head, {"bold": True})], sub],
        style=LAVENDER,
        size=11,
    )

box(
    s,
    cols_x[4],
    2.45,
    col_w,
    1.4,
    [[("Dash web app", {"bold": True})], "Fair-Rent Check, price band, marketplace, map"],
    style=INK,
    size=11,
)
box(
    s,
    cols_x[4],
    4.25,
    col_w,
    1.0,
    [[("Azure Container Apps", {"bold": True})], "public deployment by 11.12."],
    style=PLANNED,
    size=11,
    dashed=True,
)

for i in range(4):
    x_end = cols_x[i] + col_w + (0.1 if i == 0 else 0.05 if i == 1 else 0.03)
    arrow(s, x_end, 3.15, cols_x[i + 1] - (0.1 if i == 0 else 0.03), 3.15)

text(
    s,
    cols_x[3],
    5.75,
    col_w,
    0.3,
    "all runs tracked in MLflow",
    size=10.5,
    color=GREY,
    align=PP_ALIGN.CENTER,
    italic=True,
)
text(
    s,
    cols_x[4],
    5.4,
    col_w,
    0.6,
    "live address lookup: geo.admin.ch",
    size=10.5,
    color=GREY,
    align=PP_ALIGN.CENTER,
    italic=True,
)
box(s, 0.86, 6.15, 2.2, 0.36, "dated milestone", style=PLANNED, size=10.5, dashed=True)
notes(
    s,
    "Same idea as in DSPRO1, updated: the listing source is new, Comparis collected via "
    "Apify instead of our own rentumo scraper. Our Python package rentml cleans, "
    "deduplicates, anonymises and enriches, Neon stores everything, then the modelling "
    "pipeline and the Dash app. The Apify import has not started yet; the pilot results "
    "still come from the DSPRO1 snapshot. Dashed boxes are dated milestones: data freeze "
    "on 06.11. and public deployment on Azure by 11.12.",
)

# ================================================================ 14d enrichment
s = content("Data Enrichment", "Each address gets building, location and quality-of-life data")
box(
    s,
    0.86,
    1.3,
    11.54,
    0.5,
    [
        [
            ("DSPRO1: ", {"bold": True}),
            ("one API call per listing (frequent timeouts)   →   ", {}),
            ("DSPRO2: ", {"bold": True, "color": RED}),
            ("official downloads, joined locally per coordinate or EGID", {}),
        ]
    ],
    style=NEUTRAL,
    size=12.5,
)
fit_picture(s, FIG / "qoli_dimension_maps.png", 0.86, 1.95, 5.9, 3.95)
text(
    s,
    0.86,
    5.95,
    5.9,
    0.3,
    "Four core QoLI dimensions per municipality (0–100), built from these layers",
    size=10.5,
    color=GREY,
    italic=True,
    align=PP_ALIGN.CENTER,
)
layers = [
    ("GWR (EGID)", "year built, dwellings, building area"),
    ("swisstopo", "coordinates, elevation, solar, municipality"),
    ("ARE", "public-transport quality class, distance to stop"),
    ("BAFU", "road and rail noise day/night (10 m), NO₂, PM2.5"),
    ("BFS", "population grid, green share (land use), vacancy"),
    ("OpenStreetMap", "walking access to 7 amenity types (shops, doctors, schools …)"),
    ("MeteoSwiss, ESTV", "sunshine, tax burden (extended index)"),
]
for i, (source, adds) in enumerate(layers):
    y0 = 2.0 + i * 0.56
    box(s, 7.0, y0, 1.75, 0.46, source, style=INK, size=11, bold=True)
    text(s, 8.9, y0, 3.5, 0.46, adds, size=11.5, anchor=MSO_ANCHOR.MIDDLE)
notes(
    s,
    "In DSPRO1 we called geo.admin and the GWR API for every listing, which was slow and "
    "timed out. Now we download the official layers once and join them locally by "
    "coordinate or building ID. The same layers feed the rent model and the quality-of-life "
    "index; the maps show its four core dimensions per municipality.",
)

# ================================================================ 15 section 4
section(4, "First Results (Pilot)")

# ================================================================ 16 baseline
s = content("DSPRO1 Baseline vs. DSPRO2 Pilot", "MAE = mean absolute error in CHF")
box(
    s,
    0.86,
    1.45,
    3.3,
    1.25,
    [[("Baseline: DSPRO1 final model", {"bold": True})], "GradientBoosting (ALL+geo)"],
    style=INK,
    size=13,
)
arrow(s, 4.2, 2.075, 4.6, 2.075)
baseline = [
    ("MAE", "CHF 281", "avg. prediction error", CREAM),
    ("RMSE", "CHF 425", "penalises large errors", CREAM),
    ("R²", "0.71", "variance explained", INK),
]
for i, (head, value, sub, style) in enumerate(baseline):
    box(
        s,
        4.65 + i * 2.62,
        1.45,
        2.45,
        1.25,
        [[(head, {"bold": True})], [(value, {"size": 22})], sub],
        style=style,
        size=12,
    )
text(
    s,
    0.86,
    2.8,
    11.54,
    0.3,
    "DSPRO1 evaluation split, fully enriched listings only, train/eval gap 60 CHF "
    "(proposal, DSPRO1 report)",
    size=11,
    color=GREY,
)
line(s, 0.86, 3.25, 12.4, 3.25)
text(
    s,
    0.86,
    3.4,
    11.54,
    0.35,
    "DSPRO2 pilot: same DSPRO1 listings, new grouped split, test set n = 1 625",
    size=13,
    bold=True,
)
pilot = [
    ("Naive reference", "MAE CHF 355", "median CHF/m² of the municipality · R² 0.62", NEUTRAL),
    ("LightGBM, tabular", "MAE CHF 302", "flat, building and area features · R² 0.66", CREAM),
    ("LightGBM + location", "MAE CHF 261", "+ encoded municipality · RMSE 483 · R² 0.74", INK),
]
for i, (head, value, sub, style) in enumerate(pilot):
    box(
        s,
        0.86 + i * 3.9,
        3.85,
        3.74,
        1.25,
        [[(head, {"bold": True})], [(value, {"size": 20})], sub],
        style=style,
        size=12,
    )
box(
    s,
    0.86,
    5.3,
    11.54,
    0.95,
    [
        [("Interpretation", {"bold": True, "color": RED})],
        "Not yet a fair comparison: different split, and DSPRO1 used only fully enriched "
        "listings. As planned in the proposal, the new baseline is this GradientBoosting "
        "retrained on the new deduplicated data: the bar every DSPRO2 model has to clear.",
    ],
    style=(RGBColor(0xFD, 0xF5, 0xF6), RED, RGBColor(0x40, 0x40, 0x40)),
    size=12.5,
)
notes(
    s,
    "The baseline is our final DSPRO1 model: GradientBoosting on ALL+geo features, "
    "MAE 281, RMSE 425, R² 0.71 on the DSPRO1 evaluation split. The DSPRO2 pilot "
    "numbers come from a new grouped split that also contains listings without building "
    "data, so the two rows are not directly comparable yet. The naive reference uses the "
    "median price per m² of the municipality.",
)

# ================================================================ 17 RQ2
s = content("RQ2: Spatial Model", "GPBoost vs. LightGBM by listings per municipality")
fit_picture(s, FIG / "rq2_mae_by_bucket.png", 0.86, 1.35, 7.2, 4.4)
bullets(
    s,
    8.35,
    1.45,
    4.1,
    3.2,
    [
        [("MAE: ", {"bold": True}), "LightGBM 261 vs. GPBoost 269 CHF"],
        [("Difference: ", {"bold": True}), "−8.7 CHF (95 % CI −15.0 to −2.2)"],
        [
            ("Sparse areas ", {"bold": True}),
            "(< 5 listings): −18.5 CHF, again in favour of LightGBM",
        ],
        [
            ("But: ", {"bold": True}),
            "GPBoost errors are less clustered in space (Moran's I 0.017 vs. 0.025)",
        ],
    ],
    size=13.5,
    spacing=10,
)
box(
    s,
    8.35,
    4.55,
    4.05,
    1.65,
    [
        [("Decision", {"bold": True})],
        "Hypothesis not supported in the pilot. GPBoost is still untuned → tune it before the "
        "final run; LightGBM is our main model for now.",
    ],
    style=INK,
    size=12.5,
    align=PP_ALIGN.LEFT,
)
notes(
    s,
    "Bars: test error per bucket of training listings per municipality, with 95 % "
    "bootstrap intervals (dspro1_knn = DSPRO1 kNN rent features refit with "
    "LightGBM). We expected GPBoost to win in sparse municipalities; in the "
    "pilot it does not. Caveat: GPBoost was not tuned yet, so this is not final.",
)

# ================================================================ 18 RQ3
s = content(
    "RQ3: Price Intervals", "80 % interval = should contain the true rent for 8 out of 10 flats"
)
fit_picture(s, FIG / "rq3_reliability.png", 0.86, 1.3, 11.54, 3.66)
cards = [
    ("Coverage 80.2 %", "Target 80 % reached on the test set", MINT),
    (
        "All 12 groups OK",
        "Language region × price band, lowest 76.9 %; a constant band falls to 46 %",
        LAVENDER,
    ),
    (
        "18.5 % wider",
        "Mean width CHF 851 vs. 718 for a constant band → second hypothesis not supported",
        CREAM,
    ),
]
for i, (head, sub, style) in enumerate(cards):
    box(
        s,
        0.86 + i * 3.9,
        5.1,
        3.74,
        1.15,
        [[(head, {"bold": True, "size": 14})], sub],
        style=style,
        size=12,
    )
notes(
    s,
    "Bars: coverage per language region and price band. The method (Mondrian "
    "conformalized quantile regression) keeps its promise everywhere, but intervals are "
    "wider on average than one constant band. That is the price of being fair to every "
    "segment.",
)

# ================================================================ 19 RQ4
s = content(
    "RQ4: Quality-of-Life Index",
    "Core index from open data: public transport, noise, green space, services",
)
fit_picture(s, FIG / "qoli_map.png", 0.86, 1.3, 11.54, 3.66)
cards = [
    ("2 110 municipalities", "Index for all 8 125 objects, 100 m resolution", NEUTRAL),
    (
        "Robust to weights",
        "Median rank correlation 0.93 under random weights; worst 5 % below our threshold",
        CREAM,
    ),
    (
        "Validity: 2 of 4 checks",
        "Matches rent per m² and official accessibility; not vacancy or population growth",
        LAVENDER,
    ),
]
for i, (head, sub, style) in enumerate(cards):
    box(
        s,
        0.86 + i * 3.9,
        5.1,
        3.74,
        1.15,
        [[(head, {"bold": True, "size": 14})], sub],
        style=style,
        size=12,
    )
notes(
    s,
    "Left: index per inhabited hectare, right: per municipality. There is no ground "
    "truth for quality of life, so we check stability under weight changes and "
    "agreement with independent statistics. Answer so far: partially valid.",
)


# ================================================================ demo section
def screenshot(slide, name: str, x: float, y: float, w: float, h: float):
    pic = fit_picture(slide, ASSETS / name, x, y, w, h)
    pic.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    pic.line.width = Pt(1)
    return pic


s = section(5, "Demo: Web App")
notes(
    s,
    "Live demo, about 2 minutes. Start before the talk: uv run --extra app python -m "
    "rentml.webapp, then open http://127.0.0.1:8050/mietcheck. Enter Langstrasse 4 Zürich, "
    "80 m², 3.5 rooms, CHF 2400 → verdict within the expected range, estimate CHF 3451. "
    "Then /vermieter (price band and what-if) and /karte. Backup: the three hidden slides "
    "after this one contain screenshots of the same steps.",
)

s = content(
    "Demo: Fair-Rent Check",
    "The example from the start: 3.5 rooms, 80 m², Zürich, CHF 2 400 per month",
)
screenshot(s, "demo_mietcheck.jpg", 0.86, 1.3, 7.6, 4.1)
bullets(
    s,
    8.75,
    1.4,
    3.65,
    3.2,
    [
        [("Verdict: ", {"bold": True}), "within the expected range"],
        [("Estimate: ", {"bold": True}), "CHF 3 451, 80 % interval CHF 2 196–4 344"],
        [("Percentile: ", {"bold": True}), "about 23rd, so cheaper than most comparable offers"],
        [("Why: ", {"bold": True}), "municipality and district rent level (SHAP drivers)"],
    ],
    size=13,
    spacing=10,
)
box(
    s,
    8.75,
    4.55,
    3.65,
    0.85,
    "Answers «Is this rent fair?» from slide 4; «good place to live?» comes from the QoLI",
    style=INK,
    size=12,
)
text(
    s,
    0.86,
    5.6,
    11.54,
    0.6,
    "Prototype: pilot model trained on DSPRO1 data. Address lookup via geo.admin.ch; "
    "building and location data are added automatically.",
    size=11.5,
    color=GREY,
    italic=True,
)
notes(
    s,
    "Live demo: open /mietcheck, pick an address, enter 80 m², 3.5 rooms, CHF 2400. "
    "The red line is the asking rent, the diamond the estimate, the bands the 80 % "
    "interval and the middle 50 %. Backup: this screenshot.",
)

s = content("Demo: Price Band for Landlords", "Same flat, landlord view with what-if")
screenshot(s, "demo_preisband.jpg", 0.86, 1.3, 7.6, 4.6)
bullets(
    s,
    8.75,
    1.4,
    3.65,
    3.4,
    [
        [
            ("Suggested band: ", {"bold": True}),
            "CHF 2 434–3 517 (middle 50 % of comparable offers)",
        ],
        [("What-if: ", {"bold": True}), "change area or rooms and see the effect on the estimate"],
        [("Transparent: ", {"bold": True}), "shows which building data was missing"],
    ],
    size=13,
    spacing=10,
)
text(
    s,
    0.86,
    6.0,
    11.54,
    0.4,
    "Renovation, balcony or lift as what-if needs text features (RQ1), planned after the "
    "Apify data.",
    size=11.5,
    color=GREY,
    italic=True,
)

s = content(
    "Demo: Marketplace and Map", "Every listing gets a price check and a quality-of-life score"
)
screenshot(s, "demo_home.jpg", 0.86, 1.35, 5.65, 3.55)
screenshot(s, "demo_karte.jpg", 6.75, 1.35, 5.65, 3.55)
text(
    s,
    0.86,
    5.0,
    5.65,
    0.6,
    [
        [
            ("Marketplace: ", {"bold": True}),
            "price check, quality of life and value score on every listing",
        ]
    ],
    size=12.5,
    align=PP_ALIGN.CENTER,
)
text(
    s,
    6.75,
    5.0,
    5.65,
    0.6,
    [[("Map: ", {"bold": True}), "median rent per m² by canton, district and municipality"]],
    size=12.5,
    align=PP_ALIGN.CENTER,
)
text(
    s,
    0.86,
    5.85,
    11.54,
    0.4,
    "Status: prototype in progress. Listings shown are synthetic demo data (marked «Demo»); "
    "deployment follows by 11.12.",
    size=11.5,
    color=GREY,
    italic=True,
)

# ================================================================ 20 section 5
section(6, "Next Steps")

# ================================================================ 21 next steps
s = content("Next Steps", "Road to the final presentation")
plan = [
    (
        "until 06.11.",
        "Apify data",
        "Comparis listings with descriptions, enrichment, data freeze, datasheet",
        LAVENDER,
    ),
    ("until 20.11.", "Features", "Text attributes and embeddings (RQ1)", MINT),
    ("until 06.12.", "Models", "Tune GPBoost, confirmatory run, evaluation", CREAM),
    ("until 11.12.", "App", "Dash app, deployment on Azure Container Apps", ROSE),
    ("17.12.", "Final", "Final presentation; report due 31.12.", INK),
]
w, gap = 2.06, 0.31
for i, (due, head, sub, style) in enumerate(plan):
    x0 = 0.86 + i * (w + gap)
    text(s, x0, 1.5, w, 0.35, due, size=13, bold=True, align=PP_ALIGN.CENTER)
    box(s, x0, 1.95, w, 2.0, [[(head, {"bold": True, "size": 15})], "", sub], style=style, size=12)
    if i < len(plan) - 1:
        arrow(s, x0 + w + 0.03, 2.95, x0 + w + gap - 0.03, 2.95)
line(s, 0.86, 4.35, 12.4, 4.35)
text(
    s,
    0.86,
    4.55,
    11.54,
    1.3,
    [
        [
            ("Critical path: ", {"bold": True}),
            "data freeze → confirmatory run. Everything after it reuses the pipeline that "
            "already runs on the pilot data.",
        ]
    ],
    size=13,
)

# ================================================================ 22 open questions
s = content("Open Questions & Risks")
questions = [
    (
        "Data access",
        "Proposal excluded Comparis (formal approval needed in DSPRO1); "
        "now via Apify → clarify permission before the first run",
        ROSE,
    ),
    ("RQ2 still open", "Will GPBoost catch up once it is tuned?", LAVENDER),
    (
        "Interval width",
        "Can text features or better quantile models beat the constant band?",
        CREAM,
    ),
    (
        "QoLI validation",
        "Which external benchmark? The BILANZ/IAZI ranking is not available to us",
        MINT,
    ),
    ("Missing data", "51.5 % of objects lack building-register data", NEUTRAL),
]
for i, (head, sub, style) in enumerate(questions):
    box(
        s,
        0.86,
        1.4 + i * 0.8,
        11.54,
        0.66,
        [[(f"{head}:  ", {"bold": True}), (sub, {})]],
        style=style,
        size=13,
        align=PP_ALIGN.LEFT,
        margin=0.25,
    )
box(
    s,
    0.86,
    5.55,
    11.54,
    0.6,
    [
        [
            ("Feedback welcome: ", {"bold": True}),
            ("is P1 spatial model → P2 quality of life → P3 intervals the right focus?", {}),
        ]
    ],
    style=INK,
    size=13,
    rounded=False,
)

# ================================================================ 23 thank you
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, BLACK)
text(
    s,
    1.842,
    1.427,
    9.646,
    1.2,
    ["THANK YOU FOR YOUR", "ATTENTION."],
    size=26,
    bold=True,
    color=WHITE,
    anchor=MSO_ANCHOR.BOTTOM,
)
text(
    s,
    1.842,
    2.874,
    9.646,
    0.6,
    "We are happy to answer any questions or discuss the presented topics further.",
    size=13,
    color=WHITE,
)
picture(s, ASSETS / "logo_white_t.png", 1.80, 4.55, h=0.33)
text(
    s,
    1.842,
    5.106,
    9.646,
    1.44,
    [
        [("DSPRO2 Project", {"bold": True})],
        "",
        [("Team:", {"bold": True})],
        "•  Elias Martinelli",
        "•  Timo Schlumpf",
        [("Supervisor:", {"bold": True})],
        "•  Dr. Elena Nazarenko",
    ],
    size=9,
    color=WHITE,
)

# ================================================================ 24 logo
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.333, 7.5, BLACK)
picture(s, ASSETS / "logo_white_t.png", 3.197, 2.95, w=6.94)

# The demo runs live: keep the screenshot slides as hidden backup and number only the
# slides that are actually shown.
visible = 0
for slide in prs.slides:
    texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame]
    title = next((t for t in texts if t.strip()), "")
    if title.startswith("DEMO: "):
        slide._element.set("show", "0")
    else:
        visible += 1
    for sh in slide.shapes:
        if sh.has_text_frame and sh.left == Inches(12.47) and sh.text_frame.text.isdigit():
            run = sh.text_frame.paragraphs[0].runs[0]
            run.text = str(visible) if slide._element.get("show") != "0" else ""

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print(f"saved {OUT} ({len(prs.slides)} slides)")
