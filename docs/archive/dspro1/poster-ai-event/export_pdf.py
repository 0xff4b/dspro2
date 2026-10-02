"""Export the poster SVGs to print-ready PDFs and a PNG preview with headless Chrome.

Writes the A0 PDF (841 x 1189 mm) and the print-shop PDF with 3 mm bleed
(847 x 1195 mm, TrimBox = A0, BleedBox = full page).
Requires `playwright` (uses the installed Google Chrome) and `pypdf`.
Usage:  .venv/bin/python docs/dspro1/poster-ai-event/export_pdf.py
"""
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

HERE = Path(__file__).resolve().parent
W, H, BLEED = 841, 1189, 3
JOBS = [(HERE / f"DISPRO1_PosterAIEvent_Team8_A0{v}{suffix}.svg", bleed, HERE / f"preview{v.lower()}.png")
        for v in ("", "_V2_Adresse") for suffix, bleed in (("", 0), ("_Beschnitt3mm", BLEED))]
PT = 72 / 25.4


def set_boxes(pdf: Path, bleed: float) -> None:
    """Chrome rounds the page up to whole pixels and centres the content; crop to the exact size."""
    reader = PdfReader(pdf)
    writer = PdfWriter()
    w, h = (W + 2 * bleed) * PT, (H + 2 * bleed) * PT
    b = bleed * PT
    for page in reader.pages:
        mb = page.mediabox
        x0 = float(mb.left) + (float(mb.width) - w) / 2
        y0 = float(mb.bottom) + (float(mb.height) - h) / 2
        page.mediabox = page.cropbox = page.bleedbox = RectangleObject([x0, y0, x0 + w, y0 + h])
        page.trimbox = RectangleObject([x0 + b, y0 + b, x0 + w - b, y0 + h - b])
        writer.add_page(page)
    writer.add_metadata({"/Title": "Predicting Apartment Rental Prices in Switzerland – DSPRO1 Team 8"})
    with pdf.open("wb") as f:
        writer.write(f)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    for svg_path, bleed, png in JOBS:
        w, h = W + 2 * bleed, H + 2 * bleed
        html = ("<!doctype html><html><head><meta charset='utf-8'><style>"
                f"@page{{size:{w}mm {h}mm;margin:0}}html,body{{margin:0;padding:0}}"
                f"svg{{display:block;width:{w}mm;height:{h}mm}}</style></head><body>"
                + svg_path.read_text(encoding="utf-8") + "</body></html>")
        page = browser.new_page(viewport={"width": round(w / 25.4 * 96), "height": round(h / 25.4 * 96)},
                                device_scale_factor=0.5)
        page.set_content(html, wait_until="load")
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(500)
        pdf = svg_path.with_suffix(".pdf")
        page.pdf(path=str(pdf), width=f"{w}mm", height=f"{h}mm", print_background=True,
                 margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        if bleed == 0:
            page.screenshot(path=str(png))
        page.close()
        set_boxes(pdf, bleed)
        print(f"wrote {pdf.name}")
    browser.close()
