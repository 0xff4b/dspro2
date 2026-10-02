# AI-Event: A0-Poster

Poster im HSLU-/Website-Stil (Farben aus `src/plot_style.py`, Titel in Barlow Condensed, Text Arial/Helvetica).

| Datei | Zweck |
|---|---|
| `DISPRO1_PosterAIEvent_Team8_A0.pdf` | **Druckdatei**: A0 hoch (841 × 1189 mm), Vektor, Schriften eingebettet |
| `DISPRO1_PosterAIEvent_Team8_A0_Beschnitt3mm.pdf` | **Druckdatei mit 3 mm Beschnittzugabe**: 847 × 1195 mm, TrimBox = A0, BleedBox = ganze Seite |
| `DISPRO1_PosterAIEvent_Team8_A0.svg`, `…_Beschnitt3mm.svg` | Quellen, editierbar (Illustrator, Inkscape, Affinity) |
| `…_V2_Adresse.pdf` / `…_V2_Adresse_Beschnitt3mm.pdf` | **Version 2**: Sektion 06 zeigt die echte Adresssuche (Baselstrasse 30, 6003 Luzern) statt der Beispielwohnung |
| `preview.png`, `preview_v2_adresse.png` | Vorschauen |
| `build_poster.py` | Erzeugt beide SVGs (ohne und mit Beschnitt) |
| `export_pdf.py` | Erzeugt beide PDFs und die Vorschau mit Headless-Chrome, setzt TrimBox/BleedBox |
| `assets/prototype_demo.jpg` | Screenshot der lokalen App (Beispielwohnung Luzern), 3240 px ≈ 336 ppi im Druck |
| `assets/prototype_address.jpg` | Screenshot Adresssuche Baselstrasse 30, 6003 Luzern (GWR-Wohnung 2. Stock, 50 m², 2 Zimmer; Schätzung 1'963 CHF mit best_model_v3) |

## Neu erzeugen

```bash
pip install segno playwright pypdf   # einmalig; export_pdf.py nutzt das installierte Google Chrome
.venv/bin/python docs/dspro1/poster-ai-event/build_poster.py
.venv/bin/python docs/dspro1/poster-ai-event/export_pdf.py
```

## Hinweise zum Druck

- Alle Diagramme sind Vektorgrafiken und direkt im Poster gezeichnet; Mindestschrift ≈ 16 pt, Fliesstext ≈ 21–27 pt.
- Zahlen stammen aus dem Final Report (Tabellen Modellvergleich und Preisbänder) sowie aus `src/external-sources/output_csv/` (Filterstufen, Karte).
- Für die Druckerei die Datei `…_Beschnitt3mm.pdf` verwenden. Hintergrund und dunkler Footer laufen 3 mm über das Endformat hinaus; alle Inhalte liegen mindestens 30 mm innerhalb der Schnittkante. Schnittmarken sind nicht enthalten (auf 3 mm Anschnitt haben sie keinen Platz); die TrimBox definiert das Endformat.
- Farbraum RGB (Browser-Export). Die meisten Poster-Druckereien konvertieren selbst; falls CMYK/PDF-X verlangt wird, in Acrobat oder Affinity konvertieren.
- Der QR-Code führt auf die Azure-App. Vor dem Event die App über den GitHub-Workflow starten (siehe `../AZURE.md`).
- Die App verwendet `best_model_v3.joblib` (LightGBM, RMSE 393), weil `gradient_boosting_all_geo.joblib` dort als unvollständig ausgeblendet ist. Das Poster weist das im QR-Block aus.
- Im SVG ist Barlow Condensed per `@font-face` eingebettet; Layoutprogramme ignorieren das oft. Dann `src/assets/BarlowCondensed-Bold.ttf` installieren oder das PDF verwenden.

## Autorenfotos

Oben rechts stehen zwei Karten mit Foto, Name und LinkedIn-QR-Code. Die Fotos als quadratische Bilder ablegen, danach `build_poster.py` und `export_pdf.py` erneut ausführen:

- `assets/photo_elias.jpg` (oder `.png`)
- `assets/photo_timo.jpg` (oder `.png`)

Fehlt ein Foto, erscheinen die Initialen. Für den Druck mindestens 300 × 300 px (Kreis ≈ 22 mm).
