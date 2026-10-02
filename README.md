# DSPRO2 – ML-Pipeline für Schweizer Mietpreise

> HSLU DSPRO2 HS26 · Team 6 · Elias Martinelli, Timo Schlumpf
> Umsetzung der ML-Komponente aus dem Project Proposal
> (`docs/project-proposal/`): räumliches Mixed-Effects-Boosting, Textmerkmale,
> kalibrierte Intervalle pro Wohnung, Quality-of-Life-Index und Fair-Rent-Check.

Das Repository ist ein uv-Projekt. DSPRO1 (Streamlit-Prototyp) ist abgeschlossen: Doku und
Notebook liegen in `docs/archive/dspro1/`, das ausgewählte DSPRO1-Modell als Vergleichs-Baseline in
`models/baseline/`.

## Schnellstart

```bash
uv sync --all-extras
uv run python -m ipykernel install --sys-prefix --name dspro2 --display-name "DSPRO2 (uv, Python 3.12)"
uv run jupyter lab notebooks/dspro2_ml_pipeline.ipynb   # Mietmodell (RQ1-RQ3)
uv run jupyter lab notebooks/dspro2_qoli.ipynb          # Quality-of-Life-Index (P2, RQ4)
```

Im Notebook den Kernel **DSPRO2 (uv, Python 3.12)** wählen (`--sys-prefix` registriert ihn nur in
der Projekt-`.venv`; in VS Code alternativ direkt den Interpreter `.venv` wählen). Ohne `--all-extras` fehlen PyTorch,
sentence-transformers und das Anthropic-SDK; die betroffenen Abschnitte werden dann übersprungen.

## Laufmodi (Umgebungsvariablen)

| Variable | Werte | Wirkung |
|---|---|---|
| `DSPRO2_RUN_MODE` | `full` (Standard), `fast` | `fast` = kleine Budgets für einen Probelauf |
| `DSPRO2_DATA_SOURCE` | `auto` (Standard), `db`, `csv` | `auto` nutzt die Datenbank, wenn `DATABASE_URL` gesetzt ist, sonst die DSPRO1-CSVs |
| `DATABASE_URL` | Postgres-URL | Nur über `.env` oder Umgebung, nie im Code |
| `DSPRO2_RUN_LLM` | `0` / `1` | Live-Attributextraktion mit Claude (kostet Geld, braucht `ANTHROPIC_API_KEY`) |
| `DSPRO2_FETCH_OSM` | `0` / `1` | OSM-Amenities über Overpass laden (Prototyp im ML-Notebook; das QoLI-Notebook lädt sie immer, mit Cache) |
| `DSPRO2_QOLI_REFRESH` | `0` / `1` | QoLI-Notebook: gecachte Hektar-Indikatoren neu berechnen |

Nicht-interaktiv ausführen (Probelauf):

```bash
DSPRO2_RUN_MODE=fast DSPRO2_DATA_SOURCE=csv uv run jupyter nbconvert --to notebook --execute --inplace notebooks/dspro2_ml_pipeline.ipynb
```

## Daten

- **Inserate:** Standardquelle ist die DSPRO1-Datenbank (Neon). Nur mit der Datenbank gibt es
  Beschreibungstexte (`listing_details.description`) und damit RQ1 (Textablation) und die Attributextraktion.
  Ohne `DATABASE_URL` fällt das Notebook auf den versionierten DSPRO1-Snapshot in `data/dspro1_snapshot/`
  zurück (9'405 Inserate ohne Text, Stand 13.04.2026).
- **Gemeinde-/Bezirks-/Kantonshierarchie:** swissBOUNDARIES3D 2026-01 (swisstopo, OGD), wird beim ersten Lauf
  nach `data/external/` geladen.
- **OSM-Amenities:** Overpass API, ODbL, Cache in `data/cache/osm_amenities.parquet`.
- **QoLI-Geodaten (~1.8 GB, beim ersten Lauf des QoLI-Notebooks nach `data/external/qoli/`):**
  ARE ÖV-Güteklassen 2026, BAFU sonBASE Strassen-/Bahnlärm (L<sub>r,Tag</sub>/L<sub>r,Nacht</sub>, 10 m),
  BAFU PolluMap NO₂/PM2.5 2025, MeteoSchweiz Sonnenschein-Normwert 1991–2020, BFS Arealstatistik,
  BFS Erreichbarkeit/STATPOP-Hektaren 2021 (alle data.geo.admin.ch, OGD), ESTV Steuerbelastung 2018,
  BFS Leerwohnungsziffer 2026 und City Statistics (SDMX-API stats.swiss), BFS-Gemeindemutationen.
  Quellen, Lizenzen und Bezugsjahre: `rentml.qoli_sources.SOURCES` bzw. `docs/results/qoli_sources.md`.

Lokale Daten (`data/raw`, `external`, `interim`, `cache`, `app`), trainierte Modelle (`models/*.joblib`) und
`mlruns/` sind in `.gitignore`. Versioniert sind nur der DSPRO1-Snapshot und die DSPRO1-Baseline.

## Aufbau

```
.
├── pyproject.toml, uv.lock      <- Abhängigkeiten (uv), Ruff- und pytest-Konfiguration
├── src/rentml/                  <- gesamte Logik, von Notebooks und App importiert
├── tests/                       <- pytest, deterministisch, ohne Netz/Datenbank
├── notebooks/dspro2_ml_pipeline.ipynb   <- Mietmodell, Intervalle, Fair-Rent-Check
├── notebooks/dspro2_qoli.ipynb          <- Quality-of-Life-Index nach OECD/JRC (RQ4)
├── data/dspro1_snapshot/        <- DSPRO1-CSVs (versioniert, Offline-Fallback)
├── data/ (sonst lokal)          <- raw, external, interim, cache, app/map (Karten-Bundle)
├── models/baseline/             <- DSPRO1-Baseline (GradientBoosting ALL+geo, versioniert)
├── models/ (sonst lokal)        <- RentModelBundle für die Dash-App
├── pipelines/                   <- Rust-Scraper (scrapegoat, rentables-scraper), GWR/swisstopo-Anreicherung
├── docs/                        <- Proposal, Evaluationsplan, fig/, results/, archive/dspro1/
└── mlruns/ (lokal)              <- MLflow (SQLite-Backend)
```

### Module in `src/rentml/`

| Modul | Aufgabe |
|---|---|
| `config` | Konstanten des Evaluationsplans (Seeds, Quantile, α), Pfade, `.env`-Laden |
| `data` | Laden aus Neon (`DATABASE_URL`) oder den DSPRO1-CSVs, Schema-Fixes, Audit |
| `dedup`, `cleaning` | Dublettenprüfung → `object_id`; Domänenfilter, CHF/m²-Ausreisser, Mietregime |
| `geo`, `address` | swissBOUNDARIES3D: Kanton > Bezirk > Gemeinde, Sprachregion, Adress-Parsing |
| `splits`, `evaluation` | 60/20/20-Split und CV-Folds (gruppiert); Metriken, Bootstrap, Wilcoxon, Holm, MDE |
| `features`, `models` | DSPRO1-Features, hierarchisches OOF-Target-Encoding, LightGBM, Optuna |
| `spatial`, `autocorrelation` | GPBoost (Random Effects + Vecchia-GP, zweistufig); Moran's I |
| `text`, `extraction`, `extraction_llm` | Anonymisierung, Keywords, TF-IDF, Embeddings; Attribute per Regeln oder Claude |
| `fusion` | PyTorch-Netz mit Entity Embeddings (Deep-Learning-Komponente) |
| `quantile`, `conformal` | Monotone Quantil-LightGBM (Custom Pinball); CQR, Mondrian-CQR, Coverage |
| `explain`, `rentcheck` | SHAP-Treiber; Fair-Rent-Check, What-if, `RentModelBundle` für die App |
| `qoli`, `amenities` | Quality-of-Life-Index (OECD/JRC), OSM-Erreichbarkeit, Value Score |
| `qoli_sources`, `qoli_layers`, `qoli_municipal` | QoLI-Quellenregister (Datasheet) und Downloads; Punktindikatoren aus Lärm-/Luft-/Sonnenrastern, ÖV-Güteklassen, Arealstatistik (`QoliLayers`); Leerstand, City Statistics, ESTV-Steuern mit Gemeindemutationen |
| `qoli_robustness` | Cronbach-α, Perzentil-Normierung, geometrische Aggregation, Leave-one-out, Rangvergleich, bevölkerungsgewichtete Gemeindewerte |
| `tracking`, `plotting` | MLflow (SQLite) mit JSONL-Fallback; Plot-Stil, Ablationstabelle, Markdown/LaTeX |
| `mapdata`, `mapbundle` | Kartengeometrie (LV95, Coverage-Vereinfachung, Kanton/Bezirk/Gemeinde), Aggregate, Karten-Bundle |
| `webapp` | Plotly-Dash-App; Startseite ist die Schweizer Karte |
| `baseline` | Lädt die DSPRO1-Baseline und sagt auf DSPRO2-Daten vorher (nur vollständig angereicherte Objekte) |

Abbildungen landen in `docs/fig/`, Tabellen (Ablation, Modellkarte) in `docs/results/`.
Der vorregistrierte Evaluationsplan steht in `docs/EVALUATION_PLAN.md`.

## DSPRO1-Baseline

Das in DSPRO1 ausgewählte Modell (GradientBoosting, Feature-Set `ALL+geo`, R² 0.712, MAE 281 CHF auf
dem DSPRO1-Split) bleibt für den späteren Vergleich erhalten. Details: `models/baseline/README.md`.

```python
from rentml.baseline import load_dspro1_baseline
from rentml.config import ProjectPaths

baseline = load_dspro1_baseline(ProjectPaths.discover().baseline_model)
pred_chf = baseline.predict(test_df)   # NaN für Objekte ohne volle GWR/swisstopo-Anreicherung
```

Vergleichen auf demselben DSPRO2-Testset (z. B. mit `rentml.evaluation.paired_bootstrap_mae` auf den
Objekten mit `baseline.applicable(test_df)`), nicht über die DSPRO1-Kennzahlen.

## RentLens (Dash-App)

Eine Plattform mit den drei Einstiegspunkten aus dem Proposal:

| Seite | Für | Inhalt |
|---|---|---|
| `/` Karte (Startseite) | alle | Kanton/Bezirk/Gemeinde, Median CHF/m² ab 20 Objekten, Drill-down |
| `/mietcheck` | Mietende | Fair-Rent Check: Urteil unter/im/über dem 80 %-Intervall, Marktperzentil, SHAP-Treiber, erkannte Textmerkmale, Links zu BWO und Mieterverband |
| `/vermieter` | Vermietende | Angebotsband (25.–75. Perzentil), kalibriertes Intervall, What-if für Fläche und Zimmer |

```bash
uv sync --extra app
uv run python -m rentml.mapbundle --source csv   # einmalig: Karten-Bundle nach data/app/map
uv run --extra app python -m rentml.webapp       # http://127.0.0.1:8050
```

- **Modell:** `models/rent_bundle_v1.joblib` aus dem Notebook (oder `$RENTML_MODEL_BUNDLE`, `--model`).
  Fehlt es, läuft die Karte weiter und die beiden Werkzeuge zeigen einen Hinweis.
- **Adresse → Merkmale** (`rentml.geoadmin`, `rentml.estimate`): Adresssuche, GWR-Gebäude (Baujahr,
  Wohnungen, Grundfläche), ÖV-Erreichbarkeit, Solarklasse, Hektar-Bevölkerung und Höhe live von
  api3.geo.admin.ch; Gemeinde/Bezirk/Kanton per Punkt-in-Polygon auf der Kartengeometrie; danach
  dieselbe Feature-Pipeline wie im Training (Target-Encoding, abgeleitete Merkmale). Fehlende Werte
  werden angezeigt und vom Modell als NaN behandelt.
- **Unverzerrte Karte:** gezeichnet in LV95 (EPSG:2056), nicht Web Mercator; Grenzen gemeinsam
  vereinfacht (`--tolerance`, Standard 25 m).
- **Datenschutz/Lizenz:** Eingaben werden nicht gespeichert; das Karten-Bundle enthält nur Geometrien
  und Aggregate. `data/app/` ist in `.gitignore`.
- **Noch offen gegenüber dem Proposal:** QoLI, Value Score und Reliability-Layer auf der Karte
  (über `aggregate_listings(extra_cols=...)` und `meta["extra_metrics"]` vorbereitet), What-if für
  Renovation/Balkon/Lift (braucht Textmerkmale im Modell, RQ1) und Live-LLM-Extraktion (aktuell Regeln).

## Tests und Linting

```bash
uv run --extra app pytest     # ohne das Extra werden die App-Tests übersprungen
ruff check src tests && ruff format --check src tests
```

MLflow-UI: `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
