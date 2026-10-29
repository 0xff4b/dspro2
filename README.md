# DSPRO2 – RentLens: Mietpreise in der Schweiz

> HSLU DSPRO2 HS26 · Team 6 · Elias Martinelli, Timo Schlumpf
> Umsetzung des Project Proposals (`docs/project-proposal/`): räumliches Mixed-Effects-Boosting,
> Textmerkmale, kalibrierte Intervalle pro Wohnung, Quality-of-Life-Index und Fair-Rent-Check.

DSPRO1 (Streamlit-Prototyp) ist abgeschlossen und liegt in `docs/archive/dspro1/`, das damals
gewählte Modell als Vergleichs-Baseline in `models/baseline/`.

## Schnellstart

```bash
uv sync --all-extras
uv run python -m ipykernel install --sys-prefix --name dspro2 --display-name "DSPRO2 (uv, Python 3.12)"
uv run jupyter lab notebooks/dspro2_ml_pipeline.ipynb   # Mietmodell (RQ1–RQ3), schreibt Modell und Karte für die App
uv run jupyter lab notebooks/dspro2_qoli.ipynb          # Quality-of-Life-Index (RQ4)
uv run --extra app python -m rentml.app                 # App auf http://127.0.0.1:8050
```

Ohne `--all-extras` fehlen PyTorch, sentence-transformers und das Anthropic-SDK; die betroffenen
Notebook-Abschnitte werden übersprungen.

## Laufmodi (Umgebungsvariablen)

| Variable | Werte | Wirkung |
|---|---|---|
| `DSPRO2_RUN_MODE` | `full` (Standard), `fast` | `fast` = kleine Budgets für einen Probelauf |
| `DSPRO2_DATA_SOURCE` | `auto` (Standard), `db`, `csv` | `auto` nutzt die Datenbank, wenn `DATABASE_URL` gesetzt ist, sonst den DSPRO1-Snapshot |
| `DATABASE_URL` | Postgres-URL | nur über `.env` oder Umgebung, nie im Code |
| `DSPRO2_RUN_LLM` | `0` / `1` | Attributextraktion mit Claude (kostet Geld, braucht `ANTHROPIC_API_KEY`) |
| `DSPRO2_FETCH_OSM` | `0` / `1` | OSM-Amenities über Overpass laden |
| `DSPRO2_QOLI_REFRESH` | `0` / `1` | QoLI-Hektarindikatoren neu berechnen (ca. 8 min) |

```bash
DSPRO2_RUN_MODE=fast DSPRO2_DATA_SOURCE=csv uv run jupyter nbconvert --to notebook --execute --inplace notebooks/dspro2_ml_pipeline.ipynb
```

## Daten

- **Inserate:** DSPRO1-Datenbank (Neon) mit Beschreibungstexten; ohne `DATABASE_URL` der versionierte
  Snapshot in `data/dspro1_snapshot/` (9'405 Inserate ohne Text, Stand 13.04.2026, kein RQ1).
- **Geografie:** swissBOUNDARIES3D 2026-01 (swisstopo, OGD), beim ersten Lauf nach `data/external/`.
- **QoLI-Geodaten (~1.8 GB nach `data/external/qoli/`):** ARE ÖV-Güteklassen, BAFU sonBASE Lärm,
  BAFU PolluMap, MeteoSchweiz Sonnenschein, BFS Arealstatistik und STATPOP, ESTV Steuerbelastung,
  BFS Leerwohnungsziffer und City Statistics, OSM (ODbL). Quellen und Lizenzen:
  `rentml.qoli_data.SOURCES` bzw. `docs/results/qoli_sources.md`.

Lokale Daten (`data/raw`, `external`, `interim`, `cache`, `app`), trainierte Modelle und `mlruns/`
sind in `.gitignore`.

## Aufbau

```
src/rentml/      Python-Paket, von Notebooks und App importiert
tests/           pytest, ein Testmodul pro Modul, ohne Netz und Datenbank
notebooks/       ML-Pipeline (RQ1–RQ3) und QoLI (RQ4)
docs/            Proposal, Evaluationsplan, fig/, results/, archive/dspro1/
pipelines/       Rust-Scraper und GWR-/swisstopo-Anreicherung
data/, models/   Snapshot und Baseline versioniert, sonst lokal
```

| Modul | Inhalt |
|---|---|
| `config` | Pfade und Konstanten des Evaluationsplans |
| `data` | Laden (DB/CSV), Schema, Dublettenprüfung, Cleaning, Splits, Feature-Engineering |
| `geo` | Gemeinde/Bezirk/Kanton, geo.admin-Adresssuche, GWR-Wohnungen (MADD), Kartenebenen |
| `text`, `llm` | Anonymisierung, Sprache, Keywords, Attributregeln, TF-IDF, Embeddings; Claude-Extraktion |
| `models`, `nn` | Naive Baseline, LightGBM mit Target-Encoding, Optuna, GPBoost, SHAP; PyTorch-Netz |
| `intervals` | Monotone Quantil-LightGBM, (Mondrian-)CQR, Coverage, Modell-Bundle, Fair-Rent-Check |
| `evaluation` | Metriken, Bootstrap, Wilcoxon, Holm, MDE, Moran's I, Plot-Stil, MLflow |
| `qoli`, `qoli_data` | Index nach OECD/JRC und Robustheit; Quellen, Downloads, Indikatoren |
| `app`, `app_figures` | Plotly-Dash-App |

Abbildungen landen in `docs/fig/`, Tabellen in `docs/results/`, der vorregistrierte
Evaluationsplan steht in `docs/EVALUATION_PLAN.md`.

## RentLens (Dash-App)

| Seite | Inhalt |
|---|---|
| `/mietcheck` | Fair-Rent-Check: Adresse suchen, Wohnung aus dem GWR wählen (füllt Fläche und Zimmer), Urteil unter/im/über dem 80 %-Intervall, Marktperzentil, SHAP-Treiber, erkannte Textmerkmale, QoLI der Gemeinde, Links zu BWO und Mieterverband |
| `/vermieter` | Angebotsband (25.–75. Perzentil), kalibriertes Intervall, What-if über die Fläche |
| `/karte` | Median CHF/m² pro Kanton, Bezirk oder Gemeinde (ab 20 Objekten) |

- Das ML-Notebook schreibt das Modell nach `models/rent_bundle_v1.joblib` (oder
  `$RENTML_MODEL_BUNDLE`) und die Kartenebenen nach `data/app/`; das QoLI-Notebook schreibt
  `data/app/qoli_municipalities.parquet`. Fehlt das Modell, zeigen die Werkzeuge einen Hinweis.
- Adress- und Gebäudedaten kommen live von api3.geo.admin.ch, die Wohnungen eines Gebäudes vom
  BFS-Dienst MADD (eCH-0206). Eingaben werden nicht gespeichert; die App gibt keine rechtliche
  Beurteilung (Art. 269 ff. OR).

## Tests und Linting

```bash
uv run --extra app pytest
ruff check src tests && ruff format --check src tests
```

MLflow-UI: `uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
