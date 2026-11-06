# DSPRO2 RentLens – Prompt-Plan bis zur Abgabe

Stand: 3. Oktober 2026 (Ende Semesterwoche 3, Annahme Woche 1 = 14.09.2026 – bitte bestätigen).
Grundlage: Proposal (`docs/project-proposal/*.tex`), Gap-Analyse des Repos gegen das Proposal
(8 Workstreams, Vollständigkeits-Kritiker), GitHub-Board 0xff4b/dspro2 (Projekt #2, Issues #33–#106).

## 0. Befund in zehn Punkten

1. Code-Basis ist weit: 40 Module in `src/rentml` (~13.5k LOC), 36 Testdateien (~830 Tests, alle grün), Ruff sauber, zwei Notebooks komplett ausgeführt, Dash-App mit drei Seiten, QoLI mit 8 Dimensionen.
2. **Alles lief nur im CSV-Modus** (DSPRO1-Snapshot, 9'405 Inserate, **kein Beschreibungstext**). Deshalb: RQ1 unbeantwortet, kein Attributmodell, keine Übersetzung, keine fastText-Sprach-ID, Mietregime überall `unknown`, Dedup nur attributbasiert (URL-/Text-Regeln nie gelaufen).
3. RQ2-Pilot: **LightGBM-TE schlägt GPBoost** (−8.7 CHF, CI [−15.0, −2.2]). P1 ist damit ein sauber getestetes, negatives Resultat – GPBoost wurde aber nie getunt (Konfundierung, muss vor dem konfirmatorischen Lauf im Plan stehen).
4. RQ3: Mondrian-CQR ist primär und serviert (H3a gestützt 12/12 Zellen, **H3b nicht gestützt**: 851 vs. 718 CHF Breite). Reliability-Map nur als Tabelle/PNG, nicht in der App.
5. RQ4: QoLI «teilweise» (Median-ρ 0.93/0.95, p05 0.72–0.74). Nichts davon in der App; der alte 3-Dimensionen-Prototyp im ML-Notebook überschreibt `qoli_sensitivity.md` und speist die falsche Aussage «3 of 6 dimensions» in die Modellkarte.
6. **Evaluationsplan ist Entwurf**: kein Tag `eval-plan-v1`, MDE-Tabelle leer, Trainings- und Test-Reads sind bereits passiert. Entscheidungsreihenfolge im Plan (RQ2→RQ1→RQ3) widerspricht dem Proposal (RQ2→RQ3→RQ1→Refit).
7. **Keine Infrastruktur**: kein `.github/` (CI, Deploy, Cron wurden im Restructure-Commit `2d249cb` gelöscht), kein Dockerfile, kein DVC, MLflow nur SQLite lokal, keine Registry, kein `/health`, kein WSGI.
8. **Sicherheit/Lizenz, sofort**: Neon-Passwort im Klartext in `pipelines/enrichment/*.ipynb` (getrackt, öffentlich); `data/dspro1_snapshot/final_listings.csv` mit Adressen + Preisen und Roh-HTML des Scrapers getrackt – widerspricht Proposal («no listing text, price or address is redistributed»); Scraper rotiert User-Agents und nutzt Proxy-Pool – widerspricht «technical blocks are not circumvented».
9. Git-Historie wurde am 2.10. 22:44 umgeschrieben und gepusht (lokal = origin, ok), aber `model_card.md` und MLflow-Tags referenzieren den unerreichbaren Commit `8a389b0`; `.git/refs/original` liegt noch da. Lokale Altlasten: `dspro2/` 6.8 GB, `.venv-dspro1/` 2.5 GB, `tmp/`, `.azure-local/` (Azure-Token-Cache).
10. Kanban: 74 Issues, Labels, Milestones existieren, aber Board ≠ Realität (fertige Arbeit steht in Backlog), keine MoSCoW-Labels, null PRs, keine Reviews durch Timo.

## 1. Modell-, Effort- und Modus-Regeln

| Aufgabe | Modell | Effort | Modus |
|---|---|---|---|
| Wissenschaftliche/architektonische Entscheide (Evaluationsplan, Statistik, Scraper-Design, Attributmodell, Refactoring, Datasheet, Risiko, Report) | **Fable 5.1** | `/effort xhigh` (Report-Entwurf, Attributmodell: `max`) | Plan-Mode zuerst (`/plan` oder Shift+Tab), bei L/XL zusätzlich Schlüsselwort `ultracode` |
| Gut spezifizierte Implementierung (CI, Docker, Migrationen, Dash-Seiten, Tests, Doku) | **Opus 5.5** | `/effort xhigh` (**Opus-Default ist `medium` – immer explizit setzen**) | Plan-Mode bei L |
| Kleine mechanische Aufgaben (S) | Opus 5.5 | `/effort high` | direkt |
| Review nach jedem L/XL-Prompt | – | – | `/code-review high`; für die 3–4 grössten PRs `/code-review ultra` (3 Gratis-Läufe, danach Credits) |

Fakten (Doku-geprüft): Effort-Stufen `low/medium/high/xhigh/max`, gesetzt mit `/effort <stufe>`; `ultracode` aktivierst du mit `/effort ultracode` oder dem Wort `ultracode` im Prompt (Multi-Agent-Workflows für substanzielle Aufgaben – das verbraucht dein 5-Stunden-Fenster schnell: XL-Prompts am Anfang eines frischen Fensters starten). Fable 5.1 kostet 2.5× Opus 5.5 und ist für langlaufende, schwierige Arbeit gedacht; Opus 5.5 ist der Standard. Modellwechsel: `/model fable` bzw. `/model opus`.

## 2. Arbeitsablauf pro Prompt

1. Neue Session (oder `/clear`) – die Notebooks sind riesig, alter Kontext stört.
2. `/model …`, `/effort …`, bei L/XL `/plan` → Plan lesen, freigeben.
3. Branch pro Prompt (`git switch -c <issue-nr>-<kurz>`); Claude committet nur auf dein «commit».
4. Nach dem Prompt: `/code-review high` (oder `ultra`), dann `gh pr create`, Review durch Timo, Issue schliessen, Board-Spalte setzen.
5. Hinweis an Claude, wo nötig: Volllauf des ML-Notebooks dauert >30 min (nur nach Rückfrage), QoLI-Notebook lädt 1.8 GB Geodaten; `.env` nie ausgeben.

## 3. Was nur du liefern/entscheiden kannst (vorher klären)

- **Neon**: Passwort `neondb_owner` in der Neon-Console rotieren (sofort, es ist öffentlich), neue URL nur in `.env` (`DATABASE_URL=…`); Schreibrechte für Migrationen; wer ist Owner (Timo/0xff4b)?
- Entscheid öffentliches Repo vs. privat; Umgang mit CSV-Adressen/Preisen (A2).
- Cloudflare-R2-Bucket + Keys (DVC-Remote, MLflow-Artefakte) oder anderer S3-kompatibler Speicher.
- `ANTHROPIC_API_KEY` + Budget für ~2'000 Batch-Labels; Übersetzungsanbieter (DeepL/Google/LLM) + Budget.
- GPU (RTX-2000-Workstation mit CUDA in WSL2) für Attributmodell/Embeddings; HSLU-GPU/Colab als Backup.
- Azure-Subscription, OIDC/Service-Principal für GitHub Actions, GHCR-Rechte; HF-Account als Fallback.
- Semesterkalender HS26 (Woche 1 = 14.09.?), exaktes Freeze-Datum (Ende Woche 8 ≈ 06.–08.11.).
- Mit Dr. Nazarenko: Scraping-Quellen/Grauzone, negatives P1-Resultat, Entscheidungsreihenfolge RQ2→RQ3→RQ1, Datenexport-Kontakte, BILANZ/IAZI.
- Mit Timo: Rust vs. Python für Scraper v2, Hälfte der ~300 Hand-Labels, PR-Reviews, Rollen für Canvas.
- Formularpflicht-Kantone: Claude recherchiert, **du** verifizierst die Rechtsliste.

## 4. Kalender und Blöcke

| Block | Fenster | Inhalt | Deadline |
|---|---|---|---|
| A Sofort | W3–W4 (03.–11.10.) | Secrets, Hygiene, CLAUDE.md/Kanban, CI, DB-Export, Compliance, **Evaluationsplan + Tag** | Milestone Setup 09.10. |
| B Daten | W4–W8 | Schema v2, Scraper v2, Pilot (19.10.–01.11.), Enrichment, DVC/MLflow, Dedup-Validierung, Freeze | Freeze 06.–08.11., Datasheet 06.11. |
| C Modelle | W4–W9 | Baselines/RQ2-Härtung, RQ3-Härtung, Refactor + Regressionstest, konfirmatorischer Lauf nach Freeze | Model dev 29.11., Eval 06.12. |
| D Text | W4–W11 | Anonymisierer/LID/Übersetzung, Gold-Labels, LLM-Seed, Attributmodell, RQ1, Text in App | Fusion-Timebox 29.11. |
| E QoLI | W6–W9 | Konsolidierung, Service + Vermieter-View, Kartenlayer | – |
| F App/Deploy | W7–W12 | WSGI/Docker, Azure/GHCR/HF, App-Features | App 11.12. |
| G Doku | laufend | Mid-term 22.10., Datasheet 06.11., Risiko, Report-Skeleton, Canvas, Report-Draft 04.12., Final-Deck 17.12., Abgabe 31.12. | |

Kritischer Pfad: A1 → A5 (DATABASE_URL) → A7 (Tag) → B1 → B2 → B3 (Pilot muss 19.10. starten) → Freeze → C4 → D5/D6 → G6.

---

## 5. Die Prompts

Jeder Prompt ist eigenständig. Kopf jedes Prompts (nach A3 übernimmt das die Projekt-`CLAUDE.md`):

```text
Kontext: Repo ~/code/Elias-Martinelli/dspro2 = HSLU DSPRO2 «RentLens» (Team 6, Remote 0xff4b/dspro2, öffentlich).
Massgeblich: docs/project-proposal/DSPRO2_ProjectProposal_Team6_RentLens.tex, docs/EVALUATION_PLAN.md, README.md.
Regeln: uv only; nach Änderungen `uv run --extra app pytest` und `ruff check . && ruff format --check .`; nichts committen ohne mein «commit»; .env nie lesen oder ausgeben; Notebook-Vollläufe (>30 min) nur nach Rückfrage; docs/results/ und docs/fig/ sind generierte Dateien (nie von Hand editieren).
```

---

### Block A – Sofort (W3–W4)

#### A1 · Secrets entfernen, Historie aufräumen `[S] Opus 5.5 · /effort high`
Schliesst: Teil von #105. Voraussetzung: du rotierst das Neon-Passwort parallel in der Console.

```text
Sicherheitsaufgabe, read-first. (1) In pipelines/enrichment/final_records.ipynb, gwr_egid_db_sync.ipynb und
swisstopo_enrich_db_sync.ipynb steht ein Neon-Connection-String mit Passwort als os.environ-Fallback. Ersetze jeden
Fallback durch os.environ["DATABASE_URL"] ohne Default (KeyError mit klarer Meldung), ohne das Passwort je in deine
Ausgabe zu schreiben. Prüfe mit `git log -S'neondb_owner' --oneline`, seit wann es in der Historie liegt, und
fasse zusammen (ohne Secret). (2) docs/archive/dspro1/AZURE.md enthält eine Azure-Subscription-ID: durch
Platzhalter `<AZURE_SUBSCRIPTION_ID>` ersetzen. (3) Richte Secret-Scanning ein: .pre-commit-config.yaml mit
gitleaks (plus ruff, ruff-format, check-added-large-files, forbid .env), `uv add --dev pre-commit`, `uv run
pre-commit run --all-files` grün. (4) Die Historie wurde am 02.10. per filter-branch umgeschrieben und ist
gepusht; .git/refs/original existiert noch, docs/results/model_card.md Zeile 3 und MLflow-Tags referenzieren den
unerreichbaren Commit 8a389b0. Analysiere Zustand (reflog, refs/original, dangling), erkläre Folgen und schlage
Aufräumschritte vor (refs/original löschen, gc) – FRAGE vor jedem destruktiven Schritt. Ergebnis: Diff, Secret-Scan
grün, kurzes docs/SECURITY.md (Rotation erledigt am <Datum>, Regeln), Commit-Vorschlag.
```

#### A2 · Datenweitergabe, Lizenzen, Repo-Hygiene `[M] Opus 5.5 · /effort high`
Schliesst: Teil von #105. Voraussetzung: dein Entscheid öffentlich/privat.

```text
Lizenz- und Hygiene-Aufgabe. Das Proposal (Abschnitt 2, «Quality constraints») verspricht: kein Inseratetext, Preis
oder Adresse wird weitergegeben. Getrackt sind aber data/dspro1_snapshot/final_listings.csv (address, price_cold,
EGID, Koordinaten), model_wide.csv (price_cold + LV95) sowie pipelines/rentables-scraper/{output.html,
output_listings.html, element.html, proxies.txt, user_agents.txt}. (1) Erstelle eine Entscheidungsvorlage mit drei
Optionen (Repo privat; CSVs via DVC/R2 statt Git, `git rm --cached`, Historie bleibt; Adressspalte entfernen und
rentml.data CSV-Fallback anpassen) mit Konsequenzen für README/Datasheet – ich entscheide. (2) Unabhängig davon:
Roh-HTML, proxies.txt, user_agents.txt aus Git entfernen (vorher die HTML-Dateien als Test-Fixtures nach
tests/fixtures/rentumo/ kopieren und dort behalten – sie werden für den Scraper-Port gebraucht), .gitignore
ergänzen. (3) LICENSE (MIT, Copyright Team 6 2026) und docs/DATA_LICENSES.md (rentumo: keine Lizenz, nur Training;
swisstopo/BFS/BAFU/ARE/ESTV OGD mit Quellenangabe; OSM ODbL inkl. Share-alike-Hinweis für den QoLI-Parquet;
MeteoSchweiz) generiert aus rentml.qoli_sources + einer neuen Tabelle in rentml.data_sources. (4) Lokale
Altlasten: liste dspro2/ (6.8 GB, enthält gpb_models.joblib und mlruns – prüfe, ob darin Runs fehlen, die ./mlruns
nicht hat), .venv-dspro1/, tmp/, .azure-local/ (Azure-Token-Cache, root-owned), models/rent_predictor_streamlit.joblib
mit Grösse und Inhalt auf; lösche NUR nach meiner expliziten Freigabe pro Pfad. (5) README-Abschnitt «Lizenzen».
```

#### A3 · Projekt-CLAUDE.md, CONTRIBUTING, PR-Template, Kanban auf Realität bringen `[M] Opus 5.5 · /effort high`
Schliesst: #40, Teil von #34/#35; braucht `gh` (angemeldet).

```text
(1) Schreibe CLAUDE.md im Repo-Root (max. 70 Zeilen, Englisch) mit den nicht-offensichtlichen Konventionen:
uv only; Tests/Ruff-Befehle; Laufmodi DSPRO2_RUN_MODE/DSPRO2_DATA_SOURCE/DSPRO2_RUN_LLM/DSPRO2_FETCH_OSM;
docs/results und docs/fig sind generiert («no number typed by hand»); Volllauf >30 min nur nach Rückfrage; QoLI-
Notebook 1.8 GB Download; Text nur aus der DB; .env nie lesen; keine Inseratdaten in Git; Issue→Branch→PR→Review
durch Timo; Deutsch für Doku/README, Englisch für Code/Report; Evaluationsplan-Konstanten in rentml.config nicht
ohne Changelog ändern. Jeden genannten Befehl mit --help oder Datei verifizieren. (2) CONTRIBUTING.md mit der
Definition of Done aus Proposal §4 (CI grün, MLflow geloggt gemäss Plan, README/Datasheet/Modellkarte aktualisiert)
und .github/PULL_REQUEST_TEMPLATE.md als Checkliste; .github/ISSUE_TEMPLATE/task.md. (3) Kanban mit gh
(Repo 0xff4b/dspro2, Projekt #2 von 0xff4b): Labels must/should/could/wont anlegen und gemäss Proposal auf alle
dspro2-Issues setzen (core = must: Baseline, P1 GPBoost, P2 QoLI, P3 Intervalle, Daten/Plan; secondary = should:
Text, Value Score, App-Views; optional = could: Karte, Mondrian-Erweiterung, Fusion, SWISSIMAGE, RQ5); Label-
Beschreibungen korrigieren (spatial-model = P1, qol-index = P2, uncertainty = P3). Issues, deren Arbeit im Repo
nachweislich existiert, mit Kommentar «implemented in <Modul/Notebook-Abschnitt>, CSV mode» in Status Review setzen
(nicht schliessen): #41 #45 #46 #47 #49 #50 #54 #55 #57 #58 #62 #63 #64 #66 #71 #72 #73 #74 #75 #77 #78 #79
#80 #81 #82 #83 #84 #85 #87 #88 #89 #92 #96 #97 #106. Fehlende Issues anlegen: Secrets/Hygiene (A1/A2),
Schema v2, Scraper v2, Enrichment-Modul, Attributmodell, QoLI-Service, Reliability-Layer, Formularpflicht,
Risiko-Dokument, Meeting-Log. Erst Trockenlauf-Liste zeigen, dann nach meinem OK ausführen. (4) docs/PROJECT_PLAN.md:
Wochenkalender HS26 (W1 = 14.09.2026, von mir zu bestätigen) ↔ Proposal-Wochen ↔ Milestone-Daten; docs/meetings/
mit Template und Rückerfassung der Supervisor-Entscheide (Prioritäten-Umkehr, fünf Supervisor-Punkte).
```

#### A4 · CI, pre-commit, Test-Infrastruktur `[M] Opus 5.5 · /effort xhigh`
Schliesst: #43. Voraussetzung: Actions im Repo aktiv (Owner 0xff4b).

```text
Baue die CI aus Proposal §3 auf (es gibt kein .github/). (1) .github/workflows/ci.yml: ubuntu-latest, Python 3.12,
astral-sh/setup-uv mit Cache, `uv lock --check`, `uv sync --frozen --extra app --group dev`, `uv run ruff check .`,
`uv run ruff format --check .`, `uv run pytest -m "not slow and not network"` mit Coverage-Artefakt; Job-Laufzeit
<10 min; gpboost-Wheel-Installation prüfen (1.7.4, cmake nötig?). (2) Ruff an der Wurzel sauber machen: [tool.ruff]
extend-exclude = ["docs/archive"], per-file-ignores für Notebooks, die 17 E501 in notebooks/*.ipynb beheben,
pipelines/ fixen oder begründet ausschliessen. (3) pytest: Marker slow/network/gpu/db registrieren, Dev-Deps
pytest-socket, pytest-timeout, pytest-cov; addopts mit --disable-socket --allow-unix-socket --timeout=300
--durations=15; prüfen, dass MLflow-SQLite-Tests weiter laufen. (4) Zweiter Workflow nightly.yml (schedule +
workflow_dispatch): Trainings-Smoke `DSPRO2_RUN_MODE=fast DSPRO2_DATA_SOURCE=csv uv run jupyter nbconvert --execute
notebooks/dspro2_ml_pipeline.ipynb` in eine tmp-Kopie, swissBOUNDARIES3D-Download (74 MB) gecacht, ausgeführtes
Notebook als Artefakt; Laufzeit des fast-Modus messen und in README dokumentieren. (5) README-Badge, CONTRIBUTING
ergänzen. Beweise: Workflow-Datei lokal mit `act` nicht nötig – zeige mir stattdessen `uv run pytest` lokal grün und
die geplante Laufzeit-Schätzung.
```

#### A5 · DB-Export und Datenbank-Qualitätsbericht `[M] Opus 5.5 · /effort xhigh`
Schliesst: #44, #45 (DB-Teil), Teil von #106. Voraussetzung: rotiertes `DATABASE_URL` in `.env` (read-only reicht).

```text
Erster Lauf gegen die echte Neon-Datenbank (DATABASE_URL liegt in .env – nie ausgeben). Ziel ist der im Proposal
und in docs/DATENBESCHAFFUNG.md geforderte, nur lesende Qualitätsbericht. (1) Lade mit rentml.data.load_listings
(DSPRO2_DATA_SOURCE=db) und cache anonymisiert nach data/interim/listings_db.parquet; prüfe, ob LISTINGS_SQL zum
realen Schema passt (Tabellen swisstopo_details/listing_buildings/gwr_buildings haben keine DDL im Repo) und
dokumentiere Abweichungen. (2) Neues Modul src/rentml/db_audit.py (+ Tests mit synthetischem Frame), das
docs/results/db_audit.md rendert: Zeilen 9'994 vs. 9'405 vs. 8'298 erklärt; Abdeckung description (Anteil, Länge,
Median, Anteil ≥50 Zeichen), Sprachanteile de/fr/it/unknown, aux_costs/available_from/Kaution-Abdeckung,
Halbzimmer (listing_details.rooms wurde per Migration 20260413141619 auf INTEGER gecastet – wie viele 0/NULL?),
Preis-Leak-Rate der anonymisierten Texte (Ziel <1 %, 50 geflaggte Beispiele stichprobenartig prüfen), exakte
Duplikate nach listing_id UND URL/slug sowie Near-Duplikate nach Beschreibungs-Hash (erstmals möglich), Zahlen
vorher/nachher je Dedup-Regel. (3) Kapazität: pg_total_relation_size je Tabelle, Hochrechnung auf 20'000 Objekte ×
3 Beobachtungen + Texte + Übersetzungen → docs/results/db_capacity.md mit Free-Tier-Vergleich. (4) HAS_TEXT-Entscheid
protokollieren; Notebook-Abschnitte 2–4 und 7 im fast-Modus mit db laufen lassen (nur diese Abschnitte, nicht das
ganze Notebook) und cleaning_log.md/data_sources.md aus dem DB-Lauf regenerieren. Kein Rohtext darf
unanonymisiert auf Platte landen.
```

#### A6 · Compliance-Register, Outreach, externe Daten `[M] Opus 5.5 · /effort high`
Schliesst: #51 (Entwürfe), Vorarbeit #52. Nutzt WebFetch für robots.txt/AGB.

```text
(1) Neues Modul src/rentml/acquisition/sources.py mit einer SourceSpec-ähnlichen Dataclass (name, base_url,
robots_url, robots_checked_on, disallow_hits für Listing-Pfade, crawl_delay, terms_url, terms_checked_on,
terms_excerpt, decision allowed/excluded/pending, reason) + Renderer nach docs/results/acquisition_sources.md
(Tests). Kandidaten: rentumo.ch (bestehend), flatfox.ch, newhome.ch, 2–3 grosse Verwaltungen (Vorschläge machen);
ausgeschlossen laut Proposal/DATENBESCHAFFUNG: comparis.ch, homegate/immoscout (SMG-AGB). Hole robots.txt und
AGB-Seiten live (WebFetch), zitiere die relevanten Passagen mit Datum, erfinde nichts. (2) Scraper-Ist-Zustand
dokumentieren und bereinigen: pipelines/scrapegoat rotiert User-Agents (useragent.rs) und nutzt einen Proxy-Pool
(pool.rs/proxy.rs) – das widerspricht «technical blocks are not circumvented». Deaktiviere beides per Default
(fester, ehrlicher User-Agent mit Kontakt-URL), begründe im Code-Kommentar und im Register. (3) Entwürfe (Deutsch,
kurz) für: Anfrage an HSLU-Forschungskontakte via Betreuung, Anfrage an 2–3 Verwaltungen (Nettomiete, NK, Fläche,
Dezimalzimmer, Beschreibung, ID, Zeitbezug, Nutzungsumfang), Anfrage an IAZI/BILANZ für das Gemeinderanking mit
BFS-Nummern. Ablage docs/outreach/ (ohne Namen/Mailadressen). (4) rentml.qoli_municipal.load_external_ranking(csv)
mit Test auf synthetischer Datei, damit das Ranking später als Validitätskriterium läuft. (5) Gesprächsleitfaden
für Dr. Nazarenko (Quellen/Grauzone, Pilot-Zeitplan) in docs/meetings/.
```

#### A7 · Evaluationsplan finalisieren und `eval-plan-v1` vorbereiten `[M] Fable 5.1 · /effort xhigh · Plan-Mode`
Schliesst: #48. **Vor dem ersten Training auf DB-Daten.** Du setzt den Tag selbst.

```text
Du bist methodischer Reviewer und Autor zugleich. Bringe docs/EVALUATION_PLAN.md vom Entwurf zur taggbaren
Pre-Registrierung v1. Lies zuerst Proposal §2 «Pre-registered evaluation plan», den Plan, src/rentml/config.py,
docs/results/mde_table.md, rq2_*.md, rq3_intervals.md, qoli_*.md und Notebook-Abschnitte 11, 15, 19, 25 sowie das
QoLI-Notebook Abschnitte 11–12. Dann: (1) Rahmung ehrlich festhalten: der CSV-Lauf (Session 20261002-221320, 26
Runs inkl. Test-Reads) ist ein Entwicklungs-Pilot auf dem DSPRO1-Snapshot; konfirmatorisch ist der Lauf auf dem
eingefrorenen DB-Export mit eigenem Split nach dem Tag. (2) Entscheidungsreihenfolge: Proposal sagt RQ2 → RQ3
(Intervalle auf dem Gewinner) → RQ1 → Refit, der Plan §5 sagt RQ2 → RQ1 → RQ3. Stelle die Proposal-Reihenfolge
wieder her oder begründe die Abweichung im Changelog; passe Notebook-Reihenfolge (Abschnitte 17–19) entsprechend an
oder beschreibe die nötige Umstellung. (3) MDE-Tabelle für 2'000–4'000 (Proposal) plus Hinweis 1'625 realisiert und
Proxy-Herkunft (LightGBM-Fehler, ρ = 0.87) eintragen. (4) GPBoost-Tuning-Budget pre-registrieren (Optuna-Trials,
Baumparameter bei fixierten Kovarianzparametern, Grouped-CV, num_threads = 1 für den finalen Fit) und die Ablation
«GPBoost + Koordinaten in den Bäumen» als Sensitivität, nicht als Korrektur, kennzeichnen; Kovarianz-Reuse in CV
offenlegen. (5) RQ3: H3b-Sekundärkriterien (Median-Breite, Interval Score, Breite je Band), H3a ohne
Multiplizitätskorrektur + Null-Simulation, Serving-Regel (App serviert immer Quantil + Mondrian, auch wenn GPBoost
gewinnt), Fallback-Reporting kleiner it-Zellen. (6) RQ4-Regel (Median-ρ ≥ 0.90 und p05 ≥ 0.80, Konzentration 10,
500 Draws) und Kriterienliste mit Vorzeichen aus dem QoLI-Notebook übernehmen – als post hoc markieren. (7)
FREEZE_DATE als Konstante (Ende Woche 8 HS26, von mir zu bestätigen), temporaler Drift-Test, Monotonie-Abweichung
(area_per_room beschränkt Zimmer indirekt, Proposal: rooms unconstrained) als Proposal-Abweichung, neue Baseline =
DSPRO1-GB-Rezept neu trainiert auf DSPRO2-Daten als eigene Zeile, CHF/m²-Alternativziel (Sensitivitätszeile oder
dokumentierter Verzicht), Feature-Typen property type/aux costs/hectare IDs und STATENT (nutzen oder Verzicht),
Volumenziel 20'000 (DATENBESCHAFFUNG sagt 30'000). (8) Alle pre-registrierten Zahlen nach rentml.config
(OPTUNA_TRIALS, CV_FOLDS, GPBOOST_*, MORAN_K/PERMUTATIONS, SPLIT_FRACTIONS, BUCKET_EDGES, POWER_SIMS, FREEZE_DATE,
QoLI-Regel) + tests/test_eval_plan.py, der den Plan-Markdown gegen config parst. (9) Kopf mit Version/Datum/Commit,
CHANGELOG.md (Keep a Changelog) im Root, rentml.tracking loggt eval_plan_version via `git describe --tags --always`.
Liefere mir am Ende den exakten Befehl für den annotierten Tag; ich führe ihn aus.
```

---

### Block B – Daten-Pipeline (W4–W8)

#### B1 · Datenbank-Schema v2, Backfill, Integrationstest `[L] Fable 5.1 · /effort xhigh · Plan-Mode · ultracode`
Schliesst: #46. Voraussetzung: A5, Schreibrecht auf Neon, Docker Desktop für lokalen Postgres.

```text
ultracode. Entwirf und implementiere Schema v2 der Neon-Datenbank gemäss Proposal §3 («The schema is extended …
before new imports») und §2 Schema-Fixes. Lies zuerst pipelines/rentables-scraper/migrations/*.sql,
src/data/{db.rs,mod.rs}, src/rentml/data.py (LISTINGS_SQL, CANONICAL_COLUMNS) und docs/results/db_audit.md.
Anforderungen: sources(id, name, base_url, terms_url, robots_checked_at); listings bekommt source_id, url,
raw_hash, object_id (nullable, aus assign_object_ids), rent_kind ('net'|'gross'|'unknown'), rooms NUMERIC(3,1)
(Rücknahme der INTEGER-Migration, ohne Datenverlust); observations(listing_id, observed_at, price, aux_costs,
available_from, status) mit first_seen/last_seen-Views; listing_text(listing_id, description_anon, text_lang,
lang_confidence, description_de, translation_provider, translation_version, anonymizer_version); extraction_cache
und translation_cache (Proposal: Cache in Neon, nicht nur lokal); DDL für swisstopo_details/listing_buildings/
gwr_buildings (bisher ohne DDL im Repo). Umsetzung als idempotente SQL-Migrationen in db/migrations/ mit kleinem
Runner `uv run python -m rentml.db migrate --dry-run/--apply` (SQLAlchemy Core), die auch sqlx-kompatibel bleiben
(Timo's Rust-Scraper). rentml/backfill.py: observed_at = 2026-04-13 für den Bestand, object_id aus
assign_object_ids, description_anon/text_lang via _anonymize/text, Halbzimmer aus Text (recover_rooms_from_text) nur
in neue Spalten; --dry-run-Report, dann --write. load_from_db liest v2 (eine Zeile pro Beobachtung oder jüngste
Beobachtung, parametrisiert). tests/integration/test_db_roundtrip.py mit Marker db (Skip ohne TEST_DATABASE_URL),
Fixture spielt Migrationen auf einem Docker-Postgres ein. README-Abschnitt «Datenbankschema» mit ER-Skizze
(Mermaid). Risiken: Neon-Speicherlimit (siehe db_capacity.md), Rust-Scraper-Kompatibilität – beides im Plan adressieren.
```

#### B2 · Inkrementeller Scraper v2 `[XL] Fable 5.1 · /effort xhigh · Plan-Mode · ultracode`
Schliesst: Vorarbeit #52, #56, #57. Voraussetzung: B1, A6 (Register), Entscheid Python vs. Rust mit Timo.

```text
ultracode. Baue den inkrementellen Scraper aus Proposal §2 («incremental scraper … robots.txt and rate limits are
respected, technical blocks are not circumvented … observation timestamps and object identity … anonymised at
ingestion») als Python-Paket src/rentml/acquisition/ (httpx + selectolax; falls das Team Rust behält, liefere
stattdessen eine Spezifikation + Fixtures für Timo). Lies zuerst pipelines/rentables-scraper/src/{main.rs,
detail_parser.rs,data/db.rs} und die Fixtures in tests/fixtures/rentumo/ (A2). Design: BaseAdapter (list_pages,
parse_card, parse_detail), RentumoAdapter als Port (JSON-LD/Detail-Felder), AdapterRegistry aus
acquisition/sources.py (nur decision == allowed), robots.txt-Parser + Crawl-Delay + globaler Rate-Limiter, fester
User-Agent mit Kontakt-URL, kein Proxy, keine UA-Rotation, Stopp bei 403/429 (Quelle pausieren, nicht umgehen).
Ingestion nach Schema v2: Upsert listings (ON CONFLICT url), neue observations je Lauf, rooms als Decimal (3.5!),
rent_kind aus Labels (Kaltmiete/Nettomiete vs. Bruttomiete), aux_costs, available_from, raw_hash; Text sofort
anonymisieren (_anonymize), Sprache taggen, Original nie speichern (nur description_anon); Dedup-Vorschlag per
assign_object_ids als Batch-Schritt. CLI `uv run rentml-scrape --source rentumo --max-pages N --dry-run` und
`rentml-scrape report`. Tests nur mit Fixtures (kein Netz): zwei Läufe hintereinander → keine neuen listings, aber
neue observations; 3.5-Zimmer-Fixture → 3.5; Brutto-Fixture → gross; price_leak_rate < 1 %. Logging mit
Lauf-Statistik (neu/geändert/unverändert je Quelle). Abschliessend: Plan für den zweiwöchigen Pilot (B3).
```

#### B3 · Pilot-Scheduling und Wochenreport `[M] Opus 5.5 · /effort xhigh`
Schliesst: #52. Voraussetzung: B2; `DATABASE_URL` als Actions-Secret. Pilot läuft 19.10.–01.11.

```text
Richte den zweiwöchigen Scraper-Pilot aus Proposal §2 ein. (1) .github/workflows/scrape.yml: schedule (täglich
03:00 UTC) + workflow_dispatch, concurrency-Gruppe, timeout 45 min, Kill-Switch über Repository-Variable
SCRAPE_ENABLED, Secret DATABASE_URL, `uv run rentml-scrape --all-allowed`, Lauf-Statistik als Step-Summary und
Artefakt; Fallback-Anleitung für lokalen cron/WSL-Task, falls Runner-IPs blockiert werden. (2)
src/rentml/acquisition/report.py: aus den observations den Nettozuwachs eindeutiger Objekte je Woche, Quelle und
Kanton (assign_object_ids über den DB-Bestand), Vollständigkeitsquote (Fläche, Zimmer, Nettomiete, Beschreibung,
Koordinaten), Hochrechnung bis zum Freeze-Datum; Rendering nach docs/results/pilot_week_<n>.md; Tests mit
synthetischen Beobachtungen. (3) Wöchentlicher Workflow-Job, der den Report als Artefakt und als Issue-Kommentar
(#52) ablegt. (4) docs/PROJECT_PLAN.md: Pilotfenster und Entscheidungstermin «Volumenziel fixieren» eintragen.
```

#### B4 · Batch-Enrichment-Modul und 49 %-Lücke `[L] Opus 5.5 · /effort xhigh · Plan-Mode`
Schliesst: #54, #53 (Register-Teil). Voraussetzung: B1.

```text
Ersetze die DSPRO1-Notebooks pipelines/enrichment/{gwr_egid_db_sync,swisstopo_enrich_db_sync,final_records}.ipynb
durch ein getestetes Modul src/rentml/enrich.py mit CLI `uv run rentml-enrich --pending --dry-run`. Lies die drei
Notebooks und src/rentml/geoadmin.py (Einzelabfragen für die App) zuerst. Pipeline: Adresse → SearchServer (LV95,
EGID) → GWR-Gebäude (gbauj, gbaup als Fallback-Periode für year_built mit Flag, ganzwhg, garea) → ÖV-Güteklasse,
Solarklasse, Höhe, Hektar-Bevölkerung; Session mit Retry/Backoff, Rate-Limit (fair use api3.geo.admin.ch; MADD
20 req/min falls genutzt), Cache je Adresse/EGID in data/cache/enrichment.parquet, idempotente Upserts in die
Enrichment-Tabellen (Schema v2). Diagnose der 49 %-Lücke aus docs/results/… und cumulative_filters.csv: 3'190 EGIDs
ohne gbauj, 552 ohne Solar, 155 ohne Adresse → docs/results/enrichment_gap.md vorher/nachher; Re-Enrichment der
4'883 unvollständigen DSPRO1-Zeilen (nach meiner Freigabe für Schreibzugriff). Tests mit Fake-Session (Fixtures aus
echten api3-Antworten, anonymisiert). Ziel: Anteil Objekte mit year_built/oev/solar > 75 %; MLflow-Run
stage='data' mit Coverage-Metriken.
```

#### B5 · Versionierung und Tracking: DVC + R2, MLflow Neon/R2, Registry, Provenienz `[L] Opus 5.5 · /effort xhigh`
Schliesst: #41, #42. Voraussetzung: R2-Bucket + Keys, Neon-Schema «mlflow».

```text
Infrastruktur aus Proposal §3/§5. (1) DVC: `uv add --dev "dvc[s3]"`, dvc init, Remote R2 (endpointurl explizit),
`dvc add` für data/interim/*.parquet, data/app/, models/*.joblib, data/external/qoli (prüfe: nur abgeleitete Dateien
oder auch 1.8 GB Rohraster?), optional data/dspro1_snapshot gemäss Entscheid A2; README «Daten holen» (`dvc pull`);
CI holt nur, was der Smoke braucht. (2) rentml.tracking: Tracker.from_env liest MLFLOW_TRACKING_URI (Neon-Postgres)
und MLFLOW_ARTIFACT_ROOT (s3://… mit AWS_ENDPOINT_URL), boto3 als Dependency, `mlflow db upgrade`-Hinweis; ohne Env
bleibt SQLite/JSONL identisch. Pflicht-Tags/Params für jeden Run gemäss Proposal §5: seed, eval_plan_version
(git describe), split_sha256 + Fold-IDs als Artefakt, dedup_version (Hash der RULES), data_fingerprint (DVC-Hash),
rentml_version, num_threads, run_mode, data_source, git_dirty; Option, einen full-Lauf bei dirty tree abzubrechen.
(3) Model Registry: Bundle nach dem Notebook als Modellversion «rentlens-rent-bundle» registrieren (pyfunc-Wrapper
oder Artefakt-URI), Alias production; src/rentml/promote.py (`uv run rentml-promote --alias production` lädt das
Bundle für Docker/CI); App zeigt Version. Beide Notebooks übergeben tracking_uri/Tags. Tests mit temporärem
SQLite-Backend; Secrets nie in Notebook-Outputs.
```

#### B6 · Cross-Portal-Dedup validieren `[M] Opus 5.5 · /effort high`
Schliesst: #56. Voraussetzung: Pilotdaten aus ≥2 Quellen; 1–2 h Labelzeit (beide).

```text
Validiere die Dedup-Regeln aus Proposal §2 (PLZ + normalisierte Strasse, Zimmer, Fläche ±3 m², Preis ±10 %,
Beschreibungs-Hash, Koordinaten) auf den Mehrquellen-Daten des Pilots. Erzeuge je Regel 100 Kandidatenpaare aus
assign_object_ids (src/rentml/dedup.py, _dedup_rules.py) als Label-CSV data/interim/dedup_gold.csv (DVC, nicht Git)
mit den anonymisierten Feldern; nach unserem Labeln: Präzision je Regel (Ziel ≥ 0.95), transitive Komponenten-
Grössen, desc_max_group-Sensitivität, Verwaltungs-Boilerplate-Fälle; Zahlen Listings → Objekte je Quelle und gesamt
nach docs/results/dedup_report.md (Proposal: «numbers before and after deduplication are reported»). Toleranzen nur
mit Changelog-Eintrag in EVALUATION_PLAN.md ändern. Tests für neue Hilfsfunktionen.
```

#### B7 · Data Freeze, Drift-Set, Snapshot `[M] Opus 5.5 · /effort high`
Schliesst: #59. Nach dem Freeze-Datum (≈06.–08.11.); Voraussetzung: B5.

```text
Setze den Data Freeze aus Proposal §2 um. (1) Eingefrorener DB-Export: data/interim/listings_db_freeze.parquet aus
observations ≤ FREEZE_DATE (rentml.config), `dvc add`, Git-Tag-Vorschlag data-freeze-v1 (ich tagge), DVC-Hash als
MLflow-Param. (2) Neuer Notebook-Abschnitt «Temporal drift check» nach RQ3: rentml.splits.temporal_split auf
observed_at, nur ungesehene Objekte, eingefrorenes lgbm_te/GPBoost/Bundle anwenden; MAE vs. Test-MAE, Intervall-
Coverage mit Clopper-Pearson gesamt und je Mondrian-Zelle → docs/results/temporal_drift.md, rq3_drift.md, MLflow-
Run interval_drift; Modellkarte-Limitation «no temporal drift test yet» automatisch ersetzen. (3)
assert_no_group_leakage zwischen Freeze-Set und Train/Calib/Test. Drift-Set entscheidet nichts (nur Bericht).
```

---

### Block C – Modelle und Evaluation (W4–W9)

#### C1 · Baselines und RQ2-Härtung `[L] Fable 5.1 · /effort xhigh · Plan-Mode`
Schliesst: #50, #58, #78, #79, #80, #97. Voraussetzung: A7 (Budget pre-registriert). Nur CV/Kalibrierung, keine neuen Test-Reads vor dem konfirmatorischen Lauf.

```text
Härte Baseline und RQ2 gemäss Proposal §1/§2 und dem getaggten Plan, ohne das Testset erneut zu lesen (CV und
Kalibrierset genügen; Test nur im konfirmatorischen Lauf C4). Lies Notebook-Abschnitte 12–15, src/rentml/{baseline,
models,spatial,_gpboost_cov,autocorrelation,evaluation}.py und docs/results/rq2_*.md. (1) DSPRO1-Benchmark: das im
Proposal genannte GradientBoosting ALL+geo (models/baseline, rentml.baseline) auf test_df[baseline.applicable]
scoren (nur im C4-Lauf) und das GB-Rezept als «new baseline» auf den DSPRO2-Daten neu trainieren (eigene Zeile in
rq2_models/ablation/Modellkarte); Notebook-Text korrigieren (kNN-LightGBM ist nicht der DSPRO1-Champion). (2)
GPBoost-Tuning gemäss Plan: rentml.spatial_tuning mit Optuna über Baumparameter bei fixierten Kovarianzparametern
(Stage 1 je äusserem Fold oder offengelegt), Early Stopping, num_threads = 1 für den finalen Fit; Sensitivität
«GPBoost + Koordinaten in den Bäumen». gpboost-1.7.4-Abbruchbedingungen in spatial.py beachten. (3) Moran's I:
Tabelle (I, erwartet, z, p) nach docs/results/rq2_moran.md, MLflow-Metriken, paired Differenztest I(lgbm) − I(gpb)
via Cluster-Bootstrap mit festen Gewichten (Approximation benennen). (4) Spatial-CV mit paired_bootstrap_mae und
Buckets → rq2_spatial_cv.md. (5) Lernkurve mit getunten Parametern, je Sprachregion/Kanton («in which regions data
is missing»), Zahl 30'000 → 20'000 in Notebook und DATENBESCHAFFUNG.md. (6) Seed-/Split-Sensitivität (5 Seeds,
nur CV-MAE-Differenzen). (7) Train/Eval-Gap als Kennzahl je Modell (Proposal begründet DSPRO1 damit), Actual-vs-
Predicted-Plots und Prediction-Files je Modell/Stufe als MLflow-Artefakte (Proposal §5). (8) docs/results/
disclosures.md generiert: CHF/m²-Filter nutzt Ziel vor dem Split, Optuna auf denselben Folds, Kovarianz-Reuse,
Anzahl Test-Reads, Winner's Curse der RQ2-Regel, Buckets aus Trainingszählung, dirty-commit-Provenienz des Piloten.
Alle neuen Funktionen in rentml mit Tests; Notebook nur Orchestrierung. Fast-Modus-Lauf der betroffenen
Abschnitte zum Nachweis.
```

#### C2 · RQ3-Härtung und Bundle-Vertrag `[M] Opus 5.5 · /effort xhigh`
Schliesst: #81–#84 (bestätigen), Teil #95.

```text
Härte P3 (Proposal §1 «Per-apartment prediction intervals») ohne Test-Reads. Lies src/rentml/{quantile,
_quantile_grid,conformal,_interval_metrics,rentcheck,estimate}.py, Notebook-Abschnitte 18–19 und 23,
docs/results/rq3_*.md. (1) coverage_ci (Clopper-Pearson, covered-Vektor statt Rundung) und reliability_by_unit
aus den Notebook-Zellen nach rentml.conformal/_interval_metrics mit Tests (k=0, k=n, groups=None); Notebook ruft
sie auf, rq3_coverage_cells.md und rq3_reliability_map.md bleiben diff-identisch. (2) Quantilmodelle: Baumzahl-Cap
1000 ist für q50/q75/q90/q95 bindend → Cap/Lernrate anpassen, Pinball auf innerem Split vergleichen, Changelog.
(3) GPBoost-gewinnt-Kontingenz: Fallback-Zweig explizit (metadata['rq3_method_of_record'] ≠ served), Test. (4)
Punktschätzung vs. Quantil-Median (44 % der Testpunkte ausserhalb 40.–60. Perzentil): UI-Regel umsetzen (q50
anzeigen oder Fussnote), metadata['estimate_consistency']. (5) RentModelBundle: schema_version, provenance
(Bibliotheksversionen, git_commit/dirty, created_at, data_source, split_sha, eval_plan_version), JSON-Sidecar ohne
Unpickle lesbar, Prüfung beim Laden; bestehendes Bundle im fast-Modus neu erzeugen. (6)
tests/test_bundle_integration.py (skipif ohne models/rent_bundle_v1.joblib): Spaltenvertrag estimate.features ↔
bundle.feature_cols, lo < q25 ≤ q50 ≤ q75 < hi, Verdict-Konsistenz, Mondrian-Zelle gesetzt. (7) Reliability-Tabelle
als Parquet nach data/interim exportieren (für den Kartenlayer E3).
```

#### C3 · Notebook→Modul-Refactor, Regressionstest, Determinismus `[L] Fable 5.1 · /effort xhigh · Plan-Mode · ultracode`
Schliesst: Teil #96, #101. Voraussetzung: A4 (Marker), C2.

```text
ultracode. Ziel: Notebooks enthalten nur Orchestrierung und Narrativ; jede wiederverwendbare Logik liegt getestet
in rentml, und ein Regressionstest schützt beide Notebooks. Parse beide Notebooks (JSON) und liste zuerst alle
def/class in Codezellen (ca. 20 im ML-Notebook) mit Ziel-Modul. Dann: rentml.ablation (StageRunner für
fit/predict/CV je Stufe, ersetzt stage_*-Closures; explizite Parameter-Objekte statt Notebook-Globals),
rentml.reporting (Model-Card-Renderer, RQ-Entscheidungstabelle, Disclosures), conformal.candidate_intervals
(interval_methods), baseline.dspro1_knn_recipe, qoli.threshold_shares/normalize_with_quiet/value_score_by_peers,
plotting.style_map. Regel: docs/results/*.md nach einem fast-Lauf diff-identisch (bis auf Zeitstempel/Session).
tests/test_notebooks.py (Marker slow, network für QoLI): nbclient führt Kopien unter tmp_path mit
DSPRO2_PROJECT_ROOT-Umlenkung in ProjectPaths aus, prüft zentrale Ergebnisdateien; Laufzeit dokumentieren.
Determinismus: DSPRO2_DETERMINISTIC=1 (LightGBM deterministic/force_row_wise, GPBoost num_threads=1 im fast-Modus),
Golden-Metrics-Test (zwei fast-Läufe identisch). Interim-Caches mit Sidecar-JSON (sha256, seed, fractions,
rentml-Version, data_source) und fail-loud statt stillem Überschreiben des eingefrorenen Splits
(rentml.splits.load_or_create_frozen_split). Konsolenskripte [project.scripts] (rentlens, rentml-mapbundle,
rentml-doctor mit Versionen/Pfaden/Caches/Env-Flags ohne Secrets), Pins (gpboost>=1.7.4,<1.8; optuna<6; mlflow<4;
shap<1; dash<5), DSPRO2_CLAUDE_MODEL env statt Hardcode, stale «dspro2/»-Docstrings fixen, ~20 fehlende Docstrings.
Arbeite in kleinen Commits (Vorschläge), Regressionstest vor dem Umbau grün.
```

#### C4 · Konfirmatorischer Volllauf auf dem eingefrorenen DB-Export `[L] Opus 5.5 · /effort high`
Schliesst: #95, #80, #83 (bestätigt), #94 falls Text bereit. Voraussetzung: Tag, Freeze (B7), C1–C3, idealerweise D5. Laufzeit Stunden – im Hintergrund.

```text
Konfirmatorischer Lauf gemäss docs/EVALUATION_PLAN.md (Tag eval-plan-v1). Vorbedingungen prüfen und abbrechen,
falls verletzt: git_dirty == False (alles committet), `git describe --tags` liefert eval-plan-v1, data_fingerprint
des Freeze-Parquets bekannt, FREEZE_DATE gesetzt, HAS_TEXT-Status klar. Dann: DSPRO2_DATA_SOURCE=db
DSPRO2_RUN_MODE=full `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/dspro2_ml_pipeline.ipynb`
im Hintergrund mit Log; danach QoLI-Notebook im db-Modus (DSPRO2_QOLI_REFRESH=0, OSM-Cache mit Datum ins
Register); `uv run rentml-mapbundle --source db`. Neuer Split split_v1_db.parquet (fail-loud, nicht überschreiben).
Ergebnis: docs/results/*, docs/fig/*, models/rent_bundle_v1.joblib (Registry-Version + Alias), Modellkarte mit
data source db, sauberem Commit und Plan-Version. Vergleiche Pilot vs. konfirmatorisch in einer Tabelle (RQ2-
Sieger, MAE, Coverage, H3a/H3b, RQ4), prüfe Testset ≥ 2'000 oder dokumentiere, it×B-Zellen, Monotonie-Checks
PASS, Bundle reproduziert RQ3-Intervalle (same_as_rq3). Fasse die RQ-Entscheide aus Abschnitt 25 wörtlich zusammen.
Nichts committen – ich reviewe zuerst.
```

---

### Block D – Text und RQ1 (W4–W11)

#### D1 · Anonymisierer härten, Sprach-ID, Übersetzung `[L] Opus 5.5 · /effort xhigh`
Schliesst: #47. Voraussetzung: A5 (echte Texte), Entscheid Übersetzungsanbieter + Key.

```text
Textvorbereitung aus Proposal §2 auf echten DB-Texten (data/interim/listings_db.parquet, anonymisiert). (1)
Anonymisierer (src/rentml/_anonymize.py): price_leak_rate < 1 %; 50 geflaggte Texte manuell kategorisieren, neue
Muster (Währungsschreibweisen, Telefonformate, Namen ohne Anrede) als parametrisierte Tests; Laufzeit-Test muss
halten; Changelog. (2) Sprach-ID: Proposal nennt fastText lid.176. Prüfe mit uv, ob ein Python-3.12-Wheel
(fasttext-wheel) installierbar ist; sonst lingua-language-detector als dokumentierte Alternative. Neue Funktion
detect_languages mit Konfidenz, Stoppwort-Vote als Fallback für sehr kurze Texte, Modelldatei nach data/external/
(gecacht), Kreuztabelle text_lang × lang_region im Notebook, Genauigkeit auf 100 geprüften Texten → docs/results/
language_id.md. (3) src/rentml/translate.py: Provider-Interface (DeepL, Google, LLM via Claude-Batch als Option;
Fake-Client für Tests), nur description_anon wird gesendet, Platzhalter müssen die Übersetzung unverändert
überstehen (Test), Cache keyed sha256(text|provider|version) in Neon translation_cache + Parquet-Spiegel, Spalte
description_de (DE unverändert), Original + text_lang bleiben Feature/Slice; Kosten schätzen (FR/IT-Anteil × Länge)
und loggen; translation_version als MLflow-Param; Anbieter-Datenverarbeitungsbedingungen als Zeile in
rentml.data_sources (Datasheet). Wiederlauf ohne API-Aufrufe (Cache-Hit 100 %). 30 Übersetzungen Stichprobe für mich.
```

#### D2 · Gold-Set: ~300 handgelabelte Inserate `[M] Opus 5.5 · /effort high` + ~10–15 h Mensch
Schliesst: #65. Voraussetzung: D1 (Sprache), Entscheid Original oder Übersetzung labeln.

```text
Bereite das Gold-Set aus Proposal §1/§2 vor («about 300 hand-labelled listings … precision/recall per field,
stratified by original language»). Lies src/rentml/extraction.py (ATTRIBUTE_SCHEMA, 17 Felder, evaluate_extraction)
und _extraction_rules.py. (1) docs/LABELLING_GUIDE.md: Definition jedes Feldes, Grenzfälle, null vs. false,
Evidenzregel, Mietregime-Kategorien, Beispiele je Sprache. (2) src/rentml/labelling.py: stratifiziertes Sampling
100/Sprache aus dem DB-Export, Oversampling seltener Felder (Minergie, Haustiere, Regime, befristet) über
keyword_flags, Ausschluss späterer Seed-Texte, Template-CSV mit Spalten gemäss ATTRIBUTE_FIELDS, Validierung gegen
_FIELD_SPECS, Cohen's Kappa je Feld für ein Doppel-Label-Subset (60 Texte). (3) Leichtes Label-Tool: Dash-Seite
/label (lokal, nicht deployt) oder Notebook-Widget, zeigt description_anon, Formular gemäss Schema, schreibt
data/interim/extraction_gold.csv (DVC, nie Git). (4) docs/results/labelling_agreement.md aus Kappa. Zeitplan: Elias
und Timo je 150 Texte, W5–W7. Tests ohne Netz.
```

#### D3 · LLM-Seed-Set (~2'000) per Message Batches `[M] Opus 5.5 · /effort xhigh` – in der Session zuerst `/claude-api` laden
Schliesst: #64, #66. Voraussetzung: `ANTHROPIC_API_KEY`, Budgetfreigabe, D1.

```text
Lade zuerst die Skill claude-api. Setze das LLM-Seed-Set aus Proposal §1 um («seed set of about 2,000 listings
labelled by an LLM in batch mode … every field must cite its text evidence, results cached … the LLM is only used to
bootstrap labels»). Lies src/rentml/extraction_llm.py (ClaudeExtractor, ExtractionCache, batch_requests/submit_batch/
collect_batch, _NO_FORCED_TOOL_MODELS) und tests/test_extraction_llm.py. (1) Modell: Default auf claude-opus-5-5
(kein forced tool_choice auf 5.5 – der Code behandelt das; strict tool + Prompt-Anweisung), Batches halbieren die
Kosten; Kostenschätzung vorab (count_tokens auf 20 Texten × 2'000). (2) Prompt-Entwicklung auf 50 Texten (live,
DSPRO2_RUN_LLM=1) mit Vergleich gegen Regeln, PROMPT_VERSION hochzählen. (3) scripts/label_seed_set.py: Sampling
2'000 stratifiziert nach Sprache, Oversampling seltener Felder, Gold-Texte ausgeschlossen; submit → neue Funktion
wait_for_batch (Polling bis ended, Backoff) → collect; Cache data/cache/extraction.jsonl + Neon extraction_cache;
Fehlerliste; Erfolgsquote ≥ 98 %. (4) Evaluation gegen Gold (sobald D2 fertig): Precision/Recall/F1 je Feld und
Sprache → docs/results/extraction_eval_llm.md, MLflow-Params llm_model/prompt_version/cache_hash. (5) Agreement-
Check Regeln vs. LLM auf 100 Texten/Sprache. Datasheet-Zeile für Anthropic-Datenverarbeitungsbedingungen.
```

#### D4 · Selbst trainiertes Multi-Label-Attributmodell `[XL] Fable 5.1 · /effort max · Plan-Mode · ultracode · GPU`
Schliesst: #69, Vorarbeit #90. Voraussetzung: D2, D3, CUDA in WSL2 (`uv sync --extra dl`).

```text
ultracode. Baue das Kernstück des Text-Workstreams aus Proposal §1 (Text as a feature, Punkt 2): «a multi-label
attribute model that we train ourselves: a fine-tuned small German encoder with classification heads for binary and
categorical attributes and regex post-processing for numeric fields … the LLM is only used to bootstrap labels,
never at run time». Lies src/rentml/{extraction,_extraction_rules,extraction_llm,fusion,text}.py und die
Ergebnisse D2/D3. Entwürfe in Plan-Mode: Encoder-Wahl (destillierter deutscher BERT auf description_de vs.
multilingualer MiniLM auf Originaltext – begründen anhand D1-Entscheid), Kopf je Feld (binär 3-klassig
true/false/nicht erwähnt; view/parking/rent_regime kategorial), Verlustgewichtung bei Klassenungleichgewicht,
Regex-Postprocessing für floor/renovation_year/area/rooms, Training auf Seed-Labels, Gold nur Validierung/Test,
Early Stopping, Seeds. Implementierung src/rentml/attribute_model.py (+ Trainingsskript scripts/train_attribute_
model.py mit MLflow), Inferenz-API kompatibel zu rule_based_extract (ExtractedAttributes mit Evidenz-Span aus
Attention oder Regex), CPU-Export (ONNX/int8) für die App unter 0.5 GiB RAM, Fallback auf Regeln. Tests ohne
Download (Tiny-Fake-Encoder). Akzeptanz: Makro-F1 über Felder auf Gold schlägt die Regeln (je Sprache berichtet)
oder Abweichung begründet; Inferenz über alle Texte < 10 min GPU / < 1 h CPU; docs/results/attribute_model_eval.md;
attribute_model_version als MLflow-Param; Datasheet/Modellkarte-Abschnitt.
```

#### D5 · Attribute anwenden und RQ1-Ablation `[L] Fable 5.1 · /effort xhigh`
Schliesst: #62, #63, #67, #94. Voraussetzung: D1, D4 (oder Zwischenlösung LLM-Labels auf allen Texten), Plan-Reihenfolge aus A7.

```text
RQ1 gemäss Plan (Reihenfolge aus eval-plan-v1 beachten: Textablation auf dem RQ2-Gewinner). Lies Notebook-
Abschnitte 7–9 und 17, src/rentml/{text,extraction,cleaning,features}.py, docs/results/mde_table.md. (1)
ATTR_SOURCE um 'model' erweitern (Priorität Modell > vollständiger LLM-Cache > Regeln), attr_frame auf allen
Texten, rent_regime-Filter mit separater Regime-Tabelle (market vs. cooperative/cost-based/shared mit Objektzahl
und Median-CHF/m²), recover_rooms_from_text aktiv, Flächenrekonstruktion aus Text vor dem Domänenfilter prüfen oder
Abweichung im Changelog bestätigen; F_ATTR in feature_table.md. (2) Textmerkmale: keyword_flags + Prävalenz/
Prämienstatistik (docs/fig/keyword_premium), TfidfSvd (n_components/min_df per CV wählen), SentenceEmbedder
(Modellwahl nach Übersetzungsentscheid, Cache data/cache/embeddings), PCA auf Train. (3) Abschnitt 17 mit
HAS_TEXT=True: S0 → S1 (Keywords+TF-IDF) → S2 (+Embeddings) → S3 (+Attribute) → S4 optional (Fusionsnetz mit
Textzweig), Grouped-CV-Selektion, gepaarter Cluster-Bootstrap, Holm über S1–S3, Vergleich mit MDE; Ergebnisse
docs/results/rq1_text_ablation.md/.tex + Figur, ablation.md-Zeilen, Modellkarte, Summary-Text. Testset-Read nur
gemäss Plan (einmal, im konfirmatorischen Lauf C4 – sonst CV-Resultate berichten). (4) Falls eine Stufe signifikant
gewinnt: Trigger für D6 (Refit der Intervalle) dokumentieren.
```

#### D6 · Text in die App: Bundle v2, Refit der Intervalle, What-if Attribute `[L] Opus 5.5 · /effort xhigh`
Schliesst: #89 (What-if), #90. Nur wenn D5 signifikant; sonst Grösse S (Changelog «S0 wird serviert»).

```text
Proposal §1: «if a text stage lowers the MAE significantly, the intervals are refitted on that stage» und «What-if
switches for … renovation, balcony and lift». Lies src/rentml/{rentcheck,estimate,text,extraction}.py und Notebook-
Abschnitte 18–19, 23. (1) RentModelBundle v2: metadata['text_transformers'] (TfidfSvd, PCA + Embedder-Spezifikation,
Attributmodell/Extractor), feature_transform-Hook; estimate.features baut Keyword-/SVD-/Attributspalten aus
Apartment.description (fehlender/kurzer Text wie im Training behandeln). (2) RQ3-Zellen auf RQ1_SELECTED-Features
parametrisieren, Quantil- und Mondrian-CQR neu fitten, same_as_rq3 True. (3) /vermieter: Toggles Renovation,
Balkon/Terrasse, Lift → what_if mit binären Änderungen nur wenn in feature_cols; Vorzeichen-Plausibilität im
Notebook loggen; Hinweis «geschätzter Marktaufschlag, keine Monotoniegarantie». (4) Speicher/Latenz unter
0.5 GiB messen (Sentence-Transformer evtl. ONNX/quantisiert oder S2 in der App weglassen – begründen). Tests:
Bundle ohne Netz ladbar, mit/ohne Beschreibung unterschiedliche Schätzung, Coverage auf Kalibrierset ≥ 80 %.
```

#### D7 · Optionale Erweiterungen entscheiden `[S] Opus 5.5 · /effort medium`
Schliesst: #68, #70 (als could/wont). Bis 29.11. (Woche 11).

```text
Proposal-Timebox: fused Transformer, SWISSIMAGE-Embeddings und Ollama nur, falls Text-Schritte (1)–(2) bis Woche 11
fertig sind. Prüfe den Stand von D1–D5, formuliere die Entscheidung (umsetzen / verschieben / verzichten) als
Changelog-Einträge in docs/EVALUATION_PLAN.md und CHANGELOG.md, Modellkarte-Limitations und README «Noch offen»;
dokumentiere, dass rentml.fusion (MLP mit Entity-Embeddings + Zweig auf vorberechneten Embeddings) die zeitlich
begrenzte Variante ist. Bei «umsetzen»: Scope-Dokument mit GPU-Bedarf und Akzeptanzkriterien als neues Issue.
```

---

### Block E – QoLI in Produkt und Report (W6–W9)

#### E1 · QoLI-Konsolidierung `[M] Opus 5.5 · /effort xhigh`
Schliesst: #71–#75 bestätigen, #85 (Report-Teil), #53 (OSM-Register).

```text
Konsolidiere P2 (Proposal §1 QoLI, RQ4). Lies notebooks/dspro2_qoli.ipynb (alle Abschnitte), src/rentml/qoli*.py,
amenities.py, Notebook dspro2_ml_pipeline Abschnitt 22 und docs/results/qoli_*.md. (1) OSM als SourceSpec (Overpass-
URL, ODbL 1.0 mit Attribution/Share-alike, Download-Datum aus Cache-mtime, 8 Kategorien) ins Register; Test «jede
Quelle hat Lizenz». (2) Der 3-Dimensionen-Prototyp im ML-Notebook Abschnitt 22 überschreibt qoli_sensitivity.md und
erzeugt «3 of 6 dimensions» in der Modellkarte: ersetze ihn durch das Laden von data/interim/qoli_objects.parquet
und der QoLI-Ergebnistabellen; nur eine Sensitivitätstabelle in docs/results; beide Notebooks berichten dieselben
RQ4-Zahlen. (3) Notebook-Logik nach rentml.qoli/qoli_robustness: threshold_shares, normalize_with_quiet,
value_score_by_peers, RQ4-Zusammenfassung und City-Statistics-Tabelle als gerenderte docs/results-Dateien; Zahlen
exakt gleich (Diff). (4) Regionale OSM-Vollständigkeit (ρ Erreichbarkeit vs. BFS-Distanzen je Sprachregion/Kanton/
Dichte-Tertil) → docs/results/osm_completeness.md. (5) BFS STATENT (Proposal-Quelle, ungenutzt): Indikator
«Arbeitsplatz-Erreichbarkeit» oder dokumentierter Verzicht im Register; ESTV 2018 – neuere Gemeindedatei prüfen.
(6) Bevölkerungsgewichtete QoLI-Aggregation auf Bezirk/Kanton (weighted_unit_scores) als Export für die Karte.
Tests für alles Neue; QoLI-Notebook im fast/cached-Modus nachweisen.
```

#### E2 · QoLI-Service, Vermieter-Lageerklärung, Value Score, Gewichte `[L] Fable 5.1 · /effort xhigh · Plan-Mode`
Schliesst: #74 (Gewichte), #85 (App), Teil #89.

```text
Proposal §1: Vermieter-Ansicht zeigt «the QoLI dimensions as location explanation», QoLI hat «user-defined weights
across dimensions», Value Score «quality of life relative to price». Entwirf zuerst (Plan-Mode): Live-Raster-
Sampling (1.8 GB) ist in der App nicht tragbar → Hektar-Lookup aus einem Export data/app/qoli_hectares.parquet
(reli, east, north, municipality_id, normierte Dimensionen; ~30 MB) + models/qoli_spec.json. Dann: (1)
src/rentml/qoli_service.py: QoliSpec.load, nächste Hektare zu einer LV95-Koordinate (KD-Tree), Dimensionen 0–100,
Index mit Nutzergewichten, Gemeinde-Referenz aus qoli_municipalities.parquet; < 50 ms; Werte identisch zum
Notebook (Toleranz 0.1). (2) rentml.qoli.value_score_single gegen exportierte Peer-Referenzquantile
(Sprachregion × Grössenband, Mindest-n mit Fallback). (3) Dash /vermieter: Block mit vier Kern-Dimensionsbalken,
Index, Gemeindemarker, erweiterte Dimensionen markiert, Attribution (OSM ODbL, BAFU/ARE/BFS/MeteoSchweiz),
Hinweis «Hektarmittel, nicht Fassade»; /mietcheck optional Value Score mit Peer-Gruppe und n. (4) Gewichts-Slider
(0–3) + Profile gleich/Familie/Pendler (aus Notebook-Profilen in config/Spec), Summe-0 abgefangen, Default
reproduziert qoli-Spalte exakt. (5) Graceful Degradation ohne QoLI-Dateien; Services um qoli ergänzen; Tests mit
synthetischer Spec/Hektartabelle; README-Abschnitt. Image-Grösse +~35 MB einplanen (RENTML_QOLI_DIR).
```

#### E3 · Kartenlayer QoLI, Value Score, Reliability `[M] Opus 5.5 · /effort xhigh`
Schliesst: #92, #84 (Karten-Teil). Voraussetzung: E1 (Aggregation), C2 (Reliability-Parquet).

```text
Proposal §1 Karte: «drill-down … with median CHF/m² (cells with fewer than 20 listings are hidden), QoLI, value
score and, if built, the reliability layer». Lies src/rentml/{mapdata,mapbundle}.py, webapp/{home,figures,panel}.py
(extra_metrics ist vorbereitet) und tests/test_mapbundle.py. (1) build_map_bundle joint: QoLI je Gemeinde (alle
2'110, ohne Mindestanzahl – keine Inseratedaten), Bezirk/Kanton bevölkerungsgewichtet; Value Score Gemeinde-Median
ab 20 Objekten aus data/interim/qoli_objects.parquet; Reliability (coverage_80, CP-CI, n) aus dem Parquet von C2
(Kanton-Key ist Abkürzung → CANTON_ABBR-Mapping), nur Zellen ≥ MIN_MAP_CELL; meta.json extra_metrics mit Quellen
und Erklärtexten. (2) Dropdown «Kennzahl» mit QoLI, Value Score, Zuverlässigkeit 80 %-Intervall (Farbskala um 80 %
zentriert, Panel zeigt «x % (CI a–b, n)»), Dimensionsprofil im Panel bei QoLI. (3) Footer-Attribution auf allen
Kartenseiten. (4) Tests: fehlende Datei → Metrik entfällt; Join-Korrektheit; GeoJSON-Endpoint weiterhin gecacht.
Bundle neu bauen (`uv run rentml-mapbundle --source csv`, später db).
```

#### E4 · (optional) QoLI-Rohindikatoren als Ablationsstufe `[L] Fable 5.1 · /effort xhigh` – nur wenn in A7 pre-registriert

```text
Proposal §2 zählt Lärmpegel, Schadstoffkonzentrationen, Erreichbarkeit und Distanzen zu den Feature-Typen des
Preismodells. Falls im getaggten Plan als Stufe vorgesehen: Rohindikatoren aus data/interim/qoli_objects.parquet
(oev_class, oev_stop_m, noise_day/night_db, green_share, acc_total_all, no2, pm25, sunshine – NICHT value_score/
chf_per_m2) als Feature-Set F_QOLI, Leakage-Check (keine Spalte aus Mieten), Stufe in rentml.ablation mit CI und
p-Wert, Live-Features in estimate.py via qoli_service, Changelog. Sonst: Verzicht dokumentieren.
```

---

### Block F – App und Deployment (W7–W12)

#### F1 · Produktionsbetrieb: WSGI, /health, Lazy-Imports, Smoke, Dockerfile, Benchmark `[M] Opus 5.5 · /effort xhigh`
Schliesst: #87 (Infra-Teil), Vorarbeit #91. Docker Desktop muss laufen.

```text
Mache die Dash-App containerfähig für Azure Container Apps (Proposal §3: 0.25 vCPU, 0.5 GiB, 0–1 Replikate, Health-
Check). Vorlagen der Streamlit-Zeit: `git show 2d249cb^:Dockerfile`, `:.dockerignore`, `:compose.yaml`,
`:scripts/smoke_app.py` (für Dash portieren). (1) src/rentml/webapp/wsgi.py (create_app aus RENTML_MAP_BUNDLE/
RENTML_MODEL_BUNDLE/RENTML_QOLI_DIR, `server = app.server`), gunicorn im Extra app, Route /health (200, Bundle-/
Modellversion, Kartenstand, keine Eingaben), geopandas/shap/matplotlib nur lazy auf dem Serving-Pfad
(`python -c "import rentml.webapp.app"` ohne geopandas). (2) scripts/smoke_app.py: Flask test_client gegen /,
/mietcheck, /vermieter, /health, GeoJSON je Ebene, eine Schätzung mit Stub-LocationData; druckt PASS + Peak-RSS. (3)
Dockerfile multi-stage mit uv (`uv export --frozen --no-dev --extra app --no-emit-project`), python:3.12-slim mit
Digest, non-root, HEALTHCHECK /health, PORT-Env (HF-Fallback), schlankes Serving-Set (kein torch/gpboost/rasterio
im Image – prüfen, was estimate/rentcheck wirklich importieren; evtl. Extra «serve»). (4) Benchmark `docker run
--memory 512m --cpus 0.25`: Startzeit bis /health, RSS idle, RSS nach 20 Schätzungen → docs/DEPLOYMENT.md; Ziel
Peak-RSS < 400 MB, /health < 30 s; bei Überschreitung Massnahmen (GeoJSON vorgezippt via send_file, SHAP-Cache).
Tests für /health und wsgi.
```

#### F2 · Azure-Deploy-Workflow, GHCR, HF-Fallback, Bundles, Streamlit-Stilllegung `[L] Opus 5.5 · /effort xhigh`
Schliesst: #91. Voraussetzung: F1, B5 (oder Release-Assets), Azure-OIDC/GHCR-Rechte.

```text
Stelle den Deploy-Pfad aus Proposal §3 wieder her: «Docker image published to the GitHub Container Registry,
deployed by a GitHub Actions workflow (azure-prototype.yml, with deploy/start/stop/status/delete actions and a post-
deploy health check) to a container app scaling between 0 and 1 instance with 0.25 vCPU and 0.5 GiB RAM». Quellen:
`git show 2d249cb^:.github/workflows/azure-prototype.yml`, `:scripts/azure.sh`, `:scripts/az.sh`,
`:scripts/setup_github.sh`, `:scripts/check_deployment.py`, `:tests/test_azure_lifecycle.py`, docs/archive/dspro1/
AZURE.md. Portiere auf Dash (Port 8050, /health, Ressourcengruppe rg-dspro2-rentlens, switzerlandnorth), OIDC
Federated Credential neu, Repo-Variablen AZURE_* ohne Hardcodes, GHCR-Paket privat. Bundles für den Build: Variante
A GitHub-Release «bundles-v1» mit sha256 (Karte enthält nur Aggregate ≥ 20; Modell-Pickle ok?) oder Variante B
`dvc pull` aus R2 (B5) – Entscheidungsvorlage, dann umsetzen. HF Spaces Fallback: spaces/README.md (sdk: docker,
app_port 8050) + optionaler Sync-Job. Streamlit-Prototyp aus DSPRO1 (rg-dspro1-prototype): Stilllegung vorschlagen
(status → stop → delete nur nach meiner Freigabe) damit keine zwei öffentlichen Apps laufen. Lifecycle-Test
portieren. docs/DEPLOYMENT.md: Runbook, Kosten, Kaltstart, Secrets. Nichts deployen ohne mein OK.
```

#### F3 · App-Funktionen: Formularpflicht-Links, GWR-Wohnungsauswahl, Footer/Datenschutz, Attribution `[M] Opus 5.5 · /effort xhigh`
Schliesst: #88 (Rest), #89 (Rest). Rechtsliste verifizierst du.

```text
Proposal §1 Tenants/Landlords. Lies src/rentml/webapp/{tenant,landlord,components,app}.py, estimate.py, geoadmin.py.
(1) Rechtsblock: Formularpflicht-Kantone mit Link zum amtlichen Formular. Recherchiere live (WebFetch) die BWO-Seite
zur Formularpflicht/Anfechtung des Anfangsmietzinses (Art. 270 OR) und die Mieterverbands-Seite, erstelle
FORMULARPFLICHT: dict[canton_abbr, (Formularname, URL, Quelle, Stand)] NUR mit Kantonen und URLs, die du belegen
kannst; markiere Unsicherheiten – ich verifiziere die Liste vor dem Merge. Kantonsabhängige Anzeige, spezifische
BWO-/MV-URLs statt Root, Tests. (2) /vermieter «Adresse → GWR-Wohnungsauswahl»: geoadmin.dwellings_for_egid über
api3 (Layer ch.bfs.gebaeude_wohnungs_register; prüfe, welche Wohnungsfelder – warea, wazim, wstwk, ewid – die API
liefert; sonst MADD mit 20 req/min), Dropdown zur Vorbelegung von Fläche/Zimmer/Stockwerk, keine Mieten pro
Wohnung ableitbar; Tests mit Fake-Session. (3) Footer auf allen Seiten: Modellversion/Run-ID aus Bundle-Metadaten,
Datenstand, Aggregat-Schwelle 20, «Eingaben werden nicht gespeichert», Link Modellkarte; Seite /methodik (kurz,
Deutsch). (4) Attribution «© OpenStreetMap contributors (ODbL)», «© swisstopo», «BFS/BAFU/ARE/MeteoSchweiz OGD».
```

---

### Block G – Dokumentation und Abgaben

#### G1 · Mid-term-Präsentation (22.10.) `[L] Fable 5.1 · /effort xhigh`
Schliesst: #60. Entscheid Tooling (Marp/Quarto → PDF oder LaTeX Beamer; HSLU-Vorlage ist .odp).

```text
Erstelle die Mid-term-Präsentation für den 22.10.2026 (10–15 Folien, Deutsch oder Englisch – frag mich) als
Markdown-Deck (Marp oder Quarto reveal/PDF; Vorschlag machen, HSLU-Farben aus docs/archive/dspro1/presentation-
mid-term/HSLU_Vorlage.odp übernehmen) in docs/presentation-midterm/. Storyline aus Proposal und Repo: Problem und
DSPRO1-Benchmark; RentLens-Ziele und Prioritäten P1 GPBoost / P2 QoLI / P3 Intervalle; Datenaudit (9'994 →
9'405 → 8'298 → 8'125, Text-Abdeckung aus db_audit.md falls vorhanden); pre-registrierter Evaluationsplan (Tag);
RQ2-Pilot ehrlich: LightGBM-TE schlägt GPBoost (rq2_mae_by_bucket.png, gpboost_shrinkage.png), Interpretation und
offene Tuning-Frage; P3: Mondrian-CQR Coverage/Breite (quantile_reliability.png, rq3_reliability.png); P2: QoLI-
Karten (qoli_map.png, qoli_dimension_maps.png), RQ4-Stabilität; App-Screenshots (Karte, Fair-Rent-Check,
Vermieter); Datenstrategie (Pilot, Freeze 06.–08.11., 20'000-Ziel, Lernkurve); Plan bis Woche 14 mit Risiken
(Daten-Grauzone, GPU, Zeit). Jede Zahl aus docs/results/*.md, alles mit «Pilotlauf auf DSPRO1-Snapshot (CSV)»
beschriftet. Sprechernotizen je Folie, Folienaufteilung Elias/Timo vorschlagen, PDF-Export-Befehl.
```

#### G2 · Datasheet v2 (06.11.) `[L] Fable 5.1 · /effort xhigh · Plan-Mode`
Schliesst: #76, Teil #98. Voraussetzung: A2 (Lizenzentscheid), A5, A6; Zahlen nach Freeze nachtragen.

```text
Schreibe das DSPRO2-Datasheet nach HSLU-Vorlage (docs/__templates/DSPRO___Data_Sheet_Template.zip entpacken nach
docs/datasheet/, DSPRO1-Fassung docs/archive/dspro1/data-sheet/datasheet.tex als Stilreferenz). Proposal §2
verlangt: jede Quelle mit Terms/Lizenz/Bezugsjahr, Zahlen vor/nach Dedup, Anonymisierung, Übersetzungs-/LLM-
Anbieterbedingungen, Limitationen (Angebotsmieten, modellierte Lärm-/Luftdaten, OSM-Vollständigkeit), rentumo ohne
Lizenz → nur Training. Regel «keine Zahl von Hand»: erweitere die Notebooks/rentml.reporting so, dass .tex-
Fragmente (rentml.plotting.to_latex) für data_sources, qoli_sources, acquisition_sources, cleaning_log, db_audit,
dedup_report, text_coverage, language_id entstehen und per \input eingebunden werden; Volumenkette 9'994 / 9'405 /
8'298 / 8'125 / 4'536 widerspruchsfrei erklären; Abschnitt Legal/Governance mit revDSG-Handling (Anonymisierung bei
Ingestion, Prozessoren Anthropic/Übersetzer, keine Speicherung von Eingaben) und ODbL-Share-alike-Hinweis.
`latexmk -pdf` baut; Makefile-Target docs. Platzhalter-Marker für Zahlen, die erst nach dem Freeze kommen.
```

#### G3 · Risiko-Selbstbewertung (EU AI Act, revDSG) `[M] Fable 5.1 · /effort xhigh`
Schliesst: #99.

```text
Schreibe docs/RISK_ASSESSMENT.md (+ .tex-Anhang für den Report), wie im Proposal §1 versprochen («documented risk
self-assessment that uses the EU AI Act risk categories as a voluntary framework together with the revDSG
obligations that apply»). Struktur: Systembeschreibung und Nutzergruppen; Art. 5 (verbotene Praktiken) Check;
Annex III Walk-through (insb. Zugang zu wesentlichen Diensten/Kreditwürdigkeit, Screening) mit Schlussfolgerung und
Begründung; Art. 50 Transparenz (Hinweis «statistisches Modell», Intervall, Treiber); GPAI nicht relevant; revDSG:
Personendaten bei Ingestion (Adressen, Kontaktdaten in Beschreibungen), Anonymisierung (Modul _anonymize, Tests),
Zweckbindung, Speicherdauer, Auftragsbearbeiter (Anthropic Batch, Übersetzungs-API) und Auslandbekanntgabe,
Informationspflichten, keine Speicherung von App-Eingaben, Aggregat-Schwelle 20; Scraper-Ist-Zustand ehrlich
(robots.txt, Rate-Limits, Verzicht auf Proxy/UA-Rotation, AGB-Grauzone als Risiko); Missbrauchsszenarien
(Mieter-Screening, Preisdiskriminierung) und Massnahmen; jede Massnahme mit Verweis auf Modul/Test/Notebook-
Abschnitt. Ton: Selbstbewertung, keine Rechtsberatung. Modellkarte und Canvas Box 7 verlinken darauf.
```

#### G4 · Report-Skeleton und vollständige Modellkarte `[M] Opus 5.5 · /effort xhigh`
Schliesst: #96, #98. Voraussetzung: C3 (rentml.reporting).

```text
(1) docs/report/ aus docs/archive/dspro1/__templates/DSPRO__FinalReport.zip: main.tex, sections/ (Introduction,
Related Work, Data, Methods, Results je RQ, Discussion/Limitations, Conclusion, Appendix Risk/Datasheet-Verweise),
references.bib (Sigrist 2022 GPBoost, Romano et al. CQR, OECD/JRC Handbook, LightGBM, Optuna, SHAP, Vecchia), jede
Tabelle via \input{../results/*.tex} (ablation.tex, rq2_comparison.tex, rq3_intervals.tex + neue rq1/rq4/drift-
Fragmente), Figuren via \includegraphics aus docs/fig (grosse SVG-Karten als PNG), latexmk-Build, Makefile-Target
report. Proposal-Regel: kein Metrikwert von Hand. (2) Modellkarte vervollständigen (rentml.reporting): Ablations-
tabelle eingebettet, RQ1–RQ3-Entscheidungszeilen aus Abschnitt 25, QoLI/Value-Score-Abschnitt aus den QoLI-
Ergebnissen, Modellversion/Registry-Alias/eval-plan-Tag/DVC-Hash, Trainingsdatum, Disclosures-Verweis; git-Ref
muss in der aktuellen Historie auflösbar sein; Rendering als .md und .tex; tests/test_reporting.py mit
synthetischen Inputs. Nachweis per fast-Lauf der Report-Zellen.
```

#### G5 · AI Project Canvas `[M] Opus 5.5 · /effort high`
Schliesst: #86. Zahlen aus Datasheet/Risiko; Rollen mit Timo.

```text
Erstelle docs/ai-canvas/DSPRO2_AICanvas_Team6_RentLens.tex (+PDF) aus docs/__templates/DSPRO___AI_Canvas_Template.zip,
Layout-Tricks aus docs/archive/dspro1/ai-canvas/main.tex. Alle zehn Boxen; die in DSPRO1 auf DSPRO2 verschobenen
Boxen 5 (ML-Pipeline: rentml-Module, Splits, MLflow, Registry), 8 (Infrastruktur: Neon, R2/DVC, GitHub Actions,
GHCR, Azure Container Apps, GPU-Workstation), 9 (Kosten und Zeitplan: Neon-Tier, Azure Consumption, LLM-Batch,
Übersetzung, GPU-Stunden – Zahlen von mir erfragen), 10 (Team und nächste Schritte) vollständig. Box 6 Metriken:
MAE CHF, Coverage/Breite, QoLI-Rangstabilität, Stakeholder-KPI. Box 7 verweist auf docs/RISK_ASSESSMENT.md.
Geplant vs. umgesetzt explizit markieren. 1–2 Seiten, kompiliert.
```

#### G6 · Report-Entwurf (04.12.) `[XL] Fable 5.1 · /effort max · Plan-Mode · ultracode` – mehrere Sessions
Schliesst: #93. Voraussetzung: C4, D5, G2–G4. Eine Session pro Abschnitt, danach eine Review-Session.

```text
Session 1 (Plan): Lies Proposal, EVALUATION_PLAN.md (getaggt), CHANGELOG, alle docs/results/*.md, beide Notebooks
(Narrativ-Zellen), docs/RISK_ASSESSMENT.md, Datasheet, Modellkarte. Erstelle docs/report/OUTLINE.md: Gliederung
gemäss HSLU-Template, je Abschnitt Kernaussagen, zu zitierende Tabellen/Figuren (Pfade) und Quellen, Seitenbudget,
offene Punkte. Reihenfolge der Resultate wie pre-registriert (RQ2 → RQ3 → RQ1, RQ4, Drift), negative Resultate
(GPBoost, H3b) als Befund, nicht als Misserfolg.
Sessions 2–n (ultracode, je ein Abschnitt): Schreibe sections/<abschnitt>.tex auf Englisch, wissenschaftlich knapp,
jede Zahl per \input-Fragment oder \ref auf eine generierte Tabelle (nie getippt), jede Methode mit Proposal-Bezug
und Plan-Abweichung (Changelog) erwähnt; Related Work: hedonische Mietmodelle, räumliche Mixed Effects/GPBoost,
Conformal Prediction/CQR/Mondrian, Composite Indicators (OECD/JRC), Text in Immobilienbewertung – mit echten
Referenzen (WebSearch, DOI prüfen, keine erfundenen Zitate). latexmk nach jedem Abschnitt grün.
Review-Session: Lies den gesamten Entwurf als Dr. Nazarenko (Methodenkritik: Leakage, Test-Reads, Pre-Registration,
Multiplizität, Kausalsprache, Limitationen) und als Zweitgutachter (Lesbarkeit, Konsistenz Zahlen ↔ Tabellen,
Seitenlimit). Liefere eine priorisierte Änderungsliste und setze die unstrittigen Punkte um.
```

#### G7 · README, DEPLOYMENT, CHANGELOG, Meetings aktualisieren `[M] Opus 5.5 · /effort high`
Schliesst: #101. Laufend, spätestens vor jeder Abgabe.

```text
Bringe README.md auf den Stand von Proposal und Repo: Titel/Intro als RentLens; Tabelle «Deliverables» (Proposal-§
↔ Notebook-Abschnitt ↔ docs-Datei: Evaluationsplan + Tag, Datasheet, Modellkarte, Ablation, Report, Canvas, Risiko,
Meeting-Log); Reproduktion (fast/full/db, dvc pull, Secrets via .env, MLflow-Backend); Prozess (Board, PR, CI, DoD);
Deployment (Verweis docs/DEPLOYMENT.md); Lizenzen; konsistente DSPRO1-Baseline-Zahlen (README vs.
models/baseline/README.md vs. Proposal); «Noch offen» aktualisieren. CHANGELOG.md und docs/meetings/ nachführen
(Sprint-Planungen wöchentlich, Supervisor-Termine). Jeden genannten Befehl ausführen oder mit --help verifizieren.
```

#### G8 · Final-Präsentation (17.12.) und Demo-Fallback `[L] Fable 5.1 · /effort xhigh`
Schliesst: #102, Vorbereitung #103.

```text
Erstelle die Abschlusspräsentation in docs/presentation-final/ (gleiches Tooling wie Mid-term): Antworten auf RQ1–RQ4
(RQ5 falls durchgeführt) mit den finalen Tabellen/Figuren aus docs/results und docs/fig, Pre-Registration und
Abweichungen (Changelog) als Methodik-Argument, Live-Demo-Skript für die deployte App (drei vorbereitete Inserate:
unter/im/über Band, verschiedene Regionen) plus Offline-Fallback (Screenshots/kurzes Video-Skript), Limitationen und
Lessons Learned, Ausblick. Sprechernotizen, Zeitplan 20 min + Fragen, Folienaufteilung Elias/Timo.
```

#### G9 · Finale Politur und Repo-Abgabe (31.12.) `[L] Fable 5.1 · /effort max` + `/code-review ultra`
Schliesst: #104, #105.

```text
Abgabe-Review. (1) Report: Lektorat (Konsistenz Zahlen ↔ Fragmente, Abbildungsverweise, Bibliografie, Seitenlimit,
Sprache), finaler Build, PDF-Name gemäss HSLU-Konvention. (2) Repo: README-Deliverables-Links alle gültig; CI
grün; `uv sync --all-extras` + `uv run --extra app pytest` auf frischem Clone (tmp) funktioniert; `dvc pull` liefert
Bundles; keine Secrets (gitleaks), keine Inseratdaten in Git gemäss A2-Entscheid; Tags eval-plan-v1, data-freeze-v1,
release v1.0.0 vorgeschlagen; Modellkarte/Datasheet/Canvas/Risiko aktuell; Board: alle dspro2-Issues Done oder
begründet Won't; offene Punkte in docs/FUTURE_WORK.md. (3) Liefere eine Abgabe-Checkliste mit Status je Punkt.
Danach führe ich /code-review ultra auf dem Release-Branch aus.
```

#### G10 · (optional) RQ5 Usability-Test `[S] Opus 5.5 · /effort medium`
Schliesst: #100. Nur bei Restzeit in W12–W14, nach F2.

```text
Proposal RQ5 (optional): «small usability test (5–8 participants, think-aloud protocol on three prepared listings),
reported qualitatively». Erstelle docs/usability/PROTOCOL.md (Ablauf 20–30 min, Moderator/Beobachter, Aufgaben für
Mieter- und Vermieter-Ansicht inkl. Value Score, Think-aloud-Hinweise, qualitatives Kodierraster), CONSENT.md
(revDSG-konform, keine Personendaten gespeichert), LISTINGS.md (drei anonymisierte Inserate aus dem Testset mit
Modell-Output: unter/im/über Band, verschiedene Regionen) und RESULTS_TEMPLATE.md. Kein Code ausser optionalem
Demo-Button, der die drei Inserate vorbelegt.
```

---

## 6. Abhängigkeiten in Kurzform

```
A1 ─┬─ A5 ── A7(Tag) ── B1 ── B2 ── B3(Pilot 19.10.–01.11.) ── B6
    │                    │                                  └── B7(Freeze) ── C4 ── D5 ── D6 ── F3/E3
    │                    └── B4 ── B5 ─────────────────────────────────────────┘
A2, A3, A4, A6 (parallel, Woche 3–4)
C1, C2, C3 (W4–W8, parallel zu B, nur CV/Kalibrierung)        D1 ── D2 ── D3 ── D4 ── D5
E1 ── E2 ── E3 (W6–W9)                                        F1 ── F2 (W7–W11)
G1 (22.10.)  G2 (06.11.)  G3  G4 ── G5 ── G6 (04.12.) ── G7 ── G8 (17.12.) ── G9 (31.12.)
```
