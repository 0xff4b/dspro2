# Lernnotizen: DSPRO2 Mid-Term «RentLens»

Präsentation: `DSPRO2_PresentationMidTerm_Team6_RentLens.pptx` (26 sichtbare Folien, 3 ausgeblendete Demo-Backups).
Sprache der Folien: Englisch. Deshalb ist der **Sprechtext englisch**, die **Erklärungen deutsch**.
Folienzahlen beziehen sich auf die Nummer unten rechts.

---

## 0. Auf einen Blick

**Kernbotschaft in drei Sätzen** (das muss das Publikum am Ende wissen):

1. *Problem:* Mieten in der Schweiz sind intransparent, und ein Preis allein sagt nicht, ob eine Wohnung ein guter Wohnort ist.
2. *Ansatz:* RentLens schätzt pro Wohnung eine Miete **mit kalibriertem Intervall**, gibt ein Urteil (unter / im / über dem Markt) und einen **Quality-of-Life-Index**, alles in einer Web-App.
3. *Status:* Die ganze Pipeline läuft bereits als Pilot auf den DSPRO1-Daten; die neuen Daten (Comparis via Apify) kommen bis zum Data Freeze am 06.11., dann folgt der konfirmatorische Lauf.

**Zeitplan (Vorschlag für ca. 12–15 Minuten):**

| Teil | Folien | Zeit |
|---|---|---|
| 1 Problem | 1–5 | 2 min |
| 2 Goals & Approach | 6–10 | 3 min |
| 3 Current Status | 11–17 | 3 min |
| 4 First Results | 18–22 | 3 min |
| 5 Demo (live) | 23 | 2 min |
| 6 Next Steps | 24–26 | 1.5 min |

**Rollen:** Die Aufteilung zwischen Elias und Timo ist noch offen. Ein sinnvoller Schnitt wäre Teil 1–3 und Teil 4–6. Wer die Live-Demo macht, sollte auch die Resultate präsentieren.

---

## 1. Checkliste vor dem Vortrag

- [ ] Deck in PowerPoint öffnen: Schrift **Tahoma** aktiv? Folien 24–26 (Demo-Screenshots) sind ausgeblendet.
- [ ] App starten und einmal testen (Adresssuche braucht Internet, geo.admin.ch):
  ```bash
  uv run --extra app python -m rentml.webapp
  ```
  Dann `http://127.0.0.1:8050/mietcheck` im Browser offen lassen.
- [ ] Demo-Eingabe parat: **Langstrasse 4 Zürich, 80 m², 3.5 Zimmer, CHF 2400**.
- [ ] Fallback: Klappt die Demo nicht → Rechtsklick auf Folie 24–26 → «Folie einblenden».
- [ ] Zahlen-Spickzettel (Abschnitt 5) nochmals lesen.

---

## 2. Folie für Folie

### Folie 1: Titel

**Sagen:**
> Good morning. We are Team 6, Elias Martinelli and Timo Schlumpf, supervised by Dr. Elena Nazarenko. Our project is RentLens: an AI-supported rental platform for tenants and landlords that combines explainable rent prediction with a quality-of-life index.

**Erklären können:** Der Titel ist identisch mit dem Proposal. RentLens baut auf DSPRO1 auf (Mietpreisvorhersage), erweitert es aber um Unsicherheit, Wohnqualität und eine App.

### Folie 2: Agenda

**Sagen:**
> We start with the problem, then our goals and approach, the current status and first pilot results, a short live demo, and finally next steps and open questions.

### Folie 3–4: The Problem

**Sagen:**
> Imagine you find a listing: three and a half rooms, 80 square metres in Zurich, for CHF 2,400 per month. Two questions come up: is this rent fair, and is it a good place to live?
> Our DSPRO1 model could not really answer this. It gave one global error band for every flat, it systematically underestimated expensive flats, it was trained on a single snapshot in which only about 4,500 of 9,994 listings were fully enriched, and it never used the listing text.
> So in DSPRO2 we want to give every flat an estimated rent with a calibrated interval, a rent-check verdict, a quality-of-life score and a value score.

**Erklären können:**
- *Konstantes ± Band:* DSPRO1 sagte z. B. «± 425 CHF» für jede Wohnung. Eine günstige Studio-Wohnung und eine teure Attika haben aber ganz verschiedene Unsicherheit.
- *Teure Wohnungen unterschätzt:* Im obersten Preisquartil war der mittlere Fehler etwa doppelt so gross; das Modell zieht Vorhersagen zur Mitte.
- *Halbe Daten:* Von 9 994 Inseraten hatten nur ca. 4 500 alle Anreicherungsfelder (GWR etc.).
- *Text ungenutzt:* Die Beschreibung lag in der Datenbank (`listing_details.description`), wurde aber nie als Feature verwendet.

### Folie 5: Why is it relevant?

**Sagen:**
> Tenants get an explainable verdict on whether an asking rent is in line with the market, with an interval instead of one number. Landlords get a defensible price band and a what-if view before they publish a listing. And overall it brings more transparency into an opaque rental market. Important: this is not a legal assessment; the app links to the official advice pages.

**Erklären können:** «Fair» heisst bei uns *marktüblich im Vergleich zu aktuellen Angebotsmieten*, **nicht** «nicht missbräuchlich» im Sinne von Art. 269 ff. OR. Für die Anfechtung des Anfangsmietzinses (Art. 270 OR) verlinkt die App auf BWO und Mieterverband.

### Folie 6–7: Research Questions & Goals

**Sagen:**
> We agreed with our coach on three core priorities, done in this order.
> P1 is the spatial model: does boosting with spatial mixed effects, GPBoost, reduce the error in municipalities with few listings compared with LightGBM with an encoded municipality, and does it leave less spatial pattern in the errors?
> P2 is the quality-of-life index: can we build a valid index without ground truth?
> P3 are price intervals: do calibrated 80 percent intervals really contain 80 percent of rents, and are they narrower than one constant band?
> Text features are secondary, a small usability test is optional.

**Erklären können:**
- Die Nummerierung RQ1–RQ5 stammt aus dem Proposal; die **Priorität** (P1–P3) ist die Reihenfolge der Bearbeitung. Deshalb steht RQ2 vor RQ4 vor RQ3.
- *Warum diese Reihenfolge:* Zuerst muss klar sein, welches Punktmodell gewinnt (RQ2), dann werden die Intervalle für den Gewinner gebaut (RQ3). Der QoLI ist davon unabhängig.

### Folie 8: Planned Approach

**Sagen:**
> Our pipeline goes from listing to verdict. We take the listings, add building and location data, build features, compare two rent models, wrap the winner in calibrated intervals, and add the quality-of-life index and a value score in a Dash web app.
> Why this approach? Gradient boosting was our best tabular model in DSPRO1 and captures interactions like area times location. Spatial mixed effects let sparse municipalities borrow strength from their district. And conformal calibration guarantees the promised coverage without assuming a price distribution.

**Erklären können:**
- *GPBoost:* Gradient Boosting für die normalen Features **plus** Random Effects (Kanton > Bezirk > Gemeinde) **plus** ein Gaussian Process auf den Koordinaten (Mikrolage). Gemeinden mit wenig Daten werden zum Bezirkswert «geschrumpft» statt ignoriert.
- *LightGBM mit Target Encoding:* Die Gemeinde wird durch die durchschnittliche (log-)Miete in dieser Gemeinde ersetzt, **out-of-fold** berechnet, damit die eigene Miete nicht ins Feature leckt.
- *CQR:* siehe Glossar. Kurz: Quantilmodelle liefern ein Rohintervall, auf einem separaten Kalibrierungsset wird es so verbreitert oder verengt, dass genau 80 % der Mieten drin liegen.

### Folie 9: Evaluation Design

**Sagen:**
> To make our results trustworthy, we split the data once into 60 percent training, 20 percent calibration and 20 percent hold-out test. The split is grouped by object, so the same flat never appears in two sets, and stratified by canton and price.
> Our main metric is the mean absolute error in francs. Model selection uses grouped five-fold cross-validation on the training data only. Models are compared with paired tests: a bootstrap confidence interval and a Wilcoxon test, Holm-corrected. Moran's I tells us whether errors are still clustered in space. And we freeze the evaluation plan before the final run, so the test set is used only once.

**Erklären können:**
- *Warum gruppiert:* Dieselbe Wohnung wird oft mehrfach inseriert. Landet sie in Training **und** Test, misst man Auswendiglernen statt Generalisierung.
- *Warum Kalibrierungsset:* Conformal Prediction braucht Daten, die das Modell nie gesehen hat; sonst wären die Intervalle zu schmal.
- *Pre-Registration:* Hypothesen, Metriken, α und Seeds werden vorher festgelegt (`docs/EVALUATION_PLAN.md`, Tag `eval-plan-v1`). Ehrlich: Der Plan ist noch ein **Entwurf**; der Pilot lief vor dem Tag und gilt nur als Entwicklungslauf.

### Folie 10: Scope

**Sagen:**
> In scope are the new Comparis data via Apify, enrichment with official geodata and OpenStreetMap, the spatial model comparison, per-flat intervals, the quality-of-life index and value score, text features, the Dash app with public deployment on Azure, and experiment tracking with MLflow.
> Out of scope are rent forecasting over time, cooperative or subsidised rents and shared-flat rooms, a street-level map for privacy reasons, real-time feeds and any legally binding assessment. Aerial images and a fused transformer are optional only.

**Erklären können:** Genossenschafts-/subventionierte Mieten liegen weit unter Markt und würden das Ziel verzerren; sie werden ausgeschlossen und separat berichtet. Strassenebene würde fast individuelle Preise zeigen (Datenschutz) und hätte viele leere Zellen.

### Folie 11–12: Status

**Sagen:**
> Here is our progression. The proposal and research plan are done, and, a bit unusual, we already ran the full pipeline end to end on the DSPRO1 data. So every research question has a tested code path before the new data arrives. The web app prototype is in progress. Next are the Apify data and the data freeze on 6 November, then features and text, then final models and evaluation.

### Folie 13: Done vs. Planned

**Sagen:**
> To be clear about what is done and what is planned: done are the proposal, a data sourcing analysis, the cleaning pipeline from 9,405 listings to 8,125 objects, the end-to-end pilot for RQ2, RQ3 and RQ4 tracked in MLflow, a quality-of-life index for 2,110 municipalities and over 600 automated tests. In progress are the web app and the evaluation plan. Planned are the Apify data with descriptions, text features, GPBoost tuning, the confirmatory run on the frozen data and deployment.

**Erklären können:** Das Checklisten-Kriterium «clearly distinguish completed and planned work» erfüllt genau diese Folie. Wichtig: Alles in «Done» läuft **nur auf DSPRO1-Daten**.

### Folie 14: Data So Far

**Sagen:**
> Our pilot data is the DSPRO1 snapshot. We start with 9,405 raw listings. After deduplication to one row per object we remove 1,107 duplicates, about 12 percent. A few listings without a valid Swiss address or with implausible values go, and price-per-square-metre outliers. That leaves 8,125 objects, split into 4,875 training, 1,625 calibration and 1,625 test objects. Two gaps: no listing text in this snapshot, and 51.5 percent of objects lack building-register data.

**Erklären können:**
- *9 405 vs. 9 994:* 9 994 wurden in DSPRO1 extrahiert; im verwendeten CSV-Export sind 9 405 Zeilen.
- *Ausreisser:* Preis pro m² pro Gemeinde mit robustem z-Score > 3.5 (Parkplätze, Wochenpreise, Platzhalter).
- *Sprachregionen auf der Karte:* de 5 384, fr 2 331, it 572 Objekte. Das Tessin ist dünn besetzt.

### Folie 15: Data Source

**Sagen:**
> In DSPRO1 our source was Rentumo, scraped with our own Rust scraper: about 10,000 listings, one snapshot from April, and no listing text in the training data. For DSPRO2 we switch to Comparis, collected via Apify. The target is around 20,000 objects until the data freeze on 6 November, and this time the listing text is included, which makes RQ1 possible. Enrichment stays the same for both. This is a change compared with our proposal.

**Erklären können:**
- *Apify:* eine Plattform für Web-Scraping mit fertigen Scrapern («Actors»), die man konfiguriert statt selbst programmiert.
- *Abweichung vom Proposal:* Dort stand «eigener inkrementeller Scraper, Comparis bleibt ausgeschlossen». Offen ansprechen, nicht verstecken (siehe Q&A).
- *DSPRO1-Vorgeschichte (Sprechernotiz):* Comparis lehnte ab bzw. bot nur ca. 300 Datensätze mit Bewilligung an, Immoscout war nicht interessiert, Rentumo antwortete nicht.

### Folie 16: Data Pipeline High-Level Design

**Sagen:**
> Here is the pipeline. On the left the sources: Comparis via Apify for listings, official geodata and OpenStreetMap. Our Python package rentml cleans and deduplicates, anonymises the text and enriches every address. Everything is stored in a serverless PostgreSQL database on Neon. On the frozen snapshot we build features and the split, train the rent models, the intervals and the quality-of-life index, all tracked in MLflow. The result is served in a Dash web app, publicly deployed on Azure Container Apps by 11 December.

**Erklären können:**
- *Anonymisierung:* Kontaktangaben und Preise werden per Regex aus dem Text entfernt. So kann das Modell den Preis nicht aus dem Text ablesen, und es werden keine Personendaten gespeichert (revDSG).
- *Gestrichelte Kästen:* Meilensteine mit Datum (Data Freeze, Azure).
- *Ehrlich:* Der Apify-Import läuft noch nicht; der Pilot liest den DSPRO1-Snapshot.

### Folie 17: Data Enrichment

**Sagen:**
> In DSPRO1 we called an API for every single listing, which was slow and often timed out. Now we download the official layers once and join them locally by coordinate or building ID. Each address gets building data from the GWR, location data from swisstopo, public-transport quality from ARE, noise and air pollution from BAFU, population, green share and vacancy from BFS, walking access to amenities from OpenStreetMap, and sunshine and tax burden for the extended index. The maps show the four core quality-of-life dimensions per municipality.

**Erklären können:**
- *GWR / EGID:* Eidg. Gebäude- und Wohnungsregister; EGID ist die eindeutige Gebäude-ID.
- *Ehrlich:* Für den Pilot kommen die GWR-Daten noch aus der DSPRO1-Anreicherung; der GWR-Download für die Apify-Daten ist geplant.

### Folie 18–19: DSPRO1 Baseline vs. DSPRO2 Pilot

**Sagen:**
> Our baseline is the final DSPRO1 model, a gradient boosting model on all features plus geo clusters: mean absolute error 281 francs, RMSE 425, R squared 0.71, on the DSPRO1 evaluation split.
> In the DSPRO2 pilot, on a new grouped split, a naive reference that uses the median price per square metre of the municipality has an error of 355 francs. LightGBM with flat, building and area features gets 302, and adding the encoded municipality brings it down to 261 francs with an R squared of 0.74.
> This is not yet a fair comparison: different split, and DSPRO1 only used fully enriched listings. As planned, the new baseline is the DSPRO1 model retrained on the new data, and every DSPRO2 model has to beat that.

**Erklären können:**
- *MAE:* durchschnittlicher absoluter Fehler in CHF. «Im Schnitt liegen wir 261 CHF daneben.»
- *RMSE:* bestraft grosse Fehler stärker (Wurzel aus mittlerem quadriertem Fehler).
- *R²:* Anteil der erklärten Varianz (0 = so gut wie der Mittelwert, 1 = perfekt).
- *Warum RMSE im Pilot (483) höher als DSPRO1 (425), obwohl R² besser?* Anderes Testset: DSPRO2 enthält auch Objekte ohne Gebäudedaten und hat mehr Streuung. R² ist relativ zur Streuung, RMSE absolut. Deshalb «not yet comparable».
- *Grösster Hebel:* Das Encoding der Gemeinde (302 → 261 CHF).

### Folie 20: RQ2 Spatial Model

**Sagen:**
> The chart shows the test error per bucket of training listings per municipality. Our hypothesis was that GPBoost wins, especially in sparse municipalities. In the pilot it does not: LightGBM has 261 francs, GPBoost 269. The difference of minus 8.7 francs has a 95 percent confidence interval from minus 15 to minus 2, so LightGBM is significantly better. Even in municipalities with fewer than five listings, LightGBM wins by 18.5 francs. On the other hand, GPBoost's errors are less clustered in space.
> But GPBoost is not tuned yet. So we tune it before the final run, and for now LightGBM is our main model.

**Erklären können:**
- *Balken:* naive, dspro1_knn (DSPRO1-kNN-Features mit LightGBM neu trainiert), lgbm_te, gpboost; Fehlerbalken = 95 % Cluster-Bootstrap-CI.
- *Moran's I:* 0.017 (GPBoost) vs. 0.025 (LightGBM); beide nahe 0, nur deskriptiv berichtet.
- *Varianzanteile GPBoost:* Residuum 46.5 %, Gaussian Process 32.7 %, Kanton 17.4 %, Gemeinde 2.5 %, Bezirk 1.0 %; praktische Reichweite des GP 36.7 km. → Die Gemeinde allein erklärt wenig; Mikrolage (GP) und Kanton viel.
- *Zweistufiger Fit:* Die gemeinsame Schätzung kollabierte (Fehlervarianz → 0, weil mehrere Inserate dieselben Gebäudekoordinaten haben) und dauerte ~12 s pro Runde. Deshalb werden die Kovarianzparameter einmal geschätzt und dann fixiert.
- *Negatives Resultat ist ein Resultat:* Die Hypothese ist vorregistriert; wir berichten sie ehrlich.

### Folie 21: RQ3 Price Intervals

**Sagen:**
> An 80 percent interval should contain the true rent for eight out of ten flats. Our calibrated intervals reach 80.2 percent on the test set, and they hold in all twelve groups of language region times price band; the lowest is 76.9 percent, within tolerance. A constant band with the same overall coverage drops to 46 percent in its worst group.
> The downside: our intervals are on average 18.5 percent wider than the constant band, 851 versus 718 francs. So the second part of the hypothesis is not supported. That is the price of being reliable in every segment.

**Erklären können:**
- *Mondrian CQR:* Kalibrierung separat pro Gruppe (3 Sprachregionen × 4 Preisbänder = 12 Zellen), damit jede Gruppe ihre 80 % erreicht, nicht nur der Durchschnitt.
- *Warum breiter:* Teure Wohnungen brauchen in CHF sehr breite Intervalle; der Durchschnitt in CHF wird von ihnen dominiert. Das konstante Band «spart» dort und deckt dann teure Wohnungen schlecht ab.
- *Rohintervalle GPBoost:* 88.2 % Abdeckung, also zu breit ohne Kalibrierung.

### Folie 22: RQ4 Quality-of-Life Index

**Sagen:**
> The quality-of-life index is built from open data: public transport, noise, green space and access to services, following the OECD and JRC handbook. It covers all 8,125 objects and 2,110 municipalities. There is no ground truth, so we test it in two ways. First, stability: with random weights the median rank correlation is 0.93, but in the worst five percent it falls below our threshold. Second, validity: two of four checks hold. The index matches rent per square metre and official accessibility, but not vacancy or population growth. So the answer so far is: partially valid.

**Erklären können:**
- *Methode:* Min-Max-Normierung auf 0–100, gleiche Gewichte innerhalb einer Dimension, Gewichte über Dimensionen später vom Nutzer wählbar.
- *Stabilitätsregel:* Median-Spearman ≥ 0.90 **und** 5.-Perzentil ≥ 0.80. Erreicht: Median 0.926 / 0.949 (Objekte / Gemeinden), aber 5.-Perzentil nur 0.743 / 0.719 → nicht stabil.
- *Validität:* Miete pro m² ρ = 0.37 (erwartet positiv, ok); OSM vs. BFS-Distanz zu Dienstleistungen ρ = −0.74 (erwartet negativ, weil BFS eine Distanz misst, ok); Leerstand ρ ≈ −0.01 (nicht signifikant); Bevölkerungswachstum ρ = −0.06 (falsches Vorzeichen).
- *Lärm-Schwellen:* 16.1 % der Objekte über dem LSV-Immissionsgrenzwert Strasse, 42.4 % über dem WHO-Richtwert (genähert).

### Folie 23: Demo (live)

**Sagen (während der Demo):**
> Let me show you the prototype with exactly the flat from the beginning. I enter Langstrasse 4 in Zurich, 80 square metres, three and a half rooms, and an asking rent of 2,400 francs.
> The verdict: within the expected range. The model estimates 3,451 francs, with an 80 percent interval from about 2,200 to 4,340. The asking rent is around the 23rd percentile, so it is cheaper than most comparable offers. Below you see the main drivers: the rent level of the municipality and district.
> For landlords, the same flat gets a suggested listing band of about 2,430 to 3,520 francs, and with what-if you can see how area or rooms change the estimate. Finally, the map shows median rent per square metre by canton, district and municipality.

**Erklären können:**
- *Urteil:* unter / im / über dem 80-%-Intervall.
- *Angebotsband:* 25.–75. Perzentil der Quantilvorhersagen.
- *SHAP:* zeigt, welche Features die Schätzung nach oben oder unten ziehen.
- *Das Modell ist ein Pilot* auf DSPRO1-Daten; Marktplatz-Inserate sind synthetische Demo-Daten.
- *What-if Renovation/Balkon/Lift* braucht Textmerkmale (RQ1), kommt später.

### Folie 24–25: Next Steps

**Sagen:**
> Next steps: until 6 November the Apify data with descriptions, enrichment, data freeze and datasheet. Until 20 November text features for RQ1. Until 6 December tuning GPBoost, the confirmatory run and evaluation. Until 11 December the app and its deployment on Azure, and on 17 December the final presentation. The critical path is data freeze to confirmatory run; everything after it reuses the pipeline that already runs.

### Folie 26: Open Questions & Risks

**Sagen:**
> Some open questions. Data access: our proposal excluded Comparis because in DSPRO1 they asked for formal approval of automated collection. With Apify we need to clarify permission before the first run. Second, will GPBoost catch up once it is tuned? Third, can text features or better quantile models make the intervals narrower than a constant band? Fourth, which external benchmark can validate the quality-of-life index, since the BILANZ/IAZI ranking is not available to us? And 51.5 percent of objects lack building-register data.
> We would welcome your feedback: is our focus, spatial model, then quality of life, then intervals, the right one?

### Danke-Folie

> Thank you for your attention. We are happy to answer your questions.

---

## 3. Glossar (kurz und in eigenen Worten)

| Begriff | Erklärung |
|---|---|
| **MAE** | Mean Absolute Error: durchschnittlicher absoluter Fehler in CHF. Hauptmetrik, leicht verständlich. |
| **RMSE** | Root Mean Squared Error: wie MAE, aber grosse Fehler zählen stärker. |
| **R²** | Anteil der Streuung der Mieten, den das Modell erklärt. |
| **MAPE** | mittlerer prozentualer Fehler (Pilot: 14.2 %). |
| **Gradient Boosting / LightGBM** | viele kleine Entscheidungsbäume nacheinander; jeder korrigiert die Fehler der vorherigen. LightGBM ist eine schnelle Implementierung. |
| **Target Encoding (OOF)** | Kategorie (Gemeinde) wird durch die mittlere Zielgrösse ersetzt; out-of-fold, damit die eigene Miete nicht einfliesst (sonst Leakage). |
| **GPBoost** | Boosting + Mixed-Effects-Modell: Random Effects für Kanton/Bezirk/Gemeinde und Gaussian Process für die Koordinaten. Entwickelt von F. Sigrist (HSLU). |
| **Random Effect** | gruppenspezifische Abweichung (z. B. Gemeinde), die bei wenig Daten zum übergeordneten Mittel geschrumpft wird. |
| **Gaussian Process (GP)** | modelliert, dass nahe Orte ähnliche Abweichungen haben (räumliche Glätte). Vecchia-Approximation macht ihn rechenbar. |
| **Quantilregression** | sagt nicht den Mittelwert, sondern z. B. das 10. und 90. Perzentil voraus. Verlust: Pinball Loss. |
| **Monotone Constraint** | grössere Fläche → nie tiefere Schätzung (bei sonst gleichen Eingaben). Wichtig für glaubwürdiges What-if. |
| **Conformal Prediction / CQR** | nimmt die Quantil-Intervalle und korrigiert sie auf einem separaten Kalibrierungsset so, dass die gewünschte Abdeckung garantiert ist (ohne Verteilungsannahme). |
| **Mondrian CQR** | dasselbe, aber pro Gruppe separat (Sprachregion × Preisband). |
| **Coverage** | Anteil der echten Mieten, die im Intervall liegen. Ziel: 80 %. |
| **Grouped CV** | Kreuzvalidierung, bei der alle Inserate derselben Wohnung im selben Fold bleiben. |
| **Optuna** | automatische Hyperparameter-Suche (60 Versuche für LightGBM). |
| **Bootstrap-CI** | Testset tausendfach mit Zurücklegen ziehen → Unsicherheit einer Metrik oder Differenz. |
| **Wilcoxon-Test** | rangbasierter Test für gepaarte Fehler; Zweitcheck neben dem Bootstrap. |
| **Holm-Korrektur** | korrigiert p-Werte bei mehreren Tests, damit nicht zufällig etwas «signifikant» wird. |
| **MDE** | Minimum Detectable Effect: kleinster Unterschied, den das Testset zuverlässig nachweisen kann. Bei 1 625 Objekten ca. 5 % bzw. 14 CHF. |
| **Moran's I** | Mass für räumliche Autokorrelation: Haben benachbarte Wohnungen ähnliche Fehler? 0 = kein Muster. |
| **SHAP** | erklärt eine einzelne Vorhersage: wie viel jedes Feature beiträgt. |
| **QoLI** | Quality-of-Life Index, 0–100, aus offenen Daten; nach OECD/JRC-Handbuch für Composite Indicators. |
| **Value Score** | Wohnqualität im Verhältnis zum Preis. |
| **Spearman ρ** | Rangkorrelation; misst, ob zwei Rangfolgen übereinstimmen. |
| **GWR / EGID** | Gebäude- und Wohnungsregister des Bundes / eindeutige Gebäude-ID. |
| **LV95** | Schweizer Koordinatensystem (Meter). |
| **LSV** | Lärmschutz-Verordnung; Immissionsgrenzwerte für Wohnzonen 60 / 50 dB(A) Tag / Nacht. |
| **Apify** | Scraping-Plattform mit fertigen Scrapern (Actors). |
| **Neon** | serverlose PostgreSQL-Datenbank (schon in DSPRO1 genutzt). |
| **MLflow** | protokolliert jedes Experiment (Parameter, Metriken, Artefakte). |
| **Dash** | Python-Framework für interaktive Web-Apps mit Plotly. |
| **Azure Container Apps** | Cloud-Hosting für die App; skaliert auf 0, kostet im Leerlauf nichts. Schon in DSPRO1 genutzt. |
| **Data Freeze** | Stichtag (06.11.), ab dem die Daten für die finale Auswertung fix sind. |
| **Leakage** | Information aus dem Test fliesst ins Training → zu optimistische Resultate. |

---

## 4. Q&A-Vorbereitung

Antwort zuerst auf Englisch (so sagen), darunter bei Bedarf der Hintergrund.

**1. Why did GPBoost lose, if it was your main idea?**
> In the pilot it lost by 8.7 francs, but it is not tuned yet, while LightGBM was tuned with Optuna. So this is not a fair final comparison. We tune GPBoost before the confirmatory run. If it still loses, that is a valid negative result, and the variance decomposition is still useful: it shows that micro-location and canton matter much more than the municipality itself.

**2. Your intervals are wider than a constant band. Isn't that worse?**
> On average yes, by 18.5 percent. But the constant band covers only 46 percent in its worst group, so for expensive flats it is simply wrong. Our intervals hold 80 percent in every group. We prefer honest intervals over narrow ones; text features might make them narrower.

**3. Is scraping Comparis legal? Your proposal excluded it.**
> That is a fair point and a change from the proposal. In DSPRO1 Comparis asked for formal approval. We will clarify this with our supervisor before the first Apify run. Use is strictly academic and non-commercial, we do not redistribute raw listings, and the app only shows model outputs and aggregates.
>
> *Hintergrund:* Im Proposal steht zudem «technische Sperren werden nicht umgangen». Apify arbeitet oft mit Proxies; auch das vorher klären. Nicht behaupten, es sei sicher erlaubt.

**4. Why are the DSPRO1 and DSPRO2 numbers not comparable?**
> DSPRO1 used a random 80/20 split and only fully enriched listings. DSPRO2 uses a grouped split by object and includes listings with missing building data. As planned in the proposal, we retrain the DSPRO1 model on the new data, then both are compared on the same test set.

**5. How do you know the quality-of-life index is correct?**
> We cannot know it directly, there is no ground truth. So we check convergent validity with independent municipal statistics and stability under different weights. Right now it is partially valid: two of four checks hold, and the ranking is stable in the median but not in the worst cases.

**6. What does "fair" mean? Is this legal advice?**
> No. Fair means in line with current asking rents of comparable flats. It is not the legal notion of a non-abusive rent. The app gives no legal assessment and links to the official BWO and tenant-association pages.

**7. Why grouped splits? What is the leakage problem?**
> The same flat is often listed several times. If one copy is in training and another in test, the model just remembers the price. In DSPRO1 our leave-one-out kNN features on random folds made the cross-validation about 26 francs too optimistic. With grouped, out-of-fold features this optimism disappears.

**8. Is 1,625 test objects enough?**
> With 1,625 objects we can reliably detect differences of about 5 percent, or 14 francs. With the new data we aim for a larger test set; at 4,000 objects it would be about 3 percent.
>
> *Hintergrund:* Der Pilot-Vorsprung gegenüber DSPRO1 (12 CHF) liegt unter dieser Schwelle; deshalb nicht als Durchbruch verkaufen.

**9. Why LightGBM and not a neural network?**
> Tree boosting is the strongest choice for tabular data of this size. We tested a fused neural network in the pilot: 288 francs MAE versus 261 for LightGBM. A transformer over the text is an optional extension, used only if it wins the ablation.

**10. How will you use the listing text?**
> In three steps: simple features like keywords, TF-IDF and sentence embeddings; then structured attributes such as lift, balcony or renovation year, extracted by our own small model trained on about 300 hand-labelled plus 2,000 LLM-labelled listings; optionally a fused transformer. Each step must beat the previous one in the ablation.

**11. What if Apify does not deliver enough data by the freeze?**
> The pipeline already runs on the DSPRO1 data, so we can always deliver results. 20,000 is a target, not a guarantee; the learning curve shows how much data we actually need.

**12. Why Mondrian groups by language region and price band?**
> Because errors differ a lot between regions and price levels. A global calibration can reach 80 percent overall while covering expensive flats much less. Calibrating per group makes the promise hold for every segment.

**13. How do you handle personal data?**
> Contact details and prices are removed from the text at ingestion, only the anonymised text is stored. The fair-rent check stores no inputs, and the map hides cells with fewer than 20 listings.

**14. What is the difference between the calibration set and the test set?**
> The calibration set is only used to adjust the intervals. The test set is touched only once, for the final comparison. If we calibrated on the test set, the coverage would look perfect but mean nothing.

**15. What is your biggest risk?**
> Data access. Everything downstream already runs, so the critical path is getting the Apify data in time and with permission.

---

## 5. Zahlen-Spickzettel

| Was | Zahl | Quelle |
|---|---|---|
| DSPRO1 extrahierte Inserate | 9 994 (Snapshot 13.04.2026, rentumo.ch) | DSPRO1-Datasheet |
| DSPRO1 Baseline (GradientBoosting ALL+geo) | MAE 281, RMSE 425, R² 0.71, Gap 60 CHF | Proposal, `baseline.py` |
| Pilot-Rohdaten → Objekte | 9 405 → 8 298 (−1 107 Dubletten, 11.8 %) → 8 125 | `cleaning_log.md` |
| Split | 4 875 / 1 625 / 1 625 | `model_card.md` |
| Ohne GWR-Daten | 51.5 % | `feature_table.md` |
| Naive Referenz | MAE 355, R² 0.62 | `ablation.md` |
| LightGBM tabular | MAE 302, R² 0.66 | `ablation.md` |
| DSPRO1-kNN-Rezept (neu trainiert) | MAE 273, R² 0.71 | `ablation.md` |
| **LightGBM + Location** | **MAE 261** (CI 242–281), RMSE 483, R² 0.735, MAPE 14.2 % | `ablation.md` |
| Fused NN | MAE 288 | `ablation.md` |
| RQ2: LightGBM vs. GPBoost | 260.7 vs. 269.4 → −8.7 (CI −15.0 bis −2.2), p = 0.01 | `rq2_comparison.md` |
| RQ2 Gemeinden < 5 Inserate | −18.5 CHF (CI −29.3 bis −7.5) | `rq2_comparison.md` |
| Moran's I | GPBoost 0.017, LightGBM 0.025 | Pilot |
| GPBoost-Varianzanteile | Residuum 46.5 %, GP 32.7 %, Kanton 17.4 %, Gemeinde 2.5 %, Bezirk 1.0 % | `gpboost_variance_components.md` |
| RQ3 Coverage | 80.2 %; tiefste Zelle 76.9 % (it × B4, n = 13) | `rq3_intervals.md` |
| RQ3 Breite | 851 vs. 718 CHF (×1.185) | `rq3_intervals.md` |
| Konstantes Band, schlechteste Zelle | 46.2 % | `rq3_intervals.md` |
| QoLI Abdeckung | 8 125 Objekte, 2 110 Gemeinden | QoLI-Notebook |
| QoLI Stabilität | Median ρ 0.926 / 0.949; 5.-Perz. 0.743 / 0.719 (Regel: ≥ 0.90 und ≥ 0.80) | `qoli_weight_sensitivity.md` |
| QoLI Validität | Miete/m² ρ 0.37 ✓, BFS-Erreichbarkeit ρ −0.74 ✓, Leerstand ✗, Bev.-Wachstum ✗ | `qoli_convergent_validity.md` |
| MDE | 5 % / 14 CHF bei n = 1 625 | `mde_table.md` |
| kNN-Leakage DSPRO1 | 25.8 CHF zu optimistisch | `dspro1_knn_leakage.md` |
| Tests | 600+ (pytest) | Repo |
| Demo-Wohnung | Langstrasse 4 Zürich, 80 m², 3.5 Zi., CHF 2 400 → Schätzung 3 451, Intervall 2 196–4 344, ca. 23. Perzentil, Band 2 434–3 517 | Live-App |
| Termine | Data Freeze 06.11., Features 20.11., Modelle 06.12., App 11.12., Final 17.12., Abgabe 31.12. | Kanban |

---

## 6. Quellen (online geprüft)

**Quality-of-Life Index (Methode)**
- OECD & JRC (2008): *Handbook on Constructing Composite Indicators: Methodology and User Guide.* OECD Publishing, Paris. ISBN 978-92-64-04345-9. DOI [10.1787/9789264043466-en](https://doi.org/10.1787/9789264043466-en); Gratis-PDF: [jrc47008_handbook_final.pdf](https://knowledge4policy.ec.europa.eu/sites/default/files/jrc47008_handbook_final.pdf). → Normierung, Gewichtung, Aggregation, Sensitivitätsanalyse.
- OECD Better Life Index: [oecd.org/…/oecd-better-life-index](https://www.oecd.org/en/data/tools/oecd-better-life-index.html). → Vorbild für vom Nutzer wählbare Gewichte.

**Lärm-Grenzwerte Schweiz**
- Lärmschutz-Verordnung (LSV) vom 15. Dezember 1986, **SR 814.41**, Anhang 3 (Strasse) und Anhang 4 (Bahn). Empfindlichkeitsstufe II: Immissionsgrenzwert **60 dB(A) Tag / 50 dB(A) Nacht** (Planungswert 55/45, Alarmwert 70/65). Geprüft im Text Stand 1. Januar 2025 ([lexfind.ch/tolv/250001](https://www.lexfind.ch/tolv/250001/de)); offizielle Fassung auf [fedlex.admin.ch](https://www.fedlex.admin.ch), Suche «814.41».

**WHO-Richtwerte (nur Zusatzvergleich, nicht im Indexwert)**
- WHO Regional Office for Europe (2018): *Environmental Noise Guidelines for the European Region.* ISBN 978-92-890-5356-3. [who.int](https://www.who.int/europe/publications/i/item/9789289053563). Strasse 53 dB L_den (nachts 45 dB L_night); Bahn 54 dB L_den. Die WHO-Webseite datiert die Publikation auf den 30.01.2019.
- WHO (2021): *WHO global air quality guidelines: particulate matter (PM2.5 and PM10), ozone, nitrogen dioxide, sulfur dioxide and carbon monoxide.* ISBN 978-92-4-003422-8. [who.int](https://www.who.int/publications/i/item/9789240034228). Jahresmittel NO₂ 10 µg/m³, PM2.5 5 µg/m³.

**Modelle**
- Sigrist, F. (2022): *Gaussian Process Boosting.* JMLR 23(232), 1–46. [jmlr.org/papers/v23/20-322](https://jmlr.org/papers/v23/20-322.html)
- Romano, Y., Patterson, E., Candès, E. (2019): *Conformalized Quantile Regression.* NeurIPS 32. [arXiv:1905.03222](https://arxiv.org/abs/1905.03222)

**Ohne externe Quelle**
- Stabilitätsregel (Median ρ ≥ 0.90, 5.-Perzentil ≥ 0.80): eigene Festlegung im QoLI-Notebook. Das Handbuch verlangt die Sensitivitätsanalyse, nennt aber keine festen Schwellen.
- Datenquellen der Indikatoren (sonBASE, ARE, BFS, OSM usw.): Quelle, Lizenz und Referenzjahr in `docs/results/qoli_sources.md`.

**Falls gefragt wird (Englisch):**
> The index follows the OECD/JRC handbook on composite indicators. For noise we compare with the Swiss legal limits of the noise ordinance, and only as an approximate extra reference with the WHO guidelines; the WHO values do not change the index score. The stability thresholds are our own choice.

---

## 7. Nicht überverkaufen (ehrliche Punkte)

- Alle Resultate sind **Pilot** auf DSPRO1-Daten, noch nicht konfirmatorisch.
- Der Evaluationsplan ist ein **Entwurf** und noch nicht getaggt.
- GPBoost ist **nicht getunt**; das RQ2-Resultat ist vorläufig.
- Der Vorsprung gegenüber DSPRO1 ist nicht direkt vergleichbar und liegt unter der MDE.
- Die Apify-Datensammlung hat **noch nicht begonnen**, und die Erlaubnis ist ungeklärt.
- QoLI: nur teilweise valide, Stabilität im Worst Case nicht erfüllt.
- Die Marktplatz-Inserate in der App sind synthetische Demo-Daten.

Merksatz für schwierige Fragen: *«That is not known yet. Here is how we will find out.»*
