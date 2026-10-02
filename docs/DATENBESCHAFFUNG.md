# Datenbeschaffung für DSPRO2

Stand der Recherche: 21. September 2026. Grundlage: vorhandenes DSPRO2-Proposal, DSPRO1-Datasheet, Trainings-CSVs und Scraper-Code sowie die unten verlinkten Primärquellen. Dies ist eine Beschaffungsanalyse; es wurden keine neuen Inserate gesammelt und keine Anbieter angeschrieben.

## Entscheidung

Nein, nicht alles scrapen. Empfohlen ist eine Kombination aus vorhandener Datenbank, amtlichen Downloads, OpenStreetMap und einem gezielten Zugang zu zusätzlichen Inseraten. Für den ersten Textmodell- und QoLI-Prototyp ist ein erneuter Vollscrape keine Voraussetzung. Für aktuelle oder zusätzliche Angebotsmieten braucht ihr dagegen einen Export, eine vereinbarte API oder eine zulässige eigene Erhebung.

Ein Gebäude- und Wohnungsregister ersetzt keinen Mietpreisdatensatz: Gebäudemerkmale sind Merkmale des Modells, die tatsächlich angebotene Nettomiete ist die Zielvariable. Angebotsmieten, laufende Vertragsmieten und Gemeindedurchschnitte müssen getrennt bleiben.

## 1. Was bereits vorhanden ist

Direkt im Arbeitsverzeichnis gezählt:

| Datei | Datenzeilen | Inhalt |
|---|---:|---|
| data/dspro1_snapshot/model.csv | 4’536 | 12 Spalten: Nettomiete, Fläche, Zimmer, Koordinaten und bestehende Anreicherung |
| data/dspro1_snapshot/model_wide.csv | 9’405 | 7 Spalten, darunter listing_id, Nettomiete, Fläche, Zimmer, Koordinaten, Bevölkerung |

Das sind überlappende Exporte; die Zahlen dürfen nicht zu einer Zahl einzigartiger Wohnungen addiert werden. Das DSPRO1-Datasheet dokumentiert 9’994 extrahierte Inserate vom 13. April 2026.

**Besonders relevant für DSPRO2:** Laut Datasheet existiert bereits listing_details.description. Auch der vorhandene Rust-Code liest Beschreibungstexte und schreibt sie in dieses Feld. In beiden Trainings-CSVs fehlt description. Deshalb zuerst einen vollständigen Export der bestehenden Datenbank prüfen. Die tatsächliche Befüllung und Qualität der Live-Datenbank wurde in dieser Analyse nicht abgefragt.

Der Export sollte mindestens listing_id, Quelle, Quell-ID/URL, Adresse, Koordinaten, EGID soweit vorhanden, Nettomiete, Nebenkosten, Fläche, Zimmer, Beschreibung und vorhandene Erfassungszeitpunkte enthalten. Fehlende historische Zeitpunkte nicht nachträglich als beobachtet darstellen.

Die Wiederverwendung muss zum vereinbarten Nutzungsumfang passen: Das vorhandene Datasheet dokumentiert für Rentumo keine erteilte Datenlizenz. Bereits gespeicherte Daten sind deshalb nicht automatisch zur öffentlichen Weiterverteilung freigegeben.

## 2. Quellen und Bezugswege

| Datentyp | Bezugsweg | Verwendung / Einschränkung |
|---|---|---|
| Gebäude, Wohnungen, Adressen, EGID | [GWR: öffentliche Daten als ZIP und Webdienste](https://www.gwr.admin.ch/de/data/supply/public_content.html) | Für grössere Mengen Download und lokale Verknüpfung. Der Anbieter nennt für MADD 20 Abfragen pro Minute; nicht auf beliebige andere APIs übertragen. Keine Inseratetexte oder Angebotsmieten. |
| ÖV-Güteklassen und Erreichbarkeit | [ARE: Verkehrserschliessung](https://www.are.admin.ch/de/verkehrserschliessung) | Geodaten herunterladen und Wohnungskoordinaten den Flächen zuordnen. |
| Strassen- und Bahnlärm | [BAFU: Lärm-Geodaten](https://www.bafu.admin.ch/de/laerm-geodaten) | Downloads für Tag/Nacht, derzeit auf der Seite Datenstand 2021. Modellierte Umgebungsexposition; keine Garantie für die konkrete Wohnung/Fassade. |
| NO2 und PM2.5 | [BAFU: Luft-Geodaten](https://www.bafu.admin.ch/de/luft-geodaten) | Rasterdownload statt Website-Scraping. Auf der geöffneten Seite Datenstand 2025. Auflösung, Einheiten und Bezugsjahr am tatsächlich gewählten Asset prüfen. |
| Bevölkerung / Arbeitsplätze | [BFS STATPOP-Datenbeschreibung](https://dam-api.bfs.admin.ch/hub/api/dam/assets/32266056/master), [BFS STATENT-Datenbeschreibung](https://dam-api.bfs.admin.ch/hub/api/dam/assets/36073026/master) | Bezug über BFS GEOSTAT; Hektardaten und Schutzregeln berücksichtigen. Klassierte kleine Werte nicht als exakte Zählwerte interpretieren. Aktueller Downloadstand und konkreter Nutzungsumfang sind vor Import festzulegen. |
| Bodennutzung | [BFS Arealstatistik](https://opendata.swiss/en/dataset/arealstatistik-der-schweiz) | Amtliche Geodaten. Für Umgebung und Grünanteil nutzbar; Hektarinformationen bilden nicht jedes kleine Grünstück ab. |
| Einkauf, Ärzte, Schulen, Parks, Fusswege | [Geofabrik: Schweiz-Extrakt](https://download.geofabrik.de/europe/switzerland.html) | OSM als PBF oder GeoPackage herunterladen, benötigte Kategorien lokal filtern. Abdeckung ist nicht überall gleich. |
| Sonnenscheindauer / Klimanormwerte | [MeteoSchweiz: Open Climate Data](https://opendatadocs.meteoswiss.ch/c-climate-data) | Für Wohnlage eher langfristige Normwerte als Tageswetter verwenden. Keine Aussage über die Besonnung eines bestimmten Fensters. |
| Gemeinden, Bezirke, Kantone | [swissBOUNDARIES3D](https://www.swisstopo.admin.ch/de/landschaftsmodell-swissboundaries3d) | Grenzen herunterladen; Gemeindenummer und Gebietsstand mit statistischen Tabellen abgleichen. |
| Luftbilder, optional | [swisstopo: kostenlose Geodaten und SWISSIMAGE](https://www.swisstopo.admin.ch/de/faq-kostenlose-geodaten) | Nur benötigte Kacheln/Ausschnitte beziehen. Luftbilder zeigen die Umgebung, nicht den Innenausbau der Wohnung. |
| Leerstand und statistische Mietvergleiche | [BFS Datenzugänge](https://data.bfs.admin.ch/) | Tabellen/API für regionale Kontextwerte und Plausibilisierung. Aggregate sind kein Ersatz für einzelne Inserate mit Text. |
| Steuerbelastung | [ESTV: Steuerbelastung in Gemeinden](https://www.estv.admin.ch/de/steuerbelastung-in-den-gemeinden) | Geeigneten aktuellen Export über die verlinkten Angebote wählen; Belastung hängt auch von Einkommen und Haushaltstyp ab. Für den ersten QoLI nicht zwingend. |

OSM steht unter [ODbL](https://www.openstreetmap.org/copyright?locale=en-GB). Quellenangabe und Bedingungen für veröffentlichte abgeleitete Datenbanken berücksichtigen. Auch bei amtlichen Daten pro Datensatz Quelle, Bedingungen und Datenstand dokumentieren.

**Technische Empfehlung:** Nationale oder regionale Dateien einmal beziehen, versionieren und lokal verknüpfen. Nicht für jedes der geplanten 30’000 Inserate dieselben Geodaten einzeln abrufen. APIs sind besonders für neue Einzeladressen und kleine Aktualisierungen sinnvoll.

## 3. Wie an zusätzliche Mietangebote kommen?

### A. Zuerst HSLU und kooperierende Verwaltungen

Über die Betreuung nach vorhandenen Forschungszugängen fragen. Die HSLU betreibt Immobilienforschung, beispielsweise zum [Nachfragemonitor Mietwohnungen](https://www.hslu.ch/de-ch/wirtschaft/ueber-uns/personensuche/person-detail-site/?pid=4982). Daraus folgt kein zugesicherter Zugang; es ist ein sinnvoller erster Kontaktweg.

Verwaltungen können einen CSV-/JSON-Export ihrer eigenen Inserate oder einen vereinbarten Datenfeed liefern. Anfragen sollten ausdrücklich Nettomiete, Nebenkosten, Fläche, Dezimalzimmer, Lage, Beschreibung, Inserate-ID und Zeitbezug nennen. Zusätzlich klären, ob Texte fürs Training, Ergebnisse in der Demo und abgeleitete Daten veröffentlicht werden dürfen. Für den Einstieg reichen Texte; Innenraumfotos erhöhen Aufwand und Rechteklärung.

### B. Portalzugang verhandeln

Die [SMG-API-Dokumentation](https://docs.api.re.swissmarketplace.group/) beschreibt einen Zugang für Professional-/Expert-Kunden zu deren eigenen Inseraten und Anfragen, gebunden an Agentur-IDs. Das ist kein frei zugänglicher Gesamtbestand aller Schweizer Mietwohnungen.

Die [aktuellen SMG-/Homegate-AGB](https://www.homegate.ch/c/de/ueber-uns/rechtliches/agb) erfassen auch ImmoScout24 und untersagen systematisches Auslesen. Für diese Quellen daher einen gesondert vereinbarten Forschungszugang oder Export anstreben. Eine robots.txt-Prüfung allein stellt keine solche Vereinbarung her.

### C. Gezielte zusätzliche Erhebung

Wo automatisierte Erhebung zulässig bzw. vereinbart ist, kann ein kleiner, inkrementeller Scraper sinnvoll sein. Erfasst werden neue und geänderte Inserate, nicht täglich sämtliche Seiten erneut. Sperren und Zugangsbeschränkungen werden respektiert.

Ein täglicher Lauf erzeugt nicht täglich neue einzigartige Wohnungen. Eine Wohnung kann über Wochen und auf mehreren Portalen erscheinen. Deshalb Objektidentität und Beobachtungszeitpunkt getrennt speichern. Mehrere Wohnungen im selben Gebäude dürfen ebenfalls nicht versehentlich zusammenfallen.

**30’000 einzigartige Objekte sind ein Ziel des Proposals, keine gesicherte Liefermenge.** Zuerst nach einer kurzen Pilotphase die Zahl neuer, eindeutiger und hinreichend vollständiger Objekte pro Woche messen. Dann den Umfang anhand der Lernkurve und regionalen Abdeckung festlegen.

## 4. Korrekturen und Risiken für euer Proposal

- **Lärmmetrik:** Die verlinkten BAFU-Downloads heissen Lr_Tag/Lr_Nacht. Der WHO-Bezug im Proposal verwendet Lden. Diese Metriken nicht unmittelbar gleichsetzen; passende Daten oder eine fachlich dokumentierte Methodik wählen.
- **Auflösung:** Die Angabe „PolluMap 20 m“ und „adressgenauer Lärm“ erst nach Prüfung des konkreten Downloads festschreiben. Ein an einer Adresse abgefragter Rasterwert bleibt ein Rasterwert.
- **Zimmer:** Der alte Rust-Scraper verwendet Ganzzahlfelder für rooms. Vor neuen Importen Dezimalzimmer und mögliche Verluste bei 2.5-/3.5-Zimmer-Angeboten prüfen.
- **Nettomiete:** Wenn ein Inserat nur Bruttomiete enthält, ist das kein verlässlicher Kaltmietwert. Herkunft und Umrechnung dokumentieren; nicht stillschweigend gleichsetzen.
- **Textmodell:** Preisangaben in Beschreibungen entfernen, bevor das Modell aus dem Text die Miete vorhersagt. Sonst kann es die Zielvariable direkt ablesen.
- **Dubletten und Evaluation:** Dasselbe Objekt und dessen Wiederholungsinserate zusammenhalten. Trainings-/Testtrennung räumlich und für Zeitvergleiche zusätzlich zeitlich planen.
- **Historische Daten:** Jeder Umgebungsdatensatz braucht ein Bezugsjahr. Für echte zeitliche Prognosetests nur damals verfügbare Merkmale verwenden.
- **QoLI-Validierung:** Zugang und Nutzbarkeit des im Proposal genannten BILANZ/IAZI-Rankings sind noch nicht bestätigt. Kein fest eingeplanter kostenloser Validierungsdatensatz.
- **Bestands- vs. Angebotsmieten:** Amtliche Mietstatistik oder Verwaltungs-Vertragsdaten können eine andere Zielpopulation abbilden. Nicht ohne Kennzeichnung mit Inseraten vermischen.

## 5. Empfohlene Reihenfolge

1. **Bestehende Daten sichern und exportieren.** Tatsächliche Textabdeckung, fehlende Nettomieten, Halbzimmer, Adressqualität und Dubletten messen. Gültige Daten mit fehlenden Zusatzmerkmalen nicht pauschal verwerfen.
2. **Ersten DSPRO2-Datensatz bilden.** Vorhandene Inserate mit Texten sowie GWR, ÖV, Lärm und OSM verbinden. LightGBM gegen ein Textmodell mit identischer Objektmenge und identischen Testgruppen vergleichen.
3. **Parallel Datenzugang anfragen.** Über HSLU sowie zwei oder drei kooperationsbereite Verwaltungen/Portale; Export oder Feed bevorzugen. Noch keine grosse Scraping-Infrastruktur aufbauen.
4. **Neue Quelle klein erproben.** Qualitätsbericht erstellen, Lizenz/Weitergabe dokumentieren, wöchentlichen Nettozuwachs messen. Erst dann automatisieren und auf weitere Regionen erweitern.
5. **QoLI ergänzen.** Luft, Grünflächen und Klima nach dokumentierter räumlicher Verknüpfung ergänzen. Luftbilder zeitlich begrenzen und erst nach einem funktionierenden Textvergleich beginnen.

Für einen günstigen Prototyp können Downloads, lokale räumliche Verknüpfungen und die erste Modellierung auf dem vorhandenen Rechner laufen. Ein dauerhaft laufender Cloud-Server ist dafür nicht erforderlich.

Pro Datensatz eine kleine Quellenliste führen: Anbieter, URL, Bezugsdatum, fachliches Bezugsjahr, räumliche Auflösung, Koordinatensystem, Lizenz, Dateiprüfsumme und Transformationsschritte. Den Datenbankexport separat versionieren; Rohtexte und grosse Datenbestände nicht automatisch in ein öffentliches GitHub-Repository übernehmen.

**Konkreter nächster Arbeitsschritt:** Ein nur lesender Datenbank-Qualitätsbericht, der die vorhandenen 9’994 Inserate mit den verfügbaren Beschreibungen und Merkmalen abgleicht. Erst danach lässt sich belastbar sagen, welche Daten wirklich neu beschafft werden müssen.
