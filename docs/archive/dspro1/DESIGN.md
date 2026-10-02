# AI Event: App und Notebook

Das Layout kombiniert das HSLU-Projektposter mit der grosszügigen Typografie, ruhigen Flächen und dem direkten Einstieg der Referenz https://runway.com/. Logo, Texte und Diagramme gehören zum DSPRO1-Projekt; Runway-Medien werden nicht übernommen.

## Website

- Einstieg mit HSLU-Logo, Projekttitel und «Demo entdecken».
- «Beispielwohnung» funktioniert ohne externe Adressdienste. Standort, Fläche und Zimmer ändern die echte Modellvorhersage; weitere Eingaben sind feste, offen ausgewiesene Beispielwerte.
- «Adresse suchen» ist der Standard: Ein einziges Adressfeld sucht nach kurzer Tipppause ab drei Zeichen automatisch bei GeoAdmin. Die offizielle Adresse wird direkt in der Vorschlagsliste per Klick oder Tastatur bestätigt; Gebäude, Wohnungen und Schätzung laden danach automatisch. Beim Ändern der Adresse wird die alte Auswahl verworfen. Externe Dienste bleiben für echte Adressen erforderlich.
- Modellwahl und zusätzliche Eingaben befinden sich in der einklappbaren Seitenleiste.
- Logo, Schrift und vorhandene Projektgrafiken werden lokal im Container mitgeliefert.
- Ein einheitliches helles Theme verhindert einen Kontrastkonflikt mit nativen Formularen bei dunkler Systemeinstellung. Schmale Fenster, Touch-Flächen, iPhone-Sicherheitsabstände und reduzierte Animationen sind berücksichtigt.

## Notebook und Python-Diagramme

`src/plot_style.py` definiert die gemeinsame Farbpalette, Schriften, Achsen, Raster und Exportauflösung. Die App und die Stilzelle von `src/notebooks/model_v3_clean.ipynb` verwenden diese Einstellungen. Notebook-Titel und HSLU-Logo sind angepasst.

Bei einem bereits laufenden Notebook die Stilzelle erneut ausführen und danach die gewünschten Diagrammzellen. Gespeicherte Notebook-Bilder und vorhandene PNG-Grafiken ändern sich nicht rückwirkend. Datenaufbereitung, Trainingszellen, Modelle und Ergebnisse wurden für das Design nicht verändert oder neu trainiert.

## Lokal prüfen

```bash
docker build -t dspro1-streamlit:design .
docker run --rm -p 127.0.0.1:18501:8501 dspro1-streamlit:design
```

Vorschau: http://127.0.0.1:18501

`scripts/smoke_app.py` prüft im Container alle ausgelieferten Modelle und Seiten. `scripts/check_browser_ui.cjs` prüft eine laufende Vorschau mit Playwright in installiertem Chrome, Edge sowie Android- und iPhone-Emulation. Benötigt Playwright und dessen WebKit-Browser; `NODE_PATH` und `PLAYWRIGHT_BROWSERS_PATH` können auf vorhandene Installationen zeigen. Für die Live-Adressprüfung `TEST_ADDRESS` auf eine vollständige Schweizer Adresse setzen; die externen GeoAdmin-Dienste müssen erreichbar sein. Mobilprüfungen ersetzen keinen Test auf physischen Geräten.

Azure bleibt während der Designarbeit ausgeschaltet. Zum nächsten Start mit diesem Design siehe [AZURE.md](AZURE.md): im GitHub-Workflow `deploy` wählen.

## Abbildungsexport

Das Notebook schreibt weiterhin PNGs nach docs/dspro1/final-report/fig/ und zusätzlich SVGs nach docs/dspro1/fig/poster-ai-event/. Siehe [Exportanleitung](fig/poster-ai-event/README.md).
