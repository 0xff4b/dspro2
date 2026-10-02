# AI-Event: Poster-Abbildungen

Das Notebook src/notebooks/model_v3_clean.ipynb exportiert seine Abbildungen jetzt in zwei Formaten:

- PNG für den Bericht: docs/dspro1/final-report/fig/
- SVG für das Poster: docs/dspro1/fig/poster-ai-event/

Im laufenden Notebook zuerst die gemeinsame Stilzelle und die anschliessende save_fig-Zelle erneut ausführen. Danach die gewünschten Diagrammzellen oder die abschliessende Exportzelle ausführen. Bei einem frischen Kernel werden zuvor die Zellen benötigt, die die jeweiligen Daten und Modelle erzeugen.

Alle bisherigen Exportwege, einschliesslich der Wide-Experiment-Abbildungen, verwenden denselben Exporthelfer. Die SVGs werden direkt aus den Matplotlib-Figuren erzeugt; bestehende PNGs werden nicht in SVG-Hüllen verpackt. Texte bleiben als SVG-Text bearbeitbar. Die verwendeten Schriften müssen im Layoutprogramm verfügbar sein.

Vorhandene Notebook-Ausgaben und PNGs werden durch diese Codeänderung nicht rückwirkend erneuert. Modelle und Ergebnisse wurden dafür nicht neu trainiert.
