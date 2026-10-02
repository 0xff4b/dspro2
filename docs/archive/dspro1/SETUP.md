# Projekt einrichten und reparieren

## Zwei Befehle nach der Installation von Git und Python

Das Projekt nutzt Python **3.12, 64 Bit**. Auf Windows empfiehlt sich ein gewöhnlicher lokaler Ordner, beispielsweise C:/dev. Ein Windows-Python und ein WSL-Python brauchen jeweils eine eigene virtuelle Umgebung; verwende denselben Klon deshalb mit einem System.

Windows (PowerShell):

~~~powershell
git clone https://github.com/0xff4b/dspro1.git
py -3.12 dspro1/setup.py --start notebook
~~~

Linux, WSL oder macOS:

~~~bash
git clone https://github.com/0xff4b/dspro1.git
python3.12 dspro1/setup.py --start notebook
~~~

macOS: zuerst die OpenMP-Laufzeit mit **brew install libomp** installieren. Linux: libgomp1 muss vorhanden sein (Ubuntu bei Bedarf **sudo apt-get install libgomp1**). Windows benötigt für das normale Setup keinen Compiler. Der erste Download umfasst zahlreiche ML-Pakete; genügend freien Speicher und Internetzugang einplanen.

Das Setup verwendet die mitgelieferten CSVs und gespeicherten Modelle. Es prüft Git-Objekte, fehlende Dateien, Notebook-JSON, Python-Syntax, die Modell-/Trainingsdaten gegen den lokalen Commit sowie echte Vorhersagen in der App. Es führt weder das gesamte Trainingsnotebook noch Live-Scraping aus.

## Start und Paketumfang

Die folgenden Befehle gelten im Projektordner. Unter Windows **python3.12** durch **py -3.12** ersetzen.

| Zweck | Befehl |
|---|---|
| App, Notebook und ML-Umgebung einrichten | python3.12 setup.py |
| Alles einrichten und Notebook öffnen | python3.12 setup.py --start notebook |
| Alles einrichten und App starten | python3.12 setup.py --start app |
| Nur die Demo-Abhängigkeiten installieren und App starten | python3.12 setup.py --profile app --start app |
| Installation prüfen, ohne sie zu verändern | python3.12 setup.py --doctor |
| Sauberen Checkout aktualisieren und Setup ausführen | python3.12 setup.py --update |

Das erneute Setup verwendet bereits installierte passende Pakete wieder. Zwischen App- und Vollprofil werden keine zusätzlichen Pakete gelöscht. Der Notebook-Kernel heisst **DSPRO (.venv, Python 3.12)** und ist nur innerhalb der Projektumgebung registriert. JupyterLab und Streamlit starten auf der lokalen Loopback-Adresse.

Unter Linux/macOS sind **make setup**, **make app**, **make notebook** und **make doctor** gleichwertige Kurzbefehle; mit **make PYTHON=/pfad/zu/python3.12 setup** lässt sich der Interpreter vorgeben.

## Bestehende virtuelle Umgebung

Eine funktionsfähige .venv mit Python 3.12 wird wiederverwendet und mit den gesperrten Paketversionen abgeglichen. Pip wird bei Bedarf mit festgelegter Version und überprüfter Prüfsumme ergänzt, auch auf minimalen WSL-Installationen ohne ensurepip.

Ist die Umgebung defekt, von einem anderen Betriebssystem oder verwendet sie eine andere Python-Version, wird sie zuerst nach **.venv.backup-ZEITSTEMPEL** umbenannt. Das Setup legt danach eine neue .venv an. Bei Konflikten mit zusätzlich installierten Paketen:

~~~bash
python3.12 setup.py --recreate-env
~~~

Auch dabei bleibt die alte Umgebung als Backup erhalten. Vorher laufende App-/Notebook-Prozesse dieser Umgebung schliessen, insbesondere unter Windows. Backups werden nicht automatisch gelöscht. Eine aktive Conda-/pyenv-Umgebung ausserhalb des Projektordners und eine vorhandene .python-version werden nicht geändert. Die Befehle nennen deshalb Python 3.12 ausdrücklich.

## Dateien fehlen oder sind nach einem Pull beschädigt

1. **python3.12 setup.py --doctor** ausführen und die betroffenen Dateien ansehen.
2. Wenn diese Dateien auf die Version des lokalen Commits zurückgesetzt werden sollen: **python3.12 setup.py --repair**.
3. Das Setup sichert vorhandene betroffene Dateien zuerst unter **.setup-backups/files-ZEITSTEMPEL/** und stellt sie danach aus **HEAD** wieder her. Fehlende Dateien werden neu angelegt.

Wichtig: Ein abweichendes Modell oder eine abweichende CSV kann eine absichtliche Trainingsänderung sein. Das Setup kann das nicht von einem beschädigten Download unterscheiden und hält deshalb zunächst an. Eigene neue Modelle/Daten mit **--allow-local-data** erlauben; die Struktur- und Modellvorhersageprüfungen bleiben aktiv. Nicht jede beliebige Beschädigung ist erkennbar; insbesondere werden fachliche Ergebnisse eines gültigen Notebooks nicht überprüft.

Bereits vorgemerkte Änderungen im Git-Index werden nicht überschrieben. Bei ungelösten Merge-Konflikten bricht das Setup ab. Gültige lokale Code-/Notebook-Änderungen und neue eigene Dateien bleiben erhalten. **--update** arbeitet nur mit sauberem Arbeitsverzeichnis und einem Fast-Forward-Pull; es führt kein automatisches Stash, Reset oder Clean aus.

Ist die Git-Objektdatenbank selbst beschädigt, kann --repair nicht helfen: den bisherigen Ordner behalten und in einen **neuen Ordner** klonen. Eigene Änderungen anschliessend gezielt übernehmen.

Das frühere Verzeichnis **docs/dspro1/schemes mit abschliessendem Leerzeichen** wurde für Windows korrigiert. .gitattributes legt Zeilenenden fest und schützt Modelle/Bilddateien vor Textkonvertierung. Bei einem bereits fehlgeschlagenen alten Windows-Checkout ist ein neuer Klon in einen neuen Ordner oft der einfachste Einstieg.

## Docker für die Demo

Git und Docker Desktop bzw. Docker Engine mit Compose müssen vorhanden und gestartet sein:

~~~bash
git clone https://github.com/0xff4b/dspro1.git
docker compose -f dspro1/compose.yaml up --build
~~~

Danach http://localhost:8501 öffnen. Ctrl+C stoppt die App. **docker compose -f dspro1/compose.yaml down** entfernt den Container. Docker verwendet dieselbe App-Lockdatei wie das lokale Setup und kopiert keine lokalen Umgebungen oder Zugangsdaten ins Image.

## Umfang und externe Dienste

Die lokale Modell-Demo und das Hauptnotebook verwenden die eingecheckten Daten. Die Live-Adresssuche braucht Internetzugang zu GeoAdmin/GWR.

Die Datenbeschaffungs-Notebooks benötigen eigene PostgreSQL-Zugangsdaten und eine erreichbare Datenbank. Rust-Scraper benötigen separat Rust/Cargo und die jeweilige Konfiguration. Das Setup installiert die Python-Datenbankclients, richtet aber keinen Datenbankserver ein und lädt keine neuen Inserate herunter. Für **make report** sind zusätzlich LaTeX und BibTeX nötig.

Azure wird durch das lokale Setup weder gestartet noch verändert. Die Cloud-Automatisierung ist separat in [AZURE.md](AZURE.md) dokumentiert.

## Paketversionen und automatische Prüfung

requirements.txt ist die vollständige Eingabeliste; requirements-app.txt enthält die schlankere App-Liste und wird von der vollständigen Liste eingebunden. Installiert werden **requirements-full.lock** bzw. **requirements-app.lock**, jeweils mit exakten Versionen und Paket-Hashes. Die Installation akzeptiert nur fertige Wheels; fehlende Plattformpakete führen zu einem verständlichen Installationsfehler statt eines unerwarteten Compiler-Builds.

Nach einer bewussten Paketänderung beide Lockdateien regenerieren, beispielsweise in einer separaten Wartungsumgebung mit uv 0.12.17:

~~~bash
uv pip compile requirements.txt --universal --python-version 3.12 --generate-hashes --output-file requirements-full.lock
uv pip compile requirements-app.txt --universal --python-version 3.12 --generate-hashes --output-file requirements-app.lock
~~~

Anschliessend Setup und Tests ausführen. Die GitHub-Actions-Pipeline prüft einen frischen Checkout unter Windows, Ubuntu und macOS: Reparaturschutz, volle Installation, Modellvorhersagen, erneutes Setup und Doctor. Diese Pipeline veröffentlicht keine Cloud-App.

Neue DSPRO2-Dokumente gehören unter **docs/dspro2/**. Das gemeinsame Setup bleibt im Repository-Stamm.
