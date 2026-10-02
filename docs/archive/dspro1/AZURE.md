# Streamlit-Prototyp auf Azure

## Gewählte Sparvariante

Azure Container Apps **Consumption-only**, 0 bis 1 Instanz, 0.25 vCPU und 0.5 GiB RAM,
HTTPS/WebSockets auf Port 8501. Ohne Log-Analytics-Workspace, Datenbank, VM,
VNet oder kostenpflichtige dedizierte Instanz. Das Docker-Image kann in der
privaten GitHub Container Registry (GHCR) liegen, damit keine Azure-Registry
bezahlt werden muss.

Azure gewährt pro Abonnement und Monat 180'000 vCPU-Sekunden, 360'000 GiB-Sekunden
und 2 Millionen Requests. Bei dieser Konfiguration entsprechen die beiden
Rechen-Freimengen rechnerisch etwa 200 aktiven Stunden, sofern keine anderen
Apps dieselben Freimengen verbrauchen. Das ist **keine Kostenobergrenze**.
Traffic, dauerhafte Nutzung und Builds jenseits der GitHub-Actions-Freimenge
können kostenpflichtig sein. Der Screenshot der 12-Monats-Angebote ist keine
Zusicherung für alle Ressourcen und jede Nutzungsmenge.

Quellen: [Azure-Preise](https://azure.microsoft.com/en-us/pricing/details/container-apps/),
[GHCR-Abrechnung](https://docs.github.com/en/billing/concepts/product-billing/github-packages).

Geschlossene Browser-Tabs begünstigen das Herunterskalieren. Streamlit verwendet
dauerhafte WebSocket-Verbindungen; ein offener Tab kann die Instanz aktiv halten.
Nach Herunterskalieren oder Neustart gehen Sitzung und Cache verloren. Der erste
Aufruf kann wegen des Kaltstarts länger dauern. Keine Besucherdateien werden
dauerhaft gespeichert.

## Lokal mit Docker

In WSL/Linux im Projektordner:

```bash
docker compose up --build -d
# http://localhost:8501
docker compose down
```

Der Container läuft ohne Root-Rechte, mit Healthcheck, ohne Notebook-Tools und
mit festgeschriebenen Python-Paketen sowie Basis-Image-Digest.
Das unvollständige `gradient_boosting_all_geo.joblib` wird nicht ins Image
übernommen: Es benötigt ein Geo-Cluster-Feature, dessen Transformation im
Export fehlt. Die Originaldatei im Repository bleibt erhalten.
Vier andere Modelle einschließlich des Standardmodells bleiben verfügbar.

Funktionstest einschließlich Modellwechsel und aller Seiten:

```bash
docker build -t dspro1-streamlit:prototype .
docker run --rm --memory 512m --cpus 0.25 \
  -v "$PWD/scripts/smoke_app.py:/tmp/smoke_app.py:ro" \
  dspro1-streamlit:prototype python /tmp/smoke_app.py
```

## Azure anmelden und bedienen

`scripts/az.sh` verwendet eine installierte Azure CLI oder automatisch deren
Docker-Image. Die Docker-Variante legt Anmeldedaten unter `.azure-local/` ab;
dieser Ordner ist von Git und Docker ausgeschlossen. Docker muss laufen.

```bash
bash scripts/azure.sh login
bash scripts/azure.sh status
bash scripts/azure.sh stop
bash scripts/azure.sh start
bash scripts/azure.sh delete --yes
```

`stop` deaktiviert die aktiven Revisionen; ein Besucher kann sie danach nicht
automatisch starten. `start` aktiviert die letzte erfolgreiche Revision wieder.
`delete --yes` löscht die gesamte markierte Ressourcengruppe
`rg-dspro1-prototype`, einschließlich einer dort eventuell vorhandenen ACR.
Vor der Löschung werden die Ressourcen angezeigt. Fremde Ressourcengruppen
ohne die passende Besitzmarkierung werden abgelehnt. In dieser Gruppe keine
projektfremden Ressourcen ablegen.

Das Abonnement ist auf `f896f8e6-927d-4630-8d87-fdcead25030e` voreingestellt.
Overrides: `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_LOCATION`,
`AZURE_APP_NAME`, `AZURE_ENVIRONMENT_NAME`, `AZURE_REGISTRY_NAME`.

## Automatischer Upload über GitHub (empfohlen)

1. Funktionierenden GitHub-Zugang zum Projekt herstellen: `gh auth login`.
2. Einmalig Azure anmelden und OIDC einrichten:
   `bash scripts/setup_github.sh OWNER/REPOSITORY`.
   Dafür werden Rechte zum Erstellen von Identitäten und Rollenzuweisungen
   benötigt. Das Skript prüft den Repository-Zugriff zuerst.
3. In den Repository-Actions-Variablen `GHCR_USERNAME` setzen.
   Als Actions-Secret `GHCR_READ_TOKEN` einen classic PAT mit
   `read:packages` und Zugriff auf das private Image hinterlegen.
   Keinen Token in Code, Dokumentation oder Chat eintragen.
4. Änderungen ins Repository übernehmen. Unter Actions den Workflow
   **Azure Streamlit prototype** mit `deploy` starten.
5. Für automatischen Upload nach Änderungen auf `main` die Repository-Variable
   `AZURE_AUTO_DEPLOY=true` setzen.

Das Setup speichert die übrigen Azure-Variablen automatisch. Der Workflow baut
und testet zuerst, veröffentlicht dann das Image in GHCR und startet die
Azure-App. Neue GHCR-Pakete sind standardmäßig privat. Bereits existierende
Pakete behalten ihre Sichtbarkeit: ein eigenes privates Paket verwenden.
Azure-Zugriff verwendet OIDC, also kein dauerhaftes Azure-Passwort.
Die private Registry benötigt einen langlebigen Lesetoken zum Abrufen des
Images auch bei späteren Kaltstarts.

Im selben Workflow lassen sich `start`, `stop`, `status` und `delete` auswählen.
Für `delete` muss zusätzlich der Ressourcengruppenname eingegeben werden.
Vor dauerhaftem Löschen `AZURE_AUTO_DEPLOY=false` setzen. Nach dem Löschen muss
`setup_github.sh` erneut ausgeführt werden, weil die CI-Berechtigung auf die
gelöschte Gruppe begrenzt war.

Die separate Gruppe `rg-dspro1-automation` enthält nur die CI-Identität und
verursacht keine eigenen Compute-Kosten. Sie und das GHCR-Paket bleiben bei
`delete` erhalten, um spätere Deployments zu ermöglichen.

## Bereits vorhandenes Image deployen

```bash
DEPLOY_IMAGE=ghcr.io/owner/repository:version \
REGISTRY_USERNAME=github-login \
REGISTRY_PASSWORD="$GHCR_READ_TOKEN" \
bash scripts/azure.sh deploy
```

Bei öffentlichen Images die beiden Registry-Zugangsdaten weglassen.
Ein extern bereitgestelltes Image muss vorab getestet sein; der integrierte
Funktionstest wird beim lokalen Build und im GitHub-Workflow ausgeführt.

## Alternative ohne GitHub

```bash
bash scripts/azure.sh deploy
```

Ohne `DEPLOY_IMAGE` baut und testet das Skript lokal, erstellt eine **private
Azure Container Registry Basic**, lädt das Image hoch und startet die App.
Dieser Weg verursacht Registry-Grundkosten auch bei gestoppter App.
Die im Screenshot angezeigte Standard-Registry-Freimenge gilt nicht automatisch
für Basic. Für die günstigste Variante den GHCR-Weg verwenden.
Azure liest ACR-Images mit Managed Identity; der Registry-Adminzugang bleibt aus.

Nach jedem Deploy prüft das Skript den Streamlit-Health-Endpunkt und zeigt die URL.
Cloud-Bereitstellung und automatische Uploads gelten erst nach einem erfolgreichen
Azure-Deploy beziehungsweise GitHub-Workflow als geprüft.

## Aktuelle Bereitstellung

- App: https://dspro1-streamlit.thankfulpond-a9641a83.switzerlandnorth.azurecontainerapps.io
- Region: Switzerland North.
- [GitHub-Steuerung: Azure Streamlit prototype](https://github.com/0xff4b/dspro1/actions/workflows/azure-prototype.yml)
- Stand 21.09.2026: App gestoppt (alle Revisionen inaktiv); `AZURE_AUTO_DEPLOY=false`.
- Für den nächsten Event im Workflow **deploy** wählen: baut und startet den neuesten Designstand. **start** startet nur das zuletzt bereits veröffentlichte Image.
- Nach der Demo **stop** wählen. Die Automatik bleibt ausgeschaltet, bis `AZURE_AUTO_DEPLOY` ausdrücklich wieder auf `true` gesetzt wird.
- Erfolgreicher automatischer Lauf mit 0.25 vCPU / 0.5 GiB:
  https://github.com/0xff4b/dspro1/actions/runs/35548950229
- Für den initialen privaten Registry-Zugriff wurde der autorisierte
  GitHub-CLI-Token als Actions-Secret hinterlegt. Er kann durch einen dedizierten
  classic PAT mit ausschließlich `read:packages` und Paket-Zugriff ersetzt werden.

Live-Prüfung der HTTPS-Seite, der WebSocket-Verbindung und einer tatsächlich
gerenderten Modellvorhersage:

```bash
docker run --rm \
  -v "$PWD/scripts/check_deployment.py:/tmp/check_deployment.py:ro" \
  dspro1-streamlit:prototype python /tmp/check_deployment.py \
  https://dspro1-streamlit.thankfulpond-a9641a83.switzerlandnorth.azurecontainerapps.io
```

Die Löschfunktion ist durch Tests mit einer simulierten Azure CLI abgesichert;
die bereitgestellte App wird zur Prüfung nicht gelöscht.
