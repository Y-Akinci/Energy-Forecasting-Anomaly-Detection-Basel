# Energy-Forecasting-Anomaly-Detection-Basel
Wir entwickeln für Basel-Stadt ein datengetriebenes System zur Energie-Lastprognose und Anomalie-Erkennung. Auf Basis historischer Last-, Wetter- und Kalenderdaten sollen kurz- bis mittelfristige Forecasts gelingen und Abweichungen früh präventiv erkannt werden um Beschaffung, Netzbetrieb und Peak-Management verlässlicher und kosteneffizienter zu machen.


## Methodik
### CRISP-DM
Im Rahmen unseres ML-Projektes haben wir uns für die CRISP-DM-Zyklus-Vorgehensweise (Cross Industry Standard Process for Data Mining) entschieden. Dieses Vorgehensmodell gliedert den gesamten Datenanalyseprozess, von der Problemdefinition bis zur praktischen Nutzung des Modells, in sechs klar definierte Phasen.
CRISP-DM bietet uns eine strukturierte und iterative Vorgehensweise, um systematisch Daten zu verstehen, Modelle zu entwickeln und deren Nutzen im realen Umfeld zu evaluieren.
<img width="547" height="501" alt="image" src="https://github.com/user-attachments/assets/bd337948-9179-4135-807f-55f50c3d952d" />


### Business Understanding Data Understanding (Problemformalisierung)
- In dieser Phase wird das Problem fachlich verstanden und in eine formalisierte Aufgabenstellung für das Machine Learning übersetzt.
Zudem erfolgt eine erste Analyse der vorhandenen Datenquellen und -qualität.
### Data Preperation
- Die Daten werden aufbereitet, bereinigt und für das Modelltraining vorbereitet (Handling fehlender Werte,, Diskretisierung, Skalierung)
### Modeling
- Auswahl und Training des Modells (versch. Kriterien, Stärken & Schwächen und Parameter kennen)
### Evaluation
- Überprüfung der Modellleistung anhand geeigneter Metriken.
- Ziel ist es, die Ergebnisse kritisch zu bewerten, Probleme zu erkennen und den wirtschaftlichen Nutzen zu beurteilen.
### Deployment
- das entwickelte Modell oder die gewonnenen Erkenntnisse in die Praxis überführen.

(Das trainierte Prognosemodell wird als wiederverwendbare Python-Datei gespeichert. Ein separates Skript ruft das Modell regelmäßig auf und erstellt täglich eine Vorhersage des Stromverbrauchs für den Folgetag. Die Prognosen werden automatisch in einer CSV-Datei oder als Diagramm gespeichert.
Dadurch ist eine einfache, reproduzierbare Nutzung der Ergebnisse möglich, ohne manuelle Eingriffe.
### SCRUM

## Business Understanding
### Ziel und Scope
Das Projekt entwickelt ein zweiteiliges System für Basel-Stadt:
- Nachfrageprognose des Stromverbrauchs auf 15-Minuten-Basis.
- Anomalieerkennung im Verbrauchsverhalten.

Datenumfang: viertelstündlicher Netzbezug im Kanton Basel-Stadt inklusive Netzverluste.
Nicht enthalten: lokal erzeugter und direkt vor Ort verbrauchter PV-Strom.
Kundengruppen im Datensatz: grundversorgte und freie Kunden. Beide beeinflussen die Netzlast.

Zielvariable: Gesamtverbrauch (Summe beider Kundengruppen).
Für Netzführung und Lastplanung ist der gesamte Effekt im Netz relevant. Optional können später Submodelle pro Kundengruppe erstellt werden.

### Hintergrund
IWB Industrielle Werke Basel ist Versorger und Netzbetreiber: eigene Erzeugung (Wasserkraft, KWK, Solar) und marktseitiger Zukauf bei Bedarf.

Grundversorgte Kunden: beziehen Energie direkt von IWB.
Freie Kunden: kaufen Energie am Markt, nutzen aber das IWB-Netz. Deren Last ist für die Netzplanung ebenso relevant.

### Business Nutzten
#### Nachfrageprognose
1. Planungssicherheit: Eigenproduktion, Speicher, Lastverschiebung und gezielter Marktzukauf können früh geplant werden.
2. Kosteneffizienz: Weniger teurer Spot-Zukauf, optimierte Fahrpläne für Erzeugung und Speicher, geringere Regelenergie.

3. Operative Steuerung: Personaleinsatz, Instandhaltung und Anlagenbetrieb können an den erwarteten Verbrauch angepasst werden.

Hinweis: IWB kauft nicht täglich Strom, sondern nur bei Bedarf. Prognosen dienen dazu, diese Entscheidungen präziser und kostengünstiger zu treffen.

#### Anomalieerkennung
- Erkannt werden ungewöhnliche Verbrauchsmuster, keine technischen Netzstörungen.

- Vorgehen: Vergleich Ist-Verbrauch vs. erwarteter Verbrauch (Prognosefehler oder statistische Abweichung).

- Keine Netzkapazitätsdaten nötig – Fokus liegt auf Verbrauchsanomalien (sprunghafte Peaks, untypische Tagesmuster, fehlerhafte Messungen).

- Nutzen: Frühwarnung für Prüfungen, Qualitätssicherung, Messfehler-Erkennung und Auffälligkeiten im Verbrauchsverhalten.

### Technischer Zielzustand
Für das Projekt wird ein reproduzierbares Feature-Set entwickelt, das zeitliche Merkmale (z. B. Stunde, Wochentag, Monat, saisonale Muster) sowie abgeleitete Werte wie Lags, gleitende Durchschnitte und optional Feiertagsinformationen umfasst. Auf dieser Basis wird ein Regressionsmodell trainiert, um den Stromverbrauch präzise vorherzusagen und die Modellgüte anhand transparenter Metriken zu bewerten. Zusätzlich wird ein Anomalie-Flag pro Zeitintervall erzeugt, das auf Prognoseabweichungen oder unüberwachten Scores basiert. Die täglichen Prognosen und Erkennungen werden automatisch in einer CSV-Datei oder als Diagramm exportiert.

## Data Understnading

### Datenquellen:
#### Stromverbauch Daten von Basel
- [Data](data/251006_StromverbrauchBasel2012-2025.csv)
- [Link](https://de.wikipedia.org/wiki/Data_Science](https://opendata.swiss/de/dataset/kantonaler-stromverbrauch-netzlast))
#### Meteo Daten
- [Data]()
- [Link]()

### Datenstruktur:  
- Excel/csv Format
#### Stromverbauch Daten von Basel
- Total 481960 Zeilen, 11 Spalten
- Wichtige Variablen: Kundentyp, Verbrauch, Saison
#### Meteo Daten

### Erste Analysen:  
- Histogramm der Verbrauchswerte (siehe /plots/verbrauch.png)
- Stark saisonale Schwankungen im Feld 'Monat'

### Datenqualität: 
- Ausreißer im Monat Dezember 2023

### Relevante Felder:  
- Kundentyp: frei/privat
- Verbrauch: kWh pro Monat
- Zeitraum: Januar 2022–Dezember 2024


## Data Preparation

## Modeling
## Evaluation
## Demployment
