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
### Aufbau Daten
#### Features
#### Messintervall, Prognosehorizont, Use-Cases
- Intervall: 15 Minuten (historisch kontinuierlich; wenige dokumentierte Interpolationen sind vernachlässigbar).

- Horizont: kurzfristig (nächste Stunden oder der folgende Tag).

* Einsatzbereiche:
  * Optimierung der Fahrweise von Erzeugung und Speichern.
  * Präzisere Marktzukäufe bei Bedarf.
  * Automatische Erkennung von Verbrauchsanomalien zur Prüfung.

Nachhaltigkeitssteuerung: Operativ kann auch mit 15-Minuten-Forecasts der Einsatz von Speichern und erneuerbarer Energie verbessert werden. Strategische Nachhaltigkeitsplanung (z. B. Monats- oder Jahresmix) liegt ausserhalb des Projektfokus.

### Quelle
### Abgrenzungen
- In Diesem Projekt haben wir durch zeitlichen Gründen auf Spannungs- oder Frequenzdaten verzichtet und somit keine technische Netz-Anomalieanalyse durchgeführt.
- Der Datensatz misst ausschliesslich den elektrischen Energiebezug aus dem öffentlichen Netz des Kantons Basel-Stadt.
Dadurch sind Eigenverbräuche, etwa aus Photovoltaikanlagen, die direkt vor Ort genutzt werden, nicht enthalten.
Der gemessene Gesamtverbrauch kann daher niedriger erscheinen, wenn ein Teil des Strombedarfs lokal durch Eigenproduktion gedeckt wird.
Einspeisungen aus dezentralen Anlagen (z.B PV-Überschüsse) werden nicht als separate Werte erfasst, beeinflussen aber indirekt die Netzbilanz, da sie den insgesamt benötigten Netzbezug reduzieren.
- Ebenfalls wird in diesem Projekt keine Marktpreis-Optimierung durchgeführt, da der Fokus gezielt auf Prognose und Überwachung liegt.


### Umgang mit Fehlenden Werten
#### Grundversorgte und Freie Kunden

#### Zeitumstellung
Beim Import und der Prüfung der Zeitreihe zeigte sich, dass 52 Viertelstunden-Zeitpunkte fehlen. Diese treten jeweils Ende Oktober in den Jahren mit Sommerzeit-Umstellung auf. Die Ursache liegt in der Umstellung von Sommer- auf Winterzeit: lokal werden diese Stunden doppelt gemessen, in der UTC-Darstellung erscheinen sie jedoch als Lücke.
Da UTC-Zeit keine Sommerzeitkorrektur enthält, wurde sie als Standardzeitzone beibehalten, um Dubletten zu vermeiden und die Daten konsistent zu halten.
Ab 2022 scheinen die Daten bereits zeitzonenbereinigt exportiert zu werden, was auf eine System- oder Formatumstellung der Quelle hindeutet.

## Data Preparation

## Modeling
## Evaluation
## Demployment
