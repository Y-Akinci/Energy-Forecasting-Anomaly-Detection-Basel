# Energy-Forecasting-Anomaly-Detection-Basel
Machine-Learning-Modell zur Vorhersage und Anomalieerkennung im Energieverbrauch.

#@ Vorwort und Hintergrund 
xxxxxx

## Methodik
Im Rahmen unseres ML-Projektes haben wir uns für die CRISP-DM-Zyklus-Vorgehensweise (Cross Industry Standard Process for Data Mining) entschieden. Diese besteht aus sechs Phasen und bietet uns eine umfassende Orientierung und hilft uns die nächsten wichtigen Schritte im Überblick zu behalten.

### Business Understanding Data Understanding (Problemformalisierung)
- eine formalisierte Problembeschreibung ableiten
### Data Preperation
- Daten für Maschinelles Lernen der Formalisierung aufbereiten (Handling fehlender Werte,, Diskretisierung, Skalierung)
### Modeling
- Auswahl und Training des Modells (versch. Kriterien, Stärken & Schwächen und Parameter kennen)
### Evaluation
- ergebnisse beurteilen
- mögliche Probleme verstehen
- wirtschaftlicher Nutzten
### Deployment
- das entwickelte Modell oder die gewonnenen Erkenntnisse in die Praxis überführen.

(Das trainierte Prognosemodell wird als wiederverwendbare Python-Datei gespeichert. Ein separates Skript ruft das Modell regelmäßig auf und erstellt täglich eine Vorhersage des Stromverbrauchs für den Folgetag. Die Prognosen werden automatisch in einer CSV-Datei oder als Diagramm gespeichert.
Dadurch ist eine einfache, reproduzierbare Nutzung der Ergebnisse möglich, ohne manuelle Eingriffe.)

## Vorgehensweise
- Datenquellen gesucht. 
- Daten analysiert mit Phyton. Zielvariablen und Features festgelegt.
- Daten in Organge getestet geschaut was macht ein Tree, LR und Gradient Boosting. Metriken angeschaut.
- Recherche Libararys Phython für Modelltraining und Datenaufbereitung
- Weiter auf Phython mit Data Preparation, welche Daten können wichtig sein?
- Recherche und Auswahl vom Modell: Welche Modelle beachten Zeitreihen

## Business Understanding
### Business Case / Nutzen / IWB
### Formalisierung

## Data Understnading
### Aufbau Daten
Quelle
### Umgang mit Fehlenden Werten
#### Grundversorgte und Freie Kunden

#### Zeitumstellung
Beim Import und der Prüfung der Zeitreihe zeigte sich, dass 52 Viertelstunden-Zeitpunkte fehlen. Diese treten jeweils Ende Oktober in den Jahren mit Sommerzeit-Umstellung auf. Die Ursache liegt in der Umstellung von Sommer- auf Winterzeit: lokal werden diese Stunden doppelt gemessen, in der UTC-Darstellung erscheinen sie jedoch als Lücke.
Da UTC-Zeit keine Sommerzeitkorrektur enthält, wurde sie als Standardzeitzone beibehalten, um Dubletten zu vermeiden und die Daten konsistent zu halten.
Ab 2022 scheinen die Daten bereits zeitzonenbereinigt exportiert zu werden, was auf eine System- oder Formatumstellung der Quelle hindeutet.

## Modeling
## Evaluation
## Demployment
