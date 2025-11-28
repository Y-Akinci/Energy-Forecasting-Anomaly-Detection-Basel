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
Für das effiziente und zielorientierte Management verwendeten wir die agile Scrum-Methode. Ein zyklischer Framework, wo in jedem Zyklus, auch Sprint genannt, ein bestimmtes Ziel gesetzt wird. Dieser dauert mehrere Wochen und hat designierte Rollenverteilungen, wie Prodct-Owner, Scrum Master und das Entwicklungsteam. In regelmässigen Scrum-Meetings wird der Fortschritt von jedem einzelnen Teammitglied evaluiert, und neue Aufgaben werden entsprechend zugewiesen. Am Ende jedes Zyklus wird eine Sprint-Retrospective durchgeführt. Ein Rückblick in dem Positives sowie Negatives betrachtet wird um entsprechende strategische Entscheidungen für den nächsten Sprint zu treffen.

Im Rahmen des Projektes übernimmt jeweils ein Teammitglied die Rolle als Scrum-Master und leitet die Teammitglieder. Einen Product-Owner gibt es in diesem Scenario nicht, wobei unser Dozent gewisse Charakteristiken dieser Rolle übernimmt. Die Sprints sind nach dem CRISP-DM-Zyklus orientiert und sind folgendermassen strukturiert:
- Sprint 1 | Business & Data Understanding und Data Preparation
- Sprint 2 | Data Preparation
- Sprint 3 | Modelling
- Sprint 4 | Evaluation
- Sprint 5 | Deployment

[Dokumentation](scrum.md)







## Modeling
## Evaluation
## Demployment
