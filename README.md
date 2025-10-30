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

Im Rahmen des Projektes übernimmt jeweils ein Teammitglied die Rolle als Scrum-Master und leitet die Kommilitonen. Einen Product Owner gibt es in diesem Scenario nicht, wobei unser Dozent gewisse Charakteristiken dieser Rolle übernimmt.

## Business Understanding
### Ziel und Scope
Das Projekt entwickelt ein zweiteiliges System für Basel-Stadt:
- Nachfrageprognose des Stromverbrauchs auf 15-Minuten-Basis.
- Anomalieerkennung im Verbrauchsverhalten.

Datenumfang: 
Daten zum Stromverbrauch: viertelstündlicher Netzbezug im Kanton Basel-Stadt inklusive Netzverluste.
Nicht enthalten: lokal erzeugter und direkt vor Ort verbrauchter PV-Strom.
Kundengruppen im Datensatz: grundversorgte und freie Kunden. Beide beeinflussen die Netzlast.

Wetterdaten: zehnminütige Messung diverser Wetterdaten, wie Niederschlag, Sonneneinstralung und Windstärke

Zielvariable: Gesamtverbrauch (Summe beider Kundengruppen).
Für Netzführung und Lastplanung ist der gesamte Effekt im Netz relevant. Optional können später Submodelle pro Kundengruppe erstellt werden.

### Hintergrund
IWB Industrielle Werke Basel ist Versorger und Netzbetreiber: eigene Erzeugung (Wasserkraft, KWK, Solar) und marktseitiger Zukauf bei Bedarf.

Grundversorgte Kunden: beziehen Energie direkt von IWB.
Freie Kunden: kaufen Energie am Markt, nutzen aber das IWB-Netz. Deren Last ist für die Netzplanung ebenso relevant.

### Business Nutzen
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

## Data Understanding Main

### Datenquellen:
#### Stromverbauch Daten von Basel
- [Data](data/251006_StromverbrauchBasel2012-2025.csv)
- [Link](https://de.wikipedia.org/wiki/Data_Science](https://opendata.swiss/de/dataset/kantonaler-stromverbrauch-netzlast))
#### Meteo Daten Basel/Binningen BAS
- [Data]()
- [Link](https://www.meteoschweiz.admin.ch/service-und-publikationen/applikationen/ext/daten-ohne-programmierkenntnisse-herunterladen.html#lang=de&mdt=normal&pgid=&sid=BAS&col=ch.meteoschweiz.ogd-smn&di=ten-minutes&tr=historical&hdr=2020-2029)

#### Formalisierung

Instanzen: 
- Parameter Messungen im 30 Minuten Intervall
- Parameter Messungen im Tagesintervall


### Features (X)
| Features | Beschreibung | Werteskala |
|--------|----------|--------------|
| **Stromdaten** | | |
|`Start der Messung (Text)` | Ursprünglicher Zeitpunkt der Messung als Textstring. | Text |
|`Jahr` | Jahr der Messung  | 2011-2025 |
|`Monat` | Monat der Messung | 1-12 |
|`Tag` | Tag des Monats der Messung | 1-31 |
|`Wochentag` | Nummer des Wochentags | 0-6 (Montag-Sonntag) |
|`Tag des Jahres` | Fortlaufende Tagesnummer im Jahr | 1–365/366 |
|`Quartal` | Quartal des Jahres | 1-4 |
|`Woche des Jahres` | Kalenderwoche der Messung | 1–52/53 |
|`Grundversorgte Kunden` | Anzahl der Kunden, die automatisch mit Strom versorgt werden. | 0-26'090 |
|`Freie Kunden` | Anzahl der Kunden, die selbst ihren Anbieter wählen können. | 0-32'296 |
|`Stromverbrauch` | Verbrauch in Kilowattstunden (kWh) für den jeweiligen Zeitraum. | 22'322-68'374 |

| **Wetterdaten** |  |  |
| `station_abbr` | Stationskürzel (z. B. Basel = BAS) | Text |
| `reference_timestamp` | Datum und Zeit der Messung | Datum + Uhrzeit |
| `Lufttemperatur` | Lufttemperatur | -50 bis 50 °C |
| `Chilltemperatur` | Empfundene Temperatur unter Wind | -50 bis 50 °C |
| `relative Luftfeuchtigkeit` | Verhältnis aktueller zu maximal möglicher Luftfeuchte | 0–100 % |
| `Taupunkt` | Temperatur, bei der Luftfeuchte kondensiert | -50 bis 50 °C |
| `Dampfdruck` | Partialdruck des Wasserdampfs in der Luft | 0–50 hPa |
| `Luftdruck` | Druck der Luftsäule auf einer Fläche. Hoher Wert bedeutet eher schlechtes Wetter | 900–1100 hPa |
| `Böenspitze` | Maximale Windböe über 1 Sekunde | 0–50 m/s |
| `Windgeschwindigkeit` | Durchschnittliche Windgeschwindigkeit | 0–50 m/s |
| `Windrichtung` | Windrichtung in Grad | 0–360 ° |
| `Niederschlag` | Höhe von Wasseransammlung, wenn nicht versickert | 0–200 mm |
| `Globalstrahlung` | Gesamtstrahlung (direkt + diffus) auf horizontale Fläche | 0–1500 W/m² |
| `Diffustrahlung` | Streustrahlung aus der Atmosphäre ohne direkte Sonne | 0–1000 W/m² |
| `Langwellige Einstrahlung` | Wärmestrahlung vom Himmel (langwellige IR-Strahlung) | 0–500 W/m² |
| `Sonnenscheindauer` | Dauer direkter Sonneneinstrahlung in 10 Minuten | 0–10 min |

### Zielvariable (Y)
Stromverbrauch: 
- nächste 30 Minuten
- beliebiger Tag im Jahr
- beliebiger Monat im Jahr
- bestimmter Zeitraum

### Modell
Regressionsmodelle für Prognose Stromverbrauch:
- Lineare Regression


Klassifikator Anomalieerkennung:
- Logistic Regression
- Random Forest


### Data Understanding – Energieverbrauch Basel (2012–2025)

#### Datensatzübersicht
- **Datei:** `251006_StromverbrauchBasel2012-2025.csv`  
- **Zeilen:** 481’959  
- **Spalten:** 12  
- **Zeitraum:** 01.01.2012 → 29.09.2025  
- **Datenfrequenz:** 15-Minuten-Intervalle  
- **Index:** `Start der Messung` (UTC, DatetimeIndex)

---

#### Datenstruktur



| Typ | Spalten |
|------|----------|
| **Numerisch** | `Stromverbrauch`, `Grundversorgte Kunden`, `Freie Kunden`, `Jahr`, `Monat`, `Tag`, `Wochentag`, `Tag des Jahres`, `Quartal`, `Woche des Jahres` |
| **Text** | `Start der Messung (Text)` |

---

#### Datenqualität
- ✅ **Keine doppelten Zeitstempel**  
- ⚠️ **Fehlende Werte:**
  - `Grundversorgte Kunden`: ca. **62 %** fehlend  
  - `Freie Kunden`: ca. **63 %** fehlend  
- ✅ **Zeitintervalle** sind konsistent (alle 15 Minuten)  
- ✅ **Datumsindex** korrekt gesetzt (`UTC`)  

---

#### Beschreibende Statistik
| Variable | Minimum | Maximum | Mittelwert |
|-----------|----------|----------|-------------|
| **Stromverbrauch (kWh)** | 22 322 | 68 374 | **38 454** |
| **Grundversorgte Kunden** | 0 | 26 090 | **15 788** |
| **Freie Kunden** | 0 | 32 296 | **19 277** |

---

#### Korrelationen (stärkste Zusammenhänge)
- `Stromverbrauch` ↔ `Freie Kunden`: **0.92**  
- `Stromverbrauch` ↔ `Grundversorgte Kunden`: **0.87**  
- `Stromverbrauch` ↔ `Wochentag`: **–0.27**

---

#### 🔍 Zentrale Erkenntnisse
1. Der Datensatz umfasst **über 13 Jahre** Stromverbrauchsdaten für Basel.  
2. **Saisonale und wöchentliche Muster** sind erkennbar (z. B. geringerer Verbrauch am Wochenende).  
3. **Kundensegmente** (freie vs. grundversorgte Kunden) beeinflussen den Verbrauch deutlich.  
4. **Datenqualität insgesamt hoch**, nur vereinzelte Lücken.  
5. Sehr gut geeignet für **Zeitreihenanalyse** und **Machine-Learning-Prognosen**.

---
### Data Understanding – Wetterdaten Basel (2010–2024)

#### Datensatzübersicht
- **Datei:** `Wetterdaten_Basel_2010-2029_merged.csv`  
- **Zeilen:** 788’977  
- **Spalten:** 27  
- **Zeitraum:** 2010 → 2029  
- **Frequenz:** 10-Minuten-Intervalle  
- **Datenquelle:** Wetterstation Basel (BAS)

---

#### Datenstruktur
| Typ | Beispielspalten |
|------|-----------------|
| **Numerisch** | `Lufttemperatur 2 m ü. Boden`, `relative Luftfeuchtigkeit`, `Windgeschwindigkeit`, `Niederschlag`, `Globalstrahlung`, `Diffusstrahlung`, `Langwellige Einstrahlung`, `Sonnenscheindauer` |
| **Text/Objekt** | `station_abbr`, `Date and Time`, `Date`, `Time` |

---

#### Datenqualität
- ✅ **Keine doppelten Einträge**  
- ⚠️ **Fehlende Werte:**  
  - Temperatur- und Druckmessungen: < 0,3 % fehlend  
  - Wind- und Strahlungsdaten: ca. 0,2 – 0,4 % fehlend  
  - Insgesamt sehr **hohe Datenqualität** (> 99 % vollständig)  
- ✅ **Einheitliche Zeitintervalle** (10 Minuten)  
- ✅ **Numerische Typen** korrekt erkannt  

---

#### Beschreibende Statistik (Auszug)
| Variable | Minimum | Maximum | Mittelwert |
|-----------|----------|----------|-------------|
| **Lufttemperatur 2 m ü. Boden (°C)** | –19.3 | 37.4 | **11.53** |
| **relative Luftfeuchtigkeit (%)** | 11.0 | 100.0 | **73.98** |
| **Luftdruck (hPa)** | 940.1 | 1006.3 | **979.8** |
| **Windgeschwindigkeit (km/h)** | 0.0 | 55.1 | **6.78** |
| **Niederschlag (mm/10 min)** | 0.0 | 19.0 | **0.02** |
| **Globalstrahlung (W/m²)** | 0.0 | 1265.0 | **142.32** |
| **Sonnenscheindauer (min/10 min)** | 0.0 | 10.0 | **2.04** |

---

#### Korrelationen (wichtige Zusammenhänge)
- **Temperatur ↔ Taupunkt:** 0.84 → enge physikalische Beziehung  
- **Temperatur ↔ relative Luftfeuchtigkeit:** –0.54 → wärmer = trockener  
- **Windgeschwindigkeit ↔ Böenspitze:** > 0.95 → erwartungsgemäß stark korreliert  
- **Globalstrahlung ↔ Sonnenscheindauer:** 0.80 → hohe Strahlung bei viel Sonne  
- **Temperatur ↔ Globalstrahlung:** 0.50 – 0.78 → deutlicher Tagesgang-Einfluss  

---

#### Zentrale Erkenntnisse
1. **Sehr umfangreicher Datensatz** – rund 789 000 Zeilen über fast 20 Jahre.  
2. **Nahezu vollständige Messreihe**, ideal zur Kombination mit Stromverbrauchsdaten.  
3. **Starke saisonale Effekte:** höhere Globalstrahlung → höhere Temperaturen.  
4. **Hohe interne Konsistenz:** physikalische Abhängigkeiten sind plausibel.  
5. **Datensatz geeignet für Energie-, Klima- und Zeitreihenmodelle.**

---

## Data Preparation

## Modeling
## Evaluation
## Demployment
