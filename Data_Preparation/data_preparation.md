## Data Preparation
### Brainstorming & Erste Gedanken zur Data Preparation

Datums-Features müssen fest auf eine Variante reduziert werden, es gibt mehrere Datumsfeature auch in verschiedenen Datentypen.
Unterschiedliche Skalen wie bei Meteodaten können ML-Ergebnisse verzerren, deshalb später skalieren.​

Feature-Auswahl: Kategorische Daten werden zu numerischen Daten konvertiert. Textuelle Features wie „stationabbr“ oder Messstart werden entfernt.​

Fehlende Werte: Besonders viele NaNs in Spalten wie Grundversorgte Kunden, Freie Kunden, Wetterdaten (bis zu 700 NaNs, z.B. Sonnenscheindauer). Lösung: Erst mit vollständigen Daten starten, dann Varianten (z.B. lineare Interpolation) ausprobieren. In Wetterdaten abgleichen, wie oft Daten fehlen.​

Fehlerhafte Zeitabstände: Es wurden 13 Werte mit einem Abstand von 90 statt 30 Minuten gefunden und sollten per Code korrigiert werden.​

Events und externe Features: Zusätzliche Datenquellen durch Events in Basel, Ferien, Wochenenden, Nacht-Zyklus, saisonale Marker, Schulferien, Feiertagsinfos, besondere Touristenzeiten (Fantasy Basel, ArtBasel, Herbstmesse).​

Neue Features: Arbeitstag vs. Sonntag, saisonale Marker, Lag- und Rolling-Features (Verbrauch 15min, 1h, 1d, 1w vorher, Mittelwerte letzter 3h/12h/24h).​

Datenaufteilung: Trainings- und Testdaten sollen in Jahresintervallen ab 2020-2025 aufgeteilt werden. Fehlende Werte vor 2020 werden mit NaN markiert und später bewertet.​

Beachtung Strompreis/Korrelation: Untersucht werden soll die Korrelation zwischen Sonnenstrahlung und Stromverbrauch (Stichwort Solaranlagen).​

Feature-Selektion: Am Ende fixierte Liste an Features: Stromverbrauch, Tag/Monat/Jahr/Zeit, Freie Kunden, Grundversorgte Kunden, Wetterdaten (Temperatur, Druck, Windgeschwindigkeit, Chilltemperatur, Niederschlag, Sonnenscheindauer, Globalstrahlung).​

## Zusammenfassung: Zeitstempel, Intervallwahl und Feature-Entscheidungen

## 1. Zeitumstellung und fehlende Werte

Die Stromverbrauchsdaten werden in lokaler Zeit (Europe/Zurich) gemessen.  
Das führt jedes Jahr zu diesen Effekten:

### März – Zeit wird vorgestellt  
- Die Stunde 02:00–03:00 existiert nicht.  
- → 4 fehlende 15-Minuten-Messwerte.

### Oktober – Zeit wird zurückgestellt  
- Die Stunde 02:00–03:00 würde doppelt vorkommen.  
- Da das Messsystem keine doppelten lokalen Zeitstempel speichern darf,  
  wird **eine der beiden Stunden verworfen**.  
- → ebenfalls 4 fehlende Messwerte.

### Ergebnis  
Über alle Jahre fehlen genau **52 Zeitpunkte**.  
Diese Werte sind real **nie gemessen worden** und können ohne Informationsverlust entfernt werden.

---

## 2. Unterschied lokale Zeit vs. UTC

Die Originaldaten enthalten:
- `Start der Messung (Text)` → lokale Zeit (Anzeige)
- Zeitindex im DataFrame → UTC (eindeutig)

Der UTC-Index enthält die sichtbaren Zeitsprünge (März +1h, Oktober –1h),  
weil hier die fehlenden oder verworfenen Messpunkte offensichtlich werden.

Die Textspalte spielt für das Training **keine Rolle**.

---

## 3. Intervallwahl: 15 Minuten vs. 30 Minuten

### Warum 30 Minuten zuerst überlegt wurde
- Wetter: 10-Minuten-Daten → durch Sampling → 30 Minuten ohne Interpolation möglich  
  (jeder 3. Wert)
- Strom: 15-Minuten-Daten → durch Sampling → 30 Minuten möglich (jeder 2. Wert)
- Weniger Daten = stabilere Modelle

### Warum die Entscheidung am Ende **15 Minuten** war
- IWB misst und verrechnet Strom **immer in 15-Minuten-Blöcken**.
- Stromhandel in Europa läuft ebenfalls in 15-Minuten-Blöcken.
- Anomalien sind oft kürzer als 30 Minuten → 30 Minuten glätten kritische Peaks.
- Wetter lässt sich problemlos von 10 → 15 Minuten interpolieren.
- Fachlich realistischer und aussagekräftiger.

→ **15 Minuten ist der korrekte und professionelle Standard.**

---

## 4. Wetterdaten im Modell: Problem und Lösung

Ein Prognosemodell darf keine Wetterwerte verwenden,  
die **in der Zukunft liegen** (Data Leakage).

Beispiel:  
Du kannst den Stromverbrauch um 12:00 nicht mit der Temperatur um 12:00 vorhersagen,  
weil du die Temperatur um 12:00 zu diesem Zeitpunkt nicht kennst.

### Lösung  
Alle Wettervariablen werden als **Lag-Features** verwendet:

- `Temperatur(t-15min)`
- `Luftfeuchtigkeit(t-30min)`
- `Wind(t-1h)`
- usw.

Damit nutzt das Modell nur Informationen, die **real verfügbar waren**.

---

---

## 6. Datenquelle Wetter – Herkunft und regionale Relevanz

Die verwendeten Wetterdaten stammen aus externen Messstationen im Raum Basel.  
Die **exakte Station und deren geografische Abdeckung** sind aktuell noch nicht klar dokumentiert.

In der Diskussion mit Renold konnte dazu keine eindeutige Antwort gegeben werden.  
Diese Unsicherheit muss noch geklärt werden, da die räumliche Nähe der Wetterstation  
direkt die Aussagekraft für den Stromverbrauch im Versorgungsgebiet beeinflusst.

Offen:
- Welche Wetterstation wurde verwendet?
- Wie weit ist sie vom Fokusgebiet Basel entfernt?
- Wie repräsentativ sind die Messdaten für das tatsächliche Stromnetzgebiet?

---

## 6. Feature-Reduktion – Wetterdaten

Zur Vermeidung von Redundanz und unnötiger Komplexität wird das Wetter-Feature-Set reduziert.

### Behaltene Features:
- Böenspitze **(3-Sekundenböe) in m/s**
- Lufttemperatur 2 m über Boden
- Luftdruck auf Barometerhöhe
- Niederschlag
- Taupunkt
- Windgeschwindigkeit (skalar) in m/s
- relative Luftfeuchtigkeit
- Globalstrahlung
- Diffusstrahlung
- Langwellige Einstrahlung
- Chilltemperatur

### Entfernte Features:
- Alle anderen Böenspitzenmessungen  
  (Sekundenböe, km/h Varianten → redundant)
- `Station`
- `Time`, `Date`, `Date and Time` (aus Wetterquelle)
- Luftdruck auf Meeresniveau:
  - QFF
  - QNH
- Lufttemperatur Bodenoberfläche
- Windgeschwindigkeit in km/h

Ziel:  
Reduzierte, informationsreiche Features ohne doppelte Messungen.

---

## 7. Feature-Reduktion – Stromverbrauchsdaten

Aus der Stromverbrauchsquelle werden folgende zeitlichen Features entfernt:

- `Tag`
- `Quartal`
- `Jahreszeiten`

Begründung:
Diese Variablen sind abgeleitet und liefern eine geringere Auflösung.  
Sie sind redundant gegenüber:

- `Monat`
- `Tag des Jahres`
- `Wochentag`

Ein Januartag ist nicht gleich wie ein Dezembertag, obwohl beide im Winter liegen.

---

## 8. Zeitproblem zwischen UTC und lokaler Zeit

Aktuell besteht eine Inkonsistenz:

- `DateTime`-Index ist in **UTC**
- Abgeleitete Zeitfeatures (`Jahr`, `Monat`, `Tag`, `Tag des Jahres`)  
  wurden jedoch aus **lokaler Zeit (Europe/Zurich)** generiert

Dadurch entsteht eine systematische Verschiebung  
zwischen Zeitstempel und den zeitlichen Features.

---

## 9. Offene Entscheidungsoptionen

### Option A: Lokale Zeit als Standard
- `DateTime` wird von UTC → Europe/Zurich konvertiert
- Alle zeitlichen Features werden neu aus lokaler Zeit berechnet
- Feature-Lags werden ebenfalls auf lokaler Zeit aufgebaut

### Option B: UTC beibehalten
- `DateTime` bleibt UTC
- Alle zeitlichen Features werden neu aus UTC berechnet
- Alle Lag-Features werden auf UTC-Zeit bezogen

Beide Wege sind technisch möglich,  
aber es muss **eine konsistente Zeitbasis** festgelegt werden.

---

---

## Fazit

Die Aufbereitung der Strom- und Wetterdaten folgt einer klaren, fachlich begründeten Logik:

- Die **Zeitproblematik durch Sommer-/Winterzeit** wurde korrekt identifiziert und dokumentiert.  
  Die fehlenden 52 Zeitpunkte entstehen physikalisch durch die Zeitumstellung und sind keine Datenfehler.

- Die Entscheidung für ein **15-Minuten-Intervall** ist fachlich konsistent mit der Messlogik der IWB und dem Stromhandel.  
  Eine gröbere Auflösung würde relevante Muster und Anomalien verwässern.

- Bei den **Wetterdaten** wurden redundante, doppelte und wenig relevante Features systematisch entfernt.  
  Übrig bleiben nur physikalisch sinnvolle und nicht redundante Messgrössen.

- Bei den **Zeitfeatures des Stromverbrauchs** wurde unnötige Redundanz (Quartal, Jahreszeit, Tag) beseitigt,  
  ohne dabei saisonale oder wöchentliche Informationen zu verlieren.

- Das zentrale offene Thema bleibt die **Zeitbasis (UTC vs. lokale Zeit)**:
  Eine einheitliche Entscheidung ist zwingend notwendig, um Verschiebungen und Inkonsistenzen in den Features zu vermeiden.

Insgesamt wurde eine strukturierte, reduzierte und fachlich saubere Datenbasis geschaffen,  
die als Grundlage für ein valides, konsistentes und realitätsnahes Stromverbrauchs-Forecasting dienen kann.



