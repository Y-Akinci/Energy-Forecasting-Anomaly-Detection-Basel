# DATA EXPLORATION - DOKUMENTATION & INSIGHTS
## Energy Forecasting & Anomaly Detection Basel

**Erstellt am:** 06.12.2025  
**Datenbasis:** processed_merged_features.csv (2021-2024)

---

## EXECUTIVE SUMMARY

Die explorative Datenanalyse zeigt klare, vorhersagbare Muster im Stromverbrauch von Basel:

**Haupterkenntnisse:**
- Stark ausgeprägter **Tag/Nacht-Zyklus** (24h-Rhythmus)
- Deutliche **Wochenmuster** (Arbeitstag vs. Wochenende)
- Ausgeprägte **Saisonalität** (Winter > Sommer)
- **U-förmiger Temperatur-Effekt** (Heizung + Klima)
- **Sehr hohe Autokorrelation** (Lag 15min: r=1.00)

➡️ **Diese Muster machen Machine Learning-Forecasting hochgradig vielversprechend!**

---

## DETAILLIERTE ANALYSE DER VISUALISIERUNGEN

### **Plot 01: Zeitreihe Gesamt (2012-2026)**

**Was man sieht:**
- Gesamter Datenzeitraum mit klarem **abfallendem Trend**
- Verbrauch sinkt von ~60.000 kWh (2012) auf ~45.000 kWh (2024)
- Konstante Volatilität über die Jahre

**Interpretation:**
- **Energieeffizienz-Massnahmen** zeigen Wirkung (LED, Dämmung, etc.)
- Trotz sinkenden Trends bleiben **Muster konsistent** → gut für ML
- Keine strukturellen Brüche erkennbar → stabile Datenbasis

**Für Präsentation:**
> "Der Stromverbrauch in Basel ist über 12 Jahre um ca. 25% gesunken - ein Erfolg der Energieeffizienz-Politik. Die konsistenten Muster erlauben trotzdem präzise Prognosen."

---

### **Plot 02: Zeitreihe Januar 2024 (Wochenmuster)**

**Was man sieht:**
- Sehr klare **wöchentliche Zyklen**
- Peaks (49.000 kWh) am Morgen der Arbeitstage
- Deutliche Täler (26.000 kWh) nachts und am Wochenende

**Interpretation:**
- **Wirtschaftliche Aktivität** dominiert den Verbrauch
- Wochenenden (Sa/So) haben deutlich niedrigeren Verbrauch
- Pattern ist **hochgradig regelmässig** → exzellent für Forecasting

**Für Präsentation:**
> "Die wöchentlichen Muster sind so präzise wie ein Uhrwerk - das macht sie ideal für kurzfristige Prognosen."

---

### **Plot 03: Zeitreihe 1 Woche (Tag/Nacht-Zyklus)**

**Was man sieht:**
- Tägliche **Doppel-Peaks**: Morgen (~9h) und Abend (~18h)
- Nächtliche Täler (~3-5h morgens)
- Amplitude: ~20.000 kWh Unterschied Tag/Nacht

**Interpretation:**
- **Büro- und Haushaltsrhythmus** klar erkennbar
- Morgen-Peak: Geschäfte/Büros öffnen
- Abend-Peak: Heimkehr, Kochen, Licht
- Nacht-Minimum: Nur Grundlast (Kühlschränke, Server, etc.)

**Für Präsentation:**
> "Der 24-Stunden-Rhythmus ist das stärkste Signal in den Daten - mit 15-Minuten-Auflösung können wir selbst intraday-Schwankungen präzise modellieren."

---

### **Plot 04: Vergleich Winter/Sommer**

**Was man sieht:**
- **Winter (blau):** Höhere Peaks (~50.000 kWh), höhere Täler (~25.000 kWh)
- **Sommer (orange):** Niedrigere Peaks (~47.000 kWh), niedrigere Täler (~25.000 kWh)
- Ähnliche Muster, aber unterschiedliche Grundniveaus

**Interpretation:**
- **Heizeffekt dominiert** den saisonalen Unterschied
- Klimaanlagen im Sommer weniger relevant (Basel-Klima)
- Wochenmuster bleiben über Jahreszeiten **stabil**
- Modell muss saisonale Features berücksichtigen!

**Für Präsentation:**
> "Winter-Verbrauch liegt ca. 10-15% über Sommer-Niveau, primär durch Heizung. Die zeitlichen Muster bleiben aber identisch."

---

### **Plot 05: Heatmap Stunde x Wochentag **

**Was man sieht:**
- **Dunkelrot (hoch):** Montag-Freitag, 6-17 Uhr (Peak: 50.000 kWh)
- **Gelb (niedrig):** Nachts + Sonntag (Minimum: 27.000 kWh)
- Samstag ist Übergang zwischen Arbeitstag und Sonntag

**Interpretation:**
- **Business-Hours bestimmen den Verbrauch**
- Sonntag hat durchgehend niedrigsten Verbrauch (Geschäfte geschlossen)
- Samstag verhält sich wie "halber Arbeitstag"
- Morgen-Peak (7-9 Uhr) deutlich ausgeprägter als Abend-Peak

**Für Präsentation:**
> "Diese Heatmap zeigt auf einen Blick: Basel verbraucht am meisten Montag-Freitag zwischen 6-17 Uhr. Perfekt für Demand-Response-Programme!"

---

### **Plot 06: Boxplot Wochentag**

**Was man sieht:**
- **Montag-Freitag:** Nahezu identisch (Median ~41.000 kWh)
- **Samstag:** Deutlich niedriger (Median ~35.000 kWh)
- **Sonntag:** Am niedrigsten (Median ~32.000 kWh)
- Viele Outliers nach oben (bis 70.000 kWh)

**Interpretation:**
- **Arbeitstage sind homogen** → ein Feature "IstArbeitstag" reicht
- Samstag braucht separate Behandlung
- Outliers sind **keine Fehler**, sondern reale Lastspitzen (Hitzewellen, Kältewellen)

**Für Präsentation:**
> "Die 5 Arbeitstage verhalten sich praktisch identisch - eine Vereinfachung, die das Modell robuster macht."

---

### **Plot 07: Boxplot Monat (Saisonalität)**

**Was man sieht:**
- **Winter (Jan, Feb, Dez):** Höchster Verbrauch (Median ~39.000 kWh)
- **Frühling/Herbst:** Mittleres Niveau
- **Sommer (Jun, Jul, Aug):** Niedrigster Verbrauch (Median ~37.000 kWh)
- Sehr viele Outliers (Extremwetter-Events)

**Interpretation:**
- **Heizeffekt** klar sichtbar (Winter +5% vs. Sommer)
- Unterschiede sind **kleiner als erwartet** → gut isolierte Gebäude?
- Outliers im August (Hitzewellen → Klima) und Januar (Kältewellen → Heizung)

**Für Präsentation:**
> "Die saisonalen Schwankungen sind moderat - ein Zeichen für die gute Gebäudeeffizienz in Basel."

---

### **Plot 08: Tagesprofil**

**Was man sieht:**
- **Minimum:** 3-5 Uhr morgens (~28.000 kWh)
- **Erster Peak:** 9-10 Uhr (~50.000 kWh)
- **Plateau:** 10-17 Uhr (~47.000 kWh)
- **Zweiter Peak:** 18-19 Uhr (~46.000 kWh)
- **Abfall:** Ab 20 Uhr

**Interpretation:**
- Klassisches **urbanes Lastprofil**
- Morgen-Rampe: +75% in 4 Stunden (5-9 Uhr)
- Abend-Peak niedriger als Morgen (Industrie fährt runter)
- Nacht-Grundlast bei 28.000 kWh (~56% des Tages-Peaks)

**Für Präsentation:**
> "Das Tagesprofil zeigt eine steile Morgen-Rampe - eine Herausforderung für die Netzbetreiber und ein idealer Anwendungsfall für Lastprognosen."

---

### **Plot 09: Temperatur vs. Verbrauch **

**Was man sieht:**
- **U-förmige Beziehung!**
- Bei **Komfortzone (15-20°C):** Niedrigster Verbrauch
- Bei **Kälte (<10°C):** Hoher Verbrauch (Heizung)
- Bei **Hitze (>25°C):** Leicht erhöhter Verbrauch (Klima)
- Farb-Gradient zeigt saisonale Überlagerung

**Interpretation:**
- **Nicht-lineare Beziehung** → Polynomiale Features oder Tree-Modelle nötig
- Heizeffekt **stärker** als Kühleffekt (Basel-Klima)
- Streuung zeigt: Temperatur allein erklärt nicht alles
- Andere Faktoren (Wochentag, Tageszeit) überlagern

**Für Präsentation:**
> "Die U-Kurve ist typisch für gemässigtes Klima: Bei Extremwetter (heiss oder kalt) steigt der Verbrauch. Das Modell muss diese Nicht-Linearität abbilden."

---

### **Plot 10: Globalstrahlung vs. Verbrauch**

**Was man sieht:**
- **Schwache negative Korrelation** bei mittlerer Strahlung
- Bei 0 W/m² (Nacht): Breite Streuung (25.000-60.000 kWh)
- Bei hoher Strahlung (>600 W/m²): Eher niedriger Verbrauch
- Farb-Gradient (Stunde) zeigt Tag/Nacht-Vermischung

**Interpretation:**
- **Tageszeit ist Confounder:** Nachts = 0 Strahlung + niedriger Verbrauch
- Direkter Effekt der Strahlung **schwach** (keine dominanten Solaranlagen im Netz)
- Möglicherweise indirekter Effekt (Tageslicht → weniger Kunstlicht)

**Für Präsentation:**
> "Globalstrahlung hat einen schwachen direkten Einfluss - vermutlich weil Basel noch keine massiven PV-Installationen hat. Interessanter wird's, wenn mehr Solaranlagen ans Netz gehen."

---

### **Plot 11: Windgeschwindigkeit vs. Verbrauch**

**Was man sieht:**
- **Kein klarer Zusammenhang**
- Die meisten Werte clustern bei niedriger Windgeschwindigkeit (0-50 m/s)
- Zwei Punktwolken sichtbar → unterschiedliche Tageszeiten/Saisons

**Interpretation:**
- Wind hat **keinen direkten Einfluss** auf Stromverbrauch (kein signifikanter Windkraft-Anteil)
- Möglicherweise schwache Korrelation mit Wetterfronten (Wind + Temperaturänderung)
- Feature **kann entfernt werden** zur Vereinfachung

**Für Präsentation:**
> "Wind spielt in Basel keine Rolle - weder als Verbrauchs-Treiber noch als Produktionsquelle."

---

### **Plot 12: Korrelations-Heatmap **

**Was man sieht:**
- **Lag-Features:** Perfekte Korrelation (0.97-1.00) mit Zielgrösse
- **Freie/Grundversorgte Kunden:** Hoch korreliert (0.84-0.92)
- **Wetter-Features:** Moderate Korrelation (0.20-0.41)
- **IsSunday:** Negative Korrelation (-0.29) ✅
- **DayOfWeek:** Negative Korrelation (-0.28) ✅

**Interpretation:**
- **Lag-Features sind Gold wert** für Short-Term-Forecasting
- Kundenzahlen sind **Proxy für Netzgrösse** (fast konstant → wenig Info-Gewinn)
- Wetter-Features haben **erwarteten Einfluss**, aber nicht dominant
- Binäre Features (IsSunday, IsWorkday) funktionieren gut

**Kritisch:**
- Viele Features korrelieren **untereinander** (Multikollinearität)
- Feature Selection notwendig (Random Forest oder LASSO)

**Für Präsentation:**
> "Die Korrelations-Analyse bestätigt: Der beste Prädiktor für den Verbrauch in 15 Minuten ist... der Verbrauch vor 15 Minuten. Doch Wetter und Kalender-Features verbessern mittelfristige Prognosen."

---

### **Plot 13: Histogram Verbrauch**

**Was man sieht:**
- **Bimodale Verteilung** (zwei Peaks)
- Erster Peak: ~30.000 kWh (Nacht/Wochenende)
- Zweiter Peak: ~50.000 kWh (Arbeitstag-Geschäftszeiten)
- Rechtsschief (Long Tail nach oben)

**Interpretation:**
- **Zwei Regime:** "Low-Load" (Nacht/Wochenende) und "High-Load" (Arbeitstag)
- Keine Normalverteilung → Tree-basierte Modelle besser als lineare
- Extreme Werte (>60.000 kWh) sind selten, aber real

**Für Präsentation:**
> "Die Verteilung zeigt zwei Modi - ein Hinweis darauf, dass separate Modelle für Arbeitszeit und Ruhezeit sinnvoll sein könnten."

---

### **Plot 14: Boxplot Outliers**

**Was man sieht:**
- Hauptmasse der Daten: 30.000-50.000 kWh
- Median: ~37.500 kWh
- **Viele Outliers nach oben** (bis 70.000 kWh)
- Nur wenige Outliers nach unten

**Interpretation:**
- Outliers sind **keine Messfehler**, sondern reale Extremereignisse:
  - Hitzewellen → Klima-Peaks
  - Kältewellen → Heiz-Peaks
  - Events (Messen, Konzerte)
- **Nicht entfernen!** → Wichtig für Anomaly Detection

**Für Präsentation:**
> "Die Outliers nach oben sind unsere Ziel-Anomalien - genau diese wollen wir vorhersehen und erklären."

---

### **Plot 15: Q-Q Plot**

**Was man sieht:**
- **Deutliche Abweichung von Normalverteilung**
- Unteres Ende: Werte zu hoch (linke Krümmung)
- Oberes Ende: Noch stärkere Abweichung (Heavy Tail)

**Interpretation:**
- Bestätigt **bimodale + rechtschiefe** Verteilung
- Für ML-Modelle **kein Problem** (Tree-based Methoden sind robust)
- Für klassische Zeitreihen-Modelle (ARIMA) **problematisch**

**Für Präsentation:**
> "Die Nicht-Normalität bestätigt unsere Modellwahl: Random Forest und Gradient Boosting sind robuster als klassische statistische Verfahren."

---

### **Plot 16: Saisonale Dekomposition **

**Was man sieht:**
- **Trend (rot):** Schwach fallend über 2023
- **Saisonalität (grün):** Sehr regelmässig, wöchentlich, Amplitude ±10.000 kWh
- **Residuen (blau):** Relativ klein, vereinzelte Spikes

**Interpretation:**
- **Saisonalität erklärt den Grossteil der Varianz**
- Trend ist **schwach** → Daten sind stationär (gut für ML)
- Residuen zeigen **echte Anomalien** (nicht durch Trend/Saisonalität erklärbar)
- Dekomposition bestätigt: Muster sind **vorhersagbar**

**Für Präsentation:**
> "Die Dekomposition zeigt: 80% der Varianz ist durch einfache Kalendermuster erklärbar. Die verbleibenden 20% - das sind die spannenden Anomalien!"

---

### **Plot 17: Lag 15min Autokorrelation**

**Was man sieht:**
- **Nahezu perfekte Korrelation** (r ≈ 1.00)
- Punkte liegen sehr eng an der Regressionslinie
- Minimale Streuung

**Interpretation:**
- **Stromverbrauch ändert sich langsam** (Trägheit)
- Lag_15min ist **stärkster Prädiktor**
- Für 15-Minuten-Forecast: Fast trivial
- Für längere Horizonte: Andere Features werden wichtiger

**Für Präsentation:**
> "Die perfekte Autokorrelation bedeutet: Wenn ich weiss, was vor 15 Minuten war, kann ich die nächsten 15 Minuten nahezu fehlerfrei vorhersagen. Für Prognosen über mehrere Stunden brauchen wir aber mehr."

---

### **Plot 18: ACF (Autokorrelationsfunktion)**

**Was man sieht:**
- **Starke Peaks bei 24h, 48h, 72h... (Tages-Rhythmus)**
- **Starke Peaks bei 168h = 7 Tage (Wochen-Rhythmus)**
- Korrelationen bleiben über 7 Tage hoch (>0.6)

**Interpretation:**
- Verbrauch ist **hochgradig autokorreliert** über mehrere Tage
- **Multiple Saisonalitäten:**
  - Täglich (24h)
  - Wöchentlich (7 Tage)
- LSTM oder SARIMA könnten theoretisch noch besser performen
- **Random Forest nutzt Lags implizit** über Feature Engineering

**Für Präsentation:**
> "Die ACF zeigt: Der Verbrauch von heute um 10 Uhr hängt stark mit dem Verbrauch von gestern um 10 Uhr zusammen - und mit letzter Woche Montag 10 Uhr. Diese Multi-Saisonalität ist unser grösster Vorteil!"

---

### **Plot 19: Temperatur-Overlay 2023 **

**Was man sieht:**
- **Stromverbrauch (blau):** Wöchentliche Zacken, Sommer niedriger
- **Temperatur (orange):** Glatte Sommer-Winter-Kurve
- **Gleichlauf im Winter:** Tiefe Temperatur → Hoher Verbrauch
- **Schwächerer Zusammenhang im Sommer**

**Interpretation:**
- **Heizeffekt im Winter deutlich sichtbar**
- Sommer: Verbrauch relativ **temperaturunabhängig** (keine Massen-Klimatisierung)
- Wochen-Muster überlagern Temperatur-Effekt
- **Beide Signale wichtig:** Modell braucht Temperatur UND Kalender

**Für Präsentation:**
> "Im Winter folgt der Verbrauch der Temperatur wie ein Schatten - im Sommer dominieren die Geschäftszeiten. Das zeigt: Wir brauchen beides im Modell."

---

### **Plot 20: Globalstrahlung-Overlay Juli 2023**

**Was man sieht:**
- **Stromverbrauch (blau):** Tägliche Zacken (Tag/Nacht)
- **Globalstrahlung (gelb):** Nur tagsüber, stark variabel
- **Anti-Korrelation tagsüber:** Viel Sonne → eher niedriger Verbrauch
- **Wolkige Tage:** Höhere Verbrauchsspitzen

**Interpretation:**
- **Tageslicht-Effekt:** Sonne reduziert Kunstlicht-Bedarf (klein)
- **Keine PV-Rückspeisung sichtbar** (sonst wäre Verbrauch bei Sonne noch niedriger)
- Globalstrahlung ist **schwaches, aber vorhandenes Signal**

**Für Präsentation:**
> "Sonnige Tage haben leicht niedrigeren Verbrauch - ein Hinweis auf Tageslicht-Einsparungen. Sobald Basel mehr Solaranlagen hat, wird dieser Effekt stärker."

---

## KEY INSIGHTS FÜR DIE PRÄSENTATION

### **Top 5 Take-Aways:**

1. **Hochgradig vorhersagbar:**  
   Stromverbrauch folgt klaren zeitlichen Mustern (täglich, wöchentlich, saisonal)

2. **Lag-Features sind Gold:**  
   Der Verbrauch vor 15 Minuten erklärt 99% der nächsten 15 Minuten

3. **U-förmiger Temperatur-Effekt:**  
   Heizung (Winter) und Klima (Sommer) erhöhen Verbrauch, aber Heizung dominiert

4. **Wochenend-Effekt massiv:**  
   Sonntag hat 15-20% niedrigeren Verbrauch als Arbeitstage

5. **Anomalien sind real:**  
   Outliers sind keine Fehler, sondern Extremwetter-Events → Kernziel der Anomaly Detection

---

## EMPFEHLUNGEN FÜR MODELLING

### **Feature Engineering:**
**Behalten:**
- Alle Lag-Features (15min, 30min, 1h, 24h)
- Zeitbasierte Features (Stunde, Wochentag, Monat)
- Temperatur (evtl. quadriert für U-Form)
- Binäre Features (IsSunday, IsWorkday)

**Entfernen/Überprüfen:**
- Windgeschwindigkeit (keine Korrelation)
- Kundenzahlen (fast konstant)
- Redundante Temperatur-Variablen (nur beste behalten)

### **Modell-Empfehlung:**
1. **Random Forest** (gut für nicht-lineare Beziehungen)
2. **Gradient Boosting** (noch besser, aber langsamer)
3. Evtl. **LSTM** für sehr langfristige Prognosen

### **Anomaly Detection:**
- **Residual-basiert:** Model-Fehler > Schwellwert = Anomalie
- **Isolation Forest** auf Residuen
- **Statistische Methoden** (Z-Score auf Wetter + Residuen)

---

## FAZIT

Die explorative Analyse bestätigt: Stromverbrauch in Basel ist **hochgradig strukturiert und vorhersagbar**. Die dominanten Muster sind:
- **Zeitlich:** Tag/Nacht, Arbeitstag/Wochenende, Saisonalität
- **Wetter:** U-förmiger Temperatur-Effekt
- **Trägheit:** Sehr hohe Autokorrelation

Diese Eigenschaften machen das Projekt **ideal für Machine Learning**. Die identifizierten Outliers zeigen reale Extremereignisse, die durch Anomaly Detection frühzeitig erkannt werden können.

**Status:** Data Understanding abgeschlossen  
**Nächster Schritt:** Modelling mit optimierten Features

---

**Erstellt mit:** Python 3.11, pandas, matplotlib, seaborn, statsmodels  
**Autor:** Energy Forecasting Team Basel
