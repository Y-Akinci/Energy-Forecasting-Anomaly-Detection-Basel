## Business Understanding
### Ziel und Scope
Das Projekt entwickelt ein System für Basel-Stadt für die Nachfrageprognose in folgenden detailgraden:
- 15 Minuten Verbrauchsprognose
- Tagesprognosen
- Wochenprognosen

Datenumfang: 
Daten zum Stromverbrauch: viertelstündlicher Netzbezug im Kanton Basel-Stadt inklusive Netzverluste.
Nicht enthalten: lokal erzeugter und direkt vor Ort verbrauchter PV-Strom.
Kundengruppen im Datensatz: grundversorgte und freie Kunden. Beide beeinflussen die Netzlast.

Wetterdaten: zehnminütige Messung diverser Wetterdaten, wie Niederschlag, Sonneneinstralung und Windstärke

Zielvariable: Gesamtverbrauch (Summe beider Kundengruppen).
Für Netzführung und Lastplanung ist der gesamte Effekt im Netz relevant. Optional können später Submodelle pro Kundengruppe erstellt werden.

### Hintergrund
IWB Industrielle Werke Basel ist Versorger und Netzbetreiber: eigene Erzeugung (Wasserkraft, KWK, Solar) und marktseitiger Zukauf bei Bedarf.

Die Stromversorgung durch IWB funktioniert so, dass Haushalte und Unternehmen in Basel ihren Strom überwiegend aus erneuerbaren Quellen wie Schweizer Wasserkraft erhalten; dazu betreibt IWB eigene Kraftwerke und Beteiligungen in der Schweiz und Europa. Neben der eigenen Produktion kauft IWB Strom auch von anderen Anbietern am Markt hinzu, um Schwankungen auszugleichen und den Bedarf sicher zu decken – doch IWB ist in Basel der einzige Netzbetreiber und damit der zentrale Verteiler, über den jeglicher aus dem öffentlichen Netz bezogene Strom läuft.​

Stromproduktion und Bezug
IWB produziert Strom vor allem aus Wasserkraftwerken in den Alpen und mit eigenen Anlagen in Basel, ergänzt durch Windparks und Solarkraftwerke in Europa.
Der Strommix besteht zu einem sehr hohen Anteil aus Wasserkraft sowie in zunehmendem Maße Photovoltaik und Windenergie.
Zur Deckung von Spitzenbedarf oder bei Produktionsschwankungen wird Strom am Markt von anderen Produzenten dazugekauft.​

Die Rolle als einziger Verteiler
IWB besitzt das alleinige Verteilnetz im Stadtgebiet und ist damit der einzige Anbieter, der Strom physisch zu den Verbrauchern bringt.
Alle Bezugsdaten (z.B. alle 15 Minuten) zeigen den tatsächlichen Energiefluss durch das Netz; unabhängig vom Produzenten läuft alles über die IWB-Infrastruktur.
Falls Eigenverbrauch durch lokale Solaranlagen entsteht, sinkt der Bezug aus dem öffentlichen Netz entsprechend und wird in der Statistik direkt abgebildet.
Fehlende Messwerte ergänzt IWB mit Netzbilanzdaten, sodass eine vollständige Abrechnung und Dokumentation aller Energieflüsse garantiert bleibt.​

Beteiligungen und Marktintegration
Während die Herkunft des Stroms vielseitig ist (eigene Produktion, Beteiligungen, Marktkauf), bleibt die Verteil- und Abrechnungshoheit im Raum Basel exklusiv bei IWB.
Die Kunden können verschiedene Stromprodukte wählen, beispielsweise noch höheren Anteil an Solarenergie, aber die Versorgung und Netzführung ist immer an die IWB gebunden.​

So ist IWB zugleich Produzent, Einkäufer und als Betreiber des einzigen Verteilnetzes das zentrale Bindeglied für die Stromversorgung von Basel.
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

- Nutzen: Frühwarnung für Prüfungen, Qualitätssicherung, Messfehler-Erkennung und Auffälligkeiten im Verbrauchsverhalten. Anhand erkannter Anomalien wichtige Informationen ableiten und neue Features erstellen (Feature Creation)

### Technischer Zielzustand
Für das Projekt wird ein reproduzierbares Feature-Set entwickelt, das zeitliche Merkmale (z. B. Stunde, Wochentag, Monat, saisonale Muster) sowie abgeleitete Werte wie Lags, gleitende Durchschnitte und optional Feiertagsinformationen umfasst. Auf dieser Basis wird ein Regressionsmodell trainiert, um den Stromverbrauch präzise vorherzusagen und die Modellgüte anhand transparenter Metriken zu bewerten. Zusätzlich wird ein Anomalie-Flag pro Zeitintervall erzeugt, das auf Prognoseabweichungen oder unüberwachten Scores basiert. Die täglichen Prognosen und Erkennungen werden automatisch in einer CSV-Datei oder als Diagramm exportiert.

