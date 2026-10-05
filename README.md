# AURORA – wie viel ist Vorwissen über die Verhaltensbeschreibung wert? (Streamlit-Demo)

---

Interaktive Demo zu **AURORA**, der gelernten Verhaltensbeschreibung. **Achtes Stück der Konzepte-Linie „Novelty Search und Quality-Diversity“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Derselbe Hof-Roboter, dieselbe Qualität (Direktheit der Fahrt), dieselbe Auswahl der Eltern und dasselbe Archiv aus 256 CVT-Zellen wie in [`cvt-map-elites-demo`](https://github.com/sebastian-hanisch/cvt-map-elites-demo) (Stück 7). Bisher war die **Beschreibung des Verhaltens von Hand** gewählt (Endposition,
Positionen unterwegs): das ist **Vorwissen** darüber, worauf es ankommt. **AURORA** (Grillotti und Cully) braucht es nicht: ein **Autoencoder** verdichtet die **Rohdaten** einer Fahrt zu einem kurzen **Code**, der Code ist die Beschreibung, und während der Suche wird er **immer wieder neu gelernt**.
Das Stück misst, **wie viel das Vorwissen wert ist**: ob AURORA ein richtig gewähltes Vorwissen einholt und ein bewusst falsch gewähltes schlägt.

## Kernfrage

Holt eine aus den Rohdaten gelernte Verhaltensbeschreibung die handgemachte ein, und lohnt es sich, sie während der Suche neu zu lernen?

## Modell und Methodik

- **Labyrinth, Roboter, Qualität** (`aur_maze.py`, `aur_robot.py`, `aur_search.py`): wie Stück 1 bis 7; Steuerfolge von 140 Schritten, Stufe 0 (freies Feld) und Stufe 3 (Falle). **Qualität = Direktheit:** Luftlinie von Start zu Ende geteilt durch die gefahrene Weglänge (0 bis 1). Je Zelle bleibt die beste Fahrt
  (ein Kind ersetzt den Elite nur, wenn es um mehr als 10⁻¹² besser ist). **Auswahl wie in Stück 5 und 7:** 100 Kinder je Generation aus gleichverteilt gezogenen Elites, 15 % der Gene mit Gauß 0.25 verrauscht. **Immer 256 Zellen**, damit der QD-Score bei gleicher Zellzahl vergleichbar bleibt.
- **Handgemachte Beschreibungen** (`aur_space.py`): *Endposition* (2 Werte), *4 Wegpunkte* (die Position nach 1/4, 2/4, 3/4 und 4/4 der Fahrt, 8 Werte) und *Fahrstil* (Weglänge geteilt durch die größtmögliche Weglänge und das mittlere Drehgen, 2 Werte): der **Fahrstil sagt nichts über den Ort** und ist das bewusst falsche Vorwissen.
- **Gelernte Beschreibungen** (`aur_learn.py`): die **Rohdaten** sind die Position nach jedem 20. Teil der Fahrt (20 Positionen, 40 Werte, durch 24 geteilt). **PCA** (Singulärzerlegung) auf die ersten 2 oder 8 Hauptachsen; **Autoencoder** von Grund auf (NumPy, Backpropagation von Hand, Adam):
  40 → 16 (tanh) → Code (2 oder 8) → 16 (tanh) → 40, Xavier-Start mit festem Zufall, 400 Schritte mit Lernrate 0.01 über alle Daten.
- **Zellen:** der Code (bei den handgemachten die Merkmale selbst) wird mit Mittel und Standardabweichung der Trainingscodes standardisiert, ein k-Means (SciPy `kmeans2`, k-means++-Start, 25 Iterationen, fester Zufall) legt 256 Schwerpunkte hinein; die Zelle ist der nächste Schwerpunkt.
- **Einmal gelernt („fixed“):** Encoder und Schwerpunkte werden vorab auf 10 000 Zufallsfahrten im jeweiligen Labyrinth gelernt (die handgemachten Beschreibungen: 30 000 Fahrten für die Schwerpunkte) und in `spaces.npz` gespeichert; der Raum ist im Lauf fest.
- **AURORA („aur“):** die ersten 5 Generationen sind Zufallsfahrten (500), aus ihren Rohdaten wird die erste Beschreibung gelernt und alle 500 Fahrten werden eingetragen. In den Generationen 10, 30, 60, 100, 150 und 250 wird **neu gelernt** (Trainingsdaten: die 500 Fahrten, die Rohdaten aller Elites und die Kinder der Generation),
  die Elites werden nach Qualität absteigend neu eingetragen (kollidieren zwei in einer Zelle, bleibt die bessere), dann die Kinder. Bis zur ersten gelernten Beschreibung gibt es kein Archiv (die Kurven stehen bei 0).
- **Kennzahlen, nicht an der Beschreibung hängend:** **Endzellen** (die Zahl der verschiedenen 2 × 2-Zellen des Labyrinths, in denen die Elites enden, von 144) und **Wegzellen** (die 2 × 2-Zellen, die irgendeine Elite-Fahrt berührt, von 144); dazu belegte Zellen, QD-Score, mittlere Direktheit je Elite, Erfolg (Endposition im Zielkreis),
  Restweg und der Rekonstruktionsfehler des Encoders auf den Rohdaten der Elites. Beide Maße bevorzugen eine Beschreibung, die den Ort enthält; das ist Teil der Frage (siehe Grenzen).
- **Eigene Umsetzung:** PCA, Autoencoder, Zellzuordnung, Archiv und AURORA-Schleife sind eigener NumPy-/SciPy-Code; die Quelle verwaltet das Archiv anders (Abstandsschwelle statt fester Zellen) und lernt aus Sensordaten; hier sind die Rohdaten Positionen.
- **Gegenproben:** (1) die Rohdaten, die handgemachten Merkmale und der Fahrstil von Hand an einer langsamen geraden Fahrt (die Positionen liegen in gleichen Abständen; Fahrstil 0.1 und 0.5) und gegen eine skalare Neuberechnung der Fahrt; ein Fahrstil, der zwei Fahrten an verschiedenen Orten gleich beschreibt;
  (2) die PCA gegen **scikit-learn** (gleicher Unterraum, gleicher Rekonstruktionsfehler, gleiche Varianzen) und von Hand an Punkten auf einer Geraden; (3) der **Gradient des Autoencoders** gegen Differenzenquotienten, der Autoencoder lernt einen Kreisbogen, den die PCA mit einem Wert nicht abbildet; (4) die Zellzuordnung gegen eine
  Brute-Force-Suche des nächsten Schwerpunkts, die Standardisierung, das k-Means (zu wenige Punkte, identische Punkte); (5) das **Neueintragen** von Hand (Kollision: die bessere bleibt); (6) **Orakel:** MAP-Elites mit jeder der sieben gespeicherten Räume **als Ganzes** gegen eine skalare Neuimplementierung der Buchführung mit Wörterbüchern
  (skalare Merkmale, skalarer Code in reinen Schleifen für Identität, PCA und Autoencoder, Brute-Force-Zuordnung, gleiche Zufallszahlen, am Ende dieselben Zellen, Qualitäten, Endpositionen und Kurven) sowie **die AURORA-Schleife als Ganzes** (Aufwärmen, erstes Lernen, Neulernen, Neueintragen) gegen dieselbe Buchführung;
  (7) die Direktheit gegen eine skalare Neuberechnung bei Genomlängen von 5 bis 140, der Restweg gegen SciPys Breitensuche; (8) kein Verfahren benutzt das Ziel; (9) sieben Studienläufe mit gespeichertem Raum werden mit demselben Seed bitgleich reproduziert, die AURORA-Läufe fallen als frischer Lauf in die gespeicherte Verteilung (siehe Grenzen).
- **Vorgerechnete Studie** (`generate_precomputed.py` → `spaces.npz` und `precomputed_sweep.json`, rund 3 Minuten parallel): 11 Beschreibungen (3 handgemachte, PCA und Autoencoder mit 2 und 8 Werten je einmal gelernt und je als AURORA) × 2 Stufen (freies Feld, Stufe 3) × 20 Läufe à 400 Generationen
  (40 000 Evaluationen) = 440 Läufe.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. 20 Läufe je Zelle (Seeds 0 bis 19), immer 256 Zellen; „±“ ist die Standardabweichung über die Läufe, „p“ der Mann-Whitney-Test. Der Rekonstruktionsfehler ist der mittlere quadratische Fehler auf den Rohdaten der **Elites** (nicht der Trainingsdaten),
bei AURORA mit dem zuletzt gelernten Encoder.

| Stufe | Beschreibung | belegte Zellen | QD-Score | mittlere Direktheit | verschiedene Endzellen (Mittel ± SD, von 144) | Wegzellen (Mittel ± SD, von 144) | Erfolg | Restweg | Rekonstruktionsfehler |
|---|---|---|---|---|---|---|---|---|---|
| freies Feld | Handgemacht: Endposition (2 Werte) | 256.00 | 213.60 | 0.834 | 137.15 ± 1.60 | 144.00 ± 0.00 | 20 von 20 | 0.00 | – |
| freies Feld | Handgemacht: 4 Wegpunkte (8 Werte) | 255.65 | 169.94 | 0.665 | 84.30 ± 3.03 | 143.95 ± 0.22 | 20 von 20 | 0.00 | – |
| freies Feld | Handgemacht: Fahrstil (2 Werte) | 255.65 | 199.76 | 0.781 | 33.80 ± 1.44 | 137.50 ± 2.31 | 20 von 20 | 0.00 | – |
| freies Feld | PCA, einmal gelernt (2 Werte) | 256.00 | 224.39 | 0.877 | 77.00 ± 3.58 | 144.00 ± 0.00 | 20 von 20 | 0.00 | 0.0174 |
| freies Feld | PCA, einmal gelernt (8 Werte) | 254.95 | 169.97 | 0.667 | 65.70 ± 3.11 | 143.85 ± 0.37 | 20 von 20 | 0.00 | 0.0008 |
| freies Feld | Autoencoder, einmal gelernt (2 Werte) | 256.00 | 228.04 | 0.891 | 47.20 ± 2.76 | 143.60 ± 0.88 | 20 von 20 | 0.00 | 0.0272 |
| freies Feld | Autoencoder, einmal gelernt (8 Werte) | 255.00 | 181.68 | 0.712 | 73.35 ± 4.38 | 143.55 ± 0.60 | 20 von 20 | 0.00 | 0.0013 |
| freies Feld | AURORA mit PCA (2 Werte) | 256.00 | 225.19 | 0.880 | 77.15 ± 3.67 | 144.00 ± 0.00 | 20 von 20 | 0.00 | 0.0120 |
| freies Feld | AURORA mit PCA (8 Werte) | 247.40 | 168.31 | 0.680 | 61.35 ± 3.67 | 144.00 ± 0.00 | 20 von 20 | 0.00 | 0.0007 |
| freies Feld | AURORA mit Autoencoder (2 Werte) | 255.25 | 230.46 | 0.903 | 52.85 ± 4.49 | 143.95 ± 0.22 | 20 von 20 | 0.00 | 0.0123 |
| freies Feld | AURORA mit Autoencoder (8 Werte) | 243.95 | 176.54 | 0.724 | 63.65 ± 4.91 | 144.00 ± 0.00 | 20 von 20 | 0.00 | 0.0013 |
| Stufe 3 | Handgemacht: Endposition (2 Werte) | 256.00 | 208.74 | 0.815 | 101.65 ± 1.95 | 120.70 ± 2.18 | 20 von 20 | 0.65 | – |
| Stufe 3 | Handgemacht: 4 Wegpunkte (8 Werte) | 255.20 | 157.48 | 0.617 | 61.20 ± 2.89 | 90.45 ± 3.03 | 15 von 20 | 1.55 | – |
| Stufe 3 | Handgemacht: Fahrstil (2 Werte) | 255.15 | 205.31 | 0.805 | 34.95 ± 2.84 | 96.00 ± 2.41 | 19 von 20 | 1.25 | – |
| Stufe 3 | PCA, einmal gelernt (2 Werte) | 255.75 | 213.14 | 0.833 | 55.45 ± 1.85 | 102.70 ± 6.11 | 12 von 20 | 2.60 | 0.0166 |
| Stufe 3 | PCA, einmal gelernt (8 Werte) | 254.90 | 160.84 | 0.631 | 54.10 ± 2.67 | 97.80 ± 4.35 | 18 von 20 | 0.90 | 0.0008 |
| Stufe 3 | Autoencoder, einmal gelernt (2 Werte) | 255.35 | 216.46 | 0.848 | 42.40 ± 1.50 | 91.95 ± 4.22 | 5 von 20 | 4.50 | 0.0318 |
| Stufe 3 | Autoencoder, einmal gelernt (8 Werte) | 254.75 | 175.39 | 0.688 | 53.80 ± 1.51 | 90.90 ± 2.85 | 13 von 20 | 1.65 | 0.0012 |
| Stufe 3 | AURORA mit PCA (2 Werte) | 255.85 | 215.28 | 0.841 | 52.35 ± 2.60 | 98.35 ± 10.72 | 11 von 20 | 2.70 | 0.0110 |
| Stufe 3 | AURORA mit PCA (8 Werte) | 247.55 | 160.56 | 0.649 | 52.80 ± 2.59 | 99.45 ± 6.82 | 20 von 20 | 0.80 | 0.0007 |
| Stufe 3 | AURORA mit Autoencoder (2 Werte) | 255.50 | 221.54 | 0.867 | 42.05 ± 3.61 | 84.35 ± 6.55 | 10 von 20 | 2.70 | 0.0136 |
| Stufe 3 | AURORA mit Autoencoder (8 Werte) | 245.65 | 168.81 | 0.687 | 50.15 ± 3.23 | 91.85 ± 4.44 | 16 von 20 | 1.15 | 0.0011 |

| Frage | Befund |
|---|---|
| Wie viel ist das richtige Vorwissen wert? | **Sehr viel.** Im freien Feld enden die Elites bei der handgemachten Endposition in 137.15 verschiedenen Endzellen, bei 4 Wegpunkten in 84.30; die gelernten Beschreibungen kommen auf 47.20 bis 77.15 (2 Werte: PCA einmal 77.00, AURORA mit PCA 77.15, Autoencoder einmal 47.20, AURORA mit Autoencoder 52.85; 8 Werte: 65.70 / 73.35 / 61.35 / 63.65). Alle Unterschiede zur Endposition haben p < 0.001. Auf Stufe 3: 101.65 gegen 61.20 (4 Wegpunkte) und 42.05 bis 55.45 (gelernt). |
| Holt AURORA die handgemachte Beschreibung mit gleich vielen Werten ein? | **Nein.** Die gelernten 8-Werte-Beschreibungen (65.70 / 73.35 / 61.35 / 63.65 Endzellen im freien Feld) liegen unter den handgemachten 4 Wegpunkten (84.30), alle p < 0.001; auf Stufe 3 (54.10 / 53.80 / 52.80 / 50.15 gegen 61.20) ebenso. |
| Schlägt AURORA ein falsches Vorwissen? | **Bei den Endzellen ja.** Der Fahrstil (Weglänge und Drehung, ohne Ortsbezug) kommt auf 33.80 Endzellen im freien Feld und 34.95 auf Stufe 3; jede gelernte Beschreibung ist besser (alle p < 0.001). **Bei den Wegzellen auf Stufe 3 gemischt:** Fahrstil 96.00; die PCA liegt mit 102.70 / 97.80 / 98.35 / 99.45 darüber, belegt aber nur die einmal gelernte PCA mit 2 Werten (p = 0.00088; die drei anderen p > 0.05); der Autoencoder liegt mit 91.95 / 90.90 / 84.35 / 91.85 darunter (alle vier p < 0.01). Im freien Feld sind die Wegzellen fast gesättigt (144.00 bei der Endposition, 137.50 beim Fahrstil, alle anderen mindestens 143.55) und sagen nichts. |
| Lohnt es sich, neu zu lernen? | **Nicht durchgehend.** Endzellen im freien Feld, einmal gelernt gegen AURORA: PCA 2 Werte 77.00 / 77.15 (p = 0.93), PCA 8 Werte 65.70 / 61.35 (p = 0.0011), Autoencoder 2 Werte 47.20 / 52.85 (p < 0.001), Autoencoder 8 Werte 73.35 / 63.65 (p < 0.001); auf Stufe 3 55.45 / 52.35 (p < 0.001), 54.10 / 52.80 (p = 0.12), 42.40 / 42.05 (p = 0.54), 53.80 / 50.15 (p < 0.001). Von den acht Paaren ist AURORA nur einmal belegt besser (Autoencoder, 2 Werte, freies Feld), in vier belegt schlechter, in drei kein Unterschied. Beim Neueintragen gehen Zellen verloren: belegte Zellen 254.95 / 247.40 (PCA, 8 Werte) und 255.00 / 243.95 (Autoencoder, 8 Werte) von 256. |
| PCA oder Autoencoder? | **Keine einheitliche Ordnung.** Einmal gelernt, Endzellen im freien Feld: bei 2 Werten PCA 77.00 gegen Autoencoder 47.20, bei 8 Werten PCA 65.70 gegen Autoencoder 73.35 (beide p < 0.001). Der Autoencoder **rekonstruiert nicht besser** als die PCA (einmal gelernt, Fehler auf den Rohdaten der Elites: 0.0272 gegen 0.0174 bei 2 Werten und 0.0013 gegen 0.0008 bei 8 Werten im freien Feld; auf Stufe 3 0.0318 gegen 0.0166 und 0.0012 gegen 0.0008). Warum, ist nicht untersucht; naheliegend ist, dass 400 Schritte für das Netz zu wenig sind. |
| Sagt ein kleiner Rekonstruktionsfehler etwas über die Vielfalt? | **Nein.** Die PCA mit 8 Werten hat den kleinsten Fehler (0.0008), aber weniger Endzellen (65.70) als die PCA mit 2 Werten (77.00, Fehler 0.0174) im freien Feld. |
| Hilft der QD-Score beim Vergleich? | **Nicht über Beschreibungen hinweg.** Im freien Feld haben die gelernten 2-Werte-Beschreibungen einen höheren QD-Score (224.39 / 228.04 / 225.19 / 230.46) als die Endposition (213.60), bei viel weniger Endzellen; mittlere Direktheit 0.877 bis 0.903 gegen 0.834. Ein Raum mit weniger Verhaltensspielraum ist leichter zu füllen. |
| Und das Ziel (Stufe 3)? | **Die Endzellen sagen nicht, wer es erreicht.** Erfolg (von 20): Endposition 20, 4 Wegpunkte 15, Fahrstil 19; PCA einmal 12 / 18 (2 / 8 Werte), Autoencoder einmal 5 / 13, AURORA mit PCA 11 / 20, AURORA mit Autoencoder 10 / 16; Median der Evaluationen bis zum Erfolg bei der Endposition 3 526.5. Im freien Feld erreichen alle 11 Beschreibungen das Ziel in 20 von 20 Läufen. |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Die Vorab-Messreihe (8 Läufe, ohne die Aufwärmphase im Archiv) zeigte dieselbe Ordnung**, mit etwas anderen Zahlen: freies Feld, Endzellen: Endposition 136.2, gelernt 38.8 bis 78.0; Fahrstil war nicht dabei und kam erst mit der Studie hinzu.
- **Die Aufwärmphase wurde geändert.** In der Vorab-Fassung gingen die ersten vier Zufallsgenerationen nur ins Training; in der gebauten Fassung werden alle 500 Fahrten eingetragen.
- **Erwartet war ein Gewinn durch das Neulernen** (AURORAs Kernidee); gemessen ist er nur in einem von acht Paaren belegt, in vier ist AURORA belegt schlechter.
- **Der Autoencoder sollte die PCA schlagen.** Gemessen rekonstruiert er schlechter (siehe oben); die Ursache ist nicht untersucht.

## Ehrliche Grenzen

- **Die Rohdaten sind schon der Ort.** AURORA lernt in der Quelle aus Sensordaten (Bilder, Entfernungen); hier sind die Rohdaten 20 Positionen der Fahrt, also genau die Größe, auf die die Endzellen und Wegzellen zielen. Ob eine gelernte Beschreibung aus echten Sensordaten mehr findet, ist nicht gemessen;
  die Aussage der Quelle, gelernte Beschreibungen könnten handgemachte in der Vielfalt erreichen, wird hier nicht widerlegt, sondern in einer einfachen Umgebung nicht bestätigt.
- **Die Maße bevorzugen den Ort.** Endzellen und Wegzellen messen räumliche Vielfalt; eine Beschreibung der Endposition optimiert sie fast direkt. Eine andere Vielfalt (Fahrstile, Geschwindigkeitsprofile) würde den Fahrstil begünstigen; sie ist nicht gemessen.
- **Ein kleiner Autoencoder, ein Zufallsstart.** 40 → 16 → Code → 16 → 40, 400 Schritte, Xavier-Start mit festem Zufall; tiefere Netze, längeres Training, mehrere Starts, andere Aktivierungen und Standardisierung wurden nicht gemessen. Das Ergebnis „PCA mindestens so gut“ gilt für diese Einstellung.
- **Eigene Schleife.** Die Neulern-Zeitpunkte (10, 30, 60, 100, 150, 250), die Trainingsdaten und das Neueintragen mit CVT-Zellen sind eigene Festlegungen; die Referenz benutzt eine Abstandsschwelle, eine Archivgrößenregelung (AURORA-CSC) und anderes. Hier gehen beim Neueintragen Zellen verloren.
- **Zwei Stufen, eine Qualität, ein Budget.** Freies Feld und Stufe 3, Direktheit und 40 000 Evaluationen sind fest; Stufe 4 hängt am Suchraum (Stück 3) und wurde nicht neu gemessen.
- **Gelernte Läufe sind nicht plattformübergreifend bitgleich.** Die Singulärzerlegung und das Autoencoder-Training (BLAS, LAPACK) rechnen je Plattform in den letzten Stellen anders, und eine Zelle am Rand kippt dann; ein frischer AURORA-Lauf mit demselben Seed ist deshalb unter Linux ein anderer Zug aus derselben Verteilung als der gespeicherte Windows-Lauf
  (die Tests prüfen das als Band von 4 Standardabweichungen). Die Läufe mit gespeichertem Raum sind bitgleich; die Mittelwerte der Studie sind davon nicht betroffen (20 Läufe je Zelle).
- **Kleine, aber „signifikante“ Unterschiede.** Bei 20 Läufen und kleiner Streuung sind Unterschiede von wenigen Prozent mit p < 0.001 belegt; „kein Unterschied“ meint praktisch gleich (z. B. PCA 2 Werte, 77.00 gegen 77.15).
- **Das Ziel.** Das Archiv benutzt das Ziel nicht; „Erfolg“ ist eine Messgröße, kein Ziel der Suche. Die Erfolgszahlen auf Stufe 3 sind gemischt und nicht erklärt.

## Verwandte Demos im Portfolio

- [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo): Stück 1, das täuschende Labyrinth.
- [`novelty-search-demo`](https://github.com/sebastian-hanisch/novelty-search-demo): Stück 2, Novelty Search mit und ohne Archiv.
- [`behavior-space-demo`](https://github.com/sebastian-hanisch/behavior-space-demo): Stück 3, zwölf Verhaltensräume.
- [`novelty-objective-demo`](https://github.com/sebastian-hanisch/novelty-objective-demo): Stück 4, Novelty plus Ziel.
- [`map-elites-demo`](https://github.com/sebastian-hanisch/map-elites-demo): Stück 5, MAP-Elites.
- [`cma-me-demo`](https://github.com/sebastian-hanisch/cma-me-demo): Stück 6, CMA-ME und CMA-MAE.
- [`cvt-map-elites-demo`](https://github.com/sebastian-hanisch/cvt-map-elites-demo): Stück 7, CVT-MAP-Elites (Zellen und Schwerpunkte dieses Stücks).
- [`autoencoder-demo`](https://github.com/sebastian-hanisch/autoencoder-demo): der Autoencoder allein, ohne Archiv.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Die Rohdaten sind gut gewählt (hier schon der Ort) | kein Folgestück |
| Die Endzellen messen Vielfalt | kein Folgestück |
| Ein kleiner Autoencoder genügt | kein Folgestück |
| Neu lernen lohnt | kein Folgestück |
| Ein besseres Archiv löst das Ziel (Stufe 4 hängt am Suchraum) | Go-Explore und POET (Folgestücke) |

## Tests

262 Tests, rund eine Minute: Labyrinth und Roboter (aus Stück 1), PCA und Autoencoder (gegen scikit-learn, Differenzenquotienten, Kreisbogen), Merkmale, Zellzuordnung und gespeicherte Räume, Archiv, Neueintragen und Läufe (Reproduzierbarkeit, Aufwärmphase, Neulernen, wer das Ziel sieht),
**Orakel** (Direktheit, Merkmale und Code skalar, MAP-Elites mit sieben gespeicherten Räumen und die AURORA-Schleife als Ganzes gegen eine Wörterbuch-Buchführung, Restweg gegen SciPy), Auswertung und Vollständigkeit der vorgerechneten Datei (inklusive Neurechnung einzelner Läufe), Presets und Permalink, Diagramme (gesperrte Achsen, Fehlerbalken),
AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `aur_maze.py`, `aur_robot.py` | Labyrinth, Breitensuche, Täuschungsmaße; Kinematik (Genomlänge frei), Population, Pfad-Regler (wie Stück 1) |
| `aur_learn.py` | PCA und Autoencoder von Grund auf |
| `aur_space.py` | Merkmale, Verhaltensräume (Encoder und Zellen), k-Means |
| `aur_search.py` | Direktheit, Archiv, Neueintragen, Lauf (Hand, einmal gelernt, AURORA), Kennzahlen |
| `aur_evaluation.py` | Maße je Stufe, Lösbarkeit, Live-Paar, Zugriff auf Räume und Studie, Mann-Whitney-Hilfe |
| `generate_precomputed.py` | rechnet Räume und Studie vor → `spaces.npz`, `precomputed_sweep.json` |
| `aur_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `aur_presets.py`, `aur_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Grillotti, L., Cully, A.: Unsupervised behaviour discovery with quality-diversity optimisation. *IEEE Transactions on Evolutionary Computation*; arXiv:2106.05648.
- Vassiliades, V., Chatzilygeroudis, K., Mouret, J.-B.: Using centroidal Voronoi tessellations to scale up the multi-dimensional archive of phenotypic elites algorithm. *IEEE Transactions on Evolutionary Computation* (online 2017); arXiv:1610.05729.
- Mouret, J.-B., Clune, J. (2015): Illuminating search spaces by mapping elites. arXiv:1504.04909.
- Baldi, P., Hornik, K. (1989): Neural networks and principal component analysis: learning from examples without local minima. *Neural Networks* 2(1), 53–58.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Räume und Studie neu rechnen: `python generate_precomputed.py` (rund 3 Minuten auf 14 Prozessen).

Gebaut mit Streamlit und Plotly (die Simulation und der Autoencoder rechnen NumPy, das k-Means und der KD-Baum SciPy).
