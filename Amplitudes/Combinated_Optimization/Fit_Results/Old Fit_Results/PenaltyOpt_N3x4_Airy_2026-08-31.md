# Penalty-Optimierung - N3x4, airy, 2026-08-31

Kein Gitter, kein Scan: die unten als VORGEGEBEN gelisteten Groessen wurden festgehalten, die als FREI gelisteten wurden gemeinsam so gewaehlt, dass die Penalty-Zielfunktion minimal wird. Harte und atom-gewichtete Metriken werden dabei bei jedem Schritt am SELBEN Parametersatz ausgewertet - das gefundene Ergebnis ist daher fuer beide Kriterien zugleich gueltig.

## Zielfunktion

```
U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
J       = alpha*U_kombi + (1-alpha)*C_kombi   ->  min
```

Auf ROHEN (unnormierten) Metriken - dieselbe Zielfunktion, die auch run_penalty_scan.py an jedem Gitterpunkt minimiert. Eine gitterweite Normierung gibt es hier nicht und kann es nicht geben: es gibt kein Gitter, ueber das normiert werden koennte. Die Zahlen sind deshalb mit den ROHEN Spalten eines Scan-Berichts vergleichbar, nicht mit dem dortigen combined_score.

Parameter dieses Laufs: alpha = 0.700, combo_lambda = 0.750.

## Vorgaben

| Groesse | Rolle | Wert bzw. Bereich |
|---|---|---|
| Waist nach der Linse (µm) | vorgegeben | 1.1000 µm |
| Width (MHz) | **frei** | 0.2500 MHz .. 0.5000 MHz |
| Amplituden-Verhaeltnis r_x | **frei** | 0.8000 .. 2.2000 |
| Amplituden-Verhaeltnis r_y | **frei** | 0.8000 .. 2.2000 |
| Brennweite f1 (mm) | vorgegeben | 60.000 mm |
| Brennweite f2 (mm) | vorgegeben | 750.000 mm |
| Brennweite fLO (mm) | vorgegeben | 52.8800 mm |

- N_x = 3, N_y = 4, Profil = airy
- offset = 100.0000 MHz, n_grid = 400
- lambda = 795.00 nm, pitch = 5.2880 µm, theta_max = 43.000 mrad, f_band = 36.000 MHz

## Ergebnis

| Groesse | Wert | |
|---|---|---|
| Waist nach der Linse (µm) | 1.1000 µm | vorgegeben |
| Width (MHz) | 0.2645 MHz | **optimiert** |
| Amplituden-Verhaeltnis r_x | 1.6407 | **optimiert** |
| Amplituden-Verhaeltnis r_y | 2.2000 | **optimiert** |
| Brennweite f1 (mm) | 60.000 mm | vorgegeben |
| Brennweite f2 (mm) | 750.000 mm | vorgegeben |
| Brennweite fLO (mm) | 52.8800 mm | vorgegeben |

Dazu gehoerender Eingangswaist VOR der Linse: **win_input = 0.9732 mm** (aus f1/f2 oben; das ist die Groesse, die run_penalty_scan.py scannt).

### Metriken an diesem Punkt

| | hart (Pitch-Box) | atom-gewichtet | kombiniert |
|---|---|---|---|
| Uniformity | 2.1399% | 0.7029% | 2.4992% |
| Crosstalk | 5.6458% | 0.6547% | 6.8936% |

**J = 0.038175** (dieselben Einheiten wie die rohen Metriken, also Bruchteile - nicht Prozent).

## Die einzelnen Startpunkte

Mehrere Startpunkte, weil die Zielfunktion nicht glatt ist (siehe unten). Liegen die besten Laeufe dicht beieinander, ist das Optimum belastbar; streuen sie, ist es eher eine von mehreren gleichwertigen Loesungen.

| Lauf | J | Width | Amplituden-Verhaeltnis r_x | Amplituden-Verhaeltnis r_y | Auswertungen |
|---|---|---|---|---|---|
| 1 | 0.038175 | 0.2645 MHz | 1.6407 | 2.2000 | 287 |
| 2 | 0.038175 | 0.2645 MHz | 1.6407 | 2.2000 | 292 |
| 3 | 0.043083 | 0.3019 MHz | 1.1986 | 1.7071 | 264 |
| 4 | 0.043083 | 0.3019 MHz | 1.1986 | 1.7071 | 392 |
| 5 | 0.048515 | 0.3401 MHz | 1.0470 | 1.3823 | 293 |
| 6 | 0.048517 | 0.3401 MHz | 1.0452 | 1.3831 | 258 |
| 7 | 0.048862 | 0.3991 MHz | 0.9968 | 1.1755 | 386 |
| 8 | 0.050345 | 0.3596 MHz | 1.0148 | 1.2906 | 395 |

Streuung von J ueber die 8 gueltigen Laeufe: 0.012170 (31.88% des besten Werts).

## Wie genau ist das?

Die harten Metriken rauschen. Ihr globales Intensitaetsgitter wird pro Auswertung neu aus Waist und width aufgebaut, wodurch die Abtastpunkte gegenueber den Fallen wandern und die Maskengrenzen um ganze Pixel springen (gemessen: Saegezahn von 0.05-0.09 Prozentpunkten im harten Crosstalk, waehrend der atom-gewichtete an derselben Stelle glatt ist). Die Zielfunktion hat dadurch feine lokale Minima, die nichts mit Physik zu tun haben. Deshalb mehrere Startpunkte - und deshalb sollte man die letzten Nachkommastellen des Optimums nicht ernst nehmen. Ein groesseres n_grid hilft dagegen nachweislich NICHT (gemessen ueber n_grid = 1000 .. 2400: Streuung ohne Konvergenztrend).

- Startpunkte: 8, davon gueltig: 8
- Auswertungen insgesamt: 2567
- Laufzeit: 2156.0 s

