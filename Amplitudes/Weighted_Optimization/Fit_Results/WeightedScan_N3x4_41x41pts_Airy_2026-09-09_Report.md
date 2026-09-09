# WeightedScan - N3x4, 41x41 pts, airy, 2026-09-09

Metrik-Familie: **atom-gewichtet**.

Quelldatei: `C:\Users\Legion\OneDrive\Desktop\Multitone_Simulation\Amplitudes\Weighted_Optimization\Results\1.483_w_interference\scan_amp_data_weighted_N3x4_41x41pts_Airy_k1.483_interference.pkl`

An JEDEM (win_input, width)-Gitterpunkt wurde eine eigene (r_x, r_y)-Optimierung durchgefuehrt; die hier gezeigten Metriken sind die am jeweils gefundenen Optimum erreichten Werte.

## Score

Region, Bestpunkt und Score-Karte benutzen die ROHE Zielgroesse - dieselbe, die auch der Optimierer minimiert, ohne gitterweite Normierung:

```
J = alpha*Uniformity_w + (1-alpha)*Crosstalk_w
```

mit alpha = 0.700, Perzentil fuer die Region = 25.0%.

Bewusst KEINE gitterweite Min-Max-Normierung: die haengt am gescannten Fenster, dieselbe Physik ergaebe bei anderem Scan-Bereich andere Zahlen.

## Region

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach dem rohen J); 368/1681 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= 0.006987).

- win_input (vor der Linse): 1.4975 .. 1.6550 mm
- effektiver Waist (nach der Linse): 0.8086 .. 0.8936 µm
- width: 0.2500 .. 0.3050 MHz

## Bester Einzelpunkt (Minimum des rohen J)

- win_input = 1.5875 mm (0.8429 µm effektiver Waist)
- width = 0.2800 MHz
- Uniformity = 0.4000%, Crosstalk = 0.6397%
- J (Score) = 0.4719%

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 0.9537 / 1.0129

## Markierter Punkt (Stern im Plot)

Beide Koordinaten selbst vorgegeben - hier steckt keine Gerade und keine Rechnung drin:

- Waist = 1.0100 µm  (win_input = 1.3249 mm)
- width = 0.3700 MHz

Der Punkt liegt in aller Regel ZWISCHEN den Gitterpunkten - er wird auch dort gezeichnet, nicht auf ein Gitter gerundet. Der naechstgelegene tatsaechlich gerechnete Gitterpunkt liegt bei Waist = 1.0157 µm / width = 0.3700 MHz.

### Werte an diesem Punkt

| Groesse | interpoliert | naechster Gitterpunkt |
|---|---|---|
| Uniformity U_w | 0.3080 % | 0.2983 % |
| Crosstalk eta_w | 1.6667 % | 1.5597 % |
| J (Score, roh) | 0.7156 % | 0.6767 % |
| Amplituden-Verhaeltnis r_x | 1.06903 | 1.05854 |
| Amplituden-Verhaeltnis r_y | 0.91192 | 0.92066 |

Interpoliert wird bilinear zwischen den vier umliegenden Gitterpunkten; liegt einer davon ausserhalb des Scans oder ist er ungueltig, steht dort n/a. Die rechte Spalte sind die wirklich gerechneten Werte am naechstgelegenen Gitterpunkt (Waist = 1.0157 µm, width = 0.3700 MHz).

Achtung bei r_x/r_y: das sind Optimierungs-ERGEBNISSE des Scans, keine glatten Funktionen. Wer die Metriken exakt an diesem Punkt braucht, muss die Amplituden dort neu optimieren.

## Verbotener Bereich (Ueberlappung der Eck-Spots)

Die beiden diagonal gegenueberliegenden Eck-Spots des 3x4-Arrays duerfen sich nicht ueberlappen. `width` ist die Gesamtspannweite des Tonarrays - in x und y derselbe Wert -, raeumlich also

```
S(width) = 6.3162 µm/MHz * width/MHz
d        = sqrt(2) * S            (Pythagoras, Eckabstand)
d        > 2 * waist          (Bedingung: kein Ueberlapp)
```

S ist linear in width (radius_from_angle geht ueber tan, aber theta liegt bei 1.2e-3 rad - die Abweichung von der Geraden ist 5e-7 relativ). Die Bedingung ist deshalb in der (waist, width)-Ebene exakt eine Ursprungsgerade:

```
width/MHz > 0.22390 * waist/µm       (erlaubt)
```

- Steigung a = 0.223902 MHz/µm bei Faktor k = 2
- Im verbotenen Bereich (width <= a*waist): 212 von 1681 Gitterpunkten (12.6%)

**Diese Punkte wurden aus der Auswertung ausgeschlossen** (auf NaN gesetzt). Bester Punkt, Region, Talpfad und Geradenfit oben beziehen sich also nur auf den erlaubten Bereich. Der Score ist punktweise definiert (rohes J, keine gitterweite Normierung) - er aendert sich durch den Ausschluss NUR im verbotenen Bereich, nicht anderswo.

## Scan-Parameter

- N_x=3, N_y=4, Profil=airy
- Gitterpunkte: 41 x 41
- alpha = 0.700
- r_bounds = (0.1, 10.0)
- airy_scale_factor = 1.4830 (`first_zero_radius = Faktor * waist`)
- **Kohaerenz: nicht im Datensatz gespeichert** - der Scan lief vor dieser Option und hat die statische Interferenz frequenzentarteter Spots NICHT mitgerechnet. Mit neueren, kohaerent gerechneten Datensaetzen ist er deshalb nicht direkt vergleichbar.
- sigma_atom = 107.6 nm (atom_temperature=17.00 µK, trap_freq_r=60.40 kHz)
