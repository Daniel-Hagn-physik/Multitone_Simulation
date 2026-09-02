# HardScan - N3x4, 41x41 pts, airy, 2026-09-02

Metrik-Familie: **hart (globale Maske)**.

Quelldatei: `C:\Users\Legion\OneDrive\Desktop\Multitone_Simulation\Amplitudes\Hard_Optimization\Results\scan_amp_data_N3x4_41x41pts_Airy.pkl`

An JEDEM (win_input, width)-Gitterpunkt wurde eine eigene (r_x, r_y)-Optimierung durchgefuehrt; die hier gezeigten Metriken sind die am jeweils gefundenen Optimum erreichten Werte.

## Score

Region, Bestpunkt und Score-Karte benutzen die ROHE Zielgroesse - dieselbe, die auch der Optimierer minimiert, ohne gitterweite Normierung:

```
J = alpha*Uniformity + (1-alpha)*Crosstalk
```

mit alpha = 0.700, Perzentil fuer die Region = 25.0%.

Bewusst KEINE gitterweite Min-Max-Normierung: die haengt am gescannten Fenster, dieselbe Physik ergaebe bei anderem Scan-Bereich andere Zahlen.

## Region

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach dem rohen J); 305/1681 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= 0.041189).

- win_input (vor der Linse): 1.4300 .. 1.6550 mm
- effektiver Waist (nach der Linse): 0.8086 .. 0.9358 µm
- width: 0.2100 .. 0.2600 MHz

## Bester Einzelpunkt (Minimum des rohen J)

- win_input = 1.6550 mm (0.8086 µm effektiver Waist)
- width = 0.2450 MHz
- Uniformity = 3.0458%, Crosstalk = 4.3914%
- J (Score) = 3.4495%

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 1.0026 / 1.1510

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von J = alpha*Uniformity + (1-alpha)*Crosstalk (Zielgroesse), aufgetragen ueber Waist nach der Linse (µm):

```
width/MHz = 0.30541 · waist/µm - 0.0066394
```

- Steigung a = 0.30541 MHz/µm
- Achsenabschnitt b = -0.00663935 MHz
- R² = 0.9585
- gefitteter Bereich: 0.8312 .. 1.0333 µm
- verwendete Talpunkte: 15 von 18
- ausgeschlossen: 3 mit Minimum am Rand des gescannten Fensters, 0 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

- **Suchbereich eingeschraenkt auf waist 0.8200 .. 1.1000 µm und width 0.2500 .. 0.4000 MHz.** Ausserhalb wurde gar nicht erst nach einem Minimum gesucht. Ein Talpunkt auf der Grenze dieses Bereichs zaehlt trotzdem, solange es ausserhalb nicht weiter bergab geht - er ist dann ein echtes lokales Minimum, das die Grenze nur streift. Geht es draussen tiefer, faellt er als Randminimum heraus, denn dort waere nicht das Tal gefittet, sondern die eingestellte Grenze selbst.
- Talpunkte: globales Minimum der Fuehrungsgroesse je Spalte.

Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.

**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden benachbarten Gitterzeilen linear interpoliert.

## Markierter Punkt (Stern im Plot)

Selbst vorgegeben: Width = 0.3100 MHz. Die zweite Koordinate kommt aus der Talpfad-Geraden (width/MHz = 0.30541 * waist/µm -0.006639):

- Waist = 1.0368 µm  (win_input = 1.2907 mm)
- width = 0.3100 MHz

Der Punkt liegt exakt auf der Geraden und damit in aller Regel ZWISCHEN den Gitterpunkten - er wird auch dort gezeichnet, nicht auf ein Gitter gerundet. Der naechstgelegene tatsaechlich gerechnete Gitterpunkt liegt bei Waist = 1.0333 µm / width = 0.3100 MHz.

### Werte an diesem Punkt

| Groesse | interpoliert | naechster Gitterpunkt |
|---|---|---|
| Uniformity U_h | 3.2017 % | 3.1925 % |
| Crosstalk eta_h | 5.9973 % | 5.9660 % |
| J (Score, roh) | 4.0403 % | 4.0246 % |
| Amplituden-Verhaeltnis r_x | 1.00222 | 1.00236 |
| Amplituden-Verhaeltnis r_y | 1.16839 | 1.16593 |

Interpoliert wird bilinear zwischen den vier umliegenden Gitterpunkten; liegt einer davon ausserhalb des Scans oder ist er ungueltig, steht dort n/a. Die rechte Spalte sind die wirklich gerechneten Werte am naechstgelegenen Gitterpunkt (Waist = 1.0333 µm, width = 0.3100 MHz).

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
- Im verbotenen Bereich (width <= a*waist): 462 von 1681 Gitterpunkten (27.5%)

**Diese Punkte wurden aus der Auswertung ausgeschlossen** (auf NaN gesetzt). Bester Punkt, Region, Talpfad und Geradenfit oben beziehen sich also nur auf den erlaubten Bereich. Der Score ist punktweise definiert (rohes J, keine gitterweite Normierung) - er aendert sich durch den Ausschluss NUR im verbotenen Bereich, nicht anderswo.

## Scan-Parameter

- N_x=3, N_y=4, Profil=airy
- Gitterpunkte: 41 x 41
- alpha = 0.700
- r_bounds = (0.1, 10.0)
- airy_scale_factor: nicht im Datensatz gespeichert - es galt der Optimierer-Default 1.19 (`first_zero_radius = Faktor * waist`)
