# WeightedScan - N3x4, 41x41 pts, airy, 2026-09-03

Metrik-Familie: **atom-gewichtet**.

Quelldatei: `C:\Users\Legion\OneDrive\Desktop\Multitone_Simulation\Amplitudes\Weighted_Optimization\Results\zurHardMessung_scan_amp_data_weighted_N3x4_41x41pts_Airy_1000res.pkl`

An JEDEM (win_input, width)-Gitterpunkt wurde eine eigene (r_x, r_y)-Optimierung durchgefuehrt; die hier gezeigten Metriken sind die am jeweils gefundenen Optimum erreichten Werte.

## Score

Region, Bestpunkt und Score-Karte benutzen die ROHE Zielgroesse - dieselbe, die auch der Optimierer minimiert, ohne gitterweite Normierung:

```
J = alpha*Uniformity_w + (1-alpha)*Crosstalk_w
```

mit alpha = 0.700, Perzentil fuer die Region = 25.0%.

Bewusst KEINE gitterweite Min-Max-Normierung: die haengt am gescannten Fenster, dieselbe Physik ergaebe bei anderem Scan-Bereich andere Zahlen.

## Region

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach dem rohen J); 305/1681 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= 0.003153).

- win_input (vor der Linse): 1.3625 .. 1.5875 mm
- effektiver Waist (nach der Linse): 0.8429 .. 0.9821 µm
- width: 0.2200 .. 0.2600 MHz

## Bester Einzelpunkt (Minimum des rohen J)

- win_input = 1.5200 mm (0.8804 µm effektiver Waist)
- width = 0.2450 MHz
- Uniformity = 0.0683%, Crosstalk = 0.3986%
- J (Score) = 0.1674%

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 0.9417 / 0.9278

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von J = alpha*Uniformity + (1-alpha)*Crosstalk (Zielgroesse), aufgetragen ueber Waist nach der Linse (µm):

```
width/MHz = 0.2182 · waist/µm + 0.050224
```

- Steigung a = 0.218201 MHz/µm
- Achsenabschnitt b = 0.0502242 MHz
- R² = 0.9235
- gefitteter Bereich: 0.8312 .. 1.0516 µm
- verwendete Talpunkte: 10 von 10
- ausgeschlossen: 0 mit Minimum am Rand des gescannten Fensters, 0 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

- **Suchbereich eingeschraenkt auf waist 0.7871 .. 1.3000 µm und width 0.2250 .. 0.4000 MHz.** Ausserhalb wurde gar nicht erst nach einem Minimum gesucht. Ein Talpunkt auf der Grenze dieses Bereichs zaehlt trotzdem, solange es ausserhalb nicht weiter bergab geht - er ist dann ein echtes lokales Minimum, das die Grenze nur streift. Geht es draussen tiefer, faellt er als Randminimum heraus, denn dort waere nicht das Tal gefittet, sondern die eingestellte Grenze selbst.
- Talpunkte: je Spalte das LOKALE Minimum, das der Leitgeraden am naechsten liegt (Korridor +-0.030 MHz).
  Lokal heisst: beide Nachbarn vorhanden und groesser - Punkte am Rand des Scan-Fensters und Punkte, die an den ausgeschlossenen verbotenen Bereich grenzen, kommen damit gar nicht erst in Frage.
- Leitgerade aus Uniformity, atom-gewichtet: width/MHz = 0.28035 * waist/µm +0.005052 (R² = 0.9990, 30 Punkte).
- In 20 von 41 Spalten lag kein lokales Minimum im Korridor; diese Spalten fehlen im Pfad.
- **Einordnung:** die Leitgerade WAEHLT nur aus, sie verschiebt nichts - die Punkte sind echte lokale Minima der Fuehrungsgroesse und die Steigung ist deren eigene. Welcher der mehreren Minima-Zweige verfolgt wird, entscheidet aber die Leitgroesse. Diese Zahl ist also an sie gebunden und kein unabhaengiger Befund.

Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.

**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden benachbarten Gitterzeilen linear interpoliert.

## Markierter Punkt (Stern im Plot)

Selbst vorgegeben: Width = 0.2800 MHz. Die zweite Koordinate kommt aus der Talpfad-Geraden (width/MHz = 0.21820 * waist/µm +0.050224):

- Waist = 1.0530 µm  (win_input = 1.2708 mm)
- width = 0.2800 MHz

Der Punkt liegt exakt auf der Geraden und damit in aller Regel ZWISCHEN den Gitterpunkten - er wird auch dort gezeichnet, nicht auf ein Gitter gerundet. Der naechstgelegene tatsaechlich gerechnete Gitterpunkt liegt bei Waist = 1.0516 µm / width = 0.2800 MHz.

### Werte an diesem Punkt

| Groesse | interpoliert | naechster Gitterpunkt |
|---|---|---|
| Uniformity U_w | 0.0429 % | 0.0428 % |
| Crosstalk eta_w | 0.6363 % | 0.6320 % |
| J (Score, roh) | 0.2209 % | 0.2196 % |
| Amplituden-Verhaeltnis r_x | 0.89818 | 0.89899 |
| Amplituden-Verhaeltnis r_y | 0.94433 | 0.94328 |

Interpoliert wird bilinear zwischen den vier umliegenden Gitterpunkten; liegt einer davon ausserhalb des Scans oder ist er ungueltig, steht dort n/a. Die rechte Spalte sind die wirklich gerechneten Werte am naechstgelegenen Gitterpunkt (Waist = 1.0516 µm, width = 0.2800 MHz).

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
- r_bounds = (0.0, 2.0)
- airy_scale_factor: nicht im Datensatz gespeichert - es galt der Optimierer-Default 1.19 (`first_zero_radius = Faktor * waist`)
- sigma_atom = 107.6 nm (atom_temperature=17.00 µK, trap_freq_r=60.40 kHz)
