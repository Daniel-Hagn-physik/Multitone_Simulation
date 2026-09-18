# HardScan - N3x4, 41x41 pts, airy, 2026-09-18

Metrik-Familie: **hart (globale Maske)**.

Quelldatei: `C:\Users\Legion\OneDrive\Desktop\Multitone_Simulation\Amplitudes\Hard_Optimization\Results\1.483_w_interference\scan_amp_data_N3x4_41x41pts_Airy_k1.483_hard_interference.pkl`

An JEDEM (win_input, width)-Gitterpunkt wurde eine eigene (r_x, r_y)-Optimierung durchgefuehrt; die hier gezeigten Metriken sind die am jeweils gefundenen Optimum erreichten Werte.

## Score

Region, Bestpunkt und Score-Karte benutzen die ROHE Zielgroesse - dieselbe, die auch der Optimierer minimiert, ohne gitterweite Normierung:

```
J = alpha*Uniformity + (1-alpha)*Crosstalk
```

mit alpha = 0.700, Perzentil fuer die Region = 25.0%.

Bewusst KEINE gitterweite Min-Max-Normierung: die haengt am gescannten Fenster, dieselbe Physik ergaebe bei anderem Scan-Bereich andere Zahlen.

## Region

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach dem rohen J); 368/1681 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= 0.052675).

- win_input (vor der Linse): 1.5200 .. 1.7000 mm
- effektiver Waist (nach der Linse): 0.7872 .. 0.8804 µm
- width: 0.2500 .. 0.3200 MHz

## Bester Einzelpunkt (Minimum des rohen J)

- win_input = 1.7000 mm (0.7872 µm effektiver Waist)
- width = 0.2900 MHz
- Uniformity = 3.5597%, Crosstalk = 5.2771%
- J (Score) = 4.0749%

**ACHTUNG: dieser Punkt liegt auf dem Rand des gescannten Fensters.** Das ist dann kein Optimum, sondern nur die Stelle, an der der Scan aufhoert - das wahre Minimum liegt ausserhalb. Abhilfe: den Scan-Bereich erweitern. Im Plot ist der Stern deshalb offen statt gefuellt.

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 0.9696 / 1.1266

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von J = alpha*Uniformity + (1-alpha)*Crosstalk (Zielgroesse), aufgetragen ueber Waist nach der Linse (µm):

```
width/MHz = (0.3454 +- 0.0079) · waist/µm + (0.0191 +- 0.0078)
```

- Steigung a = 0.3454 +- 0.0079 MHz/µm
- Achsenabschnitt b = 0.0191 +- 0.0078 MHz
- R² = 0.9861
- gefitteter Bereich: 0.7872 .. 1.2506 µm
- verwendete Talpunkte: 29 von 29
- ausgeschlossen: 0 mit Minimum am Rand des gescannten Fensters, 0 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

Die angegebenen Fehler sind die Standardfehler der linearen Regression (`s² = SS_res/(n-2)`, daraus `sigma_a = sqrt(s²/S_tt)`). Sie sagen, wie stark a und b schwanken wuerden, wenn man den Scan mit gleichartigem Rauschen wiederholte - NICHT, wie gut eine Gerade das Problem beschreibt; dafuer steht R² daneben.

**Sie sind eine untere Schranke.** Die Talpunkte sind Ergebnisse einer Optimierung auf einem Gitter, keine unabhaengigen Messungen. Gitterschrittweite und die Wahl der Fuehrungsgroesse verschieben die Gerade um deutlich mehr, als hier herauskommt.

- **Suchbereich eingeschraenkt auf waist 0.7871 .. 1.3000 µm und width 0.2800 .. 0.4500 MHz.** Ausserhalb wurde gar nicht erst nach einem Minimum gesucht. Ein Talpunkt auf der Grenze dieses Bereichs zaehlt trotzdem, solange es ausserhalb nicht weiter bergab geht - er ist dann ein echtes lokales Minimum, das die Grenze nur streift. Geht es draussen tiefer, faellt er als Randminimum heraus, denn dort waere nicht das Tal gefittet, sondern die eingestellte Grenze selbst.
- Talpunkte: je Spalte das LOKALE Minimum, das der Leitgeraden am naechsten liegt (Korridor +-0.030 MHz).
  Lokal heisst: beide Nachbarn vorhanden und groesser - Punkte am Rand des Scan-Fensters und Punkte, die an den ausgeschlossenen verbotenen Bereich grenzen, kommen damit gar nicht erst in Frage.
- Leitgerade aus Uniformity, hart: width/MHz = 0.36208 * waist/µm +0.003525 (R² = 0.9872, 28 Punkte).
- In 1 von 41 Spalten lag kein lokales Minimum im Korridor; diese Spalten fehlen im Pfad.
- **Einordnung:** die Leitgerade WAEHLT nur aus, sie verschiebt nichts - die Punkte sind echte lokale Minima der Fuehrungsgroesse und die Steigung ist deren eigene. Welcher der mehreren Minima-Zweige verfolgt wird, entscheidet aber die Leitgroesse. Diese Zahl ist also an sie gebunden und kein unabhaengiger Befund.

Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.

**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden benachbarten Gitterzeilen linear interpoliert.

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
- r_bounds = (0.1, 5.0)
- airy_scale_factor = 1.4830 (`first_zero_radius = Faktor * waist`)
- Kohaerenz: statische Interferenz mitgerechnet, Phasendifferenz 0 (voll konstruktiv, unguenstigster Fall), 1 entartete(s) Paar(e)
