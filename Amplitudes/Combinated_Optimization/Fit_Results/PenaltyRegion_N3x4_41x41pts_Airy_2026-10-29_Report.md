# Penalty-Scan - N3x4, 41x41 pts, airy, 2026-10-29

An JEDEM (win_input, width)-Gitterpunkt wurde GENAU EINE (r_x, r_y)-Optimierung durchgefuehrt, die direkt gegen die Kombination aus hartem und atom-gewichtetem Ziel minimiert. Uniformity_hart/Crosstalk_hart UND Uniformity_weighted/Crosstalk_weighted wurden dabei am SELBEN (r_x, r_y) ausgewertet - die gefundenen Amplituden sind daher automatisch fuer BEIDE Kriterien gleichzeitig gueltig.

## Zielfunktion der Optimierung (Penalty-Term)

Pro Gitterpunkt minimiert, auf ROHEN (unnormierten) Metriken - eine gitterweite Normierung steht waehrend der Optimierung eines einzelnen Punktes noch nicht zur Verfuegung:

```
U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
J       = alpha*U_kombi + (1-alpha)*C_kombi   ->  min ueber (r_x, r_y)
```

Der Term combo_lambda*|Differenz| ist der Penalty-Term: er bestraft Amplituden, bei denen hartes und atom-gewichtetes Kriterium auseinanderlaufen.

## Score der Auswertung

Region, Bestpunkt und Score-Karte benutzen GENAU DIESE Groesse - dieselbe Formel wie oben, nur ueber das ganze Gitter statt ueber einen Punkt:

```
X_kombi = 0.5*(X_hart + X_weighted) + combo_lambda * |X_hart - X_weighted|
J       = alpha*Uniformity_kombi + (1-alpha)*Crosstalk_kombi
```

Es gibt keine gitterweite Normierung mehr. Der frueher hier verwendete normierte `combined_score` ist ersatzlos entfallen: der Optimierer hat ihn nie gesehen, er haengt am gescannten Fenster, und er hebt die atom-gewichteten Groessen gegenueber der harten Uniformity um ein Vielfaches an.

Parameter dieses Laufs: alpha = 0.700, combo_lambda = 0.750, combo_percentile = 25.0%.

## Region

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach dem rohen J); 368/1681 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= 0.0628).

- win_input (vor der Linse): 1.4750 .. 1.7000 mm
- effektiver Waist (nach der Linse): 0.7872 .. 0.9072 µm
- W: 1.7054 .. 2.0528 µm (width 0.2700 .. 0.3250 MHz)

## Bester Einzelpunkt (Minimum des rohen J)

- win_input = 1.7000 mm (0.7872 µm effektiver Waist)
- W = 1.8949 µm (width = 0.3000 MHz)
- Uniformity_hart = 3.563%, Crosstalk_hart = 5.343%
- Uniformity_weighted = 1.016%, Crosstalk_weighted = 1.399%
- Uniformity_kombi U_c = 4.200%, Crosstalk_kombi eta_c = 6.329%
- J (Score) = 4.8389%

**ACHTUNG: dieser Punkt liegt auf dem Rand des gescannten Fensters.** Das ist dann kein Optimum, sondern nur die Stelle, an der der Scan aufhoert - das wahre Minimum liegt ausserhalb. Abhilfe: den Scan-Bereich erweitern. Im Plot ist der Stern deshalb offen statt gefuellt.

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 0.9703 / 1.0979

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von Kombiniert mit Penalty, ROH (J der Optimierung), aufgetragen ueber Waist nach der Linse (µm):

```
W/µm = (1.985 +- 0.034) · waist/µm + (0.265 +- 0.034)
```

Umrechnung: W = 6.3162 µm/MHz * width (AOD-Parameter und Linsenkombination des Datensatzes). Intern, in MHz:

```
width/MHz = (0.3142 +- 0.0054) · waist/µm + (0.0420 +- 0.0054)
```

- Steigung a = 1.985 +- 0.034 µm/µm
- Achsenabschnitt b = 0.265 +- 0.034 µm
- R² = 0.9919
- gefitteter Bereich: 0.7872 .. 1.2775 µm
- verwendete Talpunkte: 30 von 30
- ausgeschlossen: 0 mit Minimum am Rand des gescannten Fensters, 0 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

Die angegebenen Fehler sind die Standardfehler der linearen Regression (`s² = SS_res/(n-2)`, daraus `sigma_a = sqrt(s²/S_tt)`). Sie sagen, wie stark a und b schwanken wuerden, wenn man den Scan mit gleichartigem Rauschen wiederholte - NICHT, wie gut eine Gerade das Problem beschreibt; dafuer steht R² daneben.

**Sie sind eine untere Schranke.** Die Talpunkte sind Ergebnisse einer Optimierung auf einem Gitter, keine unabhaengigen Messungen. Gitterschrittweite und die Wahl der Fuehrungsgroesse verschieben die Gerade um deutlich mehr: am 41x41-Datensatz liegen 0.196 (globales Minimum) und 0.283 (gefuehrt) auseinander, bei einem Standardfehler von rund 0.004.

- **Suchbereich eingeschraenkt auf waist 0.7871 .. 1.3000 µm und width 0.2800 .. 0.4500 MHz (W 1.769 .. 2.842 µm).** Ausserhalb wurde gar nicht erst nach einem Minimum gesucht. Ein Talpunkt auf der Grenze dieses Bereichs zaehlt trotzdem, solange es ausserhalb nicht weiter bergab geht - er ist dann ein echtes lokales Minimum, das die Grenze nur streift. Geht es draussen tiefer, faellt er als Randminimum heraus, denn dort waere nicht das Tal gefittet, sondern die eingestellte Grenze selbst.
- Talpunkte: je Spalte das LOKALE Minimum, das der Leitgeraden am naechsten liegt (Korridor +-0.030 MHz = +-0.189 µm).
  Lokal heisst: beide Nachbarn vorhanden und groesser - Punkte am Rand des Scan-Fensters und Punkte, die an den ausgeschlossenen verbotenen Bereich grenzen, kommen damit gar nicht erst in Frage.
- Leitgerade aus Uniformity, atom-gewichtet: W/µm = 2.15786 * waist/µm +0.03375 (width/MHz = 0.34164 * waist/µm +0.005344) (R² = 0.9977, 28 Punkte).
- **Einordnung:** die Leitgerade WAEHLT nur aus, sie verschiebt nichts - die Punkte sind echte lokale Minima der Fuehrungsgroesse und die Steigung ist deren eigene. Welcher der mehreren Minima-Zweige verfolgt wird, entscheidet aber die Leitgroesse. Diese Zahl ist also an sie gebunden und kein unabhaengiger Befund.

Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.

**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden benachbarten Gitterzeilen linear interpoliert.

## Arbeitspunkt - Zusammenfassung (Werte fuer den Text)

### Karten

- eta_h: 5.27 % .. 21.90 %
- eta_w: 0.67 % .. 8.99 %
- U_h: 3.40 % .. 33.36 %
- U_w: 0.21 % .. 4.91 %
- J_c: 4.84 % .. 31.05 %
- Minimum von U_w (0.211 %) bei omega = 1.673 µm, W = 2.65 µm (0.420 MHz)
- r_x: 0.967 .. 3.867
- r_y: 1.030 .. 5.000

### Gerade durch das Tal von J_c

```
W(omega) = a * omega + b = 1.98(3) * omega + 0.27(3) µm
```

- a = 1.9849 +- 0.0339, b = 0.2653 +- 0.0341 µm (intern a = 0.31425 +- 0.00537 MHz/µm, b = 0.04201 +- 0.00540 MHz)
- gefitteter Bereich: omega = 0.79 .. 1.28 µm, W = 1.83 .. 2.80 µm
- J_c entlang der Geraden (gefitteter Bereich): 4.85 % .. 7.11 %
- J_c entlang des Talpfads (die gefitteten Minima): 4.85 % .. 7.09 %
- U_h entlang der Geraden: 3.426 % .. 3.752 %
- U_w entlang der Geraden: 0.284 % .. 0.800 %
- eta_h entlang der Geraden: 5.288 % .. 11.406 %
- eta_w entlang der Geraden: 0.752 % .. 5.814 %
- r_x entlang der Geraden: 0.967 .. 0.977
- r_y entlang der Geraden: 1.128 .. 1.188

### Punkt auf der Geraden

- W = 2.34 µm (0.370 MHz)
- omega = 1.0437 +- 0.0026 µm (Lage auf der Geraden, korrelierte Fehler von a und b; untere Schranke)
- win vor der Linse (vor dem AOD) = 1.2821 mm

| Groesse | Wert +- sigma_num | Text | Interpolation | Scan-Rauschen |
|---|---|---|---|---|
| Amplituden-Verhaeltnis r_x | 0.97088 +- 0.00072 | 0.9709(7) | 0.00018 | 0.0007 |
| Amplituden-Verhaeltnis r_y | 1.15834 +- 0.002 | 1.158(2) | 0.00026 | 0.002 |
| Uniformity hart U_h | 3.5606 % +- 0.037 % | 3.56(4) % | 0.0032 | 0.037 |
| Crosstalk hart eta_h | 8.3992 % +- 0.035 % | 8.40(3) % | 0.0013 | 0.035 |
| Uniformity gewichtet U_w | 0.4192 % +- 0.0063 % | 0.419(6) % | 0.0022 | 0.0059 |
| Crosstalk gewichtet eta_w | 1.4211 % +- 0.032 % | 1.42(3) % | 0.025 | 0.02 |
| Uniformity kombiniert U_c | 4.3459 % +- 0.045 % | 4.35(5) % | 0.0034 | 0.045 |
| Crosstalk kombiniert eta_c | 10.1437 % +- 0.043 % | 10.14(4) % | 0.0046 | 0.043 |
| J (Score) | 6.0852 % +- 0.041 % | 6.09(4) % | 0.001 | 0.041 |

sigma_num = sqrt(Interpolation² + Scan-Rauschen²): Interpolation = |kubisch (4 Nachbarn in Waist-Richtung) - linear|, Scan-Rauschen = RMS-Residuum eines lokalen quadratischen 2D-Fits (5 width-Zeilen x 6 Waist-Spalten).

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
W/µm      > 1.41421 * waist/µm       (dasselbe in µm, W = 6.3162 µm/MHz * width)
```

- Steigung a = 0.223902 MHz/µm bei Faktor k = 2
- Im verbotenen Bereich (width <= a*waist): 212 von 1681 Gitterpunkten (12.6%)

**Diese Punkte wurden aus der Auswertung ausgeschlossen** (auf NaN gesetzt). Bester Punkt, Region, Talpfad und Geradenfit oben beziehen sich also nur auf den erlaubten Bereich. Der Score ist das rohe J und damit punktweise definiert - er aendert sich durch den Ausschluss NUR im verbotenen Bereich, nicht anderswo. (Das war anders, solange hier ein gitterweit normierter Score stand.)

## Scan-Parameter

- N_x=3, N_y=4, Profil=airy
- Gitterpunkte: 41 x 41
- r_bounds = (0.1, 5.0)
- airy_scale_factor = 1.4830 (`first_zero_radius = Faktor * waist`)
- sigma_atom = 107.6 nm (atom_temperature=17.00 µK, trap_freq_r=60.40 kHz)
