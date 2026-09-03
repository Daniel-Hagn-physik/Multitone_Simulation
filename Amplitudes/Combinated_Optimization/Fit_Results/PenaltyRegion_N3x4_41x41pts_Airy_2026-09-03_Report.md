# Penalty-Scan - N3x4, 41x41 pts, airy, 2026-09-03

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

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach dem rohen J); 305/1681 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= 4.923% (roh 0.04923)).

- win_input (vor der Linse): 1.4300 .. 1.7000 mm
- effektiver Waist (nach der Linse): 0.7872 .. 0.9358 µm
- width: 0.2100 .. 0.2550 MHz

## Bester Einzelpunkt (Minimum des rohen J)

- win_input = 1.6550 mm (0.8086 µm effektiver Waist)
- width = 0.2450 MHz
- Uniformity_hart = 3.048%, Crosstalk_hart = 4.392%
- Uniformity_weighted = 1.009%, Crosstalk_weighted = 0.781%
- Uniformity_kombi = 0.0356, Crosstalk_kombi = 0.0529 (rohe Einheiten)
- J (Score) = 4.078% (roh 0.04078)

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 1.0005 / 1.1553

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von Kombiniert mit Penalty, ROH (J der Optimierung), aufgetragen ueber Waist nach der Linse (µm):

```
width/MHz = 0.28316 · waist/µm + 0.012676
```

- Steigung a = 0.283162 MHz/µm
- Achsenabschnitt b = 0.0126764 MHz
- R² = 0.9926
- gefitteter Bereich: 0.7872 .. 1.3655 µm
- verwendete Talpunkte: 33 von 34
- ausgeschlossen: 0 mit Minimum am Rand des gescannten Fensters, 1 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

- **Suchbereich eingeschraenkt auf waist 0.7871 .. 1.4000 µm und width 0.2250 .. 0.4000 MHz.** Ausserhalb wurde gar nicht erst nach einem Minimum gesucht. Ein Talpunkt auf der Grenze dieses Bereichs zaehlt trotzdem, solange es ausserhalb nicht weiter bergab geht - er ist dann ein echtes lokales Minimum, das die Grenze nur streift. Geht es draussen tiefer, faellt er als Randminimum heraus, denn dort waere nicht das Tal gefittet, sondern die eingestellte Grenze selbst.
- Talpunkte: je Spalte das LOKALE Minimum, das der Leitgeraden am naechsten liegt (Korridor +-0.030 MHz).
  Lokal heisst: beide Nachbarn vorhanden und groesser - Punkte am Rand des Scan-Fensters und Punkte, die an den ausgeschlossenen verbotenen Bereich grenzen, kommen damit gar nicht erst in Frage.
- Leitgerade aus Uniformity, atom-gewichtet: width/MHz = 0.29490 * waist/µm +0.000746 (R² = 0.9982, 32 Punkte).
- **Einordnung:** die Leitgerade WAEHLT nur aus, sie verschiebt nichts - die Punkte sind echte lokale Minima der Fuehrungsgroesse und die Steigung ist deren eigene. Welcher der mehreren Minima-Zweige verfolgt wird, entscheidet aber die Leitgroesse. Diese Zahl ist also an sie gebunden und kein unabhaengiger Befund.

Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.

**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden benachbarten Gitterzeilen linear interpoliert.

## Markierter Punkt (Stern im Plot)

Selbst vorgegeben: Width = 0.3100 MHz. Die zweite Koordinate kommt aus der Talpfad-Geraden (width/MHz = 0.28316 * waist/µm +0.012676):

- Waist = 1.0500 µm  (win_input = 1.2744 mm)
- width = 0.3100 MHz

Der Punkt liegt exakt auf der Geraden und damit in aller Regel ZWISCHEN den Gitterpunkten - er wird auch dort gezeichnet, nicht auf ein Gitter gerundet. Der naechstgelegene tatsaechlich gerechnete Gitterpunkt liegt bei Waist = 1.0516 µm / width = 0.3100 MHz.

### Werte an diesem Punkt

| Groesse | interpoliert | naechster Gitterpunkt |
|---|---|---|
| Uniformity hart U_h | 3.2374 % | 3.2416 % |
| Uniformity gewichtet U_w | 0.6314 % | 0.6283 % |
| Crosstalk hart eta_h | 6.1183 % | 6.1327 % |
| Crosstalk gewichtet eta_w | 0.8002 % | 0.7997 % |
| Amplituden-Verhaeltnis r_x | 1.00098 | 1.00095 |
| Amplituden-Verhaeltnis r_y | 1.18155 | 1.18271 |
| J (Score, roh) | 4.9566 % | 4.9663 % |

Interpoliert wird bilinear zwischen den vier umliegenden Gitterpunkten; liegt einer davon ausserhalb des Scans oder ist er ungueltig, steht dort n/a. Die rechte Spalte sind die wirklich gerechneten Werte am naechstgelegenen Gitterpunkt (Waist = 1.0516 µm, width = 0.3100 MHz).

Achtung bei r_x/r_y: das sind Optimierungs-ERGEBNISSE des Scans, keine glatten Funktionen. Wer die Metriken exakt an diesem Punkt braucht, muss die Amplituden dort neu optimieren - dafuer gibt es `run_penalty_only.py` (Waist und width fest vorgeben, r_x/r_y frei).

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

**Diese Punkte wurden aus der Auswertung ausgeschlossen** (auf NaN gesetzt). Bester Punkt, Region, Talpfad und Geradenfit oben beziehen sich also nur auf den erlaubten Bereich. Der Score ist das rohe J und damit punktweise definiert - er aendert sich durch den Ausschluss NUR im verbotenen Bereich, nicht anderswo. (Das war anders, solange hier ein gitterweit normierter Score stand.)

## Scan-Parameter

- N_x=3, N_y=4, Profil=airy
- Gitterpunkte: 41 x 41
- r_bounds = (0.8, 2.2)
- airy_scale_factor: nicht im Datensatz gespeichert - es galt der Optimierer-Default 1.19 (`first_zero_radius = Faktor * waist`)
- sigma_atom = 107.6 nm (atom_temperature=17.00 µK, trap_freq_r=60.40 kHz)
