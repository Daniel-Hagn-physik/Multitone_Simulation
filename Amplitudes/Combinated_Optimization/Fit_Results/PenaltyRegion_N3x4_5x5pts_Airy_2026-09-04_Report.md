# Penalty-Scan - N3x4, 5x5 pts, airy, 2026-09-04

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

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach dem rohen J); 6/25 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= 6.415% (roh 0.06415)).

- win_input (vor der Linse): 1.7000 .. 1.7000 mm
- effektiver Waist (nach der Linse): 0.7872 .. 0.7872 µm
- width: 0.2500 .. 0.3000 MHz

## Bester Einzelpunkt (Minimum des rohen J)

- win_input = 1.7000 mm (0.7872 µm effektiver Waist)
- width = 0.3000 MHz
- Uniformity_hart = 3.994%, Crosstalk_hart = 5.530%
- Uniformity_weighted = 1.043%, Crosstalk_weighted = 1.417%
- Uniformity_kombi = 0.0473, Crosstalk_kombi = 0.0656 (rohe Einheiten)
- J (Score) = 5.280% (roh 0.05280)

**ACHTUNG: dieser Punkt liegt auf dem Rand des gescannten Fensters.** Das ist dann kein Optimum, sondern nur die Stelle, an der der Scan aufhoert - das wahre Minimum liegt ausserhalb. Abhilfe: den Scan-Bereich erweitern. Im Plot ist der Stern deshalb offen statt gefuellt.

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 0.9695 / 1.1238

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von Kombiniert mit Penalty, ROH (J der Optimierung), aufgetragen ueber Waist nach der Linse (µm):

```
width/MHz = 0.18918 · waist/µm + 0.16999
```

- Steigung a = 0.189183 MHz/µm
- Achsenabschnitt b = 0.169986 MHz
- R² = 0.7860
- gefitteter Bereich: 0.7872 .. 1.3055 µm
- verwendete Talpunkte: 4 von 4
- ausgeschlossen: 0 mit Minimum am Rand des gescannten Fensters, 0 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

- **Suchbereich eingeschraenkt auf waist 0.7871 .. 1.4000 µm.** Ausserhalb wurde gar nicht erst nach einem Minimum gesucht. Ein Talpunkt auf der Grenze dieses Bereichs zaehlt trotzdem, solange es ausserhalb nicht weiter bergab geht - er ist dann ein echtes lokales Minimum, das die Grenze nur streift. Geht es draussen tiefer, faellt er als Randminimum heraus, denn dort waere nicht das Tal gefittet, sondern die eingestellte Grenze selbst.
- Talpunkte: globales Minimum der Fuehrungsgroesse je Spalte.

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
- Im verbotenen Bereich (width <= a*waist): 4 von 25 Gitterpunkten (16.0%)

**Diese Punkte wurden aus der Auswertung ausgeschlossen** (auf NaN gesetzt). Bester Punkt, Region, Talpfad und Geradenfit oben beziehen sich also nur auf den erlaubten Bereich. Der Score ist das rohe J und damit punktweise definiert - er aendert sich durch den Ausschluss NUR im verbotenen Bereich, nicht anderswo. (Das war anders, solange hier ein gitterweit normierter Score stand.)

## Scan-Parameter

- N_x=3, N_y=4, Profil=airy
- Gitterpunkte: 5 x 5
- r_bounds = (0.1, 10.0)
- airy_scale_factor = 1.4830 (`first_zero_radius = Faktor * waist`)
- **Kohaerenz: nicht im Datensatz gespeichert** - der Scan lief vor dieser Option und hat die statische Interferenz frequenzentarteter Spots NICHT mitgerechnet. Mit neueren, kohaerent gerechneten Datensaetzen ist er deshalb nicht direkt vergleichbar.
- sigma_atom = 107.6 nm (atom_temperature=17.00 µK, trap_freq_r=60.40 kHz)
