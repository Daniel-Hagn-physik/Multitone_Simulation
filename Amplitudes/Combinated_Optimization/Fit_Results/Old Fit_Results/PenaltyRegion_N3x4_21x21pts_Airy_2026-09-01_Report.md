# Penalty-Scan - N3x4, 21x21 pts, airy, 2026-09-01

An JEDEM (win_input, width)-Gitterpunkt wurde GENAU EINE (r_x, r_y)-Optimierung durchgefuehrt, die direkt gegen die Kombination aus hartem und atom-gewichtetem Ziel minimiert. Uniformity_hart/Crosstalk_hart UND Uniformity_weighted/Crosstalk_weighted wurden dabei am SELBEN (r_x, r_y) ausgewertet - die gefundenen Amplituden sind daher automatisch fuer BEIDE Kriterien gleichzeitig gueltig.

## Zielfunktion der Optimierung (Penalty-Term)

Pro Gitterpunkt minimiert, auf ROHEN (unnormierten) Metriken - eine gitterweite Normierung steht waehrend der Optimierung eines einzelnen Punktes noch nicht zur Verfuegung:

```
U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
J       = alpha*U_kombi + (1-alpha)*C_kombi   ->  min ueber (r_x, r_y)
```

Der Term combo_lambda*|Differenz| ist der Penalty-Term: er bestraft Amplituden, bei denen hartes und atom-gewichtetes Kriterium auseinanderlaufen.

## Kombinationsformel der Auswertung

Fuer die folgende Region-/Score-Uebersicht (NICHT fuer die Amplituden-Suche selbst) wird jede der vier Rohgroessen unabhaengig ueber das Scan-Gitter Min-Max-normiert (X_norm in [0,1]). Daraus:

```
X_kombi        = 0.5*(X_hart_norm + X_weighted_norm)
                 + combo_lambda * |X_hart_norm - X_weighted_norm|
combined_score = alpha*Uniformity_kombi + (1-alpha)*Crosstalk_kombi
```

Parameter dieses Laufs: alpha = 0.700, combo_lambda = 0.750, combo_percentile = 25.0%.

## Region

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach combined_score); 111/441 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert combined_score <= 0.1907).

- win_input (vor der Linse): 1.3400 .. 1.7000 mm
- effektiver Waist (nach der Linse): 0.7872 .. 0.9986 µm
- width: 0.2200 .. 0.2500 MHz

## Bester Einzelpunkt (Minimum von combined_score)

- win_input = 1.7000 mm (0.7872 µm effektiver Waist)
- width = 0.2300 MHz
- Uniformity_hart = 3.164%, Crosstalk_hart = 4.113%
- Uniformity_weighted = 1.015%, Crosstalk_weighted = 0.564%
- Uniformity_kombi = 0.1467, Crosstalk_kombi = 0.0361 (normierte Einheiten)
- combined_score = 0.1135

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 0.9997 / 1.1869

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von Kombiniert mit Penalty, ROH (J der Optimierung), aufgetragen ueber Waist nach der Linse (µm):

```
width/MHz = 0.015833 · waist/µm + 0.22347
```

- Steigung a = 0.0158332 MHz/µm
- Achsenabschnitt b = 0.223473 MHz
- R² = 0.0884
- gefitteter Bereich: 0.7872 .. 1.6727 µm
- verwendete Talpunkte: 9 von 21
- ausgeschlossen: 12 mit Minimum am Rand des gescannten Fensters, 0 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.

**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden benachbarten Gitterzeilen linear interpoliert.

## Scan-Parameter

- N_x=3, N_y=4, Profil=airy
- Gitterpunkte: 21 x 21
- r_bounds = (0.1, 10.0)
- sigma_atom = 107.6 nm (atom_temperature=17.00 µK, trap_freq_r=60.40 kHz)
