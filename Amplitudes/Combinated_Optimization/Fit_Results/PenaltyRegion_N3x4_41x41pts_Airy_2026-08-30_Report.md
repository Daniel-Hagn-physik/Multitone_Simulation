# Penalty-Scan - N3x4, 41x41 pts, airy, 2026-08-30

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

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach combined_score); 421/1681 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert combined_score <= 0.1855).

- win_input (vor der Linse): 1.3625 .. 1.6100 mm
- effektiver Waist (nach der Linse): 0.8312 .. 0.9821 µm
- width: 0.2150 .. 0.2650 MHz

## Bester Einzelpunkt (Minimum von combined_score)

- win_input = 1.4525 mm (0.9213 µm effektiver Waist)
- width = 0.2700 MHz
- Uniformity_hart = 3.092%, Crosstalk_hart = 4.876%
- Uniformity_weighted = 0.771%, Crosstalk_weighted = 0.935%
- Uniformity_kombi = 0.0954, Crosstalk_kombi = 0.1188 (normierte Einheiten)
- combined_score = 0.1024

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 0.9990 / 1.1791

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von Kombiniert mit Penalty (combined_score), aufgetragen ueber Waist nach der Linse (µm):

```
width/MHz = 0.28269 · waist/µm + 0.010395
```

- Steigung a = 0.282688 MHz/µm
- Achsenabschnitt b = 0.0103953 MHz
- R² = 0.9811
- gefitteter Bereich: 0.7872 .. 1.0333 µm
- verwendete Talpunkte: 19 von 41
- ausgeschlossen: 6 mit Minimum am Rand des gescannten Fensters, 16 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.

**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden benachbarten Gitterzeilen linear interpoliert.

## Scan-Parameter

- N_x=3, N_y=4, Profil=airy
- Gitterpunkte: 41 x 41
- r_bounds = (0.8, 2.2)
- sigma_atom = 107.6 nm (atom_temperature=17.00 µK, trap_freq_r=60.40 kHz)
