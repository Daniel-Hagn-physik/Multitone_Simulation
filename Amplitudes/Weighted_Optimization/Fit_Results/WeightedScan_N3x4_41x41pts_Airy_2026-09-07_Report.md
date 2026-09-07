# WeightedScan - N3x4, 41x41 pts, airy, 2026-09-07

Metrik-Familie: **atom-gewichtet**.

Quelldatei: `C:\Users\Legion\OneDrive\Desktop\Multitone_Simulation\Amplitudes\Weighted_Optimization\Results\scan_amp_data_weighted_N3x4_41x41pts_Airy_k1.483_interference.pkl`

An JEDEM (win_input, width)-Gitterpunkt wurde eine eigene (r_x, r_y)-Optimierung durchgefuehrt; die hier gezeigten Metriken sind die am jeweils gefundenen Optimum erreichten Werte.

## Score

Region, Bestpunkt und Score-Karte benutzen die ROHE Zielgroesse - dieselbe, die auch der Optimierer minimiert, ohne gitterweite Normierung:

```
J = alpha*Uniformity_w + (1-alpha)*Crosstalk_w
```

mit alpha = 0.700, Perzentil fuer die Region = 25.0%.

Bewusst KEINE gitterweite Min-Max-Normierung: die haengt am gescannten Fenster, dieselbe Physik ergaebe bei anderem Scan-Bereich andere Zahlen.

## Region

Groesstes achsenparalleles Rechteck innerhalb der besten 25% aller Gitterpunkte (nach dem rohen J); 421/1681 Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= 0.007199).

- win_input (vor der Linse): 1.3175 .. 1.7000 mm
- effektiver Waist (nach der Linse): 0.7872 .. 1.0157 µm
- width: 0.2500 .. 0.2750 MHz

## Bester Einzelpunkt (Minimum des rohen J)

- win_input = 1.5875 mm (0.8429 µm effektiver Waist)
- width = 0.2800 MHz
- Uniformity = 0.4000%, Crosstalk = 0.6397%
- J (Score) = 0.4719%

- Amplituden-Verhaeltnisse an diesem Punkt: r_x / r_y = 0.9537 / 1.0129

## Talschnitt: Gerade durch den Talpfad

Gerade durch den Talpfad des Minimums von J = alpha*Uniformity + (1-alpha)*Crosstalk (Zielgroesse), aufgetragen ueber Waist nach der Linse (µm):

```
width/MHz = 0.27644 · waist/µm + 0.044187
```

- Steigung a = 0.276437 MHz/µm
- Achsenabschnitt b = 0.044187 MHz
- R² = 0.9767
- gefitteter Bereich: 0.8086 .. 1.0705 µm
- verwendete Talpunkte: 9 von 12
- ausgeschlossen: 0 mit Minimum am Rand des gescannten Fensters, 3 auf einem abgesetzten Nebenzweig bzw. als Rand-Kink

- Talpunkte: je Spalte das LOKALE Minimum, das der Leitgeraden am naechsten liegt (Korridor +-0.030 MHz).
  Lokal heisst: beide Nachbarn vorhanden und groesser - Punkte am Rand des Scan-Fensters und Punkte, die an den ausgeschlossenen verbotenen Bereich grenzen, kommen damit gar nicht erst in Frage.
- Leitgerade aus Uniformity, atom-gewichtet: width/MHz = 0.18931 * waist/µm +0.139585 (R² = 0.8073, 38 Punkte).
- In 29 von 41 Spalten lag kein lokales Minimum im Korridor; diese Spalten fehlen im Pfad.
- **Einordnung:** die Leitgerade WAEHLT nur aus, sie verschiebt nichts - die Punkte sind echte lokale Minima der Fuehrungsgroesse und die Steigung ist deren eigene. Welcher der mehreren Minima-Zweige verfolgt wird, entscheidet aber die Leitgroesse. Diese Zahl ist also an sie gebunden und kein unabhaengiger Befund.

Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.

**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden benachbarten Gitterzeilen linear interpoliert.

## Scan-Parameter

- N_x=3, N_y=4, Profil=airy
- Gitterpunkte: 41 x 41
- alpha = 0.700
- r_bounds = (0.1, 10.0)
- airy_scale_factor = 1.4830 (`first_zero_radius = Faktor * waist`)
- **Kohaerenz: nicht im Datensatz gespeichert** - der Scan lief vor dieser Option und hat die statische Interferenz frequenzentarteter Spots NICHT mitgerechnet. Mit neueren, kohaerent gerechneten Datensaetzen ist er deshalb nicht direkt vergleichbar.
- sigma_atom = 107.6 nm (atom_temperature=17.00 µK, trap_freq_r=60.40 kHz)
