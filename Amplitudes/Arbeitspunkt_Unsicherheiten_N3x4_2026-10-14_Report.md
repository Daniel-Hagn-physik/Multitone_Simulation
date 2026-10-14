# Arbeitspunkt mit Unsicherheiten - N3x4, Airy, combined / hard / weighted, 2026-10-14

Sammelbericht zu den letzten Auswertungen vom September. Hier steht jeder Wert, mit dem
weitergerechnet wird, zusammen mit seiner Unsicherheit und mit Angabe, woher sie kommt.
Nachgerechnet wurde mit `Amplitudes/arbeitspunkt_unsicherheiten.py` direkt aus den Scan-Dateien.
Die Werte der Reports vom 09.09. und 18.09. werden dabei auf die vierte Stelle reproduziert.

## Arbeitspunkte

| Familie | width | Waist (Atomebene) | Herkunft des Waist |
|---|---|---|---|
| **combined (Penalty)** | 0.370 MHz | **1.0437 µm** (gerundet 1.04 µm) | Talgerade combined vom 18.09.: width = 0.3142 · waist + 0.0420 |
| **hard** | 0.370 MHz | **1.0131 µm** | Talgerade hard vom 09.09. (`Thesis_HardScan`): width = 0.35599 · waist + 0.009352 |
| weighted | 0.370 MHz | 1.0437 µm (combined-Punkt) | kein eigener Punkt, nur zum Vergleich |

Die Gerade combined vom 18.09. ergibt mit ihren gerundeten Parametern 1.0439 µm. Aus den Werten,
die `run_plots.py` ausgegeben hat, laesst sich 1.0437 µm zurueckrechnen. Der Unterschied
(0.2 nm) liegt weit unter allen Unsicherheiten unten.

Feste Groessen: 3x4 Toene, f1/f2 = 75/750 mm, fLO = 52.88 mm, 795 nm, Airy mit
k = 1.483, alpha = 0.7, combo_lambda = 0.75, sigma_atom = 107.6 nm (17 µK, 60.4 kHz).

## Kurzfassung - zum Zitieren

Angegeben ist Wert ± sigma_num, die numerische Unsicherheit des Scans (Abschnitt *Methode*).
Die systematischen Beitraege stehen getrennt im Abschnitt *Systematik*. Sie sind nicht in
sigma_num enthalten.

| Groesse | combined @ 1.0437 µm | hard @ 1.0131 µm | weighted @ 1.0437 µm |
|---|---|---|---|
| Uniformity hart U_h | 3.560 ± 0.037 % | 3.473 ± 0.042 % | - |
| Uniformity gewichtet U_w | 0.4193 ± 0.0063 % | - | 0.2737 ± 0.0035 % |
| Crosstalk hart eta_h | 8.399 ± 0.035 % | 8.170 ± 0.022 % | - |
| Crosstalk gewichtet eta_w | 1.421 ± 0.032 % | - | 1.363 ± 0.030 % |
| Uniformity kombiniert U_c | 4.346 ± 0.045 % | - | - |
| Crosstalk kombiniert eta_c | 10.143 ± 0.043 % | - | - |
| J (Score, roh) | 6.085 ± 0.041 % | 4.882 ± 0.034 % | 0.600 ± 0.011 % |
| r_x | 0.9709 ± 0.0007 | 0.9688 ± 0.0006 | 1.0116 ± 0.0007 |
| r_y | 1.1583 ± 0.0020 | 1.1279 ± 0.0019 | 0.9427 ± 0.0041 |

J ist je Familie anders definiert:

- combined: J = 0.7 · U_c + 0.3 · eta_c
- hard: J = 0.7 · U_h + 0.3 · eta_h
- weighted: J = 0.7 · U_w + 0.3 · eta_w

Die drei J sind deshalb nicht untereinander vergleichbar.

## Quellen

| Familie | Scan-Datei (`Amplitudes/...`) | Scan gerechnet | Report |
|---|---|---|---|
| combined | `Combinated_Optimization/Results/1.483_w_interference/scan_amp_data_combined_N3x4_41x41pts_Airy_interfereence.pkl` | 07.09.2026 | `PenaltyRegion_N3x4_41x41pts_Airy_2026-09-18_Report.md` (Gerade), Werte am Punkt aus `run_plots.py` |
| hard | `Hard_Optimization/Results/1.483_w_interference/scan_amp_data_N3x4_41x41pts_Airy_k1.483_hard_interference.pkl` | 09.09.2026 | `Thesis_HardScan_N3x4_41x41pts_Airy_2026-09-09_Report.md` (Punkt + Gerade), `HardScan_..._2026-09-18_Report.md` |
| weighted | `Weighted_Optimization/Results/1.483_w_interference/scan_amp_data_weighted_N3x4_41x41pts_Airy_k1.483_interference.pkl` | 04.09.2026 | `WeightedScan_N3x4_41x41pts_Airy_2026-09-18_Report.md` |
| Atomposition | `Combinated_Optimization/Results/atom_offset_N3x4_Airy_w1.04um_width0.37MHz_rx0.97_ry1.16_k1.483.pkl` | 10.09.2026 | `AtomOffset_..._2026-09-10_Report.md` |

Hinweis: Im gespeicherten Report `PenaltyRegion_..._2026-09-18_Report.md` fehlt der Abschnitt
*Markierter Punkt*. Ein spaeterer Lauf ohne Vorgabe hat ihn ueberschrieben. Die combined-Werte
oben ersetzen ihn.

Gitter: 41 x 41. Die Waist-Schrittweite um den Punkt betraegt 18.3 nm (combined) bzw. 17.1 nm
(hard), die width-Schrittweite 5 kHz. width = 0.370 MHz liegt genau auf einer Gitterzeile.
Interpoliert wird also nur in der Waist-Richtung.

## Detail: combined (Penalty) @ 1.0437 µm / 0.370 MHz

| Groesse | Wert ± sigma_num | Gitter 1.0333 µm | Gitter 1.0516 µm (naechster) |
|---|---|---|---|
| U_h | 3.5604 ± 0.037 % | 3.5183 % | 3.5925 % |
| U_w | 0.4193 ± 0.0063 % | 0.4375 % | 0.4054 % |
| eta_h | 8.3989 ± 0.035 % | 8.3193 % | 8.4595 % |
| eta_w | 1.4211 ± 0.032 % | 1.4192 % | 1.4225 % |
| U_c | 4.3457 ± 0.045 % | 4.2885 % | 4.3893 % |
| eta_c | 10.143 ± 0.043 % | 10.044 % | 10.219 % |
| J | 6.0850 ± 0.041 % | 6.0153 % | 6.1381 % |
| r_x | 0.97087 ± 0.00072 | 0.96919 | 0.97216 |
| r_y | 1.1583 ± 0.0020 | 1.1479 | 1.1663 |

Unsicherheitsbudget und Empfindlichkeit:

| Groesse | Interpolation | Scan-Rauschen | sigma_num | dX pro +10 nm Waist | dX pro +5 kHz width |
|---|---|---|---|---|---|
| U_h | 0.003 | 0.037 | 0.037 | +0.033 | -0.035 |
| U_w | 0.002 | 0.006 | 0.006 | -0.019 | +0.017 |
| eta_h | 0.001 | 0.035 | 0.035 | +0.077 | -0.004 |
| eta_w | 0.025 | 0.020 | 0.032 | +0.003 | +0.014 |
| U_c | 0.003 | 0.045 | 0.045 | +0.047 | -0.048 |
| eta_c | 0.005 | 0.043 | 0.043 | +0.096 | -0.009 |
| J | 0.001 | 0.041 | 0.041 | +0.061 | -0.036 |
| r_x | 0.0002 | 0.0007 | 0.0007 | +0.0017 | -0.0025 |
| r_y | 0.0003 | 0.0020 | 0.0020 | +0.010 | -0.014 |

Die Metriken sind in Prozentpunkten angegeben, r ohne Einheit.

Exakt bei 1.0400 µm (statt 1.0437 µm) ergeben sich: U_h 3.545 %, U_w 0.426 %,
eta_h 8.371 %, eta_w 1.420 %, U_c 4.325 %, eta_c 10.108 %, J 6.060 %, r_x 0.9703,
r_y 1.1546. Die sigma_num-Werte bleiben gleich. Der Unterschied ist vollstaendig die
Waist-Empfindlichkeit oben mal 3.7 nm.

## Detail: hard @ 1.0131 µm / 0.370 MHz

| Groesse | Wert ± sigma_num | Gitter 0.9986 µm | Gitter 1.0157 µm (naechster) |
|---|---|---|---|
| U_h | 3.4734 ± 0.042 % | 3.4714 % | 3.4738 % |
| eta_h | 8.1698 ± 0.022 % | 8.0647 % | 8.1886 % |
| J | 4.8823 ± 0.034 % | 4.8494 % | 4.8882 % |
| r_x | 0.96878 ± 0.00057 | 0.96895 | 0.96875 |
| r_y | 1.1279 ± 0.0019 | 1.1163 | 1.1300 |

| Groesse | Interpolation | Scan-Rauschen | sigma_num | dX pro +10 nm Waist | dX pro +5 kHz width |
|---|---|---|---|---|---|
| U_h | 0.003 | 0.042 | 0.042 | -0.002 | +0.012 |
| eta_h | 0.00001 | 0.022 | 0.022 | +0.075 | -0.001 |
| J | 0.002 | 0.034 | 0.034 | +0.021 | +0.009 |
| r_x | 0.0001 | 0.0006 | 0.0006 | +0.0003 | -0.0005 |
| r_y | 0.0001 | 0.0018 | 0.0019 | +0.008 | -0.011 |

Am hard-Punkt liegt U_h im Tal: die Waist-Ableitung ist fast null. Das ist der Grund,
warum dieser Punkt fuer U_h robust ist.

Bei 1.0400 µm aus demselben hard-Scan: U_h 3.545 ± 0.037 %, eta_h 8.370 ± 0.035 %,
J 4.993 ± 0.034 %, r_x 0.9708, r_y 1.1525. Das entspricht dem combined-Punkt: dort dominiert
das harte Kriterium, und beide Scans liefern hier praktisch dieselben Werte.

## Detail: weighted @ 1.0437 µm / 0.370 MHz (Vergleich)

| Groesse | Wert ± sigma_num | Gitter 1.0333 µm | Gitter 1.0516 µm |
|---|---|---|---|
| U_w | 0.2737 ± 0.0035 % | 0.2803 % | 0.2687 % |
| eta_w | 1.363 ± 0.030 % | 1.370 % | 1.357 % |
| J_w | 0.600 ± 0.011 % | 0.607 % | 0.595 % |
| r_x | 1.0116 ± 0.0007 | 1.0280 | 0.9991 |
| r_y | 0.9427 ± 0.0041 | 0.9374 | 0.9467 |

Diese Werte gelten fuer die **eigenen** r_x/r_y des weighted-Scans (etwa 1.01 / 0.94), nicht
fuer die r_x/r_y des Arbeitspunkts. Mit den combined-Amplituden gelten U_w und eta_w aus der
combined-Tabelle. Am hard-Punkt (1.0131 µm) ergaebe der weighted-Scan U_w 0.30 ± 0.014 %,
eta_w 1.61 ± 0.06 %. Dort ist der Scan deutlich verrauschter: eta_w springt zwischen den
Nachbarpunkten von 1.56 auf 1.88 %.

## Methode: sigma_num

Die Scans sind deterministisch; ein statistischer Fehler im ueblichen Sinn existiert nicht. Was
streut, ist die Numerik.

1. **Interpolation:** der Unterschied zwischen linearer Interpolation (wie in `run_plots.py`) und
   einem kubischen Polynom durch die vier Nachbarpunkte der Gitterzeile.
2. **Scan-Rauschen:** An jedem Gitterpunkt wurde r_x/r_y einzeln optimiert. Zusaetzlich
   springen die harten Masken-Metriken mit dem Pixelraster (n_grid = 1000), deshalb liegen die
   Werte nicht auf einer glatten Flaeche. Gemessen wird das als RMS-Residuum eines lokalen
   quadratischen 2D-Fits (5 width-Zeilen x 6 Waist-Spalten um den Punkt, 6 Parameter,
   24 Freiheitsgrade). Echte Kruemmung, die eine Quadrik nicht erfasst, landet hier ebenfalls.
   Die Zahl ist deshalb eher grosszuegig.
3. sigma_num = sqrt(Interpolation² + Scan-Rauschen²).

Die Empfindlichkeiten dX/dWaist und dX/dwidth sind die linearen Terme desselben Fits. Mit ihnen
laesst sich jede Parameter-Unsicherheit aus dem Experiment fortpflanzen,
z. B. sigma_X = |dX/dWaist| · sigma_Waist.

**Gegenproben:**

- GUI-Export hard bei 1.04 µm / 0.37 MHz mit r = 0.97/1.16
  (`FlatMultiTone_GUI_hard_3x4_20260918_181655.txt`): U_h 3.5450 %, eta_h 8.3718 %. Der Scan
  ergibt bei 1.04 µm 3.5453 % und 8.370 %, also eine Abweichung unter 0.002 pp.
- AtomOffset-Report (eigenes Gitter auf der Region gegen das globale 1000er-Gitter): +0.025 pp
  bei U_h und +0.015 pp bei eta_h. Das ist die reine Pixelraster-Komponente und liegt innerhalb
  des Scan-Rauschens oben.

## Systematik (nicht in sigma_num)

### 1. Lage des Arbeitspunkts auf dem Talpfad

Die Fehler der Geradenfits (Standardfehler der Regression) fuehren, mit der Korrelation von a
und b, auf die Waist-Unsicherheit des Punkts bei 0.370 MHz:

| Gerade | a ± sigma_a (MHz/µm) | b ± sigma_b (MHz) | R² | Talpunkte | Waist bei 0.37 MHz |
|---|---|---|---|---|---|
| combined 18.09. | 0.3142 ± 0.0054 | 0.0420 ± 0.0054 | 0.9919 | 30 | 1.0439 ± 0.0026 µm |
| combined 09.09. | 0.3264 ± 0.0065 | 0.0309 ± 0.0063 | 0.9905 | 26 | 1.0389 ± 0.0028 µm |
| hard 18.09. | 0.3454 ± 0.0079 | 0.0191 ± 0.0078 | 0.9861 | 29 | 1.0159 ± 0.0032 µm |
| hard 09.09. | 0.3560 ± 0.0082 | 0.0094 ± 0.0079 | 0.9873 | 26 | 1.0129 ± 0.0030 µm |

Laut den Reports ist das eine **untere Schranke**. Realistischer ist der Unterschied zwischen den
Laeufen vom 09.09. und 18.09. (andere Suchgrenzen, gleicher Datensatz): 4.9 nm bei combined,
2.9 nm bei hard. Fortgepflanzt mit den Empfindlichkeiten oben:

| Groesse | combined: Fit-Fehler 2.6 nm | combined: Laufwahl 4.9 nm | hard: Fit-Fehler 3.0 nm | hard: Laufwahl 2.9 nm |
|---|---|---|---|---|
| U_h | 0.009 | 0.016 | 0.0005 | 0.0005 |
| eta_h | 0.020 | 0.038 | 0.022 | 0.022 |
| U_w | 0.005 | 0.009 | - | - |
| eta_w | 0.001 | 0.002 | - | - |
| J | 0.016 | 0.030 | 0.006 | 0.006 |

Alle Werte in Prozentpunkten. Diese Beitraege liegen unter sigma_num, die Lage auf dem Tal ist
also nicht der begrenzende Faktor.

### 2. Kohaerenz: Phase des entarteten Eckpaars

combined und hard sind mit statischer Interferenz und Tonphase 0 gerechnet, also im
ungünstigsten Fall (voll konstruktiv). Die beiden GUI-Exporte bei 1.04 µm / 0.37 MHz mit
r = 0.97/1.16 vergleichen das mit Quadratur:

| Tonphasen | U_h | eta_h | statischer Anteil |
|---|---|---|---|
| 0 (wie im Scan) | 3.5450 % | 8.3718 % | 2.99 % |
| Quadratur | 3.5468 % | 8.2294 % | 0.00 % |
| Differenz | +0.002 pp | **-0.142 pp** | |

eta_h ist also einseitig um bis zu 0.14 pp zu hoch, wenn die Phase im Aufbau nicht 0 ist.
Das ist der groesste Einzelbeitrag und viermal sigma_num. Es ist eine Modellannahme und kein
Zufallsfehler, deshalb steht er getrennt.

**weighted:** In diesem Datensatz ist keine Kohaerenz gespeichert. Der Scan lief, bevor es die
Option gab, und rechnet ohne statische Interferenz. Er ist deshalb nur bedingt mit combined und
hard vergleichbar.

### 3. r_x / r_y

r_x/r_y sind Optimierungsergebnisse an den Gitterpunkten und nur dazwischen interpoliert.
sigma_num beschreibt, wie stark sie von Punkt zu Punkt streuen. Exakte Werte an genau diesem
Punkt liefert eine Neuoptimierung mit `run_penalty_only.py` (combined) bzw. dem hard-Optimierer
mit fest vorgegebenem Waist und width.

## Positionsschwankung des Atoms (AtomOffset, 10.09.)

Gilt fuer combined bei 1.04 µm / 0.37 MHz mit r = 0.97/1.16 (glatte Werte, nicht die exakt
optimierten). Das Atom ist isotrop gaussverteilt mit sigma_pos = 127 nm je Achse: thermisch
bei 19 µK / 58.2 kHz 118 nm, technisch 47 nm nach Seubert et al. Ausgewertet per
Gauss-Hermite-Quadratur mit 7 x 7 Knoten.

| Groesse | Atom auf der Site | Mittelwert | Streuung (1 sigma, Schuss zu Schuss) |
|---|---|---|---|
| U_h | 3.57 % | 4.67 % | 1.14 pp |
| eta_h | 8.387 % | 8.41 % | 0.037 pp |
| U_w | 0.429 % | 0.71 % | 0.19 pp |
| eta_w | 1.398 % | 1.59 % | 0.19 pp |
| U_c | 4.36 % | 5.66 % | 1.39 pp |
| eta_c | 10.13 % | 10.11 % | 0.029 pp |
| J | 6.09 % | 7.00 % | 0.98 pp |

Das ist **keine Unsicherheit der Simulation**, sondern die physikalische Streuung von Schuss zu
Schuss. Bei den Uniformity-Groessen ist sie rund dreissigmal groesser als sigma_num.

Der thermische Anteil steckt in U_w und eta_w schon als Gewicht. Fuer diese beiden zaehlt
korrekt nur der technische Teil (47 nm), sonst wird er doppelt gezaehlt.

## Reproduzieren

```
cd Amplitudes
python arbeitspunkt_unsicherheiten.py
```

Das Skript gibt alle Tabellen dieses Berichts aus und schreibt sie zusaetzlich nach
`arbeitspunkt_unsicherheiten.json`. Arbeitspunkte und Geraden stehen oben im Skript in
`POINTS` und `LINES`.
