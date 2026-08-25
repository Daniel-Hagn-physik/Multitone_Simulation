# AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24 — geschlossene r_x/r_y-Formeln (zentraler Diagonalstreifen)

Automatisch generiert von `fit_central_amplitudes.py` am 2026-08-25.

Streifen: 14085/22801 Punkte (61.8%), Ridge: 6148 Punkte, Artefakt: 557 Punkte.

Bester Punkt (automatisch, best point (atom-weighted, global optimum)): waist = 1.5320 mm (vor der Linse), width = 0.2467 MHz.

## r_x

- Polynomgrad: 5, Ridge-alpha: 0.01
- R²(Block-CV): 0.9954 +/- 0.0026
- R²(volle Streifen-Daten, NICHT CV, nur Sanity-Check): 0.99840
- Max. Distillations-Fehler ggue. Pipeline-Modell: 9.06e-14
- Residuum-Modus im Diagnose-Plot: log

log(r_x) = sum_ij c_ij * waist_mm^i * width_MHz^j (21 Terme)

```
c(0,0) =  13.19960021
c(0,1) = -26.52166346
c(1,0) = -7.64899657
c(0,2) =  5.50953973
c(1,1) = -22.47693845
c(2,0) =  1.41301774
c(0,3) =  25.53336739
c(1,2) =  25.30996174
c(2,1) =  4.79698803
c(3,0) =  0.52721909
c(0,4) =  8.42163392
c(1,3) =  40.58885383
c(2,2) =  21.14883378
c(3,1) =  0.22480496
c(4,0) = -0.14410667
c(0,5) = -19.03739834
c(1,4) = -99.14631262
c(2,3) =  1.80278648
c(3,2) =  3.20339562
c(4,1) = -0.83363079
c(5,0) =  0.00058047
```

## r_y

- Polynomgrad: 4, Ridge-alpha: 0.01
- R²(Block-CV): 0.9958 +/- 0.0019
- R²(volle Streifen-Daten, NICHT CV, nur Sanity-Check): 0.99868
- Max. Distillations-Fehler ggue. Pipeline-Modell: 6.02e-14
- Residuum-Modus im Diagnose-Plot: log

log(r_y) = sum_ij c_ij * waist_mm^i * width_MHz^j (15 Terme)

```
c(0,0) =  16.94018934
c(0,1) = -42.93798106
c(1,0) = -10.54570753
c(0,2) =  23.68534994
c(1,1) = -18.03411162
c(2,0) =  1.92980593
c(0,3) =  59.97822032
c(1,2) =  59.39648006
c(2,1) =  7.68116744
c(3,0) =  0.94705135
c(0,4) = -65.14632532
c(1,3) = -68.82414564
c(2,2) =  13.15875129
c(3,1) = -2.26070182
c(4,0) = -0.32096537
```
