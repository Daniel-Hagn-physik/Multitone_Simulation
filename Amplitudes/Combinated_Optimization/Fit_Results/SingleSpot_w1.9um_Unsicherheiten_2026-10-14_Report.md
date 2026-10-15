# Single Spot bei 1.9 µm Waist - Werte und Unsicherheiten, 2026-10-14

Ein einzelner Airy-Spot, kein Tonarray. Ausgewertet wird an der Site und mit
dem Atom senkrecht versetzt (Shift r = 0 .. 1.9 µm = 0 .. 1 x Waist). Das ist
dieselbe Rechnung wie in `run_single_beam.py`. Alle Werte bei r = 0 und entlang
des Shifts sind neu gerechnet und stimmen auf alle Stellen mit den Reports vom
18.09. ueberein. Danach wurde jeder Parameter einzeln variiert.

Nachrechnen: `Amplitudes/Combinated_Optimization/single_spot_unsicherheiten.py`
(laeuft ohne PyQt, etwa 4 Minuten).

## Grundlage

| Groesse | Wert |
|---|---|
| Profil | Airy, first_zero_radius = 1.48295 x Waist |
| Waist (Atomebene) | 1.900 µm (win_input = 0.7043 mm) |
| pitch | 5.288 µm, 8 Nachbar-Sites (ein Kranz) |
| harte Region | Kreis r = 1 µm (U_h, eta_h^circ) bzw. Pitch-Quadrat 5.288 µm (eta_h^box), wandert mit dem Atom |
| gewichtet | sigma_atom = 107.6 nm (T = 17 µK, nu_r = 60.4 kHz), Sub-Gitter 241 x 241 ueber ±6 sigma |
| Penalty | alpha = 0.7, lambda = 0.75; eta_c und J benutzen eta_h^box |
| Shift | senkrecht (0, r) |

Quellen: `Fit_Results/SingleBeamOffset_Airy_w1.9um_r1um_beide-pitchbox_k1.483_2026-09-18_Report.md`
und `Fit_Results/SingleBeam_Airy_r1um_beide-pitchbox_k1.483_2026-09-18_Report.md` mit den
zugehoerigen .pkl in `Results/`.

## Welche Unsicherheiten eingehen

**Parameter (in den ± enthalten).** Die gemessenen Fallenparameter stehen im Kopf von
`weighted_multitone_flattop_optimizer.py`: pitch = 5.16(3) µm, T = 17(2) µK,
nu_r = 60.4(+2.4/−2.2) kHz. Fuer nu_r wird symmetrisch ±2.3 kHz angesetzt.

Fuer den Waist des Einzelspots gibt es keine Messung. **Angenommen sind ±70 nm**
(1.83 / 1.97 µm), uebernommen von der Fallen-Messung w0 = 0.86(7) µm. Gerechnet wird
dabei nicht linear, sondern direkt bei beiden Waists. Das ist noetig, denn eta_w sitzt
bei 1.9 µm fast im Minimum eines Airy-Rings: beide Richtungen erhoehen es, eine lineare
Fortpflanzung gaebe Unsinn. Daher sind viele Fehler **asymmetrisch**. Ist der echte
Waist-Fehler bekannt, steht in der Tabelle *Waist-Empfindlichkeit*, wie die Werte
dann skalieren.

Die Beitraege werden quadratisch addiert, getrennt nach oben und unten.

**Numerik (vernachlaessigbar).** Feinere Gitter (hart 801 statt 401, gewichtet 481 statt
241 Punkte ueber ±8 statt ±6 sigma) aendern keinen Wert um mehr als 0.0034 pp. Die
Numerik ist damit deutlich kleiner als jeder Parameterfehler und nicht mit
aufgefuehrt.

**Systematik (nicht in den ±, unten getrennt).**

- **Nachbarkranz:** gerechnet wird nur mit den 8 direkten Nachbarn.
- **pitch:** die Rechnung nimmt 5.288 µm, gemessen sind 5.16 µm.
- **Richtung des Shifts:** senkrecht gegen diagonal.

**Atomposition.** Die Streuung von Schuss zu Schuss ist eine eigene Tabelle. Sie ist
keine Unsicherheit eines Werts.

## Werte bei r = 0 mit Fehlerbudget

Alle Beitraege in Prozentpunkten (pp).

| Groesse | Wert | Numerik | Waist 1.83 / 1.97 µm | pitch ±30 nm | T ±2 µK | nu_r ±2.3 kHz | sigma_Parameter | Nachbarkranz 2 (syst.) | pitch 5.16 statt 5.288 µm (syst.) |
|---|---|---|---|---|---|---|---|---|---|
| U_h | 13.85 % | 9.7e-05 | +1.120 / -0.998 | +0.000 | +0.000 | +0.000 | +1.120 / −0.998 | +0.000 | +0.000 |
| eta_h^circ | 1.615 % | 1.7e-05 | -0.048 / +0.315 | -0.044 | +0.000 | +0.000 | +0.318 / −0.065 | +0.564 | +0.265 |
| eta_h^box | 12.49 % | 2.2e-07 | +0.039 / -0.021 | +0.006 | +0.000 | +0.000 | +0.040 / −0.022 | +2.662 | -0.020 |
| U_w | 1.064 % | 5.4e-07 | +0.082 / -0.074 | +0.000 | +0.124 | -0.080 | +0.169 / −0.165 | +0.000 | +0.000 |
| eta_w | 0.1523 % | 3.9e-09 | +0.470 / +0.088 | +0.032 | +0.008 | -0.005 | +0.471 / −0.033 | +0.392 | -0.004 |
| U_c | 17.04 % | 1.2e-04 | +1.379 / -1.229 | +0.000 | -0.031 | +0.020 | +1.380 / −1.230 | +0.000 | +0.000 |
| eta_c | 15.58 % | 2.8e-07 | -0.069 / -0.048 | -0.000 | -0.002 | +0.001 | +0.002 / −0.069 | +3.229 | -0.024 |
| J | 16.6 % | 8.5e-05 | +0.945 / -0.875 | -0.000 | -0.022 | +0.014 | +0.945 / −0.875 | +0.969 | -0.007 |

U_h und U_c werden vom Waist dominiert, U_w von der Temperatur. eta_c ist ohne
Systematik praktisch exakt (±0.07 pp). Bei den Crosstalks ist der fehlende zweite
Nachbarkranz der weitaus groesste Posten: +2.7 pp bei eta_h^box, +3.2 pp bei eta_c.

## Werte entlang des Shifts

Wert mit Parameter-Unsicherheit (Waist ±70 nm, pitch, T, nu_r), senkrechter Shift.

| r (µm) | r/w | U_h (%) | eta_h^circ (%) | eta_h^box (%) | U_w (%) | eta_w (%) | U_c (%) | eta_c (%) | J (%) |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.0 | 13.8 +1.1/−1.0 | 1.62 +0.32/−0.06 | 12.493 +0.040/−0.022 | 1.06 ± 0.17 | 0.15 +0.47/−0.03 | 17.0 +1.4/−1.2 | 15.579 +0.002/−0.069 | 16.60 ± 0.95 |
| 0.19 | 0.1 | 16.1 +1.3/−1.1 | 1.70 +0.32/−0.08 | 12.496 +0.016/−0.000 | 2.16 ± 0.25 | 0.26 +0.43/−0.03 | 19.6 +1.5/−1.4 | 15.56 +0.01/−0.10 | 18.4 ± 1.1 |
| 0.38 | 0.2 | 21.7 +1.6/−1.5 | 1.97 +0.34/−0.13 | 12.54 +0.12/−0.10 | 3.93 ± 0.42 | 0.59 +0.29/−0.01 | 26.1 +2.0/−1.8 | 15.53 +0.11/−0.20 | 22.9 ± 1.3 |
| 0.57 | 0.3 | 28.8 ± 2.1 | 2.42 +0.35/−0.20 | 12.75 ± 0.29 | 5.86 ± 0.62 | 1.18 +0.25/−0.03 | 34.5 ± 2.6 | 15.64 +0.30/−0.35 | 28.9 ± 1.7 |
| 0.76 | 0.4 | 36.8 ± 2.7 | 3.07 +0.35/−0.27 | 13.28 ± 0.49 | 7.91 ± 0.85 | 2.07 +0.33/−0.17 | 44.0 ± 3.3 | 16.08 ± 0.54 | 35.6 +2.1/−1.9 |
| 0.95 | 0.5 | 45.5 ± 3.4 | 3.94 ± 0.33 | 14.37 ± 0.72 | 10.1 ± 1.1 | 3.32 +0.35/−0.40 | 54.3 ± 4.0 | 17.13 ± 0.81 | 43.2 +2.6/−2.3 |
| 1.14 | 0.6 | 54.9 ± 4.1 | 5.05 +0.23/−0.33 | 16.26 ± 0.94 | 12.6 ± 1.4 | 5.04 +0.24/−0.52 | 65.5 ± 4.9 | 19.1 +1.1/−1.0 | 51.6 +3.1/−2.8 |
| 1.33 | 0.7 | 65.1 ± 4.9 | 6.44 +0.08/−0.23 | 19.3 ± 1.2 | 15.4 ± 1.8 | 7.33 +0.12/−0.39 | 77.5 ± 5.8 | 22.3 +1.5/−1.3 | 60.9 +3.7/−3.3 |
| 1.52 | 0.8 | 76.2 ± 5.8 | 8.20 +0.06/−0.28 | 23.7 ± 1.4 | 18.6 ± 2.3 | 10.4 +0.3/−1.0 | 90.6 ± 6.8 | 27.1 +1.9/−1.7 | 71.5 +4.3/−3.8 |
| 1.71 | 0.9 | 88.4 ± 6.8 | 10.54 +0.51/−0.70 | 30.1 ± 1.5 | 22.6 +2.9/−2.6 | 14.4 +2.1/−2.6 | 104.8 ± 7.8 | 34.0 ± 2.5 | 83.6 +4.8/−4.3 |
| 1.90 | 1.0 | 101.9 ± 7.7 | 14.0 ± 1.3 | 38.9 ± 1.7 | 27.8 +4.0/−3.5 | 19.8 ± 5.9 | 120.4 ± 8.8 | 43.7 ± 3.4 | 97.4 ± 5.2 |

### Unsicherheit des Shifts selbst

Ist r nur auf ±10 nm bekannt, aendern sich die Werte um die folgenden Betraege (pp pro
+10 nm). Fuer ein anderes sigma_r skaliert man linear.

| r (µm) | U_h | eta_h^circ | eta_h^box | U_w | eta_w | U_c | eta_c | J |
|---|---|---|---|---|---|---|---|---|
| 0.00 | +0.007 | +0.000 | -0.000 | +0.005 | +0.000 | +0.007 | -0.000 | +0.005 |
| 0.19 | +0.225 | +0.009 | +0.001 | +0.086 | +0.011 | +0.260 | -0.002 | +0.181 |
| 0.38 | +0.342 | +0.019 | +0.005 | +0.098 | +0.024 | +0.404 | +0.001 | +0.283 |
| 0.57 | +0.401 | +0.029 | +0.018 | +0.105 | +0.038 | +0.475 | +0.012 | +0.336 |
| 0.76 | +0.440 | +0.040 | +0.040 | +0.112 | +0.056 | +0.522 | +0.037 | +0.377 |
| 0.95 | +0.476 | +0.052 | +0.076 | +0.122 | +0.077 | +0.564 | +0.076 | +0.418 |
| 1.14 | +0.514 | +0.065 | +0.126 | +0.137 | +0.104 | +0.609 | +0.132 | +0.465 |
| 1.33 | +0.559 | +0.082 | +0.193 | +0.158 | +0.139 | +0.659 | +0.207 | +0.524 |
| 1.52 | +0.612 | +0.105 | +0.281 | +0.188 | +0.183 | +0.718 | +0.306 | +0.594 |
| 1.71 | +0.675 | +0.145 | +0.394 | +0.235 | +0.243 | +0.785 | +0.432 | +0.679 |
| 1.90 | +0.744 | +0.232 | +0.540 | +0.313 | +0.339 | +0.851 | +0.591 | +0.773 |

Bei r = 0 ist die Ableitung null: das Profil ist dort flach, ein kleiner Fehler im Shift
wirkt erst quadratisch. Ab r ≈ 0.2 µm reagieren U_h und J mit etwa 0.2–0.8 pp pro 10 nm.

### Waist-Empfindlichkeit

Aenderung in pp bei Waist 1.83 µm / 1.97 µm (−70 nm / +70 nm).

| r (µm) | U_h | eta_h^circ | eta_h^box | U_w | eta_w | U_c | eta_c | J |
|---|---|---|---|---|---|---|---|---|
| 0.00 | +1.120 / -0.998 | -0.048 / +0.315 | +0.039 / -0.021 | +0.082 / -0.074 | +0.470 / +0.088 | +1.379 / -1.229 | -0.069 / -0.048 | +0.945 / -0.875 |
| 0.19 | +1.269 / -1.135 | -0.065 / +0.320 | +0.003 / +0.016 | +0.167 / -0.149 | +0.425 / +0.110 | +1.544 / -1.381 | -0.103 / -0.007 | +1.050 / -0.969 |
| 0.38 | +1.645 / -1.479 | -0.114 / +0.334 | -0.102 / +0.123 | +0.307 / -0.274 | +0.293 / +0.170 | +1.979 / -1.781 | -0.201 / +0.111 | +1.325 / -1.213 |
| 0.57 | +2.149 / -1.939 | -0.186 / +0.346 | -0.262 / +0.285 | +0.464 / -0.414 | +0.090 / +0.252 | +2.570 / -2.320 | -0.350 / +0.293 | +1.694 / -1.536 |
| 0.76 | +2.734 / -2.471 | -0.263 / +0.344 | -0.458 / +0.484 | +0.642 / -0.571 | -0.154 / +0.323 | +3.257 / -2.946 | -0.534 / +0.524 | +2.120 / -1.905 |
| 0.95 | +3.388 / -3.066 | -0.319 / +0.307 | -0.669 / +0.701 | +0.849 / -0.752 | -0.384 / +0.334 | +4.023 / -3.644 | -0.741 / +0.793 | +2.594 / -2.313 |
| 1.14 | +4.113 / -3.725 | -0.321 / +0.211 | -0.879 / +0.918 | +1.102 / -0.970 | -0.510 / +0.205 | +4.865 / -4.414 | -0.971 / +1.096 | +3.114 / -2.761 |
| 1.33 | +4.915 / -4.459 | -0.221 / +0.027 | -1.071 / +1.117 | +1.426 / -1.242 | -0.371 / -0.184 | +5.787 / -5.263 | -1.245 / +1.442 | +3.678 / -3.252 |
| 1.52 | +5.802 / -5.276 | +0.035 / -0.272 | -1.232 / +1.284 | +1.867 / -1.603 | +0.312 / -1.017 | +6.786 / -6.195 | -1.618 / +1.860 | +4.265 / -3.778 |
| 1.71 | +6.764 / -6.181 | +0.508 / -0.702 | -1.351 / +1.406 | +2.511 / -2.113 | +2.059 / -2.579 | +7.827 / -7.198 | -2.204 / +2.402 | +4.817 / -4.318 |
| 1.90 | +7.741 / -7.148 | +1.258 / -1.261 | -1.410 / +1.461 | +3.546 / -2.892 | +5.925 / -5.338 | +8.790 / -8.212 | -3.244 / +3.160 | +5.180 / -4.800 |

## Systematik

### Zweiter Nachbarkranz (24 statt 8 Sites)

Airy faellt nur mit 1/r³ ab, die zweite Reihe Nachbarn traegt deshalb merklich bei. Die
Tabelle zeigt die Aenderung in pp bei ring = 2. Sie ist **einseitig**: die gerechneten
Crosstalks sind zu klein. Die Uniformity ist nicht betroffen.

| r (µm) | U_h | eta_h^circ | eta_h^box | U_w | eta_w | U_c | eta_c | J |
|---|---|---|---|---|---|---|---|---|
| 0.00 | +0.000 | +0.564 | +2.662 | +0.000 | +0.392 | +0.000 | +3.229 | +0.969 |
| 0.19 | +0.000 | +0.575 | +2.655 | +0.000 | +0.405 | +0.000 | +3.218 | +0.965 |
| 0.38 | +0.000 | +0.609 | +2.639 | +0.000 | +0.444 | +0.000 | +3.187 | +0.956 |
| 0.57 | +0.000 | +0.670 | +2.620 | +0.000 | +0.517 | +0.000 | +3.146 | +0.944 |
| 0.76 | +0.000 | +0.765 | +2.611 | +0.000 | +0.633 | +0.000 | +3.106 | +0.932 |
| 0.95 | +0.000 | +0.904 | +2.625 | +0.000 | +0.812 | +0.000 | +3.079 | +0.924 |
| 1.14 | +0.000 | +1.104 | +2.676 | +0.000 | +1.084 | +0.000 | +3.074 | +0.922 |
| 1.33 | +0.000 | +1.391 | +2.775 | +0.000 | +1.499 | +0.000 | +3.094 | +0.928 |
| 1.52 | +0.000 | +1.806 | +2.935 | +0.000 | +2.148 | +0.000 | +3.131 | +0.939 |
| 1.71 | +0.000 | +2.419 | +3.165 | +0.000 | +3.208 | +0.000 | +3.155 | +0.946 |
| 1.90 | +0.000 | +3.355 | +3.479 | +0.000 | +5.056 | +0.000 | +3.084 | +0.925 |

### pitch 5.16 µm (gemessen) statt 5.288 µm (gerechnet)

Aenderung in pp, wenn der gemessene pitch eingesetzt wird.

| r (µm) | U_h | eta_h^circ | eta_h^box | U_w | eta_w | U_c | eta_c | J |
|---|---|---|---|---|---|---|---|---|
| 0.00 | +0.000 | +0.265 | -0.020 | +0.000 | -0.004 | +0.000 | -0.024 | -0.007 |
| 0.19 | +0.000 | +0.273 | +0.007 | +0.000 | +0.020 | +0.000 | +0.004 | +0.001 |
| 0.38 | +0.000 | +0.294 | +0.087 | +0.000 | +0.089 | +0.000 | +0.087 | +0.026 |
| 0.57 | +0.000 | +0.323 | +0.222 | +0.000 | +0.192 | +0.000 | +0.230 | +0.069 |
| 0.76 | +0.000 | +0.348 | +0.415 | +0.000 | +0.311 | +0.000 | +0.441 | +0.132 |
| 0.95 | +0.000 | +0.357 | +0.672 | +0.000 | +0.414 | +0.000 | +0.737 | +0.221 |
| 1.14 | +0.000 | +0.332 | +1.004 | +0.000 | +0.447 | +0.000 | +1.143 | +0.343 |
| 1.33 | +0.000 | +0.260 | +1.429 | +0.000 | +0.327 | +0.000 | +1.705 | +0.511 |
| 1.52 | +0.000 | +0.147 | +1.975 | +0.000 | -0.082 | +0.000 | +2.490 | +0.747 |
| 1.71 | +0.000 | +0.052 | +2.686 | +0.000 | -0.989 | +0.000 | +3.604 | +1.081 |
| 1.90 | +0.000 | +0.191 | +3.622 | +0.000 | -2.695 | +0.000 | +5.201 | +1.560 |

Bei r = 0 sind die Folgen klein: eta_h^circ steigt um 0.27 pp, alles andere um weniger als
0.03 pp. Ab r ≳ 0.8 µm wird es merklich. Bei r = 1.9 µm nimmt eta_h^box um 3.6 pp zu und
eta_c um 5.2 pp, eta_w dagegen um 2.7 pp ab.

### Richtung: senkrecht gegen diagonal

Die Uniformity haengt nicht von der Richtung ab, der Crosstalk schon: die naechste
Nachbar-Site liegt in der Diagonale √2-mal weiter weg. Die Tabelle zeigt die Werte in %.
Der Report vom 18.09. hat nur senkrecht.

| r (µm) | eta_h^circ senkr. | eta_h^circ diag. | eta_h^box senkr. | eta_h^box diag. | eta_w senkr. | eta_w diag. | eta_c senkr. | eta_c diag. | J senkr. | J diag. |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1.615 | 1.615 | 12.49 | 12.49 | 0.1523 | 0.1523 | 15.58 | 15.58 | 16.6 | 16.6 |
| 0.19 | 1.703 | 1.703 | 12.5 | 12.49 | 0.2607 | 0.2604 | 15.56 | 15.55 | 18.41 | 18.41 |
| 0.38 | 1.969 | 1.969 | 12.54 | 12.52 | 0.5947 | 0.5902 | 15.53 | 15.5 | 22.94 | 22.93 |
| 0.57 | 2.42 | 2.426 | 12.75 | 12.64 | 1.182 | 1.16 | 15.64 | 15.51 | 28.86 | 28.82 |
| 0.76 | 3.071 | 3.093 | 13.28 | 12.94 | 2.069 | 2.005 | 16.08 | 15.68 | 35.64 | 35.51 |
| 0.95 | 3.939 | 4.009 | 14.37 | 13.57 | 3.324 | 3.188 | 17.13 | 16.17 | 43.18 | 42.89 |
| 1.14 | 5.05 | 5.235 | 16.26 | 14.68 | 5.038 | 4.816 | 19.07 | 17.14 | 51.55 | 50.97 |
| 1.33 | 6.444 | 6.88 | 19.27 | 16.44 | 7.331 | 7.087 | 22.25 | 18.78 | 60.93 | 59.89 |
| 1.52 | 8.204 | 9.133 | 23.74 | 19.06 | 10.36 | 10.39 | 27.09 | 21.23 | 71.53 | 69.77 |
| 1.71 | 10.54 | 12.34 | 30.11 | 22.78 | 14.37 | 15.52 | 34.05 | 24.6 | 83.61 | 80.77 |
| 1.90 | 14 | 17.17 | 38.93 | 27.88 | 19.8 | 24.39 | 43.71 | 28.75 | 97.39 | 92.9 |

Bis r ≈ 0.6 µm liegen beide Richtungen innerhalb von 0.15 pp. Darueber gilt: fuer
eta_h^box, eta_c und J ist senkrecht die unguenstigere Richtung (bei r = 1.9 µm 38.9 gegen
27.9 % bei eta_h^box). Fuer eta_h^circ und eta_w ist es umgekehrt, dort ist diagonal
unguenstiger (eta_w 24.4 gegen 19.8 %). Diese Richtungen laufen naeher an die Nachbar-Site
in der Ecke heran.

## Positionsschwankung des Atoms

Das Atom ist isotrop gaussverteilt um die Position (0, r) mit sigma_pos je Achse.
Ausgewertet per Gauss-Hermite-Quadratur mit 7 x 7 Knoten. "Streuung" ist die
Schuss-zu-Schuss-Schwankung der Groesse.

**sigma_pos = 47 nm (nur technisch).** Das ist die saubere Wahl fuer U_w und eta_w, weil
der thermische Anteil dort schon im Gewicht steckt.

| r (µm) | U_h Mittel ± Streuung (%) | eta_h^circ Mittel ± Streuung (%) | eta_h^box Mittel ± Streuung (%) | U_w Mittel ± Streuung (%) | eta_w Mittel ± Streuung (%) | U_c Mittel ± Streuung (%) | eta_c Mittel ± Streuung (%) | J Mittel ± Streuung (%) |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 14.14 ± 0.29 | 1.626 ± 0.011 | 12.49345 ± 0.00014 | 1.24 ± 0.16 | 0.166 ± 0.013 | 17.37 ± 0.32 | 15.5754 ± 0.0032 | 16.83 ± 0.23 |
| 0.19 | 16.4 ± 1.1 | 1.714 ± 0.045 | 12.4976 ± 0.0041 | 2.22 ± 0.40 | 0.274 ± 0.056 | 19.9 ± 1.2 | 15.5535 ± 0.0092 | 18.60 ± 0.85 |
| 0.38 | 21.8 ± 1.6 | 1.980 ± 0.089 | 12.549 ± 0.027 | 3.96 ± 0.46 | 0.61 ± 0.11 | 26.3 ± 1.9 | 15.5337 ± 0.0076 | 23.1 ± 1.3 |
| 0.57 | 28.9 ± 1.9 | 2.43 ± 0.14 | 12.758 ± 0.086 | 5.88 ± 0.49 | 1.20 ± 0.18 | 34.6 ± 2.2 | 15.648 ± 0.062 | 28.9 ± 1.6 |
| 0.76 | 36.9 ± 2.1 | 3.08 ± 0.19 | 13.30 ± 0.19 | 7.93 ± 0.53 | 2.09 ± 0.26 | 44.1 ± 2.5 | 16.10 ± 0.18 | 35.7 ± 1.8 |
| 0.95 | 45.6 ± 2.2 | 3.95 ± 0.24 | 14.39 ± 0.36 | 10.16 ± 0.58 | 3.35 ± 0.36 | 54.4 ± 2.7 | 17.15 ± 0.36 | 43.2 ± 2.0 |
| 1.14 | 55.0 ± 2.4 | 5.07 ± 0.31 | 16.30 ± 0.60 | 12.62 ± 0.64 | 5.06 ± 0.49 | 65.6 ± 2.9 | 19.10 ± 0.62 | 51.6 ± 2.2 |
| 1.33 | 65.2 ± 2.6 | 6.46 ± 0.39 | 19.32 ± 0.91 | 15.41 ± 0.74 | 7.36 ± 0.65 | 77.6 ± 3.1 | 22.30 ± 0.98 | 61.0 ± 2.5 |
| 1.52 | 76.3 ± 2.9 | 8.23 ± 0.50 | 23.8 ± 1.3 | 18.68 ± 0.89 | 10.40 ± 0.86 | 90.7 ± 3.4 | 27.2 ± 1.4 | 71.6 ± 2.8 |
| 1.71 | 88.5 ± 3.2 | 10.58 ± 0.69 | 30.2 ± 1.9 | 22.7 ± 1.1 | 14.4 ± 1.1 | 104.9 ± 3.7 | 34.1 ± 2.0 | 83.7 ± 3.2 |
| 1.90 | 102.0 ± 3.5 | 14.1 ± 1.1 | 39.0 ± 2.5 | 27.8 ± 1.5 | 19.9 ± 1.6 | 120.5 ± 4.0 | 43.8 ± 2.8 | 97.5 ± 3.6 |

**sigma_pos = 127 nm (thermisch + technisch).** Gilt fuer die harten Groessen, deren
Region keine thermische Mittelung enthaelt.

| r (µm) | U_h Mittel ± Streuung (%) | eta_h^circ Mittel ± Streuung (%) | eta_h^box Mittel ± Streuung (%) | U_w Mittel ± Streuung (%) | eta_w Mittel ± Streuung (%) | U_c Mittel ± Streuung (%) | eta_c Mittel ± Streuung (%) | J Mittel ± Streuung (%) |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 15.8 ± 1.8 | 1.694 ± 0.079 | 12.4972 ± 0.0090 | 1.96 ± 0.69 | 0.250 ± 0.098 | 19.3 ± 2.1 | 15.559 ± 0.016 | 18.2 ± 1.4 |
| 0.19 | 17.7 ± 2.9 | 1.78 ± 0.14 | 12.509 ± 0.029 | 2.63 ± 0.97 | 0.36 ± 0.18 | 21.5 ± 3.4 | 15.546 ± 0.020 | 19.7 ± 2.4 |
| 0.38 | 22.7 ± 4.2 | 2.05 ± 0.26 | 12.58 ± 0.11 | 4.2 ± 1.2 | 0.70 ± 0.33 | 27.4 ± 5.0 | 15.553 ± 0.062 | 23.8 ± 3.5 |
| 0.57 | 29.6 ± 5.0 | 2.51 ± 0.38 | 12.83 ± 0.28 | 6.0 ± 1.3 | 1.30 ± 0.51 | 35.4 ± 6.0 | 15.71 ± 0.23 | 29.5 ± 4.2 |
| 0.76 | 37.4 ± 5.6 | 3.16 ± 0.52 | 13.41 ± 0.58 | 8.1 ± 1.4 | 2.20 ± 0.73 | 44.8 ± 6.6 | 16.21 ± 0.54 | 36.2 ± 4.8 |
| 0.95 | 46.1 ± 6.1 | 4.04 ± 0.67 | 14.6 ± 1.0 | 10.3 ± 1.6 | 3.5 ± 1.0 | 55.0 ± 7.2 | 17.3 ± 1.1 | 43.7 ± 5.3 |
| 1.14 | 55.4 ± 6.6 | 5.17 ± 0.85 | 16.5 ± 1.7 | 12.8 ± 1.8 | 5.2 ± 1.4 | 66.1 ± 7.8 | 19.3 ± 1.8 | 52.1 ± 6.0 |
| 1.33 | 65.6 ± 7.1 | 6.6 ± 1.1 | 19.6 ± 2.6 | 15.6 ± 2.0 | 7.6 ± 1.8 | 78.1 ± 8.4 | 22.6 ± 2.7 | 61.5 ± 6.7 |
| 1.52 | 76.8 ± 7.8 | 8.4 ± 1.4 | 24.2 ± 3.7 | 18.9 ± 2.5 | 10.7 ± 2.4 | 91.2 ± 9.2 | 27.6 ± 4.0 | 72.1 ± 7.6 |
| 1.71 | 89.0 ± 8.6 | 10.9 ± 2.0 | 30.7 ± 5.2 | 23.0 ± 3.1 | 14.8 ± 3.3 | 105.5 ± 10.0 | 34.6 ± 5.6 | 84.2 ± 8.7 |
| 1.90 | 102.5 ± 9.4 | 14.7 ± 3.5 | 39.7 ± 7.1 | 28.4 ± 4.3 | 20.6 ± 4.9 | 121 ± 11 | 44.4 ± 7.6 | 98.0 ± 9.8 |

Die Werte fuer sigma_pos stammen aus dem AtomOffset-Report vom 10.09.: thermisch 118 nm
am unguenstigen Rand (19 µK, 58.2 kHz), technisch 47 nm nach Seubert et al.
(arXiv:2502.13560).

## Zum Zitieren

Fuer r = 0 (Atom auf der Site):

- U_h = 13.8 +1.1/−1.0 %
- U_w = 1.06 ± 0.17 %
- eta_h^box = 12.49 +0.04/−0.02 % (zzgl. bis +2.7 pp durch den zweiten Nachbarkranz)
- eta_w = 0.15 +0.47/−0.03 % (zzgl. bis +0.4 pp)
- J = 16.6 ± 0.9 % (zzgl. bis +1.0 pp)

Fuer einen Shift r kommt die Tabelle *Werte entlang des Shifts* dazu, und mit dem
Shift-Fehler die Tabelle *Unsicherheit des Shifts selbst*.
