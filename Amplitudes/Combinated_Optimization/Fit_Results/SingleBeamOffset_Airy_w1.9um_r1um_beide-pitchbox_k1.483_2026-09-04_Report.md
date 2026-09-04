# SingleBeamOffset - Atomposition bei festem Waist, 2026-09-04

Fester Waist, das ATOM wandert. Gefahren wird der Betrag r des Versatzes
gegen die Site-Mitte, einmal je Richtung.

- Waist: **1.9000 µm** (win_input = 0.7043 mm)
- Versatz: 0 .. 1.9000 µm  (= 1.000 x Waist)
- Stuetzstellen: 61 je Richtung
- Richtungen: senkrecht, diagonal

Eine waagerechte Richtung fehlt mit Absicht: das Strahlprofil ist
rotationssymmetrisch, waagerecht ist dasselbe wie senkrecht. Diagonal ist
es nicht - nicht wegen des Strahls, sondern wegen der Nachbar-Sites: die
liegen auf einem Quadratgitter, und in der Diagonale ist die naechste Site
sqrt(2) mal weiter weg. Genau das trennt die beiden Richtungen im
Crosstalk.

Die harte Region WANDERT MIT dem Atom: der Kreis (und, wenn gewaehlt, das Pitch-Quadrat) sitzt an der jeweiligen Atomposition. Das ist hier die sinnvolle Lesart - die Atomposition ist die abgefahrene Groesse, und die harte Region fragt, wie es DORT aussieht.

Die atom-gewichteten Groessen aendern sich in jedem Fall: das Atom sitzt im
Strahl woanders und sieht damit eine andere Intensitaet und ein anderes
Verhaeltnis zum Nachbarlicht.

Bei r = 0 stimmen alle Werte mit dem Waist-Sweep an diesem Waist ueberein.

## Ergebnis je Groesse

| Groesse | bei r = 0 | am Ende | Verlauf | Minimum |
|---|---|---|---|---|
| U_h (hart, Kreis) - senkrecht | 13.85 % | 101.9 % | steigt | 13.85 % bei r = 0.0000 µm (Rand) |
| eta_h^circ (hart, Kreis) - senkrecht | 1.615 % | 14 % | steigt | 1.615 % bei r = 0.0000 µm (Rand) |
| eta_h^box (hart, Pitch-Quadrat) - senkrecht | 12.49 % | 38.93 % | hat ein Minimum | 12.49 % bei r = 0.0317 µm |
| U_w (atom-gewichtet) - senkrecht | 1.064 % | 27.76 % | steigt | 1.064 % bei r = 0.0000 µm (Rand) |
| eta_w (atom-gewichtet) - senkrecht | 0.1523 % | 19.8 % | steigt | 0.1523 % bei r = 0.0000 µm (Rand) |
| U_c (Penalty) - senkrecht | 17.04 % | 120.4 % | steigt | 17.04 % bei r = 0.0000 µm (Rand) |
| eta_c (Penalty) - senkrecht | 15.58 % | 43.71 % | hat ein Minimum | 15.53 % bei r = 0.3483 µm |
| J (Score) - senkrecht | 16.6 % | 97.39 % | steigt | 16.6 % bei r = 0.0000 µm (Rand) |
| U_h (hart, Kreis) - diagonal | 13.85 % | 101.9 % | steigt | 13.85 % bei r = 0.0000 µm (Rand) |
| eta_h^circ (hart, Kreis) - diagonal | 1.615 % | 17.17 % | steigt | 1.615 % bei r = 0.0000 µm (Rand) |
| eta_h^box (hart, Pitch-Quadrat) - diagonal | 12.49 % | 27.88 % | hat ein Minimum | 12.49 % bei r = 0.0633 µm |
| U_w (atom-gewichtet) - diagonal | 1.064 % | 27.76 % | steigt | 1.064 % bei r = 0.0000 µm (Rand) |
| eta_w (atom-gewichtet) - diagonal | 0.1523 % | 24.39 % | steigt | 0.1523 % bei r = 0.0000 µm (Rand) |
| U_c (Penalty) - diagonal | 17.04 % | 120.4 % | steigt | 17.04 % bei r = 0.0000 µm (Rand) |
| eta_c (Penalty) - diagonal | 15.58 % | 28.75 % | hat ein Minimum | 15.49 % bei r = 0.4750 µm |
| J (Score) - diagonal | 16.6 % | 92.9 % | steigt | 16.6 % bei r = 0.0000 µm (Rand) |

"Rand" heisst: das Minimum liegt am Ende des abgefahrenen Bereichs.

## Werte

| r (µm) | r/w | U_h (v) (%) | eta_h^circ (v) (%) | eta_h^box (v) (%) | U_w (v) (%) | eta_w (v) (%) | U_c (v) (%) | eta_c (v) (%) | J (v) (%) | U_h (d) (%) | eta_h^circ (d) (%) | eta_h^box (d) (%) | U_w (d) (%) | eta_w (d) (%) | U_c (d) (%) | eta_c (d) (%) | J (d) (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.000 | 13.85 | 1.615 | 12.49 | 1.064 | 0.1523 | 17.04 | 15.58 | 16.6 | 13.85 | 1.615 | 12.49 | 1.064 | 0.1523 | 17.04 | 15.58 | 16.6 |
| 0.1900 | 0.100 | 16.14 | 1.703 | 12.5 | 2.157 | 0.2607 | 19.64 | 15.56 | 18.41 | 16.14 | 1.703 | 12.49 | 2.157 | 0.2604 | 19.64 | 15.55 | 18.41 |
| 0.3800 | 0.200 | 21.68 | 1.969 | 12.54 | 3.931 | 0.5947 | 26.11 | 15.53 | 22.94 | 21.68 | 1.969 | 12.52 | 3.931 | 0.5902 | 26.11 | 15.5 | 22.93 |
| 0.5700 | 0.300 | 28.79 | 2.42 | 12.75 | 5.856 | 1.182 | 34.52 | 15.64 | 28.86 | 28.79 | 2.426 | 12.64 | 5.856 | 1.16 | 34.52 | 15.51 | 28.82 |
| 0.7600 | 0.400 | 36.8 | 3.071 | 13.28 | 7.913 | 2.069 | 44.02 | 16.08 | 35.64 | 36.8 | 3.093 | 12.94 | 7.913 | 2.005 | 44.02 | 15.68 | 35.51 |
| 0.9500 | 0.500 | 45.5 | 3.939 | 14.37 | 10.14 | 3.324 | 54.34 | 17.13 | 43.18 | 45.5 | 4.009 | 13.57 | 10.14 | 3.188 | 54.34 | 16.17 | 42.89 |
| 1.1400 | 0.600 | 54.9 | 5.05 | 16.26 | 12.59 | 5.038 | 65.47 | 19.07 | 51.55 | 54.9 | 5.235 | 14.68 | 12.59 | 4.816 | 65.47 | 17.14 | 50.97 |
| 1.3300 | 0.700 | 65.08 | 6.444 | 19.27 | 15.38 | 7.331 | 77.5 | 22.25 | 60.93 | 65.08 | 6.88 | 16.44 | 15.38 | 7.087 | 77.5 | 18.78 | 59.89 |
| 1.5200 | 0.800 | 76.19 | 8.204 | 23.74 | 18.64 | 10.36 | 90.58 | 27.09 | 71.53 | 76.19 | 9.133 | 19.06 | 18.64 | 10.39 | 90.57 | 21.23 | 69.77 |
| 1.7100 | 0.900 | 88.4 | 10.54 | 30.11 | 22.62 | 14.37 | 104.8 | 34.05 | 83.61 | 88.4 | 12.34 | 22.78 | 22.62 | 15.52 | 104.8 | 24.6 | 80.77 |
| 1.9000 | 1.000 | 101.9 | 14 | 38.93 | 27.76 | 19.8 | 120.4 | 43.71 | 97.39 | 101.9 | 17.17 | 27.88 | 27.76 | 24.39 | 120.4 | 28.75 | 92.9 |

## Parameter

| Groesse | Wert |
|---|---|
| Profil | airy |
| airy_scale_factor | 1.48295 |
| Waist | 1.9000 µm |
| Kreisradius (hart) | 1.0000 µm |
| Crosstalk-Region (hart) | Kreis mit Radius 1 µm UND Pitch-Quadrat, Seitenlaenge 5.288 µm (in eta_c/J geht der Pitch-Quadrat, Seitenlaenge 5.288 µm ein) |
| harte Region folgt dem Atom | ja |
| pitch | 5.2880 µm |
| Nachbar-Sites | 8 |
| sigma_atom | 107.64 nm (T = 17.00 µK, nu_r = 60.40 kHz) |
| alpha / combo_lambda | 0.700 / 0.750 |
