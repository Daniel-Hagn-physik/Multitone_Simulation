# AtomOffset - Atom gegen das Multitone-Profil verschoben, 2026-09-10

Fester Parametersatz, das ATOM wandert radial von der Site-Mitte nach
aussen. Ausgewertet wie im Penalty-Fall: hart, atom-gewichtet und die
Penalty-Kombination daraus.

| Groesse | Wert |
|---|---|
| Toene | 3 x 4 (fest) |
| Linsen | f1 = 75 mm, f2 = 750 mm (fest), fLO = 52.88 mm |
| Waist (Atomebene) | 1.0400 µm |
| width | 0.3700 MHz = **d = 2.3370 µm** in der Atomebene (Spannweite des Tonarrays) |
| r_x / r_y | 0.9700 / 1.1600 |
| Profil | airy, airy_scale_factor = 1.48295 |
| Kohaerenz | an (entartete Paare: 1) |
| pitch | 5.2880 µm |
| sigma_atom (Gewicht) | 107.64 nm (T = 17.00 µK, nu_r = 60.40 kHz) |
| alpha / combo_lambda | 0.700 / 0.750 |
| Richtungen | horizontal, vertikal, diagonal |

## Was sich bewegt und was steht

Bewegt wird das Atom - und mit ihm ALLE Regionen: das Ton-Quadrat der
harten Uniformity (Seitenlaenge 2.3370 µm), das Pitch-Quadrat des
harten Crosstalks und das lokale Sub-Gitter samt Gauss-Gewichtung der
atom-gewichteten Metriken. Das Lichtfeld (12 Spots der Site, 8 Nachbarkopien
im Abstand pitch) steht. Das ist exakt dasselbe wie eine gemeinsame Drift
des ganzen Raman-Profils um -r gegen ein ruhendes Atom.

Richtungen: horizontal (r, 0) laeuft entlang der 3-Ton-Achse,
vertikal (0, r) entlang der 4-Ton-Achse, diagonal (r, r)/sqrt2 auf
eine Ecke zu, antidiagonal (-r, r)/sqrt2 auf die andere Ecke.

## Numerik und Abgleich bei r = 0

Die harten Metriken laufen NICHT ueber das globale Gitter des Optimierers:
dort sind die Regionen Masken, deren Rand bei Verschiebungen um Bruchteile
eines Pixels um ganze Pixel springt. Statt dessen liegt ein eigenes Gitter
auf der Region (201 x 201 Zellmitten, wandert mit dem Atom),
die Kurven sind dadurch glatt in r. Die gewichteten Metriken kommen
unveraendert aus dem Optimierer (atom_offset_x/y).

| Groesse | dieses Skript (Gitter auf der Region) | Optimierer (globales Gitter, n_grid = 1000) | Differenz |
|---|---|---|---|
| U_h (hart, Ton-Quadrat) | 3.5704 % | 3.5450 % | +0.0254 pp |
| eta_h (hart, Pitch-Quadrat) | 8.3869 % | 8.3718 % | +0.0151 pp |
| U_w (atom-gewichtet) | 0.4292 % | 0.4292 % | +0.0000 pp |
| eta_w (atom-gewichtet) | 1.3981 % | 1.3981 % | +0.0000 pp |

Die Differenz der harten Groessen ist der Pixelrand des globalen Gitters
(dieselbe Groessenordnung wie das in claude/combinated_optimization.md,
Nachtrag 6, gemessene Saegezahn-Rauschen).

## Positionsschwankung des Atoms - Abschaetzung

| Mechanismus | Eingabe | Hebel | Beitrag (1 sigma) | geht ein | Quelle/Hinweis |
|---|---|---|---|---|---|
| thermische Ortsbreite (nominal) | T = 17 µK, nu_r = 60.4 kHz | sigma = sqrt(hbar/(2 m w) coth(hbar w/2kT)) | 108 nm | nein | steckt als Gewicht W bereits in U_w und eta_w |
| thermische Ortsbreite (unguenstiger Rand) | T = 19 µK, nu_r = 58.2 kHz | wie oben | 118 nm | thermisch | geht in die Summe ein |
| Aenderung von sigma durch Fallenleistung | dP/P = 1 % | sigma ~ P^(-1/2) (bei festem T) | 0.542 nm | nein | keine Verschiebung, nur Breite |
| Durchhang durch Schwerkraft g/omega^2 | nu_r = 60.4 kHz | g / omega^2 | 0.0681 nm | nein | nur falls die Falle quer zur Schwerkraft steht; vernachlaessigbar |
| Relativposition Falle - Raman (gemessen, gesamt) | 47 nm | 1 | 47 nm | technisch | Groessenordnung: 47(5) nm Positionsstreuung je Site fuer Atome in AOD-Tweezern, Seubert et al., arXiv:2502.13560 (inkl. Detektionsfehler, obere Schranke; anderer Aufbau). Auf 0 setzen, wenn die Einzelbeitraege darunter eingetragen werden. |
| Raman-Strahllage vor dem AOD | 0 µrad | f1 fLO/f2 = 5.288 mm (5.288 nm/µrad) | 0 nm | technisch | Winkeljitter des Strahls, der in den AOD laeuft |
| Raman-Strahllage vor dem Objektiv | 0 µrad | fLO = 52.88 mm (52.88 nm/µrad) | 0 nm | technisch | Winkeljitter hinter dem Teleskop - der groesste Hebel (fLO) |
| AOD-Schallgeschwindigkeit | 0 ppm | f1 fLO/f2 * tan(theta_c) = 0.6358 mm (0.6358 nm/ppm, f_c = 100.2 MHz) | 0 nm | technisch | = Temperaturkoeffizient der Schallgeschwindigkeit x Temperaturschwankung des Kristalls (Datenblatt/Messung) |
| RF-Frequenzfehler | 0 Hz | 6.316 pm/Hz | 0 nm | technisch | Frequenzfehler des RF-Generators |
| Fallenposition (Tweezer-Strahllage) | 0 nm | 1 | 0 nm | technisch | Schwankung der Fallenposition in der Atomebene |
| Pitch-Unsicherheit an Site n | dp = 30 nm, n = 0 | n | 0 nm | technisch | Messung pitch = 5.16(3) µm; wirkt n-fach an der Site n |

- thermisch (unguenstiger Rand): **118.1 nm** (nominal 107.6 nm; Nullpunktsbreite 31.4 nm, mittlere Besetzung n = 5.38)
- technisch (quadratisch addiert): **47.0 nm**
- gesamt: **127.1 nm**
- benutzt fuer Plot 2 und Statistik (gesamt): **sigma_pos = 127.1 nm**, Plot 2 bis 1 sigma_pos = 127.1 nm

Alle Beitraege sind 1 sigma JE ACHSE. Fuer eine isotrope 2D-Gaussverteilung
liegt der Abstand r von der Site-Mitte mit 39.3 % Wahrscheinlichkeit unter
1 sigma, mit 86.5 % unter 2 sigma und mit 98.9 % unter 3 sigma
(Rayleigh-Verteilung). Bis 1 sigma sind es 39.3 %.

Die Hebelarme folgen aus der Optik des Optimierers
(r = f1 fLO/f2 tan(theta), theta = theta_max (f - offset)/f_band); die
Formeln stehen im Kopf von lib/position_noise.py. Technische Eingaben, die
auf 0 stehen, sind nicht gemessen - der Hebel daneben sagt, wieviel ein
Mikroradiant, ein ppm oder ein Hz ausmachen wuerde.

## Versatz bis 0.125 x width (292.1 nm)

- Bereich: 0 .. 0.2921 µm = 0.1250 d, 41 Stuetzstellen je Richtung
- Rechenzeit: 144.4 s

### Ergebnis je Groesse

| Groesse | Richtung | bei r = 0 | am Ende | Aenderung | Verlauf |
|---|---|---|---|---|---|
| U_h (hart, Ton-Quadrat) | horizontal | 3.57 % | 5.278 % | +1.71 pp | steigt |
| U_h (hart, Ton-Quadrat) | vertikal | 3.57 % | 7.665 % | +4.09 pp | steigt |
| U_h (hart, Ton-Quadrat) | diagonal | 3.57 % | 6.492 % | +2.92 pp | steigt |
| eta_h (hart, Pitch-Quadrat) | horizontal | 8.387 % | 8.46 % | +0.0727 pp | steigt |
| eta_h (hart, Pitch-Quadrat) | vertikal | 8.387 % | 8.481 % | +0.0937 pp | steigt |
| eta_h (hart, Pitch-Quadrat) | diagonal | 8.387 % | 8.444 % | +0.0571 pp | steigt |
| U_w (atom-gewichtet) | horizontal | 0.4292 % | 1.009 % | +0.58 pp | steigt |
| U_w (atom-gewichtet) | vertikal | 0.4292 % | 0.9754 % | +0.546 pp | steigt |
| U_w (atom-gewichtet) | diagonal | 0.4292 % | 1.087 % | +0.658 pp | steigt |
| eta_w (atom-gewichtet) | horizontal | 1.398 % | 2.057 % | +0.659 pp | steigt |
| eta_w (atom-gewichtet) | vertikal | 1.398 % | 1.7 % | +0.302 pp | steigt |
| eta_w (atom-gewichtet) | diagonal | 1.398 % | 1.911 % | +0.513 pp | steigt |
| U_c (Penalty) | horizontal | 4.356 % | 6.345 % | +1.99 pp | steigt |
| U_c (Penalty) | vertikal | 4.356 % | 9.338 % | +4.98 pp | steigt |
| U_c (Penalty) | diagonal | 4.356 % | 7.843 % | +3.49 pp | steigt |
| eta_c (Penalty) | horizontal | 10.13 % | 10.06 % | -0.0739 pp | hat ein Minimum |
| eta_c (Penalty) | vertikal | 10.13 % | 10.18 % | +0.0416 pp | hat ein Minimum |
| eta_c (Penalty) | diagonal | 10.13 % | 10.08 % | -0.0569 pp | faellt |
| J (Score) | horizontal | 6.089 % | 7.459 % | +1.37 pp | steigt |
| J (Score) | vertikal | 6.089 % | 9.589 % | +3.5 pp | steigt |
| J (Score) | diagonal | 6.089 % | 8.513 % | +2.42 pp | steigt |

### Werte bei den Abstaenden der Positionsschwankung

| Groesse | Richtung | r = 0 | 1 sigma_pos = 127 nm | 2 sigma_pos = 254 nm |
|---|---|---|---|---|
| U_h (hart, Ton-Quadrat) | horizontal | 3.57 % | 3.822 % (+0.252 pp) | 4.793 % (+1.22 pp) |
| U_h (hart, Ton-Quadrat) | vertikal | 3.57 % | 4.488 % (+0.918 pp) | 6.783 % (+3.21 pp) |
| U_h (hart, Ton-Quadrat) | diagonal | 3.57 % | 4.203 % (+0.633 pp) | 5.85 % (+2.28 pp) |
| eta_h (hart, Pitch-Quadrat) | horizontal | 8.387 % | 8.391 % (+0.00411 pp) | 8.431 % (+0.0437 pp) |
| eta_h (hart, Pitch-Quadrat) | vertikal | 8.387 % | 8.398 % (+0.0112 pp) | 8.45 % (+0.0632 pp) |
| eta_h (hart, Pitch-Quadrat) | diagonal | 8.387 % | 8.394 % (+0.00662 pp) | 8.425 % (+0.0382 pp) |
| U_w (atom-gewichtet) | horizontal | 0.4292 % | 0.6357 % (+0.206 pp) | 0.9383 % (+0.509 pp) |
| U_w (atom-gewichtet) | vertikal | 0.4292 % | 0.625 % (+0.196 pp) | 0.9116 % (+0.482 pp) |
| U_w (atom-gewichtet) | diagonal | 0.4292 % | 0.6411 % (+0.212 pp) | 0.9913 % (+0.562 pp) |
| eta_w (atom-gewichtet) | horizontal | 1.398 % | 1.537 % (+0.139 pp) | 1.913 % (+0.515 pp) |
| eta_w (atom-gewichtet) | vertikal | 1.398 % | 1.462 % (+0.0636 pp) | 1.634 % (+0.236 pp) |
| eta_w (atom-gewichtet) | diagonal | 1.398 % | 1.503 % (+0.105 pp) | 1.796 % (+0.398 pp) |
| U_c (Penalty) | horizontal | 4.356 % | 4.619 % (+0.263 pp) | 5.756 % (+1.4 pp) |
| U_c (Penalty) | vertikal | 4.356 % | 5.454 % (+1.1 pp) | 8.251 % (+3.9 pp) |
| U_c (Penalty) | diagonal | 4.356 % | 5.094 % (+0.738 pp) | 7.065 % (+2.71 pp) |
| eta_c (Penalty) | horizontal | 10.13 % | 10.1 % (-0.0295 pp) | 10.06 % (-0.0741 pp) |
| eta_c (Penalty) | vertikal | 10.13 % | 10.13 % (-0.00195 pp) | 10.15 % (+0.0199 pp) |
| eta_c (Penalty) | diagonal | 10.13 % | 10.12 % (-0.0181 pp) | 10.08 % (-0.0517 pp) |
| J (Score) | horizontal | 6.089 % | 6.264 % (+0.175 pp) | 7.047 % (+0.958 pp) |
| J (Score) | vertikal | 6.089 % | 6.857 % (+0.768 pp) | 8.822 % (+2.73 pp) |
| J (Score) | diagonal | 6.089 % | 6.6 % (+0.511 pp) | 7.97 % (+1.88 pp) |

### Werte

**horizontal**

| r (µm) | r/d | U_h (%) | eta_h (%) | U_w (%) | eta_w (%) | U_c (%) | eta_c (%) | J (%) |
|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.0000 | 3.57 | 8.387 | 0.4292 | 1.398 | 4.356 | 10.13 | 6.089 |
| 0.0292 | 0.0125 | 3.583 | 8.387 | 0.4436 | 1.406 | 4.367 | 10.13 | 6.097 |
| 0.0584 | 0.0250 | 3.62 | 8.387 | 0.4834 | 1.428 | 4.404 | 10.13 | 6.121 |
| 0.0876 | 0.0375 | 3.684 | 8.388 | 0.5415 | 1.465 | 4.47 | 10.12 | 6.165 |
| 0.1169 | 0.0500 | 3.78 | 8.39 | 0.6103 | 1.516 | 4.573 | 10.11 | 6.233 |
| 0.1461 | 0.0625 | 3.912 | 8.393 | 0.6837 | 1.58 | 4.719 | 10.1 | 6.332 |
| 0.1753 | 0.0750 | 4.084 | 8.399 | 0.7576 | 1.656 | 4.915 | 10.08 | 6.466 |
| 0.2045 | 0.0875 | 4.302 | 8.407 | 0.8288 | 1.743 | 5.171 | 10.07 | 6.641 |
| 0.2337 | 0.1000 | 4.572 | 8.419 | 0.8954 | 1.841 | 5.491 | 10.06 | 6.863 |
| 0.2629 | 0.1125 | 4.896 | 8.436 | 0.9557 | 1.946 | 5.881 | 10.06 | 7.134 |
| 0.2921 | 0.1250 | 5.278 | 8.46 | 1.009 | 2.057 | 6.345 | 10.06 | 7.459 |

**vertikal**

| r (µm) | r/d | U_h (%) | eta_h (%) | U_w (%) | eta_w (%) | U_c (%) | eta_c (%) | J (%) |
|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.0000 | 3.57 | 8.387 | 0.4292 | 1.398 | 4.356 | 10.13 | 6.089 |
| 0.0292 | 0.0125 | 3.623 | 8.387 | 0.4427 | 1.402 | 4.417 | 10.13 | 6.132 |
| 0.0584 | 0.0250 | 3.776 | 8.389 | 0.4803 | 1.412 | 4.6 | 10.13 | 6.26 |
| 0.0876 | 0.0375 | 4.023 | 8.392 | 0.5353 | 1.429 | 4.895 | 10.13 | 6.466 |
| 0.1169 | 0.0500 | 4.354 | 8.396 | 0.6007 | 1.452 | 5.292 | 10.13 | 6.744 |
| 0.1461 | 0.0625 | 4.759 | 8.402 | 0.6708 | 1.481 | 5.78 | 10.13 | 7.086 |
| 0.1753 | 0.0750 | 5.229 | 8.411 | 0.7413 | 1.516 | 6.351 | 10.13 | 7.486 |
| 0.2045 | 0.0875 | 5.759 | 8.422 | 0.8091 | 1.556 | 6.996 | 10.14 | 7.939 |
| 0.2337 | 0.1000 | 6.343 | 8.437 | 0.8718 | 1.601 | 7.711 | 10.15 | 8.442 |
| 0.2629 | 0.1125 | 6.979 | 8.456 | 0.9276 | 1.649 | 8.492 | 10.16 | 8.992 |
| 0.2921 | 0.1250 | 7.665 | 8.481 | 0.9754 | 1.7 | 9.338 | 10.18 | 9.589 |

**diagonal**

| r (µm) | r/d | U_h (%) | eta_h (%) | U_w (%) | eta_w (%) | U_c (%) | eta_c (%) | J (%) |
|---|---|---|---|---|---|---|---|---|
| 0.0000 | 0.0000 | 3.57 | 8.387 | 0.4292 | 1.398 | 4.356 | 10.13 | 6.089 |
| 0.0292 | 0.0125 | 3.606 | 8.387 | 0.4435 | 1.404 | 4.396 | 10.13 | 6.117 |
| 0.0584 | 0.0250 | 3.71 | 8.388 | 0.4835 | 1.421 | 4.516 | 10.13 | 6.2 |
| 0.0876 | 0.0375 | 3.879 | 8.39 | 0.5426 | 1.449 | 4.713 | 10.13 | 6.337 |
| 0.1169 | 0.0500 | 4.109 | 8.392 | 0.6141 | 1.487 | 4.983 | 10.12 | 6.523 |
| 0.1461 | 0.0625 | 4.394 | 8.396 | 0.6927 | 1.536 | 5.319 | 10.11 | 6.757 |
| 0.1753 | 0.0750 | 4.729 | 8.401 | 0.7743 | 1.595 | 5.717 | 10.1 | 7.033 |
| 0.2045 | 0.0875 | 5.109 | 8.408 | 0.8564 | 1.663 | 6.172 | 10.09 | 7.349 |
| 0.2337 | 0.1000 | 5.531 | 8.417 | 0.9368 | 1.738 | 6.68 | 10.09 | 7.702 |
| 0.2629 | 0.1125 | 5.993 | 8.429 | 1.014 | 1.822 | 7.237 | 10.08 | 8.09 |
| 0.2921 | 0.1250 | 6.492 | 8.444 | 1.087 | 1.911 | 7.843 | 10.08 | 8.513 |

## Statistik ueber die Positionsverteilung

Das Atom sitzt isotrop gaussverteilt mit sigma_pos = 127.1 nm je
Achse um die Site-Mitte. Mittelwert und Streuung jeder Groesse aus einer
Gauss-Hermite-Quadratur mit 7 x 7 = 49 Positionen
(deterministisch, keine Zufallszahlen). "Streuung" ist die
Schuss-zu-Schuss-Schwankung der Groesse, wenn jeder Schuss eine neue
Position zieht.

| Groesse | Atom auf der Site | Mittelwert | Streuung (1 sigma) | Mittel - Site | Spanne der Knoten |
|---|---|---|---|---|---|
| U_h (hart, Ton-Quadrat) | 3.57 % | 4.669 % | 1.14 pp | +1.1 pp | 3.57 .. 16.23 % |
| eta_h (hart, Pitch-Quadrat) | 8.387 % | 8.41 % | 0.0365 pp | +0.0228 pp | 8.387 .. 9.274 % |
| U_w (atom-gewichtet) | 0.4292 % | 0.7066 % | 0.188 pp | +0.277 pp | 0.4292 .. 1.709 % |
| eta_w (atom-gewichtet) | 1.398 % | 1.589 % | 0.188 pp | +0.191 pp | 1.398 .. 3.174 % |
| U_c (Penalty) | 4.356 % | 5.66 % | 1.39 pp | +1.3 pp | 4.356 .. 19.96 % |
| eta_c (Penalty) | 10.13 % | 10.11 % | 0.029 pp | -0.0192 pp | 10.06 .. 10.8 % |
| J (Score) | 6.089 % | 6.996 % | 0.976 pp | +0.907 pp | 6.089 .. 17.21 % |

Genauigkeit: U_w ist ueber der Verteilung am staerksten nichtlinear (steigt
steil an und flacht dann ab), dort konvergiert die Quadratur langsamer als
bei den uebrigen Groessen. Am Standard-Arbeitspunkt gemessen:
7 -> 11 Knoten aendert die Streuung von U_w um 0.02 pp, die aller anderen
Groessen um weniger als 0.001 pp.

Hinweis zu den atom-gewichteten Groessen: in U_w und eta_w steckt die
thermische Ortsverteilung bereits als Gewicht. Enthaelt sigma_pos den
thermischen Anteil ("gesamt"/"thermisch"), ist er dort doppelt gezaehlt -
fuer diese beiden ist "technisch" die saubere Wahl.

## Symmetrie (bei r = 0.2921 µm, Werte in %)

| Groesse | +x | -x | +y | -y | diagonal (+,+) | diagonal (-,-) | antidiagonal (-,+) | antidiagonal (+,-) |
|---|---|---|---|---|---|---|---|---|
| U_h | 5.278 | 5.278 | 7.665 | 7.665 | 6.492 | 6.492 | 6.114 | 6.114 |
| eta_h | 8.46 | 8.46 | 8.481 | 8.481 | 8.444 | 8.444 | 8.445 | 8.445 |
| U_w | 1.009 | 1.009 | 0.9754 | 0.9754 | 1.087 | 1.087 | 0.9423 | 0.9423 |
| eta_w | 2.057 | 2.057 | 1.7 | 1.7 | 1.911 | 1.911 | 1.885 | 1.885 |
| U_c | 6.345 | 6.345 | 9.338 | 9.338 | 7.843 | 7.843 | 7.407 | 7.407 |
| eta_c | 10.06 | 10.06 | 10.18 | 10.18 | 10.08 | 10.08 | 10.08 | 10.08 |
| J | 7.459 | 7.459 | 9.589 | 9.589 | 8.513 | 8.513 | 8.21 | 8.21 |

+r und -r stimmen je Achse ueberein, ebenso gegenueberliegende Punkte
einer Diagonale - die Richtung mit positivem Vorzeichen steht also fuer
beide.
Diagonale und Antidiagonale sind dagegen NICHT gleich, sobald Kohaerenz
an ist: die frequenzentarteten Eckspots (links oben, rechts unten)
interferieren statisch, das Muster ist in dieser Richtung ein anderes.

## Dateien

- `atom_offset_N3x4_Airy_w1.04um_width0.37MHz_rx0.97_ry1.16_k1.483.pkl`
- `AtomOffset_N3x4_Airy_w1.04um_width0.37MHz_rx0.97_ry1.16_k1.483_bis0.125width_2026-09-10_hard.pdf`
- `AtomOffset_N3x4_Airy_w1.04um_width0.37MHz_rx0.97_ry1.16_k1.483_bis0.125width_2026-09-10_weighted.pdf`
- `AtomOffset_N3x4_Airy_w1.04um_width0.37MHz_rx0.97_ry1.16_k1.483_bis0.125width_2026-09-10_penalty.pdf`
