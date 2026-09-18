# SingleBeam - ein Strahl, Airy, 2026-09-04

Ein EINZELNER Strahl, kein Tonarray. Abgefahren wird die einzige freie
Groesse, die dabei bleibt: der Waist.

Weder `width` noch die Amplituden-Verhaeltnisse r_x/r_y kommen vor. `width`
spannt bei einem Ton nichts auf - das Ton-Quadrat der harten
Uniformity-Region ist nicht definiert -, und ein Aussen/Innen-Verhaeltnis
braucht mindestens zwei Toene je Achse.

## Was gerechnet wurde

**Hart** - die Uniformity ueber einer Kreisregion mit Radius 1.00 µm
um die Site (Beam-Pointing-Region: das Atom kann irgendwo in diesem Kreis
sitzen), der Crosstalk ueber: **Kreis mit Radius 1 µm UND Pitch-Quadrat, Seitenlaenge 5.288 µm (in C_k/J geht der Kreis mit Radius 1 µm ein)**.

```
U_h   = std(I) / mean(I)               ueber dem Kreis
eta_h = sum(I_nachbar) / sum(I_eigen)  ueber der gewaehlten Region
```

Achtung: das Pitch-Quadrat ist eine ANDERE Flaeche als der Kreis, ueber
dem die Uniformity laeuft. Es ist die Region, die die Multitone-Skripte
fuer den Crosstalk benutzen - die Zahl ist damit direkt mit jenen Scans
vergleichbar, aber nicht mit der Uniformity daneben.


**Atom-gewichtet** - Definition unveraendert aus dem Optimierer, mit der
thermischen Ortsverteilung als Gewicht W:

```
U_w   = sqrt(<(I - <I>_W)^2>_W) / <I>_W
eta_w = sum(I_nachbar * W) / sum(I_eigen * W)
```

**Penalty-Kombination** - dieselbe Formel, die run_penalty_scan.py an jedem
Gitterpunkt minimiert:

```
U_k = 0.5*(U_h + U_w) + lambda*|U_h - U_w|
C_k = 0.5*(eta_h + eta_w) + lambda*|eta_h - eta_w|
J   = alpha*U_k + (1-alpha)*C_k
```

**Wie C_k und U_k zu lesen sind.** Solange eine der beiden Groessen
durchgehend groesser ist - hier fast immer eta_h > eta_w -, loest sich der
Betrag auf:

```
C_k = (0.5 + lambda)*eta_h - (lambda - 0.5)*eta_w
    = 1.25*eta_h - 0.25*eta_w      (bei lambda = 0.75)
```

Die KLEINERE der beiden Groessen geht also mit NEGATIVEM Vorzeichen ein.
Jedes Ringmaximum von eta_w drueckt C_k damit nach unten: Diese Dellen sind
keine Eigenschaft des Crosstalks, sondern die Aussage "hier stimmen hartes
und gewichtetes Kriterium am besten ueberein" - und genau das soll der
Penalty-Term belohnen. Bei lambda = 0.5 waere C_k exakt max(eta_h, eta_w),
darunter liegt es zwischen Mittelwert und Maximum. Fuer U_k gilt dasselbe.

I_nachbar ist die Summe der um +-pitch verschobenen Kopien desselben
Strahls: 8 Nachbar-Sites, also der Kranz direkt um die Site.
Weil das Profil translationsinvariant ist, ist das dieselbe Zahl wie "wieviel von diesem
Strahl faellt auf die Nachbar-Sites".

Nicht normiert wird bewusst: der Peak des Profils ist analytisch 1, und
beide Metriken sind gegen eine gemeinsame Skalierung von I_eigen und
I_nachbar invariant. Eine Normierung auf ein Gitter-Maximum waere hier
sogar falsch - das Maximum der Nachbar-Summe liegt ausserhalb der
ausgewerteten Region.

## Abgefahrener Bereich

- Waist nach der Linse: 0.8000 .. 2.5000 µm
- Eingangswaist vor der Linse: 0.5353 .. 1.6727 mm
- Stuetzstellen: 121

## Ergebnis je Groesse

| Groesse | Spanne | Verlauf | Minimum | bei Waist |
|---|---|---|---|---|
| U_h (hart, Kreis) | 7.879 .. 85.68 % | faellt | 7.879 % | 2.5000 µm (Rand) |
| eta_h^circ (hart, Kreis) | 0.4623 .. 6.884 % | uneindeutig | 0.4623 % | 0.8992 µm |
| eta_h^box (hart, Pitch-Quadrat) | 4.733 .. 14.18 % | uneindeutig | 4.733 % | 0.8000 µm (Rand) |
| U_w (atom-gewichtet) | 0.6163 .. 5.812 % | faellt | 0.6163 % | 2.5000 µm (Rand) |
| eta_w (atom-gewichtet) | 0.04974 .. 7.228 % | uneindeutig | 0.04974 % | 0.8425 µm |
| U_k (Penalty) | 9.695 .. 105.6 % | faellt | 9.695 % | 2.5000 µm (Rand) |
| C_k (Penalty) | 0.4843 .. 7.314 % | uneindeutig | 0.4843 % | 0.9133 µm |
| J (Score) | 8.981 .. 74.12 % | faellt | 8.981 % | 2.5000 µm (Rand) |

"Rand" heisst: das Minimum liegt am Ende des abgefahrenen Bereichs. Dann ist
es vermutlich kein Optimum, sondern nur die Grenze - weitersuchen.

## Markierter Punkt (Stern im Plot)

- Waist = 1.9000 µm (win_input = 0.7043 mm)

| Groesse | Wert |
|---|---|
| U_h | 13.85 % |
| eta_h^circ | 1.617 % |
| eta_h^box | 12.49 % |
| U_w | 1.064 % |
| eta_w | 0.155 % |
| U_k | 17.04 % |
| C_k | 1.982 % |
| J | 12.53 % |

Linear zwischen den beiden benachbarten Stuetzstellen interpoliert.
Die Kurven sind hier glatte Funktionen des Waists - anders als r_x/r_y
im Multitone-Scan, die Optimierungs-ERGEBNISSE sind; dort waere
Interpolieren irrefuehrend.

## Werte

| Waist (µm) | w_in (mm) | U_h (%) | eta_h^circ (%) | eta_h^box (%) | U_w (%) | eta_w (%) | U_k (%) | C_k (%) | J (%) |
|---|---|---|---|---|---|---|---|---|---|
| 0.8000 | 1.6727 | 85.68 | 0.4673 | 4.733 | 5.812 | 0.1427 | 105.6 | 0.5485 | 74.12 |
| 0.9700 | 1.3795 | 57.31 | 0.6095 | 6.09 | 4.002 | 0.1976 | 70.64 | 0.7125 | 49.66 |
| 1.1400 | 1.1738 | 40.58 | 0.7214 | 6.66 | 2.918 | 0.6026 | 50 | 0.7511 | 35.22 |
| 1.3100 | 1.0215 | 30.18 | 0.994 | 8.24 | 2.22 | 0.3935 | 37.17 | 1.144 | 26.36 |
| 1.4800 | 0.9042 | 23.32 | 1.451 | 10.33 | 1.745 | 0.9072 | 28.71 | 1.586 | 20.58 |
| 1.6500 | 0.8110 | 18.57 | 1.839 | 12.04 | 1.407 | 2.236 | 22.86 | 2.335 | 16.7 |
| 1.8200 | 0.7353 | 15.14 | 1.578 | 12.53 | 1.159 | 0.7237 | 18.63 | 1.791 | 13.58 |
| 1.9900 | 0.6724 | 12.58 | 2.073 | 12.48 | 0.9703 | 0.3761 | 15.49 | 2.498 | 11.59 |
| 2.1600 | 0.6195 | 10.63 | 3.961 | 12.72 | 0.8244 | 2.939 | 13.08 | 4.217 | 10.42 |
| 2.3300 | 0.5743 | 9.1 | 5.923 | 13.24 | 0.709 | 5.86 | 11.2 | 5.938 | 9.62 |
| 2.5000 | 0.5353 | 7.879 | 6.884 | 14.18 | 0.6163 | 7.228 | 9.695 | 7.314 | 8.981 |

Die vollstaendigen Kurven stehen im Datensatz (.pkl) und in den PDFs.

## Parameter

| Groesse | Wert |
|---|---|
| Profil | airy |
| airy_scale_factor | 1.48295 (first_zero_radius = Faktor * waist) |
| lambda | 795.0 nm |
| f1 / f2 / fLO | 75.00 / 750.00 / 52.88 mm |
| pitch | 5.2880 µm |
| Nachbar-Sites | 8 (1 Kranz um die Site) |
| Kreisradius (hart) | 1.0000 µm |
| Crosstalk-Region (hart) | Kreis mit Radius 1 µm UND Pitch-Quadrat, Seitenlaenge 5.288 µm (in C_k/J geht der Kreis mit Radius 1 µm ein) |
| Gitter (hart) | 401 x 401 Zellen (Mittelpunktsregel) je Region |
| sigma_atom | 107.64 nm (T = 17.00 µK, nu_r = 60.40 kHz) |
| Sub-Gitter (gewichtet) | 241 x 241 ueber +-6 sigma_atom |
| Atom-Versatz | 0.0000 / 0.0000 µm |
| alpha | 0.700 |
| combo_lambda | 0.750 |
