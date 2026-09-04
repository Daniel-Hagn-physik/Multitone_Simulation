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
sitzen), der Crosstalk ueber: **Kreis mit Radius 1 µm UND Pitch-Quadrat, Seitenlaenge 5.288 µm (in eta_c/J geht der Pitch-Quadrat, Seitenlaenge 5.288 µm ein)**.

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
U_c   = 0.5*(U_h + U_w) + lambda*|U_h - U_w|
eta_c = 0.5*(eta_h + eta_w) + lambda*|eta_h - eta_w|
J     = alpha*U_c + (1-alpha)*eta_c
```

**Wie eta_c und U_c zu lesen sind.** Solange eine der beiden Groessen
durchgehend groesser ist - hier fast immer eta_h > eta_w -, loest sich der
Betrag auf:

```
eta_c = (0.5 + lambda)*eta_h - (lambda - 0.5)*eta_w
      = 1.25*eta_h - 0.25*eta_w    (bei lambda = 0.75)
```

Die KLEINERE der beiden Groessen geht also mit NEGATIVEM Vorzeichen ein.
Jedes Ringmaximum von eta_w drueckt eta_c damit nach unten: Diese Dellen sind
keine Eigenschaft des Crosstalks, sondern die Aussage "hier stimmen hartes
und gewichtetes Kriterium am besten ueberein" - und genau das soll der
Penalty-Term belohnen. Bei lambda = 0.5 waere eta_c exakt max(eta_h, eta_w),
darunter liegt es zwischen Mittelwert und Maximum. Fuer U_c gilt dasselbe.

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

- Waist in der Atomebene: 0.7434 .. 2.5000 µm
- Waist vor der ersten Linse: 0.5353 .. 1.8001 mm
- Stuetzstellen: 121

## Ergebnis je Groesse

| Groesse | Spanne | Verlauf | Minimum | bei Waist |
|---|---|---|---|---|
| U_h (hart, Kreis) | 7.879 .. 99 % | faellt | 7.879 % | 2.5000 µm (Rand) |
| eta_h^circ (hart, Kreis) | 0.4084 .. 6.884 % | uneindeutig | 0.4084 % | 0.7434 µm (Rand) |
| eta_h^box (hart, Pitch-Quadrat) | 4.294 .. 14.18 % | uneindeutig | 4.294 % | 0.7434 µm (Rand) |
| U_w (atom-gewichtet) | 0.6163 .. 6.691 % | faellt | 0.6163 % | 2.5000 µm (Rand) |
| eta_w (atom-gewichtet) | 0.04771 .. 7.228 % | uneindeutig | 0.04771 % | 0.8312 µm |
| U_c (Penalty) | 9.695 .. 122.1 % | faellt | 9.695 % | 2.5000 µm (Rand) |
| eta_c (Penalty) | 5.33 .. 15.92 % | uneindeutig | 5.33 % | 0.7434 µm (Rand) |
| J (Score) | 11.56 .. 87.06 % | faellt | 11.56 % | 2.5000 µm (Rand) |

"Rand" heisst: das Minimum liegt am Ende des abgefahrenen Bereichs. Dann ist
es vermutlich kein Optimum, sondern nur die Grenze - weitersuchen.

## Markierter Punkt (Stern im Plot)

- Waist = 1.9000 µm (win_input = 0.7043 mm)

| Groesse | Wert |
|---|---|
| U_h | 13.85 % |
| eta_h^circ | 1.615 % |
| eta_h^box | 12.49 % |
| U_w | 1.064 % |
| eta_w | 0.1525 % |
| U_c | 17.04 % |
| eta_c | 15.58 % |
| J | 16.6 % |

Linear zwischen den beiden benachbarten Stuetzstellen interpoliert.
Die Kurven sind hier glatte Funktionen des Waists - anders als r_x/r_y
im Multitone-Scan, die Optimierungs-ERGEBNISSE sind; dort waere
Interpolieren irrefuehrend.

## Werte

| Waist (µm) | w_in (mm) | U_h (%) | eta_h^circ (%) | eta_h^box (%) | U_w (%) | eta_w (%) | U_c (%) | eta_c (%) | J (%) |
|---|---|---|---|---|---|---|---|---|---|
| 0.7434 | 1.8001 | 99 | 0.4084 | 4.294 | 6.691 | 0.1496 | 122.1 | 5.33 | 87.06 |
| 0.9191 | 1.4560 | 64.26 | 0.4727 | 5.834 | 4.444 | 0.4001 | 79.22 | 7.193 | 57.61 |
| 1.0947 | 1.2224 | 44.26 | 0.6986 | 6.465 | 3.16 | 0.4585 | 54.53 | 7.966 | 40.56 |
| 1.2704 | 1.0534 | 32.21 | 0.9123 | 7.742 | 2.359 | 0.5199 | 39.68 | 9.547 | 30.64 |
| 1.4460 | 0.9254 | 24.49 | 1.341 | 9.931 | 1.827 | 0.5729 | 30.15 | 12.27 | 24.79 |
| 1.6217 | 0.8252 | 19.25 | 1.825 | 11.83 | 1.456 | 2.221 | 23.7 | 14.23 | 20.86 |
| 1.7974 | 0.7445 | 15.54 | 1.611 | 12.53 | 1.188 | 0.9714 | 19.12 | 15.42 | 18.01 |
| 1.9730 | 0.6782 | 12.81 | 1.95 | 12.47 | 0.987 | 0.2577 | 15.77 | 15.53 | 15.69 |
| 2.1487 | 0.6228 | 10.75 | 3.815 | 12.7 | 0.8331 | 2.724 | 13.22 | 15.19 | 13.81 |
| 2.3243 | 0.5757 | 9.145 | 5.871 | 13.21 | 0.7125 | 5.783 | 11.25 | 15.07 | 12.4 |
| 2.5000 | 0.5353 | 7.879 | 6.884 | 14.18 | 0.6163 | 7.228 | 9.695 | 15.92 | 11.56 |

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
| Crosstalk-Region (hart) | Kreis mit Radius 1 µm UND Pitch-Quadrat, Seitenlaenge 5.288 µm (in eta_c/J geht der Pitch-Quadrat, Seitenlaenge 5.288 µm ein) |
| Gitter (hart) | 401 x 401 Zellen (Mittelpunktsregel) je Region |
| sigma_atom | 107.64 nm (T = 17.00 µK, nu_r = 60.40 kHz) |
| Sub-Gitter (gewichtet) | 241 x 241 ueber +-6 sigma_atom |
| Atom-Versatz | 0.0000 / 0.0000 µm |
| alpha | 0.700 |
| combo_lambda | 0.750 |
