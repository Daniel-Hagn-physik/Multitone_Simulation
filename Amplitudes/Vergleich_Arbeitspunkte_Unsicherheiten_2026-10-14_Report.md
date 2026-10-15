# Vergleich der Arbeitspunkte multi-tone / single spot - Werte und Unsicherheiten, 2026-10-14

Gehoert zur Tabelle `tab:working_points_comparison`. Alle Werte sind neu gerechnet:

- **multi-tone:** mit `lib/atom_offset.py` (Optimierer-Code, unveraendert)
- **single spot:** mit den Formeln aus `lib/single_beam.py`

Basiswerte reproduzieren die Notebooks bzw. Reports auf alle Stellen:
`Atom_Offset_Notebook.ipynb`, `Single_Spot_Notebook.ipynb`, `AtomOffset_..._2026-09-10_Report.md`
und `SingleBeamOffset_..._2026-09-18_Report.md`.

## Arbeitspunkte

| | multi-tone | single spot |
|---|---|---|
| Profil | Airy 3x4, k = 1.483 | Airy, k = 1.483 |
| Waist | 1.04 µm | 1.90 µm |
| width | 0.37 MHz (Spannweite d = 2.337 µm) | - |
| r_x / r_y | 0.97 / 1.16 (genau diese Werte) | - |
| Kohaerenz | an, Phase 0 (entartetes Eckpaar) | - |
| pitch | 5.288 µm, 8 Nachbar-Sites | 5.288 µm, 8 Nachbar-Sites |
| sigma_atom | 107.6 nm (17 µK, 60.4 kHz) | 107.6 nm |

## Was "Shift" bedeutet

**multi-tone:** Gemeint ist das Maximum ueber alle Richtungen, jeweils bis zum Rand des
Tonarrays. Waagerecht und senkrecht laeuft das bis d/2 = 1.169 µm, diagonal bis zum
Eckton bei d/2 · √2 = 1.653 µm. Das Maximum liegt in beiden Faellen **diagonal**:

| Richtung | U_w max | bei r | eta_w max | bei r |
|---|---|---|---|---|
| waagerecht | 3.75 % | 1.169 µm (Rand) | 3.25 % | 0.70 µm |
| senkrecht | 6.87 % | 1.169 µm (Rand) | 2.38 % | 0.85 µm |
| **diagonal** | **8.99 %** | 1.653 µm (Eckton) | **3.514 %** | 0.946 µm |
| antidiagonal | 7.72 % | 1.653 µm | 3.47 % | 0.91 µm |

Das eta_w-Maximum wurde mit einem Optimierer genau lokalisiert (vorher Raster 41 nm). Am
Wert aendert das nichts (3.5141 statt 3.5142 %).

**single spot:** Ausgewertet wird bei r = Waist = 1.9 µm, dem Ende des Bereichs. U_w und eta_w
steigen bis dahin monoton, das Maximum liegt also am Ende. U_w haengt nicht von der Richtung
ab. **eta_w schon:**

| Richtung | eta_w bei r = 1.9 µm |
|---|---|
| waagerecht / senkrecht | 19.80 % |
| diagonal | **24.39 %** |

Die Notebooks und der Report vom 18.09. rechnen nur waagerecht bzw. senkrecht. Soll "Shift"
in beiden Spalten dasselbe heissen (Maximum ueber die Richtungen), gehoert in die
single-spot-Spalte **24.4 %**, nicht 19.8 %.

## Unsicherheiten - was eingeht

Die Parameter des Arbeitspunkts (Waist, width, r_x, r_y) sind Vorgaben. Sie **definieren** den
Punkt und tragen keinen Fehler. Unsicher sind die Groessen, die das Experiment vorgibt.
Die gemessenen Werte stehen im Kopf von `weighted_multitone_flattop_optimizer.py`:

| Parameter | Wert | betrifft |
|---|---|---|
| Atomtemperatur T | 17 ± 2 µK | U_w, eta_w (ueber sigma_atom) |
| Fallenfrequenz nu_r | 60.4 +2.4/−2.2 kHz (gerechnet ±2.3) | U_w, eta_w |
| Fallenabstand pitch | ±0.03 µm um den Rechenwert | eta_h, eta_w |

Die Beitraege werden quadratisch addiert.

- **Numerik:** Feinere Gitter aendern die gewichteten Groessen um weniger als 0.0001 pp.
- **Pixelraster der harten Groessen:** etwa 0.03 pp bei U_h (globales 1000er-Gitter gegen
  eigenes Gitter auf der Region).

## Ergebnis

| Groesse | multi-tone | single spot |
|---|---|---|
| U_h | 3.57 ± 0.03 % | 13.85 % (keine Parameterabhaengigkeit, Numerik < 0.001 pp) |
| eta_h | 8.39 ± 0.04 % | 12.493 ± 0.006 % |
| U_w | 0.43 ± 0.05 % | 1.06 ± 0.15 % |
| U_w Shift | 9.0 ± 0.6 % | 27.8 ± 1.9 % |
| eta_w | 1.40 ± 0.03 % | 0.15 ± 0.04 % |
| eta_w Shift | 3.51 ± 0.11 % | 19.8 ± 0.5 % (waagerecht) / 24.4 ± 0.7 % (diagonal) |

### Budget multi-tone (pp, Aenderung bei +/− 1 sigma)

| Groesse | Wert | T ±2 µK | nu_r ±2.3 kHz | pitch ±30 nm | gesamt |
|---|---|---|---|---|---|
| U_w | 0.4292 % | +0.045 / −0.045 | −0.029 / +0.029 | 0 | ±0.053 |
| eta_w | 1.3981 % | +0.017 / −0.017 | −0.011 / +0.011 | +0.017 / +0.006 | +0.026 / −0.020 |
| U_w Shift | 8.988 % | +0.53 / −0.56 | −0.35 / +0.35 | 0 | ±0.64 |
| eta_w Shift | 3.514 % | −0.011 / +0.011 | +0.007 / −0.007 | −0.117 / +0.103 | ±0.11 |
| U_h | 3.570 % | - | - | 0 | ±0.03 (Raster) |
| eta_h | 8.387 % | - | - | −0.040 / +0.040 | ±0.04 |

### Budget single spot (pp)

| Groesse | Wert | T ±2 µK | nu_r ±2.3 kHz | pitch ±30 nm | gesamt |
|---|---|---|---|---|---|
| U_w | 1.0638 % | +0.124 / −0.124 | −0.079 / +0.081 | 0 | ±0.147 |
| eta_w | 0.1523 % | +0.008 / −0.008 | −0.005 / +0.005 | +0.039 / −0.025 | +0.040 / −0.027 |
| U_w Shift (r = 1.9 µm) | 27.76 % | +1.53 / −1.63 | −1.03 / +1.02 | 0 | ±1.9 |
| eta_w Shift waagerecht | 19.80 % | −0.06 / +0.06 | +0.04 / −0.04 | +0.48 / −0.54 | ±0.52 |
| eta_w Shift diagonal | 24.39 % | −0.06 / +0.06 | +0.04 / −0.04 | −0.71 / +0.71 | ±0.71 |
| eta_h^box | 12.493 % | - | - | +0.007 / −0.006 | ±0.006 |

## Systematik (nicht in den ±)

Alle Werte in pp, als Aenderung gegenueber dem Tabellenwert.

| Groesse | 2. Nachbarkranz (24 statt 8 Sites) | pitch 5.16 statt 5.288 µm | Kohaerenz aus / Quadratur (nur multi-tone) |
|---|---|---|---|
| multi-tone eta_h | +1.37 | +0.18 | −0.14 |
| multi-tone U_h | 0 | 0 | +0.02 |
| multi-tone U_w | 0 | 0 | −0.03 |
| multi-tone eta_w | +0.45 | +0.20 | −0.02 |
| multi-tone U_w Shift | 0 | 0 | −0.23 |
| multi-tone eta_w Shift | +0.59 | +0.31 | −0.41 |
| single spot eta_h^box | +2.66 | −0.02 | - |
| single spot eta_w | +0.39 | −0.004 | - |
| single spot eta_w Shift waagerecht | +5.06 | −2.69 | - |
| single spot eta_w Shift diagonal | +2.83 | +2.94 | - |

**Nachbarkranz.** Airy faellt nur mit 1/r³ ab. Mit dem zweiten Kranz steigen alle Crosstalks
deutlich, am staerksten relativ beim single spot. Das ist eine Modellgrenze beider Rechnungen.
Sie ist **einseitig**: die angegebenen Crosstalks sind eher zu klein.

**Uniformity.** Sie ist von Nachbarn und pitch nicht betroffen.

**Waist-Empfindlichkeit**, falls der Waist im Experiment nicht genau sitzt (pp pro +10 nm):

| Groesse | multi-tone (1.04 µm) | single spot (1.9 µm) |
|---|---|---|
| U_w | −0.027 | −0.011 |
| eta_w | −0.020 | −0.038 (nichtlinear, nahe Minimum) |
| U_w Shift | +0.026 | −0.45 |
| eta_w Shift | +0.11 | −0.81 (waagerecht) / +0.02 (diagonal) |

## Abweichungen zur bisherigen Tabelle

- **eta_w multi-tone:** 1.42 % war der Wert mit den exakt optimierten Amplituden
  (r_x/r_y = 0.971/1.158 bei 1.0437 µm). Bei den in der Tabelle genannten 0.97/1.16 und
  1.04 µm sind es **1.40 %**. U_w aendert sich dort von 0.419 auf 0.429 %.
- **U_w Shift single spot:** 29 % passt zu keiner Standardrechnung. Der Notebook-Wert bei
  r = ω ist **27.76 %**. 29.3 % ergaebe sich bei T = 19 µK.
- **eta_w Shift single spot:** 20.1 % passt ebenfalls nicht. Waagerecht sind es **19.80 %**,
  diagonal **24.39 %**.

## LaTeX

```latex
\begin{table}[ht!]
    \centering
    \begin{tabular}{lcc}
        \hline
        Parameter & multi-tone & Single spot \\
        \hline
        Beam waist $\omega$ & $\SI{1.04}{\micro\metre}$ & $\SI{1.9}{\micro\metre}$ \\
        Frequency width $W$ & $\SI{0.37}{\mega\hertz}$ & --- \\
        Amplitude ratio $r_x$ & $0.97$ & --- \\
        Amplitude ratio $r_y$ & $1.16$ & --- \\
        Hard uniformity $U_h$ & $3.57\% \pm 0.03\%$ & $13.85\%$ \\
        Hard crosstalk $\eta_h$ & $8.39\% \pm 0.04\%$ & $12.49\% \pm 0.01\%$ \\
        Weighted uniformity $U_w$ & $0.43\% \pm 0.05\%$ & $1.06\% \pm 0.15\%$ \\
        Shift in $U_w$ & $9.0\% \pm 0.6\%$ & $27.8\% \pm 1.9\%$ \\
        Weighted crosstalk $\eta_w$ & $1.40\% \pm 0.03\%$ & $0.15\% \pm 0.04\%$ \\
        Shift in $\eta_w$ & $3.51\% \pm 0.11\%$ & $24.4\% \pm 0.7\%$ \\
        \hline
    \end{tabular}
    \caption{Comparison of the selected working points for the multi-tone and single-spot
    approaches. Shift values are the maxima over all directions of the atom displacement,
    up to the corner tone (multi-tone) or up to $r=\omega$ (single spot). Uncertainties
    follow from the measured trap parameters ($T = \SI{17(2)}{\micro\kelvin}$,
    $\nu_r = \SI{60.4}{\kilo\hertz}$, pitch $\pm\SI{30}{\nano\metre}$); numerical errors
    are below \SI{0.03}{pp}.}
    \label{tab:working_points_comparison}
\end{table}
```
