# Fit-Formeln für r_x/r_y (gewichteter Amplituden-Scan)

Stand: 2026-08-21. Datengrundlage: `scan_amp_data_weighted_N3x4_10x10pts_Airy.pkl`
(N_x=3, N_y=4, alpha=0.7, r_bounds=(0.1, 10), win_input 0.8–1.7 mm, width
0.2–0.4 MHz). Die 7 Gitterpunkte, die sich in der Post-hoc-Nachoptimierung
(`refine_scan_amp_results_weighted()`) als Warm-Start-Artefakte herausgestellt
hatten (siehe vorheriger Chat-Abschnitt bzw. Projekt-Status-Doc), sind hier
VORHER korrigiert worden — die Formeln unten beschreiben also die bereinigte,
nicht die rohe Scan-Ausgabe. Erzeugt mit `beispiel_weighted_amp_fit_abhaengigkeiten.py`
(`fit_amplitude_dependence()`), unverändert vom deployten Skript.

Beide Achsen: x = Eingangs-Waist vor der ersten Linse in mm, y = width in MHz.

## r_x (äußeres/inneres Amplitudenverhältnis in x)

**Polynom 2. Grades** — R² = 0.384

```
r_x(x, y) = c0 + c1*x + c2*y + c3*x² + c4*y² + c5*x*y

c0 =   45.987
c1 =  -38.649
c2 = -130.460
c3 =    7.756
c4 =   80.387
c5 =   61.866
```

**Physikalisches Modell** (Intensitäts-Ausgleich, siehe Skript-Docstring) — R² = 0.000

```
r_x(waist_eff_µm, width_MHz) = exp(a·(width_MHz / waist_eff_µm)² + b)

a = -0.108
b =  0.731
```

**Kubische Spline-Interpolation** — R² = 1.000 (exakt durch jeden Datenpunkt, siehe Erklärung unten)

## r_y (äußeres/inneres Amplitudenverhältnis in y)

**Polynom 2. Grades** — R² = 0.753

```
r_y(x, y) = c0 + c1*x + c2*y + c3*x² + c4*y² + c5*x*y

c0 =   89.846
c1 =  -71.233
c2 = -293.678
c3 =   15.317
c4 =  259.129
c5 =  106.732
```

**Physikalisches Modell** — R² = 0.064

```
r_y(waist_eff_µm, width_MHz) = exp(a·(width_MHz / waist_eff_µm)² + b)

a = 7.961
b = -0.048
```

**Kubische Spline-Interpolation** — R² = 1.000

## Warum sind Polynom und physikalisches Modell hier so schwach?

R²=0.38/0.75 (Polynom) bzw. praktisch 0 (physikalisches Modell) sind keine
Fitting-Fehler, sondern eine direkte Folge der bereits diagnostizierten
Sprungstellen: die reale (r_x, r_y)-Landschaft hat in diesem Scanbereich
mehrere, teils konkurrierende lokale Optima, zwischen denen der optimale Wert
abrupt wechselt (die diagonale "Ridge" in den Heatmaps). Ein glattes Modell
mit wenigen Parametern kann eine solche unstetige Fläche grundsätzlich nicht
gut abbilden — egal wie gut der Fit-Algorithmus arbeitet. Die beigefügten
Diagnose-Plots (`FlatMultiTone_AmpFitWeighted_r_x/r_y_N3x4_10x10pts.png`)
zeigen das: die Residuen des Polynom-Modells sind genau dort am größten, wo
die Sprungzone liegt (rechts unten in der Heatmap).

Praktische Konsequenz: für eine zuverlässige Extrapolation außerhalb der
gescannten 10×10-Punkte taugen beide geschlossenen Modelle aktuell nur
eingeschränkt. Sinnvoller wäre entweder (a) ein feinerer Scan, der die
Sprungzone besser auflöst, oder (b) der Fit nur auf dem "glatten" Teilbereich
(kleines win_input/width, wo Polynom/physikalisches Modell erfahrungsgemäß
gut passen), oder (c) direkt mit der Spline arbeiten (siehe unten).

## Die Spline-Geschichte — was ist das, und bekomme ich eine Funktionsvorschrift?

Eine kubische Spline-Interpolation legt zwischen benachbarten Datenpunkten
jeweils ein kubisches Polynom-Segment, so dass Wert, erste UND zweite
Ableitung an jedem Segmentübergang exakt übereinstimmen. Ergebnis: eine
glatte Kurve (bzw. hier eine glatte Fläche, weil es zwei Variablen — waist
und width — sind), die exakt durch jeden Messpunkt geht, ohne die wilden
Schwingungen, die ein einzelnes hochgradiges globales Polynom durch dieselben
Punkte hätte (Runge-Phänomen). `scipy`s `RectBivariateSpline` macht das als
Tensorprodukt: eine bikubische Fläche aus lauter kleinen, glatt
zusammengesetzten Polynom-Flicken, ein Flicken pro Gitterzelle.

**Ist das eine "Funktionsvorschrift"?** Ja — nur keine einzelne globale
Formel wie beim Polynom oben, sondern eine STÜCKWEISE Formel:

- Formal ist die Fläche als B-Spline dargestellt:
  `S(x,y) = Σᵢ Σⱼ c_ij · B_i(x) · B_j(y)`, wobei `B_i`/`B_j` kubische
  B-Spline-Basisfunktionen sind (definiert über einen Knotenvektor) und
  `c_ij` die gefitteten Koeffizienten. Für Ihr 10×10-Gitter: 14 Knoten pro
  Achse, 100 Koeffizienten (10×10) pro Fläche (für r_x UND für r_y je einmal).
- Äquivalent, und leichter vorstellbar: der (waist, width)-Bereich wird in
  9×9 = 81 Zellen zerlegt; INNERHALB jeder Zelle ist `r_x(x,y)` tatsächlich
  ein einziges bikubisches Polynom (16 Koeffizienten, lokale Koordinaten) —
  Sie können sich das exakte Polynom für jede einzelne Zelle geben lassen,
  nur eben nicht EIN Polynom für die ganze Fläche.

**Praktisch heißt das:** wenn Sie "eine Formel, die ich in Origin/Excel/eine
Publikation eintippen kann" wollen, ist die Spline dafür nicht der richtige
Kandidat — dafür sind Polynom- oder physikalisches Modell gedacht (die hier
aber, siehe oben, für DIESEN Datensatz schlecht passen). Wenn Sie dagegen
"die beste, exakte Beschreibung der tatsächlich gemessenen Abhängigkeit,
auswertbar an jedem Zwischenpunkt" wollen, ist die Spline genau richtig — sie
reproduziert exakt das, was gescannt wurde, ohne die Sprünge künstlich
wegzuglätten.

Falls gewünscht, kann ich die vollständigen Knotenvektoren + Koeffizienten
(je 100 Zahlen für r_x/r_y) als CSV exportieren, oder ein kleines,
scipy-freies Python-Snippet schreiben, das dieselbe Fläche aus diesen Zahlen
exakt rekonstruiert (z.B. für Verwendung außerhalb dieses Projekts) — bisher
noch nicht gemacht, auf Zuruf nachreichbar.

## Dateien

- `Fit_Plots/FlatMultiTone_AmpFitWeighted_r_x_N3x4_10x10pts.png` — 5-Panel-
  Diagnose (Rohdaten, Polynom, physikalisches Modell, Spline, Residuen) für r_x.
- `Fit_Plots/FlatMultiTone_AmpFitWeighted_r_y_N3x4_10x10pts.png` — dasselbe für r_y.
