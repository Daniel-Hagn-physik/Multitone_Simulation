# Fit-Formeln für r_x/r_y — gewichteter Amplituden-Scan, 151×151 Punkte

Stand: 2026-08-24. Datengrundlage: `scan_amp_data_weighted_N3x4_151x151pts_Airy_2500res.pkl`
(N_x=3, N_y=4, alpha=0.7, r_bounds=(0.1, 10), win_input 0.8–1.7 mm vor der
ersten Linse, width 0.2–0.4 MHz, Auflösung 151×151, sigma_atom aus
atom_temperature=17 µK, trap_freq_r=60.4 kHz). Datenstand: **roh** — siehe
Abschnitt „Datenqualität" unten dazu, warum hier bewusst NICHT die
Multi-Start-Nachoptimierung (`refine_scan_amp_results_weighted()`) angewendet
wurde, anders als beim vorherigen 10×10-Datensatz.

Beide Achsen: x = Eingangs-Waist vor der ersten Linse in mm, y = width in MHz.

## Kurzfassung

Die bisherigen Modelle (Polynom 2. Grades, physikalisches Exponential-Modell)
waren nicht deshalb schlecht, weil die falschen Formeln gewählt wurden,
sondern weil die reale (r_x, r_y)-Landschaft bei dieser Auflösung eine
**echte, scharfe, räumlich zusammenhängende Struktur** zeigt (eine Front, an
der der unbeschränkte Amplituden-Optimum-Wert die r_bounds-Schranke
überschreitet, plus ein zweiter, konkurrierender lokaler-Optimum-Bereich bei
r_y) — kein glattes Polynom niedrigen Grades und kein einzelnes
Exponentialmodell kann eine solche Kante gut nachbilden, unabhängig von der
Qualität des Fit-Algorithmus.

Getestet wurden 9 Modellfamilien mit einer **ehrlichen, räumlichen
Kreuzvalidierung** (siehe unten, warum das wichtig ist). Ergebnis: eine
**Gaussian-Process-Regression auf log(r) mit Matérn-3/2-Kernel** schlägt alle
anderen Kandidaten klar und ist die empfohlene Fitfunktion. Details, Zahlen
und eine fertige, sklearn-unabhängige Predict-Funktion folgen unten.

## Warum eine neue Bewertungsmethode nötig war

Der vorherige Formeln-Bericht (10×10-Datensatz) nannte für die kubische
Spline-Interpolation R² = 1.000 als Bestwert. Das ist **kein Fehler, aber
irreführend**: eine interpolierende Spline geht per Konstruktion exakt durch
jeden Trainingspunkt — R²=1.000 sagt nur, dass die Spline sich an die
Trainingsdaten erinnert, nichts darüber, wie gut sie an einem NICHT
gescannten Zwischenpunkt vorhersagt.

Für diesen Bericht wurde stattdessen eine **räumliche Block-Kreuzvalidierung**
verwendet: das (waist, width)-Gitter wird in 5×5=25 zusammenhängende Blöcke
zerlegt, und ein Modell wird nacheinander auf 4/5 der Blöcke trainiert und auf
dem verbleibenden, komplett unbekannten Block bewertet (GroupKFold, 5 Folds,
wiederholt für alle Blockkombinationen). Das ist ein deutlich härterer,
aber ehrlicherer Test, weil ganze zusammenhängende Gebiete — auch quer über
die scharfe Ridge hinweg — nie im Training gesehen wurden. Zur Einordnung:
unter diesem Test fällt sogar ein zweiter Spline-Kandidat (`SmoothBivariateSpline`)
auf R² ≈ −339 (r_x) bzw. −1.5 (r_y) — sie überschwingt zwischen den
Trainingspunkten wild und ist für Vorhersagen an neuen Punkten unbrauchbar,
obwohl sie an den Trainingspunkten selbst exakt ist. Genau dieses Verhalten
blieb beim naiven Voll-Datensatz-R² unsichtbar.

## Modellvergleich (Block-Kreuzvalidierung, 5 Folds, R² = Mittelwert ± Std.)

| Modell | r_x R² | r_y R² |
|---|---|---|
| Polynom 2. Grades (roh) | −1.06 ± 2.09 | +0.46 ± 0.21 |
| Polynom 2. Grades (auf log r) | −0.46 ± 0.58 | +0.23 ± 0.36 |
| Polynom 4. Grades + Ridge (roh) | −0.47 ± 1.07 | +0.28 ± 0.28 |
| Polynom 4. Grades + Ridge (auf log r) | −0.02 ± 0.45 | +0.21 ± 0.25 |
| Physikalisches Modell exp(a·(w/waist)²+b) | −0.46 ± 0.89 | −0.15 ± 0.10 |
| Random Forest (300 Bäume, volle Daten) | +0.20 ± 0.32 | +0.31 ± 0.49 |
| Hist Gradient Boosting (volle Daten) | +0.17 ± 0.29 | +0.38 ± 0.38 |
| Glättende Bivariate Spline (SmoothBivariateSpline) | −339 ± 442 | −1.53 ± 2.48 |
| **Gaussian-Process-Regression, log(r), Matérn-3/2** | **+0.37 ± 0.34** | **+0.57 ± 0.21** |

(GPR wurde aus Rechenzeitgründen für den CV-Vergleich auf 800 zufälligen
Trainingspunkten pro Fold gefittet, alle anderen Modelle auf den vollen
~18 000 Trainingspunkten je Fold — GPR gewinnt also sogar mit deutlich
weniger Daten als die Baum-Modelle.)

**Wichtige Einordnung:** ein negatives R² bedeutet „schlechter als einfach
den Mittelwert zu raten" — das trifft auf Polynom und physikalisches Modell
bei r_x fast durchgängig zu. Auch das beste Modell (GPR) erklärt bei r_x nur
~37 % der Varianz unter echter Extrapolation — das ist keine Schwäche des
Fit-Algorithmus, sondern eine reale Grenze: exakt an der scharfen Ridge kann
selbst das beste Modell nicht zuverlässig vorhersagen, wie sich winzige
Verschiebungen im Scan-Bereich auf den genauen Ridge-Verlauf auswirken,
wenn dieser Bereich komplett aus dem Training ausgeschlossen wurde.

## Empfohlenes Modell: Gaussian-Process-Regression auf log(r)

```
log r(waist_mm, width_MHz) ~ GP( mean = y_train_mean,
                                  kernel = sigma_f² · Matern_3/2(d) + WhiteKernel(sigma_n²) )
r_x: sigma_f² = 0.673, length_scale = [0.202, 0.199] (standardisierte Einheiten), sigma_n² = 0.0116
r_y: sigma_f² = 0.610, length_scale = [0.209, 0.227] (standardisierte Einheiten), sigma_n² = 0.0078
```

Die kurzen length_scales (≈0.2 in standardisierten Einheiten, d.h. ≈20 % der
Datenspannweite je Achse) sind selbst ein quantitativer Beleg für die
scharfe Struktur: das Modell "lernt" von den Daten, dass es sich in dieser
kurzen Distanz stark ändern muss, statt es ihm vorzugeben.

Die finale Produktions-Version wurde auf 3000 zufällig ausgewählten
Gitterpunkten (statt allen 22 801, aus Rechenzeitgründen — GP-Training
skaliert kubisch mit der Punktzahl) trainiert und deckt damit ~13 % der
Fläche ab, aber praktisch jeden Punkt des vollen 151×151-Gitters mit einem
nahegelegenen Trainingspunkt. Die Diagnose-Plots (siehe unten) zeigen: die
Rekonstruktion der Ridge ist bis auf einen schmalen Saum von 1–2
Gitterzellen exakt — sichtbar an den fast überall weißen (=0) Residuen, mit
kleinen roten/blauen Streifen exakt entlang der schärfsten Kante.

**Das ist keine Formel im klassischen Sinn** (keine einzelne Gleichung für
Origin/Excel), sondern ein gewichteter Mittelwert über 3000 Stützpunkte
(Kernel-Regression) — mathematisch aber vollständig definiert und exakt
reproduzierbar. Eine fertige, **sklearn-unabhängige** Python-Funktion dafür
liegt bei: `gpr_amp_predict.py` + `gpr_amp_export.npz` (nur numpy nötig,
keine sklearn-Versionsabhängigkeit, Formel im Docstring erklärt, per
direktem Zahlenvergleich gegen die Original-sklearn-Vorhersage auf < 1e-13
relativer Abweichung verifiziert). Nutzung:

```python
from gpr_amp_predict import load_predictors
predict_rx, predict_ry = load_predictors("gpr_amp_export.npz")
r_x = predict_rx(waist_mm=1.2, width_MHz=0.31)
r_y = predict_ry(waist_mm=1.2, width_MHz=0.31)
```

Funktioniert auch vektorisiert (numpy-Arrays statt einzelner Zahlen).

## Sekundär-Modell: Hist Gradient Boosting

Für den Fall, dass eine reine numpy/sklearn-Abhängigkeit unpraktisch ist
oder ein Baum-basiertes Modell bevorzugt wird: `HistGradientBoostingRegressor`
(volle 22 801 Punkte, R²=0.17/0.38 unter CV, siehe Tabelle) ist deutlich
schneller zu trainieren, aber schlechter in der Vorhersagequalität als GPR.
Nicht als primäre Empfehlung, aber als Fallback dokumentiert.

## Datenqualität: warum "roh" und nicht "korrigiert"

Beim vorherigen 10×10-Datensatz waren ~30 % der Punkte isolierte
Warm-Start-Trapping-Artefakte (einzelne, vom Rest losgelöste Fehltreffer),
die per Mehrfachstart-Nachoptimierung sinnvoll korrigierbar waren. Bei
diesem 151×151-Datensatz wurde dieselbe Sprungerkennung
(`detect_amp_discontinuities`, z_thresh=3.5) angewendet — sie markiert zwar
ebenfalls ~29 % der Punkte (6706/22801), aber eine
Zusammenhangskomponenten-Analyse zeigt: **davon liegen 6705 in genau ZWEI
großen, geometrisch zusammenhängenden Flächen** (der Bound-Saturations-Front
und dem r_y-Nebenbereich, siehe Diagnose-Plot), nicht verteilt als isolierte
Einzelpunkte. Nur EIN einzelnes Pixel war ein echter isolierter Ausreißer
und wurde per Nachbar-Median korrigiert (r_x: 1.412→1.412, r_y: 0.957→0.956 —
vernachlässigbar). Eine Mehrfachstart-Nachoptimierung über 6705 Punkte hätte
Stunden gedauert und wäre methodisch fragwürdig gewesen, weil hier (anders
als beim 10×10-Fall) die überwiegende Mehrheit der "Sprünge" eine reale,
kohärente Struktur ist, keine Artefakte — Nachbar-Interpolation hätte
genau die Kante zerstört, die eigentlich die interessante Physik ist.

## Nachtrag (2026-08-24): geschlossene Formel abseits der Ridge

Auf Wunsch des Users: eine klassische geschlossene Formel NUR für den
Bereich abseits der Ridge (statt GPR). Dafür wurde die Ridge-Zone
(`detect_amp_discontinuities`, siehe oben) um 2 Gitterzellen aufgeweitet
(Sicherheitspuffer, da der Übergang direkt neben der markierten Kante noch
mitbetroffen ist) und aus dem Fit-Bereich entfernt — es bleiben 14 711 von
22 801 Punkten (64,5 %) als "glatter" Trainingsbereich.

**Ergebnis: sehr unterschiedlich für r_x und r_y.**

### r_y: funktioniert gut — kubische Formel, R²(CV) = 0.87

```
log(r_y) = c0 + c1·x + c2·y + c3·x² + c4·x·y + c5·y² + c6·x³ + c7·x²·y + c8·x·y² + c9·y³

x = waist vor der ersten Linse in mm, y = width in MHz

c0 =  14.50939296        c5 =  11.68231210
c1 =  -5.16547040        c6 =  -0.00978362
c2 = -26.58173255        c7 =  16.23136736
c3 =  -0.11302849        c8 =  64.39978519
c4 = -41.87960284        c9 =  -8.26358643
```

Block-Kreuzvalidierung (dieselbe Methode wie oben, aber nur auf dem
glatten Bereich, GEGENÜBER dem gleichen Test): **R² = 0.865 ± 0.059** — ein
deutlicher, verlässlicher Sprung ggü. den −0.15…+0.46, die im Modellvergleich
oben auf dem GESAMTEN Bereich (inkl. Ridge) erreicht wurden. Diagnose-Plot
(`..._r_y_smoothformula.png`): Rohdaten, Formel-Vorhersage und Residuen nahezu
deckungsgleich über fast die gesamte Fläche, mit kleinen Abweichungen nur
direkt am Rand der ausgeschlossenen Zonen.

**Gültigkeitsbereich:** waist 0.8–1.7 mm, width 0.2–0.4 MHz, AUSSER der
Ridge-Zone selbst (die diagonale Sprungkante, siehe
`..._diagnose_ridge.png`) und dem kleinen r_y-Nebenbereich oben rechts in der
Heatmap (dort gibt die Formel keine verlässlichen Werte — dort bitte
`gpr_amp_predict.py` verwenden oder den Rohdatenpunkt direkt nachschlagen).

### r_x: erster Versuch (nur Ridge ausgeschlossen) scheiterte

Mit NUR der Ridge-Zone ausgeschlossen (s.o., 64,5 % der Fläche) erreichte
KEIN Kandidat — Polynome 2.–6. Grades (mit/ohne Ridge-Regularisierung, versch.
Stärken), Verhältnis-Features (width/waist, 1/waist), ein 2D-Logistik-
/Sigmoid-Sättigungsmodell, eine rationale Padé-Funktion — ein positives
Kreuzvalidierungs-R² (bestes Ergebnis: Polynom 5. Grades, R²=−0.32±0.28).
Grund (per Residuen-Diagnose bestätigt): auch außerhalb der markierten Ridge
blieb eine kleine, aber steile Sättigungs-Ecke übrig (kurzer waist + schmale
width, wo r_x Richtung obere Schranke 10 ansteigt), die ein Polynom nicht
mitmodellieren konnte, ohne den Rest der Fläche zu verzerren.

### r_x — Nachtrag 2 (2026-08-24, User-Nachfrage: "die Ecken mit 10 musst du nicht mitnehmen"): funktioniert jetzt!

Genau dieser Verdacht des Users war richtig. Zusätzlich zur Ridge-Zone wurde
jetzt auch die Sättigungs-Ecke ausgeschlossen — **formfolgend statt als grobe
Box** (Zusammenhangskomponente der Punkte mit r_x>5, um 2 Gitterzellen
aufgeweitet als Sicherheitspuffer). Ein erster Versuch mit einer rechteckigen
Box (waist≤1,05mm & width≤0,26mm) hatte unnötig viel vom angrenzenden glatten
("blauen") Bereich mit weggeschnitten — auf Hinweis des Users durch die engere,
der tatsächlichen Ecken-Form folgende Maske ersetzt (waist bis max. 0,974mm,
width bis max. 0,235MHz, mit einer leicht unregelmäßigen Kontur statt einer
scharfen Boxkante). Damit bleiben 14 304 von 22 801 Punkten (62,7 %) als
Trainingsbereich — mehr als bei der Box-Variante (62,1 %), trotz vollständigem
Ausschluss der Ecke.

```
log(r_x) = Σ c_ij · x^i · y^j   (Polynom 6. Grades, 28 Terme)

x = waist vor der ersten Linse in mm, y = width in MHz

c(0,0)=13.540447   c(0,4)=-368.022315  c(3,2)=101.125246
c(0,1)= 1.962556   c(1,3)= 133.833739  c(4,1)=  6.529893
c(1,0)=10.886105   c(2,2)= 213.660505  c(5,0)= -0.246393
c(0,2)=75.059465   c(3,1)=  12.251280  c(0,6)=2137.818939
c(1,1)=-174.071313 c(4,0)= -0.803278   c(1,5)= 218.066372
c(2,0)=-1.004954   c(0,5)=-1049.695131 c(2,4)=-607.446722
c(0,3)=31.503395   c(1,4)=  67.606097  c(3,3)=-358.372353
c(1,2)=-27.176746  c(2,3)= 453.922716  c(4,2)= -46.905059
c(2,1)=-9.679756                       c(5,1)=  0.100494
c(3,0)=-1.440388                       c(6,0)=  0.160565
```

(vollständig, exakt reproduzierbar auch in `rx_smooth_formula.py`, `predict_rx()`.)

Block-Kreuzvalidierung: **R² = 0.487 ± 0.344** — etwas niedriger als bei der
(zu großzügigen) Box-Variante (0.556), aber ehrlicher, da nicht durch
zusätzlich weggeschnittenes glattes Gebiet künstlich aufgebessert. Verteilung
über die 5 Test-Blöcke diesmal gleichmäßiger (0.93, 0.70, 0.62, −0.01, 0.20 —
kein Block mehr stark negativ). Der Diagnose-Plot
(`..._r_x_smoothformula.png`) zeigt eine sehr gute Übereinstimmung über fast
die gesamte Fläche; größere Abweichungen konzentrieren sich sichtbar auf den
schmalen Rand direkt neben Ridge und Ecke.

**Gültigkeitsbereich:** waist 0.8–1.7 mm, width 0.2–0.4 MHz, AUSSER der
Ridge-Zone UND der Sättigungs-Ecke (siehe Diagnose-Plot für die genaue,
leicht unregelmäßige Kontur; `rx_smooth_formula.py`s `is_in_smooth_domain()`
nutzt dafür eine konservative rechteckige Näherung waist≤0,98mm &
width≤0,235MHz) — dort bitte `gpr_amp_predict.py` (GPR) verwenden.

**Fazit (Stand Nachtrag 3):** für r_x ebenfalls eine geschlossene Formel
verfügbar, aber mit deutlich mehr Vorsicht zu genießen als die r_y-Formel
(kompaktere 28-Term-Formel 6. Grades statt 10-Term-Formel 3. Grades, UND
niedrigeres, weniger stabiles R²). **Dieser Stand wurde durch Nachtrag 4
unten abgelöst** — dort werden r_x UND r_y auf einer saubereren,
geometrisch begründeten Fläche neu gefittet, mit deutlich besserem und
stabilerem Ergebnis für beide.

## Nachtrag 4 (2026-08-24, User-Wunsch: "Lass alles außer dem zentralen Diagonalstreifen weg, das Artefakt oben rechts bei beiden auch")

Der User bat darum, für r_x UND r_y nur noch den **"zentralen
Diagonalstreifen"** zu behalten und zusätzlich das **Artefakt oben rechts**
auszuschließen — für beide Größen gemeinsam. Das war Anlass, den bisherigen,
eher ad-hoc zusammengesetzten Ausschluss (Ridge + separat per Schwellwert
gefundene Sättigungs-Ecke, s. Nachtrag 2/3, nur für r_x) durch eine sauberere,
rein geometrische Definition zu ersetzen, die für r_x UND r_y identisch gilt.

**Was ist "der zentrale Diagonalstreifen"?** Die Sprungerkennung
(`detect_amp_discontinuities`, s.o.) liefert zwei große
Zusammenhangskomponenten: die Haupt-Ridge (6148 Punkte, die diagonale
Sprungkante durchs halbe Bild) und eine zweite, kleinere Komponente (557
Punkte) — das ist genau das **Artefakt oben rechts** (waist ≈ 1,50–1,70 mm,
width ≈ 0,361–0,40 MHz), ein isolierter zweiter Sprungbereich in der Ecke.
Werden beide (je um 2 Gitterzellen aufgeweitet als Sicherheitspuffer) aus der
Fläche entfernt, zerfällt der Rest in mehrere Stücke: ein einziges großes,
zusammenhängendes, diagonal verlaufendes Band mit 14 085 Punkten (61,8 % der
Gesamtfläche) — visuell exakt das, was man als "zentralen Diagonalstreifen"
bezeichnen würde (siehe `..._stripe_overview.png`) — plus mehrere kleine,
davon klar GETRENNTE Reste: die beiden Sättigungs-Ecken unten-links (407
Punkte, wo r_x Richtung 10 läuft) und oben-rechts (205 Punkte, ein
Rest-Fetzen neben dem Artefakt) sowie ein paar einzelne Rauschpixel (≤7
Punkte). Bemerkenswert: **innerhalb dieses einen großen Bandes sättigt
weder r_x noch r_y** — Kontrolle ergab exakt 0 Überlappung zwischen dem Band
und jeder der vier Extremwert-Flächen (r_x>5, r_x<0,5, r_y>5, r_y<0,5); r_x
bleibt dort im Bereich [0,86, 2,05], r_y im Bereich [0,90, 2,29]. Der
"zentrale Diagonalstreifen" ist also automatisch UND für beide Größen
identisch definiert — keine separate Ecken-Logik pro Größe mehr nötig.

**Ergebnis: deutlich besser als jede vorherige Version, für BEIDE Größen.**

| Größe | Grad | R² (Block-CV, nur im Streifen) | vorheriger Stand |
|---|---|---|---|
| r_x | 5 | **0,995 ± 0,003** | 0,487 ± 0,344 (Nachtrag 3) |
| r_y | 4 | **0,996 ± 0,002** | 0,865 ± 0,059 (Nachtrag 1) |

Der Sprung ist so groß, weil die vorherigen Trainingsflächen (insbesondere
für r_x) noch den Bereich JENSEITS der Ridge mit einschlossen, wo sich
r_x/r_y strukturell anders verhalten als im Streifen selbst — das hat die
Formel destabilisiert. Beschränkt man sich exakt auf den einen homogenen
Streifen, wird ein Polynom moderaten Grades fast exakt (R²≈0,995–0,996,
Streuung über die 5 CV-Blöcke jetzt klein statt groß).

### r_x — Polynom 5. Grades, 21 Terme

```
log(r_x) = Σ c_ij · x^i · y^j   (x = waist in mm, y = width in MHz)

c(0,0)= 13.199600   c(2,1)=  4.796988   c(1,4)=-99.146313
c(0,1)=-26.521663   c(3,0)=  0.527219   c(2,3)=  1.802786
c(1,0)= -7.648997   c(0,4)=  8.421634   c(3,2)=  3.203396
c(0,2)=  5.509540   c(1,3)= 40.588854   c(4,1)= -0.833631
c(1,1)=-22.476938   c(2,2)= 21.148834   c(5,0)=  0.000580
c(2,0)=  1.413018   c(3,1)=  0.224805
c(0,3)= 25.533367   c(4,0)= -0.144107
c(1,2)= 25.309962   c(0,5)=-19.037398
```

### r_y — Polynom 4. Grades, 15 Terme

```
log(r_y) = Σ c_ij · x^i · y^j   (x = waist in mm, y = width in MHz)

c(0,0)= 16.940189   c(0,3)= 59.978220   c(2,2)= 13.158751
c(0,1)=-42.937981   c(1,2)= 59.396480   c(3,1)= -2.260702
c(1,0)=-10.545708   c(2,1)=  7.681167   c(4,0)= -0.320965
c(0,2)= 23.685350   c(3,0)=  0.947051
c(1,1)=-18.034112   c(0,4)=-65.146325
                     c(1,3)=-68.824146
```

(vollständig, exakt reproduzierbar auch als Python: `rx_smooth_formula.py`,
`ry_smooth_formula.py`, jeweils `predict_rx()`/`predict_ry()`.)

Diagnose-Plots (`..._r_x_smoothformula.png`, `..._r_y_smoothformula.png`,
NEUE Version — ersetzt die Nachtrag-2/3-Plots gleichen Namens): Rohdaten,
Formel-Vorhersage und Residuen im Streifen praktisch deckungsgleich; RMS-
Residuum in log-Einheiten 0,0073 (r_x) bzw. 0,0076 (r_y), Maximum ≈0,047–0,051
(entspricht ≈5 % relativer Abweichung im schlechtesten einzelnen Punkt, meist
am Rand des Streifens direkt neben Ridge/Ecke).

**Gültigkeitsbereich (WICHTIG — enger als vorher!):** NUR innerhalb des
zentralen Diagonalstreifens, NICHT im gesamten gescannten Rechteck
0,8–1,7 mm / 0,2–0,4 MHz! Außerhalb (Ridge, beide Sättigungs-Ecken, Artefakt
oben rechts) liefert die Formel keine verlässlichen Werte — dort bitte
`gpr_amp_predict.py` (GPR, gilt überall im Scanbereich) verwenden. Die
mitgelieferten Python-Funktionen `is_in_stripe(waist_mm, width_MHz)` (in
`rx_smooth_formula.py`/`ry_smooth_formula.py`) prüfen das jetzt **exakt**
anhand der echten Streifen-Maske (nächster Gitterpunkt im Original-151×151-
Scan, gespeichert in `stripe_domain_mask.npz`) — keine grobe Box-Näherung
mehr wie in Nachtrag 3.

**Fazit:** für r_x UND r_y jetzt eine gemeinsame, geometrisch sauber
begründete, sehr genaue geschlossene Formel im zentralen Diagonalstreifen
(R²≈0,995/0,996). Dieser Stand ersetzt die Formeln aus Nachtrag 1–3. Für
Punkte außerhalb des Streifens (Ridge, Ecken, Artefakt) bleibt GPR
(`gpr_amp_predict.py`) die einzige verlässliche Option.

## Praktische Empfehlungen für die nächsten Schritte

1. **Für numerische Auswertung/Simulation:** `gpr_amp_predict.py` direkt
   verwenden — deckt den gesamten gescannten Bereich ab, mit der oben
   beschriebenen Genauigkeit (am zuverlässigsten abseits der Ridge, mit
   erhöhter Unsicherheit exakt auf der Kante).
2. **Falls eine geschlossene Formel für eine Publikation zwingend nötig
   ist:** der glatte Teil des Bereichs (kleines waist/width, unten links in
   den Heatmaps) lässt sich vermutlich gut mit dem physikalischen
   Exponentialmodell separat fitten — auf Zuruf kann ich das für einen
   sauber abgegrenzten Teilbereich nachreichen. Über die gesamte Fläche
   hinweg (inkl. Ridge) ist keine einzelne geschlossene Formel realistisch.
3. **Für eine bessere Auflösung der Ridge selbst:** entweder GPR mit mehr
   Trainingspunkten (aktuell aus Zeitgründen auf 3000 begrenzt — mit
   Sparse-/Inducing-Point-GP-Verfahren, z.B. GPyTorch oder scikit-learn mit
   Nyström-Approximation, ließen sich alle 22 801 Punkte nutzen) oder ein
   gezielter Nachscan mit feinerer Auflösung NUR entlang der Ridge.
4. Die zwei Diagnose-Plots (`_r_x.png`/`_r_y.png`) zeigen Rohdaten, GPR-Fit,
   HistGB-Fit und Residuen nebeneinander — Basis für eine visuelle Prüfung,
   ob die Genauigkeit für den geplanten Einsatzzweck ausreicht.

## Dateien

- `AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24_Formeln.md` — dieses Dokument.
- `Fit_Plots/AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24_r_x.png` — 5-Panel-Diagnose für r_x
  (GPR/HistGB, gesamte Fläche inkl. Ridge).
- `Fit_Plots/AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24_r_y.png` — dasselbe für r_y.
- `Fit_Plots/AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24_diagnose_ridge.png` — Übersichtsplot,
  der die Ridge-Struktur, die Sprungerkennung und die Schranken-Treffer zeigt (Grundlage für den
  Abschnitt „Datenqualität" oben).
- `gpr_amp_predict.py` + `gpr_amp_export.npz` — reproduzierbare, sklearn-unabhängige GPR-Predict-
  Funktion, gültig im GESAMTEN Scanbereich (auch Ridge/Ecken/Artefakt).
- `Fit_Plots/AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24_r_y_smoothformula.png` — **(Stand
  Nachtrag 4)** Diagnose-Plot zur r_y-Formel im zentralen Diagonalstreifen.
- `ry_smooth_formula.py` — **(Stand Nachtrag 4)** eigenständige Predict-Funktion für die r_y-Formel
  (nur numpy), inkl. exaktem `is_in_stripe()`-Gültigkeitscheck.
- `Fit_Plots/AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24_r_x_smoothformula.png` — **(Stand
  Nachtrag 4)** Diagnose-Plot zur r_x-Formel im zentralen Diagonalstreifen.
- `rx_smooth_formula.py` — **(Stand Nachtrag 4)** eigenständige Predict-Funktion für die r_x-Formel
  (nur numpy), inkl. exaktem `is_in_stripe()`-Gültigkeitscheck.
- `Fit_Plots/AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24_stripe_overview.png` — zeigt, wie
  der zentrale Diagonalstreifen (grün) geometrisch aus Ridge (rot) + Artefakt (blau) + Restflächen
  (schwarz) hervorgeht (siehe Nachtrag 4).
- `stripe_domain_mask.npz` — die exakte Streifen-Maske (151×151-Gitterkoordinaten + Boolean-Maske),
  von `rx_smooth_formula.py`/`ry_smooth_formula.py` für `is_in_stripe()` genutzt. Muss im selben
  Ordner wie diese beiden Skripte liegen.

Namensschema wie in `namenskonvention_fit_outputs.md` festgelegt:
`AmpFit_N{Nx}x{Ny}_{n_win}x{n_width}pts_{Profil}_{weighted|hart}_{roh|korrigiert}_{Datum}`.
