# Hard_Optimization

Scans und Auswertung der **harten** Metriken: Uniformity und Crosstalk werden
über ein globales Intensitätsgitter mit einer festen Pitch-Box-Maske
ausgewertet.

> Wie man die drei GUIs bedient, steht in **[ANLEITUNG.md](ANLEITUNG.md)** –
> geschrieben für jemanden, der die Skripte nicht selbst gebaut hat. Dieses
> Dokument ist die technische Beschreibung.

## Ich möchte … → dieses Skript

| Ich möchte … | ausführen |
|---|---|
| einen **neuen Scan bei festen Amplituden** rechnen | `run_scan.py` |
| einen **neuen Scan mit Amplituden-Optimierung** je Gitterpunkt rechnen | `run_amp_scan.py` |
| einen **vorhandenen Datensatz plotten und auswerten** | `run_plots.py` |
| die **Abhängigkeit von r_x/r_y fitten** (Flächen-Fit, Diagnose) | `run_amp_fits.py` |

Alles andere liegt in `lib/` und wird nur benutzt, nicht ausgeführt.

## Ordnerstruktur

```
Hard_Optimization/
    run_scan.py         <- ausführen: neuer Scan, feste Amplituden
    run_amp_scan.py     <- ausführen: neuer Scan, r_x/r_y je Punkt optimiert
    run_plots.py        <- ausführen: plotten/auswerten
    run_amp_fits.py     <- ausführen: Fits an die Amplituden-Abhängigkeit
    README.md           <- dieses Dokument
    ANLEITUNG.md        <- Bedienung der GUIs
    lib/                <- wird benutzt, nicht direkt ausgeführt
        __init__.py     Übersicht, welches Hauptskript was benutzt
        paths.py        Ordner-Konstanten, sys.path, Metrik-Familie
        scan_data.py    Laden/Speichern, Score/Bestpunkt/Region, verbotener Bereich
        report.py       Plots (PDF) und Markdown-Bericht
        multitone_flattop_optimizer.py          der eigentliche Optimierer
        multitone_amplitude_dependence_plots.py Scan-Plotter
        scan_checkpoint.py   stündliche Zwischenspeicherung
        resume_picker.py     Fortsetzen abgebrochener Scans
        airy_scale.py        Airy-Skalenfaktor-Definitionen
        perf_log.py          Laufzeit-Protokoll
        use_torch.py         optionale GPU-Beschleunigung
    Old_Scripts/        Archiv der abgelösten Konfig-Skripte
    Results/            gepickelte Scan-Rohdaten (.pkl)
    Bilder/2026-09-03/  PNG-Ausgaben der Scan-Plotter, tageweise
    Fit_Plots/2026-09-03/   Vektor-PDFs der Auswertung, tageweise
    Fit_Results/        Markdown-Berichte der Auswertung (flach, Datum im Namen)
```

**Bilder liegen tageweise, Daten und Berichte flach.** Jeder Lauf schreibt
seine PDFs nach `Fit_Plots/<JJJJ-MM-TT>/`, jeder Scan seine PNGs nach
`Bilder/<JJJJ-MM-TT>/`; der Ordner entsteht automatisch. Grund ist die Menge –
ein Auswertungslauf legt bis zu acht PDFs an, und nach ein paar Läufen war im
flachen Ordner nicht mehr zu sehen, was zusammengehört. Die Dateinamen tragen
das Datum weiterhin, der Ordner ist also eine zusätzliche Ordnung und nicht
die einzige Auskunft.

`Results/` und `Fit_Results/` bleiben flach: die `.pkl` ist der Datensatz und
nicht das Ergebnis eines Tages, und die Berichte will man nebeneinander lesen
können. Beide tragen das Datum im Dateinamen
(`HardScan_N3x4_41x41pts_Airy_2026-09-03_Report.md`).

Bereits vorhandene Dateien werden nicht verschoben – die Umstellung gilt ab
dem nächsten Lauf.

`lib/scan_data.py` und `lib/report.py` sind **buchstabengleich** mit den
gleichnamigen Dateien in `Weighted_Optimization/lib/`; alles, was die beiden
Ordner unterscheidet (Metrik-Familie, Dateinamens-Muster, Plot-Modul,
Berichts-Präfix), steht in `lib/paths.py`. `run_plots.py` ist ebenfalls in
beiden Ordnern identisch.

## Datensätze

| Datei | erzeugt von | enthält |
|---|---|---|
| `Results/scan_data_N…pts_….pkl` | `run_scan.py` | `uniformity_grid`, `crosstalk_grid`, `amps`, `best` |
| `Results/scan_amp_data_N…pts_….pkl` | `run_amp_scan.py` | zusätzlich `r_x_grid`, `r_y_grid`, `r_bounds` |

`run_plots.py` erkennt beide Arten automatisch (`scan_data.dataset_kind()`).

## Was `run_plots.py` erzeugt

Präfix `HardScan_N{Nx}x{Ny}_{n_win}x{n_width}pts_{Airy|Gauss}_{Datum}`:

| Datei | Fest-Amplituden | Amplituden-Scan |
|---|---|---|
| `…_metric_comparison.pdf` (Uniformity + Crosstalk) | x | x |
| `…_metric_comparison_amp.pdf` (dieselben + r_x, r_y) | – | x |
| `…_region.pdf` (Score-Karte mit Arbeitspunkt) | x | x |
| `…_valley_{X}_over_{Y}.pdf` (Talschnitt) | x | x |
| `…_line_{X}_over_{Y}.pdf` (Schnitt entlang der Geraden) | x | x |
| `…_point_cuts.pdf` (Kreuzschnitt durch den Arbeitspunkt) | – | x |
| `…_Report.md` | x | x |
| PNG-Übersicht des Scan-Plotters (optional) | x | x |

## Kreuzschnitt durch den Arbeitspunkt (`…_point_cuts.pdf`)

Zwei Schnitte durch den markierten Stern, nebeneinander: links r_x und r_y bei
**fester** width entlang des Waists, rechts bei **festem** Waist entlang der
width. Der Punkt ist in beiden Panels als senkrechte rote Linie markiert, sein
Wert je Kurve zusätzlich als Stern.

Gezeigt werden **nur r_x und r_y**, im gewohnten Aussehen der alten
Amplituden-Schnitte (`AmplitudeScanPlotter.plot_dependence_cuts`): r_x blau mit
Kreisen, r_y orange mit Quadraten, beide auf **einer** Achse — es ist dieselbe
Größe in zwei Richtungen. Uniformity und Crosstalk haben ihren Platz in den
Karten und im Talschnitt.

Das ist eine andere Frage als der Talschnitt: nicht „wie läuft das Minimum?",
sondern „wie empfindlich sind die Amplituden an meinem Arbeitspunkt?"

Gelesen wird immer **auf dem Gitter**. Liegt der Stern zwischen den
Gitterpunkten (selbst vorgegebener Punkt), laufen die Schnitte durch die
nächstgelegene Zeile bzw. Spalte, und der Titel nennt deren tatsächlichen
Wert. Interpolieren wäre hier irreführend: r_x und r_y sind
Optimierungs-Ergebnisse, keine glatten Funktionen.

Gibt es nur beim Amplituden-Scan — ein Fest-Amplituden-Scan hat keine
r_x/r_y-Gitter.

## Der Arbeitspunkt (Stern)

Welcher Punkt markiert wird, ist im Dialog frei wählbar:

* der **beste Gitterpunkt nach dem Score** (Voreinstellung) — dann zeigen
  Stern und Bericht ohne Zutun dasselbe;
* das **Minimum einer beliebigen Größe** — der Bericht bekommt dann einen
  eigenen Abschnitt dazu;
* ein **selbst vorgegebener Punkt**, in drei Varianten:
  * *nur Waist vorgeben* — die width kommt aus der Talpfad-Geraden;
  * *nur Width vorgeben* — der Waist kommt aus deren Umkehrung;
  * *Waist UND Width vorgeben* — beide Koordinaten selbst, ganz ohne Gerade.

Die beiden ersten Varianten setzen voraus, dass sich für den Datensatz
überhaupt eine Gerade legen lässt. Gibt es keine, sind sie im Dropdown
ausgegraut und die Auswahl fällt automatisch auf *Waist UND Width vorgeben*
zurück — dort wird dann eben nach beiden Koordinaten gefragt, statt
kommentarlos keinen Punkt zu zeichnen.

Ein selbst vorgegebener Punkt liegt in aller Regel zwischen den
Gitterpunkten und wird auch genau dort gezeichnet. Liegt der markierte Punkt
auf dem **Rand** des gescannten Fensters, wird der Stern **offen** statt
gefüllt gezeichnet — er ist dann kein Optimum, sondern nur das Ende des Scans.

## Der Score

Region, Bestpunkt, Score-Karte und die Führungsgröße `score` benutzen alle
dieselbe **rohe** Größe – dieselbe, die auch der Optimierer minimiert:

```
J = alpha * Uniformity + (1 - alpha) * Crosstalk
```

Bewusst **ohne** gitterweite Min-Max-Normierung. Eine solche Normierung hängt
am gescannten Fenster: dieselbe Physik ergäbe bei anderem Scan-Bereich andere
Zahlen. (Aus demselben Grund ist der normierte Score im
`Combinated_Optimization`-Ordner am 2026-09-01 entfallen.)

`alpha` und das Perzentil für die Region lassen sich in `run_plots.py` neu
setzen – Score, bester Punkt und Region werden dann aus den vorhandenen
Grids neu berechnet, ohne den teuren Scan zu wiederholen. Der Datensatz auf
der Platte bleibt unangetastet.

## Talschnitt und Gerade

Der Talschnitt folgt einer Größe (der *Führungsgröße*) und liest an genau
deren Minimum pro Spalte (bzw. Zeile) alle angehakten Größen ab – ohne
Interpolation, direkt an den Gitterpunkten.

Wie der Talpunkt gewählt wird, ist der Schalter mit dem größten Einfluss:

* **Globales Minimum je Spalte** – einfach, aber unbrauchbar, sobald das
  Minimum am Rand des Scan-Fensters oder an der Grenze des verbotenen
  Bereichs klebt.
* **Lokales Minimum nahe einer Leitgeraden** (Voreinstellung) – pro Spalte
  das lokale Minimum, das einer Leitgeraden am nächsten liegt. Lokal heißt:
  beide Nachbarn vorhanden und größer; Punkte am Scan-Rand und Punkte am
  ausgeschlossenen verbotenen Bereich fallen damit von selbst heraus. Die
  Leitgerade ist der gewöhnliche lineare Fit einer anderen Größe (Default:
  Uniformity). **Sie wählt nur aus, sie verschiebt nichts** – die Punkte sind
  echte lokale Minima. Welcher Zweig verfolgt wird, entscheidet aber sie; das
  steht so auch im Bericht.

Durch den brauchbaren Teil des Talpfads wird auf Wunsch eine Gerade gelegt.
Unbrauchbare Punkte fallen in drei Stufen heraus: Randminima, dann nur das
größte zusammenhängende Segment (Sprungerkennung), zuletzt iteratives Trimmen
abknickender Randpunkte – dasselbe Verfahren wie in
`Weighted_Optimization/lib/fit_waist_width_relation.py`, bewusst kopiert statt
importiert (jenes Modul setzt beim Import global `plt.rcParams`).

**Die Gerade gibt es nur über der µm-Achse** (effektiver Waist nach der
Linse). Über `win_input` (mm) ist der Zusammenhang gar nicht linear; über
width wäre es dieselbe Beziehung, nur andersherum aufgetragen – der
Einheitlichkeit halber ist sie auch dort gesperrt. Auf allen anderen Achsen
ist der Fit-Haken leer und grau, und „Gerade" lässt sich im Pfad-Dropdown
nicht wählen.

Im **Geradenmodus** läuft der Schnitt nicht entlang des (springenden)
Minimums, sondern entlang genau dieser Geraden, über den ganzen gescannten
Bereich – auch weit außerhalb der Punkte, aus denen sie bestimmt wurde (dort
extrapoliert, im Plot mit offenen Kreisen). Da die Gerade die Gitterpunkte
nicht trifft, werden die Werte zwischen den beiden Nachbarzeilen linear
interpoliert.

## Kohärenz: statische Interferenz der Spots

Die Töne sind untereinander **kohärent**. Ihre Kreuzterme laufen mit der
Differenzfrequenz um und mitteln sich in jeder Messung weg – **außer** bei
Paaren mit identischer Gesamtfrequenz. Beide AODs schieben die Lichtfrequenz,
ein Spot (n, m) trägt also f_s = f_x(n) + f_y(m); mit
f_x(n) = offset + width·n/(N_x−1) ist

```
f_s = 2·offset + width · ( n/(N_x−1) + m/(N_y−1) )
```

Zwei Spots sind entartet, wenn die Klammer übereinstimmt – das hängt **nur an
N_x und N_y**, nicht an offset oder width. Bei gleicher width auf beiden Achsen
gibt es deshalb immer mindestens ein entartetes Paar (die diagonal
gegenüberliegenden Eckspots); bei N_x = N_y ist jede Anti-Diagonale entartet.
Bei 3×4 ist es genau ein Paar. Der Kreuzterm dieser Paare liegt bei 0 Hz,
mittelt sich also nie weg und steht als **statisches Interferenzmuster** im
Bild:

```
<I> = Σ_s g_s²  +  Σ über entartete Paare  2 g_s g_t cos(Δφ)
```

Der erste Term ist die reine Intensitätssumme, wie sie das Projekt bis zum
2026-09-04 gerechnet hat; der zweite fehlte. Gerechnet wird mit Δφ = 0, also
**voll konstruktiv** – der ungünstigste Fall, und den soll ein Scan bewerten.
Gemessen an einem 3×4-Airy-Punkt (waist 1.1 µm, width 0.45 MHz) verschiebt das
die Uniformity um 1.6 % und den atom-gewichteten Crosstalk um 5.7 %.

Der Haken **„frequenzentartete Spots kohärent überlagern"** steht in beiden
Scan-Dialogen und ist **standardmäßig gesetzt** – die Interferenz ist da, ob
man sie mitrechnet oder nicht. Ohne Haken rechnet der Scan wie vorher.

Wo das steckt: `lib/coherence.py` (welche Spots entartet sind, plus die
Dialog-Gruppe), `lib/multitone_flattop_optimizer.py` (Felder und Kreuzterm).
Der Aufhänger ist `_profile_func()` – es liefert bei eingeschalteter Kohärenz
eine Profilfunktion mit unveränderter Signatur, die den Kreuzterm addiert.
Dadurch wirkt die Kohärenz überall gleich: Eigenintensität, Nachbarschaft und
lokales Sub-Gitter rufen dieselbe Funktion auf.

**Datensätze mit und ohne Kohärenz sind nicht vergleichbar.** Deshalb:

* Der Schlüssel `coherent` (und `n_degenerate_pairs`) steht in jeder neuen
  `.pkl`. Fehlt er, stammt der Datensatz aus der Zeit vor der Option und wurde
  inkohärent gerechnet.
* `run_plots.py` sagt es beim Laden, der Bericht schreibt es in die
  Scan-Parameter – bei fehlendem Schlüssel als deutlicher Hinweis.
* Ein Zwischenstand wird **nicht** fortgesetzt, wenn er mit dem anderen Modell
  gerechnet wurde; ein fehlender Schlüssel zählt dabei als „inkohärent" und
  nicht als „unbekannt", sonst entstünde ein halb kohärent gerechneter
  Datensatz.

## Verbotener Bereich (überlappende Eck-Spots)

Die beiden diagonal gegenüberliegenden Eck-Spots des N_x × N_y-Arrays dürfen
sich nicht überlappen. `width` ist die Gesamtspannweite des Tonarrays,
räumlich also eine Kantenlänge S; der Eckabstand ist √2·S. Die Bedingung
√2·S > k·waist ist in der (waist, width)-Ebene exakt eine Ursprungsgerade:

```
width/MHz > a * waist/µm      mit  a = k / (sqrt(2) * u)
```

`u` ist `width_to_um(1 MHz)` in µm/MHz aus der Optik des Datensatzes. Für die
Standard-Optik ist u = 6.3162 µm/MHz und damit a = 0.22390 MHz/µm bei k = 2.

Der Faktor k ist einstellbar: k = 2 heißt „die gaußäquivalenten Radien (1/e²)
berühren sich gerade"; wer die Airy-Hauptkeule bis zur ersten Nullstelle als
„den Spot" ansieht, nimmt k = 2·`airy_scale_factor`. Die GUI rechnet beide
Werte für den geladenen Datensatz vor.

Der Bereich kann nur eingezeichnet oder zusätzlich aus der Auswertung
ausgeschlossen werden. Da der Score punktweise definiert ist, ändert der
Ausschluss die Zahlen **nur im verbotenen Bereich**, nicht anderswo.

## Beschriftung und Farben

Die Metriken tragen überall den Index ihrer Familie: **U_h** und **η_h**
(hart maskiert). Das gilt in der Legende des Querschnitts, an den Colorbars, im
Titel der Score-Karte und im Bericht — beim Nebeneinanderlegen zweier PDFs
kommt so nie die Frage auf, welches jetzt welches war. Dieselbe Konvention
benutzt `Combinated_Optimization`, wo beide Familien in einem Plot vorkommen.

Die fünf Kurvenfarben des Querschnitts sind nicht nach Gefühl gewählt:
Umrechnung nach CIE-Lab, Abstand zusätzlich unter simulierter Deuteranopie und
Protanopie geprüft, Helligkeit auf L* ≤ 66 begrenzt.

| Größe | Farbe | |
|---|---|---|
| U_h | `#44AA99` | Petrol |
| η_h | `#CC3311` | Zinnober |
| J | `#000000` | Schwarz |
| r_x | `#1f77b4` | Blau (wie in den alten Amplituden-Schnitten) |
| r_y | `#ff7f0e` | Orange (dito) |

r_x und r_y behalten bewusst das Blau und Orange, das die Amplituden-Schnitte
seit jeher haben — dieselbe Größe sieht damit in beiden Plot-Arten gleich aus.
Vorher stand hier Blau für U und **Lila** (`#785EF0`) für r_x; die beiden
fallen unter Deuteranopie auf ΔE = 21.9 zusammen und waren auch normalsichtig
schwer zu trennen. Lila ist deshalb ganz raus. Der kleinste Abstand über alle
10 Paare liegt jetzt bei ΔE = 31.7 (η/r_y, die nie auf derselben Achse
liegen); die Paare, die im Bild tatsächlich übereinanderliegen, sind weit
auseinander: r_x/r_y bei 122/122/101 und U/η bei 108/56/46
(normalsichtig/deuteranop/protanop).

## Die Fit-Gerade in den Metrik-Karten

Wird sie eingezeichnet (Haken *Gerade auch in den Metrik-Vergleich
einzeichnen*), ist sie standardmäßig **eine durchgezogene Linie** über den
ganzen gescannten Bereich — eine Gerade ist eine Gerade, und aus welchem
Bereich sie bestimmt wurde, steht im Bericht.

Wer den Unterschied im Bild haben will, setzt zusätzlich den Haken *…
außerhalb des Fit-Bereichs gepunktet statt durchgezogen*: dann wird sie
zweiteilig gezeichnet, durchgezogen im gefitteten Bereich und gepunktet in der
Verlängerung. Der gepunktete Teil bekommt bewusst keinen eigenen
Legendeneintrag — die Legende der Karten soll genau einen Eintrag für die
Gerade haben.

(Technisch: `draw_fit_line_on_map(..., dashed_extrapolation=False)`. Die
Gerade in der Talschnitt-Karte ist davon unberührt — sie bleibt gestrichelt
und nur über dem Fit-Bereich gezeichnet, damit sie sich dort vom Talpfad
unterscheidet.)

## Karte des Talschnitts

Das linke Panel zeigt standardmäßig nur zwei Dinge über der Heatmap: den
**verbotenen Bereich** und die **Ausgleichsgerade**. Die einzelnen Pfadpunkte
— benutzte, ausgelassene, extrapolierte — werden dann gar nicht erst
gezeichnet und tauchen folglich auch nicht in der Legende auf; ihre Anzahl
steht im Bericht. Im Geradenmodus ist die Gerade dabei eine glatte Strecke
ohne Punktmarker.

Wer den Verlauf sehen will, setzt im Dialog den Haken *Talpfad und
ausgelassene/extrapolierte Punkte in die Karte zeichnen* — dann kommen Pfad,
Marker und die zugehörigen Legendeneinträge samt Anzahl zurück.

Eine Ausnahme gibt es: kommt gar keine Gerade zustande, bliebe die Karte sonst
leer. Dann wird der Talpfad auch ohne Haken gezeichnet.

## Plot-Stil

Alle PDFs dieses Ordners werden im LaTeX-Stil gesetzt: Serifenschrift,
Computer-Modern-Mathtext, `pdf.fonttype=42` (eingebettete TrueType- statt
Type-3-Schriften – Type 3 wird von vielen Journals abgelehnt). Der Stil wird
ausschließlich über `plt.rc_context` gesetzt, nie global, damit er nicht auf
andere Skripte abfärbt. Beschriftungen sind englisch; Dialog und Berichte
bleiben deutsch.

**Ein Maßstab für alle Bilder.** Die Schriftgrößen stehen genau einmal, und
zwar so, wie sie **im Dokument** ankommen sollen (`DOC_RC` in
`lib/report.py`, Grundschrift 9 pt bei A4-Textbreite = 16 cm = 6.3 Zoll).
Eine Figur, die breiter angelegt ist – der Talschnitt mit zwei Panels, der
Kreuzschnitt –, wird von `\includegraphics[width=\textwidth]` verkleinert;
`dokument_stil(figurbreite)` skaliert Schrift, Linienbreiten, Markergrößen
und Achsenabstände deshalb vorher um genau denselben Faktor hoch. Die
Einzeldatei sieht dadurch großschriftig aus, im Dokument stimmt es.

**Zweispaltige Figuren stehen eine Stufe kleiner.** Der Talschnitt (Karte +
Schnitt) und der Kreuzschnitt (zwei Schnitte) teilen sich die Textbreite auf
zwei Panels; jedes ist damit etwa halb so breit wie eine einzelne Karte. Ihre
Beschriftung wird deshalb mit `ZWEI_PANEL_DICHTE = 0.70` gesetzt – genauso,
wie in LaTeX eine Subfigure ihre Bildunterschrift kleiner setzt als die
Hauptabbildung. Das ist kein Geschmack: mit der vollen Dokumentgröße war die
Legende der Karte breiter als die Karte selbst und überragte den Plot, um den
es geht. Im Dokument sind es dort rund 7 pt statt 10 pt.

Gemessen über alle Plot-Arten der drei Ordner kamen vorher 4.7 bis 11.7 pt im
Dokument an (Faktor 2.5, weil die PDFs 6.2 bis 14.8 Zoll breit sind) – ein
Talschnitt war im Text unlesbar, eine Metrik-Karte größer als die
Grundschrift. Jetzt sind es überall 9.8 bis 10.2 pt, in Hard, Weighted und
Combined gleichermaßen; dieselbe Datei `lib/report.py` bzw. derselbe Block
sorgt dafür.

Zwei Folgen davon, die man beim Ansehen der Einzeldateien bemerkt: die
Colorbar des Talschnitts trägt nur noch das Symbol (`J`, `U_h`, `η_w`) statt
des ausgeschriebenen Namens – hochkant neben einer schmalen Leiste war der
lange Text höher als die Leiste –, und die Karte im Talschnitt beschriftet
ihre x-Achse kurz (`ω′ (µm)`); die lange Fassung steht weiterhin am Schnitt
daneben. Die Figurbreite des Talschnitts hängt außerdem **nicht** mehr an der
Zahl der y-Achsen: da die Schrift mitskaliert, änderte Breitermachen am Bild
im Dokument nichts mehr, es machte nur die Datei größer.

Die Karten sind auf A4-Textbreite (16 cm) gebaut – `\includegraphics[width=
\textwidth]` skaliert sie nicht mehr, die Schriftgrößen kommen also 1:1 im
Dokument an.

## Was beim Aufräumen (2026-09-02) geändert wurde

* Alle Hilfsmodule sind nach `lib/` gewandert; ausgeführt werden nur noch die
  `run_*.py`. Inhaltlich sind die verschobenen Module unverändert – nur
  `_default_dir()` zeigt jetzt eine Ebene höher, damit `Results/` und
  `Bilder/` weiterhin neben den Skripten liegen und nicht in `lib/`.
* `Winwidthscan startdialog.py` → `run_scan.py`,
  `Winwidthampscan startdialog.py` → `run_amp_scan.py` (der alte Name enthielt
  ein Leerzeichen und war damit nicht importierbar).
* **Zwei echte Fehler in `run_scan.py` behoben:** das Skript importierte
  `multitone_flattop_scan_plots`, eine Datei, die in diesem Ordner gar nicht
  existierte (nur in `Simulation_old/`) – der Dialog stürzte schon beim Import
  ab. Die dort benutzte Klasse steht jetzt als `FixedScanPlotter` im
  Plot-Modul dieses Ordners, als wortwörtliches Gegenstück zu
  `WeightedFixedScanPlotter`. Außerdem stand in `main()` ein hartes
  `import use_torch; use_torch.patch()` – ohne installiertes torch startete
  der Dialog nicht, und das Modul hieß „Use torch.py", war unter diesem Namen
  also ohnehin nie importierbar. Es heißt jetzt `lib/use_torch.py` und wird
  still übersprungen, wenn es nicht läuft.
* `run_plots.py` ist neu (vorher gab es nur `beispiel_amp_scan_ergebnisse_
  replotten.py` mit Konfig-Konstanten oben im Skript – das liegt jetzt in
  `Old_Scripts/`).
* Der Fest-Amplituden-Plot benutzt jetzt immer die µm-Achse und speichert nach
  `Bilder/` statt ins Arbeitsverzeichnis – dieselbe Konvention wie im
  Weighted-Ordner.
