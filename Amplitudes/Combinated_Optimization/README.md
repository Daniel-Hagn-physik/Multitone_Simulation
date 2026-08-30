# Combinated_Optimization

**Wer die Skripte nur benutzen will, findet die Bedienung Schritt fuer
Schritt in `ANLEITUNG.md`.** Dieses Dokument beschreibt, was die Verfahren
rechnen und wie der Code aufgebaut ist.

Drei Skripte zum Ausfuehren, ein Ordner `lib/` mit dem, was sie benutzen.

## Was will ich gerade?

| Ich moechte ... | ... dann dieses Skript ausfuehren |
|---|---|
| **neue Daten erzeugen**, bei denen die Amplituden gemeinsam auf hart + gewichtet optimiert werden (Penalty-Methode) | `run_penalty_scan.py` |
| pruefen, **ob mein vorhandener Weighted-Datensatz auch im Hard Case gut ist** | `run_hard_check.py` |
| **vorhandene Datensaetze plotten** und den Bericht (neu) erzeugen | `run_plots.py` |

Alle drei oeffnen einen Dialog, in dem die Parameter stehen. Nichts muss
im Code geaendert werden.

**Welchen Datensatz?** `run_plots.py` und `run_hard_check.py` zeigen im
Dialog ein Auswahlfeld mit allen passenden Dateien aus `Results/` (neueste
zuerst) - es wird bewusst **nichts** automatisch vorausgewaehlt, damit nie
versehentlich der falsche Datensatz ausgewertet wird. Wer immer denselben
nimmt, traegt ihn ganz oben im jeweiligen Skript ein:

```python
PKL_DATEI    = "scan_amp_data_combined_N3x4_21x21pts_Airy.pkl"   # run_plots.py
WEIGHTED_PKL = "scan_amp_data_weighted_N3x4_10x10pts_Airy.pkl"   # run_hard_check.py
```

Dateiname genuegt (wird in `Results/` gesucht), ein vollstaendiger Pfad geht
auch. "Andere Datei..." waehlt eine von ausserhalb, "Aktualisieren" liest die
Liste neu ein (z.B. nach einem gerade fertig gewordenen Scan).

---

## 1. `run_penalty_scan.py` - neue Daten mit der Penalty-Methode

An JEDEM (win_input, width)-Gitterpunkt laeuft GENAU EINE
(r_x, r_y)-Optimierung, die direkt gegen die Kombination aus hartem und
atom-gewichtetem Ziel minimiert:

```
U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
J       = alpha*U_kombi + (1-alpha)*C_kombi        ->  min ueber (r_x, r_y)
```

`combo_lambda*|Differenz|` ist der **Penalty-Term**: er bestraft
Amplituden, bei denen das harte und das atom-gewichtete Kriterium
auseinanderlaufen. Hart und gewichtet werden bei jedem Optimierungsschritt
am SELBEN (r_x, r_y) ausgewertet - die gefundene Amplitude gilt daher fuer
beide zugleich.

- Ergebnis: `Results/scan_amp_data_combined_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl`
  (unveraendertes Namensmuster - die bereits vorhandenen Datensaetze passen dazu)
- Speicherort wird VOR dem Start abgefragt, stuendliche Zwischenspeicherung,
  automatisches Fortsetzen nach einem Abbruch
- am Ende optional direkt Plots + Bericht

## 2. `run_hard_check.py` - passt der Hard Case zu meinem Weighted?

Eingabe ist ein bereits vorhandener gewichteter Amplituden-Scan
(`scan_amp_data_weighted_*.pkl` aus `Weighted_Optimization/Results/`).
An jedem Gitterpunkt werden win_input, width und die dort GEFUNDENEN
Amplituden r_x/r_y genommen und damit **genau einmal** die harten Metriken
ausgewertet.

> Keine erneute Optimierung. Keine Kombination zweier Datensaetze.
> Nur: dieselben Amplituden, einmal hart nachgerechnet.

Deshalb ist der Lauf billig (Sekunden bis wenige Minuten) und braucht
keine Zwischenspeicherung. Die Eingangsdatei wird nicht veraendert.

Ausgewertet wird zweifach:

- **Vierfeldertafel**: wie viele der unter dem gewichteten Ziel guten
  Punkte sind auch unter dem harten Ziel gut? Dazu Pearson-Korrelationen
  von Score, Uniformity und Crosstalk.
- **Consistency-Score/Region**: dieselbe Penalty-Kombination wie oben,
  hier als raeumlich zusammenhaengende Karte der Uebereinstimmung.

Ergebnis: `Results/hard_check_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl`
(eigenes Namensmuster, nie mit den Penalty-Datensaetzen zu verwechseln).

## 3. `run_plots.py` - vorhandene Datensaetze auswerten

Erkennt automatisch, welche Art Datensatz vorliegt, und erzeugt in
`Fit_Plots/` bzw. `Fit_Results/`:

| Datei | Penalty-Scan | Hard-Check |
|---|---|---|
| `..._metric_comparison.pdf` (hart vs. gewichtet, 2x2) | x | x |
| `..._region.pdf` (Score-Karte, auf Wunsch mit bestem Punkt) | x | x |
| `..._agreement.pdf` (Uebereinstimmungs-Karte, 4 Kategorien) | - | x |
| `..._score_scatter.pdf` (gewichtet vs. hart je Gitterpunkt) | - | x |
| `..._valley_{X}_over_{Y}.pdf` (Querschnitt entlang des Minimums) | x | x |
| `..._line_{X}_over_{Y}.pdf` (Querschnitt entlang der Geraden) | x | x |
| `..._Report.md` (alle Kennzahlen) | x | x |
| 6-Panel-Uebersicht + Schnitte (PNG, optional) | x | x |

**Querschnitt entlang des Minimums ("Talschnitt").** Man waehlt eine
Fuehrungsgroesse `X` (combined_score, Uniformity oder Crosstalk - hart oder
atom-gewichtet) und eine Schnittachse `Y` (Waist in µm nach der Linse,
win_input in mm vor der Linse, oder width in MHz). Pro Spalte (bzw. Zeile)
wird der Punkt gesucht, an dem `X` minimal ist; **genau an diesen Punkten**
werden dann alle angehakten Groessen abgelesen - also nicht deren eigenes
Minimum, sondern ihr Wert dort, wo die Fuehrungsgroesse am besten ist. Der
Plot zeigt links die Heatmap von `X` mit dem eingezeichneten Talpfad, rechts
den Querschnitt mit einer eigenen y-Achse je Kurve. Minima, die am Rand des
gescannten Fensters liegen (also vermutlich gar keine echten Minima sind),
werden als offene Kreise markiert und im Titel gezaehlt.

**Die Gerade in den Metrik-Karten.** Der Haken "Gerade auch in den
Metrik-Vergleich einzeichnen" in der Gruppe *Darstellung* zeichnet dieselbe
Gerade zusaetzlich in alle vier Karten von `..._metric_comparison.pdf` -
durchgezogen im gefitteten Bereich, gepunktet in der Extrapolation, auf den
gescannten width-Bereich beschnitten. Welche Groesse gefittet wird, bestimmt
"Groesse fuer Talpfad/Gerade" in der Talschnitt-Gruppe. Die Gerade ist immer
die ueber dem effektiven Waist in µm; auf der mm-Achse erscheint sie deshalb
leicht gekruemmt, weil win_input und effektiver Waist nichtlinear
zusammenhaengen (`report.line_points_for_axis` tastet sie dort dicht ab).
Laesst sich keine Gerade legen, bleiben die Karten unveraendert und es kommt
ein Hinweis auf die Konsole. Die Legende der vier Karten steht **einmal unter
der ganzen Figur** statt vier Mal mitten in den Heatmaps - dort verdeckten
die Kaesten sonst genau den Bereich, um den es geht.

Der LaTeX-Stil haengt uebrigens nicht mehr daran, ob ueber `make_all()`
aufgerufen wird: jede Plot-Funktion traegt den Dekorator `@_mit_stil`, der
`LATEX_STYLE` per `rc_context` setzt. Auch ein direkter Aufruf von
`plot_metric_comparison()` liefert damit Serifen, Computer-Modern-Mathtext
und Type-42-Schriften.

**Gerade durch den Talpfad (optional, Haken im Dialog).** Nur verfuegbar,
wenn als Achse **"Waist nach der Linse (µm)"** gewaehlt ist - nur dort ist der
Zusammenhang zwischen width und Waist linear. Bei "win_input vor der Linse
(mm)" haengen win_input und effektiver Waist nichtlinear zusammen, eine Gerade
waere dort ueber einen schmalen Bereich zwar gut angepasst, aber physikalisch
bedeutungslos; ueber "width (MHz)" ist sie ebenfalls gesperrt. Bei diesen
beiden Achsen ist der Haken leer und grau, und "Gerade" laesst sich im
Pfad-Dropdown nicht waehlen. Im vorderen Bereich
laeuft der Talpfad meist sichtbar gerade; weiter hinten faellt das Minimum auf
den Rand des gescannten Fensters oder springt auf einen zweiten Zweig. Eine
Gerade durch ALLE Talpunkte waere dadurch verfaelscht, deshalb werden die
unbrauchbaren Punkte in drei Stufen ausgeschlossen - dieselbe Logik wie in
`Weighted_Optimization/fit_waist_width_relation.py`:

1. Minima am Rand des gescannten Fensters (offene Kreise im Plot),
2. ein abgesetzter Nebenzweig: es bleibt nur das groesste zusammenhaengende
   Segment des Talverlaufs (Sprungerkennung ueber die Streuung der Schritte),
3. Rand-Kinks: die beiden Enden werden getrimmt, solange der Randpunkt
   deutlich neben der Ausgleichsgeraden liegt.

Die in Stufe 2/3 verworfenen Punkte erscheinen im Plot als schwarze Kreuze,
die Geradengleichung samt R² steht im Bericht (bewusst nicht in der Legende -
dort wuerde sie den Kasten nur aufblaehen), und der Bericht bekommt
einen eigenen Abschnitt mit Steigung, Achsenabschnitt, R², gefittetem Bereich
und der Zahl der verwendeten bzw. ausgeschlossenen Punkte. Bleiben weniger als
vier brauchbare Punkte uebrig, wird keine Gerade gezeichnet und der Bericht
sagt, warum.

**Schnitt entlang des Minimums oder entlang der Geraden.** Im Dialog waehlbar
("Schnitt entlang: Talpfad / Gerade"):

- **Talpfad** - wie oben beschrieben: echte, gerechnete Gitterwerte, aber der
  Pfad springt dort, wo das Minimum flach ist oder aus dem Scan-Fenster
  laeuft. Datei `..._valley_...pdf`.
- **Gerade** - der Schnitt folgt der gefitteten Geraden ueber den GANZEN
  gescannten Bereich, also auch weit ausserhalb der Punkte, aus denen sie
  bestimmt wurde. Dieser extrapolierte Teil ist im Plot mit offenen Kreisen
  markiert und im Titel gezaehlt. Da die Gerade die Gitterpunkte nicht trifft,
  werden die Werte pro Spalte (bzw. Zeile) linear zwischen den beiden
  benachbarten Gitterwerten interpoliert; wo die Gerade das Scan-Fenster ganz
  verlaesst, gibt es keine Daten und die Spalte faellt heraus (ebenfalls
  gezaehlt). Zum Vergleich bleibt der echte Talpfad blass im Bild. Datei
  `..._line_...pdf`.

Im Geradenmodus wird die Gerade immer bestimmt - sie ist ja der Schnitt
selbst; der Haken "Gerade durch den Talpfad legen" ist dort gesetzt und
gesperrt. Laesst sich fuer die gewaehlte Groesse keine Gerade legen, sagt der
Dialog das VOR dem Start, statt mitten in der Auswertung abzubrechen.

**Aussehen der PDFs.** Alle Plots dieses Ordners sind auf einen LaTeX-Satz
ausgelegt: englische Beschriftungen, knappe Titel, Serifenschrift mit
Computer-Modern-Mathtext (`report.LATEX_STYLE`) und `pdf.fonttype=42`, also
eingebettete TrueType- statt Type-3-Schriften - Type 3 wird von vielen
Journals und von `pdffonts`-Pruefungen abgelehnt. Der Stil wird nur ueber
`plt.rc_context` gesetzt und faerbt deshalb nicht auf andere Skripte ab.
Berichte (`.md`) und der Dialog bleiben deutsch.

Im Querschnitt bekommt nicht mehr jede Kurve eine eigene y-Achse: Kurven mit
derselben Einheit und derselben Groessenordnung teilen sich eine
(`report.group_traces_by_axis`), und `r_x`/`r_y` liegen immer zusammen. Eine
Achse nimmt nur so lange weitere Kurven auf, wie jede von ihnen noch
mindestens `AXIS_GROUP_MIN_SHARE` (10 %) der Achsenspanne fuer sich
beansprucht - sonst waere eine schwach schwankende Groesse nur noch eine
gerade Linie.

**Ausnahme fuer verrauschte Kurven.** Die harten Metriken haben ein
saegezahnfoermiges Diskretisierungsrauschen (das globale Intensitaetsgitter
wird pro Scanpunkt neu aufgespannt, die Maskengrenzen springen dabei um ganze
Pixel; die atom-gewichteten Metriken haben es nicht, ihr Sub-Gitter haengt an
der Falle). Bei Uniformity (hart) ist dieses Zickzack rund ein Drittel der
eigenen Spanne. Bekaeme die Kurve nach der 10-%-Regel eine enge eigene Achse,
wuerde das Rauschen die ganze Achse fuellen und wie ein Signal aussehen.
Deshalb duerfen Kurven, deren Spanne weniger als `AXIS_NOISE_SNR` (5) mal ihr
Zickzack betraegt, sich einer groesseren Achse anschliessen. Bleibt eine
solche Kurve doch allein, weitet `_entzerre_achse()` den Achsenbereich so
auf, dass das Zickzack hoechstens `AXIS_NOISE_TARGET` (25 %) der Achsenhoehe
einnimmt - es wird nichts abgeschnitten, die Kurve nur kleiner gezeichnet.
Aus sieben Kurven werden so typischerweise vier Achsen.

Punkte, die nicht in die Auswertung eingehen (Minimum am Rand des
Scan-Fensters, vom Fit ausgeschlossen), werden in der Karte als offene Kreise
gezeigt und **nicht** von der Pfadlinie verbunden; im Querschnitt reisst die
Kurve dort ab, und die Achsenskalierung richtet sich nur nach den benutzten
Punkten.

`alpha`, `combo_lambda` und das Perzentil lassen sich hier neu setzen -
Score, Region und die Vierfeldertafel werden dann aus den vorhandenen
Grids neu berechnet, **ohne** den teuren Scan zu wiederholen.

---

## Ordner

```
Combinated_Optimization/
    run_penalty_scan.py     <- ausfuehren: neue Daten (Penalty)
    run_hard_check.py       <- ausfuehren: Hard Case zu vorhandenem Weighted
    run_plots.py            <- ausfuehren: plotten/auswerten
    lib/                    <- wird von den drei Skripten benutzt,
                               nicht direkt ausfuehren
        paths.py            Ordner-Konstanten, Anbindung an Weighted_Optimization
        combine.py          Penalty-Kombination, Region, Laden/Speichern
        penalty_scan.py     die gemeinsame Amplituden-Optimierung
        hard_check.py       harte Nachrechnung + Konsistenz-Analyse
        report.py           Plots und Markdown-Berichte
    Results/                gespeicherte Datensaetze (.pkl)
    Fit_Plots/              Vektor-PDFs der Auswertung
    Fit_Results/            Markdown-Berichte
    Bilder/                 PNG-Ausgaben
    Old_Combine/            Archiv des verworfenen GETRENNTEN Verfahrens
```

Der eigentliche Optimierer (`MultitoneFlatTopOptimizer`), die Plot-Klassen,
`scan_checkpoint` und `perf_log` liegen unverändert in
`../Weighted_Optimization/` und werden von hier aus mitbenutzt, nicht
dupliziert.

## PyCharm markiert Importe rot?

Die Module aus `../Weighted_Optimization/` werden erst zur **Laufzeit** über
einen `sys.path`-Eintrag gefunden. PyCharms statische Analyse kennt diesen
Eintrag nicht und färbt solche Importe rot ("unresolved reference") — **der
Code läuft trotzdem**, es ist ein reines Anzeigeproblem.

Zwei Dinge dagegen:

1. Alle Importe aus `Weighted_Optimization` sind in `lib/` gebündelt. Die
   drei Skripte, die du täglich öffnest, importieren nur noch aus `lib` —
   dort ist also kein Rot mehr.
2. Dauerhaft weg bekommst du es mit einem einmaligen Handgriff: Rechtsklick
   auf den Ordner **`Weighted_Optimization`** → **"Mark Directory as"** →
   **"Sources Root"**. Danach kennt PyCharm die Module und färbt sie normal
   ein — auch in `lib/`.

## Was hier bewusst NICHT mehr drin ist

Das **getrennte** Verfahren (hart und gewichtet je einzeln optimieren,
danach die an verschiedenen r_x/r_y erreichten Metriken verrechnen) liegt
unveraendert im Archivordner `Old_Combine/` und wird nicht mehr angeboten -
es war rechnerisch nicht schluessig, weil die beiden Optimierungen am
selben Gitterpunkt unterschiedliche Amplituden liefern.

## Verifikation

Die Auswertung wurde gegen den vorhandenen Datensatz
`scan_amp_data_combined_N3x4_21x21pts_Airy.pkl` geprueft: normierte Grids,
`combined_score`, Schwellwert, Region-Maske, Region-Rechteck und bester
Punkt stimmen bit-exakt mit dem frueher erzeugten Ergebnis ueberein
(win_input = 1.7000 mm, width = 0.2300 MHz, r_x/r_y = 0.9997/1.1869,
Region 111/441 Punkte, waist 0.7872..0.9986 µm, width 0.22..0.25 MHz).
