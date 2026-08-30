# Bedienungsanleitung: Combinated_Optimization

Diese Anleitung richtet sich an jemanden, der die drei Skripte in diesem
Ordner benutzen soll, ohne sie geschrieben zu haben. Sie erklaert nur die
Bedienung. Was die Verfahren rechnen, steht in `README.md`; wie die
Ergebnisse zustande kommen, in den erzeugten Berichten.

---

## 1. Voraussetzungen

- Python mit `numpy`, `scipy`, `matplotlib` und `PyQt5`.
- Der Nachbarordner `Amplitudes/Weighted_Optimization/` muss vorhanden sein.
  Dort liegen der eigentliche Optimierer und die Plot-Klassen; dieser Ordner
  benutzt sie mit, statt sie zu kopieren. Fehlt er, brechen alle drei
  Skripte gleich beim Start mit einer klaren Meldung ab.
- Gestartet wird jedes Skript ganz normal, z.B. aus PyCharm heraus oder mit
  `python run_plots.py`. Es oeffnet sich immer erst ein Dialog; erst nach
  "OK" passiert etwas.

**PyCharm faerbt einige Importe rot** ("unresolved reference"). Das ist ein
reines Anzeigeproblem - der Ordner `Weighted_Optimization` wird erst zur
Laufzeit in den Suchpfad gehaengt. Dauerhaft weg bekommt man es mit
Rechtsklick auf `Weighted_Optimization` -> *Mark Directory as* -> *Sources
Root*.

---

## 2. Welches Skript brauche ich?

| Ich moechte ... | ausfuehren |
|---|---|
| einen **neuen Datensatz erzeugen** (dauert lange) | `run_penalty_scan.py` |
| zu einem vorhandenen **gewichteten** Scan den harten Fall nachrechnen | `run_hard_check.py` |
| einen **vorhandenen Datensatz auswerten**, Plots und Bericht erzeugen | `run_plots.py` |

Die Dateien in `lib/` werden nur benutzt, nicht direkt ausgefuehrt.

Alles landet in vier Unterordnern, die bei Bedarf automatisch angelegt
werden:

```
Results/       die Datensaetze (.pkl)
Fit_Plots/     die Plots (PDF, teils PNG)
Fit_Results/   die Berichte (Markdown)
Bilder/        Nebenausgaben des Optimierers
```

---

## 3. `run_penalty_scan.py` - neuen Datensatz erzeugen

Rechnet an jedem Punkt eines (Waist x Breite)-Gitters eine eigene
Amplituden-Optimierung. **Das dauert.** Bei 21x21 Punkten sind das 441
Optimierungen; je nach Rechner und Einstellungen Stunden bis Tage. Es wird
stuendlich zwischengespeichert (siehe *Speicherort*), ein abgebrochener Lauf
kann also fortgesetzt werden.

### Die Gruppen im Dialog

**Scan-Bereiche.** `win_input` ist der Strahlradius **vor** den Linsen in mm,
`width` die Breite in MHz. "Gitterpunkte pro Achse" gilt fuer beide Achsen -
21 bedeutet 21x21 = 441 Punkte. Fuer einen ersten Versuch 10 bis 12 nehmen;
damit sieht man in ueberschaubarer Zeit, ob die Bereiche ueberhaupt sinnvoll
liegen.

**Amplituden-Optimierung.** `r_x`/`r_y` sind die Amplituden-Verhaeltnisse, die
pro Gitterpunkt gesucht werden; min/max begrenzen die Suche. *Wichtig:* wird
der Bereich zu eng gewaehlt, klemmen viele Punkte an der Schranke und sind
dann keine echten Optima mehr - der spaetere Bericht sagt nichts darueber, man
muss es selbst im Blick behalten. `alpha` gewichtet Uniformity gegen
Crosstalk: 0.7 heisst 70 % Uniformity, 30 % Crosstalk.

**Penalty-Term und Region.** `combo_lambda` bestraft, wenn hartes und
atom-gewichtetes Kriterium auseinanderlaufen. `combo_percentile` legt fest,
welcher Anteil der besten Punkte spaeter als "gut" gilt (25 % = das beste
Viertel).

**Intensitaetsgitter.** `n_grid` ist die Aufloesung des globalen Gitters, auf
dem die *harten* Metriken ausgewertet werden. Der Aufwand waechst mit
`n_grid^2`. 1000 ist der uebliche Wert.

**Atom-Gewichtung.** Temperatur und Fallenfrequenz bestimmen, wie breit die
Atomverteilung ist, ueber die der *gewichtete* Teil mittelt.
`weighted_n_grid` ist die Aufloesung des lokalen Sub-Gitters dafuer.

**Parallelisierung und GPU.** `n_jobs` verteilt die Gitterpunkte auf
Prozesse. Voreingestellt ist "alle Kerne minus einer", damit der Rechner
bedienbar bleibt. Die GPU wird automatisch gesucht; "GPU-Erkennung
ueberspringen" erzwingt CPU.

**Speicherort.** Der Pfad wird **vor** dem Scan festgelegt, nicht danach.
Dorthin wird stuendlich zwischengespeichert und am Ende das Endergebnis
geschrieben. Liegt dort schon ein passender Zwischenstand, meldet das Skript
vor dem Start, dass es fortsetzt, und wie viele Punkte schon fertig sind.
Passt die Datei nicht zu den eingestellten Parametern, wird sie beim naechsten
Speichern ueberschrieben - dann also besser einen neuen Namen waehlen.

**"Nach dem Scan direkt auswerten"** erzeugt anschliessend die Plots und den
Bericht, ohne dass man `run_plots.py` extra starten muss.

### Wenn etwas schiefgeht

Der Fortschritt laeuft ueber ein Fortschrittsfenster; die Details stehen auf
der Konsole. Ein abgebrochener Lauf ist nicht verloren - dasselbe Skript mit
denselben Parametern und demselben Speicherort erneut starten.

---

## 4. `run_hard_check.py` - harten Fall nachrechnen

Nimmt einen bereits vorhandenen **gewichteten** Amplituden-Scan
(`scan_amp_data_weighted_*.pkl`) und wertet an jedem Gitterpunkt mit den dort
schon gefundenen Amplituden **einmal** die harten Metriken aus. Es wird nichts
neu optimiert, das dauert Sekunden bis Minuten. Die Eingangsdatei bleibt
unveraendert.

Die Frage, die das beantwortet: bleiben die Punkte, die unter dem
atom-gewichteten Ziel gut sind, auch unter dem harten Ziel gut?

**Gewichteter Amplituden-Scan (Eingabe).** Datei waehlen; das Skript prueft
sofort, ob sie ueberhaupt passt, und zeigt eine Kurzinfo an. Eine falsche
Datei wird mit einer klaren Meldung abgelehnt.

**Harte Nachrechnung.** `n_grid` wie oben - die Aufloesung des globalen
Gitters fuer die harte Auswertung.

**Vergleich.** `alpha`, `good_percentile` und `combo_lambda` bestimmen, wie
"gut" definiert wird und wie hart und gewichtet verrechnet werden. Die Werte
werden aus der Eingangsdatei vorbelegt.

**Speicherort** und **"direkt auswerten"** wie beim Scan.

Ergebnis: eine `hard_check_*.pkl` plus, auf Wunsch, Plots und Bericht. Die
Kernkennzahl steht ganz oben im Bericht: wie viel Prozent der gewichtet-guten
Punkte auch hart gut sind.

---

## 5. `run_plots.py` - auswerten und plotten

Das Skript, das man am haeufigsten braucht. Es aendert **nie** einen
Datensatz, sondern liest nur.

### Datensatz

Oben aus der Liste waehlen - sie zeigt alle `.pkl`-Dateien aus `Results/`,
neueste zuerst. Es ist bewusst nichts vorausgewaehlt, damit nie versehentlich
der falsche Datensatz ausgewertet wird. "Aktualisieren" liest den Ordner neu
ein (z.B. nach einem gerade fertig gewordenen Scan), "Andere Datei..." holt
etwas von ausserhalb.

Welche Art Datensatz vorliegt, erkennt das Skript selbst und sagt es an. Ein
rein gewichteter Scan gehoert nicht hierher, sondern zuerst durch
`run_hard_check.py`.

Wer immer denselben Datensatz auswertet, kann ihn oben im Skript bei
`PKL_DATEI` eintragen; dann ist er beim Start schon ausgewaehlt.

### Darstellung

- **Waist-Achse**: `win_input` in mm vor der Linse, oder der effektive Waist
  in µm nach der Linse. Betrifft nur die Darstellung, nicht die Rechnung.
- **Schriftgroesse Legenden**: bei vielen Kurven kleiner stellen.
- **Besten Punkt als Stern einzeichnen**: markiert in den Karten den
  Gitterpunkt mit dem kleinsten Score. Seine Zahlen stehen ohnehin im
  Bericht - ohne Haken bleiben die Heatmaps voellig frei.
- **Gerade auch in den Metrik-Vergleich einzeichnen**: siehe Abschnitt
  *Talschnitt*; die dort gefittete Gerade wird zusaetzlich in die vier
  Metrik-Karten gelegt.
- **Amplituden-Uebersicht und Schnitte mitzeichnen**: die 6-Panel-Uebersicht
  und die Schnitte des AmplitudeScanPlotter, als PNG. Kostet etwas Zeit.
- **Plots zusaetzlich interaktiv anzeigen**: oeffnet jedes Bild in einem
  Fenster. Praktisch zum Hineinzoomen, laestig bei vielen Plots.

### Querschnitt entlang des Minimums ("Talschnitt")

Das ist die eigentliche Auswertung entlang einer Linie durch das Gitter.

- **Schnitt entlang** - *Talpfad* folgt dem Minimum: pro Waist-Spalte (bzw.
  width-Zeile) wird der Punkt gesucht, an dem die Fuehrungsgroesse am
  kleinsten ist. Echte Gitterwerte, aber der Pfad springt dort, wo das
  Minimum flach ist oder aus dem Scan-Fenster laeuft. *Gerade* folgt
  stattdessen der Ausgleichsgeraden durch diesen Talpfad, ueber den ganzen
  gescannten Bereich - auch weit ausserhalb der Punkte, aus denen sie
  bestimmt wurde (dort extrapoliert, im Plot mit offenen Kreisen markiert).
- **Groesse fuer Talpfad/Gerade** - welcher Groesse gefolgt wird. Was der
  Datensatz nicht hergibt, ist ausgegraut.
- **Aufgetragen ueber** - die x-Achse des Schnitts.
- **Gerade durch den Talpfad legen** - legt eine Ausgleichsgerade durch den
  brauchbaren Teil des Talpfads. Unbrauchbare Punkte werden automatisch
  ausgeschlossen: erst Minima am Rand des Scan-Fensters (das sind keine
  echten Minima), dann ein abgesetzter Nebenzweig, zuletzt abknickende
  Randpunkte. Sie erscheinen im Plot als offene Kreise und werden nicht von
  der Linie verbunden.
- **Welche Groessen** - die sieben Haken darunter. Je mehr, desto mehr
  y-Achsen; Kurven gleicher Einheit und Groessenordnung teilen sich eine.

**Zwei Regeln, die oft ueberraschen:**

1. Die Gerade gibt es **nur** fuer die Achse "Waist nach der Linse (µm)".
   Nur dort ist der Zusammenhang zwischen width und Waist linear. Bei den
   anderen beiden Achsen ist der Haken grau und "Gerade" laesst sich im
   ersten Dropdown nicht waehlen.
2. Im Geradenmodus ist der Fit-Haken gesetzt und gesperrt - die Gerade ist
   dort ja der Schnitt selbst.

Laesst sich fuer die gewaehlte Groesse keine Gerade legen (zu wenige
brauchbare Talpunkte), sagt der Dialog das **vor** dem Start, statt mitten in
der Auswertung abzubrechen.

### Score und Region neu berechnen (optional)

Ohne Haken werden die im Datensatz gespeicherten Werte benutzt. Mit Haken
werden Score und Region aus den vorhandenen Grids **neu** berechnet - mit
anderem `alpha`, `combo_lambda` oder Perzentil, ohne den teuren Scan zu
wiederholen. Der Datensatz selbst bleibt unveraendert; auf Wunsch wird die
neu berechnete Fassung als eigene Datei mit dem Zusatz `_recombined`
gespeichert.

### Ueberschreiben

Ohne den Haken ganz unten werden gleichnamige Plots desselben Tages ohne
Rueckfrage ueberschrieben - sie sind aus dem Datensatz jederzeit
reproduzierbar. Mit Haken wird nachgefragt, **aber ueber die Konsole**, nicht
ueber ein Fenster. Wenn das Programm scheinbar haengt: in die Konsole
schauen.

---

## 6. Was am Ende herauskommt

Nach "Auswertung erzeugen" meldet ein Fenster, wohin geschrieben wurde. In
`Fit_Plots/` liegen dann, je nach Datensatz-Art und Einstellungen:

| Datei | Inhalt |
|---|---|
| `..._metric_comparison.pdf` | vier Karten: Uniformity und Crosstalk, je hart und gewichtet |
| `..._region.pdf` | die Score-Karte |
| `..._agreement.pdf` | nur Hard-Check: wo hart und gewichtet einig sind |
| `..._score_scatter.pdf` | nur Hard-Check: gewichtet gegen hart, je Gitterpunkt |
| `..._valley_...pdf` | Querschnitt entlang des Minimums |
| `..._line_...pdf` | Querschnitt entlang der Geraden |

Der Dateiname beginnt mit `PenaltyRegion_` oder `HardCheck_`, dann Tonanzahl,
Gitterpunkte, Strahlprofil und Datum. Der Bericht mit allen Zahlen -
Steigung und R² der Geraden, Region-Grenzen, bester Punkt, Scan-Parameter -
liegt als `..._Report.md` in `Fit_Results/`.

Die PDFs sind fuer einen LaTeX-Satz gemacht: englische Beschriftung,
Serifenschrift, eingebettete TrueType-Schriften. Sie lassen sich unveraendert
in ein Dokument einbinden.

---

## 7. Haeufige Stolpersteine

**"In Results/ liegen keine .pkl-Dateien."** Erst einen Scan laufen lassen
oder ueber "Andere Datei..." einen Datensatz von anderswo waehlen.

**"Diese Datei passt nicht zu diesem Ordner."** Ein rein gewichteter Scan
muss zuerst durch `run_hard_check.py`.

**Der Dialog ist hoeher als der Bildschirm.** Ist er nicht - der
Parameterbereich scrollt, die Buttons bleiben unten immer sichtbar.

**Der Scan scheint zu haengen.** Ein Gitterpunkt kann Minuten dauern. Die
Konsole zeigt den Fortschritt; stuendlich wird gespeichert.

**Ein Plot sieht zackig aus.** Die *harten* Metriken haben ein
Diskretisierungsrauschen, das die *gewichteten* nicht haben - der Grund steht
in `README.md`. Es ist echt und kein Darstellungsfehler; die Achsen sind so
skaliert, dass es nicht wie ein Signal aussieht.
