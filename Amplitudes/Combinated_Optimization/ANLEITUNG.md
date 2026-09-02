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
| **einen einzelnen Parametersatz suchen**: ein paar Groessen vorgeben, den Rest optimieren lassen | `run_penalty_only.py` |

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
- **Bester Punkt nach**: nach welcher Groesse der Stern gesetzt wird.
  Voreingestellt ist der im Datensatz gespeicherte Punkt - dann zeigen Stern
  und Bericht dasselbe. Waehlt man etwas anderes, erklaert der Bericht das in
  einem eigenen Abschnitt. **Liegt der Punkt am Rand des gescannten
  Fensters, ist der Stern offen statt gefuellt** - dort ist das Minimum
  vermutlich nur das Ende des Scans. Beim rohen J ist genau das der Fall:
  sein Minimum sitzt in beiden vorhandenen Datensaetzen auf width =
  0.200 MHz.
- **Eigener Punkt**: die letzten beiden Eintraege des Dropdowns. Du gibst
  eine Koordinate vor - Waist in µm oder width in MHz -, die andere kommt aus
  der Talpfad-Geraden. Das Feld darunter nimmt den Wert entgegen und
  beschriftet sich passend. Der Punkt liegt exakt auf der Geraden, also meist
  zwischen zwei Gitterpunkten; der Bericht nennt zusaetzlich den
  naechstgelegenen wirklich gerechneten Gitterpunkt. Ohne Talpfad-Gerade geht
  es nicht - dann kommt ein Hinweis auf der Konsole und es wird kein Punkt
  gezeichnet.
  Im Bericht stehen zu dem Punkt auch die Werte der sechs Groessen (U und
  Crosstalk je hart und gewichtet, r_x, r_y) und J - einmal interpoliert und
  einmal am naechstgelegenen Gitterpunkt. Die Interpolation ist bei r_x/r_y
  mit Vorsicht zu geniessen: das sind Optimierungsergebnisse, keine glatten
  Funktionen. Exakte Werte an einem freien Punkt liefert nur eine neue
  Optimierung dort, also `run_penalty_only.py`.
- **Gerade auch in den Metrik-Vergleich einzeichnen**: siehe Abschnitt
  *Talschnitt*; die dort gefittete Gerade wird zusaetzlich in die vier
  Metrik-Karten gelegt, als EINE durchgezogene Linie ueber den ganzen
  gescannten Bereich. Aus welchem Bereich sie bestimmt wurde, steht im
  Bericht. Die Legende der Figur hat genau einen Eintrag ("Linear model
  fit"); der Stern bekommt keinen.
- **... ausserhalb des Fit-Bereichs gepunktet statt durchgezogen**: nur wer
  den Unterschied im Bild sehen will, setzt diesen Haken - dann wird die
  Gerade zweiteilig gezeichnet, durchgezogen im gefitteten Bereich und
  gepunktet in der Verlaengerung. Der gepunktete Teil bekommt keinen eigenen
  Legendeneintrag.
- **Talpfad und ausgelassene/extrapolierte Punkte in die Karte zeichnen**
  (Gruppe *Talschnitt*): ohne Haken zeigt die Karte im linken Panel des
  Querschnitts nur den verbotenen Bereich und die Ausgleichsgerade. Die
  einzelnen Pfadpunkte werden dann gar nicht erst gezeichnet, also auch nicht
  in der Legende gefuehrt; ihre Anzahl steht im Bericht. Ausnahme: kommt gar
  keine Gerade zustande, waere die Karte sonst leer - dann wird der Talpfad
  auch ohne Haken gezeichnet.
- **Metrik-Vergleich zusaetzlich mit Amplituden (6 Karten, eigene PDF)**:
  schreibt neben `..._metric_comparison.pdf` eine zweite Datei
  `..._metric_comparison_amp.pdf` mit denselben vier Metrik-Karten plus `r_x`
  und `r_y`. Die beiden Amplituden-Karten teilen sich eine logarithmische
  Farbskala; Punkte, deren Amplitude auf einer `r_bounds`-Schranke klemmt,
  sind grau - dort steht kein freies Optimum, sondern die Schranke, und ihr
  Anteil erscheint auf der Konsole. Nicht zu verwechseln mit dem naechsten
  Haken.
- **Amplituden-Uebersicht und Schnitte mitzeichnen**: die 6-Panel-Uebersicht
  und die Schnitte des AmplitudeScanPlotter, als PNG. Kostet etwas Zeit.
- **Plots zusaetzlich interaktiv anzeigen**: oeffnet jedes Bild in einem
  Fenster. Praktisch zum Hineinzoomen, laestig bei vielen Plots.

**Welche Kurven im Schnitt stehen.** Voreingestellt sind alle ausser dem
NORMIERTEN Score: im Querschnitt steht `J` (Penalty ROH), also die Groesse,
die der Scan an jedem Gitterpunkt tatsaechlich minimiert hat. Der normierte
Score `S` bleibt anhakbar - er ist praktisch zum Sortieren, aber nichts, was
man physikalisch deuten kann, weil die Normierung jede Rohgroesse gitterweit
auf 0..1 zieht. `J` traegt eine Einheit und wird deshalb NICHT in Prozent
gezeigt, sondern roh (Werte um 0.04 statt 4 %) - genau wie im Bericht.

**Talpunkt-Auswahl** - der wichtigste Schalter der Gruppe:

- *Lokales Minimum nahe einer Leitgeraden* (Voreinstellung): pro Spalte wird
  nicht das kleinste, sondern das der Leitgeraden naechstliegende LOKALE
  Minimum genommen. Lokal heisst: beide Nachbarn existieren und sind
  groesser - Punkte am Rand des Scans und Punkte, die an den
  ausgeschlossenen verbotenen Bereich grenzen, fallen damit von selbst raus.
- *Globales Minimum je Spalte*: das alte Verhalten. Beim rohen J laeuft der
  Pfad damit auf der Grenze des verbotenen Bereichs entlang - die Steigung
  ist dann die der Grenze, nicht die der Physik.

**Leitgroesse** ist die Groesse, aus der die Leitgerade kommt (Voreinstellung
Uniformity atom-gewichtet - ihr Talpfad ist der stabilste im Projekt). Der
**Korridor** begrenzt, wie weit die Auswahl von der Leitgeraden abweichen
darf. Am 41x41-Datensatz aendert sich zwischen 0.010 und 0.100 MHz gar
nichts; am 21x21 lohnt ein Vergleich zweier Werte.

Zu wissen: die Leitgerade waehlt nur aus. Die Punkte sind echte lokale
Minima, die Steigung ist deren eigene - aber welcher Zweig verfolgt wird,
haengt an der Leitgroesse. Der Bericht sagt das ausdruecklich dazu.

**Suchbereich einschraenken** - der Haken darunter. Ohne ihn wird ueber das
ganze Scan-Fenster gesucht. Mit ihm nur innerhalb der vier Grenzen (waist
von/bis in µm, width von/bis in MHz), die beim Laden mit dem vollen Bereich
des Datensatzes vorbelegt werden; die graue Zeile darunter sagt jederzeit,
was ueberhaupt gescannt wurde.

Wozu: Datensaetze koennen MEHRERE Talzweige haben. Dann mischt der Fit sie
und die Steigung ist wertlos. Mit dem Fenster sagt man ihm, welchen Zweig
man meint.

Zwei Dinge zum Verhalten:

- Ein Talpunkt auf der Grenze zaehlt, SOLANGE es ausserhalb nicht weiter
  bergab geht - dann ist er ein echtes lokales Minimum und die Grenze hat ihn
  nur gestreift. Geht es draussen tiefer, laeuft der Zweig aus dem Fenster
  heraus; der Punkt faellt als Randminimum heraus, sonst fittete man die
  eigene Grenze statt eines Tals.
- Der Bereich gilt auch fuer die Leitgerade des gefuehrten Modus.

Kommt keine Gerade zustande, sagt der Dialog warum: wie viele Talpunkte im
Bereich liegen, wie viele Randminima es sind, wo sie liegen und an welcher
Grenze sie kleben. Meist lautet die Antwort: den Bereich weiter fassen, eine
andere Fuehrungsgroesse nehmen - oder feiner rechnen, weil der Zweig im
Scan-Fenster zu kurz ist.

Beispiel `11x11` (Airy-Faktor 1.483): ohne Fenster laeuft der gefundene Zweig
komplett auf der `r_bounds`-Schranke (r_x = r_y = 3.000 ueberall - ein
Klemm-Artefakt). Mit **Waist 0.787 bis 1.05 µm und width 0.24 bis 0.40 MHz**
und Leitgroesse *Uniformity, atom-gewichtet* kommt der freie Zweig heraus:
a = 0.37723 MHz/µm bei R² = 0.995 aus 5 von 5 Punkten (Handfit: 0.37735).
Nur eine der beiden Grenzen reicht nicht.

**Strahlprofil (nur run_penalty_scan.py / run_penalty_only.py).** Das
Dropdown *Parametrisierung* legt fest, was die Zahl „waist" physikalisch
bedeutet:

- *wie bisher (1.1900)* — der bisherige Default. Der tatsaechliche
  1/e²-Radius des Airy-Profils liegt dann bei 0.8025 x waist.
- *1/e² der Airy-Hauptkeule = waist (1.4830)* — der Waist bedeutet bei Airy
  dasselbe wie bei einem Gauss-Strahl.
- *frei eingeben* — eigener Faktor.

Der Wert wandert in die .pkl und steht im Bericht; `run_hard_check.py`
uebernimmt ihn von dort. Datensaetze mit verschiedenen Faktoren sind NICHT
vergleichbar - der Faktor setzt die physikalische Spotgroesse und damit jede
Metrik.

Die Anzeige unter dem k-Feld nennt zusaetzlich den `airy_scale_factor` des
geladenen Datensatzes und rechnet aus, bei welchem k sich die
Airy-Hauptkeulen bzw. die 1/e²-Radien gerade beruehren - man muss sich also
keine Umrechnung merken.

### Verbotener Bereich (Ueberlappung der Eck-Spots)

Die beiden diagonal gegenueberliegenden Eck-Spots duerfen sich nicht
ueberlappen. `width` ist die Gesamtspannweite des Tonarrays, raeumlich also
eine Kantenlaenge S; der Eckabstand ist sqrt(2)*S. Aus `sqrt(2)*S > k*waist`
wird in der (waist, width)-Ebene eine Ursprungsgerade - alles darunter ist
verboten (dicke Spots, eng beieinander).

- **Faktor k**: k = 2 heisst "die Spot-Radien beruehren sich gerade" und
  meint den gaussaequivalenten Radius (1/e^2). Wer beim Airy-Profil die
  Hauptkeule bis zur ersten Nullstelle als den Spot ansieht, nimmt
  k = 2*1.19 = 2.38. Unter dem Feld steht sofort, wie die Grenze dann lautet
  und wie viele Gitterpunkte betroffen waeren - man muss den Lauf nicht
  starten, um das zu sehen.
- **Einzeichnen**: rote Grenzlinie und Schraffur in allen Karten. Aendert
  keine einzige Zahl. Auf der mm-Achse ist die Grenze gekruemmt, das ist
  richtig so (win_input und Waist haengen reziprok zusammen).
- **Ausschliessen**: die verbotenen Punkte werden wie ungueltig behandelt.
  Bester Punkt, Region, Talpfad und Geradenfit sehen sie dann nicht mehr.
  Der normierte Score aendert sich dabei ueberall, weil die Normierung ueber
  das ganze Gitter laeuft - die rohen Metriken nicht. Die .pkl auf der
  Platte wird nicht angefasst.

Beide Haken sind unabhaengig. Nur einzeichnen ist der ehrlichere erste
Blick; ausschliessen lohnt, wenn der Talpfad in den verbotenen Bereich
laeuft.

### Querschnitt entlang des Minimums ("Talschnitt")

**Zuerst das Wichtigste zum Dropdown "Groesse fuer Talpfad/Gerade":** dort
stehen zwei Penalty-Eintraege, und sie sind nicht dasselbe.

- *Kombiniert mit Penalty, normiert (combined_score)* - die Groesse, die auch
  die Score-Karte und die Region benutzen. Dafuer werden die vier Gitter
  vorher einzeln auf 0..1 gezogen. Der Optimierer hat diese Groesse nie
  gesehen; sie entsteht erst bei der Auswertung.
- *Kombiniert mit Penalty, ROH (J der Optimierung)* - genau das, was der Scan
  an jedem Gitterpunkt minimiert hat.

Die Gerade kann je nach Wahl deutlich anders ausfallen - beim
41x41-Datensatz 0.283 MHz/µm gegen 0.196, und die beiden Talpfade liegen
sogar in verschiedenen Teilen der Karte. Welche Groesse gemeint war, steht im
Dateinamen und im Bericht. Wer die Steigung weiterverwendet, sollte wissen,
welche der beiden es war.

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

## 5b. `run_penalty_only.py` - Parametersatz suchen statt scannen

Dieses Skript beantwortet eine andere Frage als die drei anderen. Es
erzeugt keinen Datensatz und wertet keinen aus, sondern sucht **einen
einzigen Parametersatz**:

> "Waist und Brennweiten habe ich - wie muessen width, r_x und r_y sein?"

Es braucht keine Eingangsdatei. Ergebnis ist ein Markdown-Bericht in
`Fit_Results/`, sonst nichts (keine .pkl, keine Plots - es gibt hier
nichts zu plotten, das Ergebnis ist ein Punkt).

### Groessen - vorgeben oder optimieren lassen

Die Tabelle oben im Dialog hat eine Zeile je Groesse. In der zweiten
Spalte steht, was mit ihr passieren soll:

- **vorgeben** - die Groesse wird auf den Wert in der Spalte *Wert*
  festgehalten. Die Felder *von*/*bis* sind dann grau.
- **optimieren** - die Groesse wird mitoptimiert und darf zwischen *von*
  und *bis* liegen. Das Feld *Wert* ist dann grau.

Waehlbar sind: **Waist** (µm), **Width** (MHz), **r_x**, **r_y** und die
drei Brennweiten **f1**, **f2**, **fLO** (mm). Voreingestellt ist genau
der Fall aus der Frage oben: Waist und Brennweiten vorgegeben, width und
die beiden Amplituden-Verhaeltnisse frei.

Zwei Dinge, die von aussen ueberraschen:

- Der Waist ist der **effektive Waist nach der Linse in µm**, nicht der
  Eingangswaist `win_input` in mm, den der 2D-Scan abfaehrt. Das ist die
  Groesse, die man vorgibt, wenn man sie kennt. Welchen `win_input` man
  dafuer einstellen muss, rechnet der Bericht am Ende aus den Brennweiten
  zurueck und nennt ihn in mm.
- Die Brennweiten sind **keine Anzeigegroesse**. Sie legen fest, wo die
  Spots in der Fallenebene liegen - also auch, welche Laenge ein
  gegebenes `width` ueberhaupt bedeutet. Sie zu aendern aendert das
  Ergebnis wirklich.

Ist *bis* nicht groesser als *von*, sagt der Dialog das beim Start und
laeuft nicht los.

### Penalty-Zielfunktion

`alpha` und `combo_lambda` wie im Scan - dieselbe Formel, dieselben rohen
Metriken. Ein Ergebnis hier und ein Gitterpunkt des Scans sind deshalb
direkt vergleichbar.

### Aufbau

`N_x`, `N_y`, Profil, `offset` und `n_grid`. Zu `n_grid`: groesser ist
quadratisch langsamer und macht das Ergebnis nachweislich **nicht**
genauer (siehe "zackige harte Kurven" weiter unten). 400 bis 1000 sind
sinnvoll.

### Suche

- **Startpunkte**: wie viele Optimierungen von verschiedenen Startwerten
  aus laufen. Der erste Start ist immer die Mitte aller Bereiche, die
  uebrigen decken sie gleichmaessig ab. Mehr Startpunkte kosten linear
  mehr Zeit, sind aber der einzige Schutz gegen die feinen lokalen Minima,
  die das Rauschen der harten Metriken erzeugt. `1` = nur der Mittelpunkt.
- **Parallele Prozesse**: verteilt die Startpunkte auf mehrere Kerne.
- **max. Iterationen je Start**: Obergrenze fuer einen einzelnen Lauf.

**Laufzeit.** Eine einzelne Auswertung dauert grob 1-2 s bei
`n_grid = 400` und mehrere Sekunden bei `n_grid = 1000`; ein
Nelder-Mead-Lauf mit drei freien Groessen braucht ungefaehr 130 davon.
Acht Startpunkte auf vier Kernen sind also eher eine halbe Stunde als
eine Minute. Fuer einen ersten Blick: `n_grid = 400`, 2-3 Startpunkte.

### Was im Bericht steht

Die Vorgaben (was war fest, was frei, mit welchem Bereich), das gefundene
Optimum mit allen sieben Groessen, der zugehoerige `win_input` in mm, die
volle Metrik-Aufschluesselung (hart / gewichtet / kombiniert, je fuer
Uniformity und Crosstalk) und `J` - und die Tabelle **aller** Startpunkte.
Die ist der eigentliche Wert: liegen die besten Laeufe dicht beieinander,
ist das Optimum belastbar; streuen sie, hat man eine von mehreren
gleichwertigen Loesungen gefunden und sollte die letzten Nachkommastellen
nicht ernst nehmen.

---

## 6. Was am Ende herauskommt

Nach "Auswertung erzeugen" meldet ein Fenster, wohin geschrieben wurde. In
`Fit_Plots/` liegen dann, je nach Datensatz-Art und Einstellungen:

| Datei | Inhalt |
|---|---|
| `..._metric_comparison.pdf` | vier Karten: Uniformity und Crosstalk, je hart und gewichtet |
| `..._metric_comparison_amp.pdf` | dieselben vier Karten plus r_x und r_y (nur mit dem entsprechenden Haken) |
| `..._region.pdf` | die Score-Karte |
| `..._agreement.pdf` | nur Hard-Check: wo hart und gewichtet einig sind |
| `..._score_scatter.pdf` | nur Hard-Check: gewichtet gegen hart, je Gitterpunkt |
| `..._valley_...pdf` | Querschnitt entlang des Minimums |
| `..._line_...pdf` | Querschnitt entlang der Geraden |

Der Dateiname beginnt mit `PenaltyRegion_` oder `HardCheck_`, dann Tonanzahl,
Gitterpunkte, Strahlprofil und Datum. Der Bericht mit allen Zahlen -
Steigung und R² der Geraden, Region-Grenzen, bester Punkt, Scan-Parameter -
liegt als `..._Report.md` in `Fit_Results/`. Der Bericht von
`run_penalty_only.py` heisst `PenaltyOpt_...md` und liegt im selben
Ordner - dazu gibt es keine Plots.

Die PDFs sind fuer einen LaTeX-Satz gemacht: englische Beschriftung,
Serifenschrift, eingebettete TrueType-Schriften. Sie lassen sich unveraendert
in ein Dokument einbinden.

Die Metrik-Karten sind dabei gleich auf A4 gebaut: die 6-Karten-Fassung fuellt
eine Seite (Textbreite bei 2.5-cm-Raendern, Hoehe abzueglich Platz fuer die
`\caption`), die 4-Karten-Fassung etwa eine halbe. Einbinden mit

```latex
\includegraphics[width=\textwidth]{..._metric_comparison_amp.pdf}
```

verkleinert die Datei praktisch nicht mehr - deshalb kommen die Schriften in
der eingestellten Groesse im Dokument an. Eine Ueberschrift ueber der Figur
gibt es nicht; dort gehoert die `\caption` hin. Der rote Stern steht als
*Working point* in der Legende unter der Abbildung.

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
