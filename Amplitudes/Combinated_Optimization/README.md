# Combinated_Optimization

**Wer die Skripte nur benutzen will, findet die Bedienung Schritt fuer
Schritt in `ANLEITUNG.md`.** Dieses Dokument beschreibt, was die Verfahren
rechnen und wie der Code aufgebaut ist.

Vier Skripte zum Ausfuehren, ein Ordner `lib/` mit dem, was sie benutzen.

## Was will ich gerade?

| Ich moechte ... | ... dann dieses Skript ausfuehren |
|---|---|
| **neue Daten erzeugen**, bei denen die Amplituden gemeinsam auf hart + gewichtet optimiert werden (Penalty-Methode) | `run_penalty_scan.py` |
| pruefen, **ob mein vorhandener Weighted-Datensatz auch im Hard Case gut ist** | `run_hard_check.py` |
| **vorhandene Datensaetze plotten** und den Bericht (neu) erzeugen | `run_plots.py` |
| **einen einzelnen Parametersatz suchen**: einen Teil der Groessen vorgeben, die uebrigen gegen die Penalty optimieren lassen (kein Gitter) | `run_penalty_only.py` |
| fuer **einen festen Parametersatz** (waist, width, r_x, r_y) sehen, wie Uniformity und Crosstalk reagieren, wenn **das Atom nicht auf der Site-Mitte sitzt** - samt Abschaetzung der zu erwartenden Positionsschwankung | `run_atom_offset.py` |

Alle vier oeffnen einen Dialog, in dem die Parameter stehen. Nichts muss
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

(`run_penalty_only.py` braucht keinen Datensatz - es rechnet von Grund auf.)

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
| `..._metric_comparison_amp.pdf` (dieselben vier plus r_x/r_y, 3x2, optional) | x | x |
| `..._region.pdf` (Score-Karte, auf Wunsch mit bestem Punkt) | x | x |
| `..._agreement.pdf` (Uebereinstimmungs-Karte, 4 Kategorien) | - | x |
| `..._score_scatter.pdf` (gewichtet vs. hart je Gitterpunkt) | - | x |
| `..._valley_{X}_over_{Y}.pdf` (Querschnitt entlang des Minimums) | x | x |
| `..._line_{X}_over_{Y}.pdf` (Querschnitt entlang der Geraden) | x | x |
| `..._line_panels_over_{Y}.pdf` (die sechs Groessen einzeln, entlang der Geraden) | x | x |
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

**Beschriftung und Farben der Kurven.** Im Querschnitt heissen die Kurven mit
Symbolen statt ausgeschrieben: $U_w$/$U_h$ fuer die Uniformity, $\eta_w$/$\eta_h$
fuer den Crosstalk (Index w = atom-gewichtet, h = harte Pitch-Box-Maske), $S$
fuer den combined score, dazu $r_x$ und $r_y$. Bei bis zu sieben
Legendeneintraegen unter dem Panel ist "Crosstalk (hard)" schlicht zu lang;
ausgeschrieben steht es weiterhin an der Colorbar der Karte links, die das
Symbol jetzt mit auffuehrt. Die sieben Farben sind als Satz mit moeglichst
grossem kleinstem Farbabstand gewaehlt (CIE-Lab, zusaetzlich unter simulierter
Rot-Gruen-Schwaeche geprueft, Helligkeit begrenzt) und danach so zugeordnet,
dass gerade die Kurven, die im Bild uebereinanderliegen, am weitesten
auseinanderliegen - $U_h$ und $r_y$ im oberen Drittel, $U_w$ und $r_x$ unten,
$r_x$ und $r_y$ auf ihrer gemeinsamen Achse. Ist eine Kurvenfarbe fuer eine
Achsenbeschriftung zu hell, wird nur die Achse abgedunkelt, die Linie behaelt
ihre Farbe.

**Welcher Groesse der Talpfad folgt - normiert oder roh.** Das Dropdown
"Groesse fuer Talpfad/Gerade" hat zwei Penalty-Eintraege, und der Unterschied
ist erheblich:

- **`combined_score` (normiert)** - jedes der vier Gitter wird vorher einzeln
  min-max ueber das Scan-Fenster auf 0..1 gezogen, dann kombiniert. Das macht
  die vier vergleichbar, hebt aber die atom-gewichteten Groessen gegenueber
  der harten Uniformity an, deren rohe Spanne ein Vielfaches groesser ist.
  Diese Groesse hat der Optimierer **nie gesehen** - sie entsteht erst bei der
  Auswertung.
- **`J` roh** - genau die Zielfunktion, die der Scan an jedem Gitterpunkt ueber
  (r_x, r_y) minimiert hat. Wird aus den gespeicherten rohen Gittern
  nachgerechnet (`report._grid_for(..., "penalty_raw")`), kostet also keinen
  neuen Scan, und haengt nicht am gescannten Fenster.

Am 41x41-Datensatz laufen die beiden Talpfaeder in **verschiedenen Teilen der
Karte**: der normierte Fit sitzt bei 0.79-1.03 µm mit Steigung 0.283, der rohe
bei 1.03-1.67 µm mit Steigung 0.196. Der Dateiname und der Bericht nennen
jeweils, welche Groesse gemeint war. Vorsicht bei der Interpretation: auf dem
rohen Zweig klebten in jenem Datensatz 19 der 23 gefitteten Talpunkte an der
`r_bounds`-Schranke - das sind keine freien Optima.

**Sechs Karten statt vier.** Der Haken "Metrik-Vergleich zusaetzlich mit
Amplituden" schreibt eine zweite Datei `..._metric_comparison_amp.pdf`: die
vier Metrik-Karten unveraendert, darunter `r_x` und `r_y` - also die
Amplituden, bei denen die Metriken darueber ausgewertet wurden. Die
2x2-Datei bleibt daneben bestehen; beide bekommen Stern und Gerade nach
denselben Regeln.

Drei Dinge sind an den beiden Amplituden-Karten bewusst anders als an den
Metrik-Karten:

- **Gemeinsame Farbskala** fuer `r_x` und `r_y`, damit ablesbar bleibt, dass
  `r_y` systematisch hoeher liegt. Innerhalb einer einzelnen Karte sieht die
  Struktur dadurch flacher aus.
- **Logarithmisch.** `r_x`/`r_y` sind Verhaeltnisse, die sinnvolle Einheit
  ist der Faktor. Ausserdem hat die Verteilung einen langen Schwanz - im
  21x21-Datensatz liegt der Median bei 1.11, aber 5 % der Punkte ueber 3;
  linear waere die Karte fast einfarbig. Bewusst immer logarithmisch, auch
  bei kleiner Spanne: eine Skala, die je nach Datensatz umschaltet, macht
  den Vergleich zweier Aufloesungen unmoeglich.
- **Grau = auf der `r_bounds`-Schranke.** Diese Punkte sind aus der Skala
  herausgerechnet und grau ueberzeichnet, denn dort steht kein freies
  Optimum, sondern die Schranke - im 41x41-Datensatz betrifft das 18.6 % der
  `r_y`-Werte. Ihr Anteil erscheint zusaetzlich auf der Konsole. Ungueltige
  (NaN-)Punkte bleiben davon unberuehrt und weiss.

**Wie der Talpunkt je Spalte gewaehlt wird.** Der Schalter mit dem groessten
Einfluss auf die Steigung - groesser als Achse, Perzentil oder alpha.

- `global` - der kleinste Wert der Spalte. Beim rohen `J` unbrauchbar: dessen
  globales Minimum liegt am Rand des gescannten Fensters bzw. direkt an der
  Grenze des verbotenen Bereichs. Gemessen am 41x41-Datensatz mit
  ausgeschlossenem verbotenen Bereich: die Gerade bekommt die Steigung
  0.22413, die Grenze selbst hat 0.22390 - der Pfad klebt an der Schranke.
- `guided` (Voreinstellung) - pro Spalte das LOKALE Minimum, das einer
  LEITGERADEN am naechsten liegt. Lokal heisst: beide Nachbarn vorhanden und
  groesser. Damit fallen die Raender des Scan-Fensters UND alle Punkte, die
  an den ausgeschlossenen verbotenen Bereich grenzen, von selbst heraus.
  Die Leitgerade ist der gewoehnliche (globale) Fit einer anderen Groesse,
  voreingestellt `uniformity_weighted`; der Korridor um sie ist einstellbar.

Ergebnis am 41x41-Datensatz fuer `penalty_raw`:

| Auswahl | a (MHz/µm) | R² | Punkte |
|---|---|---|---|
| global | 0.19571 | 0.993 | 23 von 41 |
| guided (Leitgroesse U_w) | **0.28316** | **0.993** | 33 von 41 |
| Leitgerade U_w selbst | 0.29480 | 0.995 | 22 von 41 |

Die 0.28316 bleiben **unveraendert** fuer Korridore von +-0.010 bis
+-0.100 MHz - diese Unempfindlichkeit gegen den einzigen freien Parameter
ist das eigentliche Argument fuer das Verfahren. Beim groberen
21x21-Gitter reagiert es dagegen merklich (0.290 bis 0.297 je nach
Korridor); dort lohnt es, zwei Werte zu vergleichen.

**Suchbereich einschraenken (Haken im Dialog).** Beide Verfahren oben
suchen zunaechst im ganzen Scan-Fenster. Hat ein Datensatz MEHRERE Talzweige,
mischt der Fit sie und die Steigung wird sinnlos. Mit dem Haken
"Suchbereich einschraenken" gibt man vier Grenzen vor (waist von/bis in µm,
width von/bis in MHz); ausserhalb wird gar nicht erst nach einem Minimum
gesucht. Die Felder sind mit dem vollen Bereich des geladenen Datensatzes
vorbelegt, ein Bereich, der den ganzen Scan umfasst, gilt als "keine
Einschraenkung".

**Was mit Punkten auf der Grenze passiert.** Legt man das Fenster eng um
einen Zweig, liegen dessen Punkte zwangslaeufig teils auf der Grenze. Solche
Punkte pauschal zu verwerfen waere zu streng, sie alle zu nehmen zu lasch.
Entschieden wird deshalb am UNGEFENSTERTEN Gitter:

- Geht es ausserhalb der Grenze nicht weiter bergab, ist der Punkt ein
  echtes lokales Minimum, das die Grenze nur streift - er zaehlt.
- Ist der Nachbar draussen kleiner, laeuft der Zweig aus dem Fenster heraus;
  das "Minimum" darin ist nur die Grenze. Der Punkt faellt als Randminimum
  heraus, sonst haette man die eigene Grenze gefittet.

Ohne Suchbereich aendert diese Regel nichts: ein globales Spaltenminimum kann
per Definition keinen kleineren Nachbarn haben. Der Bereich gilt AUCH fuer die
Leitgerade des gefuehrten Modus, sonst waehlte sie aus dem falschen Zweig aus.

**Wenn keine Gerade herauskommt**, sagt der Dialog jetzt warum: wie viele
Talpunkte im Bereich liegen, wie viele davon Randminima sind, wo diese liegen
und an welcher Grenze sie kleben. Im Talpfad-Modus ist das eine Rueckfrage
(der Schnitt selbst geht ja auch ohne Gerade), im Geradenmodus ein Abbruch.
Geprueft wird mit genau den Einstellungen, mit denen dann auch gerechnet wird -
frueher pruefte der Dialog nur Groesse und Achse und ignorierte Auswahl,
Korridor und Suchbereich.

Beispiel `11x11`-Datensatz (Airy-Faktor 1.483): ohne Einschraenkung hat er
zwei Zweige, und der gefundene laeuft komplett an der `r_bounds`-Schranke
entlang (`r_x = r_y = 3.000` an allen sechs Punkten - ein Klemm-Artefakt,
keine Physik). Mit **waist 0.787 .. 1.05 µm UND width 0.24 .. 0.40 MHz**
trifft der Fit den freien Zweig:

| Einstellung | a (MHz/µm) | R² | Punkte |
|---|---|---|---|
| ohne Einschraenkung (U_w) | 0.03078 | 0.06 | 9 von 11 |
| Fenster, U_w, global oder guided | **0.37723** | **0.995** | 5 von 5 |
| Handfit des freien Zweigs | 0.37735 | 0.995 | 6 |

Nur eine der beiden Grenzen reicht nicht: `waist <= 1.05` allein liefert gar
keinen Fit, `width >= 0.24` allein bleibt beim Artefakt. Beim rohen `J`
liegen in diesem Fenster alle Spaltenminima auf der unteren width-Grenze und
werden folgerichtig als Randminima verworfen - hier muss man `uniformity_weighted`
als Fuehrungsgroesse fitten oder den gefuehrten Modus nehmen (dort a = 0.2419
bei R² = 0.90, das grobe 11x11-Gitter gibt nicht mehr her).

Ein engeres Fenster **waist 0.8 .. 1.2 µm, width 0.30 .. 0.40 MHz** ergibt mit
`uniformity_weighted` a = 0.35747 bei R² = 0.9970 aus 4 von 6 Punkten. Die
beiden fehlenden liegen bei waist 1.07 und 1.15 auf der oberen width-Grenze:
dort will der Zweig nach 0.42 bzw. 0.44 MHz, was der Scan (max. 0.40 MHz) nicht
mehr hergibt. Ueber waist ~1.10 hinaus ist dieser Zweig in diesem Datensatz
also gar nicht vermessen - eine groessere waist-Obergrenze bringt nichts,
sondern nur ein Scan bis rund 0.62 MHz width.

**Was dabei ehrlich zu sagen ist:** die Leitgerade WAEHLT nur AUS, sie
verschiebt nichts - die Punkte sind echte lokale Minima der
Fuehrungsgroesse und die Steigung ist deren eigene (0.283 gegen 0.295 der
Leitgeraden, sie ist also keine Kopie). Welcher der mehreren
Minima-Zweige verfolgt wird, entscheidet aber die Leitgroesse. Diese Zahl
ist damit an sie gebunden und kein unabhaengiger Befund. Genau dieser Satz
steht auch im erzeugten Bericht.

**Der Score ist das rohe J - der normierte ist entfallen (2026-09-01).**
`combine_grids()` normierte frueher jede der vier Rohgroessen gitterweit
min-max und bildete daraus einen `combined_score`. Das ist ersatzlos
gestrichen. Region, Bestpunkt und Score-Karte benutzen jetzt genau die
Groesse, die der Scan an jedem Gitterpunkt minimiert hat:

```
X_kombi = 0.5*(X_hart + X_weighted) + combo_lambda * |X_hart - X_weighted|
J       = alpha*Uniformity_kombi + (1-alpha)*Crosstalk_kombi
```

Gruende: der Optimierer hat den normierten Score nie gesehen; er haengt am
gescannten Fenster (dieselbe Physik, anderer Scan-Bereich, andere Zahlen);
und er hebt die atom-gewichteten Groessen um das Fuenf- bis Zehnfache an,
weil deren rohe Spannen 4-7 pp betragen gegen 41 pp bei U_hart.

Der Schluessel im gespeicherten dict heisst weiterhin `combined_score`
(Dateiformat), traegt aber jetzt J; neue Datensaetze bekommen zusaetzlich
`score_is_raw=True`. **Aeltere Dateien tragen dort noch den normierten
Wert - `run_plots.py` rechnet Score, Region und Bestpunkt beim Laden
deshalb grundsaetzlich aus den rohen Gittern neu** und sagt es auf der
Konsole. Die Datei selbst bleibt unangetastet.

Nebeneffekt, der die Sache einfacher macht: J ist punktweise definiert. Der
Ausschluss des verbotenen Bereichs aendert den Score deshalb nur DORT und
nicht mehr im ganzen Gitter.

**Bester Punkt nach frei waehlbarer Groesse.** Der Stern folgt einem
Dropdown in der Gruppe *Darstellung*: voreingestellt der im Datensatz
gespeicherte Punkt (dann zeigen Stern und Bericht ohne Zutun dasselbe),
sonst das Minimum einer beliebigen Fuehrungsgroesse; der Bericht bekommt
dann einen eigenen Abschnitt dazu.

Die letzten beiden Eintraege des Dropdowns sind etwas anderes: **ein selbst
vorgegebener Punkt.** Man gibt EINE Koordinate vor - Waist in µm oder width
in MHz -, die zweite kommt aus der Talpfad-Geraden
(`width = a*waist + b` bzw. deren Umkehrung). Der Punkt liegt damit exakt auf
der Geraden und in aller Regel ZWISCHEN den Gitterpunkten; er wird auch dort
gezeichnet und nicht auf ein Gitter gerundet, Legende "selected point". Der
Bericht nennt beide Koordinaten, die verwendete Geradengleichung und den
naechstgelegenen tatsaechlich gerechneten Gitterpunkt. Liegt der Punkt
ausserhalb des gescannten Fensters, wird er trotzdem gezeichnet, aber Bericht
und Konsole sagen, dass es dort keine Daten gibt. Ohne brauchbare Gerade gibt
es keinen Punkt und einen Hinweis auf der Konsole.

**Der Bericht nennt zu jedem markierten Punkt die Werte der sechs Groessen**
(Uniformity und Crosstalk je hart und gewichtet, r_x, r_y) plus J. Beim
selbst gewaehlten Punkt zweispaltig: bilinear zwischen den vier umliegenden
Gitterpunkten interpoliert, daneben die wirklich gerechneten Werte am
naechstgelegenen Gitterpunkt als Anker; liegt eine Ecke ausserhalb des Scans
oder ist ungueltig, steht dort n/a. Beim besten Punkt nach einer Groesse ist
es ein Gitterpunkt, dort genuegt eine Spalte.

Dazu ein Hinweis im Bericht, der leicht uebersehen wird: r_x und r_y sind
Optimierungs-ERGEBNISSE des Scans, keine glatten Funktionen - eine
Interpolation zwischen ihnen ist nur eine Schaetzung. Wer die Metriken exakt
an einem freien Punkt braucht, muss die Amplituden dort neu optimieren, und
dafuer gibt es `run_penalty_only.py`.

Der zugehoerige `win_input` wird ueber die reziproke Kaskade
zurueckgerechnet und durch Vorwaertsrechnen mit `win_input_to_win()`
gegengeprueft - passt die Probe nicht, gibt es `None` statt einer falschen
Zahl.

**Liegt der Punkt auf dem Rand des gescannten Fensters, wird der Stern
OFFEN gezeichnet** und Bericht wie Konsole warnen. Das ist kein Randfall:
das Minimum des rohen J liegt in BEIDEN vorhandenen Datensaetzen auf
width = 0.200 MHz, dem unteren Fensterrand (41x41: 1.3400 mm, 21x21:
1.1150 mm) - und damit zugleich im verbotenen Bereich. Wer den echten
Bestpunkt sucht, braucht einen Scan mit groesserem width-Bereich oder
schliesst den verbotenen Bereich aus.

**Kurven im Querschnitt.** `TRACE_ORDER` fuehrt neun Groessen. Gezeigt wird
als Score `penalty_raw` = `J`, die Zielfunktion der Optimierung; der frueher
ebenfalls angebotene NORMIERTE Score ist am 2026-09-01 ersatzlos entfallen.
`J` traegt dabei das Schwarz des Satzes.

**`J` wird in Prozent gezeigt (2026-09-07).** Bis dahin stand es roh im Bild
(0.058 statt 5.8 %), mit der Begruendung, es trage eine eigene Einheit. Das
stimmt nicht: `J = alpha*U_c + (1-alpha)*eta_c` ist eine Linearkombination
zweier Bruchteile und damit selbst einer. Roh gezeigt brauchte es
zwangslaeufig eine eigene y-Achse - in Prozent steht es auf DERSELBEN wie
`U_c` und `eta_c`, und aus drei Kurven auf zwei Achsen wird eine Achse.

Umgestellt ist es ueberall, damit nicht zwei `J`-Skalen nebeneinander stehen:
Schnittkurven, die Colorbar der Score-Karte (`plot_region`), die Colorbar der
Karte im Talschnitt und der Bericht (bester Punkt und Wertetabelle). Auch
`U_c`/`eta_c` stehen im Bericht jetzt in Prozent statt roh.

Die Karte im Talschnitt skaliert seitdem generell wie die Kurve daneben: ist
die Fuehrungsgroesse eine Prozentgroesse, zeigt auch die Colorbar Prozent.
Vorher stand dort 0.01, wo die Kurve daneben 1.0 % zeigte - dieselbe Groesse
in zwei Einheiten im selben Bild.

**Der Airy-Skalenfaktor ist einstellbar (2026-09-01).** `first_zero_radius =
airy_scale_factor * waist` - der Faktor setzt die physikalische Spotgroesse
und damit JEDE Metrik eines Scans, nicht nur den verbotenen Bereich.

Bisher war er fest auf dem Optimierer-Default 1.19 und wurde nicht
mitgespeichert. Jetzt:

- `run_penalty_scan.py` und `run_penalty_only.py` haben in der Gruppe
  *Strahlprofil* ein Dropdown **Parametrisierung** mit zwei benannten
  Konventionen plus Freieingabe (`combine.AIRY_SCALE_CHOICES`):

  | Eintrag | Faktor | Bedeutung |
  |---|---|---|
  | wie bisher | 1.19 | der historische Default; 1/e²-Radius liegt bei 0.8025 · waist |
  | 1/e² der Airy-Hauptkeule = waist | 1.482951 | der Waist bedeutet bei Airy dasselbe wie bei einem Gauss-Strahl |
  | frei eingeben | — | eigener Wert im Feld darunter |

  Bei einer benannten Konvention zeigt das Zahlenfeld nur den Wert; tippen
  laesst es sich nur bei „frei eingeben". Im Scan-Dialog steht darunter live,
  was der Wert konkret bedeutet (erste Nullstelle, tatsaechlicher
  1/e²-Radius, und bei welchem k sich zwei Hauptkeulen beruehren).

  Herleitung des zweiten Wertes: `(2·J₁(u)/u)²` faellt bei
  u = 2.583838989865 auf e⁻², die erste Nullstelle liegt bei
  u = 3.831705970207512, also Faktor = 3.8317…/2.5838… = 1.482951.
  Gegengerechnet: damit ist der 1/e²-Radius 1.000000168 · waist.
- Der Wert wandert in die .pkl (`airy_scale_factor`) und steht im Bericht
  unter *Scan-Parameter*. Aeltere Dateien fuehren ihn nicht - dort sagt der
  Bericht ausdruecklich, dass der Default 1.19 galt.
- `run_hard_check.py` uebernimmt ihn ueber `INHERITED_KEYS` aus dem
  gewichteten Datensatz. Ohne das wuerde die harte Nachrechnung mit einer
  anderen Optik laufen als der Scan, den sie prueft.

**Wieviel haengt daran:** ein einzelner Punkt (win_input 1.45 mm,
width 0.27 MHz, r_x/r_y = 1.0/1.18), einmal mit 1.19 und einmal mit 1.4830
gerechnet:

| | 1.19 | 1.4830 |
|---|---|---|
| R_1 | 1.0982 µm | 1.3686 µm |
| Uniformity hart | 3.7675 % | 6.5344 % |
| Crosstalk hart | 4.9117 % | 6.5168 % |
| Uniformity gewichtet | 0.7684 % | 0.3506 % |
| Crosstalk gewichtet | 0.9547 % | 1.1674 % |

Das ist kein Feinschliff - Datensaetze mit verschiedenen Faktoren sind nicht
vergleichbar.

**Welcher Wert ist richtig?** 1.19 entspricht keiner gaengigen Konvention
(gleicher 1/e²-Radius: 1.4830; bester Gauss-Fit an die Hauptkeule: 1.4499;
gleiche FWHM: 1.3956). Bei 1.19 liegt der tatsaechliche 1/e²-Radius des
Airy-Profils bei 0.8025 * waist, also 20 % unter der Zahl, die "waist"
heisst; bei 1.4830 liegt er exakt auf waist. Das ist eine physikalische
Entscheidung, keine Code-Frage - deshalb ist der Faktor jetzt ein Feld und
keine Konstante.

**Verbotener Bereich: Ueberlappung der Eck-Spots.** Die beiden diagonal
gegenueberliegenden ECK-Spots des Arrays duerfen sich nicht ueberlappen.
`width` ist die Gesamtspannweite des Tonarrays - in x und y derselbe Wert,
nachgesehen in `_compute_centers_for_width()` -, raeumlich also eine
Kantenlaenge `S = u * width` mit `u = 6.3162 µm/MHz` fuer die Optik dieses
Projekts. Der Eckabstand ist damit `d = sqrt(2) * S`, und die Bedingung
`d > k * waist` (Default `k = 2`, die Radien beruehren sich gerade) ist in
der (waist, width)-Ebene exakt eine Ursprungsgerade:

```
width/MHz > k / (sqrt(2) * u) * waist/µm  =  0.22390 * waist/µm   (k = 2)
```

`S` ist linear in `width` - `radius_from_angle` geht zwar ueber `tan`, aber
theta liegt bei 1.2e-3 rad, die Abweichung von der Geraden ist 5e-7 relativ
und damit weit unter jeder Gitterweite (an einem Punkt von Hand
gegengerechnet). Der verbotene Bereich liegt UNTERHALB der Geraden: dicke
Spots, eng beieinander.

Der Faktor `k` ist frei einstellbar. `k = 2` ist der gaussaequivalente
Radius (1/e^2), wie vom Nutzer vorgegeben; wer beim Airy-Profil die
Hauptkeule bis zur ersten Nullstelle (1.19*w_0) als "den Spot" ansieht,
nimmt `k = 2.38`. Beim 41x41-Datensatz sind damit 27.5 % bzw. 48.7 % aller
Gitterpunkte verboten.

Zwei unabhaengige Haken in der Gruppe *Verbotener Bereich*:

- **Einzeichnen** - Grenzgerade plus schraffierte Flaeche in allen Karten
  (Metrik-Vergleich, Region, Talschnitt, Uebereinstimmungs-Karte). Auf der
  mm-Achse ist die Grenze gekruemmt, weil `waist ~ 1/win_input`; sie wird
  dort dicht abgetastet. Aendert keine Zahl.
- **Ausschliessen** - alle Gitter im verbotenen Bereich werden auf NaN
  gesetzt, danach werden Score, Region und Bestpunkt neu gerechnet. Wichtig:
  die Min-Max-Normierung in `combine_grids()` laeuft ueber das ganze Gitter,
  der normierte `combined_score` aendert sich dadurch UEBERALL. Die rohen
  Metriken bleiben unberuehrt, die Datei auf der Platte auch.

Der Bericht bekommt in beiden Faellen einen Abschnitt mit Formel, Steigung
und Anzahl betroffener Punkte.

**Die Gerade in den Metrik-Karten.** Der Haken "Gerade auch in den
Metrik-Vergleich einzeichnen" in der Gruppe *Darstellung* zeichnet dieselbe
Gerade zusaetzlich in alle vier Karten von `..._metric_comparison.pdf` -
durchgezogen im gefitteten Bereich, gepunktet in der Extrapolation, auf den
gescannten width-Bereich beschnitten. Der extrapolierte Teil bekommt dabei
KEINEN eigenen Legendeneintrag: der Unterschied zwischen Fit und
Verlaengerung steckt im Linienformat, nicht in einem zweiten Kasten. Wie weit
gefittet wurde, steht im Bericht. Welche Groesse gefittet wird, bestimmt
"Groesse fuer Talpfad/Gerade" in der Talschnitt-Gruppe. Die Gerade ist immer
die ueber dem effektiven Waist in µm; auf der mm-Achse erscheint sie deshalb
leicht gekruemmt, weil win_input und effektiver Waist nichtlinear
zusammenhaengen (`report.line_points_for_axis` tastet sie dort dicht ab).
Laesst sich keine Gerade legen, bleiben die Karten unveraendert und es kommt
ein Hinweis auf die Konsole. Die Legende der vier Karten steht **einmal unter
der ganzen Figur** statt vier Mal mitten in den Heatmaps - dort verdeckten
die Kaesten sonst genau den Bereich, um den es geht - und hat genau **einen
Eintrag**: "Linear model fit" (auch dann, wenn die Gerade extrapoliert
gezeichnet wird). Der beste Punkt wird weiterhin als roter Stern
eingezeichnet (Haken "Besten Punkt als Stern einzeichnen"), bekommt hier aber
keinen Legendeneintrag; ein roter Stern in einer Metrik-Karte erklaert sich
selbst. Ohne Gerade hat die Figur gar keine Legende.

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

Die in Stufe 2/3 verworfenen Punkte erscheinen im Plot als offene Kreise -
dieselbe Markierung wie die Randminima, denn fuer den Betrachter ist beides
dasselbe ("steckt nicht in der Auswertung") -,
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

**Der Geradenmodus zeigt drei Kurven, nicht sieben (2026-09-07).** Neben der
Karte standen frueher alle angehakten Groessen - bis zu sieben Kurven auf vier
gestaffelten y-Achsen. Das war nicht mehr zu lesen. Dort stehen jetzt fest die
Zielgroesse und ihre beiden Haelften:

```
U_c = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
eta_c analog,   J = alpha*U_c + (1-alpha)*eta_c
```

`U_c` und `eta_c` sind beide in %, teilen sich also eine Achse - macht **zwei**
y-Achsen statt vier. Die beiden werden wie `penalty_raw` aus den gespeicherten
ROHEN Gittern nachgerechnet (`report._grid_for`), kosten also keinen neuen Scan
und gibt es auch fuer aeltere Datensaetze. Als Kurven sind sie im Talpfad-Modus
anhakbar, aber nicht voreingestellt.

**Und daneben: dieselben sechs Felder wie die Metrik-Uebersicht, nur als
Schnitte.** Datei `..._line_panels_over_{Y}.pdf`, seitenfuellend auf A4 wie der
6-Karten-Plot und in genau derselben Anordnung - `U_h` oben links, `U_w` oben
rechts, `eta_h` und `eta_w` in der Mitte, `r_x` unten links, `r_y` unten
rechts. Wo dort die Karte steht, steht hier der Schnitt entlang der Geraden.
Jedes Feld hat seine eigene y-Achse, also gibt es das Staffel-Problem gar
nicht erst. Extrapolierte Stellen sind mit offenen Kreisen markiert, wie in
der Karte.

An den y-Achsen dieser sechs Felder steht nur das Symbol - `$U_h$ (%)`,
`$U_w$ (%)`, `$\eta_h$ (%)`, `$\eta_w$ (%)`, `$r_x$`, `$r_y$`. Was die Groesse
ist, sagt schon der Titel darueber ("Uniformity (hard mask)"); den
ausgeschriebenen Namen zweimal ins selbe Feld zu schreiben kostet nur Platz.

Die Datei entsteht automatisch, sobald "Schnitt entlang: Gerade" gewaehlt ist.

**Zwei Schalter fuer den Schnitt daneben:**

- **"Maximale y-Achsen im Schnitt"** (Vorgabe 2, Bereich 1-5). Normalerweise
  buendelt `group_traces_by_axis` selbst - gleiche Einheit, gleiche
  Groessenordnung, eine Achse. `_limit_axis_count` legt zusaetzlich Gruppen
  zusammen, bis die Grenze eingehalten ist: zuerst zwei Gruppen mit GLEICHER
  Einheit (das ist nur eine Frage der Skalierung), erst danach verschiedene.
  Fuer `J`/`U_c`/`eta_c` greift die Grenze seit der Prozent-Darstellung gar
  nicht mehr - sie ist die Absicherung, nicht der Normalfall.
- **Die Gerade heisst in JEDER 2D-Karte "Linear fit"** (`report.FIT_LINE_LABEL`)
  - Metrik-Karten, Score-Karte und die Karte im Schnittplot. Dort stand vorher
  "cut along the fitted line (33/41 pts)" bzw. "Linear model fit"; die Zahl der
  benutzten Punkte gehoert in den Bericht, nicht in die Legende.

**Was ausser der Geraden in die 2D-Karten kommt - vier Schalter.**
`report.MAP_MARK_KEYS` / `MAP_MARKS_DEFAULT`, im Dialog vier Haken. Es sind
bewusst VIER und nicht einer, weil es vier verschiedene Aussagen sind:

| Schalter | Vorgabe | was er zeichnet |
|---|---|---|
| `extension` | **an** | die Gerade ueber den gefitteten Bereich hinaus |
| `dashed` | aus | diesen verlaengerten Teil gestrichelt statt durchgezogen |
| `path` | aus | den Talpfad als Linie, an den ausgelassenen Stellen abbrechend |
| `used` | aus | die Talpunkte, die IN den Fit eingegangen sind (gefuellt) |
| `unused` | aus | die herausgefallenen Talpunkte (offene Kreise, mit Zahl) |

Die Verlaengerung sieht seit dem 2026-09-07 aus wie der Rest der Geraden - sie
IST dieselbe Gerade. Vorher war sie fest gepunktet; das sagte zwar "hier wurde
nicht gefittet", machte die Gerade aber optisch zweiteilig. Wer die Aussage
braucht, setzt `dashed`.

Sie gelten fuer ALLE 2D-Karten - Metrik-Vergleich, Score-Karte und die Karte
im Schnittplot -, damit ein Dokument nicht zwei Bildsprachen mischt.
Voreingestellt ist nur die Verlaengerung: das ist das Bild fuers Dokument, die
drei anderen sind die Herleitung. `report.map_marks(**kwargs)` baut das dict
und wirft bei unbekannten Namen einen `ValueError` - ein vertippter Schalter
soll nicht stillschweigend nichts tun.

Zwei Sonderfaelle: im Talpfad-Modus ist `path` wirkungslos (dort IST der Pfad
der Schnitt und wird immer gezeichnet), und die Score-Karte bekommt die Gerade
ueberhaupt erst, seit sie `fit_line` durchgereicht bekommt.
Die Kurven-Haken im Dialog sind dann grau - sie wirken nur noch im
Talpfad-Modus, wo die alte Fassung mit freier Kurvenauswahl unveraendert
bleibt.

Im Geradenmodus wird die Gerade immer bestimmt - sie ist ja der Schnitt
selbst; der Haken "Gerade durch den Talpfad legen" ist dort gesetzt und
gesperrt. Laesst sich fuer die gewaehlte Groesse keine Gerade legen, sagt der
Dialog das VOR dem Start, statt mitten in der Auswertung abzubrechen.

**Die Waist-Achse heisst "Waist in the atomic plane" (2026-09-07).** Frueher
stand dort `Waist at focus $\omega'$ ($\mu$m, after lenses)`. Der Strich an
omega sollte "nach der Linse" heissen, sagte das aber nur dem, der die
Konvention schon kennt - im Fliesstext eines Dokuments liest er sich wie eine
Ableitung. Jetzt: `Waist in the atomic plane $\omega$ ($\mu$m)`, ohne Strich,
und die Sache steht im Namen statt in einem Zeichen. Der Text liegt als
`report.WAIST_UM_LABEL` an genau einer Stelle und gilt fuer alle Plots -
Karten, Region, Talschnitt, Panel-Plot. Der Input-Waist vor den Linsen heisst
unveraendert `$\omega_{\mathrm{in}}$` (`report.WIN_INPUT_LABEL`).

**Einheiten stehen in eckigen Klammern (2026-09-07).** `Width [MHz]`,
`Waist in the atomic plane $\omega$ [$\mu$m]`, `$U_h$ [%]` - in allen Plots,
an Achsen wie an Colorbars. Runde Klammern bleiben den ERKLAERUNGEN
vorbehalten, die keine Einheit sind: `(hard mask)`, `(atom-weighted)`,
`(raw, lower is better)`, `(before lenses)`, `(log)`. Dimensionslose Groessen
- `J`, `r_x`, `r_y` - bekommen gar keine Klammer.

**Der Querschnitt hat dieselbe Schrift wie die Karten (2026-09-07).** Er war
11.4 Zoll breit und wurde im Dokument auf 55 % geschrumpft, waehrend die
Kartenplots seit dem A4-Umbau 1:1 ankommen - nebeneinander im selben Dokument
sah das aus wie zwei Schriftgroessen. Jetzt `CUT_FIGSIZE = (6.3, 3.9)` Zoll und
`MAP_STYLE` wie die Karten.

**Die y-Achsen im Querschnitt sind schwarz.** Sie trugen frueher die Farbe
ihrer Kurve (abgedunkelt ueber `_achsenfarbe`), was Achse und Kurve zuordnete -
bei mehreren Achsen aber ebenso viele verschiedenfarbige Beschriftungen ergab.
In einem Schriftsatz ist das unruhig, und die Zuordnung leistet die Legende
ohnehin. `_achsenfarbe` und `MULTI_TRACE_AXIS_COLOR` sind damit entfallen.

**Der markierte Punkt steht in ALLEN Plots (2026-09-07).** Frueher zeigten ihn
nur die Metrik-Karten und die Score-Karte. Jetzt auch:

- die Karte im Schnittplot - als Stern, wie in jeder anderen Karte;
- der Querschnitt daneben - als senkrechte Linie an seiner Stelle auf der
  Schnittachse (in einem Schnitt ist von einem Punkt nur die eine Koordinate
  uebrig, also eine Linie);
- alle sechs Felder des Panel-Plots - dieselbe Linie an derselben Stelle, so
  liest man die sechs Werte am Arbeitspunkt in einem Blick ab.

**Die Legende nennt seine Koordinaten und sonst nichts**: `omega = 1.262 µm,
width = 0.370 MHz`. Beide, auch wenn nur eine davon vorgegeben wurde - die
andere kommt aus der Geraden und ist genauso ein Ergebnis. Das Wort "Working
point" steht bewusst NICHT davor: der rote Stern ist in einer Karte mit genau
einem Stern selbsterklaerend, und die Legende soll die Zahlen tragen, nicht
eine Bezeichnung fuer sie. (`WORKING_POINT_LABEL` bleibt als Rueckfallwert
fuer den Fall, dass der Punkt keine Koordinaten hat.)

Dafuer traegt jeder Punkt jetzt `waist_um` mit (`report._mit_waist_um`);
vorher hatte ihn nur der manuell gesetzte Punkt, und die Legende haette bei
den anderen `nan` angezeigt.

**Die einzelnen extrapolierten Stuetzstellen** - die offenen Kreise auf der
Geraden bzw. auf den Schnittkurven - erscheinen nur noch, wenn auch die
Talpunkte eingeschaltet sind (`used`/`unused`). Sie gehoeren zur Herleitung;
im fertigen Bild unterbrechen sie eine sonst glatte Linie ohne Gewinn. Das
gilt fuer den Schnittplot wie fuer den Panel-Plot.

**Die Colorbar der Karte im Schnittplot traegt ihre Einheit** (`... [%]`),
seit die Karte wie die Kurve daneben in Prozent skaliert - vorher standen dort
Prozentzahlen ohne Prozentzeichen.

Der Schnittplot hat **zwei** Legenden, je eine unter ihrem Panel: Karte und
Querschnitt zeigen verschiedene Dinge, und in einem gemeinsamen Kasten muss der
Leser erst sortieren, was wohin gehoert. Die Karte ist schmaler, ihre Legende
deshalb einspaltig; die des Querschnitts nimmt bis zu drei Spalten.

`CUT_FIGSIZE = (6.3, 3.6)` und `CUT_WIDTH_RATIOS = [1.0, 1.0]` sind nicht
geschaetzt, sondern an den uebrigen Plots GEMESSEN. Eine Achse ist dort:

| Plot | Achse (Zoll) | Breite/Hoehe |
|---|---|---|
| 2x2-Karten | 2.04 x 2.34 | 0.87 |
| 3x2-Karten | 2.00 x 2.37 | 0.84 |
| Panel-Schnitte | 2.35 x 2.50 | 0.94 |
| **Schnittplot** | **2.03 x 2.40** | **0.84** |

Vorher stand hier 4.0 Zoll bei 1 : 1.15, was 1.81 x 2.75 Zoll ergab
(Breite/Hoehe 0.66) - sichtbar hochkant gegenueber allen anderen Plots. Beim
Formatieren solcher Figuren lohnt es, die Achsenmasse mit
`ax.get_position()` gegen die anderen Plots zu HALTEN, statt die Figurhoehe zu
raten: der Aufbau des Schnittplots (Colorbar auf der einen, y-Achse auf der
anderen Seite) frisst rund 2.3 Zoll Breite, die in einer reinen Kartenfigur
nicht anfallen.

**Die x-Achse traegt dort die Kurzform** `Waist $\omega$ [$\mu$m]`
(`report.short_axis_label`). Bei Textbreite bleiben zwei Panels nebeneinander
rund 2.5 Zoll breit - der ausgeschriebene Name lief ueber den Figurrand hinaus,
gemessen fiel das schliessende "]" weg. In den einzeln stehenden Karten steht
weiterhin der volle Name.

**Der Bericht listet alle Werte am markierten Punkt** - `U_h`, `U_w`,
`eta_h`, `eta_w`, `U_c`, `eta_c`, `r_x`, `r_y` und `J`, bilinear interpoliert
UND am naechstgelegenen Gitterpunkt. `U_c`/`eta_c` sind seit dem 2026-09-07
dabei (`POINT_VALUE_KEYS`).

**Der Geradenfit bringt seine Unsicherheit mit (2026-09-07).** `sigma_a` und
`sigma_b` stehen im Fit-dict und im Bericht (`a = 0.2949 +- 0.0023`), berechnet
mit der ueblichen Formel der linearen Regression - gegengeprueft gegen
`np.polyfit(..., cov=True)`, Abweichung in der sechsten Stelle:

```
s²    = SS_res / (n - 2)
sig_a = sqrt(s² / S_tt),   S_tt = sum (t - mean t)²
sig_b = sig_a * sqrt(sum t² / n)
```

Der Bericht sagt dabei ausdruecklich, was die Zahl NICHT ist. Die Talpunkte
sind Ergebnisse einer Optimierung auf einem Gitter, keine unabhaengigen
Messungen - die Streuung durch Gitterschrittweite und Wahl der Fuehrungsgroesse
ist um ein Vielfaches groesser: am 41x41-Datensatz liegen 0.196 (globales
Minimum) und 0.283 (gefuehrt) auseinander, bei sigma_a von rund 0.004. Die
Fehlerangabe ist damit eine untere Schranke, kein Gesamtfehler.

**Die Legende bricht auf zwei Eintraege pro Zeile um**, sobald eine Zeile zu
voll wird (`report.legend_ncol`). Die Regel ist zweistufig: passt alles in eine
Zeile, kommt alles in eine Zeile; sonst zwei, zentriert. Geschaetzt wird ueber
die DARGESTELLTE Zeichenzahl - `_label_laenge` rechnet Mathtext heraus, weil
`$\omega$` neun Zeichen im Code und eines im Bild ist. Ohne das waere jeder
Eintrag mit Formelzeichen viel zu lang geschaetzt und die Legende immer
einspaltig.

**Aussehen der PDFs.** Alle Plots dieses Ordners sind auf einen LaTeX-Satz
ausgelegt: englische Beschriftungen, knappe Titel, Serifenschrift mit
Computer-Modern-Mathtext (`report.LATEX_STYLE`) und `pdf.fonttype=42`, also
eingebettete TrueType- statt Type-3-Schriften - Type 3 wird von vielen
Journals und von `pdffonts`-Pruefungen abgelehnt. Der Stil wird nur ueber
`plt.rc_context` gesetzt und faerbt deshalb nicht auf andere Skripte ab.
Berichte (`.md`) und der Dialog bleiben deutsch.

**Die Metrik-Karten sind auf A4 gebaut (2026-09-01).** Frueher war die Figur
12 Zoll breit. `\includegraphics[width=\textwidth]` schrumpfte sie im Dokument
auf Textbreite - Faktor 0.52 -, und aus 10-pt-Beschriftung wurden 5 pt. Jetzt
wird die Datei gleich in der Groesse erzeugt, in der sie im Dokument landet:

- `PAGE_FIGSIZE = (6.3, 9.0)` Zoll fuer die 3x2-Fassung: Textbreite bei
  2.5-cm-Raendern, Hoehe = Texthoehe abzueglich rund 1.6 cm fuer die
  `\caption`. Die Abbildung fuellt damit eine Seite.
- `HALF_PAGE_FIGSIZE = (6.3, 6.2)` Zoll fuer die 2x2-Fassung - gleiche
  Kartenhoehe, also etwa eine halbe Seite. Sie auf volle Seitenhoehe zu
  ziehen wuerde die vier Karten nur unnatuerlich strecken.

Weil im Dokument nicht mehr skaliert wird, sind die Schriften groesser als im
Rest des Ordners (`report.MAP_STYLE`: Achsen 11 pt, Titel 11.5 pt, Ticks und
Colorbar 9.5 pt). Zwei weitere Aenderungen holen Platz fuer die Karten selbst:

- **Keine Ueberschrift ueber der Figur.** In einem LaTeX-Dokument steht dort
  die `\caption`; zwei Titel uebereinander sind einer zu viel. Was die Karten
  zeigen, steht in ihren eigenen Titeln.
- **Achsenbeschriftung nur aussen herum** (`sharex`/`sharey`): alle Karten
  haben dieselben Achsen, also steht die x-Beschriftung nur unter der
  untersten Reihe und die y-Beschriftung nur links. Das gibt rund einen
  halben Zoll je eingesparter Zeile - der direkt in die Kartenhoehe geht,
  und die Amplitudenkarten unten waren vorher die kleinsten.

**Der Stern heisst "Working point"** und steht in der gemeinsamen Legende
unter der Figur - einmal, obwohl er in jeder Karte gezeichnet wird. Der Name
gilt fuer beide Faelle, den selbst gesetzten Punkt und den gefundenen besten;
liegt er am Rand des gescannten Fensters, wird daraus "Working point (at scan
edge)" und der Stern bleibt offen.

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

## 4. `run_penalty_only.py` - ein einzelner Parametersatz statt eines Gitters

Beantwortet die Frage, die der Scan nicht direkt beantwortet:

> "Waist und Brennweiten habe ich vorgegeben - wie muessen width, r_x und
> r_y sein?"

Im Dialog wird fuer jede der sieben Groessen einzeln eingestellt, ob sie
**vorgegeben** wird (fester Wert) oder **optimiert** (dann mit einem
Bereich, in dem sie liegen darf):

| Groesse | Einheit | |
|---|---|---|
| Waist nach der Linse | µm | der EFFEKTIVE Waist, nicht `win_input` |
| Width | MHz | |
| r_x, r_y | - | Amplituden-Verhaeltnisse |
| f1, f2, fLO | mm | die drei Brennweiten |

Die freien Groessen werden anschliessend **gemeinsam** so gewaehlt, dass
dieselbe Zielfunktion `J` minimal wird, die auch `run_penalty_scan.py` an
jedem Gitterpunkt minimiert - dieselbe Formel, dieselben rohen
(unnormierten) Metriken, dasselbe eine `_evaluate(..., weighted=True)` pro
Auswertung. Ein Ergebnis hier und ein Gitterpunkt dort sind deshalb direkt
vergleichbar.

**Zu den Brennweiten.** f1, f2 und fLO sind keine Anzeigegroesse: sie
bestimmen ueber `radius_from_angle()` die Spot-Positionen in der
Fallenebene und damit, welche Laenge ein gegebenes `width` ueberhaupt
bedeutet. Weil der Waist hier direkt in µm vorgegeben wird, ist das der
wirksame Weg; die zweite Rolle der Brennweiten (Umrechnung
Eingangswaist -> effektiver Waist) benutzt der Bericht, um am Ende zu
sagen, welchen **win_input in mm** man fuer den gefundenen Waist
einstellen muss.

**Warum mehrere Startpunkte.** Die harten Metriken rauschen (siehe
`ANLEITUNG.md`, "zackige harte Kurven"), die Zielfunktion hat dadurch
feine lokale Minima ohne physikalische Bedeutung. Deshalb laufen per
Default acht Nelder-Mead-Optimierungen: die erste aus der Mitte aller
Bereiche, die uebrigen von Latin-Hypercube-Punkten (fester Seed, also
reproduzierbar). Der Bericht listet **alle** Laeufe mit ihrem J - liegen
die besten dicht beieinander, ist das Optimum belastbar; streuen sie, ist
es eine von mehreren gleichwertigen Loesungen. `Startpunkte = 1` ist genau
der eine Lauf aus der Mitte.

Optimiert wird intern in normierten Koordinaten `u` in `[0, 1]` je freier
Groesse. Nelder-Mead baut seinen Startsimplex aus festen relativen
Schritten; mit `waist ~ 1` (µm), `width ~ 0.3` (MHz) und `f2 ~ 750` (mm)
im selben Vektor waere er um drei Groessenordnungen verzerrt und der Lauf
wuerde faktisch nur noch f2 variieren.

- Ergebnis: `Fit_Results/PenaltyOpt_N{Nx}x{Ny}_{Profil}_{Datum}.md`
- **Keine .pkl, keine Plots** - das Ergebnis ist ein Punkt, kein Feld
- Braucht keinen vorhandenen Datensatz

Der Bericht enthaelt: die Vorgaben (was war fest, was frei, mit welchem
Bereich), das Optimum mit allen sieben Groessen, den zugehoerigen
win_input in mm, die volle Metrik-Aufschluesselung (hart / gewichtet /
kombiniert fuer Uniformity und Crosstalk) und J, die Tabelle aller
Startpunkte samt Streuung, sowie einen Abschnitt dazu, wie genau das
Ergebnis ueberhaupt sein kann.

---

## 5. `run_atom_offset.py` - Atom gegen das Profil verschieben

Beantwortet:

> "Mein Parametersatz steht. Was passiert mit Uniformity und Crosstalk,
> wenn das Atom nicht genau auf der Site-Mitte sitzt - und wie weit sitzt
> es im Labor ueberhaupt daneben?"

Eingetragen werden **waist** (µm, Atomebene), **width** (MHz), **r_x** und
**r_y**. Fest wie ueberall hier: 3x4 Toene, f1 = 75 mm, f2 = 750 mm.

**Was sich bewegt.** Das Atom wandert radial von der Site-Mitte nach
aussen. Mit ihm wandern ALLE Regionen: das
Ton-Quadrat der harten Uniformity, das Pitch-Quadrat des harten Crosstalks
und die Gauss-Gewichtung von U_w/eta_w (`atom_offset_x/y` des Optimierers).
Das Lichtfeld (12 Spots, 8 Nachbarkopien) steht. Das ist exakt dasselbe wie
eine gemeinsame Drift des ganzen Raman-Profils um -r.

**Richtung.** Die Metriken haengen von der Richtung des Versatzes ab (das
3x4-Array ist nicht rotationssymmetrisch, die Nachbarn liegen auf einem
Quadratgitter). Vorgabe ist deshalb das **Richtungsmittel**: an jedem r wird
an 6 Winkeln im Halbkreis gerechnet und gemittelt - eine Kurve je Groesse,
der Erwartungswert fuer einen Versatz um r in zufaelliger Richtung. Der
Halbkreis genuegt, weil die Anordnung punktsymmetrisch ist (+r = -r bis auf
3e-6 pp); spiegelsymmetrisch ist sie mit Kohaerenz nicht (0.25 pp). 6 Winkel
weichen von 16 um < 0.002 pp ab. Die kombinierten Groessen werden je Winkel
gebildet und dann gemittelt. Optional: Spannweite ueber die Richtungen als
blasses Band, oder statt des Mittels einzelne Richtungen (horizontal =
3-Ton-Achse, vertikal = 4-Ton-Achse, diagonal, antidiagonal).

**Warum eigene harte Metriken.** Auf dem globalen Gitter des Optimierers
sind die Regionen Masken; verschiebt man sie um Bruchteile eines Pixels,
springt ihr Rand um ganze Pixel. Bei Versaetzen von einigen 10 nm waere
dieses Rauschen groesser als der Effekt. `lib/atom_offset.py` legt deshalb
ein Gitter aus Zellmitten direkt AUF die Region, das mit dem Atom wandert.
Bei r = 0 stimmt es mit dem Optimierer ueberein (U_h +0.002 pp, eta_h
-0.02 pp - der Pixelrand), der Bericht zeigt den Abgleich bei jedem Lauf.

**Was immer kommt und was optional ist.**

- Immer (Basisfall): Plot 1, r = 0 .. 0.125 x width (width ueber die Optik
  in µm umgerechnet, = Spannweite des Tonarrays `d`; bei 0.45 MHz 2.842 µm),
  mit den **atom-gewichteten** Metriken U_w, eta_w.
- Optional, Gruppe **"Harte Metriken"** (Vorgabe aus): eigener Plot fuer
  U_h, eta_h, beide im Bericht, dazu der Abgleich. Ohne harte und ohne
  kombinierte Groessen wird das Harte gar nicht gerechnet (etwa doppelt so
  schnell).
- Optional, Gruppe **"Kombinierte Groessen"** (Vorgabe aus): eigener Plot
  fuer U_c, eta_c, J und die drei Groessen im Bericht. Rechnet die harten
  Metriken automatisch mit (geplottet werden sie nur mit eigenem Haken).
- Optional, Gruppe **"Positionsschwankung des Atoms"** (Vorgabe aus): die
  Abschaetzung unten, darin Plot 2 (r = 0 .. k x sigma_pos), die Statistik
  und die sigma-Linien in Plot 1 (die nur zusammen mit Plot 2).

**Die Abschaetzung** (`lib/position_noise.py`), alles 1 sigma je Achse,
unabhaengige Beitraege quadratisch addiert:

| Beitrag | Umrechnung in die Atomebene |
|---|---|
| thermische Ortsbreite im Fallenpotential | sigma^2 = hbar/(2 m omega) coth(hbar omega/2kT), unguenstiger Rand (T + dT, nu_r - dnu) |
| Raman-Strahllage vor dem AOD | f1 fLO / f2 = 5.29 nm/µrad |
| Raman-Strahllage vor dem Objektiv | fLO = 52.9 nm/µrad |
| AOD-Schallgeschwindigkeit dv/v | f1 fLO/f2 tan(theta_c) = 0.64 nm/ppm |
| RF-Frequenzfehler | 6.3 pm/Hz |
| Fallenposition, Pitch-Unsicherheit x Site-Index | direkt |
| gemessene Relativposition (gesamt) | direkt, Default 47 nm (Groessenordnung aus arXiv:2502.13560) |

Defaults: T = 17(2) µK, nu_r = 60.4(-2.2) kHz -> thermisch 118 nm; technisch
nur die 47 nm; zusammen **127 nm**. Nicht gemessene technische Groessen
stehen auf 0 - der Hebel daneben sagt, was sie ausmachen wuerden.

**Statistik.** Sitzt das Atom mit sigma_pos gaussverteilt, wie gross sind
Mittelwert und Schuss-zu-Schuss-Streuung jeder Groesse? Gauss-Hermite-
Quadratur, 7 x 7 Positionen.

**Symmetrie.** +r und -r sind je Achse gleich. Diagonale und Antidiagonale
sind es mit Kohaerenz NICHT: die frequenzentarteten Eckspots interferieren
statisch (am Standard-Arbeitspunkt bei r = 0.355 µm: U_w 2.41 % diagonal,
2.98 % antidiagonal). Die Tabelle steht in jedem Bericht.

- Plots: `Fit_Plots/<Datum>/AtomOffset_N3x4_Airy_w..um_width..MHz_rx.._ry.._k.._bis0.125width_<Datum>_weighted.pdf`, `_hard.pdf` nur mit harten, `_penalty.pdf` nur mit kombinierten Groessen, dasselbe mit `bis1sigma` nur mit Positionsschwankung
- Bericht: `Fit_Results/AtomOffset_..._<Datum>_Report.md`
- Datensatz: `Results/atom_offset_....pkl` (optional)
- Rechenzeit mit den Defaults (nur gewichtet, Richtungsmittel) etwa eine halbe Minute; mit harten Metriken etwa doppelt so lang, mit Positionsschwankung (Plot 2 + Statistik) noch einmal mehr

---

## Ordner

```
Combinated_Optimization/
    run_penalty_scan.py     <- ausfuehren: neue Daten (Penalty)
    run_penalty_only.py     <- ausfuehren: ein Parametersatz, kein Gitter
    run_hard_check.py       <- ausfuehren: Hard Case zu vorhandenem Weighted
    run_plots.py            <- ausfuehren: plotten/auswerten
    run_atom_offset.py      <- ausfuehren: Atom gegen das Profil verschieben
    lib/                    <- wird von den drei Skripten benutzt,
                               nicht direkt ausfuehren
        paths.py            Ordner-Konstanten, Anbindung an Weighted_Optimization
        combine.py          Penalty-Kombination, Region, Laden/Speichern
        penalty_scan.py     die gemeinsame Amplituden-Optimierung (Gitter)
        penalty_opt.py      dieselbe Zielfunktion ohne Gitter (freie Parameter)
        hard_check.py       harte Nachrechnung + Konsistenz-Analyse
        report.py           Plots und Markdown-Berichte
        atom_offset.py      Atom-Versatz: Metriken mit mitwandernden Regionen
        atom_offset_report.py  Plots und Bericht dazu
        position_noise.py   Abschaetzung der Positionsschwankung
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
