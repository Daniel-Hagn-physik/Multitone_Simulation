# Anleitung – Hard_Optimization

Für jemanden, der diese Skripte nicht geschrieben hat. Technische Details
stehen in [README.md](README.md).

Was dieser Ordner rechnet: die **harten** Metriken. Uniformity und Crosstalk
werden über ein globales Intensitätsgitter mit einer festen Pitch-Box-Maske
um jeden Spot ausgewertet. Der Nachbarordner `Weighted_Optimization` rechnet
dieselben Größen atom-gewichtet, `Combinated_Optimization` beides zugleich.

---

## Voraussetzungen

Python mit `numpy`, `scipy`, `matplotlib` und `PyQt5`. `torch` ist optional –
wird es gefunden und hat der Rechner eine CUDA-GPU, rechnen die Scans dort;
sonst still auf der CPU.

**PyCharm zeigt rote Importe?** Die Module liegen in `lib/` und werden erst
zur Laufzeit über einen `sys.path`-Eintrag gefunden. PyCharms statische
Analyse kennt den nicht. Der Code läuft trotzdem. Dauerhaft weg bekommt man
das Rot mit einem Handgriff:

> Rechtsklick auf den Ordner `lib` → *Mark Directory as* → *Sources Root*

---

## Welches Skript?

| Ich möchte … | ausführen |
|---|---|
| einen **neuen Scan bei festen Amplituden** rechnen | `run_scan.py` |
| einen **neuen Scan mit Amplituden-Optimierung** je Gitterpunkt | `run_amp_scan.py` |
| einen **vorhandenen Datensatz plotten und auswerten** | `run_plots.py` |
| die **Abhängigkeit von r_x/r_y fitten** | `run_amp_fits.py` |

Alles in `lib/` wird nur benutzt und nie direkt gestartet.

---

## `run_scan.py` – Scan bei festen Amplituden

Rechnet Uniformity und Crosstalk über ein Gitter aus Eingangs-Waist und
width, **ohne** an irgendeinem Punkt zu optimieren. Die Amplituden stehen
vorher fest.

Die Dialoggruppen:

* **Fortsetzen (unfertige Datensätze)** – findet Zwischenstände in
  `Results/`, die zu den eingestellten Parametern passen. Wählt man einen
  aus, werden die Eingabefelder gesperrt und der Scan macht dort weiter, wo
  er abgebrochen ist.
* **Tone Count** – Anzahl Töne in x und y (N_x, N_y).
* **Amplitudes per Axis** – die festen Amplituden, ein Feld je Ton.
* **Scan Ranges** – von/bis und Anzahl Punkte für Waist und width. Der Waist
  lässt sich wahlweise als `win_input` in mm vor der Linse oder als
  effektiver Waist in µm an der Fokusebene eingeben; das Skript rechnet um.
* **Strahlprofil (Airy)** – der Skalenfaktor legt fest, was die Zahl „waist"
  physikalisch bedeutet (`first_zero_radius = Faktor · waist`). Er steht in
  jedem erzeugten Datensatz mit drin.
* **Save Location** – Pfad wird **vor** dem Scan festgelegt. Dorthin geht
  stündlich eine Zwischensicherung, und am Ende der saubere Endstand.

Am Ende erscheinen die beiden Heatmaps (Uniformity, Crosstalk) und landen als
PNG in `Bilder/`. Der Datensatz liegt in `Results/`.

---

## `run_amp_scan.py` – Scan mit Amplituden-Optimierung

Wie oben, aber an **jedem** Gitterpunkt wird zusätzlich ein eigenes
(r_x, r_y)-Paar optimiert. Das ist um Größenordnungen teurer – ein feines
Gitter kann Tage laufen, deshalb die Zwischenspeicherung.

Zusätzliche Gruppen:

* **Amplitude-Ratio Optimization** – Startwert und `r_bounds`, die Schranken,
  innerhalb derer r_x/r_y liegen dürfen. Punkte, an denen das Optimum an eine
  Schranke stößt, sind **keine freien Optima**; `run_plots.py` zeichnet sie
  in den Amplituden-Karten grau und nennt ihren Anteil.
* **Intensity Grid** – Auflösung des globalen Intensitätsgitters
  (`n_grid`). Größer heißt genauer, aber quadratisch teurer.
* **Parallelization** – `n_jobs > 1` verteilt die Gitterpunkte auf mehrere
  Prozesse.
* **GPU Acceleration** – wird automatisch versucht; der Haken erzwingt CPU.

---

## `run_plots.py` – auswerten

Braucht keinen neuen Scan. Datensatz auswählen, Haken setzen, fertig.

### Datensatz

Das Dropdown listet alles aus `Results/`, neueste zuerst. Es ist **nichts**
vorausgewählt – so wird nie versehentlich die falsche Datei ausgewertet. Wer
immer denselben Datensatz auswertet, kann ihn oben im Skript bei `PKL_DATEI`
eintragen. „Andere Datei…" holt einen Datensatz von außerhalb.

Sobald etwas geladen ist, steht darunter, was es ist: Amplituden-Scan oder
Fest-Amplituden-Scan, Gittergröße, alpha, bester Punkt.

### Darstellung

* **Waist-Achse** – `win_input` in mm vor der Linse, oder effektiver Waist in
  µm nach der Linse. Auf der µm-Achse läuft die Achse rückwärts (großer
  Eingangs-Waist = kleiner Fokus); die Heatmaps werden entsprechend
  gespiegelt.
* **Punkt als Stern einzeichnen** und das Dropdown darunter – welcher Punkt
  markiert wird. Voreingestellt ist der beste Gitterpunkt nach dem Score;
  dann zeigen Stern und Bericht ohne Zutun dasselbe. Die drei letzten
  Einträge sind etwas anderes: dort gibt **man selbst** den Punkt vor.
  *Nur Waist vorgeben* bzw. *nur Width vorgeben* brauchen nur eine Koordinate
  – die zweite kommt aus der Talpfad-Geraden, der Punkt liegt dann exakt auf
  ihr. Das setzt aber voraus, dass es überhaupt eine Gerade gibt: **lässt sich
  für den Datensatz keine legen, sind diese beiden Einträge ausgegraut und die
  Auswahl springt auf *Waist UND Width vorgeben***, wo nach beiden Werten
  gefragt wird. Ein selbst gesetzter Punkt liegt in aller Regel zwischen den
  Gitterpunkten und wird auch genau dort gezeichnet.
* **Querschnitt durch den markierten Punkt (r_x, r_y) als PDF** – zwei
  Schnitte durch den Stern in einer Datei (`…_point_cuts.pdf`): links r_x und
  r_y bei fester width entlang des Waists, rechts bei festem Waist entlang der
  width, der Punkt jeweils als senkrechte rote Linie. Gezeigt werden nur r_x
  und r_y, im gewohnten Aussehen der alten Amplituden-Schnitte: r_x blau mit
  Kreisen, r_y orange mit Quadraten, beide auf einer Achse. Beantwortet eine
  andere Frage als der Talschnitt weiter unten: nicht „wie läuft das
  Minimum?", sondern „wie empfindlich sind die Amplituden an meinem
  Arbeitspunkt?" Gelesen wird auf dem Gitter – liegt der Stern dazwischen,
  laufen die Schnitte durch die nächste Zeile bzw. Spalte, und der Titel sagt
  das. Nur beim Amplituden-Scan.
* **Gerade auch in den Metrik-Vergleich einzeichnen** – die Gerade aus der
  Talschnitt-Gruppe erscheint zusätzlich in den Karten, als **eine
  durchgezogene Linie** über den ganzen gescannten Bereich. Aus welchem
  Bereich sie bestimmt wurde, steht im Bericht.
* **… außerhalb des Fit-Bereichs gepunktet statt durchgezogen** – nur wer den
  Unterschied im Bild sehen will, setzt diesen Haken: dann ist die Gerade
  zweiteilig, durchgezogen im gefitteten Bereich und gepunktet in der
  Verlängerung. Der gepunktete Teil bekommt keinen eigenen Legendeneintrag.
  Der Haken ist nur bedienbar, solange die Gerade überhaupt gezeichnet wird.
* **Metrik-Vergleich zusätzlich mit Amplituden** – schreibt eine zweite PDF
  mit r_x und r_y als zweiter Zeile. Nur beim Amplituden-Scan verfügbar.
* **PNG-Übersicht des Scan-Plotters** – das gewohnte Bild der Scan-Skripte,
  zusätzlich zu den PDFs.

### Querschnitt entlang des Minimums (Talschnitt)

Der interessanteste Teil. Idee: einer Größe folgen und pro Waist-Spalte den
Punkt suchen, an dem sie minimal ist. **Genau dort** werden alle angehakten
Größen abgelesen – also nicht deren eigenes Minimum, sondern ihr Wert dort,
wo die Führungsgröße am besten ist.

* **Schnitt entlang** – *Talpfad* (echte Gitterminima, springt aber dort, wo
  das Minimum flach ist oder aus dem Fenster läuft) oder *Gerade* (der
  Schnitt folgt der gefitteten Geraden über den ganzen Bereich, Werte
  zwischen den Gitterzeilen linear interpoliert).
* **Größe für Talpfad/Gerade** – die Führungsgröße. Voreingestellt J, die
  Größe, die auch optimiert wird.
* **Talpunkt-Auswahl** – *Globales Minimum je Spalte* oder *Lokales Minimum
  nahe einer Leitgeraden*. Das ist der Schalter mit dem größten Einfluss auf
  die herauskommende Steigung. Klebt das globale Minimum am Rand des
  Scan-Fensters oder an der Grenze des verbotenen Bereichs, ist es keins –
  dann hilft nur die geführte Variante.
* **Leitgröße** und **Korridor** – nur im geführten Modus. Die Leitgerade ist
  der gewöhnliche lineare Fit einer anderen Größe (Default: Uniformity); pro
  Spalte wird das lokale Minimum genommen, das ihr am nächsten liegt.
* **Aufgetragen über** – Waist (µm oder mm) oder width.
* **Gerade durch den Talpfad legen** – nur über der µm-Achse (siehe unten).
* **Talpfad und ausgelassene/extrapolierte Punkte in die Karte zeichnen** –
  ohne Haken zeigt das linke Panel nur den verbotenen Bereich und die
  Ausgleichsgerade; die Pfadpunkte werden gar nicht erst gezeichnet und stehen
  folglich auch nicht in der Legende (ihre Anzahl steht im Bericht). Mit Haken
  kommen Pfad, Marker und Legendeneinträge zurück. Kommt gar keine Gerade
  zustande, wird der Talpfad auch ohne Haken gezeichnet – sonst wäre die Karte
  leer.
* **Welche Größen** – je mehr Haken, desto mehr y-Achsen. Kurven derselben
  Einheit und Größenordnung teilen sich automatisch eine Achse. Die
  Führungsgröße wird immer mitgezeichnet. (Der Querschnitt durch den Punkt
  weiter oben hängt nicht daran – der zeigt immer r_x und r_y.)
* **Suchbereich einschränken** – gedacht für Datensätze mit mehreren
  Talzweigen: hier sagt man dem Fit, welcher gemeint ist.

### Verbotener Bereich

Die diagonal gegenüberliegenden Eck-Spots dürfen sich nicht überlappen. Der
Haken zeichnet die Grenze nur ein; der zweite Haken schließt die Punkte
darunter aus der Auswertung aus (Score, Bestpunkt, Region, Talpfad). Der
Faktor k sagt, ab wann „überlappt": k = 2 lässt die 1/e²-Radien sich gerade
berühren. Das graue Feld darunter rechnet für den geladenen Datensatz vor,
wie viele Gitterpunkte betroffen wären.

### Score und Region neu berechnen

Ohne Haken gelten alpha und Perzentil aus dem Datensatz. Mit Haken werden
Score, bester Punkt und Region daraus neu bestimmt – ohne neuen Scan. Der
Datensatz auf der Platte bleibt unangetastet; auf Wunsch wird die neue
Fassung als eigene Datei (`…_recomputed.pkl`) daneben gelegt.

---

## Zwei Regeln, die von außen überraschen

1. **Die Gerade gibt es nur über der µm-Achse.** Über `win_input` (mm) ist
   der Zusammenhang zwischen Waist und width gar nicht linear – eine Gerade
   wäre dort über einen schmalen Bereich zwar gut angepasst, aber
   physikalisch bedeutungslos. Über width wäre es dieselbe Beziehung, nur
   andersherum aufgetragen; der Einheitlichkeit halber ist sie auch dort
   gesperrt. Auf den anderen Achsen ist der Haken deshalb leer und grau.
2. **Im Geradenmodus ist der Fit-Haken gesetzt und gesperrt.** Dort *ist* die
   Gerade der Schnitt – ohne sie gäbe es nichts zu zeichnen.

---

## Häufige Stolpersteine

* **„In Results/ liegen keine .pkl-Dateien."** Erst einen Scan rechnen
  (`run_scan.py` oder `run_amp_scan.py`), oder über „Andere Datei…" einen
  Datensatz von anderswo holen.
* **„Keine Gerade möglich."** Der Dialog sagt im Klartext, woran es liegt –
  meistens: fast alle Talminima liegen am Rand des gescannten Fensters, das
  Tal verlässt den Scan also. Abhilfe: Bereich weiter fassen, andere
  Führungsgröße, oder einen Scan mit größerem width-Bereich rechnen.
* **Der Scan scheint zu hängen.** Amplituden-Scans optimieren pro Gitterpunkt
  und brauchen bei feinen Gittern Stunden bis Tage. Die Konsole zeigt den
  Fortschritt; stündlich wird zwischengespeichert, ein Abbruch ist also nicht
  schlimm.
* **Die Überschreib-Rückfrage sieht man nicht.** Sie läuft über die
  **Konsole**, nicht über ein Fenster. In `run_plots.py` ist sie
  standardmäßig aus (Plots sind aus dem Datensatz jederzeit reproduzierbar).
* **Amplituden-Karten sind grau.** Diese Punkte liegen auf einer
  `r_bounds`-Schranke – dort steht kein freies Optimum, sondern die Schranke.
  Wenn das viele Punkte betrifft, war der erlaubte Bereich zu eng.
* **Der Stern ist offen statt gefüllt.** Der markierte Punkt liegt auf dem
  Rand des gescannten Fensters. Das ist dann kein Optimum, sondern nur die
  Stelle, an der der Scan aufhört – das wahre Minimum liegt außerhalb.
* **„nur Waist vorgeben" ist ausgegraut.** Für diesen Datensatz (und diese
  Talschnitt-Einstellungen) lässt sich keine Gerade durch den Talpfad legen –
  ohne sie ist die zweite Koordinate nicht bestimmbar. Entweder eine andere
  Führungsgröße wählen, oder gleich *Waist UND Width vorgeben* nehmen.
