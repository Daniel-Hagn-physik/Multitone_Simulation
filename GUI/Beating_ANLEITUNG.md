# Beating_Multitone_GUI.py

Zeitaufgelöste, **kohärente** Simulation des Multitone-FlatTop-Musters.
Ergänzt `Multitone_Lens_GUI.py` / `Weighted_Multitone_Lens_GUI.py`, ersetzt
sie nicht.

## Warum

Die bisherigen GUIs summieren die **Intensitäten** der Töne:

```
I(x,y) = Σ_s a_s · |u(r − c_s)|²
```

Physikalisch überlagern sich die Töne aber als **Felder**, und jeder Ton trägt
eine andere AOD-Frequenz:

```
E(x,y,t) = Σ_s A_s · u(r − c_s) · exp(i·2π·f_s·t + i·φ_s)
I(x,y,t) = |E|² = Σ_s A_s²|u_s|²                                  ← das Zeitmittel
                + 2 Σ_{s<s'} A_s A_s' u_s u_s' · cos(2π Δf t + Δφ)  ← das Beating
```

Der zweite Term mittelt sich über eine Grundperiode weg — **sofern jede
Differenzfrequenz Δf von Null verschieden ist**. Genau das ist die
Voraussetzung, unter der das bisherige inkohärente Bild gilt, und genau die
ist bei gleicher `width` auf beiden Achsen verletzt (siehe unten).

## Frequenzen

```
f_x(n) = offset + width_x · n/(N_x−1)
f_y(m) = offset + width_y · m/(N_y−1)
f_s(n,m) = f_x(n) + f_y(m)          (beide AODs schieben die Lichtfrequenz)
```

In |E|² steht nur die **Differenz** zweier Spot-Frequenzen. Daraus folgt:

* Die **Wellenlänge** und die **Offset-Frequenz** ändern kein einziges
  Beating. Ein konstanter Versatz kürzt sich aus jeder Differenz heraus.
  (Beide sind trotzdem Eingabefelder — sie gehen in die Geometrie ein:
  λ in die Waist-Umrechnung, offset in die Strahlablenkung.)
* Bei `width_x = width_y = width` sind alle Differenzen Vielfache von
  `f_0 = width / kgV(N_x−1, N_y−1)`. Für 3×4 also `width/6`:
  bei width = 0.35 MHz sind das **58.33 kHz**, Grundperiode **T_0 = 17.14 µs**.

## Der Befund: frequenzentartete Spots

Zwei Spots mit **identischer** Gesamtfrequenz haben einen Kreuzterm bei 0 Hz.
Der läuft nie um, mittelt sich nie weg und erscheint als **statische
Interferenz** — eine systematische Verzerrung des Profils, die die
inkohärenten GUIs nicht sehen.

Bei gleicher `width` auf beiden Achsen gibt es **immer** mindestens ein
solches Paar: die beiden diagonal gegenüberliegenden Eck-Spots,
(n,m) = (0, N_y−1) und (N_x−1, 0), tragen beide `2·offset + width`.

Richtig schlimm wird es, wenn `N_x−1` und `N_y−1` einen gemeinsamen Teiler
haben. Gemessen als maximale Abweichung des Zeitmittels vom inkohärenten
Bild, in Prozent des inkohärenten Maximums (waist = 1.05 µm, width = 0.35 MHz,
Gauß, r_x = r_y = 1):

| N_x×N_y | entartete Paare | Abweichung |
|---|---|---|
| 3×4 | 1 | 1.09 % |
| 2×5 | 1 | 1.00 % |
| 4×5 | 1 | 0.56 % |
| 5×6 | 1 | 0.34 % |
| **3×3** | **5** | **61 %** |
| **3×5** | **7** | **58 %** |
| **4×4** | **14** | **140 %** |
| **5×5** | **30** | **209 %** |
| **6×6** | **55** | **276 %** |

Quadratische Gitter (N_x = N_y) sind der ungünstigste Fall: dort ist jede
Anti-Diagonale n+m = const vollständig entartet. **Für diese Konfigurationen
beschreibt die inkohärente Intensitätssumme das Profil nicht mehr** — die
Uniformity-Zahlen aus den bestehenden Scans sind dort nicht das, was am Atom
ankommt.

Das gewählte 3×4 ist unter diesem Gesichtspunkt eine gute Wahl: `kgV(2,3) = 6`
ist maximal, es bleibt nur die eine unvermeidbare Eck-Entartung mit gut 1 %.

Aufheben lässt sich auch die nur durch **unterschiedliche `width` in x und y**
(Häkchen `width_y = width_x` abwählen). Das ändert allerdings den Spot-Abstand
in y — es ist ein Eingriff in die Geometrie, keine reine Frequenzmaßnahme.

Vorbehalt: das alles setzt voraus, dass die beiden AODs phasenstarr vom selben
Takt laufen und das Licht beide Beugungen kohärent durchläuft. Ist das der
Fall, ist die relative Phase Δφ der entarteten Paare fest, aber **beliebig**
und driftet langsam mit Weglänge und Temperatur — die statische Verzerrung
wandert dann. Mit dem Würfel-Knopf in der Phasengruppe lässt sich dieser Fall durchspielen.

## Tonphasen — was sie können und was nicht

Der Kreuzterm **eines** Spotpaares lautet

```
2 · A_s A_s' · u_s(r) u_s'(r) · cos(2π Δf t + Δφ)
```

Für sich allein lässt sich ein einzelnes Paar durch keine Phase dämpfen — Δφ
verschiebt nur, *wann* das Maximum liegt. Weil sich aber viele Paare dieselbe
Differenzfrequenz teilen (bei 3×4 bis zu elf), addieren sich ihre Beiträge als
Zeiger und können sich teilweise auslöschen. Die zeitliche Varianz ist damit
sehr wohl phasenabhängig — aber nur begrenzt:

| | σ_t/⟨I⟩ im Plateau |
|---|---|
| alle Phasen 0 | 136 % |
| Schroeder | 90 % |
| bestes mit Tonphasen | 70 % |
| absolute Schranke (freie Paarphasen) | 28.7 % |

**Verworfen und entfernt.** Es gab einmal drei weitere Optimierer — auf
breitbandige Zeitvariation, auf minimale Spitzenintensität und auf ein
„Ruhefenster". Alle drei liefen auf nahezu denselben Phasensatz hinaus (72 %,
70 %, 70 %), und für den gepulsten Betrieb ist ohnehin die Pulsfläche die
maßgebliche Größe. Ebenfalls entfernt: der Modus *frei je Spot* (er zeigte, dass
auch volle Freiheit nur auf 55 % kommt — Punkt gemacht, Knopf weg) und das
Newman-Preset (lieferte auf drei Stellen dasselbe wie Schroeder).

**Geblieben** sind die Presets 0 / Schroeder / würfeln, der Quadratur-Haken und
der Puls-Optimierer.

## Wo die Unruhe sitzt: das Spektrum

I(r,t) ist eine Fourierreihe in f₀, und `Var_t(I) = Σ_{d≠0} |D_d|²` zerfällt in
Beiträge einzelner Schwebungsfrequenzen d·f₀. Das Panel **Spektrum der
Schwebung** zeigt sie, mit ν_r und 2ν_r eingezeichnet.

Bei 3×4, alle Phasen 0, als σ_d/⟨I⟩ im Plateau:

| d | Paare | f = d·f₀ | Phasen 0 | breitband optimiert | untere Schranke |
|---|---|---|---|---|---|
| 1 | 10 | 58.3 kHz | 53 % | 12 % | 6.7 % |
| 2 | 11 | 116.7 kHz | 92 % | 38 % | 19.7 % |
| 3 | 10 | 175 kHz | 64 % | 44 % | 12.8 % |
| 4 | 9 | 233 kHz | 35 % | 33 % | 8.3 % |
| 5 | 6 | 292 kHz | 40 % | 12 % | 11.0 % |
| 6…10 | 7…2 | 350…583 kHz | 11…2 % | 11…2 % | — |
| 12 | **1** | 700 kHz | 0.3 % | **0.3 %** | 0.3 % |
| **gesamt** | | | **136 %** | **70 %** | **28.7 %** |

Die Zeile d = 12 ist der Beweis, dass es keine Lösung gibt: diese
Differenzfrequenz erzeugt **ein einziges** Spotpaar (die beiden diagonalen
Eckspots). Es gibt keinen zweiten Zeiger, gegen den sich der Term auslöschen
ließe — sein Beitrag ist streng phasenunabhängig. Dasselbe Argument, auf alle
Ordnungen angewandt, ergibt die **absolute Schranke von 28.7 %** — und die gilt
schon für frei wählbare *Paar*phasen, die physikalisch gar nicht einstellbar
sind. Mit freien Spotphasen kommt man auf 55 %, mit den real einstellbaren
Tonphasen auf 70 %.

**Ein zeitlich stabiles Profil ist über Phasen also nicht erreichbar.**

Was Phasen dagegen sehr wohl können: die Unruhe **umverteilen**. Die Falle
reagiert nicht auf jede Frequenz gleich — weit oberhalb ν_r mittelt das Atom
weg, gefährlich sind ν_r (Aufheizen) und 2ν_r (parametrische Resonanz). Die
Optimierung lässt sich deshalb auf ein Frequenzfenster richten
(*Ziel: nur nahe ν_r und 2·ν_r*):

| Ziel | erreicht | breitbandig danach | d=1 | d=2 |
|---|---|---|---|---|
| Phasen 0 | — | 136 % | 53 % | 92 % |
| breitbandig | 70 % | 70 % | 12 % | 38 % |
| nur d = 1 | **11 %** | 77 % | 11 % | 49 % |
| nur d = 2 | 37 % | 74 % | 18 % | 37 % |
| d = 1 und d = 2 | 40 % | 70 % | 12 % | 38 % |

Die Komponente direkt bei ν_r lässt sich von 53 % auf **11 %** drücken — Faktor
fünf. Bemerkenswert: die breitbandige Optimierung erreicht bei d = 1 und d = 2
schon dasselbe wie die gezielte. Es gibt hier keinen Zielkonflikt, ein
Phasensatz bedient beides.

Eine Eichfreiheit sollte man kennen: φ_s → α + β·k_s lässt **jedes** |D_d|
unverändert (ein linearer Phasenverlauf über die Töne ist wirkungslos). Von den
sieben Tonphasen bleiben damit nur vier physikalisch wirksame Freiheitsgrade —
daher die enge Schranke.

## Ruhefenster: stabil auf Zeit statt global

Global stabil geht nicht (siehe oben). **Auf einem begrenzten Zeitfenster
schon** — und das ist meist das, was zählt, wenn der Puls ohnehin nur ein paar
Mikrosekunden dauert.

Der Knopf **Ruhefenster optimieren** sucht die Tonphasen, bei denen das Profil
während eines Fensters der Länge T_win dem Zeitmittel ⟨I⟩ möglichst nahe kommt.

> **Referenz ist ⟨I⟩, nicht der Mittelwert im Fenster.** Das ist keine
> Feinheit: gegen den Fenstermittelwert optimiert, erfüllt der Optimierer die
> Forderung, indem er das Licht im Fenster schlicht herunterfährt — perfekt
> flach bei 0.3 · ⟨I⟩ und mit völlig falschem Profil. Genau das ist beim Bauen
> passiert, bevor die Referenz korrigiert wurde.

Die Kennzahl wird in zwei Teile zerlegt, weil sie physikalisch Verschiedenes
bedeuten. α(t) ist der beste gemeinsame Skalenfaktor: springt nur α, **atmet
die ganze Fallentiefe** und das FlatTop-Profil bleibt heil. Was nach Abzug von
α übrig bleibt, ist echte **Formänderung** — das ist es, was die Uniformity
zwischen den Sites verdirbt.

Arbeitspunkt 3×4 Airy, waist 1.1 µm, width 0.45 MHz, r_x = 1 / r_y = 1.2,
T₀ = 13.33 µs, gemessen im Plateau:

| Fenster | gesamt | davon Niveau | davon Form | Fallentiefe |
|---|---|---|---|---|
| 0.5 µs | 17 % | — | — | — |
| 1 µs | **21 %** | 7 % | 22 % | 0.80 … 1.06 × |
| 2 µs | 31 % | 15 % | 28 % | 0.60 … 1.10 × |
| 3 µs | **42 %** | 25 % | 33 % | 0.55 … 1.00 × |
| 5 µs | 49 % | — | — | — |
| ganze Periode | 73 % | — | — | — |

Zum Vergleich, ohne Phasenwahl: bei allen Phasen 0 sind es im 3-µs-Fenster
135 %, und die Fallentiefe schwankt zwischen **0.03 und 3.24 × nominal** — die
Falle geht also einmal praktisch aus. Mit dem Fenster-Optimum bleibt sie im
Fenster zwischen 0.55 und 1.00.

Die Lage des Fensters ist **keine** zusätzliche Freiheit: eine Zeitverschiebung
ist selbst ein Phasensatz (φ_s → φ_s + 2π f_s t₀, und das ist separabel, also
als Tonphasen darstellbar). Das Fenster liegt deshalb fest bei t = 0, und der
Optimierer schiebt es implizit an die günstigste Stelle.

Erkauft wird das Fenster **außerhalb** des Fensters: der 3-µs-Satz ist über die
ganze Periode gerechnet mit 81 % schlechter als der global optimierte mit 73 %.
Wer den Rest der Periode nicht braucht, tauscht hier richtig.

## Uniformity im Zeitverlauf

Das Panel **Uniformity U(t) der drei Gebiete** zeigt U = std/mean — dieselbe
Definition wie im Rest des Projekts — zu jedem Zeitpunkt einzeln, für alle drei
Auswerte-Gebiete: Plateau, die Spot-Zentren (also die Fallentiefen
untereinander) und den Kreis mit dem eingestellten Radius. Gepunktet liegt
darin jeweils **U(⟨I⟩)**, der Wert aus dem Zeitmittel — genau die Zahl, die die
Scan-Pipeline optimiert.

Der Abstand zwischen der Kurve und ihrer gepunkteten Linie ist die eigentliche
Aussage:

| Gebiet | U(⟨I⟩) (was die Pipeline sieht) | U(t) im Mittel, Phasen 0 |
|---|---|---|
| Plateau | 17.9 % | 62 % |
| die 12 Spot-Zentren | **6.1 %** | **51 %** |
| Kreis r = 2 µm | 28.0 % | 67 % |

An den Spot-Zentren ist die momentane Uniformity also rund **achtmal** so
schlecht wie der Wert, auf den hin optimiert wird. Ob das zählt, hängt daran,
ob das Atom die Schwebung auflöst — bei f₀ = 75 kHz gegen ν_r = 60.4 kHz tut
es das.

### Live mitverfolgen

Mit dem Haken **U(t) live mitschreiben** (Gruppe *Aktionen*) wächst die Kurve
während der Animation mit: durchgezogen bis zum aktuellen Zeitpunkt, der
restliche Verlauf blass dahinter, ein Punkt auf dem Momentanwert. Im Titel
stehen die drei Momentanwerte als Zahlen — das ist die Anzeige, die man beim
Zuschauen tatsächlich abliest.

Damit das zügig läuft, tauscht das GUI beim Weiterschalten nur die Daten der
vorhandenen Zeichenobjekte aus, statt alle vier Achsen neu aufzubauen. Der
schnelle Pfad liefert pixelgleiche Ausgabe wie der vollständige Aufbau (auf
allen sechs Panels per Hash geprüft) und greift nur, solange Panel, Datenform
und Fadenkreuz unverändert sind — sonst wird automatisch vollständig gezeichnet.

Falls die Anzeige sich merkwürdig verhält, schaltet der Haken **Schnellzeichnen
beim Abspielen** in der Gruppe *Aktionen* auf den vollständigen Neuaufbau
zurück.

> Hier stand einmal ein Absatz über *Blitting* — einen eingefrorenen
> Hintergrund, über den nur die beweglichen Objekte gezeichnet werden. Das war
> im Testcontainer zwanzigmal schneller, hat sich im echten Fenster aber nicht
> bewährt und ist wieder entfernt. Der Grund: bei Blitting sind die beweglichen
> Objekte als `animated` markiert und verschwinden bei jedem normalen Neuzeichnen
> des Fensters (Größe ändern, verschieben, in den Vordergrund holen), während
> der eingefrorene Hintergrund dann nicht mehr zur Fenstergröße passt. Im
> Offscreen-Test ändert sich die Fenstergröße nie, deshalb fiel es dort nicht
> auf.

### Phasen auf Uniformity optimieren

Die Zielgröße im Optimierer ist umschaltbar. Die drei Ziele führen auf
verschiedene Phasen und auf einen echten Zielkonflikt (3-µs-Fenster,
Zielgebiet Spot-Zentren):

| Zielgröße | Fallentiefe α im Fenster | U Plateau | U Zentren | U Kreis |
|---|---|---|---|---|
| — (alle Phasen 0) | 0.03 … 3.23 × | 62 % | 51 % | 67 % |
| Abweichung von ⟨I⟩ | **0.57 … 1.02 ×** | 51 % | 45 % | 55 % |
| Uniformity U(t) | 0.17 … 1.21 × | **38 %** | **23 %** | **44 %** |
| Fallentiefe + Uniformity | 0.53 … 1.52 × | 49 % | 31 % | 56 % |

* **Abweichung von ⟨I⟩** hält die Fallentiefe fast perfekt (0.57 … 1.02), lässt
  die Uniformity aber bei 45 %.
* **Uniformity** halbiert U an den Spot-Zentren auf 23 % — dafür atmet die
  Tiefe wieder um den Faktor sieben.
* **Fallentiefe + Uniformity** ist der Kompromiss. Bewusst *nicht* als Summe
  aus ⟨I⟩-Abweichung und U formuliert — das zählt den Formfehler doppelt und
  landet schlechter als beide Einzelziele. Stattdessen √(α-Schwankung² + U²),
  also genau die beiden Größen, die für die Falle zählen.

Beides zugleich geht nicht: eine gleichmäßige Ausleuchtung bei gleichzeitig
konstanter Höhe ist genau die Forderung „I(r,t) = ⟨I⟩(r)", und die ist über
Phasen nicht erfüllbar. Man wählt, was wichtiger ist — gleiche Tiefen oder
konstante Tiefe.

## Kann man es nicht einfach inkohärent überlagern?

Nein und ja — die Frage zerfällt in zwei Teile.

**Nein, nicht momentan.** „Inkohärent" ist keine Einstellung, sondern das
Ergebnis einer Mittelung. Alle Töne kommen aus demselben Laser und demselben
Takt, sie sind zu jedem Zeitpunkt kohärent. Was man „inkohärente Überlagerung"
nennt, ist genau das Zeitmittel über die Schwebung — und das rechnet das GUI
bereits als ⟨I⟩.

**Ja, für das Zeitmittel — und das ist ein exakter Gewinn.** Der eine Teil der
Kohärenz, der sich *nicht* wegmittelt, sind die frequenzentarteten Paare: ihr
Kreuzterm liegt bei 0 Hz. Er lautet

```
2 g_s g_s' · cos(φ_s − φ_s')
```

und ist **exakt null bei einer Phasendifferenz von 90°**. Dann, und nur dann,
ist das Zeitmittel exakt die inkohärente Intensitätssumme, die die übrigen GUIs
und die Scan-Pipeline rechnen.

Am Arbeitspunkt genügt dafür eine einzige Zahl:

| φ_x(2) | statischer Anteil |
|---|---|
| 0° | 7.07 % |
| 45° | 5.00 % |
| **90° (oder 270°)** | **0.0000 %** |

Und es kostet fast nichts: mit erzwungener Quadratur geht die optimierte
Uniformity an den Spot-Zentren von 23.7 % auf 24.3 %. Der Haken **entartete
Paare in Quadratur halten** ist deshalb voreingestellt und wirkt in allen drei
Optimierern.

### Warum echte Inkohärenz *schlechter* wäre

Naheliegender Gedanke: die Tonphasen schnell würfeln, dann mittelt sich alles
weg. Nachgerechnet ist das ein schwerer Fehler.

Die kohärente Schwebung erzeugt ein **Linienspektrum** bei Vielfachen von f₀.
Am Arbeitspunkt sind das 75, 150, 225 … kHz — und ν_r = 60.4 kHz sowie
2ν_r = 120.8 kHz liegen **zwischen** den Linien. Die Falle bekommt auf ihren
Resonanzen praktisch keine Leistung, trotz 100 % Modulationstiefe.

Würfelt man die Phasen, wird aus dem Linienspektrum ein Kontinuum, und das legt
Rauschleistung genau auf ν_r und 2ν_r. Gemessen am Musterzentrum, Rauschen in
einem 16-kHz-Band:

| Ansteuerung | bei ν_r | bei 2ν_r |
|---|---|---|
| statisch (kohärent) | ~10⁻¹⁸ | ~10⁻¹⁹ |
| gewürfelt @ 300 kHz | 1.8·10⁻¹ | 2.3·10⁻¹ |
| gewürfelt @ 20 MHz | 9.5·10⁻³ | 8.6·10⁻³ |

Das ist kein knapper Vergleich. **Der diskrete Charakter des Spektrums ist
der Schutz** — man sollte ihn nicht zerstören.

> Vorbehalt: die Rechnung nimmt eine feste Fallenfrequenz an. Weil die
> Modulation auch ν_r selbst moduliert, ist die Antwort in Wirklichkeit
> verbreitert, und die 15 Größenordnungen sind eine Idealisierung. Die
> Richtung der Aussage bleibt.

### Die daraus folgende Entwurfsregel

Nicht die Modulationstiefe ist die relevante Größe, sondern ob eine
Schwebungslinie auf ν_r oder 2ν_r fällt. Das GUI prüft das laufend und meldet
den Abstand der nächsten Linie:

| width | f₀ | Abstand zu ν_r | Abstand zu 2ν_r | |
|---|---|---|---|---|
| 0.35 MHz (alt) | 58.3 kHz | **2.1 kHz** | **4.1 kHz** | kritisch |
| 0.45 MHz (jetzt) | 75.0 kHz | 14.6 kHz | 29.2 kHz | unkritisch |

Der Wechsel auf width = 0.45 MHz hat also nebenbei genau das Richtige getan.

## Gepulster Betrieb: die Pulsfläche ist die relevante Größe

Wird das Profil **gepulst** eingestrahlt, um Rabi-Übergänge zu treiben, ist
weder die Momentanintensität noch das Zeitmittel die maßgebliche Größe, sondern
die **akkumulierte Rabi-Fläche**

```
θ(r) = ∫_{t₀}^{t₀+T_p} Ω(r,t) dt
```

Der Puls integriert über die Schwebung. Ihre Gleichmäßigkeit über das
Auswertegebiet bestimmt, wie einheitlich der Drehwinkel der Atome wird.

Weil I eine Fourierreihe in f₀ ist, lässt sich das Integral für Ω ~ I
geschlossen angeben — kein Zeitschritt, keine Diskretisierungsfehler:

```
θ = T_p·D₀ + 2 Re[ Σ_{d>0} D_d · G_d ],   G_d = (e^{i d ω₀ (t₀+T_p)} − e^{i d ω₀ t₀}) / (i d ω₀)
```

### Die harte Bedingung: der Puls muss getriggert sein

π-Puls-Dauer T = 1/(2 f_Rabi), also 5 µs bei 0.1 MHz bis 0.5 µs bei 1 MHz —
gegen eine Schwebungsperiode von 13.33 µs. **Ohne feste Lage des Pulses im
Schwebungszyklus schwankt die Pulsfläche von Schuss zu Schuss:**

| f_Rabi | T_π | T_π / T₀ | Fläche schwankt um |
|---|---|---|---|
| 0.1 MHz | 5.00 µs | 0.375 | 2.2 × |
| 0.2 MHz | 2.50 µs | 0.188 | 18.7 × |
| 0.5 MHz | 1.00 µs | 0.075 | 55.6 × |
| 1.0 MHz | 0.50 µs | 0.037 | 73.1 × |

Faktor 73 im Drehwinkel macht kohärente Operationen unmöglich. Der Puls **muss**
auf die AWG-Wellenform getriggert werden. Das GUI weist darauf hin und nennt den
Faktor.

### Ist der Puls getriggert, ist die Lage sogar ein Vorteil

Mit fester Pulslage t₀ und optimierten Tonphasen wird die Uniformity der
Pulsfläche im 2-µm-Kreis **besser als der Zeitmittelwert** — der Puls
integriert die Schwebung so, dass sie die räumliche Ungleichmäßigkeit teilweise
kompensiert:

| f_Rabi | U bei t₀ = 0 | bestes t₀, Phasen 0 | Phasen **und** t₀ optimiert | t₀ |
|---|---|---|---|---|
| 0.1 MHz | 35.3 % | 18.6 % | **14.2 %** | 9.64 µs |
| 0.2 MHz | 46.4 % | 20.3 % | **11.4 %** | 0.64 µs |
| 0.5 MHz | 57.0 % | 31.9 % | **18.4 %** | 4.93 µs |
| 1.0 MHz | 67.3 % | 34.2 % | **19.5 %** | 2.06 µs |

Zum Vergleich: **U(⟨I⟩) = 27.8 %** im selben Kreis — das ist der Wert, den das
inkohärente Bild liefert. Bei 0.1 und 0.2 MHz liegt die optimierte Pulsfläche
deutlich darunter.

### Welche Physik im Anregungsmodell steckt

Die Anregung kommt allein aus der Pulsfläche: P(r) = sin²(θ(r)/2) mit
θ(r) = ∫Ω dt. Für ein **resonantes** Zweiniveausystem mit beliebig
zeitabhängigem Ω ist das **exakt**, keine Näherung — gegen eine
schrittweise Integration der Schrödingergleichung auf sechs Stellen geprüft.

Das **Kopplungsgesetz** ist die entscheidende Wahl:

* **Ω ~ I** — beide Raman-Äste kommen aus diesem Multiton-Strahl (z. B. ein EOM
  setzt die 3.035-GHz-Seitenbänder hinter den AOD). Dann paart sich jeder Ton
  mit seinem eigenen Seitenband, alle Paare sind zweiphotonen-resonant, und die
  kohärente Summe über alle Tonpaare ergibt exakt Ω ~ |E|² = I. **Hier ist das
  Modell exakt.**
* **Ω ~ √I** — nur ein Ast läuft über den AOD. Dann ist Ω ~ E **komplex**, und
  jeder Ton sitzt bei seiner eigenen Zweiphotonen-Verstimmung (Vielfache von f₀,
  bis ±450 kHz — in derselben Größenordnung wie Ω selbst). Das GUI behält nur
  |Ω| ~ √I und lässt die Phase weg; gegen die exakte komplexe Lösung kostet das
  bis zu 2.5 % Anregung bei 0.2 MHz. Dieser Zweig ist als Anhaltspunkt zu lesen.

Die **Lichtverschiebung** ist über das Feld `light shift eta` = δ/Ω drin. Bei
einem Raman-Übergang skalieren δ und Ω mit derselben Intensität, ihr Verhältnis
ist also orts- und zeitunabhängig, und das Problem schließt sich wieder:

```
P = 1/(1+η²) · sin²( √(1+η²) · θ/2 )
```

numerisch bestätigt. Die Lichtverschiebung **deckelt also den Kontrast, erzeugt
aber keine zusätzliche räumliche Ungleichmäßigkeit** — alle Uniformity-Zahlen
dieses GUIs bleiben davon unberührt. Kompensiert man den Mittelwert durch
Verstimmen des Raman, holt man das meiste zurück (bei η = 0.5: 0.77 → 0.93).

**Nicht im Modell**, und beim Vergleich mit der Messung zu bedenken: statische
Zweiphotonen-Verstimmung und Magnetfeldshifts; spontane Emission über den
Zwischenzustand; die Zeeman-Substruktur von Rb-85 (F = 2 mit 5, F = 3 mit 7
Unterzuständen, je eigener Clebsch-Gordan-Faktor — ein sauberer π-Puls setzt ein
geschlossenes Paar voraus); Atombewegung während des Pulses (jeder Ort wird
eingefroren behandelt, was zum Schuss-zu-Schuss-Zittern des Strahlzeigers passt,
nicht zu einem im Topf schwingenden Atom); endliche Pulsflanken und die
Füllzeit des AOD; Polarisation und Vektor-Lichtverschiebung.

Die 3.035 GHz selbst tauchen in keiner Formel auf: sie entscheiden nur, **welche**
Lichtkomponenten das Raman-Paar bilden — die AOD-Töne liegen nur einige 100 kHz
auseinander. Ob die Aufspaltung innerhalb oder außerhalb des Multiton-Pfades
überbrückt wird, ist genau das, was das Kopplungsgesetz oben kodiert.

### Der Trigger, praktisch

„Triggern" heißt nur: jeden Schuss an derselben Stelle des Schwebungszyklus
beginnen. Der AWG spielt die Multiton-Wellenform mit Periode T₀ = 13.33 µs;
der Lichtpuls braucht eine feste Verzögerung dazu. Zwei Wege:

* Der **AWG erzeugt den Puls selbst** (Burst statt Dauerbetrieb) — dann liegt
  die Phase per Konstruktion fest. Vorbehalt: der AOD braucht seine Füllzeit
  (Strahldurchmesser / Schallgeschwindigkeit), bei ~2 mm Strahl je nach
  Kristall 0.5 bis 3 µs. Bei einem 5-µs-Puls spürbar, bei 0.5 µs unbrauchbar.
* Die Töne laufen durch, ein separater **AOM schneidet den Puls heraus** —
  dann Marker-Ausgang des AWG als Startsignal, Loop-Länge genau T₀ (oder ein
  Vielfaches), einstellbare Verzögerung darauf.

### Flacher Punkt statt bestem Punkt

Die Flächenkurve A(t₀) hat steile und flache Stellen. Auf einer flachen Stelle
verschwindet dA/dt₀, und Zeit-Jitter wirkt erst in zweiter Ordnung. Das ist
weit mehr wert als das letzte Prozent Uniformity:

| t₀ liegt … | U | Toleranz für 1 % Flächenfehler |
|---|---|---|
| auf dem U-Minimum (steil) | 18.3 % | 44 ns |
| auf einer flachen Stelle | 19.9 % | **12 800 ns** |

Faktor 290 an Timing-Toleranz für 1.6 Prozentpunkte. Bei 1 MHz ist es 5 ns
gegen 533 ns — dort existenziell. Der Knopf **Move t_0 to a flat point of the
area** setzt t₀ dorthin.

### Was man am Ende misst

Das Panel **Rabi oscillation** zeigt die Anregung über die Pulsdauer, gemittelt
über das Zielgebiet, für drei Fälle: ideal ohne Schwebung, getriggert beim
aktuellen t₀, und ungetriggert (über zufällige Pulslagen gemittelt). Bei
0.1 MHz und optimierten Phasen:

| | Kontrast |
|---|---|
| ideal | 100 % |
| getriggert auf den flachen Punkt | 84 % |
| getriggert auf den steilen Punkt | 78 % |
| ohne Trigger | 76 % |

Bei 1 MHz dagegen 80 % mit gegen 41 % ohne Trigger — je kürzer der Puls, desto
mehr hängt alles am Timing.

Die Kurven werden erst berechnet, wenn das Panel gewählt ist (etwa eine
Sekunde), und danach zwischengespeichert.

### Bedienung
### Bedienung

Die Gruppe **Gepulster Betrieb** hat drei Eingaben und einen Knopf:

| Feld | Bedeutung |
|---|---|
| f_Rabi | Ω/2π; daraus folgt T_π = 1/(2 f_Rabi) |
| Pulsbeginn t₀ | Lage des Pulses im Schwebungszyklus |
| Kopplung | Ω ~ I (Zweiphotonen-Raman, beide Äste aus diesem Profil) oder Ω ~ √I (dieses Profil ist nur ein Ast) |
| *Move t_0 to a flat point* | setzt t₀ auf eine jitter-tolerante Stelle |
| *Optimise phases and t_0* | minimiert U(θ) im Zielgebiet, etwa 30 s |

Das Panel **Pulsfläche: U über den Pulsbeginn** zeigt U(θ) als Funktion von t₀
über einen ganzen Schwebungszyklus, mit dem besten t₀ markiert und der
Zeitmittel-Referenz als gepunkteter Linie.

Die Kopplung macht einen spürbaren Unterschied: mit Ω ~ √I liegt das beste t₀
bei 11.1 % statt 20.3 %, weil die Wurzel den Dynamikbereich staucht.

**Zielgebiet ist jetzt der 2-µm-Kreis** um die MusterMitte — Voreinstellung im
Feld *Zielgebiet*, Radius im Feld daneben.

## Was die Modulation tatsächlich reduziert

Nur zwei Dinge:

**1. Weniger Überlapp.** n_eff(r) = I_max/⟨I⟩ ist exakt die Zahl der am Ort
wirksam überlappenden Töne, und zugleich der Faktor, um den die
Momentanintensität bei Rephasierung über dem Zeitmittel liegt. Bei waist
1.05 µm (Spot-Abstand 1.11 / 0.74 µm) liegt n_eff im Plateau bei 5, bei waist
0.35 µm bei 1.1 — und dort ist die Modulationstiefe nur noch 6 %. Der Haken:
ohne Überlapp kein flaches Dach. **Modulationstiefe und Flachheit sind dieselbe
Größe von zwei Seiten.**

**2. Die Schwebung aus dem Ansprechbereich des Atoms schieben.** Alle
Beat-Frequenzen skalieren mit `width`, die Spot-Abstände dagegen mit dem
**Produkt** (f1/f2)·width. Beide Größen lassen sich also gegeneinander
verrechnen: `width` ×10 und `f1/f2` ÷10 ergibt exakt dasselbe Spot-Muster
(numerisch auf 6·10⁻⁶ geprüft), aber zehnfach schnellere Schwebung.

| | jetzt | Beispiel-Alternative |
|---|---|---|
| width | 0.35 MHz | 3.5 MHz |
| f1 / f2 | 75 / 750 mm | 7.5 / 750 mm |
| Spot-Muster | — | identisch |
| f_0 | 58.3 kHz | 583 kHz |
| f_0 / ν_r (ν_r = 60.4 kHz) | 0.97 | 9.7 |
| nötiger Eingangswaist für w_out = 1.05 µm | 1.27 mm | 0.127 mm |

Bei f_0 ≈ ν_r sieht das Atom die Modulation quasi-resonant; erst bei f_0 ≫ ν_r
mittelt es sie weg, und dann — und erst dann — ist das inkohärente Bild der
bestehenden GUIs die richtige Beschreibung. Der Preis ist ein zehnfach
kleinerer Strahl auf dem AOD; die Zahl der auflösbaren Spots (τ·Δf) bleibt
dabei unverändert, weil Apertur und Bandbreite gegenläufig skalieren.

## Bedienung

Alles wird eingetippt, es gibt keine Slider außer für die Zeitachse.

**Voreingestellt ist der Arbeitspunkt**, das GUI startet immer damit:

| | |
|---|---|
| Töne | 3 × 4 |
| Profil | **Airy**, Faktor 1.4830 → erster Nullring 1.631 µm |
| waist | **1.10 µm** (entspricht 0.973 mm vor der Linse) |
| width | **0.45 MHz** (x und y gekoppelt) |
| Amplituden | r_x = 1.0, r_y = **1.2** → amp_y = [1.2, 1, 1, 1.2] |
| Optik | f1 = **60 mm**, f2 = 750 mm, λ = 795 nm, Offset 100 MHz |

Daraus folgt: Spot-Abstand 1.137 µm in x und 0.758 µm in y, Grundperiode
T₀ = 13.33 µs (f₀ = width/6 = 75.0 kHz, also 1.24 · ν_r), eine
frequenzentartete Eck-Paarung mit 7.1 % statischem Anteil, und ohne
Phasenoptimierung σ_t/⟨I⟩ = 107 % im Plateau (mit Optimierung 73 %).

| Gruppe | Feld | Bedeutung |
|---|---|---|
| Töne | N_x, N_y | Tonzahl je Achse |
| Strahlprofil | Gauß / Airy | Feldprofil eines Spots. Beim Airy tragen die Ringe ein **negatives** Vorzeichen — für die kohärente Summe wesentlich. |
| | Airy-Faktor | `first_zero_radius = Faktor · waist`, Voreinstellung 1.4830 aus `airy_scale.py` |
| Strahl / Optik | Modus | waist nach der Linse (µm) oder davor (mm); die jeweils andere Größe wird nachgezogen |
| | width x / width y | Frequenzspanne je Achse; Häkchen koppelt beide |
| | Wellenlänge, Offset | gehen in die Geometrie ein, **nicht** in die Beat-Frequenzen |
| | f1, f2 | Teleskop |
| Amplituden | r_x, r_y | Außen/Innen-Verhältnis wie in `amps_from_ratio()`: `amp_x = [r_x, 1, …, 1, r_x]`. Das sind **Intensitäts**-Gewichte, das Feld trägt die Wurzel. |
| Zeitachse | Perioden | Fensterlänge in Grundperioden |
| | Frames/Periode | Abtastung. **Muss > 2·(höchste Beat-Frequenz / f_0) sein**, sonst Aliasing — das GUI warnt und nennt die nötige Zahl. |
| Tonphasen | φ_x, φ_y je Ton (Grad) | eintippbar; Knöpfe für 0 / Schroeder / Newman / würfeln / Spitze minimieren |
| | Play / fps / t | Animation bzw. manuelles Durchfahren |

Ein Klick in die 2D-Karte setzt das Fadenkreuz; Orts-Zeit-Karte, Schnitt und
I(t) beziehen sich danach auf diesen Punkt.

## Die vier Plots

1. **I(x,y,t)** — Momentaufnahme. Kreise markieren frequenzentartete Spots.
2. **Orts-Zeit-Karte I(x,t)** entlang des x-Schnitts über das ganze Fenster —
   hier stehen die Schwebungsstreifen als Muster, ohne Animation lesbar.
3. **x-Schnitt** bei t, mit Min/Max-Hüllkurve über die Zeit und dem Zeitmittel
   als gestrichelter Referenz — das Zeitmittel ist das, was die bisherigen
   GUIs zeigen.
4. **I(t) am Fadenkreuz** über dieselbe Zeitspanne, mit Modulationstiefe.

## Farbskala

Stehen alle Tonphasen auf 0, rephasieren die Töne einmal pro Grundperiode zu
einem kurzen Puls — bei 3×4 mit **7.5× dem Maximum des Zeitmittels**. Eine
feste Skala auf dieses Maximum lässt alle übrigen Frames fast schwarz, deshalb
ist das 99.5-Perzentil voreingestellt. Der Spitzenwert steht im Titel.

## Zum Vergleich mit den anderen GUIs

Die Statuszeile prüft bei jedem Lauf, ob das Zeitmittel des Würfels mit der
inkohärenten Summe `Σ_s a_s |u_s|²` übereinstimmt — also mit genau dem, was
`Multitone_Lens_GUI.py` rechnet. Ohne entartete Spots stimmt es auf ~1e-6
genau. Weicht es ab, steht der Betrag dort und kommt von der statischen
Interferenz, nicht von der Numerik.

## Ein-Linsen-Aufbau (Labor, Messbild)

Der Aufbau für das Messbild hat kein Teleskop und kein fLO — hinter dem AOD
steht **eine** Linse. Checkbox **Use one lens** in *Beam / optics* schaltet
darauf um; `f1`/`f2` werden dann ignoriert, stattdessen zählt `f (single
lens)`. Es ändern sich genau zwei Formeln:

| | Teleskop-Aufbau | eine Linse |
|---|---|---|
| Ortsablage | `r = (f1·fLO/f2)·tan θ` | `r = f·tan θ` |
| Waist | `w₀ = (f1/f2)·λ·fLO/(π w_in)` | `w₀ = λ·f/(π w_in)` |

Der AOD bleibt unangetastet: `θ(f) = θ_max·(f−offset)/f_band` mit θ_max =
43 mrad und f_band = 36 MHz, also v_ac = λ·f_band/θ_max = 665.6 m/s.

### Die eine Zahl, auf die es ankommt

Aus beiden Formeln folgt der Tonabstand, der genau einen Spotabstand von einem
Waist erzeugt:

```
df(pitch/waist = 1) = w₀ / (dr/df) = v_ac / (π·w_in)
```

**Die Brennweite kürzt sich heraus.** Sie legt nur fest, wie groß das Bild
insgesamt wird, nicht, wie viele Töne man braucht. Für w_in = 1.75 mm sind es
121.06 kHz — bei f = 45 mm gehört dazu w₀ = 6.51 µm und pitch = 6.51 µm.

### Ziel-Beating-Periode

`|E|²` enthält nur Differenzen der Tonfrequenzen, die Grundperiode ist
`T₀ = 1/f₀` mit `f₀ = ggT` aller Differenzfrequenzen. Bei gleicher width auf
beiden Achsen ist

```
f₀ = width / kgV(N_x−1, N_y−1)     ⇔     width = kgV(N_x−1, N_y−1) / T
```

Damit steht die width fest, sobald das Gitter steht — der Knopf **Design for a
target beating period …** sucht die Gitter ab und listet die Sätze, die die
geforderte Periode *exakt* treffen, sortiert nach Abweichung von einem
Ziel-`pitch/waist`. Doppelklick oder *Markierten Satz übernehmen* setzt N_x,
N_y, width_x, width_y, Brennweite, Eingangswaist und Belichtungszeit im
Haupt-GUI und rechnet neu; Bilder pro Periode, Periodenzahl und
Gitterauflösung werden dabei auf etwas Bezahlbares gesetzt.

Die Suche hat einen unangenehmen Zug, den man kennen sollte: bei **gleicher**
width zwingt eine lange Periode entweder zu dichten Spots oder zu vielen
Tönen. Für T = 100 µs (f₀ = 10 kHz) und pitch/waist ≈ 1 braucht es
`kgV/(N−1) ≈ 12`, also teilerfremde `N−1` um 12 herum — **13×14 Töne bei
width = 1.56 MHz**. Wer weniger Töne will, hat zwei Wege:

* **dichter setzen** — `pitch/waist` kleiner als 1 ist kein Fehler. Die Spots
  überlappen dann stärker, das Zeitmittel wird *glatter*, nur das Feld kleiner.
  Das Ziel-`pitch/waist` im Dialog ist genau dafür da.
* **ungleiche widths zulassen** (Checkbox im Dialog). `df_x = 120 kHz` und
  `df_y = 130 kHz` haben ebenfalls ggT 10 kHz, brauchen aber nur **3×4 Töne**
  (width_x = 0.24 MHz, width_y = 0.39 MHz) — und heben die Frequenzentartung
  nebenbei auf. Der Spotabstand in y ist dann 7 % größer als in x.

### Belichtungszeit der Kamera

Das Feld **Camera exposure** in *Time axis* rechnet mit, was eine Kamera mit
endlicher Belichtung sieht. Eine Belichtung `t_exp` ist ein Boxcar in der Zeit;
auf dem Fourierkoeffizienten der Beat-Ordnung `d` wirkt sie als

```
D_d → D_d · sinc(d·f₀·t_exp),        sinc(z) = sin(πz)/(πz)
```

Bei t_exp = 20 µs und f₀ = 10 kHz bleibt die Grundschwingung mit 94 % fast voll
stehen, während 120/130 kHz auf 8 % gedämpft werden — die Kamera sieht gerade
das langsame Beating und mittelt das schnelle weg. Im 13×14-Satz sinkt die
Modulationstiefe im Plateau dadurch von 100 % auf 24 %, σ_t/⟨I⟩ von 86 % auf
13 %. Das Zeitmittel bleibt unberührt (sinc(0) = 1).

Die Belichtung wirkt auf **alle Bilder und die σ_t-Karte**, nicht auf die
Puls- und Rabi-Analyse: das Atom integriert nicht.

### Rechenzeit bei vielen Tönen

`time_stats_exact()` läuft nicht mehr über alle Spotpaare (O(S²)), sondern
komprimiert die Spots auf ihre **Beat-Ordnungen** und wertet I(t) mit einer FFT
an 2K+1 Stützstellen exakt aus. 182 Spots auf einem 140²-Gitter mit 625 Bildern
brauchen damit rund 3 s statt Minuten. Die alte Fassung steht als
`time_stats_exact_pairs()` daneben und stimmt auf 1e-15 damit überein.

Puls- und Spektrumanalyse laufen weiter über alle Paare und werden ab 64 Spots
übersprungen; die Checkbox *pulse / spectrum also at many spots* erzwingt sie.

### Kamera-Bildserie

Knopf **Camera frame series …** in *Actions*. Das Fenster zeigt, was eine Kamera
aufnimmt, wenn man den Trigger-Delay in Schritten über eine Grundperiode
durchfährt: obere Reihe die Rohbilder, untere Reihe die **Abweichung vom
Zeitmittel** — das ist die Größe, die man im Labor auswertet, weil der
Untergrund herausfällt und die Schwebung als Vorzeichenmuster dasteht.

Einstellbar sind Belichtung, Schrittweite des Trigger-Delays (getrennt, weil
jedes Bild ein eigener Schuss ist und man auch überlappend abtasten kann) und
die Bildzahl. Voreinstellung: lückenlos über genau eine Periode.

Das Fenster ist **nicht modal** und bleibt offen. Der Ablauf zum Ausprobieren
von Phasen ist: Phasen im Haupt-GUI setzen → *Recompute* → im Serienfenster
*Neu zeichnen*. Die Kennzahlzeile darunter nennt den Hub pro Pixel im Plateau
(Median und p90) und ob das Gesamtlicht im Plateau mitschwankt oder ob es eine
reine Umverteilung ist — Letzteres ist der bessere Fall, weil es dann nicht mit
einer Laserleistungsdrift verwechselt werden kann.

Für 13×14 bei 20 µs Belichtung:

| Tonphasen | Hub/Pixel Median | p90 | Plateau-Gesamtlicht | Crest RF x/y |
|---|---|---|---|---|
| alle 0 | 16 % | 36 % | schwankt 15 % | 5.10 / 5.29 |
| **Schroeder** | **52 %** | 68 % | 0.9 % | **1.88 / 2.00** |
| zufällig | 65 % | 102 % | 3 % | 2.6 / 2.6 |

Schroeder-Phasen sind hier in jeder Hinsicht die richtige Wahl: dreimal so viel
Kontrast wie bei Phasen 0, ein um den Faktor 2.7 kleinerer Crest-Faktor der RF,
und eine reine Umverteilung ohne Helligkeitsänderung.

### Crest-Faktor: Formel und eine korrigierte Mittelung

`schroeder_phases(N)` setzt die klassische Schroeder-Vorschrift für gleiche
Amplituden,

```
phi_n = -pi * n(n-1)/N ,        n = 0 ... N-1
```

ein quadratischer Phasenverlauf, also ein linearer Frequenz-Chirp über die
Töne. Die Momentanleistung verteilt sich damit gleichmäßig über die
Hüllkurvenperiode statt in einen Puls zu rephasieren. Sie folgt aus Schroeders
allgemeiner Formel für eine Leistungsverteilung p_l (Σp_l = 1),
`phi_n = -2π Σ_{l<n} (n-l)·p_l`, mit p_l = 1/N.

`crest_factor()` rechnet ohne Träger: mit `s(t) = Re[e^{2πi f_c t}·A(t)]` ist
Spitze/Effektivwert `= √2·max|A| / rms|A|`. **Das Mittelungsfenster ist eine
Periode der Hüllkurve, 1/f_env mit f_env = ggT der Tonabstände — nicht
1/span.** Eine frühere Fassung nahm 1/span; bei 13 Tönen mit 130 kHz Abstand
ist die Spanne das Zwölffache des Abstands, das Fenster liegt dann komplett
innerhalb der Rephasierungsspitze und der Effektivwert kommt viel zu groß
heraus: 2.19 statt der korrekten 5.10 = √(2N) bei Phasen 0. Der Fehler wächst
mit der Tonzahl, also genau dort, wo die Zahl gebraucht wird. Probe: Phasen 0
muss exakt √(2N) geben. Die RF-Amplituden (r_x, r_y) gehen jetzt mit ein.

Die Bilder werden nicht aus dem Würfel gemittelt, sondern exakt gerechnet
(`camera_frames_exact()`): I(t) ist ein trigonometrisches Polynom in
`exp(2πi f₀t)`, die Belichtung ab t₀ wirkt darauf als

```
I_cam(t₀) = Σ_d C_d · sinc(d·f₀·t_exp) · exp(2πi·d·f₀·(t₀ + t_exp/2))
```

für **beliebige** t₀ und t_exp — kein Zeitraster, keine Rundung, kein Aliasing.
Probe: `t_exp = T₀` muss exakt das Zeitmittel geben, die untere Reihe wird dann
null.

## Pulsflaeche und Trigger-Jitter

Knopf **Pulse area / trigger jitter …** in *Actions*. Ein Rechteckpuls der Länge
T_p wird auf das Zeitfenster höchster Intensität getriggert, seine akkumulierte
Rabi-Fläche

```
θ(r, t₀) = ∫_{t₀}^{t₀+T_p} Ω(r,t) dt
```

berechnet und über den Trigger-Fehler durchgefahren. Der große Plot unten ist
genau das: Fläche über dem Delay, 0 = perfekt getriggert, ±Grenze frei
einstellbar (Vorgabe ±1 µs, 401 Punkte).

### Der Punkt, den man leicht übersieht

Ein Puls ist derselbe Boxcar in der Zeit wie eine Kamerabelichtung, nur
tausendmal kürzer. Auf die Beat-Ordnung d wirkt er als `sinc(d·f₀·T_p)`. Bei
T_p = 1 µs und f₀ = 10 kHz bleibt die Grundschwingung bei 0.9999 und 120 kHz
noch bei 0.976 — der Puls **mittelt die Schwebung nicht weg, er tastet sie ab**.
Deshalb hängt die Fläche empfindlich vom Trigger ab, und zwar auf der Zeitskala
der *schnellen* Beats (einige µs), nicht auf der der Grundperiode.

### Normierung

```
Ω(r,t) = 2π·f_rabi · g(r,t)/⟨g⟩ ,     g = I  bzw.  √I
```

⟨g⟩ ist das Mittel über **Zeit und Bereich** — f_rabi ist also die
Rabi-Frequenz, die man auf dem zeitgemittelten Profil messen würde. Ein Puls auf
dieser Referenz hat exakt θ = 2π·f_rabi·T_p; bei 1 MHz und 1 µs also 2π. Der
π-Puls wäre 0.5 µs lang, dafür gibt es den Knopf.

### Trigger

Der automatische Trigger sucht das Maximum der mittleren Fläche über eine ganze
Grundperiode (mit parabolischer Nachziehung). Dort ist dθ/d(Delay) = 0, der
Jitter geht also nur in **zweiter** Ordnung ein — der eigentliche Grund, auf ein
Maximum und nicht auf eine Flanke zu triggern. Der Trigger lässt sich auch von
Hand setzen und ins Haupt-GUI übernehmen.

### Was für 13×14 mit Schroeder-Phasen herauskommt

Bei Ω/2π = 1 MHz, T_p = 1 µs, Auswertebereich Plateau:

| | |
|---|---|
| optimaler Trigger | t₀ = 2.89 µs im Zyklus |
| θ(0) | 2.30 π (Referenz 2.00 π, also +15 %) |
| Spanne über die Periode | 1.87 … 2.30 π |
| Abweichung bei ±1 µs | −9.7 % |
| Toleranz für 1 % Flächenfehler | ±290 ns |
| σ_θ/⟨θ⟩ im Plateau | **74 %** (Zeitmittel: 11 %) |

Die letzte Zeile ist die eigentliche Nachricht: ein 1-µs-Puls sieht praktisch
eine Momentaufnahme, und die ist räumlich siebenmal so ungleichmäßig wie das
Zeitmittel. Mit längeren Pulsen läuft das zurück — T_p = 5 µs gibt 44 %,
T_p = 20 µs gibt 23 %, und die Trigger-Toleranz wird gleichzeitig unkritisch
(±1 µs kostet dann nur noch 0.5 % Fläche).

### Rechnung

Für Ω ∼ I geschlossen und exakt über dieselben Fourierkoeffizienten wie die
Kamerabilder: die mittlere Fläche für beliebig viele Delays ist ein
Matrix-Vektor-Produkt (`beat_coeffs_mean()` + `pulse_area_curve()`), gegen die
alte paarweise `PulseArea` auf 1e-16 geprüft. Für Ω ∼ √I gibt es keine
geschlossene Form; dort wird ⟨√I⟩ auf einem feinen Zeitraster per FFT
ausgewertet und das Integral als laufende Summe gebildet (`sqrt_mean_series()`,
`sqrt_area_curve()`), Übereinstimmung mit der numerischen Referenz auf 1e-3.

Gespeichert werden PDF, eine Parameterdatei und `_scan.txt` mit der Kurve als
Zahlenkolonnen (Delay, θ/π, Abweichung in %).

### Anregung, nicht Fläche

Die Fläche θ ist linear in Ω, die Anregung ist es nicht. Weil sin² nichtlinear
ist, gilt ⟨sin²(θ/2)⟩ ≠ sin²(⟨θ⟩/2), und bei 74–77 % räumlicher Streuung liegen
die beiden weit auseinander. Das Fenster mittelt deshalb die **Anregung über die
Orte**, nicht die Fläche: für 13×14 bei T_p = 0.5 µs (dem nominellen π-Puls)
kommt ⟨sin²(θ/2)⟩ = 0.48 heraus statt der 0.94, die man aus der mittleren Fläche
naiv ablesen würde. Beide Zahlen stehen in der Infozeile nebeneinander.

Der differentielle Lichtshift η aus dem Haupt-GUI geht mit ein:
`P = 1/(1+η²)·sin²(√(1+η²)·θ/2)`. Bei η = 0.442 (Δ = −8 GHz) sinkt die Anregung
weiter von 0.48 auf 0.41. η ist orts- und zeitunabhängig, weil Ω und δ_LS
denselben Faktor I tragen — der Lichtshift deckelt den Kontrast, fügt aber keine
zusätzliche räumliche Struktur hinzu.

## Leistung und Intensität

Knopf **Power / intensity …** in *Actions*. Die Simulation kennt I nur bis auf
eine Konstante; eine physikalische Vorgabe legt sie fest, und der Dialog rechnet
in beide Richtungen — Rabi-Frequenz → Leistung oder Leistung → Rabi-Frequenz.

Die Kette:

1. **Atomphysik** aus `kern/rb85_raman.py`: adiabatische Elimination der
   5P-Zustände mit voller, vorzeichenrichtiger Summe über D1 *und* D2 samt
   Hyperfeinstruktur. Ergebnis `Ω = C_rabi·I`, `δ_LS = C_shift·I`,
   `Γ = C_scatter·I`, alle linear in I. Für Δ = −8 GHz, σ⁺/σ⁺ kopropagierend,
   β = 0.5, m_F = 0: **C_rabi = 71.98 rad/s pro W/m²**, η = −0.442.
2. **Bezug**: Ω/2π gilt für das *Zeitmittel* der Intensität im gewählten Bereich
   — dieselbe Konvention wie im Pulsfenster.
3. **Fläche**: `P = I_ref·A_eff` mit `A_eff = ∫I dA / ⟨I⟩_Bereich`. Das
   Flächenintegral kommt **analytisch** aus der Summe der Einzelspots
   (`profile_total_power()`), nicht aus dem Rechengitter — beim Airy-Profil
   fehlen dort 0.55 % der Leistung in den abgeschnittenen Ringen.
4. **Aufteilung**: Spot (n,m) ∝ a_x(n)²·a_y(m)²; RF-Ton n der x-Achse ∝
   a_x(n)²/Σa_x², weil er alle Spots seiner Spalte speist.

Für den Ein-Linsen-Satz 13×14 bei Ω/2π = 1 MHz:

| | |
|---|---|
| Referenzintensität | 8.73 W/cm² |
| A_eff | 9469 µm² |
| Leistung im Profil | **0.83 mW** |
| pro Spot | 4.5 µW (hellster 6.2) |
| pro RF-Ton x / y | 64 µW / 59 µW (stärkster y-Ton 80 µW) |
| Streuung pro 1-µs-Puls | 0.81 % |

Zum Vergleich der Standard-Arbeitspunkt 3×4 mit 1.1 µm Waist: A_eff = 15 µm²,
also 1.3 µW — der Faktor 600 ist reine Fläche.

**Nicht enthalten**: Beugungseffizienz des AOD, Transmission der Optik,
Intermodulation. Angegeben ist die Leistung *im Profil*, also nach dem AOD.

Fehlt `arc` (das `rb85_raman` braucht), bleibt der Dialog benutzbar: C_rabi wird
dann als Eingabefeld freigeschaltet, voreingestellt auf 71.985.

### Atomgewichtet — die einzige Auswertung, die eine Atom-Größe liefert

Voreingestellter Bereich im Pulsfenster. Statt über eine Fläche zu mitteln, auf
der gar kein einzelnes Atom sitzt, wird mit der thermischen
Aufenthaltswahrscheinlichkeit gewichtet — dieselbe Konstruktion wie die
gewichtete Uniformity in `Weighted_Multitone_Lens_GUI.py`:

```
σ² = ħ/(2mω) · coth(ħω / 2k_BT) ,    W(r) = exp(−|r−r_atom|²/2σ²)
```

T und ν_r sind Eingabefelder (Vorgabe 17 µK und 60.4 kHz aus der Messung, Rb-85
→ σ = 108 nm). Ausgewertet wird auf einem eigenen feinen Gitter von ±4σ um den
Atomort; das globale Rechengitter ist mit 0.6 µm pro Zelle für ein Atom von
0.1 µm hoffnungslos zu grob.

**Das ändert die Aussage vollständig.** Für 13×14, Ω/2π = 1 MHz, T_p = 1 µs:

| Bereich | θ(0) | σ_θ/⟨θ⟩ | Abweichung bei ±1 µs | 1 %-Toleranz |
|---|---|---|---|---|
| **Atom (108 nm)** | **6.35 π** | **0.5 %** | **44 %** | **±125 ns** |
| Kreis r = 1 µm | 6.22 π | 4.2 % | 42 % | — |
| Plateau (90 µm) | 2.30 π | 74 % | 9.7 % | ±290 ns |
| Spotzentren | 2.36 π | 71 % | 12 % | — |

Zwei Dinge stehen hier auf dem Kopf:

* **Die Gleichmäßigkeit ist kein Problem.** Über 108 nm variiert die
  Momentanintensität um 0.5 %, das Zeitmittel um unter 0.005 %. Die 74 % des
  Plateaus sind eine Kamera-Größe — sie beschreiben, wie unterschiedlich zwei
  *weit auseinanderliegende* Punkte des Feldes bestrahlt werden, nicht was ein
  Atom erlebt.
* **Der Trigger ist viel kritischer als das Plateau-Mittel vermuten lässt.**
  Ein Punkt sieht die volle Schwebung; im Plateau-Mittel löschen sich die
  Beiträge teilweise aus. ±1 µs kosten 44 % statt 9.7 % der Fläche, und für 1 %
  Genauigkeit bleiben ±125 ns statt ±290 ns.

Weil die Streuung im Atom-Gewicht so klein ist, fallen dort auch ⟨sin²(θ/2)⟩ und
sin²(⟨θ⟩/2) zusammen (0.270 gegen 0.269) — der Unterschied, der im Plateau
Faktor zwei ausmacht, ist eine Eigenschaft der Mittelungsfläche, nicht der
Physik am Atom.

Die Karte oben in der Mitte zeigt im atomgewichteten Fall den lokalen Ausschnitt
in Nanometern mit den 1σ- und 2σ-Ringen des Atoms.

## Startwerte

Das GUI öffnet jetzt mit dem aktuellen Arbeitspunkt: **3×4 Töne, f1 = 75 mm,
f2 = 750 mm, width = 0.37 MHz, Waist 1.05 µm, r_x = 0.97, r_y = 1.16**, Airy,
795 nm, 100 MHz Offset, Ω/2π = 1 MHz. Daraus folgt f₀ = 61.67 kHz und
T₀ = 16.22 µs.

## Leistungskette: was wo gebraucht wird

Das Leistungsfenster rechnet die ganze Kette durch. Eingaben: Leistung vor dem
AOD, Beugungseffizienz je AOD (der Spot wird **zweimal** gebeugt, x und y
multiplizieren sich), Transmission der Optik. Der Bezugsbereich hat jetzt
ebenfalls die atomgewichtete Option — sonst kalibriert man f_rabi auf eine
Fläche und rechnet die Pulsfläche am Atom.

Für den Arbeitspunkt bei Ω/2π = 1 MHz am Atom, Δ = −8 GHz:

| | |
|---|---|
| Referenzintensität am Atom | 8.73 W/cm² |
| A_eff | 14.4 µm² |
| **Leistung im Profil** | **1.26 µW** |
| pro Spot | 0.105 µW (hellster 0.125) |
| pro RF-Ton x / y | 0.42 / 0.32 µW |
| 300 mW × 0.7 × 0.7 × 0.8 | 117.6 mW verfügbar |
| **Reserve** | **≈ 9·10⁴** |

Die Leistung ist also um Größenordnungen kein Problem — nötig wäre eine
Gesamteffizienz von 4·10⁻⁶. Das ist die eigentliche Nachricht: **der Aufbau ist
nicht leistungs-, sondern streubegrenzt.** Bei Δ = −8 GHz streut ein 1-µs-Puls
mit 0.81 %; weil Ω ∝ I/Δ und Γ ∝ I/Δ² geht, sinkt die Streuung bei fester
Rabi-Frequenz mit der Verstimmung. Das Fenster rechnet dazu Δ mit denselben
Koeffizienten wirklich durch (kein 1/Δ-Extrapolieren) und meldet, bis wohin die
Leistung reicht — hier bis über die abgesuchten 3 THz hinaus, wo die Näherung
selbst zu kippen beginnt, weil D2 nur 7.1 THz entfernt liegt und mit
umgekehrtem Vorzeichen beiträgt.

## Layout des Pulsfensters

Rechts groß die **ortsaufgelöste** Karte θ(r) — das Einzige im Fenster, das
keine Mittelung enthält. Links untereinander die drei Kurven, die alle
Ortsmittel sind: Fläche über eine Grundperiode, Delay-Scan, räumliche Streuung.
Der Delay-Plot trägt den Hinweis „intensitätsgewichtetes Ortsmittel" direkt
unter der Achse; die Anregungskurve ist aus dem Streuungsplot heraus (die Zahl
steht weiterhin in der Infozeile).
