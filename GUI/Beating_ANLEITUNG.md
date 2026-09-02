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

Für sich allein genommen lässt sich ein einzelnes Paar durch keine Phase
dämpfen — Δφ verschiebt nur, *wann* das Maximum liegt.

Entscheidend ist aber, dass sich **viele Paare dieselbe Differenzfrequenz
teilen** (bei 3×4 bis zu elf Paare pro Frequenz). Ihre Beiträge addieren sich
als Zeiger, und diese Summe kann durch geeignete Phasen teilweise ausgelöscht
werden. Schreibt man I(r,t) als Fourierreihe in f₀,

```
I(r,t) = Σ_d D_d(r) e^{i d ω₀ t}      D_d = Σ_{k_s−k_s' = d} g_s g_s' e^{i(φ_s−φ_s')}
⟨I⟩(r) = D_0(r)                        Var_t(I) = Σ_{d≠0} |D_d|²
```

dann ist die zeitliche Varianz **sehr wohl phasenabhängig**. (Diese Form ist
exakt und braucht keine Zeitschleife — das GUI rechnet σ_t/⟨I⟩ damit und kann
deshalb hunderte Startpunkte optimieren.)

Gemessen bei 3×4, waist 1.05 µm, width 0.35 MHz, als RMS von σ_t(I)/⟨I⟩ im
Plateau:

| Phasensatz | σ_t/⟨I⟩ im Plateau | Modulationstiefe | Spitze / max⟨I⟩ |
|---|---|---|---|
| alle 0 | 136 % | 99.7 % | 7.49 × |
| Schroeder | 90 % | 98.3 % | 3.93 × |
| Newman | 90 % | 98.3 % | 3.91 × |
| auf minimale **Spitze** optimiert | 72 % | 96.9 % | **2.47 ×** |
| auf minimale **Variation** optimiert | **70 %** | 96.4 % | 2.63 × |
| frei je Spot (unphysikalisch) | 55 % | — | 2.26 × |

**Faktor 1.9 ist drin, mehr nicht.** Auf null bringen lässt es sich prinzipiell
nicht: Differenzfrequenzen, die nur von einem einzigen Paar erzeugt werden (bei
3×4 etwa d = 12, die beiden diagonalen Eckspots), haben keinen Partner zum
Auslöschen. Selbst mit völlig freien Spotphasen — die sich mit zwei AODs gar
nicht ansteuern lassen — kommt man nur auf 55 %.

Bemerkenswert: die Optimierung auf **Plateau**, auf die **Spot-Zentren** und auf
den **2-µm-Kreis** führt auf praktisch denselben Wert (70.2 / 70.5 / 70.2 %).
Es gibt hier also keinen Zielkonflikt, ein Phasensatz bedient alle drei.

Ebenso bemerkenswert: die Optimierung auf minimale **Spitze** landet fast am
selben Punkt (72 % statt 70 %). Wer den AOD schonen will, bekommt die ruhigere
Intensität fast geschenkt.

Die **Modulationstiefe** (I_max−I_min)/(I_max+I_min) bleibt dagegen bei 96 bis
100 %, egal welche Phasen. Sie wird vom Minimum bestimmt, und das geht in einem
so stark überlappenden Muster fast überall durch null. σ_t/⟨I⟩ ist deshalb die
aussagekräftigere Größe für „wie unruhig ist das Licht".

**Was physikalisch einstellbar ist:** nur die N_x + N_y Phasen der RF-Töne.
Ein Spot (n,m) trägt φ_x(n) + φ_y(m) — bei 3×4 kommen die zwölf Spotphasen also
aus sieben Freiheitsgraden, von denen zwei global sind, bleiben fünf. Der
Eintrag *frei je Spot* im GUI umgeht das bewusst und dient nur als Gegenprobe.

Die Knöpfe: **Ruhefenster optimieren** (siehe unten — meist der praktisch
relevante), **Zeitliche Variation minimieren** (Zielgebiet wählbar: Plateau,
Spot-Zentren oder Kreis mit einstellbarem Radius) und **Spitze minimieren**.
Beide schreiben das Ergebnis in die Eingabefelder, von wo aus sich weiter
probieren lässt.

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
