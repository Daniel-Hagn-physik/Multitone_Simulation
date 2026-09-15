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

## Dateien

Stand 2026-09-11. Seit dem Umbau liegt die gesamte Physik in einer eigenen
Datei; das GUI selbst enthält nur noch Eingaben, Ablauf und Zeichnen.

| Datei | Inhalt |
|---|---|
| `Beating_Multitone_GUI.py` | **starten.** Fenster: Eingaben, `recompute()`, Zeichnen, Speichern |
| `kern/beating_physik.py` | **die ganze Physik und Numerik** — Frequenzen, Geometrie, Felder, Beat-Ordnungen, exakte Zeitstatistik, Pulsfläche, Atomgewichtung. Kein Qt, nur numpy/scipy. Oben in der Datei stehen die Formeln, was das Modell *nicht* enthält, und ein Inhaltsverzeichnis der Abschnitte 0–11. |
| `kern/rb85_raman.py` | Raman-Koeffizienten für Rb-85 (ARC) |
| `kern/beating_profil.py` | Brücke zur Rabi-Rechnung (`Rabi_Rb85_GUI.py`); importiert `beating_physik.py`, nicht mehr das GUI |
| `one_lens_design.py` | Fenster *Design for a target beating period …* |
| `camera_series.py` | Fenster *Camera frame series …* |
| `pulse_timing.py` | Fenster *Pulse area / trigger jitter …* |
| `power_budget.py` | Fenster *Power / intensity …* |

Die vier Fenster-Dateien müssen neben dem GUI liegen; die Physik holen sie
sich selbst aus `kern/`. Alle Funktionsnamen, die in dieser Anleitung
auftauchen (`time_stats_exact()`, `crest_factor()`, `camera_frames_exact()`,
`profile_total_power()`, …), stehen in `kern/beating_physik.py`.

## Startwerte

Das GUI öffnet immer mit diesem Arbeitspunkt:

| | |
|---|---|
| Töne | 3 × 4 |
| Profil | **Airy**, Faktor **1.4830 (fest)** → erster Nullring 1.542 µm |
| waist | **1.04 µm** nach den Linsen (entspricht 1.287 mm davor) |
| width | **0.37 MHz**, x und y gekoppelt |
| Amplituden | r_x = **0.97**, r_y = **1.16** → amp_x = [0.97, 1, 0.97], amp_y = [1.16, 1, 1, 1.16] |
| Optik | f1 = **75 mm**, f2 = 750 mm, fLO = 52.88 mm, λ = 795 nm, Offset 100 MHz |
| Puls | f_Rabi = 1 MHz, Ω ~ I, η = 0, t₀ = 0 |
| Zeitachse | 3 Perioden, 60 Frames/Periode, Gitter 200² |
| Pulsfenster | Δ = **+50 GHz** (blau), 10 µW im Profil, T_p = 1 µs, atomgewichtet (17 µK, ν_r = 60.4 kHz) |
| Leistungsfenster | Δ = **+50 GHz** |

Daraus folgt, bei allen Tonphasen 0:

| | |
|---|---|
| f₀ = width/6 | 61.67 kHz → T₀ = 16.22 µs |
| Spotabstand | 1.169 µm in x, 0.779 µm in y |
| Frequenzentartung | eine Eck-Paarung, statischer Anteil 7.16 % |
| σ_t/⟨I⟩ im Plateau | 95.7 % (Schroeder: 91.4 %) |
| Spitze der Rephasierung | 4.25 × max⟨I⟩ |
| Crest-Faktor RF x / y | 2.45 / 2.83 (Schroeder: 2.16 / 2.00), RF-Spannungen √r |
| **nächste Beat-Linie zu ν_r = 60.4 kHz** | **1.3 kHz** (zu 2ν_r: 2.5 kHz) — das GUI meldet das rot, siehe *Die daraus folgende Entwurfsregel* |

**Der Airy-Faktor ist nicht mehr einstellbar.** Er sollte früher aus
`airy_scale.py` kommen; diese Datei liegt aber in `Flattop GUI/`, der Import
schlug stillschweigend fehl, und benutzt wurde ohnehin immer der Ersatzwert
1.4830. Jetzt steht er als `AIRY_FACTOR` in `kern/beating_physik.py`, und das
GUI zeigt ihn nur an.

## Bedienung

Alles wird eingetippt; Slider gibt es nur für die Zeitachse und die
Abspielgeschwindigkeit. Gerechnet wird mit **Recompute** oder, mit dem Haken
**Recompute automatically**, bei jeder Änderung. Ein Klick in die 2D-Karte
setzt das Fadenkreuz; Orts-Zeit-Karte, Schnitt und I(t) beziehen sich danach
auf diesen Punkt.

| Gruppe | Feld | Bedeutung |
|---|---|---|
| Tones | N_x, N_y | Tonzahl je Achse |
| Beam profile | Profile | Gauss oder Airy. Beim Airy tragen die Ringe ein **negatives** Vorzeichen — für die kohärente Summe wesentlich. |
| | Airy factor | fest 1.4830, `first_zero_radius = Faktor · waist` (nur Anzeige) |
| Beam / optics | Use one lens, f (single lens), *Design for a target beating period …* | Ein-Linsen-Aufbau, siehe unten |
| | Mode, waist, waist_in | waist nach der Linse (µm) oder davor (mm); die jeweils andere Größe wird nachgezogen |
| | width x / width y, `width_y = width_x` | Frequenzspanne je Achse; der Haken koppelt beide |
| | Wavelength, Offset f0 | gehen in die Geometrie ein, **nicht** in die Beat-Frequenzen |
| | f1, f2 | Teleskop (ignoriert bei *Use one lens*) |
| Amplitudes | r_x, r_y | Außen/Innen-Verhältnis `amp_x = [r_x, 1, …, 1, r_x]`. Das sind **Intensitäts**-Gewichte, das Feld trägt die Wurzel. |
| Pulsed operation | f_Rabi, Pulse start t_0, Coupling, light shift eta | siehe *Gepulster Betrieb* |
| | *Move t_0 to a flat point …* | siehe *Flacher Punkt statt bestem Punkt*. Der Optimierer-Knopf sitzt im Kasten *Tone phases* |
| Tone phases | 0 / Schroeder / Kitayoshi / randomise | Presets für alle Tonphasen |
| | max. crest factor | die einzige Stellschraube: wie hart die RF-Kette gefahren werden darf. Siehe *Phasen für das Atom optimieren* |
| | *Optimise phases and t_0 for even illumination of the atom* | startet die Suche. Steht direkt bei den Einstellungen, die er liest |
| | evaluation circle r | Auswertekreis für Plots, U(t) und die Nebenfenster — **nicht** für die Phasensuche |
| | Pulse length T_p, *= pi pulse* | Pulslänge, mit der der Optimierer rechnet (Standard 1 µs). Unabhängig von f_Rabi — das Experiment legt den Puls fest, nicht die π-Bedingung |
| | atom T, trap frequency nu_r | Atomtemperatur und radiale Fallenfrequenz → σ (angezeigt). Nur beim atomgewichteten Ziel |
| | Target region + Radius | Zielgebiet nur des **Regions**-Ziels: Plateau, Spot centres oder Circle around the centre (Standard, 2 µm) |
| | keep degenerate pairs in quadrature | hält frequenzentartete Paare bei 90° (Standard: an) |
| | φ_x, φ_y je Ton (Grad) | eintippbar |
| Time axis | Periods, Frames/period, Grid resolution | Fensterlänge, Abtastung, Gitter. Frames/period **muss > 2·(höchste Beat-Frequenz / f₀)** sein, sonst Aliasing — das GUI warnt und nennt die nötige Zahl. |
| | Camera exposure | Boxcar-Belichtung, siehe *Belichtungszeit der Kamera* |
| | t, Play, fps | Zeitpunkt bzw. Animation |
| | Trap frequency nu_r | nur für die Prüfung, ob eine Beat-Linie auf ν_r oder 2ν_r fällt |
| Actions | Recompute automatically, Recompute | |
| | Panel top right | Auswahl des Plots oben rechts, siehe *Die Plots* |
| | Fast drawing during playback, pulse / spectrum also at many spots, record U(t) live | Zeichen- und Rechenoptionen |
| | Colour scale | siehe *Farbskala* |
| | *Power / intensity …*, *Pulse area / trigger jitter …*, *Camera frame series …* | Nebenfenster, siehe unten |
| | Save view as PDF | PDF nach `GUI/Bilder`, Parameter als `.txt` daneben |

## Die Plots

Links groß **I(x,y,t)**, die Momentaufnahme; Kreise markieren
frequenzentartete Spots. Rechts untereinander:

1. **Panel oben rechts**, wählbar unter *Panel top right*:

   | Auswahl | zeigt |
   |---|---|
   | Space-time map I(x, t) | Orts-Zeit-Karte entlang des x-Schnitts — die Schwebungsstreifen ohne Animation lesbar |
   | Enhancement n_eff = I_max / ⟨I⟩ | Zahl der am Ort wirksam überlappenden Töne |
   | Modulation depth | (I_max − I_min)/(I_max + I_min) |
   | temporal variation σ_t/⟨I⟩ | exakt aus den Beat-Ordnungen, ohne Zeitraster |
   | Spectrum of the beating | σ_d/⟨I⟩ je Beat-Ordnung, mit ν_r und 2ν_r |
   | Uniformity U(t) of the three regions | siehe *Uniformity im Zeitverlauf* |
   | Pulse area: U over the pulse start | U(θ) über t₀ |
   | Pulse area A(t_0) and flat points | mittlere Fläche über t₀, flache Stellen markiert |
   | Rabi oscillation | Anregung über der Pulslänge, siehe *Was man am Ende misst* |

2. **x cut** bei t, mit Min/Max-Hüllkurve über die Zeit und dem Zeitmittel
   als gestrichelter Referenz — das Zeitmittel ist das, was die inkohärenten
   GUIs zeigen.
3. **I(t) am Fadenkreuz** über dieselbe Zeitspanne, mit Modulationstiefe.

## Farbskala

Stehen alle Tonphasen auf 0, rephasieren die Töne einmal pro Grundperiode zu
einem kurzen Puls — am Startpunkt mit **4.25× dem Maximum des Zeitmittels**
(bei r_x = r_y = 1 und width 0.35 MHz waren es 7.5×). Eine feste Skala auf
dieses Maximum lässt alle übrigen Frames fast schwarz, deshalb ist das
99.5-Perzentil voreingestellt. Der Spitzenwert steht im Titel. Weitere
Stellungen: Maximum über die Zeit, Maximum des Zeitmittels, je Frame.

## Zum Vergleich mit den anderen GUIs

Die Statuszeile prüft bei jedem Lauf, ob das Zeitmittel des Würfels mit der
inkohärenten Summe `Σ_s a_s |u_s|²` übereinstimmt — also mit genau dem, was
`Multitone_Lens_GUI.py` rechnet. Ohne entartete Spots stimmt es auf ~1e-6
genau. Weicht es ab, steht der Betrag dort und kommt von der statischen
Interferenz, nicht von der Numerik.

> **Zu den Zahlen in den folgenden Abschnitten.** Sie sind Befunde aus dem
> jeweiligen Stand der Untersuchung und nicht auf die aktuellen Startwerte
> umgerechnet. Meist gilt 3×4 Airy mit waist 1.05–1.1 µm und width
> 0.35–0.45 MHz (T₀ = 17.14 bzw. 13.33 µs); wo ein anderer Satz gerechnet
> wurde (13×14, Ein-Linsen-Aufbau), steht es dabei. Die Aussagen selbst gelten
> unverändert.

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
wandert dann. Mit dem Preset *randomise* in der Gruppe *Tone phases* lässt sich dieser Fall durchspielen.

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

**Geblieben** sind die Presets 0 / Schroeder / Kitayoshi / randomise, der
Quadratur-Haken und der Puls-Optimierer. Die Abschnitte *Ruhefenster* und
*Phasen auf Uniformity optimieren* weiter unten beschreiben die entfernten
Optimierer; ihre Zahlen bleiben als Befund stehen.

Alle Zahlen dieser Tabelle gelten für das **Plateau**, also für eine Kamera.
Was ein einzelnes **Atom** sieht, ist eine andere Frage und steht in
*Phasen für das Atom optimieren* — dort ist die Zielgröße nicht σ_t/⟨I⟩ über
eine Fläche, sondern der Hub der **Pulsfläche am Atomort** über den
Trigger-Zeitpunkt.

## Phasen für das Atom optimieren

Alle bisherigen Zielgrößen mitteln über eine **Fläche** — ein Plateau, einen
Kreis, die Spotzentren. Auf keiner davon sitzt ein einzelnes Atom. Ein Atom
sitzt an **einem** Ort und ist dort um σ_thermal unscharf (108 nm bei 17 µK und
ν_r = 60.4 kHz); was es sieht, ist die mit seiner Aufenthaltswahrscheinlichkeit
W(r) gewichtete Intensität — dieselbe Gewichtung, die
`Weighted_Multitone_Lens_GUI.py` und das Pulsfenster benutzen.

Und weil das Atom an einem Ort sitzt, ist auch die *Zeit*frage eine andere:
bei perfektem Trigger friert jeder Schuss auf denselben Wert ein. Was von
Schuss zu Schuss variiert, ist **wann** der Puls im Schwebungszyklus landet.
Genau das ist die Zielgröße.

### Die Struktur wird dadurch einfacher, nicht komplizierter

Das Ortsmittel vertauscht mit allem Weiteren, darf also ganz nach vorne:

```
I_W(t) = Σ_r W(r) I(r,t) / Σ_r W(r) = Σ_d C_d e^{2πi d f₀ t}
C_d    = Σ_{k_s − k_s' = d}  ⟨g_s g_s'⟩_W · e^{i(φ_s − φ_s')}
```

Aus einem Feld über tausenden Pixeln wird **ein** komplexer Vektor mit so
vielen Einträgen, wie es Beat-Ordnungen gibt — bei 3×4 zwölf. Die
Paarprodukte ⟨g_s g_s'⟩_W hängen nicht von den Phasen ab und werden einmal
vorberechnet. Eine Auswertung kostet danach rund 13 µs statt einer
Matrixrechnung pro Pixel; deshalb sind 300 Startpunkte praktisch gratis, und
die braucht man auch — die Zielfunktion steckt voller lokaler Minima.

Ein Rechteckpuls der Länge T_p ab t₀ ist derselbe Boxcar wie eine Belichtung,
also ein sinc auf jeder Ordnung:

```
A(t₀) = (1/T_p) ∫_{t₀}^{t₀+T_p} I_W(t) dt
      = Σ_d C_d · sinc(d f₀ T_p) · e^{2πi d f₀ (t₀ + T_p/2)}
```

Bei T_p = 1 µs und f₀ = 61.67 kHz ist sinc(0.0617) = 0.994 auf der Grundordnung
und immer noch 0.30 bei d = 12: **der Puls mittelt die Schwebung nicht weg, er
tastet sie ab.** Erst bei T_p = T₀ wird jedes sinc(d) = 0 und der Hub exakt
null — der Puls deckt dann genau eine volle Periode ab. (Das ist zugleich der
Testfall, an dem die Implementierung geprüft ist: R kommt dort auf 5·10⁻¹⁷
heraus.)

Wegen Parseval ist der Effektivwert über eine ganze Grundperiode geschlossen
angebbar, ganz ohne Zeitraster:

```
R = sqrt( 2 · Σ_{d>0} |C_d · sinc(d f₀ T_p)|² ) / C_0
```

**R ist die Zielgröße**: die relative Schwankung der Pulsfläche am Atomort über
den Trigger-Zeitpunkt. R = 0 hieße, der Puls liefert bei *jedem* Trigger
dieselbe Rabi-Fläche.

### Zwei Zielgrößen, die gegeneinander ziehen

R ist die richtige Größe, **wenn der Trigger unsicher ist**. Ist der Puls
sauber getriggert, ist sie die falsche: dann friert jeder Schuss auf denselben
Wert ein, und was zählt, ist allein, wie viel Rabi-Fläche das Atom bei diesem
einen t₀ bekommt, also

```
peak_ratio = max_t A(t) / ⟨A⟩
```

Weil ⟨A⟩ praktisch phasenunabhängig ist (1.9128 gegen 1.9119 zwischen dem
schlechtesten und dem besten Satz — ⟨A⟩ hängt nur über die frequenzentarteten
Paare überhaupt von den Phasen ab), ist dieses Verhältnis zugleich der
**absolute** Vergleich zwischen Phasensätzen: ein Faktor zwei hier ist ein
Faktor zwei in der nötigen Laserleistung.

Arbeitspunkt, T_p = 1 µs, σ = 108 nm:

| Phasen | R | A(t₀)/⟨A⟩ | Crest x / y | 1 %-Fenster |
|---|---|---|---|---|
| alle 0 | 115 % | 4.25 | 2.45 / 2.83 | 277 ns |
| Schroeder | 72 % | 2.70 | 2.16 / 2.00 | 260 ns |
| Kitayoshi | 72 % | 2.47 | 2.16 / 2.00 | 262 ns |
| auf **R** optimiert | **41 %** | **1.47** | 1.82 / 2.21 | 317 ns |
| auf **A(t₀)** optimiert, C = 2.0 | 71 % | **2.89** | 1.99 / 1.99 | ≈ 260 ns |
| freie Spotphasen (nicht fahrbar) | 24.9 % | — | — | — |

**Die beiden ziehen gegeneinander.** R zu minimieren senkt die Helligkeit am
Trigger von 4.25 auf 1.47 — das Atom bekommt bei bestem Trigger dann weniger
als die Hälfte dessen, was schon das Schroeder-Preset liefert. Deshalb ist das
**Standardziel die Helligkeit**, und R nur die Option für den ungetriggerten
Fall.

Die Quadratur kostet beim R-Ziel 1 Prozentpunkt (41.1 statt 40.1 %) — sie kann
anbleiben.

In beiden Fällen wird t₀ danach auf das **Maximum** von A(t₀) gelegt. Dort ist
dA/dt₀ = 0, Jitter geht also nur quadratisch ein.

### Abhängigkeit von der Pulslänge

| T_p | 0.2 µs | 0.5 µs | 1 µs | 2 µs | 5 µs | 16.22 µs = T₀ |
|---|---|---|---|---|---|---|
| R optimiert | 44 % | 43 % | 41 % | 33 % | 15 % | 0 % |
| Crest x / y | 1.82/2.20 | 1.82/2.20 | 1.82/2.21 | 1.82/2.21 | 1.82/2.22 | 1.84/2.18 |

Bemerkenswert: der Crestfaktor am Optimum ist über den ganzen Bereich
derselbe. Der Optimierer landet unabhängig von T_p in derselben
RF-freundlichen Ecke.

### Was der Crestfaktor ist, und warum er in der Zielfunktion steht

Der AWG erzeugt je Achse die Summe der Töne,
s(t) = Σ_n a_n cos(2π f_n t + φ_n). Der **Crestfaktor** ist Spitzenwert durch
Effektivwert dieses Signals. Er misst, wie ungleichmäßig sich die Leistung über
die Zeit verteilt: bei einem einzigen Ton ist er √2 = 1.414, bei N gleichen
Tönen *mit gleicher Phase* rephasieren alle einmal pro Hüllkurvenperiode und er
erreicht sein Maximum

```
crest = √(2N)     →  2.449 bei 3 Tönen, 2.828 bei 4 Tönen
```

Genau diese Werte stehen im GUI bei φ = 0 (2.45 / 2.83). Deshalb gibt es
Schroeder- und Kitayoshi-Phasen überhaupt: sie drücken den Crest, ohne die
Amplituden anzutasten.

**Warum das Licht kostet.** Ein Verstärker, der durch seine *Spitzenspannung*
begrenzt ist — er geht in Kompression, nicht in Erwärmung — kann bei gegebenem
V_max nur eine mittlere Leistung

```
P_mittel ≤ V_max² / (R · crest²)
```

liefern. Die gebeugte optische Leistung folgt der mittleren RF-Leistung, und
ein Spot wird je Achse einmal gebeugt. Die Spotintensität trägt also
1/(C_x²·C_y²). (Fährt man trotzdem in die Kompression, kommen
Intermodulationsprodukte dazu: neue Frequenzen an Summen und Differenzen der
Töne, also Geisterspots, und die Amplitudenverhältnisse verschieben sich — das
mühsam eingestellte Profil stimmt dann nicht mehr.)

**Die Konsequenz.** Der Rephasing-Peak, der die Pulsfläche hell macht, ist
*derselbe Vorgang*, der den Crest hochtreibt — die Töne addieren sich in Phase,
optisch wie elektrisch. Man kann nicht das eine haben und das andere nicht.
Unter Spitzenbegrenzung kostet ein höherer Crest aber mehr mittlere Leistung,
als der hellere Peak einbringt. Die richtige Zielgröße ist deshalb

```
F = max_t A(t) / (crest_x² · crest_y²)
```

die Rabi-Fläche **pro Volt Verstärker-Reserve**. Der Crestfaktor ist damit
keine Nebenbedingung mehr, sondern Teil der Antwort — es gibt nichts
einzustellen.

| Phasen | F | A(t₀)/⟨A⟩ | Crest x / y | R |
|---|---|---|---|---|
| alle 0 | 0.089 | 4.25 | 2.45 / 2.83 | 115 % |
| Kitayoshi | 0.133 | 2.47 | 2.16 / 2.00 | 72 % |
| Schroeder | 0.145 | 2.70 | 2.16 / 2.00 | 72 % |
| auf R optimiert | 0.091 | 1.47 | 1.82 / 2.21 | 41 % |
| **auf F optimiert** | **0.239** | 2.47 | **1.82 / 1.77** | 62 % |

**2.7-fache Pulsfläche gegenüber φ = 0, 1.6-fache gegenüber Schroeder**, bei
gleichem Verstärker. Und φ = 0 ist unter dieser Normierung nicht nur ungünstig
für die RF-Kette, sondern schlicht der schlechteste Punkt. Bemerkenswert auch:
das R-Optimum liegt mit 0.091 kaum über φ = 0 — es erkauft Ruhe mit Helligkeit,
und zwar teuer.

> **Wäre die Kette durch die mittlere Leistung begrenzt** (thermisch limitierter
> AOD), sähe es anders aus: der Crest wäre dann nur eine Verzerrungsgrenze, die
> richtige Zielgröße wäre max_t A(t) allein, und man bräuchte eine
> Crest-*Schranke*, weil die Antwort sonst immer φ = 0 lautet. Für diesen
> Aufbau gilt das nicht.

### Was *nicht* funktioniert hat

Die naheliegende Formulierung für einen getriggerten Puls — „minimiere die
Empfindlichkeit gegen Trigger-Jitter", also den Effektivwert von A über ein
Fenster ±Δ um t₀ — ist **degeneriert** und wurde verworfen. Bei realistischem
Jitter ist sie ohnehin bedeutungslos (±50 ns ergibt schon für φ = 0 nur
0.06 % Flächenfehler), und der Optimierer nutzt die verbleibende Freiheit, um
einen *flachen, aber dunklen* Wendepunkt zu suchen: A(t₀) = 0.78·⟨A⟩ bei
gleichzeitig **schlechterer** Gesamtschwebung (R = 0.87 gegen 0.72 bei
Schroeder). Das Atom bekäme weniger Licht und die Schwebung wäre größer.

Ebenfalls verworfen: R durch A(t₀) zu teilen, um Schwebung und Helligkeit in
einer Zahl zu fassen. Das unterscheidet nicht — φ = 0 kommt auf 0.270,
Schroeder auf 0.268, das Optimum auf 0.280. Ein schärferer Peak kauft die
Helligkeit exakt proportional zur Schwebung ein.

### Der Ortsterm

Die räumliche Streuung über die Atomwolke, σ_θ/⟨θ⟩_W bei σ = 108 nm, ist
**nicht** Teil der Zielfunktion. Sie ist klein, aber nicht null und verhält
sich nicht monoton mit R: 0.7 % bei φ = 0 (im Rephasing-Maximum ist das
Profil die glatte kohärente Summe), 18.8 % bei Schroeder, 6.0 % am Optimum.
Wer sie mitnehmen will, muss sie explizit dazunehmen — sie wandert dann in
die Zielfunktion, nicht in die Anzeige.

### Bedienung

Alles zur Phasenoptimierung sitzt jetzt in **einem** Kasten, *Tone phases*:
Presets, Zielauswahl, eine Klartextzeile, die sagt was das gewählte Ziel tut,
die Eingaben (T_p, atom T, ν_r bzw. Zielregion), der Quadratur-Haken, der
Optimierer-Knopf und darunter die Phasenfelder mit der Ergebniszeile. Der Knopf
beschriftet sich nach dem Ziel um; die Felder des nicht gewählten Ziels werden
ausgegraut. Rund 20–40 Sekunden.

* **pulse area at the atom, per amplifier headroom** — Standard, 250
  Startpunkte. Ergebniszeile: F mit den Faktoren gegenüber φ = 0 und
  Schroeder, dazu A(t₀)/⟨A⟩, t₀, die beiden Crestfaktoren und R.
* **beating at the atom, minimal swing (untriggered)** — 300 Startpunkte.
  Ergebniszeile: R mit dem Vergleichswert für alle Phasen 0, t₀, A(t₀)/⟨A⟩
  und die Crestfaktoren.
* **pulse area, spatial spread over a region** — das alte Ziel, unverändert:
  std/mean von θ(r) über eine harte Maske zu *einem* Zeitpunkt. Es beantwortet
  eine Kamera-Frage und bleibt für den Vergleich mit den inkohärenten GUIs
  stehen.

Die Eingabefelder des jeweils nicht gewählten Ziels werden ausgegraut.

Code: `AtomBeating` (mit `ripple()` und `peak_ratio()`) und
`atom_beating_at_centre` in Abschnitt 12 von `kern/beating_physik.py`,
`CrestBasis` in Abschnitt 5.

## Wo die Unruhe sitzt: das Spektrum

I(r,t) ist eine Fourierreihe in f₀, und `Var_t(I) = Σ_{d≠0} |D_d|²` zerfällt in
Beiträge einzelner Schwebungsfrequenzen d·f₀. Das Panel **Spectrum of the
beating** zeigt sie, mit ν_r und 2ν_r eingezeichnet.

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
Optimierung ließ sich deshalb auf ein Frequenzfenster richten (*Ziel: nur nahe
ν_r und 2·ν_r*; diese Zielwahl ist inzwischen entfernt, der Befund gilt):

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

> **Entfernt.** Den Knopf *Ruhefenster optimieren* gibt es nicht mehr (siehe
> *Tonphasen*). Der Abschnitt bleibt als Befund stehen.

Der Knopf **Ruhefenster optimieren** suchte die Tonphasen, bei denen das Profil
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

Das Panel **Uniformity U(t) of the three regions** zeigt U = std/mean — dieselbe
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

Mit dem Haken **record U(t) live** (Gruppe *Actions*) wächst die Kurve
während der Animation mit: durchgezogen bis zum aktuellen Zeitpunkt, der
restliche Verlauf blass dahinter, ein Punkt auf dem Momentanwert. Im Titel
stehen die drei Momentanwerte als Zahlen — das ist die Anzeige, die man beim
Zuschauen tatsächlich abliest.

Damit das zügig läuft, tauscht das GUI beim Weiterschalten nur die Daten der
vorhandenen Zeichenobjekte aus, statt alle vier Achsen neu aufzubauen. Der
schnelle Pfad liefert pixelgleiche Ausgabe wie der vollständige Aufbau (auf
allen sechs Panels per Hash geprüft) und greift nur, solange Panel, Datenform
und Fadenkreuz unverändert sind — sonst wird automatisch vollständig gezeichnet.

Falls die Anzeige sich merkwürdig verhält, schaltet der Haken **Fast drawing
during playback** in der Gruppe *Actions* auf den vollständigen Neuaufbau
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

> **Entfernt.** Diese Zielwahl gibt es nicht mehr; übrig ist nur der
> Puls-Optimierer. Die Tabelle bleibt als Befund stehen.

Die Zielgröße im Optimierer war umschaltbar. Die drei Ziele führen auf
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
Uniformity an den Spot-Zentren von 23.7 % auf 24.3 %. Der Haken **keep
degenerate pairs in quadrature** ist deshalb voreingestellt und wirkt im
Puls-Optimierer.

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
| 0.35 MHz | 58.3 kHz | **2.1 kHz** | **4.1 kHz** | kritisch |
| **0.37 MHz (Startwert)** | **61.67 kHz** | **1.3 kHz** | **2.5 kHz** | **kritisch** |
| 0.45 MHz | 75.0 kHz | 14.6 kHz | 29.2 kHz | unkritisch |

Der Wechsel auf width = 0.45 MHz hatte also nebenbei genau das Richtige getan.
**Die aktuellen Startwerte (0.37 MHz) liegen wieder fast genau auf ν_r** — das
GUI meldet es rot in der Gruppe *Time axis*. Nach dem Kriterium des GUI
(nächste Linie näher als 6 % an ν_r bzw. 2ν_r, ν_r = 60.4 kHz) sind bei 3×4
die widths 0.23–0.25, 0.35–0.38 und 0.69–0.76 MHz kritisch; 0.39–0.68 MHz ist
frei.

> Gilt für Beleuchtung über **viele Fallenperioden** (Dauerlicht, lange
> Belichtung, dichte Pulsfolgen). Ein einzelner Puls, der kürzer ist als 1/ν_r
> (π-Puls 0.5 µs gegen 16.6 µs Fallenperiode), ist spektral ~1/T_p breit; dort
> spielt die Lage der Linien keine Rolle, sondern nur der Stoß des Pulses.

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
Eine feste Verstimmung ist allerdings nicht mehr ∝ Ω; mit Beating schwankt der
Rest zeitlich, und die geschlossene Form gilt dann nicht mehr exakt.

**Das gilt nur für Ω ~ I.** Bei Ω ~ √I folgt die Verschiebung des Multiton-Asts
weiter I, die Rabi-Frequenz aber √I — das Verhältnis ist nicht konstant, die
Formel oben ist dort falsch. Seit 2026-09-11 wird in diesem Fall numerisch
propagiert (`sqrt_law_excitation()`); η bedeutet dort den Anteil von δ/Ω, den
der Multiton-Ast bei der Kalibrierintensität verursacht (bei gleich starken
Ästen die Hälfte des Gesamt-η, die andere Hälfte vom sauberen Ast gilt als
kompensiert). Größenordnung des Gesamt-η: ω_HF/Δ, also ≈ 0.06 bei +50 GHz und
≈ 0.44 bei −8 GHz.

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

Die Gruppe **Pulsed operation** hat vier Eingaben und zwei Knöpfe:

| Feld | Bedeutung |
|---|---|
| f_Rabi | Ω/2π; daraus folgt T_π = 1/(2 f_Rabi) |
| Pulse start t_0 | Lage des Pulses im Schwebungszyklus |
| Coupling | Ω ~ I (Zweiphotonen-Raman, beide Äste aus diesem Profil) oder Ω ~ √I (dieses Profil ist nur ein Ast) |
| light shift eta | δ/Ω, deckelt den Kontrast, siehe *Welche Physik im Anregungsmodell steckt* |
| *Move t_0 to a flat point* | setzt t₀ auf eine jitter-tolerante Stelle |
| *Optimise phases and t_0* | minimiert U(θ) im Zielgebiet, etwa 30 s |

Das Panel **Pulse area: U over the pulse start** zeigt U(θ) als Funktion von t₀
über einen ganzen Schwebungszyklus, mit dem besten t₀ markiert und der
Zeitmittel-Referenz als gepunkteter Linie.

Die Kopplung macht einen spürbaren Unterschied: mit Ω ~ √I liegt das beste t₀
bei 11.1 % statt 20.3 %, weil die Wurzel den Dynamikbereich staucht.

**Zielgebiet ist der 2-µm-Kreis** um die Mustermitte — Voreinstellung
*Circle around the centre* im Feld *Target region* (Gruppe *Tone phases*),
Radius im Feld daneben.

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
Ziel-`pitch/waist`. Doppelklick oder *Apply the selected set* setzt N_x,
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

`time_stats_exact()` (in `kern/beating_physik.py`) läuft nicht mehr über alle Spotpaare (O(S²)), sondern
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
*Redraw*. Die Kennzahlzeile darunter nennt den Hub pro Pixel im Plateau
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
muss exakt √(2N) geben. Die RF-Amplituden gehen mit ein, und zwar als
**Spannungen** √r_x, √r_y (r ist ein Leistungsverhältnis; bis 2026-09-11 stand
dort fälschlich r).

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
zusätzliche räumliche Struktur hinzu. (Für Ω ~ I; bei Ω ~ √I wird propagiert,
siehe *Welche Physik im Anregungsmodell steckt*.)

## Leistung und Intensität

Knopf **Power / intensity …** in *Actions*. Die Simulation kennt I nur bis auf
eine Konstante; eine physikalische Vorgabe legt sie fest, und der Dialog rechnet
in beide Richtungen — Rabi-Frequenz → Leistung oder Leistung → Rabi-Frequenz.

Die Kette:

1. **Atomphysik** aus `kern/rb85_raman.py`: adiabatische Elimination der
   5P-Zustände mit voller, vorzeichenrichtiger Summe über D1 *und* D2 samt
   Hyperfeinstruktur. Ergebnis `Ω = C_rabi·I`, `δ_LS = C_shift·I`,
   `Γ = C_scatter·I`, alle linear in I. Für Δ = −8 GHz (damalige Voreinstellung,
   jetzt +50 GHz: C_rabi = 11.66, η = −0.061), σ⁺/σ⁺ kopropagierend,
   β = 0.5, m_F = 0: **C_rabi = 71.98 rad/s pro W/m²**, η = −0.442.
2. **Bezug**: Ω/2π gilt für das *Zeitmittel* der Intensität im gewählten Bereich
   — dieselbe Konvention wie im Pulsfenster.
3. **Fläche**: `P = I_ref·A_eff` mit `A_eff = ∫I dA / ⟨I⟩_Bereich`. Das
   Flächenintegral kommt **analytisch** aus `profile_total_power()`, nicht aus
   dem Rechengitter — beim Airy-Profil fehlen dort je nach Rand bis ~5 % der
   Leistung in den abgeschnittenen Ringen. Jeder Spot trägt sein
   Intensitätsgewicht a (Summe über a, **nicht a²**), dazu kommt der statische
   Kreuzterm frequenzentarteter Spots (3×4, Airy, Phasen 0: +1.1 %).
4. **Aufteilung**: a_x, a_y sind RF-**Leistungs**verhältnisse (Beugungseffizienz
   ∝ RF-Leistung, gebeugtes Feld ∝ RF-Spannung). Spot (n,m) ∝ a_x(n)·a_y(m);
   RF-Ton n der x-Achse ∝ a_x(n)/Σa_x, weil er alle Spots seiner Spalte speist.
   Die RF-**Spannung** eines Tons ist √a_x(n) — das braucht ein AWG mit
   Spannungsamplituden (r = 1.16 → 1.077, +0.64 dB).

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

> Die Zahlen dieses Abschnitts und von *Leistungskette* stammen von **vor der
> Korrektur 2026-09-11**: A_eff, Leistungen und Aufteilung rechneten mit a²
> statt a. Bei r ≠ 1 waren A_eff und die nötige Leistung um Σa²/Σa zu groß
> (3×4 mit r_x/r_y = 0.97/1.16: 6.4 %, abzüglich +1.1 % statischer Kreuzterm
> bei Phasen 0 → neu A_eff × 0.95), und die Spots/Töne am Rand bekamen r² statt
> r. Bei r_x = r_y = 1 ändert sich nur der statische Kreuzterm.

**Nicht enthalten**: Beugungseffizienz des AOD, Transmission der Optik,
Intermodulation. Angegeben ist die Leistung *im Profil*, also nach dem AOD.

Fehlt `arc` (das `rb85_raman` braucht), bleibt der Dialog benutzbar: C_rabi wird
dann als Eingabefeld freigeschaltet, voreingestellt auf 71.985 — das ist der
Wert bei −8 GHz, bei der Voreinstellung +50 GHz also von Hand auf 11.66 zu
setzen.

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

## Leistungskette: was wo gebraucht wird

Das Leistungsfenster rechnet die ganze Kette durch. Eingaben: Leistung vor dem
AOD, Beugungseffizienz je AOD (der Spot wird **zweimal** gebeugt, x und y
multiplizieren sich), Transmission der Optik. Der Bezugsbereich hat jetzt
ebenfalls die atomgewichtete Option — sonst kalibriert man f_rabi auf eine
Fläche und rechnet die Pulsfläche am Atom.

Für den damaligen Arbeitspunkt (waist 1.05 µm) bei Ω/2π = 1 MHz am Atom,
Δ = −8 GHz (Voreinstellung jetzt +50 GHz):

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

## Wo die Bilder landen

`OUT_DIR_CANDIDATES` zeigt jetzt auf **`GUI/Bilder`** — unter Windows zuerst
absolut auf `C:\Users\Legion\OneDrive\Desktop\Multitone_Simulation\GUI\Bilder`,
sonst relativ zur Datei (`<Ordner der GUI>/../Bilder`), damit ein Klon woanders
neben sich selbst schreibt statt auf einen fremden Desktop.

Der Dateiname nennt Beating und Arbeitspunkt:

```
Beating_N3x4_w1.050um_width0.3700MHz_rx0.97_ry1.16_T0-16.22us
        _frabi1.000MHz_tp1.000us_atom_2026-09-09_081823_*.pdf
```

Ein Speichervorgang im Pulsfenster legt ab:

| Datei | Inhalt |
|---|---|
| `…_curves.pdf` | die drei Zeitplots, ohne Erklärungen |
| `…_map.pdf` | die 2D-Karte allein, ohne Titel |
| `…_report.md` | **alle physikalischen Parameter** (Markdown-Tabellen) |
| `…_period.txt`, `…_delayscan.txt`, `…_intensity.txt` | die Kurven als Zahlenspalten |

Der Report enthält Profil (N, widths, r_x/r_y, Waist, Aufbau, λ, Offset, f₀/T₀,
Tonphasen), Auswertebereich (σ, T, ν_r, gewichtet oder harte Maske), Puls
(Ω/2π, T_p, Gesetz, Pulsstart und -ende, ⟨θ⟩, σ_θ, Anregung, η, Toleranz),
Leistung (Δ, C_rabi, I_ref, A_eff, P im Profil, pro Spot, pro RF-Ton), Optik
und AOD (θ_max, f_band, v_ac, Profil, Gitter) sowie die Namen der erzeugten
PDFs.

**t₀ ist der Pulsanfang**, nicht die Mitte: der Puls läuft von t₀ bis t₀ + T_p.
Das T_p/2 in der Boxcar-Formel ist nur die Phase des Fensterschwerpunkts in der
Fourierdarstellung und verschiebt die aufgetragene Zeit nicht. Im obersten Plot
markiert die senkrechte rote Linie t₀, die Schattierung die Pulsdauer.

## Kitayoshi-Phasen

Neben `0`, `Schroeder` und `randomise` gibt es jetzt den Preset **`Kitayoshi`**.
Die Definition (Kitayoshi, Sumida, Shirakawa, Takeshita, *DSP Synthesized
Signal Source for Analog Testing Stimulus and New Test Method*, IEEE Int. Test
Conf. 1985, S. 825–834) ist eine kumulative Summe:

```
φ_k = φ₀ − (2π/N) · Σ_{j=1..k} j = π/2 − π·k(k+1)/N ,   k = 0 … N−1
```

φ₀ = π/2 ist der Startwert, N die Tonanzahl. Der Zuwachs von Ton zu Ton wächst
linear (`Δφ_k = −2πk/N`), die Phase selbst also quadratisch — genau wie bei
Schroeder, wo

```
φ_n = −π·n(n−1)/N ,   n = 0 … N−1
```

gilt. Beide sind dieselbe quadratische Familie, Kitayoshi nur um einen Index
verschoben (`k(k+1)` statt `n(n−1)`) und um π/2 versetzt. Der Sinn ist in
beiden Fällen derselbe: eine quadratische Phase entspricht einem linearen
Frequenzsweep über die Periode, das Signal wird zum gestreckten Chirp und der
Crest-Faktor fällt von √(2N) auf ≈ √2.

Der **Crest-Faktor ist deshalb identisch** — für gleiche Amplituden liefert die
Implementierung

| N | 0 | Schroeder | Kitayoshi |
|---|---|---|---|
| 3 | 2.449 | 2.160 | 2.160 |
| 4 | 2.828 | 2.000 | 2.000 |
| 13 | 5.099 | 1.878 | 1.878 |
| 14 | 5.292 | 1.943 | 1.943 |

Was sich unterscheidet, ist die **Zuordnung** der Phasen zu den Tönen: Kitayoshi
läuft in der anderen Richtung durch die Parabel und startet bei π/2. Im
Multitone-Profil sind Töne aber nicht austauschbar — jeder gehört zu einem
Beugungswinkel — also ergibt sich ein anderes räumliches Muster und ein anderes
Beating, obwohl der zeitliche Crest-Faktor gleich bleibt. Genau dafür ist der
Preset gedacht: eine zweite Variante mit gleich gutem Crest-Faktor, die man im
Labor durchprobieren kann.

## Normierung im Pulsfenster: Leistung → Ω statt Ω → Leistung

Bisher war Ω/2π = 1 MHz gesetzt und die Leistung ergab sich daraus. Das ist
umgedreht worden, weil in der Praxis die Leistung die Vorgabe ist und die
Pulslänge (1 µs) feststeht. Die Combo **`normalise by`** hat drei Stellungen:

| Modus | Eingabe | Ergebnis |
|---|---|---|
| `power in the profile -> Omega` (Standard) | P im Profil | Ω/2π, P pro Spot |
| `power per spot (mean) -> Omega` | P pro Spot (Mittel) | Ω/2π, P im Profil |
| `Omega given -> power` | Ω/2π | P im Profil und pro Spot |

Alle drei Felder werden nach jeder Rechnung konsistent zurückgeschrieben; nur
das zum Modus gehörende ist editierbar. Der Weg ist

```
I_ref = P / A_eff ,   A_eff = ∫I dA / ⟨I⟩_Bereich ,   Ω = C_rabi · I_ref
```

mit C_rabi aus `rb85_raman.py` bei der eingestellten Verstimmung. A_eff wird
jetzt **vor** der Modus-Verzweigung berechnet, damit alle drei Richtungen
dieselbe Fläche benutzen.

### Neuer Arbeitspunkt: Δ = +50 GHz blau, P = 10 µW

Als Standard steht jetzt eine **blaue Verstimmung von +50 GHz** und ein üblicher
Wert von **10 µW im Profil** (3×4 Töne → 0.83 µW pro Spot im Mittel, 1.00 µW im
hellsten). Damit:

| Größe | Wert |
|---|---|
| C_rabi (+50 GHz) | 11.66 rad/s pro W/m² (bei −8 GHz: 71.98) |
| η = δ_LS/Ω | −0.061 (bei −8 GHz: −0.442) |
| Kontrastgrenze 1/(1+η²) | 0.9963 (bei −8 GHz: 0.8365) |
| A_eff | 14.44 µm² |
| I_ref | 69.3 W/cm² |
| **Ω/2π** | **1.286 MHz** |
| Streuung pro 1-µs-Puls | 0.087 % |
| für einen π-Puls in 1 µs nötig | 3.89 µW |

> Vor der Korrektur 2026-09-11 gerechnet. Mit Leistung ∝ a statt a² (und dem
> statischen Kreuzterm) bei denselben 10 µW, headless nachgerechnet: A_eff
> 14.39 → 13.67 µm² (Phasen 0) bzw. 14.40 → 13.40 µm² (Schroeder), I_ref und
> Ω/2π entsprechend +5.3 % bzw. +7.5 %, π-Puls-Leistung 3.88 → 3.68 µW,
> hellster Spot 1.00 → 0.91 µW. Auch ⟨θ⟩ in der folgenden Tabelle steigt um
> diese Faktoren; σ_θ und der Abfall bei ±1 µs bleiben gleich. Die Anregung bei
> θ ≈ 11 π reagiert auf 5 % in θ empfindlich und ist neu im GUI abzulesen.

Die blaue Verstimmung ist der eigentliche Gewinn: Ω ∝ 1/Δ, die Streurate aber
∝ 1/Δ², und die Ungleichheit der beiden Raman-Zweige (η) fällt ebenfalls,
sodass die Kontrastobergrenze von 84 % auf 99.6 % steigt. Die Leistung bleibt
dabei völlig unkritisch — 10 µW im Profil gegen ≈ 118 mW verfügbar.

Mit 10 µW und 1 µs ergibt sich (atomgewichtet, 3×4-Arbeitspunkt):

| Phasen | ⟨θ⟩ | σ_θ | Anregung | Abfall bei ±1 µs |
|---|---|---|---|---|
| 0 | 11.37 π | 0.77 % | 0.687 | 45.1 % |
| Schroeder | 6.98 π | 18.7 % | 0.500 | 42.5 % |
| Kitayoshi | 6.37 π | 22.0 % | 0.500 | 42.4 % |

Die Flächen liegen deutlich über π, weil 10 µW ein runder Vorgabewert ist und
kein π-Puls-Punkt; der π-Puls bei 1 µs bräuchte 3.89 µW. Für die
Trigger-Diskussion ist das ohne Belang — der Delay-Scan ist in θ linear in der
Leistung, die relative Empfindlichkeit ändert sich nicht.

### Warum Δ vorher nichts geändert hat

Zwei Fehler, beide behoben:

1. **Kein Auto-Recompute.** Nur `layout` und `title inside the figure` waren mit
   einem Signal verbunden; jede Zahl — auch Δ — wurde erst beim Druck auf
   `Recompute` wirksam. Das Fenster zeigte dann eine Abbildung, die nicht mehr
   zu den Zahlen darüber gehörte. Jetzt hängen alle physikalischen Eingaben an
   einem entprellten Timer (250 ms), sodass eine Änderung von selbst
   durchrechnet. Das Zurückschreiben von P/Ω in `recompute()` ist mit
   `blockSignals` geschützt und kann keine Rückkopplung auslösen.
2. **η kam aus dem Haupt-GUI.** Die Anregungskurve benutzte `state["eta_ls"]`
   statt des zur eingestellten Verstimmung gehörenden η. Jetzt wird
   `self._coef["eta"]` aus `rb85_raman` genommen; der Haupt-GUI-Wert ist nur
   noch Notnagel, wenn das Modul nicht ladbar ist.

Zusätzlich meldet die Notizzeile jetzt **rot und fett**, wenn `rb85_raman`
(also `arc`) nicht importiert werden konnte — dann ist C_rabi auf seinem
−8-GHz-Wert eingefroren und Δ ist tatsächlich wirkungslos. Sie zeigt außerdem
η mit an.

Zur Kontrolle (10 µW im Profil, 3×4, atomgewichtet, T_p = 1 µs):

| Δ | C_rabi [rad/s pro W/m²] | η | Ω/2π | ⟨θ⟩ |
|---|---|---|---|---|
| −8 GHz | 71.98 | −0.442 | 7.93 MHz | 70.2 π |
| +20 GHz | 29.05 | −0.155 | 3.20 MHz | 28.3 π |
| +50 GHz | 11.66 | −0.061 | 1.28 MHz | 11.4 π |
| +200 GHz | 2.98 | −0.015 | 0.33 MHz | 2.9 π |

Bei fester Leistung ist Ω ∝ C_rabi ∝ 1/Δ, die Streurate ∝ 1/Δ² — deshalb wird
weiter außen zwar die Fläche kleiner, das Verhältnis Fläche zu Streuung aber
besser.

### Warum sich der 2D-Plot mit Δ nicht ändern *kann*

Das ist kein Fehler, sondern die Struktur der Rechnung:

```
θ(r) = Ω(r) · T_p = C_rabi(Δ) · I(r) · T_p
```

Bei **fester Leistung** ist Δ ein reiner skalarer Vorfaktor — jeder Pixel wird
mit derselben Zahl multipliziert. `imshow` skaliert die Farbskala automatisch
auf min/max, also ist das Bild danach **Pixel für Pixel identisch**; nur die
Zahlen an der Farbleiste wandern (von ~62–71 π bei −8 GHz auf ~10.0–11.4 π bei
+50 GHz). Numerisch geprüft: das Verhältnis der beiden Karten ist auf 10⁻⁹
konstant, der Kontrast (max−min)/mean bleibt exakt 0.127711.

Sichtbar wird Δ erst in einer Größe, die **nichtlinear** in θ ist. Deshalb hat
das Fenster jetzt die Combo **`map shows`**:

| Einstellung | Dargestellt | Δ-Abhängigkeit |
|---|---|---|
| `pulse area theta(r)` | θ(r)/π, Farbskala autoskaliert | nur die Farbleiste |
| `excitation p(r)` | p = sin²(√(1+η²)·θ/2)/(1+η²), Skala fest 0…1 | volle Musteränderung |

Die Anregungskarte trägt beides: θ selbst *und* η(Δ), das die Kontrastobergrenze
1/(1+η²) setzt. Bei 10 µW im Profil und T_p = 1 µs:

| Δ | C_rabi | η | ⟨θ⟩ | p: min … max | ⟨p⟩ |
|---|---|---|---|---|---|
| −8 GHz | 71.98 | −0.442 | 70.2 π | 0.000 … 0.836 | 0.447 |
| +20 GHz | 29.05 | −0.155 | 28.3 π | 0.000 … 0.977 | 0.465 |
| +50 GHz | 11.66 | −0.061 | 11.4 π | 0.007 … 0.996 | 0.796 |
| +200 GHz | 2.98 | −0.015 | 2.9 π | 0.597 … 0.985 | 0.907 |

Bei −8 GHz laufen über die Atomwolke mehrere volle Rabi-Zyklen — die Karte ist
ein Streifenmuster, das Atom sitzt auf mehreren Fransen gleichzeitig, und die
Anregung mittelt sich zu ½ weg. Je weiter blau, desto weniger Zyklen passen in
die Wolke und desto homogener wird p. Genau das ist die Aussage, die man aus
der θ-Karte nicht ablesen kann.

Die beiden Karten gehen jetzt auch unter verschiedenen Namen ins Bilderverzeichnis:
`…_areamap.pdf` bzw. `…_excmap.pdf`, damit eine gespeicherte θ-Karte nicht von
einer p-Karte überschrieben wird.

## Umbau 2026-09-11: Physik in `kern/beating_physik.py`

* **Alles Physikalische** aus `Beating_Multitone_GUI.py` — Konstanten,
  Frequenzen und Geometrie, Felder, Beat-Frequenzen und Entartung, Tonphasen,
  Zeitreihe, exakte Zeitstatistik, Gütemaße, Leistung, Atomgewichtung,
  Pulsfläche — steht jetzt in `kern/beating_physik.py`, nach Themen in
  Abschnitte 0–11 sortiert. Die Funktionen sind unverändert übernommen (per
  Syntaxbaum-Vergleich geprüft); das GUI schrumpft von 3313 auf gut 2050
  Zeilen.
* Die Nebenfenster bekommen die Funktionen nicht mehr als Dictionary `fns` vom
  GUI gereicht, sondern importieren `beating_physik` selbst.
* `kern/` ist ein Paket (`kern/__init__.py`). GUI und Nebenfenster importieren
  `from kern.beating_physik import …` bzw. `from kern import beating_physik` —
  ohne Umweg über `sys.path`, damit PyCharm den Import auflöst und nicht rot
  markiert. `kern/beating_profil.py` und `Rabi_Rb85_GUI.py` bleiben beim
  bisherigen Weg (`kern/` im `sys.path`).
  `one_lens_design.py` nimmt θ_max, f_band, `amps_from_ratio()` und das
  Spotprofil ebenfalls von dort, statt eigene Kopien zu halten.
* `kern/beating_profil.py` importiert nicht mehr das ganze GUI (und damit Qt),
  sondern nur `beating_physik.py`. fLO, θ_max und f_band gehen jetzt als
  Argumente an `compute_centers_and_freqs()`, statt vorübergehend die
  Modulkonstanten zu überschreiben. Sein Arbeitspunkt `WP` entspricht den
  Startwerten des GUI (waist 1.04 µm, width 0.37 MHz, r_x/r_y = 0.97/1.16,
  f_Rabi = 1 MHz) mit Δ = +50 GHz.
* Airy-Faktor fest 1.4830, waist-Startwert 1.04 µm, Δ-Voreinstellung im
  Leistungsfenster +50 GHz (wie im Pulsfenster).
* `np.trapezoid` fällt unter numpy < 2 automatisch auf `np.trapz` zurück.

Kontrolle: altes und neues GUI mit identischen Eingaben headless durchgerechnet
(Startpunkt, Schroeder, Kitayoshi, Ω ~ √I, ungleiche widths, Gauß, Ein-Linse
mit Belichtung, flacher Punkt, Puls-Optimierer, alle 9 Panels, alle drei
Nebenfenster, Kandidatensuche des Ein-Linsen-Dialogs, `beating_profil.profil()`)
— alle Arrays und Anzeigetexte **bitgleich**.

## Physik-Korrekturen 2026-09-11

Beim Durchgehen der Physik gefunden und behoben, in beiden Repos gleich:

* **Leistung ∝ a, nicht a².** `amps` sind Intensitätsgewichte, das Feld trägt
  √a (`build_field_stack()`). `profile_total_power()`, die Aufteilung in
  `power_budget.py` und `pulse_timing.py` und `plateau_ripple()` in
  `one_lens_design.py` rechneten mit a². Neu in `kern/beating_physik.py`:
  `spot_power_shares()`, `tone_power_shares()`, `rf_voltage_ratios()`.
  Bestätigt gegen ein Gitterintegral des Zeitmittels (Gauß exakt, Airy bis auf
  0.12 % Abschneiden bei 200 µm).
* **Statischer Kreuzterm in der Gesamtleistung.** Frequenzentartete Spots
  tragen `2·√(a_s a_s')·cos Δφ·∫u_s u_s' dA` bei. Das Überlappintegral ist
  analytisch (`spot_overlap_integral()`: Gauß `πw²/2·exp(−d²/2w²)`, Airy
  `P_Spot·u(d)`). 3×4, Airy, Phasen 0: +1.1 % — nicht „weit unter einem
  Promille“, wie es vorher hieß (die alte Aussage bezog sich auf 13×14).
* **`kalibrieren_auf_leistung()`** in `kern/beating_profil.py` summierte über das
  abgeschnittene Rechengitter (Airy: −4.7 %, Intensität also ~5 % zu hoch).
  Jetzt analytisch über `profile_total_power()`. Betrifft `Rabi_Rb85_GUI.py`
  und `beispiele/rabi_pro_atom.py`.
* **Lichtverschiebung bei Ω ~ √I.** Die geschlossene Form mit η setzt δ/Ω =
  const voraus und wurde trotzdem angewandt (Rabi-Panel, Pulsfenster). Jetzt
  numerische Propagation (`sqrt_law_excitation()`), geprüft gegen
  `solve_ivp` (Abweichung < 10⁻⁵) und gegen die geschlossene Form bei η = 0.
  Bei η = 0.3 wichen die Rabi-Kurven vorher um bis zu 5 Prozentpunkte ab.
* **Pulsfenster bei Ω ~ √I** stürzte mit Normierung über die Leistung ab (A_eff
  ist dort nicht definiert, alles NaN). Es schaltet jetzt auf *Omega given*.
* **Crest-Faktor** mit RF-Spannungen √r statt r (3×4 Startwert: y 2.82 → 2.83).
* **Texte:** Resonanzprüfung gilt nur für Beleuchtung über viele
  Fallenperioden; `compute_grid()` nennt „keine Nachbar-Sites“ jetzt als
  Annahme (kohärent getriebene Nachbarn würden linear im Nachbarfeld
  interferieren); η-Tooltip: Größenordnung ω_HF/Δ statt ω_HF/(2Δ), η = 0 heißt
  „vernachlässigt“, nicht „kompensiert“.

Unverändert und weiter offen: C_rabi-Ersatzwert ohne `arc` ist der −8-GHz-Wert;
`Rabi_Rb85_GUI.py` steht noch auf dem alten Arbeitspunkt.

Kontrolle: altes und neues GUI headless mit identischen Eingaben. Unverändert
(bitgleich): f₀, Zeitmittel, Varianz, σ_t/⟨I⟩, Pulsflächen-Uniformity,
Rabi-Kurven für Ω ~ I. Geändert nur die oben genannten Größen.

## Umbau 2026-09-14: Phasenoptimierung auf das Atom

* **Neues Ziel `beating at the atom`** (Standard) im Kasten *Tone phases*:
  minimiert R = σ_t₀(A)/⟨A⟩, den Hub der atomgewichteten Pulsfläche über den
  Trigger-Zeitpunkt, und legt t₀ anschließend auf das Maximum von A. Am
  Arbeitspunkt 115 % → 41 %; der Crestfaktor fällt dabei von 2.45/2.83 auf
  1.82/2.20 mit. Einzelheiten in *Phasen für das Atom optimieren*.
* **Abschnitt 12 in `kern/beating_physik.py`**: `AtomBeating` (geschlossene
  Form über Parseval, ~13 µs je Auswertung) und `atom_beating_at_centre`.
  Geprüft gegen eine Brute-Force-Zeitreihe (200 000 Stützstellen, Abweichung
  < 10⁻⁵ relativ), gegen den Effektivwert der Kurve (bitgleich) und am
  Grenzfall T_p = T₀ (R = 5·10⁻¹⁷).
* **Neue Eingaben**: Pulslänge T_p für die Optimierung (Standard 1 µs,
  unabhängig von f_Rabi — das Experiment legt den Puls fest, nicht die
  π-Bedingung; der alte Optimierer benutzte stillschweigend 1/(2 f_Rabi)),
  dazu atom T und ν_r mit σ-Anzeige. Beide Optimierer rechnen jetzt mit
  demselben T_p.
* **Der alte Optimierer** heißt `pulse area, spatial spread over a region` und
  ist inhaltlich unverändert; nur die Pulslänge kommt jetzt aus dem neuen Feld.
  Die Eingabefelder des jeweils nicht gewählten Ziels werden ausgegraut.
* **Keine Crest-Nebenbedingung**, obwohl φ = 0 für die RF-Kette ausscheidet:
  Helligkeit am Trigger und Crestfaktor messen dasselbe (beide: wie stark
  rephasieren die Töne), das R-Optimum unterbietet φ = 0 also von selbst.
  Pareto-Tabelle im Abschnitt.
* **Verworfen**: die Jitter-Zielfunktion (degeneriert — findet flache *dunkle*
  Stellen mit schlechterer Gesamtschwebung) und R/A(t₀) als Einzelzahl
  (unterscheidet nicht). Beides mit Zahlen im Abschnitt dokumentiert.

Kontrolle: GUI headless gebaut und beide Ziele durchgerechnet; das
Regions-Ziel liefert unverändert dieselben Phasen wie vorher, sofern T_p auf
1/(2 f_Rabi) gesetzt wird.

## Nachtrag 2026-09-15: Helligkeit statt Schwebung als Standardziel

Der Umbau vom 14. hat auf R optimiert — die *ungetriggerte* Größe. Für einen
getriggerten Puls ist das die falsche Zielfunktion, und zwar spürbar: das
R-Optimum liefert A(t₀) = 1.47·⟨A⟩, das Schroeder-Preset 2.70. Der Optimierer
machte den Puls also **dunkler als ein Preset**, um eine Schwankung zu
unterdrücken, die ein fester Trigger ohnehin einfriert.

* **Neues Standardziel** `pulse area at the atom, max at the trigger (crest
  limited)`: maximiert `max_t A(t)/⟨A⟩` unter Crest ≤ C. Bei C = 2.0 kommt
  2.89·⟨A⟩ heraus, also fast das Doppelte des R-Optimums und über beiden
  Presets.
* **Neues Eingabefeld `max. crest factor`** (Standard 2.0). Es ist die einzige
  Nebenbedingung, die das Problem wohlgestellt macht — ohne sie ist die Antwort
  immer φ = 0. Die Schranke ist immer bindend.
* Das R-Ziel bleibt als `beating at the atom, minimal swing (untriggered)`
  erhalten, das Regions-Ziel unverändert. Die Combobox hat jetzt drei Einträge.
* **`CrestBasis`** (Abschnitt 5 von `beating_physik.py`): crest_factor mit
  vorberechnetem Zeitraster. Schranke und Anzeige benutzen dasselbe Objekt, der
  angezeigte Wert kann also nicht mit dem Limit streiten, gegen das optimiert
  wurde. Gegen `crest_factor()` auf vier Stellen identisch.
* **Zeitbasis in `AtomBeating` gecacht.** `peak_ratio()` baute vorher bei jedem
  Aufruf eine exp-Matrix (n_t × Ordnungen) — in der Schleife der teuerste
  Posten überhaupt. Laufzeit von 186 s auf 22 s.
* Die Suche zielt auf C − 0.015 statt auf C: ein weicher Strafterm bleibt
  knapp über seiner Schwelle stehen, und ein angezeigter Crest von 2.02 gegen
  ein Limit von 2.00 ist genau die Sorte Zahl, die man nicht erklären möchte.

Kontrolle: GUI headless, alle drei Ziele; C = 1.9 / 2.0 / 2.2 / 2.5 liefert
2.65 / 2.89 / 3.36 / 3.83 bei Crest 1.89 / 1.99 / 2.19 / 2.49 — die Schranke
wird in keinem Fall überschritten. Das R-Ziel gibt unverändert 41.1 %.

## Nachtrag 2026-09-15 (2): Crestfaktor gehört in die Zielfunktion

Auf die Frage, warum der Crestfaktor überhaupt relevant ist, kam heraus, dass
die Zielfunktion vom Vormittag immer noch falsch normiert war. Sie maximierte
A(t₀) bei *fester* optischer Leistung unter einer Crest-Schranke. Die Leistung
ist aber nicht fest: bei einem spitzenspannungsbegrenzten Verstärker ist
P_mittel ~ 1/crest², und beide Achsen beugen. Ein höherer Crest kostet also
mehr, als der hellere Peak einbringt.

* **Neue Zielgröße** `F = max_t A(t) / (crest_x² · crest_y²)` — Rabi-Fläche pro
  Volt Verstärker-Reserve. Ergebnis: Crest 1.82/1.77, F = 0.239, das ist
  2.7× φ = 0 und 1.6× Schroeder.
* **Das Feld `max. crest factor` ist wieder verschwunden.** Es war nur nötig,
  solange die Zielfunktion falsch normiert war — jetzt steckt der Trade-off in
  der Zielgröße selbst und es gibt keinen freien Parameter mehr.
* **UI aufgeräumt**: der Optimierer-Knopf sitzt jetzt im Kasten *Tone phases*
  direkt bei den Einstellungen, die er liest (vorher zwei Kästen weiter oben in
  *Pulsed operation*), und unter der Zielauswahl steht eine Klartextzeile, die
  sagt, was maximiert bzw. minimiert wird.
* Quadratur-Strafterm auf 0.10 hochgesetzt — gegen eine Zielgröße von 0.24
  wurde er bei 0.02 einfach weggekauft.

Kontrolle: 400 Startpunkte, F = 0.2385 mit Quadratur / 0.2503 ohne; GUI
headless mit allen drei Zielen. Referenzwerte φ=0 0.0887, Schroeder 0.1451,
Kitayoshi 0.1326, R-Optimum 0.0908.

## Nachtrag 2026-09-15 (3): ein Ziel, und es ist die Gleichmäßigkeit

Die drei Ziele waren verwirrend und alle drei beantworteten eine Frage, die das
Experiment nicht stellt. Die eigentliche Frage lautet:

> Sitzt der Interferenzfleck während des Pulses **mittig** auf dem Atom, und
> ist die aufgesammelte Pulsfläche über die Ausdehnung des Atoms
> **gleichmäßig**?

Läuft der Fleck halb am Atom vorbei, ist θ auf der einen Seite der
Aufenthaltsverteilung größer als auf der anderen. Der Drehwinkel hängt dann
davon ab, wo das Atom gerade war — und kein Nachkalibrieren des Mittelwerts
repariert das. Genau das misst

```
U_W = σ_W(θ) / ⟨θ⟩_W
```

Ein seitlicher Versatz erzeugt einen linearen Gradienten quer durch die Wolke,
und ein Gradient ist das, was U_W sieht: **„gleichmäßig" und „mittig" sind
dieselbe Bedingung**, deshalb gibt es nur eine Zahl zu minimieren. Der
Schwerpunktversatz wird trotzdem in Nanometern ausgegeben.

### Rechnung

θ(r) ist linear in den Paarprodukten, θ(r) = Σ_p a_p·(g_s g_s')_p(r) mit
a_p = Re[e^{i(φ_s−φ_s')}·c_{d(p)}]. Damit sind ⟨θ⟩_W = Σ_p a_p m_p und
⟨θ²⟩_W = Σ_pq a_p a_q M_pq, und M ist eine feste 79×79-Matrix. Eine Auswertung
kostet ein aᵀMa statt einer Rechnung über 961 Pixel. Geprüft gegen
`PulseArea.theta` auf demselben Gitter: identisch in allen Stellen.

### Der Zielkonflikt

Der Moment, in dem die Töne rephasieren, ist der Moment, in dem der Fleck
sauber, symmetrisch und mittig ist — genau das macht die Beleuchtung
gleichmäßig. Dasselbe Rephasieren treibt den Crest auf √(2N). **Gleichmäßige
Beleuchtung und schonende RF-Kette stehen direkt gegeneinander**, und die
Crest-Schranke ist, wo man sich auf dieser Kurve platziert.

| Phasen | U_W | Versatz | A(t₀)/⟨A⟩ | Crest x / y |
|---|---|---|---|---|
| optimiert, C = 1.9 | 0.80 % | 0.1 nm | 0.99 | 1.89 / 1.89 |
| optimiert, C = 2.5 | 0.18 % | 0.0 nm | 1.13 | 2.45 / 2.49 |
| alle Phasen 0 | 0.18 % | 0.0 nm | 1.35 | 2.45 / 2.83 |
| ganz ohne Schranke | 0.09 % | 0.0 nm | 3.81 | 2.45 / 2.82 |
| Schroeder | 18.4 % | 19.4 nm | 1.00 | 2.16 / 2.00 |

Schroeder minimiert den Crest und weiß nichts von einem Atom — daher die 18 %.
Ohne Schranke läuft die Suche in eine **lineare Phasenrampe**, und die ist
φ = 0 seitlich verschoben: für das Atom ideal, für den Verstärker der
schlechteste Fall. Eine Rampe ändert den Crest nämlich gar nicht, sie
verschiebt nur die Hüllkurve in der Zeit.

Nebenbedingung außerdem: A(t₀) ≥ ⟨A⟩, damit der Trigger nicht in eine Delle
fällt. Bei C = 1.9 ist sie bindend (0.99) — die Gleichmäßigkeit möchte dort
noch dunkler werden.

### Entfernt

Die Zielauswahl mit drei Einträgen, dazu *pulse area per amplifier headroom*,
*minimal swing* und *spatial spread over a region*, sowie das Feld
*Target region*. Ihre Zahlen stehen in den Abschnitten oben und in den
Nachträgen (1) und (2) als Befund. Der Auswertekreis (*evaluation circle r*)
bleibt — er gehört zu den Plots und den Nebenfenstern, nicht zur Phasensuche.

### Bedienung

Alles in einem Kasten, *Tone phases*: Presets, eine Klartextzeile die sagt was
gesucht wird, T_p, **max. crest factor**, atom T und ν_r mit σ-Anzeige, der
Auswertekreis, der Quadratur-Haken, der Knopf, darunter die Phasenfelder und
die Ergebniszeile. Rund 2–3 Minuten für 120 Startpunkte.
