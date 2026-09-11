# Rabi Rb85 GUI

Rabi-Oszillationen auf dem Zweiphotonen-Raman-Übergang von Rb-85,
`|5S₁ᐟ₂, F=2, m⟩ ↔ |5S₁ᐟ₂, F=3, m⟩`, mit echten Atomdaten aus ARC.

## Was du ausführst

```
python Rabi_Rb85_GUI.py
```

**Das ist alles.** Der Rest wird importiert oder ist optional.

## Aufbau

```
Beating Analyse/
├── Rabi_Rb85_GUI.py          ← STARTEN
├── Beating_Multitone_GUI.py  ← euer Beating-GUI (starten, wenn ihr das wollt)
├── README_Rabi_Rb85.md
├── kern/                     ← Hilfsmodule, NICHT starten
│   ├── rb85_raman.py             die Physik (ARC, Raman-Koeffizienten)
│   ├── beating_physik.py         Physik des Beating-GUI (Felder, Beating, Pulsfläche)
│   └── beating_profil.py         Brücke zum Beating-GUI
└── beispiele/                ← optionale Rechenbeispiele
    ├── beispiel_profil.py        eigenes I(t) einspeisen
    └── rabi_pro_atom.py          ein Atom unter einem 3×4-Flattop
```

`kern/rb85_raman.py` und `kern/beating_profil.py` lassen sich zwar direkt
starten (sie geben dann ein paar Kennzahlen aus), aber ihr Zweck ist der
Import. Die Suchpfade werden relativ zur jeweiligen **Datei** gesetzt, nicht
zum Arbeitsverzeichnis — Start aus PyCharm und aus der Konsole verhalten sich
also gleich.

Bilder gehen nach `GUI/Bilder`, wenn dieser Ordner unter `GUI/` liegt, sonst
nach `Bilder/` neben dem Skript. Zu jedem PDF wird eine `.json` mit allen
Parametern geschrieben.

## Abhängigkeiten

```
pip install -r requirements.txt
```

## Was „physikalisch genau" hier heißt

Effektives Zwei-Niveau-System, aber **die Koeffizienten sind keine freien
Parameter**. Die 5P-Zustände werden adiabatisch eliminiert — die Summe läuft
dabei über *beide* Feinstrukturlinien und *alle* ihre Hyperfeinniveaus, mit
vorzeichenbehafteten Dipolmatrixelementen aus ARC. Heraus kommen drei
Koeffizienten, alle linear in der Intensität:

```
Omega(I)    = C_rabi    * I     Zweiphotonen-Rabifrequenz
delta_LS(I) = C_shift   * I     differentieller Lichtshift
Gamma(I)    = C_scatter * I     Streurate
```

Weil Ω und δ_LS denselben Faktor I tragen, ist η = δ_LS/Ω eine **reine Zahl** —
unabhängig von Intensität, Ort und Zeit. Genau das schließt das
Zwei-Niveau-Problem: der Lichtshift deckelt den Kontrast auf 1/(1+η²) und
beschleunigt die Rotation um √(1+η²), erzeugt aber keine eigene Struktur.

Bei konstanter Intensität ist die Lösung dann exakt das Lehrbuchergebnis
`P = Ω²/(Ω²+δ²)·sin²(√(Ω²+δ²)t/2)`, und das GUI benutzt diese geschlossene
Form. Sobald ein Term dazukommt, der **nicht** mit I skaliert — eine
Zeeman-Verstimmung, eine statische Zweiphotonen-Verstimmung — kommutiert H(t)
zu verschiedenen Zeiten nicht mehr, die geschlossene Form gilt nicht, und
`excitation_series()` integriert schrittweise.

## Geometrie, die angenommen wird

Kopropagierend, σ⁺ auf beiden Zweigen. Das ist keine Kosmetik: der **skalare**
Anteil des AC-Stark-Operators kann einen Hyperfeinübergang gar nicht treiben,
nur der **Vektor**-Anteil, und der geht wie ε₁\* × ε₂ — für zwei *linear*
polarisierte Zweige exakt null. σ⁺/σ⁺ gibt ε\* × ε = ẑ und treibt Δm_F = 0.

Kopropagierend heißt außerdem Δk = ω_hfs/c = 64 m⁻¹, Rückstoßgeschwindigkeit
5·10⁻⁸ m/s: dopplerfrei und bewegungsunabhängig.

## Panels

1. **Zeeman-Spektrum** — Anregung über der Zweiphotonen-Verstimmung, alle fünf
   m_F. Beantwortet: trennt mein Bias-Feld den Uhrenübergang sauber ab?
2. **Verstimmungsbudget** — η, Kontrastdeckel und Streuung über Δ. Die beiden
   laufen gegeneinander; hier sieht man den Kompromiss.
3. **Zwischenzustände** — Beitrag jedes F' zu Ω, symlog mit Vorzeichen.
4. **Leistungsskalierung** — Ω/2π und t_π über der Leistung.
5. **m_F-Vergleich** — alle fünf Unterzustände einzeln plus thermisches Mittel.

## Multiton-Profil statt Gaußstrahl

Im GUI: Strahl → Vorgabe → **"Multiton-Profil (Leistung je Profil)"**. Dann
wird die Gruppe *Multiton-Profil* aktiv, mit denselben Parameternamen wie im
Beating GUI (N_x, N_y, waist, width, r_x, r_y, f1, f2, fLO, Offset, Tonphasen)
plus **Leistung je Profil** und **Pulsstart t₀**. Voraussetzung: `beating_profil.py`
und `beating_physik.py` liegen zusammen in `kern/` (das Beating-GUI selbst wird
dafür nicht gebraucht).

Zwei Dinge, die dort anders sind als beim Gaußstrahl:

* **Jeder Ort wird einzeln propagiert** und erst danach gemittelt. Die
  Intensitätsstreuung über den Auswertekreis ist genau das, was den Kontrast
  frisst — wer vorher mittelt, rechnet sie weg.
* **t₀ ist ein Parameter, kein Detail.** Die Pulsfläche hängt stark davon ab,
  wo im Beat-Zyklus der Puls startet. Der Knopf *besten t₀ suchen* nimmt die
  gleichmäßigste Fläche, aber nur unter Startzeiten nahe am Median — sonst
  landet die Suche auf der Flanke des Rephasierungs-Kamms, wo die Fläche ein
  Vielfaches ist und der „π-Puls" keiner mehr.

Die Infozeile zeigt **t_π/T₀**. Ist das ≪ 1, mittelt der Puls das Beating
nicht weg, sondern sieht eine eingefrorene Momentaufnahme — dann wird die
Sache schlechter, nicht besser, wenn man die Leistung erhöht.

## Als Modul: eigenes I(t) einspeisen

Für ein beliebiges zeitabhängiges Profil geht I(t) direkt an das Physik-Modul:

```python
import numpy as np, rb85_raman as R
raman = R.RamanRb85(delta_Hz=-8e9)
t  = np.linspace(0, 8e-6, 6001)
It = ...                                  # euer I(t) in W/m^2
P  = R.excitation_series(t, It, raman, m=0, B_gauss=3.0)
```

Siehe `beispiel_profil.py`.

## Verifikation

* geschlossene Form gegen schrittweise Integration bei konstantem I:
  **10⁻¹⁴** (Maschinengenauigkeit — der Mittelpunkt-Schritt ist dort exakt)
* zeitabhängiges I mit δ = 0 gegen das Pulsflächen-Theorem sin²(θ/2): **10⁻¹⁴**
* P_max = 1/(1+η²) und t_π = 1/(2·f_Rabi·√(1+η²)) auf 4 Stellen
* Lichtshift-Kompensation → 100.000 % (ohne Streuung)
* ARC gegen Steck `rubidium85numbers.pdf`: reduzierte Matrixelemente
  4.2321 / 5.9783 e·a₀ (Steck 4.231 / 5.977), Γ/2π = 5.7478 / 6.0659 MHz,
  ω_hfs = 3.0357324 GHz — alle auf 4+ Stellen
* eigene Ω = d·E₀/ħ-Konvention gegen `arc.getRabiFrequency`: Verhältnis 1.000000
* Paritätstest η_{σ−}(m) = η_{σ+}(−m): exakt erfüllt
* 60 Kombinationen aus Panel × m_F × B-Feld: 0 ≤ P ≤ 1, alles endlich

## Was NICHT drin ist

Atombewegung und Falle · endliche Pulsflanken · Polarisationsfehler ·
Rückpumpen aus den dunklen m = ±3 in F=3 · der angeregte Zustand jenseits der
adiabatischen Elimination (bei |Δ| von GHz gegen Γ/2π = 5.75 MHz sehr gut, und
die Streuung, die er kostet, *ist* als Überlebensfaktor drin).
