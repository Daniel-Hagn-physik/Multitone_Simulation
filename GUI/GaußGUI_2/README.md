# Gaußstrahl-Propagation – GUI

Berechnet die Propagation eines Gaußstrahls durch eine frei zusammengestellte
Optik-Kette und liefert am Ausgang **Waist-Radius, Waist-Position und
Divergenzwinkel** – getrennt für die **tangentiale** und die **sagittale**
Ebene, damit gekippte Flächen (Kristalle, Brewster-Platten, Faltspiegel)
korrekten Astigmatismus erzeugen.

## Installation & Start

```bash
pip install PySide6 matplotlib numpy
python gui.py
```

`gaussbeam.py` ist die Physik-Engine und hat außer `numpy` keine Abhängigkeit –
du kannst sie auch ohne GUI im Skript benutzen.

## Dateien

| Datei | Inhalt |
|---|---|
| `gaussbeam.py` | Physik: q-Parameter, ABCD-Matrizen, Komponenten, Optimierer |
| `gui.py` | PySide6-Oberfläche mit Plot |
| `test_gaussbeam.py` | 23 analytische Gegenrechnungen (`python test_gaussbeam.py`) |

## Bedienung

1. **Eingangsstrahl** links oben: Wellenlänge, Waist-Radius w0 (1/e²-Feldradius),
   Strahlqualität. Für die Divergenz hast du drei Möglichkeiten:
   * *ideal gaußisch* – θ = λ/(π·w0), M² = 1
   * *M² vorgeben*
   * *Divergenz vorgeben* – daraus wird M² = π·w0·θ/λ berechnet
   „kollimiert“ heißt: der Waist liegt genau an der ersten Komponente
   (ebene Phasenfront). Sonst gibst du den Abstand Waist → erste Komponente an
   (negativ = Waist liegt hinter der ersten Komponente).
2. **Komponenten** in Reihenfolge des Strahlwegs hinzufügen. Häkchen = aktiv,
   so kannst du ein Element testweise herausnehmen ohne es zu löschen.
3. **Parameter** des markierten Elements unten links ändern – es wird sofort
   neu gerechnet.
4. **Ziel-Optimierung**: einen Parameter freigeben (z. B. den Abstand vor der
   Linse), Suchbereich und Ziel wählen (Waist-Größe, Waist-Position,
   kollimieren, Astigmatismus wegkippen …). Grobes Raster + Golden-Section.

## Komponenten

| Element | Parameter |
|---|---|
| Abstand / Freiraum | Länge (im jeweils aktuellen Medium) |
| Dünne Linse | f, Kippwinkel (f_t = f·cosθ, f_s = f/cosθ) |
| Dicke Linse | R1, R2, Mittendicke, n, Kippwinkel |
| Grenzfläche | R (0 = plan), Index danach, Einfallswinkel |
| Kristall / planparallele Platte | Dicke, n, Einfallswinkel |
| Kristall im Brewsterwinkel | Dicke, n (Winkel wird gesetzt) |
| Gekrümmter Spiegel | R, Einfallswinkel (f_t = R·cosθ/2, f_s = R/(2cosθ)) |
| Freie ABCD-Matrix | A, B, C, D |

Der Brechungsindex wird durch das System durchgereicht: nach einer Grenzfläche
mit n2 = 1.5 rechnet der nächste „Abstand“ automatisch im Glas.

## Konventionen (wichtig fürs Weiterbauen)

* Strahlvektor (y, θ) mit **echten** Winkeln, deshalb det(M) = n1/n2 an einer
  Grenzfläche.
* q im **lokalen** Medium: `1/q = 1/R − i·λ_eff/(π·n·w²)` mit `λ_eff = M²·λ₀`.
  Freiraum: `q → q + L` (geometrische Länge). Element: `q → (Aq+B)/(Cq+D)`.
* Gekippte sphärische Fläche (aus den Coddington-Gleichungen), mit
  `P = (n₂cosθ₂ − n₁cosθ₁)/R`:

  ```
  tangential:  A = cosθ₂/cosθ₁     C = −P/(n₂cosθ₁cosθ₂)   D = n₁cosθ₁/(n₂cosθ₂)
  sagittal:    A = 1               C = −P/n₂               D = n₁/n₂
  ```

* Kippwinkel > 0 heißt: Einfallswinkel gegen die Flächennormale, die
  Einfallsebene ist die *tangentiale* Ebene.
* `z_to_waist > 0`: der Waist liegt **hinter** dem Ausgang (der Strahl läuft
  noch zusammen). `< 0`: der Waist liegt bereits im System.

Testbeispiel: eine gekippte Platte wirkt wie ein Freiraum der Länge
`t/(n·cosθ₂)` (sagittal) bzw. `t·cos²θ₁/(n·cos³θ₂)` (tangential) – genau das
prüft `test_gaussbeam.py` nach.

## Engine ohne GUI

```python
from gaussbeam import InputBeam, OpticalSystem, Distance, ThinLens, BrewsterCrystal

sys_ = OpticalSystem(
    InputBeam(wavelength=800e-9, w0=1e-3),          # kollimiert, 1 mm Waist
    [Distance(L=300.0), ThinLens(f=100.0), Distance(L=90.0),
     BrewsterCrystal(t=3.0, n=1.76), Distance(L=50.0)],
)
r = sys_.propagate()
print(r.tangential.w0, r.tangential.z_to_waist, r.tangential.theta)
print("Astigmatismus:", r.astigmatism)
```

Parameter werden in Anzeige-Einheiten übergeben (mm, µm, Grad), intern in SI
gespeichert; `comp.get("f")` liefert Meter, `comp.set("f", 0.1)` setzt Meter.
