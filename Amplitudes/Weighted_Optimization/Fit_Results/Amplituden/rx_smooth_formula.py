"""
rx_smooth_formula.py
=====================
Geschlossene Formel (Polynom 5. Grades) fuer r_x(waist_mm, width_MHz),
gefittet NUR auf dem "zentralen Diagonalstreifen" des gewichteten
151x151-Amplituden-Scans (scan_amp_data_weighted_N3x4_151x151pts_Airy_2500res.pkl).

Was ist der "zentrale Diagonalstreifen"? Es ist die EINE grosse, zusammen-
haengende, glatte Region der Heatmap, die uebrig bleibt, wenn man
    - die Ridge (die diagonale Sprungkante/Unstetigkeit),
    - das kleine Artefakt oben rechts (isolierte zweite Sprung-Komponente,
      waist ~1.5-1.7mm, width ~0.36-0.4MHz), und
    - die beiden Saettigungs-Ecken (wo r_x oder r_y an die Schranken 0.1
      bzw. 10 laufen)
entfernt. Diese vier Bereiche sind in den Rohdaten bereits geometrisch
sauber getrennt (keine manuelle Box noetig) - siehe stripe_overview.png:
die verbleibende gruene Flaeche ist ein einzelnes zusammenhaengendes,
diagonal verlaufendes Band durch die Mitte des Scanbereichs, exakt was
"zentraler Diagonalstreifen" heisst. Innerhalb dieses Streifens bewegen
sich r_x UND r_y beide nur im moderaten Bereich (r_x in [0.86, 2.05],
r_y in [0.90, 2.29]) - keine Saettigung, kein Sprung.

R²(Block-Kreuzvalidierung, 5x5 Bloecke, NUR im Streifen) = 0.995 +/- 0.003
- eine sehr deutliche Verbesserung gegenueber der vorherigen Version dieser
  Datei (R²=0.487+/-0.344, gefittet auf "alles ausser Ridge+Ecke", was noch
  den Bereich JENSEITS der Ridge mit einschloss - dort verhaelt sich r_x
  anders, was die Formel destabilisiert hat).

WICHTIG - Gueltigkeitsbereich (NICHT gueltig ausserhalb!):
- NUR innerhalb des zentralen Diagonalstreifens, NICHT im gesamten
  gescannten Rechteck (0.8-1.7mm / 0.2-0.4MHz)! Ausserhalb des Streifens
  (Ridge, Ecken, Artefakt) bitte gpr_amp_predict.py (GPR, gilt ueberall)
  verwenden.
- is_in_stripe() unten prueft das EXAKT anhand der echten Streifen-Maske
  (naechster Gitterpunkt im 151x151-Scan, aus stripe_domain_mask.npz),
  nicht ueber eine grobe rechteckige Naeherung.
"""
import numpy as np
import os

_TERMS = [
    (0, 0, 13.19960021), (0, 1, -26.52166346), (1, 0, -7.64899657),
    (0, 2, 5.50953973), (1, 1, -22.47693845), (2, 0, 1.41301774),
    (0, 3, 25.53336739), (1, 2, 25.30996174), (2, 1, 4.79698803),
    (3, 0, 0.52721909), (0, 4, 8.42163392), (1, 3, 40.58885383),
    (2, 2, 21.14883378), (3, 1, 0.22480496), (4, 0, -0.14410667),
    (0, 5, -19.03739834), (1, 4, -99.14631262), (2, 3, 1.80278648),
    (3, 2, 3.20339562), (4, 1, -0.83363079), (5, 0, 0.00058047),
]

_DOMAIN = dict(waist_mm=(0.8, 1.7), width_MHz=(0.2, 0.4))
_NPZ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stripe_domain_mask.npz")


def predict_rx(waist_mm, width_MHz):
    """r_x(waist_mm, width_MHz) - vektorisiert (float oder numpy-Array).
    Nur gueltig innerhalb des zentralen Diagonalstreifens - siehe is_in_stripe()."""
    x = np.atleast_1d(np.asarray(waist_mm, dtype=float))
    y = np.atleast_1d(np.asarray(width_MHz, dtype=float))
    log_r = np.zeros_like(x)
    for i, j, c in _TERMS:
        log_r = log_r + c * (x**i) * (y**j)
    r_x = np.exp(log_r)
    return r_x if r_x.size > 1 else float(r_x[0])


_stripe_cache = None


def _load_stripe():
    global _stripe_cache
    if _stripe_cache is None:
        data = np.load(_NPZ_PATH)
        _stripe_cache = (data["waist_mm"], data["width_MHz"], data["stripe_mask"])
    return _stripe_cache


def is_in_stripe(waist_mm, width_MHz):
    """Exakte Pruefung: liegt der Punkt (naechster Gitterpunkt im Original-
    151x151-Scan) innerhalb des zentralen Diagonalstreifens? Ausserhalb davon
    (Ridge, Saettigungs-Ecken, Artefakt oben rechts) ist predict_rx() NICHT
    gueltig - dort bitte gpr_amp_predict.py verwenden."""
    waist_grid, width_grid, mask = _load_stripe()
    x = np.atleast_1d(np.asarray(waist_mm, dtype=float))
    y = np.atleast_1d(np.asarray(width_MHz, dtype=float))
    wlo, whi = _DOMAIN["waist_mm"]
    dlo, dhi = _DOMAIN["width_MHz"]
    in_box = (x >= wlo) & (x <= whi) & (y >= dlo) & (y <= dhi)
    ix = np.clip(np.round((x - waist_grid[0]) / (waist_grid[1] - waist_grid[0])).astype(int), 0, len(waist_grid) - 1)
    iy = np.clip(np.round((y - width_grid[0]) / (width_grid[1] - width_grid[0])).astype(int), 0, len(width_grid) - 1)
    ok = in_box & mask[iy, ix]
    return ok if ok.size > 1 else bool(ok[0])


if __name__ == "__main__":
    print("r_x(1.2mm, 0.30MHz) =", predict_rx(1.2, 0.30),
          "  im Streifen?", is_in_stripe(1.2, 0.30))
    print("r_x(0.85mm, 0.21MHz) [Saettigungs-Ecke!] =", predict_rx(0.85, 0.21),
          "  im Streifen?", is_in_stripe(0.85, 0.21))
    print("r_x(1.6mm, 0.38MHz) [Artefakt-Ecke!] =", predict_rx(1.6, 0.38),
          "  im Streifen?", is_in_stripe(1.6, 0.38))
