"""
ry_smooth_formula.py
=====================
Geschlossene Formel (Polynom 4. Grades) fuer r_y(waist_mm, width_MHz),
gefittet NUR auf dem "zentralen Diagonalstreifen" des gewichteten
151x151-Amplituden-Scans (scan_amp_data_weighted_N3x4_151x151pts_Airy_2500res.pkl).

Siehe rx_smooth_formula.py fuer die ausfuehrliche Erklaerung, was der
"zentrale Diagonalstreifen" ist (dieselbe Streifen-Maske wird fuer r_x UND
r_y verwendet, da Ridge und Artefakt fuer beide an denselben Stellen liegen).

R²(Block-Kreuzvalidierung, 5x5 Bloecke, NUR im Streifen) = 0.996 +/- 0.002
- eine Verbesserung gegenueber der vorherigen Version dieser Datei
  (R²=0.865+/-0.059, gefittet auf "alles ausser Ridge", was noch den
  Saettigungsbereich JENSEITS der Ridge mit einschloss).

WICHTIG - Gueltigkeitsbereich (NICHT gueltig ausserhalb!):
- NUR innerhalb des zentralen Diagonalstreifens, NICHT im gesamten
  gescannten Rechteck (0.8-1.7mm / 0.2-0.4MHz)! Ausserhalb des Streifens
  (Ridge, Saettigungs-Ecken, Artefakt oben rechts) bitte gpr_amp_predict.py
  (GPR, gilt ueberall) verwenden.
- is_in_stripe() unten prueft das EXAKT anhand der echten Streifen-Maske
  (naechster Gitterpunkt im 151x151-Scan, aus stripe_domain_mask.npz).
"""
import numpy as np
import os

_TERMS = [
    (0, 0, 16.94018934), (0, 1, -42.93798106), (1, 0, -10.54570753),
    (0, 2, 23.68534994), (1, 1, -18.03411162), (2, 0, 1.92980593),
    (0, 3, 59.97822032), (1, 2, 59.39648006), (2, 1, 7.68116744),
    (3, 0, 0.94705135), (0, 4, -65.14632532), (1, 3, -68.82414564),
    (2, 2, 13.15875129), (3, 1, -2.26070182), (4, 0, -0.32096537),
]

_DOMAIN = dict(waist_mm=(0.8, 1.7), width_MHz=(0.2, 0.4))
_NPZ_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stripe_domain_mask.npz")


def predict_ry(waist_mm, width_MHz):
    """r_y(waist_mm, width_MHz) - vektorisiert (float oder numpy-Array).
    Nur gueltig innerhalb des zentralen Diagonalstreifens - siehe is_in_stripe()."""
    x = np.atleast_1d(np.asarray(waist_mm, dtype=float))
    y = np.atleast_1d(np.asarray(width_MHz, dtype=float))
    log_r = np.zeros_like(x)
    for i, j, c in _TERMS:
        log_r = log_r + c * (x**i) * (y**j)
    r_y = np.exp(log_r)
    return r_y if r_y.size > 1 else float(r_y[0])


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
    (Ridge, Saettigungs-Ecken, Artefakt oben rechts) ist predict_ry() NICHT
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
    print("r_y(1.2mm, 0.30MHz) =", predict_ry(1.2, 0.30),
          "  im Streifen?", is_in_stripe(1.2, 0.30))
    print("r_y(0.85mm, 0.21MHz) [Saettigungs-Ecke!] =", predict_ry(0.85, 0.21),
          "  im Streifen?", is_in_stripe(0.85, 0.21))
    print("r_y(1.6mm, 0.38MHz) [Artefakt-Ecke!] =", predict_ry(1.6, 0.38),
          "  im Streifen?", is_in_stripe(1.6, 0.38))
