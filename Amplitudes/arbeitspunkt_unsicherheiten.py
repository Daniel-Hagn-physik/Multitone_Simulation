"""Werte und numerische Unsicherheiten am Arbeitspunkt (3x4, Airy, k = 1.483).

Liest die drei 41x41-Scans (combined / hard / weighted) und wertet sie am
jeweiligen Arbeitspunkt aus:
  - linear interpoliert in der Waist-Richtung (width liegt exakt auf der
    Gitterzeile 0.370 MHz, also identisch zur bilinearen Interpolation
    von run_plots.py),
  - Interpolationsfehler = |kubisch (4 Nachbarn) - linear|,
  - Scan-Rauschen = RMS-Residuum eines lokalen 2D-Quadratfits
    (5 width-Zeilen x 6 Waist-Spalten um den Punkt, 6 Parameter),
  - Sensitivitaeten dX/dWaist und dX/dwidth aus demselben Fit,
  - Lage des Arbeitspunkts aus dem Geradenfit (korrelierte Fehler von a, b).
Aufruf aus dem Ordner Amplitudes:  python arbeitspunkt_unsicherheiten.py
"""
import json, pickle
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent
FILES = {
    "combined": "Combinated_Optimization/Results/1.483_w_interference/scan_amp_data_combined_N3x4_41x41pts_Airy_interfereence.pkl",
    "hard":     "Hard_Optimization/Results/1.483_w_interference/scan_amp_data_N3x4_41x41pts_Airy_k1.483_hard_interference.pkl",
    "weighted": "Weighted_Optimization/Results/1.483_w_interference/scan_amp_data_weighted_N3x4_41x41pts_Airy_k1.483_interference.pkl",
}
WIDTH_MHZ = 0.37
# Arbeitspunkte (Waist in um): combined aus der Talgeraden vom 18.09.,
# hard aus der Talgeraden vom 09.09. (Thesis_HardScan), weighted am combined-Punkt
POINTS = {"combined": [1.0437, 1.04], "hard": [1.0131, 1.04], "weighted": [1.0437, 1.0131]}
# Geradenfits (a, b, sigma_a, fit-Bereich waist) fuer die Lage des Punkts
LINES = {
    "combined 18.09": (0.3142, 0.0420, 0.0054, 0.7872, 1.2775),
    "combined 09.09": (0.3264, 0.0309, 0.0065, 0.7872, 1.1764),
    "hard 18.09":     (0.3454, 0.0191, 0.0079, 0.7872, 1.2506),
    "hard 09.09":     (0.3560, 0.0094, 0.0082, 0.7872, 1.1764),
}


def grids(name, d):
    if name == "combined":
        g = {"U_h": d["uniformity_grid"], "U_w": d["uniformity_weighted_grid"],
             "eta_h": d["crosstalk_grid"], "eta_w": d["eta_weighted_grid"],
             "U_c": d["uniformity_kombi"], "eta_c": d["crosstalk_kombi"],
             "J": d["combined_score"]}
    elif name == "hard":
        g = {"U_h": d["uniformity_grid"], "eta_h": d["crosstalk_grid"]}
        g["J"] = 0.7 * g["U_h"] + 0.3 * g["eta_h"]
    else:
        g = {"U_w": d["uniformity_weighted_grid"], "eta_w": d["eta_weighted_grid"]}
        g["J"] = 0.7 * g["U_w"] + 0.3 * g["eta_w"]
    g = {k: v * 100.0 for k, v in g.items()}          # Prozent
    g["r_x"], g["r_y"] = d["r_x_grid"], d["r_y_grid"]
    return g


def waist_axis(d):
    return (d["f1"] / d["f2"]) * d["lambda_opt"] * d["fLO"] / (np.pi * d["win_input_vals"]) * 1e6


def evaluate(name, W0):
    d = pickle.load(open(BASE / FILES[name], "rb"))
    w, wd = waist_axis(d), d["width_vals"] / 1e6
    i = int(np.argmin(abs(wd - WIDTH_MHZ)))
    j = int(np.where(w >= W0)[0].max())                 # w faellt mit j
    t = (W0 - w[j]) / (w[j + 1] - w[j])
    ii, jj = np.arange(i - 2, i + 3), np.arange(j - 2, j + 4)
    WW, FF = np.meshgrid(w[jj], wd[ii])
    x, y = (WW - W0).ravel(), ((FF - WIDTH_MHZ) * 1e3).ravel()
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    out = {"waist_grid": [float(w[j + 1]), float(w[j])], "q": {}}
    for k, Z in grids(name, d).items():
        lin = Z[i, j] * (1 - t) + Z[i, j + 1] * t
        cub = np.polyval(np.polyfit(w[j - 1:j + 3], Z[i, j - 1:j + 3], 3), W0)
        zz = Z[np.ix_(ii, jj)].ravel(); m = np.isfinite(zz)
        c, *_ = np.linalg.lstsq(A[m], zz[m], rcond=None)
        res = zz[m] - A[m] @ c
        noise = float(np.sqrt((res ** 2).sum() / (m.sum() - 6)))
        out["q"][k] = dict(lin=float(lin), near_lo=float(Z[i, j + 1]), near_hi=float(Z[i, j]),
                           d_interp=float(abs(cub - lin)), noise=noise,
                           sigma=float(np.hypot(cub - lin, noise)),
                           per10nm=float(c[1] * 0.01), per5kHz=float(c[2] * 5))
    return out


def line_position(a, b, sa, lo, hi, w_axis):
    t = np.sort(w_axis[(w_axis >= lo - 1e-4) & (w_axis <= hi + 1e-4)])
    n, tb = len(t), t.mean(); Stt = ((t - tb) ** 2).sum(); s2 = sa ** 2 * Stt
    w0 = (WIDTH_MHZ - b) / a
    sy = np.sqrt(s2 * (1 / n + (w0 - tb) ** 2 / Stt))
    return dict(n=n, waist=float(w0), sigma_nm=float(sy / a * 1e3))


if __name__ == "__main__":
    R = {n: {f"{W:.4f}": evaluate(n, W) for W in Ws} for n, Ws in POINTS.items()}
    d = pickle.load(open(BASE / FILES["hard"], "rb")); wax = waist_axis(d)
    R["lines"] = {k: line_position(*v, wax) for k, v in LINES.items()}
    json.dump(R, open(BASE / "arbeitspunkt_unsicherheiten.json", "w"), indent=1)
    for n in POINTS:
        for W, r in R[n].items():
            print(f"## {n} @ {W} um / {WIDTH_MHZ} MHz  (Gitter {r['waist_grid']})")
            for k, q in r["q"].items():
                print(f"  {k:6s} {q['lin']:.5g} +- {q['sigma']:.2g}  (interp {q['d_interp']:.2g}, "
                      f"rausch {q['noise']:.2g})  Gitter {q['near_lo']:.5g}/{q['near_hi']:.5g}  "
                      f"d/10nm {q['per10nm']:+.3g}  d/5kHz {q['per5kHz']:+.3g}")
    for k, v in R["lines"].items():
        print(k, v)
