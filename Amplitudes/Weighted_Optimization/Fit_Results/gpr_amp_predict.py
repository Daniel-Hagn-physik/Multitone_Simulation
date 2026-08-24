"""
gpr_amp_predict.py
===================
Eigenständiger, sklearn-unabhaengiger Predictor fuer die gefitteten
r_x(waist,width)- und r_y(waist,width)-Gaussian-Process-Modelle aus
AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24.

Warum nicht einfach das sklearn-Pickle laden? Ein sklearn-Pickle ist an die
exakte sklearn-Version gebunden, mit der es erzeugt wurde, und laedt in einer
neuen/anderen Umgebung u.U. gar nicht mehr. Dieses Skript reimplementiert die
GP-Posterior-Mean-Formel von Hand (Matern-3/2-Kernel) anhand der exportierten
rohen Zahlen (gpr_amp_export.npz: Trainingspunkte, alpha-Koeffizienten,
Kernel-Hyperparameter) - funktioniert mit jeder halbwegs aktuellen
numpy-Installation, keine sklearn-Abhaengigkeit.

Formel (Standard-GP-Regression, siehe z.B. Rasmussen & Williams Gl. 2.23-2.24):
    mu(x) = k(x, X_train) @ alpha * y_train_std + y_train_mean
mit k(x, x') = sigma_f^2 * Matern_3/2(r), r = |x_scaled - x'_scaled| (pro Achse
mit eigener length_scale reskaliert), alpha bereits (K + sigma_n^2 I)^-1 (y-mean)
(steckt in gp.alpha_ aus dem sklearn-Fit, hier direkt uebernommen). Die
WhiteKernel-Rauschkomponente traegt an neuen Testpunkten NICHT zur
Kovarianz mit den Trainingspunkten bei (nur auf der Diagonale w"ahrend des
Trainings) - deshalb hier nicht Teil von k(x, X_train).

Nutzung:
    import numpy as np
    from gpr_amp_predict import load_predictors
    predict_rx, predict_ry = load_predictors("gpr_amp_export.npz")
    r_x = predict_rx(waist_mm=1.2, width_MHz=0.31)
    r_y = predict_ry(waist_mm=1.2, width_MHz=0.31)
    # vektorisiert:
    waists = np.linspace(0.8, 1.7, 50)
    widths = np.full_like(waists, 0.3)
    r_x_vals = predict_rx(waists, widths)
"""
import numpy as np


def _matern32(d):
    """Matern-Kernel mit nu=3/2, d = euklidischer Abstand in reskalierten (dimensionslosen) Koordinaten."""
    sqrt3_d = np.sqrt(3.0) * d
    return (1.0 + sqrt3_d) * np.exp(-sqrt3_d)


def _make_predictor(p):
    X_train_scaled = p["X_train_scaled"]        # (n,2), bereits standardisiert
    alpha = p["alpha"]                           # (n,)
    y_mean = float(p["y_train_mean"])
    y_std = float(p["y_train_std"])
    scaler_mean = p["scaler_mean"]                # (2,) [waist_mm, width_MHz]
    scaler_scale = p["scaler_scale"]               # (2,)
    sigma_f2 = float(p["sigma_f2"])
    length_scale = p["length_scale"]              # (2,) in standardisierten Einheiten

    def predict(waist_mm, width_MHz):
        waist_mm = np.atleast_1d(np.asarray(waist_mm, dtype=float))
        width_MHz = np.atleast_1d(np.asarray(width_MHz, dtype=float))
        X = np.column_stack([waist_mm, width_MHz])
        X_scaled = (X - scaler_mean) / scaler_scale
        # anisotrope Distanz: pro Achse durch length_scale teilen, dann euklidisch
        diff = (X_scaled[:, None, :] - X_train_scaled[None, :, :]) / length_scale
        d = np.sqrt(np.sum(diff**2, axis=-1))      # (n_test, n_train)
        K_trans = sigma_f2 * _matern32(d)
        log_r = K_trans @ alpha * y_std + y_mean
        r = np.exp(log_r)                          # Modell wurde auf log(r) trainiert
        return r if r.size > 1 else float(r[0])

    return predict


def load_predictors(npz_path):
    data = np.load(npz_path)
    keys = ("X_train_scaled", "alpha", "y_train_mean", "y_train_std",
            "scaler_mean", "scaler_scale", "sigma_f2", "length_scale")
    p_rx = {k: data[f"r_x__{k}"] for k in keys}
    p_ry = {k: data[f"r_y__{k}"] for k in keys}
    return _make_predictor(p_rx), _make_predictor(p_ry)


if __name__ == "__main__":
    predict_rx, predict_ry = load_predictors("gpr_amp_export.npz")
    print("r_x(1.2mm, 0.30MHz) =", predict_rx(1.2, 0.30))
    print("r_y(1.2mm, 0.30MHz) =", predict_ry(1.2, 0.30))
