"""
Weighted_Optimization/beispiel_weighted_amp_fit_abhaengigkeiten.py
=====================================================================

Port von beispiel_amp_fit_abhaengigkeiten.py (Original-Ordner) für den
ATOM-GEWICHTETEN Optimizer. Gleiche ZWEI Fits wie im Original, aber
gespeist aus den GEWICHTETEN Scan-Varianten (uniformity_weighted/
eta_weighted statt uniformity/crosstalk - siehe weighted_amp_scan_methods.py
und den Modul-Docstring von weighted_multitone_amplitude_dependence_plots.py
für den physikalischen Hintergrund):

1. Amplituden-Abhängigkeit r_x(waist, width), r_y(waist, width): aus dem
   GEWICHTETEN Amplituden-Scan (scan_amp_data_weighted_...pkl, erzeugt von
   scan_win_width_amplitude_dependence_weighted() in
   weighted_amp_scan_methods.py). An jedem Gitterpunkt wurden dort r_x/r_y
   so optimiert, dass sie das ATOM-GEWICHTETE Ziel alpha*uniformity_weighted
   + (1-alpha)*eta_weighted minimieren (statt der harten Masken-Metriken im
   Original) - diese r_x/r_y-Werte werden hier gegen waist/width gefittet,
   mit denselben DREI Modellen wie im Original zum Vergleich (Polynom 2.
   Grades / physikalisches Exponentialmodell / glatte 2D-Spline-
   Interpolation, siehe _fit_interpolation()). WICHTIG: r_x_opt_weighted
   kann von r_x_opt (Original, hartes Ziel) abweichen, weil die gewichtete
   Metrik die tatsächliche thermische Ausdehnung des Atoms berücksichtigt
   statt einer harten Kastenmaske - genau DAS ist die Antwort auf die
   Frage, ob/wie sich die "richtige" Amplituden-Wahl ändert, wenn man die
   Atomphysik statt einer geometrischen Näherung zugrunde legt.

2. Linearer Zusammenhang zwischen effektivem Waist (µm) und width (MHz):
   wie im Original aus einem FEST-AMPLITUDE-Scan, hier aber aus dem
   GEWICHTETEN Fest-Amplitude-Scan (scan_data_weighted_...pkl, erzeugt von
   scan_win_width_weighted_uniformity()) - das Uniformity-Tal wird also in
   uniformity_weighted statt uniformity gesucht. Identischer Kink-Ausschluss
   (_fit_line_dropping_kinks(), unverändert übernommen) wie im Original.

Beide Teile geben wiederverwendbare Fit-Funktionen zurück - siehe main()
bzw. direkt fit_amplitude_dependence()/fit_waist_width_relation():

    from beispiel_weighted_amp_fit_abhaengigkeiten import (
        load_amp_scan_results, fit_amplitude_dependence, fit_waist_width_relation,
    )
    amp_results = load_amp_scan_results("scan_amp_data_weighted_....pkl")
    amp_fits = fit_amplitude_dependence(amp_results)
    r_x_pred = amp_fits["r_x"]["poly"]["predict"](win_mm, width_mhz)

ACHTUNG bei eigenständiger Verwendung von fit_amplitude_dependence(): wie im
Original bereinigt diese Funktion selbst KEINE Ausreißer - vorher ggf.
detect_amp_outliers()/clean_amp_scan_results() aufrufen (wie in main()).

Ausgabe: Konsole (Koeffizienten, R², Fit-Formeln) UND, sofern PLOT_FITS=True
(Default), je ein Diagnose-Plot pro Fit, gespeichert in "Fit_Plots" NEBEN
DIESEM Skript (also Weighted_Optimization/Fit_Plots - eigener Ordner,
getrennt von Weighted_Optimization/Bilder, exakt wie im Original).

WICHTIG - richtige Dateien auswählen:
  - AMP_PKL_DATEI  -> eine "scan_amp_data_weighted_...pkl"-Datei (mit
    r_x_grid/r_y_grid, optimiert unter dem GEWICHTETEN Ziel - erzeugt von
    opt.scan_win_width_amplitude_dependence_weighted() +
    opt.save_scan_amp_results_weighted())
  - FIXED_PKL_DATEI -> eine "scan_data_weighted_...pkl"-Datei (mit
    uniformity_weighted_grid/eta_weighted_grid bei FESTEN Amplituden -
    erzeugt von opt.scan_win_width_weighted_uniformity() +
    opt.save_scan_weighted_results())
Beide Dateien werden unabhängig voneinander geladen; fehlt eine, wird nur
der jeweils andere Teil übersprungen.

Nutzung:
    python beispiel_weighted_amp_fit_abhaengigkeiten.py
Ggf. vorher AMP_PKL_DATEI / FIXED_PKL_DATEI unten anpassen. Falls die pkl-
Dateien noch nicht existieren: siehe weighted_amp_scan_methods.py, dort
steht ein vollständiges Beispiel, wie man sie mit dem gewichteten Optimizer
erzeugt.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline

from weighted_multitone_amplitude_dependence_plots import (
    load_amp_scan_results,
    win_input_to_win,
    summarize_amp_bounds,
    detect_amp_outliers,
    clean_amp_scan_results,
    resolve_save_path,
    AmplitudeScanPlotter,
    DEFAULT_RESULTS_DIR,
)


def _default_dir(name):
    """Wie weighted_multitone_amplitude_dependence_plots._default_dir(): legt
    einen Ordner neben DIESEM Skript an (bei Bedarf automatisch)."""
    candidate = Path(__file__).resolve().parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = Path(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# Eigener Ordner für die Diagnose-Plots dieses Skripts - getrennt vom
# "Bilder"-Ordner (identisches Muster wie im Original).
FIT_PLOTS_DIR = _default_dir("Fit_Plots")


# ======================================================================
# Konfiguration - hier anpassen
# ======================================================================

# GEWICHTETER Amplituden-Scan (mit r_x_grid/r_y_grid, optimiert unter dem
# atom-gewichteten Ziel) für Fit 1 - siehe
# scan_win_width_amplitude_dependence_weighted() in weighted_amp_scan_methods.py.
AMP_PKL_DATEI = r"scan_amp_data_weighted_N3x4_10x10pts_Airy_400res_Ausreißer.pkl"

# GEWICHTETER Fest-Amplitude-Scan (mit uniformity_weighted_grid/
# eta_weighted_grid, feste Amplituden) für Fit 2 - siehe
# scan_win_width_weighted_uniformity() in weighted_amp_scan_methods.py.
FIXED_PKL_DATEI = r"C:\Users\Legion\PycharmProjects\Lern-repo\Optimierung Niklas+Claude\Weighted_Optimization\Results\scan_data_weighted_N3x4_31x31pts_Airy.pkl"

# "before_lens" (mm vor der ersten Linse) oder "after_lens" (µm effektiver
# Waist an der Fokusebene) - nur die ANZEIGE-/Fit-Achse für den POLYNOM-Fit
# von r_x/r_y (Teil 1), identisch zum Original.
WIN_AXIS = "before_lens"

# Ausreißer (r_x/r_y an r_bounds-Schranke) VOR dem Fit aus dem Amplituden-
# Scan entfernen? Identisch zum Original.
AMP_CLEAN_OUTLIERS = True
AMP_CLEAN_STRATEGY = "interpolate"
AMP_CLEAN_BOUNDS = ("lower",)

# Diagnose-Plots erzeugen?
PLOT_FITS = True

SHOW = True
SAVE = True
ASK_BEFORE_SAVE = True


# ======================================================================
# Kleine Hilfsfunktionen
# ======================================================================
def _load_pkl(pkl_datei, expect_amp):
    """Wie im Original, aber sucht nach den GEWICHTETEN Dateinamen-Mustern
    (scan_amp_data_weighted_*.pkl / scan_data_weighted_*.pkl)."""
    try:
        return load_amp_scan_results(pkl_datei)
    except FileNotFoundError:
        pattern = "scan_amp_data_weighted_*.pkl" if expect_amp else "scan_data_weighted_*.pkl"
        vorhandene = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob(pattern))
        print(f"'{pkl_datei}' wurde weder im aktuellen Ordner noch in "
              f"'{DEFAULT_RESULTS_DIR}' gefunden.")
        if vorhandene:
            print(f"Vorhandene Dateien ({pattern}) in Results:")
            for name in vorhandene:
                print(f"  - {name}")
        else:
            print(f"Keine passenden Dateien ({pattern}) in '{DEFAULT_RESULTS_DIR}' gefunden. "
                  f"Siehe weighted_amp_scan_methods.py für ein Beispiel, wie man sie erzeugt.")
        return None


def _win_axis_values(win_input_vals, win_axis, f1, f2, lambda_opt, fLO):
    if win_axis == "before_lens":
        return win_input_vals * 1e3, "waist before lens (mm)"
    elif win_axis == "after_lens":
        x = np.array([win_input_to_win(w, f1, f2, lambda_opt, fLO) for w in win_input_vals]) * 1e6
        return x, r"effective waist at focal plane ($\mu$m)"
    else:
        raise ValueError("win_axis muss 'before_lens' oder 'after_lens' sein.")


def _waist_eff_um(win_input_vals, f1, f2, lambda_opt, fLO):
    return np.array([win_input_to_win(w, f1, f2, lambda_opt, fLO) for w in win_input_vals]) * 1e6


def _win_axis_for_plot(win_input_vals, win_axis, f1, f2, lambda_opt, fLO):
    x_vals, x_label = _win_axis_values(win_input_vals, win_axis, f1, f2, lambda_opt, fLO)
    reversed_ = len(x_vals) > 1 and x_vals[0] > x_vals[-1]
    if reversed_:
        x_vals = x_vals[::-1]
    return x_vals, x_label, reversed_


def _finish_fig(fig, filename, out_dir, show, save, confirm_overwrite=None):
    out_dir = Path(out_dir) if out_dir is not None else FIT_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if save:
        out_file = resolve_save_path(out_dir, filename, confirm_overwrite=confirm_overwrite)
        fig.savefig(out_file, dpi=AmplitudeScanPlotter.SCAN2D_SAVE_DPI, bbox_inches="tight")
        print(f"Figure saved: {out_file}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _r2(y, y_fit):
    y = np.asarray(y, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ======================================================================
# Teil 1: r_x(waist, width) / r_y(waist, width) aus dem GEWICHTETEN
# Amplituden-Scan - Fit-Logik selbst 1:1 UNVERÄNDERT vom Original (r_x_grid/
# r_y_grid heißen identisch, egal unter welchem Ziel sie optimiert wurden).
# ======================================================================
def _fit_poly2(x, y, z):
    """z ~ c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*x*y (kleinste Quadrate)."""
    A = np.column_stack([np.ones_like(x), x, y, x ** 2, y ** 2, x * y])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    z_fit = A @ coeffs

    def predict(x_new, y_new):
        x_new = np.asarray(x_new, dtype=float)
        y_new = np.asarray(y_new, dtype=float)
        c0, c1, c2, c3, c4, c5 = coeffs
        return c0 + c1 * x_new + c2 * y_new + c3 * x_new ** 2 + c4 * y_new ** 2 + c5 * x_new * y_new

    return dict(
        coeffs=coeffs, r2=_r2(z, z_fit), predict=predict,
        formula="r = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*x*y",
    )


def _fit_physical(waist_um, width_mhz, z):
    """r = exp(a*(width/w_eff)^2 + b) - siehe Modul-Docstring des Originals
    für die physikalische Begründung. Gilt für r_x_opt_weighted genauso wie
    für r_x_opt (Original): der intensitätsabhängige Amplituden-Ausgleich
    ist unabhängig davon, ob man ihn über die harte Maske oder die
    atom-gewichtete Metrik bestimmt."""
    ratio = width_mhz / waist_um

    def model(ratio_, a, b):
        return np.exp(a * ratio_ ** 2 + b)

    try:
        popt, _ = curve_fit(model, ratio, z, p0=(1.0, 0.0), maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    z_fit = model(ratio, *popt)

    def predict(waist_um_new, width_mhz_new):
        waist_um_new = np.asarray(waist_um_new, dtype=float)
        width_mhz_new = np.asarray(width_mhz_new, dtype=float)
        return model(width_mhz_new / waist_um_new, *popt)

    return dict(
        params=dict(a=popt[0], b=popt[1]), r2=_r2(z, z_fit), predict=predict,
        formula="r = exp(a*(width_MHz/waist_eff_um)^2 + b)",
    )


def _fit_interpolation(x_vals, width_mhz, Z):
    """Glatte 2D-Spline-Interpolation - siehe Modul-Docstring des Originals
    (unverändert übernommen, RectBivariateSpline mit s=0)."""
    x_vals = np.asarray(x_vals, dtype=float)
    width_mhz = np.asarray(width_mhz, dtype=float)

    order_x = np.argsort(x_vals)
    x_sorted = x_vals[order_x]
    Z_sorted = Z[:, order_x]
    order_y = np.argsort(width_mhz)
    y_sorted = width_mhz[order_y]
    Z_sorted = Z_sorted[order_y, :]

    x_unique, x_idx = np.unique(x_sorted, return_index=True)
    y_unique, y_idx = np.unique(y_sorted, return_index=True)
    Z_sorted = Z_sorted[np.ix_(y_idx, x_idx)]

    kx = min(3, len(x_unique) - 1)
    ky = min(3, len(y_unique) - 1)
    if kx < 1 or ky < 1 or not np.all(np.isfinite(Z_sorted)):
        return None
    try:
        spline = RectBivariateSpline(y_unique, x_unique, Z_sorted, kx=kx, ky=ky, s=0)
    except Exception:
        return None

    x_lo, x_hi = x_unique[0], x_unique[-1]
    y_lo, y_hi = y_unique[0], y_unique[-1]

    def predict(x_new, y_new):
        x_new = np.asarray(x_new, dtype=float)
        y_new = np.asarray(y_new, dtype=float)
        shape = np.broadcast(x_new, y_new).shape
        x_b = np.clip(np.broadcast_to(x_new, shape), x_lo, x_hi).ravel()
        y_b = np.clip(np.broadcast_to(y_new, shape), y_lo, y_hi).ravel()
        result = spline.ev(y_b, x_b)
        return result.reshape(shape) if shape else float(result[0])

    z_fit = predict(x_vals[None, :] * np.ones_like(Z), width_mhz[:, None] * np.ones_like(Z))
    return dict(
        r2=_r2(Z, z_fit), predict=predict,
        formula="cubic 2D spline interpolation (exact through data, clipped outside scan range)",
    )


def fit_amplitude_dependence(results, win_axis=WIN_AXIS):
    """Wie im Original - fittet r_x(waist,width)/r_y(waist,width) mit DREI
    Modellen. Einzige Änderung: die r_x_grid/r_y_grid-Werte im übergebenen
    results-dict stammen hier vom GEWICHTETEN Amplituden-Scan (optimiert
    unter alpha*uniformity_weighted+(1-alpha)*eta_weighted statt der harten
    Masken-Metriken) - die Fit-Logik selbst braucht dafür keinerlei Anpassung."""
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]

    x_vals, x_label = _win_axis_values(win_input_vals, win_axis, f1, f2, lambda_opt, fLO)
    width_mhz = width_vals * 1e-6
    waist_um = _waist_eff_um(win_input_vals, f1, f2, lambda_opt, fLO)

    X, Y = np.meshgrid(x_vals, width_mhz)
    Waist_um_grid, Width_mhz_grid = np.meshgrid(waist_um, width_mhz)

    out = {}
    for key, grid_key in (("r_x", "r_x_grid"), ("r_y", "r_y_grid")):
        Z = results[grid_key]
        mask = np.isfinite(Z)
        n_valid = int(mask.sum())
        if n_valid < 6:
            print(f"  [{key}] zu wenige gültige Gitterpunkte ({n_valid}) für einen Fit - übersprungen.")
            out[key] = dict(poly=None, physical=None)
            continue

        poly_fit = _fit_poly2(X[mask], Y[mask], Z[mask])
        physical_fit = _fit_physical(Waist_um_grid[mask], Width_mhz_grid[mask], Z[mask])
        interp_fit = _fit_interpolation(x_vals, width_mhz, Z) if n_valid == Z.size else None
        out[key] = dict(poly=poly_fit, physical=physical_fit, interp=interp_fit)

        print(f"\n-- {key}_weighted (x-Achse: {x_label}) --")
        print(f"  Polynom 2. Grades:  R²={poly_fit['r2']:.4f}   [{poly_fit['formula']}]")
        print(f"    Koeffizienten (c0..c5): {np.array2string(poly_fit['coeffs'], precision=5, suppress_small=True)}")
        if physical_fit is not None:
            print(f"  Physikalisch (exp): R²={physical_fit['r2']:.4f}   [{physical_fit['formula']}]")
            print(f"    a={physical_fit['params']['a']:.5g}, b={physical_fit['params']['b']:.5g}")
        else:
            print("  Physikalisch (exp): Fit nicht konvergiert - übersprungen.")
        if interp_fit is not None:
            print(f"  Spline-Interpolation: R²={interp_fit['r2']:.4f}   [{interp_fit['formula']}]")
        else:
            print("  Spline-Interpolation: übersprungen (Gitter hat Lücken/NaN - vorher bereinigen, siehe AMP_CLEAN_OUTLIERS).")

        kandidaten = [("Polynom-Modell", poly_fit), ("physikalisches Modell", physical_fit),
                      ("Spline-Interpolation", interp_fit)]
        kandidaten = [(name, f) for name, f in kandidaten if f is not None]
        bester_name, bester_fit = max(kandidaten, key=lambda kv: kv[1]["r2"])
        print(f"    -> '{bester_name}' beschreibt {key}_weighted hier am besten (höchstes R²={bester_fit['r2']:.4f}).")
        if bester_name == "Spline-Interpolation":
            print("       (Hinweis: die Interpolation geht per Konstruktion fast exakt durch die "
                  "Datenpunkte, daher fast immer das höchste R² - dafür keine geschlossene Formel "
                  "und unzuverlässig außerhalb des gescannten Bereichs, siehe _fit_interpolation().)")

    return out


def _key_latex(key):
    """'r_x'/'r_y' -> LaTeX-kompatibles mathtext-Label ('$r_x$'/'$r_y$'),
    identisch zum Original - dass es sich um die unter dem GEWICHTETEN Ziel
    gefundenen r_x/r_y handelt, steht im Panel-/Figure-Titel (siehe
    plot_amplitude_fit()), nicht im Symbol selbst."""
    return rf"${key}$"


def plot_amplitude_fit(results, fits, key, win_axis=WIN_AXIS, out_dir=None,
                        show=True, save=True, confirm_overwrite=None):
    """Wie im Original, aber Titel/Dateiname markieren zusätzlich, dass die
    zugrundeliegenden r_x/r_y-Werte unter dem ATOM-GEWICHTETEN Ziel
    gefunden wurden."""
    grid_key = f"{key}_grid"
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]

    x_vals, x_label, reversed_ = _win_axis_for_plot(win_input_vals, win_axis, f1, f2, lambda_opt, fLO)
    width_mhz = width_vals * 1e-6
    waist_um = _waist_eff_um(win_input_vals, f1, f2, lambda_opt, fLO)
    if reversed_:
        waist_um = waist_um[::-1]

    Z_plot = results[grid_key][:, ::-1] if reversed_ else results[grid_key]

    X_disp, Y_disp = np.meshgrid(x_vals, width_mhz)
    Waist_um_grid, Width_mhz_grid = np.meshgrid(waist_um, width_mhz)

    poly = fits.get(key, {}).get("poly")
    physical = fits.get(key, {}).get("physical")
    interp = fits.get(key, {}).get("interp")
    key_tex = _key_latex(key)

    panels = [("Data (atom-weighted optimum)", Z_plot)]
    if poly is not None:
        panels.append((rf"Polynomial fit ($R^2={poly['r2']:.3f}$)", poly["predict"](X_disp, Y_disp)))
    if physical is not None:
        panels.append((rf"Physical model ($R^2={physical['r2']:.3f}$)",
                        physical["predict"](Waist_um_grid, Width_mhz_grid)))
    if interp is not None:
        panels.append((rf"Spline interpolation ($R^2={interp['r2']:.3f}$)",
                        interp["predict"](X_disp, Y_disp)))

    if len(panels) == 1:
        print(f"  [{key}] keine Fits verfügbar - Plot wird übersprungen.")
        return None

    # Wie im Original: nur poly/physical als Kandidaten für die Residuen -
    # die Interpolation geht per Konstruktion praktisch exakt durch jeden
    # Punkt (Residuum ~0 überall, ergäbe eine leere/weiße Fläche).
    candidates = []
    if poly is not None:
        candidates.append(("polynomial model", poly["r2"], poly["predict"](X_disp, Y_disp)))
    if physical is not None:
        candidates.append(("physical model", physical["r2"], physical["predict"](Waist_um_grid, Width_mhz_grid)))
    if candidates:
        best_name, best_r2, best_pred = max(candidates, key=lambda c: c[1])
        residual = Z_plot - best_pred
    else:
        best_name, residual = None, None

    vmin = min(np.nanmin(p[1]) for p in panels)
    vmax = max(np.nanmax(p[1]) for p in panels)

    n_panels = len(panels) + (1 if residual is not None else 0)
    with plt.rc_context(AmplitudeScanPlotter.SCAN2D_RC):
        fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (title, Zp) in zip(axes[:len(panels)], panels):
            im = ax.pcolormesh(x_vals, width_mhz, Zp, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel("width (MHz)")
        fig.colorbar(im, ax=list(axes[:len(panels)]), label=f"{key_tex} (outer/inner, atom-weighted optimum)",
                     shrink=0.85)

        if residual is not None:
            ax = axes[-1]
            r_max = np.nanmax(np.abs(residual)) or 1.0
            im_r = ax.pcolormesh(x_vals, width_mhz, residual, shading="auto", cmap="RdBu_r",
                                  vmin=-r_max, vmax=r_max)
            ax.set_title(f"Residuals (data $-$ {best_name})")
            ax.set_xlabel(x_label)
            ax.set_ylabel("width (MHz)")
            fig.colorbar(im_r, ax=ax, label=rf"$\Delta$ {key_tex}", shrink=0.85)

        fig.suptitle(rf"Fit diagnostics (atom-weighted): {key_tex}(waist, width)")

    n_win, n_width = len(win_input_vals), len(width_vals)
    filename = f"FlatMultiTone_AmpFitWeighted_{key}_N{results.get('N_x', '?')}x{results.get('N_y', '?')}_{n_win}x{n_width}pts.png"
    _finish_fig(fig, filename, out_dir, show, save, confirm_overwrite)
    return fig


# ======================================================================
# Teil 2: linearer Zusammenhang Waist(µm) <-> Width(MHz) aus dem
# GEWICHTETEN Fest-Amplitude-Scan (Uniformity_w-Tal statt Uniformity-Tal)
# ======================================================================
def extract_uniformity_valley(results):
    """Wie im Original, aber sucht das Minimum in uniformity_weighted_grid
    statt uniformity_grid - das Uniformity_w-Tal ist das gewichtete
    Analogon der geometrischen waist-width-Kopplung."""
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    Z = results["uniformity_weighted_grid"]                # shape (n_width, n_win)
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]

    waist_um = _waist_eff_um(win_input_vals, f1, f2, lambda_opt, fLO)
    width_mhz = width_vals * 1e-6

    valley_waist, valley_width = [], []
    for j in range(len(win_input_vals)):
        col = Z[:, j]
        if not np.any(np.isfinite(col)):
            continue
        i_min = int(np.nanargmin(col))
        width_at_min = width_mhz[i_min]
        if 0 < i_min < len(col) - 1 and np.isfinite(col[i_min - 1]) and np.isfinite(col[i_min + 1]):
            y0, y1, y2 = col[i_min - 1], col[i_min], col[i_min + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
                width_at_min = float(np.interp(i_min + delta, np.arange(len(width_mhz)), width_mhz))
        valley_waist.append(waist_um[j])
        valley_width.append(width_at_min)

    return np.array(valley_waist), np.array(valley_width)


def _fit_line_dropping_kinks(x, y, min_points=4, min_keep_frac=0.5, mad_factor=3.0):
    """UNVERÄNDERT vom Original übernommen - siehe dort für die volle
    Dokumentation der Kink-Ausschluss-Logik. Funktioniert identisch,
    unabhängig davon, ob die y-Werte aus dem Uniformity- oder dem
    Uniformity_w-Tal stammen."""
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    n = len(x_sorted)
    min_keep = max(min_points, int(np.ceil(min_keep_frac * n)))

    lo, hi = 0, n
    m, b = np.polyfit(x_sorted, y_sorted, 1)

    while hi - lo > min_keep:
        xs, ys = x_sorted[lo:hi], y_sorted[lo:hi]
        m, b = np.polyfit(xs, ys, 1)
        resid = ys - (m * xs + b)

        interior = resid[1:-1] if len(resid) > 2 else resid
        mad = np.median(np.abs(interior - np.median(interior)))
        scale = 1.4826 * mad
        floor = 0.02 * (np.max(y_sorted) - np.min(y_sorted) + 1e-12)
        threshold = mad_factor * max(scale, floor)

        left_r, right_r = abs(resid[0]), abs(resid[-1])
        if max(left_r, right_r) <= threshold:
            break
        if left_r >= right_r:
            lo += 1
        else:
            hi -= 1

    xs, ys = x_sorted[lo:hi], y_sorted[lo:hi]
    m, b = np.polyfit(xs, ys, 1)
    r2 = _r2(ys, m * xs + b)

    excl_mask = np.ones(n, dtype=bool)
    excl_mask[lo:hi] = False
    return m, b, r2, xs, ys, x_sorted[excl_mask], y_sorted[excl_mask]


def fit_waist_width_relation(results):
    """Wie im Original, aber die Talpunkte kommen aus dem GEWICHTETEN
    Uniformity_w-Tal (extract_uniformity_valley() oben) statt aus dem
    harten Uniformity-Tal."""
    waist_um, width_mhz = extract_uniformity_valley(results)
    if len(waist_um) < 2:
        print("  Zu wenige gültige Talpunkte für einen linearen Fit gefunden.")
        return None

    m, b, r2, waist_used, width_used, waist_excluded, width_excluded = \
        _fit_line_dropping_kinks(waist_um, width_mhz)

    def predict(waist_um_new):
        return m * np.asarray(waist_um_new, dtype=float) + b

    def predict_inv(width_mhz_new):
        return (np.asarray(width_mhz_new, dtype=float) - b) / m

    print(f"  width_MHz ≈ {m:.5g} * waist_um + {b:.5g}   (R²={r2:.4f}, "
          f"n={len(waist_used)} von {len(waist_um)} Uniformity_w-Talpunkten verwendet)")
    if len(waist_excluded):
        print(f"  {len(waist_excluded)} Punkt(e) am Rand wegen Abknicken vom Fit "
              f"ausgeschlossen (waist_um={np.array2string(waist_excluded, precision=1)}).")
    if len(waist_used) < 5:
        print("  Hinweis: sehr wenige Punkte (grobes Gitter) - Fit entsprechend unsicher; "
              "für ein belastbareres Ergebnis ein feineres Fest-Amplitude-Gitter scannen.")

    return dict(slope=m, intercept=b, r2=r2, predict=predict, predict_inv=predict_inv,
                waist_um=waist_used, width_mhz=width_used,
                waist_um_excluded=waist_excluded, width_mhz_excluded=width_excluded,
                waist_um_all=waist_um, width_mhz_all=width_mhz)


def plot_waist_width_fit(results, lin_fit, out_dir=None, show=True, save=True, confirm_overwrite=None):
    """Wie im Original, aber Heatmap/Titel zeigen Uniformity_w (atom-
    gewichtet) statt der harten Uniformity."""
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]

    waist_um = _waist_eff_um(win_input_vals, f1, f2, lambda_opt, fLO)
    width_mhz = width_vals * 1e-6
    reversed_ = len(waist_um) > 1 and waist_um[0] > waist_um[-1]
    x_vals = waist_um[::-1] if reversed_ else waist_um
    Z_plot = (results["uniformity_weighted_grid"][:, ::-1] if reversed_
              else results["uniformity_weighted_grid"]) * 100.0

    sigma_atom = results.get("sigma_atom")

    with plt.rc_context(AmplitudeScanPlotter.SCAN2D_RC):
        fig, ax = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
        im = ax.pcolormesh(x_vals, width_mhz, Z_plot, shading="auto", cmap="viridis_r")
        fig.colorbar(im, ax=ax, label=r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)")

        if lin_fit is not None and len(lin_fit.get("waist_um_excluded", [])):
            ax.plot(lin_fit["waist_um_excluded"], lin_fit["width_mhz_excluded"], "x",
                    color="lightgray", markeredgecolor="dimgray", markersize=8, markeredgewidth=2,
                    label="valley point, excluded from fit")
            ax.plot(lin_fit["waist_um"], lin_fit["width_mhz"], "o", color="white",
                    markeredgecolor="black", markersize=6, label="valley point, used for fit")
        else:
            valley_waist, valley_width = extract_uniformity_valley(results)
            ax.plot(valley_waist, valley_width, "o", color="white", markeredgecolor="black",
                    markersize=6, label="Uniformity$_w$ minimum per waist column")

        if lin_fit is not None:
            waist_used = lin_fit["waist_um"]
            xs = np.linspace(waist_used.min(), waist_used.max(), 100)
            ax.plot(xs, lin_fit["predict"](xs), "-", color="red", linewidth=3.5, label="Fit")

        ax.set_xlabel(r"effective waist at focal plane ($\mu$m)")
        ax.set_ylabel("width (MHz)")
        title = "Waist-width coupling (fixed-amplitude scan, atom-weighted)"
        if sigma_atom is not None and np.isfinite(sigma_atom):
            title += rf" ($\sigma_{{\mathrm{{atom}}}}$={sigma_atom*1e9:.1f} nm)"
        ax.set_title(title)
        ax.legend(loc="best", fontsize=10)

    n_win, n_width = len(win_input_vals), len(width_vals)
    filename = f"FlatMultiTone_WaistWidthFitWeighted_N{results.get('N_x', '?')}x{results.get('N_y', '?')}_{n_win}x{n_width}pts.png"
    _finish_fig(fig, filename, out_dir, show, save, confirm_overwrite)
    return fig


# ======================================================================
# main
# ======================================================================
def main():
    print("=" * 70)
    print("1) Amplituden-Fit (ATOM-WEIGHTED): r_x(waist, width), r_y(waist, width)")
    print("   Quelle: gewichteter Amplituden-Scan (r_x_grid/r_y_grid pro Punkt")
    print("   unter alpha*uniformity_weighted+(1-alpha)*eta_weighted optimiert)")
    print("=" * 70)

    amp_fits = None
    amp_results = _load_pkl(AMP_PKL_DATEI, expect_amp=True)
    if amp_results is not None:
        required = ("r_x_grid", "r_y_grid", "win_input_vals", "width_vals", "f1", "f2", "lambda_opt", "fLO")
        missing = [k for k in required if k not in amp_results]
        if missing:
            print(f"'{AMP_PKL_DATEI}' fehlen benötigte Schlüssel: {missing} - kein Amplituden-Fit möglich.")
        else:
            if AMP_CLEAN_OUTLIERS:
                summarize_amp_bounds(amp_results)
                mask = detect_amp_outliers(amp_results, bounds=AMP_CLEAN_BOUNDS)
                if mask.sum():
                    print(f"{int(mask.sum())} Ausreißer-Punkt(e) an Schranke(n) {AMP_CLEAN_BOUNDS} "
                          f"gefunden - bereinige mit '{AMP_CLEAN_STRATEGY}' vor dem Fit.")
                    amp_results = clean_amp_scan_results(amp_results, mask=mask, strategy=AMP_CLEAN_STRATEGY, verbose=False)
                else:
                    print(f"Keine Ausreißer an Schranke(n) {AMP_CLEAN_BOUNDS} gefunden.")
            amp_fits = fit_amplitude_dependence(amp_results, win_axis=WIN_AXIS)

    print("\n" + "=" * 70)
    print("2) Waist-Width-Kopplung aus dem Uniformity_w-Tal (ATOM-WEIGHTED, feste Amplitude)")
    print("   Quelle: gewichteter Fest-Amplitude-Scan (keine Amplituden-Optimierung pro Punkt)")
    print("=" * 70)

    waist_width_fit = None
    fixed_results = _load_pkl(FIXED_PKL_DATEI, expect_amp=False)
    if fixed_results is not None:
        required = ("uniformity_weighted_grid", "win_input_vals", "width_vals", "f1", "f2", "lambda_opt", "fLO")
        missing = [k for k in required if k not in fixed_results]
        if missing:
            print(f"'{FIXED_PKL_DATEI}' fehlen benötigte Schlüssel: {missing} - kein Waist-Width-Fit möglich.")
        else:
            waist_width_fit = fit_waist_width_relation(fixed_results)

    if PLOT_FITS:
        print("\n" + "=" * 70)
        print("3) Diagnose-Plots")
        print(f"   Ordner: {FIT_PLOTS_DIR}")
        print("=" * 70)

        save = SAVE
        if SAVE and ASK_BEFORE_SAVE:
            try:
                antwort = input("Diagnose-Plots in 'Fit_Plots' speichern? [y/N]: ").strip().lower()
                save = antwort in ("y", "yes", "j", "ja")
                if not save:
                    print("-> Bilder werden NICHT gespeichert (nur angezeigt, falls SHOW=True).")
            except EOFError:
                print("(ASK_BEFORE_SAVE=True, aber keine Eingabe möglich (kein Terminal) - "
                      "verwende SAVE={} wie konfiguriert.)".format(SAVE))

        if amp_fits is not None:
            for key in ("r_x", "r_y"):
                plot_amplitude_fit(amp_results, amp_fits, key, win_axis=WIN_AXIS,
                                    show=SHOW, save=save)
        if waist_width_fit is not None:
            plot_waist_width_fit(fixed_results, waist_width_fit, show=SHOW, save=save)

    print("\n" + "=" * 70)
    print("Fertig. amp_fits / waist_width_fit enthalten die 'predict'-Funktionen "
          "zum Wiederverwenden (siehe Modul-Docstring für ein Import-Beispiel).")
    print("=" * 70)

    return amp_fits, waist_width_fit


if __name__ == "__main__":
    main()
