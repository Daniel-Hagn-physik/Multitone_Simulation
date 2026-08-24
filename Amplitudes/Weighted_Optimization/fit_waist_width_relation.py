"""
fit_waist_width_relation.py
=============================
EIN Skript: laedt den GEWICHTETEN Fest-Amplitude-Scan
(scan_data_weighted_...pkl, mit uniformity_weighted_grid - KEINE r_x_grid/
r_y_grid, die Amplituden sind hier NICHT pro Gitterpunkt optimiert, sondern
fest/konstant), extrahiert das Uniformity_w-Tal (fuer jede Waist-Spalte der
width-Wert mit minimaler Uniformity_w), vergleicht per Kreuzvalidierung
mehrere Kandidaten-Funktionen fuer den Zusammenhang width(waist) - und
plottet das Ergebnis. Alles in einem Skript, direkt ausfuehrbar.

Abgrenzung zu fit_central_amplitudes.py (frueher fit_stripe_closed_form.py):
- fit_central_amplitudes.py: r_x(waist, width), r_y(waist, width) aus dem
  AMPLITUDEN-optimierten Scan (scan_amp_data_weighted_...pkl) - die
  Amplituden wurden an JEDEM Gitterpunkt separat optimiert. Rechnet in mm
  VOR der ersten Linse (win_input, der tatsaechliche Scan-Parameter).
- DIESES Skript: der Zusammenhang zwischen effektivem Waist (an der
  Fokusebene, NACH allen Linsen, in Mikrometern) und width (MHz) bei FESTEN/
  KONSTANTEN Amplituden - die Frage "bei welcher width ist der Flat-Top fuer
  einen gegebenen Waist am gleichmaessigsten (minimale Uniformity_w)?". Das
  ist die Ablösung des Fit-Teils, der bisher (mit einer fest angenommenen
  linearen Funktion) in beispiel_weighted_amp_fit_abhaengigkeiten.py steckte
  (fit_waist_width_relation()/_fit_line_dropping_kinks()) - der DORTIGE
  zweite Fit-Teil (r_x/r_y-Amplitudenabhaengigkeit) wird nicht mehr
  gebraucht, weil fit_central_amplitudes.py ihn ersetzt hat.

Achsen-Konvention (auf User-Wunsch konsequent unterschieden - siehe auch
fit_central_amplitudes.py):
- fit_central_amplitudes.py -> Plots in Fit_Plots/mm_waist/ (mm VOR der Linse).
- DIESES Skript -> Plots in Fit_Plots/um_waist/ (Mikrometer NACH der Linse,
  effektiver Waist an der Fokusebene - win_input_to_win() rechnet um, aus
  weighted_multitone_amplitude_dependence_plots.py importiert, damit die
  Umrechnungsformel nicht dupliziert wird).

Modell-Wahl: bisher wurde IMMER eine lineare Funktion angenommen. Dieses
Skript prueft das nicht nur an, sondern vergleicht per Block-Kreuzvalidierung
(Talpunkte entlang der Waist-Achse in Bloecke geteilt, analog zur
raeumlichen Block-CV in fit_central_amplitudes.py) mehrere Kandidaten:
linear, quadratisch, kubisch, Potenzgesetz (w=a*waist^p), reziprok
(w=a/waist+b) und eine affine+reziproke Korrektur (w=a*waist+b+c/waist) -
und waehlt automatisch die beste.

Physikalische Zusatzueberlegung fuer den reziproken Kandidaten: der
effektive Waist an der Fokusebene ist selbst umgekehrt proportional zum
VOR der Linse gescannten win_input (waist_um = C/win_input, siehe
win_input_to_win()). Ist der WAHRE Zusammenhang linear in win_input (mm vor
der Linse, wie tatsaechlich gescannt), dann ist er durch diese 1/x-
Transformation zwangslaeufig reziprok in waist_um (NACH der Linse) - genau
deshalb ist "reziprok" hier kein willkuerlicher Kandidat, sondern ein
physikalisch motivierter.

Nutzung:
    python fit_waist_width_relation.py
(vorher ggf. PKL_DATEI unten anpassen). Ausgabe: Konsole (Modellvergleich,
Formel) UND, sofern PLOT_FITS=True (Default), ein Diagnose-Plot (Heatmap +
Talpunkte + Fit-Kurve) in Fit_Plots/um_waist/.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq

from weighted_multitone_amplitude_dependence_plots import (
    load_amp_scan_results,  # generischer Pickle-Loader (Name historisch, laedt JEDES
                            # von save_scan_weighted_results()/save_scan_amp_results_weighted()
                            # erzeugte dict - hier fuer die Fest-Amplitude-Datei genutzt)
    win_input_to_win,
    resolve_save_path,
    DEFAULT_RESULTS_DIR,
)


def _default_dir(name):
    """Legt einen Ordner neben DIESEM Skript an (bei Bedarf automatisch) -
    identisches Muster wie in fit_central_amplitudes.py."""
    candidate = Path(__file__).resolve().parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = Path(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# Waist-Achsen-Konvention: DIESES Skript rechnet im effektiven Waist NACH
# der Linse (Mikrometer) - Plots landen deshalb in einem eigenen Unterordner
# "um_waist", getrennt von fit_central_amplitudes.py's "mm_waist"
# (Waist VOR der Linse) - auf Wunsch, damit an Ordnernamen sofort erkennbar
# ist, welche Waist-Konvention ein Plot verwendet.
FIT_PLOTS_DIR = _default_dir("Fit_Plots/um_waist")
FIT_RESULTS_DIR = _default_dir("Fit_Results")

# LaTeX-freundlicher Plot-Stil - identisch zu fit_central_amplitudes.py.
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


# ======================================================================
# Konfiguration - hier anpassen
# ======================================================================

# GEWICHTETER Fest-Amplitude-Scan (mit uniformity_weighted_grid, OHNE
# r_x_grid/r_y_grid - NICHT dieselbe Datei wie fuer fit_central_amplitudes.py!).
PKL_DATEI = r"scan_data_weighted_N3x4_151x151pts_Airy_2500res.pkl"

OUTPUT_PREFIX = "WaistWidthFit_N3x4_151x151pts_Airy_weighted_2026-08-24"

# Block-Kreuzvalidierung entlang der (sortierten) Waist-Achse.
CV_BLOCKS = 5

# Kandidaten-Funktionen fuer width_mhz(waist_um) - siehe MODELS unten fuer
# die tatsaechlichen Definitionen/Startwerte.
CANDIDATE_MODELS = ("linear", "quadratic", "cubic", "power", "reciprocal", "affine_reciprocal")

PLOT_FITS = True
SHOW = False
SAVE = True
ASK_BEFORE_SAVE = True
PDF_FIGSIZE = (7.5, 5.8)
PDF_RASTER_DPI = 300


# ======================================================================
# 1) Uniformity_w-Tal extrahieren
# ======================================================================
def extract_valley(results):
    """Fuer jede Waist-Spalte (win_input-Wert) den width-Wert, an dem
    uniformity_weighted_grid minimal ist (mit parabolischer Subpixel-
    Interpolation um das Minimum, wie im bisherigen
    beispiel_weighted_amp_fit_abhaengigkeiten.py). Gibt (waist_um, width_mhz)
    zurueck - waist_um ist der EFFEKTIVE Waist an der Fokusebene.

    Spalten, bei denen argmin GENAU am Rand des gescannten width-Bereichs
    liegt (i_min == 0 oder i_min == n_width-1), werden verworfen: dort liegt
    das wahre Uniformity_w-Minimum ausserhalb des gescannten width-Fensters
    (0.2-0.4 MHz) - der Randwert ist dann kein echtes Minimum, sondern ein
    Artefakt der endlichen Scan-Breite, und wuerde die Kurvenform verfaelschen
    (identisches Prinzip wie das Entfernen der Saettigungs-Ecken im
    fit_central_amplitudes.py-Stripe-Verfahren)."""
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    Z = results["uniformity_weighted_grid"]  # shape (n_width, n_win)
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]

    waist_um = np.array([win_input_to_win(w, f1, f2, lambda_opt, fLO) for w in win_input_vals]) * 1e6
    width_mhz = width_vals * 1e-6
    n_width = len(width_mhz)

    valley_waist, valley_width = [], []
    n_boundary = 0
    for j in range(len(win_input_vals)):
        col = Z[:, j]
        if not np.any(np.isfinite(col)):
            continue
        i_min = int(np.nanargmin(col))
        if i_min == 0 or i_min == n_width - 1:
            n_boundary += 1
            continue
        width_at_min = width_mhz[i_min]
        if np.isfinite(col[i_min - 1]) and np.isfinite(col[i_min + 1]):
            y0, y1, y2 = col[i_min - 1], col[i_min], col[i_min + 1]
            denom = y0 - 2 * y1 + y2
            if denom != 0:
                delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
                width_at_min = float(np.interp(i_min + delta, np.arange(len(width_mhz)), width_mhz))
        valley_waist.append(waist_um[j])
        valley_width.append(width_at_min)

    if n_boundary:
        print(f"   ({n_boundary} Spalte(n) verworfen: Uniformity_w-Minimum lag am Rand des "
              f"gescannten width-Bereichs [{width_mhz.min():.2f}, {width_mhz.max():.2f}] MHz - "
              f"dort wurde das wahre Minimum vom Scan-Fenster abgeschnitten.)")

    return np.array(valley_waist), np.array(valley_width)


def drop_disconnected_branch(x, y, jump_factor=6.0):
    """Nach dem Entfernen der Randsaettigungs-Spalten (siehe extract_valley)
    kann eine ZWEITE, davon getrennte Punktwolke uebrigbleiben: fuer
    waist_um oberhalb eines gewissen Werts liegt das wahre Uniformity_w-
    Minimum ausserhalb des gescannten width-Fensters (das erzeugt die
    Randsaettigung), aber es existiert dort zusaetzlich ein ZWEITES,
    lokales Minimum INNERHALB des Fensters - bei einem anderen (meist
    niedrigeren) width-Wert als die Fortsetzung des Haupttals erwarten
    liesse. Das fuehrt zu einem SPRUNG (keinem sanften Kink) im sortierten
    (waist,width)-Talverlauf, den drop_edge_kinks (das nur die beiden
    Rand-Residuen einer global gefitteten Geraden prueft) nicht zuverlaessig
    erkennt. Diese Funktion sucht direkt nach Sprüngen in der sortierten
    Punktfolge (|delta_width| eines Schritts deutlich groesser als der
    typische Schritt, robust via MAD) und behaelt nur das GROESSTE
    zusammenhaengende Segment - die uebrigen Punkte gelten als nicht
    zuverlaessig demselben physikalischen Tal zugehoerig und werden
    ausgeschlossen (nicht geloescht: sie werden im Plot als 'excluded'
    markiert, damit nichts unter den Tisch faellt)."""
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    n = len(x_sorted)
    if n < 4:
        return x_sorted, y_sorted, np.array([]), np.array([])

    steps = np.abs(np.diff(y_sorted))
    mad = np.median(np.abs(steps - np.median(steps)))
    scale = 1.4826 * mad
    floor = 0.05 * (np.max(y_sorted) - np.min(y_sorted) + 1e-12)
    threshold = jump_factor * max(scale, floor)

    jump_idx = np.where(steps > threshold)[0]  # Sprung zwischen i und i+1
    if len(jump_idx) == 0:
        return x_sorted, y_sorted, np.array([]), np.array([])

    boundaries = [0] + [int(i) + 1 for i in jump_idx] + [n]
    segments = [(boundaries[k], boundaries[k + 1]) for k in range(len(boundaries) - 1)]
    lo, hi = max(segments, key=lambda seg: seg[1] - seg[0])

    keep = np.zeros(n, dtype=bool)
    keep[lo:hi] = True
    return x_sorted[keep], y_sorted[keep], x_sorted[~keep], y_sorted[~keep]


def drop_edge_kinks(x, y, min_points=4, min_keep_frac=0.5, mad_factor=3.0):
    """Robustes, iteratives Trimmen von Rand-'Kinks' (Gitterrand-Artefakte,
    an denen das Tal abknickt) - UNVERAENDERTE Logik aus
    beispiel_weighted_amp_fit_abhaengigkeiten.py's _fit_line_dropping_kinks(),
    hier verselbststaendigt (nur der Trimm-Schritt, nicht an ein bestimmtes
    Modell gebunden) und VOR dem Modellvergleich einmal angewendet, damit
    alle Kandidaten-Funktionen auf demselben bereinigten Punktesatz verglichen
    werden. Referenz-Fit fuer die Ausreisser-Erkennung bleibt eine einfache
    Gerade (robust, modellunabhaengig als Trend-Referenz geeignet)."""
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    n = len(x_sorted)
    min_keep = max(min_points, int(np.ceil(min_keep_frac * n)))

    lo, hi = 0, n
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

    keep = np.zeros(n, dtype=bool)
    keep[lo:hi] = True
    return x_sorted[keep], y_sorted[keep], x_sorted[~keep], y_sorted[~keep]


# ======================================================================
# 2) Kandidaten-Modelle
# ======================================================================
def _f_linear(x, a, b):
    return a * x + b


def _f_quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def _f_cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def _f_power(x, a, p):
    return a * np.power(x, p)


def _f_reciprocal(x, a, b):
    return a / x + b


def _f_affine_reciprocal(x, a, b, c):
    return a * x + b + c / x


def _initial_guess(name, x, y):
    m, b = np.polyfit(x, y, 1)
    guesses = {
        "linear": [m, b],
        "quadratic": [0.0, m, b],
        "cubic": [0.0, 0.0, m, b],
        "power": [y.mean() / x.mean(), 1.0],
        "reciprocal": [m * x.mean()**2, y.mean() - m * x.mean()],
        "affine_reciprocal": [m, b, 0.0],
    }
    return guesses[name]


MODELS = {
    "linear": (_f_linear, "width = a*waist + b"),
    "quadratic": (_f_quadratic, "width = a*waist^2 + b*waist + c"),
    "cubic": (_f_cubic, "width = a*waist^3 + b*waist^2 + c*waist + d"),
    "power": (_f_power, "width = a*waist^p"),
    "reciprocal": (_f_reciprocal, "width = a/waist + b"),
    "affine_reciprocal": (_f_affine_reciprocal, "width = a*waist + b + c/waist"),
}

# Anzahl freier Parameter pro Modell - fuer die Occam's-razor-Tie-Break-Regel
# in main(): bei (annaehernd) gleich guter Block-CV wird das Modell mit den
# WENIGSTEN Parametern gewaehlt, nicht einfach das erste beste (rein
# numerische) Maximum. Sonst gewinnt bei einem fast perfekt linearen
# Datenbereich zufaellig ein unnoetig komplexes Modell (z.B. cubic), nur weil
# mehr Parameter minimal besser zum CV-Rauschen passen.
PARAM_COUNT = {name: len(_initial_guess(name, np.array([1.0, 2.0]), np.array([1.0, 2.0])))
                for name in MODELS}


def _fit_model(name, x, y):
    func, _ = MODELS[name]
    p0 = _initial_guess(name, x, y)
    popt, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)
    return popt


def _r2(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ======================================================================
# 3) Block-Kreuzvalidierung ueber alle Kandidaten
# ======================================================================
def cv_compare_models(x, y, n_blocks=CV_BLOCKS, candidates=CANDIDATE_MODELS, verbose=True):
    """Sortiert (x,y) nach x, teilt in n_blocks zusammenhaengende Bloecke,
    und bewertet jeden Kandidaten per Leave-one-block-out. Gibt ein dict
    {name: {"r2_mean":..., "r2_std":..., "popt":...}} zurueck, sortiert nach
    absteigendem r2_mean nicht erzwungen (Aufrufer waehlt selbst das Maximum)."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    n = len(x_sorted)
    block_id = np.minimum((np.arange(n) * n_blocks) // n, n_blocks - 1)

    results = {}
    for name in candidates:
        scores = []
        for b in range(n_blocks):
            train = block_id != b
            test = block_id == b
            if train.sum() < len(_initial_guess(name, x_sorted, y_sorted)) + 1 or test.sum() < 2:
                continue
            try:
                popt = _fit_model(name, x_sorted[train], y_sorted[train])
                pred = MODELS[name][0](x_sorted[test], *popt)
                if not np.all(np.isfinite(pred)):
                    continue
                scores.append(_r2(y_sorted[test], pred))
            except (RuntimeError, ValueError):
                continue
        if scores:
            m, s = float(np.mean(scores)), float(np.std(scores))
        else:
            m, s = float("nan"), float("nan")
        results[name] = dict(r2_mean=m, r2_std=s, n_folds_ok=len(scores))
        if verbose:
            print(f"    {name:<18s} R2 = {m:.4f} +/- {s:.4f}  ({len(scores)}/{n_blocks} Bloecke ausgewertet)")
    return results


# ======================================================================
# 4) Plot
# ======================================================================
def _finish_fig(fig, filename, out_dir, show, save):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save:
        out_file = resolve_save_path(out_dir, filename, confirm_overwrite=None)
        fig.savefig(out_file, format="pdf", dpi=PDF_RASTER_DPI, bbox_inches="tight")
        print(f"Plot gespeichert: {out_file}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_waist_width_fit(results, waist_used, width_used, waist_excluded, width_excluded,
                          model_name, popt, out_dir=FIT_PLOTS_DIR, prefix=OUTPUT_PREFIX,
                          show=SHOW, save=SAVE):
    """Heatmap von Uniformity_w (effektiver Waist in µm auf der x-Achse) mit
    Talpunkten und Fit-Kurve - Englisch, Vektor-PDF, LaTeX-tauglich."""
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]

    waist_um_axis = np.array([win_input_to_win(w, f1, f2, lambda_opt, fLO) for w in win_input_vals]) * 1e6
    width_mhz_axis = width_vals * 1e-6
    reversed_ = waist_um_axis[0] > waist_um_axis[-1]
    x_axis = waist_um_axis[::-1] if reversed_ else waist_um_axis
    Z = (results["uniformity_weighted_grid"][:, ::-1] if reversed_
         else results["uniformity_weighted_grid"]) * 100.0

    fig, ax = plt.subplots(figsize=PDF_FIGSIZE, constrained_layout=True)
    im = ax.pcolormesh(x_axis, width_mhz_axis, Z, shading="auto", cmap="viridis_r")
    fig.colorbar(im, ax=ax, label=r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)")

    if len(waist_excluded):
        ax.plot(waist_excluded, width_excluded, "x", color="lightgray", markeredgecolor="dimgray",
                 markersize=8, markeredgewidth=2, label="valley point, excluded (scan-window artifact)")
    ax.plot(waist_used, width_used, "o", color="white", markeredgecolor="black",
             markersize=6, label="valley point, used for fit")

    xs = np.linspace(waist_used.min(), waist_used.max(), 200)
    func, formula_str = MODELS[model_name]
    ax.plot(xs, func(xs, *popt), "-", color="red", linewidth=3.0, label=f"fit ({model_name})")

    ax.set_xlabel(r"effective waist at focal plane ($\mu$m)")
    ax.set_ylabel("width (MHz)")
    ax.set_title("Waist-width coupling (fixed-amplitude scan)")
    ax.legend(loc="best", fontsize=9)

    _finish_fig(fig, f"{prefix}_waist_width_fit.pdf", out_dir, show, save)


# ======================================================================
# 5) Predictor + Inverse
# ======================================================================
def make_predictor(model_name, popt):
    func, _ = MODELS[model_name]

    def predict_width_mhz(waist_um):
        x = np.atleast_1d(np.asarray(waist_um, dtype=float))
        w = func(x, *popt)
        return w if w.size > 1 else float(w[0])

    def predict_waist_um(width_mhz, bracket):
        """Numerische Umkehrung (Bisektion) - funktioniert fuer jedes der
        MODELS, solange width(waist) im uebergebenen bracket=(lo,hi) monoton
        ist (im Talbereich der Fall)."""
        target = float(width_mhz)
        lo, hi = bracket
        g = lambda x: func(x, *popt) - target
        return brentq(g, lo, hi)

    return predict_width_mhz, predict_waist_um


# ======================================================================
# main
# ======================================================================
def main():
    print(f"Lade '{PKL_DATEI}' ...")
    try:
        results = load_amp_scan_results(PKL_DATEI)
    except FileNotFoundError:
        vorhandene = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob("scan_data_weighted_*.pkl"))
        print(f"'{PKL_DATEI}' wurde weder im aktuellen Ordner noch in '{DEFAULT_RESULTS_DIR}' gefunden.")
        if vorhandene:
            print("Vorhandene Fest-Amplitude-Scan-Dateien ('scan_data_weighted_...pkl'):")
            for name in vorhandene:
                print(f"  - {name}")
        return
    if "uniformity_weighted_grid" not in results:
        print("Diese Datei enthaelt kein 'uniformity_weighted_grid' - ist das wirklich ein "
              "Fest-Amplitude-Scan (scan_data_weighted_...pkl), kein Amplituden-Scan "
              "(scan_amp_data_weighted_...pkl)?")
        return

    print("\n1) Uniformity_w-Tal extrahieren (ein Punkt pro Waist-Spalte) ...")
    waist_all, width_all = extract_valley(results)
    print(f"   {len(waist_all)} Talpunkte gefunden "
          f"(waist_um {waist_all.min():.2f}-{waist_all.max():.2f}).")

    waist_main, width_main, waist_branch, width_branch = drop_disconnected_branch(waist_all, width_all)
    if len(waist_branch):
        print(f"   {len(waist_branch)} Punkt(e) gehoeren zu einem davon getrennten "
              f"Nebenzweig (Sprung im Talverlauf - vermutlich ein zweites, lokales "
              f"Uniformity_w-Minimum ausserhalb des Hauptzweigs) und werden verworfen.")

    waist_used, width_used, waist_excl_k, width_excl_k = drop_edge_kinks(waist_main, width_main)
    print(f"   {len(waist_excl_k)} Rand-Kink-Punkt(e) zusaetzlich ausgeschlossen, "
          f"{len(waist_used)} verbleiben.")
    waist_excl = np.concatenate([waist_branch, waist_excl_k])
    width_excl = np.concatenate([width_branch, width_excl_k])

    print(f"\n2) Block-Kreuzvalidierung ueber {len(CANDIDATE_MODELS)} Kandidaten-Modelle ...")
    cv_results = cv_compare_models(waist_used, width_used)
    valid = [n for n in cv_results if np.isfinite(cv_results[n]["r2_mean"])]
    best_r2 = max(cv_results[n]["r2_mean"] for n in valid)
    # Occam's razor: bei (annaehernd) gleich gutem Block-CV-R^2 (Toleranz
    # 1e-3) gewinnt das Modell mit den WENIGSTEN freien Parametern, nicht
    # einfach das rein numerische Maximum - siehe PARAM_COUNT weiter oben.
    tol = 1e-3
    tied = [n for n in valid if best_r2 - cv_results[n]["r2_mean"] <= tol]
    best_name = min(tied, key=lambda n: (PARAM_COUNT[n], CANDIDATE_MODELS.index(n)))
    if len(tied) > 1:
        print(f"   ({len(tied)} Modelle praktisch gleichauf (R² innerhalb {tol}): "
              f"{', '.join(tied)} -> einfachstes (wenigste Parameter) gewaehlt.)")
    print(f"   -> bestes Modell: {best_name}  (R²(Block-CV) = "
          f"{cv_results[best_name]['r2_mean']:.4f} +/- {cv_results[best_name]['r2_std']:.4f})")

    print(f"\n3) Finaler Fit ({best_name}) auf allen {len(waist_used)} bereinigten Talpunkten ...")
    popt = _fit_model(best_name, waist_used, width_used)
    func, formula_str = MODELS[best_name]
    pred_full = func(waist_used, *popt)
    r2_fulldata = _r2(width_used, pred_full)
    param_names = "abcdefgh"[:len(popt)]
    param_str = ", ".join(f"{n}={v:.6g}" for n, v in zip(param_names, popt))
    print(f"   {formula_str}")
    print(f"   {param_str}")
    print(f"   R²(volle bereinigte Daten, NICHT CV, nur Sanity-Check) = {r2_fulldata:.5f}")

    if PLOT_FITS:
        print(f"\n4) Diagnose-Plot erzeugen (Ordner: {FIT_PLOTS_DIR}) ...")
        save = SAVE
        if SAVE and ASK_BEFORE_SAVE:
            try:
                antwort = input("Diagnose-Plot in 'Fit_Plots/um_waist' speichern? [y/N]: ").strip().lower()
                save = antwort in ("y", "yes", "j", "ja")
                if not save:
                    print("-> Bild wird NICHT gespeichert (nur angezeigt, falls SHOW=True).")
            except EOFError:
                print(f"(ASK_BEFORE_SAVE=True, aber keine Eingabe möglich (kein Terminal) - "
                      f"verwende SAVE={SAVE} wie konfiguriert.)")
        plot_waist_width_fit(results, waist_used, width_used, waist_excl, width_excl,
                              best_name, popt, show=SHOW, save=save)

    predict_width_mhz, predict_waist_um = make_predictor(best_name, popt)
    print(f"\nFertig. Modell '{best_name}' gilt fuer waist_um in "
          f"[{waist_used.min():.2f}, {waist_used.max():.2f}] (der bereinigte Talbereich) - "
          f"ausserhalb ist Extrapolation, mit Vorsicht zu geniessen.")
    return dict(model=best_name, popt=popt, cv_results=cv_results,
                waist_used=waist_used, width_used=width_used,
                waist_excluded=waist_excl, width_excluded=width_excl,
                predict_width_mhz=predict_width_mhz, predict_waist_um=predict_waist_um)


if __name__ == "__main__":
    main()

