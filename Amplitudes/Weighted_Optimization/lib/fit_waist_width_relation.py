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

NEU (2026-08-27, auf User-Wunsch): "ich möchte lineare Funktionen an meine
Datensätze. Bei einem scan über konstante Amplituden sollen das lineare
Funktionen sein, keine Quadratischen" - fuer GENAU diesen Fit (Fest-
Amplitude-Scan) wird das Modell jetzt standardmaessig auf "linear"
ERZWUNGEN statt es automatisch per Block-CV auswaehlen zu lassen (siehe
FORCE_MODEL weiter unten). Die Block-CV laeuft trotzdem weiter und wird
im Konsolen-Log angezeigt (informativ, z.B. um zu sehen, wie gut linear
im Vergleich zu den anderen Kandidaten abschneidet) - sie entscheidet nur
nicht mehr ueber das tatsaechlich verwendete Modell. FORCE_MODEL=None
(bzw. "auto") reaktiviert die alte automatische Modellwahl.

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
Formel) UND, sofern PLOT_FITS=True (Default), ein Diagnose-Plot in
Fit_Plots/um_waist/ - links Heatmap von Uniformity_w + Talpunkte + Fit-Kurve
(wie bisher), rechts (letztes Panel) ein SCHNITT entlang der Fit-Kurve:
Uniformity_w ausgewertet exakt auf der gefitteten Linie width=f(waist) statt
im Tal-Minimum, zum Vergleich mit dem tatsaechlichen Minimum je Waist-Spalte
- zeigt an, wie gut/schlecht die Gleichmaessigkeit ist, wenn man strikt nach
der (meist linearen) Fit-Formel faehrt, inklusive der Luecke dort, wo die
Fit-Kurve das gescannte width-Fenster verlaesst (siehe cut_along_fit()). Auf
User-Wunsch (NEU) zeigt bei SHOW_CROSSTALK=True (Default) ein zusaetzliches
MITTLERES Panel dieselbe 2D-Heatmap-Ansicht fuer Crosstalk_w
(eta_weighted_grid, direkt vergleichbar mit der Uniformity-Heatmap links,
dieselben Talpunkte/Fit-Kurve/bester Punkt ueberlagert), und das rechte
Schnitt-Panel zeigt zusaetzlich Crosstalk_w entlang der Fit-Kurve auf einer
zweiten y-Achse - beides gemeinsam an/abschaltbar ueber SHOW_CROSSTALK bzw.
den show_crosstalk-Parameter von main() (bei False: normaler 1x2-Plot wie
zuvor, nur Uniformity).
"""
from pathlib import Path
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
from scipy.interpolate import RegularGridInterpolator

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
    candidate = Path(__file__).resolve().parent.parent / name
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

# Modell-Erzwingung (NEU, 2026-08-27, auf User-Wunsch): bei einem Fest-
# Amplitude-Scan soll der Zusammenhang width(waist) IMMER als lineare
# Funktion gefittet werden - nicht ueber die automatische Block-CV-
# Modellwahl (cv_compare_models() + Occam's-Razor-Tie-Break, siehe main())
# potenziell ein quadratisches/kubisches/etc. Modell waehlen koennen, was
# v.a. bei verrauschten/kleinen Datensaetzen (z.B. mit Atom-Offset, wenige
# Talpunkte) passieren kann, obwohl der physikalisch erwartete Zusammenhang
# linear ist (siehe Docstring oben).
# - "linear" (Default): width(waist) wird immer linear gefittet. Die
#   Block-CV ueber alle CANDIDATE_MODELS laeuft trotzdem (nur zu
#   Informationszwecken, im Konsolen-Log sichtbar).
# - None oder "auto": alte automatische Modellwahl reaktivieren (bestes
#   Modell per Block-CV + Occam's-Razor-Tie-Break, wie vor diesem Update).
# - einer der anderen CANDIDATE_MODELS-Namen: erzwingt stattdessen DIESES
#   Modell (z.B. fuer gezielte Vergleiche).
FORCE_MODEL = "linear"

PLOT_FITS = True
SHOW = False
SAVE = True
ASK_BEFORE_SAVE = True
PDF_FIGSIZE = (12.5, 5.5)  # 1x2-Layout: Uniformity-Heatmap links, Schnitt-Panel rechts
                            # (SHOW_CROSSTALK=False)
PDF_FIGSIZE_CROSSTALK = (21.0, 5.5)  # 1x3-Layout bei SHOW_CROSSTALK=True: zusaetzlich eine
                                      # Crosstalk-Heatmap (2D, wie links fuer Uniformity) in
                                      # der Mitte, Schnitt-Panel rechts - insgesamt breiter,
                                      # u.a. damit die (wegen twinx() ausgelagerte, siehe
                                      # plot_waist_width_fit()) Legende des Schnitt-Panels
                                      # Platz hat, ohne die Panels selbst zu quetschen.
PDF_RASTER_DPI = 300
CUT_N_POINTS = 400  # Aufloesung des Uniformity-Schnitts entlang der Fit-Kurve

# Grafische Feintuning-Optionen (auf User-Wunsch, per run_all_fits.py von
# aussen ueberschreibbar, siehe main()-Parameter unten) - identisches Muster
# wie in fit_central_amplitudes.py.
LEGEND_FONTSIZE = 9
DRAW_BEST_POINT = False

# Auf User-Wunsch: im rechten Panel (Schnitt entlang der Fit-Kurve) neben
# Uniformity_w zusaetzlich Crosstalk_w (eta_weighted_grid) zeigen - auf einer
# zweiten y-Achse (twinx), damit beide Kurven auch bei unterschiedlicher
# Groessenordnung gut lesbar bleiben. Per main()-Parameter/run_all_fits.py
# von aussen ueberschreibbar, identisches Muster wie DRAW_BEST_POINT.
SHOW_CROSSTALK = True

# Bildstil fuer den optionalen Best-Point-Marker - identisch zu
# AmplitudeScanPlotter.BEST_POINT_STYLE und zu fit_central_amplitudes.py,
# damit ein "bester Punkt" im ganzen Projekt visuell gleich aussieht.
BEST_POINT_STYLE = dict(
    marker="*", markersize=16, markeredgecolor="white",
    markeredgewidth=1.1, color="red", zorder=8, linestyle="none",
)


# ======================================================================
# 0) Bester Punkt (optional, automatisch OHNE vorgegebene Koordinaten)
# ======================================================================
def _find_best_point(results):
    """Anders als fit_central_amplitudes.py's _find_best_point() muss hier
    NICHTS berechnet werden: der Fest-Amplitude-Scan speichert den besten
    gefundenen Punkt bereits direkt im results-dict unter dem Schluessel
    'best' (win_input, width, uniformity_weighted, eta_weighted, combined) -
    dieselbe Groesse, die auch die uebrigen Plots im Projekt
    (_mark_point()-Konvention) verwenden. Braucht KEINE von Hand
    vorgegebenen Koordinaten - komplett automatisch.

    Gibt dict(waist_um=..., width_mhz=..., label=...) zurueck, oder None,
    falls kein 'best'-Eintrag vorhanden/gueltig ist."""
    best = results.get("best")
    if not best or best.get("win_input") is None:
        return None
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]
    waist_um = win_input_to_win(best["win_input"], f1, f2, lambda_opt, fLO) * 1e6
    width_mhz = best["width"] * 1e-6
    label = "best point (scan optimum)"
    uniformity_percent = None
    if best.get("uniformity_weighted") is not None:
        uniformity_percent = float(best["uniformity_weighted"]) * 100.0
    crosstalk_percent = None
    if best.get("eta_weighted") is not None:
        crosstalk_percent = float(best["eta_weighted"]) * 100.0
    return dict(waist_um=float(waist_um), width_mhz=float(width_mhz), label=label,
                uniformity_percent=uniformity_percent, crosstalk_percent=crosstalk_percent)


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

    return np.array(valley_waist), np.array(valley_width), n_boundary


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
    median_step = np.median(steps)
    mad = np.median(np.abs(steps - median_step))
    scale = 1.4826 * mad
    # Numerischer Sicherheits-Boden (falls MAD==0, z.B. bei exakt aequidistanten
    # Schritten): an den TYPISCHEN Schrittabstand gekoppelt, NICHT an die
    # gesamte y-Spannweite. Ein an y-range gekoppelter Boden (frueher:
    # 0.05*y-range) haengt von der Anzahl Scanpunkte ab, waehrend der
    # physikalische Sprung selbst (zweiter Uniformity_w-Nebenzweig) eine
    # nahezu feste absolute Groesse hat - bei grober Aufloesung (wenige
    # Scanpunkte, z.B. 40x40) uebertraf der alte, y-range-gekoppelte Boden den
    # echten Sprung knapp und verhinderte so dessen Erkennung (siehe Nachtrag
    # in status.md).
    floor = 0.1 * max(median_step, 1e-9)
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
# 4) Schnitt entlang der Fit-Kurve: Uniformity_w(waist) fuer width=fit(waist)
# ======================================================================
def _uniformity_interpolator(results):
    """Baut EINEN RegularGridInterpolator auf dem ORIGINAL-Scan-Gitter
    (win_input_vals x width_vals, beide bereits regelmaessig/aufsteigend -
    NICHT das nichtlinear transformierte waist_um-Gitter, um keine
    zusaetzliche Interpolations-Ungenauigkeit durch die 1/x-Umrechnung
    einzuschleppen). Wird von cut_along_fit() UND von der
    Talpunkt-Vergleichskurve in plot_waist_width_fit() genutzt, damit beide
    exakt dieselbe (unverfaelschte, nicht z.B. achsen-geflippte) Grundlage
    verwenden. Rueckgabewert erwartet Punkte als (width_Hz, win_input_m)."""
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    Z = results["uniformity_weighted_grid"]  # shape (n_width, n_win)
    return RegularGridInterpolator((width_vals, win_input_vals), Z,
                                    bounds_error=False, fill_value=np.nan)


def _crosstalk_interpolator(results):
    """Wie _uniformity_interpolator(), nur fuer Crosstalk_w
    (eta_weighted_grid) statt Uniformity_w - identisches Gitter, identische
    Konvention. Gibt None zurueck, falls diese Datei kein
    'eta_weighted_grid' enthaelt (defensiv, damit ein aelteres/anderes pkl
    nicht mit einem KeyError abbricht, sondern der Crosstalk-Teil einfach
    weggelassen wird - siehe SHOW_CROSSTALK)."""
    if "eta_weighted_grid" not in results:
        return None
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    Z = results["eta_weighted_grid"]  # shape (n_width, n_win)
    return RegularGridInterpolator((width_vals, win_input_vals), Z,
                                    bounds_error=False, fill_value=np.nan)


def _waist_um_to_win_input(results, waist_um):
    """Umkehrung von win_input_to_win(): waist_um -> win_input (m). Dieselbe
    Formel wie in weighted_multitone_amplitude_dependence_plots.py, nur nach
    win_input aufgeloest (f1/f2*lambda_opt*fLO/(pi*win_input) = waist)."""
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]
    waist_m = np.asarray(waist_um, dtype=float) * 1e-6
    return (f1 * lambda_opt * fLO) / (f2 * np.pi * waist_m)


def cut_along_fit(results, model_name, popt, waist_lo, waist_hi, n=CUT_N_POINTS):
    """Wertet Uniformity_w (und, sofern vorhanden, Crosstalk_w) NICHT im
    Tal-Minimum aus, sondern GENAU entlang der gefitteten Kurve
    width = f(waist) - also entlang der Trajektorie, die man tatsaechlich
    befahren wuerde, wenn man width strikt nach der Fit-Formel aus dem
    gewuenschten waist berechnet. Punkte, an denen die Fit-Kurve das
    gescannte width-Fenster (0.2-0.4 MHz) verlaesst, werden als NaN
    zurueckgegeben (erscheinen im Plot als Luecke - genau dort waere die
    Fit-Vorhersage mit den vorhandenen Scan-Daten nicht ueberpruefbar).

    Gibt (waist_cut_um, width_cut_mhz, uniformity_cut_percent,
    crosstalk_cut_percent) zurueck - crosstalk_cut_percent ist None, falls
    'eta_weighted_grid' in dieser Datei fehlt."""
    waist_cut_um = np.linspace(waist_lo, waist_hi, n)
    func, _ = MODELS[model_name]
    width_cut_mhz = func(waist_cut_um, *popt)

    win_input_cut = _waist_um_to_win_input(results, waist_cut_um)
    width_cut_hz = width_cut_mhz * 1e6
    points = np.column_stack([width_cut_hz, win_input_cut])

    interp_u = _uniformity_interpolator(results)
    uniformity_cut = interp_u(points) * 100.0

    interp_c = _crosstalk_interpolator(results)
    crosstalk_cut = interp_c(points) * 100.0 if interp_c is not None else None

    return waist_cut_um, width_cut_mhz, uniformity_cut, crosstalk_cut


def write_formula_doc(prefix, model_name, popt, r2_cv, r2_cv_std, r2_fulldata,
                       waist_used, waist_excluded, best_point,
                       n_boundary=0, out_dir=FIT_RESULTS_DIR, forced=False):
    """Schreibt AUTOMATISCH ein Formel-Dokument (Markdown) mit dem gerade
    gewaehlten Modell/Parametern/R² - analog zu fit_central_amplitudes.py's
    write_formula_doc(), bei jedem Lauf frisch generiert.

    forced (NEU, 2026-08-27): True, wenn model_name ueber FORCE_MODEL
    erzwungen wurde statt automatisch per Block-CV gewaehlt - wird im
    Dokument entsprechend vermerkt, damit spaeter nachvollziehbar bleibt,
    warum genau dieses Modell verwendet wurde."""
    func, formula_str = MODELS[model_name]
    param_names = "abcdefgh"[:len(popt)]
    param_lines = "\n".join(f"{n} = {v:.8g}" for n, v in zip(param_names, popt))
    auswahl_text = f"ERZWUNGEN, FORCE_MODEL='{model_name}'" if forced else "Block-CV + Occam's-Razor-Tie-Break"
    lines = [
        f"# {prefix} — Waist-Width-Fit (Fest-Amplitude-Scan)",
        "",
        f"Automatisch generiert von `fit_waist_width_relation.py` am {date.today().isoformat()}.",
        "",
        f"- Gewaehltes Modell ({auswahl_text}): **{model_name}**",
        f"- Formel: `{formula_str}`",
        f"- R²(Block-CV): {r2_cv:.4f} +/- {r2_cv_std:.4f}",
        f"- R²(volle bereinigte Daten, NICHT CV, nur Sanity-Check): {r2_fulldata:.5f}",
        f"- Gueltigkeitsbereich (zuverlaessiger Talbereich): "
        f"waist_um ∈ [{waist_used.min():.4f}, {waist_used.max():.4f}]",
        f"- Davon ausgeschlossene Talpunkte: {n_boundary + len(waist_excluded)} "
        f"({n_boundary} Scan-Fenster-Artefakte (Rand) + {len(waist_excluded)} Nebenzweig/Rand-Kink-Punkte)",
        "",
        "```",
        param_lines,
        "```",
        "",
    ]
    if best_point is not None:
        lines += [
            f"Bester Punkt (automatisch, {best_point['label']}): "
            f"waist = {best_point['waist_um']:.4f} µm (nach der Linse), "
            f"width = {best_point['width_mhz']:.4f} MHz"
            + (f", Uniformity_w = {best_point['uniformity_percent']:.4f} %"
               if best_point['uniformity_percent'] is not None else "")
            + (f", Crosstalk_w = {best_point['crosstalk_percent']:.4f} %"
               if best_point.get('crosstalk_percent') is not None else "") + ".",
            "",
        ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wie in fit_central_amplitudes.py: reproduzierbar aus der pkl-Datei
    # ableitbar, deshalb bei jedem Lauf ueberschreiben statt interaktiv fragen.
    out_file = resolve_save_path(out_dir, f"{prefix}_Formel.md", confirm_overwrite=lambda p: True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Formel-Dokument gespeichert: {out_file}")
    return out_file


# ======================================================================
# 5) Plot
# ======================================================================
def _finish_fig(fig, filename, out_dir, show, save, ask_before_save=True):
    """ask_before_save steuert (Bugfix): frueher wurde beim UEBERSCHREIBEN
    eines bereits vorhandenen Plots IMMER interaktiv nachgefragt
    (confirm_overwrite=None), unabhaengig von ASK_BEFORE_SAVE - das brach
    "ein Klick and go" (run_all_fits.py) beim ZWEITEN Lauf, sobald die PDFs
    schon existierten (kein Terminal fuer input() vorhanden -> EOFError).
    Jetzt: bei ask_before_save=False wird wie bei den Formel-Dokumenten ohne
    Rueckfrage ueberschrieben (reproduzierbar aus der pkl-Datei); nur beim
    manuellen, interaktiven Aufruf (ask_before_save=True, Default) bleibt die
    Rueckfrage bestehen."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if save:
        confirm_overwrite = None if ask_before_save else (lambda p: True)
        out_file = resolve_save_path(out_dir, filename, confirm_overwrite=confirm_overwrite)
        fig.savefig(out_file, format="pdf", dpi=PDF_RASTER_DPI, bbox_inches="tight")
        print(f"Plot gespeichert: {out_file}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_waist_width_fit(results, waist_used, width_used, waist_excluded, width_excluded,
                          model_name, popt, out_dir=FIT_PLOTS_DIR, prefix=OUTPUT_PREFIX,
                          show=SHOW, save=SAVE, best_point=None, legend_fontsize=LEGEND_FONTSIZE,
                          show_crosstalk=SHOW_CROSSTALK, ask_before_save=ASK_BEFORE_SAVE):
    """Links Heatmap von Uniformity_w (effektiver Waist in µm auf der
    x-Achse) mit Talpunkten und Fit-Kurve; bei SHOW_CROSSTALK=True (NEU)
    zusaetzlich in der Mitte dieselbe 2D-Ansicht fuer Crosstalk_w
    (eta_weighted_grid) - identische Talpunkte/Fit-Kurve/bester-Punkt-
    Ueberlagerung, nur mit anderer Heatmap-Groesse dahinter, damit beide
    Groessen direkt vergleichbar sind; rechts ein SCHNITT entlang dieser
    Fit-Kurve - Uniformity_w(waist) (und optional Crosstalk_w), ausgewertet
    exakt auf der gefitteten Linie width=f(waist), nicht im Tal-Minimum.
    Zeigt, wie gut (bzw. schlecht, sobald man den zuverlaessigen Fit-Bereich
    verlaesst) die Gleichmaessigkeit/das Crosstalk ist, wenn man strikt nach
    der Fit-Formel faehrt. Englisch, Vektor-PDF, LaTeX-tauglich."""
    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    f1, f2, lambda_opt, fLO = results["f1"], results["f2"], results["lambda_opt"], results["fLO"]

    waist_um_axis = np.array([win_input_to_win(w, f1, f2, lambda_opt, fLO) for w in win_input_vals]) * 1e6
    width_mhz_axis = width_vals * 1e-6
    reversed_ = waist_um_axis[0] > waist_um_axis[-1]
    x_axis = waist_um_axis[::-1] if reversed_ else waist_um_axis
    Z = (results["uniformity_weighted_grid"][:, ::-1] if reversed_
         else results["uniformity_weighted_grid"]) * 100.0
    crosstalk_available = "eta_weighted_grid" in results
    show_crosstalk_2d = show_crosstalk and crosstalk_available
    if show_crosstalk_2d:
        Z_ct = (results["eta_weighted_grid"][:, ::-1] if reversed_
                else results["eta_weighted_grid"]) * 100.0
    elif show_crosstalk and not crosstalk_available:
        print("   (SHOW_CROSSTALK=True, aber diese Datei enthaelt kein 'eta_weighted_grid' - "
              "Crosstalk-Heatmap/-Kurve wird ausgelassen.)")

    figsize = PDF_FIGSIZE_CROSSTALK if show_crosstalk_2d else PDF_FIGSIZE
    if show_crosstalk_2d:
        fig, (ax, ax_ct2d, ax_cut) = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    else:
        fig, (ax, ax_cut) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    xs = np.linspace(waist_used.min(), waist_used.max(), 200)
    func, formula_str = MODELS[model_name]

    def _draw_valley_and_fit(ax_hm):
        """Talpunkte + Fit-Kurve + bester Punkt - identisch fuer die
        Uniformity- UND (falls gezeigt) die Crosstalk-Heatmap, damit beide
        Panels 1:1 vergleichbar sind (dieselben (waist,width)-Koordinaten,
        nur eine andere Groesse als Hintergrundfarbe)."""
        if len(waist_excluded):
            ax_hm.plot(waist_excluded, width_excluded, "x", color="lightgray", markeredgecolor="dimgray",
                        markersize=8, markeredgewidth=2, label="valley point, excluded (scan-window artifact)")
        ax_hm.plot(waist_used, width_used, "o", color="white", markeredgecolor="black",
                    markersize=6, label="valley point, used for fit")
        ax_hm.plot(xs, func(xs, *popt), "-", color="red", linewidth=3.0, label=f"fit ({model_name})")
        if best_point is not None:
            ax_hm.plot(best_point["waist_um"], best_point["width_mhz"],
                        label=best_point["label"], **BEST_POINT_STYLE)
        ax_hm.set_xlabel(r"effective waist at focal plane ($\mu$m)")
        ax_hm.set_ylabel("width (MHz)")

    # --- linkes Panel: Uniformity-Heatmap + Talpunkte + Fit-Kurve (wie bisher) ---
    im = ax.pcolormesh(x_axis, width_mhz_axis, Z, shading="auto", cmap="viridis_r")
    fig.colorbar(im, ax=ax, label=r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)")
    _draw_valley_and_fit(ax)
    ax.set_title("Uniformity (fixed-amplitude scan)" if show_crosstalk_2d
                 else "Waist-width coupling (fixed-amplitude scan)")
    ax.legend(loc="best", fontsize=legend_fontsize)

    # --- mittleres Panel (NEU, nur falls SHOW_CROSSTALK=True): dieselbe
    # 2D-Ansicht, aber fuer Crosstalk_w statt Uniformity_w - auf User-Wunsch
    # ("bitte noch die 2D Ansicht des Crosstalks dazu"). ---
    if show_crosstalk_2d:
        im_ct = ax_ct2d.pcolormesh(x_axis, width_mhz_axis, Z_ct, shading="auto", cmap="magma_r")
        fig.colorbar(im_ct, ax=ax_ct2d, label=r"Crosstalk$_w$ ($\eta$) (%)")
        _draw_valley_and_fit(ax_ct2d)
        ax_ct2d.set_title("Crosstalk (fixed-amplitude scan)")
        ax_ct2d.legend(loc="best", fontsize=legend_fontsize)

    # --- rechtes Panel: Uniformity_w (und optional Crosstalk_w) entlang der
    # Fit-Kurve (NEU) ---
    waist_cut, width_cut, uniformity_cut, crosstalk_cut = cut_along_fit(
        results, model_name, popt, waist_lo=x_axis.min(), waist_hi=x_axis.max())

    ax_cut.axvspan(waist_used.min(), waist_used.max(), color="tab:green", alpha=0.10,
                    label="reliable fit domain")
    l1, = ax_cut.plot(waist_cut, uniformity_cut, "-", color="red", linewidth=2.0,
                        label=f"Uniformity, along fit ({model_name})")
    # Talpunkt-Uniformity (echtes Minimum je Spalte) zum Vergleich - zeigt,
    # wie nah die (einfache) Fit-Kurve am tatsaechlichen Optimum je Waist liegt.
    # Dieselbe Interpolationsgrundlage wie cut_along_fit() (siehe
    # _uniformity_interpolator), nicht das oben ggf. geflippte Plot-Z.
    interp_used = _uniformity_interpolator(results)
    win_input_used = _waist_um_to_win_input(results, waist_used)
    width_used_hz = width_used * 1e6
    points_used = np.column_stack([width_used_hz, win_input_used])
    valley_uniformity_used = interp_used(points_used) * 100.0
    l2, = ax_cut.plot(waist_used, valley_uniformity_used, "o", color="black", markersize=4,
                        label="Uniformity, true valley minimum")

    handles = [l1, l2]
    ax_cut.set_xlabel(r"effective waist at focal plane ($\mu$m)")
    ax_cut.set_ylabel(r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)", color="red")
    ax_cut.tick_params(axis="y", labelcolor="red")

    # Crosstalk_w (eta_weighted_grid) auf einer zweiten y-Achse (auf
    # User-Wunsch, per SHOW_CROSSTALK an/abschaltbar) - eigene Achse statt
    # gemeinsam mit Uniformity_w, damit beide Kurven unabhaengig von ihrer
    # relativen Groessenordnung gut lesbar bleiben.
    if show_crosstalk_2d:
        ax_ct = ax_cut.twinx()
        l3, = ax_ct.plot(waist_cut, crosstalk_cut, "--", color="tab:blue", linewidth=2.0,
                           label=f"Crosstalk, along fit ({model_name})")
        interp_c_used = _crosstalk_interpolator(results)
        crosstalk_used = interp_c_used(points_used) * 100.0
        l4, = ax_ct.plot(waist_used, crosstalk_used, "s", color="navy", markersize=4,
                           label="Crosstalk, at valley point")
        ax_ct.set_ylabel(r"Crosstalk$_w$ ($\eta$) (%)", color="tab:blue")
        ax_ct.tick_params(axis="y", labelcolor="tab:blue")
        handles += [l3, l4]

        if best_point is not None and best_point.get("crosstalk_percent") is not None:
            (bp_ct,) = ax_ct.plot(best_point["waist_um"], best_point["crosstalk_percent"],
                                    label=best_point["label"], **BEST_POINT_STYLE)

    if best_point is not None and best_point["uniformity_percent"] is not None:
        (bp_u,) = ax_cut.plot(best_point["waist_um"], best_point["uniformity_percent"],
                                label=best_point["label"], **BEST_POINT_STYLE)
        # Bester Punkt nur EINMAL in der Legende (Uniformity-Achse reicht -
        # bei show_crosstalk erscheint er ggf. zusaetzlich, aber unbeschriftet,
        # auf der Crosstalk-Achse, siehe oben).
        handles.append(bp_u)

    ax_cut.set_title("Uniformity / Crosstalk cut along the fitted line"
                      if show_crosstalk_2d else "Uniformity cut along the fitted line")
    # Gleiche x-Achse wie das Heatmap-Panel (statt Auto-Crop auf den letzten
    # endlichen Punkt) - macht sichtbar, dass die Kurve jenseits des
    # zuverlaessigen Bereichs abbricht, weil die Fit-Vorhersage dort das
    # gescannte width-Fenster verlaesst (siehe cut_along_fit()-Docstring).
    ax_cut.set_xlim(x_axis.min(), x_axis.max())
    if show_crosstalk_2d:
        # loc="best" beruecksichtigt bei einer zweiten y-Achse (twinx) NICHT
        # deren Kurven (matplotlib kennt nur die Artists der eigenen Achse) -
        # koennte die Legende also mitten auf die (ggf. stark oszillierende)
        # Crosstalk-Kurve legen, die je nach Datensatz unterschiedlich hoch
        # ausschlagen kann. Robuste Loesung: Legende komplett AUSSERHALB des
        # Panels platzieren (rechts daneben, siehe PDF_FIGSIZE_CROSSTALK) -
        # dort ueberlappt sie nie mit irgendwelchen Kurven, unabhaengig von
        # deren Form.
        ax_cut.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.16, 1.0),
                       fontsize=legend_fontsize, borderaxespad=0.0)
    else:
        # Ohne Crosstalk (nur EINE Kurve, monoton) reicht die normale
        # In-Panel-Platzierung wie bisher - keine Notwendigkeit fuer die
        # breitere Figur/externe Legende.
        ax_cut.legend(handles=handles, loc="best", fontsize=legend_fontsize)

    _finish_fig(fig, f"{prefix}_waist_width_fit.pdf", out_dir, show, save, ask_before_save=ask_before_save)


# ======================================================================
# 6) Predictor + Inverse
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
def main(pkl_datei=None, output_prefix=None, draw_best_point=None, legend_fontsize=None,
         ask_before_save=None, save=None, show=None, show_crosstalk=None, force_model=None):
    """Alle Parameter sind optional - None faellt auf die Modul-Konfiguration
    oben zurueck (PKL_DATEI/OUTPUT_PREFIX/DRAW_BEST_POINT/LEGEND_FONTSIZE/
    ASK_BEFORE_SAVE/SAVE/SHOW/SHOW_CROSSTALK/FORCE_MODEL). Identisches Muster
    wie in fit_central_amplitudes.py's main() - genutzt von run_all_fits.py.

    force_model (NEU, 2026-08-27): None faellt auf FORCE_MODEL zurueck
    (Default "linear" - siehe dortiger Kommentar). Zum expliziten
    Reaktivieren der automatischen Modellwahl "auto" uebergeben (nicht
    None, das wuerde nur den Modul-Default uebernehmen)."""
    pkl_datei = PKL_DATEI if pkl_datei is None else pkl_datei
    output_prefix = OUTPUT_PREFIX if output_prefix is None else output_prefix
    draw_best_point = DRAW_BEST_POINT if draw_best_point is None else draw_best_point
    legend_fontsize = LEGEND_FONTSIZE if legend_fontsize is None else legend_fontsize
    ask_before_save = ASK_BEFORE_SAVE if ask_before_save is None else ask_before_save
    save = SAVE if save is None else save
    show = SHOW if show is None else show
    show_crosstalk = SHOW_CROSSTALK if show_crosstalk is None else show_crosstalk
    force_model = FORCE_MODEL if force_model is None else force_model
    if isinstance(force_model, str) and force_model.lower() == "auto":
        force_model = None
    if force_model is not None and force_model not in MODELS:
        raise ValueError(f"force_model='{force_model}' ist kein bekanntes Modell "
                          f"(bekannt: {', '.join(MODELS)}, oder None/'auto' fuer automatische Wahl).")

    print(f"Lade '{pkl_datei}' ...")
    try:
        results = load_amp_scan_results(pkl_datei)
    except FileNotFoundError:
        vorhandene = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob("scan_data_weighted_*.pkl"))
        print(f"'{pkl_datei}' wurde weder im aktuellen Ordner noch in '{DEFAULT_RESULTS_DIR}' gefunden.")
        if vorhandene:
            print("Vorhandene Fest-Amplitude-Scan-Dateien ('scan_data_weighted_...pkl'):")
            for name in vorhandene:
                print(f"  - {name}")
        return None
    if "uniformity_weighted_grid" not in results:
        print("Diese Datei enthaelt kein 'uniformity_weighted_grid' - ist das wirklich ein "
              "Fest-Amplitude-Scan (scan_data_weighted_...pkl), kein Amplituden-Scan "
              "(scan_amp_data_weighted_...pkl)?")
        return None

    best_point = _find_best_point(results) if draw_best_point else None
    if draw_best_point:
        if best_point is not None:
            print(f"   Bester Punkt (automatisch, aus results['best']): "
                  f"waist_um={best_point['waist_um']:.4f}, width_mhz={best_point['width_mhz']:.4f}")
        else:
            print("   DRAW_BEST_POINT=True, aber kein gueltiger 'best'-Eintrag in dieser Datei - "
                  "kein Punkt eingezeichnet.")

    print("\n1) Uniformity_w-Tal extrahieren (ein Punkt pro Waist-Spalte) ...")
    waist_all, width_all, n_boundary = extract_valley(results)
    if len(waist_all) == 0:
        width_vals = results["width_vals"]
        print(f"   FEHLER: Alle {n_boundary} Spalten (win_input-Werte) wurden als "
              f"Scan-Fenster-Artefakt verworfen - es bleibt KEIN einziger Talpunkt "
              f"uebrig, ein Fit ist damit nicht moeglich.\n"
              f"   Das bedeutet: das Uniformity_w-Minimum liegt in JEDER Spalte am "
              f"Rand des gescannten width-Fensters [{width_vals.min()*1e-6:.2f}, "
              f"{width_vals.max()*1e-6:.2f}] MHz - das wahre Optimum liegt fuer "
              f"diesen Datensatz komplett AUSSERHALB des gescannten width-Bereichs "
              f"(z.B. bei einem Atom-Offset, der die optimale width-Spannweite stark "
              f"nach oben oder unten verschiebt). Abhilfe: den Scan fuer diesen "
              f"Offset mit einem breiteren width-Bereich wiederholen, bis das echte "
              f"Minimum innerhalb des Fensters liegt.")
        return None
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

    if force_model is not None:
        # NEU (2026-08-27, auf User-Wunsch): Modell erzwungen statt
        # automatisch gewaehlt - die Block-CV oben laeuft trotzdem (rein
        # informativ, im Log sichtbar), entscheidet hier aber nicht mehr.
        best_name = force_model
        if best_name in cv_results and np.isfinite(cv_results[best_name]["r2_mean"]):
            print(f"   -> Modell ERZWUNGEN (force_model='{best_name}'): R²(Block-CV) = "
                  f"{cv_results[best_name]['r2_mean']:.4f} +/- {cv_results[best_name]['r2_std']:.4f} "
                  f"(automatische Modellwahl oben nur zu Informationszwecken - siehe FORCE_MODEL).")
        else:
            print(f"   -> Modell ERZWUNGEN (force_model='{best_name}') - Block-CV fuer dieses "
                  f"Modell hier nicht auswertbar (zu wenige Punkte pro Block?), wird trotzdem verwendet.")
    else:
        best_name_raw = max(valid, key=lambda n: cv_results[n]["r2_mean"])
        best_r2 = cv_results[best_name_raw]["r2_mean"]
        # Occam's razor: bei (annaehernd) gleich gutem Block-CV-R^2 gewinnt das
        # Modell mit den WENIGSTEN freien Parametern, nicht einfach das rein
        # numerische Maximum - siehe PARAM_COUNT weiter oben. Die Toleranz dafuer
        # ("annaehernd gleich gut") war frueher ein FESTER Wert (1e-3) - das
        # ignoriert, wie verrauscht die Block-CV selbst ist (nur n_blocks=5
        # Stichproben fuer r2_std). Bei sehr sauberen Datensaetzen (r2_std ~
        # 1e-4) ist 1e-3 eine sinnvolle Toleranz; bei verrauschteren Datensaetzen
        # (r2_std ~ 0.06, z.B. mit Atom-Offset) lag der feste 1e-3-Wert WEIT
        # innerhalb der CV-Streuung und liess die Modellwahl faktisch vom
        # CV-Stichprobenrauschen entscheiden - zwei kaum unterscheidbare Modelle
        # (hier: linear vs. power mit Exponent ~1.0) konnten dadurch je nach
        # Datensatz unterschiedlich "gewinnen", obwohl der Unterschied statistisch
        # nicht signifikant war. Fix: 1-Standardfehler-Regel (Standard in der
        # CV-Modellwahl, z.B. LASSO/Elastic-Net) - als "gleich gut" gilt jetzt
        # alles innerhalb des Standardfehlers des besten Modells
        # (r2_std / sqrt(n_folds_ok)), zusaetzlich zum alten 1e-3-Boden fuer den
        # Fall extrem kleiner Streuung.
        best_folds = max(cv_results[best_name_raw]["n_folds_ok"], 1)
        se_best = cv_results[best_name_raw]["r2_std"] / np.sqrt(best_folds)
        tol = max(1e-3, se_best)
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

    do_save = False
    if PLOT_FITS:
        print(f"\n4) Diagnose-Plot erzeugen (Ordner: {FIT_PLOTS_DIR}) ...")
        do_save = save
        if save and ask_before_save:
            try:
                antwort = input("Diagnose-Plot in 'Fit_Plots/um_waist' speichern? [y/N]: ").strip().lower()
                do_save = antwort in ("y", "yes", "j", "ja")
                if not do_save:
                    print("-> Bild wird NICHT gespeichert (nur angezeigt, falls show=True).")
            except EOFError:
                print(f"(ask_before_save=True, aber keine Eingabe möglich (kein Terminal) - "
                      f"verwende save={save} wie konfiguriert.)")
        plot_waist_width_fit(results, waist_used, width_used, waist_excl, width_excl,
                              best_name, popt, prefix=output_prefix, show=show, save=do_save,
                              best_point=best_point, legend_fontsize=legend_fontsize,
                              show_crosstalk=show_crosstalk, ask_before_save=ask_before_save)

    formula_doc = None
    if do_save:
        formula_doc = write_formula_doc(
            output_prefix, best_name, popt, cv_results[best_name]["r2_mean"],
            cv_results[best_name]["r2_std"], r2_fulldata, waist_used, waist_excl, best_point,
            n_boundary=n_boundary, forced=(force_model is not None))

    predict_width_mhz, predict_waist_um = make_predictor(best_name, popt)
    print(f"\nFertig. Modell '{best_name}' gilt fuer waist_um in "
          f"[{waist_used.min():.2f}, {waist_used.max():.2f}] (der bereinigte Talbereich) - "
          f"ausserhalb ist Extrapolation, mit Vorsicht zu geniessen.")
    return dict(model=best_name, popt=popt, cv_results=cv_results,
                waist_used=waist_used, width_used=width_used,
                waist_excluded=waist_excl, width_excluded=width_excl,
                n_boundary=n_boundary,
                predict_width_mhz=predict_width_mhz, predict_waist_um=predict_waist_um,
                best_point=best_point, formula_doc=formula_doc,
                forced_model=force_model)


if __name__ == "__main__":
    main()

