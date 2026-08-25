"""
fit_central_amplitudes.py
==========================
(Frueher: fit_stripe_closed_form.py - umbenannt, weil dieses Skript
speziell die AMPLITUDEN r_x/r_y aus dem amplituden-optimierten Scan fittet
(mm-Waist VOR der Linse) - zur Abgrenzung von fit_waist_width_relation.py,
das den Waist-Width-Zusammenhang bei FESTER Amplitude behandelt, um-Waist
NACH der Linse.)

EIN Skript, das den kompletten Fit-Prozess für die geschlossenen r_x/r_y-
Formeln im "zentralen Diagonalstreifen" enthält: pkl-Datei einlesen ->
Streifen-Bereich bestimmen -> Polynom-Fit (mit Kreuzvalidierung) -> Formeln
plotten. Alles in einem Skript, direkt ausführbar (`python
fit_central_amplitudes.py`).

Hintergrund/Motivation: der vorherige Foschungsstand
(rx_smooth_formula.py / ry_smooth_formula.py in Fit_Results\\) enthielt NUR
das fertige Ergebnis (hartcodierte Koeffizienten) - der Fit-Prozess selbst
lief in Wegwerf-Skripten, die nicht mitgeliefert wurden. Dieses Skript holt
das nach: es reproduziert den gesamten Weg von den Rohdaten bis zu den
Koeffizienten, nachvollziehbar und exakt wiederholbar (deterministisch -
gleiche Eingabedatei => gleiche Koeffizienten).

Was macht der Fit-Prozess konkret (siehe Funktionen unten, in dieser
Reihenfolge aufgerufen)?

1. `load_amp_scan_results()` (aus weighted_multitone_amplitude_dependence_plots.py)
   laedt die .pkl-Datei (r_x_grid, r_y_grid, win_input_vals, width_vals).

2. `find_stripe_mask()` bestimmt den "zentralen Diagonalstreifen":
   - `detect_amp_discontinuities()` (ebenfalls aus dem Plots-Modul) markiert
     Gitterpunkte, an denen (r_x, r_y) sprunghaft von ihren Nachbarn abweicht.
   - Die beiden groessten Zusammenhangskomponenten dieser Maske sind die
     Haupt-Ridge (die diagonale Sprungkante) und ein isoliertes Artefakt
     (typischerweise oben rechts in der Heatmap) - beide werden (mit ein paar
     Gitterzellen Sicherheitspuffer) aus der Flaeche entfernt.
   - Von den verbleibenden Punkten bildet der GROESSTE zusammenhaengende Rest
     den "zentralen Diagonalstreifen" - dort saettigen weder r_x noch r_y
     (keine Naehe zu den r_bounds-Schranken), im Gegensatz zu den kleineren
     Rest-Flaechen (Saettigungs-Ecken), die automatisch mit ausgeschlossen
     werden.

3. `cv_sweep_degree_alpha()` waehlt per raeumlicher Block-Kreuzvalidierung
   (Gitter in 5x5 Bloecke geteilt, GroupKFold, damit ganze zusammenhaengende
   Gebiete nie gleichzeitig in Training UND Test liegen) automatisch den
   besten Polynomgrad und die beste Ridge-Regularisierungsstaerke fuer
   log(r_x) bzw. log(r_y) - kein manuell geratener Grad.

4. `fit_and_distill()` fittet das finale Modell (PolynomialFeatures +
   StandardScaler + Ridge, auf log(r)) auf ALLEN Streifen-Punkten und
   "distilliert" es anschliessend in eine algebraisch exakte, direkt
   lesbare Polynom-Formel in den ROHEN Koordinaten (waist_mm, width_MHz) -
   per kleinste-Quadrate-Fit auf einem dichten Hilfsgitter (Rekonstruktions-
   fehler typischerweise < 1e-10, siehe Konsolen-Ausgabe).

5. `plot_stripe_fit()` / `plot_stripe_overview()` erzeugen die Diagnose-Plots
   (Rohdaten vs. Formel vs. Residuen; Uebersicht ueber Streifen/Ridge/
   Artefakt) und speichern sie in Fit_Plots\\mm_waist\\ (mm-Waist VOR der
   Linse, der tatsaechliche Scan-Parameter - getrennt von
   Fit_Plots\\um_waist\\, wo fit_waist_width_relation.py seine Plots ablegt;
   identisches Dateinamensschema wie der Rest des Projekts, siehe
   namenskonvention_fit_outputs.md). Alle
   Plot-Beschriftungen sind auf ENGLISCH, das Layout ist ein kompaktes
   2x2-Gitter (statt 1x4) mit groesseren Schriftgroessen, und gespeichert
   wird als Vektor-PDF (nicht PNG) - alles darauf ausgelegt, die Plots direkt
   per `\\includegraphics{...}` in ein LaTeX-Dokument einzubetten, ohne dass
   Beschriftungen beim Skalieren auf Spaltenbreite unleserlich klein werden.

Benoetigt zusaetzlich zu den ueblichen Projekt-Abhaengigkeiten: scikit-learn
(`pip install scikit-learn`) fuer PolynomialFeatures/StandardScaler/Ridge/
GroupKFold.

Nutzung:
    python fit_central_amplitudes.py
(vorher ggf. PKL_DATEI unten anpassen). Ausgabe: Konsole (Koeffizienten, R²,
Fit-Formeln) UND, sofern PLOT_FITS=True (Default), Diagnose-Plots in
Fit_Plots\\mm_waist\\ neben diesem Skript.

Kann auch importiert werden, z.B.:
    from fit_central_amplitudes import load_amp_scan_results, find_stripe_mask, \\
        cv_sweep_degree_alpha, fit_and_distill, make_predictor
    results = load_amp_scan_results("scan_amp_data_weighted_....pkl")
    stripe, waist_mm, width_MHz, extra = find_stripe_mask(results)
    best = cv_sweep_degree_alpha(results["r_x_grid"], stripe, waist_mm, width_MHz)
    terms, coef, info = fit_and_distill(results["r_x_grid"], stripe, waist_mm, width_MHz,
                                         best["degree"], best["alpha"])
    predict_rx = make_predictor(terms, coef)
    r_x = predict_rx(1.2, 0.30)
"""
from pathlib import Path
from datetime import date

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

from weighted_multitone_amplitude_dependence_plots import (
    load_amp_scan_results,
    detect_amp_discontinuities,
    resolve_save_path,
    DEFAULT_RESULTS_DIR,
)

# Matplotlib-Stil fuer Plots, die 1:1 in ein LaTeX-Dokument eingebettet werden:
# serifenbetonte Schrift (passt zu Computer Modern/Times, dem LaTeX-Standard),
# und Schriftgroessen, die auch nach dem Skalieren auf \textwidth noch lesbar
# sind (siehe PDF_FIGSIZE/PDF_RASTER_DPI unten - die Panels sind bewusst als
# 2x2-Gitter statt 1x4 angeordnet, damit der Skalierungsfaktor beim Einbetten
# nicht so extrem ausfaellt).
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


def _default_dir(name):
    """Legt einen Ordner neben DIESEM Skript an (bei Bedarf automatisch) -
    identisches Muster wie im Rest des Projekts."""
    candidate = Path(__file__).resolve().parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = Path(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# mm-Waist VOR der Linse (der tatsaechliche Scan-Parameter win_input) -
# eigener Unterordner, getrennt von Fit_Plots/um_waist (siehe
# fit_waist_width_relation.py), damit an der Ordnerstruktur sofort erkennbar
# ist, welche Waist-Konvention ein Plot verwendet.
FIT_PLOTS_DIR = _default_dir("Fit_Plots/mm_waist")
FIT_RESULTS_DIR = _default_dir("Fit_Results")


# ======================================================================
# Konfiguration - hier anpassen
# ======================================================================

# Gewichteter Amplituden-Scan (mit r_x_grid/r_y_grid) - Basis fuer den Fit.
PKL_DATEI = r"scan_amp_data_weighted_N3x4_151x151pts_Airy_2500res.pkl"

# Dateipraefix fuer alle Ausgaben (Formel-Dokument + Plots), siehe
# namenskonvention_fit_outputs.md - bei einer anderen PKL_DATEI entsprechend
# anpassen (N_x, N_y, Aufloesung, Profil, weighted/hart, roh/korrigiert, Datum).
OUTPUT_PREFIX = "AmpFit_N3x4_151x151pts_Airy_weighted_roh_2026-08-24"

# Sicherheitspuffer (Gitterzellen), um den Ridge/das Artefakt beim
# Streifen-Ausschluss aufzuweiten.
DILATE_ITERATIONS = 2

# Raeumliche Block-Kreuzvalidierung: n x n Bloecke, GroupKFold-Folds.
CV_BLOCKS_PER_AXIS = 5
CV_FOLDS = 5
CV_DEGREES = (2, 3, 4, 5, 6)
CV_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)

# Aufloesung des Hilfsgitters fuer die Distillation (roher Polynom-Fit).
DISTILL_GRID_N = 250

PLOT_FITS = True
SHOW = False
SAVE = True
ASK_BEFORE_SAVE = True

# Plot-Ausgabe: Vektor-PDF statt PNG (verlustfrei einbettbar/skalierbar in
# LaTeX, z.B. \includegraphics[width=\textwidth]{...}). PDF_FIGSIZE ist die
# physische Groesse der 2x2-Diagnose-Plots in Zoll (Breite, Hoehe) -
# absichtlich kompakt gewaehlt (statt der vorherigen 1x4-Anordnung), damit
# der Skalierungsfaktor beim Einbetten in eine Textspalte moderat bleibt und
# die oben gesetzten Schriftgroessen im gedruckten Dokument nicht winzig
# wirken. PDF_RASTER_DPI betrifft nur die rasterisierten imshow-Heatmaps
# innerhalb der PDF (Titel/Achsen/Ticks bleiben als Vektor-Text ohnehin in
# jeder Groesse gestochen scharf).
PDF_FIGSIZE = (9.0, 8.0)
PDF_RASTER_DPI = 300

# Residuum-Panel: "log" (Default) zeigt ln(r_data) - ln(r_fit) - das ist die
# Groesse, auf der tatsaechlich gefittet wurde (Ridge lernt log(r), siehe
# fit_and_distill()), und ist ausserdem naeherungsweise der RELATIVE Fehler
# (data-fit)/data, weil d(ln x) = dx/x - bei einer Groesse mit so grosser
# Spannweite (r_x/r_y reichen von ~0.86 bis ~2.3 im Streifen) aussagekraeftiger
# als eine absolute Differenz. "linear" zeigt stattdessen die einfache
# Differenz r_data - r_fit in den rohen Einheiten von r_x/r_y - nicht
# aufwendiger zu berechnen, nur eine andere Interpretation.
RESIDUAL_MODE = "log"  # "log" oder "linear"

# Grafische Feintuning-Optionen (auf User-Wunsch, per run_all_fits.py von
# aussen ueberschreibbar, siehe main()-Parameter unten):
# - LEGEND_FONTSIZE: Schriftgroesse der Legende, die NUR erscheint, wenn ein
#   bester Punkt eingezeichnet wird (siehe DRAW_BEST_POINT/_find_best_point).
# - DRAW_BEST_POINT: zeichnet den global besten Punkt (Minimum von
#   alpha*Uniformity + (1-alpha)*Crosstalk UEBER DEM GESAMTEN Scan-Gitter,
#   nicht nur im Streifen) als roten Stern ein - OHNE dass Koordinaten von
#   Hand angegeben werden muessen (siehe _find_best_point()). Default False,
#   da der Punkt fuer den reinen r_x/r_y-Formel-Fit nicht zwingend gebraucht
#   wird, aber fuer den Gesamtueberblick nuetzlich sein kann.
LEGEND_FONTSIZE = 9
DRAW_BEST_POINT = False

# NEU (auf User-Wunsch): Rohdaten-Uebersicht (Uniformity, Crosstalk, r_x,
# r_y - ALLE vier auf dem GESAMTEN Scan-Gitter, keine Streifen-Einschraenkung)
# als eigener Plot, BEVOR der eigentliche Fit laeuft (siehe main()). Default
# True, da sie ohne Zusatzaufwand aus denselben bereits geladenen Daten
# entsteht - ueber run_all_fits.py bzw. den main()-Parameter
# draw_dataset_overview abschaltbar.
DRAW_DATASET_OVERVIEW = True

# Bildstil fuer den optionalen Best-Point-Marker - identisch zu
# AmplitudeScanPlotter.BEST_POINT_STYLE in
# weighted_multitone_amplitude_dependence_plots.py (roter Stern, weisser
# Rand), damit ein "bester Punkt" im ganzen Projekt visuell gleich aussieht.
BEST_POINT_STYLE = dict(
    marker="*", markersize=16, markeredgecolor="white",
    markeredgewidth=1.1, color="red", zorder=8, linestyle="none",
)


# ======================================================================
# 0) Bester Punkt (optional, automatisch OHNE vorgegebene Koordinaten)
# ======================================================================
def _find_best_point(results):
    """Findet den global besten Punkt UEBER DEM GESAMTEN Scan-Gitter (nicht
    nur im Streifen) - dieselbe Definition wie
    AmplitudeScanPlotter._resolve_mark_point() im Rest des Projekts: Minimum
    von combined = alpha*Uniformity + (1-alpha)*Crosstalk, bevorzugt die
    atom-gewichteten Metriken (uniformity_weighted_grid/eta_weighted_grid),
    sonst die harten (uniformity_grid/crosstalk_grid). Braucht KEINE von
    Hand vorgegebenen Koordinaten - komplett automatisch.

    Gibt dict(waist_mm=..., width_MHz=..., label=...) zurueck, oder None,
    falls keine der beiden Metrik-Grid-Paare in results vorhanden ist (z.B.
    bei einem minimalen/reduzierten pkl ohne Uniformity/Crosstalk-Daten)."""
    has_weighted = "uniformity_weighted_grid" in results and "eta_weighted_grid" in results
    has_hard = "uniformity_grid" in results and "crosstalk_grid" in results
    if not has_weighted and not has_hard:
        return None

    if has_weighted:
        U, C = results["uniformity_weighted_grid"], results["eta_weighted_grid"]
        metric_tag = "atom-weighted"
    else:
        U, C = results["uniformity_grid"], results["crosstalk_grid"]
        metric_tag = "hard mask"
    alpha = results.get("alpha", 1.0)
    combined = alpha * U + (1 - alpha) * C
    if not np.any(np.isfinite(combined)):
        return None
    i, j = np.unravel_index(np.nanargmin(combined), combined.shape)

    win_input_vals = results["win_input_vals"]
    width_vals = results["width_vals"]
    waist_mm = float(win_input_vals[j]) * 1000.0
    width_MHz = float(width_vals[i]) / 1e6
    label = f"best point ({metric_tag}, global optimum)"
    return dict(waist_mm=waist_mm, width_MHz=width_MHz, label=label)


# ======================================================================
# 1) Streifen-Bereich bestimmen
# ======================================================================
def _meshgrid_mm_mhz(results):
    """Baut die (waist_mm, width_MHz)-Koordinatengitter aus win_input_vals/
    width_vals - dieselbe Umrechnung wie bisher in find_stripe_mask(), hier
    verselbststaendigt, damit plot_dataset_overview() (NEU, siehe unten) sie
    ohne vorherige Streifen-Bestimmung nutzen kann (die Uebersicht soll ja
    VOR dem eigentlichen Fit stehen, siehe main())."""
    win = np.asarray(results["win_input_vals"], dtype=float)
    width = np.asarray(results["width_vals"], dtype=float)
    WI, WW = np.meshgrid(win, width, indexing="xy")
    return WI * 1000.0, WW / 1e6  # waist_mm, width_MHz


def find_stripe_mask(results, z_thresh=3.5, min_neighbors=2, dilate=DILATE_ITERATIONS):
    """Bestimmt den "zentralen Diagonalstreifen": die groesste zusammen-
    haengende Flaeche, die uebrig bleibt, wenn man Ridge + isoliertes
    Artefakt (beide per detect_amp_discontinuities() gefunden, dann um
    `dilate` Gitterzellen aufgeweitet) aus der Flaeche entfernt.

    Rueckgabe: (stripe_mask, waist_mm, width_MHz, extra) - stripe_mask und
    die beiden Koordinaten-Gitter haben dieselbe Form wie r_x_grid/r_y_grid;
    extra ist ein dict mit Diagnose-Infos (ridge_mask, artifact_mask,
    component_sizes, ...) fuer die Uebersichts-Plots.
    """
    waist_mm, width_MHz = _meshgrid_mm_mhz(results)

    mask, jump_z = detect_amp_discontinuities(results, z_thresh=z_thresh, min_neighbors=min_neighbors)
    lbl, n_components = ndimage.label(mask, structure=np.ones((3, 3)))
    if n_components == 0:
        raise RuntimeError("Keine Diskontinuitaeten gefunden - Streifen-Logik setzt mind. eine Ridge voraus.")
    sizes = ndimage.sum(mask, lbl, range(1, n_components + 1))
    order = np.argsort(sizes)[::-1]

    ridge_mask = lbl == (order[0] + 1)
    artifact_mask = (lbl == (order[1] + 1)) if n_components > 1 else np.zeros_like(ridge_mask)

    ridge_dil = ndimage.binary_dilation(ridge_mask, iterations=dilate)
    artifact_dil = ndimage.binary_dilation(artifact_mask, iterations=dilate)
    background = ~ridge_dil & ~artifact_dil

    lbl2, n2 = ndimage.label(background, structure=np.ones((3, 3)))
    if n2 == 0:
        raise RuntimeError("Nach Entfernen von Ridge+Artefakt bleibt keine Flaeche mehr uebrig.")
    sizes2 = ndimage.sum(background, lbl2, range(1, n2 + 1))
    order2 = np.argsort(sizes2)[::-1]
    stripe_mask = lbl2 == (order2[0] + 1)

    extra = dict(
        ridge_mask=ridge_mask, artifact_mask=artifact_mask,
        ridge_dil=ridge_dil, artifact_dil=artifact_dil,
        n_components=n_components, component_sizes=sizes,
        stripe_size=int(stripe_mask.sum()), total_size=int(mask.size),
        n_background_components=n2, background_sizes=sizes2,
    )
    return stripe_mask, waist_mm, width_MHz, extra


# ======================================================================
# 2) Block-Kreuzvalidierung: bester Polynomgrad + Ridge-alpha
# ======================================================================
def _spatial_groups(stripe_mask, n_blocks):
    ny, nx = stripe_mask.shape
    ys, xs = np.where(stripe_mask)
    block_i = (ys * n_blocks) // ny
    block_j = (xs * n_blocks) // nx
    groups = block_i * n_blocks + block_j
    return ys, xs, groups


def cv_sweep_degree_alpha(target_grid, stripe_mask, waist_mm, width_MHz,
                           degrees=CV_DEGREES, alphas=CV_ALPHAS,
                           n_blocks=CV_BLOCKS_PER_AXIS, n_folds=CV_FOLDS, verbose=True):
    """Räumliche Block-Kreuzvalidierung (GroupKFold auf n_blocks x n_blocks
    Gitterbloecken) fuer log(target_grid), NUR auf den stripe_mask-Punkten.
    Testet alle (degree, alpha)-Kombinationen und gibt das beste Ergebnis
    zurueck: {"degree":..., "alpha":..., "r2_mean":..., "r2_std":..., "all": [...]}."""
    ys, xs, groups = _spatial_groups(stripe_mask, n_blocks)
    X = np.column_stack([waist_mm[ys, xs], width_MHz[ys, xs]])
    y = np.log(target_grid[ys, xs])

    all_results = []
    best = None
    for degree in degrees:
        for alpha in alphas:
            ug = np.unique(groups)
            if len(ug) < n_folds:
                continue
            gkf = GroupKFold(n_splits=n_folds)
            scores = []
            for tr, te in gkf.split(X, y, groups):
                if len(np.unique(groups[tr])) < 2 or len(te) < 5:
                    continue
                model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=alpha))
                model.fit(X[tr], y[tr])
                pred = model.predict(X[te])
                ss_res = np.sum((y[te] - pred) ** 2)
                ss_tot = np.sum((y[te] - y[te].mean()) ** 2)
                scores.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
            m, s = np.mean(scores), np.std(scores)
            all_results.append(dict(degree=degree, alpha=alpha, r2_mean=m, r2_std=s))
            if verbose:
                print(f"    degree={degree} alpha={alpha:<6g}  R2 = {m:.4f} +/- {s:.4f}")
            if best is None or (not np.isnan(m) and m > best["r2_mean"]):
                best = dict(degree=degree, alpha=alpha, r2_mean=m, r2_std=s)
    best["all"] = all_results
    return best


# ======================================================================
# 3) Finaler Fit + Distillation in eine geschlossene Polynom-Formel
# ======================================================================
def _poly_terms(degree):
    terms = []
    for total in range(degree + 1):
        for i in range(total + 1):
            j = total - i
            terms.append((i, j))
    return terms


def fit_and_distill(target_grid, stripe_mask, waist_mm, width_MHz, degree, alpha,
                     grid_n=DISTILL_GRID_N, domain=((0.8, 1.7), (0.2, 0.4))):
    """Fittet PolynomialFeatures+StandardScaler+Ridge (auf log(target)) auf
    ALLEN stripe_mask-Punkten, und "distilliert" das Ergebnis in eine
    algebraisch exakte Polynom-Formel in rohen (waist_mm, width_MHz)-
    Koordinaten (kleinste-Quadrate-Fit auf einem dichten Hilfsgitter).

    Rueckgabe: (terms, coef, info) - terms: Liste von (i, j)-Exponenten-
    Paaren, coef: dazugehoerige Koeffizienten (log(r) = sum coef * x^i * y^j),
    info: dict mit R²(volle Daten, NICHT CV - nur Sanity-Check) und dem
    maximalen Rekonstruktionsfehler ggue. dem Pipeline-Modell."""
    ys, xs = np.where(stripe_mask)
    X = np.column_stack([waist_mm[ys, xs], width_MHz[ys, xs]])
    y = np.log(target_grid[ys, xs])

    model = make_pipeline(PolynomialFeatures(degree), StandardScaler(), Ridge(alpha=alpha))
    model.fit(X, y)

    (x_lo, x_hi), (y_lo, y_hi) = domain
    gx = np.linspace(x_lo, x_hi, grid_n)
    gy = np.linspace(y_lo, y_hi, grid_n)
    GX, GY = np.meshgrid(gx, gy, indexing="xy")
    Xd = np.column_stack([GX.ravel(), GY.ravel()])
    yd_pred = model.predict(Xd)

    terms = _poly_terms(degree)
    A = np.column_stack([(Xd[:, 0] ** i) * (Xd[:, 1] ** j) for (i, j) in terms])
    coef, *_ = np.linalg.lstsq(A, yd_pred, rcond=None)

    recon_err = float(np.max(np.abs(A @ coef - yd_pred)))

    A_data = np.column_stack([(X[:, 0] ** i) * (X[:, 1] ** j) for (i, j) in terms])
    log_pred_data = A_data @ coef
    ss_res = np.sum((y - log_pred_data) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_fulldata = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    info = dict(degree=degree, alpha=alpha, n_terms=len(terms),
                max_distillation_error=recon_err, r2_full_data_sanity_check=r2_fulldata,
                n_train_points=len(ys))
    return terms, coef, info


def make_predictor(terms, coef):
    """Baut aus (terms, coef) eine aufrufbare predict(waist_mm, width_MHz)-Funktion."""
    def predict(waist_mm, width_MHz):
        x = np.atleast_1d(np.asarray(waist_mm, dtype=float))
        y = np.atleast_1d(np.asarray(width_MHz, dtype=float))
        log_r = np.zeros_like(x)
        for (i, j), c in zip(terms, coef):
            log_r = log_r + c * (x ** i) * (y ** j)
        r = np.exp(log_r)
        return r if r.size > 1 else float(r[0])
    return predict


def print_formula(name, terms, coef):
    print(f"\nlog({name}) = sum_ij c_ij * waist_mm^i * width_MHz^j  ({len(terms)} Terme)")
    for (i, j), c in zip(terms, coef):
        print(f"    c({i},{j}) = {c: .8f}")


def write_formula_doc(prefix, coefficients, extra, best_point,
                       out_dir=FIT_RESULTS_DIR, residual_mode=RESIDUAL_MODE):
    """Schreibt AUTOMATISCH ein Formel-Dokument (Markdown) mit den gerade
    gefundenen Koeffizienten/R²/Domaeneninfos - ersetzt das bisher von Hand
    gepflegte `..._Formeln.md` durch eine bei jedem Lauf frisch generierte
    Version (deterministisch, kein Nachpflegen mehr noetig). `coefficients`
    ist das dict aus main() (ein Eintrag je "r_x"/"r_y" mit terms/coef/cv/
    distill), `extra` die Streifen-Diagnose aus find_stripe_mask()."""
    lines = [
        f"# {prefix} — geschlossene r_x/r_y-Formeln (zentraler Diagonalstreifen)",
        "",
        f"Automatisch generiert von `fit_central_amplitudes.py` am {date.today().isoformat()}.",
        "",
        f"Streifen: {extra['stripe_size']}/{extra['total_size']} Punkte "
        f"({100 * extra['stripe_size'] / extra['total_size']:.1f}%), "
        f"Ridge: {int(extra['ridge_mask'].sum())} Punkte, "
        f"Artefakt: {int(extra['artifact_mask'].sum())} Punkte.",
        "",
    ]
    if best_point is not None:
        lines += [
            f"Bester Punkt (automatisch, {best_point['label']}): "
            f"waist = {best_point['waist_mm']:.4f} mm (vor der Linse), "
            f"width = {best_point['width_MHz']:.4f} MHz.",
            "",
        ]
    for name in ("r_x", "r_y"):
        if name not in coefficients:
            continue
        c = coefficients[name]
        cv, distill = c["cv"], c["distill"]
        lines += [
            f"## {name}",
            "",
            f"- Polynomgrad: {cv['degree']}, Ridge-alpha: {cv['alpha']}",
            f"- R²(Block-CV): {cv['r2_mean']:.4f} +/- {cv['r2_std']:.4f}",
            f"- R²(volle Streifen-Daten, NICHT CV, nur Sanity-Check): "
            f"{distill['r2_full_data_sanity_check']:.5f}",
            f"- Max. Distillations-Fehler ggue. Pipeline-Modell: "
            f"{distill['max_distillation_error']:.2e}",
            f"- Residuum-Modus im Diagnose-Plot: {residual_mode}",
            "",
            f"log({name}) = sum_ij c_ij * waist_mm^i * width_MHz^j "
            f"({len(c['terms'])} Terme)",
            "",
            "```",
        ]
        for (i, j), coef_val in zip(c["terms"], c["coef"]):
            lines.append(f"c({i},{j}) = {coef_val: .8f}")
        lines += ["```", ""]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Formel-Dokument soll bei jedem Lauf frisch ueberschrieben werden (es
    # ist reproduzierbar aus der pkl-Datei ableitbar, anders als die
    # Diagnose-Plots gibt es keinen Grund, alte Versionen mit _2/_3-Suffix
    # aufzuheben) - deshalb IMMER ueberschreiben statt interaktiv zu fragen.
    out_file = resolve_save_path(out_dir, f"{prefix}_Formeln.md", confirm_overwrite=lambda p: True)
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"Formel-Dokument gespeichert: {out_file}")
    return out_file


# ======================================================================
# 4) Plots
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


def plot_dataset_overview(results, waist_mm, width_MHz, out_dir=FIT_PLOTS_DIR, prefix=OUTPUT_PREFIX,
                           show=SHOW, save=SAVE, best_point=None, legend_fontsize=LEGEND_FONTSIZE,
                           ask_before_save=ASK_BEFORE_SAVE):
    """NEU (User-Wunsch): 2x2-Rohdaten-Uebersicht (Englisch, Vektor-PDF) fuer
    den amplituden-optimierten Scan - Uniformity, Crosstalk und BEIDE
    Amplituden r_x/r_y, alle auf dem GESAMTEN Scan-Gitter (keine
    Streifen-Einschraenkung wie bei plot_stripe_fit()/plot_stripe_overview()
    weiter unten, die NUR den zentralen Diagonalstreifen zeigen) - gedacht
    als Blick auf die Rohdaten-Landschaft, BEVOR der eigentliche Polynom-Fit
    passiert (siehe main(), wird dort als erster Plot erzeugt).

    Bevorzugt die atom-gewichteten Metriken (uniformity_weighted_grid/
    eta_weighted_grid), faellt sonst auf die harten (uniformity_grid/
    crosstalk_grid) zurueck - dieselbe Praeferenz wie _find_best_point().
    Gibt None zurueck (ohne zu plotten), falls keines der beiden Paare in
    der Datei vorhanden ist."""
    has_weighted = "uniformity_weighted_grid" in results and "eta_weighted_grid" in results
    has_hard = "uniformity_grid" in results and "crosstalk_grid" in results
    if not has_weighted and not has_hard:
        print("   (Kein Uniformity/Crosstalk-Grid in dieser Datei gefunden - "
              "Dataset-Overview-Plot wird ausgelassen.)")
        return None
    if has_weighted:
        U = results["uniformity_weighted_grid"] * 100.0
        C = results["eta_weighted_grid"] * 100.0
        metric_tag = "atom-weighted"
    else:
        U = results["uniformity_grid"] * 100.0
        C = results["crosstalk_grid"] * 100.0
        metric_tag = "hard mask"

    rx_grid = np.asarray(results["r_x_grid"], dtype=float)
    ry_grid = np.asarray(results["r_y_grid"], dtype=float)

    extent = [waist_mm.min(), waist_mm.max(), width_MHz.min(), width_MHz.max()]
    fig, axes = plt.subplots(2, 2, figsize=PDF_FIGSIZE)
    ax_u, ax_c, ax_rx, ax_ry = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    im0 = ax_u.imshow(U, origin="lower", extent=extent, aspect="auto", cmap="viridis_r")
    ax_u.set_title(f"Uniformity ({metric_tag})")
    plt.colorbar(im0, ax=ax_u, label=r"Uniformity ($\sigma/\mu$) (%)")

    im1 = ax_c.imshow(C, origin="lower", extent=extent, aspect="auto", cmap="magma_r")
    ax_c.set_title(f"Crosstalk ({metric_tag})")
    plt.colorbar(im1, ax=ax_c, label=r"Crosstalk ($\eta$) (%)")

    im2 = ax_rx.imshow(rx_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax_rx.set_title(r"r$_x$")
    plt.colorbar(im2, ax=ax_rx)

    im3 = ax_ry.imshow(ry_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax_ry.set_title(r"r$_y$")
    plt.colorbar(im3, ax=ax_ry)

    if best_point is not None:
        for ax in (ax_u, ax_c, ax_rx, ax_ry):
            ax.plot(best_point["waist_mm"], best_point["width_MHz"],
                    label=best_point["label"], **BEST_POINT_STYLE)
        ax_u.legend(loc="best", fontsize=legend_fontsize, framealpha=0.9)

    for ax in axes.ravel():
        ax.set_xlabel("waist (mm)")
        ax.set_ylabel("width (MHz)")
    plt.tight_layout()
    _finish_fig(fig, f"{prefix}_dataset_overview.pdf", out_dir, show, save, ask_before_save=ask_before_save)


def plot_stripe_fit(name, raw_grid, predict_fn, stripe_mask, waist_mm, width_MHz,
                     out_dir=FIT_PLOTS_DIR, prefix=OUTPUT_PREFIX, show=SHOW, save=SAVE,
                     residual_mode=RESIDUAL_MODE, best_point=None, legend_fontsize=LEGEND_FONTSIZE,
                     ask_before_save=ASK_BEFORE_SAVE):
    """2x2-Diagnose-Plot (Englisch, Vektor-PDF): r_x/r_y (voller Scan),
    r_x/r_y Data (nur Streifen), r_x/r_y Fit (nur Streifen), Residuum
    "Data - Fit" - je nach residual_mode entweder in log-Einheiten
    (Default; log(r) wurde gefittet, daher IST das die tatsaechliche
    Fit-Residuum-Groesse - kein separates ln(...) noetig, es ist per
    Definition schon "Data - Fit" im gefitteten Raum) oder in rohen
    r_x/r_y-Einheiten (residual_mode="linear").

    Beschriftung: "r_x"/"r_y" wird als r$_x$/r$_y$ gesetzt (nur die
    Subskript-Ziffer in Mathtext, "r" bleibt aufrechte Textschrift) - das
    ist die uebliche Text-Umgebung-Schreibweise (r\\textsubscript{x} in
    LaTeX), passender fuer gemischte Titel wie "r_x Data" als ein
    durchgehend kursiv gesetztes "$r_x$"."""
    pred = predict_fn(waist_mm.ravel(), width_MHz.ravel()).reshape(raw_grid.shape)
    if residual_mode == "log":
        resid = np.log(raw_grid) - np.log(pred)
    elif residual_mode == "linear":
        resid = raw_grid - pred
    else:
        raise ValueError("residual_mode muss 'log' oder 'linear' sein.")
    resid_title = "Data - Fit"
    masked_raw = np.where(stripe_mask, raw_grid, np.nan)
    masked_pred = np.where(stripe_mask, pred, np.nan)
    masked_resid = np.where(stripe_mask, resid, np.nan)

    sym = r"r$_x$" if name == "r_x" else r"r$_y$"
    extent = [waist_mm.min(), waist_mm.max(), width_MHz.min(), width_MHz.max()]
    fig, axes = plt.subplots(2, 2, figsize=PDF_FIGSIZE)
    ax_raw, ax_data, ax_formula, ax_resid = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    im0 = ax_raw.imshow(raw_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax_raw.set_title(sym)
    plt.colorbar(im0, ax=ax_raw)
    if best_point is not None:
        ax_raw.plot(best_point["waist_mm"], best_point["width_MHz"],
                    label=best_point["label"], **BEST_POINT_STYLE)
        ax_raw.legend(loc="best", fontsize=legend_fontsize, framealpha=0.9)

    vmin, vmax = np.nanmin(masked_raw), np.nanmax(masked_raw)
    im1 = ax_data.imshow(masked_raw, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_data.set_title(f"{sym} Data")
    plt.colorbar(im1, ax=ax_data)

    im2 = ax_formula.imshow(masked_pred, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_formula.set_title(f"{sym} Fit")
    plt.colorbar(im2, ax=ax_formula)

    rmax = np.nanmax(np.abs(masked_resid))
    im3 = ax_resid.imshow(masked_resid, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-rmax, vmax=rmax)
    ax_resid.set_title(resid_title)
    plt.colorbar(im3, ax=ax_resid)

    for ax in axes.ravel():
        ax.set_xlabel("waist (mm)")
        ax.set_ylabel("width (MHz)")
    plt.tight_layout()
    _finish_fig(fig, f"{prefix}_{name}_smoothformula.pdf", out_dir, show, save, ask_before_save=ask_before_save)

    rms = float(np.sqrt(np.nanmean(masked_resid ** 2)))
    print(f"{name}: RMS(Residuum, {residual_mode}) = {rms:.5f}   max|Residuum| = {rmax:.5f}")
    return rms, float(rmax)


def plot_stripe_overview(rx_grid, ry_grid, stripe_mask, waist_mm, width_MHz, extra,
                          out_dir=FIT_PLOTS_DIR, prefix=OUTPUT_PREFIX, show=SHOW, save=SAVE,
                          best_point=None, legend_fontsize=LEGEND_FONTSIZE,
                          ask_before_save=ASK_BEFORE_SAVE):
    """2x2-Uebersichtsplot (Englisch, LaTeX-tauglich als Vektor-PDF): r_x/r_y
    roh + Streifen(gruen)/Ridge(rot)/Artefakt(blau)."""
    extent = [waist_mm.min(), waist_mm.max(), width_MHz.min(), width_MHz.max()]
    fig, axes = plt.subplots(2, 2, figsize=PDF_FIGSIZE)
    ax_rx, ax_ry, ax_stripe, ax_overlay = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    im0 = ax_rx.imshow(rx_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax_rx.set_title(r"r$_x$"); plt.colorbar(im0, ax=ax_rx)
    im1 = ax_ry.imshow(ry_grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax_ry.set_title(r"r$_y$"); plt.colorbar(im1, ax=ax_ry)
    if best_point is not None:
        for ax in (ax_rx, ax_ry):
            ax.plot(best_point["waist_mm"], best_point["width_MHz"],
                    label=best_point["label"], **BEST_POINT_STYLE)
        ax_rx.legend(loc="best", fontsize=legend_fontsize, framealpha=0.9)
    ax_stripe.imshow(stripe_mask, origin="lower", extent=extent, aspect="auto", cmap="Greens")
    ax_stripe.set_title(f"central diagonal stripe (n={int(stripe_mask.sum())})")

    overlay = np.zeros(rx_grid.shape + (3,))
    overlay[..., 0] = extra["ridge_dil"]
    overlay[..., 1] = stripe_mask * 0.6
    overlay[..., 2] = extra["artifact_dil"]
    ax_overlay.imshow(overlay, origin="lower", extent=extent, aspect="auto")
    ax_overlay.set_title("red = ridge, green = stripe, blue = artifact")
    for ax in axes.ravel():
        ax.set_xlabel("waist (mm)"); ax.set_ylabel("width (MHz)")
    plt.tight_layout()
    _finish_fig(fig, f"{prefix}_stripe_overview.pdf", out_dir, show, save, ask_before_save=ask_before_save)


# ======================================================================
# main: alles zusammen - laden, Streifen bestimmen, fitten, plotten
# ======================================================================
def main(pkl_datei=None, output_prefix=None, draw_best_point=None, legend_fontsize=None,
         ask_before_save=None, save=None, show=None, draw_dataset_overview=None):
    """Alle Parameter sind optional - None faellt auf die Modul-Konfiguration
    oben zurueck (PKL_DATEI/OUTPUT_PREFIX/DRAW_BEST_POINT/LEGEND_FONTSIZE/
    ASK_BEFORE_SAVE/SAVE/SHOW/DRAW_DATASET_OVERVIEW). Explizite Werte
    ueberschreiben das gezielt - genutzt von run_all_fits.py, um z.B.
    ASK_BEFORE_SAVE=False (kein Terminal-Prompt, "ein Klick and go") oder
    DRAW_BEST_POINT/LEGEND_FONTSIZE zentral von aussen zu setzen, ohne die
    Modul-Konstanten direkt zu patchen."""
    pkl_datei = PKL_DATEI if pkl_datei is None else pkl_datei
    output_prefix = OUTPUT_PREFIX if output_prefix is None else output_prefix
    draw_best_point = DRAW_BEST_POINT if draw_best_point is None else draw_best_point
    legend_fontsize = LEGEND_FONTSIZE if legend_fontsize is None else legend_fontsize
    ask_before_save = ASK_BEFORE_SAVE if ask_before_save is None else ask_before_save
    save = SAVE if save is None else save
    show = SHOW if show is None else show
    draw_dataset_overview = DRAW_DATASET_OVERVIEW if draw_dataset_overview is None else draw_dataset_overview

    print(f"Lade '{pkl_datei}' ...")
    try:
        results = load_amp_scan_results(pkl_datei)
    except FileNotFoundError:
        vorhandene = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob("scan_amp_data_weighted_*.pkl"))
        print(f"'{pkl_datei}' wurde weder im aktuellen Ordner noch in '{DEFAULT_RESULTS_DIR}' gefunden.")
        if vorhandene:
            print("Vorhandene Dateien:")
            for name in vorhandene:
                print(f"  - {name}")
        return None

    rx_grid = np.asarray(results["r_x_grid"], dtype=float)
    ry_grid = np.asarray(results["r_y_grid"], dtype=float)

    print("\n1) Bestimme zentralen Diagonalstreifen (Ridge + Artefakt ausschliessen) ...")
    stripe_mask, waist_mm, width_MHz, extra = find_stripe_mask(results)
    n_stripe, n_total = extra["stripe_size"], extra["total_size"]
    print(f"   Ridge: {int(extra['ridge_mask'].sum())} Punkte, "
          f"Artefakt: {int(extra['artifact_mask'].sum())} Punkte, "
          f"Streifen: {n_stripe}/{n_total} Punkte ({100*n_stripe/n_total:.1f}%)")

    best_point = _find_best_point(results) if draw_best_point else None
    if draw_best_point:
        if best_point is not None:
            print(f"   Bester Punkt (automatisch): waist={best_point['waist_mm']:.4f} mm, "
                  f"width={best_point['width_MHz']:.4f} MHz ({best_point['label']})")
        else:
            print("   DRAW_BEST_POINT=True, aber weder gewichtete noch harte "
                  "Uniformity/Crosstalk-Grids in dieser Datei gefunden - kein Punkt eingezeichnet.")

    # Speichern-Entscheidung EINMAL hier treffen (statt wie vorher erst kurz
    # vor plot_stripe_fit/plot_stripe_overview) - gilt jetzt fuer ALLE Plots
    # inkl. der neuen Dataset-Overview weiter unten, damit nicht zweimal
    # nachgefragt wird und alle Plots konsistent behandelt werden.
    do_save = save
    if PLOT_FITS and save and ask_before_save:
        try:
            antwort = input("Diagnose-Plots in 'Fit_Plots/mm_waist' speichern? [y/N]: ").strip().lower()
            do_save = antwort in ("y", "yes", "j", "ja")
            if not do_save:
                print("-> Bilder werden NICHT gespeichert (nur angezeigt, falls show=True).")
        except EOFError:
            print(f"(ask_before_save=True, aber keine Eingabe möglich (kein Terminal) - "
                  f"verwende save={save} wie konfiguriert.)")
    elif not PLOT_FITS:
        do_save = False

    if PLOT_FITS and draw_dataset_overview:
        print(f"\nErzeuge Datensatz-Uebersicht (Uniformity/Crosstalk/r_x/r_y, gesamtes Gitter, "
              f"VOR dem Fit) ...")
        plot_dataset_overview(results, waist_mm, width_MHz, prefix=output_prefix,
                               show=show, save=do_save, best_point=best_point,
                               legend_fontsize=legend_fontsize, ask_before_save=ask_before_save)

    predictors = {}
    coefficients = {}
    for name, grid in (("r_x", rx_grid), ("r_y", ry_grid)):
        print(f"\n2) Block-Kreuzvalidierung ({name}): Grad x Ridge-alpha durchsuchen ...")
        best = cv_sweep_degree_alpha(grid, stripe_mask, waist_mm, width_MHz)
        print(f"   -> bester Grad = {best['degree']}, alpha = {best['alpha']}, "
              f"R²(Block-CV) = {best['r2_mean']:.4f} +/- {best['r2_std']:.4f}")

        print(f"3) Finaler Fit + Distillation in geschlossene Formel ({name}) ...")
        terms, coef, info = fit_and_distill(grid, stripe_mask, waist_mm, width_MHz,
                                             best["degree"], best["alpha"])
        print(f"   {info['n_terms']} Terme, max. Distillations-Fehler = {info['max_distillation_error']:.2e}, "
              f"R²(volle Streifen-Daten, NICHT CV, nur Sanity-Check) = {info['r2_full_data_sanity_check']:.5f}")
        print_formula(name, terms, coef)

        predictors[name] = make_predictor(terms, coef)
        coefficients[name] = dict(terms=terms, coef=coef.tolist(), cv=best, distill=info)

    if PLOT_FITS:
        print(f"\n4) Diagnose-Plots erzeugen (Ordner: {FIT_PLOTS_DIR}) ...")
        # do_save wurde bereits weiter oben bestimmt (vor der Dataset-Overview) -
        # hier nur noch wiederverwendet, damit nicht ein zweites Mal nachgefragt wird.
        plot_stripe_fit("r_x", rx_grid, predictors["r_x"], stripe_mask, waist_mm, width_MHz,
                         prefix=output_prefix, show=show, save=do_save,
                         best_point=best_point, legend_fontsize=legend_fontsize,
                         ask_before_save=ask_before_save)
        plot_stripe_fit("r_y", ry_grid, predictors["r_y"], stripe_mask, waist_mm, width_MHz,
                         prefix=output_prefix, show=show, save=do_save,
                         best_point=best_point, legend_fontsize=legend_fontsize,
                         ask_before_save=ask_before_save)
        plot_stripe_overview(rx_grid, ry_grid, stripe_mask, waist_mm, width_MHz, extra,
                              prefix=output_prefix, show=show, save=do_save,
                              best_point=best_point, legend_fontsize=legend_fontsize,
                              ask_before_save=ask_before_save)

    formula_doc = None
    if do_save:
        formula_doc = write_formula_doc(output_prefix, coefficients, extra, best_point)

        npz_path = FIT_RESULTS_DIR / "stripe_domain_mask.npz"
        np.savez(npz_path, waist_mm=waist_mm[0, :], width_MHz=width_MHz[:, 0], stripe_mask=stripe_mask)
        print(f"Streifen-Maske gespeichert: {npz_path}")

    print("\nFertig. r_x/r_y sind jetzt NUR innerhalb des zentralen Diagonalstreifens gueltig "
          "(siehe Konsolen-Ausgabe/Plots oben) - ausserhalb (Ridge, Saettigungs-Ecken, Artefakt) "
          "bitte gpr_amp_predict.py (GPR, gilt ueberall) verwenden.")
    return dict(predictors=predictors, coefficients=coefficients, stripe_mask=stripe_mask,
                waist_mm=waist_mm, width_MHz=width_MHz, extra=extra, best_point=best_point,
                formula_doc=formula_doc)


if __name__ == "__main__":
    main()
