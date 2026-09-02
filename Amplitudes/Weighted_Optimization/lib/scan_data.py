"""
lib/scan_data.py - Datensaetze laden, einordnen, auswerten.

Der rechnerische Kern hinter run_plots.py: aus den beiden Metrik-Gittern
eines Scans werden Score, bester Punkt und Region bestimmt, dazu kommen der
verbotene Bereich (ueberlappende Eck-Spots) und das Laden/Speichern.

Diese Datei ist in Hard_Optimization/lib und Weighted_Optimization/lib
IDENTISCH - alles Ordner-Spezifische (welche Gitter-Schluessel, welche
Dateinamens-Muster, welches Plot-Modul) steht in paths.py.

Zwei Datensatz-Arten kommen vor:

    "fixed"  feste Amplituden, ein Scan ueber (win_input, width)
             -> run_scan.py, scan_data*.pkl
    "amp"    pro Gitterpunkt eine eigene (r_x, r_y)-Optimierung
             -> run_amp_scan.py, scan_amp_data*.pkl

Der Score ist in beiden Faellen dieselbe Groesse, die auch der Optimierer
benutzt:

    J = alpha * Uniformity + (1 - alpha) * Crosstalk

- also ROH, ohne gitterweite Normierung. (Der Combinated_Optimization-Ordner
hat 2026-09-01 aus denselben Gruenden auf die rohe Groesse umgestellt: eine
gitterweite Min-Max-Normierung haengt am gescannten Fenster, dieselbe Physik
ergaebe bei anderem Scan-Bereich andere Zahlen.)
"""

import pickle
from pathlib import Path as FilePath

import numpy as np

from . import paths
from .paths import width_to_um, win_input_to_win


# ======================================================================
# Welche Gitter hat dieser Datensatz?
# ======================================================================
KIND_LABELS = {
    "amp": "Amplituden-Scan (r_x/r_y je Gitterpunkt optimiert)",
    "fixed": "Fest-Amplituden-Scan (feste Amplituden, nur win/width variiert)",
    "unknown": "unbekannt",
}

# Welche Metrik-Familie im Datensatz steckt - unabhaengig davon, in welchem
# Ordner er liegt. Der eigene Ordner (paths.FLAVOR) hat Vorrang; ein
# Datensatz aus dem jeweils anderen Ordner (oder ein kombinierter mit beiden
# Familien) laesst sich damit trotzdem plotten.
FLAVOR_KEYS = {
    "hard": ("uniformity_grid", "crosstalk_grid"),
    "weighted": ("uniformity_weighted_grid", "eta_weighted_grid"),
}
FLAVOR_LABELS = {"hard": "hart (globale Maske)", "weighted": "atom-gewichtet"}


def flavor_of(results):
    """'hard' oder 'weighted' - welche Metrik-Familie dieser Datensatz
    hergibt. Bevorzugt die des eigenen Ordners; None, wenn keine da ist."""
    kandidaten = [paths.FLAVOR] + [f for f in FLAVOR_KEYS if f != paths.FLAVOR]
    for name in kandidaten:
        u_key, c_key = FLAVOR_KEYS[name]
        if results.get(u_key) is not None and results.get(c_key) is not None:
            return name
    return None


def metric_keys(results):
    """(uniformity_key, crosstalk_key) dieses Datensatzes."""
    name = flavor_of(results)
    if name is None:
        raise ValueError(
            "Der Datensatz enthaelt weder ein hartes noch ein atom-gewichtetes "
            "Metrik-Paar - das sieht nicht nach einem Scan dieses Projekts aus.")
    return FLAVOR_KEYS[name]


def metric_grids(results):
    """(Uniformity, Crosstalk) als float-Arrays."""
    u_key, c_key = metric_keys(results)
    return (np.asarray(results[u_key], dtype=float),
            np.asarray(results[c_key], dtype=float))


def has_amplitudes(results):
    return (results.get('r_x_grid') is not None
            and results.get('r_y_grid') is not None)


def dataset_kind(results):
    """'amp', 'fixed' oder 'unknown'."""
    if flavor_of(results) is None:
        return "unknown"
    if not ('win_input_vals' in results and 'width_vals' in results):
        return "unknown"
    return "amp" if has_amplitudes(results) else "fixed"


def is_foreign(results):
    """True, wenn der Datensatz die Metriken des ANDEREN Ordners mitbringt -
    dann wurde vermutlich die falsche Datei gewaehlt, und das sagt die GUI
    auch."""
    name = flavor_of(results)
    return name is not None and name != paths.FLAVOR


# ======================================================================
# Score, bester Punkt, Region
# ======================================================================
DEFAULT_PERCENTILE = 25.0


def score_from(U, C, alpha):
    """Die Groesse, gegen die auch optimiert wird: alpha*U + (1-alpha)*C.
    Roh, ohne gitterweite Normierung."""
    return alpha * np.asarray(U, dtype=float) + (1.0 - alpha) * np.asarray(C, dtype=float)


def largest_rectangle(mask):
    """Groesstes achsenparalleles Rechteck vollstaendig innerhalb der
    True-Punkte von `mask`.

    Gibt (flaeche, (row0, row1, col0, col1)) zurueck, Grenzen INKLUSIVE;
    bei leerer Maske (0, None). Standard-Verfahren: zeilenweise Histogramm
    der Saeulenhoehen, darin per Stack das groesste Rechteck.
    """
    mask = np.asarray(mask, dtype=bool)
    n_rows, n_cols = mask.shape
    best_area = 0
    best_bounds = None
    heights = np.zeros(n_cols, dtype=int)

    for i in range(n_rows):
        heights = np.where(mask[i], heights + 1, 0)
        stack = []  # (start_index, height)
        for j in range(n_cols + 1):
            current = int(heights[j]) if j < n_cols else 0
            start = j
            while stack and stack[-1][1] >= current:
                idx, ht = stack.pop()
                area = ht * (j - idx)
                if area > best_area:
                    best_area = area
                    best_bounds = (i - ht + 1, i, idx, j - 1)
                start = idx
            stack.append((start, current))

    return best_area, best_bounds


def analyse(results, alpha=None, percentile=None):
    """Score, bester Punkt und Region aus den vorhandenen Gittern.

    Gibt ein neues dict zurueck (das Original bleibt unangetastet), in dem
    zusaetzlich stehen:

        score        das Gitter alpha*U + (1-alpha)*C
        best         bester Gitterpunkt nach diesem Score, inkl. at_edge
        region       groesstes Rechteck in den besten `percentile`%
        alpha, combo_percentile

    Der urspruenglich gespeicherte `best` (nur bei Fest-Amplituden-Scans
    vorhanden) bleibt als `best_stored` erhalten - so kann der Bericht
    ehrlich sagen, welcher Punkt gemeint ist.
    """
    results = dict(results)
    alpha = float(results.get('alpha', 0.7)) if alpha is None else float(alpha)
    percentile = (float(results.get('combo_percentile', DEFAULT_PERCENTILE))
                  if percentile is None else float(percentile))

    U, C = metric_grids(results)
    win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
    width_vals = np.asarray(results['width_vals'], dtype=float)
    score = score_from(U, C, alpha)
    finite = np.isfinite(score)

    if 'best_stored' not in results and results.get('best'):
        results['best_stored'] = dict(results['best'])

    best = dict(win_input=None, width=None, uniformity=None, crosstalk=None,
                score=None, row=None, col=None, at_edge=False)
    if np.any(finite):
        i, j = np.unravel_index(
            int(np.argmin(np.where(finite, score, np.inf))), score.shape)
        best = dict(
            win_input=float(win_input_vals[j]), width=float(width_vals[i]),
            uniformity=float(U[i, j]), crosstalk=float(C[i, j]),
            score=float(score[i, j]), row=int(i), col=int(j),
            # Liegt der beste Punkt auf dem Rand des gescannten Fensters,
            # ist er vermutlich gar kein Optimum, sondern nur der Rand.
            at_edge=bool(i in (0, score.shape[0] - 1)
                         or j in (0, score.shape[1] - 1)),
        )
        if has_amplitudes(results):
            best['r_x'] = float(np.asarray(results['r_x_grid'], dtype=float)[i, j])
            best['r_y'] = float(np.asarray(results['r_y_grid'], dtype=float)[i, j])

    region = dict(win_input_min=None, win_input_max=None, width_min=None,
                  width_max=None, mask=np.zeros_like(score, dtype=bool),
                  threshold=None, n_points_total=int(score.size),
                  n_points_region=0, row_bounds=None, col_bounds=None)
    if np.any(finite):
        threshold = float(np.nanpercentile(score[finite], percentile))
        mask = finite & (score <= threshold)
        area, bounds = largest_rectangle(mask)
        region.update(mask=mask, threshold=threshold,
                      n_points_region=int(mask.sum()))
        if bounds is not None and area > 0:
            r0, r1, c0, c1 = bounds
            region.update(
                win_input_min=float(win_input_vals[c0]),
                win_input_max=float(win_input_vals[c1]),
                width_min=float(width_vals[r0]), width_max=float(width_vals[r1]),
                row_bounds=(int(r0), int(r1)), col_bounds=(int(c0), int(c1)),
            )

    results.update(score=score, best=best, region=region,
                   alpha=alpha, combo_percentile=percentile)
    return results


# ======================================================================
# Verbotener Bereich: ueberlappende Eck-Spots
# ======================================================================
# Wortwoertlich dieselbe Herleitung wie in
# Combinated_Optimization/lib/combine.py - hier noch einmal, damit dieser
# Ordner ohne den anderen auskommt.
#
# Die beiden diagonal gegenueberliegenden ECK-Spots des N_x x N_y-Arrays
# duerfen sich nicht ueberlappen. `width` ist die GESAMTSPANNWEITE des
# Tonarrays in Frequenz - und zwar dieselbe fuer x und y. Raeumlich
# entspricht das der Kantenlaenge
#
#     S(width) = width_to_um(width, ...)                          [µm]
#
# Die beiden Ecken liegen damit bei (0, 0) und (S, S), ihr Abstand ist
# d = sqrt(2) * S. Nicht-Ueberlappung heisst d > k * waist (k = 2 als
# Default: die beiden Spot-Radien beruehren sich dann gerade). S ist linear
# in width, die Bedingung ist deshalb in der (waist, width)-Ebene exakt eine
# Ursprungsgerade:
#
#     width/MHz > a * waist/µm     mit     a = k / (sqrt(2) * u)
#
# Der verbotene Bereich liegt UNTERHALB dieser Geraden: grosser Waist bei
# kleiner width - dicke Spots, die eng beieinander sitzen.
FORBIDDEN_FACTOR_DEFAULT = 2.0

# Fallbacks, falls ein aelterer Datensatz theta_max/f_band nicht
# mitgespeichert hat - dieselben Werte wie die Optimierer-Defaults.
THETA_MAX_DEFAULT = 43e-3
F_BAND_DEFAULT = 36e6


def um_per_MHz(results):
    """Wieviele µm an der Trap-Ebene entsprechen 1 MHz width? Benutzt
    width_to_um() des Projekts, nicht eine nachgebaute Formel."""
    return float(width_to_um(
        1e6, results['f1'], results['f2'], results['fLO'],
        results.get('theta_max') or THETA_MAX_DEFAULT,
        results.get('f_band') or F_BAND_DEFAULT))


def waist_um_vals(results):
    """Effektiver Waist (µm nach der Linse) je win_input-Spalte."""
    return np.array([win_input_to_win(w, results['f1'], results['f2'],
                                      results['lambda_opt'], results['fLO'])
                     for w in np.asarray(results['win_input_vals'], dtype=float)]) * 1e6


def forbidden_boundary(results, factor=FORBIDDEN_FACTOR_DEFAULT):
    """Die Grenzgerade des verbotenen Bereichs, oder None (nur wenn es gar
    keine zwei Eck-Spots gibt, N_x < 2 UND N_y < 2)."""
    n_achsen = int(results.get('N_x', 1) > 1) + int(results.get('N_y', 1) > 1)
    if n_achsen == 0:
        return None
    u = um_per_MHz(results)
    diag = np.sqrt(float(n_achsen))          # sqrt(2) beim ueblichen 2D-Array
    slope = float(factor) / (diag * u)
    return dict(factor=float(factor), um_per_MHz=u, n_axes=n_achsen,
                diag_factor=float(diag), slope=slope)


def forbidden_mask(results, factor=FORBIDDEN_FACTOR_DEFAULT):
    """Maske (Zeilen=width, Spalten=win_input): True, wo sich die Eck-Spots
    ueberlappen wuerden. None, wenn es keine Grenze gibt."""
    grenze = forbidden_boundary(results, factor)
    if grenze is None:
        return None
    waist = waist_um_vals(results)                                  # µm
    width = np.asarray(results['width_vals'], dtype=float) * 1e-6   # MHz
    W, WD = np.meshgrid(waist, width)
    # Erlaubt ist width > slope*waist; Gleichheit heisst "beruehren sich
    # gerade" und zaehlt damit als verboten.
    return WD <= grenze['slope'] * W


GRID_KEYS_FORBIDDEN = ("uniformity_grid", "crosstalk_grid",
                       "uniformity_weighted_grid", "eta_weighted_grid",
                       "r_x_grid", "r_y_grid")


def mask_forbidden_grids(results, factor=FORBIDDEN_FACTOR_DEFAULT):
    """Kopie des Datensatzes, in der alle Gitter im verbotenen Bereich auf
    NaN stehen. Gibt (results, maske) zurueck; maske ist None, wenn es keine
    Grenze gibt.

    Score/Region/Bestpunkt rechnet der Aufrufer danach mit analyse() neu.
    Weil der Score hier ROH ist (keine gitterweite Normierung), aendert der
    Ausschluss die Zahlen NUR im verbotenen Bereich, nicht anderswo.
    """
    maske = forbidden_mask(results, factor)
    if maske is None:
        return dict(results), None
    neu = dict(results)
    for key in GRID_KEYS_FORBIDDEN:
        if neu.get(key) is not None:
            grid = np.array(neu[key], dtype=float)
            if grid.shape == maske.shape:
                grid[maske] = np.nan
                neu[key] = grid
    neu['forbidden_factor'] = float(factor)
    neu['forbidden_excluded'] = True
    return neu, maske


# ======================================================================
# Laden / Speichern
# ======================================================================
def resolve_pickle_path(out_dir, filename, overwrite=False):
    """Vollstaendiger Pfad in out_dir. Bei overwrite=False und bereits
    existierender Datei wird ein freier Name mit Zaehler (_2, _3, ...)
    gewaehlt."""
    out_dir = FilePath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    if overwrite or not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    counter = 2
    while (out_dir / f"{stem}_{counter}{suffix}").exists():
        counter += 1
    return out_dir / f"{stem}_{counter}{suffix}"


def save_results(results, filepath, overwrite=False):
    """Ergebnis-dict als .pkl speichern."""
    filepath = FilePath(filepath)
    if not overwrite:
        filepath = resolve_pickle_path(filepath.parent, filepath.name, overwrite=False)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as fh:
        pickle.dump(results, fh)
    print(f"Datensatz gespeichert: {filepath}")
    return filepath


def load_results(filepath):
    """Ergebnis-dict aus .pkl laden. `filepath` darf ein voller Pfad oder ein
    blosser Dateiname sein (dann wird in Results/ und im Ordner selbst
    gesucht)."""
    candidate = FilePath(filepath)
    if not candidate.exists():
        for base in (paths.DEFAULT_RESULTS_DIR, paths.BASE_DIR):
            alt = base / candidate.name
            if alt.exists():
                candidate = alt
                break
    if not candidate.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")
    with open(candidate, 'rb') as fh:
        results = pickle.load(fh)
    if not isinstance(results, dict):
        raise ValueError(f"{candidate} enthaelt kein Ergebnis-dict, sondern "
                         f"{type(results).__name__}.")
    results['_source_path'] = str(candidate)
    return results


def describe(results):
    """Kurze, menschenlesbare Einordnung - die GUI zeigt das direkt an."""
    kind = dataset_kind(results)
    fam = flavor_of(results)
    n_win = len(np.asarray(results.get('win_input_vals', [])))
    n_width = len(np.asarray(results.get('width_vals', [])))
    lines = [
        f"Art: {KIND_LABELS.get(kind, kind)}",
        f"Metriken: {FLAVOR_LABELS.get(fam, 'unbekannt')}",
        f"Gitter: {n_win} x {n_width} Punkte, N_x={results.get('N_x')}, "
        f"N_y={results.get('N_y')}, Profil={results.get('profile')}",
    ]
    if is_foreign(results):
        lines.append(
            "ACHTUNG: dieser Datensatz enthaelt die Metriken des ANDEREN Ordners "
            f"({FLAVOR_LABELS.get(fam)}). Er laesst sich hier plotten, gehoert "
            "aber vermutlich in den anderen Ordner.")
    if results.get('_checkpoint') or results.get('checkpoint'):
        lines.append("ACHTUNG: das ist ein unvollstaendiger Zwischenstand (Checkpoint).")
    if results.get('alpha') is not None:
        lines.append(f"alpha = {float(results['alpha']):.3f}")
    if results.get('r_bounds') is not None:
        lines.append(f"r_bounds = {tuple(results['r_bounds'])}")
    best = results.get('best') or results.get('best_stored') or {}
    if best.get('win_input') is not None:
        lines.append(f"Bester Punkt: win_input={best['win_input'] * 1e3:.4f} mm, "
                     f"width={best['width'] * 1e-6:.4f} MHz")
    return "\n".join(lines)


def looks_like_scan(results):
    """(ok, fehlende_schluessel) - taugt die Datei ueberhaupt als Scan?"""
    needed = ['win_input_vals', 'width_vals']
    missing = [k for k in needed if k not in results]
    if flavor_of(results) is None:
        missing.append("uniformity_*_grid/crosstalk_grid")
    return (len(missing) == 0), missing
