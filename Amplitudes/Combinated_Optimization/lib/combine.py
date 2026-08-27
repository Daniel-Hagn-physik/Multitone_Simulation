"""
lib/combine.py - die Penalty-Kombination aus hart + atom-gewichtet.

Hier steht der rechnerische Kern, den BEIDE Datensatz-Arten teilen:

1. penalty_objective() - die ROHE Penalty-Zielfunktion, die waehrend der
   Optimierung an jedem einzelnen Gitterpunkt minimiert wird (siehe
   penalty_scan.py). "Roh" = ohne gitterweite Normierung, weil die
   waehrend der Optimierung eines EINZELNEN Punktes noch gar nicht zur
   Verfuegung steht (sie braucht alle Gitterpunkte).

       U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
       C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
       J       = alpha*U_kombi + (1-alpha)*C_kombi

   Der Term combo_lambda*|Differenz| ist der Penalty-Term: er bestraft
   Amplituden, bei denen hartes und atom-gewichtetes Kriterium
   auseinanderlaufen. Ohne ihn (combo_lambda=0) waere es der reine
   Mittelwert beider Kriterien.

2. combine_grids() - dieselbe Formel als NACHBEARBEITUNG ueber das
   fertige Gitter, diesmal auf gitterweit Min-Max-normierten Groessen
   (damit Uniformity und Crosstalk vergleichbar skaliert in den Score
   eingehen), plus Bestpunkt und Region (groesstes achsenparalleles
   Rechteck innerhalb der besten combo_percentile% aller Punkte).

Punkt 2 ist reine Auswertung/Visualisierung ueber bereits fertige Grids
und hat keinen Einfluss mehr auf die Wahl von r_x/r_y aus Punkt 1.

Verifiziert: combine_grids() reproduziert die Grids, den Schwellwert,
die Maske, das Region-Rechteck und den Bestpunkt des vorhandenen
Datensatzes scan_amp_data_combined_N3x4_21x21pts_Airy.pkl bit-exakt.
"""

import pickle
from pathlib import Path as FilePath

import numpy as np

from . import paths


# ======================================================================
# 1) Die rohe Penalty-Zielfunktion (waehrend der Optimierung)
# ======================================================================
def penalty_pair(x_hard, x_weighted, combo_lambda):
    """Mittelwert + Penalty-Term fuer EINE Groesse (Uniformity oder
    Crosstalk). Arbeitet elementweise, also auch auf ganzen Grids."""
    return 0.5 * (x_hard + x_weighted) + combo_lambda * np.abs(x_hard - x_weighted)


def penalty_objective(U_hard, C_hard, U_weighted, C_weighted, alpha, combo_lambda):
    """Die Zielgroesse J, die pro Gitterpunkt ueber (r_x, r_y) minimiert
    wird. Auf ROHEN (unnormierten) Metriken - siehe Modul-Docstring."""
    U_kombi = penalty_pair(U_hard, U_weighted, combo_lambda)
    C_kombi = penalty_pair(C_hard, C_weighted, combo_lambda)
    return alpha * U_kombi + (1.0 - alpha) * C_kombi


# ======================================================================
# 2) Kombination/Region als Nachbearbeitung ueber das fertige Gitter
# ======================================================================
def normalize01(grid):
    """Min-Max-Normierung ueber alle endlichen Werte des Gitters.
    NaN bleibt NaN. Konstantes Gitter -> 0."""
    grid = np.asarray(grid, dtype=float)
    finite = np.isfinite(grid)
    if not np.any(finite):
        return np.full_like(grid, np.nan)
    lo = np.nanmin(grid[finite])
    hi = np.nanmax(grid[finite])
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return np.where(finite, 0.0, np.nan)
    return (grid - lo) / span


def largest_rectangle(mask):
    """Groesstes achsenparalleles Rechteck, das vollstaendig innerhalb der
    True-Punkte von `mask` liegt.

    Gibt (flaeche, (row0, row1, col0, col1)) zurueck, Grenzen INKLUSIVE.
    Bei leerer Maske: (0, None).

    Standard-Verfahren: zeilenweise Histogramm der Saeulenhoehen, darin
    per Stack das groesste Rechteck.
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


def combine_grids(U_hard, C_hard, U_weighted, C_weighted,
                  win_input_vals, width_vals,
                  alpha=0.7, combo_lambda=0.75, combo_percentile=25.0):
    """Normierung -> Penalty-Kombination -> combined_score -> Bestpunkt +
    Region. Gibt das Teil-dict zurueck, das in jedes Ergebnis-dict
    einfliesst (siehe build_penalty_results()/hard_check.py)."""
    U_hard = np.asarray(U_hard, dtype=float)
    C_hard = np.asarray(C_hard, dtype=float)
    U_weighted = np.asarray(U_weighted, dtype=float)
    C_weighted = np.asarray(C_weighted, dtype=float)
    win_input_vals = np.asarray(win_input_vals, dtype=float)
    width_vals = np.asarray(width_vals, dtype=float)

    U_hard_norm = normalize01(U_hard)
    C_hard_norm = normalize01(C_hard)
    U_weighted_norm = normalize01(U_weighted)
    C_weighted_norm = normalize01(C_weighted)

    U_kombi = penalty_pair(U_hard_norm, U_weighted_norm, combo_lambda)
    C_kombi = penalty_pair(C_hard_norm, C_weighted_norm, combo_lambda)
    combined_score = alpha * U_kombi + (1.0 - alpha) * C_kombi

    finite = np.isfinite(combined_score)

    best = dict(win_input=None, width=None,
                uniformity_hart=None, crosstalk_hart=None,
                uniformity_weighted=None, crosstalk_weighted=None,
                uniformity_kombi=None, crosstalk_kombi=None, combined_score=None)
    if np.any(finite):
        i, j = np.unravel_index(
            int(np.argmin(np.where(finite, combined_score, np.inf))), combined_score.shape)
        best = dict(
            win_input=float(win_input_vals[j]), width=float(width_vals[i]),
            uniformity_hart=float(U_hard[i, j]), crosstalk_hart=float(C_hard[i, j]),
            uniformity_weighted=float(U_weighted[i, j]), crosstalk_weighted=float(C_weighted[i, j]),
            uniformity_kombi=float(U_kombi[i, j]), crosstalk_kombi=float(C_kombi[i, j]),
            combined_score=float(combined_score[i, j]),
        )

    region = dict(win_input_min=None, win_input_max=None, width_min=None, width_max=None,
                  mask=np.zeros_like(combined_score, dtype=bool), threshold=None,
                  n_points_total=int(combined_score.size), n_points_region=0,
                  row_bounds=None, col_bounds=None)
    if np.any(finite):
        threshold = float(np.nanpercentile(combined_score[finite], combo_percentile))
        mask = finite & (combined_score <= threshold)
        area, bounds = largest_rectangle(mask)
        region = dict(
            mask=mask, threshold=threshold,
            n_points_total=int(combined_score.size), n_points_region=int(mask.sum()),
            win_input_min=None, win_input_max=None, width_min=None, width_max=None,
            row_bounds=None, col_bounds=None,
        )
        if bounds is not None and area > 0:
            r0, r1, c0, c1 = bounds
            region.update(
                win_input_min=float(win_input_vals[c0]), win_input_max=float(win_input_vals[c1]),
                width_min=float(width_vals[r0]), width_max=float(width_vals[r1]),
                row_bounds=(int(r0), int(r1)), col_bounds=(int(c0), int(c1)),
            )

    return dict(
        uniformity_hart_norm=U_hard_norm, crosstalk_hart_norm=C_hard_norm,
        uniformity_weighted_norm=U_weighted_norm, crosstalk_weighted_norm=C_weighted_norm,
        uniformity_kombi=U_kombi, crosstalk_kombi=C_kombi,
        combined_score=combined_score,
        best=best, region=region,
        alpha=alpha, combo_lambda=combo_lambda, combo_percentile=combo_percentile,
    )


# ======================================================================
# Ergebnis-dict des Penalty-Scans
# ======================================================================
def build_penalty_results(scan, alpha=0.7, combo_lambda=0.75, combo_percentile=25.0):
    """Baut aus den rohen Scan-Grids das vollstaendige Ergebnis-dict.

    `scan` enthaelt die vier Metrik-Grids, r_x_grid/r_y_grid und die
    Geometrie-/Atom-Parameter (siehe penalty_scan._current_results()).

    r_x_grid_hart/_weighted (bzw. r_y_...) werden identisch zu
    r_x_grid/r_y_grid gesetzt: bei der Penalty-Methode gibt es pro Punkt
    nur EIN gemeinsames Optimum, hart und gewichtet wurden am SELBEN
    (r_x, r_y) ausgewertet. Die beiden Zusatz-Schluessel existieren
    ausschliesslich, damit Auswertungs-Code, der sie erwartet,
    unveraendert funktioniert.
    """
    win_input_vals = np.asarray(scan['win_input_vals'], dtype=float)
    width_vals = np.asarray(scan['width_vals'], dtype=float)

    U_hard = np.asarray(scan['uniformity_grid'], dtype=float)
    C_hard = np.asarray(scan['crosstalk_grid'], dtype=float)
    U_weighted = np.asarray(scan['uniformity_weighted_grid'], dtype=float)
    C_weighted = np.asarray(scan['eta_weighted_grid'], dtype=float)

    combo = combine_grids(U_hard, C_hard, U_weighted, C_weighted,
                          win_input_vals, width_vals,
                          alpha=alpha, combo_lambda=combo_lambda,
                          combo_percentile=combo_percentile)

    r_x_grid = np.asarray(scan['r_x_grid'], dtype=float)
    r_y_grid = np.asarray(scan['r_y_grid'], dtype=float)

    out = dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_grid=U_hard, crosstalk_grid=C_hard,
        uniformity_weighted_grid=U_weighted, eta_weighted_grid=C_weighted,
        r_x_grid=r_x_grid, r_y_grid=r_y_grid, r_grid_source="joint",
        r_x_grid_hart=r_x_grid, r_y_grid_hart=r_y_grid,
        r_x_grid_weighted=r_x_grid, r_y_grid_weighted=r_y_grid,
        r_bounds=scan.get('r_bounds'),
        joint_optimization=True,
        dataset_kind="penalty",
        N_x=scan['N_x'], N_y=scan['N_y'],
        f1=scan['f1'], f2=scan['f2'], fLO=scan['fLO'],
        lambda_opt=scan['lambda_opt'], theta_max=scan['theta_max'], f_band=scan['f_band'],
        profile=scan['profile'],
        sigma_atom=scan.get('sigma_atom'),
        atom_mass=scan.get('atom_mass'), atom_temperature=scan.get('atom_temperature'),
        trap_freq_r=scan.get('trap_freq_r'),
    )
    out.update(combo)
    return out


def recombine_from_grids(results, alpha=None, combo_lambda=None, combo_percentile=None):
    """Score/Region eines bereits vorhandenen Datensatzes mit ANDEREN
    Kombinationsparametern neu berechnen - ohne teuren Re-Scan.

    Jeder Parameter, der None bleibt, behaelt den im Datensatz
    gespeicherten Wert. Die Metrik-Grids und r_x/r_y bleiben unangetastet
    (sie sind das Ergebnis der Optimierung und aendern sich nicht).
    """
    results = dict(results)
    alpha = results.get('alpha', 0.7) if alpha is None else alpha
    combo_lambda = results.get('combo_lambda', 0.75) if combo_lambda is None else combo_lambda
    combo_percentile = (results.get('combo_percentile', 25.0)
                        if combo_percentile is None else combo_percentile)

    combo = combine_grids(
        results['uniformity_grid'], results['crosstalk_grid'],
        results['uniformity_weighted_grid'], results['eta_weighted_grid'],
        results['win_input_vals'], results['width_vals'],
        alpha=alpha, combo_lambda=combo_lambda, combo_percentile=combo_percentile,
    )
    results.update(combo)
    return results


# ======================================================================
# Speichern / Laden
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
    """Ergebnis-dict als .pkl speichern.

    overwrite=True ueberschreibt GENAU den angegebenen Pfad. Das ist der
    Normalfall am Ende eines Scans: derselbe Pfad wurde vorher schon als
    Checkpoint-Pfad benutzt, dort liegt also bereits ein Zwischenstand,
    der jetzt bewusst durch den sauberen Endstand ersetzt werden soll -
    statt daneben eine verwirrende "_2"-Doppeldatei anzulegen.
    """
    filepath = FilePath(filepath)
    if not overwrite:
        filepath = resolve_pickle_path(filepath.parent, filepath.name, overwrite=False)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as fh:
        pickle.dump(results, fh)
    print(f"Datensatz gespeichert: {filepath}")
    return filepath


def load_results(filepath):
    """Ergebnis-dict aus .pkl laden. `filepath` darf ein voller Pfad oder
    ein blosser Dateiname sein (dann wird in Results/ und neben
    Combinated_Optimization/ gesucht)."""
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
    results['_source_path'] = str(candidate)
    return results


def describe(results):
    """Kurze, menschenlesbare Einordnung eines geladenen Datensatzes -
    genutzt von den GUIs, um sofort anzuzeigen, was da geladen wurde."""
    kind = dataset_kind(results)
    n_win = len(np.asarray(results.get('win_input_vals', [])))
    n_width = len(np.asarray(results.get('width_vals', [])))
    lines = [
        f"Art: {KIND_LABELS.get(kind, kind)}",
        f"Gitter: {n_win} x {n_width} Punkte, N_x={results.get('N_x')}, "
        f"N_y={results.get('N_y')}, Profil={results.get('profile')}",
    ]
    if results.get('checkpoint') or results.get('_checkpoint'):
        lines.append("ACHTUNG: das ist ein unvollstaendiger Zwischenstand (Checkpoint).")
    params = [f"{name}={results[name]}" for name in ('alpha', 'combo_lambda', 'combo_percentile')
              if results.get(name) is not None]
    if params:
        lines.append(", ".join(params))
    best = results.get('best') or {}
    if best.get('win_input') is not None:
        lines.append(f"Bester Punkt: win_input={best['win_input'] * 1e3:.4f} mm, "
                     f"width={best['width'] * 1e-6:.4f} MHz")
    return "\n".join(lines)


KIND_LABELS = {
    "penalty": "Penalty-Scan (gemeinsame Amplituden-Optimierung, hart + gewichtet)",
    "hard_check": "Hard-Check (harte Metriken bei den Amplituden eines gewichteten Scans)",
    "weighted_amp": "Gewichteter Amplituden-Scan (Eingangsdatei fuer den Hard-Check)",
    "unknown": "unbekannt",
}


def dataset_kind(results):
    """Erkennt, um welche Art Datensatz es sich handelt."""
    if results.get('dataset_kind') in KIND_LABELS:
        return results['dataset_kind']
    has_hard = 'uniformity_grid' in results and 'crosstalk_grid' in results
    has_weighted = 'uniformity_weighted_grid' in results and 'eta_weighted_grid' in results
    has_r = 'r_x_grid' in results and 'r_y_grid' in results
    if has_hard and has_weighted and has_r:
        # Sowohl die alten scan_amp_data_combined_*.pkl (joint_optimization=True)
        # als auch neue Penalty-Scans landen hier.
        return "penalty"
    if has_weighted and has_r and not has_hard:
        return "weighted_amp"
    return "unknown"


def looks_like_weighted_amp_scan(results):
    """Pruefung fuer den Hard-Check: enthaelt die Datei einen
    amplituden-optimierten GEWICHTETEN Scan (r_x/r_y je Gitterpunkt +
    gewichtete Metriken)?"""
    needed = ('win_input_vals', 'width_vals', 'r_x_grid', 'r_y_grid',
              'uniformity_weighted_grid', 'eta_weighted_grid')
    missing = [k for k in needed if k not in results]
    return (len(missing) == 0), missing
