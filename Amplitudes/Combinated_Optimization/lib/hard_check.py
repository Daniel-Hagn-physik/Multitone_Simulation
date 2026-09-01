"""
lib/hard_check.py - harte Metriken zu einem vorhandenen GEWICHTETEN Scan
nachrechnen.

Die Frage, die dieses Modul beantwortet: bleiben die (waist, width,
r_x, r_y)-Punkte, die unter dem atom-GEWICHTETEN Ziel gut sind, auch
unter dem HARTEN (globale Maske) Ziel gut?

Vorgehen - bewusst OHNE erneute Optimierung:

    Fuer jeden Gitterpunkt eines bereits vorhandenen gewichteten
    Amplituden-Scans werden win_input, width und die dort GEFUNDENEN
    Amplituden r_x/r_y genommen und damit GENAU EINMAL die harten
    Metriken ausgewertet (_evaluate(..., weighted=False)).

Das ist um Groessenordnungen billiger als ein Scan: eine einzelne
Auswertung pro Punkt statt Dutzenden bis Hunderten Nelder-Mead-Schritten.
Ein kompletter Lauf dauert typischerweise Sekunden bis wenige Minuten -
deshalb gibt es hier bewusst KEINE Checkpoint-/Fortsetzen-Mechanik.

Ausgewertet wird anschliessend auf zwei Arten:

1. Vierfeldertafel (consistency_crosstab): unabhaengig voneinander wird
   je eine "gut"-Menge fuer den gewichteten und den harten Score
   bestimmt (die besten good_percentile% der Punkte). Die Kernkennzahl
   ist der Anteil der gewichtet-guten Punkte, die auch hart-gut sind.
   Dazu Pearson-Korrelationen von Score, Uniformity und Crosstalk.

2. Consistency-Score/Region ueber dieselbe Penalty-Kombination wie beim
   Penalty-Scan (lib/combine.py) - hier nicht als Optimierungsziel,
   sondern als raeumlich zusammenhaengende Zweitsicht: wo im
   (waist, width)-Feld stimmen beide Kriterien ueberein?
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import paths  # noqa: F401  (setzt sys.path auf Weighted_Optimization)
from .combine import combine_grids, looks_like_weighted_amp_scan

from weighted_multitone_flattop_optimizer import (  # noqa: E402
    MultitoneFlatTopOptimizer, amps_from_ratios,
)


# Parameter, die aus einem gewichteten Scan-Datensatz uebernommen werden,
# damit die Nachrechnung dieselbe Geometrie/Physik verwendet.
INHERITED_KEYS = ('N_x', 'N_y', 'f1', 'f2', 'fLO', 'lambda_opt', 'theta_max',
                  'f_band', 'profile', 'airy_scale_factor',
                  'atom_mass', 'atom_temperature', 'trap_freq_r')
# airy_scale_factor MUSS mituebernommen werden: er setzt die physikalische
# Spotgroesse. Rechnete die Nachrechnung mit einem anderen Faktor als der
# gewichtete Scan, verglichen wir zwei verschiedene Optiken miteinander.
# Datensaetze von vor dem 2026-09-01 fuehren ihn nicht - dort greift der
# Optimierer-Default 1.19, also derselbe Wert, mit dem sie gerechnet wurden.


def optimizer_from_results(results, n_grid=None, extra_params=None):
    """Baut eine MultitoneFlatTopOptimizer-Instanz mit den Parametern des
    uebergebenen Datensatzes."""
    params = {k: results[k] for k in INHERITED_KEYS if k in results and results[k] is not None}
    if n_grid is not None:
        params['n_grid'] = n_grid
    if extra_params:
        params.update(extra_params)
    return MultitoneFlatTopOptimizer(out_dir=str(paths.DEFAULT_IMAGES_DIR), **params)


def _evaluate_hard_point(opt, win_input_val, width_val, r_x, r_y):
    """Harte Uniformity/Crosstalk bei GEGEBENEN Amplituden - eine einzige
    Auswertung, keine Optimierung. Gibt (U_hart, C_hart) oder None."""
    if not (np.isfinite(r_x) and np.isfinite(r_y)):
        return None
    try:
        win_eff = opt.win_input_to_win(win_input_val)
    except ValueError:
        return None
    point_grid = opt._build_dynamic_grid(win_eff, width_val)
    amps = amps_from_ratios(r_x, r_y, opt.N_x, opt.N_y)
    val = opt._evaluate(win_eff, width_val, amps=amps, grid=point_grid, weighted=False)
    if val is None:
        return None
    U_h, C_h = val.get('uniformity'), val.get('eta')
    if U_h is None or C_h is None or not (np.isfinite(U_h) and np.isfinite(C_h)):
        return None
    return float(U_h), float(C_h)


def _worker(task):
    """Worker fuer n_jobs > 1 (Modulebene, damit picklebar)."""
    i, j, win_input_val, width_val, r_x, r_y, optimizer_kwargs = task
    opt = MultitoneFlatTopOptimizer(out_dir=str(paths.DEFAULT_IMAGES_DIR), **optimizer_kwargs)
    out = _evaluate_hard_point(opt, win_input_val, width_val, r_x, r_y)
    if out is None:
        return (i, j, None, None)
    return (i, j, out[0], out[1])


# ======================================================================
# Konsistenz-Analyse
# ======================================================================
def _pearson(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    a, b = a[ok], b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def consistency_crosstab(U_hard, C_hard, U_weighted, C_weighted, alpha=0.7,
                         good_percentile=25.0):
    """Vierfeldertafel "gewichtet gut" x "hart gut" plus Korrelationen.

    "gut" heisst jeweils: unter den besten good_percentile% der gueltigen
    Punkte nach dem jeweiligen Score alpha*Uniformity + (1-alpha)*Crosstalk.
    Beide Mengen werden UNABHAENGIG voneinander bestimmt, sind also
    (bis auf Bindungen) gleich gross.
    """
    U_hard = np.asarray(U_hard, dtype=float)
    C_hard = np.asarray(C_hard, dtype=float)
    U_weighted = np.asarray(U_weighted, dtype=float)
    C_weighted = np.asarray(C_weighted, dtype=float)

    score_hard = alpha * U_hard + (1.0 - alpha) * C_hard
    score_weighted = alpha * U_weighted + (1.0 - alpha) * C_weighted
    valid = np.isfinite(score_hard) & np.isfinite(score_weighted)

    out = dict(
        good_percentile=good_percentile, alpha=alpha,
        score_hard=score_hard, score_weighted=score_weighted, valid_mask=valid,
        n_valid=int(valid.sum()),
        n_weighted_good=0, n_hard_good=0, n_both_good=0,
        n_only_weighted_good=0, n_only_hard_good=0, n_neither_good=0,
        fraction_weighted_good_also_hard_good=None,
        threshold_weighted=None, threshold_hard=None,
        good_weighted_mask=np.zeros_like(valid), good_hard_mask=np.zeros_like(valid),
        agreement_map=np.full(valid.shape, np.nan),
        pearson_score=None, pearson_uniformity=None, pearson_crosstalk=None,
    )
    if not np.any(valid):
        return out

    thr_w = float(np.nanpercentile(score_weighted[valid], good_percentile))
    thr_h = float(np.nanpercentile(score_hard[valid], good_percentile))
    good_w = valid & (score_weighted <= thr_w)
    good_h = valid & (score_hard <= thr_h)

    both = good_w & good_h
    only_w = good_w & ~good_h
    only_h = good_h & ~good_w
    neither = valid & ~good_w & ~good_h

    # 0 = keines gut, 1 = nur gewichtet gut, 2 = nur hart gut, 3 = beide gut
    agreement = np.full(valid.shape, np.nan)
    agreement[neither] = 0
    agreement[only_w] = 1
    agreement[only_h] = 2
    agreement[both] = 3

    out.update(
        threshold_weighted=thr_w, threshold_hard=thr_h,
        good_weighted_mask=good_w, good_hard_mask=good_h,
        n_weighted_good=int(good_w.sum()), n_hard_good=int(good_h.sum()),
        n_both_good=int(both.sum()), n_only_weighted_good=int(only_w.sum()),
        n_only_hard_good=int(only_h.sum()), n_neither_good=int(neither.sum()),
        fraction_weighted_good_also_hard_good=(float(both.sum()) / float(good_w.sum())
                                               if good_w.sum() else None),
        agreement_map=agreement,
        pearson_score=_pearson(score_weighted[valid], score_hard[valid]),
        pearson_uniformity=_pearson(U_weighted[valid], U_hard[valid]),
        pearson_crosstalk=_pearson(C_weighted[valid], C_hard[valid]),
    )
    return out


# ======================================================================
# Der Hard-Check
# ======================================================================
def run_hard_check(weighted_results, n_grid=None, n_jobs=1,
                   pool_initializer=None, pool_initargs=(),
                   alpha=None, combo_lambda=0.75, good_percentile=25.0,
                   extra_params=None, progress_callback=None, verbose=True):
    """Rechnet die harten Metriken zu einem gewichteten Amplituden-Scan
    nach und gibt das fertige Ergebnis-dict zurueck.

    weighted_results: geladener gewichteter Scan (siehe
        combine.load_results()). Muss r_x_grid/r_y_grid und die
        gewichteten Metrik-Grids enthalten.
    n_grid: Aufloesung des globalen Intensitaetsgitters fuer die HARTE
        Auswertung. None = Default des Optimierers.
    alpha: Gewicht Uniformity/Crosstalk. None = Wert aus dem gewichteten
        Datensatz (dort war er das Optimierungsziel).
    """
    ok, missing = looks_like_weighted_amp_scan(weighted_results)
    if not ok:
        raise ValueError(
            "Die uebergebene Datei sieht nicht wie ein amplituden-optimierter GEWICHTETER "
            f"Scan aus - es fehlen die Schluessel: {missing}. Erwartet wird eine Datei vom "
            "Typ scan_amp_data_weighted_*.pkl."
        )

    alpha = weighted_results.get('alpha', 0.7) if alpha is None else alpha

    win_input_vals = np.asarray(weighted_results['win_input_vals'], dtype=float)
    width_vals = np.asarray(weighted_results['width_vals'], dtype=float)
    r_x_grid = np.asarray(weighted_results['r_x_grid'], dtype=float)
    r_y_grid = np.asarray(weighted_results['r_y_grid'], dtype=float)
    U_weighted = np.asarray(weighted_results['uniformity_weighted_grid'], dtype=float)
    C_weighted = np.asarray(weighted_results['eta_weighted_grid'], dtype=float)

    opt = optimizer_from_results(weighted_results, n_grid=n_grid, extra_params=extra_params)

    n_width, n_win_input = r_x_grid.shape
    total = n_width * n_win_input
    U_hard = np.full((n_width, n_win_input), np.nan)
    C_hard = np.full((n_width, n_win_input), np.nan)

    n_jobs_resolved = 1 if n_jobs in (None, 0) else n_jobs
    if verbose:
        print("\n" + "=" * 70)
        print(f"HARD-CHECK: {n_win_input}x{n_width} Punkte, pro Punkt EINE harte Auswertung "
              f"bei den vom gewichteten Scan gefundenen Amplituden "
              f"(keine erneute Optimierung), n_grid={opt.n_grid}, n_jobs={n_jobs_resolved}")
        print("=" * 70)

    cancelled = False
    done = 0

    if n_jobs_resolved <= 1:
        for i in range(n_width):
            if cancelled:
                break
            for j in range(n_win_input):
                out = _evaluate_hard_point(opt, win_input_vals[j], width_vals[i],
                                           r_x_grid[i, j], r_y_grid[i, j])
                if out is not None:
                    U_hard[i, j], C_hard[i, j] = out
                done += 1
                if progress_callback is not None:
                    if progress_callback(done, total) is False:
                        cancelled = True
                        break
    else:
        optimizer_kwargs = {k: getattr(opt, k) for k in opt.DEFAULTS}
        tasks = [(i, j, win_input_vals[j], width_vals[i], r_x_grid[i, j], r_y_grid[i, j],
                  optimizer_kwargs)
                 for i in range(n_width) for j in range(n_win_input)]
        with ProcessPoolExecutor(max_workers=n_jobs_resolved,
                                 initializer=pool_initializer,
                                 initargs=pool_initargs) as executor:
            futures = {executor.submit(_worker, task): task for task in tasks}
            for future in as_completed(futures):
                try:
                    i, j, U_h, C_h = future.result()
                except Exception:
                    task = futures[future]
                    i, j, U_h, C_h = task[0], task[1], None, None
                if U_h is not None:
                    U_hard[i, j] = U_h
                    C_hard[i, j] = C_h
                done += 1
                if progress_callback is not None and not cancelled:
                    if progress_callback(done, total) is False:
                        cancelled = True
                        for f in futures:
                            f.cancel()

    # ---------------- Auswertung ----------------
    crosstab = consistency_crosstab(U_hard, C_hard, U_weighted, C_weighted,
                                    alpha=alpha, good_percentile=good_percentile)
    combo = combine_grids(U_hard, C_hard, U_weighted, C_weighted,
                          win_input_vals, width_vals,
                          alpha=alpha, combo_lambda=combo_lambda,
                          combo_percentile=good_percentile)

    results = dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_grid=U_hard, crosstalk_grid=C_hard,
        uniformity_weighted_grid=U_weighted, eta_weighted_grid=C_weighted,
        r_x_grid=r_x_grid, r_y_grid=r_y_grid, r_grid_source="weighted",
        r_x_grid_hart=r_x_grid, r_y_grid_hart=r_y_grid,
        r_x_grid_weighted=r_x_grid, r_y_grid_weighted=r_y_grid,
        r_bounds=weighted_results.get('r_bounds'),
        joint_optimization=False,
        dataset_kind="hard_check",
        n_grid_hard=opt.n_grid,
        source_weighted_file=weighted_results.get('_source_path'),
        good_percentile=good_percentile,
        consistency=crosstab,
        N_x=opt.N_x, N_y=opt.N_y, f1=opt.f1, f2=opt.f2, fLO=opt.fLO,
        lambda_opt=opt.lambda_opt, theta_max=opt.theta_max, f_band=opt.f_band,
        profile=opt.profile,
        airy_scale_factor=getattr(opt, "airy_scale_factor", None),
        sigma_atom=opt.sigma_atom,
        atom_mass=opt.atom_mass, atom_temperature=opt.atom_temperature,
        trap_freq_r=opt.trap_freq_r,
    )
    results.update(combo)

    if verbose:
        print_summary(results)

    return results


def recheck_from_grids(results, alpha=None, combo_lambda=None, good_percentile=None):
    """Konsistenz-Analyse und Consistency-Score mit anderen Parametern neu
    berechnen - ohne die (teure) Nachrechnung zu wiederholen."""
    results = dict(results)
    alpha = results.get('alpha', 0.7) if alpha is None else alpha
    combo_lambda = results.get('combo_lambda', 0.75) if combo_lambda is None else combo_lambda
    good_percentile = (results.get('good_percentile', 25.0)
                       if good_percentile is None else good_percentile)

    results['consistency'] = consistency_crosstab(
        results['uniformity_grid'], results['crosstalk_grid'],
        results['uniformity_weighted_grid'], results['eta_weighted_grid'],
        alpha=alpha, good_percentile=good_percentile,
    )
    results.update(combine_grids(
        results['uniformity_grid'], results['crosstalk_grid'],
        results['uniformity_weighted_grid'], results['eta_weighted_grid'],
        results['win_input_vals'], results['width_vals'],
        alpha=alpha, combo_lambda=combo_lambda, combo_percentile=good_percentile,
    ))
    results['good_percentile'] = good_percentile
    return results


def print_summary(results):
    c = results.get('consistency') or {}
    print("\n" + "=" * 70)
    print(f"Gueltige Punkte: {c.get('n_valid')}  |  gewichtet gut: {c.get('n_weighted_good')}  "
          f"|  hart gut: {c.get('n_hard_good')}  |  beide gut: {c.get('n_both_good')}")
    frac = c.get('fraction_weighted_good_also_hard_good')
    if frac is not None:
        print(f"KERNKENNZAHL: {frac * 100:.1f}% der gewichtet-guten Punkte sind auch hart gut "
              f"({c.get('n_both_good')}/{c.get('n_weighted_good')}).")
    for label, key in (("Score", 'pearson_score'), ("Uniformity", 'pearson_uniformity'),
                       ("Crosstalk", 'pearson_crosstalk')):
        val = c.get(key)
        if val is not None:
            print(f"  Pearson r ({label}, gewichtet vs. hart) = {val:.4f}")
    region = results.get('region') or {}
    if region.get('win_input_min') is not None:
        print(f"Validierte Region (Consistency-Score): win_input in "
              f"[{region['win_input_min'] * 1e3:.4f}, {region['win_input_max'] * 1e3:.4f}] mm, "
              f"width in [{region['width_min'] * 1e-6:.4f}, {region['width_max'] * 1e-6:.4f}] MHz")
    print("=" * 70)
