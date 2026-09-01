"""
lib/penalty_scan.py - die PENALTY-METHODE (gemeinsame Amplituden-Optimierung).

Das ist das Verfahren, das run_penalty_scan.py ausfuehrt und das die
vorhandenen scan_amp_data_combined_*.pkl erzeugt hat.

An JEDEM (win_input, width)-Gitterpunkt laeuft GENAU EINE Nelder-Mead-
Optimierung ueber (r_x, r_y), die direkt

    J(r_x, r_y) = alpha * U_kombi + (1 - alpha) * C_kombi

minimiert, mit dem Penalty-Term aus lib/combine.py:

    U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
    C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|

Entscheidend: U_hart/C_hart (harte Maske) UND U_w/C_w (atom-gewichtet)
werden bei JEDEM Optimierungsschritt am SELBEN (r_x, r_y) ausgewertet -
ein einziger _evaluate(..., weighted=True)-Aufruf liefert beide
Metrik-Paare zugleich. Die gefundene Amplitude ist damit automatisch fuer
BEIDE Kriterien gleichzeitig gueltig.

(Abgrenzung: das frueher hier vorhandene GETRENNTE Verfahren - hart und
gewichtet je einzeln optimieren, danach die an verschiedenen r_x/r_y
erreichten Metriken verrechnen - ist bewusst nicht mehr enthalten. Es
liegt unveraendert im Archivordner Old_Combine/.)

Zwischenspeicherung: der Scan sichert stuendlich unter checkpoint_path
und setzt bei einem erneuten Start mit denselben Parametern automatisch
dort fort, wo er stehen geblieben ist.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize

from . import paths  # noqa: F401  (setzt sys.path auf Weighted_Optimization)
from .combine import penalty_objective, build_penalty_results

from weighted_multitone_flattop_optimizer import (  # noqa: E402
    MultitoneFlatTopOptimizer, amps_from_ratios,
)
import scan_checkpoint  # noqa: E402
import perf_log  # noqa: E402


# ======================================================================
# Duenne Huelle um alles, was aus Weighted_Optimization kommt.
#
# Die drei Haupt-Skripte importieren AUSSCHLIESSLICH aus lib - dadurch
# taucht der (in PyCharm rot markierte, weil erst zur Laufzeit ueber
# sys.path aufgeloeste) Import dieser Fremdmodule nur noch hier auf.
# Siehe den Hinweis in lib/paths.py, wie sich das Rot dauerhaft
# abstellen laesst ("Mark Directory as -> Sources Root").
# ======================================================================
def make_optimizer(**params):
    """MultitoneFlatTopOptimizer mit Ausgabe in Combinated_Optimization/Bilder."""
    return MultitoneFlatTopOptimizer(out_dir=str(paths.DEFAULT_IMAGES_DIR), **params)


def peek_checkpoint(checkpoint_path, win_input_range, width_range, n_points,
                    N_x, N_y, alpha, r_bounds, combo_lambda,
                    airy_scale_factor=None, optics_match=None):
    """Schaut VOR dem Scan nach, ob unter checkpoint_path ein passender
    Zwischenstand liegt.

    Gibt (n_fertige_punkte, n_punkte_gesamt) zurueck, wenn fortgesetzt
    werden kann, sonst None.
    """
    if not checkpoint_path:
        return None
    resumed = scan_checkpoint.load_resumable(
        checkpoint_path, win_input_range, width_range, n_points, n_points, N_x, N_y,
        extra_match=dict(alpha=alpha, r_bounds=r_bounds, combo_lambda=combo_lambda,
                         joint_optimization=True),
        airy_scale_factor=airy_scale_factor,
        optics_match=optics_match,
        verbose=False,
    )
    if resumed is None:
        return None
    grid = resumed["uniformity_grid"]
    return scan_checkpoint.count_done(grid), int(np.asarray(grid).size)


def enable_perf_log():
    perf_log.enable()


def log_duration(label, seconds):
    perf_log.log(f"{label}: {seconds:.1f}s")


def setup_gpu(force_cpu=False, n_jobs=1):
    """Versucht GPU-Beschleunigung (CUDA via PyTorch), sofern nicht
    ausdruecklich abgewaehlt.

    Gibt (n_jobs, pool_initializer, meldung) zurueck - bei aktiver GPU
    wird auf einen Prozess reduziert.
    """
    if not force_cpu:
        try:
            import weighted_use_torch as use_torch
            if use_torch.cuda_available():
                use_torch.patch()
                return 1, use_torch.patch, ("GPU (CUDA via PyTorch) gefunden - "
                                            "nutze GPU-Beschleunigung.")
        except ImportError:
            pass
    grund = "CPU erzwungen" if force_cpu else "Keine GPU/CUDA gefunden"
    return n_jobs, None, f"{grund} - nutze parallelisierte CPU mit n_jobs={n_jobs}."


NELDER_MEAD_OPTIONS = {'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300}
INVALID_PENALTY = 1e10


# ======================================================================
# Auswertung EINES Punktes
# ======================================================================
def _optimize_point(opt, win_input_val, width_val, x0, r_bounds, alpha, combo_lambda):
    """Eine vollstaendige (r_x, r_y)-Optimierung an einem Gitterpunkt.

    Gibt (U_hart, C_hart, U_weighted, C_weighted, r_x, r_y) zurueck, oder
    None, wenn der Punkt ungueltig ist (z.B. win_input ausserhalb des
    physikalisch moeglichen Bereichs, oder keine gewichtete Auswertung
    moeglich).
    """
    try:
        win_eff = opt.win_input_to_win(win_input_val)
    except ValueError:
        return None

    point_grid = opt._build_dynamic_grid(win_eff, width_val)

    def objective(p):
        amps = amps_from_ratios(p[0], p[1], opt.N_x, opt.N_y)
        val = opt._evaluate(win_eff, width_val, amps=amps, grid=point_grid, weighted=True)
        if val is None:
            return INVALID_PENALTY
        U_h, C_h = val['uniformity'], val['eta']
        U_w, C_w = val.get('uniformity_weighted'), val.get('eta_weighted')
        if U_w is None or C_w is None or not (np.isfinite(U_w) and np.isfinite(C_w)):
            return INVALID_PENALTY
        return float(penalty_objective(U_h, C_h, U_w, C_w, alpha, combo_lambda))

    result = minimize(objective, x0=list(x0), method='Nelder-Mead',
                      bounds=[r_bounds, r_bounds], options=NELDER_MEAD_OPTIONS)
    r_opt = result.x

    amps_opt = amps_from_ratios(r_opt[0], r_opt[1], opt.N_x, opt.N_y)
    details = opt._evaluate(win_eff, width_val, amps=amps_opt, grid=point_grid, weighted=True)
    if details is None:
        return None
    U_w_final = details.get('uniformity_weighted')
    C_w_final = details.get('eta_weighted')
    if U_w_final is None or C_w_final is None or not (np.isfinite(U_w_final) and np.isfinite(C_w_final)):
        return None

    return (float(details['uniformity']), float(details['eta']),
            float(U_w_final), float(C_w_final), float(r_opt[0]), float(r_opt[1]))


def _worker(task):
    """Worker fuer n_jobs > 1. Muss auf Modulebene stehen, damit
    ProcessPoolExecutor ihn picklen kann."""
    (i, j, win_input_val, width_val, x0, r_bounds, alpha, combo_lambda, optimizer_kwargs) = task
    opt = make_optimizer(**optimizer_kwargs)
    out = _optimize_point(opt, win_input_val, width_val, x0, r_bounds, alpha, combo_lambda)
    if out is None:
        return (i, j, None, None, None, None, None, None)
    return (i, j) + out


# ======================================================================
# Der Scan
# ======================================================================
def run_penalty_scan(opt, win_input_range, width_range,
                     n_win_input=15, n_width=15,
                     alpha=0.7, combo_lambda=0.75, combo_percentile=25.0,
                     r_bounds=(0.1, 10.0), r0=(1.0, 1.0), warm_start=True,
                     n_jobs=1, pool_initializer=None, pool_initargs=(),
                     progress_callback=None, verbose=True,
                     checkpoint_path=None,
                     checkpoint_interval_s=scan_checkpoint.CHECKPOINT_INTERVAL_S):
    """Fuehrt den Penalty-Scan aus und gibt das fertige Ergebnis-dict
    zurueck (dasselbe Format wie die vorhandenen
    scan_amp_data_combined_*.pkl).

    opt: eine MultitoneFlatTopOptimizer-Instanz mit den gewuenschten
         Geometrie-/Atom-Parametern.
    n_jobs > 1 verteilt die Gitterpunkte auf Prozesse; warm_start
         (Startwert der Optimierung = Optimum des Vorgaengerpunktes)
         entfaellt dann, da die Punkte unabhaengig laufen.
    """
    win_input_vals = np.linspace(win_input_range[0], win_input_range[1], n_win_input)
    width_vals = np.linspace(width_range[0], width_range[1], n_width)

    extra_match = dict(alpha=alpha, r_bounds=r_bounds, combo_lambda=combo_lambda,
                       joint_optimization=True)
    resumed = scan_checkpoint.load_resumable(
        checkpoint_path, win_input_range, width_range, n_win_input, n_width,
        opt.N_x, opt.N_y, extra_match=extra_match,
        airy_scale_factor=opt.airy_scale_factor,
        optics_match=dict(n_grid=opt.n_grid, weighted_n_grid=opt.weighted_n_grid,
                          atom_offset_x=getattr(opt, 'atom_offset_x', 0.0),
                          atom_offset_y=getattr(opt, 'atom_offset_y', 0.0)),
        verbose=verbose,
    )

    if resumed is not None:
        uniformity_grid = np.asarray(resumed['uniformity_grid'], dtype=float).copy()
        crosstalk_grid = np.asarray(resumed['crosstalk_grid'], dtype=float).copy()
        uniformity_weighted_grid = np.asarray(resumed['uniformity_weighted_grid'], dtype=float).copy()
        eta_weighted_grid = np.asarray(resumed['eta_weighted_grid'], dtype=float).copy()
        r_x_grid = np.asarray(resumed['r_x_grid'], dtype=float).copy()
        r_y_grid = np.asarray(resumed['r_y_grid'], dtype=float).copy()
        n_done_before = scan_checkpoint.count_done(uniformity_grid)
        if verbose:
            print(f"[Checkpoint] Setze Penalty-Scan fort: {n_done_before}/{uniformity_grid.size} "
                  f"Punkte bereits vorhanden ({checkpoint_path}).")
    else:
        shape = (n_width, n_win_input)
        uniformity_grid = np.full(shape, np.nan)
        crosstalk_grid = np.full(shape, np.nan)
        uniformity_weighted_grid = np.full(shape, np.nan)
        eta_weighted_grid = np.full(shape, np.nan)
        r_x_grid = np.full(shape, np.nan)
        r_y_grid = np.full(shape, np.nan)
        n_done_before = 0

    ckpt = scan_checkpoint.CheckpointWriter(checkpoint_path, checkpoint_interval_s, verbose=verbose)

    def _scan_dict():
        return dict(
            win_input_vals=win_input_vals, width_vals=width_vals,
            uniformity_grid=uniformity_grid, crosstalk_grid=crosstalk_grid,
            uniformity_weighted_grid=uniformity_weighted_grid,
            eta_weighted_grid=eta_weighted_grid,
            r_x_grid=r_x_grid, r_y_grid=r_y_grid,
            alpha=alpha, r_bounds=r_bounds, combo_lambda=combo_lambda,
            combo_percentile=combo_percentile,
            joint_optimization=True, dataset_kind="penalty",
            sigma_atom=opt.sigma_atom,
            N_x=opt.N_x, N_y=opt.N_y, f1=opt.f1, f2=opt.f2, fLO=opt.fLO,
            lambda_opt=opt.lambda_opt, theta_max=opt.theta_max, f_band=opt.f_band,
            profile=opt.profile,
            # Der Airy-Skalenfaktor bestimmt die physikalische Spotgroesse
            # (first_zero_radius = airy_scale_factor * win) und damit JEDE
            # Metrik dieses Scans. Er gehoert deshalb in den Datensatz -
            # aeltere Dateien haben ihn nicht, dort gilt der Default 1.19.
            airy_scale_factor=getattr(opt, "airy_scale_factor", None),
            atom_mass=opt.atom_mass, atom_temperature=opt.atom_temperature,
            trap_freq_r=opt.trap_freq_r,
            # n_grid/weighted_n_grid (und der Atom-Offset) bestimmen, WIE fein
            # ausgewertet wurde. Ein fortgesetzter Scan mit anderer Aufloesung
            # haette sonst zwei verschiedene Aufloesungen in einem Datensatz.
            n_grid=opt.n_grid, weighted_n_grid=opt.weighted_n_grid,
            atom_offset_x=getattr(opt, "atom_offset_x", 0.0),
            atom_offset_y=getattr(opt, "atom_offset_y", 0.0),
        )

    def _is_done(i, j):
        return (np.isfinite(uniformity_grid[i, j]) and np.isfinite(uniformity_weighted_grid[i, j])
                and np.isfinite(r_x_grid[i, j]) and np.isfinite(r_y_grid[i, j]))

    total = n_width * n_win_input
    cancelled = False

    # Startpunkt fuer warm_start: das Optimum des zuletzt berechneten Punktes.
    last_r = [float(r0[0]), float(r0[1])]
    finite_mask = (np.isfinite(r_x_grid) & np.isfinite(r_y_grid)
                   & np.isfinite(uniformity_grid) & np.isfinite(uniformity_weighted_grid))
    if np.any(finite_mask):
        flat_idx = np.flatnonzero(finite_mask.ravel())[-1]
        i_last, j_last = np.unravel_index(flat_idx, r_x_grid.shape)
        last_r = [float(r_x_grid[i_last, j_last]), float(r_y_grid[i_last, j_last])]

    n_jobs_resolved = 1 if n_jobs in (None, 0) else n_jobs
    if n_jobs_resolved == -1:
        n_jobs_resolved = os.cpu_count() or 1

    if verbose:
        print("\n" + "=" * 70)
        print(f"PENALTY-SCAN: {n_win_input}x{n_width} Punkte, pro Punkt EINE (r_x, r_y)-"
              f"Optimierung direkt gegen die Kombination aus hart + gewichtet "
              f"(alpha={alpha}, combo_lambda={combo_lambda}), N_x={opt.N_x}, N_y={opt.N_y}, "
              f"n_jobs={n_jobs_resolved}")
        print("=" * 70)

    if n_jobs_resolved <= 1:
        # -------------------- sequentiell, mit warm_start --------------------
        done = 0
        for i, width_val in enumerate(width_vals):
            if cancelled:
                break
            for j, win_input_val in enumerate(win_input_vals):
                if _is_done(i, j):
                    if warm_start:
                        last_r = [float(r_x_grid[i, j]), float(r_y_grid[i, j])]
                else:
                    x0 = last_r if warm_start else [float(r0[0]), float(r0[1])]
                    out = _optimize_point(opt, win_input_val, width_val, x0,
                                          r_bounds, alpha, combo_lambda)
                    if out is not None:
                        (uniformity_grid[i, j], crosstalk_grid[i, j],
                         uniformity_weighted_grid[i, j], eta_weighted_grid[i, j],
                         r_x_grid[i, j], r_y_grid[i, j]) = out
                        if warm_start:
                            last_r = [float(out[4]), float(out[5])]

                done += 1
                if progress_callback is not None:
                    if progress_callback(done, total) is False:
                        cancelled = True
                        break
                ckpt.maybe_save(_scan_dict, done=done, total=total)
            if verbose and n_width >= 10 and (i % max(1, n_width // 10) == 0):
                print(f"  ... width-Zeile {i + 1}/{n_width}")

    else:
        # -------------------- parallel ueber Prozesse --------------------
        optimizer_kwargs = {k: getattr(opt, k) for k in opt.DEFAULTS}
        tasks = [
            (i, j, win_input_val, width_val, list(r0), r_bounds, alpha, combo_lambda,
             optimizer_kwargs)
            for i, width_val in enumerate(width_vals)
            for j, win_input_val in enumerate(win_input_vals)
            if not _is_done(i, j)
        ]
        done = n_done_before

        if verbose:
            skipped = total - len(tasks)
            skip_msg = f" ({skipped} bereits aus Checkpoint vorhanden)" if skipped else ""
            print(f"Verteile {len(tasks)} Punkte auf {n_jobs_resolved} Prozesse{skip_msg} "
                  f"(warm_start wird bei n_jobs>1 nicht verwendet)...")

        if not tasks:
            if verbose:
                print("  Alle Punkte bereits vorhanden - nichts zu tun.")
        else:
            with ProcessPoolExecutor(max_workers=n_jobs_resolved,
                                     initializer=pool_initializer,
                                     initargs=pool_initargs) as executor:
                futures = {executor.submit(_worker, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        i, j, U_h, C_h, U_w, C_w, r_x, r_y = future.result()
                    except Exception:
                        task = futures[future]
                        i, j = task[0], task[1]
                        U_h = C_h = U_w = C_w = r_x = r_y = None

                    if U_h is not None:
                        uniformity_grid[i, j] = U_h
                        crosstalk_grid[i, j] = C_h
                        uniformity_weighted_grid[i, j] = U_w
                        eta_weighted_grid[i, j] = C_w
                        r_x_grid[i, j] = r_x
                        r_y_grid[i, j] = r_y

                    done += 1
                    if progress_callback is not None and not cancelled:
                        if progress_callback(done, total) is False:
                            cancelled = True
                            for f in futures:
                                f.cancel()
                    ckpt.maybe_save(_scan_dict, done=done, total=total)
                    if verbose and total >= 10 and done % max(1, total // 10) == 0:
                        print(f"  ... {done}/{total} Punkte fertig")

    if ckpt.active:
        ckpt.maybe_save(_scan_dict, done=done, total=total, force=True)

    results = build_penalty_results(_scan_dict(), alpha=alpha, combo_lambda=combo_lambda,
                                    combo_percentile=combo_percentile)
    opt.results['scan2d_amp_penalty'] = results

    if verbose:
        print_summary(results, combo_percentile)

    return results


def print_summary(results, combo_percentile=None):
    """Kurze Konsolen-Zusammenfassung nach dem Scan."""
    combo_percentile = results.get('combo_percentile') if combo_percentile is None else combo_percentile
    best, region = results['best'], results['region']
    print("\n" + "=" * 70)
    if best['win_input'] is not None:
        j = int(np.argmin(np.abs(results['win_input_vals'] - best['win_input'])))
        i = int(np.argmin(np.abs(results['width_vals'] - best['width'])))
        print(f"Bester Punkt: win_input={best['win_input'] * 1e3:.4f} mm, "
              f"width={best['width'] * 1e-6:.4f} MHz, "
              f"r_x={results['r_x_grid'][i, j]:.4f}, r_y={results['r_y_grid'][i, j]:.4f}")
        print(f"  Uniformity_hart={best['uniformity_hart'] * 100:.2f}%, "
              f"Crosstalk_hart={best['crosstalk_hart'] * 100:.3f}%, "
              f"Uniformity_w={best['uniformity_weighted'] * 100:.2f}%, "
              f"Crosstalk_w={best['crosstalk_weighted'] * 100:.3f}%")
    if region['win_input_min'] is not None:
        print(f"Region (beste {combo_percentile:.0f}% des Scores, groesstes Rechteck; "
              f"{region['n_points_region']}/{region['n_points_total']} Punkte im "
              f"Akzeptanzbereich): win_input in [{region['win_input_min'] * 1e3:.4f}, "
              f"{region['win_input_max'] * 1e3:.4f}] mm, width in "
              f"[{region['width_min'] * 1e-6:.4f}, {region['width_max'] * 1e-6:.4f}] MHz")
    else:
        print("Region: kein gueltiges Rechteck gefunden (zu wenige valide Punkte).")
    print("=" * 70)
