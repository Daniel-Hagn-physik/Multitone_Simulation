"""
Weighted_Optimization/weighted_amp_scan_methods.py
====================================================

Ergänzt `weighted_multitone_flattop_optimizer.MultitoneFlatTopOptimizer` um
ZWEI neue 2D-Scan-Methoden für die atom-gewichteten Metriken
(uniformity_weighted, eta_weighted - siehe `_evaluate_weighted_only()` /
`_evaluate_weighted_metrics()` in der Optimizer-Datei), analog zu den
bereits vorhandenen Scans für die harten Masken-Metriken:

  - scan_win_width_uniformity()                 -> HIER: scan_win_width_weighted_uniformity()
  - scan_win_width_amplitude_dependence()        -> HIER: scan_win_width_amplitude_dependence_weighted()

WARUM ein eigenes Skript statt die Optimizer-Datei direkt zu ändern?
Die beiden neuen Scan-Methoden werden hier als freistehende Funktionen
definiert und am Ende per Monkey-Patch auf die Klasse gehängt
(`MultitoneFlatTopOptimizer.scan_win_width_weighted_uniformity = ...`).
Dadurch bleibt `weighted_multitone_flattop_optimizer.py` unangetastet
(kein Risiko, etwas an der ohnehin schon komplexen Optimierungslogik zu
zerbrechen), man kann diese Erweiterung aber trotzdem ganz normal als
Methode aufrufen:

    from weighted_multitone_flattop_optimizer import MultitoneFlatTopOptimizer
    import weighted_amp_scan_methods  # nur importieren reicht - patcht beim Import

    opt = MultitoneFlatTopOptimizer(out_dir=".")
    opt.scan_win_width_weighted_uniformity(
        win_input_range=(0.5e-3, 3.5e-3), width_range=(0.15e6, 0.6e6),
        n_win_input=40, n_width=40,
    )
    opt.save_scan_weighted_results()   # -> Results/scan_data_weighted_....pkl

    opt.scan_win_width_amplitude_dependence_weighted(
        win_input_range=(0.5e-3, 3.5e-3), width_range=(0.15e6, 0.6e6),
        n_win_input=15, n_width=15, n_jobs=4,          # perf: wie im Original, mehrere Prozesse
        pool_initializer=None,                          # usetorch: siehe Docstring unten
    )
    opt.save_scan_amp_results_weighted()  # -> Results/scan_amp_data_weighted_....pkl

Beide neuen Scans landen (dank _default_dir() in der Optimizer-Datei, die
sich am __file__ DIESES Skripts NICHT beteiligt - _resolve_pickle_path()
schreibt weiterhin in DEFAULT_RESULTS_DIR aus weighted_multitone_flattop_
optimizer.py) im selben "Results"-Ordner wie die unveränderten
scan_data_.../scan_amp_data_...-Dateien, aber mit dem Zusatz "_weighted"
im Dateinamen, damit nichts überschrieben wird und beim Laden sofort klar
ist, welche Datei die atom-gewichteten Metriken enthält.

PERF ("perf, usetorch etc" - siehe Auftrag): beide neuen Scans erben exakt
dieselbe n_jobs/ProcessPoolExecutor/pool_initializer-Infrastruktur wie
scan_win_width_amplitude_dependence() im Original (n_jobs=1 sequentiell
mit warm_start, n_jobs>1 parallel über mehrere Prozesse OHNE warm_start,
pool_initializer/pool_initargs zum erneuten Anwenden eines use_torch.py-
Monkey-Patches in jedem Worker-Prozess - siehe Docstring von
scan_win_width_amplitude_dependence_weighted() unten für Details). Der
gewichtete Amplituden-Scan ist dabei sogar SCHNELLER pro Punkt als das
Original: _evaluate_weighted_only() baut - anders als _evaluate() - kein
teures globales/dynamisches Rechengitter auf, sondern nur ein kleines
lokales Sub-Grid um die betrachtete Site (siehe _build_local_weighted_grid()
in der Optimizer-Datei).

WICHTIG - was hier NICHT gemacht wird: die harten Masken-Metriken
(uniformity/eta) werden in den neuen Scans NICHT mitberechnet (aus
Performance-Gründen - genau das vermeiden, was _evaluate_weighted_only()
gegenüber _evaluate() einspart). Wer beide Metrik-Arten für denselben
Scan braucht, ruft zusätzlich die UNVERÄNDERTEN Original-Methoden
(scan_win_width_uniformity()/scan_win_width_amplitude_dependence()) auf -
die existieren unverändert auf derselben Klasse.
"""

import os
import pickle
from pathlib import Path as FilePath
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize

import scan_checkpoint
from weighted_multitone_flattop_optimizer import (
    MultitoneFlatTopOptimizer,
    amps_from_ratios,
    _resolve_pickle_path,
    DEFAULT_RESULTS_DIR,
)


# ======================================================================
# Teil 1: Fest-Amplitude-Scan der gewichteten Uniformity/Crosstalk
# (Analogon zu scan_win_width_uniformity())
# ======================================================================
def scan_win_width_weighted_uniformity(self, win_input_range, width_range,
                                        n_win_input=40, n_width=40,
                                        amps=None, alpha=0.9, verbose=True,
                                        progress_callback=None,
                                        checkpoint_path=None,
                                        checkpoint_interval_s=scan_checkpoint.CHECKPOINT_INTERVAL_S):
    """
    Wie scan_win_width_uniformity(), aber mit den ATOM-GEWICHTETEN Metriken
    (uniformity_weighted, eta_weighted - siehe _evaluate_weighted_only() in
    der Optimizer-Datei) statt der harten Masken-Metriken. Amplituden sind
    FEST (amps=None -> alle Amplituden = 1), an jedem Gitterpunkt wird nur
    AUSGEWERTET, nicht optimiert.

    win_input_range / width_range: (min, max) in SI-Einheiten (Meter, Hz),
    wie beim Original.

    alpha: Gewichtung für den kombinierten Score
    alpha*uniformity_weighted + (1-alpha)*eta_weighted, mit dem der einzelne
    "beste" Punkt (best-Eintrag im Ergebnis-dict) bestimmt wird - dieselbe
    Konvention wie im Original (Default 0.9).

    progress_callback: wie im Original, optionales Callable(done, total)
    -> bool|None; explizites False bricht kooperativ ab.

    checkpoint_path / checkpoint_interval_s: wie bei scan_win_width_
    uniformity() im Original - siehe scan_checkpoint.py. Speichert
    stündlich (Default) den Zwischenstand unter checkpoint_path und setzt
    einen dort bereits vorhandenen, zu diesem Scan (Gitter/N_x/N_y/amps/
    alpha) passenden Zwischenstand automatisch fort.

    Speichert das Ergebnis in self.results['scan2d_weighted'] und gibt
    (win_input_vals, width_vals, uniformity_weighted_grid, eta_weighted_grid)
    zurück. NaN-Einträge bedeuten ungültige/nicht auswertbare bzw. (bei
    Abbruch) noch nicht berechnete Punkte.
    """
    win_input_vals = np.linspace(win_input_range[0], win_input_range[1], n_win_input)
    width_vals = np.linspace(width_range[0], width_range[1], n_width)

    resumed = scan_checkpoint.load_resumable(
        checkpoint_path, win_input_range, width_range, n_win_input, n_width,
        self.N_x, self.N_y, extra_match=dict(amps=amps, alpha=alpha), verbose=verbose,
    )
    if resumed is not None:
        uniformity_weighted_grid = np.asarray(resumed['uniformity_weighted_grid'], dtype=float).copy()
        eta_weighted_grid = np.asarray(resumed['eta_weighted_grid'], dtype=float).copy()
        n_done_before = scan_checkpoint.count_done(uniformity_weighted_grid)
        if verbose:
            print(f"[Checkpoint] Setze Scan fort: {n_done_before}/{uniformity_weighted_grid.size} "
                  f"Punkte bereits vorhanden ({checkpoint_path}).")
    else:
        uniformity_weighted_grid = np.full((n_width, n_win_input), np.nan)
        eta_weighted_grid = np.full((n_width, n_win_input), np.nan)
    ckpt = scan_checkpoint.CheckpointWriter(checkpoint_path, checkpoint_interval_s, verbose=verbose)

    total = n_width * n_win_input
    done = 0
    cancelled = False

    def _current_results():
        return dict(
            win_input_vals=win_input_vals, width_vals=width_vals,
            uniformity_weighted_grid=uniformity_weighted_grid, eta_weighted_grid=eta_weighted_grid,
            amps=amps, alpha=alpha,
            best=scan_checkpoint.best_point(uniformity_weighted_grid, eta_weighted_grid, win_input_vals,
                                             width_vals, alpha, "uniformity_weighted", "eta_weighted"),
            sigma_atom=self.sigma_atom,
            N_x=self.N_x, N_y=self.N_y, f1=self.f1, f2=self.f2, fLO=self.fLO,
            lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
            profile=self.profile,
            atom_mass=self.atom_mass, atom_temperature=self.atom_temperature, trap_freq_r=self.trap_freq_r,
            atom_offset_x=self.atom_offset_x, atom_offset_y=self.atom_offset_y, pitch=self.pitch,
        )

    if verbose:
        print("\n" + "=" * 60)
        print(f"2D scan (ATOM-WEIGHTED): uniformity_weighted & eta_weighted over win_input "
              f"({n_win_input} points) x width ({n_width} points), N_x={self.N_x}, N_y={self.N_y}, "
              f"sigma_atom={self.sigma_atom * 1e9:.1f} nm")
        if self.atom_offset_x != 0.0 or self.atom_offset_y != 0.0:
            print(f"Atom/neighbor position offset: atom_offset_x={self.atom_offset_x * 1e6:+.3f} um "
                  f"({self.atom_offset_x / self.pitch:+.3f} x pitch), "
                  f"atom_offset_y={self.atom_offset_y * 1e6:+.3f} um "
                  f"({self.atom_offset_y / self.pitch:+.3f} x pitch)")
        print("=" * 60)

    for i, width_val in enumerate(width_vals):
        if cancelled:
            break
        for j, win_input_val in enumerate(win_input_vals):
            if np.isfinite(uniformity_weighted_grid[i, j]):
                pass  # bereits aus Checkpoint vorhanden - nicht neu berechnen
            else:
                try:
                    win_eff = self.win_input_to_win(win_input_val)
                except ValueError:
                    win_eff = None
                if win_eff is not None:
                    details = self._evaluate_weighted_only(win_eff, width_val, amps=amps)
                    if details is not None:
                        uniformity_weighted_grid[i, j] = details['uniformity_weighted']
                        eta_weighted_grid[i, j] = details['eta_weighted']

            done += 1
            if progress_callback is not None:
                if progress_callback(done, total) is False:
                    cancelled = True
                    break
            ckpt.maybe_save(_current_results, done=done, total=total)
        if verbose and n_width >= 10 and (i % max(1, n_width // 10) == 0):
            print(f"  ... width row {i + 1}/{n_width}")

    if ckpt.active:
        ckpt.maybe_save(_current_results, done=done, total=total, force=True)

    combined_grid = alpha * uniformity_weighted_grid + (1 - alpha) * eta_weighted_grid

    best = dict(win_input=None, width=None, uniformity_weighted=None, eta_weighted=None, combined=None)
    if np.any(np.isfinite(combined_grid)):
        idx_min = np.unravel_index(np.nanargmin(combined_grid), combined_grid.shape)
        best.update(
            win_input=win_input_vals[idx_min[1]],
            width=width_vals[idx_min[0]],
            uniformity_weighted=uniformity_weighted_grid[idx_min],
            eta_weighted=eta_weighted_grid[idx_min],
            combined=combined_grid[idx_min],
        )

    self.results['scan2d_weighted'] = dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_weighted_grid=uniformity_weighted_grid, eta_weighted_grid=eta_weighted_grid,
        amps=amps, alpha=alpha, best=best, sigma_atom=self.sigma_atom,
    )

    if verbose:
        valid = np.isfinite(uniformity_weighted_grid)
        status = "cancelled" if cancelled else "completed"
        print(f"Scan {status} ({np.sum(valid)}/{uniformity_weighted_grid.size} valid points, "
              f"{done}/{total} computed).")
        if best["win_input"] is not None:
            print(f"Combined-optimal point (alpha={alpha}): win_input={best['win_input'] * 1e3:.4f} mm, "
                  f"width={best['width'] * 1e-6:.3f} MHz  ->  "
                  f"Uniformity_w={best['uniformity_weighted'] * 100:.2f}%, "
                  f"Eta_w={best['eta_weighted'] * 100:.3f}%")
        print("=" * 60)

    return win_input_vals, width_vals, uniformity_weighted_grid, eta_weighted_grid


def get_scan_weighted_results(self):
    """Wie get_scan_results(), aber für scan_win_width_weighted_uniformity()."""
    if 'scan2d_weighted' not in self.results:
        raise RuntimeError("scan_win_width_weighted_uniformity() must be called first.")
    res = dict(self.results['scan2d_weighted'])  # shallow copy, nicht self.results mutieren
    res.update(
        N_x=self.N_x, N_y=self.N_y,
        f1=self.f1, f2=self.f2, fLO=self.fLO,
        lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
        profile=self.profile,
        atom_mass=self.atom_mass, atom_temperature=self.atom_temperature, trap_freq_r=self.trap_freq_r,
        atom_offset_x=self.atom_offset_x, atom_offset_y=self.atom_offset_y, pitch=self.pitch,
    )
    return res


def save_scan_weighted_results(self, filepath=None, overwrite=False):
    """
    Wie save_scan_results(), aber für get_scan_weighted_results(). Default-
    Dateiname bei filepath=None: "scan_data_weighted_N{N_x}x{N_y}_
    {n_win}x{n_width}pts_{Airy|Gauss}.pkl" in DEFAULT_RESULTS_DIR (identischer
    Ordner wie bei save_scan_results(), da beide auf dieselbe
    _resolve_pickle_path()/DEFAULT_RESULTS_DIR aus der Optimizer-Datei
    zurückgreifen) - der "_weighted"-Zusatz im Namen verhindert Kollisionen
    mit den unveränderten scan_data_...pkl-Dateien.

    overwrite (NEU, 2026-08-26): bei filepath!=None und overwrite=True wird
    eine bereits vorhandene Datei unter GENAU diesem Pfad direkt überschrieben
    statt (wie sonst über _resolve_pickle_path()) einen freien "_2"-Namen zu
    waehlen. Gebraucht von den GUI-Start-Dialogen, die denselben Pfad zuvor
    schon als checkpoint_path an den Scan uebergeben haben (siehe
    scan_checkpoint.py) - dort liegt unter diesem Pfad bereits eine (mit
    '_checkpoint': True markierte) Zwischenstand-Datei, die hier ganz
    bewusst durch den sauberen Endstand ERSETZT werden soll, statt
    daneben eine verwirrende Doppel-Datei zu erzeugen.
    """
    if filepath is None:
        res = self.results.get('scan2d_weighted', {})
        n_win = len(res.get('win_input_vals', []))
        n_width = len(res.get('width_vals', []))
        profile_tag = "Airy" if self.profile == "airy" else "Gauss" if self.profile == "gaussian" else self.profile
        filename = f"scan_data_weighted_N{self.N_x}x{self.N_y}_{n_win}x{n_width}pts_{profile_tag}.pkl"
        filepath = _resolve_pickle_path(DEFAULT_RESULTS_DIR, filename)
    else:
        filepath = FilePath(filepath)
        if filepath.exists() and not overwrite:
            filepath = _resolve_pickle_path(filepath.parent, filepath.name)

    with open(filepath, 'wb') as f:
        pickle.dump(self.get_scan_weighted_results(), f)
    print(f"Weighted scan results saved: {filepath}")
    return filepath


# ======================================================================
# Teil 2: Amplituden-Abhängigkeits-Scan UNTER dem gewichteten Ziel
# (Analogon zu scan_win_width_amplitude_dependence())
# ======================================================================
def scan_win_width_amplitude_dependence_weighted(self, win_input_range, width_range,
                                                   n_win_input=15, n_width=15,
                                                   alpha=0.7, r_bounds=(0.0, 2.0), r0=(1.0, 1.0),
                                                   warm_start=True, verbose=True,
                                                   progress_callback=None, n_jobs=1,
                                                   pool_initializer=None, pool_initargs=(),
                                                   checkpoint_path=None,
                                                   checkpoint_interval_s=scan_checkpoint.CHECKPOINT_INTERVAL_S):
    """
    Wie scan_win_width_amplitude_dependence(), aber die pro Gitterpunkt
    optimierten Amplitudenverhältnisse (r_x, r_y) minimieren hier
    alpha*uniformity_weighted + (1-alpha)*eta_weighted (ATOM-GEWICHTET)
    statt alpha*uniformity + (1-alpha)*eta (harte Masken). Das Ergebnis -
    r_x_opt_weighted(win_input, width) und r_y_opt_weighted(win_input, width)
    - kann von den r_x_opt/r_y_opt des Originals abweichen, weil die
    gewichtete Metrik die tatsächliche thermische Ausdehnung des Atoms
    berücksichtigt statt eine harte Kastenmaske.

    r_bounds/r0/warm_start/alpha: identische Bedeutung wie im Original.

    PERF: _evaluate_weighted_only() (siehe Optimizer-Datei) baut - anders
    als _evaluate() - KEIN teures globales/dynamisches Rechengitter auf,
    sondern nur ein kleines lokales Sub-Grid um die betrachtete Site. Ein
    einzelner Punkt dieses Scans ist daher i.d.R. spürbar schneller als
    ein Punkt von scan_win_width_amplitude_dependence() - für einen ersten
    Überblick lohnt sich trotzdem ein kleines Testgitter (n_win_input/
    n_width ~12-20), da pro Punkt weiterhin eine vollständige Nelder-Mead-
    Optimierung läuft.

    PERF/usetorch (n_jobs, pool_initializer, pool_initargs): 1:1 identisch
    zur Infrastruktur von scan_win_width_amplitude_dependence() im
    Original:
      - n_jobs=1 (Default): sequentiell im Hauptprozess, mit warm_start.
      - n_jobs=N>1: verteilt alle Punkte auf N Worker-Prozesse
        (ProcessPoolExecutor), warm_start entfällt (jeder Punkt startet
        bei r0), erwarteter Speedup nahezu linear mit der Kernzahl.
      - n_jobs=-1: nutzt alle verfügbaren CPU-Kerne (os.cpu_count()).
      - pool_initializer/pool_initargs: werden 1:1 an ProcessPoolExecutor
        durchgereicht und einmal pro Worker-PROZESS ausgeführt - z.B.
        pool_initializer=use_torch.patch, falls im Hauptprozess ein
        use_torch.py-Monkey-Patch (dyn_gaussian_2d_weighted_distance_
        from_centers/dyn_airy_2d_weighted_distance_from_centers auf eine
        GPU/PyTorch-Version umgebogen) aktiv ist und auch in den Worker-
        Prozessen gelten soll (jeder Prozess importiert das Modul beim
        Start sonst frisch/ungepatcht, siehe Docstring im Original).
      - WICHTIG unter Windows (multiprocessing='spawn'): Aufruf mit
        n_jobs>1 MUSS innerhalb eines `if __name__ == "__main__":`-Blocks
        erfolgen.

    checkpoint_path / checkpoint_interval_s: wie bei scan_win_width_
    amplitude_dependence() im Original - siehe scan_checkpoint.py.
    Speichert stündlich (Default) den Zwischenstand (alle 4 Grids) unter
    checkpoint_path und setzt einen dort bereits vorhandenen, zu diesem
    Scan (Gitter/N_x/N_y/alpha/r_bounds) passenden Zwischenstand
    automatisch fort; last_r wird beim Fortsetzen vom letzten bereits
    berechneten Gitterpunkt übernommen (statt wieder bei r0 zu beginnen).

    Speichert das Ergebnis in self.results['scan2d_amp_weighted'] und gibt
    (win_input_vals, width_vals, uniformity_weighted_grid, eta_weighted_grid,
    r_x_grid, r_y_grid) zurück.
    """
    win_input_vals = np.linspace(win_input_range[0], win_input_range[1], n_win_input)
    width_vals = np.linspace(width_range[0], width_range[1], n_width)

    resumed = scan_checkpoint.load_resumable(
        checkpoint_path, win_input_range, width_range, n_win_input, n_width,
        self.N_x, self.N_y, extra_match=dict(alpha=alpha, r_bounds=r_bounds), verbose=verbose,
    )
    if resumed is not None:
        uniformity_weighted_grid = np.asarray(resumed['uniformity_weighted_grid'], dtype=float).copy()
        eta_weighted_grid = np.asarray(resumed['eta_weighted_grid'], dtype=float).copy()
        r_x_grid = np.asarray(resumed['r_x_grid'], dtype=float).copy()
        r_y_grid = np.asarray(resumed['r_y_grid'], dtype=float).copy()
        n_done_before = scan_checkpoint.count_done(uniformity_weighted_grid)
        if verbose:
            print(f"[Checkpoint] Setze Scan fort: {n_done_before}/{uniformity_weighted_grid.size} "
                  f"Punkte bereits vorhanden ({checkpoint_path}).")
    else:
        uniformity_weighted_grid = np.full((n_width, n_win_input), np.nan)
        eta_weighted_grid = np.full((n_width, n_win_input), np.nan)
        r_x_grid = np.full((n_width, n_win_input), np.nan)
        r_y_grid = np.full((n_width, n_win_input), np.nan)
        n_done_before = 0
    ckpt = scan_checkpoint.CheckpointWriter(checkpoint_path, checkpoint_interval_s, verbose=verbose)

    def _current_results():
        return dict(
            win_input_vals=win_input_vals, width_vals=width_vals,
            uniformity_weighted_grid=uniformity_weighted_grid, eta_weighted_grid=eta_weighted_grid,
            r_x_grid=r_x_grid, r_y_grid=r_y_grid,
            alpha=alpha, r_bounds=r_bounds, sigma_atom=self.sigma_atom,
            N_x=self.N_x, N_y=self.N_y, f1=self.f1, f2=self.f2, fLO=self.fLO,
            lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
            profile=self.profile,
            atom_mass=self.atom_mass, atom_temperature=self.atom_temperature, trap_freq_r=self.trap_freq_r,
        )

    total = n_width * n_win_input
    done = 0
    cancelled = False
    last_r = [float(r0[0]), float(r0[1])]
    finite_mask = np.isfinite(r_x_grid) & np.isfinite(r_y_grid)
    if np.any(finite_mask):
        flat_idx = np.flatnonzero(finite_mask.ravel())[-1]
        i_last, j_last = np.unravel_index(flat_idx, r_x_grid.shape)
        last_r = [float(r_x_grid[i_last, j_last]), float(r_y_grid[i_last, j_last])]

    n_jobs_resolved = 1 if n_jobs in (None, 0) else n_jobs
    if n_jobs_resolved == -1:
        n_jobs_resolved = os.cpu_count() or 1

    if verbose:
        print("\n" + "=" * 60)
        print(f"2D Amplituden-Abhängigkeits-Scan (ATOM-WEIGHTED): {n_win_input}x{n_width} Punkte, "
              f"pro Punkt Optimierung von (r_x, r_y) unter uniformity_weighted/eta_weighted, "
              f"N_x={self.N_x}, N_y={self.N_y}, alpha={alpha}, n_jobs={n_jobs_resolved}")
        print("=" * 60)

    if n_jobs_resolved <= 1:
        # ------------------------------------------------------------
        # Sequentiell im Hauptprozess, mit warm_start (siehe Docstring)
        # ------------------------------------------------------------
        for i, width_val in enumerate(width_vals):
            if cancelled:
                break
            for j, win_input_val in enumerate(win_input_vals):
                if np.isfinite(uniformity_weighted_grid[i, j]) and np.isfinite(r_x_grid[i, j]) and np.isfinite(r_y_grid[i, j]):
                    # bereits aus Checkpoint vorhanden - nicht neu berechnen, aber
                    # last_r fuer warm_start konsistent mitfuehren
                    if warm_start:
                        last_r = [float(r_x_grid[i, j]), float(r_y_grid[i, j])]
                else:
                    try:
                        win_eff = self.win_input_to_win(win_input_val)
                    except ValueError:
                        win_eff = None

                    if win_eff is not None:
                        def objective(p, win_eff=win_eff, width_val=width_val):
                            amps = amps_from_ratios(p[0], p[1], self.N_x, self.N_y)
                            val = self._evaluate_weighted_only(win_eff, width_val, amps=amps)
                            if val is None:
                                return 1e10
                            return alpha * val['uniformity_weighted'] + (1 - alpha) * val['eta_weighted']

                        x0 = last_r if warm_start else [float(r0[0]), float(r0[1])]
                        result = minimize(
                            objective, x0=x0, method='Nelder-Mead',
                            bounds=[r_bounds, r_bounds],
                            options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300},
                        )
                        r_opt = result.x
                        amps_opt = amps_from_ratios(r_opt[0], r_opt[1], self.N_x, self.N_y)
                        details = self._evaluate_weighted_only(win_eff, width_val, amps=amps_opt)

                        if details is not None:
                            uniformity_weighted_grid[i, j] = details['uniformity_weighted']
                            eta_weighted_grid[i, j] = details['eta_weighted']
                            r_x_grid[i, j] = r_opt[0]
                            r_y_grid[i, j] = r_opt[1]
                            if warm_start:
                                last_r = [float(r_opt[0]), float(r_opt[1])]

                done += 1
                if progress_callback is not None:
                    if progress_callback(done, total) is False:
                        cancelled = True
                        break
                ckpt.maybe_save(_current_results, done=done, total=total)
            if verbose and n_width >= 10 and (i % max(1, n_width // 10) == 0):
                print(f"  ... width row {i + 1}/{n_width}")

    else:
        # ------------------------------------------------------------
        # Parallel über mehrere Prozesse (ProcessPoolExecutor) - siehe
        # Docstring: identisches Muster wie im Original.
        # ------------------------------------------------------------
        optimizer_kwargs = {k: getattr(self, k) for k in self.DEFAULTS}
        tasks = [
            (i, j, win_input_val, width_val, list(r0), r_bounds, alpha, optimizer_kwargs)
            for i, width_val in enumerate(width_vals)
            for j, win_input_val in enumerate(win_input_vals)
            if not (np.isfinite(uniformity_weighted_grid[i, j]) and np.isfinite(r_x_grid[i, j])
                    and np.isfinite(r_y_grid[i, j]))
        ]
        done = n_done_before

        if verbose:
            skipped = total - len(tasks)
            skip_msg = f" ({skipped} bereits aus Checkpoint vorhanden)" if skipped else ""
            print(f"Verteile {len(tasks)} Punkte auf {n_jobs_resolved} Prozesse{skip_msg} "
                  f"(warm_start wird bei n_jobs>1 ignoriert)...")

        if not tasks:
            if verbose:
                print("  Alle Punkte bereits vorhanden - nichts zu tun.")
        else:
            with ProcessPoolExecutor(max_workers=n_jobs_resolved,
                                      initializer=pool_initializer,
                                      initargs=pool_initargs) as executor:
                futures = {executor.submit(_amp_dependence_worker_weighted, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        i, j, uniformity_w, eta_w, r_x, r_y = future.result()
                    except Exception:
                        task = futures[future]
                        i, j = task[0], task[1]
                        uniformity_w = eta_w = r_x = r_y = None

                    if uniformity_w is not None:
                        uniformity_weighted_grid[i, j] = uniformity_w
                        eta_weighted_grid[i, j] = eta_w
                        r_x_grid[i, j] = r_x
                        r_y_grid[i, j] = r_y

                    done += 1
                    if progress_callback is not None and not cancelled:
                        if progress_callback(done, total) is False:
                            cancelled = True
                            for f in futures:
                                f.cancel()
                    ckpt.maybe_save(_current_results, done=done, total=total)
                    if verbose and total >= 10 and done % max(1, total // 10) == 0:
                        print(f"  ... {done}/{total} Punkte fertig")

    if ckpt.active:
        ckpt.maybe_save(_current_results, done=done, total=total, force=True)

    self.results['scan2d_amp_weighted'] = dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_weighted_grid=uniformity_weighted_grid, eta_weighted_grid=eta_weighted_grid,
        r_x_grid=r_x_grid, r_y_grid=r_y_grid,
        alpha=alpha, r_bounds=r_bounds, sigma_atom=self.sigma_atom,
    )

    if verbose:
        valid = np.isfinite(uniformity_weighted_grid)
        status = "cancelled" if cancelled else "completed"
        print(f"Scan {status} ({np.sum(valid)}/{uniformity_weighted_grid.size} valid points, "
              f"{done}/{total} computed).")
        print("=" * 60)

    return win_input_vals, width_vals, uniformity_weighted_grid, eta_weighted_grid, r_x_grid, r_y_grid


def _amp_dependence_worker_weighted(task):
    """
    Wie _amp_dependence_worker() in der Optimizer-Datei, aber für die
    gewichtete Zielfunktion: task = (i, j, win_input_val, width_val, x0,
    r_bounds, alpha, optimizer_kwargs). Gibt (i, j, uniformity_weighted,
    eta_weighted, r_x, r_y) zurück, oder (i, j, None, None, None, None) bei
    einem ungültigen Punkt.
    """
    i, j, win_input_val, width_val, x0, r_bounds, alpha, optimizer_kwargs = task

    opt = MultitoneFlatTopOptimizer(out_dir="..", **optimizer_kwargs)
    try:
        win_eff = opt.win_input_to_win(win_input_val)
    except ValueError:
        return (i, j, None, None, None, None)

    def objective(p):
        amps = amps_from_ratios(p[0], p[1], opt.N_x, opt.N_y)
        val = opt._evaluate_weighted_only(win_eff, width_val, amps=amps)
        if val is None:
            return 1e10
        return alpha * val['uniformity_weighted'] + (1 - alpha) * val['eta_weighted']

    result = minimize(
        objective, x0=list(x0), method='Nelder-Mead',
        bounds=[r_bounds, r_bounds],
        options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300},
    )
    r_opt = result.x
    amps_opt = amps_from_ratios(r_opt[0], r_opt[1], opt.N_x, opt.N_y)
    details = opt._evaluate_weighted_only(win_eff, width_val, amps=amps_opt)
    if details is None:
        return (i, j, None, None, None, None)

    return (i, j, float(details['uniformity_weighted']), float(details['eta_weighted']),
            float(r_opt[0]), float(r_opt[1]))


def get_scan_amp_results_weighted(self):
    """Wie get_scan_amp_results(), aber für scan_win_width_amplitude_dependence_weighted()."""
    if 'scan2d_amp_weighted' not in self.results:
        raise RuntimeError("scan_win_width_amplitude_dependence_weighted() must be called first.")
    res = dict(self.results['scan2d_amp_weighted'])  # shallow copy
    res.update(
        N_x=self.N_x, N_y=self.N_y,
        f1=self.f1, f2=self.f2, fLO=self.fLO,
        lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
        profile=self.profile,
        atom_mass=self.atom_mass, atom_temperature=self.atom_temperature, trap_freq_r=self.trap_freq_r,
    )
    return res


def save_scan_amp_results_weighted(self, filepath=None, overwrite=False):
    """
    Wie save_scan_amp_results(), aber für get_scan_amp_results_weighted().
    Default-Dateiname: "scan_amp_data_weighted_N{N_x}x{N_y}_{n_win}x{n_width}
    pts_{Airy|Gauss}.pkl" in DEFAULT_RESULTS_DIR.

    overwrite: siehe save_scan_weighted_results() oben - identische
    Begründung/Verwendung (checkpoint_path == finaler Speicherpfad in den
    GUI-Start-Dialogen).
    """
    if filepath is None:
        res = self.results.get('scan2d_amp_weighted', {})
        n_win = len(res.get('win_input_vals', []))
        n_width = len(res.get('width_vals', []))
        profile_tag = "Airy" if self.profile == "airy" else "Gauss" if self.profile == "gaussian" else self.profile
        filename = f"scan_amp_data_weighted_N{self.N_x}x{self.N_y}_{n_win}x{n_width}pts_{profile_tag}.pkl"
        filepath = _resolve_pickle_path(DEFAULT_RESULTS_DIR, filename)
    else:
        filepath = FilePath(filepath)
        if filepath.exists() and not overwrite:
            filepath = _resolve_pickle_path(filepath.parent, filepath.name)

    with open(filepath, 'wb') as f:
        pickle.dump(self.get_scan_amp_results_weighted(), f)
    print(f"Weighted amplitude-dependence scan results saved: {filepath}")
    return filepath


# ======================================================================
# Teil 3: Post-hoc-Nachoptimierung auffälliger (r_x, r_y)-Sprungstellen
# (2026-08-21, auf User-Wunsch - siehe Diagnose im Projekt-Status-Doc)
# ======================================================================
_DEFAULT_REFINE_STARTS = ((1.0, 1.0), (1.0, 10.0), (10.0, 1.0), (5.0, 5.0), (0.5, 0.5))


def refine_scan_amp_results_weighted(self, results=None, mask=None, z_thresh=3.5,
                                      extra_starts=None, alpha=None, r_bounds=None,
                                      tol_rel=1e-3, verbose=True):
    """
    Post-hoc-Nachoptimierung für scan_win_width_amplitude_dependence_weighted()-
    Ergebnisse.

    HINTERGRUND (Diagnose 2026-08-21, siehe Projekt-Status-Doc): der Scan
    läuft standardmäßig sequentiell mit Warm-Start (jeder Gitterpunkt startet
    die Nelder-Mead-Optimierung beim (r_x, r_y)-Ergebnis des Vorgängerpunkts).
    Das ist schnell und meistens robust, kann aber in Bereichen, wo die
    gewichtete Zielfunktion mehrere fast gleich gute lokale Optima hat
    (bestätigt z.B. bei width=0.356MHz zwischen win_input=1.6mm und 1.7mm),
    in die falsche Senke laufen und dort auch für nachfolgende Punkte
    hängen bleiben. Diese Funktion prüft NUR die auffälligen Punkte
    (detect_amp_discontinuities()) noch einmal nach - mit mehreren festen
    Startpunkten statt eines einzigen Warm-Starts - und übernimmt einen
    Alternativpunkt NUR, wenn er den kombinierten Zielwert tatsächlich
    messbar verbessert. Viele erkannte Sprünge sind KEINE Artefakte, sondern
    echte, benachbarte Optima etwa gleicher Güte (siehe Diagnose) - diese
    bleiben unverändert, weil keine Mehrfachstart-Optimierung dort etwas
    Besseres findet.

    results: dict wie von get_scan_amp_results_weighted()/
        load_amp_scan_results() (z.B. eine geladene .pkl-Datei). Default:
        self.get_scan_amp_results_weighted() (der zuletzt in DIESER
        Optimizer-Instanz gelaufene gewichtete Amplituden-Scan).
        WICHTIG: `self` muss zu `results` PASSEN (gleiche N_x/N_y, f1/f2/
        fLO/lambda_opt/theta_max/f_band, profile, atom_mass/atom_temperature/
        trap_freq_r) - sonst werden falsche Punkte nachgerechnet. Beim
        direkten Weiterverwenden von `self` nach einem eigenen
        scan_win_width_amplitude_dependence_weighted()-Aufruf ist das
        automatisch der Fall. Beim Nachladen einer alten .pkl-Datei muss
        eine neue MultitoneFlatTopOptimizer-Instanz mit exakt den im dict
        gespeicherten Parametern konstruiert werden (weighted_n_grid/
        weighted_n_sigma sind NICHT im dict gespeichert - dort greift dann
        der Klassen-Default, siehe DEFAULTS in weighted_multitone_flattop_
        optimizer.py; für eine exakte Reproduktion des Original-Scans daher
        nach Möglichkeit denselben weighted_n_grid-Wert wie beim Original-
        Scan explizit angeben).
    mask: eigene boolsche (n_width, n_win_input)-Maske statt der
        automatischen Erkennung über detect_amp_discontinuities(z_thresh).
    z_thresh: Schwelle für detect_amp_discontinuities() (nur bei mask=None).
    extra_starts: zusätzliche (r_x, r_y)-Startpunkte für die Mehrfachstart-
        Optimierung, ZUSÄTZLICH zum ursprünglich gemeldeten Punkt selbst
        (der immer als ein Start mitläuft - ein bereits optimaler Punkt kann
        dadurch nie verschlechtert werden). Default: 5 über den r_bounds-
        Bereich verteilte Punkte (siehe _DEFAULT_REFINE_STARTS), auf
        r_bounds geclippt.
    alpha/r_bounds: Default aus results['alpha']/results['r_bounds'].
    tol_rel: relative Mindestverbesserung des kombinierten Zielwerts
        (alpha*uniformity_weighted+(1-alpha)*eta_weighted), ab der ein
        gefundener Alternativpunkt tatsächlich übernommen wird (Default
        0.1%% - verhindert, dass reines Optimierungs-/Numerik-Rauschen als
        "Verbesserung" gewertet wird).
    verbose: Fortschritts-/Zusammenfassungsausgabe.

    Gibt (refined_results, report) zurück:
      - refined_results: BEREINIGTE KOPIE von results (Original bleibt
        unangetastet, gleiches Muster wie clean_amp_scan_results()).
      - report: Liste von dicts, EIN Eintrag pro tatsächlich geprüftem
        Punkt (nicht nur den geänderten - volle Transparenz, welche Punkte
        angeschaut UND welche davon verändert wurden): i, j, win_input_mm,
        width_mhz, r_x_before, r_y_before, combined_before, r_x_after,
        r_y_after, combined_after, changed (bool).
    """
    from weighted_multitone_amplitude_dependence_plots import detect_amp_discontinuities

    if results is None:
        results = self.get_scan_amp_results_weighted()

    if mask is None:
        mask, _jump_z = detect_amp_discontinuities(results, z_thresh=z_thresh)

    alpha = alpha if alpha is not None else results.get('alpha', 0.7)
    r_bounds = r_bounds if r_bounds is not None else results.get('r_bounds', (0.0, 2.0))
    lo, hi = r_bounds

    starts_raw = list(extra_starts) if extra_starts is not None else list(_DEFAULT_REFINE_STARTS)
    fixed_starts = [(min(max(sx, lo), hi), min(max(sy, lo), hi)) for sx, sy in starts_raw]

    win_input_vals = results['win_input_vals']
    width_vals = results['width_vals']
    r_x_grid = results['r_x_grid']
    r_y_grid = results['r_y_grid']

    cleaned = dict(results)
    r_x_out = r_x_grid.copy()
    r_y_out = r_y_grid.copy()
    uniformity_out = results.get('uniformity_weighted_grid')
    uniformity_out = uniformity_out.copy() if uniformity_out is not None else None
    eta_out = results.get('eta_weighted_grid')
    eta_out = eta_out.copy() if eta_out is not None else None

    idxs = list(zip(*np.nonzero(mask)))
    report = []
    n_changed = 0

    if verbose:
        print(f"refine_scan_amp_results_weighted: {len(idxs)} auffällige Punkt(e) "
              f"(z_thresh={z_thresh}) werden mit {1 + len(fixed_starts)} Startpunkten "
              f"nachgerechnet...")

    for i, j in idxs:
        win_input_val = win_input_vals[j]
        width_val = width_vals[i]
        try:
            win_eff = self.win_input_to_win(win_input_val)
        except ValueError:
            continue

        def objective(p, win_eff=win_eff, width_val=width_val):
            amps = amps_from_ratios(p[0], p[1], self.N_x, self.N_y)
            val = self._evaluate_weighted_only(win_eff, width_val, amps=amps)
            if val is None:
                return 1e10
            return alpha * val['uniformity_weighted'] + (1 - alpha) * val['eta_weighted']

        r_x_before = float(r_x_grid[i, j])
        r_y_before = float(r_y_grid[i, j])
        combined_before = objective([r_x_before, r_y_before])

        best_combined = combined_before
        best_r = (r_x_before, r_y_before)
        for x0 in [(r_x_before, r_y_before)] + fixed_starts:
            result = minimize(
                objective, x0=list(x0), method='Nelder-Mead',
                bounds=[r_bounds, r_bounds],
                options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300},
            )
            if result.fun < best_combined:
                best_combined = result.fun
                best_r = (float(result.x[0]), float(result.x[1]))

        improvement = combined_before - best_combined
        threshold = max(tol_rel * abs(combined_before), 1e-9)
        changed = improvement > threshold

        if changed:
            n_changed += 1
            r_x_out[i, j] = best_r[0]
            r_y_out[i, j] = best_r[1]
            amps_best = amps_from_ratios(best_r[0], best_r[1], self.N_x, self.N_y)
            details = self._evaluate_weighted_only(win_eff, width_val, amps=amps_best)
            if details is not None:
                if uniformity_out is not None:
                    uniformity_out[i, j] = details['uniformity_weighted']
                if eta_out is not None:
                    eta_out[i, j] = details['eta_weighted']

        report.append(dict(
            i=int(i), j=int(j),
            win_input_mm=float(win_input_val) * 1e3, width_mhz=float(width_val) * 1e-6,
            r_x_before=r_x_before, r_y_before=r_y_before, combined_before=float(combined_before),
            r_x_after=best_r[0], r_y_after=best_r[1], combined_after=float(best_combined),
            changed=changed,
        ))
        if verbose:
            tag = "GEÄNDERT" if changed else "bestätigt"
            print(f"  win={win_input_val * 1e3:.2f}mm, width={width_val * 1e-6:.3f}MHz: "
                  f"({r_x_before:.3f},{r_y_before:.3f})->({best_r[0]:.3f},{best_r[1]:.3f}) "
                  f"[{combined_before:.5f}->{best_combined:.5f}]  [{tag}]")

    cleaned['r_x_grid'] = r_x_out
    cleaned['r_y_grid'] = r_y_out
    if uniformity_out is not None:
        cleaned['uniformity_weighted_grid'] = uniformity_out
    if eta_out is not None:
        cleaned['eta_weighted_grid'] = eta_out

    if verbose:
        print(f"refine_scan_amp_results_weighted: {n_changed}/{len(idxs)} Punkt(e) tatsächlich "
              f"verbessert und übernommen (Rest: bestätigt als bereits optimal bzw. keine "
              f"bessere Alternative gefunden).")

    return cleaned, report


# ======================================================================
# Monkey-Patch: die neuen Methoden ganz normal auf der Klasse verfügbar
# machen (opt.scan_win_width_weighted_uniformity(...) statt einer
# freistehenden Funktion) - passiert automatisch beim Import dieses Moduls.
# ======================================================================
MultitoneFlatTopOptimizer.scan_win_width_weighted_uniformity = scan_win_width_weighted_uniformity
MultitoneFlatTopOptimizer.get_scan_weighted_results = get_scan_weighted_results
MultitoneFlatTopOptimizer.save_scan_weighted_results = save_scan_weighted_results
MultitoneFlatTopOptimizer.scan_win_width_amplitude_dependence_weighted = scan_win_width_amplitude_dependence_weighted
MultitoneFlatTopOptimizer.get_scan_amp_results_weighted = get_scan_amp_results_weighted
MultitoneFlatTopOptimizer.save_scan_amp_results_weighted = save_scan_amp_results_weighted
MultitoneFlatTopOptimizer.refine_scan_amp_results_weighted = refine_scan_amp_results_weighted


if __name__ == "__main__":
    print(__doc__)
