"""
Combinated_Optimization/combined_amp_scan_methods.py
=======================================================

Wie combined_scan_methods.py, aber fuer den AMPLITUDEN-OPTIMIERTEN Scan
(an jedem (win_input, width)-Gitterpunkt wird eine eigene (r_x, r_y)-
Optimierung durchgefuehrt - siehe scan_win_width_amplitude_dependence()/
scan_win_width_amplitude_dependence_weighted()), statt der festen
Amplituden aus combined_scan_methods.py.

Kombinationsprinzip identisch zu combined_scan_methods.py (Normierung +
Mittelwert + Uneinigkeits-Strafterm auf uniformity_grid/crosstalk_grid
[hart] vs. uniformity_weighted_grid/eta_weighted_grid [gewichtet] -> Score
-> Region) - die Kernrechnung (`_combine_grids()`) wird direkt aus
combined_scan_methods.py importiert und wiederverwendet, NICHT dupliziert.

Es gibt hier ZWEI unterschiedliche Verfahren, um an die pro Punkt
optimierten Amplituden-Verhaeltnisse r_x/r_y zu kommen (siehe Chat
"Amplituden Abhängigkeit"):

1. GETRENNT (scan_win_width_amplitude_dependence_combined(), aelteres
   Verfahren, siehe auch combine_existing_datasets.py): hart und
   gewichtet werden an jedem Punkt UNABHAENGIG voneinander optimiert
   (jeweils gegen ihr EIGENES Ziel) - r_x_grid_hart/r_y_grid_hart und
   r_x_grid_weighted/r_y_grid_weighted koennen sich daher unterscheiden.
   Erst HINTERHER werden die dabei (an VERSCHIEDENEN r_x/r_y) erreichten
   Uniformity/Crosstalk-Werte kombiniert - fuer die Region-/Score-
   Uebersicht taugt das, aber der resultierende "primaere" r_x/r_y-Satz
   (r_grid_source) ist dann jeweils nur fuer EINES der beiden Kriterien
   tatsaechlich optimal, nicht fuer die Kombination.

2. GEMEINSAM/JOINT (scan_win_width_amplitude_dependence_combined_joint(),
   siehe deren Docstring - EMPFOHLEN): an jedem Punkt wird nur EINE
   Nelder-Mead-Optimierung ueber (r_x, r_y) durchgefuehrt, die DIREKT
   gegen die Kombination aus hart+gewichtet minimiert (beide Metriken an
   JEDEM Optimierungsschritt am SELBEN r_x/r_y ausgewertet, ueber
   MultitoneFlatTopOptimizer._evaluate(..., weighted=True) - eine
   Auswertung liefert bereits beide Metrik-Paare). r_x_grid_hart und
   r_x_grid_weighted sind hier daher immer IDENTISCH (kommen aus
   derselben Optimierung) - die gefundene Amplitude ist automatisch fuer
   BEIDE Kriterien gleichzeitig gueltig, nicht nur fuer eines.

Fuer die Kompatibilitaet mit der UNVERAENDERTEN AmplitudeScanPlotter
(weighted_multitone_amplitude_dependence_plots.py, die ein einzelnes
r_x_grid/r_y_grid erwartet) wird in BEIDEN Faellen zusaetzlich EIN
"primaerer" Satz (r_x_grid/r_y_grid) mitgeliefert - das erlaubt, das
kombinierte Ergebnis-dict direkt und unveraendert an AmplitudeScanPlotter
zu uebergeben (6-Panel-Ansicht: Uniformity/Crosstalk je hart+gewichtet,
plus r_x, r_y - das unterstuetzt AmplitudeScanPlotter bereits nativ,
siehe deren Docstring "wer beide Scans zusammenfuehrt ... bekommt beide
Paare gleichzeitig plotbar").

Nutzung (empfohlen, gemeinsame/jointe Optimierung):
    from combined_amp_scan_methods import MultitoneFlatTopOptimizer

    opt = MultitoneFlatTopOptimizer(out_dir=".", N_x=3, N_y=4)
    opt.scan_win_width_amplitude_dependence_combined_joint(
        win_input_range=(0.8e-3, 1.7e-3), width_range=(0.2e6, 0.4e6),
        n_win_input=15, n_width=15, n_jobs=4,
    )
    opt.save_scan_amp_results_combined()   # -> Results/scan_amp_data_combined_....pkl
"""

import os
import sys
import pickle
from pathlib import Path as FilePath
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.optimize import minimize

_WEIGHTED_DIR = FilePath(__file__).resolve().parent.parent / "Weighted_Optimization"
if str(_WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(_WEIGHTED_DIR))

from weighted_multitone_flattop_optimizer import (  # noqa: E402
    MultitoneFlatTopOptimizer,
    _resolve_pickle_path,
    amps_from_ratios,
)
import weighted_amp_scan_methods  # noqa: E402,F401  # Import-Nebeneffekt: patcht
# scan_win_width_amplitude_dependence_weighted()/get_scan_amp_results_weighted()/
# save_scan_amp_results_weighted() auf MultitoneFlatTopOptimizer.

# Kernrechnung (Normierung + Mittelwert + Uneinigkeits-Strafterm + Region)
# UND der gemeinsame Results/Bilder-Ordner werden direkt aus
# combined_scan_methods.py wiederverwendet - EIN gemeinsamer Ordner fuer
# beide Scan-Arten in Combinated_Optimization.
from combined_scan_methods import (  # noqa: E402
    DEFAULT_RESULTS_DIR, DEFAULT_IMAGES_DIR, _combine_grids, _derive_checkpoint_paths,
)
import scan_checkpoint  # noqa: E402  # liegt (identische Kopie) in Weighted_Optimization,
# ueber denselben sys.path-Eintrag oben erreichbar.


def build_combined_amp_scan_results(hard, weighted, alpha=0.7, combo_lambda=0.75,
                                     combo_percentile=25.0, r_grid_source="weighted"):
    """Baut das vollstaendige kombinierte Ergebnis-dict aus den RAW-Outputs
    von get_scan_amp_results() (hart) und get_scan_amp_results_weighted()
    (weighted). r_grid_source ('weighted' oder 'hart') waehlt, welcher der
    beiden r_x/r_y-Saetze als 'primaerer' r_x_grid/r_y_grid im Ergebnis
    landet (fuer die Kompatibilitaet mit AmplitudeScanPlotter) - BEIDE
    Saetze bleiben in jedem Fall zusaetzlich unter r_x_grid_hart/
    r_x_grid_weighted (bzw. r_y_...) erhalten."""
    win_input_vals = np.asarray(hard['win_input_vals'])
    width_vals = np.asarray(hard['width_vals'])
    if not (np.array_equal(win_input_vals, weighted['win_input_vals'])
            and np.array_equal(width_vals, weighted['width_vals'])):
        raise ValueError(
            "Harter und gewichteter Amplituden-Scan liegen auf unterschiedlichen "
            "(waist,width)-Gittern - scan_win_width_amplitude_dependence_combined() "
            "ruft beide Scans intern mit identischen win_input_range/width_range/"
            "n_win_input/n_width auf derselben Optimizer-Instanz auf, das sollte "
            "nicht passieren."
        )
    if r_grid_source not in ("hart", "weighted"):
        raise ValueError(f"r_grid_source muss 'hart' oder 'weighted' sein, nicht {r_grid_source!r}.")

    U_hard = np.asarray(hard['uniformity_grid'])
    C_hard = np.asarray(hard['crosstalk_grid'])
    U_w = np.asarray(weighted['uniformity_weighted_grid'])
    C_w = np.asarray(weighted['eta_weighted_grid'])

    combo = _combine_grids(U_hard, C_hard, U_w, C_w, win_input_vals, width_vals,
                            alpha, combo_lambda, combo_percentile)

    r_x_hart = np.asarray(hard['r_x_grid'])
    r_y_hart = np.asarray(hard['r_y_grid'])
    r_x_weighted = np.asarray(weighted['r_x_grid'])
    r_y_weighted = np.asarray(weighted['r_y_grid'])
    r_x_grid = r_x_weighted if r_grid_source == "weighted" else r_x_hart
    r_y_grid = r_y_weighted if r_grid_source == "weighted" else r_y_hart

    return dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_grid=U_hard, crosstalk_grid=C_hard,
        uniformity_weighted_grid=U_w, eta_weighted_grid=C_w,
        r_x_grid=r_x_grid, r_y_grid=r_y_grid, r_grid_source=r_grid_source,
        r_x_grid_hart=r_x_hart, r_y_grid_hart=r_y_hart,
        r_x_grid_weighted=r_x_weighted, r_y_grid_weighted=r_y_weighted,
        r_bounds=hard.get('r_bounds'),
        N_x=hard['N_x'], N_y=hard['N_y'],
        f1=hard['f1'], f2=hard['f2'], fLO=hard['fLO'],
        lambda_opt=hard['lambda_opt'], theta_max=hard['theta_max'], f_band=hard['f_band'],
        profile=hard['profile'],
        sigma_atom=weighted.get('sigma_atom'),
        atom_mass=weighted.get('atom_mass'), atom_temperature=weighted.get('atom_temperature'),
        trap_freq_r=weighted.get('trap_freq_r'),
        **combo,
    )


def build_combined_amp_scan_results_joint(hard, weighted, alpha=0.7, combo_lambda=0.75,
                                           combo_percentile=25.0):
    """Wie build_combined_amp_scan_results(), aber fuer die GEMEINSAME
    (jointe) Optimierung (siehe scan_win_width_amplitude_dependence_
    combined_joint() und Modul-Docstring, Punkt 2).

    Im Unterschied zu build_combined_amp_scan_results() gibt es hier nur
    EIN r_x_grid/r_y_grid - hard und weighted wurden am SELBEN Punkt (r_x,
    r_y) ausgewertet, nicht an zwei unabhaengig gefundenen Optima. Es gibt
    daher auch kein r_grid_source-Argument: r_x_grid_hart/r_x_grid_weighted
    (bzw. r_y_...) werden trotzdem beide gesetzt (identisch zueinander),
    ausschliesslich fuer die Kompatibilitaet mit Code, der diese Schluessel
    erwartet (z.B. fit_combined_amp_region.py._r_at_best()).

    Die Region-/Score-Uebersicht (combo, ueber _combine_grids()) verwendet
    weiterhin die gitterweite Min-Max-Normierung - das ist reine
    Nachbearbeitung/Visualisierung ueber die BEREITS fertigen Grids und hat
    keinen Einfluss mehr auf die (schon abgeschlossene) Wahl von r_x/r_y
    selbst, die waehrend der Optimierung OHNE Normierung (roh) minimiert
    wurde (siehe Docstring von scan_win_width_amplitude_dependence_
    combined_joint() fuer die Begruendung: eine gitterweite Normierung
    steht waehrend der Optimierung eines einzelnen Punktes noch nicht zur
    Verfuegung)."""
    win_input_vals = np.asarray(hard['win_input_vals'])
    width_vals = np.asarray(hard['width_vals'])

    U_hard = np.asarray(hard['uniformity_grid'])
    C_hard = np.asarray(hard['crosstalk_grid'])
    U_w = np.asarray(weighted['uniformity_weighted_grid'])
    C_w = np.asarray(weighted['eta_weighted_grid'])

    combo = _combine_grids(U_hard, C_hard, U_w, C_w, win_input_vals, width_vals,
                            alpha, combo_lambda, combo_percentile)

    r_x_grid = np.asarray(hard['r_x_grid'])
    r_y_grid = np.asarray(hard['r_y_grid'])

    return dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_grid=U_hard, crosstalk_grid=C_hard,
        uniformity_weighted_grid=U_w, eta_weighted_grid=C_w,
        r_x_grid=r_x_grid, r_y_grid=r_y_grid, r_grid_source="joint",
        r_x_grid_hart=r_x_grid, r_y_grid_hart=r_y_grid,
        r_x_grid_weighted=r_x_grid, r_y_grid_weighted=r_y_grid,
        r_bounds=hard.get('r_bounds'),
        joint_optimization=True,
        N_x=hard['N_x'], N_y=hard['N_y'],
        f1=hard['f1'], f2=hard['f2'], fLO=hard['fLO'],
        lambda_opt=hard['lambda_opt'], theta_max=hard['theta_max'], f_band=hard['f_band'],
        profile=hard['profile'],
        sigma_atom=weighted.get('sigma_atom'),
        atom_mass=weighted.get('atom_mass'), atom_temperature=weighted.get('atom_temperature'),
        trap_freq_r=weighted.get('trap_freq_r'),
        **combo,
    )


# ======================================================================
# Worker-Funktion fuer die GEMEINSAME (jointe) Optimierung bei n_jobs>1
# (muss auf Modulebene stehen, siehe _amp_dependence_worker() im Original -
# ProcessPoolExecutor kann keine lokal definierten/gebundenen Funktionen
# picklen).
# ======================================================================
def _amp_dependence_worker_combined_joint(task):
    """
    task: (i, j, win_input_val, width_val, x0, r_bounds, alpha, combo_lambda,
           optimizer_kwargs)

    Fuehrt GENAU EINEN (win_input, width)-Punkt der gemeinsamen (jointen)
    kombinierten Amplituden-Optimierung aus (siehe scan_win_width_amplitude_
    dependence_combined_joint() fuer die volle Erklaerung): EINE Nelder-
    Mead-Optimierung ueber (r_x, r_y), die
        J(r_x,r_y) = alpha*Uniformity_kombi_roh + (1-alpha)*Crosstalk_kombi_roh
    minimiert (roh = ohne gitterweite Normierung - siehe dortiger
    Docstring), wobei Uniformity_hart/Crosstalk_hart UND
    Uniformity_w/Crosstalk_w bei JEDEM Optimierungsschritt am SELBEN
    (r_x,r_y) ausgewertet werden (eine einzige _evaluate(...,
    weighted=True)-Auswertung liefert beide Metrik-Paare zugleich).

    Gibt (i, j, uniformity_hart, crosstalk_hart, uniformity_weighted,
    crosstalk_weighted, r_x, r_y) zurueck, oder (i, j, None, None, None,
    None, None, None) bei einem ungueltigen Punkt (z.B. win_input<=0,
    leere Maske, oder sigma_atom ungueltig/keine gewichtete Auswertung
    moeglich).
    """
    (i, j, win_input_val, width_val, x0, r_bounds, alpha, combo_lambda,
     optimizer_kwargs) = task

    opt = MultitoneFlatTopOptimizer(out_dir="..", **optimizer_kwargs)
    try:
        win_eff = opt.win_input_to_win(win_input_val)
    except ValueError:
        return (i, j, None, None, None, None, None, None)

    point_grid = opt._build_dynamic_grid(win_eff, width_val)

    def objective(p):
        amps = amps_from_ratios(p[0], p[1], opt.N_x, opt.N_y)
        val = opt._evaluate(win_eff, width_val, amps=amps, grid=point_grid, weighted=True)
        if val is None:
            return 1e10
        U_h, C_h = val['uniformity'], val['eta']
        U_w, C_w = val.get('uniformity_weighted'), val.get('eta_weighted')
        if U_w is None or C_w is None or not (np.isfinite(U_w) and np.isfinite(C_w)):
            return 1e10
        U_kombi = 0.5 * (U_h + U_w) + combo_lambda * abs(U_h - U_w)
        C_kombi = 0.5 * (C_h + C_w) + combo_lambda * abs(C_h - C_w)
        return alpha * U_kombi + (1.0 - alpha) * C_kombi

    result = minimize(
        objective, x0=list(x0), method='Nelder-Mead',
        bounds=[r_bounds, r_bounds],
        options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300},
    )
    r_opt = result.x
    amps_opt = amps_from_ratios(r_opt[0], r_opt[1], opt.N_x, opt.N_y)
    details = opt._evaluate(win_eff, width_val, amps=amps_opt, grid=point_grid, weighted=True)
    if details is None:
        return (i, j, None, None, None, None, None, None)
    U_w_final = details.get('uniformity_weighted')
    C_w_final = details.get('eta_weighted')
    if U_w_final is None or C_w_final is None or not (np.isfinite(U_w_final) and np.isfinite(C_w_final)):
        return (i, j, None, None, None, None, None, None)

    return (i, j, float(details['uniformity']), float(details['eta']),
            float(U_w_final), float(C_w_final), float(r_opt[0]), float(r_opt[1]))


# ======================================================================
# Monkey-Patch auf MultitoneFlatTopOptimizer - identisches Muster wie
# combined_scan_methods.py/weighted_amp_scan_methods.py.
# ======================================================================
def scan_win_width_amplitude_dependence_combined(self, win_input_range, width_range,
                                                   n_win_input=15, n_width=15,
                                                   alpha=0.7, r_bounds=(0.0, 2.0), r0=(1.0, 1.0),
                                                   warm_start=True,
                                                   combo_lambda=0.75, combo_percentile=25.0,
                                                   r_grid_source="weighted",
                                                   verbose=True, progress_callback=None,
                                                   n_jobs=1, pool_initializer=None, pool_initargs=(),
                                                   checkpoint_path=None,
                                                   checkpoint_interval_s=scan_checkpoint.CHECKPOINT_INTERVAL_S):
    """
    ACHTUNG: dies ist das AELTERE, GETRENNTE Verfahren (siehe Modul-
    Docstring, Punkt 1) - hart und gewichtet werden UNABHAENGIG voneinander
    optimiert und koennen daher unterschiedliche r_x/r_y liefern. Fuer eine
    Amplitude, die direkt auf den kombinierten Fall zugeschnitten ist,
    stattdessen scan_win_width_amplitude_dependence_combined_joint()
    verwenden (empfohlen, siehe deren Docstring).

    Fuehrt NACHEINANDER die unveraenderte harte
    scan_win_width_amplitude_dependence() und die atom-gewichtete
    scan_win_width_amplitude_dependence_weighted() mit IDENTISCHEN
    win_input_range/width_range/n_win_input/n_width/alpha/r_bounds/r0/
    warm_start/n_jobs auf dieser Optimizer-Instanz aus (beide Scans bauen
    ihr Gitter identisch per np.linspace() auf) und kombiniert die
    Ergebnisse anschliessend (siehe build_combined_amp_scan_results() und
    Modul-Docstring).

    ACHTUNG Laufzeit: JEDER der insgesamt 2*n_win_input*n_width Punkte
    kostet eine vollstaendige Nelder-Mead-(r_x,r_y)-Optimierung - deutlich
    teurer als der Fest-Amplituden-Scan (combined_scan_methods.py). Fuer
    einen ersten Durchlauf n_win_input/n_width klein halten (12-20) und
    n_jobs>1 (bzw. -1 fuer alle Kerne) nutzen, siehe Docstring von
    scan_win_width_amplitude_dependence().

    combo_lambda/combo_percentile: wie in combined_scan_methods.py.
    r_grid_source: siehe build_combined_amp_scan_results().

    checkpoint_path / checkpoint_interval_s: wie bei scan_win_width_
    combined_uniformity() in combined_scan_methods.py - EIN vom Nutzer
    gewaehlter Pfad fuer den gesamten kombinierten Scan wird intern in
    ZWEI eigene Pfade fuer den harten und den gewichteten Teilscan
    aufgeteilt (siehe _derive_checkpoint_paths()), die unabhaengig
    voneinander stuendlich (Default) ihren Zwischenstand sichern und beim
    Neustart jeweils automatisch an der Stelle fortsetzen, an der sie
    stehen geblieben sind.

    Speichert das Ergebnis in self.results['scan2d_amp_combined'] und gibt
    es zurueck.
    """
    total_each = n_win_input * n_width
    ckpt_hard, ckpt_weighted = _derive_checkpoint_paths(checkpoint_path)

    def hard_progress(done, total):
        if progress_callback is None:
            return True
        return progress_callback(done, 2 * total_each)

    def weighted_progress(done, total):
        if progress_callback is None:
            return True
        return progress_callback(total_each + done, 2 * total_each)

    if verbose:
        print("\n" + "=" * 70)
        print("KOMBINIERTER Amplituden-Abhaengigkeits-Scan: 1/2 harte Masken-Metriken "
              "(pro Punkt r_x/r_y-Optimierung unter uniformity/crosstalk) ...")
        print("=" * 70)
    self.scan_win_width_amplitude_dependence(
        win_input_range=win_input_range, width_range=width_range,
        n_win_input=n_win_input, n_width=n_width, alpha=alpha, r_bounds=r_bounds, r0=r0,
        warm_start=warm_start, verbose=verbose, progress_callback=hard_progress,
        n_jobs=n_jobs, pool_initializer=pool_initializer, pool_initargs=pool_initargs,
        checkpoint_path=ckpt_hard, checkpoint_interval_s=checkpoint_interval_s,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("KOMBINIERTER Amplituden-Abhaengigkeits-Scan: 2/2 atom-gewichtete Metriken "
              "(pro Punkt r_x/r_y-Optimierung unter uniformity_w/crosstalk_w) ...")
        print("=" * 70)
    self.scan_win_width_amplitude_dependence_weighted(
        win_input_range=win_input_range, width_range=width_range,
        n_win_input=n_win_input, n_width=n_width, alpha=alpha, r_bounds=r_bounds, r0=r0,
        warm_start=warm_start, verbose=verbose, progress_callback=weighted_progress,
        n_jobs=n_jobs, pool_initializer=pool_initializer, pool_initargs=pool_initargs,
        checkpoint_path=ckpt_weighted, checkpoint_interval_s=checkpoint_interval_s,
    )

    hard = self.get_scan_amp_results()
    weighted = self.get_scan_amp_results_weighted()
    combined = build_combined_amp_scan_results(
        hard, weighted, alpha=alpha, combo_lambda=combo_lambda,
        combo_percentile=combo_percentile, r_grid_source=r_grid_source,
    )
    self.results['scan2d_amp_combined'] = combined

    if verbose:
        print("\n" + "=" * 70)
        b, r = combined['best'], combined['region']
        if b['win_input'] is not None:
            print(f"Kombiniert bester Punkt: win_input={b['win_input']*1e3:.4f} mm, "
                  f"width={b['width']*1e-6:.3f} MHz -> "
                  f"Uniformity_hart={b['uniformity_hart']*100:.2f}%, "
                  f"Crosstalk_hart={b['crosstalk_hart']*100:.3f}%, "
                  f"Uniformity_w={b['uniformity_weighted']*100:.2f}%, "
                  f"Crosstalk_w={b['crosstalk_weighted']*100:.3f}%")
        if r['win_input_min'] is not None:
            print(f"Kombinierte Region (beste {combo_percentile:.0f}% des Scores, groesstes "
                  f"eingeschriebenes Rechteck aus {r['n_points_region']}/{r['n_points_total']} "
                  f"Punkten im Akzeptanzbereich): "
                  f"win_input in [{r['win_input_min']*1e3:.4f}, {r['win_input_max']*1e3:.4f}] mm, "
                  f"width in [{r['width_min']*1e-6:.4f}, {r['width_max']*1e-6:.4f}] MHz")
        else:
            print("Kombinierte Region: kein gueltiges Rechteck gefunden (zu wenige valide Punkte).")
        print("=" * 70)

    return combined


def scan_win_width_amplitude_dependence_combined_joint(self, win_input_range, width_range,
                                                         n_win_input=15, n_width=15,
                                                         alpha=0.7, r_bounds=(0.0, 2.0), r0=(1.0, 1.0),
                                                         warm_start=True,
                                                         combo_lambda=0.75, combo_percentile=25.0,
                                                         verbose=True, progress_callback=None,
                                                         n_jobs=1, pool_initializer=None, pool_initargs=(),
                                                         checkpoint_path=None,
                                                         checkpoint_interval_s=scan_checkpoint.CHECKPOINT_INTERVAL_S):
    """
    GEMEINSAME (jointe) kombinierte Amplituden-Optimierung - EMPFOHLEN
    (siehe Chat "Amplituden Abhängigkeit" und Modul-Docstring, Punkt 2).

    Im Unterschied zu scan_win_width_amplitude_dependence_combined() (das
    hart und gewichtet an jedem Punkt GETRENNT optimiert und danach nur
    deren - an potenziell verschiedenen r_x/r_y erreichten - Metriken
    verrechnet) wird hier an JEDEM (win_input, width)-Gitterpunkt NUR EINE
    Nelder-Mead-Optimierung ueber (r_x, r_y) durchgefuehrt, die DIREKT

        J(r_x, r_y) = alpha * Uniformity_kombi_roh(r_x, r_y)
                      + (1 - alpha) * Crosstalk_kombi_roh(r_x, r_y)

    minimiert, mit

        Uniformity_kombi_roh = 0.5*(Uniformity_hart + Uniformity_w)
                                + combo_lambda * |Uniformity_hart - Uniformity_w|
        Crosstalk_kombi_roh  = 0.5*(Crosstalk_hart + Crosstalk_w)
                                + combo_lambda * |Crosstalk_hart - Crosstalk_w|

    wobei Uniformity_hart/Crosstalk_hart (harte Maske) UND
    Uniformity_w/Crosstalk_w (atom-gewichtet) bei JEDEM Optimierungs-
    schritt am SELBEN (r_x, r_y) ausgewertet werden - eine einzige
    MultitoneFlatTopOptimizer._evaluate(..., weighted=True)-Auswertung
    liefert bereits beide Metrik-Paare zugleich (die zusaetzliche
    atom-gewichtete Auswertung ist ein guenstiges lokales Sub-Grid, siehe
    _evaluate_weighted_metrics()). Die gefundene Amplitude ist dadurch
    automatisch fuer BEIDE Kriterien gleichzeitig gueltig, nicht nur fuer
    eines der beiden wie beim getrennten Verfahren.

    "roh" = OHNE gitterweite Min-Max-Normierung: eine Normierung ueber das
    GESAMTE Scan-Gitter (wie sie combined_scan_methods._combine_grids() fuer
    die Region-/Score-Uebersicht verwendet) steht waehrend der Optimierung
    EINES EINZELNEN Punktes noch nicht zur Verfuegung, da dafuer alle
    Gitterpunkte bereits bekannt sein muessten. combo_lambda wirkt hier
    daher direkt auf die RAW-Werte von Uniformity/Crosstalk (beide ohnehin
    dimensionslose Groessen aehnlicher Groessenordnung). Die anschliessende
    Region-/Score-Heatmap (build_combined_amp_scan_results_joint()) nutzt
    weiterhin die gitterweite Normierung wie gehabt - das ist reine
    Nachbearbeitung/Visualisierung ueber die bereits fertigen Grids, ohne
    Rueckwirkung auf die schon getroffene Amplituden-Wahl.

    alpha/r_bounds/r0/warm_start/verbose/progress_callback/n_jobs/
    pool_initializer/pool_initargs: identische Bedeutung wie bei
    scan_win_width_amplitude_dependence() (siehe deren Docstring in
    weighted_multitone_flattop_optimizer.py) - hier auf EINE statt zwei
    Optimierungen pro Punkt angewendet, daher insgesamt nur
    n_win_input*n_width (statt 2*n_win_input*n_width) Nelder-Mead-Laeufe
    noetig - typischerweise GUENSTIGER als das getrennte Verfahren, da die
    zusaetzliche atom-gewichtete Auswertung pro Optimierungsschritt
    vergleichsweise billig ist (kleines lokales Sub-Grid statt einer
    zweiten vollen Optimierung).

    checkpoint_path / checkpoint_interval_s: EIN einziger Checkpoint-Pfad
    fuer den GESAMTEN gemeinsamen Scan (anders als beim getrennten
    Verfahren gibt es hier keine Aufteilung in einen harten und einen
    gewichteten Teilscan-Pfad, da es nur noch EINEN Scan gibt) - wird
    direkt (ohne _derive_checkpoint_paths()) verwendet.

    Speichert das Ergebnis in self.results['scan2d_amp_combined'] (SELBER
    Schluessel wie beim getrennten Verfahren scan_win_width_amplitude_
    dependence_combined() - get_scan_amp_results_combined()/
    save_scan_amp_results_combined() funktionieren daher unveraendert,
    unabhaengig davon, welches der beiden Verfahren den Scan erzeugt hat)
    und gibt es zurueck.
    """
    win_input_vals = np.linspace(win_input_range[0], win_input_range[1], n_win_input)
    width_vals = np.linspace(width_range[0], width_range[1], n_width)

    resumed = scan_checkpoint.load_resumable(
        checkpoint_path, win_input_range, width_range, n_win_input, n_width,
        self.N_x, self.N_y,
        extra_match=dict(alpha=alpha, r_bounds=r_bounds, combo_lambda=combo_lambda,
                          joint_optimization=True),
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
            print(f"[Checkpoint] Setze gemeinsamen (jointen) Scan fort: "
                  f"{n_done_before}/{uniformity_grid.size} Punkte bereits vorhanden "
                  f"({checkpoint_path}).")
    else:
        uniformity_grid = np.full((n_width, n_win_input), np.nan)
        crosstalk_grid = np.full((n_width, n_win_input), np.nan)
        uniformity_weighted_grid = np.full((n_width, n_win_input), np.nan)
        eta_weighted_grid = np.full((n_width, n_win_input), np.nan)
        r_x_grid = np.full((n_width, n_win_input), np.nan)
        r_y_grid = np.full((n_width, n_win_input), np.nan)
        n_done_before = 0
    ckpt = scan_checkpoint.CheckpointWriter(checkpoint_path, checkpoint_interval_s, verbose=verbose)

    def _current_results():
        return dict(
            win_input_vals=win_input_vals, width_vals=width_vals,
            uniformity_grid=uniformity_grid, crosstalk_grid=crosstalk_grid,
            uniformity_weighted_grid=uniformity_weighted_grid, eta_weighted_grid=eta_weighted_grid,
            r_x_grid=r_x_grid, r_y_grid=r_y_grid,
            alpha=alpha, r_bounds=r_bounds, combo_lambda=combo_lambda,
            joint_optimization=True, sigma_atom=self.sigma_atom,
            N_x=self.N_x, N_y=self.N_y, f1=self.f1, f2=self.f2, fLO=self.fLO,
            lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
            profile=self.profile,
            atom_mass=self.atom_mass, atom_temperature=self.atom_temperature,
            trap_freq_r=self.trap_freq_r,
        )

    total = n_width * n_win_input
    done = 0
    cancelled = False
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
        print(f"GEMEINSAMER (jointer) Amplituden-Abhaengigkeits-Scan: {n_win_input}x{n_width} "
              f"Punkte, pro Punkt EINE (r_x, r_y)-Optimierung direkt gegen die Kombination aus "
              f"hart+gewichtet (alpha={alpha}, combo_lambda={combo_lambda}), N_x={self.N_x}, "
              f"N_y={self.N_y}, n_jobs={n_jobs_resolved}")
        print("=" * 70)

    if n_jobs_resolved <= 1:
        # ------------------------------------------------------------
        # Sequentiell im Hauptprozess, mit warm_start - identisches Muster
        # wie scan_win_width_amplitude_dependence(), nur mit EINEM
        # kombinierten Ziel statt alpha*uniformity + (1-alpha)*eta.
        # ------------------------------------------------------------
        for i, width_val in enumerate(width_vals):
            if cancelled:
                break
            for j, win_input_val in enumerate(win_input_vals):
                already_done = (np.isfinite(uniformity_grid[i, j])
                                 and np.isfinite(uniformity_weighted_grid[i, j])
                                 and np.isfinite(r_x_grid[i, j]) and np.isfinite(r_y_grid[i, j]))
                if already_done:
                    if warm_start:
                        last_r = [float(r_x_grid[i, j]), float(r_y_grid[i, j])]
                else:
                    try:
                        win_eff = self.win_input_to_win(win_input_val)
                    except ValueError:
                        win_eff = None

                    if win_eff is not None:
                        point_grid = self._build_dynamic_grid(win_eff, width_val)

                        def objective(p, win_eff=win_eff, width_val=width_val, point_grid=point_grid):
                            amps = amps_from_ratios(p[0], p[1], self.N_x, self.N_y)
                            val = self._evaluate(win_eff, width_val, amps=amps, grid=point_grid,
                                                  weighted=True)
                            if val is None:
                                return 1e10
                            U_h, C_h = val['uniformity'], val['eta']
                            U_w, C_w = val.get('uniformity_weighted'), val.get('eta_weighted')
                            if U_w is None or C_w is None or not (np.isfinite(U_w) and np.isfinite(C_w)):
                                return 1e10
                            U_kombi = 0.5 * (U_h + U_w) + combo_lambda * abs(U_h - U_w)
                            C_kombi = 0.5 * (C_h + C_w) + combo_lambda * abs(C_h - C_w)
                            return alpha * U_kombi + (1.0 - alpha) * C_kombi

                        x0 = last_r if warm_start else [float(r0[0]), float(r0[1])]
                        result = minimize(
                            objective, x0=x0, method='Nelder-Mead',
                            bounds=[r_bounds, r_bounds],
                            options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300},
                        )
                        r_opt = result.x
                        amps_opt = amps_from_ratios(r_opt[0], r_opt[1], self.N_x, self.N_y)
                        details = self._evaluate(win_eff, width_val, amps=amps_opt, grid=point_grid,
                                                  weighted=True)
                        U_w_final = details.get('uniformity_weighted') if details is not None else None
                        C_w_final = details.get('eta_weighted') if details is not None else None
                        valid = (details is not None and U_w_final is not None and C_w_final is not None
                                 and np.isfinite(U_w_final) and np.isfinite(C_w_final))

                        if valid:
                            uniformity_grid[i, j] = details['uniformity']
                            crosstalk_grid[i, j] = details['eta']
                            uniformity_weighted_grid[i, j] = U_w_final
                            eta_weighted_grid[i, j] = C_w_final
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
        # Parallel ueber mehrere Prozesse (ProcessPoolExecutor) - jeder
        # Gitterpunkt ist unabhaengig, warm_start entfaellt.
        # ------------------------------------------------------------
        optimizer_kwargs = {k: getattr(self, k) for k in self.DEFAULTS}
        tasks = [
            (i, j, win_input_val, width_val, list(r0), r_bounds, alpha, combo_lambda, optimizer_kwargs)
            for i, width_val in enumerate(width_vals)
            for j, win_input_val in enumerate(win_input_vals)
            if not (np.isfinite(uniformity_grid[i, j]) and np.isfinite(uniformity_weighted_grid[i, j])
                    and np.isfinite(r_x_grid[i, j]) and np.isfinite(r_y_grid[i, j]))
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
                futures = {executor.submit(_amp_dependence_worker_combined_joint, task): task
                           for task in tasks}
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
                    ckpt.maybe_save(_current_results, done=done, total=total)
                    if verbose and total >= 10 and done % max(1, total // 10) == 0:
                        print(f"  ... {done}/{total} Punkte fertig")

    if ckpt.active:
        ckpt.maybe_save(_current_results, done=done, total=total, force=True)

    hard = dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_grid=uniformity_grid, crosstalk_grid=crosstalk_grid,
        r_x_grid=r_x_grid, r_y_grid=r_y_grid, r_bounds=r_bounds,
        N_x=self.N_x, N_y=self.N_y, f1=self.f1, f2=self.f2, fLO=self.fLO,
        lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
        profile=self.profile,
    )
    weighted = dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_weighted_grid=uniformity_weighted_grid, eta_weighted_grid=eta_weighted_grid,
        r_x_grid=r_x_grid, r_y_grid=r_y_grid,
        sigma_atom=self.sigma_atom, atom_mass=self.atom_mass,
        atom_temperature=self.atom_temperature, trap_freq_r=self.trap_freq_r,
    )
    combined = build_combined_amp_scan_results_joint(
        hard, weighted, alpha=alpha, combo_lambda=combo_lambda, combo_percentile=combo_percentile,
    )
    self.results['scan2d_amp_combined'] = combined

    if verbose:
        print("\n" + "=" * 70)
        b, r = combined['best'], combined['region']
        if b['win_input'] is not None:
            print(f"Gemeinsam (joint) optimiert, bester Punkt: win_input={b['win_input']*1e3:.4f} mm, "
                  f"width={b['width']*1e-6:.3f} MHz, r_x={r_x_grid[int(np.argmin(np.abs(width_vals-b['width']))), int(np.argmin(np.abs(win_input_vals-b['win_input'])))]:.4f}, "
                  f"r_y={r_y_grid[int(np.argmin(np.abs(width_vals-b['width']))), int(np.argmin(np.abs(win_input_vals-b['win_input'])))]:.4f} -> "
                  f"Uniformity_hart={b['uniformity_hart']*100:.2f}%, "
                  f"Crosstalk_hart={b['crosstalk_hart']*100:.3f}%, "
                  f"Uniformity_w={b['uniformity_weighted']*100:.2f}%, "
                  f"Crosstalk_w={b['crosstalk_weighted']*100:.3f}%")
        if r['win_input_min'] is not None:
            print(f"Kombinierte Region (beste {combo_percentile:.0f}% des Scores, groesstes "
                  f"eingeschriebenes Rechteck aus {r['n_points_region']}/{r['n_points_total']} "
                  f"Punkten im Akzeptanzbereich): "
                  f"win_input in [{r['win_input_min']*1e3:.4f}, {r['win_input_max']*1e3:.4f}] mm, "
                  f"width in [{r['width_min']*1e-6:.4f}, {r['width_max']*1e-6:.4f}] MHz")
        else:
            print("Kombinierte Region: kein gueltiges Rechteck gefunden (zu wenige valide Punkte).")
        print("=" * 70)

    return combined


def get_scan_amp_results_combined(self):
    """Wie get_scan_amp_results()/get_scan_amp_results_weighted(), aber fuer
    scan_win_width_amplitude_dependence_combined()."""
    if 'scan2d_amp_combined' not in self.results:
        raise RuntimeError("scan_win_width_amplitude_dependence_combined() must be called first.")
    return dict(self.results['scan2d_amp_combined'])


def save_scan_amp_results_combined(self, filepath=None, overwrite=False):
    """Wie save_scan_amp_results()/save_scan_amp_results_weighted(), aber
    fuer get_scan_amp_results_combined(). Default-Dateiname bei
    filepath=None: "scan_amp_data_combined_N{N_x}x{N_y}_{n_win}x{n_width}pts_
    {Airy|Gauss}.pkl" - im SELBEN Results-Ordner wie
    scan_data_combined_...pkl (combined_scan_methods.py), da beide
    Combinated_Optimization/Results als DEFAULT_RESULTS_DIR teilen.

    overwrite (NEU, 2026-08-27, siehe Chat "Amplituden Abhängigkeit" - analoger
    Fix zu save_scan_weighted_results()/save_scan_amp_results_weighted() in
    weighted_amp_scan_methods.py, Nachtrag 23 - dort urspruenglich versaeumt,
    auf Combinated_Optimization zu uebertragen): bei filepath!=None und
    overwrite=True wird eine bereits vorhandene Datei unter GENAU diesem Pfad
    direkt ueberschrieben statt (wie sonst ueber _resolve_pickle_path()) einen
    freien "_2"-Namen zu waehlen. Noetig, weil combined_winwidthampscan_
    startdialog.py denselben Pfad zuvor schon als checkpoint_path an den Scan
    uebergeben hat - dort liegt unter diesem Pfad bereits eine (mit
    '_checkpoint': True markierte) Zwischenstand-Datei, die hier ganz bewusst
    durch den sauberen Endstand ERSETZT werden soll, statt daneben eine
    verwirrende Doppel-Datei zu erzeugen."""
    if filepath is None:
        res = self.results.get('scan2d_amp_combined', {})
        n_win = len(res.get('win_input_vals', []))
        n_width = len(res.get('width_vals', []))
        profile_tag = "Airy" if self.profile == "airy" else "Gauss" if self.profile == "gaussian" else self.profile
        filename = f"scan_amp_data_combined_N{self.N_x}x{self.N_y}_{n_win}x{n_width}pts_{profile_tag}.pkl"
        filepath = _resolve_pickle_path(DEFAULT_RESULTS_DIR, filename)
    else:
        filepath = FilePath(filepath)
        if filepath.exists() and not overwrite:
            filepath = _resolve_pickle_path(filepath.parent, filepath.name)

    with open(filepath, 'wb') as f:
        pickle.dump(self.get_scan_amp_results_combined(), f)
    print(f"Combined amplitude-dependence scan results saved: {filepath}")
    return filepath


MultitoneFlatTopOptimizer.scan_win_width_amplitude_dependence_combined = scan_win_width_amplitude_dependence_combined
MultitoneFlatTopOptimizer.scan_win_width_amplitude_dependence_combined_joint = scan_win_width_amplitude_dependence_combined_joint
MultitoneFlatTopOptimizer.get_scan_amp_results_combined = get_scan_amp_results_combined
MultitoneFlatTopOptimizer.save_scan_amp_results_combined = save_scan_amp_results_combined
