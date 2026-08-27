"""
Combinated_Optimization/combined_scan_methods.py
=================================================

Kombiniert die harten Masken-Metriken (uniformity/crosstalk, aus der
UNVERAENDERTEN Methode `scan_win_width_uniformity()`) und die atom-
gewichteten Metriken (uniformity_weighted/eta_weighted, aus
`scan_win_width_weighted_uniformity()`, monkey-gepatcht via
weighted_amp_scan_methods.py) zu EINER gemeinsamen Bewertung pro
(win_input, width)-Gitterpunkt des Fest-Amplituden-Scans.

WARUM das ueberhaupt zusammenpasst (siehe Chat-Verlauf "Amplituden
Abhaengigkeit"): `MultitoneFlatTopOptimizer._evaluate()` und die harte
`scan_win_width_uniformity()` sind in weighted_multitone_flattop_optimizer.py
UNVERAENDERT (identischer Code) gegenueber der Original-Optimierung in
Hard_Optimization/multitone_flattop_optimizer.py enthalten - es wird also
KEIN zweites Modul/keine zweite Optimizer-Klasse gebraucht. Beide Scans
(`scan_win_width_uniformity()` und `scan_win_width_weighted_uniformity()`)
bauen ihre Achsen IDENTISCH per `np.linspace(range[0], range[1], n)` auf;
werden sie mit denselben `win_input_range`/`width_range`/`n_win_input`/
`n_width`-Parametern auf DERSELBEN Optimizer-Instanz aufgerufen, liegen
beide Ergebnis-Grids exakt auf demselben Gitter - punktweise Kombination
ist damit ohne Interpolation moeglich.

Kombinationsprinzip (siehe Chat, "Mittelwert mit Uneinigkeits-Strafterm"):
1. Jede der vier Rohgroessen (uniformity_grid, crosstalk_grid,
   uniformity_weighted_grid, eta_weighted_grid) wird unabhaengig ueber das
   Scan-Gitter auf [0, 1] normiert (Min-Max, NaN-sicher).
2. Kombinierte Uniformity/Crosstalk:
       X_kombi = 0.5*(X_hart_norm + X_weighted_norm)
                 + combo_lambda * |X_hart_norm - X_weighted_norm|
   Der erste Term ist der Mittelwert, der zweite bestraft genau die Faelle,
   in denen hart und weighted stark auseinanderlaufen (das eigentliche
   Problem: "gutes globales Ergebnis muss nicht lokal gut sein und
   umgekehrt"), statt es im Mittelwert zu verstecken.
3. Gesamt-Score = alpha*Uniformity_kombi + (1-alpha)*Crosstalk_kombi
   (dieselbe alpha-Konvention wie ueberall sonst im Projekt).
4. Region = groesstes achsenparalleles Rechteck, das vollstaendig in der
   Menge der "besten combo_percentile % aller Gitterpunkte" (nach
   Gesamt-Score) liegt - das ist der tatsaechliche, direkt nutzbare
   (waist,width)-Bereich.

Nutzung (siehe combined_winwidthscan_startdialog.py fuer die GUI):

    from combined_scan_methods import MultitoneFlatTopOptimizer, DEFAULT_RESULTS_DIR

    opt = MultitoneFlatTopOptimizer(out_dir=".")
    opt.scan_win_width_combined_uniformity(
        win_input_range=(0.8e-3, 1.7e-3), width_range=(0.2e6, 0.4e6),
        n_win_input=40, n_width=40,
    )
    opt.save_scan_combined_results()   # -> Results/scan_data_combined_....pkl

Zum guenstigen Nach-Tunen von combo_lambda/combo_percentile/alpha OHNE die
(teuren) Scans neu laufen zu lassen: `recombine_from_grids()` unten - nimmt
ein bereits kombiniertes results-dict (z.B. aus einer geladenen pkl-Datei)
und berechnet Uniformity_kombi/Crosstalk_kombi/Score/Region aus den darin
weiterhin vollstaendig enthaltenen Rohgrids neu.
"""

import sys
import pickle
from pathlib import Path as FilePath

import numpy as np

# ----------------------------------------------------------------------
# Weighted_Optimization ist Geschwisterordner von Combinated_Optimization
# (beide direkt unter Amplitudes/) - wird hier auf sys.path gesetzt, damit
# die dortigen, UNVERAENDERTEN Module importiert werden koennen, statt
# ca. 14.000 Zeilen Optimizer-/Scan-Code zu duplizieren. Aenderungen an
# Weighted_Optimization (z.B. neue Optimizer-Parameter) wirken sich damit
# automatisch auch hier aus.
# ----------------------------------------------------------------------
_WEIGHTED_DIR = FilePath(__file__).resolve().parent.parent / "Weighted_Optimization"
if str(_WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(_WEIGHTED_DIR))

from weighted_multitone_flattop_optimizer import (  # noqa: E402
    MultitoneFlatTopOptimizer,
    _resolve_pickle_path,
)
import weighted_amp_scan_methods  # noqa: E402,F401  # Import-Nebeneffekt: patcht
# scan_win_width_weighted_uniformity()/get_scan_weighted_results()/
# save_scan_weighted_results() auf MultitoneFlatTopOptimizer.
import scan_checkpoint  # noqa: E402  # liegt (identische Kopie) in Weighted_Optimization,
# ueber denselben sys.path-Eintrag oben erreichbar.


def _derive_checkpoint_paths(checkpoint_path, tag_hard="hart", tag_weighted="weighted"):
    """Leitet aus EINEM vom Nutzer gewaehlten checkpoint_path ZWEI eigene
    Zwischenspeicher-Pfade ab - je einen fuer den harten und den
    gewichteten Teilscan (siehe scan_win_width_combined_uniformity()/
    scan_win_width_amplitude_dependence_combined() in combined_amp_scan_
    methods.py). Beide Teilscans speichern damit unabhaengig voneinander
    stuendlich ihren eigenen Zwischenstand; beim Fortsetzen wird jeder
    Teilscan automatisch an GENAU der Stelle wieder aufgenommen, an der er
    (unabhaengig vom jeweils anderen Teilscan) stehen geblieben ist - z.B.
    ist der harte Teilscan evtl. schon fertig (wird dann sofort komplett
    geladen) waehrend der gewichtete Teilscan noch weiterlaufen muss.
    checkpoint_path=None ergibt (None, None) - Zwischenspeicherung bleibt
    dann komplett deaktiviert, wie bei den Einzel-Scans."""
    if checkpoint_path is None:
        return None, None
    checkpoint_path = FilePath(checkpoint_path)
    hard_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{tag_hard}_checkpoint{checkpoint_path.suffix}")
    weighted_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_{tag_weighted}_checkpoint{checkpoint_path.suffix}")
    return hard_path, weighted_path


# ======================================================================
# Eigene Default-Ordner - bewusst NICHT die aus weighted_multitone_flattop_
# optimizer.py/weighted_amp_scan_methods.py uebernommen, da jene relativ zu
# IHREM __file__ (also Weighted_Optimization/) aufloesen. Combinated_
# Optimization soll komplett eigenstaendig sein ("alles rein, was ich
# brauche" - siehe Chat), daher eigene Results/Bilder-Ordner direkt hier.
# ======================================================================
def _default_dir(name):
    candidate = FilePath(__file__).resolve().parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = FilePath(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DEFAULT_RESULTS_DIR = _default_dir("Results")
DEFAULT_IMAGES_DIR = _default_dir("Bilder")


# ======================================================================
# Normierung + Kombination (reine Numpy-Nachbearbeitung, keine
# Optimizer-Aufrufe - beliebig oft/billig wiederholbar)
# ======================================================================
def _normalize01(grid):
    """Min-Max-Normierung auf [0, 1], NaN-sicher. Punkte, die in KEINEM der
    vier Grids endlich sind, bleiben NaN; ist ein Grid komplett konstant
    (max==min), wird es auf 0 gesetzt (kein Beitrag zur Uneinigkeit)."""
    grid = np.asarray(grid, dtype=float)
    finite = np.isfinite(grid)
    out = np.full_like(grid, np.nan)
    if not finite.any():
        return out
    lo = np.nanmin(grid)
    hi = np.nanmax(grid)
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-300:
        out[finite] = 0.0
        return out
    out[finite] = (grid[finite] - lo) / (hi - lo)
    return out


def _largest_true_rectangle(mask):
    """Groesstes achsenparalleles, vollstaendig True-Rechteck in einem 2D
    Boolean-Array. Standard-Histogramm-Algorithmus (monotoner Stack),
    O(rows*cols). Gibt (row0, row1, col0, col1) (inklusive Indexgrenzen)
    zurueck, oder None falls mask ueberall False ist."""
    mask = np.asarray(mask, dtype=bool)
    rows, cols = mask.shape
    if not mask.any():
        return None

    height = np.zeros(cols, dtype=int)
    best_area = 0
    best = None

    for r in range(rows):
        height = np.where(mask[r], height + 1, 0)
        stack = []  # Liste von (start_col, h)
        for c in range(cols + 1):
            h = int(height[c]) if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (c - s)
                if area > best_area:
                    best_area = area
                    best = (r - sh + 1, r, s, c - 1)
                start = s
            stack.append((start, h))

    return best


def _combine_grids(U_hard, C_hard, U_w, C_w, win_input_vals, width_vals,
                    alpha, combo_lambda, combo_percentile):
    """Kernrechnung: Normierung -> Uniformity_kombi/Crosstalk_kombi
    (Mittelwert + Uneinigkeits-Strafterm) -> Gesamt-Score -> bester Punkt
    -> Region (Perzentil-Maske + groesstes eingeschriebenes Rechteck).
    Gibt ein dict mit allen abgeleiteten Groessen zurueck (siehe
    build_combined_scan_results() fuer die vollstaendige Zusammenstellung)."""
    U_hard_n = _normalize01(U_hard)
    C_hard_n = _normalize01(C_hard)
    U_w_n = _normalize01(U_w)
    C_w_n = _normalize01(C_w)

    U_kombi = 0.5 * (U_hard_n + U_w_n) + combo_lambda * np.abs(U_hard_n - U_w_n)
    C_kombi = 0.5 * (C_hard_n + C_w_n) + combo_lambda * np.abs(C_hard_n - C_w_n)
    combined_score = alpha * U_kombi + (1.0 - alpha) * C_kombi

    best = dict(win_input=None, width=None, uniformity_hart=None, crosstalk_hart=None,
                uniformity_weighted=None, crosstalk_weighted=None,
                uniformity_kombi=None, crosstalk_kombi=None, combined_score=None)
    if np.any(np.isfinite(combined_score)):
        idx_min = np.unravel_index(np.nanargmin(combined_score), combined_score.shape)
        best.update(
            win_input=win_input_vals[idx_min[1]], width=width_vals[idx_min[0]],
            uniformity_hart=U_hard[idx_min], crosstalk_hart=C_hard[idx_min],
            uniformity_weighted=U_w[idx_min], crosstalk_weighted=C_w[idx_min],
            uniformity_kombi=U_kombi[idx_min], crosstalk_kombi=C_kombi[idx_min],
            combined_score=combined_score[idx_min],
        )

    finite = np.isfinite(combined_score)
    region = dict(win_input_min=None, win_input_max=None, width_min=None, width_max=None,
                  mask=None, threshold=None, n_points_total=int(finite.sum()),
                  n_points_region=0, row_bounds=None, col_bounds=None)
    if finite.any():
        threshold = float(np.nanpercentile(combined_score, combo_percentile))
        mask = finite & (combined_score <= threshold)
        region.update(mask=mask, threshold=threshold, n_points_region=int(mask.sum()))
        rect = _largest_true_rectangle(mask)
        if rect is not None:
            r0, r1, c0, c1 = rect
            region.update(
                win_input_min=float(win_input_vals[c0]), win_input_max=float(win_input_vals[c1]),
                width_min=float(width_vals[r0]), width_max=float(width_vals[r1]),
                row_bounds=(r0, r1), col_bounds=(c0, c1),
            )

    return dict(
        uniformity_hart_norm=U_hard_n, crosstalk_hart_norm=C_hard_n,
        uniformity_weighted_norm=U_w_n, crosstalk_weighted_norm=C_w_n,
        uniformity_kombi=U_kombi, crosstalk_kombi=C_kombi,
        combined_score=combined_score, best=best, region=region,
        alpha=alpha, combo_lambda=combo_lambda, combo_percentile=combo_percentile,
    )


def build_combined_scan_results(hard, weighted, alpha=0.9, combo_lambda=0.75,
                                 combo_percentile=25.0):
    """Baut das vollstaendige kombinierte Ergebnis-dict aus den RAW-Outputs
    von get_scan_results() (hart) und get_scan_weighted_results() (weighted)
    - wird von scan_win_width_combined_uniformity() nach beiden Scans
    aufgerufen. Prueft zuerst, dass beide tatsaechlich auf demselben Gitter
    liegen (siehe Modul-Docstring - sollte bei identischen Scan-Parametern
    immer der Fall sein, wird hier trotzdem defensiv geprueft)."""
    win_input_vals = np.asarray(hard['win_input_vals'])
    width_vals = np.asarray(hard['width_vals'])
    if not (np.array_equal(win_input_vals, weighted['win_input_vals'])
            and np.array_equal(width_vals, weighted['width_vals'])):
        raise ValueError(
            "Harter und gewichteter Scan liegen auf unterschiedlichen "
            "(waist,width)-Gittern - scan_win_width_combined_uniformity() ruft "
            "beide Scans intern mit identischen win_input_range/width_range/"
            "n_win_input/n_width auf derselben Optimizer-Instanz auf, das "
            "sollte nicht passieren. Wurden hard/weighted von aussen manuell "
            "zusammengestellt, muessen sie exakt dasselbe Gitter haben."
        )

    U_hard = np.asarray(hard['uniformity_grid'])
    C_hard = np.asarray(hard['crosstalk_grid'])
    U_w = np.asarray(weighted['uniformity_weighted_grid'])
    C_w = np.asarray(weighted['eta_weighted_grid'])

    combo = _combine_grids(U_hard, C_hard, U_w, C_w, win_input_vals, width_vals,
                            alpha, combo_lambda, combo_percentile)

    return dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_grid=U_hard, crosstalk_grid=C_hard,
        uniformity_weighted_grid=U_w, eta_weighted_grid=C_w,
        amps=hard.get('amps'),
        N_x=hard['N_x'], N_y=hard['N_y'],
        f1=hard['f1'], f2=hard['f2'], fLO=hard['fLO'],
        lambda_opt=hard['lambda_opt'], theta_max=hard['theta_max'], f_band=hard['f_band'],
        profile=hard['profile'],
        sigma_atom=weighted.get('sigma_atom'),
        atom_mass=weighted.get('atom_mass'), atom_temperature=weighted.get('atom_temperature'),
        trap_freq_r=weighted.get('trap_freq_r'),
        atom_offset_x=weighted.get('atom_offset_x'), atom_offset_y=weighted.get('atom_offset_y'),
        pitch=weighted.get('pitch'),
        **combo,
    )


def recombine_from_grids(results, alpha=None, combo_lambda=None, combo_percentile=None):
    """Berechnet Uniformity_kombi/Crosstalk_kombi/Score/beste-Punkt/Region
    aus einem BEREITS kombinierten results-dict (z.B. per Pickle geladen)
    neu - OHNE die Scans zu wiederholen. Ueberschreibt nur die drei
    Kombinations-Parameter, die explizit angegeben werden (sonst bleiben
    die im dict gespeicherten Werte erhalten). Gibt ein NEUES dict zurueck
    (results bleibt unveraendert). Genutzt von fit_combined_region.py, um
    combo_lambda/combo_percentile/alpha guenstig nachzujustieren."""
    alpha = results['alpha'] if alpha is None else alpha
    combo_lambda = results['combo_lambda'] if combo_lambda is None else combo_lambda
    combo_percentile = results['combo_percentile'] if combo_percentile is None else combo_percentile

    combo = _combine_grids(
        results['uniformity_grid'], results['crosstalk_grid'],
        results['uniformity_weighted_grid'], results['eta_weighted_grid'],
        results['win_input_vals'], results['width_vals'],
        alpha, combo_lambda, combo_percentile,
    )
    out = dict(results)
    out.update(combo)
    return out


# ======================================================================
# Monkey-Patch auf MultitoneFlatTopOptimizer - identisches Muster wie
# weighted_amp_scan_methods.py (siehe dortiger Docstring): freistehende
# Funktionen definieren, dann als Methode anhaengen. So bleibt
# weighted_multitone_flattop_optimizer.py unangetastet.
# ======================================================================
def scan_win_width_combined_uniformity(self, win_input_range, width_range,
                                        n_win_input=40, n_width=40,
                                        amps=None, alpha=0.9,
                                        combo_lambda=0.75, combo_percentile=25.0,
                                        verbose=True, progress_callback=None,
                                        checkpoint_path=None,
                                        checkpoint_interval_s=scan_checkpoint.CHECKPOINT_INTERVAL_S):
    """
    Fuehrt NACHEINANDER die unveraenderte harte scan_win_width_uniformity()
    und die atom-gewichtete scan_win_width_weighted_uniformity() mit
    IDENTISCHEN win_input_range/width_range/n_win_input/n_width/amps/alpha
    auf dieser Optimizer-Instanz aus (beide Scans bauen ihr Gitter identisch
    per np.linspace() auf - die Ergebnis-Grids liegen daher exakt
    uebereinander, keine Interpolation noetig) und kombiniert die vier
    Ergebnis-Grids anschliessend zu Uniformity_kombi/Crosstalk_kombi/
    combined_score/region (siehe build_combined_scan_results() und
    Modul-Docstring fuer das Kombinationsprinzip).

    combo_lambda: Gewicht des Uneinigkeits-Strafterms |hart_norm - weighted_norm|
    in der Kombination (0 = reiner Mittelwert, hoehere Werte bestrafen
    Punkte, an denen hart und weighted stark auseinanderlaufen, staerker).
    combo_percentile: welcher Anteil (in %) der Gitterpunkte mit dem besten
    (kleinsten) combined_score als "Region" gilt, aus der das groesste
    eingeschriebene Rechteck extrahiert wird.

    progress_callback: wie bei den Einzel-Scans, wird ueber BEIDE Scans
    hinweg auf den gemeinsamen Fortschritt (0..2*n_win_input*n_width)
    umgerechnet, damit z.B. eine QProgressDialog durchgehend laeuft statt
    zweimal bei 0 neu zu starten.

    checkpoint_path / checkpoint_interval_s: EIN vom Nutzer gewaehlter Pfad
    fuer den GESAMTEN kombinierten Scan - wird intern in ZWEI eigene Pfade
    fuer den harten und den gewichteten Teilscan aufgeteilt (siehe
    _derive_checkpoint_paths(), z.B. "meine_datei_hart_checkpoint.pkl" /
    "meine_datei_weighted_checkpoint.pkl" neben dem gewaehlten Pfad), die
    beide unabhaengig voneinander stuendlich (Default) ihren Zwischenstand
    sichern. Bei einem Neustart wird JEDER Teilscan automatisch an der
    Stelle fortgesetzt, an der er zuletzt stehen geblieben ist (z.B. der
    harte Teilscan schon fertig -> sofort geladen, gewichteter Teilscan
    setzt fort) - siehe scan_win_width_uniformity()/scan_win_width_
    weighted_uniformity() fuer die Fortsetzungslogik selbst.

    Speichert das Ergebnis in self.results['scan2d_combined'] (zusaetzlich
    zu den ohnehin von den Einzel-Scans gesetzten self.results['scan2d']/
    self.results['scan2d_weighted']) und gibt es zurueck.
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
        print("KOMBINIERTER Fest-Amplituden-Scan: 1/2 harte Masken-Metriken "
              "(uniformity/crosstalk) ...")
        print("=" * 70)
    self.scan_win_width_uniformity(
        win_input_range=win_input_range, width_range=width_range,
        n_win_input=n_win_input, n_width=n_width, amps=amps, alpha=alpha,
        verbose=verbose, progress_callback=hard_progress,
        checkpoint_path=ckpt_hard, checkpoint_interval_s=checkpoint_interval_s,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("KOMBINIERTER Fest-Amplituden-Scan: 2/2 atom-gewichtete Metriken "
              "(uniformity_w/crosstalk_w) ...")
        print("=" * 70)
    self.scan_win_width_weighted_uniformity(
        win_input_range=win_input_range, width_range=width_range,
        n_win_input=n_win_input, n_width=n_width, amps=amps, alpha=alpha,
        verbose=verbose, progress_callback=weighted_progress,
        checkpoint_path=ckpt_weighted, checkpoint_interval_s=checkpoint_interval_s,
    )

    hard = self.get_scan_results()
    weighted = self.get_scan_weighted_results()
    combined = build_combined_scan_results(
        hard, weighted, alpha=alpha, combo_lambda=combo_lambda, combo_percentile=combo_percentile,
    )
    self.results['scan2d_combined'] = combined

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


def get_scan_combined_results(self):
    """Wie get_scan_results()/get_scan_weighted_results(), aber fuer
    scan_win_width_combined_uniformity()."""
    if 'scan2d_combined' not in self.results:
        raise RuntimeError("scan_win_width_combined_uniformity() must be called first.")
    return dict(self.results['scan2d_combined'])


def save_scan_combined_results(self, filepath=None):
    """Wie save_scan_results()/save_scan_weighted_results(), aber fuer
    get_scan_combined_results(). Default-Dateiname bei filepath=None:
    "scan_data_combined_N{N_x}x{N_y}_{n_win}x{n_width}pts_{Airy|Gauss}.pkl"
    in DIESEM (Combinated_Optimization/Results, s.o.) Ordner - bewusst NICHT
    derselbe Ordner wie bei den Einzel-Scans, damit Combinated_Optimization
    eigenstaendig bleibt."""
    if filepath is None:
        res = self.results.get('scan2d_combined', {})
        n_win = len(res.get('win_input_vals', []))
        n_width = len(res.get('width_vals', []))
        profile_tag = "Airy" if self.profile == "airy" else "Gauss" if self.profile == "gaussian" else self.profile
        filename = f"scan_data_combined_N{self.N_x}x{self.N_y}_{n_win}x{n_width}pts_{profile_tag}.pkl"
        filepath = _resolve_pickle_path(DEFAULT_RESULTS_DIR, filename)
    else:
        filepath = FilePath(filepath)
        if filepath.exists():
            filepath = _resolve_pickle_path(filepath.parent, filepath.name)

    with open(filepath, 'wb') as f:
        pickle.dump(self.get_scan_combined_results(), f)
    print(f"Combined scan results saved: {filepath}")
    return filepath


MultitoneFlatTopOptimizer.scan_win_width_combined_uniformity = scan_win_width_combined_uniformity
MultitoneFlatTopOptimizer.get_scan_combined_results = get_scan_combined_results
MultitoneFlatTopOptimizer.save_scan_combined_results = save_scan_combined_results


def load_combined_scan_results(filepath):
    """Generischer Pickle-Loader (siehe load_amp_scan_results() im Original)
    - sucht bei einem reinen Dateinamen zuerst im aktuellen Arbeitsverzeichnis,
    dann in DEFAULT_RESULTS_DIR (Combinated_Optimization/Results)."""
    path = FilePath(filepath)
    if not path.is_absolute() and not path.exists() and len(path.parts) == 1:
        candidate = DEFAULT_RESULTS_DIR / path.name
        if candidate.exists():
            path = candidate
    with open(path, 'rb') as f:
        return pickle.load(f)
