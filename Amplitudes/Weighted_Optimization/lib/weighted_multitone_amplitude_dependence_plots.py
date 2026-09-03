"""
Weighted_Optimization/weighted_multitone_amplitude_dependence_plots.py
=========================================================================

Port von `multitone_amplitude_dependence_plots.py` (Original-Ordner) für den
GEWICHTETEN Optimizer (`weighted_multitone_flattop_optimizer.py` +
`weighted_amp_scan_methods.py`, beide in diesem Ordner).

WAS "gewichtet" hier bedeutet: der Optimizer berechnet zusätzlich zu den
harten Masken-Metriken (uniformity, crosstalk/eta) eine ATOM-THERMISCH
gewichtete Variante (uniformity_weighted, eta_weighted - siehe
_evaluate_weighted_only()/weighted_uniformity()/weighted_crosstalk() in
weighted_multitone_flattop_optimizer.py), die berücksichtigt, dass ein Atom
in der Falle keine Punktmasse ist, sondern eine thermische Ortsverteilung
mit Breite sigma_atom hat. weighted_amp_scan_methods.py (in diesem Ordner)
ergänzt dafür ZWEI neue Scan-Methoden auf MultitoneFlatTopOptimizer:

  - scan_win_width_weighted_uniformity()             (Fest-Amplitude-Scan,
    Analogon zu scan_win_width_uniformity())
  - scan_win_width_amplitude_dependence_weighted()    (r_x/r_y werden pro
    Punkt UNTER dem gewichteten Ziel optimiert, Analogon zu
    scan_win_width_amplitude_dependence())

Genau wie im Original ist DIESES Modul ein reines Plotting-Modul OHNE
Abhängigkeit vom Optimizer (oder scipy) - es lädt nur die von
save_scan_weighted_results()/save_scan_amp_results_weighted() gepickelten
dicts. Kleine Hilfsfunktionen (win_input_to_win, amps_from_ratio, ...)
sind bewusst noch einmal dupliziert statt importiert (wie im Original),
damit dieses Modul für sich allein funktioniert.

Was ist neu/anders gegenüber dem Original:

1. AmplitudeScanPlotter (für scan_amp_data_weighted_...pkl, d.h.
   get_scan_amp_results_weighted()) akzeptiert jetzt SOWOHL Dictionaries
   mit den harten Masken-Grids (uniformity_grid/crosstalk_grid) ALS AUCH
   mit den gewichteten Grids (uniformity_weighted_grid/eta_weighted_grid)
   ALS AUCH beide zusammen - r_x_grid/r_y_grid werden weiterhin immer
   erwartet (die Amplituden-Abhängigkeit ist unabhängig davon, welche
   Metrik optimiert wurde, immer der Kern der Fragestellung). Neue Methoden
   plot_scan2d_uniformity_weighted()/plot_scan2d_crosstalk_weighted() sowie
   eine dynamisch mitwachsende plot_scan2d_combined()-Ansicht (4 Panels bei
   nur harten ODER nur gewichteten Metriken, 6 Panels wenn beide vorhanden
   sind).

2. NEUE Klasse WeightedFixedScanPlotter für scan_data_weighted_...pkl (d.h.
   get_scan_weighted_results() aus dem Fest-Amplitude-Scan) - Analogon zu
   dem (hier nicht vorliegenden) ScanPlotter aus multitone_flattop_scan_
   plots.py, nur für die gewichteten Metriken statt uniformity/crosstalk.

3. plot_dependence_cuts() ist UNVERÄNDERT vom Original übernommen: r_x/r_y
   heißen in beiden Scan-Varianten identisch (r_x_grid/r_y_grid), die
   Methode braucht daher keine Anpassung, egal ob die Grids unter dem
   harten oder dem gewichteten Ziel gefunden wurden.

Typische Verwendung:

    from weighted_multitone_amplitude_dependence_plots import (
        load_amp_scan_results, AmplitudeScanPlotter, WeightedFixedScanPlotter,
        DEFAULT_RESULTS_DIR,
    )

    # r_x/r_y-Abhängigkeit + gewichtete Uniformity/Crosstalk am Optimum:
    results = load_amp_scan_results(DEFAULT_RESULTS_DIR / "scan_amp_data_weighted_N3x4_15x15pts_Airy.pkl")
    plotter = AmplitudeScanPlotter(results)
    plotter.plot_scan2d_combined(show=True, save=True)       # 4-6 Heatmaps, je nachdem was vorhanden ist
    plotter.plot_dependence_cuts(show=True, save=True)        # r_x/r_y als Kurven (unverändert vom Original)

    # Waist-Width-Kopplung aus dem Fest-Amplitude-Scan (gewichtet):
    results_fixed = load_amp_scan_results(DEFAULT_RESULTS_DIR / "scan_data_weighted_N3x4_31x31pts_Airy.pkl")
    fixed_plotter = WeightedFixedScanPlotter(results_fixed)
    fixed_plotter.plot_scan2d_weighted_combined(show=True, save=True)
"""

from datetime import date
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ======================================================================
# Default-Ordner (siehe Modul-Docstring von weighted_multitone_flattop_
# optimizer.py: _default_dir() dort löst - da __file__ jetzt in
# Weighted_Optimization/ liegt - automatisch auf Weighted_Optimization/
# Results bzw. Weighted_Optimization/Bilder auf; hier identische Logik,
# damit dieses Plot-Modul in denselben Ordnern sucht/speichert.)
# ======================================================================
def _default_dir(name):
    candidate = Path(__file__).resolve().parent.parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = Path(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DEFAULT_RESULTS_DIR = _default_dir("Results")
# Bilder werden tageweise abgelegt (Bilder/JJJJ-MM-TT). Der Dateiname
# traegt weiterhin alles, was die Datei identifiziert - der Ordner sagt
# nur, wann sie entstanden ist.
DEFAULT_IMAGES_DIR = _default_dir(f"Bilder/{date.today().isoformat()}")


# ======================================================================
# Laden der gepickelten Rohdaten (format-agnostisch - funktioniert für
# scan_data_weighted_...pkl UND scan_amp_data_weighted_...pkl UND, da nur
# ein generisches dict gepickelt/entpickelt wird, auch für die
# UNVERÄNDERTEN scan_data_...pkl/scan_amp_data_...pkl-Dateien des
# Originals.)
# ======================================================================
def load_amp_scan_results(filepath):
    """Lädt ein von save_scan_weighted_results()/save_scan_amp_results_weighted()
    (oder den unveränderten Original-Varianten) erzeugtes Pickle und gibt das
    enthaltene dict zurück.

    filepath: vollständiger Pfad, ODER nur ein Dateiname - in diesem Fall
    wird zuerst im aktuellen Arbeitsverzeichnis und danach in
    DEFAULT_RESULTS_DIR gesucht."""
    path = Path(filepath)
    if not path.is_absolute() and not path.exists() and len(path.parts) == 1:
        candidate = DEFAULT_RESULTS_DIR / path.name
        if candidate.exists():
            path = candidate
    with open(path, 'rb') as f:
        return pickle.load(f)


# ======================================================================
# Leichte, freistehende Physik-Hilfsfunktionen (Duplikate - siehe
# Modul-Docstring: dieses Modul bleibt bewusst ohne Abhängigkeit vom
# Optimizer, identisch zum Original)
# ======================================================================
def radius_from_angle(theta, f1, f2, fLO):
    return (f1 * fLO / f2) * np.tan(theta)


def win_input_to_win(win_input, f1, f2, lambda_opt, fLO):
    """Siehe MultitoneFlatTopOptimizer.win_input_to_win() für die Herleitung
    (3-Linsen-Kaskade f1 -> f2 Teleskop + fLO Fokussierlinse)."""
    if win_input <= 0:
        raise ValueError("win_input muss > 0 sein.")
    return (f1 / f2) * (lambda_opt * fLO) / (np.pi * win_input)


def width_to_um(width_hz, f1, f2, fLO, theta_max, f_band):
    """Siehe MultitoneFlatTopOptimizer.width_to_um()."""
    theta_width = theta_max * width_hz / f_band
    return radius_from_angle(theta_width, f1, f2, fLO) * 1e6


def amps_from_ratio(r, N):
    """Duplikat von weighted_multitone_flattop_optimizer.amps_from_ratio() - siehe dort."""
    amp = np.ones(N, dtype=float)
    if N >= 2:
        amp[0] = r
        amp[-1] = r
    elif N == 1:
        amp[0] = r
    return amp


# ======================================================================
# Ausreißer-Erkennung/-Bereinigung VOR dem Plotten (unverändert vom
# Original - r_x_grid/r_y_grid sind in BEIDEN Amplituden-Scan-Varianten
# [hart/gewichtet] identisch aufgebaut, siehe Modul-Docstring)
# ======================================================================
def summarize_amp_bounds(results, r_bounds=None, atol=1e-9, verbose=True):
    """Siehe multitone_amplitude_dependence_plots.summarize_amp_bounds() (Original)
    für die ausführliche Begründung - unverändert übernommen."""
    r_bounds = r_bounds if r_bounds is not None else results.get('r_bounds', (0.0, 2.0))
    lo, hi = r_bounds
    counts = {}
    for name, G in (("r_x", results['r_x_grid']), ("r_y", results['r_y_grid'])):
        valid = np.isfinite(G)
        counts[name] = dict(
            lower=int(np.sum(valid & np.isclose(G, lo, atol=atol))),
            upper=int(np.sum(valid & np.isclose(G, hi, atol=atol))),
            total=int(valid.sum()),
        )
    if verbose:
        print(f"Schranken-Analyse (r_bounds=({lo}, {hi})):")
        for name, c in counts.items():
            print(f"  {name}: {c['lower']}/{c['total']} Punkt(e) an unterer Schranke ({lo}), "
                  f"{c['upper']}/{c['total']} an oberer Schranke ({hi}).")
        print("  -> viele/zusammenhängende Treffer an einer Schranke deuten eher auf eine "
              "ECHTE Randsättigung hin (NICHT automatisch bereinigen); einzelne/verstreute "
              "Treffer eher auf einen Ausreißer.")
    return counts


def detect_amp_outliers(results, r_bounds=None, atol=1e-9, bounds=("lower", "upper")):
    """Siehe multitone_amplitude_dependence_plots.detect_amp_outliers() (Original) - unverändert."""
    r_bounds = r_bounds if r_bounds is not None else results.get('r_bounds', (0.0, 2.0))
    lo, hi = r_bounds
    targets = []
    if "lower" in bounds:
        targets.append(lo)
    if "upper" in bounds:
        targets.append(hi)
    RX = results['r_x_grid']
    RY = results['r_y_grid']
    mask = np.zeros(RX.shape, dtype=bool)
    for G in (RX, RY):
        valid = np.isfinite(G)
        for target in targets:
            mask |= valid & np.isclose(G, target, atol=atol)
    return mask


def detect_amp_discontinuities(results, z_thresh=3.5, min_neighbors=2):
    """
    Findet Gitterpunkte, an denen (r_x, r_y) sprunghaft von ihren direkten
    Nachbarn abweicht - im Unterschied zu detect_amp_outliers() (das NUR
    exakte r_bounds-Treffer wie r=0/r=2 findet) erkennt diese Funktion auch
    Sprünge INNERHALB des erlaubten Bereichs, wie sie z.B. entstehen, wenn
    die sequentielle Warm-Start-Nelder-Mead-Optimierung von
    scan_win_width_amplitude_dependence_weighted() an einer Stelle in ein
    lokales statt das globale Optimum läuft (siehe Diagnose vom 2026-08-21,
    Projekt-Status-Doc). WICHTIG: ein erkannter Sprung ist noch KEIN Beweis
    für ein Artefakt - an vielen Stellen ist er die reale Signatur zweier
    fast gleich guter, benachbarter lokaler Optima der gewichteten Metrik.
    Diese Funktion liefert nur die KANDIDATEN; ob ein Kandidat tatsächlich
    ein Artefakt ist, klärt erst refine_scan_amp_results_weighted() (in
    weighted_amp_scan_methods.py) durch eine echte Mehrfachstart-Nachrechnung.

    Methode: für jeden Gitterpunkt wird der euklidische Abstand von
    (r_x, r_y) zum Median seiner bis zu 4 direkten Gitternachbarn
    (oben/unten/links/rechts, NaN-Nachbarn werden ignoriert) berechnet
    ("jump"). Diese jump-Werte werden robust standardisiert (Median + MAD,
    MAD*1.4826 als Sigma-Äquivalent, über alle bewertbaren Punkte im
    Gitter) und Punkte mit jump_z > z_thresh markiert. Punkte mit weniger
    als `min_neighbors` gültigen Nachbarn (z.B. isolierte Ecken bei sehr
    kleinen Gittern) werden nicht bewertet.

    Gibt (mask, jump_z) zurück: mask ist eine boolsche (n_width,
    n_win_input)-Maske, jump_z das zugrunde liegende standardisierte
    Sprungmaß (NaN wo nicht bewertbar) - nützlich, um z_thresh zu tunen
    oder die "Auffälligkeit" eines Punktes zu quantifizieren.
    """
    r_x = np.asarray(results['r_x_grid'], dtype=float)
    r_y = np.asarray(results['r_y_grid'], dtype=float)
    n_width, n_win = r_x.shape
    jump = np.full((n_width, n_win), np.nan)

    for i in range(n_width):
        for j in range(n_win):
            if not (np.isfinite(r_x[i, j]) and np.isfinite(r_y[i, j])):
                continue
            neigh_rx, neigh_ry = [], []
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n_width and 0 <= jj < n_win \
                        and np.isfinite(r_x[ii, jj]) and np.isfinite(r_y[ii, jj]):
                    neigh_rx.append(r_x[ii, jj])
                    neigh_ry.append(r_y[ii, jj])
            if len(neigh_rx) < min_neighbors:
                continue
            jump[i, j] = np.hypot(r_x[i, j] - np.median(neigh_rx), r_y[i, j] - np.median(neigh_ry))

    valid = jump[np.isfinite(jump)]
    mask = np.zeros((n_width, n_win), dtype=bool)
    if valid.size < 4:
        return mask, jump

    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    scale = max(mad * 1.4826, 1e-9)
    jump_z = (jump - med) / scale
    mask = np.isfinite(jump_z) & (jump_z > z_thresh)
    return mask, jump_z


def _fill_from_neighbors(grid, bad_mask):
    """Siehe multitone_amplitude_dependence_plots._fill_from_neighbors() (Original) - unverändert."""
    out = grid.copy()
    ny, nx = grid.shape
    bad = bad_mask | ~np.isfinite(grid)
    for i in range(ny):
        for j in range(nx):
            if not bad[i, j]:
                continue
            neighbor_vals = [
                grid[ii, jj]
                for ii, jj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                if 0 <= ii < ny and 0 <= jj < nx and not bad[ii, jj]
            ]
            if neighbor_vals:
                out[i, j] = float(np.median(neighbor_vals))
    return out


# Grid-Keys, die zusätzlich zu r_x_grid/r_y_grid bei einer Bereinigung
# konsistent mitgezogen werden sollen, falls vorhanden (harte Masken-Grids
# UND/ODER gewichtete Grids - je nachdem, was im geladenen dict steckt).
_CLEANABLE_METRIC_KEYS = ("uniformity_grid", "crosstalk_grid",
                           "uniformity_weighted_grid", "eta_weighted_grid")


def clean_amp_scan_results(results, mask=None, strategy="interpolate", r_bounds=None,
                            atol=1e-9, bounds=("lower", "upper"), verbose=True):
    """Wie multitone_amplitude_dependence_plots.clean_amp_scan_results() (Original), mit
    EINER Anpassung: welche Metrik-Grids neben r_x_grid/r_y_grid mitbereinigt werden,
    wird jetzt dynamisch aus den tatsächlich im dict vorhandenen Keys bestimmt
    (_CLEANABLE_METRIC_KEYS), statt hart auf uniformity_grid/crosstalk_grid
    festgelegt zu sein - so funktioniert die Funktion unverändert für Scans mit
    harten Metriken, gewichteten Metriken, oder beiden zusammen."""
    if mask is None:
        mask = detect_amp_outliers(results, r_bounds=r_bounds, atol=atol, bounds=bounds)

    n_found = int(mask.sum())
    if verbose:
        print(f"clean_amp_scan_results: {n_found} Ausreißer-Punkt(e) gefunden "
              f"(r_x/r_y an r_bounds-Schranke), strategy='{strategy}'.")
    if n_found == 0:
        return dict(results)

    cleaned = dict(results)  # shallow copy - Original bleibt unangetastet
    keys = ["r_x_grid", "r_y_grid"] + [k for k in _CLEANABLE_METRIC_KEYS if k in results]

    if strategy == "nan":
        for key in keys:
            G = results[key].copy()
            G[mask] = np.nan
            cleaned[key] = G

    elif strategy == "interpolate":
        for key in keys:
            cleaned[key] = _fill_from_neighbors(results[key], mask)

    elif strategy in ("drop_columns", "drop_rows"):
        if strategy == "drop_columns":
            bad_axis = np.any(mask, axis=0)
            keep = ~bad_axis
            if verbose:
                print(f"  -> entferne {int(bad_axis.sum())} von {len(keep)} win_input-Spalte(n).")
            cleaned['win_input_vals'] = results['win_input_vals'][keep]
            for key in keys:
                cleaned[key] = results[key][:, keep]
        else:
            bad_axis = np.any(mask, axis=1)
            keep = ~bad_axis
            if verbose:
                print(f"  -> entferne {int(bad_axis.sum())} von {len(keep)} width-Zeile(n).")
            cleaned['width_vals'] = results['width_vals'][keep]
            for key in keys:
                cleaned[key] = results[key][keep, :]
    else:
        raise ValueError(
            f"strategy muss 'interpolate', 'drop_columns', 'drop_rows' oder 'nan' sein, nicht {strategy!r}."
        )

    return cleaned


# ======================================================================
# Kollisionsschutz beim Speichern (identisch zum Original)
# ======================================================================
def resolve_save_path(out_dir, filename, confirm_overwrite=None):
    path = Path(out_dir) / filename
    if not path.exists():
        return path

    if confirm_overwrite is None:
        def confirm_overwrite(existing_path):
            answer = input(
                f"'{existing_path.name}' existiert bereits. Überschreiben? [y/N]: "
            ).strip().lower()
            return answer in ("y", "yes", "j", "ja")

    if confirm_overwrite(path):
        return path

    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            print(f"Bestehende Datei bleibt erhalten, speichere stattdessen als: {candidate.name}")
            return candidate
        counter += 1


# ======================================================================
# Plotter für den Amplituden-Abhängigkeits-Scan (r_x/r_y-Optimum, +
# optional harte und/oder gewichtete Uniformity/Crosstalk am Optimum)
# ======================================================================
class AmplitudeScanPlotter:
    """
    Wie multitone_amplitude_dependence_plots.AmplitudeScanPlotter (Original),
    erweitert um die gewichteten Metriken (uniformity_weighted_grid/
    eta_weighted_grid) - siehe Modul-Docstring oben für den Überblick.

    Erwartet IMMER: r_x_grid, r_y_grid, win_input_vals, width_vals (die
    Amplituden-Abhängigkeit ist der Kern jedes Amplituden-Scans, unabhängig
    davon, welches Ziel pro Punkt optimiert wurde). Von den vier
    Metrik-Grids (uniformity_grid, crosstalk_grid, uniformity_weighted_grid,
    eta_weighted_grid) muss mindestens EIN Paar vorhanden sein - z.B. liefert
    get_scan_amp_results() (Original, hartes Ziel) nur das erste Paar,
    get_scan_amp_results_weighted() (gewichtetes Ziel) nur das zweite Paar;
    wer beide Scans zusammenführt (z.B. per dict.update()), bekommt beide
    Paare gleichzeitig plotbar.
    """

    SCAN2D_RC = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
    SCAN2D_SAVE_DPI = 300

    # Alle vier Metrik-Grids sind dimensionslose Verhältnisse (0-1) und
    # werden auf der Colorbar als Prozent dargestellt (Rohdaten in results
    # bleiben unverändert als Fraktion). r_x_grid/r_y_grid bleiben unskaliert.
    PERCENT_METRIC_KEYS = ("uniformity_grid", "crosstalk_grid",
                            "uniformity_weighted_grid", "eta_weighted_grid")

    BEST_POINT_STYLE = dict(
        marker="*", markersize=20, markeredgecolor="white",
        markeredgewidth=1.3, color="red", zorder=8,
    )

    def __init__(self, results, out_dir=None, confirm_overwrite=None):
        missing = [k for k in ("r_x_grid", "r_y_grid", "win_input_vals", "width_vals")
                   if k not in results]
        if missing:
            raise ValueError(
                "results fehlen die Schlüssel " + ", ".join(missing) + " - AmplitudeScanPlotter "
                "braucht mindestens das Ergebnis eines Amplituden-Scans (get_scan_amp_results() "
                "oder get_scan_amp_results_weighted())."
            )
        has_hard = "uniformity_grid" in results and "crosstalk_grid" in results
        has_weighted = "uniformity_weighted_grid" in results and "eta_weighted_grid" in results
        if not has_hard and not has_weighted:
            raise ValueError(
                "results hat weder uniformity_grid/crosstalk_grid (hartes Ziel) noch "
                "uniformity_weighted_grid/eta_weighted_grid (gewichtetes Ziel) - nichts zum "
                "Plotten außer r_x/r_y. Kam das dict aus scan_win_width_amplitude_dependence() "
                "bzw. ..._weighted()?"
            )
        self.has_hard = has_hard
        self.has_weighted = has_weighted
        self.results = results
        self.out_dir = Path(out_dir) if out_dir is not None else DEFAULT_IMAGES_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_overwrite = confirm_overwrite

    # ------------------------------------------------------------------
    def _win_input_to_win(self, win_input):
        r = self.results
        return win_input_to_win(win_input, r['f1'], r['f2'], r['lambda_opt'], r['fLO'])

    def _profile_tag(self):
        profile = self.results.get('profile')
        if profile == 'airy':
            return 'Airy'
        if profile == 'gaussian':
            return 'Gauss'
        return 'ProfilUnbekannt'

    def _filetag(self):
        r = self.results
        n_win = len(r.get('win_input_vals', []))
        n_width = len(r.get('width_vals', []))
        tag = "Weighted" if (self.has_weighted and not self.has_hard) else \
              "HardAndWeighted" if (self.has_weighted and self.has_hard) else ""
        suffix = f"_{tag}" if tag else ""
        return f"N{r['N_x']}x{r['N_y']}_{n_win}x{n_width}pts_{self._profile_tag()}{suffix}"

    def _finish_figure(self, fig, filename, show, save, dpi=150):
        if save:
            out_file = resolve_save_path(self.out_dir, filename, confirm_overwrite=self.confirm_overwrite)
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {out_file}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _point_info_lines(self, win_input_val, width_val, win_eff, r_x, r_y,
                           uniformity=None, crosstalk=None,
                           uniformity_weighted=None, eta_weighted=None):
        r = self.results
        amp_x = amps_from_ratio(r_x, r['N_x'])
        amp_y = amps_from_ratio(r_y, r['N_y'])
        lines = [
            rf"$\omega_{{\mathrm{{in}}}}$ (before lenses):  {win_input_val*1e3:.4f} mm",
            rf"$\omega'$ (at focus):            {win_eff*1e6:.4f} µm" if win_eff is not None
            else r"$\omega'$ (at focus):            invalid ($\omega_{\mathrm{in}} \leq 0$)",
            rf"width:                       {width_val*1e-6:.4f} MHz",
        ]
        if uniformity is not None and np.isfinite(uniformity):
            lines.append(rf"Uniformity ($\sigma/\mu$, hard mask):  {uniformity*100:.3f} %")
            lines.append(rf"Crosstalk ($\eta$, hard mask):  {crosstalk*100:.3f} %")
        if uniformity_weighted is not None and np.isfinite(uniformity_weighted):
            lines.append(rf"Uniformity$_w$ ($\sigma_w/\mu_w$, atom-weighted):  {uniformity_weighted*100:.3f} %")
            lines.append(rf"Crosstalk$_w$ ($\eta_w$, atom-weighted):  {eta_weighted*100:.3f} %")
        if np.isfinite(r_x):
            lines.append(rf"$r_x$ (outer/inner, $N_x$={r['N_x']}):  {r_x:.4f}")
            lines.append(rf"$r_y$ (outer/inner, $N_y$={r['N_y']}):  {r_y:.4f}")
            lines.append(rf"$a_x$:  {np.array2string(amp_x, precision=3)}")
            lines.append(rf"$a_y$:  {np.array2string(amp_y, precision=3)}")
        else:
            lines.append("Invalid point (no optimum found).")
        return lines

    # ------------------------------------------------------------------
    def _win_axis_values(self, win_input_vals, win_axis):
        if win_axis == "before_lens":
            return win_input_vals * 1e3, r"Input waist $\omega_{\mathrm{in}}$ (before lenses, mm)", False
        elif win_axis == "after_lens":
            x = np.array([self._win_input_to_win(w) for w in win_input_vals]) * 1e6
            if len(x) > 1 and x[0] > x[-1]:
                return x[::-1], r"Waist at focus $\omega'$ (after lenses, µm)", True
            return x, r"Waist at focus $\omega'$ (after lenses, µm)", False
        else:
            raise ValueError(f"win_axis muss 'before_lens' oder 'after_lens' sein, nicht {win_axis!r}.")

    def _win_axis_single_value(self, win_input_val, win_axis):
        return self._win_axis_values(np.array([win_input_val]), win_axis)[0][0]

    def _resolve_mark_point(self, win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        RX = r['r_x_grid']

        manual = win_input_fixed is not None or width_fixed is not None
        has_valid = np.any(np.isfinite(RX))
        if not manual and not has_valid:
            return None

        # "Bestes" Kombinationsziel für den Default-Markierungspunkt: nimmt
        # bevorzugt die harten Masken-Metriken (falls vorhanden, für
        # Rückwärtskompatibilität zum Original), sonst die gewichteten.
        idx_min = (0, 0)
        if has_valid:
            if self.has_hard:
                U, C = r['uniformity_grid'], r['crosstalk_grid']
                alpha = r.get('alpha', 1.0)
                combined = alpha * U + (1 - alpha) * C
            else:
                U, C = r['uniformity_weighted_grid'], r['eta_weighted_grid']
                alpha = r.get('alpha', 1.0)
                combined = alpha * U + (1 - alpha) * C
            idx_min = np.unravel_index(np.nanargmin(combined), combined.shape)

        i = idx_min[0] if width_fixed is None else int(np.argmin(np.abs(width_vals - width_fixed)))
        if win_input_fixed is None:
            j = idx_min[1]
        elif win_input_fixed_axis == "before_lens":
            j = int(np.argmin(np.abs(win_input_vals - win_input_fixed)))
        elif win_input_fixed_axis == "after_lens":
            win_eff_vals = np.array([self._win_input_to_win(w) for w in win_input_vals])
            j = int(np.argmin(np.abs(win_eff_vals - win_input_fixed)))
        else:
            raise ValueError(
                f"win_input_fixed_axis muss 'before_lens' oder 'after_lens' sein, "
                f"nicht {win_input_fixed_axis!r}."
            )
        label = "selected point (cut origin)" if manual else "best point (global optimum)"
        return i, j, label

    # ------------------------------------------------------------------
    # Einzelne Heatmaps
    # ------------------------------------------------------------------
    def plot_scan2d_rx(self, show=True, save=True, cmap="viridis", vmin=None, vmax=None,
                        win_axis="before_lens", mark_best_point=True,
                        win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        self._plot_scan2d_metric(
            grid_key="r_x_grid", colorbar_label=r"$r_x$ (outer/inner)",
            title_metric=r"Optimal $r_x$", filename_suffix="rx",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def plot_scan2d_ry(self, show=True, save=True, cmap="viridis", vmin=None, vmax=None,
                        win_axis="before_lens", mark_best_point=True,
                        win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        self._plot_scan2d_metric(
            grid_key="r_y_grid", colorbar_label=r"$r_y$ (outer/inner)",
            title_metric=r"Optimal $r_y$", filename_suffix="ry",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def _require(self, key, what):
        if key not in self.results:
            raise ValueError(
                f"results hat kein '{key}' - {what} nicht verfügbar. Kam das dict aus dem "
                f"richtigen Scan (hartes vs. gewichtetes Ziel)?"
            )

    def plot_scan2d_uniformity(self, show=True, save=True, cmap="viridis_r", vmin=None, vmax=None,
                                win_axis="before_lens", mark_best_point=True,
                                win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """Heatmap der HARTEN Uniformity (uniformity_grid) am (r_x,r_y)-Optimum.
        Braucht uniformity_grid im results-dict (Original-Ziel, siehe
        scan_win_width_amplitude_dependence()) - für die atom-gewichtete
        Variante siehe plot_scan2d_uniformity_weighted()."""
        self._require("uniformity_grid", "harte Uniformity-Heatmap")
        alpha = self.results.get('alpha')
        self._plot_scan2d_metric(
            grid_key="uniformity_grid", colorbar_label="Uniformity (σ/μ) at r-optimum (%)",
            title_metric=f"Uniformity (r-optimum, α={alpha:.2f})" if alpha is not None else "Uniformity (r-optimum)",
            filename_suffix="Uniformity",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def plot_scan2d_crosstalk(self, show=True, save=True, cmap="Oranges", vmin=None, vmax=None,
                               win_axis="before_lens", mark_best_point=True,
                               win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """Heatmap des HARTEN Crosstalk (crosstalk_grid) am (r_x,r_y)-Optimum -
        für die atom-gewichtete Variante siehe plot_scan2d_crosstalk_weighted()."""
        self._require("crosstalk_grid", "harte Crosstalk-Heatmap")
        alpha = self.results.get('alpha')
        self._plot_scan2d_metric(
            grid_key="crosstalk_grid", colorbar_label="Crosstalk (η) at r-optimum (%)",
            title_metric=f"Crosstalk (r-optimum, α={alpha:.2f})" if alpha is not None else "Crosstalk (r-optimum)",
            filename_suffix="Crosstalk",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def plot_scan2d_uniformity_weighted(self, show=True, save=True, cmap="viridis_r", vmin=None, vmax=None,
                                         win_axis="before_lens", mark_best_point=True,
                                         win_input_fixed=None, width_fixed=None,
                                         win_input_fixed_axis="before_lens"):
        """NEU (gegenüber dem Original): Heatmap der ATOM-GEWICHTETEN
        Uniformity (uniformity_weighted_grid, sigma_w/mu_w) am (r_x,r_y)-
        Optimum unter dem gewichteten Ziel - siehe
        scan_win_width_amplitude_dependence_weighted() in
        weighted_amp_scan_methods.py. Braucht uniformity_weighted_grid im
        results-dict."""
        self._require("uniformity_weighted_grid", "gewichtete Uniformity-Heatmap")
        alpha = self.results.get('alpha')
        self._plot_scan2d_metric(
            grid_key="uniformity_weighted_grid",
            colorbar_label=r"Uniformity$_w$ ($\sigma_w/\mu_w$) at r-optimum (%)",
            title_metric=(rf"Uniformity$_w$ (r-optimum, atom-weighted, $\alpha$={alpha:.2f})"
                          if alpha is not None else r"Uniformity$_w$ (r-optimum, atom-weighted)"),
            filename_suffix="UniformityWeighted",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def plot_scan2d_crosstalk_weighted(self, show=True, save=True, cmap="Oranges", vmin=None, vmax=None,
                                        win_axis="before_lens", mark_best_point=True,
                                        win_input_fixed=None, width_fixed=None,
                                        win_input_fixed_axis="before_lens"):
        """NEU (gegenüber dem Original): Heatmap des ATOM-GEWICHTETEN
        Crosstalk (eta_weighted_grid) am (r_x,r_y)-Optimum unter dem
        gewichteten Ziel. Braucht eta_weighted_grid im results-dict."""
        self._require("eta_weighted_grid", "gewichtete Crosstalk-Heatmap")
        alpha = self.results.get('alpha')
        self._plot_scan2d_metric(
            grid_key="eta_weighted_grid",
            colorbar_label=r"Crosstalk$_w$ ($\eta_w$) at r-optimum (%)",
            title_metric=(rf"Crosstalk$_w$ (r-optimum, atom-weighted, $\alpha$={alpha:.2f})"
                          if alpha is not None else r"Crosstalk$_w$ (r-optimum, atom-weighted)"),
            filename_suffix="CrosstalkWeighted",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def _plot_scan2d_metric(self, grid_key, colorbar_label, title_metric, filename_suffix,
                             cmap, vmin, vmax, show, save, win_axis="before_lens",
                             mark_best_point=True, win_input_fixed=None, width_fixed=None,
                             win_input_fixed_axis="before_lens"):
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        Z = r[grid_key]

        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        win_order = win_input_vals[::-1] if reversed_ else win_input_vals
        Z_plot = Z[:, ::-1] if reversed_ else Z

        if grid_key in self.PERCENT_METRIC_KEYS:
            Z_plot = Z_plot * 100.0
            plot_vmin = None if vmin is None else vmin * 100.0
            plot_vmax = None if vmax is None else vmax * 100.0
        else:
            plot_vmin, plot_vmax = vmin, vmax

        with plt.rc_context(self.SCAN2D_RC):
            fig = plt.figure(figsize=(9, 8.9))
            ax = fig.add_axes([0.12, 0.36, 0.75, 0.56])
            info_ax = fig.add_axes([0.06, 0.03, 0.88, 0.24])
            info_ax.axis('off')

            im = ax.pcolormesh(
                x_vals, width_vals * 1e-6, Z_plot,
                shading='auto', cmap=cmap, vmin=plot_vmin, vmax=plot_vmax,
            )
            fig.colorbar(im, ax=ax, label=colorbar_label)

            dx = (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 0.1
            dy = (width_vals[1] - width_vals[0]) * 1e-6 if len(width_vals) > 1 else 0.05
            x_pad, y_pad = dx / 2, dy / 2
            ax.set_xlim(x_vals[0] - x_pad, x_vals[-1] + x_pad)
            ax.set_ylim(width_vals[0] * 1e-6 - y_pad, width_vals[-1] * 1e-6 + y_pad)

            ax.set_xlabel(x_label)
            ax.set_ylabel("width (MHz)")
            ax.set_title(title_metric)

            selection_rect = Rectangle(
                (0, 0), dx, dy, edgecolor="red", facecolor="none",
                linewidth=2.5, zorder=6,
            )
            selection_rect.set_visible(False)
            ax.add_patch(selection_rect)

            RX = r['r_x_grid']
            RY = r['r_y_grid']
            U = r.get('uniformity_grid')
            C = r.get('crosstalk_grid')
            Uw = r.get('uniformity_weighted_grid')
            Ew = r.get('eta_weighted_grid')

            if mark_best_point:
                mark = self._resolve_mark_point(win_input_fixed, width_fixed, win_input_fixed_axis)
                if mark is not None:
                    i_mark, j_mark, mark_label = mark
                    x_mark = self._win_axis_single_value(win_input_vals[j_mark], win_axis)
                    y_mark = width_vals[i_mark] * 1e-6
                    ax.plot(x_mark, y_mark, linestyle="none",
                            label=mark_label, **self.BEST_POINT_STYLE)
                    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

            info_text = info_ax.text(
                0.0, 1.0, "Click on a point in the heatmap to show all parameters "
                           "(incl. amplitude ratios r_x, r_y).",
                va='top', ha='left', fontsize=13, family='monospace',
                transform=info_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9),
            )

            def _on_click(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return
                j_plot = int(np.argmin(np.abs(x_vals - event.xdata)))
                win_input_actual = win_order[j_plot]
                width_val = event.ydata * 1e6
                j = int(np.argmin(np.abs(win_input_vals - win_input_actual)))
                i = int(np.argmin(np.abs(width_vals - width_val)))
                win_input_actual = win_input_vals[j]
                width_actual = width_vals[i]

                selection_rect.set_xy((x_vals[j_plot] - dx / 2, width_actual * 1e-6 - dy / 2))
                selection_rect.set_visible(True)

                try:
                    win_eff = self._win_input_to_win(win_input_actual)
                except ValueError:
                    win_eff = None

                lines = self._point_info_lines(
                    win_input_actual, width_actual, win_eff, RX[i, j], RY[i, j],
                    uniformity=U[i, j] if U is not None else None,
                    crosstalk=C[i, j] if C is not None else None,
                    uniformity_weighted=Uw[i, j] if Uw is not None else None,
                    eta_weighted=Ew[i, j] if Ew is not None else None,
                )
                info_text.set_text("\n".join(lines))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', _on_click)

            tag = self._filetag()
            self._finish_figure(
                fig, f"FlatMultiTone_AmpScan_{filename_suffix}_{tag}.png",
                show, save, dpi=self.SCAN2D_SAVE_DPI,
            )

    # ------------------------------------------------------------------
    # Kombinierte Ansicht: dynamisch 4-6 Heatmaps, je nachdem welche
    # Metrik-Grids im results-dict vorhanden sind (hart und/oder gewichtet)
    # + immer r_x, r_y.
    # ------------------------------------------------------------------
    def plot_scan2d_combined(self, show=True, save=True,
                              cmap_uniformity="viridis_r", cmap_crosstalk="Oranges",
                              cmap_rx="viridis", cmap_ry="viridis",
                              win_axis="before_lens", mark_best_point=True,
                              win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """
        Wie plot_scan2d_combined() im Original, aber die Anzahl der Panels
        passt sich dynamisch an: 4 Panels (Uniformity, Crosstalk, r_x, r_y),
        falls nur die harten Metriken vorhanden sind (wie im Original) ODER
        falls nur die gewichteten vorhanden sind (dann Uniformity_w/
        Crosstalk_w statt der harten); 6 Panels, falls BEIDE Metrik-Arten im
        results-dict stecken (z.B. nach manuellem Zusammenführen zweier
        Scans) - dann werden alle vier Metrik-Heatmaps PLUS r_x/r_y gezeigt.
        """
        if save:
            fig_save = self._build_combined_figure(interactive=False,
                                                     cmap_uniformity=cmap_uniformity, cmap_crosstalk=cmap_crosstalk,
                                                     cmap_rx=cmap_rx, cmap_ry=cmap_ry,
                                                     win_axis=win_axis, mark_best_point=mark_best_point,
                                                     win_input_fixed=win_input_fixed, width_fixed=width_fixed,
                                                     win_input_fixed_axis=win_input_fixed_axis)
            tag = self._filetag()
            out_file = resolve_save_path(
                self.out_dir, f"FlatMultiTone_AmpScan_Combined_{tag}.png",
                confirm_overwrite=self.confirm_overwrite,
            )
            fig_save.savefig(out_file, dpi=self.SCAN2D_SAVE_DPI, bbox_inches='tight')
            print(f"Figure saved: {out_file}")
            plt.close(fig_save)

        if show:
            self._build_combined_figure(interactive=True,
                                         cmap_uniformity=cmap_uniformity, cmap_crosstalk=cmap_crosstalk,
                                         cmap_rx=cmap_rx, cmap_ry=cmap_ry,
                                         win_axis=win_axis, mark_best_point=mark_best_point,
                                         win_input_fixed=win_input_fixed, width_fixed=width_fixed,
                                         win_input_fixed_axis=win_input_fixed_axis)
            plt.show()

    def _combined_panel_list(self, cmap_uniformity, cmap_crosstalk, cmap_rx, cmap_ry):
        r = self.results
        alpha = r.get('alpha')
        alpha_tag = f", α={alpha:.2f}" if alpha is not None else ""
        panels = []
        if self.has_hard:
            panels.append(("uniformity_grid", "Uniformity (σ/μ) at r-optimum (%)",
                            f"Uniformity (r-optimum{alpha_tag})", cmap_uniformity))
            panels.append(("crosstalk_grid", "Crosstalk (η) at r-optimum (%)",
                            f"Crosstalk (r-optimum{alpha_tag})", cmap_crosstalk))
        if self.has_weighted:
            panels.append(("uniformity_weighted_grid", r"Uniformity$_w$ ($\sigma_w/\mu_w$) at r-optimum (%)",
                            rf"Uniformity$_w$ (r-optimum, atom-weighted{alpha_tag})", cmap_uniformity))
            panels.append(("eta_weighted_grid", r"Crosstalk$_w$ ($\eta_w$) at r-optimum (%)",
                            rf"Crosstalk$_w$ (r-optimum, atom-weighted{alpha_tag})", cmap_crosstalk))
        panels.append(("r_x_grid", r"$r_x$ (outer/inner)", r"Optimal $r_x$ ($N_x$=%d)" % r['N_x'], cmap_rx))
        panels.append(("r_y_grid", r"$r_y$ (outer/inner)", r"Optimal $r_y$ ($N_y$=%d)" % r['N_y'], cmap_ry))
        return panels

    def _build_combined_figure(self, interactive, cmap_uniformity, cmap_crosstalk, cmap_rx, cmap_ry,
                                win_axis="before_lens", mark_best_point=True,
                                win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        RX = r['r_x_grid']
        RY = r['r_y_grid']
        U = r.get('uniformity_grid')
        C = r.get('crosstalk_grid')
        Uw = r.get('uniformity_weighted_grid')
        Ew = r.get('eta_weighted_grid')

        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        win_order = win_input_vals[::-1] if reversed_ else win_input_vals

        panels = self._combined_panel_list(cmap_uniformity, cmap_crosstalk, cmap_rx, cmap_ry)
        n_panels = len(panels)
        ncols = 2
        nrows = (n_panels + 1) // ncols

        dx = (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 0.1
        dy = (width_vals[1] - width_vals[0]) * 1e-6 if len(width_vals) > 1 else 0.05

        mark = self._resolve_mark_point(win_input_fixed, width_fixed, win_input_fixed_axis) if mark_best_point else None
        if mark is not None:
            i_mark, j_mark, _ = mark
            x_mark = self._win_axis_single_value(win_input_vals[j_mark], win_axis)
            y_mark = width_vals[i_mark] * 1e-6

        # Interaktive Fenster duerfen nicht linear mit der Panel-Zahl wachsen:
        # bei 6 Panels (3 Zeilen - Uniformity/Crosstalk hart+gewichtet PLUS
        # r_x/r_y, z.B. beim kombinierten Amplituden-Scan) wurde das Fenster
        # bisher >19 Zoll hoch und ragte auf normalen Bildschirmen weit unten
        # heraus (siehe Chat "Amplituden Abhängigkeit": "die Box mit den
        # Parametern kann ich nicht sehen, weil sie zu weit unten ist").
        # Zeilenhoehe daher ab 3 Zeilen kappen, statt linear weiterwachsen zu
        # lassen. Ausserdem: constrained_layout statt manuellem hspace/
        # subplots_adjust - reserviert automatisch genug Platz fuer Titel/
        # Colorbars/Achsenbeschriftungen und verhindert damit zuverlaessig
        # ueberlappende Ueberschriften, unabhaengig von der Panel-Zahl.
        row_h = 5.9 if nrows <= 2 else 3.7
        info_h = 1.1 if nrows <= 2 else 0.9
        title_fs = 17 if nrows <= 2 else 13

        with plt.rc_context(self.SCAN2D_RC):
            if interactive:
                fig = plt.figure(figsize=(7.5 * ncols, row_h * nrows + info_h),
                                  constrained_layout=True)
                gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[info_h] + [row_h] * nrows)
                # Parameter-Box bewusst als ERSTE (statt bisher letzte) Zeile:
                # bei einem zu grossen Fenster schneiden die meisten
                # Fensterverwaltungen den UNTEREN Rand ab, nicht den oberen -
                # so bleibt die Box beim Oeffnen immer sofort sichtbar.
                info_ax = fig.add_subplot(gs[0, :])
                info_ax.axis('off')
                axes = [fig.add_subplot(gs[1 + k // ncols, k % ncols]) for k in range(n_panels)]
            else:
                fig, axes2d = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 5.5 * nrows),
                                            constrained_layout=True)
                axes = list(np.atleast_1d(axes2d).ravel())[:n_panels]
                for extra_ax in list(np.atleast_1d(axes2d).ravel())[n_panels:]:
                    extra_ax.axis('off')

            selection_rects = []
            for ax, (grid_key, cbar_label, title, cmap) in zip(axes, panels):
                Z = r[grid_key]
                Z_plot = Z[:, ::-1] if reversed_ else Z
                if grid_key in self.PERCENT_METRIC_KEYS:
                    Z_plot = Z_plot * 100.0
                im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading='auto', cmap=cmap)
                x_pad, y_pad = dx / 2, dy / 2
                ax.set_xlim(x_vals[0] - x_pad, x_vals[-1] + x_pad)
                ax.set_ylim(width_vals[0] * 1e-6 - y_pad, width_vals[-1] * 1e-6 + y_pad)
                fig.colorbar(im, ax=ax, label=cbar_label)
                ax.set_xlabel(x_label)
                ax.set_ylabel("width (MHz)")
                ax.set_title(title, fontsize=title_fs)

                if mark is not None:
                    ax.plot(x_mark, y_mark, linestyle="none", **self.BEST_POINT_STYLE)

                if interactive:
                    rect = Rectangle((0, 0), dx, dy, edgecolor="red", facecolor="none",
                                      linewidth=2.2, zorder=6)
                    rect.set_visible(False)
                    ax.add_patch(rect)
                    selection_rects.append(rect)

            if not interactive:
                return fig

            info_text = info_ax.text(
                0.0, 1.0, "Click on a point in any heatmap to show all parameters "
                           "(incl. amplitude ratios r_x, r_y and the full amplitude arrays).",
                va='top', ha='left', fontsize=13, family='monospace',
                transform=info_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9),
            )

            def _on_click(event):
                if event.inaxes not in axes or event.xdata is None or event.ydata is None:
                    return
                j_plot = int(np.argmin(np.abs(x_vals - event.xdata)))
                win_input_actual = win_order[j_plot]
                width_val = event.ydata * 1e6
                j = int(np.argmin(np.abs(win_input_vals - win_input_actual)))
                i = int(np.argmin(np.abs(width_vals - width_val)))
                win_input_actual = win_input_vals[j]
                width_actual = width_vals[i]

                xy = (x_vals[j_plot] - dx / 2, width_actual * 1e-6 - dy / 2)
                for rect in selection_rects:
                    rect.set_xy(xy)
                    rect.set_visible(True)

                try:
                    win_eff = self._win_input_to_win(win_input_actual)
                except ValueError:
                    win_eff = None

                lines = self._point_info_lines(
                    win_input_actual, width_actual, win_eff, RX[i, j], RY[i, j],
                    uniformity=U[i, j] if U is not None else None,
                    crosstalk=C[i, j] if C is not None else None,
                    uniformity_weighted=Uw[i, j] if Uw is not None else None,
                    eta_weighted=Ew[i, j] if Ew is not None else None,
                )
                info_text.set_text("\n".join(lines))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', _on_click)
            return fig

    # ------------------------------------------------------------------
    # Abhängigkeits-Schnitte: UNVERÄNDERT vom Original übernommen - r_x_grid/
    # r_y_grid heißen in beiden Scan-Varianten (hart/gewichtet) identisch,
    # diese Methode braucht daher keinerlei Anpassung (siehe Modul-Docstring).
    # ------------------------------------------------------------------
    def plot_dependence_cuts(self, win_input_fixed=None, width_fixed=None, show=True, save=True,
                              win_axis="before_lens", legend_fontsize=None, mark_best_point=True,
                              win_input_fixed_axis="before_lens"):
        """Siehe multitone_amplitude_dependence_plots.AmplitudeScanPlotter.plot_dependence_cuts()
        (Original) für die volle Dokumentation - Verhalten hier 1:1 identisch."""
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        RX = r['r_x_grid']
        RY = r['r_y_grid']
        alpha = r.get('alpha', 1.0)

        mark = self._resolve_mark_point(win_input_fixed, width_fixed, win_input_fixed_axis)
        if mark is None:
            raise RuntimeError(
                "Keine gültigen Scan-Punkte vorhanden, und kein eigener Schnittpunkt "
                "(win_input_fixed/width_fixed) angegeben."
            )
        i_fixed, j_fixed, mark_label = mark

        x_win, x_win_label, _ = self._win_axis_values(win_input_vals, win_axis)

        legend_kwargs = {} if legend_fontsize is None else {"fontsize": legend_fontsize}

        with plt.rc_context(self.SCAN2D_RC):
            # constrained_layout statt tight_layout(): tight_layout() reserviert
            # nach einem fig.suptitle() (siehe unten) nicht zuverlaessig genug
            # Platz, wodurch der Suptitel mit den Panel-Ueberschriften
            # ueberlappen konnte (siehe Chat "Amplituden Abhängigkeit").
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

            ax_left.plot(x_win, RX[i_fixed, :], 'o-', label=r"$r_x$ ($N_x$=%d)" % r['N_x'])
            ax_left.plot(x_win, RY[i_fixed, :], 's-', label=r"$r_y$ ($N_y$=%d)" % r['N_y'])
            ax_left.set_xlabel(x_win_label)
            ax_left.set_ylabel("outer/inner ratio")
            ax_left.set_title(f"width fixed = {width_vals[i_fixed]*1e-6:.4f} MHz")
            ax_left.grid(True, alpha=0.3)

            ax_right.plot(width_vals * 1e-6, RX[:, j_fixed], 'o-', label=r"$r_x$ ($N_x$=%d)" % r['N_x'])
            ax_right.plot(width_vals * 1e-6, RY[:, j_fixed], 's-', label=r"$r_y$ ($N_y$=%d)" % r['N_y'])
            ax_right.set_xlabel("width (MHz)")
            ax_right.set_ylabel("outer/inner ratio")
            ax_right.set_title(f"win_input fixed = {win_input_vals[j_fixed]*1e3:.4f} mm")
            ax_right.grid(True, alpha=0.3)

            if mark_best_point:
                x_mark_win = self._win_axis_single_value(win_input_vals[j_fixed], win_axis)
                ax_left.axvline(x_mark_win, color="red", linestyle="--", linewidth=1.3, alpha=0.7)
                ax_left.plot(x_mark_win, RX[i_fixed, j_fixed], linestyle="none",
                             label=mark_label, **self.BEST_POINT_STYLE)
                ax_left.plot(x_mark_win, RY[i_fixed, j_fixed], linestyle="none", **self.BEST_POINT_STYLE)

                width_mark = width_vals[i_fixed] * 1e-6
                ax_right.axvline(width_mark, color="red", linestyle="--", linewidth=1.3, alpha=0.7)
                ax_right.plot(width_mark, RX[i_fixed, j_fixed], linestyle="none",
                              label=mark_label, **self.BEST_POINT_STYLE)
                ax_right.plot(width_mark, RY[i_fixed, j_fixed], linestyle="none", **self.BEST_POINT_STYLE)

            ax_left.legend(**legend_kwargs)
            ax_right.legend(**legend_kwargs)

            fig.suptitle(f"Amplitude-ratio dependence on waist and width (α={alpha:.2f})",
                         fontsize=15, fontweight='bold')

            tag = self._filetag()
            self._finish_figure(fig, f"FlatMultiTone_AmpScan_DependenceCuts_{tag}.png", show, save)


# ======================================================================
# NEUE Klasse: Plotter für den Fest-Amplitude-Scan der GEWICHTETEN
# Metriken (scan_win_width_weighted_uniformity() / get_scan_weighted_results(),
# siehe weighted_amp_scan_methods.py) - Analogon zum (hier nicht
# vorliegenden) ScanPlotter aus multitone_flattop_scan_plots.py, nur mit
# uniformity_weighted/eta_weighted statt uniformity/crosstalk.
# ======================================================================
class WeightedFixedScanPlotter:
    """
    Erzeugt Heatmap-Plots für den Fest-Amplitude-Scan der atom-gewichteten
    Uniformity/Crosstalk (uniformity_weighted_grid/eta_weighted_grid über
    win_input x width, bei FESTEN Amplituden - keine r_x/r_y-Optimierung
    pro Punkt) - aus einem von get_scan_weighted_results()/
    load_amp_scan_results() gelieferten dict.
    """

    SCAN2D_RC = AmplitudeScanPlotter.SCAN2D_RC
    SCAN2D_SAVE_DPI = AmplitudeScanPlotter.SCAN2D_SAVE_DPI
    BEST_POINT_STYLE = AmplitudeScanPlotter.BEST_POINT_STYLE

    def __init__(self, results, out_dir=None, confirm_overwrite=None):
        missing = [k for k in ("uniformity_weighted_grid", "eta_weighted_grid",
                                "win_input_vals", "width_vals") if k not in results]
        if missing:
            raise ValueError(
                "results fehlen die Schlüssel " + ", ".join(missing) + " - das sieht nicht "
                "nach den Rohdaten von scan_win_width_weighted_uniformity() "
                "(weighted_amp_scan_methods.py) aus."
            )
        self.results = results
        self.out_dir = Path(out_dir) if out_dir is not None else DEFAULT_IMAGES_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_overwrite = confirm_overwrite

    def _win_input_to_win(self, win_input):
        r = self.results
        return win_input_to_win(win_input, r['f1'], r['f2'], r['lambda_opt'], r['fLO'])

    def _profile_tag(self):
        profile = self.results.get('profile')
        if profile == 'airy':
            return 'Airy'
        if profile == 'gaussian':
            return 'Gauss'
        return 'ProfilUnbekannt'

    def _filetag(self):
        r = self.results
        n_win = len(r.get('win_input_vals', []))
        n_width = len(r.get('width_vals', []))
        return f"N{r['N_x']}x{r['N_y']}_{n_win}x{n_width}pts_{self._profile_tag()}_Weighted"

    def _finish_figure(self, fig, filename, show, save, dpi=150):
        if save:
            out_file = resolve_save_path(self.out_dir, filename, confirm_overwrite=self.confirm_overwrite)
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {out_file}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _win_axis_values(self, win_input_vals, win_axis):
        if win_axis == "before_lens":
            return win_input_vals * 1e3, r"Input waist $\omega_{\mathrm{in}}$ (before lenses, mm)", False
        elif win_axis == "after_lens":
            x = np.array([self._win_input_to_win(w) for w in win_input_vals]) * 1e6
            if len(x) > 1 and x[0] > x[-1]:
                return x[::-1], r"Waist at focus $\omega'$ (after lenses, µm)", True
            return x, r"Waist at focus $\omega'$ (after lenses, µm)", False
        else:
            raise ValueError(f"win_axis muss 'before_lens' oder 'after_lens' sein, nicht {win_axis!r}.")

    def _mark_point(self):
        r = self.results
        best = r.get('best', {})
        if best.get('win_input') is None:
            return None
        j = int(np.argmin(np.abs(r['win_input_vals'] - best['win_input'])))
        i = int(np.argmin(np.abs(r['width_vals'] - best['width'])))
        return i, j

    def _plot_one(self, grid_key, colorbar_label, title, cmap, filename_suffix,
                   win_axis, show, save):
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        Z_plot = (r[grid_key][:, ::-1] if reversed_ else r[grid_key]) * 100.0

        with plt.rc_context(self.SCAN2D_RC):
            fig, ax = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
            im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
            fig.colorbar(im, ax=ax, label=colorbar_label)

            mark = self._mark_point()
            if mark is not None:
                i_mark, j_mark = mark
                x_mark = self._win_axis_values(np.array([win_input_vals[j_mark]]), win_axis)[0][0]
                y_mark = width_vals[i_mark] * 1e-6
                ax.plot(x_mark, y_mark, linestyle="none", label="best point (global optimum)",
                        **self.BEST_POINT_STYLE)
                ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

            ax.set_xlabel(x_label)
            ax.set_ylabel("width (MHz)")
            ax.set_title(title)

        tag = self._filetag()
        self._finish_figure(fig, f"FlatMultiTone_Scan_{filename_suffix}_{tag}.png", show, save,
                             dpi=self.SCAN2D_SAVE_DPI)
        return fig

    def plot_scan2d_weighted_uniformity(self, show=True, save=True, cmap="viridis_r", win_axis="before_lens"):
        """Heatmap von uniformity_weighted (%) über (win_input, width) bei festen Amplituden."""
        return self._plot_one(
            "uniformity_weighted_grid", r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)",
            "Uniformity_w (atom-weighted, fixed-amplitude scan)", cmap, "UniformityWeighted",
            win_axis, show, save,
        )

    def plot_scan2d_weighted_crosstalk(self, show=True, save=True, cmap="Oranges", win_axis="before_lens"):
        """Heatmap von eta_weighted (%) über (win_input, width) bei festen Amplituden."""
        return self._plot_one(
            "eta_weighted_grid", r"Crosstalk$_w$ ($\eta_w$) (%)",
            "Crosstalk_w (atom-weighted, fixed-amplitude scan)", cmap, "CrosstalkWeighted",
            win_axis, show, save,
        )

    def plot_scan2d_weighted_combined(self, show=True, save=True,
                                       cmap_uniformity="viridis_r", cmap_crosstalk="Oranges",
                                       win_axis="before_lens"):
        """Beide Heatmaps (Uniformity_w, Crosstalk_w) nebeneinander in einer Figure."""
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        mark = self._mark_point()

        panels = [
            ("uniformity_weighted_grid", r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)",
             "Uniformity_w (atom-weighted)", cmap_uniformity),
            ("eta_weighted_grid", r"Crosstalk$_w$ ($\eta_w$) (%)",
             "Crosstalk_w (atom-weighted)", cmap_crosstalk),
        ]

        with plt.rc_context(self.SCAN2D_RC):
            fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
            for ax, (grid_key, cbar_label, title, cmap) in zip(axes, panels):
                Z_plot = (r[grid_key][:, ::-1] if reversed_ else r[grid_key]) * 100.0
                im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
                fig.colorbar(im, ax=ax, label=cbar_label)
                ax.set_xlabel(x_label)
                ax.set_ylabel("width (MHz)")
                ax.set_title(title)
                if mark is not None:
                    i_mark, j_mark = mark
                    x_mark = self._win_axis_values(np.array([win_input_vals[j_mark]]), win_axis)[0][0]
                    y_mark = width_vals[i_mark] * 1e-6
                    ax.plot(x_mark, y_mark, linestyle="none", label="best point (global optimum)",
                            **self.BEST_POINT_STYLE)
                    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

            sigma_atom = r.get('sigma_atom')
            suptitle = "Waist-width scan: atom-weighted metrics (fixed-amplitude)"
            if sigma_atom is not None and np.isfinite(sigma_atom):
                suptitle += rf" ($\sigma_{{\mathrm{{atom}}}}$={sigma_atom*1e9:.1f} nm)"
            fig.suptitle(suptitle)

        tag = self._filetag()
        self._finish_figure(fig, f"FlatMultiTone_Scan_WeightedCombined_{tag}.png", show, save,
                             dpi=self.SCAN2D_SAVE_DPI)
        return fig

    # Einheitlicher Name fuer beide Ordner: Hard_Optimization/lib/
    # multitone_amplitude_dependence_plots.FixedScanPlotter heisst die Methode
    # plot_scan2d_combined(). lib/report.py ist in beiden Ordnern dieselbe
    # Datei und ruft deshalb diesen Namen auf; der alte bleibt gueltig.
    plot_scan2d_combined = plot_scan2d_weighted_combined


if __name__ == "__main__":
    print(__doc__)
