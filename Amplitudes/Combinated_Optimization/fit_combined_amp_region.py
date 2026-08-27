"""
fit_combined_amp_region.py
============================
Wie fit_combined_region.py, aber fuer den KOMBINIERTEN AMPLITUDEN-
OPTIMIERTEN Scan (scan_amp_data_combined_...pkl, aus
combined_winwidthampscan_startdialog.py / combined_amp_scan_methods.py).

Laedt das kombinierte Ergebnis-dict, berechnet daraus (guenstig, OHNE
Re-Scan - siehe recombine_from_grids() in combined_scan_methods.py, wird
hier unveraendert wiederverwendet: die vier fuer die Kombination noetigen
Rohgrids heissen in beiden Scan-Arten identisch) die kombinierte Region und
schreibt:

    CombinedAmpRegion_N{Nx}x{Ny}_{n_win}x{n_width}pts_{Profil}_{Datum}

gefolgt von:
    _metric_comparison.pdf  - 2x2 hart vs. weighted (Uniformity + Crosstalk)
    _region.pdf              - kombinierter Score + Region-Rechteck
    _Region.md                - Formel-Dokument (Kombinationsformel, Region,
                                 bester Punkt inkl. r_x/r_y AUS BEIDEN Scans)

Zusaetzlich (im Unterschied zum Fest-Amplituden-Pendant) werden die
Standard-Uebersichtsplots des Amplituden-Scans selbst erzeugt - unveraendert
ueber AmplitudeScanPlotter (erkennt has_hard=has_weighted=True automatisch):
    FlatMultiTone_AmpScan_Combined_....png       (6-Panel: Uniformity/
                                                    Crosstalk je hart+gewichtet,
                                                    plus r_x, r_y)
    FlatMultiTone_AmpScan_DependenceCuts_....png (waist-/width-Schnitte)
beide in Fit_Plots (nicht Bilder), damit alle Ausgaben dieses Skripts an
einem Ort landen.

Da combo_lambda/combo_percentile/alpha aus den bereits im pkl gespeicherten
Rohgrids (uniformity_grid, crosstalk_grid, uniformity_weighted_grid,
eta_weighted_grid) neu kombiniert werden koennen, lassen sich diese drei
Parameter hier per Config (oder ueber main()-Argumente) frei nachjustieren,
OHNE den (teuren, da pro Punkt eine (r_x,r_y)-Optimierung noetig ist) Scan
zu wiederholen.

Nutzung:
    python fit_combined_amp_region.py
(vorher ggf. PKL_DATEI unten anpassen, oder automatische Erkennung der
neuesten scan_amp_data_combined_*.pkl in Results/ nutzen lassen).
"""
import sys
from pathlib import Path
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

_WEIGHTED_DIR = Path(__file__).resolve().parent.parent / "Weighted_Optimization"
if str(_WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(_WEIGHTED_DIR))

from weighted_multitone_amplitude_dependence_plots import (  # noqa: E402
    win_input_to_win, resolve_save_path, AmplitudeScanPlotter,
)

from combined_scan_methods import (  # noqa: E402
    DEFAULT_RESULTS_DIR, load_combined_scan_results, recombine_from_grids,
)
from combined_scan_plots import CombinedFixedScanPlotter  # noqa: E402


def _default_dir(name):
    candidate = Path(__file__).resolve().parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = Path(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


FIT_PLOTS_DIR = _default_dir("Fit_Plots")
FIT_RESULTS_DIR = _default_dir("Fit_Results")

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
PKL_DATEI = r"scan_amp_data_combined_N3x4_15x15pts_Airy.pkl"

# None = aus dem pkl uebernehmen (dort beim Scan festgelegt); explizit
# gesetzt = guenstiges Nachjustieren OHNE Re-Scan (siehe recombine_from_grids()).
ALPHA = None
COMBO_LAMBDA = None
COMBO_PERCENTILE = None

WIN_AXIS = "before_lens"  # "before_lens" (mm vor der Linse) oder "after_lens" (µm nach der Linse)

SAVE = True
SHOW = False
ASK_BEFORE_SAVE = True

# Grafische Feintuning-Optionen, analog zu fit_combined_region.py -
# per run_all_fits_combined.py von aussen ueberschreibbar.
LEGEND_FONTSIZE = 9
DRAW_BEST_POINT = True

# Zusaetzlich die Standard-Amplituden-Scan-Uebersichtsplots (6-Panel +
# Dependence-Cuts, ueber AmplitudeScanPlotter, unveraendertes Modul)
# erzeugen? (siehe Modul-Docstring)
PLOT_AMP_OVERVIEW = True


def _resolve_pkl(configured_name):
    """Wie _resolve_pkl() in fit_combined_region.py: faellt auf die
    neueste scan_amp_data_combined_*.pkl in Results/ zurueck, falls der
    eingetragene Name nicht existiert."""
    here = Path(__file__).resolve().parent / configured_name
    in_results = DEFAULT_RESULTS_DIR / configured_name
    if here.exists() or in_results.exists():
        return configured_name

    kandidaten = sorted(DEFAULT_RESULTS_DIR.glob("scan_amp_data_combined_*.pkl"),
                         key=lambda p: p.stat().st_mtime)
    if not kandidaten:
        print(f"WARNUNG: weder '{configured_name}' noch irgendeine 'scan_amp_data_combined_*.pkl'-Datei "
              f"in '{DEFAULT_RESULTS_DIR}' gefunden.")
        return configured_name

    neueste = kandidaten[-1]
    if neueste.name != configured_name:
        print(f"Hinweis: '{configured_name}' nicht gefunden - verwende stattdessen "
              f"die neueste passende Datei: '{neueste.name}'.")
    return neueste.name


def _output_prefix(results):
    n_win = len(results['win_input_vals'])
    n_width = len(results['width_vals'])
    profile_tag = "Airy" if results['profile'] == 'airy' else "Gauss" if results['profile'] == 'gaussian' else results['profile']
    today = date.today().isoformat()
    return f"CombinedAmpRegion_N{results['N_x']}x{results['N_y']}_{n_win}x{n_width}pts_{profile_tag}_{today}"


def _fmt_range_um(results, win_input_lo, win_input_hi):
    """win_input-Grenzen (Meter, vor der Linse) als effektiver Waist (µm,
    nach der Linse) - Achtung, win_input_to_win() ist monoton FALLEND,
    daher kann sich die Reihenfolge umkehren."""
    a = win_input_to_win(win_input_lo, results['f1'], results['f2'], results['lambda_opt'], results['fLO']) * 1e6
    b = win_input_to_win(win_input_hi, results['f1'], results['f2'], results['lambda_opt'], results['fLO']) * 1e6
    return (min(a, b), max(a, b))


def _r_at_best(results):
    """Liest r_x/r_y AUS BEIDEN Scans (hart und gewichtet) am besten
    kombinierten Gitterpunkt aus - existieren nur bei einem Ergebnis aus
    scan_win_width_amplitude_dependence_combined() (die vier Zusatz-
    Schluessel r_x_grid_hart/r_y_grid_hart/r_x_grid_weighted/
    r_y_grid_weighted, siehe combined_amp_scan_methods.py)."""
    b = results['best']
    if b['win_input'] is None:
        return None
    j = int(np.argmin(np.abs(results['win_input_vals'] - b['win_input'])))
    i = int(np.argmin(np.abs(results['width_vals'] - b['width'])))
    out = {}
    for key in ('r_x_grid_hart', 'r_y_grid_hart', 'r_x_grid_weighted', 'r_y_grid_weighted'):
        if key in results:
            out[key] = float(np.asarray(results[key])[i, j])
    return out or None


def write_region_doc(results, output_path):
    """Wie write_region_doc() in fit_combined_region.py, zusaetzlich mit
    den an dem besten Punkt gefundenen Amplituden-Verhaeltnissen r_x/r_y.
    Wird bei jedem Lauf neu erzeugt und OHNE Rueckfrage ueberschrieben
    (vollstaendig aus dem pkl + den Kombinationsparametern reproduzierbar,
    analog zu den anderen automatisch generierten Formel-Dokumenten im
    Projekt, siehe namenskonvention_fit_outputs.md).

    results.get('joint_optimization'): unterscheidet, ob r_x/r_y aus der
    GEMEINSAMEN (jointen) Optimierung stammen (scan_win_width_amplitude_
    dependence_combined_joint() - EIN gemeinsames r_x/r_y pro Punkt,
    direkt gegen die Kombination minimiert) oder aus dem AELTEREN,
    GETRENNTEN Verfahren (scan_win_width_amplitude_dependence_combined() -
    zwei unabhaengig gefundene r_x/r_y-Saetze, die sich unterscheiden
    koennen)."""
    r = results
    b = r['best']
    reg = r['region']
    joint = bool(r.get('joint_optimization'))

    lines = []
    lines.append(f"# Combined Amplitude-Dependence Region - N{r['N_x']}x{r['N_y']}, "
                  f"{len(r['win_input_vals'])}x{len(r['width_vals'])} pts, "
                  f"{r['profile']}, {date.today().isoformat()}\n")
    if joint:
        lines.append(
            "An JEDEM (win_input, width)-Gitterpunkt wurde GENAU EINE (r_x, r_y)-"
            "Optimierung durchgefuehrt, die DIREKT gegen die Kombination aus "
            "hartem und atom-gewichtetem Ziel minimiert (siehe "
            "scan_win_width_amplitude_dependence_combined_joint() in "
            "combined_amp_scan_methods.py) - Uniformity_hart/Crosstalk_hart UND "
            "Uniformity_weighted/Crosstalk_weighted wurden dabei am SELBEN r_x/r_y "
            "ausgewertet, r_x/r_y sind daher automatisch fuer BEIDE Kriterien "
            "gleichzeitig gueltig.\n"
        )
    else:
        lines.append(
            "An JEDEM (win_input, width)-Gitterpunkt wurde eine eigene (r_x, r_y)-"
            "Optimierung durchgefuehrt (einmal unter dem harten, einmal unter dem "
            "atom-gewichteten Ziel, GETRENNT - siehe "
            "scan_win_width_amplitude_dependence_combined() in "
            "combined_amp_scan_methods.py) - die folgende Kombination bezieht "
            "sich auf die dabei jeweils UNABHAENGIG erreichten Uniformity/"
            "Crosstalk-Werte; r_x/r_y koennen zwischen hart und weighted "
            "abweichen (siehe unten).\n"
        )
    lines.append("## Kombinationsformel\n")
    lines.append(
        (
            "Fuer die folgende Region-/Score-Uebersicht (NICHT fuer die "
            "Amplituden-Suche selbst - siehe oben) wird jede der vier "
            "Rohgroessen (Uniformity_hart, Crosstalk_hart, Uniformity_weighted, "
            "Crosstalk_weighted, jeweils am gemeinsamen (r_x,r_y)-Optimum) "
            "unabhaengig ueber das Scan-Gitter Min-Max-normiert (X_norm in "
            "[0,1]).\n\n" if joint else
            "Jede der vier Rohgroessen (Uniformity_hart, Crosstalk_hart, "
            "Uniformity_weighted, Crosstalk_weighted - jeweils am eigenen "
            "(r_x,r_y)-Optimum) wird unabhaengig ueber das Scan-Gitter Min-Max-"
            "normiert (X_norm in [0,1]).\n\n"
        ) +
        "Daraus:\n\n"
        "```\n"
        "X_kombi = 0.5*(X_hart_norm + X_weighted_norm)\n"
        "          + combo_lambda * |X_hart_norm - X_weighted_norm|\n"
        "combined_score = alpha*Uniformity_kombi + (1-alpha)*Crosstalk_kombi\n"
        "```\n\n"
        f"Parameter dieses Laufs: alpha = {r['alpha']:.3f}, "
        f"combo_lambda = {r['combo_lambda']:.3f}, "
        f"combo_percentile = {r['combo_percentile']:.1f}%.\n"
    )

    lines.append("## Region\n")
    if reg['win_input_min'] is not None:
        um_lo, um_hi = _fmt_range_um(r, reg['win_input_min'], reg['win_input_max'])
        lines.append(
            f"Groesstes achsenparalleles Rechteck innerhalb der besten "
            f"{r['combo_percentile']:.0f}% aller Gitterpunkte (nach combined_score); "
            f"{reg['n_points_region']}/{reg['n_points_total']} Gitterpunkte insgesamt im "
            f"Akzeptanzbereich (Schwellwert combined_score <= {reg['threshold']:.4f}).\n\n"
            f"- win_input (vor der Linse): {reg['win_input_min']*1e3:.4f} .. "
            f"{reg['win_input_max']*1e3:.4f} mm\n"
            f"- effektiver Waist (nach der Linse): {um_lo:.4f} .. {um_hi:.4f} µm\n"
            f"- width: {reg['width_min']*1e-6:.4f} .. {reg['width_max']*1e-6:.4f} MHz\n"
        )
    else:
        lines.append("Kein gueltiges Rechteck gefunden (zu wenige valide Punkte im Scan).\n")

    lines.append("## Bester Einzelpunkt (Minimum von combined_score)\n")
    if b['win_input'] is not None:
        um = win_input_to_win(b['win_input'], r['f1'], r['f2'], r['lambda_opt'], r['fLO']) * 1e6
        lines.append(
            f"- win_input = {b['win_input']*1e3:.4f} mm ({um:.4f} µm effektiver Waist)\n"
            f"- width = {b['width']*1e-6:.4f} MHz\n"
            f"- Uniformity_hart = {b['uniformity_hart']*100:.3f}%, "
            f"Crosstalk_hart = {b['crosstalk_hart']*100:.3f}%\n"
            f"- Uniformity_weighted = {b['uniformity_weighted']*100:.3f}%, "
            f"Crosstalk_weighted = {b['crosstalk_weighted']*100:.3f}%\n"
            f"- Uniformity_kombi = {b['uniformity_kombi']:.4f}, "
            f"Crosstalk_kombi = {b['crosstalk_kombi']:.4f} (normierte Einheiten)\n"
            f"- combined_score = {b['combined_score']:.4f}\n"
        )
        r_best = _r_at_best(r)
        if r_best is not None:
            if joint:
                lines.append(
                    "\nAn diesem Gitterpunkt gefundene Amplituden-Verhaeltnisse (EINE "
                    "gemeinsame (r_x,r_y)-Optimierung - hart und weighted wurden am "
                    "SELBEN Punkt ausgewertet, daher identisch fuer beide):\n\n"
                    f"- r_x/r_y = {r_best.get('r_x_grid_hart', float('nan')):.4f} / "
                    f"{r_best.get('r_y_grid_hart', float('nan')):.4f}\n"
                )
            else:
                lines.append(
                    "\nAn diesem Gitterpunkt gefundene Amplituden-Verhaeltnisse "
                    "(koennen zwischen hart und gewichtet abweichen, da beide "
                    "Scans unterschiedliche Zielfunktionen optimieren):\n\n"
                    f"- r_x/r_y (hart)      = {r_best.get('r_x_grid_hart', float('nan')):.4f} / "
                    f"{r_best.get('r_y_grid_hart', float('nan')):.4f}\n"
                    f"- r_x/r_y (weighted)  = {r_best.get('r_x_grid_weighted', float('nan')):.4f} / "
                    f"{r_best.get('r_y_grid_weighted', float('nan')):.4f}\n"
                    f"- primaerer r_x/r_y   = {r.get('r_grid_source')!r} "
                    f"-> r_x={r['r_x_grid'][int(np.argmin(np.abs(r['width_vals']-b['width']))), int(np.argmin(np.abs(r['win_input_vals']-b['win_input'])))]:.4f}, "
                    f"r_y={r['r_y_grid'][int(np.argmin(np.abs(r['width_vals']-b['width']))), int(np.argmin(np.abs(r['win_input_vals']-b['win_input'])))]:.4f} "
                    "(dieser Satz steckt in r_x_grid/r_y_grid und wird von AmplitudeScanPlotter angezeigt)\n"
                )
    else:
        lines.append("Kein gueltiger Punkt gefunden.\n")

    grid_note = ("EINE gemeinsame (r_x,r_y)-Optimierung, gegen die Kombination" if joint
                 else "eigene (r_x,r_y)-Optimierung, zweimal - hart + gewichtet, GETRENNT")
    lines.append("## Scan-Parameter\n")
    lines.append(
        f"- N_x={r['N_x']}, N_y={r['N_y']}, Profil={r['profile']}\n"
        f"- Gitterpunkte: {len(r['win_input_vals'])} x {len(r['width_vals'])} "
        f"(JEDER Punkt = {grid_note})\n"
        f"- r_bounds = {r.get('r_bounds')}\n"
        f"- sigma_atom = {r.get('sigma_atom', float('nan'))*1e9:.1f} nm "
        f"(atom_temperature={r.get('atom_temperature', float('nan'))*1e6:.2f} µK, "
        f"trap_freq_r={r.get('trap_freq_r', float('nan'))*1e-3:.2f} kHz)\n"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Region-Dokument geschrieben: {output_path}")


def main(pkl_datei=None, alpha=None, combo_lambda=None, combo_percentile=None,
         win_axis=None, draw_best_point=None, legend_fontsize=None,
         ask_before_save=None, save=None, show=None, plot_amp_overview=None):
    pkl_datei = PKL_DATEI if pkl_datei is None else pkl_datei
    alpha = ALPHA if alpha is None else alpha
    combo_lambda = COMBO_LAMBDA if combo_lambda is None else combo_lambda
    combo_percentile = COMBO_PERCENTILE if combo_percentile is None else combo_percentile
    win_axis = WIN_AXIS if win_axis is None else win_axis
    draw_best_point = DRAW_BEST_POINT if draw_best_point is None else draw_best_point
    legend_fontsize = LEGEND_FONTSIZE if legend_fontsize is None else legend_fontsize
    ask_before_save = ASK_BEFORE_SAVE if ask_before_save is None else ask_before_save
    save = SAVE if save is None else save
    show = SHOW if show is None else show
    plot_amp_overview = PLOT_AMP_OVERVIEW if plot_amp_overview is None else plot_amp_overview

    resolved_name = _resolve_pkl(pkl_datei)
    raw = load_combined_scan_results(resolved_name)
    results = recombine_from_grids(raw, alpha=alpha, combo_lambda=combo_lambda,
                                    combo_percentile=combo_percentile)

    prefix = _output_prefix(results)

    confirm_overwrite = None if ask_before_save else (lambda existing_path: True)

    plotter = CombinedFixedScanPlotter(results, out_dir=FIT_PLOTS_DIR, confirm_overwrite=confirm_overwrite)
    # CombinedFixedScanPlotter selbst speichert als PNG (fuer die interaktive
    # GUI, siehe combined_winwidthampscan_startdialog.py) - fuer die
    # LaTeX-taugliche PDF-Ausgabe (Konvention der anderen Fit-Skripte, Vektor-
    # PDF statt PNG) wird hier direkt mit dem Matplotlib-Objekt gearbeitet,
    # nutzt aber dieselben Hilfsmethoden (_win_axis_values, _draw_region_and_mark).
    with plt.rc_context({"legend.fontsize": legend_fontsize}):
        cmp_path = _save_metric_comparison_pdf(plotter, prefix, draw_best_point, win_axis, show, save)
        region_path = _save_region_pdf(plotter, prefix, draw_best_point, win_axis, show, save)

    amp_overview_paths = None
    if plot_amp_overview:
        # AmplitudeScanPlotter erkennt automatisch, dass sowohl harte als
        # auch gewichtete Metrik-Grids vorhanden sind (has_hard=
        # has_weighted=True) und zeichnet die 6-Panel-Uebersicht
        # (Uniformity/Crosstalk je hart+gewichtet, plus r_x/r_y - der
        # "primaere" Satz, siehe r_grid_source in combined_amp_scan_methods.py)
        # sowie die waist-/width-Schnitte - unveraendertes Modul, hier nur in
        # Fit_Plots statt Bilder umgeleitet.
        amp_plotter = AmplitudeScanPlotter(results, out_dir=FIT_PLOTS_DIR, confirm_overwrite=confirm_overwrite)
        overview_path = amp_plotter.plot_scan2d_combined(show=show, save=save)
        cuts_path = amp_plotter.plot_dependence_cuts(show=show, save=save)
        amp_overview_paths = dict(overview=overview_path, dependence_cuts=cuts_path)

    formula_doc = None
    if save:
        formula_doc = FIT_RESULTS_DIR / f"{prefix}_Region.md"
        write_region_doc(results, formula_doc)

    best_point = None
    if results['best']['win_input'] is not None:
        b = results['best']
        um = win_input_to_win(b['win_input'], results['f1'], results['f2'],
                               results['lambda_opt'], results['fLO']) * 1e6
        best_point = dict(win_input_mm=b['win_input'] * 1e3, waist_um=um,
                           width_MHz=b['width'] * 1e-6, combined_score=b['combined_score'],
                           r_at_best=_r_at_best(results))

    region_summary = None
    if results['region']['win_input_min'] is not None:
        reg = results['region']
        um_lo, um_hi = _fmt_range_um(results, reg['win_input_min'], reg['win_input_max'])
        region_summary = dict(
            win_input_mm=(reg['win_input_min'] * 1e3, reg['win_input_max'] * 1e3),
            waist_um=(um_lo, um_hi),
            width_MHz=(reg['width_min'] * 1e-6, reg['width_max'] * 1e-6),
            n_points_region=reg['n_points_region'], n_points_total=reg['n_points_total'],
        )

    return dict(
        results=results, prefix=prefix, formula_doc=formula_doc,
        best_point=best_point, region=region_summary,
        plots=dict(metric_comparison=cmp_path, region=region_path,
                    amp_overview=amp_overview_paths),
    )


def _save_metric_comparison_pdf(plotter, prefix, draw_best_point, win_axis, show, save):
    import matplotlib.pyplot as plt
    r = plotter.results
    win_input_vals = r['win_input_vals']
    width_vals = r['width_vals']
    x_vals, x_label, reversed_ = plotter._win_axis_values(win_input_vals, win_axis)

    panels = [
        ("uniformity_grid", r"Uniformity ($\sigma/\mu$) (%)", "Uniformity (hard, global)", "viridis_r"),
        ("uniformity_weighted_grid", r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)",
         "Uniformity$_w$ (atom-weighted, local)", "viridis_r"),
        ("crosstalk_grid", r"Crosstalk ($\eta$) (%)", "Crosstalk (hard, global)", "Oranges"),
        ("eta_weighted_grid", r"Crosstalk$_w$ ($\eta_w$) (%)",
         "Crosstalk$_w$ (atom-weighted, local)", "Oranges"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), constrained_layout=True)
    for ax, (grid_key, cbar_label, title, cmap) in zip(axes.flat, panels):
        Z_plot = (r[grid_key][:, ::-1] if reversed_ else r[grid_key]) * 100.0
        im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel("width (MHz)")
        ax.set_title(title)
        if draw_best_point:
            plotter._draw_region_and_mark(ax, win_axis)

    fig.suptitle("Amplitude-optimized scan: hard vs. atom-weighted metrics "
                  "(each point its own (r_x,r_y) optimum)")

    out_path = None
    if save:
        out_path = resolve_save_path(plotter.out_dir, f"{prefix}_metric_comparison.pdf",
                                      confirm_overwrite=plotter.confirm_overwrite)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def _save_region_pdf(plotter, prefix, draw_best_point, win_axis, show, save):
    import matplotlib.pyplot as plt
    r = plotter.results
    win_input_vals = r['win_input_vals']
    width_vals = r['width_vals']
    x_vals, x_label, reversed_ = plotter._win_axis_values(win_input_vals, win_axis)
    Z_plot = r['combined_score'][:, ::-1] if reversed_ else r['combined_score']

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap="magma_r")
    fig.colorbar(im, ax=ax, label="combined score (normalized, smaller = better)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("width (MHz)")

    region = r.get('region', {})
    pct = r.get('combo_percentile')
    n_region = region.get('n_points_region')
    n_total = region.get('n_points_total')
    title = "Combined region (hard + atom-weighted, amplitude-optimized)"
    if pct is not None and n_region is not None:
        title += f"\nbest {pct:.0f}% of score, largest rectangle: {n_region}/{n_total} points"
    ax.set_title(title)

    if draw_best_point:
        plotter._draw_region_and_mark(ax, win_axis)

    out_path = None
    if save:
        out_path = resolve_save_path(plotter.out_dir, f"{prefix}_region.pdf",
                                      confirm_overwrite=plotter.confirm_overwrite)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
