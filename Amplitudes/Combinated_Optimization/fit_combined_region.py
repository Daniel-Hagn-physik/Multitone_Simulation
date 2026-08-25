"""
fit_combined_region.py
========================
Laedt einen kombinierten Fest-Amplituden-Scan (scan_data_combined_...pkl,
aus combined_winwidthscan_startdialog.py / combined_scan_methods.py),
berechnet daraus (guenstig, OHNE Re-Scan - siehe recombine_from_grids() in
combined_scan_methods.py) die kombinierte Region und schreibt Plots +
ein Formel-/Region-Dokument, in Anlehnung an die Namenskonvention der
uebrigen Fit-Skripte im Projekt (siehe status.md/
namenskonvention_fit_outputs.md):

    CombinedRegion_N{Nx}x{Ny}_{n_win}x{n_width}pts_{Profil}_{Datum}

gefolgt von:
    _metric_comparison.pdf  - 2x2 hart vs. weighted (Uniformity + Crosstalk)
    _region.pdf              - kombinierter Score + Region-Rechteck
    _Region.md                - Formel-Dokument (Kombinationsformel, Region,
                                 bester Punkt, alle vier Rohmetriken daran)

Da combo_lambda/combo_percentile/alpha aus den bereits im pkl gespeicherten
Rohgrids (uniformity_grid, crosstalk_grid, uniformity_weighted_grid,
eta_weighted_grid) neu kombiniert werden koennen, lassen sich diese drei
Parameter hier per Config (oder ueber main()-Argumente) frei nachjustieren,
OHNE den (teuren) Scan zu wiederholen.

Nutzung:
    python fit_combined_region.py
(vorher ggf. PKL_DATEI unten anpassen, oder automatische Erkennung der
neuesten scan_data_combined_*.pkl in Results/ nutzen lassen).
"""
import sys
from pathlib import Path
from datetime import date

import numpy as np
import matplotlib.pyplot as plt

_WEIGHTED_DIR = Path(__file__).resolve().parent.parent / "Weighted_Optimization"
if str(_WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(_WEIGHTED_DIR))

from weighted_multitone_amplitude_dependence_plots import win_input_to_win, resolve_save_path  # noqa: E402

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
PKL_DATEI = r"scan_data_combined_N3x4_40x40pts_Airy.pkl"

# None = aus dem pkl uebernehmen (dort beim Scan festgelegt); explizit
# gesetzt = guenstiges Nachjustieren OHNE Re-Scan (siehe recombine_from_grids()).
ALPHA = None
COMBO_LAMBDA = None
COMBO_PERCENTILE = None

WIN_AXIS = "before_lens"  # "before_lens" (mm vor der Linse) oder "after_lens" (µm nach der Linse)

SAVE = True
SHOW = False
ASK_BEFORE_SAVE = True

# Grafische Feintuning-Optionen, analog zu den anderen Fit-Skripten -
# per run_all_fits_combined.py von aussen ueberschreibbar.
LEGEND_FONTSIZE = 9
DRAW_BEST_POINT = True


def _resolve_pkl(configured_name):
    """Wie _resolve_pkl() in run_all_fits.py: faellt auf die neueste
    scan_data_combined_*.pkl in Results/ zurueck, falls der eingetragene
    Name nicht existiert."""
    here = Path(__file__).resolve().parent / configured_name
    in_results = DEFAULT_RESULTS_DIR / configured_name
    if here.exists() or in_results.exists():
        return configured_name

    kandidaten = sorted(DEFAULT_RESULTS_DIR.glob("scan_data_combined_*.pkl"),
                         key=lambda p: p.stat().st_mtime)
    if not kandidaten:
        print(f"WARNUNG: weder '{configured_name}' noch irgendeine 'scan_data_combined_*.pkl'-Datei "
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
    return f"CombinedRegion_N{results['N_x']}x{results['N_y']}_{n_win}x{n_width}pts_{profile_tag}_{today}"


def _fmt_range_um(results, win_input_lo, win_input_hi):
    """win_input-Grenzen (Meter, vor der Linse) als effektiver Waist (µm,
    nach der Linse) - Achtung, win_input_to_win() ist monoton FALLEND,
    daher kann sich die Reihenfolge umkehren."""
    a = win_input_to_win(win_input_lo, results['f1'], results['f2'], results['lambda_opt'], results['fLO']) * 1e6
    b = win_input_to_win(win_input_hi, results['f1'], results['f2'], results['lambda_opt'], results['fLO']) * 1e6
    return (min(a, b), max(a, b))


def write_region_doc(results, output_path):
    """Schreibt das Formel-/Region-Dokument - wird bei jedem Lauf neu
    erzeugt und OHNE Rueckfrage ueberschrieben (vollstaendig aus dem pkl +
    den Kombinationsparametern reproduzierbar, analog zu den anderen
    automatisch generierten Formel-Dokumenten im Projekt, siehe
    namenskonvention_fit_outputs.md, Addendum 2026-08-25)."""
    r = results
    b = r['best']
    reg = r['region']

    lines = []
    lines.append(f"# Combined Region - N{r['N_x']}x{r['N_y']}, "
                  f"{len(r['win_input_vals'])}x{len(r['width_vals'])} pts, "
                  f"{r['profile']}, {date.today().isoformat()}\n")
    lines.append("## Kombinationsformel\n")
    lines.append(
        "Jede der vier Rohgroessen (Uniformity_hart, Crosstalk_hart, "
        "Uniformity_weighted, Crosstalk_weighted) wird unabhaengig ueber das "
        "Scan-Gitter Min-Max-normiert (X_norm in [0,1]). Daraus:\n\n"
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
    else:
        lines.append("Kein gueltiger Punkt gefunden.\n")

    lines.append("## Scan-Parameter\n")
    lines.append(
        f"- N_x={r['N_x']}, N_y={r['N_y']}, Profil={r['profile']}\n"
        f"- Gitterpunkte: {len(r['win_input_vals'])} x {len(r['width_vals'])}\n"
        f"- sigma_atom = {r.get('sigma_atom', float('nan'))*1e9:.1f} nm "
        f"(atom_temperature={r.get('atom_temperature', float('nan'))*1e6:.2f} µK, "
        f"trap_freq_r={r.get('trap_freq_r', float('nan'))*1e-3:.2f} kHz)\n"
        f"- atom_offset_x={r.get('atom_offset_x', 0.0)*1e6:+.3f} µm, "
        f"atom_offset_y={r.get('atom_offset_y', 0.0)*1e6:+.3f} µm\n"
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Region-Dokument geschrieben: {output_path}")


def main(pkl_datei=None, alpha=None, combo_lambda=None, combo_percentile=None,
         win_axis=None, draw_best_point=None, legend_fontsize=None,
         ask_before_save=None, save=None, show=None):
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

    resolved_name = _resolve_pkl(pkl_datei)
    raw = load_combined_scan_results(resolved_name)
    results = recombine_from_grids(raw, alpha=alpha, combo_lambda=combo_lambda,
                                    combo_percentile=combo_percentile)

    prefix = _output_prefix(results)

    confirm_overwrite = None if ask_before_save else (lambda existing_path: True)

    plotter = CombinedFixedScanPlotter(results, out_dir=FIT_PLOTS_DIR, confirm_overwrite=confirm_overwrite)
    # CombinedFixedScanPlotter selbst speichert als PNG (fuer die interaktive
    # GUI, siehe combined_winwidthscan_startdialog.py) - fuer die
    # LaTeX-taugliche PDF-Ausgabe (Konvention der anderen Fit-Skripte, Vektor-
    # PDF statt PNG) wird hier direkt mit dem Matplotlib-Objekt gearbeitet,
    # nutzt aber dieselben Hilfsmethoden (_win_axis_values, _draw_region_and_mark).
    with plt.rc_context({"legend.fontsize": legend_fontsize}):
        cmp_path = _save_metric_comparison_pdf(plotter, prefix, draw_best_point, win_axis, show, save)
        region_path = _save_region_pdf(plotter, prefix, draw_best_point, win_axis, show, save)

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
                           width_MHz=b['width'] * 1e-6, combined_score=b['combined_score'])

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
        plots=dict(metric_comparison=cmp_path, region=region_path),
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

    fig.suptitle("Fixed-amplitude scan: hard vs. atom-weighted metrics")

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
    title = "Combined region (hard + atom-weighted)"
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
