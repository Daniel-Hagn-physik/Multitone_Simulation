"""
lib/report.py - Plots (Vektor-PDF) und Markdown-Berichte.

Wird von run_plots.py benutzt, und am Ende von run_penalty_scan.py /
run_hard_check.py fuer die automatische Erst-Auswertung.

Beide Datensatz-Arten teilen sich denselben Satz Standard-Plots
(Metrik-Vergleich hart vs. gewichtet, Score-/Region-Karte, Amplituden-
Uebersicht); der Hard-Check bekommt zusaetzlich zwei eigene Plots
(Uebereinstimmungs-Karte und Streudiagramm), fuer die es beim
Penalty-Scan keine Entsprechung gibt.

Dateinamens-Praefix (Konvention des Projekts):

    PenaltyRegion_N{Nx}x{Ny}_{n_win}x{n_width}pts_{Airy|Gauss}_{Datum}
    HardCheck_N{Nx}x{Ny}_{n_win}x{n_width}pts_{Airy|Gauss}_{Datum}

gefolgt von _metric_comparison.pdf / _region.pdf / _agreement.pdf /
_score_scatter.pdf bzw. _Report.md.
"""

from datetime import date

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import paths
from .combine import dataset_kind

from weighted_multitone_amplitude_dependence_plots import (  # noqa: E402
    AmplitudeScanPlotter, resolve_save_path, win_input_to_win,
)


BEST_POINT_STYLE = dict(marker='*', color='red', markersize=16,
                        markeredgecolor='white', markeredgewidth=1.2, linestyle='none')
REGION_STYLE = dict(edgecolor='red', facecolor='none', linewidth=1.8, linestyle='--')


# ======================================================================
# Achsen-Hilfen (identische Konvention wie AmplitudeScanPlotter)
# ======================================================================
def win_axis_values(results, win_axis):
    """(x_werte, achsenbeschriftung, umgedreht?) fuer die gewuenschte
    Waist-Konvention. 'after_lens' ist monoton fallend in win_input,
    daher wird dort ggf. umgedreht."""
    win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
    if win_axis == "before_lens":
        return win_input_vals * 1e3, r"Input waist $\omega_{\mathrm{in}}$ (before lenses, mm)", False
    if win_axis == "after_lens":
        x = np.array([win_input_to_win(w, results['f1'], results['f2'],
                                       results['lambda_opt'], results['fLO'])
                      for w in win_input_vals]) * 1e6
        label = r"Waist at focus $\omega'$ (after lenses, µm)"
        if len(x) > 1 and x[0] > x[-1]:
            return x[::-1], label, True
        return x, label, False
    raise ValueError(f"win_axis muss 'before_lens' oder 'after_lens' sein, nicht {win_axis!r}.")


def _x_of_index(results, j, win_axis):
    x_vals, _, reversed_ = win_axis_values(results, win_axis)
    n = len(x_vals)
    return x_vals[n - 1 - j] if reversed_ else x_vals[j]


def draw_region_and_mark(ax, results, win_axis, draw_region=True, draw_best=True):
    """Zeichnet das Region-Rechteck und den besten Punkt in eine
    vorhandene Heatmap-Achse."""
    width_vals = np.asarray(results['width_vals'], dtype=float)
    region = results.get('region') or {}
    best = results.get('best') or {}

    if draw_region and region.get('col_bounds') is not None:
        c0, c1 = region['col_bounds']
        r0, r1 = region['row_bounds']
        x0 = _x_of_index(results, c0, win_axis)
        x1 = _x_of_index(results, c1, win_axis)
        y0 = width_vals[r0] * 1e-6
        y1 = width_vals[r1] * 1e-6
        ax.add_patch(Rectangle((min(x0, x1), min(y0, y1)),
                               abs(x1 - x0), abs(y1 - y0),
                               label="region (largest rectangle)", **REGION_STYLE))

    if draw_best and best.get('win_input') is not None:
        win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
        j = int(np.argmin(np.abs(win_input_vals - best['win_input'])))
        x = _x_of_index(results, j, win_axis)
        ax.plot([x], [best['width'] * 1e-6], label="best point", **BEST_POINT_STYLE)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", framealpha=0.85)


def output_prefix(results):
    n_win = len(np.asarray(results['win_input_vals']))
    n_width = len(np.asarray(results['width_vals']))
    tag = paths.profile_tag_of(results.get('profile'))
    kind = dataset_kind(results)
    stem = "HardCheck" if kind == "hard_check" else "PenaltyRegion"
    return (f"{stem}_N{results['N_x']}x{results['N_y']}_{n_win}x{n_width}pts_{tag}_"
            f"{date.today().isoformat()}")


def _finish(fig, out_dir, filename, save, show, confirm_overwrite):
    out_path = None
    if save:
        out_path = resolve_save_path(out_dir, filename, confirm_overwrite=confirm_overwrite)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Plot gespeichert: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# ======================================================================
# Standard-Plots (beide Datensatz-Arten)
# ======================================================================
def plot_metric_comparison(results, prefix, out_dir=None, win_axis="before_lens",
                           draw_best_point=True, save=True, show=False,
                           confirm_overwrite=None):
    """2x2: harte vs. atom-gewichtete Uniformity und Crosstalk."""
    out_dir = paths.FIT_PLOTS_DIR if out_dir is None else out_dir
    width_vals = np.asarray(results['width_vals'], dtype=float)
    x_vals, x_label, reversed_ = win_axis_values(results, win_axis)

    kind = dataset_kind(results)
    if kind == "hard_check":
        suptitle = ("Hard check: metrics of the weighted scan vs. hard metrics "
                    "recomputed at the SAME amplitudes")
    else:
        suptitle = ("Penalty scan: hard vs. atom-weighted metrics "
                    "(one joint (r_x, r_y) optimum per point)")

    panels = [
        ("uniformity_grid", r"Uniformity ($\sigma/\mu$) (%)", "Uniformity (hard, global mask)", "viridis_r"),
        ("uniformity_weighted_grid", r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)",
         "Uniformity$_w$ (atom-weighted, local)", "viridis_r"),
        ("crosstalk_grid", r"Crosstalk ($\eta$) (%)", "Crosstalk (hard, global mask)", "Oranges"),
        ("eta_weighted_grid", r"Crosstalk$_w$ ($\eta_w$) (%)",
         "Crosstalk$_w$ (atom-weighted, local)", "Oranges"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0), constrained_layout=True)
    for ax, (grid_key, cbar_label, title, cmap) in zip(axes.flat, panels):
        Z = np.asarray(results[grid_key], dtype=float)
        Z_plot = (Z[:, ::-1] if reversed_ else Z) * 100.0
        im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel("width (MHz)")
        ax.set_title(title)
        if draw_best_point:
            draw_region_and_mark(ax, results, win_axis)
    fig.suptitle(suptitle)
    return _finish(fig, out_dir, f"{prefix}_metric_comparison.pdf", save, show, confirm_overwrite)


def plot_region(results, prefix, out_dir=None, win_axis="before_lens",
                draw_best_point=True, save=True, show=False, confirm_overwrite=None):
    """Score-Heatmap mit Region-Rechteck und bestem Punkt."""
    out_dir = paths.FIT_PLOTS_DIR if out_dir is None else out_dir
    width_vals = np.asarray(results['width_vals'], dtype=float)
    x_vals, x_label, reversed_ = win_axis_values(results, win_axis)
    Z = np.asarray(results['combined_score'], dtype=float)
    Z_plot = Z[:, ::-1] if reversed_ else Z

    kind = dataset_kind(results)
    score_label = ("consistency score (normalized, smaller = better)" if kind == "hard_check"
                   else "combined score (normalized, smaller = better)")
    title = ("Hard check: consistency region (hard vs. weighted agree)" if kind == "hard_check"
             else "Penalty scan: combined region (hard + atom-weighted)")

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap="magma_r")
    fig.colorbar(im, ax=ax, label=score_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel("width (MHz)")

    region = results.get('region') or {}
    pct = results.get('combo_percentile')
    if pct is not None and region.get('n_points_region') is not None:
        title += (f"\nbest {pct:.0f}% of score, largest rectangle: "
                  f"{region['n_points_region']}/{region['n_points_total']} points in range")
    ax.set_title(title)

    if draw_best_point:
        draw_region_and_mark(ax, results, win_axis)
    return _finish(fig, out_dir, f"{prefix}_region.pdf", save, show, confirm_overwrite)


def plot_amplitudes(results, prefix, out_dir=None, save=True, show=False,
                    confirm_overwrite=None):
    """Die 6-Panel-Uebersicht und die Schnitte des vorhandenen
    AmplitudeScanPlotter (unveraendertes Modul aus Weighted_Optimization,
    hier nur in Fit_Plots umgeleitet)."""
    out_dir = paths.FIT_PLOTS_DIR if out_dir is None else out_dir
    plotter = AmplitudeScanPlotter(results, out_dir=out_dir, confirm_overwrite=confirm_overwrite)
    overview = plotter.plot_scan2d_combined(show=show, save=save)
    cuts = plotter.plot_dependence_cuts(show=show, save=save)
    return dict(overview=overview, dependence_cuts=cuts)


# ======================================================================
# Zusatz-Plots nur fuer den Hard-Check
# ======================================================================
AGREEMENT_COLORS = ["#d9d9d9", "#4c78a8", "#f58518", "#54a24b"]
AGREEMENT_LABELS = ["neither good", "only weighted good", "only hard good", "both good"]


def plot_agreement_map(results, prefix, out_dir=None, win_axis="before_lens",
                       save=True, show=False, confirm_overwrite=None):
    """Karte der vier Kategorien: wo sind gewichtet und hart einig?"""
    out_dir = paths.FIT_PLOTS_DIR if out_dir is None else out_dir
    c = results.get('consistency') or {}
    agreement = np.asarray(c.get('agreement_map'), dtype=float)
    width_vals = np.asarray(results['width_vals'], dtype=float)
    x_vals, x_label, reversed_ = win_axis_values(results, win_axis)
    Z = agreement[:, ::-1] if reversed_ else agreement

    cmap = matplotlib.colors.ListedColormap(AGREEMENT_COLORS)
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
    im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z, shading="auto", cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(AGREEMENT_LABELS)
    ax.set_xlabel(x_label)
    ax.set_ylabel("width (MHz)")
    pct = c.get('good_percentile')
    frac = c.get('fraction_weighted_good_also_hard_good')
    title = f"Agreement map (top {pct:.0f}% by each objective)" if pct is not None else "Agreement map"
    if frac is not None:
        title += f"\n{frac * 100:.1f}% of weighted-good points are also hard-good"
    ax.set_title(title)
    return _finish(fig, out_dir, f"{prefix}_agreement.pdf", save, show, confirm_overwrite)


def plot_score_scatter(results, prefix, out_dir=None, save=True, show=False,
                       confirm_overwrite=None):
    """Streudiagramm: gewichteter Score vs. nachgerechneter harter Score."""
    out_dir = paths.FIT_PLOTS_DIR if out_dir is None else out_dir
    c = results.get('consistency') or {}
    sw = np.asarray(c.get('score_weighted'), dtype=float).ravel()
    sh = np.asarray(c.get('score_hard'), dtype=float).ravel()
    good_w = np.asarray(c.get('good_weighted_mask'), dtype=bool).ravel()
    good_h = np.asarray(c.get('good_hard_mask'), dtype=bool).ravel()
    ok = np.isfinite(sw) & np.isfinite(sh)

    # Figur bewusst etwas groesser und mit einzeiligem Titel: ein
    # zweizeiliger Titel ueberlappte hier mit dem gedrehten y-Achsenlabel.
    fig, ax = plt.subplots(figsize=(7.6, 7.0), constrained_layout=True)
    both = ok & good_w & good_h
    only_w = ok & good_w & ~good_h
    rest = ok & ~good_w
    ax.scatter(sw[rest] * 100, sh[rest] * 100, s=18, c="#bbbbbb", label="rest")
    ax.scatter(sw[only_w] * 100, sh[only_w] * 100, s=26, c="#f58518", label="weighted good, hard not")
    ax.scatter(sw[both] * 100, sh[both] * 100, s=32, c="#54a24b", label="good under both")
    ax.set_xlabel("weighted score (%)")
    ax.set_ylabel("hard score, recomputed at the same amplitudes (%)")
    ax.set_title("Weighted vs. recomputed hard score, per grid point")

    # Die interessanten (guten) Punkte draengen sich sonst alle in der
    # linken unteren Ecke, weil die schlechten Punkte um Groessenordnungen
    # hoehere Scores haben. Bei grosser Spannweite daher log-log.
    def _needs_log(values):
        pos = values[values > 0]
        return pos.size > 2 and (pos.max() / pos.min()) > 20.0

    if _needs_log(sw[ok]) and _needs_log(sh[ok]):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("weighted score (%, log scale)")
        ax.set_ylabel("hard score, recomputed at the same amplitudes (%, log scale)")

    lines = []
    if c.get('pearson_score') is not None:
        lines.append(f"Pearson r (score) = {c['pearson_score']:.4f}")
    if c.get('fraction_weighted_good_also_hard_good') is not None:
        lines.append(f"weighted-good also hard-good: "
                     f"{c['fraction_weighted_good_also_hard_good'] * 100:.1f}% "
                     f"({c.get('n_both_good')}/{c.get('n_weighted_good')})")
    if lines:
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    ax.legend(loc="lower right", framealpha=0.85)
    return _finish(fig, out_dir, f"{prefix}_score_scatter.pdf", save, show, confirm_overwrite)


# ======================================================================
# Markdown-Berichte
# ======================================================================
def _waist_um(results, win_input):
    return win_input_to_win(win_input, results['f1'], results['f2'],
                            results['lambda_opt'], results['fLO']) * 1e6


def _waist_range_um(results, lo, hi):
    a, b = _waist_um(results, lo), _waist_um(results, hi)
    return min(a, b), max(a, b)


def _r_at_best(results):
    best = results.get('best') or {}
    if best.get('win_input') is None:
        return None
    j = int(np.argmin(np.abs(np.asarray(results['win_input_vals']) - best['win_input'])))
    i = int(np.argmin(np.abs(np.asarray(results['width_vals']) - best['width'])))
    return float(np.asarray(results['r_x_grid'])[i, j]), float(np.asarray(results['r_y_grid'])[i, j])


def _scan_parameter_lines(results):
    n_win = len(np.asarray(results['win_input_vals']))
    n_width = len(np.asarray(results['width_vals']))
    lines = [
        "## Scan-Parameter",
        "",
        f"- N_x={results['N_x']}, N_y={results['N_y']}, Profil={results.get('profile')}",
        f"- Gitterpunkte: {n_win} x {n_width}",
    ]
    if results.get('r_bounds') is not None:
        lines.append(f"- r_bounds = {tuple(results['r_bounds'])}")
    sigma = results.get('sigma_atom')
    if sigma is not None:
        lines.append(
            f"- sigma_atom = {float(sigma) * 1e9:.1f} nm "
            f"(atom_temperature={float(results['atom_temperature']) * 1e6:.2f} µK, "
            f"trap_freq_r={float(results['trap_freq_r']) * 1e-3:.2f} kHz)")
    return lines


def _region_lines(results, heading, intro):
    region = results.get('region') or {}
    lines = [f"## {heading}", ""]
    if region.get('win_input_min') is None:
        lines += ["Kein gueltiges Rechteck gefunden (zu wenige valide Punkte).", ""]
        return lines
    um_lo, um_hi = _waist_range_um(results, region['win_input_min'], region['win_input_max'])
    thr = region.get('threshold')
    lines += [
        intro.format(pct=results.get('combo_percentile', float('nan')),
                     n_region=region['n_points_region'], n_total=region['n_points_total'],
                     threshold=(f"{thr:.4f}" if thr is not None else "n/a")),
        "",
        f"- win_input (vor der Linse): {region['win_input_min'] * 1e3:.4f} .. "
        f"{region['win_input_max'] * 1e3:.4f} mm",
        f"- effektiver Waist (nach der Linse): {um_lo:.4f} .. {um_hi:.4f} µm",
        f"- width: {region['width_min'] * 1e-6:.4f} .. {region['width_max'] * 1e-6:.4f} MHz",
        "",
    ]
    return lines


def _best_point_lines(results, heading):
    best = results.get('best') or {}
    lines = [f"## {heading}", ""]
    if best.get('win_input') is None:
        lines += ["Kein gueltiger Punkt im Gitter.", ""]
        return lines
    r_at_best = _r_at_best(results)
    lines += [
        f"- win_input = {best['win_input'] * 1e3:.4f} mm "
        f"({_waist_um(results, best['win_input']):.4f} µm effektiver Waist)",
        f"- width = {best['width'] * 1e-6:.4f} MHz",
        f"- Uniformity_hart = {best['uniformity_hart'] * 100:.3f}%, "
        f"Crosstalk_hart = {best['crosstalk_hart'] * 100:.3f}%",
        f"- Uniformity_weighted = {best['uniformity_weighted'] * 100:.3f}%, "
        f"Crosstalk_weighted = {best['crosstalk_weighted'] * 100:.3f}%",
        f"- Uniformity_kombi = {best['uniformity_kombi']:.4f}, "
        f"Crosstalk_kombi = {best['crosstalk_kombi']:.4f} (normierte Einheiten)",
        f"- combined_score = {best['combined_score']:.4f}",
        "",
    ]
    if r_at_best is not None:
        lines += [f"- Amplituden-Verhaeltnisse an diesem Punkt: "
                  f"r_x / r_y = {r_at_best[0]:.4f} / {r_at_best[1]:.4f}", ""]
    return lines


FORMULA_BLOCK = [
    "```",
    "X_kombi        = 0.5*(X_hart_norm + X_weighted_norm)",
    "                 + combo_lambda * |X_hart_norm - X_weighted_norm|",
    "combined_score = alpha*Uniformity_kombi + (1-alpha)*Crosstalk_kombi",
    "```",
]


def write_penalty_report(results, output_path):
    """Markdown-Bericht fuer einen Penalty-Scan."""
    n_win = len(np.asarray(results['win_input_vals']))
    n_width = len(np.asarray(results['width_vals']))
    lines = [
        f"# Penalty-Scan - N{results['N_x']}x{results['N_y']}, {n_win}x{n_width} pts, "
        f"{results.get('profile')}, {date.today().isoformat()}",
        "",
        "An JEDEM (win_input, width)-Gitterpunkt wurde GENAU EINE (r_x, r_y)-Optimierung "
        "durchgefuehrt, die direkt gegen die Kombination aus hartem und atom-gewichtetem "
        "Ziel minimiert. Uniformity_hart/Crosstalk_hart UND Uniformity_weighted/"
        "Crosstalk_weighted wurden dabei am SELBEN (r_x, r_y) ausgewertet - die gefundenen "
        "Amplituden sind daher automatisch fuer BEIDE Kriterien gleichzeitig gueltig.",
        "",
        "## Zielfunktion der Optimierung (Penalty-Term)",
        "",
        "Pro Gitterpunkt minimiert, auf ROHEN (unnormierten) Metriken - eine gitterweite "
        "Normierung steht waehrend der Optimierung eines einzelnen Punktes noch nicht zur "
        "Verfuegung:",
        "",
        "```",
        "U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|",
        "C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|",
        "J       = alpha*U_kombi + (1-alpha)*C_kombi   ->  min ueber (r_x, r_y)",
        "```",
        "",
        "Der Term combo_lambda*|Differenz| ist der Penalty-Term: er bestraft Amplituden, "
        "bei denen hartes und atom-gewichtetes Kriterium auseinanderlaufen.",
        "",
        "## Kombinationsformel der Auswertung",
        "",
        "Fuer die folgende Region-/Score-Uebersicht (NICHT fuer die Amplituden-Suche selbst) "
        "wird jede der vier Rohgroessen unabhaengig ueber das Scan-Gitter Min-Max-normiert "
        "(X_norm in [0,1]). Daraus:",
        "",
    ] + FORMULA_BLOCK + [
        "",
        f"Parameter dieses Laufs: alpha = {results.get('alpha'):.3f}, "
        f"combo_lambda = {results.get('combo_lambda'):.3f}, "
        f"combo_percentile = {results.get('combo_percentile'):.1f}%.",
        "",
    ]
    lines += _region_lines(
        results, "Region",
        "Groesstes achsenparalleles Rechteck innerhalb der besten {pct:.0f}% aller "
        "Gitterpunkte (nach combined_score); {n_region}/{n_total} Gitterpunkte insgesamt "
        "im Akzeptanzbereich (Schwellwert combined_score <= {threshold}).")
    lines += _best_point_lines(results, "Bester Einzelpunkt (Minimum von combined_score)")
    lines += _scan_parameter_lines(results)
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))
    print(f"Bericht geschrieben: {output_path}")
    return output_path


def write_hard_check_report(results, output_path):
    """Markdown-Bericht fuer einen Hard-Check."""
    c = results.get('consistency') or {}
    n_win = len(np.asarray(results['win_input_vals']))
    n_width = len(np.asarray(results['width_vals']))

    def fmt_r(key):
        val = c.get(key)
        return f"{val:.4f}" if val is not None else "n/a"

    frac = c.get('fraction_weighted_good_also_hard_good')
    lines = [
        f"# Hard-Check - N{results['N_x']}x{results['N_y']}, {n_win}x{n_width} pts, "
        f"{results.get('profile')}, {date.today().isoformat()}",
        "",
        "Ausgangspunkt ist ein bereits vorhandener, amplituden-optimierter GEWICHTETER "
        "Scan. An jedem Gitterpunkt wurden win_input, width und die dort gefundenen "
        "Amplituden r_x/r_y genommen und damit GENAU EINMAL die harten Metriken "
        "ausgewertet - KEINE erneute Optimierung. Die Frage: bleiben die unter dem "
        "atom-gewichteten Ziel guten Punkte auch unter dem harten Ziel gut?",
        "",
    ]
    if results.get('source_weighted_file'):
        lines += [f"Quelldatei (gewichteter Scan): `{results['source_weighted_file']}`", ""]
    lines += [f"Aufloesung des globalen Intensitaetsgitters fuer die harte Auswertung: "
              f"n_grid = {results.get('n_grid_hard')}.", ""]

    lines += [
        "## Kernkennzahl",
        "",
        (f"**{frac * 100:.1f}%** der unter dem gewichteten Ziel guten Punkte sind auch unter "
         f"dem harten Ziel gut ({c.get('n_both_good')} von {c.get('n_weighted_good')})."
         if frac is not None else
         "Nicht bestimmbar (keine gueltigen Punkte)."),
        "",
        f"\"Gut\" heisst jeweils: unter den besten {c.get('good_percentile', float('nan')):.0f}% "
        f"aller gueltigen Punkte nach dem eigenen Score alpha*Uniformity + (1-alpha)*Crosstalk "
        f"(alpha = {c.get('alpha', float('nan')):.3f}). Beide Mengen werden unabhaengig "
        f"voneinander bestimmt.",
        "",
        "## Vierfeldertafel",
        "",
        "| | hart gut | hart nicht gut |",
        "|---|---|---|",
        f"| **gewichtet gut** | {c.get('n_both_good')} | {c.get('n_only_weighted_good')} |",
        f"| **gewichtet nicht gut** | {c.get('n_only_hard_good')} | {c.get('n_neither_good')} |",
        "",
        f"Gueltige Punkte insgesamt: {c.get('n_valid')} "
        f"(gewichtet gut: {c.get('n_weighted_good')}, hart gut: {c.get('n_hard_good')}).",
        "",
        "## Korrelationen (gewichtet vs. hart, ueber alle gueltigen Punkte)",
        "",
        f"- Pearson r (Score) = {fmt_r('pearson_score')}",
        f"- Pearson r (Uniformity) = {fmt_r('pearson_uniformity')}",
        f"- Pearson r (Crosstalk) = {fmt_r('pearson_crosstalk')}",
        "",
        "## Consistency-Score (raeumlich zusammenhaengende Zweitsicht)",
        "",
        "Dieselbe Penalty-Kombination wie beim Penalty-Scan, hier aber nicht als "
        "Optimierungsziel, sondern als Mass fuer die Uebereinstimmung: jede der vier "
        "Rohgroessen wird gitterweit Min-Max-normiert, dann",
        "",
    ] + [line.replace("combined_score", "consistency  ") for line in FORMULA_BLOCK] + [
        "",
        "(im gespeicherten Datensatz heisst diese Groesse weiterhin `combined_score` - "
        "dieselbe Formel wie beim Penalty-Scan, nur anders interpretiert.)",
        "",
        f"mit alpha = {results.get('alpha'):.3f}, combo_lambda = "
        f"{results.get('combo_lambda'):.3f}.",
        "",
    ]
    lines += _region_lines(
        results, "Validierte Region",
        "Groesstes achsenparalleles Rechteck innerhalb der besten {pct:.0f}% aller "
        "Gitterpunkte (nach Consistency-Score); {n_region}/{n_total} Gitterpunkte insgesamt "
        "im Akzeptanzbereich (Schwellwert <= {threshold}).")
    lines += _best_point_lines(results, "Bester Einzelpunkt (Minimum des Consistency-Score)")
    lines += _scan_parameter_lines(results)
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))
    print(f"Bericht geschrieben: {output_path}")
    return output_path


# ======================================================================
# Sammel-Aufruf: alles auf einmal
# ======================================================================
def make_all(results, win_axis="before_lens", draw_best_point=True,
             plot_amplitudes_overview=True, save=True, show=False,
             ask_before_save=True, legend_fontsize=9,
             plots_dir=None, results_dir=None):
    """Erzeugt alle zum Datensatz passenden Plots und den Bericht.

    Gibt ein dict mit den Pfaden zurueck. Wird sowohl von run_plots.py als
    auch am Ende der beiden Scan-Skripte aufgerufen.
    """
    plots_dir = paths.FIT_PLOTS_DIR if plots_dir is None else plots_dir
    results_dir = paths.FIT_RESULTS_DIR if results_dir is None else results_dir
    confirm_overwrite = None if ask_before_save else (lambda existing_path: True)

    prefix = output_prefix(results)
    kind = dataset_kind(results)
    out = dict(prefix=prefix, kind=kind, plots={}, report=None)

    with plt.rc_context({"legend.fontsize": legend_fontsize}):
        out['plots']['metric_comparison'] = plot_metric_comparison(
            results, prefix, out_dir=plots_dir, win_axis=win_axis,
            draw_best_point=draw_best_point, save=save, show=show,
            confirm_overwrite=confirm_overwrite)
        out['plots']['region'] = plot_region(
            results, prefix, out_dir=plots_dir, win_axis=win_axis,
            draw_best_point=draw_best_point, save=save, show=show,
            confirm_overwrite=confirm_overwrite)
        if kind == "hard_check" and results.get('consistency'):
            out['plots']['agreement'] = plot_agreement_map(
                results, prefix, out_dir=plots_dir, win_axis=win_axis,
                save=save, show=show, confirm_overwrite=confirm_overwrite)
            out['plots']['score_scatter'] = plot_score_scatter(
                results, prefix, out_dir=plots_dir, save=save, show=show,
                confirm_overwrite=confirm_overwrite)
        if plot_amplitudes_overview:
            out['plots']['amplitudes'] = plot_amplitudes(
                results, prefix, out_dir=plots_dir, save=save, show=show,
                confirm_overwrite=confirm_overwrite)

    if save:
        report_path = results_dir / f"{prefix}_Report.md"
        if kind == "hard_check":
            write_hard_check_report(results, report_path)
        else:
            write_penalty_report(results, report_path)
        out['report'] = report_path

    return out
