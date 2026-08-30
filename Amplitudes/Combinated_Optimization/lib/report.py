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
_score_scatter.pdf / _valley_{Groesse}_over_{Achse}.pdf bzw. _Report.md.

Der Talschnitt (_valley_...) ist ein Querschnitt entlang des Minimums:
einer Groesse wird gefolgt, und genau an deren Minimum pro Spalte bzw.
Zeile werden alle uebrigen Groessen abgelesen. Optional wird durch den
vorderen, geraden Teil dieses Talpfads eine Gerade gelegt (unbrauchbare
Punkte werden dabei automatisch ausgeschlossen, siehe unten) - und der
Schnitt laesst sich statt entlang des Minimums auch entlang genau dieser
Geraden legen, ueber den ganzen gescannten Bereich hinweg (_line_...).
"""

import functools
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


# Plot-Stil fuer alle PDFs dieses Moduls: Serifen + Computer-Modern-Mathtext,
# damit die Grafiken in einem LaTeX-Dokument nicht aus dem Satz fallen.
# pdf.fonttype=42 bettet TrueType statt Type-3 ein - Type-3-Schriften werden
# von vielen Journals und von pdffonts-Pruefungen abgelehnt.
LATEX_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

WIDTH_LABEL = "Width (MHz)"


def _mit_stil(func):
    """Zeichnet die Funktion im LATEX_STYLE - auch beim Einzelaufruf, nicht
    nur ueber make_all(). Bewusst ueber rc_context statt global, damit der
    Stil nicht auf andere Skripte abfaerbt."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(LATEX_STYLE):
            return func(*args, **kwargs)
    return wrapper

BEST_POINT_STYLE = dict(marker='*', color='red', markersize=16,
                        markeredgecolor='white', markeredgewidth=1.2, linestyle='none')
# Das groesste Rechteck ("Region") wird weiterhin berechnet und steht mit
# seinen Grenzen im Bericht - eingezeichnet wird es nicht mehr.


# ======================================================================
# Achsen-Hilfen (identische Konvention wie AmplitudeScanPlotter)
# ======================================================================
def win_axis_values(results, win_axis):
    """(x_werte, achsenbeschriftung, umgedreht?) fuer die gewuenschte
    Waist-Konvention. 'after_lens' ist monoton fallend in win_input,
    daher wird dort ggf. umgedreht."""
    win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
    if win_axis == "before_lens":
        return win_input_vals * 1e3, r"Input waist $\omega_{\mathrm{in}}$ (mm, before lenses)", False
    if win_axis == "after_lens":
        x = np.array([win_input_to_win(w, results['f1'], results['f2'],
                                       results['lambda_opt'], results['fLO'])
                      for w in win_input_vals]) * 1e6
        label = r"Waist at focus $\omega'$ ($\mu$m, after lenses)"
        if len(x) > 1 and x[0] > x[-1]:
            return x[::-1], label, True
        return x, label, False
    raise ValueError(f"win_axis muss 'before_lens' oder 'after_lens' sein, nicht {win_axis!r}.")


def _x_of_index(results, j, win_axis):
    x_vals, _, reversed_ = win_axis_values(results, win_axis)
    n = len(x_vals)
    return x_vals[n - 1 - j] if reversed_ else x_vals[j]


def draw_best_point_marker(ax, results, win_axis, legend=True):
    """Markiert den besten Gitterpunkt in einer vorhandenen Heatmap-Achse.
    `legend=False`, wenn der Aufrufer die Legende selbst setzt (z.B. eine
    gemeinsame fuer mehrere Panels)."""
    best = results.get('best') or {}
    if best.get('win_input') is None:
        return
    win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
    j = int(np.argmin(np.abs(win_input_vals - best['win_input'])))
    x = _x_of_index(results, j, win_axis)
    ax.plot([x], [best['width'] * 1e-6], label="best point", **BEST_POINT_STYLE)
    handles, _ = ax.get_legend_handles_labels()
    if legend and handles:
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
@_mit_stil
def plot_metric_comparison(results, prefix, out_dir=None, win_axis="before_lens",
                           draw_best_point=True, save=True, show=False,
                           confirm_overwrite=None, fit_line=None):
    """2x2: harte vs. atom-gewichtete Uniformity und Crosstalk.

    fit_line: Ergebnis von fit_valley_line() - dann wird die Gerade in alle
    vier Karten eingezeichnet (siehe draw_fit_line_on_map)."""
    out_dir = paths.FIT_PLOTS_DIR if out_dir is None else out_dir
    width_vals = np.asarray(results['width_vals'], dtype=float)
    x_vals, x_label, reversed_ = win_axis_values(results, win_axis)

    kind = dataset_kind(results)
    # Knappe Titel: der Rest steht im Bericht, nicht im Bild.
    suptitle = ("Weighted vs. recomputed hard metrics" if kind == "hard_check"
                else "Hard vs. atom-weighted metrics")

    panels = [
        ("uniformity_grid", r"Uniformity $\sigma/\mu$ (%)", "Uniformity (hard mask)", "viridis_r"),
        ("uniformity_weighted_grid", r"Uniformity$_w$ $\sigma_w/\mu_w$ (%)",
         r"Uniformity$_w$ (atom-weighted)", "viridis_r"),
        ("crosstalk_grid", r"Crosstalk $\eta$ (%)", "Crosstalk (hard mask)", "Oranges"),
        ("eta_weighted_grid", r"Crosstalk$_w$ $\eta_w$ (%)",
         r"Crosstalk$_w$ (atom-weighted)", "Oranges"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.2), constrained_layout=True)
    for ax, (grid_key, cbar_label, title, cmap) in zip(axes.flat, panels):
        Z = np.asarray(results[grid_key], dtype=float)
        Z_plot = (Z[:, ::-1] if reversed_ else Z) * 100.0
        im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(WIDTH_LABEL)
        ax.set_title(title)
        if fit_line is not None:
            draw_fit_line_on_map(ax, results, fit_line, win_axis)
        if draw_best_point:
            # Die Legende kommt EINMAL unter die ganze Figur - vier gleiche
            # Kaesten mitten in den Karten verdecken sonst genau den
            # Bereich, um den es geht.
            draw_best_point_marker(ax, results, win_axis, legend=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center",
                   ncol=min(4, len(handles)), framealpha=0.9)
    fig.suptitle(suptitle)
    return _finish(fig, out_dir, f"{prefix}_metric_comparison.pdf",
                   save, show, confirm_overwrite)


@_mit_stil
def plot_region(results, prefix, out_dir=None, win_axis="before_lens",
                draw_best_point=True, save=True, show=False, confirm_overwrite=None):
    """Score-Heatmap, auf Wunsch mit dem besten Punkt."""
    out_dir = paths.FIT_PLOTS_DIR if out_dir is None else out_dir
    width_vals = np.asarray(results['width_vals'], dtype=float)
    x_vals, x_label, reversed_ = win_axis_values(results, win_axis)
    Z = np.asarray(results['combined_score'], dtype=float)
    Z_plot = Z[:, ::-1] if reversed_ else Z

    kind = dataset_kind(results)
    score_label = ("Consistency score (normalized, lower is better)" if kind == "hard_check"
                   else "Combined score (normalized, lower is better)")
    title = "Consistency region" if kind == "hard_check" else "Combined region"

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap="magma_r")
    fig.colorbar(im, ax=ax, label=score_label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(WIDTH_LABEL)

    region = results.get('region') or {}
    pct = results.get('combo_percentile')
    if pct is not None and region.get('n_points_region') is not None:
        title += (f"\nbest {pct:.0f}% of all grid points: "
                  f"{region['n_points_region']}/{region['n_points_total']}")
    ax.set_title(title)

    if draw_best_point:
        draw_best_point_marker(ax, results, win_axis)
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


@_mit_stil
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
    ax.set_ylabel(WIDTH_LABEL)
    pct = c.get('good_percentile')
    frac = c.get('fraction_weighted_good_also_hard_good')
    title = f"Agreement map (best {pct:.0f}% per objective)" if pct is not None else "Agreement map"
    if frac is not None:
        title += f"\n{frac * 100:.1f}% of weighted-good points are hard-good too"
    ax.set_title(title)
    return _finish(fig, out_dir, f"{prefix}_agreement.pdf", save, show, confirm_overwrite)


@_mit_stil
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
    ax.set_xlabel("Weighted score (%)")
    ax.set_ylabel("Recomputed hard score (%)")
    ax.set_title("Weighted vs. recomputed hard score")

    # Die interessanten (guten) Punkte draengen sich sonst alle in der
    # linken unteren Ecke, weil die schlechten Punkte um Groessenordnungen
    # hoehere Scores haben. Bei grosser Spannweite daher log-log.
    def _needs_log(values):
        pos = values[values > 0]
        return pos.size > 2 and (pos.max() / pos.min()) > 20.0

    if _needs_log(sw[ok]) and _needs_log(sh[ok]):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Weighted score (%, log)")
        ax.set_ylabel("Recomputed hard score (%, log)")

    lines = []
    if c.get('pearson_score') is not None:
        lines.append(f"Pearson $r$ (score) = {c['pearson_score']:.4f}")
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


def write_penalty_report(results, output_path, valley_line=None, valley_axis_label=None,
                         valley_path_mode="valley"):
    """Markdown-Bericht fuer einen Penalty-Scan.

    valley_line: Ergebnis von fit_valley_line() - ist es gesetzt (auch als
    None bei fehlgeschlagenem Fit, dann als leeres dict uebergeben nicht
    noetig), bekommt der Bericht einen Abschnitt zur Talpfad-Geraden."""
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
    if valley_axis_label is not None:
        lines += _valley_line_report_lines(valley_line, valley_axis_label,
                                           path_mode=valley_path_mode)
    lines += _scan_parameter_lines(results)
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))
    print(f"Bericht geschrieben: {output_path}")
    return output_path


def write_hard_check_report(results, output_path, valley_line=None,
                            valley_axis_label=None, valley_path_mode="valley"):
    """Markdown-Bericht fuer einen Hard-Check.

    valley_line/valley_axis_label wie bei write_penalty_report()."""
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
    if valley_axis_label is not None:
        lines += _valley_line_report_lines(valley_line, valley_axis_label,
                                           path_mode=valley_path_mode)
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
             plots_dir=None, results_dir=None,
             valley_cut=False, valley_axis="waist_um", valley_follow="combined",
             valley_traces=None, valley_fit_line=False, valley_path_mode="valley",
             fit_line_on_maps=False):
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

    # Die Gerade fuer die Metrik-Karten: dieselbe wie im Talschnitt, also
    # ueber der µm-Achse und der eingestellten Fuehrungsgroesse - egal,
    # welche Waist-Achse die Karten selbst benutzen.
    karten_fit = (fit_valley_line(results, axis=VALLEY_FIT_AXIS, follow=valley_follow)
                  if fit_line_on_maps else None)
    if fit_line_on_maps and karten_fit is None:
        print("Hinweis: keine Gerade fuer die Metrik-Karten - fuer "
              f"{_follow_label(valley_follow)} bleiben zu wenige brauchbare "
              "Talpunkte uebrig.")
    out['map_fit_line'] = karten_fit

    with plt.rc_context({**LATEX_STYLE, "legend.fontsize": legend_fontsize}):
        out['plots']['metric_comparison'] = plot_metric_comparison(
            results, prefix, out_dir=plots_dir, win_axis=win_axis,
            draw_best_point=draw_best_point, save=save, show=show,
            confirm_overwrite=confirm_overwrite, fit_line=karten_fit)
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
        if valley_cut:
            # Im Geradenmodus IST die Gerade der Schnitt - sie wird dann
            # immer bestimmt, unabhaengig vom fit_line-Schalter.
            braucht_fit = ((valley_fit_line or valley_path_mode == "line")
                           and valley_fit_supported(valley_axis))
            out['plots']['valley_cut'] = plot_valley_cut(
                results, prefix, axis=valley_axis, follow=valley_follow,
                traces=valley_traces, out_dir=plots_dir, save=save, show=show,
                confirm_overwrite=confirm_overwrite, legend_fontsize=legend_fontsize,
                fit_line=valley_fit_line and valley_fit_supported(valley_axis),
                path_mode=valley_path_mode)
            if braucht_fit:
                out['valley_line'] = fit_valley_line(
                    results, axis=valley_axis, follow=valley_follow)
        if plot_amplitudes_overview:
            out['plots']['amplitudes'] = plot_amplitudes(
                results, prefix, out_dir=plots_dir, save=save, show=show,
                confirm_overwrite=confirm_overwrite)

    if save:
        report_path = results_dir / f"{prefix}_Report.md"
        zeige_gerade = (valley_cut and valley_fit_supported(valley_axis)
                        and (valley_fit_line or valley_path_mode == "line"))
        axis_label = dict(VALLEY_AXIS_CHOICES).get(valley_axis) if zeige_gerade else None
        if kind == "hard_check":
            write_hard_check_report(results, report_path,
                                    valley_line=out.get('valley_line'),
                                    valley_axis_label=axis_label,
                                    valley_path_mode=valley_path_mode)
        else:
            write_penalty_report(results, report_path,
                                 valley_line=out.get('valley_line'),
                                 valley_axis_label=axis_label,
                                 valley_path_mode=valley_path_mode)
        out['report'] = report_path

    return out


# ======================================================================
# Querschnitt entlang des Minimums ("Talschnitt")
# ======================================================================
# Idee: einer Groesse folgen (der "Fuehrungsgroesse") und pro Spalte bzw.
# Zeile den Punkt suchen, an dem sie minimal ist. Genau an diesen Punkten
# werden dann ALLE gewuenschten Groessen abgelesen - also nicht deren
# eigenes Minimum, sondern ihr Wert dort, wo die Fuehrungsgroesse am
# besten ist. Das beantwortet die Frage: "wenn ich dem Optimum von X
# folge, was machen dabei die anderen Groessen und die Amplituden?"

# Waehlbare Fuehrungsgroessen: key -> (Anzeigename, wie berechnet)
FOLLOW_CHOICES = [
    ("combined", "Kombiniert mit Penalty (combined_score)"),
    ("uniformity_weighted", "Uniformity, atom-gewichtet"),
    ("crosstalk_weighted", "Crosstalk, atom-gewichtet"),
    ("uniformity_hard", "Uniformity, hart (globale Maske)"),
    ("crosstalk_hard", "Crosstalk, hart (globale Maske)"),
    ("score_weighted", "alpha*Uniformity + (1-alpha)*Crosstalk, gewichtet"),
    ("score_hard", "alpha*Uniformity + (1-alpha)*Crosstalk, hart"),
]

# Waehlbare Kurven im Querschnitt: key -> (Label, Einheit, Farbe, in % ?)
TRACE_SPECS = {
    "uniformity_weighted": (r"Uniformity$_w$", "%", "#4c78a8", True),
    "crosstalk_weighted": (r"Crosstalk$_w$", "%", "#f58518", True),
    "uniformity_hard": ("Uniformity (hard)", "%", "#54a24b", True),
    "crosstalk_hard": ("Crosstalk (hard)", "%", "#b279a2", True),
    "combined": ("Combined score", "norm.", "#333333", False),
    "r_x": (r"$r_x$", "", "#e45756", False),
    "r_y": (r"$r_y$", "", "#72b7b2", False),
}

# Reihenfolge der y-Achsen im Querschnitt (nur die angehakten erscheinen)
TRACE_ORDER = ["uniformity_weighted", "crosstalk_weighted", "uniformity_hard",
               "crosstalk_hard", "combined", "r_x", "r_y"]

VALLEY_AXIS_CHOICES = [
    ("waist_um", "Waist nach der Linse (µm)"),
    ("waist_mm", "win_input vor der Linse (mm)"),
    ("width", "width (MHz)"),
]

# Schnittachse -> Waist-Konvention der x-Achse/Heatmap
_VALLEY_AXIS_TO_WIN_AXIS = {"waist_um": "after_lens", "waist_mm": "before_lens",
                            "width": "after_lens"}


def _grid_for(results, key, alpha=None):
    """Das Gitter zu einem Fuehrungs-/Kurven-Schluessel. None, wenn der
    Datensatz die noetigen Groessen nicht enthaelt."""
    alpha = float(results.get("alpha", 0.7)) if alpha is None else alpha
    U_h = results.get("uniformity_grid")
    C_h = results.get("crosstalk_grid")
    U_w = results.get("uniformity_weighted_grid")
    C_w = results.get("eta_weighted_grid")
    table = {
        "uniformity_hard": U_h,
        "crosstalk_hard": C_h,
        "uniformity_weighted": U_w,
        "crosstalk_weighted": C_w,
        "combined": results.get("combined_score"),
        "r_x": results.get("r_x_grid"),
        "r_y": results.get("r_y_grid"),
    }
    if key == "score_hard":
        if U_h is None or C_h is None:
            return None
        return alpha * np.asarray(U_h, float) + (1 - alpha) * np.asarray(C_h, float)
    if key == "score_weighted":
        if U_w is None or C_w is None:
            return None
        return alpha * np.asarray(U_w, float) + (1 - alpha) * np.asarray(C_w, float)
    grid = table.get(key)
    return None if grid is None else np.asarray(grid, dtype=float)


def available_follow_keys(results):
    """Welche Fuehrungsgroessen der Datensatz hergibt."""
    return [key for key, _label in FOLLOW_CHOICES if _grid_for(results, key) is not None]


def available_trace_keys(results):
    """Welche Kurven der Datensatz hergibt."""
    return [key for key in TRACE_ORDER if _grid_for(results, key) is not None]


def extract_valley(results, axis="waist_um", follow="combined"):
    """Sucht den Talpfad der Fuehrungsgroesse und liest dort alle Groessen ab.

    axis="waist_um"/"waist_mm": pro waist-Spalte den width-Index mit dem
        kleinsten Wert der Fuehrungsgroesse.
    axis="width": pro width-Zeile den waist-Index.

    Gibt ein dict zurueck mit x (die Schnittachse, aufsteigend sortiert),
    x_label, den Koordinaten jedes Talpunkts (waist_um/waist_mm/width_MHz),
    den Gitterindizes (rows/cols) und values[key] fuer jede verfuegbare
    Groesse - jeweils GENAU am Talpunkt abgelesen, ohne Interpolation.
    """
    if axis not in _VALLEY_AXIS_TO_WIN_AXIS:
        raise ValueError(f"axis muss eine von {list(_VALLEY_AXIS_TO_WIN_AXIS)} sein, nicht {axis!r}.")
    target = _grid_for(results, follow)
    if target is None:
        raise ValueError(f"Der Datensatz enthaelt die Fuehrungsgroesse {follow!r} nicht.")

    win_input_vals = np.asarray(results["win_input_vals"], dtype=float)
    width_vals = np.asarray(results["width_vals"], dtype=float)
    waist_um = np.array([win_input_to_win(w, results["f1"], results["f2"],
                                          results["lambda_opt"], results["fLO"])
                         for w in win_input_vals]) * 1e6

    finite = np.isfinite(target)
    safe = np.where(finite, target, np.inf)

    if axis == "width":
        gueltig = np.flatnonzero(finite.any(axis=1))       # Zeilen mit Daten
        rows = gueltig
        cols = np.argmin(safe[gueltig, :], axis=1)
        x = width_vals[rows] * 1e-6          # width_vals stehen in Hz
        x_label = WIDTH_LABEL
    else:
        gueltig = np.flatnonzero(finite.any(axis=0))       # Spalten mit Daten
        cols = gueltig
        rows = np.argmin(safe[:, gueltig], axis=0)
        x = waist_um[cols] if axis == "waist_um" else win_input_vals[cols] * 1e3
        x_label = ("Waist at focus $\\omega'$ ($\\mu$m, after lenses)" if axis == "waist_um"
                   else "Input waist $\\omega_{\\mathrm{in}}$ (mm, before lenses)")

    reihenfolge = np.argsort(x)          # x aufsteigend, damit die Linie sauber laeuft
    rows, cols, x = rows[reihenfolge], cols[reihenfolge], x[reihenfolge]

    # Liegt das Minimum am Rand des gescannten Fensters, ist es vermutlich
    # gar kein echtes Minimum, sondern nur der Rand des Scans - das wahre
    # Optimum liegt dann ausserhalb. Solche Punkte werden markiert, damit
    # man sie im Plot nicht fuer bare Muenze nimmt.
    if axis == "width":
        am_rand = (cols == 0) | (cols == target.shape[1] - 1)
    else:
        am_rand = (rows == 0) | (rows == target.shape[0] - 1)

    values = {}
    for key in available_trace_keys(results):
        grid = _grid_for(results, key)
        werte = grid[rows, cols]
        label, unit, color, as_percent = TRACE_SPECS[key]
        values[key] = werte * 100.0 if as_percent else werte

    waist_heat = (win_input_vals[cols] * 1e3 if _VALLEY_AXIS_TO_WIN_AXIS[axis] == "before_lens"
                  else waist_um[cols])
    return dict(
        path_mode="valley",
        axis=axis, follow=follow, x=x, x_label=x_label,
        rows=rows, cols=cols, values=values, boundary=am_rand,
        # fuer die Nachrechnung der harten Metriken: der Ort jedes Punktes
        win_input=win_input_vals[cols],
        n_boundary=int(am_rand.sum()),
        extrapolated=np.zeros(len(x), dtype=bool), n_extrapolated=0,
        waist_um=waist_um[cols], waist_mm=win_input_vals[cols] * 1e3,
        width_MHz=width_vals[rows] * 1e-6,
        # Koordinaten fuer die Heatmap: x = Waist in der Konvention der Achse,
        # y = width. Beim Talpfad liegen sie auf Gitterpunkten, beim
        # Geradenschnitt dazwischen - beides wird gleich gezeichnet.
        x_heat=waist_heat, y_heat=width_vals[rows] * 1e-6,
        n_points=len(x), n_total=(target.shape[1] if axis != "width" else target.shape[0]),
        n_outside=0,
        alpha=float(results.get("alpha", 0.7)),
    )


# Fuer die Plots: knappe englische Bezeichnungen. FOLLOW_CHOICES bleibt
# deutsch, weil es die Eintraege des Dialogs sind.
FOLLOW_PLOT_LABELS = {
    "combined": "Combined score (penalty)",
    "uniformity_weighted": r"Uniformity$_w$ (atom-weighted)",
    "crosstalk_weighted": r"Crosstalk$_w$ (atom-weighted)",
    "uniformity_hard": "Uniformity (hard mask)",
    "crosstalk_hard": "Crosstalk (hard mask)",
    "score_weighted": r"$\alpha\,U_w + (1-\alpha)\,C_w$",
    "score_hard": r"$\alpha\,U + (1-\alpha)\,C$",
}


def follow_plot_label(follow):
    """Englische Kurzbezeichnung fuer Plot-Titel und Colorbar."""
    return FOLLOW_PLOT_LABELS.get(follow, follow)


def _follow_label(follow):
    for key, label in FOLLOW_CHOICES:
        if key == follow:
            return label
    return follow


# ----------------------------------------------------------------------
# Achsen-Buendelung im Querschnitt
# ----------------------------------------------------------------------
# Eine eigene y-Achse je Kurve wird ab vier Kurven unleserlich. Kurven
# mit DERSELBEN EINHEIT und derselben Groessenordnung teilen sich deshalb
# eine Achse; erst wenn sich die Wertebereiche um mehr als eine
# Groessenordnung unterscheiden, kommt eine zweite dazu (sonst kleben die
# kleinen Kurven platt am unteren Rand). r_x und r_y landen immer
# zusammen - es ist dieselbe Groesse in zwei Richtungen.
AXIS_GROUP_MAX_RATIO = 10.0
# Zweite Bedingung: eine Kurve, die sich eine Achse teilt, muss dort noch
# etwas zu sehen geben. Uniformity (hart) schwankt z.B. nur zwischen 3.11 und
# 3.29 % - auf einer Achse, die bis 5.9 % reicht, waere davon eine glatte
# Linie uebrig. Jede Kurve muss deshalb mindestens diesen Anteil der
# gemeinsamen Achsenspanne fuer sich beanspruchen.
AXIS_GROUP_MIN_SHARE = 0.10
# ... mit einer Ausnahme: eine Kurve, deren eigene Schwankung groesstenteils
# Rauschen ist, soll KEINE enge eigene Achse bekommen. Sonst fuellt das
# Zickzack die ganze Achse und sieht nach viel aus, obwohl es nichts
# bedeutet - Uniformity (hart) ist so ein Fall (Spanne 0.33 %-Punkte,
# Zickzack 0.11). Solche Kurven duerfen sich einer groesseren Achse
# anschliessen, auf der ihr Rauschen klein wird. Mass ist das Verhaeltnis
# aus Spanne und mittlerer zweiter Differenz.
AXIS_NOISE_SNR = 5.0
# Bleibt eine verrauschte Kurve trotzdem allein auf ihrer Achse, wird der
# Achsenbereich so aufgeweitet, dass das Zickzack hoechstens diesen Anteil
# der Achsenhoehe einnimmt.
AXIS_NOISE_TARGET = 0.25
AXIS_GROUP_ALWAYS_TOGETHER = ("r_x", "r_y")
MULTI_TRACE_AXIS_COLOR = "#333333"


def _wertebereich(werte):
    """(kleinster, groesster) endlicher Betrag > 0, oder None."""
    v = np.abs(np.asarray(werte, dtype=float))
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return None
    return float(v.min()), float(v.max())


def _zickzack(werte):
    """Mass fuer hochfrequentes Rauschen: mittlerer Betrag der zweiten
    Differenz. 0, wenn zu wenige Punkte."""
    v = np.asarray(werte, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(np.abs(np.diff(v, 2)))) if len(v) > 3 else 0.0


def _ist_verrauscht(werte, snr=AXIS_NOISE_SNR):
    """Ist die eigene Schwankung dieser Kurve groesstenteils Rauschen?"""
    v = np.asarray(werte, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return False
    spanne = float(v.max() - v.min())
    zack = _zickzack(v)
    return zack > 0 and spanne / zack < snr


def _passt_dazu(bereiche, kandidaten, values, max_ratio=AXIS_GROUP_MAX_RATIO,
                min_share=AXIS_GROUP_MIN_SHARE):
    """Duerfen sich diese Kurven eine y-Achse teilen? Ja, wenn sie in
    derselben Groessenordnung liegen und jede von ihnen entweder einen
    nennenswerten Teil der gemeinsamen Achsenspanne fuellt ODER ohnehin
    verrauscht ist (dann ist eine grosse Achse sogar die ehrlichere
    Darstellung)."""
    lo = min(bereiche[k][0] for k in kandidaten)
    hi = max(bereiche[k][1] for k in kandidaten)
    if lo <= 0 or hi / lo > max_ratio:
        return False
    spanne = hi - lo
    if spanne <= 0:
        return True
    for k in kandidaten:
        anteil = (bereiche[k][1] - bereiche[k][0]) / spanne
        if anteil < min_share and not _ist_verrauscht(values[k]):
            return False
    return True


def group_traces_by_axis(keys, values, max_ratio=AXIS_GROUP_MAX_RATIO):
    """Kurven auf moeglichst wenige y-Achsen verteilen.

    Zuerst nach Einheit trennen (Prozent, normiert, dimensionslos), dann
    innerhalb einer Einheit nach Groessenordnung buendeln: eine Achse
    nimmt so lange weitere Kurven auf, wie ihr gemeinsamer Wertebereich
    nicht mehr als `max_ratio` umspannt.

    Gibt eine Liste von Schluessel-Listen zurueck, in der Reihenfolge von
    TRACE_ORDER.
    """
    nach_einheit = {}
    for key in keys:
        nach_einheit.setdefault(TRACE_SPECS[key][1], []).append(key)

    gruppen = []
    for einheit, einheit_keys in nach_einheit.items():
        zusammen = [k for k in einheit_keys if k in AXIS_GROUP_ALWAYS_TOGETHER]
        einzeln = [k for k in einheit_keys if k not in AXIS_GROUP_ALWAYS_TOGETHER]
        if zusammen:
            gruppen.append(zusammen)

        bereiche = {k: _wertebereich(values[k]) for k in einzeln}
        # Kurven ohne brauchbaren Bereich (alles NaN/0) bekommen eine
        # eigene Achse, damit sie die Buendelung nicht durcheinanderbringen.
        ohne = [k for k in einzeln if bereiche[k] is None]
        mit = sorted([k for k in einzeln if bereiche[k] is not None],
                     key=lambda k: bereiche[k][0])

        aktuell = []
        for key in mit:
            if aktuell and _passt_dazu(bereiche, aktuell + [key], values,
                                       max_ratio=max_ratio):
                aktuell.append(key)
            else:
                if aktuell:
                    gruppen.append(aktuell)
                aktuell = [key]
        if aktuell:
            gruppen.append(aktuell)
        gruppen.extend([k] for k in ohne)

    rang = {k: i for i, k in enumerate(TRACE_ORDER)}
    for g in gruppen:
        g.sort(key=lambda k: rang.get(k, 99))
    gruppen.sort(key=lambda g: rang.get(g[0], 99))
    return gruppen


def _entzerre_achse(ax, werte_liste, ziel=AXIS_NOISE_TARGET):
    """Achsenbereich aufweiten, falls das Zickzack sonst die Achse
    dominiert. Es wird nichts abgeschnitten - die Kurve wird nur kleiner
    gezeichnet, damit hochfrequentes Rauschen nicht wie ein Signal
    aussieht."""
    endlich = [np.asarray(v, dtype=float)[np.isfinite(np.asarray(v, dtype=float))]
               for v in werte_liste]
    endlich = [v for v in endlich if v.size]
    if not endlich:
        return
    lo = min(float(v.min()) for v in endlich)
    hi = max(float(v.max()) for v in endlich)
    spanne = hi - lo
    zack = max(_zickzack(v) for v in endlich)
    if spanne <= 0 or zack <= 0 or zack / spanne <= ziel:
        return
    soll = zack / ziel
    mitte = 0.5 * (lo + hi)
    ax.set_ylim(mitte - soll / 2, mitte + soll / 2)


def _axis_label_for_group(gruppe):
    labels = [TRACE_SPECS[k][0] for k in gruppe]
    einheit = TRACE_SPECS[gruppe[0]][1]
    text = ", ".join(labels)
    return f"{text} ({einheit})" if einheit else text


def _break_at(werte, unbenutzt):
    """Kopie mit NaN an den nicht benutzten Stellen - so laesst
    matplotlib die Linie dort abreissen, statt quer durchs Bild zu
    verbinden."""
    v = np.array(werte, dtype=float)
    if unbenutzt is not None and len(unbenutzt) == len(v):
        v[np.asarray(unbenutzt, dtype=bool)] = np.nan
    return v


def _unused_mask(valley, fit):
    """Welche Punkte des Pfads gehen nicht in die Auswertung ein? Am Rand
    des Scan-Fensters gefundene Minima immer; zusaetzlich die vom Fit
    ausgeschlossenen, falls einer vorliegt."""
    unbenutzt = np.asarray(valley["boundary"], dtype=bool).copy()
    if fit is not None and fit.get("excluded_mask") is not None:
        raus = np.asarray(fit["excluded_mask"], dtype=bool)
        if len(raus) == len(unbenutzt):
            unbenutzt |= raus
    return unbenutzt


def plot_valley_cut(results, prefix, axis="waist_um", follow="combined", traces=None,
                    out_dir=None, save=True, show=False, confirm_overwrite=None,
                    legend_fontsize=9, fit_line=False, path_mode="valley"):
    """Querschnitt: links die Heatmap der Fuehrungsgroesse mit dem Pfad,
    rechts der Schnitt entlang dieses Pfads.

    traces: Liste der anzuzeigenden Schluessel (siehe TRACE_ORDER). None =
    alle, die der Datensatz hergibt. Die Fuehrungsgroesse selbst wird immer
    mitgezeichnet, auch wenn sie nicht in traces steht.

    fit_line: zusaetzlich eine Gerade durch den brauchbaren (vorderen) Teil
    des Talpfads legen und in die Heatmap zeichnen - siehe
    _fit_line_through_valley().

    path_mode: "valley" = Schnitt entlang des Minimums (Talpfad),
    "line" = Schnitt entlang genau dieser Geraden, ueber den ganzen
    gescannten Bereich (also auch extrapoliert) - siehe extract_line_cut().

    Beschriftungen sind durchgehend englisch und mathtext-basiert, damit
    die PDFs unveraendert in einen LaTeX-Satz passen.
    """
    out_dir = paths.FIT_PLOTS_DIR if out_dir is None else out_dir
    valley = extract_path(results, axis=axis, follow=follow, path_mode=path_mode)

    verfuegbar = available_trace_keys(results)
    gewaehlt = list(verfuegbar) if traces is None else [k for k in TRACE_ORDER
                                                       if k in traces and k in verfuegbar]
    follow_als_kurve = {"score_hard": "uniformity_hard",
                        "score_weighted": "uniformity_weighted"}.get(follow, follow)
    if follow_als_kurve in verfuegbar and follow_als_kurve not in gewaehlt:
        gewaehlt.insert(0, follow_als_kurve)
    if not gewaehlt:
        raise ValueError("Keine einzige darstellbare Kurve ausgewaehlt.")

    win_axis = _VALLEY_AXIS_TO_WIN_AXIS[axis]
    x_heat, x_heat_label, reversed_ = win_axis_values(results, win_axis)
    width_vals = np.asarray(results["width_vals"], dtype=float)
    target = _grid_for(results, follow)
    Z = target[:, ::-1] if reversed_ else target

    # Die Buendelung (und damit die Achsenskalierung) richtet sich nur nach
    # den BENUTZTEN Punkten. Sonst zieht ein einzelner Ausreisser am Rand des
    # Scan-Fensters die Achse auf und drueckt die eigentliche Kurve platt.
    fit_fuer_maske = (valley["fit"] if path_mode == "line"
                      else (_fit_line_through_valley(valley, _VALLEY_AXIS_TO_WIN_AXIS[axis])
                            if fit_line else None))
    unbenutzt = (np.zeros(valley["n_points"], dtype=bool) if path_mode == "line"
                 else _unused_mask(valley, fit_fuer_maske))
    werte_benutzt = {k: _break_at(v, unbenutzt) for k, v in valley["values"].items()}
    gruppen = group_traces_by_axis(gewaehlt, werte_benutzt)
    extra = max(0, len(gruppen) - 1)

    with plt.rc_context(LATEX_STYLE):
        fig, (ax_map, ax_cut) = plt.subplots(
            1, 2, figsize=(11.4 + 0.85 * extra, 5.2), constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 1.35]})

        # ---------------- links: Karte mit Pfad ----------------
        im = ax_map.pcolormesh(x_heat, width_vals * 1e-6, Z, shading="auto", cmap="magma_r")
        fig.colorbar(im, ax=ax_map, label=follow_plot_label(follow))
        ax_map.set_xlabel(x_heat_label)
        ax_map.set_ylabel(WIDTH_LABEL)

        x_pfad = np.asarray(valley["x_heat"], dtype=float)
        y_pfad = np.asarray(valley["y_heat"], dtype=float)

        if path_mode == "line":
            fit = valley["fit"]
            ax_map.set_title("Linear fit to the minimum path")
            # Der echte Talpfad bleibt blass im Bild - nur so ist zu sehen,
            # wie weit die Gerade von den tatsaechlichen Minima abweicht.
            tal = extract_valley(results, axis=axis, follow=follow)
            tal_unbenutzt = _unused_mask(tal, fit)
            ax_map.plot(_break_at(tal["x_heat"], tal_unbenutzt),
                        _break_at(tal["y_heat"], tal_unbenutzt),
                        linewidth=0.9, color="#8a8a8a", marker="o", markersize=2.8,
                        label="minimum path (reference)")
            if tal_unbenutzt.any():
                ax_map.plot(np.asarray(tal["x_heat"])[tal_unbenutzt],
                            np.asarray(tal["y_heat"])[tal_unbenutzt],
                            label=f"not used ({int(tal_unbenutzt.sum())})",
                            **UNUSED_STYLE)
            ax_map.plot(x_pfad, y_pfad, color=VALLEY_FIT_STYLE["color"], linewidth=2.0,
                        marker="o", markersize=3.2, markeredgecolor="white",
                        markeredgewidth=0.4,
                        label=(f"cut along the fitted line "
                               f"({valley['n_points']}/{valley['n_total']} pts)"))
            extrap = np.asarray(valley["extrapolated"], dtype=bool)
            if extrap.any():
                ax_map.plot(x_pfad[extrap], y_pfad[extrap],
                            label=f"extrapolated ({int(extrap.sum())})",
                            **EXTRAPOLATED_MARKER)
        else:
            ax_map.set_title("Minimum path")
            fit = fit_fuer_maske
            ax_map.plot(_break_at(x_pfad, unbenutzt), _break_at(y_pfad, unbenutzt),
                        linewidth=1.2, color="red", marker="o", markersize=3.4,
                        markeredgecolor="white", markeredgewidth=0.5,
                        label=f"minimum path ({int((~unbenutzt).sum())}/"
                              f"{valley['n_total']} pts)")
            if unbenutzt.any():
                ax_map.plot(x_pfad[unbenutzt], y_pfad[unbenutzt],
                            label=f"not used ({int(unbenutzt.sum())})", **UNUSED_STYLE)
            if fit is not None:
                draw_valley_line(ax_map, fit)
        ax_map.legend(loc="lower right", framealpha=0.9, fontsize=legend_fontsize)

        # ---------------- rechts: Querschnitt, eine y-Achse je Gruppe ----------------
        linien = []
        for position, gruppe in enumerate(gruppen):
            ax = ax_cut if position == 0 else ax_cut.twinx()
            if position > 1:
                ax.spines["right"].set_position(("outward", 52 * (position - 1)))
                ax.set_frame_on(True)
                ax.patch.set_visible(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.spines["right"].set_visible(True)
            for key in gruppe:
                label, _unit, color, _pct = TRACE_SPECS[key]
                marker = "o" if key == follow_als_kurve else None
                # Nicht benutzte Punkte tauchen hier gar nicht auf: die Linie
                # reisst dort ab. Welche das sind, zeigt die Karte links.
                linie, = ax.plot(valley["x"], werte_benutzt[key],
                                 color=color, marker=marker, markersize=2.8,
                                 linewidth=1.5, label=label)
                linien.append(linie)
            _entzerre_achse(ax, [werte_benutzt[k] for k in gruppe])
            achsen_label = _axis_label_for_group(gruppe)
            achsen_farbe = (TRACE_SPECS[gruppe[0]][2] if len(gruppe) == 1
                            else MULTI_TRACE_AXIS_COLOR)
            ax.set_ylabel(achsen_label, color=achsen_farbe)
            ax.tick_params(axis="y", colors=achsen_farbe)
            if position > 0:
                ax.spines["right"].set_color(achsen_farbe)

        ax_cut.set_xlabel(valley["x_label"])
        titel = ("Values along the fitted line" if path_mode == "line"
                 else "Values along the minimum path")
        ax_cut.set_title(titel)
        ax_cut.grid(True, alpha=0.25)
        ax_cut.legend(linien, [ln.get_label() for ln in linien],
                      loc="upper center", bbox_to_anchor=(0.5, -0.13),
                      ncol=min(4, len(linien)), framealpha=0.9, fontsize=legend_fontsize)

        achse_tag = {"waist_um": "waist_um", "waist_mm": "waist_mm", "width": "width"}[axis]
        pfad_tag = "line" if path_mode == "line" else "valley"
        dateiname = f"{prefix}_{pfad_tag}_{follow}_over_{achse_tag}.pdf"
        return _finish(fig, out_dir, dateiname, save, show, confirm_overwrite)


# ======================================================================
# Gerade durch den vorderen Teil des Talpfads
# ======================================================================
# Im vorderen Bereich laeuft der Talpfad sichtbar gerade; weiter hinten
# faellt das Minimum auf den Rand des gescannten Fensters oder springt auf
# einen zweiten, davon getrennten Nebenzweig. Eine Gerade durch ALLE
# Talpunkte waere dadurch verfaelscht - deshalb werden die unbrauchbaren
# Punkte in drei Stufen aussortiert:
#
#   1. Minima am Rand des gescannten Fensters (extract_valley() markiert
#      sie bereits als `boundary`) - dort ist der Wert kein echtes
#      Minimum, sondern der abgeschnittene Fensterrand.
#   2. Sprungerkennung: bleibt danach eine zweite, abgesetzte Punktwolke
#      uebrig, wird nur das groesste zusammenhaengende Segment behalten.
#   3. Rand-Kinks: an den beiden Enden wird iterativ abgeschnitten,
#      solange der Randpunkt deutlich neben der Ausgleichsgeraden liegt.
#
# Ausgeschlossene Punkte werden nicht verschwiegen, sondern im Plot
# markiert und im Bericht gezaehlt.
#
# Stufe 2 und 3 folgen drop_disconnected_branch() / drop_edge_kinks() aus
# Weighted_Optimization/fit_waist_width_relation.py. Bewusst hierher
# kopiert statt importiert: jenes Modul setzt beim Import global
# plt.rcParams (Serifen-Stil) und legt Ordner an - das wuerde das Aussehen
# aller uebrigen Plots dieses Ordners still veraendern. Zwei bewusste
# Abweichungen (beide unten am Ort kommentiert): die Funktionen geben eine
# Maske statt geteilter Arrays zurueck und sortieren nicht selbst, weil
# hier die Reihenfolge ENTLANG DES PFADS gilt und nicht die nach x; und die
# Sprung-Schwelle bekommt einen Boden in Hoehe des typischen Schritts.

VALLEY_FIT_STYLE = dict(color="#00c2ff", linewidth=2.2, linestyle="--")
# Punkte, die nicht in die Auswertung eingehen (Minimum am Rand des
# Scan-Fensters oder vom Fit ausgeschlossen): offen gezeichnet und NICHT
# durch die Pfadlinie verbunden. Randminima und Fit-Ausschluesse teilen sich
# bewusst EINE Markierung - fuer den Betrachter ist beides dasselbe: "dieser
# Punkt steckt nicht in der Auswertung".
UNUSED_STYLE = dict(linestyle="none", marker="o", markersize=5.5,
                    markerfacecolor="none", markeredgecolor="#222222",
                    markeredgewidth=1.2)
VALLEY_FIT_MIN_POINTS = 4
VALLEY_JUMP_FACTOR = 6.0

# Die Gerade gibt es NUR fuer den effektiven Waist in µm nach der Linse.
# Ueber win_input (mm) ist der Zusammenhang gar nicht linear - win_input und
# der effektive Waist haengen nichtlinear zusammen, eine Gerade waere dort
# ueber einen schmalen Bereich zwar hinreichend gut angepasst, aber physikalisch
# bedeutungslos und ausserhalb sofort falsch. Ueber width (MHz) waere es
# dieselbe Beziehung, nur andersherum aufgetragen - auf ausdruecklichen Wunsch
# des Nutzers bleibt sie auch dort gesperrt, damit "Gerade" im Dialog
# eindeutig an der µm-Achse haengt.
VALLEY_FIT_AXIS = "waist_um"


def valley_fit_supported(axis):
    """Laesst sich fuer diese Schnittachse eine Gerade legen? Nur fuer den
    effektiven Waist in µm - siehe VALLEY_FIT_AXIS."""
    return axis == VALLEY_FIT_AXIS


def valley_fit_axis_hint():
    """Ein Satz, der erklaert, warum es fuer die anderen Achsen keine Gerade
    gibt - fuer GUI-Tooltips und Fehlermeldungen."""
    return (f"Die Gerade gibt es nur fuer die Achse "
            f"\"{dict(VALLEY_AXIS_CHOICES)[VALLEY_FIT_AXIS]}\" - nur dort ist der "
            f"Zusammenhang zwischen width und Waist linear.")


def _branch_mask(u, jump_factor=VALLEY_JUMP_FACTOR):
    """Maske des groessten zusammenhaengenden Segments von `u` (Werte in
    Reihenfolge ENTLANG DES TALPFADS, nicht sortiert). Ein Sprung ist ein
    Schritt |du|, der deutlich groesser ist als der typische Schritt."""
    u = np.asarray(u, dtype=float)
    n = len(u)
    keep = np.ones(n, dtype=bool)
    if n < 4:
        return keep

    steps = np.abs(np.diff(u))
    median_step = float(np.median(steps))
    mad = float(np.median(np.abs(steps - median_step)))
    # Der Talpfad wird OHNE Interpolation direkt an den Gitterpunkten
    # abgelesen, seine Schritte sind also Vielfache der Gitterweite. Bei
    # einem groben Gitter sind viele Schritte exakt gleich gross, die MAD
    # wird dann 0 und eine allein auf ihr beruhende Schwelle faellt unter
    # eine einzige Gitterzelle - jeder normale Schritt gaelte als Sprung.
    # Deshalb ist der typische Schritt selbst die untere Schranke der
    # Skala: ein Sprung ist, was `jump_factor` TYPISCHE Schritte
    # ueberspringt.
    scale = max(1.4826 * mad, median_step, 1e-12)
    threshold = jump_factor * scale

    jump_idx = np.where(steps > threshold)[0]
    if len(jump_idx) == 0:
        return keep

    grenzen = [0] + [int(i) + 1 for i in jump_idx] + [n]
    segmente = [(grenzen[k], grenzen[k + 1]) for k in range(len(grenzen) - 1)]
    lo, hi = max(segmente, key=lambda seg: seg[1] - seg[0])
    keep[:] = False
    keep[lo:hi] = True
    return keep


def _edge_kink_mask(t, u, min_points=VALLEY_FIT_MIN_POINTS, min_keep_frac=0.5,
                    mad_factor=3.0):
    """Maske nach iterativem Trimmen der beiden Enden: solange der jeweilige
    Randpunkt deutlich neben der Ausgleichsgeraden des aktuellen Fensters
    liegt, faellt er weg. `t`/`u` in Reihenfolge entlang des Talpfads."""
    t = np.asarray(t, dtype=float)
    u = np.asarray(u, dtype=float)
    n = len(t)
    keep = np.zeros(n, dtype=bool)
    min_keep = max(min_points, int(np.ceil(min_keep_frac * n)))

    lo, hi = 0, n
    while hi - lo > min_keep:
        ts, us = t[lo:hi], u[lo:hi]
        m, b = np.polyfit(ts, us, 1)
        resid = us - (m * ts + b)

        innen = resid[1:-1] if len(resid) > 2 else resid
        mad = float(np.median(np.abs(innen - np.median(innen))))
        scale = 1.4826 * mad
        floor = 0.02 * (float(np.max(u)) - float(np.min(u)) + 1e-12)
        threshold = mad_factor * max(scale, floor)

        links, rechts = abs(resid[0]), abs(resid[-1])
        if max(links, rechts) <= threshold:
            break
        if links >= rechts:
            lo += 1
        else:
            hi -= 1

    keep[lo:hi] = True
    return keep


def _valley_waist(valley, win_axis):
    """Waist-Werte des Talpfads in DEN Koordinaten, in denen die Heatmap
    gezeichnet wird - nur so ist die gefittete Gerade im Bild auch
    tatsaechlich gerade."""
    if win_axis == "before_lens":
        return (np.asarray(valley["waist_mm"], dtype=float), "mm",
                r"$\omega_{\mathrm{in}}$", "win_input")
    return (np.asarray(valley["waist_um"], dtype=float), "µm", r"$\omega'$", "waist")


def _fit_line_through_valley(valley, win_axis):
    """Gerade durch den brauchbaren Teil des Talpfads.

    Gefittet wird immer ENTLANG des Pfads: bei einem Schnitt ueber den
    Waist ist der Waist die unabhaengige Groesse (width = a*waist + b), bei
    einem Schnitt ueber width ist es width (waist = a*width + b). So bleibt
    die Reihenfolge der Punkte die des Pfads, und die Gerade laesst sich in
    beiden Faellen unverzerrt in die Heatmap zeichnen.

    Gibt None zurueck, wenn zu wenige brauchbare Punkte uebrig bleiben -
    oder wenn die Schnittachse gar keinen Fit erlaubt (siehe
    VALLEY_FIT_AXIS).
    """
    if not valley_fit_supported(valley["axis"]):
        return None
    waist, waist_unit, waist_tex, waist_plain = _valley_waist(valley, win_axis)
    width = np.asarray(valley["width_MHz"], dtype=float)
    rand = np.asarray(valley["boundary"], dtype=bool)

    ueber_width = valley["axis"] == "width"
    if ueber_width:
        t, t_unit, t_tex, t_plain = width, "MHz", "width", "width"
        u, u_unit, u_tex, u_plain = waist, waist_unit, waist_tex, waist_plain
    else:
        t, t_unit, t_tex, t_plain = waist, waist_unit, waist_tex, waist_plain
        u, u_unit, u_tex, u_plain = width, "MHz", "width", "width"

    keep = ~rand                                     # Stufe 1: Randminima
    if int(keep.sum()) < VALLEY_FIT_MIN_POINTS:
        return None
    keep[keep] = _branch_mask(u[keep])                # Stufe 2: Nebenzweig
    if int(keep.sum()) < VALLEY_FIT_MIN_POINTS:
        return None
    keep[keep] = _edge_kink_mask(t[keep], u[keep])    # Stufe 3: Rand-Kinks
    if int(keep.sum()) < VALLEY_FIT_MIN_POINTS:
        return None

    ss_tot = float(np.sum((u[keep] - np.mean(u[keep])) ** 2))
    if ss_tot > 0:
        a, b = np.polyfit(t[keep], u[keep], 1)
        ss_res = float(np.sum((u[keep] - (a * t[keep] + b)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
    else:
        # Entartet: alle verbliebenen Punkte haben denselben u-Wert. Dann
        # ist die Gerade waagerecht, und ein R² gibt es nicht (die Varianz,
        # die es erklaeren soll, ist null). polyfit lieferte hier sonst eine
        # Steigung in der Groessenordnung 1e-15 - Rundungsrauschen, das im
        # Bericht wie ein Ergebnis aussaehe.
        a, b, r2 = 0.0, float(np.mean(u[keep])), float("nan")

    raus = (~keep) & (~rand)                          # in Stufe 2/3 verworfen
    # Beide Masken beziehen sich auf die Punkte des Talpfads in dessen
    # Reihenfolge - der Plot bricht die Linie damit an den nicht benutzten
    # Stellen ab, statt quer darueber hinweg zu verbinden.
    t_ends = np.array([float(np.min(t[keep])), float(np.max(t[keep]))])
    u_ends = a * t_ends + b
    # Endpunkte der Strecke in Heatmap-Koordinaten (x = Waist, y = width)
    if ueber_width:
        x_line, y_line = u_ends, t_ends
    else:
        x_line, y_line = t_ends, u_ends

    return dict(
        a=float(a), b=float(b), r2=float(r2),
        t_tex=t_tex, t_plain=t_plain, t_unit=t_unit,
        u_tex=u_tex, u_plain=u_plain, u_unit=u_unit,
        t_min=float(t_ends[0]), t_max=float(t_ends[1]),
        x_line=x_line, y_line=y_line,
        waist_used=waist[keep], width_used=width[keep],
        waist_excluded=waist[raus], width_excluded=width[raus],
        waist_boundary=waist[rand], width_boundary=width[rand],
        used_mask=keep, excluded_mask=raus,
        n_used=int(keep.sum()), n_excluded=int(raus.sum()),
        n_boundary=int(rand.sum()), n_total=int(len(waist)),
        axis=valley["axis"], follow=valley["follow"],
    )


def fit_valley_line(results, axis="waist_um", follow="combined"):
    """Talpfad bestimmen und eine Gerade durch dessen brauchbaren Teil
    legen - fuer den Aufruf von aussen (make_all, eigene Skripte).

    None, wenn die Achse keinen Fit erlaubt (nur VALLEY_FIT_AXIS) oder zu
    wenige brauchbare Talpunkte uebrig bleiben."""
    if not valley_fit_supported(axis):
        return None
    valley = extract_valley(results, axis=axis, follow=follow)
    return _fit_line_through_valley(valley, _VALLEY_AXIS_TO_WIN_AXIS[axis])


def _r2_text(r2):
    return "n/a" if not np.isfinite(r2) else f"{r2:.4f}"


def valley_line_formula(fit, latex=False):
    """Die Geradengleichung als Text, z.B.
    'width/MHz = 0.29455 * waist/µm + 0.000042'."""
    t = fit["t_tex"] if latex else fit["t_plain"]
    u = fit["u_tex"] if latex else fit["u_plain"]
    zeichen = "-" if fit["b"] < 0 else "+"
    return (f"{u}/{fit['u_unit']} = {fit['a']:.5g} · {t}/{fit['t_unit']} "
            f"{zeichen} {abs(fit['b']):.5g}")


def _valley_line_report_lines(fit, axis_label, path_mode="valley"):
    """Abschnitt fuer den Markdown-Bericht."""
    lines = ["## Talschnitt: Gerade durch den Talpfad", ""]
    if fit is None:
        lines += [
            f"Keine Gerade bestimmt: nach dem Ausschluss der unbrauchbaren Talpunkte "
            f"blieben weniger als {VALLEY_FIT_MIN_POINTS} uebrig. Das heisst in aller "
            f"Regel, dass das Minimum ueber weite Teile des Scans am Rand des "
            f"gescannten Fensters liegt - dann hilft nur ein Scan mit groesserem "
            f"Bereich.",
            "",
        ]
        return lines
    lines += [
        f"Gerade durch den Talpfad des Minimums von {_follow_label(fit['follow'])}, "
        f"aufgetragen ueber {axis_label}:",
        "",
        "```",
        valley_line_formula(fit),
        "```",
        "",
        f"- Steigung a = {fit['a']:.6g} {fit['u_unit']}/{fit['t_unit']}",
        f"- Achsenabschnitt b = {fit['b']:.6g} {fit['u_unit']}",
        f"- R² = {_r2_text(fit['r2'])}"
        + ("  (die verbliebenen Punkte liegen alle auf demselben Wert - "
           "eine Varianz, die eine Gerade erklaeren koennte, gibt es hier nicht)"
           if not np.isfinite(fit['r2']) else ""),
        f"- gefitteter Bereich: {fit['t_min']:.4f} .. {fit['t_max']:.4f} {fit['t_unit']}",
        f"- verwendete Talpunkte: {fit['n_used']} von {fit['n_total']}",
        f"- ausgeschlossen: {fit['n_boundary']} mit Minimum am Rand des gescannten "
        f"Fensters, {fit['n_excluded']} auf einem abgesetzten Nebenzweig bzw. als "
        f"Rand-Kink",
        "",
        "Ausschluss-Verfahren (dieselbe Logik wie in `fit_waist_width_relation.py`): "
        "zuerst die Randminima, dann nur das groesste zusammenhaengende Segment des "
        "Talverlaufs (Sprungerkennung ueber die Streuung der Schritte), zuletzt "
        "iteratives Trimmen der beiden Enden, solange der Randpunkt deutlich neben "
        "der Ausgleichsgeraden liegt.",
        "",
    ]
    if path_mode == "line":
        lines += [
            "**Der Querschnitt wurde entlang genau dieser Geraden gelegt**, nicht "
            "entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, also "
            "auch ausserhalb des oben genannten Fit-Bereichs (dort ist er "
            "extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die "
            "Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den beiden "
            "benachbarten Gitterzeilen linear interpoliert.",
            "",
        ]
    return lines


def line_points_for_axis(results, fit, win_axis, n=240):
    """Die Fit-Gerade als Punktfolge in den Koordinaten EINER Karte.

    Die Gerade ist in width ueber dem effektiven Waist (µm) definiert. Auf
    einer µm-Achse ist sie deshalb wirklich gerade; auf der mm-Achse
    (win_input vor der Linse) nicht, weil win_input und effektiver Waist
    nichtlinear zusammenhaengen - dort wird sie deshalb dicht abgetastet
    und als Polygonzug gezeichnet, nicht als Strecke.

    Gibt (x_innen, y_innen, x_aussen, y_aussen) zurueck: innerhalb und
    ausserhalb des Bereichs, aus dem die Gerade bestimmt wurde.
    """
    win_input_vals = np.asarray(results["win_input_vals"], dtype=float)
    win_input = np.linspace(win_input_vals.min(), win_input_vals.max(), int(n))
    waist_um = np.array([win_input_to_win(w, results["f1"], results["f2"],
                                          results["lambda_opt"], results["fLO"])
                         for w in win_input]) * 1e6
    width = fit["a"] * waist_um + fit["b"]
    x = waist_um if win_axis == "after_lens" else win_input * 1e3
    innen = (waist_um >= fit["t_min"]) & (waist_um <= fit["t_max"])
    # Auf den gescannten width-Bereich beschneiden: sonst zieht die Gerade
    # die y-Achse der Karte auf und die Heatmap schrumpft auf einen Streifen.
    width_vals = np.asarray(results["width_vals"], dtype=float) * 1e-6
    sichtbar = (width >= width_vals.min()) & (width <= width_vals.max())
    ordnung = np.argsort(x)
    x, width, innen, sichtbar = x[ordnung], width[ordnung], innen[ordnung], sichtbar[ordnung]
    return (np.where(innen & sichtbar, x, np.nan), np.where(innen & sichtbar, width, np.nan),
            np.where(~innen & sichtbar, x, np.nan), np.where(~innen & sichtbar, width, np.nan))


def draw_fit_line_on_map(ax, results, fit, win_axis):
    """Die Fit-Gerade in eine beliebige (Waist, width)-Karte zeichnen -
    durchgezogen im gefitteten Bereich, gepunktet in der Extrapolation.

    Die Fitparameter stehen bewusst NICHT in der Legende: Steigung,
    Achsenabschnitt, R2 und der gefittete Bereich sind im Bericht
    nachzulesen und wuerden den Kasten nur aufblaehen."""
    if fit is None:
        return
    x_in, y_in, x_out, y_out = line_points_for_axis(results, fit, win_axis)
    ax.plot(x_in, y_in, color=VALLEY_FIT_STYLE["color"], linewidth=2.0,
            linestyle="-", label="linear fit")
    if np.any(np.isfinite(x_out)):
        ax.plot(x_out, y_out, color=VALLEY_FIT_STYLE["color"], linewidth=1.4,
                linestyle=":", label="fit, extrapolated")


def draw_valley_line(ax, fit, legend_fontsize=None):
    """Gerade in die Heatmap zeichnen. `fit=None` zeichnet nichts. Die vom
    Fit ausgeschlossenen Punkte markiert der Aufrufer bereits ueber die
    gemeinsame \"not used\"-Markierung."""
    if fit is None:
        return
    ax.plot(fit["x_line"], fit["y_line"], label="linear fit", **VALLEY_FIT_STYLE)


# ======================================================================
# Schnitt entlang der Geraden (statt entlang des Minimums)
# ======================================================================
# Der Talpfad springt dort, wo das Minimum flach ist oder aus dem
# gescannten Fenster laeuft. Die Gerade aus _fit_line_through_valley()
# tut das nicht: sie ist ueber den GANZEN gescannten Bereich definiert,
# auch weit ausserhalb der Punkte, aus denen sie bestimmt wurde. Ein
# Schnitt entlang dieser Geraden ist deshalb glatt und zeigt, was die
# Metriken taeten, wenn man der linearen Beziehung folgte statt dem
# tatsaechlichen (teils verrauschten) Minimum.
#
# Die Gerade trifft die Gitterpunkte nicht - deshalb wird pro Spalte
# (bzw. Zeile) LINEAR zwischen den beiden benachbarten Gitterwerten
# interpoliert. Wo die Gerade das gescannte Fenster verlaesst, gibt es
# keine Daten; solche Spalten fallen aus dem Schnitt heraus und werden
# gezaehlt (`n_outside`). Punkte ausserhalb des Bereichs, aus dem die
# Gerade bestimmt wurde, sind echte EXTRAPOLATION und werden als solche
# markiert (`extrapolated`) - im Plot mit offenen Kreisen.

PATH_MODE_CHOICES = [
    ("valley", "Talpfad (Minimum der Fuehrungsgroesse)"),
    ("line", "Gerade durch den Talpfad (auch extrapoliert)"),
]

EXTRAPOLATED_MARKER = dict(linestyle="none", marker="o", markersize=6.0,
                           markerfacecolor="white", markeredgecolor="#00647f",
                           markeredgewidth=1.3)


def _interp_at(werte, koordinate, ziel):
    """Linear zwischen den beiden Nachbarn interpolieren. Gibt NaN
    zurueck, wenn `ziel` ausserhalb des Koordinatenbereichs liegt."""
    ordnung = np.argsort(koordinate)
    c = np.asarray(koordinate, dtype=float)[ordnung]
    v = np.asarray(werte, dtype=float)[ordnung]
    if not np.isfinite(ziel) or ziel < c[0] or ziel > c[-1]:
        return float("nan")
    i = int(np.clip(np.searchsorted(c, ziel), 1, len(c) - 1))
    spanne = c[i] - c[i - 1]
    t = 0.0 if spanne == 0 else (ziel - c[i - 1]) / spanne
    return float((1.0 - t) * v[i - 1] + t * v[i])


def extract_line_cut(results, axis="waist_um", follow="combined", fit=None):
    """Schnitt entlang der Fit-Geraden statt entlang des Minimums.

    Liefert dasselbe dict-Format wie extract_valley(), damit beide von
    plot_valley_cut() gleich behandelt werden - nur sind die Werte hier
    zwischen den Gitterzeilen interpoliert statt direkt abgelesen.

    fit: Ergebnis von fit_valley_line(); None = selbst bestimmen.
    """
    if axis not in _VALLEY_AXIS_TO_WIN_AXIS:
        raise ValueError(f"axis muss eine von {list(_VALLEY_AXIS_TO_WIN_AXIS)} sein, nicht {axis!r}.")
    if not valley_fit_supported(axis):
        raise ValueError(
            f"Ein Schnitt entlang der Geraden ist fuer die Achse {axis!r} nicht "
            f"vorgesehen. {valley_fit_axis_hint()}")
    win_axis = _VALLEY_AXIS_TO_WIN_AXIS[axis]
    if fit is None:
        fit = fit_valley_line(results, axis=axis, follow=follow)
    if fit is None:
        raise ValueError(
            "Fuer diesen Datensatz liess sich keine Gerade durch den Talpfad legen "
            "(zu wenige brauchbare Talpunkte) - ein Schnitt entlang der Geraden ist "
            "damit nicht moeglich. Bitte den Talpfad-Modus verwenden.")

    win_input_vals = np.asarray(results["win_input_vals"], dtype=float)
    width_vals = np.asarray(results["width_vals"], dtype=float)
    width_mhz = width_vals * 1e-6
    waist_um = np.array([win_input_to_win(w, results["f1"], results["f2"],
                                          results["lambda_opt"], results["fLO"])
                         for w in win_input_vals]) * 1e6
    waist_mm = win_input_vals * 1e3
    waist_heat = waist_mm if win_axis == "before_lens" else waist_um

    keys = available_trace_keys(results)
    gitter = {key: _grid_for(results, key) for key in keys}
    ziel_gitter = _grid_for(results, follow)

    ueber_width = axis == "width"
    x_liste, waist_liste, width_liste, extrap_liste, index_liste = [], [], [], [], []
    werte_liste = {key: [] for key in keys}
    n_aussen = 0

    # t ist die unabhaengige Groesse des Fits (siehe _fit_line_through_valley):
    # beim Schnitt ueber den Waist der Waist, beim Schnitt ueber width die width.
    laufindex = range(len(width_mhz)) if ueber_width else range(len(waist_heat))
    for k in laufindex:
        if ueber_width:
            t = float(width_mhz[k])
            u = fit["a"] * t + fit["b"]                  # Waist auf der Geraden
            if not np.isfinite(u) or u < np.min(waist_heat) or u > np.max(waist_heat):
                n_aussen += 1
                continue
            if not np.isfinite(_interp_at(ziel_gitter[k, :], waist_heat, u)):
                n_aussen += 1
                continue
            waist_hier, width_hier = u, t
            for key in keys:
                werte_liste[key].append(_interp_at(gitter[key][k, :], waist_heat, u))
        else:
            t = float(waist_heat[k])
            u = fit["a"] * t + fit["b"]                  # width auf der Geraden
            if not np.isfinite(u) or u < np.min(width_mhz) or u > np.max(width_mhz):
                n_aussen += 1
                continue
            if not np.isfinite(_interp_at(ziel_gitter[:, k], width_mhz, u)):
                n_aussen += 1
                continue
            waist_hier, width_hier = t, u
            for key in keys:
                werte_liste[key].append(_interp_at(gitter[key][:, k], width_mhz, u))

        x_liste.append(t)
        index_liste.append(k)
        waist_liste.append(waist_hier)
        width_liste.append(width_hier)
        # ausserhalb des Bereichs, aus dem die Gerade bestimmt wurde
        extrap_liste.append(t < fit["t_min"] or t > fit["t_max"])

    if not x_liste:
        raise ValueError(
            "Die Gerade verlaesst das gescannte Fenster ueberall - es bleibt kein "
            "einziger Punkt fuer den Schnitt uebrig.")

    x = np.array(x_liste, dtype=float)
    ordnung = np.argsort(x)
    x = x[ordnung]
    waist_arr = np.array(waist_liste, dtype=float)[ordnung]
    width_arr = np.array(width_liste, dtype=float)[ordnung]
    extrap = np.array(extrap_liste, dtype=bool)[ordnung]
    indizes = np.array(index_liste, dtype=int)[ordnung]
    # win_input je Schnittpunkt. Beim Schnitt ueber den Waist sitzt jeder
    # Punkt auf einer Scan-Spalte, der Wert ist also exakt; beim Schnitt
    # ueber width liegt der Waist zwischen den Spalten und win_input wird
    # mitinterpoliert.
    if ueber_width:
        win_input_out = np.array([_interp_at(win_input_vals, waist_heat, u)
                                  for u in waist_arr], dtype=float)
    else:
        win_input_out = win_input_vals[indizes]

    values = {}
    for key in keys:
        werte = np.array(werte_liste[key], dtype=float)[ordnung]
        _label, _unit, _color, as_percent = TRACE_SPECS[key]
        values[key] = werte * 100.0 if as_percent else werte

    if ueber_width:
        x_label = WIDTH_LABEL
        waist_um_out = waist_arr if win_axis == "after_lens" else np.full_like(waist_arr, np.nan)
        waist_mm_out = waist_arr if win_axis == "before_lens" else np.full_like(waist_arr, np.nan)
    else:
        x_label = ("Waist at focus $\\omega'$ ($\\mu$m, after lenses)" if axis == "waist_um"
                   else "Input waist $\\omega_{\\mathrm{in}}$ (mm, before lenses)")
        waist_um_out = waist_arr if win_axis == "after_lens" else np.full_like(waist_arr, np.nan)
        waist_mm_out = waist_arr if win_axis == "before_lens" else np.full_like(waist_arr, np.nan)

    return dict(
        path_mode="line", fit=fit,
        axis=axis, follow=follow, x=x, x_label=x_label,
        rows=None, cols=None, values=values,
        boundary=np.zeros(len(x), dtype=bool), n_boundary=0,
        win_input=win_input_out,
        extrapolated=extrap, n_extrapolated=int(extrap.sum()),
        waist_um=waist_um_out, waist_mm=waist_mm_out, width_MHz=width_arr,
        x_heat=waist_arr, y_heat=width_arr,
        n_points=len(x),
        n_total=(len(width_mhz) if ueber_width else len(waist_heat)),
        n_outside=n_aussen,
        alpha=float(results.get("alpha", 0.7)),
    )


def extract_path(results, axis="waist_um", follow="combined", path_mode="valley"):
    """Talpfad ODER Geradenschnitt - je nach `path_mode`."""
    if path_mode == "valley":
        return extract_valley(results, axis=axis, follow=follow)
    if path_mode == "line":
        return extract_line_cut(results, axis=axis, follow=follow)
    raise ValueError(f"path_mode muss 'valley' oder 'line' sein, nicht {path_mode!r}.")


def path_mode_label(path_mode):
    for key, label in PATH_MODE_CHOICES:
        if key == path_mode:
            return label
    return path_mode
