"""
fit_uniformity_valley_overview.py
==================================
NEU (User-Wunsch): "Ich möchte in run_all_fits die Möglichkeit haben, alle
Datensätze entlang der minimalen Uniformity zu betrachten. Also über Waist
oder Width aufgetragen die Werte der minimalen Uniformity und den
dazugehörigen Crosstalk sowie r_x und r_y bei diesen Werten."

KORREKTUR 1 (2026-08-25, direkter Nachtrag): "nicht ganz, Für den Waist gibt
es in jeder Spalte einen Wert, der für das Minimum aus GEWICHTETEM CROSSTALK
UND UNIFORMITY minimal war - nicht nur Uniformity. Entlang diesem bitte den
Plot." Daraufhin wurde die Minimierung auf die KOMBINIERTE Metrik
`combined = alpha*Uniformity_w + (1-alpha)*Crosstalk_w` umgestellt (dieselbe
Zielgroesse wie `fit_central_amplitudes._find_best_point()`).

Ausserdem (Korrektur 1, Teil 2): "Gerne im Plot mehr Achsen einbauen, damit
Crosstalk und Uniformity nicht relativ zueinander so komisch skaliert
werden muessen" - Uniformity_w und Crosstalk_w liegen typischerweise in
STARK unterschiedlichen Groessenordnungen (Uniformity oft nur 0.01-0.1%,
Crosstalk mehrere Prozent) - auf einer gemeinsamen Prozent-Achse wuerde die
Uniformity-Kurve dadurch praktisch unsichtbar flach erscheinen. Deshalb
bekommt JEDE der vier Groessen (Uniformity, Crosstalk, r_x, r_y) ihre EIGENE
y-Achse (versetzte Spines, klassisches Matplotlib-Mehrfachachsen-Muster)
statt sich zwei Achsen zu teilen - das bleibt unabhaengig von der
gewaehlten Minimierungs-Metrik so.

KORREKTUR 2 (2026-08-25, direkter Nachtrag): "gib mir lieber den schnitt
entlang der minimalen Uniformity über den Waist bitte" - der User moechte
den Schnitt DOCH lieber wieder entlang des reinen Uniformity_w-Minimums
sehen (nicht der kombinierten Metrik aus Korrektur 1). Da beide Sichtweisen
schon einmal explizit gewuenscht wurden, wird die Minimierungs-Metrik jetzt
KONFIGURIERBAR gemacht (`MINIMIZE_METRIC`, analog zu `VALLEY_AXIS`) statt
starr auf eine der beiden Varianten festgelegt - Default jetzt wieder
"uniformity" (reines Uniformity_w-Minimum je Spalte/Zeile), "combined"
bleibt als Option verfuegbar, falls spaeter doch wieder die kombinierte
Metrik gewuenscht wird.

Arbeitet auf DERSELBEN Datei wie fit_central_amplitudes.py (der amplituden-
optimierte Scan, scan_amp_data_weighted_...pkl) - nur dort gibt es sowohl
Uniformity/Crosstalk ALS AUCH r_x_grid/r_y_grid in einer einzigen Datei.

Idee: fuer jede Spalte (fester waist) bzw. jede Zeile (fester width) des
Scan-Gitters wird der Gitterpunkt gesucht, an dem die gewaehlte Metrik
(Uniformity_w allein ODER die kombinierte Metrik alpha*Uniformity_w +
(1-alpha)*Crosstalk_w, siehe MINIMIZE_METRIC) minimal ist ("Tal"-Punkt) -
KEINE Subpixel-Interpolation wie in fit_waist_width_relation.py, sondern
bewusst das einfache Gitter-Minimum, damit Uniformity_w, Crosstalk_w, r_x
und r_y direkt am selben Gitterpunkt abgelesen werden koennen (kein
zusaetzliches Interpolieren mehrerer Groessen noetig - auf User-Wunsch
"moeglichst leicht zu bedienen und nicht zu unuebersichtlich").

Ergebnis: EIN Plot mit zwei Panels nebeneinander:
- links: die Heatmap der MINIMIERTEN Metrik (gesamtes Gitter, dieselbe
  Groesse, die minimiert wurde) mit allen gefundenen Minimalpunkten als
  rote Punkte ueberlagert - zeigt auf einen Blick, wo das Tal verlaeuft.
- rechts: der Querschnitt entlang der gewaehlten Achse (waist ODER width) -
  Uniformity_w, Crosstalk_w, r_x und r_y AN GENAU DIESEN Minimalpunkten,
  jede Groesse auf ihrer EIGENEN y-Achse.

Welche Achse als unabhaengige Variable dient (waist oder width), wird
GEFRAGT: beim direkten, interaktiven Aufruf
(`python fit_uniformity_valley_overview.py`) per Terminal-Eingabe; beim
Aufruf ueber run_all_fits.py ("ein Klick and go") wird stattdessen der von
aussen uebergebene `axis`-Parameter verwendet (dort zentral als
VALLEY_AXIS konfigurierbar) - es wird NIE interaktiv nachgefragt, wenn
`axis` explizit uebergeben wurde, damit kein Terminal-Prompt die
nicht-interaktive Pipeline blockiert (identisches Muster wie
`ask_before_save` in den beiden anderen Fit-Skripten). Die Minimierungs-
Metrik (`metric`) hat KEINE eigene interaktive Abfrage (haette den Prompt
unnoetig verkompliziert) - sie wird ausschliesslich ueber MINIMIZE_METRIC
bzw. den main()-Parameter `metric` gesteuert.

Kann eigenstaendig direkt ausgefuehrt werden ODER von run_all_fits.py als
Modul importiert und per main(...) aufgerufen werden - dupliziert dafuer
keinen Code, sondern importiert die noetigen Bausteine (Koordinatengitter,
Bester-Punkt-Logik, Speicherlogik, Plot-Stil) direkt aus
fit_central_amplitudes.py.

Nutzung:
    python fit_uniformity_valley_overview.py
(fragt interaktiv nach der Achse, sofern nicht am Prompt Enter gedrueckt
wird - dann gilt VALLEY_AXIS unten als Default; die Metrik ist unten in
MINIMIZE_METRIC einstellbar).
"""
import numpy as np
import matplotlib.pyplot as plt

import fit_central_amplitudes as fca
from fit_central_amplitudes import (
    _meshgrid_mm_mhz, _find_best_point, BEST_POINT_STYLE, FIT_PLOTS_DIR, _finish_fig,
)
from weighted_multitone_amplitude_dependence_plots import load_amp_scan_results, DEFAULT_RESULTS_DIR


# ======================================================================
# Konfiguration - hier anpassen (Defaults uebernommen von fit_central_amplitudes.py,
# da derselbe Datensatz verwendet wird)
# ======================================================================
PKL_DATEI = fca.PKL_DATEI
OUTPUT_PREFIX = fca.OUTPUT_PREFIX

# "waist" oder "width" - Default, falls beim interaktiven Aufruf einfach nur
# Enter gedrueckt wird, bzw. der Wert, den run_all_fits.py verwendet (dort
# als VALLEY_AXIS zentral einstellbar).
VALLEY_AXIS = "waist"

# "uniformity" (Default, auf User-Wunsch): pro Spalte/Zeile wird der
# Gitterpunkt mit minimaler Uniformity_w gesucht.
# "combined": stattdessen der Gitterpunkt, an dem alpha*Uniformity_w +
# (1-alpha)*Crosstalk_w minimal ist (dieselbe Zielgroesse wie
# fit_central_amplitudes._find_best_point(), hier spalten-/zeilenweise statt
# global) - war zwischenzeitlich der Default, siehe Korrektur 1/2 oben.
MINIMIZE_METRIC = "uniformity"

LEGEND_FONTSIZE = fca.LEGEND_FONTSIZE
DRAW_BEST_POINT = fca.DRAW_BEST_POINT
SHOW = fca.SHOW
SAVE = fca.SAVE
ASK_BEFORE_SAVE = fca.ASK_BEFORE_SAVE

# Breiter als PDF_FIGSIZE aus fit_central_amplitudes.py (das ist fuer ein
# 2x2-Gitter gedacht) - hier 1x2, rechtes Panel bekommt VIER eigene
# y-Achsen (Uniformity/Crosstalk/r_x/r_y, siehe plot_valley_overview()) plus
# eine Legende AUSSERHALB des Panels - deutlich mehr Platzbedarf nach rechts
# als bei den anderen Fit-Skripten (dort nur EINE zusaetzliche twinx()-Achse).
PDF_FIGSIZE_VALLEY = (19.0, 5.5)


def _make_patch_spines_invisible(ax):
    """Standard-Matplotlib-Rezept fuer eine zusaetzliche, nach aussen
    versetzte y-Achse: der automatisch erzeugte Rahmen/Patch der Twin-Axes
    wuerde sonst die dahinterliegenden Achsen/Kurven verdecken."""
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ======================================================================
# Talpunkte entlang einer Achse extrahieren
# ======================================================================
def extract_valley(results, axis=VALLEY_AXIS, metric=MINIMIZE_METRIC):
    """Fuer axis="waist": pro waist-Spalte den width-Index suchen, an dem
    die gewaehlte Metrik minimal ist ("Tal" ueber width). Fuer
    axis="width": analog umgekehrt, pro width-Zeile das waist-Minimum
    suchen.

    metric="uniformity" (Default): minimiert Uniformity_w allein.
    metric="combined": minimiert alpha*Uniformity_w + (1-alpha)*Crosstalk_w
    (dieselbe Zielgroesse wie fit_central_amplitudes._find_best_point(),
    hier spalten-/zeilenweise statt global).

    Bevorzugt die atom-gewichteten Metriken (uniformity_weighted_grid/
    eta_weighted_grid), faellt sonst auf die harten (uniformity_grid/
    crosstalk_grid) zurueck - dieselbe Praeferenz wie in
    fit_central_amplitudes._find_best_point(). Gibt None zurueck, falls
    keines der beiden Paare in der Datei vorhanden ist.

    Rueckgabe: dict mit
      axis: "waist"/"width"
      metric: "uniformity"/"combined", metric_label: Anzeigename dafuer
      x: die unabhaengige Achse (waist_mm bzw. width_MHz), aufsteigend
      waist_mm, width_MHz: tatsaechliche Koordinaten jedes Minimalpunkts
      uniformity_percent, crosstalk_percent, r_x, r_y: Werte GENAU an
        diesen Minimalpunkten (direkter Gitterwert, keine Interpolation)
      target_percent: der Wert der MINIMIERTEN Groesse (*100) an diesen
        Punkten (per Definition das Minimum je Spalte/Zeile)
      alpha: das verwendete alpha (aus der pkl-Datei, Default 1.0)
      metric_tag: "atom-weighted" oder "hard mask"
    """
    has_weighted = "uniformity_weighted_grid" in results and "eta_weighted_grid" in results
    has_hard = "uniformity_grid" in results and "crosstalk_grid" in results
    if not has_weighted and not has_hard:
        return None
    if has_weighted:
        U, C = results["uniformity_weighted_grid"], results["eta_weighted_grid"]
        metric_tag = "atom-weighted"
    else:
        U, C = results["uniformity_grid"], results["crosstalk_grid"]
        metric_tag = "hard mask"
    alpha = float(results.get("alpha", 1.0))

    if metric == "uniformity":
        target = U
        metric_label = "Uniformity"
    elif metric == "combined":
        target = alpha * U + (1 - alpha) * C
        metric_label = "combined score"
    else:
        raise ValueError("metric muss 'uniformity' oder 'combined' sein.")

    rx_grid = np.asarray(results["r_x_grid"], dtype=float)
    ry_grid = np.asarray(results["r_y_grid"], dtype=float)
    waist_mm, width_MHz = _meshgrid_mm_mhz(results)  # Form (n_width, n_waist)

    if axis == "waist":
        i_min = np.nanargmin(target, axis=0)      # je Spalte (waist) das width-Minimum
        rows, cols = i_min, np.arange(target.shape[1])
        x = waist_mm[0, :]
    elif axis == "width":
        j_min = np.nanargmin(target, axis=1)      # je Zeile (width) das waist-Minimum
        rows, cols = np.arange(target.shape[0]), j_min
        x = width_MHz[:, 0]
    else:
        raise ValueError("axis muss 'waist' oder 'width' sein.")

    return dict(
        axis=axis,
        metric=metric,
        metric_label=metric_label,
        x=x,
        waist_mm=waist_mm[rows, cols],
        width_MHz=width_MHz[rows, cols],
        uniformity_percent=U[rows, cols] * 100.0,
        crosstalk_percent=C[rows, cols] * 100.0,
        target_percent=target[rows, cols] * 100.0,
        r_x=rx_grid[rows, cols],
        r_y=ry_grid[rows, cols],
        alpha=alpha,
        metric_tag=metric_tag,
    )


# ======================================================================
# Plot
# ======================================================================
def plot_valley_overview(results, valley, waist_mm, width_MHz,
                          out_dir=FIT_PLOTS_DIR, prefix=OUTPUT_PREFIX,
                          show=SHOW, save=SAVE, best_point=None,
                          legend_fontsize=LEGEND_FONTSIZE, ask_before_save=ASK_BEFORE_SAVE):
    """1x2-Plot: links Heatmap der MINIMIERTEN Metrik (Uniformity_w allein
    ODER die kombinierte Metrik, je nach valley["metric"]) + Minimalpunkte
    (rot); rechts Querschnitt mit VIER eigenen y-Achsen (Uniformity,
    Crosstalk, r_x, r_y), damit die stark unterschiedlichen Groessenordnungen
    von Uniformity und Crosstalk sich nicht eine gemeinsame, "komisch"
    wirkende Skala teilen muessen."""
    axis = valley["axis"]
    metric = valley["metric"]
    has_weighted = "uniformity_weighted_grid" in results
    if has_weighted:
        U, C = results["uniformity_weighted_grid"], results["eta_weighted_grid"]
    else:
        U, C = results["uniformity_grid"], results["crosstalk_grid"]
    if metric == "uniformity":
        target_percent = U * 100.0
        heatmap_label = r"Uniformity ($\sigma/\mu$) (%)"
        heatmap_title = f"Uniformity ({valley['metric_tag']}) — minimum points"
    else:
        target_percent = (valley["alpha"] * U + (1 - valley["alpha"]) * C) * 100.0
        heatmap_label = r"$\alpha\,$Uniformity$\,+(1-\alpha)\,$Crosstalk (%)"
        heatmap_title = (f"Combined score ({valley['metric_tag']}, "
                          rf"$\alpha$={valley['alpha']:.2g}) — minimum points")

    extent = [waist_mm.min(), waist_mm.max(), width_MHz.min(), width_MHz.max()]
    fig, (ax_hm, ax_cut) = plt.subplots(1, 2, figsize=PDF_FIGSIZE_VALLEY)

    # --- links: Heatmap der minimierten Metrik + Minimalpunkte ---------
    im = ax_hm.imshow(target_percent, origin="lower", extent=extent, aspect="auto", cmap="viridis_r")
    ax_hm.set_title(heatmap_title)
    plt.colorbar(im, ax=ax_hm, label=heatmap_label)

    minimum_label = f"{valley['metric_label']} minimum (per {'width' if axis == 'waist' else 'waist'})"
    ax_hm.plot(valley["waist_mm"], valley["width_MHz"], marker="o", markersize=3,
               markeredgewidth=0, linestyle="none", color="red", zorder=6, label=minimum_label)
    if best_point is not None:
        ax_hm.plot(best_point["waist_mm"], best_point["width_MHz"],
                   label=best_point["label"], **BEST_POINT_STYLE)
    ax_hm.legend(loc="best", fontsize=legend_fontsize, framealpha=0.9)
    ax_hm.set_xlabel("waist (mm)")
    ax_hm.set_ylabel("width (MHz)")

    # --- rechts: Querschnitt, VIER eigene y-Achsen ----------------------
    xlabel = "waist (mm)" if axis == "waist" else "width (MHz)"
    x = valley["x"]

    ax_u = ax_cut                     # 1. Achse: Uniformity (ganz links, Standard-Position)
    ax_ct = ax_cut.twinx()            # 2. Achse: Crosstalk (rechts, Standard-Position)
    ax_rx = ax_cut.twinx()            # 3. Achse: r_x (rechts, nach aussen versetzt)
    ax_ry = ax_cut.twinx()            # 4. Achse: r_y (rechts, noch weiter aussen versetzt)

    _make_patch_spines_invisible(ax_rx)
    ax_rx.spines["right"].set_visible(True)
    ax_rx.spines["right"].set_position(("axes", 1.16))
    _make_patch_spines_invisible(ax_ry)
    ax_ry.spines["right"].set_visible(True)
    ax_ry.spines["right"].set_position(("axes", 1.32))

    lines = []
    c_u, c_ct, c_rx, c_ry = "tab:blue", "tab:orange", "tab:green", "tab:red"

    lines += ax_u.plot(x, valley["uniformity_percent"], color=c_u, marker=".", label="Uniformity (at min)")
    ax_u.set_ylabel("Uniformity (%)", color=c_u)
    ax_u.tick_params(axis="y", colors=c_u)
    ax_u.spines["left"].set_color(c_u)

    lines += ax_ct.plot(x, valley["crosstalk_percent"], color=c_ct, marker=".", label="Crosstalk (at min)")
    ax_ct.set_ylabel("Crosstalk (%)", color=c_ct)
    ax_ct.tick_params(axis="y", colors=c_ct)
    ax_ct.spines["right"].set_color(c_ct)

    lines += ax_rx.plot(x, valley["r_x"], color=c_rx, linestyle="--", label=r"r$_x$ (at min)")
    ax_rx.set_ylabel(r"r$_x$ (ratio)", color=c_rx)
    ax_rx.tick_params(axis="y", colors=c_rx)
    ax_rx.spines["right"].set_color(c_rx)

    lines += ax_ry.plot(x, valley["r_y"], color=c_ry, linestyle="--", label=r"r$_y$ (at min)")
    ax_ry.set_ylabel(r"r$_y$ (ratio)", color=c_ry)
    ax_ry.tick_params(axis="y", colors=c_ry)
    ax_ry.spines["right"].set_color(c_ry)

    ax_u.set_xlabel(xlabel)

    if best_point is not None:
        best_x = best_point["waist_mm"] if axis == "waist" else best_point["width_MHz"]
        best_line = ax_u.axvline(best_x, color="black", linestyle=":", linewidth=1.2, alpha=0.8,
                                  label=best_point["label"])
        lines.append(best_line)

    labels = [ln.get_label() for ln in lines]
    ax_u.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.42, 1.0),
                fontsize=legend_fontsize, framealpha=0.9)
    ax_u.set_title(f"Cross-section along {valley['metric_label']} minimum (vs. {xlabel.split(' ')[0]})")

    plt.tight_layout()
    axis_suffix = "over_waist" if axis == "waist" else "over_width"
    _finish_fig(fig, f"{prefix}_valley_{axis_suffix}.pdf", out_dir, show, save, ask_before_save=ask_before_save)


# ======================================================================
# main
# ======================================================================
def main(pkl_datei=None, output_prefix=None, axis=None, metric=None, draw_best_point=None,
         legend_fontsize=None, ask_before_save=None, save=None, show=None):
    """Alle Parameter optional - None faellt auf die Modul-Konfiguration
    oben zurueck. `axis` ist der einzige Parameter mit Sonderverhalten: wird
    er nicht uebergeben (None), wird interaktiv gefragt (nur sinnvoll bei
    direktem, manuellem Aufruf) - run_all_fits.py uebergibt ihn IMMER
    explizit (VALLEY_AXIS), damit hier nie nachgefragt wird. `metric`
    ("uniformity"/"combined") hat keine interaktive Abfrage, faellt bei
    None einfach auf MINIMIZE_METRIC zurueck."""
    pkl_datei = PKL_DATEI if pkl_datei is None else pkl_datei
    output_prefix = OUTPUT_PREFIX if output_prefix is None else output_prefix
    metric = MINIMIZE_METRIC if metric is None else metric
    draw_best_point = DRAW_BEST_POINT if draw_best_point is None else draw_best_point
    legend_fontsize = LEGEND_FONTSIZE if legend_fontsize is None else legend_fontsize
    ask_before_save = ASK_BEFORE_SAVE if ask_before_save is None else ask_before_save
    save = SAVE if save is None else save
    show = SHOW if show is None else show

    if axis is None:
        try:
            antwort = input(f"{('Uniformity' if metric == 'uniformity' else 'Kombiniertes Minimum')}"
                             f"-Tal ueber welche Achse auftragen? [w]aist / [b]reite(width) "
                             f"(Enter = '{VALLEY_AXIS}'): ").strip().lower()
        except EOFError:
            antwort = ""
        if antwort in ("w", "waist"):
            axis = "waist"
        elif antwort in ("b", "width", "breite"):
            axis = "width"
        else:
            axis = VALLEY_AXIS
    if axis not in ("waist", "width"):
        raise ValueError("axis muss 'waist' oder 'width' sein.")

    print(f"Lade '{pkl_datei}' ...")
    try:
        results = load_amp_scan_results(pkl_datei)
    except FileNotFoundError:
        vorhandene = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob("scan_amp_data_weighted_*.pkl"))
        print(f"'{pkl_datei}' wurde weder im aktuellen Ordner noch in '{DEFAULT_RESULTS_DIR}' gefunden.")
        if vorhandene:
            print("Vorhandene Dateien:")
            for name in vorhandene:
                print(f"  - {name}")
        return None

    waist_mm, width_MHz = _meshgrid_mm_mhz(results)
    valley = extract_valley(results, axis=axis, metric=metric)
    if valley is None:
        print("   Weder gewichtete noch harte Uniformity/Crosstalk-Grids in dieser Datei "
              "gefunden - Valley-Overview wird uebersprungen.")
        return None
    print(f"{valley['metric_label']}-Minimum entlang '{axis}' extrahiert ({valley['metric_tag']}), "
          f"{len(valley['x'])} Punkte, Minimum in "
          f"[{valley['target_percent'].min():.3f}%, {valley['target_percent'].max():.3f}%].")

    best_point = _find_best_point(results) if draw_best_point else None

    plot_valley_overview(results, valley, waist_mm, width_MHz, prefix=output_prefix,
                          show=show, save=save, best_point=best_point,
                          legend_fontsize=legend_fontsize, ask_before_save=ask_before_save)

    print("\nFertig.")
    return dict(valley=valley, best_point=best_point, axis=axis, metric=metric)


if __name__ == "__main__":
    main()
