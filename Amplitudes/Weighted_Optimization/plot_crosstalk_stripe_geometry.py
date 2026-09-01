"""
Geometrie der Crosstalk-Streifen: warum sie entlang width laufen, leicht
kippen und dabei einknicken
=========================================================================

Drei Karten ueber (waist, width), englisch beschriftet, LaTeX-tauglich:

  A  Simulation - der ZAEHLER der gewichteten Crosstalk-Definition,
     <I_Nachbar>_W. Hier steckt die gesamte Streifenstruktur.
  B  Analytisches Modell mit NUR dem naechsten Nachbarspot. Ergibt perfekt
     gerade, leicht gekippte Baender - also Kippung ja, Knick nein.
  C  Dasselbe Modell mit ALLEN 8*N_x*N_y Nachbarspots. Jetzt knicken die
     Baender genauso wie in der Simulation.

Damit ist die Ursachenkette isoliert:

* Streifen laufen entlang width, weil der Waist die Ringskala setzt
  (u ~ 1/w_0, ueber den Scanbereich rund 3 Ringperioden), width dagegen fast
  nichts bewegt (Delta_u < pi ueber die ganze Achse).
* Sie KIPPEN, weil width den Abstand Atom <-> naechster Nachbarspot doch
  linear verkleinert: r_min = pitch - halbe Spannweite(width). Die
  Ringbedingung u = const heisst w_0 ~ r, also wandert dasselbe Ringminimum
  proportional zu kleinerem Waist.
* Sie KNICKEN, weil es nicht EIN Ringsystem ist, sondern eines pro
  Nachbarspot (r = 4.0 ... 8.8 um). Jedes hat in 1/w_0 seine eigene Periode
  (~ 1/r), die Ueberlagerung schwebt. Wo die Schwebung umschlaegt, biegt sich
  das Band - und stellenweise loest sich ein Minimum kurz in eine Schulter
  auf.
* Der NENNER <I_eigen>_W traegt nichts dazu bei: er ist bei jeder width
  streng monoton im Waist (per Skript geprueft, keine einzige lokale
  Extremstelle) und liefert nur einen glatten Untergrund.

Aufruf
------
Einfach ausfuehren; liegt neben `weighted_multitone_flattop_optimizer.py`.
Laufzeit ~2 min fuer die Default-Aufloesung (Panel A ist der teure Teil:
n_waist * n_width Auswertungen auf dem lokalen Sub-Grid).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ======================================================================
# Konfiguration
# ======================================================================

OPT_DIR = None                        # None = Ordner dieses Skripts

WIN_INPUT_RANGE = (0.8e-3, 1.7e-3)    # Eingangswaist vor der Linse (m)
WIDTH_RANGE = (0.2e6, 0.4e6)          # Hz
N_WAIST = 101
N_WIDTH = 51

# Sub-Grid fuer die gewichteten Metriken. 81 statt der Default-241 ist hier
# voellig ausreichend (gegen 241 geprueft: 9 uebereinstimmende Stellen) und
# rund 15x schneller.
WEIGHTED_N_GRID = 81

AMPLITUDES = None                     # None = alle Amplituden 1

# Airy-Skalenfaktor, siehe airy_scale.py. None = AIRY_SCALE_DIALOG_DEFAULT
# (1.482951). Der Dateiname bekommt bei jedem Faktor ausser 1.19 ein
# Kuerzel, damit nicht vergleichbare Abbildungen sich nicht ueberschreiben.
AIRY_SCALE_FACTOR = None

OUT_DIR = "Bilder"
OUT_NAME = "crosstalk_stripe_geometry"
OUT_NAME_TERMS = "crosstalk_terms_vs_width"   # zweite Abbildung: Zaehler/Nenner
WAIST_CUTS = (0.87e-6, 1.31e-6)               # Waists fuer die zweite Abbildung
OUT_NAME_BAND = "crosstalk_band_trajectory"   # dritte Abbildung: Bandverlauf
BAND_START_UM = None                          # welches Band verfolgt wird (Startwert am unteren
                                              # Rand, in um). None = automatisch das mittlere
                                              # Minimum der untersten width-Zeile - noetig, weil
                                              # die Baender mit dem Airy-Skalenfaktor wandern.
SAVE_PNG = True
SHOW = False

LATEX_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COL_RIDGE = "#b3402a"
COL_MUTED = "#6b6b6b"
COL_NUM = "#1f4e79"
COL_DEN = "#e08214"
COL_ETA = "#1b1b1b"

J1_FIRST_ZERO = 3.83170597

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OPT_DIR if OPT_DIR else _here)
import weighted_multitone_flattop_optimizer as wmfo  # noqa: E402
import airy_scale  # noqa: E402


def resolved_scale_factor():
    """Der tatsaechlich verwendete Airy-Skalenfaktor."""
    return (airy_scale.AIRY_SCALE_DIALOG_DEFAULT if AIRY_SCALE_FACTOR is None
            else float(AIRY_SCALE_FACTOR))


# ======================================================================
# Rechnen
# ======================================================================

def neighbour_radii(opt, width_hz):
    """Abstaende Atom (Site-Mitte) zu ALLEN Spots der 8 Nachbar-Sites."""
    cx, cy, rc = opt._compute_centers_for_width(width_hz)
    dx, dy = cx - rc, cy - rc
    return np.concatenate([
        np.sqrt((dx + ix * opt.pitch) ** 2 + (dy + iy * opt.pitch) ** 2)
        for ix in (-1, 0, 1) for iy in (-1, 0, 1) if not (ix == 0 and iy == 0)
    ])


def airy_tail(u):
    """Asymptotik der Airy-Intensitaet: (8/pi) * cos^2(u - 3pi/4) / u^3."""
    return (8.0 / np.pi) * np.cos(u - 0.75 * np.pi) ** 2 / u ** 3


def model_map(opt, waists, widths, nearest_only=False):
    """Analytisches Nachbarlicht am Atomort, ohne jede Simulation."""
    out = np.zeros((len(widths), len(waists)))
    for i, W in enumerate(widths):
        r = neighbour_radii(opt, W)
        if nearest_only:
            r = np.array([r.min()])
        for j, w in enumerate(waists):
            u = J1_FIRST_ZERO * r / (opt.airy_scale_factor * w)
            out[i, j] = np.sum(airy_tail(u))
    return out


def weighted_terms(opt, waist_m, width_hz, amps=None):
    """
    Zaehler <I_Nachbar>_W und Nenner <I_eigen>_W der gewichteten
    Crosstalk-Definition getrennt - dieselben Bausteine wie in
    _evaluate_weighted_metrics(), nur nicht sofort dividiert.
    """
    cx, cy, rc = opt._compute_centers_for_width(width_hz)
    profile_func = opt._profile_func()
    scale = opt._profile_scale(waist_m)
    if amps is None:
        amp_spots = np.ones(len(cx))
    else:
        amps = np.asarray(amps, dtype=float)
        amp_spots = np.repeat(amps[:opt.N_x], opt.N_y) * np.tile(amps[opt.N_x:], opt.N_x)

    ax, ay = rc + opt.atom_offset_x, rc + opt.atom_offset_y
    Xs, Ys = opt._build_local_weighted_grid(ax, ay)
    I_own = profile_func(Xs, Ys, cx, cy, scale, amp_spots)
    I_nb = opt._local_neighbor_intensity(Xs, Ys, cx, cy, scale, amp_spots, profile_func)
    Wt = wmfo.atom_weight_2d(Xs, Ys, ax, ay, opt.sigma_atom)
    norm = np.sum(Wt)
    return np.sum(I_nb * Wt) / norm, np.sum(I_own * Wt) / norm


def simulate_maps(opt, waists, widths, amps=None):
    num = np.zeros((len(widths), len(waists)))
    den = np.zeros_like(num)
    for i, W in enumerate(widths):
        for j, w in enumerate(waists):
            num[i, j], den[i, j] = weighted_terms(opt, w, W, amps)
        print("  width %.3f MHz fertig" % (W / 1e6))
    return num, den


def ridge_points(M, waists):
    """Lokale Minima entlang der waist-Achse, Zeile fuer Zeile."""
    xs, ys = [], []
    for i in range(M.shape[0]):
        row = M[i]
        for j in range(1, M.shape[1] - 1):
            if row[j] <= row[j - 1] and row[j] <= row[j + 1]:
                xs.append(waists[j])
                ys.append(i)
    return np.array(xs), np.array(ys, dtype=int)


def monotone_check(M):
    """Anzahl Zeilen mit lokalen Extrema entlang waist (0 = ueberall monoton)."""
    bad = 0
    for row in M:
        d = np.diff(row)
        if np.any(np.sign(d[:-1]) * np.sign(d[1:]) < 0):
            bad += 1
    return bad


# ======================================================================
# Abbildung
# ======================================================================

def make_figure(waists, widths, num, mod_one, mod_all):
    wu = waists * 1e6
    wd = widths / 1e6

    with plt.rc_context(LATEX_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True,
                                 sharex=True, sharey=True)
        panels = [
            (axes[0], num, r"(a) simulation: $\langle I_\mathrm{nb}\rangle_W$"),
            (axes[1], mod_one, "(b) model, nearest spot only"),
            (axes[2], mod_all, "(c) model, all neighbour spots"),
        ]
        for ax, M, title in panels:
            ax.pcolormesh(wu, wd, M, shading="nearest", cmap="Blues",
                          norm=LogNorm(vmin=M.min(), vmax=M.max()), rasterized=True)
            rx, ri = ridge_points(M, waists)
            ax.plot(rx * 1e6, wd[ri], ".", ms=3.5, color=COL_RIDGE)
            ax.set_title(title)
            ax.set_xlabel(r"Waist $w_0$ ($\mu$m, after lenses)")
            for sp in ax.spines.values():
                sp.set_color(COL_MUTED)
            ax.tick_params(colors=COL_MUTED)
            ax.xaxis.label.set_color("#1b1b1b")
        axes[2].annotate(r"Airy scale $k = %.4g$" % resolved_scale_factor(),
                         (0.97, 0.03), xycoords="axes fraction", ha="right",
                         color="white", fontsize=8)
        axes[0].set_ylabel("Width (MHz)")
        axes[0].yaxis.label.set_color("#1b1b1b")
        axes[0].set_xlim(wu.min(), wu.max())
        axes[0].set_ylim(wd.min(), wd.max())
        axes[2].plot([], [], ".", ms=5, color=COL_RIDGE, label="crosstalk minima")
        axes[2].legend(loc="upper right", framealpha=0.9, edgecolor="none")

        out_dir = OUT_DIR if os.path.isabs(OUT_DIR) else os.path.join(_here, OUT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        stem = OUT_NAME + airy_scale.scale_tag(resolved_scale_factor())
        pdf_path = os.path.join(out_dir, stem + ".pdf")
        fig.savefig(pdf_path)
        print("gespeichert:", pdf_path)
        if SAVE_PNG:
            png_path = os.path.join(out_dir, stem + ".png")
            fig.savefig(png_path, dpi=200)
            print("gespeichert:", png_path)
        if SHOW:
            plt.show()
        else:
            plt.close(fig)


def subpixel_minima(row, waists):
    """Lokale Minima mit parabolischer Subpixel-Interpolation im Log-Raum."""
    out = []
    idx = np.arange(len(waists))
    for j in range(1, len(row) - 1):
        if row[j] <= row[j - 1] and row[j] <= row[j + 1]:
            y0, y1, y2 = np.log(row[j - 1]), np.log(row[j]), np.log(row[j + 1])
            denom = y0 - 2 * y1 + y2
            dj = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            out.append(np.interp(j + dj, idx, waists))
    return np.array(out)


def track_band(M, waists, start_m, max_jump=0.12e-6):
    """Verfolgt EIN Minimum zeilenweise (nearest neighbour) ueber die width-Achse."""
    pos = np.full(M.shape[0], np.nan)
    cur = start_m
    for i in range(M.shape[0]):
        m = subpixel_minima(M[i], waists)
        if len(m) == 0:
            continue
        k = int(np.argmin(np.abs(m - cur)))
        if abs(m[k] - cur) <= max_jump:
            cur = m[k]
            pos[i] = cur
    return pos


def make_band_figure(waists, widths, num, mod_one, mod_all):
    """
    Dritte Abbildung: Lage EINES Bandes ueber die width-Achse.

    Zeigt den eigentlichen Punkt: die GESAMTE Verschiebung ist in beiden
    Faellen praktisch gleich (~-14 %). Beim Ein-Spot-Modell kommt sie als
    gleichmaessige Schraege, bei der vollen Ueberlagerung als Treppe -
    lange fast senkrechte Stuecke, unterbrochen von wenigen Spruengen.
    """
    wd = widths / 1e6
    if BAND_START_UM is None:
        cand = subpixel_minima(num[0], waists)
        if len(cand) == 0:
            print("kein Band in der untersten width-Zeile gefunden - Abbildung uebersprungen")
            return
        start = float(cand[len(cand) // 2])
        print("verfolgtes Band startet bei w_0 = %.3f um (automatisch gewaehlt)" % (start * 1e6))
    else:
        start = BAND_START_UM * 1e-6
    curves = [
        (track_band(mod_one, waists, start), COL_DEN, "model, nearest spot only"),
        (track_band(mod_all, waists, start), COL_NUM, "model, all neighbour spots"),
        (track_band(num, waists, start), COL_ETA, "simulation"),
    ]
    with plt.rc_context(LATEX_STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 4.0), constrained_layout=True)
        for pos, col, lab in curves:
            ax.plot(pos * 1e6, wd, lw=1.8, color=col, label=lab)
        ax.set_xlabel(r"Band position, waist $w_0$ ($\mu$m)")
        ax.set_ylabel("Width (MHz)")
        ax.set_title(r"Same total shift, different shape ($k = %.4g$)"
                     % resolved_scale_factor())
        ax.grid(True, lw=0.4, color="0.9")
        ax.set_axisbelow(True)
        ax.legend(loc="lower left")
        allpos = np.concatenate([p[np.isfinite(p)] for p, _, _ in curves]) * 1e6
        pad = 0.03 * (allpos.max() - allpos.min())
        ax.set_xlim(allpos.min() - pad, allpos.max() + pad)
        ax.set_ylim(wd.min(), wd.max())
        for sp in ax.spines.values():
            sp.set_color(COL_MUTED)
        ax.tick_params(colors=COL_MUTED)
        ax.xaxis.label.set_color("#1b1b1b")
        ax.yaxis.label.set_color("#1b1b1b")

        out_dir = OUT_DIR if os.path.isabs(OUT_DIR) else os.path.join(_here, OUT_DIR)
        stem = OUT_NAME_BAND + airy_scale.scale_tag(resolved_scale_factor())
        pdf_path = os.path.join(out_dir, stem + ".pdf")
        fig.savefig(pdf_path)
        print("gespeichert:", pdf_path)
        if SAVE_PNG:
            fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=200)
        if SHOW:
            plt.show()
        else:
            plt.close(fig)

    for pos, _, lab in curves:
        ok = pos[np.isfinite(pos)]
        slope = np.diff(pos) / np.diff(wd)
        print("%-28s Gesamtverschiebung %.1f %%, lokale Steigung Median %.2f um/MHz "
              "(min %.2f)" % (lab, 100 * (ok[-1] / ok[0] - 1),
                              np.nanmedian(slope) * 1e6, np.nanmin(slope) * 1e6))


def make_terms_figure(waists, widths, num, den):
    """
    Zweite Abbildung: warum eta_w entlang width so wenig tut. Zaehler und
    Nenner getrennt, jeweils auf ihren Wert am linken Rand normiert - damit
    sieht man sofort, dass das Nachbarlicht praktisch konstant bleibt und
    der Anstieg von eta_w fast vollstaendig aus dem SCHRUMPFENDEN Nenner
    (Eigenlicht am Atomort) kommt.
    """
    eta = num / den * 100.0
    wd = widths / 1e6

    with plt.rc_context(LATEX_STYLE):
        fig, axes = plt.subplots(1, len(WAIST_CUTS), figsize=(9.0, 3.6),
                                 constrained_layout=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, w_cut in zip(axes, WAIST_CUTS):
            j = int(np.argmin(np.abs(waists - w_cut)))
            for series, col, lab in ((num[:, j], COL_NUM, r"neighbour light $\langle I_\mathrm{nb}\rangle_W$"),
                                     (den[:, j], COL_DEN, r"own light $\langle I_\mathrm{own}\rangle_W$"),
                                     (eta[:, j], COL_ETA, r"crosstalk$_w$ = ratio")):
                ax.semilogy(wd, series / series[0], lw=1.8, color=col, label=lab)
            ax.axhline(1.0, lw=0.8, ls=":", color=COL_MUTED)
            ax.set_title(r"$w_0 = %.2f\,\mu$m" % (waists[j] * 1e6))
            ax.set_xlabel("Width (MHz)")
            ax.grid(True, lw=0.4, color="0.9")
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_color(COL_MUTED)
            ax.tick_params(colors=COL_MUTED)
            ax.xaxis.label.set_color("#1b1b1b")
        from matplotlib.ticker import FixedLocator, FixedFormatter
        ticks = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
        for ax in axes:
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_minor_locator(FixedLocator([]))
            ax.yaxis.set_major_formatter(FixedFormatter([("%g" % t) for t in ticks]))
        axes[0].set_ylabel("Relative to width $=$ 0.20 MHz")
        axes[0].yaxis.label.set_color("#1b1b1b")
        axes[0].legend(loc="lower left")

        out_dir = OUT_DIR if os.path.isabs(OUT_DIR) else os.path.join(_here, OUT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        stem = OUT_NAME_TERMS + airy_scale.scale_tag(resolved_scale_factor())
        pdf_path = os.path.join(out_dir, stem + ".pdf")
        fig.savefig(pdf_path)
        print("gespeichert:", pdf_path)
        if SAVE_PNG:
            fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=200)
        if SHOW:
            plt.show()
        else:
            plt.close(fig)

        for w_cut in WAIST_CUTS:
            j = int(np.argmin(np.abs(waists - w_cut)))
            print("w_0 = %.3f um, width %.2f -> %.2f MHz:  Zaehler x%.2f, Nenner x%.2f, eta_w x%.2f"
                  % (waists[j] * 1e6, wd[0], wd[-1], num[-1, j] / num[0, j],
                     den[-1, j] / den[0, j], eta[-1, j] / eta[0, j]))


def main():
    opt = wmfo.MultitoneFlatTopOptimizer(out_dir=os.path.join(_here, OUT_DIR),
                                         weighted_n_grid=WEIGHTED_N_GRID,
                                         airy_scale_factor=resolved_scale_factor())
    print("airy_scale_factor = %.6g" % opt.airy_scale_factor)
    print(airy_scale.describe(opt.airy_scale_factor))
    win_input = np.linspace(*WIN_INPUT_RANGE, N_WAIST)
    waists = (opt.f1 / opt.f2) * (opt.lambda_opt * opt.fLO) / (np.pi * win_input)
    widths = np.linspace(*WIDTH_RANGE, N_WIDTH)

    r_lo, r_hi = neighbour_radii(opt, widths[0]), neighbour_radii(opt, widths[-1])
    print("naechster Nachbarspot: %.3f um (width %.2f MHz) -> %.3f um (width %.2f MHz), "
          "also %.1f %% naeher"
          % (r_lo.min() * 1e6, widths[0] / 1e6, r_hi.min() * 1e6, widths[-1] / 1e6,
             100 * (1 - r_hi.min() / r_lo.min())))
    print("Spanne aller Nachbarspot-Abstaende: %.2f ... %.2f um"
          % (r_lo.min() * 1e6, r_lo.max() * 1e6))

    print("simuliere Zaehler/Nenner (%d x %d Punkte) ..." % (N_WIDTH, N_WAIST))
    num, den = simulate_maps(opt, waists, widths, AMPLITUDES)
    print("Zeilen mit lokalen Extrema entlang waist: Zaehler %d/%d, Nenner %d/%d"
          % (monotone_check(num), N_WIDTH, monotone_check(den), N_WIDTH))

    mod_one = model_map(opt, waists, widths, nearest_only=True)
    mod_all = model_map(opt, waists, widths, nearest_only=False)
    make_figure(waists, widths, num, mod_one, mod_all)
    make_terms_figure(waists, widths, num, den)
    make_band_figure(waists, widths, num, mod_one, mod_all)


if __name__ == "__main__":
    main()
