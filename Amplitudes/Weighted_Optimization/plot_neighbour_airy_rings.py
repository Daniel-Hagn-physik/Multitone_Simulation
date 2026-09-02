"""
Airy-Ringe der Nachbar-Sites: warum der atom-gewichtete Crosstalk Streifen hat
=============================================================================

Erzeugt eine zweiteilige Abbildung (LaTeX-tauglich, englisch beschriftet):

  links   Nachbar-Intensitaet entlang der Verbindungslinie Atom -> Nachbar-Site
          (y = 0), fuer zwei Waists: einen im Crosstalk-Minimum, einen im
          Maximum. Man sieht direkt, dass das Atom im einen Fall auf einem
          dunklen, im anderen auf einem hellen Airy-Ring des Nachbarn sitzt.
          Der Bereich reicht bis in das Hauptmaximum der Nachbar-Site hinein.

  rechts  eta_w(waist) bei fester width, mit genau diesen beiden Punkten
          markiert.

Hintergrund
-----------
`weighted_crosstalk()` ist eta_w = sum(I_Nachbar*W) / sum(I_eigen*W).
`_local_neighbor_intensity()` legt 8 Kopien der gesamten N_x x N_y-Struktur um
+/- pitch herum. Beim Airy-Profil faellt die Flanke eines Spots asymptotisch wie

    I(u) ~ (8/pi) * cos^2(u - 3pi/4) / u^3 ,   u = 3.8317 * r / (1.19 * w_0)

also 1/u^3 mit aufmoduliertem cos^2 - das sind die Ringe, Periode pi in u.
Der Waist bewegt weder das Atom noch den Nachbarn, er dilatiert nur das
Ringsystem (Ringradien ~ w_0). Beim Scannen des Waists wandern die Ringe
deshalb ueber den festen Atomort: dunkler Ring = Crosstalk-Minimum, heller
Ring = Maximum. Das erzeugt die Streifen in der (waist, width)-Karte.

Entlang der width-Achse passiert das NICHT, weil der Nachbarabstand durch den
festen pitch gesetzt ist: ueber den ganzen width-Bereich aendert sich u um
weniger als pi, also nicht einmal um eine Ringperiode.

Aufruf
------
Einfach ausfuehren. Liegt neben `weighted_multitone_flattop_optimizer.py`
(Ordner `Weighted_Optimization/`); von woanders aus den Pfad in OPT_DIR
setzen. Alle Stellschrauben stehen unten im Konfigurationsblock.

Laufzeit: ~1-2 Minuten fuer N_WAIST = 181 Punkte (jeder Punkt ist eine
Auswertung auf dem lokalen 241x241-Sub-Grid).
"""

import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ======================================================================
# Konfiguration
# ======================================================================

# Ordner mit weighted_multitone_flattop_optimizer.py. None = derselbe Ordner
# wie dieses Skript.
OPT_DIR = None

WIDTH_HZ = 0.30e6          # feste width, an der geschnitten wird
WIN_INPUT_RANGE = (0.8e-3, 1.7e-3)   # Eingangswaist vor der Linse (m)
N_WAIST = 181              # Punkte fuer die eta_w(waist)-Kurve

# Ortsachse des linken Panels, relativ zum Atom (m). Der Nachbar sitzt bei
# pitch = 5.288 um, deshalb reicht der Default bis knapp dahinter, damit das
# Hauptmaximum der Nachbar-Site mit im Bild ist.
X_RANGE = (-0.6e-6, 6.4e-6)
N_X = 4000

# Welche beiden Waists gezeigt werden. None = automatisch das benachbarte
# Minimum/Maximum-Paar mit dem groessten Verhaeltnis. Sonst z.B. 0.863e-6.
WAIST_MIN_M = None
WAIST_MAX_M = None

AMPLITUDES = None          # None = alle Amplituden 1; sonst Array der Laenge N_x+N_y

# Airy-Skalenfaktor: first_zero_radius = AIRY_SCALE_FACTOR * waist.
# None = airy_scale.AIRY_SCALE_DIALOG_DEFAULT, also 1.482951 ("1/e^2 der
# Airy-Hauptkeule liegt bei waist") - derselbe Default wie in den
# Scan-Startdialogen seit 2026-09-01. Fuer den Vergleich mit aelteren
# Datensaetzen hier airy_scale.AIRY_SCALE_LEGACY (1.19) eintragen; der
# Dateiname bekommt dann automatisch wieder KEIN Kuerzel, waehrend jeder
# andere Faktor eines bekommt (z.B. "_k1.483"), damit sich zwei nicht
# vergleichbare Abbildungen nie gegenseitig ueberschreiben.
AIRY_SCALE_FACTOR = None

OUT_DIR = "Bilder"
OUT_NAME = "neighbour_airy_rings"   # ohne Endung; es werden .pdf und .png geschrieben
SAVE_PNG = True
SHOW = False

# LaTeX-tauglicher Stil - bewusst NUR per rc_context gesetzt, damit er nicht
# global auf andere Skripte abfaerbt. pdf.fonttype=42 bettet TrueType statt
# Type-3-Schriften ein (Type 3 wird von vielen Journals abgelehnt).
LATEX_STYLE = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COL_MIN = "#1f4e79"   # Kurve im Crosstalk-Minimum
COL_MAX = "#e08214"   # Kurve im Crosstalk-Maximum
COL_ATOM = "#b3402a"  # Atomort / thermische Breite
COL_MUTED = "#6b6b6b"


# ======================================================================
# Optimierer importieren
# ======================================================================

# Die Module liegen seit dem Aufraeumen (2026-09-02) im Unterordner lib/;
# OPT_DIR (oben im Skript) uebersteuert das weiterhin, falls der Optimierer
# woanders liegt.
_here = os.path.dirname(os.path.abspath(__file__))
_lib = os.path.join(_here, "lib")
sys.path.insert(0, OPT_DIR if OPT_DIR else _lib)
import weighted_multitone_flattop_optimizer as wmfo  # noqa: E402
import airy_scale  # noqa: E402


def resolved_scale_factor():
    """Der tatsaechlich verwendete Airy-Skalenfaktor."""
    return (airy_scale.AIRY_SCALE_DIALOG_DEFAULT if AIRY_SCALE_FACTOR is None
            else float(AIRY_SCALE_FACTOR))


def waist_from_win_input(opt, win_input):
    """Effektiver Waist nach der Linse (m) aus dem Eingangswaist davor (m)."""
    return (opt.f1 / opt.f2) * (opt.lambda_opt * opt.fLO) / (np.pi * np.asarray(win_input))


def spot_amplitudes(opt, amps=None):
    """Amplitude pro Spot aus den N_x + N_y Achsen-Amplituden (wie in _evaluate())."""
    n_spots = opt.N_x * opt.N_y
    if amps is None:
        return np.ones(n_spots)
    amps = np.asarray(amps, dtype=float)
    amp_x = amps[:opt.N_x]
    amp_y = amps[opt.N_x:opt.N_x + opt.N_y]
    return np.repeat(amp_x, opt.N_y) * np.tile(amp_y, opt.N_x)


def crosstalk_vs_waist(opt, waists, width_hz, amps=None):
    """eta_w (in %) fuer eine Reihe von Waists bei fester width."""
    out = np.full(len(waists), np.nan)
    for i, w in enumerate(waists):
        res = opt._evaluate_weighted_only(w, width_hz, amps=amps)
        if res is not None:
            out[i] = res["eta_weighted"] * 100.0
    return out


def neighbour_cut(opt, waist_m, width_hz, x_rel, amps=None):
    """
    Nachbar-Intensitaet entlang y = 0, aufgetragen ueber den Abstand x_rel vom
    Atom (das in der Site-Mitte sitzt). Nutzt exakt dieselbe Funktion wie die
    gewichtete Metrik, damit hier nichts nachgebaut wird.
    """
    centers_x, centers_y, r_center = opt._compute_centers_for_width(width_hz)
    profile_func = opt._profile_func()
    scale = opt._profile_scale(waist_m)
    amp_spots = spot_amplitudes(opt, amps)

    X = (r_center + np.asarray(x_rel)).reshape(1, -1)
    Y = np.full_like(X, r_center)
    return opt._local_neighbor_intensity(
        X, Y, centers_x, centers_y, scale, amp_spots, profile_func
    )[0]


def pick_extrema(waists, eta):
    """
    Sucht das benachbarte (Minimum, Maximum)-Paar mit dem groessten Verhaeltnis.
    Gibt (waist_min, waist_max) zurueck - "min"/"max" bezieht sich auf eta_w,
    nicht auf den Waist.
    """
    lo = [j for j in range(1, len(eta) - 1) if eta[j] <= eta[j - 1] and eta[j] <= eta[j + 1]]
    hi = [j for j in range(1, len(eta) - 1) if eta[j] >= eta[j - 1] and eta[j] >= eta[j + 1]]
    if not lo or not hi:
        raise RuntimeError(
            "Keine Oszillation gefunden - Waist-Bereich zu schmal oder zu grob abgetastet."
        )
    best = max(
        ((j, k) for j in lo for k in hi),
        key=lambda p: eta[p[1]] / eta[p[0]] - 0.02 * abs(p[0] - p[1]),
    )
    return waists[best[0]], waists[best[1]]


# ======================================================================
# Abbildung
# ======================================================================

def make_figure(opt, waists, eta, waist_min, waist_max, width_hz, x_rel, amps=None):
    k = opt.airy_scale_factor
    cut_min = neighbour_cut(opt, waist_min, width_hz, x_rel, amps)
    cut_max = neighbour_cut(opt, waist_max, width_hz, x_rel, amps)
    # BEWUSST NICHT normiert: die Airy-Funktion (2*J1(u)/u)^2 hat bei u -> 0 den
    # Wert 1, ein einzelner Spot hat also per Konstruktion Peakintensitaet 1.
    # Damit ist die Kurve schon in einer physikalisch sinnvollen Einheit, und
    # die beiden Waists bleiben direkt vergleichbar. Wuerde man jede Kurve auf
    # ihr EIGENES Maximum normieren, teilte man mit zwei verschiedenen Zahlen
    # (hier 1.92 vs 2.42, weil der breitere Spot die Luecken zwischen den
    # y-Reihen staerker auffuellt) und schrumpfte den echten Kontrast am
    # Atomort von 3.3 auf scheinbare 2.6.

    e_min = eta[np.argmin(np.abs(waists - waist_min))]
    e_max = eta[np.argmin(np.abs(waists - waist_max))]
    i0 = np.argmin(np.abs(x_rel))
    ratio = cut_max[i0] / cut_min[i0]

    sigma_atom = opt.sigma_atom * 1e6
    pitch_um = opt.pitch * 1e6
    x_um = np.asarray(x_rel) * 1e6

    with plt.rc_context(LATEX_STYLE):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)

        # --- links: Schnitt entlang y = 0 -----------------------------
        axL.semilogy(x_um, cut_min, lw=1.6, color=COL_MIN,
                     label=r"$w_0 = %.3f\,\mu$m (crosstalk min.)" % (waist_min * 1e6))
        axL.semilogy(x_um, cut_max, lw=1.6, color=COL_MAX,
                     label=r"$w_0 = %.3f\,\mu$m (crosstalk max.)" % (waist_max * 1e6))
        axL.axvspan(-2 * sigma_atom, 2 * sigma_atom, color=COL_ATOM, alpha=0.16, lw=0)
        axL.axvline(0.0, color=COL_ATOM, lw=1.0)
        axL.plot([0, 0], [cut_min[i0], cut_max[i0]], "o", ms=5, color=COL_ATOM, zorder=5)
        axL.annotate(r"atom, $\times %.1f$" % ratio,
                     (0.0, np.sqrt(cut_min[i0] * cut_max[i0])),
                     xytext=(11, -3), ha="left", textcoords="offset points",
                     color=COL_ATOM, fontsize=8,
                     bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        axL.annotate("neighbour site", (pitch_um, max(cut_min.max(), cut_max.max())),
                     xytext=(0, 6), textcoords="offset points", ha="center",
                     color=COL_MUTED, fontsize=8)
        axL.set_xlim(x_um.min(), x_um.max())
        axL.axhline(1.0, lw=0.9, ls=":", color=COL_MUTED)
        axL.annotate("single spot peak", (x_um.min(), 1.0), xytext=(4, 4),
                     textcoords="offset points", color=COL_MUTED, fontsize=8)
        axL.set_ylim(min(cut_min.min(), cut_max.min()) * 0.7,
                     max(cut_min.max(), cut_max.max()) * 3.0)
        axL.set_xlabel(r"$x$ ($\mu$m), $0 =$ atom, %.2f $=$ neighbour site" % pitch_um)
        axL.set_ylabel(r"Neighbour intensity (single-spot peak $= 1$)")
        axL.set_title("Airy rings of the neighbour site at the atom")
        axL.annotate(r"Airy scale $k = %.4g$  ($R_1 = k\,w_0$)" % k,
                     (0.985, 0.045), xycoords="axes fraction", ha="right",
                     color=COL_MUTED, fontsize=8)
        axL.legend(loc="upper left")

        # --- rechts: eta_w(waist) -------------------------------------
        axR.plot(waists * 1e6, eta, lw=1.8, color=COL_MUTED)
        axR.plot([waist_min * 1e6], [e_min], "o", ms=6, color=COL_MIN)
        axR.plot([waist_max * 1e6], [e_max], "o", ms=6, color=COL_MAX)
        axR.annotate("min", (waist_min * 1e6, e_min), xytext=(5, -12),
                     textcoords="offset points", color=COL_MIN, fontsize=8)
        axR.annotate("max", (waist_max * 1e6, e_max), xytext=(5, 4),
                     textcoords="offset points", color=COL_MAX, fontsize=8)
        axR.set_xlabel(r"Waist $w_0$ ($\mu$m, after lenses)")
        axR.set_ylabel(r"Crosstalk$_w$ (\%)" if matplotlib.rcParams["text.usetex"]
                       else r"Crosstalk$_w$ (%)")
        axR.set_title(r"Width $= %.2f$ MHz: one ring per stripe" % (width_hz / 1e6))

        for ax in (axL, axR):
            ax.grid(True, lw=0.4, color="0.9")
            ax.set_axisbelow(True)
            for sp in ax.spines.values():
                sp.set_color(COL_MUTED)
            ax.tick_params(colors=COL_MUTED)
            ax.xaxis.label.set_color("#1b1b1b")
            ax.yaxis.label.set_color("#1b1b1b")

        out_dir = OUT_DIR if os.path.isabs(OUT_DIR) else os.path.join(_here, OUT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        stem = OUT_NAME + airy_scale.scale_tag(k)
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

    return dict(waist_min=waist_min, waist_max=waist_max, eta_min=e_min,
                eta_max=e_max, ratio_at_atom=ratio)


def main():
    k = resolved_scale_factor()
    opt = wmfo.MultitoneFlatTopOptimizer(out_dir=os.path.join(_here, OUT_DIR),
                                         airy_scale_factor=k)
    print("pitch = %.3f um, sigma_atom = %.1f nm, Profil = %s, airy_scale_factor = %.6g"
          % (opt.pitch * 1e6, opt.sigma_atom * 1e9, opt.profile, opt.airy_scale_factor))
    print(airy_scale.describe(opt.airy_scale_factor))
    print("Ringabstand in Ortskoordinaten: dr = pi*%.2f*w_0/3.8317 = %.3f*w_0"
          % (opt.airy_scale_factor, np.pi * opt.airy_scale_factor / 3.83170597))

    win_input = np.linspace(WIN_INPUT_RANGE[0], WIN_INPUT_RANGE[1], N_WAIST)
    waists = waist_from_win_input(opt, win_input)

    print("berechne eta_w(waist) an %d Punkten bei width = %.3f MHz ..."
          % (N_WAIST, WIDTH_HZ / 1e6))
    eta = crosstalk_vs_waist(opt, waists, WIDTH_HZ, amps=AMPLITUDES)

    if WAIST_MIN_M is None or WAIST_MAX_M is None:
        w_min, w_max = pick_extrema(waists, eta)
    else:
        w_min, w_max = WAIST_MIN_M, WAIST_MAX_M
    print("gewaehlt: Minimum bei w_0 = %.4f um, Maximum bei w_0 = %.4f um"
          % (w_min * 1e6, w_max * 1e6))

    x_rel = np.linspace(X_RANGE[0], X_RANGE[1], N_X)
    info = make_figure(opt, waists, eta, w_min, w_max, WIDTH_HZ, x_rel, amps=AMPLITUDES)
    print("eta_w: %.3f %% (min) vs %.3f %% (max); Nachbarlicht am Atomort Faktor %.2f"
          % (info["eta_min"], info["eta_max"], info["ratio_at_atom"]))
    return info


if __name__ == "__main__":
    main()
