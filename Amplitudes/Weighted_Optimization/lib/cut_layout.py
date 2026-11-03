"""Karten + Schnitt entlang der Fit-Geraden in EINER Zeile.

Neues Standard-Layout (Oktober 2026) fuer die Uebersicht mit Amplituden:

  Combinated_Optimization:
    ..._metric_cuts_U_eta.pdf   2x3: Zeile U   = U_h-Karte, U_w-Karte, Schnitt
                                     Zeile eta = eta_h-Karte, eta_w-Karte, Schnitt
    ..._metric_cuts_r.pdf       1x3: r_x-Karte, r_y-Karte, Schnitt

  Hard_ / Weighted_Optimization:
    ..._metric_cuts_U_eta.pdf   2x2: Zeile U   = U-Karte, Schnitt
                                     Zeile eta = eta-Karte, Schnitt
    ..._metric_cuts_r.pdf       1x3: r_x-Karte, r_y-Karte, Schnitt

Das alte Raster (2x3 bzw. 2x2) bleibt im GUI waehlbar.

Gestaltung (so mit Daniel abgestimmt):
  - Figurbreite exakt 16 cm (\\linewidth), Schrift 10 pt / Ticks 9 pt.
  - Colorbars waagerecht UEBER den Karten, Beschriftung oben.
  - Width-Achse in µm ("Width $W$ (µm)"), µ aufrecht.
  - Fit-Gerade in den Karten ohne Legendeneintrag.
  - Schnitt: y-Achse rechts, oben eine Width-Achse mit Ticks genau ueber
    den Waist-Ticks. Liegen zwei Kurven weit auseinander, wird die Achse
    unterbrochen - beide Teile mit DEMSELBEN Massstab, damit die Kaestchen
    oben und unten gleich gross sind.
  - Legende: oberste Kurve zuerst; y-Beschriftung von unten nach oben.
  - Teilbild-Buchstaben oben links ueber der y-Achse, auf Hoehe der
    Colorbar-Beschriftung.

Diese Datei steht wortgleich in Hard_, Weighted_ und
Combinated_Optimization/lib - wer hier etwas aendert, aendert es in allen
drei Ordnern.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator


CUT_LAYOUT_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Masse in cm
FIG_WIDTH_CM = 16.0
MAP_W, MAP_H, CUT_W = 3.75, 3.6, 3.85
LEFT_MARGIN = 1.7        # y-Beschriftung + Ticks der ersten Karte
GAP_MAP_MAP = 0.65       # zwischen zwei Karten (zweite ohne y-Zahlen)
GAP_MAP_CUT = 0.7        # Karte -> Schnitt (Schnitt hat seine Achse rechts)
RIGHT_MARGIN = 1.6       # rechte y-Achse des Schnitts (Inhalt = 16 cm, Start bei 1.7 cm)
ROW_BOTTOM = 1.15        # x-Beschriftung
ROW_TOP = 1.6            # Colorbar bzw. obere Width-Achse
BREAK_GAP = 0.12         # Luecke der unterbrochenen Achse
TAG_DX = 0.5             # Buchstabe links von der Achse
# Figurhoehe je Zeilenzahl, wie in Daniels Referenz-PDFs (Oktober 2026):
# 1 Zeile 6.4 cm (PenaltyRegion_amplitudes_1x3), 2 Zeilen 12.6 cm
# (PenaltyRegion_uniformity_crosstalk_2x3). Die unterste Zeile beginnt immer
# bei ROW_BOTTOM; andere Zeilenzahlen: Zeilenabstand * n + 0.1.
FIG_HEIGHT_CM = {1: 6.4, 2: 12.6}
# Schnitt: hoechstens so viele Kaestchen (Tick-Abstaende) auf der y-Achse.
# Bleibt Platz, wird er oben als Kopfraum fuer die Legende aufgefuellt.
CUT_MAX_BOXES = 4

FIT_COLOR = "#00BFFF"
WIDTH_UNIT_LABEL = "Width $W$ (µm)"
WAIST_LABEL = r"Waist $\omega$ (µm)"


def _cm(fig, x, y, w, h):
    W, H = fig.get_size_inches() * 2.54
    return [x / W, y / H, w / W, h / H]


def nice_ticks(lo, hi, n):
    """Genau n (oder so nah daran wie moeglich) runde Marken in [lo, hi]."""
    lo, hi = sorted((float(lo), float(hi)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return []
    dec = 10.0 ** math.floor(math.log10(hi - lo))
    best = None
    for p in (dec * 0.1, dec, dec * 10):
        for f in (1.0, 2.0, 2.5, 4.0, 5.0, 8.0):
            s = f * p
            first = math.ceil(lo / s - 1e-9) * s
            t = np.arange(first, hi + 0.5 * s, s)
            t = t[(t >= lo - 1e-12) & (t <= hi + 1e-12)]
            if t.size == 0:
                continue
            key = (0 if t.size == n else 1, 0 if t.size > n else 1, abs(t.size - n), -s)
            if best is None or key < best[0]:
                best = (key, t)
    t = best[1]
    if t.size > n:
        t = t[t.size - n:]
    return [float(v) for v in t]


def _fmt(step):
    nd = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    if abs(step * 10 ** nd - round(step * 10 ** nd)) > 1e-6:
        nd += 1
    return "%." + str(nd) + "f"


# ----------------------------------------------------------------------
# unterbrochene Achse mit gleichem Massstab
# ----------------------------------------------------------------------
_STEPS = [f * 10.0 ** k for k in range(-4, 4) for f in (1.0, 2.0, 2.5, 5.0)]


def _segment(lo, hi, step):
    """(Anzahl Kaestchen, erster Tick) fuer einen Datenbereich bei
    Schrittweite step. Bevorzugt Ticks auf Vielfachen von step, sonst auf
    step/2 bzw. step/5 - immer mit moeglichst wenigen Kaestchen."""
    tol = 0.12 * step  # Kurve darf knapp (12 %) unter/ueber der Randmarke liegen
    best = None
    # Nur feinere Raster, deren Werte mit den Nachkommastellen von step
    # darstellbar sind - sonst stuenden an der Achse Zahlen wie 0.963.
    nd = int(_fmt(step)[2:-1])
    feine = [f for f in (step, step / 2.0, step / 5.0)
             if abs(round(f, nd) - f) < 1e-12 * max(1.0, abs(f))]
    for rang, fein in enumerate(feine):
        t0 = math.floor((lo + tol) / fein + 1e-9) * fein
        n = max(1, math.ceil((hi - tol - t0) / step - 1e-9))
        # zentrieren, falls auf dem feinen Raster moeglich
        if fein < step:
            mitte = lo - (n * step - (hi - lo)) / 2.0
            t0c = round(mitte / fein) * fein
            if t0c - tol <= lo and t0c + n * step + tol >= hi:
                t0 = t0c
        if t0 - tol > lo or t0 + n * step + tol < hi:
            continue
        key = (n, rang)
        if best is None or key < best[0]:
            best = (key, n, t0)
    if best is None:
        t0 = math.floor(lo / step) * step
        return max(1, math.ceil((hi - t0) / step)), t0
    return best[1], best[2]


def broken_axis_plan(lower, upper, max_boxes=CUT_MAX_BOXES):
    """Ticks und Grenzen fuer beide Teile einer unterbrochenen Achse.

    lower/upper: (min, max) der Kurven unten bzw. oben. Oben kommt ein
    Kaestchen Kopfraum fuer die Legende dazu. Gibt dict mit 'lo'/'up'
    = (lim, ticks) und 'step' zurueck.
    """
    for step in _STEPS:
        nl, tl = _segment(lower[0], lower[1], step)
        nu, tu = _segment(upper[0], upper[1], step)
        if nl + nu + 1 <= max_boxes:
            break
    pad = 0.2 * step
    kopf = max(1, max_boxes - nl - nu)                         # Rest = Kopfraum
    ticks_l = [tl + k * step for k in range(nl + 1)]
    ticks_u = [tu + k * step for k in range(nu + 1 + kopf)]
    lim_l = (min(ticks_l[0] - pad, lower[0] - 0.02 * step),
             max(ticks_l[-1] + pad, lower[1] + 0.02 * step))
    # Liegt die obere Kurve knapp unter ihrer untersten Marke, bekommt sie
    # ein halbes Kaestchen Platz nach unten.
    unten_u = ticks_u[0] - (0.5 * step if upper[0] < ticks_u[0] else pad)
    lim_u = (min(unten_u, upper[0] - 0.02 * step), ticks_u[-1])
    return dict(step=step, lo=(lim_l, ticks_l), up=(lim_u, ticks_u))


def _should_break(traces):
    """Zwei Gruppen weit auseinander? -> (untere Indizes, obere Indizes)."""
    if len(traces) < 2:
        return None
    bereiche = []
    for i, tr in enumerate(traces):
        y = np.asarray(tr["y"], dtype=float)
        y = y[np.isfinite(y)]
        if y.size:
            bereiche.append((float(y.mean()), float(y.min()), float(y.max()), i))
    if len(bereiche) < 2:
        return None
    bereiche.sort()
    luecken = [(bereiche[k + 1][1] - max(b[2] for b in bereiche[:k + 1]), k)
               for k in range(len(bereiche) - 1)]
    luecke, k = max(luecken)
    gesamt = max(b[2] for b in bereiche) - min(b[1] for b in bereiche)
    if gesamt <= 0 or luecke / gesamt < 0.4:
        return None
    unten = [b[3] for b in bereiche[:k + 1]]
    oben = [b[3] for b in bereiche[k + 1:]]
    return unten, oben


def _range_of(traces, idx):
    werte = np.concatenate([np.asarray(traces[i]["y"], dtype=float) for i in idx])
    werte = werte[np.isfinite(werte)]
    return float(werte.min()), float(werte.max())


def _break_marks(au, al):
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=7, linestyle="none",
              color="k", mec="k", mew=0.8, clip_on=False)
    au.plot([0, 1], [0, 0], transform=au.transAxes, **kw)
    al.plot([0, 1], [1, 1], transform=al.transAxes, **kw)


def _style_cut_axis(ax, xlim):
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_xlim(*xlim)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")


# ----------------------------------------------------------------------
# Bausteine
# ----------------------------------------------------------------------
def _draw_map(fig, rect, spec, row, show_ylabel):
    ax = fig.add_axes(rect)
    x = np.asarray(spec["x"], dtype=float)
    y = np.asarray(spec["y"], dtype=float)
    Z = np.asarray(spec["Z"], dtype=float)
    norm = spec.get("norm")
    grenzen = dict(norm=norm) if norm is not None else dict(
        vmin=spec.get("vmin"), vmax=spec.get("vmax"))
    pc = ax.pcolormesh(x, y, Z, shading="auto", cmap=spec["cmap"],
                       rasterized=True, **grenzen)
    fl = row.get("fitline")
    if fl is not None:
        x_in, y_in, x_out, y_out = fl
        ax.plot(x_out, y_out, color=FIT_COLOR, lw=1.5)
        ax.plot(x_in, y_in, color=FIT_COLOR, lw=1.5)
    bp = row.get("best_point")
    if bp is not None:
        ax.plot([bp[0]], [bp[1]], marker="*", color="red", markersize=11,
                markeredgecolor="k", markeredgewidth=0.5, linestyle="none")
    # pcolormesh mit Zentren: die Grenzen sind die halben Zellen aussen
    ax.set_xlim(*_edges(x))
    ax.set_ylim(*_edges(y))
    ax.xaxis.set_major_locator(FixedLocator(nice_ticks(*ax.get_xlim(), 3)))
    ax.yaxis.set_major_locator(FixedLocator(nice_ticks(*ax.get_ylim(), 5)))
    ax.set_xlabel(row.get("xlabel", WAIST_LABEL))
    if show_ylabel:
        ax.set_ylabel(row.get("ylabel", WIDTH_UNIT_LABEL))
    else:
        ax.tick_params(labelleft=False)

    cax = ax.inset_axes([0.0, 1.05, 1.0, 0.055])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cb.set_label(spec["cbar"], labelpad=3)
    ticks = spec.get("cbar_ticks")
    if ticks is None:
        lo, hi = (norm.vmin, norm.vmax) if norm is not None else pc.get_clim()
        ticks = nice_ticks(lo, hi, 4)
    cax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    cax.xaxis.set_minor_locator(NullLocator())
    cax.xaxis.set_major_formatter(FormatStrFormatter(spec.get("cbar_fmt", "%g")))
    ax._cut_layout_cax = cax
    return ax


def _edges(v):
    v = np.sort(np.asarray(v, dtype=float))
    if v.size < 2:
        return float(v[0]) - 0.5, float(v[0]) + 0.5
    return float(v[0] - 0.5 * (v[1] - v[0])), float(v[-1] + 0.5 * (v[-1] - v[-2]))


def _top_width_axis(ax, cut, xticks):
    a, b = cut.get("width_line", (None, None))
    if a is None or not np.isfinite(a) or a == 0:
        return
    sek = ax.secondary_xaxis("top", functions=(
        lambda t: np.asarray(t, dtype=float) * a + b,
        lambda u: (np.asarray(u, dtype=float) - b) / a))
    tw = [a * t + b for t in xticks]
    sek.set_xticks(tw)
    sek.set_xticklabels(["%.2f" % v for v in tw])
    sek.set_xlabel(cut.get("top_label", WIDTH_UNIT_LABEL), labelpad=3)


def _ylabel(traces, unit):
    reihen = sorted(traces, key=lambda t: np.nanmean(np.asarray(t["y"], float)))
    text = ", ".join(t["label"] for t in reihen)
    return text + (" (%s)" % unit if unit else "")


def _legend(ax, traces, handles):
    if len(traces) < 2:
        return
    ordnung = sorted(range(len(traces)),
                     key=lambda i: -np.nanmean(np.asarray(traces[i]["y"], float)))
    ax.legend([handles[i] for i in ordnung], [traces[i]["label"] for i in ordnung],
              loc="upper left", framealpha=0.9)


def _draw_cut(fig, x0, y0, cut):
    """Gibt die Achse zurueck, ueber der der Teilbild-Buchstabe steht."""
    traces = cut["traces"]
    x = np.asarray(cut["x"], dtype=float)
    # x-Bereich: 2 % Rand, dann nach aussen auf runde Werte (unten 0.01,
    # oben 0.05) - so wie in Daniels Referenz-PDFs (0.77 .. 1.30 µm).
    xa, xb = float(np.nanmin(x)), float(np.nanmax(x))
    rand = 0.02 * (xb - xa)
    xlim = (math.floor((xa - rand) * 100.0 + 1e-9) / 100.0,
            math.ceil((xb + rand) * 20.0 - 1e-9) / 20.0)
    xticks = nice_ticks(xlim[0], xlim[1], 3)
    unit = cut.get("unit", "")
    teilung = _should_break(traces)
    wp = cut.get("working_point_x")

    if teilung is None:
        ax = fig.add_axes(_cm(fig, x0, y0, CUT_W, MAP_H))
        _style_cut_axis(ax, xlim)
        hs = [ax.plot(x, tr["y"], color=tr["color"], lw=1.6)[0] for tr in traces]
        lo, hi = _range_of(traces, range(len(traces)))
        span = max(hi - lo, abs(hi) * 0.05, 1e-12)
        if unit == "%" and lo >= 0 and lo < 0.5 * hi:
            unten = 0.0
        else:
            unten = lo - 0.15 * span
        # Kleinste runde Schrittweite, mit der die Daten in hoechstens
        # CUT_MAX_BOXES Kaestchen passen; die Achse endet auf der obersten
        # Marke (z.B. eta: 0 / 5 / 10 / 15).
        for step in _STEPS:
            t0 = math.floor(unten / step + 1e-9) * step
            n = max(1, math.ceil((hi - t0) / step - 1e-9))
            if n <= CUT_MAX_BOXES:
                break
        ticks = [t0 + k * step for k in range(n + 1)]
        unten, oben = (0.0 if unten == 0.0 else min(unten, t0)), ticks[-1]
        if unten > ticks[0]:
            unten = ticks[0]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FormatStrFormatter(_fmt(step)))
        ax.set_ylim(unten, oben)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.set_xlabel(cut.get("xlabel", WAIST_LABEL))
        ax.set_ylabel(_ylabel(traces, unit))
        if wp is not None:
            ax.axvline(wp, color="red", lw=1.2, ls="--", alpha=0.8)
        _legend(ax, traces, hs)
        _top_width_axis(ax, cut, xticks)
        return ax

    unten_idx, oben_idx = teilung
    plan = broken_axis_plan(_range_of(traces, unten_idx), _range_of(traces, oben_idx))
    (lim_l, t_l), (lim_u, t_u) = plan["lo"], plan["up"]
    rl, ru = lim_l[1] - lim_l[0], lim_u[1] - lim_u[0]
    hl = (MAP_H - BREAK_GAP) * rl / (rl + ru)
    hu = MAP_H - BREAK_GAP - hl
    al = fig.add_axes(_cm(fig, x0, y0, CUT_W, hl))
    au = fig.add_axes(_cm(fig, x0, y0 + hl + BREAK_GAP, CUT_W, hu), sharex=al)
    fmt = FormatStrFormatter(_fmt(plan["step"]))
    for a, lim, t in ((au, lim_u, t_u), (al, lim_l, t_l)):
        _style_cut_axis(a, xlim)
        a.set_ylim(*lim)
        a.yaxis.set_major_locator(FixedLocator(t))
        a.yaxis.set_major_formatter(fmt)
    hs = []
    for i, tr in enumerate(traces):
        a = au if i in oben_idx else al
        hs.append(a.plot(x, tr["y"], color=tr["color"], lw=1.6)[0])
    au.spines["bottom"].set_visible(False)
    al.spines["top"].set_visible(False)
    au.tick_params(bottom=False, labelbottom=False)
    al.xaxis.set_major_locator(FixedLocator(xticks))
    al.set_xlabel(cut.get("xlabel", WAIST_LABEL))
    _break_marks(au, al)
    if wp is not None:
        for a in (au, al):
            a.axvline(wp, color="red", lw=1.2, ls="--", alpha=0.8)
    _legend(au, traces, hs)
    _top_width_axis(au, cut, xticks)
    au._cut_layout_partner = al
    au._cut_layout_ylabel = _ylabel(traces, unit)
    return au


def _place_broken_ylabel(fig, au):
    """y-Beschriftung der unterbrochenen Achse: mittig ueber BEIDE Teile,
    rechts neben den breitesten Tick-Zahlen."""
    al = au._cut_layout_partner
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    rechts = au.get_window_extent(r).x1
    for a in (au, al):
        for lab in a.get_yticklabels():
            if lab.get_text():
                rechts = max(rechts, lab.get_window_extent(r).x1)
    inv = fig.transFigure.inverted()
    x = inv.transform((rechts + 4.0, 0))[0]
    y = 0.5 * (au.get_position().y1 + al.get_position().y0)
    fig.text(x, y, au._cut_layout_ylabel, rotation=90, ha="left", va="center",
             fontsize=plt.rcParams["axes.labelsize"])


# ----------------------------------------------------------------------
# Haupteinstieg
# ----------------------------------------------------------------------
def render_rows(rows, out_path=None, tags=None, finish=None):
    """rows: Liste von dict(maps=[...], cut={...}|None, fitline=..., best_point=...).

    maps: dict(x, y, Z, cmap, norm|vmin/vmax, cbar, cbar_ticks=None, cbar_fmt)
          x = Waist (µm), y = Width (µm) als Zellzentren, Z[len(y), len(x)].
    cut:  dict(x, traces=[dict(y, color, label)], unit="%"|"",
               width_line=(a, b) mit W/µm = a*waist/µm + b,
               working_point_x=None)
    fitline: (x_in, y_in, x_out, y_out) in Kartenkoordinaten oder None.

    Gibt die Figur zurueck; mit out_path wird sie als PDF gespeichert.
    finish: optional fig -> Ergebnis, wird noch INNERHALB des Stils
    aufgerufen (Speichern mit den richtigen Schriften) und sein Ergebnis
    zurueckgegeben.
    """
    n_maps = max(len(r["maps"]) for r in rows)
    inhalt = (LEFT_MARGIN + n_maps * MAP_W + (n_maps - 1) * GAP_MAP_MAP
              + (GAP_MAP_CUT + CUT_W if any(r.get("cut") for r in rows) else 0.0)
              + RIGHT_MARGIN)
    x_start = LEFT_MARGIN + max(0.0, (FIG_WIDTH_CM - inhalt) / 2.0)
    pitch = ROW_BOTTOM + MAP_H + ROW_TOP
    hoehe = FIG_HEIGHT_CM.get(len(rows), pitch * len(rows) + 0.1)
    tags = list(tags) if tags is not None else ["(%s)" % c for c in "abcdefghijkl"]

    with plt.rc_context(CUT_LAYOUT_RC):
        fig = plt.figure(figsize=(FIG_WIDTH_CM / 2.54, hoehe / 2.54))
        beschriftungen = []        # (Achse, Text, Referenz-Colorbar)
        gebrochen = []
        k = 0
        for zeile, row in enumerate(rows):
            y0 = ROW_BOTTOM + (len(rows) - 1 - zeile) * pitch
            x = x_start
            ref = None
            for m, spec in enumerate(row["maps"]):
                ax = _draw_map(fig, _cm(fig, x, y0, MAP_W, MAP_H), spec, row,
                               show_ylabel=(m == 0))
                ref = ref or ax._cut_layout_cax
                beschriftungen.append((ax, tags[k], ref)); k += 1
                x += MAP_W + GAP_MAP_MAP
            if row.get("cut"):
                x = x - GAP_MAP_MAP + GAP_MAP_CUT
                ax = _draw_cut(fig, x, y0, row["cut"])
                if hasattr(ax, "_cut_layout_partner"):
                    gebrochen.append(ax)
                beschriftungen.append((ax, tags[k], ref)); k += 1

        for au in gebrochen:
            _place_broken_ylabel(fig, au)

        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        W = fig.get_size_inches()[0] * 2.54
        for ax, text, ref in beschriftungen:
            p = ax.get_position()
            if ref is not None:
                bb = ref.xaxis.label.get_window_extent(r)
                y = inv.transform((0, 0.5 * (bb.y0 + bb.y1)))[1]
            else:
                y = p.y1 + 1.27 / (fig.get_size_inches()[1] * 2.54)
            fig.text(p.x0 - TAG_DX / W, y, text, ha="left", va="center", fontsize=10)

        if out_path is not None:
            fig.savefig(out_path, dpi=600)
        if finish is not None:
            return finish(fig)
    return fig
