"""
lib/atom_offset_report.py - Plots und Bericht zum Atom-Versatz (Multitone)
=========================================================================

Die Kurven sehen aus wie die des Einzelstrahl-Positions-Sweeps
(single_beam_report.make_offset_plots): dieselbe Figurgroesse, derselbe
Grundstil, dieselben Farben je Groesse, dieselbe Achsen-Buendelung. Neu
sind nur die Richtungen:

    horizontal   durchgezogen   Exponent ->
    vertikal     gestrichelt    Exponent ^
    diagonal     gepunktet      Exponent Pfeil nach rechts oben
    antidiagonal strichpunkt    Exponent Pfeil nach links oben

Die Richtung steckt also im Linienstil UND im Exponenten, die Groesse in
der Farbe. Die Legende hat eine Zeile je Groesse und eine Spalte je
Richtung.

Registrierung in single_beam_report
-----------------------------------
Die Achsen-Buendelung (gruppen_fuer_achsen) und die Achsenbeschriftung
lesen die Tabellen TRACES, TRACE_ORDER, IMMER_ZUSAMMEN und RICHTUNG_STIL
aus single_beam_report. Die Richtungen dieses Moduls werden dort beim
Import EINGETRAGEN (nur hinzugefuegt, nichts ueberschrieben - die Namen
x/y/xy/antixy kommen dort nicht vor). Eine zweite Fassung dieser Logik
waere die schlechtere Loesung: die beiden Plotarten sollen im selben
Dokument nebeneinander stehen koennen.
"""

from datetime import date
from pathlib import Path as FilePath

import numpy as np
import matplotlib.pyplot as plt

from . import paths
from . import atom_offset as ao
from . import position_noise as pn
from . import single_beam as sb
from . import single_beam_report as sbr
from .report import BEST_POINT_STYLE, _entzerre_achse, _finish


# ======================================================================
# Richtungen in single_beam_report eintragen
# ======================================================================
RICHTUNG_EXPONENT = {
    "x": r"\rightarrow",
    "y": r"\uparrow",
    "xy": r"\nearrow",
    "antixy": r"\nwarrow",
}
RICHTUNG_STIL = {"x": "-", "y": "--", "xy": ":", "antixy": "-.", "mittel": "-"}

BASIS_HART = ("uniformity_hart", "crosstalk_hart")
BASIS_GEWICHTET = ("uniformity_weighted", "crosstalk_weighted")
BASIS_PENALTY = ("uniformity_kombi", "crosstalk_kombi", "combined_score")
BASIS_ALLE = BASIS_HART + BASIS_GEWICHTET + BASIS_PENALTY


def _registrieren():
    """Idempotent: traegt die Richtungs-Kurven in die Tabellen von
    single_beam_report ein."""
    for basis in BASIS_ALLE:
        for rid, exp in RICHTUNG_EXPONENT.items():
            key = f"{basis}__{rid}"
            if key in sbr.TRACES:
                continue
            label, einheit, farbe, prozent = sbr.TRACES[basis]
            sbr.TRACES[key] = (sbr._mit_exponent(label, exp), einheit, farbe, prozent)
            sbr.TEXT_LABELS[key] = f"{sbr.TEXT_LABELS[basis]} ({ao.RICHTUNG_TEXT[rid]})"
            sbr.TRACE_ORDER.append(key)
    # Richtungsmittel: <U_w>_phi - eine Kurve je Groesse.
    for basis in BASIS_ALLE:
        key = f"{basis}__mittel"
        if key in sbr.TRACES:
            continue
        label, einheit, farbe, prozent = sbr.TRACES[basis]
        kern = label.strip("$")
        sbr.TRACES[key] = ("$\\langle " + kern + "\\rangle_\\varphi$", einheit, farbe, prozent)
        sbr.TEXT_LABELS[key] = f"{sbr.TEXT_LABELS[basis]} (Richtungsmittel)"
        sbr.TRACE_ORDER.append(key)
    for rid, stil in RICHTUNG_STIL.items():
        sbr.RICHTUNG_STIL.setdefault(rid, stil)
    for basis in BASIS_ALLE:
        gruppe = tuple(f"{basis}__{rid}" for rid in RICHTUNG_EXPONENT)
        if gruppe not in sbr.IMMER_ZUSAMMEN:
            sbr.IMMER_ZUSAMMEN.append(gruppe)


_registrieren()


# ======================================================================
# Namen
# ======================================================================
KLARTEXT = {
    "uniformity_hart": "U_h (hart, Ton-Quadrat)",
    "crosstalk_hart": "eta_h (hart, Pitch-Quadrat)",
    "uniformity_weighted": "U_w (atom-gewichtet)",
    "crosstalk_weighted": "eta_w (atom-gewichtet)",
    "uniformity_kombi": "U_c (Penalty)",
    "crosstalk_kombi": "eta_c (Penalty)",
    "combined_score": "J (Score)",
}
KURZ = {
    "uniformity_hart": "U_h", "crosstalk_hart": "eta_h",
    "uniformity_weighted": "U_w", "crosstalk_weighted": "eta_w",
    "uniformity_kombi": "U_c", "crosstalk_kombi": "eta_c",
    "combined_score": "J",
}


def bereich_tag(results):
    """Namenskuerzel fuer den abgefahrenen Bereich."""
    b = results.get("bereich", {})
    if b.get("art") == "width":
        return f"bis{b['faktor']:.4g}width"
    if b.get("art") == "schwankung":
        return f"bis{b['faktor']:.3g}sigma"
    return f"bis{float(np.max(results['offset_um'])):.4g}um"


def prefix(results, tag=None):
    """AtomOffset_N3x4_Airy_w1.1um_width0.45MHz_rx1_ry1.2_k1.483_bis0.125width_2026-09-10"""
    tag = date.today().isoformat() if tag is None else str(tag)
    return f"AtomOffset_{ao.name_kern(results['params'])}_{bereich_tag(results)}_{tag}"


def bericht_prefix(params, tag=None):
    tag = date.today().isoformat() if tag is None else str(tag)
    return f"AtomOffset_{ao.name_kern(params)}_{tag}"


# ======================================================================
# Kurvenfigur
# ======================================================================
OFFSET_LABEL = r"Atom offset from site centre $r$ [$\mathrm{\mu m}$]"

LEGENDEN_CHOICES = [
    ("unten", "unter dem Plot (Zeile je Groesse, Spalte je Richtung)"),
    ("upper left", "oben links im Plot"),
    ("upper right", "oben rechts im Plot"),
]


def _zweite_achse_width(ax, results):
    d = float(results["width_um"])
    if not np.isfinite(d) or d <= 0:
        return None
    sek = ax.secondary_xaxis(
        "top", functions=(lambda v: np.asarray(v, dtype=float) / d,
                          lambda v: np.asarray(v, dtype=float) * d))
    sek.set_xlabel(r"$r\,/\,d$,  $d$ = array width = %.3f$\,\mathrm{\mu m}$" % d)
    return sek


def _zweite_achse_sigma(ax, results):
    s_um = (results.get("bereich") or {}).get("bezug_m")
    if not s_um:
        return _zweite_achse_width(ax, results)
    s_um = float(s_um) * 1e6
    sek = ax.secondary_xaxis(
        "top", functions=(lambda v: np.asarray(v, dtype=float) / s_um,
                          lambda v: np.asarray(v, dtype=float) * s_um))
    sek.set_xlabel(r"$r\,/\,\sigma_\mathrm{pos}$,  $\sigma_\mathrm{pos}$ = %.1f$\,\mathrm{nm}$"
                   % (s_um * 1e3))
    return sek


def plot_offset(results, basis_keys, filename, out_dir=None, achsen="auto",
                legende="unten", dichte=sbr.SCHRIFT_DICHTE, marker=False,
                markierungen=None, save=True, show=False, confirm_overwrite=None,
                titel=None, band=False):
    """Eine Figur: die Groessen `basis_keys` ueber r, je gerechneter Richtung
    bzw. - im Richtungsmittel - eine Kurve je Groesse.

    markierungen: Liste (r_m, text) - senkrechte Linien, z.B. sigma_pos.
    band: nur im Richtungsmittel - Minimum bis Maximum ueber die Winkel als
        blasse Flaeche in der Farbe der Kurve.
    """
    out_dir = paths.fit_plots_dir() if out_dir is None else out_dir
    richtungen = [r for r in results["richtungen"] if r in RICHTUNG_STIL]
    keys = [f"{b}__{r}" for b in basis_keys for r in richtungen
            if f"{b}__{r}" in results]
    if not keys:
        return None
    # Reihenfolge der Legende: Groesse fuer Groesse, darin die Richtungen.
    werte = sbr.werte_in_prozent(results, keys)
    gruppen = sbr.gruppen_fuer_achsen(keys, werte, modus=achsen)
    x = np.asarray(results["offset_um"], dtype=float)
    zweite = (_zweite_achse_sigma if (results.get("bereich") or {}).get("art") == "schwankung"
              else _zweite_achse_width)

    with sbr.kurven_stil(dichte):
        fig, ax0 = plt.subplots(figsize=sbr.CURVE_FIGSIZE, constrained_layout=True)
        handles = {}
        achsen_liste = []
        for position, gruppe in enumerate(gruppen):
            ax = ax0 if position == 0 else ax0.twinx()
            achsen_liste.append(ax)
            if position > 1:
                ax.spines["right"].set_position(
                    ("outward", sbr.S(sbr.ZUSATZ_ACHSE_PT) * (position - 1)))
                ax.set_frame_on(True)
                ax.patch.set_visible(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.spines["right"].set_visible(True)
            band_werte = []
            for key in gruppe:
                label, _einheit, farbe, prozent = sbr.TRACES[key]
                rid = key.rsplit("__", 1)[-1]
                basis = key.rsplit("__", 1)[0]
                if band and rid == "mittel" and f"{basis}__min" in results:
                    f = 100.0 if prozent else 1.0
                    lo = np.asarray(results[f"{basis}__min"], dtype=float) * f
                    hi = np.asarray(results[f"{basis}__max"], dtype=float) * f
                    ax.fill_between(x, lo, hi, color=farbe, alpha=0.15, linewidth=0,
                                    zorder=1)
                    band_werte += [lo, hi]
                linie, = ax.plot(x, werte[key], color=farbe,
                                 linestyle=RICHTUNG_STIL[rid], linewidth=sbr.S(1.6),
                                 label=label, marker="o" if marker else None,
                                 markersize=sbr.S(2.6))
                handles[key] = linie
            if achsen == "log":
                ax.set_yscale("log")
            else:
                _entzerre_achse(ax, [werte[k] for k in gruppe] + band_werte)
            if legende.startswith("upper"):
                sbr._luft_nach_oben(ax, [werte[k] for k in gruppe] + band_werte,
                                    anteil=0.12 * len(richtungen) + 0.1)
            ax.set_ylabel(sbr._achsen_label(gruppe))

        # Senkrechte Markierungen. Liegen zwei dicht beieinander (sigma_atom
        # und sigma_pos tun das), steht die Beschriftung der linken links
        # von ihrer Linie und die der rechten rechts davon.
        sichtbar = sorted((float(r_m) * 1e6, text) for r_m, text in (markierungen or [])
                          if x[0] < float(r_m) * 1e6 <= x[-1])
        for i, (r_um, text) in enumerate(sichtbar):
            ax0.axvline(r_um, color=BEST_POINT_STYLE["color"], linestyle=(0, (4, 3)),
                        linewidth=sbr.S(1.0), alpha=0.75, zorder=1.5)
            links = len(sichtbar) > 1 and i < len(sichtbar) - 1
            ax0.text(r_um, 0.97, (text + " ") if links else (" " + text),
                     transform=ax0.get_xaxis_transform(),
                     color=BEST_POINT_STYLE["color"], ha="right" if links else "left",
                     va="top", fontsize=plt.rcParams["legend.fontsize"])

        ax0.set_xlim(x[0], x[-1])
        ax0.set_xlabel(OFFSET_LABEL)
        if titel:
            ax0.set_title(titel)
        ax0.grid(True, alpha=0.25)
        sek = zweite(ax0, results)

        # Zeile je Groesse, Spalte je Richtung - so bleibt die Legende unter
        # dem Plot zwei bis drei Zeilen hoch. matplotlib fuellt spaltenweise,
        # also Richtung fuer Richtung uebergeben.
        reihenfolge = [f"{b}__{r}" for r in richtungen for b in basis_keys
                       if f"{b}__{r}" in handles]
        # Eine Kurve je Groesse (Richtungsmittel oder nur eine Richtung):
        # alle nebeneinander in einer Zeile.
        ncol = len(richtungen) if len(richtungen) > 1 else max(len(reihenfolge), 1)
        if legende == "unten":
            fig.legend([handles[k] for k in reihenfolge],
                       [handles[k].get_label() for k in reihenfolge],
                       loc="outside lower center", ncol=ncol, framealpha=0.95)
        else:
            leg = ax0.legend([handles[k] for k in reihenfolge],
                             [handles[k].get_label() for k in reihenfolge],
                             loc=legende, ncol=ncol, framealpha=0.95)
            leg.set_in_layout(False)

        if sek is not None:
            sbr._ticks_entzerren(fig, sek)

        return _finish(fig, out_dir, filename, save, show, confirm_overwrite)


def make_plots(results, sigma_pos_m=None, sigma_atom_m=None, out_dir=None,
               getrennt=True, penalty_plot=False, achsen="auto", legende="unten",
               dichte=sbr.SCHRIFT_DICHTE, marker=False, save=True, show=False,
               confirm_overwrite=None, hart=False, band=False):
    """Gewichtet ueber r (immer), hart nur mit hart=True, die Penalty-Groessen
    (U_c, eta_c, J) nur mit penalty_plot=True. sigma_pos/sigma_atom werden nur markiert, wenn sie
    uebergeben werden. Uniformity und Crosstalk bleiben immer zusammen in
    einer Figur."""
    pfx = prefix(results)
    markierungen = []
    if sigma_pos_m:
        markierungen.append((sigma_pos_m, r"$\sigma_\mathrm{pos}$"))
    if sigma_atom_m:
        markierungen.append((sigma_atom_m, r"$\sigma_\mathrm{atom}$"))
    gemeinsam = dict(out_dir=out_dir, achsen=achsen, legende=legende, dichte=dichte,
                     marker=marker, markierungen=markierungen, save=save, show=show,
                     confirm_overwrite=confirm_overwrite, band=band)
    pfade = []
    if hart and not getrennt:
        pfade.append(plot_offset(results, BASIS_HART + BASIS_GEWICHTET,
                                 f"{pfx}_metrics.pdf", **gemeinsam))
    else:
        pfade.append(plot_offset(results, BASIS_GEWICHTET, f"{pfx}_weighted.pdf", **gemeinsam))
        if hart:
            pfade.append(plot_offset(results, BASIS_HART, f"{pfx}_hard.pdf", **gemeinsam))
    if penalty_plot:
        pfade.append(plot_offset(results, BASIS_PENALTY, f"{pfx}_penalty.pdf", **gemeinsam))
    return [p for p in pfade if p is not None]


# ======================================================================
# Bericht
# ======================================================================
def _p(v, stellen=4):
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:.{stellen}g} %"


def _pp(v):
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:+.3g} pp"


def basis_liste(kombiniert=False, hart=False):
    """Die Groessen, die in Bericht und Tabellen erscheinen. Basisfall sind
    die atom-gewichteten; hart und kombiniert kommen nur auf Wunsch dazu."""
    return (BASIS_GEWICHTET + (BASIS_HART if hart else ())
            + (BASIS_PENALTY if kombiniert else ()))


def _ergebnis_tabelle(results, basis=BASIS_ALLE):
    zeilen = ["| Groesse | Richtung | bei r = 0 | am Ende | Aenderung | Verlauf |",
              "|---|---|---|---|---|---|"]
    for b in basis:
        for rid in results["richtungen"]:
            v = np.asarray(results[f"{b}__{rid}"], dtype=float)
            zeilen.append(
                f"| {KLARTEXT[b]} | {ao.RICHTUNG_TEXT[rid]} | {_p(v[0])} | {_p(v[-1])} "
                f"| {_pp(v[-1] - v[0])} | {sb.monotone_kind(v)} |")
    return "\n".join(zeilen)


def _werte_tabellen(results, n_zeilen=11, basis=BASIS_ALLE):
    r_um = np.asarray(results["offset_um"], dtype=float)
    idx = np.unique(np.linspace(0, r_um.size - 1, min(n_zeilen, r_um.size)).astype(int))
    bloecke = []
    for rid in results["richtungen"]:
        kopf = ("| r (µm) | r/d | " + " | ".join(f"{KURZ[b]} (%)" for b in basis) + " |")
        zeilen = [f"**{ao.RICHTUNG_TEXT[rid]}**", "", kopf,
                  "|" + "---|" * (len(basis) + 2)]
        for i in idx:
            zeilen.append(f"| {r_um[i]:.4f} | {r_um[i] / results['width_um']:.4f} | "
                          + " | ".join(f"{results[f'{b}__{rid}'][i] * 100:.4g}"
                                       for b in basis) + " |")
        bloecke.append("\n".join(zeilen))
    return "\n\n".join(bloecke)


def _abschaetzung_block(ab, umfang, k_faktor):
    zeilen = ["| Mechanismus | Eingabe | Hebel | Beitrag (1 sigma) | geht ein | Quelle/Hinweis |",
              "|---|---|---|---|---|---|"]
    for b in ab["beitraege"]:
        ein = {"thermisch": "thermisch", "technisch": "technisch", "info": "nein"}[b["art"]]
        zeilen.append(f"| {b['name']} | {b['eingabe']} | {b['hebel']} "
                      f"| {b['wert_m'] * 1e9:.3g} nm | {ein} | {b['quelle']} |")
    sigma = {"gesamt": ab["sigma_gesamt_m"], "technisch": ab["sigma_tech_m"],
             "thermisch": ab["sigma_therm_max_m"]}[umfang]
    return "\n".join(zeilen) + f"""

- thermisch (unguenstiger Rand): **{ab['sigma_therm_max_m'] * 1e9:.1f} nm** (nominal {ab['sigma_therm_nom_m'] * 1e9:.1f} nm; Nullpunktsbreite {ab['nullpunkt_m'] * 1e9:.1f} nm, mittlere Besetzung n = {ab['n_mittel']:.2f})
- technisch (quadratisch addiert): **{ab['sigma_tech_m'] * 1e9:.1f} nm**
- gesamt: **{ab['sigma_gesamt_m'] * 1e9:.1f} nm**
- benutzt fuer Plot 2 und Statistik ({umfang}): **sigma_pos = {sigma * 1e9:.1f} nm**, Plot 2 bis {k_faktor:g} sigma_pos = {k_faktor * sigma * 1e9:.1f} nm

Alle Beitraege sind 1 sigma JE ACHSE. Fuer eine isotrope 2D-Gaussverteilung
liegt der Abstand r von der Site-Mitte mit 39.3 % Wahrscheinlichkeit unter
1 sigma, mit 86.5 % unter 2 sigma und mit 98.9 % unter 3 sigma
(Rayleigh-Verteilung). Bis {k_faktor:g} sigma sind es {pn.anteil_innerhalb(k_faktor) * 100:.1f} %.
"""


def _stellen_tabelle(results, abstaende, basis=BASIS_ALLE):
    """Werte bei bestimmten Abstaenden (interpoliert) gegen r = 0."""
    zeilen = ["| Groesse | Richtung | r = 0 | " + " | ".join(t for _r, t in abstaende) + " |",
              "|---|---|---|" + "---|" * len(abstaende)]
    werte_je = [(ao.werte_bei(results, r), t) for r, t in abstaende]
    for b in basis:
        for rid in results["richtungen"]:
            key = f"{b}__{rid}"
            v0 = float(results[key][0])
            zellen = []
            for w, _t in werte_je:
                v = w[key]
                zellen.append("ausserhalb" if not np.isfinite(v)
                              else f"{v * 100:.4g} % ({(v - v0) * 100:+.3g} pp)")
            zeilen.append(f"| {KLARTEXT[b]} | {ao.RICHTUNG_TEXT[rid]} | {v0 * 100:.4g} % | "
                          + " | ".join(zellen) + " |")
    return "\n".join(zeilen)


def _statistik_tabelle(st, basis=BASIS_ALLE):
    zeilen = ["| Groesse | Atom auf der Site | Mittelwert | Streuung (1 sigma) | Mittel - Site | Spanne der Knoten |",
              "|---|---|---|---|---|---|"]
    for b in basis:
        s = st[b]
        zeilen.append(
            f"| {KLARTEXT[b]} | {s['null'] * 100:.4g} % | {s['mittel'] * 100:.4g} % "
            f"| {s['streuung'] * 100:.3g} pp | {(s['mittel'] - s['null']) * 100:+.3g} pp "
            f"| {s['min_knoten'] * 100:.4g} .. {s['max_knoten'] * 100:.4g} % |")
    return "\n".join(zeilen)


def _symmetrie_tabelle(sym, basis=BASIS_ALLE):
    namen = list(sym)
    zeilen = ["| Groesse | " + " | ".join(namen) + " |",
              "|---|" + "---|" * len(namen)]
    for b in basis:
        zeilen.append(f"| {KURZ[b]} | "
                      + " | ".join(f"{sym[n][b] * 100:.4g}" for n in namen) + " |")
    return "\n".join(zeilen)


def _abgleich_tabelle(null, ref):  # nur mit harten Metriken aufgerufen
    if ref is None:
        return "Der Optimierer lieferte bei r = 0 kein Ergebnis - kein Abgleich moeglich."
    zeilen = ["| Groesse | dieses Skript (Gitter auf der Region) | Optimierer (globales Gitter, n_grid = 1000) | Differenz |",
              "|---|---|---|---|"]
    for k in ("uniformity_hart", "crosstalk_hart", "uniformity_weighted", "crosstalk_weighted"):
        zeilen.append(f"| {KLARTEXT[k]} | {null[k] * 100:.4f} % | {ref[k] * 100:.4f} % "
                      f"| {(null[k] - ref[k]) * 100:+.4f} pp |")
    return "\n".join(zeilen)


def _richtungen_text(lauf, erster):
    if (erster or {}).get("modus") == "mittel" or lauf.get("modus") == "mittel":
        n_w = int((erster or {}).get("winkel_rad", np.zeros(lauf.get("n_winkel", 6))).size)
        return (f"Mittel ueber die Richtung ({n_w} Winkel im Halbkreis, "
                f"alle {180.0 / max(n_w, 1):g} Grad)")
    return ", ".join(ao.RICHTUNG_TEXT[r] for r in (erster or {}).get("richtungen", ()))


def _spanne_tabelle(results, basis, stellen):
    """Richtungsmittel: Minimum, Mittel und Maximum ueber die Winkel an
    ausgewaehlten Abstaenden (interpoliert zwischen den Stuetzstellen)."""
    r = np.asarray(results["offset"], dtype=float)
    kopf = "| Groesse | " + " | ".join(t for _r, t in stellen) + " |"
    zeilen = [kopf, "|---|" + "---|" * len(stellen)]
    for b in basis:
        zellen = []
        for r_m, _t in stellen:
            if r_m > r[-1] + 1e-15:
                zellen.append("ausserhalb")
                continue
            mi = float(np.interp(r_m, r, results[f"{b}__min"])) * 100
            mw = float(np.interp(r_m, r, results[f"{b}__mittel"])) * 100
            ma = float(np.interp(r_m, r, results[f"{b}__max"])) * 100
            zellen.append(f"{mw:.4g} % ({mi:.4g} .. {ma:.4g})")
        zeilen.append(f"| {KLARTEXT[b]} | " + " | ".join(zellen) + " |")
    return "\n".join(zeilen)


def write_report(lauf, out_dir=None, dateiname=None, kombiniert=None, hart=None):
    """Markdown-Bericht zu einem ganzen Lauf (ein oder zwei Sweeps,
    Abschaetzung, Statistik, Symmetrie, Abgleich).

    Basisfall sind die atom-gewichteten Metriken.
    hart: U_h, eta_h mit in die Tabellen. None nimmt lauf["hart"].
    kombiniert: U_c, eta_c und J mit in die Tabellen. None nimmt
        lauf["kombiniert"].
    Die Abschnitte zur Positionsschwankung erscheinen nur, wenn der Lauf
    eine Abschaetzung traegt."""
    out_dir = paths.FIT_RESULTS_DIR if out_dir is None else out_dir
    p = lauf["params"]
    dateiname = f"{bericht_prefix(p)}_Report.md" if dateiname is None else dateiname
    sweeps = lauf.get("sweeps", [])
    erster = sweeps[0] if sweeps else None
    ab = lauf.get("abschaetzung")
    sigma_pos = lauf.get("sigma_pos_m") if ab else None
    if kombiniert is None:
        kombiniert = bool(lauf.get("kombiniert", False))
    if hart is None:
        hart = bool(lauf.get("hart_plot", lauf.get("hart", p.get("hart", True))))
    # U_c/eta_c/J brauchen die harten Metriken zum RECHNEN; ob U_h und eta_h
    # selbst im Bericht stehen, entscheidet trotzdem nur `hart`.
    basis = basis_liste(kombiniert, hart)

    if hart:
        numerik = f"""## Numerik und Abgleich bei r = 0

Die harten Metriken laufen NICHT ueber das globale Gitter des Optimierers:
dort sind die Regionen Masken, deren Rand bei Verschiebungen um Bruchteile
eines Pixels um ganze Pixel springt. Statt dessen liegt ein eigenes Gitter
auf der Region ({p['hard_n_grid']} x {p['hard_n_grid']} Zellmitten, wandert mit dem Atom),
die Kurven sind dadurch glatt in r. Die gewichteten Metriken kommen
unveraendert aus dem Optimierer (atom_offset_x/y).

{_abgleich_tabelle(lauf['null'], lauf.get('referenz')) if lauf.get('abgleich', True) else 'Kein Abgleich (im Dialog abgewaehlt).'}

Die Differenz der harten Groessen ist der Pixelrand des globalen Gitters
(dieselbe Groessenordnung wie das in claude/combinated_optimization.md,
Nachtrag 6, gemessene Saegezahn-Rauschen).
"""
    else:
        numerik = """## Numerik

Die atom-gewichteten Metriken kommen unveraendert aus dem Optimierer
(atom_offset_x/y, lokales Sub-Gitter um das Atom). Harte Metriken wurden
nicht gerechnet, deshalb gibt es auch keinen Abgleich.
"""

    if (erster or {}).get("modus") == "mittel":
        richtung_satz = """Die Kurven sind ueber die RICHTUNG des Versatzes gemittelt: an jedem
Abstand r wird an gleichmaessig verteilten Winkeln im Halbkreis gerechnet
und gemittelt - der Erwartungswert fuer ein Atom, das um r in zufaelliger
Richtung verschoben ist. Der Halbkreis genuegt, weil die Anordnung
punktsymmetrisch ist (+r und -r geben denselben Wert, siehe
Symmetrie-Tabelle). Die kombinierten Groessen werden je Winkel gebildet und
dann gemittelt. Wie stark die einzelnen Richtungen vom Mittel abweichen,
steht je Bereich in der Tabelle "Spannweite ueber die Richtungen"."""
    else:
        richtung_satz = f"""Richtungen: horizontal (r, 0) laeuft entlang der {ao.N_X_FIXED}-Ton-Achse,
vertikal (0, r) entlang der {ao.N_Y_FIXED}-Ton-Achse, diagonal (r, r)/sqrt2 auf
eine Ecke zu, antidiagonal (-r, r)/sqrt2 auf die andere Ecke."""

    regionen = ("das Ton-Quadrat der harten Uniformity (Seitenlaenge "
                f"{lauf['uniformity_side_um']:.4f} µm), das Pitch-Quadrat des harten "
                "Crosstalks und das lokale Sub-Gitter samt Gauss-Gewichtung der "
                "atom-gewichteten Metriken" if hart else
                "das lokale Sub-Gitter samt Gauss-Gewichtung der atom-gewichteten Metriken")
    was = "die atom-gewichteten Metriken (Basisfall)"
    if hart:
        was += ", die harten"
    if kombiniert:
        was += " und die Penalty-Kombination U_c, eta_c, J"

    teile = [f"""# AtomOffset - Atom gegen das Multitone-Profil verschoben, {date.today().isoformat()}

Fester Parametersatz, das ATOM wandert radial von der Site-Mitte nach
aussen. Ausgewertet werden {was}.

| Groesse | Wert |
|---|---|
| Toene | {ao.N_X_FIXED} x {ao.N_Y_FIXED} (fest) |
| Linsen | f1 = {ao.F1_FIXED * 1e3:.0f} mm, f2 = {ao.F2_FIXED * 1e3:.0f} mm (fest), fLO = {lauf['fLO'] * 1e3:.2f} mm |
| Waist (Atomebene) | {p['waist'] * 1e6:.4f} µm |
| width | {p['width'] * 1e-6:.4f} MHz = **d = {lauf['width_um']:.4f} µm** in der Atomebene (Spannweite des Tonarrays) |
| r_x / r_y | {p['r_x']:.4f} / {p['r_y']:.4f} |
| Profil | {p['profile']}, airy_scale_factor = {p['airy_scale_factor']:.6g} |
| Kohaerenz | {"an" if p['coherent'] else "aus"} (entartete Paare: {lauf['n_degenerate_pairs']}) |
| pitch | {lauf['pitch'] * 1e6:.4f} µm |
| sigma_atom (Gewicht) | {lauf['sigma_atom'] * 1e9:.2f} nm (T = {p['atom_temperature'] * 1e6:.2f} µK, nu_r = {p['trap_freq_r'] * 1e-3:.2f} kHz) |
| harte Metriken | {"ja" if hart else "nein"} |
| kombinierte Groessen | {f"ja (alpha = {p['alpha']:.3f}, combo_lambda = {p['combo_lambda']:.3f})" if kombiniert else "nein"} |
| Positionsschwankung | {"ja" if ab else "nein"} |
| Richtung | {_richtungen_text(lauf, erster)} |

## Was sich bewegt und was steht

Bewegt wird das Atom - und mit ihm ALLE Regionen: {regionen}. Das
Lichtfeld (12 Spots der Site, 8 Nachbarkopien im Abstand pitch) steht. Das
ist exakt dasselbe wie eine gemeinsame Drift des ganzen Raman-Profils um -r
gegen ein ruhendes Atom.

{richtung_satz}

{numerik}"""]

    if ab:
        teile.append(f"""## Positionsschwankung des Atoms - Abschaetzung

{_abschaetzung_block(ab, lauf['umfang'], lauf['k_faktor'])}
Die Hebelarme folgen aus der Optik des Optimierers
(r = f1 fLO/f2 tan(theta), theta = theta_max (f - offset)/f_band); die
Formeln stehen im Kopf von lib/position_noise.py. Technische Eingaben, die
auf 0 stehen, sind nicht gemessen - der Hebel daneben sagt, wieviel ein
Mikroradiant, ein ppm oder ein Hz ausmachen wuerde.
""")

    for res in sweeps:
        b = res["bereich"]
        r_um = np.asarray(res["offset_um"], dtype=float)
        abstaende = []
        if sigma_pos:
            for k in (1.0, 2.0, 3.0):
                if k * sigma_pos * 1e6 <= r_um[-1] + 1e-12:
                    abstaende.append((k * sigma_pos,
                                      f"{k:g} sigma_pos = {k * sigma_pos * 1e9:.0f} nm"))
        stellen = (f"""### Werte bei den Abstaenden der Positionsschwankung

{_stellen_tabelle(res, abstaende, basis)}

""" if abstaende else "")
        abgebr = "\n\n**Der Lauf wurde abgebrochen - fehlende Werte sind NaN.**" if res.get("abgebrochen") else ""
        spanne = ""
        if res.get("modus") == "mittel":
            orte = [(r_m, t) for r_m, t in abstaende]
            orte.append((r_um[-1] * 1e-6, f"Ende, r = {r_um[-1] * 1e3:.0f} nm"))
            spanne = (f"### Spannweite ueber die Richtungen\n\n"
                      f"Mittel ueber die Winkel, in Klammern Minimum .. Maximum.\n\n"
                      f"{_spanne_tabelle(res, basis, orte)}\n\n")
        teile.append(f"""## Versatz {b['text']}

- Bereich: 0 .. {r_um[-1]:.4f} µm = {r_um[-1] / res['width_um']:.4f} d, {r_um.size} Stuetzstellen je Richtung
- Rechenzeit: {res.get('dauer_s', float('nan')):.1f} s{abgebr}

### Ergebnis je Groesse

{_ergebnis_tabelle(res, basis)}

{stellen}{spanne}### Werte

{_werte_tabellen(res, basis=basis)}
""")

    st = lauf.get("statistik")
    if st:
        teile.append(f"""## Statistik ueber die Positionsverteilung

Das Atom sitzt isotrop gaussverteilt mit sigma_pos = {st['sigma'] * 1e9:.1f} nm je
Achse um die Site-Mitte. Mittelwert und Streuung jeder Groesse aus einer
Gauss-Hermite-Quadratur mit {st['ordnung']} x {st['ordnung']} = {st['n_auswertungen']} Positionen
(deterministisch, keine Zufallszahlen). "Streuung" ist die
Schuss-zu-Schuss-Schwankung der Groesse, wenn jeder Schuss eine neue
Position zieht.

{_statistik_tabelle(st, basis)}

Genauigkeit: U_w ist ueber der Verteilung am staerksten nichtlinear (steigt
steil an und flacht dann ab), dort konvergiert die Quadratur langsamer als
bei den uebrigen Groessen. Am Standard-Arbeitspunkt gemessen:
7 -> 11 Knoten aendert die Streuung von U_w um 0.02 pp, die aller anderen
Groessen um weniger als 0.001 pp.

Hinweis zu den atom-gewichteten Groessen: in U_w und eta_w steckt die
thermische Ortsverteilung bereits als Gewicht. Enthaelt sigma_pos den
thermischen Anteil ("gesamt"/"thermisch"), ist er dort doppelt gezaehlt -
fuer diese beiden ist "technisch" die saubere Wahl.
""")

    sym = lauf.get("symmetrie")
    if sym:
        teile.append(f"""## Symmetrie (bei r = {lauf['symmetrie_r_m'] * 1e6:.4f} µm, Werte in %)

{_symmetrie_tabelle(sym, basis)}

+r und -r stimmen je Achse ueberein, ebenso gegenueberliegende Punkte
einer Diagonale - die Richtung mit positivem Vorzeichen steht also fuer
beide.
Diagonale und Antidiagonale sind dagegen NICHT gleich, sobald Kohaerenz
an ist: die frequenzentarteten Eckspots (links oben, rechts unten)
interferieren statisch, das Muster ist in dieser Richtung ein anderes.
""")

    dateien = lauf.get("dateien") or []
    if dateien:
        teile.append("## Dateien\n\n" + "\n".join(f"- `{FilePath(d).name}`" for d in dateien) + "\n")

    out_path = FilePath(out_dir) / dateiname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(teile), encoding="utf-8")
    print(f"Bericht gespeichert: {out_path}")
    return out_path
