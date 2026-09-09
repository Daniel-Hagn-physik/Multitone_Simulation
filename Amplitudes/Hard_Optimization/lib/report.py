"""
lib/report.py - Plots (Vektor-PDF) und Markdown-Bericht zu einem Datensatz.

Wird von run_plots.py benutzt und laesst sich auch aus eigenen Skripten
aufrufen (make_all(...)).

Diese Datei ist in Hard_Optimization/lib und Weighted_Optimization/lib
IDENTISCH. Alles Ordner-Spezifische kommt aus paths.py (welches Plot-Modul,
welche Metrik-Familie, welcher Dateinamens-Praefix); die Rechenlogik steckt
in scan_data.py.

Dateinamens-Praefix (Konvention des Projekts):

    HardScan_N{Nx}x{Ny}_{n_win}x{n_width}pts_{Airy|Gauss}_{Datum}
    WeightedScan_N{Nx}x{Ny}_{n_win}x{n_width}pts_{Airy|Gauss}_{Datum}

gefolgt von

    _metric_comparison.pdf       Uniformity und Crosstalk nebeneinander
    _metric_comparison_amp.pdf   dieselben zwei plus r_x und r_y (2x2)
    _region.pdf                  Score-Karte mit Arbeitspunkt
    _valley_{X}_over_{Y}.pdf     Querschnitt entlang des Minimums von X
    _line_{X}_over_{Y}.pdf       Querschnitt entlang der Fit-Geraden
    _point_cuts.pdf              Kreuzschnitt DURCH den markierten Punkt
                                 (r_x und r_y bei fester width bzw. festem Waist)
    _Report.md                   Bericht mit allen Kennzahlen

Der Talschnitt (_valley_...) ist ein Querschnitt entlang des Minimums:
einer Groesse wird gefolgt, und genau an deren Minimum pro Spalte bzw.
Zeile werden alle uebrigen Groessen abgelesen. Optional wird durch den
brauchbaren Teil dieses Talpfads eine Gerade gelegt (unbrauchbare Punkte
werden automatisch ausgeschlossen) - und der Schnitt laesst sich statt
entlang des Minimums auch entlang genau dieser Geraden legen, ueber den
ganzen gescannten Bereich hinweg (_line_...).

Das Verfahren ist buchstabengleich das aus
Combinated_Optimization/lib/report.py, nur mit EINEM Metrik-Paar statt
zweien - damit die Ergebnisse der drei Ordner direkt vergleichbar sind.
"""

import contextlib
from datetime import date

import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

from . import paths
from . import scan_data
from .scan_data import (
    FORBIDDEN_FACTOR_DEFAULT, dataset_kind, forbidden_boundary, forbidden_mask,
    metric_grids, has_amplitudes, score_from, waist_um_vals,
)
from .paths import (
    AmplitudeScanPlotter, FixedScanPlotter, resolve_save_path, win_input_to_win,
)


# ======================================================================
# EIN Massstab fuer alle PDFs dieses Projekts
# ======================================================================
# Alle Grafiken sind zum Einbinden mit \includegraphics[width=\textwidth]
# gedacht (A4, 2.5-cm-Raender -> 16 cm = 6.3 Zoll). Eine Figur, die breiter
# angelegt ist, wird dabei VERKLEINERT - und mit ihr Schrift und Linien.
#
# Genau das war das Problem: dieselben 11 pt kamen im Dokument je nach Plot
# als 4.7 bis 11.7 pt an (gemessen ueber alle Plot-Arten der drei Ordner,
# Faktor 2.5), weil die PDFs 6.2 bis 14.8 Zoll breit sind. Ein Talschnitt war
# im Text unlesbar, eine Metrik-Karte groesser als die Grundschrift.
#
# Loesung: die Zielgroessen stehen EINMAL, und zwar so, wie sie IM DOKUMENT
# ankommen sollen (DOC_RC). Beim Zeichnen werden sie mit dem Faktor
# Figurbreite/Textbreite vorskaliert - eine 13-Zoll-Figur bekommt also die
# doppelte Schrift und doppelt dicke Linien, damit nach dem Verkleinern
# wieder genau DOC_RC herauskommt. Die Einzeldatei sieht dadurch
# "grossschriftig" aus; im Dokument stimmt es.
#
# Dieselbe Datei liegt in Hard_Optimization/lib und Weighted_Optimization/lib;
# Combinated_Optimization/lib/report.py benutzt denselben Block, damit die
# drei Ordner im selben Dokument nicht auseinanderfallen.
TEXT_WIDTH_IN = 6.3

# So sollen die Grafiken im Dokument ankommen (Grundschrift dort ~11 pt).
DOC_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # Groessen in pt - werden vorskaliert
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.titlesize": 11.0,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "grid.linewidth": 0.8,
    "patch.linewidth": 0.8,
    "hatch.linewidth": 1.0,
}

# Welche rcParams eine Laenge in Punkten sind und deshalb mitskalieren.
_SKALIERBARE_RC = tuple(k for k in DOC_RC if k not in (
    "font.family", "mathtext.fontset", "pdf.fonttype", "ps.fonttype"))

# Aktueller Vorskalierungsfaktor. Wird von dokument_stil() gesetzt und von
# S()/SD() gelesen - die Zeichenfunktionen geben ihn nicht durch, weil sie
# sonst alle einen zusaetzlichen Parameter braeuchten (draw_best_point_marker,
# draw_forbidden_region, draw_fit_line_on_map, ...). Es wird immer genau eine
# Figur zur Zeit gezeichnet, deshalb reicht ein Modul-Zustand.
_SKALA = 1.0


def _skalierte_rc(faktor, **ueberschreiben):
    rc = dict(DOC_RC)
    for key in _SKALIERBARE_RC:
        rc[key] = DOC_RC[key] * faktor
    for key, wert in ueberschreiben.items():
        if wert is None:
            continue
        rc[key] = wert * faktor if key in _SKALIERBARE_RC else wert
    return rc


def _skalierte_rc_dotted(faktor, ueberschreiben):
    """Wie _skalierte_rc(), aber mit einem fertigen dict statt Schluesselworten."""
    return _skalierte_rc(faktor, **ueberschreiben)


@contextlib.contextmanager
def dokument_stil(fig_width_in, legend_fontsize=None, dichte=1.0,
                  **ueberschreiben):
    """Zeichen-Kontext fuer eine Figur dieser Breite.

    Setzt die rcParams auf DOC_RC * dichte * (fig_width / TEXT_WIDTH_IN) und
    merkt sich den Faktor fuer S()/SD().

    dichte < 1 setzt die Schrift EINE STUFE KLEINER, gemessen am Plot - fuer
    Figuren, die sich die Textbreite auf zwei Panels aufteilen (siehe
    ZWEI_PANEL_DICHTE). Nicht kosmetisch: mit voller Groesse ueberragte die
    Legende die Karte, um die es geht.

    legend_fontsize (aus dem Dialog) ist ebenfalls eine DOKUMENT-Groesse und
    wird mitskaliert; weitere rcParams lassen sich unter ihrem gepunkteten
    Namen als Schluesselwort mit Unterstrichen uebergeben
    (axes_titlesize=... -> "axes.titlesize")."""
    global _SKALA
    faktor = float(dichte) * float(fig_width_in) / TEXT_WIDTH_IN
    rc_namen = {k.replace(".", "_"): k for k in DOC_RC}
    ueber = {rc_namen.get(k, k.replace("_", ".")): v
             for k, v in ueberschreiben.items()}
    if legend_fontsize is not None:
        ueber["legend.fontsize"] = legend_fontsize
    vorher = _SKALA
    _SKALA = faktor
    try:
        with plt.rc_context(_skalierte_rc(faktor, **ueber)):
            yield faktor
    finally:
        _SKALA = vorher


def S(laenge):
    """Eine in Dokument-Punkten gedachte Laenge auf die aktuelle Figur
    umrechnen (Linienbreiten, Markergroessen)."""
    return laenge * _SKALA


def SA(flaeche):
    """Wie S(), aber fuer scatter(s=...): das ist eine FLAECHE in pt^2, die
    also quadratisch mitwachsen muss, sonst schrumpfen die Punkte relativ."""
    return flaeche * _SKALA ** 2


def SD(stil, **ueberschreiben):
    """Dasselbe fuer ein Stil-dict: alle Laengen-Eintraege mitskalieren."""
    laengen = ("linewidth", "markersize", "markeredgewidth", "elinewidth",
               "capsize", "capthick", "width")
    neu = {k: (v * _SKALA if k in laengen and isinstance(v, (int, float)) else v)
           for k, v in stil.items()}
    if isinstance(stil.get("s"), (int, float)):
        neu["s"] = SA(stil["s"])
    neu.update(ueberschreiben)
    return neu


# Die Figurgroessen. Breite = Textbreite, wo es geht; die zweispaltigen
# Querschnitte bleiben breiter (sie brauchen den Platz) und werden ueber
# dokument_stil() ausgeglichen.
HALF_PAGE_FIGSIZE = (6.3, 6.2)   # 2x2 Karten
ROW_FIGSIZE = (6.3, 3.2)         # 1x2 Karten
REGION_FIGSIZE = (6.3, 4.4)      # eine Karte

# Platz, den jede zusaetzliche y-Achse im Querschnitt rechts braucht - Zahlen,
# Teilstriche und die gedrehte Beschriftung, gemessen in DOKUMENT-Punkten.
# Steht hier oben, weil zwei Stellen ihn brauchen: die Breite der Figur
# (plot_valley_cut) und der Versatz der Achsen selbst.
ZUSATZ_ACHSE_PT = 40.0

# Zweispaltige Figuren (Talschnitt: Karte + Schnitt; Kreuzschnitt: zwei
# Schnitte) teilen sich die Textbreite auf zwei Panels. Jedes Panel ist damit
# etwa halb so breit wie eine einzelne Karte - die Beschriftung darf dort eine
# Stufe kleiner stehen, so wie in LaTeX eine Subfigure ihre Bildunterschrift
# kleiner setzt als die Hauptabbildung.
#
# Das ist nicht Geschmack: mit der vollen Dokumentgroesse war die Legende der
# Karte breiter als die Karte selbst und ueberragte den eigentlichen Plot.
# 0.7 ist gemessen die Grenze, ab der die Legende in die Karte passt und
# gleichzeitig im Dokument noch lesbar bleibt (Grundschrift 6.3 pt,
# Titel 7 pt bei Textbreite).
# Seit die Legenden der beiden zweispaltigen Figuren (Talschnitt,
# Kreuzschnitt) nicht mehr IM Bild stehen, ueberragen sie die Panels nicht
# mehr - der Grund fuer die kleinere Stufe ist damit entfallen, beide zeichnen
# jetzt mit dichte=1.0. Die Konstante bleibt fuer eigene Skripte stehen.
ZWEI_PANEL_DICHTE = 0.70

# Querschnitt: Karte und Schnitt nebeneinander, auf voller Textbreite.
#
# Diese Geometrie ist in Hard_, Weighted_ und Combinated_Optimization
# ABSICHTLICH DIESELBE (6.3 x 3.6 Zoll, zwei gleich breite Panels, Legenden
# unter den Panels): der Plot "J als Karte + Schnitt daneben" soll aus allen
# drei Ordnern und ueber alle Datensaetze hinweg dasselbe Bild ergeben.
# 6.3 Zoll ist die Textbreite (A4, 2.5-cm-Raender), die Datei wird im
# Dokument also nicht mehr skaliert. Wer hier etwas aendert, aendert es in
# allen drei Ordnern.
VALLEY_FIGSIZE = (6.3, 3.6)
VALLEY_WIDTH_RATIOS = [1.0, 1.0]
# Beide Legenden stehen UNTER ihrem Panel, nicht im Bild: in der Karte
# ueberdeckte die Legende (mit den Koordinaten des Arbeitspunkts) die
# Heatmap, um die es geht.
VALLEY_LEGEND_ANCHOR = (0.5, -0.16)

# Wohin die beiden Legenden des Schnittplots gehoeren.
#
# "below"  - je eine Figur-Legende unter ihrem Panel ("outside"-Ort).
#            constrained_layout reserviert den Platz, die Panels werden
#            entsprechend kleiner, nichts ueberlappt. Voreinstellung.
# "inside" - im Panel unten rechts, eine Stufe kleiner. Spart die Zeilen
#            unter der Figur; ob die Legende die Kurven verdeckt, haengt am
#            Datensatz, deshalb ist es die Wahl des Nutzers und nicht die
#            Voreinstellung.
VALLEY_LEGEND_CHOICES = [
    ("below", "Unter die Panels"),
    ("inside", "Ins Panel, unten rechts (kleinere Schrift)"),
]
VALLEY_LEGEND_PLACEMENT_DEFAULT = "below"
# Beide Legenden des Schnittplots stehen eine Stufe kleiner als die
# Achsenbeschriftung. Nicht kosmetisch, sondern gemessen: in voller Groesse
# stossen die beiden Kaesten unter der Figur in der Mitte aneinander (die
# Koordinaten des Arbeitspunkts stehen in beiden), das "MHz" des linken wurde
# abgeschnitten. Bei 0.85 bleibt in allen geprueften Faellen eine Luecke von
# mindestens 0.15 Figurbreiten; im Panel passt der Kasten damit ebenfalls
# neben die Kurven.
VALLEY_LEGEND_SCALE = 1.0
# ... und sie darf ihr Panel nicht ueberragen. Reicht 0.85 dafuer nicht (der
# Eintrag des Arbeitspunkts ist mit seinen Koordinaten der laengste), wird die
# Schrift so weit nachgezogen, dass der Kasten hineinpasst - bis zu dieser
# Untergrenze, gemessen an der Ausgangsgroesse.
VALLEY_LEGEND_MIN_SCALE = 0.75
# Absolute Untergrenze der Legendenschrift. Die Zeilenaufteilung hat
# Vorrang: passt eine Zeile nur kleiner, wird sie kleiner gesetzt - aber
# nie unter diesen Wert, sonst waere sie gedruckt nicht mehr lesbar.
VALLEY_LEGEND_MIN_PT = 6.0

WORKING_POINT_LABEL = "Working point"

WORKING_POINT_LINE_STYLE = dict(color="red", linewidth=1.2, linestyle="--",
                                alpha=0.8)


def working_point_label(best, at_edge=False):
    """Legendentext des Arbeitspunkts: seine beiden Koordinaten, sonst nichts.

    Wo er liegt, soll man ablesen koennen, ohne im Bericht nachzuschlagen.
    Beide Koordinaten, auch wenn nur eine vorgegeben wurde - die andere kommt
    aus der Geraden und ist genauso ein Ergebnis. Das Wort "Working point"
    steht bewusst nicht davor: ein einzelner roter Stern ist selbsterklaerend,
    und die Legende soll die Zahlen tragen.
    """
    # `at_edge` steuert nur noch die FORM des Sterns (offen statt gefuellt);
    # der Zusatz "(at scan edge)" stand frueher auch im Legendentext und
    # machte ihn doppelt so lang - dass der Punkt am Rand liegt, sagen der
    # offene Stern und der Bericht.
    if not best or best.get('width') is None or best.get('waist_um') is None:
        return WORKING_POINT_LABEL
    # Waist auf zwei Nachkommastellen: das Gitter ist feiner, die dritte
    # Stelle unterscheidet aber keine Arbeitspunkte mehr, die man von Hand
    # ansteuern wuerde - und der Text steht in jeder Legende.
    return (r"$\omega$ = %.2f $\mu$m, width = %.3f MHz"
            % (best['waist_um'], best['width'] * 1e-6))


def koordinaten_label(axis, wert):
    """Eine einzelne Koordinate in der Schreibweise der Legenden:
    "omega = 1.05 um", "omega_in = 1.70 mm" oder "width = 0.370 MHz".
    Der Waist auf zwei Nachkommastellen, die width auf drei."""
    if axis == "width":
        return r"width = %.3f MHz" % float(wert)
    if axis == "waist_mm":
        return r"%s = %.3f mm" % (WIN_INPUT_SYMBOL, float(wert))
    return r"%s = %.2f $\mu$m" % (WAIST_UM_SYMBOL, float(wert))


def working_point_axis_label(best, axis):
    """Beschriftung des Arbeitspunkts in einem SCHNITT.

    Dort ist von ihm nur die eine Koordinate der Schnittachse uebrig - der
    senkrechte Strich markiert genau sie -, und nur die steht deshalb in der
    Legende. Die vollstaendigen Koordinaten traegt der Stern in der Karte
    daneben. Ausgeschrieben waere der Eintrag laenger als alle Kurvennamen
    zusammen und der Legendenkasten breiter als das Panel, in dem er steht."""
    if not best:
        return WORKING_POINT_LABEL
    if axis == "width":
        wert = best.get('width')
        return (WORKING_POINT_LABEL if wert is None
                else koordinaten_label(axis, wert * 1e-6))
    if axis == "waist_mm":
        wert = best.get('win_input')
        return (WORKING_POINT_LABEL if wert is None
                else koordinaten_label(axis, wert * 1e3))
    wert = best.get('waist_um')
    return (WORKING_POINT_LABEL if wert is None
            else koordinaten_label("waist_um", wert))


def draw_working_point_line(ax, best, axis, label=True, text=None):
    """Der Arbeitspunkt in einem KURVENplot: eine senkrechte Linie an seiner
    Stelle auf der Schnittachse.

    In einer Karte ist der Punkt ein Punkt; in einem Schnitt ueber den Waist
    (oder ueber width) ist von ihm nur die eine Koordinate uebrig - also eine
    Linie.

    Der Legendentext ist die Koordinate der Schnittachse - genau die, die der
    Strich markiert (siehe working_point_axis_label). Die vollstaendigen
    Koordinaten traegt der Stern in der Karte. `text` ueberschreibt ihn.
    Gibt die Linie zurueck (oder False, wenn nichts gezeichnet wurde).
    """
    if not best or best.get('width') is None:
        return False
    if axis == "width":
        x = best['width'] * 1e-6
    elif axis == "waist_um":
        x = best.get('waist_um')
    else:
        x = best.get('win_input')
        x = None if x is None else x * 1e3
    if x is None or not np.isfinite(x):
        return False
    beschriftung = text or working_point_axis_label(best, axis)
    return ax.axvline(x, label=(beschriftung if label else "_nolegend_"),
                      **SD(WORKING_POINT_LINE_STYLE))
# Achsenbeschriftung - WORTGLEICH mit Combinated_Optimization/lib/report.py,
# damit Abbildungen aus den drei Ordnern in einem Dokument nebeneinander
# nicht auseinanderfallen. Einheiten stehen in eckigen Klammern; die runden
# bleiben dem erklaerenden Zusatz vorbehalten.
#
# Der Waist NACH den Linsen ist der in der Atomebene wirksame. Frueher hiess
# er "Waist at focus omega' (µm, after lenses)"; der Strich sollte "nach der
# Linse" heissen, sagte das aber nur fuer den, der die Konvention schon kennt.
# "in the atomic plane" sagt es direkt und macht den Strich ueberfluessig.
WAIST_UM_LABEL = r"Waist in the atomic plane $\omega$ [$\mu$m]"
# Kurzform fuer zwei Panels nebeneinander: dort bleiben je rund 2.5 Zoll, und
# der ausgeschriebene Name laeuft ueber den Figurrand hinaus.
WAIST_UM_LABEL_SHORT = r"Waist $\omega$ [$\mu$m]"
WAIST_UM_SYMBOL = r"$\omega$"
WIN_INPUT_LABEL = r"Input waist $\omega_{\mathrm{in}}$ [mm] (before lenses)"
WIN_INPUT_LABEL_SHORT = r"Input waist $\omega_{\mathrm{in}}$ [mm]"
WIN_INPUT_SYMBOL = r"$\omega_{\mathrm{in}}$"

WIDTH_LABEL = "Width [MHz]"

# ----------------------------------------------------------------------
# Beschriftung der Metriken - haengt an der Familie dieses Ordners
# ----------------------------------------------------------------------
_WEIGHTED = (paths.FLAVOR == "weighted")
# Der Index sagt, WELCHE Metrik-Familie gemeint ist: h = harte
# Pitch-Box-Maske, w = atom-gewichtet. Er steht ueberall - in der Legende des
# Querschnitts, an den Colorbars, im Titel der Score-Karte und im Bericht -
# damit beim Nebeneinanderlegen zweier PDFs nie die Frage aufkommt, welches
# jetzt welches war. Dieselbe Konvention benutzt Combinated_Optimization, wo
# beide Familien in EINEM Plot vorkommen.
_IDX = "w" if _WEIGHTED else "h"
U_SYMBOL = r"$U_%s$" % _IDX
C_SYMBOL = r"$\eta_%s$" % _IDX
# Colorbar-Beschriftung der Karten: NUR das Symbol, wie an der y-Achse der
# Schnitte. Was die Groesse ist, steht im Titel ueber der Karte - die
# ausgeschriebene Definition daneben noch einmal zu wiederholen, kostete
# Breite und stand in jeder der vier Karten ein zweites Mal.
U_CBAR = U_SYMBOL + " [%]"
C_CBAR = C_SYMBOL + " [%]"
U_TITLE = (r"Uniformity $U_w$ (atom-weighted)" if _WEIGHTED
           else r"Uniformity $U_h$ (hard metric)")
C_TITLE = (r"Crosstalk $\eta_w$ (atom-weighted)" if _WEIGHTED
           else r"Crosstalk $\eta_h$ (hard metric)")
FAMILY_TITLE = "atom-weighted" if _WEIGHTED else "hard metric"


# ======================================================================
# Wie der Talpunkt je Spalte gewaehlt wird
# ======================================================================
# "global": das kleinste Gitter der Spalte. Einfach - aber wenn das Minimum
#     am Rand des gescannten Fensters oder an der Grenze des verbotenen
#     Bereichs klebt, ist es keins.
#
# "guided": pro Spalte das LOKALE Minimum, das einer LEITGERADEN am
#     naechsten liegt. Die Leitgerade ist der gewoehnliche lineare Fit einer
#     anderen Groesse (Default: Uniformity) auf demselben Datensatz.
#
#     Lokales Minimum heisst hier: beide Nachbarn existieren UND sind
#     groesser. Damit fallen automatisch heraus (a) die Raender des
#     gescannten Fensters und (b) alle Punkte, die direkt an den
#     ausgeschlossenen verbotenen Bereich grenzen.
#
#     WICHTIG UND EHRLICH ZU BENENNEN: die Leitgerade waehlt AUS, sie
#     verschiebt nichts. Die Punkte sind echte lokale Minima der
#     Fuehrungsgroesse, und die Steigung, die herauskommt, ist deren eigene.
#     Trotzdem ist das Verfahren an die Leitgroesse gebunden - welcher der
#     mehreren lokalen Minima-Zweige verfolgt wird, entscheidet sie. Das
#     gehoert in den Bericht und steht dort auch.
VALLEY_SELECT_CHOICES = [
    ("guided", "Lokales Minimum nahe einer Leitgeraden"),
    ("global", "Globales Minimum je Spalte"),
]
GUIDE_FOLLOW_DEFAULT = "uniformity"
GUIDE_HALFWIDTH_DEFAULT = 0.03          # MHz, halbe Korridorbreite


def valley_select_label(select):
    return dict(VALLEY_SELECT_CHOICES).get(select, select)


# Die frueheren Dekoratoren @_mit_stil/@_mit_kartenstil sind entfallen: der
# Stil haengt jetzt an der Figurbreite, die erst INNERHALB der Zeichenfunktion
# feststeht. Jede von ihnen oeffnet deshalb selbst ein dokument_stil(...).


# Am Rand des gescannten Fensters wird der Stern OFFEN gezeichnet: dort ist
# das "Minimum" vermutlich nur der Fensterrand.
BEST_POINT_EDGE_STYLE = dict(marker='*', markerfacecolor='none',
                             markeredgecolor='red', color='red',
                             markersize=16, markeredgewidth=1.6, linestyle='none')
BEST_POINT_STYLE = dict(marker='*', color='red', markersize=16,
                        markeredgecolor='white', markeredgewidth=1.2, linestyle='none')
# Wortgleich mit Combinated_Optimization - derselbe Plot, dieselbe Legende.
FIT_LINE_LABEL = "Linear fit"


# ======================================================================
# Achsen-Hilfen (identische Konvention wie AmplitudeScanPlotter)
# ======================================================================
def win_axis_values(results, win_axis):
    """(x_werte, achsenbeschriftung, umgedreht?) fuer die gewuenschte
    Waist-Konvention. 'after_lens' ist monoton fallend in win_input, daher
    wird dort ggf. umgedreht."""
    win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
    if win_axis == "before_lens":
        return (win_input_vals * 1e3,
                WIN_INPUT_LABEL, False)
    if win_axis == "after_lens":
        x = np.array([win_input_to_win(w, results['f1'], results['f2'],
                                       results['lambda_opt'], results['fLO'])
                      for w in win_input_vals]) * 1e6
        label = WAIST_UM_LABEL
        if len(x) > 1 and x[0] > x[-1]:
            return x[::-1], label, True
        return x, label, False
    raise ValueError(f"win_axis muss 'before_lens' oder 'after_lens' sein, nicht {win_axis!r}.")


# Kurzform derselben Achsenbeschriftung fuer schmale Panels (die Karte im
# Talschnitt). Die lange Fassung ist breiter als das Panel selbst und ragte
# links aus der Figur heraus; welche Konvention gemeint ist, sagt das Symbol
# (omega' = nach den Linsen, omega_in = davor) und die Achse des Schnitts
# daneben, die die lange Fassung weiterhin traegt.
_KURZE_ACHSENLABEL = {
    WIN_INPUT_LABEL: WIN_INPUT_LABEL_SHORT,
    WAIST_UM_LABEL: WAIST_UM_LABEL_SHORT,
}


def kurzes_achsenlabel(label):
    """Kurzform, wenn es eine gibt - sonst unveraendert."""
    return _KURZE_ACHSENLABEL.get(label, label)


def _x_of_index(results, j, win_axis):
    x_vals, _, reversed_ = win_axis_values(results, win_axis)
    n = len(x_vals)
    return x_vals[n - 1 - j] if reversed_ else x_vals[j]


def draw_best_point_marker(ax, results, win_axis, legend=True, best=None,
                           label=None):
    """Markiert den Arbeitspunkt in einer vorhandenen Heatmap-Achse.

    best=None nimmt den von analyse() bestimmten Punkt. Liegt der Punkt am
    Rand des gescannten Fensters, wird der Stern OFFEN gezeichnet - er ist
    dann vermutlich keiner.

    `legend=False`, wenn der Aufrufer die Legende selbst setzt (z.B. eine
    gemeinsame fuer mehrere Panels). `label` steuert davon unabhaengig, ob
    der Stern ueberhaupt einen Legendeneintrag bekommt."""
    if label is None:
        label = legend
    if best is None:
        best = results.get('best') or {}
    if not best or best.get('win_input') is None:
        return
    if best.get('off_grid'):
        # Selbst gewaehlter Punkt: er liegt zwischen den Gitterpunkten -
        # also auch genau dort zeichnen, nicht auf eine Spalte runden.
        x = (best['waist_um'] if win_axis == "after_lens"
             else best['win_input'] * 1e3)
    else:
        win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
        j = int(np.argmin(np.abs(win_input_vals - best['win_input'])))
        x = _x_of_index(results, j, win_axis)
    am_rand = bool(best.get('at_edge'))
    stil = SD(BEST_POINT_EDGE_STYLE if am_rand else BEST_POINT_STYLE)
    best = _mit_waist_um(results, best)
    beschriftung = working_point_label(best, am_rand)
    ax.plot([x], [best['width'] * 1e-6],
            label=(beschriftung if label else "_nolegend_"), **stil)
    handles, _ = ax.get_legend_handles_labels()
    if legend and handles:
        ax.legend(loc="best", framealpha=0.85)


# ======================================================================
# Verbotener Bereich (ueberlappende Eck-Spots)
# ======================================================================
# Herleitung und Formel stehen in lib/scan_data.py. Hier nur das Zeichnen:
# eine Grenzgerade plus schraffierte Flaeche darunter. Schraffur statt
# Volltonflaeche und halbtransparent: die Heatmap darunter soll lesbar
# bleiben.
FORBIDDEN_LINE_STYLE = dict(color="#d62728", linewidth=1.8, linestyle="-")
FORBIDDEN_FILL_STYLE = dict(facecolor="none", edgecolor="#d62728",
                            hatch="///", linewidth=0.0, alpha=0.55)
FORBIDDEN_LABEL = "Corner spots overlap"


def forbidden_curve(results, win_axis, factor=FORBIDDEN_FACTOR_DEFAULT, n=400):
    """(x, y_grenze) der Grenzlinie in den Koordinaten einer Karte.

    Auf der µm-Achse ist die Grenze eine Gerade durch den Ursprung. Auf der
    mm-Achse ist sie es NICHT (waist ~ 1/win_input), die Gerade wird dort zu
    einer Hyperbel - deshalb wird sie dicht abgetastet und als Polygonzug
    gezeichnet."""
    grenze = forbidden_boundary(results, factor)
    if grenze is None:
        return None
    win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
    dicht = np.linspace(win_input_vals.min(), win_input_vals.max(), int(n))
    hilfs = dict(results)
    hilfs['win_input_vals'] = dicht
    waist = waist_um_vals(hilfs)                       # µm
    y = grenze['slope'] * waist                        # MHz
    x = waist if win_axis == "after_lens" else dicht * 1e3
    ordnung = np.argsort(x)
    return x[ordnung], y[ordnung]


def draw_forbidden_region(ax, results, win_axis, factor=FORBIDDEN_FACTOR_DEFAULT,
                          legend=True):
    """Grenzlinie und schraffierte verbotene Flaeche in eine vorhandene
    Karte zeichnen, ohne deren Achsengrenzen zu veraendern."""
    kurve = forbidden_curve(results, win_axis, factor)
    if kurve is None:
        return
    x, y = kurve
    width_vals = np.asarray(results['width_vals'], dtype=float) * 1e-6
    y_unten = float(width_vals.min())
    # Auf den gescannten width-Bereich beschneiden - ohne das zoege die
    # Grenze die y-Achse auf und die Heatmap schrumpfte auf einen Streifen.
    y_oben = float(width_vals.max())
    y_clip = np.clip(y, y_unten, y_oben)
    sichtbar = y > y_unten
    if not np.any(sichtbar):
        return
    ax.fill_between(x, y_unten, y_clip, where=sichtbar,
                    label=(FORBIDDEN_LABEL if legend else "_nolegend_"),
                    **SD(FORBIDDEN_FILL_STYLE))
    innen = (y >= y_unten) & (y <= y_oben)
    if np.any(innen):
        ax.plot(x[innen], y[innen], label="_nolegend_", **SD(FORBIDDEN_LINE_STYLE))


def score_grid(results):
    """Das Score-Gitter - IMMER das rohe alpha*U + (1-alpha)*C, aus den
    gespeicherten Rohgittern nachgerechnet."""
    return _grid_for(results, "score")


# ======================================================================
# Arbeitspunkt - nach frei waehlbarer Groesse
# ======================================================================
BEST_POINT_FOLLOW_STORED = "__stored__"
# Selbst gewaehlter Punkt, drei Varianten:
#
#   ..._WAIST / ..._WIDTH  eine Koordinate wird vorgegeben, die andere kommt
#                          aus der Talpfad-Geraden. Setzt also voraus, dass es
#                          ueberhaupt eine brauchbare Gerade gibt - sonst ist
#                          die zweite Koordinate nicht bestimmbar.
#   ..._BOTH               beide Koordinaten werden vorgegeben. Braucht keine
#                          Gerade und ist deshalb der Rueckfall, wenn keine
#                          zustande kommt.
#
# In allen drei Faellen liegt der Punkt in aller Regel ZWISCHEN den
# Gitterpunkten - er wird deshalb auch dort gezeichnet und nicht auf ein
# Gitter gerundet.
BEST_POINT_MANUAL_WAIST = "__manual_waist__"
BEST_POINT_MANUAL_WIDTH = "__manual_width__"
BEST_POINT_MANUAL_BOTH = "__manual_both__"
# Die Varianten, die eine Talpfad-Gerade brauchen.
MANUAL_LINE_KEYS = (BEST_POINT_MANUAL_WAIST, BEST_POINT_MANUAL_WIDTH)
MANUAL_POINT_KEYS = MANUAL_LINE_KEYS + (BEST_POINT_MANUAL_BOTH,)
MANUAL_POINT_UNITS = {BEST_POINT_MANUAL_WAIST: "µm", BEST_POINT_MANUAL_WIDTH: "MHz"}


def best_point_choices(results):
    """Eintraege fuer das Dialog-Dropdown: der bestimmte Bestpunkt, jede
    Groesse, die der Datensatz hergibt, und die drei Eigenvorgaben."""
    eintraege = [(BEST_POINT_FOLLOW_STORED, "bester Gitterpunkt nach dem Score")]
    for key, label in FOLLOW_CHOICES:
        if _grid_for(results, key) is not None:
            eintraege.append((key, label))
    eintraege += [
        (BEST_POINT_MANUAL_WAIST, "eigener Punkt: nur Waist vorgeben (µm, Rest aus der Geraden)"),
        (BEST_POINT_MANUAL_WIDTH, "eigener Punkt: nur Width vorgeben (MHz, Rest aus der Geraden)"),
        (BEST_POINT_MANUAL_BOTH, "eigener Punkt: Waist UND Width vorgeben"),
    ]
    return eintraege


def win_input_for_waist_um(results, waist_um):
    """Eingangswaist (m) zu einem effektiven Waist in µm.

    Die Kaskade ist reziprok (siehe win_input_to_win), die Umkehrung also
    dieselbe Formel mit vertauschten Rollen. Das Ergebnis wird NICHT
    geglaubt, sondern durch Vorwaertsrechnen mit der Projektfunktion
    gegengeprueft - passt es nicht, gibt es None statt einer falschen Zahl."""
    if not np.isfinite(waist_um) or waist_um <= 0:
        return None
    waist_m = float(waist_um) * 1e-6
    try:
        win_input = ((results['f1'] / results['f2'])
                     * (results['lambda_opt'] * results['fLO']) / (np.pi * waist_m))
        zurueck = win_input_to_win(win_input, results['f1'], results['f2'],
                                   results['lambda_opt'], results['fLO'])
    except (ValueError, ZeroDivisionError, KeyError, TypeError):
        return None
    if not np.isfinite(zurueck) or abs(zurueck - waist_m) > 1e-12 + 1e-9 * abs(waist_m):
        return None
    return float(win_input)


def manual_point_on_line(results, follow, value, fit):
    """Der selbst gewaehlte Punkt auf der Talpfad-Geraden.

    follow = BEST_POINT_MANUAL_WAIST: `value` ist der Waist in µm, die width
        kommt aus der Geraden (width = a*waist + b).
    follow = BEST_POINT_MANUAL_WIDTH: `value` ist die width in MHz, der
        Waist kommt aus der Umkehrung ((width - b)/a).

    None, wenn es keine Gerade gibt oder die Umkehrung nicht moeglich ist
    (Steigung 0). `outside` markiert Punkte ausserhalb des gescannten
    Fensters - dort gibt es keine Daten, der Stern haengt in der Luft."""
    if fit is None or value is None or not np.isfinite(value):
        return None
    a, b = float(fit["a"]), float(fit["b"])
    if follow == BEST_POINT_MANUAL_WAIST:
        waist_um = float(value)
        width_mhz = a * waist_um + b
    elif follow == BEST_POINT_MANUAL_WIDTH:
        if a == 0:
            return None
        width_mhz = float(value)
        waist_um = (width_mhz - b) / a
    else:
        return None
    if not (np.isfinite(waist_um) and np.isfinite(width_mhz)):
        return None
    win_input = win_input_for_waist_um(results, waist_um)
    if win_input is None:
        return None

    return _manual_point(results, follow, waist_um, width_mhz, win_input,
                         given=float(value), fit_a=a, fit_b=b)


def manual_point_free(results, waist_um, width_mhz):
    """Der selbst gewaehlte Punkt aus BEIDEN vorgegebenen Koordinaten.

    Braucht keine Talpfad-Gerade - das ist der Weg, wenn sich fuer den
    Datensatz keine legen laesst (dann waere die zweite Koordinate in
    manual_point_on_line() gar nicht bestimmbar).
    """
    if waist_um is None or width_mhz is None:
        return None
    if not (np.isfinite(waist_um) and np.isfinite(width_mhz)):
        return None
    win_input = win_input_for_waist_um(results, float(waist_um))
    if win_input is None:
        return None
    return _manual_point(results, BEST_POINT_MANUAL_BOTH, float(waist_um),
                         float(width_mhz), win_input)


def _manual_point(results, follow, waist_um, width_mhz, win_input,
                  given=None, fit_a=None, fit_b=None):
    """Gemeinsamer Rueckgabewert aller drei Eigenvorgaben."""
    waist_grid = waist_um_of(results)
    width_grid = np.asarray(results['width_vals'], dtype=float) * 1e-6
    j = int(np.argmin(np.abs(waist_grid - waist_um)))
    i = int(np.argmin(np.abs(width_grid - width_mhz)))
    ausserhalb = bool(waist_um < waist_grid.min() or waist_um > waist_grid.max()
                      or width_mhz < width_grid.min() or width_mhz > width_grid.max())
    return dict(
        follow=follow, given=given,
        # `off_grid`: der Punkt liegt zwischen den Gitterpunkten und wird
        # genau dort gezeichnet, nicht auf eine Scan-Spalte gerundet.
        off_grid=True, on_line=(fit_a is not None),
        waist_um=float(waist_um), width=float(width_mhz) * 1e6,
        win_input=float(win_input),
        row=i, col=j, at_edge=False, outside=ausserhalb,
        nearest_waist_um=float(waist_grid[j]), nearest_width=float(width_grid[i]) * 1e6,
        fit_a=fit_a, fit_b=fit_b,
    )


def _mit_waist_um(results, best):
    """`waist_um` nachtragen, wo es fehlt - Legende und Bericht brauchen es
    in jedem Fall, die Karten sind auf der µm-Achse beschriftet."""
    if best is None or best.get('win_input') is None or 'waist_um' in best:
        return best
    best = dict(best)
    best['waist_um'] = float(_waist_um(results, best['win_input']))
    return best


def best_point_by(results, follow=None, value=None, fit=None, value2=None):
    """Der beste Gitterpunkt nach einer frei gewaehlten Groesse.

    follow=None oder BEST_POINT_FOLLOW_STORED -> der von analyse()
    bestimmte Punkt (results['best']).

    Der Rueckgabewert enthaelt immer `at_edge`: liegt der Punkt auf dem Rand
    des gescannten Fensters, ist er vermutlich gar kein Optimum, sondern nur
    der Rand. Der Plot zeichnet solche Punkte als OFFENEN Stern, der Bericht
    warnt."""
    win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
    width_vals = np.asarray(results['width_vals'], dtype=float)
    if follow == BEST_POINT_MANUAL_BOTH:
        return manual_point_free(results, value, value2)
    if follow in MANUAL_LINE_KEYS:
        return manual_point_on_line(results, follow, value, fit)
    if follow in (None, BEST_POINT_FOLLOW_STORED):
        best = dict(results.get('best') or {})
        if best.get('win_input') is None:
            return None
        if 'at_edge' not in best:
            j = int(np.argmin(np.abs(win_input_vals - best['win_input'])))
            i = int(np.argmin(np.abs(width_vals - best['width'])))
            best['row'], best['col'] = i, j
            best['at_edge'] = bool(i in (0, len(width_vals) - 1)
                                   or j in (0, len(win_input_vals) - 1))
        best['follow'] = BEST_POINT_FOLLOW_STORED
        return best
    grid = _grid_for(results, follow)
    if grid is None:
        return None
    finite = np.isfinite(grid)
    if not np.any(finite):
        return None
    i, j = np.unravel_index(int(np.argmin(np.where(finite, grid, np.inf))), grid.shape)
    return dict(
        win_input=float(win_input_vals[j]), width=float(width_vals[i]),
        row=int(i), col=int(j), value=float(grid[i, j]), follow=follow,
        at_edge=bool(i in (0, grid.shape[0] - 1) or j in (0, grid.shape[1] - 1)),
    )


def output_prefix(results):
    n_win = len(np.asarray(results['win_input_vals']))
    n_width = len(np.asarray(results['width_vals']))
    tag = paths.profile_tag_of(results.get('profile'))
    return (f"{paths.REPORT_STEM}_N{results['N_x']}x{results['N_y']}_"
            f"{n_win}x{n_width}pts_{tag}_{date.today().isoformat()}")


def _finish(fig, out_dir, filename, save, show, confirm_overwrite, tight=True):
    """Speichern und schliessen.

    tight=False speichert in der EINGESTELLTEN Figurgroesse statt auf den
    Inhalt zugeschnitten. Das ist fuer den Schnittplot wichtig: mit
    bbox_inches='tight' haengt die Seitengroesse an der Breite der
    Legendentexte, und die aendert sich mit dem Datensatz (die Koordinaten
    des Arbeitspunkts stehen darin). Zwei Dateien desselben Plots waren so
    verschieden gross - im Dokument nebeneinander sofort sichtbar. Ohne
    'tight' ist die Seite immer exakt VALLEY_FIGSIZE; constrained_layout
    sorgt dafuer, dass Legenden und Beschriftung darin Platz finden."""
    out_path = None
    if save:
        out_path = resolve_save_path(out_dir, filename, confirm_overwrite=confirm_overwrite)
        fig.savefig(out_path, dpi=300, **({'bbox_inches': 'tight'} if tight else {}))
        print(f"Plot gespeichert: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


# ======================================================================
# Metrik-Karten
# ======================================================================
AMPLITUDE_CMAP = "cividis"
# Farbe fuer Gitterpunkte, deren Amplitude auf einer r_bounds-Schranke
# klemmt. Neutrales Grau, also erkennbar KEIN Wert der Skala - das ist die
# Aussage: dort steht kein freies Optimum, sondern die Schranke.
R_CLAMP_COLOR = "#9e9e9e"
R_CLAMP_LABEL = "at optimizer bound $r_{\\mathrm{bounds}}$"
# Toleranz, ab der ein Amplitudenwert als "auf der Schranke" gilt.
R_BOUND_TOL = 1e-6


def metric_panels(results):
    """Die beiden Metrik-Karten dieses Datensatzes."""
    u_key, c_key = scan_data.metric_keys(results)
    return [
        dict(key=u_key, cbar=U_CBAR, title=U_TITLE, cmap="viridis_r", scale=100.0),
        dict(key=c_key, cbar=C_CBAR, title=C_TITLE, cmap="Oranges", scale=100.0),
    ]


def _label_laenge(text):
    """Ungefaehre DARGESTELLTE Laenge eines Legendentextes.

    Mathtext zaehlt anders als der Rohstring: "$\\omega$" sind neun Zeichen
    im Code und eines im Bild. Ohne diese Korrektur waere jeder Eintrag mit
    Formelzeichen viel zu lang geschaetzt.
    """
    s = re.sub(r"\\[a-zA-Z]+", "x", str(text))
    return len(s.replace("$", "").replace("{", "").replace("}", ""))


# Ein unsichtbarer Platzhalter, um eine Legendenzeile aufzufuellen.
_LEER_HANDLE_STIL = dict(linestyle="none", marker="none")


def _legende_fontsize(fontsize=None, scale=VALLEY_LEGEND_SCALE):
    """Schriftgroesse der Legenden im Schnittplot: eine Stufe kleiner als die
    uebergebene bzw. die eingestellte. Die rcParams sind zu diesem Zeitpunkt
    bereits auf die Figurbreite skaliert, der Faktor wirkt also relativ."""
    wert = fontsize
    if wert is None:
        wert = plt.rcParams.get("legend.fontsize") or plt.rcParams.get("font.size")
    try:
        wert = float(wert)
    except (TypeError, ValueError):
        wert = 9.0
    return wert * float(scale)


def _legende_breite(fig, legende, renderer):
    """Breite der Legende als Anteil der Figurbreite."""
    kasten = legende.get_window_extent(renderer)
    return kasten.transformed(fig.transFigure.inverted()).width


def _eine_legende(fig, ax, zeilen, ncol, placement, aussen_loc, groesse):
    """Eine der beiden Legenden des Schnittplots. `zeilen` ist die gewuenschte
    ZEILEN-Aufteilung der Eintraege (z.B. die Kurven in einer Zeile, der
    Arbeitspunkt in der naechsten).

    placement="below": Figur-Legende an `aussen_loc` ("outside lower left"
        bzw. "... right"), mehrspaltig gemaess `zeilen`. constrained_layout
        reserviert den Platz unterhalb der Achsenbeschriftung.
    placement="inside": Achsenlegende unten rechts, mit DERSELBEN
        Zeilenaufteilung wie unter der Figur - U, eta und J stehen auch hier
        nebeneinander. BEWUSST ausserhalb
        der Layout-Rechnung (`set_in_layout(False)`), sonst schruempfte
        constrained_layout das Panel um genau die Flaeche, die die Legende
        darin einnimmt."""
    eintraege = [h for zeile in zeilen for h in zeile]
    if not eintraege:
        return None
    handles, labels = _legende_zeilen(zeilen, ncol)
    if placement == "inside":
        legende = ax.legend(handles, labels, loc="lower right", ncol=ncol,
                            framealpha=0.9, fontsize=groesse)
        legende.set_in_layout(False)
        return legende
    return fig.legend(handles, labels, loc=aussen_loc, ncol=ncol,
                      framealpha=0.9, fontsize=groesse)


def _legenden_zentrieren(fig, paare):
    """Jede Legende waagerecht mittig unter ihr Panel.

    "outside lower left/right" laesst constrained_layout den Platz unter der
    Figur reservieren - richtet den Kasten dann aber am Figurrand aus, nicht
    am Panel; unter der Karte stand er dadurch weit links. Deshalb: einmal
    zeichnen (damit das Layout steht), die Layout-Rechnung einfrieren und die
    Kaesten auf die Mitte ihres Panels schieben. Die Hoehe bleibt, wie das
    Layout sie bestimmt hat.

    Nur fuer die Figur-Legenden; im Panel stehen sie ohnehin an ihrer Ecke."""
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig.set_layout_engine("none")
        for legende, ax in paare:
            if legende is None:
                continue
            kasten = legende.get_window_extent(renderer)
            kasten = kasten.transformed(fig.transFigure.inverted())
            panel = ax.get_position()
            legende.set_loc("lower center")
            legende.set_bbox_to_anchor((panel.x0 + panel.width / 2, kasten.y0),
                                       transform=fig.transFigure)
    except Exception:
        pass


def _schnitt_legenden(fig, ax_map, karten_zeilen, ax_cut, cut_zeilen, ncol,
                      placement, fontsize=None):
    """Beide Legenden des Schnittplots.

    Keine der beiden darf ihr Panel ueberragen: die Kartenlegende schoebe
    sich sonst neben die Karte, und unter der Figur stiessen die beiden
    Kaesten in der Mitte aneinander.

    Passt ein Kasten nicht, wird AUSSCHLIESSLICH die Schrift verkleinert.
    Die uebergebene ZEILENAUFTEILUNG ist gesetzt: U, eta und J stehen
    nebeneinander, und wenn das nur bei kleinerer Schrift geht, dann eben
    bei kleinerer Schrift. Sie untereinander zu setzen zoege die Legende in
    die Laenge und risse eine Luecke zwischen Panel und Kasten.

    JEDE Legende wird nur so weit verkleinert, wie IHR Panel es verlangt -
    vorher diktierte die Kartenlegende (ihr Stern-Eintrag traegt beide
    Koordinaten und ist der laengste Text der Figur) auch die Groesse der
    Schnittlegende.

    Gemessen statt geschaetzt; die Kastenbreite ist in guter Naeherung
    proportional zur Schriftgroesse, ein Durchgang genuegt also."""
    basis = _legende_fontsize(fontsize)
    karte = _eine_legende(fig, ax_map, karten_zeilen, 1, placement,
                          "outside lower left", basis)
    schnitt = _eine_legende(fig, ax_cut, cut_zeilen, ncol, placement,
                            "outside lower right", basis)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        renderer = None

    def passt(legende, ax):
        return _legende_breite(fig, legende, renderer) <= ax.get_position().width

    def weg(legende):
        if placement == "inside":
            legende.remove()
        else:
            fig.legends.remove(legende)

    if renderer is not None and schnitt is not None and not passt(schnitt, ax_cut):
        breite = _legende_breite(fig, schnitt, renderer)
        groesse = max(VALLEY_LEGEND_MIN_PT,
                      basis * ax_cut.get_position().width / breite * 0.97)
        weg(schnitt)
        schnitt = _eine_legende(fig, ax_cut, cut_zeilen, ncol, placement,
                                "outside lower right", groesse)

    if renderer is not None and karte is not None and not passt(karte, ax_map):
        breite = _legende_breite(fig, karte, renderer)
        groesse = max(VALLEY_LEGEND_MIN_PT,
                      basis * ax_map.get_position().width / breite * 0.97)
        weg(karte)
        karte = _eine_legende(fig, ax_map, karten_zeilen, 1, placement,
                              "outside lower left", groesse)

    if placement != "inside":
        _legenden_zentrieren(fig, ((karte, ax_map), (schnitt, ax_cut)))
    return karte, schnitt


def _legende_zeilen(zeilen, ncol):
    """(handles, labels) so umsortiert, dass matplotlibs SPALTENWEISE
    Fuellung die uebergebenen ZEILEN ergibt.

    matplotlib verteilt Legendeneintraege spaltenweise. Eine gewuenschte
    ZEILEN-Aufteilung - die Kurven oben, der Arbeitspunkt in einer eigenen
    Zeile darunter - kommt dabei nur heraus, wenn man die Eintraege vorher
    umordnet und kurze Zeilen mit unsichtbaren Platzhaltern auffuellt."""
    gitter = [list(z) + [None] * (ncol - len(z)) for z in zeilen]
    handles, labels = [], []
    for spalte in range(ncol):
        for zeile in gitter:
            eintrag = zeile[spalte]
            if eintrag is None:
                handles.append(plt.Line2D([], [], **_LEER_HANDLE_STIL))
                labels.append("")
            else:
                handles.append(eintrag)
                labels.append(eintrag.get_label())
    return handles, labels


def legend_ncol(labels, max_chars=78, symbol_breite=6):
    """Wie viele Legendeneintraege nebeneinander gesetzt werden.

    Die Regel ist bewusst grob und in zwei Stufen:

    1. Passt ALLES in eine Zeile (geschaetzte Breite <= `max_chars`), kommt
       auch alles in eine Zeile.
    2. Sonst ZWEI pro Zeile, zentriert. Drei waeren technisch oft noch
       moeglich, aber eine Legende, die die Figur fast ausfuellt, liest sich
       schlechter als zwei ruhige Spalten.

    `symbol_breite` ist der Platz fuer Linie/Marker und den Abstand vor dem
    Text, ebenfalls in Zeichen gerechnet.

    Eins nur dann, wenn schon ein einzelner Eintrag breiter ist als die
    Zeile - dann hilft auch Zweispaltigkeit nicht.
    """
    n = len(labels)
    if n <= 1:
        return max(1, n)
    laengen = [_label_laenge(l) + symbol_breite for l in labels]
    if sum(laengen) <= max_chars:
        return n
    zwei = sorted(laengen)[-2:]
    return 2 if sum(zwei) <= max_chars else 1


def log_ticks(vmin, vmax, max_ticks=7):
    """"Runde" Ticks fuer eine logarithmische Amplituden-Colorbar.

    Matplotlibs Voreinstellung beschriftet eine Log-Achse ueber weniger als
    einer Dekade mit "$6\\times10^0$" - fuer Werte zwischen 1 und 8 ist das
    unlesbar. Stattdessen eine feste Leiter runder Faktoren, ausgeduennt,
    bis hoechstens max_ticks uebrig sind."""
    if not (vmin > 0 and vmax > vmin):
        return None
    leiter = np.array([1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0])
    dekaden = range(int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax))) + 1)
    kandidaten = np.concatenate([leiter * 10.0 ** d for d in dekaden])
    drin = np.unique(kandidaten[(kandidaten >= vmin) & (kandidaten <= vmax)])
    if drin.size == 0:
        return None
    while drin.size > max_ticks:
        drin = drin[::2]
    return drin if drin.size >= 2 else None


def _r_clamp_mask(grid, bounds):
    """Boolesche Maske der Punkte, die auf einer r_bounds-Schranke liegen."""
    if bounds is None or len(bounds) != 2:
        return np.zeros(grid.shape, dtype=bool)
    lo, hi = float(bounds[0]), float(bounds[1])
    endlich = np.isfinite(grid)
    return endlich & ((np.abs(grid - lo) <= R_BOUND_TOL)
                      | (np.abs(grid - hi) <= R_BOUND_TOL))


def amplitude_panels(results):
    """Die beiden Amplituden-Karten r_x und r_y - oder None, wenn der
    Datensatz keine Amplituden mitfuehrt (Fest-Amplituden-Scan).

    Drei Entscheidungen, alle aus echten Datensaetzen begruendet:

    1. GEMEINSAME Farbskala fuer r_x und r_y, damit direkt ablesbar ist,
       dass r_y systematisch ueber r_x liegt.
    2. LOGARITHMISCHE Skala. r_x/r_y sind Verhaeltnisse - die sinnvolle
       Einheit ist der Faktor, nicht die Differenz; die Verteilung hat einen
       langen Schwanz. Bewusst IMMER logarithmisch: eine Skala, die je nach
       Datensatz umschaltet, macht den Vergleich zweier Aufloesungen
       unmoeglich.
    3. Punkte, deren Amplitude auf einer r_bounds-SCHRANKE klemmt, werden aus
       der Skala herausgenommen und grau ueberzeichnet. Dort steht kein
       freies Optimum, sondern die Schranke."""
    if not has_amplitudes(results):
        return None
    bounds = results.get('r_bounds')
    grids, masken, freie = {}, {}, []
    for key in ("r_x_grid", "r_y_grid"):
        grid = np.asarray(results[key], dtype=float)
        maske = _r_clamp_mask(grid, bounds)
        grids[key], masken[key] = grid, maske
        freie.append(grid[np.isfinite(grid) & ~maske])
    frei = np.concatenate(freie)
    if frei.size == 0:
        alle = np.concatenate([grids[k][np.isfinite(grids[k])].ravel() for k in grids])
        if alle.size == 0:
            return None
        frei, masken = alle, {k: np.zeros(grids[k].shape, dtype=bool) for k in grids}
    vmin, vmax = float(frei.min()), float(frei.max())
    if vmin <= 0:                            # LogNorm braucht positive Grenzen
        vmin = float(frei[frei > 0].min()) if (frei > 0).any() else 1e-3
    if not vmax > vmin:                      # konstantes Gitter
        vmin, vmax = vmin / 1.5, vmax * 1.5
    norm = LogNorm(vmin=vmin, vmax=vmax)
    geklemmt = any(m.any() for m in masken.values())
    return [
        dict(key="r_x_grid", cbar=r"$r_x$", title=r"Amplitude ratio $r_x$",
             cmap=AMPLITUDE_CMAP, scale=1.0, norm=norm,
             mask=masken["r_x_grid"], clamped=geklemmt),
        dict(key="r_y_grid", cbar=r"$r_y$", title=r"Amplitude ratio $r_y$",
             cmap=AMPLITUDE_CMAP, scale=1.0, norm=norm,
             mask=masken["r_y_grid"], clamped=geklemmt),
    ]


def r_bounds_clamped_fraction(results):
    """(Anteil r_x, Anteil r_y) der Gitterpunkte, deren Amplitude auf einer
    r_bounds-Schranke liegt - oder None. Solche Punkte sind KEINE freien
    Optima: der Optimierer wollte weiter und durfte nicht."""
    bounds = results.get('r_bounds')
    if bounds is None or len(bounds) != 2 or not has_amplitudes(results):
        return None
    anteile = []
    for key in ("r_x_grid", "r_y_grid"):
        grid = np.asarray(results[key], dtype=float)
        gueltig = np.isfinite(grid)
        if not gueltig.any():
            return None
        anteile.append(float(_r_clamp_mask(grid, bounds).sum()) / float(gueltig.sum()))
    return tuple(anteile)


def plot_metric_comparison(results, prefix, out_dir=None, win_axis="before_lens",
                           draw_best_point=True, save=True, show=False,
                           confirm_overwrite=None, fit_line=None,
                           with_amplitudes=False, forbidden_factor=None,
                           best_point=None, fit_line_dashed_extrapolation=False):
    """Metrik-Karten: Uniformity und Crosstalk nebeneinander; mit
    with_amplitudes=True zusaetzlich r_x und r_y als zweite Zeile (2x2).

    with_amplitudes schreibt in eine EIGENE Datei
    (..._metric_comparison_amp.pdf), damit die gewohnte Fassung unveraendert
    daneben bestehen bleibt.

    fit_line: Ergebnis von fit_valley_line() - dann wird die Gerade in alle
    Karten eingezeichnet, als durchgezogene Linie ueber den ganzen gescannten
    Bereich. Mit fit_line_dashed_extrapolation=True wird der Teil ausserhalb
    des Fit-Bereichs stattdessen gepunktet (siehe draw_fit_line_on_map)."""
    out_dir = paths.fit_plots_dir() if out_dir is None else out_dir
    width_vals = np.asarray(results['width_vals'], dtype=float)
    x_vals, x_label, reversed_ = win_axis_values(results, win_axis)

    panels = metric_panels(results)
    if with_amplitudes:
        amp = amplitude_panels(results)
        if amp is None:
            raise ValueError(
                "Der Datensatz fuehrt keine Amplituden mit (r_x_grid/r_y_grid "
                "fehlen) - die 4-Karten-Uebersicht ist hier nicht moeglich. "
                "Das ist bei einem Fest-Amplituden-Scan normal.")
        panels = panels + amp

    # Keine Ueberschrift ueber der Figur: in einem LaTeX-Dokument steht dort
    # die \caption. Was der Plot zeigt, steht in den Titeln der Karten.
    n_rows = (len(panels) + 1) // 2
    figsize = HALF_PAGE_FIGSIZE if n_rows >= 2 else ROW_FIGSIZE
    # Alle Karten haben dieselben Achsen - die Beschriftung deshalb nur
    # einmal aussen herum.
    with dokument_stil(figsize[0]):
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharex="all", sharey="all",
                                 squeeze=False, constrained_layout=True)
        for ax, panel in zip(axes.flat, panels):
            Z = np.asarray(results[panel['key']], dtype=float) * panel['scale']
            maske = panel.get('mask')
            if maske is not None:
                # Geklemmte Punkte aus der Farbskala nehmen; sie kommen gleich
                # als eigene graue Ebene darueber.
                Z = np.where(maske, np.nan, Z)
            Z_plot = Z[:, ::-1] if reversed_ else Z
            norm = panel.get('norm')
            # pcolormesh nimmt norm ODER vmin/vmax, nie beides.
            grenzen = (dict(norm=norm) if norm is not None
                       else dict(vmin=panel.get('vmin'), vmax=panel.get('vmax')))
            im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto",
                               cmap=panel['cmap'], **grenzen)
            ticks = log_ticks(norm.vmin, norm.vmax) if norm is not None else None
            cbar = fig.colorbar(im, ax=ax, label=panel['cbar'],
                                **({} if ticks is None else dict(ticks=ticks)))
            if ticks is not None:
                cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
                cbar.ax.minorticks_off()
            if maske is not None and maske.any():
                M = np.where(maske, 1.0, np.nan)
                M_plot = M[:, ::-1] if reversed_ else M
                # NaN zeichnet matplotlib durchsichtig - uebrig bleibt genau der
                # geklemmte Bereich in Grau, ungueltige Punkte bleiben weiss.
                ax.pcolormesh(x_vals, width_vals * 1e-6, M_plot, shading="auto",
                              cmap=ListedColormap([R_CLAMP_COLOR]), vmin=0.0, vmax=1.0)
            ax.set_title(panel['title'])
            if forbidden_factor is not None:
                draw_forbidden_region(ax, results, win_axis, forbidden_factor)
            if fit_line is not None:
                draw_fit_line_on_map(ax, results, fit_line, win_axis,
                                     dashed_extrapolation=fit_line_dashed_extrapolation)
            if draw_best_point:
                # Gezeichnet wird der Stern in jeder Karte, den Legendeneintrag
                # holt sich aber nur die erste.
                draw_best_point_marker(ax, results, win_axis, legend=False,
                                       label=(ax is axes.flat[0]), best=best_point)
        for ax in axes[-1, :]:
            ax.set_xlabel(x_label)
        for ax in axes[:, 0]:
            ax.set_ylabel(WIDTH_LABEL)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        if any(p.get('clamped') for p in panels):
            # Das Grau erklaert sich nicht von selbst - ein Eintrag in der
            # ohnehin vorhandenen gemeinsamen Legende, kein zweiter Kasten.
            handles.append(Patch(facecolor=R_CLAMP_COLOR, edgecolor="none"))
            labels.append(R_CLAMP_LABEL)
        if handles:
            fig.legend(handles, labels, loc="outside lower center",
                       ncol=legend_ncol(labels), framealpha=0.9)
        dateiname = (f"{prefix}_metric_comparison_amp.pdf" if with_amplitudes
                     else f"{prefix}_metric_comparison.pdf")
        return _finish(fig, out_dir, dateiname, save, show, confirm_overwrite)


def plot_region(results, prefix, out_dir=None, win_axis="before_lens",
                draw_best_point=True, save=True, show=False, confirm_overwrite=None,
                forbidden_factor=None, best_point=None):
    """Score-Heatmap, auf Wunsch mit dem Arbeitspunkt."""
    out_dir = paths.fit_plots_dir() if out_dir is None else out_dir
    width_vals = np.asarray(results['width_vals'], dtype=float)
    x_vals, x_label, reversed_ = win_axis_values(results, win_axis)
    Z = score_grid(results)
    if Z is None:
        raise ValueError("Der Datensatz enthaelt kein Score-Gitter.")
    Z = np.asarray(Z, dtype=float) * 100.0
    Z_plot = Z[:, ::-1] if reversed_ else Z

    alpha = float(results.get('alpha', 0.7))
    score_label = r"$J_%s$ [%%]" % _IDX
    title = f"Combined score ({FAMILY_TITLE}, " + rf"$\alpha$ = {alpha:.2f})"

    with dokument_stil(REGION_FIGSIZE[0]):
        fig, ax = plt.subplots(figsize=REGION_FIGSIZE, constrained_layout=True)
        im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap="magma_r")
        fig.colorbar(im, ax=ax, label=score_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(WIDTH_LABEL)

        region = results.get('region') or {}
        pct = results.get('combo_percentile')
        if pct is not None and region.get('n_points_region'):
            title += (f"\nbest {pct:.0f}% of all grid points: "
                      f"{region['n_points_region']}/{region['n_points_total']}")
        ax.set_title(title)

        if forbidden_factor is not None:
            draw_forbidden_region(ax, results, win_axis, forbidden_factor)
        if draw_best_point:
            draw_best_point_marker(ax, results, win_axis, legend=False,
                                   label=True, best=best_point)
        # Legende UNTER die Karte, nicht hinein: mit den Koordinaten des
        # Arbeitspunkts und dem verbotenen Bereich war sie breiter als die
        # halbe Karte und verdeckte je nach Datensatz einen anderen Teil der
        # Heatmap - dieselbe Anordnung wie im Schnittplot.
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper center",
                      bbox_to_anchor=(0.5, -0.16),
                      ncol=legend_ncol(labels), framealpha=0.9)
        return _finish(fig, out_dir, f"{prefix}_region.pdf", save, show, confirm_overwrite)


def _plotter_punkt(best):
    """`win_input_fixed`/`width_fixed` fuer die Scan-Plotter aus dem
    Arbeitspunkt.

    Ohne sie legen die Plotter ihre Schnitte durch den GLOBAL besten
    Gitterpunkt - nicht durch den im Dialog eingestellten Punkt. Genau den
    zeigen aber alle uebrigen Plots; zwei verschiedene Punkte in einer
    Auswertung sind eine Fehlerquelle. Die Plotter runden den Wert selbst auf
    die naechste Gitterzeile/-spalte und beschriften ihn dann als
    "selected point (cut origin)" statt "best point (global optimum)".
    Leeres dict, wenn es keinen Arbeitspunkt gibt - dann bleibt es beim
    bisherigen Verhalten."""
    if not best:
        return {}
    punkt = {}
    if best.get('win_input') is not None:
        punkt['win_input_fixed'] = float(best['win_input'])
    if best.get('width') is not None:
        punkt['width_fixed'] = float(best['width'])
    return punkt


def plot_overview(results, out_dir=None, save=True, show=False,
                  confirm_overwrite=None, best_point=None,
                  win_axis="after_lens"):
    """Die Uebersicht des jeweiligen Scan-Plotters (unveraenderte Module aus
    lib/): beim Amplituden-Scan die 6-Panel-Uebersicht plus die Schnitte,
    beim Fest-Amplituden-Scan die zwei Heatmaps.

    `best_point`: der Arbeitspunkt - die Schnitte laufen dann durch ihn und
    nicht durch das globale Optimum (siehe _plotter_punkt).
    `win_axis`: dieselbe Waist-Konvention wie in allen uebrigen Plots - wer
    alles in µm anzeigen laesst, bekommt sie auch hier."""
    out_dir = paths.fit_plots_dir() if out_dir is None else out_dir
    if has_amplitudes(results):
        plotter = AmplitudeScanPlotter(results, out_dir=out_dir,
                                       confirm_overwrite=confirm_overwrite)
        punkt = _plotter_punkt(best_point)
        overview = plotter.plot_scan2d_combined(show=show, save=save,
                                                win_axis=win_axis, **punkt)
        cuts = plotter.plot_dependence_cuts(show=show, save=save,
                                            win_axis=win_axis, **punkt)
        return dict(overview=overview, dependence_cuts=cuts)
    plotter = FixedScanPlotter(results, out_dir=out_dir,
                               confirm_overwrite=confirm_overwrite)
    overview = plotter.plot_scan2d_combined(show=show, save=save,
                                            win_axis=win_axis)
    return dict(overview=overview, dependence_cuts=None)


# ======================================================================
# Querschnitt DURCH DEN MARKIERTEN PUNKT (Kreuzschnitt)
# ======================================================================
# Zwei Schnitte durch den Stern: einmal bei fester width entlang des Waists,
# einmal bei festem Waist entlang der width. Das beantwortet die Frage "wie
# empfindlich sind r_x und r_y an meinem Arbeitspunkt?" - eine andere Frage
# als der Talschnitt, der dem Minimum folgt.
#
# Gelesen wird immer auf dem GITTER: liegt der Stern zwischen den Punkten
# (selbst vorgegebener Punkt), laufen die Schnitte durch die naechstgelegene
# Zeile bzw. Spalte. Das steht dann auch im Titel - interpolieren waere hier
# irrefuehrend, weil r_x/r_y Optimierungs-Ergebnisse sind und keine glatten
# Funktionen.
# In dieser Grafik geht es AUSSCHLIESSLICH um die Amplituden - Uniformity
# und Crosstalk haben ihren Platz in den Karten und im Talschnitt. Das
# Aussehen ist bewusst das der alten Amplituden-Schnitte
# (AmplitudeScanPlotter.plot_dependence_cuts): r_x blau mit Kreisen, r_y
# orange mit Quadraten, beide auf EINER Achse (es ist dieselbe Groesse in
# zwei Richtungen), der Punkt als rote gestrichelte Senkrechte.
POINT_CUT_TRACES = ("r_x", "r_y")
POINT_CUT_STYLE = {
    "r_x": dict(color="#1f77b4", marker="o"),
    "r_y": dict(color="#ff7f0e", marker="s"),
}
POINT_MARK_STYLE = dict(color="red", linestyle="--", linewidth=1.3, alpha=0.7)
# Derselbe Zuschnitt wie der Talschnitt (VALLEY_FIGSIZE): zwei Panels auf
# Textbreite, feste Seitengroesse. Die Figuren eines Dokuments sollen
# nebeneinander nach demselben Format aussehen.
POINT_CUT_FIGSIZE = (6.3, 3.6)


def point_cut_indices(results, best=None):
    """(Zeile, Spalte) des markierten Punktes im Gitter, oder None."""
    best = (results.get('best') or {}) if best is None else best
    if not best or best.get('win_input') is None:
        return None
    if best.get('row') is not None and best.get('col') is not None:
        return int(best['row']), int(best['col'])
    win_input_vals = np.asarray(results['win_input_vals'], dtype=float)
    width_vals = np.asarray(results['width_vals'], dtype=float)
    j = int(np.argmin(np.abs(win_input_vals - best['win_input'])))
    i = int(np.argmin(np.abs(width_vals - best['width'])))
    return i, j


def plot_point_cuts(results, prefix, best=None, out_dir=None,
                    win_axis="after_lens", save=True, show=False,
                    confirm_overwrite=None, legend_fontsize=9):
    """Kreuzschnitt durch den markierten Punkt - r_x und r_y.

    Links: bei FESTER width entlang des Waists. Rechts: bei FESTEM Waist
    entlang der width. Der Punkt ist in beiden Panels als senkrechte rote
    Linie markiert, sein Wert je Kurve als Stern.
    """
    out_dir = paths.fit_plots_dir() if out_dir is None else out_dir
    verfuegbar = available_trace_keys(results)
    gewaehlt = [k for k in POINT_CUT_TRACES if k in verfuegbar]
    if not gewaehlt:
        raise ValueError(
            "Der Datensatz fuehrt keine Amplituden mit (r_x_grid/r_y_grid) - "
            "einen Schnitt durch den Punkt gibt es nur beim Amplituden-Scan.")

    stelle = point_cut_indices(results, best)
    if stelle is None:
        raise ValueError("Es ist kein Punkt markiert, durch den geschnitten "
                         "werden koennte.")
    i, j = stelle
    best = (results.get('best') or {}) if best is None else best

    width_mhz = np.asarray(results['width_vals'], dtype=float) * 1e-6
    if win_axis == "after_lens":
        x_col = waist_um_of(results)
        x_label = WAIST_UM_LABEL
        waist_einheit = "$\\mu$m"
    else:
        x_col = np.asarray(results['win_input_vals'], dtype=float) * 1e3
        x_label = WIN_INPUT_LABEL
        waist_einheit = "mm"
    ordnung = np.argsort(x_col)

    # Die Koordinaten des ARBEITSPUNKTS - dieselben Zahlen wie ueberall sonst.
    #
    # Zwei Regeln, die beide gelten muessen:
    #
    # 1. Innerhalb dieser Figur benennen der Titel des einen Panels und der
    #    Legendeneintrag des anderen DIESELBE Linie ("Cut at width = 0.290
    #    MHz" links, "width = 0.290 MHz" rechts). Beide kommen deshalb aus
    #    x_mark/y_mark, nicht einmal daher und einmal von woanders - sonst
    #    stehen nach dem Runden zwei verschiedene Werte fuer dieselbe Linie
    #    im Bild.
    # 2. Ueber die ganze Auswertung hinweg zeigt jeder Plot denselben
    #    Arbeitspunkt mit denselben Zahlen: der Stern in den Karten, der
    #    Strich im Talschnitt, die Panel-Schnitte und der Bericht. Deshalb
    #    die Koordinaten des Punkts und NICHT die der Gitterlinie - bei einem
    #    frei gewaehlten Punkt liegen die dazwischen und koennen auf der
    #    angezeigten Stelle anders runden (gemessen: 0.2865 MHz gewaehlt,
    #    Gitterzeile 0.2850 -> "0.286" gegen "0.285").
    #
    # Die Kurven selbst stammen aus der naechstgelegenen Gitterzeile/-spalte
    # (i, j) - naeher als eine halbe Gitterweite laesst sich ein Schnitt
    # durch einen freien Punkt nicht legen. Welche das ist, steht im Bericht.
    if best.get('off_grid'):
        x_mark = (float(best['waist_um']) if win_axis == "after_lens"
                  else float(best['win_input']) * 1e3)
        y_mark = float(best['width']) * 1e-6
    else:
        x_mark, y_mark = float(x_col[j]), float(width_mhz[i])

    # Jedes Panel schneidet bei EINER festen Groesse und laeuft ueber die
    # andere: der Titel nennt die feste, die rote Linie in der Legende die
    # laufende - dieselbe Aufteilung wie im Talschnitt, wo der Stern in der
    # Karte die feste Koordinate traegt und der Strich die der Schnittachse.
    waist_axis = "waist_um" if win_axis == "after_lens" else "waist_mm"
    panels = [
        ("waist", x_col[ordnung], x_label,
         "Cut at " + koordinaten_label("width", y_mark), x_mark, waist_axis),
        ("width", width_mhz, WIDTH_LABEL,
         "Cut at " + koordinaten_label(waist_axis, x_mark), y_mark, "width"),
    ]

    with dokument_stil(POINT_CUT_FIGSIZE[0], legend_fontsize=legend_fontsize):
        fig, axes = plt.subplots(1, 2, figsize=POINT_CUT_FIGSIZE, sharey=True,
                                 constrained_layout=True)
        for ax, (achse, x, xl, titel, marke, marken_axis) in zip(axes, panels):
            for key in gewaehlt:
                grid = _grid_for(results, key)
                werte = grid[i, :][ordnung] if achse == "waist" else grid[:, j]
                ax.plot(x, werte, linewidth=S(1.5), markersize=S(3.4),
                        label=TRACE_SPECS[key][0], **POINT_CUT_STYLE[key])
            # Die Linie traegt die Koordinate, an der sie steht - in jedem
            # Panel also die Groesse der jeweiligen x-Achse.
            ax.axvline(marke, label=koordinaten_label(marken_axis, marke),
                       **SD(POINT_MARK_STYLE))
            # Der Wert je Kurve genau am Punkt - auf dem Gitter abgelesen.
            # Etwas kleinerer Stern als in den Karten: hier sitzt er auf einer
            # Kurve und soll sie nicht verdecken.
            stern_stil = SD(BEST_POINT_STYLE, markersize=S(12))
            for key in gewaehlt:
                grid = _grid_for(results, key)
                ax.plot([marke], [grid[i, j]], **stern_stil)
            # Kurzform wie im Talschnitt: zwei Panels auf Textbreite, die
            # ausgeschriebene Fassung lief aus der Figur.
            ax.set_xlabel(kurzes_achsenlabel(xl))
            ax.set_title(titel)
            # Wie in der Talschnitt-Karte: die automatische Teilung kennt die
            # Breite der Zahlen nicht und setzte im schmalen Panel mehr
            # Striche, als nebeneinander passen.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            ax.grid(True, alpha=0.25)
        # sharey=True: das Raster auf der linken Achse gilt fuer beide.
        _achsenraster(axes[0], TRACE_SPECS[gewaehlt[0]][1])
        axes[0].set_ylabel(r"Amplitude ratio $r_x$, $r_y$")

        # Je Panel eine eigene Legende OBEN RECHTS im Bild: die Linie heisst
        # in den beiden Panels verschieden (jeweils ihre eigene Koordinate),
        # eine gemeinsame Legende unter der Figur koennte das nicht zeigen.
        # `set_in_layout(False)`, damit constrained_layout die Panels nicht um
        # die Flaeche der Legende schrumpft - wie im Talschnitt.
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                continue
            legende = ax.legend(handles, labels, loc="upper right",
                                ncol=legend_ncol(labels), framealpha=0.9,
                                fontsize=_legende_fontsize(legend_fontsize))
            legende.set_in_layout(False)
        # tight=False: feste Seitengroesse wie beim Talschnitt.
        return _finish(fig, out_dir, f"{prefix}_point_cuts.pdf", save, show,
                       confirm_overwrite, tight=False)


# ======================================================================
# Markdown-Bericht
# ======================================================================
def _waist_um(results, win_input):
    return win_input_to_win(win_input, results['f1'], results['f2'],
                            results['lambda_opt'], results['fLO']) * 1e6


def _waist_range_um(results, lo, hi):
    a, b = _waist_um(results, lo), _waist_um(results, hi)
    return min(a, b), max(a, b)


# Die Groessen, die im Bericht an einem Punkt aufgelistet werden.
POINT_VALUE_KEYS = ["uniformity", "crosstalk", "score", "r_x", "r_y"]

POINT_VALUE_LABELS = {
    "uniformity": "Uniformity U_" + _IDX,
    "crosstalk": "Crosstalk eta_" + _IDX,
    "score": "J (Score, roh)",
    "r_x": "Amplituden-Verhaeltnis r_x",
    "r_y": "Amplituden-Verhaeltnis r_y",
}


def _bilinear(grid, waist_grid, width_grid, waist_um, width_mhz):
    """Bilineare Interpolation eines Gitters an einer beliebigen Stelle.
    NaN in einer der vier Ecken -> NaN (nicht heimlich ueberbruecken);
    ausserhalb des Gitters -> NaN."""
    grid = np.asarray(grid, dtype=float)
    ordnung = np.argsort(waist_grid)             # waist faellt mit dem Spaltenindex
    w_sortiert = np.asarray(waist_grid, dtype=float)[ordnung]
    if not (w_sortiert[0] <= waist_um <= w_sortiert[-1]):
        return float("nan")
    if not (width_grid[0] <= width_mhz <= width_grid[-1]):
        return float("nan")
    jb = int(np.clip(np.searchsorted(w_sortiert, waist_um), 1, len(w_sortiert) - 1))
    ib = int(np.clip(np.searchsorted(width_grid, width_mhz), 1, len(width_grid) - 1))
    j0, j1 = int(ordnung[jb - 1]), int(ordnung[jb])
    i0, i1 = ib - 1, ib
    dw = w_sortiert[jb] - w_sortiert[jb - 1]
    dh = width_grid[i1] - width_grid[i0]
    tw = 0.0 if dw == 0 else (waist_um - w_sortiert[jb - 1]) / dw
    th = 0.0 if dh == 0 else (width_mhz - width_grid[i0]) / dh
    ecken = np.array([grid[i0, j0], grid[i0, j1], grid[i1, j0], grid[i1, j1]], dtype=float)
    if not np.isfinite(ecken).all():
        return float("nan")
    return float((1 - th) * ((1 - tw) * ecken[0] + tw * ecken[1])
                 + th * ((1 - tw) * ecken[2] + tw * ecken[3]))


def values_at_point(results, waist_um, width_mhz):
    """Die Groessen an einer beliebigen Stelle der (waist, width)-Ebene.

    Gibt je Groesse (interpoliert, am naechsten Gitterpunkt) zurueck, beides
    bereits in der Anzeige-Einheit. Interpoliert wird bilinear zwischen den
    vier umliegenden Gitterpunkten; der Gitterwert daneben ist eine wirklich
    gerechnete Zahl und dient als Anker."""
    waist_grid = waist_um_of(results)
    width_grid = np.asarray(results['width_vals'], dtype=float) * 1e-6
    j = int(np.argmin(np.abs(waist_grid - waist_um)))
    i = int(np.argmin(np.abs(width_grid - width_mhz)))
    out = {}
    for key in POINT_VALUE_KEYS:
        grid = _grid_for(results, key)
        if grid is None:
            continue
        _label, _unit, _color, als_prozent = TRACE_SPECS[key]
        faktor = 100.0 if als_prozent else 1.0
        interp = _bilinear(grid, waist_grid, width_grid, waist_um, width_mhz)
        out[key] = (interp * faktor, float(grid[i, j]) * faktor)
    return out, (float(waist_grid[j]), float(width_grid[i]))


def _point_value_table(results, waist_um, width_mhz, auf_gitter=False):
    """Markdown-Tabelle der Groessen an einem Punkt.

    auf_gitter=True: der Punkt IST ein Gitterpunkt, dann gibt es nur eine
    Wertespalte - interpolieren waere dort dieselbe Zahl noch einmal."""
    werte, (w_nah, h_nah) = values_at_point(results, waist_um, width_mhz)
    if not werte:
        return []
    if auf_gitter:
        zeilen = ["| Groesse | Wert |", "|---|---|"]
        for key, (_interp, gitter) in werte.items():
            _label, _unit, _color, als_prozent = TRACE_SPECS[key]
            einheit = " %" if als_prozent else ""
            fmt = "%.4f" if als_prozent else "%.5f"
            zeilen.append("| %s | %s |"
                          % (POINT_VALUE_LABELS[key], (fmt % gitter) + einheit))
        zeilen.append("")
        return zeilen
    zeilen = ["| Groesse | interpoliert | naechster Gitterpunkt |", "|---|---|---|"]
    for key, (interp, gitter) in werte.items():
        _label, _unit, _color, als_prozent = TRACE_SPECS[key]
        einheit = " %" if als_prozent else ""
        fmt = "%.4f" if als_prozent else "%.5f"
        interp_txt = "n/a" if not np.isfinite(interp) else (fmt % interp) + einheit
        zeilen.append("| %s | %s | %s |"
                      % (POINT_VALUE_LABELS[key], interp_txt, (fmt % gitter) + einheit))
    zeilen += [
        "",
        f"Interpoliert wird bilinear zwischen den vier umliegenden Gitterpunkten; "
        f"liegt einer davon ausserhalb des Scans oder ist er ungueltig, steht dort "
        f"n/a. Die rechte Spalte sind die wirklich gerechneten Werte am "
        f"naechstgelegenen Gitterpunkt (Waist = {w_nah:.4f} µm, "
        f"width = {h_nah:.4f} MHz).",
    ]
    if has_amplitudes(results):
        zeilen += [
            "",
            "Achtung bei r_x/r_y: das sind Optimierungs-ERGEBNISSE des Scans, keine "
            "glatten Funktionen. Wer die Metriken exakt an diesem Punkt braucht, muss "
            "die Amplituden dort neu optimieren.",
        ]
    return zeilen


def _manual_point_lines(punkt, results=None):
    """Berichtsabschnitt fuer einen selbst vorgegebenen Punkt."""
    zeilen = ["## Markierter Punkt (Stern im Plot)", ""]
    if punkt['follow'] == BEST_POINT_MANUAL_BOTH:
        zeilen += [
            "Beide Koordinaten selbst vorgegeben - hier steckt keine Gerade und "
            "keine Rechnung drin:",
        ]
    else:
        vorgabe = "Waist" if punkt['follow'] == BEST_POINT_MANUAL_WAIST else "Width"
        einheit = MANUAL_POINT_UNITS[punkt['follow']]
        zeilen += [
            f"Selbst vorgegeben: {vorgabe} = {punkt['given']:.4f} {einheit}. Die zweite "
            f"Koordinate kommt aus der Talpfad-Geraden "
            f"(width/MHz = {punkt['fit_a']:.5f} * waist/µm {punkt['fit_b']:+.6f}):",
        ]
    zeilen += [
        "",
        f"- Waist = {punkt['waist_um']:.4f} µm  (win_input = "
        f"{punkt['win_input'] * 1e3:.4f} mm)",
        f"- width = {punkt['width'] * 1e-6:.4f} MHz",
        "",
        ("Der Punkt liegt in aller Regel ZWISCHEN den Gitterpunkten - er wird auch "
         "dort gezeichnet, nicht auf ein Gitter gerundet."
         if punkt['follow'] == BEST_POINT_MANUAL_BOTH else
         "Der Punkt liegt exakt auf der Geraden und damit in aller Regel ZWISCHEN "
         "den Gitterpunkten - er wird auch dort gezeichnet, nicht auf ein Gitter "
         "gerundet.")
        + " Der naechstgelegene tatsaechlich gerechnete Gitterpunkt liegt "
        f"bei Waist = {punkt['nearest_waist_um']:.4f} µm / width = "
        f"{punkt['nearest_width'] * 1e-6:.4f} MHz.",
    ]
    if punkt.get('outside'):
        zeilen += [
            "",
            "**ACHTUNG: dieser Punkt liegt ausserhalb des gescannten Fensters.** "
            "Dort gibt es keine Daten"
            + ("." if punkt['follow'] == BEST_POINT_MANUAL_BOTH
               else "; die Gerade ist hier reine Extrapolation."),
        ]
    if results is not None:
        zeilen += ["", "### Werte an diesem Punkt", ""]
        zeilen += _point_value_table(results, punkt['waist_um'], punkt['width'] * 1e-6)
    zeilen.append("")
    return zeilen


def _best_point_marker_lines(best_point, results=None):
    """Nur noetig, wenn der Stern NICHT den bestimmten Bestpunkt zeigt -
    sonst stuende dieselbe Zahl zweimal im Bericht."""
    if not best_point or best_point.get('follow') in (None, BEST_POINT_FOLLOW_STORED):
        return []
    if best_point.get('follow') in MANUAL_POINT_KEYS:
        return _manual_point_lines(best_point, results)
    zeilen = [
        "## Markierter Punkt (Stern im Plot)",
        "",
        f"Der Stern zeigt NICHT den oben genannten Punkt, sondern das Minimum von "
        f"{_follow_label(best_point['follow'])}:",
        "",
        f"- win_input = {best_point['win_input'] * 1e3:.4f} mm",
        f"- width = {best_point['width'] * 1e-6:.4f} MHz",
        f"- Wert dort = {best_point.get('value', float('nan')):.6g}",
    ]
    if best_point.get('at_edge'):
        zeilen += [
            "",
            "**ACHTUNG: dieser Punkt liegt auf dem Rand des gescannten Fensters** - "
            "also vermutlich kein Optimum, sondern nur das Ende des Scans. Der Stern "
            "ist deshalb offen gezeichnet. Abhilfe: Scan-Bereich erweitern.",
        ]
    if results is not None:
        waist_um = _waist_um(results, best_point['win_input'])
        zusatz = _point_value_table(results, waist_um, best_point['width'] * 1e-6,
                                    auf_gitter=True)
        if zusatz:
            zeilen += ["", "### Werte an diesem Punkt", ""] + zusatz
    zeilen.append("")
    return zeilen


def _forbidden_lines(results, factor, excluded):
    """Berichtsabschnitt zum verbotenen Bereich. Leere Liste, wenn keiner
    verlangt wurde."""
    if factor is None:
        return []
    grenze = forbidden_boundary(results, factor)
    if grenze is None:
        return ["## Verbotener Bereich (Ueberlappung der Eck-Spots)", "",
                "Bei diesem Datensatz gibt es keine zwei Eck-Spots "
                "(N_x < 2 und N_y < 2) - es kann nichts ueberlappen.", ""]
    maske = forbidden_mask(results, factor)
    n_verboten, n_gesamt = int(maske.sum()), int(maske.size)
    achsen = "x und y" if grenze['n_axes'] == 2 else "einer Achse"
    lines = [
        "## Verbotener Bereich (Ueberlappung der Eck-Spots)",
        "",
        "Die beiden diagonal gegenueberliegenden Eck-Spots des "
        f"{results['N_x']}x{results['N_y']}-Arrays duerfen sich nicht ueberlappen. "
        "`width` ist die Gesamtspannweite des Tonarrays - in "
        f"{achsen} derselbe Wert -, raeumlich also",
        "",
        "```",
        f"S(width) = {grenze['um_per_MHz']:.4f} µm/MHz * width/MHz",
        f"d        = sqrt({grenze['n_axes']}) * S            (Pythagoras, Eckabstand)",
        f"d        > {grenze['factor']:.4g} * waist          (Bedingung: kein Ueberlapp)",
        "```",
        "",
        "S ist linear in width (radius_from_angle geht ueber tan, aber theta liegt "
        "bei 1.2e-3 rad - die Abweichung von der Geraden ist 5e-7 relativ). Die "
        "Bedingung ist deshalb in der (waist, width)-Ebene exakt eine Ursprungsgerade:",
        "",
        "```",
        f"width/MHz > {grenze['slope']:.5f} * waist/µm       (erlaubt)",
        "```",
        "",
        f"- Steigung a = {grenze['slope']:.6f} MHz/µm bei Faktor k = {grenze['factor']:.4g}",
        f"- Im verbotenen Bereich (width <= a*waist): {n_verboten} von {n_gesamt} "
        f"Gitterpunkten ({100.0 * n_verboten / n_gesamt:.1f}%)",
    ]
    if excluded:
        lines += [
            "",
            "**Diese Punkte wurden aus der Auswertung ausgeschlossen** (auf NaN gesetzt). "
            "Bester Punkt, Region, Talpfad und Geradenfit oben beziehen sich also nur "
            "auf den erlaubten Bereich. Der Score ist punktweise definiert (rohes J, "
            "keine gitterweite Normierung) - er aendert sich durch den Ausschluss NUR "
            "im verbotenen Bereich, nicht anderswo.",
        ]
    else:
        lines += [
            "",
            "Die Grenze ist nur eingezeichnet. Alle Zahlen dieses Berichts - bester "
            "Punkt, Region, Talpfad, Geradenfit - enthalten die verbotenen Punkte "
            "weiterhin.",
        ]
    lines.append("")
    return lines


def _scan_parameter_lines(results):
    n_win = len(np.asarray(results['win_input_vals']))
    n_width = len(np.asarray(results['width_vals']))
    lines = [
        "## Scan-Parameter",
        "",
        f"- N_x={results['N_x']}, N_y={results['N_y']}, Profil={results.get('profile')}",
        f"- Gitterpunkte: {n_win} x {n_width}",
        f"- alpha = {float(results.get('alpha', 0.7)):.3f}",
    ]
    if results.get('r_bounds') is not None:
        lines.append(f"- r_bounds = {tuple(results['r_bounds'])}")
    amps = results.get('amps')
    if amps is not None:
        a = np.asarray(amps, dtype=float).ravel()
        lines.append(f"- feste Amplituden: {np.array2string(a, precision=4)}")
    if str(results.get('profile')).lower() == "airy":
        faktor = results.get('airy_scale_factor')
        if faktor is None:
            lines.append(
                "- airy_scale_factor: nicht im Datensatz gespeichert - es galt der "
                "Optimierer-Default 1.19 (`first_zero_radius = Faktor * waist`)")
        else:
            lines.append(
                f"- airy_scale_factor = {float(faktor):.4f} "
                f"(`first_zero_radius = Faktor * waist`)")
    # Kohaerenz: mit oder ohne den statischen Interferenzanteil der
    # frequenzentarteten Spots gerechnet? Fehlt der Schluessel, stammt der
    # Datensatz aus der Zeit vor der Option und wurde inkohaerent gerechnet.
    koh = results.get('coherent')
    paare = results.get('n_degenerate_pairs')
    zusatz = "" if paare is None else f", {int(paare)} entartete(s) Paar(e)"
    if koh is None:
        lines.append(
            "- **Kohaerenz: nicht im Datensatz gespeichert** - der Scan lief vor "
            "dieser Option und hat die statische Interferenz frequenzentarteter "
            "Spots NICHT mitgerechnet. Mit neueren, kohaerent gerechneten "
            "Datensaetzen ist er deshalb nicht direkt vergleichbar.")
    elif koh:
        lines.append(
            f"- Kohaerenz: statische Interferenz mitgerechnet, Phasendifferenz 0 "
            f"(voll konstruktiv, unguenstigster Fall){zusatz}")
    else:
        lines.append(
            f"- Kohaerenz: AUS - reine Intensitaetssumme, statische Interferenz "
            f"nicht enthalten{zusatz}")

    sigma = results.get('sigma_atom')
    if sigma is not None:
        lines.append(
            f"- sigma_atom = {float(sigma) * 1e9:.1f} nm "
            f"(atom_temperature={float(results['atom_temperature']) * 1e6:.2f} µK, "
            f"trap_freq_r={float(results['trap_freq_r']) * 1e-3:.2f} kHz)")
    for key, label in (('atom_offset_x', 'atom_offset_x'), ('atom_offset_y', 'atom_offset_y')):
        if results.get(key):
            lines.append(f"- {label} = {float(results[key]) * 1e6:.4f} µm")
    return lines


def _region_lines(results):
    region = results.get('region') or {}
    lines = ["## Region", ""]
    if region.get('win_input_min') is None:
        lines += ["Kein gueltiges Rechteck gefunden (zu wenige valide Punkte).", ""]
        return lines
    um_lo, um_hi = _waist_range_um(results, region['win_input_min'], region['win_input_max'])
    thr = region.get('threshold')
    lines += [
        f"Groesstes achsenparalleles Rechteck innerhalb der besten "
        f"{results.get('combo_percentile', float('nan')):.0f}% aller Gitterpunkte "
        f"(nach dem rohen J); {region['n_points_region']}/{region['n_points_total']} "
        f"Gitterpunkte insgesamt im Akzeptanzbereich (Schwellwert J <= "
        f"{('%.6f' % thr) if thr is not None else 'n/a'}).",
        "",
        f"- win_input (vor der Linse): {region['win_input_min'] * 1e3:.4f} .. "
        f"{region['win_input_max'] * 1e3:.4f} mm",
        f"- effektiver Waist (nach der Linse): {um_lo:.4f} .. {um_hi:.4f} µm",
        f"- width: {region['width_min'] * 1e-6:.4f} .. {region['width_max'] * 1e-6:.4f} MHz",
        "",
    ]
    return lines


def _best_point_lines(results):
    best = results.get('best') or {}
    lines = ["## Bester Einzelpunkt (Minimum des rohen J)", ""]
    if best.get('win_input') is None:
        lines += ["Kein gueltiger Punkt im Gitter.", ""]
        return lines
    lines += [
        f"- win_input = {best['win_input'] * 1e3:.4f} mm "
        f"({_waist_um(results, best['win_input']):.4f} µm effektiver Waist)",
        f"- width = {best['width'] * 1e-6:.4f} MHz",
        f"- Uniformity = {best['uniformity'] * 100:.4f}%, "
        f"Crosstalk = {best['crosstalk'] * 100:.4f}%",
        f"- J (Score) = {best['score'] * 100:.4f}%",
        "",
    ]
    if best.get('at_edge'):
        lines += [
            "**ACHTUNG: dieser Punkt liegt auf dem Rand des gescannten Fensters.** "
            "Das ist dann kein Optimum, sondern nur die Stelle, an der der Scan "
            "aufhoert - das wahre Minimum liegt ausserhalb. Abhilfe: den Scan-"
            "Bereich erweitern. Im Plot ist der Stern deshalb offen statt gefuellt.",
            "",
        ]
    if best.get('r_x') is not None:
        lines += [f"- Amplituden-Verhaeltnisse an diesem Punkt: "
                  f"r_x / r_y = {best['r_x']:.4f} / {best['r_y']:.4f}", ""]
    stored = results.get('best_stored') or {}
    if stored.get('win_input') is not None:
        gleich = (abs(stored['win_input'] - best['win_input']) < 1e-12
                  and abs(stored['width'] - best['width']) < 1e-6)
        if not gleich:
            lines += [
                f"(Im Datensatz stand ein anderer bester Punkt: win_input = "
                f"{stored['win_input'] * 1e3:.4f} mm, width = "
                f"{stored['width'] * 1e-6:.4f} MHz - dort galt ein anderes alpha "
                f"oder ein anderer Ausschluss.)",
                "",
            ]
    return lines


def write_report(results, output_path, valley_line=None, valley_axis_label=None,
                 valley_path_mode="valley", forbidden_factor=None,
                 forbidden_excluded=False, best_point=None):
    """Markdown-Bericht zu einem Scan dieses Ordners.

    valley_line: Ergebnis von fit_valley_line() - ist valley_axis_label
    gesetzt, bekommt der Bericht einen Abschnitt zur Talpfad-Geraden."""
    n_win = len(np.asarray(results['win_input_vals']))
    n_width = len(np.asarray(results['width_vals']))
    kind = dataset_kind(results)
    alpha = float(results.get('alpha', 0.7))
    u_name = "Uniformity_w" if _WEIGHTED else "Uniformity"
    c_name = "Crosstalk_w" if _WEIGHTED else "Crosstalk"

    if kind == "amp":
        einleitung = (
            "An JEDEM (win_input, width)-Gitterpunkt wurde eine eigene "
            "(r_x, r_y)-Optimierung durchgefuehrt; die hier gezeigten Metriken sind "
            "die am jeweils gefundenen Optimum erreichten Werte.")
    else:
        einleitung = (
            "Scan ueber (win_input, width) bei FESTEN Amplituden - es wurde an "
            "keinem Punkt optimiert, die Amplituden standen vorher fest.")

    lines = [
        f"# {paths.REPORT_STEM} - N{results['N_x']}x{results['N_y']}, "
        f"{n_win}x{n_width} pts, {results.get('profile')}, {date.today().isoformat()}",
        "",
        f"Metrik-Familie: **{scan_data.FLAVOR_LABELS.get(scan_data.flavor_of(results))}**.",
        "",
        einleitung,
        "",
        "## Score",
        "",
        "Region, Bestpunkt und Score-Karte benutzen die ROHE Zielgroesse - dieselbe, "
        "die auch der Optimierer minimiert, ohne gitterweite Normierung:",
        "",
        "```",
        f"J = alpha*{u_name} + (1-alpha)*{c_name}",
        "```",
        "",
        f"mit alpha = {alpha:.3f}, Perzentil fuer die Region = "
        f"{float(results.get('combo_percentile', scan_data.DEFAULT_PERCENTILE)):.1f}%.",
        "",
        "Bewusst KEINE gitterweite Min-Max-Normierung: die haengt am gescannten "
        "Fenster, dieselbe Physik ergaebe bei anderem Scan-Bereich andere Zahlen.",
        "",
    ]
    if results.get('_source_path'):
        lines[4:4] = [f"Quelldatei: `{results['_source_path']}`", ""]

    lines += _region_lines(results)
    lines += _best_point_lines(results)
    if valley_axis_label is not None:
        lines += _valley_line_report_lines(valley_line, valley_axis_label,
                                           path_mode=valley_path_mode)
    lines += _best_point_marker_lines(best_point, results)
    lines += _forbidden_lines(results, forbidden_factor, forbidden_excluded)
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
             plot_scan_overview=False, save=True, show=False,
             ask_before_save=True, legend_fontsize=9,
             plots_dir=None, results_dir=None,
             valley_cut=False, valley_axis="waist_um", valley_follow="score",
             valley_select="guided", valley_guide_follow=GUIDE_FOLLOW_DEFAULT,
             valley_guide_halfwidth=GUIDE_HALFWIDTH_DEFAULT,
             valley_waist_range=None, valley_width_range=None,
             valley_traces=None, valley_fit_line=False, valley_path_mode="valley",
             fit_line_on_maps=False, fit_line_dashed_extrapolation=False,
             amplitude_maps=False,
             forbidden_factor=None, forbidden_excluded=False,
             forbidden_draw=True, best_point_follow=None, best_point_value=None,
             best_point_value2=None, point_cuts=False,
             valley_map_show_path=False,
             valley_legend_placement=VALLEY_LEGEND_PLACEMENT_DEFAULT,
             panel_legend_placement=VALLEY_LEGEND_PLACEMENT_DEFAULT):
    """Erzeugt alle zum Datensatz passenden Plots und den Bericht.

    Gibt ein dict mit den Pfaden zurueck. `results` sollte vorher durch
    scan_data.analyse() gelaufen sein (run_plots.py macht das); ist kein
    Score da, wird analyse() hier nachgeholt."""
    if results.get('score') is None:
        results = scan_data.analyse(results)
    plots_dir = paths.fit_plots_dir() if plots_dir is None else plots_dir
    results_dir = paths.FIT_RESULTS_DIR if results_dir is None else results_dir
    confirm_overwrite = None if ask_before_save else (lambda existing_path: True)

    prefix = output_prefix(results)
    out = dict(prefix=prefix, kind=dataset_kind(results), plots={}, report=None,
               plots_dir=plots_dir, results_dir=results_dir)

    # Die Gerade fuer die Metrik-Karten: dieselbe wie im Talschnitt, also
    # ueber der µm-Achse und mit der eingestellten Fuehrungsgroesse - egal,
    # welche Waist-Achse die Karten selbst benutzen.
    karten_fit = (fit_valley_line(results, axis=VALLEY_FIT_AXIS, follow=valley_follow,
                                  select=valley_select,
                                  guide_follow=valley_guide_follow,
                                  guide_halfwidth=valley_guide_halfwidth,
                                  waist_range=valley_waist_range,
                                  width_range=valley_width_range)
                  if fit_line_on_maps else None)
    if fit_line_on_maps and karten_fit is None:
        print("Hinweis: keine Gerade fuer die Metrik-Karten - fuer "
              f"{_follow_label(valley_follow)} bleiben zu wenige brauchbare "
              "Talpunkte uebrig.")
    out['map_fit_line'] = karten_fit

    # forbidden_factor steuert BEIDES: das Einzeichnen und den Abschnitt im
    # Bericht. Wer die Punkte ausschliesst, ohne die Grenze zu zeichnen,
    # setzt forbidden_draw=False - der Bericht nennt die Grenze trotzdem.
    zeichnen = forbidden_factor if forbidden_draw else None

    stern_fit = None
    if draw_best_point and best_point_follow in MANUAL_LINE_KEYS:
        stern_fit = karten_fit if karten_fit is not None else fit_valley_line(
            results, axis=VALLEY_FIT_AXIS, follow=valley_follow,
            select=valley_select, guide_follow=valley_guide_follow,
            guide_halfwidth=valley_guide_halfwidth,
            waist_range=valley_waist_range, width_range=valley_width_range)
        if stern_fit is None:
            print("Hinweis: fuer den selbst gewaehlten Punkt gibt es keine "
                  "Talpfad-Gerade - ohne sie laesst sich die zweite Koordinate "
                  "nicht bestimmen. Bitte im Dialog beide Koordinaten vorgeben "
                  "(\"Waist UND Width vorgeben\").")
    stern = (best_point_by(results, best_point_follow, value=best_point_value,
                           fit=stern_fit, value2=best_point_value2)
             if draw_best_point else None)
    # waist_um nachtragen: Legende und die senkrechte Linie im Schnitt
    # brauchen es, best_point_by liefert es nur beim manuellen Punkt.
    stern = _mit_waist_um(results, stern)
    out['best_point'] = stern
    if stern is not None and stern.get('outside'):
        print("Hinweis: der selbst gewaehlte Punkt liegt ausserhalb des "
              "gescannten Fensters - dort gibt es keine Daten.")
    if stern is not None and stern.get('at_edge'):
        print("Hinweis: der markierte Punkt liegt auf dem Rand des gescannten "
              "Fensters - dort ist das Minimum vermutlich nur der Fensterrand. Im "
              "Plot ist der Stern deshalb offen statt gefuellt.")
    if (draw_best_point and stern is None
            and best_point_follow not in (None, BEST_POINT_FOLLOW_STORED)):
        print(f"Hinweis: {_follow_label(best_point_follow)} gibt es in diesem "
              "Datensatz nicht - der Stern zeigt den besten Gitterpunkt.")
        stern = best_point_by(results, None)
        out['best_point'] = stern

    # Kein gemeinsamer rc_context mehr: jede Zeichenfunktion oeffnet ihr
    # eigenes dokument_stil(), weil der Massstab an ihrer Figurbreite
    # haengt. legend_fontsize wird als DOKUMENT-Groesse durchgereicht.
    out['plots']['metric_comparison'] = plot_metric_comparison(
        results, prefix, out_dir=plots_dir, win_axis=win_axis,
        draw_best_point=draw_best_point, save=save, show=show,
        confirm_overwrite=confirm_overwrite, fit_line=karten_fit,
        fit_line_dashed_extrapolation=fit_line_dashed_extrapolation,
        forbidden_factor=zeichnen, best_point=stern)
    if amplitude_maps:
        if amplitude_panels(results) is None:
            # Kein Abbruch: die uebrigen Plots sollen nicht an einer
            # fehlenden Zusatzkarte scheitern.
            print("Hinweis: keine Amplituden-Karten - der Datensatz fuehrt "
                  "keine Amplituden (r_x_grid/r_y_grid) mit. Bei einem "
                  "Fest-Amplituden-Scan ist das normal.")
        else:
            out['plots']['metric_comparison_amp'] = plot_metric_comparison(
                results, prefix, out_dir=plots_dir, win_axis=win_axis,
                draw_best_point=draw_best_point, save=save, show=show,
                confirm_overwrite=confirm_overwrite, fit_line=karten_fit,
                fit_line_dashed_extrapolation=fit_line_dashed_extrapolation,
                with_amplitudes=True, forbidden_factor=zeichnen,
                best_point=stern)
            geklemmt = r_bounds_clamped_fraction(results)
            if geklemmt is not None and max(geklemmt) > 0:
                lo, hi = results['r_bounds']
                print(f"Hinweis zu den Amplituden-Karten: r_bounds = "
                      f"({lo:g}, {hi:g}); auf der Schranke liegen "
                      f"{geklemmt[0] * 100:.1f}% der r_x- und "
                      f"{geklemmt[1] * 100:.1f}% der r_y-Werte. Diese Punkte "
                      "sind keine freien Optima - im Bild sind sie ein Plateau.")
    out['plots']['region'] = plot_region(
        results, prefix, out_dir=plots_dir, win_axis=win_axis,
        draw_best_point=draw_best_point, save=save, show=show,
        confirm_overwrite=confirm_overwrite, forbidden_factor=zeichnen,
        best_point=stern)
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
            path_mode=valley_path_mode, forbidden_factor=zeichnen,
            select=valley_select, guide_follow=valley_guide_follow,
            guide_halfwidth=valley_guide_halfwidth,
            waist_range=valley_waist_range, width_range=valley_width_range,
            map_show_path=valley_map_show_path,
            legend_placement=valley_legend_placement,
            draw_best_point=draw_best_point, best_point=stern)
        if braucht_fit:
            out['valley_line'] = fit_valley_line(
                results, axis=valley_axis, follow=valley_follow,
                select=valley_select, guide_follow=valley_guide_follow,
                guide_halfwidth=valley_guide_halfwidth,
                waist_range=valley_waist_range, width_range=valley_width_range)
        if valley_path_mode == "line":
            # Zweite Datei: dieselbe Gerade, aber jede Groesse der Uebersicht
            # in einem eigenen Feld - in derselben Anordnung wie die Karten.
            panels = plot_line_panels(
                results, prefix, axis=valley_axis, follow=valley_follow,
                out_dir=plots_dir, save=save, show=show,
                confirm_overwrite=confirm_overwrite,
                legend_fontsize=legend_fontsize, select=valley_select,
                guide_follow=valley_guide_follow,
                guide_halfwidth=valley_guide_halfwidth,
                waist_range=valley_waist_range, width_range=valley_width_range,
                draw_best_point=draw_best_point, best_point=stern,
                show_extrapolated=valley_map_show_path,
                legend_placement=panel_legend_placement)
            if panels is not None:
                out['plots']['line_panels'] = panels
    if point_cuts:
        if stern is None:
            print("Hinweis: kein Schnitt durch den Punkt - es ist kein Punkt "
                  "markiert (Haken \"Punkt als Stern einzeichnen\").")
        else:
            try:
                out['plots']['point_cuts'] = plot_point_cuts(
                    results, prefix, best=stern, out_dir=plots_dir,
                    win_axis=win_axis, save=save, show=show,
                    confirm_overwrite=confirm_overwrite,
                    legend_fontsize=legend_fontsize)
            except ValueError as exc:
                # Kein Abbruch: die uebrigen Plots sollen nicht an einer
                # fehlenden Zusatzgrafik scheitern.
                print(f"Hinweis: kein Schnitt durch den Punkt - {exc}")
    if plot_scan_overview:
        out['plots']['overview'] = plot_overview(
            results, out_dir=plots_dir, save=save, show=show,
            confirm_overwrite=confirm_overwrite, best_point=stern,
            win_axis=win_axis)

    if save:
        report_path = results_dir / f"{prefix}_Report.md"
        zeige_gerade = (valley_cut and valley_fit_supported(valley_axis)
                        and (valley_fit_line or valley_path_mode == "line"))
        axis_label = dict(VALLEY_AXIS_CHOICES).get(valley_axis) if zeige_gerade else None
        write_report(results, report_path,
                     valley_line=out.get('valley_line'),
                     valley_axis_label=axis_label,
                     valley_path_mode=valley_path_mode,
                     forbidden_factor=forbidden_factor,
                     forbidden_excluded=forbidden_excluded,
                     best_point=stern)
        out['report'] = report_path

    return out


# ======================================================================
# Querschnitt entlang des Minimums ("Talschnitt")
# ======================================================================
# Idee: einer Groesse folgen (der "Fuehrungsgroesse") und pro Spalte bzw.
# Zeile den Punkt suchen, an dem sie minimal ist. Genau an diesen Punkten
# werden dann ALLE gewuenschten Groessen abgelesen - also nicht deren
# eigenes Minimum, sondern ihr Wert dort, wo die Fuehrungsgroesse am besten
# ist. Das beantwortet die Frage: "wenn ich dem Optimum von X folge, was
# machen dabei die anderen Groessen und die Amplituden?"

# Waehlbare Fuehrungsgroessen: key -> Anzeigename (deutsch, Dialog-Eintraege)
FOLLOW_CHOICES = [
    ("score", "J = alpha*Uniformity + (1-alpha)*Crosstalk (Zielgroesse)"),
    ("uniformity", "Uniformity" + (", atom-gewichtet" if _WEIGHTED else ", hart")),
    ("crosstalk", "Crosstalk" + (", atom-gewichtet" if _WEIGHTED else ", hart")),
]

# Waehlbare Kurven im Querschnitt: key -> (Label, Einheit, Farbe, in % ?)
#
# BESCHRIFTUNG: reine Symbole statt ausgeschriebener Namen - im Querschnitt
# stehen die Legendeneintraege nebeneinander unter dem Panel. Der Index h/w
# haengt an der Metrik-Familie dieses Ordners (siehe U_SYMBOL oben).
#
# FARBEN: r_x und r_y behalten das Blau und Orange, das die Amplituden-
# Schnitte des AmplitudeScanPlotter seit jeher haben (matplotlib-Standard
# C0/C1) - dieselbe Groesse sieht damit in beiden Plot-Arten gleich aus. Die
# uebrigen drei sind darauf abgestimmt und NICHT nach Gefuehl gewaehlt:
# Umrechnung nach CIE-Lab, Abstand zusaetzlich unter simulierter Deuteranopie
# und Protanopie geprueft, Helligkeit auf L* <= 66 begrenzt, damit keine Linie
# auf weissem Grund verblasst.
#
# Vorher standen hier Blau (#0072B2) fuer U und Lila (#785EF0) fuer r_x - die
# beiden fallen unter Deuteranopie auf dE = 21.9 zusammen und waren auch
# normalsichtig schwer zu trennen. Lila ist deshalb ganz raus. Der jetzige
# Satz hat als kleinsten Abstand ueber ALLE 10 Paare dE = 31.7 (das Paar
# eta/r_y, die nie auf derselben Achse liegen); die Paare, die im Bild
# tatsaechlich uebereinanderliegen, sind deutlich weiter auseinander:
#   r_x / r_y  (gemeinsame Achse)   dE = 122 / 122 / 101  (normal/deut/prot)
#   U   / eta  (beide in %)         dE = 108 /  56 /  46
TRACE_SPECS = {
    "uniformity": (U_SYMBOL, "%", "#44AA99", True),
    "crosstalk": (C_SYMBOL, "%", "#CC3311", True),
    "score": (r"$J_%s$" % _IDX, "%", "#000000", True),
    "r_x": (r"$r_x$", "", "#1f77b4", False),
    "r_y": (r"$r_y$", "", "#ff7f0e", False),
}

# Reihenfolge der y-Achsen im Querschnitt (nur die angehakten erscheinen)
TRACE_ORDER = ["uniformity", "crosstalk", "score", "r_x", "r_y"]

# Der schlanke Schnitt: NUR die Zielgroesse J. Uniformity, Crosstalk, r_x und
# r_y stehen bereits in der 2x2-Kartenuebersicht und im Kreuzschnitt; neben
# der Karte ergaben sie fuenf Kurven auf drei gestaffelten y-Achsen. Das ist
# die VOREINSTELLUNG der Haken in run_plots.py, keine feste Liste - wer mehr
# sehen will, hakt mehr an. (Combinated_Optimization zeigt hier J_c, U_c und
# eta_c, weil dort die beiden Haelften von J die eigentliche Aussage sind.)
LINE_CUT_TRACES = ["score"]

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
    if scan_data.flavor_of(results) is None:
        return None
    U, C = metric_grids(results)
    if key == "uniformity":
        return U
    if key == "crosstalk":
        return C
    if key == "score":
        return score_from(U, C, alpha)
    if key in ("r_x", "r_y"):
        grid = results.get(f"{key}_grid")
        return None if grid is None else np.asarray(grid, dtype=float)
    return None


def available_follow_keys(results):
    """Welche Fuehrungsgroessen der Datensatz hergibt."""
    return [key for key, _label in FOLLOW_CHOICES if _grid_for(results, key) is not None]


def available_trace_keys(results):
    """Welche Kurven der Datensatz hergibt."""
    return [key for key in TRACE_ORDER if _grid_for(results, key) is not None]


def waist_um_of(results):
    """Effektiver Waist (µm) je win_input-Spalte."""
    return waist_um_vals(results)


def _window_tol(werte):
    """Winzige Toleranz fuer die Bereichsgrenzen: wer im GUI genau den
    angezeigten Randwert eintippt, meint diesen Gitterpunkt."""
    spanne = float(np.max(werte) - np.min(werte))
    return max(1e-9, 1e-4 * spanne)


def search_window_mask(results, waist_range=None, width_range=None):
    """True, wo der Talpfad ueberhaupt gesucht werden darf.

    Beide Bereiche sind (von, bis) in µm bzw. MHz, None = keine
    Einschraenkung. Gebraucht, wenn ein Datensatz mehrere Talzweige hat und
    man dem Fit sagen will, welcher gemeint ist."""
    waist = waist_um_of(results)
    width = np.asarray(results["width_vals"], dtype=float) * 1e-6
    ok_col = np.ones(len(waist), dtype=bool)
    ok_row = np.ones(len(width), dtype=bool)
    if waist_range is not None:
        lo, hi = sorted(float(v) for v in waist_range)
        tol = _window_tol(waist)
        ok_col = (waist >= lo - tol) & (waist <= hi + tol)
    if width_range is not None:
        lo, hi = sorted(float(v) for v in width_range)
        tol = _window_tol(width)
        ok_row = (width >= lo - tol) & (width <= hi + tol)
    return ok_row[:, None] & ok_col[None, :]


def _apply_search_window(grid, results, waist_range, width_range):
    """Kopie des Gitters, ausserhalb des Suchbereichs auf NaN."""
    if waist_range is None and width_range is None:
        return np.asarray(grid, dtype=float)
    maske = search_window_mask(results, waist_range, width_range)
    return np.where(maske, np.asarray(grid, dtype=float), np.nan)


def _boundary_mask(target, rows, cols, axis):
    """Welche Talpunkte sind Randminima?

    Beurteilt auf dem UNGEFENSTERTEN Gitter. Rand heisst:

    1. am Rand des gescannten Gitters,
    2. ein Nachbar fehlt - NaN im Scan oder ausgeschlossener verbotener
       Bereich; das "Minimum" ist dann nur die Stelle, wo die Daten aufhoeren,
    3. ein Nachbar ist ECHT KLEINER als der Punkt selbst.

    Punkt 3 ist der Fall "Suchbereich": legt man ein Fenster um einen
    Talzweig, liegen seine Punkte zwangslaeufig teils auf der Fenstergrenze.
    Ein solcher Punkt ist trotzdem brauchbar, SOLANGE es ausserhalb nicht
    weiter bergab geht. Ohne Suchbereich aendert Punkt 3 nichts."""
    n_rows, n_cols = target.shape
    rand = np.zeros(len(rows), dtype=bool)
    for k, (i, j) in enumerate(zip(rows, cols)):
        wert = target[i, j]
        if axis == "width":
            am = (j == 0) or (j == n_cols - 1)
            nachbarn = () if am else (target[i, j - 1], target[i, j + 1])
        else:
            am = (i == 0) or (i == n_rows - 1)
            nachbarn = () if am else (target[i - 1, j], target[i + 1, j])
        if not am:
            am = not all(np.isfinite(v) for v in nachbarn)
        if not am and np.isfinite(wert):
            am = any(v < wert for v in nachbarn)
        rand[k] = bool(am)
    return rand


def _local_min_indices(spalte):
    """Indizes echter lokaler Minima: beide Nachbarn vorhanden und groesser.

    Ein Punkt am Rand des Gitters oder mit einem NaN-Nachbarn zaehlt NICHT -
    genau das schliesst die Punkte aus, die nur am Scan-Fenster oder am
    ausgeschlossenen verbotenen Bereich anliegen."""
    s = np.asarray(spalte, dtype=float)
    idx = []
    for i in range(1, len(s) - 1):
        if not np.isfinite(s[i - 1:i + 2]).all():
            continue
        if s[i] < s[i - 1] and s[i] < s[i + 1]:
            idx.append(i)
    return idx


def guide_line(results, guide_follow=GUIDE_FOLLOW_DEFAULT,
               waist_range=None, width_range=None):
    """Die Leitgerade: gewoehnlicher Fit der Leitgroesse ueber der µm-Achse.
    None, wenn der Datensatz sie nicht hergibt.

    Der Suchbereich gilt AUCH fuer die Leitgerade - sonst waere sie auf einem
    anderen Talzweig bestimmt als der Fit, den sie fuehren soll."""
    if _grid_for(results, guide_follow) is None:
        return None
    return fit_valley_line(results, axis=VALLEY_FIT_AXIS, follow=guide_follow,
                           select="global", waist_range=waist_range,
                           width_range=width_range)


def extract_valley(results, axis="waist_um", follow="score",
                   select="global", guide_follow=GUIDE_FOLLOW_DEFAULT,
                   guide_halfwidth=GUIDE_HALFWIDTH_DEFAULT,
                   waist_range=None, width_range=None):
    """Sucht den Talpfad der Fuehrungsgroesse und liest dort alle Groessen ab.

    axis="waist_um"/"waist_mm": pro waist-Spalte den width-Index mit dem
        kleinsten Wert der Fuehrungsgroesse.
    axis="width": pro width-Zeile den waist-Index.

    Gibt ein dict zurueck mit x (die Schnittachse, aufsteigend sortiert),
    x_label, den Koordinaten jedes Talpunkts, den Gitterindizes (rows/cols)
    und values[key] fuer jede verfuegbare Groesse - jeweils GENAU am Talpunkt
    abgelesen, ohne Interpolation."""
    if axis not in _VALLEY_AXIS_TO_WIN_AXIS:
        raise ValueError(f"axis muss eine von {list(_VALLEY_AXIS_TO_WIN_AXIS)} sein, nicht {axis!r}.")
    target = _grid_for(results, follow)
    if target is None:
        raise ValueError(f"Der Datensatz enthaelt die Fuehrungsgroesse {follow!r} nicht.")

    win_input_vals = np.asarray(results["win_input_vals"], dtype=float)
    width_vals = np.asarray(results["width_vals"], dtype=float)
    waist_um = waist_um_of(results)

    # Suchbereich: alles ausserhalb ist fuer die Minimumsuche nicht
    # vorhanden. Abgelesen werden die Werte spaeter trotzdem aus den
    # ungekuerzten Gittern.
    target_voll = np.asarray(target, dtype=float)
    target = _apply_search_window(target, results, waist_range, width_range)
    finite = np.isfinite(target)
    if not finite.any():
        raise ValueError(
            "Im eingestellten Suchbereich liegt kein einziger gueltiger "
            "Gitterpunkt der Fuehrungsgroesse.")
    safe = np.where(finite, target, np.inf)

    fuehrung = None
    n_ohne_kandidat = 0
    if select == "guided":
        fuehrung = guide_line(results, guide_follow,
                              waist_range=waist_range, width_range=width_range)
        if fuehrung is None:
            print(f"Hinweis: keine Leitgerade aus {_follow_label(guide_follow)} - "
                  "der Talpfad faellt auf das globale Minimum je Spalte zurueck.")
            select = "global"

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
        x_label = (WAIST_UM_LABEL if axis == "waist_um"
                   else WIN_INPUT_LABEL)

    if select == "guided":
        # Die Leitgerade ist immer width = a*waist + b ueber der µm-Achse
        # (VALLEY_FIT_AXIS); beim Schnitt ueber width wird sie umgestellt.
        width_mhz_all = width_vals * 1e-6
        neue_rows, neue_cols = [], []
        for zeile, spalte in zip(rows, cols):
            if axis == "width":
                if fuehrung["a"] == 0:
                    n_ohne_kandidat += 1
                    continue
                ziel = (width_mhz_all[zeile] - fuehrung["b"]) / fuehrung["a"]
                kandidaten = _local_min_indices(target[zeile, :])
                koord = waist_um
            else:
                ziel = fuehrung["a"] * waist_um[spalte] + fuehrung["b"]
                kandidaten = _local_min_indices(target[:, spalte])
                koord = width_mhz_all
            im_korridor = [i for i in kandidaten
                           if abs(koord[i] - ziel) <= guide_halfwidth]
            if not im_korridor:
                n_ohne_kandidat += 1
                continue
            treffer = min(im_korridor, key=lambda i: abs(koord[i] - ziel))
            if axis == "width":
                neue_rows.append(zeile); neue_cols.append(treffer)
            else:
                neue_rows.append(treffer); neue_cols.append(spalte)
        if len(neue_rows) < VALLEY_FIT_MIN_POINTS:
            print("Hinweis: die Leitgerade findet fast nirgends ein lokales "
                  "Minimum im Korridor - der Talpfad faellt auf das globale "
                  "Minimum je Spalte zurueck.")
            select = "global"
            n_ohne_kandidat = 0
        else:
            rows = np.array(neue_rows, dtype=int)
            cols = np.array(neue_cols, dtype=int)
            x = (width_vals[rows] * 1e-6 if axis == "width"
                 else (waist_um[cols] if axis == "waist_um"
                       else win_input_vals[cols] * 1e3))

    reihenfolge = np.argsort(x)          # x aufsteigend, damit die Linie sauber laeuft
    rows, cols, x = rows[reihenfolge], cols[reihenfolge], x[reihenfolge]

    # Liegt das Minimum am Rand des gescannten Fensters, ist es vermutlich
    # gar kein echtes Minimum, sondern nur der Rand des Scans.
    am_rand = _boundary_mask(target_voll, rows, cols, axis)
    if select == "guided":
        # Im gefuehrten Modus sind alle Punkte per Konstruktion echte lokale
        # Minima mit zwei vorhandenen Nachbarn - es gibt keine Randminima.
        am_rand = np.zeros(len(rows), dtype=bool)

    values = {}
    for key in available_trace_keys(results):
        grid = _grid_for(results, key)
        werte = grid[rows, cols]
        _label, _unit, _color, as_percent = TRACE_SPECS[key]
        values[key] = werte * 100.0 if as_percent else werte

    waist_heat = (win_input_vals[cols] * 1e3
                  if _VALLEY_AXIS_TO_WIN_AXIS[axis] == "before_lens"
                  else waist_um[cols])
    return dict(
        path_mode="valley",
        axis=axis, follow=follow, x=x, x_label=x_label,
        rows=rows, cols=cols, values=values, boundary=am_rand,
        win_input=win_input_vals[cols],
        n_boundary=int(am_rand.sum()),
        extrapolated=np.zeros(len(x), dtype=bool), n_extrapolated=0,
        waist_um=waist_um[cols], waist_mm=win_input_vals[cols] * 1e3,
        width_MHz=width_vals[rows] * 1e-6,
        # Koordinaten fuer die Heatmap: x = Waist in der Konvention der
        # Achse, y = width. Beim Talpfad liegen sie auf Gitterpunkten, beim
        # Geradenschnitt dazwischen - beides wird gleich gezeichnet.
        x_heat=waist_heat, y_heat=width_vals[rows] * 1e-6,
        n_points=len(x),
        n_total=(target.shape[1] if axis != "width" else target.shape[0]),
        n_outside=n_ohne_kandidat,
        waist_range=(None if waist_range is None else tuple(float(v) for v in waist_range)),
        width_range=(None if width_range is None else tuple(float(v) for v in width_range)),
        select=select, guide_follow=(guide_follow if select == "guided" else None),
        guide_halfwidth=(float(guide_halfwidth) if select == "guided" else None),
        guide=(fuehrung if select == "guided" else None),
        alpha=float(results.get("alpha", 0.7)),
    )


# Fuer die Plots: knappe englische Bezeichnungen (FOLLOW_CHOICES bleibt
# deutsch, das sind die Dialog-Eintraege). Das Symbol steht mit dabei, damit
# Colorbar und Legendeneintrag derselben Groesse zusammenpassen.
FOLLOW_PLOT_LABELS = {
    "score": r"Objective $J = \alpha U_%s + (1-\alpha)\eta_%s$" % (_IDX, _IDX),
    "uniformity": U_TITLE,
    "crosstalk": C_TITLE,
}


# Fuer die Colorbar des Talschnitts noch einmal kuerzer: dort steht die
# Beschriftung HOCHKANT neben einer schmalen Farbleiste, und seit die Schrift
# am Dokument haengt (dokument_stil) ist der ausgeschriebene Score laenger als
# die Leiste hoch ist - er lief oben aus der Figur. Das Symbol genuegt: was J,
# U und eta bedeuten, steht im Bericht und in den Titeln der Metrik-Karten.
FOLLOW_CBAR_LABELS = {
    "score": r"$J_%s$" % _IDX,
    "uniformity": U_SYMBOL,
    "crosstalk": C_SYMBOL,
}

# Ueberschrift der KARTE im Schnittplot. Sie nennt, WAS die Karte zeigt -
# nicht, in welchem Modus geschnitten wurde: "Linear fit" bzw. "Minimum path"
# beschrieb die Linie darin, nicht die Heatmap darunter. Welcher Modus laeuft,
# sagen die Linie selbst und ihr Legendeneintrag; der Titel der Karte gehoert
# der Groesse. Dieselbe Formulierung wie ueber der J-Karte (plot_region).
FOLLOW_MAP_TITLES = {
    "score": "Combined score (%s)" % FAMILY_TITLE,
    "uniformity": U_TITLE,
    "crosstalk": C_TITLE,
}


def follow_map_title(follow):
    return FOLLOW_MAP_TITLES.get(follow, follow_plot_label(follow))


def follow_plot_label(follow):
    """Englische Kurzbezeichnung fuer Plot-Titel und Colorbar."""
    return FOLLOW_PLOT_LABELS.get(follow, follow)


def follow_cbar_label(follow):
    """Ganz kurze Fassung fuer die hochkant stehende Colorbar-Beschriftung."""
    return FOLLOW_CBAR_LABELS.get(follow, follow_plot_label(follow))


def _follow_label(follow):
    for key, label in FOLLOW_CHOICES:
        if key == follow:
            return label
    return follow


# ----------------------------------------------------------------------
# Achsen-Buendelung im Querschnitt
# ----------------------------------------------------------------------
# Eine eigene y-Achse je Kurve wird ab vier Kurven unleserlich. Kurven mit
# DERSELBEN EINHEIT und derselben Groessenordnung teilen sich deshalb eine
# Achse; erst wenn sich die Wertebereiche um mehr als eine Groessenordnung
# unterscheiden, kommt eine zweite dazu. r_x und r_y landen immer zusammen -
# es ist dieselbe Groesse in zwei Richtungen.
AXIS_GROUP_MAX_RATIO = 10.0
# Zweite Bedingung: eine Kurve, die sich eine Achse teilt, muss dort noch
# etwas zu sehen geben - sie muss mindestens diesen Anteil der gemeinsamen
# Achsenspanne fuer sich beanspruchen.
AXIS_GROUP_MIN_SHARE = 0.10
# ... mit einer Ausnahme: eine Kurve, deren eigene Schwankung groesstenteils
# Rauschen ist, soll KEINE enge eigene Achse bekommen. Sonst fuellt das
# Zickzack die ganze Achse und sieht nach viel aus, obwohl es nichts
# bedeutet. Mass ist das Verhaeltnis aus Spanne und mittlerer zweiter
# Differenz.
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
    """Duerfen sich diese Kurven eine y-Achse teilen?"""
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

    Zuerst nach Einheit trennen (Prozent, dimensionslos), dann innerhalb
    einer Einheit nach Groessenordnung buendeln. Gibt eine Liste von
    Schluessel-Listen zurueck, in der Reihenfolge von TRACE_ORDER."""
    nach_einheit = {}
    for key in keys:
        nach_einheit.setdefault(TRACE_SPECS[key][1], []).append(key)

    gruppen = []
    for _einheit, einheit_keys in nach_einheit.items():
        zusammen = [k for k in einheit_keys if k in AXIS_GROUP_ALWAYS_TOGETHER]
        einzeln = [k for k in einheit_keys if k not in AXIS_GROUP_ALWAYS_TOGETHER]
        if zusammen:
            gruppen.append(zusammen)

        bereiche = {k: _wertebereich(values[k]) for k in einzeln}
        # Kurven ohne brauchbaren Bereich (alles NaN/0) bekommen eine eigene
        # Achse, damit sie die Buendelung nicht durcheinanderbringen.
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
    """Achsenbereich aufweiten, falls das Zickzack sonst die Achse dominiert.
    Es wird nichts abgeschnitten - die Kurve wird nur kleiner gezeichnet,
    damit hochfrequentes Rauschen nicht wie ein Signal aussieht."""
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


# Feinste Tick-Schrittweite je Einheit - die UNTERGRENZE der Aufloesung
# einer Schnitt-Achse.
#
# Ohne sie zoomt matplotlib auf die tatsaechliche Schwankung: r_x laeuft in
# manchen Datensaetzen nur zwischen 0.9995 und 1.0020, das
# Diskretisierungsrauschen des Intensitaetsgitters fuellt dann die ganze
# Achse und sieht wie ein Verlauf aus. Mit dem Raster steht r_x da, was es
# ist: konstant 1.00 auf zwei Nachkommastellen.
#
#   ""  (r_x, r_y)  -> 0.01, also hoechstens zwei Nachkommastellen
#   "%" (U, eta, J) -> 0.1,  also hoechstens eine
ACHSEN_MINDESTSCHRITT = {"": 0.01, "%": 0.1}
# So viele Schritte muessen mindestens auf die Achse passen (sonst wird sie
# aufgeweitet) ...
ACHSEN_MIN_SCHRITTE = 4
# ... und hoechstens so viele Striche sollen es werden (sonst wird die
# Schrittweite vergroessert: 1, 2, 5, 10, 20, 50, ... x Mindestschritt).
ACHSEN_MAX_TICKS = 7
_RASTER_FAKTOREN = [f * 10 ** k for k in range(7) for f in (1, 2, 5)]


def _achsenraster(ax, einheit):
    """Feste Tick-Aufloesung fuer eine Schnitt-Achse.

    Weitet den Bereich auf, bis mindestens ACHSEN_MIN_SCHRITTE Schritte der
    Einheit hineinpassen, waehlt daraus eine Schrittweite mit hoechstens
    ACHSEN_MAX_TICKS Strichen und rundet die Grenzen nach aussen auf ein
    Vielfaches davon. Die Zahl der Nachkommastellen folgt der Schrittweite -
    ein Tick zeigt nie mehr Stellen, als er unterscheidet."""
    schritt0 = ACHSEN_MINDESTSCHRITT.get(einheit)
    if schritt0 is None:
        return
    lo, hi = (float(v) for v in ax.get_ylim())
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    mitte = 0.5 * (lo + hi)
    spanne = max(hi - lo, schritt0 * ACHSEN_MIN_SCHRITTE)
    lo, hi = mitte - spanne / 2, mitte + spanne / 2
    schritt = schritt0 * _RASTER_FAKTOREN[-1]
    for faktor in _RASTER_FAKTOREN:
        if spanne / (schritt0 * faktor) <= ACHSEN_MAX_TICKS:
            schritt = schritt0 * faktor
            break
    ax.set_ylim(np.floor(lo / schritt) * schritt,
                np.ceil(hi / schritt) * schritt)
    ax.yaxis.set_major_locator(MultipleLocator(schritt))
    stellen = max(0, int(np.ceil(-np.log10(schritt) - 1e-9)))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{stellen}f"))


# Eine helle Kurvenfarbe ist als LINIE gut zu sehen, als Achsenbeschriftung
# auf weissem Grund aber zu blass. Ticks und Achsenlabel werden deshalb
# abgedunkelt, wenn die Farbe zu hell ist - die Linie selbst behaelt ihre
# Farbe, sonst gehoerten Kurve und Achse optisch nicht mehr zusammen.
AXIS_LABEL_MAX_LUMA = 0.45


def _achsenfarbe(color):
    r, g, b = matplotlib.colors.to_rgb(color)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luma <= AXIS_LABEL_MAX_LUMA:
        return color
    f = AXIS_LABEL_MAX_LUMA / luma
    return (r * f, g * f, b * f)


def _axis_label_for_group(gruppe):
    labels = [TRACE_SPECS[k][0] for k in gruppe]
    einheit = TRACE_SPECS[gruppe[0]][1]
    text = ", ".join(labels)
    return f"{text} [{einheit}]" if einheit else text


def _break_at(werte, unbenutzt):
    """Kopie mit NaN an den nicht benutzten Stellen - so laesst matplotlib
    die Linie dort abreissen, statt quer durchs Bild zu verbinden."""
    v = np.array(werte, dtype=float)
    if unbenutzt is not None and len(unbenutzt) == len(v):
        v[np.asarray(unbenutzt, dtype=bool)] = np.nan
    return v


def _unused_mask(valley, fit):
    """Welche Punkte des Pfads gehen nicht in die Auswertung ein? Am Rand des
    Scan-Fensters gefundene Minima immer; zusaetzlich die vom Fit
    ausgeschlossenen, falls einer vorliegt."""
    unbenutzt = np.asarray(valley["boundary"], dtype=bool).copy()
    if fit is not None and fit.get("excluded_mask") is not None:
        raus = np.asarray(fit["excluded_mask"], dtype=bool)
        if len(raus) == len(unbenutzt):
            unbenutzt |= raus
    return unbenutzt


def plot_valley_cut(results, prefix, axis="waist_um", follow="score", traces=None,
                    out_dir=None, save=True, show=False, confirm_overwrite=None,
                    legend_fontsize=9, fit_line=False, path_mode="valley",
                    forbidden_factor=None, select="global",
                    guide_follow=GUIDE_FOLLOW_DEFAULT,
                    guide_halfwidth=GUIDE_HALFWIDTH_DEFAULT,
                    waist_range=None, width_range=None,
                    map_show_path=False, draw_best_point=True, best_point=None,
                    legend_placement=VALLEY_LEGEND_PLACEMENT_DEFAULT):
    """Querschnitt: links die Heatmap der Fuehrungsgroesse mit dem Pfad,
    rechts der Schnitt entlang dieses Pfads.

    traces: Liste der anzuzeigenden Schluessel (siehe TRACE_ORDER). None =
    alle, die der Datensatz hergibt. Die Fuehrungsgroesse selbst wird immer
    mitgezeichnet.

    fit_line: zusaetzlich eine Gerade durch den brauchbaren Teil des
    Talpfads legen und in die Heatmap zeichnen.

    path_mode: "valley" = Schnitt entlang des Minimums, "line" = Schnitt
    entlang genau dieser Geraden ueber den ganzen gescannten Bereich.

    map_show_path: was in der KARTE (linkes Panel) ausser der Heatmap zu
    sehen ist. Standardmaessig nur der verbotene Bereich und die
    Ausgleichsgerade - die einzelnen Pfadpunkte (benutzt, ausgelassen,
    extrapoliert) werden dann WEDER gezeichnet NOCH benannt; ihre Zahlen
    stehen im Bericht. Mit map_show_path=True kommen sie samt Legende zurueck.

    Ausnahme: gibt es gar keine Gerade, bliebe die Karte sonst leer - dann
    wird der Talpfad auch ohne Haken gezeichnet."""
    out_dir = paths.fit_plots_dir() if out_dir is None else out_dir
    valley = extract_path(results, axis=axis, follow=follow, path_mode=path_mode,
                          select=select, guide_follow=guide_follow,
                          guide_halfwidth=guide_halfwidth,
                          waist_range=waist_range, width_range=width_range)

    verfuegbar = available_trace_keys(results)
    if traces is None:
        # Ohne Vorgabe der schlanke Satz (LINE_CUT_TRACES), nicht mehr alles,
        # was der Datensatz hergibt.
        gewaehlt = [k for k in LINE_CUT_TRACES if k in verfuegbar] or list(verfuegbar)
    else:
        gewaehlt = [k for k in TRACE_ORDER if k in traces and k in verfuegbar]
    if follow in verfuegbar and follow not in gewaehlt:
        gewaehlt.insert(0, follow)
    if not gewaehlt:
        raise ValueError("Keine einzige darstellbare Kurve ausgewaehlt.")

    win_axis = _VALLEY_AXIS_TO_WIN_AXIS[axis]
    x_heat, x_heat_label, reversed_ = win_axis_values(results, win_axis)
    width_vals = np.asarray(results["width_vals"], dtype=float)
    target = _grid_for(results, follow)
    skala = 100.0 if TRACE_SPECS.get(follow, (None, None, None, False))[3] else 1.0
    Z = (target[:, ::-1] if reversed_ else target) * skala

    # Die Buendelung (und damit die Achsenskalierung) richtet sich nur nach
    # den BENUTZTEN Punkten. Sonst zieht ein einzelner Ausreisser am Rand des
    # Scan-Fensters die Achse auf und drueckt die eigentliche Kurve platt.
    fit_fuer_maske = (valley["fit"] if path_mode == "line"
                      else (_fit_line_through_valley(valley, win_axis)
                            if fit_line else None))
    unbenutzt = (np.zeros(valley["n_points"], dtype=bool) if path_mode == "line"
                 else _unused_mask(valley, fit_fuer_maske))
    werte_benutzt = {k: _break_at(v, unbenutzt) for k, v in valley["values"].items()}
    gruppen = group_traces_by_axis(gewaehlt, werte_benutzt)
    extra = max(0, len(gruppen) - 1)

    # Feste Figurgroesse - und zwar unabhaengig von der Zahl der y-Achsen.
    #
    # Das war frueher anders (11.4 + 0.85 je zusaetzlicher Achse), und mit
    # fester Schriftgroesse ergab das auch Sinn: die breitere Figur wurde im
    # Dokument staerker verkleinert, die Schrift wurde also kleiner und die
    # Achsen passten. Genau dieses Kleinerwerden ist der Fehler, den
    # dokument_stil() behebt - seither skaliert die Schrift mit der Breite
    # mit, und Breiter-Machen aendert am Bild IM DOKUMENT nichts mehr
    # (alles waechst gleichmaessig, die Datei wird nur groesser).
    #
    # Was im Dokument ankommt, haengt damit allein an dieser Geometrie:
    # 6.3 x 2.87 Zoll bei Textbreite. Der Platz fuer zusaetzliche y-Achsen
    # muss darin stecken - er kostet Panel-Breite, und das ist ehrlich so:
    # fuenf y-Achsen nebeneinander sind fuenf y-Achsen nebeneinander.
    # dichte=1.0 (nicht ZWEI_PANEL_DICHTE): seit die Legenden UNTER den
    # Panels stehen, ueberragt die Kartenlegende die Karte nicht mehr - der
    # Grund fuer die kleinere Stufe ist damit weg, und die Beschriftung darf
    # so gross sein wie in den uebrigen Dokument-Abbildungen. Combinated_
    # Optimization setzt fuer denselben Plot dieselben Groessen.
    breite, hoehe = VALLEY_FIGSIZE
    with dokument_stil(breite, legend_fontsize=legend_fontsize, dichte=1.0):
        fig, (ax_map, ax_cut) = plt.subplots(
            1, 2, figsize=(breite, hoehe), constrained_layout=True,
            gridspec_kw={"width_ratios": list(VALLEY_WIDTH_RATIOS)})

        # ---------------- links: Karte mit Pfad ----------------
        # Knappe Karte (Default): nur der verbotene Bereich und die Gerade.
        # Die einzelnen Pfadpunkte werden dann gar nicht erst gezeichnet -
        # nicht bloss aus der Legende genommen. Ihre Zahlen stehen im Bericht.
        #
        # Ohne Gerade waere die Karte sonst leer; dann wird der Pfad auch ohne
        # Haken gezeichnet, sonst zeigte das linke Panel nur die Heatmap.
        zeige_pfad = bool(map_show_path) or (
            fit_fuer_maske is None and path_mode != "line")

        im = ax_map.pcolormesh(x_heat, width_vals * 1e-6, Z, shading="auto", cmap="magma_r")
        label = follow_cbar_label(follow) + (" [%]" if skala == 100.0 else "")
        fig.colorbar(im, ax=ax_map, label=label)
        ax_map.set_xlabel(kurzes_achsenlabel(x_heat_label))
        # Die Karte ist das schmalere der beiden Panels; die automatische
        # Teilung setzt dort mehr Striche, als Zahlen nebeneinander passen
        # (sie kennt die Textbreite nicht) - sie klebten aneinander.
        ax_map.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax_map.set_ylabel(WIDTH_LABEL)
        if forbidden_factor is not None:
            draw_forbidden_region(ax_map, results, win_axis, forbidden_factor)

        x_pfad = np.asarray(valley["x_heat"], dtype=float)
        y_pfad = np.asarray(valley["y_heat"], dtype=float)

        if path_mode == "line":
            fit = valley["fit"]
            ax_map.set_title(follow_map_title(follow))
            if zeige_pfad:
                # Der echte Talpfad blass im Bild - nur so ist zu sehen, wie
                # weit die Gerade von den tatsaechlichen Minima abweicht.
                tal = extract_valley(results, axis=axis, follow=follow, select=select,
                                     guide_follow=guide_follow,
                                     guide_halfwidth=guide_halfwidth,
                                     waist_range=waist_range, width_range=width_range)
                tal_unbenutzt = _unused_mask(tal, fit)
                ax_map.plot(_break_at(tal["x_heat"], tal_unbenutzt),
                            _break_at(tal["y_heat"], tal_unbenutzt),
                            linewidth=S(0.9), color="#8a8a8a", marker="o",
                            markersize=S(2.8), label="minimum path (reference)")
                if tal_unbenutzt.any():
                    ax_map.plot(np.asarray(tal["x_heat"])[tal_unbenutzt],
                                np.asarray(tal["y_heat"])[tal_unbenutzt],
                                label=f"not used ({int(tal_unbenutzt.sum())})",
                                **SD(UNUSED_STYLE))
            # Die Gerade laeuft IMMER durch das ganze Bild - dieselbe
            # Funktion und derselbe Anblick wie in den Metrik-Karten.
            #
            # Vorher wurde hier statt ihrer der SCHNITTPFAD gezeichnet. Der
            # endet, wo die Gerade das gescannte width-Fenster verlaesst
            # (oder wo ein Suchbereich sie abschneidet) - die Linie hoerte
            # dadurch mitten in der Karte auf, obwohl dieselbe Gerade in der
            # Kartenuebersicht daneben durchlief.
            draw_fit_line_on_map(ax_map, results, fit, win_axis)
            extrap = np.asarray(valley["extrapolated"], dtype=bool)
            if zeige_pfad:
                # Die tatsaechlich geschnittenen Stellen als Punkte AUF der
                # Geraden - ohne eigene Linie, die ist schon gezeichnet.
                ax_map.plot(x_pfad, y_pfad, linestyle="none", marker="o",
                            markersize=S(3.2), color=VALLEY_FIT_STYLE["color"],
                            markeredgecolor="white", markeredgewidth=S(0.4),
                            label=(f"cut along the fitted line "
                                   f"({valley['n_points']}/{valley['n_total']} pts)"))
                if extrap.any():
                    ax_map.plot(x_pfad[extrap], y_pfad[extrap],
                                label=f"extrapolated ({int(extrap.sum())})",
                                **SD(EXTRAPOLATED_MARKER))
        else:
            ax_map.set_title(follow_map_title(follow))
            fit = fit_fuer_maske
            if zeige_pfad:
                ax_map.plot(_break_at(x_pfad, unbenutzt), _break_at(y_pfad, unbenutzt),
                            linewidth=S(1.2), color="red", marker="o", markersize=S(3.4),
                            markeredgecolor="white", markeredgewidth=S(0.5),
                            label=f"minimum path ({int((~unbenutzt).sum())}/"
                                  f"{valley['n_total']} pts)")
                if unbenutzt.any():
                    ax_map.plot(x_pfad[unbenutzt], y_pfad[unbenutzt],
                                label=f"not used ({int(unbenutzt.sum())})",
                                **SD(UNUSED_STYLE))
            if fit is not None:
                # Auch hier durch das ganze Bild, nicht nur von der ersten
                # bis zur letzten benutzten Stuetzstelle.
                draw_fit_line_on_map(ax_map, results, fit, win_axis)
        if draw_best_point:
            draw_best_point_marker(ax_map, results, win_axis, legend=False,
                                   label=True, best=best_point)
        # Ohne einen einzigen Eintrag gar keinen leeren Kasten zeichnen.
        # Die Eintraege der Karte nur SAMMELN - gezeichnet werden beide
        # Legenden erst am Ende, gemeinsam und in derselben Schriftgroesse
        # (siehe _schnitt_legenden).
        karten_handles, _karten_labels = ax_map.get_legend_handles_labels()
        karten_zeilen = [[h] for h in karten_handles]

        # ------------- rechts: Querschnitt, eine y-Achse je Gruppe -------------
        linien = []
        for position, gruppe in enumerate(gruppen):
            ax = ax_cut if position == 0 else ax_cut.twinx()
            if position > 1:
                ax.spines["right"].set_position(
                    ("outward", S(ZUSATZ_ACHSE_PT) * (position - 1)))
                ax.set_frame_on(True)
                ax.patch.set_visible(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.spines["right"].set_visible(True)
            for key in gruppe:
                label, _unit, color, _pct = TRACE_SPECS[key]
                marker = "o" if key == follow else None
                # Nicht benutzte Punkte tauchen hier gar nicht auf: die Linie
                # reisst dort ab. Welche das sind, zeigt die Karte links.
                linie, = ax.plot(valley["x"], werte_benutzt[key],
                                 color=color, marker=marker, markersize=S(2.8),
                                 linewidth=S(1.5), label=label)
                linien.append(linie)
            _entzerre_achse(ax, [werte_benutzt[k] for k in gruppe])
            _achsenraster(ax, TRACE_SPECS[gruppe[0]][1])
            achsen_label = _axis_label_for_group(gruppe)
            achsen_farbe = _achsenfarbe(TRACE_SPECS[gruppe[0]][2] if len(gruppe) == 1
                                        else MULTI_TRACE_AXIS_COLOR)
            ax.set_ylabel(achsen_label, color=achsen_farbe)
            ax.tick_params(axis="y", colors=achsen_farbe)
            if position > 0:
                ax.spines["right"].set_color(achsen_farbe)

        # Der Strich bekommt einen eigenen Eintrag in der Legende des
        # Schnitts, und zwar denselben Text wie der Stern in der Karte
        # links: seine beiden Koordinaten. Beide meinen denselben Punkt -
        # zwei verschiedene Namen dafuer in einer Figur waeren eine Frage,
        # die sich der Leser nicht stellen soll.
        marke = None
        if draw_best_point:
            gezeichnet = draw_working_point_line(
                ax_cut, _mit_waist_um(results, best_point), axis, label=True)
            marke = gezeichnet if gezeichnet is not False else None
        # Auch hier die Kurzform: bei zwei Panels auf Textbreite lief die
        # ausgeschriebene Fassung rechts aus der Figur heraus. Welche
        # Konvention gemeint ist, sagt das Symbol - und die Karte links
        # traegt dieselbe Achse.
        ax_cut.set_xlabel(kurzes_achsenlabel(valley["x_label"]))
        # Ueberschrift des rechten Panels. Im Geradenmodus ist der Schnitt
        # entlang der Fit-Geraden gelegt, im Talmodus entlang des Minimums -
        # beides heisst "Cross-section along ...", damit die beiden Fassungen
        # nebeneinander als dasselbe Bild erkennbar bleiben.
        # In beiden Pfadmodi derselbe Titel: was geschnitten wurde, sagt der
        # Titel der Karte links ("Linear fit" bzw. "Minimum path").
        ax_cut.set_title("Cross section")
        ax_cut.grid(True, alpha=0.25)
        # Die Kurven in eine Zeile (bei vielen umgebrochen), der
        # Arbeitspunkt in eine eigene Zeile darunter.
        # Die Kurven bleiben NEBENEINANDER in einer Zeile. Der Arbeitspunkt
        # traegt nur noch die Koordinate der Schnittachse und ist damit kurz
        # genug, um sich eine Zeile mit einer Kurve zu teilen - mindestens
        # zwei Eintraege pro Zeile, statt ihn immer allein zu setzen.
        ncol = legend_ncol([ln.get_label() for ln in linien])
        if marke is not None:
            ncol = max(2, ncol)
        zeilen = [linien[i:i + ncol] for i in range(0, len(linien), ncol)]
        if marke is not None:
            if zeilen and len(zeilen[-1]) < ncol:
                zeilen[-1].append(marke)
            else:
                zeilen.append([marke])
        _schnitt_legenden(fig, ax_map, karten_zeilen, ax_cut, zeilen,
                          ncol=ncol, placement=legend_placement)

        pfad_tag = "line" if path_mode == "line" else "valley"
        dateiname = f"{prefix}_{pfad_tag}_{follow}_over_{axis}.pdf"
        # tight=False: feste Seitengroesse, unabhaengig von Datensatz und
        # Legendentext - siehe _finish().
        return _finish(fig, out_dir, dateiname, save, show, confirm_overwrite,
                       tight=False)


# ======================================================================
# Gerade durch den brauchbaren Teil des Talpfads
# ======================================================================
# Im vorderen Bereich laeuft der Talpfad sichtbar gerade; weiter hinten
# faellt das Minimum auf den Rand des gescannten Fensters oder springt auf
# einen zweiten, davon getrennten Nebenzweig. Eine Gerade durch ALLE
# Talpunkte waere dadurch verfaelscht - deshalb werden die unbrauchbaren
# Punkte in drei Stufen aussortiert:
#
#   1. Minima am Rand des gescannten Fensters (extract_valley() markiert sie
#      bereits als `boundary`).
#   2. Sprungerkennung: bleibt danach eine zweite, abgesetzte Punktwolke
#      uebrig, wird nur das groesste zusammenhaengende Segment behalten.
#   3. Rand-Kinks: an den beiden Enden wird iterativ abgeschnitten, solange
#      der Randpunkt deutlich neben der Ausgleichsgeraden liegt.
#
# Ausgeschlossene Punkte werden nicht verschwiegen, sondern im Plot markiert
# und im Bericht gezaehlt.
#
# Stufe 2 und 3 folgen drop_disconnected_branch()/drop_edge_kinks() aus
# fit_waist_width_relation.py. Bewusst kopiert statt importiert: jenes Modul
# setzt beim Import global plt.rcParams (Serifen-Stil) und legt Ordner an -
# das wuerde das Aussehen aller uebrigen Plots still veraendern. Zwei
# bewusste Abweichungen: die Funktionen geben eine Maske statt geteilter
# Arrays zurueck und sortieren nicht selbst (hier gilt die Reihenfolge
# ENTLANG DES PFADS), und die Sprung-Schwelle bekommt einen Boden in Hoehe
# des typischen Schritts.

VALLEY_FIT_STYLE = dict(color="#00c2ff", linewidth=2.2, linestyle="--")
# Punkte, die nicht in die Auswertung eingehen: offen gezeichnet und NICHT
# durch die Pfadlinie verbunden. Randminima und Fit-Ausschluesse teilen sich
# bewusst EINE Markierung - fuer den Betrachter ist beides dasselbe.
UNUSED_STYLE = dict(linestyle="none", marker="o", markersize=5.5,
                    markerfacecolor="none", markeredgecolor="#222222",
                    markeredgewidth=1.2)
VALLEY_FIT_MIN_POINTS = 4
VALLEY_JUMP_FACTOR = 6.0

# Die Gerade gibt es NUR fuer den effektiven Waist in µm nach der Linse.
# Ueber win_input (mm) ist der Zusammenhang gar nicht linear; ueber width
# (MHz) waere es dieselbe Beziehung, nur andersherum aufgetragen - der
# Einheitlichkeit mit Combinated_Optimization halber bleibt sie auch dort
# gesperrt, damit "Gerade" im Dialog eindeutig an der µm-Achse haengt.
VALLEY_FIT_AXIS = "waist_um"


def valley_fit_supported(axis):
    """Laesst sich fuer diese Schnittachse eine Gerade legen? Nur fuer den
    effektiven Waist in µm - siehe VALLEY_FIT_AXIS."""
    return axis == VALLEY_FIT_AXIS


def valley_fit_axis_hint():
    """Ein Satz fuer GUI-Tooltips und Fehlermeldungen."""
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
    # Deshalb ist der typische Schritt selbst die untere Schranke der Skala.
    #
    # Faellt MEHR als die Haelfte der Schritte auf exakt 0 - der Pfad bleibt
    # dann ueber mehrere Spalten in derselben Gitterzeile, der Normalfall bei
    # flachem Tal oder feinem width-Gitter -, wird auch dieser Median 0 und
    # die Schwelle rutscht wieder auf ~0: jeder einzelne Gitterschritt gaelte
    # als Sprung, uebrig bliebe die laengste Folge gleicher Werte und damit
    # eine entartete, waagerechte "Gerade". Ein Nullschritt ist aber gar kein
    # Schritt - der typische Schritt ist der Median derer, die einer sind.
    schritte_echt = steps[steps > 0]
    median_echt = float(np.median(schritte_echt)) if schritte_echt.size else 0.0
    scale = max(1.4826 * mad, median_step, median_echt, 1e-12)
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

    Gefittet wird immer ENTLANG des Pfads: bei einem Schnitt ueber den Waist
    ist der Waist die unabhaengige Groesse (width = a*waist + b), bei einem
    Schnitt ueber width ist es width. None, wenn zu wenige brauchbare Punkte
    uebrig bleiben oder die Schnittachse gar keinen Fit erlaubt."""
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
        sigma_a, sigma_b = _fit_uncertainty(t[keep], u[keep], ss_res)
    else:
        # Entartet: alle verbliebenen Punkte haben denselben u-Wert. Dann ist
        # die Gerade waagerecht, und ein R² gibt es nicht. polyfit lieferte
        # hier sonst eine Steigung der Groessenordnung 1e-15 -
        # Rundungsrauschen, das im Bericht wie ein Ergebnis aussaehe.
        a, b, r2 = 0.0, float(np.mean(u[keep])), float("nan")
        sigma_a = sigma_b = float("nan")

    raus = (~keep) & (~rand)                          # in Stufe 2/3 verworfen
    t_ends = np.array([float(np.min(t[keep])), float(np.max(t[keep]))])
    u_ends = a * t_ends + b
    # Endpunkte der Strecke in Heatmap-Koordinaten (x = Waist, y = width)
    if ueber_width:
        x_line, y_line = u_ends, t_ends
    else:
        x_line, y_line = t_ends, u_ends

    return dict(
        a=float(a), b=float(b), r2=float(r2),
        sigma_a=float(sigma_a), sigma_b=float(sigma_b),
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
        select=valley.get("select", "global"),
        guide_follow=valley.get("guide_follow"),
        guide_halfwidth=valley.get("guide_halfwidth"),
        guide=valley.get("guide"),
        n_no_candidate=int(valley.get("n_outside") or 0),
        n_columns=int(valley.get("n_total") or 0),
        waist_range=valley.get("waist_range"), width_range=valley.get("width_range"),
    )


def valley_fit_diagnosis(results, axis="waist_um", follow="score",
                         select="global", guide_follow=GUIDE_FOLLOW_DEFAULT,
                         guide_halfwidth=GUIDE_HALFWIDTH_DEFAULT,
                         waist_range=None, width_range=None):
    """Warum kommt keine Gerade heraus? Klartext statt "kein Fit".

    None, wenn eine Gerade zustande kommt."""
    if not valley_fit_supported(axis):
        return valley_fit_axis_hint()
    try:
        valley = extract_valley(results, axis=axis, follow=follow, select=select,
                                guide_follow=guide_follow,
                                guide_halfwidth=guide_halfwidth,
                                waist_range=waist_range, width_range=width_range)
    except ValueError as exc:
        return str(exc)
    if _fit_line_through_valley(valley, _VALLEY_AXIS_TO_WIN_AXIS[axis]) is not None:
        return None

    rand = np.asarray(valley["boundary"], dtype=bool)
    n_ges = int(len(rand))
    n_rand = int(rand.sum())
    n_frei = n_ges - n_rand
    zeilen = [
        f"Im gesuchten Bereich liegen {n_ges} Talpunkte ({_follow_label(follow)}).",
        f"Davon sind {n_rand} Randminima und fallen heraus, "
        f"{n_frei} bleiben - gebraucht werden {VALLEY_FIT_MIN_POINTS}.",
    ]
    if n_rand:
        waist = np.asarray(valley["waist_um"], dtype=float)[rand]
        width = np.asarray(valley["width_MHz"], dtype=float)[rand]
        orte = ", ".join(f"{a:.3f} µm / {b:.3f} MHz" for a, b in zip(waist, width))
        zeilen += ["", "Die Randminima liegen bei: " + orte + "."]
        gr = _search_window_edges(results, waist_range, width_range, waist, width)
        if gr:
            zeilen += [
                "",
                "Sie kleben an " + gr + ". Dort geht es ausserhalb weiter bergab: "
                "der Talzweig verlaesst den eingestellten Bereich, das Minimum "
                "darin ist nur dessen Rand. Solche Punkte in den Fit zu nehmen "
                "hiesse, die eigene Grenze zu fitten."]
    if n_frei >= VALLEY_FIT_MIN_POINTS:
        zeilen += ["",
                   "Die verbleibenden Punkte wurden in Stufe 2/3 verworfen "
                   "(abgesetzter Nebenzweig oder Kink am Rand)."]
    zeilen += ["",
               "Moeglichkeiten: den Bereich weiter fassen, eine andere "
               "Fuehrungsgroesse waehlen - oder feiner rechnen, wenn der Zweig im "
               "Scan-Fenster schlicht zu kurz ist."]
    return "\n".join(zeilen)


def _search_window_edges(results, waist_range, width_range, waist, width):
    """Klartext, an welcher Grenze des Suchbereichs die Punkte kleben."""
    treffer = []
    for bereich, werte, name, einheit in (
            (waist_range, waist, "Waist", "µm"),
            (width_range, width, "width", "MHz")):
        if bereich is None or not len(werte):
            continue
        lo, hi = sorted(float(v) for v in bereich)
        tol = 1e-3 * max(abs(hi - lo), 1e-12)
        if np.any(np.abs(werte - lo) <= tol):
            treffer.append(f"der unteren {name}-Grenze ({lo:.3f} {einheit})")
        if np.any(np.abs(werte - hi) <= tol):
            treffer.append(f"der oberen {name}-Grenze ({hi:.3f} {einheit})")
    return " und ".join(treffer)


def fit_valley_line(results, axis="waist_um", follow="score",
                    select="global", guide_follow=GUIDE_FOLLOW_DEFAULT,
                    guide_halfwidth=GUIDE_HALFWIDTH_DEFAULT,
                    waist_range=None, width_range=None):
    """Talpfad bestimmen und eine Gerade durch dessen brauchbaren Teil legen.

    None, wenn die Achse keinen Fit erlaubt (nur VALLEY_FIT_AXIS) oder zu
    wenige brauchbare Talpunkte uebrig bleiben."""
    if not valley_fit_supported(axis):
        return None
    valley = extract_valley(results, axis=axis, follow=follow, select=select,
                            guide_follow=guide_follow, guide_halfwidth=guide_halfwidth,
                            waist_range=waist_range, width_range=width_range)
    return _fit_line_through_valley(valley, _VALLEY_AXIS_TO_WIN_AXIS[axis])


def _fit_uncertainty(t, u, ss_res):
    """Standardfehler von Steigung und Achsenabschnitt einer Geraden.

    Die uebliche Formel der linearen Regression:

        s^2    = SS_res / (n - 2)              (Reststreuung)
        S_tt   = sum (t - mean(t))^2
        sig_a  = sqrt(s^2 / S_tt)
        sig_b  = sig_a * sqrt(sum t^2 / n)

    Sie beschreibt, wie stark a und b schwanken wuerden, wenn man den Scan
    mit gleichartigem Rauschen wiederholte - NICHT, wie gut die Gerade das
    Problem beschreibt; dafuer steht R^2 daneben.

    Weil die Talpunkte Ergebnisse einer Optimierung auf einem Gitter sind
    und keine unabhaengigen Messungen, ist die Zahl eine untere Schranke.

    nan bei weniger als drei Punkten - durch zwei geht eine Gerade exakt.
    """
    n = int(len(t))
    if n < 3:
        return float("nan"), float("nan")
    S_tt = float(np.sum((t - np.mean(t)) ** 2))
    if S_tt <= 0:
        return float("nan"), float("nan")
    s2 = float(ss_res) / (n - 2)
    sigma_a = float(np.sqrt(s2 / S_tt))
    sigma_b = float(sigma_a * np.sqrt(float(np.sum(np.asarray(t, float) ** 2)) / n))
    return sigma_a, sigma_b


def _mit_fehler(wert, sigma, stellen=6):
    """'0.28316 +- 0.00440' - oder nur den Wert, wenn es keinen Fehler gibt.

    Der Fehler bekommt zwei signifikante Stellen; mehr suggeriert eine
    Genauigkeit, die eine Schaetzung aus wenigen Punkten nicht hat.
    """
    if sigma is None or not np.isfinite(sigma) or sigma <= 0:
        return f"{wert:.{stellen}g}"
    nk = max(0, int(-np.floor(np.log10(sigma))) + 1)
    return f"{wert:.{nk}f} +- {sigma:.{nk}f}"


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


def valley_line_formula_with_error(fit):
    """Wie valley_line_formula, aber mit den Standardfehlern - fuer den
    Bericht, wo Platz dafuer ist."""
    zeichen = "-" if fit["b"] < 0 else "+"
    a_txt = _mit_fehler(fit["a"], fit.get("sigma_a"), stellen=5)
    b_txt = _mit_fehler(abs(fit["b"]), fit.get("sigma_b"), stellen=5)
    if " +- " in a_txt:
        a_txt = f"({a_txt})"
    if " +- " in b_txt:
        b_txt = f"({b_txt})"
    return (f"{fit['u_plain']}/{fit['u_unit']} = {a_txt} · "
            f"{fit['t_plain']}/{fit['t_unit']} {zeichen} {b_txt}")


def _valley_selection_lines(fit):
    """Wie die Talpunkte gewaehlt wurden - ohne das ist die Steigung nicht
    nachvollziehbar."""
    if fit is None:
        return []
    bereich = []
    if fit.get("waist_range") or fit.get("width_range"):
        teile = []
        if fit.get("waist_range"):
            teile.append("waist %.4f .. %.4f µm" % fit["waist_range"])
        if fit.get("width_range"):
            teile.append("width %.4f .. %.4f MHz" % fit["width_range"])
        bereich = [
            "- **Suchbereich eingeschraenkt auf " + " und ".join(teile) + ".** "
            "Ausserhalb wurde gar nicht erst nach einem Minimum gesucht. Ein "
            "Talpunkt auf der Grenze dieses Bereichs zaehlt trotzdem, solange es "
            "ausserhalb nicht weiter bergab geht - er ist dann ein echtes lokales "
            "Minimum, das die Grenze nur streift. Geht es draussen tiefer, faellt "
            "er als Randminimum heraus, denn dort waere nicht das Tal gefittet, "
            "sondern die eingestellte Grenze selbst.",
        ]
    if fit.get("select") != "guided":
        return bereich + ["- Talpunkte: globales Minimum der Fuehrungsgroesse je Spalte."]
    fuehrung = fit.get("guide") or {}
    zeilen = [
        f"- Talpunkte: je Spalte das LOKALE Minimum, das der Leitgeraden am "
        f"naechsten liegt (Korridor +-{fit.get('guide_halfwidth', 0):.3f} MHz).",
        "  Lokal heisst: beide Nachbarn vorhanden und groesser - Punkte am Rand des "
        "Scan-Fensters und Punkte, die an den ausgeschlossenen verbotenen Bereich "
        "grenzen, kommen damit gar nicht erst in Frage.",
    ]
    if fuehrung:
        zeilen.append(
            f"- Leitgerade aus {_follow_label(fit.get('guide_follow'))}: "
            f"width/MHz = {fuehrung['a']:.5f} * waist/µm {fuehrung['b']:+.6f} "
            f"(R² = {_r2_text(fuehrung['r2'])}, {fuehrung['n_used']} Punkte).")
    if fit.get("n_no_candidate"):
        zeilen.append(
            f"- In {fit['n_no_candidate']} von {fit.get('n_columns', 0)} Spalten lag "
            f"kein lokales Minimum im Korridor; diese Spalten fehlen im Pfad.")
    zeilen = bereich + zeilen
    zeilen.append(
        "- **Einordnung:** die Leitgerade WAEHLT nur aus, sie verschiebt nichts - "
        "die Punkte sind echte lokale Minima der Fuehrungsgroesse und die Steigung "
        "ist deren eigene. Welcher der mehreren Minima-Zweige verfolgt wird, "
        "entscheidet aber die Leitgroesse. Diese Zahl ist also an sie gebunden und "
        "kein unabhaengiger Befund.")
    return zeilen


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
        valley_line_formula_with_error(fit),
        "```",
        "",
        f"- Steigung a = {_mit_fehler(fit['a'], fit.get('sigma_a'))} "
        f"{fit['u_unit']}/{fit['t_unit']}",
        f"- Achsenabschnitt b = {_mit_fehler(fit['b'], fit.get('sigma_b'))} "
        f"{fit['u_unit']}",
        f"- R² = {_r2_text(fit['r2'])}"
        + ("  (die verbliebenen Punkte liegen alle auf demselben Wert - eine "
           "Varianz, die eine Gerade erklaeren koennte, gibt es hier nicht)"
           if not np.isfinite(fit['r2']) else ""),
        f"- gefitteter Bereich: {fit['t_min']:.4f} .. {fit['t_max']:.4f} {fit['t_unit']}",
        f"- verwendete Talpunkte: {fit['n_used']} von {fit['n_total']}",
        f"- ausgeschlossen: {fit['n_boundary']} mit Minimum am Rand des gescannten "
        f"Fensters, {fit['n_excluded']} auf einem abgesetzten Nebenzweig bzw. als "
        f"Rand-Kink",
        "",
    ]
    if np.isfinite(fit.get('sigma_a', float('nan'))):
        lines += [
            "Die angegebenen Fehler sind die Standardfehler der linearen "
            "Regression (`s² = SS_res/(n-2)`, daraus `sigma_a = sqrt(s²/S_tt)`). "
            "Sie sagen, wie stark a und b schwanken wuerden, wenn man den Scan "
            "mit gleichartigem Rauschen wiederholte - NICHT, wie gut eine Gerade "
            "das Problem beschreibt; dafuer steht R² daneben.",
            "",
            "**Sie sind eine untere Schranke.** Die Talpunkte sind Ergebnisse "
            "einer Optimierung auf einem Gitter, keine unabhaengigen Messungen. "
            "Gitterschrittweite und die Wahl der Fuehrungsgroesse verschieben die "
            "Gerade um deutlich mehr, als hier herauskommt.",
            "",
        ]
    lines += _valley_selection_lines(fit)
    lines += [
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
            "entlang des Minimums - und zwar ueber den ganzen gescannten Bereich, "
            "also auch ausserhalb des oben genannten Fit-Bereichs (dort ist er "
            "extrapoliert; im Plot mit offenen Kreisen markiert). Da die Gerade die "
            "Gitterpunkte nicht trifft, sind die abgelesenen Werte zwischen den "
            "beiden benachbarten Gitterzeilen linear interpoliert.",
            "",
        ]
    return lines


def line_points_for_axis(results, fit, win_axis, n=240):
    """Die Fit-Gerade als Punktfolge in den Koordinaten EINER Karte.

    Die Gerade ist in width ueber dem effektiven Waist (µm) definiert. Auf
    einer µm-Achse ist sie deshalb wirklich gerade; auf der mm-Achse nicht,
    weil win_input und effektiver Waist nichtlinear zusammenhaengen - dort
    wird sie dicht abgetastet und als Polygonzug gezeichnet.

    Gibt (x_innen, y_innen, x_aussen, y_aussen) zurueck: innerhalb und
    ausserhalb des Bereichs, aus dem die Gerade bestimmt wurde."""
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


def draw_fit_line_on_map(ax, results, fit, win_axis, dashed_extrapolation=False):
    """Die Fit-Gerade in eine beliebige (Waist, width)-Karte zeichnen, ueber
    den ganzen gescannten Bereich.

    Standardmaessig eine EINZIGE durchgezogene Linie - eine Gerade ist eine
    Gerade, und der Bereich, aus dem sie bestimmt wurde, steht im Bericht.

    dashed_extrapolation=True zeichnet sie wie frueher zweiteilig:
    durchgezogen im gefitteten Teil, gepunktet ausserhalb. Der gepunktete Teil
    bekommt bewusst KEINEN eigenen Legendeneintrag - die Legende der Karten
    soll genau einen Eintrag haben; der Unterschied steckt im Linienformat.
    Auch die Fitparameter stehen nicht in der Legende, sie sind im Bericht
    nachzulesen."""
    if fit is None:
        return
    x_in, y_in, x_out, y_out = line_points_for_axis(results, fit, win_axis)
    if not dashed_extrapolation:
        # Beide Teile wieder zu einem Zug zusammensetzen: sie sind
        # komplementaer (line_points_for_axis setzt jeweils NaN, wo der andere
        # Teil liegt) und bereits nach x sortiert.
        x_all = np.where(np.isfinite(x_in), x_in, x_out)
        y_all = np.where(np.isfinite(y_in), y_in, y_out)
        ax.plot(x_all, y_all, color=VALLEY_FIT_STYLE["color"], linewidth=S(2.0),
                linestyle="-", label=FIT_LINE_LABEL)
        return
    ax.plot(x_in, y_in, color=VALLEY_FIT_STYLE["color"], linewidth=S(2.0),
            linestyle="-", label=FIT_LINE_LABEL)
    if np.any(np.isfinite(x_out)):
        ax.plot(x_out, y_out, color=VALLEY_FIT_STYLE["color"], linewidth=S(1.4),
                linestyle=":", label="_nolegend_")


def draw_valley_line(ax, fit, legend_fontsize=None):
    """Gerade in die Heatmap zeichnen. `fit=None` zeichnet nichts."""
    if fit is None:
        return
    ax.plot(fit["x_line"], fit["y_line"], label=FIT_LINE_LABEL, **SD(VALLEY_FIT_STYLE))


# ======================================================================
# Schnitt entlang der Geraden (statt entlang des Minimums)
# ======================================================================
# Der Talpfad springt dort, wo das Minimum flach ist oder aus dem gescannten
# Fenster laeuft. Die Gerade aus _fit_line_through_valley() tut das nicht:
# sie ist ueber den GANZEN gescannten Bereich definiert. Ein Schnitt entlang
# dieser Geraden ist deshalb glatt und zeigt, was die Metriken taeten, wenn
# man der linearen Beziehung folgte statt dem tatsaechlichen (teils
# verrauschten) Minimum.
#
# Die Gerade trifft die Gitterpunkte nicht - deshalb wird pro Spalte (bzw.
# Zeile) LINEAR zwischen den beiden benachbarten Gitterwerten interpoliert.
# Wo die Gerade das gescannte Fenster verlaesst, gibt es keine Daten; solche
# Spalten fallen heraus und werden gezaehlt (`n_outside`). Punkte ausserhalb
# des Bereichs, aus dem die Gerade bestimmt wurde, sind echte EXTRAPOLATION
# und werden als solche markiert (`extrapolated`).

PATH_MODE_CHOICES = [
    ("valley", "Talpfad (Minimum der Fuehrungsgroesse)"),
    ("line", "Gerade durch den Talpfad (auch extrapoliert)"),
]

EXTRAPOLATED_MARKER = dict(linestyle="none", marker="o", markersize=6.0,
                           markerfacecolor="white", markeredgecolor="#00647f",
                           markeredgewidth=1.3)


def _interp_at(werte, koordinate, ziel):
    """Linear zwischen den beiden Nachbarn interpolieren. NaN, wenn `ziel`
    ausserhalb des Koordinatenbereichs liegt."""
    ordnung = np.argsort(koordinate)
    c = np.asarray(koordinate, dtype=float)[ordnung]
    v = np.asarray(werte, dtype=float)[ordnung]
    if not np.isfinite(ziel) or ziel < c[0] or ziel > c[-1]:
        return float("nan")
    i = int(np.clip(np.searchsorted(c, ziel), 1, len(c) - 1))
    spanne = c[i] - c[i - 1]
    t = 0.0 if spanne == 0 else (ziel - c[i - 1]) / spanne
    return float((1.0 - t) * v[i - 1] + t * v[i])


# ======================================================================
# Panel-Plot: dieselben Karten, aber als Schnitte
# ======================================================================
# Anordnung EXAKT wie plot_metric_comparison(with_amplitudes=True), also
# Uniformity und Crosstalk oben, r_x und r_y unten - nur steht in jedem Feld
# der SCHNITT entlang der Geraden statt der Karte. Wer die beiden Dateien
# nebeneinander legt, findet jede Groesse an derselben Stelle.
#
# Welche Karte zu welcher Kurve gehoert, sagt diese Tabelle: die Karten sind
# ueber Gitternamen definiert, die Kurven ueber TRACE_SPECS-Schluessel.
_PANEL_KEY_TO_TRACE = {
    "uniformity_grid": "uniformity",
    "uniformity_weighted_grid": "uniformity",
    "crosstalk_grid": "crosstalk",
    "eta_weighted_grid": "crosstalk",
    "r_x_grid": "r_x",
    "r_y_grid": "r_y",
}


def line_panels(results):
    """Die Felder des Panel-Plots, in der Reihenfolge der Metrik-Karten."""
    felder = list(metric_panels(results))
    amp = amplitude_panels(results)
    if amp is not None:
        felder = felder + amp
    verfuegbar = available_trace_keys(results)
    raus = []
    for panel in felder:
        key = _PANEL_KEY_TO_TRACE.get(panel["key"])
        if key is None or key not in verfuegbar:
            continue
        # Die y-Achse traegt nur das Symbol; was die Groesse ist, sagt der
        # Titel darueber. Zweimal derselbe Name im selben Feld kostet nur
        # Platz.
        einheit = TRACE_SPECS[key][1]
        ylabel = TRACE_SPECS[key][0] + (f" [{einheit}]" if einheit else "")
        raus.append(dict(trace=key, title=panel["title"], ylabel=ylabel))
    return raus


def plot_line_panels(results, prefix, axis="waist_um", follow="score",
                     out_dir=None, save=True, show=False, confirm_overwrite=None,
                     legend_fontsize=9, select="global",
                     guide_follow=GUIDE_FOLLOW_DEFAULT,
                     guide_halfwidth=GUIDE_HALFWIDTH_DEFAULT,
                     waist_range=None, width_range=None, path=None,
                     draw_best_point=True, best_point=None,
                     show_extrapolated=False,
                     legend_placement=VALLEY_LEGEND_PLACEMENT_DEFAULT):
    """Jede Groesse der Uebersicht als Schnitt entlang der Geraden.

    `path`: fertiges Ergebnis von extract_line_cut() - sonst wird es selbst
    geholt. Der Aufrufer reicht es durch, damit Gerade und Schnitt in beiden
    Dateien garantiert dieselben sind.

    Gibt None zurueck, wenn der Datensatz keine der Groessen hergibt.
    """
    out_dir = paths.fit_plots_dir() if out_dir is None else out_dir
    if path is None:
        path = extract_line_cut(results, axis=axis, follow=follow, select=select,
                                guide_follow=guide_follow,
                                guide_halfwidth=guide_halfwidth,
                                waist_range=waist_range, width_range=width_range)
    felder = line_panels(results)
    if not felder:
        return None

    x = np.asarray(path["x"], dtype=float)
    extrap = (np.asarray(path["extrapolated"], dtype=bool) if show_extrapolated
              else np.zeros(len(x), dtype=bool))

    n_rows = (len(felder) + 1) // 2
    figsize = HALF_PAGE_FIGSIZE if n_rows >= 2 else ROW_FIGSIZE
    with dokument_stil(figsize[0], legend_fontsize=legend_fontsize):
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharex="all",
                                 squeeze=False, constrained_layout=True)
        for ax, feld in zip(axes.flat, felder):
            y = np.asarray(path["values"][feld["trace"]], dtype=float)
            ax.plot(x, y, color=TRACE_SPECS[feld["trace"]][2], linewidth=S(1.6))
            # Bei einer Legende JE PANEL braucht auch jedes Panel seine
            # eigenen Eintraege; bei der gemeinsamen Legende unter der Figur
            # genuegt einer, sonst stuende dasselbe vier- bzw. sechsmal drin.
            eigene_legende = (legend_placement == "inside")
            if extrap.any():
                ax.plot(x[extrap], y[extrap],
                        label=(f"extrapolated ({int(extrap.sum())})"
                               if eigene_legende else "_nolegend_"),
                        **SD(EXTRAPOLATED_MARKER))
            if draw_best_point:
                # In jedem Feld dieselbe Stelle - so liest man alle Werte am
                # Arbeitspunkt in einem Blick ab.
                draw_working_point_line(
                    ax, best_point, axis,
                    label=(eigene_legende or ax is axes.flat[0]))
            _achsenraster(ax, TRACE_SPECS[feld["trace"]][1])
            ax.set_ylabel(feld["ylabel"])
            ax.set_title(feld["title"])
            ax.grid(True, alpha=0.25)
        for ax in axes.flat[len(felder):]:
            ax.set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xlabel(path["x_label"])

        if legend_placement == "inside":
            # Je Panel eine eigene Legende oben rechts, wie im Kreuzschnitt.
            # `set_in_layout(False)`: sonst schruempfte constrained_layout
            # jedes Panel um die Flaeche seiner Legende.
            for ax in axes.flat[:len(felder)]:
                handles, labels = ax.get_legend_handles_labels()
                if not handles:
                    continue
                # Volle Legendenschrift, nicht die verkleinerte des
                # Schnittplots: hier steht je Feld nur EIN kurzer Eintrag,
                # der passt ohne Nachhelfen neben die Kurve.
                legende = ax.legend(handles, labels, loc="upper right",
                                    ncol=legend_ncol(labels), framealpha=0.9)
                legende.set_in_layout(False)
        else:
            handles, labels = axes.flat[0].get_legend_handles_labels()
            if extrap.any():
                handles.append(plt.Line2D([], [], **SD(EXTRAPOLATED_MARKER)))
                labels.append(f"extrapolated ({int(extrap.sum())})")
            if handles:
                fig.legend(handles, labels, loc="outside lower center",
                           ncol=legend_ncol(labels), framealpha=0.9)

        dateiname = f"{prefix}_line_panels_over_{axis}.pdf"
        return _finish(fig, out_dir, dateiname, save, show, confirm_overwrite)


def extract_line_cut(results, axis="waist_um", follow="score", fit=None,
                     select="global", guide_follow=GUIDE_FOLLOW_DEFAULT,
                     guide_halfwidth=GUIDE_HALFWIDTH_DEFAULT,
                     waist_range=None, width_range=None):
    """Schnitt entlang der Fit-Geraden statt entlang des Minimums.

    Liefert dasselbe dict-Format wie extract_valley(), damit beide von
    plot_valley_cut() gleich behandelt werden - nur sind die Werte hier
    zwischen den Gitterzeilen interpoliert statt direkt abgelesen."""
    if axis not in _VALLEY_AXIS_TO_WIN_AXIS:
        raise ValueError(f"axis muss eine von {list(_VALLEY_AXIS_TO_WIN_AXIS)} sein, nicht {axis!r}.")
    if not valley_fit_supported(axis):
        raise ValueError(
            f"Ein Schnitt entlang der Geraden ist fuer die Achse {axis!r} nicht "
            f"vorgesehen. {valley_fit_axis_hint()}")
    win_axis = _VALLEY_AXIS_TO_WIN_AXIS[axis]
    if fit is None:
        fit = fit_valley_line(results, axis=axis, follow=follow, select=select,
                              guide_follow=guide_follow, guide_halfwidth=guide_halfwidth,
                              waist_range=waist_range, width_range=width_range)
    if fit is None:
        raise ValueError(
            "Fuer diesen Datensatz liess sich keine Gerade durch den Talpfad legen "
            "(zu wenige brauchbare Talpunkte) - ein Schnitt entlang der Geraden ist "
            "damit nicht moeglich. Bitte den Talpfad-Modus verwenden.")

    win_input_vals = np.asarray(results["win_input_vals"], dtype=float)
    width_vals = np.asarray(results["width_vals"], dtype=float)
    width_mhz = width_vals * 1e-6
    waist_um = waist_um_of(results)
    waist_mm = win_input_vals * 1e3
    waist_heat = waist_mm if win_axis == "before_lens" else waist_um

    keys = available_trace_keys(results)
    gitter = {key: _grid_for(results, key) for key in keys}
    ziel_gitter = _grid_for(results, follow)

    ueber_width = axis == "width"
    x_liste, waist_liste, width_liste, extrap_liste, index_liste = [], [], [], [], []
    werte_liste = {key: [] for key in keys}
    n_aussen = 0

    # t ist die unabhaengige Groesse des Fits: beim Schnitt ueber den Waist
    # der Waist, beim Schnitt ueber width die width.
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
    # win_input je Schnittpunkt: beim Schnitt ueber den Waist sitzt jeder
    # Punkt auf einer Scan-Spalte, der Wert ist also exakt; beim Schnitt ueber
    # width liegt der Waist zwischen den Spalten und wird mitinterpoliert.
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
    else:
        x_label = (WAIST_UM_LABEL if axis == "waist_um"
                   else WIN_INPUT_LABEL)
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


def extract_path(results, axis="waist_um", follow="score", path_mode="valley",
                 select="global", guide_follow=GUIDE_FOLLOW_DEFAULT,
                 guide_halfwidth=GUIDE_HALFWIDTH_DEFAULT,
                 waist_range=None, width_range=None):
    """Talpfad ODER Geradenschnitt - je nach `path_mode`."""
    if path_mode == "valley":
        return extract_valley(results, axis=axis, follow=follow, select=select,
                              guide_follow=guide_follow, guide_halfwidth=guide_halfwidth,
                              waist_range=waist_range, width_range=width_range)
    if path_mode == "line":
        return extract_line_cut(results, axis=axis, follow=follow, select=select,
                                guide_follow=guide_follow, guide_halfwidth=guide_halfwidth,
                                waist_range=waist_range, width_range=width_range)
    raise ValueError(f"path_mode muss 'valley' oder 'line' sein, nicht {path_mode!r}.")


def path_mode_label(path_mode):
    for key, label in PATH_MODE_CHOICES:
        if key == path_mode:
            return label
    return path_mode
