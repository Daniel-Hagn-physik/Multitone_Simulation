"""
lib/single_beam_report.py - Plots und Bericht fuer den Einzelstrahl-Sweep.

Alle Kurven laufen ueber dem WAIST; oben traegt jede Figur eine zweite
x-Achse mit dem zugehoerigen Waist vor der ersten Linse (die beiden
haengen ueber waist = C/win_input zusammen, die Achse ist deshalb nicht
linear - das ist kein Fehler, sondern die Beziehung selbst).

Uniformity und Crosstalk stehen IMMER zusammen in einer Figur. Ob sie sich
eine y-Achse teilen, entscheidet dieselbe Regel wie im Querschnitt der
Multitone-Auswertung (`report.group_traces_by_axis`): gleiche Einheit,
Wertebereiche innerhalb einer Groessenordnung, und jede Kurve muss auf der
gemeinsamen Achse noch etwas zu sehen geben. Sonst bekommt die zweite
Kurve ihre eigene Achse rechts, in ihrer Farbe beschriftet.

Stil, Schriftgroessen und Farben kommen unveraendert aus `report.py` -
dieselben PDFs sollen neben den Multitone-Bildern im selben Dokument
stehen, ohne dass man den Bruch sieht.
"""

from datetime import date
from pathlib import Path as FilePath

import numpy as np
import matplotlib.pyplot as plt

from . import paths  # noqa: F401
from . import single_beam as sb

# Bewusst aus report.py uebernommen statt neu geschrieben: Massstab,
# Farben und die Achsen-Buendelung sollen in beiden Auswertungen dieselben
# sein. Die mit _ beginnenden Namen sind modulintern gedacht, werden hier
# aber genau so benutzt wie dort - eine zweite Fassung waere die
# schlechtere Loesung.
from .report import (  # noqa: E402
    AXIS_GROUP_MAX_RATIO, BEST_POINT_STYLE, TRACE_SPECS,
    ZUSATZ_ACHSE_PT, S, SD, _entzerre_achse, _finish, _ist_verrauscht,
    _wertebereich, dokument_stil,
)


# ======================================================================
# Welche Kurve wie heisst und aussieht
# ======================================================================
# Die vier Metriken erben Symbol und Farbe aus TRACE_SPECS (dort sind die
# Farbabstaende geprueft, auch unter Farbenblindheit). Die beiden
# Kombi-Groessen sind neu; sie bekommen die beiden Farben, die in diesem
# Ordner sonst r_x/r_y tragen - Amplituden gibt es beim Einzelstrahl nicht,
# der geprueffte Satz bleibt also vollstaendig und kollisionsfrei.
#
# Eintrag: (Symbol, Einheit, Farbe, in Prozent rechnen) - genau die Form,
# die TRACE_SPECS in report.py hat. Einen festen Linienstil gibt es NICHT:
# gezeichnet wird durchgezogen, und nur wo zwei Kurven im Bild aufeinander
# liegen, wird eine von ihnen gestrichelt (siehe _entwirre_ueberlapp).
TRACES = {
    "uniformity_hart":      TRACE_SPECS["uniformity_hard"],
    "crosstalk_hart":       TRACE_SPECS["crosstalk_hard"],
    # Werden beide Regionen gerechnet, tragen die zwei Kurven ihre Region
    # im Exponenten. Der Kreis behaelt das Amber von eta_h - er ist
    # dieselbe Groesse -, das Pitch-Quadrat bekommt das Violett, das in
    # diesem Ordner sonst r_x traegt (Amplituden gibt es beim Einzelstrahl
    # nicht).
    "crosstalk_hart_kreis": (r"$\eta_h^{\mathrm{circ}}$",) + TRACE_SPECS["crosstalk_hard"][1:],
    "crosstalk_hart_pitch": (r"$\eta_h^{\mathrm{box}}$", "%", "#785EF0", True),
    "uniformity_weighted":  TRACE_SPECS["uniformity_weighted"],
    "crosstalk_weighted":   TRACE_SPECS["crosstalk_weighted"],
    "uniformity_kombi":     (r"$U_c$", "%", "#785EF0", True),
    "crosstalk_kombi":      (r"$\eta_c$", "%", "#CC3311", True),
    "combined_score":       TRACE_SPECS["penalty_raw"],
}

# Reihenfolge, in der Achsen und Legendeneintraege sortiert werden.
TRACE_ORDER = ["uniformity_hart", "crosstalk_hart",
               "crosstalk_hart_kreis", "crosstalk_hart_pitch",
               "uniformity_weighted", "crosstalk_weighted",
               "uniformity_kombi", "crosstalk_kombi", "combined_score"]

# Dieselben Groessen in Klartext - fuer Bericht und Meldungsfenster, wo
# kein Mathe-Satz stattfindet und "\\eta_h" nur haesslich waere.
TEXT_LABELS = {
    "uniformity_hart": "U_h",
    "crosstalk_hart": "eta_h",
    "crosstalk_hart_kreis": "eta_h^circ",
    "crosstalk_hart_pitch": "eta_h^box",
    "uniformity_weighted": "U_w",
    "crosstalk_weighted": "eta_w",
    "uniformity_kombi": "U_c",
    "crosstalk_kombi": "eta_c",
    "combined_score": "J",
}

# ----------------------------------------------------------------------
# Positions-Sweep: dieselben Groessen, einmal je Richtung
# ----------------------------------------------------------------------
# Symbol und Farbe bleiben, die Richtung kommt als Exponent dazu
# (U_w^v / U_w^d) - dieselbe Schreibweise wie bei eta_h^circ/eta_h^box.
# Die Richtung wird ZUSAETZLICH ueber den Linienstil kodiert (senkrecht
# durchgezogen, diagonal gestrichelt): die beiden Kurven einer Groesse
# laufen dicht beieinander, und Farbe allein traegt das nicht.
RICHTUNG_EXPONENT = {"vertikal": r"\mathrm{v}", "diagonal": r"\mathrm{d}"}
RICHTUNG_STIL = {"vertikal": "-", "diagonal": "--"}
RICHTUNG_TEXT = {"vertikal": "senkrecht", "diagonal": "diagonal"}


def _mit_exponent(label, exponent):
    """$U_w$ -> $U_w^{\mathrm{v}}$.

    Traegt das Symbol schon einen Exponenten (eta_h^{circ}), wird der
    Zusatz hineingeschrieben statt ein zweiter Exponent angehaengt - zwei
    hochgestellte Bloecke hintereinander sind kein gueltiges LaTeX."""
    kern = label.strip("$")
    if "^{" in kern and kern.endswith("}"):
        kern = kern[:-1] + "," + exponent + "}"
    else:
        kern = kern + "^{" + exponent + "}"
    return "$" + kern + "$"


for _basis in list(TRACES):
    for _richtung, _exp in RICHTUNG_EXPONENT.items():
        _label, _einheit, _farbe, _prozent = TRACES[_basis]
        TRACES[f"{_basis}__{_richtung}"] = (
            _mit_exponent(_label, _exp), _einheit, _farbe, _prozent)
        TEXT_LABELS[f"{_basis}__{_richtung}"] = (
            f"{TEXT_LABELS[_basis]} ({_richtung[0]})")
TRACE_ORDER = TRACE_ORDER + [f"{b}__{r}" for b in list(TRACE_ORDER)
                             for r in RICHTUNG_EXPONENT]

WEIGHTED_KEYS = ("uniformity_weighted", "crosstalk_weighted")
PENALTY_KEYS = ("uniformity_kombi", "crosstalk_kombi", "combined_score")


def hard_keys(results):
    """Die harten Kurven dieses Datensatzes. Sind beide Regionen gerechnet,
    sind es drei: die eine Uniformity und die zwei Crosstalks."""
    if "crosstalk_hart_kreis" in results:
        return ("uniformity_hart", "crosstalk_hart_kreis", "crosstalk_hart_pitch")
    return ("uniformity_hart", "crosstalk_hart")


def kurven_keys(results):
    """Alle Kurven fuer Tabellen und den Arbeitspunkt - ohne Dubletten.

    Im Modus "beide" ist `crosstalk_hart` eine der beiden Regionen-Kurven
    (die, die in die Penalty eingeht); sie noch einmal aufzufuehren waere
    dieselbe Zahl unter zwei Namen."""
    keys = [k for k in TRACE_ORDER if k in results]
    if "crosstalk_hart_kreis" in keys and "crosstalk_hart" in keys:
        keys.remove("crosstalk_hart")
    return keys


def zeilen_namen(results):
    """Klartext-Bezeichnung je Kurve fuer den Bericht."""
    p = results["params"]
    region = "Kreis" if sb.penalty_region(p) == "kreis" else "Pitch-Quadrat"
    return {
        "uniformity_hart": "U_h (hart, Kreis)",
        "crosstalk_hart": f"eta_h (hart, {region})",
        "crosstalk_hart_kreis": "eta_h^circ (hart, Kreis)",
        "crosstalk_hart_pitch": "eta_h^box (hart, Pitch-Quadrat)",
        "uniformity_weighted": "U_w (atom-gewichtet)",
        "crosstalk_weighted": "eta_w (atom-gewichtet)",
        "uniformity_kombi": "U_c (Penalty)",
        "crosstalk_kombi": "eta_c (Penalty)",
        "combined_score": "J (Score)",
    }

# Eine Kurvenfigur, Breite = Textbreite. Die Hoehe ist so gewaehlt, dass
# Titel, zweite x-Achse oben und die Legende unter dem Panel Platz haben,
# ohne dass das Panel selbst flach wird.
CURVE_FIGSIZE = (6.3, 4.4)

# Schrift- und Linienstaerke, gemessen an DOC_RC aus report.py (dessen
# Grundschrift ist auf eine ueber die ganze Textbreite eingebundene Karte
# ausgelegt). Eine Kurvenfigur wird im Text oft kleiner gesetzt und lebt
# von wenigen Beschriftungen - hier ist mehr Gewicht besser lesbar. Der
# Faktor geht als `dichte` in dokument_stil() und skaliert Schrift,
# Linien, Marker und Teilstriche gemeinsam.
SCHRIFT_DICHTE = 1.45

# Luft ueber den Kurven, damit die Legende oben links nicht auf ihnen
# liegt - als Anteil der Datenspanne der jeweiligen Achse.
LEGENDEN_LUFT = 0.28

# Wo die Legende steht. "upper left" ist die Vorgabe; jeder
# matplotlib-loc-String geht.
LEGENDEN_ORT = "upper left"

WAIST_LABEL = r"Waist in the atomic plane $w$ (µm)"
WIN_INPUT_LABEL = r"Waist before first lens $w_\mathrm{in}$ (mm)"

# Mindestabstand zweier Teilstriche der oberen Achse, als Anteil der
# Achsenbreite (siehe _zweite_x_achse).
TICK_MIN_ABSTAND = 0.07

# Diese beiden gehoeren immer auf DIESELBE y-Achse - es ist dieselbe
# Groesse ueber zwei Regionen, und genau ihr Verhaeltnis will man ablesen.
# Auf getrennten Achsen saehen sie gleich gross aus. Dasselbe Prinzip wie
# AXIS_GROUP_ALWAYS_TOGETHER (r_x/r_y) in report.py.
IMMER_ZUSAMMEN = [("crosstalk_hart_kreis", "crosstalk_hart_pitch")] + [
    tuple(f"{basis}__{richtung}" for richtung in RICHTUNG_EXPONENT)
    for basis in ("uniformity_hart", "crosstalk_hart", "crosstalk_hart_kreis",
                  "crosstalk_hart_pitch", "uniformity_weighted",
                  "crosstalk_weighted", "uniformity_kombi", "crosstalk_kombi",
                  "combined_score")
]

# Linienstile: alles durchgezogen - ausser dort, wo zwei Kurven im Bild
# uebereinander liegen. Dann taeuscht eine durchgezogene Linie eine
# Einzelkurve vor, und die untere ist schlicht nicht mehr da; deshalb wird
# die spaeter gezeichnete der beiden gestrichelt.
#
# Entschieden wird das am fertigen Bild, nicht an den Rohwerten: zwei
# Kurven koennen auf verschiedenen y-Achsen laufen und trotzdem exakt
# uebereinander liegen (U_c und J tun das regelmaessig). Verglichen werden
# deshalb die Lagen in Achsen-Anteilen, nachdem alle Achsengrenzen stehen.
STRICHEL_ZYKLUS = ("--", "-.", (0, (5, 1, 1, 1, 1, 1)))

# Zwei Kurven gelten als uebereinanderliegend, wenn sie sich ueber
# UEBERLAPP_ANTEIL des Bereichs um weniger als UEBERLAPP_TOLERANZ der
# Achsenhoehe unterscheiden.
UEBERLAPP_TOLERANZ = 0.02
UEBERLAPP_ANTEIL = 0.5


def _normierte_lage(ax, werte):
    """Die Kurve in Achsen-Anteilen (0 = unten, 1 = oben), so wie sie im
    Bild liegt. None, wenn sich das nicht bestimmen laesst."""
    lo, hi = ax.get_ylim()
    v = np.asarray(werte, dtype=float)
    if ax.get_yscale() == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log10(np.where(v > 0, v, np.nan))
        if lo <= 0 or hi <= 0:
            return None
        lo, hi = np.log10(lo), np.log10(hi)
    if not np.isfinite(hi - lo) or hi <= lo:
        return None
    return (v - lo) / (hi - lo)


def _liegen_uebereinander(a, b):
    if a is None or b is None:
        return False
    gueltig = np.isfinite(a) & np.isfinite(b)
    if not np.any(gueltig):
        return False
    nah = np.abs(a[gueltig] - b[gueltig]) < UEBERLAPP_TOLERANZ
    return float(np.mean(nah)) >= UEBERLAPP_ANTEIL


def _entwirre_ueberlapp(gezeichnet):
    """gezeichnet: Liste (linie, ax, werte) in Zeichenreihenfolge.

    Jede Kurve, die auf einer frueheren liegt, bekommt den naechsten
    Strichel-Stil. Die erste bleibt durchgezogen - es soll ja gerade
    sichtbar werden, dass da zwei sind."""
    lagen = [_normierte_lage(ax, werte) for _linie, ax, werte in gezeichnet]
    naechster = 0
    for i in range(1, len(gezeichnet)):
        if any(_liegen_uebereinander(lagen[i], lagen[j]) for j in range(i)):
            gezeichnet[i][0].set_linestyle(STRICHEL_ZYKLUS[naechster % len(STRICHEL_ZYKLUS)])
            naechster += 1

# Achsen-Modi (Dialog)
ACHSEN_CHOICES = [
    ("auto", "automatisch: gemeinsame Achse bei gleicher Groessenordnung"),
    ("eine", "immer EINE gemeinsame y-Achse (linear)"),
    ("log", "eine gemeinsame y-Achse, logarithmisch"),
    ("getrennt", "immer eine eigene y-Achse je Kurve"),
]


def hard_title(results):
    """Ueberschrift der harten Figur. Nimmt der Crosstalk das Pitch-Quadrat,
    stehen BEIDE Regionen im Titel - sonst liest man zwei Kurven als eine
    Aussage ueber dieselbe Flaeche, und das waeren sie dann nicht."""
    p = results["params"]
    radius_um = p["hard_radius"] * 1e6
    region = p.get("hard_crosstalk_region", "kreis")
    if region == "kreis":
        return r"Single beam, hard region (circle $R = %.2f$ µm)" % radius_um
    if region == "beide":
        return (r"Single beam, hard region ($U_h$ and $\eta_h^{\mathrm{circ}}$: "
                r"circle $R = %.2f$ µm, $\eta_h^{\mathrm{box}}$: pitch box "
                r"$p = %.3f$ µm)" % (radius_um, p["pitch"] * 1e6))
    return (r"Single beam, hard region ($U_h$: circle $R = %.2f$ µm, "
            r"$\eta_h$: pitch box $p = %.3f$ µm)"
            % (radius_um, p["pitch"] * 1e6))


def output_prefix(results, tag=None):
    """SingleBeam_{Profil}_r{Radius}um{_pitchbox}{_kFaktor}_{Datum} -
    dasselbe Muster wie die Praefixe der Multitone-Auswertung (Art,
    Konfiguration, Datum)."""
    params = results["params"]
    profil = paths.profile_tag_of(params["profile"])
    radius_um = float(params["hard_radius"]) * 1e6
    tag = date.today().isoformat() if tag is None else str(tag)
    return (f"SingleBeam_{profil}_r{radius_um:g}um"
            f"{sb.crosstalk_region_tag(params)}{sb.airy_tag(params)}_{tag}")


# ======================================================================
# Werte und Achsen-Buendelung
# ======================================================================
def werte_in_prozent(results, keys):
    """Die Kurven in der Einheit, in der sie gezeichnet werden."""
    werte = {}
    for key in keys:
        _label, _einheit, _farbe, prozent = TRACES[key]
        v = np.asarray(results[key], dtype=float)
        werte[key] = v * 100.0 if prozent else v
    return werte


# Wieviel der gemeinsamen Achsenspanne eine Kurve mit ihrer eigenen Spanne
# mindestens fuellen muss, damit sie sich die Achse teilen darf. Strenger
# als AXIS_GROUP_MIN_SHARE (0.10) in report.py: dort sind die Kurven flache
# Querschnitte, hier laufen sie ueber den ganzen Bereich, und mit 0.10
# waeren auch Kurven zusammengekommen, von denen eine nur noch das untere
# Achsenzehntel gefuellt haette.
#
# Angefangen wird mit der Kurve, die den GROESSTEN Bereich abdeckt, und
# dann werden die kleineren dazugenommen, solange sie sichtbar bleiben.
# Nach dem kleinsten Wert zu sortieren waere naheliegender, gibt aber die
# falsche Antwort: dann setzt sich eine kleine Kurve an den Anfang und
# nimmt die naechstgroessere mit, waehrend die beiden, die eigentlich
# aufeinander liegen (U_c und J), auf zwei Achsen landen.
SICHTBAR_MIN_ANTEIL = 0.25


def _passt_zusammen(bereiche, kandidaten, werte, min_share=SICHTBAR_MIN_ANTEIL):
    """Duerfen sich diese Kurven eine y-Achse teilen?

    Gefragt wird nur: bleibt auf der gemeinsamen Achse jede von ihnen noch
    sichtbar? Mass dafuer ist, welchen Anteil der gemeinsamen Achsenspanne
    eine Kurve mit ihrer eigenen Spanne fuellt (dieselbe Bedingung wie
    AXIS_GROUP_MIN_SHARE in report.py, samt der dortigen Ausnahme fuer
    verrauschte Kurven).

    Anders als report._passt_dazu() wird NICHT zusaetzlich verlangt, dass
    der gemeinsame Wertebereich hoechstens eine Groessenordnung umspannt.
    Diese Bedingung ist fuer Karten-Querschnitte gedacht, wo die Kurven
    flach sind; hier fallen U_c und J ueber den Waist selbst um mehr als
    das Zehnfache, und die Bedingung haette sie auf getrennte Achsen
    geschoben, obwohl sie fast aufeinander liegen - genau die Kurven also,
    die zusammengehoeren.
    """
    lo = min(bereiche[k][0] for k in kandidaten)
    hi = max(bereiche[k][1] for k in kandidaten)
    spanne = hi - lo
    if spanne <= 0:
        return True
    for k in kandidaten:
        anteil = (bereiche[k][1] - bereiche[k][0]) / spanne
        if anteil < min_share and not _ist_verrauscht(werte[k]):
            return False
    return True


def gruppen_fuer_achsen(keys, werte, modus="auto", max_ratio=AXIS_GROUP_MAX_RATIO):
    """Aufteilung der Kurven auf y-Achsen.

    'auto' buendelt nach Sichtbarkeit (siehe _passt_zusammen), 'eine' und
    'log' erzwingen eine gemeinsame Achse, 'getrennt' je eine eigene.
    """
    keys = [k for k in TRACE_ORDER if k in keys]
    if modus in ("eine", "log"):
        return [list(keys)]
    if modus == "getrennt":
        return [[k] for k in keys]

    # Feste Gruppen zuerst: dieselbe Groesse in zwei Regionen bzw. zwei
    # Richtungen gehoert auf eine Achse, sonst saehe sie gleich gross aus.
    # Sie gehen als EINHEIT in die weitere Buendelung ein - eine feste
    # Gruppe darf sich also mit anderen Kurven eine Achse teilen, solange
    # dort jede sichtbar bleibt (U_c und J landen so zusammen).
    fest = []
    vergeben = set()
    for gruppe in IMMER_ZUSAMMEN:
        dabei = [k for k in keys if k in gruppe and k not in vergeben]
        if len(dabei) > 1:
            fest.append(dabei)
            vergeben.update(dabei)

    bereiche = {k: _wertebereich(werte[k]) for k in keys}
    einheiten = fest + [[k] for k in keys if k not in vergeben]
    ohne = [e for e in einheiten if all(bereiche[k] is None for k in e)]
    mit = [e for e in einheiten if any(bereiche[k] is not None for k in e)]
    def _spanne(einheit):
        gueltig = [bereiche[k] for k in einheit if bereiche[k] is not None]
        return max(b[1] for b in gueltig) - min(b[0] for b in gueltig)

    mit.sort(key=_spanne, reverse=True)

    gruppen = []
    aktuell = []
    for einheit in mit:
        kandidaten = [k for k in aktuell + einheit if bereiche[k] is not None]
        if aktuell and _passt_zusammen(bereiche, kandidaten, werte):
            aktuell = aktuell + einheit
        else:
            if aktuell:
                gruppen.append(aktuell)
            aktuell = list(einheit)
    if aktuell:
        gruppen.append(aktuell)
    gruppen.extend(ohne)

    rang = {k: i for i, k in enumerate(TRACE_ORDER)}
    for g in gruppen:
        g.sort(key=lambda k: rang.get(k, 99))
    gruppen.sort(key=lambda g: rang.get(g[0], 99))
    return gruppen


def _achsen_label(gruppe):
    """Beschriftung der y-Achse.

    Im Positions-Sweep steht dieselbe Groesse zweimal auf der Achse, einmal
    je Richtung. Die Achse traegt sie deshalb nur EINMAL - welche Kurve
    welche Richtung ist, sagt die Legende. Sonst stuende dort viermal
    dasselbe Symbol mit wechselndem Exponenten."""
    labels = []
    for key in gruppe:
        basis = key.rsplit("__", 1)[0] if key.rsplit("__", 1)[-1] in RICHTUNG_STIL else key
        label = TRACES[basis][0]
        if label not in labels:
            labels.append(label)
    einheit = TRACES[gruppe[0]][1]
    text = ", ".join(labels)
    return f"{text} ({einheit})" if einheit else text


# Leiter, aus der die Schrittweite der oberen Achse gewaehlt wird, und wie
# viele Teilstriche sie hoechstens ergeben darf. Genommen wird die feinste
# Schrittweite, die darunter bleibt.
TICK_SCHRITTE_MM = (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0)
TICK_MAX_ANZAHL = 10


def _zweite_x_achse(ax, results):
    """Oben der Waist vor der ersten Linse. waist = C/win_input, die
    Umrechnung ist ihre eigene Umkehrung bis auf die Konstante - beide
    Richtungen also dieselbe Funktion.

    Die Teilstriche werden SELBST gesetzt, und zwar in zwei Schritten:

    1. Der ERSTE Teilstrich ist der Startwert der Auftragung selbst - der
       Wert, den man im Dialog eingegeben hat (z.B. 1.8 mm). Er soll
       dastehen; ein Ticker, der stattdessen bei 2.0 anfaengt, beschriftet
       eine Stelle, die gar nicht gerechnet wurde. Von dort geht es in
       runden Schritten abwaerts (TICK_SCHRITTE_MM).
    2. Danach wird nach dem Umrechnen ausgeduennt: bei einer
       1/x-Beziehung draengeln sich die grossen win_input-Werte am linken
       Rand zu einem Klumpen. Liegen zwei Teilstriche im Bild naeher
       als TICK_MIN_ABSTAND der Achsenbreite beieinander, faellt der
       kleinere weg - bei 1.8 mm Start faellt so die 1.6 heraus und es
       bleiben 1.8, 1.4, 1.2, 1.0, 0.8, 0.6.
    """
    p = results["params"]
    const_mm_um = 1e3 * 1e6 * (p["f1"] / p["f2"]) * p["lambda_opt"] * p["fLO"] / np.pi

    def kehrwert(v):
        v = np.asarray(v, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(v > 0, const_mm_um / np.where(v > 0, v, np.nan), np.nan)

    sek = ax.secondary_xaxis("top", functions=(kehrwert, kehrwert))
    sek.set_xlabel(WIN_INPUT_LABEL)

    x_lo, x_hi = ax.get_xlim()
    x_lo = max(x_lo, float(np.min(results["waist_um"])) * 0.5)
    if x_lo <= 0 or x_hi <= x_lo:
        return sek
    win_lo, win_hi = sorted(float(v) for v in kehrwert(np.array([x_lo, x_hi])))

    # Der Startwert der Auftragung: der groesste tatsaechlich gerechnete
    # win_input, also der linke Rand der Kurven.
    start = float(np.nanmax(np.asarray(results["win_input_mm"], dtype=float)))

    schritt = TICK_SCHRITTE_MM[-1]
    for kandidat in TICK_SCHRITTE_MM:
        if (start - win_lo) / kandidat <= TICK_MAX_ANZAHL:
            schritt = kandidat
            break

    # Runde Werte unterhalb des Startwerts, dazu der Startwert selbst.
    leiter = np.arange(np.ceil(win_lo / schritt) * schritt,
                       start - 1e-9, schritt)
    kandidaten = [start] + sorted((float(t) for t in leiter), reverse=True)

    # Aufsteigend in x heisst absteigend in win_input.
    behalten = []
    for t in kandidaten:
        if not (win_lo - 1e-9 <= t <= win_hi + 1e-9):
            continue
        x = float(kehrwert(np.array([t]))[0])
        if all(abs(x - float(kehrwert(np.array([u]))[0])) > TICK_MIN_ABSTAND * (x_hi - x_lo)
               for u in behalten):
            behalten.append(t)
    if len(behalten) >= 2:
        sek.set_xticks(sorted(behalten))
    return sek


def _ticks_entzerren(fig, sek, luft_pt=6.0):
    """Teilstriche der oberen Achse ausduennen, deren BESCHRIFTUNGEN sich
    ueberlappen.

    Der Abstand in TICK_MIN_ABSTAND ist ein Anteil der Achsenbreite und
    weiss nichts davon, wie breit "1.10" gegen "1.8" gesetzt wird. Hier
    wird deshalb einmal gezeichnet und dann wirklich gemessen: von links
    nach rechts bleibt jeder Teilstrich, dessen Beschriftung die letzte
    behaltene nicht beruehrt. Von links, damit der Startwert der
    Auftragung in jedem Fall stehen bleibt - er ist der eine Wert, der
    dastehen soll.
    """
    ticks = list(sek.get_xticks())
    if len(ticks) < 2:
        return
    fig.canvas.draw()
    labels = [t for t in sek.get_xticklabels() if t.get_text()]
    if len(labels) != len(ticks):
        return
    reihenfolge = sorted(range(len(ticks)),
                         key=lambda i: labels[i].get_window_extent().x0)
    behalten = []
    letzte = None
    for i in reihenfolge:
        kasten = labels[i].get_window_extent()
        if letzte is not None and kasten.x0 < letzte.x1 + luft_pt:
            continue
        behalten.append(ticks[i])
        letzte = kasten
    if 2 <= len(behalten) < len(ticks):
        sek.set_xticks(sorted(behalten))


# ======================================================================
# Die Kurvenfigur
# ======================================================================
def arbeitspunkt_werte(results, waist_um, keys=None):
    """Die Kurvenwerte (in Prozent) am markierten Waist, linear zwischen
    den beiden benachbarten Stuetzstellen interpoliert.

    Bewusst interpoliert und nicht auf die naechste Stuetzstelle gerundet:
    die Kurven sind hier glatte Funktionen des Waists (anders als r_x/r_y
    im Multitone-Scan, die Optimierungs-ERGEBNISSE sind - dort waere
    Interpolieren irrefuehrend)."""
    keys = kurven_keys(results) if keys is None else [k for k in TRACE_ORDER if k in keys]
    x = np.asarray(results["waist_um"], dtype=float)
    werte = werte_in_prozent(results, keys)
    return {k: float(np.interp(waist_um, x, werte[k])) for k in keys}


def _zeichne_arbeitspunkt(ax0, achsen_und_gruppen, x, werte, waist_um):
    """Senkrechte Linie plus je einen Stern auf jeder Kurve. Gibt das
    Handle fuer die Legende zurueck."""
    # Die senkrechte Linie ist bewusst gestrichelt: sie ist keine Messkurve,
    # sondern eine Markierung, und soll auch dort als solche erkennbar sein,
    # wo sie eine Kurve kreuzt.
    linie = ax0.axvline(waist_um, color=BEST_POINT_STYLE["color"],
                        linestyle="--", linewidth=S(1.1), alpha=0.8, zorder=1.5,
                        label=r"$w$ = %.3f µm" % waist_um)
    for ax, gruppe in achsen_und_gruppen:
        for key in gruppe:
            y = float(np.interp(waist_um, x, werte[key]))
            ax.plot([waist_um], [y], zorder=6,
                    **SD(BEST_POINT_STYLE, markersize=S(11), markeredgewidth=S(1.0)))
    return linie


def _luft_nach_oben(ax, werte_liste, anteil=LEGENDEN_LUFT):
    """Achsenobergrenze anheben, damit die Legende oben links nicht auf
    den Kurven liegt. Es wird nichts abgeschnitten, nur Platz gemacht."""
    if ax.get_yscale() == "log":
        return
    endlich = [v[np.isfinite(v)] for v in (np.asarray(w, dtype=float) for w in werte_liste)]
    endlich = [v for v in endlich if v.size]
    if not endlich:
        return
    lo_daten = min(float(v.min()) for v in endlich)
    hi_daten = max(float(v.max()) for v in endlich)
    lo, hi = ax.get_ylim()
    spanne = max(hi - lo, hi_daten - lo_daten)
    if spanne <= 0:
        return
    ax.set_ylim(lo, max(hi, hi_daten + anteil * spanne))


def plot_curves(results, keys, filename, titel=None, out_dir=None, achsen="auto",
                save=True, show=False, confirm_overwrite=None, marker=False,
                arbeitspunkt=None, legende=LEGENDEN_ORT, dichte=SCHRIFT_DICHTE,
                x_key="waist_um", x_label=None, zweite_achse=None, stile=None):
    """Eine Figur mit den angegebenen Kurven ueber dem Waist.

    keys: Schluessel aus TRACES. Uniformity und Crosstalk gehoeren immer
    zusammen in einen Aufruf - getrennt zu zeichnen waere die eine
    Darstellung, die beim Vergleich nichts taugt.

    titel: None (Vorgabe) laesst die Ueberschrift weg. Was sie sagen wuerde
    (Region, sigma_atom, Penalty-Parameter), steht im Bericht und im
    Dateinamen - in einer Abbildung mit eigener Bildunterschrift ist sie
    doppelt.

    arbeitspunkt: Waist in µm, der als senkrechte Linie und als Stern auf
    jeder Kurve markiert wird. None = keiner.

    x_key/x_label/zweite_achse: die x-Achse. Vorgabe ist der Waist mit dem
    Eingangswaist obendrueber; der Positions-Sweep setzt hier den Versatz
    des Atoms ein.

    stile: Linienstil je Kurve. Ohne Angabe wird alles durchgezogen
    gezeichnet und nur bei Ueberlapp gestrichelt (_entwirre_ueberlapp).
    Der Positions-Sweep gibt Stile vor - dort steht der Stil fuer die
    RICHTUNG und darf nicht automatisch vergeben werden.
    """
    out_dir = paths.fit_plots_dir() if out_dir is None else out_dir
    x_label = WAIST_LABEL if x_label is None else x_label
    zweite_achse = _zweite_x_achse if zweite_achse is None else zweite_achse
    keys = [k for k in TRACE_ORDER if k in keys]
    werte = werte_in_prozent(results, keys)
    gruppen = gruppen_fuer_achsen(keys, werte, modus=achsen)
    x = np.asarray(results[x_key], dtype=float)

    with dokument_stil(CURVE_FIGSIZE[0], dichte=dichte):
        fig, ax0 = plt.subplots(figsize=CURVE_FIGSIZE, constrained_layout=True)

        linien = []
        gezeichnet = []
        achsen_und_gruppen = []
        for position, gruppe in enumerate(gruppen):
            ax = ax0 if position == 0 else ax0.twinx()
            achsen_und_gruppen.append((ax, gruppe))
            if position > 1:
                ax.spines["right"].set_position(
                    ("outward", S(ZUSATZ_ACHSE_PT) * (position - 1)))
                ax.set_frame_on(True)
                ax.patch.set_visible(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.spines["right"].set_visible(True)

            for key in gruppe:
                label, _einheit, farbe, _prozent = TRACES[key]
                linie, = ax.plot(
                    x, werte[key], color=farbe,
                    linestyle="-" if stile is None else stile.get(key, "-"),
                    linewidth=S(1.6), label=label,
                    marker="o" if marker else None, markersize=S(2.6))
                linien.append(linie)
                gezeichnet.append([linie, ax, werte[key]])

            if achsen == "log":
                ax.set_yscale("log")
            else:
                _entzerre_achse(ax, [werte[k] for k in gruppe])
            if legende and legende.startswith("upper"):
                _luft_nach_oben(ax, [werte[k] for k in gruppe])

            # Achsen bleiben schwarz. Welche Kurve welche Farbe hat, steht
            # in der Legende; eingefaerbte Achsen wiederholen das nur und
            # machen die Zahlen schlechter lesbar (helle Kurvenfarben
            # verblassen auf weissem Grund).
            ax.set_ylabel(_achsen_label(gruppe))

        # Erst jetzt stehen alle Achsengrenzen - vorher waere die Lage im
        # Bild noch nicht bekannt. Mit vorgegebenen Stilen entfaellt das:
        # dort traegt der Stil bereits eine Bedeutung.
        if stile is None:
            _entwirre_ueberlapp(gezeichnet)

        if arbeitspunkt is not None:
            linien.append(_zeichne_arbeitspunkt(
                ax0, achsen_und_gruppen, x, werte, float(arbeitspunkt)))

        ax0.set_xlabel(x_label)
        if titel:
            ax0.set_title(titel)
        ax0.grid(True, alpha=0.25)
        sek = zweite_achse(ax0, results) if zweite_achse is not None else None

        # Die Legende steht IM Bild (Vorgabe oben links). Damit sie nicht in
        # die Layout-Rechnung eingeht und das Panel schrumpft, wird sie aus
        # dem Layout genommen - dieselbe Stelle, an der report.py das fuer
        # die Kartenlegende des Talschnitts tut. Der Platz kommt statt
        # dessen aus _luft_nach_oben().
        leg = ax0.legend(linien, [ln.get_label() for ln in linien],
                         loc=legende, framealpha=0.95,
                         ncol=1 if len(linien) <= 3 else 2)
        leg.set_in_layout(False)

        # Zum Schluss, wenn das Layout steht: Teilstriche der oberen Achse
        # ausduennen, deren Beschriftungen sich beruehren.
        if sek is not None:
            _ticks_entzerren(fig, sek)

        return _finish(fig, out_dir, filename, save, show, confirm_overwrite)


def make_plots(results, out_dir=None, getrennt=True, achsen="auto",
               penalty_plot=True, save=True, show=False, confirm_overwrite=None,
               marker=False, prefix=None, titel=False, arbeitspunkt=None,
               legende=LEGENDEN_ORT, dichte=SCHRIFT_DICHTE):
    """Alle Figuren eines Laufs.

    getrennt=True : harte und gewichtete Metriken in je einer eigenen Figur
    getrennt=False: alle vier in derselben Figur (hart durchgezogen,
                    gewichtet gestrichelt)
    penalty_plot  : zusaetzlich U_c, eta_c und J
    titel         : Ueberschriften zeichnen (Vorgabe: nein - in einer
                    Abbildung mit Bildunterschrift sind sie doppelt)
    arbeitspunkt  : Waist in µm zum Einzeichnen; None nimmt
                    results["arbeitspunkt_um"], falls gesetzt

    Uniformity und Crosstalk bleiben in jedem Fall zusammen.
    """
    prefix = output_prefix(results) if prefix is None else prefix
    if arbeitspunkt is None:
        arbeitspunkt = results.get("arbeitspunkt_um")
    gemeinsam = dict(out_dir=out_dir, achsen=achsen, save=save, show=show,
                     confirm_overwrite=confirm_overwrite, marker=marker,
                     arbeitspunkt=arbeitspunkt, legende=legende, dichte=dichte)
    p = results["params"]
    pfade = []

    if getrennt:
        pfade.append(plot_curves(
            results, hard_keys(results), f"{prefix}_hard.pdf",
            hard_title(results) if titel else None, **gemeinsam))
        pfade.append(plot_curves(
            results, WEIGHTED_KEYS, f"{prefix}_weighted.pdf",
            (r"Single beam, atom-weighted ($\sigma_\mathrm{atom} = %.0f$ nm)"
             % (results["sigma_atom"] * 1e9)) if titel else None, **gemeinsam))
    else:
        pfade.append(plot_curves(
            results, tuple(hard_keys(results)) + tuple(WEIGHTED_KEYS), f"{prefix}_metrics.pdf",
            "Single beam: hard region and atom-weighted" if titel else None,
            **gemeinsam))

    if penalty_plot:
        pfade.append(plot_curves(
            results, PENALTY_KEYS, f"{prefix}_penalty.pdf",
            (r"Penalty combination ($\alpha = %.2f$, $\lambda = %.2f$)"
             % (p["alpha"], p["combo_lambda"])) if titel else None, **gemeinsam))

    return [q for q in pfade if q is not None]


# ======================================================================
# Positions-Sweep: Kurven ueber dem Atom-Versatz
# ======================================================================
OFFSET_LABEL = r"Atom offset from site centre $r$ (µm)"
OFFSET_LABEL_REL = r"$r / w$"


def _zweite_x_achse_offset(ax, results):
    """Oben derselbe Versatz in Einheiten des Waists - r/w = 1 ist der
    Rand des Strahls und zugleich das Ende des abgefahrenen Bereichs."""
    waist_um = float(results["waist_um"])
    if not np.isfinite(waist_um) or waist_um <= 0:
        return None
    sek = ax.secondary_xaxis(
        "top", functions=(lambda v: np.asarray(v, dtype=float) / waist_um,
                          lambda v: np.asarray(v, dtype=float) * waist_um))
    sek.set_xlabel(OFFSET_LABEL_REL)
    return sek


def offset_stile(keys):
    """Linienstil je Kurve: die RICHTUNG steckt im Stil, die Groesse in der
    Farbe. Zwei Kurven derselben Groesse laufen dicht beieinander - Farbe
    allein traegt das nicht."""
    stile = {}
    for key in keys:
        richtung = key.rsplit("__", 1)[-1]
        stile[key] = RICHTUNG_STIL.get(richtung, "-")
    return stile


def offset_keys(results, basis_keys):
    """Aus Basis-Groessen die tatsaechlich vorhandenen Richtungs-Kurven."""
    return tuple(f"{basis}__{richtung}"
                 for basis in basis_keys
                 for richtung in results.get("richtungen", ())
                 if f"{basis}__{richtung}" in results)


def offset_prefix(results, tag=None):
    """SingleBeamOffset_{Profil}_w{Waist}um_r{Radius}um..._{Datum}"""
    p = results["params"]
    profil = paths.profile_tag_of(p["profile"])
    radius_um = float(p["hard_radius"]) * 1e6
    tag = date.today().isoformat() if tag is None else str(tag)
    return (f"SingleBeamOffset_{profil}_w{results['waist_um']:g}um"
            f"_r{radius_um:g}um{sb.crosstalk_region_tag(p)}{sb.airy_tag(p)}_{tag}")


def make_offset_plots(results, out_dir=None, getrennt=True, achsen="auto",
                      penalty_plot=True, save=True, show=False,
                      confirm_overwrite=None, marker=False, prefix=None,
                      titel=False, legende=LEGENDEN_ORT, dichte=SCHRIFT_DICHTE):
    """Dieselben drei Figuren wie beim Waist-Sweep, nur ueber dem Versatz
    des Atoms - und mit beiden Richtungen nebeneinander."""
    prefix = offset_prefix(results) if prefix is None else prefix
    hart_basis = (("uniformity_hart", "crosstalk_hart_kreis", "crosstalk_hart_pitch")
                  if f"crosstalk_hart_kreis__{results['richtungen'][0]}" in results
                  else ("uniformity_hart", "crosstalk_hart"))

    def figur(basis_keys, dateiname, titeltext):
        keys = offset_keys(results, basis_keys)
        if not keys:
            return None
        return plot_curves(
            results, keys, dateiname, titeltext if titel else None,
            out_dir=out_dir, achsen=achsen, save=save, show=show,
            confirm_overwrite=confirm_overwrite, marker=marker,
            legende=legende, dichte=dichte, stile=offset_stile(keys),
            x_key="offset_um", x_label=OFFSET_LABEL,
            zweite_achse=_zweite_x_achse_offset)

    pfade = []
    if getrennt:
        pfade.append(figur(hart_basis, f"{prefix}_hard.pdf",
                           "Atom position sweep, hard region"))
        pfade.append(figur(WEIGHTED_KEYS, f"{prefix}_weighted.pdf",
                           "Atom position sweep, atom-weighted"))
    else:
        pfade.append(figur(tuple(hart_basis) + tuple(WEIGHTED_KEYS),
                           f"{prefix}_metrics.pdf",
                           "Atom position sweep"))
    if penalty_plot:
        pfade.append(figur(PENALTY_KEYS, f"{prefix}_penalty.pdf",
                           "Atom position sweep, penalty combination"))
    return [q for q in pfade if q is not None]


def _offset_zeile(results, key, name):
    werte = np.asarray(results[key], dtype=float) * 100.0
    endlich = werte[np.isfinite(werte)]
    best = results["best"].get(key)
    if endlich.size == 0 or best is None:
        return f"| {name} | n/a | n/a | n/a | n/a |"
    rand = " (Rand)" if best["at_edge"] else ""
    return (f"| {name} | {werte[0]:.4g} % | {werte[-1]:.4g} % "
            f"| {sb.monotone_kind(results[key])} "
            f"| {best['value'] * 100:.4g} % bei r = {best['offset_um']:.4f} µm{rand} |")


def write_offset_report(results, out_dir=None, prefix=None, dateiname=None):
    """Markdown-Bericht des Positions-Sweeps."""
    out_dir = paths.FIT_RESULTS_DIR if out_dir is None else out_dir
    prefix = offset_prefix(results) if prefix is None else prefix
    dateiname = f"{prefix}_Report.md" if dateiname is None else dateiname

    p = results["params"]
    r_um = np.asarray(results["offset_um"], dtype=float)
    folgt = results.get("hard_follows_atom", True)

    hart_satz = (
        "Die harte Region WANDERT MIT dem Atom: der Kreis (und, wenn "
        "gewaehlt, das Pitch-Quadrat) sitzt an der jeweiligen Atomposition. "
        "Das ist hier die sinnvolle Lesart - die Atomposition ist die "
        "abgefahrene Groesse, und die harte Region fragt, wie es DORT "
        "aussieht."
        if folgt else
        "Die harte Region BLEIBT AUF DER SITE. Die harten Kurven sind "
        "deshalb ueber dem Versatz konstant; sie stehen nur zum Vergleich "
        "daneben.")

    zeilen = []
    namen = zeilen_namen(results)
    for key in results["kurven"]:
        basis, richtung = key.rsplit("__", 1)
        zeilen.append(_offset_zeile(
            results, key, f"{namen[basis]} - {RICHTUNG_TEXT[richtung]}"))

    tabelle = "\n".join(
        ["| Groesse | bei r = 0 | am Ende | Verlauf | Minimum |",
         "|---|---|---|---|---|"] + zeilen)

    text = f"""# SingleBeamOffset - Atomposition bei festem Waist, {date.today().isoformat()}

Fester Waist, das ATOM wandert. Gefahren wird der Betrag r des Versatzes
gegen die Site-Mitte, einmal je Richtung.

- Waist: **{results['waist_um']:.4f} µm** (win_input = {results['win_input_mm']:.4f} mm)
- Versatz: 0 .. {r_um.max():.4f} µm  (= {r_um.max() / results['waist_um']:.3f} x Waist)
- Stuetzstellen: {r_um.size} je Richtung
- Richtungen: {", ".join(RICHTUNG_TEXT[r] for r in results['richtungen'])}

Eine waagerechte Richtung fehlt mit Absicht: das Strahlprofil ist
rotationssymmetrisch, waagerecht ist dasselbe wie senkrecht. Diagonal ist
es nicht - nicht wegen des Strahls, sondern wegen der Nachbar-Sites: die
liegen auf einem Quadratgitter, und in der Diagonale ist die naechste Site
sqrt(2) mal weiter weg. Genau das trennt die beiden Richtungen im
Crosstalk.

## Was sich bewegt und was steht

Bewegt wird NUR das Atom - und mit ihm die Bereiche, ueber die ausgewertet
wird: das lokale Sub-Gitter und die Gauss-Gewichtung W sitzen zentriert auf
der jeweiligen Atomposition.

Das LICHTFELD bleibt stehen: der Spot sitzt im Ursprung auf der Site, die
8 Nachbarkopien bei +-pitch darum herum. Wuerden Spot und Nachbarn
mitwandern, waere es eine reine Translation der ganzen Anordnung - bei
einem translationsinvarianten Profil kaeme dann fuer jeden Versatz genau
dasselbe heraus und alle Kurven waeren konstant.

Das ist dieselbe Semantik wie `atom_offset_x/y` im Multitone-Optimierer
(Atom relativ zu seiner Site und damit gleichermassen relativ zu allen
8 Nachbar-Sites versetzt). Sie deckt auch den Fall Strahl-Drift ab: wenn
die Pointing-Drift alle Strahlen gemeinsam verschiebt (gleicher AOD), ist
das aequivalent dazu, das Atom um -r zu verschieben - und weil der Satz der
8 Nachbarn inversionssymmetrisch ist, kommt exakt dasselbe heraus. Nur der
unphysikalische Fall "ein einzelner Strahl driftet, die Nachbarstrahlen
nicht" waere etwas anderes.

{hart_satz}

Die atom-gewichteten Groessen aendern sich in jedem Fall: das Atom sitzt im
Strahl woanders und sieht damit eine andere Intensitaet und ein anderes
Verhaeltnis zum Nachbarlicht.

Bei r = 0 stimmen alle Werte mit dem Waist-Sweep an diesem Waist ueberein.

## Ergebnis je Groesse

{tabelle}

"Rand" heisst: das Minimum liegt am Ende des abgefahrenen Bereichs.

## Werte

{_offset_werte_tabelle(results)}

## Parameter

| Groesse | Wert |
|---|---|
| Profil | {p['profile']} |
| airy_scale_factor | {p['airy_scale_factor']:.6g} |
| Waist | {results['waist_um']:.4f} µm |
| Kreisradius (hart) | {p['hard_radius'] * 1e6:.4f} µm |
| Crosstalk-Region (hart) | {sb.crosstalk_region_text(p)} |
| harte Region folgt dem Atom | {"ja" if folgt else "nein"} |
| pitch | {p['pitch'] * 1e6:.4f} µm |
| Nachbar-Sites | {(2 * int(p['neighbour_ring']) + 1) ** 2 - 1} |
| sigma_atom | {results['sigma_atom'] * 1e9:.2f} nm (T = {p['atom_temperature'] * 1e6:.2f} µK, nu_r = {p['trap_freq_r'] * 1e-3:.2f} kHz) |
| alpha / combo_lambda | {p['alpha']:.3f} / {p['combo_lambda']:.3f} |
"""

    out_path = FilePath(out_dir) / dateiname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Bericht gespeichert: {out_path}")
    return out_path


def _offset_werte_tabelle(results, n_zeilen=11):
    r_um = np.asarray(results["offset_um"], dtype=float)
    keys = list(results["kurven"])
    idx = np.unique(np.linspace(0, r_um.size - 1, min(n_zeilen, r_um.size)).astype(int))
    kopf = ("| r (µm) | r/w | "
            + " | ".join(f"{TEXT_LABELS[k]} (%)" for k in keys) + " |")
    trenner = "|" + "---|" * (len(keys) + 2)
    zeilen = [kopf, trenner]
    for i in idx:
        werte = [results[k][i] * 100.0 for k in keys]
        zeilen.append("| {:.4f} | {:.3f} | ".format(
            r_um[i], r_um[i] / results["waist_um"])
            + " | ".join(f"{v:.4g}" for v in werte) + " |")
    return "\n".join(zeilen)


# ======================================================================
# Bericht
# ======================================================================
def _zeile(results, key, name):
    werte = np.asarray(results[key], dtype=float) * 100.0
    endlich = werte[np.isfinite(werte)]
    best = results["best"].get(key)
    if endlich.size == 0 or best is None:
        return f"| {name} | n/a | n/a | n/a | n/a |"
    rand = " (Rand)" if best["at_edge"] else ""
    return (f"| {name} | {endlich.min():.4g} .. {endlich.max():.4g} % "
            f"| {sb.monotone_kind(results[key])} "
            f"| {best['value'] * 100:.4g} % "
            f"| {best['waist_um']:.4f} µm{rand} |")


def _arbeitspunkt_block(results):
    """Abschnitt "Markierter Punkt" - leer, wenn keiner gesetzt ist."""
    waist_um = results.get("arbeitspunkt_um")
    if waist_um is None:
        return ""
    p = results["params"]
    win_mm = float(sb.win_input_from_waist(
        waist_um * 1e-6, p["f1"], p["f2"], p["lambda_opt"], p["fLO"])) * 1e3
    werte = arbeitspunkt_werte(results, waist_um)
    zeilen = [
        "## Markierter Punkt (Stern im Plot)",
        "",
        f"- Waist = {waist_um:.4f} µm (win_input = {win_mm:.4f} mm)",
        "",
        "| Groesse | Wert |",
        "|---|---|",
    ]
    for key in kurven_keys(results):
        zeilen.append(f"| {TEXT_LABELS[key]} | {werte[key]:.4g} % |")
    zeilen += [
        "",
        "Linear zwischen den beiden benachbarten Stuetzstellen interpoliert.",
        "Die Kurven sind hier glatte Funktionen des Waists - anders als r_x/r_y",
        "im Multitone-Scan, die Optimierungs-ERGEBNISSE sind; dort waere",
        "Interpolieren irrefuehrend.",
        "",
    ]
    return "\n".join(zeilen)


def _ergebnis_tabelle(results):
    namen = zeilen_namen(results)
    kopf = ["| Groesse | Spanne | Verlauf | Minimum | bei Waist |",
            "|---|---|---|---|---|"]
    return "\n".join(kopf + [_zeile(results, k, namen[k]) for k in kurven_keys(results)])


def _werte_tabelle(results, n_zeilen=11):
    waist = np.asarray(results["waist_um"], dtype=float)
    win = np.asarray(results["win_input_mm"], dtype=float)
    keys = kurven_keys(results)
    idx = np.unique(np.linspace(0, waist.size - 1, min(n_zeilen, waist.size)).astype(int))
    kopf = ("| Waist (µm) | w_in (mm) | "
            + " | ".join(f"{TEXT_LABELS[k]} (%)" for k in keys) + " |")
    trenner = "|" + "---|" * (len(keys) + 2)
    zeilen = [kopf, trenner]
    for i in idx:
        werte = [results[k][i] * 100.0 for k in keys]
        zeilen.append("| {:.4f} | {:.4f} | ".format(waist[i], win[i])
                      + " | ".join(f"{v:.4g}" for v in werte) + " |")
    return "\n".join(zeilen)


def write_report(results, out_dir=None, prefix=None, dateiname=None):
    """Markdown-Bericht - dieselbe Rolle wie die Berichte der
    Multitone-Auswertung: was gerechnet wurde, womit, und was herauskam."""
    out_dir = paths.FIT_RESULTS_DIR if out_dir is None else out_dir
    prefix = output_prefix(results) if prefix is None else prefix
    dateiname = f"{prefix}_Report.md" if dateiname is None else dateiname

    p = results["params"]
    waist = np.asarray(results["waist_um"], dtype=float)
    win = np.asarray(results["win_input_mm"], dtype=float)
    radius_um = p["hard_radius"] * 1e6

    # Steht nur da, wenn der Crosstalk eine andere Flaeche nimmt als die
    # Uniformity - dann muss es dastehen.
    pitch_hinweis = "" if p.get("hard_crosstalk_region", "kreis") == "kreis" else (
        "\nAchtung: das Pitch-Quadrat ist eine ANDERE Flaeche als der Kreis, ueber\n"
        "dem die Uniformity laeuft. Es ist die Region, die die Multitone-Skripte\n"
        "fuer den Crosstalk benutzen - die Zahl ist damit direkt mit jenen Scans\n"
        "vergleichbar, aber nicht mit der Uniformity daneben.\n")

    text = f"""# SingleBeam - ein Strahl, {paths.profile_tag_of(p['profile'])}, {date.today().isoformat()}

Ein EINZELNER Strahl, kein Tonarray. Abgefahren wird die einzige freie
Groesse, die dabei bleibt: der Waist.

Weder `width` noch die Amplituden-Verhaeltnisse r_x/r_y kommen vor. `width`
spannt bei einem Ton nichts auf - das Ton-Quadrat der harten
Uniformity-Region ist nicht definiert -, und ein Aussen/Innen-Verhaeltnis
braucht mindestens zwei Toene je Achse.

## Was gerechnet wurde

**Hart** - die Uniformity ueber einer Kreisregion mit Radius {radius_um:.2f} µm
um die Site (Beam-Pointing-Region: das Atom kann irgendwo in diesem Kreis
sitzen), der Crosstalk ueber: **{sb.crosstalk_region_text(p)}**.

```
U_h   = std(I) / mean(I)               ueber dem Kreis
eta_h = sum(I_nachbar) / sum(I_eigen)  ueber der gewaehlten Region
```
{pitch_hinweis}

**Atom-gewichtet** - Definition unveraendert aus dem Optimierer, mit der
thermischen Ortsverteilung als Gewicht W:

```
U_w   = sqrt(<(I - <I>_W)^2>_W) / <I>_W
eta_w = sum(I_nachbar * W) / sum(I_eigen * W)
```

**Penalty-Kombination** - dieselbe Formel, die run_penalty_scan.py an jedem
Gitterpunkt minimiert:

```
U_c   = 0.5*(U_h + U_w) + lambda*|U_h - U_w|
eta_c = 0.5*(eta_h + eta_w) + lambda*|eta_h - eta_w|
J     = alpha*U_c + (1-alpha)*eta_c
```

**Wie eta_c und U_c zu lesen sind.** Solange eine der beiden Groessen
durchgehend groesser ist - hier fast immer eta_h > eta_w -, loest sich der
Betrag auf:

```
eta_c = (0.5 + lambda)*eta_h - (lambda - 0.5)*eta_w
      = {0.5 + p['combo_lambda']:.2f}*eta_h - {p['combo_lambda'] - 0.5:.2f}*eta_w    (bei lambda = {p['combo_lambda']:.2f})
```

Die KLEINERE der beiden Groessen geht also mit NEGATIVEM Vorzeichen ein.
Jedes Ringmaximum von eta_w drueckt eta_c damit nach unten: Diese Dellen sind
keine Eigenschaft des Crosstalks, sondern die Aussage "hier stimmen hartes
und gewichtetes Kriterium am besten ueberein" - und genau das soll der
Penalty-Term belohnen. Bei lambda = 0.5 waere eta_c exakt max(eta_h, eta_w),
darunter liegt es zwischen Mittelwert und Maximum. Fuer U_c gilt dasselbe.

I_nachbar ist die Summe der um +-pitch verschobenen Kopien desselben
Strahls: {(2 * int(p['neighbour_ring']) + 1) ** 2 - 1} Nachbar-Sites, also {'der Kranz direkt um die Site' if int(p['neighbour_ring']) == 1 else str(int(p['neighbour_ring'])) + ' Kraenze um die Site'}.
Weil das Profil translationsinvariant ist, ist das dieselbe Zahl wie "wieviel von diesem
Strahl faellt auf die Nachbar-Sites".

Nicht normiert wird bewusst: der Peak des Profils ist analytisch 1, und
beide Metriken sind gegen eine gemeinsame Skalierung von I_eigen und
I_nachbar invariant. Eine Normierung auf ein Gitter-Maximum waere hier
sogar falsch - das Maximum der Nachbar-Summe liegt ausserhalb der
ausgewerteten Region.

## Abgefahrener Bereich

- Waist in der Atomebene: {waist.min():.4f} .. {waist.max():.4f} µm
- Waist vor der ersten Linse: {win.min():.4f} .. {win.max():.4f} mm
- Stuetzstellen: {waist.size}{' (Lauf wurde abgebrochen)' if results.get('abgebrochen') else ''}

## Ergebnis je Groesse

{_ergebnis_tabelle(results)}

"Rand" heisst: das Minimum liegt am Ende des abgefahrenen Bereichs. Dann ist
es vermutlich kein Optimum, sondern nur die Grenze - weitersuchen.

{_arbeitspunkt_block(results)}
## Werte

{_werte_tabelle(results)}

Die vollstaendigen Kurven stehen im Datensatz (.pkl) und in den PDFs.

## Parameter

| Groesse | Wert |
|---|---|
| Profil | {p['profile']} |
| airy_scale_factor | {p['airy_scale_factor']:.6g} (first_zero_radius = Faktor * waist) |
| lambda | {p['lambda_opt'] * 1e9:.1f} nm |
| f1 / f2 / fLO | {p['f1'] * 1e3:.2f} / {p['f2'] * 1e3:.2f} / {p['fLO'] * 1e3:.2f} mm |
| pitch | {p['pitch'] * 1e6:.4f} µm |
| Nachbar-Sites | {(2 * int(p['neighbour_ring']) + 1) ** 2 - 1} ({int(p['neighbour_ring'])} Kranz um die Site) |
| Kreisradius (hart) | {radius_um:.4f} µm |
| Crosstalk-Region (hart) | {sb.crosstalk_region_text(p)} |
| Gitter (hart) | {int(p['hard_n_grid'])} x {int(p['hard_n_grid'])} Zellen (Mittelpunktsregel) je Region |
| sigma_atom | {results['sigma_atom'] * 1e9:.2f} nm (T = {p['atom_temperature'] * 1e6:.2f} µK, nu_r = {p['trap_freq_r'] * 1e-3:.2f} kHz) |
| Sub-Gitter (gewichtet) | {int(p['weighted_n_grid'])} x {int(p['weighted_n_grid'])} ueber +-{p['weighted_n_sigma']:g} sigma_atom |
| Atom-Versatz | {p['atom_offset_x'] * 1e6:.4f} / {p['atom_offset_y'] * 1e6:.4f} µm |
| alpha | {p['alpha']:.3f} |
| combo_lambda | {p['combo_lambda']:.3f} |
"""

    out_path = FilePath(out_dir) / dateiname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Bericht gespeichert: {out_path}")
    return out_path
