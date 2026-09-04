"""
lib/single_beam.py - EIN einzelner Strahl statt eines Tonarrays.

Worum es geht
-------------
Alles andere in diesem Ordner rechnet ein Multitone-Array: N_x x N_y Toene,
ein `width`, das die Spots aufspannt, und Amplituden-Verhaeltnisse r_x/r_y,
mit denen die Randtoene gegen die inneren gewichtet werden.

Bei EINEM Ton gibt es beides nicht mehr:

- `width` spannt nichts auf. Ein einzelner Punkt hat keine Ausdehnung, das
  "Ton-Quadrat" der harten Uniformity-Region ist nicht definiert.
- `r_x`/`r_y` sind bedeutungslos: das Verhaeltnis aussen/innen braucht
  mindestens zwei Toene pro Achse.

Uebrig bleibt genau eine freie Groesse: der WAIST. Dieses Modul faehrt sie
ab und liefert fuer jeden Waist

    U_h, eta_h    hart   - ueber einer KREISREGION mit Radius `hard_radius`
                           (Default 2 um) um die Site, nicht ueber dem
                           Ton-Quadrat und nicht ueber der Pitch-Box.
                           Gedacht als Beam-Pointing-Region: das Atom kann
                           irgendwo in diesem Kreis sitzen.
    U_w, eta_w    atom-gewichtet - unveraendert die Definition aus
                           weighted_multitone_flattop_optimizer.py
                           (thermische Ortsverteilung als Gewicht).
    U_c, eta_c, J   Penalty-Kombination - unveraendert die Formel aus
                           combine.py, also dieselbe, die
                           run_penalty_scan.py an jedem Gitterpunkt
                           minimiert.

Die Formeln werden NICHT hier neu geschrieben, sondern importiert:
`weighted_uniformity`, `weighted_crosstalk`, `atom_weight_2d`,
`sigma_thermal` und die Profilfunktionen kommen aus dem Optimierer,
`penalty_pair`/`penalty_objective` aus `combine.py`. Dieses Modul steuert
nur bei, WO ausgewertet wird (Kreis statt Kasten) und dass es nur einen
Spot gibt.

Warum hier nicht normiert wird
------------------------------
`_evaluate()` im Optimierer teilt die Intensitaet durch ihr Maximum auf dem
Rechengitter, und `create_neighbourhood()` normiert jede Nachbarkopie auf
deren eigenes Maximum. Beides ist bei einem einzelnen Spot ueberfluessig
und waere sogar gefaehrlich:

- Der Peak des Profils ist analytisch 1 (Airy: (2*J1(u)/u)^2 -> 1 fuer
  u -> 0; Gauss: exp(0) = 1). Es gibt nichts zu normieren.
- Das Maximum der NACHBAR-Summe liegt ~pitch entfernt und damit ausserhalb
  des ausgewerteten Kreises. Eine Normierung auf das dort sichtbare
  Maximum wuerde die Nachbarintensitaet massiv aufblasen - genau der
  Fehler, den `_local_neighbor_intensity()` im Optimierer beschreibt.

Beide Metriken sind ohnehin invariant gegen eine GEMEINSAME Skalierung von
I_own und I_neighbor (U ist ein Variationskoeffizient, eta ein Verhaeltnis
zweier Summen). Es wird deshalb schlicht mit dem analytischen Peak 1
gerechnet.

Was der Crosstalk bei einem Strahl bedeutet
-------------------------------------------
Unveraendert `sum(I_neighbor) / sum(I_own)` ueber der Region, wobei
I_neighbor die Summe der um +-pitch verschobenen Kopien desselben Strahls
ist. Weil das Profil translationsinvariant ist, ist das dieselbe Zahl wie
"wieviel von MEINEM Strahl faellt auf die Nachbar-Sites" - fuer einen
einzelnen Adressierstrahl also genau die interessierende Groesse.
"""

import pickle
from pathlib import Path as FilePath

import numpy as np

from . import paths  # noqa: F401  (setzt sys.path fuer die Importe unten)
from .combine import penalty_pair, penalty_objective

from weighted_multitone_flattop_optimizer import (  # noqa: E402
    PROFILE_FUNCS, RB85_MASS, atom_weight_2d, beam_radius_scaled,
    sigma_thermal, weighted_crosstalk, weighted_uniformity,
)

from airy_scale import AIRY_SCALE_LEGACY, scale_tag  # noqa: E402


# ======================================================================
# Voreinstellungen
# ======================================================================
# Optik: die Werte, mit denen die vorhandenen Datensaetze dieses Projekts
# gerechnet wurden (f1 = 75 mm, siehe MultitoneFlatTopOptimizer.DEFAULTS) -
# damit ein Waist hier dieselbe Zahl bedeutet wie dort. Im Dialog frei
# aenderbar.
DEFAULTS = dict(
    # --- Strahlprofil ---
    profile="airy",
    airy_scale_factor=AIRY_SCALE_LEGACY,   # first_zero_radius = Faktor * waist

    # --- Optik (nur fuer die Umrechnung win_input <-> Waist) ---
    lambda_opt=795e-9,
    f1=75e-3,
    f2=750e-3,
    fLO=52.88e-3,

    # --- Gitter der Fallen (fuer den Crosstalk) ---
    pitch=5.288e-6,
    neighbour_ring=1,          # 1 -> die 8 direkten Nachbarn (wie im Optimierer)

    # --- Harte Region: Kreis statt Kasten ---
    hard_radius=1.0e-6,        # Radius (nicht Durchmesser) der Beam-Pointing-Region
    hard_n_grid=401,           # Zellen pro Achse (Mittelpunktsregel)
    # Region fuer den harten CROSSTALK - die Uniformity nimmt immer den Kreis:
    #   "kreis" derselbe Kreis wie die Uniformity (Beam-Pointing-Region)
    #   "pitch" das Pitch-Quadrat mit Seitenlaenge pitch, wie in den
    #           Multitone-Skripten (_build_masks -> overlap_mask_pitch)
    #   "beide" beide gleichzeitig - zwei Crosstalk-Kurven nebeneinander
    hard_crosstalk_region="beide",
    # Nur bei "beide": welche der beiden in die Penalty-Kombination (und
    # damit in eta_c und J) eingeht. Die Kombination braucht EINE Definition
    # von eta_h; welche das ist, muss dastehen und nicht geraten werden.
    penalty_crosstalk_region="kreis",

    # --- Atom-gewichtete Metriken (Definition wie im Optimierer) ---
    atom_mass=RB85_MASS,
    atom_temperature=17e-6,    # K
    trap_freq_r=60.4e3,        # Hz, radiale Fallenfrequenz nu_r
    weighted_n_sigma=6,        # Ausdehnung des lokalen Sub-Grids in sigma_atom
    weighted_n_grid=241,       # Punkte pro Achse im lokalen Sub-Grid
    atom_offset_x=0.0,         # Meter, Versatz des Atoms gegen die Site-Mitte
    atom_offset_y=0.0,

    # Positions-Sweep (sweep_offset): wandert die harte Region mit dem
    # Atom? True ist die sinnvolle Vorgabe - dort ist die Atomposition die
    # abgefahrene Groesse. False laesst sie auf der Site, dann sind die
    # harten Kurven ueber dem Versatz konstant.
    offset_hard_follows_atom=True,

    # --- Penalty-Kombination (identisch zu run_penalty_scan.py) ---
    alpha=0.7,
    combo_lambda=0.75,
)

# Die Groessen, die als Kurve ueber dem Waist herauskommen. Reihenfolge und
# Schluessel werden von single_beam_report.py und vom Bericht benutzt.
METRIC_KEYS = (
    "uniformity_hart", "crosstalk_hart",
    "crosstalk_hart_kreis", "crosstalk_hart_pitch",
    "uniformity_weighted", "crosstalk_weighted",
    "uniformity_kombi", "crosstalk_kombi",
    "combined_score",
)


def vorhandene_metriken(results):
    """Die Kurven, die dieser Datensatz wirklich enthaelt, in der
    Reihenfolge von METRIC_KEYS. Die beiden Regionen-Kurven gibt es nur im
    Modus "beide"."""
    return tuple(k for k in METRIC_KEYS if k in results)

PKL_GLOB = "single_beam_*.pkl"

# Der Startbereich, mit dem der Dialog aufgeht - der Arbeitsbereich dieses
# Aufbaus. Festgelegt wird er an seinen beiden Enden:
#
#   - Anfang der Auftragung: 1.8 mm VOR der ersten Linse (der kleinste
#     Waist in der Atomebene, also der linke Rand der Figuren)
#   - Ende: 2.5 µm in der Atomebene
#
# Die jeweils andere Schreibweise wird daraus mit den Default-Brennweiten
# ausgerechnet, statt sie ein zweites Mal von Hand hinzuschreiben - so
# beschreiben beide garantiert DASSELBE Fenster, und beim Umschalten der
# Einheit im Dialog wechselt nur die Schreibweise, nicht der Bereich.
WIN_INPUT_START_DEFAULT_MM = 1.8
WAIST_ENDE_DEFAULT_UM = 2.5

_C_MM_UM = 1e3 * 1e6 * (DEFAULTS["f1"] / DEFAULTS["f2"]) * \
    DEFAULTS["lambda_opt"] * DEFAULTS["fLO"] / np.pi

WAIST_RANGE_DEFAULT_UM = (round(_C_MM_UM / WIN_INPUT_START_DEFAULT_MM, 4),
                          WAIST_ENDE_DEFAULT_UM)
WIN_INPUT_RANGE_DEFAULT_MM = (round(_C_MM_UM / WAIST_ENDE_DEFAULT_UM, 4),
                              WIN_INPUT_START_DEFAULT_MM)

# Voreingestellter Arbeitspunkt (Waist in µm): wird in den Figuren markiert
# und im Bericht ausgewertet.
ARBEITSPUNKT_DEFAULT_UM = 1.9

# ----------------------------------------------------------------------
# Positions-Sweep: fester Waist, das ATOM wandert
# ----------------------------------------------------------------------
# Gefahren wird der Betrag r des Versatzes; die Richtung sagt, wie er sich
# auf x und y verteilt. Eine dritte, waagerechte Richtung braucht es nicht:
# das Profil ist rotationssymmetrisch, waagerecht ist dasselbe wie
# senkrecht. Diagonal ist es NICHT dasselbe - nicht wegen des Strahls,
# sondern wegen der Nachbar-Sites: die liegen auf einem Quadratgitter, und
# in der Diagonale ist die naechste Site sqrt(2) mal weiter weg.
OFFSET_RICHTUNGEN = [
    ("vertikal", "senkrecht  (0, r)"),
    ("diagonal", "diagonal  (r/sqrt2, r/sqrt2)"),
]

OFFSET_STUETZSTELLEN_DEFAULT = 61


def offset_vektor(richtung, r):
    """(dx, dy) zu einem Versatz-BETRAG r in der gewaehlten Richtung."""
    if richtung == "vertikal":
        return 0.0, float(r)
    if richtung == "diagonal":
        return float(r) / np.sqrt(2.0), float(r) / np.sqrt(2.0)
    raise ValueError(f"Unbekannte Richtung '{richtung}'.")


def offset_key(basis, richtung):
    """Schluessel einer Kurve im Positions-Sweep: Groesse + Richtung."""
    return f"{basis}__{richtung}"

# (Schluessel, Anzeigetext) - die Wahl steht im Dialog und im Bericht.
CROSSTALK_REGION_CHOICES = [
    ("kreis", "derselbe Kreis wie die Uniformity"),
    ("pitch", "Pitch-Quadrat (Seitenlaenge = pitch), wie im Multitone-Fall"),
    ("beide", "beide gleichzeitig - zwei Crosstalk-Kurven"),
]

# Nur bei "beide" gefragt: welche Region die Penalty-Kombination benutzt.
PENALTY_REGION_CHOICES = [
    ("kreis", "Kreis"),
    ("pitch", "Pitch-Quadrat"),
]

# Die beiden harten Crosstalk-Kurven, wenn beide gerechnet werden. Bei nur
# einer Region heisst sie schlicht "crosstalk_hart" - so bleiben die
# Datensaetze und Plots der Ein-Regionen-Faelle unveraendert.
CROSSTALK_KEYS = dict(kreis="crosstalk_hart_kreis", pitch="crosstalk_hart_pitch")


def beide_regionen(params):
    return params.get("hard_crosstalk_region", "kreis") == "beide"


def penalty_region(params):
    """Welche Region in eta_c/J eingeht. Ausserhalb von "beide" ist das die
    einzige gerechnete."""
    region = params.get("hard_crosstalk_region", "kreis")
    if region != "beide":
        return region
    return params.get("penalty_crosstalk_region", "kreis")


def crosstalk_region_tag(params):
    """Namenskuerzel fuer die Crosstalk-Region - leer beim Kreis (dem
    Default). Zwei Laeufe mit verschiedenen Regionen sind nicht
    vergleichbar und duerfen sich nicht ueberschreiben."""
    region = params.get("hard_crosstalk_region", "kreis")
    if region == "kreis":
        return ""
    if region == "pitch":
        return "_pitchbox"
    return "_beide" if penalty_region(params) == "kreis" else "_beide-pitchbox"


def region_text(region, params):
    """Klartext EINER Region."""
    if region == "kreis":
        return "Kreis mit Radius %.4g µm" % (float(params["hard_radius"]) * 1e6)
    return "Pitch-Quadrat, Seitenlaenge %.4g µm" % (float(params["pitch"]) * 1e6)


def crosstalk_region_text(params):
    """Klartext fuer Bericht und Plot-Titel."""
    region = params.get("hard_crosstalk_region", "kreis")
    if region != "beide":
        return region_text(region, params)
    return ("%s UND %s (in eta_c/J geht der %s ein)"
            % (region_text("kreis", params), region_text("pitch", params),
               region_text(penalty_region(params), params)))


def profile_tag_of(profile):
    return paths.profile_tag_of(profile)


def airy_tag(params):
    """Namenskuerzel fuer den airy_scale_factor - leer beim historischen
    1.19, sonst z.B. "_k1.483". Datensaetze mit verschiedenen Faktoren sind
    NICHT vergleichbar (siehe airy_scale.py); das Kuerzel im Dateinamen
    verhindert, dass einer den anderen ueberschreibt. Beim Gauss-Profil
    spielt der Faktor keine Rolle und das Kuerzel entfaellt."""
    if params.get("profile") != "airy":
        return ""
    return scale_tag(params["airy_scale_factor"])


# ======================================================================
# Umrechnung Waist in der Atomebene <-> Waist vor der ersten Linse
# ======================================================================
def waist_from_win_input(win_input, f1, f2, lambda_opt, fLO):
    """Waist in der Atomebene (m) aus dem Waist vor der ersten Linse (m).

    Dieselbe Beziehung wie MultitoneFlatTopOptimizer.win_input_to_win():

        waist = (f1/f2) * lambda_opt * fLO / (pi * win_input)
    """
    return beam_radius_scaled(f1, f2, lambda_opt, fLO, np.asarray(win_input, dtype=float))


def win_input_from_waist(waist, f1, f2, lambda_opt, fLO):
    """Die Umkehrung. Die Beziehung ist ihre eigene Umkehrfunktion bis auf
    die Konstante: beide Richtungen sind waist = C / win_input mit
    C = (f1/f2) * lambda_opt * fLO / pi."""
    waist = np.asarray(waist, dtype=float)
    const = (f1 / f2) * lambda_opt * fLO / np.pi
    return const / waist


def profile_scale(waist, profile, airy_scale_factor):
    """Der Skalenparameter, den die Profilfunktion erwartet: sigma beim
    Gauss-Profil, first_zero_radius = airy_scale_factor * waist beim Airy."""
    if profile == "airy":
        return airy_scale_factor * waist
    return waist


# ======================================================================
# Geometrie: ein Spot, seine Nachbarn
# ======================================================================
def neighbour_centers(pitch, ring=1):
    """Zentren der Nachbar-Sites um den Ursprung, ohne den Ursprung selbst.

    ring=1 sind die 8 direkten Nachbarn - genau die, die
    create_neighbourhood() im Optimierer summiert. ring=2 nimmt den
    naechsten Kranz dazu (24 Sites); bei Airy faellt der Beitrag mit 1/r^3
    ab, ist also klein, aber nicht null.
    """
    idx = np.arange(-ring, ring + 1)
    ix, iy = np.meshgrid(idx, idx)
    keep = ~((ix == 0) & (iy == 0))
    return ix[keep].ravel() * pitch, iy[keep].ravel() * pitch


def _own_intensity(X, Y, scale, profile_func):
    """Der eine Spot, im Ursprung. Peak analytisch 1, siehe Modul-Docstring."""
    return profile_func(X, Y, np.zeros(1), np.zeros(1), scale, np.ones(1))


def _neighbour_intensity(X, Y, scale, profile_func, pitch, ring):
    """Summe der Nachbarkopien - ohne jede Renormierung, aus demselben
    Grund, aus dem _local_neighbor_intensity() im Optimierer sie weglaesst."""
    cx, cy = neighbour_centers(pitch, ring)
    return profile_func(X, Y, cx, cy, scale, np.ones(cx.size))


# ======================================================================
# Die harten Metriken auf der Kreisregion
# ======================================================================
def _zellmitten(halbe_breite, n):
    """n gleich breite Zellen ueber [-halbe_breite, +halbe_breite], und
    zwar deren MITTEN.

    Nicht np.linspace(-a, a, n): das legt Punkte auf beide Raender und
    gewichtet die Randzeilen damit doppelt so stark, wie ihnen zusteht -
    ein Fehler, der mit 1/n abfaellt und beim Pitch-Quadrat (wo der Rand
    mitten im Signal liegt) gemessen 0.6 % ausmachte. Mit den Zellmitten
    ist es die Mittelpunktsregel, und die Summe konvergiert quadratisch.

    Bei ungeradem n liegt eine Mitte genau auf 0, der Profil-Peak wird
    also weiterhin getroffen."""
    schritt = 2.0 * halbe_breite / n
    return -halbe_breite + (np.arange(n) + 0.5) * schritt


def _gitter_kreis(radius, n, zentrum=(0.0, 0.0)):
    """Umschriebenes Quadrat mit n Zellen je Achse, plus die Kreismaske.
    Die Zellen sind flaechengleich - die Summen ueber die Maske sind also
    proportional zu den Integralen ueber den Kreis.

    `zentrum` verschiebt die Region; der Strahl bleibt im Ursprung."""
    cx, cy = zentrum
    xs = _zellmitten(radius, n)
    X, Y = np.meshgrid(xs + cx, xs + cy)
    return X, Y, ((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2


def _gitter_pitchbox(pitch, n, zentrum=(0.0, 0.0)):
    """Das Pitch-Quadrat (Seitenlaenge = pitch) um die Site. Hier liegt
    JEDER Gitterpunkt in der Region, eine Maske braucht es nicht - das
    Gitter IST die Region. Dieselbe Region, die overlap_mask_pitch() im
    Multitone-Optimierer aus dem globalen Gitter ausschneidet."""
    cx, cy = zentrum
    xs = _zellmitten(0.5 * pitch, n)
    X, Y = np.meshgrid(xs + cx, xs + cy)
    return X, Y, np.ones_like(X, dtype=bool)


def hard_metrics(waist, params, zentrum=(0.0, 0.0)):
    """Die harten Metriken. Rueckgabe: dict mit "uniformity", "kreis" und
    "pitch" - die nicht angeforderten Crosstalk-Werte sind NaN.

    Uniformity IMMER ueber dem Kreis mit Radius params['hard_radius'] um
    die Site - das ist die Beam-Pointing-Region, und sie ist der Grund,
    warum es hier ueberhaupt eine harte Region gibt.

    Crosstalk wahlweise (params['hard_crosstalk_region']):

        "kreis"  ueber demselben Kreis - eine Region fuer beide Groessen
        "pitch"  ueber dem Pitch-Quadrat mit Seitenlaenge pitch, also der
                 Region, die die Multitone-Skripte fuer den Crosstalk
                 benutzen. Dann sind die Zahlen mit jenen Scans direkt
                 vergleichbar, beziehen sich aber auf eine ANDERE Flaeche
                 als die Uniformity daneben.
        "beide"  beide, in einem Durchgang. Die Uniformity wird dabei nur
                 EINMAL gerechnet - sie haengt gar nicht an der
                 Crosstalk-Region -, die beiden Crosstalks laufen ueber
                 ihre je eigenen Gitter.

        U_h   = std(I) / mean(I)               ueber dem Kreis
        eta_h = sum(I_neighbor) / sum(I_own)   ueber der gewaehlten Region

    Beide Regionen liegen auf `zentrum` (Default: der Site, also dort, wo
    auch der Strahl steht). Im Waist-Sweep bleibt das so, auch wenn
    atom_offset_x/y gesetzt sind - genau wie im Multitone-Optimierer, wo
    der Versatz nur die atom-gewichteten Metriken betrifft: die harte
    Region ist eine geometrisch vorgegebene Region, keine Aussage darueber,
    wo das Atom gerade sitzt.

    Der Positions-Sweep (sweep_offset) kann sie auf Wunsch dem Atom
    nachfuehren - dort IST die Atomposition ja die abgefahrene Groesse.

    Beim Pitch-Quadrat wird ein ZWEITES Gitter aufgebaut statt eines
    gemeinsamen, groesseren: sonst haengt die Aufloesung des Kreises daran,
    wie gross der Pitch gerade ist, und U_h aenderte sich mit einer
    Einstellung, die es gar nicht betrifft.
    """
    radius = float(params["hard_radius"])
    n = int(params["hard_n_grid"])
    region = params.get("hard_crosstalk_region", "kreis")
    profile_func = PROFILE_FUNCS[params["profile"]]
    scale = profile_scale(waist, params["profile"], params["airy_scale_factor"])
    pitch = float(params["pitch"])
    ring = int(params["neighbour_ring"])

    def auf_gitter(X, Y, inside):
        I_own = _own_intensity(X, Y, scale, profile_func)[inside]
        I_nb = _neighbour_intensity(X, Y, scale, profile_func, pitch, ring)[inside]
        return I_own, I_nb

    X, Y, inside = _gitter_kreis(radius, n, zentrum)
    I_own_kreis, I_nb_kreis = auf_gitter(X, Y, inside)

    mean_own = np.mean(I_own_kreis)
    uniformity = (float(np.std(I_own_kreis) / mean_own)
                  if np.isfinite(mean_own) and mean_own != 0 else np.nan)

    werte = dict(uniformity=uniformity, kreis=np.nan, pitch=np.nan)

    if region in ("kreis", "beide"):
        summe = np.sum(I_own_kreis)
        werte["kreis"] = float(np.sum(I_nb_kreis) / summe) if summe != 0 else np.nan

    if region in ("pitch", "beide"):
        I_own_box, I_nb_box = auf_gitter(*_gitter_pitchbox(pitch, n, zentrum))
        summe = np.sum(I_own_box)
        werte["pitch"] = float(np.sum(I_nb_box) / summe) if summe != 0 else np.nan

    return werte


# ======================================================================
# Die atom-gewichteten Metriken
# ======================================================================
def weighted_metrics(waist, params, sigma_atom=None):
    """Uniformity und Crosstalk mit der thermischen Atom-Verteilung als
    Gewicht - Definition unveraendert aus weighted_uniformity() und
    weighted_crosstalk() des Optimierers:

        U_w   = sqrt(<(I-<I>_W)^2>_W) / <I>_W
        eta_w = sum(I_neighbor * W) / sum(I_own * W)

    Ausgewertet auf einem lokalen Sub-Grid von +-weighted_n_sigma *
    sigma_atom um die Atomposition (Site-Mitte plus atom_offset_x/y).
    """
    if sigma_atom is None:
        sigma_atom = sigma_thermal(
            params["atom_mass"], 2 * np.pi * params["trap_freq_r"], params["atom_temperature"]
        )
    if not np.isfinite(sigma_atom) or sigma_atom <= 0:
        return np.nan, np.nan

    profile_func = PROFILE_FUNCS[params["profile"]]
    scale = profile_scale(waist, params["profile"], params["airy_scale_factor"])

    atom_x = float(params["atom_offset_x"])
    atom_y = float(params["atom_offset_y"])
    half = float(params["weighted_n_sigma"]) * sigma_atom
    n = int(params["weighted_n_grid"])
    xs = np.linspace(atom_x - half, atom_x + half, n)
    ys = np.linspace(atom_y - half, atom_y + half, n)
    X, Y = np.meshgrid(xs, ys)

    I_own = _own_intensity(X, Y, scale, profile_func)
    I_nb = _neighbour_intensity(
        X, Y, scale, profile_func, params["pitch"], int(params["neighbour_ring"])
    )
    W = atom_weight_2d(X, Y, atom_x, atom_y, sigma_atom)

    return float(weighted_uniformity(I_own, W)), float(weighted_crosstalk(I_own, I_nb, W))


# ======================================================================
# Der Sweep
# ======================================================================
def build_waist_grid(mode, lo, hi, n, params):
    """Die abzufahrenden Waists (in Metern) plus die zugehoerigen
    Eingangswaists.

    mode='after_lens': lo/hi sind Waists in um, linear abgetastet.
    mode='before_lens': lo/hi sind Eingangswaists in mm, LINEAR in
        win_input abgetastet - die daraus folgenden Waists liegen dann
        nicht aequidistant, weil waist ~ 1/win_input. Genau so wird auch
        in den Multitone-Scans gescannt (dort ist win_input die
        Scan-Achse), deshalb bleibt es hier dabei.
    """
    if n < 2:
        raise ValueError("Mindestens zwei Stuetzstellen noetig.")
    if lo >= hi:
        raise ValueError("Der Bereich muss aufsteigend sein (von < bis).")

    if mode == "before_lens":
        win_input = np.linspace(lo * 1e-3, hi * 1e-3, int(n))
        waist = waist_from_win_input(
            win_input, params["f1"], params["f2"], params["lambda_opt"], params["fLO"])
        order = np.argsort(waist)
        return waist[order], win_input[order]

    if mode == "after_lens":
        waist = np.linspace(lo * 1e-6, hi * 1e-6, int(n))
        win_input = win_input_from_waist(
            waist, params["f1"], params["f2"], params["lambda_opt"], params["fLO"])
        return waist, win_input

    raise ValueError(f"Unbekannter Modus '{mode}' (erwartet 'before_lens' oder 'after_lens').")


def sweep(waist_vals, params, progress=None):
    """Rechnet alle Metriken fuer jeden Waist in `waist_vals` (Meter).

    progress: optionales Callable progress(i, n) -> bool. Gibt es False
    zurueck, wird abgebrochen und der Rest bleibt NaN (fuer den
    Abbrechen-Knopf des Dialogs).

    Rueckgabe: dict mit den Kurven, den Parametern und den Bestpunkten -
    dasselbe dict, das gepickelt und geplottet wird.
    """
    merged = dict(DEFAULTS)
    merged.update(params or {})

    waist_vals = np.asarray(waist_vals, dtype=float)
    n = waist_vals.size

    sigma_atom = sigma_thermal(
        merged["atom_mass"], 2 * np.pi * merged["trap_freq_r"], merged["atom_temperature"]
    )

    U_h = np.full(n, np.nan)
    C_kreis = np.full(n, np.nan)
    C_pitch = np.full(n, np.nan)
    U_w = np.full(n, np.nan)
    C_w = np.full(n, np.nan)

    abgebrochen = False
    for i, waist in enumerate(waist_vals):
        if progress is not None and progress(i, n) is False:
            abgebrochen = True
            break
        if waist <= 0:
            continue
        hart = hard_metrics(waist, merged)
        U_h[i] = hart["uniformity"]
        C_kreis[i] = hart["kreis"]
        C_pitch[i] = hart["pitch"]
        U_w[i], C_w[i] = weighted_metrics(waist, merged, sigma_atom=sigma_atom)

    # Die Kurve, die als "der" harte Crosstalk gilt: bei einer Region die
    # eine, bei "beide" die im Dialog dafuer bestimmte. Nur sie geht in die
    # Penalty-Kombination ein - eta_c und J brauchen EINE Definition.
    C_h = C_kreis if penalty_region(merged) == "kreis" else C_pitch

    U_c = penalty_pair(U_h, U_w, merged["combo_lambda"])
    eta_c = penalty_pair(C_h, C_w, merged["combo_lambda"])
    J = penalty_objective(U_h, C_h, U_w, C_w, merged["alpha"], merged["combo_lambda"])

    win_input = win_input_from_waist(
        waist_vals, merged["f1"], merged["f2"], merged["lambda_opt"], merged["fLO"])

    results = dict(
        kind="single_beam",
        waist=waist_vals,
        waist_um=waist_vals * 1e6,
        win_input=win_input,
        win_input_mm=win_input * 1e3,
        uniformity_hart=U_h, crosstalk_hart=C_h,
        uniformity_weighted=U_w, crosstalk_weighted=C_w,
        uniformity_kombi=U_c, crosstalk_kombi=eta_c,
        combined_score=J,
        sigma_atom=float(sigma_atom),
        params=merged,
        abgebrochen=abgebrochen,
    )
    if beide_regionen(merged):
        results["crosstalk_hart_kreis"] = C_kreis
        results["crosstalk_hart_pitch"] = C_pitch

    results["best"] = best_points(results)
    return results


def best_points(results, keys=None):
    """Je Groesse die Stelle mit dem kleinsten Wert, plus ob die am Rand
    des abgefahrenen Bereichs liegt (dann ist es vermutlich kein Optimum,
    sondern nur der Rand).

    Traegt der Datensatz eine Waist-Achse, steht sie mit im Ergebnis;
    beim Positions-Sweep ist es die Versatz-Achse."""
    keys = vorhandene_metriken(results) if keys is None else keys
    achse = "offset_um" if "offset_um" in results else "waist_um"
    x = np.asarray(results[achse], dtype=float)
    best = {}
    for key in keys:
        werte = np.asarray(results[key], dtype=float)
        finite = np.isfinite(werte)
        if not np.any(finite):
            best[key] = None
            continue
        i = int(np.argmin(np.where(finite, werte, np.inf)))
        eintrag = dict(index=i, value=float(werte[i]),
                       at_edge=bool(i in (0, werte.size - 1)))
        eintrag[achse] = float(x[i])
        if "win_input_mm" in results and achse == "waist_um":
            eintrag["win_input_mm"] = float(np.asarray(results["win_input_mm"])[i])
        best[key] = eintrag
    return best


def monotone_kind(werte):
    """'faellt', 'steigt', 'hat ein Minimum' oder 'uneindeutig' - fuer den
    Bericht. Rein beschreibend, ohne Glaettung: entschieden wird an den
    Vorzeichen der Differenzen der endlichen Werte."""
    werte = np.asarray(werte, dtype=float)
    finite = werte[np.isfinite(werte)]
    if finite.size < 3:
        return "zu wenige Punkte"
    d = np.diff(finite)
    if np.all(d == 0):
        return "konstant"
    if np.all(d >= 0):
        return "steigt"
    if np.all(d <= 0):
        return "faellt"
    i = int(np.argmin(finite))
    if 0 < i < finite.size - 1 and np.all(d[:i] <= 0) and np.all(d[i:] >= 0):
        return "hat ein Minimum"
    return "uneindeutig"


# ======================================================================
# Positions-Sweep: fester Waist, das Atom wandert
# ======================================================================
def sweep_offset(waist, params, n=None, r_max=None,
                 richtungen=("vertikal", "diagonal"), progress=None):
    """Alle Metriken ueber dem VERSATZ des Atoms, bei festem Waist.

    r laeuft von 0 (Atom auf der Site) bis `r_max` - Default ist der Waist
    selbst, also bis dorthin, wo die Intensitaet auf 1/e^2 abgefallen
    waere. Fuer jede Richtung in `richtungen` gibt es einen vollen Satz
    Kurven; die Schluessel tragen die Richtung als Endung
    (offset_key(), z.B. "uniformity_weighted__diagonal").

    Was sich mit dem Versatz aendert:

    - Die atom-gewichteten Groessen immer: das Atom sitzt im Strahl
      woanders, sieht also eine andere Intensitaet und ein anderes
      Verhaeltnis zum Nachbarlicht.
    - Die HARTEN Groessen nur, wenn `offset_hard_follows_atom` gesetzt ist
      (Default). Dann wandert die Kreis- bzw. Pitch-Region mit dem Atom.
      Das ist hier die sinnvolle Lesart - die Atomposition IST die
      abgefahrene Groesse, und die harte Region fragt "wie sieht es DA
      aus". Steht der Schalter auf False, bleibt die Region auf der Site,
      und die harten Kurven sind ueber dem Versatz konstant - was auch eine
      Aussage ist, nur eben eine langweilige.

    Bei r = 0 stimmen die Werte exakt mit dem Waist-Sweep an diesem Waist
    ueberein; der Positions-Sweep faengt also dort an, wo jener steht.
    """
    merged = dict(DEFAULTS)
    merged.update(params or {})
    waist = float(waist)
    n = int(OFFSET_STUETZSTELLEN_DEFAULT if n is None else n)
    r_max = float(waist if r_max is None else r_max)
    folgt = bool(merged.get("offset_hard_follows_atom", True))
    richtungen = tuple(richtungen)
    if not richtungen:
        raise ValueError("Mindestens eine Richtung waehlen.")
    if n < 2:
        raise ValueError("Mindestens zwei Stuetzstellen noetig.")

    sigma_atom = sigma_thermal(
        merged["atom_mass"], 2 * np.pi * merged["trap_freq_r"], merged["atom_temperature"]
    )
    r_vals = np.linspace(0.0, r_max, n)

    results = dict(
        kind="single_beam_offset",
        waist=waist, waist_um=waist * 1e6,
        win_input_mm=float(win_input_from_waist(
            waist, merged["f1"], merged["f2"], merged["lambda_opt"], merged["fLO"])) * 1e3,
        offset=r_vals, offset_um=r_vals * 1e6,
        offset_in_waist=r_vals / waist if waist > 0 else np.full(n, np.nan),
        richtungen=richtungen,
        sigma_atom=float(sigma_atom),
        params=merged,
        hard_follows_atom=folgt,
        abgebrochen=False,
    )

    gesamt = n * len(richtungen)
    getan = 0
    keys = []
    for richtung in richtungen:
        U_h = np.full(n, np.nan)
        C_kreis = np.full(n, np.nan)
        C_pitch = np.full(n, np.nan)
        U_w = np.full(n, np.nan)
        C_w = np.full(n, np.nan)

        for i, r in enumerate(r_vals):
            if progress is not None and progress(getan, gesamt) is False:
                results["abgebrochen"] = True
                break
            getan += 1
            dx, dy = offset_vektor(richtung, r)
            hart = hard_metrics(waist, merged, zentrum=(dx, dy) if folgt else (0.0, 0.0))
            U_h[i] = hart["uniformity"]
            C_kreis[i] = hart["kreis"]
            C_pitch[i] = hart["pitch"]
            punkt = dict(merged, atom_offset_x=dx, atom_offset_y=dy)
            U_w[i], C_w[i] = weighted_metrics(waist, punkt, sigma_atom=sigma_atom)

        C_h = C_kreis if penalty_region(merged) == "kreis" else C_pitch
        werte = {
            "uniformity_hart": U_h,
            "crosstalk_hart": C_h,
            "uniformity_weighted": U_w,
            "crosstalk_weighted": C_w,
            "uniformity_kombi": penalty_pair(U_h, U_w, merged["combo_lambda"]),
            "crosstalk_kombi": penalty_pair(C_h, C_w, merged["combo_lambda"]),
            "combined_score": penalty_objective(
                U_h, C_h, U_w, C_w, merged["alpha"], merged["combo_lambda"]),
        }
        if beide_regionen(merged):
            werte["crosstalk_hart_kreis"] = C_kreis
            werte["crosstalk_hart_pitch"] = C_pitch

        for basis in METRIC_KEYS:
            if basis not in werte:
                continue
            key = offset_key(basis, richtung)
            results[key] = werte[basis]
            # Im Modus "beide" ist crosstalk_hart eine der beiden
            # Regionen-Kurven; sie noch einmal aufzufuehren waere dieselbe
            # Zahl unter zwei Namen (dieselbe Regel wie kurven_keys()).
            if not (basis == "crosstalk_hart" and beide_regionen(merged)):
                keys.append(key)

        if results["abgebrochen"]:
            break

    results["kurven"] = tuple(keys)
    results["best"] = best_points(results, keys)
    return results


def offset_pkl_name(params, waist_um, n_points):
    """single_beam_offset_{Profil}_w{Waist}um_{n}pts_r{Radius}um....pkl"""
    tag = paths.profile_tag_of(params["profile"])
    radius_um = float(params["hard_radius"]) * 1e6
    return (f"single_beam_offset_{tag}_w{waist_um:g}um_{int(n_points)}pts"
            f"_r{radius_um:g}um{crosstalk_region_tag(params)}{airy_tag(params)}.pkl")


# ======================================================================
# Speichern / Laden
# ======================================================================
def pkl_name(params, n_points):
    """single_beam_{Profil}_{n}pts_r{Radius}um{_pitchbox}{_kFaktor}.pkl"""
    tag = paths.profile_tag_of(params["profile"])
    radius_um = float(params["hard_radius"]) * 1e6
    return (f"single_beam_{tag}_{int(n_points)}pts_r{radius_um:g}um"
            f"{crosstalk_region_tag(params)}{airy_tag(params)}.pkl")


def save_results(results, filepath):
    filepath = FilePath(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as fh:
        pickle.dump(results, fh)
    return filepath


def load_results(filepath):
    with open(FilePath(filepath), "rb") as fh:
        results = pickle.load(fh)
    if results.get("kind") not in ("single_beam", "single_beam_offset"):
        raise ValueError(
            f"{filepath} ist kein Einzelstrahl-Datensatz (kind={results.get('kind')!r}). "
            f"Die Multitone-Datensaetze wertet run_plots.py aus.")
    return results
