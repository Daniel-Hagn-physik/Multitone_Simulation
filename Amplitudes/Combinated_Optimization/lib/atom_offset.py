"""
lib/atom_offset.py - Das Atom gegen das Multitone-Profil verschieben
====================================================================

Fester Parametersatz (waist, width, r_x, r_y; 3x4 Toene, f1 = 75 mm,
f2 = 750 mm), und das ATOM wandert radial nach aussen - horizontal,
vertikal, diagonal (optional antidiagonal). Ausgewertet wird wie im
Penalty-Fall: harte und atom-gewichtete Metriken und ihre Kombination

    U_c   = 0.5*(U_h + U_w) + combo_lambda*|U_h - U_w|
    eta_c = 0.5*(C_h + C_w) + combo_lambda*|C_h - C_w|
    J     = alpha*U_c + (1-alpha)*eta_c

(Formel unveraendert aus lib/combine.py).

Was sich bewegt und was steht
-----------------------------
BEWEGT wird das Atom - und mit ihm ALLE Regionen, ueber die ausgewertet
wird:

- die harte Uniformity-Region: das Ton-Quadrat (Seitenlaenge = Spannweite
  des Tonarrays, dieselbe Groesse wie _resolve_uniformity_side() im
  Optimierer), zentriert auf dem Atom;
- die harte Crosstalk-Region: das Pitch-Quadrat, zentriert auf dem Atom;
- das lokale Sub-Gitter und die Gauss-Gewichtung W der gewichteten
  Metriken (das ist atom_offset_x/y des Optimierers).

STEHEN bleibt das Lichtfeld: die 12 Spots der eigenen Site und die 8
Nachbarkopien im Abstand pitch. Das ist exakt dasselbe wie eine
gemeinsame Drift des ganzen Raman-Profils um -r gegen ein ruhendes Atom.

Warum die harten Metriken hier NICHT ueber das globale Gitter laufen
--------------------------------------------------------------------
Der Optimierer rechnet U_h und eta_h auf einem globalen Gitter mit
n_grid x n_grid Punkten und schneidet die Regionen als Masken heraus. Wird
die Maske um Bruchteile eines Pixels verschoben, springt ihr Rand um ganze
Pixel - genau das Saegezahn-Rauschen, das in claude/combinated_optimization.md
(Nachtrag 6) gemessen ist, und bei Verschiebungen von einigen 10 nm waere es
groesser als der Effekt. Deshalb liegt hier ein eigenes Gitter AUF der
Region: Zellmitten (Mittelpunktsregel) eines Quadrats um das Atom. Es
wandert mit, die Abtastung aendert sich also stetig mit r.

Bei r = 0 stimmt das mit dem Optimierer ueberein (siehe
reference_at_zero(); gemessen bei waist 1.1 µm / width 0.45 MHz:
U_h 5.871 % gegen 5.869 %, eta_h 9.170 % gegen 9.190 % - der Rest ist der
Pixelrand des globalen Gitters).

Symmetrie
---------
Horizontal und vertikal sind spiegelsymmetrisch (+r und -r geben dasselbe).
Die beiden Diagonalen sind es NICHT ganz: mit Kohaerenz interferieren die
frequenzentarteten Eckspots (0, w) und (w, 0) statisch, das Muster ist also
gegen die Antidiagonale anders als gegen die Diagonale. Fuer die harten
Groessen ist der Unterschied klein, fuer U_w merklich (bei 1.1 µm /
0.45 MHz und r = 0.355 µm: U_w 2.41 % diagonal gegen 2.98 % antidiagonal).
symmetry_check() rechnet ihn am Ende des Bereichs aus, er steht im Bericht,
und die Antidiagonale laesst sich als vierte Richtung dazunehmen.
"""

import pickle
import time
from datetime import date
from pathlib import Path as FilePath

import numpy as np

from . import paths
from .combine import penalty_pair, penalty_objective

from weighted_multitone_flattop_optimizer import (  # noqa: E402
    MultitoneFlatTopOptimizer, amps_from_ratios,
)
from airy_scale import AIRY_SCALE_DIALOG_DEFAULT, scale_tag  # noqa: E402


# ======================================================================
# Feste Geometrie und Defaults
# ======================================================================
N_X_FIXED = 3
N_Y_FIXED = 4
F1_FIXED = 75e-3
F2_FIXED = 750e-3

DEFAULTS = dict(
    # Arbeitspunkt (Standard-Arbeitspunkt des Projekts)
    waist=1.1e-6,            # m, Waist in der Atomebene
    width=0.45e6,            # Hz
    r_x=1.0,
    r_y=1.2,
    # Profil
    profile="airy",
    airy_scale_factor=AIRY_SCALE_DIALOG_DEFAULT,
    coherent=True,
    # Atom (gewichtete Metriken)
    atom_temperature=17e-6,  # K
    trap_freq_r=60.4e3,      # Hz
    weighted_n_grid=241,
    # harte Metriken: rechnen ja/nein (Basisfall sind die atom-gewichteten;
    # die kombinierten Groessen brauchen die harten und schalten sie ein)
    # und Zellen je Achse auf der Region
    hart=False,
    hard_n_grid=201,
    # Penalty
    alpha=0.7,
    combo_lambda=0.75,
)

# (id, Klartext, Einheitsvektor). Die id steckt in den Kurvenschluesseln
# ("uniformity_hart__x"); sie ist bewusst NICHT "vertikal"/"diagonal", weil
# single_beam_report.py diese Namen fuer den Einzelstrahl schon belegt.
RICHTUNGEN = [
    ("x", "horizontal", (1.0, 0.0)),
    ("y", "vertikal", (0.0, 1.0)),
    ("xy", "diagonal", (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0))),
    ("antixy", "antidiagonal", (-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0))),
]
RICHTUNG_TEXT = {rid: text for rid, text, _v in RICHTUNGEN}
RICHTUNG_TEXT["mittel"] = "Mittel ueber die Richtung"
RICHTUNG_VEKTOR = {rid: v for rid, _t, v in RICHTUNGEN}
RICHTUNGEN_DEFAULT = ("x", "y", "xy")

# Richtungsmittel: so viele Winkel im HALBkreis [0, pi). Der halbe Kreis
# genuegt, weil die Anordnung punktsymmetrisch ist (M(-r) = M(r), gemessen:
# Abweichung 3e-6 pp bei r = 0.355 µm). Spiegelsymmetrisch ist sie mit
# Kohaerenz NICHT (0.25 pp), deshalb kein Viertelkreis. Die Funktion ist
# periodisch und glatt, das gleichgewichtete Mittel konvergiert spektral:
# 6 Winkel weichen von 16 um < 0.002 pp ab (U_w), alle anderen < 0.001 pp.
RICHTUNGSMODI = [
    ("mittel", "ueber alle Richtungen gemittelt (eine Kurve je Groesse)"),
    ("einzeln", "einzelne Richtungen (eine Kurve je Richtung)"),
]
N_WINKEL_DEFAULT = 6

WIDTH_ANTEIL_DEFAULT = 0.125
STUETZSTELLEN_DEFAULT = 41

METRIC_KEYS = ("uniformity_hart", "crosstalk_hart",
               "uniformity_weighted", "crosstalk_weighted",
               "uniformity_kombi", "crosstalk_kombi", "combined_score")

KIND = "multitone_atom_offset"


# ======================================================================
# Optimierer und Geometrie
# ======================================================================
def merged(params=None):
    p = dict(DEFAULTS)
    p.update(params or {})
    return p


def make_optimizer(params=None):
    """MultitoneFlatTopOptimizer mit fester 3x4-Geometrie und 75/750 mm.

    n_grid wird klein gehalten: das globale Gitter braucht hier nur
    reference_at_zero(), und die baut sich ihr eigenes."""
    p = merged(params)
    return MultitoneFlatTopOptimizer(
        out_dir=str(paths.DEFAULT_IMAGES_DIR),
        f1=F1_FIXED, f2=F2_FIXED, N_x=N_X_FIXED, N_y=N_Y_FIXED,
        profile=p["profile"], airy_scale_factor=p["airy_scale_factor"],
        coherent=bool(p["coherent"]),
        atom_temperature=p["atom_temperature"], trap_freq_r=p["trap_freq_r"],
        weighted_n_grid=int(p["weighted_n_grid"]),
        win=p["waist"], width=p["width"], n_grid=200,
    )


def width_um(params=None, opt=None):
    """width in der Atomebene (µm) = Spannweite des Tonarrays, aeusserster
    bis aeusserster Ton. In x und y gleich (multitone_frequencies laeuft in
    beiden Achsen von offset bis offset + width)."""
    p = merged(params)
    opt = make_optimizer(p) if opt is None else opt
    return float(opt.width_to_um(p["width"]))


def _amps(p):
    return amps_from_ratios(p["r_x"], p["r_y"], N_X_FIXED, N_Y_FIXED)


def _zellmitten(cx, cy, seite, n):
    """Zellmitten eines n x n-Gitters auf dem Quadrat der Seitenlaenge
    `seite` um (cx, cy) - Mittelpunktsregel, jede Zelle gleich gross."""
    s = (np.arange(n) + 0.5) / n * seite - 0.5 * seite
    return np.meshgrid(cx + s, cy + s)


class _Geometrie:
    """Alles, was fuer einen Parametersatz nur einmal gerechnet wird."""

    def __init__(self, opt, p):
        self.opt = opt
        self.p = p
        self.win = float(p["waist"])
        self.width = float(p["width"])
        self.amps = _amps(p)
        self.cx, self.cy, self.rc = opt._compute_centers_for_width(self.width)
        self.pf = opt._profile_func()
        self.scale = opt._profile_scale(self.win)
        a_x = self.amps[:N_X_FIXED]
        a_y = self.amps[N_X_FIXED:]
        self.amp_spots = np.repeat(a_x, N_Y_FIXED) * np.tile(a_y, N_X_FIXED)
        self.seite_uni = float(opt._resolve_uniformity_side(self.cx, self.cy, self.rc))
        self.pitch = float(opt.pitch)
        self.n = int(p["hard_n_grid"])
        self.hart = bool(p.get("hart", True))


# ======================================================================
# Metriken an EINER Atomposition
# ======================================================================
def metrics_at(geo, dx, dy):
    """Alle sieben Groessen bei Atomversatz (dx, dy) in Metern.

    Rueckgabe: dict mit den METRIC_KEYS (Anteile, nicht Prozent). Ist
    geo.hart aus, werden die harten Metriken (und damit U_c, eta_c, J)
    nicht gerechnet und stehen auf NaN - das spart gut die Haelfte der Zeit."""
    opt = geo.opt
    ax_ = geo.rc + dx
    ay_ = geo.rc + dy

    if not geo.hart:
        opt.atom_offset_x = float(dx)
        opt.atom_offset_y = float(dy)
        try:
            w = opt._evaluate_weighted_only(geo.win, geo.width, amps=geo.amps)
        finally:
            opt.atom_offset_x = 0.0
            opt.atom_offset_y = 0.0
        aus = {k: np.nan for k in METRIC_KEYS}
        aus["uniformity_weighted"] = float(w["uniformity_weighted"]) if w else np.nan
        aus["crosstalk_weighted"] = float(w["eta_weighted"]) if w else np.nan
        return aus

    # --- hart: Uniformity ueber dem Ton-Quadrat um das Atom ---
    X, Y = _zellmitten(ax_, ay_, geo.seite_uni, geo.n)
    I = geo.pf(X, Y, geo.cx, geo.cy, geo.scale, geo.amp_spots)
    mittel = float(np.mean(I))
    U_h = float(np.std(I) / mittel) if mittel != 0 else np.nan

    # --- hart: Crosstalk ueber dem Pitch-Quadrat um das Atom ---
    X, Y = _zellmitten(ax_, ay_, geo.pitch, geo.n)
    I_eigen = geo.pf(X, Y, geo.cx, geo.cy, geo.scale, geo.amp_spots)
    I_nachbar = opt._local_neighbor_intensity(
        X, Y, geo.cx, geo.cy, geo.scale, geo.amp_spots, geo.pf)
    summe = float(np.sum(I_eigen))
    C_h = float(np.sum(I_nachbar) / summe) if summe != 0 else np.nan

    # --- atom-gewichtet: Optimierer mit atom_offset_x/y ---
    opt.atom_offset_x = float(dx)
    opt.atom_offset_y = float(dy)
    try:
        w = opt._evaluate_weighted_only(geo.win, geo.width, amps=geo.amps)
    finally:
        opt.atom_offset_x = 0.0
        opt.atom_offset_y = 0.0
    U_w = float(w["uniformity_weighted"]) if w else np.nan
    C_w = float(w["eta_weighted"]) if w else np.nan

    lam = float(geo.p["combo_lambda"])
    return dict(
        uniformity_hart=U_h, crosstalk_hart=C_h,
        uniformity_weighted=U_w, crosstalk_weighted=C_w,
        uniformity_kombi=float(penalty_pair(U_h, U_w, lam)),
        crosstalk_kombi=float(penalty_pair(C_h, C_w, lam)),
        combined_score=float(penalty_objective(U_h, C_h, U_w, C_w,
                                               float(geo.p["alpha"]), lam)),
    )


def reference_at_zero(geo):
    """Dieselben vier Rohgroessen bei r = 0 ueber den Optimierer selbst
    (globales Gitter, Masken) - als Abgleich im Bericht."""
    opt = geo.opt
    n_vorher = opt.n_grid
    opt.n_grid = 1000
    try:
        grid = opt._build_dynamic_grid(geo.win, geo.width)
        r = opt._evaluate(geo.win, geo.width, amps=geo.amps, grid=grid, weighted=True)
    finally:
        opt.n_grid = n_vorher
    if r is None:
        return None
    return dict(uniformity_hart=float(r["uniformity"]), crosstalk_hart=float(r["eta"]),
                uniformity_weighted=float(r["uniformity_weighted"]),
                crosstalk_weighted=float(r["eta_weighted"]), n_grid=1000)


# ======================================================================
# Sweep ueber den Versatz
# ======================================================================
def sweep(params, r_max, n=STUETZSTELLEN_DEFAULT, richtungen=RICHTUNGEN_DEFAULT,
          progress=None, bereich=None, geo=None, modus="einzeln",
          n_winkel=N_WINKEL_DEFAULT):
    """Metriken ueber dem Versatz r = 0 .. r_max.

    modus="einzeln": je Richtung aus `richtungen` eine Kurve
        ("uniformity_hart__x", ...).
    modus="mittel": EINE Kurve je Groesse, gemittelt ueber die Richtung des
        Versatzes: n_winkel Winkel gleichmaessig im Halbkreis [0, pi), an
        jedem die volle Auswertung, dann das Mittel ("uniformity_hart__mittel").
        Dazu Minimum und Maximum ueber die Winkel ("__min", "__max") und alle
        Einzelwerte ("__winkel", Form n x n_winkel). Die kombinierten
        Groessen werden je Winkel gebildet und DANN gemittelt - also das
        Mittel von J, nicht J der Mittelwerte.

    bereich: dict, das beschreibt, woher r_max kommt - wird nur
        mitgespeichert (Bericht, Achsenbeschriftung). Erwartete Schluessel:
        art ("width" | "schwankung" | "frei"), faktor, bezug_m, text.
    progress(i, gesamt) -> False bricht ab.
    """
    p = merged(params)
    if modus == "mittel":
        n_winkel = int(n_winkel)
        if n_winkel < 1:
            raise ValueError("Mindestens ein Winkel noetig.")
        winkel = np.arange(n_winkel) * np.pi / n_winkel
        richtungen = ("mittel",)
    elif modus == "einzeln":
        winkel = None
        richtungen = tuple(richtungen)
    else:
        raise ValueError(f"Unbekannter modus {modus!r}.")
    if not richtungen:
        raise ValueError("Mindestens eine Richtung waehlen.")
    n = int(n)
    if n < 2:
        raise ValueError("Mindestens zwei Stuetzstellen noetig.")
    r_max = float(r_max)
    if not (np.isfinite(r_max) and r_max > 0):
        raise ValueError(f"r_max muss > 0 sein (ist {r_max}).")

    if geo is None:
        geo = _Geometrie(make_optimizer(p), p)
    r_vals = np.linspace(0.0, r_max, n)
    d_um = float(geo.opt.width_to_um(geo.width))

    results = dict(
        kind=KIND,
        params=p,
        N_x=N_X_FIXED, N_y=N_Y_FIXED, f1=F1_FIXED, f2=F2_FIXED,
        fLO=geo.opt.fLO, pitch=geo.pitch, lambda_opt=geo.opt.lambda_opt,
        waist_um=geo.win * 1e6,
        width_MHz=geo.width * 1e-6,
        width_um=d_um,
        uniformity_side_um=geo.seite_uni * 1e6,
        sigma_atom=float(geo.opt.sigma_atom),
        n_degenerate_pairs=int(geo.opt.n_degenerate_pairs()),
        offset=r_vals, offset_um=r_vals * 1e6,
        offset_in_width=r_vals * 1e6 / d_um if d_um > 0 else np.full(n, np.nan),
        richtungen=richtungen,
        modus=modus,
        winkel_rad=None if winkel is None else winkel,
        bereich=dict(bereich or dict(art="frei", faktor=None, bezug_m=None,
                                     text=f"bis {r_max * 1e6:.4g} µm")),
        abgebrochen=False,
        datum=date.today().isoformat(),
    )

    gesamt = 1 + (n - 1) * (len(winkel) if modus == "mittel" else len(richtungen))
    getan = 0
    t0 = time.perf_counter()

    null = metrics_at(geo, 0.0, 0.0)
    getan += 1
    if progress is not None and progress(getan, gesamt) is False:
        results["abgebrochen"] = True

    keys = []
    if modus == "mittel":
        einzel = {k: np.full((n, len(winkel)), np.nan) for k in METRIC_KEYS}
        for k in METRIC_KEYS:
            einzel[k][0, :] = null[k]
        for i in range(1, n):
            if results["abgebrochen"]:
                break
            for j, phi in enumerate(winkel):
                m = metrics_at(geo, np.cos(phi) * r_vals[i], np.sin(phi) * r_vals[i])
                for k in METRIC_KEYS:
                    einzel[k][i, j] = m[k]
                getan += 1
                if progress is not None and progress(getan, gesamt) is False:
                    results["abgebrochen"] = True
                    break
        for k in METRIC_KEYS:
            zeilen = einzel[k]
            voll = np.all(np.isfinite(zeilen), axis=1)
            mittel = np.full(n, np.nan)
            mittel[voll] = np.mean(zeilen[voll], axis=1)
            unten = np.full(n, np.nan)
            oben = np.full(n, np.nan)
            unten[voll] = np.min(zeilen[voll], axis=1)
            oben[voll] = np.max(zeilen[voll], axis=1)
            results[f"{k}__mittel"] = mittel
            results[f"{k}__min"] = unten
            results[f"{k}__max"] = oben
            results[f"{k}__winkel"] = zeilen
            keys.append(f"{k}__mittel")
        results["kurven"] = tuple(keys)
        results["dauer_s"] = time.perf_counter() - t0
        return results

    for rid in richtungen:
        ex, ey = RICHTUNG_VEKTOR[rid]
        werte = {k: np.full(n, np.nan) for k in METRIC_KEYS}
        for k in METRIC_KEYS:
            werte[k][0] = null[k]
        for i in range(1, n):
            if results["abgebrochen"]:
                break
            m = metrics_at(geo, ex * r_vals[i], ey * r_vals[i])
            for k in METRIC_KEYS:
                werte[k][i] = m[k]
            getan += 1
            if progress is not None and progress(getan, gesamt) is False:
                results["abgebrochen"] = True
        for k in METRIC_KEYS:
            key = f"{k}__{rid}"
            results[key] = werte[k]
            keys.append(key)

    results["kurven"] = tuple(keys)
    results["dauer_s"] = time.perf_counter() - t0
    return results


def symmetry_check(geo, r):
    """Werte bei +r und -r je Achse sowie auf beiden Diagonalen - fuer den
    Bericht. Rueckgabe: {Name: metrics_at-dict}."""
    s = r / np.sqrt(2.0)
    punkte = {
        "+x": (r, 0.0), "-x": (-r, 0.0),
        "+y": (0.0, r), "-y": (0.0, -r),
        "diagonal (+,+)": (s, s), "diagonal (-,-)": (-s, -s),
        "antidiagonal (-,+)": (-s, s), "antidiagonal (+,-)": (s, -s),
    }
    return {name: metrics_at(geo, dx, dy) for name, (dx, dy) in punkte.items()}


# ======================================================================
# Statistik ueber die 2D-Positionsverteilung
# ======================================================================
GH_ORDNUNG_DEFAULT = 7


def statistik(geo, sigma, ordnung=GH_ORDNUNG_DEFAULT, progress=None):
    """Mittelwert und Streuung jeder Groesse, wenn das Atom isotrop
    gaussverteilt mit `sigma` je Achse um die Site-Mitte sitzt.

    Gauss-Hermite-Quadratur mit `ordnung` Knoten je Achse (7 -> 49
    Auswertungen). Die Metriken sind glatt in (dx, dy), die Quadratur ist
    fuer Mittelwert und Varianz damit sehr genau, ohne dass gewuerfelt
    werden muss (reproduzierbar).
    """
    t, w = np.polynomial.hermite.hermgauss(int(ordnung))
    x = np.sqrt(2.0) * float(sigma) * t
    gew = w / np.sqrt(np.pi)
    werte = {k: [] for k in METRIC_KEYS}
    gewichte = []
    gesamt = x.size ** 2
    i = 0
    for a, wa in zip(x, gew):
        for b, wb in zip(x, gew):
            m = metrics_at(geo, a, b)
            for k in METRIC_KEYS:
                werte[k].append(m[k])
            gewichte.append(wa * wb)
            i += 1
            if progress is not None and progress(i, gesamt) is False:
                return None
    gewichte = np.asarray(gewichte)
    null = metrics_at(geo, 0.0, 0.0)
    aus = dict(sigma=float(sigma), ordnung=int(ordnung), n_auswertungen=int(gesamt))
    for k in METRIC_KEYS:
        v = np.asarray(werte[k], dtype=float)
        mittel = float(np.sum(gewichte * v))
        streu = float(np.sqrt(max(np.sum(gewichte * (v - mittel) ** 2), 0.0)))
        aus[k] = dict(null=float(null[k]), mittel=mittel, streuung=streu,
                      min_knoten=float(np.min(v)), max_knoten=float(np.max(v)))
    return aus


# ======================================================================
# Werte an bestimmten Abstaenden (Bericht)
# ======================================================================
def werte_bei(results, r_m):
    """{Kurvenschluessel: Wert} bei Versatz r_m, linear interpoliert. Liegt
    r_m ausserhalb des gerechneten Bereichs, ist der Wert NaN."""
    r = np.asarray(results["offset"], dtype=float)
    aus = {}
    for key in results["kurven"]:
        v = np.asarray(results[key], dtype=float)
        if r_m < r[0] - 1e-15 or r_m > r[-1] + 1e-15:
            aus[key] = np.nan
        else:
            aus[key] = float(np.interp(r_m, r, v))
    return aus


# ======================================================================
# Dateinamen, Speichern, Laden
# ======================================================================
def name_kern(p):
    """N3x4_Airy_w1.1um_width0.45MHz_rx1_ry1.2_k1.483 (+ _inkohaerent)."""
    profil = paths.profile_tag_of(p["profile"])
    k = scale_tag(p["airy_scale_factor"]) if p["profile"] == "airy" else ""
    koh = "" if p.get("coherent", True) else "_inkohaerent"
    return (f"N{N_X_FIXED}x{N_Y_FIXED}_{profil}_w{p['waist'] * 1e6:.4g}um"
            f"_width{p['width'] * 1e-6:.4g}MHz_rx{p['r_x']:.4g}_ry{p['r_y']:.4g}{k}{koh}")


def pkl_name(p):
    return f"atom_offset_{name_kern(merged(p))}.pkl"


def save_results(lauf, filepath):
    filepath = FilePath(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as fh:
        pickle.dump(lauf, fh)
    return filepath


def load_results(filepath):
    with open(FilePath(filepath), "rb") as fh:
        lauf = pickle.load(fh)
    if lauf.get("kind") != KIND + "_lauf":
        raise ValueError(f"{filepath} ist kein Atom-Versatz-Datensatz "
                         f"(kind={lauf.get('kind')!r}).")
    return lauf


# ======================================================================
# Ein ganzer Lauf (was run_atom_offset.py ausfuehrt)
# ======================================================================
def lauf_rechnen(params, abschaetzung=None, umfang="gesamt", k_faktor=1.0,
                 plot_width=True, width_anteil=WIDTH_ANTEIL_DEFAULT,
                 plot_schwankung=False, n=STUETZSTELLEN_DEFAULT,
                 richtungen=RICHTUNGEN_DEFAULT, statistik_an=False,
                 gh_ordnung=GH_ORDNUNG_DEFAULT, abgleich=True, kombiniert=False,
                 modus="mittel", n_winkel=N_WINKEL_DEFAULT, progress=None):
    """Rechnet alles und gibt das Lauf-dict zurueck (ohne Plots/Bericht).

    abschaetzung: position_noise.Abschaetzung, oder None - dann gibt es
        weder den Sweep bis zur Positionsschwankung noch die Statistik.
    kombiniert: wird nur mitgespeichert (Plots/Bericht). U_c, eta_c und J
        rechnet metrics_at() ohnehin mit, sie kosten nichts.
    progress(text, i, gesamt) -> False bricht ab.
    """
    p = merged(params)
    # Die kombinierten Groessen bestehen aus hart UND gewichtet.
    p["hart"] = bool(p.get("hart", False) or kombiniert)
    opt = make_optimizer(p)
    geo = _Geometrie(opt, p)
    d_m = float(opt.width_to_um(geo.width)) * 1e-6
    if abschaetzung is None:
        plot_schwankung = False
        statistik_an = False
        sigma_pos = 0.0
    else:
        sigma_pos = float(abschaetzung.sigma(umfang))

    lauf = dict(
        kind=KIND + "_lauf", params=p, datum=date.today().isoformat(),
        width_um=d_m * 1e6, uniformity_side_um=geo.seite_uni * 1e6,
        fLO=opt.fLO, pitch=geo.pitch, sigma_atom=float(opt.sigma_atom),
        n_degenerate_pairs=int(opt.n_degenerate_pairs()),
        abschaetzung=None if abschaetzung is None else abschaetzung.als_dict(),
        umfang=umfang, k_faktor=float(k_faktor), sigma_pos_m=sigma_pos,
        kombiniert=bool(kombiniert), hart=p["hart"], modus=modus, n_winkel=int(n_winkel),
        abgleich=bool(abgleich),
        sweeps=[], statistik=None, symmetrie=None, referenz=None,
        abgebrochen=False,
    )

    def melder(text):
        if progress is None:
            return None

        def f(i, gesamt):
            ok = progress(text, i, gesamt)
            if ok is False:
                lauf["abgebrochen"] = True
            return ok
        return f

    lauf["null"] = metrics_at(geo, 0.0, 0.0)
    # Der Abgleich prueft das mitwandernde Gitter der HARTEN Metriken. Die
    # gewichteten kommen unveraendert aus dem Optimierer - ohne harte gibt es
    # nichts abzugleichen.
    if abgleich and p["hart"]:
        if progress is not None:
            progress("Abgleich mit dem Optimierer bei r = 0 ...", 0, 1)
        lauf["referenz"] = reference_at_zero(geo)

    if plot_width and not lauf["abgebrochen"]:
        r_max = float(width_anteil) * d_m
        lauf["sweeps"].append(sweep(
            p, r_max, n=n, richtungen=richtungen, geo=geo, modus=modus, n_winkel=n_winkel,
            progress=melder(f"Versatz bis {width_anteil:g} d = {r_max * 1e9:.0f} nm ..."),
            bereich=dict(art="width", faktor=float(width_anteil), bezug_m=d_m,
                         text=f"bis {width_anteil:g} x width ({r_max * 1e9:.1f} nm)")))

    if plot_schwankung and sigma_pos > 0 and not lauf["abgebrochen"]:
        r_max = float(k_faktor) * sigma_pos
        lauf["sweeps"].append(sweep(
            p, r_max, n=n, richtungen=richtungen, geo=geo, modus=modus, n_winkel=n_winkel,
            progress=melder(f"Versatz bis {k_faktor:g} sigma_pos = {r_max * 1e9:.0f} nm ..."),
            bereich=dict(art="schwankung", faktor=float(k_faktor), bezug_m=sigma_pos,
                         text=f"bis {k_faktor:g} x Positionsschwankung ({r_max * 1e9:.1f} nm)")))

    if statistik_an and sigma_pos > 0 and not lauf["abgebrochen"]:
        lauf["statistik"] = statistik(geo, sigma_pos, ordnung=gh_ordnung,
                                      progress=melder("Statistik ueber die Positionsverteilung ..."))

    if not lauf["abgebrochen"]:
        r_sym = max([float(np.max(s["offset"])) for s in lauf["sweeps"]] or [0.125 * d_m])
        lauf["symmetrie_r_m"] = r_sym
        if progress is not None:
            progress("Symmetrie-Pruefung ...", 0, 1)
        lauf["symmetrie"] = symmetry_check(geo, r_sym)
    return lauf
