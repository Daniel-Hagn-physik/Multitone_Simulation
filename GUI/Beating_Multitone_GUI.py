"""
Beating Multitone GUI (PyQt5)
=============================
Zeitaufgeloeste, KOHAERENTE Simulation des Multitone-FlatTop-Musters.

Warum ein eigenes GUI?
----------------------
Multitone_Lens_GUI.py und Weighted_Multitone_Lens_GUI.py summieren die
INTENSITAETEN der einzelnen Toene:

    I(x,y) = sum_s a_s * |u(r - c_s)|^2

Das ist das Zeitmittel. Physikalisch ueberlagern sich die Toene aber als
FELDER, und weil jeder Ton eine andere AOD-Frequenz traegt, laeuft die
relative Phase zwischen zwei Toenen mit ihrer Differenzfrequenz um:

    E(x,y,t) = sum_s A_s * u(r - c_s) * exp(i*2pi*f_s*t + i*phi_s)
    I(x,y,t) = |E(x,y,t)|^2
             = sum_s A_s^2 |u_s|^2                          <- das Zeitmittel
             + 2 * sum_{s<s'} A_s A_s' u_s u_s'
                     * cos(2pi (f_s - f_s') t + phi_s - phi_s')   <- das Beating

Der zweite Term verschwindet im Mittel ueber eine Grundperiode - genau
deshalb stimmt das bisherige inkohaerente Bild als Zeitmittel, verschweigt
aber die Momentanwerte. Sichtbar wird das Beating dort, wo sich zwei Spots
raeumlich ueberlappen (u_s * u_s' != 0), also zwischen den Fallen.

Frequenzen
----------
Die Toene liegen wie im AWG bei

    f_x(n) = offset + width * n/(N_x-1),   n = 0 .. N_x-1
    f_y(m) = offset + width * m/(N_y-1),   m = 0 .. N_y-1

Beide AODs schieben die Lichtfrequenz, ein Spot (n,m) traegt also
f_s = f_x(n) + f_y(m). In |E|^2 taucht nur die DIFFERENZ zweier
Spot-Frequenzen auf:

    f_s - f_s' = width * ( dn/(N_x-1) + dm/(N_y-1) )

Alle diese Differenzen sind ganzzahlige Vielfache von

    f_beat_0 = width / kgV(N_x-1, N_y-1)

fuer 3x4 also width/6. Bei width = 0.35 MHz sind das 58.3 kHz, Grundperiode
T_0 = 17.1 us. Die absolute Lichtfrequenz und der Offset kuerzen sich
komplett heraus - Wellenlaenge und Offset aendern die Geometrie (Waist-
Umrechnung bzw. Strahlablenkung), aber keine einzige Beat-Frequenz.

Feldamplituden
--------------
Die uebrigen Skripte fuehren `amps` als INTENSITAETS-Gewichte
(I += a * profil). Das Feld traegt daher sqrt(a). Die Amplituden kommen aus
der bekannten Aussen/Innen-Parametrisierung (amps_from_ratio):

    amp_x = [r_x, 1, ..., 1, r_x],   amp_y = [r_y, 1, ..., 1, r_y]
    a_spot(n,m) = amp_x[n] * amp_y[m]

Feldprofil eines Spots (auf 1 normiert im Zentrum):
    Gauss:  u(r) = exp(-r^2 / w^2)          -> |u|^2 = exp(-2r^2/w^2)
    Airy:   u(r) = 2*J1(k r)/(k r)          -> |u|^2 = (2*J1/u)^2
            k = 3.83170597 / (Skalenfaktor * waist)
Beim Airy-Profil ist u in den Ringen NEGATIV - dieses Vorzeichen ist fuer
die kohaerente Summe wesentlich und wird hier bewusst mitgenommen.

Bedienung
---------
Alle Parameter werden eingetippt (keine Slider ausser der Zeitachse).
Voreingestellt ist der Arbeitspunkt: 3x4 Toene, AIRY-Profil, waist 1.1 um,
width 0.45 MHz, r_x = 1.0, r_y = 1.2, 795 nm, 100 MHz Offset,
f1 = 60 mm, f2 = 750 mm.

Start:
    python Beating_Multitone_GUI.py
"""

import sys
import math
import datetime
from functools import reduce
from pathlib import Path as FilePath

import numpy as np
from scipy.special import j1

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QGroupBox, QScrollArea, QSplitter, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer

try:
    import airy_scale
    AIRY_SCALE_DEFAULT = airy_scale.AIRY_SCALE_DIALOG_DEFAULT
except Exception:      # airy_scale.py liegt normalerweise daneben
    airy_scale = None
    AIRY_SCALE_DEFAULT = 1.4830


# ============================================================
# Feste Optik-Konstanten (identisch zu den anderen GUIs)
# offset und lambda sind hier EINGABEFELDER, keine Konstanten mehr.
# ============================================================
fLO = 52.88e-3          # m   Fokussierlinse
theta_max = 43e-3       # rad maximaler Ablenkwinkel
f_band = 36e6           # Hz  AOD-Bandbreite
pitch = 5.288e-6        # m   physikalischer Atomabstand (nur informativ)

OUT_DIR_CANDIDATES = [
    FilePath(r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\PythonCode\Multitone_FlatTop"),
]


def _resolve_out_dir():
    for cand in OUT_DIR_CANDIDATES:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except Exception:
            continue
    fallback = FilePath.cwd() / "FlatMultiTone_Output"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ============================================================
# Physik / Numerik
# ============================================================
def multitone_frequencies(N, offset, width):
    """Diskrete AWG-Frequenzen: offset + width * n/(N-1).

    Fuer N == 1 wird der einzelne Ton auf die MITTE des Bereichs gelegt
    (offset + width/2), damit er mit r_center zusammenfaellt - genau wie in
    Multitone_Lens_GUI.multitone_frequencies()."""
    if N <= 1:
        return np.array([offset + width / 2.0], dtype=float)
    return width * np.arange(N) / (N - 1) + offset


def angle_from_frequency(f, offset, theta_max_, f_band_):
    return theta_max_ * (f - offset) / f_band_


def radius_from_angle(theta, f1, f2, fLO_):
    return (f1 * fLO_ / f2) * np.tan(theta)


def conjugate_waist(w, f1, f2, lam):
    """Eingangs- <-> Ausgangswaist durch das Teleskop f1->f2 plus fLO.
        w_out = (f1/f2) * (lam * fLO) / (pi * w_in)
    Die Beziehung ist symmetrisch, dieselbe Funktion rechnet beide
    Richtungen."""
    if w <= 0:
        return float("nan")
    return (f1 / f2) * (lam * fLO) / np.pi / w


def compute_centers_and_freqs(N_x, N_y, width_x, width_y, f1, f2, offset):
    """Spot-Zentren UND die Frequenz jedes Spots.

    Reihenfolge der Spots ist identisch zu compute_centers() der anderen
    GUIs (fx aussen, fy innen), damit amp_spots = repeat(amp_x, N_y) *
    tile(amp_y, N_x) unveraendert passt.

    x und y bekommen hier eine EIGENE width. Die uebrigen GUIs setzen beide
    gleich; genau das erzeugt aber Frequenzentartungen (siehe
    degenerate_groups()), und getrennte widths sind der einzige Weg, sie
    aufzuheben - ein konstanter Frequenzversatz einer Achse hilft NICHT, weil
    er alle Spot-Frequenzen um denselben Betrag verschiebt und damit jede
    Differenz unveraendert laesst."""
    fx_freq = multitone_frequencies(N_x, offset, width_x)
    fy_freq = multitone_frequencies(N_y, offset, width_y)
    f_center_x = offset + width_x / 2.0
    f_center_y = offset + width_y / 2.0
    r_center_x = radius_from_angle(
        angle_from_frequency(f_center_x, offset, theta_max, f_band), f1, f2, fLO)
    r_center_y = radius_from_angle(
        angle_from_frequency(f_center_y, offset, theta_max, f_band), f1, f2, fLO)

    centers_x, centers_y, f_spots = [], [], []
    for fx in fx_freq:
        rx = radius_from_angle(angle_from_frequency(fx, offset, theta_max, f_band), f1, f2, fLO)
        for fy in fy_freq:
            ry = radius_from_angle(angle_from_frequency(fy, offset, theta_max, f_band), f1, f2, fLO)
            centers_x.append(rx)
            centers_y.append(ry)
            # Beide AODs schieben die Lichtfrequenz -> der Spot traegt die Summe.
            f_spots.append(fx + fy)
    return (np.array(centers_x), np.array(centers_y), np.array(f_spots),
            r_center_x, r_center_y, fx_freq, fy_freq)


def amps_from_ratio(r, N):
    """Aussen/Innen-Parametrisierung, buchstabengleich zu
    Amplitudes/.../multitone_flattop_optimizer.amps_from_ratio():
    die beiden aeussersten Toene bekommen r, alle inneren bleiben 1."""
    amp = np.ones(N, dtype=float)
    if N >= 2:
        amp[0] = r
        amp[-1] = r
    elif N == 1:
        amp[0] = r
    return amp


def amp_spots_from_ratios(r_x, r_y, N_x, N_y):
    """Intensitaets-Gewicht je Spot, Reihenfolge wie compute_centers_and_freqs()."""
    amp_x = amps_from_ratio(r_x, N_x)
    amp_y = amps_from_ratio(r_y, N_y)
    return np.repeat(amp_x, N_y) * np.tile(amp_y, N_x)


def spot_field(X, Y, cx, cy, width_param, use_airy, airy_factor):
    """FELD (nicht Intensitaet) eines einzelnen Spots, im Zentrum auf 1
    normiert. Beim Airy-Profil bleibt das Vorzeichen der Ringe erhalten -
    fuer die kohaerente Summe ist es wesentlich."""
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    if not use_airy:
        return np.exp(-r2 / width_param ** 2)
    first_zero_radius = airy_factor * width_param
    k = 3.83170597 / first_zero_radius
    u = k * np.sqrt(r2)
    out = np.ones_like(u)
    m = u > 1e-12
    out[m] = 2.0 * j1(u[m]) / u[m]
    return out


def build_field_stack(X, Y, centers_x, centers_y, amp_spots, width_param,
                      use_airy, airy_factor):
    """Stapel A_s * u_s(x,y) fuer alle Spots. A_s = sqrt(Intensitaets-Gewicht),
    weil `amps` im uebrigen Projekt Intensitaeten gewichtet."""
    S = len(centers_x)
    F = np.empty((S,) + X.shape, dtype=np.float64)
    A = np.sqrt(np.clip(amp_spots, 0.0, None))
    for s in range(S):
        F[s] = A[s] * spot_field(X, Y, centers_x[s], centers_y[s],
                                 width_param, use_airy, airy_factor)
    return F


def unique_beat_frequencies(f_spots, rtol=1e-10):
    """Alle vorkommenden positiven Differenzfrequenzen, aufsteigend, ohne
    Dubletten.

    Die Spot-Frequenzen liegen bei ~200 MHz, die interessanten Differenzen
    bei ~50 kHz. Deshalb wird vor dem Differenzbilden das Minimum abgezogen:
    mathematisch aendert das keine Differenz, numerisch verschwindet aber die
    Ausloeschung von acht signifikanten Stellen. Anschliessend werden nur
    Werte zusammengefasst, die naeher als rtol * f_max beieinander liegen -
    ohne diesen Schritt taeuscht schon eine Rundung von 1e-4 Hz dem
    nachfolgenden ggT inkommensurable Frequenzen vor."""
    f = np.asarray(f_spots, dtype=float)
    if f.size < 2:
        return np.array([])
    f = f - f.min()
    d = np.abs(f[:, None] - f[None, :])[np.triu_indices(f.size, k=1)]
    d = np.sort(d[d > 0])
    if d.size == 0:
        return np.array([])
    tol = d[-1] * rtol
    groups = [[d[0]]]
    for v in d[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return np.array([float(np.mean(g)) for g in groups])


def _gcd_float(a, b, scale, rel_tol=1e-8):
    """Groesster gemeinsamer Teiler zweier Frequenzen.

    `scale` ist die Bezugsgroesse fuer die Toleranz (die groesste
    vorkommende Frequenz) - sie darf NICHT mitschrumpfen, sonst wird das
    Abbruchkriterium im Verlauf des Euklid-Verfahrens immer schaerfer und
    bricht an numerischem Rauschen ab."""
    a, b = abs(float(a)), abs(float(b))
    if a < b:
        a, b = b, a
    for _ in range(500):
        if b <= scale * rel_tol:
            return a
        a, b = b, a - b * np.floor(a / b)
    return 0.0


def fundamental_beat_frequency(f_spots, rel_tol=1e-6):
    """Grundfrequenz des Beatings: groesster gemeinsamer Teiler ALLER
    vorkommenden Differenzfrequenzen.

    Rein numerisch aus den Spot-Frequenzen bestimmt, damit auch getrennte
    widths fuer x und y richtig behandelt werden. Bei gleicher width auf
    beiden Achsen kommt genau das analytische Ergebnis
    width / kgV(N_x-1, N_y-1) heraus (fuer 3x4 also width/6).

    Rueckgabe 0.0 bedeutet: kein gemeinsames Vielfaches - die
    Differenzfrequenzen sind inkommensurabel, das Signal ist NICHT periodisch
    und laesst sich nicht in ganzen Perioden darstellen."""
    d = unique_beat_frequencies(f_spots)
    if d.size == 0:
        return 0.0
    scale = float(d[-1])
    g = float(d[0])
    for v in d[1:]:
        g = _gcd_float(g, v, scale)
        if g <= 0:
            return 0.0
    ratios = d / g
    if not np.all(np.abs(ratios - np.round(ratios)) < rel_tol * np.maximum(1.0, ratios)):
        return 0.0
    return g


def degenerate_groups(f_spots, f0, tol=1e-6):
    """Gruppen von Spots, die EXAKT dieselbe Gesamtfrequenz tragen.

    Wichtig, weil das Zeitmittel nur dann der inkohaerenten Intensitaetssumme
    der uebrigen GUIs entspricht, wenn ALLE Kreuzterme mit einer von Null
    verschiedenen Frequenz umlaufen. Zwei Spots mit gleichem f_s haben einen
    Kreuzterm bei 0 Hz - der laeuft nie um, mittelt sich nie weg und
    erscheint als STATISCHE Interferenz.

    Bei gleichem offset und gleicher width auf beiden Achsen ist
        f_s(n,m) = 2*offset + width * ( n/(N_x-1) + m/(N_y-1) ),
    fuer 3x4 also 2*offset + width*(3n+2m)/6. Der Wert 3n+2m = 6 kommt
    zweimal vor: (n,m) = (0,3) und (2,0) - die beiden diagonal
    gegenueberliegenden Eck-Spots sind frequenzentartet.
    """
    if len(f_spots) < 2:
        return []
    f = np.asarray(f_spots, dtype=float)
    f = f - f.min()          # gegen Ausloeschung, siehe unique_beat_frequencies()
    scale = f0 if f0 > 0 else max(1.0, float(np.max(np.abs(f))))
    key = np.round(f / (scale * tol)).astype(np.int64)
    groups = []
    for val in np.unique(key):
        idx = np.flatnonzero(key == val)
        if idx.size > 1:
            groups.append(idx)
    return groups


def min_frames_per_period(f_spots, f0):
    """Kleinste Frame-Zahl pro Grundperiode, die die SCHNELLSTE vorkommende
    Beat-Frequenz noch aufloest.

    Das Zeitmittel ueber ein Fenster aus ganzen Perioden ist nur dann exakt,
    wenn keine Beat-Harmonische auf ein Vielfaches der Abtastrate faellt.
    Die hoechste Differenzfrequenz ist f_max = M * f0; Nyquist verlangt
    mehr als 2*M Abtastpunkte pro Grundperiode. Mit gleicher width auf
    beiden Achsen ist M klein (12 bei 3x4), mit getrennten widths kann f0
    sehr klein und M entsprechend gross werden - dann taeuschen zu wenige
    Frames ein voellig falsches Zeitmittel vor."""
    d = unique_beat_frequencies(f_spots)
    if d.size == 0 or f0 <= 0:
        return 1
    return int(2 * math.ceil(d[-1] / f0)) + 1


# ============================================================
# Tonphasen
# ============================================================
# Einstellbar sind physikalisch nur die N_x + N_y PHASEN DER RF-TOENE.
# Die Phase eines Spots (n,m) ist deren Summe:
#
#     phi_spot(n,m) = phi_x(n) + phi_y(m)
#
# Bei 3x4 kommen die zwoelf Spotphasen also aus sieben Freiheitsgraden, sie
# sind NICHT unabhaengig waehlbar.
#
# WAS PHASEN KOENNEN UND WAS NICHT
# Der Kreuzterm EINES Spotpaares lautet
#
#     2 * A_s A_s' * u_s(r) u_s'(r) * cos(2*pi*df*t + dphi)
#
# Die Phase steht nur im Kosinus. Fuer sich allein genommen laesst sich ein
# einzelnes Paar durch keine Phase daempfen - sie verschiebt nur, WANN das
# Maximum liegt.
#
# Entscheidend ist aber, dass sich viele Paare DIESELBE Differenzfrequenz
# teilen (bei 3x4 bis zu elf Paare pro Frequenz). Ihre Beitraege addieren
# sich als Zeiger:
#
#     D_d = sum_{k_s - k_s' = d} g_s g_s' e^{i(phi_s - phi_s')}
#
# und diese Summe KANN durch geeignete Phasen teilweise ausgeloescht werden.
# Die zeitliche Varianz ist genau sum_{d != 0} |D_d|^2, also sehr wohl
# phasenabhaengig. Gemessen bei 3x4 sinkt sigma_t/<I> im Plateau von 136 %
# (alle Phasen 0) auf 70 % im Optimum.
#
# Auf null bringen laesst es sich nicht: Differenzfrequenzen, die nur von
# einem einzigen Paar erzeugt werden (bei 3x4 z.B. d = 12), haben keinen
# Partner zum Ausloeschen. Auch mit voellig freien Spotphasen - die sich mit
# zwei AODs gar nicht ansteuern lassen - kommt man nur auf 55 %.


def schroeder_phases(N):
    """Schroeder-Phasen (M. R. Schroeder, IEEE Trans. Inf. Theory 16, 85 (1970)),
    der Standard fuer Multiton-Ansteuerung bei gleichen Amplituden:
        phi_n = -pi * n(n-1)/N
    Minimiert naeherungsweise den Crest-Faktor des Summensignals."""
    n = np.arange(N)
    return -np.pi * n * (n - 1) / max(N, 1)


def newman_phases(N):
    """Newman-Phasen, Alternative zu Schroeder: phi_n = pi (n-1)^2 / N."""
    n = np.arange(N)
    return np.pi * (n - 1) ** 2 / max(N, 1)


def spot_phases_from_tones(phase_x, phase_y, N_x, N_y):
    """phi_spot(n,m) = phi_x(n) + phi_y(m), in der Spot-Reihenfolge von
    compute_centers_and_freqs() (fx aussen, fy innen)."""
    return np.repeat(phase_x, N_y) + np.tile(phase_y, N_x)


def crest_factor(f_tones, phases, n_samples=20000, f_ref=None):
    """Crest-Faktor des RF-Summensignals einer Achse: Spitzenamplitude
    geteilt durch den Effektivwert. Massgeblich dafuer, wie stark der AOD
    kurzzeitig ausgesteuert wird."""
    f = np.asarray(f_tones, dtype=float)
    if f.size == 0:
        return float("nan")
    if f_ref is None or f_ref <= 0:
        span = np.ptp(f)
        f_ref = span if span > 0 else 1.0
    t = np.linspace(0.0, 1.0 / f_ref, n_samples, endpoint=False)
    sig = np.zeros_like(t)
    for fi, pi_ in zip(f, phases):
        sig += np.cos(2 * np.pi * fi * t + pi_)
    rms = np.sqrt(np.mean(sig ** 2))
    return float(np.max(np.abs(sig)) / rms) if rms > 0 else float("nan")


# ============================================================
# Exakte Zeitstatistik ohne Zeitschleife
# ============================================================
# Mit g_s(r) = A_s u_s(r) (reell) und z_s = g_s e^{i phi_s} ist
#
#     I(r,t) = |sum_s z_s e^{i w_s t}|^2 = sum_d D_d(r) e^{i d w_0 t}
#     D_d(r) = sum_{k_s - k_s' = d} g_s g_s' e^{i(phi_s - phi_s')}
#
# wobei k_s die Ordnung der Spot-Frequenz in Einheiten von f_0 ist. Daraus
# folgen Mittelwert und Varianz in geschlossener Form:
#
#     <I>(r)    = D_0(r)                (enthaelt die statischen Terme
#                                        frequenzentarteter Paare!)
#     Var_t(I)  = sum_{d != 0} |D_d|^2 = 2 * sum_{d > 0} |D_d|^2
#
# Das ist exakt - keine Abtastung, kein Aliasing - und um Groessenordnungen
# schneller als eine Zeitreihe. Genau das braucht die Phasenoptimierung.


def beat_orders(f_spots, f0):
    """Ordnung k_s jeder Spot-Frequenz in Einheiten von f_0, ganzzahlig."""
    f = np.asarray(f_spots, dtype=float)
    if f0 <= 0:
        return np.zeros(f.size, dtype=int)
    return np.round((f - f.min()) / f0).astype(int)


def pair_lists(k):
    """Fuer jede Ordnungsdifferenz d >= 0 die Liste der Spot-Paare (s, s')
    mit k_s - k_s' = d. d = 0 enthaelt sowohl die Diagonale als auch die
    frequenzentarteten Paare."""
    k = np.asarray(k)
    S = k.size
    out = {}
    for i in range(S):
        for j in range(S):
            d = int(k[i] - k[j])
            if d >= 0:
                out.setdefault(d, []).append((i, j))
    return {d: np.array(v) for d, v in out.items()}


def time_stats_exact(F, k, phases):
    """Zeitmittel und Zeitvarianz von I(r,t), exakt und ohne Zeitschleife.

    Rueckgabe (mean_map, var_map) in der Form von F[0]."""
    S = F.shape[0]
    shape = F.shape[1:]
    G = F.reshape(S, -1)
    e = np.exp(1j * np.asarray(phases, dtype=float))
    pl = pair_lists(k)

    ps = pl[0]
    w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
    mean = np.zeros(G.shape[1])
    for (i, j), wij in zip(ps, w):
        mean += float(np.real(wij)) * G[i] * G[j]

    var = np.zeros(G.shape[1])
    for d, ps in pl.items():
        if d == 0:
            continue
        w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
        Dre = np.zeros(G.shape[1])
        Dim = np.zeros(G.shape[1])
        for (i, j), wij in zip(ps, w):
            gg = G[i] * G[j]
            Dre += wij.real * gg
            Dim += wij.imag * gg
        var += 2.0 * (Dre * Dre + Dim * Dim)
    return mean.reshape(shape), var.reshape(shape)


class VariationObjective:
    """Vorberechneter Operator fuer die mittlere quadratische zeitliche
    Schwankung in einem Gebiet.

    Die Ortssumme laesst sich vor die Phasen ziehen:

        sum_r Var(r)/<I>(r)^2 = sum_{d>0} 2 * w_d^H M_d w_d
        M_d[p,q] = sum_r (g_s g_s')_p (g_s g_s')_q / <I>(r)^2

    Die M_d sind kleine Matrizen (hoechstens Spotzahl x Spotzahl) und werden
    einmal gebaut. Danach kostet eine Auswertung Mikrosekunden statt
    Millisekunden - erst das macht eine Mehrfachstart-Optimierung ueber die
    Tonphasen praktikabel.

    Der Nenner <I>(r) wird bei phases_ref eingefroren. Er haengt ueber die
    frequenzentarteten Paare selbst schwach von den Phasen ab; ihn mit zu
    variieren wuerde das Ziel unstetig machen, ohne etwas zu gewinnen."""

    def __init__(self, F, k, mask, phases_ref=None):
        S = F.shape[0]
        G = F.reshape(S, -1)[:, mask.ravel()]
        if phases_ref is None:
            phases_ref = np.zeros(S)
        mean, _ = time_stats_exact(F, k, phases_ref)
        mu2 = np.maximum(mean.ravel()[mask.ravel()], 1e-300) ** 2
        self.n = int(mask.sum())
        self.M, self.P = {}, {}
        for d, ps in pair_lists(k).items():
            if d == 0:
                continue
            prod = np.stack([G[i] * G[j] for i, j in ps])
            self.M[d] = (prod / mu2) @ prod.T
            self.P[d] = ps

    def components(self, phases):
        """Beitrag jeder Ordnungsdifferenz d einzeln, als relatives RMS.

        sigma_d = sqrt(2 |D_d|^2 / <I>^2), gemittelt ueber das Gebiet. Die
        Quadratsumme ueber alle d ergibt rms(). Damit laesst sich sehen, bei
        WELCHER Frequenz die Unruhe sitzt - entscheidend, weil die Falle nur
        auf Komponenten nahe nu_r und 2*nu_r wirklich reagiert."""
        e = np.exp(1j * np.asarray(phases, dtype=float))
        out = {}
        for d, M in self.M.items():
            ps = self.P[d]
            w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
            v = 2.0 * float(np.real(np.conj(w) @ M @ w))
            out[d] = float(np.sqrt(max(v, 0.0) / max(self.n, 1)))
        return out

    def rms_weighted(self, phases, weights):
        """Wie rms(), aber jede Ordnung d mit weights[d] gewichtet. Mit einer
        Gewichtung, die nur die Ordnungen nahe nu_r und 2*nu_r zaehlt, laesst
        sich gezielt das unterdruecken, worauf die Falle anspricht - auf
        Kosten der Ordnungen, die ihr egal sind."""
        e = np.exp(1j * np.asarray(phases, dtype=float))
        tot = 0.0
        for d, M in self.M.items():
            wgt = weights.get(d, 0.0)
            if wgt == 0.0:
                continue
            ps = self.P[d]
            w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
            tot += wgt * 2.0 * float(np.real(np.conj(w) @ M @ w))
        return float(np.sqrt(max(tot, 0.0) / max(self.n, 1)))

    def lower_bound(self, F, k, mask):
        """Untere Schranke fuer rms(), gueltig selbst bei voellig freien
        PAARphasen (die physikalisch nicht einstellbar sind - sie folgen aus
        den Spotphasen). Je Ordnung bleibt mindestens
        max(0, 2*max|c_p| - sum|c_p|) stehen: eine Frequenz, die nur EIN
        Spotpaar erzeugt, hat keinen Partner zum Ausloeschen."""
        S = F.shape[0]
        G = F.reshape(S, -1)[:, mask.ravel()]
        mean, _ = time_stats_exact(F, k, np.zeros(S))
        mu2 = np.maximum(mean.ravel()[mask.ravel()], 1e-300) ** 2
        tot = 0.0
        for d, ps in self.P.items():
            P = np.abs(np.stack([G[i] * G[j] for i, j in ps]))
            lo = np.maximum(0.0, 2 * P.max(axis=0) - P.sum(axis=0))
            tot += 2.0 * float(np.sum(lo ** 2 / mu2))
        return float(np.sqrt(tot / max(self.n, 1)))

    def rms(self, phases):
        """Wurzel aus der mittleren quadratischen relativen Schwankung,
        also der Mittelwert von sigma_t(I)/<I> im Gebiet (quadratisch)."""
        e = np.exp(1j * np.asarray(phases, dtype=float))
        tot = 0.0
        for d, M in self.M.items():
            ps = self.P[d]
            w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
            tot += 2.0 * float(np.real(np.conj(w) @ M @ w))
        return float(np.sqrt(max(tot, 0.0) / max(self.n, 1)))


class UniformitySeries:
    """Zeitverlauf der Uniformity U(t) = std(I)/mean(I) in einem Gebiet.

    Dieselbe Definition wie in den uebrigen GUIs und Optimierern des Projekts
    (dort auf dem Zeitmittel ausgewertet) - hier aber zu jedem Zeitpunkt
    einzeln. Der Vergleich der beiden ist der eigentliche Punkt: die Pipeline
    optimiert U(<I>), das Atom sieht U(t).

    Grosse Gebiete werden auf hoechstens `max_points` Stuetzstellen
    ausgeduennt; U ist ein Verhaeltnis von Flaechenmitteln und aendert sich
    dadurch nur in der dritten Stelle, die Optimierung wird aber schnell
    genug fuer hunderte Startpunkte."""

    def __init__(self, F, f_spots, mask, t, max_points=400, k=None):
        S = F.shape[0]
        idx = np.flatnonzero(mask.ravel())
        if idx.size > max_points:
            idx = idx[:: int(np.ceil(idx.size / max_points))]
        self.g = F.reshape(S, -1)[:, idx]
        self.t = np.asarray(t, dtype=float)
        self.W = np.exp(2j * np.pi * np.outer(self.t, np.asarray(f_spots, dtype=float)))
        # Zeitmittel auf denselben Stuetzstellen, fuer den Niveau-Faktor alpha(t)
        if k is not None:
            mean, _ = time_stats_exact(F, k, np.zeros(S))
            self.mu = mean.ravel()[idx]
            self.mu_norm = float(np.sum(self.mu * self.mu))
        else:
            self.mu = None

    def series(self, phases):
        Z = self.g * np.exp(1j * np.asarray(phases, dtype=float))[:, None]
        E = self.W @ Z
        I = E.real ** 2 + E.imag ** 2
        mu = I.mean(axis=1)
        return I.std(axis=1) / np.maximum(mu, 1e-300)

    def mean_u(self, phases):
        return float(np.mean(self.series(phases)))

    def level_and_u(self, phases):
        """(RMS von alpha(t)-1, mittleres U(t)) in einem Durchgang.

        alpha(t) ist der beste gemeinsame Skalenfaktor gegen das Zeitmittel:
        er misst, wie stark die Fallentiefe atmet, U(t) wie ungleich die
        Tiefen untereinander sind. Das sind die beiden Groessen, die fuer die
        Falle zaehlen - die reine Abweichung von <I> mischt sie."""
        Z = self.g * np.exp(1j * np.asarray(phases, dtype=float))[:, None]
        E = self.W @ Z
        I = E.real ** 2 + E.imag ** 2
        mu_t = I.mean(axis=1)
        u = float(np.mean(I.std(axis=1) / np.maximum(mu_t, 1e-300)))
        if self.mu is None or self.mu_norm <= 0:
            return 0.0, u
        alpha = (I @ self.mu) / self.mu_norm
        return float(np.sqrt(np.mean((alpha - 1.0) ** 2))), u


def uniformity_of(I, mask):
    """U = std/mean im Gebiet - die Projektkonvention."""
    v = I[mask]
    m = float(np.mean(v))
    return float(np.std(v) / m) if m > 0 else float("nan")


class WindowObjective:
    """RMS-Abweichung vom ZEITMITTEL <I> waehrend eines Zeitfensters [0, T_win].

    Ziel ist ein Fenster, in dem das Profil dem Zeitmittel nahekommt - nicht
    bloss "flach". Der Unterschied ist wesentlich: gegen den FENSTERmittelwert
    optimiert, erfuellt der Optimierer die Forderung, indem er das Licht im
    Fenster einfach herunterfaehrt. Flach, aber auf falschem Niveau und mit
    falschem Profil. Referenz ist deshalb <I>(r).

        I(r,t) - <I>(r) = 2 Re[ sum_{d>0} D_d(r) e^{i d w0 t} ]

    Wie bei VariationObjective laesst sich die Orts- UND die Zeitsumme vor die
    Phasen ziehen. Mit v_p = e^{i(phi_s - phi_s')} und
    prod_p(r) = g_s(r) g_s'(r):

        sum_{r,t} (I-<I>)^2/<I>^2 = 2 ( v M1 v* + Re[v M2 v] )
        M1[p,q] = A[d_p,d_q] * S1[p,q],   A[d,d'] = sum_t e^{i(d-d') w0 t}
        M2[p,q] = B[d_p,d_q] * S1[p,q],   B[d,d'] = sum_t e^{i(d+d') w0 t}
        S1[p,q] = sum_r prod_p prod_q / <I>^2

    ACHTUNG bei der Reihenfolge: der erste Term ist v @ M1 @ conj(v), NICHT
    conj(v) @ M1 @ v. M1 ist nicht hermitesch (A haengt an d_p, d_q, S1 ist
    symmetrisch), die beiden Ausdruecke sind verschieden - die falsche
    Variante liefert negative Werte und ein Scheinoptimum."""

    def __init__(self, F, k, mask, f0, t_win, n_t=60):
        S = F.shape[0]
        G = F.reshape(S, -1)[:, mask.ravel()]
        mean, _ = time_stats_exact(F, k, np.zeros(S))
        mu2 = np.maximum(mean.ravel()[mask.ravel()], 1e-300) ** 2
        self.n_r = int(mask.sum())
        self.t = np.linspace(0.0, t_win, n_t)
        self.n_t = n_t

        ds, pairs, prods = [], [], []
        for d, ps in pair_lists(k).items():
            if d == 0:
                continue
            for (i, j) in ps:
                ds.append(d); pairs.append((i, j)); prods.append(G[i] * G[j])
        self.pairs = np.array(pairs)
        if not ds:
            self.M1 = self.M2 = None
            return
        ds = np.array(ds)
        P = np.stack(prods)
        S1 = (P / mu2) @ P.T
        e = np.exp(2j * np.pi * f0 * np.outer(ds, self.t))       # e^{i d w0 t}
        A = e @ e.conj().T
        Bm = e @ e.T
        self.M1 = A * S1
        self.M2 = Bm * S1

    def rms(self, phases):
        if self.M1 is None:
            return 0.0
        e = np.exp(1j * np.asarray(phases, dtype=float))
        v = e[self.pairs[:, 0]] * np.conj(e[self.pairs[:, 1]])
        q = 2.0 * (float(np.real(v @ self.M1 @ np.conj(v)))
                   + float(np.real(v @ self.M2 @ v))) / (self.n_r * self.n_t)
        return float(np.sqrt(max(q, 0.0)))


def window_stats(F, k, phases, mask, f_spots, t_win, n_t=120):
    """Kennzahlen im Zeitfenster, zerlegt in Niveau und Form.

    alpha(t) ist der beste gemeinsame Skalenfaktor: springt nur alpha, atmet
    die ganze Fallentiefe und das FlatTop-Profil bleibt heil. Was nach Abzug
    von alpha uebrig bleibt, ist echte Formaenderung - genau das, was die
    Uniformity zwischen den Sites verdirbt."""
    S = F.shape[0]
    mean, _ = time_stats_exact(F, k, np.zeros(S))
    mu = mean[mask]
    t = np.linspace(0.0, t_win, n_t)
    cube = intensity_cube(F, f_spots, phases, t).astype(np.float64)[:, mask]
    total = float(np.sqrt(np.mean(((cube - mu) / mu) ** 2)))
    denom = float(np.sum(mu * mu))
    alpha = (cube @ mu) / denom if denom > 0 else np.ones(n_t)
    level = float(np.sqrt(np.mean((alpha - 1.0) ** 2)))
    shape = float(np.sqrt(np.mean(((cube - alpha[:, None] * mu) / mu) ** 2)))
    return dict(total=total, level=level, shape=shape,
                alpha_min=float(alpha.min()), alpha_max=float(alpha.max()))


def quadrature_penalty(phases, degen):
    """Wie weit sind frequenzentartete Paare von der Quadratur entfernt?

    Ein Paar mit gleicher Gesamtfrequenz hat einen Kreuzterm bei 0 Hz:
    2 g_s g_s' cos(phi_s - phi_s'). Der laeuft nie um und mittelt sich nie
    weg - ER IST ABER EXAKT NULL, wenn die beiden Spots in Quadratur stehen
    (Phasendifferenz 90 oder 270 Grad). Genau dann, und nur dann, ist das
    Zeitmittel exakt die inkohaerente Intensitaetssumme, die die uebrigen
    GUIs und die Scan-Pipeline rechnen.

    Rueckgabe 0 heisst: alle entarteten Paare stehen in Quadratur."""
    if not degen:
        return 0.0
    pen = 0.0
    for grp in degen:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                pen += abs(np.cos(float(phases[grp[i]]) - float(phases[grp[j]])))
    return pen


def resonance_check(beats, nu_r, tol_rel=0.06):
    """Faellt eine Schwebungslinie mit nu_r oder 2*nu_r zusammen?

    Die kohaerente Schwebung erzeugt ein LINIENspektrum bei Vielfachen von
    f_0. Liegt keine Linie nahe nu_r oder 2*nu_r, bekommt die Falle trotz
    100 % Modulationstiefe praktisch keine Leistung auf ihren Resonanzen -
    das ist der entscheidende Punkt, nicht die Groesse der Modulation.

    Rueckgabe: (abstand_nu, abstand_2nu, kritisch) mit den Abstaenden der
    naechstgelegenen Linie in Hz."""
    if beats is None or len(beats) == 0 or nu_r <= 0:
        return float("nan"), float("nan"), False
    b = np.asarray(beats, dtype=float)
    d1 = float(np.min(np.abs(b - nu_r)))
    d2 = float(np.min(np.abs(b - 2 * nu_r)))
    crit = (d1 < tol_rel * nu_r) or (d2 < tol_rel * 2 * nu_r)
    return d1, d2, crit


def compute_grid(centers_x, centers_y, win_eff, resolution, pad_factor=2.5):
    """Quadratisches Grid, das alle Spots plus einen Rand von pad_factor
    Strahlradien umfasst. Anders als in den Metrik-GUIs braucht es hier
    KEINE Pitch-Kopien der Nachbarn - Beating entsteht zwischen den Toenen
    desselben Musters, nicht zwischen Nachbarfallen."""
    pad = pad_factor * win_eff
    x_lo, x_hi = centers_x.min() - pad, centers_x.max() + pad
    y_lo, y_hi = centers_y.min() - pad, centers_y.max() + pad
    # quadratisch machen, damit der 2D-Plot nicht verzerrt
    cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    half = 0.5 * max(x_hi - x_lo, y_hi - y_lo)
    x = np.linspace(cx - half, cx + half, resolution)
    y = np.linspace(cy - half, cy + half, resolution)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def intensity_cube(F, f_spots, phases, t):
    """I(x,y,t) = |sum_s F_s exp(i(2pi f_s t + phi_s))|^2 fuer alle t.

    Rueckgabe float32, um den Speicher im Rahmen zu halten (bei 200x200 und
    240 Frames sind das 38 MB)."""
    n_t = len(t)
    S, ny, nx = F.shape
    cube = np.empty((n_t, ny, nx), dtype=np.float32)
    for k in range(n_t):
        ph = 2.0 * np.pi * f_spots * t[k] + phases
        E_re = np.tensordot(np.cos(ph), F, axes=(0, 0))
        E_im = np.tensordot(np.sin(ph), F, axes=(0, 0))
        cube[k] = (E_re * E_re + E_im * E_im).astype(np.float32)
    return cube


# ============================================================
# Hauptfenster
# ============================================================
class BeatingMultitoneWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beating Multitone GUI - kohaerente Zeitentwicklung")
        self.resize(1600, 950)

        self.state = {
            "N_x": 3,
            "N_y": 4,
            # Startwerte des Arbeitspunkts - so wird das GUI immer geoeffnet.
            "use_airy": True,
            "airy_factor": AIRY_SCALE_DEFAULT,
            "win": 1.10e-6,        # m, Waist NACH den Linsen
            "win_in": None,        # m, Waist VOR den Linsen
            "win_mode": "output",
            "width_x": 0.45e6,     # Hz, Frequenzspanne der x-Toene
            "width_y": 0.45e6,     # Hz, Frequenzspanne der y-Toene
            "link_width": True,    # width_y folgt width_x
            "r_x": 1.0,
            "r_y": 1.2,
            "lambda_opt": 795e-9,  # m
            "offset": 100e6,       # Hz
            "f1": 60e-3,
            "f2": 750e-3,
            "grid_n": 200,
            "n_periods": 3,
            "frames_per_period": 60,
            "phase_x": np.zeros(3),      # rad, Phase je x-Ton
            "phase_y": np.zeros(4),      # rad, Phase je y-Ton
            "phase_scope": "tone",       # 'tone' (physikalisch) | 'free' (Diagnose)
            "t_win": 3.0e-6,             # s, Laenge des Ruhefensters
            "phase_free": np.zeros(12),  # rad, freie Phase je Spot (unphysikalisch)
            "auto_update": False,
        }
        self.state["win_in"] = conjugate_waist(
            self.state["win"], self.state["f1"], self.state["f2"], self.state["lambda_opt"])

        self.cache = {}
        self._panel_cbar = None
        self._opt_note = ""
        self._art = {}          # dauerhafte Zeichenobjekte fuer den schnellen Pfad
        self._last_panel = None
        self.frame_idx = 0
        self._building = True
        self.out_dir = _resolve_out_dir()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self._build_ui()
        self._building = False
        self.recompute()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.fig = Figure(figsize=(10, 8.5))
        self.fig.set_constrained_layout(True)
        self.canvas = FigureCanvas(self.fig)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.55, 1], height_ratios=[1.5, 1, 1])
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_st = self.fig.add_subplot(gs[0, 1])
        self.ax_cut = self.fig.add_subplot(gs[1, 1])
        self.ax_time = self.fig.add_subplot(gs[2, 1])
        splitter.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setAlignment(Qt.AlignTop)
        pl.addWidget(self._group_tones())
        pl.addWidget(self._group_profile())
        pl.addWidget(self._group_beam())
        pl.addWidget(self._group_amps())
        pl.addWidget(self._group_phases())
        pl.addWidget(self._group_time())
        pl.addWidget(self._group_actions())
        pl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(360)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1150, 420])

    # -- kleine Helfer fuer die Eingabefelder ----------------
    def _dspin(self, value, lo, hi, dec, step, suffix=""):
        w = QDoubleSpinBox()
        w.setDecimals(dec)
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(value)
        w.setKeyboardTracking(False)
        if suffix:
            w.setSuffix(" " + suffix)
        w.valueChanged.connect(self._on_param_changed)
        return w

    def _ispin(self, value, lo, hi, suffix=""):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(value)
        w.setKeyboardTracking(False)
        if suffix:
            w.setSuffix(" " + suffix)
        w.valueChanged.connect(self._on_param_changed)
        return w

    def _group_tones(self):
        g = QGroupBox("Toene")
        lay = QGridLayout(g)
        self.sp_nx = self._ispin(self.state["N_x"], 1, 20)
        self.sp_ny = self._ispin(self.state["N_y"], 1, 20)
        lay.addWidget(QLabel("N_x"), 0, 0); lay.addWidget(self.sp_nx, 0, 1)
        lay.addWidget(QLabel("N_y"), 1, 0); lay.addWidget(self.sp_ny, 1, 1)
        self.lbl_freqs = QLabel("-")
        self.lbl_freqs.setWordWrap(True)
        self.lbl_freqs.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_freqs, 2, 0, 1, 2)
        return g

    def _group_profile(self):
        g = QGroupBox("Strahlprofil")
        lay = QGridLayout(g)
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems(["Gauss", "Airy"])
        self.cmb_profile.setCurrentIndex(1 if self.state["use_airy"] else 0)
        self.cmb_profile.currentIndexChanged.connect(self._on_param_changed)
        self.sp_airy = self._dspin(self.state["airy_factor"], 0.1, 5.0, 4, 0.01)
        self.sp_airy.setToolTip("first_zero_radius = Faktor * waist.\n"
                                "1.4830 = gleiche 1/e^2-Breite wie der Gauss,\n"
                                "1.19 = historischer Wert.")
        lay.addWidget(QLabel("Profil"), 0, 0); lay.addWidget(self.cmb_profile, 0, 1)
        lay.addWidget(QLabel("Airy-Faktor"), 1, 0); lay.addWidget(self.sp_airy, 1, 1)
        return g

    def _group_beam(self):
        g = QGroupBox("Strahl / Optik")
        lay = QGridLayout(g)
        self.cmb_winmode = QComboBox()
        self.cmb_winmode.addItems(["Waist nach Linse (um)", "Waist vor Linse (mm)"])
        self.cmb_winmode.currentIndexChanged.connect(self._on_winmode_changed)
        self.sp_win = self._dspin(self.state["win"] * 1e6, 0.05, 50.0, 4, 0.01, "um")
        self.sp_win_in = self._dspin(self.state["win_in"] * 1e3, 0.01, 50.0, 4, 0.01, "mm")
        self.sp_width = self._dspin(self.state["width_x"] * 1e-6, 0.0, 36.0, 5, 0.01, "MHz")
        self.sp_width_y = self._dspin(self.state["width_y"] * 1e-6, 0.0, 36.0, 5, 0.01, "MHz")
        self.cb_link_width = QCheckBox("width_y = width_x")
        self.cb_link_width.setChecked(self.state["link_width"])
        self.cb_link_width.stateChanged.connect(self._on_link_width_changed)
        self.cb_link_width.setToolTip(
            "Die uebrigen GUIs setzen beide widths gleich. Genau das erzeugt\n"
            "Frequenzentartungen: bei 3x4 tragen die Spots (n,m) = (0,3) und\n"
            "(2,0) exakt dieselbe Gesamtfrequenz und interferieren statisch.\n"
            "Getrennte widths heben das auf - aendern aber den Spot-Abstand in y.")
        self.sp_lambda = self._dspin(self.state["lambda_opt"] * 1e9, 200.0, 2000.0, 2, 1.0, "nm")
        self.sp_offset = self._dspin(self.state["offset"] * 1e-6, 0.0, 500.0, 4, 1.0, "MHz")
        self.sp_f1 = self._dspin(self.state["f1"] * 1e3, 1.0, 2000.0, 2, 5.0, "mm")
        self.sp_f2 = self._dspin(self.state["f2"] * 1e3, 1.0, 2000.0, 2, 5.0, "mm")

        rows = [("Modus", self.cmb_winmode), ("waist", self.sp_win),
                ("waist_in", self.sp_win_in),
                ("width x", self.sp_width), ("width y", self.sp_width_y),
                ("", self.cb_link_width),
                ("Wellenlaenge", self.sp_lambda), ("Offset f0", self.sp_offset),
                ("f1", self.sp_f1), ("f2", self.sp_f2)]
        for i, (name, w) in enumerate(rows):
            lay.addWidget(QLabel(name), i, 0)
            lay.addWidget(w, i, 1)
        hint = QLabel("Wellenlaenge und Offset aendern die Geometrie,\n"
                      "aber KEINE Beat-Frequenz - in |E|^2 stehen nur\n"
                      "Differenzen der Tonfrequenzen, und ein konstanter\n"
                      "Versatz kuerzt sich aus jeder Differenz heraus.")
        hint.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(hint, len(rows), 0, 1, 2)
        self._sync_winmode_enabled()
        self._sync_width_enabled()
        return g

    def _group_amps(self):
        g = QGroupBox("Amplituden (aussen/innen)")
        lay = QGridLayout(g)
        self.sp_rx = self._dspin(self.state["r_x"], 0.0, 10.0, 4, 0.01)
        self.sp_ry = self._dspin(self.state["r_y"], 0.0, 10.0, 4, 0.01)
        lay.addWidget(QLabel("r_x"), 0, 0); lay.addWidget(self.sp_rx, 0, 1)
        lay.addWidget(QLabel("r_y"), 1, 0); lay.addWidget(self.sp_ry, 1, 1)
        self.lbl_amps = QLabel("-")
        self.lbl_amps.setWordWrap(True)
        self.lbl_amps.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_amps, 2, 0, 1, 2)
        return g

    def _group_time(self):
        g = QGroupBox("Zeitachse")
        lay = QGridLayout(g)
        self.sp_periods = self._ispin(self.state["n_periods"], 1, 50)
        self.sp_fpp = self._ispin(self.state["frames_per_period"], 8, 400)
        self.sp_grid = self._ispin(self.state["grid_n"], 60, 600)
        lay.addWidget(QLabel("Perioden"), 0, 0); lay.addWidget(self.sp_periods, 0, 1)
        self.sp_fpp.setToolTip(
            "Muss groesser sein als 2 * (hoechste Beat-Frequenz / f_0),\n"
            "sonst wird das Beating falsch abgetastet (Aliasing) und schon\n"
            "das Zeitmittel stimmt nicht mehr. Das GUI warnt, wenn es zu\n"
            "wenige sind, und nennt die noetige Zahl.")
        lay.addWidget(QLabel("Frames/Periode"), 1, 0); lay.addWidget(self.sp_fpp, 1, 1)
        lay.addWidget(QLabel("Grid-Aufloesung"), 2, 0); lay.addWidget(self.sp_grid, 2, 1)

        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.setMinimum(0)
        self.slider_t.setMaximum(0)
        self.slider_t.valueChanged.connect(self._on_time_slider)
        lay.addWidget(QLabel("t"), 4, 0); lay.addWidget(self.slider_t, 4, 1)

        row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._on_play_clicked)
        row.addWidget(self.btn_play)
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 60)      # Frames pro Sekunde
        self.slider_speed.setValue(20)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        row.addWidget(QLabel("fps"))
        row.addWidget(self.slider_speed)
        holder = QWidget(); holder.setLayout(row)
        lay.addWidget(holder, 5, 0, 1, 2)

        self.lbl_beat = QLabel("-")
        self.lbl_beat.setWordWrap(True)
        self.lbl_beat.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_beat, 6, 0, 1, 2)

        self.lbl_degen = QLabel("-")
        self.lbl_degen.setWordWrap(True)
        self.lbl_degen.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_degen, 7, 0, 1, 2)

        self.lbl_res = QLabel("")
        self.lbl_res.setWordWrap(True)
        self.lbl_res.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_res, 8, 0, 1, 2)
        return g

    def _group_phases(self):
        g = QGroupBox("Tonphasen")
        outer = QVBoxLayout(g)

        self.cmb_phase_scope = QComboBox()
        self.cmb_phase_scope.addItems([
            "je Ton (physikalisch: N_x + N_y Werte)",
            "frei je Spot (unphysikalisch, nur zum Testen)",
        ])
        self.cmb_phase_scope.setToolTip(
            "Einstellbar sind im Aufbau nur die Phasen der RF-Toene.\n"
            "Ein Spot (n,m) traegt phi_x(n) + phi_y(m) - die zwoelf\n"
            "Spotphasen kommen bei 3x4 also aus sieben Freiheitsgraden.\n\n"
            "Der zweite Eintrag vergibt trotzdem jedem Spot eine eigene,\n"
            "unabhaengige Zufallsphase. Das laesst sich nicht ansteuern und\n"
            "dient nur der Probe, dass auch volle Freiheit die Modulation\n"
            "nicht beseitigt.")
        self.cmb_phase_scope.currentIndexChanged.connect(self._on_phase_scope_changed)
        outer.addWidget(self.cmb_phase_scope)

        btns = QHBoxLayout()
        for label, fn in (("0", "zero"), ("Schroeder", "schroeder"),
                          ("Newman", "newman"), ("wuerfeln", "random")):
            b = QPushButton(label)
            b.clicked.connect(lambda _, k=fn: self._apply_phase_preset(k))
            btns.addWidget(b)
        holder = QWidget(); holder.setLayout(btns)
        outer.addWidget(holder)

        row = QHBoxLayout()
        row.addWidget(QLabel("Zielgebiet"))
        self.cmb_opt_region = QComboBox()
        self.cmb_opt_region.addItems(["Plateau (<I> > 50 % max)", "Spot-Zentren",
                                      "Kreis um die Mitte"])
        row.addWidget(self.cmb_opt_region)
        self.sp_opt_radius = QDoubleSpinBox()
        self.sp_opt_radius.setRange(0.1, 20.0); self.sp_opt_radius.setDecimals(2)
        self.sp_opt_radius.setSingleStep(0.1); self.sp_opt_radius.setValue(2.0)
        self.sp_opt_radius.setSuffix(" um")
        row.addWidget(self.sp_opt_radius)
        holder2 = QWidget(); holder2.setLayout(row)
        outer.addWidget(holder2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Ziel"))
        self.cmb_opt_weight = QComboBox()
        self.cmb_opt_weight.addItems([
            "breitbandig (alle Frequenzen)",
            "nur nahe nu_r und 2*nu_r (Falle)",
        ])
        self.cmb_opt_weight.setToolTip(
            "Die Falle reagiert nicht auf jede Schwebungsfrequenz gleich.\n"
            "Komponenten weit oberhalb der Fallenfrequenz mittelt das Atom\n"
            "weg; gefaehrlich sind die nahe nu_r (Aufheizen) und 2*nu_r\n"
            "(parametrische Resonanz).\n\n"
            "Der zweite Eintrag opfert die harmlosen Ordnungen, um die\n"
            "gefaehrlichen zu druecken - das bringt dort deutlich mehr als\n"
            "die breitbandige Optimierung.")
        row3.addWidget(self.cmb_opt_weight)
        row3.addWidget(QLabel("nu_r"))
        self.sp_nu_r = QDoubleSpinBox()
        self.sp_nu_r.setRange(0.1, 5000.0); self.sp_nu_r.setDecimals(1)
        self.sp_nu_r.setValue(60.4); self.sp_nu_r.setSuffix(" kHz")
        self.sp_nu_r.setKeyboardTracking(False)
        self.sp_nu_r.valueChanged.connect(self._on_param_changed)
        row3.addWidget(self.sp_nu_r)
        holder3 = QWidget(); holder3.setLayout(row3)
        outer.addWidget(holder3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Ruhefenster"))
        self.sp_twin = QDoubleSpinBox()
        self.sp_twin.setRange(0.05, 200.0); self.sp_twin.setDecimals(2)
        self.sp_twin.setSingleStep(0.5); self.sp_twin.setValue(3.0)
        self.sp_twin.setSuffix(" us"); self.sp_twin.setKeyboardTracking(False)
        self.sp_twin.valueChanged.connect(self._on_param_changed)
        row4.addWidget(self.sp_twin)
        self.cmb_opt_target = QComboBox()
        self.cmb_opt_target.addItems([
            "Abweichung von <I>",
            "Uniformity U(t)",
            "Fallentiefe + Uniformity",
        ])
        self.cmb_opt_target.setToolTip(
            "Zwei verschiedene Ziele:\n\n"
            "'Abweichung von <I>' haelt Hoehe UND Form nahe am Zeitmittel.\n"
            "'Uniformity U(t)' haelt nur die Gleichmaessigkeit INNERHALB des\n"
            "Gebiets klein - das Niveau darf dabei atmen.\n\n"
            "'Fallentiefe + Uniformity' kombiniert beide Groessen, die fuer\n"
            "die Falle zaehlen: alpha(t) (atmet die Tiefe?) und U(t) (sind\n"
            "die Tiefen untereinander gleich?).\n\n"
            "Sie fuehren auf verschiedene Phasen: auf U optimiert kommt man\n"
            "an den Spot-Zentren im 3-us-Fenster auf 23 % statt 45 %.")
        row4.addWidget(self.cmb_opt_target)
        self.btn_min_win = QPushButton("Ruhefenster optimieren")
        self.btn_min_win.setToolTip(
            "Sucht die Tonphasen, bei denen das Profil waehrend eines Fensters\n"
            "der eingestellten Laenge dem ZEITMITTEL <I> moeglichst nahe kommt.\n\n"
            "Referenz ist bewusst <I> und nicht der Mittelwert im Fenster: sonst\n"
            "erfuellt der Optimierer die Forderung, indem er das Licht im Fenster\n"
            "herunterfaehrt - flach, aber auf falschem Niveau.\n\n"
            "Das Fenster liegt bei t = 0; seine Lage ist keine zusaetzliche\n"
            "Freiheit, weil eine Zeitverschiebung selbst ein Phasensatz ist.")
        self.btn_min_win.clicked.connect(self._on_minimize_window)
        row4.addWidget(self.btn_min_win)
        holder4 = QWidget(); holder4.setLayout(row4)
        outer.addWidget(holder4)

        self.cb_quad = QCheckBox("entartete Paare in Quadratur halten")
        self.cb_quad.setChecked(True)
        self.cb_quad.setToolTip(
            "Frequenzentartete Spots haben einen Kreuzterm bei 0 Hz, der sich\n"
            "nie wegmittelt - die statische Verzerrung, die die inkohaerenten\n"
            "GUIs nicht sehen.\n\n"
            "Bei einer Phasendifferenz von 90 Grad ist dieser Term EXAKT null.\n"
            "Dann - und nur dann - ist das Zeitmittel exakt die inkohaerente\n"
            "Summe. Kostet fast nichts: die Uniformity im Optimum geht von\n"
            "23.5 auf 24.3 Prozent.")
        self.cb_quad.stateChanged.connect(self._on_param_changed)
        outer.addWidget(self.cb_quad)

        self.btn_min_var = QPushButton("Zeitliche Variation minimieren")
        self.btn_min_var.setToolTip(
            "Sucht die Tonphasen mit der kleinsten mittleren relativen\n"
            "Schwankung sigma_t(I)/<I> im gewaehlten Gebiet.\n\n"
            "Anders als bei der Spitze geht das wirklich: viele Spotpaare\n"
            "teilen sich dieselbe Differenzfrequenz, und ihre Kreuzterme\n"
            "koennen sich teilweise ausloeschen. Auf null bringen laesst es\n"
            "sich nicht - Paare mit einer nur einmal vorkommenden\n"
            "Differenzfrequenz haben keinen Partner zum Ausloeschen.")
        self.btn_min_var.clicked.connect(self._on_minimize_variation)
        outer.addWidget(self.btn_min_var)

        self.btn_min_peak = QPushButton("Spitze minimieren (dauert einige Sekunden)")
        self.btn_min_peak.setToolTip(
            "Sucht den Phasensatz mit der kleinsten Spitzenintensitaet.\n"
            "Das senkt die Belastung des AOD und die Ueberhoehung im Bild -\n"
            "die Modulationstiefe im Plateau bleibt davon fast unberuehrt.")
        self.btn_min_peak.clicked.connect(self._on_minimize_peak)
        outer.addWidget(self.btn_min_peak)

        self.phase_grid_host = QWidget()
        self.phase_grid = QGridLayout(self.phase_grid_host)
        self.phase_grid.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.phase_grid_host)
        self.phase_spins_x, self.phase_spins_y = [], []
        self._rebuild_phase_fields()

        self.lbl_phase = QLabel("-")
        self.lbl_phase.setWordWrap(True)
        self.lbl_phase.setStyleSheet("color: #555; font-size: 10px;")
        outer.addWidget(self.lbl_phase)
        return g

    def _rebuild_phase_fields(self):
        """Legt die Eingabefelder neu an, wenn sich N_x oder N_y aendert.
        Vorhandene Werte werden dabei so weit wie moeglich uebernommen."""
        while self.phase_grid.count():
            item = self.phase_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.phase_spins_x, self.phase_spins_y = [], []
        N_x, N_y = self.state["N_x"], self.state["N_y"]
        for name, N, key, store in (("phi_x", N_x, "phase_x", self.phase_spins_x),
                                    ("phi_y", N_y, "phase_y", self.phase_spins_y)):
            old = np.asarray(self.state[key], dtype=float)
            vals = np.zeros(N)
            vals[:min(N, old.size)] = old[:min(N, old.size)]
            self.state[key] = vals
            col = 0 if key == "phase_x" else 1
            self.phase_grid.addWidget(QLabel(name + " (Grad)"), 0, col)
            for i in range(N):
                sp = QDoubleSpinBox()
                sp.setRange(-720.0, 720.0)
                sp.setDecimals(1)
                sp.setSingleStep(15.0)
                sp.setWrapping(True)
                sp.setKeyboardTracking(False)
                sp.setValue(float(np.degrees(vals[i])))
                sp.valueChanged.connect(self._on_param_changed)
                self.phase_grid.addWidget(sp, i + 1, col)
                store.append(sp)
        self._sync_phase_fields_enabled()

    def _sync_phase_fields_enabled(self):
        on = self.cmb_phase_scope.currentIndex() == 0
        self.phase_grid_host.setEnabled(on)
        self.btn_min_peak.setEnabled(on)
        self.btn_min_var.setEnabled(on)
        self.btn_min_win.setEnabled(on)

    def _on_phase_scope_changed(self, idx):
        self.state["phase_scope"] = "tone" if idx == 0 else "free"
        if self.state["phase_scope"] == "free":
            rng = np.random.default_rng()
            self.state["phase_free"] = rng.uniform(
                0.0, 2 * np.pi, self.state["N_x"] * self.state["N_y"])
        self._sync_phase_fields_enabled()
        self.recompute()

    def _apply_phase_preset(self, kind):
        N_x, N_y = self.state["N_x"], self.state["N_y"]
        if kind == "zero":
            px, py = np.zeros(N_x), np.zeros(N_y)
        elif kind == "schroeder":
            px, py = schroeder_phases(N_x), schroeder_phases(N_y)
        elif kind == "newman":
            px, py = newman_phases(N_x), newman_phases(N_y)
        else:
            rng = np.random.default_rng()
            px = rng.uniform(0, 2 * np.pi, N_x)
            py = rng.uniform(0, 2 * np.pi, N_y)
        self._write_phase_fields(px, py)
        self.recompute()

    def _write_phase_fields(self, px, py):
        self.state["phase_x"], self.state["phase_y"] = np.asarray(px), np.asarray(py)
        for sp, v in zip(self.phase_spins_x, px):
            sp.blockSignals(True); sp.setValue(float(np.degrees(v)) % 360.0); sp.blockSignals(False)
        for sp, v in zip(self.phase_spins_y, py):
            sp.blockSignals(True); sp.setValue(float(np.degrees(v)) % 360.0); sp.blockSignals(False)

    def _target_mask(self, mean_map, X, Y, centers_x, centers_y, x, y):
        """Gebiet, auf das die Phasenoptimierung zielt."""
        idx = self.cmb_opt_region.currentIndex()
        if idx == 0:
            return mean_map > 0.5 * mean_map.max() if mean_map.max() > 0 \
                else np.ones_like(mean_map, bool)
        if idx == 1:
            m = np.zeros(mean_map.shape, bool)
            for cxi, cyi in zip(centers_x, centers_y):
                m[int(np.argmin(np.abs(y - cyi))), int(np.argmin(np.abs(x - cxi)))] = True
            return m
        r = self.sp_opt_radius.value() * 1e-6
        cx0, cy0 = float(np.mean(centers_x)), float(np.mean(centers_y))
        m = ((X - cx0) ** 2 + (Y - cy0) ** 2) <= r ** 2
        return m if m.any() else np.ones_like(mean_map, bool)

    def _trap_weights(self, orders, f0):
        """Gewicht je Ordnungsdifferenz d. Breitbandig: alles 1. Sonst nur die
        Ordnungen, die nahe nu_r oder 2*nu_r liegen (Fenster +-35 %)."""
        if self.cmb_opt_weight.currentIndex() == 0:
            return {int(d): 1.0 for d in orders}, "breitbandig"
        nu = self.sp_nu_r.value() * 1e3
        w = {}
        for d in orders:
            f = d * f0
            if abs(f - nu) < 0.35 * nu or abs(f - 2 * nu) < 0.35 * nu:
                w[int(d)] = 1.0
        if not w:                      # keine Ordnung im Fenster
            return {int(d): 1.0 for d in orders}, "breitbandig (keine Ordnung nahe nu_r)"
        return w, f"nahe nu_r={nu * 1e-3:.1f} kHz"

    def _on_minimize_variation(self):
        """Sucht die Tonphasen mit der kleinsten zeitlichen Schwankung.

        Nutzt die geschlossene Form (VariationObjective) - eine Auswertung
        kostet Mikrosekunden, deshalb sind hunderte Startpunkte moeglich."""
        from scipy.optimize import minimize
        self._read_widgets()
        s = self.state
        N_x, N_y = s["N_x"], s["N_y"]
        n_free = max(0, N_x - 1) + max(0, N_y - 1)
        if n_free == 0:
            self.lbl_status.setText("Nur ein Ton je Achse - keine Phasen frei.")
            return
        self.lbl_status.setText("baue Operator ...")
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        QApplication.processEvents()

        cxs, cys, f_spots, _, _, _, _ = compute_centers_and_freqs(
            N_x, N_y, s["width_x"], s["width_y"], s["f1"], s["f2"], s["offset"])
        f0 = fundamental_beat_frequency(f_spots)
        if f0 <= 0:
            self.lbl_status.setText("Kein gemeinsames f_0 - Optimierung nicht definiert.")
            return
        amp = amp_spots_from_ratios(s["r_x"], s["r_y"], N_x, N_y)
        win_eff = s["win"] * (s["airy_factor"] if s["use_airy"] else 1.0)
        xg, yg, Xg, Yg = compute_grid(cxs, cys, win_eff, 110)
        Fg = build_field_stack(Xg, Yg, cxs, cys, amp, s["win"], s["use_airy"], s["airy_factor"])
        k = beat_orders(f_spots, f0)
        mean0, _ = time_stats_exact(Fg, k, np.zeros(len(f_spots)))
        mask = self._target_mask(mean0, Xg, Yg, cxs, cys, xg, yg)
        obj = VariationObjective(Fg, k, mask)
        weights, wname = self._trap_weights(sorted(obj.M.keys()), f0)

        degen_g = degenerate_groups(f_spots, f0)

        def cost(v):
            px = np.concatenate(([0.0], v[:N_x - 1])) if N_x > 1 else np.zeros(1)
            py = np.concatenate(([0.0], v[N_x - 1:])) if N_y > 1 else np.zeros(1)
            ph = spot_phases_from_tones(px, py, N_x, N_y)
            pen = (2.0 * quadrature_penalty(ph, degen_g)
                   if self.cb_quad.isChecked() else 0.0)
            return obj.rms_weighted(ph, weights) + pen

        ph_now = spot_phases_from_tones(s["phase_x"], s["phase_y"], N_x, N_y)
        start = obj.rms_weighted(ph_now, weights)
        bound = obj.lower_bound(Fg, k, mask)
        self.lbl_status.setText("suche Phasen mit minimaler zeitlicher Variation ...")
        QApplication.processEvents()
        rng = np.random.default_rng(5)
        best_f, best_v = cost(np.zeros(n_free)), np.zeros(n_free)
        for n in range(250):
            r = minimize(cost, rng.uniform(0, 2 * np.pi, n_free), method="Nelder-Mead",
                         options=dict(maxiter=2000, xatol=1e-5, fatol=1e-10))
            if r.fun < best_f:
                best_f, best_v = float(r.fun), r.x
            if n % 50 == 0:
                QApplication.processEvents()
        px = np.concatenate(([0.0], best_v[:N_x - 1])) if N_x > 1 else np.zeros(1)
        py = np.concatenate(([0.0], best_v[N_x - 1:])) if N_y > 1 else np.zeros(1)
        self._write_phase_fields(px, py)
        ph_best = spot_phases_from_tones(px, py, N_x, N_y)
        self._opt_note = (
            f"Optimierung ({wname}): Zielgroesse {start * 100:.1f} % -> {best_f * 100:.1f} %"
            f"   |   breitbandig danach {obj.rms(ph_best) * 100:.1f} %"
            f"   |   absolute Schranke {bound * 100:.1f} % (selbst bei freien Paarphasen)")
        self.recompute()

    def _on_minimize_window(self):
        """Sucht die Tonphasen mit dem besten Ruhefenster."""
        from scipy.optimize import minimize
        self._read_widgets()
        s = self.state
        N_x, N_y = s["N_x"], s["N_y"]
        n_free = max(0, N_x - 1) + max(0, N_y - 1)
        if n_free == 0:
            self.lbl_status.setText("Nur ein Ton je Achse - keine Phasen frei.")
            return
        self.lbl_status.setText("baue Operator fuer das Ruhefenster ...")
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        QApplication.processEvents()

        cxs, cys, f_spots, _, _, _, _ = compute_centers_and_freqs(
            N_x, N_y, s["width_x"], s["width_y"], s["f1"], s["f2"], s["offset"])
        f0 = fundamental_beat_frequency(f_spots)
        if f0 <= 0:
            self.lbl_status.setText("Kein gemeinsames f_0 - Optimierung nicht definiert.")
            return
        amp = amp_spots_from_ratios(s["r_x"], s["r_y"], N_x, N_y)
        win_eff = s["win"] * (s["airy_factor"] if s["use_airy"] else 1.0)
        xg, yg, Xg, Yg = compute_grid(cxs, cys, win_eff, 110)
        Fg = build_field_stack(Xg, Yg, cxs, cys, amp, s["win"], s["use_airy"], s["airy_factor"])
        k = beat_orders(f_spots, f0)
        mean0, _ = time_stats_exact(Fg, k, np.zeros(len(f_spots)))
        mask = self._target_mask(mean0, Xg, Yg, cxs, cys, xg, yg)
        obj = WindowObjective(Fg, k, mask, f0, s["t_win"])
        degen_g = degenerate_groups(f_spots, f0)
        t_w = np.linspace(0.0, s["t_win"], 50)
        useries = UniformitySeries(Fg, f_spots, mask, t_w, k=k)
        mode = self.cmb_opt_target.currentIndex()
        # Beide Groessen sind dimensionslose relative Masse in derselben
        # Groessenordnung; fuer "beides" reicht deshalb die schlichte Summe.
        def cost(v):
            px = np.concatenate(([0.0], v[:N_x - 1])) if N_x > 1 else np.zeros(1)
            py = np.concatenate(([0.0], v[N_x - 1:])) if N_y > 1 else np.zeros(1)
            ph = spot_phases_from_tones(px, py, N_x, N_y)
            pen = (2.0 * quadrature_penalty(ph, degen_g)
                   if self.cb_quad.isChecked() else 0.0)
            if mode == 1:
                return useries.mean_u(ph) + pen
            if mode == 0:
                return obj.rms(ph) + pen
            # "beides": konstante Fallentiefe UND gleichmaessige Tiefen.
            # Bewusst nicht rms(<I>) + U - das zaehlt den Formfehler doppelt
            # und landet auf einem Kompromiss, der beide Einzelziele verfehlt.
            lvl, u = useries.level_and_u(ph)
            return float(np.hypot(lvl, u)) + pen

        start = cost(np.concatenate((s["phase_x"][1:], s["phase_y"][1:])))
        self.lbl_status.setText(
            f"suche Phasen fuer ein ruhiges Fenster von {s['t_win'] * 1e6:.2f} us ...")
        QApplication.processEvents()
        rng = np.random.default_rng(7)
        best_f, best_v = cost(np.zeros(n_free)), np.zeros(n_free)
        for n in range(300):
            r = minimize(cost, rng.uniform(0, 2 * np.pi, n_free), method="Nelder-Mead",
                         options=dict(maxiter=2500, xatol=1e-5, fatol=1e-11))
            if r.fun < best_f:
                best_f, best_v = float(r.fun), r.x
            if n % 60 == 0:
                QApplication.processEvents()
        px = np.concatenate(([0.0], best_v[:N_x - 1])) if N_x > 1 else np.zeros(1)
        py = np.concatenate(([0.0], best_v[N_x - 1:])) if N_y > 1 else np.zeros(1)
        self._write_phase_fields(px, py)
        ph_best = spot_phases_from_tones(px, py, N_x, N_y)
        st = window_stats(Fg, k, ph_best, mask, f_spots, s["t_win"])
        u_ref = uniformity_of(mean0, mask)
        self._opt_note = (
            f"Ruhefenster {s['t_win'] * 1e6:.2f} us "
            f"[{self.cmb_opt_target.currentText()}]: {start * 100:.0f} % -> {best_f * 100:.1f} %\n"
            f"   Abweichung von <I> {st['total'] * 100:.0f} % "
            f"(Niveau {st['level'] * 100:.0f} %, Form {st['shape'] * 100:.0f} %, "
            f"Tiefe {st['alpha_min']:.2f}..{st['alpha_max']:.2f} x)   |   "
            f"U(t) im Fenster {useries.mean_u(ph_best) * 100:.0f} % "
            f"gegen U(<I>) = {u_ref * 100:.1f} %")
        self.recompute()

    def _on_minimize_peak(self):
        """Sucht die Tonphasen mit der kleinsten Spitzenintensitaet.

        Bewertet wird auf einem bewusst groben Grid und mit wenigen Frames -
        die Spitze ist eine glatte Funktion der Phasen, die Optimierung
        braucht dafuer keine volle Aufloesung. Der gefundene Satz wird in die
        Eingabefelder geschrieben und dann normal durchgerechnet."""
        from scipy.optimize import minimize
        self._read_widgets()
        s = self.state
        self.lbl_status.setText("suche Phasen mit minimaler Spitze ...")
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        QApplication.processEvents()

        cxs, cys, f_spots, _, _, _, _ = compute_centers_and_freqs(
            s["N_x"], s["N_y"], s["width_x"], s["width_y"], s["f1"], s["f2"], s["offset"])
        f0 = fundamental_beat_frequency(f_spots)
        if f0 <= 0:
            self.lbl_status.setText("Kein Beating - nichts zu minimieren.")
            return
        amp = amp_spots_from_ratios(s["r_x"], s["r_y"], s["N_x"], s["N_y"])
        win_eff = s["win"] * (s["airy_factor"] if s["use_airy"] else 1.0)
        _, _, Xc, Yc = compute_grid(cxs, cys, win_eff, 70)
        Fc = build_field_stack(Xc, Yc, cxs, cys, amp, s["win"], s["use_airy"], s["airy_factor"])
        tc = np.arange(120) / 120 * (1.0 / f0)
        N_x, N_y = s["N_x"], s["N_y"]

        degen_g = degenerate_groups(f_spots, f0)
        peak_scale = float(np.einsum("sij,sij->ij", Fc, Fc).max())

        def peak(v):
            px = np.concatenate(([0.0], v[:N_x - 1])) if N_x > 1 else np.zeros(1)
            py = np.concatenate(([0.0], v[N_x - 1:])) if N_y > 1 else np.zeros(1)
            ph = spot_phases_from_tones(px, py, N_x, N_y)
            pen = (2.0 * peak_scale * quadrature_penalty(ph, degen_g)
                   if self.cb_quad.isChecked() else 0.0)
            return float(intensity_cube(Fc, f_spots, ph, tc).max()) + pen

        n_free = max(0, N_x - 1) + max(0, N_y - 1)
        if n_free == 0:
            self.lbl_status.setText("Nur ein Ton je Achse - keine Phasen frei.")
            return
        rng = np.random.default_rng(0)
        best_v, best_f = np.zeros(n_free), peak(np.zeros(n_free))
        for _ in range(12):                      # Mehrfachstart gegen lokale Minima
            v0 = rng.uniform(0, 2 * np.pi, n_free)
            r = minimize(peak, v0, method="Nelder-Mead",
                         options=dict(maxiter=400, xatol=2e-3, fatol=1e-4))
            if r.fun < best_f:
                best_f, best_v = float(r.fun), r.x
            QApplication.processEvents()
        px = np.concatenate(([0.0], best_v[:N_x - 1])) if N_x > 1 else np.zeros(1)
        py = np.concatenate(([0.0], best_v[N_x - 1:])) if N_y > 1 else np.zeros(1)
        self._write_phase_fields(px, py)
        self.recompute()

    def _group_actions(self):
        g = QGroupBox("Aktionen")
        lay = QVBoxLayout(g)
        self.cb_auto = QCheckBox("Automatisch neu rechnen")
        self.cb_auto.setChecked(self.state["auto_update"])
        self.cb_auto.stateChanged.connect(self._on_auto_changed)
        lay.addWidget(self.cb_auto)
        self.btn_update = QPushButton("Neu berechnen")
        self.btn_update.clicked.connect(lambda: self.recompute())
        lay.addWidget(self.btn_update)
        lay.addWidget(QLabel("Panel oben rechts"))
        self.cmb_panel = QComboBox()
        self.cmb_panel.addItems([
            "Orts-Zeit-Karte I(x, t)",
            "Ueberhoehung n_eff = I_max / <I>",
            "Modulationstiefe (I_max-I_min)/(I_max+I_min)",
            "zeitliche Variation sigma_t / <I>",
            "Spektrum der Schwebung",
            "Uniformity U(t) der drei Gebiete",
        ])
        self.cmb_panel.setToolTip(
            "n_eff ist die Zahl der am jeweiligen Ort wirksam ueberlappenden\n"
            "Toene: 1, wo ein einzelner Spot dominiert, bis hinauf zur\n"
            "Spotzahl im Plateau. Sie ist zugleich der Faktor, um den die\n"
            "Momentanintensitaet bei Rephasierung ueber dem Zeitmittel liegt.")
        self.cmb_panel.currentIndexChanged.connect(lambda _: self.draw_frame(full=True))
        self.cb_fastdraw = QCheckBox("Schnellzeichnen beim Abspielen")
        self.cb_fastdraw.setChecked(True)
        self.cb_fastdraw.setToolTip(
            "Beim Weiterschalten werden nur die Daten der vorhandenen\n"
            "Zeichenobjekte ausgetauscht, statt alle vier Achsen neu\n"
            "aufzubauen - etwa doppelt so schnell.\n\n"
            "Falls die Anzeige sich merkwuerdig verhaelt: Haken weg, dann\n"
            "wird jedes Bild vollstaendig neu gezeichnet.")
        self.cb_fastdraw.stateChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cb_fastdraw)

        self.cb_live = QCheckBox("U(t) live mitschreiben")
        self.cb_live.setChecked(True)
        self.cb_live.setToolTip(
            "Im Panel 'Uniformity U(t)' wird die Kurve nur bis zum aktuellen\n"
            "Zeitpunkt gezeichnet und waechst mit der Animation mit; der\n"
            "restliche Verlauf steht blass dahinter. Die Zahlen im Titel sind\n"
            "die Momentanwerte.")
        self.cb_live.stateChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cb_live)
        lay.addWidget(self.cmb_panel)
        lay.addWidget(QLabel("Farbskala"))
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems([
            "fest: 99.5-Perzentil ueber die Zeit",
            "fest: Maximum ueber die Zeit",
            "fest: Maximum des Zeitmittels",
            "pro Frame neu",
        ])
        self.cmb_scale.setToolTip(
            "Stehen alle Tonphasen auf 0, rephasieren alle Toene einmal pro\n"
            "Grundperiode zu einem kurzen Puls, der um ein Vielfaches ueber\n"
            "dem Zeitmittel liegt. Eine feste Skala auf DIESES Maximum laesst\n"
            "alle uebrigen Frames fast schwarz - deshalb ist das Perzentil\n"
            "die Voreinstellung. 'pro Frame' zeigt jede Aufnahme voll\n"
            "ausgesteuert, macht Frames aber untereinander unvergleichbar.")
        self.cmb_scale.currentIndexChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cmb_scale)
        self.btn_save = QPushButton("Ansicht als PNG speichern")
        self.btn_save.clicked.connect(self._on_save_clicked)
        lay.addWidget(self.btn_save)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_status)
        return g

    # --------------------------------------------------------
    # Eingaben -> state
    # --------------------------------------------------------
    def _sync_winmode_enabled(self):
        out = self.state["win_mode"] == "output"
        self.sp_win.setEnabled(out)
        self.sp_win_in.setEnabled(not out)

    def _sync_width_enabled(self):
        self.sp_width_y.setEnabled(not self.cb_link_width.isChecked())

    def _on_link_width_changed(self, _):
        self.state["link_width"] = self.cb_link_width.isChecked()
        self._sync_width_enabled()
        self._on_param_changed()

    def _on_winmode_changed(self, idx):
        self.state["win_mode"] = "output" if idx == 0 else "input"
        self._sync_winmode_enabled()
        self._on_param_changed()

    def _read_widgets(self):
        s = self.state
        s["N_x"] = self.sp_nx.value()
        s["N_y"] = self.sp_ny.value()
        s["use_airy"] = self.cmb_profile.currentIndex() == 1
        s["airy_factor"] = self.sp_airy.value()
        s["link_width"] = self.cb_link_width.isChecked()
        s["width_x"] = self.sp_width.value() * 1e6
        if s["link_width"]:
            s["width_y"] = s["width_x"]
            self.sp_width_y.blockSignals(True)
            self.sp_width_y.setValue(s["width_x"] * 1e-6)
            self.sp_width_y.blockSignals(False)
        else:
            s["width_y"] = self.sp_width_y.value() * 1e6
        s["lambda_opt"] = self.sp_lambda.value() * 1e-9
        s["offset"] = self.sp_offset.value() * 1e6
        s["f1"] = self.sp_f1.value() * 1e-3
        s["f2"] = self.sp_f2.value() * 1e-3
        s["r_x"] = self.sp_rx.value()
        s["r_y"] = self.sp_ry.value()
        s["grid_n"] = self.sp_grid.value()
        s["n_periods"] = self.sp_periods.value()
        s["frames_per_period"] = self.sp_fpp.value()
        s["phase_scope"] = "tone" if self.cmb_phase_scope.currentIndex() == 0 else "free"
        if len(self.phase_spins_x) != s["N_x"] or len(self.phase_spins_y) != s["N_y"]:
            self._rebuild_phase_fields()
            if s["phase_scope"] == "free":
                rng = np.random.default_rng()
                s["phase_free"] = rng.uniform(0.0, 2 * np.pi, s["N_x"] * s["N_y"])
        s["t_win"] = self.sp_twin.value() * 1e-6
        s["phase_x"] = np.radians([sp.value() for sp in self.phase_spins_x])
        s["phase_y"] = np.radians([sp.value() for sp in self.phase_spins_y])

        # Waist: die jeweils aktive Groesse ist die Stellgroesse, die andere
        # wird nachgezogen und im Feld angezeigt.
        if s["win_mode"] == "output":
            s["win"] = self.sp_win.value() * 1e-6
            s["win_in"] = conjugate_waist(s["win"], s["f1"], s["f2"], s["lambda_opt"])
            self.sp_win_in.blockSignals(True)
            self.sp_win_in.setValue(s["win_in"] * 1e3)
            self.sp_win_in.blockSignals(False)
        else:
            s["win_in"] = self.sp_win_in.value() * 1e-3
            s["win"] = conjugate_waist(s["win_in"], s["f1"], s["f2"], s["lambda_opt"])
            self.sp_win.blockSignals(True)
            self.sp_win.setValue(s["win"] * 1e6)
            self.sp_win.blockSignals(False)

    def _on_param_changed(self, *args):
        if self._building:
            return
        if self.state["auto_update"]:
            self.recompute()
        else:
            self.lbl_status.setText("Parameter geaendert - 'Neu berechnen' druecken.")

    def _on_auto_changed(self, _):
        self.state["auto_update"] = self.cb_auto.isChecked()
        if self.state["auto_update"]:
            self.recompute()

    def _on_speed_changed(self, val):
        if self.timer.isActive():
            self.timer.start(max(1, int(1000 / val)))

    def _on_play_clicked(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start(max(1, int(1000 / self.slider_speed.value())))
            self.btn_play.setText("Pause")

    def _on_timer_tick(self):
        n = self.cache.get("n_frames", 0)
        if n == 0:
            return
        self.frame_idx = (self.frame_idx + 1) % n
        self.slider_t.blockSignals(True)
        self.slider_t.setValue(self.frame_idx)
        self.slider_t.blockSignals(False)
        self.draw_frame()

    def _on_time_slider(self, val):
        self.frame_idx = int(val)
        self.draw_frame()

    def _on_canvas_click(self, event):
        """Klick in die 2D-Karte verschiebt das Fadenkreuz - Schnitt, Orts-Zeit-
        Karte und I(t) beziehen sich danach auf diesen Punkt."""
        if event.inaxes is not self.ax_main or not self.cache:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_um = self.cache["x"] * 1e6
        y_um = self.cache["y"] * 1e6
        self.cache["col"] = int(np.argmin(np.abs(x_um - event.xdata)))
        self.cache["row"] = int(np.argmin(np.abs(y_um - event.ydata)))
        self._rebuild_traces()
        self.draw_frame(full=True)

    # --------------------------------------------------------
    # Rechnen
    # --------------------------------------------------------
    def recompute(self):
        self._read_widgets()
        s = self.state

        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        centers_x, centers_y, f_spots, r_center_x, r_center_y, fx_freq, fy_freq = \
            compute_centers_and_freqs(s["N_x"], s["N_y"], s["width_x"], s["width_y"],
                                      s["f1"], s["f2"], s["offset"])
        amp_spots = amp_spots_from_ratios(s["r_x"], s["r_y"], s["N_x"], s["N_y"])
        n_spots = len(f_spots)

        win_eff = s["win"] * (s["airy_factor"] if s["use_airy"] else 1.0)
        x, y, X, Y = compute_grid(centers_x, centers_y, win_eff, s["grid_n"])

        F = build_field_stack(X, Y, centers_x, centers_y, amp_spots,
                              s["win"], s["use_airy"], s["airy_factor"])

        f0 = fundamental_beat_frequency(f_spots)
        degen = degenerate_groups(f_spots, f0)
        # Zeitfenster. f0 > 0: das Signal ist streng periodisch mit T0 = 1/f0,
        # das Fenster umfasst dann ganze Perioden und das Zeitmittel ueber das
        # Fenster ist exakt das Zeitmittel ueberhaupt.
        # f0 == 0 bei vorhandenen Beats: die Differenzfrequenzen sind
        # inkommensurabel (getrennte widths), es gibt gar keine gemeinsame
        # Periode. Dann nehmen wir 1/f_min als Anzeigemassstab und sagen das
        # auch dazu - das Fenstermittel ist dann nur eine Naeherung.
        beats = unique_beat_frequencies(f_spots)
        need_fpp = min_frames_per_period(f_spots, f0)
        n_frames = s["n_periods"] * s["frames_per_period"]
        periodic = f0 > 0
        if periodic:
            T0 = 1.0 / f0
        elif beats.size:
            T0 = 1.0 / beats[0]
        else:
            T0 = float("nan")
        if np.isfinite(T0):
            # offenes Intervall: der letzte Frame ist NICHT identisch mit dem
            # ersten, sonst ruckelt die Schleife und das Zeitmittel bekaeme
            # einen Punkt doppelt.
            t = np.arange(n_frames) / n_frames * (s["n_periods"] * T0)
        else:
            t = np.zeros(1)
            n_frames = 1

        if s["phase_scope"] == "free":
            free = np.asarray(s["phase_free"], dtype=float)
            if free.size != len(f_spots):
                free = np.random.default_rng().uniform(0.0, 2 * np.pi, len(f_spots))
                s["phase_free"] = free
            phases = free
        else:
            phases = spot_phases_from_tones(s["phase_x"], s["phase_y"], s["N_x"], s["N_y"])

        cost = n_frames * len(f_spots) * X.size
        if cost > 4e9:
            QMessageBox.warning(self, "Zu gross",
                                "Diese Kombination aus Grid, Frames und Tonzahl waere sehr "
                                "langsam. Bitte Grid-Aufloesung oder Frames/Periode "
                                "verkleinern.")
            return
        self.lbl_status.setText("rechne ...")
        QApplication.processEvents()

        cube = intensity_cube(F, f_spots, phases, t)

        # Referenzen. I_avg ist das Zeitmittel; ueber eine ganze Grundperiode
        # muss es exakt der inkohaerenten Summe sum_s A_s^2 |u_s|^2 der
        # bisherigen GUIs entsprechen - genau das prueft resid.
        I_avg = cube.mean(axis=0).astype(np.float64)
        I_incoh = np.einsum("sij,sij->ij", F, F)
        denom = np.max(I_incoh) if np.max(I_incoh) > 0 else 1.0
        resid = float(np.max(np.abs(I_avg - I_incoh)) / denom)

        norm = float(np.max(I_incoh)) if np.max(I_incoh) > 0 else 1.0

        # Ueberhoehung und Modulationstiefe. n_eff = I_max/<I> ist die Zahl der
        # am jeweiligen Ort wirksam ueberlappenden Toene: 1, wo ein einzelner
        # Spot dominiert, bis hinauf zur Spotzahl, wo alle gleich beitragen.
        k_orders = beat_orders(f_spots, f0) if f0 > 0 else np.zeros(len(f_spots), int)
        mean_exact, var_exact = time_stats_exact(F, k_orders, phases)
        sigma_rel = np.sqrt(np.maximum(var_exact, 0.0)) / np.maximum(mean_exact, 1e-300)

        I_max_map = cube.max(axis=0).astype(np.float64)
        I_min_map = cube.min(axis=0).astype(np.float64)
        n_eff = I_max_map / np.maximum(I_avg, 1e-300)
        depth = (I_max_map - I_min_map) / np.maximum(I_max_map + I_min_map, 1e-300)
        plateau = I_avg > 0.5 * I_avg.max() if I_avg.max() > 0 else np.zeros_like(I_avg, bool)
        if plateau.any():
            try:
                obj_spec = VariationObjective(F, k_orders, plateau)
                spectrum = obj_spec.components(phases)
            except Exception:
                spectrum = {}
            sigma_rms = float(np.sqrt(np.mean(sigma_rel[plateau] ** 2)))
            try:
                win_st = window_stats(F, k_orders, phases, plateau, f_spots, s["t_win"])
            except Exception:
                win_st = None
            # Uniformity U(t) fuer die drei Auswerte-Gebiete. Die Referenz ist
            # jeweils U(<I>) - genau die Zahl, die die uebrigen GUIs und die
            # Scan-Pipeline ausgeben.
            sites_mask = np.zeros(I_avg.shape, bool)
            for cxi, cyi in zip(centers_x, centers_y):
                sites_mask[int(np.argmin(np.abs(y - cyi))),
                           int(np.argmin(np.abs(x - cxi)))] = True
            r_circ = self.sp_opt_radius.value() * 1e-6
            circ_mask = ((X - float(np.mean(centers_x))) ** 2
                         + (Y - float(np.mean(centers_y))) ** 2) <= r_circ ** 2
            u_regions = {
                "Plateau": plateau,
                "Spot-Zentren": sites_mask,
                f"Kreis r={r_circ * 1e6:.1f} um": circ_mask,
            }
            u_series, u_ref = {}, {}
            for rname, rmask in u_regions.items():
                if not rmask.any():
                    continue
                try:
                    u_series[rname] = UniformitySeries(F, f_spots, rmask, t).series(phases)
                    u_ref[rname] = uniformity_of(mean_exact, rmask)
                except Exception:
                    pass
            n_eff_med = float(np.median(n_eff[plateau]))
            depth_med = float(np.median(depth[plateau]))
            depth_min = float(depth[plateau].min())
        else:
            n_eff_med = depth_med = depth_min = sigma_rms = float("nan")
            spectrum = {}
            win_st = None
            u_series, u_ref = {}, {}
        if s["phase_scope"] == "tone":
            crest_x = crest_factor(fx_freq, s["phase_x"])
            crest_y = crest_factor(fy_freq, s["phase_y"])
        else:
            # Freie Spot-Phasen lassen sich durch kein RF-Signal erzeugen -
            # ein Crest-Faktor der Toene ist dann nicht definiert.
            crest_x = crest_y = float("nan")

        self.cache = {
            "x": x, "y": y, "X": X, "Y": Y,
            "centers_x": centers_x, "centers_y": centers_y, "f_spots": f_spots,
            "r_center_x": r_center_x, "r_center_y": r_center_y,
            "fx_freq": fx_freq, "fy_freq": fy_freq,
            "amp_spots": amp_spots, "t": t, "n_frames": n_frames,
            "cube": cube, "I_avg": I_avg, "I_incoh": I_incoh,
            "norm": norm, "resid": resid, "T0": T0, "f0": f0, "degen": degen,
            "periodic": periodic, "beats": beats, "need_fpp": need_fpp,
            "I_max_map": I_max_map, "I_min_map": I_min_map,
            "n_eff": n_eff, "depth": depth, "plateau": plateau,
            "sigma_rel": sigma_rel, "sigma_rms": sigma_rms, "spectrum": spectrum,
            "win_st": win_st, "t_win": s["t_win"],
            "u_series": u_series, "u_ref": u_ref,
            "mean_exact": mean_exact, "var_exact": var_exact,
            "n_eff_med": n_eff_med, "depth_med": depth_med, "depth_min": depth_min,
            "crest_x": crest_x, "crest_y": crest_y, "phases": phases,
            "row": int(np.argmin(np.abs(y - r_center_y))),
            "col": int(np.argmin(np.abs(x - r_center_x))),
            "cube_max": float(cube.max()),
            "cube_p995": float(np.percentile(cube, 99.5)),
        }
        self._rebuild_traces()

        self.frame_idx = min(self.frame_idx, n_frames - 1)
        self.slider_t.blockSignals(True)
        self.slider_t.setMaximum(max(0, n_frames - 1))
        self.slider_t.setValue(self.frame_idx)
        self.slider_t.blockSignals(False)

        crest_txt = ("Crest-Faktor RF: nicht definiert (freie Spot-Phasen)"
                     if not np.isfinite(crest_x)
                     else f"Crest-Faktor RF: x {crest_x:.2f}, y {crest_y:.2f}")
        self.lbl_phase.setText(
            f"Spitze: {cube.max() / norm:.2f} x max<I>   |   " + crest_txt + "\n"
            f"Im Plateau: sigma_t/<I> = {sigma_rms * 100:.1f} %,  "
            f"n_eff (Ueberlapp) = {n_eff_med:.1f},  "
            f"Modulationstiefe = {depth_med * 100:.1f} %\n"
            + (f"Ruhefenster {s['t_win'] * 1e6:.2f} us: {win_st['total'] * 100:.0f} % von <I> "
               f"(Niveau {win_st['level'] * 100:.0f} %, Form {win_st['shape'] * 100:.0f} %, "
               f"Tiefe {win_st['alpha_min']:.2f}..{win_st['alpha_max']:.2f} x)\n"
               if win_st else "")
            + (f"Entartete Paare: Phasendifferenz "
               f"{np.degrees(phases[degen[0][0]] - phases[degen[0][1]]) % 180:.1f} Grad "
               f"-> statischer Anteil {resid * 100:.2f} % "
               f"({'in Quadratur, Zeitmittel = inkohaerente Summe' if resid < 1e-6 else 'nicht in Quadratur'})\n"
               if degen else "")
            + (self._opt_note + "\n" if self._opt_note else "")
            + "Phasen koennen sigma_t/<I> senken (Kreuzterme gleicher "
              "Differenzfrequenz loeschen sich teilweise aus), aber nicht auf null.")
        self._opt_note = ""
        self._update_labels()
        self.draw_frame(full=True)
        if degen:
            note = (f"Zeitmittel weicht um {resid * 100:.3f} % vom inkohaerenten Bild ab - "
                    f"das ist KEIN Rechenfehler, sondern die statische Interferenz der "
                    f"frequenzentarteten Spots.")
        else:
            note = (f"Zeitmittel == inkohaerente Summe bis auf {resid:.1e} - "
                    f"das bisherige Bild ist exakt das Zeitmittel.")
        if periodic and s["frames_per_period"] < need_fpp:
            note += (f"\nWARNUNG: {s['frames_per_period']} Frames/Periode reichen nicht. "
                     f"Die schnellste Beat-Frequenz ist {beats[-1] * 1e-3:.1f} kHz = "
                     f"{beats[-1] / f0:.0f} x f_0; fuer Nyquist braucht es mindestens "
                     f"{need_fpp}. Das Zeitmittel und die Huellkurven sind sonst falsch "
                     f"(Aliasing).")
        self.lbl_status.setText(
            f"fertig: {n_frames} Frames, {len(f_spots)} Spots, Grid {s['grid_n']}^2.\n" + note)
        if periodic and s["frames_per_period"] < need_fpp:
            self.lbl_status.setStyleSheet("color: #a00; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        if was_running:
            self.timer.start(max(1, int(1000 / self.slider_speed.value())))

    def _rebuild_traces(self):
        """Schnitte, Orts-Zeit-Karte und I(t) fuer das aktuelle Fadenkreuz."""
        c = self.cache
        if not c:
            return
        row, col = c["row"], c["col"]
        c["st_map"] = c["cube"][:, row, :]        # I(x, t) entlang des x-Schnitts
        c["trace"] = c["cube"][:, row, col]       # I(t) am Fadenkreuz
        c["cut_x_min"] = c["cube"][:, row, :].min(axis=0)
        c["cut_x_max"] = c["cube"][:, row, :].max(axis=0)
        c["cut_y_min"] = c["cube"][:, :, col].min(axis=0)
        c["cut_y_max"] = c["cube"][:, :, col].max(axis=0)

    def _update_labels(self):
        c, s = self.cache, self.state
        fx = c["fx_freq"] * 1e-6
        fy = c["fy_freq"] * 1e-6
        self.lbl_freqs.setText(
            "f_x = " + ", ".join(f"{v:.4f}" for v in fx) + " MHz\n"
            "f_y = " + ", ".join(f"{v:.4f}" for v in fy) + " MHz\n"
            f"{len(c['f_spots'])} Spots, Pitch der Toene "
            f"{abs(c['centers_x'][-1] - c['centers_x'][0]) * 1e6 / max(1, s['N_x'] - 1):.3f} um in x")

        self.lbl_amps.setText(
            "amp_x = [" + ", ".join(f"{v:g}" for v in amps_from_ratio(s["r_x"], s["N_x"])) + "]\n"
            "amp_y = [" + ", ".join(f"{v:g}" for v in amps_from_ratio(s["r_y"], s["N_y"])) + "]\n"
            "(Intensitaets-Gewichte; das Feld traegt die Wurzel)")

        beats = c["beats"]
        if c["periodic"]:
            shown = ", ".join(f"{b * 1e-3:.1f}" for b in beats[:8])
            more = " ..." if len(beats) > 8 else ""
            self.lbl_beat.setText(
                f"Grundfrequenz f_0 = ggT aller Differenzen = {c['f0'] * 1e-3:.3f} kHz\n"
                f"Grundperiode T_0 = {c['T0'] * 1e6:.3f} us\n"
                f"Fenster = {s['n_periods']} x T_0 = {s['n_periods'] * c['T0'] * 1e6:.2f} us\n"
                f"Beat-Frequenzen [kHz]: {shown}{more}")
        elif beats.size:
            shown = ", ".join(f"{b * 1e-3:.2f}" for b in beats[:8])
            more = " ..." if len(beats) > 8 else ""
            self.lbl_beat.setText(
                "NICHT PERIODISCH: die Differenzfrequenzen haben kein gemeinsames\n"
                "Vielfaches (getrennte widths). Angezeigt wird 1/f_min als Massstab:\n"
                f"T_ref = {c['T0'] * 1e6:.3f} us,  Fenster = {s['n_periods']} x T_ref = "
                f"{s['n_periods'] * c['T0'] * 1e6:.2f} us\n"
                f"Das Fenstermittel ist damit nur eine Naeherung.\n"
                f"Beat-Frequenzen [kHz]: {shown}{more}")
        else:
            self.lbl_beat.setText("Kein Beating: nur ein Spot bzw. width = 0.")

        # Entscheidend ist nicht die Groesse der Modulation, sondern ob eine
        # Schwebungslinie auf nu_r oder 2*nu_r faellt: das Spektrum ist
        # diskret, dazwischen bekommt die Falle praktisch keine Leistung.
        nu = self.sp_nu_r.value() * 1e3
        d1, d2, crit = resonance_check(beats, nu)
        if np.isfinite(d1):
            txt = (f"Naechste Schwebungslinie zu nu_r ({nu * 1e-3:.1f} kHz): "
                   f"{d1 * 1e-3:.1f} kHz entfernt;  zu 2*nu_r: {d2 * 1e-3:.1f} kHz.")
            if crit:
                self.lbl_res.setText(
                    "ACHTUNG - " + txt + "\n"
                    "Eine Linie liegt auf einer Fallenresonanz. Das Spektrum ist "
                    "diskret; dass eine Linie ausgerechnet dort sitzt, ist der "
                    "gefaehrliche Fall - nicht die Modulationstiefe. width so "
                    "waehlen, dass kein Vielfaches von f_0 dorthin faellt.")
                self.lbl_res.setStyleSheet("color: #a00; font-size: 10px; font-weight: bold;")
            else:
                self.lbl_res.setText(
                    txt + "\nKeine Linie auf einer Fallenresonanz - trotz voller "
                    "Modulationstiefe bekommt die Falle dort kaum Leistung.")
                self.lbl_res.setStyleSheet("color: #060; font-size: 10px;")
        else:
            self.lbl_res.setText("")

        degen = c.get("degen", [])
        if degen:
            lines = []
            for grp in degen:
                pos = " = ".join(f"({c['centers_x'][i] * 1e6:.2f}, "
                                 f"{c['centers_y'][i] * 1e6:.2f})" for i in grp)
                lines.append(f"  {pos} um  bei {c['f_spots'][grp[0]] * 1e-6:.5f} MHz")
            self.lbl_degen.setText(
                "ACHTUNG - frequenzentartete Spots:\n" + "\n".join(lines) +
                "\nDiese Paare haben einen Kreuzterm bei 0 Hz. Der laeuft nie um "
                "und mittelt sich nie weg: STATISCHE Interferenz, die die "
                "inkohaerenten GUIs nicht sehen. Aufheben laesst sich das nur "
                "durch UNTERSCHIEDLICHE width in x und y - ein konstanter "
                "Frequenzversatz einer Achse hilft nicht, er kuerzt sich aus "
                "jeder Differenz heraus.")
            self.lbl_degen.setStyleSheet(
                "color: #a00; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_degen.setText("Keine frequenzentarteten Spots - jeder Kreuzterm "
                                   "laeuft um, das Zeitmittel ist exakt das "
                                   "inkohaerente Bild.")
            self.lbl_degen.setStyleSheet("color: #060; font-size: 10px;")

    # --------------------------------------------------------
    # Zeichnen
    # --------------------------------------------------------
    def _u_title(self, c, k):
        """Titel des U(t)-Panels mit den MOMENTANwerten - die Zahl, die man
        beim Zuschauen eigentlich ablesen will."""
        us = c.get("u_series", {})
        if not us:
            return "Uniformity im Zeitverlauf"
        # Kurze, LAENGENSTABILE Beschriftung: beim Blitting laeuft das Layout
        # nicht mehr nach, ein waschsender Titel wuerde ueber den Rand laufen.
        short = {"Plateau": "Plat", "Spot-Zentren": "Zentr"}
        now = " ".join(f"{short.get(rn, 'Kreis')} {u[k] * 100:3.0f}%" for rn, u in us.items())
        return "U jetzt:  " + now + "\n(gepunktet: aus dem Zeitmittel)"

    def _fast_frame(self, c, k):
        """Nur die Daten austauschen, statt alle vier Achsen neu aufzubauen.

        Ein vollstaendiger Neuaufbau kostet rund 50 ms, also hoechstens 20
        Bilder je Sekunde - fuer eine live mitlaufende Anzeige zu wenig. Auf
        diesem Weg bleiben nur die Aufrufe uebrig, die sich je Frame wirklich
        aendern. Gibt False zurueck, wenn etwas fehlt; dann zeichnet der
        Aufrufer vollstaendig."""
        if not self.cb_fastdraw.isChecked():
            return False
        art = self._art
        if not art or art.get("panel") != self.cmb_panel.currentIndex():
            return False
        # Der schnelle Pfad tauscht nur Daten aus. Passt die Form nicht mehr
        # zu den vorhandenen Objekten (Gitteraufloesung, Zeitraster, Fadenkreuz
        # geaendert), muss vollstaendig neu gezeichnet werden - sonst stuenden
        # Bild und Achsen nicht mehr zueinander.
        if (art.get("shape") != c["cube"].shape
                or art.get("rowcol") != (c["row"], c["col"])):
            return False
        try:
            norm = c["norm"]
            t_us = c["t"] * 1e6
            cut_row, cut_col = c["row"], c["col"]
            I_now = c["cube"][k] / norm
            art["main_im"].set_data(I_now)
            if self.cmb_scale.currentIndex() == 3:          # pro Frame neu
                art["main_im"].set_clim(0.0, max(float(I_now.max()), 1e-12))
            T0_us = c["T0"] * 1e6
            per = (f"   ({t_us[k] / T0_us:.3f} $T_0$)"
                   if np.isfinite(T0_us) and T0_us > 0 else "")
            art["main_title"].set_text(
                f"I(x, y, t) momentan   |   t = {t_us[k]:8.4f} us" + per +
                f"\nSpitze ueber die Zeit: {c['cube_max'] / norm:.2f} x das "
                f"Maximum des Zeitmittels")
            art["cut_now"].set_ydata(c["cube"][k, cut_row, :] / norm)
            tr = c["trace"] / norm
            art["time_cur"].set_xdata([t_us[k], t_us[k]])
            art["time_dot"].set_data([t_us[k]], [tr[k]])
            if "st_cur" in art:
                art["st_cur"].set_ydata([t_us[k], t_us[k]])
            if "u_cur" in art:
                art["u_cur"].set_xdata([t_us[k], t_us[k]])
                for ln, u in art.get("u_live", []):
                    ln.set_data(t_us[:k + 1], u[:k + 1] * 100)
                for dt, u in art.get("u_dot", []):
                    dt.set_data([t_us[k]], [u[k] * 100])
                if "u_title" in art:
                    art["u_title"].set_text(self._u_title(c, k))
        except Exception:
            return False
        self.canvas.draw_idle()
        return True

    def draw_frame(self, full=False):
        c, s = self.cache, self.state
        if not c:
            return
        k = self.frame_idx
        if not full and self._fast_frame(c, k):
            return
        self._art = {}
        x_um, y_um = c["x"] * 1e6, c["y"] * 1e6
        extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
        norm = c["norm"]
        t_us = c["t"] * 1e6
        cut_row, cut_col = c["row"], c["col"]   # NICHT row/col nennen:
        # "col" wurde in Panel-Schleifen schon zweimal als Farbvariable
        # wiederverwendet und hat den Spaltenindex ueberschrieben.
        I_now = c["cube"][k] / norm

        mode = self.cmb_scale.currentIndex()
        if mode == 0:
            vmax = c["cube_p995"] / norm
        elif mode == 1:
            vmax = c["cube_max"] / norm
        elif mode == 2:
            vmax = float(c["I_avg"].max()) / norm
        else:
            vmax = float(I_now.max())
        vmax = max(vmax, 1e-12)

        # ---- 2D-Karte ----
        self.ax_main.clear()
        self._art["main_im"] = self.ax_main.imshow(
            I_now, extent=extent, origin="lower", cmap="inferno",
            vmin=0.0, vmax=vmax, aspect="equal")
        self.ax_main.plot(c["centers_x"] * 1e6, c["centers_y"] * 1e6, "w+",
                          markersize=5, markeredgewidth=0.8, alpha=0.55,
                          label="Spot-Zentren")
        for grp in c.get("degen", []):
            self.ax_main.plot(c["centers_x"][grp] * 1e6, c["centers_y"][grp] * 1e6,
                              "o", mfc="none", mec="deepskyblue", ms=11, mew=1.6,
                              label="frequenzentartet")
        self.ax_main.axhline(y_um[cut_row], color="cyan", lw=0.8, alpha=0.8)
        self.ax_main.axvline(x_um[cut_col], color="magenta", lw=0.8, alpha=0.8)
        self.ax_main.set_xlabel("x (um)")
        self.ax_main.set_ylabel("y (um)")
        T0_us = c["T0"] * 1e6
        per_lbl = (f"   ({t_us[k] / T0_us:.3f} $T_0$)" if np.isfinite(T0_us) and T0_us > 0
                   else "")
        self._art["main_title"] = self.ax_main.set_title(
            f"I(x, y, t) momentan   |   t = {t_us[k]:8.4f} us" + per_lbl +
            f"\nSpitze ueber die Zeit: {c['cube_max'] / norm:.2f} x das Maximum "
            f"des Zeitmittels", fontsize=9)
        h, l = self.ax_main.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        self.ax_main.legend(uniq.values(), uniq.keys(), loc="upper right",
                            fontsize=7, framealpha=0.4)

        # ---- Panel oben rechts: Orts-Zeit-Karte, n_eff oder Modulationstiefe ----
        self.ax_st.clear()
        panel = self.cmb_panel.currentIndex()
        # Colorbar und Seitenverhaeltnis gehoeren nur zu den Karten-Panels
        # (1..3). Bleiben sie beim Umschalten stehen, quetschen sie die
        # Orts-Zeit-Karte bzw. das Balkendiagramm.
        if panel in (0, 4, 5):
            if self._panel_cbar is not None:
                try:
                    self._panel_cbar.remove()
                except Exception:
                    pass
                self._panel_cbar = None
            self.ax_st.set_aspect("auto")
        if panel == 0:
            t_hi = t_us[-1] if t_us[-1] > t_us[0] else t_us[0] + 1.0
            self.ax_st.imshow(c["st_map"] / norm, origin="lower", aspect="auto",
                              cmap="inferno", vmin=0.0, vmax=vmax,
                              extent=[x_um[0], x_um[-1], t_us[0], t_hi])
            self._art["st_cur"] = self.ax_st.axhline(t_us[k], color="w", lw=1.0)
            self.ax_st.set_xlabel("x (um)", fontsize=8)
            self.ax_st.set_ylabel("t (us)", fontsize=8)
            self.ax_st.set_title(f"Orts-Zeit-Karte I(x, t) bei y = {y_um[cut_row]:.3f} um",
                                 fontsize=9)
        elif panel == 5:
            us = c.get("u_series", {})
            if us:
                # feste Farbreihenfolge, nie durchrotiert
                cols = ["#3b6ea5", "#b3402f", "#4c8b5b"]
                live = self.cb_live.isChecked()
                self._art["u_live"] = []
                self._art["u_dot"] = []
                for i, (rname, u) in enumerate(us.items()):
                    hue = cols[i % len(cols)]
                    if live:
                        # ganzer Verlauf blass als Orientierung, darueber die
                        # mitlaufende Spur bis zum aktuellen Zeitpunkt
                        self.ax_st.plot(t_us, u * 100, color=hue, lw=1.0, alpha=0.22)
                        (ln,) = self.ax_st.plot(t_us[:k + 1], u[:k + 1] * 100,
                                                color=hue, lw=1.8, label=rname)
                        (dt,) = self.ax_st.plot([t_us[k]], [u[k] * 100], "o",
                                                color=hue, ms=5)
                        self._art["u_live"].append((ln, u))
                        self._art["u_dot"].append((dt, u))
                    else:
                        self.ax_st.plot(t_us, u * 100, color=hue, lw=1.5, label=rname)
                    ref = c["u_ref"].get(rname)
                    if ref is not None and np.isfinite(ref):
                        self.ax_st.axhline(ref * 100, color=hue, ls=":", lw=1.1)
                        self.ax_st.annotate(f"U($\\langle I\\rangle$) = {ref * 100:.1f} %",
                                            (t_us[-1], ref * 100), xytext=(-2, 3),
                                            textcoords="offset points", ha="right",
                                            fontsize=6.5, color=hue)
                tw = c.get("t_win", 0.0) * 1e6
                if tw > 0:
                    self.ax_st.axvspan(0.0, min(tw, t_us[-1]), color="#3b6ea5",
                                       alpha=0.12, lw=0)
                self._art["u_cur"] = self.ax_st.axvline(t_us[k], color="k", lw=0.9)
                self.ax_st.set_xlabel("t (us)", fontsize=8)
                self.ax_st.set_ylabel("U = std/mean (%)", fontsize=8)
                self._art["u_title"] = self.ax_st.set_title(
                    self._u_title(c, k), fontsize=8.5)
                self.ax_st.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
                self.ax_st.set_ylim(bottom=0)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "keine Uniformity verfuegbar",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        elif panel == 4:
            spec = c.get("spectrum", {})
            if spec:
                ds = np.array(sorted(spec))
                freqs = ds * c["f0"] * 1e-3
                vals = np.array([spec[int(d)] * 100 for d in ds])
                width_bar = 0.7 * (freqs[1] - freqs[0]) if len(freqs) > 1 else 10.0
                self.ax_st.bar(freqs, vals, width=width_bar, color="#3b6ea5",
                               edgecolor="none")
                nu = self.sp_nu_r.value()
                # ACHTUNG: hier bloss keine Variable "col" verwenden - das ist
                # weiter oben der Spaltenindex des Fadenkreuzes.
                for f_mark, lab in ((nu, "$\\nu_r$"), (2 * nu, "$2\\nu_r$")):
                    if freqs[0] - width_bar <= f_mark <= freqs[-1] + width_bar:
                        self.ax_st.axvline(f_mark, color="#b3402f", lw=1.4, ls="--")
                        self.ax_st.annotate(lab, (f_mark, self.ax_st.get_ylim()[1]),
                                            xytext=(3, -2), textcoords="offset points",
                                            color="#b3402f", fontsize=9, ha="left", va="top")
                self.ax_st.set_ylim(0, max(vals.max() * 1.28, 1e-3))
                self.ax_st.set_xlabel("Schwebungsfrequenz (kHz)", fontsize=8)
                self.ax_st.set_ylabel("$\\sigma_d / \\langle I\\rangle$ (%)", fontsize=8)
                self.ax_st.set_title(
                    f"Spektrum der Schwebung im Plateau   "
                    f"(gesamt {c['sigma_rms'] * 100:.0f} %)", fontsize=9)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "kein Spektrum verfuegbar",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        else:
            if panel == 1:
                dat, cmap = c["n_eff"], "viridis"
                title = (f"Ueberhoehung $n_{{eff}} = I_{{max}}/\\langle I\\rangle$   "
                         f"(Plateau: {c['n_eff_med']:.1f})")
                lim = (1.0, max(2.0, float(len(c["f_spots"]))))
            elif panel == 2:
                dat, cmap = c["depth"] * 100.0, "magma"
                title = (f"Modulationstiefe in %   "
                         f"(Plateau: {c['depth_med'] * 100:.0f} %)")
                lim = (0.0, 100.0)
            else:
                dat, cmap = c["sigma_rel"] * 100.0, "cividis"
                title = (f"zeitliche Variation $\\sigma_t/\\langle I\\rangle$ in %   "
                         f"(Plateau: {c['sigma_rms'] * 100:.0f} %)")
                lim = (0.0, float(np.nanpercentile(c["sigma_rel"], 99) * 100))
            im = self.ax_st.imshow(dat, extent=extent, origin="lower", cmap=cmap,
                                   aspect="equal", vmin=lim[0], vmax=lim[1])
            self.ax_st.contour(x_um, y_um, c["I_avg"] / norm,
                               levels=[0.5 * c["I_avg"].max() / norm],
                               colors="w", linewidths=0.9, linestyles="--")
            self.ax_st.set_xlabel("x (um)", fontsize=8)
            self.ax_st.set_ylabel("y (um)", fontsize=8)
            self.ax_st.set_title(title, fontsize=8.5)
            if self._panel_cbar is not None:
                try:
                    self._panel_cbar.remove()
                except Exception:
                    pass
            self._panel_cbar = self.fig.colorbar(im, ax=self.ax_st, fraction=0.046, pad=0.03)
            self._panel_cbar.ax.tick_params(labelsize=7)
        self.ax_st.tick_params(labelsize=7)

        # ---- Schnitte mit Huellkurve ----
        self.ax_cut.clear()
        self.ax_cut.fill_between(x_um, c["cut_x_min"] / norm, c["cut_x_max"] / norm,
                                 color="tab:orange", alpha=0.22, lw=0,
                                 label="Min/Max ueber t")
        self.ax_cut.plot(x_um, c["I_avg"][cut_row, :] / norm, "k--", lw=1.0,
                         label="Zeitmittel")
        (self._art["cut_now"],) = self.ax_cut.plot(
            x_um, c["cube"][k, cut_row, :] / norm, color="tab:orange", lw=1.3,
            label="momentan")
        self.ax_cut.axvline(x_um[cut_col], color="magenta", lw=0.8, alpha=0.7)
        self.ax_cut.set_xlabel("x (um)", fontsize=8)
        self.ax_cut.set_ylabel("I / I_max", fontsize=8)
        self.ax_cut.set_title("x-Schnitt", fontsize=9)
        self.ax_cut.tick_params(labelsize=7)
        self.ax_cut.legend(fontsize=6.5, loc="upper right", framealpha=0.5)

        # ---- I(t) am Fadenkreuz ----
        self.ax_time.clear()
        tr = c["trace"] / norm
        self.ax_time.plot(t_us, tr, color="tab:blue", lw=1.0)
        tw = c.get("t_win", 0.0) * 1e6
        if tw > 0:
            self.ax_time.axvspan(0.0, min(tw, t_us[-1]), color="#3b6ea5", alpha=0.15, lw=0)
        self._art["time_cur"] = self.ax_time.axvline(t_us[k], color="k", lw=0.9)
        (self._art["time_dot"],) = self.ax_time.plot([t_us[k]], [tr[k]], "o",
                                                     color="tab:red", ms=4)
        mean_v = float(tr.mean())
        self.ax_time.axhline(mean_v, color="k", ls="--", lw=0.8)
        if np.isfinite(T0_us) and T0_us > 0:
            for p in range(1, s["n_periods"]):
                self.ax_time.axvline(p * T0_us, color="gray", ls=":", lw=0.6)
        lo, hi = float(tr.min()), float(tr.max())
        depth = (hi - lo) / (hi + lo) if (hi + lo) > 0 else 0.0
        self.ax_time.set_xlabel("t (us)", fontsize=8)
        self.ax_time.set_ylabel("I / I_max", fontsize=8)
        ws = c.get("win_st")
        extra = (f"\nFenster {tw:.1f} us: {ws['total'] * 100:.0f} % von "
                 f"$\\langle I\\rangle$" if ws else "")
        self.ax_time.set_title(
            f"I(t) am Fadenkreuz - Modulation {depth * 100:.0f} %" + extra, fontsize=8.5)
        self.ax_time.tick_params(labelsize=7)

        prof = "Airy" if s["use_airy"] else "Gauss"
        self.fig.suptitle(
            f"N = {s['N_x']}x{s['N_y']} Toene, {prof}, waist = {s['win'] * 1e6:.3f} um, "
            f"width = {s['width_x'] * 1e-6:.4f}/{s['width_y'] * 1e-6:.4f} MHz (x/y), "
            f"r_x = {s['r_x']:g}, r_y = {s['r_y']:g}, "
            f"lambda = {s['lambda_opt'] * 1e9:.1f} nm, f0 = {s['offset'] * 1e-6:.3f} MHz\n"
            + (f"Grundperiode T_0 = {T0_us:.3f} us  (f_0 = {c['f0'] * 1e-3:.3f} kHz), "
               f"Fenster = {s['n_periods']} Perioden" if c["periodic"]
               else f"nicht periodisch - Massstab T_ref = {T0_us:.3f} us, "
                    f"Fenster = {s['n_periods']} x T_ref"),
            fontsize=10)
        self._art["panel"] = self.cmb_panel.currentIndex()
        self._art["shape"] = c["cube"].shape
        self._art["rowcol"] = (c["row"], c["col"])
        self.canvas.draw_idle()

    # --------------------------------------------------------
    def _on_save_clicked(self):
        self.draw_frame(full=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        s = self.state
        name = (f"Beating_N{s['N_x']}x{s['N_y']}_"
                f"{'Airy' if s['use_airy'] else 'Gauss'}_"
                f"w{s['win'] * 1e6:.3f}um_width{s['width_x'] * 1e-6:.4f}MHz_"
                f"frame{self.frame_idx:04d}_{stamp}.png")
        path = self.out_dir / name
        try:
            self.fig.savefig(path, dpi=200)
            self.lbl_status.setText(f"gespeichert: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(exc))


def main():
    app = QApplication(sys.argv)
    win = BeatingMultitoneWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
