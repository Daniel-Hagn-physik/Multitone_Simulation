"""
Multitone FlatTop Optimizer
============================

Objektorientierte Umstrukturierung des ursprünglichen Skripts zur Optimierung
eines Multitone-AOD-Spot-Arrays hinsichtlich Uniformity (Gleichförmigkeit der
Intensität im Zentrum) und Crosstalk (Überlapp mit Nachbar-Sites).

Aufbau:
- Reine Mathematik-/Physik-Hilfsfunktionen bleiben als freistehende Funktionen
  (multitone_frequencies, angle_from_frequency, ... , create_neighbourhood).
- Der gesamte zustandsbehaftete Workflow (Parameter, Grid, Ergebnisse,
  Optimierungsstufen) steckt in der Klasse `MultitoneFlatTopOptimizer`.

WICHTIG - Berechnung und Plotten sind jetzt getrennt (2 Skripte):
Diese Datei enthält NUR Berechnung, kein matplotlib-Plotten des 2D-Scans mehr.
Die Plot-Funktionen für den 2D-Scan (scan_win_width_uniformity) leben in
`multitone_flattop_scan_plots.py` und lesen ihre Daten aus einem einfachen
dict/Pickle - dadurch muss der (potenziell langsame) Scan nicht neu berechnet
werden, nur weil sich am Plot-Styling etwas ändert:

    # Skript 1: Berechnung (einmal laufen lassen, ggf. langsam)
    opt = MultitoneFlatTopOptimizer(out_dir=".")
    opt.scan_win_width_uniformity(
        win_input_range=(0.5e-3, 3.5e-3), width_range=(0.15e6, 0.6e6),
        n_win_input=40, n_width=40,
    )
    opt.save_scan_results("scan_data.pkl")   # Rohdaten sichern

    # Skript 2: Plotten (beliebig oft, ohne Neuberechnung)
    from multitone_flattop_scan_plots import load_scan_results, ScanPlotter
    results = load_scan_results("scan_data.pkl")
    plotter = ScanPlotter(results, out_dir=".")
    plotter.plot_scan2d_combined(show=True, save=True)

Die übrigen Plot-Methoden (plot_initial, plot_win_optimized, ...) für die
Einzelauswertungen (compute_initial/optimize_win/width/combined) bleiben
unverändert direkt in dieser Klasse, da sie im Vergleich zum 2D-Scan schnell
sind und bislang nicht separat iteriert wurden.

Typische Verwendung:

    opt = MultitoneFlatTopOptimizer(out_dir=r"C:\\pfad\\zum\\output")
    opt.compute_initial()
    opt.plot_initial()

    opt.optimize_win()          # 1. Iteration: win optimieren
    opt.optimize_width()        # 2. Iteration: width optimieren (nutzt win_optimal)
    opt.optimize_combined()     # 3. Iteration: win + width + Amplituden gemeinsam

    opt.summary()

Jede Stufe kann auch unabhängig / mehrfach mit anderen Startwerten aufgerufen
werden, z.B.:

    opt.optimize_win(alpha=0.5, x0=2e-6)

Zusätzlich: 2D-Scan der Uniformity/Crosstalk über (Eingangswaist vor der 1.
Linse, width) - reine Berechnung, siehe oben für das Plotten:

    opt.scan_win_width_uniformity(
        win_input_range=(0.05e-6, 2.0e-6),
        width_range=(0.05e6, 0.8e6),
        amps=None,   # oder z.B. np.concatenate([amp_x, amp_y])
    )
"""

from datetime import date
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from matplotlib.path import Path
from pathlib import Path as FilePath
from scipy.special import j0, j1
from matplotlib.patches import Rectangle
from concurrent.futures import ProcessPoolExecutor, as_completed

import scan_checkpoint


# ======================================================================
# Default-Ordner für Zwischenergebnisse (gepickelte Scan-Rohdaten aus
# save_scan_results()/save_scan_amp_results()) bzw. für Bilder. Beide liegen
# direkt neben diesem Skript (also im Projektordner "Optimierung
# Niklas+Claude") und werden bei Bedarf automatisch angelegt - vorher zeigte
# DEFAULT_RESULTS_DIR fälschlich auf einen anderen Projektordner
# ("Strahlanalyse\AOD_Simulation\Results"), das ist seit 2026-08-18 auf
# expliziten Wunsch des Users korrigiert. Fällt auf einen Ordner relativ
# zum aktuellen Arbeitsverzeichnis zurück, falls der Pfad neben dem Skript
# aus irgendeinem Grund nicht beschreibbar ist (z.B. anderer Rechner).
# ======================================================================
def _default_dir(name):
    candidate = FilePath(__file__).resolve().parent.parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = FilePath(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DEFAULT_RESULTS_DIR = _default_dir("Results")
# Bilder werden tageweise abgelegt (Bilder/JJJJ-MM-TT). Der Dateiname
# traegt weiterhin alles, was die Datei identifiziert - der Ordner sagt
# nur, wann sie entstanden ist.
DEFAULT_IMAGES_DIR = _default_dir(f"Bilder/{date.today().isoformat()}")


def _resolve_pickle_path(out_dir, filename):
    """
    Wie multitone_flattop_scan_plots.resolve_save_path(), aber ohne
    matplotlib-Abhängigkeit (reine Dateisystem-Logik) - genutzt von
    save_scan_results(). Fragt bei existierender Datei nach Überschreiben
    (Konsole y/N) und wählt sonst einen freien Namen mit angehängtem
    Zähler (_2, _3, ...).

    WICHTIG: input() wird NUR aufgerufen, wenn tatsächlich eine
    interaktive Konsole angeschlossen ist (sys.stdin.isatty()). Sonst
    (z.B. bestimmte IDE-Ausführungsarten, Hintergrundprozesse) würde
    input() für immer auf eine Eingabe warten, die nie kommt, und das
    Skript scheinbar "hängen" lassen, ohne Fehlermeldung - in diesem Fall
    wird direkt und ohne Nachfrage der nächste freie Name gewählt.
    """
    path = FilePath(out_dir) / filename
    if not path.exists():
        return path

    if sys.stdin is not None and sys.stdin.isatty():
        answer = input(f"'{path.name}' existiert bereits. Überschreiben? [y/N]: ").strip().lower()
        if answer in ("y", "yes", "j", "ja"):
            return path
    else:
        print(f"'{path.name}' existiert bereits (keine interaktive Konsole - "
              f"überschreibe nicht automatisch, wähle stattdessen einen neuen Namen).")

    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            print(f"Bestehende Datei bleibt erhalten, speichere stattdessen als: {candidate.name}")
            return candidate
        counter += 1


# ======================================================================
# Freistehende Mathematik-/Physik-Hilfsfunktionen (zustandslos)
# ======================================================================

def multitone_frequencies(N, offset, width):
    """Erzeugt diskrete Frequenzen wie im AWG: width * n/(N-1) + offset"""
    if N <= 1:
        return np.array([offset], dtype=float)
    return width * np.arange(N) / (N - 1) + offset


def angle_from_frequency(f, offset, theta_max, f_band):
    """
    Winkel theta aus Frequenz f:
    theta = theta_max * (f - offset) / f_band
    """
    return theta_max * (f - offset) / f_band


def radius_from_angle(theta, f1, f2, fLO):
    """
    Räumlicher Abstand r aus Winkel theta:
    r = (f1 * fLO / f2) * tan(theta)
    """
    return (f1 * fLO / f2) * np.tan(theta)


def beam_radius_scaled(f1, f2, lambda_opt, fLO, win):
    """
    Strahlradius w im Ortsraum (in der Fokus-/Trap-Ebene), ausgehend vom
    Eingangswaist win VOR der ersten Linse (Teleskop f1 -> f2):

    w = (f1 / f2) * (lambda_opt * fLO) / (pi * win)
    """
    return (f1 / f2) * (lambda_opt * fLO) / (np.pi * win)


def gaussian_2d_distance_from_centers(X, Y, centers_x, centers_y, sigma):
    """2D Intensität als Summe von Gauß-Profilen mit gleicher Breite sigma."""
    I = np.zeros_like(X, dtype=float)
    for cx, cy in zip(centers_x, centers_y):
        I += np.exp(-2 * ((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2))
    return I


def numpy_gaussian_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, sigma, amps):
    """Wie gaussian_2d_distance_from_centers, aber mit individueller Amplitude pro Spot."""
    I = np.zeros_like(X, dtype=float)
    for cx, cy, a in zip(centers_x, centers_y, amps):
        I += a * np.exp(-2 * ((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2))
    return I


# ======================================================================
# Monkey-Patch-Indirektion für GPU-Beschleunigung (Übernahme aus Niklas'
# use_torch.py-Ansatz): dyn_gaussian_2d_weighted_distance_from_centers ist
# ein MODUL-LEVEL-Funktionszeiger, Default = die obige NumPy-Implementierung.
# gaussian_2d_weighted_distance_from_centers() (der Name, den PROFILE_FUNCS
# und create_neighbourhood() tatsächlich verwenden) leitet nur noch an
# diesen Zeiger weiter. Ein externes Modul (siehe use_torch.py) kann damit
# zur Laufzeit dyn_gaussian_2d_weighted_distance_from_centers auf eine
# GPU/PyTorch-Version umbiegen (import multitone_flattop_optimizer as mfo;
# mfo.dyn_gaussian_2d_weighted_distance_from_centers = cuda_version) - OHNE
# dass sich an dieser Datei sonst irgendetwas ändern muss: _evaluate(),
# create_neighbourhood(), scan_win_width_uniformity(),
# scan_win_width_amplitude_dependence() etc. rufen alle weiterhin nur
# gaussian_2d_weighted_distance_from_centers() auf und merken vom Patch
# nichts direkt.
# ======================================================================
dyn_gaussian_2d_weighted_distance_from_centers = numpy_gaussian_2d_weighted_distance_from_centers


def gaussian_2d_weighted_distance_from_centers(*args, **kwargs):
    return dyn_gaussian_2d_weighted_distance_from_centers(*args, **kwargs)


def bessel_2d_distance_from_centers(X, Y, centers_x, centers_y, width):
    """Summe von radialsymmetrischen Bessel-Profilen: I = sum_i J0(r_i / width)."""
    I = np.zeros_like(X, dtype=float)
    for cx, cy in zip(centers_x, centers_y):
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        I += j0(r / width)
    return I


def airy_2d_distance_from_centers(X, Y, centers_x, centers_y, first_zero_radius):
    """Summe von Airy-Profilen (z.B. für beugungsbegrenzte Spots)."""
    I = np.zeros_like(X, dtype=float)
    k = 3.83170597 / first_zero_radius
    for cx, cy in zip(centers_x, centers_y):
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        u = k * r
        airy = np.ones_like(u)
        mask = u > 1e-12
        airy[mask] = (2 * j1(u[mask]) / u[mask]) ** 2
        I += airy
    return I


def numpy_airy_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, first_zero_radius, amps):
    """Wie airy_2d_distance_from_centers, aber mit individueller Amplitude pro Spot."""
    I = np.zeros_like(X, dtype=float)
    k = 3.83170597 / first_zero_radius
    for cx, cy, a in zip(centers_x, centers_y, amps):
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        u = k * r
        airy = np.ones_like(u)
        mask = u > 1e-12
        airy[mask] = (2 * j1(u[mask]) / u[mask]) ** 2
        I += a * airy
    return I


# Monkey-Patch-Indirektion analog zu gaussian_2d_weighted_distance_from_centers
# oben: dyn_airy_2d_weighted_distance_from_centers ist der Funktionszeiger
# (Default = NumPy-Implementierung), airy_2d_weighted_distance_from_centers()
# leitet nur noch weiter. Siehe use_torch.py.
dyn_airy_2d_weighted_distance_from_centers = numpy_airy_2d_weighted_distance_from_centers


def airy_2d_weighted_distance_from_centers(*args, **kwargs):
    return dyn_airy_2d_weighted_distance_from_centers(*args, **kwargs)


def overlap_mask_circular_3sigma(X, Y, center_x, center_y, radius):
    R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    return R <= radius


def overlap_mask_pitch(X, Y, center_x, center_y, side_length):
    half_side = side_length / 2
    return (
        (np.abs(X - center_x) <= half_side) &
        (np.abs(Y - center_y) <= half_side)
    )


def create_neighbourhood(X, Y, pitch, centers_x, centers_y, w_in, amps=None, profile_func=None):
    """
    Summiert die (auf 1 normierten) Intensitätsbeiträge der 8 Nachbar-Sites.

    profile_func: gewichtete Profilfunktion mit Signatur
    f(X, Y, centers_x, centers_y, scale, amps). Default: Gauß-Profil.
    """
    if profile_func is None:
        profile_func = gaussian_2d_weighted_distance_from_centers
    if amps is None:
        amps = np.ones(len(centers_x))
    I_neighbor = np.zeros_like(X)
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            if ix == 0 and iy == 0:
                continue
            shifted_x = centers_x + ix * pitch
            shifted_y = centers_y + iy * pitch
            I_spot = profile_func(X, Y, shifted_x, shifted_y, w_in, amps)
            I_spot = I_spot / np.max(I_spot)
            I_neighbor += I_spot
    return I_neighbor


# ======================================================================
# Symmetrische Amplituden-Parametrisierung (outer/inner-Verhältnis)
# ======================================================================
def amps_from_ratio(r, N):
    """
    Baut einen Amplituden-Vektor der Länge N nach der symmetrischen
    "outer/inner"-Parametrisierung: die beiden äußersten Töne bekommen
    die Amplitude r, alle dazwischenliegenden (inneren) Töne bleiben
    fest bei 1. r ist also als Außen/Innen-Verhältnis definiert
    (innen = 1), wie besprochen.

    N=4 (gerade, kein mittlerer Ton):  [r, 1, 1, r]
    N=3 (ungerade, 1 mittlerer Ton):   [r, 1, r]
    N=2:                                [r, r]  (beide Töne sind "außen")
    N=1:                                [r]     (Entartungsfall)

    Für N_x=3 und N_y=4 (die hier fest verwendete Tonanzahl) entspricht
    das genau: amp_x = [r_x, 1, r_x], amp_y = [r_y, 1, 1, r_y].
    """
    amp = np.ones(N, dtype=float)
    if N >= 2:
        amp[0] = r
        amp[-1] = r
    elif N == 1:
        amp[0] = r
    return amp


def amps_from_ratios(r_x, r_y, N_x, N_y):
    """
    Kombiniert amps_from_ratio() für beide Achsen zu dem vollen
    Amplituden-Vektor [amp_x_0..amp_x_{N_x-1}, amp_y_0..amp_y_{N_y-1}],
    wie er von MultitoneFlatTopOptimizer._evaluate()/optimize_combined()
    erwartet wird (amps=amps_from_ratios(r_x, r_y, self.N_x, self.N_y)).
    """
    return np.concatenate([amps_from_ratio(r_x, N_x), amps_from_ratio(r_y, N_y)])


# Verfügbare Strahlprofile: Name -> gewichtete Profilfunktion f(X,Y,cx,cy,scale,amps)
PROFILE_FUNCS = {
    "gaussian": gaussian_2d_weighted_distance_from_centers,
    "airy": airy_2d_weighted_distance_from_centers,
}


# ======================================================================
# Klasse: Kapselt Parameter, Grid, Ergebnisse und Optimierungsstufen
# ======================================================================

class MultitoneFlatTopOptimizer:
    """
    Optimiert ein Multitone-AOD-Spot-Array hinsichtlich Uniformity und
    Crosstalk. Kapselt Eingangsparameter, Grid-Setup, Ausgangsberechnung
    sowie die einzelnen Optimierungsstufen als separat aufrufbare Methoden.
    """

    # ------------------------------------------------------------------
    # Feste Grenzwerte für die Optimierung (praxis-/hardwarebedingt)
    # ------------------------------------------------------------------
    MAX_TONES_PER_AXIS = 10          # nicht mehr als 10x10 Töne
    WIDTH_BOUNDS = (1e3, 0.8e6)      # width nur bis 0.8 MHz
    WIN_BOUNDS = (0, 2.5e-6)         # win (Waist in der Ebene) nur bis 2.5 µm
    WIN_BOUNDS_COMBINED = (0.1e-6, 2.5e-6)  # wie WIN_BOUNDS, aber L-BFGS-B mag echte untere Grenze

    DEFAULTS = dict(
        offset=100e6,
        width=0.35e6,
        N_x=3,
        N_y=4,
        f1=75e-3,
        f2=750e-3,
        fLO=52.88e-3,
        theta_max=43e-3,
        f_band=36e6,
        lambda_opt=795e-9,
        win=1.2e-6,
        pitch=5.288e-6,
        uniformity_side_length=None,  # None = automatisch "Ton-Quadrat" (siehe _build_masks)
        integration_radius=0.45e-6,
        threshindex=2,
        levels=(np.exp(-2), 0.5, 0.9),
        useconvexhull=False,
        use_Levels=False,
        n_grid=1000,
        profile="airy",           # "gaussian" oder "airy"
        airy_scale_factor=1.19,   # first_zero_radius = airy_scale_factor * win
    )

    def __init__(self, out_dir=None, **params):
        # Default (out_dir=None): DEFAULT_IMAGES_DIR, d.h. der "Bilder"-Ordner
        # neben diesem Skript - wird bei Bedarf automatisch angelegt. Vorher
        # war der Default "." (aktuelles Arbeitsverzeichnis), wodurch Bilder
        # je nach Startverzeichnis an unerwarteten Orten landen konnten.
        self.out_dir = FilePath(out_dir) if out_dir is not None else DEFAULT_IMAGES_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)

        merged = dict(self.DEFAULTS)
        merged.update(params)
        self.results = {}
        self._initial_guesses = {}  # profilabhängige Startwerte, siehe set_initial_guess()
        self.set_parameters(clear_results=False, **merged)

    # ------------------------------------------------------------------
    # Parameter setzen / Grid aufbauen
    # ------------------------------------------------------------------
    def set_parameters(self, clear_results=True, **kwargs):
        """
        Setzt/aktualisiert beliebige Eingangsparameter (z.B. offset, width,
        N_x, N_y, win, pitch, ...) und baut Frequenz-/Ortsraum-Geometrie
        sowie das Rechengitter neu auf.

        clear_results: Falls True (Default) werden zwischengespeicherte
        Optimierungsergebnisse verworfen, da sie sich auf die alte Geometrie
        bezogen haben.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.profile not in PROFILE_FUNCS:
            raise ValueError(f"Unbekanntes profile='{self.profile}'. Verfügbar: {list(PROFILE_FUNCS)}")

        if self.N_x > self.MAX_TONES_PER_AXIS or self.N_y > self.MAX_TONES_PER_AXIS:
            raise ValueError(
                f"N_x={self.N_x}, N_y={self.N_y}: maximal {self.MAX_TONES_PER_AXIS}x{self.MAX_TONES_PER_AXIS} "
                f"Töne sind vorgesehen."
            )

        self.levels = list(self.levels)
        self.N = self.N_x + self.N_y
        self.drawpoints = self.N < 16  # bei vielen Spots keine roten Punkte zeichnen

        self._setup_geometry()

        if clear_results:
            self.results = {}
            self._initial_guesses = {}

        return self

    def set_initial_guess(self, win=None, width=None, amp_x=None, amp_y=None, profile=None):
        """
        Hinterlegt eine eigene Startschätzung für optimize_combined() (win, width,
        Amplituden pro Achse). Wird automatisch verwendet, wenn optimize_combined()
        ohne explizites x0 aufgerufen wird.

        profile=None (Default) bezieht sich auf das aktuell aktive Profil
        (self.profile). So lassen sich für 'gaussian' und 'airy' unterschiedliche
        Vermutungen hinterlegen, z.B.:

            opt.set_profile('gaussian')
            opt.set_initial_guess(win=1.8e-6, width=2.9e5, amp_x=[..], amp_y=[..])

            opt.set_profile('airy')
            opt.set_initial_guess(win=2.4e-6, width=2.9e5, amp_x=[..], amp_y=[..])

        Einzelne Felder können weggelassen werden; für sie greift beim Optimieren
        weiterhin der bisherige Fallback (Ergebnis von optimize_win()/optimize_width()
        bzw. Gleichverteilung).
        """
        key = profile if profile is not None else self.profile
        if key not in PROFILE_FUNCS:
            raise ValueError(f"Unbekanntes profile='{key}'. Verfügbar: {list(PROFILE_FUNCS)}")

        current = self._initial_guesses.get(key, {})
        if win is not None:
            current["win"] = float(win)
        if width is not None:
            current["width"] = float(width)
        if amp_x is not None:
            amp_x = np.asarray(amp_x, dtype=float)
            if len(amp_x) != self.N_x:
                raise ValueError(f"amp_x muss Länge N_x={self.N_x} haben, hat aber {len(amp_x)}.")
            current["amp_x"] = amp_x
        if amp_y is not None:
            amp_y = np.asarray(amp_y, dtype=float)
            if len(amp_y) != self.N_y:
                raise ValueError(f"amp_y muss Länge N_y={self.N_y} haben, hat aber {len(amp_y)}.")
            current["amp_y"] = amp_y

        self._initial_guesses[key] = current
        return self

    def set_profile(self, profile, airy_scale_factor=None, clear_results=True):
        """
        Wechselt das Strahlprofil ('gaussian' oder 'airy').

        airy_scale_factor: optional neuer Umrechnungsfaktor first_zero_radius = factor * win
        (nur für profile='airy' relevant, Default bleibt sonst unverändert).
        """
        kwargs = dict(profile=profile)
        if airy_scale_factor is not None:
            kwargs["airy_scale_factor"] = airy_scale_factor
        return self.set_parameters(clear_results=clear_results, **kwargs)

    def _profile_func(self):
        """Gibt die aktuell gewählte gewichtete Profilfunktion zurück."""
        return PROFILE_FUNCS[self.profile]

    def _profile_scale(self, win_val):
        """
        Übersetzt den Optimierungsparameter 'win' in den für das jeweilige
        Profil benötigten Skalenparameter: sigma beim Gauß-Profil,
        first_zero_radius = airy_scale_factor * win beim Airy-Profil.
        """
        if self.profile == "airy":
            return self.airy_scale_factor * win_val
        return win_val

    def win_input_to_win(self, win_input):
        """
        Rechnet den Eingangswaist VOR der ersten Linse (win_input, in Metern)
        über eine 3-Linsen-Kaskade (f1, f2, fLO), jeweils im Abstand der
        Brennweiten (4f-Relay), in den in der Fokus-/Trap-Ebene wirksamen
        Waist um, der intern als self.win / sigma_spot für die
        Profilfunktionen verwendet wird:

        1. Teleskop f1 -> f2 (Abstand jeweils f1+f2): bildet win_input mit
           dem Faktor (f2/f1) auf einen Zwischenwaist ab.
        2. Fokussierlinse fLO (im Abstand fLO davor/danach): fokussiert den
           Zwischenwaist beugungsbegrenzt gemäß w' = lambda_opt * fLO / (pi * w).

        Zusammen ergibt das:

            win_eff = (f1/f2) * (lambda_opt * fLO) / (pi * win_input)

        Nutzt dafür die freistehende Funktion beam_radius_scaled() mit den
        aktuell gesetzten Parametern self.f1, self.f2, self.lambda_opt, self.fLO.

        Typische Größenordnungen: win_input im mm-Bereich (Kollimationswaist
        vor dem AOD/Teleskop), win_eff im µm-Bereich (Fokus-/Trap-Waist).
        """
        if win_input <= 0:
            raise ValueError("win_input muss > 0 sein.")
        return beam_radius_scaled(self.f1, self.f2, self.lambda_opt, self.fLO, win_input)

    def width_to_um(self, width_hz):
        """
        Rechnet einen Frequenzabstand `width` (Hz) über dieselbe
        Winkel/Radius-Beziehung wie der Rest der Optik (self.theta_max,
        self.f_band, self.f1, self.f2, self.fLO) in die zugehörige
        räumliche Ausdehnung (µm) an der Trap-Ebene um - d.h. welche
        physikalische Länge ein gegebener width-Wert entspricht.
        """
        theta_width = self.theta_max * width_hz / self.f_band
        return radius_from_angle(theta_width, self.f1, self.f2, self.fLO) * 1e6

    def _setup_geometry(self):
        """Berechnet Ausgangs-Spotpositionen (aus offset/width) und das feste Rechengitter X, Y."""
        self.f_center = self.offset + self.width / 2

        self.centers_r_x, self.centers_r_y, self.r_center = self._compute_centers_for_width(self.width)

        self.sigma_spot = self.win
        self.first_zero_radius = self._profile_scale(self.win)

        margin = 10 * self.sigma_spot
        self.margin = margin

        # Das Rechengitter wird einmal aus den Ausgangsparametern aufgebaut und
        # bleibt für alle nachfolgenden Optimierungsstufen (win/width/combined)
        # fix - genau wie im Ausgangsskript.
        self.x = np.linspace(-margin / 2, np.max(np.abs(self.centers_r_x)) + margin / 2, self.n_grid)
        self.y = np.linspace(-margin / 2, np.max(np.abs(self.centers_r_y)) + margin / 2, self.n_grid)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.mid_x_idx = len(self.x) // 2
        self.mid_y_idx = len(self.y) // 2

    def _compute_centers_for_width(self, width_val):
        """Berechnet Spot-Zentren im Ortsraum + Zentrumsposition für einen gegebenen width-Wert."""
        fx_freq = multitone_frequencies(self.N_x, self.offset, width_val)
        fy_freq = multitone_frequencies(self.N_y, self.offset, width_val)

        f_center = self.offset + width_val / 2
        theta_center = angle_from_frequency(f_center, self.offset, self.theta_max, self.f_band)
        r_center = radius_from_angle(theta_center, self.f1, self.f2, self.fLO)

        centers_x, centers_y = [], []
        for fx in fx_freq:
            theta_x = angle_from_frequency(fx, self.offset, self.theta_max, self.f_band)
            r_x = radius_from_angle(theta_x, self.f1, self.f2, self.fLO)
            for fy in fy_freq:
                theta_y = angle_from_frequency(fy, self.offset, self.theta_max, self.f_band)
                r_y = radius_from_angle(theta_y, self.f1, self.f2, self.fLO)
                centers_x.append(r_x)
                centers_y.append(r_y)

        return np.array(centers_x), np.array(centers_y), r_center

    # ------------------------------------------------------------------
    # Masken (Uniformity-Region / Crosstalk-Region)
    # ------------------------------------------------------------------
    def _resolve_uniformity_side(self, centers_x, centers_y, r_center):
        """
        Liefert die tatsächlich verwendete Uniformity-Seitenlänge:
        self.uniformity_side_length, falls explizit gesetzt, sonst automatisch
        das "Ton-Quadrat" (Ausdehnung der übergebenen Spot-Zentren).
        """
        side = self.uniformity_side_length
        if side is None:
            half_extent = max(
                np.max(np.abs(centers_x - r_center)),
                np.max(np.abs(centers_y - r_center)),
            )
            side = 2 * half_extent
        return side

    def _build_masks(self, X, Y, r_center, I, centers_x, centers_y):
        """
        Erstellt Uniformity- und Crosstalk-Maske je nach gewähltem Modus.

        Uniformity-Region: Falls useconvexhull/use_Levels nicht aktiv sind und
        self.uniformity_side_length nicht explizit gesetzt ist (Default: None),
        wird automatisch das "Ton-Quadrat" verwendet - ein Quadrat, das exakt
        die Ausdehnung der aktuell übergebenen Spot-Zentren (centers_x, centers_y)
        umfasst. Das passt sich bei jeder Optimierungsstufe (win/width/combined)
        automatisch an die jeweils aktuellen Spot-Positionen an.
        """
        hull_vertices = None
        if self.useconvexhull:
            points = np.column_stack([centers_x, centers_y])
            hull = ConvexHull(points)
            hull_vertices = hull.points[hull.vertices]
            polygon_path = Path(hull_vertices)
            grid_points = np.column_stack([X.ravel(), Y.ravel()])
            mask_uniformity = polygon_path.contains_points(grid_points).reshape(X.shape)
        elif self.use_Levels:
            mask_uniformity = I >= self.levels[self.threshindex]
        else:
            side = self._resolve_uniformity_side(centers_x, centers_y, r_center)
            mask_uniformity = overlap_mask_pitch(X, Y, r_center, r_center, side)

        mask_crosstalk = overlap_mask_pitch(X, Y, r_center, r_center, self.pitch)
        return mask_uniformity, mask_crosstalk, hull_vertices

    # ------------------------------------------------------------------
    # Kernfunktion: Uniformity + Crosstalk für gegebene (win, width, amps)
    # ------------------------------------------------------------------
    def _build_dynamic_grid(self, win_val, width_val, centers_x=None, centers_y=None):
        """
        Builds a grid sized specifically for the given win_val/width_val -
        mirroring the GUI's compute_grid(), which recomputes the margin
        from the CURRENT win/width on every call. Used by
        scan_win_width_uniformity() so grid accuracy doesn't silently
        degrade as win_val strays from the win the optimizer happened to
        be constructed with (the class's own self.X/self.Y grid is fixed
        at construction time and does NOT adapt to win_val/width_val
        passed into _evaluate() later - fine for optimize_win/width/combined,
        which refine near a similar starting point, but wrong for a 2D
        scan where win legitimately varies over a wide range).
        """
        if centers_x is None or centers_y is None:
            centers_x, centers_y, _ = self._compute_centers_for_width(width_val)
        scale = self._profile_scale(win_val)
        side_u = self.uniformity_side_length
        if side_u is None:
            r_center = self._compute_centers_for_width(width_val)[2]
            half_extent = max(np.max(np.abs(centers_x - r_center)), np.max(np.abs(centers_y - r_center)))
            side_u = 2 * half_extent
        margin = max(10 * scale, 1.3 * side_u, 1.3 * self.pitch, 3.0 * self.pitch)
        x = np.linspace(-margin / 2, np.max(np.abs(centers_x)) + margin / 2, self.n_grid)
        y = np.linspace(-margin / 2, np.max(np.abs(centers_y)) + margin / 2, self.n_grid)
        X, Y = np.meshgrid(x, y)
        return X, Y

    def _evaluate(self, win_val, width_val, amps=None, grid=None):
        """
        Berechnet Intensitätsverteilung, Nachbar-Intensität, Uniformity und
        Crosstalk für gegebenes win, width und optionale Amplituden pro
        Achse (amps = [amp_x_0..amp_x_{N_x-1}, amp_y_0..amp_y_{N_y-1}]).

        Ersetzt die vier fast identischen calculate_uniformity(_width)/
        calculate_overlapp(_width)-Funktionen des Ausgangsskripts durch
        eine einzige Kernberechnung.

        grid: optionales (X, Y)-Tupel. Ohne Angabe wird das feste,
        beim Konstruieren aufgebaute self.X/self.Y verwendet (Standard-
        verhalten für optimize_win/width/combined, die win/width nahe am
        Startwert verfeinern). scan_win_width_uniformity() übergibt
        stattdessen ein pro Punkt über _build_dynamic_grid() neu passend
        aufgebautes Gitter, da win dort über einen weiten Bereich variiert
        und ein fixes Gitter sonst je nach Abstand zum Konstruktions-win
        systematisch falsche (zu eng geschnittene) Ergebnisse liefert.

        Gibt None zurück, falls die Parameter zu einem ungültigen Zustand
        führen (z.B. leere Maske) - das ist der Penalty-Fall für die
        Optimierung.
        """
        if win_val <= 0:
            return None

        centers_x, centers_y, r_center = self._compute_centers_for_width(width_val)
        profile_func = self._profile_func()
        scale = self._profile_scale(win_val)  # sigma (Gauß) bzw. first_zero_radius (Airy)

        X, Y = grid if grid is not None else (self.X, self.Y)

        # Amplituden pro Achse -> pro Spot; ohne Angabe wird gleichmäßig (1) gewichtet.
        if amps is not None:
            amps = np.asarray(amps)
            amp_x = amps[:self.N_x]
            amp_y = amps[self.N_x:self.N_x + self.N_y]
            amp_spots = np.repeat(amp_x, self.N_y) * np.tile(amp_y, self.N_x)
        else:
            amp_spots = np.ones(len(centers_x))

        I_new = profile_func(X, Y, centers_x, centers_y, scale, amp_spots)

        if np.max(I_new) == 0:
            return None
        I_new = I_new / np.max(I_new)

        try:
            I_neighbor = create_neighbourhood(X, Y, self.pitch, centers_x, centers_y,
                                               w_in=scale, amps=amp_spots, profile_func=profile_func)
            mask_uniformity, mask_crosstalk, hull_vertices = self._build_masks(
                X, Y, r_center, I_new, centers_x, centers_y
            )
        except Exception:
            return None

        I_inside_uni = I_new[mask_uniformity]
        I_inside_cross = I_new[mask_crosstalk]
        I_neighbor_inside = I_neighbor[mask_crosstalk]

        if (len(I_inside_uni) == 0 or np.mean(I_inside_uni) == 0
                or len(I_inside_cross) == 0 or np.sum(I_inside_cross) == 0):
            return None

        uniformity = np.std(I_inside_uni) / np.mean(I_inside_uni)
        eta = np.sum(I_neighbor_inside) / np.sum(I_inside_cross)

        return dict(
            uniformity=uniformity, eta=eta,
            I=I_new, I_neighbor=I_neighbor,
            centers_x=centers_x, centers_y=centers_y, r_center=r_center,
            mask_uniformity=mask_uniformity, mask_crosstalk=mask_crosstalk,
            hull_vertices=hull_vertices,
            win=win_val, width=width_val, amps=amps,
        )

    # ------------------------------------------------------------------
    # Ausgangszustand: Berechnung + Plot
    # ------------------------------------------------------------------
    def compute_initial(self, verbose=True):
        """Berechnet Intensitätsverteilung, Uniformity und Crosstalk für die aktuellen Eingangsparameter."""
        details = self._evaluate(self.win, self.width)
        if details is None:
            raise RuntimeError("Ungültige Ausgangsparameter: Maske/Intensität leer.")

        if verbose:
            print(f"Mittlere Intensität in der Region: {np.mean(details['I'][details['mask_uniformity']]):.4f}")
            print(f"Standardabweichung: {np.std(details['I'][details['mask_uniformity']]):.4f}")
            print(f"Uniformity (σ/μ): {details['uniformity']:.4f} ({details['uniformity']*100:.2f}%)")
            print(f"Überlapp der Nachbarn in die Zentralstruktur: {details['eta']*100:.3f}%")

        self.results['initial'] = details
        return details

    def plot_initial(self, show=True, save=True):
        """Plottet den Ausgangszustand (Intensität + Schnitte, sowie Crosstalk-Nachbarn)."""
        if 'initial' not in self.results:
            self.compute_initial()
        res = self.results['initial']
        tag = self._filetag()

        self._plot_intensity(
            res, title=f"Original: win = {self.win:.3e} m, Uniformity = {res['uniformity']*100:.2f}%",
            filename=f"FlatMultiTone_Original_{tag}.png", show=show, save=save,
        )
        self._plot_neighbor(
            res, title=f"Neighbors (Original): Crosstalk = {res['eta']*100:.3f}%",
            filename=f"FlatMultiTone_NeighborOriginal_{tag}.png", show=show, save=save,
        )

    # ------------------------------------------------------------------
    # 1. Iteration: win optimieren
    # ------------------------------------------------------------------
    def optimize_win(self, alpha=0.7, x0=None, bounds=None, show=True, save=True, verbose=True):
        """
        Optimiert win (Strahlradius/Waist in der Ebene) bei fixem width zur
        Minimierung von Uniformity + Crosstalk.

        bounds: Default self.WIN_BOUNDS (0, 2.5 µm) - Waist in der Ebene nur bis 2.5 µm.
        """
        bounds = bounds if bounds is not None else self.WIN_BOUNDS
        x0_val = self.win if x0 is None else x0

        def objective(p):
            val = self._evaluate(p[0], self.width)
            if val is None:
                return 1e10
            return alpha * val['uniformity'] + (1 - alpha) * val['eta']

        if verbose:
            print("\n" + "=" * 60)
            print("Starte Optimierung von win zur Minimierung von Uniformity + Crosstalk...")
            print("=" * 60)

        result = minimize(
            objective, x0=[x0_val], method='Nelder-Mead',
            options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 1000},
            bounds=[bounds],
        )
        win_optimal = result.x[0]
        details = self._evaluate(win_optimal, self.width)

        if verbose:
            print(f"\nOptimales win: {win_optimal:.6e} m")
            print(f"Optimale Uniformity: {details['uniformity']:.4f} ({details['uniformity']*100:.2f}%)")
            print(f"Optimaler Overlapp: {details['eta']:.4f} ({details['eta']*100:.2f}%)")
            if 'initial' in self.results:
                u0 = self.results['initial']['uniformity']
                print(f"Verbesserung ggü. Original: {((u0 - details['uniformity']) / u0 * 100):.2f}%")
            print("=" * 60)

        self.results['win_opt'] = dict(win_optimal=win_optimal, nfev=result.nfev, nit=result.nit, **details)

        if show or save:
            self.plot_win_optimized(show=show, save=save)

        return win_optimal

    def plot_win_optimized(self, show=True, save=True):
        if 'win_opt' not in self.results:
            raise RuntimeError("optimize_win() muss zuerst aufgerufen werden.")
        res = self.results['win_opt']
        tag = self._filetag()

        self._plot_intensity(
            res, title=f"Optimized: win = {res['win_optimal']:.3e} m, Uniformity = {res['uniformity']*100:.2f}%",
            filename=f"FlatMultiTone_Optimized_{tag}.png", show=show, save=save,
            xlim=(-5, 10), ylim=(-5, 10),
        )
        self._plot_neighbor(
            res, title=f"Neighbors (win-optimized): Crosstalk = {res['eta']*100:.3f}%",
            filename=f"FlatMultiTone_NeighborWinOptimized_{tag}.png", show=show, save=save,
        )

    # ------------------------------------------------------------------
    # 2. Iteration: width (Frequenzabstände) optimieren
    # ------------------------------------------------------------------
    def optimize_width(self, alpha=0.9, x0=None, win_val=None, bounds=None, show=True, save=True, verbose=True):
        """
        Optimiert width (Frequenzabstände der Spots) bei fixem win zur
        Minimierung von Uniformity + Crosstalk.

        win_val: welcher win-Wert dabei benutzt wird. Default: das Ergebnis
        von optimize_win(), falls vorhanden, sonst der aktuelle self.win.

        bounds: Default self.WIDTH_BOUNDS (1 kHz, 0.8 MHz) - width nur bis 0.8 MHz.
        """
        bounds = bounds if bounds is not None else self.WIDTH_BOUNDS
        sigma = win_val if win_val is not None else self.results.get('win_opt', {}).get('win_optimal', self.win)
        x0_val = self.width if x0 is None else x0

        def objective(p):
            val = self._evaluate(sigma, p[0])
            if val is None:
                return 1e10
            return alpha * val['uniformity'] + (1 - alpha) * val['eta']

        if verbose:
            print("\n" + "=" * 60)
            print("ZWEITE ITERATION: Optimierung der Frequenzabstände (width)")
            print(f"win bleibt auf: {sigma:.3e}")
            print("=" * 60)

        result = minimize(
            objective, x0=[x0_val], method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-7, 'maxiter': 1000},
            bounds=[bounds],
        )
        width_optimal = result.x[0]
        details = self._evaluate(sigma, width_optimal)

        if verbose:
            print(f"\nOptimale width: {width_optimal:.6e} Hz")
            print(f"Uniformity (width-Opt.): {details['uniformity']:.4f} ({details['uniformity']*100:.2f}%)")
            print(f"Crosstalk (width-Opt.): {details['eta']:.4f} ({details['eta']*100:.3f}%)")

        self.results['width_opt'] = dict(width_optimal=width_optimal, win_used=sigma,
                                          nfev=result.nfev, nit=result.nit, **details)

        if show or save:
            self.plot_width_optimized(show=show, save=save)

        return width_optimal

    def plot_width_optimized(self, show=True, save=True):
        if 'width_opt' not in self.results:
            raise RuntimeError("optimize_width() muss zuerst aufgerufen werden.")
        res = self.results['width_opt']
        tag = self._filetag()

        self._plot_intensity(
            res, title=f"Frequency optimized: width = {res['width_optimal']:.3e} Hz, "
                       f"Uniformity = {res['uniformity']*100:.2f}%",
            filename=f"FlatMultiTone_FrequencyOptimized_{tag}.png", show=show, save=save,
        )
        self._plot_neighbor(
            res, title=f"Neighbors (width-optimized): Crosstalk = {res['eta']*100:.3f}%",
            filename=f"FlatMultiTone_NeighborWidthOptimized_{tag}.png", show=show, save=save,
        )

    # ------------------------------------------------------------------
    # 3. Iteration: win + width + Amplituden gemeinsam optimieren
    # ------------------------------------------------------------------
    def optimize_combined(self, alpha=0.9, x0=None, bounds=None, show=True, save=True, verbose=True):
        """
        Optimiert win, width sowie individuelle Amplituden pro Achse
        (amp_x_0..amp_x_{N_x-1}, amp_y_0..amp_y_{N_y-1}) gemeinsam.

        Startwerte (Priorität, falls x0 nicht explizit übergeben wird):
        1. per set_initial_guess() hinterlegte, profilabhängige Vermutung
        2. Ergebnisse von optimize_win()/optimize_width(), falls vorhanden
        3. aktuelles self.win/self.width, Amplituden = 1

        Ein guter Startwert (z.B. aus set_initial_guess()) reduziert bei
        L-BFGS-B typischerweise deutlich die Anzahl nötiger Funktions-
        auswertungen bis zur Konvergenz.

        bounds: Default self.WIN_BOUNDS_COMBINED (0.1 - 2.5 µm) für win,
        self.WIDTH_BOUNDS (1 kHz - 0.8 MHz) für width, (0, 2) für jede Amplitude.
        """
        win0 = self.results.get('win_opt', {}).get('win_optimal', self.win)
        width0 = self.results.get('width_opt', {}).get('width_optimal', self.width)
        amp_x0 = np.ones(self.N_x)
        amp_y0 = np.ones(self.N_y)

        # Eigene, per set_initial_guess() hinterlegte Vermutung (profilabhängig)
        # überschreibt die obigen Fallbacks, sofern gesetzt.
        guess = self._initial_guesses.get(self.profile, {})
        win0 = guess.get('win', win0)
        width0 = guess.get('width', width0)
        amp_x0 = guess.get('amp_x', amp_x0)
        amp_y0 = guess.get('amp_y', amp_y0)

        if x0 is None:
            x0 = np.concatenate(([win0, width0], amp_x0, amp_y0))
        if bounds is None:
            bounds = [self.WIN_BOUNDS_COMBINED, self.WIDTH_BOUNDS] + [(0.0, 2.0)] * (self.N_x + self.N_y)

        def objective(p):
            val = self._evaluate(p[0], p[1], amps=p[2:])
            if val is None:
                return 1e10
            return alpha * val['uniformity'] + (1 - alpha) * val['eta']

        if verbose:
            print("\n" + "=" * 60)
            print("DRITTE ITERATION: Optimierung von win, width und Amplituden pro Achse")
            print("=" * 60)
            print("\nStarte kombinierte Optimierung (win + width + Amplituden)...")

        result = minimize(
            objective, x0=x0, method='L-BFGS-B', bounds=bounds,
            options={'ftol': 1e-9, 'maxiter': 2000},
        )

        win_opt, width_opt = result.x[0], result.x[1]
        amps_opt = result.x[2:]
        details = self._evaluate(win_opt, width_opt, amps=amps_opt)

        if verbose:
            print(f"\nOptimales win: {win_opt:.6e} m")
            print(f"Optimale width: {width_opt:.6e} Hz")
            print(f"Optimierte amp_x: {amps_opt[:self.N_x]}")
            print(f"Optimierte amp_y: {amps_opt[self.N_x:]}")
            print(f"Uniformity (win+width+amps): {details['uniformity']:.4f} ({details['uniformity']*100:.2f}%)")
            print(f"Crosstalk (win+width+amps): {details['eta']:.4f} ({details['eta']*100:.3f}%)")
            print(f"Funktionsauswertungen (nfev): {result.nfev}, Iterationen (nit): {result.nit}")

        # 'details' enthält bereits 'amps' (aus _evaluate); win_optimal/width_optimal ergänzen
        self.results['combined_opt'] = dict(win_optimal=win_opt, width_optimal=width_opt,
                                             nfev=result.nfev, nit=result.nit, **details)

        if show or save:
            self.plot_combined_optimized(show=show, save=save)

        return result.x

    def plot_combined_optimized(self, show=True, save=True):
        if 'combined_opt' not in self.results:
            raise RuntimeError("optimize_combined() muss zuerst aufgerufen werden.")
        res = self.results['combined_opt']
        tag = self._filetag()

        self._plot_intensity(
            res,
            title=(f"win+width+Amp optimized: win = {res['win_optimal']:.3e} m, "
                   f"width = {res['width_optimal']:.3e} Hz, "
                   f"Uniformity = {res['uniformity']*100:.2f}%, Crosstalk = {res['eta']*100:.2f}%"),
            filename=f"FlatMultiTone_CombinedOptimized_{tag}.png", show=show, save=save,
        )
        self._plot_neighbor(
            res, title=f"Neighbors (win+width+Amp-optimiert): Crosstalk = {res['eta']*100:.3f}%",
            filename=f"FlatMultiTone_NeighborCombinedOptimized_{tag}.png", show=show, save=save,
        )

    # ------------------------------------------------------------------
    # 2D-Scan: Uniformity über (Eingangswaist vor 1. Linse) x width
    # ------------------------------------------------------------------
    def scan_win_width_uniformity(self, win_input_range, width_range,
                                   n_win_input=40, n_width=40,
                                   amps=None, alpha=0.9, verbose=True,
                                   progress_callback=None,
                                   checkpoint_path=None,
                                   checkpoint_interval_s=scan_checkpoint.CHECKPOINT_INTERVAL_S):
        """
        Scans BOTH the uniformity and the crosstalk in the spot square over
        two axes at fixed tone count (N_x, N_y):

          - x-axis: input waist before the first lens, win_input (meters)
          - y-axis: width, i.e. the frequency spacing of the tones (Hz)

        For each grid point, win_input is converted via win_input_to_win()
        (telescope imaging f1 -> f2, plus focusing lens fLO) into the waist
        used internally at the focal/trap plane. Uniformity and crosstalk
        come from the same _evaluate() call per point, so there is no extra
        computational cost in tracking both.

        Each point uses its OWN correctly-sized grid (via
        _build_dynamic_grid(), matching the GUI's per-call compute_grid()),
        not the fixed self.X/self.Y grid built at construction time - since
        win_eff varies over a wide range across the scan, a fixed grid
        sized for the constructor's win would silently clip/misjudge the
        intensity profile for any win_eff far from that value.

        This method does PURE COMPUTATION - no plotting. To plot the
        result, either:
          - call get_scan_results() / save_scan_results(filepath) and hand
            that off to multitone_flattop_scan_plots.ScanPlotter, so
            re-styling the plots never requires re-running this (usually
            much more expensive) scan, or
          - for quick one-off use, import ScanPlotter directly and pass
            get_scan_results() straight in (still no need to touch this
            module again for plot-only changes).

        win_input_range / width_range: (min, max) tuples in SI units
        (meters and Hz respectively).
        amps: optional fixed amplitudes [amp_x_0..amp_x_{N_x-1},
              amp_y_0..amp_y_{N_y-1}]. If omitted, all amplitudes are set
              to 1 (see the selection dialog in the GUI to set them
              beforehand).
        alpha: weight for the combined score alpha*uniformity + (1-alpha)*eta
               used to pick the single "best" point marked with a cross on
               both plots - the same weighting convention used by
               optimize_win()/optimize_width()/optimize_combined() to find
               good starting parameters (default 0.9, same as those methods).

        progress_callback: optional Callable(done, total) -> bool|None,
        called after every computed grid point (done = number of points
        computed so far, total = n_win_input * n_width). This can drive,
        e.g., a QProgressDialog/QProgressBar or a tqdm bar. If the callback
        explicitly returns False, the scan is cooperatively cancelled
        (e.g. on "Cancel" in a dialog) - the grid computed so far (with NaN
        for the remaining points) is still returned.

        checkpoint_path / checkpoint_interval_s: falls checkpoint_path
        gesetzt ist, wird alle checkpoint_interval_s Sekunden (Default 1h,
        siehe scan_checkpoint.py) der bisherige Fortschritt unter genau
        diesem Pfad gespeichert - dieselbe Art dict, die auch
        get_scan_results() liefert (mit NaN fuer noch nicht berechnete
        Punkte). Existiert unter checkpoint_path bereits eine zu diesem
        Scan (identisches Gitter/N_x/N_y/amps/alpha) passende (Teil-)Datei,
        wird sie beim Start automatisch geladen und NUR die darin noch
        fehlenden Punkte werden neu berechnet - so laesst sich ein
        abgebrochener/abgestuerzter Scan an der Stelle fortsetzen, an der
        er stehen geblieben ist, statt komplett neu zu beginnen.

        Stores the result in self.results['scan2d'] and returns
        (win_input_vals, width_vals, uniformity_grid, crosstalk_grid). NaN
        entries in the grids mean invalid/non-evaluable or not-yet-computed
        (cancelled) parameter combinations.
        """
        win_input_vals = np.linspace(win_input_range[0], win_input_range[1], n_win_input)
        width_vals = np.linspace(width_range[0], width_range[1], n_width)

        resumed = scan_checkpoint.load_resumable(
            checkpoint_path, win_input_range, width_range, n_win_input, n_width,
            self.N_x, self.N_y, extra_match=dict(amps=amps, alpha=alpha),
            airy_scale_factor=self.airy_scale_factor,
            optics_match=dict(n_grid=self.n_grid), verbose=verbose,
        )
        if resumed is not None:
            uniformity_grid = np.asarray(resumed['uniformity_grid'], dtype=float).copy()
            crosstalk_grid = np.asarray(resumed['crosstalk_grid'], dtype=float).copy()
            n_done_before = scan_checkpoint.count_done(uniformity_grid)
            if verbose:
                print(f"[Checkpoint] Setze Scan fort: {n_done_before}/{uniformity_grid.size} "
                      f"Punkte bereits vorhanden ({checkpoint_path}).")
        else:
            uniformity_grid = np.full((n_width, n_win_input), np.nan)
            crosstalk_grid = np.full((n_width, n_win_input), np.nan)
        ckpt = scan_checkpoint.CheckpointWriter(checkpoint_path, checkpoint_interval_s, verbose=verbose)

        total = n_width * n_win_input
        done = 0
        cancelled = False

        def _current_results():
            return dict(
                win_input_vals=win_input_vals, width_vals=width_vals,
                uniformity_grid=uniformity_grid, crosstalk_grid=crosstalk_grid,
                amps=amps, alpha=alpha,
                best=scan_checkpoint.best_point(uniformity_grid, crosstalk_grid, win_input_vals,
                                                 width_vals, alpha, "uniformity", "crosstalk"),
                N_x=self.N_x, N_y=self.N_y, f1=self.f1, f2=self.f2, fLO=self.fLO,
                lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
                profile=self.profile,
                airy_scale_factor=self.airy_scale_factor,
                # n_grid/weighted_n_grid (und der Atom-Offset) bestimmen, WIE fein
                # ausgewertet wurde. Ein fortgesetzter Scan mit anderer Aufloesung
                # haette sonst zwei verschiedene Aufloesungen in einem Datensatz.
                n_grid=self.n_grid,
            )

        if verbose:
            print("\n" + "=" * 60)
            print(f"2D scan: uniformity & crosstalk over win_input ({n_win_input} points) x "
                  f"width ({n_width} points), N_x={self.N_x}, N_y={self.N_y}")
            print("=" * 60)

        for i, width_val in enumerate(width_vals):
            if cancelled:
                break
            for j, win_input_val in enumerate(win_input_vals):
                if np.isfinite(uniformity_grid[i, j]):
                    pass  # bereits aus Checkpoint vorhanden - nicht neu berechnen
                else:
                    try:
                        win_eff = self.win_input_to_win(win_input_val)
                    except ValueError:
                        win_eff = None
                    if win_eff is not None:
                        point_grid = self._build_dynamic_grid(win_eff, width_val)
                        details = self._evaluate(win_eff, width_val, amps=amps, grid=point_grid)
                        if details is not None:
                            uniformity_grid[i, j] = details['uniformity']
                            crosstalk_grid[i, j] = details['eta']

                done += 1
                if progress_callback is not None:
                    if progress_callback(done, total) is False:
                        cancelled = True
                        break
                ckpt.maybe_save(_current_results, done=done, total=total)
            if verbose and n_width >= 10 and (i % max(1, n_width // 10) == 0):
                print(f"  ... width row {i + 1}/{n_width}")

        if ckpt.active:
            ckpt.maybe_save(_current_results, done=done, total=total, force=True)

        combined_grid = alpha * uniformity_grid + (1 - alpha) * crosstalk_grid

        best = dict(win_input=None, width=None, uniformity=None, crosstalk=None, combined=None)
        if np.any(np.isfinite(combined_grid)):
            idx_min = np.unravel_index(np.nanargmin(combined_grid), combined_grid.shape)
            best.update(
                win_input=win_input_vals[idx_min[1]],
                width=width_vals[idx_min[0]],
                uniformity=uniformity_grid[idx_min],
                crosstalk=crosstalk_grid[idx_min],
                combined=combined_grid[idx_min],
            )

        self.results['scan2d'] = dict(
            win_input_vals=win_input_vals, width_vals=width_vals,
            uniformity_grid=uniformity_grid, crosstalk_grid=crosstalk_grid,
            amps=amps, alpha=alpha, best=best,
        )

        if verbose:
            valid = np.isfinite(uniformity_grid)
            status = "cancelled" if cancelled else "completed"
            print(f"Scan {status} ({np.sum(valid)}/{uniformity_grid.size} valid points, "
                  f"{done}/{total} computed).")
            if best["win_input"] is not None:
                print(f"Combined-optimal point (alpha={alpha}): win_input={best['win_input']*1e3:.4f} mm, "
                      f"width={best['width']*1e-6:.3f} MHz  ->  "
                      f"Uniformity={best['uniformity']*100:.2f}%, Crosstalk={best['crosstalk']*100:.3f}%")
            print("=" * 60)

        return win_input_vals, width_vals, uniformity_grid, crosstalk_grid

    def get_scan_results(self):
        """
        Returns a self-contained, plain dict with everything
        multitone_flattop_scan_plots.ScanPlotter needs to render the 2D
        scan plots - WITHOUT re-running scan_win_width_uniformity(). Pass
        this (or a pickled/reloaded copy from save_scan_results()) straight
        into ScanPlotter(...).
        """
        if 'scan2d' not in self.results:
            raise RuntimeError("scan_win_width_uniformity() must be called first.")
        res = dict(self.results['scan2d'])  # shallow copy, don't mutate self.results
        res.update(
            N_x=self.N_x, N_y=self.N_y,
            f1=self.f1, f2=self.f2, fLO=self.fLO,
            lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
            profile=self.profile,
            airy_scale_factor=self.airy_scale_factor,
            # n_grid/weighted_n_grid (und der Atom-Offset) bestimmen, WIE fein
            # ausgewertet wurde. Ein fortgesetzter Scan mit anderer Aufloesung
            # haette sonst zwei verschiedene Aufloesungen in einem Datensatz.
            n_grid=self.n_grid,
        )
        return res

    def save_scan_results(self, filepath=None):
        """
        Pickles get_scan_results() to `filepath`, so the 2D-scan plots can
        be (re-)generated later - e.g. from a separate plotting script or
        session - without re-running the (usually much more expensive)
        scan. Load it back with multitone_flattop_scan_plots.load_scan_results().

        filepath: optional. If omitted, saves into DEFAULT_RESULTS_DIR (dem
        "Results"-Ordner neben diesem Skript, wird bei Bedarf automatisch
        angelegt) unter einem automatisch generierten Namen, der Tonanzahl,
        Scan-Auflösung UND Strahlprofil kodiert, z.B.
        "scan_data_N3x4_40x40pts_Airy.pkl". If that file already exists, asks
        whether to overwrite (like the image-saving code) and otherwise
        picks a free name with an incrementing suffix (_2, _3, ...) so a
        previous result is never silently lost.
        """
        if filepath is None:
            res = self.results.get('scan2d', {})
            n_win = len(res.get('win_input_vals', []))
            n_width = len(res.get('width_vals', []))
            profile_tag = "Airy" if self.profile == "airy" else "Gauss" if self.profile == "gaussian" else self.profile
            filename = f"scan_data_N{self.N_x}x{self.N_y}_{n_win}x{n_width}pts_{profile_tag}.pkl"
            filepath = _resolve_pickle_path(DEFAULT_RESULTS_DIR, filename)
        else:
            filepath = FilePath(filepath)
            if filepath.exists():
                filepath = _resolve_pickle_path(filepath.parent, filepath.name)

        with open(filepath, 'wb') as f:
            pickle.dump(self.get_scan_results(), f)
        print(f"Scan results saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # 2D-Scan: Amplituden-Abhängigkeit von Waist und Width
    #
    # Anders als scan_win_width_uniformity() (feste Amplituden, Uniformity
    # + Crosstalk werden nur AUSGEWERTET), wird hier an JEDEM (win_input,
    # width)-Gitterpunkt zusätzlich ÜBER die Amplituden-Verhältnisse
    # (r_x, r_y) optimiert (siehe amps_from_ratio()/amps_from_ratios()),
    # um diejenigen Amplituden zu finden, die an genau diesem Punkt das
    # kombinierte Ziel (Uniformity + Crosstalk, wie überall sonst im
    # Optimizer) minimieren. Das Ergebnis - r_x_opt(win_input, width) und
    # r_y_opt(win_input, width) - IST die gesuchte Abhängigkeit der
    # Amplituden von Waist und Width.
    # ------------------------------------------------------------------
    def scan_win_width_amplitude_dependence(self, win_input_range, width_range,
                                             n_win_input=15, n_width=15,
                                             alpha=0.7, r_bounds=(0.0, 2.0), r0=(1.0, 1.0),
                                             warm_start=True, verbose=True,
                                             progress_callback=None, n_jobs=1,
                                             pool_initializer=None, pool_initargs=(),
                                             checkpoint_path=None,
                                             checkpoint_interval_s=scan_checkpoint.CHECKPOINT_INTERVAL_S):
        """
        Findet für jeden Punkt eines (win_input, width)-Gitters die
        Amplituden-Verhältnisse (r_x, r_y) - äußere/innere Amplitude,
        innen fest = 1, siehe amps_from_ratio() - die dort das kombinierte
        Ziel alpha*uniformity + (1-alpha)*eta minimieren, bei fester
        Tonanzahl (N_x, N_y). Uniformity UND Crosstalk werden also
        gemeinsam minimiert (dieselbe Art Gewichtung wie bei
        optimize_win()/optimize_width()/scan_win_width_uniformity()
        im übrigen Optimizer, dort mit anderen Default-Werten für alpha).

        Das ist die Umkehrung von scan_win_width_uniformity(): dort waren
        die Amplituden fest und Uniformity/Crosstalk wurden nur an jedem
        Punkt ausgewertet. Hier wird an JEDEM Punkt eine eigene kleine
        2-Parameter-Optimierung (r_x, r_y) durchgeführt (Nelder-Mead,
        wie optimize_win()/optimize_width()) - das Ergebnis ist direkt
        die gesuchte Abhängigkeit: r_x_opt(win_input, width) und
        r_y_opt(win_input, width).

        alpha: Gewichtung des pro Punkt minimierten Ziels,
        alpha*uniformity + (1-alpha)*eta. Default alpha=0.7 - Uniformity
        UND Crosstalk werden gemeinsam minimiert, mit etwas mehr Gewicht
        auf Crosstalk als bei optimize_width()/optimize_combined()
        (dort Default 0.9). alpha=1.0 würde stattdessen NUR die Uniformity
        minimieren (Crosstalk wird in
        jedem Fall mit den gefundenen r_x/r_y ausgewertet und mitgespeichert,
        unabhängig davon, ob es in die Optimierung eingeht).

        r_bounds: Schranken, die für r_x UND r_y gleichermaßen gelten
        (beide sind Außen/Innen-Verhältnisse derselben Art).

        warm_start: Falls True (Default) startet die Optimierung jedes
        Punkts beim (r_x, r_y)-Optimum des VORHERIGEN Punkts (Scan-
        Reihenfolge: width äußere Schleife, win_input innere Schleife)
        statt immer bei r0 - benachbarte Gitterpunkte haben meist
        ähnliche Optima, das beschleunigt die Konvergenz spürbar und ist
        robuster gegen lokale Minima. r0 wird nur für den allerersten
        Punkt (und bei warm_start=False für jeden Punkt) verwendet.

        WICHTIG - Kosten: Für jeden der n_win_input * n_width Gitterpunkte
        läuft eine vollständige Nelder-Mead-Optimierung (typischerweise
        30-100 _evaluate()-Aufrufe für ein 2-Parameter-Problem). Das ist
        deutlich teurer als scan_win_width_uniformity() (1 Auswertung pro
        Punkt) - für einen ersten Durchlauf n_win_input/n_width eher klein
        halten (z.B. 12-20) und n_grid moderat (150-250).

        progress_callback: wie bei scan_win_width_uniformity(), optionales
        Callable(done, total) -> bool|None; explizites False bricht den
        Scan kooperativ ab (bereits berechnete, schon fertige Punkte
        bleiben erhalten; bei n_jobs>1 können bereits laufende Worker-
        Prozesse nicht mitten in der Optimierung abgebrochen werden -
        deren Ergebnis wird noch übernommen, nur NICHT GESTARTETE Punkte
        entfallen).

        pool_initializer, pool_initargs: nur relevant bei n_jobs>1.
        Werden unverändert an ProcessPoolExecutor(initializer=...,
        initargs=...) durchgereicht und einmal pro Worker-PROZESS (nicht
        pro Punkt) beim Start ausgeführt. Grund: ein evtl. im Hauptprozess
        aktiver Monkey-Patch (z.B. use_torch.patch(), das
        dyn_gaussian_2d_weighted_distance_from_centers/
        dyn_airy_2d_weighted_distance_from_centers im Hauptprozess auf
        eine GPU/PyTorch-Version umbiegt) gilt NUR für den Hauptprozess -
        jeder Worker-Prozess importiert das Modul beim Start frisch/
        unge patcht. Ohne pool_initializer würden Worker-Prozesse bei
        n_jobs>1 also unbemerkt auf die NumPy-Variante zurückfallen, obwohl
        im Hauptprozess CUDA aktiv aussieht. Beispiel:
        `scan_win_width_amplitude_dependence(..., n_jobs=4,
        pool_initializer=use_torch.patch)` sorgt dafür, dass jeder der 4
        Worker-Prozesse den Patch selbst nochmal anwendet.

        n_jobs: Anzahl paralleler Prozesse für die äußere (win_input,
        width)-Schleife. JEDER Gitterpunkt ist komplett unabhängig
        (eigenes Grid, eigene Optimierung) - das lässt sich daher ohne
        Weiteres auf mehrere CPU-Kerne verteilen (ProcessPoolExecutor,
        kein GPU-Nutzen hier: die Rechnung ist vergleichsweise kleine,
        viele-Male-wiederholte NumPy-Arbeit mit SciPy-Optimierung
        dazwischen - dafür ist Multiprocessing über mehrere CPU-Kerne der
        praktikable Hebel, nicht CUDA/GPU).

        - n_jobs=1 (Default): sequentiell im Hauptprozess, WIE BISHER,
          inkl. warm_start (siehe oben) - unverändertes Verhalten.
        - n_jobs=N (N>1): verteilt alle Punkte auf N Worker-Prozesse.
          warm_start wird dabei automatisch ignoriert (jeder Punkt läuft
          unabhängig und startet immer bei r0), da die Bearbeitungs-
          reihenfolge unter mehreren Prozessen nicht mehr sequentiell
          garantiert ist. Erwarteter Speedup: nahezu linear mit der
          Kernzahl abzüglich Prozess-Overhead (z.B. ~6-7x auf 8 Kernen).
        - n_jobs=-1: nutzt alle verfügbaren CPU-Kerne (os.cpu_count()).

        WICHTIG für n_jobs>1 unter Windows (multiprocessing nutzt dort
        'spawn' statt 'fork'): der Aufruf MUSS innerhalb eines
        `if __name__ == "__main__":`-Blocks erfolgen (wie in
        WinWidthAmpScan_StartDialog.py bereits der Fall), sonst startet
        jeder Kindprozess das Skript erneut von vorne.

        checkpoint_path / checkpoint_interval_s: wie bei
        scan_win_width_uniformity() - falls checkpoint_path gesetzt ist,
        wird alle checkpoint_interval_s Sekunden (Default 1h) der bisherige
        Fortschritt (alle 4 Grids: uniformity, crosstalk, r_x, r_y) unter
        genau diesem Pfad gespeichert. Existiert dort bereits eine zum
        aktuellen Scan (identisches Gitter/N_x/N_y/alpha/r_bounds)
        passende (Teil-)Datei, wird sie beim Start automatisch geladen und
        NUR die darin noch fehlenden Punkte werden neu berechnet - so
        lässt sich ein abgebrochener/abgestürzter Scan (der hier wegen der
        pro-Punkt-Optimierung besonders lange dauern kann) an der Stelle
        fortsetzen, an der er stehen geblieben ist. Bei warm_start=True
        wird last_r beim Fortsetzen sinnvoll vom letzten bereits
        berechneten Gitterpunkt (in Scan-Reihenfolge) übernommen statt
        wieder bei r0 zu beginnen.

        Speichert das Ergebnis in self.results['scan2d_amp'] und gibt
        (win_input_vals, width_vals, uniformity_grid, crosstalk_grid,
        r_x_grid, r_y_grid) zurück. NaN-Einträge bedeuten ungültige/nicht
        auswertbare bzw. (bei Abbruch) noch nicht berechnete Punkte.
        """
        win_input_vals = np.linspace(win_input_range[0], win_input_range[1], n_win_input)
        width_vals = np.linspace(width_range[0], width_range[1], n_width)

        resumed = scan_checkpoint.load_resumable(
            checkpoint_path, win_input_range, width_range, n_win_input, n_width,
            self.N_x, self.N_y, extra_match=dict(alpha=alpha, r_bounds=r_bounds),
            airy_scale_factor=self.airy_scale_factor,
            optics_match=dict(n_grid=self.n_grid), verbose=verbose,
        )
        if resumed is not None:
            uniformity_grid = np.asarray(resumed['uniformity_grid'], dtype=float).copy()
            crosstalk_grid = np.asarray(resumed['crosstalk_grid'], dtype=float).copy()
            r_x_grid = np.asarray(resumed['r_x_grid'], dtype=float).copy()
            r_y_grid = np.asarray(resumed['r_y_grid'], dtype=float).copy()
            n_done_before = scan_checkpoint.count_done(uniformity_grid)
            if verbose:
                print(f"[Checkpoint] Setze Scan fort: {n_done_before}/{uniformity_grid.size} "
                      f"Punkte bereits vorhanden ({checkpoint_path}).")
        else:
            uniformity_grid = np.full((n_width, n_win_input), np.nan)
            crosstalk_grid = np.full((n_width, n_win_input), np.nan)
            r_x_grid = np.full((n_width, n_win_input), np.nan)
            r_y_grid = np.full((n_width, n_win_input), np.nan)
            n_done_before = 0
        ckpt = scan_checkpoint.CheckpointWriter(checkpoint_path, checkpoint_interval_s, verbose=verbose)

        def _current_results_amp():
            return dict(
                win_input_vals=win_input_vals, width_vals=width_vals,
                uniformity_grid=uniformity_grid, crosstalk_grid=crosstalk_grid,
                r_x_grid=r_x_grid, r_y_grid=r_y_grid,
                alpha=alpha, r_bounds=r_bounds,
                N_x=self.N_x, N_y=self.N_y, f1=self.f1, f2=self.f2, fLO=self.fLO,
                lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
                profile=self.profile,
                airy_scale_factor=self.airy_scale_factor,
                # n_grid/weighted_n_grid (und der Atom-Offset) bestimmen, WIE fein
                # ausgewertet wurde. Ein fortgesetzter Scan mit anderer Aufloesung
                # haette sonst zwei verschiedene Aufloesungen in einem Datensatz.
                n_grid=self.n_grid,
            )

        total = n_width * n_win_input
        done = 0
        cancelled = False
        last_r = [float(r0[0]), float(r0[1])]
        finite_mask = np.isfinite(r_x_grid) & np.isfinite(r_y_grid)
        if np.any(finite_mask):
            flat_idx = np.flatnonzero(finite_mask.ravel())[-1]
            i_last, j_last = np.unravel_index(flat_idx, r_x_grid.shape)
            last_r = [float(r_x_grid[i_last, j_last]), float(r_y_grid[i_last, j_last])]

        n_jobs_resolved = 1 if n_jobs in (None, 0) else n_jobs
        if n_jobs_resolved == -1:
            n_jobs_resolved = os.cpu_count() or 1

        if verbose:
            print("\n" + "=" * 60)
            print(f"2D Amplituden-Abhängigkeits-Scan: {n_win_input}x{n_width} Punkte, "
                  f"pro Punkt Optimierung von (r_x, r_y), N_x={self.N_x}, N_y={self.N_y}, "
                  f"alpha={alpha}, n_jobs={n_jobs_resolved}")
            print("=" * 60)

        if n_jobs_resolved <= 1:
            # ------------------------------------------------------------
            # Sequentiell im Hauptprozess, mit warm_start (siehe Docstring)
            # ------------------------------------------------------------
            for i, width_val in enumerate(width_vals):
                if cancelled:
                    break
                for j, win_input_val in enumerate(win_input_vals):
                    if np.isfinite(uniformity_grid[i, j]) and np.isfinite(r_x_grid[i, j]) and np.isfinite(r_y_grid[i, j]):
                        # bereits aus Checkpoint vorhanden - nicht neu berechnen, aber
                        # last_r fuer warm_start konsistent mitfuehren
                        if warm_start:
                            last_r = [float(r_x_grid[i, j]), float(r_y_grid[i, j])]
                    else:
                        try:
                            win_eff = self.win_input_to_win(win_input_val)
                        except ValueError:
                            win_eff = None

                        if win_eff is not None:
                            point_grid = self._build_dynamic_grid(win_eff, width_val)

                            def objective(p, win_eff=win_eff, width_val=width_val, point_grid=point_grid):
                                amps = amps_from_ratios(p[0], p[1], self.N_x, self.N_y)
                                val = self._evaluate(win_eff, width_val, amps=amps, grid=point_grid)
                                if val is None:
                                    return 1e10
                                return alpha * val['uniformity'] + (1 - alpha) * val['eta']

                            x0 = last_r if warm_start else [float(r0[0]), float(r0[1])]
                            result = minimize(
                                objective, x0=x0, method='Nelder-Mead',
                                bounds=[r_bounds, r_bounds],
                                options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300},
                            )
                            r_opt = result.x
                            amps_opt = amps_from_ratios(r_opt[0], r_opt[1], self.N_x, self.N_y)
                            details = self._evaluate(win_eff, width_val, amps=amps_opt, grid=point_grid)

                            if details is not None:
                                uniformity_grid[i, j] = details['uniformity']
                                crosstalk_grid[i, j] = details['eta']
                                r_x_grid[i, j] = r_opt[0]
                                r_y_grid[i, j] = r_opt[1]
                                if warm_start:
                                    last_r = [float(r_opt[0]), float(r_opt[1])]

                    done += 1
                    if progress_callback is not None:
                        if progress_callback(done, total) is False:
                            cancelled = True
                            break
                    ckpt.maybe_save(_current_results_amp, done=done, total=total)
                if verbose and n_width >= 10 and (i % max(1, n_width // 10) == 0):
                    print(f"  ... width row {i + 1}/{n_width}")

        else:
            # ------------------------------------------------------------
            # Parallel über mehrere Prozesse (ProcessPoolExecutor) - jeder
            # Gitterpunkt ist unabhängig, warm_start entfällt (jeder Punkt
            # startet bei r0), siehe Docstring.
            # ------------------------------------------------------------
            optimizer_kwargs = {k: getattr(self, k) for k in self.DEFAULTS}
            tasks = [
                (i, j, win_input_val, width_val, list(r0), r_bounds, alpha, optimizer_kwargs)
                for i, width_val in enumerate(width_vals)
                for j, win_input_val in enumerate(win_input_vals)
                if not (np.isfinite(uniformity_grid[i, j]) and np.isfinite(r_x_grid[i, j])
                        and np.isfinite(r_y_grid[i, j]))
            ]
            done = n_done_before

            if verbose:
                skipped = total - len(tasks)
                skip_msg = f" ({skipped} bereits aus Checkpoint vorhanden)" if skipped else ""
                print(f"Verteile {len(tasks)} Punkte auf {n_jobs_resolved} Prozesse{skip_msg} "
                      f"(warm_start wird bei n_jobs>1 ignoriert)...")

            if not tasks:
                if verbose:
                    print("  Alle Punkte bereits vorhanden - nichts zu tun.")
            else:
                with ProcessPoolExecutor(max_workers=n_jobs_resolved,
                                          initializer=pool_initializer,
                                          initargs=pool_initargs) as executor:
                    futures = {executor.submit(_amp_dependence_worker, task): task for task in tasks}
                    for future in as_completed(futures):
                        try:
                            i, j, uniformity, eta, r_x, r_y = future.result()
                        except Exception:
                            task = futures[future]
                            i, j = task[0], task[1]
                            uniformity = eta = r_x = r_y = None

                        if uniformity is not None:
                            uniformity_grid[i, j] = uniformity
                            crosstalk_grid[i, j] = eta
                            r_x_grid[i, j] = r_x
                            r_y_grid[i, j] = r_y

                        done += 1
                        if progress_callback is not None and not cancelled:
                            if progress_callback(done, total) is False:
                                cancelled = True
                                # nur noch nicht gestartete Futures lassen sich wirklich
                                # abbrechen - bereits laufende Worker liefern ihr
                                # Ergebnis trotzdem noch (wird oben weiter verwertet)
                                for f in futures:
                                    f.cancel()
                        ckpt.maybe_save(_current_results_amp, done=done, total=total)
                        if verbose and total >= 10 and done % max(1, total // 10) == 0:
                            print(f"  ... {done}/{total} Punkte fertig")

        if ckpt.active:
            ckpt.maybe_save(_current_results_amp, done=done, total=total, force=True)

        self.results['scan2d_amp'] = dict(
            win_input_vals=win_input_vals, width_vals=width_vals,
            uniformity_grid=uniformity_grid, crosstalk_grid=crosstalk_grid,
            r_x_grid=r_x_grid, r_y_grid=r_y_grid,
            alpha=alpha, r_bounds=r_bounds,
        )

        if verbose:
            valid = np.isfinite(uniformity_grid)
            status = "cancelled" if cancelled else "completed"
            print(f"Scan {status} ({np.sum(valid)}/{uniformity_grid.size} valid points, "
                  f"{done}/{total} computed).")
            print("=" * 60)

        return win_input_vals, width_vals, uniformity_grid, crosstalk_grid, r_x_grid, r_y_grid

    def get_scan_amp_results(self):
        """
        Wie get_scan_results(), aber für scan_win_width_amplitude_dependence():
        gibt ein eigenständiges dict zurück (inkl. N_x, N_y, f1, f2, fLO,
        lambda_opt, theta_max, f_band), das
        multitone_amplitude_dependence_plots.AmplitudeScanPlotter direkt
        zum Plotten übergeben werden kann, ohne den Scan neu zu berechnen.
        """
        if 'scan2d_amp' not in self.results:
            raise RuntimeError("scan_win_width_amplitude_dependence() must be called first.")
        res = dict(self.results['scan2d_amp'])  # shallow copy, don't mutate self.results
        res.update(
            N_x=self.N_x, N_y=self.N_y,
            f1=self.f1, f2=self.f2, fLO=self.fLO,
            lambda_opt=self.lambda_opt, theta_max=self.theta_max, f_band=self.f_band,
            profile=self.profile,
            airy_scale_factor=self.airy_scale_factor,
            # n_grid/weighted_n_grid (und der Atom-Offset) bestimmen, WIE fein
            # ausgewertet wurde. Ein fortgesetzter Scan mit anderer Aufloesung
            # haette sonst zwei verschiedene Aufloesungen in einem Datensatz.
            n_grid=self.n_grid,
        )
        return res

    def save_scan_amp_results(self, filepath=None):
        """
        Wie save_scan_results(), aber für get_scan_amp_results() (Amplituden-
        Abhängigkeits-Scan). Default-Dateiname bei filepath=None:
        "scan_amp_data_N{N_x}x{N_y}_{n_win}x{n_width}pts_{Airy|Gauss}.pkl" in
        DEFAULT_RESULTS_DIR (dem "Results"-Ordner neben diesem Skript, wird
        bei Bedarf automatisch angelegt), mit demselben Kollisionsschutz
        (Nachfrage/automatischer Zähler-Suffix) wie save_scan_results(). Der
        Dateiname kodiert damit direkt Tonanzahl, Anzahl der Gitterpunkte UND
        das verwendete Strahlprofil (Airy/Gauss).
        """
        if filepath is None:
            res = self.results.get('scan2d_amp', {})
            n_win = len(res.get('win_input_vals', []))
            n_width = len(res.get('width_vals', []))
            profile_tag = "Airy" if self.profile == "airy" else "Gauss" if self.profile == "gaussian" else self.profile
            filename = f"scan_amp_data_N{self.N_x}x{self.N_y}_{n_win}x{n_width}pts_{profile_tag}.pkl"
            filepath = _resolve_pickle_path(DEFAULT_RESULTS_DIR, filename)
        else:
            filepath = FilePath(filepath)
            if filepath.exists():
                filepath = _resolve_pickle_path(filepath.parent, filepath.name)

        with open(filepath, 'wb') as f:
            pickle.dump(self.get_scan_amp_results(), f)
        print(f"Amplitude-dependence scan results saved: {filepath}")
        return filepath

    # ------------------------------------------------------------------
    # Komfort: alles nacheinander ausführen
    # ------------------------------------------------------------------
    def run_all(self, show=True, save=True):
        """Führt Ausgangsberechnung + alle drei Optimierungsstufen nacheinander aus."""
        self.compute_initial()
        self.plot_initial(show=show, save=save)
        self.optimize_win(show=show, save=save)
        self.optimize_width(show=show, save=save)
        self.optimize_combined(show=show, save=save)
        self.summary()

    def summary(self):
        """Druckt eine Zusammenfassung aller bisher berechneten Stufen."""
        print("\n" + "=" * 60)
        print("ZUSAMMENFASSUNG ALLER OPTIMIERUNGEN")
        print("=" * 60)
        if 'initial' in self.results:
            r = self.results['initial']
            print(f"Original:              Uniformity = {r['uniformity']*100:.2f}%, Crosstalk = {r['eta']*100:.3f}%")
        if 'win_opt' in self.results:
            r = self.results['win_opt']
            print(f"win-optimiert:         Uniformity = {r['uniformity']*100:.2f}%, Crosstalk = {r['eta']*100:.3f}%")
        if 'width_opt' in self.results:
            r = self.results['width_opt']
            print(f"width-optimiert:       Uniformity = {r['uniformity']*100:.2f}%, Crosstalk = {r['eta']*100:.3f}%")
        if 'combined_opt' in self.results:
            r = self.results['combined_opt']
            print(f"win+width+amp-optim.:  Uniformity = {r['uniformity']*100:.2f}%, Crosstalk = {r['eta']*100:.3f}%")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Generische Plot-Helfer (von allen Stufen wiederverwendet)
    # ------------------------------------------------------------------
    def _filetag(self):
        return f"{self.offset*1e-6, self.width*1e-6, self.N_x, self.N_y}"

    def _plot_intensity(self, res, title, filename, show=True, save=True, xlim=None, ylim=None):
        """2-Panel-Plot: Intensitätsverteilung links, Schnitte entlang x/y rechts."""
        I = res['I']
        centers_x, centers_y = res['centers_x'], res['centers_y']
        r_center = res['r_center']
        x, y = self.x, self.y

        fig = plt.figure(figsize=(12, 7))
        fig.suptitle(title, fontsize=13, fontweight='bold')

        ax_2d = plt.subplot(1, 2, 1)
        im = ax_2d.imshow(
            I, origin="lower",
            extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
            aspect="equal", cmap="viridis",
        )
        if self.drawpoints:
            ax_2d.scatter(centers_x*1e6, centers_y*1e6, c="red", s=20, label="Spot Centers")

        self._draw_region_overlay(ax_2d, res, r_center)

        ax_2d.set_xlabel("$x$ (µm)")
        ax_2d.set_ylabel("$y$ (µm)")
        if xlim is not None:
            ax_2d.set_xlim(*xlim)
        if ylim is not None:
            ax_2d.set_ylim(*ylim)
        ax_2d.legend()
        plt.colorbar(im, ax=ax_2d, label="Normalized Intensity")

        ax_x = plt.subplot(2, 2, 2)
        ax_x.plot(x*1e6, I[self.mid_y_idx, :], 'b-', linewidth=2)
        ax_x.set_xlabel("$x$ (µm)")
        ax_x.set_ylabel("Intensity")
        ax_x.set_title("Cut along x through center")
        ax_x.grid(True, alpha=0.3)

        ax_y = plt.subplot(2, 2, 4)
        ax_y.plot(y*1e6, I[:, self.mid_x_idx], 'g-', linewidth=2)
        ax_y.set_xlabel("$y$ (µm)")
        ax_y.set_ylabel("Intensity")
        ax_y.set_title("Cut along y through center")
        ax_y.grid(True, alpha=0.3)

        plt.tight_layout()
        self._finish_figure(fig, filename, show, save)

    def _plot_neighbor(self, res, title, filename, show=True, save=True):
        """1-Panel-Plot der Nachbar-(Crosstalk-)Intensität mit Maske und Mittelpunkt."""
        I_neighbor = res['I_neighbor']
        centers_x, centers_y = res['centers_x'], res['centers_y']
        r_center = res['r_center']
        x, y = self.x, self.y

        fig = plt.figure(figsize=(8, 7))
        fig.suptitle(title, fontsize=13, fontweight='bold')

        ax = plt.subplot(1, 1, 1)
        im = ax.imshow(
            I_neighbor, origin="lower",
            extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
            aspect="equal", cmap="viridis",
        )
        if self.drawpoints:
            ax.scatter(centers_x*1e6, centers_y*1e6, c="red", s=20, label="Spots")

        half_side = self.pitch / 2
        rect = Rectangle(
            ((r_center - half_side)*1e6, (r_center - half_side)*1e6),
            self.pitch*1e6, self.pitch*1e6,
            edgecolor="red", facecolor="none", linewidth=2, label="Mask",
        )
        ax.add_patch(rect)
        ax.plot(r_center*1e6, r_center*1e6, "r+", markersize=12, markeredgewidth=2, label="Center")

        plt.colorbar(im, ax=ax, label="Intensity (Neighbors)")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.legend()
        plt.tight_layout()
        self._finish_figure(fig, filename, show, save)

    def _draw_region_overlay(self, ax, res, r_center):
        """Zeichnet je nach Modus ConvexHull / Level-Konturen / Crosstalk+Uniformity-Rechtecke."""
        if self.useconvexhull and res.get('hull_vertices') is not None:
            hv = res['hull_vertices']
            hv_closed = np.vstack([hv, hv[0]])
            ax.plot(hv_closed[:, 0]*1e6, hv_closed[:, 1]*1e6, 'b--', linewidth=2, label="Uniformity region")
        elif self.use_Levels:
            cs = ax.contour(
                self.X*1e6, self.Y*1e6, res['I'],
                levels=self.levels, colors=['blue', 'red', 'green'], linewidths=2,
            )
            ax.clabel(cs, inline=True, fontsize=8)
        else:
            half_side = self.pitch / 2
            rect = Rectangle(
                ((r_center - half_side)*1e6, (r_center - half_side)*1e6),
                self.pitch*1e6, self.pitch*1e6,
                edgecolor="red", facecolor="none", linewidth=2, label="Crosstalk region",
            )
            ax.add_patch(rect)

            side_u = self._resolve_uniformity_side(res['centers_x'], res['centers_y'], r_center)
            half_side_u = side_u / 2
            rect_u = Rectangle(
                ((r_center - half_side_u)*1e6, (r_center - half_side_u)*1e6),
                side_u*1e6, side_u*1e6,
                edgecolor="cyan", facecolor="none", linewidth=2, label="Uniformity region",
            )
            ax.add_patch(rect_u)

            ax.plot(r_center*1e6, r_center*1e6, "r+", markersize=12, markeredgewidth=2, label="Center")

    def _finish_figure(self, fig, filename, show, save, dpi=150):
        if save:
            out_file = self.out_dir / filename
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {out_file.name}")
        if show:
            plt.show()
        else:
            plt.close(fig)


# ======================================================================
# Worker-Funktion für scan_win_width_amplitude_dependence(n_jobs>1)
#
# MUSS auf Modulebene stehen (nicht als Methode/Closure) - nur so kann
# sie von einem ProcessPoolExecutor an die Worker-Prozesse gepickelt
# werden (auf Windows zwingend, da dort 'spawn' statt 'fork' verwendet
# wird und Worker-Prozesse das Zielobjekt daher importieren/pickeln
# müssen können, statt einfach den Speicher zu teilen).
#
# Jeder Worker-Prozess baut sich aus optimizer_kwargs (siehe
# MultitoneFlatTopOptimizer.DEFAULTS) eine EIGENE, leichte
# MultitoneFlatTopOptimizer-Instanz - das ist billig (kein Grid wird
# dabei gebaut, das passiert erst on-demand in _build_dynamic_grid()
# für genau diesen einen Punkt) und vermeidet, dass die (nicht
# picklebare) self-Closure der eigentlichen objective()-Funktion über
# Prozessgrenzen transportiert werden müsste.
# ======================================================================
def _amp_dependence_worker(task):
    """
    task: (i, j, win_input_val, width_val, x0, r_bounds, alpha, optimizer_kwargs)

    Führt GENAU EINEN (win_input, width)-Punkt aus scan_win_width_amplitude_dependence()
    aus: baut das passende dynamische Grid, optimiert (r_x, r_y) via Nelder-Mead
    und wertet das Ergebnis aus.

    Gibt (i, j, uniformity, crosstalk, r_x, r_y) zurück, oder (i, j, None, None, None, None)
    bei einem ungültigen Punkt (z.B. win_input <= 0 oder leere Maske).
    """
    i, j, win_input_val, width_val, x0, r_bounds, alpha, optimizer_kwargs = task

    opt = MultitoneFlatTopOptimizer(out_dir=".", **optimizer_kwargs)
    try:
        win_eff = opt.win_input_to_win(win_input_val)
    except ValueError:
        return (i, j, None, None, None, None)

    point_grid = opt._build_dynamic_grid(win_eff, width_val)

    def objective(p):
        amps = amps_from_ratios(p[0], p[1], opt.N_x, opt.N_y)
        val = opt._evaluate(win_eff, width_val, amps=amps, grid=point_grid)
        if val is None:
            return 1e10
        return alpha * val['uniformity'] + (1 - alpha) * val['eta']

    result = minimize(
        objective, x0=list(x0), method='Nelder-Mead',
        bounds=[r_bounds, r_bounds],
        options={'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300},
    )
    r_opt = result.x
    amps_opt = amps_from_ratios(r_opt[0], r_opt[1], opt.N_x, opt.N_y)
    details = opt._evaluate(win_eff, width_val, amps=amps_opt, grid=point_grid)
    if details is None:
        return (i, j, None, None, None, None)

    return (i, j, float(details['uniformity']), float(details['eta']),
            float(r_opt[0]), float(r_opt[1]))


# ======================================================================
# Beispiel-Nutzung (nur wenn das Skript direkt ausgeführt wird)
# ======================================================================
if __name__ == "__main__":
    out_dir = r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\PythonCode\Multitone_FlatTop"

    opt = MultitoneFlatTopOptimizer(
        out_dir=out_dir,
        offset=100e6,
        width=0.15e6,
        N_x=2,
        N_y=3,
        win=0.9e-6,
    )

    # Ausgangszustand
    # opt.compute_initial()
    # opt.plot_initial()

    # Optimierungsstufen einzeln aufrufbar:
    # opt.optimize_win()
    # opt.optimize_width()

    # Optional: eigene Startschätzung für die kombinierte Optimierung (win, width,
    # Amplituden) hinterlegen, um sie zu beschleunigen. Ohne diesen Aufruf nutzt
    # optimize_combined() automatisch die Ergebnisse von optimize_win()/optimize_width().
    opt.set_initial_guess(
        win=0.9e-6,
        width=1.5e5,
        amp_x=[1, 1],
        amp_y=[1, 0.1, 1],
    )
    opt.optimize_combined()

    opt.summary()

    # Beispiel: unterschiedliche Startschätzungen für Gauß und Airy hinterlegen
    # und zwischen den Profilen wechseln:
    #
    # opt.set_profile("gaussian")
    # opt.set_initial_guess(win=1.7e-6, width=2.9e5, amp_x=[1, 1, 1], amp_y=[1, 1, 1, 1])
    # opt.optimize_combined()
    #
    # opt.set_profile("airy")
    # opt.set_initial_guess(win=2.3e-6, width=2.9e5, amp_x=[1, 1, 1], amp_y=[1, 1, 1, 1])
    # opt.optimize_combined()

    # Alternativ alles in einem Rutsch:
    # opt.run_all()

    # Parameter später ändern und neu rechnen, z.B. anderes Spot-Raster:
    # opt.set_parameters(N_x=4, N_y=4, pitch=5.0e-6)
    # opt.run_all()

    # Beispiel: 2D-Scan der Uniformity/Crosstalk über Eingangswaist vor der
    # 1. Linse (win_input) und width, bei fester Tonanzahl und festen
    # Amplituden - reine Berechnung, dann Rohdaten für's Plotten sichern
    # (siehe multitone_flattop_scan_plots.py für das eigentliche Plotten):
    #
    # opt.scan_win_width_uniformity(
    #     win_input_range=(0.05e-6, 2.0e-6),
    #     width_range=(0.05e6, 0.8e6),
    #     n_win_input=50, n_width=50,
    #     amps=np.concatenate([np.ones(opt.N_x), np.ones(opt.N_y)]),
    # )
    # opt.save_scan_results("scan_data.pkl")