"""
Multitone FlatTop Optimizer
============================

Objektorientierte Umstrukturierung des ursprünglichen Skripts zur Optimierung
eines Multitone-AOD_GUIs-Spot-Arrays hinsichtlich Uniformity (Gleichförmigkeit der
Intensität im Zentrum) und Crosstalk (Überlapp mit Nachbar-Sites).

Aufbau:
- Reine Mathematik-/Physik-Hilfsfunktionen bleiben als freistehende Funktionen
  (multitone_frequencies, angle_from_frequency, ... , create_neighbourhood).
- Der gesamte zustandsbehaftete Workflow (Parameter, Grid, Ergebnisse, Plots,
  Optimierungsstufen) steckt in der Klasse `MultitoneFlatTopOptimizer`.

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

Zusätzlich: 2D-Scan der Uniformity über (Eingangswaist vor der 1. Linse, width):

    opt.scan_win_width_uniformity(
        win_input_range=(0.05e-6, 2.0e-6),
        width_range=(0.1e6, 0.6e6),
        amps=None,   # oder z.B. np.concatenate([amp_x, amp_y])
    )
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from matplotlib.path import Path
from pathlib import Path as FilePath
from scipy.special import j0, j1
from matplotlib.patches import Rectangle


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


def gaussian_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, sigma, amps):
    """Wie gaussian_2d_distance_from_centers, aber mit individueller Amplitude pro Spot."""
    I = np.zeros_like(X, dtype=float)
    for cx, cy, a in zip(centers_x, centers_y, amps):
        I += a * np.exp(-2 * ((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2))
    return I


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


def airy_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, first_zero_radius, amps):
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


# Verfügbare Strahlprofile: Name -> gewichtete Profilfunktion f(X,Y,cx,cy,scale,amps)
PROFILE_FUNCS = {
    "gaussian": gaussian_2d_weighted_distance_from_centers,
    "airy": airy_2d_weighted_distance_from_centers,
}


# ----------------------------------------------------------------------
# Symmetrische Amplituden-Parametrisierung (pro Achse):
#
# Statt N unabhängiger Amplituden pro Achse zu optimieren, wird die
# Symmetrie um die Achsenmitte ausgenutzt: der innerste Ton (ungerades N)
# bzw. das innerste Paar (gerades N) ist immer auf 1 fixiert. Jeder
# weitere "Ring" nach außen (Spiegelpaar mit gleichem Abstand zur Mitte)
# bekommt genau EINEN freien Verhältnisparameter (Amplitude relativ zum
# inneren Ton = 1). Das reduziert z.B.
#   N=3 (außen-innen-außen)              -> 1 freier Parameter (außen/innen)
#   N=4 (außen-innen-innen-außen)        -> 1 freier Parameter (außen/innen)
#   N=5                                  -> 2 freie Parameter
# auf jeweils die tatsächlich sinnvollen Freiheitsgrade und eliminiert so
# redundante Amplitudenparameter aus der Optimierung.
# ----------------------------------------------------------------------

def symmetric_amp_ring_count(N):
    """Anzahl freier Verhältnisparameter für N symmetrisch angeordnete Töne
    (alles außer dem innersten Ton/Paar, der fest auf 1 steht)."""
    return -(-N // 2) - 1  # == ceil(N/2) - 1, ohne math.ceil-Import


def build_symmetric_amps(N, ratios):
    """
    Baut ein symmetrisches Amplitudenarray der Länge N aus den freien
    Ring-Verhältnisparametern `ratios` (Länge symmetric_amp_ring_count(N)):
    der innerste Ton (N ungerade) bzw. das innerste Paar (N gerade) ist 1,
    jeder weitere Ring nach außen bekommt den nächsten Wert aus `ratios`
    (symmetrisch für beide Seiten). Fehlen Werte (z.B. bei zu kurzem
    ratios-Array), wird für die fehlenden äußeren Ringe 1 angenommen.
    """
    ratios = list(ratios)
    amp = np.ones(N, dtype=float)
    if N % 2 == 1:
        center = N // 2
        for k in range(1, N // 2 + 1):
            r = ratios[k - 1] if (k - 1) < len(ratios) else 1.0
            amp[center - k] = r
            amp[center + k] = r
    else:
        half = N // 2
        for k in range(1, half):
            r = ratios[k - 1] if (k - 1) < len(ratios) else 1.0
            amp[half - 1 - k] = r
            amp[half + k] = r
    return amp


def extract_ring_ratios(amp):
    """
    Kehrt build_symmetric_amps() um: liest aus einem (näherungsweise)
    symmetrischen Amplitudenarray die Ring-Verhältnisparameter aus
    (Mittelwert der beiden Spiegelwerte pro Ring, falls das Array nicht
    exakt symmetrisch ist). Nützlich, um eine per set_initial_guess()
    hinterlegte volle Amplitudenschätzung (Länge N) in die reduzierte
    Ring-Parametrisierung zu übersetzen.
    """
    amp = np.asarray(amp, dtype=float)
    N = len(amp)
    ratios = []
    if N % 2 == 1:
        center = N // 2
        for k in range(1, N // 2 + 1):
            ratios.append(0.5 * (amp[center - k] + amp[center + k]))
    else:
        half = N // 2
        for k in range(1, half):
            ratios.append(0.5 * (amp[half - 1 - k] + amp[half + k]))
    return np.array(ratios, dtype=float)


# ======================================================================
# Klasse: Kapselt Parameter, Grid, Ergebnisse und Optimierungsstufen
# ======================================================================

class MultitoneFlatTopOptimizer:
    """
    Optimiert ein Multitone-AOD_GUIs-Spot-Array hinsichtlich Uniformity und
    Crosstalk. Kapselt Eingangsparameter, Grid-Setup, Ausgangsberechnung
    sowie die einzelnen Optimierungsstufen als separat aufrufbare Methoden.
    """

    # ------------------------------------------------------------------
    # Feste Grenzwerte für die Optimierung (praxis-/hardwarebedingt)
    # ------------------------------------------------------------------
    MAX_TONES_PER_AXIS = 10          # nicht mehr als 10x10 Töne
    WIDTH_BOUNDS = (0.1e6, 0.6e6)      # width nur von 0.1 bis 0.6 MHz
    WIN_BOUNDS = (0, 2.5e-6)         # win (Waist in der Ebene) nur bis 2.5 µm
    WIN_BOUNDS_COMBINED = (0.1e-6, 2.5e-6)  # wie WIN_BOUNDS, aber L-BFGS-B mag echte untere Grenze

    DEFAULTS = dict(
        offset=100e6,
        width=0.3e6,
        N_x=3,
        N_y=4,
        f1=75e-3,
        f2=750e-3,
        fLO=52.88e-3,
        theta_max=43e-3,
        f_band=36e6,
        lambda_opt=795e-9,
        win=1e-6,
        pitch=5.288e-6,
        uniformity_side_length=None,  # None = automatisch "Ton-Quadrat" (siehe _build_masks)
        integration_radius=0.45e-6,
        threshindex=2,
        levels=(np.exp(-2), 0.5, 0.9),
        useconvexhull=False,
        use_Levels=False,
        n_grid=1000,
        profile="airy",       # "gaussian" oder "airy"
        airy_scale_factor=1.19,   # first_zero_radius = airy_scale_factor * win
    )

    def __init__(self, out_dir=None, **params):
        self.out_dir = FilePath(out_dir) if out_dir is not None else FilePath("../AOD_GUIs")
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

        amp_x/amp_y: volle Amplitudenarrays (Länge N_x bzw. N_y), müssen
        symmetrisch um die Achsenmitte sein (innerster Ton bzw. innerstes
        Paar = 1, äußere Töne paarweise gleich) - das entspricht genau der
        Parametrisierung, die optimize_combined() intern verwendet (siehe
        dort: pro Achse nur EIN freier Verhältnisparameter pro Ring). Aus
        amp_x/amp_y werden intern per extract_ring_ratios() automatisch die
        passenden Ring-Verhältnisse für optimize_combined() abgeleitet, z.B.
        für N_x=3: amp_x=[1.05, 1.0, 1.05] -> Verhältnis außen/innen = 1.05.
        Für N_y=4: amp_y=[1.1, 1.0, 1.0, 1.1] -> Verhältnis außen/innen = 1.1.

        profile=None (Default) bezieht sich auf das aktuell aktive Profil
        (self.profile). So lassen sich für 'gaussian' und 'airy' unterschiedliche
        Vermutungen hinterlegen, z.B.:

            opt.set_profile('gaussian')
            opt.set_initial_guess(win=1.8e-6, width=2.9e5, amp_x=[..], amp_y=[..])

            opt.set_profile('airy')
            opt.set_initial_guess(win=2.4e-6, width=2.9e5, amp_x=[..], amp_y=[..])

        Einzelne Felder können weggelassen werden; für sie greift beim Optimieren
        weiterhin der bisherige Fallback (Ergebnis von optimize_win()/optimize_width()
        bzw. Gleichverteilung/Verhältnis 1).
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
        vor dem AOD_GUIs/Teleskop), win_eff im µm-Bereich (Fokus-/Trap-Waist).
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
    def _evaluate(self, win_val, width_val, amps=None):
        """
        Berechnet Intensitätsverteilung, Nachbar-Intensität, Uniformity und
        Crosstalk für gegebenes win, width und optionale Amplituden pro
        Achse (amps = [amp_x_0..amp_x_{N_x-1}, amp_y_0..amp_y_{N_y-1}]).

        Ersetzt die vier fast identischen calculate_uniformity(_width)/
        calculate_overlapp(_width)-Funktionen des Ausgangsskripts durch
        eine einzige Kernberechnung.

        Gibt None zurück, falls die Parameter zu einem ungültigen Zustand
        führen (z.B. leere Maske) - das ist der Penalty-Fall für die
        Optimierung.
        """
        if win_val <= 0:
            return None

        centers_x, centers_y, r_center = self._compute_centers_for_width(width_val)
        profile_func = self._profile_func()
        scale = self._profile_scale(win_val)  # sigma (Gauß) bzw. first_zero_radius (Airy)

        # Amplituden pro Achse -> pro Spot; ohne Angabe wird gleichmäßig (1) gewichtet.
        if amps is not None:
            amps = np.asarray(amps)
            amp_x = amps[:self.N_x]
            amp_y = amps[self.N_x:self.N_x + self.N_y]
            amp_spots = np.repeat(amp_x, self.N_y) * np.tile(amp_y, self.N_x)
        else:
            amp_spots = np.ones(len(centers_x))

        I_new = profile_func(self.X, self.Y, centers_x, centers_y, scale, amp_spots)

        if np.max(I_new) == 0:
            return None
        I_new = I_new / np.max(I_new)

        try:
            I_neighbor = create_neighbourhood(self.X, self.Y, self.pitch, centers_x, centers_y,
                                               w_in=scale, amps=amp_spots, profile_func=profile_func)
            mask_uniformity, mask_crosstalk, hull_vertices = self._build_masks(
                self.X, self.Y, r_center, I_new, centers_x, centers_y
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
    def optimize_width(self, alpha=0.7, x0=None, win_val=None, bounds=None, show=True, save=True, verbose=True):
        """
        Optimiert width (Frequenzabstände der Spots) bei fixem win zur
        Minimierung von Uniformity + Crosstalk.

        win_val: welcher win-Wert dabei benutzt wird. Default: das Ergebnis
        von optimize_win(), falls vorhanden, sonst der aktuelle self.win.

        bounds: Default self.WIDTH_BOUNDS (0.1 - 0.6 MHz) - width nur von 0.1 bis 0.6 MHz.
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
    def optimize_combined(self, alpha=0.7, x0=None, bounds=None, show=True, save=True, verbose=True):
        """
        Optimiert win, width sowie die Amplituden pro Achse gemeinsam - die
        Amplituden werden dabei NICHT als N_x+N_y unabhängige Werte
        behandelt, sondern nutzen die Symmetrie um die Achsenmitte aus, um
        redundante Parameter zu eliminieren (siehe build_symmetric_amps()):
        der innerste Ton je Achse (ungerades N) bzw. das innerste Paar
        (gerades N) ist immer auf 1 fixiert; jeder weitere Ring nach außen
        (Spiegelpaar mit gleichem Abstand zur Mitte) bekommt genau EINEN
        freien Verhältnisparameter (Amplitude relativ zum inneren Ton = 1).

        Beispiele:
        - N=4 (z.B. N_y): innen (die beiden mittleren Töne) = 1, außen
          (die beiden äußeren Töne) = EIN gemeinsames Verhältnis r_außen/innen.
          -> 1 freier Amplitudenparameter statt 4.
        - N=3 (z.B. N_x): außen-innen-außen, die beiden äußeren Töne sind
          gleich und werden im Verhältnis zum inneren Ton (=1) optimiert.
          -> 1 freier Amplitudenparameter statt 3.
        Allgemein: symmetric_amp_ring_count(N) freie Parameter pro Achse.

        Der Optimierungsvektor ist also
        [win, width, ratio_x_1..ratio_x_{K_x}, ratio_y_1..ratio_y_{K_y}]
        mit K_x=symmetric_amp_ring_count(N_x), K_y=symmetric_amp_ring_count(N_y).

        Startwerte (Priorität, falls x0 nicht explizit übergeben wird):
        1. per set_initial_guess() hinterlegte, profilabhängige Vermutung
           (volle amp_x/amp_y-Arrays werden dabei per extract_ring_ratios()
           automatisch in die reduzierte Ring-Parametrisierung übersetzt)
        2. Ergebnisse von optimize_win()/optimize_width(), falls vorhanden
        3. aktuelles self.win/self.width, alle Ring-Verhältnisse = 1

        Ein guter Startwert (z.B. aus set_initial_guess()) reduziert bei
        L-BFGS-B typischerweise deutlich die Anzahl nötiger Funktions-
        auswertungen bis zur Konvergenz.

        bounds: Default self.WIN_BOUNDS_COMBINED (0.1 - 2.5 µm) für win,
        self.WIDTH_BOUNDS (0.1 - 0.6 MHz) für width, (0, 2) für jeden
        Amplituden-Verhältnisparameter.
        """
        K_x = symmetric_amp_ring_count(self.N_x)
        K_y = symmetric_amp_ring_count(self.N_y)

        win0 = self.results.get('win_opt', {}).get('win_optimal', self.win)
        width0 = self.results.get('width_opt', {}).get('width_optimal', self.width)
        ratio_x0 = np.ones(K_x)
        ratio_y0 = np.ones(K_y)

        # Eigene, per set_initial_guess() hinterlegte Vermutung (profilabhängig)
        # überschreibt die obigen Fallbacks, sofern gesetzt. Volle amp_x/amp_y-
        # Arrays werden dabei in Ring-Verhältnisse übersetzt.
        guess = self._initial_guesses.get(self.profile, {})
        win0 = guess.get('win', win0)
        width0 = guess.get('width', width0)
        if 'amp_x' in guess:
            ratio_x0 = extract_ring_ratios(guess['amp_x'])[:K_x]
        if 'amp_y' in guess:
            ratio_y0 = extract_ring_ratios(guess['amp_y'])[:K_y]

        if x0 is None:
            x0 = np.concatenate(([win0, width0], ratio_x0, ratio_y0))
        if bounds is None:
            bounds = [self.WIN_BOUNDS_COMBINED, self.WIDTH_BOUNDS] + [(0.0, 2.0)] * (K_x + K_y)

        def _amps_from_ratios(p):
            ratio_x = p[2:2 + K_x]
            ratio_y = p[2 + K_x:2 + K_x + K_y]
            amp_x = build_symmetric_amps(self.N_x, ratio_x)
            amp_y = build_symmetric_amps(self.N_y, ratio_y)
            return np.concatenate([amp_x, amp_y])

        def objective(p):
            val = self._evaluate(p[0], p[1], amps=_amps_from_ratios(p))
            if val is None:
                return 1e10
            return alpha * val['uniformity'] + (1 - alpha) * val['eta']

        if verbose:
            print("\n" + "=" * 60)
            print("DRITTE ITERATION: Optimierung von win, width und Amplituden-Verhältnissen pro Achse")
            print(f"Freie Amplitudenparameter: {K_x} (N_x={self.N_x}) + {K_y} (N_y={self.N_y})")
            print("=" * 60)
            print("\nStarte kombinierte Optimierung (win + width + Amplituden-Verhältnisse)...")

        result = minimize(
            objective, x0=x0, method='L-BFGS-B', bounds=bounds,
            options={'ftol': 1e-9, 'maxiter': 2000},
        )

        win_opt, width_opt = result.x[0], result.x[1]
        ratio_x_opt = result.x[2:2 + K_x]
        ratio_y_opt = result.x[2 + K_x:2 + K_x + K_y]
        amp_x_opt = build_symmetric_amps(self.N_x, ratio_x_opt)
        amp_y_opt = build_symmetric_amps(self.N_y, ratio_y_opt)
        amps_opt = np.concatenate([amp_x_opt, amp_y_opt])
        details = self._evaluate(win_opt, width_opt, amps=amps_opt)

        if verbose:
            print(f"\nOptimales win: {win_opt:.6e} m")
            print(f"Optimale width: {width_opt:.6e} Hz")
            print(f"Optimierte Ring-Verhältnisse amp_x (außen->innen): {ratio_x_opt}")
            print(f"Optimierte Ring-Verhältnisse amp_y (außen->innen): {ratio_y_opt}")
            print(f"Daraus resultierende amp_x: {amp_x_opt}")
            print(f"Daraus resultierende amp_y: {amp_y_opt}")
            print(f"Uniformity (win+width+amps): {details['uniformity']:.4f} ({details['uniformity']*100:.2f}%)")
            print(f"Crosstalk (win+width+amps): {details['eta']:.4f} ({details['eta']*100:.3f}%)")
            print(f"Funktionsauswertungen (nfev): {result.nfev}, Iterationen (nit): {result.nit}")

        # 'details' enthält bereits 'amps' (volles, aus den Ring-Verhältnissen
        # abgeleitetes Array, aus _evaluate); win_optimal/width_optimal sowie
        # die Ring-Verhältnisse selbst ergänzen.
        self.results['combined_opt'] = dict(
            win_optimal=win_opt, width_optimal=width_opt,
            ratio_x_optimal=ratio_x_opt, ratio_y_optimal=ratio_y_opt,
            nfev=result.nfev, nit=result.nit, **details,
        )

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
                                   amps=None, alpha=0.7, show=True, save=True, verbose=True,
                                   progress_callback=None):
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

        Stores the result in self.results['scan2d'] and returns
        (win_input_vals, width_vals, uniformity_grid, crosstalk_grid). NaN
        entries in the grids mean invalid/non-evaluable or not-yet-computed
        (cancelled) parameter combinations.

        If show or save is True, plots BOTH heatmaps side by side in ONE
        figure (plot_scan2d_combined()) - this is the default "final" plot.
        The individual plot_scan2d_uniformity()/plot_scan2d_crosstalk() are
        still available separately if only one heatmap is needed.
        """
        win_input_vals = np.linspace(win_input_range[0], win_input_range[1], n_win_input)
        width_vals = np.linspace(width_range[0], width_range[1], n_width)

        uniformity_grid = np.full((n_width, n_win_input), np.nan)
        crosstalk_grid = np.full((n_width, n_win_input), np.nan)
        total = n_width * n_win_input
        done = 0
        cancelled = False

        if verbose:
            print("\n" + "=" * 60)
            print(f"2D scan: uniformity & crosstalk over win_input ({n_win_input} points) x "
                  f"width ({n_width} points), N_x={self.N_x}, N_y={self.N_y}")
            print("=" * 60)

        for i, width_val in enumerate(width_vals):
            if cancelled:
                break
            for j, win_input_val in enumerate(win_input_vals):
                try:
                    win_eff = self.win_input_to_win(win_input_val)
                except ValueError:
                    win_eff = None
                if win_eff is not None:
                    details = self._evaluate(win_eff, width_val, amps=amps)
                    if details is not None:
                        uniformity_grid[i, j] = details['uniformity']
                        crosstalk_grid[i, j] = details['eta']

                done += 1
                if progress_callback is not None:
                    if progress_callback(done, total) is False:
                        cancelled = True
                        break
            if verbose and n_width >= 10 and (i % max(1, n_width // 10) == 0):
                print(f"  ... width row {i + 1}/{n_width}")

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

        if show or save:
            self.plot_scan2d_combined(show=show, save=save)

        return win_input_vals, width_vals, uniformity_grid, crosstalk_grid

    # Font sizes / dpi used for the scan-2D figures so the saved PNGs are
    # suitable for direct inclusion in a LaTeX document (readable at
    # reduced print size, serif font matching typical LaTeX body text).
    SCAN2D_RC = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
    SCAN2D_SAVE_DPI = 300

    def plot_scan2d_uniformity(self, show=True, save=True, cmap="viridis", vmax=None):
        """Plots the uniformity heatmap from scan_win_width_uniformity(). See
        _plot_scan2d_metric() for the shared layout/interactivity details."""
        self._plot_scan2d_metric(
            grid_key="uniformity_grid", colorbar_label="Uniformity (σ/μ)",
            title_metric="Uniformity", filename_suffix="Uniformity",
            cmap=cmap, vmax=vmax, show=show, save=save,
        )

    def plot_scan2d_crosstalk(self, show=True, save=True, cmap="YlOrRd", vmax=None):
        """Plots the crosstalk heatmap from scan_win_width_uniformity(). See
        _plot_scan2d_metric() for the shared layout/interactivity details.
        Default colormap avoids near-black low values (unlike e.g. 'magma'),
        which read poorly for a quantity where black could be misread as
        "no data"."""
        self._plot_scan2d_metric(
            grid_key="crosstalk_grid", colorbar_label="Crosstalk (η)",
            title_metric="Crosstalk", filename_suffix="Crosstalk",
            cmap=cmap, vmax=vmax, show=show, save=save,
        )

    def plot_scan2d(self, show=True, save=True, cmap="viridis", vmax=None):
        """Deprecated alias for plot_scan2d_uniformity(), kept for backward compatibility."""
        self.plot_scan2d_uniformity(show=show, save=save, cmap=cmap, vmax=vmax)

    def _scan2d_info_lines(self, win_input_val, width_val, win_eff, details, amps):
        """Builds the info-panel text lines for a single (win_input, width)
        scan point, using LaTeX-style math symbols/subscripts instead of
        underscored parameter names (e.g. omega_in instead of win_input).
        Shared by _plot_scan2d_metric() and plot_scan2d_combined()."""
        lines = [
            rf"$\omega_{{\mathrm{{in}}}}$ (before lenses):  {win_input_val*1e3:.4f} mm",
            rf"$\omega'$ (at focus):            {win_eff*1e6:.4f} µm" if win_eff is not None
            else r"$\omega'$ (at focus):            invalid ($\omega_{\mathrm{in}} \leq 0$)",
            rf"width:                       {width_val*1e-6:.4f} MHz",
            rf"$f_1$ / $f_2$ / $f_{{LO}}$:            {self.f1*1e3:.1f} mm / {self.f2*1e3:.1f} mm / {self.fLO*1e3:.2f} mm",
        ]
        if details is not None:
            lines.append(rf"Uniformity ($\sigma/\mu$):        {details['uniformity']*100:.3f} %")
            lines.append(rf"Crosstalk ($\eta$):              {details['eta']*100:.3f} %")
        else:
            lines.append("Uniformity / Crosstalk:      invalid point")

        if amps is not None:
            amps_arr = np.asarray(amps)
            amp_x = amps_arr[:self.N_x]
            amp_y = amps_arr[self.N_x:self.N_x + self.N_y]
            lines.append(rf"$a_x$:  {np.array2string(amp_x, precision=3)}")
            lines.append(rf"$a_y$:  {np.array2string(amp_y, precision=3)}")
        return lines

    def plot_scan2d_combined(self, show=True, save=True,
                              cmap_uniformity="viridis", cmap_crosstalk="Oranges",
                              vmax_uniformity=None, vmax_crosstalk=None):
        """
        Plots uniformity and crosstalk side by side (the "final" scan
        plot) - this is what scan_win_width_uniformity() saves by default.

        Each heatmap gets its OWN colorbar (own cmap, own vmin=0/vmax
        auto-scaled to its own data) - uniformity and crosstalk differ
        strongly in typical magnitude, so a single shared colorbar washes
        out the contrast of whichever one has the smaller range. A single
        shared legend (one entry, wrapped onto two lines to stay compact)
        sits inside the uniformity plot's corner instead of duplicating it
        per subplot.

        cmap_crosstalk defaults to 'Oranges' (light -> dark orange, no
        near-black tones and clearly distinct from the 'viridis' used for
        uniformity), since colormaps with a black low end (e.g. 'magma')
        read poorly for a quantity where black could be misread as "no
        data".

        Two independent figures are built as needed:
        - The SAVED figure (save=True) is clean: no info panel, no red
          selection frame - just the two heatmaps, their colorbars, and
          the shared legend stating the combined-optimal point's axis
          values (omega_in in mm, width in MHz and its equivalent spatial
          spot-separation length in µm via width_to_um()).
        - The SHOWN figure (show=True) additionally has the interactive
          info panel and click-to-highlight red frame (same behavior as
          plot_scan2d_uniformity()/plot_scan2d_crosstalk()): clicking a
          point in either heatmap highlights that cell in BOTH heatmaps
          and prints its full details (incl. waist before/after the
          lenses, amplitudes) into the panel below.

        The figure uses a serif font and higher DPI (see SCAN2D_RC /
        SCAN2D_SAVE_DPI) so the saved PNG is suitable for direct inclusion
        in a LaTeX document.
        """
        if 'scan2d' not in self.results:
            raise RuntimeError("scan_win_width_uniformity() must be called first.")

        if save:
            fig_save = self._build_scan2d_combined_figure(
                interactive=False, cmap_uniformity=cmap_uniformity, cmap_crosstalk=cmap_crosstalk,
                vmax_uniformity=vmax_uniformity, vmax_crosstalk=vmax_crosstalk,
            )
            tag = self._filetag()
            out_file = self.out_dir / f"FlatMultiTone_Scan2D_Combined_{tag}.png"
            fig_save.savefig(out_file, dpi=self.SCAN2D_SAVE_DPI, bbox_inches='tight')
            print(f"Figure saved: {out_file.name}")
            plt.close(fig_save)

        if show:
            self._build_scan2d_combined_figure(
                interactive=True, cmap_uniformity=cmap_uniformity, cmap_crosstalk=cmap_crosstalk,
                vmax_uniformity=vmax_uniformity, vmax_crosstalk=vmax_crosstalk,
            )
            plt.show()

    def _build_scan2d_combined_figure(self, interactive, cmap_uniformity, cmap_crosstalk,
                                       vmax_uniformity, vmax_crosstalk):
        """
        Builds (but does not show/save) the side-by-side uniformity +
        crosstalk figure used by plot_scan2d_combined(). With
        interactive=True, adds the click-to-highlight red frame and the
        info panel below both heatmaps; with interactive=False, produces
        the clean, LaTeX-ready layout without those elements. Each panel
        keeps its own colorbar; both share ONE legend (see
        plot_scan2d_combined()).
        """
        res = self.results['scan2d']
        win_input_vals = res['win_input_vals']
        width_vals = res['width_vals']
        U = res['uniformity_grid']
        C = res['crosstalk_grid']
        amps = res.get('amps')
        best = res.get('best', {})

        optimal_label = None
        if best.get("win_input") is not None:
            width_um = self.width_to_um(best["width"])
            # wrapped onto two lines so the legend stays compact
            optimal_label = (
                rf"Optimal: $\omega_{{\mathrm{{in}}}}$={best['win_input']*1e3:.3f} mm, "
                rf"width={best['width']*1e-6:.3f} MHz"
                "\n"
                rf"($\approx${width_um:.3f} µm profile width)"
            )

        with plt.rc_context(self.SCAN2D_RC):
            if interactive:
                fig = plt.figure(figsize=(15.5, 7.4))
                gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.32, wspace=0.28)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                info_ax = fig.add_subplot(gs[1, :])
                info_ax.axis('off')
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.4))

            panels = [
                (ax1, U, "Uniformity (σ/μ)", "Uniformity", cmap_uniformity, vmax_uniformity),
                (ax2, C, "Crosstalk (η)", "Crosstalk", cmap_crosstalk, vmax_crosstalk),
            ]

            dx = (win_input_vals[1] - win_input_vals[0]) * 1e3 if len(win_input_vals) > 1 else 0.1
            dy = (width_vals[1] - width_vals[0]) * 1e-6 if len(width_vals) > 1 else 0.05
            selection_rects = []

            for ax, Z, colorbar_label, title_metric, cmap, vmax in panels:
                im = ax.pcolormesh(
                    win_input_vals * 1e3, width_vals * 1e-6, Z,
                    shading='auto', cmap=cmap, vmax=vmax,
                )
                x_pad = dx / 2
                y_pad = dy / 2
                ax.set_xlim(win_input_vals[0] * 1e3 - x_pad, win_input_vals[-1] * 1e3 + x_pad)
                ax.set_ylim(width_vals[0] * 1e-6 - y_pad, width_vals[-1] * 1e-6 + y_pad)

                fig.colorbar(im, ax=ax, label=colorbar_label)

                if optimal_label is not None:
                    ax.plot(
                        best["win_input"] * 1e3, best["width"] * 1e-6,
                        marker="+", color="red", markeredgecolor="red",
                        markersize=18, markeredgewidth=3, linestyle="none", zorder=7,
                        label=(optimal_label if ax is ax1 else None),
                    )

                ax.set_xlabel(r"Input waist $\omega_{\mathrm{in}}$ (mm)")
                ax.set_ylabel("width (MHz)")
                ax.set_title(title_metric)

                if interactive:
                    rect = Rectangle((0, 0), dx, dy, edgecolor="red", facecolor="none",
                                      linewidth=2.5, zorder=6)
                    rect.set_visible(False)
                    ax.add_patch(rect)
                    selection_rects.append(rect)

            # ONE legend (single entry, wrapped onto two lines), placed
            # inside ax1's own corner rather than duplicated per subplot.
            if optimal_label is not None:
                ax1.legend(loc="upper right", framealpha=0.9, fontsize=12 if interactive else None)

            if not interactive:
                fig.tight_layout()
                return fig

            fig.subplots_adjust(left=0.055, right=0.95, top=0.93, bottom=0.05)

            # Interactive-only: info panel (pre-filled with the optimal
            # point) + click handler highlighting the same cell on both
            # heatmaps and printing its full details.
            if best.get("win_input") is not None:
                win_eff_best = self.win_input_to_win(best["win_input"])
                details_best = self._evaluate(win_eff_best, best["width"], amps=amps)
                initial_lines = ["Optimal (combined), see cross:"]
                initial_lines += self._scan2d_info_lines(best["win_input"], best["width"], win_eff_best, details_best, amps)
                initial_lines.append("")
                initial_lines.append("Click any point in either heatmap to inspect it instead.")
                initial_msg = "\n".join(initial_lines)
            else:
                initial_msg = "Click on a point in either heatmap to show all parameters (incl. waist before/after the lenses)."

            info_text = info_ax.text(
                0.0, 1.0, initial_msg, va='top', ha='left', fontsize=14,
                family='monospace', transform=info_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9),
            )

            def _on_click(event):
                if event.inaxes not in (ax1, ax2) or event.xdata is None or event.ydata is None:
                    return

                win_input_val = event.xdata * 1e-3   # mm -> m
                width_val = event.ydata * 1e6        # MHz -> Hz

                j = int(np.argmin(np.abs(win_input_vals - win_input_val)))
                i = int(np.argmin(np.abs(width_vals - width_val)))
                win_input_actual = win_input_vals[j]
                width_actual = width_vals[i]

                xy = (win_input_actual * 1e3 - dx / 2, width_actual * 1e-6 - dy / 2)
                for rect in selection_rects:
                    rect.set_xy(xy)
                    rect.set_visible(True)

                try:
                    win_eff = self.win_input_to_win(win_input_actual)
                except ValueError:
                    win_eff = None

                details = self._evaluate(win_eff, width_actual, amps=amps) if win_eff is not None else None

                lines = self._scan2d_info_lines(win_input_actual, width_actual, win_eff, details, amps)
                info_text.set_text("\n".join(lines))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', _on_click)
            return fig

    def _plot_scan2d_metric(self, grid_key, colorbar_label, title_metric, filename_suffix,
                             cmap, vmax, show, save):
        """
        Shared plotting routine for plot_scan2d_uniformity() /
        plot_scan2d_crosstalk() - both are standalone 2D heatmaps over the
        same (win_input, width) grid, differing only in which precomputed
        grid is displayed. For the combined side-by-side figure (the
        default "final" plot), see plot_scan2d_combined() instead.

        x-axis: input waist before the 1st lens, omega_in, in mm.
        y-axis: width in MHz.

        A red cross marks the single combined-optimal point (from the
        alpha-weighted uniformity+crosstalk score computed in
        scan_win_width_uniformity()) - the same point on both the
        uniformity and the crosstalk plot, analogous to the "good starting
        parameters" found by optimize_win()/optimize_width()/optimize_combined().
        Its exact values are shown in the info panel below the plot
        (legend only carries a short label, to avoid clutter).

        Interactive: clicking a point on the heatmap outlines the selected
        grid cell with a red frame and shows, in the info panel, all
        parameters for that point - including the waist before the lens
        (omega_in, mm) AND the resulting waist at the focal plane after the
        lenses (omega', µm), width, uniformity, crosstalk, and the
        amplitudes used (if any). All parameter names are rendered with
        LaTeX-style math symbols/subscripts instead of underscored names.

        The figure uses a serif font and higher DPI (see SCAN2D_RC /
        SCAN2D_SAVE_DPI) so the saved PNG is suitable for direct inclusion
        in a LaTeX document.
        """
        if 'scan2d' not in self.results:
            raise RuntimeError("scan_win_width_uniformity() must be called first.")
        res = self.results['scan2d']
        win_input_vals = res['win_input_vals']
        width_vals = res['width_vals']
        Z = res[grid_key]
        amps = res.get('amps')
        best = res.get('best', {})

        with plt.rc_context(self.SCAN2D_RC):
            fig = plt.figure(figsize=(9, 8.9))
            ax = fig.add_axes([0.12, 0.36, 0.75, 0.56])
            info_ax = fig.add_axes([0.06, 0.03, 0.88, 0.24])
            info_ax.axis('off')

            im = ax.pcolormesh(
                win_input_vals * 1e3, width_vals * 1e-6, Z,
                shading='auto', cmap=cmap, vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label=colorbar_label)

            # Cross marking the combined-optimal point (good starting
            # parameters, same weighting as optimize_win/width/combined),
            # shown identically on both the uniformity and crosstalk plot.
            # Legend stays short on purpose - exact values are shown in the
            # info panel below instead of being crammed into the legend.
            if best.get("win_input") is not None:
                ax.plot(
                    best["win_input"] * 1e3, best["width"] * 1e-6,
                    marker="+", color="red", markeredgecolor="red",
                    markersize=20, markeredgewidth=3, linestyle="none", zorder=7,
                    label="Optimal (combined)",
                )
                ax.legend(loc="upper right", framealpha=0.9)

            ax.set_xlabel(r"Input waist $\omega_{\mathrm{in}}$ (mm)")
            ax.set_ylabel("width (MHz)")
            ax.set_title(rf"{title_metric} in spot square: $N_x$={self.N_x}, $N_y$={self.N_y}")

            # Red frame highlighting the currently selected (clicked) grid cell.
            # Cell size is derived from the (uniform) grid spacing so the frame
            # exactly outlines one pixel of the heatmap.
            dx = (win_input_vals[1] - win_input_vals[0]) * 1e3 if len(win_input_vals) > 1 else 0.1
            dy = (width_vals[1] - width_vals[0]) * 1e-6 if len(width_vals) > 1 else 0.05
            selection_rect = Rectangle(
                (0, 0), dx, dy, edgecolor="red", facecolor="none",
                linewidth=2.5, zorder=6,
            )
            selection_rect.set_visible(False)
            ax.add_patch(selection_rect)

            # Info panel starts out showing the combined-optimal point
            # (the same one marked with the cross), so its exact values are
            # visible immediately without requiring a click.
            if best.get("win_input") is not None:
                win_eff_best = self.win_input_to_win(best["win_input"])
                details_best = self._evaluate(win_eff_best, best["width"], amps=amps)
                initial_lines = ["Optimal (combined), see cross:"]
                initial_lines += self._scan2d_info_lines(best["win_input"], best["width"], win_eff_best, details_best, amps)
                initial_lines.append("")
                initial_lines.append("Click any point in the heatmap to inspect it instead.")
                initial_msg = "\n".join(initial_lines)
            else:
                initial_msg = "Click on a point in the heatmap to show all parameters (incl. waist before/after the lenses)."

            info_text = info_ax.text(
                0.0, 1.0, initial_msg, va='top', ha='left', fontsize=14,
                family='monospace', transform=info_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9),
            )

            def _on_click(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return

                win_input_val = event.xdata * 1e-3   # mm -> m
                width_val = event.ydata * 1e6        # MHz -> Hz

                # find the nearest actually computed grid point
                j = int(np.argmin(np.abs(win_input_vals - win_input_val)))
                i = int(np.argmin(np.abs(width_vals - width_val)))
                win_input_actual = win_input_vals[j]
                width_actual = width_vals[i]

                # highlight the selected cell with a red frame
                selection_rect.set_xy((win_input_actual * 1e3 - dx / 2, width_actual * 1e-6 - dy / 2))
                selection_rect.set_visible(True)

                try:
                    win_eff = self.win_input_to_win(win_input_actual)
                except ValueError:
                    win_eff = None

                details = self._evaluate(win_eff, width_actual, amps=amps) if win_eff is not None else None

                lines = self._scan2d_info_lines(win_input_actual, width_actual, win_eff, details, amps)
                info_text.set_text("\n".join(lines))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', _on_click)

            tag = self._filetag()
            self._finish_figure(
                fig, f"FlatMultiTone_Scan2D_{filename_suffix}_{tag}.png",
                show, save, dpi=self.SCAN2D_SAVE_DPI,
            )

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
# Beispiel-Nutzung (nur wenn das Skript direkt ausgeführt wird)
# ======================================================================
if __name__ == "__main__":
    #out_dir = r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\PythonCode\Multitone_FlatTop"
    out_dir = r"C:\Users\Admin\Documents\GitHub\Lern - repo\Strahlanalyse\AOD_Simulation\Results"

    opt = MultitoneFlatTopOptimizer(
        out_dir=out_dir,
        offset=100e6,
        width=0.3e6,
        N_x=3,
        N_y=4,
        win=1.0e-6,
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
    #
    # amp_x/amp_y müssen symmetrisch sein (innen = 1, außen paarweise gleich) -
    # optimize_combined() optimiert intern ohnehin nur noch das jeweils EINE
    # freie Verhältnis außen/innen pro Achse (siehe build_symmetric_amps()):
    # N_x=3 (außen-innen-außen) -> 1 freier Parameter, N_y=4 (außen-innen-
    # innen-außen) -> ebenfalls 1 freier Parameter.
    opt.set_initial_guess(
        win=1e-6,
        width=3e5,
        amp_x=[1.05, 1.0, 1.05],
        amp_y=[1.1, 1.0, 1.0, 1.1],
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

    # Beispiel: 2D-Scan der Uniformity über Eingangswaist vor der 1. Linse (win_input)
    # und width, bei fester Tonanzahl und festen Amplituden:
    #
    # opt.scan_win_width_uniformity(
    #     win_input_range=(0.05e-6, 2.0e-6),
    #     width_range=(0.1e6, 0.6e6),
    #     n_win_input=50, n_width=50,
    #     amps=np.concatenate([np.ones(opt.N_x), np.ones(opt.N_y)]),
    # )