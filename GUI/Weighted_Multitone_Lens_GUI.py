"""
Weighted FlatMultiTone GUI (PyQt5-Version)
============================================
Erweiterung von Multitone_Lens_GUI.py um ATOM-GEWICHTETE Uniformity-/
Crosstalk-Metriken: statt zweier hart-begrenzter, per Drag&Drop verschieb-
barer Boxen (Uniformity-Region / Crosstalk-Region) wird optional die
tatsächliche thermische Ortsverteilung des Atoms in der Dipolfalle
(Gauß-Gewichtung mit sigma_atom aus Fallenfrequenz, Temperatur und
Atommasse) als kontinuierliche Gewichtsfunktion verwendet - siehe
Diskussion/Herleitung im MultitoneFlatTopOptimizer (multitone_flattop_
optimizer.py), insbesondere den dortigen Bugfix zur korrekten
Nachbar-Normierung auf kleinen lokalen Fenstern (create_neighbourhood()
normiert pro Fenster auf dessen eigenes Maximum - auf einem winzigen,
lokal um die Site zentrierten Fenster ist das falsch, weil der echte
Peak jeder Nachbarkopie ~pitch entfernt und damit außerhalb des Fensters
liegt; siehe local_neighbor_intensity() weiter unten für die korrigierte
Variante).

NEU gegenüber Multitone_Lens_GUI.py:
- Checkbox "Use atom-weighted uniformity/crosstalk": schaltet zwischen der
  bisherigen harten Box-Definition und der atom-gewichteten Definition um.
- Neue Gruppe "Atom (weighted metrics)": Spezies (Rb-85/Rb-87), Temperatur
  und Fallenfrequenz sind über Slider einstellbar (Startwerte aus der
  Messung: Rb-85, T = 17 µK, nu_r = 60.4 kHz) und fließen live in
  sigma_atom = sigma_thermal(m, omega_r, T) ein.
- Im gewichteten Modus ersetzen sigma_atom-Konturen (1/2/3 sigma) im
  Hauptplot sowie eine gefüllte Wahrscheinlichkeitsdichte-Kurve in beiden
  Schnittplots die bisherigen festen Rechteck-Regionen. Der bisherige
  "Neighbor regions"-Plot wird durch eine lokale, auf die Atom-Skala
  (sigma_atom) intelligent neu skalierte Nahansicht ersetzt - andernfalls
  wäre auf dem globalen, µm-skalierten Gitter (Auflösung GRID_N über den
  ganzen Pitch-Bereich) die für sigma_atom ~ 100 nm relevante Struktur
  hoffnungslos unteraufgelöst bzw. unsichtbar.
- Beide Schnittplots werden im gewichteten Modus mit eigens dafür fein
  aufgelösten, lokal um das Zentrum liegenden Datenreihen (statt des
  groben globalen Gitters) neu gezeichnet und automatisch auf ein Fenster
  von einigen sigma_atom gezoomt.

Voraussetzung: PyQt5 muss installiert sein.
    pip install PyQt5

Start:
    python Weighted_Multitone_Lens_GUI.py
"""

import sys
import datetime
from pathlib import Path as FilePath

import numpy as np
from scipy.special import j1
from scipy.constants import hbar, k as kB, atomic_mass

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle, Circle

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QGroupBox, QScrollArea, QSplitter, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt

from lens_design_dialog import LensDesignDialog
import airy_scale

# ----------------------------------------------------------------------
# Airy-Skalenfaktor: first_zero_radius = AIRY_SCALE_FACTOR * waist
# ----------------------------------------------------------------------
# Legt fest, was die Zahl "waist" beim Airy-Profil physikalisch bedeutet.
# Voreingestellt ist 1.4830 - damit hat die Airy-Hauptkeule denselben
# 1/e^2-Radius wie ein Gauss-Strahl mit diesem Waist. Der historische Wert
# 1.19 (kein 1/e^2-Radius, sondern 0.8025 * waist) bleibt im Dialog
# waehlbar. Die Definitionen stehen in airy_scale.py, einer identischen
# Kopie der Datei aus Amplitudes/Weighted_Optimization/.
AIRY_SCALE_FACTOR = airy_scale.AIRY_SCALE_DIALOG_DEFAULT



# ============================================================
# Physikalische Konstanten (fix, nicht über die GUI einstellbar)
# f1 und f2 sind jetzt über die GUI einstellbar (siehe self.state) und
# daher hier keine festen Konstanten mehr.
# ============================================================
offset = 100e6        # Hz
fLO = 52.88e-3         # m
theta_max = 43e-3      # rad
f_band = 36e6          # Hz
lambda_opt = 795e-9    # m
pitch = 5.288e-6       # m, physikalischer Atomabstand (Verschiebung der Nachbar-Kopien)

GRID_N = 260            # Auflösung für die interaktive Vorschau (hartes Grid)
GRID_N_HIGHRES = 1000   # Auflösung für den "Speichern"-Button (hartes Grid)

# Auflösung/Ausdehnung für die ATOM-GEWICHTETE lokale Auswertung. Diese ist
# unabhängig vom (groben, µm-skalierten) globalen Grid, weil sigma_atom
# typischerweise ~100 nm beträgt - ein Vielfaches kleiner als GRID_N/den
# Pitch-Bereich. Ohne eigenes, fein aufgelöstes lokales Fenster wäre die
# gewichtete Auswertung numerisch unterabgetastet (siehe Modul-Docstring).
WEIGHTED_N_SIGMA = 6          # Ausdehnung des lokalen Auswertungs-/Sichtfensters in sigma_atom
WEIGHTED_GRID_N = 161          # Auflösung des lokalen 2D-Fensters (interaktiv)
WEIGHTED_GRID_N_HIGHRES = 301  # Auflösung des lokalen 2D-Fensters (Export)
WEIGHTED_CUT_POINTS = 400      # Punkte pro Schnittlinie im gewichteten Modus (interaktiv)
WEIGHTED_CUT_POINTS_HIGHRES = 800

# Schriftgrößen für den High-Resolution-Export (PNG, das z.B. in LaTeX/PDF
# eingebunden wird -> muss auch nach dem Verkleinern im Dokument lesbar sein).
EXPORT_FONTSIZE_SUPTITLE = 15
EXPORT_FONTSIZE_TITLE = 13
EXPORT_FONTSIZE_LABEL = 13
EXPORT_FONTSIZE_TICK = 11
EXPORT_FONTSIZE_LEGEND = 11

out_dir = FilePath(r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\PythonCode\Multitone_FlatTop")
try:
    out_dir.mkdir(parents=True, exist_ok=True)
except Exception:
    out_dir = FilePath.cwd() / "FlatMultiTone_Output"
    out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Hilfsfunktionen (Physik/Numerik) — unverändert zur bisherigen Version
# ============================================================
def multitone_frequencies(N, offset, width):
    """Erzeugt diskrete Frequenzen wie im AWG: width * n/(N-1) + offset.

    Für N=1 (ein einzelner Ton) wird die Frequenz auf die MITTE des
    Scan-Bereichs gelegt (offset + width/2) statt auf offset selbst.
    Andernfalls wäre der einzelne Ton nicht deckungsgleich mit r_center
    (das über f_center = offset + width/2 definiert ist, siehe
    compute_centers()) - genau das führte zu einem sichtbar nicht
    zentrierten Atom/Spot, sobald N_x oder N_y auf 1 gesetzt wurde,
    während es für N>=2 unauffällig blieb (dort ist f_center automatisch
    die Mitte des linear aufgespannten Frequenzbereichs)."""
    if N <= 1:
        return np.array([offset + width / 2], dtype=float)
    return width * np.arange(N) / (N - 1) + offset


def angle_from_frequency(f, offset, theta_max, f_band):
    return theta_max * (f - offset) / f_band


def radius_from_angle(theta, f1, f2, fLO):
    return (f1 * fLO / f2) * np.tan(theta)


def gaussian_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, sigma, amps):
    I = np.zeros_like(X, dtype=float)
    for cx, cy, a in zip(centers_x, centers_y, amps):
        I += a * np.exp(-2 * ((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2))
    return I


def airy_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, first_zero_radius, amps):
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


def compute_intensity_profile(X, Y, centers_x, centers_y, width_param, amps, use_airy,
                              airy_scale_factor=None):
    if use_airy:
        factor = AIRY_SCALE_FACTOR if airy_scale_factor is None else float(airy_scale_factor)
        first_zero_radius = factor * width_param
        return airy_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, first_zero_radius, amps)
    else:
        return gaussian_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, width_param, amps)


def create_neighbourhood(X, Y, pitch, centers_x, centers_y, w_in, amps=None, use_airy=False):
    """
    Unveraendert zur Originaldatei: Summiert die (pro Nachbar-Kopie auf 1
    normierten) Intensitaetsbeitraege der 8 Nachbar-Sites. NUR fuer das
    GLOBALE, hinreichend breite Grid korrekt (siehe Modul-Docstring) - wird
    daher ausschliesslich fuer die harte Box-Metrik und den bisherigen
    "Neighbor regions"-Plot verwendet, NICHT fuer die atom-gewichtete
    Auswertung (dafuer: local_neighbor_intensity()).
    """
    if amps is None:
        amps = np.ones(len(centers_x))
    I_neighbor = np.zeros_like(X)
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            if ix == 0 and iy == 0:
                continue
            shifted_x = centers_x + ix * pitch
            shifted_y = centers_y + iy * pitch
            I_spot = compute_intensity_profile(X, Y, shifted_x, shifted_y, w_in, amps, use_airy)
            I_spot = I_spot / np.max(I_spot)
            I_neighbor += I_spot
    return I_neighbor


def overlap_mask_pitch(X, Y, center_x, center_y, side_length):
    half_side = side_length / 2
    return (np.abs(X - center_x) <= half_side) & (np.abs(Y - center_y) <= half_side)


def compute_centers(N_x, N_y, width, f1, f2):
    fx_freq = multitone_frequencies(N_x, offset, width)
    fy_freq = multitone_frequencies(N_y, offset, width)
    f_center = offset + width / 2
    theta_center = angle_from_frequency(f_center, offset, theta_max, f_band)
    r_center = radius_from_angle(theta_center, f1, f2, fLO)

    centers_x = []
    centers_y = []
    for fx in fx_freq:
        theta_x = angle_from_frequency(fx, offset, theta_max, f_band)
        r_x = radius_from_angle(theta_x, f1, f2, fLO)
        for fy in fy_freq:
            theta_y = angle_from_frequency(fy, offset, theta_max, f_band)
            r_y = radius_from_angle(theta_y, f1, f2, fLO)
            centers_x.append(r_x)
            centers_y.append(r_y)
    return np.array(centers_x), np.array(centers_y), r_center


def conjugate_waist(w, f1, f2):
    """
    Rechnet zwischen dem Eingangswaist (vor f1) und dem resultierenden Waist
    nach dem Teleskop f1->f2 (vor fLO-Fokussierung) um. Die Beziehung ist
    symmetrisch (Fourier-Transformationseigenschaft eines Gauß-Strahls durch
    eine Linse): dieselbe Funktion liefert in beide Richtungen den jeweils
    anderen Waist.
        w_out = (f1/f2) * (lambda_opt * fLO) / (pi * w_in)
    """
    C = (f1 / f2) * (lambda_opt * fLO) / np.pi
    return C / w


LENS_FOCAL_LENGTHS_MM = [30, 35, 40, 45, 50, 60, 75, 80.3, 100, 125, 150,
                          200, 250, 300, 400, 500, 750, 1000]


def closest_focal_index(f_m, options_mm=LENS_FOCAL_LENGTHS_MM):
    """Index des Eintrags in options_mm, der f_m (in Metern) am nächsten kommt."""
    f_mm = f_m * 1e3
    diffs = [abs(f_mm - v) for v in options_mm]
    return diffs.index(min(diffs))


def compute_grid(centers_x, centers_y, win, uniformity_side_length, crosstalk_side_length, resolution):
    # Die Marge muss auch den Pitch-Versatz der Nachbar-Kopien abdecken:
    # create_neighbourhood() verschiebt Kopien der Spots um +-pitch in x und y.
    # War die Marge kleiner als das, fielen Nachbarn (v.a. bei kleinem N_x/N_y)
    # bereits außerhalb des berechneten Grids und wurden dadurch schon in den
    # Rohdaten abgeschnitten - nicht nur in der Ansicht.
    margin = max(10 * win, 1.3 * uniformity_side_length, 1.3 * crosstalk_side_length, 3.0 * pitch)
    x = np.linspace(-margin / 2, np.max(np.abs(centers_x)) + margin / 2, resolution)
    y = np.linspace(-margin / 2, np.max(np.abs(centers_y)) + margin / 2, resolution)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def width_to_um(width_hz, f1, f2):
    """Rechnet die (Frequenz-)Breite `width` (Slider in MHz, state['width'] in Hz)
    über dieselbe Winkel/Radius-Beziehung wie der Rest der Optik in eine
    räumliche Größe in µm um - exakt analog zur Scanning-Range-Berechnung,
    die theta_max verwendet (radius_from_angle(theta_max, f1, f2, fLO)),
    hier eben mit dem durch `width` erzeugten Winkel."""
    theta_width = theta_max * width_hz / f_band
    return radius_from_angle(theta_width, f1, f2, fLO) * 1e6


def configure_neighbor_view(ax, r_center_um, pitch_um, zoom_factor=1.3):
    """Setzt die Achsengrenzen des Nachbar-Subplots auf ein festes, quadratisches
    Fenster um das Zentrum, das groß genug ist, um alle acht pitch-verschobenen
    Nachbar-Kopien vollständig zu zeigen - unabhängig von der (meist deutlich
    größeren) Ausdehnung des Hauptplots, dessen extent hier sonst 1:1 übernommen
    würde und die Nachbarn winzig bzw. am Rand abgeschnitten aussehen ließe.
    (Nur im HARTEN Modus verwendet - im gewichteten Modus übernimmt
    configure_local_view() diese Rolle mit einem an sigma_atom skalierten
    Fenster.)"""
    half = pitch_um * zoom_factor
    ax.set_xlim(r_center_um - half, r_center_um + half)
    ax.set_ylim(r_center_um - half, r_center_um + half)
    ax.set_aspect("equal", adjustable="box")


# ============================================================
# NEU: Atom-Gewichtung (thermische Ortsverteilung im harmonischen
# Fallenpotential) - siehe MultitoneFlatTopOptimizer.sigma_thermal() /
# atom_weight_2d() / weighted_uniformity() / weighted_crosstalk() für die
# ausführliche physikalische Herleitung und Diskussion.
# ============================================================
RB_MASSES_KG = {
    "Rb-85": 84.911789738 * atomic_mass,
    "Rb-87": 86.909180527 * atomic_mass,
}


def sigma_thermal(m, omega, T):
    """
    Thermische 1-sigma-Breite der 2D-Ortsverteilung eines Atoms der Masse m
    im isotropen harmonischen Fallenpotential (Frequenz omega = 2*pi*nu_r)
    bei Temperatur T, inklusive Nullpunkts-Korrektur (coth-Term):

        sigma^2 = hbar/(2*m*omega) * coth(hbar*omega / (2*kB*T))
    """
    if omega <= 0 or T <= 0:
        return float("nan")
    x = hbar * omega / (2 * kB * T)
    return np.sqrt(hbar / (2 * m * omega) / np.tanh(x))


def atom_weight_2d(X, Y, center_x, center_y, sigma):
    """Unnormierte gaussfoermige Gewichtsfunktion W(x,y) der thermischen
    Aufenthaltswahrscheinlichkeit eines Atoms um eine Site."""
    return np.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * sigma ** 2))


def weighted_uniformity(I, W):
    """Gewichteter Variationskoeffizient sigma_w/mu_w (Analogon zu std/mean
    der harten Masken-Definition), siehe Modul-Docstring."""
    norm = np.sum(W)
    if norm == 0:
        return float("nan")
    mu_w = np.sum(I * W) / norm
    if mu_w == 0:
        return float("nan")
    var_w = np.sum((I - mu_w) ** 2 * W) / norm
    return np.sqrt(var_w) / mu_w


def weighted_crosstalk(I_own, I_neighbor, W):
    """Gewichtetes Verhaeltnis Nachbar-/Eigenintensitaet, siehe Modul-Docstring."""
    denom = np.sum(I_own * W)
    if denom == 0:
        return float("nan")
    return np.sum(I_neighbor * W) / denom


def local_neighbor_intensity(X, Y, pitch, centers_x, centers_y, width_param, amps, use_airy):
    """
    Wie create_neighbourhood(), aber OHNE dessen interne Renormierung
    'I_spot /= np.max(I_spot)' pro Nachbar-Kopie.

    Diese Renormierung ist NUR korrekt, wenn das übergebene Grid breit
    genug ist, um den tatsächlichen Peak JEDES verschobenen Nachbarn zu
    erfassen (der per Definition ~pitch von der betrachteten Site entfernt
    liegt). Auf dem winzigen lokalen Fenster für die atom-gewichtete
    Auswertung (~WEIGHTED_N_SIGMA * sigma_atom breit) liegt der
    Nachbar-Peak dagegen weit AUSSERHALB des Fensters - np.max() sieht dort
    nur einen kleinen Schwanzwert, normiert fälschlich darauf und bläst die
    Intensität massiv auf (siehe MultitoneFlatTopOptimizer-Bugfix).

    Da das Profil translationsinvariant ist (die Summe hängt nur von
    Abständen zu den - verschobenen - Zentren ab), hat jede Nachbarkopie
    exakt dieselbe Peak-Intensität wie die eigene Site; die Division
    entfällt daher ersatzlos.
    """
    I_neighbor = np.zeros_like(X, dtype=float)
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            if ix == 0 and iy == 0:
                continue
            shifted_x = centers_x + ix * pitch
            shifted_y = centers_y + iy * pitch
            I_neighbor += compute_intensity_profile(X, Y, shifted_x, shifted_y, width_param, amps, use_airy)
    return I_neighbor


def build_local_weighted_grid(center_x, center_y, sigma, n_sigma, n_grid):
    """Feines, lokal um (center_x, center_y) zentriertes 2D-Sub-Grid mit
    Ausdehnung +/- n_sigma * sigma - siehe Modul-Docstring."""
    half_extent = n_sigma * sigma
    xs = np.linspace(center_x - half_extent, center_x + half_extent, n_grid)
    ys = np.linspace(center_y - half_extent, center_y + half_extent, n_grid)
    Xs, Ys = np.meshgrid(xs, ys)
    return xs, ys, Xs, Ys


def build_local_cut_lines(center_x, center_y, sigma, n_sigma, n_points):
    """Feine 1D-Linien (x- und y-Schnitt) durch (center_x, center_y), fürs
    intelligente Neu-Skalieren der Schnittplots im gewichteten Modus."""
    half_extent = n_sigma * sigma
    x_line = np.linspace(center_x - half_extent, center_x + half_extent, n_points)
    y_line = np.linspace(center_y - half_extent, center_y + half_extent, n_points)
    return x_line, y_line


# ============================================================
# Hauptfenster
# ============================================================
class WeightedFlatMultiToneWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Weighted FlatMultiTone GUI (PyQt5)")
        self.resize(1500, 900)

        self.state = {
            "N_x": 3,
            "N_y": 4,
            "win": 1.05e-6,        # Waist NACH den Linsen (das, was tatsächlich für die Rechnung genutzt wird)
            "win_in": None,       # Waist VOR den Linsen (nur informativ bzw. im Eingangswaist-Modus die Stellgröße)
            "win_mode": "output", # 'output' = win direkt einstellen (bisheriges Verhalten)
                                  # 'input'  = win_in einstellen, win wird daraus berechnet
            "f1": 75e-3,
            "f2": 750e-3,
            "width": 0.35e6,
            "uniformity_side_length": 2.6e-6,
            "crosstalk_side_length": pitch,
            "custom_amps": False,
            "use_airy": False,
            "amp_x": np.ones(3),
            "amp_y": np.ones(4),
            "cut_row_idx": None,   # Fadenkreuz: Zeilenindex (bestimmt Schnitt entlang x, feste y-Position)
            "cut_col_idx": None,   # Fadenkreuz: Spaltenindex (bestimmt Schnitt entlang y, feste x-Position)
            # Vom Lens Design Tool (3-Linsen-Modell) übernommene, genauere Werte.
            # None = noch nicht/nicht mehr gültig -> interne Formel wird verwendet.
            "external_scan_range_um": None,
            "external_central_profile_um": None,

            # --- NEU: atom-gewichtete Uniformity/Crosstalk ---
            "weighted_mode": False,
            "atom_species": "Rb-85",
            "atom_temperature": 17e-6,   # K  (Startwert aus der Messung)
            "atom_trap_freq": 60.4e3,    # Hz (Startwert aus der Messung, nu_r)
            "sigma_atom": None,           # wird laufend nachgerechnet (Cache)

            # --- NEU: Atom-Position innerhalb des Lichtmusters ---
            # Verschiebung des Atoms relativ zum Zentrum des Lichtflecks
            # (r_center), unabhängig einstellbar in x und y, begrenzt auf
            # +/- pitch/2 (weiter kann sich das Atom nicht von "seinem" Site
            # entfernen, ohne dass ein Nachbar-Site naeher waere). Fliesst in
            # ALLE Uniformity-/Crosstalk-Metriken (hart wie gewichtet) sowie
            # in die Fadenkreuz-Standardposition ein - siehe _atom_center().
            "atom_offset_x": 0.0,   # m
            "atom_offset_y": 0.0,   # m

            # --- NEU: manueller Update-Modus ---
            # Ist "manual_update_mode" True, loesen Slider/Checkboxen/Drag&Drop
            # etc. KEIN sofortiges Neuzeichnen mehr aus - stattdessen wird nur
            # der interne state aktualisiert und "_pending_manual_update"
            # gesetzt; erst ein Klick auf den "Update now"-Button (siehe
            # on_manual_update_clicked()) fuehrt dann ein vollstaendiges
            # full_update(force=True) aus.
            "manual_update_mode": False,
        }
        self.state["win_in"] = conjugate_waist(self.state["win"], self.state["f1"], self.state["f2"])
        self.state["sigma_atom"] = self._current_sigma_atom()
        self.lens_dialog = None
        self.cache = {}
        self.dragging_target = None
        self._pending_manual_update = False

        # Widgets, die dynamisch neu erzeugt werden (Amplituden-Panel)
        self.amp_spinboxes_x = []
        self.amp_spinboxes_y = []

        # Artefakt-Referenzen für den gewichteten Modus (werden bei Bedarf
        # angelegt/entfernt, siehe _redraw_weighted_overlays()/_redraw_hard_rectangles())
        self.sigma_circles = []
        self.pdf_fill_x = None
        self.pdf_fill_y = None

        self._build_ui()
        self._connect_signals()
        self.full_update()

    # --------------------------------------------------------
    # UI-Aufbau
    # --------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ---------------- linke Seite: Matplotlib-Canvas ----------------
        self.fig = Figure(figsize=(9, 8))
        self.fig.set_constrained_layout(True)  # verhindert automatisch überlappende Titel/Labels
        self.canvas = FigureCanvas(self.fig)

        # height_ratios: die Nachbar-/Lokalansicht (Zeile 0) bleibt groß wie ein
        # "richtiger" 2D-Plot, die beiden Schnittplots (Zeile 1/2) dürfen kleiner
        # sein. Die Gesamthöhe (und damit die Höhe des Hauptplots in Spalte 0,
        # der über alle drei Zeilen geht) ändert sich dadurch nicht.
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.6, 1], height_ratios=[1.8, 1, 1])
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_neighbor = self.fig.add_subplot(gs[0, 1])
        self.ax_cut_x = self.fig.add_subplot(gs[1, 1])
        self.ax_cut_y = self.fig.add_subplot(gs[2, 1])

        splitter.addWidget(self.canvas)

        # ---------------- rechte Seite: Steuerbereich ----------------
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setAlignment(Qt.AlignTop)

        control_layout.addWidget(self._build_update_mode_group())
        control_layout.addWidget(self._build_grid_group())
        control_layout.addWidget(self._build_lens_group())
        control_layout.addWidget(self._build_param_group())
        control_layout.addWidget(self._build_profile_group())
        control_layout.addWidget(self._build_amp_group())
        control_layout.addWidget(self._build_atom_group())
        control_layout.addWidget(self._build_atom_position_group())
        control_layout.addWidget(self._build_region_group())
        control_layout.addWidget(self._build_crosshair_group())
        control_layout.addWidget(self._build_save_group())
        control_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(control_container)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(430)

        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)

        # Drag & Drop im Hauptplot
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("button_release_event", self.on_release)

    def _build_update_mode_group(self):
        """NEU: Umschalter zwischen dem bisherigen Live-Update (GUI zeichnet
        bei jeder Änderung sofort neu) und einem manuellen Modus, bei dem
        Änderungen nur den internen state aktualisieren und die eigentliche
        Neuberechnung/Neuzeichnung erst per Klick auf "Update now" passiert."""
        box = QGroupBox("Update Mode")
        layout = QVBoxLayout(box)

        self.cb_manual_update = QCheckBox("Manual update (apply changes only via button)")
        self.cb_manual_update.setChecked(self.state["manual_update_mode"])
        layout.addWidget(self.cb_manual_update)

        self.btn_manual_update = QPushButton("Update now")
        self.btn_manual_update.setEnabled(self.state["manual_update_mode"])
        layout.addWidget(self.btn_manual_update)

        self.label_manual_update_status = QLabel("")
        self.label_manual_update_status.setWordWrap(True)
        layout.addWidget(self.label_manual_update_status)

        return box

    def _build_grid_group(self):
        box = QGroupBox("Spot Grid")
        layout = QGridLayout(box)

        layout.addWidget(QLabel("N_x:"), 0, 0)
        self.spin_nx = QSpinBox()
        self.spin_nx.setRange(1, 30)
        self.spin_nx.setValue(self.state["N_x"])
        layout.addWidget(self.spin_nx, 0, 1)

        layout.addWidget(QLabel("N_y:"), 1, 0)
        self.spin_ny = QSpinBox()
        self.spin_ny.setRange(1, 30)
        self.spin_ny.setValue(self.state["N_y"])
        layout.addWidget(self.spin_ny, 1, 1)

        return box

    def _build_param_group(self):
        box = QGroupBox("Beam Parameters")
        layout = QGridLayout(box)
        row = 0

        self.cb_win_mode = QCheckBox("Set input waist (before the lenses)")
        self.cb_win_mode.setChecked(self.state["win_mode"] == "input")
        layout.addWidget(self.cb_win_mode, row, 0, 1, 2)
        row += 1

        self.label_win_in = QLabel()
        layout.addWidget(self.label_win_in, row, 0, 1, 2)
        row += 1
        self.slider_win_in = QSlider(Qt.Horizontal)
        self.slider_win_in.setRange(5, 500)   # entspricht 0.05 - 5.00 mm (Faktor 100)
        self.slider_win_in.setValue(int(round(self.state["win_in"] * 1e3 * 100)))
        layout.addWidget(self.slider_win_in, row, 0, 1, 2)
        row += 1

        self.label_win = QLabel()
        layout.addWidget(self.label_win, row, 0, 1, 2)
        row += 1
        self.slider_win = QSlider(Qt.Horizontal)
        self.slider_win.setRange(10, 500)   # entspricht 0.10 - 5.00 µm (Faktor 100)
        self.slider_win.setValue(int(round(self.state["win"] * 1e6 * 100)))
        layout.addWidget(self.slider_win, row, 0, 1, 2)
        row += 1

        self.label_scan_range = QLabel()
        self.label_scan_range.setWordWrap(True)
        layout.addWidget(self.label_scan_range, row, 0, 1, 2)
        row += 1

        self.label_central_profile = QLabel()
        self.label_central_profile.setWordWrap(True)
        layout.addWidget(self.label_central_profile, row, 0, 1, 2)
        row += 1

        self.label_width = QLabel()
        layout.addWidget(self.label_width, row, 0, 1, 2)
        row += 1
        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 300)  # entspricht 0.01 - 3.00 MHz (Faktor 100)
        self.slider_width.setValue(int(round(self.state["width"] * 1e-6 * 100)))
        layout.addWidget(self.slider_width, row, 0, 1, 2)

        self._update_win_mode_enabled_state()
        self._update_param_labels()
        return box

    def _build_lens_group(self):
        box = QGroupBox("Optics (Focal Lengths)")
        layout = QGridLayout(box)

        layout.addWidget(QLabel("f1:"), 0, 0)
        self.combo_f1 = QComboBox()
        for mm in LENS_FOCAL_LENGTHS_MM:
            self.combo_f1.addItem(f"{mm:g} mm", mm * 1e-3)
        self.combo_f1.setCurrentIndex(closest_focal_index(self.state["f1"]))
        layout.addWidget(self.combo_f1, 0, 1)

        layout.addWidget(QLabel("f2:"), 1, 0)
        self.combo_f2 = QComboBox()
        for mm in LENS_FOCAL_LENGTHS_MM:
            self.combo_f2.addItem(f"{mm:g} mm", mm * 1e-3)
        self.combo_f2.setCurrentIndex(closest_focal_index(self.state["f2"]))
        layout.addWidget(self.combo_f2, 1, 1)

        self.label_lenses = QLabel()
        layout.addWidget(self.label_lenses, 2, 0, 1, 2)
        self._update_lens_label()

        self.btn_open_lens_designer = QPushButton("Open Lens Design Tool (3-Lens Model)...")
        layout.addWidget(self.btn_open_lens_designer, 3, 0, 1, 2)

        self.label_lens_designer_status = QLabel("")
        self.label_lens_designer_status.setWordWrap(True)
        self.label_lens_designer_status.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.label_lens_designer_status, 4, 0, 1, 2)

        return box

    def _build_profile_group(self):
        box = QGroupBox("Intensity Profile")
        layout = QVBoxLayout(box)
        self.cb_airy = QCheckBox("Airy disk instead of Gaussian")
        self.cb_airy.setChecked(self.state["use_airy"])
        layout.addWidget(self.cb_airy)
        # Parametrisierung des Airy-Profils - nur wirksam, solange die
        # Checkbox darueber gesetzt ist (beim Gauss-Profil gibt es keinen
        # Skalenfaktor, deshalb wird die Gruppe dann ausgegraut).
        self.airy_group = airy_scale.AiryScaleGroup(AIRY_SCALE_FACTOR)
        self.airy_group.setEnabled(self.state["use_airy"])
        self.airy_group.factor_spin.valueChanged.connect(self.on_airy_scale_changed)
        layout.addWidget(self.airy_group)
        return box

    def _build_amp_group(self):
        self.amp_group = QGroupBox("Amplitudes")
        outer_layout = QVBoxLayout(self.amp_group)

        self.cb_custom_amps = QCheckBox("Use individual amplitudes")
        self.cb_custom_amps.setChecked(self.state["custom_amps"])
        outer_layout.addWidget(self.cb_custom_amps)

        self.amp_scroll = QScrollArea()
        self.amp_scroll.setWidgetResizable(True)
        self.amp_scroll.setMaximumHeight(220)
        self.amp_inner = QWidget()
        self.amp_grid = QGridLayout(self.amp_inner)
        self.amp_scroll.setWidget(self.amp_inner)
        outer_layout.addWidget(self.amp_scroll)

        self.btn_reset_amps = QPushButton("Reset amplitudes (= 1)")
        outer_layout.addWidget(self.btn_reset_amps)

        self.rebuild_amplitude_widgets()
        return self.amp_group

    def _build_atom_group(self):
        """NEU: Atom-Parameter fürs gewichtete Modell (sigma_atom) - Startwerte
        aus der Messung (Rb-85, T = 17 µK, nu_r = 60.4 kHz)."""
        self.atom_group = QGroupBox("Atom (weighted metrics)")
        layout = QGridLayout(self.atom_group)
        row = 0

        self.cb_weighted = QCheckBox("Use atom-weighted uniformity/crosstalk")
        self.cb_weighted.setChecked(self.state["weighted_mode"])
        layout.addWidget(self.cb_weighted, row, 0, 1, 2)
        row += 1

        layout.addWidget(QLabel("Species:"), row, 0)
        self.combo_species = QComboBox()
        for name in RB_MASSES_KG:
            self.combo_species.addItem(name)
        self.combo_species.setCurrentText(self.state["atom_species"])
        layout.addWidget(self.combo_species, row, 1)
        row += 1

        self.label_atom_T = QLabel()
        layout.addWidget(self.label_atom_T, row, 0, 1, 2)
        row += 1
        self.slider_atom_T = QSlider(Qt.Horizontal)
        self.slider_atom_T.setRange(1, 500)   # 1 - 500 µK (Faktor 1)
        self.slider_atom_T.setValue(int(round(self.state["atom_temperature"] * 1e6)))
        layout.addWidget(self.slider_atom_T, row, 0, 1, 2)
        row += 1

        self.label_atom_nu = QLabel()
        layout.addWidget(self.label_atom_nu, row, 0, 1, 2)
        row += 1
        self.slider_atom_nu = QSlider(Qt.Horizontal)
        self.slider_atom_nu.setRange(1, 300)  # 1 - 300 kHz (Faktor 1)
        self.slider_atom_nu.setValue(int(round(self.state["atom_trap_freq"] * 1e-3)))
        layout.addWidget(self.slider_atom_nu, row, 0, 1, 2)
        row += 1

        self.label_sigma_atom = QLabel()
        self.label_sigma_atom.setWordWrap(True)
        layout.addWidget(self.label_sigma_atom, row, 0, 1, 2)

        self._update_atom_labels()
        return self.atom_group

    def _build_atom_position_group(self):
        """NEU: Verschiebung des Atoms relativ zum Zentrum des Lichtflecks
        (dem Site, an dem die Falle nominell sitzt) - z.B. um eine
        Fehljustage/einen Versatz zwischen Dipolfalle und Lichtmuster zu
        untersuchen. Wirkt auf ALLE Uniformity-/Crosstalk-Metriken (hart wie
        atom-gewichtet) sowie auf die Standardposition des Fadenkreuzes -
        siehe _atom_center(). Begrenzt auf +/- pitch/2 in jeder Dimension
        (siehe _max_atom_offset())."""
        self.atom_pos_group = QGroupBox("Atom Position (offset from light-pattern center)")
        layout = QGridLayout(self.atom_pos_group)
        row = 0

        max_offset_nm = int(round(self._max_atom_offset() * 1e9))

        self.label_atom_offset_x = QLabel()
        layout.addWidget(self.label_atom_offset_x, row, 0, 1, 2)
        row += 1
        self.slider_atom_offset_x = QSlider(Qt.Horizontal)
        self.slider_atom_offset_x.setRange(-max_offset_nm, max_offset_nm)
        self.slider_atom_offset_x.setValue(int(round(self.state["atom_offset_x"] * 1e9)))
        layout.addWidget(self.slider_atom_offset_x, row, 0, 1, 2)
        row += 1

        self.label_atom_offset_y = QLabel()
        layout.addWidget(self.label_atom_offset_y, row, 0, 1, 2)
        row += 1
        self.slider_atom_offset_y = QSlider(Qt.Horizontal)
        self.slider_atom_offset_y.setRange(-max_offset_nm, max_offset_nm)
        self.slider_atom_offset_y.setValue(int(round(self.state["atom_offset_y"] * 1e9)))
        layout.addWidget(self.slider_atom_offset_y, row, 0, 1, 2)
        row += 1

        self.btn_center_atom = QPushButton("Center atom (Δ = 0)")
        layout.addWidget(self.btn_center_atom, row, 0, 1, 2)
        row += 1

        note = QLabel(f"(limited to ± pitch/2 = ±{self._max_atom_offset()*1e6:.3f} µm per axis)")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note, row, 0, 1, 2)

        self._update_atom_offset_labels()
        return self.atom_pos_group

    def _build_region_group(self):
        self.region_group = QGroupBox("Regions (Uniformity / Crosstalk)")
        layout = QVBoxLayout(self.region_group)
        self.btn_uniform_to_spots = QPushButton("Uniformity = spot square")
        self.btn_crosstalk_to_pitch = QPushButton("Crosstalk = pitch")
        layout.addWidget(self.btn_uniform_to_spots)
        layout.addWidget(self.btn_crosstalk_to_pitch)
        note = QLabel("(disabled in weighted mode - the atom-weighted region has\nno hard edge, see 'Atom' group above)")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(note)
        self._update_region_controls_enabled()
        return self.region_group

    def _build_crosshair_group(self):
        box = QGroupBox("Crosshair")
        layout = QVBoxLayout(box)
        self.btn_reset_crosshair = QPushButton("Reset crosshair to center")
        layout.addWidget(self.btn_reset_crosshair)
        return box

    def _build_save_group(self):
        box = QGroupBox("Export")
        layout = QVBoxLayout(box)
        self.btn_save = QPushButton("Save high-resolution")
        layout.addWidget(self.btn_save)
        self.label_save_status = QLabel("")
        # Nur der Dateiname wird angezeigt (nicht der ggf. sehr lange UNC-Pfad),
        # damit das Label die rechte Spalte nicht mehr in die Breite zieht: ein
        # langer Pfad ohne Leerzeichen wird von QLabel trotz setWordWrap(True)
        # nicht umgebrochen und hat zuvor die Buttons/Labels verschoben. Der
        # vollständige Pfad steht weiterhin als Tooltip zur Verfügung.
        self.label_save_status.setWordWrap(True)
        self.label_save_status.setMaximumWidth(360)
        layout.addWidget(self.label_save_status)
        return box

    def _update_param_labels(self):
        self.label_win_in.setText(f"win before lenses = {self.state['win_in']*1e3:.3f} mm")
        self.label_win.setText(f"win after lenses   = {self.state['win']*1e6:.3f} µm")

        if self.state["external_scan_range_um"] is not None:
            r_um = self.state["external_scan_range_um"]
            n_atoms = r_um * 1e-6 / pitch
            source_note = "  (from 3-lens model)"
        else:
            r_um = radius_from_angle(theta_max, self.state["f1"], self.state["f2"], fLO) * 1e6
            n_atoms = r_um * 1e-6 / pitch
            source_note = ""
        self.label_scan_range.setText(
            f"Scanning range r (full angle){source_note} = {r_um:.3f} µm\n"
            f"≈ {n_atoms:.2f} atoms (at pitch {pitch*1e6:.3f} µm)"
        )

        central_um = width_to_um(self.state["width"], self.state["f1"], self.state["f2"])
        if self.state["external_central_profile_um"] is not None:
            diam_um = self.state["external_central_profile_um"]
            diam_note = ", 3-lens model"
        else:
            diam_um = 2 * self.state["win"] * 1e6
            diam_note = ""
        self.label_central_profile.setText(
            f"Central profile size (from width) = {central_um:.3f} µm\n"
            f"(waist on both sides added{diam_note} = {diam_um:.3f} µm)"
        )

        self.label_width.setText(f"width = {self.state['width']*1e-6:.2f} MHz")

        # win beeinflusst das waist/sigma_atom-Verhaeltnis im Atom-Label
        if hasattr(self, "label_sigma_atom"):
            self._update_atom_labels()

    def _update_win_mode_enabled_state(self):
        """Nur der Slider des aktiven Modus ist bedienbar; beide Werte bleiben sichtbar."""
        is_input_mode = self.state["win_mode"] == "input"
        self.slider_win_in.setEnabled(is_input_mode)
        self.slider_win.setEnabled(not is_input_mode)

    def _update_lens_label(self):
        f1_mm = self.state["f1"] * 1e3
        f2_mm = self.state["f2"] * 1e3
        magnification = f2_mm / f1_mm
        self.label_lenses.setText(
            f"Current: f1 = {f1_mm:g} mm    f2 = {f2_mm:g} mm    (M = f2/f1 = {magnification:.2f})"
        )

    def _current_sigma_atom(self):
        """Berechnet sigma_atom aus den aktuellen Atom-Parametern im state."""
        m = RB_MASSES_KG[self.state["atom_species"]]
        omega = 2 * np.pi * self.state["atom_trap_freq"]
        return sigma_thermal(m, omega, self.state["atom_temperature"])

    def _atom_center(self):
        """Tatsächliche Position des Atoms: das Zentrum des Lichtflecks
        (r_center, für x und y identisch, siehe compute_centers()) plus die
        über die Slider einstellbare Verschiebung atom_offset_x/y. Diese
        Position - NICHT mehr einfach (r_center, r_center) - ist ab jetzt
        der maßgebliche Bezugspunkt für Uniformity-/Crosstalk-Regionen
        (hart wie gewichtet), die sigma_atom-Konturen/-Kreise und die
        Standard-/Reset-Position des Fadenkreuzes."""
        r_center = self.cache.get("r_center", 0.0)
        return r_center + self.state["atom_offset_x"], r_center + self.state["atom_offset_y"]

    def _max_atom_offset(self):
        """Maximal erlaubte Verschiebung des Atoms in jeder Dimension:
        +/- die Hälfte des Site-Pitch (weiter entfernt läge das Atom näher
        an einem Nachbar-Site als am eigenen)."""
        return pitch / 2

    def _update_atom_labels(self):
        sigma = self._current_sigma_atom()
        self.state["sigma_atom"] = sigma
        self.label_atom_T.setText(f"Temperature T = {self.state['atom_temperature']*1e6:.1f} µK")
        self.label_atom_nu.setText(f"Trap frequency ν_r = {self.state['atom_trap_freq']*1e-3:.1f} kHz")
        if sigma is not None and np.isfinite(sigma) and sigma > 0:
            ratio_pitch = pitch / sigma
            ratio_win = self.state["win"] / sigma
            self.label_sigma_atom.setText(
                f"σ_atom = {sigma*1e9:.1f} nm\n"
                f"pitch / σ_atom = {ratio_pitch:.1f}      waist / σ_atom = {ratio_win:.1f}"
            )
        else:
            self.label_sigma_atom.setText("σ_atom: ungültig (Parameter prüfen)")

    def _update_atom_offset_labels(self):
        max_offset = self._max_atom_offset()
        dx, dy = self.state["atom_offset_x"], self.state["atom_offset_y"]
        self.label_atom_offset_x.setText(
            f"Δx = {dx*1e6:+.3f} µm  ({dx/max_offset*100:+.0f} % of pitch/2)"
        )
        self.label_atom_offset_y.setText(
            f"Δy = {dy*1e6:+.3f} µm  ({dy/max_offset*100:+.0f} % of pitch/2)"
        )

    def _update_region_controls_enabled(self):
        is_weighted = self.state["weighted_mode"]
        self.btn_uniform_to_spots.setEnabled(not is_weighted)
        self.btn_crosstalk_to_pitch.setEnabled(not is_weighted)

    def _mark_pending(self):
        """Wird von allen *_update()-Methoden aufgerufen, wenn im manuellen
        Modus eine Neuberechnung/Neuzeichnung wegen einer Zustandsänderung
        eigentlich fällig wäre, aber (mangels Klick auf 'Update now')
        übersprungen wird - vermerkt das nur fürs Status-Label."""
        self._pending_manual_update = True
        self._update_manual_status_label()

    def _update_manual_status_label(self):
        if not self.state.get("manual_update_mode", False):
            self.label_manual_update_status.setText("")
            return
        if self._pending_manual_update:
            self.label_manual_update_status.setText("Changes pending – click 'Update now'")
            self.label_manual_update_status.setStyleSheet("color: #b36b00; font-weight: bold; font-size: 10px;")
        else:
            self.label_manual_update_status.setText("Up to date")
            self.label_manual_update_status.setStyleSheet("color: gray; font-size: 10px;")

    # --------------------------------------------------------
    # Signale verbinden
    # --------------------------------------------------------
    def _connect_signals(self):
        self.cb_manual_update.stateChanged.connect(self.on_manual_mode_toggle)
        self.btn_manual_update.clicked.connect(self.on_manual_update_clicked)
        self.spin_nx.valueChanged.connect(self.on_nx_changed)
        self.spin_ny.valueChanged.connect(self.on_ny_changed)
        self.combo_f1.currentIndexChanged.connect(self.on_f1_changed)
        self.combo_f2.currentIndexChanged.connect(self.on_f2_changed)
        self.btn_open_lens_designer.clicked.connect(self.on_open_lens_designer)
        self.cb_win_mode.stateChanged.connect(self.on_win_mode_toggle)
        self.slider_win_in.valueChanged.connect(self.on_win_in_changed)
        self.slider_win.valueChanged.connect(self.on_win_changed)
        self.slider_width.valueChanged.connect(self.on_width_changed)
        self.cb_airy.stateChanged.connect(self.on_profile_toggle)
        self.cb_custom_amps.stateChanged.connect(self.on_amp_toggle)
        self.btn_reset_amps.clicked.connect(self.on_reset_amps)
        self.cb_weighted.stateChanged.connect(self.on_weighted_toggle)
        self.combo_species.currentTextChanged.connect(self.on_species_changed)
        self.slider_atom_T.valueChanged.connect(self.on_atom_T_changed)
        self.slider_atom_nu.valueChanged.connect(self.on_atom_nu_changed)
        self.slider_atom_offset_x.valueChanged.connect(self.on_atom_offset_x_changed)
        self.slider_atom_offset_y.valueChanged.connect(self.on_atom_offset_y_changed)
        self.btn_center_atom.clicked.connect(self.on_center_atom_clicked)
        self.btn_uniform_to_spots.clicked.connect(self.on_set_uniform_to_spots)
        self.btn_crosstalk_to_pitch.clicked.connect(self.on_set_crosstalk_to_pitch)
        self.btn_reset_crosshair.clicked.connect(self.on_reset_crosshair)
        self.btn_save.clicked.connect(self.on_save)

    # --------------------------------------------------------
    # Amplituden-Panel dynamisch (neu) aufbauen
    # --------------------------------------------------------
    def rebuild_amplitude_widgets(self):
        # alte Widgets entfernen
        while self.amp_grid.count():
            item = self.amp_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.amp_spinboxes_x = []
        self.amp_spinboxes_y = []

        self.amp_scroll.setVisible(self.state["custom_amps"])
        self.btn_reset_amps.setVisible(self.state["custom_amps"])
        if not self.state["custom_amps"]:
            return

        N_x, N_y = self.state["N_x"], self.state["N_y"]
        col_pairs = 2  # 2 (Label+Spinbox)-Paare pro Zeile
        row = 0
        col = 0

        def add_entry(label_text, value, callback):
            nonlocal row, col
            lbl = QLabel(label_text)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(3)
            spin.setValue(value)
            spin.valueChanged.connect(callback)
            self.amp_grid.addWidget(lbl, row, col * 2)
            self.amp_grid.addWidget(spin, row, col * 2 + 1)
            col += 1
            if col >= col_pairs:
                col = 0
                row += 1
            return spin

        for i in range(N_x):
            spin = add_entry(f"x{i}", self.state["amp_x"][i],
                              lambda val, idx=i: self.on_amp_x_changed(idx, val))
            self.amp_spinboxes_x.append(spin)

        for j in range(N_y):
            spin = add_entry(f"y{j}", self.state["amp_y"][j],
                              lambda val, idx=j: self.on_amp_y_changed(idx, val))
            self.amp_spinboxes_y.append(spin)

    # --------------------------------------------------------
    # Berechnungs- / Zeichenroutinen
    # --------------------------------------------------------
    def current_amp_spots(self):
        if self.state["custom_amps"]:
            amp_x = self.state["amp_x"]
            amp_y = self.state["amp_y"]
        else:
            amp_x = np.ones(self.state["N_x"])
            amp_y = np.ones(self.state["N_y"])
        return np.repeat(amp_x, self.state["N_y"]) * np.tile(amp_y, self.state["N_x"])

    def compute_masks_and_metrics(self):
        """
        Berechnet IMMER beide Metrik-Familien (harte Box + atom-gewichtet),
        unabhängig vom aktiven Modus - so bleibt beim Umschalten der
        Checkbox sofort ein konsistenter Vergleichswert verfügbar, ohne
        alles neu rechnen zu müssen. cache['uniformity']/['crosstalk']
        halten jeweils die AKTIVE (im Titel angezeigte) Metrik.
        """
        X, Y = self.cache["X"], self.cache["Y"]
        atom_cx, atom_cy = self._atom_center()
        I_ort = self.cache["I_ort"]
        I_neighbor = self.cache["I_neighbor"]

        # --- harte Box-Metriken (um die tatsächliche Atom-Position, siehe _atom_center()) ---
        mask_uniformity = overlap_mask_pitch(X, Y, atom_cx, atom_cy, self.state["uniformity_side_length"])
        mask_crosstalk = overlap_mask_pitch(X, Y, atom_cx, atom_cy, self.state["crosstalk_side_length"])

        I_inside_uniform = I_ort[mask_uniformity]
        I_inside_cross = I_ort[mask_crosstalk]
        I_neighbor_inside = I_neighbor[mask_crosstalk]

        if len(I_inside_uniform) == 0 or np.mean(I_inside_uniform) == 0:
            uniformity_hard = float("nan")
        else:
            uniformity_hard = np.std(I_inside_uniform) / np.mean(I_inside_uniform)

        if len(I_inside_cross) == 0 or np.sum(I_inside_cross) == 0:
            crosstalk_hard = float("nan")
        else:
            crosstalk_hard = np.sum(I_neighbor_inside) / np.sum(I_inside_cross)

        self.cache["mask_uniformity"] = mask_uniformity
        self.cache["mask_crosstalk"] = mask_crosstalk
        self.cache["uniformity_hard"] = uniformity_hard
        self.cache["crosstalk_hard"] = crosstalk_hard

        # --- atom-gewichtete Metriken (auf feinem lokalen Sub-Grid) ---
        sigma_atom = self._current_sigma_atom()
        self.state["sigma_atom"] = sigma_atom
        centers_x, centers_y = self.cache["centers_x"], self.cache["centers_y"]
        amp_spots = self.current_amp_spots()

        if sigma_atom is not None and np.isfinite(sigma_atom) and sigma_atom > 0:
            xs, ys, Xs, Ys = build_local_weighted_grid(
                atom_cx, atom_cy, sigma_atom, WEIGHTED_N_SIGMA, WEIGHTED_GRID_N
            )
            I_own_raw = compute_intensity_profile(
                Xs, Ys, centers_x, centers_y, self.state["win"], amp_spots, self.state["use_airy"]
            )
            I_neigh_raw = local_neighbor_intensity(
                Xs, Ys, pitch, centers_x, centers_y, self.state["win"], amp_spots, self.state["use_airy"]
            )
            W = atom_weight_2d(Xs, Ys, atom_cx, atom_cy, sigma_atom)

            # uniformity_weighted/eta_weighted sind skaleninvariante Verhältnisse -
            # eine gemeinsame Normierung von I_own/I_neigh ändert den Wert nicht,
            # wird hier nur für die Anzeige (imshow-Farbskala) durchgeführt.
            uniformity_weighted = weighted_uniformity(I_own_raw, W)
            eta_weighted = weighted_crosstalk(I_own_raw, I_neigh_raw, W)

            # NEU: Aufschlüsselung von eta_weighted nach Kanten- (oben/unten/
            # links/rechts, Abstand=pitch) vs. Diagonal-Nachbarn (Ecken,
            # Abstand=sqrt(2)*pitch) - als Vertrauens-Check und Diagnose, siehe
            # weighted_crosstalk_breakdown() im MultitoneFlatTopOptimizer für
            # die ausführliche Begründung. edge_sum + diag_sum muss exakt
            # eta_weighted ergeben (Intensitäten addieren sich immer, keine
            # destruktive Interferenz zwischen den Nachbarkopien).
            denom = np.sum(I_own_raw * W)
            if denom != 0:
                per_direction = {}
                for ix in (-1, 0, 1):
                    for iy in (-1, 0, 1):
                        if ix == 0 and iy == 0:
                            continue
                        I_dir = compute_intensity_profile(
                            Xs, Ys, centers_x + ix * pitch, centers_y + iy * pitch,
                            self.state["win"], amp_spots, self.state["use_airy"]
                        )
                        per_direction[(ix, iy)] = np.sum(I_dir * W) / denom
                edge_keys = [k for k in per_direction if abs(k[0]) + abs(k[1]) == 1]
                diag_keys = [k for k in per_direction if abs(k[0]) == 1 and abs(k[1]) == 1]
                self.cache["crosstalk_edge_per_neighbor"] = sum(per_direction[k] for k in edge_keys) / len(edge_keys)
                self.cache["crosstalk_diag_per_neighbor"] = sum(per_direction[k] for k in diag_keys) / len(diag_keys)
            else:
                self.cache["crosstalk_edge_per_neighbor"] = float("nan")
                self.cache["crosstalk_diag_per_neighbor"] = float("nan")

            peak = np.max(I_own_raw)
            if peak > 0:
                self.cache["I_own_local"] = I_own_raw / peak
                self.cache["I_neighbor_local"] = I_neigh_raw / peak
            else:
                self.cache["I_own_local"] = I_own_raw
                self.cache["I_neighbor_local"] = I_neigh_raw
            self.cache["local_x"] = xs
            self.cache["local_y"] = ys
            self.cache["local_X"] = Xs
            self.cache["local_Y"] = Ys
            self.cache["W_local"] = W
        else:
            uniformity_weighted = float("nan")
            eta_weighted = float("nan")
            self.cache["crosstalk_edge_per_neighbor"] = float("nan")
            self.cache["crosstalk_diag_per_neighbor"] = float("nan")

        self.cache["uniformity_weighted"] = uniformity_weighted
        self.cache["crosstalk_weighted"] = eta_weighted

        # --- aktive Metrik (für Titel/Export) ---
        if self.state["weighted_mode"]:
            self.cache["uniformity"] = uniformity_weighted
            self.cache["crosstalk"] = eta_weighted
        else:
            self.cache["uniformity"] = uniformity_hard
            self.cache["crosstalk"] = crosstalk_hard

    def _local_cut_arrays(self, n_points):
        """Feine 1D-Schnittlinien (x und y) durch die tatsächliche Atom-
        Position (_atom_center(), i.e. r_center + Atom-Offset) für den
        gewichteten Modus, plus normierte Intensität und normierte
        Wahrscheinlichkeitsdichte entlang jeder Linie. Ersetzt im
        gewichteten Modus die groben Schnitte aus dem globalen GRID_N/
        GRID_N_HIGHRES-Gitter, das für sigma_atom-Skalen (~100 nm) zu grob
        aufgelöst wäre."""
        atom_cx, atom_cy = self._atom_center()
        centers_x, centers_y = self.cache["centers_x"], self.cache["centers_y"]
        amp_spots = self.current_amp_spots()
        sigma = self.state["sigma_atom"]

        x_line, y_line = build_local_cut_lines(atom_cx, atom_cy, sigma, WEIGHTED_N_SIGMA, n_points)

        I_x = compute_intensity_profile(
            x_line, np.full_like(x_line, atom_cy), centers_x, centers_y,
            self.state["win"], amp_spots, self.state["use_airy"]
        )
        I_y = compute_intensity_profile(
            np.full_like(y_line, atom_cx), y_line, centers_x, centers_y,
            self.state["win"], amp_spots, self.state["use_airy"]
        )
        peak = max(np.max(I_x), np.max(I_y), 1e-300)
        I_x = I_x / peak
        I_y = I_y / peak

        pdf_x = np.exp(-(x_line - atom_cx) ** 2 / (2 * sigma ** 2))
        pdf_y = np.exp(-(y_line - atom_cy) ** 2 / (2 * sigma ** 2))
        return x_line, y_line, I_x, I_y, pdf_x, pdf_y

    def redraw_regions(self):
        """Zeichnet je nach Modus entweder die harten Rechtecke (+ Nachbar-
        Plot + volle Schnitte) oder die atom-gewichteten Overlays (sigma-
        Konturen + lokale Nahansicht + fein aufgelöste, gezoomte Schnitte
        mit Wahrscheinlichkeitsdichte-Kurve)."""
        if self.state["weighted_mode"]:
            self._redraw_weighted_overlays()
        else:
            self._redraw_hard_rectangles()

    def _clear_weighted_artists(self):
        for c in self.sigma_circles:
            c.remove()
        self.sigma_circles = []
        if self.pdf_fill_x is not None:
            self.pdf_fill_x.remove()
            self.pdf_fill_x = None
        if self.pdf_fill_y is not None:
            self.pdf_fill_y.remove()
            self.pdf_fill_y = None

    def _redraw_hard_rectangles(self):
        # falls zuvor im gewichteten Modus: Overlays aus dem letzten Zustand entfernen
        self._clear_weighted_artists()

        self.handle_uniformity.set_visible(True)
        self.handle_crosstalk.set_visible(True)
        self.rect_uniformity.set_visible(True)
        self.rect_crosstalk.set_visible(True)

        r_center_um = self.cache["r_center"] * 1e6
        atom_cx, atom_cy = self._atom_center()
        atom_cx_um, atom_cy_um = atom_cx * 1e6, atom_cy * 1e6
        side_u_um = self.state["uniformity_side_length"] * 1e6
        side_c_um = self.state["crosstalk_side_length"] * 1e6
        half_u = side_u_um / 2
        half_c = side_c_um / 2

        self.rect_uniformity.set_xy((atom_cx_um - half_u, atom_cy_um - half_u))
        self.rect_uniformity.set_width(2 * half_u)
        self.rect_uniformity.set_height(2 * half_u)
        self.rect_uniformity.set_label(f"Uniformity region ({side_u_um:.3f} µm)")

        self.rect_crosstalk.set_xy((atom_cx_um - half_c, atom_cy_um - half_c))
        self.rect_crosstalk.set_width(2 * half_c)
        self.rect_crosstalk.set_height(2 * half_c)
        self.rect_crosstalk.set_label(f"Crosstalk region ({side_c_um:.3f} µm)")

        self.handle_uniformity.set_data([atom_cx_um + half_u], [atom_cy_um + half_u])
        self.handle_crosstalk.set_data([atom_cx_um + half_c], [atom_cy_um + half_c])

        legend = self.ax_main.legend(loc="upper right", fontsize=8)
        legend.set_zorder(10)
        legend.get_frame().set_alpha(1.0)

        # Nachbar-Plot: klassische "Neighbor regions"-Ansicht (fixes Pitch-Fenster,
        # weiterhin um das Lichtfleck-Zentrum r_center - das zeigt die Geometrie
        # der Nachbar-Spots; die tatsächliche Atom-Position wird zusätzlich als
        # Marker eingezeichnet, falls sie von r_center abweicht).
        self.ax_neighbor.clear()
        extent = [self.cache["x"][0] * 1e6, self.cache["x"][-1] * 1e6,
                  self.cache["y"][0] * 1e6, self.cache["y"][-1] * 1e6]
        self.im_neighbor = self.ax_neighbor.imshow(
            self.cache["I_neighbor"], origin="lower", extent=extent, aspect="equal", cmap="viridis"
        )
        configure_neighbor_view(self.ax_neighbor, r_center_um, pitch * 1e6)
        if abs(self.state["atom_offset_x"]) > 0 or abs(self.state["atom_offset_y"]) > 0:
            self.ax_neighbor.plot(atom_cx_um, atom_cy_um, "+", color="red", markersize=10,
                                   markeredgewidth=1.5, zorder=6, label="Atom")
        self.ax_neighbor.set_xlabel("Position $x$ (µm)", fontsize=8)
        self.ax_neighbor.set_ylabel("Position $y$ (µm)", fontsize=8)
        self.ax_neighbor.set_title("Neighbor regions (crosstalk contribution)", fontsize=8)
        self.ax_neighbor.tick_params(labelsize=7)

        # Schnittplots zurück auf die volle, globale Domäne mit den bisherigen Bändern
        x, y = self.cache["x"], self.cache["y"]
        I_ort = self.cache["I_ort"]
        row_idx, col_idx = self.state["cut_row_idx"], self.state["cut_col_idx"]
        self.line_cut_x.set_data(x * 1e6, I_ort[row_idx, :])
        self.line_cut_y.set_data(y * 1e6, I_ort[:, col_idx])
        self.ax_cut_x.set_xlim(x[0] * 1e6, x[-1] * 1e6)
        self.ax_cut_y.set_xlim(y[0] * 1e6, y[-1] * 1e6)
        self.ax_cut_x.set_ylim(0, 1.05)
        self.ax_cut_y.set_ylim(0, 1.05)

        for attr in ("cut_x_span_uniformity", "cut_x_span_crosstalk",
                     "cut_y_span_uniformity", "cut_y_span_crosstalk"):
            old = getattr(self, attr, None)
            if old is not None:
                old.remove()

        self.cut_x_span_crosstalk = self.ax_cut_x.axvspan(
            atom_cx_um - half_c, atom_cx_um + half_c,
            facecolor=(1, 0, 0, 0.12), edgecolor="red", linewidth=1.2, zorder=0
        )
        self.cut_x_span_uniformity = self.ax_cut_x.axvspan(
            atom_cx_um - half_u, atom_cx_um + half_u,
            facecolor=(0, 1, 1, 0.18), edgecolor="cyan", linewidth=1.2, zorder=1
        )
        self.cut_y_span_crosstalk = self.ax_cut_y.axvspan(
            atom_cy_um - half_c, atom_cy_um + half_c,
            facecolor=(1, 0, 0, 0.12), edgecolor="red", linewidth=1.2, zorder=0
        )
        self.cut_y_span_uniformity = self.ax_cut_y.axvspan(
            atom_cy_um - half_u, atom_cy_um + half_u,
            facecolor=(0, 1, 1, 0.18), edgecolor="cyan", linewidth=1.2, zorder=1
        )
        legend_x = self.ax_cut_x.legend(fontsize=7, loc="upper right") if self.ax_cut_x.get_legend_handles_labels()[0] else None

    def _redraw_weighted_overlays(self):
        """Atom-gewichteter Modus: keine harten Rechtecke, stattdessen
        sigma_atom-Konturen im Hauptplot, eine auf sigma_atom neu skalierte
        lokale Nahansicht statt des Nachbar-Plots, und in beiden Schnitt-
        plots eine gefüllte Wahrscheinlichkeitsdichte-Kurve statt der
        festen farbigen Bänder."""
        # bisherige Rechtecke/Handles ausblenden (Legende zeigt sie dann nicht mehr)
        self.handle_uniformity.set_visible(False)
        self.handle_crosstalk.set_visible(False)
        self.handle_uniformity.set_data([], [])
        self.handle_crosstalk.set_data([], [])
        self.rect_uniformity.set_visible(False)
        self.rect_crosstalk.set_visible(False)
        old_legend = self.ax_main.get_legend()
        if old_legend is not None:
            old_legend.remove()

        self._clear_weighted_artists()

        atom_cx, atom_cy = self._atom_center()
        atom_cx_um, atom_cy_um = atom_cx * 1e6, atom_cy * 1e6
        sigma = self.state["sigma_atom"]
        sigma_um = sigma * 1e6 if (sigma is not None and np.isfinite(sigma)) else None

        # Hauptplot (große Ansicht): DEZENTE 1/2/3-sigma-Ringe (keine Legende)
        # an der eigenen Site UND an den 8 pitch-verschobenen Nachbar-Sites,
        # nur zur groben räumlichen Orientierung. Die Ringe folgen der
        # tatsächlichen Atom-Position (inkl. Offset) - der Offset wird als
        # für alle Sites identisch angenommen (systematischer Versatz
        # Atom-Falle vs. Lichtmuster).
        if sigma_um is not None and sigma_um > 0:
            pitch_um = pitch * 1e6
            for ix in (-1, 0, 1):
                for iy in (-1, 0, 1):
                    cx = atom_cx_um + ix * pitch_um
                    cy = atom_cy_um + iy * pitch_um
                    for n in (1, 2, 3):
                        circ = Circle((cx, cy), n * sigma_um, edgecolor="white", facecolor="none",
                                       linewidth=0.6, alpha=0.35, zorder=6)
                        self.ax_main.add_patch(circ)
                        self.sigma_circles.append(circ)

        # --- lokale Nahansicht statt "Neighbor regions" ---
        self.ax_neighbor.clear()
        if "local_x" in self.cache and sigma_um is not None and sigma_um > 0:
            xs, ys = self.cache["local_x"], self.cache["local_y"]
            I_own_local = self.cache["I_own_local"]
            extent_local = [xs[0] * 1e6, xs[-1] * 1e6, ys[0] * 1e6, ys[-1] * 1e6]
            self.im_neighbor = self.ax_neighbor.imshow(
                I_own_local, origin="lower", extent=extent_local, aspect="equal", cmap="viridis"
            )
            # 1/2/3-sigma-Konturen NUR hier (gezoomte Ansicht), dezent mit
            # Inline-Beschriftung direkt auf der Linie statt einer Legende.
            levels = [np.exp(-4.5), np.exp(-2.0), np.exp(-0.5)]  # 3σ, 2σ, 1σ Konturwerte der Gauss-Gewichtung
            level_labels = {levels[0]: "3σ", levels[1]: "2σ", levels[2]: "1σ"}
            Xs_um, Ys_um = self.cache["local_X"] * 1e6, self.cache["local_Y"] * 1e6
            try:
                cs = self.ax_neighbor.contour(Xs_um, Ys_um, self.cache["W_local"], levels=levels,
                                               colors="white", linewidths=0.8, linestyles="--", alpha=0.8)
                self.ax_neighbor.clabel(cs, fmt=level_labels, fontsize=6, inline=True)
            except Exception:
                pass  # Konturen sind rein informativ - bei degenerierten Werten einfach auslassen
            self.ax_neighbor.set_xlabel("Position $x$ (µm)", fontsize=8)
            self.ax_neighbor.set_ylabel("Position $y$ (µm)", fontsize=8)
            self.ax_neighbor.set_title(
                f"Local view (±{WEIGHTED_N_SIGMA}σ, σ={sigma*1e9:.0f} nm)",
                fontsize=8
            )
            self.ax_neighbor.tick_params(labelsize=7)
        else:
            self.ax_neighbor.text(0.5, 0.5, "σ_atom invalid", ha="center", va="center", transform=self.ax_neighbor.transAxes)

        # --- Schnittplots: intelligent auf sigma_atom neu skaliert + PDF-Kurve ---
        if sigma_um is not None and sigma_um > 0:
            x_line, y_line, I_x, I_y, pdf_x, pdf_y = self._local_cut_arrays(WEIGHTED_CUT_POINTS)
            self.line_cut_x.set_data(x_line * 1e6, I_x)
            self.line_cut_y.set_data(y_line * 1e6, I_y)
            self.ax_cut_x.set_xlim(x_line[0] * 1e6, x_line[-1] * 1e6)
            self.ax_cut_y.set_xlim(y_line[0] * 1e6, y_line[-1] * 1e6)
            self.ax_cut_x.set_ylim(0, 1.05)
            self.ax_cut_y.set_ylim(0, 1.05)

            self.pdf_fill_x = self.ax_cut_x.fill_between(
                x_line * 1e6, 0, pdf_x, color="magenta", alpha=0.25, zorder=0,
                label="Atom probability density (norm.)"
            )
            self.pdf_fill_y = self.ax_cut_y.fill_between(
                y_line * 1e6, 0, pdf_y, color="magenta", alpha=0.25, zorder=0,
                label="Atom probability density (norm.)"
            )
            self.ax_cut_x.legend(fontsize=7, loc="upper right")
            self.ax_cut_y.legend(fontsize=7, loc="upper right")

    def update_title(self):
        u = self.cache.get("uniformity", float("nan"))
        c = self.cache.get("crosstalk", float("nan"))
        u_h = self.cache.get("uniformity_hard", float("nan"))
        c_h = self.cache.get("crosstalk_hard", float("nan"))
        u_w = self.cache.get("uniformity_weighted", float("nan"))
        c_w = self.cache.get("crosstalk_weighted", float("nan"))
        c_edge = self.cache.get("crosstalk_edge_per_neighbor", float("nan"))
        c_diag = self.cache.get("crosstalk_diag_per_neighbor", float("nan"))
        profile_label = "Airy" if self.state["use_airy"] else "Gaussian"

        if self.state["weighted_mode"]:
            main_line = (f"Profile = {profile_label}      Uniformity_w (σ_w/μ_w) = {u*100:.2f} %      "
                         f"Crosstalk_w (η_w) = {c*100:.3f} %")
            sub_line = f"(hart, zum Vergleich: Uniformity = {u_h*100:.2f} %, Crosstalk = {c_h*100:.3f} %)"
            breakdown_line = (f"η_w per neighbor: edge (N/S/E/W) = {c_edge*100:.4f} %, "
                               f"diagonal (corners) = {c_diag*100:.4f} %  (4 each, sum = η_w)")
            self.ax_main.set_title(f"{main_line}\n{sub_line}\n{breakdown_line}", fontsize=9, fontweight="bold")
            return
        else:
            main_line = (f"Profile = {profile_label}      Uniformity (σ/μ) = {u*100:.2f} %      "
                         f"Crosstalk (η) = {c*100:.3f} %")
            sub_line = f"(gewichtet, zum Vergleich: Uniformity_w = {u_w*100:.2f} %, Crosstalk_w = {c_w*100:.3f} %)"

        self.ax_main.set_title(f"{main_line}\n{sub_line}", fontsize=10, fontweight="bold")

    def build_rectangles_and_handles(self):
        self.rect_uniformity = Rectangle((0, 0), 0, 0, edgecolor="cyan", facecolor="none",
                                          linewidth=2, label="Uniformity region")
        self.rect_crosstalk = Rectangle((0, 0), 0, 0, edgecolor="red", facecolor="none",
                                         linewidth=2, label="Crosstalk region")
        self.ax_main.add_patch(self.rect_uniformity)
        self.ax_main.add_patch(self.rect_crosstalk)

        (self.handle_uniformity,) = self.ax_main.plot([], [], "s", color="cyan", markersize=10,
                                                        markeredgecolor="black", zorder=6)
        (self.handle_crosstalk,) = self.ax_main.plot([], [], "s", color="red", markersize=10,
                                                       markeredgecolor="black", zorder=6)

    def build_crosshair(self):
        """Dünne, durchgehende Fadenkreuz-Linien."""
        (self.crosshair_h_thin,) = self.ax_main.plot([], [], "-", color="black",
                                                       linewidth=0.8, zorder=7)
        (self.crosshair_v_thin,) = self.ax_main.plot([], [], "-", color="black",
                                                       linewidth=0.8, zorder=7)

    def redraw_crosshair(self):
        """Positioniert die Fadenkreuz-Linien anhand von state['cut_row_idx']/['cut_col_idx']
        und aktualisiert die Titel der Schnittplots mit der aktuellen Position."""
        x_um = self.cache["x"] * 1e6
        y_um = self.cache["y"] * 1e6
        row_idx = self.state["cut_row_idx"]
        col_idx = self.state["cut_col_idx"]
        x0 = x_um[col_idx]
        y0 = y_um[row_idx]

        self.crosshair_h_thin.set_data([x_um[0], x_um[-1]], [y0, y0])
        self.crosshair_v_thin.set_data([x0, x0], [y_um[0], y_um[-1]])

        if not self.state["weighted_mode"]:
            self.ax_cut_x.set_title(f"Cut along x  (y = {y0:.3f} µm)", fontsize=8)
            self.ax_cut_y.set_title(f"Cut along y  (x = {x0:.3f} µm)", fontsize=8)
        else:
            atom_cx, atom_cy = self._atom_center()
            self.ax_cut_x.set_title(f"Cut along x through atom  (y = {atom_cy*1e6:.3f} µm)", fontsize=8)
            self.ax_cut_y.set_title(f"Cut along y through atom  (x = {atom_cx*1e6:.3f} µm)", fontsize=8)

    def update_crosshair_from_event(self, xdata_um, ydata_um):
        """Bewegt das Fadenkreuz zur Mausposition und aktualisiert (im harten
        Modus) beide Schnittplots live, ohne Uniformity/Crosstalk (Masken)
        neu zu berechnen -> bleibt beim Ziehen flüssig. Im gewichteten
        Modus sind die Schnittplots fest auf den Atom-Ort/sigma_atom
        gezoomt (das Fadenkreuz dient dort nur der Orientierung im
        Hauptplot, die Cut-Kurven bleiben unverändert)."""
        x = self.cache["x"]
        y = self.cache["y"]
        x_m = xdata_um * 1e-6
        y_m = ydata_um * 1e-6
        col_idx = int(np.argmin(np.abs(x - x_m)))
        row_idx = int(np.argmin(np.abs(y - y_m)))
        self.state["cut_row_idx"] = row_idx
        self.state["cut_col_idx"] = col_idx

        if self.state.get("manual_update_mode", False):
            self._mark_pending()
            return

        if not self.state["weighted_mode"]:
            I_ort = self.cache["I_ort"]
            self.line_cut_x.set_ydata(I_ort[row_idx, :])
            self.line_cut_y.set_ydata(I_ort[:, col_idx])

        self.redraw_crosshair()
        self.canvas.draw_idle()

    def _sync_waists(self):
        """Rechnet je nach Modus win <-> win_in um den Linsen konsistent."""
        f1 = self.state["f1"]
        f2 = self.state["f2"]
        if self.state["win_mode"] == "input":
            self.state["win"] = conjugate_waist(self.state["win_in"], f1, f2)
        else:
            self.state["win_in"] = conjugate_waist(self.state["win"], f1, f2)

    def _sync_waist_sliders(self):
        """Zieht die Position beider Waist-Slider mit dem aktuellen state nach,
        auch für den gerade nicht aktiv bedienten (nur-Anzeige) Slider."""
        self.slider_win.blockSignals(True)
        self.slider_win.setValue(int(round(self.state["win"] * 1e6 * 100)))
        self.slider_win.blockSignals(False)

        self.slider_win_in.blockSignals(True)
        self.slider_win_in.setValue(int(round(self.state["win_in"] * 1e3 * 100)))
        self.slider_win_in.blockSignals(False)

    def full_update(self, force=False):
        """Neu: Zentren, Grid, Intensitätsbilder (bei N_x, N_y, win, width, f1, f2).

        Wenn "manual_update_mode" aktiv ist und der Aufruf nicht explizit mit
        force=True erzwungen wird (das tut nur on_manual_update_clicked()),
        wird die eigentliche Neuberechnung/Neuzeichnung übersprungen und nur
        vermerkt, dass eine Aktualisierung aussteht - siehe _mark_pending()."""
        self._sync_waists()

        if self.state.get("manual_update_mode", False) and not force:
            self._mark_pending()
            return

        centers_x, centers_y, r_center = compute_centers(
            self.state["N_x"], self.state["N_y"], self.state["width"],
            self.state["f1"], self.state["f2"]
        )
        x, y, X, Y = compute_grid(
            centers_x, centers_y, self.state["win"],
            self.state["uniformity_side_length"], self.state["crosstalk_side_length"],
            GRID_N
        )

        amp_spots = self.current_amp_spots()
        I_ort = compute_intensity_profile(X, Y, centers_x, centers_y, self.state["win"], amp_spots, self.state["use_airy"])
        if np.max(I_ort) > 0:
            I_ort = I_ort / np.max(I_ort)
        I_neighbor = create_neighbourhood(X, Y, pitch, centers_x, centers_y, self.state["win"],
                                           amps=amp_spots, use_airy=self.state["use_airy"])

        # Fadenkreuz bei jeder Grid-Neuberechnung auf die tatsächliche
        # Atom-Position setzen (r_center + Atom-Offset, siehe _atom_center()) -
        # NICHT einfach len(...)//2, das bei einer geraden Gitterauflösung
        # (GRID_N) um einen halben Pixel neben dem geometrischen Zentrum liegt
        # und zusätzlich den Atom-Offset ignorieren würde.
        atom_cx_target = r_center + self.state["atom_offset_x"]
        atom_cy_target = r_center + self.state["atom_offset_y"]
        mid_y_idx = int(np.argmin(np.abs(y - atom_cy_target)))
        mid_x_idx = int(np.argmin(np.abs(x - atom_cx_target)))
        self.state["cut_row_idx"] = mid_y_idx
        self.state["cut_col_idx"] = mid_x_idx

        self.cache.update({
            "x": x, "y": y, "X": X, "Y": Y,
            "centers_x": centers_x, "centers_y": centers_y, "r_center": r_center,
            "I_ort": I_ort, "I_neighbor": I_neighbor,
        })

        extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]

        self.ax_main.clear()
        self.ax_neighbor.clear()
        self.ax_cut_x.clear()
        self.ax_cut_y.clear()
        # clear() entfernt bereits vorhandene Bänder/Patches implizit -> alte Referenzen ungültig machen
        self.cut_x_span_uniformity = None
        self.cut_x_span_crosstalk = None
        self.cut_y_span_uniformity = None
        self.cut_y_span_crosstalk = None
        self.sigma_circles = []
        self.pdf_fill_x = None
        self.pdf_fill_y = None

        self.im_main = self.ax_main.imshow(I_ort, origin="lower", extent=extent, aspect="equal", cmap="viridis")

        self.ax_main.scatter(centers_x * 1e6, centers_y * 1e6, c="white", edgecolors="black", s=20, zorder=5)

        self.ax_main.set_xlabel("Position $x$ (µm)")
        self.ax_main.set_ylabel("Position $y$ (µm)")

        (self.line_cut_x,) = self.ax_cut_x.plot(x * 1e6, I_ort[mid_y_idx, :], "b-", linewidth=2,
                                                  label="Intensity", zorder=3)
        self.ax_cut_x.set_xlabel("Position $x$ (µm)", fontsize=8)
        self.ax_cut_x.set_ylabel("Intensity", fontsize=8)
        self.ax_cut_x.tick_params(labelsize=7)
        self.ax_cut_x.grid(True, alpha=0.3)
        self.ax_cut_x.set_ylim(0, 1.05)

        (self.line_cut_y,) = self.ax_cut_y.plot(y * 1e6, I_ort[:, mid_x_idx], "g-", linewidth=2,
                                                  label="Intensity", zorder=3)
        self.ax_cut_y.set_xlabel("Position $y$ (µm)", fontsize=8)
        self.ax_cut_y.set_ylabel("Intensity", fontsize=8)
        self.ax_cut_y.tick_params(labelsize=7)
        self.ax_cut_y.grid(True, alpha=0.3)
        self.ax_cut_y.set_ylim(0, 1.05)

        self.build_rectangles_and_handles()
        self.build_crosshair()

        self.compute_masks_and_metrics()
        self.redraw_regions()
        self.redraw_crosshair()
        self.update_title()
        self._update_param_labels()
        self._update_atom_labels()
        self._sync_waist_sliders()
        self._update_region_controls_enabled()

        self.canvas.draw_idle()

    def medium_update(self, force=False):
        """Nur Intensitätsbilder neu (bei Amplituden-/Profiländerung), Grid/Zentren bleiben.

        Siehe full_update() zum "manual_update_mode"-Gate."""
        if self.state.get("manual_update_mode", False) and not force:
            self._mark_pending()
            return

        if "X" not in self.cache:
            self.full_update(force=force)
            return

        X, Y = self.cache["X"], self.cache["Y"]
        centers_x, centers_y = self.cache["centers_x"], self.cache["centers_y"]
        amp_spots = self.current_amp_spots()

        I_ort = compute_intensity_profile(X, Y, centers_x, centers_y, self.state["win"], amp_spots, self.state["use_airy"])
        if np.max(I_ort) > 0:
            I_ort = I_ort / np.max(I_ort)
        I_neighbor = create_neighbourhood(X, Y, pitch, centers_x, centers_y, self.state["win"],
                                           amps=amp_spots, use_airy=self.state["use_airy"])

        self.cache["I_ort"] = I_ort
        self.cache["I_neighbor"] = I_neighbor

        self.im_main.set_data(I_ort)

        row_idx = self.state["cut_row_idx"]
        col_idx = self.state["cut_col_idx"]
        if not self.state["weighted_mode"]:
            self.line_cut_x.set_ydata(I_ort[row_idx, :])
            self.line_cut_y.set_ydata(I_ort[:, col_idx])

        self.compute_masks_and_metrics()
        self.redraw_regions()
        self.update_title()
        self.canvas.draw_idle()

    def fast_update(self, force=False):
        """Nur Masken/Kennzahlen/Rechtecke neu (beim Ziehen der Regionen, nur harter Modus relevant).

        Siehe full_update() zum "manual_update_mode"-Gate."""
        if self.state.get("manual_update_mode", False) and not force:
            self._mark_pending()
            return

        if "X" not in self.cache:
            self.full_update(force=force)
            return
        self.compute_masks_and_metrics()
        self.redraw_regions()
        self.update_title()
        self.canvas.draw_idle()

    def atom_update(self, force=False):
        """NEU: leichte Update-Stufe für Änderungen an den Atom-Parametern
        (Spezies, Temperatur, Fallenfrequenz) - Grid/Intensitätsbilder
        bleiben unverändert, nur sigma_atom und alles davon Abhängige
        (gewichtete Metriken, Konturen, lokale Nahansicht, gezoomte
        Schnittplots) werden neu berechnet.

        Die Atom-Labels (Temperatur/Fallenfrequenz/sigma_atom-Text) werden
        immer sofort aktualisiert (billig, reine Text-Anzeige) - nur die
        eigentliche Neuberechnung/Neuzeichnung des Plots respektiert das
        "manual_update_mode"-Gate, siehe full_update()."""
        self._update_atom_labels()

        if self.state.get("manual_update_mode", False) and not force:
            self._mark_pending()
            return

        if "X" not in self.cache:
            return
        self.compute_masks_and_metrics()
        self.redraw_regions()
        self.redraw_crosshair()
        self.update_title()
        self.canvas.draw_idle()

    # --------------------------------------------------------
    # Callbacks: Qt-Widgets
    # --------------------------------------------------------
    def on_manual_mode_toggle(self, checked_state):
        self.state["manual_update_mode"] = bool(checked_state)
        self.btn_manual_update.setEnabled(self.state["manual_update_mode"])
        if not self.state["manual_update_mode"] and self._pending_manual_update:
            # Zurück in den Live-Modus: eine noch ausstehende Änderung sofort anwenden.
            self.on_manual_update_clicked()
        else:
            self._update_manual_status_label()

    def on_manual_update_clicked(self):
        """Führt die im manuellen Modus zurückgehaltene(n) Änderung(en) jetzt
        vollständig aus. full_update(force=True) berechnet dabei sicherheits-
        halber immer ALLES neu (Grid, Zentren, Intensitätsbilder, Masken/
        Kennzahlen, Overlays) - unabhängig davon, ob z.B. nur eine Amplitude,
        ein Atom-Parameter oder eine Region-Größe geändert wurde. Da
        full_update() das Fadenkreuz routinemäßig auf die Mitte zurücksetzt
        (siehe dortiger Kommentar), wird die zuletzt vom Nutzer gewählte
        Fadenkreuz-Position hier zusätzlich gesichert und - falls weiterhin
        gültig - wiederhergestellt."""
        saved_row = self.state.get("cut_row_idx")
        saved_col = self.state.get("cut_col_idx")

        self.full_update(force=True)

        if saved_row is not None and saved_col is not None and "x" in self.cache:
            n_y = len(self.cache["y"])
            n_x = len(self.cache["x"])
            if saved_row < n_y and saved_col < n_x:
                self.state["cut_row_idx"] = saved_row
                self.state["cut_col_idx"] = saved_col
                if not self.state["weighted_mode"]:
                    I_ort = self.cache["I_ort"]
                    self.line_cut_x.set_ydata(I_ort[saved_row, :])
                    self.line_cut_y.set_ydata(I_ort[:, saved_col])
                self.redraw_crosshair()

        self._pending_manual_update = False
        self._update_manual_status_label()
        self.canvas.draw_idle()

    def on_nx_changed(self, value):
        self.state["N_x"] = value
        self.state["amp_x"] = np.ones(value)
        self.rebuild_amplitude_widgets()
        self.full_update()

    def on_ny_changed(self, value):
        self.state["N_y"] = value
        self.state["amp_y"] = np.ones(value)
        self.rebuild_amplitude_widgets()
        self.full_update()

    def _clear_external_optics(self):
        """Invalidiert Werte, die zuletzt vom Lens Design Tool übernommen wurden,
        sobald der Nutzer f1/f2/win/win_in/Modus manuell im Hauptfenster ändert."""
        self.state["external_scan_range_um"] = None
        self.state["external_central_profile_um"] = None

    def on_win_changed(self, value):
        self._clear_external_optics()
        self.state["win"] = value / 100.0 * 1e-6
        self._update_param_labels()
        self.full_update()

    def on_win_in_changed(self, value):
        self._clear_external_optics()
        self.state["win_in"] = value / 100.0 * 1e-3
        self._update_param_labels()
        self.full_update()

    def on_win_mode_toggle(self, checked_state):
        self._clear_external_optics()
        self.state["win_mode"] = "input" if checked_state else "output"
        self._update_win_mode_enabled_state()
        self.full_update()

    def on_f1_changed(self, index):
        self._clear_external_optics()
        self.state["f1"] = self.combo_f1.itemData(index)
        self._update_lens_label()
        self.full_update()

    def on_f2_changed(self, index):
        self._clear_external_optics()
        self.state["f2"] = self.combo_f2.itemData(index)
        self._update_lens_label()
        self.full_update()

    def on_width_changed(self, value):
        self.state["width"] = value / 100.0 * 1e6
        self._update_param_labels()
        self.full_update()

    def on_profile_toggle(self, checked_state):
        self.state["use_airy"] = bool(checked_state)
        if hasattr(self, "airy_group"):
            self.airy_group.setEnabled(self.state["use_airy"])
        self.medium_update()

    def on_airy_scale_changed(self, value):
        """Neuer Airy-Skalenfaktor: setzt das Modul-Global, das
        compute_intensity_profile() benutzt, und rechnet neu. Wirkt nur
        beim Airy-Profil."""
        global AIRY_SCALE_FACTOR
        AIRY_SCALE_FACTOR = float(value)
        if self.state["use_airy"]:
            self.medium_update()

    def on_amp_toggle(self, checked_state):
        self.state["custom_amps"] = bool(checked_state)
        self.rebuild_amplitude_widgets()
        self.medium_update()

    def on_amp_x_changed(self, idx, value):
        self.state["amp_x"][idx] = value
        self.medium_update()

    def on_amp_y_changed(self, idx, value):
        self.state["amp_y"][idx] = value
        self.medium_update()

    def on_reset_amps(self):
        self.state["amp_x"] = np.ones(self.state["N_x"])
        self.state["amp_y"] = np.ones(self.state["N_y"])
        for spin in self.amp_spinboxes_x:
            spin.blockSignals(True)
            spin.setValue(1.0)
            spin.blockSignals(False)
        for spin in self.amp_spinboxes_y:
            spin.blockSignals(True)
            spin.setValue(1.0)
            spin.blockSignals(False)
        self.medium_update()

    def on_weighted_toggle(self, checked_state):
        self.state["weighted_mode"] = bool(checked_state)
        self._update_region_controls_enabled()
        self.full_update()

    def on_species_changed(self, text):
        self.state["atom_species"] = text
        self.atom_update()

    def on_atom_T_changed(self, value):
        self.state["atom_temperature"] = value * 1e-6
        self.atom_update()

    def on_atom_nu_changed(self, value):
        self.state["atom_trap_freq"] = value * 1e3
        self.atom_update()

    def on_atom_offset_x_changed(self, value):
        self.state["atom_offset_x"] = value * 1e-9
        self._update_atom_offset_labels()
        self.fast_update()

    def on_atom_offset_y_changed(self, value):
        self.state["atom_offset_y"] = value * 1e-9
        self._update_atom_offset_labels()
        self.fast_update()

    def on_center_atom_clicked(self):
        self.state["atom_offset_x"] = 0.0
        self.state["atom_offset_y"] = 0.0
        self.slider_atom_offset_x.blockSignals(True)
        self.slider_atom_offset_x.setValue(0)
        self.slider_atom_offset_x.blockSignals(False)
        self.slider_atom_offset_y.blockSignals(True)
        self.slider_atom_offset_y.setValue(0)
        self.slider_atom_offset_y.blockSignals(False)
        self._update_atom_offset_labels()
        self.fast_update()

    def on_set_uniform_to_spots(self):
        if "centers_x" not in self.cache:
            return
        cx, cy, rc = self.cache["centers_x"], self.cache["centers_y"], self.cache["r_center"]
        half_extent = max(np.max(np.abs(cx - rc)), np.max(np.abs(cy - rc)))
        self.state["uniformity_side_length"] = 2 * half_extent
        self.fast_update()

    def on_set_crosstalk_to_pitch(self):
        self.state["crosstalk_side_length"] = pitch
        self.fast_update()

    def on_reset_crosshair(self):
        """Setzt das Fadenkreuz auf die tatsächliche Atom-Position zurück
        (r_center + Atom-Offset, siehe _atom_center()) - nicht auf den rein
        geometrischen Gittermittelpunkt, siehe Kommentar in full_update()."""
        if "x" not in self.cache:
            return
        atom_cx, atom_cy = self._atom_center()
        row_idx = int(np.argmin(np.abs(self.cache["y"] - atom_cy)))
        col_idx = int(np.argmin(np.abs(self.cache["x"] - atom_cx)))
        self.state["cut_row_idx"] = row_idx
        self.state["cut_col_idx"] = col_idx

        if self.state.get("manual_update_mode", False):
            self._mark_pending()
            return

        if not self.state["weighted_mode"]:
            I_ort = self.cache["I_ort"]
            self.line_cut_x.set_ydata(I_ort[row_idx, :])
            self.line_cut_y.set_ydata(I_ort[:, col_idx])

        self.redraw_crosshair()
        self.canvas.draw_idle()

    def on_open_lens_designer(self):
        if self.lens_dialog is not None and self.lens_dialog.isVisible():
            self.lens_dialog.raise_()
            self.lens_dialog.activateWindow()
            return

        self.lens_dialog = LensDesignDialog(
            parent=self,
            f1_mm=self.state["f1"] * 1e3,
            f2_mm=self.state["f2"] * 1e3,
            w0_mm=self.state["win_in"] * 1e3,
            wavelength_nm=lambda_opt * 1e9,
            atom_size_um=pitch * 1e6,
        )
        self.lens_dialog.optics_changed.connect(self.on_external_optics_changed)
        self.label_lens_designer_status.setText("Lens Design Tool open \u2013 values apply live.")
        self.lens_dialog.show()

    def on_external_optics_changed(self, data):
        """Wird bei jeder Neuberechnung im Lens Design Tool live aufgerufen und
        übernimmt die (physikalisch genaueren) Ergebnisse in das Hauptfenster."""
        idx1 = closest_focal_index(data["f1_mm"] * 1e-3)
        idx2 = closest_focal_index(data["f2_mm"] * 1e-3)

        self.combo_f1.blockSignals(True)
        self.combo_f1.setCurrentIndex(idx1)
        self.combo_f1.blockSignals(False)
        self.combo_f2.blockSignals(True)
        self.combo_f2.setCurrentIndex(idx2)
        self.combo_f2.blockSignals(False)
        self.state["f1"] = self.combo_f1.itemData(idx1)
        self.state["f2"] = self.combo_f2.itemData(idx2)
        self._update_lens_label()

        # win direkt vom simulierten Ergebnis übernehmen (genauer als die interne Formel,
        # da beliebige, auch nicht-ideale Linsenabstände berücksichtigt werden)
        self.state["win"] = data["w_screen_um"] * 1e-6
        self.state["win_in"] = data["w0_mm"] * 1e-3
        self.state["win_mode"] = "output"
        self.cb_win_mode.blockSignals(True)
        self.cb_win_mode.setChecked(False)
        self.cb_win_mode.blockSignals(False)
        self._update_win_mode_enabled_state()

        self.slider_win.blockSignals(True)
        self.slider_win.setValue(int(round(self.state["win"] * 1e6 * 100)))
        self.slider_win.blockSignals(False)
        self.slider_win_in.blockSignals(True)
        self.slider_win_in.setValue(int(round(self.state["win_in"] * 1e3 * 100)))
        self.slider_win_in.blockSignals(False)

        self.state["external_scan_range_um"] = data["scan_range_um"]
        self.state["external_central_profile_um"] = data["central_profile_um"]

        self.label_lens_designer_status.setText(
            f"Last synced from Lens Design Tool: f1={data['f1_mm']:g} mm, f2={data['f2_mm']:g} mm, "
            f"f3={data['f3_mm']:g} mm"
        )

        self.full_update()

    def on_save(self):
        self._sync_waists()  # sicherstellen, dass win/win_in konsistent sind (z.B. falls Modus zuletzt geändert wurde)
        centers_x, centers_y, r_center = compute_centers(
            self.state["N_x"], self.state["N_y"], self.state["width"],
            self.state["f1"], self.state["f2"]
        )
        atom_cx = r_center + self.state["atom_offset_x"]
        atom_cy = r_center + self.state["atom_offset_y"]
        x, y, X, Y = compute_grid(
            centers_x, centers_y, self.state["win"],
            self.state["uniformity_side_length"], self.state["crosstalk_side_length"],
            GRID_N_HIGHRES
        )
        amp_spots = self.current_amp_spots()
        I_ort = compute_intensity_profile(X, Y, centers_x, centers_y, self.state["win"], amp_spots, self.state["use_airy"])
        I_ort /= np.max(I_ort)
        I_neighbor = create_neighbourhood(X, Y, pitch, centers_x, centers_y, self.state["win"],
                                           amps=amp_spots, use_airy=self.state["use_airy"])

        weighted = self.state["weighted_mode"]
        sigma_atom = self._current_sigma_atom()
        self.state["sigma_atom"] = sigma_atom
        crosstalk_edge = crosstalk_diag = float("nan")

        if weighted and sigma_atom is not None and np.isfinite(sigma_atom) and sigma_atom > 0:
            xs, ys, Xs, Ys = build_local_weighted_grid(
                atom_cx, atom_cy, sigma_atom, WEIGHTED_N_SIGMA, WEIGHTED_GRID_N_HIGHRES
            )
            I_own_raw = compute_intensity_profile(Xs, Ys, centers_x, centers_y, self.state["win"],
                                                    amp_spots, self.state["use_airy"])
            I_neigh_raw = local_neighbor_intensity(Xs, Ys, pitch, centers_x, centers_y, self.state["win"],
                                                     amp_spots, self.state["use_airy"])
            W = atom_weight_2d(Xs, Ys, atom_cx, atom_cy, sigma_atom)
            uniformity = weighted_uniformity(I_own_raw, W)
            crosstalk = weighted_crosstalk(I_own_raw, I_neigh_raw, W)
            peak_local = np.max(I_own_raw)
            I_own_local = I_own_raw / peak_local if peak_local > 0 else I_own_raw

            # Aufschlüsselung nach Kanten-/Diagonal-Nachbarn (siehe compute_masks_and_metrics())
            denom_bd = np.sum(I_own_raw * W)
            if denom_bd != 0:
                per_direction = {}
                for ix in (-1, 0, 1):
                    for iy in (-1, 0, 1):
                        if ix == 0 and iy == 0:
                            continue
                        I_dir = compute_intensity_profile(
                            Xs, Ys, centers_x + ix * pitch, centers_y + iy * pitch,
                            self.state["win"], amp_spots, self.state["use_airy"]
                        )
                        per_direction[(ix, iy)] = np.sum(I_dir * W) / denom_bd
                edge_keys = [k for k in per_direction if abs(k[0]) + abs(k[1]) == 1]
                diag_keys = [k for k in per_direction if abs(k[0]) == 1 and abs(k[1]) == 1]
                crosstalk_edge = sum(per_direction[k] for k in edge_keys) / len(edge_keys)
                crosstalk_diag = sum(per_direction[k] for k in diag_keys) / len(diag_keys)
            else:
                crosstalk_edge = crosstalk_diag = float("nan")

            x_line, y_line = build_local_cut_lines(atom_cx, atom_cy, sigma_atom, WEIGHTED_N_SIGMA,
                                                     WEIGHTED_CUT_POINTS_HIGHRES)
            I_x_line = compute_intensity_profile(x_line, np.full_like(x_line, atom_cy), centers_x, centers_y,
                                                  self.state["win"], amp_spots, self.state["use_airy"])
            I_y_line = compute_intensity_profile(np.full_like(y_line, atom_cx), y_line, centers_x, centers_y,
                                                  self.state["win"], amp_spots, self.state["use_airy"])
            peak_line = max(np.max(I_x_line), np.max(I_y_line), 1e-300)
            I_x_line /= peak_line
            I_y_line /= peak_line
            pdf_x = np.exp(-(x_line - atom_cx) ** 2 / (2 * sigma_atom ** 2))
            pdf_y = np.exp(-(y_line - atom_cy) ** 2 / (2 * sigma_atom ** 2))
        else:
            mask_u = overlap_mask_pitch(X, Y, atom_cx, atom_cy, self.state["uniformity_side_length"])
            mask_c = overlap_mask_pitch(X, Y, atom_cx, atom_cy, self.state["crosstalk_side_length"])
            uniformity = np.std(I_ort[mask_u]) / np.mean(I_ort[mask_u])
            crosstalk = np.sum(I_neighbor[mask_c]) / np.sum(I_ort[mask_c])

        extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]

        # Aktuelle Fadenkreuz-Position (aus der interaktiven Ansicht) auf das
        # hochauflösende Grid übertragen, statt immer die Mitte zu nehmen
        if self.state["cut_row_idx"] is not None and "y" in self.cache and not weighted:
            y_pos_m = self.cache["y"][self.state["cut_row_idx"]]
            x_pos_m = self.cache["x"][self.state["cut_col_idx"]]
        else:
            y_pos_m = atom_cy
            x_pos_m = atom_cx
        mid_y_idx = int(np.argmin(np.abs(y - y_pos_m)))
        mid_x_idx = int(np.argmin(np.abs(x - x_pos_m)))
        y_pos_um = y[mid_y_idx] * 1e6
        x_pos_um = x[mid_x_idx] * 1e6

        profile_label = "Airy" if self.state["use_airy"] else "Gaussian"
        metric_label = "Uniformity_w/Crosstalk_w (atom-weighted)" if weighted else "Uniformity/Crosstalk (hard box)"

        suptitle_text = (
            f"f1={self.state['f1']*1e3:g} mm, f2={self.state['f2']*1e3:g} mm, "
            f"win_in={self.state['win_in']*1e3:.3f} mm, win={self.state['win']*1e6:.3f} µm, "
            f"width={self.state['width']*1e-6:.3f} MHz, Profile={profile_label}, {metric_label}: "
            f"Uniformity={uniformity*100:.2f}%, Crosstalk={crosstalk*100:.3f}%"
        )
        if weighted and np.isfinite(crosstalk_edge) and np.isfinite(crosstalk_diag):
            suptitle_text += (f"\nη_w per neighbor: edge (N/S/E/W) = {crosstalk_edge*100:.4f} %, "
                               f"diagonal (corners) = {crosstalk_diag*100:.4f} %  (4 each, sum = η_w)")

        fig_save = plt.figure(figsize=(13, 10))
        fig_save.set_constrained_layout(True)
        fig_save.suptitle(suptitle_text, fontsize=EXPORT_FONTSIZE_SUPTITLE, fontweight="bold")

        gs_save = fig_save.add_gridspec(3, 2, width_ratios=[1.6, 1], height_ratios=[1.8, 1, 1])
        ax1 = fig_save.add_subplot(gs_save[:, 0])
        ax2 = fig_save.add_subplot(gs_save[0, 1])
        ax3 = fig_save.add_subplot(gs_save[1, 1])
        ax4 = fig_save.add_subplot(gs_save[2, 1])

        ax1.imshow(I_ort, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        ax1.scatter(centers_x * 1e6, centers_y * 1e6, c="white", edgecolors="black", s=15, zorder=5)
        ax1.axhline(y_pos_um, color="black", linewidth=0.8, zorder=6)
        ax1.axvline(x_pos_um, color="black", linewidth=0.8, zorder=6)
        r_um = r_center * 1e6
        atom_cx_um, atom_cy_um = atom_cx * 1e6, atom_cy * 1e6

        if weighted and sigma_atom is not None and np.isfinite(sigma_atom) and sigma_atom > 0:
            # große Ansicht: dezente 1/2/3-sigma-Ringe an eigener Site + 8
            # Nachbarn, keine Legende (siehe interaktive Ansicht /
            # _redraw_weighted_overlays()) - zentriert auf die tatsächliche
            # Atom-Position (inkl. Offset).
            sigma_um = sigma_atom * 1e6
            pitch_um = pitch * 1e6
            for ix in (-1, 0, 1):
                for iy in (-1, 0, 1):
                    for n in (1, 2, 3):
                        ax1.add_patch(Circle((atom_cx_um + ix * pitch_um, atom_cy_um + iy * pitch_um), n * sigma_um,
                                              edgecolor="white", facecolor="none",
                                              linewidth=0.6, alpha=0.35, zorder=6))
        else:
            side_u_um = self.state["uniformity_side_length"] * 1e6
            side_c_um = self.state["crosstalk_side_length"] * 1e6
            half_u = side_u_um / 2
            half_c = side_c_um / 2
            ax1.add_patch(Rectangle((atom_cx_um - half_u, atom_cy_um - half_u), 2 * half_u, 2 * half_u,
                                     edgecolor="cyan", facecolor="none", linewidth=2,
                                     label=f"Uniformity region ({side_u_um:.3f} µm)"))
            ax1.add_patch(Rectangle((atom_cx_um - half_c, atom_cy_um - half_c), 2 * half_c, 2 * half_c,
                                     edgecolor="red", facecolor="none", linewidth=2,
                                     label=f"Crosstalk region ({side_c_um:.3f} µm)"))
            leg1 = ax1.legend(fontsize=EXPORT_FONTSIZE_LEGEND)
            leg1.set_zorder(10)
            leg1.get_frame().set_alpha(1.0)

        ax1.set_xlabel("Position $x$ (µm)", fontsize=EXPORT_FONTSIZE_LABEL)
        ax1.set_ylabel("Position $y$ (µm)", fontsize=EXPORT_FONTSIZE_LABEL)
        ax1.tick_params(labelsize=EXPORT_FONTSIZE_TICK)

        if weighted and sigma_atom is not None and np.isfinite(sigma_atom) and sigma_atom > 0:
            extent_local = [xs[0] * 1e6, xs[-1] * 1e6, ys[0] * 1e6, ys[-1] * 1e6]
            ax2.imshow(I_own_local, origin="lower", extent=extent_local, aspect="equal", cmap="viridis")
            # 1/2/3-sigma-Konturen NUR hier (gezoomte Ansicht), dezent mit
            # Inline-Beschriftung direkt auf der Linie statt einer Legende.
            levels = [np.exp(-4.5), np.exp(-2.0), np.exp(-0.5)]
            level_labels = {levels[0]: "3σ", levels[1]: "2σ", levels[2]: "1σ"}
            Xs_um, Ys_um = Xs * 1e6, Ys * 1e6
            try:
                cs2 = ax2.contour(Xs_um, Ys_um, W, levels=levels, colors="white", linewidths=0.8,
                                   linestyles="--", alpha=0.8)
                ax2.clabel(cs2, fmt=level_labels, fontsize=6, inline=True)
            except Exception:
                pass
            ax2.set_title(f"Local view (±{WEIGHTED_N_SIGMA}σ, σ={sigma_atom*1e9:.0f} nm)",
                          fontsize=EXPORT_FONTSIZE_TITLE)
        else:
            ax2.imshow(I_neighbor, origin="lower", extent=extent, aspect="equal", cmap="viridis")
            configure_neighbor_view(ax2, r_um, pitch * 1e6)
            if abs(self.state["atom_offset_x"]) > 0 or abs(self.state["atom_offset_y"]) > 0:
                ax2.plot(atom_cx_um, atom_cy_um, "+", color="red", markersize=10,
                         markeredgewidth=1.5, zorder=6, label="Atom")
                ax2.legend(fontsize=EXPORT_FONTSIZE_LEGEND * 0.8, loc="upper right")
            ax2.set_title("Neighbor regions", fontsize=EXPORT_FONTSIZE_TITLE)
        ax2.set_xlabel("Position $x$ (µm)", fontsize=EXPORT_FONTSIZE_LABEL)
        ax2.set_ylabel("Position $y$ (µm)", fontsize=EXPORT_FONTSIZE_LABEL)
        ax2.tick_params(labelsize=EXPORT_FONTSIZE_TICK)

        if weighted and sigma_atom is not None and np.isfinite(sigma_atom) and sigma_atom > 0:
            ax3.plot(x_line * 1e6, I_x_line, "b-", linewidth=2, label="Intensity", zorder=3)
            ax3.fill_between(x_line * 1e6, 0, pdf_x, color="magenta", alpha=0.25, zorder=0,
                              label="Atom probability density (norm.)")
            ax3.legend(fontsize=EXPORT_FONTSIZE_LEGEND * 0.7, loc="upper right")
            ax3.set_title(f"Cut along x through atom  (y = {atom_cy_um:.3f} µm)", fontsize=EXPORT_FONTSIZE_TITLE)

            ax4.plot(y_line * 1e6, I_y_line, "g-", linewidth=2, label="Intensity", zorder=3)
            ax4.fill_between(y_line * 1e6, 0, pdf_y, color="magenta", alpha=0.25, zorder=0,
                              label="Atom probability density (norm.)")
            ax4.legend(fontsize=EXPORT_FONTSIZE_LEGEND * 0.7, loc="upper right")
            ax4.set_title(f"Cut along y through atom  (x = {atom_cx_um:.3f} µm)", fontsize=EXPORT_FONTSIZE_TITLE)
        else:
            ax3.plot(x * 1e6, I_ort[mid_y_idx, :], "b-", linewidth=2)
            ax3.set_title(f"Cut along x  (y = {y_pos_um:.3f} µm)", fontsize=EXPORT_FONTSIZE_TITLE)

            ax4.plot(y * 1e6, I_ort[:, mid_x_idx], "g-", linewidth=2)
            ax4.set_title(f"Cut along y  (x = {x_pos_um:.3f} µm)", fontsize=EXPORT_FONTSIZE_TITLE)

        ax3.set_xlabel("Position $x$ (µm)", fontsize=EXPORT_FONTSIZE_LABEL)
        ax3.set_ylabel("Intensity", fontsize=EXPORT_FONTSIZE_LABEL)
        ax3.tick_params(labelsize=EXPORT_FONTSIZE_TICK)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)

        ax4.set_xlabel("Position $y$ (µm)", fontsize=EXPORT_FONTSIZE_LABEL)
        ax4.set_ylabel("Intensity", fontsize=EXPORT_FONTSIZE_LABEL)
        ax4.tick_params(labelsize=EXPORT_FONTSIZE_TICK)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "weighted" if weighted else "hard"
        out_file = out_dir / f"FlatMultiTone_GUI_{mode_tag}_{timestamp}.png"
        try:
            fig_save.savefig(out_file, dpi=150, bbox_inches="tight")
            self.label_save_status.setText(f"Saved: {out_file.name}")
            self.label_save_status.setToolTip(str(out_file))
        except Exception as e:
            self.label_save_status.setText(f"Error while saving: {e}")
            self.label_save_status.setToolTip(str(out_file))
            QMessageBox.warning(self, "Save failed", str(e))
        finally:
            plt.close(fig_save)

    # --------------------------------------------------------
    # Drag & Drop im Hauptplot: Region-Handles ODER Fadenkreuz
    # (matplotlib-Events auf dem Qt-Canvas)
    # --------------------------------------------------------
    def on_button_press(self, event):
        if event.inaxes != self.ax_main or event.xdata is None or event.ydata is None:
            return

        # Priorität: liegt der Klick nah genug an einem Region-Handle, dieses greifen.
        # Im gewichteten Modus sind handle_uniformity/handle_crosstalk leer (siehe
        # _redraw_weighted_overlays()), sodass diese Schleife dort automatisch
        # nichts findet und direkt auf Fadenkreuz-Verhalten zurückfällt.
        click_disp = np.array(self.ax_main.transData.transform((event.xdata, event.ydata)))
        for target, handle in (("uniformity", self.handle_uniformity), ("crosstalk", self.handle_crosstalk)):
            hx, hy = handle.get_data()
            if len(hx) == 0:
                continue
            handle_disp = np.array(self.ax_main.transData.transform((hx[0], hy[0])))
            if np.hypot(*(handle_disp - click_disp)) < 12:
                self.dragging_target = target
                return

        # sonst: Fadenkreuz an die Klickposition bewegen (und bei Ziehen live mitführen)
        self.dragging_target = "crosshair"
        self.update_crosshair_from_event(event.xdata, event.ydata)

    def on_motion(self, event):
        if self.dragging_target is None:
            return
        if event.inaxes != self.ax_main or event.xdata is None or event.ydata is None:
            return

        if self.dragging_target == "crosshair":
            self.update_crosshair_from_event(event.xdata, event.ydata)
            return

        if self.state["weighted_mode"]:
            # Regionen koennen im gewichteten Modus nicht gezogen werden
            # (keine harten Grenzen mehr) - sollte wegen on_button_press
            # ohnehin nie erreicht werden, ist hier nur als Sicherheitsnetz.
            return

        atom_cx, atom_cy = self._atom_center()
        dx = event.xdata - atom_cx * 1e6
        dy = event.ydata - atom_cy * 1e6
        half_um = max(abs(dx), abs(dy))
        half_um = float(np.clip(half_um, 0.02, 25.0))
        side_m = 2 * half_um * 1e-6

        if self.dragging_target == "uniformity":
            self.state["uniformity_side_length"] = side_m
        else:
            self.state["crosstalk_side_length"] = side_m

        self.fast_update()

    def on_release(self, event):
        self.dragging_target = None


def main():
    app = QApplication(sys.argv)
    window = WeightedFlatMultiToneWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()