"""
FlatMultiTone GUI (PyQt5-Version)
==================================
Identische Funktionalität wie das matplotlib-widgets-Skript, aber mit nativen
Qt-Widgets für die Steuerung. Vorteil: Qt-Layouts ordnen alle Bedienelemente
automatisch überlappungsfrei an, unabhängig von Fenstergröße oder Bildschirm-DPI.

Voraussetzung: PyQt5 muss installiert sein.
    pip install PyQt5

Start:
    python FlatMultiTone_GUI_PyQt5.py

Einstellbar:
- N_x, N_y (Spinboxen, ganze Zahlen)
- win, width (Slider mit Live-Wertanzeige)
- Airy-Scheibchen statt Gauß-Profil (Checkbox)
- Individuelle Amplituden pro Ton (Checkbox schaltet Eingabefelder frei)
- Uniformity-Region / Crosstalk-Region: per Drag&Drop der farbigen Eckmarker im
  Hauptplot in der Größe veränderbar, plus Reset-Buttons
  ("Uniformity = Spot-Quadrat", "Crosstalk = Pitch")
- Hochauflösend speichern: schreibt eine PNG-Datei mit Intensität, Nachbarn und
  beiden Schnittplots
"""

import sys
import datetime
from pathlib import Path as FilePath

import numpy as np
from scipy.special import j1

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QGroupBox, QScrollArea, QSplitter, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt


# ============================================================
# Physikalische Konstanten (fix, nicht über die GUI einstellbar)
# ============================================================
offset = 100e6        # Hz
f1 = 45e-3             # m
f2 = 300e-3            # m
fLO = 52.88e-3         # m
theta_max = 43e-3      # rad
f_band = 36e6          # Hz
lambda_opt = 795e-9    # m
pitch = 5.288e-6       # m, physikalischer Atomabstand (Verschiebung der Nachbar-Kopien)

GRID_N = 260            # Auflösung für die interaktive Vorschau
GRID_N_HIGHRES = 1000   # Auflösung für den "Speichern"-Button

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
    if N <= 1:
        return np.array([offset], dtype=float)
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


def compute_intensity_profile(X, Y, centers_x, centers_y, width_param, amps, use_airy):
    if use_airy:
        first_zero_radius = 1.19 * width_param
        return airy_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, first_zero_radius, amps)
    else:
        return gaussian_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, width_param, amps)


def create_neighbourhood(X, Y, pitch, centers_x, centers_y, w_in, amps=None, use_airy=False):
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


def compute_centers(N_x, N_y, width):
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


def compute_grid(centers_x, centers_y, win, uniformity_side_length, crosstalk_side_length, resolution):
    margin = max(10 * win, 1.3 * uniformity_side_length, 1.3 * crosstalk_side_length)
    x = np.linspace(-margin / 2, np.max(np.abs(centers_x)) + margin / 2, resolution)
    y = np.linspace(-margin / 2, np.max(np.abs(centers_y)) + margin / 2, resolution)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


# ============================================================
# Hauptfenster
# ============================================================
class FlatMultiToneWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlatMultiTone GUI (PyQt5)")
        self.resize(1500, 900)

        self.state = {
            "N_x": 3,
            "N_y": 4,
            "win": 1.6e-6,
            "width": 0.3e6,
            "uniformity_side_length": 2.6e-6,
            "crosstalk_side_length": pitch,
            "custom_amps": False,
            "use_airy": False,
            "amp_x": np.ones(3),
            "amp_y": np.ones(4),
            "cut_row_idx": None,   # Fadenkreuz: Zeilenindex (bestimmt Schnitt entlang x, feste y-Position)
            "cut_col_idx": None,   # Fadenkreuz: Spaltenindex (bestimmt Schnitt entlang y, feste x-Position)
        }
        self.cache = {}
        self.dragging_target = None

        # Widgets, die dynamisch neu erzeugt werden (Amplituden-Panel)
        self.amp_spinboxes_x = []
        self.amp_spinboxes_y = []

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

        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.6, 1])
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_neighbor = self.fig.add_subplot(gs[0, 1])
        self.ax_cut_x = self.fig.add_subplot(gs[1, 1])
        self.ax_cut_y = self.fig.add_subplot(gs[2, 1])

        splitter.addWidget(self.canvas)

        # ---------------- rechte Seite: Steuerbereich ----------------
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setAlignment(Qt.AlignTop)

        control_layout.addWidget(self._build_grid_group())
        control_layout.addWidget(self._build_param_group())
        control_layout.addWidget(self._build_profile_group())
        control_layout.addWidget(self._build_amp_group())
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

    def _build_grid_group(self):
        box = QGroupBox("Spot-Gitter")
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
        box = QGroupBox("Strahlparameter")
        layout = QGridLayout(box)

        self.label_win = QLabel()
        layout.addWidget(self.label_win, 0, 0, 1, 2)
        self.slider_win = QSlider(Qt.Horizontal)
        self.slider_win.setRange(10, 500)   # entspricht 0.10 - 5.00 µm (Faktor 100)
        self.slider_win.setValue(int(round(self.state["win"] * 1e6 * 100)))
        layout.addWidget(self.slider_win, 1, 0, 1, 2)

        self.label_width = QLabel()
        layout.addWidget(self.label_width, 2, 0, 1, 2)
        self.slider_width = QSlider(Qt.Horizontal)
        self.slider_width.setRange(1, 300)  # entspricht 0.01 - 3.00 MHz (Faktor 100)
        self.slider_width.setValue(int(round(self.state["width"] * 1e-6 * 100)))
        layout.addWidget(self.slider_width, 3, 0, 1, 2)

        self._update_param_labels()
        return box

    def _build_profile_group(self):
        box = QGroupBox("Intensitätsprofil")
        layout = QVBoxLayout(box)
        self.cb_airy = QCheckBox("Airy-Scheibchen statt Gauß")
        self.cb_airy.setChecked(self.state["use_airy"])
        layout.addWidget(self.cb_airy)
        return box

    def _build_amp_group(self):
        self.amp_group = QGroupBox("Amplituden")
        outer_layout = QVBoxLayout(self.amp_group)

        self.cb_custom_amps = QCheckBox("Individuelle Amplituden verwenden")
        self.cb_custom_amps.setChecked(self.state["custom_amps"])
        outer_layout.addWidget(self.cb_custom_amps)

        self.amp_scroll = QScrollArea()
        self.amp_scroll.setWidgetResizable(True)
        self.amp_scroll.setMaximumHeight(220)
        self.amp_inner = QWidget()
        self.amp_grid = QGridLayout(self.amp_inner)
        self.amp_scroll.setWidget(self.amp_inner)
        outer_layout.addWidget(self.amp_scroll)

        self.btn_reset_amps = QPushButton("Reset Amplituden (= 1)")
        outer_layout.addWidget(self.btn_reset_amps)

        self.rebuild_amplitude_widgets()
        return self.amp_group

    def _build_region_group(self):
        box = QGroupBox("Regionen (Uniformity / Crosstalk)")
        layout = QVBoxLayout(box)
        self.btn_uniform_to_spots = QPushButton("Uniformity = Spot-Quadrat")
        self.btn_crosstalk_to_pitch = QPushButton("Crosstalk = Pitch")
        layout.addWidget(self.btn_uniform_to_spots)
        layout.addWidget(self.btn_crosstalk_to_pitch)
        return box

    def _build_crosshair_group(self):
        box = QGroupBox("Fadenkreuz")
        layout = QVBoxLayout(box)
        self.btn_reset_crosshair = QPushButton("Fadenkreuz auf Mitte zurücksetzen")
        layout.addWidget(self.btn_reset_crosshair)
        return box

    def _build_save_group(self):
        box = QGroupBox("Export")
        layout = QVBoxLayout(box)
        self.btn_save = QPushButton("Hochauflösend speichern")
        layout.addWidget(self.btn_save)
        self.label_save_status = QLabel("")
        self.label_save_status.setWordWrap(True)
        layout.addWidget(self.label_save_status)
        return box

    def _update_param_labels(self):
        self.label_win.setText(f"win = {self.state['win']*1e6:.2f} µm")
        self.label_width.setText(f"width = {self.state['width']*1e-6:.2f} MHz")

    # --------------------------------------------------------
    # Signale verbinden
    # --------------------------------------------------------
    def _connect_signals(self):
        self.spin_nx.valueChanged.connect(self.on_nx_changed)
        self.spin_ny.valueChanged.connect(self.on_ny_changed)
        self.slider_win.valueChanged.connect(self.on_win_changed)
        self.slider_width.valueChanged.connect(self.on_width_changed)
        self.cb_airy.stateChanged.connect(self.on_profile_toggle)
        self.cb_custom_amps.stateChanged.connect(self.on_amp_toggle)
        self.btn_reset_amps.clicked.connect(self.on_reset_amps)
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
            spin.setRange(0.0, 2.0)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
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
        X, Y = self.cache["X"], self.cache["Y"]
        r_center = self.cache["r_center"]
        I_ort = self.cache["I_ort"]
        I_neighbor = self.cache["I_neighbor"]

        mask_uniformity = overlap_mask_pitch(X, Y, r_center, r_center, self.state["uniformity_side_length"])
        mask_crosstalk = overlap_mask_pitch(X, Y, r_center, r_center, self.state["crosstalk_side_length"])

        I_inside_uniform = I_ort[mask_uniformity]
        I_inside_cross = I_ort[mask_crosstalk]
        I_neighbor_inside = I_neighbor[mask_crosstalk]

        if len(I_inside_uniform) == 0 or np.mean(I_inside_uniform) == 0:
            uniformity = float("nan")
        else:
            uniformity = np.std(I_inside_uniform) / np.mean(I_inside_uniform)

        if len(I_inside_cross) == 0 or np.sum(I_inside_cross) == 0:
            crosstalk = float("nan")
        else:
            crosstalk = np.sum(I_neighbor_inside) / np.sum(I_inside_cross)

        self.cache["uniformity"] = uniformity
        self.cache["crosstalk"] = crosstalk

    def redraw_rectangles(self):
        r_center_um = self.cache["r_center"] * 1e6
        side_u_um = self.state["uniformity_side_length"] * 1e6
        side_c_um = self.state["crosstalk_side_length"] * 1e6
        half_u = side_u_um / 2
        half_c = side_c_um / 2

        self.rect_uniformity.set_xy((r_center_um - half_u, r_center_um - half_u))
        self.rect_uniformity.set_width(2 * half_u)
        self.rect_uniformity.set_height(2 * half_u)
        self.rect_uniformity.set_label(f"Uniformity-Bereich ({side_u_um:.3f} µm)")

        self.rect_crosstalk.set_xy((r_center_um - half_c, r_center_um - half_c))
        self.rect_crosstalk.set_width(2 * half_c)
        self.rect_crosstalk.set_height(2 * half_c)
        self.rect_crosstalk.set_label(f"Crosstalk-Bereich ({side_c_um:.3f} µm)")

        self.handle_uniformity.set_data([r_center_um + half_u], [r_center_um + half_u])
        self.handle_crosstalk.set_data([r_center_um + half_c], [r_center_um + half_c])

        self.ax_main.legend(loc="upper right", fontsize=15)

    def update_title(self):
        u = self.cache.get("uniformity", float("nan"))
        c = self.cache.get("crosstalk", float("nan"))
        profile_label = "Airy" if self.state["use_airy"] else "Gauß"
        self.ax_main.set_title(
            f"Profil = {profile_label}    Uniformity (σ/μ) = {u*100:.2f} %    "
            f"Crosstalk (η) = {c*100:.3f} %",
            fontsize=17, fontweight="bold"
        )

    def build_rectangles_and_handles(self):
        self.rect_uniformity = Rectangle((0, 0), 0, 0, edgecolor="cyan", facecolor="none",
                                          linewidth=2, label="Uniformity-Bereich")
        self.rect_crosstalk = Rectangle((0, 0), 0, 0, edgecolor="red", facecolor="none",
                                         linewidth=2, label="Crosstalk-Bereich")
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

        self.ax_cut_x.set_title(f"Schnitt entlang x  (y = {y0:.3f} µm)", fontsize=8)
        self.ax_cut_y.set_title(f"Schnitt entlang y  (x = {x0:.3f} µm)", fontsize=8)

    def update_crosshair_from_event(self, xdata_um, ydata_um):
        """Bewegt das Fadenkreuz zur Mausposition und aktualisiert beide Schnittplots live,
        ohne Uniformity/Crosstalk (Masken) neu zu berechnen -> bleibt beim Ziehen flüssig."""
        x = self.cache["x"]
        y = self.cache["y"]
        x_m = xdata_um * 1e-6
        y_m = ydata_um * 1e-6
        col_idx = int(np.argmin(np.abs(x - x_m)))
        row_idx = int(np.argmin(np.abs(y - y_m)))
        self.state["cut_row_idx"] = row_idx
        self.state["cut_col_idx"] = col_idx

        I_ort = self.cache["I_ort"]
        self.line_cut_x.set_ydata(I_ort[row_idx, :])
        self.line_cut_y.set_ydata(I_ort[:, col_idx])

        self.redraw_crosshair()
        self.canvas.draw_idle()

    def full_update(self):
        """Neu: Zentren, Grid, Intensitätsbilder (bei N_x, N_y, win, width)."""
        centers_x, centers_y, r_center = compute_centers(self.state["N_x"], self.state["N_y"], self.state["width"])
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

        mid_y_idx = len(y) // 2
        mid_x_idx = len(x) // 2
        # Fadenkreuz bei jeder Grid-Neuberechnung wieder auf die Mitte setzen
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

        self.im_main = self.ax_main.imshow(I_ort, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        self.im_neighbor = self.ax_neighbor.imshow(I_neighbor, origin="lower", extent=extent, aspect="equal", cmap="viridis")

        self.ax_main.scatter(centers_x * 1e6, centers_y * 1e6, c="white", edgecolors="black", s=20, zorder=5)

        self.ax_main.set_xlabel("Ort $x$ (µm)",fontsize=12)
        self.ax_main.set_ylabel("Ort $y$ (µm)",fontsize=12)
        self.ax_neighbor.set_xlabel("Ort $x$ (µm)", fontsize=8)
        self.ax_neighbor.set_ylabel("Ort $y$ (µm)", fontsize=8)
        self.ax_neighbor.set_title("Nachbarregionen (Crosstalk-Beitrag)", fontsize=8)
        self.ax_neighbor.tick_params(labelsize=7)

        (self.line_cut_x,) = self.ax_cut_x.plot(x * 1e6, I_ort[self.state["cut_row_idx"], :], "b-", linewidth=2)
        self.ax_cut_x.set_xlabel("Ort $x$ (µm)", fontsize=10)
        self.ax_cut_x.set_ylabel("Intensität", fontsize=10)
        self.ax_cut_x.tick_params(labelsize=8)
        self.ax_cut_x.grid(True, alpha=0.3)
        self.ax_cut_x.set_ylim(0, 1.05)

        (self.line_cut_y,) = self.ax_cut_y.plot(y * 1e6, I_ort[:, self.state["cut_col_idx"]], "g-", linewidth=2)
        self.ax_cut_y.set_xlabel("Ort $y$ (µm)", fontsize=10)
        self.ax_cut_y.set_ylabel("Intensität", fontsize=10)
        self.ax_cut_y.tick_params(labelsize=8)
        self.ax_cut_y.grid(True, alpha=0.3)
        self.ax_cut_y.set_ylim(0, 1.05)

        self.build_rectangles_and_handles()
        self.build_crosshair()

        self.compute_masks_and_metrics()
        self.redraw_rectangles()
        self.redraw_crosshair()
        self.update_title()

        self.canvas.draw_idle()

    def medium_update(self):
        """Nur Intensitätsbilder neu (bei Amplituden-/Profiländerung), Grid/Zentren bleiben."""
        if "X" not in self.cache:
            self.full_update()
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
        self.im_neighbor.set_data(I_neighbor)

        row_idx = self.state["cut_row_idx"]
        col_idx = self.state["cut_col_idx"]
        self.line_cut_x.set_ydata(I_ort[row_idx, :])
        self.line_cut_y.set_ydata(I_ort[:, col_idx])

        self.compute_masks_and_metrics()
        self.redraw_rectangles()
        self.update_title()
        self.canvas.draw_idle()

    def fast_update(self):
        """Nur Masken/Kennzahlen/Rechtecke neu (beim Ziehen der Regionen)."""
        if "X" not in self.cache:
            self.full_update()
            return
        self.compute_masks_and_metrics()
        self.redraw_rectangles()
        self.update_title()
        self.canvas.draw_idle()

    # --------------------------------------------------------
    # Callbacks: Qt-Widgets
    # --------------------------------------------------------
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

    def on_win_changed(self, value):
        self.state["win"] = value / 100.0 * 1e-6
        self._update_param_labels()
        self.full_update()

    def on_width_changed(self, value):
        self.state["width"] = value / 100.0 * 1e6
        self._update_param_labels()
        self.full_update()

    def on_profile_toggle(self, checked_state):
        self.state["use_airy"] = bool(checked_state)
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
        if "x" not in self.cache:
            return
        row_idx = len(self.cache["y"]) // 2
        col_idx = len(self.cache["x"]) // 2
        self.state["cut_row_idx"] = row_idx
        self.state["cut_col_idx"] = col_idx

        I_ort = self.cache["I_ort"]
        self.line_cut_x.set_ydata(I_ort[row_idx, :])
        self.line_cut_y.set_ydata(I_ort[:, col_idx])

        self.redraw_crosshair()
        self.canvas.draw_idle()

    def on_save(self):
        centers_x, centers_y, r_center = compute_centers(self.state["N_x"], self.state["N_y"], self.state["width"])
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

        mask_u = overlap_mask_pitch(X, Y, r_center, r_center, self.state["uniformity_side_length"])
        mask_c = overlap_mask_pitch(X, Y, r_center, r_center, self.state["crosstalk_side_length"])
        uniformity = np.std(I_ort[mask_u]) / np.mean(I_ort[mask_u])
        crosstalk = np.sum(I_neighbor[mask_c]) / np.sum(I_ort[mask_c])

        extent = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]

        # Aktuelle Fadenkreuz-Position (aus der interaktiven Ansicht) auf das
        # hochauflösende Grid übertragen, statt immer die Mitte zu nehmen
        if self.state["cut_row_idx"] is not None and "y" in self.cache:
            y_pos_m = self.cache["y"][self.state["cut_row_idx"]]
            x_pos_m = self.cache["x"][self.state["cut_col_idx"]]
        else:
            y_pos_m = 0.0
            x_pos_m = 0.0
        mid_y_idx = int(np.argmin(np.abs(y - y_pos_m)))
        mid_x_idx = int(np.argmin(np.abs(x - x_pos_m)))
        y_pos_um = y[mid_y_idx] * 1e6
        x_pos_um = x[mid_x_idx] * 1e6

        profile_label = "Airy" if self.state["use_airy"] else "Gauß"
        fig_save = plt.figure(figsize=(12, 9))
        fig_save.suptitle(
            f"win={self.state['win']:.3e} m, width={self.state['width']:.3e} Hz, Profil={profile_label}, "
            f"Uniformity={uniformity*100:.2f}%, Crosstalk={crosstalk*100:.3f}%",
            fontsize=12, fontweight="bold"
        )

        ax1 = fig_save.add_subplot(2, 2, 1)
        ax1.imshow(I_ort, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        ax1.scatter(centers_x * 1e6, centers_y * 1e6, c="white", edgecolors="black", s=15, zorder=5)
        ax1.axhline(y_pos_um, color="black", linewidth=0.8, zorder=6)
        ax1.axvline(x_pos_um, color="black", linewidth=0.8, zorder=6)
        side_u_um = self.state["uniformity_side_length"] * 1e6
        side_c_um = self.state["crosstalk_side_length"] * 1e6
        half_u = side_u_um / 2
        half_c = side_c_um / 2
        r_um = r_center * 1e6
        ax1.add_patch(Rectangle((r_um - half_u, r_um - half_u), 2 * half_u, 2 * half_u,
                                 edgecolor="cyan", facecolor="none", linewidth=2,
                                 label=f"Uniformity-Bereich ({side_u_um:.3f} µm)"))
        ax1.add_patch(Rectangle((r_um - half_c, r_um - half_c), 2 * half_c, 2 * half_c,
                                 edgecolor="red", facecolor="none", linewidth=2,
                                 label=f"Crosstalk-Bereich ({side_c_um:.3f} µm)"))
        ax1.set_xlabel("Ort $x$ (µm)")
        ax1.set_ylabel("Ort $y$ (µm)")
        ax1.legend(fontsize=8)

        ax2 = fig_save.add_subplot(2, 2, 2)
        ax2.imshow(I_neighbor, origin="lower", extent=extent, aspect="equal", cmap="viridis")
        ax2.set_xlabel("Ort $x$ (µm)")
        ax2.set_ylabel("Ort $y$ (µm)")
        ax2.set_title("Nachbarregionen")

        ax3 = fig_save.add_subplot(2, 2, 3)
        ax3.plot(x * 1e6, I_ort[mid_y_idx, :], "b-", linewidth=2)
        ax3.set_xlabel("Ort $x$ (µm)")
        ax3.set_ylabel("Intensität")
        ax3.set_title(f"Schnitt entlang x  (y = {y_pos_um:.3f} µm)")
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.05)

        ax4 = fig_save.add_subplot(2, 2, 4)
        ax4.plot(y * 1e6, I_ort[:, mid_x_idx], "g-", linewidth=2)
        ax4.set_xlabel("Ort $y$ (µm)")
        ax4.set_ylabel("Intensität")
        ax4.set_title(f"Schnitt entlang y  (x = {x_pos_um:.3f} µm)")
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1.05)

        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = out_dir / f"FlatMultiTone_GUI_{timestamp}.png"
        try:
            fig_save.savefig(out_file, dpi=150, bbox_inches="tight")
            self.label_save_status.setText(f"Gespeichert:\n{out_file}")
        except Exception as e:
            self.label_save_status.setText(f"Fehler beim Speichern: {e}")
            QMessageBox.warning(self, "Speichern fehlgeschlagen", str(e))
        finally:
            plt.close(fig_save)

    # --------------------------------------------------------
    # Drag & Drop im Hauptplot: Region-Handles ODER Fadenkreuz
    # (matplotlib-Events auf dem Qt-Canvas)
    # --------------------------------------------------------
    def on_button_press(self, event):
        if event.inaxes != self.ax_main or event.xdata is None or event.ydata is None:
            return

        # Priorität: liegt der Klick nah genug an einem Region-Handle, dieses greifen
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

        r_center_um = self.cache["r_center"] * 1e6
        dx = event.xdata - r_center_um
        dy = event.ydata - r_center_um
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
    window = FlatMultiToneWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()