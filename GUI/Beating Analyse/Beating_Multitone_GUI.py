"""
Beating Multitone GUI (PyQt5)
=============================
Time-resolved, COHERENT simulation of the multitone flat-top pattern.

Why a separate GUI?
-------------------
Multitone_Lens_GUI.py and Weighted_Multitone_Lens_GUI.py sum the INTENSITIES
of the individual tones - that is the time average. Physically the tones
superpose as FIELDS, each with its own AOD frequency, so the pattern beats at
the difference frequencies. This GUI shows the instantaneous intensity, the
beat spectrum, the static interference of frequency degenerate spots and what
all of that means for a camera image and for a Raman pulse.

Files
-----
    Beating_Multitone_GUI.py   this window: inputs, recompute(), drawing, saving
    kern/beating_physik.py     ALL physics and numerics - frequencies, geometry,
                               fields, beat orders, exact time statistics, pulse
                               area, atom weighting. No Qt. The formulae and
                               what the model leaves out are described at the
                               top of that file.
    kern/rb85_raman.py         Rb-85 Raman coefficients (ARC)
    kern/beating_profil.py     bridge to the Rabi calculation (Rabi_Rb85_GUI.py)
    one_lens_design.py         sub-window: single lens, target beating period
    camera_series.py           sub-window: camera frame series
    pulse_timing.py            sub-window: pulse area and trigger jitter
    power_budget.py            sub-window: power and intensity

Start values
------------
The GUI always opens with this working point:

    tones            3 x 4
    profile          Airy, factor 1.4830 (fixed) -> first zero ring 1.542 um
    waist            1.04 um after the lenses (1.287 mm before them)
    width            0.37 MHz, x and y coupled
    amplitudes       r_x = 0.97, r_y = 1.16
    optics           f1 = 75 mm, f2 = 750 mm, fLO = 52.88 mm
    light / AOD      795 nm, offset 100 MHz
    pulse            f_Rabi = 1 MHz, Omega ~ I, eta = 0

From it follow f_0 = width/6 = 61.67 kHz, T_0 = 16.22 us, a tone spacing of
1.169 um in x and 0.779 um in y, and one frequency degenerate corner pair
with a static contribution of 7.16 % of the maximum.
The pulse-area window opens with Delta = +50 GHz (blue) and 10 uW in the
profile.

Operation
---------
All parameters are typed in (no sliders except the time axis). Press
'Recompute' or tick 'Recompute automatically'. A click into the 2D map moves
the crosshair; the space-time map, the cut and I(t) refer to it.

Start:
    python Beating_Multitone_GUI.py
"""

import os
import sys
import datetime
from pathlib import Path as FilePath

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QGroupBox, QScrollArea, QSplitter, QComboBox, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer

_HERE = FilePath(__file__).resolve().parent

# All physics lives in kern/beating_physik.py. kern is a package next to this
# file, so the import resolves the same way for Python and for PyCharm.
from kern.beating_physik import (
    # constants
    fLO, theta_max, f_band, AIRY_FACTOR,
    # frequencies, geometry, amplitudes, fields
    conjugate_waist, compute_centers_and_freqs, amps_from_ratio,
    amp_spots_from_ratios, build_field_stack, compute_grid,
    # beat frequencies and phases
    unique_beat_frequencies, fundamental_beat_frequency, degenerate_groups,
    min_frames_per_period, resonance_check, schroeder_phases,
    kitayoshi_phases, spot_phases_from_tones, quadrature_penalty, crest_factor,
    rf_voltage_ratios,
    # time evolution and statistics
    intensity_cube, boxcar_in_time, beat_orders, time_stats_exact,
    uniformity_of, VariationObjective, UniformitySeries, PulseArea,
)

try:
    import one_lens_design
except Exception:      # one_lens_design.py sits next to this file
    one_lens_design = None

try:
    import camera_series
except Exception:      # camera_series.py sits next to this file
    camera_series = None

try:
    import pulse_timing
except Exception:      # pulse_timing.py sits next to this file
    pulse_timing = None

try:
    import power_budget
except Exception:      # power_budget.py sits next to this file
    power_budget = None


# Where images and their parameter files go. GUI/Bilder is the agreed place;
# the path relative to this file is the fallback, so a clone somewhere else
# still writes next to itself instead of into a stranger's Desktop.
OUT_DIR_CANDIDATES = (
    [FilePath(r"C:\Users\Legion\OneDrive\Desktop\Multitone_Simulation\GUI\Bilder")]
    if os.name == "nt" else []) + [
    _HERE.parent / "Bilder",
    _HERE / "Bilder",
    FilePath(r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\PythonCode\Multitone_FlatTop"),
]


def short_name(path, keep=34):
    """Shorten a file name for display, with an ellipsis in the middle.

    A saved file name is long and contains no spaces, so it cannot be wrapped;
    the full path goes into the tooltip instead."""
    name = str(getattr(path, "name", path))
    if len(name) <= keep:
        return name
    head = keep // 2 - 2
    return name[:head] + "..." + name[-(keep - head - 3):]


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
# Main window
# ============================================================
class BeatingMultitoneWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beating Multitone GUI - coherent time evolution")
        self.resize(1600, 950)

        self.state = {
            "N_x": 3,
            "N_y": 4,
            # Start values of the working point - the GUI always opens like this.
            "use_airy": True,
            "airy_factor": AIRY_FACTOR,  # fixed, see kern/beating_physik.py
            "win": 1.04e-6,        # m, waist AFTER the lenses
            "win_in": None,        # m, waist BEFORE the lenses
            "win_mode": "output",
            "width_x": 0.37e6,     # Hz, frequency span of the x tones
            "width_y": 0.37e6,     # Hz, frequency span of the y tones
            "link_width": True,    # width_y follows width_x
            "r_x": 0.97,
            "r_y": 1.16,
            "lambda_opt": 795e-9,  # m
            "offset": 100e6,       # Hz
            "f1": 75e-3,
            "f2": 750e-3,
            # --- single lens build (lab setup for the camera image) ---
            "one_lens": False,     # True: no telescope, no fLO, only f_single
            "f_single": 45e-3,     # m, the one lens behind the AOD
            "win_in_single": 1.75e-3,  # m, waist in front of that lens
            "target_period": 100e-6,   # s, wanted beating period (design dialog)
            "t_exp": 0.0,          # s, camera exposure; 0 = instantaneous
            "n_max": 20,           # search limit of the design dialog
            "heavy_analysis": False,   # force pulse/spectrum at many spots
            "grid_n": 200,
            "n_periods": 3,
            "frames_per_period": 60,
            "phase_x": np.zeros(3),      # rad, phase per x tone
            "phase_y": np.zeros(4),      # rad, phase per y tone
            # Phases are always TONE phases: a spot (n,m) carries
            # phi_x(n) + phi_y(m). Free spot phases were a cross-check and
            # have been removed - they could not be driven anyway and did
            # not remove the modulation either.
            "f_rabi": 1.0e6,             # Hz, Rabi frequency Omega/2pi
            "pulse_t0": 0.0,             # s, start time of the pulse in the beat cycle
            "rabi_law": "I",             # 'I' (two-photon Raman) | 'sqrtI'
            "eta_ls": 0.0,               # differential light shift / Rabi frequency
            "auto_update": False,
        }
        self.state["win_in"] = conjugate_waist(
            self.state["win"], self.state["f1"], self.state["f2"],
            self.state["lambda_opt"], self.state["one_lens"], self.state["f_single"])

        self.cache = {}
        self._panel_cbar = None
        self._opt_note = ""
        self._rabi_cache = None
        self._art = {}          # persistent drawing objects for the fast path
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

    def _tame_info_labels(self):
        """Keep long, unbreakable texts from stretching the control panel.

        A saved file name has no spaces, so word wrap cannot break it and the
        label reports a very wide sizeHint - which the layout honours by
        widening the whole panel and squashing every button. Ignoring the
        horizontal sizeHint makes the label adapt to the panel instead of the
        other way round.
        """
        for lab in self.findChildren(QLabel):
            if lab.wordWrap():
                lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                lab.setMinimumWidth(1)

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
        pl.addWidget(self._group_pulse())
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

    # -- small helpers for the input fields ------------------
        self._tame_info_labels()

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
        g = QGroupBox("Tones")
        lay = QGridLayout(g)
        self.sp_nx = self._ispin(self.state["N_x"], 1, 64)
        self.sp_ny = self._ispin(self.state["N_y"], 1, 64)
        lay.addWidget(QLabel("N_x"), 0, 0); lay.addWidget(self.sp_nx, 0, 1)
        lay.addWidget(QLabel("N_y"), 1, 0); lay.addWidget(self.sp_ny, 1, 1)
        self.lbl_freqs = QLabel("-")
        self.lbl_freqs.setWordWrap(True)
        self.lbl_freqs.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_freqs, 2, 0, 1, 2)
        return g

    def _group_profile(self):
        g = QGroupBox("Beam profile")
        lay = QGridLayout(g)
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems(["Gauss", "Airy"])
        self.cmb_profile.setCurrentIndex(1 if self.state["use_airy"] else 0)
        self.cmb_profile.currentIndexChanged.connect(self._on_param_changed)
        self.lbl_airy = QLabel(f"{AIRY_FACTOR:.4f}  (fixed)")
        self.lbl_airy.setToolTip("first_zero_radius = factor * waist.\n"
                                 "1.4830 = same 1/e^2 width as the Gauss.\n"
                                 "Fixed: AIRY_FACTOR in kern/beating_physik.py.")
        lay.addWidget(QLabel("Profile"), 0, 0); lay.addWidget(self.cmb_profile, 0, 1)
        lay.addWidget(QLabel("Airy factor"), 1, 0); lay.addWidget(self.lbl_airy, 1, 1)
        return g

    def _group_beam(self):
        g = QGroupBox("Beam / optics")
        lay = QGridLayout(g)
        self.cmb_winmode = QComboBox()
        self.cmb_winmode.addItems(["Waist after lens (um)", "Waist before lens (mm)"])
        self.cmb_winmode.currentIndexChanged.connect(self._on_winmode_changed)
        self.sp_win = self._dspin(self.state["win"] * 1e6, 0.05, 50.0, 4, 0.01, "um")
        self.sp_win_in = self._dspin(self.state["win_in"] * 1e3, 0.01, 50.0, 4, 0.01, "mm")
        self.sp_width = self._dspin(self.state["width_x"] * 1e-6, 0.0, 36.0, 5, 0.01, "MHz")
        self.sp_width_y = self._dspin(self.state["width_y"] * 1e-6, 0.0, 36.0, 5, 0.01, "MHz")
        self.cb_link_width = QCheckBox("width_y = width_x")
        self.cb_link_width.setChecked(self.state["link_width"])
        self.cb_link_width.stateChanged.connect(self._on_link_width_changed)
        self.cb_link_width.setToolTip(
            "The other GUIs set both widths equal. That is exactly what\n"
            "produces frequency degeneracies: at 3x4 the spots (n,m) = (0,3)\n"
            "and (2,0) carry exactly the same total frequency and interfere\n"
            "statically. Separate widths lift this - but change the spot\n"
            "spacing in y.")
        self.sp_lambda = self._dspin(self.state["lambda_opt"] * 1e9, 200.0, 2000.0, 2, 1.0, "nm")
        self.sp_offset = self._dspin(self.state["offset"] * 1e-6, 0.0, 500.0, 4, 1.0, "MHz")
        self.sp_f1 = self._dspin(self.state["f1"] * 1e3, 1.0, 2000.0, 2, 5.0, "mm")
        self.sp_f2 = self._dspin(self.state["f2"] * 1e3, 1.0, 2000.0, 2, 5.0, "mm")

        self.cb_one_lens = QCheckBox("Use one lens")
        self.cb_one_lens.setChecked(self.state["one_lens"])
        self.cb_one_lens.setToolTip(
            "The lab setup for the camera image has no telescope and no fLO:\n"
            "behind the AOD stands ONE lens. Then\n"
            "    r(f) = f_lens * tan(theta(f))     w_0 = lam*f_lens/(pi*w_in)\n"
            "instead of the (f1*fLO/f2) chain. The AOD itself is unchanged,\n"
            "theta(f) = theta_max*(f-offset)/f_band stays as it is.\n\n"
            "f1 and f2 are ignored while this is ticked.")
        self.cb_one_lens.stateChanged.connect(self._on_one_lens_changed)
        self.sp_fsingle = self._dspin(self.state["f_single"] * 1e3, 1.0, 2000.0, 3, 1.0, "mm")
        self.btn_one_lens = QPushButton("Design for a target beating period ...")
        self.btn_one_lens.setToolTip(
            "Opens the design window: focal length, waist in front of the\n"
            "lens, wanted beating period and camera exposure go in, tone\n"
            "numbers and frequency widths that hit that period EXACTLY come\n"
            "out.")
        self.btn_one_lens.clicked.connect(self._open_one_lens_dialog)

        rows = [("", self.cb_one_lens), ("f (single lens)", self.sp_fsingle),
                ("", self.btn_one_lens),
                ("Mode", self.cmb_winmode), ("waist", self.sp_win),
                ("waist_in", self.sp_win_in),
                ("width x", self.sp_width), ("width y", self.sp_width_y),
                ("", self.cb_link_width),
                ("Wavelength", self.sp_lambda), ("Offset f0", self.sp_offset),
                ("f1", self.sp_f1), ("f2", self.sp_f2)]
        for i, (name, w) in enumerate(rows):
            lay.addWidget(QLabel(name), i, 0)
            lay.addWidget(w, i, 1)
        hint = QLabel("Wavelength and offset change the geometry, but NO\n"
                      "beat frequency - |E|^2 contains only differences\n"
                      "of the tone frequencies, and a constant offset\n"
                      "cancels out of every difference.")
        hint.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(hint, len(rows), 0, 1, 2)
        self._sync_winmode_enabled()
        self._sync_width_enabled()
        self._sync_one_lens_enabled()
        return g

    def _group_amps(self):
        g = QGroupBox("Amplitudes (outer/inner)")
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

    def _group_pulse(self):
        g = QGroupBox("Pulsed operation")
        lay = QGridLayout(g)
        self.sp_frabi = QDoubleSpinBox()
        self.sp_frabi.setRange(0.001, 100.0); self.sp_frabi.setDecimals(4)
        self.sp_frabi.setSingleStep(0.05)
        self.sp_frabi.setValue(self.state["f_rabi"] * 1e-6)
        self.sp_frabi.setSuffix(" MHz"); self.sp_frabi.setKeyboardTracking(False)
        self.sp_frabi.setToolTip("Rabi frequency Omega/2pi. From it follows the "
                                 "pi pulse duration T = 1/(2 f_Rabi).")
        self.sp_frabi.valueChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("f_Rabi"), 0, 0); lay.addWidget(self.sp_frabi, 0, 1)

        self.sp_t0 = QDoubleSpinBox()
        self.sp_t0.setRange(0.0, 1000.0); self.sp_t0.setDecimals(3)
        self.sp_t0.setSingleStep(0.1); self.sp_t0.setValue(0.0)
        self.sp_t0.setSuffix(" us"); self.sp_t0.setKeyboardTracking(False)
        self.sp_t0.setToolTip(
            "Start time of the pulse WITHIN the beat cycle.\n"
            "Without a fixed timing the pulse area varies from shot to shot -\n"
            "at the working point by up to a factor of 73. The pulse must\n"
            "therefore be triggered on the AWG waveform.")
        self.sp_t0.valueChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("Pulse start t_0"), 1, 0); lay.addWidget(self.sp_t0, 1, 1)

        self.cmb_rabi_law = QComboBox()
        self.cmb_rabi_law.addItems(["Omega ~ I (two-photon Raman)",
                                    "Omega ~ sqrt(I) (single photon / one branch)"])
        self.cmb_rabi_law.setToolTip(
            "If both Raman branches come from this profile, Omega ~ I.\n"
            "If this profile is only ONE branch, Omega ~ E ~ sqrt(I).\n"
            "That changes the weighting and hence the uniformity.")
        self.cmb_rabi_law.currentIndexChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("Coupling"), 2, 0); lay.addWidget(self.cmb_rabi_law, 2, 1)

        self.sp_eta = QDoubleSpinBox()
        self.sp_eta.setRange(0.0, 5.0); self.sp_eta.setDecimals(3)
        self.sp_eta.setSingleStep(0.05); self.sp_eta.setValue(0.0)
        self.sp_eta.setKeyboardTracking(False)
        self.sp_eta.setToolTip(
            "Differential light shift divided by the Rabi frequency.\n\n"
            "Omega ~ I: both scale with the SAME intensity, so their ratio is\n"
            "constant in space and time. The light shift adds no extra spatial\n"
            "non-uniformity - it caps the contrast at 1/(1+eta^2) and rescales\n"
            "the Rabi frequency by sqrt(1+eta^2). Exact, closed form.\n\n"
            "Omega ~ sqrt(I): the shift follows I, the Rabi frequency sqrt(I).\n"
            "eta is then the part of delta/Omega from THIS (multitone) leg at\n"
            "the calibration intensity - half the total for equal legs; the\n"
            "clean leg's constant shift counts as compensated. The excitation\n"
            "is propagated numerically.\n\n"
            "Order of magnitude of the total (D1, sigma+/sigma+, equal legs):\n"
            "eta ~ omega_HF/Delta, about 0.06 at +50 GHz.\n"
            "eta = 0 means the light shift is NEGLECTED. A fixed detuning only\n"
            "compensates the mean shift; with beating the shift varies in time.")
        self.sp_eta.valueChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("light shift eta"), 3, 0); lay.addWidget(self.sp_eta, 3, 1)

        self.btn_snap_flat = QPushButton("Move t_0 to a flat point of the area")
        self.btn_snap_flat.setToolTip(
            "Puts the pulse start on a stationary point of the pulse area,\n"
            "where dA/dt_0 = 0 and timing jitter therefore only enters in\n"
            "second order.\n\n"
            "This is worth far more than the last percent of uniformity: at\n"
            "0.1 MHz the tolerance for a 1 % area error rises from 44 ns to\n"
            "12.8 us, while the uniformity only goes from 18.3 to 19.9 %.")
        self.btn_snap_flat.clicked.connect(self._on_snap_flat)
        lay.addWidget(self.btn_snap_flat, 4, 0, 1, 2)

        self.btn_opt_pulse = QPushButton("Optimise phases and t_0 for the pulse area")
        self.btn_opt_pulse.setToolTip(
            "Minimises the uniformity of the accumulated Rabi area in the\n"
            "selected target region - the quantity that really counts for\n"
            "pulsed driving. Optimises tone phases AND pulse timing jointly.\n"
            "Takes about half a minute.")
        self.btn_opt_pulse.clicked.connect(self._on_optimize_pulse)
        lay.addWidget(self.btn_opt_pulse, 5, 0, 1, 2)

        self.lbl_pulse = QLabel("-")
        self.lbl_pulse.setWordWrap(True)
        self.lbl_pulse.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_pulse, 6, 0, 1, 2)
        return g

    def _group_time(self):
        g = QGroupBox("Time axis")
        lay = QGridLayout(g)
        self.sp_periods = self._ispin(self.state["n_periods"], 1, 50)
        self.sp_fpp = self._ispin(self.state["frames_per_period"], 8, 2000)
        self.sp_grid = self._ispin(self.state["grid_n"], 60, 600)
        lay.addWidget(QLabel("Periods"), 0, 0); lay.addWidget(self.sp_periods, 0, 1)
        self.sp_fpp.setToolTip(
            "Must be larger than 2 * (highest beat frequency / f_0),\n"
            "otherwise the beating is sampled incorrectly (aliasing) and\n"
            "even the time average is wrong. The GUI warns when there are\n"
            "too few and states the required number.")
        lay.addWidget(QLabel("Frames/period"), 1, 0); lay.addWidget(self.sp_fpp, 1, 1)
        lay.addWidget(QLabel("Grid resolution"), 2, 0); lay.addWidget(self.sp_grid, 2, 1)
        self.sp_texp = QDoubleSpinBox()
        self.sp_texp.setRange(0.0, 100000.0); self.sp_texp.setDecimals(3)
        self.sp_texp.setSingleStep(5.0); self.sp_texp.setSuffix(" us")
        self.sp_texp.setValue(self.state["t_exp"] * 1e6)
        self.sp_texp.setKeyboardTracking(False)
        self.sp_texp.setToolTip(
            "Camera exposure. 0 = instantaneous intensity, as before.\n\n"
            "A finite exposure is a boxcar in time: the Fourier coefficient\n"
            "of beat order d is multiplied by sinc(d*f_0*t_exp). Slow beats\n"
            "survive, fast ones are averaged away - at t_exp = 20 us and\n"
            "f_0 = 10 kHz the fundamental keeps 94 %, while 120 kHz is left\n"
            "with 8 %. All images and the sigma_t map then show what the\n"
            "CAMERA records; the pulse and Rabi analysis stays instantaneous,\n"
            "because the atom does not integrate.")
        self.sp_texp.valueChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("Camera exposure"), 3, 0); lay.addWidget(self.sp_texp, 3, 1)

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
        self.slider_speed.setRange(1, 60)      # frames per second
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

        row_nu = QHBoxLayout()
        row_nu.addWidget(QLabel("Trap frequency nu_r"))
        self.sp_nu_r = QDoubleSpinBox()
        self.sp_nu_r.setRange(0.1, 5000.0); self.sp_nu_r.setDecimals(1)
        self.sp_nu_r.setValue(60.4); self.sp_nu_r.setSuffix(" kHz")
        self.sp_nu_r.setKeyboardTracking(False)
        self.sp_nu_r.setToolTip(
            "Only for checking whether a beat line falls on nu_r or\n"
            "2*nu_r. The spectrum is discrete - if no line sits there,\n"
            "the trap receives hardly any power despite full modulation.\n\n"
            "That holds for illumination over MANY trap periods. A single\n"
            "pulse shorter than 1/nu_r is spectrally broad; there the line\n"
            "positions do not matter, only the kick of the pulse.")
        self.sp_nu_r.valueChanged.connect(self._on_param_changed)
        row_nu.addWidget(self.sp_nu_r)
        holder_nu = QWidget(); holder_nu.setLayout(row_nu)
        lay.addWidget(holder_nu, 8, 0, 1, 2)

        self.lbl_res = QLabel("")
        self.lbl_res.setWordWrap(True)
        self.lbl_res.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_res, 9, 0, 1, 2)
        return g

    def _group_phases(self):
        g = QGroupBox("Tone phases")
        outer = QVBoxLayout(g)

        btns = QHBoxLayout()
        for label, fn in (("0", "zero"), ("Schroeder", "schroeder"),
                          ("Kitayoshi", "kitayoshi"), ("randomise", "random")):
            b = QPushButton(label)
            b.clicked.connect(lambda _, k=fn: self._apply_phase_preset(k))
            btns.addWidget(b)
        holder = QWidget(); holder.setLayout(btns)
        outer.addWidget(holder)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target region"))
        self.cmb_opt_region = QComboBox()
        self.cmb_opt_region.addItems(["Plateau (<I> > 50 % max)", "Spot centres",
                                      "Circle around the centre"])
        self.cmb_opt_region.setCurrentIndex(2)   # default: the 2 um circle
        row.addWidget(self.cmb_opt_region)
        self.sp_opt_radius = QDoubleSpinBox()
        self.sp_opt_radius.setRange(0.1, 20.0); self.sp_opt_radius.setDecimals(2)
        self.sp_opt_radius.setSingleStep(0.1); self.sp_opt_radius.setValue(2.0)
        self.sp_opt_radius.setSuffix(" um")
        row.addWidget(self.sp_opt_radius)
        holder2 = QWidget(); holder2.setLayout(row)
        outer.addWidget(holder2)

        self.cb_quad = QCheckBox("keep degenerate pairs in quadrature")
        self.cb_quad.setChecked(True)
        self.cb_quad.setToolTip(
            "Frequency degenerate spots have a cross term at 0 Hz that never\n"
            "averages away - the static distortion the incoherent GUIs do\n"
            "not see.\n\n"
            "At a phase difference of 90 degrees this term is EXACTLY zero.\n"
            "Then - and only then - the time average is exactly the\n"
            "incoherent sum. Costs almost nothing: the uniformity at the\n"
            "optimum goes from 23.5 to 24.3 percent.")
        self.cb_quad.stateChanged.connect(self._on_param_changed)
        outer.addWidget(self.cb_quad)

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
        """Creates the input fields anew when N_x or N_y changes.
        Existing values are carried over as far as possible."""
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
            self.phase_grid.addWidget(QLabel(name + " (degrees)"), 0, col)
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

    def _apply_phase_preset(self, kind):
        N_x, N_y = self.state["N_x"], self.state["N_y"]
        if kind == "zero":
            px, py = np.zeros(N_x), np.zeros(N_y)
        elif kind == "schroeder":
            px, py = schroeder_phases(N_x), schroeder_phases(N_y)
        elif kind == "kitayoshi":
            px, py = kitayoshi_phases(N_x), kitayoshi_phases(N_y)
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
        """Region that the phase optimisation targets."""
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

    def _pulse_setup(self):
        """Common setup for everything that has to do with the pulse."""
        s = self.state
        cxs, cys, f_spots, _, _, _, _ = compute_centers_and_freqs(
            s["N_x"], s["N_y"], s["width_x"], s["width_y"], s["f1"], s["f2"],
            s["offset"], s["one_lens"], s["f_single"])
        f0 = fundamental_beat_frequency(f_spots)
        amp = amp_spots_from_ratios(s["r_x"], s["r_y"], s["N_x"], s["N_y"])
        win_eff = s["win"] * (s["airy_factor"] if s["use_airy"] else 1.0)
        xg, yg, Xg, Yg = compute_grid(cxs, cys, win_eff, 120)
        Fg = build_field_stack(Xg, Yg, cxs, cys, amp, s["win"], s["use_airy"],
                               s["airy_factor"])
        k = beat_orders(f_spots, f0) if f0 > 0 else np.zeros(len(f_spots), int)
        mean0, _ = time_stats_exact(Fg, k, np.zeros(len(f_spots)))
        mask = self._target_mask(mean0, Xg, Yg, cxs, cys, xg, yg)
        return dict(f_spots=f_spots, f0=f0, F=Fg, k=k, mask=mask, mean0=mean0,
                    T0=(1.0 / f0 if f0 > 0 else float("nan")))

    def _on_snap_flat(self):
        """Move the pulse start to the flattest useful point of the area curve."""
        pz = self.cache.get("pulse")
        if pz is None or "t0_flat" not in pz:
            self.lbl_status.setText("No pulse data available.")
            return
        self.sp_t0.blockSignals(True)
        self.sp_t0.setValue(pz["t0_flat"] * 1e6)
        self.sp_t0.blockSignals(False)
        self.state["pulse_t0"] = pz["t0_flat"]
        self._opt_note = (f"t_0 moved to the flat point at {pz['t0_flat'] * 1e6:.3f} us "
                          f"(U there {pz['u_flat'] * 100:.1f} %)")
        self.recompute()

    def _on_optimize_pulse(self):
        """Searches for tone phases AND pulse timing with the most uniform
        pulse area.

        This is the only optimiser left. The earlier objectives (broadband
        time variation, peak intensity, quiet window) have been removed:
        they all ended up at nearly the same phase set, and for pulsed
        operation the pulse area is the relevant quantity anyway."""
        from scipy.optimize import minimize
        self._read_widgets()
        s = self.state
        N_x, N_y = s["N_x"], s["N_y"]
        n_free = max(0, N_x - 1) + max(0, N_y - 1)
        st = self._pulse_setup()
        if st["f0"] <= 0 or n_free == 0:
            self.lbl_status.setText("No beating or no free phases.")
            return
        T0 = st["T0"]
        t_p = 1.0 / (2.0 * s["f_rabi"])
        pa = PulseArea(st["F"], st["k"], st["f0"], st["mask"], law=s["rabi_law"])
        degen_g = degenerate_groups(st["f_spots"], st["f0"])

        def cost(v):
            px = np.concatenate(([0.0], v[:N_x - 1])) if N_x > 1 else np.zeros(1)
            py = np.concatenate(([0.0], v[N_x - 1:n_free])) if N_y > 1 else np.zeros(1)
            ph = spot_phases_from_tones(px, py, N_x, N_y)
            u = pa.uniformity(ph, v[-1] % T0, t_p, st["f_spots"])
            if not np.isfinite(u):
                return 1e6
            pen = (2.0 * quadrature_penalty(ph, degen_g)
                   if self.cb_quad.isChecked() else 0.0)
            return u + pen

        self.lbl_status.setText("searching phases and pulse timing ...")
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        QApplication.processEvents()
        rng = np.random.default_rng(3)
        n_start = 35 if s["rabi_law"] == "I" else 12
        best_f, best_v = 1e18, None
        for n in range(n_start):
            v0 = np.concatenate((rng.uniform(0, 2 * np.pi, n_free), [rng.uniform(0, T0)]))
            r = minimize(cost, v0, method="Nelder-Mead",
                         options=dict(maxiter=2500, xatol=1e-7, fatol=1e-10))
            if r.fun < best_f:
                best_f, best_v = float(r.fun), r.x
            if n % 10 == 0:
                QApplication.processEvents()
        px = np.concatenate(([0.0], best_v[:N_x - 1])) if N_x > 1 else np.zeros(1)
        py = np.concatenate(([0.0], best_v[N_x - 1:n_free])) if N_y > 1 else np.zeros(1)
        self._write_phase_fields(px, py)
        t0 = float(best_v[-1] % T0)
        self.sp_t0.blockSignals(True); self.sp_t0.setValue(t0 * 1e6); self.sp_t0.blockSignals(False)
        self.state["pulse_t0"] = t0
        self._opt_note = (f"Pulse area: U = {best_f * 100:.1f} % at t_0 = {t0 * 1e6:.3f} us "
                          f"(T_pi = {t_p * 1e6:.3f} us)")
        self.recompute()

    def _group_actions(self):
        g = QGroupBox("Actions")
        lay = QVBoxLayout(g)
        self.cb_auto = QCheckBox("Recompute automatically")
        self.cb_auto.setChecked(self.state["auto_update"])
        self.cb_auto.stateChanged.connect(self._on_auto_changed)
        lay.addWidget(self.cb_auto)
        self.btn_update = QPushButton("Recompute")
        self.btn_update.clicked.connect(lambda: self.recompute())
        lay.addWidget(self.btn_update)
        lay.addWidget(QLabel("Panel top right"))
        self.cmb_panel = QComboBox()
        self.cmb_panel.addItems([
            "Space-time map I(x, t)",
            "Enhancement n_eff = I_max / <I>",
            "Modulation depth (I_max-I_min)/(I_max+I_min)",
            "temporal variation sigma_t / <I>",
            "Spectrum of the beating",
            "Uniformity U(t) of the three regions",
            "Pulse area: U over the pulse start",
            "Pulse area A(t_0) and flat points",
            "Rabi oscillation",
        ])
        self.cmb_panel.setToolTip(
            "n_eff is the number of tones effectively overlapping at the\n"
            "respective position: 1 where a single spot dominates, up to the\n"
            "number of spots in the plateau. It is at the same time the\n"
            "factor by which the instantaneous intensity exceeds the time\n"
            "average on rephasing.")
        self.cmb_panel.currentIndexChanged.connect(lambda _: self.draw_frame(full=True))
        self.cb_fastdraw = QCheckBox("Fast drawing during playback")
        self.cb_fastdraw.setChecked(True)
        self.cb_fastdraw.setToolTip(
            "When stepping forward only the data of the existing drawing\n"
            "objects are exchanged instead of rebuilding all four axes -\n"
            "about twice as fast.\n\n"
            "If the display behaves strangely: uncheck, then every image is\n"
            "redrawn completely.")
        self.cb_fastdraw.stateChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cb_fastdraw)

        self.cb_heavy = QCheckBox("pulse / spectrum also at many spots")
        self.cb_heavy.setChecked(self.state["heavy_analysis"])
        self.cb_heavy.setToolTip(
            "The pulse-area and spectrum analysis runs over ALL spot pairs and\n"
            "therefore costs O(S^2) - fine at 12 spots, minutes at 180. Above\n"
            "64 spots it is skipped unless this is ticked. Images, uniformity\n"
            "and the sigma_t map are unaffected, they use the fast path.")
        self.cb_heavy.stateChanged.connect(self._on_param_changed)
        lay.addWidget(self.cb_heavy)

        self.cb_live = QCheckBox("record U(t) live")
        self.cb_live.setChecked(True)
        self.cb_live.setToolTip(
            "In the 'Uniformity U(t)' panel the curve is only drawn up to the\n"
            "current instant and grows along with the animation; the rest of\n"
            "the trace stands pale behind it. The numbers in the title are\n"
            "the instantaneous values.")
        self.cb_live.stateChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cb_live)
        lay.addWidget(self.cmb_panel)
        lay.addWidget(QLabel("Colour scale"))
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems([
            "fixed: 99.5 percentile over time",
            "fixed: maximum over time",
            "fixed: maximum of the time average",
            "per frame",
        ])
        self.cmb_scale.setToolTip(
            "With all tone phases at 0, all tones rephase once per\n"
            "fundamental period into a short pulse that exceeds the time\n"
            "average by a large factor. A fixed scale on THIS maximum leaves\n"
            "all remaining frames almost black - which is why the percentile\n"
            "is the default. 'per frame' shows every image fully scaled, but\n"
            "makes frames incomparable to each other.")
        self.cmb_scale.currentIndexChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cmb_scale)
        self.btn_power = QPushButton("Power / intensity ...")
        self.btn_power.setToolTip(
            "Opens the power budget: from an assumed Rabi frequency (or from\n"
            "an available power, the other way round) it gives the intensity,\n"
            "the power in the profile, per spot and per RF tone, plus the\n"
            "photon scattering per pulse.\n\n"
            "Atomic coefficients come from kern/rb85_raman.py.")
        self.btn_power.clicked.connect(self._open_power_budget)
        lay.addWidget(self.btn_power)
        self.btn_pulse = QPushButton("Pulse area / trigger jitter ...")
        self.btn_pulse.setToolTip(
            "Opens a window for the pulsed case: a rectangular pulse of a\n"
            "given length is triggered onto the time window of highest\n"
            "intensity, its Rabi area is computed, and the area is scanned\n"
            "over the trigger error.\n\n"
            "A pulse is the same boxcar in time as a camera exposure, only\n"
            "much shorter - at 1 us it damps 120 kHz only to 0.98, so it\n"
            "samples the beating instead of averaging it away.")
        self.btn_pulse.clicked.connect(self._open_pulse_timing)
        lay.addWidget(self.btn_pulse)
        self.btn_camera = QPushButton("Camera frame series ...")
        self.btn_camera.setToolTip(
            "Opens a window with the images a camera really records when the\n"
            "trigger delay is stepped over one beat period: top row the raw\n"
            "frames, bottom row the deviation from the time average.\n\n"
            "The window stays open. Change the tone phases here, press\n"
            "'Recompute', then 'Neu zeichnen' there - that is how to compare\n"
            "phase sets by what the camera would show.")
        self.btn_camera.clicked.connect(self._open_camera_series)
        lay.addWidget(self.btn_camera)
        self.btn_save = QPushButton("Save view as PDF")
        self.btn_save.clicked.connect(self._on_save_clicked)
        lay.addWidget(self.btn_save)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_status)
        return g

    # --------------------------------------------------------
    # Inputs -> state
    # --------------------------------------------------------
    def _sync_winmode_enabled(self):
        out = self.state["win_mode"] == "output"
        self.sp_win.setEnabled(out)
        self.sp_win_in.setEnabled(not out)

    def _sync_one_lens_enabled(self):
        """f1/f2 belong to the telescope build, f_single to the other one -
        exactly one pair is meaningful at a time."""
        one = self.state["one_lens"]
        self.sp_fsingle.setEnabled(one)
        self.sp_f1.setEnabled(not one)
        self.sp_f2.setEnabled(not one)

    def _on_one_lens_changed(self, _):
        newly_on = self.cb_one_lens.isChecked() and not self.state["one_lens"]
        self.state["one_lens"] = self.cb_one_lens.isChecked()
        self._sync_one_lens_enabled()
        if newly_on:
            # Switching the build changes the geometry completely; the design
            # window is the only sensible next step, so it opens by itself.
            if self._open_one_lens_dialog():
                return
        self._on_param_changed()

    def _open_one_lens_dialog(self):
        """Design window: target beating period -> tone numbers and widths.

        Returns True when a parameter set was taken over."""
        if one_lens_design is None:
            QMessageBox.warning(
                self, "Module missing",
                "one_lens_design.py was not found next to this file.")
            return False
        self._read_widgets()
        dlg = one_lens_design.OneLensDesignDialog(self, self.state)
        if dlg.exec_() != dlg.Accepted or not dlg.result_candidate:
            return False
        r = dlg.result_candidate
        s = self.state
        s["one_lens"] = True
        s["f_single"] = r["f_single"]
        s["target_period"] = r["target_period"]
        s["n_max"] = r["n_max"]
        s["t_exp"] = r["t_exp"]
        s["win_mode"] = "input"          # in front of the lens is what is known
        s["win_in"] = r["win_in"]
        s["win_in_single"] = r["win_in"]
        blocked = [self.cb_one_lens, self.sp_fsingle, self.cmb_winmode,
                   self.sp_win_in, self.sp_nx, self.sp_ny, self.sp_width,
                   self.sp_width_y, self.cb_link_width, self.sp_texp]
        for w in blocked:
            w.blockSignals(True)
        try:
            self.cb_one_lens.setChecked(True)
            self.sp_fsingle.setValue(r["f_single"] * 1e3)
            self.cmb_winmode.setCurrentIndex(1)
            self.sp_win_in.setValue(r["win_in"] * 1e3)
            self.sp_nx.setValue(r["N_x"])
            self.sp_ny.setValue(r["N_y"])
            self.cb_link_width.setChecked(bool(r["link_width"]))
            self.sp_width.setValue(r["width_x"] * 1e-6)
            self.sp_width_y.setValue(r["width_y"] * 1e-6)
            self.sp_texp.setValue(r["t_exp"] * 1e6)
        finally:
            for w in blocked:
                w.blockSignals(False)
        self._sync_one_lens_enabled()
        self._sync_winmode_enabled()
        self._sync_width_enabled()
        self._rebuild_phase_fields()
        # Time sampling. The highest beat order is
        #     K = (N_x-1)*k_x + (N_y-1)*k_y,   k = df * T_target,
        # Nyquist wants more than 2K samples per fundamental period. With a
        # hundred and more tones that is a lot of frames, so the window is
        # cut back to a single period and the grid is coarsened - otherwise
        # the cube alone would be hundreds of MB.
        k_x = int(round(r["width_x"] / max(r["N_x"] - 1, 1) * r["target_period"]))
        k_y = int(round(r["width_y"] / max(r["N_y"] - 1, 1) * r["target_period"]))
        need = 2 * ((r["N_x"] - 1) * k_x + (r["N_y"] - 1) * k_y) + 1
        n_spots = r["N_x"] * r["N_y"]
        for w, val in ((self.sp_fpp, int(min(self.sp_fpp.maximum(), max(60, need)))),
                       (self.sp_periods, 1 if need > 200 else self.sp_periods.value()),
                       (self.sp_grid, min(self.sp_grid.value(), 140)
                        if n_spots > 100 else self.sp_grid.value())):
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self.recompute()
        return True

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
        s["airy_factor"] = AIRY_FACTOR
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
        s["one_lens"] = self.cb_one_lens.isChecked()
        s["f_single"] = self.sp_fsingle.value() * 1e-3
        s["t_exp"] = self.sp_texp.value() * 1e-6
        s["heavy_analysis"] = self.cb_heavy.isChecked()
        s["r_x"] = self.sp_rx.value()
        s["r_y"] = self.sp_ry.value()
        s["grid_n"] = self.sp_grid.value()
        s["n_periods"] = self.sp_periods.value()
        s["frames_per_period"] = self.sp_fpp.value()
        if len(self.phase_spins_x) != s["N_x"] or len(self.phase_spins_y) != s["N_y"]:
            self._rebuild_phase_fields()
        s["f_rabi"] = self.sp_frabi.value() * 1e6
        s["pulse_t0"] = self.sp_t0.value() * 1e-6
        s["rabi_law"] = "I" if self.cmb_rabi_law.currentIndex() == 0 else "sqrtI"
        s["eta_ls"] = self.sp_eta.value()
        s["phase_x"] = np.radians([sp.value() for sp in self.phase_spins_x])
        s["phase_y"] = np.radians([sp.value() for sp in self.phase_spins_y])

        # Waist: whichever quantity is active is the control variable, the
        # other one is updated accordingly and shown in its field.
        if s["win_mode"] == "output":
            s["win"] = self.sp_win.value() * 1e-6
            s["win_in"] = conjugate_waist(s["win"], s["f1"], s["f2"], s["lambda_opt"],
                                          s["one_lens"], s["f_single"])
            self.sp_win_in.blockSignals(True)
            self.sp_win_in.setValue(s["win_in"] * 1e3)
            self.sp_win_in.blockSignals(False)
        else:
            s["win_in"] = self.sp_win_in.value() * 1e-3
            s["win"] = conjugate_waist(s["win_in"], s["f1"], s["f2"], s["lambda_opt"],
                                       s["one_lens"], s["f_single"])
            self.sp_win.blockSignals(True)
            self.sp_win.setValue(s["win"] * 1e6)
            self.sp_win.blockSignals(False)

    def _on_param_changed(self, *args):
        if self._building:
            return
        if self.state["auto_update"]:
            self.recompute()
        else:
            self.lbl_status.setText("Parameter changed - press 'Recompute'.")

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
        """A click in the 2D map moves the crosshair - the cut, the
        space-time map and I(t) then refer to that point."""
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
    # Computation
    # --------------------------------------------------------
    def recompute(self):
        self._read_widgets()
        s = self.state

        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        centers_x, centers_y, f_spots, r_center_x, r_center_y, fx_freq, fy_freq = \
            compute_centers_and_freqs(s["N_x"], s["N_y"], s["width_x"], s["width_y"],
                                      s["f1"], s["f2"], s["offset"],
                                      s["one_lens"], s["f_single"])
        amp_spots = amp_spots_from_ratios(s["r_x"], s["r_y"], s["N_x"], s["N_y"])
        n_spots = len(f_spots)

        win_eff = s["win"] * (s["airy_factor"] if s["use_airy"] else 1.0)
        x, y, X, Y = compute_grid(centers_x, centers_y, win_eff, s["grid_n"])

        F = build_field_stack(X, Y, centers_x, centers_y, amp_spots,
                              s["win"], s["use_airy"], s["airy_factor"])

        f0 = fundamental_beat_frequency(f_spots)
        degen = degenerate_groups(f_spots, f0)
        # Time window. f0 > 0: the signal is strictly periodic with T0 = 1/f0,
        # the window then covers whole periods and the time average over the
        # window is exactly the time average as such.
        # f0 == 0 with beats present: the difference frequencies are
        # incommensurable (separate widths), there is no common period at
        # all. Then we take 1/f_min as a display scale and say so - the
        # window average is only an approximation in that case.
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
            # open interval: the last frame is NOT identical to the first,
            # otherwise the loop stutters and the time average would get one
            # point twice.
            t = np.arange(n_frames) / n_frames * (s["n_periods"] * T0)
        else:
            t = np.zeros(1)
            n_frames = 1

        phases = spot_phases_from_tones(s["phase_x"], s["phase_y"], s["N_x"], s["N_y"])

        cost = n_frames * len(f_spots) * X.size
        if cost > 4e9:
            QMessageBox.warning(self, "Too large",
                                "This combination of grid, frames and number of tones "
                                "would be very slow. Please reduce the grid resolution "
                                "or the frames per period.")
            return
        self.lbl_status.setText("computing ...")
        QApplication.processEvents()

        cube = intensity_cube(F, f_spots, phases, t)

        # Camera exposure: cyclic running mean over the frames that fall into
        # t_exp. Exact in the same sense as the frame sampling itself - if
        # there are enough frames for the fastest beat (the warning below),
        # the running mean is the boxcar. The mean over whole periods is
        # untouched by it, only the envelopes and the modulation change.
        n_win = int(round(s["t_exp"] * n_frames / max(s["n_periods"] * T0, 1e-30))) \
            if (s["t_exp"] > 0 and np.isfinite(T0)) else 0
        t_exp_eff = (n_win * (s["n_periods"] * T0) / n_frames) if n_win > 1 else 0.0
        if n_win > 1:
            cube = boxcar_in_time(cube, n_win)

        # References. I_avg is the time average; over a whole fundamental
        # period it must be exactly the incoherent sum sum_s A_s^2 |u_s|^2 of
        # the previous GUIs - that is precisely what resid checks.
        I_avg = cube.mean(axis=0).astype(np.float64)
        I_incoh = np.einsum("sij,sij->ij", F, F)
        denom = np.max(I_incoh) if np.max(I_incoh) > 0 else 1.0
        resid = float(np.max(np.abs(I_avg - I_incoh)) / denom)

        norm = float(np.max(I_incoh)) if np.max(I_incoh) > 0 else 1.0

        # Enhancement and modulation depth. n_eff = I_max/<I> is the number of
        # tones effectively overlapping at the respective position: 1 where a
        # single spot dominates, up to the number of spots where all
        # contribute equally.
        k_orders = beat_orders(f_spots, f0) if f0 > 0 else np.zeros(len(f_spots), int)
        mean_exact, var_exact = time_stats_exact(F, k_orders, phases,
                                                 f0=f0, t_exp=t_exp_eff)
        sigma_rel = np.sqrt(np.maximum(var_exact, 0.0)) / np.maximum(mean_exact, 1e-300)

        I_max_map = cube.max(axis=0).astype(np.float64)
        I_min_map = cube.min(axis=0).astype(np.float64)
        n_eff = I_max_map / np.maximum(I_avg, 1e-300)
        depth = (I_max_map - I_min_map) / np.maximum(I_max_map + I_min_map, 1e-300)
        plateau = I_avg > 0.5 * I_avg.max() if I_avg.max() > 0 else np.zeros_like(I_avg, bool)
        if plateau.any():
            heavy_ok = s["heavy_analysis"] or n_spots <= 64
            try:
                if not heavy_ok:
                    raise RuntimeError("skipped, too many spots")
                obj_spec = VariationObjective(F, k_orders, plateau)
                spectrum = obj_spec.components(phases)
            except Exception:
                spectrum = {}
            sigma_rms = float(np.sqrt(np.mean(sigma_rel[plateau] ** 2)))
            # Uniformity U(t) for the three evaluation regions. The reference
            # is U(<I>) in each case - exactly the number the other GUIs and
            # the scan pipeline report.
            sites_mask = np.zeros(I_avg.shape, bool)
            for cxi, cyi in zip(centers_x, centers_y):
                sites_mask[int(np.argmin(np.abs(y - cyi))),
                           int(np.argmin(np.abs(x - cxi)))] = True
            r_circ = self.sp_opt_radius.value() * 1e-6
            circ_mask = ((X - float(np.mean(centers_x))) ** 2
                         + (Y - float(np.mean(centers_y))) ** 2) <= r_circ ** 2
            u_regions = {
                "Plateau": plateau,
                "Spot centres": sites_mask,
                f"Circle r={r_circ * 1e6:.1f} um": circ_mask,
            }
            # --- pulsed operation: uniformity of the accumulated Rabi area ---
            try:
                if not heavy_ok:
                    raise RuntimeError("skipped, too many spots")
                t_p = 1.0 / (2.0 * s["f_rabi"])
                pa = PulseArea(F, k_orders, f0, circ_mask, law=s["rabi_law"])
                t0_scan = np.linspace(0.0, T0, 90, endpoint=False) if periodic else np.zeros(1)
                u_t0 = np.array([pa.uniformity(phases, tt, t_p, f_spots) for tt in t0_scan])
                area_t0 = np.array([pa.theta(phases, tt, t_p, f_spots).mean()
                                    for tt in t0_scan])
                # Sensitivity of the pulse area to timing jitter. On a flat
                # part of the area curve the jitter only enters in second
                # order - that is worth far more than the last percent of
                # uniformity, see _on_snap_flat().
                a_mean = float(np.mean(area_t0)) if area_t0.size else 1.0
                dA = (np.gradient(area_t0, t0_scan) if len(t0_scan) > 2
                      else np.zeros_like(area_t0))
                i_now = int(np.argmin(np.abs(t0_scan - s["pulse_t0"])))
                slope = abs(float(dA[i_now])) if dA.size else 0.0
                jit = (0.01 * a_mean / slope) if slope > 0 else float("inf")
                flat = np.flatnonzero(np.diff(np.sign(dA)) != 0) if dA.size else np.array([])
                i_flat = int(flat[int(np.argmin(u_t0[flat]))]) if flat.size else i_now
                pulse = dict(t_p=t_p, t0=s["pulse_t0"], t0_scan=t0_scan, u=u_t0,
                             area=area_t0, area_norm=area_t0 / max(a_mean, 1e-300),
                             dA=dA, jitter=jit, i_flat=i_flat,
                             t0_flat=float(t0_scan[i_flat]), u_flat=float(u_t0[i_flat]),
                             u_now=pa.uniformity(phases, s["pulse_t0"], t_p, f_spots),
                             theta=pa.theta(phases, s["pulse_t0"], t_p, f_spots),
                             u_ref=uniformity_of(mean_exact, circ_mask))
            except Exception:
                pulse = None

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
            u_series, u_ref = {}, {}
            pulse = None
        # The RF signal carries VOLTAGES; r is an RF power ratio -> sqrt(r).
        crest_x = crest_factor(fx_freq, s["phase_x"],
                               amps=rf_voltage_ratios(s["r_x"], s["N_x"]))
        crest_y = crest_factor(fy_freq, s["phase_y"],
                               amps=rf_voltage_ratios(s["r_y"], s["N_y"]))

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
            "F_stack": F, "k_orders": k_orders, "region_mask": circ_mask,
            "u_series": u_series, "u_ref": u_ref, "pulse": pulse,
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

        crest_txt = ("Crest factor RF: not defined (free spot phases)"
                     if not np.isfinite(crest_x)
                     else f"Crest factor RF: x {crest_x:.2f}, y {crest_y:.2f}")
        self.lbl_phase.setText(
            f"Peak: {cube.max() / norm:.2f} x max<I>   |   " + crest_txt + "\n"
            f"In the plateau: sigma_t/<I> = {sigma_rms * 100:.1f} %,  "
            f"n_eff (overlap) = {n_eff_med:.1f},  "
            f"modulation depth = {depth_med * 100:.1f} %\n"
            + (f"Degenerate pairs: phase difference "
               f"{np.degrees(phases[degen[0][0]] - phases[degen[0][1]]) % 180:.1f} degrees "
               f"-> static contribution {resid * 100:.2f} % "
               f"({'in quadrature, time average = incoherent sum' if resid < 1e-6 else 'not in quadrature'})\n"
               if degen else "")
            + (self._opt_note + "\n" if self._opt_note else "")
            + "Phases can lower sigma_t/<I> (cross terms of equal difference "
              "frequency partially cancel), but not to zero.")
        self._opt_note = ""
        if pulse is not None:
            sw = (pulse["area"].max() / pulse["area"].min()
                  if pulse["area"].min() > 0 else float("inf"))
            jt = pulse["jitter"]
            jtxt = ("insensitive to first order" if not np.isfinite(jt)
                    else f"{jt * 1e9:.0f} ns per 1 % area error")
            self.lbl_pulse.setText(
                f"pi pulse T = {pulse['t_p'] * 1e6:.3f} us "
                f"({pulse['t_p'] / T0 * 100:.1f} % of one beat period)\n"
                f"U(pulse area) at t_0 = {s['pulse_t0'] * 1e6:.3f} us: "
                f"{pulse['u_now'] * 100:.1f} %   |   best t_0: {pulse['u'].min() * 100:.1f} % "
                f"at {pulse['t0_scan'][int(np.argmin(pulse['u']))] * 1e6:.3f} us\n"
                f"timing tolerance here: {jtxt}   |   flat point at "
                f"{pulse['t0_flat'] * 1e6:.3f} us (U {pulse['u_flat'] * 100:.1f} %)\n"
                f"without a fixed pulse timing the area varies by a factor of {sw:.1f} - "
                f"the pulse MUST be triggered on the AWG waveform.\n"
                f"Reference U from the time average: {pulse['u_ref'] * 100:.1f} %")
            self.lbl_pulse.setStyleSheet(
                "color: #a00; font-size: 10px;" if sw > 1.5
                else "color: #555; font-size: 10px;")
        self._update_labels()
        self.draw_frame(full=True)
        if degen:
            note = (f"Time average deviates by {resid * 100:.3f} % from the incoherent "
                    f"picture - this is NOT a computational error, but the static "
                    f"interference of the frequency degenerate spots.")
        else:
            note = (f"Time average == incoherent sum to within {resid:.1e} - "
                    f"the previous picture is exactly the time average.")
        if periodic and s["frames_per_period"] < need_fpp:
            note += (f"\nWARNING: {s['frames_per_period']} frames/period are not enough. "
                     f"The fastest beat frequency is {beats[-1] * 1e-3:.1f} kHz = "
                     f"{beats[-1] / f0:.0f} x f_0; Nyquist requires at least "
                     f"{need_fpp}. Otherwise the time average and the envelopes are "
                     f"wrong (aliasing).")
        if s["one_lens"]:
            build = (f"single lens f = {s['f_single'] * 1e3:.1f} mm, "
                     f"waist before it {s['win_in'] * 1e3:.3f} mm "
                     f"-> w_0 = {s['win'] * 1e6:.3f} um")
        else:
            build = (f"telescope f1/f2 = {s['f1'] * 1e3:.0f}/{s['f2'] * 1e3:.0f} mm "
                     f"+ fLO = {fLO * 1e3:.2f} mm")
        if t_exp_eff > 0:
            build += (f"   |   camera exposure {t_exp_eff * 1e6:.2f} us "
                      f"({n_win} frames), fundamental kept at "
                      f"{abs(float(np.sinc(f0 * t_exp_eff))) * 100:.0f} %")
        elif s["t_exp"] > 0:
            build += "   |   exposure < one frame, ignored"
        if not (s["heavy_analysis"] or n_spots <= 64):
            build += "   |   pulse/spectrum skipped (>64 spots)"
        self.lbl_status.setText(
            f"done: {n_frames} frames, {len(f_spots)} spots, grid {s['grid_n']}^2.\n"
            + build + "\n" + note)
        if periodic and s["frames_per_period"] < need_fpp:
            self.lbl_status.setStyleSheet("color: #a00; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        if was_running:
            self.timer.start(max(1, int(1000 / self.slider_speed.value())))

    def _rebuild_traces(self):
        """Cuts, space-time map and I(t) for the current crosshair."""
        c = self.cache
        if not c:
            return
        row, col = c["row"], c["col"]
        c["st_map"] = c["cube"][:, row, :]        # I(x, t) along the x cut
        c["trace"] = c["cube"][:, row, col]       # I(t) at the crosshair
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
            f"{len(c['f_spots'])} spots, pitch of the tones "
            f"{abs(c['centers_x'][-1] - c['centers_x'][0]) * 1e6 / max(1, s['N_x'] - 1):.3f} um in x")

        self.lbl_amps.setText(
            "amp_x = [" + ", ".join(f"{v:g}" for v in amps_from_ratio(s["r_x"], s["N_x"])) + "]\n"
            "amp_y = [" + ", ".join(f"{v:g}" for v in amps_from_ratio(s["r_y"], s["N_y"])) + "]\n"
            "(intensity weights = RF power ratios; field and RF voltage carry\n"
            " the square root: V_x = [" + ", ".join(f"{v:.4g}" for v in rf_voltage_ratios(s["r_x"], s["N_x"]))
            + "], V_y = [" + ", ".join(f"{v:.4g}" for v in rf_voltage_ratios(s["r_y"], s["N_y"])) + "])")

        beats = c["beats"]
        if c["periodic"]:
            shown = ", ".join(f"{b * 1e-3:.1f}" for b in beats[:8])
            more = " ..." if len(beats) > 8 else ""
            self.lbl_beat.setText(
                f"Fundamental frequency f_0 = gcd of all differences = {c['f0'] * 1e-3:.3f} kHz\n"
                f"Fundamental period T_0 = {c['T0'] * 1e6:.3f} us\n"
                f"Window = {s['n_periods']} x T_0 = {s['n_periods'] * c['T0'] * 1e6:.2f} us\n"
                f"Beat frequencies [kHz]: {shown}{more}")
        elif beats.size:
            shown = ", ".join(f"{b * 1e-3:.2f}" for b in beats[:8])
            more = " ..." if len(beats) > 8 else ""
            self.lbl_beat.setText(
                "NOT PERIODIC: the difference frequencies have no common\n"
                "multiple (separate widths). Shown is 1/f_min as a scale:\n"
                f"T_ref = {c['T0'] * 1e6:.3f} us,  window = {s['n_periods']} x T_ref = "
                f"{s['n_periods'] * c['T0'] * 1e6:.2f} us\n"
                f"The window average is therefore only an approximation.\n"
                f"Beat frequencies [kHz]: {shown}{more}")
        else:
            self.lbl_beat.setText("No beating: only one spot, or width = 0.")

        # What matters is not the size of the modulation, but whether a beat
        # line falls on nu_r or 2*nu_r: the spectrum is discrete, in between
        # the trap receives practically no power.
        nu = self.sp_nu_r.value() * 1e3
        d1, d2, crit = resonance_check(beats, nu)
        if np.isfinite(d1):
            txt = (f"Nearest beat line to nu_r ({nu * 1e-3:.1f} kHz): "
                   f"{d1 * 1e-3:.1f} kHz away;  to 2*nu_r: {d2 * 1e-3:.1f} kHz.")
            if crit:
                self.lbl_res.setText(
                    "CAUTION - " + txt + "\n"
                    "A line sits on a trap resonance. The spectrum is "
                    "discrete; that a line sits exactly there is the "
                    "dangerous case - not the modulation depth. Choose width "
                    "so that no multiple of f_0 falls there. Relevant for "
                    "illumination over many trap periods, not for a single "
                    "pulse shorter than 1/nu_r.")
                self.lbl_res.setStyleSheet("color: #a00; font-size: 10px; font-weight: bold;")
            else:
                self.lbl_res.setText(
                    txt + "\nNo line on a trap resonance - despite full "
                    "modulation depth the trap receives hardly any power there "
                    "(for illumination over many trap periods).")
                self.lbl_res.setStyleSheet("color: #060; font-size: 10px;")
        else:
            self.lbl_res.setText("")

        degen = c.get("degen", [])
        if degen:
            lines = []
            for grp in degen:
                pos = " = ".join(f"({c['centers_x'][i] * 1e6:.2f}, "
                                 f"{c['centers_y'][i] * 1e6:.2f})" for i in grp)
                lines.append(f"  {pos} um  at {c['f_spots'][grp[0]] * 1e-6:.5f} MHz")
            self.lbl_degen.setText(
                "CAUTION - frequency degenerate spots:\n" + "\n".join(lines) +
                "\nThese pairs have a cross term at 0 Hz. It never runs and "
                "never averages away: STATIC interference that the incoherent "
                "GUIs do not see. It can only be lifted by DIFFERENT widths "
                "in x and y - a constant frequency offset on one axis does "
                "not help, it cancels out of every difference.")
            self.lbl_degen.setStyleSheet(
                "color: #a00; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_degen.setText("No frequency degenerate spots - every cross "
                                   "term runs, the time average is exactly the "
                                   "incoherent picture.")
            self.lbl_degen.setStyleSheet("color: #060; font-size: 10px;")

    # --------------------------------------------------------
    # Drawing
    # --------------------------------------------------------
    def _rabi_key(self, c):
        s = self.state
        return (s["N_x"], s["N_y"], s["width_x"], s["width_y"], s["win"], s["use_airy"],
                s["airy_factor"], s["r_x"], s["r_y"], s["f1"], s["f2"], s["offset"],
                s["f_rabi"], s["pulse_t0"], s["rabi_law"], s["eta_ls"], s["grid_n"],
                self.cmb_opt_region.currentIndex(), self.sp_opt_radius.value(),
                tuple(np.round(c.get("phases", []), 9)))

    def _draw_rabi_panel(self, c):
        """Excitation over pulse length - computed only when the panel is shown.

        The curves need a few hundred evaluations of the pulse area, which is
        too much to do on every recompute; the result is therefore cached and
        only rebuilt when something it depends on has changed."""
        pz = c.get("pulse")
        if pz is None or not c.get("periodic", False):
            self.ax_st.text(0.5, 0.5, "no pulse data available",
                            ha="center", va="center", transform=self.ax_st.transAxes)
            return
        key = self._rabi_key(c)
        if self._rabi_cache is None or self._rabi_cache[0] != key:
            self.lbl_status.setText("computing the Rabi curves ...")
            QApplication.processEvents()
            s = self.state
            t_pi = pz["t_p"]
            fast = s["rabi_law"] == "I"
            pa = PulseArea(c["F_stack"], c["k_orders"], c["f0"], c["region_mask"],
                           law=s["rabi_law"], n_t=60, max_points=350 if fast else 120)
            t_p = np.linspace(1e-12, 2.2 * t_pi, 90 if fast else 45)
            ideal, trig, none = pa.rabi_curves(c["phases"], pz["t0"], t_p, t_pi,
                                               c["f_spots"],
                                               n_random=14 if fast else 6, T0=c["T0"],
                                               eta=s["eta_ls"])
            self._rabi_cache = (key, t_p, ideal, trig, none)
            self.lbl_status.setText("")
        _, t_p, ideal, trig, none = self._rabi_cache
        ct = lambda P: (P.max() - P.min()) * 100
        self.ax_st.plot(t_p * 1e6, ideal * 100, color="#888", ls="--", lw=1.4,
                        label="ideal, no beating")
        self.ax_st.plot(t_p * 1e6, trig * 100, color="#4c8b5b", lw=2.0,
                        label=f"triggered at $t_0$ (contrast {ct(trig):.0f} %)")
        self.ax_st.plot(t_p * 1e6, none * 100, color="#1b1b1b", ls=":", lw=1.6,
                        label=f"untriggered (contrast {ct(none):.0f} %)")
        self.ax_st.axvline(pz["t_p"] * 1e6, color="#888", lw=0.9)
        self.ax_st.set_xlabel("pulse length (us)", fontsize=8)
        self.ax_st.set_ylabel("excitation (%)", fontsize=8)
        eta = self.state["eta_ls"]
        cap = ("" if eta <= 0 else
               f"   |   light shift eta = {eta:g} caps the contrast at "
               f"{100 / (1 + eta ** 2):.0f} %")
        self.ax_st.set_title("Rabi oscillation, averaged over the target region" + cap,
                             fontsize=8.5)
        self.ax_st.set_ylim(0, 105)
        self.ax_st.legend(fontsize=6.5, loc="lower right", framealpha=0.85)
        self.ax_st.spines["top"].set_visible(False)
        self.ax_st.spines["right"].set_visible(False)

    def _u_title(self, c, k):
        """Title of the U(t) panel with the INSTANTANEOUS values - the number
        one actually wants to read off while watching."""
        us = c.get("u_series", {})
        if not us:
            return "Uniformity over time"
        # Short, LENGTH-STABLE labelling: with blitting the layout no longer
        # follows, a growing title would run over the edge.
        short = {"Plateau": "Plat", "Spot centres": "Ctr"}
        now = " ".join(f"{short.get(rn, 'Circ')} {u[k] * 100:3.0f}%" for rn, u in us.items())
        return "U now:  " + now + "\n(dotted: from the time average)"

    def _fast_frame(self, c, k):
        """Only exchange the data instead of rebuilding all four axes.

        A complete rebuild costs about 50 ms, so at most 20 images per
        second - too few for a live display. This way only the calls that
        really change per frame remain. Returns False if something is
        missing; then the caller draws completely."""
        if not self.cb_fastdraw.isChecked():
            return False
        art = self._art
        if not art or art.get("panel") != self.cmb_panel.currentIndex():
            return False
        # The fast path only exchanges data. If the shape no longer matches
        # the existing objects (grid resolution, time sampling or crosshair
        # changed), everything must be redrawn - otherwise image and axes
        # would no longer belong together.
        if (art.get("shape") != c["cube"].shape
                or art.get("rowcol") != (c["row"], c["col"])):
            return False
        try:
            norm = c["norm"]
            t_us = c["t"] * 1e6
            cut_row, cut_col = c["row"], c["col"]
            I_now = c["cube"][k] / norm
            art["main_im"].set_data(I_now)
            if self.cmb_scale.currentIndex() == 3:          # per frame
                art["main_im"].set_clim(0.0, max(float(I_now.max()), 1e-12))
            T0_us = c["T0"] * 1e6
            per = (f"   ({t_us[k] / T0_us:.3f} $T_0$)"
                   if np.isfinite(T0_us) and T0_us > 0 else "")
            art["main_title"].set_text(
                f"I(x, y, t) instantaneous   |   t = {t_us[k]:8.4f} us" + per +
                f"\nPeak over time: {c['cube_max'] / norm:.2f} x the "
                f"maximum of the time average")
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
        cut_row, cut_col = c["row"], c["col"]   # do NOT call them row/col:
        # "col" has already been reused twice as a colour variable in the
        # panel loops and overwrote the column index.
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

        # ---- 2D map ----
        self.ax_main.clear()
        self._art["main_im"] = self.ax_main.imshow(
            I_now, extent=extent, origin="lower", cmap="inferno",
            vmin=0.0, vmax=vmax, aspect="equal")
        self.ax_main.plot(c["centers_x"] * 1e6, c["centers_y"] * 1e6, "w+",
                          markersize=5, markeredgewidth=0.8, alpha=0.55,
                          label="spot centres")
        for grp in c.get("degen", []):
            self.ax_main.plot(c["centers_x"][grp] * 1e6, c["centers_y"][grp] * 1e6,
                              "o", mfc="none", mec="deepskyblue", ms=11, mew=1.6,
                              label="frequency degenerate")
        self.ax_main.axhline(y_um[cut_row], color="cyan", lw=0.8, alpha=0.8)
        self.ax_main.axvline(x_um[cut_col], color="magenta", lw=0.8, alpha=0.8)
        self.ax_main.set_xlabel("x (um)")
        self.ax_main.set_ylabel("y (um)")
        T0_us = c["T0"] * 1e6
        per_lbl = (f"   ({t_us[k] / T0_us:.3f} $T_0$)" if np.isfinite(T0_us) and T0_us > 0
                   else "")
        self._art["main_title"] = self.ax_main.set_title(
            f"I(x, y, t) instantaneous   |   t = {t_us[k]:8.4f} us" + per_lbl +
            f"\nPeak over time: {c['cube_max'] / norm:.2f} x the maximum "
            f"of the time average", fontsize=9)
        h, l = self.ax_main.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        self.ax_main.legend(uniq.values(), uniq.keys(), loc="upper right",
                            fontsize=7, framealpha=0.4)

        # ---- panel top right: space-time map, n_eff or modulation depth ----
        self.ax_st.clear()
        panel = self.cmb_panel.currentIndex()
        # Colorbar and aspect ratio belong only to the map panels (1..3). If
        # they remain when switching, they squeeze the space-time map or the
        # bar chart.
        if panel in (0, 4, 5, 6, 7, 8):
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
            self.ax_st.set_title(f"Space-time map I(x, t) at y = {y_um[cut_row]:.3f} um",
                                 fontsize=9)
        elif panel == 8:
            self._draw_rabi_panel(c)
        elif panel == 7:
            pz = c.get("pulse")
            if pz is not None and "area_norm" in pz and len(pz["t0_scan"]) > 1:
                tt = pz["t0_scan"] * 1e6
                self.ax_st.plot(tt, pz["area_norm"], color="#3b6ea5", lw=1.8,
                                label="pulse area / mean")
                self.ax_st.axhline(1.0, color="#888", ls="--", lw=0.9)
                i_f = pz["i_flat"]
                self.ax_st.plot([tt[i_f]], [pz["area_norm"][i_f]], "o",
                                color="#4c8b5b", ms=8, label="flat point (jitter-tolerant)")
                self.ax_st.axvline(pz["t0"] * 1e6, color="#b3402f", lw=1.4,
                                   label="current $t_0$")
                jt = pz["jitter"]
                jtxt = ("no first-order sensitivity" if not np.isfinite(jt)
                        else f"{jt * 1e9:.0f} ns per 1 % area error")
                self.ax_st.set_xlabel("pulse start $t_0$ (us)", fontsize=8)
                self.ax_st.set_ylabel("pulse area / mean", fontsize=8)
                self.ax_st.set_title(
                    f"Pulse area over the start time   |   at the current $t_0$: {jtxt}",
                    fontsize=8.5)
                self.ax_st.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "no pulse data available",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        elif panel == 6:
            pz = c.get("pulse")
            if pz is not None and len(pz["t0_scan"]) > 1:
                tt = pz["t0_scan"] * 1e6
                self.ax_st.plot(tt, pz["u"] * 100, color="#3b6ea5", lw=1.6,
                                label="U of the pulse area")
                self.ax_st.axhline(pz["u_ref"] * 100, color="#4c8b5b", ls=":", lw=1.2,
                                   label="U from the time average")
                i_best = int(np.argmin(pz["u"]))
                self.ax_st.plot([tt[i_best]], [pz["u"][i_best] * 100], "o",
                                color="#4c8b5b", ms=6)
                self.ax_st.annotate(f"best t_0 = {tt[i_best]:.2f} us\n"
                                    f"U = {pz['u'][i_best] * 100:.1f} %",
                                    (tt[i_best], pz["u"][i_best] * 100),
                                    xytext=(6, 8), textcoords="offset points",
                                    fontsize=7, color="#2f6b45")
                self.ax_st.axvline(pz["t0"] * 1e6, color="#b3402f", lw=1.4)
                self.ax_st.set_xlabel("pulse start t_0 (us)", fontsize=8)
                self.ax_st.set_ylabel("U of the pulse area (%)", fontsize=8)
                self.ax_st.set_title(
                    f"pi pulse {pz['t_p'] * 1e6:.2f} us   |   at t_0 = "
                    f"{pz['t0'] * 1e6:.2f} us: U = {pz['u_now'] * 100:.1f} %",
                    fontsize=8.5)
                self.ax_st.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
                self.ax_st.set_ylim(bottom=0)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "no pulse data available",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        elif panel == 5:
            us = c.get("u_series", {})
            if us:
                # fixed colour order, never rotated through
                cols = ["#3b6ea5", "#b3402f", "#4c8b5b"]
                live = self.cb_live.isChecked()
                self._art["u_live"] = []
                self._art["u_dot"] = []
                for i, (rname, u) in enumerate(us.items()):
                    hue = cols[i % len(cols)]
                    if live:
                        # whole trace pale for orientation, on top of it the
                        # running trace up to the current instant
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
                pz_ = c.get("pulse")
                if pz_ is not None:
                    self.ax_st.axvspan(pz_["t0"] * 1e6,
                                       min((pz_["t0"] + pz_["t_p"]) * 1e6, t_us[-1]),
                                       color="#3b6ea5", alpha=0.12, lw=0)
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
                self.ax_st.text(0.5, 0.5, "no uniformity available",
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
                # CAUTION: do not use a variable "col" here - further up that
                # is the column index of the crosshair.
                for f_mark, lab in ((nu, "$\\nu_r$"), (2 * nu, "$2\\nu_r$")):
                    if freqs[0] - width_bar <= f_mark <= freqs[-1] + width_bar:
                        self.ax_st.axvline(f_mark, color="#b3402f", lw=1.4, ls="--")
                        self.ax_st.annotate(lab, (f_mark, self.ax_st.get_ylim()[1]),
                                            xytext=(3, -2), textcoords="offset points",
                                            color="#b3402f", fontsize=9, ha="left", va="top")
                self.ax_st.set_ylim(0, max(vals.max() * 1.28, 1e-3))
                self.ax_st.set_xlabel("beat frequency (kHz)", fontsize=8)
                self.ax_st.set_ylabel("$\\sigma_d / \\langle I\\rangle$ (%)", fontsize=8)
                self.ax_st.set_title(
                    f"Spectrum of the beating in the plateau   "
                    f"(total {c['sigma_rms'] * 100:.0f} %)", fontsize=9)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "no spectrum available",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        else:
            if panel == 1:
                dat, cmap = c["n_eff"], "viridis"
                title = (f"Enhancement $n_{{eff}} = I_{{max}}/\\langle I\\rangle$   "
                         f"(plateau: {c['n_eff_med']:.1f})")
                lim = (1.0, max(2.0, float(len(c["f_spots"]))))
            elif panel == 2:
                dat, cmap = c["depth"] * 100.0, "magma"
                title = (f"Modulation depth in %   "
                         f"(plateau: {c['depth_med'] * 100:.0f} %)")
                lim = (0.0, 100.0)
            else:
                dat, cmap = c["sigma_rel"] * 100.0, "cividis"
                title = (f"temporal variation $\\sigma_t/\\langle I\\rangle$ in %   "
                         f"(plateau: {c['sigma_rms'] * 100:.0f} %)")
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

        # ---- cuts with envelope ----
        self.ax_cut.clear()
        self.ax_cut.fill_between(x_um, c["cut_x_min"] / norm, c["cut_x_max"] / norm,
                                 color="tab:orange", alpha=0.22, lw=0,
                                 label="min/max over t")
        self.ax_cut.plot(x_um, c["I_avg"][cut_row, :] / norm, "k--", lw=1.0,
                         label="time average")
        (self._art["cut_now"],) = self.ax_cut.plot(
            x_um, c["cube"][k, cut_row, :] / norm, color="tab:orange", lw=1.3,
            label="instantaneous")
        self.ax_cut.axvline(x_um[cut_col], color="magenta", lw=0.8, alpha=0.7)
        self.ax_cut.set_xlabel("x (um)", fontsize=8)
        self.ax_cut.set_ylabel("I / I_max", fontsize=8)
        self.ax_cut.set_title("x cut", fontsize=9)
        self.ax_cut.tick_params(labelsize=7)
        self.ax_cut.legend(fontsize=6.5, loc="upper right", framealpha=0.5)

        # ---- I(t) at the crosshair ----
        self.ax_time.clear()
        tr = c["trace"] / norm
        self.ax_time.plot(t_us, tr, color="tab:blue", lw=1.0)
        pz = c.get("pulse")
        if pz is not None:
            self.ax_time.axvspan(pz["t0"] * 1e6,
                                 min((pz["t0"] + pz["t_p"]) * 1e6, t_us[-1]),
                                 color="#3b6ea5", alpha=0.15, lw=0)
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
        extra = (f"\npi pulse {pz['t_p'] * 1e6:.2f} us from {pz['t0'] * 1e6:.2f} us: "
                 f"U(area) {pz['u_now'] * 100:.0f} %" if pz else "")
        self.ax_time.set_title(
            f"I(t) at the crosshair - modulation {depth * 100:.0f} %" + extra, fontsize=8.5)
        self.ax_time.tick_params(labelsize=7)

        prof = "Airy" if s["use_airy"] else "Gauss"
        self.fig.suptitle(
            f"N = {s['N_x']}x{s['N_y']} tones, {prof}, waist = {s['win'] * 1e6:.3f} um, "
            f"width = {s['width_x'] * 1e-6:.4f}/{s['width_y'] * 1e-6:.4f} MHz (x/y), "
            f"r_x = {s['r_x']:g}, r_y = {s['r_y']:g}, "
            f"lambda = {s['lambda_opt'] * 1e9:.1f} nm, f0 = {s['offset'] * 1e-6:.3f} MHz\n"
            + (f"Fundamental period T_0 = {T0_us:.3f} us  (f_0 = {c['f0'] * 1e-3:.3f} kHz), "
               f"window = {s['n_periods']} periods" if c["periodic"]
               else f"not periodic - scale T_ref = {T0_us:.3f} us, "
                    f"window = {s['n_periods']} x T_ref"),
            fontsize=10)
        self._art["panel"] = self.cmb_panel.currentIndex()
        self._art["shape"] = c["cube"].shape
        self._art["rowcol"] = (c["row"], c["col"])
        self.canvas.draw_idle()

    # --------------------------------------------------------
    def _open_power_budget(self):
        """Leistung und Intensitaet des Profils, eigenes Fenster."""
        if power_budget is None:
            QMessageBox.warning(
                self, "Module missing",
                "power_budget.py was not found next to this file.")
            return
        if not self.cache:
            self.recompute()
        dlg = getattr(self, "_power_dlg", None)
        if dlg is None:
            dlg = power_budget.PowerBudgetDialog(self)
            self._power_dlg = dlg
        else:
            dlg.recompute()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_pulse_timing(self):
        """Pulsflaeche und Trigger-Jitter, eigenes Fenster."""
        if pulse_timing is None:
            QMessageBox.warning(
                self, "Module missing",
                "pulse_timing.py was not found next to this file.")
            return
        if not self.cache:
            self.recompute()
        dlg = getattr(self, "_pulse_dlg", None)
        if dlg is None:
            dlg = pulse_timing.PulseTimingDialog(self)
            self._pulse_dlg = dlg
        else:
            dlg.recompute()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_camera_series(self):
        """Bildserie einer Kamera ueber eine Grundperiode, eigenes Fenster."""
        if camera_series is None:
            QMessageBox.warning(
                self, "Module missing",
                "camera_series.py was not found next to this file.")
            return
        if not self.cache:
            self.recompute()
        dlg = getattr(self, "_camera_dlg", None)
        if dlg is None:
            dlg = camera_series.CameraSeriesDialog(self)
            self._camera_dlg = dlg
        else:
            dlg.redraw()
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_save_clicked(self):
        self.draw_frame(full=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        s = self.state
        # Vector PDF instead of a raster image: the outputs end up in LaTeX
        # documents, where a PNG is needlessly blurry after downscaling.
        tag = (f"_1lens{s['f_single'] * 1e3:.0f}mm" if s["one_lens"] else "")
        tag += (f"_texp{s['t_exp'] * 1e6:.0f}us" if s["t_exp"] > 0 else "")
        name = (f"Beating_N{s['N_x']}x{s['N_y']}_"
                f"{'Airy' if s['use_airy'] else 'Gauss'}_"
                f"w{s['win'] * 1e6:.3f}um_width{s['width_x'] * 1e-6:.4f}MHz"
                f"{tag}_frame{self.frame_idx:04d}_{stamp}")
        path = self.out_dir / (name + ".pdf")
        try:
            self.fig.savefig(path, format="pdf", bbox_inches="tight")
            # Parameter daneben als Textdatei - ein Bild ohne die Zahlen,
            # aus denen es entstanden ist, ist spaeter nicht rekonstruierbar.
            c = self.cache
            lines = [f"{stamp}", "",
                     "build: " + ("single lens f = %.3f mm, waist before lens "
                                  "%.4f mm" % (s["f_single"] * 1e3, s["win_in"] * 1e3)
                                  if s["one_lens"] else
                                  "telescope f1 = %.2f mm, f2 = %.2f mm, fLO = %.2f mm"
                                  % (s["f1"] * 1e3, s["f2"] * 1e3, fLO * 1e3)),
                     f"AOD: theta_max = {theta_max * 1e3:.1f} mrad, "
                     f"f_band = {f_band * 1e-6:.1f} MHz, "
                     f"offset = {s['offset'] * 1e-6:.4f} MHz",
                     f"lambda = {s['lambda_opt'] * 1e9:.2f} nm",
                     f"N_x x N_y = {s['N_x']} x {s['N_y']}",
                     f"width_x = {s['width_x'] * 1e-6:.6f} MHz, "
                     f"width_y = {s['width_y'] * 1e-6:.6f} MHz",
                     f"waist (focus) = {s['win'] * 1e6:.4f} um, "
                     f"profile = {'Airy' if s['use_airy'] else 'Gauss'} "
                     f"(factor {s['airy_factor']:.4f})",
                     f"r_x = {s['r_x']:.4f}, r_y = {s['r_y']:.4f}",
                     f"camera exposure = {s['t_exp'] * 1e6:.3f} us",
                     f"grid = {s['grid_n']}^2, frames/period = "
                     f"{s['frames_per_period']}, periods = {s['n_periods']}",
                     f"frame index = {self.frame_idx}"]
            if c:
                lines += [f"f_0 = {c['f0'] * 1e-3:.6f} kHz  ->  T_0 = "
                          f"{c['T0'] * 1e6:.4f} us",
                          f"spots = {len(c['f_spots'])}, degenerate groups = "
                          f"{len(c['degen'])}, resid = {c['resid'] * 100:.4f} %",
                          f"sigma_t/<I> (plateau) = {c['sigma_rms'] * 100:.2f} %, "
                          f"modulation depth = {c['depth_med'] * 100:.2f} %"]
            lines.append("phases x [deg] = "
                         + ", ".join(f"{v:.2f}" for v in np.degrees(s["phase_x"])))
            lines.append("phases y [deg] = "
                         + ", ".join(f"{v:.2f}" for v in np.degrees(s["phase_y"])))
            (self.out_dir / (name + ".txt")).write_text("\n".join(lines),
                                                        encoding="utf-8")
            self.lbl_status.setText(f"saved: {short_name(path)} (+ .txt)")
            self.lbl_status.setToolTip(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "Saving failed", str(exc))


def main():
    app = QApplication(sys.argv)
    win = BeatingMultitoneWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
