"""
Interactive Gaussian beam propagation through a 3-lens system (PyQt5 dialog).
Lenses & screen can be dragged in the plot, all values are freely editable
(mouse wheel over a field scales its value).

This module can be used standalone:
    python3 lens_design_dialog.py
or imported and opened as a dialog from another PyQt5 application (see
FlatMultiTone_GUI_PyQt5.py, button "Open Lens Design Tool..."). When used as
a dialog, it emits the Qt signal `optics_changed` (a dict) every time the
beam is recomputed, so a parent window can live-sync its own beam-size /
scanning-range display.

Requires: pip install PyQt5 matplotlib numpy
"""

import sys
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QLabel, QPushButton, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

# ============================================================
# Standard focal lengths (mm) - informational only, not a hard restriction
# ============================================================
STANDARD_F_MM = [
    -1000, -750, -500, -300, -200, -150, -100, -75, -50, -40, -30, -25,
    25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400,
    500, 600, 750, 1000, 1500, 2000,
]


def nearest_standard(value):
    return min(STANDARD_F_MM, key=lambda f: abs(f - value))


# ============================================================
# Wavelength (nm) -> RGB (0..1), for a physically plausible beam color
# ============================================================
def wavelength_to_rgb(nm):
    if 380 <= nm < 440:
        r, g, b = -(nm - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= nm < 490:
        r, g, b = 0.0, (nm - 440) / (490 - 440), 1.0
    elif 490 <= nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / (510 - 490)
    elif 510 <= nm < 580:
        r, g, b = (nm - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= nm < 645:
        r, g, b = 1.0, -(nm - 645) / (645 - 580), 0.0
    elif 645 <= nm <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.6, 0.6, 0.6
    factor = 1.0
    if 380 <= nm < 420:
        factor = 0.3 + 0.7 * (nm - 380) / (420 - 380)
    elif 700 <= nm <= 780:
        factor = 0.3 + 0.7 * (780 - nm) / (780 - 700)
    gamma = 0.8
    adj = lambda c: max(c * factor, 0.0) ** gamma
    return (adj(r), adj(g), adj(b))


# ============================================================
# Gaussian beam physics (q-parameter as Python complex, ABCD)
# ============================================================
def prop_q_free(q, d):
    return q + d


def prop_q_lens(q, f):
    return q / (-q / f + 1)


def beam_width(q, wavelength_mm):
    return np.sqrt((wavelength_mm / np.pi) * (abs(q) ** 2) / q.imag)


def prop_ray_free(ray, d):
    y, th = ray
    return (y + d * th, th)


def prop_ray_lens(ray, f):
    y, th = ray
    return (y, th - y / f)


def compute_beam(x1, x2, x3, xscreen, f1, f2, f3, w0_mm, wavelength_nm, theta0_mrad):
    wavelength_mm = wavelength_nm * 1e-6
    theta0 = theta0_mrad * 1e-3
    zR = np.pi * w0_mm ** 2 / wavelength_mm

    q = 0 + 1j * zR
    ray = (0.0, theta0)
    z = 0.0

    segs = [
        ("prop", x1 - 0), ("lens", f1, x1),
        ("prop", x2 - x1), ("lens", f2, x2),
        ("prop", x3 - x2), ("lens", f3, x3),
        ("prop", xscreen - x3),
    ]

    z_list = [0.0]
    w_list = [beam_width(q, wavelength_mm)]
    y_list = [ray[0]]
    lens_info = []

    for seg in segs:
        if seg[0] == "prop":
            length = max(seg[1], 0.0)
            n = max(6, min(140, round(length / 4)))
            for i in range(1, n + 1):
                dz = length * i / n
                q_local = prop_q_free(q, dz)
                ray_local = prop_ray_free(ray, dz)
                z_list.append(z + dz)
                w_list.append(beam_width(q_local, wavelength_mm))
                y_list.append(ray_local[0])
            q = prop_q_free(q, length)
            ray = prop_ray_free(ray, length)
            z += length
        else:
            _, f, xpos = seg
            lens_info.append({"x": xpos, "f": f, "w": beam_width(q, wavelength_mm), "y": ray[0]})
            q = prop_q_lens(q, f)
            ray = prop_ray_lens(ray, f)

    z_arr, w_arr, y_arr = np.array(z_list), np.array(w_list), np.array(y_list)
    imin = int(np.argmin(w_arr))
    return {
        "z": z_arr, "w": w_arr, "y": y_arr, "lenses": lens_info, "zR": zR,
        "waist_min": (z_arr[imin], w_arr[imin]),
        "final": (w_arr[-1], y_arr[-1]),
    }


# ============================================================
# Qt stylesheet (dark theme, applies to all widgets)
# ============================================================
DARK_QSS = """
QDialog, QWidget { background-color: #141619; color: #e8eaed; font-size: 12px; }
QLabel { color: #9aa1ab; }
QLineEdit {
    background-color: #0d0f11; color: #e8eaed;
    border: 1px solid #333944; border-radius: 4px;
    padding: 4px 6px; min-height: 18px;
}
QLineEdit:focus { border: 1px solid #5eead4; }
QPushButton {
    background-color: #2a3138; color: #e8eaed;
    border: 1px solid #3a4650; border-radius: 5px;
    padding: 7px 12px;
}
QPushButton:hover { background-color: #3a4650; }
QPushButton:pressed { background-color: #1c2126; }
"""


# ============================================================
# QLineEdit with mouse-wheel support (scroll = scale the value)
# ============================================================
class ScrollLineEdit(QLineEdit):
    def __init__(self, step, decimals=2, parent=None):
        super().__init__(parent)
        self.step = step
        self.decimals = decimals
        self.setAlignment(Qt.AlignRight)

    def wheelEvent(self, event):
        try:
            val = float(self.text())
        except ValueError:
            val = 0.0
        direction = 1 if event.angleDelta().y() > 0 else -1
        val += direction * self.step
        self.setText(f"{val:.{self.decimals}f}")
        self.editingFinished.emit()  # triggers the same handler as manual typing
        event.accept()


# ============================================================
# Main dialog
# ============================================================
class LensDesignDialog(QDialog):
    """3-lens Gaussian beam design tool.

    Emits `optics_changed` (dict) every time the beam is recomputed, with keys:
        f1_mm, f2_mm, f3_mm, w0_mm, wavelength_nm, atom_size_um,
        w_screen_um          - 1/e^2 beam RADIUS at the screen, in µm
        central_profile_um   - full 1/e^2-to-1/e^2 width at the screen (= 2*w), in µm
        scan_range_um        - scanning range r for the full 43 mrad deflection angle, in µm
        scan_range_atoms     - scan_range_um / atom_size_um
    """

    optics_changed = pyqtSignal(dict)

    GAP = 15.0  # mm minimum spacing between elements

    def __init__(self, parent=None, f1_mm=75.0, f2_mm=750.0, f3_mm=52.88,
                 w0_mm=1.0, theta0_mrad=3.0, wavelength_nm=795.0, atom_size_um=5.288):
        super().__init__(parent)
        self.setWindowTitle("Gaussian Beam · 3-Lens System")
        self.setStyleSheet(DARK_QSS)
        self.setWindowModality(Qt.NonModal)

        # -------- default / initial state --------
        self.f1_default, self.f2_default, self.f3_default = f1_mm, f2_mm, f3_mm
        self.f1, self.f2, self.f3 = f1_mm, f2_mm, f3_mm
        self.w0_mm = w0_mm
        self.theta0_mrad = theta0_mrad
        self.wavelength_nm = wavelength_nm
        self.atom_size_um = atom_size_um
        self.dragging = None

        self.x1 = self.f1
        self.x2 = self.x1 + self.f1 + self.f2
        self.x3 = self.x2 + self.f2 + self.f3
        self.xscreen = self.x3 + self.f3

        self._build_ui()
        self.redraw()
        self.resize(1200, 900)

    # -------------------- state / physics helpers --------------------
    def set_confocal(self, full_reset=False):
        """full_reset=True: d1=f1, d2=f1+f2, d3=f2+f3, d4=f3.
        full_reset=False: x1 and d4 are kept, only d2/d3 recomputed from f1,f2,f3."""
        x1 = self.f1 if full_reset else self.x1
        x2 = x1 + self.f1 + self.f2
        x3 = x2 + self.f2 + self.f3
        d4 = self.f3 if full_reset else max(self.xscreen - self.x3, self.GAP)
        self.x1, self.x2, self.x3 = x1, x2, x3
        self.xscreen = x3 + d4

    def _set_d1(self, v):
        v = max(self.GAP, v)
        self.x1 = min(v, self.x2 - self.GAP)

    def _set_d2(self, v):
        v = max(self.GAP, v)
        self.x2 = self.x1 + v
        if self.x2 > self.x3 - self.GAP:
            self.x2 = self.x3 - self.GAP

    def _set_d3(self, v):
        v = max(self.GAP, v)
        self.x3 = self.x2 + v
        if self.x3 > self.xscreen - self.GAP:
            self.x3 = self.xscreen - self.GAP

    def _set_d4(self, v):
        self.xscreen = self.x3 + max(self.GAP, v)

    def _apply_f1(self, v):
        self.f1 = v
        self.set_confocal(full_reset=False)

    def _apply_f2(self, v):
        self.f2 = v
        self.set_confocal(full_reset=False)

    def _apply_f3(self, v):
        self.f3 = v
        self.set_confocal(full_reset=False)

    def positions_ok(self):
        return 0 < self.x1 < self.x2 < self.x3 < self.xscreen

    def _system_matrix_to_screen(self):
        """Full ABCD matrix from the source (z=0) to the screen.
        y_screen = A*y0 + B*theta0. Since y0=0 (beam starts on axis),
        y_screen = B*theta0 -> B directly gives the angle->position gain."""
        def prop(d):
            return np.array([[1.0, d], [0.0, 1.0]])

        def lens(f):
            return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])

        segs = [prop(self.x1), lens(self.f1),
                prop(self.x2 - self.x1), lens(self.f2),
                prop(self.x3 - self.x2), lens(self.f3),
                prop(self.xscreen - self.x3)]
        M = np.eye(2)
        for seg in segs:
            M = seg @ M
        return M

    def _build_stats_box(self):
        box = QWidget()
        box.setObjectName("statsBox")
        box.setStyleSheet(
            "QWidget#statsBox { background-color: #1c1f24; border: 1px solid #2c313a; border-radius: 8px; }"
        )
        stats_grid = QGridLayout(box)
        stats_grid.setContentsMargins(14, 10, 14, 10)
        stats_grid.setHorizontalSpacing(24)
        stats_grid.setVerticalSpacing(8)

        panel_title = QLabel("Beam Size &amp; Scanning Range")
        panel_title.setStyleSheet("font-size: 12px; font-weight: 600; color: #e8eaed; border: none;")
        stats_grid.addWidget(panel_title, 0, 0, 1, 2)

        def add_stat(row, col, key_text, colspan=1):
            cell = QWidget()
            v = QVBoxLayout(cell)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(1)
            klbl = QLabel(key_text)
            klbl.setStyleSheet("font-size: 10px; color: #7c8390; border: none;")
            klbl.setWordWrap(True)
            vlbl = QLabel("\u2013")
            vlbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #5eead4; "
                                "font-family: monospace; border: none;")
            vlbl.setWordWrap(True)
            v.addWidget(klbl)
            v.addWidget(vlbl)
            stats_grid.addWidget(cell, row, col, 1, colspan)
            return vlbl

        self.lbl_stat_zr = add_stat(1, 0, "Rayleigh length z_R")
        self.lbl_stat_wscreen = add_stat(1, 1, "Beam radius at screen (1/e\u00b2)")
        self.lbl_stat_wmin = add_stat(2, 0, "Minimum beam width (position)")
        self.lbl_stat_yscreen = add_stat(2, 1, "Offset at screen (due to \u03b80)")
        self.lbl_stat_central = add_stat(3, 0, "Central profile size (1/e\u00b2 to 1/e\u00b2) at screen", colspan=2)
        self.lbl_stat_scan = add_stat(4, 0, "Scanning range (43 mrad full angle, at screen)", colspan=2)
        stats_grid.setColumnMinimumWidth(0, 210)
        stats_grid.setColumnMinimumWidth(1, 210)
        return box

    # -------------------- UI construction --------------------
    def _make_field(self, label_text, initial, step, decimals, callback):
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        lbl = QLabel(label_text)
        edit = ScrollLineEdit(step, decimals)
        edit.setText(f"{initial:.{decimals}f}")
        edit.setFixedWidth(78)
        edit.editingFinished.connect(callback)
        v.addWidget(lbl)
        v.addWidget(edit)
        return container, edit

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        header = QVBoxLayout()
        title = QLabel("Gaussian Beam · 3-Lens System")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #e8eaed;")
        header.addWidget(title)

        hint = QLabel("Drag lenses & screen in the plot. "
                       "Mouse wheel over a field scales its value. "
                       "Vertical axis is strongly exaggerated. "
                       "Results are applied to the main window live, as you change them.")
        hint.setWordWrap(True)
        header.addWidget(hint)

        root.addLayout(header)

        # ---- Plot ----
        self.fig = Figure(figsize=(11, 4.8), facecolor="#141619")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(380)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#1b1e23")
        root.addWidget(self.canvas, stretch=1)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # ---- Input fields: f1, f2, f3, w0 ----
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        f1_c, self.edit_f1 = self._make_field("f1 [mm]", self.f1, 1.0, 2, self._on_f1)
        f2_c, self.edit_f2 = self._make_field("f2 [mm]", self.f2, 1.0, 2, self._on_f2)
        f3_c, self.edit_f3 = self._make_field("f3 [mm]", self.f3, 0.5, 2, self._on_f3)
        w0_c, self.edit_w0 = self._make_field("w0 start [mm]", self.w0_mm, 0.05, 3, self._on_w0)

        self.lbl_f1_std = QLabel("")
        self.lbl_f2_std = QLabel("")
        self.lbl_f1_std.setStyleSheet("font-size: 10px;")
        self.lbl_f2_std.setStyleSheet("font-size: 10px;")
        f1_c.layout().addWidget(self.lbl_f1_std)
        f2_c.layout().addWidget(self.lbl_f2_std)

        grid.addWidget(f1_c, 0, 0)
        grid.addWidget(f2_c, 0, 1)
        grid.addWidget(f3_c, 0, 2)
        grid.addWidget(w0_c, 0, 3)

        th_c, self.edit_theta = self._make_field("theta0 [mrad]", self.theta0_mrad, 0.5, 2, self._on_theta)
        wl_c, self.edit_wl = self._make_field("lambda [nm]", self.wavelength_nm, 5.0, 1, self._on_wl)
        atom_c, self.edit_atom = self._make_field("Atom size [\u00b5m]", self.atom_size_um, 0.1, 3, self._on_atom)
        grid.addWidget(th_c, 1, 0)
        grid.addWidget(wl_c, 1, 1)
        grid.addWidget(atom_c, 1, 2)

        d1_c, self.edit_d1 = self._make_field("d1 source\u2192L1 [mm]", self.x1, 5.0, 1, self._on_d1)
        d2_c, self.edit_d2 = self._make_field("d2 L1\u2192L2 [mm]", self.x2 - self.x1, 5.0, 1, self._on_d2)
        d3_c, self.edit_d3 = self._make_field("d3 L2\u2192L3 [mm]", self.x3 - self.x2, 5.0, 1, self._on_d3)
        d4_c, self.edit_d4 = self._make_field("d4 L3\u2192screen [mm]", self.xscreen - self.x3, 5.0, 1, self._on_d4)
        grid.addWidget(d1_c, 2, 0)
        grid.addWidget(d2_c, 2, 1)
        grid.addWidget(d3_c, 2, 2)
        grid.addWidget(d4_c, 2, 3)

        row_fields = QHBoxLayout()
        row_fields.addLayout(grid)
        row_fields.addStretch(1)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        self.btn_confocal = QPushButton("Set confocal  (d = sum of current f)")
        self.btn_reset = QPushButton("Reset  (f1/f2/f3 \u2192 initial, d = sum of f)")
        self.btn_confocal.clicked.connect(self._on_confocal_click)
        self.btn_reset.clicked.connect(self._on_reset_click)
        btn_row.addWidget(self.btn_confocal)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)

        left_col = QVBoxLayout()
        left_col.addLayout(row_fields)
        left_col.addLayout(btn_row)
        left_col.addStretch(1)

        # ---- Bottom row: fields+buttons on the left, stats panel on the right ----
        lower_row = QHBoxLayout()
        lower_row.setSpacing(16)
        lower_row.addLayout(left_col, stretch=2)

        self.stats_box = self._build_stats_box()
        lower_row.addWidget(self.stats_box, stretch=1)

        root.addLayout(lower_row)

    # -------------------- input field handlers --------------------
    def _on_f1(self):
        try:
            self._apply_f1(float(self.edit_f1.text()))
            self.redraw()
        except ValueError:
            pass

    def _on_f2(self):
        try:
            self._apply_f2(float(self.edit_f2.text()))
            self.redraw()
        except ValueError:
            pass

    def _on_f3(self):
        try:
            self._apply_f3(float(self.edit_f3.text()))
            self.redraw()
        except ValueError:
            pass

    def _on_w0(self):
        try:
            self.w0_mm = float(self.edit_w0.text())
            self.redraw()
        except ValueError:
            pass

    def _on_theta(self):
        try:
            self.theta0_mrad = float(self.edit_theta.text())
            self.redraw()
        except ValueError:
            pass

    def _on_wl(self):
        try:
            self.wavelength_nm = float(self.edit_wl.text())
            self.redraw()
        except ValueError:
            pass

    def _on_atom(self):
        try:
            self.atom_size_um = float(self.edit_atom.text())
            self.redraw()
        except ValueError:
            pass

    def _on_d1(self):
        try:
            self._set_d1(float(self.edit_d1.text()))
            self.redraw()
        except ValueError:
            pass

    def _on_d2(self):
        try:
            self._set_d2(float(self.edit_d2.text()))
            self.redraw()
        except ValueError:
            pass

    def _on_d3(self):
        try:
            self._set_d3(float(self.edit_d3.text()))
            self.redraw()
        except ValueError:
            pass

    def _on_d4(self):
        try:
            self._set_d4(float(self.edit_d4.text()))
            self.redraw()
        except ValueError:
            pass

    def _on_confocal_click(self):
        self.set_confocal(full_reset=True)
        self.redraw()

    def _on_reset_click(self):
        self.f1, self.f2, self.f3 = self.f1_default, self.f2_default, self.f3_default
        self.edit_f1.setText(f"{self.f1:.2f}")
        self.edit_f2.setText(f"{self.f2:.2f}")
        self.edit_f3.setText(f"{self.f3:.2f}")
        self.set_confocal(full_reset=True)
        self.redraw()

    # -------------------- drag & drop in the plot --------------------
    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        candidates = {"x1": self.x1, "x2": self.x2, "x3": self.x3, "xscreen": self.xscreen}
        xr = self.ax.get_xlim()
        tol = (xr[1] - xr[0]) * 0.015
        nearest = min(candidates, key=lambda k: abs(candidates[k] - event.xdata))
        if abs(candidates[nearest] - event.xdata) < tol:
            self.dragging = nearest

    def _on_motion(self, event):
        if self.dragging is None or event.inaxes != self.ax or event.xdata is None:
            return
        val = float(event.xdata)
        g = self.GAP
        if self.dragging == "x1":
            self.x1 = max(g, min(val, self.x2 - g))
        elif self.dragging == "x2":
            self.x2 = max(self.x1 + g, min(val, self.x3 - g))
        elif self.dragging == "x3":
            self.x3 = max(self.x2 + g, min(val, self.xscreen - g))
        elif self.dragging == "xscreen":
            self.xscreen = max(self.x3 + g, val)
        self.redraw()

    def _on_release(self, event):
        self.dragging = None

    # -------------------- drawing --------------------
    def _draw_lens(self, cx, top, bottom, converging, color="#8fd3ff"):
        cap = (bottom - top) * 0.08
        d = 1 if converging else -1
        halfw = (bottom - top) * 0.04
        self.ax.plot([cx, cx], [top, bottom], color=color, lw=2.4, solid_capstyle="round")
        self.ax.plot([cx - halfw, cx], [top + d * cap, top], color=color, lw=2.4, solid_capstyle="round")
        self.ax.plot([cx + halfw, cx], [top + d * cap, top], color=color, lw=2.4, solid_capstyle="round")
        self.ax.plot([cx - halfw, cx], [bottom - d * cap, bottom], color=color, lw=2.4, solid_capstyle="round")
        self.ax.plot([cx + halfw, cx], [bottom - d * cap, bottom], color=color, lw=2.4, solid_capstyle="round")

    def _update_standard_labels(self):
        def status(f):
            if f in STANDARD_F_MM:
                return "\u2713 standard value", "#5eead4"
            return f"not a standard value (nearest: {nearest_standard(f)}mm)", "#c9a227"
        t1, c1 = status(self.f1)
        t2, c2 = status(self.f2)
        self.lbl_f1_std.setText(t1)
        self.lbl_f1_std.setStyleSheet(f"font-size: 10px; color: {c1};")
        self.lbl_f2_std.setText(t2)
        self.lbl_f2_std.setStyleSheet(f"font-size: 10px; color: {c2};")

    def _sync_distance_fields(self):
        self.edit_d1.setText(f"{self.x1:.1f}")
        self.edit_d2.setText(f"{self.x2 - self.x1:.1f}")
        self.edit_d3.setText(f"{self.x3 - self.x2:.1f}")
        self.edit_d4.setText(f"{self.xscreen - self.x3:.1f}")

    def redraw(self):
        self._update_standard_labels()
        self._sync_distance_fields()

        self.ax.clear()
        self.ax.set_facecolor("#1b1e23")

        if not self.positions_ok():
            self.ax.text(0.5, 0.5, "Invalid arrangement", transform=self.ax.transAxes,
                          ha="center", color="white")
            self.canvas.draw_idle()
            return

        beam = compute_beam(self.x1, self.x2, self.x3, self.xscreen,
                             self.f1, self.f2, self.f3,
                             self.w0_mm, self.wavelength_nm, self.theta0_mrad)
        color = wavelength_to_rgb(self.wavelength_nm)

        z, w, y = beam["z"], beam["w"], beam["y"]
        upper, lower = y + w, y - w

        self.ax.axhline(0, color="#454c56", ls="--", lw=1, zorder=1)
        self.ax.fill_between(z, lower, upper, color=color, alpha=0.25, zorder=2)
        self.ax.plot(z, upper, color=color, lw=1.2, zorder=3)
        self.ax.plot(z, lower, color=color, lw=1.2, zorder=3)
        self.ax.plot(z, y, color=color, lw=1, ls=":", zorder=3)

        self.ax.plot([0], [0], "o", color="#e8eaed", ms=5, zorder=4)
        self.ax.text(0, self.ax.get_ylim()[0], "Source", color="#9aa1ab", fontsize=8,
                     ha="center", va="top")

        yr = max(np.max(upper) - np.min(lower), 1e-6)
        for i, info in enumerate(beam["lenses"]):
            half = max(yr * 0.12, info["w"] * 1.6)
            top, bottom = info["y"] + half, info["y"] - half
            self._draw_lens(info["x"], top, bottom, converging=info["f"] > 0)
            self.ax.text(info["x"], bottom, f"L{i+1}: f={info['f']:g}mm\nx={info['x']:.0f}mm",
                         color="#9aa1ab", fontsize=8, ha="center", va="top")

        ylim = self.ax.get_ylim()
        self.ax.axvline(self.xscreen, color="#e8b23a", lw=2.5, zorder=3)
        self.ax.text(self.xscreen, ylim[0], "Screen", color="#9aa1ab", fontsize=8,
                     ha="center", va="top")

        self.ax.set_xlabel("z [mm]", color="#9aa1ab")
        self.ax.set_ylabel("y [mm]  (vertical exaggerated)", color="#9aa1ab")
        self.ax.tick_params(colors="#9aa1ab")
        for spine in self.ax.spines.values():
            spine.set_color("#333944")

        M = self._system_matrix_to_screen()
        B_mm_per_rad = M[0, 1]
        full_angle_rad = 43e-3   # full angle, not the half angle
        scan_full_um = abs(B_mm_per_rad) * full_angle_rad * 1000.0
        scan_full_atoms = scan_full_um / self.atom_size_um if self.atom_size_um != 0 else float("nan")

        w_screen_um = beam["final"][0] * 1000.0
        central_profile_um = 2.0 * w_screen_um

        self.lbl_stat_zr.setText(f"{beam['zR']:.1f} mm")
        self.lbl_stat_wscreen.setText(f"{w_screen_um:.4f} \u00b5m")
        self.lbl_stat_wmin.setText(f"{beam['waist_min'][1]*1000:.1f} \u00b5m  @ z={beam['waist_min'][0]:.0f} mm")
        self.lbl_stat_yscreen.setText(f"{beam['final'][1]*1000:.4f} \u00b5m")
        self.lbl_stat_central.setText(f"{central_profile_um:.4f} \u00b5m")
        self.lbl_stat_scan.setText(
            f"{scan_full_um:.4f} \u00b5m   (at 43 mrad full angle)\n"
            f"\u00f7 {self.atom_size_um:g} \u00b5m:  {scan_full_atoms:.4f}"
        )

        self.fig.tight_layout()
        self.canvas.draw_idle()

        # Live-notify a parent/host application (e.g. FlatMultiTone GUI)
        self.optics_changed.emit({
            "f1_mm": self.f1, "f2_mm": self.f2, "f3_mm": self.f3,
            "w0_mm": self.w0_mm,
            "wavelength_nm": self.wavelength_nm,
            "atom_size_um": self.atom_size_um,
            "w_screen_um": w_screen_um,
            "central_profile_um": central_profile_um,
            "scan_range_um": scan_full_um,
            "scan_range_atoms": scan_full_atoms,
        })


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = LensDesignDialog()
    dlg.show()
    sys.exit(app.exec_())