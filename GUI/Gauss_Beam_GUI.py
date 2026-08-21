"""
Interaktive Gaußstrahl-Propagation durch ein 3-Linsen-System (PyQt5-Version).
Linsen & Schirm per Drag & Drop verschiebbar, alle Werte frei eingebbar (Mausrad = skalieren).

Ausführen: python3 gauss_beam_gui_pyqt5.py
Benötigt:  pip install PyQt5 matplotlib numpy
"""

import sys
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLineEdit, QLabel, QPushButton, QSizePolicy,
)
from PyQt5.QtCore import Qt

# ============================================================
# Standardbrennweiten (mm) - nur als Hinweis, keine Einschränkung
# ============================================================
STANDARD_F_MM = [
    -1000, -750, -500, -300, -200, -150, -100, -75, -50, -40, -30, -25,
    25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400,
    500, 600, 750, 1000, 1500, 2000,
]

def nearest_standard(value):
    return min(STANDARD_F_MM, key=lambda f: abs(f - value))

# ============================================================
# Wellenlänge (nm) -> RGB (0..1), für physikalisch passende Strahlfarbe
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
# Gaußstrahl-Physik (q-Parameter als Python complex, ABCD)
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
# Qt-Stylesheet (dunkles Theme, gilt für alle Widgets)
# ============================================================
DARK_QSS = """
QMainWindow, QWidget { background-color: #141619; color: #e8eaed; font-size: 12px; }
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
# QLineEdit mit Mausrad-Unterstützung (scrollen = Wert skalieren)
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
        self.editingFinished.emit()  # löst denselben Handler wie manuelles Eintippen aus
        event.accept()

# ============================================================
# Hauptfenster
# ============================================================
class MainWindow(QMainWindow):
    GAP = 15.0  # mm Mindestabstand zwischen Elementen

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaußstrahl · 3-Linsen-System")
        self.setStyleSheet(DARK_QSS)

        # -------- Default-Zustand --------
        self.f1_default, self.f2_default, self.f3_default = 75.0, 750.0, 52.88
        self.f1, self.f2, self.f3 = self.f1_default, self.f2_default, self.f3_default
        self.w0_mm = 1.0
        self.theta0_mrad = 3.0
        self.wavelength_nm = 795.0
        self.atom_size_um = 5.288
        self.dragging = None

        self.x1 = self.f1
        self.x2 = self.x1 + self.f1 + self.f2
        self.x3 = self.x2 + self.f2 + self.f3
        self.xscreen = self.x3 + self.f3

        self._build_ui()
        self.redraw()
        self.resize(1200, 860)

    # -------------------- Zustand / Physik-Hilfsfunktionen --------------------
    def set_confocal(self, full_reset=False):
        """full_reset=True: d1=f1, d2=f1+f2, d3=f2+f3, d4=f3.
        full_reset=False: x1 und d4 bleiben erhalten, nur d2/d3 neu aus f1,f2,f3."""
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
        """Gesamte ABCD-Matrix von der Quelle (z=0) bis zum Schirm.
        y_Schirm = A*y0 + B*theta0. Da y0=0 (Strahl startet auf der Achse),
        gilt y_Schirm = B*theta0 -> B beschreibt direkt die Winkel->Position-Verstärkung."""
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

        panel_title = QLabel("Strahlgröße &amp; Scanning Range")
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
            vlbl = QLabel("–")
            vlbl.setStyleSheet("font-size: 13px; font-weight: 600; color: #5eead4; "
                                "font-family: monospace; border: none;")
            vlbl.setWordWrap(True)
            v.addWidget(klbl)
            v.addWidget(vlbl)
            stats_grid.addWidget(cell, row, col, 1, colspan)
            return vlbl

        self.lbl_stat_zr = add_stat(1, 0, "Rayleigh-Länge z_R")
        self.lbl_stat_wscreen = add_stat(1, 1, "Strahlradius am Schirm")
        self.lbl_stat_wmin = add_stat(2, 0, "Minimale Strahlbreite (Position)")
        self.lbl_stat_yscreen = add_stat(2, 1, "Versatz am Schirm (durch \u03b80)")
        self.lbl_stat_scan = add_stat(3, 0, "Scanning Range (43 mrad Vollwinkel, am Schirm)", colspan=2)
        stats_grid.setColumnMinimumWidth(0, 210)
        stats_grid.setColumnMinimumWidth(1, 210)
        return box

    # -------------------- UI-Aufbau --------------------
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
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        header = QVBoxLayout()
        title = QLabel("Gaußstrahl · 3-Linsen-System")
        title.setStyleSheet("font-size: 16px; font-weight: 600; color: #e8eaed;")
        header.addWidget(title)

        hint = QLabel("Linsen & Schirm im Diagramm per Drag & Drop verschieben. "
                       "Mausrad über einem Feld skaliert dessen Wert. "
                       "Vertikale Achse stark überhöht dargestellt.")
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

        # ---- Eingabefelder: f1, f2, f3, w0 ----
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(10)

        f1_c, self.edit_f1 = self._make_field("f1 [mm]", self.f1, 1.0, 2, self._on_f1)
        f2_c, self.edit_f2 = self._make_field("f2 [mm]", self.f2, 1.0, 2, self._on_f2)
        f3_c, self.edit_f3 = self._make_field("f3 [mm]", self.f3, 0.5, 2, self._on_f3)
        w0_c, self.edit_w0 = self._make_field("w0 Start [mm]", self.w0_mm, 0.05, 3, self._on_w0)

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
        atom_c, self.edit_atom = self._make_field("Atomgröße [µm]", self.atom_size_um, 0.1, 3, self._on_atom)
        grid.addWidget(th_c, 1, 0)
        grid.addWidget(wl_c, 1, 1)
        grid.addWidget(atom_c, 1, 2)

        d1_c, self.edit_d1 = self._make_field("d1 Quelle\u2192L1 [mm]", self.x1, 5.0, 1, self._on_d1)
        d2_c, self.edit_d2 = self._make_field("d2 L1\u2192L2 [mm]", self.x2 - self.x1, 5.0, 1, self._on_d2)
        d3_c, self.edit_d3 = self._make_field("d3 L2\u2192L3 [mm]", self.x3 - self.x2, 5.0, 1, self._on_d3)
        d4_c, self.edit_d4 = self._make_field("d4 L3\u2192Schirm [mm]", self.xscreen - self.x3, 5.0, 1, self._on_d4)
        grid.addWidget(d1_c, 2, 0)
        grid.addWidget(d2_c, 2, 1)
        grid.addWidget(d3_c, 2, 2)
        grid.addWidget(d4_c, 2, 3)

        row_fields = QHBoxLayout()
        row_fields.addLayout(grid)
        row_fields.addStretch(1)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        self.btn_confocal = QPushButton("Konfokal setzen  (d = Summe der aktuellen f)")
        self.btn_reset = QPushButton("Zurücksetzen  (f1/f2/f3 → Standard, d = Summe der f)")
        self.btn_confocal.clicked.connect(self._on_confocal_click)
        self.btn_reset.clicked.connect(self._on_reset_click)
        btn_row.addWidget(self.btn_confocal)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)

        left_col = QVBoxLayout()
        left_col.addLayout(row_fields)
        left_col.addLayout(btn_row)
        left_col.addStretch(1)

        # ---- Untere Zeile: Felder+Buttons links, Kennzahlen-Panel im freien Bereich rechts ----
        lower_row = QHBoxLayout()
        lower_row.setSpacing(16)
        lower_row.addLayout(left_col, stretch=2)

        self.stats_box = self._build_stats_box()
        lower_row.addWidget(self.stats_box, stretch=1)

        root.addLayout(lower_row)

    # -------------------- Eingabefeld-Handler --------------------
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

    # -------------------- Drag & Drop im Plot --------------------
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

    # -------------------- Zeichnen --------------------
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
                return "✓ Standardwert", "#5eead4"
            return f"kein Standardwert (nächster: {nearest_standard(f)}mm)", "#c9a227"
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
            self.ax.text(0.5, 0.5, "Ungültige Anordnung", transform=self.ax.transAxes,
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
        self.ax.text(0, self.ax.get_ylim()[0], "Quelle", color="#9aa1ab", fontsize=8,
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
        self.ax.text(self.xscreen, ylim[0], "Schirm", color="#9aa1ab", fontsize=8,
                     ha="center", va="top")

        self.ax.set_xlabel("z [mm]", color="#9aa1ab")
        self.ax.set_ylabel("y [mm]  (vertikal überhöht)", color="#9aa1ab")
        self.ax.tick_params(colors="#9aa1ab")
        for spine in self.ax.spines.values():
            spine.set_color("#333944")

        d1, d2, d3, d4 = self.x1, self.x2 - self.x1, self.x3 - self.x2, self.xscreen - self.x3

        M = self._system_matrix_to_screen()
        B_mm_per_rad = M[0, 1]
        full_angle_rad = 43e-3   # voller Winkel, nicht der Halbwinkel
        scan_full_um = abs(B_mm_per_rad) * full_angle_rad * 1000.0
        scan_full_atoms = scan_full_um / self.atom_size_um if self.atom_size_um != 0 else float("nan")

        self.lbl_stat_zr.setText(f"{beam['zR']:.1f} mm")
        self.lbl_stat_wscreen.setText(f"{beam['final'][0]*1000:.4f} µm")
        self.lbl_stat_wmin.setText(f"{beam['waist_min'][1]*1000:.1f} µm  @ z={beam['waist_min'][0]:.0f} mm")
        self.lbl_stat_yscreen.setText(f"{beam['final'][1]*1000:.4f} µm")
        self.lbl_stat_scan.setText(
            f"{scan_full_um:.4f} \u00b5m   (bei 43 mrad Vollwinkel)\n"
            f"\u00f7 {self.atom_size_um:g} \u00b5m:  {scan_full_atoms:.4f}"
        )

        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())