"""
Gaussstrahl-Propagations-GUI (PySide6 / PyQt6).

Start:  python gui.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import List, Optional

# ---------------------------------------------------------------- Qt-Binding
QT_BINDING = None
try:
    from PySide6 import QtCore, QtGui, QtWidgets
    QT_BINDING = "pyside6"
    os.environ.setdefault("QT_API", "pyside6")
    Signal = QtCore.Signal
except ImportError:  # pragma: no cover
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        QT_BINDING = "pyqt6"
        os.environ.setdefault("QT_API", "pyqt6")
        Signal = QtCore.pyqtSignal
    except ImportError:
        sys.exit("Weder PySide6 noch PyQt6 gefunden.\n"
                 "Bitte installieren:  pip install PySide6 matplotlib numpy")

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from gaussbeam import (COMPONENT_TYPES, GOALS, BeamError, Component, Distance,
                       InputBeam, OpticalSystem, Param, PLANES, ThinLens,
                       optimize)

MM = 1e-3
UM = 1e-6
NM = 1e-9
MRAD = 1e-3


def spin(value=0.0, lo=-1e9, hi=1e9, dec=4, step=1.0, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setDecimals(dec)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(value)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(" " + suffix)
    s.setMinimumWidth(120)
    return s


# ============================================================================
#  Eingangsstrahl-Panel
# ============================================================================
class BeamPanel(QtWidgets.QGroupBox):
    changed = Signal()

    def __init__(self):
        super().__init__("Eingangsstrahl")
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        self.lam = spin(1064.0, 100.0, 20000.0, 2, 10.0, "nm")
        self.n0 = spin(1.0, 1.0, 5.0, 5, 0.01)
        self.w0 = spin(500.0, 0.01, 1e6, 3, 10.0, "um")

        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["ideal gaussisch (M2 = 1)",
                            "M2 vorgeben",
                            "Divergenz (Halbwinkel) vorgeben"])
        self.m2 = spin(1.0, 1.0, 100.0, 3, 0.1)
        self.theta = spin(0.677, 1e-6, 1e5, 5, 0.05, "mrad")

        self.collimated = QtWidgets.QCheckBox("kollimiert (Waist an der 1. Komponente)")
        self.collimated.setChecked(True)
        self.zw = spin(0.0, -1e6, 1e6, 4, 10.0, "mm")
        self.zw.setToolTip("Abstand vom Eingangswaist bis zur ersten Komponente.\n"
                           "0 = der Strahl startet im Waist (kollimiert).")

        self.astig_in = QtWidgets.QCheckBox("Eingang bereits astigmatisch")
        self.w0s = spin(500.0, 0.01, 1e6, 3, 10.0, "um")
        self.zws = spin(0.0, -1e6, 1e6, 4, 10.0, "mm")

        form.addRow("Wellenlaenge (Vakuum)", self.lam)
        form.addRow("Brechzahl vor Optik", self.n0)
        form.addRow("Waist-Radius w0 (1/e2)", self.w0)
        form.addRow("Strahlqualitaet", self.mode)
        form.addRow("M2", self.m2)
        form.addRow("Divergenz (halb)", self.theta)
        form.addRow(self.collimated)
        form.addRow("Abstand Waist -> Optik", self.zw)
        form.addRow(self.astig_in)
        form.addRow("  w0 sagittal", self.w0s)
        form.addRow("  Abstand sagittal", self.zws)

        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("color:#555;")
        self.info.setWordWrap(True)
        form.addRow(self.info)

        for w in (self.lam, self.n0, self.w0, self.m2, self.theta, self.zw,
                  self.w0s, self.zws):
            w.valueChanged.connect(self._emit)
        self.mode.currentIndexChanged.connect(self._emit)
        self.collimated.toggled.connect(self._emit)
        self.astig_in.toggled.connect(self._emit)
        self._sync_enabled()

    def _sync_enabled(self):
        m = self.mode.currentIndex()
        self.m2.setEnabled(m == 1)
        self.theta.setEnabled(m == 2)
        self.zw.setEnabled(not self.collimated.isChecked())
        if self.collimated.isChecked():
            self.zw.blockSignals(True)
            self.zw.setValue(0.0)
            self.zw.blockSignals(False)
        on = self.astig_in.isChecked()
        self.w0s.setEnabled(on)
        self.zws.setEnabled(on and not self.collimated.isChecked())

    def _emit(self, *_):
        self._sync_enabled()
        self.changed.emit()

    def beam(self) -> InputBeam:
        lam = self.lam.value() * NM
        w0 = self.w0.value() * UM
        n0 = self.n0.value()
        m = self.mode.currentIndex()
        if m == 0:
            m2 = 1.0
        elif m == 1:
            m2 = self.m2.value()
        else:
            m2 = math.pi * n0 * w0 * (self.theta.value() * MRAD) / lam
            m2 = max(m2, 1e-9)
        b = InputBeam(wavelength=lam, w0=w0, m2=m2, n0=n0,
                      z_waist=0.0 if self.collimated.isChecked() else self.zw.value() * MM)
        if self.astig_in.isChecked():
            b.w0_sag = self.w0s.value() * UM
            b.z_waist_sag = 0.0 if self.collimated.isChecked() else self.zws.value() * MM
        self.info.setText(
            f"-> M2 = {b.m2:.4g},  Divergenz (halb) = {b.divergence()/MRAD:.4g} mrad,  "
            f"z_R = {b.rayleigh()/MM:.4g} mm")
        return b

    def set_beam(self, b: InputBeam):
        for w in (self.lam, self.n0, self.w0, self.m2, self.theta, self.zw,
                  self.w0s, self.zws):
            w.blockSignals(True)
        self.lam.setValue(b.wavelength / NM)
        self.n0.setValue(b.n0)
        self.w0.setValue(b.w0 / UM)
        self.m2.setValue(b.m2)
        self.mode.setCurrentIndex(0 if abs(b.m2 - 1.0) < 1e-12 else 1)
        self.collimated.setChecked(abs(b.z_waist) < 1e-15)
        self.zw.setValue(b.z_waist / MM)
        self.astig_in.setChecked(b.w0_sag is not None)
        if b.w0_sag is not None:
            self.w0s.setValue(b.w0_sag / UM)
            self.zws.setValue((b.z_waist_sag or 0.0) / MM)
        for w in (self.lam, self.n0, self.w0, self.m2, self.theta, self.zw,
                  self.w0s, self.zws):
            w.blockSignals(False)
        self._emit()


# ============================================================================
#  Parameter-Editor fuer die ausgewaehlte Komponente
# ============================================================================
class ParamEditor(QtWidgets.QGroupBox):
    changed = Signal()

    def __init__(self):
        super().__init__("Parameter")
        self.form = QtWidgets.QFormLayout(self)
        self.widgets = {}
        self.comp: Optional[Component] = None
        self.placeholder = QtWidgets.QLabel("Keine Komponente ausgewaehlt.")
        self.placeholder.setStyleSheet("color:#777;")
        self.form.addRow(self.placeholder)

    def _clear(self):
        while self.form.count():
            item = self.form.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.widgets = {}

    def show_component(self, comp: Optional[Component]):
        self._clear()
        self.comp = comp
        if comp is None:
            lab = QtWidgets.QLabel("Keine Komponente ausgewaehlt.")
            lab.setStyleSheet("color:#777;")
            self.form.addRow(lab)
            self.setTitle("Parameter")
            return
        self.setTitle(f"Parameter - {comp.label}")
        for p in comp.params:
            s = spin(p.to_display(comp.get(p.key)), p.vmin, p.vmax, p.decimals,
                     1.0, "" if p.unit == "-" else p.unit)
            if p.tip:
                s.setToolTip(p.tip)
            s.valueChanged.connect(lambda v, key=p.key: self._on_change(key, v))
            self.widgets[p.key] = s
            self.form.addRow(p.label, s)

    def _on_change(self, key, value):
        if self.comp is None:
            return
        self.comp.set(key, self.comp.param(key).to_si(value))
        self.changed.emit()

    def refresh_values(self):
        if self.comp is None:
            return
        for key, w in self.widgets.items():
            w.blockSignals(True)
            w.setValue(self.comp.param(key).to_display(self.comp.get(key)))
            w.blockSignals(False)


# ============================================================================
#  Optimierungs-Dialog
# ============================================================================
class OptimizeDialog(QtWidgets.QDialog):
    def __init__(self, system: OpticalSystem, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ziel-Optimierung")
        self.system = system
        self.result_value: Optional[float] = None

        lay = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        lay.addLayout(form)

        self.comp_box = QtWidgets.QComboBox()
        for i, c in enumerate(system.components):
            self.comp_box.addItem(f"{i+1}. {c.label}", i)
        self.par_box = QtWidgets.QComboBox()
        self.lo = spin(0.0, -1e9, 1e9, 4, 1.0)
        self.hi = spin(200.0, -1e9, 1e9, 4, 1.0)

        self.goal_box = QtWidgets.QComboBox()
        for k, v in GOALS.items():
            self.goal_box.addItem(v, k)
        self.target = spin(100.0, -1e9, 1e9, 4, 1.0)
        self.plane_box = QtWidgets.QComboBox()
        self.plane_box.addItems(["tangential", "sagittal", "mittel"])

        form.addRow("Komponente", self.comp_box)
        form.addRow("freier Parameter", self.par_box)
        form.addRow("Suchbereich von", self.lo)
        form.addRow("bis", self.hi)
        form.addRow("Ziel", self.goal_box)
        form.addRow("Zielwert", self.target)
        form.addRow("Ebene", self.plane_box)

        self.out = QtWidgets.QLabel("-")
        self.out.setWordWrap(True)
        self.out.setStyleSheet("font-family: monospace;")
        lay.addWidget(self.out)

        btns = QtWidgets.QDialogButtonBox()
        self.run_btn = btns.addButton("Suchen", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.apply_btn = btns.addButton("Uebernehmen", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        btns.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.apply_btn.setEnabled(False)
        btns.rejected.connect(self.reject)
        self.run_btn.clicked.connect(self.run)
        self.apply_btn.clicked.connect(self.accept)
        lay.addWidget(btns)

        self.comp_box.currentIndexChanged.connect(self._fill_params)
        self.goal_box.currentIndexChanged.connect(self._sync_target)
        self._fill_params()
        self._sync_target()

    def _fill_params(self):
        self.par_box.clear()
        i = self.comp_box.currentData()
        if i is None:
            return
        for p in self.system.components[i].params:
            self.par_box.addItem(f"{p.label} [{p.unit}]", p.key)
        p0 = self.system.components[i].params[0]
        cur = p0.to_display(self.system.components[i].get(p0.key))
        self.lo.setValue(min(0.0, cur))
        self.hi.setValue(max(cur * 3.0 if cur else 100.0, cur + 100.0))

    def _sync_target(self):
        goal = self.goal_box.currentData()
        units = {"waist_size": "um", "w_end": "um", "waist_pos": "mm"}
        u = units.get(goal)
        self.target.setEnabled(u is not None)
        self.target.setSuffix(" " + u if u else "")

    def _target_si(self):
        goal = self.goal_box.currentData()
        if goal in ("waist_size", "w_end"):
            return self.target.value() * UM
        if goal == "waist_pos":
            return self.target.value() * MM
        return 0.0

    def run(self):
        i = self.comp_box.currentData()
        key = self.par_box.currentData()
        if i is None or key is None:
            return
        comp = self.system.components[i]
        p = comp.param(key)
        lo, hi = sorted((p.to_si(self.lo.value()), p.to_si(self.hi.value())))
        try:
            res = optimize(self.system, i, key, lo, hi,
                           goal=self.goal_box.currentData(),
                           target=self._target_si(),
                           plane=self.plane_box.currentText())
        except Exception as exc:
            self.out.setText(f"Fehler: {exc}")
            return
        if not res.ok or not math.isfinite(res.cost):
            self.out.setText(res.message)
            self.apply_btn.setEnabled(False)
            return
        self.result_index = i
        self.result_key = key
        self.result_value = res.value
        goal = self.goal_box.currentData()
        if goal in ("waist_size", "w_end", "round"):
            err, unit, scale = res.cost / UM, "um", UM
        elif goal in ("waist_pos", "collimate"):
            err, unit, scale = res.cost / MM, "mm", MM
        else:
            err, unit, scale = res.cost / UM, "um", UM
        msg = (f"Optimum: {p.label} = {p.to_display(res.value):.6g} {p.unit}\n"
               f"Restabweichung = {err:.4g} {unit}")
        ref = {"waist_size": self.target.value(), "w_end": self.target.value()}.get(goal)
        if abs(res.value - lo) < 1e-12 or abs(res.value - hi) < 1e-12:
            msg += "\nACHTUNG: Optimum liegt am Rand des Suchbereichs - Bereich erweitern."
        if goal not in ("min_w_end",) and err > (abs(ref) * 0.05 if ref else 0.5):
            msg += "\nACHTUNG: Ziel wird nicht getroffen - anderer Parameter noetig?"
        self.out.setText(msg)
        self.apply_btn.setEnabled(True)


# ============================================================================
#  Hauptfenster
# ============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaussstrahl-Propagation")
        self.resize(1400, 860)
        self.system = OpticalSystem()
        self._build()
        self._demo()
        self.recalc()

    # ------------------------------------------------------------------ UI
    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # ---- linke Spalte
        left = QtWidgets.QVBoxLayout()
        left_w = QtWidgets.QWidget()
        left_w.setLayout(left)
        left_w.setMinimumWidth(430)
        left_w.setMaximumWidth(520)

        self.beam_panel = BeamPanel()
        self.beam_panel.changed.connect(self.recalc)
        left.addWidget(self.beam_panel)

        comp_box = QtWidgets.QGroupBox("Komponenten (Reihenfolge = Strahlweg)")
        cl = QtWidgets.QVBoxLayout(comp_box)
        self.list = QtWidgets.QListWidget()
        self.list.currentRowChanged.connect(self._sel_changed)
        self.list.itemChanged.connect(self._item_changed)
        cl.addWidget(self.list)

        row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QToolButton()
        self.add_btn.setText("Hinzufuegen")
        self.add_btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(self.add_btn)
        for kind, cls in COMPONENT_TYPES.items():
            act = menu.addAction(cls.label)
            act.triggered.connect(lambda _=False, k=kind: self.add_component(k))
        self.add_btn.setMenu(menu)
        row.addWidget(self.add_btn)
        for text, slot in (("Entfernen", self.remove_component),
                           ("Hoch", lambda: self.move_component(-1)),
                           ("Runter", lambda: self.move_component(+1)),
                           ("Kopie", self.duplicate_component)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        cl.addLayout(row)
        left.addWidget(comp_box, 1)

        self.editor = ParamEditor()
        self.editor.changed.connect(self.recalc)
        left.addWidget(self.editor)
        root.addWidget(left_w)

        # ---- rechte Spalte
        right = QtWidgets.QVBoxLayout()
        self.fig = Figure(figsize=(7, 4.2), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        toprow = QtWidgets.QHBoxLayout()
        toprow.addWidget(NavToolbar(self.canvas, self))
        self.show_tail = QtWidgets.QCheckBox("Freiraum hinter dem letzten Element mitzeichnen")
        self.show_tail.setChecked(True)
        self.show_tail.toggled.connect(self.recalc)
        toprow.addWidget(self.show_tail)
        toprow.addStretch(1)
        right.addLayout(toprow)
        right.addWidget(self.canvas, 3)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Groesse", "tangential", "sagittal"])
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(250)
        right.addWidget(self.table, 2)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        right.addWidget(self.status)
        root.addLayout(right, 1)

        # ---- Toolbar
        tb = self.addToolBar("Aktionen")
        tb.setMovable(False)
        for text, slot in (("Neu berechnen", self.recalc),
                           ("Ziel-Optimierung...", self.open_optimizer),
                           ("Speichern...", self.save),
                           ("Laden...", self.load),
                           ("Plot exportieren...", self.export_plot)):
            a = QtGui.QAction(text, self)
            a.triggered.connect(slot)
            tb.addAction(a)

    def _demo(self):
        self.system.components = [
            Distance(L=200.0),
            ThinLens(f=100.0),
            Distance(L=150.0),
            COMPONENT_TYPES["crystal"](t=10.0, n=1.76, theta=30.0),
            Distance(L=100.0),
        ]
        self._refill_list()

    # -------------------------------------------------------- Listenpflege
    def _refill_list(self, select: int = 0):
        self.list.blockSignals(True)
        self.list.clear()
        for i, c in enumerate(self.system.components):
            it = QtWidgets.QListWidgetItem(f"{i+1}. {c.label}  -  {c.summary()}")
            it.setFlags(it.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.CheckState.Checked if c.enabled
                             else QtCore.Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        if self.system.components:
            self.list.setCurrentRow(max(0, min(select, len(self.system.components) - 1)))
        else:
            self.editor.show_component(None)

    def _update_list_texts(self):
        self.list.blockSignals(True)
        for i, c in enumerate(self.system.components):
            it = self.list.item(i)
            if it is not None:
                it.setText(f"{i+1}. {c.label}  -  {c.summary()}")
        self.list.blockSignals(False)

    def _sel_changed(self, row):
        if 0 <= row < len(self.system.components):
            self.editor.show_component(self.system.components[row])
        else:
            self.editor.show_component(None)

    def _item_changed(self, item):
        row = self.list.row(item)
        if 0 <= row < len(self.system.components):
            self.system.components[row].enabled = (
                item.checkState() == QtCore.Qt.CheckState.Checked)
            self.recalc()

    def add_component(self, kind: str):
        comp = COMPONENT_TYPES[kind]()
        row = self.list.currentRow()
        idx = row + 1 if row >= 0 else len(self.system.components)
        self.system.components.insert(idx, comp)
        self._refill_list(idx)
        self.recalc()

    def duplicate_component(self):
        row = self.list.currentRow()
        if row < 0:
            return
        c = Component.from_dict(self.system.components[row].to_dict())
        self.system.components.insert(row + 1, c)
        self._refill_list(row + 1)
        self.recalc()

    def remove_component(self):
        row = self.list.currentRow()
        if row < 0:
            return
        self.system.components.pop(row)
        self._refill_list(row)
        self.recalc()

    def move_component(self, delta):
        row = self.list.currentRow()
        new = row + delta
        if row < 0 or not (0 <= new < len(self.system.components)):
            return
        cs = self.system.components
        cs[row], cs[new] = cs[new], cs[row]
        self._refill_list(new)
        self.recalc()

    # -------------------------------------------------------------- Rechnen
    def recalc(self):
        self.system.beam = self.beam_panel.beam()
        self._update_list_texts()
        try:
            res = self.system.propagate()
        except (BeamError, ValueError, ZeroDivisionError) as exc:
            self.status.setText(f"<b style='color:#b00'>Fehler:</b> {exc}")
            return
        self.status.setText("")
        self._fill_table(res)
        self._plot(res)

    def _fill_table(self, res):
        rows = [
            ("Strahlradius w am Ausgang [um]", res.tangential.w / UM, res.sagittal.w / UM),
            ("Waist-Radius w0 [um]", res.tangential.w0 / UM, res.sagittal.w0 / UM),
            ("Waist-Durchmesser 2w0 [um]", 2 * res.tangential.w0 / UM, 2 * res.sagittal.w0 / UM),
            ("Abstand Ausgang -> Waist [mm]", res.tangential.z_to_waist / MM,
             res.sagittal.z_to_waist / MM),
            ("Rayleigh-Laenge zR [mm]", res.tangential.z_rayleigh / MM,
             res.sagittal.z_rayleigh / MM),
            ("Divergenz halb [mrad]", res.tangential.theta / MRAD, res.sagittal.theta / MRAD),
            ("Divergenz voll [mrad]", 2 * res.tangential.theta / MRAD,
             2 * res.sagittal.theta / MRAD),
            ("Divergenz halb [Grad]", math.degrees(res.tangential.theta),
             math.degrees(res.sagittal.theta)),
            ("Phasenfront-Radius R [mm]", res.tangential.R / MM, res.sagittal.R / MM),
        ]
        extra = [
            ("Astigmatismus (Waistabstand t-s) [mm]", res.astigmatism / MM, None),
            ("Brechzahl am Ausgang", res.n_out, None),
            ("optische Weglaenge gesamt [mm]", res.z_total / MM, None),
        ]
        self.table.setRowCount(len(rows) + len(extra))
        for r, (name, t, s) in enumerate(rows + extra):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            for col, v in ((1, t), (2, s)):
                if v is None:
                    txt = ""
                elif not math.isfinite(v):
                    txt = "unendlich (kollimiert)"
                else:
                    txt = f"{v:.6g}"
                it = QtWidgets.QTableWidgetItem(txt)
                it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight |
                                    QtCore.Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, col, it)

    def _plot(self, res):
        from gaussbeam import w_of_q
        self.ax.clear()
        z = list(res.samples.z)
        wt = list(res.samples.wt)
        ws = list(res.samples.ws)

        # Nachlauf hinter der letzten Komponente, damit ein Waist im Freiraum
        # sichtbar wird
        tail = 0.0
        if self.show_tail.isChecked():
            cand = max(res.tangential.z_to_waist + 2 * res.tangential.z_rayleigh,
                       res.sagittal.z_to_waist + 2 * res.sagittal.z_rayleigh, 0.0)
            tail = min(cand, 20.0 * max(res.z_total, 1e-3))
        if tail > 0:
            for dz in np.linspace(0.0, tail, 400)[1:]:
                z.append(res.z_total + dz)
                wt.append(w_of_q(res.q_t + dz, res.n_out, res.lam_eff))
                ws.append(w_of_q(res.q_s + dz, res.n_out, res.lam_eff))

        z = np.array(z) / MM
        wt = np.array(wt) / UM
        ws = np.array(ws) / UM
        self.ax.plot(z, wt, color="#1f77b4", lw=1.6, label="tangential (Einfallsebene)")
        self.ax.plot(z, -wt, color="#1f77b4", lw=1.6)
        self.ax.plot(z, ws, color="#d62728", lw=1.3, ls="--", label="sagittal")
        self.ax.plot(z, -ws, color="#d62728", lw=1.3, ls="--")
        self.ax.fill_between(z, -wt, wt, color="#1f77b4", alpha=0.10)
        self.ax.axhline(0, color="#999", lw=0.6)

        zmin, zmax = float(z.min()), float(z.max())
        pad = 0.02 * max(zmax - zmin, 1e-9)
        self.ax.set_xlim(zmin - pad, zmax + pad)
        if tail > 0:
            self.ax.axvline(res.z_total / MM, color="#2ca02c", lw=1.2, ls=":")
            self.ax.text(res.z_total / MM, self.ax.get_ylim()[1], " Ausgang",
                         color="#2ca02c", fontsize=8, va="top")

        seen = set()
        for m in res.markers:
            zz = m.z / MM
            if round(zz, 9) in seen:
                continue
            seen.add(round(zz, 9))
            self.ax.axvline(zz, color="#444", lw=0.8, alpha=0.45)

        offside = []
        ylo, yhi = self.ax.get_ylim()
        for r, col, name, ytxt, va in ((res.tangential, "#1f77b4", "Waist t",
                                        0.80 * yhi, "bottom"),
                                       (res.sagittal, "#d62728", "Waist s",
                                        0.80 * ylo, "top")):
            zw = (res.z_total + r.z_to_waist) / MM
            txt = f"{name}: w0={r.w0/UM:.1f} um @ {r.z_to_waist/MM:+.1f} mm"
            if zmin - pad <= zw <= zmax + pad:
                self.ax.plot([zw], [0], marker="|", ms=16, color=col, zorder=5)
                self.ax.annotate(txt.replace(": ", ":\n"), xy=(zw, 0),
                                 xytext=(zw, ytxt), fontsize=8, color=col,
                                 ha="center", va=va,
                                 arrowprops=dict(arrowstyle="-", color=col,
                                                 lw=0.7, alpha=0.5),
                                 bbox=dict(fc="white", ec="none", alpha=0.75))
            else:
                offside.append(txt)
        if offside:
            self.ax.text(0.01, 0.98, "\n".join(offside), transform=self.ax.transAxes,
                         fontsize=8, va="top", ha="left",
                         bbox=dict(fc="white", ec="#bbb", alpha=0.85))

        self.ax.set_xlabel("z entlang des Strahls [mm]  (0 = erste Komponente)")
        self.ax.set_ylabel("Strahlradius w [um]")
        self.ax.set_title("Strahlradius entlang des Systems")
        self.ax.grid(alpha=0.25)
        self.ax.legend(loc="upper right", fontsize=8)
        self.canvas.draw_idle()

    # ------------------------------------------------------------- Aktionen
    def open_optimizer(self):
        if not self.system.components:
            return
        dlg = OptimizeDialog(self.system, self)
        if dlg.exec() and dlg.result_value is not None:
            self.system.components[dlg.result_index].set(dlg.result_key, dlg.result_value)
            self._refill_list(dlg.result_index)
            self.editor.show_component(self.system.components[dlg.result_index])
            self.recalc()

    def save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Aufbau speichern", "aufbau.json", "JSON (*.json)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.system.to_dict(), fh, indent=2)

    def load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Aufbau laden", "", "JSON (*.json)")
        if not path:
            return
        with open(path, encoding="utf-8") as fh:
            self.system = OpticalSystem.from_dict(json.load(fh))
        self.beam_panel.set_beam(self.system.beam)
        self._refill_list()
        self.recalc()

    def export_plot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Plot exportieren", "strahlverlauf.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.fig.savefig(path, dpi=200)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
