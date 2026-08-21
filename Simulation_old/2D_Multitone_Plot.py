"""
Start dialog for the 2D scan (input waist before 1st lens vs. width -> uniformity).

At program start, asks for the following parameters via an input dialog:
- Tone count N_x, N_y
- Amplitudes per axis (amp_x, amp_y)
- Scan ranges for win_input (input waist before the 1st lens, in mm) and
  width (frequency spacing, in MHz)
- Number of grid points per axis

f1 = 75 mm and f2 = 750 mm are fixed (default values of the optimizer, see
MultitoneFlatTopOptimizer.DEFAULTS).

After confirmation, MultitoneFlatTopOptimizer.scan_win_width_uniformity() is
called and the uniformity is plotted as a 2D heatmap (win_input x width).

Usage:
    python WinWidthScan_StartDialog.py
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog,
)
from PyQt5.QtCore import Qt

from Strahlanalyse.AOD_Simulation.Multitone_Simulation import MultitoneFlatTopOptimizer


class StartParametersDialog(QDialog):
    """Asks for N_x, N_y, amplitudes, and the scan ranges for win_input/width."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multitone FlatTop - 2D Scan Parameters")

        self.amp_x_boxes = []
        self.amp_y_boxes = []

        main_layout = QVBoxLayout(self)

        # -- fixed optical parameters (display only) --
        info_label = QLabel("Fixed optical parameters: f1 = 75 mm, f2 = 750 mm")
        info_label.setStyleSheet("font-style: italic;")
        main_layout.addWidget(info_label)

        # -- tone count --
        tone_group = QGroupBox("Tone Count")
        tone_layout = QFormLayout()
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(1, 10)
        self.nx_spin.setValue(3)
        self.nx_spin.valueChanged.connect(self._rebuild_amp_fields)
        self.ny_spin = QSpinBox()
        self.ny_spin.setRange(1, 10)
        self.ny_spin.setValue(4)
        self.ny_spin.valueChanged.connect(self._rebuild_amp_fields)
        tone_layout.addRow("N_x:", self.nx_spin)
        tone_layout.addRow("N_y:", self.ny_spin)
        tone_group.setLayout(tone_layout)
        main_layout.addWidget(tone_group)

        # -- amplitudes --
        self.amp_group = QGroupBox("Amplitudes per Axis (weight, 0 - 2)")
        self.amp_layout = QGridLayout()
        self.amp_group.setLayout(self.amp_layout)
        main_layout.addWidget(self.amp_group)
        self._rebuild_amp_fields()

        # -- scan ranges --
        scan_group = QGroupBox("Scan Ranges")
        scan_layout = QFormLayout()

        self.win_input_min = self._make_spin(0.5, 0.001, 50.0, 0.05)
        self.win_input_max = self._make_spin(3.5, 0.001, 50.0, 0.05)
        scan_layout.addRow("win_input min (mm):", self.win_input_min)
        scan_layout.addRow("win_input max (mm):", self.win_input_max)

        self.width_min = self._make_spin(0.15, 0.001, 5.0, 0.01)
        self.width_max = self._make_spin(0.6, 0.001, 5.0, 0.01)
        scan_layout.addRow("width min (MHz):", self.width_min)
        scan_layout.addRow("width max (MHz):", self.width_max)

        self.n_points = QSpinBox()
        self.n_points.setRange(5, 300)
        self.n_points.setValue(5)
        scan_layout.addRow("Grid points per axis:", self.n_points)

        scan_group.setLayout(scan_layout)
        main_layout.addWidget(scan_group)

        # -- buttons --
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Start Scan")
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)

    @staticmethod
    def _make_spin(value, minimum, maximum, step):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(3)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _rebuild_amp_fields(self):
        # remove old widgets
        while self.amp_layout.count():
            item = self.amp_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.amp_x_boxes = []
        self.amp_y_boxes = []

        n_x = self.nx_spin.value()
        n_y = self.ny_spin.value()

        self.amp_layout.addWidget(QLabel("amp_x:"), 0, 0)
        for i in range(n_x):
            box = self._make_spin(1.0, 0.0, 2.0, 0.05)
            self.amp_layout.addWidget(box, 0, i + 1)
            self.amp_x_boxes.append(box)

        self.amp_layout.addWidget(QLabel("amp_y:"), 1, 0)
        for i in range(n_y):
            box = self._make_spin(1.0, 0.0, 2.0, 0.05)
            self.amp_layout.addWidget(box, 1, i + 1)
            self.amp_y_boxes.append(box)

    def _on_accept(self):
        if self.win_input_min.value() >= self.win_input_max.value():
            QMessageBox.warning(self, "Invalid Range",
                                 "win_input min must be smaller than win_input max.")
            return
        if self.width_min.value() >= self.width_max.value():
            QMessageBox.warning(self, "Invalid Range",
                                 "width min must be smaller than width max.")
            return
        self.accept()

    def get_values(self):
        """Returns all entered parameters as a dict (SI units: meters, Hz)."""
        amp_x = np.array([b.value() for b in self.amp_x_boxes])
        amp_y = np.array([b.value() for b in self.amp_y_boxes])
        return dict(
            N_x=self.nx_spin.value(),
            N_y=self.ny_spin.value(),
            amp_x=amp_x,
            amp_y=amp_y,
            win_input_range=(self.win_input_min.value() * 1e-3, self.win_input_max.value() * 1e-3),
            width_range=(self.width_min.value() * 1e6, self.width_max.value() * 1e6),
            n_points=self.n_points.value(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = StartParametersDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)

    params = dialog.get_values()
    amps = np.concatenate([params["amp_x"], params["amp_y"]])

    opt = MultitoneFlatTopOptimizer(
        out_dir="../AOD_GUIs",
        f1=75e-3,
        f2=750e-3,
        N_x=params["N_x"],
        N_y=params["N_y"],
    )

    total_points = params["n_points"] * params["n_points"]
    progress = QProgressDialog("Computing uniformity & crosstalk scan...", "Cancel", 0, total_points)
    progress.setWindowTitle("2D Scan Running")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)  # show immediately, even for short scans
    progress.setValue(0)

    def on_progress(done, total):
        progress.setValue(done)
        # keep the UI responsive, otherwise the bar freezes during computation
        QApplication.processEvents()
        if progress.wasCanceled():
            return False  # signals scan_win_width_uniformity() to cancel
        return True

    opt.scan_win_width_uniformity(
        win_input_range=params["win_input_range"],
        width_range=params["width_range"],
        n_win_input=params["n_points"],
        n_width=params["n_points"],
        amps=amps,
        show=True,
        save=True,
        progress_callback=on_progress,
    )

    progress.setValue(total_points)


if __name__ == "__main__":
    main()