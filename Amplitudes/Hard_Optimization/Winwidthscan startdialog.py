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

After confirmation, MultitoneFlatTopOptimizer.scan_win_width_uniformity()
computes the scan (pure computation, no plotting). The raw results are then
saved to disk (so the scan never has to be re-run just to re-style a plot
later) and handed to multitone_flattop_scan_plots.ScanPlotter, which draws
the uniformity/crosstalk heatmaps side by side.

Usage:
    python WinWidthScan_StartDialog.py
"""

import sys
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QFileDialog, QComboBox, QLineEdit,
)
from PyQt5.QtCore import Qt

from multitone_flattop_optimizer import MultitoneFlatTopOptimizer, DEFAULT_RESULTS_DIR
from multitone_flattop_scan_plots import ScanPlotter, win_input_to_win
import perf_log
import scan_checkpoint

# Feste Optik-Parameter für die Umrechnung win_input (vor der Linse) <->
# win_eff (nach der Linse/am Fokus) in diesem Dialog. f1/f2 sind wie überall
# sonst in diesem Skript fest (siehe info_label unten); fLO/lambda_opt
# kommen aus denselben Defaults, die auch der Optimizer intern benutzt.
_F1 = 75e-3
_F2 = 750e-3
_FLO = MultitoneFlatTopOptimizer.DEFAULTS['fLO']
_LAMBDA_OPT = MultitoneFlatTopOptimizer.DEFAULTS['lambda_opt']


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
        self.scan_layout = scan_layout  # gebraucht, um Zeilen-Labels später umzubenennen

        # Abfrage, auf welchen Waist sich die beiden Felder darunter beziehen
        # sollen - VOR dem eigentlichen Setzen des Bereichs. "vor der Linse"
        # (win_input, mm) ist das bisherige Verhalten; "nach der Linse"
        # (win_eff, µm, am Fokus) rechnet die eingegebenen Werte beim Start
        # des Scans automatisch über win_input_to_win() zurück in win_input -
        # der Scan selbst arbeitet intern immer in win_input.
        self.waist_mode_combo = QComboBox()
        self.waist_mode_combo.addItem("vor der Linse (Input-Waist, mm)", "win_input")
        self.waist_mode_combo.addItem("nach der Linse (Fokus-Waist, µm)", "win_eff")
        self.waist_mode_combo.setToolTip(
            "Legt fest, worauf sich die beiden Felder 'min'/'max' darunter\n"
            "beziehen: der Waist VOR der ersten Linse (win_input, mm) oder\n"
            "der effektive Waist NACH der Linse/am Fokus (win_eff, µm).\n"
            "Beim Wechsel werden die aktuell eingetragenen Werte automatisch\n"
            "umgerechnet (win_eff = K / win_input, daher vertauschen sich\n"
            "dabei min und max)."
        )
        self._current_waist_mode = "win_input"
        self.waist_mode_combo.currentIndexChanged.connect(self._on_waist_mode_changed)
        scan_layout.addRow("Waist-Eingabe bezieht sich auf:", self.waist_mode_combo)

        self.win_input_min = self._make_spin(0.5, 0.001, 50.0, 0.05)
        self.win_input_max = self._make_spin(3.5, 0.001, 50.0, 0.05)
        scan_layout.addRow("win_input min (mm):", self.win_input_min)
        scan_layout.addRow("win_input max (mm):", self.win_input_max)

        self.width_min = self._make_spin(0.15, 0.001, 5.0, 0.01)
        self.width_max = self._make_spin(0.5, 0.001, 5.0, 0.01)
        scan_layout.addRow("width min (MHz):", self.width_min)
        scan_layout.addRow("width max (MHz):", self.width_max)

        self.n_points = QSpinBox()
        self.n_points.setRange(5, 300)
        self.n_points.setValue(40)
        scan_layout.addRow("Grid points per axis:", self.n_points)

        self.n_grid = QSpinBox()
        self.n_grid.setRange(50, 2000)
        self.n_grid.setValue(200)
        self.n_grid.setToolTip(
            "Resolution of the intensity grid used for EACH scan point.\n"
            "Cost scales as n_grid^2 per point, multiplied by the number of\n"
            "scan points above - the default of 1000 (used elsewhere in the\n"
            "optimizer) is far too slow for a full 2D scan. 150-300 is\n"
            "usually plenty accurate for uniformity/crosstalk values."
        )
        scan_layout.addRow("Intensity grid resolution (n_grid):", self.n_grid)

        scan_group.setLayout(scan_layout)
        main_layout.addWidget(scan_group)

        # -- save location (asked UP FRONT, before the scan starts - see
        # Chat "Amplituden Abhängigkeit": Scans können Tage dauern, daher
        # wird stündlich unter GENAU diesem Pfad zwischengespeichert; ein
        # abgebrochener Scan wird beim erneuten Start automatisch an dieser
        # Stelle fortgesetzt (siehe main()/scan_checkpoint.py) --
        self._save_path_auto = True
        save_group = QGroupBox("Save Location (Zwischenspeicherung + Endergebnis)")
        save_layout = QHBoxLayout()
        self.save_path_edit = QLineEdit()
        self.save_path_edit.textEdited.connect(self._on_save_path_edited)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_save_path)
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(browse_btn)
        save_group.setLayout(save_layout)
        main_layout.addWidget(save_group)
        save_info = QLabel(
            "Der Scan wird unter diesem Pfad stündlich zwischengespeichert. Sollte der "
            "Prozess abbrechen, wird ein erneuter Start mit denselben Parametern und "
            "demselben Pfad automatisch an der Stelle fortgesetzt, an der er stehen "
            "geblieben ist."
        )
        save_info.setWordWrap(True)
        save_info.setStyleSheet("font-style: italic; color: gray;")
        main_layout.addWidget(save_info)

        self.nx_spin.valueChanged.connect(self._update_default_save_path)
        self.ny_spin.valueChanged.connect(self._update_default_save_path)
        self.n_points.valueChanged.connect(self._update_default_save_path)
        self._update_default_save_path()

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

    def _on_waist_mode_changed(self):
        """Wechselt die Beschriftung der win_input-min/max-Felder zwischen
        'vor der Linse' (mm) und 'nach der Linse' (µm) und rechnet die
        aktuell eingetragenen Werte passend um, damit sie beim Wechsel
        physikalisch sinnvoll bleiben (win_eff = K / win_input ist fallend,
        daher tauschen min/max dabei die Rollen)."""
        new_mode = self.waist_mode_combo.currentData()
        if new_mode == self._current_waist_mode:
            return

        old_min, old_max = self.win_input_min.value(), self.win_input_max.value()
        label_min = self.scan_layout.labelForField(self.win_input_min)
        label_max = self.scan_layout.labelForField(self.win_input_max)

        if new_mode == "win_eff":
            # bisher mm (win_input) eingetragen -> jetzt µm (win_eff) anzeigen
            new_min = win_input_to_win(old_max * 1e-3, _F1, _F2, _LAMBDA_OPT, _FLO) * 1e6
            new_max = win_input_to_win(old_min * 1e-3, _F1, _F2, _LAMBDA_OPT, _FLO) * 1e6
            label_min.setText("Fokus-Waist min (µm):")
            label_max.setText("Fokus-Waist max (µm):")
        else:
            # bisher µm (win_eff) eingetragen -> jetzt mm (win_input) anzeigen
            new_min = win_input_to_win(old_max * 1e-6, _F1, _F2, _LAMBDA_OPT, _FLO) * 1e3
            new_max = win_input_to_win(old_min * 1e-6, _F1, _F2, _LAMBDA_OPT, _FLO) * 1e3
            label_min.setText("win_input min (mm):")
            label_max.setText("win_input max (mm):")

        self.win_input_min.setValue(new_min)
        self.win_input_max.setValue(new_max)
        self._current_waist_mode = new_mode

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

    def _default_save_filename(self):
        n = self.n_points.value()
        return f"scan_data_N{self.nx_spin.value()}x{self.ny_spin.value()}_{n}x{n}pts.pkl"

    def _update_default_save_path(self):
        if self._save_path_auto:
            self.save_path_edit.setText(str(DEFAULT_RESULTS_DIR / self._default_save_filename()))

    def _on_save_path_edited(self, _text):
        self._save_path_auto = False

    def _on_browse_save_path(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Speicherort wählen", self.save_path_edit.text(), "Pickle files (*.pkl)"
        )
        if path:
            self.save_path_edit.setText(path)
            self._save_path_auto = False

    def _on_accept(self):
        if self.win_input_min.value() >= self.win_input_max.value():
            QMessageBox.warning(self, "Invalid Range",
                                 "win_input min must be smaller than win_input max.")
            return
        if self.width_min.value() >= self.width_max.value():
            QMessageBox.warning(self, "Invalid Range",
                                 "width min must be smaller than width max.")
            return
        if not self.save_path_edit.text().strip():
            QMessageBox.warning(self, "Invalid Save Location",
                                 "Bitte einen Speicherort für die Zwischen-/Endergebnisse angeben.")
            return
        self.accept()

    def get_values(self):
        """Returns all entered parameters as a dict (SI units: meters, Hz).

        win_input_range wird IMMER in win_input (vor der Linse, Meter)
        zurückgegeben, egal welcher Modus in waist_mode_combo gewählt war -
        scan_win_width_uniformity() erwartet ausschließlich win_input.
        Bei Modus 'win_eff' werden die eingegebenen Fokus-Waist-Werte (µm)
        dafür über win_input_to_win() zurückgerechnet (die Formel ist ihre
        eigene Umkehrfunktion); da win_eff fallend in win_input ist,
        vertauschen sich dabei min und max.
        """
        amp_x = np.array([b.value() for b in self.amp_x_boxes])
        amp_y = np.array([b.value() for b in self.amp_y_boxes])

        raw_min = self.win_input_min.value()
        raw_max = self.win_input_max.value()
        if self._current_waist_mode == "win_eff":
            win_input_range = (
                win_input_to_win(raw_max * 1e-6, _F1, _F2, _LAMBDA_OPT, _FLO),
                win_input_to_win(raw_min * 1e-6, _F1, _F2, _LAMBDA_OPT, _FLO),
            )
        else:
            win_input_range = (raw_min * 1e-3, raw_max * 1e-3)

        return dict(
            N_x=self.nx_spin.value(),
            N_y=self.ny_spin.value(),
            amp_x=amp_x,
            amp_y=amp_y,
            win_input_range=win_input_range,
            width_range=(self.width_min.value() * 1e6, self.width_max.value() * 1e6),
            n_points=self.n_points.value(),
            n_grid=self.n_grid.value(),
            # welche Achse beim direkten Plotten am Ende von main() genutzt
            # werden soll (siehe ScanPlotter.plot_scan2d_combined(x_axis=...)) -
            # standardmäßig dieselbe, auf die sich der Scan-Bereich hier bezog
            waist_mode=self._current_waist_mode,
            save_path=self.save_path_edit.text().strip(),
        )


def main():
    #Niklas Aufruf über Conda/Konsole Whatever
    # if "--use-cuda" in sys.argv:
    #     print("Enabling cuda")
    #     import use_torch
    #     use_torch.patch()
    # if "--profile" in sys.argv:
    #     print("Enabling profile logging")
    #     perf_log.enable()

    print("Enabling cuda")
    import use_torch
    use_torch.patch()

    print("Enabling profile logging")
    perf_log.enable()

    app = QApplication(sys.argv)

    dialog = StartParametersDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)

    params = dialog.get_values()
    amps = np.concatenate([params["amp_x"], params["amp_y"]])
    save_path = params["save_path"]

    opt = MultitoneFlatTopOptimizer(
        out_dir=".",
        f1=75e-3,
        f2=750e-3,
        N_x=params["N_x"],
        N_y=params["N_y"],
        n_grid=params["n_grid"],
    )

    # Vor dem (potenziell tagelangen) Scan prüfen, ob unter dem gewählten
    # Speicherort bereits ein zu diesen Parametern passender Zwischenstand
    # liegt (z.B. von einem abgebrochenen vorherigen Lauf) - und den
    # Nutzer informieren, ob automatisch fortgesetzt oder neu gestartet wird.
    if save_path and Path(save_path).exists():
        resumable = scan_checkpoint.load_resumable(
            save_path, params["win_input_range"], params["width_range"],
            params["n_points"], params["n_points"], params["N_x"], params["N_y"],
            extra_match=dict(amps=amps, alpha=0.9), verbose=False,
        )
        if resumable is not None:
            n_done = scan_checkpoint.count_done(resumable["uniformity_grid"])
            total_pts = params["n_points"] * params["n_points"]
            QMessageBox.information(
                None, "Zwischenstand gefunden",
                f"Unter '{Path(save_path).name}' wurde ein zu diesen Scan-Parametern "
                f"passender Zwischenstand mit {n_done}/{total_pts} bereits berechneten "
                f"Punkten gefunden.\n\nDer Scan wird automatisch an dieser Stelle "
                f"fortgesetzt.",
            )
        else:
            QMessageBox.information(
                None, "Vorhandene Datei passt nicht",
                f"Die Datei '{Path(save_path).name}' existiert bereits, passt aber nicht "
                f"zu den aktuell gewählten Scan-Parametern (anderes Gitter/Tonzahl/"
                f"Amplituden) oder konnte nicht gelesen werden.\n\nDer Scan startet "
                f"komplett neu und überschreibt die Datei beim nächsten "
                f"Zwischenspeichern.",
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

    perf_measure = perf_log.Measurement()
    perf_measure.start()

    # Pure computation - no plotting happens here. This means re-running
    # just the plotting step later (e.g. from a saved .pkl) never needs to
    # repeat this (potentially slow) scan.
    opt.scan_win_width_uniformity(
        win_input_range=params["win_input_range"],
        width_range=params["width_range"],
        n_win_input=params["n_points"],
        n_width=params["n_points"],
        amps=amps,
        progress_callback=on_progress,
        checkpoint_path=save_path or None,
    )

    duration = perf_measure.stop()
    perf_log.log(f"Optimization time: {duration/1e9}s")

    progress.setValue(total_points)

    # Der Speicherort wurde bereits VOR dem Scan festgelegt (s.o.) und diente
    # dort bereits als checkpoint_path (stündliche Zwischenspeicherung) - hier
    # nur noch der finale, "saubere" Endstand (ohne Checkpoint-Markerfelder),
    # der denselben Pfad überschreibt. Kein erneuter Dateidialog mehr nötig.
    if save_path:
        opt.save_scan_results(save_path)
    else:
        opt.save_scan_results()  # kein Pfad angegeben -> Results-Ordner-Default

    def qt_confirm_overwrite(existing_path):
        answer = QMessageBox.question(
            None, "File already exists",
            f"'{existing_path.name}' already exists. Overwrite it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    plotter = ScanPlotter(opt.get_scan_results(), out_dir=".", confirm_overwrite=qt_confirm_overwrite)
    plotter.plot_scan2d_combined(show=True, save=True, x_axis=params["waist_mode"])


if __name__ == "__main__":
    main()