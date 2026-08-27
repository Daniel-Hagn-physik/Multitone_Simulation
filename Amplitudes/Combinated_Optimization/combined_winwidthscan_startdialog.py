"""
Combinated_Optimization/combined_winwidthscan_startdialog.py
==============================================================

GUI-Startdialog fuer den KOMBINIERTEN Fest-Amplituden-Scan (hart +
atom-gewichtet, siehe combined_scan_methods.py) - das Gegenstueck zu
Weighted_Optimization/weighted_winwidthscan_startdialog.py, aber mit EINEM
zusaetzlichen Scan-Durchlauf (die unveraenderte harte
scan_win_width_uniformity()) und einer neuen Gruppe "Combination" fuer die
Kombinationsparameter (combo_lambda, combo_percentile). Alle uebrigen
Felder (Tonanzahl, feste Amplituden pro Ton, Scan-Bereiche, Atom-Gewichtung,
GPU/Perf) sind inhaltlich identisch zum gewichteten Original-Dialog - siehe
dessen Docstring fuer Details zu den einzelnen Feldern.

Warum ueberhaupt ein kombinierter Scan (siehe Chat "Amplituden
Abhaengigkeit"): weder die rein harte (globale) noch die rein
atom-gewichtete (lokale) Metrik allein liefert einen Bereich, der
garantiert sowohl global als auch lokal gut ist - ein gutes globales
Ergebnis muss lokal nicht gut sein und umgekehrt. Dieser Dialog fuehrt
BEIDE Scans auf derselben Optimizer-Instanz mit identischen Parametern aus
(daher liegen die Ergebnis-Grids exakt auf demselben Gitter) und leitet
daraus einen kombinierten Score + eine tatsaechlich nutzbare
(win_input,width)-Region ab (groesstes Rechteck innerhalb der besten
combo_percentile% aller Gitterpunkte) - siehe combined_scan_methods.py fuer
das Kombinationsprinzip.

Nutzung:
    python combined_winwidthscan_startdialog.py
"""

import sys
from pathlib import Path as FilePath

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QFileDialog, QComboBox, QCheckBox, QLineEdit,
)
from PyQt5.QtCore import Qt

_WEIGHTED_DIR = FilePath(__file__).resolve().parent.parent / "Weighted_Optimization"
if str(_WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(_WEIGHTED_DIR))

from weighted_multitone_flattop_optimizer import MultitoneFlatTopOptimizer  # noqa: E402
from weighted_multitone_amplitude_dependence_plots import win_input_to_win  # noqa: E402
import perf_log  # noqa: E402  (aus Weighted_Optimization, per sys.path - reine Utility, keine Kopplung)
import scan_checkpoint  # noqa: E402  (identische Kopie, ebenfalls aus Weighted_Optimization erreichbar)

from combined_scan_methods import (  # noqa: E402,F401
    DEFAULT_RESULTS_DIR, DEFAULT_IMAGES_DIR, _derive_checkpoint_paths,
)
import combined_scan_methods  # noqa: E402,F401  # Import-Nebeneffekt: patcht scan_win_width_combined_uniformity() etc.
from combined_scan_plots import CombinedFixedScanPlotter  # noqa: E402

# Feste Optik-Parameter fuer die win_input<->win_eff-Umrechnung in diesem
# Dialog - identisch zu den anderen Startdialogs im Projekt.
_F1 = 75e-3
_F2 = 750e-3
_FLO = MultitoneFlatTopOptimizer.DEFAULTS['fLO']
_LAMBDA_OPT = MultitoneFlatTopOptimizer.DEFAULTS['lambda_opt']
_PITCH = MultitoneFlatTopOptimizer.DEFAULTS['pitch']

_WAIST_MODE_TO_WIN_AXIS = {"win_input": "before_lens", "win_eff": "after_lens"}


class StartParametersDialog(QDialog):
    """Fragt Tonanzahl, feste Amplituden pro Ton, Scan-Bereiche
    (win_input/width), Atom-Gewichtungsparameter UND die neuen
    Kombinationsparameter (combo_lambda, combo_percentile) ab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multitone FlatTop - Combined (Hard + Weighted) Fixed-Amplitude Scan")

        self.amp_x_boxes = []
        self.amp_y_boxes = []

        main_layout = QVBoxLayout(self)

        info_label = QLabel(
            "Fixed optical parameters: f1 = 75 mm, f2 = 750 mm\n"
            "Amplitudes are FIXED per tone. Runs BOTH the hard (global mask) scan\n"
            "AND the atom-weighted (local) scan on the same grid, then combines them\n"
            "(see 'Combination' group below) into one usable region."
        )
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
        self.scan_layout = scan_layout

        self.waist_mode_combo = QComboBox()
        self.waist_mode_combo.addItem("vor der Linse (Input-Waist, mm)", "win_input")
        self.waist_mode_combo.addItem("nach der Linse (Fokus-Waist, µm)", "win_eff")
        self.waist_mode_combo.setToolTip(
            "Legt fest, worauf sich die beiden Felder 'min'/'max' darunter\n"
            "beziehen: der Waist VOR der ersten Linse (win_input, mm) oder\n"
            "der effektive Waist NACH der Linse/am Fokus (win_eff, µm)."
        )
        self._current_waist_mode = "win_input"
        self.waist_mode_combo.currentIndexChanged.connect(self._on_waist_mode_changed)
        scan_layout.addRow("Waist-Eingabe bezieht sich auf:", self.waist_mode_combo)

        self.win_input_min = self._make_spin(0.8, 0.001, 50.0, 0.05)
        self.win_input_max = self._make_spin(1.7, 0.001, 50.0, 0.05)
        scan_layout.addRow("win_input min (mm):", self.win_input_min)
        scan_layout.addRow("win_input max (mm):", self.win_input_max)

        self.width_min = self._make_spin(0.2, 0.001, 5.0, 0.01)
        self.width_max = self._make_spin(0.4, 0.001, 5.0, 0.01)
        scan_layout.addRow("width min (MHz):", self.width_min)
        scan_layout.addRow("width max (MHz):", self.width_max)

        self.n_points = QSpinBox()
        self.n_points.setRange(5, 300)
        self.n_points.setValue(40)
        self.n_points.setToolTip(
            "Gitterpunkte pro Achse. Der kombinierte Scan rechnet BEIDE\n"
            "Einzel-Scans auf diesem Gitter - Laufzeit ist entsprechend die\n"
            "Summe aus hartem + gewichtetem Einzel-Scan (siehe Fortschrittsbalken)."
        )
        scan_layout.addRow("Grid points per axis:", self.n_points)

        self.n_grid = QSpinBox()
        self.n_grid.setRange(50, 2500)
        self.n_grid.setValue(1000)
        self.n_grid.setToolTip(
            "Resolution of the GLOBAL intensity grid, used by the HARD scan\n"
            "(_build_dynamic_grid() pro Punkt). Siehe 'Atom Weighting' ->\n"
            "weighted_n_grid unten fuer die Aufloesung, die fuer die\n"
            "gewichtete Metrik massgeblich ist."
        )
        scan_layout.addRow("Intensity grid resolution (n_grid):", self.n_grid)

        self.alpha = self._make_spin(0.9, 0.0, 1.0, 0.05)
        self.alpha.setToolTip(
            "Gewicht in alpha*Uniformity + (1-alpha)*Crosstalk - wird HIER an\n"
            "DREI Stellen verwendet: fuer den 'besten Punkt' des harten\n"
            "Einzel-Scans, des gewichteten Einzel-Scans, UND fuer den\n"
            "kombinierten Gesamt-Score (alpha*Uniformity_kombi +\n"
            "(1-alpha)*Crosstalk_kombi). Gleiche Konvention wie ueberall\n"
            "sonst im Projekt (Default dort meist 0.9)."
        )
        scan_layout.addRow("alpha (Uniformity/Crosstalk-Gewicht):", self.alpha)

        scan_group.setLayout(scan_layout)
        main_layout.addWidget(scan_group)

        # -- atom weighting --
        atom_group = QGroupBox("Atom Weighting (nur fuer den gewichteten Teil)")
        atom_layout = QFormLayout()

        atom_info = QLabel(
            "sigma_atom (Breite der Gauss-Gewichtung) wird aus atom_temperature\n"
            "und trap_freq_r ueber sigma_thermal() berechnet (atom_mass fest auf\n"
            "Rb-85, siehe DEFAULTS im Optimizer). Beeinflusst NUR die\n"
            "atom-gewichtete Haelfte des kombinierten Scans."
        )
        atom_info.setStyleSheet("font-style: italic;")
        atom_layout.addRow(atom_info)

        self.atom_temperature_uK = self._make_spin(17.0, 0.001, 1000.0, 0.5)
        atom_layout.addRow("atom_temperature (µK):", self.atom_temperature_uK)

        self.trap_freq_r_kHz = self._make_spin(60.4, 0.001, 10000.0, 1.0)
        atom_layout.addRow("trap_freq_r (kHz):", self.trap_freq_r_kHz)

        self.weighted_n_grid = QSpinBox()
        self.weighted_n_grid.setRange(21, 2001)
        self.weighted_n_grid.setSingleStep(20)
        self.weighted_n_grid.setValue(241)
        atom_layout.addRow("weighted_n_grid:", self.weighted_n_grid)

        offset_info = QLabel(
            f"Atom/Nachbar-Versatz als Bruchteil von pitch (aktuell {_PITCH * 1e6:.3f} µm);\n"
            f"0 = Atom exakt auf der Site (Default)."
        )
        offset_info.setStyleSheet("font-style: italic;")
        atom_layout.addRow(offset_info)

        self.atom_offset_frac_x = self._make_spin(0.0, -0.5, 0.5, 0.05)
        atom_layout.addRow("Atom offset x (x pitch, -0.5..0.5):", self.atom_offset_frac_x)

        self.atom_offset_frac_y = self._make_spin(0.0, -0.5, 0.5, 0.05)
        atom_layout.addRow("Atom offset y (x pitch, -0.5..0.5):", self.atom_offset_frac_y)

        atom_group.setLayout(atom_layout)
        main_layout.addWidget(atom_group)

        # -- Kombination (NEU) --
        combo_group = QGroupBox("Combination (hart + gewichtet -> Region)")
        combo_layout = QFormLayout()

        combo_info = QLabel(
            "Uniformity_kombi = 0.5*(hart_norm + weighted_norm)\n"
            "                   + combo_lambda * |hart_norm - weighted_norm|\n"
            "(analog fuer Crosstalk_kombi; hart/weighted jeweils Min-Max-normiert\n"
            "ueber das Scan-Gitter). Der zweite Term bestraft Punkte, an denen\n"
            "hart und gewichtet stark auseinanderlaufen, statt es im Mittelwert\n"
            "zu verstecken. Die 'Region' ist das groesste Rechteck innerhalb der\n"
            "besten combo_percentile% aller Punkte (nach combined_score =\n"
            "alpha*Uniformity_kombi + (1-alpha)*Crosstalk_kombi)."
        )
        combo_info.setStyleSheet("font-style: italic;")
        combo_layout.addRow(combo_info)

        self.combo_lambda = self._make_spin(0.75, 0.0, 5.0, 0.05)
        self.combo_lambda.setToolTip(
            "Gewicht des Uneinigkeits-Strafterms. 0 = reiner Mittelwert aus\n"
            "hart und weighted. Hoeher = Punkte mit grosser Diskrepanz zwischen\n"
            "hart und weighted werden staerker bestraft."
        )
        combo_layout.addRow("combo_lambda (Uneinigkeits-Gewicht):", self.combo_lambda)

        self.combo_percentile = self._make_spin(25.0, 1.0, 100.0, 5.0)
        self.combo_percentile.setToolTip(
            "Anteil (in %) der Gitterpunkte mit dem besten (kleinsten)\n"
            "combined_score, aus dem die Region (groesstes eingeschriebenes\n"
            "Rechteck) extrahiert wird. Kleiner = strengere/kleinere Region."
        )
        combo_layout.addRow("combo_percentile (% beste Punkte):", self.combo_percentile)

        combo_group.setLayout(combo_layout)
        main_layout.addWidget(combo_group)

        # -- GPU / logging --
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QFormLayout()
        gpu_info = QLabel(
            "GPU-Beschleunigung (CUDA via PyTorch) wird automatisch versucht;\n"
            "ohne installiertes torch/CUDA faellt es sauber auf CPU zurueck."
        )
        gpu_info.setStyleSheet("font-style: italic;")
        gpu_layout.addRow(gpu_info)

        self.force_cpu = QCheckBox("Force CPU only (skip GPU auto-detect)")
        gpu_layout.addRow(self.force_cpu)

        self.enable_perf_log = QCheckBox("Enable perf logging")
        gpu_layout.addRow(self.enable_perf_log)

        gpu_group.setLayout(gpu_layout)
        main_layout.addWidget(gpu_group)

        # -- save location (asked UP FRONT, before the scan starts - see
        # Chat "Amplituden Abhängigkeit": der kombinierte Scan fuehrt ZWEI
        # Teilscans nacheinander aus und kann daher besonders lange laufen;
        # der hier gewaehlte EINE Pfad wird intern in zwei eigene
        # Zwischenspeicher-Pfade fuer den harten und den gewichteten
        # Teilscan aufgeteilt (siehe _derive_checkpoint_paths() in
        # combined_scan_methods.py), die unabhaengig voneinander stuendlich
        # sichern und beim erneuten Start automatisch je an ihrer Stelle
        # fortsetzen --
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
            "Der Scan wird unter diesem Pfad stündlich zwischengespeichert (intern "
            "aufgeteilt in einen harten und einen gewichteten Teilscan-Zwischenstand). "
            "Sollte der Prozess abbrechen, wird ein erneuter Start mit denselben "
            "Parametern und demselben Pfad automatisch an der Stelle fortgesetzt, an "
            "der er stehen geblieben ist."
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
        ok_btn = QPushButton("Start Combined Scan")
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
        new_mode = self.waist_mode_combo.currentData()
        if new_mode == self._current_waist_mode:
            return

        old_min, old_max = self.win_input_min.value(), self.win_input_max.value()
        label_min = self.scan_layout.labelForField(self.win_input_min)
        label_max = self.scan_layout.labelForField(self.win_input_max)

        if new_mode == "win_eff":
            new_min = win_input_to_win(old_max * 1e-3, _F1, _F2, _LAMBDA_OPT, _FLO) * 1e6
            new_max = win_input_to_win(old_min * 1e-3, _F1, _F2, _LAMBDA_OPT, _FLO) * 1e6
            label_min.setText("Fokus-Waist min (µm):")
            label_max.setText("Fokus-Waist max (µm):")
        else:
            new_min = win_input_to_win(old_max * 1e-6, _F1, _F2, _LAMBDA_OPT, _FLO) * 1e3
            new_max = win_input_to_win(old_min * 1e-6, _F1, _F2, _LAMBDA_OPT, _FLO) * 1e3
            label_min.setText("win_input min (mm):")
            label_max.setText("win_input max (mm):")

        self.win_input_min.setValue(new_min)
        self.win_input_max.setValue(new_max)
        self._current_waist_mode = new_mode

    def _rebuild_amp_fields(self):
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
        return f"scan_data_combined_N{self.nx_spin.value()}x{self.ny_spin.value()}_{n}x{n}pts.pkl"

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
            alpha=self.alpha.value(),
            combo_lambda=self.combo_lambda.value(),
            combo_percentile=self.combo_percentile.value(),
            atom_temperature=self.atom_temperature_uK.value() * 1e-6,
            trap_freq_r=self.trap_freq_r_kHz.value() * 1e3,
            weighted_n_grid=self.weighted_n_grid.value(),
            atom_offset_x=self.atom_offset_frac_x.value() * _PITCH,
            atom_offset_y=self.atom_offset_frac_y.value() * _PITCH,
            force_cpu=self.force_cpu.isChecked(),
            enable_perf_log=self.enable_perf_log.isChecked(),
            waist_mode=self._current_waist_mode,
            save_path=self.save_path_edit.text().strip(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = StartParametersDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)

    params = dialog.get_values()
    amps = np.concatenate([params["amp_x"], params["amp_y"]])

    use_gpu = False
    if not params["force_cpu"]:
        try:
            import weighted_use_torch as use_torch
            use_gpu = use_torch.cuda_available()
        except ImportError:
            use_gpu = False

    if use_gpu:
        print("GPU (CUDA via PyTorch) gefunden - nutze GPU-Beschleunigung (weighted_use_torch.patch()).")
        use_torch.patch()
    else:
        if params["force_cpu"]:
            print("CPU erzwungen (force_cpu) - nutze CPU (NumPy).")
        else:
            print("Keine GPU/CUDA gefunden - nutze CPU (NumPy).")

    if params["enable_perf_log"]:
        perf_log.enable()

    opt = MultitoneFlatTopOptimizer(
        out_dir=DEFAULT_IMAGES_DIR,
        f1=75e-3,
        f2=750e-3,
        N_x=params["N_x"],
        N_y=params["N_y"],
        n_grid=params["n_grid"],
        atom_temperature=params["atom_temperature"],
        trap_freq_r=params["trap_freq_r"],
        weighted_n_grid=params["weighted_n_grid"],
        atom_offset_x=params["atom_offset_x"],
        atom_offset_y=params["atom_offset_y"],
    )

    save_path = params["save_path"]

    # Vor dem (potenziell tagelangen) kombinierten Scan pruefen, ob unter
    # den beiden ABGELEITETEN Teilscan-Pfaden (siehe _derive_checkpoint_
    # paths()) bereits zu diesen Parametern passende Zwischenstaende
    # liegen - und den Nutzer informieren, welcher Teilscan (falls
    # ueberhaupt) fortgesetzt wird.
    if save_path:
        ckpt_hard, ckpt_weighted = _derive_checkpoint_paths(save_path)
        status_lines = []
        for label, ckpt_path, grid_key, extra in (
            ("harten", ckpt_hard, "uniformity_grid", dict(amps=amps, alpha=params["alpha"])),
            ("gewichteten", ckpt_weighted, "uniformity_weighted_grid", dict(amps=amps, alpha=params["alpha"])),
        ):
            if not FilePath(ckpt_path).exists():
                continue
            resumable = scan_checkpoint.load_resumable(
                ckpt_path, params["win_input_range"], params["width_range"],
                params["n_points"], params["n_points"], params["N_x"], params["N_y"],
                extra_match=extra, verbose=False,
            )
            total_pts = params["n_points"] * params["n_points"]
            if resumable is not None:
                n_done = scan_checkpoint.count_done(resumable[grid_key])
                status_lines.append(f"- {label.capitalize()} Teilscan: {n_done}/{total_pts} Punkte "
                                     f"bereits vorhanden, wird fortgesetzt.")
            else:
                status_lines.append(f"- {label.capitalize()} Teilscan: vorhandene Datei passt nicht zu "
                                     f"den aktuellen Parametern, startet neu.")
        if status_lines:
            QMessageBox.information(
                None, "Zwischenstand gefunden",
                "Zu diesem kombinierten Scan wurden bereits Zwischenstaende gefunden:\n\n"
                + "\n".join(status_lines),
            )

    total_points = 2 * params["n_points"] * params["n_points"]
    progress = QProgressDialog("Computing combined (hard + weighted) uniformity & crosstalk scan...",
                                "Cancel", 0, total_points)
    progress.setWindowTitle("Combined Fixed-Amplitude 2D Scan Running")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    def on_progress(done, total):
        progress.setValue(done)
        QApplication.processEvents()
        if progress.wasCanceled():
            return False
        return True

    perf_measure = perf_log.Measurement()
    perf_measure.start()

    opt.scan_win_width_combined_uniformity(
        win_input_range=params["win_input_range"],
        width_range=params["width_range"],
        n_win_input=params["n_points"],
        n_width=params["n_points"],
        amps=amps,
        alpha=params["alpha"],
        combo_lambda=params["combo_lambda"],
        combo_percentile=params["combo_percentile"],
        progress_callback=on_progress,
        checkpoint_path=save_path or None,
    )

    duration = perf_measure.stop()
    perf_log.log(f"Optimization time: {duration/1e9}s")

    progress.setValue(total_points)

    # Der Speicherort wurde bereits VOR dem Scan festgelegt (s.o.) und diente
    # dort bereits als checkpoint_path (stündliche Zwischenspeicherung der
    # beiden abgeleiteten Teilscan-Zwischenstaende) - hier nur noch der
    # finale, kombinierte Endstand, der denselben Pfad überschreibt. Kein
    # erneuter Dateidialog mehr nötig.
    if save_path:
        opt.save_scan_combined_results(save_path)
    else:
        opt.save_scan_combined_results()

    def qt_confirm_overwrite(existing_path):
        answer = QMessageBox.question(
            None, "File already exists",
            f"'{existing_path.name}' already exists. Overwrite it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    plotter = CombinedFixedScanPlotter(opt.get_scan_combined_results(), confirm_overwrite=qt_confirm_overwrite)
    win_axis = _WAIST_MODE_TO_WIN_AXIS[params["waist_mode"]]
    plotter.plot_metric_comparison(show=True, save=True, win_axis=win_axis)
    plotter.plot_combined_region(show=True, save=True, win_axis=win_axis)


if __name__ == "__main__":
    main()
