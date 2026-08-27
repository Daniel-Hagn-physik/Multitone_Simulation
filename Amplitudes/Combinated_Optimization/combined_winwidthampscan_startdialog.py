"""
Combinated_Optimization/combined_winwidthampscan_startdialog.py
===================================================================

GUI-Startdialog fuer den KOMBINIERTEN AMPLITUDEN-OPTIMIERTEN Scan (hart +
atom-gewichtet, siehe combined_amp_scan_methods.py) - das Gegenstueck zu
Weighted_Optimization/weighted_winwidthampscan_startdialog.py, aber mit
EINEM zusaetzlichen Scan-Durchlauf (die unveraenderte harte
scan_win_width_amplitude_dependence()) und einer neuen Gruppe
"Combination" fuer combo_lambda/combo_percentile - analog zu
combined_winwidthscan_startdialog.py (Fest-Amplituden-Variante), siehe
dessen Docstring fuer das Kombinationsprinzip.

Unterschied zum Fest-Amplituden-Dialog: hier wird an JEDEM (win_input,
width)-Gitterpunkt eine eigene (r_x, r_y)-Optimierung durchgefuehrt
(deutlich teurer, siehe Parallelisierung/n_jobs unten) statt nur mit
festen Amplituden auszuwerten. N_x=3, N_y=4 sind wie im Original-Dialog
fest verdrahtet (Amplituden-Parametrisierung amp_x=[r_x,1,r_x],
amp_y=[r_y,1,1,r_y]).

Nutzung:
    python combined_winwidthampscan_startdialog.py
"""

import os
import sys
from pathlib import Path as FilePath

from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QFileDialog, QCheckBox, QLineEdit, QScrollArea, QWidget,
)
from PyQt5.QtCore import Qt

_WEIGHTED_DIR = FilePath(__file__).resolve().parent.parent / "Weighted_Optimization"
if str(_WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(_WEIGHTED_DIR))

from weighted_multitone_flattop_optimizer import MultitoneFlatTopOptimizer  # noqa: E402
from weighted_multitone_amplitude_dependence_plots import AmplitudeScanPlotter  # noqa: E402
import perf_log  # noqa: E402  (aus Weighted_Optimization, per sys.path)
import scan_checkpoint  # noqa: E402  (identische Kopie, ebenfalls aus Weighted_Optimization erreichbar)

from combined_scan_methods import (  # noqa: E402,F401
    DEFAULT_RESULTS_DIR, DEFAULT_IMAGES_DIR, _derive_checkpoint_paths,
)
import combined_amp_scan_methods  # noqa: E402,F401  # Import-Nebeneffekt: patcht
# scan_win_width_amplitude_dependence_combined() etc.
from combined_scan_plots import CombinedFixedScanPlotter  # noqa: E402


N_X_FIXED = 3
N_Y_FIXED = 4


class StartParametersDialog(QDialog):
    """Fragt Scan-Bereiche (win_input/width), r-Bounds, Gitteraufloesung,
    Atom-Gewichtungsparameter, Parallelisierung UND die neuen
    Kombinationsparameter (combo_lambda, combo_percentile) ab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multitone FlatTop - Combined (Hard + Weighted) Amplitude-Dependence Scan")

        # Der Dialog ist inzwischen (mit "Combination"/"Save Location" usw.)
        # gewachsen und passt auf manchen Bildschirmen nicht mehr komplett in
        # der Hoehe - Inhalt daher in einer QScrollArea, waehrend die
        # Start/Cancel-Buttons AUSSERHALB davon (also immer sichtbar am
        # unteren Rand) bleiben. Siehe Chat "Amplituden Abhängigkeit":
        # "das GUI ist so lang, dass ich den Button unten nicht sehe".
        outer_layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        main_layout = QVBoxLayout(content)
        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)

        info_label = QLabel(
            f"Fixed: f1 = 75 mm, f2 = 750 mm, N_x = {N_X_FIXED}, N_y = {N_Y_FIXED}\n"
            f"Amplitudes: amp_x = [r_x, 1, r_x],  amp_y = [r_y, 1, 1, r_y]  (outer/inner ratio)\n"
            f"With 'joint optimization' (default, see 'Combination' group below): ONE (r_x, r_y)\n"
            f"optimization per grid point, minimizing the hard+weighted combination directly - the\n"
            f"found amplitude is valid for both criteria at once. Without it: hard and weighted are\n"
            f"optimized SEPARATELY (their own r_x/r_y each, can differ) and only their resulting\n"
            f"metrics are combined afterwards."
        )
        info_label.setStyleSheet("font-style: italic;")
        main_layout.addWidget(info_label)

        # -- scan ranges --
        scan_group = QGroupBox("Scan Ranges")
        scan_layout = QFormLayout()

        self.win_input_min = self._make_spin(0.8, 0.001, 50.0, 0.05)
        self.win_input_max = self._make_spin(1.7, 0.001, 50.0, 0.05)
        scan_layout.addRow("win_input min (mm):", self.win_input_min)
        scan_layout.addRow("win_input max (mm):", self.win_input_max)

        self.width_min = self._make_spin(0.2, 0.001, 5.0, 0.01)
        self.width_max = self._make_spin(0.4, 0.001, 5.0, 0.01)
        scan_layout.addRow("width min (MHz):", self.width_min)
        scan_layout.addRow("width max (MHz):", self.width_max)

        self.n_points = QSpinBox()
        self.n_points.setRange(3, 200)
        self.n_points.setValue(15)
        self.n_points.setToolTip(
            "Grid points per axis. EVERY grid point runs its own (r_x, r_y)\n"
            "optimization TWICE (hard + weighted) - keep this low (12-20) for\n"
            "a first pass. Total cost roughly = 2x the single-scan cost at the\n"
            "same point count."
        )
        scan_layout.addRow("Grid points per axis:", self.n_points)

        scan_group.setLayout(scan_layout)
        main_layout.addWidget(scan_group)

        # -- amplitude-ratio bounds & optimization settings --
        amp_group = QGroupBox("Amplitude-Ratio Optimization")
        amp_layout = QFormLayout()

        self.r_min = self._make_spin(0.1, 0.0, 10.0, 0.05)
        self.r_max = self._make_spin(10.0, 0.0, 10.0, 0.05)
        amp_layout.addRow("r_x / r_y min:", self.r_min)
        amp_layout.addRow("r_x / r_y max:", self.r_max)

        self.alpha = self._make_spin(0.7, 0.0, 1.0, 0.05)
        self.alpha.setToolTip(
            "With joint optimization (default, see 'Combination' below): weight in\n"
            "the SINGLE per-point objective alpha*Uniformity_kombi +\n"
            "(1-alpha)*Crosstalk_kombi that is minimized directly over (r_x, r_y).\n"
            "Without joint optimization: weight of BOTH separate per-point objectives\n"
            "(hard: alpha*uniformity + (1-alpha)*eta; weighted: alpha*uniformity_w +\n"
            "(1-alpha)*eta_w), AND of the combined score afterwards. Same convention\n"
            "throughout either way."
        )
        amp_layout.addRow("alpha:", self.alpha)

        amp_group.setLayout(amp_layout)
        main_layout.addWidget(amp_group)

        # -- intensity grid resolution --
        grid_group = QGroupBox("Intensity Grid")
        grid_layout = QFormLayout()
        self.n_grid = QSpinBox()
        self.n_grid.setRange(50, 5000)
        self.n_grid.setValue(1000)
        self.n_grid.setToolTip(
            "Resolution of the GLOBAL intensity grid, used by the HARD scan.\n"
            "See 'Atom Weighting' -> weighted_n_grid below for the weighted part."
        )
        grid_layout.addRow("n_grid:", self.n_grid)
        grid_group.setLayout(grid_layout)
        main_layout.addWidget(grid_group)

        # -- atom weighting --
        atom_group = QGroupBox("Atom Weighting (nur fuer den gewichteten Teil)")
        atom_layout = QFormLayout()

        atom_info = QLabel(
            "sigma_atom wird aus atom_temperature und trap_freq_r berechnet\n"
            "(atom_mass fest auf Rb-85). Beeinflusst NUR die atom-gewichtete\n"
            "Haelfte des kombinierten Scans."
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

        atom_group.setLayout(atom_layout)
        main_layout.addWidget(atom_group)

        # -- Kombination (NEU) --
        combo_group = QGroupBox("Combination (hart + gewichtet -> Region)")
        combo_layout = QFormLayout()

        combo_info = QLabel(
            "Uniformity_kombi/Crosstalk_kombi = Mittelwert + combo_lambda * |Differenz|\n"
            "der harten und gewichteten Metrik. Region = groesstes Rechteck in den\n"
            "besten combo_percentile% aller Punkte nach combined_score."
        )
        combo_info.setStyleSheet("font-style: italic;")
        combo_layout.addRow(combo_info)

        self.joint_optimization = QCheckBox("Gemeinsame (jointe) Amplituden-Optimierung (empfohlen)")
        self.joint_optimization.setChecked(True)
        self.joint_optimization.setToolTip(
            "EMPFOHLEN (siehe Chat 'Amplituden Abhängigkeit'): An jedem Gitterpunkt wird\n"
            "GENAU EINE (r_x, r_y)-Optimierung durchgeführt, die DIREKT gegen die (rohe,\n"
            "unnormierte) Kombination alpha*Uniformity_kombi + (1-alpha)*Crosstalk_kombi\n"
            "minimiert - hart und gewichtet werden dabei bei JEDEM Optimierungsschritt am\n"
            "SELBEN (r_x,r_y) ausgewertet. Die gefundene Amplitude ist damit automatisch\n"
            "für BEIDE Kriterien gleichzeitig gültig, nicht nur für eines.\n\n"
            "Deaktiviert (altes Verhalten): hart und gewichtet werden GETRENNT optimiert\n"
            "(je ihr eigenes Ziel) - können unterschiedliche r_x/r_y liefern; erst danach\n"
            "werden die dabei erreichten Metriken (an verschiedenen Punkten!) kombiniert.\n"
            "Kostet außerdem ~2x so viele Optimierungsläufe wie die gemeinsame Variante."
        )
        combo_layout.addRow(self.joint_optimization)

        self.combo_lambda = self._make_spin(0.75, 0.0, 5.0, 0.05)
        combo_layout.addRow("combo_lambda (Uneinigkeits-Gewicht):", self.combo_lambda)

        self.combo_percentile = self._make_spin(25.0, 1.0, 100.0, 5.0)
        combo_layout.addRow("combo_percentile (% beste Punkte):", self.combo_percentile)

        self.r_grid_source = QCheckBox("Primaeres r_x/r_y aus dem HARTEN Scan (statt gewichtet)")
        self.r_grid_source.setToolTip(
            "Nur relevant, wenn 'Gemeinsame (jointe) Amplituden-Optimierung' oben\n"
            "DEAKTIVIERT ist (im jointen Modus ist r_x/r_y ohnehin fuer hart UND\n"
            "gewichtet identisch). Betrifft dann NUR, welcher r_x/r_y-Satz als\n"
            "'primaerer' r_x_grid/r_y_grid im Ergebnis-dict landet (z.B. fuer die\n"
            "6-Panel-Uebersicht/Dependence-Cuts). BEIDE Saetze (hart UND gewichtet)\n"
            "bleiben in jedem Fall vollstaendig im gespeicherten pkl erhalten.\n"
            "Default: gewichtet (unangehakt), da das die tatsaechliche thermische\n"
            "Ausdehnung des Atoms beruecksichtigt."
        )
        combo_layout.addRow(self.r_grid_source)
        self.joint_optimization.toggled.connect(lambda checked: self.r_grid_source.setDisabled(checked))
        self.r_grid_source.setDisabled(self.joint_optimization.isChecked())

        combo_group.setLayout(combo_layout)
        main_layout.addWidget(combo_group)

        # -- parallelization --
        cpu_count = os.cpu_count() or 1
        parallel_group = QGroupBox("Parallelization")
        parallel_layout = QFormLayout()
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(1, max(1, cpu_count))
        self.n_jobs.setValue(max(1, cpu_count - 1))
        self.n_jobs.setToolTip(
            f"Number of parallel processes for EACH of the two scans\n"
            f"(this machine reports {cpu_count} CPU(s)). n_jobs=1 keeps the\n"
            f"sequential, warm-started behavior; n_jobs>1 disables warm-start."
        )
        parallel_layout.addRow(f"n_jobs (1-{cpu_count}):", self.n_jobs)
        parallel_group.setLayout(parallel_layout)
        main_layout.addWidget(parallel_group)

        # -- GPU / logging --
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QFormLayout()
        gpu_info = QLabel(
            "GPU-Beschleunigung (CUDA via PyTorch) wird automatisch versucht;\n"
            "ohne installiertes torch/CUDA faellt es auf die parallelisierte\n"
            "CPU (n_jobs oben) zurueck."
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
        # Chat "Amplituden Abhängigkeit": dieser Scan fuehrt ZWEI besonders
        # teure (pro Punkt eine volle Nelder-Mead-Optimierung) Teilscans
        # nacheinander aus und kann daher Tage laufen; der hier gewaehlte
        # EINE Pfad wird intern in zwei eigene Zwischenspeicher-Pfade fuer
        # den harten und den gewichteten Teilscan aufgeteilt (siehe
        # _derive_checkpoint_paths() in combined_scan_methods.py), die
        # unabhaengig voneinander stuendlich sichern und beim erneuten
        # Start automatisch je an ihrer Stelle fortsetzen --
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

        self.n_points.valueChanged.connect(self._update_default_save_path)
        self._update_default_save_path()

        # -- buttons -- (bewusst AUSSERHALB der QScrollArea, an outer_layout,
        # damit Start/Cancel immer sichtbar bleiben, egal wie lang der
        # scrollbare Inhalt oben ist)
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Start Combined Scan")
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        outer_layout.addLayout(btn_layout)

        # Anfangsgroesse an die tatsaechliche Bildschirmhoehe anpassen (statt
        # blind auf die volle Inhaltshoehe zu wachsen) - bei kleineren
        # Bildschirmen/Aufloesungen greift dann automatisch der Scrollbalken
        # der QScrollArea oben, die Buttons bleiben trotzdem erreichbar.
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail_height = screen.availableGeometry().height()
            dialog_height = min(760, int(avail_height * 0.85))
        else:
            dialog_height = 760
        self.resize(640, dialog_height)

    @staticmethod
    def _make_spin(value, minimum, maximum, step):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(3)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _default_save_filename(self):
        n = self.n_points.value()
        return f"scan_amp_data_combined_N{N_X_FIXED}x{N_Y_FIXED}_{n}x{n}pts_Airy.pkl"

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
        if self.r_min.value() >= self.r_max.value():
            QMessageBox.warning(self, "Invalid Range",
                                 "r_x/r_y min must be smaller than r_x/r_y max.")
            return
        if not self.save_path_edit.text().strip():
            QMessageBox.warning(self, "Invalid Save Location",
                                 "Bitte einen Speicherort für die Zwischen-/Endergebnisse angeben.")
            return
        self.accept()

    def get_values(self):
        return dict(
            win_input_range=(self.win_input_min.value() * 1e-3, self.win_input_max.value() * 1e-3),
            width_range=(self.width_min.value() * 1e6, self.width_max.value() * 1e6),
            n_points=self.n_points.value(),
            r_bounds=(self.r_min.value(), self.r_max.value()),
            alpha=self.alpha.value(),
            combo_lambda=self.combo_lambda.value(),
            combo_percentile=self.combo_percentile.value(),
            joint_optimization=self.joint_optimization.isChecked(),
            r_grid_source="hart" if self.r_grid_source.isChecked() else "weighted",
            n_grid=self.n_grid.value(),
            atom_temperature=self.atom_temperature_uK.value() * 1e-6,
            trap_freq_r=self.trap_freq_r_kHz.value() * 1e3,
            weighted_n_grid=self.weighted_n_grid.value(),
            n_jobs=self.n_jobs.value(),
            force_cpu=self.force_cpu.isChecked(),
            enable_perf_log=self.enable_perf_log.isChecked(),
            save_path=self.save_path_edit.text().strip(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = StartParametersDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)

    params = dialog.get_values()

    n_jobs = params["n_jobs"]
    pool_initializer = None
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
        pool_initializer = use_torch.patch
        n_jobs = 1
    else:
        if params["force_cpu"]:
            print(f"CPU erzwungen (force_cpu) - nutze parallelisierte CPU mit n_jobs={n_jobs}.")
        else:
            print(f"Keine GPU/CUDA gefunden - nutze parallelisierte CPU mit n_jobs={n_jobs}.")

    if params["enable_perf_log"]:
        perf_log.enable()

    opt = MultitoneFlatTopOptimizer(
        out_dir=DEFAULT_IMAGES_DIR,
        f1=75e-3,
        f2=750e-3,
        N_x=N_X_FIXED,
        N_y=N_Y_FIXED,
        n_grid=params["n_grid"],
        atom_temperature=params["atom_temperature"],
        trap_freq_r=params["trap_freq_r"],
        weighted_n_grid=params["weighted_n_grid"],
    )

    save_path = params["save_path"]
    joint = params["joint_optimization"]

    # Vor dem (potenziell tagelangen) kombinierten Scan pruefen, ob bereits
    # ein zu diesen Parametern passender Zwischenstand vorliegt. Im jointen
    # Modus gibt es nur EINEN Scan/Checkpoint-Pfad (save_path direkt); im
    # alten getrennten Modus wird save_path intern in ZWEI abgeleitete
    # Teilscan-Pfade aufgeteilt (siehe _derive_checkpoint_paths()).
    if save_path:
        if joint:
            resumable = scan_checkpoint.load_resumable(
                save_path, params["win_input_range"], params["width_range"],
                params["n_points"], params["n_points"], N_X_FIXED, N_Y_FIXED,
                extra_match=dict(alpha=params["alpha"], r_bounds=params["r_bounds"],
                                  combo_lambda=params["combo_lambda"], joint_optimization=True),
                verbose=False,
            )
            if resumable is not None:
                n_done = scan_checkpoint.count_done(resumable["uniformity_grid"])
                total_pts = params["n_points"] * params["n_points"]
                QMessageBox.information(
                    None, "Zwischenstand gefunden",
                    f"Zu diesem gemeinsamen (jointen) kombinierten Scan wurde bereits ein "
                    f"Zwischenstand gefunden:\n\n{n_done}/{total_pts} Punkte bereits "
                    f"vorhanden, wird fortgesetzt.",
                )
            elif FilePath(save_path).exists():
                QMessageBox.information(
                    None, "Zwischenstand gefunden",
                    "Unter diesem Pfad liegt bereits eine Datei, die aber nicht zu den "
                    "aktuellen Scan-Parametern passt - der Scan startet komplett neu.",
                )
        else:
            ckpt_hard, ckpt_weighted = _derive_checkpoint_paths(save_path)
            status_lines = []
            for label, ckpt_path, grid_key in (
                ("harten", ckpt_hard, "uniformity_grid"),
                ("gewichteten", ckpt_weighted, "uniformity_weighted_grid"),
            ):
                if not FilePath(ckpt_path).exists():
                    continue
                resumable = scan_checkpoint.load_resumable(
                    ckpt_path, params["win_input_range"], params["width_range"],
                    params["n_points"], params["n_points"], N_X_FIXED, N_Y_FIXED,
                    extra_match=dict(alpha=params["alpha"], r_bounds=params["r_bounds"]), verbose=False,
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

    # Joint: EINE Optimierung pro Punkt -> n_points^2 Laeufe insgesamt.
    # Getrennt (alt): ZWEI Optimierungen pro Punkt -> 2*n_points^2 Laeufe.
    total_points = (params["n_points"] * params["n_points"] if joint
                     else 2 * params["n_points"] * params["n_points"])
    progress = QProgressDialog("Computing combined (hard + weighted) amplitude-dependence scan...",
                                "Cancel", 0, total_points)
    progress.setWindowTitle("Combined Amplitude-Dependence Scan Running")
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

    if joint:
        opt.scan_win_width_amplitude_dependence_combined_joint(
            win_input_range=params["win_input_range"],
            width_range=params["width_range"],
            n_win_input=params["n_points"],
            n_width=params["n_points"],
            r_bounds=params["r_bounds"],
            alpha=params["alpha"],
            combo_lambda=params["combo_lambda"],
            combo_percentile=params["combo_percentile"],
            progress_callback=on_progress,
            n_jobs=n_jobs,
            pool_initializer=pool_initializer,
            checkpoint_path=save_path or None,
        )
    else:
        opt.scan_win_width_amplitude_dependence_combined(
            win_input_range=params["win_input_range"],
            width_range=params["width_range"],
            n_win_input=params["n_points"],
            n_width=params["n_points"],
            r_bounds=params["r_bounds"],
            alpha=params["alpha"],
            combo_lambda=params["combo_lambda"],
            combo_percentile=params["combo_percentile"],
            r_grid_source=params["r_grid_source"],
            progress_callback=on_progress,
            n_jobs=n_jobs,
            pool_initializer=pool_initializer,
            checkpoint_path=save_path or None,
        )

    duration = perf_measure.stop()
    perf_log.log(f"Optimization time: {duration / 1e9}s")

    progress.setValue(total_points)

    # Der Speicherort wurde bereits VOR dem Scan festgelegt (s.o.) und diente
    # dort bereits als checkpoint_path (stündliche Zwischenspeicherung der
    # beiden abgeleiteten Teilscan-Zwischenstaende) - hier nur noch der
    # finale, kombinierte Endstand, der denselben Pfad überschreibt. Kein
    # erneuter Dateidialog mehr nötig.
    if save_path:
        opt.save_scan_amp_results_combined(save_path)
    else:
        opt.save_scan_amp_results_combined()

    def qt_confirm_overwrite(existing_path):
        answer = QMessageBox.question(
            None, "File already exists",
            f"'{existing_path.name}' already exists. Overwrite it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    results = opt.get_scan_amp_results_combined()

    # AmplitudeScanPlotter erkennt automatisch, dass sowohl harte als auch
    # gewichtete Metrik-Grids vorhanden sind (has_hard=has_weighted=True) und
    # zeichnet die 6-Panel-Uebersicht (Uniformity/Crosstalk je hart+gewichtet,
    # plus r_x/r_y) - unveraendertes Modul, keine eigene Plot-Logik noetig.
    amp_plotter = AmplitudeScanPlotter(results, confirm_overwrite=qt_confirm_overwrite)
    amp_plotter.plot_scan2d_combined(show=True, save=True)
    amp_plotter.plot_dependence_cuts(show=True, save=True)

    # Zusaetzlich die kombinierte Region (Score-Heatmap + Rechteck).
    region_plotter = CombinedFixedScanPlotter(results, confirm_overwrite=qt_confirm_overwrite)
    region_plotter.plot_metric_comparison(show=True, save=True)
    region_plotter.plot_combined_region(show=True, save=True)


if __name__ == "__main__":
    main()
