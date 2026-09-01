"""
Weighted_Optimization/weighted_winwidthampscan_startdialog.py
================================================================

Port von `Winwidthampscan startdialog.py` (Original-Ordner) fuer den
ATOM-GEWICHTETEN Amplituden-Abhaengigkeits-Scan. Fragt (wie das Original)
nach Scan-Bereichen fuer win_input (Eingangs-Waist vor der 1. Linse, mm) und
width (Frequenzabstand, MHz) und startet dann
`opt.scan_win_width_amplitude_dependence_weighted()`
(weighted_amp_scan_methods.py) statt der harten
`scan_win_width_amplitude_dependence()` - an JEDEM (win_input, width)-
Gitterpunkt werden r_x/r_y so optimiert, dass
alpha*uniformity_weighted + (1-alpha)*eta_weighted minimiert wird (die
ATOM-THERMISCH gewichtete Uniformity/Crosstalk-Metrik, siehe
weighted_multitone_flattop_optimizer.py), statt der harten
Rechteck-Masken-Metrik.

Einziger STRUKTURELLER Unterschied zum Original-Dialog: eine zusaetzliche
Gruppe "Atom Weighting" fragt nach den drei Parametern, die sigma_atom (die
Breite der Gauss-Gewichtung der tatsaechlichen thermischen Ortsverteilung
eines Atoms in der Falle) bestimmen - atom_temperature, trap_freq_r,
weighted_n_grid (Aufloesung des lokalen Sub-Grids um jede Site, siehe
_build_local_weighted_grid() in der Optimizer-Datei). Alles andere (Scan-
Bereiche, r-Bounds, alpha, n_grid, Parallelisierung, GPU-Auto-Detect mit
CPU-Fallback, perf-Logging, Speichern+Neu-Plotten) ist 1:1 identisch zum
Original uebernommen.

Amplituden-Parametrisierung (UNVERAENDERT, "gewichtet" betrifft nur die
Metrik, nicht die Amplituden-Darstellung selbst) fuer N_x=3, N_y=4:

    N_x=3:  amp_x = [r_x, 1, r_x]        (outer tones = r_x, center = 1)
    N_y=4:  amp_y = [r_y, 1, 1, r_y]     (outer tones = r_y, inner pair = 1)

GPU-Beschleunigung: wie im Original automatisch versucht (kein Zwang), aber
ueber `weighted_use_torch.py` statt `use_torch.py` - das Original-Modul
wuerde die Funktionszeiger im FALSCHEN Optimizer-Modul patchen und dadurch
wirkungslos bleiben (siehe Docstring von weighted_use_torch.py).

f1 = 75 mm und f2 = 750 mm sind fest (Default-Werte des Optimizers). N_x=3,
N_y=4 sind fest, passend zur obigen Amplituden-Parametrisierung.

Nach Bestaetigung berechnet
MultitoneFlatTopOptimizer.scan_win_width_amplitude_dependence_weighted() den
Scan (reine Berechnung, kein Plotten). Die Rohdaten werden anschliessend
gespeichert (Dateiname-Schema "scan_amp_data_weighted_...pkl", damit der
Scan fuer spaeteres Neu-Plotten nie wiederholt werden muss) und an
weighted_multitone_amplitude_dependence_plots.AmplitudeScanPlotter
uebergeben, der die kombinierten Heatmaps (Uniformity_w, Crosstalk_w, r_x,
r_y) sowie die Dependence-Cut-Liniendiagramme zeichnet.

Nutzung:
    python weighted_winwidthampscan_startdialog.py
"""

import os
import sys
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QFileDialog, QCheckBox, QLineEdit,
    QScrollArea, QWidget,
)
from PyQt5.QtCore import Qt

from weighted_multitone_flattop_optimizer import (
    MultitoneFlatTopOptimizer, DEFAULT_RESULTS_DIR, DEFAULT_IMAGES_DIR,
)
import weighted_amp_scan_methods  # nur Import noetig - patcht die neuen Scan-Methoden auf die Klasse
from weighted_multitone_amplitude_dependence_plots import AmplitudeScanPlotter
import perf_log
import scan_checkpoint
import airy_scale
import resume_picker


N_X_FIXED = 3
N_Y_FIXED = 4


class StartParametersDialog(QDialog):
    """Fragt Scan-Bereiche (win_input/width), r-Bounds, Gitteraufloesung UND
    die Atom-Gewichtungsparameter (sigma_atom-Bestimmung) ab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multitone FlatTop - Weighted Amplitude-Dependence Scan Parameters")

        # Der Parameterbereich liegt in einer QScrollArea, die Buttons
        # bewusst DARUNTER und ausserhalb - so bleiben "Start"/"Cancel"
        # immer sichtbar, egal wie viele Gruppen der Dialog enthaelt
        # (gleiche Loesung wie in den Combinated_Optimization-Dialogen).
        outer_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        main_layout = QVBoxLayout(scroll_content)
        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)

        # -- Fortsetzen: unfertigen Datensatz auswaehlen --
        # Steht bewusst GANZ OBEN: die Entscheidung "neu oder fortsetzen"
        # kommt vor allen Parametern, denn beim Fortsetzen kommen die
        # Parameter aus der Datei und die Felder darunter werden gesperrt.
        self.resume_group = resume_picker.ResumePickerGroup(
            DEFAULT_RESULTS_DIR, kind=resume_picker.KIND_WEIGHTED_AMP,
            on_change=self._on_resume_changed)
        main_layout.addWidget(self.resume_group)


        # -- fixed parameters (display only) --
        info_label = QLabel(
            f"Fixed: f1 = 75 mm, f2 = 750 mm, N_x = {N_X_FIXED}, N_y = {N_Y_FIXED}\n"
            f"Amplitudes: amp_x = [r_x, 1, r_x],  amp_y = [r_y, 1, 1, r_y]  "
            f"(outer/inner ratio, inner = 1)\n"
            f"Objective: alpha*uniformity_weighted + (1-alpha)*eta_weighted "
            f"(ATOM-WEIGHTED, see 'Atom Weighting' below)"
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
            "Grid points per axis. Unlike the plain uniformity/crosstalk scan,\n"
            "EVERY grid point here runs its own (r_x, r_y) optimization\n"
            "(~30-100 evaluations) - keep this low (12-20) for a first pass.\n"
            "Note: the weighted evaluation itself is cheaper per point than\n"
            "the hard-masked one (no global grid, only a small local sub-grid),\n"
            "so this scan tends to be faster than the un-weighted equivalent\n"
            "at the same point count."
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
            "Weight of the per-point objective:\n"
            "alpha*uniformity_weighted + (1-alpha)*eta_weighted.\n"
            "Default 0.7 - both are minimized together (alpha=1.0 would\n"
            "minimize uniformity_weighted only). eta_weighted at the found\n"
            "optimum is recorded either way."
        )
        amp_layout.addRow("alpha (0.7 = uniformity_w+eta_w):", self.alpha)

        amp_group.setLayout(amp_layout)
        main_layout.addWidget(amp_group)

        # -- intensity grid resolution (hard/global grid, used by win_input_to_win etc.) --
        grid_group = QGroupBox("Intensity Grid")
        grid_layout = QFormLayout()
        self.n_grid = QSpinBox()
        self.n_grid.setRange(50, 5000)
        self.n_grid.setValue(1000)
        self.n_grid.setToolTip(
            "Resolution of the GLOBAL intensity grid (used by geometry/lens\n"
            "helper methods, not by the weighted evaluation itself - see\n"
            "'Atom Weighting' -> weighted_n_grid below for the grid that\n"
            "actually matters for uniformity_weighted/eta_weighted)."
        )
        grid_layout.addRow("n_grid:", self.n_grid)
        grid_group.setLayout(grid_layout)
        main_layout.addWidget(grid_group)

        # -- atom weighting (sigma_atom determination + local sub-grid) --
        atom_group = QGroupBox("Atom Weighting")
        atom_layout = QFormLayout()

        atom_info = QLabel(
            "sigma_atom (width of the Gaussian atom-position weighting) is\n"
            "computed from atom_temperature and trap_freq_r via sigma_thermal()\n"
            "(atom_mass fixed to Rb-85, see DEFAULTS in the optimizer file)."
        )
        atom_info.setStyleSheet("font-style: italic;")
        atom_layout.addRow(atom_info)

        self.atom_temperature_uK = self._make_spin(17.0, 0.001, 1000.0, 0.5)
        self.atom_temperature_uK.setToolTip(
            "Atom temperature in the trap (microkelvin). Higher T -> wider\n"
            "thermal spread -> larger sigma_atom -> the weighted metric\n"
            "'sees' a larger region around each site."
        )
        atom_layout.addRow("atom_temperature (µK):", self.atom_temperature_uK)

        self.trap_freq_r_kHz = self._make_spin(60.4, 0.001, 10000.0, 1.0)
        self.trap_freq_r_kHz.setToolTip(
            "Radial trap frequency nu_r (kHz). Higher nu_r -> tighter\n"
            "confinement -> smaller sigma_atom."
        )
        atom_layout.addRow("trap_freq_r (kHz):", self.trap_freq_r_kHz)

        self.weighted_n_grid = QSpinBox()
        self.weighted_n_grid.setRange(21, 2001)
        self.weighted_n_grid.setSingleStep(20)
        self.weighted_n_grid.setValue(241)
        self.weighted_n_grid.setToolTip(
            "Points per axis of the small LOCAL sub-grid built around each\n"
            "evaluated site (extent +/- weighted_n_sigma * sigma_atom, see\n"
            "_build_local_weighted_grid() in the optimizer file) - this is\n"
            "the resolution that actually matters for uniformity_weighted/\n"
            "eta_weighted. Keep odd-ish/moderate (151-301) for a good\n"
            "cost/accuracy tradeoff."
        )
        atom_layout.addRow("weighted_n_grid:", self.weighted_n_grid)

        atom_group.setLayout(atom_layout)
        main_layout.addWidget(atom_group)

        # -- parallelization --
        cpu_count = os.cpu_count() or 1
        parallel_group = QGroupBox("Parallelization")
        parallel_layout = QFormLayout()
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(1, max(1, cpu_count))
        self.n_jobs.setValue(max(1, cpu_count - 1))
        self.n_jobs.setToolTip(
            "Number of parallel processes for the (win_input, width) grid.\n"
            "Every point is independent, so this scales close to linearly\n"
            f"with cores (this machine reports {cpu_count} CPU(s)).\n"
            "n_jobs=1 keeps the sequential, warm-started behavior.\n"
            "n_jobs>1 disables warm-start between neighbouring points\n"
            "(each point starts fresh from r0=(1,1))."
        )
        parallel_layout.addRow(f"n_jobs (1-{cpu_count}):", self.n_jobs)
        parallel_group.setLayout(parallel_layout)
        main_layout.addWidget(parallel_group)

        # -- GPU acceleration / logging (monkey-patch, see weighted_use_torch.py) --
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QFormLayout()

        gpu_info = QLabel(
            "By default, GPU acceleration (CUDA via PyTorch,\n"
            "weighted_use_torch.patch()) is tried automatically at scan\n"
            "start. If torch is not installed or no CUDA GPU is found, it\n"
            "falls back to the parallelized CPU path (n_jobs above)\n"
            "automatically - no action needed either way."
        )
        gpu_info.setStyleSheet("font-style: italic;")
        gpu_layout.addRow(gpu_info)

        self.force_cpu = QCheckBox("Force CPU only (skip GPU auto-detect)")
        self.force_cpu.setToolTip(
            "Skips the automatic GPU auto-detect/patch entirely and always\n"
            "uses the plain NumPy path (optionally parallelized via n_jobs\n"
            "above). Useful for a reproducible baseline run, or if you\n"
            "specifically want to avoid the not-yet-verified CUDA/PyTorch\n"
            "code path (see weighted_use_torch.py docstring)."
        )
        gpu_layout.addRow(self.force_cpu)

        self.enable_perf_log = QCheckBox("Enable perf logging")
        self.enable_perf_log.setToolTip(
            "Prints total optimization time via perf_log - useful to\n"
            "compare CUDA on/off or different n_jobs settings."
        )
        gpu_layout.addRow(self.enable_perf_log)

        gpu_group.setLayout(gpu_layout)
        main_layout.addWidget(gpu_group)

        # -- save location (asked UP FRONT, before the scan starts - see
        # Chat "Amplituden Abhängigkeit": dieser Scan kann besonders lange
        # laufen (pro Punkt eine volle Nelder-Mead-Optimierung), daher wird
        # stündlich unter GENAU diesem Pfad zwischengespeichert; ein
        # abgebrochener Scan wird beim erneuten Start automatisch an dieser
        # Stelle fortgesetzt (siehe main()/scan_checkpoint.py) --
        self._save_path_auto = True
        # -- Strahlprofil (Airy-Skalenfaktor) --
        # Legt fest, was die Zahl "waist" physikalisch bedeutet:
        #     first_zero_radius = airy_scale_factor * waist
        # Voreingestellt ist 1.4830 (1/e^2-Radius der Airy-Hauptkeule = waist,
        # also dieselbe Bedeutung wie bei einem Gauss-Strahl). Der historische
        # Wert 1.19 bleibt waehlbar. Datensaetze mit verschiedenen Faktoren
        # sind NICHT vergleichbar - der Faktor wandert deshalb in die .pkl und
        # (wenn er nicht 1.19 ist) in den vorgeschlagenen Dateinamen.
        self.airy_group = airy_scale.AiryScaleGroup()
        main_layout.addWidget(self.airy_group)
        self.airy_group.factor_spin.valueChanged.connect(
            lambda _v: self._update_default_save_path())

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
        outer_layout.addLayout(btn_layout)

        # Anfangsgroesse an die tatsaechlich verfuegbare Bildschirmhoehe
        # kappen - bei kleineren Bildschirmen greift der Scrollbalken.
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry().height() if screen is not None else 900
        self.resize(max(self.width(), 660), min(820, int(avail * 0.85)))

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
        tag = airy_scale.scale_tag(self.airy_group.value())
        return f"scan_amp_data_weighted_N{N_X_FIXED}x{N_Y_FIXED}_{n}x{n}pts_Airy{tag}.pkl"

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

    def _on_resume_changed(self, entry):
        """Auswahl im Fortsetzen-Feld hat sich geaendert.

        Wird schon beim Aufbau der Gruppe einmal aufgerufen, also bevor die
        uebrigen Felder ueberhaupt existieren - deshalb die hasattr-Pruefung.
        """
        if not hasattr(self, "save_path_edit"):
            return
        if entry is None:
            resume_picker.set_inputs_locked(self, False, self.resume_group)
            self._save_path_auto = True
            self._update_default_save_path()
            return
        # Werte aus der Datei anzeigen, dann sperren. Massgeblich ist
        # ohnehin resume_group.apply() in get_values().
        resume_picker.apply_display(self, entry["results"])
        resume_picker.set_inputs_locked(self, True, self.resume_group)
        self._save_path_auto = False
        self.save_path_edit.setText(entry["path"])

    def get_values(self):
        """Wie _get_values_raw(), aber beim Fortsetzen aus dem gewaehlten
        Datensatz ueberschrieben - siehe resume_picker.apply_to_params()."""
        return self.resume_group.apply(self._get_values_raw())

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

    def _get_values_raw(self):
        """Returns all entered parameters as a dict (SI units: meters, Hz, K)."""
        return dict(
            win_input_range=(self.win_input_min.value() * 1e-3, self.win_input_max.value() * 1e-3),
            width_range=(self.width_min.value() * 1e6, self.width_max.value() * 1e6),
            n_points=self.n_points.value(),
            r_bounds=(self.r_min.value(), self.r_max.value()),
            alpha=self.alpha.value(),
            n_grid=self.n_grid.value(),
            atom_temperature=self.atom_temperature_uK.value() * 1e-6,
            trap_freq_r=self.trap_freq_r_kHz.value() * 1e3,
            weighted_n_grid=self.weighted_n_grid.value(),
            n_jobs=self.n_jobs.value(),
            force_cpu=self.force_cpu.isChecked(),
            enable_perf_log=self.enable_perf_log.isChecked(),
            airy_scale_factor=self.airy_group.value(),
            save_path=self.save_path_edit.text().strip(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = StartParametersDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)

    params = dialog.get_values()

    # ------------------------------------------------------------------
    # GPU auto-detect with automatic fallback to the parallelized CPU path -
    # identisches Muster wie im Original-Dialog, nur ueber
    # weighted_use_torch.py statt use_torch.py (patcht die Funktionszeiger im
    # RICHTIGEN, gewichteten Optimizer-Modul - siehe dortiger Docstring).
    # ------------------------------------------------------------------
    n_jobs = params["n_jobs"]
    pool_initializer = None
    use_gpu = False

    if not params["force_cpu"]:
        try:
            import weighted_use_torch as use_torch
            use_gpu = use_torch.cuda_available()
        except ImportError:
            use_gpu = False  # weighted_use_torch.py itself missing - fall back silently

    if use_gpu:
        print("GPU (CUDA via PyTorch) gefunden - nutze GPU-Beschleunigung (weighted_use_torch.patch()).")
        use_torch.patch()
        pool_initializer = use_torch.patch
        n_jobs = 1
    else:
        if params["force_cpu"]:
            print(f"CPU erzwungen (force_cpu) - nutze parallelisierte CPU mit n_jobs={n_jobs}.")
        else:
            print(f"Keine GPU/CUDA gefunden (torch nicht installiert oder keine CUDA-GPU) "
                  f"- nutze parallelisierte CPU mit n_jobs={n_jobs}.")

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
        airy_scale_factor=params["airy_scale_factor"],
    )

    save_path = params["save_path"]

    # Vor dem (potenziell tagelangen) Scan pruefen, ob unter dem gewaehlten
    # Speicherort bereits ein zu diesen Parametern passender Zwischenstand
    # liegt (z.B. von einem abgebrochenen vorherigen Lauf) - und den
    # Nutzer informieren, ob automatisch fortgesetzt oder neu gestartet wird.
    if save_path and Path(save_path).exists():
        resumable = scan_checkpoint.load_resumable(
            save_path, params["win_input_range"], params["width_range"],
            params["n_points"], params["n_points"], N_X_FIXED, N_Y_FIXED,
            extra_match=dict(alpha=params["alpha"], r_bounds=params["r_bounds"]),
            airy_scale_factor=params["airy_scale_factor"], verbose=False,
        )
        if resumable is not None:
            n_done = scan_checkpoint.count_done(resumable["uniformity_weighted_grid"])
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
                f"zu den aktuell gewählten Scan-Parametern (anderes Gitter/alpha/"
                f"r_bounds) oder konnte nicht gelesen werden.\n\nDer Scan startet "
                f"komplett neu und überschreibt die Datei beim nächsten "
                f"Zwischenspeichern.",
            )

    total_points = params["n_points"] * params["n_points"]
    progress = QProgressDialog("Computing weighted amplitude-dependence scan...", "Cancel", 0, total_points)
    progress.setWindowTitle("Weighted Amplitude-Dependence Scan Running")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)  # show immediately, even for short scans
    progress.setValue(0)

    def on_progress(done, total):
        progress.setValue(done)
        # keep the UI responsive, otherwise the bar freezes during computation
        QApplication.processEvents()
        if progress.wasCanceled():
            return False  # signals scan_win_width_amplitude_dependence_weighted() to cancel
        return True

    # Pure computation - no plotting happens here. This means re-running
    # just the plotting step later (e.g. from a saved .pkl) never needs to
    # repeat this scan.
    perf_measure = perf_log.Measurement()
    perf_measure.start()

    opt.scan_win_width_amplitude_dependence_weighted(
        win_input_range=params["win_input_range"],
        width_range=params["width_range"],
        n_win_input=params["n_points"],
        n_width=params["n_points"],
        r_bounds=params["r_bounds"],
        alpha=params["alpha"],
        progress_callback=on_progress,
        n_jobs=n_jobs,
        pool_initializer=pool_initializer,
        checkpoint_path=save_path or None,
    )

    duration = perf_measure.stop()
    perf_log.log(f"Optimization time: {duration / 1e9}s")

    progress.setValue(total_points)

    # Der Speicherort wurde bereits VOR dem Scan festgelegt (s.o.) und diente
    # dort bereits als checkpoint_path (stündliche Zwischenspeicherung) - hier
    # nur noch der finale, "saubere" Endstand (ohne Checkpoint-Markerfelder),
    # der denselben Pfad überschreibt. Kein erneuter Dateidialog mehr nötig.
    if save_path:
        # overwrite=True (Fix 2026-08-26, siehe Chat "Amplituden Abhängigkeit"):
        # unter save_path liegt bereits der (checkpoint_path-)Zwischenstand aus
        # dem Scan-Aufruf oben - der soll hier durch den sauberen Endstand
        # ERSETZT werden, nicht daneben eine verwirrende "_2"-Datei erzeugen
        # (vorher fehlte overwrite=True hier komplett - siehe overwrite-
        # Docstring von save_scan_amp_results_weighted()).
        opt.save_scan_amp_results_weighted(save_path, overwrite=True)
    else:
        opt.save_scan_amp_results_weighted()  # kein Pfad angegeben -> Results-Ordner-Default

    def qt_confirm_overwrite(existing_path):
        answer = QMessageBox.question(
            None, "File already exists",
            f"'{existing_path.name}' already exists. Overwrite it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    # out_dir bewusst NICHT gesetzt: AmplitudeScanPlotter speichert dann
    # automatisch in seinen eigenen DEFAULT_IMAGES_DIR (den "Bilder"-Ordner
    # neben weighted_multitone_amplitude_dependence_plots.py).
    plotter = AmplitudeScanPlotter(opt.get_scan_amp_results_weighted(), confirm_overwrite=qt_confirm_overwrite)
    plotter.plot_scan2d_combined(show=True, save=True)
    plotter.plot_dependence_cuts(show=True, save=True)


if __name__ == "__main__":
    main()
