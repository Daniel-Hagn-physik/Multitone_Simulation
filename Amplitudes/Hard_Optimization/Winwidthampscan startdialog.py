"""
Start dialog for the amplitude-dependence scan (input waist before 1st lens
vs. width -> optimal amplitude ratios r_x, r_y that minimize Uniformity and
Crosstalk together).

Unlike WinWidthScan_StartDialog.py (fixed amplitudes, scans Uniformity/
Crosstalk directly), this dialog does NOT ask for per-tone amplitudes.
Instead, for the fixed tone count N_x=3 / N_y=4, amplitudes follow the
symmetric outer/inner parametrization (see amps_from_ratio() in
multitone_flattop_optimizer.py):

    N_x=3:  amp_x = [r_x, 1, r_x]        (outer tones = r_x, center = 1)
    N_y=4:  amp_y = [r_y, 1, 1, r_y]     (outer tones = r_y, inner pair = 1)

At EACH (win_input, width) grid point, r_x and r_y are themselves optimized
to minimize alpha*uniformity + (1-alpha)*eta there (default alpha=0.7 -
see MultitoneFlatTopOptimizer.scan_win_width_amplitude_dependence()). The
result - r_x_opt(win_input, width) and r_y_opt(win_input, width) - is the
"amplitude dependence on waist and width" the project is about.

Asks for:
- Scan ranges for win_input (input waist before the 1st lens, in mm) and
  width (frequency spacing, in MHz)
- Number of grid points per axis (kept low by default - this scan runs an
  inner optimization at EVERY point, so it is much slower than the plain
  uniformity/crosstalk scan)
- Bounds for r_x / r_y (both share the same bounds, since both are
  outer/inner ratios of the same kind)
- Intensity grid resolution (n_grid), same meaning/cost tradeoff as in
  WinWidthScan_StartDialog.py
- Number of parallel processes (n_jobs), used as the CPU fallback (see
  below). Every (win_input, width) grid point is fully independent, so
  this scan parallelizes across CPU cores almost linearly. n_jobs>1
  disables the warm-start between neighbouring points (see
  scan_win_width_amplitude_dependence() docstring for details).

GPU acceleration (automatic, no checkbox needed): at scan start, main()
tries CUDA/PyTorch acceleration first via use_torch.cuda_available() +
use_torch.patch() (see use_torch.py, and the dyn_gaussian_.../
dyn_airy_..._distance_from_centers monkey-patch pointers in
multitone_flattop_optimizer.py). If torch is not installed, or is
installed but finds no CUDA GPU, this silently falls back to the
parallelized CPU path above (n_jobs) instead - no user action required
either way. When GPU acceleration IS used, n_jobs is forced to 1 (a
single process already benefits from the GPU-accelerated _evaluate()
calls; multiple processes would each open their own CUDA context on the
same GPU, which is unnecessary overhead here). A "Force CPU only"
checkbox lets you skip the GPU auto-detect entirely, e.g. for a
reproducible baseline run.

f1 = 75 mm and f2 = 750 mm are fixed (default values of the optimizer, see
MultitoneFlatTopOptimizer.DEFAULTS). N_x=3, N_y=4 are fixed, matching the
amplitude parametrization above.

After confirmation, MultitoneFlatTopOptimizer.scan_win_width_amplitude_dependence()
computes the scan (pure computation, no plotting). The raw results are then
saved to disk (so the scan never has to be re-run just to re-style a plot
later) and handed to
multitone_amplitude_dependence_plots.AmplitudeScanPlotter, which draws the
combined heatmaps (Uniformity, Crosstalk, r_x, r_y) plus the dependence-cut
line plots.

Usage:
    python WinWidthAmpScan_StartDialog.py
"""

import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QFileDialog, QCheckBox,
)
from PyQt5.QtCore import Qt

from multitone_flattop_optimizer import MultitoneFlatTopOptimizer, DEFAULT_RESULTS_DIR, DEFAULT_IMAGES_DIR
from multitone_amplitude_dependence_plots import AmplitudeScanPlotter
import perf_log


N_X_FIXED = 3
N_Y_FIXED = 4


class StartParametersDialog(QDialog):
    """Asks for the scan ranges (win_input/width), r-bounds, and grid resolution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multitone FlatTop - Amplitude-Dependence Scan Parameters")

        main_layout = QVBoxLayout(self)

        # -- fixed parameters (display only) --
        info_label = QLabel(
            f"Fixed: f1 = 75 mm, f2 = 750 mm, N_x = {N_X_FIXED}, N_y = {N_Y_FIXED}\n"
            f"Amplitudes: amp_x = [r_x, 1, r_x],  amp_y = [r_y, 1, 1, r_y]  "
            f"(outer/inner ratio, inner = 1)"
        )
        info_label.setStyleSheet("font-style: italic;")
        main_layout.addWidget(info_label)

        # -- scan ranges --
        scan_group = QGroupBox("Scan Ranges")
        scan_layout = QFormLayout()

        self.win_input_min = self._make_spin(0.5, 0.001, 50.0, 0.05)
        self.win_input_max = self._make_spin(1.75, 0.001, 50.0, 0.05)
        scan_layout.addRow("win_input min (mm):", self.win_input_min)
        scan_layout.addRow("win_input max (mm):", self.win_input_max)

        self.width_min = self._make_spin(0.2, 0.001, 5.0, 0.01)
        self.width_max = self._make_spin(0.4, 0.001, 5.0, 0.01)
        scan_layout.addRow("width min (MHz):", self.width_min)
        scan_layout.addRow("width max (MHz):", self.width_max)

        self.n_points = QSpinBox()
        self.n_points.setRange(3, 100)
        self.n_points.setValue(15)
        self.n_points.setToolTip(
            "Grid points per axis. Unlike the plain uniformity/crosstalk scan,\n"
            "EVERY grid point here runs its own (r_x, r_y) optimization\n"
            "(~30-100 evaluations) - keep this low (12-20) for a first pass."
        )
        scan_layout.addRow("Grid points per axis:", self.n_points)

        scan_group.setLayout(scan_layout)
        main_layout.addWidget(scan_group)

        # -- amplitude-ratio bounds & optimization settings --
        amp_group = QGroupBox("Amplitude-Ratio Optimization")
        amp_layout = QFormLayout()

        self.r_min = self._make_spin(0.0, 0.0, 10.0, 0.05)
        self.r_max = self._make_spin(2.0, 0.0, 10.0, 0.05)
        amp_layout.addRow("r_x / r_y min:", self.r_min)
        amp_layout.addRow("r_x / r_y max:", self.r_max)

        self.alpha = self._make_spin(0.7, 0.0, 1.0, 0.05)
        self.alpha.setToolTip(
            "Weight of the per-point objective: alpha*uniformity + (1-alpha)*eta.\n"
            "Default 0.7 - Uniformity AND Crosstalk are minimized together\n"
            "(alpha=1.0 would minimize Uniformity only).\n"
            "Crosstalk at the found optimum is recorded either way."
        )
        amp_layout.addRow("alpha (0.7 = uniformity+crosstalk):", self.alpha)

        amp_group.setLayout(amp_layout)
        main_layout.addWidget(amp_group)

        # -- intensity grid resolution --
        grid_group = QGroupBox("Intensity Grid")
        grid_layout = QFormLayout()
        self.n_grid = QSpinBox()
        self.n_grid.setRange(50, 2000)
        self.n_grid.setValue(200)
        self.n_grid.setToolTip(
            "Resolution of the intensity grid used for EACH _evaluate() call.\n"
            "Cost scales as n_grid^2, multiplied by scan points AND inner\n"
            "optimization evaluations - keep this moderate (150-250)."
        )
        grid_layout.addRow("n_grid:", self.n_grid)
        grid_group.setLayout(grid_layout)
        main_layout.addWidget(grid_group)

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

        # -- GPU acceleration / logging (monkey-patch, see use_torch.py) --
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QFormLayout()

        gpu_info = QLabel(
            "By default, GPU acceleration (CUDA via PyTorch, use_torch.patch())\n"
            "is tried automatically at scan start. If torch is not installed or\n"
            "no CUDA GPU is found, it falls back to the parallelized CPU path\n"
            "(n_jobs below) automatically - no action needed either way."
        )
        gpu_info.setStyleSheet("font-style: italic;")
        gpu_layout.addRow(gpu_info)

        self.force_cpu = QCheckBox("Force CPU only (skip GPU auto-detect)")
        self.force_cpu.setToolTip(
            "Skips the automatic GPU auto-detect/patch entirely and always\n"
            "uses the plain NumPy path (optionally parallelized via n_jobs\n"
            "above). Useful for a reproducible baseline run, or if you\n"
            "specifically want to avoid the not-yet-verified CUDA/PyTorch\n"
            "code path (see use_torch.py docstring)."
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
        self.accept()

    def get_values(self):
        """Returns all entered parameters as a dict (SI units: meters, Hz)."""
        return dict(
            win_input_range=(self.win_input_min.value() * 1e-3, self.win_input_max.value() * 1e-3),
            width_range=(self.width_min.value() * 1e6, self.width_max.value() * 1e6),
            n_points=self.n_points.value(),
            r_bounds=(self.r_min.value(), self.r_max.value()),
            alpha=self.alpha.value(),
            n_grid=self.n_grid.value(),
            n_jobs=self.n_jobs.value(),
            force_cpu=self.force_cpu.isChecked(),
            enable_perf_log=self.enable_perf_log.isChecked(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = StartParametersDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)

    params = dialog.get_values()

    # ------------------------------------------------------------------
    # GPU auto-detect with automatic fallback to the parallelized CPU path:
    # by default (force_cpu unchecked) we TRY CUDA/PyTorch first via
    # use_torch.cuda_available() (which never raises, even if torch is not
    # installed at all). Only if that comes back False (no torch, or torch
    # without a usable CUDA GPU) do we fall back to the plain NumPy path,
    # parallelized across n_jobs processes as configured above.
    #
    # When GPU acceleration IS used, n_jobs is forced to 1: each process
    # would open its own CUDA context on the same physical GPU, which adds
    # memory/init overhead and is unnecessary here anyway - the GPU patch
    # already accelerates every single _evaluate() call, so a single
    # (still warm-started) process already gets the benefit. If you
    # deliberately want multiple processes EACH using the GPU, that is
    # still possible from a script via pool_initializer=use_torch.patch,
    # just not exposed as a GUI default here.
    # ------------------------------------------------------------------
    n_jobs = params["n_jobs"]
    pool_initializer = None
    use_gpu = False

    if not params["force_cpu"]:
        try:
            import use_torch
            use_gpu = use_torch.cuda_available()
        except ImportError:
            use_gpu = False  # use_torch.py itself missing - fall back silently

    if use_gpu:
        print("GPU (CUDA via PyTorch) gefunden - nutze GPU-Beschleunigung (use_torch.patch()).")
        use_torch.patch()
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
    )

    total_points = params["n_points"] * params["n_points"]
    progress = QProgressDialog("Computing amplitude-dependence scan...", "Cancel", 0, total_points)
    progress.setWindowTitle("Amplitude-Dependence Scan Running")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)  # show immediately, even for short scans
    progress.setValue(0)

    def on_progress(done, total):
        progress.setValue(done)
        # keep the UI responsive, otherwise the bar freezes during computation
        QApplication.processEvents()
        if progress.wasCanceled():
            return False  # signals scan_win_width_amplitude_dependence() to cancel
        return True

    # Pure computation - no plotting happens here. This means re-running
    # just the plotting step later (e.g. from a saved .pkl) never needs to
    # repeat this (much more expensive than the plain uniformity/crosstalk
    # scan, since every point runs its own r_x/r_y optimization) scan.
    perf_measure = perf_log.Measurement()
    perf_measure.start()

    opt.scan_win_width_amplitude_dependence(
        win_input_range=params["win_input_range"],
        width_range=params["width_range"],
        n_win_input=params["n_points"],
        n_width=params["n_points"],
        r_bounds=params["r_bounds"],
        alpha=params["alpha"],
        progress_callback=on_progress,
        n_jobs=n_jobs,
        pool_initializer=pool_initializer,
    )

    duration = perf_measure.stop()
    perf_log.log(f"Optimization time: {duration / 1e9}s")

    progress.setValue(total_points)

    # Persist the raw results so this exact (usually slow) scan never has to
    # be re-run just to change plot styling later. Filename encodes tone
    # count, grid resolution AND the active beam profile (Airy/Gauss), so
    # raw-data files and their re-plotted images stay unambiguous - matches
    # the auto-generated name save_scan_amp_results() would pick anyway.
    profile_tag = "Airy" if opt.profile == "airy" else "Gauss" if opt.profile == "gaussian" else opt.profile
    default_scan_data_path = str(
        DEFAULT_RESULTS_DIR / f"scan_amp_data_N{N_X_FIXED}x{N_Y_FIXED}_"
                               f"{params['n_points']}x{params['n_points']}pts_{profile_tag}.pkl"
    )
    scan_data_path = QFileDialog.getSaveFileName(
        None, "Save amplitude-dependence scan data (for re-plotting later)",
        default_scan_data_path, "Pickle files (*.pkl)"
    )[0]
    if scan_data_path:
        opt.save_scan_amp_results(scan_data_path)
    else:
        opt.save_scan_amp_results()  # user cancelled the dialog -> fall back to the Results-folder default

    def qt_confirm_overwrite(existing_path):
        answer = QMessageBox.question(
            None, "File already exists",
            f"'{existing_path.name}' already exists. Overwrite it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    # out_dir bewusst NICHT gesetzt: AmplitudeScanPlotter speichert dann
    # automatisch in seinen eigenen DEFAULT_IMAGES_DIR (den "Bilder"-Ordner
    # neben multitone_amplitude_dependence_plots.py).
    plotter = AmplitudeScanPlotter(opt.get_scan_amp_results(), confirm_overwrite=qt_confirm_overwrite)
    plotter.plot_scan2d_combined(show=True, save=True)
    plotter.plot_dependence_cuts(show=True, save=True)


if __name__ == "__main__":
    main()