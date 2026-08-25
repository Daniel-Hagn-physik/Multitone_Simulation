"""
Weighted_Optimization/weighted_winwidthscan_startdialog.py
================================================================

Port von `Winwidthscan startdialog.py` (Original-Ordner, "alte Version ohne
Amplitudenmodulation") fuer den ATOM-GEWICHTETEN Fest-Amplituden-Scan.
Amplituden werden hier - anders als bei weighted_winwidthampscan_startdialog.py
- NICHT pro Gitterpunkt optimiert, sondern vom Nutzer FEST vorgegeben (ein
Wert pro Ton in amp_x/amp_y); an jedem (win_input, width)-Gitterpunkt wird
nur AUSGEWERTET, nicht optimiert. Dafuer ruft dieses Skript
`opt.scan_win_width_weighted_uniformity()` (weighted_amp_scan_methods.py)
auf statt der harten `scan_win_width_uniformity()` - die berichteten
Uniformity_w/Crosstalk_w-Werte je Gitterpunkt sind die ATOM-THERMISCH
gewichtete Metrik (siehe weighted_multitone_flattop_optimizer.py), nicht die
harte Rechteck-Masken-Metrik.

Zweck (siehe Chat): diese Fest-Amplituden-Scans liefern die Datengrundlage
fuer den linearen waist(um)-width-Fit (`fit_waist_width_relation()`, "Teil 2"
von beispiel_weighted_amp_fit_abhaengigkeiten.py) - dafuer braucht es
mehrere Scans bei verschiedenen FESTEN Amplituden-Sets, nicht den
amplituden-optimierten Scan.

Einziger STRUKTURELLER Unterschied zum Original-Dialog: eine zusaetzliche
Gruppe "Atom Weighting" fragt nach den drei Parametern, die sigma_atom (die
Breite der Gauss-Gewichtung der tatsaechlichen thermischen Ortsverteilung
eines Atoms in der Falle) bestimmen - atom_temperature, trap_freq_r,
weighted_n_grid (Aufloesung des lokalen Sub-Grids um jede Site, siehe
_build_local_weighted_grid() in der Optimizer-Datei) - sowie ein "alpha"-Feld
fuer den kombinierten Score (alpha*uniformity_weighted + (1-alpha)*eta_weighted),
mit dem der "beste" Punkt im Scan bestimmt wird (im Original gibt es dieses
alpha nicht, weil dort gar kein "bester Punkt" markiert wird - hier schon,
siehe scan_win_width_weighted_uniformity()). Alles andere (Tonanzahl,
Amplituden pro Ton, Scan-Bereiche, Waist-Eingabemodus
vor/nach der Linse, n_grid) ist 1:1 identisch zum Original uebernommen.

NEU (2026-08-25): zusaetzlich zwei Felder "Atom offset x/y" (jeweils als
Bruchteil von pitch, dem raeumlichen Tonabstand, -0.5..0.5) in derselben
"Atom Weighting"-Gruppe. Damit laesst sich das Atom - und die 8 relativ zu
ihm ueber pitch platzierten Nachbar-Sites, die in die atom-gewichtete
Crosstalk-Metrik eingehen - VOR der Berechnung des Datensatzes um bis zu
einem halben Tonabstand in x UND y aus der Site-Mitte verschieben (0/0 =
bisheriges Verhalten, Atom exakt zentriert). Wird als atom_offset_x/
atom_offset_y (Meter) an den Optimizer durchgereicht - siehe
_evaluate_weighted_metrics() in weighted_multitone_flattop_optimizer.py fuer
die Details der Verschiebung.

GPU-Beschleunigung: wie bei den anderen Weighted_Optimization-Skripten
automatisch versucht (kein Zwang), aber ueber `weighted_use_torch.py` statt
`use_torch.py` - das Original-Modul wuerde die Funktionszeiger im FALSCHEN
Optimizer-Modul patchen und dadurch wirkungslos bleiben (siehe Docstring von
weighted_use_torch.py). Anders als im Original (dort unconditional/hart
`import use_torch; use_torch.patch()`, was ohne installiertes torch
crashen wuerde) wird hier - wie bei weighted_winwidthampscan_startdialog.py -
per try/except ImportError sauber auf CPU zurueckgefallen.

f1 = 75 mm und f2 = 750 mm sind fest (Default-Werte des Optimizers), wie im
Original.

Nach Bestaetigung berechnet
MultitoneFlatTopOptimizer.scan_win_width_weighted_uniformity() den Scan
(reine Berechnung, kein Plotten). Die Rohdaten werden anschliessend
gespeichert (Dateiname-Schema "scan_data_weighted_...pkl", damit der Scan
fuer spaeteres Neu-Plotten nie wiederholt werden muss) und an
weighted_multitone_amplitude_dependence_plots.WeightedFixedScanPlotter
uebergeben, der die Uniformity_w/Crosstalk_w-Heatmaps nebeneinander
zeichnet.

Nutzung:
    python weighted_winwidthscan_startdialog.py
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QFileDialog, QComboBox, QCheckBox,
)
from PyQt5.QtCore import Qt

from weighted_multitone_flattop_optimizer import (
    MultitoneFlatTopOptimizer, DEFAULT_RESULTS_DIR, DEFAULT_IMAGES_DIR,
)
import weighted_amp_scan_methods  # nur Import noetig - patcht scan_win_width_weighted_uniformity() etc. auf die Klasse
from weighted_multitone_amplitude_dependence_plots import WeightedFixedScanPlotter, win_input_to_win
import perf_log

# Feste Optik-Parameter fuer die Umrechnung win_input (vor der Linse) <->
# win_eff (nach der Linse/am Fokus) in diesem Dialog - identisch zum
# Original-Dialog. f1/f2 sind wie ueberall sonst in diesem Skript fest;
# fLO/lambda_opt kommen aus denselben Defaults, die auch der Optimizer
# intern benutzt.
_F1 = 75e-3
_F2 = 750e-3
_FLO = MultitoneFlatTopOptimizer.DEFAULTS['fLO']
_LAMBDA_OPT = MultitoneFlatTopOptimizer.DEFAULTS['lambda_opt']
# pitch = raeumlicher Tonabstand (Periodizitaet der Nachbar-Sites, siehe
# self.pitch/create_neighbourhood() in der Optimizer-Datei) - gebraucht, um
# den unten abgefragten Atom/Nachbar-Versatz als Bruchteil von pitch in
# Meter umzurechnen.
_PITCH = MultitoneFlatTopOptimizer.DEFAULTS['pitch']

# Mapping zwischen dem waist_mode-Wert dieses Dialogs ("win_input"/"win_eff",
# identisch zum Original) und dem win_axis-Argument von
# WeightedFixedScanPlotter.plot_scan2d_weighted_combined()
# ("before_lens"/"after_lens", andere Konvention als beim Original-ScanPlotter).
_WAIST_MODE_TO_WIN_AXIS = {"win_input": "before_lens", "win_eff": "after_lens"}


class StartParametersDialog(QDialog):
    """Fragt Tonanzahl, feste Amplituden pro Ton, Scan-Bereiche (win_input/
    width), Gitteraufloesung UND die Atom-Gewichtungsparameter ab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multitone FlatTop - Weighted Fixed-Amplitude Scan Parameters")

        self.amp_x_boxes = []
        self.amp_y_boxes = []

        main_layout = QVBoxLayout(self)

        # -- fixed optical parameters (display only) --
        info_label = QLabel(
            "Fixed optical parameters: f1 = 75 mm, f2 = 750 mm\n"
            "Amplitudes are FIXED per tone (no per-point optimization) - metric "
            "is ATOM-WEIGHTED (Uniformity_w/Crosstalk_w, see 'Atom Weighting' below)."
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
        self.scan_layout = scan_layout  # gebraucht, um Zeilen-Labels spaeter umzubenennen

        # Abfrage, auf welchen Waist sich die beiden Felder darunter beziehen
        # sollen - identisch zum Original (siehe dortiger Kommentar).
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
        scan_layout.addRow("Grid points per axis:", self.n_points)

        self.n_grid = QSpinBox()
        self.n_grid.setRange(50, 2500)
        self.n_grid.setValue(1000)
        self.n_grid.setToolTip(
            "Resolution of the GLOBAL intensity grid (used by geometry/lens\n"
            "helper methods, not by the weighted evaluation itself - see\n"
            "'Atom Weighting' -> weighted_n_grid below for the grid that\n"
            "actually matters for uniformity_weighted/eta_weighted)."
        )
        scan_layout.addRow("Intensity grid resolution (n_grid):", self.n_grid)

        self.alpha = self._make_spin(0.7, 0.0, 1.0, 0.05)
        self.alpha.setToolTip(
            "Weight of the combined score used ONLY to mark the 'best' grid\n"
            "point in the plots: alpha*uniformity_weighted + (1-alpha)*eta_weighted.\n"
            "Default 0.9 (same convention as scan_win_width_weighted_uniformity()).\n"
            "Does NOT affect the amplitudes themselves - they stay fixed as\n"
            "entered above."
        )
        scan_layout.addRow("alpha (best-point marker):", self.alpha)

        scan_group.setLayout(scan_layout)
        main_layout.addWidget(scan_group)

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
            "Radial trap frequency nu_r (kHz, LINEAR frequency - the optimizer\n"
            "converts internally to omega_r = 2*pi*nu_r). Higher nu_r -> tighter\n"
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

        offset_info = QLabel(
            f"Optional: shifts the atom - and with it the local sub-grid used\n"
            f"for uniformity_weighted/eta_weighted, i.e. the atom AND the 8\n"
            f"neighbor-site images it sees - away from the exact site center,\n"
            f"before the dataset is computed. Given as a fraction of the tone\n"
            f"spacing pitch (currently {_PITCH * 1e6:.3f} um); 0 = atom exactly on\n"
            f"the site (previous/default behavior)."
        )
        offset_info.setStyleSheet("font-style: italic;")
        atom_layout.addRow(offset_info)

        self.atom_offset_frac_x = self._make_spin(0.0, -0.5, 0.5, 0.05)
        self.atom_offset_frac_x.setToolTip(
            "Atom/neighbor offset in x, as a fraction of the tone spacing\n"
            "(pitch). Range +/- 0.5 = up to half a tone spacing in either\n"
            "direction. Applied to atom_offset_x (meters) on the optimizer."
        )
        atom_layout.addRow("Atom offset x (x pitch, -0.5..0.5):", self.atom_offset_frac_x)

        self.atom_offset_frac_y = self._make_spin(0.0, -0.5, 0.5, 0.05)
        self.atom_offset_frac_y.setToolTip(
            "Atom/neighbor offset in y, as a fraction of the tone spacing\n"
            "(pitch). Range +/- 0.5 = up to half a tone spacing in either\n"
            "direction. Applied to atom_offset_y (meters) on the optimizer."
        )
        atom_layout.addRow("Atom offset y (x pitch, -0.5..0.5):", self.atom_offset_frac_y)

        atom_group.setLayout(atom_layout)
        main_layout.addWidget(atom_group)

        # -- GPU acceleration / logging (monkey-patch, see weighted_use_torch.py) --
        gpu_group = QGroupBox("GPU Acceleration")
        gpu_layout = QFormLayout()

        gpu_info = QLabel(
            "By default, GPU acceleration (CUDA via PyTorch,\n"
            "weighted_use_torch.patch()) is tried automatically at scan\n"
            "start. If torch is not installed or no CUDA GPU is found, it\n"
            "falls back to the plain CPU path automatically - no action\n"
            "needed either way."
        )
        gpu_info.setStyleSheet("font-style: italic;")
        gpu_layout.addRow(gpu_info)

        self.force_cpu = QCheckBox("Force CPU only (skip GPU auto-detect)")
        self.force_cpu.setToolTip(
            "Skips the automatic GPU auto-detect/patch entirely and always\n"
            "uses the plain NumPy path. Useful for a reproducible baseline\n"
            "run, or if you specifically want to avoid the not-yet-verified\n"
            "CUDA/PyTorch code path (see weighted_use_torch.py docstring)."
        )
        gpu_layout.addRow(self.force_cpu)

        self.enable_perf_log = QCheckBox("Enable perf logging")
        self.enable_perf_log.setToolTip(
            "Prints total optimization time via perf_log - useful to\n"
            "compare CUDA on/off runs."
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

    def _on_waist_mode_changed(self):
        """Wie im Original: wechselt die Beschriftung der win_input-min/max-
        Felder zwischen 'vor der Linse' (mm) und 'nach der Linse' (µm) und
        rechnet die aktuell eingetragenen Werte passend um."""
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
        """Returns all entered parameters as a dict (SI units: meters, Hz, K).

        win_input_range wird IMMER in win_input (vor der Linse, Meter)
        zurueckgegeben, egal welcher Modus in waist_mode_combo gewaehlt war -
        scan_win_width_weighted_uniformity() erwartet ausschliesslich
        win_input (identisches Verhalten zum Original-Dialog).
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
            alpha=self.alpha.value(),
            atom_temperature=self.atom_temperature_uK.value() * 1e-6,
            trap_freq_r=self.trap_freq_r_kHz.value() * 1e3,
            weighted_n_grid=self.weighted_n_grid.value(),
            atom_offset_x=self.atom_offset_frac_x.value() * _PITCH,
            atom_offset_y=self.atom_offset_frac_y.value() * _PITCH,
            force_cpu=self.force_cpu.isChecked(),
            enable_perf_log=self.enable_perf_log.isChecked(),
            # welche Achse beim direkten Plotten am Ende von main() genutzt
            # werden soll - standardmaessig dieselbe, auf die sich der
            # Scan-Bereich hier bezog
            waist_mode=self._current_waist_mode,
        )


def main():
    app = QApplication(sys.argv)

    dialog = StartParametersDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)

    params = dialog.get_values()
    amps = np.concatenate([params["amp_x"], params["amp_y"]])

    # ------------------------------------------------------------------
    # GPU auto-detect mit automatischem CPU-Fallback - ueber
    # weighted_use_torch.py statt use_torch.py (patcht die Funktionszeiger im
    # RICHTIGEN, gewichteten Optimizer-Modul - siehe dortiger Docstring).
    # Anders als im Original-Dialog (hartes, ungeschuetztes
    # `import use_torch; use_torch.patch()`) sauber per try/except, damit das
    # Skript auch ohne installiertes torch laeuft.
    # ------------------------------------------------------------------
    use_gpu = False
    if not params["force_cpu"]:
        try:
            import weighted_use_torch as use_torch
            use_gpu = use_torch.cuda_available()
        except ImportError:
            use_gpu = False  # weighted_use_torch.py fehlt - stiller Fallback auf CPU

    if use_gpu:
        print("GPU (CUDA via PyTorch) gefunden - nutze GPU-Beschleunigung (weighted_use_torch.patch()).")
        use_torch.patch()
    else:
        if params["force_cpu"]:
            print("CPU erzwungen (force_cpu) - nutze CPU (NumPy).")
        else:
            print("Keine GPU/CUDA gefunden (torch nicht installiert oder keine CUDA-GPU) - nutze CPU (NumPy).")

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

    total_points = params["n_points"] * params["n_points"]
    progress = QProgressDialog("Computing weighted uniformity & crosstalk scan...", "Cancel", 0, total_points)
    progress.setWindowTitle("Weighted Fixed-Amplitude 2D Scan Running")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)  # show immediately, even for short scans
    progress.setValue(0)

    def on_progress(done, total):
        progress.setValue(done)
        # keep the UI responsive, otherwise the bar freezes during computation
        QApplication.processEvents()
        if progress.wasCanceled():
            return False  # signals scan_win_width_weighted_uniformity() to cancel
        return True

    perf_measure = perf_log.Measurement()
    perf_measure.start()

    # Pure computation - no plotting happens here. This means re-running
    # just the plotting step later (e.g. from a saved .pkl) never needs to
    # repeat this (potentially slow) scan.
    opt.scan_win_width_weighted_uniformity(
        win_input_range=params["win_input_range"],
        width_range=params["width_range"],
        n_win_input=params["n_points"],
        n_width=params["n_points"],
        amps=amps,
        alpha=params["alpha"],
        progress_callback=on_progress,
    )

    duration = perf_measure.stop()
    perf_log.log(f"Optimization time: {duration/1e9}s")

    progress.setValue(total_points)

    # Persist the raw results so this exact scan never has to be re-run
    # just to change plot styling later. Defaults to DEFAULT_RESULTS_DIR mit
    # auto-generiertem Namen (siehe save_scan_weighted_results()); der Nutzer
    # kann in der Dialogbox weiterhin einen anderen Ort/Namen waehlen.
    profile_tag = "Airy" if opt.profile == "airy" else "Gauss" if opt.profile == "gaussian" else opt.profile
    default_scan_data_path = str(
        DEFAULT_RESULTS_DIR / f"scan_data_weighted_N{params['N_x']}x{params['N_y']}_"
                               f"{params['n_points']}x{params['n_points']}pts_{profile_tag}.pkl"
    )
    scan_data_path = QFileDialog.getSaveFileName(
        None, "Save weighted scan data (for re-plotting later)", default_scan_data_path, "Pickle files (*.pkl)"
    )[0]
    if scan_data_path:
        opt.save_scan_weighted_results(scan_data_path)
    else:
        opt.save_scan_weighted_results()  # user cancelled the dialog -> fall back to the Results-folder default

    def qt_confirm_overwrite(existing_path):
        answer = QMessageBox.question(
            None, "File already exists",
            f"'{existing_path.name}' already exists. Overwrite it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    # out_dir bewusst NICHT gesetzt: WeightedFixedScanPlotter speichert dann
    # automatisch in seinen eigenen DEFAULT_IMAGES_DIR (den "Bilder"-Ordner
    # neben weighted_multitone_amplitude_dependence_plots.py).
    plotter = WeightedFixedScanPlotter(opt.get_scan_weighted_results(), confirm_overwrite=qt_confirm_overwrite)
    plotter.plot_scan2d_weighted_combined(
        show=True, save=True, win_axis=_WAIST_MODE_TO_WIN_AXIS[params["waist_mode"]],
    )


if __name__ == "__main__":
    main()
