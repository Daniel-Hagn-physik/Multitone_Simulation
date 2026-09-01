"""
run_penalty_scan.py  -  NEUE DATEN MIT DER PENALTY-METHODE ERZEUGEN
===================================================================

    ==> Dieses Skript ausfuehren, wenn ein NEUER Datensatz gescannt
    ==> werden soll, bei dem die Amplituden gemeinsam auf hart UND
    ==> atom-gewichtet optimiert werden (Penalty-Term).

Was passiert: an JEDEM (win_input, width)-Gitterpunkt laeuft GENAU EINE
(r_x, r_y)-Optimierung, die direkt gegen die Kombination aus hartem und
atom-gewichtetem Ziel minimiert:

    U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
    C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
    J       = alpha*U_kombi + (1-alpha)*C_kombi        ->  min

Der Term combo_lambda*|Differenz| ist der Penalty-Term: er bestraft
Amplituden, bei denen die beiden Kriterien auseinanderlaufen. Hart und
gewichtet werden bei jedem Optimierungsschritt am SELBEN (r_x, r_y)
ausgewertet - die gefundene Amplitude gilt daher fuer beide zugleich.

Ergebnis: Results/scan_amp_data_combined_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl
(dasselbe Namensmuster wie bisher, vorhandene Datensaetze passen dazu).

Der Scan kann lange laufen. Der Speicherort wird deshalb VOR dem Start
abgefragt, stuendlich zwischengespeichert, und ein erneuter Start mit
denselben Parametern und demselben Pfad setzt automatisch dort fort, wo
er stehen geblieben ist.

Die anderen beiden Hauptskripte:
    run_hard_check.py  -  vorhandenen GEWICHTETEN Scan hart nachrechnen
    run_plots.py       -  vorhandene Datensaetze plotten/auswerten
"""

import os
import sys
import time
from pathlib import Path as FilePath

from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QFileDialog, QCheckBox, QLineEdit, QComboBox, QScrollArea,
    QWidget,
)
from PyQt5.QtCore import Qt

sys.path.insert(0, str(FilePath(__file__).resolve().parent))

# Alles, was dieses Skript braucht, kommt aus lib - auch der Optimierer und
# die Zwischenspeicherung, die eigentlich in ../Weighted_Optimization liegen
# (lib/penalty_scan.py kapselt sie, siehe Hinweis in lib/paths.py).
from lib import paths, combine, penalty_scan, report  # noqa: E402
import resume_picker            # aus ../Weighted_Optimization (siehe lib/paths.py)


N_X_FIXED = 3
N_Y_FIXED = 4


class PenaltyScanDialog(QDialog):
    """Parameter fuer den Penalty-Scan. Der gesamte Inhalt liegt in einer
    QScrollArea, Start/Cancel bleiben ausserhalb und damit immer
    sichtbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Penalty-Scan - gemeinsame Amplituden-Optimierung (hart + gewichtet)")

        outer_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        main_layout = QVBoxLayout(content)
        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)

        # -- Fortsetzen: unfertigen Datensatz auswaehlen --
        # Steht bewusst GANZ OBEN: die Entscheidung "neu oder fortsetzen"
        # kommt vor allen Parametern, denn beim Fortsetzen kommen die
        # Parameter aus der Datei und die Felder darunter werden gesperrt.
        self.resume_group = resume_picker.ResumePickerGroup(
            paths.DEFAULT_RESULTS_DIR, kind=resume_picker.KIND_PENALTY,
            on_change=self._on_resume_changed)
        main_layout.addWidget(self.resume_group)


        info = QLabel(
            f"Fest: f1 = 75 mm, f2 = 750 mm, N_x = {N_X_FIXED}, N_y = {N_Y_FIXED}\n"
            f"Amplituden: amp_x = [r_x, 1, r_x],  amp_y = [r_y, 1, 1, r_y]\n\n"
            f"Pro Gitterpunkt EINE (r_x, r_y)-Optimierung direkt gegen\n"
            f"    J = alpha*U_kombi + (1-alpha)*C_kombi,\n"
            f"    X_kombi = 0.5*(X_hart + X_gewichtet) + combo_lambda*|X_hart - X_gewichtet|.\n"
            f"combo_lambda ist der Penalty-Term: er bestraft Amplituden, bei denen das harte\n"
            f"und das atom-gewichtete Kriterium auseinanderlaufen."
        )
        info.setStyleSheet("font-style: italic;")
        main_layout.addWidget(info)

        # -- Scan-Bereiche --
        scan_group = QGroupBox("Scan-Bereiche")
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
        self.n_points.setValue(21)
        self.n_points.setToolTip(
            "Gitterpunkte pro Achse. JEDER Gitterpunkt bekommt eine eigene\n"
            "(r_x, r_y)-Optimierung - fuer einen ersten Durchlauf eher\n"
            "niedrig ansetzen (12-21)."
        )
        scan_layout.addRow("Gitterpunkte pro Achse:", self.n_points)
        scan_group.setLayout(scan_layout)
        main_layout.addWidget(scan_group)

        # -- Amplituden-Optimierung --
        amp_group = QGroupBox("Amplituden-Optimierung")
        amp_layout = QFormLayout()
        self.r_min = self._make_spin(0.1, 0.0, 10.0, 0.05)
        self.r_max = self._make_spin(10.0, 0.0, 10.0, 0.05)
        amp_layout.addRow("r_x / r_y min:", self.r_min)
        amp_layout.addRow("r_x / r_y max:", self.r_max)
        self.alpha = self._make_spin(0.7, 0.0, 1.0, 0.05)
        self.alpha.setToolTip(
            "Gewicht von Uniformity gegenueber Crosstalk in der Zielfunktion\n"
            "J = alpha*U_kombi + (1-alpha)*C_kombi. Wird auch fuer die\n"
            "anschliessende Score-/Region-Auswertung verwendet."
        )
        amp_layout.addRow("alpha (Uniformity vs. Crosstalk):", self.alpha)
        amp_group.setLayout(amp_layout)
        main_layout.addWidget(amp_group)

        # -- Penalty / Kombination --
        combo_group = QGroupBox("Penalty-Term und Region")
        combo_layout = QFormLayout()
        combo_info = QLabel(
            "combo_lambda = 0: reiner Mittelwert beider Kriterien (kein Penalty).\n"
            "Groesser: Amplituden, bei denen hart und gewichtet auseinanderlaufen,\n"
            "werden staerker bestraft.\n"
            "combo_percentile steuert nur die Auswertung: die Region ist das groesste\n"
            "Rechteck innerhalb der besten combo_percentile% aller Punkte."
        )
        combo_info.setStyleSheet("font-style: italic;")
        combo_layout.addRow(combo_info)
        self.combo_lambda = self._make_spin(0.75, 0.0, 5.0, 0.05)
        combo_layout.addRow("combo_lambda (Penalty-Gewicht):", self.combo_lambda)
        self.combo_percentile = self._make_spin(25.0, 1.0, 100.0, 5.0)
        combo_layout.addRow("combo_percentile (% beste Punkte):", self.combo_percentile)
        combo_group.setLayout(combo_layout)
        main_layout.addWidget(combo_group)

        # -- Intensitaetsgitter --
        grid_group = QGroupBox("Intensitaetsgitter")
        grid_layout = QFormLayout()
        self.n_grid = QSpinBox()
        self.n_grid.setRange(50, 5000)
        self.n_grid.setValue(1000)
        self.n_grid.setToolTip("Aufloesung des GLOBALEN Gitters (harte Metriken).")
        grid_layout.addRow("n_grid:", self.n_grid)
        grid_group.setLayout(grid_layout)
        main_layout.addWidget(grid_group)

        # -- Strahlprofil --
        profile_group = QGroupBox("Strahlprofil (Airy)")
        profile_layout = QFormLayout()
        profile_info = QLabel(
            "Das Profil in der Ebene ist Airy. Der Skalenfaktor uebersetzt den\n"
            "Scan-Parameter waist in die Airy-Skala:\n"
            "    first_zero_radius = Faktor * waist")
        profile_info.setStyleSheet("font-style: italic;")
        profile_layout.addRow(profile_info)
        self.airy_scale_mode = QComboBox()
        for _key, label, _wert in combine.AIRY_SCALE_CHOICES:
            self.airy_scale_mode.addItem(label)
        self.airy_scale_mode.setToolTip(
            "Was die Zahl 'waist' physikalisch bedeuten soll.\n\n"
            "  wie bisher (1.19): der historische Default. Der tatsaechliche\n"
            "  1/e^2-Radius des Airy-Profils liegt dann bei 0.8025 * waist.\n\n"
            "  1/e^2 der Airy-Hauptkeule = waist (1.4830): der Waist bedeutet\n"
            "  dann bei Airy dasselbe wie bei einem Gauss-Strahl.\n\n"
            "  frei eingeben: eigener Wert im Feld darunter.")
        self.airy_scale_mode.currentIndexChanged.connect(
            lambda _i: self._sync_airy_mode())
        profile_layout.addRow("Parametrisierung:", self.airy_scale_mode)
        self.airy_scale_factor = self._make_spin(1.19, 0.1, 5.0, 0.01)
        self.airy_scale_factor.setDecimals(6)
        self.airy_scale_factor.setToolTip(
            "first_zero_radius = Faktor * waist. Der Faktor setzt die physikalische\n"
            "Spotgroesse und damit JEDE Metrik dieses Scans - nicht nur den verbotenen\n"
            "Bereich in der Auswertung.\n"
            "\n"
            "Der bisherige Default 1.19 entspricht KEINER der gaengigen Konventionen\n"
            "(nachgerechnet):\n"
            "   gleicher 1/e^2-Radius wie ein Gauss  -> 1.4830\n"
            "   bester Gauss-Fit an die Hauptkeule   -> 1.4499\n"
            "   gleiche FWHM                         -> 1.3956\n"
            "   gleicher Radius fuer 50 % Leistung   -> 1.3425\n"
            "\n"
            "Bei 1.19 liegt der tatsaechliche 1/e^2-Radius des Airy-Profils bei\n"
            "0.8025 * waist, also 20 % unter der Zahl, die spaeter 'waist' heisst.\n"
            "\n"
            "Der gewaehlte Wert wird in die .pkl geschrieben; run_hard_check.py\n"
            "uebernimmt ihn von dort, damit nicht zwei verschiedene Optiken\n"
            "verglichen werden.")
        profile_layout.addRow("airy_scale_factor:", self.airy_scale_factor)
        self.airy_scale_hint = QLabel("")
        self.airy_scale_hint.setWordWrap(True)
        self.airy_scale_hint.setStyleSheet("color: gray;")
        profile_layout.addRow(self.airy_scale_hint)
        self.airy_scale_factor.valueChanged.connect(lambda _v: self._sync_airy_hint())
        profile_group.setLayout(profile_layout)
        main_layout.addWidget(profile_group)
        self._sync_airy_mode()

        # -- Atom-Gewichtung --
        atom_group = QGroupBox("Atom-Gewichtung (fuer den gewichteten Anteil)")
        atom_layout = QFormLayout()
        atom_info = QLabel("sigma_atom wird aus atom_temperature und trap_freq_r berechnet\n"
                           "(atom_mass fest auf Rb-85).")
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

        # -- Parallelisierung / GPU --
        cpu_count = os.cpu_count() or 1
        parallel_group = QGroupBox("Parallelisierung und GPU")
        parallel_layout = QFormLayout()
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(1, max(1, cpu_count))
        self.n_jobs.setValue(max(1, cpu_count - 1))
        self.n_jobs.setToolTip(
            f"Anzahl paralleler Prozesse (dieser Rechner meldet {cpu_count} CPU(s)).\n"
            f"n_jobs=1 behaelt den sequentiellen Warm-Start bei (Startwert der\n"
            f"Optimierung = Optimum des Vorgaengerpunktes); n_jobs>1 nicht."
        )
        parallel_layout.addRow(f"n_jobs (1-{cpu_count}):", self.n_jobs)
        self.force_cpu = QCheckBox("GPU-Erkennung ueberspringen (nur CPU)")
        parallel_layout.addRow(self.force_cpu)
        self.enable_perf_log = QCheckBox("Perf-Logging aktivieren")
        parallel_layout.addRow(self.enable_perf_log)
        parallel_group.setLayout(parallel_layout)
        main_layout.addWidget(parallel_group)

        # -- Speicherort --
        self._save_path_auto = True
        save_group = QGroupBox("Speicherort (Zwischenspeicherung + Endergebnis)")
        save_layout = QHBoxLayout()
        self.save_path_edit = QLineEdit()
        self.save_path_edit.textEdited.connect(self._on_save_path_edited)
        browse_btn = QPushButton("Durchsuchen...")
        browse_btn.clicked.connect(self._on_browse_save_path)
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(browse_btn)
        save_group.setLayout(save_layout)
        main_layout.addWidget(save_group)
        save_info = QLabel(
            "Der Scan wird unter diesem Pfad stuendlich zwischengespeichert. Bricht der "
            "Prozess ab, setzt ein erneuter Start mit denselben Parametern und demselben "
            "Pfad automatisch an der Abbruchstelle fort. Am Ende wird der Zwischenstand "
            "durch das fertige Ergebnis ersetzt (keine Doppel-Datei)."
        )
        save_info.setWordWrap(True)
        save_info.setStyleSheet("font-style: italic; color: gray;")
        main_layout.addWidget(save_info)

        self.auto_report = QCheckBox("Nach dem Scan direkt auswerten (Plots + Bericht)")
        self.auto_report.setChecked(True)
        self.auto_report.setToolTip(
            "Erzeugt sofort dieselben Plots und denselben Markdown-Bericht wie\n"
            "run_plots.py. Kann spaeter jederzeit mit run_plots.py wiederholt werden."
        )
        main_layout.addWidget(self.auto_report)

        self.n_points.valueChanged.connect(self._update_default_save_path)
        self._update_default_save_path()

        # Buttons bewusst AUSSERHALB der ScrollArea -> immer sichtbar.
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Penalty-Scan starten")
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        outer_layout.addLayout(btn_layout)

        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry().height() if screen is not None else 900
        self.resize(660, min(760, int(avail * 0.85)))

    def _sync_airy_mode(self):
        """Bei einer benannten Konvention steht der Wert fest - das Feld
        zeigt ihn dann nur an. Nur 'frei eingeben' laesst ihn tippen."""
        _key, _label, wert = combine.AIRY_SCALE_CHOICES[
            self.airy_scale_mode.currentIndex()]
        if wert is None:
            self.airy_scale_factor.setEnabled(True)
        else:
            self.airy_scale_factor.setEnabled(False)
            self.airy_scale_factor.setValue(float(wert))
        self._sync_airy_hint()

    def _sync_airy_hint(self):
        """Was der eingestellte Faktor konkret bedeutet - damit man den Wert
        nicht gegen eine Tabelle im README pruefen muss."""
        f = self.airy_scale_factor.value()
        e2 = combine.airy_e2_radius_factor(f)
        self.airy_scale_hint.setText(
            f"Bei Faktor {f:.4f}: erste Nullstelle bei {f:.4f} x waist, "
            f"tatsaechlicher 1/e^2-Radius bei {e2:.4f} x waist. "
            f"Zwei Spots beruehren sich mit ihren Hauptkeulen bei einem "
            f"Abstand von {2 * f:.4f} x waist (= k im verbotenen Bereich).")

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
        return paths.penalty_pkl_name(N_X_FIXED, N_Y_FIXED, n, n, "airy")

    def _update_default_save_path(self):
        if self._save_path_auto:
            self.save_path_edit.setText(str(paths.DEFAULT_RESULTS_DIR / self._default_save_filename()))

    def _on_save_path_edited(self, _text):
        self._save_path_auto = False

    def _on_browse_save_path(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Speicherort waehlen", self.save_path_edit.text(), "Pickle-Dateien (*.pkl)")
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
            QMessageBox.warning(self, "Ungueltiger Bereich",
                                "win_input min muss kleiner als win_input max sein.")
            return
        if self.width_min.value() >= self.width_max.value():
            QMessageBox.warning(self, "Ungueltiger Bereich",
                                "width min muss kleiner als width max sein.")
            return
        if self.r_min.value() >= self.r_max.value():
            QMessageBox.warning(self, "Ungueltiger Bereich",
                                "r_x/r_y min muss kleiner als r_x/r_y max sein.")
            return
        if not self.save_path_edit.text().strip():
            QMessageBox.warning(self, "Kein Speicherort",
                                "Bitte einen Speicherort fuer Zwischen-/Endergebnis angeben.")
            return
        self.accept()

    def _get_values_raw(self):
        return dict(
            win_input_range=(self.win_input_min.value() * 1e-3, self.win_input_max.value() * 1e-3),
            width_range=(self.width_min.value() * 1e6, self.width_max.value() * 1e6),
            n_points=self.n_points.value(),
            r_bounds=(self.r_min.value(), self.r_max.value()),
            alpha=self.alpha.value(),
            combo_lambda=self.combo_lambda.value(),
            combo_percentile=self.combo_percentile.value(),
            n_grid=self.n_grid.value(),
            airy_scale_factor=self.airy_scale_factor.value(),
            atom_temperature=self.atom_temperature_uK.value() * 1e-6,
            trap_freq_r=self.trap_freq_r_kHz.value() * 1e3,
            weighted_n_grid=self.weighted_n_grid.value(),
            n_jobs=self.n_jobs.value(),
            force_cpu=self.force_cpu.isChecked(),
            enable_perf_log=self.enable_perf_log.isChecked(),
            save_path=self.save_path_edit.text().strip(),
            auto_report=self.auto_report.isChecked(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = PenaltyScanDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)
    params = dialog.get_values()

    # ---------------- GPU / Parallelisierung ----------------
    n_jobs, pool_initializer, meldung = penalty_scan.setup_gpu(
        force_cpu=params["force_cpu"], n_jobs=params["n_jobs"])
    print(meldung)

    if params["enable_perf_log"]:
        penalty_scan.enable_perf_log()

    opt = penalty_scan.make_optimizer(
        f1=75e-3, f2=750e-3, N_x=N_X_FIXED, N_y=N_Y_FIXED,
        n_grid=params["n_grid"],
        airy_scale_factor=params["airy_scale_factor"],
        atom_temperature=params["atom_temperature"],
        trap_freq_r=params["trap_freq_r"],
        weighted_n_grid=params["weighted_n_grid"],
    )

    save_path = params["save_path"]

    # ---------------- Zwischenstand vorab pruefen ----------------
    if save_path:
        resumable = penalty_scan.peek_checkpoint(
            save_path, params["win_input_range"], params["width_range"],
            params["n_points"], N_X_FIXED, N_Y_FIXED,
            alpha=params["alpha"], r_bounds=params["r_bounds"],
            combo_lambda=params["combo_lambda"],
            airy_scale_factor=params["airy_scale_factor"],
            optics_match=dict(n_grid=params["n_grid"],
                              weighted_n_grid=params["weighted_n_grid"]))
        if resumable is not None:
            n_done, total_pts = resumable
            QMessageBox.information(
                None, "Zwischenstand gefunden",
                f"Zu diesem Penalty-Scan liegt bereits ein Zwischenstand vor:\n\n"
                f"{n_done}/{total_pts} Punkte vorhanden, wird fortgesetzt.")
        elif FilePath(save_path).exists():
            QMessageBox.information(
                None, "Datei vorhanden",
                "Unter diesem Pfad liegt bereits eine Datei, die nicht zu den aktuellen "
                "Scan-Parametern passt - der Scan startet komplett neu und ersetzt sie.")

    # ---------------- Scan ----------------
    total_points = params["n_points"] ** 2
    progress = QProgressDialog("Penalty-Scan laeuft (gemeinsame Amplituden-Optimierung)...",
                               "Abbrechen", 0, total_points)
    progress.setWindowTitle("Penalty-Scan")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    def on_progress(done, total):
        progress.setValue(done)
        QApplication.processEvents()
        return not progress.wasCanceled()

    t_start = time.perf_counter()

    results = penalty_scan.run_penalty_scan(
        opt,
        win_input_range=params["win_input_range"],
        width_range=params["width_range"],
        n_win_input=params["n_points"], n_width=params["n_points"],
        alpha=params["alpha"], combo_lambda=params["combo_lambda"],
        combo_percentile=params["combo_percentile"],
        r_bounds=params["r_bounds"],
        n_jobs=n_jobs, pool_initializer=pool_initializer,
        progress_callback=on_progress,
        checkpoint_path=save_path or None,
    )

    progress.setValue(total_points)
    penalty_scan.log_duration("Scan-Dauer", time.perf_counter() - t_start)

    # ---------------- Speichern ----------------
    saved_path = None
    if save_path:
        saved_path = combine.save_results(results, save_path, overwrite=True)

    # ---------------- Auswertung ----------------
    report_info = None
    if params["auto_report"]:
        try:
            report_info = report.make_all(results, ask_before_save=False, show=False)
        except Exception as exc:  # Auswertung darf den fertigen Scan nie gefaehrden
            QMessageBox.warning(None, "Auswertung fehlgeschlagen",
                                f"Der Scan ist gespeichert, die automatische Auswertung ist "
                                f"aber fehlgeschlagen:\n\n{exc!r}\n\n"
                                f"Sie laesst sich jederzeit mit run_plots.py nachholen.")

    lines = ["Penalty-Scan abgeschlossen."]
    if saved_path:
        lines.append(f"\nDatensatz: {saved_path}")
    best = results.get('best') or {}
    if best.get('win_input') is not None:
        lines.append(f"\nBester Punkt: win_input = {best['win_input'] * 1e3:.4f} mm, "
                     f"width = {best['width'] * 1e-6:.4f} MHz")
    if report_info and report_info.get('report'):
        lines.append(f"\nBericht: {report_info['report']}")
        lines.append(f"Plots: {paths.FIT_PLOTS_DIR}")
    QMessageBox.information(None, "Fertig", "\n".join(lines))


if __name__ == "__main__":
    main()
