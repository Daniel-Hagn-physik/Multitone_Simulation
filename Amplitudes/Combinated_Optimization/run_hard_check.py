"""
run_hard_check.py  -  PASST DER HARD CASE ZU MEINEM WEIGHTED-DATENSATZ?
=======================================================================

    ==> Dieses Skript ausfuehren, wenn es bereits einen GEWICHTETEN
    ==> Amplituden-Scan gibt (scan_amp_data_weighted_*.pkl) und geprueft
    ==> werden soll, ob die dort gefundenen guten Punkte auch unter dem
    ==> HARTEN Kriterium gut sind.

Was passiert: an jedem Gitterpunkt des uebergebenen gewichteten Scans
werden win_input, width und die dort GEFUNDENEN Amplituden r_x/r_y
genommen und damit GENAU EINMAL die harten Metriken ausgewertet.

    KEINE erneute Optimierung. KEINE Kombination zweier Datensaetze.
    Nur: dieselben Amplituden, einmal hart nachgerechnet.

Deshalb ist der Lauf billig (eine Auswertung statt einer ganzen
Optimierung pro Punkt) - typischerweise Sekunden bis wenige Minuten, und
daher ohne Zwischenspeicherung.

Ausgewertet wird zweifach:
  - Vierfeldertafel: wie viele der unter dem gewichteten Ziel guten
    Punkte sind auch unter dem harten Ziel gut? (Kernkennzahl) Dazu
    Pearson-Korrelationen von Score, Uniformity und Crosstalk.
  - Consistency-Score/Region: dieselbe Penalty-Kombination wie beim
    Penalty-Scan, hier als raeumlich zusammenhaengende Karte der
    Uebereinstimmung.

Ergebnis: Results/hard_check_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl
(eigenes Namensmuster - nie mit den Penalty-Datensaetzen zu verwechseln).
Die uebergebene gewichtete Datei wird NICHT veraendert.

Die anderen beiden Hauptskripte:
    run_penalty_scan.py  -  neuen Datensatz mit der Penalty-Methode scannen
    run_plots.py         -  vorhandene Datensaetze plotten/auswerten
"""

import os
import sys
import time
from pathlib import Path as FilePath

from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QFileDialog, QCheckBox, QLineEdit, QComboBox, QScrollArea, QWidget,
)
from PyQt5.QtCore import Qt

sys.path.insert(0, str(FilePath(__file__).resolve().parent))

# Alles, was dieses Skript braucht, kommt aus lib - auch der Optimierer, der
# eigentlich in ../Weighted_Optimization liegt (lib/hard_check.py kapselt ihn,
# siehe Hinweis in lib/paths.py).
from lib import paths, combine, hard_check, report  # noqa: E402


# ======================================================================
# HIER DIE GEWICHTETE EINGANGSDATEI EINTRAGEN (optional)
# ======================================================================
# Leer lassen ("") -> im Dialog aus einer Liste aller gewichteten Scans
# auswaehlen. Es wird NICHTS automatisch vorausgewaehlt.
#
# Oder hier fest eintragen, dann ist diese Datei beim Start bereits
# ausgewaehlt. Dateiname genuegt, vollstaendiger Pfad geht auch:
#
#   WEIGHTED_PKL = "scan_amp_data_weighted_N3x4_10x10pts_Airy.pkl"
#
WEIGHTED_PKL = ""
# ======================================================================

# Durchsucht werden ../Weighted_Optimization/Results (dort entstehen die
# gewichteten Scans) und das eigene Results/.
WEIGHTED_GLOB = "scan_amp_data_weighted_*.pkl"
NO_SELECTION = "\u2014 bitte auswaehlen \u2014"


class HardCheckDialog(QDialog):
    """Waehlt die gewichtete Eingangsdatei und die Parameter der
    Nachrechnung. Zeigt sofort an, was geladen wurde."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hard-Check - harte Metriken zu einem gewichteten Scan nachrechnen")
        self.loaded = None
        self._save_path_auto = True

        outer_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        main_layout = QVBoxLayout(content)
        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)

        info = QLabel(
            "An jedem Gitterpunkt des gewaehlten GEWICHTETEN Scans werden win_input, width\n"
            "und die dort gefundenen Amplituden r_x/r_y genommen und damit einmal die HARTEN\n"
            "Metriken ausgewertet - ohne erneute Optimierung. Die Eingangsdatei bleibt\n"
            "unveraendert."
        )
        info.setStyleSheet("font-style: italic;")
        main_layout.addWidget(info)

        # -- Eingangsdatei --
        input_group = QGroupBox("Gewichteter Amplituden-Scan (Eingabe)")
        input_layout = QVBoxLayout()
        row = QHBoxLayout()
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(360)
        self.file_combo.currentIndexChanged.connect(lambda _i: self._try_load())
        refresh_btn = QPushButton("Aktualisieren")
        refresh_btn.setToolTip("Die Results-Ordner erneut einlesen.")
        refresh_btn.clicked.connect(self._fill_file_combo)
        browse_btn = QPushButton("Andere Datei...")
        browse_btn.setToolTip("Eine Datei ausserhalb der Results-Ordner waehlen.")
        browse_btn.clicked.connect(self._on_browse_input)
        row.addWidget(self.file_combo, stretch=1)
        row.addWidget(refresh_btn)
        row.addWidget(browse_btn)
        input_layout.addLayout(row)
        self.input_info = QLabel("Noch kein Datensatz gewaehlt.")
        self.input_info.setWordWrap(True)
        self.input_info.setStyleSheet("color: gray;")
        input_layout.addWidget(self.input_info)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # -- Harte Nachrechnung --
        recompute_group = QGroupBox("Harte Nachrechnung")
        recompute_layout = QFormLayout()
        self.n_grid = QSpinBox()
        self.n_grid.setRange(50, 5000)
        self.n_grid.setValue(1000)
        self.n_grid.setToolTip(
            "Aufloesung des globalen Intensitaetsgitters fuer die HARTE Auswertung.\n"
            "Der gewichtete Scan speichert seinen eigenen Wert nicht mit - hier also\n"
            "denselben Wert eintragen, mit dem der gewichtete Scan gelaufen ist,\n"
            "damit die beiden Seiten vergleichbar bleiben."
        )
        recompute_layout.addRow("n_grid:", self.n_grid)
        cpu_count = os.cpu_count() or 1
        self.n_jobs = QSpinBox()
        self.n_jobs.setRange(1, max(1, cpu_count))
        self.n_jobs.setValue(max(1, cpu_count - 1))
        recompute_layout.addRow(f"n_jobs (1-{cpu_count}):", self.n_jobs)
        recompute_group.setLayout(recompute_layout)
        main_layout.addWidget(recompute_group)

        # -- Vergleich --
        compare_group = QGroupBox("Vergleich")
        compare_layout = QFormLayout()
        compare_info = QLabel(
            "alpha: Gewicht von Uniformity gegenueber Crosstalk in BEIDEN Scores.\n"
            "good_percentile: was als \"gut\" zaehlt - die besten X% nach dem jeweiligen\n"
            "Score, fuer gewichtet und hart unabhaengig bestimmt.\n"
            "combo_lambda: Penalty-Gewicht des Consistency-Scores (Region-Karte)."
        )
        compare_info.setStyleSheet("font-style: italic;")
        compare_layout.addRow(compare_info)
        self.alpha = self._make_spin(0.7, 0.0, 1.0, 0.05)
        compare_layout.addRow("alpha:", self.alpha)
        self.good_percentile = self._make_spin(25.0, 1.0, 100.0, 5.0)
        compare_layout.addRow("good_percentile (% beste Punkte):", self.good_percentile)
        self.combo_lambda = self._make_spin(0.75, 0.0, 5.0, 0.05)
        compare_layout.addRow("combo_lambda (Penalty-Gewicht):", self.combo_lambda)
        compare_group.setLayout(compare_layout)
        main_layout.addWidget(compare_group)

        # -- Speicherort --
        save_group = QGroupBox("Speicherort des Ergebnisses")
        save_layout = QHBoxLayout()
        self.save_path_edit = QLineEdit()
        self.save_path_edit.textEdited.connect(self._on_save_path_edited)
        save_browse = QPushButton("Durchsuchen...")
        save_browse.clicked.connect(self._on_browse_save)
        save_layout.addWidget(self.save_path_edit)
        save_layout.addWidget(save_browse)
        save_group.setLayout(save_layout)
        main_layout.addWidget(save_group)

        self.auto_report = QCheckBox("Nach der Nachrechnung direkt auswerten (Plots + Bericht)")
        self.auto_report.setChecked(True)
        main_layout.addWidget(self.auto_report)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Hard-Check starten")
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        outer_layout.addLayout(btn_layout)

        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry().height() if screen is not None else 900
        self.resize(720, min(700, int(avail * 0.85)))

        # Erst jetzt fuellen: das currentIndexChanged-Signal loest _try_load()
        # aus, das auf die Felder der uebrigen Gruppen zugreift.
        self._fill_file_combo()

    @staticmethod
    def _make_spin(value, minimum, maximum, step):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(3)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    # ------------------------------------------------------------------
    def _fill_file_combo(self):
        """Listet die gewichteten Amplituden-Scans auf, neueste zuerst.

        Vorausgewaehlt wird NUR, was oben in WEIGHTED_PKL eingetragen ist -
        sonst bleibt "bitte auswaehlen" stehen.
        """
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        self.file_combo.addItem(NO_SELECTION, userData=None)

        gefunden = []
        for ordner in (paths.WEIGHTED_DIR / "Results", paths.DEFAULT_RESULTS_DIR):
            if ordner.is_dir():
                gefunden.extend(ordner.glob(WEIGHTED_GLOB))
        gefunden = sorted(set(gefunden), key=lambda p: p.stat().st_mtime, reverse=True)

        for pfad in gefunden:
            # Ordner mit anzeigen - der Name allein waere zwischen den beiden
            # Results-Ordnern nicht eindeutig.
            self.file_combo.addItem(f"{pfad.parent.parent.name}/Results/{pfad.name}",
                                    userData=str(pfad))
        self.file_combo.blockSignals(False)

        if not gefunden:
            self.input_info.setText(
                f"Keine Datei nach dem Muster {WEIGHTED_GLOB} gefunden (gesucht in "
                f"{paths.WEIGHTED_DIR / 'Results'} und {paths.DEFAULT_RESULTS_DIR}). "
                f"Ueber \"Andere Datei...\" laesst sich eine von anderswo waehlen.")
            return

        if WEIGHTED_PKL:
            self._select_path(WEIGHTED_PKL)
        else:
            self.input_info.setText(
                f"{len(gefunden)} gewichtete(r) Scan(s) gefunden - bitte oben auswaehlen. "
                f"(Wer immer denselben nimmt, kann ihn oben im Skript bei WEIGHTED_PKL "
                f"eintragen.)")

    def _select_path(self, pfad):
        """Waehlt den angegebenen Pfad aus - haengt ihn an, falls er nicht
        in einem der durchsuchten Ordner liegt."""
        kandidat = FilePath(pfad)
        if not kandidat.is_absolute() and not kandidat.exists():
            for ordner in (paths.WEIGHTED_DIR / "Results", paths.DEFAULT_RESULTS_DIR):
                if (ordner / kandidat.name).exists():
                    kandidat = ordner / kandidat.name
                    break
        ziel = str(kandidat)

        for index in range(self.file_combo.count()):
            if self.file_combo.itemData(index) == ziel:
                self.file_combo.setCurrentIndex(index)
                return
        if kandidat.exists():
            self.file_combo.addItem(kandidat.name, userData=ziel)
            self.file_combo.setCurrentIndex(self.file_combo.count() - 1)
        else:
            self.input_info.setText(
                f"Die in WEIGHTED_PKL eingetragene Datei wurde nicht gefunden:\n{pfad}\n"
                f"Bitte oben aus der Liste auswaehlen.")

    def _current_path(self):
        return self.file_combo.currentData()

    def _on_browse_input(self):
        start_dir = str(paths.WEIGHTED_DIR / "Results")
        path, _ = QFileDialog.getOpenFileName(
            self, "Gewichteten Amplituden-Scan waehlen", start_dir, "Pickle-Dateien (*.pkl)")
        if path:
            self._select_path(path)

    def _try_load(self):
        """Laedt die Datei sofort und zeigt an, was drinsteht."""
        path = self._current_path()
        self.loaded = None
        if not path:
            self.input_info.setText("Noch kein Datensatz gewaehlt.")
            return
        try:
            results = combine.load_results(path)
        except Exception as exc:
            self.input_info.setText(f"Datei konnte nicht geladen werden: {exc!r}")
            return
        ok, missing = combine.looks_like_weighted_amp_scan(results)
        if not ok:
            self.input_info.setText(
                f"Diese Datei sieht NICHT wie ein amplituden-optimierter gewichteter Scan aus "
                f"(es fehlen: {', '.join(missing)}). Erwartet wird scan_amp_data_weighted_*.pkl.")
            return

        # Ein Penalty-Scan oder ein fertiger Hard-Check enthaelt zufaellig
        # dieselben Schluessel und wuerde die Pruefung oben bestehen - ihn hier
        # nachzurechnen waere aber sinnlos: die Amplituden stammen dort aus der
        # gemeinsamen Optimierung, und die harten Metriken stehen bereits drin.
        kind = combine.dataset_kind(results)
        if kind in ("penalty", "hard_check"):
            self.input_info.setText(
                f"Das ist bereits ein {combine.KIND_LABELS[kind]}.\n"
                f"Hier wird ein REIN GEWICHTETER Amplituden-Scan gebraucht "
                f"(scan_amp_data_weighted_*.pkl) - einer, in dem die harten Metriken noch "
                f"fehlen. Diesen Datensatz plottet man stattdessen mit run_plots.py.")
            return
        self.loaded = results
        self.input_info.setText(combine.describe(results))
        if results.get('alpha') is not None:
            self.alpha.setValue(float(results['alpha']))
        self._update_default_save_path()

    def _default_save_filename(self):
        r = self.loaded or {}
        n_win = len(r.get('win_input_vals', []))
        n_width = len(r.get('width_vals', []))
        return paths.hardcheck_pkl_name(r.get('N_x', 3), r.get('N_y', 4),
                                        n_win, n_width, r.get('profile', 'airy'))

    def _update_default_save_path(self):
        if self._save_path_auto and self.loaded is not None:
            self.save_path_edit.setText(
                str(paths.DEFAULT_RESULTS_DIR / self._default_save_filename()))

    def _on_save_path_edited(self, _text):
        self._save_path_auto = False

    def _on_browse_save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Speicherort waehlen", self.save_path_edit.text(), "Pickle-Dateien (*.pkl)")
        if path:
            self.save_path_edit.setText(path)
            self._save_path_auto = False

    def _on_accept(self):
        self._try_load()
        if self.loaded is None:
            QMessageBox.warning(self, "Keine gueltige Eingangsdatei",
                                "Bitte oben einen amplituden-optimierten GEWICHTETEN Scan "
                                "(scan_amp_data_weighted_*.pkl) auswaehlen.")
            return
        if not self.save_path_edit.text().strip():
            QMessageBox.warning(self, "Kein Speicherort",
                                "Bitte einen Speicherort fuer das Ergebnis angeben.")
            return
        self.accept()

    def get_values(self):
        return dict(
            weighted_results=self.loaded,
            n_grid=self.n_grid.value(),
            n_jobs=self.n_jobs.value(),
            alpha=self.alpha.value(),
            good_percentile=self.good_percentile.value(),
            combo_lambda=self.combo_lambda.value(),
            save_path=self.save_path_edit.text().strip(),
            auto_report=self.auto_report.isChecked(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = HardCheckDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)
    params = dialog.get_values()

    weighted = params["weighted_results"]
    total_points = (len(weighted['win_input_vals']) * len(weighted['width_vals']))

    progress = QProgressDialog("Harte Metriken werden bei den vorhandenen Amplituden "
                               "nachgerechnet...", "Abbrechen", 0, total_points)
    progress.setWindowTitle("Hard-Check")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    def on_progress(done, total):
        progress.setValue(done)
        QApplication.processEvents()
        return not progress.wasCanceled()

    t_start = time.perf_counter()

    try:
        results = hard_check.run_hard_check(
            weighted, n_grid=params["n_grid"], n_jobs=params["n_jobs"],
            alpha=params["alpha"], combo_lambda=params["combo_lambda"],
            good_percentile=params["good_percentile"],
            progress_callback=on_progress, verbose=True,
        )
    except ValueError as exc:
        progress.close()
        QMessageBox.critical(None, "Hard-Check nicht moeglich", str(exc))
        sys.exit(1)

    progress.setValue(total_points)
    print(f"Hard-Check-Dauer: {time.perf_counter() - t_start:.1f}s")

    saved_path = combine.save_results(results, params["save_path"], overwrite=True)

    report_info = None
    if params["auto_report"]:
        try:
            report_info = report.make_all(results, ask_before_save=False, show=False)
        except Exception as exc:
            QMessageBox.warning(None, "Auswertung fehlgeschlagen",
                                f"Der Hard-Check ist gespeichert, die automatische Auswertung "
                                f"ist aber fehlgeschlagen:\n\n{exc!r}\n\n"
                                f"Sie laesst sich mit run_plots.py nachholen.")

    c = results.get('consistency') or {}
    frac = c.get('fraction_weighted_good_also_hard_good')
    lines = ["Hard-Check abgeschlossen.", ""]
    if frac is not None:
        lines.append(f"{frac * 100:.1f}% der unter dem gewichteten Ziel guten Punkte sind "
                     f"auch unter dem harten Ziel gut ({c.get('n_both_good')}/"
                     f"{c.get('n_weighted_good')}).")
    if c.get('pearson_score') is not None:
        lines.append(f"Pearson r (Score, gewichtet vs. hart) = {c['pearson_score']:.4f}")
    region = results.get('region') or {}
    if region.get('win_input_min') is not None:
        lines.append(f"\nValidierte Region: win_input {region['win_input_min'] * 1e3:.4f} .. "
                     f"{region['win_input_max'] * 1e3:.4f} mm, width "
                     f"{region['width_min'] * 1e-6:.4f} .. {region['width_max'] * 1e-6:.4f} MHz")
    lines.append(f"\nDatensatz: {saved_path}")
    if report_info and report_info.get('report'):
        lines.append(f"Bericht: {report_info['report']}")
        lines.append(f"Plots: {paths.FIT_PLOTS_DIR}")
    QMessageBox.information(None, "Fertig", "\n".join(lines))


if __name__ == "__main__":
    main()
