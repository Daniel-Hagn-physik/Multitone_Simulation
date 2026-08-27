"""
run_plots.py  -  VORHANDENE DATENSAETZE PLOTTEN UND AUSWERTEN
=============================================================

    ==> Dieses Skript ausfuehren, wenn ein Datensatz bereits vorliegt
    ==> und nur Plots und Bericht (neu) erzeugt werden sollen.

Versteht beide Datensatz-Arten aus diesem Ordner und erkennt automatisch,
welche vorliegt:

  - Penalty-Scan   (scan_amp_data_combined_*.pkl, aus run_penalty_scan.py -
                    auch die bereits vorhandenen Datensaetze)
  - Hard-Check     (hard_check_*.pkl, aus run_hard_check.py)

Erzeugt in Fit_Plots/ bzw. Fit_Results/:

  {Praefix}_metric_comparison.pdf   hart vs. atom-gewichtet, 2x2
  {Praefix}_region.pdf              Score-Karte mit Region und bestem Punkt
  {Praefix}_agreement.pdf           nur Hard-Check: Uebereinstimmungs-Karte
  {Praefix}_score_scatter.pdf       nur Hard-Check: gewichtet vs. hart
  {Praefix}_Report.md               Bericht mit allen Kennzahlen
  (optional) die 6-Panel-Uebersicht und die Schnitte des AmplitudeScanPlotter

alpha, combo_lambda und das Perzentil koennen hier neu gesetzt werden -
Score, Region und (beim Hard-Check) die Vierfeldertafel werden dann aus
den vorhandenen Grids neu berechnet, OHNE den teuren Scan zu wiederholen.
Der Datensatz selbst wird dabei nicht veraendert; auf Wunsch laesst sich
die neu berechnete Fassung unter neuem Namen speichern.

Die anderen beiden Hauptskripte:
    run_penalty_scan.py  -  neuen Datensatz mit der Penalty-Methode scannen
    run_hard_check.py    -  vorhandenen GEWICHTETEN Scan hart nachrechnen
"""

import sys
from pathlib import Path as FilePath

from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QFileDialog, QCheckBox, QComboBox, QScrollArea, QWidget,
)

sys.path.insert(0, str(FilePath(__file__).resolve().parent))

from lib import paths  # noqa: E402
from lib import combine, hard_check, report  # noqa: E402


# ======================================================================
# HIER DEN DATENSATZ EINTRAGEN (optional)
# ======================================================================
# Leer lassen ("") -> im Dialog aus einer Liste aller Dateien in Results/
# auswaehlen. Es wird NICHTS automatisch vorausgewaehlt.
#
# Oder hier fest eintragen, dann ist dieser Datensatz beim Start bereits
# ausgewaehlt. Es genuegt der Dateiname (wird in Results/ gesucht), ein
# vollstaendiger Pfad geht auch:
#
#   PKL_DATEI = "scan_amp_data_combined_N3x4_21x21pts_Airy.pkl"
#   PKL_DATEI = r"C:\...\Results\hard_check_N3x4_10x10pts_Airy.pkl"
#
PKL_DATEI = ""
# ======================================================================


WIN_AXIS_CHOICES = [
    ("win_input, mm vor der Linse", "before_lens"),
    ("effektiver Waist, µm nach der Linse", "after_lens"),
]

NO_SELECTION = "\u2014 bitte auswaehlen \u2014"


class PlotsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auswertung - Plots und Bericht zu einem vorhandenen Datensatz")
        self.loaded = None

        outer_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        main_layout = QVBoxLayout(content)
        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)

        info = QLabel(
            "Datensatz waehlen - die Art (Penalty-Scan oder Hard-Check) wird automatisch\n"
            "erkannt und bestimmt, welche Plots und welcher Bericht erzeugt werden."
        )
        info.setStyleSheet("font-style: italic;")
        main_layout.addWidget(info)

        # -- Datensatz --
        input_group = QGroupBox("Datensatz")
        input_layout = QVBoxLayout()
        row = QHBoxLayout()
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(360)
        self.file_combo.currentIndexChanged.connect(lambda _i: self._try_load())
        refresh_btn = QPushButton("Aktualisieren")
        refresh_btn.setToolTip("Results/ erneut einlesen (z.B. nach einem gerade "
                               "fertig gewordenen Scan).")
        refresh_btn.clicked.connect(self._fill_file_combo)
        browse_btn = QPushButton("Andere Datei...")
        browse_btn.setToolTip("Eine Datei ausserhalb von Results/ waehlen.")
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

        # -- Darstellung --
        display_group = QGroupBox("Darstellung")
        display_layout = QFormLayout()
        self.win_axis = QComboBox()
        for label, _value in WIN_AXIS_CHOICES:
            self.win_axis.addItem(label)
        display_layout.addRow("Waist-Achse:", self.win_axis)
        self.legend_fontsize = self._make_spin(9.0, 4.0, 24.0, 1.0, decimals=0)
        display_layout.addRow("Schriftgroesse Legenden:", self.legend_fontsize)
        self.draw_best_point = QCheckBox("Region und besten Punkt einzeichnen")
        self.draw_best_point.setChecked(True)
        display_layout.addRow(self.draw_best_point)
        self.plot_amplitudes = QCheckBox("Amplituden-Uebersicht und Schnitte mitzeichnen")
        self.plot_amplitudes.setChecked(True)
        self.plot_amplitudes.setToolTip(
            "Die 6-Panel-Uebersicht (Uniformity/Crosstalk je hart und gewichtet, plus\n"
            "r_x/r_y) und die waist-/width-Schnitte des AmplitudeScanPlotter - als PNG."
        )
        display_layout.addRow(self.plot_amplitudes)
        self.show_interactive = QCheckBox("Plots zusaetzlich interaktiv anzeigen")
        display_layout.addRow(self.show_interactive)
        display_group.setLayout(display_layout)
        main_layout.addWidget(display_group)

        # -- Neuberechnung --
        recombine_group = QGroupBox("Score und Region neu berechnen (optional)")
        recombine_layout = QFormLayout()
        recombine_info = QLabel(
            "Ohne Haken werden die im Datensatz gespeicherten Werte verwendet.\n"
            "Mit Haken werden Score, Region und (beim Hard-Check) die Vierfeldertafel\n"
            "aus den vorhandenen Grids neu berechnet - kein erneuter Scan noetig."
        )
        recombine_info.setStyleSheet("font-style: italic;")
        recombine_layout.addRow(recombine_info)
        self.do_recombine = QCheckBox("Mit abweichenden Parametern neu berechnen")
        self.do_recombine.toggled.connect(self._on_recombine_toggled)
        recombine_layout.addRow(self.do_recombine)
        self.alpha = self._make_spin(0.7, 0.0, 1.0, 0.05)
        recombine_layout.addRow("alpha:", self.alpha)
        self.combo_lambda = self._make_spin(0.75, 0.0, 5.0, 0.05)
        recombine_layout.addRow("combo_lambda (Penalty-Gewicht):", self.combo_lambda)
        self.combo_percentile = self._make_spin(25.0, 1.0, 100.0, 5.0)
        recombine_layout.addRow("Perzentil (% beste Punkte):", self.combo_percentile)
        self.save_recombined = QCheckBox("Neu berechnete Fassung als eigene Datei speichern")
        self.save_recombined.setToolTip(
            "Legt eine neue .pkl neben dem Original an (Suffix _recombined).\n"
            "Der urspruengliche Datensatz bleibt unveraendert."
        )
        recombine_layout.addRow(self.save_recombined)
        recombine_group.setLayout(recombine_layout)
        main_layout.addWidget(recombine_group)
        self._on_recombine_toggled(False)

        self.ask_before_save = QCheckBox("Vor dem Ueberschreiben vorhandener Plots nachfragen")
        self.ask_before_save.setChecked(False)
        self.ask_before_save.setToolTip(
            "Ohne Haken werden gleichnamige Plots des heutigen Tages ohne Rueckfrage\n"
            "ueberschrieben (sie sind aus dem Datensatz jederzeit reproduzierbar).\n"
            "Achtung: die Rueckfrage laeuft ueber die Konsole, nicht ueber ein Fenster."
        )
        main_layout.addWidget(self.ask_before_save)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Auswertung erzeugen")
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
    def _make_spin(value, minimum, maximum, step, decimals=3):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def _on_recombine_toggled(self, checked):
        for widget in (self.alpha, self.combo_lambda, self.combo_percentile,
                       self.save_recombined):
            widget.setEnabled(bool(checked))

    def _fill_file_combo(self):
        """Listet alle .pkl-Dateien aus Results/ auf, neueste zuerst.

        Vorausgewaehlt wird NUR, was oben in PKL_DATEI eingetragen ist -
        sonst bleibt "bitte auswaehlen" stehen, damit nie versehentlich
        der falsche Datensatz ausgewertet wird.
        """
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        self.file_combo.addItem(NO_SELECTION, userData=None)

        dateien = sorted(paths.DEFAULT_RESULTS_DIR.glob("*.pkl"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for pfad in dateien:
            self.file_combo.addItem(pfad.name, userData=str(pfad))
        self.file_combo.blockSignals(False)

        if not dateien:
            self.input_info.setText(
                f"In {paths.DEFAULT_RESULTS_DIR} liegen keine .pkl-Dateien. "
                f"Zuerst run_penalty_scan.py oder run_hard_check.py ausfuehren - "
                f"oder ueber \"Andere Datei...\" einen Datensatz von anderswo waehlen.")
            return

        # Vorauswahl nur, wenn oben im Skript etwas eingetragen ist.
        if PKL_DATEI:
            self._select_path(PKL_DATEI)
        else:
            self.input_info.setText(
                f"{len(dateien)} Datei(en) in Results/ gefunden - bitte oben auswaehlen. "
                f"(Wer immer denselben Datensatz auswertet, kann ihn oben im Skript "
                f"bei PKL_DATEI eintragen.)")

    def _select_path(self, pfad):
        """Waehlt den angegebenen Pfad im Dropdown aus - haengt ihn an,
        falls er nicht aus Results/ stammt."""
        kandidat = FilePath(pfad)
        if not kandidat.is_absolute() and not kandidat.exists():
            kandidat = paths.DEFAULT_RESULTS_DIR / kandidat.name
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
                f"Die in PKL_DATEI eingetragene Datei wurde nicht gefunden:\n{pfad}\n"
                f"Bitte oben aus der Liste auswaehlen.")

    def _current_path(self):
        return self.file_combo.currentData()

    def _on_browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Datensatz waehlen", str(paths.DEFAULT_RESULTS_DIR), "Pickle-Dateien (*.pkl)")
        if path:
            self._select_path(path)

    def _try_load(self):
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
        kind = combine.dataset_kind(results)
        if kind not in ("penalty", "hard_check"):
            self.input_info.setText(
                "Diese Datei passt nicht zu diesem Ordner. Erwartet wird ein Penalty-Scan "
                "(scan_amp_data_combined_*.pkl) oder ein Hard-Check (hard_check_*.pkl).\n"
                "Ein rein GEWICHTETER Scan gehoert zuerst durch run_hard_check.py.")
            return
        self.loaded = results
        self.input_info.setText(combine.describe(results))
        if results.get('alpha') is not None:
            self.alpha.setValue(float(results['alpha']))
        if results.get('combo_lambda') is not None:
            self.combo_lambda.setValue(float(results['combo_lambda']))
        if results.get('combo_percentile') is not None:
            self.combo_percentile.setValue(float(results['combo_percentile']))

    def _on_accept(self):
        self._try_load()
        if self.loaded is None:
            QMessageBox.warning(
                self, "Kein gueltiger Datensatz",
                "Bitte oben einen Datensatz auswaehlen - einen Penalty-Scan "
                "(scan_amp_data_combined_*.pkl) oder einen Hard-Check "
                "(hard_check_*.pkl).")
            return
        self.accept()

    def get_values(self):
        return dict(
            results=self.loaded,
            win_axis=WIN_AXIS_CHOICES[self.win_axis.currentIndex()][1],
            legend_fontsize=int(self.legend_fontsize.value()),
            draw_best_point=self.draw_best_point.isChecked(),
            plot_amplitudes=self.plot_amplitudes.isChecked(),
            show=self.show_interactive.isChecked(),
            do_recombine=self.do_recombine.isChecked(),
            alpha=self.alpha.value(),
            combo_lambda=self.combo_lambda.value(),
            combo_percentile=self.combo_percentile.value(),
            save_recombined=self.save_recombined.isChecked(),
            ask_before_save=self.ask_before_save.isChecked(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = PlotsDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)
    params = dialog.get_values()

    results = params["results"]
    source_path = results.get('_source_path')
    kind = combine.dataset_kind(results)

    saved_recombined = None
    if params["do_recombine"]:
        if kind == "hard_check":
            results = hard_check.recheck_from_grids(
                results, alpha=params["alpha"], combo_lambda=params["combo_lambda"],
                good_percentile=params["combo_percentile"])
        else:
            results = combine.recombine_from_grids(
                results, alpha=params["alpha"], combo_lambda=params["combo_lambda"],
                combo_percentile=params["combo_percentile"])
        if params["save_recombined"] and source_path:
            src = FilePath(source_path)
            target = src.with_name(f"{src.stem}_recombined{src.suffix}")
            saved_recombined = combine.save_results(results, target, overwrite=False)

    try:
        out = report.make_all(
            results,
            win_axis=params["win_axis"],
            draw_best_point=params["draw_best_point"],
            plot_amplitudes_overview=params["plot_amplitudes"],
            save=True, show=params["show"],
            ask_before_save=params["ask_before_save"],
            legend_fontsize=params["legend_fontsize"],
        )
    except Exception as exc:
        QMessageBox.critical(None, "Auswertung fehlgeschlagen", f"{exc!r}")
        sys.exit(1)

    lines = [f"Auswertung fertig ({combine.KIND_LABELS.get(out['kind'], out['kind'])}).", ""]
    lines.append(f"Plots: {paths.FIT_PLOTS_DIR}")
    if out.get('report'):
        lines.append(f"Bericht: {out['report']}")
    if saved_recombined:
        lines.append(f"Neu berechnete Fassung: {saved_recombined}")
    QMessageBox.information(None, "Fertig", "\n".join(lines))


if __name__ == "__main__":
    main()
