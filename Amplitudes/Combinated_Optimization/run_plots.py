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

  {Praefix}_metric_comparison.pdf   hart vs. atom-gewichtet, 2x2 (auf Wunsch
                                    mit eingezeichneter Talpfad-Geraden)
  {Praefix}_region.pdf              Score-Karte mit Region und bestem Punkt
  {Praefix}_agreement.pdf           nur Hard-Check: Uebereinstimmungs-Karte
  {Praefix}_score_scatter.pdf       nur Hard-Check: gewichtet vs. hart
  {Praefix}_valley_{X}_over_{Y}.pdf Querschnitt entlang des Minimums von X,
                                    aufgetragen ueber Y (Waist oder width),
                                    auf Wunsch mit Gerade durch den Talpfad
  {Praefix}_line_{X}_over_{Y}.pdf   derselbe Querschnitt, aber entlang der
                                    Geraden statt entlang des Minimums
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
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout,
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
        self.draw_best_point = QCheckBox("Besten Punkt als Stern einzeichnen")
        self.draw_best_point.setChecked(True)
        self.draw_best_point.setToolTip(
            "Markiert in den Karten den Gitterpunkt mit dem kleinsten Score als roten\n"
            "Stern. Seine Zahlen stehen ohnehin im Bericht - ohne Haken bleiben die\n"
            "Heatmaps voellig frei.\n\n"
            "Das groesste Rechteck (\"Region\") wird nicht mehr eingezeichnet; seine\n"
            "Grenzen stehen weiterhin im Bericht.")
        display_layout.addRow(self.draw_best_point)
        self.fit_line_on_maps = QCheckBox(
            "Gerade auch in den Metrik-Vergleich einzeichnen (2x2-Karten)")
        self.fit_line_on_maps.setToolTip(
            "Zeichnet die Gerade, die unten durch den Talpfad gelegt wird, zusaetzlich\n"
            "in alle vier Karten von ..._metric_comparison.pdf - durchgezogen im\n"
            "gefitteten Bereich, gepunktet in der Extrapolation.\n\n"
            "Welche Groesse gefittet wird, bestimmt \"Groesse fuer Talpfad/Gerade\"\n"
            "in der Talschnitt-Gruppe. Die Gerade ist immer die ueber dem effektiven\n"
            "Waist in µm - auf einer mm-Achse erscheint sie deshalb leicht gekruemmt,\n"
            "weil win_input und effektiver Waist nichtlinear zusammenhaengen.\n\n"
            "Ohne brauchbare Gerade bleiben die Karten unveraendert (Hinweis auf der\n"
            "Konsole).")
        display_layout.addRow(self.fit_line_on_maps)
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

        # -- Querschnitt entlang des Minimums (Talschnitt) --
        valley_group = QGroupBox("Querschnitt entlang des Minimums (Talschnitt)")
        valley_layout = QVBoxLayout()
        valley_info = QLabel(
            "Einer Groesse folgen und pro Spalte (bzw. Zeile) den Punkt suchen, an dem sie\n"
            "minimal ist. GENAU an diesen Punkten werden dann die angehakten Groessen\n"
            "abgelesen - also nicht deren eigenes Minimum, sondern ihr Wert dort, wo die\n"
            "Fuehrungsgroesse am besten ist. Jede Kurve bekommt eine eigene y-Achse.\n"
            "Alternativ laeuft der Schnitt statt entlang des (springenden) Minimums\n"
            "entlang der Geraden, die durch den Talpfad gelegt wurde."
        )
        valley_info.setStyleSheet("font-style: italic;")
        valley_layout.addWidget(valley_info)

        self.do_valley = QCheckBox("Talschnitt erzeugen")
        self.do_valley.setChecked(True)
        self.do_valley.toggled.connect(self._on_valley_toggled)
        valley_layout.addWidget(self.do_valley)

        valley_form = QFormLayout()
        self.valley_path_mode = QComboBox()
        for _key, label in report.PATH_MODE_CHOICES:
            self.valley_path_mode.addItem(label)
        self.valley_path_mode.setToolTip(
            "Talpfad: pro Spalte (bzw. Zeile) der Punkt mit dem kleinsten Wert der\n"
            "Fuehrungsgroesse - echte Gitterwerte, aber der Pfad springt dort, wo das\n"
            "Minimum flach ist oder aus dem Scan-Fenster laeuft.\n\n"
            "Gerade: der Schnitt folgt der Geraden, die durch den Talpfad gefittet\n"
            "wurde - ueber den GANZEN gescannten Bereich, also auch weit ausserhalb\n"
            "der Punkte, aus denen sie bestimmt wurde (dort extrapoliert, im Plot mit\n"
            "offenen Kreisen markiert). Da die Gerade die Gitterpunkte nicht trifft,\n"
            "werden die Werte zwischen den beiden Nachbarzeilen linear interpoliert."
        )
        self.valley_path_mode.currentIndexChanged.connect(
            lambda _i: self._sync_valley_fit_state())
        valley_form.addRow("Schnitt entlang:", self.valley_path_mode)
        self.valley_follow = QComboBox()
        for _key, label in report.FOLLOW_CHOICES:
            self.valley_follow.addItem(label)
        self.valley_follow.setToolTip(
            "Welcher Groesse der Talpfad folgt - und damit auch, worauf die Gerade\n"
            "gefittet wird.\n\n"
            "Die beiden Penalty-Eintraege unterscheiden sich, und der Unterschied ist\n"
            "nicht klein:\n\n"
            "  NORMIERT (combined_score): jedes der vier Gitter wird vorher einzeln\n"
            "  min-max ueber das Scan-Fenster auf 0..1 gezogen. Das macht die vier\n"
            "  vergleichbar, hebt aber die atom-gewichteten Groessen gegenueber der\n"
            "  harten Uniformity an - deren rohe Spanne ist ein Vielfaches groesser.\n"
            "  Diese Groesse hat der Optimierer nie gesehen.\n\n"
            "  ROH (J): genau die Zielfunktion, die der Scan an jedem Gitterpunkt\n"
            "  ueber (r_x, r_y) minimiert hat. Keine Normierung, haengt damit auch\n"
            "  nicht am gescannten Fenster.\n\n"
            "Die Steigung der Geraden kann sich zwischen beiden deutlich\n"
            "unterscheiden. Welche gemeint war, steht im Bericht und im Dateinamen."
        )
        valley_form.addRow("Groesse fuer Talpfad/Gerade:", self.valley_follow)
        self.valley_axis = QComboBox()
        for _key, label in report.VALLEY_AXIS_CHOICES:
            self.valley_axis.addItem(label)
        self.valley_axis.setToolTip(
            "Bei Waist: pro Waist-Spalte wird ueber width minimiert.\n"
            "Bei width: pro width-Zeile wird ueber den Waist minimiert.\n\n"
            + report.valley_fit_axis_hint()
        )
        self.valley_axis.currentIndexChanged.connect(
            lambda _i: self._sync_valley_fit_state())
        valley_form.addRow("Aufgetragen ueber:", self.valley_axis)
        valley_layout.addLayout(valley_form)

        self.valley_fit_line = QCheckBox("Gerade durch den Talpfad legen (linearer Fit)")
        self.valley_fit_line.setChecked(True)
        self._fit_line_gemerkt = True
        self._fit_line_axis_tooltip = (
            report.valley_fit_axis_hint() + "\n"
            "Bei den anderen Achsen ist die Gerade deshalb gesperrt.")
        self._fit_line_mode_tooltip = (
            "Im Geradenmodus wird die Gerade immer bestimmt - sie ist ja der "
            "Schnitt selbst.")
        self._fit_line_tooltip = (
            "Legt eine Gerade durch den brauchbaren Teil des Talpfads und zeichnet sie\n"
            "in die Heatmap. Unbrauchbare Punkte werden automatisch ausgeschlossen:\n"
            "erst Minima am Rand des gescannten Fensters (das sind keine echten\n"
            "Minima), dann ein abgesetzter Nebenzweig, zuletzt abknickende Randpunkte.\n"
            "Die ausgeschlossenen Punkte werden im Plot markiert und im Bericht\n"
            "gezaehlt - dasselbe Verfahren wie in fit_waist_width_relation.py.")
        self.valley_fit_line.setToolTip(self._fit_line_tooltip)
        valley_layout.addWidget(self.valley_fit_line)


        traces_label = QLabel("Welche Groessen sollen entlang dieses Wegs gezeigt werden?")
        valley_layout.addWidget(traces_label)

        # Checkboxen zweispaltig, damit die Gruppe nicht zu hoch wird.
        self.trace_boxes = {}
        traces_grid = QGridLayout()
        for position, key in enumerate(report.TRACE_ORDER):
            klartext = {
                "uniformity_weighted": "Uniformity, atom-gewichtet",
                "crosstalk_weighted": "Crosstalk, atom-gewichtet",
                "uniformity_hard": "Uniformity, hart",
                "crosstalk_hard": "Crosstalk, hart",
                "combined": "combined score (Penalty)",
                "r_x": "r_x (Amplituden-Verhaeltnis x)",
                "r_y": "r_y (Amplituden-Verhaeltnis y)",
            }[key]
            box = QCheckBox(klartext)
            box.setChecked(True)
            self.trace_boxes[key] = box
            traces_grid.addWidget(box, position // 2, position % 2)
        valley_layout.addLayout(traces_grid)

        traces_hint = QLabel(
            "Je mehr Haken, desto mehr y-Achsen - mit allen sieben wird es voll. "
            "Die Fuehrungsgroesse wird immer mitgezeichnet, auch ohne Haken."
        )
        traces_hint.setWordWrap(True)
        traces_hint.setStyleSheet("color: gray;")
        valley_layout.addWidget(traces_hint)

        valley_group.setLayout(valley_layout)
        main_layout.addWidget(valley_group)

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

    def _on_valley_toggled(self, checked):
        for widget in [self.valley_path_mode, self.valley_follow, self.valley_axis,
                       *self.trace_boxes.values()]:
            widget.setEnabled(bool(checked))
        self._sync_valley_fit_state()

    def _sync_valley_fit_state(self):
        """Haelt Pfad-Dropdown und Fit-Haken im Einklang mit der gewaehlten
        Achse. Zwei Regeln:

        - Eine Gerade gibt es nur fuer die µm-Achse (report.VALLEY_FIT_AXIS).
          Bei den anderen Achsen ist der Haken gesperrt und leer, und
          "Gerade" laesst sich im Pfad-Dropdown gar nicht erst waehlen.
        - Im Geradenmodus IST die Gerade der Schnitt, der Haken ist dort
          gesetzt und gesperrt.

        Was der Nutzer zuletzt selbst eingestellt hat, wird gemerkt und
        wiederhergestellt, sobald der Haken wieder frei ist.
        """
        if self.valley_fit_line.isEnabled():
            self._fit_line_gemerkt = self.valley_fit_line.isChecked()

        moeglich = report.valley_fit_supported(self._current_axis())
        for index, (key, _label) in enumerate(report.PATH_MODE_CHOICES):
            item = self.valley_path_mode.model().item(index)
            if item is not None:
                item.setEnabled(moeglich or key != "line")
        if not moeglich and self._current_path_mode() == "line":
            self.valley_path_mode.setCurrentIndex(0)      # zurueck auf Talpfad

        aktiv = self.do_valley.isChecked()
        if not moeglich:
            self.valley_fit_line.setChecked(False)
            self.valley_fit_line.setEnabled(False)
            self.valley_fit_line.setToolTip(self._fit_line_axis_tooltip)
        elif self._current_path_mode() == "line":
            self.valley_fit_line.setChecked(True)
            self.valley_fit_line.setEnabled(False)
            self.valley_fit_line.setToolTip(self._fit_line_mode_tooltip)
        else:
            self.valley_fit_line.setChecked(bool(self._fit_line_gemerkt))
            self.valley_fit_line.setEnabled(aktiv)
            self.valley_fit_line.setToolTip(self._fit_line_tooltip)

    def _current_path_mode(self):
        return report.PATH_MODE_CHOICES[self.valley_path_mode.currentIndex()][0]

    def _current_axis(self):
        return report.VALLEY_AXIS_CHOICES[self.valley_axis.currentIndex()][0]

    def _sync_valley_options(self):
        """Nur die Groessen anbieten, die der geladene Datensatz hergibt."""
        if self.loaded is None:
            return
        moegliche_follow = report.available_follow_keys(self.loaded)
        for index, (key, _label) in enumerate(report.FOLLOW_CHOICES):
            item = self.valley_follow.model().item(index)
            if item is not None:
                item.setEnabled(key in moegliche_follow)
        if report.FOLLOW_CHOICES[self.valley_follow.currentIndex()][0] not in moegliche_follow:
            for index, (key, _label) in enumerate(report.FOLLOW_CHOICES):
                if key in moegliche_follow:
                    self.valley_follow.setCurrentIndex(index)
                    break
        moegliche_traces = report.available_trace_keys(self.loaded)
        aktiv = self.do_valley.isChecked()
        for key, box in self.trace_boxes.items():
            verfuegbar = key in moegliche_traces
            box.setEnabled(aktiv and verfuegbar)
            if not verfuegbar:
                box.setChecked(False)
                box.setToolTip("In diesem Datensatz nicht enthalten.")
        self._sync_valley_fit_state()

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
        self._sync_valley_options()

    def _on_accept(self):
        self._try_load()
        if self.loaded is None:
            QMessageBox.warning(
                self, "Kein gueltiger Datensatz",
                "Bitte oben einen Datensatz auswaehlen - einen Penalty-Scan "
                "(scan_amp_data_combined_*.pkl) oder einen Hard-Check "
                "(hard_check_*.pkl).")
            return
        if self.do_valley.isChecked() and self._current_path_mode() == "line":
            werte = self.get_values()
            if not report.valley_fit_supported(werte["valley_axis"]):
                QMessageBox.warning(self, "Gerade nur fuer die µm-Achse",
                                    report.valley_fit_axis_hint())
                return
            if report.fit_valley_line(self.loaded, axis=werte["valley_axis"],
                                      follow=werte["valley_follow"]) is None:
                QMessageBox.warning(
                    self, "Keine Gerade moeglich",
                    "Fuer diese Kombination aus Groesse und Achse laesst sich keine "
                    "Gerade durch den Talpfad legen: nach dem Ausschluss der "
                    "unbrauchbaren Talpunkte bleiben zu wenige uebrig.\n\n"
                    "Das heisst meist, dass das Minimum ueber weite Teile des Scans am "
                    "Rand des gescannten Fensters liegt.\n\n"
                    "Bitte eine andere Groesse waehlen oder auf \"Talpfad\" "
                    "umschalten.")
                return
        self.accept()

    def get_values(self):
        return dict(
            results=self.loaded,
            win_axis=WIN_AXIS_CHOICES[self.win_axis.currentIndex()][1],
            legend_fontsize=int(self.legend_fontsize.value()),
            draw_best_point=self.draw_best_point.isChecked(),
            fit_line_on_maps=self.fit_line_on_maps.isChecked(),
            plot_amplitudes=self.plot_amplitudes.isChecked(),
            show=self.show_interactive.isChecked(),
            do_recombine=self.do_recombine.isChecked(),
            alpha=self.alpha.value(),
            combo_lambda=self.combo_lambda.value(),
            combo_percentile=self.combo_percentile.value(),
            save_recombined=self.save_recombined.isChecked(),
            ask_before_save=self.ask_before_save.isChecked(),
            do_valley=self.do_valley.isChecked(),
            valley_follow=report.FOLLOW_CHOICES[self.valley_follow.currentIndex()][0],
            valley_axis=report.VALLEY_AXIS_CHOICES[self.valley_axis.currentIndex()][0],
            valley_traces=[key for key, box in self.trace_boxes.items() if box.isChecked()],
            valley_fit_line=self.valley_fit_line.isChecked(),
            valley_path_mode=self._current_path_mode(),
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
            fit_line_on_maps=params["fit_line_on_maps"],
            plot_amplitudes_overview=params["plot_amplitudes"],
            save=True, show=params["show"],
            ask_before_save=params["ask_before_save"],
            legend_fontsize=params["legend_fontsize"],
            valley_cut=params["do_valley"],
            valley_axis=params["valley_axis"],
            valley_follow=params["valley_follow"],
            valley_traces=params["valley_traces"],
            valley_fit_line=params["valley_fit_line"],
            valley_path_mode=params["valley_path_mode"],
        )
    except Exception as exc:
        QMessageBox.critical(None, "Auswertung fehlgeschlagen", f"{exc!r}")
        sys.exit(1)

    lines = [f"Auswertung fertig ({combine.KIND_LABELS.get(out['kind'], out['kind'])}).", ""]
    lines.append(f"Plots: {paths.FIT_PLOTS_DIR}")
    if params["do_valley"]:
        lines.append("")
        lines.append(f"Schnitt entlang: "
                     f"{report.path_mode_label(params['valley_path_mode'])}")
        if params["valley_fit_line"] or params["valley_path_mode"] == "line":
            fit = out.get('valley_line')
            if fit is None:
                lines.append("Talpfad-Gerade: zu wenige brauchbare Talpunkte - "
                             "siehe Bericht.")
            else:
                lines.append(f"Talpfad-Gerade: {report.valley_line_formula(fit)}")
                lines.append(f"   R² = {report._r2_text(fit['r2'])}, "
                             f"{fit['n_used']} von {fit['n_total']} Talpunkten verwendet")
        lines.append("")
    if out.get('report'):
        lines.append(f"Bericht: {out['report']}")
    if saved_recombined:
        lines.append(f"Neu berechnete Fassung: {saved_recombined}")
    QMessageBox.information(None, "Fertig", "\n".join(lines))


if __name__ == "__main__":
    main()
