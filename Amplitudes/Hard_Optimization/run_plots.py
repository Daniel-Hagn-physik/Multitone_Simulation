"""
run_plots.py  -  VORHANDENE DATENSAETZE PLOTTEN UND AUSWERTEN
=============================================================

    ==> Dieses Skript ausfuehren, wenn ein Datensatz bereits vorliegt
    ==> und nur Plots und Bericht (neu) erzeugt werden sollen.

Versteht beide Datensatz-Arten dieses Ordners und erkennt automatisch,
welche vorliegt:

  - Amplituden-Scan       (scan_amp_data*.pkl, aus run_amp_scan.py):
                          pro Gitterpunkt eine eigene (r_x, r_y)-Optimierung
  - Fest-Amplituden-Scan  (scan_data*.pkl, aus run_scan.py):
                          feste Amplituden, nur win_input und width variiert

Erzeugt in Fit_Plots/ bzw. Fit_Results/:

  {Praefix}_metric_comparison.pdf     Uniformity und Crosstalk nebeneinander
                                      (auf Wunsch mit der Talpfad-Geraden)
  {Praefix}_metric_comparison_amp.pdf dieselben plus r_x und r_y (2x2)
  {Praefix}_region.pdf                Score-Karte mit Arbeitspunkt
  {Praefix}_valley_{X}_over_{Y}.pdf   Querschnitt entlang des Minimums von X,
                                      aufgetragen ueber Y (Waist oder width),
                                      auf Wunsch mit Gerade durch den Talpfad
  {Praefix}_line_{X}_over_{Y}.pdf     derselbe Querschnitt, aber entlang der
                                      Geraden statt entlang des Minimums
  {Praefix}_Report.md                 Bericht mit allen Kennzahlen
  (optional) die PNG-Uebersicht des jeweiligen Scan-Plotters

alpha und das Perzentil koennen hier neu gesetzt werden - Score, bester
Punkt und Region werden dann aus den vorhandenen Grids neu berechnet, OHNE
den teuren Scan zu wiederholen. Der Datensatz auf der Platte wird dabei nicht
veraendert; auf Wunsch laesst sich die neu berechnete Fassung unter neuem
Namen speichern.

Die anderen Haupt-Skripte dieses Ordners:
    run_scan.py      -  neuer Scan bei FESTEN Amplituden
    run_amp_scan.py  -  neuer Scan MIT Amplituden-Optimierung je Gitterpunkt
"""

import math
import sys
from pathlib import Path as FilePath

from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QFileDialog, QCheckBox, QComboBox, QScrollArea, QWidget,
)

sys.path.insert(0, str(FilePath(__file__).resolve().parent))

from lib import paths  # noqa: E402
from lib import report, scan_data  # noqa: E402


# ======================================================================
# HIER DEN DATENSATZ EINTRAGEN (optional)
# ======================================================================
# Leer lassen ("") -> im Dialog aus einer Liste aller Dateien in Results/
# auswaehlen. Es wird NICHTS automatisch vorausgewaehlt.
#
# Oder hier fest eintragen, dann ist dieser Datensatz beim Start bereits
# ausgewaehlt. Es genuegt der Dateiname (wird in Results/ gesucht), ein
# vollstaendiger Pfad geht auch.
PKL_DATEI = ""
# ======================================================================


WIN_AXIS_CHOICES = [
    ("win_input, mm vor der Linse", "before_lens"),
    ("effektiver Waist, µm nach der Linse", "after_lens"),
]

NO_SELECTION = "— bitte auswaehlen —"

# Klartext fuer die Kurven-Checkboxen des Talschnitts. Der Zusatz haengt an
# der Metrik-Familie dieses Ordners (paths.FLAVOR) - dadurch bleibt diese
# Datei in Hard_Optimization und Weighted_Optimization buchstabengleich.
_FAM = " (atom-gewichtet)" if paths.FLAVOR == "weighted" else " (hart, globale Maske)"
TRACE_LABELS = {
    "uniformity": "Uniformity" + _FAM,
    "crosstalk": "Crosstalk" + _FAM,
    "score": "J = alpha*Uniformity + (1-alpha)*Crosstalk (Score)",
    "r_x": "r_x (Amplituden-Verhaeltnis x)",
    "r_y": "r_y (Amplituden-Verhaeltnis y)",
}


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
            "Datensatz waehlen - die Art (Amplituden-Scan oder Fest-Amplituden-Scan)\n"
            "wird automatisch erkannt und bestimmt, welche Plots erzeugt werden."
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

        self.draw_best_point = QCheckBox("Punkt als Stern einzeichnen")
        self.draw_best_point.setChecked(True)
        self.draw_best_point.toggled.connect(self._on_draw_best_point_toggled)
        self.draw_best_point.setToolTip(
            "Markiert einen Punkt in den Karten als roten Stern. WELCHEN, bestimmt\n"
            "das Dropdown darunter - der beste Punkt nach einer Groesse, oder ein\n"
            "selbst vorgegebener. Ohne Haken bleiben die Heatmaps voellig frei.\n\n"
            "Das groesste Rechteck (\"Region\") wird nicht eingezeichnet; seine\n"
            "Grenzen stehen im Bericht.")
        display_layout.addRow(self.draw_best_point)

        self.best_point_follow = QComboBox()
        self.best_point_follow.setToolTip(
            "Nach welcher Groesse der Stern gesetzt wird.\n\n"
            "\"bester Gitterpunkt nach dem Score\" ist der Punkt, den auch der\n"
            "Bericht unter \"Bester Einzelpunkt\" nennt - Stern und Bericht zeigen\n"
            "dann ohne Zutun dasselbe.\n\n"
            "Waehlt man etwas anderes, bekommt der Bericht einen eigenen Abschnitt\n"
            "dazu. Punkte am Rand des Scan-Fensters werden als OFFENER Stern\n"
            "gezeichnet - sie sind kein Optimum, sondern nur das Ende des Scans.\n\n"
            "Die drei letzten Eintraege sind etwas anderes: dort gibst DU den Punkt\n"
            "vor. Bei den beiden \"nur ...\"-Varianten genuegt eine Koordinate, die\n"
            "zweite kommt aus der Talpfad-Geraden - die gibt es aber nur, wenn sich\n"
            "fuer den Datensatz ueberhaupt eine legen laesst. Sonst sind sie\n"
            "ausgegraut und es bleibt \"Waist UND Width vorgeben\".")
        self.best_point_follow.currentIndexChanged.connect(
            lambda _i: self._sync_manual_point())
        display_layout.addRow("Punkt:", self.best_point_follow)

        self.best_point_value = self._make_spin(1.0, 0.0001, 1000.0, 0.01, decimals=4)
        self.best_point_value.setToolTip(
            "Der selbst vorgegebene Wert - Waist in µm oder width in MHz, je nach\n"
            "Auswahl darueber.\n\n"
            "Liegt der Punkt ausserhalb des gescannten Fensters, wird er trotzdem\n"
            "gezeichnet - der Bericht sagt dann, dass es dort keine Daten gibt.")
        self.best_point_value_label = QLabel("Vorgabe:")
        display_layout.addRow(self.best_point_value_label, self.best_point_value)

        # Zweites Feld: nur fuer "Waist UND Width vorgeben". Das ist der Weg,
        # wenn sich keine Talpfad-Gerade legen laesst - dann waere die zweite
        # Koordinate sonst gar nicht bestimmbar.
        self.best_point_value2 = self._make_spin(0.25, 0.0001, 1000.0, 0.005, decimals=4)
        self.best_point_value2.setToolTip(
            "Die width des selbst vorgegebenen Punktes, in MHz.\n\n"
            "Dieses Feld gibt es nur bei \"Waist UND Width vorgeben\" - dort wird\n"
            "keine Gerade gebraucht, der Punkt steht einfach da, wo Du ihn hinsetzt.")
        self.best_point_value2_label = QLabel("Vorgabe width (MHz):")
        display_layout.addRow(self.best_point_value2_label, self.best_point_value2)

        self.point_cuts = QCheckBox(
            "Querschnitt durch den markierten Punkt (r_x, r_y) als PDF")
        self.point_cuts.setToolTip(
            "Zwei Schnitte durch den Stern, nebeneinander in einer PDF\n"
            "(..._point_cuts.pdf): links r_x und r_y bei FESTER width entlang des\n"
            "Waists, rechts bei FESTEM Waist entlang der width. Der Punkt selbst\n"
            "ist in beiden Panels als senkrechte rote Linie markiert.\n\n"
            "Beantwortet eine andere Frage als der Talschnitt weiter unten: nicht\n"
            "\"wie laeuft das Minimum?\", sondern \"wie empfindlich sind die\n"
            "Amplituden an meinem Arbeitspunkt?\"\n\n"
            "Gezeigt werden nur r_x und r_y, im gewohnten Aussehen der alten\n"
            "Amplituden-Schnitte: r_x blau mit Kreisen, r_y orange mit Quadraten,\n"
            "beide auf einer Achse. Uniformity und Crosstalk haben ihren Platz in\n"
            "den Karten und im Talschnitt.\n\n"
            "Gelesen wird auf dem Gitter. Liegt der Stern zwischen den Gitterpunkten\n"
            "(selbst vorgegebener Punkt), laufen die Schnitte durch die naechste\n"
            "Zeile bzw. Spalte - das steht dann im Titel. Interpolieren waere hier\n"
            "irrefuehrend: r_x/r_y sind Optimierungs-Ergebnisse, keine glatten\n"
            "Funktionen.\n\n"
            "Gibt es nur beim Amplituden-Scan.")
        display_layout.addRow(self.point_cuts)

        self.fit_line_on_maps = QCheckBox(
            "Gerade auch in den Metrik-Vergleich einzeichnen")
        self.fit_line_on_maps.setToolTip(
            "Zeichnet die Gerade, die unten durch den Talpfad gelegt wird, zusaetzlich\n"
            "in die Karten von ..._metric_comparison.pdf - durchgezogen im gefitteten\n"
            "Bereich, gepunktet in der Extrapolation.\n\n"
            "Welche Groesse gefittet wird, bestimmt \"Groesse fuer Talpfad/Gerade\"\n"
            "in der Talschnitt-Gruppe. Die Gerade ist immer die ueber dem effektiven\n"
            "Waist in µm - auf einer mm-Achse erscheint sie deshalb leicht gekruemmt,\n"
            "weil win_input und effektiver Waist nichtlinear zusammenhaengen.\n\n"
            "Gezeichnet wird eine durchgezogene Linie ueber den ganzen gescannten\n"
            "Bereich - aus welchem Bereich sie bestimmt wurde, steht im Bericht.\n\n"
            "Ohne brauchbare Gerade bleiben die Karten unveraendert (Hinweis auf der\n"
            "Konsole).")
        self.fit_line_on_maps.toggled.connect(self._sync_fit_line_style)
        display_layout.addRow(self.fit_line_on_maps)

        self.fit_line_dashed = QCheckBox(
            "   ... ausserhalb des Fit-Bereichs gepunktet statt durchgezogen")
        self.fit_line_dashed.setToolTip(
            "Ohne Haken ist die Gerade in den Karten eine einzige durchgezogene\n"
            "Linie - eine Gerade ist eine Gerade, und aus welchem Bereich sie\n"
            "bestimmt wurde, steht im Bericht.\n\n"
            "Mit Haken wird sie zweiteilig gezeichnet: durchgezogen im gefitteten\n"
            "Bereich, gepunktet in der Verlaengerung. Der gepunktete Teil bekommt\n"
            "keinen eigenen Legendeneintrag - der Unterschied steckt allein im\n"
            "Linienformat.")
        display_layout.addRow(self.fit_line_dashed)

        self.amplitude_maps = QCheckBox(
            "Metrik-Vergleich zusaetzlich mit Amplituden (4 Karten, eigene PDF)")
        self.amplitude_maps.setToolTip(
            "Schreibt neben ..._metric_comparison.pdf eine zweite Datei\n"
            "..._metric_comparison_amp.pdf: dieselben zwei Metrik-Karten, plus r_x\n"
            "und r_y - also die Amplituden, bei denen die Metriken darueber\n"
            "ausgewertet wurden.\n\n"
            "r_x und r_y teilen sich eine logarithmische Farbskala (Verhaeltnisse,\n"
            "und die Verteilung hat einen langen Schwanz). Punkte, deren Amplitude\n"
            "auf einer r_bounds-Schranke klemmt, sind grau: dort steht kein freies\n"
            "Optimum. Ihr Anteil erscheint auf der Konsole.\n\n"
            "Gibt es nur beim Amplituden-Scan - ein Fest-Amplituden-Scan hat keine\n"
            "r_x/r_y-Gitter.")
        display_layout.addRow(self.amplitude_maps)

        self.plot_overview = QCheckBox("PNG-Uebersicht des Scan-Plotters mitzeichnen")
        self.plot_overview.setToolTip(
            "Beim Amplituden-Scan die 6-Panel-Uebersicht und die waist-/width-\n"
            "Schnitte des AmplitudeScanPlotter, beim Fest-Amplituden-Scan die\n"
            "beiden Heatmaps - als PNG, im gewohnten Aussehen der Scan-Skripte.")
        display_layout.addRow(self.plot_overview)

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
            "werden die Werte zwischen den beiden Nachbarzeilen linear interpoliert.")
        self.valley_path_mode.currentIndexChanged.connect(
            lambda _i: self._sync_valley_fit_state())
        valley_form.addRow("Schnitt entlang:", self.valley_path_mode)

        self.valley_follow = QComboBox()
        for _key, label in report.FOLLOW_CHOICES:
            self.valley_follow.addItem(label)
        self.valley_follow.setToolTip(
            "Welcher Groesse der Talpfad folgt - und damit auch, worauf die Gerade\n"
            "gefittet wird.\n\n"
            "J ist die Zielgroesse, die auch der Optimierer minimiert, und zugleich\n"
            "der Score von Region und Bestpunkt. Roh, ohne Normierung, haengt damit\n"
            "auch nicht am gescannten Fenster.\n\n"
            "Welche Groesse gemeint war, steht im Bericht und im Dateinamen.")
        valley_form.addRow("Groesse fuer Talpfad/Gerade:", self.valley_follow)

        self.valley_select = QComboBox()
        for _key, label in report.VALLEY_SELECT_CHOICES:
            self.valley_select.addItem(label)
        self.valley_select.setToolTip(
            "WIE der Talpunkt je Spalte gewaehlt wird - der Schalter mit dem\n"
            "groessten Einfluss auf die Steigung.\n\n"
            "  Globales Minimum: der kleinste Wert der Spalte. Einfach, aber\n"
            "  unbrauchbar, sobald das Minimum am Rand des Scan-Fensters oder an der\n"
            "  Grenze des verbotenen Bereichs klebt.\n\n"
            "  Lokales Minimum nahe einer Leitgeraden: pro Spalte das lokale\n"
            "  Minimum, das der Leitgeraden am naechsten liegt. Lokal heisst: beide\n"
            "  Nachbarn vorhanden und groesser - Punkte am Scan-Rand und Punkte, die\n"
            "  an den ausgeschlossenen verbotenen Bereich grenzen, fallen damit von\n"
            "  selbst heraus.\n\n"
            "Die Leitgerade waehlt nur AUS, sie verschiebt nichts; die Punkte sind\n"
            "echte lokale Minima. Welcher Zweig verfolgt wird, entscheidet aber die\n"
            "Leitgroesse - das steht so auch im Bericht.")
        self.valley_select.currentIndexChanged.connect(
            lambda _i: self._sync_guide_state())
        valley_form.addRow("Talpunkt-Auswahl:", self.valley_select)

        self.valley_guide_follow = QComboBox()
        for _key, label in report.FOLLOW_CHOICES:
            self.valley_guide_follow.addItem(label)
        vorgabe = [k for k, _l in report.FOLLOW_CHOICES].index(report.GUIDE_FOLLOW_DEFAULT)
        self.valley_guide_follow.setCurrentIndex(vorgabe)
        self.valley_guide_follow.setToolTip(
            "Welche Groesse die Leitgerade liefert. Voreingestellt ist die\n"
            "Uniformity: ihr Talpfad ist der glatteste, und er liefert die Steigung,\n"
            "an der sich der Fit orientieren soll.\n\n"
            "Die Leitgerade wird immer mit dem GLOBALEN Minimum bestimmt - sonst\n"
            "braeuchte sie selbst wieder eine Leitgerade.")
        valley_form.addRow("Leitgroesse:", self.valley_guide_follow)

        self.valley_guide_halfwidth = self._make_spin(
            report.GUIDE_HALFWIDTH_DEFAULT, 0.001, 1.0, 0.005, decimals=3)
        self.valley_guide_halfwidth.setToolTip(
            "Halbe Breite des Korridors um die Leitgerade, in MHz. Lokale Minima\n"
            "weiter weg werden nicht in Betracht gezogen.\n\n"
            "Bei feinen Gittern ist das Ergebnis ueber einen weiten Bereich\n"
            "unempfindlich gegen diesen Wert - das ist das eigentliche Argument fuer\n"
            "das Verfahren. Bei groben Gittern lohnt es, zwei Werte zu vergleichen.")
        valley_form.addRow("Korridor um die Leitgerade (+- MHz):",
                           self.valley_guide_halfwidth)

        self.valley_axis = QComboBox()
        for _key, label in report.VALLEY_AXIS_CHOICES:
            self.valley_axis.addItem(label)
        self.valley_axis.setToolTip(
            "Bei Waist: pro Waist-Spalte wird ueber width minimiert.\n"
            "Bei width: pro width-Zeile wird ueber den Waist minimiert.\n\n"
            + report.valley_fit_axis_hint())
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

        self.valley_map_show_path = QCheckBox(
            "Talpfad und ausgelassene/extrapolierte Punkte in die Karte zeichnen")
        self.valley_map_show_path.setToolTip(
            "Ohne Haken zeigt die KARTE (linkes Panel) nur den verbotenen Bereich\n"
            "und die Ausgleichsgerade. Die einzelnen Pfadpunkte werden dann gar\n"
            "nicht erst gezeichnet - also auch nicht in der Legende gefuehrt.\n"
            "Ihre Anzahl steht im Bericht.\n\n"
            "Mit Haken kommen der Talpfad, die nicht benutzten und die\n"
            "extrapolierten Punkte zurueck, jeweils mit ihrer Anzahl in der\n"
            "Legende.\n\n"
            "Ausnahme: kommt gar keine Gerade zustande, waere die Karte sonst\n"
            "leer - dann wird der Talpfad auch ohne Haken gezeichnet.")
        valley_layout.addWidget(self.valley_map_show_path)

        traces_label = QLabel("Welche Groessen sollen entlang dieses Wegs gezeigt werden?")
        valley_layout.addWidget(traces_label)

        # Checkboxen zweispaltig, damit die Gruppe nicht zu hoch wird.
        self.trace_boxes = {}
        traces_grid = QGridLayout()
        for position, key in enumerate(report.TRACE_ORDER):
            box = QCheckBox(TRACE_LABELS[key])
            box.setChecked(True)
            self.trace_boxes[key] = box
            traces_grid.addWidget(box, position // 2, position % 2)
        valley_layout.addLayout(traces_grid)

        traces_hint = QLabel(
            "Je mehr Haken, desto mehr y-Achsen. Die Fuehrungsgroesse wird immer "
            "mitgezeichnet, auch ohne Haken.")
        traces_hint.setWordWrap(True)
        traces_hint.setStyleSheet("color: gray;")
        valley_layout.addWidget(traces_hint)

        self.valley_limit = QCheckBox(
            "Suchbereich einschraenken (sagen, wo der Talpfad gesucht wird)")
        self.valley_limit.setToolTip(
            "Ohne Haken wird ueber das ganze Scan-Fenster gesucht.\n\n"
            "Mit Haken nur innerhalb der vier Grenzen darunter - gedacht fuer\n"
            "Datensaetze mit MEHREREN Talzweigen, wo man dem Fit sagen muss,\n"
            "welcher gemeint ist.\n\n"
            "Minima, die am Rand des eingestellten Bereichs liegen, zaehlen wie\n"
            "Minima am Rand des Scan-Fensters und fallen aus dem Fit heraus -\n"
            "sonst wuerde die Einschraenkung selbst kuenstliche Talpunkte erzeugen.\n\n"
            "Der Bereich gilt AUCH fuer die Leitgerade des gefuehrten Modus.")
        self.valley_limit.toggled.connect(lambda _b: self._sync_valley_limit())
        valley_layout.addWidget(self.valley_limit)

        limit_form = QFormLayout()
        self.waist_von = self._make_spin(0.0, 0.0, 1000.0, 0.05, decimals=4)
        self.waist_bis = self._make_spin(0.0, 0.0, 1000.0, 0.05, decimals=4)
        self.width_von = self._make_spin(0.0, 0.0, 1000.0, 0.01, decimals=4)
        self.width_bis = self._make_spin(0.0, 0.0, 1000.0, 0.01, decimals=4)
        for w in (self.waist_von, self.waist_bis):
            w.setToolTip("Grenzen in µm (effektiver Waist nach der Linse).")
        for w in (self.width_von, self.width_bis):
            w.setToolTip("Grenzen in MHz.")
        waist_zeile = QHBoxLayout()
        waist_zeile.addWidget(self.waist_von)
        waist_zeile.addWidget(QLabel("bis"))
        waist_zeile.addWidget(self.waist_bis)
        limit_form.addRow("Waist von (µm):", waist_zeile)
        width_zeile = QHBoxLayout()
        width_zeile.addWidget(self.width_von)
        width_zeile.addWidget(QLabel("bis"))
        width_zeile.addWidget(self.width_bis)
        limit_form.addRow("width von (MHz):", width_zeile)
        valley_layout.addLayout(limit_form)

        self.valley_limit_info = QLabel("")
        self.valley_limit_info.setWordWrap(True)
        self.valley_limit_info.setStyleSheet("color: gray;")
        valley_layout.addWidget(self.valley_limit_info)

        valley_group.setLayout(valley_layout)
        main_layout.addWidget(valley_group)

        # -- Verbotener Bereich --
        forbidden_group = QGroupBox("Verbotener Bereich (Ueberlappung der Eck-Spots)")
        forbidden_layout = QVBoxLayout()
        forbidden_info = QLabel(
            "Die beiden diagonal gegenueberliegenden Eck-Spots duerfen sich nicht\n"
            "ueberlappen. width ist die Gesamtspannweite des Tonarrays, raeumlich also\n"
            "eine Kantenlaenge S; der Eckabstand ist sqrt(2)*S. Die Bedingung\n"
            "sqrt(2)*S > k*waist ist in der (waist, width)-Ebene eine Ursprungsgerade;\n"
            "darunter liegt der verbotene Bereich (dicke Spots, eng beieinander)."
        )
        forbidden_info.setStyleSheet("font-style: italic;")
        forbidden_layout.addWidget(forbidden_info)

        self.forbidden_draw = QCheckBox("Verbotenen Bereich in die Karten einzeichnen")
        self.forbidden_draw.setToolTip(
            "Grenzgerade plus schraffierte Flaeche darunter, in allen Karten.\n"
            "Auf der mm-Achse ist die Grenze gekruemmt, weil win_input und\n"
            "effektiver Waist reziprok zusammenhaengen.\n\n"
            "Aendert keine einzige Zahl - dafuer ist der Haken darunter da.")
        self.forbidden_draw.toggled.connect(self._sync_forbidden_state)
        forbidden_layout.addWidget(self.forbidden_draw)

        self.forbidden_exclude = QCheckBox(
            "Punkte im verbotenen Bereich aus der Auswertung ausschliessen")
        self.forbidden_exclude.setToolTip(
            "Setzt alle Gitter im verbotenen Bereich auf NaN und rechnet Score,\n"
            "Region und Bestpunkt daraus neu. Talpfad und Geradenfit sehen diese\n"
            "Punkte dann nicht mehr.\n\n"
            "Der Score ist das rohe J und damit punktweise definiert - er aendert\n"
            "sich durch den Ausschluss nur im verbotenen Bereich, nicht anderswo.\n\n"
            "Der Original-Datensatz auf der Platte wird nicht angefasst.")
        self.forbidden_exclude.toggled.connect(self._sync_forbidden_state)
        forbidden_layout.addWidget(self.forbidden_exclude)

        forbidden_form = QFormLayout()
        self.forbidden_factor = self._make_spin(
            scan_data.FORBIDDEN_FACTOR_DEFAULT, 0.1, 20.0, 0.1, decimals=3)
        self.forbidden_factor.setToolTip(
            "k in der Bedingung Abstand > k * waist.\n\n"
            "k = 2: die gaussaequivalenten Radien (1/e^2) beruehren sich gerade.\n"
            "k = 2.38 = 2*1.19: dasselbe fuer die Airy-Hauptkeule bis zur ersten\n"
            "Nullstelle - das ist der Radius, den man optisch als \"den Spot\" sieht.")
        self.forbidden_factor.valueChanged.connect(self._sync_forbidden_info)
        forbidden_form.addRow("Faktor k (Abstand > k * waist):", self.forbidden_factor)
        forbidden_layout.addLayout(forbidden_form)

        self.forbidden_info = QLabel("")
        self.forbidden_info.setWordWrap(True)
        self.forbidden_info.setStyleSheet("color: gray;")
        forbidden_layout.addWidget(self.forbidden_info)

        forbidden_group.setLayout(forbidden_layout)
        main_layout.addWidget(forbidden_group)

        # -- Neuberechnung --
        recompute_group = QGroupBox("Score und Region neu berechnen (optional)")
        recompute_layout = QFormLayout()
        recompute_info = QLabel(
            "Ohne Haken gelten die im Datensatz gespeicherten Werte fuer alpha und das\n"
            "Perzentil. Mit Haken werden Score, bester Punkt und Region daraus neu\n"
            "berechnet - kein erneuter Scan noetig, der Datensatz bleibt unveraendert."
        )
        recompute_info.setStyleSheet("font-style: italic;")
        recompute_layout.addRow(recompute_info)
        self.do_recompute = QCheckBox("Mit abweichenden Parametern neu berechnen")
        self.do_recompute.toggled.connect(self._on_recompute_toggled)
        recompute_layout.addRow(self.do_recompute)
        self.alpha = self._make_spin(0.7, 0.0, 1.0, 0.05)
        self.alpha.setToolTip(
            "Gewicht der Uniformity im Score J = alpha*U + (1-alpha)*Crosstalk.\n"
            "alpha = 1 waere reine Uniformity, alpha = 0 reiner Crosstalk.")
        recompute_layout.addRow("alpha:", self.alpha)
        self.percentile = self._make_spin(scan_data.DEFAULT_PERCENTILE, 1.0, 100.0, 5.0)
        self.percentile.setToolTip(
            "Wieviel Prozent der besten Gitterpunkte als \"Region\" gelten.\n"
            "Daraus wird das groesste achsenparallele Rechteck bestimmt; seine\n"
            "Grenzen stehen im Bericht.")
        recompute_layout.addRow("Perzentil (% beste Punkte):", self.percentile)
        self.save_recomputed = QCheckBox("Neu berechnete Fassung als eigene Datei speichern")
        self.save_recomputed.setToolTip(
            "Legt eine neue .pkl neben dem Original an (Suffix _recomputed).\n"
            "Der urspruengliche Datensatz bleibt unveraendert.")
        recompute_layout.addRow(self.save_recomputed)
        recompute_group.setLayout(recompute_layout)
        main_layout.addWidget(recompute_group)

        self._on_recompute_toggled(False)
        self._sync_fit_line_style()
        self._sync_forbidden_state()
        self._sync_guide_state()
        self._fill_best_point_combo()
        self._sync_valley_limit()

        # Ob es eine Talpfad-Gerade gibt, haengt an genau diesen Einstellungen -
        # und davon haengt ab, ob die beiden Ein-Koordinaten-Vorgaben ueberhaupt
        # waehlbar sind. Deshalb bei jeder Aenderung nachziehen.
        for widget in (self.valley_follow, self.valley_select, self.valley_guide_follow):
            widget.currentIndexChanged.connect(lambda _i: self._sync_manual_point())
        self.valley_guide_halfwidth.valueChanged.connect(
            lambda _v: self._sync_manual_point())
        for widget in (self.waist_von, self.waist_bis, self.width_von, self.width_bis):
            widget.valueChanged.connect(lambda _v: self._sync_manual_point())
        self.valley_limit.toggled.connect(lambda _b: self._sync_manual_point())

        self.ask_before_save = QCheckBox("Vor dem Ueberschreiben vorhandener Plots nachfragen")
        self.ask_before_save.setChecked(False)
        self.ask_before_save.setToolTip(
            "Ohne Haken werden gleichnamige Plots des heutigen Tages ohne Rueckfrage\n"
            "ueberschrieben (sie sind aus dem Datensatz jederzeit reproduzierbar).\n"
            "Achtung: die Rueckfrage laeuft ueber die Konsole, nicht ueber ein Fenster.")
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

    # ------------------------------------------------------------------
    # Zustandspflege
    # ------------------------------------------------------------------
    def _on_valley_toggled(self, checked):
        for widget in [self.valley_path_mode, self.valley_follow, self.valley_axis,
                       self.valley_select, self.valley_guide_follow,
                       self.valley_guide_halfwidth, self.valley_limit,
                       self.valley_map_show_path,
                       *self.trace_boxes.values()]:
            widget.setEnabled(bool(checked))
        self._sync_valley_fit_state()
        self._sync_guide_state()
        self._sync_valley_limit()

    def _sync_valley_fit_state(self):
        """Haelt Pfad-Dropdown und Fit-Haken im Einklang mit der gewaehlten
        Achse. Zwei Regeln:

        - Eine Gerade gibt es nur fuer die µm-Achse (report.VALLEY_FIT_AXIS).
          Bei den anderen Achsen ist der Haken gesperrt und leer, und
          "Gerade" laesst sich im Pfad-Dropdown gar nicht erst waehlen.
        - Im Geradenmodus IST die Gerade der Schnitt, der Haken ist dort
          gesetzt und gesperrt.

        Was der Nutzer zuletzt selbst eingestellt hat, wird gemerkt und
        wiederhergestellt, sobald der Haken wieder frei ist."""
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

    def _sync_valley_limit(self):
        aktiv = self.do_valley.isChecked() and self.valley_limit.isChecked()
        for w in (self.waist_von, self.waist_bis, self.width_von, self.width_bis):
            w.setEnabled(aktiv)
        if self.loaded is None:
            self.valley_limit_info.setText("Noch kein Datensatz geladen.")
            return
        voll = self._dataset_ranges()
        if voll is None:
            self.valley_limit_info.setText("")
            return
        (w0, w1), (h0, h1) = voll
        self.valley_limit_info.setText(
            f"Gescannt: waist {w0:.4f} .. {w1:.4f} µm, width {h0:.4f} .. {h1:.4f} MHz."
            + ("" if aktiv else " (Suche laeuft ueber den ganzen Bereich)"))

    def _dataset_ranges(self):
        """(waist_min, waist_max), (width_min, width_max) des Datensatzes."""
        if self.loaded is None:
            return None
        import numpy as _np
        waist = report.waist_um_of(self.loaded)
        width = _np.asarray(self.loaded["width_vals"], dtype=float) * 1e-6
        return ((float(waist.min()), float(waist.max())),
                (float(width.min()), float(width.max())))

    def _fill_valley_limit(self):
        """Grenzen mit dem vollen Bereich des Datensatzes vorbelegen - aber
        nur, solange der Haken aus ist (sonst wuerde man dem Nutzer seine
        Eingabe ueberschreiben)."""
        voll = self._dataset_ranges()
        if voll is None or self.valley_limit.isChecked():
            self._sync_valley_limit()
            return
        (w0, w1), (h0, h1) = voll
        # nach aussen runden: die angezeigten 4 Stellen duerfen den
        # aeussersten Gitterpunkt nicht abschneiden
        self.waist_von.setValue(math.floor(w0 * 1e4) / 1e4)
        self.waist_bis.setValue(math.ceil(w1 * 1e4) / 1e4)
        self.width_von.setValue(math.floor(h0 * 1e4) / 1e4)
        self.width_bis.setValue(math.ceil(h1 * 1e4) / 1e4)
        self._sync_valley_limit()

    def _current_valley_ranges(self):
        """(waist_range, width_range) fuer report - None, wo nicht
        eingeschraenkt wird. Ein Bereich, der den ganzen Scan umfasst, wird
        als 'keine Einschraenkung' behandelt."""
        if not self.valley_limit.isChecked():
            return None, None
        voll = self._dataset_ranges()
        wr = (self.waist_von.value(), self.waist_bis.value())
        hr = (self.width_von.value(), self.width_bis.value())
        if voll is not None:
            (w0, w1), (h0, h1) = voll
            tw = 1e-4 * (w1 - w0)
            th = 1e-4 * (h1 - h0)
            if wr[0] <= w0 + tw and wr[1] >= w1 - tw:
                wr = None
            if hr[0] <= h0 + th and hr[1] >= h1 - th:
                hr = None
        return wr, hr

    def _sync_guide_state(self):
        """Leitgroesse und Korridor nur im gefuehrten Modus bedienbar."""
        aktiv = self.do_valley.isChecked() and self._current_select() == "guided"
        self.valley_guide_follow.setEnabled(aktiv)
        self.valley_guide_halfwidth.setEnabled(aktiv)

    def _current_select(self):
        return report.VALLEY_SELECT_CHOICES[self.valley_select.currentIndex()][0]

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
                box.setToolTip("In diesem Datensatz nicht enthalten "
                               "(Fest-Amplituden-Scans haben kein r_x/r_y).")
        # Amplituden-Karten gibt es nur beim Amplituden-Scan.
        hat_amps = scan_data.has_amplitudes(self.loaded)
        self.amplitude_maps.setEnabled(hat_amps)
        self.point_cuts.setEnabled(hat_amps)
        if not hat_amps:
            self.amplitude_maps.setChecked(False)
            self.point_cuts.setChecked(False)
            self.point_cuts.setToolTip(
                "Nur beim Amplituden-Scan - ein Fest-Amplituden-Scan hat keine "
                "r_x/r_y-Gitter, durch die man schneiden koennte.")
        self._sync_valley_fit_state()

    def _sync_forbidden_state(self, _checked=None):
        aktiv = self.forbidden_draw.isChecked() or self.forbidden_exclude.isChecked()
        self.forbidden_factor.setEnabled(aktiv)
        self._sync_forbidden_info()

    def _sync_forbidden_info(self, _wert=None):
        """Zeigt Steigung und Anzahl betroffener Punkte fuer den geladenen
        Datensatz - sonst muesste man den Lauf starten, um zu sehen, ob der
        gewaehlte Faktor ueberhaupt etwas abschneidet."""
        if self.loaded is None:
            self.forbidden_info.setText("Noch kein Datensatz geladen.")
            return
        faktor = self.forbidden_factor.value()
        grenze = scan_data.forbidden_boundary(self.loaded, faktor)
        if grenze is None:
            self.forbidden_info.setText(
                "Dieser Datensatz hat keine zwei Eck-Spots - es kann nichts ueberlappen.")
            return
        maske = scan_data.forbidden_mask(self.loaded, faktor)
        n, gesamt = int(maske.sum()), int(maske.size)
        text = (f"Grenze: width/MHz > {grenze['slope']:.5f} * waist/µm "
                f"({grenze['um_per_MHz']:.4f} µm/MHz). "
                f"Betroffen: {n} von {gesamt} Gitterpunkten ({100.0 * n / gesamt:.1f}%).")
        # Beim Airy-Profil ist "die Hauptkeulen beruehren sich" ein konkreter
        # k-Wert - und der haengt am Skalenfaktor des Datensatzes.
        if str(self.loaded.get('profile')).lower() == "airy":
            skala = self.loaded.get('airy_scale_factor')
            quelle = "im Datensatz" if skala is not None else "Default, nicht gespeichert"
            skala = 1.19 if skala is None else float(skala)
            text += (f"\nDieser Datensatz: airy_scale_factor = {skala:.4f} ({quelle}) - "
                     f"die Airy-Hauptkeulen beruehren sich bei k = {2 * skala:.4f}, "
                     f"die 1/e²-Radien bei k = {2 * 0.67433 * skala:.4f}.")
        self.forbidden_info.setText(text)

    def _sync_fit_line_style(self, _checked=None):
        """Der Punktier-Haken ergibt nur Sinn, wenn ueberhaupt eine Gerade in
        die Karten gezeichnet wird."""
        an = self.fit_line_on_maps.isChecked()
        self.fit_line_dashed.setEnabled(an)
        if not an:
            self.fit_line_dashed.setChecked(False)

    def _on_recompute_toggled(self, checked):
        for widget in (self.alpha, self.percentile, self.save_recomputed):
            widget.setEnabled(bool(checked))

    # ------------------------------------------------------------------
    # Datensatz-Auswahl
    # ------------------------------------------------------------------
    def _fill_file_combo(self):
        """Listet alle .pkl-Dateien aus Results/ auf, neueste zuerst.

        Vorausgewaehlt wird NUR, was oben in PKL_DATEI eingetragen ist -
        sonst bleibt "bitte auswaehlen" stehen, damit nie versehentlich der
        falsche Datensatz ausgewertet wird."""
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
                f"Zuerst run_scan.py oder run_amp_scan.py ausfuehren - oder ueber "
                f"\"Andere Datei...\" einen Datensatz von anderswo waehlen.")
            return

        if PKL_DATEI:
            self._select_path(PKL_DATEI)
        else:
            self.input_info.setText(
                f"{len(dateien)} Datei(en) in Results/ gefunden - bitte oben auswaehlen. "
                f"(Wer immer denselben Datensatz auswertet, kann ihn oben im Skript "
                f"bei PKL_DATEI eintragen.)")

    def _select_path(self, pfad):
        """Waehlt den angegebenen Pfad im Dropdown aus - haengt ihn an, falls
        er nicht aus Results/ stammt."""
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
            self, "Datensatz waehlen", str(paths.DEFAULT_RESULTS_DIR),
            "Pickle-Dateien (*.pkl)")
        if path:
            self._select_path(path)

    def _try_load(self):
        path = self._current_path()
        self.loaded = None
        if not path:
            self.input_info.setText("Noch kein Datensatz gewaehlt.")
            return
        try:
            results = scan_data.load_results(path)
        except Exception as exc:
            self.input_info.setText(f"Datei konnte nicht geladen werden: {exc!r}")
            return
        ok, fehlt = scan_data.looks_like_scan(results)
        if not ok:
            self.input_info.setText(
                "Diese Datei sieht nicht nach einem Scan dieses Projekts aus - es "
                "fehlen: " + ", ".join(fehlt) + ".")
            return
        self.loaded = results
        self.input_info.setText(scan_data.describe(results))
        if results.get('alpha') is not None:
            self.alpha.setValue(float(results['alpha']))
        if results.get('combo_percentile') is not None:
            self.percentile.setValue(float(results['combo_percentile']))
        self._sync_valley_options()
        self._sync_forbidden_state()
        self._fill_best_point_combo()
        self._fill_valley_limit()

    def _on_draw_best_point_toggled(self, an):
        self.best_point_follow.setEnabled(bool(an))
        self._sync_manual_point()

    def _fill_best_point_combo(self):
        """Eintraege haengen am geladenen Datensatz - erst danach steht fest,
        welche Groessen es ueberhaupt gibt."""
        vorher = self._current_best_point_follow()
        self.best_point_combo_keys = (
            [report.BEST_POINT_FOLLOW_STORED] if self.loaded is None
            else [k for k, _l in report.best_point_choices(self.loaded)])
        labels = (["bester Gitterpunkt nach dem Score"] if self.loaded is None
                  else [l for _k, l in report.best_point_choices(self.loaded)])
        self.best_point_follow.blockSignals(True)
        self.best_point_follow.clear()
        self.best_point_follow.addItems(labels)
        if vorher in self.best_point_combo_keys:
            self.best_point_follow.setCurrentIndex(self.best_point_combo_keys.index(vorher))
        self.best_point_follow.blockSignals(False)
        self.best_point_follow.setEnabled(self.draw_best_point.isChecked())
        self._sync_manual_point()

    def _line_available(self):
        """Laesst sich mit den aktuellen Einstellungen ueberhaupt eine
        Talpfad-Gerade legen? Ohne sie sind die beiden Eigenvorgaben, die
        nur EINE Koordinate brauchen, nicht bestimmbar."""
        if self.loaded is None:
            return False
        try:
            wr, hr = self._current_valley_ranges()
            fit = report.fit_valley_line(
                self.loaded, axis=report.VALLEY_FIT_AXIS,
                follow=report.FOLLOW_CHOICES[self.valley_follow.currentIndex()][0],
                select=self._current_select(),
                guide_follow=report.FOLLOW_CHOICES[
                    self.valley_guide_follow.currentIndex()][0],
                guide_halfwidth=self.valley_guide_halfwidth.value(),
                waist_range=wr, width_range=hr)
        except Exception:
            return False
        return fit is not None

    def _sync_manual_point(self):
        """Die Wertfelder gibt es nur fuer die Eigenvorgaben - welches, haengt
        an der gewaehlten Variante.

        Zusaetzlich: die beiden Varianten mit nur EINER Koordinate brauchen
        eine Talpfad-Gerade. Gibt es keine, werden sie ausgegraut und der
        Eintrag faellt auf "Waist UND Width vorgeben" zurueck - dort wird nach
        beiden Koordinaten gefragt."""
        gerade = self._line_available()
        for index, key in enumerate(getattr(self, "best_point_combo_keys", [])):
            item = self.best_point_follow.model().item(index)
            if item is None:
                continue
            erlaubt = gerade or key not in report.MANUAL_LINE_KEYS
            item.setEnabled(erlaubt)
            item.setToolTip("" if erlaubt else
                            "Fuer diesen Datensatz (und diese Talschnitt-Einstellungen) "
                            "laesst sich keine Gerade durch den Talpfad legen - ohne sie "
                            "ist die zweite Koordinate nicht bestimmbar. Bitte "
                            "\"Waist UND Width vorgeben\" nehmen.")

        key = self._current_best_point_follow()
        if key in report.MANUAL_LINE_KEYS and not gerade:
            # Nicht kommentarlos nichts zeichnen: auf die Variante umschalten,
            # die ohne Gerade auskommt.
            if report.BEST_POINT_MANUAL_BOTH in self.best_point_combo_keys:
                self.best_point_follow.blockSignals(True)
                self.best_point_follow.setCurrentIndex(
                    self.best_point_combo_keys.index(report.BEST_POINT_MANUAL_BOTH))
                self.best_point_follow.blockSignals(False)
                key = report.BEST_POINT_MANUAL_BOTH

        an = self.draw_best_point.isChecked()
        eins = an and key in (report.BEST_POINT_MANUAL_WAIST,
                              report.BEST_POINT_MANUAL_WIDTH,
                              report.BEST_POINT_MANUAL_BOTH)
        zwei = an and key == report.BEST_POINT_MANUAL_BOTH
        self.best_point_value.setEnabled(eins)
        self.best_point_value_label.setEnabled(eins)
        self.best_point_value2.setEnabled(zwei)
        self.best_point_value2_label.setEnabled(zwei)
        if key == report.BEST_POINT_MANUAL_WIDTH:
            self.best_point_value_label.setText("Vorgabe width (MHz):")
            self.best_point_value.setRange(0.0001, 100.0)
            self.best_point_value.setSingleStep(0.005)
        elif key in (report.BEST_POINT_MANUAL_WAIST, report.BEST_POINT_MANUAL_BOTH):
            self.best_point_value_label.setText("Vorgabe Waist (µm):")
            self.best_point_value.setRange(0.0001, 100.0)
            self.best_point_value.setSingleStep(0.01)
        else:
            self.best_point_value_label.setText("Vorgabe:")

    def _current_best_point_follow(self):
        keys = getattr(self, "best_point_combo_keys", None)
        if not keys:
            return report.BEST_POINT_FOLLOW_STORED
        i = self.best_point_follow.currentIndex()
        return keys[i] if 0 <= i < len(keys) else report.BEST_POINT_FOLLOW_STORED

    # ------------------------------------------------------------------
    def _on_accept(self):
        self._try_load()
        if self.loaded is None:
            QMessageBox.warning(
                self, "Kein gueltiger Datensatz",
                "Bitte oben einen Datensatz auswaehlen - einen Amplituden-Scan "
                f"({paths.AMP_PKL_GLOB}) oder einen Fest-Amplituden-Scan "
                f"({paths.FIXED_PKL_GLOB}).")
            return
        punkt = self._current_best_point_follow()
        if self.draw_best_point.isChecked() and punkt in report.MANUAL_LINE_KEYS \
                and not self._line_available():
            QMessageBox.warning(
                self, "Keine Gerade fuer den eigenen Punkt",
                "Fuer diesen Datensatz laesst sich mit den eingestellten "
                "Talschnitt-Optionen keine Gerade durch den Talpfad legen. Damit "
                "ist die zweite Koordinate nicht bestimmbar.\n\n"
                "Bitte oben \"eigener Punkt: Waist UND Width vorgeben\" waehlen "
                "und beide Werte eintragen.")
            return
        if self.draw_best_point.isChecked() and punkt == report.BEST_POINT_MANUAL_BOTH:
            if self.loaded is not None and report.win_input_for_waist_um(
                    self.loaded, self.best_point_value.value()) is None:
                QMessageBox.warning(
                    self, "Waist nicht umrechenbar",
                    "Zu dem eingetragenen Waist laesst sich kein Eingangs-Waist "
                    "bestimmen. Bitte einen anderen Wert eintragen.")
                return
        braucht_gerade = self.do_valley.isChecked() and (
            self._current_path_mode() == "line" or self.valley_fit_line.isChecked())
        if braucht_gerade:
            werte = self.get_values()
            if not report.valley_fit_supported(werte["valley_axis"]):
                QMessageBox.warning(self, "Gerade nur fuer die µm-Achse",
                                    report.valley_fit_axis_hint())
                return
            # Genau die Einstellungen pruefen, mit denen dann auch gerechnet
            # wird - sonst meldet der Dialog etwas anderes, als hinterher
            # herauskommt. Der Score haengt an alpha, also vorher anwenden.
            probe = scan_data.analyse(
                self.loaded,
                alpha=(werte["alpha"] if werte["do_recompute"] else None),
                percentile=(werte["percentile"] if werte["do_recompute"] else None))
            grund = report.valley_fit_diagnosis(
                probe, axis=werte["valley_axis"],
                follow=werte["valley_follow"], select=werte["valley_select"],
                guide_follow=werte["valley_guide_follow"],
                guide_halfwidth=werte["valley_guide_halfwidth"],
                waist_range=werte["valley_waist_range"],
                width_range=werte["valley_width_range"])
            if grund is not None:
                if self._current_path_mode() == "line":
                    # Im Geradenmodus IST die Gerade der Schnitt - ohne sie
                    # gibt es nichts zu zeichnen.
                    QMessageBox.warning(
                        self, "Keine Gerade moeglich",
                        grund + "\n\nIm Geradenmodus ist die Gerade der Schnitt "
                        "selbst - bitte etwas davon aendern oder auf \"Talpfad\" "
                        "umschalten.")
                    return
                antwort = QMessageBox.question(
                    self, "Keine Gerade moeglich",
                    grund + "\n\nDer Talschnitt selbst wird trotzdem gezeichnet, "
                    "nur ohne Gerade. Fortfahren?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if antwort != QMessageBox.Yes:
                    return
        self.accept()

    def get_values(self):
        return dict(
            results=self.loaded,
            win_axis=WIN_AXIS_CHOICES[self.win_axis.currentIndex()][1],
            legend_fontsize=int(self.legend_fontsize.value()),
            draw_best_point=self.draw_best_point.isChecked(),
            best_point_follow=self._current_best_point_follow(),
            best_point_value=self.best_point_value.value(),
            best_point_value2=self.best_point_value2.value(),
            fit_line_on_maps=self.fit_line_on_maps.isChecked(),
            fit_line_dashed_extrapolation=self.fit_line_dashed.isChecked(),
            amplitude_maps=self.amplitude_maps.isChecked(),
            point_cuts=self.point_cuts.isChecked(),
            plot_overview=self.plot_overview.isChecked(),
            show=self.show_interactive.isChecked(),
            do_recompute=self.do_recompute.isChecked(),
            alpha=self.alpha.value(),
            percentile=self.percentile.value(),
            save_recomputed=self.save_recomputed.isChecked(),
            ask_before_save=self.ask_before_save.isChecked(),
            do_valley=self.do_valley.isChecked(),
            valley_follow=report.FOLLOW_CHOICES[self.valley_follow.currentIndex()][0],
            valley_axis=report.VALLEY_AXIS_CHOICES[self.valley_axis.currentIndex()][0],
            valley_traces=[key for key, box in self.trace_boxes.items() if box.isChecked()],
            valley_fit_line=self.valley_fit_line.isChecked(),
            valley_map_show_path=self.valley_map_show_path.isChecked(),
            valley_path_mode=self._current_path_mode(),
            valley_select=self._current_select(),
            valley_guide_follow=report.FOLLOW_CHOICES[
                self.valley_guide_follow.currentIndex()][0],
            valley_guide_halfwidth=self.valley_guide_halfwidth.value(),
            valley_waist_range=self._current_valley_ranges()[0],
            valley_width_range=self._current_valley_ranges()[1],
            forbidden_draw=self.forbidden_draw.isChecked(),
            forbidden_exclude=self.forbidden_exclude.isChecked(),
            forbidden_factor=self.forbidden_factor.value(),
        )


def main():
    app = QApplication(sys.argv)

    dialog = PlotsDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)
    params = dialog.get_values()

    results = params["results"]
    source_path = results.get('_source_path')

    # Erst die verbotenen Punkte herausnehmen, dann auswerten - sonst
    # bezoegen Bestpunkt und Region sie noch mit ein.
    n_verboten = None
    if params["forbidden_exclude"]:
        results, verboten = scan_data.mask_forbidden_grids(
            results, params["forbidden_factor"])
        n_verboten = None if verboten is None else int(verboten.sum())

    # Score, Region und Bestpunkt werden IMMER hier bestimmt: die
    # Amplituden-Scans speichern gar keinen, und die Fest-Amplituden-Scans
    # einen mit ihrem eigenen alpha. So zeigen Karte und Bericht garantiert
    # dieselbe Groesse.
    results = scan_data.analyse(
        results,
        alpha=(params["alpha"] if params["do_recompute"] else None),
        percentile=(params["percentile"] if params["do_recompute"] else None))

    saved_recomputed = None
    if params["do_recompute"] and params["save_recomputed"] and source_path:
        src = FilePath(source_path)
        target = src.with_name(f"{src.stem}_recomputed{src.suffix}")
        saved_recomputed = scan_data.save_results(results, target, overwrite=False)

    try:
        out = report.make_all(
            results,
            win_axis=params["win_axis"],
            draw_best_point=params["draw_best_point"],
            best_point_follow=params["best_point_follow"],
            best_point_value=params["best_point_value"],
            best_point_value2=params["best_point_value2"],
            point_cuts=params["point_cuts"],
            valley_map_show_path=params["valley_map_show_path"],
            fit_line_on_maps=params["fit_line_on_maps"],
            fit_line_dashed_extrapolation=params["fit_line_dashed_extrapolation"],
            amplitude_maps=params["amplitude_maps"],
            forbidden_factor=(params["forbidden_factor"]
                              if (params["forbidden_draw"] or params["forbidden_exclude"])
                              else None),
            forbidden_draw=params["forbidden_draw"],
            forbidden_excluded=params["forbidden_exclude"],
            plot_scan_overview=params["plot_overview"],
            save=True, show=params["show"],
            ask_before_save=params["ask_before_save"],
            legend_fontsize=params["legend_fontsize"],
            valley_cut=params["do_valley"],
            valley_axis=params["valley_axis"],
            valley_follow=params["valley_follow"],
            valley_traces=params["valley_traces"],
            valley_fit_line=params["valley_fit_line"],
            valley_path_mode=params["valley_path_mode"],
            valley_select=params["valley_select"],
            valley_guide_follow=params["valley_guide_follow"],
            valley_guide_halfwidth=params["valley_guide_halfwidth"],
            valley_waist_range=params["valley_waist_range"],
            valley_width_range=params["valley_width_range"],
        )
    except Exception as exc:
        QMessageBox.critical(None, "Auswertung fehlgeschlagen", f"{exc!r}")
        sys.exit(1)

    lines = [f"Auswertung fertig "
             f"({scan_data.KIND_LABELS.get(out['kind'], out['kind'])}).", ""]
    # Der Tagesordner, in den dieser Lauf tatsaechlich geschrieben hat -
    # nicht die Konstante, die beim Import galt.
    lines.append(f"Plots: {out.get('plots_dir', paths.fit_plots_dir())}")
    lines.append(f"Bericht: {out.get('results_dir', paths.FIT_RESULTS_DIR)}")
    if params["forbidden_draw"] or params["forbidden_exclude"]:
        grenze = scan_data.forbidden_boundary(results, params["forbidden_factor"])
        lines.append("")
        if grenze is None:
            lines.append("Verbotener Bereich: bei diesem Datensatz gibt es keine "
                         "zwei Eck-Spots.")
        else:
            lines.append(f"Verbotener Bereich (k = {params['forbidden_factor']:g}): "
                         f"width/MHz > {grenze['slope']:.5f} * waist/µm")
            if n_verboten is not None:
                lines.append(f"   {n_verboten} Gitterpunkte ausgeschlossen")
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
    if saved_recomputed:
        lines.append(f"Neu berechnete Fassung: {saved_recomputed}")
    QMessageBox.information(None, "Fertig", "\n".join(lines))


if __name__ == "__main__":
    main()
