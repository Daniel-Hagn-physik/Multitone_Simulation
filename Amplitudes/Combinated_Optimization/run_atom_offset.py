"""
run_atom_offset.py  -  ATOM GEGEN DAS MULTITONE-PROFIL VERSCHIEBEN
==================================================================

    ==> Dieses Skript ausfuehren, wenn die Frage lautet:
    ==>
    ==>     "Ich habe EINEN Parametersatz (waist, width, r_x, r_y). Wie
    ==>      aendern sich Uniformity und Crosstalk, wenn das Atom nicht
    ==>      genau auf der Site-Mitte sitzt?"

Eingetragen werden waist, width, r_x und r_y. Fest sind wie in den
anderen Combined-Skripten 3x4 Toene und die Linsen f1 = 75 mm, f2 = 750 mm.

Das Atom wandert radial von der Site-Mitte nach aussen - horizontal,
vertikal, diagonal (und auf Wunsch antidiagonal). Mit ihm wandern ALLE
Regionen: das Ton-Quadrat der harten Uniformity, das Pitch-Quadrat des
harten Crosstalks und die Gauss-Gewichtung der atom-gewichteten Metriken.
Das Lichtfeld steht. Ausgewertet wird wie im Penalty-Fall
(U_h, eta_h, U_w, eta_w und daraus U_c, eta_c, J).

Immer (Basisfall):

    Plot 1   r = 0 .. 0.125 x width (width in µm umgerechnet), die
             atom-gewichteten Metriken U_w, eta_w - gemittelt ueber die
             Richtung des Versatzes, also EINE Kurve je Groesse
             (auf Wunsch statt dessen einzelne Richtungen)

Optional (je eine ankreuzbare Gruppe, Vorgabe AUS):

    Harte Metriken         U_h, eta_h als eigener Plot und im Bericht
    Kombinierte Groessen   U_c, eta_c, J als eigener Plot und im Bericht
                           (rechnet die harten automatisch mit)
    Positionsschwankung    Abschaetzung (lib/position_noise.py): thermische
                           Ortsbreite plus technische Beitraege, ueber den
                           geometrischen Hebel der Optik in nm umgerechnet;
                           darin wiederum waehlbar
                             Plot 2  r = 0 .. k x sigma_pos
                             Statistik ueber die Positionsverteilung
                             sigma_pos/sigma_atom als Linie in Plot 1
                             (nur zusammen mit Plot 2)

Ergebnis:
    Fit_Plots/<Datum>/AtomOffset_..._bis0.125width_<Datum>_weighted.pdf
                                                  _hard.pdf       (optional)
                                                  _penalty.pdf    (optional)
    Fit_Plots/<Datum>/AtomOffset_..._bis1sigma_<Datum>_*.pdf       (optional)
    Fit_Results/AtomOffset_..._<Datum>_Report.md
    Results/atom_offset_....pkl                      (optional)

Die anderen Hauptskripte:
    run_penalty_scan.py  -  Multitone-Gitter mit der Penalty-Methode
    run_penalty_only.py  -  ein Parametersatz, Amplituden optimieren
    run_single_beam.py   -  ein einzelner Airy-Strahl
    run_hard_check.py    -  Hard Case zu einem vorhandenen Weighted-Scan
    run_plots.py         -  vorhandene Multitone-Datensaetze auswerten
"""

import sys
import time
from pathlib import Path as FilePath

from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt

sys.path.insert(0, str(FilePath(__file__).resolve().parent))

# Alles kommt aus lib - auch das, was eigentlich in ../Weighted_Optimization
# liegt (siehe Hinweis in lib/paths.py).
from lib import paths  # noqa: E402
from lib import atom_offset as ao  # noqa: E402
from lib import atom_offset_report as aor  # noqa: E402
from lib import position_noise as pn  # noqa: E402
from lib import single_beam_report as sbr  # noqa: E402

from airy_scale import (  # noqa: E402
    AIRY_SCALE_CHOICES, AIRY_SCALE_DIALOG_DEFAULT, choice_index_for, describe,
)
import coherence  # noqa: E402

# Grobe Rechenzeit je Atomposition bei 201 Zellen je Achse (gemessen im
# Testcontainer). Nur fuer die Anzeige im Dialog.
SEKUNDEN_JE_PUNKT_201 = 0.45


class AtomOffsetDialog(QDialog):
    """Arbeitspunkt, Profil, Atom, Versatz, Abschaetzung, Penalty, Ausgabe."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atom-Versatz im Multitone-Profil (Penalty-Fall)")

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inhalt = QWidget()
        haupt = QVBoxLayout(inhalt)
        scroll.setWidget(inhalt)
        outer.addWidget(scroll)

        info = QLabel(
            "Ein fester Parametersatz; das ATOM wandert radial von der Site-Mitte nach aussen.\n"
            "Mit ihm wandern alle Regionen (Ton-Quadrat, Pitch-Quadrat, Gauss-Gewichtung),\n"
            "das Lichtfeld steht. Ausgewertet wie im Penalty-Fall.")
        info.setStyleSheet("font-style: italic;")
        haupt.addWidget(info)

        d = ao.DEFAULTS
        e = pn.DEFAULTS

        # ------------------------------------------------------------------
        # Arbeitspunkt
        # ------------------------------------------------------------------
        g = QGroupBox(f"Arbeitspunkt (fest: {ao.N_X_FIXED}x{ao.N_Y_FIXED} Toene, "
                      f"f1 = {ao.F1_FIXED * 1e3:.0f} mm, f2 = {ao.F2_FIXED * 1e3:.0f} mm)")
        f = QFormLayout()
        self.waist = self._spin(d["waist"] * 1e6, (0.05, 10.0), 4)
        self.width = self._spin(d["width"] * 1e-6, (0.001, 5.0), 4)
        self.r_x = self._spin(d["r_x"], (0.0, 10.0), 4)
        self.r_y = self._spin(d["r_y"], (0.0, 10.0), 4)
        self.waist.setToolTip("Waist in der Atomebene (nach der Linse).")
        self.r_x.setToolTip("Aussen/Innen-Verhaeltnis der Amplituden in x: [r_x, 1, r_x]")
        self.r_y.setToolTip("Aussen/Innen-Verhaeltnis der Amplituden in y: [r_y, 1, 1, r_y]")
        self.width_info = QLabel()
        self.width_info.setStyleSheet("color: #555;")
        f.addRow("Waist (µm)", self.waist)
        f.addRow("width (MHz)", self.width)
        f.addRow("", self.width_info)
        f.addRow("r_x", self.r_x)
        f.addRow("r_y", self.r_y)
        g.setLayout(f)
        haupt.addWidget(g)

        # ------------------------------------------------------------------
        # Strahlprofil + Kohaerenz
        # ------------------------------------------------------------------
        g = QGroupBox("Strahlprofil (Airy)")
        f = QFormLayout()
        self.airy_modus = QComboBox()
        for _key, text, _wert in AIRY_SCALE_CHOICES:
            self.airy_modus.addItem(text)
        self.airy_modus.setCurrentIndex(choice_index_for(AIRY_SCALE_DIALOG_DEFAULT))
        self.airy_faktor = self._spin(AIRY_SCALE_DIALOG_DEFAULT, (0.1, 10.0), 6)
        self.airy_hinweis = QLabel(describe(AIRY_SCALE_DIALOG_DEFAULT))
        self.airy_hinweis.setWordWrap(True)
        self.airy_hinweis.setStyleSheet("color: #555;")
        f.addRow("Waist-Bedeutung", self.airy_modus)
        f.addRow("airy_scale_factor", self.airy_faktor)
        f.addRow("", self.airy_hinweis)
        g.setLayout(f)
        haupt.addWidget(g)
        self.airy_modus.currentIndexChanged.connect(self._airy_modus_gewechselt)
        self.airy_faktor.valueChanged.connect(lambda v: self.airy_hinweis.setText(describe(v)))
        self._airy_modus_gewechselt()

        self.coherence_group = coherence.CoherenceGroup(ao.N_X_FIXED, ao.N_Y_FIXED)
        haupt.addWidget(self.coherence_group)

        # ------------------------------------------------------------------
        # Atom
        # ------------------------------------------------------------------
        g = QGroupBox("Atom (Gewichtung und thermische Ortsbreite)")
        f = QFormLayout()
        self.temperatur = self._spin(d["atom_temperature"] * 1e6, (0.01, 1000.0), 3)
        self.nu_r = self._spin(d["trap_freq_r"] * 1e-3, (0.1, 1000.0), 3)
        self.sub_gitter = QSpinBox()
        self.sub_gitter.setRange(31, 2001)
        self.sub_gitter.setSingleStep(20)
        self.sub_gitter.setValue(int(d["weighted_n_grid"]))
        self.temperatur.setToolTip("Geht in die Gewichtung von U_w/eta_w UND in die "
                                   "thermische Ortsbreite der Abschaetzung ein.")
        f.addRow("Temperatur T (µK)", self.temperatur)
        f.addRow("Fallenfrequenz nu_r (kHz)", self.nu_r)
        f.addRow("Sub-Gitter: Punkte je Achse", self.sub_gitter)
        g.setLayout(f)
        haupt.addWidget(g)

        # ------------------------------------------------------------------
        # Versatz
        # ------------------------------------------------------------------
        g = QGroupBox("Versatz des Atoms")
        f = QFormLayout()
        self.modus = QComboBox()
        for _key, text in ao.RICHTUNGSMODI:
            self.modus.addItem(text)
        self.modus.setToolTip(
            "Gemittelt: an jedem Abstand r wird an gleichmaessig verteilten Winkeln\n"
            "gerechnet und gemittelt - der Erwartungswert fuer ein Atom, das um r in\n"
            "zufaelliger Richtung sitzt. Eine Kurve je Groesse.\n"
            "Einzeln: je angehakter Richtung eine Kurve.")
        f.addRow("Richtung", self.modus)
        self.n_winkel = QSpinBox()
        self.n_winkel.setRange(1, 64)
        self.n_winkel.setValue(ao.N_WINKEL_DEFAULT)
        self.n_winkel.setToolTip(
            "Winkel im Halbkreis (die Anordnung ist punktsymmetrisch). 6 = alle 30 Grad;\n"
            "gegen 16 Winkel < 0.002 pp Unterschied.")
        f.addRow("  Winkel im Halbkreis", self.n_winkel)
        self.band = QCheckBox("  Spannweite ueber die Richtungen als blasses Band zeichnen")
        f.addRow(self.band)
        zeile = QWidget()
        hl = QHBoxLayout(zeile)
        hl.setContentsMargins(0, 0, 0, 0)
        self.richtung_boxen = {}
        for rid, text, _v in ao.RICHTUNGEN:
            box = QCheckBox(text)
            box.setChecked(rid in ao.RICHTUNGEN_DEFAULT)
            self.richtung_boxen[rid] = box
            hl.addWidget(box)
        self.richtung_boxen["antixy"].setToolTip(
            "Mit Kohaerenz ist die Antidiagonale NICHT gleich der Diagonale: die\n"
            "frequenzentarteten Eckspots interferieren statisch. Der Unterschied\n"
            "steht in jedem Fall in der Symmetrie-Tabelle des Berichts.")
        f.addRow("  einzelne Richtungen", zeile)

        self.n_punkte = QSpinBox()
        self.n_punkte.setRange(3, 2001)
        self.n_punkte.setValue(ao.STUETZSTELLEN_DEFAULT)
        f.addRow("Stuetzstellen in r", self.n_punkte)

        self.plot1 = QCheckBox("Plot 1: bis zu einem Anteil der width")
        self.plot1.setChecked(True)
        self.anteil = self._spin(ao.WIDTH_ANTEIL_DEFAULT, (0.001, 2.0), 4)
        self.anteil_info = QLabel()
        self.anteil_info.setStyleSheet("color: #555;")
        f.addRow(self.plot1)
        f.addRow("  Anteil der width", self._mit_label(self.anteil, self.anteil_info))

        self.zeit_info = QLabel()
        self.zeit_info.setStyleSheet("color: #555;")
        f.addRow("", self.zeit_info)
        g.setLayout(f)
        haupt.addWidget(g)

        # ------------------------------------------------------------------
        # Harte Metriken (optional)
        # ------------------------------------------------------------------
        g = QGroupBox("Harte Metriken U_h, eta_h (optional)")
        g.setCheckable(True)
        g.setChecked(False)
        g.setToolTip("Angehakt: eigener Plot fuer U_h und eta_h, und beide stehen im Bericht.\n"
                     "Basisfall sind die atom-gewichteten Metriken; ohne harte (und ohne\n"
                     "kombinierte) wird das Harte gar nicht gerechnet - etwa doppelt so schnell.")
        self.g_hart = g
        f = QFormLayout()
        self.hart_gitter = QSpinBox()
        self.hart_gitter.setRange(51, 2001)
        self.hart_gitter.setSingleStep(50)
        self.hart_gitter.setValue(int(d["hard_n_grid"]))
        self.hart_gitter.setToolTip(
            "Zellen je Achse des Gitters, das auf der harten Region liegt und mit\n"
            "dem Atom wandert. 201 weicht von 801 um < 0.001 pp (eta_h) ab.")
        f.addRow("harte Region: Zellen je Achse", self.hart_gitter)
        self.abgleich = QCheckBox("Abgleich mit dem Optimierer bei r = 0 (einige Sekunden)")
        self.abgleich.setChecked(True)
        self.abgleich.setToolTip(
            "Rechnet r = 0 zusaetzlich mit dem Optimierer (globales Gitter) und stellt\n"
            "beide Zahlen im Bericht nebeneinander - Plausibilitaetspruefung des\n"
            "mitwandernden Gitters der harten Metriken.")
        f.addRow(self.abgleich)
        g.setLayout(f)
        haupt.addWidget(g)

        # ------------------------------------------------------------------
        # Abschaetzung der Positionsschwankung
        # ------------------------------------------------------------------
        g = QGroupBox("Positionsschwankung des Atoms (optional, 1 sigma je Achse)")
        g.setCheckable(True)
        g.setChecked(False)
        g.setToolTip("Angehakt: Abschaetzung, Plot 2 bis zur Schwankung, Statistik und\n"
                     "Markierung in Plot 1 - je nach den Haken in dieser Gruppe.")
        self.g_schwankung = g
        f = QFormLayout()
        hinweis = QLabel(
            "T und nu_r kommen aus der Gruppe 'Atom'. Technische Werte auf 0 sind nicht\n"
            "gemessen; der Hebel steht im Tooltip und im Bericht. Wer Einzelbeitraege\n"
            "eintraegt, setzt die 'Relativposition (gesamt)' auf 0.")
        hinweis.setStyleSheet("color: #555;")
        f.addRow(hinweis)
        self.schwankung = {}
        hebel = pn.hebel(width=d["width"])
        hebel_text = {
            "theta_vor_aod_urad": f"Hebel {hebel['m_pro_rad_vor_aod'] * 1e3:.3f} nm/µrad",
            "theta_vor_obj_urad": f"Hebel {hebel['m_pro_rad_vor_obj'] * 1e3:.2f} nm/µrad",
            "dv_ppm": f"Hebel {hebel['m_pro_dv_rel'] * 1e3:.3f} nm/ppm",
            "df_Hz": f"Hebel {hebel['m_pro_Hz'] * 1e12:.2f} pm/Hz",
        }
        for key, text, einheit, default, quelle in pn.EINGABEN_THERMISCH + pn.EINGABEN_TECHNISCH:
            if key in ("T_uK", "nu_kHz"):
                continue
            box = self._spin(default, (0.0, 1e6), 3)
            tip = quelle + (("\n" + hebel_text[key]) if key in hebel_text else "")
            box.setToolTip(tip)
            self.schwankung[key] = box
            f.addRow(f"{text} ({einheit})", box)
            box.valueChanged.connect(self._aktualisieren)
        self.ergebnis = QLabel()
        self.ergebnis.setWordWrap(True)
        self.ergebnis.setStyleSheet("font-weight: bold;")
        f.addRow(self.ergebnis)

        self.umfang = QComboBox()
        for _key, text in pn.UMFANG_CHOICES:
            self.umfang.addItem(text)
        self.umfang.setToolTip(
            "Welche Schwankung Plot 2 und die Statistik aufspannt.\n"
            "In U_w und eta_w steckt die thermische Ortsbreite bereits als Gewicht -\n"
            "fuer diese beiden ist 'nur technisch' die saubere Wahl.")
        f.addRow("sigma_pos aus", self.umfang)

        self.plot2 = QCheckBox("Plot 2: bis zur abgeschaetzten Positionsschwankung")
        self.plot2.setChecked(True)
        self.k_faktor = self._spin(1.0, (0.1, 10.0), 2)
        self.k_info = QLabel()
        self.k_info.setStyleSheet("color: #555;")
        f.addRow(self.plot2)
        f.addRow("  bis k x sigma_pos, k =", self._mit_label(self.k_faktor, self.k_info))

        self.statistik = QCheckBox("Statistik ueber die 2D-Positionsverteilung (Mittel, Streuung)")
        self.statistik.setChecked(True)
        self.gh_ordnung = QSpinBox()
        self.gh_ordnung.setRange(3, 15)
        self.gh_ordnung.setValue(ao.GH_ORDNUNG_DEFAULT)
        self.gh_ordnung.setToolTip("Gauss-Hermite-Knoten je Achse; 7 -> 49 Positionen.")
        f.addRow(self.statistik)
        f.addRow("  Knoten je Achse", self.gh_ordnung)

        # Die roten Linien gehoeren zu Plot 2: ohne ihn gibt es keinen Grund,
        # sigma_pos/sigma_atom in Plot 1 einzuzeichnen.
        self.markierung = QCheckBox("  sigma_pos und sigma_atom als rote Linie in Plot 1 (nur mit Plot 2)")
        self.markierung.setChecked(True)
        f.addRow(self.markierung)
        self.plot2.toggled.connect(self.markierung.setEnabled)
        g.setLayout(f)
        haupt.addWidget(g)

        # ------------------------------------------------------------------
        # Penalty
        # ------------------------------------------------------------------
        g = QGroupBox("Kombinierte Groessen U_c, eta_c, J (optional)")
        g.setCheckable(True)
        g.setChecked(False)
        g.setToolTip("Angehakt: eigener Plot fuer U_c, eta_c und J, und die drei Groessen\n"
                     "stehen im Bericht. Sie bestehen aus hart UND gewichtet - die harten\n"
                     "Metriken werden dafuer automatisch mitgerechnet (geplottet werden sie\n"
                     "nur mit Haken bei 'Harte Metriken').")
        self.g_kombi = g
        f = QFormLayout()
        self.alpha = self._spin(d["alpha"], (0.0, 1.0), 3)
        self.combo_lambda = self._spin(d["combo_lambda"], (0.0, 10.0), 3)
        self.alpha.setToolTip("J = alpha*U_c + (1-alpha)*eta_c")
        self.combo_lambda.setToolTip("U_c = 0.5*(U_h + U_w) + lambda*|U_h - U_w|")
        f.addRow("alpha", self.alpha)
        f.addRow("combo_lambda", self.combo_lambda)
        g.setLayout(f)
        haupt.addWidget(g)

        # ------------------------------------------------------------------
        # Plots
        # ------------------------------------------------------------------
        g = QGroupBox("Plots")
        f = QFormLayout()
        self.getrennt = QCheckBox("hart und atom-gewichtet in GETRENNTE Plots (nur mit harten)")
        self.getrennt.setChecked(True)
        self.achsen = QComboBox()
        for _key, text in sbr.ACHSEN_CHOICES:
            self.achsen.addItem(text)
        self.legende = QComboBox()
        for _key, text in aor.LEGENDEN_CHOICES:
            self.legende.addItem(text)
        self.schrift = self._spin(sbr.SCHRIFT_DICHTE, (0.6, 3.0), 2)
        self.schrift.setSingleStep(0.05)
        self.marker = QCheckBox("Stuetzstellen als Marker zeichnen")
        self.zeigen = QCheckBox("Plots am Ende anzeigen")
        self.ueberschreiben = QCheckBox("vorhandene Dateien ohne Rueckfrage ueberschreiben")
        f.addRow(self.getrennt)
        f.addRow("y-Achsen", self.achsen)
        f.addRow("Legende", self.legende)
        f.addRow("Schrift/Linien-Faktor", self.schrift)
        f.addRow(self.marker)
        f.addRow(self.zeigen)
        f.addRow(self.ueberschreiben)
        g.setLayout(f)
        haupt.addWidget(g)

        # ------------------------------------------------------------------
        # Speichern
        # ------------------------------------------------------------------
        g = QGroupBox("Speichern")
        f = QFormLayout()
        self.bericht = QCheckBox("Markdown-Bericht schreiben (Fit_Results/)")
        self.bericht.setChecked(True)
        self.datensatz = QCheckBox("Datensatz als .pkl speichern (Results/)")
        self.datensatz.setChecked(True)
        f.addRow(self.bericht)
        f.addRow(self.datensatz)
        ziel = QLabel(f"Plots: {paths.FIT_PLOTS_ROOT}\\<Datum>\n"
                      f"Bericht: {paths.FIT_RESULTS_DIR}\n"
                      f"Datensatz: {paths.DEFAULT_RESULTS_DIR}")
        ziel.setStyleSheet("color: #555;")
        f.addRow("", ziel)
        g.setLayout(f)
        haupt.addWidget(g)

        # ------------------------------------------------------------------
        # Knoepfe (ausserhalb der Scroll-Flaeche, immer sichtbar)
        # ------------------------------------------------------------------
        knoepfe = QHBoxLayout()
        knoepfe.addStretch(1)
        abbrechen = QPushButton("Abbrechen")
        abbrechen.clicked.connect(self.reject)
        starten = QPushButton("Rechnen")
        starten.setDefault(True)
        starten.clicked.connect(self._pruefen_und_annehmen)
        knoepfe.addWidget(abbrechen)
        knoepfe.addWidget(starten)
        outer.addLayout(knoepfe)

        for widget in (self.waist, self.width, self.temperatur, self.nu_r, self.anteil,
                       self.k_faktor, self.airy_faktor):
            widget.valueChanged.connect(self._aktualisieren)
        for widget in (self.n_punkte, self.hart_gitter, self.gh_ordnung, self.n_winkel):
            widget.valueChanged.connect(self._aktualisieren)
        for box in list(self.richtung_boxen.values()) + [self.plot1, self.plot2,
                                                          self.statistik, self.abgleich,
                                                          self.g_schwankung, self.g_hart,
                                                          self.g_kombi]:
            box.toggled.connect(self._aktualisieren)
        self.umfang.currentIndexChanged.connect(self._aktualisieren)
        self.modus.currentIndexChanged.connect(self._aktualisieren)
        self._aktualisieren()

        hoehe = 820
        screen = QApplication.primaryScreen()
        if screen is not None:
            hoehe = min(hoehe, int(0.85 * screen.availableGeometry().height()))
        self.resize(700, hoehe)

    # ------------------------------------------------------------------
    # Kleinkram
    # ------------------------------------------------------------------
    @staticmethod
    def _spin(wert, spanne, stellen):
        box = QDoubleSpinBox()
        box.setDecimals(stellen)
        box.setRange(*spanne)
        box.setSingleStep(10 ** -min(stellen, 3))
        box.setValue(wert)
        return box

    @staticmethod
    def _mit_label(widget, label):
        zeile = QWidget()
        layout = QHBoxLayout(zeile)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        layout.addWidget(label, 1)
        return zeile

    def _airy_modus_gewechselt(self):
        wert = AIRY_SCALE_CHOICES[self.airy_modus.currentIndex()][2]
        frei = wert is None
        self.airy_faktor.setEnabled(frei)
        if not frei:
            self.airy_faktor.setValue(float(wert))

    def _richtungen(self):
        return tuple(rid for rid, _t, _v in ao.RICHTUNGEN if self.richtung_boxen[rid].isChecked())

    def _modus(self):
        return ao.RICHTUNGSMODI[self.modus.currentIndex()][0]

    def _hart_rechnen(self):
        return self.g_hart.isChecked() or self.g_kombi.isChecked()

    def _abschaetzung(self):
        eingaben = {k: box.value() for k, box in self.schwankung.items()}
        eingaben["T_uK"] = self.temperatur.value()
        eingaben["nu_kHz"] = self.nu_r.value()
        return pn.abschaetzen(eingaben, width=self.width.value() * 1e6)

    def _aktualisieren(self, *_):
        """Alle abhaengigen Anzeigen: width in µm, Bereiche, Abschaetzung,
        grobe Rechenzeit."""
        # width in µm ueber dieselbe Umrechnung wie der Optimierer
        d_um = float(ao.MultitoneFlatTopOptimizer.width_to_um(
            _WidthHelfer, self.width.value() * 1e6))
        self.width_info.setText(
            f"= d = {d_um:.4f} µm in der Atomebene (Spannweite des Tonarrays, "
            f"auch Seitenlaenge der harten Uniformity-Region)")
        self.anteil_info.setText(f"= {self.anteil.value() * d_um * 1e3:.1f} nm")

        ab = self._abschaetzung()
        umfang = pn.UMFANG_CHOICES[self.umfang.currentIndex()][0]
        sigma = ab.sigma(umfang)
        self.k_info.setText(f"= {self.k_faktor.value() * sigma * 1e9:.1f} nm  "
                            f"({pn.anteil_innerhalb(self.k_faktor.value()) * 100:.0f} % "
                            f"der Atome liegen innerhalb)")
        self.ergebnis.setText(pn.kurztext(ab, umfang))

        gemittelt = self._modus() == "mittel"
        self.n_winkel.setEnabled(gemittelt)
        self.band.setEnabled(gemittelt)
        for box in self.richtung_boxen.values():
            box.setEnabled(not gemittelt)
        hart = self._hart_rechnen()
        self.getrennt.setEnabled(self.g_hart.isChecked())

        je_r = self.n_winkel.value() if gemittelt else max(len(self._richtungen()), 1)
        n = self.n_punkte.value()
        schwankung = self.g_schwankung.isChecked()
        punkte = 1  # r = 0
        laeufe = int(self.plot1.isChecked()) + int(schwankung and self.plot2.isChecked())
        punkte += laeufe * (1 + (n - 1) * je_r)
        if schwankung and self.statistik.isChecked():
            punkte += self.gh_ordnung.value() ** 2 + 1
        punkte += 8  # Symmetrie
        s_gewichtet = 0.5 * SEKUNDEN_JE_PUNKT_201
        s_hart = 0.5 * SEKUNDEN_JE_PUNKT_201 * (self.hart_gitter.value() / 201.0) ** 2
        s_pro = s_gewichtet + (s_hart if hart else 0.0)
        abgleich = hart and self.g_hart.isChecked() and self.abgleich.isChecked()
        sekunden = punkte * s_pro + (6.0 if abgleich else 0.0)
        self.zeit_info.setText(f"{punkte} Atompositionen, grob {sekunden / 60:.1f} min"
                               + ("" if hart else " (nur atom-gewichtet)"))

    def _pruefen_und_annehmen(self):
        if self._modus() == "einzeln" and not self._richtungen():
            QMessageBox.warning(self, "Richtungen", "Mindestens eine Richtung anhaken.")
            return
        schwankung = self.g_schwankung.isChecked()
        plot2 = schwankung and self.plot2.isChecked()
        statistik = schwankung and self.statistik.isChecked()
        if not (self.plot1.isChecked() or plot2 or statistik):
            QMessageBox.warning(self, "Nichts zu tun",
                                "Plot 1 anhaken - oder die Positionsschwankung mit Plot 2 "
                                "bzw. Statistik.")
            return
        umfang = pn.UMFANG_CHOICES[self.umfang.currentIndex()][0]
        if (plot2 or statistik) and not self._abschaetzung().sigma(umfang) > 0:
            QMessageBox.warning(self, "Keine Schwankung",
                                "Die gewaehlte Positionsschwankung ist 0 - Plot 2 und "
                                "Statistik brauchen einen Wert > 0.")
            return
        self.accept()

    def werte(self):
        params = ao.merged(dict(
            waist=self.waist.value() * 1e-6,
            width=self.width.value() * 1e6,
            r_x=self.r_x.value(),
            r_y=self.r_y.value(),
            profile="airy",
            airy_scale_factor=float(self.airy_faktor.value()),
            coherent=self.coherence_group.value(),
            atom_temperature=self.temperatur.value() * 1e-6,
            trap_freq_r=self.nu_r.value() * 1e3,
            weighted_n_grid=int(self.sub_gitter.value()),
            hard_n_grid=int(self.hart_gitter.value()),
            hart=self._hart_rechnen(),
            alpha=self.alpha.value(),
            combo_lambda=self.combo_lambda.value(),
        ))
        return dict(
            params=params,
            schwankung=self.g_schwankung.isChecked(),
            abschaetzung=self._abschaetzung() if self.g_schwankung.isChecked() else None,
            kombiniert=self.g_kombi.isChecked(),
            hart_plot=self.g_hart.isChecked(),
            modus=self._modus(),
            n_winkel=int(self.n_winkel.value()),
            band=self._modus() == "mittel" and self.band.isChecked(),
            umfang=pn.UMFANG_CHOICES[self.umfang.currentIndex()][0],
            k_faktor=self.k_faktor.value(),
            plot1=self.plot1.isChecked(),
            anteil=self.anteil.value(),
            plot2=self.g_schwankung.isChecked() and self.plot2.isChecked(),
            n=int(self.n_punkte.value()),
            richtungen=self._richtungen(),
            statistik=self.g_schwankung.isChecked() and self.statistik.isChecked(),
            gh_ordnung=int(self.gh_ordnung.value()),
            abgleich=self.g_hart.isChecked() and self.abgleich.isChecked(),
            getrennt=self.getrennt.isChecked(),
            achsen=sbr.ACHSEN_CHOICES[self.achsen.currentIndex()][0],
            legende=aor.LEGENDEN_CHOICES[self.legende.currentIndex()][0],
            dichte=float(self.schrift.value()),
            markierung=(self.g_schwankung.isChecked() and self.plot2.isChecked()
                        and self.markierung.isChecked()),
            marker=self.marker.isChecked(),
            zeigen=self.zeigen.isChecked(),
            ueberschreiben=self.ueberschreiben.isChecked(),
            bericht=self.bericht.isChecked(),
            datensatz=self.datensatz.isChecked(),
        )


class _WidthHelfer:
    """Minimaler Traeger der Optik-Konstanten fuer width_to_um(), damit der
    Dialog die Umrechnung live zeigen kann, ohne jedes Mal einen ganzen
    Optimierer zu bauen. Werte = Optimierer-Defaults mit f1/f2 fest."""
    theta_max = ao.MultitoneFlatTopOptimizer.DEFAULTS["theta_max"]
    f_band = ao.MultitoneFlatTopOptimizer.DEFAULTS["f_band"]
    fLO = ao.MultitoneFlatTopOptimizer.DEFAULTS["fLO"]
    f1 = ao.F1_FIXED
    f2 = ao.F2_FIXED


def _nachfragen(pfad):
    antwort = QMessageBox.question(
        None, "Datei vorhanden",
        f"'{FilePath(pfad).name}' existiert bereits. Ueberschreiben?\n\n"
        f"Nein = unter neuem Namen (_2, _3, ...) speichern.",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    return antwort == QMessageBox.Yes


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    dialog = AtomOffsetDialog()
    if dialog.exec_() != QDialog.Accepted:
        print("Abgebrochen.")
        return
    e = dialog.werte()
    ab = e["abschaetzung"]
    if ab is not None:
        print(pn.kurztext(ab, e["umfang"]))

    fortschritt = QProgressDialog("Rechne ...", "Abbrechen", 0, 1)
    fortschritt.setWindowTitle("Atom-Versatz")
    fortschritt.setWindowModality(Qt.WindowModal)
    fortschritt.setMinimumDuration(0)
    fortschritt.setValue(0)

    def melden(text, i, gesamt):
        if fortschritt.labelText() != text:
            fortschritt.setLabelText(text)
        fortschritt.setMaximum(max(int(gesamt), 1))
        fortschritt.setValue(min(int(i), int(gesamt)))
        app.processEvents()
        return not fortschritt.wasCanceled()

    start = time.time()
    lauf = ao.lauf_rechnen(
        e["params"], ab, umfang=e["umfang"], k_faktor=e["k_faktor"],
        plot_width=e["plot1"], width_anteil=e["anteil"],
        plot_schwankung=e["plot2"], n=e["n"], richtungen=e["richtungen"],
        statistik_an=e["statistik"], gh_ordnung=e["gh_ordnung"],
        abgleich=e["abgleich"], kombiniert=e["kombiniert"],
        modus=e["modus"], n_winkel=e["n_winkel"], progress=melden)
    fortschritt.close()
    dauer = time.time() - start

    if lauf.get("abgebrochen"):
        QMessageBox.information(
            None, "Abgebrochen",
            "Der Lauf wurde abgebrochen. Was fertig ist, wird geplottet und "
            "gespeichert; fehlende Werte sind NaN.")

    geschrieben = []
    if e["datensatz"]:
        pfad = paths.DEFAULT_RESULTS_DIR / ao.pkl_name(e["params"])
        geschrieben.append(ao.save_results(lauf, pfad))
        print(f"Datensatz gespeichert: {pfad}")

    confirm = (lambda _pfad: True) if e["ueberschreiben"] else _nachfragen
    for res in lauf["sweeps"]:
        markiert = e["markierung"] and res["bereich"]["art"] == "width"
        geschrieben += aor.make_plots(
            res,
            sigma_pos_m=lauf["sigma_pos_m"] if markiert else None,
            sigma_atom_m=lauf["sigma_atom"] if markiert else None,
            getrennt=e["getrennt"], penalty_plot=e["kombiniert"],
            hart=e["hart_plot"], band=e["band"],
            achsen=e["achsen"], legende=e["legende"], dichte=e["dichte"],
            marker=e["marker"], show=e["zeigen"], confirm_overwrite=confirm)

    lauf["dateien"] = [str(p) for p in geschrieben]
    lauf["hart_plot"] = e["hart_plot"]
    if e["bericht"]:
        geschrieben.append(aor.write_report(lauf, hart=e["hart_plot"]))

    text = _zusammenfassung(lauf, dauer, geschrieben)
    print(text)
    QMessageBox.information(None, "Fertig", text)


def _zusammenfassung(lauf, dauer, geschrieben):
    zeilen = [f"Gerechnet in {dauer:.0f} s.", ""]
    if lauf.get("abschaetzung"):
        zeilen += [f"Positionsschwankung ({lauf['umfang']}): sigma_pos = "
                   f"{lauf['sigma_pos_m'] * 1e9:.1f} nm je Achse", ""]
    st = lauf.get("statistik")
    if st:
        zeilen.append("Atom gaussverteilt mit sigma_pos - Mittel +- Streuung (auf der Site):")
        for basis in aor.basis_liste(lauf.get("kombiniert", False), lauf.get("hart_plot", False)):
            s = st[basis]
            zeilen.append(f"  {aor.KURZ[basis]:6s} {s['mittel'] * 100:7.3f} % +- "
                          f"{s['streuung'] * 100:.3f} pp   ({s['null'] * 100:.3f} %)")
        zeilen.append("")
    if geschrieben:
        zeilen.append("Geschrieben:")
        zeilen += [f"  {FilePath(p).name}" for p in geschrieben if p]
    return "\n".join(zeilen)


if __name__ == "__main__":
    main()
