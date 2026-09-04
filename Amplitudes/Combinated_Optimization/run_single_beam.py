"""
run_single_beam.py  -  EIN STRAHL, NUR DER WAIST
================================================

    ==> Dieses Skript ausfuehren, wenn es nicht um ein Tonarray geht,
    ==> sondern um einen EINZELNEN Strahl, und die Frage lautet:
    ==>
    ==>     "Wie laufen Uniformity und Crosstalk ueber dem Waist -
    ==>      hart im 2-µm-Kreis, atom-gewichtet, und beides zusammen
    ==>      nach der Penalty-Formel?"

Vorgegeben wird ein WAISTBEREICH - wahlweise vor der ersten Linse in mm
oder in der Atomebene in µm. Heraus kommen Kurven ueber dem Waist, ein
Markdown-Bericht und (auf Wunsch) der gepickelte Datensatz.

Warum es hier weder width noch r_x/r_y gibt
-------------------------------------------
Bei einem Ton spannt `width` nichts auf: die harte Uniformity-Region der
Multitone-Skripte ist das Quadrat, das die Spot-Zentren aufspannen, und ein
einzelner Punkt spannt kein Quadrat auf. An seine Stelle tritt hier eine
KREISREGION mit frei einstellbarem Radius (Default 2 µm) um die Site - die
Beam-Pointing-Region: das Atom kann irgendwo in diesem Kreis sitzen.

Ein Amplituden-Verhaeltnis aussen/innen braucht mindestens zwei Toene je
Achse. Bei einem Ton ist es bedeutungslos und kommt deshalb nicht vor.

Die drei Metrik-Familien
------------------------
    hart            U_h, eta_h    ueber dem Kreis
    atom-gewichtet  U_w, eta_w    Definition unveraendert aus dem Optimierer
    Penalty         U_c, eta_c, J   Formel unveraendert aus lib/combine.py

Ergebnis:
    Fit_Plots/<Datum>/SingleBeam_..._hard.pdf      (bzw. _metrics.pdf)
    Fit_Plots/<Datum>/SingleBeam_..._weighted.pdf
    Fit_Plots/<Datum>/SingleBeam_..._penalty.pdf
    Fit_Results/SingleBeam_..._Report.md
    Results/single_beam_....pkl                    (optional)

Die anderen Hauptskripte:
    run_penalty_scan.py  -  Multitone-Gitter mit der Penalty-Methode
    run_penalty_only.py  -  ein Parametersatz statt eines Gitters
    run_hard_check.py    -  Hard Case zu einem vorhandenen Weighted-Scan
    run_plots.py         -  vorhandene Multitone-Datensaetze auswerten
"""

import sys
import time
from pathlib import Path as FilePath

from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressDialog,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt

sys.path.insert(0, str(FilePath(__file__).resolve().parent))

# Alles kommt aus lib - auch das, was eigentlich in ../Weighted_Optimization
# liegt (siehe Hinweis in lib/paths.py).
from lib import paths, single_beam as sb, single_beam_report as sbr  # noqa: E402

from airy_scale import (  # noqa: E402
    AIRY_SCALE_CHOICES, AIRY_SCALE_DIALOG_DEFAULT, choice_index_for, describe,
)


# Wieviele Kraenze von Nachbar-Sites zur Auswahl stehen. Ein Kranz sind die
# 8 direkten Nachbarn (3x3 ohne Mitte), zwei Kraenze 24 (5x5 ohne Mitte).
NACHBAR_KRAENZE = [1, 2, 3]

# Wo die Legende stehen darf (matplotlib-loc, Anzeigetext).
LEGENDEN_ORTE = [
    ("upper left", "oben links (im Plot)"),
    ("upper right", "oben rechts (im Plot)"),
    ("lower left", "unten links (im Plot)"),
    ("lower right", "unten rechts (im Plot)"),
]

# Woher der markierte Waist kommt.
ARBEITSPUNKT_QUELLEN = [
    ("wert", "Waist selbst vorgeben"),
    ("min_J", "Minimum von J"),
    ("min_eta_c", "Minimum von eta_c"),
    ("min_U_h", "Minimum von U_h"),
]

WAIST_MODES = [
    ("after_lens", "Waist in der ATOMEBENE (µm)"),
    ("before_lens", "Waist VOR DER ERSTEN LINSE (mm)"),
]


class SingleBeamDialog(QDialog):
    """Waistbereich, Optik, Region, Atom, Penalty, Ausgabe."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einzelstrahl - Uniformity und Crosstalk ueber dem Waist")

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inhalt = QWidget()
        haupt = QVBoxLayout(inhalt)
        scroll.setWidget(inhalt)
        outer.addWidget(scroll)

        info = QLabel(
            "Ein einzelner Strahl, kein Tonarray. Abgefahren wird die einzige freie\n"
            "Groesse, die dabei bleibt: der Waist. width und r_x/r_y gibt es hier nicht -\n"
            "ein Punkt spannt kein Quadrat auf, und ein Aussen/Innen-Verhaeltnis braucht\n"
            "mindestens zwei Toene je Achse."
        )
        info.setStyleSheet("font-style: italic;")
        haupt.addWidget(info)

        d = sb.DEFAULTS

        # ------------------------------------------------------------------
        # Waistbereich
        # ------------------------------------------------------------------
        g_bereich = QGroupBox("Waistbereich")
        f = QFormLayout()
        self.modus = QComboBox()
        for _key, text in WAIST_MODES:
            self.modus.addItem(text)
        self.modus.setToolTip(
            "Vor der Linse wird LINEAR in win_input abgetastet; die zugehoerigen\n"
            "Waists liegen dann nicht aequidistant (waist ~ 1/win_input). Genau so\n"
            "scannen auch die Multitone-Skripte.")
        f.addRow("Bereich angeben als", self.modus)

        self.von = self._spin(sb.WAIST_RANGE_DEFAULT_UM[0], (1e-3, 1e4), 4)
        self.bis = self._spin(sb.WAIST_RANGE_DEFAULT_UM[1], (1e-3, 1e4), 4)
        self.n_punkte = QSpinBox()
        self.n_punkte.setRange(2, 20001)
        self.n_punkte.setValue(121)
        self.einheit_von = QLabel("µm")
        self.einheit_bis = QLabel("µm")
        f.addRow("von", self._mit_einheit(self.von, self.einheit_von))
        f.addRow("bis", self._mit_einheit(self.bis, self.einheit_bis))
        f.addRow("Stuetzstellen", self.n_punkte)
        g_bereich.setLayout(f)
        haupt.addWidget(g_bereich)
        self.modus.currentIndexChanged.connect(self._modus_gewechselt)

        # ------------------------------------------------------------------
        # Strahlprofil
        # ------------------------------------------------------------------
        g_profil = QGroupBox("Strahlprofil")
        f = QFormLayout()
        self.profil = QComboBox()
        self.profil.addItems(["airy", "gaussian"])
        f.addRow("Profil", self.profil)

        self.airy_modus = QComboBox()
        for _key, text, _wert in AIRY_SCALE_CHOICES:
            self.airy_modus.addItem(text)
        self.airy_modus.setCurrentIndex(choice_index_for(AIRY_SCALE_DIALOG_DEFAULT))
        self.airy_faktor = self._spin(AIRY_SCALE_DIALOG_DEFAULT, (0.1, 10.0), 6)
        self.airy_hinweis = QLabel(describe(AIRY_SCALE_DIALOG_DEFAULT))
        self.airy_hinweis.setWordWrap(True)
        self.airy_hinweis.setStyleSheet("color: #555;")
        f.addRow("Waist-Bedeutung (Airy)", self.airy_modus)
        f.addRow("airy_scale_factor", self.airy_faktor)
        f.addRow("", self.airy_hinweis)
        g_profil.setLayout(f)
        haupt.addWidget(g_profil)
        self.airy_modus.currentIndexChanged.connect(self._airy_modus_gewechselt)
        self.airy_faktor.valueChanged.connect(
            lambda v: self.airy_hinweis.setText(describe(v)))
        self._airy_modus_gewechselt()

        # ------------------------------------------------------------------
        # Optik und Fallengitter
        # ------------------------------------------------------------------
        g_optik = QGroupBox("Optik und Fallengitter")
        f = QFormLayout()
        self.f1 = self._spin(d["f1"] * 1e3, (1.0, 5000.0), 3)
        self.f2 = self._spin(d["f2"] * 1e3, (1.0, 5000.0), 3)
        self.fLO = self._spin(d["fLO"] * 1e3, (1.0, 5000.0), 3)
        self.lam = self._spin(d["lambda_opt"] * 1e9, (200.0, 2000.0), 2)
        self.pitch = self._spin(d["pitch"] * 1e6, (0.1, 100.0), 4)
        self.nachbarn = QComboBox()
        for kranz in NACHBAR_KRAENZE:
            self.nachbarn.addItem(
                f"{(2 * kranz + 1) ** 2 - 1} Sites"
                + ("  (die direkten Nachbarn)" if kranz == 1 else f"  ({kranz} Kraenze)"))
        self.nachbarn.setCurrentIndex(NACHBAR_KRAENZE.index(int(d["neighbour_ring"])))
        self.nachbarn.setToolTip(
            "Wieviele Nachbar-Sites in den Crosstalk eingehen.\n\n"
            "8 Sites = der Kranz direkt um die Site herum (die 3x3-Umgebung ohne\n"
            "die Mitte) - genau das, was der Multitone-Optimierer rechnet, und die\n"
            "einzige Wahl, mit der die Zahlen mit dessen Scans vergleichbar sind.\n\n"
            "24 Sites nimmt den naechsten Kranz dazu (5x5 ohne Mitte). Bei Airy\n"
            "faellt der Beitrag mit 1/r^3 ab - gemessen bringt der zweite Kranz\n"
            "rund 20 % mehr Crosstalk, ist also klein, aber nicht null.")
        f.addRow("f1 (mm)", self.f1)
        f.addRow("f2 (mm)", self.f2)
        f.addRow("fLO (mm)", self.fLO)
        f.addRow("Wellenlaenge (nm)", self.lam)
        f.addRow("pitch (µm)", self.pitch)
        f.addRow("Nachbar-Sites (Crosstalk)", self.nachbarn)
        hinweis_optik = QLabel(
            "f1 = 75 mm ist der Wert, mit dem die vorhandenen Datensaetze dieses\n"
            "Projekts gerechnet wurden - damit bedeutet ein Waist hier dieselbe Zahl\n"
            "wie dort. Steht im Aufbau eine andere Brennweite, hier aendern.")
        hinweis_optik.setStyleSheet("color: #555;")
        f.addRow("", hinweis_optik)
        g_optik.setLayout(f)
        haupt.addWidget(g_optik)

        # ------------------------------------------------------------------
        # Harte Region
        # ------------------------------------------------------------------
        g_hart = QGroupBox("Harte Region")
        f = QFormLayout()
        self.radius = self._spin(d["hard_radius"] * 1e6, (0.05, 20.0), 4)
        self.radius.setToolTip(
            "RADIUS, nicht Durchmesser. 2 µm ist die Beam-Pointing-Region:\n"
            "das Atom kann irgendwo in diesem Kreis sitzen.\n\n"
            "Die Uniformity nimmt IMMER diesen Kreis - er ist der Grund, warum es\n"
            "hier ueberhaupt eine harte Region gibt.")
        self.crosstalk_region = QComboBox()
        for _key, text in sb.CROSSTALK_REGION_CHOICES:
            self.crosstalk_region.addItem(text)
        self.crosstalk_region.setCurrentIndex(
            [k for k, _t in sb.CROSSTALK_REGION_CHOICES].index(
                sb.DEFAULTS["hard_crosstalk_region"]))
        self.penalty_region = QComboBox()
        for _key, text in sb.PENALTY_REGION_CHOICES:
            self.penalty_region.addItem(text)
        self.penalty_region.setToolTip(
            "Nur bei 'beide' zu entscheiden: die Penalty-Kombination braucht EINE\n"
            "Definition von eta_h. eta_c und J werden mit der hier gewaehlten Region\n"
            "gerechnet; die andere Kurve steht daneben zum Vergleich.")
        self.crosstalk_region.setToolTip(
            "Ueber welcher Flaeche der HARTE Crosstalk gerechnet wird.\n\n"
            "Kreis: eine Region fuer beide Groessen - U_h und eta_h sagen dann\n"
            "etwas ueber dieselbe Flaeche aus.\n\n"
            "Pitch-Quadrat: die Region, die die Multitone-Skripte fuer den\n"
            "Crosstalk benutzen (Seitenlaenge = pitch). Damit ist eta_h direkt\n"
            "mit jenen Scans vergleichbar - bezieht sich dann aber auf eine\n"
            "ANDERE Flaeche als die Uniformity daneben. Der Bericht und der\n"
            "Plot-Titel sagen das dazu, und der Dateiname bekommt '_pitchbox',\n"
            "damit sich die beiden Faelle nie ueberschreiben.\n\n"
            "beide: beides in einem Lauf. Die harte Figur zeigt dann zwei\n"
            "Crosstalk-Kurven (eta_h^circ und eta_h^box) neben derselben\n"
            "Uniformity - die haengt nicht an der Crosstalk-Region und wird nur\n"
            "einmal gerechnet. Fuer eta_c und J wird die Region darunter benutzt.")
        self.hart_gitter = QSpinBox()
        self.hart_gitter.setRange(51, 4001)
        self.hart_gitter.setSingleStep(50)
        self.hart_gitter.setValue(int(d["hard_n_grid"]))
        self.hart_gitter.setToolTip(
            "Zellen je Achse in der jeweiligen Region (Mittelpunktsregel).\n"
            "401 liefert bereits fuenf bis sechs stabile Stellen - geprueft gegen\n"
            "3201, fuer beide Crosstalk-Regionen.")
        f.addRow("Kreisradius (µm)", self.radius)
        f.addRow("Crosstalk-Region", self.crosstalk_region)
        f.addRow("davon in eta_c / J", self.penalty_region)
        f.addRow("Gitterzellen je Achse", self.hart_gitter)
        g_hart.setLayout(f)
        haupt.addWidget(g_hart)
        self.crosstalk_region.currentIndexChanged.connect(self._region_umschalten)
        self._region_umschalten()

        # ------------------------------------------------------------------
        # Atom
        # ------------------------------------------------------------------
        g_atom = QGroupBox("Atom (gewichtete Metriken)")
        f = QFormLayout()
        self.temperatur = self._spin(d["atom_temperature"] * 1e6, (0.01, 1000.0), 3)
        self.nu_r = self._spin(d["trap_freq_r"] * 1e-3, (0.1, 1000.0), 3)
        self.n_sigma = self._spin(float(d["weighted_n_sigma"]), (1.0, 20.0), 2)
        self.sub_gitter = QSpinBox()
        self.sub_gitter.setRange(31, 2001)
        self.sub_gitter.setSingleStep(20)
        self.sub_gitter.setValue(int(d["weighted_n_grid"]))
        self.versatz_x = self._spin(0.0, (-10.0, 10.0), 4)
        self.versatz_y = self._spin(0.0, (-10.0, 10.0), 4)
        f.addRow("Temperatur (µK)", self.temperatur)
        f.addRow("Fallenfrequenz nu_r (kHz)", self.nu_r)
        f.addRow("Sub-Gitter: +- n sigma", self.n_sigma)
        f.addRow("Sub-Gitter: Punkte je Achse", self.sub_gitter)
        f.addRow("Atom-Versatz x (µm)", self.versatz_x)
        f.addRow("Atom-Versatz y (µm)", self.versatz_y)
        g_atom.setLayout(f)
        haupt.addWidget(g_atom)

        # ------------------------------------------------------------------
        # Penalty
        # ------------------------------------------------------------------
        g_pen = QGroupBox("Penalty-Kombination")
        f = QFormLayout()
        self.alpha = self._spin(d["alpha"], (0.0, 1.0), 3)
        self.combo_lambda = self._spin(d["combo_lambda"], (0.0, 10.0), 3)
        self.alpha.setToolTip("J = alpha*U_c + (1-alpha)*eta_c")
        self.combo_lambda.setToolTip(
            "U_c = 0.5*(U_h + U_w) + lambda*|U_h - U_w|; 0 waere der reine Mittelwert.")
        f.addRow("alpha", self.alpha)
        f.addRow("combo_lambda", self.combo_lambda)
        g_pen.setLayout(f)
        haupt.addWidget(g_pen)

        # ------------------------------------------------------------------
        # Plots
        # ------------------------------------------------------------------
        g_plot = QGroupBox("Plots")
        f = QFormLayout()
        self.getrennt = QCheckBox("hart und atom-gewichtet in GETRENNTE Plots")
        self.getrennt.setChecked(True)
        self.getrennt.setToolTip(
            "Angehakt: zwei Figuren (hard / weighted).\n"
            "Nicht angehakt: alle vier Kurven in einer Figur (hart durchgezogen,\n"
            "gewichtet gestrichelt).\n\n"
            "Uniformity und Crosstalk bleiben in jedem Fall zusammen.")
        self.penalty_plot = QCheckBox("eigener Plot fuer die Penalty-Kombination (U_c, eta_c, J)")
        self.penalty_plot.setChecked(True)
        self.achsen = QComboBox()
        for _key, text in sbr.ACHSEN_CHOICES:
            self.achsen.addItem(text)
        self.achsen.setToolTip(
            "'automatisch' folgt derselben Regel wie der Querschnitt der\n"
            "Multitone-Auswertung: gemeinsame y-Achse, solange die Wertebereiche\n"
            "innerhalb einer Groessenordnung liegen und jede Kurve auf der\n"
            "gemeinsamen Achse noch etwas zu sehen gibt - sonst eine zweite Achse\n"
            "rechts, in der Farbe ihrer Kurve.")
        self.legende = QComboBox()
        for _key, text in LEGENDEN_ORTE:
            self.legende.addItem(text)
        self.legende.setToolTip(
            "Die Legende steht im Plot. Damit sie nicht auf den Kurven liegt,\n"
            "wird bei den oberen Positionen etwas Luft nach oben gemacht -\n"
            "abgeschnitten wird nichts.")
        self.schrift = self._spin(sbr.SCHRIFT_DICHTE, (0.6, 3.0), 2)
        self.schrift.setSingleStep(0.05)
        self.schrift.setToolTip(
            "Faktor auf Schrift, Linien, Marker und Teilstriche - gemeinsam,\n"
            "damit das Bild stimmig bleibt. 1.00 ist der Massstab, den die\n"
            "Multitone-Karten benutzen (auf volle Textbreite ausgelegt); eine\n"
            "Kurvenfigur wird im Text meist kleiner gesetzt und braucht mehr.")
        self.titel = QCheckBox("Ueberschrift zeichnen")
        self.titel.setToolTip(
            "Aus: die Abbildung traegt keinen Titel - was er sagen wuerde\n"
            "(Region, sigma_atom, Penalty-Parameter), steht im Bericht und im\n"
            "Dateinamen, und in LaTeX steht die Bildunterschrift darunter.")
        self.marker = QCheckBox("Stuetzstellen als Marker zeichnen")
        self.zeigen = QCheckBox("Plots am Ende anzeigen")
        self.ueberschreiben = QCheckBox("vorhandene Dateien ohne Rueckfrage ueberschreiben")
        f.addRow(self.getrennt)
        f.addRow(self.penalty_plot)
        f.addRow("y-Achsen", self.achsen)
        f.addRow("Legende", self.legende)
        f.addRow("Schrift/Linien-Faktor", self.schrift)
        f.addRow(self.titel)
        f.addRow(self.marker)
        f.addRow(self.zeigen)
        f.addRow(self.ueberschreiben)
        g_plot.setLayout(f)
        haupt.addWidget(g_plot)

        # ------------------------------------------------------------------
        # Arbeitspunkt
        # ------------------------------------------------------------------
        g_ap = QGroupBox("Arbeitspunkt einzeichnen")
        f = QFormLayout()
        self.ap_an = QCheckBox("senkrechte Linie und Stern auf jeder Kurve")
        self.ap_an.setChecked(True)
        self.ap_an.setToolTip(
            "Markiert einen Waist in allen Figuren. Der Bericht bekommt dazu\n"
            "einen Abschnitt mit den Werten aller Groessen an dieser Stelle,\n"
            "zwischen den benachbarten Stuetzstellen interpoliert.")
        self.ap_quelle = QComboBox()
        for _key, text in ARBEITSPUNKT_QUELLEN:
            self.ap_quelle.addItem(text)
        self.ap_wert = self._spin(sb.ARBEITSPUNKT_DEFAULT_UM, (0.01, 100.0), 4)
        f.addRow(self.ap_an)
        f.addRow("woher", self.ap_quelle)
        f.addRow("Waist (µm)", self.ap_wert)
        g_ap.setLayout(f)
        haupt.addWidget(g_ap)
        self.ap_an.toggled.connect(self._arbeitspunkt_umschalten)
        self.ap_quelle.currentIndexChanged.connect(self._arbeitspunkt_umschalten)
        self._arbeitspunkt_umschalten()

        # ------------------------------------------------------------------
        # Positions-Sweep
        # ------------------------------------------------------------------
        g_pos = QGroupBox("Atomposition durchfahren (bei festem Waist)")
        f = QFormLayout()
        self.pos_an = QCheckBox("zusaetzlich das Atom aus der Mitte herausfahren")
        self.pos_an.setChecked(True)
        self.pos_an.setToolTip(
            "Zweiter Lauf mit FESTEM Waist: nicht der Strahl wird veraendert,\n"
            "sondern die Position des Atoms. Eigene Figuren, eigener Bericht,\n"
            "eigener Datensatz.")
        self.pos_waist = self._spin(sb.ARBEITSPUNKT_DEFAULT_UM, (0.01, 100.0), 4)
        self.pos_waist.setToolTip("Der Waist, bei dem das Atom durchgefahren wird.")
        self.pos_bis = QComboBox()
        self.pos_bis.addItems(["bis zum Waist", "eigener Wert"])
        self.pos_bis_wert = self._spin(sb.ARBEITSPUNKT_DEFAULT_UM, (0.001, 100.0), 4)
        self.pos_n = QSpinBox()
        self.pos_n.setRange(2, 5001)
        self.pos_n.setValue(sb.OFFSET_STUETZSTELLEN_DEFAULT)
        self.pos_vertikal = QCheckBox("senkrecht  (0, r)")
        self.pos_vertikal.setChecked(True)
        self.pos_diagonal = QCheckBox("diagonal  (r/sqrt2, r/sqrt2)")
        self.pos_diagonal.setChecked(True)
        for box in (self.pos_vertikal, self.pos_diagonal):
            box.setToolTip(
                "Eine waagerechte Richtung fehlt mit Absicht: das Strahlprofil ist\n"
                "rotationssymmetrisch, waagerecht ist dasselbe wie senkrecht.\n"
                "Diagonal ist es NICHT - nicht wegen des Strahls, sondern wegen der\n"
                "Nachbar-Sites: die liegen auf einem Quadratgitter, und diagonal ist\n"
                "die naechste Site sqrt(2) mal weiter weg.")
        self.pos_folgt = QCheckBox("harte Region folgt dem Atom")
        self.pos_folgt.setChecked(bool(d["offset_hard_follows_atom"]))
        self.pos_folgt.setToolTip(
            "An: Kreis (und ggf. Pitch-Quadrat) sitzen an der jeweiligen\n"
            "Atomposition - hier die sinnvolle Lesart, denn die Atomposition IST\n"
            "die abgefahrene Groesse.\n\n"
            "Aus: die Region bleibt auf der Site. Dann sind U_h und eta_h ueber\n"
            "dem Versatz konstant - auch eine Aussage, nur eben eine langweilige.")
        f.addRow(self.pos_an)
        f.addRow("Waist (µm)", self.pos_waist)
        f.addRow("Versatz bis", self.pos_bis)
        f.addRow("eigener Wert (µm)", self.pos_bis_wert)
        f.addRow("Stuetzstellen", self.pos_n)
        f.addRow(self.pos_vertikal)
        f.addRow(self.pos_diagonal)
        f.addRow(self.pos_folgt)
        g_pos.setLayout(f)
        haupt.addWidget(g_pos)
        self.pos_an.toggled.connect(self._position_umschalten)
        self.pos_bis.currentIndexChanged.connect(self._position_umschalten)
        self._position_umschalten()

        # ------------------------------------------------------------------
        # Speichern
        # ------------------------------------------------------------------
        g_save = QGroupBox("Speichern")
        f = QFormLayout()
        self.bericht = QCheckBox("Markdown-Bericht schreiben (Fit_Results/)")
        self.bericht.setChecked(True)
        self.datensatz = QCheckBox("Datensatz als .pkl speichern (Results/)")
        self.datensatz.setChecked(True)
        self.pkl_name = QLineEdit()
        self.pkl_name.setPlaceholderText("automatisch aus Profil, Punktzahl und Radius")
        f.addRow(self.bericht)
        f.addRow(self.datensatz)
        f.addRow("Dateiname (.pkl)", self.pkl_name)
        ziel = QLabel(f"Plots: {paths.FIT_PLOTS_ROOT}\\<Datum>\n"
                      f"Bericht: {paths.FIT_RESULTS_DIR}\n"
                      f"Datensatz: {paths.DEFAULT_RESULTS_DIR}")
        ziel.setStyleSheet("color: #555;")
        f.addRow("", ziel)
        g_save.setLayout(f)
        haupt.addWidget(g_save)

        # ------------------------------------------------------------------
        # Knoepfe
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

        self.resize(640, 820)

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
    def _mit_einheit(widget, label):
        zeile = QWidget()
        layout = QHBoxLayout(zeile)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        layout.addWidget(label)
        return zeile

    def _modus_gewechselt(self):
        vor_der_linse = self.modus.currentIndex() == 1
        einheit = "mm" if vor_der_linse else "µm"
        self.einheit_von.setText(einheit)
        self.einheit_bis.setText(einheit)
        # Startwerte je Einheit. Die beiden Bereiche beschreiben dasselbe
        # Fenster (siehe single_beam.WAIST_RANGE_DEFAULT_UM) - es wechselt
        # also nur die Schreibweise, nicht der abgefahrene Bereich.
        if vor_der_linse:
            self.von.setValue(sb.WIN_INPUT_RANGE_DEFAULT_MM[0])
            self.bis.setValue(sb.WIN_INPUT_RANGE_DEFAULT_MM[1])
        else:
            self.von.setValue(sb.WAIST_RANGE_DEFAULT_UM[0])
            self.bis.setValue(sb.WAIST_RANGE_DEFAULT_UM[1])

    def _position_umschalten(self):
        an = self.pos_an.isChecked()
        for widget in (self.pos_waist, self.pos_bis, self.pos_n,
                       self.pos_vertikal, self.pos_diagonal, self.pos_folgt):
            widget.setEnabled(an)
        self.pos_bis_wert.setEnabled(an and self.pos_bis.currentIndex() == 1)

    def _region_umschalten(self):
        """Welche Region in die Penalty geht, ist nur bei 'beide' eine
        Frage - sonst gibt es nur die eine."""
        beide = (sb.CROSSTALK_REGION_CHOICES[
            self.crosstalk_region.currentIndex()][0] == "beide")
        self.penalty_region.setEnabled(beide)

    def _arbeitspunkt_umschalten(self):
        an = self.ap_an.isChecked()
        self.ap_quelle.setEnabled(an)
        # Der Waist ist nur bei "selbst vorgeben" von Hand einzustellen;
        # sonst kommt er aus dem Ergebnis und wird nachher eingetragen.
        self.ap_wert.setEnabled(an and self.ap_quelle.currentIndex() == 0)

    def _airy_modus_gewechselt(self):
        wert = AIRY_SCALE_CHOICES[self.airy_modus.currentIndex()][2]
        frei = wert is None
        self.airy_faktor.setEnabled(frei)
        if not frei:
            self.airy_faktor.setValue(float(wert))

    def _pruefen_und_annehmen(self):
        if self.von.value() >= self.bis.value():
            QMessageBox.warning(self, "Bereich", "'von' muss kleiner als 'bis' sein.")
            return
        radius_um = self.radius.value()
        pitch_um = self.pitch.value()
        if radius_um >= pitch_um / 2:
            antwort = QMessageBox.question(
                self, "Kreis groesser als der halbe Pitch",
                f"Der Kreisradius ({radius_um:.3f} µm) ist nicht kleiner als der halbe "
                f"Pitch ({pitch_um / 2:.3f} µm). Die Region reicht dann bis zur "
                f"Nachbar-Site; Uniformity und Crosstalk sind dort nur noch schwer zu "
                f"lesen.\n\nTrotzdem rechnen?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if antwort != QMessageBox.Yes:
                return
        if self.pos_an.isChecked() and not (
                self.pos_vertikal.isChecked() or self.pos_diagonal.isChecked()):
            QMessageBox.warning(
                self, "Positions-Sweep",
                "Mindestens eine Richtung anhaken - oder den Positions-Sweep abwaehlen.")
            return
        self.accept()

    # ------------------------------------------------------------------
    # Auswertung des Dialogs
    # ------------------------------------------------------------------
    def werte(self):
        params = dict(sb.DEFAULTS)
        params.update(
            profile=self.profil.currentText(),
            airy_scale_factor=float(self.airy_faktor.value()),
            f1=self.f1.value() * 1e-3,
            f2=self.f2.value() * 1e-3,
            fLO=self.fLO.value() * 1e-3,
            lambda_opt=self.lam.value() * 1e-9,
            pitch=self.pitch.value() * 1e-6,
            neighbour_ring=NACHBAR_KRAENZE[self.nachbarn.currentIndex()],
            hard_radius=self.radius.value() * 1e-6,
            hard_n_grid=int(self.hart_gitter.value()),
            hard_crosstalk_region=sb.CROSSTALK_REGION_CHOICES[
                self.crosstalk_region.currentIndex()][0],
            penalty_crosstalk_region=sb.PENALTY_REGION_CHOICES[
                self.penalty_region.currentIndex()][0],
            atom_temperature=self.temperatur.value() * 1e-6,
            trap_freq_r=self.nu_r.value() * 1e3,
            weighted_n_sigma=float(self.n_sigma.value()),
            weighted_n_grid=int(self.sub_gitter.value()),
            atom_offset_x=self.versatz_x.value() * 1e-6,
            atom_offset_y=self.versatz_y.value() * 1e-6,
            alpha=self.alpha.value(),
            combo_lambda=self.combo_lambda.value(),
            offset_hard_follows_atom=self.pos_folgt.isChecked(),
        )
        return dict(
            params=params,
            modus=WAIST_MODES[self.modus.currentIndex()][0],
            von=self.von.value(),
            bis=self.bis.value(),
            n_punkte=int(self.n_punkte.value()),
            getrennt=self.getrennt.isChecked(),
            penalty_plot=self.penalty_plot.isChecked(),
            achsen=sbr.ACHSEN_CHOICES[self.achsen.currentIndex()][0],
            legende=LEGENDEN_ORTE[self.legende.currentIndex()][0],
            dichte=float(self.schrift.value()),
            titel=self.titel.isChecked(),
            position_an=self.pos_an.isChecked(),
            position_waist=float(self.pos_waist.value()),
            position_r_max=(None if self.pos_bis.currentIndex() == 0
                            else float(self.pos_bis_wert.value())),
            position_n=int(self.pos_n.value()),
            position_richtungen=tuple(
                r for r, box in (("vertikal", self.pos_vertikal),
                                 ("diagonal", self.pos_diagonal)) if box.isChecked()),
            arbeitspunkt_an=self.ap_an.isChecked(),
            arbeitspunkt_quelle=ARBEITSPUNKT_QUELLEN[self.ap_quelle.currentIndex()][0],
            arbeitspunkt_wert=float(self.ap_wert.value()),
            marker=self.marker.isChecked(),
            zeigen=self.zeigen.isChecked(),
            ueberschreiben=self.ueberschreiben.isChecked(),
            bericht=self.bericht.isChecked(),
            datensatz=self.datensatz.isChecked(),
            pkl_name=self.pkl_name.text().strip(),
        )


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    dialog = SingleBeamDialog()
    if dialog.exec_() != QDialog.Accepted:
        print("Abgebrochen.")
        return
    e = dialog.werte()

    waist_vals, _win_vals = sb.build_waist_grid(
        e["modus"], e["von"], e["bis"], e["n_punkte"], e["params"])

    fortschritt = QProgressDialog("Rechne Metriken ueber dem Waist ...", "Abbrechen",
                                  0, len(waist_vals))
    fortschritt.setWindowTitle("Einzelstrahl")
    fortschritt.setWindowModality(Qt.WindowModal)
    fortschritt.setMinimumDuration(0)
    fortschritt.setValue(0)

    def melden(i, n):
        fortschritt.setValue(i)
        app.processEvents()
        return not fortschritt.wasCanceled()

    start = time.time()
    results = sb.sweep(waist_vals, e["params"], progress=melden)
    fortschritt.setValue(len(waist_vals))
    fortschritt.close()
    dauer = time.time() - start

    if results.get("abgebrochen"):
        QMessageBox.information(
            None, "Abgebrochen",
            "Der Lauf wurde abgebrochen. Die bereits gerechneten Punkte werden "
            "geplottet, der Rest bleibt leer.")

    geschrieben = []

    if e["datensatz"]:
        name = e["pkl_name"] or sb.pkl_name(e["params"], len(waist_vals))
        if not name.lower().endswith(".pkl"):
            name += ".pkl"
        pfad = FilePath(name)
        if not pfad.is_absolute():
            pfad = paths.DEFAULT_RESULTS_DIR / pfad
        geschrieben.append(sb.save_results(results, pfad))
        print(f"Datensatz gespeichert: {pfad}")

    results["arbeitspunkt_um"] = _arbeitspunkt(results, e)

    confirm = (lambda _pfad: True) if e["ueberschreiben"] else None
    geschrieben += sbr.make_plots(
        results, getrennt=e["getrennt"], achsen=e["achsen"],
        penalty_plot=e["penalty_plot"], marker=e["marker"],
        titel=e["titel"], legende=e["legende"], dichte=e["dichte"],
        show=e["zeigen"], confirm_overwrite=confirm)

    if e["bericht"]:
        geschrieben.append(sbr.write_report(results))

    if e["position_an"]:
        geschrieben += _positions_sweep(app, e, confirm)

    zusammenfassung = _zusammenfassung(results, dauer, geschrieben)
    print(zusammenfassung)
    QMessageBox.information(None, "Fertig", zusammenfassung)


def _positions_sweep(app, e, confirm):
    """Zweiter Lauf: fester Waist, das Atom wandert. Eigene Figuren,
    eigener Bericht, eigener Datensatz."""
    waist = e["position_waist"] * 1e-6
    r_max = None if e["position_r_max"] is None else e["position_r_max"] * 1e-6
    n = e["position_n"]
    richtungen = e["position_richtungen"]

    fortschritt = QProgressDialog(
        "Fahre die Atomposition durch ...", "Abbrechen", 0, n * len(richtungen))
    fortschritt.setWindowTitle("Einzelstrahl - Atomposition")
    fortschritt.setWindowModality(Qt.WindowModal)
    fortschritt.setMinimumDuration(0)
    fortschritt.setValue(0)

    def melden(i, gesamt):
        fortschritt.setValue(i)
        app.processEvents()
        return not fortschritt.wasCanceled()

    results = sb.sweep_offset(waist, e["params"], n=n, r_max=r_max,
                              richtungen=richtungen, progress=melden)
    fortschritt.setValue(n * len(richtungen))
    fortschritt.close()

    geschrieben = []
    if e["datensatz"]:
        pfad = paths.DEFAULT_RESULTS_DIR / sb.offset_pkl_name(
            e["params"], e["position_waist"], n)
        geschrieben.append(sb.save_results(results, pfad))
        print(f"Datensatz gespeichert: {pfad}")

    geschrieben += sbr.make_offset_plots(
        results, getrennt=e["getrennt"], achsen=e["achsen"],
        penalty_plot=e["penalty_plot"], marker=e["marker"],
        titel=e["titel"], legende=e["legende"], dichte=e["dichte"],
        show=e["zeigen"], confirm_overwrite=confirm)

    if e["bericht"]:
        geschrieben.append(sbr.write_offset_report(results))
    return geschrieben


def _arbeitspunkt(results, e):
    """Der zu markierende Waist in µm, oder None.

    Bei "Minimum von ..." wird die Stuetzstelle genommen, an der die
    gewaehlte Groesse am kleinsten ist - nicht interpoliert: zwischen den
    Stuetzstellen liegt kein gerechneter Wert, und ein Minimum
    dazwischenzuschaetzen waere eine Genauigkeit, die die Rechnung nicht
    hergibt. Wer es genauer braucht, faehrt den Bereich enger ab.
    """
    if not e["arbeitspunkt_an"]:
        return None
    quelle = e["arbeitspunkt_quelle"]
    if quelle == "wert":
        return float(e["arbeitspunkt_wert"])
    key = {"min_J": "combined_score", "min_eta_c": "crosstalk_kombi",
           "min_U_h": "uniformity_hart"}[quelle]
    best = results["best"].get(key)
    return None if best is None else float(best["waist_um"])


def _zusammenfassung(results, dauer, geschrieben):
    zeilen = [f"{len(results['waist_um'])} Waists gerechnet in {dauer:.1f} s.", ""]
    namen = sbr.zeilen_namen(results)
    # Die Kombi-Groessen stehen unten beim Arbeitspunkt; hier interessiert,
    # wo die einzelnen Metriken ihr Minimum haben.
    gezeigt = [k for k in sbr.kurven_keys(results)
               if k not in ("uniformity_kombi", "crosstalk_kombi")]
    for key in gezeigt:
        name = namen[key]
        best = results["best"].get(key)
        if best is None:
            zeilen.append(f"{name}: keine gueltigen Werte")
            continue
        rand = "  <- am Rand des Bereichs" if best["at_edge"] else ""
        zeilen.append(f"{name}: min {best['value'] * 100:.4g} % "
                      f"bei Waist {best['waist_um']:.4f} µm "
                      f"(w_in {best['win_input_mm']:.4f} mm){rand}")
    ap = results.get("arbeitspunkt_um")
    if ap is not None:
        zeilen.append("")
        zeilen.append(f"Markierter Arbeitspunkt: Waist {ap:.4f} µm")
        werte = sbr.arbeitspunkt_werte(results, ap)
        zeilen.append("  " + ",  ".join(
            f"{sbr.TEXT_LABELS[k]} = {v:.4g} %" for k, v in werte.items()))

    if geschrieben:
        zeilen.append("")
        zeilen.append("Geschrieben:")
        zeilen += [f"  {FilePath(p).name}" for p in geschrieben if p]
    return "\n".join(zeilen)


if __name__ == "__main__":
    main()
