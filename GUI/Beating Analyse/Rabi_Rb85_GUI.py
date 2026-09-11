"""
Rabi Rb85 GUI (PyQt5)
=====================
Two-photon Raman Rabi oscillations on the Rb-85 ground-state hyperfine
transition |5S_1/2, F=2, m> <-> |5S_1/2, F=3, m>, with real atomic data.

What this is
------------
An effective TWO-LEVEL model whose coefficients are not free parameters.  The
5P manifold is adiabatically eliminated, but the elimination runs over BOTH
fine-structure lines and ALL their hyperfine levels, with signed dipole matrix
elements taken from ARC.  Three numbers come out, all linear in intensity:

    Omega(I)    = C_rabi    * I     two-photon Rabi frequency
    delta_LS(I) = C_shift   * I     differential light shift
    Gamma(I)    = C_scatter * I     photon scattering rate

Because Omega and delta_LS carry the same factor I, their ratio eta is a pure
number.  That is what closes the two-level problem: the light shift caps the
contrast at 1/(1+eta^2) and speeds the rotation up by sqrt(1+eta^2), but it
adds no structure of its own.  At constant intensity the solution is then the
textbook

    P = Omega^2/(Omega^2+delta^2) * sin^2( sqrt(Omega^2+delta^2) t/2 )

and this GUI uses that closed form.  As soon as something enters that does NOT
scale with I - a Zeeman detuning, a static two-photon detuning - the closed
form is no longer valid and rb85_raman.excitation_series() integrates instead.

Geometry assumed
----------------
Co-propagating legs, sigma+ on both.  The scalar part of the AC Stark operator
cannot drive a hyperfine transition at all; only the vector part can, and it
goes as eps_1* x eps_2 - which VANISHES for two linearly polarised legs.
sigma+/sigma+ gives eps* x eps = z and drives Delta m_F = 0, the clock
transition.  Co-propagating also means Delta k = w_hfs/c = 64 1/m: the
transition is Doppler-free and insensitive to the atom's motion.

Structured profiles
-------------------
The beam here is a plain Gaussian, I0 = 2P/(pi w^2).  To feed a structured or
time-dependent profile instead (the multitone flat-top, say) hand its I(t) to
rb85_raman.excitation_series() - the physics module takes an arbitrary
intensity history and nothing else has to change.

What is NOT in it
-----------------
  * atomic motion and the trap (irrelevant for co-propagating Raman, but the
    trap's own light shift is not modelled either)
  * finite pulse rise time
  * polarisation errors and the resulting vector light shifts
  * repumping out of m = +/-3 in F = 3, which are dark for this drive
  * the excited state beyond adiabatic elimination - at |Delta| of a few GHz
    against Gamma/2pi = 5.75 MHz that is a very good approximation, and the
    scattering it costs IS accounted for, as a survival factor

Start:
    python Rabi_Rb85_GUI.py
"""

import sys
import json
import datetime
from pathlib import Path as FilePath

# Die Hilfsmodule liegen in kern/ - dieses Skript bleibt oben, damit klar ist,
# was man startet. Der Pfad wird relativ zur DATEI gesetzt, nicht zum
# Arbeitsverzeichnis, damit der Start aus PyCharm und aus der Konsole gleich
# funktioniert.
sys.path.insert(0, str(FilePath(__file__).resolve().parent / "kern"))

import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QGroupBox,
    QScrollArea, QSplitter, QComboBox, QMessageBox
)
from PyQt5.QtCore import Qt

import rb85_raman as R

# beating_profil zieht kern/beating_physik.py herein. Fehlt das (weil dieses
# Skript irgendwo ohne das Beating GUI liegt), soll das Fenster trotzdem
# starten - nur der Profil-Modus ist dann nicht waehlbar.
try:
    import beating_profil as BP
    HAT_PROFIL = True
except Exception as _exc:                     # noqa: BLE001
    BP = None
    HAT_PROFIL = False
    PROFIL_FEHLER = str(_exc)


# Where the PDFs go, resolved relative to THIS FILE rather than to the working
# directory (which depends on how the script was launched).
#
# Project convention is GUI/Bilder - but only when this script actually sits
# somewhere under GUI/.  As a standalone project the parent is the folder that
# holds ALL projects, and writing a Bilder/ there would be plainly wrong, so in
# that case the images stay next to the script.
_HERE = FilePath(__file__).resolve().parent
OUT_DIR_CANDIDATES = ([_HERE.parent / "Bilder"] if _HERE.parent.name.lower() == "gui"
                      else []) + [_HERE / "Bilder"]


def _resolve_out_dir():
    for cand in OUT_DIR_CANDIDATES:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except Exception:
            continue
    fb = FilePath.cwd() / "Bilder"
    fb.mkdir(parents=True, exist_ok=True)
    return fb


def _json_default(o):
    """Fallback fuer json.dumps.

    Der Zustand enthaelt numpy-Arrays (die Tonphasen). Ein blosses
    default=float wirft darauf ("only size-1 arrays can be converted"), und die
    Ausnahme landete frueher in einem MODALEN Dialog - das Fenster blieb
    stehen, bis jemand klickt."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return float(o)


def short_name(path, keep=34):
    name = str(getattr(path, "name", path))
    if len(name) <= keep:
        return name
    head = keep // 2 - 2
    return name[:head] + "..." + name[-(keep - head - 3):]


M_CHOICES = [("Uhrenzustand $m_F = 0$", 0), ("$m_F = +1$", 1), ("$m_F = -1$", -1),
             ("$m_F = +2$", 2), ("$m_F = -2$", -2),
             ("thermisch (alle $m_F$ gleich besetzt)", None)]

PANELS = ["Zeeman-Spektrum (Anregung über Zweiphotonen-Verstimmung)",
          "Verstimmungsbudget: eta, Kontrastdeckel, Streuung",
          "Beiträge der Zwischenzustände zu Omega",
          "Leistung: Omega/2pi und t_pi",
          "m_F-Vergleich",
          "Multiton-Profil: Karte und I(t)"]


class RabiRb85Window(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rabi Rb85 GUI - Zweiphotonen-Raman, F=2 <-> F=3")
        self.resize(1500, 880)

        self.state = {
            "delta": -8.0e9,        # Hz, from the 5P_1/2 centroid, negative = red
            "beta": 0.5,            # power fraction in the F=2 leg
            "q": +1,                # sigma+
            "power": 1.0e-6,        # W
            "waist": 1.1e-6,        # m
            "drive_mode": "rabi",   # 'power' | 'intensity' | 'rabi'
            "intensity": 1.75e4,    # W/m^2
            "f_rabi": 0.2e6,        # Hz, only in 'rabi' mode
            "m": 0,                 # None = thermal
            "B": 0.0,               # G
            # Multiton-Profil (nur im Modus "Profil")
            "p_N_x": 3, "p_N_y": 4,
            "p_waist": 1.10e-6,
            "p_width_x": 0.45e6, "p_width_y": 0.45e6,
            "p_r_x": 1.0, "p_r_y": 1.2,
            "p_f1": 75e-3, "p_f2": 750e-3, "p_fLO": 52.88e-3,
            "p_offset": 100e6,
            "p_airy": True, "p_airy_factor": 1.4830,
            "p_radius": 1.0e-6,
            "p_leistung": 222e-9,     # W je Profil
            "p_t0": 0.0,              # s, Pulsstart im Beat-Zyklus
            "p_phase_x": None, "p_phase_y": None,
            "t_max": 7.0e-6,        # s
            "n_t": 1200,
            "with_shift": True,
            "compensate": False,
            "with_scatter": True,
            "show_ideal": True,
            "title_in_plot": True,
            "auto_update": True,
        }
        self.out_dir = _resolve_out_dir()
        self.cache = {}
        self._building = True
        self._build_ui()
        self._building = False
        self.recompute()

    # ============================================================ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.fig = Figure(figsize=(11, 7.5))
        self.fig.set_constrained_layout(True)
        self.canvas = FigureCanvas(self.fig)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
        self.ax_rabi = self.fig.add_subplot(gs[0, 0])
        self.ax_aux = self.fig.add_subplot(gs[0, 1])
        splitter.addWidget(self.canvas)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setAlignment(Qt.AlignTop)
        pl.addWidget(self._group_raman())
        pl.addWidget(self._group_beam())
        pl.addWidget(self._group_profil())
        pl.addWidget(self._group_atom())
        pl.addWidget(self._group_pulse())
        pl.addWidget(self._group_options())
        pl.addWidget(self._group_actions())
        pl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(390)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1060, 440])

    # -- helpers, same shape as in Beating_Multitone_GUI ---------
    def _dspin(self, value, lo, hi, dec, step, suffix=""):
        w = QDoubleSpinBox()
        w.setDecimals(dec); w.setRange(lo, hi); w.setSingleStep(step)
        w.setValue(value); w.setKeyboardTracking(False)
        if suffix:
            w.setSuffix(" " + suffix)
        w.valueChanged.connect(self._on_param_changed)
        return w

    def _ispin(self, value, lo, hi, suffix=""):
        w = QSpinBox()
        w.setRange(lo, hi); w.setValue(value); w.setKeyboardTracking(False)
        if suffix:
            w.setSuffix(" " + suffix)
        w.valueChanged.connect(self._on_param_changed)
        return w

    def _group_raman(self):
        g = QGroupBox("Raman")
        lay = QGridLayout(g)
        self.sp_delta = self._dspin(self.state["delta"] / 1e9, -2000.0, -3.2,
                                    3, 0.5, "GHz")
        self.sp_delta.setToolTip(
            "Verstimmung des F=2-Zweigs von der 5P_1/2-Zentroide.\n"
            "Negativ = rot. Sie legt gleichzeitig zwei Dinge fest:\n"
            "  eta ~ w_hfs/|Delta|  (Kontrastdeckel)\n"
            "  Streuung ~ pi*Gamma/|Delta|  (Verlust)\n"
            "Beide gegenläufig - siehe Panel 'Verstimmungsbudget'.\n\n"
            "Unterhalb von |Delta| = 3.036 GHz wechselt eta das Vorzeichen,\n"
            "weil der jeweils andere Zweig dann auf der anderen Seite der\n"
            "Resonanz liegt. Deshalb ist der Bereich hier begrenzt.")
        lay.addWidget(QLabel("Verstimmung Delta"), 0, 0)
        lay.addWidget(self.sp_delta, 0, 1)

        self.sp_beta = self._dspin(self.state["beta"], 0.02, 0.98, 3, 0.05)
        self.sp_beta.setToolTip(
            "Anteil der optischen Leistung im Zweig, der F=2 adressiert.\n"
            "Omega ~ sqrt(beta(1-beta)) ist bei 0.5 maximal.\n"
            "Ein 'magisches' beta, das eta zu Null macht, gibt es NICHT:\n"
            "beide Beiträge zum differentiellen Shift haben dasselbe\n"
            "Vorzeichen. Das Optimum liegt bei beta ~ 0.31 und bringt nur\n"
            "wenige Prozent.")
        lay.addWidget(QLabel("Leistungsanteil beta"), 1, 0)
        lay.addWidget(self.sp_beta, 1, 1)

        self.cmb_q = QComboBox()
        self.cmb_q.addItems(["sigma+ auf beiden Zweigen",
                             "sigma- auf beiden Zweigen"])
        self.cmb_q.setToolTip(
            "Nur der VEKTOR-Anteil des AC-Stark-Operators kann einen\n"
            "Hyperfeinübergang treiben, und der geht wie eps_1* x eps_2.\n"
            "Für zwei LINEAR polarisierte Zweige ist das exakt Null -\n"
            "der Übergang wird dann gar nicht getrieben.\n"
            "Gleichsinnig zirkular gibt eps* x eps = z und treibt Delta m = 0.")
        self.cmb_q.currentIndexChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("Polarisation"), 2, 0)
        lay.addWidget(self.cmb_q, 2, 1)
        return g

    def _group_beam(self):
        g = QGroupBox("Strahl")
        lay = QGridLayout(g)
        self.cmb_drive = QComboBox()
        self.cmb_drive.addItems(["Leistung + Waist", "Intensität direkt",
                                 "Rabi-Frequenz vorgeben",
                                 "Multiton-Profil (Leistung je Profil)"])
        if not HAT_PROFIL:
            self.cmb_drive.model().item(3).setEnabled(False)
        self.cmb_drive.setCurrentIndex(
            {"power": 0, "intensity": 1, "rabi": 2}[self.state["drive_mode"]])
        self.cmb_drive.setToolTip(
            "'Rabi-Frequenz vorgeben' rechnet rückwärts: es zeigt, welche\n"
            "Intensität und Leistung nötig wären. Nützlich, um zu sehen,\n"
            "dass Leistung bei diesem Übergang praktisch nie die\n"
            "Randbedingung ist.")
        self.cmb_drive.currentIndexChanged.connect(self._on_drive_changed)
        lay.addWidget(QLabel("Vorgabe"), 0, 0); lay.addWidget(self.cmb_drive, 0, 1)

        self.sp_power = self._dspin(self.state["power"] * 1e6, 1e-4, 1e5, 4,
                                    0.1, "uW")
        lay.addWidget(QLabel("Leistung"), 1, 0); lay.addWidget(self.sp_power, 1, 1)
        self.sp_waist = self._dspin(self.state["waist"] * 1e6, 0.1, 5000.0, 3,
                                    0.1, "um")
        self.sp_waist.setToolTip("1/e^2-Radius des Feldes. I0 = 2P/(pi w^2).")
        lay.addWidget(QLabel("Waist"), 2, 0); lay.addWidget(self.sp_waist, 2, 1)
        self.sp_int = self._dspin(self.state["intensity"] / 1e4, 1e-6, 1e6, 5,
                                  0.1, "W/cm2")
        lay.addWidget(QLabel("Intensität"), 3, 0); lay.addWidget(self.sp_int, 3, 1)
        self.sp_frabi = self._dspin(self.state["f_rabi"] / 1e6, 1e-4, 50.0, 4,
                                    0.05, "MHz")
        lay.addWidget(QLabel("f_Rabi (Vorgabe)"), 4, 0)
        lay.addWidget(self.sp_frabi, 4, 1)
        self._sync_drive_enabled()
        return g


    def _group_profil(self):
        """Alle Parameter des Multiton-Profils - dieselben Namen wie im
        Beating GUI, damit ein Arbeitspunkt 1:1 uebertragbar ist."""
        g = QGroupBox("Multiton-Profil")
        self.grp_profil = g
        lay = QGridLayout(g)
        st = self.state
        r = 0

        def zeile(label, w, tip=None):
            nonlocal r
            if tip:
                w.setToolTip(tip)
            lay.addWidget(QLabel(label), r, 0); lay.addWidget(w, r, 1)
            r += 1
            return w

        self.sp_pNx = zeile("N_x", self._ispin(st["p_N_x"], 1, 60))
        self.sp_pNy = zeile("N_y", self._ispin(st["p_N_y"], 1, 60))
        self.cmb_pprof = QComboBox(); self.cmb_pprof.addItems(["Gauss", "Airy"])
        self.cmb_pprof.setCurrentIndex(1 if st["p_airy"] else 0)
        self.cmb_pprof.currentIndexChanged.connect(self._on_param_changed)
        zeile("Profil", self.cmb_pprof)
        self.sp_pairy = zeile("Airy-Faktor",
                              self._dspin(st["p_airy_factor"], 0.1, 5.0, 4, 0.01),
                              "erster Nullring = Faktor x Waist")
        self.sp_pwaist = zeile("waist", self._dspin(st["p_waist"] * 1e6, 0.05, 500.0,
                                                    3, 0.05, "um"))
        self.sp_pwx = zeile("width_x", self._dspin(st["p_width_x"] / 1e6, 0.0, 36.0,
                                                   4, 0.05, "MHz"))
        self.sp_pwy = zeile("width_y", self._dspin(st["p_width_y"] / 1e6, 0.0, 36.0,
                                                   4, 0.05, "MHz"))
        self.sp_prx = zeile("r_x", self._dspin(st["p_r_x"], 0.0, 5.0, 3, 0.05))
        self.sp_pry = zeile("r_y", self._dspin(st["p_r_y"], 0.0, 5.0, 3, 0.05))
        self.sp_pf1 = zeile("f1", self._dspin(st["p_f1"] * 1e3, 1.0, 5000.0, 3, 5.0, "mm"))
        self.sp_pf2 = zeile("f2", self._dspin(st["p_f2"] * 1e3, 1.0, 5000.0, 3, 5.0, "mm"))
        self.sp_pflo = zeile("fLO", self._dspin(st["p_fLO"] * 1e3, 1.0, 5000.0, 3, 1.0, "mm"),
                             "Fokussierlinse. Im Beating GUI eine Konstante\n"
                             "(fLO in kern/beating_physik.py);\n"
                             "hier einstellbar, weil sie die Spotgroesse und den\n"
                             "Ablenkbereich zugleich setzt.")
        self.sp_poff = zeile("Offset", self._dspin(st["p_offset"] / 1e6, 0.0, 1000.0,
                                                   3, 1.0, "MHz"),
                             "faellt aus jeder Differenzfrequenz heraus - er\n"
                             "verschiebt die Geometrie, nicht das Beating.")
        self.sp_prad = zeile("Auswerteradius",
                             self._dspin(st["p_radius"] * 1e6, 0.05, 50.0, 3, 0.1, "um"),
                             "Kreis um die Profilmitte, ueber den gemittelt wird.\n"
                             "Das ist die empfindlichste Einstellung ueberhaupt:\n"
                             "waehlt man ihn groesser als das Plateau, dominiert\n"
                             "die Flanke die Uniformity und damit den Kontrast.")
        self.sp_pP = zeile("Leistung je Profil",
                           self._dspin(st["p_leistung"] * 1e9, 1e-3, 1e9, 4, 50.0, "nW"),
                           "Gesamtleistung IN DIESEM Profil, nicht vor dem AOD.\n"
                           "P_Profil = P_vor_AOD * Wirkungsgrad / N_gleichzeitig")

        self.cmb_pphase = QComboBox()
        self.cmb_pphase.addItems(["Tonphasen alle 0", "Schroeder", "zufällig"])
        self.cmb_pphase.setToolTip(
            "Nur die N_x + N_y TONphasen sind einstellbar; ein Spot (n,m)\n"
            "traegt phi_x(n) + phi_y(m). Bei allen 0 rephasieren alle Toene\n"
            "einmal je Grundperiode zu einem Kamm - dort ist die Pulsflaeche\n"
            "ein Vielfaches. Schroeder-Phasen brechen das auf.")
        self.cmb_pphase.currentIndexChanged.connect(self._on_param_changed)
        zeile("Tonphasen", self.cmb_pphase)

        self.sp_pt0 = zeile("Pulsstart t_0",
                            self._dspin(st["p_t0"] * 1e6, 0.0, 1000.0, 4, 0.1, "us"),
                            "Startzeitpunkt IM Beat-Zyklus. Die Pulsflaeche haengt\n"
                            "stark davon ab - ohne festen Trigger streut sie von\n"
                            "Schuss zu Schuss um Groessenordnungen.")
        self.btn_pt0 = QPushButton("besten t_0 suchen")
        self.btn_pt0.setToolTip(
            "Sucht die gleichmaessigste Pulsflaeche - aber nur unter den\n"
            "Startzeiten, deren Flaeche nahe am Median liegt. Ohne diese\n"
            "Einschraenkung landet die Suche auf der Flanke des Kamms, wo die\n"
            "Flaeche ein Vielfaches ist und der 'pi-Puls' keiner mehr ist.")
        self.btn_pt0.clicked.connect(self._on_bester_t0)
        lay.addWidget(self.btn_pt0, r, 0, 1, 2); r += 1

        self.lbl_profil = QLabel("-")
        self.lbl_profil.setWordWrap(True)
        self.lbl_profil.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_profil, r, 0, 1, 2)
        if not HAT_PROFIL:
            self.lbl_profil.setText("kern/beating_physik.py nicht ladbar:\n"
                                    + PROFIL_FEHLER)
        g.setEnabled(False)
        return g

    def _profil_wp(self):
        """Zustand -> Parametersatz fuer beating_profil."""
        st = self.state
        wp = dict(BP.WP)
        wp.update(N_x=st["p_N_x"], N_y=st["p_N_y"], use_airy=st["p_airy"],
                  airy_factor=st["p_airy_factor"], waist=st["p_waist"],
                  width_x=st["p_width_x"], width_y=st["p_width_y"],
                  r_x=st["p_r_x"], r_y=st["p_r_y"], offset=st["p_offset"],
                  f1=st["p_f1"], f2=st["p_f2"], fLO=st["p_fLO"],
                  radius=st["p_radius"], m=st["m"] or 0, B_gauss=st["B"],
                  delta=st["delta"],
                  phase_x=st["p_phase_x"], phase_y=st["p_phase_y"])
        return wp

    def _on_bester_t0(self):
        c = self.cache
        if not c.get("prof"):
            return
        self.lbl_status.setText("suche t_0 ...")
        QApplication.processEvents()
        t0s, A, U = BP.flaeche_ueber_t0(c["prof"], c["t_pi"], n=100,
                                        skala=c["skala"], raman=c["raman"])
        med = np.median(A)
        ok = np.flatnonzero(np.abs(A - med) < 0.25 * med)
        i = int(ok[np.argmin(U[ok])]) if ok.size else int(np.argmin(U))
        self.sp_pt0.setValue(float(t0s[i]) * 1e6)
        self.lbl_status.setText(f"t_0 = {t0s[i]*1e6:.3f} us, U(Fläche) = {U[i]*100:.1f} %")

    def _group_atom(self):
        g = QGroupBox("Atom")
        lay = QGridLayout(g)
        self.cmb_m = QComboBox()
        self.cmb_m.addItems([n for n, _ in M_CHOICES])
        self.cmb_m.setToolTip(
            "Rb-85 hat I = 5/2, also 5 Unterzustände in F=2 und 7 in F=3.\n"
            "Für Delta m = 0 koppeln nur m = -2 ... +2; m = +/-3 in F=3\n"
            "haben keinen Partner und sind dunkel.\n\n"
            "Die relativen Kopplungen (aus der vollen ARC-Summe) sind\n"
            "ungefähr 0.76 : 0.95 : 1 : 0.94 : 0.73 - ohne Pumpen flopt\n"
            "jedes m_F mit eigener Frequenz, und die Ensemble-Kurve dämpft\n"
            "QUASIPERIODISCH mit Wiederbelebungen. Das unterscheidet sie von\n"
            "einer Dämpfung durch Intensitätsstreuung, die monoton ist.")
        self.cmb_m.currentIndexChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("Unterzustand"), 0, 0); lay.addWidget(self.cmb_m, 0, 1)

        self.sp_B = self._dspin(self.state["B"], 0.0, 200.0, 3, 0.5, "G")
        self.sp_B.setToolTip(
            "Bias-Feld. Für Delta m = 0 verschiebt es den Übergang um\n"
            "(g_3 - g_2) m = 0.933 MHz/G * m - linear in m, also NICHT für\n"
            "den Uhrenzustand. Der spürt nur die gemessene quadratische\n"
            "Verschiebung, 1296.8 Hz/G^2.\n\n"
            "Praktischer Nutzen: schon wenige Gauss schieben alle m =/= 0\n"
            "weit aus der Resonanz, und der Uhrenübergang steht spektral\n"
            "allein da. Dann verschwindet die Clebsch-Gordan-Dämpfung und\n"
            "übrig bleibt nur ein Kontrast-SOCKEL von 1/5 durch die nicht\n"
            "adressierte Population - im Fit klar unterscheidbar.")
        lay.addWidget(QLabel("Bias-Feld B"), 1, 0); lay.addWidget(self.sp_B, 1, 1)
        return g

    def _group_pulse(self):
        g = QGroupBox("Puls")
        lay = QGridLayout(g)
        self.sp_tmax = self._dspin(self.state["t_max"] * 1e6, 0.01, 5000.0, 3,
                                   1.0, "us")
        lay.addWidget(QLabel("max. Pulslänge"), 0, 0); lay.addWidget(self.sp_tmax, 0, 1)
        self.sp_nt = self._ispin(self.state["n_t"], 100, 20000)
        lay.addWidget(QLabel("Punkte"), 1, 0); lay.addWidget(self.sp_nt, 1, 1)
        self.btn_auto_t = QPushButton("Auf 3 pi-Pulse setzen")
        self.btn_auto_t.clicked.connect(self._on_auto_t)
        lay.addWidget(self.btn_auto_t, 2, 0, 1, 2)
        return g

    def _group_options(self):
        g = QGroupBox("Optionen")
        lay = QVBoxLayout(g)
        self.cb_shift = QCheckBox("differentiellen Lichtshift einrechnen")
        self.cb_shift.setChecked(self.state["with_shift"])
        self.cb_shift.setToolTip(
            "delta_LS = eta * Omega mit ortsunabhängigem eta. Deckelt den\n"
            "Kontrast auf 1/(1+eta^2) und beschleunigt die Rotation um\n"
            "sqrt(1+eta^2). Ausschalten zeigt, wie viel er wirklich kostet.")
        self.cb_shift.stateChanged.connect(self._on_param_changed)
        lay.addWidget(self.cb_shift)

        self.cb_comp = QCheckBox("Lichtshift durch Verstimmung kompensieren")
        self.cb_comp.setChecked(self.state["compensate"])
        self.cb_comp.setToolTip(
            "Setzt die Zweiphotonen-Verstimmung auf -delta_LS. Bei KONSTANTER\n"
            "Intensität ist das perfekt und holt den vollen Kontrast zurück -\n"
            "das ist der billigste Gewinn in diesem ganzen Fenster.\n"
            "Bei einem strukturierten Profil kompensiert es nur den Mittelwert.")
        self.cb_comp.stateChanged.connect(self._on_param_changed)
        lay.addWidget(self.cb_comp)

        self.cb_scatter = QCheckBox("spontane Streuung einrechnen")
        self.cb_scatter.setChecked(self.state["with_scatter"])
        self.cb_scatter.setToolTip(
            "Als Überlebensfaktor exp(-Gamma t). Streuung ist VERLUST, nicht\n"
            "Kontrastverlust: sie holt Population aus dem Zwei-Niveau-System\n"
            "heraus (auch in die dunklen m = +/-3). Im Fit ist das ein\n"
            "abfallender Umschlag über alle Kurven, keine Dämpfung der\n"
            "Oszillation - deshalb lohnt es, sie getrennt zeigen zu können.")
        self.cb_scatter.stateChanged.connect(self._on_param_changed)
        lay.addWidget(self.cb_scatter)

        self.cb_ideal = QCheckBox("ideale Kurve (eta = 0) mitzeichnen")
        self.cb_ideal.setChecked(self.state["show_ideal"])
        self.cb_ideal.stateChanged.connect(self._on_param_changed)
        lay.addWidget(self.cb_ideal)
        return g

    def _group_actions(self):
        g = QGroupBox("Actions")
        lay = QVBoxLayout(g)
        self.cb_auto = QCheckBox("automatisch neu rechnen")
        self.cb_auto.setChecked(self.state["auto_update"])
        self.cb_auto.stateChanged.connect(self._on_param_changed)
        lay.addWidget(self.cb_auto)
        self.btn_update = QPushButton("Neu rechnen")
        self.btn_update.clicked.connect(lambda: self.recompute())
        lay.addWidget(self.btn_update)

        lay.addWidget(QLabel("Panel rechts"))
        self.cmb_panel = QComboBox()
        self.cmb_panel.addItems(PANELS)
        self.cmb_panel.currentIndexChanged.connect(lambda _: self.draw())
        lay.addWidget(self.cmb_panel)

        self.cb_title = QCheckBox("Parameter als Titel ins Bild")
        self.cb_title.setChecked(self.state["title_in_plot"])
        self.cb_title.stateChanged.connect(self._on_param_changed)
        lay.addWidget(self.cb_title)

        self.btn_save = QPushButton("Als PDF speichern")
        self.btn_save.setToolTip(
            "Vektor-PDF nach GUI/Bilder, plus eine .json mit allen Parametern\n"
            "unter demselben Namen - damit ein Bild später rekonstruierbar ist.")
        self.btn_save.clicked.connect(self._on_save_clicked)
        lay.addWidget(self.btn_save)

        self.lbl_info = QLabel("-")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color: #333; font-size: 10px;")
        lay.addWidget(self.lbl_info)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_status)
        return g

    # ======================================================== plumbing
    def _sync_drive_enabled(self):
        mode = self.cmb_drive.currentIndex()
        self.sp_power.setEnabled(mode == 0)
        self.sp_waist.setEnabled(mode in (0, 2))
        self.sp_int.setEnabled(mode == 1)
        self.sp_frabi.setEnabled(mode == 2)
        if hasattr(self, "grp_profil"):
            self.grp_profil.setEnabled(mode == 3)

    def _on_drive_changed(self, _):
        self._sync_drive_enabled()
        self._on_param_changed()

    def _on_param_changed(self, *_):
        if self._building:
            return
        if self.cb_auto.isChecked():
            self.recompute()

    def _on_auto_t(self):
        c = self.cache
        if c.get("t_pi"):
            self.sp_tmax.setValue(3.0 * c["t_pi"] * 1e6)

    def _read_widgets(self):
        s = self.state
        s["delta"] = self.sp_delta.value() * 1e9
        s["beta"] = self.sp_beta.value()
        s["q"] = +1 if self.cmb_q.currentIndex() == 0 else -1
        s["drive_mode"] = ["power", "intensity", "rabi",
                           "profil"][self.cmb_drive.currentIndex()]
        s["power"] = self.sp_power.value() * 1e-6
        s["waist"] = self.sp_waist.value() * 1e-6
        s["intensity"] = self.sp_int.value() * 1e4
        s["f_rabi"] = self.sp_frabi.value() * 1e6
        s["m"] = M_CHOICES[self.cmb_m.currentIndex()][1]
        s["B"] = self.sp_B.value()
        if hasattr(self, "sp_pNx"):
            s["p_N_x"] = self.sp_pNx.value(); s["p_N_y"] = self.sp_pNy.value()
            s["p_airy"] = self.cmb_pprof.currentIndex() == 1
            s["p_airy_factor"] = self.sp_pairy.value()
            s["p_waist"] = self.sp_pwaist.value() * 1e-6
            s["p_width_x"] = self.sp_pwx.value() * 1e6
            s["p_width_y"] = self.sp_pwy.value() * 1e6
            s["p_r_x"] = self.sp_prx.value(); s["p_r_y"] = self.sp_pry.value()
            s["p_f1"] = self.sp_pf1.value() * 1e-3
            s["p_f2"] = self.sp_pf2.value() * 1e-3
            s["p_fLO"] = self.sp_pflo.value() * 1e-3
            s["p_offset"] = self.sp_poff.value() * 1e6
            s["p_radius"] = self.sp_prad.value() * 1e-6
            s["p_leistung"] = self.sp_pP.value() * 1e-9
            s["p_t0"] = self.sp_pt0.value() * 1e-6
            art = ["null", "schroeder", "zufall"][self.cmb_pphase.currentIndex()]
            s["p_phase_x"] = BP.phasen(art, s["p_N_x"]) if BP else None
            s["p_phase_y"] = BP.phasen(art, s["p_N_y"]) if BP else None
        s["t_max"] = self.sp_tmax.value() * 1e-6
        s["n_t"] = self.sp_nt.value()
        s["with_shift"] = self.cb_shift.isChecked()
        s["compensate"] = self.cb_comp.isChecked()
        s["with_scatter"] = self.cb_scatter.isChecked()
        s["show_ideal"] = self.cb_ideal.isChecked()
        s["title_in_plot"] = self.cb_title.isChecked()
        return s

    # ======================================================== physics
    def recompute(self):
        s = self._read_widgets()
        raman = R.RamanRb85(s["delta"], s["beta"], s["q"])

        prof = skala = None
        if s["drive_mode"] == "profil":
            self.lbl_status.setText("baue Profil ...")
            QApplication.processEvents()
            prof = BP.profil(self._profil_wp())
            skala = BP.kalibrieren_auf_leistung(prof, s["p_leistung"])
            tt = np.linspace(0, prof["T0"], 2001)
            # I ist hier der ORTS- UND ZEITMITTELWERT in der Region - nur fuer
            # die Kennzahlen. Die Kurve selbst wird pixelweise gerechnet.
            I = float(BP._I_roh(prof, tt).mean() * skala)
            power = s["p_leistung"]
        elif s["drive_mode"] == "power":
            I = R.gaussian_peak_intensity(s["power"], s["waist"])
            power = R.power_for_intensity(I, s["waist"])
        elif s["drive_mode"] == "intensity":
            I = s["intensity"]
            power = R.power_for_intensity(I, s["waist"])
        else:
            I = raman.intensity_for_rabi(s["f_rabi"], 0)
            power = R.power_for_intensity(I, s["waist"])

        t = np.linspace(0.0, s["t_max"], s["n_t"])
        kw = dict(B_gauss=s["B"], compensate=s["compensate"],
                  with_shift=s["with_shift"], with_scatter=s["with_scatter"])

        def one(m):
            if prof is not None:
                # Strukturiertes Profil: jeder Ort hat sein eigenes I(t), also
                # wird pixelweise propagiert und erst danach gemittelt.
                _, I_t = BP.intensitaet(prof, s["p_t0"], s["t_max"],
                                        n_t=s["n_t"], skala=skala, raman=raman)
                return R.excitation_series_multi(t, I_t, raman, m=m, **kw)
            # Konstantes I: geschlossene Form, ausser ein statischer
            # Zeeman-Term bricht die Vertauschbarkeit.
            if abs(R.zeeman_detuning(m, s["B"])) > 0:
                return R.excitation_series(t, np.full_like(t, I), raman, m=m, **kw)
            return R.excitation_constant(t, I, raman, m=m, **kw)

        if s["m"] is None:
            P = R.thermal_average(one, raman)
            m_ref = 0
        else:
            P = one(s["m"])
            m_ref = s["m"]

        cm = raman.coeff(m_ref)
        Om = cm["C_rabi"] * I
        eta = cm["eta"] if s["with_shift"] else 0.0
        if s["compensate"]:
            eta = 0.0
        g = np.sqrt(1 + eta ** 2)
        t_pi = np.pi / (Om * g) if Om > 0 else np.nan
        P_ideal = np.sin(Om * t / 2) ** 2
        Gam = 0.5 * (cm["C_scatter_2"] + cm["C_scatter_3"]) * I

        self.cache = dict(raman=raman, t=t, P=P, P_ideal=P_ideal, I=I,
                          power=power, Om=Om, eta=eta, t_pi=t_pi, Gam=Gam,
                          cm=cm, m_ref=m_ref, s=dict(s), prof=prof, skala=skala)
        self.lbl_status.setText("")
        self._update_info()
        self._update_profil_info()
        self.draw()

    def _update_info(self):
        c, s = self.cache, self.state
        cm, r = c["cm"], c["raman"]
        cg = r.cg()
        pk = c["P"].max()
        self.lbl_info.setText(
            f"Omega/2pi = {c['Om'] / 2 / np.pi / 1e3:.2f} kHz   |   "
            f"t_pi = {c['t_pi'] * 1e6:.4f} us\n"
            f"I = {c['I'] / 1e4:.4g} W/cm2   |   P = {c['power'] * 1e6:.4g} uW "
            f"bei w = {s['waist'] * 1e6:.2f} um\n"
            f"eta = {cm['eta']:+.4f}  ->  Kontrastdeckel "
            f"{r.contrast_cap(c['m_ref']) * 100:.2f} %"
            + ("  (kompensiert)" if s["compensate"] else "") + "\n"
            f"Streuung: {c['Gam'] * c['t_pi'] * 100:.4f} % pro pi-Puls   |   "
            f"1/Gamma = {1 / c['Gam'] * 1e6:.1f} us\n"
            f"gemeinsamer Lichtshift: "
            f"{cm['C_shift_common'] * c['I'] / 2 / np.pi / 1e3:+.1f} kHz\n"
            f"Zeeman bei {s['B']:.2f} G: "
            f"{R.zeeman_detuning(c['m_ref'], s['B']) / 2 / np.pi / 1e3:+.2f} kHz\n"
            f"CG-Gewichte: " + ", ".join(f"{m:+d}:{cg[m]:.3f}" for m in (-2, -1, 0, 1, 2))
            + "\neta je m_F: " + ", ".join(f"{m:+d}:{r.coeff(m)['eta']:+.2f}"
                                           for m in (-2, -1, 0, 1, 2))
            + f"\nerreichte Anregung: {pk * 100:.2f} %")

    def _update_profil_info(self):
        c = self.cache
        prof = c.get("prof")
        if prof is None:
            return
        T0 = prof["T0"]
        A_eff = c["power"] / c["I"] if c["I"] > 0 else float("nan")
        verh = c["t_pi"] / T0
        if not prof["periodisch"]:
            self.lbl_profil.setText(
                f"ein einziger Ton: kein Beating, I konstant\n"
                f"{prof['maske'].sum()} Pixel, gerechnet {prof['idx'].size}\n"
                f"effektive Flaeche {A_eff*1e12:.2f} um^2, "
                f"<I> = {c['I']/1e4:.4g} W/cm^2")
            return
        wie = ("Beating eingefroren - der Puls mittelt NICHTS weg" if verh < 0.02
               else "Puls mittelt teilweise" if verh < 0.5
               else "Puls mittelt ueber mehrere Beat-Perioden")
        self.lbl_profil.setText(
            f"T_0 = {T0*1e6:.3f} us (f_0 = {prof['f0']/1e3:.2f} kHz), "
            f"{prof['maske'].sum()} Pixel, gerechnet {prof['idx'].size}\n"
            f"effektive Flaeche {A_eff*1e12:.2f} um^2, "
            f"<I> = {c['I']/1e4:.4g} W/cm^2\n"
            f"t_pi/T_0 = {verh:.2e}  ->  {wie}\n"
            + (f"{len(prof['degen'])} frequenzentartete Paare\n"
               if prof["degen"] else "")
            + (f"hoechste Beat-Linie {prof['beats'][-1]/1e3:.0f} kHz "
               f"= {prof['beats'][-1]/prof['f0']:.0f} x f_0"
               if prof["periodisch"] else
               "ein einziger Ton - kein Beating, I ist konstant"))

    # ======================================================== drawing
    def draw(self):
        c = self.cache
        if not c:
            return
        s = c["s"]
        # clear() keeps the scale a previous panel set, and a log x-axis
        # then swallows the next panel silently - reset both explicitly.
        # Eine Colorbar bringt eine EIGENE Achse mit. ax.clear() entfernt sie
        # nicht, und wenn man nur die Achse loescht, zeigt das Colorbar-Objekt
        # ins Leere und der naechste draw() wirft. Also erst das Objekt loesen.
        if getattr(self, "_cbar", None) is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self._drop_twins()
        for ax in (self.ax_rabi, self.ax_aux):
            ax.clear()
            ax.set_xscale("linear"); ax.set_yscale("linear")

        # ---- left: the Rabi oscillation itself ----
        a = self.ax_rabi
        t_us = c["t"] * 1e6
        if s["show_ideal"]:
            a.plot(t_us, c["P_ideal"] * 100, color="#8a8a8a", ls="--", lw=1.2,
                   label="ideal (eta = 0, keine Streuung)")
        lab = dict(M_CHOICES)  # name -> m
        mname = [n for n, v in M_CHOICES if v == s["m"]][0]
        a.plot(t_us, c["P"] * 100, color="#2f6f4e", lw=1.9, label=mname)
        if c["eta"] != 0:
            a.axhline(100 / (1 + c["eta"] ** 2), color="#bbb", lw=0.9)
            a.text(t_us[-1], 100 / (1 + c["eta"] ** 2), " Deckel", va="center",
                   fontsize=7, color="#888")
        if np.isfinite(c["t_pi"]):
            a.axvline(c["t_pi"] * 1e6, color="#b3452c", lw=0.9, ls=":")
            a.text(c["t_pi"] * 1e6, 2, "  $t_\\pi$", fontsize=8, color="#b3452c")
        a.set_xlabel("Pulslänge $t_p$ ($\\mu$s)")
        a.set_ylabel("Anregung (%)")
        a.set_ylim(0, 102)
        a.set_title("Rabi-Oszillation, $|F{=}2,m\\rangle \\leftrightarrow |F{=}3,m\\rangle$",
                    fontsize=9.5)
        a.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)

        {0: self._panel_zeeman, 1: self._panel_budget, 2: self._panel_parts,
         3: self._panel_power, 4: self._panel_mf,
         5: self._panel_profil}[self.cmb_panel.currentIndex()]()
        self.ax_aux.spines["top"].set_visible(False)
        self.ax_aux.spines["right"].set_visible(False)

        if s["title_in_plot"]:
            self.fig.suptitle(self._param_line(), fontsize=8.5)
        else:
            self.fig.suptitle("")
        self.canvas.draw_idle()

    def _drop_twins(self):
        """Remove twin axes created by earlier panels.

        ax.twinx() adds a NEW axes to the figure; ax.clear() does not remove
        it, so switching panels back and forth would otherwise stack ghost
        axes with stale ticks on top of each other."""
        keep = {self.ax_rabi, self.ax_aux}
        for ax in list(self.fig.axes):
            if ax not in keep:
                ax.remove()

    def _param_line(self):
        c, s = self.cache, self.state
        return (f"Rb-85 Zweiphotonen-Raman  |  $\\Delta$ = {s['delta']/1e9:g} GHz von "
                f"$5P_{{1/2}}$, $\\beta$ = {s['beta']:g}, "
                f"$\\sigma^{'+' if s['q'] > 0 else '-'}$  |  "
                f"I = {c['I']/1e4:.4g} W/cm$^2$, w = {s['waist']*1e6:.2f} $\\mu$m, "
                f"B = {s['B']:g} G  |  "
                f"$\\eta$ = {c['cm']['eta']:+.3f}, "
                f"$\\Omega/2\\pi$ = {c['Om']/2/np.pi/1e3:.1f} kHz")

    # -- panels --------------------------------------------------
    def _panel_zeeman(self):
        """Excitation versus a deliberate two-photon detuning, per m_F.

        This is the panel that answers the practical question: does my bias
        field separate the clock transition from its neighbours by enough that
        I can address it alone?"""
        a, c, s = self.ax_aux, self.cache, self.state
        r = c["raman"]
        dets = np.linspace(-4e6, 4e6, 1200)
        t_p = c["t_pi"]
        for m, col in zip((-2, -1, 0, 1, 2),
                          ["#9ecae1", "#4292c6", "#2f6f4e", "#b3452c", "#f4a582"]):
            cm = r.coeff(m)
            Om = cm["C_rabi"] * c["I"]
            de = 2 * np.pi * dets + R.zeeman_detuning(m, s["B"])
            if s["with_shift"] and not s["compensate"]:
                de = de + cm["C_shift"] * c["I"]
            W = np.hypot(Om, de)
            P = (Om / W) ** 2 * np.sin(W * t_p / 2) ** 2
            a.plot(dets / 1e6, P * 100, color=col, lw=1.4, label=f"$m_F={m:+d}$")
        a.set_xlabel("Zweiphotonen-Verstimmung (MHz)")
        a.set_ylabel("Anregung (%)")
        a.set_title(f"Zeeman-Spektrum bei B = {s['B']:g} G, $t_p = t_\\pi$",
                    fontsize=9)
        a.legend(fontsize=7, ncol=2)
        a.set_ylim(0, 102)

    def _panel_budget(self):
        a, c, s = self.ax_aux, self.cache, self.state
        Ds = np.linspace(-60e9, -3.5e9, 200)
        eta, cap, sc = [], [], []
        for D in Ds:
            r = R.RamanRb85(D, s["beta"], s["q"])
            cm = r.coeff(0)
            I = r.intensity_for_rabi(2e5, 0)
            eta.append(abs(cm["eta"])); cap.append(r.contrast_cap(0))
            sc.append(0.5 * (cm["C_scatter_2"] + cm["C_scatter_3"]) * I / (2 * 2e5))
        a.plot(Ds / 1e9, eta, color="#1b4f8a", lw=1.6, label="$|\\eta|$")
        a.plot(Ds / 1e9, cap, color="#2f6f4e", lw=1.6,
               label="Kontrastdeckel $1/(1+\\eta^2)$")
        a.axvline(s["delta"] / 1e9, color="#b3452c", lw=1.0, ls=":")
        a2 = a.twinx(); a2.spines["top"].set_visible(False)
        a2.plot(Ds / 1e9, np.array(sc) * 100, color="#b3452c", lw=1.3, ls="--")
        a2.set_yscale("log")
        a2.set_ylabel("Streuung pro $\\pi$-Puls (%)", color="#b3452c", fontsize=8)
        a2.tick_params(axis="y", colors="#b3452c", labelsize=7)
        a.set_xlabel("$\\Delta$ von der $D_1$-Zentroide (GHz)")
        a.set_ylabel("$|\\eta|$ , Kontrastdeckel")
        a.set_ylim(0, 1.15); a.legend(fontsize=7.5, loc="center right")
        a.set_title("Verstimmungsbudget bei $\\Omega/2\\pi$ = 200 kHz", fontsize=9)

    def _panel_parts(self):
        """Contribution of each intermediate hyperfine level to Omega.

        The bars alternate in sign - that interference is exactly what a
        'sum over F' with unsigned Clebsch-Gordan factors' would get wrong."""
        a, c = self.ax_aux, self.cache
        parts = c["cm"]["parts"]
        keys = list(parts.keys())
        vals = np.array([parts[k] for k in keys])
        tot = vals.sum()
        cols = ["#2f6f4e" if v > 0 else "#b3452c" for v in vals]
        pct = vals / abs(tot) * 100
        a.barh(range(len(keys)), pct, color=cols)
        # D2 sits 7.1 THz away while D1 is a few GHz away, so its terms are
        # orders of magnitude smaller and simply invisible on a linear axis.
        # symlog keeps both readable AND keeps the sign visible.
        a.set_xscale("symlog", linthresh=0.01)
        # The value goes into the tick label, not next to the bar: a negative
        # bar's annotation would otherwise sit on top of the axis labels.
        a.set_yticks(range(len(keys)))
        a.set_yticklabels([f"{k}   {p:+.3g}%" for k, p in zip(keys, pct)],
                          fontsize=7.5)
        a.axvline(0, color="#333", lw=0.8)
        a.set_xlabel("Beitrag zu $\\Omega$ (% der Summe, symlog)")
        same = bool(np.all(vals > 0) or np.all(vals < 0))
        a.set_title("Zwischenzustände einzeln"
                    + ("  -  hier alle gleichsinnig" if same
                       else "  -  Vorzeichen heben sich teilweise auf"),
                    fontsize=9)
        a.invert_yaxis()

    def _panel_power(self):
        a, c, s = self.ax_aux, self.cache, self.state
        P = np.logspace(-3, 3, 200) * 1e-6
        I = R.gaussian_peak_intensity(P, s["waist"])
        Om = c["cm"]["C_rabi"] * I
        g = np.sqrt(1 + (0.0 if s["compensate"] else c["cm"]["eta"]) ** 2)
        a.loglog(P * 1e6, Om / 2 / np.pi / 1e3, color="#1b4f8a", lw=1.6)
        a.set_xlabel("Leistung ($\\mu$W)")
        a.set_ylabel("$\\Omega/2\\pi$ (kHz)", color="#1b4f8a")
        a.tick_params(axis="y", colors="#1b4f8a")
        a2 = a.twinx(); a2.spines["top"].set_visible(False)
        a2.loglog(P * 1e6, np.pi / (Om * g) * 1e6, color="#b3452c", lw=1.4, ls="--")
        a2.set_ylabel("$t_\\pi$ ($\\mu$s)", color="#b3452c")
        a2.tick_params(axis="y", colors="#b3452c")
        a.axvline(c["power"] * 1e6, color="#888", lw=0.9, ls=":")
        a.set_title(f"Leistungsskalierung bei w = {s['waist']*1e6:.2f} $\\mu$m",
                    fontsize=9)

    def _panel_mf(self):
        a, c, s = self.ax_aux, self.cache, self.state
        r, t = c["raman"], c["t"]
        kw = dict(B_gauss=s["B"], compensate=s["compensate"],
                  with_shift=s["with_shift"], with_scatter=s["with_scatter"])
        for m, col in zip((-2, -1, 0, 1, 2),
                          ["#9ecae1", "#4292c6", "#2f6f4e", "#b3452c", "#f4a582"]):
            if abs(R.zeeman_detuning(m, s["B"])) > 0:
                P = R.excitation_series(t, np.full_like(t, c["I"]), r, m=m, **kw)
            else:
                P = R.excitation_constant(t, c["I"], r, m=m, **kw)
            a.plot(t * 1e6, P * 100, color=col, lw=1.2, label=f"$m_F={m:+d}$")
        Pth = R.thermal_average(
            lambda m: (R.excitation_series(t, np.full_like(t, c["I"]), r, m=m, **kw)
                       if abs(R.zeeman_detuning(m, s["B"])) > 0
                       else R.excitation_constant(t, c["I"], r, m=m, **kw)), r)
        a.plot(t * 1e6, Pth * 100, color="#1b1b1b", lw=2.0, ls=":",
               label="thermisch")
        a.set_xlabel("Pulslänge $t_p$ ($\\mu$s)")
        a.set_ylabel("Anregung (%)")
        a.set_ylim(0, 102); a.legend(fontsize=7, ncol=2)
        a.set_title("Die fünf Unterzustände einzeln", fontsize=9)
        # sigma+ light breaks the +m / -m symmetry: the vector light shift is
        # linear in m, so eta runs from +0.30 (m=+1) to -2.55 (m=-2) at
        # Delta = -8 GHz.  The mirror check eta_{sigma-}(m) = eta_{sigma+}(-m)
        # holds exactly, which is what says this is physics and not a bug.
        a.text(0.02, 0.02, "$\\sigma^+$ bricht die $\\pm m$-Symmetrie\n"
                           "(Vektor-Lichtshift $\\propto m$)",
               transform=a.transAxes, fontsize=6.5, color="#666", va="bottom")

    def _panel_profil(self):
        """Zeitmittel des Profils mit dem Auswertekreis."""
        a, c = self.ax_aux, self.cache
        prof = c.get("prof")
        if prof is None:
            a.text(0.5, 0.5, "nur im Modus 'Multiton-Profil'", ha="center",
                   va="center", transform=a.transAxes, fontsize=9)
            return
        t = np.linspace(0, prof["T0"], 401)
        alle = np.arange(prof["F"][0].size)
        Iavg = BP._I_roh(prof, t, alle).mean(0).reshape(prof["X"].shape)
        um = 1e6
        ext = [(prof["x"][0] - prof["rcx"]) * um, (prof["x"][-1] - prof["rcx"]) * um,
               (prof["y"][0] - prof["rcy"]) * um, (prof["y"][-1] - prof["rcy"]) * um]
        im = a.imshow(Iavg / Iavg.max(), origin="lower", extent=ext, cmap="magma")
        th = np.linspace(0, 2 * np.pi, 200)
        rr = self.state["p_radius"] * um
        a.plot(rr * np.cos(th), rr * np.sin(th), color="#7fd6ff", lw=1.2)
        lim = max(rr * 2.2, 2.0)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
        a.set_xlabel("$x$ ($\\mu$m)")
        a.set_ylabel("$y$ ($\\mu$m)")
        a.set_title("Zeitmittel des Profils, Auswertekreis", fontsize=9)
        self._cbar = self.fig.colorbar(im, ax=a, fraction=0.046)

    # ======================================================== export
    def _on_save_clicked(self):
        self.draw()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        s, c = self.state, self.cache
        mtag = "thermal" if s["m"] is None else "m%+d" % s["m"]
        base = (f"RabiRb85_D{s['delta']/1e9:+.2f}GHz_"
                f"I{c['I']/1e4:.4g}Wcm2_B{s['B']:g}G_{mtag}_{stamp}")
        pdf = self.out_dir / (base + ".pdf")
        js = self.out_dir / (base + ".json")
        try:
            # KEIN bbox_inches="tight": die Figure laeuft mit
            # constrained_layout, und beides zusammen laesst matplotlib bei
            # einer Colorbar im Bild in eine Endlosschleife laufen.
            # constrained_layout setzt den Rand ohnehin schon eng.
            self.fig.savefig(pdf, format="pdf")
            js.write_text(json.dumps(
                {**{k: v for k, v in s.items()},
                 "derived": dict(intensity_W_m2=c["I"], power_W=c["power"],
                                 Omega_rad_s=c["Om"], eta=c["cm"]["eta"],
                                 t_pi_s=c["t_pi"], Gamma_scatter_1_s=c["Gam"],
                                 contrast_cap=c["raman"].contrast_cap(c["m_ref"]),
                                 cg_ratios={str(k): v for k, v
                                            in c["raman"].cg().items()}),
                 "panel": PANELS[self.cmb_panel.currentIndex()]},
                indent=2, default=_json_default))
            self.lbl_status.setText(f"gespeichert: {short_name(pdf)}")
            self.lbl_status.setToolTip(str(pdf))
        except Exception as exc:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(exc))


def main():
    app = QApplication(sys.argv)
    win = RabiRb85Window()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
