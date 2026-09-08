"""Leistung und Intensitaet des Multitone-Profils.

WAS HIER GERECHNET WIRD
Die Simulation kennt die Intensitaet nur bis auf eine Konstante - die Felder
sind auf 1 im Spotzentrum normiert, Watt kommen darin nicht vor. Die Konstante
faellt, sobald man EINE physikalische Vorgabe macht. Es gibt zwei natuerliche:

    "ich will Omega/2pi = X"      -> daraus folgt die Leistung
    "ich habe P Milliwatt"        -> daraus folgt Omega/2pi

Beide Richtungen stehen zur Auswahl; gerechnet wird immer dieselbe Kette.

DIE KETTE
1. Atomphysik. kern/rb85_raman.py eliminiert die 5P-Zustaende adiabatisch, mit
   voller Summe ueber D1 UND D2 samt Hyperfeinstruktur und mit Vorzeichen, und
   liefert

       Omega = C_rabi * I ,     delta_LS = C_shift * I ,   Gamma = C_scatter * I

   alle linear in der Intensitaet. Aus C_rabi folgt die Intensitaet, die die
   gewuenschte Rabi-Frequenz macht.

2. Bezugspunkt. Diese Intensitaet ist die des ZEITMITTELS im Auswertebereich -
   dieselbe Konvention wie im Pulsfenster, also die Rabi-Frequenz, die man im
   Labor ohne Trigger misst.

3. Flaeche. Fuer ein Flattop ist die Leistung nicht I mal irgendeiner
   Strahlflaeche, sondern

       P = I_ref * A_eff ,      A_eff = (Integral I dA) / <I>_Bereich .

   Das Flaechenintegral kommt analytisch aus der Summe der Einzelspots
   (profile_total_power im Haupt-GUI), nicht aus dem Rechengitter - das
   schneidet die Airy-Ringe ab und liegt um ein halbes Prozent zu tief.

4. Aufteilung. Die Leistung eines Spots (n,m) geht mit a_x(n)^2 * a_y(m)^2,
   die eines RF-TONS n der x-Achse mit a_x(n)^2 / sum a_x^2 - der Ton speist
   ja alle Spots seiner Spalte. Bei r_x = r_y = 1 sind alle Toene gleich, bei
   r != 1 nicht, und das ist die Zahl, die der AWG-Kanal aushalten muss.

WAS NICHT DRIN STECKT
Beugungseffizienz des AOD, Transmission der Optik, Intermodulation. Die
angegebene Leistung ist die IM PROFIL, also nach dem AOD. Was vorne
hineinmuss, ist erheblich mehr und haengt am Aufbau.
"""

import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QGroupBox, QGridLayout, QMessageBox, QTextEdit
)


def _load_raman():
    """kern/rb85_raman.py laden, ohne das GUI zu sprengen, wenn arc fehlt.

    Die Kopie in kern/ ist die gepflegte; die daneben liegende aeltere wird
    nur als Rueckfallebene genommen."""
    for mod in ("kern.rb85_raman", "rb85_raman"):
        try:
            return __import__(mod, fromlist=["RamanRb85"]), None
        except Exception as exc:                     # arc fehlt, Pfad falsch, ...
            last = exc
    return None, last


class PowerBudgetDialog(QDialog):
    """Nicht-modales Fenster; das Haupt-GUI bleibt bedienbar."""

    # C_rabi fuer Delta = -8 GHz, beta = 0.5, sigma+/sigma+, m_F = 0.
    # Nur Rueckfallwert, wenn rb85_raman nicht importiert werden kann.
    C_RABI_FALLBACK = 71.985

    def __init__(self, parent, fns):
        super().__init__(parent)
        self.setWindowTitle("Leistung und Intensitaet des Profils")
        self.resize(980, 760)
        self.parent_win = parent
        self.fns = fns
        self._raman_mod, self._raman_err = _load_raman()
        self._text = ""

        root = QVBoxLayout(self)
        root.addWidget(self._group_inputs())

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setLineWrapMode(QTextEdit.NoWrap)
        self.out.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        root.addWidget(self.out, 1)

        row = QHBoxLayout()
        self.btn_calc = QPushButton("Neu rechnen")
        self.btn_calc.clicked.connect(self.recompute)
        self.btn_save = QPushButton("Als Textdatei speichern")
        self.btn_save.clicked.connect(self._on_save)
        btn_close = QPushButton("Schliessen")
        btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_calc)
        row.addStretch(1)
        row.addWidget(self.btn_save)
        row.addWidget(btn_close)
        root.addLayout(row)

        self.recompute()

    # ------------------------------------------------------------
    def _dspin(self, val, lo, hi, dec, step, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi); w.setDecimals(dec); w.setSingleStep(step)
        if suffix:
            w.setSuffix(" " + suffix)
        w.setValue(val); w.setKeyboardTracking(False)
        return w

    def _group_inputs(self):
        g = QGroupBox("Vorgabe")
        lay = QGridLayout(g)
        s = self.parent_win.state

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Rabi-Frequenz vorgeben -> Leistung",
                                "Leistung vorgeben -> Rabi-Frequenz"])
        self.cmb_mode.currentIndexChanged.connect(self._sync_mode)

        self.sp_frabi = self._dspin((s.get("f_rabi") or 1e6) * 1e-6,
                                    0.0001, 1000.0, 4, 0.1, "MHz")
        self.sp_power = self._dspin(1.0, 1e-6, 1e6, 5, 0.1, "mW")
        self.sp_delta = self._dspin(-8.0, -2000.0, 2000.0, 3, 1.0, "GHz")
        self.sp_delta.setToolTip("Raman-Verstimmung vom 5P_1/2-Schwerpunkt.\n"
                                 "Negativ = rot. Omega geht wie I/Delta, die\n"
                                 "Streurate wie I/Delta^2 - mehr Verstimmung\n"
                                 "bei mehr Leistung kauft also Qualitaet.")
        self.sp_beta = self._dspin(0.5, 0.01, 0.99, 3, 0.05)
        self.sp_beta.setToolTip("Leistungsaufteilung auf die beiden Raman-Zweige.\n"
                                "0.5 = haelftig, das Optimum fuer Omega.")
        self.sp_m = QSpinBox(); self.sp_m.setRange(-2, 2); self.sp_m.setValue(0)
        self.sp_m.setToolTip("m_F des adressierten Uhrenzustands.")
        self.sp_tp = self._dspin(1.0, 0.0001, 10000.0, 4, 0.1, "us")
        self.sp_tp.setToolTip("Pulslaenge, nur fuer die Streuwahrscheinlichkeit\n"
                              "und die Pulsflaeche in der Ausgabe.")
        self.cmb_region = QComboBox()
        self.cmb_region.addItems(["atomgewichtet (thermisch)", "Plateau",
                                  "Kreis (Radius aus dem Haupt-GUI)",
                                  "Spotzentren"])
        self.cmb_region.setToolTip(
            "Worauf sich die Rabi-Frequenz bezieht: das Zeitmittel der\n"
            "Intensitaet in diesem Bereich. 'atomgewichtet' ist die einzige\n"
            "Wahl, die dieselbe Groesse benutzt wie das Pulsfenster - sonst\n"
            "kalibriert man f_rabi auf eine Flaeche und rechnet die\n"
            "Pulsflaeche am Atom.")
        self.sp_T = self._dspin(17.0, 0.01, 10000.0, 2, 1.0, "uK")
        self.sp_nu = self._dspin(60.4, 0.1, 100000.0, 2, 1.0, "kHz")
        self.sp_pin = self._dspin(300.0, 0.001, 1e6, 3, 10.0, "mW")
        self.sp_pin.setToolTip("Optische Leistung VOR dem AOD - was der Strahl\n"
                               "mitbringt.")
        self.sp_eff_x = self._dspin(0.70, 0.001, 1.0, 4, 0.05)
        self.sp_eff_x.setToolTip(
            "Gesamte Beugungseffizienz des x-AOD in die 1. Ordnung, ueber ALLE\n"
            "seine Toene zusammen - die Groesse, die man am Aufbau misst.\n"
            "Die Aufteilung auf die einzelnen Toene steckt schon in r_x.")
        self.sp_eff_y = self._dspin(0.70, 0.001, 1.0, 4, 0.05)
        self.sp_eff_y.setToolTip("Dasselbe fuer den y-AOD. Beide multiplizieren\n"
                                 "sich - der Spot wird zweimal gebeugt.")
        self.sp_trans = self._dspin(0.80, 0.001, 1.0, 4, 0.05)
        self.sp_trans.setToolTip("Transmission der Optik zwischen AOD und Atom.")
        self.sp_crabi = self._dspin(self.C_RABI_FALLBACK, 1e-6, 1e9, 4, 1.0)
        self.sp_crabi.setToolTip("C_rabi in rad/s pro W/m^2. Wird aus\n"
                                 "kern/rb85_raman.py geholt; von Hand nur\n"
                                 "editierbar, wenn das Modul fehlt.")
        self.sp_crabi.setEnabled(self._raman_mod is None)

        rows = [("Modus", self.cmb_mode), ("Omega/2pi", self.sp_frabi),
                ("Leistung im Profil", self.sp_power),
                ("Verstimmung Delta", self.sp_delta),
                ("Aufteilung beta", self.sp_beta), ("m_F", self.sp_m),
                ("Pulslaenge", self.sp_tp), ("Bezugsbereich", self.cmb_region),
                ("C_rabi [rad/s / (W/m^2)]", self.sp_crabi),
                ("Atom T", self.sp_T), ("Fallenfrequenz nu_r", self.sp_nu),
                ("Leistung vor dem AOD", self.sp_pin),
                ("Beugung AOD x", self.sp_eff_x),
                ("Beugung AOD y", self.sp_eff_y),
                ("Transmission Optik", self.sp_trans)]
        for i, (name, w) in enumerate(rows):
            lay.addWidget(QLabel(name), i % 3, 2 * (i // 3))
            lay.addWidget(w, i % 3, 2 * (i // 3) + 1)
        self.lbl_src = QLabel("-")
        self.lbl_src.setWordWrap(True)
        self.lbl_src.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_src, 3, 0, 1, 6)
        self._sync_mode()
        return g

    def _sync_mode(self, *_):
        by_rabi = self.cmb_mode.currentIndex() == 0
        self.sp_frabi.setEnabled(by_rabi)
        self.sp_power.setEnabled(not by_rabi)

    def _reference(self, c):
        """<I> im gewaehlten Bereich, in Simulationseinheiten, plus Name.

        Im atomgewichteten Fall auf einem eigenen feinen Gitter um den
        Atomort, mit W(r) als Gewicht - dieselbe Groesse, auf die auch das
        Pulsfenster normiert."""
        if self.cmb_region.currentIndex() != 0:
            mask, name = self._mask(c)
            if not mask.any():
                return float("nan"), name
            return float(np.mean(c["mean_exact"][mask])), name
        s = self.parent_win.state
        sig = self.fns["sigma_thermal"](self.sp_nu.value() * 1e3,
                                        self.sp_T.value() * 1e-6)
        _, _, _, _, F, W = self.fns["atom_local_stack"](
            c["centers_x"], c["centers_y"], c["amp_spots"], s["win"],
            s["use_airy"], s["airy_factor"], sig)
        me, _ = self.fns["time_stats_exact"](F, c["k_orders"], c["phases"])
        wv = W.ravel() / W.sum()
        return float(me.ravel() @ wv), "Atom (sigma = %.1f nm)" % (sig * 1e9)

    def _mask(self, c):
        i = self.cmb_region.currentIndex()
        if i == 2:
            return c["region_mask"], "Kreis"
        if i == 3:
            m = np.zeros(c["I_avg"].shape, bool)
            x, y = c["x"], c["y"]
            for cxi, cyi in zip(c["centers_x"], c["centers_y"]):
                m[int(np.argmin(np.abs(y - cyi))),
                  int(np.argmin(np.abs(x - cxi)))] = True
            return m, "Spotzentren"
        return c["plateau"], "Plateau"

    # ------------------------------------------------------------
    def _coefficients(self):
        """C_rabi, C_shift, C_scatter (F=3), eta - aus rb85_raman oder Rueckfall."""
        if self._raman_mod is None:
            return dict(C_rabi=self.sp_crabi.value(), eta=float("nan"),
                        C_scatter_3=float("nan"), C_scatter_2=float("nan"),
                        source="Rueckfallwert von Hand (rb85_raman nicht "
                               "ladbar: %s)" % self._raman_err)
        r = self._raman_mod.RamanRb85(delta_Hz=self.sp_delta.value() * 1e9,
                                      beta=self.sp_beta.value(), q=1)
        c = r.coeff(self.sp_m.value())
        self.sp_crabi.blockSignals(True)
        self.sp_crabi.setValue(c["C_rabi"])
        self.sp_crabi.blockSignals(False)
        return dict(C_rabi=c["C_rabi"], eta=c["eta"],
                    C_scatter_3=c["C_scatter_3"], C_scatter_2=c["C_scatter_2"],
                    source="kern/rb85_raman.py, sigma+/sigma+ kopropagierend, "
                           "volle Summe ueber D1 und D2")

    def recompute(self):
        c = getattr(self.parent_win, "cache", {}) or {}
        if not c or "mean_exact" not in c:
            QMessageBox.information(self, "Nichts zu rechnen",
                                    "Im Haupt-GUI erst 'Recompute' druecken.")
            return
        s = self.parent_win.state
        I_ref_sim, region = self._reference(c)
        if not np.isfinite(I_ref_sim) or I_ref_sim <= 0:
            QMessageBox.information(self, "Leerer Bereich", "Der Bereich ist leer.")
            return

        coef = self._coefficients()
        self.lbl_src.setText("C_rabi = %.4f rad/s pro W/m^2   (%s)"
                             % (coef["C_rabi"], coef["source"]))

        # --- Profilgeometrie in Simulationseinheiten ---
        amp = c["amp_spots"]
        P_sim = self.fns["profile_total_power"](amp, s["win"], s["use_airy"],
                                                s["airy_factor"])
        A_eff = P_sim / I_ref_sim                      # m^2
        I_peak_sim = float(np.max(c["mean_exact"]))

        # --- die eine physikalische Vorgabe ---
        t_p = self.sp_tp.value() * 1e-6
        if self.cmb_mode.currentIndex() == 0:
            f_rabi = self.sp_frabi.value() * 1e6
            I_ref = 2 * np.pi * f_rabi / coef["C_rabi"]
            P_tot = I_ref * A_eff
            self.sp_power.blockSignals(True)
            self.sp_power.setValue(P_tot * 1e3)
            self.sp_power.blockSignals(False)
        else:
            P_tot = self.sp_power.value() * 1e-3
            I_ref = P_tot / A_eff
            f_rabi = coef["C_rabi"] * I_ref / (2 * np.pi)
            self.sp_frabi.blockSignals(True)
            self.sp_frabi.setValue(f_rabi * 1e-6)
            self.sp_frabi.blockSignals(False)

        scale = I_ref / I_ref_sim                      # W/m^2 pro Simulationseinheit

        # --- Aufteilung auf Spots und RF-Toene ---
        w_spot = amp ** 2
        P_spot = P_tot * w_spot / w_spot.sum()
        ax = self.fns["amps_from_ratio"](s["r_x"], s["N_x"]) ** 2
        ay = self.fns["amps_from_ratio"](s["r_y"], s["N_y"]) ** 2
        P_tx = P_tot * ax / ax.sum()
        P_ty = P_tot * ay / ay.sum()

        I_peak_avg = I_peak_sim * scale
        I_peak_inst = float(c.get("cube_max", np.nan)) * scale
        gam3 = coef["C_scatter_3"] * I_ref
        gam2 = coef["C_scatter_2"] * I_ref
        p_sc = 1.0 - np.exp(-gam3 * t_p) if np.isfinite(gam3) else float("nan")
        theta_ref = 2 * f_rabi * t_p                   # in Einheiten pi

        u = "µ"
        L = []
        L.append("Bezug: Omega/2pi gilt fuer das ZEITMITTEL der Intensitaet im "
                 "Bereich '%s'." % region)
        L.append("")
        L.append("ATOM")
        L.append("  Delta                     %12.3f GHz   (beta = %.3f, m_F = %+d)"
                 % (self.sp_delta.value(), self.sp_beta.value(), self.sp_m.value()))
        L.append("  C_rabi                    %12.4f rad/s pro W/m^2"
                 % coef["C_rabi"])
        if np.isfinite(coef["eta"]):
            L.append("  eta = delta_LS/Omega      %12.4f        -> Kontrast <= %.4f"
                     % (coef["eta"], 1.0 / (1.0 + coef["eta"] ** 2)))
        L.append("")
        L.append("PROFIL")
        L.append("  Toene                     %12d  x  %-6d (= %d Spots)"
                 % (s["N_x"], s["N_y"], len(amp)))
        L.append("  Waist                     %12.4f %sm" % (s["win"] * 1e6, u))
        L.append("  Flaeche eines Spots       %12.4f %sm^2"
                 % (self.fns["single_spot_power"](s["win"], s["use_airy"],
                                                  s["airy_factor"]) * 1e12, u))
        L.append("  effektive Flaeche A_eff   %12.1f %sm^2   (= Integral I dA / <I>)"
                 % (A_eff * 1e12, u))
        L.append("")
        L.append("INTENSITAET")
        L.append("  Referenz <I> im Bereich   %12.4f W/cm^2" % (I_ref * 1e-4))
        L.append("  Spitze des Zeitmittels    %12.4f W/cm^2   (%.2f x Referenz)"
                 % (I_peak_avg * 1e-4, I_peak_avg / I_ref))
        if np.isfinite(I_peak_inst):
            L.append("  Spitze momentan           %12.4f W/cm^2   (%.2f x Referenz)"
                     % (I_peak_inst * 1e-4, I_peak_inst / I_ref))
            L.append("     (aus dem Zeitwuerfel; haengt an Bildern/Periode und "
                     "an der eingestellten Belichtung)")
        L.append("")
        L.append("LEISTUNG   -   im Profil, also NACH dem AOD")
        L.append("  gesamt                    %12.5f mW" % (P_tot * 1e3))
        L.append("  pro Spot   Mittel         %12.5f %sW" % (P_spot.mean() * 1e6, u))
        L.append("             hellster       %12.5f %sW" % (P_spot.max() * 1e6, u))
        L.append("             schwaechster   %12.5f %sW" % (P_spot.min() * 1e6, u))
        L.append("  pro RF-Ton x  Mittel      %12.5f %sW   (%d Toene)"
                 % (P_tx.mean() * 1e6, u, s["N_x"]))
        L.append("                staerkster  %12.5f %sW" % (P_tx.max() * 1e6, u))
        L.append("  pro RF-Ton y  Mittel      %12.5f %sW   (%d Toene)"
                 % (P_ty.mean() * 1e6, u, s["N_y"]))
        L.append("                staerkster  %12.5f %sW" % (P_ty.max() * 1e6, u))
        if abs(s["r_x"] - 1) < 1e-9 and abs(s["r_y"] - 1) < 1e-9:
            L.append("     (r_x = r_y = 1: alle Toene gleich stark)")
        else:
            L.append("     (r_x = %.4f, r_y = %.4f: die aeusseren Toene tragen "
                     "r^2 der inneren)" % (s["r_x"], s["r_y"]))
        L.append("")
        L.append("PULS   T_p = %.4f %ss" % (t_p * 1e6, u))
        L.append("  Flaeche auf der Referenz  %12.4f pi        (pi-Puls bei %.4f %ss)"
                 % (theta_ref, 0.5e6 / f_rabi if f_rabi > 0 else float("nan"), u))
        if np.isfinite(gam3):
            L.append("  Streurate F=3 / F=2       %12.1f / %.1f 1/s" % (gam3, gam2))
            L.append("  Streuung pro Puls         %12.4f %%" % (100 * p_sc))
        L.append("")
        L.append("LEISTUNGSKETTE   -   zweimal gebeugt, x-AOD und y-AOD")
        P_in = self.sp_pin.value() * 1e-3
        ex, ey, tr = (self.sp_eff_x.value(), self.sp_eff_y.value(),
                      self.sp_trans.value())
        chain = ex * ey * tr
        P_avail = P_in * chain
        L.append("  vor dem AOD               %12.4f mW" % (P_in * 1e3))
        L.append("  x %5.3f (AOD x) x %5.3f (AOD y) x %5.3f (Optik) = %.4f"
                 % (ex, ey, tr, chain))
        L.append("  verfuegbar im Profil      %12.4f mW" % (P_avail * 1e3))
        L.append("  davon gebraucht           %12.5f mW   (%.4f %% der "
                 "verfuegbaren)" % (P_tot * 1e3, 100 * P_tot / max(P_avail, 1e-30)))
        if P_avail > 0:
            head = P_avail / max(P_tot, 1e-30)
            L.append("  Reserve                   %12.1f x" % head)
            L.append("  benoetigte Beugung        %12.3e  (x*y*Optik, fuer "
                     "genau %.4f MHz)" % (chain / head, f_rabi * 1e-6))
            L.append("")
            L.append("  Die Reserve ist keine Einladung, mehr Leistung aufs Atom")
            L.append("  zu geben: Omega geht wie I/Delta, die Streurate wie")
            L.append("  I/Delta^2. Bei FESTER Rabi-Frequenz sinkt die Streuung")
            L.append("  also mit der Verstimmung, und die Reserve ist genau das")
            L.append("  Budget dafuer. Statt zu extrapolieren wird Delta hier mit")
            L.append("  denselben Koeffizienten wirklich durchgerechnet:")
            best = self._solve_delta(f_rabi, A_eff, P_avail)
            if best is None:
                L.append("    (keine Loesung im abgesuchten Bereich)")
            else:
                dd, I_d, g3_d = best
                cap = abs(abs(dd) - 3000.0) < 1e-6
                L.append("    Delta = %.1f GHz -> %.1f GHz  bei gleicher "
                         "Rabi-Frequenz%s" % (self.sp_delta.value(), dd,
                                              "  (Rand der Suche!)" if cap else ""))
                if cap:
                    L.append("    Die Leistung reicht also noch ueber 3 THz "
                             "hinaus - dort ist die Naeherung 'weit von D1,")
                    L.append("    weit von D2' aber nicht mehr gut, D2 liegt "
                             "7.1 THz entfernt und der Beitrag dreht das")
                    L.append("    Vorzeichen. Die Leistung ist nicht die "
                             "Grenze, die Optik und der AOD sind es.")
                L.append("    dort I = %.1f W/cm^2, P = %.4f mW "
                         "(= die verfuegbare Leistung)"
                         % (I_d * 1e-4, I_d * A_eff * 1e3))
                L.append("    Streuung pro Puls        %.4f %% -> %.5f %%"
                         % (100 * p_sc, 100 * (1 - np.exp(-g3_d * t_p))))
        L.append("")
        L.append("NICHT enthalten: Intermodulation im AOD, Fuellzeit, "
                 "Polarisationsfehler.")
        self._text = "\n".join(L)
        self.out.setPlainText(self._text)

    # ------------------------------------------------------------
    def _solve_delta(self, f_rabi, A_eff, P_avail, d_max=3000.0, n=600):
        """Groesste Verstimmung, bei der die verfuegbare Leistung noch fuer
        f_rabi reicht - mit den echten Koeffizienten, nicht mit Omega ~ 1/Delta.

        Rueckgabe (Delta_GHz, I [W/m^2], Streurate F=3 [1/s]) oder None."""
        if self._raman_mod is None or P_avail <= 0:
            return None
        sign = -1.0 if self.sp_delta.value() < 0 else 1.0
        grid = np.geomspace(abs(self.sp_delta.value()) or 1.0, d_max, n)
        best = None
        for dg in grid:
            try:
                r = self._raman_mod.RamanRb85(delta_Hz=sign * dg * 1e9,
                                              beta=self.sp_beta.value(), q=1)
                cc = r.coeff(self.sp_m.value())
            except Exception:
                continue
            I = 2 * np.pi * f_rabi / cc["C_rabi"]
            if I * A_eff <= P_avail:
                best = (sign * dg, I, cc["C_scatter_3"] * I)
        return best

    def _on_save(self):
        if not self._text:
            return
        s = self.parent_win.state
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name = "Leistungsbudget_N{}x{}_{}.txt".format(s["N_x"], s["N_y"], stamp)
        try:
            (self.parent_win.out_dir / name).write_text(
                stamp + "\n\n" + self._text + "\n", encoding="utf-8")
            self.lbl_src.setText("gespeichert: " + name)
        except Exception as exc:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(exc))
