"""Power and intensity of the multitone profile.

WHAT IS COMPUTED HERE
The simulation knows the intensity only up to a constant - the fields are
normalised to 1 at the spot centre, watts do not appear in them. The constant
is fixed as soon as ONE physical input is given. There are two natural ones:

    "I want Omega/2pi = X"      -> the power follows
    "I have P milliwatt"        -> Omega/2pi follows

Both directions are selectable; the same chain is computed either way.

THE CHAIN
1. Atomic physics. kern/rb85_raman.py adiabatically eliminates the 5P states,
   with the full, sign-correct sum over D1 AND D2 including hyperfine
   structure, and returns

       Omega = C_rabi * I ,     delta_LS = C_shift * I ,   Gamma = C_scatter * I

   all linear in the intensity. C_rabi gives the intensity that makes the
   wanted Rabi frequency.

2. Reference. That intensity is the one of the TIME AVERAGE in the evaluation
   region - the same convention as in the pulse window, i.e. the Rabi
   frequency one measures in the lab without a trigger. With the atom weighted
   option it is the mean weighted with the atomic position probability
   density; the other options are hard masks and average over areas on which
   no single atom sits.

3. Area. For a flat top the power is not I times some beam area but

       P = I_ref * A_eff ,      A_eff = (integral I dA) / <I>_region .

   The area integral comes analytically from the sum of the single spots
   (profile_total_power in the main GUI), not from the computation grid - that
   one truncates the Airy rings and comes out half a percent low.

4. Splitting. The power of spot (n,m) goes as a_x(n)^2 * a_y(m)^2, that of RF
   TONE n of the x axis as a_x(n)^2 / sum a_x^2 - the tone feeds all spots of
   its column. At r_x = r_y = 1 all tones are equal, at r != 1 they are not,
   and that is the number the AWG channel has to survive.

5. Power chain. Before the AOD, times the diffraction efficiency of BOTH AODs
   (the spot is diffracted twice) times the optics transmission. What comes
   out is what is available in the profile; the ratio to what is needed is the
   headroom, and the headroom is really a detuning budget: Omega goes as
   I/Delta and the scattering rate as I/Delta^2, so at fixed Rabi frequency
   more detuning plus more power buys less scattering.

WHAT IS NOT IN HERE
Intermodulation in the AOD, its fill time, polarisation errors.
"""

import datetime

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QPushButton, QComboBox, QGroupBox, QGridLayout, QMessageBox, QTextEdit
)


def _load_raman():
    """Load kern/rb85_raman.py without breaking the GUI when arc is missing.

    The copy in kern/ is the maintained one; the older sibling next to it is
    only a fallback."""
    for mod in ("kern.rb85_raman", "rb85_raman"):
        try:
            return __import__(mod, fromlist=["RamanRb85"]), None
        except Exception as exc:                     # arc fehlt, Pfad falsch, ...
            last = exc
    return None, last


class PowerBudgetDialog(QDialog):
    """Modeless window; the main GUI stays usable."""

    # C_rabi for Delta = -8 GHz, beta = 0.5, sigma+/sigma+, m_F = 0.
    # Fallback only, used when rb85_raman cannot be imported.
    C_RABI_FALLBACK = 71.985

    def __init__(self, parent, fns):
        super().__init__(parent)
        self.setWindowTitle("Power and intensity of the profile")
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
        self.btn_calc = QPushButton("Recompute")
        self.btn_calc.clicked.connect(self.recompute)
        self.btn_save = QPushButton("Save as text file")
        self.btn_save.clicked.connect(self._on_save)
        btn_close = QPushButton("Close")
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
        g = QGroupBox("Input")
        lay = QGridLayout(g)
        s = self.parent_win.state

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["given Rabi frequency -> power",
                                "given power -> Rabi frequency"])
        self.cmb_mode.currentIndexChanged.connect(self._sync_mode)

        self.sp_frabi = self._dspin((s.get("f_rabi") or 1e6) * 1e-6,
                                    0.0001, 1000.0, 4, 0.1, "MHz")
        self.sp_power = self._dspin(1.0, 1e-6, 1e6, 5, 0.1, "mW")
        self.sp_delta = self._dspin(-8.0, -2000.0, 2000.0, 3, 1.0, "GHz")
        self.sp_delta.setToolTip("Raman detuning from the 5P_1/2 centroid.\n"
                                 "Negative = red. Omega goes as I/Delta, the\n"
                                 "scattering rate as I/Delta^2 - more detuning\n"
                                 "with more power therefore buys quality.")
        self.sp_beta = self._dspin(0.5, 0.01, 0.99, 3, 0.05)
        self.sp_beta.setToolTip("Power split between the two Raman legs.\n"
                                "0.5 = even, the optimum for Omega.")
        self.sp_m = QSpinBox(); self.sp_m.setRange(-2, 2); self.sp_m.setValue(0)
        self.sp_m.setToolTip("m_F of the addressed clock state.")
        self.sp_tp = self._dspin(1.0, 0.0001, 10000.0, 4, 0.1, "us")
        self.sp_tp.setToolTip("Pulse length, only for the scattering probability\n"
                              "and the pulse area in the output.")
        self.cmb_region = QComboBox()
        self.cmb_region.addItems(["atom weighted (thermal)", "Plateau",
                                  "Kreis (Radius aus dem Haupt-GUI)",
                                  "spot centres"])
        self.cmb_region.setToolTip(
            "What the Rabi frequency refers to: the time average of the\n"
            "intensity in this region. 'atom weighted' is the only choice\n"
            "that uses the same quantity as the pulse window - otherwise one\n"
            "calibrates f_rabi on an area and computes the pulse area at the\n"
            "atom.")
        self.sp_T = self._dspin(17.0, 0.01, 10000.0, 2, 1.0, "uK")
        self.sp_nu = self._dspin(60.4, 0.1, 100000.0, 2, 1.0, "kHz")
        self.sp_pin = self._dspin(300.0, 0.001, 1e6, 3, 10.0, "mW")
        self.sp_pin.setToolTip("Optical power BEFORE the AOD - what the beam\n"
                               "brings along.")
        self.sp_eff_x = self._dspin(0.70, 0.001, 1.0, 4, 0.05)
        self.sp_eff_x.setToolTip(
            "Total diffraction efficiency of the x-AOD into the 1st order, over\n"
            "ALL of its tones together - the quantity one measures at the\n"
            "setup. The split between the tones is already in r_x.")
        self.sp_eff_y = self._dspin(0.70, 0.001, 1.0, 4, 0.05)
        self.sp_eff_y.setToolTip("Same for the y-AOD. The two multiply - the spot\n"
                                 "is diffracted twice.")
        self.sp_trans = self._dspin(0.80, 0.001, 1.0, 4, 0.05)
        self.sp_trans.setToolTip("Transmission of the optics between AOD and atom.")
        self.sp_crabi = self._dspin(self.C_RABI_FALLBACK, 1e-6, 1e9, 4, 1.0)
        self.sp_crabi.setToolTip("C_rabi in rad/s per W/m^2. Taken from\n"
                                 "kern/rb85_raman.py; editable by hand only\n"
                                 "when that module is missing.")
        self.sp_crabi.setEnabled(self._raman_mod is None)

        rows = [("mode", self.cmb_mode), ("Omega/2pi", self.sp_frabi),
                ("power in the profile", self.sp_power),
                ("detuning Delta", self.sp_delta),
                ("power split beta", self.sp_beta), ("m_F", self.sp_m),
                ("pulse length", self.sp_tp), ("reference region", self.cmb_region),
                ("C_rabi [rad/s / (W/m^2)]", self.sp_crabi),
                ("atom T", self.sp_T), ("trap frequency nu_r", self.sp_nu),
                ("power before the AOD", self.sp_pin),
                ("diffraction AOD x", self.sp_eff_x),
                ("diffraction AOD y", self.sp_eff_y),
                ("optics transmission", self.sp_trans)]
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
        """<I> in the chosen region, in simulation units, plus its name.

        In the atom weighted case on a dedicated fine grid around the atom,
        with W(r) as the weight - the same quantity the pulse window
        normalises to."""
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
        """C_rabi, C_shift, C_scatter (F=3), eta - from rb85_raman or fallback."""
        if self._raman_mod is None:
            return dict(C_rabi=self.sp_crabi.value(), eta=float("nan"),
                        C_scatter_3=float("nan"), C_scatter_2=float("nan"),
                        source="manual fallback (rb85_raman not loadable: "
                               "%s)" % self._raman_err)
        r = self._raman_mod.RamanRb85(delta_Hz=self.sp_delta.value() * 1e9,
                                      beta=self.sp_beta.value(), q=1)
        c = r.coeff(self.sp_m.value())
        self.sp_crabi.blockSignals(True)
        self.sp_crabi.setValue(c["C_rabi"])
        self.sp_crabi.blockSignals(False)
        return dict(C_rabi=c["C_rabi"], eta=c["eta"],
                    C_scatter_3=c["C_scatter_3"], C_scatter_2=c["C_scatter_2"],
                    source="kern/rb85_raman.py, sigma+/sigma+ co-propagating, "
                           "full sum over D1 and D2")

    def recompute(self):
        c = getattr(self.parent_win, "cache", {}) or {}
        if not c or "mean_exact" not in c:
            QMessageBox.information(self, "Nothing to compute",
                                    "Press 'Recompute' in the main GUI first.")
            return
        s = self.parent_win.state
        I_ref_sim, region = self._reference(c)
        if not np.isfinite(I_ref_sim) or I_ref_sim <= 0:
            QMessageBox.information(self, "Empty region", "The region is empty.")
            return

        coef = self._coefficients()
        self.lbl_src.setText("C_rabi = %.4f rad/s per W/m^2   (%s)"
                             % (coef["C_rabi"], coef["source"]))

        # --- Profilgeometrie in Simulationseinheiten ---
        amp = c["amp_spots"]
        P_sim = self.fns["profile_total_power"](amp, s["win"], s["use_airy"],
                                                s["airy_factor"])
        A_eff = P_sim / I_ref_sim                      # m^2
        I_peak_sim = float(np.max(c["mean_exact"]))

        # --- die eine physikalische Input ---
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

        # --- split over spots and RF tones ---
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
        L.append("Reference: Omega/2pi refers to the TIME AVERAGE of the intensity in the "
                 "region '%s'." % region)
        L.append("")
        L.append("ATOM")
        L.append("  Delta                     %12.3f GHz   (beta = %.3f, m_F = %+d)"
                 % (self.sp_delta.value(), self.sp_beta.value(), self.sp_m.value()))
        L.append("  C_rabi                    %12.4f rad/s per W/m^2"
                 % coef["C_rabi"])
        if np.isfinite(coef["eta"]):
            L.append("  eta = delta_LS/Omega      %12.4f        -> contrast <= %.4f"
                     % (coef["eta"], 1.0 / (1.0 + coef["eta"] ** 2)))
        L.append("")
        L.append("PROFILE")
        L.append("  tones                     %12d  x  %-6d (= %d spots)"
                 % (s["N_x"], s["N_y"], len(amp)))
        L.append("  waist                     %12.4f %sm" % (s["win"] * 1e6, u))
        L.append("  area of one spot          %12.4f %sm^2"
                 % (self.fns["single_spot_power"](s["win"], s["use_airy"],
                                                  s["airy_factor"]) * 1e12, u))
        L.append("  effective area A_eff     %12.1f %sm^2   (= integral I dA / <I>)"
                 % (A_eff * 1e12, u))
        L.append("")
        L.append("INTENSITY")
        L.append("  reference <I> (MEAN)    %12.4f W/cm^2" % (I_ref * 1e-4))
        L.append("  peak of the time avg.   %12.4f W/cm^2   (%.2f x reference)"
                 % (I_peak_avg * 1e-4, I_peak_avg / I_ref))
        if np.isfinite(I_peak_inst):
            L.append("  peak instantaneous          %12.4f W/cm^2   (%.2f x reference)"
                     % (I_peak_inst * 1e-4, I_peak_inst / I_ref))
            L.append("     (from the time cube; depends on frames per period "
                     "and on the exposure set in the main GUI)")
        L.append("")
        L.append("POWER   -   in the profile, i.e. AFTER the AODs")
        L.append("  total                     %12.5f mW" % (P_tot * 1e3))
        L.append("  per spot   mean           %12.5f %sW" % (P_spot.mean() * 1e6, u))
        L.append("             brightest      %12.5f %sW" % (P_spot.max() * 1e6, u))
        L.append("             weakest       %12.5f %sW" % (P_spot.min() * 1e6, u))
        L.append("  per RF tone x  mean      %12.5f %sW   (%d tones)"
                 % (P_tx.mean() * 1e6, u, s["N_x"]))
        L.append("                strongest %12.5f %sW" % (P_tx.max() * 1e6, u))
        L.append("  per RF tone y  mean      %12.5f %sW   (%d tones)"
                 % (P_ty.mean() * 1e6, u, s["N_y"]))
        L.append("                strongest %12.5f %sW" % (P_ty.max() * 1e6, u))
        if abs(s["r_x"] - 1) < 1e-9 and abs(s["r_y"] - 1) < 1e-9:
            L.append("     (r_x = r_y = 1: all tones equal)")
        else:
            L.append("     (r_x = %.4f, r_y = %.4f: the outer tones carry r^2 of the inner "
                     "ones)" % (s["r_x"], s["r_y"]))
        L.append("")
        L.append("PULSE   T_p = %.4f %ss" % (t_p * 1e6, u))
        L.append("  area at the reference   %12.4f pi        (pi pulse at %.4f %ss)"
                 % (theta_ref, 0.5e6 / f_rabi if f_rabi > 0 else float("nan"), u))
        if np.isfinite(gam3):
            L.append("  scattering rate F=3/F=2      %12.1f / %.1f 1/s" % (gam3, gam2))
            L.append("  scattering per pulse        %12.4f %%" % (100 * p_sc))
        L.append("")
        L.append("POWER CHAIN   -   diffracted twice, x-AOD and y-AOD")
        P_in = self.sp_pin.value() * 1e-3
        ex, ey, tr = (self.sp_eff_x.value(), self.sp_eff_y.value(),
                      self.sp_trans.value())
        chain = ex * ey * tr
        P_avail = P_in * chain
        L.append("  before the AOD              %12.4f mW" % (P_in * 1e3))
        L.append("  x %5.3f (AOD x) x %5.3f (AOD y) x %5.3f (optics) = %.4f"
                 % (ex, ey, tr, chain))
        L.append("  available in profile     %12.4f mW" % (P_avail * 1e3))
        L.append("  of which needed           %12.5f mW   (%.4f %% of what is "
                 "available)" % (P_tot * 1e3, 100 * P_tot / max(P_avail, 1e-30)))
        if P_avail > 0:
            head = P_avail / max(P_tot, 1e-30)
            L.append("  headroom                  %12.1f x" % head)
            L.append("  diffraction needed        %12.3e  (x*y*optics, for exactly "
                     "%.4f MHz)" % (chain / head, f_rabi * 1e-6))
            L.append("")
            L.append("  The headroom is not an invitation to put more power on the")
            L.append("  atom: Omega goes as I/Delta, the scattering rate as I/Delta^2.")
            L.append("  At FIXED Rabi frequency the scattering therefore drops with")
            L.append("  the detuning, and the headroom is exactly the budget for it.")
            L.append("  Instead of extrapolating, Delta is really solved for here")
            L.append("  with the same coefficients:")
            best = self._solve_delta(f_rabi, A_eff, P_avail)
            if best is None:
                L.append("    (no solution within the searched range)")
            else:
                dd, I_d, g3_d = best
                cap = abs(abs(dd) - 3000.0) < 1e-6
                L.append("    Delta = %.1f GHz -> %.1f GHz  at the same Rabi "
                         "frequency%s" % (self.sp_delta.value(), dd,
                                              "  (edge of the search!)" if cap else ""))
                if cap:
                    L.append("    So the power is enough beyond 3 THz - but there the "
                             "assumption 'far from D1,")
                    L.append("    far from D2' no longer holds: D2 is 7.1 THz away and "
                             "its contribution flips the")
                    L.append("    sign. Power is not the limit here, the optics and the "
                             "AOD are.")
                L.append("    there I = %.1f W/cm^2, P = %.4f mW "
                         "(= the available power)"
                         % (I_d * 1e-4, I_d * A_eff * 1e3))
                L.append("    scattering per pulse       %.4f %% -> %.5f %%"
                         % (100 * p_sc, 100 * (1 - np.exp(-g3_d * t_p))))
        L.append("")
        L.append("NOT included: intermodulation in the AOD, fill time, polarisation "
                 "errors.")
        self._text = "\n".join(L)
        self.out.setPlainText(self._text)

    # ------------------------------------------------------------
    def _solve_delta(self, f_rabi, A_eff, P_avail, d_max=3000.0, n=600):
        """Largest detuning at which the available power still delivers f_rabi -
        with the real coefficients, not with Omega ~ 1/Delta.

        Returns (Delta_GHz, I [W/m^2], scattering rate F=3 [1/s]) or None."""
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
        name = "PowerBudget_N{}x{}_w{:.3f}um_width{:.4f}MHz_{}_report.md".format(
            s["N_x"], s["N_y"], s["win"] * 1e6, s["width_x"] * 1e-6, stamp)
        try:
            md = ("# Power and intensity of the multitone profile\n\n"
                  "Generated %s by `power_budget.py` "
                  "(Beating_Multitone_GUI).\n\n```\n%s\n```\n"
                  % (stamp, self._text))
            (self.parent_win.out_dir / name).write_text(md, encoding="utf-8")
            self.lbl_src.setText("saved: " + name)
        except Exception as exc:
            QMessageBox.critical(self, "Saving failed", str(exc))
