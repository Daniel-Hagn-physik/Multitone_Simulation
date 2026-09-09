"""Pulse area and trigger jitter.

THE QUESTION
A rectangular pulse of length T_p is triggered onto a time window of high
intensity. How large is the accumulated Rabi area

    theta(r, t_0) = int_{t_0}^{t_0+T_p} Omega(r,t) dt ,

and how much does it change when the trigger is off by up to +-Delta?

THE POINT THAT IS EASILY MISSED
A pulse is the same boxcar in time as a camera exposure, only a thousand times
shorter. On beat order d it acts as sinc(d*f_0*T_p). At T_p = 1 us and
f_0 = 10 kHz that is 0.9999 for the fundamental and still 0.976 at 120 kHz -
the pulse does NOT average the beating away, it SAMPLES it. That is why the
area depends sensitively on the trigger, and on the time scale of the FAST
beats (a few us), not of the fundamental period.

NORMALISATION
Omega is fixed by the profile up to a constant; the constant comes from the
assumed Rabi frequency:

    Omega(r,t) = 2*pi*f_rabi * g(r,t) / <g> ,   g = I  or  sqrt(I),

with <g> the mean over time AND the evaluation region - the calibration one
would do in the lab: f_rabi is the Rabi frequency measured on the time
averaged profile. A pulse of length T_p at that reference intensity has
exactly theta = 2*pi*f_rabi*T_p.

WHERE AVERAGING HAPPENS
theta(r,t_0) is a field in space. Every curve in this window is a WEIGHTED
SPATIAL MEAN of it,

    <theta>_W = sum_r W(r) theta(r) / sum_r W(r) ,

with W the atomic position probability density in the atom weighted mode and
W = 1 on a hard mask otherwise. Only the map shows theta(r) point by point.
The excitation is the one quantity where the order matters, because sin^2 is
non-linear - it is applied per point and averaged afterwards.
"""

import datetime

import numpy as np

import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch, Ellipse
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPushButton, QComboBox, QGroupBox, QGridLayout, QMessageBox
)

try:
    from power_budget import _load_raman
except Exception:                                    # sibling module missing
    def _load_raman():
        return None, "power_budget.py not found"

A4_PORTRAIT = (8.27, 11.69)                          # inch
TRACE_MAX_PERIODS = 40.0     # beat periods the top plot will ever draw
C_RABI_FALLBACK = 71.985                             # rad/s per W/m^2 at -8 GHz


class PulseTimingDialog(QDialog):
    """Modeless window; the main GUI stays usable."""

    def __init__(self, parent, fns):
        super().__init__(parent)
        self.setWindowTitle("Pulse area and trigger jitter")
        self.resize(1180, 900)
        self.parent_win = parent
        self.fns = fns
        self._last = None
        self._raman_mod, _ = _load_raman()

        root = QVBoxLayout(self)
        root.addWidget(self._group_inputs())

        self.fig = Figure(figsize=(11, 8.6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)

        self.lbl_info = QLabel("-")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size: 11px;")
        root.addWidget(self.lbl_info)

        row = QHBoxLayout()
        self.btn_draw = QPushButton("Recompute")
        self.btn_draw.clicked.connect(self.recompute)
        self.btn_apply = QPushButton("Send trigger to main GUI")
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_save = QPushButton("Save PDF (LaTeX, A4)")
        self.btn_save.clicked.connect(self._on_save)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_draw)
        row.addStretch(1)
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_save)
        row.addWidget(btn_close)
        root.addLayout(row)

        self._wire_auto()
        self.recompute()

    # --------------------------------------------------------- live inputs
    def _wire_auto(self):
        """Every physical input recomputes by itself.

        Without this the detuning (and every other spin box) silently did
        nothing until 'Recompute' was pressed - the window then showed a
        figure that no longer belonged to the numbers above it.  The timer
        debounces so that dragging a spin box does not queue ten full
        recomputes; the write-back of P/Omega inside recompute() is guarded
        by blockSignals and therefore cannot feed back in here."""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.recompute)
        for w in (self.sp_pprof, self.sp_pspot, self.sp_frabi, self.sp_delta,
                  self.sp_tp, self.sp_delay, self.sp_T, self.sp_nu, self.sp_t0):
            w.valueChanged.connect(self._queue)
        self.sp_pts.valueChanged.connect(self._queue)
        for w in (self.cmb_norm, self.cmb_region):
            w.currentIndexChanged.connect(self._queue)
        self.cb_auto.stateChanged.connect(self._queue)

    def _queue(self, *_):
        self._timer.start()

    # ------------------------------------------------------------ UI
    def _dspin(self, val, lo, hi, dec, step, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi); w.setDecimals(dec); w.setSingleStep(step)
        if suffix:
            w.setSuffix(" " + suffix)
        w.setValue(val); w.setKeyboardTracking(False)
        return w

    def _group_inputs(self):
        g = QGroupBox("Pulse, trigger and evaluation")
        lay = QGridLayout(g)
        s = self.parent_win.state

        self.cmb_norm = QComboBox()
        self.cmb_norm.addItems(["power in the profile -> Omega",
                                "power per spot (mean) -> Omega",
                                "Omega given -> power"])
        self.cmb_norm.setToolTip(
            "Which single number fixes the scale. The simulation knows the\n"
            "SHAPE of I(r,t) only; one physical input turns it into watts and\n"
            "rad/s. Everything relative - the curve shapes, the optimal\n"
            "trigger, sigma_theta, the deviation at a given jitter - is\n"
            "independent of this choice.")
        self.cmb_norm.currentIndexChanged.connect(self._sync_norm)
        self.sp_pprof = self._dspin(10.0, 1e-6, 1e9, 5, 1.0, "uW")
        self.sp_pprof.setToolTip("Optical power in the whole multitone profile,\n"
                                 "i.e. after both AODs.")
        self.sp_pspot = self._dspin(10.0 / 12.0, 1e-9, 1e9, 6, 0.1, "uW")
        self.sp_pspot.setToolTip("Mean optical power per spot; the profile power\n"
                                 "is this times the number of spots.")
        self.sp_frabi = self._dspin((s.get("f_rabi") or 1e6) * 1e-6,
                                    0.0001, 100000.0, 4, 0.1, "MHz")
        self.sp_frabi.setToolTip(
            "Two-photon Rabi frequency Omega/2pi, referred to the TIME\n"
            "AVERAGED intensity in the evaluation region. Input in the third\n"
            "mode, computed output in the first two.")
        self.sp_tp = self._dspin(1.0, 0.0001, 1.0e6, 4, 0.1, "us")
        self.sp_tp.setToolTip(
            "Pulse length (rectangular pulse). Long pulses are allowed: for\n"
            "Omega ~ I the area is exact for ANY T_p (the boxcar is a sinc on\n"
            "the beat orders, not a time integral), so nothing gets slower.\n"
            "The time axis of the top plot follows T_p instead of staying at\n"
            "three beat periods.")
        self.btn_pi = QPushButton("set to pi pulse")
        self.btn_pi.setToolTip("T_p = 1/(2 f_rabi): the length that gives\n"
                               "theta = pi at the reference intensity.")
        self.btn_pi.clicked.connect(self._set_pi)

        self.sp_delay = self._dspin(1.0, 0.0001, 1.0e6, 4, 0.1, "us")
        self.sp_delay.setToolTip("Half width of the delay scan: 0 is the perfect\n"
                                 "trigger, +-this value the limit.")
        self.sp_pts = QSpinBox(); self.sp_pts.setRange(21, 4001)
        self.sp_pts.setValue(401); self.sp_pts.setSingleStep(50)

        self.cmb_region = QComboBox()
        self.cmb_region.addItems(["atom weighted (thermal)", "plateau",
                                  "circle (radius from main GUI)",
                                  "spot centres"])
        self.cmb_region.setToolTip(
            "What the curves are averaged over.\n\n"
            "'atom weighted' is the only choice that gives an ATOM quantity: a\n"
            "Gaussian weight of the position probability density around the\n"
            "atom, sigma from T and nu_r as in Weighted_Multitone_Lens_GUI.\n"
            "The others average over areas on which no single atom sits.")
        self.sp_T = self._dspin(17.0, 0.01, 10000.0, 2, 1.0, "uK")
        self.sp_T.setToolTip("Atom temperature; with nu_r it gives\n"
                             "sigma^2 = hbar/(2 m w) coth(hbar w / 2 kB T).")
        self.sp_nu = self._dspin(60.4, 0.1, 100000.0, 2, 1.0, "kHz")
        self.sp_nu.setToolTip("Radial trap frequency nu_r.")
        self.sp_delta = self._dspin(50.0, -3000.0, 3000.0, 3, 1.0, "GHz")
        self.sp_delta.setToolTip("Raman detuning from the 5P_1/2 centroid.\n"
                                 "Positive = blue. It sets C_rabi and hence the\n"
                                 "Rabi frequency a given power produces.")

        self.cb_auto = QCheckBox("trigger automatically on max. area")
        self.cb_auto.setChecked(True)
        self.cb_auto.setToolTip(
            "Finds the maximum of the mean pulse area over one full beat\n"
            "period. There the derivative with respect to the delay vanishes,\n"
            "so jitter enters only quadratically - the actual reason to\n"
            "trigger on a maximum and not on a slope.")
        self.cb_auto.stateChanged.connect(
            lambda _: self.sp_t0.setEnabled(not self.cb_auto.isChecked()))
        self.sp_t0 = self._dspin(0.0, 0.0, 1e6, 4, 1.0, "us")
        self.sp_t0.setEnabled(False)

        self.cmb_map = QComboBox()
        self.cmb_map.addItems(["pulse area theta(r)", "excitation p(r)"])
        self.cmb_map.setToolTip(
            "theta(r) = C_rabi(Delta) * I(r) * T_p is LINEAR in the detuning:\n"
            "at fixed power Delta is a pure scalar prefactor, so the map is\n"
            "rescaled but pixel for pixel the same picture - only the colour\n"
            "bar numbers move. The excitation\n\n"
            "    p = sin^2(sqrt(1+eta^2) theta / 2) / (1 + eta^2)\n\n"
            "is non-linear in theta and also carries eta(Delta), so it is the\n"
            "map that actually changes shape when you tune Delta.")
        self.cmb_map.currentIndexChanged.connect(lambda _: self._redraw_only())

        self.cmb_layout = QComboBox()
        self.cmb_layout.addItems(["screen (map on the right)",
                                  "A4 portrait (stacked, for LaTeX)"])
        self.cmb_layout.setToolTip(
            "The A4 layout stacks all four panels and fills a portrait page,\n"
            "leaving room for a caption underneath. The PDF export always\n"
            "uses it.")
        self.cmb_layout.currentIndexChanged.connect(lambda _: self._redraw_only())
        self.cb_suptitle = QCheckBox("title inside the figure")
        self.cb_suptitle.setChecked(True)
        self.cb_suptitle.setToolTip("Switch off for LaTeX: the caption belongs\n"
                                    "in \\caption{}, not in the graphic.")
        self.cb_suptitle.stateChanged.connect(lambda _: self._redraw_only())

        rows = [("normalise by", self.cmb_norm),
                ("P in the profile", self.sp_pprof),
                ("P per spot (mean)", self.sp_pspot),
                ("Omega/2pi", self.sp_frabi),
                ("detuning Delta", self.sp_delta),
                ("pulse length T_p", self.sp_tp), ("", self.btn_pi),
                ("delay range +-", self.sp_delay), ("points", self.sp_pts),
                ("region", self.cmb_region), ("map shows", self.cmb_map),
                ("atom T", self.sp_T), ("trap frequency nu_r", self.sp_nu),
                ("", self.cb_auto), ("trigger t_0", self.sp_t0),
                ("layout", self.cmb_layout), ("", self.cb_suptitle)]
        n_row = 5
        for i, (name, w) in enumerate(rows):
            lay.addWidget(QLabel(name), i % n_row, 2 * (i // n_row))
            lay.addWidget(w, i % n_row, 2 * (i // n_row) + 1)
        self._sync_norm()
        self.lbl_note = QLabel("-")
        self.lbl_note.setStyleSheet("color: #555; font-size: 10px;")
        self.lbl_note.setWordWrap(True)
        lay.addWidget(self.lbl_note, n_row, 0, 1,
                      2 * ((len(rows) + n_row - 1) // n_row))
        return g

    def _sync_norm(self, *_):
        i = self.cmb_norm.currentIndex()
        self.sp_pprof.setEnabled(i == 0)
        self.sp_pspot.setEnabled(i == 1)
        self.sp_frabi.setEnabled(i == 2)

    def _set_pi(self):
        f = self.sp_frabi.value() * 1e6
        if f > 0:
            self.sp_tp.setValue(1e6 / (2.0 * f))

    # ------------------------------------------------------- regions
    def _atom_sigma(self):
        return self.fns["sigma_thermal"](self.sp_nu.value() * 1e3,
                                         self.sp_T.value() * 1e-6)

    def _mask(self, c):
        idx = self.cmb_region.currentIndex()
        if idx == 2:
            return c["region_mask"], "circle"
        if idx == 3:
            m = np.zeros(c["I_avg"].shape, bool)
            x, y = c["x"], c["y"]
            for cxi, cyi in zip(c["centers_x"], c["centers_y"]):
                m[int(np.argmin(np.abs(y - cyi))),
                  int(np.argmin(np.abs(x - cxi)))] = True
            return m, "spot centres"
        return c["plateau"], "plateau"

    def _fields(self, c):
        """Field stack, mask, weight and grid for the chosen evaluation.

        In the atom weighted case a dedicated fine local grid is built around
        the atom: the global one has ~0.6 um per cell, far too coarse for an
        atom of 0.1 um."""
        if self.cmb_region.currentIndex() != 0:
            mask, name = self._mask(c)
            return c["F_stack"], mask, None, c["x"], c["y"], name, None
        s = self.parent_win.state
        sig = self._atom_sigma()
        xs, ys, Xs, Ys, F, W = self.fns["atom_local_stack"](
            c["centers_x"], c["centers_y"], c["amp_spots"], s["win"],
            s["use_airy"], s["airy_factor"], sig)
        return F, None, W, xs, ys, "atom, sigma = %.1f nm" % (sig * 1e9), sig

    # ------------------------------------------------------- physics
    def _c_rabi(self):
        """C_rabi plus the scattering and light shift coefficients."""
        if self._raman_mod is None:
            self._coef = None
            return C_RABI_FALLBACK, "fallback constant"
        try:
            r = self._raman_mod.RamanRb85(delta_Hz=self.sp_delta.value() * 1e9,
                                          beta=0.5, q=1)
            self._coef = r.coeff(0)
            return self._coef["C_rabi"], "kern/rb85_raman.py"
        except Exception:
            self._coef = None
            return C_RABI_FALLBACK, "fallback constant"

    def recompute(self):
        c = getattr(self.parent_win, "cache", {}) or {}
        if not c or "F_stack" not in c:
            QMessageBox.information(self, "Nothing to compute",
                                    "Press 'Recompute' in the main GUI first.")
            return
        f0, T0 = c["f0"], c["T0"]
        if not (f0 > 0 and np.isfinite(T0)):
            QMessageBox.information(
                self, "No beat period",
                "Without a common fundamental period there is no defined "
                "trigger instant within the cycle.")
            return
        F, mask, W, gx, gy, region, sig_atom = self._fields(c)
        if mask is not None and not mask.any():
            QMessageBox.information(self, "Empty region",
                                    "The chosen region contains no points.")
            return

        k, ph = c["k_orders"], c["phases"]
        law = self.parent_win.state.get("rabi_law", "I")
        t_p = self.sp_tp.value() * 1e-6
        half = self.sp_delay.value() * 1e-6
        n_pts = self.sp_pts.value()

        # --- mean area as a function of the trigger, over one period ---
        n_scan = int(max(2000, min(40000, 40 * T0 / max(t_p, 1e-12))))
        t_scan = np.arange(n_scan) / n_scan * T0
        if law == "I":
            coef, orders = self.fns["beat_coeffs_mean"](F, k, ph, mask, W)
            area = lambda tt: self.fns["pulse_area_curve"](coef, orders, f0, t_p, tt)
            g_ref = float(np.real(coef[0]))
            # instantaneous weighted mean intensity, from the same coefficients
            g_of_t = lambda tt: np.real(
                np.exp(2j * np.pi * f0 * np.outer(np.atleast_1d(tt), orders)) @ coef)
        else:
            wser, dt = self.fns["sqrt_mean_series"](F, k, ph, f0, mask, W,
                                                    oversample=10)
            area = lambda tt: self.fns["sqrt_area_curve"](wser, dt, t_p, tt)
            g_ref = float(np.mean(wser))
            g_of_t = lambda tt: np.interp(
                np.mod(np.atleast_1d(tt), wser.size * dt),
                np.arange(wser.size) * dt, wser, period=wser.size * dt)
        a_scan = area(t_scan)

        # --- the ONE physical input that turns the shape into watts ---
        # A_eff is needed first: it converts a profile power into the
        # reference intensity <I> that f_rabi is defined on.
        s = self.parent_win.state
        C_rabi, c_src = self._c_rabi()
        amp = c["amp_spots"]
        P_sim = self.fns["profile_total_power"](amp, s["win"], s["use_airy"],
                                                s["airy_factor"])
        A_eff = P_sim / max(g_ref, 1e-300) if law == "I" else float("nan")
        mode = self.cmb_norm.currentIndex()
        if mode == 2:
            f_rabi = self.sp_frabi.value() * 1e6
            I_ref = 2 * np.pi * f_rabi / C_rabi
            P_tot = I_ref * A_eff
        else:
            P_tot = (self.sp_pprof.value() * 1e-6 if mode == 0
                     else self.sp_pspot.value() * 1e-6 * len(amp))
            I_ref = P_tot / max(A_eff, 1e-300)
            f_rabi = C_rabi * I_ref / (2 * np.pi)
        for wdg, val in ((self.sp_pprof, P_tot * 1e6),
                         (self.sp_pspot, P_tot * 1e6 / max(len(amp), 1)),
                         (self.sp_frabi, f_rabi * 1e-6)):
            wdg.blockSignals(True)
            wdg.setValue(val)
            wdg.blockSignals(False)

        # --- pick the trigger; refine the maximum parabolically ---
        if self.cb_auto.isChecked():
            i = int(np.argmax(a_scan))
            y0, y1, y2 = a_scan[(i - 1) % n_scan], a_scan[i], a_scan[(i + 1) % n_scan]
            den = y0 - 2 * y1 + y2
            shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            t_opt = float((i + np.clip(shift, -1, 1)) / n_scan * T0)
            self.sp_t0.blockSignals(True)
            self.sp_t0.setValue(t_opt * 1e6)
            self.sp_t0.blockSignals(False)
        else:
            t_opt = self.sp_t0.value() * 1e-6

        delays = np.linspace(-half, half, n_pts)
        a_delay = area(t_opt + delays)

        scale = 2 * np.pi * f_rabi / max(g_ref, 1e-300)
        th_scan = a_scan * scale
        th_delay = a_delay * scale
        th0 = float(area(np.array([t_opt]))[0] * scale)

        # --- instantaneous trace over three beat periods ---
        # The trace has to SHOW the pulse: at T_p = 200 us and T0 = 16 us a
        # fixed window of three beat periods cuts off twelve thirteenths of
        # it. The span therefore follows the pulse, but is capped so that a
        # 10 ms pulse does not try to draw 600 oscillations into 8 cm.
        n_beat = t_p * f0                       # beat periods inside the pulse
        n_need = (t_opt + t_p) / T0 + 0.5
        n_show = float(min(max(3.0, n_need), TRACE_MAX_PERIODS))
        trace_cut = n_need > TRACE_MAX_PERIODS + 1e-9
        n_pt = int(np.clip(240 * n_show, 3000, 30000))
        t_trace = np.linspace(0.0, n_show * T0, n_pt)
        g_trace = np.asarray(g_of_t(t_trace), dtype=float) / max(g_ref, 1e-300)

        # --- spatial spread and excitation at a few delays ---
        n_u = 21
        d_u = np.linspace(-half, half, n_u)
        if mask is None:
            Fm = F.reshape(F.shape[0], -1)
            wpix = np.asarray(W, dtype=float).ravel()
        else:
            idx = np.flatnonzero(mask.ravel())
            Fm = F.reshape(F.shape[0], -1)[:, idx]
            wpix = np.ones(idx.size)
        wpix = wpix / wpix.sum()
        if law == "I":
            th_pix = self.fns["camera_frames_exact"](Fm, k, ph, f0, t_p,
                                                     t_opt + d_u) * t_p
        else:
            # 41 samples across the window alias badly once the pulse spans
            # more than a beat period - the sampling has to follow T_p.
            n_sub = int(np.clip(41 * max(1.0, n_beat), 41, 2001))
            th_pix = np.stack([self.fns["sqrt_area_map"](Fm, k, ph, f0, t_p, tt,
                                                         n_sub=n_sub)
                               for tt in (t_opt + d_u)])
        mu = th_pix @ wpix
        var = ((th_pix - mu[:, None]) ** 2) @ wpix
        u_delay = np.where(mu > 0, np.sqrt(var) / np.maximum(mu, 1e-300), np.nan)

        # sin^2 is non-linear: apply per point, average afterwards.
        # eta belongs to the detuning set HERE, not to whatever the main GUI
        # was last set to - otherwise the excitation curve ignores Delta.
        eta = (float(self._coef["eta"]) if getattr(self, "_coef", None)
               else float(self.parent_win.state.get("eta_ls", 0.0) or 0.0))
        gsh = np.sqrt(1.0 + eta ** 2)
        p_of = lambda th: np.sin(gsh * th * scale / 2.0) ** 2 / (1.0 + eta ** 2)
        exc_delay = p_of(th_pix) @ wpix

        if mask is None:
            me_map, _ = self.fns["time_stats_exact"](F, k, ph)
            me = me_map.ravel()
            mu_a = float(me @ wpix)
            u_avg = float(np.sqrt(((me - mu_a) ** 2) @ wpix) / mu_a) if mu_a > 0 \
                else float("nan")
        else:
            me = c["mean_exact"][mask]
            u_avg = float(np.std(me) / np.mean(me)) if np.mean(me) > 0 else float("nan")

        # --- map, point by point ---
        if law == "I":
            th_map = self.fns["pulse_area_map"](F, k, ph, f0, t_p, t_opt) * scale
        else:
            n_sub = int(np.clip(41 * max(1.0, n_beat), 41, 2001))
            th_map = self.fns["sqrt_area_map"](F, k, ph, f0, t_p, t_opt,
                                               n_sub=n_sub) * scale
        if mask is None:
            th_map = th_map.reshape(len(gy), len(gx))

        # --- how the power splits over spots and RF tones ---
        w2 = amp ** 2
        P_spot = P_tot * w2 / w2.sum()
        ax_ = self.fns["amps_from_ratio"](s["r_x"], s["N_x"]) ** 2
        ay_ = self.fns["amps_from_ratio"](s["r_y"], s["N_y"]) ** 2
        P_tx = P_tot * ax_ / ax_.sum()
        P_ty = P_tot * ay_ / ay_.sum()

        self._last = dict(
            t_scan=t_scan, th_scan=th_scan, delays=delays, th_delay=th_delay,
            th0=th0, t_opt=t_opt, T0=T0, f0=f0, t_p=t_p, f_rabi=f_rabi,
            d_u=d_u, u_delay=u_delay, exc_delay=exc_delay, eta=eta,
            th_map=th_map, mask=mask, region=region, law=law, half=half,
            u_avg=u_avg, th_ref=2 * f_rabi * t_p, gx=gx, gy=gy,
            sig_atom=sig_atom, t_trace=t_trace, g_trace=g_trace,
            n_beat=n_beat, trace_cut=trace_cut,
            atom_xy=(float(np.mean(c["centers_x"])), float(np.mean(c["centers_y"]))),
            I_ref=I_ref, A_eff=A_eff, P_tot=P_tot, P_spot=P_spot,
            P_tx=P_tx, P_ty=P_ty, C_rabi=C_rabi, c_src=c_src,
            delta=self.sp_delta.value(), T_atom=self.sp_T.value(),
            nu_trap=self.sp_nu.value(),
            gam3=(self._coef["C_scatter_3"] * I_ref
                  if getattr(self, "_coef", None) else float("nan")),
            eta_atom=(self._coef["eta"] if getattr(self, "_coef", None)
                      else float("nan")),
            # power that would make the pulse a pi pulse on the reference
            P_pi=P_tot / max(2 * f_rabi * t_p, 1e-300))
        self._redraw_only()

    def _redraw_only(self):
        if self._last is not None:
            self._draw(self.fig, self.cmb_layout.currentIndex() == 1,
                       self.cb_suptitle.isChecked())
            self.canvas.draw_idle()
            self._write_info()

    # -------------------------------------------------------- figure
    # The panels are separate methods so that the screen view, the curve PDF
    # and the map PDF are literally the same drawings, never a second version
    # that can drift out of step.
    @staticmethod
    def _t_unit(span_s):
        """Scale and label for a time axis, so 200 us and 20 ms both read well."""
        if span_s >= 1e-3:
            return 1e3, "ms"
        if span_s >= 1e-6:
            return 1e6, r"$\mu$s"
        return 1e9, "ns"

    def _panel_intensity(self, ax, L, title=True):
        end = float(L["t_trace"][-1])
        u, ul = self._t_unit(end)
        # Many oscillations in one axis turn into a solid band; thinning the
        # line keeps the envelope - which is what matters there - readable.
        lw = 0.9 if L["n_beat"] <= 6 else (0.6 if L["n_beat"] <= 20 else 0.4)
        ax.plot(L["t_trace"] * u, L["g_trace"], lw=lw, color="#1f77b4")
        ax.axhline(1.0, color="#888", lw=0.7, ls="--")
        # Repeats of the pulse are only worth drawing while it is short enough
        # that more than one fits into the window.
        n_rep = 4 if L["t_p"] < L["T0"] else 1
        for n in range(n_rep):
            t0n = (L["t_opt"] + n * L["T0"]) * u
            if t0n > end * u:
                continue
            ax.axvspan(t0n, min(t0n + L["t_p"] * u, end * u),
                       color="#d62728", alpha=0.22, lw=0)
            ax.axvline(t0n, color="#d62728", lw=1.1)
        ax.annotate(r"$t_0$", xy=(L["t_opt"] * u, ax.get_ylim()[1]),
                    xytext=(3, -11), textcoords="offset points",
                    color="#d62728", fontsize=10)
        if L["n_beat"] >= 1.0:
            ax.annotate(("pulse spans %.2f beat periods%s"
                         % (L["n_beat"], ", axis truncated"
                            if L["trace_cut"] else "")),
                        xy=(0.985, 0.06), xycoords="axes fraction",
                        ha="right", va="bottom", fontsize=8, color="#d62728")
        ax.set_xlim(0, end * u)
        ax.set_xlabel(r"$t$ [%s]" % ul)
        gname = "I" if L["law"] == "I" else r"\sqrt{I}"
        ax.set_ylabel(r"$%s(t)\,/\,\langle %s\rangle_t$" % (gname, gname))
        if title:
            ax.set_title("Profile intensity, normalised to its time average",
                         fontsize=10)
        ax.grid(alpha=0.25)

    def _panel_period(self, ax, L, title=True):
        u, ul = self._t_unit(L["T0"])
        ax.plot(L["t_scan"] * u, L["th_scan"] / np.pi, lw=0.9, color="#1f77b4")
        ax.axvline(L["t_opt"] * u, color="#d62728", lw=1.1)
        ax.set_xlim(0, L["T0"] * u)
        ax.set_xlabel(r"pulse start $t_0$ [%s]" % ul)
        ax.set_ylabel(r"$\langle\theta\rangle\,/\,\pi$")
        if title:
            ax.set_title("Mean pulse area over one beat period", fontsize=10)
        ax.grid(alpha=0.25)
        # A pulse longer than the beat period averages the modulation away
        # (the boxcar sinc kills order d at d f0 T_p = 1). The absolute area
        # then sits on a huge offset and the swing is invisible - so give the
        # panel the same relative axis the delay scan has.
        m = float(np.mean(L["th_scan"]))
        if m > 0:
            axr = ax.twinx()
            axr.set_ylim((np.array(ax.get_ylim()) / (m / np.pi) - 1.0) * 100.0)
            axr.set_ylabel("deviation from\nmean [%]", fontsize=8)
            axr.tick_params(labelsize=8)
            pp = (L["th_scan"].max() - L["th_scan"].min()) / m * 100.0
            # left-aligned: the magnifier lens sits around t_opt, usually to
            # the right, and the two would overlap.
            ax.annotate("peak-to-peak %.3g %%" % pp, xy=(0.015, 0.06),
                        xycoords="axes fraction", ha="left", va="bottom",
                        fontsize=8, color="#555")
            return axr

    def _panel_zoom(self, ax, L):
        u, ul = self._t_unit(2.0 * L["half"])
        d = L["delays"] * u
        ax.plot(d, L["th_delay"] / np.pi, lw=1.6, color="#1f77b4")
        ax.axvline(0.0, color="#888", lw=0.8)
        ax.plot([0.0], [L["th0"] / np.pi], "o", color="#d62728", ms=5)
        ax.set_xlim(d[0], d[-1])
        ax.set_xlabel(r"trigger error [%s]" % ul)
        ax.set_ylabel(r"$\langle\theta\rangle\,/\,\pi$")
        ax.grid(alpha=0.25)
        axr = ax.twinx()
        axr.set_ylim((np.array(ax.get_ylim()) / (L["th0"] / np.pi) - 1.0) * 100.0)
        axr.set_ylabel("deviation [%]")
        return axr

    def _panel_map(self, fig, ax, L, title=True):
        # The global coordinate origin is the position a tone AT THE RF OFFSET
        # would have; the whole pattern therefore sits at positive x, y. For
        # the atom map that is useless - there the natural origin is the atom
        # itself, so the local grid is plotted RELATIVE to it.
        x, y = L["gx"], L["gy"]
        atom = L["sig_atom"] is not None
        sc = 1e9 if atom else 1e6
        if atom:
            x = x - 0.5 * (x[0] + x[-1])
            y = y - 0.5 * (y[0] + y[-1])
        ext = [x[0] * sc, x[-1] * sc, y[0] * sc, y[-1] * sc]
        # theta is linear in C_rabi(Delta): at fixed power the detuning only
        # rescales the whole map, so with an autoscaled colour map the picture
        # cannot change - only the numbers on the bar. The excitation is the
        # non-linear function of it and is where Delta becomes visible.
        if self.cmb_map.currentIndex() == 1:
            gsh = np.sqrt(1.0 + L["eta"] ** 2)
            dat = np.sin(gsh * L["th_map"] / 2.0) ** 2 / (1.0 + L["eta"] ** 2)
            kw = dict(vmin=0.0, vmax=1.0)
            cb_lab = r"$p_\uparrow(r)$"
            ttl = "Excitation point by point"
        else:
            dat = L["th_map"] / np.pi
            kw = {}
            cb_lab = r"$\theta(r)\,/\,\pi$"
            ttl = "Pulse area point by point"
        im = ax.imshow(dat, extent=ext, origin="lower", cmap="magma", **kw)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cb_lab, fontsize=10)
        # A white line on the bright end of any colormap is invisible; the dark
        # stroke around it keeps the rings readable wherever they fall.
        stroke = [pe.withStroke(linewidth=2.6, foreground="black")]
        if atom:
            cx0, cy0 = 0.0, 0.0                     # atom at the origin now
            ang = np.linspace(0, 2 * np.pi, 300)
            for n in (1, 2):
                ax.plot((cx0 + n * L["sig_atom"] * np.cos(ang)) * sc,
                        (cy0 + n * L["sig_atom"] * np.sin(ang)) * sc,
                        color="white", lw=1.4, path_effects=stroke)
                ax.text(cx0 * sc, (cy0 + n * L["sig_atom"]) * sc,
                        r"$%d\sigma$" % n, color="white", fontsize=11,
                        ha="center", va="bottom", path_effects=stroke)
        elif L["mask"] is not None:
            ax.contour(x * sc, y * sc, L["mask"].astype(float), levels=[0.5],
                       colors="white", linewidths=1.4)
        if atom:
            ax.set_xlabel(r"$x - x_{\mathrm{atom}}$ [nm]")
            ax.set_ylabel(r"$y - y_{\mathrm{atom}}$ [nm]")
        else:
            ax.set_xlabel(r"x [$\mu$m]")
            ax.set_ylabel(r"y [$\mu$m]")
        if title:
            ax.set_title(ttl, fontsize=10)

    def _draw_curves(self, fig, titles=True):
        """The three time-domain panels - what goes into the curve PDF."""
        L = self._last
        fig.clear()
        gs = fig.add_gridspec(3, 1, hspace=0.55, left=0.14, right=0.88,
                              top=0.96, bottom=0.08)
        axI = fig.add_subplot(gs[0, 0])
        axP = fig.add_subplot(gs[1, 0])
        axZ = fig.add_subplot(gs[2, 0])
        self._panel_intensity(axI, L, titles)
        self._panel_period(axP, L, titles)
        self._panel_zoom(axZ, L)
        self._magnifier(fig, axP, axZ, L)
        return axI, axP, axZ

    def _draw(self, fig, a4, suptitle):
        L = self._last
        s = self.parent_win.state
        if a4:
            fig.set_size_inches(*A4_PORTRAIT)
            fig.clear()
            gs = fig.add_gridspec(4, 1, hspace=0.62,
                                  height_ratios=[0.9, 0.9, 0.9, 1.45],
                                  left=0.14, right=0.88, top=0.945, bottom=0.05)
            axI = fig.add_subplot(gs[0, 0])
            axP = fig.add_subplot(gs[1, 0])
            axZ = fig.add_subplot(gs[2, 0])
            axM = fig.add_subplot(gs[3, 0])
        else:
            fig.set_size_inches(11, 8.6)
            fig.clear()
            gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15], hspace=0.6,
                                  wspace=0.3, left=0.09, right=0.95,
                                  top=0.9, bottom=0.09)
            axI = fig.add_subplot(gs[0, 0])
            axP = fig.add_subplot(gs[1, 0])
            axZ = fig.add_subplot(gs[2, 0])
            axM = fig.add_subplot(gs[:, 1])
        self._panel_intensity(axI, L)
        self._panel_period(axP, L)
        self._panel_zoom(axZ, L)
        self._magnifier(fig, axP, axZ, L)
        self._panel_map(fig, axM, L)
        if suptitle:
            gname = "I" if L["law"] == "I" else r"\sqrt{I}"
            fig.suptitle(
                r"%d$\times$%d tones, width %.4f / %.4f MHz  |  "
                r"$\Omega/2\pi$ = %.3f MHz, $T_p$ = %.3f $\mu$s, "
                r"$\Omega\sim %s$  |  region: %s"
                % (s["N_x"], s["N_y"], s["width_x"] * 1e-6, s["width_y"] * 1e-6,
                   L["f_rabi"] * 1e-6, L["t_p"] * 1e6, gname, L["region"]),
                fontsize=9.5, y=0.985)

    def _magnifier(self, fig, ax_from, ax_to, L):
        """Lens around the zoom window plus two lines down to the panel below.

        Without it the two panels show the same curve twice and nothing says
        that one is a detail of the other."""
        # Must use the SAME unit as the panel it draws on, or the lens lands
        # somewhere else entirely once that panel switches to ms.
        u, _ = self._t_unit(L["T0"])
        # The lens only means something while the zoom really is a small
        # detail of the panel above; past about a third of the period it grows
        # over the whole axis and says nothing.
        if 2.0 * L["half"] > 0.35 * L["T0"]:
            return
        lo, hi = (L["t_opt"] - L["half"]) * u, (L["t_opt"] + L["half"]) * u
        x0, x1 = ax_from.get_xlim()
        y0, y1 = ax_from.get_ylim()
        cx = 0.5 * (lo + hi)
        wx = max(hi - lo, 0.03 * (x1 - x0))
        lens = Ellipse((cx, 0.5 * (y0 + y1)), 1.6 * wx, 0.95 * (y1 - y0),
                       fill=False, ec="#d62728", lw=1.4, zorder=5)
        lens.set_clip_on(False)          # the window may sit at the period edge
        ax_from.add_patch(lens)
        for xa, xb in ((cx - 0.8 * wx, 0.0), (cx + 0.8 * wx, 1.0)):
            fig.add_artist(ConnectionPatch(
                xyA=(np.clip(xa, x0, x1), y0), coordsA=ax_from.transData,
                xyB=(xb, 1.0), coordsB=ax_to.transAxes,
                color="#d62728", lw=0.9, ls="--", alpha=0.75))

    # ---------------------------------------------------------- text
    def _write_info(self):
        L = self._last
        d = L["delays"] * 1e6
        rel = np.abs(L["th_delay"] / L["th0"] - 1.0) if L["th0"] else np.zeros_like(d)
        edge = 100 * float(max(rel[0], rel[-1]))
        over = np.flatnonzero(rel > 0.01)
        if over.size:
            t_tol = float(np.min(np.abs(L["delays"][over])))
            tol = ("%.0f ns until 1 %% area error" % (t_tol * 1e9)
                   if t_tol < 1e-6 else
                   "%.3f us until 1 %% area error" % (t_tol * 1e6))
        else:
            tol = "below 1 %% area error over the whole range"
        mid = len(L["u_delay"]) // 2
        self.lbl_info.setText(
            "MEAN over {}:  <theta>_W(0) = {:.4f} pi  ->  excitation "
            "<sin^2(theta/2)>_W = {:.3f}{}   |   sigma_theta/<theta>_W = "
            "{:.2f} %  (time average in the same region: {:.2f} %)\n"
            "the pulse STARTS at t_0 = {:.4f} us and ends at t_0 + T_p = "
            "{:.4f} us (t_0 is the start, never the centre);  at +-{:.3f} us "
            "trigger error the mean area deviates by {:.2f} %, {}\n"
            "POWER: P_profile = {:.4f} uW -> I_ref = {:.2f} W/cm^2 (mean over "
            "time AND region) -> Omega/2pi = {:.4f} MHz  |  per spot {:.4f} uW "
            "mean, {:.4f} uW brightest  |  per RF tone x {:.4f} uW, y {:.4f} uW "
            "(strongest)  |  scattering {:.4f} % per pulse  |  a pi pulse of "
            "{:.3f} us would need {:.4f} uW".format(
                L["region"], L["th0"] / np.pi, float(L["exc_delay"][mid]),
                ", eta = %+.3f" % L["eta"] if L["eta"] else "",
                100 * float(L["u_delay"][mid]), 100 * L["u_avg"],
                L["t_opt"] * 1e6, (L["t_opt"] + L["t_p"]) * 1e6,
                L["half"] * 1e6, edge, tol,
                L["P_tot"] * 1e6, L["I_ref"] * 1e-4, L["f_rabi"] * 1e-6,
                L["P_spot"].mean() * 1e6, L["P_spot"].max() * 1e6,
                L["P_tx"].max() * 1e6, L["P_ty"].max() * 1e6,
                100 * (1 - np.exp(-L["gam3"] * L["t_p"]))
                if np.isfinite(L["gam3"]) else float("nan"),
                L["t_p"] * 1e6, L["P_pi"] * 1e6))
        broken = L["c_src"].startswith("fallback")
        self.lbl_note.setStyleSheet(
            "color: #b00; font-size: 10px; font-weight: bold;" if broken
            else "color: #555; font-size: 10px;")
        self.lbl_note.setText(
            ("DETUNING HAS NO EFFECT: rb85_raman could not be imported, so "
             "C_rabi is frozen at its -8 GHz value. " if broken else "") +
            "Omega = 2 pi f_rabi * g/<g>, <g> the mean over time AND region. A "
            "pulse at that reference has theta = {:.4f} pi. C_rabi = {:.4f} "
            "rad/s per W/m^2 at Delta = {:.1f} GHz ({}), eta = {:+.4f};  "
            "A_eff = {:.2f} um^2. "
            "Power is the one IN THE PROFILE, i.e. after both AODs."
            .format(L["th_ref"], L["C_rabi"], L["delta"], L["c_src"],
                    L["eta"], L["A_eff"] * 1e12))

    # ------------------------------------------------------- actions
    def _on_apply(self):
        if self._last is None:
            return
        s = self.parent_win.state
        s["f_rabi"] = self._last["f_rabi"]
        s["pulse_t0"] = self._last["t_opt"]
        for w, v in ((self.parent_win.sp_frabi, self._last["f_rabi"] * 1e-6),
                     (self.parent_win.sp_t0, self._last["t_opt"] * 1e6)):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        self.lbl_note.setText("trigger t_0 = %.4f us and f_rabi = %.4f MHz sent "
                              "to the main GUI." % (self._last["t_opt"] * 1e6,
                                                    self._last["f_rabi"] * 1e-6))

    def _file_tag(self):
        """File name that says beating, working point and pulse at a glance."""
        L, s = self._last, self.parent_win.state
        return ("Beating_N{}x{}_w{:.3f}um_width{:.4f}MHz_rx{:.2f}_ry{:.2f}"
                "_T0-{:.2f}us_frabi{:.3f}MHz_tp{:.3f}us_{}"
                .format(s["N_x"], s["N_y"], s["win"] * 1e6, s["width_x"] * 1e-6,
                        s["r_x"], s["r_y"], L["T0"] * 1e6, L["f_rabi"] * 1e-6,
                        L["t_p"] * 1e6,
                        L["region"].split(",")[0].replace(" ", "")))

    def _on_save(self):
        """Two PDFs plus a report - all into the image folder of the GUI.

        Neither PDF carries a title: what each panel shows is written out in
        the report next to it, and in a document it belongs in the caption.
        The map is its own file because there it wants its own place."""
        if self._last is None:
            return
        L = self._last
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        base = self._file_tag() + "_" + stamp
        out = self.parent_win.out_dir
        try:
            import matplotlib
            old_ft = matplotlib.rcParams.get("pdf.fonttype")
            matplotlib.rcParams["pdf.fonttype"] = 42          # editable text

            f_cur = Figure(figsize=(A4_PORTRAIT[0], 0.62 * A4_PORTRAIT[1]),
                           dpi=100)
            FigureCanvas(f_cur)
            self._draw_curves(f_cur, titles=False)
            p_cur = out / (base + "_curves.pdf")
            f_cur.savefig(p_cur, format="pdf", bbox_inches="tight")

            f_map = Figure(figsize=(5.6, 4.6), dpi=100)
            FigureCanvas(f_map)
            ax = f_map.add_subplot(111)
            self._panel_map(f_map, ax, L, title=False)
            map_sfx = "_excmap.pdf" if self.cmb_map.currentIndex() == 1 \
                else "_areamap.pdf"
            p_map = out / (base + map_sfx)
            f_map.savefig(p_map, format="pdf", bbox_inches="tight")

            if old_ft is not None:
                matplotlib.rcParams["pdf.fonttype"] = old_ft

            np.savetxt(out / (base + "_delayscan.txt"),
                       np.column_stack([L["delays"] * 1e6, L["th_delay"] / np.pi,
                                        (L["th_delay"] / L["th0"] - 1) * 100]),
                       header="delay_us\ttheta_over_pi_weighted_mean\t"
                              "rel_deviation_percent", delimiter="\t", fmt="%.9g")
            np.savetxt(out / (base + "_period.txt"),
                       np.column_stack([L["t_scan"] * 1e6, L["th_scan"] / np.pi]),
                       header="t0_us\ttheta_over_pi_weighted_mean",
                       delimiter="\t", fmt="%.9g")
            np.savetxt(out / (base + "_intensity.txt"),
                       np.column_stack([L["t_trace"] * 1e6, L["g_trace"]]),
                       header="t_us\tI_over_time_mean", delimiter="\t", fmt="%.9g")
            (out / (base + "_report.md")).write_text(
                self._param_text(stamp, [p_cur.name, p_map.name]),
                encoding="utf-8")
            self.lbl_note.setText(
                "saved to {} :  {}_curves.pdf, {}, _report.md and three data "
                "files".format(out, base, map_sfx[1:]))
        except Exception as exc:
            QMessageBox.critical(self, "Saving failed", str(exc))

    def _param_text(self, stamp, files=()):
        """The report as Markdown - readable as text, renders as a table."""
        L, s = self._last, self.parent_win.state
        mid = len(L["u_delay"]) // 2
        rel = np.abs(L["th_delay"] / L["th0"] - 1.0)
        gname = "I" if L["law"] == "I" else "sqrt(I)"
        build = ("single lens f = %.3f mm, waist before lens %.4f mm"
                 % (s["f_single"] * 1e3, s["win_in"] * 1e3) if s["one_lens"] else
                 "telescope f1 = %.2f mm, f2 = %.2f mm"
                 % (s["f1"] * 1e3, s["f2"] * 1e3))

        def tbl(rows):
            out = ["| quantity | value |", "|---|---|"]
            out += ["| %s | %s |" % r for r in rows]
            return out

        P = ["# Pulse area and trigger jitter", "",
             "Generated %s by `pulse_timing.py` (Beating_Multitone_GUI)." % stamp,
             "", "## Profile", ""]
        P += tbl([
            ("N_x x N_y", "%d x %d" % (s["N_x"], s["N_y"])),
            ("width_x / width_y", "%.6f / %.6f MHz"
             % (s["width_x"] * 1e-6, s["width_y"] * 1e-6)),
            ("r_x / r_y", "%.4f / %.4f" % (s["r_x"], s["r_y"])),
            ("waist", "%.4f um" % (s["win"] * 1e6)),
            ("beam profile", "%s, airy factor %.4f"
             % ("Airy" if s["use_airy"] else "Gauss", s["airy_factor"])),
            ("build", build),
            ("lambda / RF offset", "%.2f nm / %.4f MHz"
             % (s["lambda_opt"] * 1e9, s["offset"] * 1e-6)),
            ("AOD", "theta_max = 43.0 mrad, f_band = 36.0 MHz, "
                    "v_ac = %.1f m/s" % (s["lambda_opt"] * 36e6 / 43e-3)),
            ("f_0 / T_0", "%.6f kHz / %.4f us" % (L["f0"] * 1e-3, L["T0"] * 1e6)),
            ("grid / frames / periods", "%d^2 / %d / %d"
             % (s["grid_n"], s["frames_per_period"], s["n_periods"])),
            ("tone phases x [deg]", ", ".join("%.2f" % v for v in
                                              np.degrees(s["phase_x"]))),
            ("tone phases y [deg]", ", ".join("%.2f" % v for v in
                                              np.degrees(s["phase_y"]))),
        ])
        P += ["", "## Evaluation region", ""]
        reg_rows = [("region", L["region"]),
                    ("atom position (global)", "x = %.4f um, y = %.4f um"
                     % (L["atom_xy"][0] * 1e6, L["atom_xy"][1] * 1e6))]
        if L["sig_atom"] is not None:
            reg_rows += [("atom T / nu_r", "%.2f uK / %.2f kHz"
                          % (L["T_atom"], L["nu_trap"])),
                         ("sigma", "%.2f nm" % (L["sig_atom"] * 1e9)),
                         ("weighting", "Gaussian position probability density "
                                       "W(r), **not** a hard mask")]
        else:
            reg_rows += [("weighting", "hard mask, uniform (W = 1)")]
        P += tbl(reg_rows)
        P += ["", "All curves are weighted **spatial means** "
              "`<x>_W = sum_r W(r) x(r) / sum_r W(r)`; only the map is point by "
              "point. The excitation is averaged after the sin^2, not before.",
              "", "## Pulse", ""]
        P += tbl([
            ("Omega/2pi (reference)", "%.6f MHz, on the time averaged "
                                      "intensity in the region"
             % (L["f_rabi"] * 1e-6)),
            ("coupling law", "Omega ~ %s" % gname),
            ("T_p", "%.6f us (pi pulse at %.6f us)"
             % (L["t_p"] * 1e6, 0.5e6 / L["f_rabi"])),
            ("pulse start t_0", "%.6f us in the cycle (t_0 is the START)"
             % (L["t_opt"] * 1e6)),
            ("pulse end t_0 + T_p", "%.6f us" % ((L["t_opt"] + L["t_p"]) * 1e6)),
            ("**mean area** <theta>_W(0)", "**%.6f pi**" % (L["th0"] / np.pi)),
            ("area at the reference", "%.6f pi" % L["th_ref"]),
            ("sigma_theta / <theta>_W", "%.4f %% (spatial spread in the region)"
             % (100 * L["u_delay"][mid])),
            ("same for the time average", "%.4f %%" % (100 * L["u_avg"])),
            ("<sin^2(theta/2)>_W", "%.6f (per point, then averaged)"
             % L["exc_delay"][mid]),
            ("eta (light shift)", "%.6f" % L["eta"]),
            ("deviation at +-%.3f us" % (L["half"] * 1e6),
             "%.4f %%" % (100 * max(rel[0], rel[-1]))),
        ])
        P += ["", "## Power (in the profile, i.e. after both AODs)", ""]
        P += tbl([
            ("Delta", "%.3f GHz" % L["delta"]),
            ("C_rabi", "%.4f rad/s per W/m^2 (%s)" % (L["C_rabi"], L["c_src"])),
            ("I_ref", "%.6f W/cm^2 (mean over time AND region)"
             % (L["I_ref"] * 1e-4)),
            ("A_eff", "%.4f um^2 (= integral I dA / <I>)" % (L["A_eff"] * 1e12)),
            ("**P_profile**", "**%.8f mW**" % (L["P_tot"] * 1e3)),
            ("P per spot", "%.8f uW mean, %.8f uW brightest, %.8f uW weakest"
             % (L["P_spot"].mean() * 1e6, L["P_spot"].max() * 1e6,
                L["P_spot"].min() * 1e6)),
            ("P per RF tone x", "%.8f uW mean, %.8f uW strongest"
             % (L["P_tx"].mean() * 1e6, L["P_tx"].max() * 1e6)),
            ("P per RF tone y", "%.8f uW mean, %.8f uW strongest"
             % (L["P_ty"].mean() * 1e6, L["P_ty"].max() * 1e6)),
            ("scattering rate F=3", "%.1f 1/s" % L["gam3"]),
            ("scattering per pulse", "%.4f %%"
             % (100 * (1 - np.exp(-L["gam3"] * L["t_p"])))),
            ("eta (from Delta)", "%.4f -> contrast <= %.4f"
             % (L["eta_atom"], 1.0 / (1.0 + L["eta_atom"] ** 2))),
            ("P for a pi pulse of T_p", "%.6f uW" % (L["P_pi"] * 1e6)),
            ("not included", "AOD diffraction efficiency, optics transmission, "
                             "intermodulation"),
        ])
        P += ["", "## Figures", "",
              "Neither PDF carries a title; this is what the panels show.", "",
              "**`_curves.pdf`**, three panels top to bottom:", "",
              "1. Profile intensity `I(t)` divided by its own time average, over "
              "three beat periods. The red line marks the pulse **start** t_0, "
              "the shading the pulse duration T_p.",
              "2. Mean pulse area `<theta>/pi` versus the pulse start t_0, over "
              "one full beat period. The red ellipse is the magnifier for "
              "panel 3.",
              "3. The same curve zoomed around the chosen t_0. x is the trigger "
              "error, the right axis the deviation from the area at zero error.",
              "",
              "**`_map.pdf`**: pulse area `theta(r)/pi` point by point, no "
              "averaging. The white rings are 1 and 2 sigma of the atomic "
              "position distribution (or the outline of the hard mask). In the "
              "atom weighted case the axes are RELATIVE to the atom, so the "
              "origin is the atom; the global origin would be the position of "
              "a tone at the bare RF offset, which is why the pattern as a "
              "whole sits at positive coordinates.",
              "", "## Files", ""]
        P += ["- `%s`" % f for f in files] or ["- -"]
        P += ["- `%s_period.txt`, `%s_delayscan.txt`, `%s_intensity.txt` "
              "(data columns)" % ((files[0].rsplit("_curves", 1)[0],) * 3
                                  if files else ("<base>",) * 3),
              "", "## Summary line from the GUI", "", "```",
              self.lbl_info.text(), "```", ""]
        return "\n".join(P)
