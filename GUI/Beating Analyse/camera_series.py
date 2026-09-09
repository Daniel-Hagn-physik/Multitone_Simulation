"""Camera frame series over one beat period.

What the window shows: what a camera with a finite exposure really records
when the trigger delay is stepped over one fundamental period - top row the
raw frames, bottom row the deviation from the time average. The latter is what
one evaluates in the lab: the background drops out and the beating stands
there as a signed pattern.

The frames are NOT averaged from the cube, they are computed exactly. I(t) is
a trigonometric polynomial in exp(2 pi i f_0 t),

    I(r,t) = sum_d C_d(r) exp(2 pi i d f_0 t) ,

and an exposure t_exp starting at t_0 is a boxcar on it:

    I_cam(r, t_0) = sum_d C_d(r) * sinc(d f_0 t_exp)
                                 * exp(2 pi i d f_0 (t_0 + t_exp/2))

with sinc(z) = sin(pi z)/(pi z). This holds for ARBITRARY t_0 and t_exp - no
rounding to a time grid, no aliasing. The C_d come from an FFT over the beat
orders, i.e. without any loop over spot pairs.

The phases are taken fresh from the main GUI on every draw. Change the phases
there, press 'Recompute', then 'Redraw' here - that is the workflow this
window was built for.

NO SPATIAL AVERAGING happens here at all: the image IS the spatial resolution.
Only the two figures of merit below the plot average - the per-pixel swing
(median and p90 over the plateau pixels) and the total light in the plateau.
"""

import datetime

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPushButton, QComboBox, QGroupBox, QGridLayout, QMessageBox
)

A4_LANDSCAPE = (11.69, 8.27)          # inch - a frame series is wide by nature


class CameraSeriesDialog(QDialog):
    """Modeless window; the main GUI stays usable."""

    def __init__(self, parent, frame_fn):
        super().__init__(parent)
        self.setWindowTitle("Camera frame series over one beat period")
        self.resize(1500, 800)
        self.parent_win = parent
        self.frame_fn = frame_fn          # camera_frames_exact() of the main GUI
        self._last = None
        self._T0_seen = None              # period the times were set up for

        root = QVBoxLayout(self)
        root.addWidget(self._group_inputs())

        self.fig = Figure(figsize=(15, 6.4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)

        self.lbl_info = QLabel("-")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size: 11px;")
        root.addWidget(self.lbl_info)

        row = QHBoxLayout()
        self.btn_draw = QPushButton("Redraw")
        self.btn_draw.clicked.connect(self.redraw)
        self.btn_save = QPushButton("Save PDF (LaTeX, A4)")
        self.btn_save.clicked.connect(self._on_save)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_draw)
        row.addStretch(1)
        row.addWidget(self.btn_save)
        row.addWidget(btn_close)
        root.addLayout(row)

        self.redraw()

    # ------------------------------------------------------------
    def _group_inputs(self):
        g = QGroupBox("Acquisition")
        lay = QGridLayout(g)
        c = getattr(self.parent_win, "cache", {}) or {}
        T0 = c.get("T0", 100e-6)
        t_exp0 = self.parent_win.state.get("t_exp") or 20e-6
        # A series needs several frames per period. If the exposure set in the
        # main GUI does not fit (say 20 us at a period of 13 us), start from a
        # fifth of the period instead.
        if T0 and np.isfinite(T0) and t_exp0 > 0.5 * T0:
            t_exp0 = T0 / 5.0

        self.sp_texp = QDoubleSpinBox()
        self.sp_texp.setRange(0.0001, 100000.0); self.sp_texp.setDecimals(4)
        self.sp_texp.setSingleStep(5.0); self.sp_texp.setSuffix(" us")
        self.sp_texp.setValue(t_exp0 * 1e6)
        self.sp_texp.setKeyboardTracking(False)
        self.sp_texp.setToolTip(
            "Exposure time per frame. It damps beat order d by\n"
            "sinc(d*f_0*t_exp): slow components survive, fast ones are averaged\n"
            "away. At t_exp = T_0 every frame is the time average and the\n"
            "bottom row vanishes - that is the built-in check.")
        self.sp_texp.valueChanged.connect(self._on_texp_changed)

        self.sp_step = QDoubleSpinBox()
        self.sp_step.setRange(0.0001, 100000.0); self.sp_step.setDecimals(4)
        self.sp_step.setSingleStep(5.0); self.sp_step.setSuffix(" us")
        self.sp_step.setValue(t_exp0 * 1e6)
        self.sp_step.setKeyboardTracking(False)
        self.sp_step.setToolTip(
            "Step of the trigger delay from frame to frame. Equal to the\n"
            "exposure means gapless; smaller means oversampled - equally\n"
            "feasible in the lab, because every frame is its own shot.")
        self.sp_step.valueChanged.connect(lambda _: self._sync_n())

        self.sp_n = QSpinBox()
        self.sp_n.setRange(1, 12)
        self.sp_n.setValue(max(1, min(12, int(round(T0 / max(t_exp0, 1e-12))))))
        self.sp_n.setToolTip("Number of frames. Default: as many as fit into one\n"
                             "fundamental period.")

        self.cmb_row2 = QComboBox()
        self.cmb_row2.addItems(["deviation from the time average",
                                "ratio to the time average",
                                "no second row"])
        self.cb_common = QCheckBox("common colour scale")
        self.cb_common.setChecked(True)
        self.cb_common.setToolTip(
            "Off: every frame is scaled to its own maximum. That looks more\n"
            "contrasty but makes the frames incomparable - exactly the mistake\n"
            "one does not want to make at the setup.")
        self.cb_suptitle = QCheckBox("title inside the figure")
        self.cb_suptitle.setChecked(True)
        self.cb_suptitle.setToolTip("Switch off for LaTeX: the caption belongs\n"
                                    "in \\caption{}, not in the graphic.")

        rows = [("exposure", self.sp_texp), ("trigger delay step", self.sp_step),
                ("frames", self.sp_n), ("bottom row", self.cmb_row2),
                ("", self.cb_common), ("", self.cb_suptitle)]
        for i, (name, w) in enumerate(rows):
            lay.addWidget(QLabel(name), 0, 2 * i)
            lay.addWidget(w, 0, 2 * i + 1)
        self.lbl_period = QLabel("-")
        self.lbl_period.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_period, 1, 0, 1, 2 * len(rows))
        return g

    def _on_texp_changed(self, _):
        self.sp_step.blockSignals(True)
        self.sp_step.setValue(self.sp_texp.value())
        self.sp_step.blockSignals(False)
        self._sync_n()

    def _apply_defaults(self, T0):
        """Set the times up for a new fundamental period.

        Only when T_0 has changed - a deliberately chosen series (a shorter
        excerpt, oversampled steps) must not be thrown away by a 'Redraw'."""
        t_exp = self.parent_win.state.get("t_exp") or 0.0
        if not (0 < t_exp <= 0.5 * T0):
            t_exp = T0 / 5.0
        for w, v in ((self.sp_texp, t_exp * 1e6), (self.sp_step, t_exp * 1e6),
                     (self.sp_n, max(1, min(12, int(round(T0 / t_exp)))))):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        self._T0_seen = T0

    def _sync_n(self):
        c = getattr(self.parent_win, "cache", {}) or {}
        T0 = c.get("T0", None)
        if T0 and np.isfinite(T0):
            step = self.sp_step.value() * 1e-6
            self.sp_n.setValue(max(1, min(12, int(round(T0 / max(step, 1e-12))))))

    # ------------------------------------------------------------
    def redraw(self):
        c = getattr(self.parent_win, "cache", {}) or {}
        if not c or "F_stack" not in c:
            QMessageBox.information(self, "Nothing to draw",
                                    "Press 'Recompute' in the main GUI first.")
            return
        f0, T0 = c["f0"], c["T0"]
        if not (f0 > 0 and np.isfinite(T0)):
            QMessageBox.information(
                self, "No beat period",
                "The difference frequencies have no common divisor - there is "
                "no beat period for a series to run over.")
            return
        if self._T0_seen is None or abs(T0 - self._T0_seen) > 1e-3 * T0:
            self._apply_defaults(T0)

        t_exp = self.sp_texp.value() * 1e-6
        step = self.sp_step.value() * 1e-6
        n = self.sp_n.value()
        t0 = np.arange(n) * step

        frames = self.frame_fn(c["F_stack"], c["k_orders"], c["phases"],
                               f0, t_exp, t0)
        self._last = dict(frames=frames, mean=c["mean_exact"],
                          plateau=c["plateau"], t0=t0, t_exp=t_exp, step=step,
                          f0=f0, T0=T0, x=c["x"], y=c["y"])
        self._draw(self.fig, False)
        self.canvas.draw_idle()
        self._write_info()

    def _draw(self, fig, a4):
        L = self._last
        frames, mean = L["frames"], L["mean"]
        n = len(frames)
        x, y = L["x"], L["y"]
        ext = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]
        mode = self.cmb_row2.currentIndex()
        n_rows = 1 if mode == 2 else 2
        fig.clear()
        fig.set_size_inches(*(A4_LANDSCAPE if a4 else (15, 6.4)))
        axes = fig.subplots(n_rows, n, squeeze=False)
        vmax = float(frames.max()) if self.cb_common.isChecked() else None
        norm = max(float(mean.max()), 1e-300)

        for i in range(n):
            a = axes[0][i]
            a.imshow(frames[i] / norm, extent=ext, origin="lower", cmap="inferno",
                     vmin=0.0, vmax=(vmax / norm) if vmax else None)
            a.set_title(r"$t_0$ = %.1f - %.1f $\mu$s"
                        % (L["t0"][i] * 1e6, (L["t0"][i] + L["t_exp"]) * 1e6),
                        fontsize=9)
            a.set_xticks([]); a.set_yticks([])
            if mode == 2:
                continue
            b = axes[1][i]
            if mode == 0:
                d = (frames[i] - mean) / norm * 100.0
                lim = float(np.max(np.abs(frames - mean))) / norm * 100.0
                im = b.imshow(d, extent=ext, origin="lower", cmap="coolwarm",
                              vmin=-lim, vmax=lim)
            else:
                d = frames[i] / np.maximum(mean, norm * 1e-3)
                im = b.imshow(d, extent=ext, origin="lower", cmap="coolwarm",
                              vmin=0.0, vmax=2.0)
            b.set_xticks([]); b.set_yticks([])
        axes[0][0].set_ylabel("camera frame", fontsize=9)
        if mode != 2:
            axes[1][0].set_ylabel("frame $-$ time average" if mode == 0
                                  else "frame / time average", fontsize=9)
            cb = fig.colorbar(im, ax=[axes[1][j] for j in range(n)],
                              fraction=0.02, pad=0.01)
            cb.set_label(r"% of max$\langle I\rangle$" if mode == 0 else "ratio",
                         fontsize=9)

        if self.cb_suptitle.isChecked():
            s = self.parent_win.state
            build = ("single lens f = %.1f mm, $w_{in}$ = %.3f mm"
                     % (s["f_single"] * 1e3, s["win_in"] * 1e3)) if s["one_lens"] \
                else ("telescope %.0f/%.0f mm" % (s["f1"] * 1e3, s["f2"] * 1e3))
            fig.suptitle(
                r"%d$\times$%d tones, width %.4f / %.4f MHz  |  %s, "
                r"$w_0$ = %.2f $\mu$m  |  $T_0$ = %.2f $\mu$s, exposure "
                r"%.2f $\mu$s  -  no spatial averaging, this is the image itself"
                % (s["N_x"], s["N_y"], s["width_x"] * 1e-6, s["width_y"] * 1e-6,
                   build, s["win"] * 1e6, L["T0"] * 1e6, L["t_exp"] * 1e6),
                fontsize=9.5)

    def _write_info(self):
        L = self._last
        frames, mean, plateau = L["frames"], L["mean"], L["plateau"]
        n = len(frames)
        if plateau.any() and n > 1:
            rel = (frames.max(0) - frames.min(0)) / np.maximum(mean, 1e-300)
            med = float(np.median(rel[plateau]))
            p90 = float(np.percentile(rel[plateau], 90))
            tot = np.array([float(fr[plateau].sum()) for fr in frames])
            tot = tot / max(tot.mean(), 1e-300)
            glob = float(tot.max() - tot.min())
        else:
            med = p90 = glob = float("nan")
        supp = abs(float(np.sinc(L["f0"] * L["t_exp"])))
        head = ("only one frame - a swing needs at least two"
                if not np.isfinite(med) else
                "per-pixel swing (max-min)/mean:  median {:.0f} %, p90 {:.0f} % "
                "(MEDIAN over the plateau pixels).  Total light in the plateau "
                "varies by {:.1f} % ({})".format(
                    100 * med, 100 * p90, 100 * glob,
                    "pure redistribution, no brightness change" if glob < 0.05
                    else "visible as a brightness change too"))
        self.lbl_info.setText(
            head + ".\nThe exposure leaves {:.0f} % of the fundamental "
            "standing; {} frames of {:.2f} us at a step of {:.2f} us cover "
            "{:.0f} % of one period.".format(
                100 * supp, n, L["t_exp"] * 1e6, L["step"] * 1e6,
                100 * min(1.0, n * L["step"] / L["T0"])))
        self.lbl_period.setText(
            "Fundamental period T_0 = %.3f us (f_0 = %.4f kHz).  Exposure = T_0 "
            "gives exactly the time average - the bottom row must then vanish."
            % (L["T0"] * 1e6, L["f0"] * 1e-3))

    # ------------------------------------------------------------
    def _on_save(self):
        if self._last is None:
            return
        s = self.parent_win.state
        L = self._last
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name = ("CameraSeries_N{}x{}_texp{:.0f}us_step{:.0f}us_{}"
                .format(s["N_x"], s["N_y"], L["t_exp"] * 1e6, L["step"] * 1e6,
                        stamp))
        out = self.parent_win.out_dir
        try:
            import matplotlib
            old_ft = matplotlib.rcParams.get("pdf.fonttype")
            matplotlib.rcParams["pdf.fonttype"] = 42
            tmp = Figure(figsize=A4_LANDSCAPE, dpi=100)
            FigureCanvas(tmp)
            self._draw(tmp, True)
            tmp.savefig(out / (name + ".pdf"), format="pdf", bbox_inches="tight")
            if old_ft is not None:
                matplotlib.rcParams["pdf.fonttype"] = old_ft
            rows = [
                ("N_x x N_y", "%d x %d" % (s["N_x"], s["N_y"])),
                ("width_x / width_y", "%.6f / %.6f MHz"
                 % (s["width_x"] * 1e-6, s["width_y"] * 1e-6)),
                ("r_x / r_y", "%.4f / %.4f" % (s["r_x"], s["r_y"])),
                ("build", "single lens f = %.3f mm, w_in = %.4f mm"
                 % (s["f_single"] * 1e3, s["win_in"] * 1e3) if s["one_lens"]
                 else "telescope f1 = %.2f mm, f2 = %.2f mm"
                 % (s["f1"] * 1e3, s["f2"] * 1e3)),
                ("waist (focus)", "%.4f um" % (s["win"] * 1e6)),
                ("beam profile", "%s, airy factor %.4f"
                 % ("Airy" if s["use_airy"] else "Gauss", s["airy_factor"])),
                ("lambda / RF offset", "%.2f nm / %.4f MHz"
                 % (s["lambda_opt"] * 1e9, s["offset"] * 1e-6)),
                ("f_0 / T_0", "%.6f kHz / %.4f us"
                 % (L["f0"] * 1e-3, L["T0"] * 1e6)),
                ("exposure / delay step", "%.4f / %.4f us"
                 % (L["t_exp"] * 1e6, L["step"] * 1e6)),
                ("frames", "%d" % len(L["frames"])),
                ("tone phases x [deg]", ", ".join("%.2f" % v for v in
                                                  np.degrees(s["phase_x"]))),
                ("tone phases y [deg]", ", ".join("%.2f" % v for v in
                                                  np.degrees(s["phase_y"]))),
            ]
            md = ["# Camera frame series", "",
                  "Generated %s by `camera_series.py` "
                  "(Beating_Multitone_GUI)." % stamp, "",
                  "## Parameters", "",
                  "| quantity | value |", "|---|---|"]
            md += ["| %s | %s |" % r for r in rows]
            md += ["", "## Figure", "",
                   "Top row: the raw camera frames. Bottom row: the deviation "
                   "from the time average - that is what one evaluates, "
                   "because the background drops out. **No spatial averaging "
                   "anywhere**; the image is the spatial resolution.", "",
                   "## Summary line from the GUI", "", "```",
                   self.lbl_info.text(), "```",
                   "", "## Files", "", "- `%s.pdf`" % name, ""]
            (out / (name + "_report.md")).write_text("\n".join(md),
                                                     encoding="utf-8")
            self.lbl_period.setText("saved: " + name + ".pdf (+ _report.md)")
        except Exception as exc:
            QMessageBox.critical(self, "Saving failed", str(exc))
