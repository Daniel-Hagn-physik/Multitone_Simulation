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

----------------------------------------------------------------------
Comparison with a measured image
----------------------------------------------------------------------
The second half of this window puts the predicted image next to the one the
camera actually took. A measured frame is loaded from file (png, tif, bmp,
npy, csv - whatever the camera software wrote), brought into the same
orientation, cut to the profile and plotted beside the simulated frame for the
same exposure.

What is deliberately NOT drawn: position axes. Neither the magnification at
the camera nor the pixel the profile sits on is known well enough to put a
micrometre scale under the measurement, and a wrong scale is worse than
none. Both panels are therefore cut to the same fraction of the profile size
and shown without ticks - what is compared is the SHAPE, not the position.
Each panel is normalised to its own maximum, because the absolute scale
(diffraction efficiency, attenuation, camera gain) is not the subject here.

Which simulated frame to compare against is a real question: an untriggered
camera catches an unknown trigger delay t_0. Hence the choice between the
time average (right when the exposure covers the whole period) and the
best-matching frame, found by maximising the correlation over the frames of
the series - and over a small pixel shift, so that a slightly off-centre cut
does not spoil the number.

The figure is saved at 16 cm width with 10 pt type (kern/plotstil.py), i.e.
it goes into LaTeX with \\includegraphics[width=\\textwidth] unscaled.
"""

import datetime
import io
from pathlib import Path

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPushButton, QComboBox, QGroupBox, QGridLayout, QMessageBox,
    QFileDialog
)

# The physics: kern/beating_physik.py (kern is a package next to this file).
from kern import beating_physik as phys
# One width, one font size for every saved PDF - see kern/plotstil.py.
from kern.plotstil import FIG_WIDTH_IN, style_figure

A4_LANDSCAPE = (11.69, 8.27)          # inch - a frame series is wide by nature

IMAGE_FILTER = ("camera image (*.png *.tif *.tiff *.bmp *.jpg *.jpeg *.pgm"
                " *.npy *.npz *.csv *.txt *.dat);;all files (*)")


# ============================================================
# Loading and preparing a measured frame - numpy only
# ============================================================

def load_measured_image(path):
    """A measured camera frame as a 2-D float array, row 0 at the BOTTOM.

    Takes whatever the camera software wrote: png/tif/bmp/jpg through PIL
    (16 bit included), .npy/.npz, or a text table. A colour image is reduced
    to luminance, a stack of frames is averaged - both are reported back in
    `note` so that it does not happen silently.

    Image files count rows from the top, the simulation plots with
    origin='lower'. The array is flipped once here, so that from this point on
    BOTH follow the same convention and 'flip y' in the window means a real
    camera orientation, not a plotting detail.
    """
    path = Path(path)
    ext = path.suffix.lower()
    notes = []
    sat_max = None

    if ext == ".npy":
        a = np.load(str(path))
    elif ext == ".npz":
        z = np.load(str(path))
        key = list(z.keys())[0]
        a = z[key]
        notes.append("npz, array '%s'" % key)
    elif ext in (".csv", ".txt", ".dat"):
        txt = path.read_text(errors="ignore")
        for dl in (",", ";", "\t", None):
            try:
                a = np.loadtxt(io.StringIO(txt), delimiter=dl)
                break
            except Exception:
                a = None
        if a is None:
            raise ValueError("could not read %s as a number table" % path.name)
    else:
        a = None
        try:
            from PIL import Image
            im = Image.open(str(path))
            a = np.asarray(im)
            notes.append("PIL, mode %s" % im.mode)
        except Exception:
            pass
        if a is None:
            try:
                import tifffile
                a = tifffile.imread(str(path))
                notes.append("tifffile")
            except Exception:
                pass
        if a is None:
            import matplotlib.image as mpimg
            a = mpimg.imread(str(path))
            notes.append("matplotlib.imread")

    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        sat_max = float(np.iinfo(a.dtype).max)
    if a.ndim == 3:
        if a.shape[-1] in (3, 4):
            w = np.array([0.2126, 0.7152, 0.0722])
            a = np.tensordot(a[..., :3].astype(float), w, axes=([-1], [0]))
            notes.append("colour -> luminance")
        else:
            notes.append("stack of %d frames -> averaged" % a.shape[0])
            a = a.astype(float).mean(0)
    if a.ndim != 2:
        raise ValueError("%s is not a 2-D image (shape %s)"
                         % (path.name, a.shape))
    a = a.astype(float)

    # Saturation: an image that runs into the top of the range cannot be
    # compared to a simulation in shape - the plateau is then flat by the
    # camera, not by the optics. Worth a word, so it is counted here.
    frac_sat = float("nan")
    if sat_max:
        frac_sat = float(np.mean(a >= 0.999 * sat_max))
    elif a.max() > 0:
        frac_sat = float(np.mean(a >= 0.999 * a.max()))

    a = a[::-1]                        # file top row -> plot top row
    return a, ", ".join(notes), frac_sat


def _box_mean(a, k):
    """Moving average over a k x k window, k odd, edges extended.

    Only for finding the profile: a single hot pixel must not decide where the
    cut-out sits. Done with cumulative sums, so no scipy is needed.
    """
    k = int(max(1, k)) | 1
    if k == 1:
        return np.asarray(a, float)
    pad = k // 2
    b = np.pad(np.asarray(a, float), pad, mode="edge")
    c = np.cumsum(np.cumsum(b, 0), 1)
    c = np.pad(c, ((1, 0), (1, 0)))
    H, W = a.shape
    return (c[k:k + H, k:k + W] - c[0:H, k:k + W]
            - c[k:k + H, 0:W] + c[0:H, 0:W]) / float(k * k)


def crop_to_profile(img, frac=0.25, margin=1.5, aspect=None):
    """Window around the illuminated part: (row slice, col slice).

    The box is the extent of everything above `frac` of the (smoothed) peak,
    blown up by `margin` and centred on that extent. `aspect` = width/height
    forces the shape of the window, so that the simulated and the measured
    panel show the same kind of field and neither is stretched.
    """
    img = np.asarray(img, float)
    H, W = img.shape
    sm = _box_mean(img, max(3, (min(H, W) // 40) * 2 + 1))
    pk = float(np.percentile(sm, 99.9))
    mask = sm > frac * pk if pk > 0 else np.zeros_like(sm, bool)
    if not mask.any():
        return slice(0, H), slice(0, W)
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    rc = 0.5 * (rows[0] + rows[-1])
    cc = 0.5 * (cols[0] + cols[-1])
    hh = 0.5 * (rows[-1] - rows[0] + 1) * margin
    hw = 0.5 * (cols[-1] - cols[0] + 1) * margin
    if aspect:
        hh = max(hh, hw / aspect)
        hw = aspect * hh
    hh = min(hh, H / 2.0)
    hw = min(hw, W / 2.0)
    r0 = int(round(min(max(rc - hh, 0), H - 2 * hh)))
    c0 = int(round(min(max(cc - hw, 0), W - 2 * hw)))
    return (slice(r0, min(H, r0 + int(round(2 * hh)))),
            slice(c0, min(W, c0 + int(round(2 * hw)))))


def resample_nn(img, shape):
    """Nearest neighbour onto `shape` - for correlating two different grids.

    Nearest neighbour and not something smoother on purpose: it invents no
    intermediate values, and for a correlation coefficient over some 10^4
    pixels the difference is in the third decimal.
    """
    H, W = img.shape
    ri = np.clip(((np.arange(shape[0]) + 0.5) * H / shape[0]).astype(int), 0, H - 1)
    ci = np.clip(((np.arange(shape[1]) + 0.5) * W / shape[1]).astype(int), 0, W - 1)
    return img[np.ix_(ri, ci)]


def _corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / d) if d > 0 else float("nan")


def best_alignment(sim, meas, rmax=6):
    """Correlation of the two panels, maximised over an integer pixel shift.

    The cut-out comes from a centre of gravity, i.e. it can be off by a few
    pixels; without this, that alone would cost correlation and the number
    would say more about the cut than about the physics. Returns
    (r, shift_row, shift_col) with the shift applied to the measurement.
    """
    best = (-2.0, 0, 0)
    for dr in range(-rmax, rmax + 1):
        for dc in range(-rmax, rmax + 1):
            r = _corr(sim, np.roll(np.roll(meas, dr, 0), dc, 1))
            if r > best[0]:
                best = (r, dr, dc)
    return best


class CameraSeriesDialog(QDialog):
    """Modeless window; the main GUI stays usable."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Camera frame series over one beat period")
        self.resize(1500, 800)
        self.parent_win = parent
        self._last = None
        self._T0_seen = None              # period the times were set up for
        self._meas = None                 # measured frame, bottom row first
        self._meas_path = None
        self._meas_note = ""
        self._meas_sat = float("nan")
        self._cmp = None                  # comparison window, created on demand

        root = QVBoxLayout(self)
        root.addWidget(self._group_inputs())

        self.fig = Figure(figsize=(15, 6.4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)

        self.lbl_info = QLabel("-")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size: 11px;")
        root.addWidget(self.lbl_info)
        root.addWidget(self._group_measure())

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

        frames = phys.camera_frames_exact(c["F_stack"], c["k_orders"], c["phases"],
                                          f0, t_exp, t0)
        self._last = dict(frames=frames, mean=c["mean_exact"],
                          plateau=c["plateau"], t0=t0, t_exp=t_exp, step=step,
                          f0=f0, T0=T0, x=c["x"], y=c["y"])
        self._draw(self.fig, False)
        self.canvas.draw_idle()
        self._write_info()
        self._refresh_panel_combo()
        if self._cmp is not None and self._cmp.isVisible():
            self._cmp.refresh()

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

    # ============================================================
    # Comparison with a measured image
    # ============================================================
    def _group_measure(self):
        g = QGroupBox("Comparison with a measured image")
        lay = QGridLayout(g)

        btn_open = QPushButton("Choose image file ...")
        btn_open.clicked.connect(self._on_browse)
        btn_open.setToolTip(
            "The frame the camera took: png, tif, bmp, jpg, npy/npz or a\n"
            "number table (csv/txt). A colour image is reduced to luminance,\n"
            "a stack of frames is averaged.")
        self.lbl_file = QLabel("no file chosen")
        self.lbl_file.setStyleSheet("color: #555;")

        self.cmb_rot = QComboBox()
        self.cmb_rot.addItems(["0 deg", "90 deg", "180 deg", "270 deg"])
        self.cmb_rot.setToolTip(
            "Camera orientation. Which way the sensor sits relative to the\n"
            "x and y axis of the simulation is not known from the data - it is\n"
            "set here, by looking at the picture.")
        self.cb_flipx = QCheckBox("mirror x")
        self.cb_flipy = QCheckBox("mirror y")
        for w in (self.cmb_rot, self.cb_flipx, self.cb_flipy):
            w.setToolTip(self.cmb_rot.toolTip())
            (w.currentIndexChanged if isinstance(w, QComboBox)
             else w.stateChanged).connect(lambda *_: self._refresh_cmp())

        self.sp_bg = QDoubleSpinBox()
        self.sp_bg.setRange(0.0, 50.0); self.sp_bg.setDecimals(1)
        self.sp_bg.setValue(5.0); self.sp_bg.setSuffix(" %")
        self.sp_bg.setToolTip(
            "Background: this percentile of the measured image is subtracted\n"
            "and negative values are clipped. Dark current and stray light are\n"
            "an offset the simulation does not have - without this, every\n"
            "comparison of shape is pulled towards grey.")
        self.sp_bg.valueChanged.connect(lambda *_: self._refresh_cmp())

        self.cb_crop = QCheckBox("cut to the profile")
        self.cb_crop.setChecked(True)
        self.cb_crop.setToolTip(
            "Cuts BOTH panels to the illuminated part, with the same margin\n"
            "and the same shape of window. Only then does the profile fill the\n"
            "same fraction of both panels and the shapes can be compared by\n"
            "eye - the magnification at the camera is not known.")
        self.sp_margin = QDoubleSpinBox()
        self.sp_margin.setRange(1.0, 6.0); self.sp_margin.setDecimals(2)
        self.sp_margin.setSingleStep(0.1); self.sp_margin.setValue(1.50)
        self.sp_margin.setPrefix("x "); self.sp_margin.setToolTip(
            "Size of the window in units of the profile extent.")
        for w in (self.cb_crop, self.sp_margin):
            (w.stateChanged if isinstance(w, QCheckBox)
             else w.valueChanged).connect(lambda *_: self._refresh_cmp())

        self.cmb_panel = QComboBox()
        self.cmb_panel.setToolTip(
            "Which simulated frame goes next to the measurement.\n"
            "'time average': right when the exposure covers a whole period.\n"
            "'best match': the frame of the series with the highest\n"
            "correlation - the honest choice for an UNTRIGGERED camera, where\n"
            "the trigger delay t_0 of the shot is unknown.")
        self.cmb_panel.currentIndexChanged.connect(lambda *_: self._refresh_cmp())

        self.cb_align = QCheckBox("align (pixel shift)")
        self.cb_align.setChecked(True)
        self.cb_align.setToolTip(
            "Allows a shift of up to 6 pixels when correlating, so that the\n"
            "number says something about the shape and not about how well the\n"
            "centre of gravity hit the centre. The shift is reported.")
        self.sp_fw = QDoubleSpinBox()
        self.sp_fw.setRange(0.30, 1.00); self.sp_fw.setDecimals(2)
        self.sp_fw.setSingleStep(0.05); self.sp_fw.setValue(0.85)
        self.sp_fw.setSuffix(" x linewidth")
        self.sp_fw.setToolTip(
            "How wide the figure will stand in the document. It is SAVED at\n"
            "exactly that width, so \\includegraphics[width=0.85\\linewidth]\n"
            "does not scale it and the 10 pt inside are 10 pt on paper.\n"
            "Saving at full width and letting LaTeX shrink it to 0.85 is what\n"
            "makes the labels too small - that is this field, not a font size.")
        self.sp_fw.valueChanged.connect(lambda *_: self._refresh_cmp())

        self.cb_labels = QCheckBox("captions in the figure")
        self.cb_labels.setToolTip(
            "Off (the default): two bare panels, simulation LEFT, measurement\n"
            "RIGHT - the order says which is which, and what the figure shows\n"
            "belongs in \\caption{} in LaTeX, not in the graphic.\n"
            "On: a title over each panel and a headline with the parameters.")
        self.cb_cuts = QCheckBox("with cuts")
        self.cb_cuts.setToolTip(
            "Adds a row with the cut through the middle of each panel, over\n"
            "the panel width as the coordinate - no micrometres, because the\n"
            "scale at the camera is not known. This is the quantitative part\n"
            "of the comparison.")
        for w in (self.cb_align, self.cb_labels, self.cb_cuts):
            w.stateChanged.connect(lambda *_: self._refresh_cmp())

        btn_cmp = QPushButton("Show comparison")
        btn_cmp.clicked.connect(self._open_compare)
        btn_cmp.setToolTip("Opens the figure; the PDF is saved from there.")

        r0 = [("", btn_open), ("", self.lbl_file)]
        for i, (name, w) in enumerate(r0):
            lay.addWidget(w, 0, i)
        lay.setColumnStretch(1, 1)
        r1 = [("rotation", self.cmb_rot), ("", self.cb_flipx),
              ("", self.cb_flipy), ("background", self.sp_bg)]
        for i, (name, w) in enumerate(r1):
            if name:
                lay.addWidget(QLabel(name), 1, 2 * i)
            lay.addWidget(w, 1, 2 * i + 1)
        r2 = [("", self.cb_crop), ("window", self.sp_margin),
              ("simulated panel", self.cmb_panel), ("", self.cb_align),
              ("", self.cb_labels), ("", self.cb_cuts),
              ("width in the document", self.sp_fw)]
        for i, (name, w) in enumerate(r2):
            if name:
                lay.addWidget(QLabel(name), 2, 2 * i)
            lay.addWidget(w, 2, 2 * i + 1)
        lay.addWidget(btn_cmp, 0, 2, 1, 4)
        return g

    def _refresh_panel_combo(self):
        """Keeps the list of panels in step with the series."""
        if self._last is None:
            return
        want = ["time average", "best match to the measurement"]
        want += ["frame %d  (t_0 = %.1f us)" % (i + 1, t * 1e6)
                 for i, t in enumerate(self._last["t0"])]
        cur = self.cmb_panel.currentText()
        if [self.cmb_panel.itemText(i)
                for i in range(self.cmb_panel.count())] == want:
            return
        self.cmb_panel.blockSignals(True)
        self.cmb_panel.clear()
        self.cmb_panel.addItems(want)
        idx = want.index(cur) if cur in want else 1
        self.cmb_panel.setCurrentIndex(idx)
        self.cmb_panel.blockSignals(False)

    def _on_browse(self):
        start = self._meas_path or getattr(self.parent_win, "out_dir", None)
        fn, _ = QFileDialog.getOpenFileName(
            self, "Measured camera frame", str(start or ""), IMAGE_FILTER)
        if not fn:
            return
        try:
            img, note, sat = load_measured_image(fn)
        except Exception as exc:
            QMessageBox.critical(self, "Could not read the image", str(exc))
            return
        self._meas = img
        self._meas_path = Path(fn)
        self._meas_note = note
        self._meas_sat = sat
        self.lbl_file.setText("%s   (%d x %d px%s)"
                              % (self._meas_path.name, img.shape[1],
                                 img.shape[0], ", " + note if note else ""))
        self._open_compare()

    def _refresh_cmp(self):
        if self._cmp is not None and self._cmp.isVisible():
            self._cmp.refresh()

    # ------------------------------------------------------------
    def _meas_oriented(self):
        """The measured frame with rotation, mirroring and background removed."""
        a = np.array(self._meas, float)
        k = self.cmb_rot.currentIndex()
        if k:
            a = np.rot90(a, k)
        if self.cb_flipx.isChecked():
            a = a[:, ::-1]
        if self.cb_flipy.isChecked():
            a = a[::-1, :]
        q = self.sp_bg.value()
        if q > 0:
            a = a - float(np.percentile(a, q))
        return np.maximum(a, 0.0)

    def compare_data(self):
        """Everything the comparison figure and its report need.

        Both panels are cut with the same rule and the same relative margin,
        the measurement additionally to the shape of the simulated window, so
        that nothing has to be stretched. The correlation is computed on the
        simulation grid.
        """
        if self._last is None or self._meas is None:
            return None
        L = self._last
        frames, mean = L["frames"], L["mean"]
        margin = self.sp_margin.value()

        if self.cb_crop.isChecked():
            rs, cs = crop_to_profile(mean, margin=margin)
        else:
            rs, cs = slice(None), slice(None)
        sim_mean = mean[rs, cs]
        sim_frames = np.array([f[rs, cs] for f in frames])
        aspect = sim_mean.shape[1] / float(sim_mean.shape[0])

        meas = self._meas_oriented()
        if self.cb_crop.isChecked():
            mr, mc = crop_to_profile(meas, margin=margin, aspect=aspect)
            meas = meas[mr, mc]

        # correlation on the simulation grid
        m_on_sim = resample_nn(meas, sim_mean.shape)
        rmax = 6 if self.cb_align.isChecked() else 0
        cands = [("time average", sim_mean, float("nan"))]
        cands += [("frame %d" % (i + 1), sim_frames[i], L["t0"][i])
                  for i in range(len(sim_frames))]
        scored = []
        for name, panel, t0 in cands:
            r, dr, dc = best_alignment(panel, m_on_sim, rmax)
            scored.append((name, panel, t0, r, dr, dc))
        best = max(scored[1:] or scored, key=lambda s: s[3])

        k = self.cmb_panel.currentIndex()
        if k <= 0:
            chosen = scored[0]
            label = r"simulation, time average"
        elif k == 1:
            chosen = best
            label = (r"simulation, best match ($t_0$ = %.1f $\mu$s)"
                     % (chosen[2] * 1e6))
        else:
            chosen = scored[min(k - 1, len(scored) - 1)]
            label = (r"simulation, $t_0$ = %.1f $\mu$s" % (chosen[2] * 1e6))

        name, panel, t0, r, dr, dc = chosen
        meas_shown = np.roll(np.roll(m_on_sim, dr, 0), dc, 1) if (dr or dc) \
            else m_on_sim

        # figures of merit in the plateau of each panel, same definition
        def _u(img):
            m = img > 0.5 * img.max() if img.max() > 0 else np.zeros_like(img, bool)
            return (phys.uniformity_of(img, m) if m.any() else float("nan"),
                    int(m.sum()))
        u_sim, n_sim = _u(panel)
        u_meas, n_meas = _u(meas_shown)

        mode = self.cmb_panel.currentText()
        sim_name = name if k <= 0 or k > 1 else (
            "%s -> %s (t_0 = %.1f us)" % (mode, name, t0 * 1e6))
        return dict(sim=panel, meas=meas_shown, meas_raw=meas, label=label,
                    sim_name=sim_name, t0=t0, r=r, dr=dr, dc=dc,
                    r_best=best[3], t0_best=best[2], r_avg=scored[0][3],
                    u_sim=u_sim, u_meas=u_meas, n_sim=n_sim, n_meas=n_meas,
                    t_exp=L["t_exp"], T0=L["T0"], f0=L["f0"],
                    margin=margin, sat=self._meas_sat,
                    file=self._meas_path, note=self._meas_note)

    # ------------------------------------------------------------
    def draw_compare(self, fig, D):
        """The figure itself: two panels, no position axes.

        Each panel normalised to its own maximum - the absolute scale is set
        by diffraction efficiency, attenuation and camera gain and says
        nothing about the profile. One colour bar for both, because both run
        from 0 to 1.
        """
        cuts = self.cb_cuts.isChecked()
        fig.clear()
        if cuts:
            gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.52])
        else:
            gs = fig.add_gridspec(1, 2)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])

        # Fixed order, and it is the only thing that names the panels when the
        # captions are off: SIMULATION LEFT, MEASUREMENT RIGHT.
        S = D["sim"] / max(float(D["sim"].max()), 1e-300)
        M = D["meas"] / max(float(D["meas"].max()), 1e-300)
        lab = self.cb_labels.isChecked()
        for a, img, ttl in ((ax0, S, D["label"]), (ax1, M, "measurement")):
            im = a.imshow(img, origin="lower", cmap="inferno", vmin=0.0, vmax=1.0,
                          interpolation="nearest")
            if lab:
                a.set_title(ttl)
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_linewidth(0.6)
        cb = fig.colorbar(im, ax=[ax0, ax1], orientation="horizontal",
                          fraction=0.055, pad=0.03, aspect=45)
        # Each panel is normalised to its own maximum - that belongs in the
        # caption, not on the colour bar.
        cb.set_label(r"$I/I_{\mathrm{max}}$")
        cb.outline.set_linewidth(0.6)

        if cuts:
            ny, nx = S.shape
            u = (np.arange(nx) + 0.5) / nx - 0.5
            v = (np.arange(ny) + 0.5) / ny - 0.5
            axh = fig.add_subplot(gs[1, 0])
            axv = fig.add_subplot(gs[1, 1])
            axh.plot(u, S[ny // 2, :], "-", lw=1.2, label="simulation")
            axh.plot(u, M[ny // 2, :], "-", lw=1.2, label="measurement")
            axv.plot(v, S[:, nx // 2], "-", lw=1.2)
            axv.plot(v, M[:, nx // 2], "-", lw=1.2)
            for a, t in ((axh, "cut through the middle, horizontal"),
                         (axv, "vertical")):
                a.set_xlabel("position / panel width")
                # Headroom above 1 so that the legend has somewhere to sit
                # that no curve can reach - both are normalised to 1.
                a.set_ylim(0, 1.35); a.set_xlim(u[0], u[-1])
                a.grid(alpha=0.25, lw=0.5)
                a.set_title(t, fontsize=9)
            axh.set_ylabel(r"$I/I_{\mathrm{max}}$")
            axh.legend(frameon=False, loc="upper center", ncol=2,
                       borderaxespad=0.2, handlelength=1.6)

        if lab:
            s = self.parent_win.state
            build = ("single lens f = %.1f mm, $w_{in}$ = %.2f mm"
                     % (s["f_single"] * 1e3, s["win_in"] * 1e3)) \
                if s["one_lens"] else ("telescope %.0f/%.0f mm"
                                       % (s["f1"] * 1e3, s["f2"] * 1e3))
            # Two lines on purpose: at 16 cm and 10 pt a single line of this
            # would run off both sides of the figure.
            fig.suptitle((r"%d$\times$%d tones, %s" "\n"
                          r"exposure %.1f $\mu$s, $T_0$ = %.1f $\mu$s, "
                          r"correlation $r$ = %.3f")
                         % (s["N_x"], s["N_y"], build, D["t_exp"] * 1e6,
                            D["T0"] * 1e6, D["r"]))
        return fig

    def compare_text(self, D):
        sh = ("" if not (D["dr"] or D["dc"])
              else "  shifted by (%+d, %+d) px for the alignment." % (D["dr"], D["dc"]))
        sat = ("" if not np.isfinite(D["sat"]) else
               "  %.2f %% of the measured pixels sit at the top of the range%s."
               % (100 * D["sat"],
                  " - SATURATED, the plateau is then flat by the camera"
                  if D["sat"] > 0.01 else ""))
        return ("Correlation with the shown panel r = %.3f (time average "
                "%.3f, best frame %.3f at t_0 = %.1f us).%s\n"
                "U = std/mean above half maximum: simulation %.1f %% over "
                "%d px, measurement %.1f %% over %d px.%s"
                % (D["r"], D["r_avg"], D["r_best"], D["t0_best"] * 1e6, sh,
                   100 * D["u_sim"], D["n_sim"], 100 * D["u_meas"],
                   D["n_meas"], sat))

    def fig_size(self):
        """Width and height of the comparison figure in inch.

        The width is the one the figure will have IN THE DOCUMENT - saving at
        full text width and letting \\includegraphics shrink it to 0.85 would
        shrink the 10 pt type to 8.5 pt with it. See kern/plotstil.py.
        """
        w = FIG_WIDTH_IN * self.sp_fw.value()
        return w, w * (0.86 if self.cb_cuts.isChecked() else 0.56)

    # ------------------------------------------------------------
    def _open_compare(self):
        if self._last is None:
            QMessageBox.information(self, "Nothing to compare",
                                    "Press 'Redraw' first.")
            return
        if self._meas is None:
            self._on_browse()
            return
        if self._cmp is None:
            self._cmp = _CompareWindow(self)
        self._cmp.show(); self._cmp.raise_(); self._cmp.activateWindow()
        self._cmp.refresh()

    def save_compare(self):
        """The figure at 16 cm width with 10 pt type, plus a small report."""
        D = self.compare_data()
        if D is None:
            return None
        s = self.parent_win.state
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name = ("CompareMeasurement_N{}x{}_texp{:.0f}us_{}"
                .format(s["N_x"], s["N_y"], D["t_exp"] * 1e6, stamp))
        out = self.parent_win.out_dir
        import matplotlib
        old_ft = matplotlib.rcParams.get("pdf.fonttype")
        matplotlib.rcParams["pdf.fonttype"] = 42
        try:
            w_in, h_in = self.fig_size()
            tmp = Figure(figsize=(w_in, h_in), dpi=100,
                         constrained_layout=True)
            FigureCanvas(tmp)
            self.draw_compare(tmp, D)
            style_figure(tmp)
            tmp.savefig(out / (name + ".pdf"), format="pdf")
        finally:
            if old_ft is not None:
                matplotlib.rcParams["pdf.fonttype"] = old_ft
        rows = [
            ("measured file", str(D["file"])),
            ("read as", D["note"] or "-"),
            ("orientation", "rotation %s, mirror x %s, mirror y %s"
             % (self.cmb_rot.currentText(), self.cb_flipx.isChecked(),
                self.cb_flipy.isChecked())),
            ("background removed", "%.1f %% percentile" % self.sp_bg.value()),
            ("window", ("profile extent x %.2f, same rule for both panels"
                        % D["margin"]) if self.cb_crop.isChecked()
             else "full frames"),
            ("simulated panel", D["sim_name"]),
            ("N_x x N_y", "%d x %d" % (s["N_x"], s["N_y"])),
            ("width_x / width_y", "%.6f / %.6f MHz"
             % (s["width_x"] * 1e-6, s["width_y"] * 1e-6)),
            ("build", "single lens f = %.3f mm, w_in = %.4f mm"
             % (s["f_single"] * 1e3, s["win_in"] * 1e3) if s["one_lens"]
             else "telescope f1 = %.2f mm, f2 = %.2f mm"
             % (s["f1"] * 1e3, s["f2"] * 1e3)),
            ("waist (focus)", "%.4f um" % (s["win"] * 1e6)),
            ("exposure", "%.4f us" % (D["t_exp"] * 1e6)),
            ("f_0 / T_0", "%.6f kHz / %.4f us" % (D["f0"] * 1e-3, D["T0"] * 1e6)),
            ("tone phases x [deg]", ", ".join("%.2f" % v for v in
                                              np.degrees(s["phase_x"]))),
            ("tone phases y [deg]", ", ".join("%.2f" % v for v in
                                              np.degrees(s["phase_y"]))),
        ]
        md = ["# Predicted vs. measured camera frame", "",
              "Generated %s by `camera_series.py` (Beating_Multitone_GUI)."
              % stamp, "",
              "Saved at %.2f cm width with %.0f pt type - goes into LaTeX with "
              "`\\includegraphics[width=%.2f\\linewidth]` unscaled, so the "
              "10 pt in the figure are 10 pt on paper."
              % (self.fig_size()[0] * 2.54, 10, self.sp_fw.value()), "",
              "## Parameters", "", "| quantity | value |", "|---|---|"]
        md += ["| %s | %s |" % r for r in rows]
        md += ["", "## Numbers", "", "```", self.compare_text(D), "```", "",
               "## What is not in the figure", "",
               "No position axes. Neither the magnification at the camera nor "
               "the pixel the profile sits on is known well enough for a "
               "micrometre scale, and a wrong scale is worse than none. Both "
               "panels are cut to the same fraction of the profile size and "
               "each is normalised to its own maximum: what is compared is the "
               "shape.", "", "Left panel: simulation. Right panel: the "
               "measured frame. Without captions in the graphic that order is "
               "what names them - say it in the LaTeX caption.", "",
               "## Files", "", "- `%s.pdf`" % name, ""]
        (out / (name + "_report.md")).write_text("\n".join(md), encoding="utf-8")
        return name


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


class _CompareWindow(QDialog):
    """Just the frame around the comparison figure: look at it, save it.

    Deliberately a window of its own and not another panel in the series
    window: the figure is meant to go into the document at 16 cm, and at that
    aspect ratio it would be squeezed in next to the series.
    """

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.setWindowTitle("Predicted vs. measured camera frame")
        self.resize(1000, 760)
        root = QVBoxLayout(self)
        self.fig = Figure(figsize=(FIG_WIDTH_IN, 0.86 * FIG_WIDTH_IN), dpi=110,
                          constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)
        self.lbl = QLabel("-")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("font-size: 11px;")
        root.addWidget(self.lbl)
        row = QHBoxLayout()
        b_re = QPushButton("Redraw")
        b_re.clicked.connect(self.refresh)
        b_sv = QPushButton("Save PDF (LaTeX, 16 cm)")
        b_sv.clicked.connect(self._save)
        b_cl = QPushButton("Close")
        b_cl.clicked.connect(self.close)
        row.addWidget(b_re); row.addStretch(1); row.addWidget(b_sv)
        row.addWidget(b_cl)
        root.addLayout(row)

    def refresh(self):
        D = self.owner.compare_data()
        if D is None:
            self.lbl.setText("no measured image loaded")
            return
        # Same proportions as the PDF, so that what is judged here is what
        # ends up in the document.
        self.fig.set_size_inches(*self.owner.fig_size())
        self.owner.draw_compare(self.fig, D)
        self.canvas.draw_idle()
        self.lbl.setText(self.owner.compare_text(D))

    def _save(self):
        try:
            name = self.owner.save_compare()
        except Exception as exc:
            QMessageBox.critical(self, "Saving failed", str(exc))
            return
        if name:
            self.lbl.setText("saved: " + name + ".pdf (+ _report.md)\n"
                             + self.lbl.text())
