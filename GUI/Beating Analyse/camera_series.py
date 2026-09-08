"""Bildserie einer Kamera ueber eine Beating-Periode.

Was das Fenster zeigt: was eine Kamera mit endlicher Belichtung wirklich
aufnimmt, wenn man den Trigger-Delay in Schritten ueber eine Grundperiode
durchfaehrt - obere Reihe die Rohbilder, untere Reihe die Abweichung vom
Zeitmittel. Letztere ist das, was man im Labor auswertet: der Untergrund faellt
heraus und die Schwebung steht als Vorzeichenmuster da.

Die Bilder werden NICHT aus dem Wuerfel gemittelt, sondern exakt gerechnet.
I(t) ist ein trigonometrisches Polynom in exp(2 pi i f_0 t),

    I(r,t) = sum_d C_d(r) exp(2 pi i d f_0 t) ,

und eine Belichtung t_exp ab t_0 ist darauf ein Boxcar:

    I_cam(r, t_0) = sum_d C_d(r) * sinc(d f_0 t_exp)
                                 * exp(2 pi i d f_0 (t_0 + t_exp/2))

mit sinc(z) = sin(pi z)/(pi z). Das gilt fuer BELIEBIGE t_0 und t_exp - keine
Rundung auf ein Zeitraster, kein Aliasing. Die C_d kommen aus einer FFT ueber
die Beat-Ordnungen, also ohne Schleife ueber Spotpaare.

Die Phasen kommen bei jedem Zeichnen frisch aus dem Haupt-GUI. Phasen dort
umstellen, 'Recompute', hier 'Neu zeichnen' - das ist der Arbeitsablauf, fuer
den das Fenster gebaut ist.
"""

import datetime

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPushButton, QComboBox, QGroupBox, QGridLayout, QMessageBox
)


class CameraSeriesDialog(QDialog):
    """Nicht-modales Fenster; das Haupt-GUI bleibt bedienbar."""

    def __init__(self, parent, frame_fn):
        super().__init__(parent)
        self.setWindowTitle("Kamera-Bildserie ueber eine Beating-Periode")
        self.resize(1500, 780)
        self.parent_win = parent
        self.frame_fn = frame_fn          # camera_frames_exact() des Haupt-GUI
        self._last = None
        self._T0_seen = None               # Periode, auf die eingestellt wurde

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
        self.btn_draw = QPushButton("Neu zeichnen")
        self.btn_draw.clicked.connect(self.redraw)
        self.btn_save = QPushButton("Als PDF speichern")
        self.btn_save.clicked.connect(self._on_save)
        btn_close = QPushButton("Schliessen")
        btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_draw)
        row.addStretch(1)
        row.addWidget(self.btn_save)
        row.addWidget(btn_close)
        root.addLayout(row)

        self.redraw()

    # ------------------------------------------------------------
    def _group_inputs(self):
        g = QGroupBox("Aufnahme")
        lay = QGridLayout(g)
        c = getattr(self.parent_win, "cache", {}) or {}
        T0 = c.get("T0", 100e-6)
        t_exp0 = self.parent_win.state.get("t_exp") or 20e-6
        # Eine Serie braucht mehrere Bilder pro Periode. Passt die im
        # Haupt-GUI eingestellte Belichtung nicht dazu (z.B. 20 us bei einer
        # Periode von 13 us), wird auf ein Fuenftel der Periode aufgesetzt.
        if T0 and np.isfinite(T0) and t_exp0 > 0.5 * T0:
            t_exp0 = T0 / 5.0

        self.sp_texp = QDoubleSpinBox()
        self.sp_texp.setRange(0.0001, 100000.0); self.sp_texp.setDecimals(4)
        self.sp_texp.setSingleStep(5.0); self.sp_texp.setSuffix(" us")
        self.sp_texp.setValue(t_exp0 * 1e6)
        self.sp_texp.setKeyboardTracking(False)
        self.sp_texp.setToolTip(
            "Belichtungszeit pro Bild. Sie daempft die Beat-Ordnung d mit\\n"
            "sinc(d*f_0*t_exp): langsame Anteile bleiben stehen, schnelle\\n"
            "werden weggemittelt. Bei t_exp = T_0 ist jedes Bild das\\n"
            "Zeitmittel und die untere Reihe wird null - das ist die Probe.")
        self.sp_texp.valueChanged.connect(self._on_texp_changed)

        self.sp_step = QDoubleSpinBox()
        self.sp_step.setRange(0.0001, 100000.0); self.sp_step.setDecimals(4)
        self.sp_step.setSingleStep(5.0); self.sp_step.setSuffix(" us")
        self.sp_step.setValue(t_exp0 * 1e6)
        self.sp_step.setKeyboardTracking(False)
        self.sp_step.setToolTip(
            "Schrittweite des Trigger-Delays von Bild zu Bild. Gleich der\\n"
            "Belichtung heisst: lueckenlos aneinander. Kleiner heisst\\n"
            "ueberlappend abgetastet - im Labor genauso machbar, weil jedes\\n"
            "Bild ein eigener Schuss ist.")
        self.sp_step.valueChanged.connect(lambda _: self._sync_n())

        self.sp_n = QSpinBox()
        self.sp_n.setRange(1, 12)
        self.sp_n.setValue(max(1, min(12, int(round(T0 / max(t_exp0, 1e-12))))))
        self.sp_n.setToolTip("Anzahl Bilder. Voreinstellung: so viele, wie eine\\n"
                             "Grundperiode fassen.")

        self.cmb_row2 = QComboBox()
        self.cmb_row2.addItems(["Abweichung vom Zeitmittel",
                                "Verhaeltnis zum Zeitmittel",
                                "keine zweite Reihe"])
        self.cb_common = QCheckBox("gemeinsame Farbskala")
        self.cb_common.setChecked(True)
        self.cb_common.setToolTip(
            "Aus: jedes Bild wird einzeln voll ausgesteuert. Das sieht\\n"
            "kontrastreicher aus, macht die Bilder aber untereinander\\n"
            "unvergleichbar - genau der Fehler, den man am Messplatz nicht\\n"
            "machen will.")

        rows = [("Belichtung", self.sp_texp), ("Schritt Trigger-Delay", self.sp_step),
                ("Bilder", self.sp_n), ("Untere Reihe", self.cmb_row2),
                ("", self.cb_common)]
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
        """Zeiten auf eine neue Grundperiode aufsetzen.

        Nur wenn sich T_0 geaendert hat - eine vom Benutzer absichtlich
        gesetzte Serie (kuerzerer Ausschnitt, ueberlappende Abtastung) soll
        ein 'Neu zeichnen' nicht zurueckwerfen."""
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
            QMessageBox.information(self, "Nichts zu zeichnen",
                                    "Im Haupt-GUI erst 'Recompute' druecken.")
            return
        f0, T0 = c["f0"], c["T0"]
        if not (f0 > 0 and np.isfinite(T0)):
            QMessageBox.information(
                self, "Keine Periode",
                "Die Differenzfrequenzen haben keinen gemeinsamen Teiler - es "
                "gibt keine Beating-Periode, ueber die eine Serie laufen "
                "koennte.")
            return

        if self._T0_seen is None or abs(T0 - self._T0_seen) > 1e-3 * T0:
            self._apply_defaults(T0)

        t_exp = self.sp_texp.value() * 1e-6
        step = self.sp_step.value() * 1e-6
        n = self.sp_n.value()
        t0 = np.arange(n) * step

        frames = self.frame_fn(c["F_stack"], c["k_orders"], c["phases"],
                               f0, t_exp, t0)
        mean = c["mean_exact"]
        plateau = c["plateau"]
        self._last = dict(frames=frames, mean=mean, t0=t0, t_exp=t_exp,
                          step=step, f0=f0, T0=T0)

        x, y = c["x"], c["y"]
        ext = [x[0] * 1e6, x[-1] * 1e6, y[0] * 1e6, y[-1] * 1e6]
        mode = self.cmb_row2.currentIndex()
        n_rows = 1 if mode == 2 else 2
        self.fig.clear()
        axes = self.fig.subplots(n_rows, n, squeeze=False)
        vmax = float(frames.max()) if self.cb_common.isChecked() else None
        norm = max(float(mean.max()), 1e-300)

        for i in range(n):
            a = axes[0][i]
            a.imshow(frames[i] / norm, extent=ext, origin="lower", cmap="inferno",
                     vmin=0.0, vmax=(vmax / norm) if vmax else None)
            a.set_title("t$_0$ = %.1f - %.1f $\\mu$s"
                        % (t0[i] * 1e6, (t0[i] + t_exp) * 1e6), fontsize=9)
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
        axes[0][0].set_ylabel("Kamerabild", fontsize=9)
        if mode != 2:
            axes[1][0].set_ylabel("Bild $-$ Zeitmittel" if mode == 0
                                  else "Bild / Zeitmittel", fontsize=9)
            cb = self.fig.colorbar(im, ax=[axes[1][j] for j in range(n)],
                                   fraction=0.02, pad=0.01)
            cb.set_label("% von max$\\langle I\\rangle$" if mode == 0
                         else "Verhaeltnis", fontsize=9)

        s = self.parent_win.state
        build = ("eine Linse f = %.1f mm, $w_{in}$ = %.3f mm"
                 % (s["f_single"] * 1e3, s["win_in"] * 1e3)) if s["one_lens"] \
            else ("Teleskop %.0f/%.0f mm" % (s["f1"] * 1e3, s["f2"] * 1e3))
        self.fig.suptitle(
            "%d$\\times$%d Toene, width %.4f / %.4f MHz  |  %s, $w_0$ = %.2f $\\mu$m  |  "
            "T$_0$ = %.2f $\\mu$s, Belichtung %.2f $\\mu$s"
            % (s["N_x"], s["N_y"], s["width_x"] * 1e-6, s["width_y"] * 1e-6,
               build, s["win"] * 1e6, T0 * 1e6, t_exp * 1e6), fontsize=10)
        self.canvas.draw_idle()

        # Kennzahlen - das, wonach man das Bild am Messplatz beurteilt
        if plateau.any() and n > 1:
            rel = (frames.max(0) - frames.min(0)) / np.maximum(mean, 1e-300)
            med = float(np.median(rel[plateau]))
            p90 = float(np.percentile(rel[plateau], 90))
            tot = np.array([float(fr[plateau].sum()) for fr in frames])
            tot = tot / max(tot.mean(), 1e-300)
            glob = float(tot.max() - tot.min())
        else:
            med = p90 = glob = float("nan")
        supp = abs(float(np.sinc(f0 * t_exp)))
        head = ("nur ein Bild - fuer einen Hub braucht es mindestens zwei"
                if not np.isfinite(med) else
                "Hub pro Pixel im Plateau (max-min)/Mittel:  Median {:.0f} %,  "
                "p90 {:.0f} %.   Gesamtlicht im Plateau schwankt um {:.1f} % "
                "({})".format(100 * med, 100 * p90, 100 * glob,
                              "reine Umverteilung, keine Helligkeits"
                              "aenderung" if glob < 0.05 else
                              "auch als Helligkeitsaenderung sichtbar"))
        self.lbl_info.setText(
            head + ".\nDie Belichtung laesst von der Grundschwingung {:.0f} % "
            "stehen; {} Bilder a {:.2f} us im Abstand {:.2f} us decken {:.0f} % "
            "einer Periode ab.".format(
                100 * supp, n, t_exp * 1e6, step * 1e6,
                100 * min(1.0, n * step / T0)))
        self.lbl_period.setText(
            "Grundperiode T_0 = %.3f us (f_0 = %.4f kHz).   Belichtung = T_0 "
            "liefert exakt das Zeitmittel - die untere Reihe muss dann "
            "verschwinden." % (T0 * 1e6, f0 * 1e-3))

    # ------------------------------------------------------------
    def _on_save(self):
        if self._last is None:
            return
        s = self.parent_win.state
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name = ("Kameraserie_N{}x{}_texp{:.0f}us_step{:.0f}us_{}"
                .format(s["N_x"], s["N_y"], self._last["t_exp"] * 1e6,
                        self._last["step"] * 1e6, stamp))
        out = self.parent_win.out_dir
        try:
            self.fig.savefig(out / (name + ".pdf"), format="pdf",
                             bbox_inches="tight")
            lines = [stamp, "",
                     "N_x x N_y = %d x %d" % (s["N_x"], s["N_y"]),
                     "width_x = %.6f MHz, width_y = %.6f MHz"
                     % (s["width_x"] * 1e-6, s["width_y"] * 1e-6),
                     "build: " + ("single lens f = %.3f mm, w_in = %.4f mm"
                                  % (s["f_single"] * 1e3, s["win_in"] * 1e3)
                                  if s["one_lens"] else
                                  "telescope f1 = %.2f mm, f2 = %.2f mm"
                                  % (s["f1"] * 1e3, s["f2"] * 1e3)),
                     "waist (focus) = %.4f um" % (s["win"] * 1e6),
                     "T_0 = %.4f us (f_0 = %.6f kHz)"
                     % (self._last["T0"] * 1e6, self._last["f0"] * 1e-3),
                     "exposure = %.4f us, delay step = %.4f us, frames = %d"
                     % (self._last["t_exp"] * 1e6, self._last["step"] * 1e6,
                        len(self._last["t0"])),
                     "phases x [deg] = " + ", ".join(
                         "%.2f" % v for v in np.degrees(s["phase_x"])),
                     "phases y [deg] = " + ", ".join(
                         "%.2f" % v for v in np.degrees(s["phase_y"])),
                     "", self.lbl_info.text()]
            (out / (name + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            self.lbl_period.setText("gespeichert: " + name + ".pdf (+ .txt)")
        except Exception as exc:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(exc))
