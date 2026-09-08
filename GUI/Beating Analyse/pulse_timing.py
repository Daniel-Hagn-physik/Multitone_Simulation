"""Pulsflaeche und Trigger-Jitter.

DIE FRAGE
Ein Rechteckpuls der Laenge T_p wird auf ein Zeitfenster hoher Intensitaet
getriggert. Wie gross ist die akkumulierte Rabi-Flaeche

    theta(r, t_0) = int_{t_0}^{t_0+T_p} Omega(r,t) dt ,

und wie stark aendert sie sich, wenn der Trigger um bis zu +-Delta danebenliegt?

DER PUNKT, DEN MAN LEICHT UEBERSIEHT
Ein Puls ist derselbe Boxcar in der Zeit wie eine Kamerabelichtung, nur
tausendmal kuerzer. Auf die Beat-Ordnung d wirkt er als sinc(d*f_0*T_p). Bei
T_p = 1 us und f_0 = 10 kHz ist sinc(0.012) = 0.9999 fuer die Grundschwingung
und immer noch 0.976 bei 120 kHz - der Puls mittelt die Schwebung also NICHT
weg, er tastet sie ab. Deshalb ist die Flaeche empfindlich auf den Trigger,
und zwar auf der Zeitskala der SCHNELLEN Beats (einige us), nicht auf der der
Grundperiode (100 us). Genau das macht den Scan ueber +-1 us interessant.

NORMIERUNG
Omega ist bis auf eine Konstante durch das Profil festgelegt; die Konstante
kommt aus der angenommenen Rabi-Frequenz. Hier gilt

    Omega(r,t) = 2*pi*f_rabi * g(r,t) / <g> ,   g = I  bzw.  sqrt(I),

mit <g> dem Mittel ueber Zeit UND Bereich - also die Kalibrierung, die man im
Labor machen wuerde: f_rabi ist die Rabi-Frequenz, die man auf dem
zeitgemittelten Profil misst. Ein Puls der Laenge T_p auf dieser
Referenzintensitaet hat dann exakt die Flaeche theta = 2*pi*f_rabi*T_p; bei
f_rabi = 1 MHz und T_p = 1 us also 2*pi, einen 2pi-Puls. Der pi-Puls waere
0.5 us lang - dafuer gibt es den Knopf.

AM OPTIMUM IST DIE KURVE FLACH
Der automatisch gesuchte Trigger ist das Maximum der mittleren Flaeche. Dort
ist die Ableitung nach dem Delay null, der Jitter geht also nur in zweiter
Ordnung ein. Das ist der eigentliche Grund, warum man auf ein Maximum triggert
und nicht auf eine Flanke.
"""

import datetime

import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, QSpinBox,
    QCheckBox, QPushButton, QComboBox, QGroupBox, QGridLayout, QMessageBox
)


class PulseTimingDialog(QDialog):
    """Nicht-modales Fenster; das Haupt-GUI bleibt bedienbar."""

    def __init__(self, parent, fns):
        super().__init__(parent)
        self.setWindowTitle("Pulsflaeche und Trigger-Jitter")
        self.resize(1450, 860)
        self.parent_win = parent
        self.fns = fns                    # Rechenfunktionen des Haupt-GUI
        self._last = None

        root = QVBoxLayout(self)
        root.addWidget(self._group_inputs())

        self.fig = Figure(figsize=(14, 7.4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        root.addWidget(self.canvas, 1)

        self.lbl_info = QLabel("-")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("font-size: 11px;")
        root.addWidget(self.lbl_info)

        row = QHBoxLayout()
        self.btn_draw = QPushButton("Neu rechnen")
        self.btn_draw.clicked.connect(self.recompute)
        self.btn_apply = QPushButton("Trigger ins Haupt-GUI uebernehmen")
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_save = QPushButton("Als PDF speichern")
        self.btn_save.clicked.connect(self._on_save)
        btn_close = QPushButton("Schliessen")
        btn_close.clicked.connect(self.close)
        row.addWidget(self.btn_draw)
        row.addStretch(1)
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_save)
        row.addWidget(btn_close)
        root.addLayout(row)

        self.recompute()

    # ------------------------------------------------------------
    def _dspin(self, val, lo, hi, dec, step, suffix):
        w = QDoubleSpinBox()
        w.setRange(lo, hi); w.setDecimals(dec); w.setSingleStep(step)
        w.setSuffix(" " + suffix); w.setValue(val); w.setKeyboardTracking(False)
        return w

    def _group_inputs(self):
        g = QGroupBox("Puls und Trigger")
        lay = QGridLayout(g)
        s = self.parent_win.state

        self.sp_frabi = self._dspin((s.get("f_rabi") or 1e6) * 1e-6,
                                    0.001, 100.0, 4, 0.1, "MHz")
        self.sp_frabi.setToolTip(
            "Angenommene Rabi-Frequenz Omega/2pi des Zwei-Photonen-Uebergangs,\n"
            "gemessen auf dem ZEITGEMITTELTEN Profil im Auswertebereich. Sie\n"
            "setzt nur den Massstab der Flaeche, nicht ihre Form.")
        self.sp_tp = self._dspin(1.0, 0.001, 10000.0, 4, 0.1, "us")
        self.sp_tp.setToolTip("Pulslaenge (Rechteckpuls).")
        self.btn_pi = QPushButton("auf pi-Puls setzen")
        self.btn_pi.setToolTip("T_p = 1/(2*f_rabi) - die Laenge, die auf der\n"
                               "Referenzintensitaet gerade theta = pi ergibt.")
        self.btn_pi.clicked.connect(self._set_pi)

        self.sp_delay = self._dspin(1.0, 0.001, 10000.0, 4, 0.1, "us")
        self.sp_delay.setToolTip("Halbe Breite des Delay-Scans: 0 ist der\n"
                                 "perfekte Trigger, +-dieser Wert die Grenze.")
        self.sp_pts = QSpinBox(); self.sp_pts.setRange(21, 4001)
        self.sp_pts.setValue(401); self.sp_pts.setSingleStep(50)

        self.cmb_region = QComboBox()
        self.cmb_region.addItems(["atomgewichtet (thermisch)", "Plateau",
                                  "Kreis (Radius aus dem Haupt-GUI)",
                                  "Spotzentren"])
        self.cmb_region.setToolTip(
            "Worueber gemittelt wird.\n\n"
            "'atomgewichtet' ist die einzige Option, die eine Atom-Groesse\n"
            "liefert: ein Gauss-Gewicht der Aufenthaltswahrscheinlichkeit um\n"
            "den Atomort, sigma aus T und nu_r wie in\n"
            "Weighted_Multitone_Lens_GUI. Die anderen mitteln ueber Flaechen,\n"
            "auf denen gar kein einzelnes Atom sitzt - 'Plateau' ist eine\n"
            "Kamera-Groesse, nicht das, was ein Atom sieht.")
        self.sp_T = self._dspin(17.0, 0.01, 10000.0, 2, 1.0, "uK")
        self.sp_T.setToolTip("Atomtemperatur; zusammen mit nu_r ergibt sich\n"
                             "sigma^2 = hbar/(2 m w) coth(hbar w/(2 kB T)).")
        self.sp_nu = self._dspin(60.4, 0.1, 100000.0, 2, 1.0, "kHz")
        self.sp_nu.setToolTip("Radiale Fallenfrequenz nu_r.")
        self.cb_auto = QCheckBox("Trigger automatisch auf max. Flaeche")
        self.cb_auto.setChecked(True)
        self.cb_auto.setToolTip(
            "Sucht das Maximum der mittleren Pulsflaeche ueber eine ganze\n"
            "Grundperiode. Dort ist die Ableitung nach dem Delay null, der\n"
            "Jitter geht also nur quadratisch ein - der eigentliche Grund,\n"
            "auf ein Maximum und nicht auf eine Flanke zu triggern.")
        self.cb_auto.stateChanged.connect(
            lambda _: self.sp_t0.setEnabled(not self.cb_auto.isChecked()))
        self.sp_t0 = self._dspin(0.0, 0.0, 1e6, 4, 1.0, "us")
        self.sp_t0.setEnabled(False)
        self.sp_t0.setToolTip("Trigger von Hand, gemessen ab dem Anfang der\n"
                              "Grundperiode.")

        rows = [("Rabi-Frequenz", self.sp_frabi), ("Pulslaenge", self.sp_tp),
                ("", self.btn_pi),
                ("Delay-Bereich +-", self.sp_delay), ("Punkte", self.sp_pts),
                ("Bereich", self.cmb_region),
                ("", self.cb_auto), ("Trigger t_0", self.sp_t0),
                ("Atom T", self.sp_T), ("Fallenfrequenz nu_r", self.sp_nu)]
        for i, (name, w) in enumerate(rows):
            lay.addWidget(QLabel(name), i % 3, 2 * (i // 3))
            lay.addWidget(w, i % 3, 2 * (i // 3) + 1)
        self.lbl_note = QLabel("-")
        self.lbl_note.setStyleSheet("color: #555; font-size: 10px;")
        self.lbl_note.setWordWrap(True)
        lay.addWidget(self.lbl_note, 3, 0, 1, 6)
        return g

    def _set_pi(self):
        f = self.sp_frabi.value() * 1e6
        if f > 0:
            self.sp_tp.setValue(1e6 / (2.0 * f))

    # ------------------------------------------------------------
    def _atom_sigma(self):
        return self.fns["sigma_thermal"](self.sp_nu.value() * 1e3,
                                         self.sp_T.value() * 1e-6)

    def _mask(self, c):
        idx = self.cmb_region.currentIndex()
        if idx == 2:
            return c["region_mask"], "Kreis"
        if idx == 3:
            m = np.zeros(c["I_avg"].shape, bool)
            x, y = c["x"], c["y"]
            for cxi, cyi in zip(c["centers_x"], c["centers_y"]):
                m[int(np.argmin(np.abs(y - cyi))), int(np.argmin(np.abs(x - cxi)))] = True
            return m, "Spotzentren"
        return c["plateau"], "Plateau"

    def _fields(self, c):
        """Feld-Stack, Maske, Gewicht und Gitter fuer die gewaehlte Auswertung.

        Im atomgewichteten Fall wird ein eigenes, feines lokales Gitter um den
        Atomort aufgespannt - das globale ist mit 0.6 um pro Zelle viel zu
        grob fuer ein Atom von 0.1 um."""
        if self.cmb_region.currentIndex() != 0:
            mask, name = self._mask(c)
            return c["F_stack"], mask, None, c["x"], c["y"], name, None
        s = self.parent_win.state
        sig = self._atom_sigma()
        xs, ys, Xs, Ys, F, W = self.fns["atom_local_stack"](
            c["centers_x"], c["centers_y"], c["amp_spots"], s["win"],
            s["use_airy"], s["airy_factor"], sig)
        return F, None, W, xs, ys, "Atom (sigma = %.1f nm)" % (sig * 1e9), sig

    def recompute(self):
        c = getattr(self.parent_win, "cache", {}) or {}
        if not c or "F_stack" not in c:
            QMessageBox.information(self, "Nichts zu rechnen",
                                    "Im Haupt-GUI erst 'Recompute' druecken.")
            return
        f0, T0 = c["f0"], c["T0"]
        if not (f0 > 0 and np.isfinite(T0)):
            QMessageBox.information(
                self, "Keine Periode",
                "Ohne gemeinsame Grundperiode gibt es keinen definierten "
                "Trigger-Zeitpunkt im Zyklus.")
            return
        F, mask, W, gx, gy, region_name, sig_atom = self._fields(c)
        if mask is not None and not mask.any():
            QMessageBox.information(self, "Leerer Bereich",
                                    "Der gewaehlte Bereich enthaelt keine Punkte.")
            return

        k, ph = c["k_orders"], c["phases"]
        law = self.parent_win.state.get("rabi_law", "I")
        f_rabi = self.sp_frabi.value() * 1e6
        t_p = self.sp_tp.value() * 1e-6
        half = self.sp_delay.value() * 1e-6
        n_pts = self.sp_pts.value()

        # --- mittlere Flaeche als Funktion des Triggers, ueber eine Periode ---
        n_scan = int(max(2000, min(40000, 40 * T0 / max(t_p, 1e-12))))
        t_scan = np.arange(n_scan) / n_scan * T0
        if law == "I":
            coef, orders = self.fns["beat_coeffs_mean"](F, k, ph, mask, W)
            area = lambda tt: self.fns["pulse_area_curve"](coef, orders, f0, t_p, tt)
            g_ref = float(np.real(coef[0]))          # <I> ueber Zeit und Bereich
        else:
            w, dt = self.fns["sqrt_mean_series"](F, k, ph, f0, mask, W, oversample=10)
            area = lambda tt: self.fns["sqrt_area_curve"](w, dt, t_p, tt)
            g_ref = float(np.mean(w))                # <sqrt(I)>
        a_scan = area(t_scan)

        # --- Trigger waehlen; das Maximum parabolisch nachziehen ---
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

        # --- Delay-Scan ---
        delays = np.linspace(-half, half, n_pts)
        a_delay = area(t_opt + delays)

        # Flaeche -> theta. Die Flaeche hat die Einheit "g mal Zeit"; geteilt
        # durch die Referenz <g> wird daraus eine Zeit, mal 2 pi f_rabi eine
        # Phase. Ein Puls auf der Referenzintensitaet hat damit genau
        # theta = 2 pi f_rabi T_p.
        scale = 2 * np.pi * f_rabi / max(g_ref, 1e-300)
        th_scan = a_scan * scale
        th_delay = a_delay * scale
        th0 = float(area(np.array([t_opt]))[0] * scale)

        # --- Uniformitaet an einigen Delays: nur die Maskenspalten ---
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
            th_pix = np.stack([self.fns["sqrt_area_map"](Fm, k, ph, f0, t_p, tt)
                               for tt in (t_opt + d_u)])
        mu = th_pix @ wpix
        var = ((th_pix - mu[:, None]) ** 2) @ wpix
        u_delay = np.where(mu > 0, np.sqrt(var) / np.maximum(mu, 1e-300), np.nan)

        # Anregung. sin^2 ist nichtlinear, deshalb ist der Mittelwert der
        # Anregung NICHT die Anregung des Mittelwerts - bei 74 % Streuung
        # liegen die beiden weit auseinander. Gemittelt wird ueber die Orte,
        # so wie es das Experiment tut.
        # Der differentielle Lichtshift eta = delta_LS/Omega ist ortsunabhaengig
        # (Omega und delta_LS teilen denselben Faktor I) und schliesst das
        # Zweiniveauproblem:  P = 1/(1+eta^2) sin^2( sqrt(1+eta^2) theta/2 ).
        eta = float(self.parent_win.state.get("eta_ls", 0.0) or 0.0)
        gsh = np.sqrt(1.0 + eta ** 2)
        p_of = lambda th: np.sin(gsh * th * scale / 2.0) ** 2 / (1.0 + eta ** 2)
        exc_delay = p_of(th_pix) @ wpix

        # --- Karte bei Delay 0 ---
        if law == "I":
            th_map = self.fns["pulse_area_map"](F, k, ph, f0, t_p, t_opt) * scale
        else:
            th_map = self.fns["sqrt_area_map"](F, k, ph, f0, t_p, t_opt) * scale
        if mask is None:
            th_map = th_map.reshape(len(gy), len(gx))

        # Vergleichsmass: die raeumliche Streuung des ZEITMITTELS im selben
        # Bereich. Ein Puls, der lang gegen T_0 ist, laeuft darauf zu; die
        # Differenz zum Pulswert ist genau das, was die Schwebung anrichtet.
        if mask is None:
            me_map, _ = self.fns["time_stats_exact"](F, k, ph)
            me = me_map.ravel()
            mu_a = float(me @ wpix)
            u_avg = float(np.sqrt(((me - mu_a) ** 2) @ wpix) / mu_a) if mu_a > 0 \
                else float("nan")
        else:
            me = c["mean_exact"][mask]
            u_avg = float(np.std(me) / np.mean(me)) if np.mean(me) > 0 else float("nan")

        self._last = dict(exc_delay=exc_delay, eta=eta,
                          t_scan=t_scan, th_scan=th_scan, delays=delays,
                          th_delay=th_delay, th0=th0, t_opt=t_opt, T0=T0, f0=f0,
                          t_p=t_p, f_rabi=f_rabi, d_u=d_u, u_delay=u_delay,
                          th_map=th_map, mask=mask, region=region_name, law=law,
                          half=half, u_avg=u_avg, th_ref=2 * f_rabi * t_p,
                          gx=gx, gy=gy, sig_atom=sig_atom)
        self._draw(c)

    # ------------------------------------------------------------
    def _draw(self, c):
        L = self._last
        self.fig.clear()
        # Die Karte ist das Einzige, was Ortsaufloesung zeigt - sie bekommt
        # deshalb die ganze rechte Spalte. Links die drei Kurven, die alle
        # ORTSMITTEL sind, untereinander.
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.0, 1.35],
                                   hspace=1.30, wspace=0.2)
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax3 = self.fig.add_subplot(gs[1, 0])
        ax4 = self.fig.add_subplot(gs[2, 0])
        ax2 = self.fig.add_subplot(gs[:, 1])

        # (1) Flaeche ueber eine ganze Grundperiode
        ax1.plot(L["t_scan"] * 1e6, L["th_scan"] / np.pi, lw=0.9, color="#1f77b4")
        ax1.axvline(L["t_opt"] * 1e6, color="#d62728", lw=1.2)
        ax1.axhline(L["th0"] / np.pi, color="#d62728", lw=0.6, ls=":")
        ax1.set_xlabel("Trigger $t_0$ [$\\mu$s]")
        ax1.set_ylabel(r"$\theta$ / $\pi$")
        ax1.set_title("Pulsflaeche ueber eine Grundperiode (T$_0$ = %.1f $\\mu$s)"
                      % (L["T0"] * 1e6), fontsize=10)
        ax1.grid(alpha=0.25)

        # (2) Karte bei Delay 0
        x, y = L["gx"], L["gy"]
        atom = L["sig_atom"] is not None
        sc = 1e9 if atom else 1e6
        ext = [x[0] * sc, x[-1] * sc, y[0] * sc, y[-1] * sc]
        im = ax2.imshow(L["th_map"] / np.pi, extent=ext, origin="lower",
                        cmap="viridis")
        if atom:
            cx0, cy0 = 0.5 * (x[0] + x[-1]), 0.5 * (y[0] + y[-1])
            th = np.linspace(0, 2 * np.pi, 200)
            for n in (1, 2):
                ax2.plot((cx0 + n * L["sig_atom"] * np.cos(th)) * sc,
                         (cy0 + n * L["sig_atom"] * np.sin(th)) * sc,
                         "w-", lw=0.8, alpha=0.8)
        elif L["mask"] is not None:
            ax2.contour(x * sc, y * sc, L["mask"].astype(float), levels=[0.5],
                        colors="w", linewidths=0.8)
        self.fig.colorbar(im, ax=ax2, fraction=0.046).set_label(r"$\theta$ / $\pi$",
                                                                fontsize=9)
        ax2.set_title("$\\theta(r)$ PUNKTWEISE, beim gewaehlten Trigger (Delay 0)"
                      + ("\n1$\\sigma$ und 2$\\sigma$ der Atomverteilung" if atom
                         else ""), fontsize=10)
        unit = "nm" if atom else "$\\mu$m"
        ax2.set_xlabel("x [%s]" % unit); ax2.set_ylabel("y [%s]" % unit)

        # (3) der Delay-Scan
        d = L["delays"] * 1e6
        ax3.plot(d, L["th_delay"] / np.pi, lw=1.6, color="#1f77b4",
                 label=r"$\theta$ / $\pi$")
        ax3.axvline(0.0, color="#888", lw=0.8)
        ax3.plot([0.0], [L["th0"] / np.pi], "o", color="#d62728", ms=5)
        ax3.set_xlabel(
            "Trigger-Fehler [$\\mu$s]   (0 = perfekt)\n"
            "INTENSITAETSGEWICHTETES ORTSMITTEL ueber %s;\n"
            "die Karte rechts zeigt $\\theta(r)$ punktweise" % L["region"],
            fontsize=8.5)
        ax3.set_ylabel(r"$\langle\theta\rangle$ / $\pi$   (Ortsmittel)")
        ax3.grid(alpha=0.25)
        ax3.set_xlim(d[0], d[-1])

        axr = ax3.twinx()
        rel = (L["th_delay"] / L["th0"] - 1.0) * 100.0 if L["th0"] else np.zeros_like(d)
        axr.plot(d, rel, lw=0.0)
        axr.set_ylim((np.array(ax3.get_ylim()) / (L["th0"] / np.pi) - 1.0) * 100.0)
        axr.set_ylabel("Abweichung von der Sollflaeche [%]")

        ax3.set_title("Pulsflaeche ueber dem Trigger-Fehler", fontsize=10)

        # (4) raeumliche Streuung der Flaeche ueber demselben Delay
        ax4.plot(L["d_u"] * 1e6, 100 * L["u_delay"], "s-", ms=3, lw=1.2,
                 color="#2ca02c")
        if np.isfinite(L["u_avg"]):
            ax4.axhline(100 * L["u_avg"], color="#888", ls="--", lw=0.9)
            ax4.text(0.02, 0.08, "Zeitmittel: %.2f %%" % (100 * L["u_avg"]),
                     transform=ax4.transAxes, fontsize=8, color="#555")
        ax4.set_xlabel("Trigger-Fehler [$\\mu$s]")
        ax4.set_ylabel(r"$\sigma_\theta/\langle\theta\rangle$ [%]")
        ax4.set_title("raeumliche Streuung der Flaeche", fontsize=10)
        ax4.grid(alpha=0.25)

        s = self.parent_win.state
        self.fig.suptitle(
            "%d$\\times$%d Toene, width %.4f / %.4f MHz  |  $\\Omega/2\\pi$ = %.3f MHz, "
            "T$_p$ = %.3f $\\mu$s, %s  |  Bereich: %s"
            % (s["N_x"], s["N_y"], s["width_x"] * 1e-6, s["width_y"] * 1e-6,
               L["f_rabi"] * 1e-6, L["t_p"] * 1e6,
               r"$\Omega\sim I$" if L["law"] == "I" else r"$\Omega\sim\sqrt{I}$",
               L["region"]), fontsize=10)
        self.canvas.draw_idle()

        # --- Kennzahlen ---
        rel_abs = np.abs(L["th_delay"] / L["th0"] - 1.0) if L["th0"] else np.zeros_like(d)
        edge = 100 * float(max(rel_abs[0], rel_abs[-1]))
        over = np.flatnonzero(rel_abs > 0.01)
        if over.size:
            tol = float(np.min(np.abs(L["delays"][over]))) * 1e9
            tol_txt = "%.0f ns bis 1 %% Flaechenfehler" % tol
        else:
            tol_txt = "ueber den ganzen Bereich unter 1 %% Flaechenfehler"
        exc_mean = float(L["exc_delay"][len(L["exc_delay"]) // 2])
        exc_naiv = np.sin(L["th0"] / 2.0) ** 2
        supp = abs(float(np.sinc(L["f0"] * L["t_p"])))
        k_half = 0.6034 / max(L["f0"] * L["t_p"], 1e-30)     # sinc = 0.5
        self.lbl_info.setText(
            "theta(0) = {:.4f} pi  ->  Anregung <sin^2(theta/2)> = {:.3f} "
            "(aus <theta> naiv gerechnet waere es {:.3f}{})   |   "
            "raeumlich: sigma_theta/<theta> = {:.2f} % im {}"
            "  (Zeitmittel im selben Bereich: {:.2f} %)\n"
            "Trigger bei t_0 = {:.4f} us im Zyklus. Am Rand des Scans "
            "(+-{:.3f} us) weicht die Flaeche um {:.2f} % ab; {}.\n"
            "Der Puls laesst Beat-Ordnungen bis d = {:.0f} (= {:.0f} kHz) mit "
            "mehr als 50 % durch - er mittelt die Schwebung nicht weg, er "
            "tastet sie ab.".format(
                L["th0"] / np.pi, exc_mean, exc_naiv,
                ", eta = %+.3f" % L["eta"] if L["eta"] else "",
                100 * float(L["u_delay"][len(L["u_delay"]) // 2]), L["region"],
                100 * L["u_avg"],
                L["t_opt"] * 1e6, L["half"] * 1e6, edge, tol_txt,
                k_half, k_half * L["f0"] * 1e-3))
        self.lbl_note.setText(
            "Normierung: Omega = 2 pi f_rabi * g/<g> mit <g> dem Mittel ueber Zeit "
            "UND Bereich - f_rabi ist also die Rabi-Frequenz auf dem "
            "zeitgemittelten Profil. Ein Puls auf dieser Referenz haette exakt "
            "theta = 2 pi f_rabi T_p = {:.4f} pi.".format(
                2 * L["f_rabi"] * L["t_p"]))

    # ------------------------------------------------------------
    def _on_apply(self):
        if self._last is None:
            return
        s = self.parent_win.state
        s["f_rabi"] = self._last["f_rabi"]
        s["pulse_t0"] = self._last["t_opt"]
        for w, v in ((self.parent_win.sp_frabi, self._last["f_rabi"] * 1e-6),
                     (self.parent_win.sp_t0, self._last["t_opt"] * 1e6)):
            w.blockSignals(True); w.setValue(v); w.blockSignals(False)
        self.lbl_note.setText("Trigger t_0 = %.4f us und f_rabi = %.4f MHz ins "
                              "Haupt-GUI uebernommen."
                              % (self._last["t_opt"] * 1e6,
                                 self._last["f_rabi"] * 1e-6))

    def _on_save(self):
        if self._last is None:
            return
        s = self.parent_win.state
        L = self._last
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        name = ("Pulsflaeche_N{}x{}_frabi{:.3f}MHz_tp{:.3f}us_{}"
                .format(s["N_x"], s["N_y"], L["f_rabi"] * 1e-6, L["t_p"] * 1e6, stamp))
        out = self.parent_win.out_dir
        try:
            self.fig.savefig(out / (name + ".pdf"), format="pdf", bbox_inches="tight")
            head = "delay_us\ttheta_over_pi\trel_deviation_percent"
            data = np.column_stack([L["delays"] * 1e6, L["th_delay"] / np.pi,
                                    (L["th_delay"] / L["th0"] - 1) * 100])
            np.savetxt(out / (name + "_scan.txt"), data, header=head,
                       delimiter="\t", fmt="%.9g")
            lines = [stamp, "",
                     "N_x x N_y = %d x %d" % (s["N_x"], s["N_y"]),
                     "width_x = %.6f MHz, width_y = %.6f MHz"
                     % (s["width_x"] * 1e-6, s["width_y"] * 1e-6),
                     "waist = %.4f um, profile = %s"
                     % (s["win"] * 1e6, "Airy" if s["use_airy"] else "Gauss"),
                     "f_0 = %.6f kHz, T_0 = %.4f us" % (L["f0"] * 1e-3, L["T0"] * 1e6),
                     "law: Omega ~ %s" % ("I" if L["law"] == "I" else "sqrt(I)"),
                     "f_rabi = %.6f MHz (on the time averaged profile)"
                     % (L["f_rabi"] * 1e-6),
                     "T_p = %.6f us, trigger t_0 = %.6f us"
                     % (L["t_p"] * 1e6, L["t_opt"] * 1e6),
                     "region = %s" % L["region"],
                     "theta(0) = %.6f pi" % (L["th0"] / np.pi),
                     "eta (differential light shift) = %.6f" % L["eta"],
                     "phases x [deg] = " + ", ".join("%.2f" % v for v in
                                                     np.degrees(s["phase_x"])),
                     "phases y [deg] = " + ", ".join("%.2f" % v for v in
                                                     np.degrees(s["phase_y"])),
                     "", self.lbl_info.text()]
            (out / (name + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            self.lbl_note.setText("gespeichert: " + name + ".pdf (+ .txt, _scan.txt)")
        except Exception as exc:
            QMessageBox.critical(self, "Speichern fehlgeschlagen", str(exc))
