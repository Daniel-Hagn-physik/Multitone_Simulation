"""Ein-Linsen-Modus fuer das Beating-GUI: Optik, Parametersuche, Dialog.

WARUM EINE EIGENE DATEI
Der Laboraufbau fuer das Messbild ist nicht der Aufbau, fuer den die GUIs
gebaut wurden. Dort steht hinter dem AOD ein Teleskop f1 -> f2 und danach die
Fokussierlinse fLO; hier steht hinter dem AOD nur EINE Linse. Das aendert zwei
Groessen, und nur diese zwei:

    Ortsablage im Fokus      r(f)  = f_lens * tan(theta(f))
    Waist im Fokus           w_0   = lambda * f_lens / (pi * w_in)

Der Winkel selbst bleibt der des AOD,

    theta(f) = theta_max * (f - offset) / f_band ,

denn der AOD wird nicht angefasst. Aus beidem folgt die einzige Zahl, auf die
es bei der Auslegung ankommt - der Tonabstand, der genau einen Spotabstand von
einem Waist erzeugt:

    df(pitch/waist = 1) = w_0 / (dr/df) = v_ac / (pi * w_in)

mit v_ac = lambda * f_band / theta_max der Schallgeschwindigkeit im AOD. Die
Brennweite kuerzt sich heraus. Fuer w_in = 1.75 mm und v_ac = 665.6 m/s sind
das 121.1 kHz - unabhaengig davon, welche Linse man nimmt. Die Linse bestimmt
nur, wie gross das Bild insgesamt wird (w_0 = 6.51 um bei f = 45 mm), nicht,
wie viele Toene man braucht.

DIE BEATING-PERIODE
|E|^2 enthaelt nur Differenzen der Tonfrequenzen. Die Grundperiode ist
T_0 = 1/f_0 mit f_0 = ggT aller Differenzfrequenzen. Bei gleicher width auf
beiden Achsen ist

    df_x = width/(N_x-1),  df_y = width/(N_y-1),
    f_0  = ggT(df_x, df_y) = width / kgV(N_x-1, N_y-1) .

Eine Zielperiode T legt damit die width fest, sobald das Gitter steht:

    width = kgV(N_x-1, N_y-1) / T .

Das ist die ganze Suche. Sie hat einen unangenehmen Zug: bei gleicher width
zwingt eine lange Periode entweder zu dichten Spots oder zu vielen Toenen.
Fuer T = 100 us (f_0 = 10 kHz) und pitch/waist = 1 braucht es kgV/(N-1) = 12,
also teilerfremde N_x-1, N_y-1 um 12 herum - 13x14 Toene bei width 1.56 MHz.
Wer weniger Toene will, muss entweder dichter setzen (kleineres pitch/waist,
was dem Flattop nicht schadet, nur das Feld kleiner macht) oder ungleiche
widths zulassen (Option unten): df_x = 120 kHz, df_y = 130 kHz haben ebenfalls
ggT 10 kHz, brauchen aber nur 3x4 Toene.

KAMERA
Eine Belichtungszeit T_exp ist ein Boxcar in der Zeit. Auf den Fourierkoeffi-
zienten D_d der Intensitaet (Ordnung d, Frequenz d*f_0) wirkt sie als

    D_d -> D_d * sinc(d * f_0 * T_exp)   mit sinc(z) = sin(pi z)/(pi z).

Bei T_exp = 20 us und f_0 = 10 kHz bleibt die Grundschwingung mit 0.94 fast
voll stehen, waehrend die schnellen Anteile bei 120/130 kHz auf 0.08 gedaempft
werden. Die Kamera sieht also gerade das langsame Beating und mittelt das
schnelle weg - genau das ist der Sinn der Auslegung.
"""

import math
from math import gcd

import numpy as np
from scipy.special import j1

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QDoubleSpinBox,
    QSpinBox, QCheckBox, QPushButton, QGroupBox, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QHeaderView, QMessageBox
)
from PyQt5.QtCore import Qt


# ============================================================
# AOD - unveraendert gegenueber dem bisherigen Aufbau
# ============================================================
THETA_MAX = 43e-3       # rad, maximaler Ablenkwinkel
F_BAND = 36e6           # Hz, AOD-Bandbreite


def acoustic_velocity(lam, theta_max=THETA_MAX, f_band=F_BAND):
    """v_ac = lambda * f_band / theta_max, aus der Bragg-Beziehung."""
    return lam * f_band / theta_max


def pitch_per_hz(f_lens, theta_max=THETA_MAX, f_band=F_BAND):
    """dr/df im Fokus der einen Linse, in m/Hz (Kleinwinkelnaeherung)."""
    return f_lens * theta_max / f_band


def focus_waist(lam, f_lens, w_in):
    """Waist im Fokus bei kollimiertem Eingangsstrahl mit Waist w_in."""
    if w_in <= 0:
        return float("nan")
    return lam * f_lens / (math.pi * w_in)


def df_unit_ratio(lam, f_lens, w_in, theta_max=THETA_MAX, f_band=F_BAND):
    """Tonabstand, der pitch/waist = 1 ergibt. Von f_lens unabhaengig."""
    w0 = focus_waist(lam, f_lens, w_in)
    return w0 / pitch_per_hz(f_lens, theta_max, f_band)


# ============================================================
# Profil (identisch zum GUI, nur als Intensitaet)
# ============================================================
def _profile_intensity(r2, waist, use_airy, airy_factor):
    if not use_airy:
        return np.exp(-2.0 * r2 / waist ** 2)
    first_zero = airy_factor * waist
    k = 3.83170597 / first_zero
    u = k * np.sqrt(r2)
    out = np.ones_like(u)
    m = u > 1e-12
    out[m] = 2.0 * j1(u[m]) / u[m]
    return out ** 2


def _amps_from_ratio(r, N):
    a = np.ones(N, dtype=float)
    if N >= 2:
        a[0] = r
        a[-1] = r
    elif N == 1:
        a[0] = r
    return a


def plateau_ripple(N_x, N_y, pitch_x, pitch_y, waist, use_airy, airy_factor,
                   r_x=1.0, r_y=1.0, n=81):
    """Restwelligkeit des ZEITMITTELS ueber dem Spot-Rechteck.

    Das Zeitmittel ist die inkohaerente Summe der Einzelintensitaeten - genau
    die Groesse, die die anderen GUIs als Flattop optimieren. Ausgewertet wird
    das Rechteck zwischen den aeussersten Spotzentren; zurueckgegeben wird
    (I_max - I_min) / (I_max + I_min) darauf.
    """
    if N_x < 1 or N_y < 1:
        return float("nan")
    cx = (np.arange(N_x) - (N_x - 1) / 2.0) * pitch_x
    cy = (np.arange(N_y) - (N_y - 1) / 2.0) * pitch_y
    hx = max(abs(cx[0]), 1e-12) if N_x > 1 else 0.6 * waist
    hy = max(abs(cy[0]), 1e-12) if N_y > 1 else 0.6 * waist
    x = np.linspace(-hx, hx, n)
    y = np.linspace(-hy, hy, n)
    X, Y = np.meshgrid(x, y)
    ax = _amps_from_ratio(r_x, N_x)
    ay = _amps_from_ratio(r_y, N_y)
    I = np.zeros_like(X)
    for i in range(N_x):
        for j in range(N_y):
            a2 = (ax[i] * ay[j]) ** 2
            I += a2 * _profile_intensity((X - cx[i]) ** 2 + (Y - cy[j]) ** 2,
                                         waist, use_airy, airy_factor)
    lo, hi = float(I.min()), float(I.max())
    if hi + lo <= 0:
        return float("nan")
    return (hi - lo) / (hi + lo)


# ============================================================
# Frequenzentartung
# ============================================================
def degenerate_pair_count(kx, ky, N_x, N_y):
    """Anzahl Spotpaare mit exakt gleicher Gesamtfrequenz.

    Der Spot (n,m) traegt n*kx + m*ky in Einheiten von f_0. Die eine
    unvermeidbare Entartung der Ecken zaehlt mit; alles darueber kommt von
    gemeinsamen Teilern und ist ein Warnsignal (siehe Projektnotiz
    'Frequenzentartung im Multitone-Muster').
    """
    counts = {}
    for n in range(N_x):
        for m in range(N_y):
            key = n * kx + m * ky
            counts[key] = counts.get(key, 0) + 1
    return sum(c * (c - 1) // 2 for c in counts.values())


# ============================================================
# Kandidatensuche
# ============================================================
class Candidate:
    __slots__ = ("N_x", "N_y", "width_x", "width_y", "df_x", "df_y",
                 "pitch_x", "pitch_y", "rho_x", "rho_y", "span_x", "span_y",
                 "n_tones", "degen", "ripple", "score", "k_x", "k_y")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def search_candidates(target_period, lam, f_lens, w_in, waist,
                      rho_target=1.0, n_max=20, fix_nx=0, fix_ny=0,
                      equal_width=True, use_airy=True, airy_factor=1.483,
                      r_x=1.0, r_y=1.0, theta_max=THETA_MAX, f_band=F_BAND,
                      n_keep=25, tone_penalty=0.12):
    """Parametersaetze, die EXAKT die geforderte Beating-Periode liefern.

    Gleiche width (equal_width=True):
        Es gibt genau einen Satz pro Gitter (N_x, N_y):
            width = kgV(N_x-1, N_y-1) / T .
    Ungleiche width:
        df_x = k_x*f_0, df_y = k_y*f_0 mit ggT(k_x,k_y) = 1; gesucht werden
        die k, die pitch/waist am besten treffen.

    Sortiert nach Abweichung von rho_target (logarithmisch, damit zu dicht
    und zu weit gleich bewertet werden) plus einer milden Strafe auf die
    Tonzahl.
    """
    if target_period <= 0:
        return []
    f0 = 1.0 / target_period
    drdf = pitch_per_hz(f_lens, theta_max, f_band)
    df1 = waist / drdf                      # Tonabstand fuer pitch/waist = 1
    out = []

    nx_range = [fix_nx] if fix_nx >= 2 else range(2, n_max + 1)
    ny_range = [fix_ny] if fix_ny >= 2 else range(2, n_max + 1)

    for N_x in nx_range:
        a = N_x - 1
        for N_y in ny_range:
            b = N_y - 1
            if equal_width:
                L = a * b // gcd(a, b)
                kx, ky = L // a, L // b
                width_x = width_y = f0 * L
                if width_x > f_band:
                    continue
            else:
                # k_x, k_y frei und teilerfremd: das Ziel ist df ~ df1
                k_star = df1 / f0
                lo = max(1, int(k_star * 0.4))
                hi = max(lo + 1, int(k_star * 2.2) + 1)
                best = None
                for kx in range(lo, hi + 1):
                    for ky in range(lo, hi + 1):
                        if gcd(kx, ky) != 1:
                            continue
                        wx, wy = f0 * kx * a, f0 * ky * b
                        if wx > f_band or wy > f_band:
                            continue
                        s = math.hypot(math.log(kx * f0 / df1),
                                       math.log(ky * f0 / df1))
                        if best is None or s < best[0]:
                            best = (s, kx, ky, wx, wy)
                if best is None:
                    continue
                _, kx, ky, width_x, width_y = best

            df_x, df_y = f0 * kx, f0 * ky
            rho_x, rho_y = df_x / df1, df_y / df1
            # Tonzahl: zweiseitig um den Standard-Arbeitspunkt (3x4 = 12
            # Toene) bewertet. Einseitig gestraft wuerde die Suche bei
            # ungleichen widths, wo rho immer perfekt getroffen wird, auf
            # 2x2 zusammenfallen - ein Gitter, das kein Flattop mehr ist.
            score = (math.hypot(math.log(rho_x / rho_target),
                                math.log(rho_y / rho_target))
                     + tone_penalty * abs(math.log(N_x * N_y / 12.0))
                     + 0.02 * abs(math.log(N_x / N_y)))   # eher quadratisch
            out.append(Candidate(
                N_x=N_x, N_y=N_y, width_x=width_x, width_y=width_y,
                df_x=df_x, df_y=df_y, k_x=kx, k_y=ky,
                pitch_x=df_x * drdf, pitch_y=df_y * drdf,
                rho_x=rho_x, rho_y=rho_y,
                span_x=(N_x - 1) * df_x * drdf, span_y=(N_y - 1) * df_y * drdf,
                n_tones=N_x * N_y,
                degen=degenerate_pair_count(kx, ky, N_x, N_y),
                ripple=float("nan"), score=score))

    out.sort(key=lambda c: c.score)
    out = out[:n_keep]
    for c in out:
        c.ripple = plateau_ripple(c.N_x, c.N_y, c.pitch_x, c.pitch_y, waist,
                                  use_airy, airy_factor, r_x, r_y)
    return out


def camera_visibility(f0, t_exp):
    """Anteil der Grundschwingung, der eine Belichtung t_exp ueberlebt."""
    if t_exp <= 0:
        return 1.0
    return abs(float(np.sinc(f0 * t_exp)))


def suppression_table(f0, t_exp, orders):
    """sinc-Daempfung pro Beat-Ordnung, fuer den Infotext."""
    o = np.asarray(orders, dtype=float)
    return np.abs(np.sinc(o * f0 * t_exp))


# ============================================================
# Dialog
# ============================================================
class OneLensDesignDialog(QDialog):
    """Auslegung des Ein-Linsen-Aufbaus auf eine Ziel-Beating-Periode.

    Der Dialog rechnet nur - uebernommen wird erst auf Knopfdruck, und dann
    genau die markierte Zeile.
    """

    # (Ueberschrift, Nachkommastellen; None = ganzzahlig)
    COLS = [("Rang", None), ("N_x", None), ("N_y", None),
            ("width_x [MHz]", 4), ("width_y [MHz]", 4),
            ("df_x [kHz]", 1), ("df_y [kHz]", 1),
            ("pitch_x [um]", 3), ("pitch_y [um]", 3),
            ("pitch/waist x", 3), ("pitch/waist y", 3),
            ("Feld [um]", 1), ("Toene", None), ("entartet", None),
            ("Ripple [%]", 2)]

    def __init__(self, parent=None, state=None):
        super().__init__(parent)
        self.setWindowTitle("Ein-Linsen-Aufbau - Parameter fuer eine Ziel-Beating-Periode")
        self.resize(1180, 760)
        s = state or {}
        self.result_candidate = None
        self._cands = []

        self.lam = float(s.get("lambda_opt", 795e-9))
        self.use_airy = bool(s.get("use_airy", True))
        self.airy_factor = float(s.get("airy_factor", 1.483))
        self.r_x = float(s.get("r_x", 1.0))
        self.r_y = float(s.get("r_y", 1.0))

        root = QVBoxLayout(self)
        root.addWidget(self._group_inputs(s))
        root.addWidget(self._group_derived())

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels([c[0] for c in self.COLS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.doubleClicked.connect(lambda _: self._on_apply())
        root.addWidget(self.table, 1)

        self.lbl_detail = QLabel("-")
        self.lbl_detail.setWordWrap(True)
        self.lbl_detail.setStyleSheet("color: #333; font-size: 11px;")
        root.addWidget(self.lbl_detail)

        row = QHBoxLayout()
        self.btn_search = QPushButton("Vorschlaege berechnen")
        self.btn_search.clicked.connect(self._on_search)
        self.btn_apply = QPushButton("Markierten Satz uebernehmen")
        self.btn_apply.clicked.connect(self._on_apply)
        self.btn_apply.setEnabled(False)
        btn_close = QPushButton("Schliessen")
        btn_close.clicked.connect(self.reject)
        row.addWidget(self.btn_search)
        row.addStretch(1)
        row.addWidget(self.btn_apply)
        row.addWidget(btn_close)
        root.addLayout(row)

        self._update_derived()
        self._on_search()

    # ---------------- UI ----------------
    def _dspin(self, val, lo, hi, dec, step, suffix=""):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(dec)
        w.setSingleStep(step)
        w.setValue(val)
        if suffix:
            w.setSuffix(" " + suffix)
        w.valueChanged.connect(self._update_derived)
        return w

    def _group_inputs(self, s):
        g = QGroupBox("Aufbau und Ziel")
        lay = QGridLayout(g)

        # The waist in front of the lens is a property of the SINGLE LENS
        # build; state["win_in"] holds the telescope-conjugate value while
        # the other build is active, which would be the wrong number here.
        f_default = float(s.get("f_single", 45e-3)) * 1e3
        if s.get("one_lens"):
            w_default = float(s.get("win_in") or 1.75e-3) * 1e3
        else:
            w_default = float(s.get("win_in_single") or 1.75e-3) * 1e3
        self.sp_f = self._dspin(f_default, 1.0, 2000.0, 3, 1.0, "mm")
        self.sp_win_in = self._dspin(w_default, 0.01, 50.0, 4, 0.05, "mm")
        self.sp_T = self._dspin(float(s.get("target_period", 100e-6)) * 1e6,
                                0.1, 100000.0, 3, 10.0, "us")
        self.sp_texp = self._dspin(float(s.get("t_exp") or 20e-6) * 1e6,
                                   0.0, 100000.0, 3, 5.0, "us")
        self.sp_rho = self._dspin(1.0, 0.05, 5.0, 3, 0.05)
        self.sp_rho.setToolTip(
            "Ziel fuer Spotabstand / Waist. 1.0 entspricht dem bisherigen\n"
            "Arbeitspunkt (pitch 1.137 um bei waist 1.10 um). Kleinere Werte\n"
            "sind kein Fehler: dichter gesetzte Spots geben ein glatteres\n"
            "Flattop, nur ein kleineres Feld bei gleicher Tonzahl.")
        self.sp_nmax = QSpinBox()
        self.sp_nmax.setRange(2, 64)
        self.sp_nmax.setValue(int(s.get("n_max", 20)))
        self.sp_nx_fix = QSpinBox()
        self.sp_nx_fix.setRange(0, 64)
        self.sp_nx_fix.setSpecialValueText("frei")
        self.sp_ny_fix = QSpinBox()
        self.sp_ny_fix.setRange(0, 64)
        self.sp_ny_fix.setSpecialValueText("frei")
        self.cb_unequal = QCheckBox("ungleiche width_x / width_y zulassen")
        self.cb_unequal.setToolTip(
            "Aus. Dann wird width_x = width_y gesucht, wie in allen anderen\n"
            "GUIs. Eingeschaltet duerfen die beiden Achsen verschiedene\n"
            "Spannen haben - dieselbe Beating-Periode kommt dann mit sehr viel\n"
            "weniger Toenen aus (3x4 statt 13x14), und die Frequenzentartung\n"
            "faellt nebenbei weg. Der Spotabstand in y aendert sich dadurch.")

        rows = [("Brennweite f", self.sp_f),
                ("Waist vor der Linse", self.sp_win_in),
                ("Ziel-Beating-Periode", self.sp_T),
                ("Belichtungszeit Kamera", self.sp_texp),
                ("Ziel pitch/waist", self.sp_rho),
                ("N max", self.sp_nmax),
                ("N_x festhalten", self.sp_nx_fix),
                ("N_y festhalten", self.sp_ny_fix),
                ("", self.cb_unequal)]
        for i, (name, w) in enumerate(rows):
            lay.addWidget(QLabel(name), i % 5, 2 * (i // 5))
            lay.addWidget(w, i % 5, 2 * (i // 5) + 1)
        return g

    def _group_derived(self):
        g = QGroupBox("Was daraus folgt")
        lay = QVBoxLayout(g)
        self.lbl_derived = QLabel("-")
        self.lbl_derived.setWordWrap(True)
        self.lbl_derived.setStyleSheet("font-size: 11px;")
        lay.addWidget(self.lbl_derived)
        return g

    # ---------------- Rechnen ----------------
    def _inputs(self):
        return dict(
            f_lens=self.sp_f.value() * 1e-3,
            w_in=self.sp_win_in.value() * 1e-3,
            T=self.sp_T.value() * 1e-6,
            t_exp=self.sp_texp.value() * 1e-6,
            rho=self.sp_rho.value(),
            n_max=self.sp_nmax.value(),
            fix_nx=self.sp_nx_fix.value(),
            fix_ny=self.sp_ny_fix.value(),
            equal=not self.cb_unequal.isChecked())

    def _update_derived(self, *_):
        p = self._inputs()
        w0 = focus_waist(self.lam, p["f_lens"], p["w_in"])
        drdf = pitch_per_hz(p["f_lens"])
        df1 = w0 / drdf
        f0 = 1.0 / p["T"] if p["T"] > 0 else float("nan")
        vis = camera_visibility(f0, p["t_exp"])
        zR = math.pi * w0 ** 2 / self.lam
        self.lbl_derived.setText(
            "v_ac = {:.1f} m/s   |   Waist im Fokus w_0 = {:.3f} um   "
            "(Rayleigh {:.1f} um)   |   dr/df = {:.4f} um/kHz\n"
            "Tonabstand fuer pitch/waist = 1: {:.2f} kHz   ->   "
            "pitch = {:.3f} um\n"
            "Ziel f_0 = {:.3f} kHz. Belichtung {:.1f} us laesst davon "
            "{:.0f} % stehen und daempft {:.0f} kHz auf {:.0f} %; "
            "{:.1f} Bilder pro Beating-Periode.".format(
                acoustic_velocity(self.lam), w0 * 1e6, zR * 1e6, drdf * 1e9,
                df1 * 1e-3, w0 * 1e6, f0 * 1e-3, p["t_exp"] * 1e6, 100 * vis,
                df1 * 1e-3, 100 * camera_visibility(df1, p["t_exp"]),
                p["T"] / p["t_exp"] if p["t_exp"] > 0 else float("inf")))

    def _on_search(self):
        p = self._inputs()
        w0 = focus_waist(self.lam, p["f_lens"], p["w_in"])
        self._cands = search_candidates(
            p["T"], self.lam, p["f_lens"], p["w_in"], w0,
            rho_target=p["rho"], n_max=p["n_max"],
            fix_nx=p["fix_nx"], fix_ny=p["fix_ny"], equal_width=p["equal"],
            use_airy=self.use_airy, airy_factor=self.airy_factor,
            r_x=self.r_x, r_y=self.r_y)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._cands))
        for i, c in enumerate(self._cands):
            vals = [i + 1, c.N_x, c.N_y, c.width_x * 1e-6, c.width_y * 1e-6,
                    c.df_x * 1e-3, c.df_y * 1e-3, c.pitch_x * 1e6,
                    c.pitch_y * 1e6, c.rho_x, c.rho_y,
                    max(c.span_x, c.span_y) * 1e6, c.n_tones, c.degen,
                    100 * c.ripple]
            for j, (v, (_, dec)) in enumerate(zip(vals, self.COLS)):
                it = QTableWidgetItem()
                # Als Zahl ablegen, nicht als Text - sonst sortiert die
                # Tabelle lexikographisch und 10 steht vor 9.
                it.setData(Qt.DisplayRole,
                           int(v) if dec is None else round(float(v), dec))
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == 0:
                    it.setData(Qt.UserRole, i)
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.AscendingOrder)
        if self._cands:
            self.table.selectRow(0)
        self._update_derived()

    def _selected(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        it = self.table.item(rows[0].row(), 0)
        if it is None:
            return None
        idx = it.data(Qt.UserRole)
        if idx is None or idx >= len(self._cands):
            return None
        return self._cands[idx]

    def _on_row_selected(self):
        c = self._selected()
        self.btn_apply.setEnabled(c is not None)
        if c is None:
            self.lbl_detail.setText("-")
            return
        p = self._inputs()
        f0 = 1.0 / p["T"]
        k_max = (c.N_x - 1) * c.k_x + (c.N_y - 1) * c.k_y
        supp = suppression_table(f0, p["t_exp"], np.arange(1, k_max + 1))
        n_survive = int(np.sum(supp > 0.5))
        self.lbl_detail.setText(
            "{}x{} Toene, width_x = {:.4f} MHz, width_y = {:.4f} MHz.   "
            "f_0 = {:.3f} kHz exakt, Periode {:.2f} us.\n"
            "Feld zwischen den aeusseren Spots {:.1f} x {:.1f} um bei Waist "
            "{:.2f} um; Ripple des Zeitmittels darauf {:.2f} %.   "
            "Hoechste Beat-Frequenz {:.2f} MHz (Ordnung {}).\n"
            "Von {} Ordnungen ueberstehen {} die Belichtung mit mehr als 50 %; "
            "{} frequenzentartete Spotpaare.".format(
                c.N_x, c.N_y, c.width_x * 1e-6, c.width_y * 1e-6,
                f0 * 1e-3, p["T"] * 1e6,
                c.span_x * 1e6, c.span_y * 1e6,
                focus_waist(self.lam, p["f_lens"], p["w_in"]) * 1e6,
                100 * c.ripple, k_max * f0 * 1e-6, k_max,
                k_max, n_survive, c.degen))

    def _on_apply(self):
        c = self._selected()
        if c is None:
            return
        p = self._inputs()
        if c.n_tones > 300:
            r = QMessageBox.question(
                self, "Viele Toene",
                "Dieser Satz hat {} Spots. Die kohaerente Zeitentwicklung "
                "skaliert quadratisch damit - Gitteraufloesung und Bilder pro "
                "Periode besser vorher heruntersetzen.\n\nTrotzdem "
                "uebernehmen?".format(c.n_tones),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if r != QMessageBox.Yes:
                return
        self.result_candidate = dict(
            N_x=c.N_x, N_y=c.N_y, width_x=c.width_x, width_y=c.width_y,
            f_single=p["f_lens"], win_in=p["w_in"], t_exp=p["t_exp"],
            target_period=p["T"], n_max=p["n_max"],
            link_width=abs(c.width_x - c.width_y) < 1e-6)
        self.accept()
