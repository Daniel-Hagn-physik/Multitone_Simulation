"""
Beating Multitone GUI (PyQt5)
=============================
Time-resolved, COHERENT simulation of the multitone flat-top pattern.

Why a separate GUI?
-------------------
Multitone_Lens_GUI.py and Weighted_Multitone_Lens_GUI.py sum the
INTENSITIES of the individual tones:

    I(x,y) = sum_s a_s * |u(r - c_s)|^2

That is the time average. Physically, however, the tones superpose as
FIELDS, and because every tone carries a different AOD frequency, the
relative phase between two tones runs at their difference frequency:

    E(x,y,t) = sum_s A_s * u(r - c_s) * exp(i*2pi*f_s*t + i*phi_s)
    I(x,y,t) = |E(x,y,t)|^2
             = sum_s A_s^2 |u_s|^2                          <- the time average
             + 2 * sum_{s<s'} A_s A_s' u_s u_s'
                     * cos(2pi (f_s - f_s') t + phi_s - phi_s')   <- the beating

The second term vanishes on average over one fundamental period - that is
exactly why the previous incoherent picture is correct as a time average,
but it says nothing about the instantaneous values. The beating becomes
visible wherever two spots overlap spatially (u_s * u_s' != 0), that is,
between the traps.

Frequencies
-----------
As in the AWG, the tones sit at

    f_x(n) = offset + width * n/(N_x-1),   n = 0 .. N_x-1
    f_y(m) = offset + width * m/(N_y-1),   m = 0 .. N_y-1

Both AODs shift the light frequency, so a spot (n,m) carries
f_s = f_x(n) + f_y(m). Only the DIFFERENCE of two spot frequencies
appears in |E|^2:

    f_s - f_s' = width * ( dn/(N_x-1) + dm/(N_y-1) )

All of these differences are integer multiples of

    f_beat_0 = width / lcm(N_x-1, N_y-1)

so width/6 for 3x4. At width = 0.35 MHz that is 58.3 kHz, fundamental
period T_0 = 17.1 us. The absolute light frequency and the offset cancel
completely - wavelength and offset change the geometry (waist conversion
and beam deflection, respectively), but not a single beat frequency.

Field amplitudes
----------------
The other scripts carry `amps` as INTENSITY weights (I += a * profile).
The field therefore carries sqrt(a). The amplitudes come from the familiar
outer/inner parametrisation (amps_from_ratio):

    amp_x = [r_x, 1, ..., 1, r_x],   amp_y = [r_y, 1, ..., 1, r_y]
    a_spot(n,m) = amp_x[n] * amp_y[m]

Field profile of a single spot (normalised to 1 at the centre):
    Gauss:  u(r) = exp(-r^2 / w^2)          -> |u|^2 = exp(-2r^2/w^2)
    Airy:   u(r) = 2*J1(k r)/(k r)          -> |u|^2 = (2*J1/u)^2
            k = 3.83170597 / (scale factor * waist)
For the Airy profile u is NEGATIVE in the rings - this sign matters for
the coherent sum and is deliberately kept here.

Operation
---------
All parameters are typed in (no sliders except the time axis).
The default is the working point: 3x4 tones, AIRY profile, waist 1.1 um,
width 0.45 MHz, r_x = 1.0, r_y = 1.2, 795 nm, 100 MHz offset,
f1 = 60 mm, f2 = 750 mm.

Start:
    python Beating_Multitone_GUI.py

Physics used, and what is NOT in it
-----------------------------------
The excitation is computed from the accumulated pulse area alone:

    theta(r) = int Omega(r,t) dt        P(r) = sin^2(theta(r)/2)

For a resonant two-level system with an arbitrary time-dependent Omega this is
EXACT, not an approximation - checked against a step-by-step integration of the
Schroedinger equation (agreement to six digits). Everything the beating does to
a pulse enters through theta.

The coupling law is selectable, and the choice is not cosmetic:
  * Omega ~ I: both Raman legs come from this multitone beam (for instance an
    EOM adds the 3.035 GHz sideband after the AOD). Then every tone pairs with
    its own sideband, all pairs are two-photon resonant, and the coherent sum
    over all tone pairs gives exactly Omega ~ |E|^2 = I. The model is exact here.
  * Omega ~ sqrt(I): only one leg passes the AOD, the other is a separate clean
    beam. Then Omega ~ E is COMPLEX, and each tone sits at its own two-photon
    detuning (multiples of f_0, up to a few hundred kHz - comparable to the Rabi
    frequency itself). This code keeps only |Omega| ~ sqrt(I) and drops the
    phase; measured against the exact complex solution that costs up to about
    2.5 % in excitation at 0.2 MHz. Treat this branch as indicative.

The differential light shift is included through `eta` = delta/Omega. For a
Raman transition both scale with the same intensity, so their ratio is constant
in space and time and the two-level problem still closes:

    P = 1/(1+eta^2) * sin^2( sqrt(1+eta^2) * theta/2 )

verified numerically. The light shift therefore caps the contrast but adds NO
extra spatial non-uniformity - the uniformity numbers of this GUI are unaffected
by it. Compensating the mean shift by detuning the Raman recovers most of the
loss (at eta = 0.5: 0.77 -> 0.93 in the excitation).

NOT modelled, and to be kept in mind:
  * static two-photon detuning, magnetic-field shifts, and any residual
    UNcompensated light-shift gradient
  * spontaneous emission through the intermediate state
  * the Zeeman substructure of Rb-85 (F = 2 has 5, F = 3 has 7 sublevels, each
    with its own Clebsch-Gordan factor) - a clean pi pulse assumes one closed pair
  * atomic motion during the pulse; every position is treated as frozen, which
    is the right picture for shot-to-shot beam-pointing scatter, not for one
    atom oscillating in its trap
  * finite rise time of the pulse and the acoustic fill time of the AOD
  * polarisation and vector light shifts

The 3.035 GHz hyperfine splitting itself never appears in the formulae: it only
decides WHICH light components form the Raman pair, and the AOD tones are only
some 100 kHz apart. Whether the splitting is bridged inside or outside the
multitone path is exactly what the coupling law above encodes.
"""

import sys
import math
import datetime
from functools import reduce
from pathlib import Path as FilePath

import numpy as np
from scipy.special import j1

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QGroupBox, QScrollArea, QSplitter, QComboBox, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer

try:
    import airy_scale
    AIRY_SCALE_DEFAULT = airy_scale.AIRY_SCALE_DIALOG_DEFAULT
except Exception:      # airy_scale.py normally sits next to this file
    airy_scale = None
    AIRY_SCALE_DEFAULT = 1.4830


# ============================================================
# Fixed optics constants (identical to the other GUIs)
# offset and lambda are INPUT FIELDS here, no longer constants.
# ============================================================
fLO = 52.88e-3          # m   focusing lens
theta_max = 43e-3       # rad maximum deflection angle
f_band = 36e6           # Hz  AOD bandwidth
pitch = 5.288e-6        # m   physical atom spacing (for information only)

OUT_DIR_CANDIDATES = [
    FilePath(r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\PythonCode\Multitone_FlatTop"),
]


def short_name(path, keep=34):
    """Shorten a file name for display, with an ellipsis in the middle.

    A saved file name is long and contains no spaces, so it cannot be wrapped;
    the full path goes into the tooltip instead."""
    name = str(getattr(path, "name", path))
    if len(name) <= keep:
        return name
    head = keep // 2 - 2
    return name[:head] + "..." + name[-(keep - head - 3):]


def _resolve_out_dir():
    for cand in OUT_DIR_CANDIDATES:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except Exception:
            continue
    fallback = FilePath.cwd() / "FlatMultiTone_Output"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ============================================================
# Physics / numerics
# ============================================================
def multitone_frequencies(N, offset, width):
    """Discrete AWG frequencies: offset + width * n/(N-1).

    For N == 1 the single tone is placed at the CENTRE of the range
    (offset + width/2) so that it coincides with r_center - exactly as in
    Multitone_Lens_GUI.multitone_frequencies()."""
    if N <= 1:
        return np.array([offset + width / 2.0], dtype=float)
    return width * np.arange(N) / (N - 1) + offset


def angle_from_frequency(f, offset, theta_max_, f_band_):
    return theta_max_ * (f - offset) / f_band_


def radius_from_angle(theta, f1, f2, fLO_):
    return (f1 * fLO_ / f2) * np.tan(theta)


def conjugate_waist(w, f1, f2, lam):
    """Input <-> output waist through the telescope f1->f2 plus fLO.
        w_out = (f1/f2) * (lam * fLO) / (pi * w_in)
    The relation is symmetric, the same function computes both
    directions."""
    if w <= 0:
        return float("nan")
    return (f1 / f2) * (lam * fLO) / np.pi / w


def compute_centers_and_freqs(N_x, N_y, width_x, width_y, f1, f2, offset):
    """Spot centres AND the frequency of every spot.

    The spot ordering is identical to compute_centers() of the other
    GUIs (fx outer, fy inner), so that amp_spots = repeat(amp_x, N_y) *
    tile(amp_y, N_x) still fits unchanged.

    Here x and y get their OWN width. The other GUIs set both equal;
    that is precisely what produces frequency degeneracies (see
    degenerate_groups()), and separate widths are the only way to lift
    them - a constant frequency offset on one axis does NOT help, because
    it shifts all spot frequencies by the same amount and therefore leaves
    every difference unchanged."""
    fx_freq = multitone_frequencies(N_x, offset, width_x)
    fy_freq = multitone_frequencies(N_y, offset, width_y)
    f_center_x = offset + width_x / 2.0
    f_center_y = offset + width_y / 2.0
    r_center_x = radius_from_angle(
        angle_from_frequency(f_center_x, offset, theta_max, f_band), f1, f2, fLO)
    r_center_y = radius_from_angle(
        angle_from_frequency(f_center_y, offset, theta_max, f_band), f1, f2, fLO)

    centers_x, centers_y, f_spots = [], [], []
    for fx in fx_freq:
        rx = radius_from_angle(angle_from_frequency(fx, offset, theta_max, f_band), f1, f2, fLO)
        for fy in fy_freq:
            ry = radius_from_angle(angle_from_frequency(fy, offset, theta_max, f_band), f1, f2, fLO)
            centers_x.append(rx)
            centers_y.append(ry)
            # Both AODs shift the light frequency -> the spot carries the sum.
            f_spots.append(fx + fy)
    return (np.array(centers_x), np.array(centers_y), np.array(f_spots),
            r_center_x, r_center_y, fx_freq, fy_freq)


def amps_from_ratio(r, N):
    """Outer/inner parametrisation, literally the same as
    Amplitudes/.../multitone_flattop_optimizer.amps_from_ratio():
    the two outermost tones get r, all inner ones stay at 1."""
    amp = np.ones(N, dtype=float)
    if N >= 2:
        amp[0] = r
        amp[-1] = r
    elif N == 1:
        amp[0] = r
    return amp


def amp_spots_from_ratios(r_x, r_y, N_x, N_y):
    """Intensity weight per spot, ordering as in compute_centers_and_freqs()."""
    amp_x = amps_from_ratio(r_x, N_x)
    amp_y = amps_from_ratio(r_y, N_y)
    return np.repeat(amp_x, N_y) * np.tile(amp_y, N_x)


def spot_field(X, Y, cx, cy, width_param, use_airy, airy_factor):
    """FIELD (not intensity) of a single spot, normalised to 1 at the
    centre. For the Airy profile the sign of the rings is kept - it is
    essential for the coherent sum."""
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    if not use_airy:
        return np.exp(-r2 / width_param ** 2)
    first_zero_radius = airy_factor * width_param
    k = 3.83170597 / first_zero_radius
    u = k * np.sqrt(r2)
    out = np.ones_like(u)
    m = u > 1e-12
    out[m] = 2.0 * j1(u[m]) / u[m]
    return out


def build_field_stack(X, Y, centers_x, centers_y, amp_spots, width_param,
                      use_airy, airy_factor):
    """Stack A_s * u_s(x,y) for all spots. A_s = sqrt(intensity weight),
    because `amps` weights intensities everywhere else in the project."""
    S = len(centers_x)
    F = np.empty((S,) + X.shape, dtype=np.float64)
    A = np.sqrt(np.clip(amp_spots, 0.0, None))
    for s in range(S):
        F[s] = A[s] * spot_field(X, Y, centers_x[s], centers_y[s],
                                 width_param, use_airy, airy_factor)
    return F


def unique_beat_frequencies(f_spots, rtol=1e-10):
    """All positive difference frequencies that occur, ascending, without
    duplicates.

    The spot frequencies sit at ~200 MHz, the interesting differences at
    ~50 kHz. The minimum is therefore subtracted before forming the
    differences: mathematically that changes no difference, numerically it
    removes the cancellation of eight significant digits. Afterwards only
    values closer together than rtol * f_max are merged - without this step
    a rounding of 1e-4 Hz already fools the following gcd into seeing
    incommensurable frequencies."""
    f = np.asarray(f_spots, dtype=float)
    if f.size < 2:
        return np.array([])
    f = f - f.min()
    d = np.abs(f[:, None] - f[None, :])[np.triu_indices(f.size, k=1)]
    d = np.sort(d[d > 0])
    if d.size == 0:
        return np.array([])
    tol = d[-1] * rtol
    groups = [[d[0]]]
    for v in d[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return np.array([float(np.mean(g)) for g in groups])


def _gcd_float(a, b, scale, rel_tol=1e-8):
    """Greatest common divisor of two frequencies.

    `scale` is the reference quantity for the tolerance (the largest
    frequency that occurs) - it must NOT shrink along, otherwise the
    stopping criterion becomes ever sharper during the Euclidean algorithm
    and the iteration aborts on numerical noise."""
    a, b = abs(float(a)), abs(float(b))
    if a < b:
        a, b = b, a
    for _ in range(500):
        if b <= scale * rel_tol:
            return a
        a, b = b, a - b * np.floor(a / b)
    return 0.0


def fundamental_beat_frequency(f_spots, rel_tol=1e-6):
    """Fundamental frequency of the beating: greatest common divisor of ALL
    difference frequencies that occur.

    Determined purely numerically from the spot frequencies, so that
    separate widths for x and y are handled correctly as well. For equal
    width on both axes exactly the analytic result
    width / lcm(N_x-1, N_y-1) comes out (so width/6 for 3x4).

    A return value of 0.0 means: no common multiple - the difference
    frequencies are incommensurable, the signal is NOT periodic and cannot
    be shown in whole periods."""
    d = unique_beat_frequencies(f_spots)
    if d.size == 0:
        return 0.0
    scale = float(d[-1])
    g = float(d[0])
    for v in d[1:]:
        g = _gcd_float(g, v, scale)
        if g <= 0:
            return 0.0
    ratios = d / g
    if not np.all(np.abs(ratios - np.round(ratios)) < rel_tol * np.maximum(1.0, ratios)):
        return 0.0
    return g


def degenerate_groups(f_spots, f0, tol=1e-6):
    """Groups of spots that carry EXACTLY the same total frequency.

    Important, because the time average only equals the incoherent
    intensity sum of the other GUIs if ALL cross terms run at a non-zero
    frequency. Two spots with the same f_s have a cross term at 0 Hz - it
    never runs, never averages away and shows up as STATIC interference.

    For equal offset and equal width on both axes,
        f_s(n,m) = 2*offset + width * ( n/(N_x-1) + m/(N_y-1) ),
    so 2*offset + width*(3n+2m)/6 for 3x4. The value 3n+2m = 6 occurs
    twice: (n,m) = (0,3) and (2,0) - the two diagonally opposite corner
    spots are frequency degenerate.
    """
    if len(f_spots) < 2:
        return []
    f = np.asarray(f_spots, dtype=float)
    f = f - f.min()          # against cancellation, see unique_beat_frequencies()
    scale = f0 if f0 > 0 else max(1.0, float(np.max(np.abs(f))))
    key = np.round(f / (scale * tol)).astype(np.int64)
    groups = []
    for val in np.unique(key):
        idx = np.flatnonzero(key == val)
        if idx.size > 1:
            groups.append(idx)
    return groups


def min_frames_per_period(f_spots, f0):
    """Smallest number of frames per fundamental period that still resolves
    the FASTEST beat frequency that occurs.

    The time average over a window of whole periods is only exact if no
    beat harmonic falls on a multiple of the sampling rate. The highest
    difference frequency is f_max = M * f0; Nyquist requires more than
    2*M sampling points per fundamental period. With equal width on both
    axes M is small (12 for 3x4), with separate widths f0 can become very
    small and M correspondingly large - then too few frames pretend a
    completely wrong time average."""
    d = unique_beat_frequencies(f_spots)
    if d.size == 0 or f0 <= 0:
        return 1
    return int(2 * math.ceil(d[-1] / f0)) + 1


# ============================================================
# Tone phases
# ============================================================
# Physically, only the N_x + N_y PHASES OF THE RF TONES can be set. The
# phase of a spot (n,m) is their sum:
#
#     phi_spot(n,m) = phi_x(n) + phi_y(m)
#
# For 3x4 the twelve spot phases therefore come from seven degrees of
# freedom, they are NOT independently selectable.
#
# WHAT PHASES CAN AND CANNOT DO
# The cross term of ONE spot pair reads
#
#     2 * A_s A_s' * u_s(r) u_s'(r) * cos(2*pi*df*t + dphi)
#
# The phase only appears inside the cosine. Taken on its own, a single
# pair cannot be damped by any phase - the phase only shifts WHEN the
# maximum occurs.
#
# What matters, though, is that many pairs share THE SAME difference
# frequency (for 3x4 up to eleven pairs per frequency). Their
# contributions add as phasors:
#
#     D_d = sum_{k_s - k_s' = d} g_s g_s' e^{i(phi_s - phi_s')}
#
# and this sum CAN be partially cancelled by suitable phases. The
# temporal variance is exactly sum_{d != 0} |D_d|^2, so it does depend on
# the phases. Measured at 3x4, sigma_t/<I> in the plateau drops from 136 %
# (all phases 0) to 70 % at the optimum.
#
# It cannot be brought to zero: difference frequencies produced by only a
# single pair (for 3x4 e.g. d = 12) have no partner to cancel against.
# Even with completely free spot phases - which cannot be driven with two
# AODs at all - one only reaches 55 %.


def schroeder_phases(N):
    """Schroeder phases (M. R. Schroeder, IEEE Trans. Inf. Theory 16, 85 (1970)),
    the standard for multitone driving with equal amplitudes:
        phi_n = -pi * n(n-1)/N
    Approximately minimises the crest factor of the summed signal."""
    n = np.arange(N)
    return -np.pi * n * (n - 1) / max(N, 1)


def spot_phases_from_tones(phase_x, phase_y, N_x, N_y):
    """phi_spot(n,m) = phi_x(n) + phi_y(m), in the spot ordering of
    compute_centers_and_freqs() (fx outer, fy inner)."""
    return np.repeat(phase_x, N_y) + np.tile(phase_y, N_x)


def crest_factor(f_tones, phases, n_samples=20000, f_ref=None):
    """Crest factor of the summed RF signal of one axis: peak amplitude
    divided by the rms value. Decisive for how strongly the AOD is driven
    for short times."""
    f = np.asarray(f_tones, dtype=float)
    if f.size == 0:
        return float("nan")
    if f_ref is None or f_ref <= 0:
        span = np.ptp(f)
        f_ref = span if span > 0 else 1.0
    t = np.linspace(0.0, 1.0 / f_ref, n_samples, endpoint=False)
    sig = np.zeros_like(t)
    for fi, pi_ in zip(f, phases):
        sig += np.cos(2 * np.pi * fi * t + pi_)
    rms = np.sqrt(np.mean(sig ** 2))
    return float(np.max(np.abs(sig)) / rms) if rms > 0 else float("nan")


# ============================================================
# Exact time statistics without a time loop
# ============================================================
# With g_s(r) = A_s u_s(r) (real) and z_s = g_s e^{i phi_s},
#
#     I(r,t) = |sum_s z_s e^{i w_s t}|^2 = sum_d D_d(r) e^{i d w_0 t}
#     D_d(r) = sum_{k_s - k_s' = d} g_s g_s' e^{i(phi_s - phi_s')}
#
# where k_s is the order of the spot frequency in units of f_0. From this
# the mean and the variance follow in closed form:
#
#     <I>(r)    = D_0(r)                (contains the static terms of
#                                        frequency degenerate pairs!)
#     Var_t(I)  = sum_{d != 0} |D_d|^2 = 2 * sum_{d > 0} |D_d|^2
#
# This is exact - no sampling, no aliasing - and orders of magnitude
# faster than a time series. This is exactly what the phase optimisation
# needs.


def beat_orders(f_spots, f0):
    """Order k_s of every spot frequency in units of f_0, integer."""
    f = np.asarray(f_spots, dtype=float)
    if f0 <= 0:
        return np.zeros(f.size, dtype=int)
    return np.round((f - f.min()) / f0).astype(int)


def pair_lists(k):
    """For every order difference d >= 0 the list of spot pairs (s, s')
    with k_s - k_s' = d. d = 0 contains both the diagonal and the
    frequency degenerate pairs."""
    k = np.asarray(k)
    S = k.size
    out = {}
    for i in range(S):
        for j in range(S):
            d = int(k[i] - k[j])
            if d >= 0:
                out.setdefault(d, []).append((i, j))
    return {d: np.array(v) for d, v in out.items()}


def time_stats_exact(F, k, phases):
    """Time average and time variance of I(r,t), exact and without a time
    loop.

    Returns (mean_map, var_map) in the shape of F[0]."""
    S = F.shape[0]
    shape = F.shape[1:]
    G = F.reshape(S, -1)
    e = np.exp(1j * np.asarray(phases, dtype=float))
    pl = pair_lists(k)

    ps = pl[0]
    w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
    mean = np.zeros(G.shape[1])
    for (i, j), wij in zip(ps, w):
        mean += float(np.real(wij)) * G[i] * G[j]

    var = np.zeros(G.shape[1])
    for d, ps in pl.items():
        if d == 0:
            continue
        w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
        Dre = np.zeros(G.shape[1])
        Dim = np.zeros(G.shape[1])
        for (i, j), wij in zip(ps, w):
            gg = G[i] * G[j]
            Dre += wij.real * gg
            Dim += wij.imag * gg
        var += 2.0 * (Dre * Dre + Dim * Dim)
    return mean.reshape(shape), var.reshape(shape)


class VariationObjective:
    """Precomputed operator for the mean square temporal fluctuation in a
    region.

    The spatial sum can be pulled in front of the phases:

        sum_r Var(r)/<I>(r)^2 = sum_{d>0} 2 * w_d^H M_d w_d
        M_d[p,q] = sum_r (g_s g_s')_p (g_s g_s')_q / <I>(r)^2

    The M_d are small matrices (at most number of spots x number of spots)
    and are built once. After that an evaluation costs microseconds instead
    of milliseconds - only this makes a multi-start optimisation over the
    tone phases practical.

    The denominator <I>(r) is frozen at phases_ref. Through the frequency
    degenerate pairs it does depend weakly on the phases itself; varying it
    along would make the objective discontinuous without gaining
    anything."""

    def __init__(self, F, k, mask, phases_ref=None):
        S = F.shape[0]
        G = F.reshape(S, -1)[:, mask.ravel()]
        if phases_ref is None:
            phases_ref = np.zeros(S)
        mean, _ = time_stats_exact(F, k, phases_ref)
        mu2 = np.maximum(mean.ravel()[mask.ravel()], 1e-300) ** 2
        self.n = int(mask.sum())
        self.M, self.P = {}, {}
        for d, ps in pair_lists(k).items():
            if d == 0:
                continue
            prod = np.stack([G[i] * G[j] for i, j in ps])
            self.M[d] = (prod / mu2) @ prod.T
            self.P[d] = ps

    def components(self, phases):
        """Contribution of every order difference d separately, as relative
        RMS.

        sigma_d = sqrt(2 |D_d|^2 / <I>^2), averaged over the region. The sum
        of squares over all d gives rms(). This shows at WHICH frequency the
        unrest sits - decisive, because the trap really only responds to
        components near nu_r and 2*nu_r."""
        e = np.exp(1j * np.asarray(phases, dtype=float))
        out = {}
        for d, M in self.M.items():
            ps = self.P[d]
            w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
            v = 2.0 * float(np.real(np.conj(w) @ M @ w))
            out[d] = float(np.sqrt(max(v, 0.0) / max(self.n, 1)))
        return out

    def rms_weighted(self, phases, weights):
        """Like rms(), but every order d weighted with weights[d]. With a
        weighting that only counts the orders near nu_r and 2*nu_r, one can
        specifically suppress what the trap responds to - at the cost of the
        orders it does not care about."""
        e = np.exp(1j * np.asarray(phases, dtype=float))
        tot = 0.0
        for d, M in self.M.items():
            wgt = weights.get(d, 0.0)
            if wgt == 0.0:
                continue
            ps = self.P[d]
            w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
            tot += wgt * 2.0 * float(np.real(np.conj(w) @ M @ w))
        return float(np.sqrt(max(tot, 0.0) / max(self.n, 1)))

    def lower_bound(self, F, k, mask):
        """Lower bound for rms(), valid even for completely free PAIR phases
        (which are not physically adjustable - they follow from the spot
        phases). Per order at least max(0, 2*max|c_p| - sum|c_p|) remains: a
        frequency that is produced by only ONE spot pair has no partner to
        cancel against."""
        S = F.shape[0]
        G = F.reshape(S, -1)[:, mask.ravel()]
        mean, _ = time_stats_exact(F, k, np.zeros(S))
        mu2 = np.maximum(mean.ravel()[mask.ravel()], 1e-300) ** 2
        tot = 0.0
        for d, ps in self.P.items():
            P = np.abs(np.stack([G[i] * G[j] for i, j in ps]))
            lo = np.maximum(0.0, 2 * P.max(axis=0) - P.sum(axis=0))
            tot += 2.0 * float(np.sum(lo ** 2 / mu2))
        return float(np.sqrt(tot / max(self.n, 1)))

    def rms(self, phases):
        """Square root of the mean square relative fluctuation, that is the
        average of sigma_t(I)/<I> in the region (in the quadratic sense)."""
        e = np.exp(1j * np.asarray(phases, dtype=float))
        tot = 0.0
        for d, M in self.M.items():
            ps = self.P[d]
            w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
            tot += 2.0 * float(np.real(np.conj(w) @ M @ w))
        return float(np.sqrt(max(tot, 0.0) / max(self.n, 1)))


class UniformitySeries:
    """Time evolution of the uniformity U(t) = std(I)/mean(I) in a region.

    The same definition as in the other GUIs and optimisers of the project
    (evaluated on the time average there) - but here at every instant
    separately. Comparing the two is the actual point: the pipeline
    optimises U(<I>), the atom sees U(t).

    Large regions are thinned down to at most `max_points` sampling points;
    U is a ratio of area averages and therefore changes only in the third
    digit, but the optimisation becomes fast enough for hundreds of
    starting points."""

    def __init__(self, F, f_spots, mask, t, max_points=400, k=None):
        S = F.shape[0]
        idx = np.flatnonzero(mask.ravel())
        if idx.size > max_points:
            idx = idx[:: int(np.ceil(idx.size / max_points))]
        self.g = F.reshape(S, -1)[:, idx]
        self.t = np.asarray(t, dtype=float)
        self.W = np.exp(2j * np.pi * np.outer(self.t, np.asarray(f_spots, dtype=float)))
        # Time average on the same sampling points, for the level factor alpha(t)
        if k is not None:
            mean, _ = time_stats_exact(F, k, np.zeros(S))
            self.mu = mean.ravel()[idx]
            self.mu_norm = float(np.sum(self.mu * self.mu))
        else:
            self.mu = None

    def series(self, phases):
        Z = self.g * np.exp(1j * np.asarray(phases, dtype=float))[:, None]
        E = self.W @ Z
        I = E.real ** 2 + E.imag ** 2
        mu = I.mean(axis=1)
        return I.std(axis=1) / np.maximum(mu, 1e-300)

    def mean_u(self, phases):
        return float(np.mean(self.series(phases)))

    def level_and_u(self, phases):
        """(RMS of alpha(t)-1, mean U(t)) in a single pass.

        alpha(t) is the best common scale factor against the time average:
        it measures how strongly the trap depth breathes, U(t) how unequal
        the depths are among each other. These are the two quantities that
        matter for the trap - the plain deviation from <I> mixes them."""
        Z = self.g * np.exp(1j * np.asarray(phases, dtype=float))[:, None]
        E = self.W @ Z
        I = E.real ** 2 + E.imag ** 2
        mu_t = I.mean(axis=1)
        u = float(np.mean(I.std(axis=1) / np.maximum(mu_t, 1e-300)))
        if self.mu is None or self.mu_norm <= 0:
            return 0.0, u
        alpha = (I @ self.mu) / self.mu_norm
        return float(np.sqrt(np.mean((alpha - 1.0) ** 2))), u


def uniformity_of(I, mask):
    """U = std/mean in the region - the project convention."""
    v = I[mask]
    m = float(np.mean(v))
    return float(np.std(v) / m) if m > 0 else float("nan")


class PulseArea:
    """Accumulated Rabi area of a pulse,

        theta(r) = int_{t0}^{t0+Tp} Omega(r,t) dt

    For a pulsed drive the relevant quantity is NOT the instantaneous
    intensity but this area: the pulse integrates over the beating. Its
    uniformity across the evaluation region determines how uniform the
    rotation angle of the atoms becomes.

    For Omega ~ I (two-photon Raman, both branches from this profile) the
    integral can be given in closed form, because I is a Fourier series in
    f_0:

        theta = Tp * D_0 + 2 Re[ sum_{d>0} D_d * G_d ]
        G_d   = ( e^{i d w0 (t0+Tp)} - e^{i d w0 t0} ) / (i d w0)

    No time step, no discretisation errors - and fast enough to optimise
    phases AND pulse timing.

    For Omega ~ sqrt(I) (this profile is only ONE branch) there is no
    closed form; then the integration is done numerically."""

    def __init__(self, F, k, f0, mask, law="I", n_t=160, max_points=None):
        S = F.shape[0]
        idx = np.flatnonzero(mask.ravel())
        if max_points is not None and idx.size > max_points:
            # Thinning out for the Rabi curves: they need many evaluations, and
            # the excitation is an area average that barely moves with the
            # number of sampling points.
            idx = idx[:: int(np.ceil(idx.size / max_points))]
        self.G = F.reshape(S, -1)[:, idx]
        self.pl = pair_lists(k)
        self.prod = {d: np.stack([self.G[i] * self.G[j] for i, j in ps])
                     for d, ps in self.pl.items()}
        self.w0 = 2 * np.pi * f0
        self.law = law
        self.n_t = n_t
        self.idx = idx

    def theta(self, phases, t0, t_p, f_spots=None):
        if self.law != "I":
            # No closed form for sqrt(I): integrate numerically - but only on
            # the columns this object actually holds. Going through the full
            # grid here made the Rabi curves unusably slow.
            t = np.linspace(t0, t0 + t_p, self.n_t)
            Z = self.G * np.exp(1j * np.asarray(phases, dtype=float))[:, None]
            E = np.exp(2j * np.pi * np.outer(t, np.asarray(f_spots, dtype=float))) @ Z
            w = np.sqrt(np.maximum(E.real ** 2 + E.imag ** 2, 0.0))
            return w.mean(axis=0) * t_p
        e = np.exp(1j * np.asarray(phases, dtype=float))
        out = np.zeros(self.G.shape[1])
        for d, ps in self.pl.items():
            w = e[ps[:, 0]] * np.conj(e[ps[:, 1]])
            D = np.tensordot(w, self.prod[d], axes=(0, 0))
            if d == 0:
                out += t_p * D.real
            else:
                dw = d * self.w0
                Gd = (np.exp(1j * dw * (t0 + t_p)) - np.exp(1j * dw * t0)) / (1j * dw)
                out += 2.0 * np.real(D * Gd)
        return out

    def rabi_curves(self, phases, t0, t_p_values, t_pi, f_spots=None, n_random=14,
                    T0=None, eta=0.0):
        """Excitation versus pulse length - what an experiment actually measures.

        For every pulse length the accumulated area theta(r) is evaluated and
        the excitation averaged over the region:

            P = < sin^2( pi/2 * theta(r) / theta_pi ) >

        theta_pi is the mean area of a pi pulse at the working point, i.e. the
        pulse is calibrated on the mean as one would do in the lab.

        Returns three curves: the ideal case without beating, the triggered
        case at the given t0, and the untriggered case averaged over random
        pulse timings."""
        theta_pi = float(np.mean(self.theta(phases, t0, t_pi, f_spots)))
        if theta_pi <= 0:
            n = len(t_p_values)
            return np.zeros(n), np.zeros(n), np.zeros(n)
        # With a differential light shift proportional to the same intensity
        # (delta = eta*Omega) the two-level problem still has a closed form:
        #     P = 1/(1+eta^2) * sin^2( sqrt(1+eta^2) * theta/2 )
        # verified against a step-by-step integration of the Schroedinger
        # equation. eta is constant in space and time because Omega and delta
        # share the same intensity dependence - the shift caps the contrast but
        # adds no extra non-uniformity.
        g = np.sqrt(1.0 + eta ** 2)
        amp = 1.0 / (1.0 + eta ** 2)
        exc = lambda th: amp * np.sin(g * np.pi / 2 * th / theta_pi) ** 2
        ideal = exc(np.asarray(t_p_values) / t_pi * theta_pi)
        trig = np.array([float(np.mean(exc(self.theta(phases, t0, tp, f_spots))))
                         for tp in t_p_values])
        if T0 is None or not np.isfinite(T0) or T0 <= 0 or n_random < 2:
            return ideal, trig, trig
        t0s = np.linspace(0.0, T0, n_random, endpoint=False)
        none = np.array([float(np.mean([np.mean(exc(self.theta(phases, tt, tp, f_spots)))
                                        for tt in t0s])) for tp in t_p_values])
        return ideal, trig, none

    def uniformity(self, phases, t0, t_p, f_spots=None):
        th = self.theta(phases, t0, t_p, f_spots)
        m = float(np.mean(th))
        return float(np.std(th) / m) if m > 0 else float("nan")


def quadrature_penalty(phases, degen):
    """How far are frequency degenerate pairs from quadrature?

    A pair with the same total frequency has a cross term at 0 Hz:
    2 g_s g_s' cos(phi_s - phi_s'). It never runs and never averages away -
    BUT IT IS EXACTLY ZERO if the two spots are in quadrature (phase
    difference 90 or 270 degrees). Then, and only then, the time average is
    exactly the incoherent intensity sum computed by the other GUIs and by
    the scan pipeline.

    A return value of 0 means: all degenerate pairs are in quadrature."""
    if not degen:
        return 0.0
    pen = 0.0
    for grp in degen:
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                pen += abs(np.cos(float(phases[grp[i]]) - float(phases[grp[j]])))
    return pen


def resonance_check(beats, nu_r, tol_rel=0.06):
    """Does a beat line coincide with nu_r or 2*nu_r?

    The coherent beating produces a LINE spectrum at multiples of f_0. If
    no line lies near nu_r or 2*nu_r, the trap receives practically no
    power on its resonances despite 100 % modulation depth - that is the
    decisive point, not the size of the modulation.

    Returns: (distance_nu, distance_2nu, critical) with the distances of
    the nearest line in Hz."""
    if beats is None or len(beats) == 0 or nu_r <= 0:
        return float("nan"), float("nan"), False
    b = np.asarray(beats, dtype=float)
    d1 = float(np.min(np.abs(b - nu_r)))
    d2 = float(np.min(np.abs(b - 2 * nu_r)))
    crit = (d1 < tol_rel * nu_r) or (d2 < tol_rel * 2 * nu_r)
    return d1, d2, crit


def compute_grid(centers_x, centers_y, win_eff, resolution, pad_factor=2.5):
    """Square grid covering all spots plus a margin of pad_factor beam
    radii. Unlike in the metric GUIs, NO pitch copies of the neighbours are
    needed here - beating arises between the tones of the same pattern, not
    between neighbouring traps."""
    pad = pad_factor * win_eff
    x_lo, x_hi = centers_x.min() - pad, centers_x.max() + pad
    y_lo, y_hi = centers_y.min() - pad, centers_y.max() + pad
    # make it square so that the 2D plot is not distorted
    cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    half = 0.5 * max(x_hi - x_lo, y_hi - y_lo)
    x = np.linspace(cx - half, cx + half, resolution)
    y = np.linspace(cy - half, cy + half, resolution)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def intensity_cube(F, f_spots, phases, t):
    """I(x,y,t) = |sum_s F_s exp(i(2pi f_s t + phi_s))|^2 for all t.

    Returns float32 to keep the memory within bounds (at 200x200 and
    240 frames that is 38 MB)."""
    n_t = len(t)
    S, ny, nx = F.shape
    cube = np.empty((n_t, ny, nx), dtype=np.float32)
    for k in range(n_t):
        ph = 2.0 * np.pi * f_spots * t[k] + phases
        E_re = np.tensordot(np.cos(ph), F, axes=(0, 0))
        E_im = np.tensordot(np.sin(ph), F, axes=(0, 0))
        cube[k] = (E_re * E_re + E_im * E_im).astype(np.float32)
    return cube


# ============================================================
# Main window
# ============================================================
class BeatingMultitoneWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beating Multitone GUI - coherent time evolution")
        self.resize(1600, 950)

        self.state = {
            "N_x": 3,
            "N_y": 4,
            # Start values of the working point - the GUI always opens like this.
            "use_airy": True,
            "airy_factor": AIRY_SCALE_DEFAULT,
            "win": 1.10e-6,        # m, waist AFTER the lenses
            "win_in": None,        # m, waist BEFORE the lenses
            "win_mode": "output",
            "width_x": 0.45e6,     # Hz, frequency span of the x tones
            "width_y": 0.45e6,     # Hz, frequency span of the y tones
            "link_width": True,    # width_y follows width_x
            "r_x": 1.0,
            "r_y": 1.2,
            "lambda_opt": 795e-9,  # m
            "offset": 100e6,       # Hz
            "f1": 60e-3,
            "f2": 750e-3,
            "grid_n": 200,
            "n_periods": 3,
            "frames_per_period": 60,
            "phase_x": np.zeros(3),      # rad, phase per x tone
            "phase_y": np.zeros(4),      # rad, phase per y tone
            # Phases are always TONE phases: a spot (n,m) carries
            # phi_x(n) + phi_y(m). Free spot phases were a cross-check and
            # have been removed - they could not be driven anyway and did
            # not remove the modulation either.
            "f_rabi": 0.2e6,             # Hz, Rabi frequency Omega/2pi
            "pulse_t0": 0.0,             # s, start time of the pulse in the beat cycle
            "rabi_law": "I",             # 'I' (two-photon Raman) | 'sqrtI'
            "eta_ls": 0.0,               # differential light shift / Rabi frequency
            "auto_update": False,
        }
        self.state["win_in"] = conjugate_waist(
            self.state["win"], self.state["f1"], self.state["f2"], self.state["lambda_opt"])

        self.cache = {}
        self._panel_cbar = None
        self._opt_note = ""
        self._rabi_cache = None
        self._art = {}          # persistent drawing objects for the fast path
        self._last_panel = None
        self.frame_idx = 0
        self._building = True
        self.out_dir = _resolve_out_dir()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_tick)

        self._build_ui()
        self._building = False
        self.recompute()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _tame_info_labels(self):
        """Keep long, unbreakable texts from stretching the control panel.

        A saved file name has no spaces, so word wrap cannot break it and the
        label reports a very wide sizeHint - which the layout honours by
        widening the whole panel and squashing every button. Ignoring the
        horizontal sizeHint makes the label adapt to the panel instead of the
        other way round.
        """
        for lab in self.findChildren(QLabel):
            if lab.wordWrap():
                lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                lab.setMinimumWidth(1)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        self.fig = Figure(figsize=(10, 8.5))
        self.fig.set_constrained_layout(True)
        self.canvas = FigureCanvas(self.fig)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.55, 1], height_ratios=[1.5, 1, 1])
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_st = self.fig.add_subplot(gs[0, 1])
        self.ax_cut = self.fig.add_subplot(gs[1, 1])
        self.ax_time = self.fig.add_subplot(gs[2, 1])
        splitter.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setAlignment(Qt.AlignTop)
        pl.addWidget(self._group_tones())
        pl.addWidget(self._group_profile())
        pl.addWidget(self._group_beam())
        pl.addWidget(self._group_amps())
        pl.addWidget(self._group_pulse())
        pl.addWidget(self._group_phases())
        pl.addWidget(self._group_time())
        pl.addWidget(self._group_actions())
        pl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(360)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([1150, 420])

    # -- small helpers for the input fields ------------------
        self._tame_info_labels()

    def _dspin(self, value, lo, hi, dec, step, suffix=""):
        w = QDoubleSpinBox()
        w.setDecimals(dec)
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(value)
        w.setKeyboardTracking(False)
        if suffix:
            w.setSuffix(" " + suffix)
        w.valueChanged.connect(self._on_param_changed)
        return w

    def _ispin(self, value, lo, hi, suffix=""):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(value)
        w.setKeyboardTracking(False)
        if suffix:
            w.setSuffix(" " + suffix)
        w.valueChanged.connect(self._on_param_changed)
        return w

    def _group_tones(self):
        g = QGroupBox("Tones")
        lay = QGridLayout(g)
        self.sp_nx = self._ispin(self.state["N_x"], 1, 20)
        self.sp_ny = self._ispin(self.state["N_y"], 1, 20)
        lay.addWidget(QLabel("N_x"), 0, 0); lay.addWidget(self.sp_nx, 0, 1)
        lay.addWidget(QLabel("N_y"), 1, 0); lay.addWidget(self.sp_ny, 1, 1)
        self.lbl_freqs = QLabel("-")
        self.lbl_freqs.setWordWrap(True)
        self.lbl_freqs.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_freqs, 2, 0, 1, 2)
        return g

    def _group_profile(self):
        g = QGroupBox("Beam profile")
        lay = QGridLayout(g)
        self.cmb_profile = QComboBox()
        self.cmb_profile.addItems(["Gauss", "Airy"])
        self.cmb_profile.setCurrentIndex(1 if self.state["use_airy"] else 0)
        self.cmb_profile.currentIndexChanged.connect(self._on_param_changed)
        self.sp_airy = self._dspin(self.state["airy_factor"], 0.1, 5.0, 4, 0.01)
        self.sp_airy.setToolTip("first_zero_radius = factor * waist.\n"
                                "1.4830 = same 1/e^2 width as the Gauss,\n"
                                "1.19 = historical value.")
        lay.addWidget(QLabel("Profile"), 0, 0); lay.addWidget(self.cmb_profile, 0, 1)
        lay.addWidget(QLabel("Airy factor"), 1, 0); lay.addWidget(self.sp_airy, 1, 1)
        return g

    def _group_beam(self):
        g = QGroupBox("Beam / optics")
        lay = QGridLayout(g)
        self.cmb_winmode = QComboBox()
        self.cmb_winmode.addItems(["Waist after lens (um)", "Waist before lens (mm)"])
        self.cmb_winmode.currentIndexChanged.connect(self._on_winmode_changed)
        self.sp_win = self._dspin(self.state["win"] * 1e6, 0.05, 50.0, 4, 0.01, "um")
        self.sp_win_in = self._dspin(self.state["win_in"] * 1e3, 0.01, 50.0, 4, 0.01, "mm")
        self.sp_width = self._dspin(self.state["width_x"] * 1e-6, 0.0, 36.0, 5, 0.01, "MHz")
        self.sp_width_y = self._dspin(self.state["width_y"] * 1e-6, 0.0, 36.0, 5, 0.01, "MHz")
        self.cb_link_width = QCheckBox("width_y = width_x")
        self.cb_link_width.setChecked(self.state["link_width"])
        self.cb_link_width.stateChanged.connect(self._on_link_width_changed)
        self.cb_link_width.setToolTip(
            "The other GUIs set both widths equal. That is exactly what\n"
            "produces frequency degeneracies: at 3x4 the spots (n,m) = (0,3)\n"
            "and (2,0) carry exactly the same total frequency and interfere\n"
            "statically. Separate widths lift this - but change the spot\n"
            "spacing in y.")
        self.sp_lambda = self._dspin(self.state["lambda_opt"] * 1e9, 200.0, 2000.0, 2, 1.0, "nm")
        self.sp_offset = self._dspin(self.state["offset"] * 1e-6, 0.0, 500.0, 4, 1.0, "MHz")
        self.sp_f1 = self._dspin(self.state["f1"] * 1e3, 1.0, 2000.0, 2, 5.0, "mm")
        self.sp_f2 = self._dspin(self.state["f2"] * 1e3, 1.0, 2000.0, 2, 5.0, "mm")

        rows = [("Mode", self.cmb_winmode), ("waist", self.sp_win),
                ("waist_in", self.sp_win_in),
                ("width x", self.sp_width), ("width y", self.sp_width_y),
                ("", self.cb_link_width),
                ("Wavelength", self.sp_lambda), ("Offset f0", self.sp_offset),
                ("f1", self.sp_f1), ("f2", self.sp_f2)]
        for i, (name, w) in enumerate(rows):
            lay.addWidget(QLabel(name), i, 0)
            lay.addWidget(w, i, 1)
        hint = QLabel("Wavelength and offset change the geometry, but NO\n"
                      "beat frequency - |E|^2 contains only differences\n"
                      "of the tone frequencies, and a constant offset\n"
                      "cancels out of every difference.")
        hint.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(hint, len(rows), 0, 1, 2)
        self._sync_winmode_enabled()
        self._sync_width_enabled()
        return g

    def _group_amps(self):
        g = QGroupBox("Amplitudes (outer/inner)")
        lay = QGridLayout(g)
        self.sp_rx = self._dspin(self.state["r_x"], 0.0, 10.0, 4, 0.01)
        self.sp_ry = self._dspin(self.state["r_y"], 0.0, 10.0, 4, 0.01)
        lay.addWidget(QLabel("r_x"), 0, 0); lay.addWidget(self.sp_rx, 0, 1)
        lay.addWidget(QLabel("r_y"), 1, 0); lay.addWidget(self.sp_ry, 1, 1)
        self.lbl_amps = QLabel("-")
        self.lbl_amps.setWordWrap(True)
        self.lbl_amps.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_amps, 2, 0, 1, 2)
        return g

    def _group_pulse(self):
        g = QGroupBox("Pulsed operation")
        lay = QGridLayout(g)
        self.sp_frabi = QDoubleSpinBox()
        self.sp_frabi.setRange(0.001, 100.0); self.sp_frabi.setDecimals(4)
        self.sp_frabi.setSingleStep(0.05); self.sp_frabi.setValue(0.2)
        self.sp_frabi.setSuffix(" MHz"); self.sp_frabi.setKeyboardTracking(False)
        self.sp_frabi.setToolTip("Rabi frequency Omega/2pi. From it follows the "
                                 "pi pulse duration T = 1/(2 f_Rabi).")
        self.sp_frabi.valueChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("f_Rabi"), 0, 0); lay.addWidget(self.sp_frabi, 0, 1)

        self.sp_t0 = QDoubleSpinBox()
        self.sp_t0.setRange(0.0, 1000.0); self.sp_t0.setDecimals(3)
        self.sp_t0.setSingleStep(0.1); self.sp_t0.setValue(0.0)
        self.sp_t0.setSuffix(" us"); self.sp_t0.setKeyboardTracking(False)
        self.sp_t0.setToolTip(
            "Start time of the pulse WITHIN the beat cycle.\n"
            "Without a fixed timing the pulse area varies from shot to shot -\n"
            "at the working point by up to a factor of 73. The pulse must\n"
            "therefore be triggered on the AWG waveform.")
        self.sp_t0.valueChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("Pulse start t_0"), 1, 0); lay.addWidget(self.sp_t0, 1, 1)

        self.cmb_rabi_law = QComboBox()
        self.cmb_rabi_law.addItems(["Omega ~ I (two-photon Raman)",
                                    "Omega ~ sqrt(I) (single photon / one branch)"])
        self.cmb_rabi_law.setToolTip(
            "If both Raman branches come from this profile, Omega ~ I.\n"
            "If this profile is only ONE branch, Omega ~ E ~ sqrt(I).\n"
            "That changes the weighting and hence the uniformity.")
        self.cmb_rabi_law.currentIndexChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("Coupling"), 2, 0); lay.addWidget(self.cmb_rabi_law, 2, 1)

        self.sp_eta = QDoubleSpinBox()
        self.sp_eta.setRange(0.0, 5.0); self.sp_eta.setDecimals(3)
        self.sp_eta.setSingleStep(0.05); self.sp_eta.setValue(0.0)
        self.sp_eta.setKeyboardTracking(False)
        self.sp_eta.setToolTip(
            "Differential light shift divided by the Rabi frequency.\n\n"
            "For a two-photon Raman transition both scale with the SAME\n"
            "intensity, so their ratio is constant in space and time. The\n"
            "light shift therefore adds no extra spatial non-uniformity - it\n"
            "only caps the contrast at 1/(1+eta^2) and rescales the Rabi\n"
            "frequency by sqrt(1+eta^2). That closed form is exact and is what\n"
            "the Rabi panel uses.\n\n"
            "Order of magnitude for a Lambda system: eta ~ omega_HF/(2*Delta),\n"
            "so 3.035 GHz over twice the Raman detuning. eta = 0 assumes the\n"
            "shift is perfectly compensated.")
        self.sp_eta.valueChanged.connect(self._on_param_changed)
        lay.addWidget(QLabel("light shift eta"), 3, 0); lay.addWidget(self.sp_eta, 3, 1)

        self.btn_snap_flat = QPushButton("Move t_0 to a flat point of the area")
        self.btn_snap_flat.setToolTip(
            "Puts the pulse start on a stationary point of the pulse area,\n"
            "where dA/dt_0 = 0 and timing jitter therefore only enters in\n"
            "second order.\n\n"
            "This is worth far more than the last percent of uniformity: at\n"
            "0.1 MHz the tolerance for a 1 % area error rises from 44 ns to\n"
            "12.8 us, while the uniformity only goes from 18.3 to 19.9 %.")
        self.btn_snap_flat.clicked.connect(self._on_snap_flat)
        lay.addWidget(self.btn_snap_flat, 4, 0, 1, 2)

        self.btn_opt_pulse = QPushButton("Optimise phases and t_0 for the pulse area")
        self.btn_opt_pulse.setToolTip(
            "Minimises the uniformity of the accumulated Rabi area in the\n"
            "selected target region - the quantity that really counts for\n"
            "pulsed driving. Optimises tone phases AND pulse timing jointly.\n"
            "Takes about half a minute.")
        self.btn_opt_pulse.clicked.connect(self._on_optimize_pulse)
        lay.addWidget(self.btn_opt_pulse, 5, 0, 1, 2)

        self.lbl_pulse = QLabel("-")
        self.lbl_pulse.setWordWrap(True)
        self.lbl_pulse.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_pulse, 6, 0, 1, 2)
        return g

    def _group_time(self):
        g = QGroupBox("Time axis")
        lay = QGridLayout(g)
        self.sp_periods = self._ispin(self.state["n_periods"], 1, 50)
        self.sp_fpp = self._ispin(self.state["frames_per_period"], 8, 400)
        self.sp_grid = self._ispin(self.state["grid_n"], 60, 600)
        lay.addWidget(QLabel("Periods"), 0, 0); lay.addWidget(self.sp_periods, 0, 1)
        self.sp_fpp.setToolTip(
            "Must be larger than 2 * (highest beat frequency / f_0),\n"
            "otherwise the beating is sampled incorrectly (aliasing) and\n"
            "even the time average is wrong. The GUI warns when there are\n"
            "too few and states the required number.")
        lay.addWidget(QLabel("Frames/period"), 1, 0); lay.addWidget(self.sp_fpp, 1, 1)
        lay.addWidget(QLabel("Grid resolution"), 2, 0); lay.addWidget(self.sp_grid, 2, 1)

        self.slider_t = QSlider(Qt.Horizontal)
        self.slider_t.setMinimum(0)
        self.slider_t.setMaximum(0)
        self.slider_t.valueChanged.connect(self._on_time_slider)
        lay.addWidget(QLabel("t"), 4, 0); lay.addWidget(self.slider_t, 4, 1)

        row = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._on_play_clicked)
        row.addWidget(self.btn_play)
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 60)      # frames per second
        self.slider_speed.setValue(20)
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        row.addWidget(QLabel("fps"))
        row.addWidget(self.slider_speed)
        holder = QWidget(); holder.setLayout(row)
        lay.addWidget(holder, 5, 0, 1, 2)

        self.lbl_beat = QLabel("-")
        self.lbl_beat.setWordWrap(True)
        self.lbl_beat.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_beat, 6, 0, 1, 2)

        self.lbl_degen = QLabel("-")
        self.lbl_degen.setWordWrap(True)
        self.lbl_degen.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_degen, 7, 0, 1, 2)

        row_nu = QHBoxLayout()
        row_nu.addWidget(QLabel("Trap frequency nu_r"))
        self.sp_nu_r = QDoubleSpinBox()
        self.sp_nu_r.setRange(0.1, 5000.0); self.sp_nu_r.setDecimals(1)
        self.sp_nu_r.setValue(60.4); self.sp_nu_r.setSuffix(" kHz")
        self.sp_nu_r.setKeyboardTracking(False)
        self.sp_nu_r.setToolTip(
            "Only for checking whether a beat line falls on nu_r or\n"
            "2*nu_r. The spectrum is discrete - if no line sits there,\n"
            "the trap receives hardly any power despite full modulation.")
        self.sp_nu_r.valueChanged.connect(self._on_param_changed)
        row_nu.addWidget(self.sp_nu_r)
        holder_nu = QWidget(); holder_nu.setLayout(row_nu)
        lay.addWidget(holder_nu, 8, 0, 1, 2)

        self.lbl_res = QLabel("")
        self.lbl_res.setWordWrap(True)
        self.lbl_res.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_res, 9, 0, 1, 2)
        return g

    def _group_phases(self):
        g = QGroupBox("Tone phases")
        outer = QVBoxLayout(g)

        btns = QHBoxLayout()
        for label, fn in (("0", "zero"), ("Schroeder", "schroeder"),
                          ("randomise", "random")):
            b = QPushButton(label)
            b.clicked.connect(lambda _, k=fn: self._apply_phase_preset(k))
            btns.addWidget(b)
        holder = QWidget(); holder.setLayout(btns)
        outer.addWidget(holder)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target region"))
        self.cmb_opt_region = QComboBox()
        self.cmb_opt_region.addItems(["Plateau (<I> > 50 % max)", "Spot centres",
                                      "Circle around the centre"])
        self.cmb_opt_region.setCurrentIndex(2)   # default: the 2 um circle
        row.addWidget(self.cmb_opt_region)
        self.sp_opt_radius = QDoubleSpinBox()
        self.sp_opt_radius.setRange(0.1, 20.0); self.sp_opt_radius.setDecimals(2)
        self.sp_opt_radius.setSingleStep(0.1); self.sp_opt_radius.setValue(2.0)
        self.sp_opt_radius.setSuffix(" um")
        row.addWidget(self.sp_opt_radius)
        holder2 = QWidget(); holder2.setLayout(row)
        outer.addWidget(holder2)

        self.cb_quad = QCheckBox("keep degenerate pairs in quadrature")
        self.cb_quad.setChecked(True)
        self.cb_quad.setToolTip(
            "Frequency degenerate spots have a cross term at 0 Hz that never\n"
            "averages away - the static distortion the incoherent GUIs do\n"
            "not see.\n\n"
            "At a phase difference of 90 degrees this term is EXACTLY zero.\n"
            "Then - and only then - the time average is exactly the\n"
            "incoherent sum. Costs almost nothing: the uniformity at the\n"
            "optimum goes from 23.5 to 24.3 percent.")
        self.cb_quad.stateChanged.connect(self._on_param_changed)
        outer.addWidget(self.cb_quad)

        self.phase_grid_host = QWidget()
        self.phase_grid = QGridLayout(self.phase_grid_host)
        self.phase_grid.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.phase_grid_host)
        self.phase_spins_x, self.phase_spins_y = [], []
        self._rebuild_phase_fields()

        self.lbl_phase = QLabel("-")
        self.lbl_phase.setWordWrap(True)
        self.lbl_phase.setStyleSheet("color: #555; font-size: 10px;")
        outer.addWidget(self.lbl_phase)
        return g

    def _rebuild_phase_fields(self):
        """Creates the input fields anew when N_x or N_y changes.
        Existing values are carried over as far as possible."""
        while self.phase_grid.count():
            item = self.phase_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.phase_spins_x, self.phase_spins_y = [], []
        N_x, N_y = self.state["N_x"], self.state["N_y"]
        for name, N, key, store in (("phi_x", N_x, "phase_x", self.phase_spins_x),
                                    ("phi_y", N_y, "phase_y", self.phase_spins_y)):
            old = np.asarray(self.state[key], dtype=float)
            vals = np.zeros(N)
            vals[:min(N, old.size)] = old[:min(N, old.size)]
            self.state[key] = vals
            col = 0 if key == "phase_x" else 1
            self.phase_grid.addWidget(QLabel(name + " (degrees)"), 0, col)
            for i in range(N):
                sp = QDoubleSpinBox()
                sp.setRange(-720.0, 720.0)
                sp.setDecimals(1)
                sp.setSingleStep(15.0)
                sp.setWrapping(True)
                sp.setKeyboardTracking(False)
                sp.setValue(float(np.degrees(vals[i])))
                sp.valueChanged.connect(self._on_param_changed)
                self.phase_grid.addWidget(sp, i + 1, col)
                store.append(sp)

    def _apply_phase_preset(self, kind):
        N_x, N_y = self.state["N_x"], self.state["N_y"]
        if kind == "zero":
            px, py = np.zeros(N_x), np.zeros(N_y)
        elif kind == "schroeder":
            px, py = schroeder_phases(N_x), schroeder_phases(N_y)
        else:
            rng = np.random.default_rng()
            px = rng.uniform(0, 2 * np.pi, N_x)
            py = rng.uniform(0, 2 * np.pi, N_y)
        self._write_phase_fields(px, py)
        self.recompute()

    def _write_phase_fields(self, px, py):
        self.state["phase_x"], self.state["phase_y"] = np.asarray(px), np.asarray(py)
        for sp, v in zip(self.phase_spins_x, px):
            sp.blockSignals(True); sp.setValue(float(np.degrees(v)) % 360.0); sp.blockSignals(False)
        for sp, v in zip(self.phase_spins_y, py):
            sp.blockSignals(True); sp.setValue(float(np.degrees(v)) % 360.0); sp.blockSignals(False)

    def _target_mask(self, mean_map, X, Y, centers_x, centers_y, x, y):
        """Region that the phase optimisation targets."""
        idx = self.cmb_opt_region.currentIndex()
        if idx == 0:
            return mean_map > 0.5 * mean_map.max() if mean_map.max() > 0 \
                else np.ones_like(mean_map, bool)
        if idx == 1:
            m = np.zeros(mean_map.shape, bool)
            for cxi, cyi in zip(centers_x, centers_y):
                m[int(np.argmin(np.abs(y - cyi))), int(np.argmin(np.abs(x - cxi)))] = True
            return m
        r = self.sp_opt_radius.value() * 1e-6
        cx0, cy0 = float(np.mean(centers_x)), float(np.mean(centers_y))
        m = ((X - cx0) ** 2 + (Y - cy0) ** 2) <= r ** 2
        return m if m.any() else np.ones_like(mean_map, bool)

    def _pulse_setup(self):
        """Common setup for everything that has to do with the pulse."""
        s = self.state
        cxs, cys, f_spots, _, _, _, _ = compute_centers_and_freqs(
            s["N_x"], s["N_y"], s["width_x"], s["width_y"], s["f1"], s["f2"], s["offset"])
        f0 = fundamental_beat_frequency(f_spots)
        amp = amp_spots_from_ratios(s["r_x"], s["r_y"], s["N_x"], s["N_y"])
        win_eff = s["win"] * (s["airy_factor"] if s["use_airy"] else 1.0)
        xg, yg, Xg, Yg = compute_grid(cxs, cys, win_eff, 120)
        Fg = build_field_stack(Xg, Yg, cxs, cys, amp, s["win"], s["use_airy"],
                               s["airy_factor"])
        k = beat_orders(f_spots, f0) if f0 > 0 else np.zeros(len(f_spots), int)
        mean0, _ = time_stats_exact(Fg, k, np.zeros(len(f_spots)))
        mask = self._target_mask(mean0, Xg, Yg, cxs, cys, xg, yg)
        return dict(f_spots=f_spots, f0=f0, F=Fg, k=k, mask=mask, mean0=mean0,
                    T0=(1.0 / f0 if f0 > 0 else float("nan")))

    def _on_snap_flat(self):
        """Move the pulse start to the flattest useful point of the area curve."""
        pz = self.cache.get("pulse")
        if pz is None or "t0_flat" not in pz:
            self.lbl_status.setText("No pulse data available.")
            return
        self.sp_t0.blockSignals(True)
        self.sp_t0.setValue(pz["t0_flat"] * 1e6)
        self.sp_t0.blockSignals(False)
        self.state["pulse_t0"] = pz["t0_flat"]
        self._opt_note = (f"t_0 moved to the flat point at {pz['t0_flat'] * 1e6:.3f} us "
                          f"(U there {pz['u_flat'] * 100:.1f} %)")
        self.recompute()

    def _on_optimize_pulse(self):
        """Searches for tone phases AND pulse timing with the most uniform
        pulse area.

        This is the only optimiser left. The earlier objectives (broadband
        time variation, peak intensity, quiet window) have been removed:
        they all ended up at nearly the same phase set, and for pulsed
        operation the pulse area is the relevant quantity anyway."""
        from scipy.optimize import minimize
        self._read_widgets()
        s = self.state
        N_x, N_y = s["N_x"], s["N_y"]
        n_free = max(0, N_x - 1) + max(0, N_y - 1)
        st = self._pulse_setup()
        if st["f0"] <= 0 or n_free == 0:
            self.lbl_status.setText("No beating or no free phases.")
            return
        T0 = st["T0"]
        t_p = 1.0 / (2.0 * s["f_rabi"])
        pa = PulseArea(st["F"], st["k"], st["f0"], st["mask"], law=s["rabi_law"])
        degen_g = degenerate_groups(st["f_spots"], st["f0"])

        def cost(v):
            px = np.concatenate(([0.0], v[:N_x - 1])) if N_x > 1 else np.zeros(1)
            py = np.concatenate(([0.0], v[N_x - 1:n_free])) if N_y > 1 else np.zeros(1)
            ph = spot_phases_from_tones(px, py, N_x, N_y)
            u = pa.uniformity(ph, v[-1] % T0, t_p, st["f_spots"])
            if not np.isfinite(u):
                return 1e6
            pen = (2.0 * quadrature_penalty(ph, degen_g)
                   if self.cb_quad.isChecked() else 0.0)
            return u + pen

        self.lbl_status.setText("searching phases and pulse timing ...")
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        QApplication.processEvents()
        rng = np.random.default_rng(3)
        n_start = 35 if s["rabi_law"] == "I" else 12
        best_f, best_v = 1e18, None
        for n in range(n_start):
            v0 = np.concatenate((rng.uniform(0, 2 * np.pi, n_free), [rng.uniform(0, T0)]))
            r = minimize(cost, v0, method="Nelder-Mead",
                         options=dict(maxiter=2500, xatol=1e-7, fatol=1e-10))
            if r.fun < best_f:
                best_f, best_v = float(r.fun), r.x
            if n % 10 == 0:
                QApplication.processEvents()
        px = np.concatenate(([0.0], best_v[:N_x - 1])) if N_x > 1 else np.zeros(1)
        py = np.concatenate(([0.0], best_v[N_x - 1:n_free])) if N_y > 1 else np.zeros(1)
        self._write_phase_fields(px, py)
        t0 = float(best_v[-1] % T0)
        self.sp_t0.blockSignals(True); self.sp_t0.setValue(t0 * 1e6); self.sp_t0.blockSignals(False)
        self.state["pulse_t0"] = t0
        self._opt_note = (f"Pulse area: U = {best_f * 100:.1f} % at t_0 = {t0 * 1e6:.3f} us "
                          f"(T_pi = {t_p * 1e6:.3f} us)")
        self.recompute()

    def _group_actions(self):
        g = QGroupBox("Actions")
        lay = QVBoxLayout(g)
        self.cb_auto = QCheckBox("Recompute automatically")
        self.cb_auto.setChecked(self.state["auto_update"])
        self.cb_auto.stateChanged.connect(self._on_auto_changed)
        lay.addWidget(self.cb_auto)
        self.btn_update = QPushButton("Recompute")
        self.btn_update.clicked.connect(lambda: self.recompute())
        lay.addWidget(self.btn_update)
        lay.addWidget(QLabel("Panel top right"))
        self.cmb_panel = QComboBox()
        self.cmb_panel.addItems([
            "Space-time map I(x, t)",
            "Enhancement n_eff = I_max / <I>",
            "Modulation depth (I_max-I_min)/(I_max+I_min)",
            "temporal variation sigma_t / <I>",
            "Spectrum of the beating",
            "Uniformity U(t) of the three regions",
            "Pulse area: U over the pulse start",
            "Pulse area A(t_0) and flat points",
            "Rabi oscillation",
        ])
        self.cmb_panel.setToolTip(
            "n_eff is the number of tones effectively overlapping at the\n"
            "respective position: 1 where a single spot dominates, up to the\n"
            "number of spots in the plateau. It is at the same time the\n"
            "factor by which the instantaneous intensity exceeds the time\n"
            "average on rephasing.")
        self.cmb_panel.currentIndexChanged.connect(lambda _: self.draw_frame(full=True))
        self.cb_fastdraw = QCheckBox("Fast drawing during playback")
        self.cb_fastdraw.setChecked(True)
        self.cb_fastdraw.setToolTip(
            "When stepping forward only the data of the existing drawing\n"
            "objects are exchanged instead of rebuilding all four axes -\n"
            "about twice as fast.\n\n"
            "If the display behaves strangely: uncheck, then every image is\n"
            "redrawn completely.")
        self.cb_fastdraw.stateChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cb_fastdraw)

        self.cb_live = QCheckBox("record U(t) live")
        self.cb_live.setChecked(True)
        self.cb_live.setToolTip(
            "In the 'Uniformity U(t)' panel the curve is only drawn up to the\n"
            "current instant and grows along with the animation; the rest of\n"
            "the trace stands pale behind it. The numbers in the title are\n"
            "the instantaneous values.")
        self.cb_live.stateChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cb_live)
        lay.addWidget(self.cmb_panel)
        lay.addWidget(QLabel("Colour scale"))
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems([
            "fixed: 99.5 percentile over time",
            "fixed: maximum over time",
            "fixed: maximum of the time average",
            "per frame",
        ])
        self.cmb_scale.setToolTip(
            "With all tone phases at 0, all tones rephase once per\n"
            "fundamental period into a short pulse that exceeds the time\n"
            "average by a large factor. A fixed scale on THIS maximum leaves\n"
            "all remaining frames almost black - which is why the percentile\n"
            "is the default. 'per frame' shows every image fully scaled, but\n"
            "makes frames incomparable to each other.")
        self.cmb_scale.currentIndexChanged.connect(lambda _: self.draw_frame(full=True))
        lay.addWidget(self.cmb_scale)
        self.btn_save = QPushButton("Save view as PDF")
        self.btn_save.clicked.connect(self._on_save_clicked)
        lay.addWidget(self.btn_save)
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        lay.addWidget(self.lbl_status)
        return g

    # --------------------------------------------------------
    # Inputs -> state
    # --------------------------------------------------------
    def _sync_winmode_enabled(self):
        out = self.state["win_mode"] == "output"
        self.sp_win.setEnabled(out)
        self.sp_win_in.setEnabled(not out)

    def _sync_width_enabled(self):
        self.sp_width_y.setEnabled(not self.cb_link_width.isChecked())

    def _on_link_width_changed(self, _):
        self.state["link_width"] = self.cb_link_width.isChecked()
        self._sync_width_enabled()
        self._on_param_changed()

    def _on_winmode_changed(self, idx):
        self.state["win_mode"] = "output" if idx == 0 else "input"
        self._sync_winmode_enabled()
        self._on_param_changed()

    def _read_widgets(self):
        s = self.state
        s["N_x"] = self.sp_nx.value()
        s["N_y"] = self.sp_ny.value()
        s["use_airy"] = self.cmb_profile.currentIndex() == 1
        s["airy_factor"] = self.sp_airy.value()
        s["link_width"] = self.cb_link_width.isChecked()
        s["width_x"] = self.sp_width.value() * 1e6
        if s["link_width"]:
            s["width_y"] = s["width_x"]
            self.sp_width_y.blockSignals(True)
            self.sp_width_y.setValue(s["width_x"] * 1e-6)
            self.sp_width_y.blockSignals(False)
        else:
            s["width_y"] = self.sp_width_y.value() * 1e6
        s["lambda_opt"] = self.sp_lambda.value() * 1e-9
        s["offset"] = self.sp_offset.value() * 1e6
        s["f1"] = self.sp_f1.value() * 1e-3
        s["f2"] = self.sp_f2.value() * 1e-3
        s["r_x"] = self.sp_rx.value()
        s["r_y"] = self.sp_ry.value()
        s["grid_n"] = self.sp_grid.value()
        s["n_periods"] = self.sp_periods.value()
        s["frames_per_period"] = self.sp_fpp.value()
        if len(self.phase_spins_x) != s["N_x"] or len(self.phase_spins_y) != s["N_y"]:
            self._rebuild_phase_fields()
        s["f_rabi"] = self.sp_frabi.value() * 1e6
        s["pulse_t0"] = self.sp_t0.value() * 1e-6
        s["rabi_law"] = "I" if self.cmb_rabi_law.currentIndex() == 0 else "sqrtI"
        s["eta_ls"] = self.sp_eta.value()
        s["phase_x"] = np.radians([sp.value() for sp in self.phase_spins_x])
        s["phase_y"] = np.radians([sp.value() for sp in self.phase_spins_y])

        # Waist: whichever quantity is active is the control variable, the
        # other one is updated accordingly and shown in its field.
        if s["win_mode"] == "output":
            s["win"] = self.sp_win.value() * 1e-6
            s["win_in"] = conjugate_waist(s["win"], s["f1"], s["f2"], s["lambda_opt"])
            self.sp_win_in.blockSignals(True)
            self.sp_win_in.setValue(s["win_in"] * 1e3)
            self.sp_win_in.blockSignals(False)
        else:
            s["win_in"] = self.sp_win_in.value() * 1e-3
            s["win"] = conjugate_waist(s["win_in"], s["f1"], s["f2"], s["lambda_opt"])
            self.sp_win.blockSignals(True)
            self.sp_win.setValue(s["win"] * 1e6)
            self.sp_win.blockSignals(False)

    def _on_param_changed(self, *args):
        if self._building:
            return
        if self.state["auto_update"]:
            self.recompute()
        else:
            self.lbl_status.setText("Parameter changed - press 'Recompute'.")

    def _on_auto_changed(self, _):
        self.state["auto_update"] = self.cb_auto.isChecked()
        if self.state["auto_update"]:
            self.recompute()

    def _on_speed_changed(self, val):
        if self.timer.isActive():
            self.timer.start(max(1, int(1000 / val)))

    def _on_play_clicked(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.timer.start(max(1, int(1000 / self.slider_speed.value())))
            self.btn_play.setText("Pause")

    def _on_timer_tick(self):
        n = self.cache.get("n_frames", 0)
        if n == 0:
            return
        self.frame_idx = (self.frame_idx + 1) % n
        self.slider_t.blockSignals(True)
        self.slider_t.setValue(self.frame_idx)
        self.slider_t.blockSignals(False)
        self.draw_frame()

    def _on_time_slider(self, val):
        self.frame_idx = int(val)
        self.draw_frame()

    def _on_canvas_click(self, event):
        """A click in the 2D map moves the crosshair - the cut, the
        space-time map and I(t) then refer to that point."""
        if event.inaxes is not self.ax_main or not self.cache:
            return
        if event.xdata is None or event.ydata is None:
            return
        x_um = self.cache["x"] * 1e6
        y_um = self.cache["y"] * 1e6
        self.cache["col"] = int(np.argmin(np.abs(x_um - event.xdata)))
        self.cache["row"] = int(np.argmin(np.abs(y_um - event.ydata)))
        self._rebuild_traces()
        self.draw_frame(full=True)

    # --------------------------------------------------------
    # Computation
    # --------------------------------------------------------
    def recompute(self):
        self._read_widgets()
        s = self.state

        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        centers_x, centers_y, f_spots, r_center_x, r_center_y, fx_freq, fy_freq = \
            compute_centers_and_freqs(s["N_x"], s["N_y"], s["width_x"], s["width_y"],
                                      s["f1"], s["f2"], s["offset"])
        amp_spots = amp_spots_from_ratios(s["r_x"], s["r_y"], s["N_x"], s["N_y"])
        n_spots = len(f_spots)

        win_eff = s["win"] * (s["airy_factor"] if s["use_airy"] else 1.0)
        x, y, X, Y = compute_grid(centers_x, centers_y, win_eff, s["grid_n"])

        F = build_field_stack(X, Y, centers_x, centers_y, amp_spots,
                              s["win"], s["use_airy"], s["airy_factor"])

        f0 = fundamental_beat_frequency(f_spots)
        degen = degenerate_groups(f_spots, f0)
        # Time window. f0 > 0: the signal is strictly periodic with T0 = 1/f0,
        # the window then covers whole periods and the time average over the
        # window is exactly the time average as such.
        # f0 == 0 with beats present: the difference frequencies are
        # incommensurable (separate widths), there is no common period at
        # all. Then we take 1/f_min as a display scale and say so - the
        # window average is only an approximation in that case.
        beats = unique_beat_frequencies(f_spots)
        need_fpp = min_frames_per_period(f_spots, f0)
        n_frames = s["n_periods"] * s["frames_per_period"]
        periodic = f0 > 0
        if periodic:
            T0 = 1.0 / f0
        elif beats.size:
            T0 = 1.0 / beats[0]
        else:
            T0 = float("nan")
        if np.isfinite(T0):
            # open interval: the last frame is NOT identical to the first,
            # otherwise the loop stutters and the time average would get one
            # point twice.
            t = np.arange(n_frames) / n_frames * (s["n_periods"] * T0)
        else:
            t = np.zeros(1)
            n_frames = 1

        phases = spot_phases_from_tones(s["phase_x"], s["phase_y"], s["N_x"], s["N_y"])

        cost = n_frames * len(f_spots) * X.size
        if cost > 4e9:
            QMessageBox.warning(self, "Too large",
                                "This combination of grid, frames and number of tones "
                                "would be very slow. Please reduce the grid resolution "
                                "or the frames per period.")
            return
        self.lbl_status.setText("computing ...")
        QApplication.processEvents()

        cube = intensity_cube(F, f_spots, phases, t)

        # References. I_avg is the time average; over a whole fundamental
        # period it must be exactly the incoherent sum sum_s A_s^2 |u_s|^2 of
        # the previous GUIs - that is precisely what resid checks.
        I_avg = cube.mean(axis=0).astype(np.float64)
        I_incoh = np.einsum("sij,sij->ij", F, F)
        denom = np.max(I_incoh) if np.max(I_incoh) > 0 else 1.0
        resid = float(np.max(np.abs(I_avg - I_incoh)) / denom)

        norm = float(np.max(I_incoh)) if np.max(I_incoh) > 0 else 1.0

        # Enhancement and modulation depth. n_eff = I_max/<I> is the number of
        # tones effectively overlapping at the respective position: 1 where a
        # single spot dominates, up to the number of spots where all
        # contribute equally.
        k_orders = beat_orders(f_spots, f0) if f0 > 0 else np.zeros(len(f_spots), int)
        mean_exact, var_exact = time_stats_exact(F, k_orders, phases)
        sigma_rel = np.sqrt(np.maximum(var_exact, 0.0)) / np.maximum(mean_exact, 1e-300)

        I_max_map = cube.max(axis=0).astype(np.float64)
        I_min_map = cube.min(axis=0).astype(np.float64)
        n_eff = I_max_map / np.maximum(I_avg, 1e-300)
        depth = (I_max_map - I_min_map) / np.maximum(I_max_map + I_min_map, 1e-300)
        plateau = I_avg > 0.5 * I_avg.max() if I_avg.max() > 0 else np.zeros_like(I_avg, bool)
        if plateau.any():
            try:
                obj_spec = VariationObjective(F, k_orders, plateau)
                spectrum = obj_spec.components(phases)
            except Exception:
                spectrum = {}
            sigma_rms = float(np.sqrt(np.mean(sigma_rel[plateau] ** 2)))
            # Uniformity U(t) for the three evaluation regions. The reference
            # is U(<I>) in each case - exactly the number the other GUIs and
            # the scan pipeline report.
            sites_mask = np.zeros(I_avg.shape, bool)
            for cxi, cyi in zip(centers_x, centers_y):
                sites_mask[int(np.argmin(np.abs(y - cyi))),
                           int(np.argmin(np.abs(x - cxi)))] = True
            r_circ = self.sp_opt_radius.value() * 1e-6
            circ_mask = ((X - float(np.mean(centers_x))) ** 2
                         + (Y - float(np.mean(centers_y))) ** 2) <= r_circ ** 2
            u_regions = {
                "Plateau": plateau,
                "Spot centres": sites_mask,
                f"Circle r={r_circ * 1e6:.1f} um": circ_mask,
            }
            # --- pulsed operation: uniformity of the accumulated Rabi area ---
            try:
                t_p = 1.0 / (2.0 * s["f_rabi"])
                pa = PulseArea(F, k_orders, f0, circ_mask, law=s["rabi_law"])
                t0_scan = np.linspace(0.0, T0, 90, endpoint=False) if periodic else np.zeros(1)
                u_t0 = np.array([pa.uniformity(phases, tt, t_p, f_spots) for tt in t0_scan])
                area_t0 = np.array([pa.theta(phases, tt, t_p, f_spots).mean()
                                    for tt in t0_scan])
                # Sensitivity of the pulse area to timing jitter. On a flat
                # part of the area curve the jitter only enters in second
                # order - that is worth far more than the last percent of
                # uniformity, see _on_snap_flat().
                a_mean = float(np.mean(area_t0)) if area_t0.size else 1.0
                dA = (np.gradient(area_t0, t0_scan) if len(t0_scan) > 2
                      else np.zeros_like(area_t0))
                i_now = int(np.argmin(np.abs(t0_scan - s["pulse_t0"])))
                slope = abs(float(dA[i_now])) if dA.size else 0.0
                jit = (0.01 * a_mean / slope) if slope > 0 else float("inf")
                flat = np.flatnonzero(np.diff(np.sign(dA)) != 0) if dA.size else np.array([])
                i_flat = int(flat[int(np.argmin(u_t0[flat]))]) if flat.size else i_now
                pulse = dict(t_p=t_p, t0=s["pulse_t0"], t0_scan=t0_scan, u=u_t0,
                             area=area_t0, area_norm=area_t0 / max(a_mean, 1e-300),
                             dA=dA, jitter=jit, i_flat=i_flat,
                             t0_flat=float(t0_scan[i_flat]), u_flat=float(u_t0[i_flat]),
                             u_now=pa.uniformity(phases, s["pulse_t0"], t_p, f_spots),
                             theta=pa.theta(phases, s["pulse_t0"], t_p, f_spots),
                             u_ref=uniformity_of(mean_exact, circ_mask))
            except Exception:
                pulse = None

            u_series, u_ref = {}, {}
            for rname, rmask in u_regions.items():
                if not rmask.any():
                    continue
                try:
                    u_series[rname] = UniformitySeries(F, f_spots, rmask, t).series(phases)
                    u_ref[rname] = uniformity_of(mean_exact, rmask)
                except Exception:
                    pass
            n_eff_med = float(np.median(n_eff[plateau]))
            depth_med = float(np.median(depth[plateau]))
            depth_min = float(depth[plateau].min())
        else:
            n_eff_med = depth_med = depth_min = sigma_rms = float("nan")
            spectrum = {}
            u_series, u_ref = {}, {}
            pulse = None
        crest_x = crest_factor(fx_freq, s["phase_x"])
        crest_y = crest_factor(fy_freq, s["phase_y"])

        self.cache = {
            "x": x, "y": y, "X": X, "Y": Y,
            "centers_x": centers_x, "centers_y": centers_y, "f_spots": f_spots,
            "r_center_x": r_center_x, "r_center_y": r_center_y,
            "fx_freq": fx_freq, "fy_freq": fy_freq,
            "amp_spots": amp_spots, "t": t, "n_frames": n_frames,
            "cube": cube, "I_avg": I_avg, "I_incoh": I_incoh,
            "norm": norm, "resid": resid, "T0": T0, "f0": f0, "degen": degen,
            "periodic": periodic, "beats": beats, "need_fpp": need_fpp,
            "I_max_map": I_max_map, "I_min_map": I_min_map,
            "n_eff": n_eff, "depth": depth, "plateau": plateau,
            "sigma_rel": sigma_rel, "sigma_rms": sigma_rms, "spectrum": spectrum,
            "F_stack": F, "k_orders": k_orders, "region_mask": circ_mask,
            "u_series": u_series, "u_ref": u_ref, "pulse": pulse,
            "mean_exact": mean_exact, "var_exact": var_exact,
            "n_eff_med": n_eff_med, "depth_med": depth_med, "depth_min": depth_min,
            "crest_x": crest_x, "crest_y": crest_y, "phases": phases,
            "row": int(np.argmin(np.abs(y - r_center_y))),
            "col": int(np.argmin(np.abs(x - r_center_x))),
            "cube_max": float(cube.max()),
            "cube_p995": float(np.percentile(cube, 99.5)),
        }
        self._rebuild_traces()

        self.frame_idx = min(self.frame_idx, n_frames - 1)
        self.slider_t.blockSignals(True)
        self.slider_t.setMaximum(max(0, n_frames - 1))
        self.slider_t.setValue(self.frame_idx)
        self.slider_t.blockSignals(False)

        crest_txt = ("Crest factor RF: not defined (free spot phases)"
                     if not np.isfinite(crest_x)
                     else f"Crest factor RF: x {crest_x:.2f}, y {crest_y:.2f}")
        self.lbl_phase.setText(
            f"Peak: {cube.max() / norm:.2f} x max<I>   |   " + crest_txt + "\n"
            f"In the plateau: sigma_t/<I> = {sigma_rms * 100:.1f} %,  "
            f"n_eff (overlap) = {n_eff_med:.1f},  "
            f"modulation depth = {depth_med * 100:.1f} %\n"
            + (f"Degenerate pairs: phase difference "
               f"{np.degrees(phases[degen[0][0]] - phases[degen[0][1]]) % 180:.1f} degrees "
               f"-> static contribution {resid * 100:.2f} % "
               f"({'in quadrature, time average = incoherent sum' if resid < 1e-6 else 'not in quadrature'})\n"
               if degen else "")
            + (self._opt_note + "\n" if self._opt_note else "")
            + "Phases can lower sigma_t/<I> (cross terms of equal difference "
              "frequency partially cancel), but not to zero.")
        self._opt_note = ""
        if pulse is not None:
            sw = (pulse["area"].max() / pulse["area"].min()
                  if pulse["area"].min() > 0 else float("inf"))
            jt = pulse["jitter"]
            jtxt = ("insensitive to first order" if not np.isfinite(jt)
                    else f"{jt * 1e9:.0f} ns per 1 % area error")
            self.lbl_pulse.setText(
                f"pi pulse T = {pulse['t_p'] * 1e6:.3f} us "
                f"({pulse['t_p'] / T0 * 100:.1f} % of one beat period)\n"
                f"U(pulse area) at t_0 = {s['pulse_t0'] * 1e6:.3f} us: "
                f"{pulse['u_now'] * 100:.1f} %   |   best t_0: {pulse['u'].min() * 100:.1f} % "
                f"at {pulse['t0_scan'][int(np.argmin(pulse['u']))] * 1e6:.3f} us\n"
                f"timing tolerance here: {jtxt}   |   flat point at "
                f"{pulse['t0_flat'] * 1e6:.3f} us (U {pulse['u_flat'] * 100:.1f} %)\n"
                f"without a fixed pulse timing the area varies by a factor of {sw:.1f} - "
                f"the pulse MUST be triggered on the AWG waveform.\n"
                f"Reference U from the time average: {pulse['u_ref'] * 100:.1f} %")
            self.lbl_pulse.setStyleSheet(
                "color: #a00; font-size: 10px;" if sw > 1.5
                else "color: #555; font-size: 10px;")
        self._update_labels()
        self.draw_frame(full=True)
        if degen:
            note = (f"Time average deviates by {resid * 100:.3f} % from the incoherent "
                    f"picture - this is NOT a computational error, but the static "
                    f"interference of the frequency degenerate spots.")
        else:
            note = (f"Time average == incoherent sum to within {resid:.1e} - "
                    f"the previous picture is exactly the time average.")
        if periodic and s["frames_per_period"] < need_fpp:
            note += (f"\nWARNING: {s['frames_per_period']} frames/period are not enough. "
                     f"The fastest beat frequency is {beats[-1] * 1e-3:.1f} kHz = "
                     f"{beats[-1] / f0:.0f} x f_0; Nyquist requires at least "
                     f"{need_fpp}. Otherwise the time average and the envelopes are "
                     f"wrong (aliasing).")
        self.lbl_status.setText(
            f"done: {n_frames} frames, {len(f_spots)} spots, grid {s['grid_n']}^2.\n" + note)
        if periodic and s["frames_per_period"] < need_fpp:
            self.lbl_status.setStyleSheet("color: #a00; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("color: #555; font-size: 10px;")
        if was_running:
            self.timer.start(max(1, int(1000 / self.slider_speed.value())))

    def _rebuild_traces(self):
        """Cuts, space-time map and I(t) for the current crosshair."""
        c = self.cache
        if not c:
            return
        row, col = c["row"], c["col"]
        c["st_map"] = c["cube"][:, row, :]        # I(x, t) along the x cut
        c["trace"] = c["cube"][:, row, col]       # I(t) at the crosshair
        c["cut_x_min"] = c["cube"][:, row, :].min(axis=0)
        c["cut_x_max"] = c["cube"][:, row, :].max(axis=0)
        c["cut_y_min"] = c["cube"][:, :, col].min(axis=0)
        c["cut_y_max"] = c["cube"][:, :, col].max(axis=0)

    def _update_labels(self):
        c, s = self.cache, self.state
        fx = c["fx_freq"] * 1e-6
        fy = c["fy_freq"] * 1e-6
        self.lbl_freqs.setText(
            "f_x = " + ", ".join(f"{v:.4f}" for v in fx) + " MHz\n"
            "f_y = " + ", ".join(f"{v:.4f}" for v in fy) + " MHz\n"
            f"{len(c['f_spots'])} spots, pitch of the tones "
            f"{abs(c['centers_x'][-1] - c['centers_x'][0]) * 1e6 / max(1, s['N_x'] - 1):.3f} um in x")

        self.lbl_amps.setText(
            "amp_x = [" + ", ".join(f"{v:g}" for v in amps_from_ratio(s["r_x"], s["N_x"])) + "]\n"
            "amp_y = [" + ", ".join(f"{v:g}" for v in amps_from_ratio(s["r_y"], s["N_y"])) + "]\n"
            "(intensity weights; the field carries the square root)")

        beats = c["beats"]
        if c["periodic"]:
            shown = ", ".join(f"{b * 1e-3:.1f}" for b in beats[:8])
            more = " ..." if len(beats) > 8 else ""
            self.lbl_beat.setText(
                f"Fundamental frequency f_0 = gcd of all differences = {c['f0'] * 1e-3:.3f} kHz\n"
                f"Fundamental period T_0 = {c['T0'] * 1e6:.3f} us\n"
                f"Window = {s['n_periods']} x T_0 = {s['n_periods'] * c['T0'] * 1e6:.2f} us\n"
                f"Beat frequencies [kHz]: {shown}{more}")
        elif beats.size:
            shown = ", ".join(f"{b * 1e-3:.2f}" for b in beats[:8])
            more = " ..." if len(beats) > 8 else ""
            self.lbl_beat.setText(
                "NOT PERIODIC: the difference frequencies have no common\n"
                "multiple (separate widths). Shown is 1/f_min as a scale:\n"
                f"T_ref = {c['T0'] * 1e6:.3f} us,  window = {s['n_periods']} x T_ref = "
                f"{s['n_periods'] * c['T0'] * 1e6:.2f} us\n"
                f"The window average is therefore only an approximation.\n"
                f"Beat frequencies [kHz]: {shown}{more}")
        else:
            self.lbl_beat.setText("No beating: only one spot, or width = 0.")

        # What matters is not the size of the modulation, but whether a beat
        # line falls on nu_r or 2*nu_r: the spectrum is discrete, in between
        # the trap receives practically no power.
        nu = self.sp_nu_r.value() * 1e3
        d1, d2, crit = resonance_check(beats, nu)
        if np.isfinite(d1):
            txt = (f"Nearest beat line to nu_r ({nu * 1e-3:.1f} kHz): "
                   f"{d1 * 1e-3:.1f} kHz away;  to 2*nu_r: {d2 * 1e-3:.1f} kHz.")
            if crit:
                self.lbl_res.setText(
                    "CAUTION - " + txt + "\n"
                    "A line sits on a trap resonance. The spectrum is "
                    "discrete; that a line sits exactly there is the "
                    "dangerous case - not the modulation depth. Choose width "
                    "so that no multiple of f_0 falls there.")
                self.lbl_res.setStyleSheet("color: #a00; font-size: 10px; font-weight: bold;")
            else:
                self.lbl_res.setText(
                    txt + "\nNo line on a trap resonance - despite full "
                    "modulation depth the trap receives hardly any power there.")
                self.lbl_res.setStyleSheet("color: #060; font-size: 10px;")
        else:
            self.lbl_res.setText("")

        degen = c.get("degen", [])
        if degen:
            lines = []
            for grp in degen:
                pos = " = ".join(f"({c['centers_x'][i] * 1e6:.2f}, "
                                 f"{c['centers_y'][i] * 1e6:.2f})" for i in grp)
                lines.append(f"  {pos} um  at {c['f_spots'][grp[0]] * 1e-6:.5f} MHz")
            self.lbl_degen.setText(
                "CAUTION - frequency degenerate spots:\n" + "\n".join(lines) +
                "\nThese pairs have a cross term at 0 Hz. It never runs and "
                "never averages away: STATIC interference that the incoherent "
                "GUIs do not see. It can only be lifted by DIFFERENT widths "
                "in x and y - a constant frequency offset on one axis does "
                "not help, it cancels out of every difference.")
            self.lbl_degen.setStyleSheet(
                "color: #a00; font-size: 10px; font-weight: bold;")
        else:
            self.lbl_degen.setText("No frequency degenerate spots - every cross "
                                   "term runs, the time average is exactly the "
                                   "incoherent picture.")
            self.lbl_degen.setStyleSheet("color: #060; font-size: 10px;")

    # --------------------------------------------------------
    # Drawing
    # --------------------------------------------------------
    def _rabi_key(self, c):
        s = self.state
        return (s["N_x"], s["N_y"], s["width_x"], s["width_y"], s["win"], s["use_airy"],
                s["airy_factor"], s["r_x"], s["r_y"], s["f1"], s["f2"], s["offset"],
                s["f_rabi"], s["pulse_t0"], s["rabi_law"], s["eta_ls"], s["grid_n"],
                self.cmb_opt_region.currentIndex(), self.sp_opt_radius.value(),
                tuple(np.round(c.get("phases", []), 9)))

    def _draw_rabi_panel(self, c):
        """Excitation over pulse length - computed only when the panel is shown.

        The curves need a few hundred evaluations of the pulse area, which is
        too much to do on every recompute; the result is therefore cached and
        only rebuilt when something it depends on has changed."""
        pz = c.get("pulse")
        if pz is None or not c.get("periodic", False):
            self.ax_st.text(0.5, 0.5, "no pulse data available",
                            ha="center", va="center", transform=self.ax_st.transAxes)
            return
        key = self._rabi_key(c)
        if self._rabi_cache is None or self._rabi_cache[0] != key:
            self.lbl_status.setText("computing the Rabi curves ...")
            QApplication.processEvents()
            s = self.state
            t_pi = pz["t_p"]
            fast = s["rabi_law"] == "I"
            pa = PulseArea(c["F_stack"], c["k_orders"], c["f0"], c["region_mask"],
                           law=s["rabi_law"], n_t=60, max_points=350 if fast else 120)
            t_p = np.linspace(1e-12, 2.2 * t_pi, 90 if fast else 45)
            ideal, trig, none = pa.rabi_curves(c["phases"], pz["t0"], t_p, t_pi,
                                               c["f_spots"],
                                               n_random=14 if fast else 6, T0=c["T0"],
                                               eta=s["eta_ls"])
            self._rabi_cache = (key, t_p, ideal, trig, none)
            self.lbl_status.setText("")
        _, t_p, ideal, trig, none = self._rabi_cache
        ct = lambda P: (P.max() - P.min()) * 100
        self.ax_st.plot(t_p * 1e6, ideal * 100, color="#888", ls="--", lw=1.4,
                        label="ideal, no beating")
        self.ax_st.plot(t_p * 1e6, trig * 100, color="#4c8b5b", lw=2.0,
                        label=f"triggered at $t_0$ (contrast {ct(trig):.0f} %)")
        self.ax_st.plot(t_p * 1e6, none * 100, color="#1b1b1b", ls=":", lw=1.6,
                        label=f"untriggered (contrast {ct(none):.0f} %)")
        self.ax_st.axvline(pz["t_p"] * 1e6, color="#888", lw=0.9)
        self.ax_st.set_xlabel("pulse length (us)", fontsize=8)
        self.ax_st.set_ylabel("excitation (%)", fontsize=8)
        eta = self.state["eta_ls"]
        cap = ("" if eta <= 0 else
               f"   |   light shift eta = {eta:g} caps the contrast at "
               f"{100 / (1 + eta ** 2):.0f} %")
        self.ax_st.set_title("Rabi oscillation, averaged over the target region" + cap,
                             fontsize=8.5)
        self.ax_st.set_ylim(0, 105)
        self.ax_st.legend(fontsize=6.5, loc="lower right", framealpha=0.85)
        self.ax_st.spines["top"].set_visible(False)
        self.ax_st.spines["right"].set_visible(False)

    def _u_title(self, c, k):
        """Title of the U(t) panel with the INSTANTANEOUS values - the number
        one actually wants to read off while watching."""
        us = c.get("u_series", {})
        if not us:
            return "Uniformity over time"
        # Short, LENGTH-STABLE labelling: with blitting the layout no longer
        # follows, a growing title would run over the edge.
        short = {"Plateau": "Plat", "Spot centres": "Ctr"}
        now = " ".join(f"{short.get(rn, 'Circ')} {u[k] * 100:3.0f}%" for rn, u in us.items())
        return "U now:  " + now + "\n(dotted: from the time average)"

    def _fast_frame(self, c, k):
        """Only exchange the data instead of rebuilding all four axes.

        A complete rebuild costs about 50 ms, so at most 20 images per
        second - too few for a live display. This way only the calls that
        really change per frame remain. Returns False if something is
        missing; then the caller draws completely."""
        if not self.cb_fastdraw.isChecked():
            return False
        art = self._art
        if not art or art.get("panel") != self.cmb_panel.currentIndex():
            return False
        # The fast path only exchanges data. If the shape no longer matches
        # the existing objects (grid resolution, time sampling or crosshair
        # changed), everything must be redrawn - otherwise image and axes
        # would no longer belong together.
        if (art.get("shape") != c["cube"].shape
                or art.get("rowcol") != (c["row"], c["col"])):
            return False
        try:
            norm = c["norm"]
            t_us = c["t"] * 1e6
            cut_row, cut_col = c["row"], c["col"]
            I_now = c["cube"][k] / norm
            art["main_im"].set_data(I_now)
            if self.cmb_scale.currentIndex() == 3:          # per frame
                art["main_im"].set_clim(0.0, max(float(I_now.max()), 1e-12))
            T0_us = c["T0"] * 1e6
            per = (f"   ({t_us[k] / T0_us:.3f} $T_0$)"
                   if np.isfinite(T0_us) and T0_us > 0 else "")
            art["main_title"].set_text(
                f"I(x, y, t) instantaneous   |   t = {t_us[k]:8.4f} us" + per +
                f"\nPeak over time: {c['cube_max'] / norm:.2f} x the "
                f"maximum of the time average")
            art["cut_now"].set_ydata(c["cube"][k, cut_row, :] / norm)
            tr = c["trace"] / norm
            art["time_cur"].set_xdata([t_us[k], t_us[k]])
            art["time_dot"].set_data([t_us[k]], [tr[k]])
            if "st_cur" in art:
                art["st_cur"].set_ydata([t_us[k], t_us[k]])
            if "u_cur" in art:
                art["u_cur"].set_xdata([t_us[k], t_us[k]])
                for ln, u in art.get("u_live", []):
                    ln.set_data(t_us[:k + 1], u[:k + 1] * 100)
                for dt, u in art.get("u_dot", []):
                    dt.set_data([t_us[k]], [u[k] * 100])
                if "u_title" in art:
                    art["u_title"].set_text(self._u_title(c, k))
        except Exception:
            return False
        self.canvas.draw_idle()
        return True

    def draw_frame(self, full=False):
        c, s = self.cache, self.state
        if not c:
            return
        k = self.frame_idx
        if not full and self._fast_frame(c, k):
            return
        self._art = {}
        x_um, y_um = c["x"] * 1e6, c["y"] * 1e6
        extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
        norm = c["norm"]
        t_us = c["t"] * 1e6
        cut_row, cut_col = c["row"], c["col"]   # do NOT call them row/col:
        # "col" has already been reused twice as a colour variable in the
        # panel loops and overwrote the column index.
        I_now = c["cube"][k] / norm

        mode = self.cmb_scale.currentIndex()
        if mode == 0:
            vmax = c["cube_p995"] / norm
        elif mode == 1:
            vmax = c["cube_max"] / norm
        elif mode == 2:
            vmax = float(c["I_avg"].max()) / norm
        else:
            vmax = float(I_now.max())
        vmax = max(vmax, 1e-12)

        # ---- 2D map ----
        self.ax_main.clear()
        self._art["main_im"] = self.ax_main.imshow(
            I_now, extent=extent, origin="lower", cmap="inferno",
            vmin=0.0, vmax=vmax, aspect="equal")
        self.ax_main.plot(c["centers_x"] * 1e6, c["centers_y"] * 1e6, "w+",
                          markersize=5, markeredgewidth=0.8, alpha=0.55,
                          label="spot centres")
        for grp in c.get("degen", []):
            self.ax_main.plot(c["centers_x"][grp] * 1e6, c["centers_y"][grp] * 1e6,
                              "o", mfc="none", mec="deepskyblue", ms=11, mew=1.6,
                              label="frequency degenerate")
        self.ax_main.axhline(y_um[cut_row], color="cyan", lw=0.8, alpha=0.8)
        self.ax_main.axvline(x_um[cut_col], color="magenta", lw=0.8, alpha=0.8)
        self.ax_main.set_xlabel("x (um)")
        self.ax_main.set_ylabel("y (um)")
        T0_us = c["T0"] * 1e6
        per_lbl = (f"   ({t_us[k] / T0_us:.3f} $T_0$)" if np.isfinite(T0_us) and T0_us > 0
                   else "")
        self._art["main_title"] = self.ax_main.set_title(
            f"I(x, y, t) instantaneous   |   t = {t_us[k]:8.4f} us" + per_lbl +
            f"\nPeak over time: {c['cube_max'] / norm:.2f} x the maximum "
            f"of the time average", fontsize=9)
        h, l = self.ax_main.get_legend_handles_labels()
        uniq = dict(zip(l, h))
        self.ax_main.legend(uniq.values(), uniq.keys(), loc="upper right",
                            fontsize=7, framealpha=0.4)

        # ---- panel top right: space-time map, n_eff or modulation depth ----
        self.ax_st.clear()
        panel = self.cmb_panel.currentIndex()
        # Colorbar and aspect ratio belong only to the map panels (1..3). If
        # they remain when switching, they squeeze the space-time map or the
        # bar chart.
        if panel in (0, 4, 5, 6, 7, 8):
            if self._panel_cbar is not None:
                try:
                    self._panel_cbar.remove()
                except Exception:
                    pass
                self._panel_cbar = None
            self.ax_st.set_aspect("auto")
        if panel == 0:
            t_hi = t_us[-1] if t_us[-1] > t_us[0] else t_us[0] + 1.0
            self.ax_st.imshow(c["st_map"] / norm, origin="lower", aspect="auto",
                              cmap="inferno", vmin=0.0, vmax=vmax,
                              extent=[x_um[0], x_um[-1], t_us[0], t_hi])
            self._art["st_cur"] = self.ax_st.axhline(t_us[k], color="w", lw=1.0)
            self.ax_st.set_xlabel("x (um)", fontsize=8)
            self.ax_st.set_ylabel("t (us)", fontsize=8)
            self.ax_st.set_title(f"Space-time map I(x, t) at y = {y_um[cut_row]:.3f} um",
                                 fontsize=9)
        elif panel == 8:
            self._draw_rabi_panel(c)
        elif panel == 7:
            pz = c.get("pulse")
            if pz is not None and "area_norm" in pz and len(pz["t0_scan"]) > 1:
                tt = pz["t0_scan"] * 1e6
                self.ax_st.plot(tt, pz["area_norm"], color="#3b6ea5", lw=1.8,
                                label="pulse area / mean")
                self.ax_st.axhline(1.0, color="#888", ls="--", lw=0.9)
                i_f = pz["i_flat"]
                self.ax_st.plot([tt[i_f]], [pz["area_norm"][i_f]], "o",
                                color="#4c8b5b", ms=8, label="flat point (jitter-tolerant)")
                self.ax_st.axvline(pz["t0"] * 1e6, color="#b3402f", lw=1.4,
                                   label="current $t_0$")
                jt = pz["jitter"]
                jtxt = ("no first-order sensitivity" if not np.isfinite(jt)
                        else f"{jt * 1e9:.0f} ns per 1 % area error")
                self.ax_st.set_xlabel("pulse start $t_0$ (us)", fontsize=8)
                self.ax_st.set_ylabel("pulse area / mean", fontsize=8)
                self.ax_st.set_title(
                    f"Pulse area over the start time   |   at the current $t_0$: {jtxt}",
                    fontsize=8.5)
                self.ax_st.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "no pulse data available",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        elif panel == 6:
            pz = c.get("pulse")
            if pz is not None and len(pz["t0_scan"]) > 1:
                tt = pz["t0_scan"] * 1e6
                self.ax_st.plot(tt, pz["u"] * 100, color="#3b6ea5", lw=1.6,
                                label="U of the pulse area")
                self.ax_st.axhline(pz["u_ref"] * 100, color="#4c8b5b", ls=":", lw=1.2,
                                   label="U from the time average")
                i_best = int(np.argmin(pz["u"]))
                self.ax_st.plot([tt[i_best]], [pz["u"][i_best] * 100], "o",
                                color="#4c8b5b", ms=6)
                self.ax_st.annotate(f"best t_0 = {tt[i_best]:.2f} us\n"
                                    f"U = {pz['u'][i_best] * 100:.1f} %",
                                    (tt[i_best], pz["u"][i_best] * 100),
                                    xytext=(6, 8), textcoords="offset points",
                                    fontsize=7, color="#2f6b45")
                self.ax_st.axvline(pz["t0"] * 1e6, color="#b3402f", lw=1.4)
                self.ax_st.set_xlabel("pulse start t_0 (us)", fontsize=8)
                self.ax_st.set_ylabel("U of the pulse area (%)", fontsize=8)
                self.ax_st.set_title(
                    f"pi pulse {pz['t_p'] * 1e6:.2f} us   |   at t_0 = "
                    f"{pz['t0'] * 1e6:.2f} us: U = {pz['u_now'] * 100:.1f} %",
                    fontsize=8.5)
                self.ax_st.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
                self.ax_st.set_ylim(bottom=0)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "no pulse data available",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        elif panel == 5:
            us = c.get("u_series", {})
            if us:
                # fixed colour order, never rotated through
                cols = ["#3b6ea5", "#b3402f", "#4c8b5b"]
                live = self.cb_live.isChecked()
                self._art["u_live"] = []
                self._art["u_dot"] = []
                for i, (rname, u) in enumerate(us.items()):
                    hue = cols[i % len(cols)]
                    if live:
                        # whole trace pale for orientation, on top of it the
                        # running trace up to the current instant
                        self.ax_st.plot(t_us, u * 100, color=hue, lw=1.0, alpha=0.22)
                        (ln,) = self.ax_st.plot(t_us[:k + 1], u[:k + 1] * 100,
                                                color=hue, lw=1.8, label=rname)
                        (dt,) = self.ax_st.plot([t_us[k]], [u[k] * 100], "o",
                                                color=hue, ms=5)
                        self._art["u_live"].append((ln, u))
                        self._art["u_dot"].append((dt, u))
                    else:
                        self.ax_st.plot(t_us, u * 100, color=hue, lw=1.5, label=rname)
                    ref = c["u_ref"].get(rname)
                    if ref is not None and np.isfinite(ref):
                        self.ax_st.axhline(ref * 100, color=hue, ls=":", lw=1.1)
                        self.ax_st.annotate(f"U($\\langle I\\rangle$) = {ref * 100:.1f} %",
                                            (t_us[-1], ref * 100), xytext=(-2, 3),
                                            textcoords="offset points", ha="right",
                                            fontsize=6.5, color=hue)
                pz_ = c.get("pulse")
                if pz_ is not None:
                    self.ax_st.axvspan(pz_["t0"] * 1e6,
                                       min((pz_["t0"] + pz_["t_p"]) * 1e6, t_us[-1]),
                                       color="#3b6ea5", alpha=0.12, lw=0)
                self._art["u_cur"] = self.ax_st.axvline(t_us[k], color="k", lw=0.9)
                self.ax_st.set_xlabel("t (us)", fontsize=8)
                self.ax_st.set_ylabel("U = std/mean (%)", fontsize=8)
                self._art["u_title"] = self.ax_st.set_title(
                    self._u_title(c, k), fontsize=8.5)
                self.ax_st.legend(fontsize=6.5, loc="upper right", framealpha=0.85)
                self.ax_st.set_ylim(bottom=0)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "no uniformity available",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        elif panel == 4:
            spec = c.get("spectrum", {})
            if spec:
                ds = np.array(sorted(spec))
                freqs = ds * c["f0"] * 1e-3
                vals = np.array([spec[int(d)] * 100 for d in ds])
                width_bar = 0.7 * (freqs[1] - freqs[0]) if len(freqs) > 1 else 10.0
                self.ax_st.bar(freqs, vals, width=width_bar, color="#3b6ea5",
                               edgecolor="none")
                nu = self.sp_nu_r.value()
                # CAUTION: do not use a variable "col" here - further up that
                # is the column index of the crosshair.
                for f_mark, lab in ((nu, "$\\nu_r$"), (2 * nu, "$2\\nu_r$")):
                    if freqs[0] - width_bar <= f_mark <= freqs[-1] + width_bar:
                        self.ax_st.axvline(f_mark, color="#b3402f", lw=1.4, ls="--")
                        self.ax_st.annotate(lab, (f_mark, self.ax_st.get_ylim()[1]),
                                            xytext=(3, -2), textcoords="offset points",
                                            color="#b3402f", fontsize=9, ha="left", va="top")
                self.ax_st.set_ylim(0, max(vals.max() * 1.28, 1e-3))
                self.ax_st.set_xlabel("beat frequency (kHz)", fontsize=8)
                self.ax_st.set_ylabel("$\\sigma_d / \\langle I\\rangle$ (%)", fontsize=8)
                self.ax_st.set_title(
                    f"Spectrum of the beating in the plateau   "
                    f"(total {c['sigma_rms'] * 100:.0f} %)", fontsize=9)
                self.ax_st.spines["top"].set_visible(False)
                self.ax_st.spines["right"].set_visible(False)
            else:
                self.ax_st.text(0.5, 0.5, "no spectrum available",
                                ha="center", va="center", transform=self.ax_st.transAxes)
        else:
            if panel == 1:
                dat, cmap = c["n_eff"], "viridis"
                title = (f"Enhancement $n_{{eff}} = I_{{max}}/\\langle I\\rangle$   "
                         f"(plateau: {c['n_eff_med']:.1f})")
                lim = (1.0, max(2.0, float(len(c["f_spots"]))))
            elif panel == 2:
                dat, cmap = c["depth"] * 100.0, "magma"
                title = (f"Modulation depth in %   "
                         f"(plateau: {c['depth_med'] * 100:.0f} %)")
                lim = (0.0, 100.0)
            else:
                dat, cmap = c["sigma_rel"] * 100.0, "cividis"
                title = (f"temporal variation $\\sigma_t/\\langle I\\rangle$ in %   "
                         f"(plateau: {c['sigma_rms'] * 100:.0f} %)")
                lim = (0.0, float(np.nanpercentile(c["sigma_rel"], 99) * 100))
            im = self.ax_st.imshow(dat, extent=extent, origin="lower", cmap=cmap,
                                   aspect="equal", vmin=lim[0], vmax=lim[1])
            self.ax_st.contour(x_um, y_um, c["I_avg"] / norm,
                               levels=[0.5 * c["I_avg"].max() / norm],
                               colors="w", linewidths=0.9, linestyles="--")
            self.ax_st.set_xlabel("x (um)", fontsize=8)
            self.ax_st.set_ylabel("y (um)", fontsize=8)
            self.ax_st.set_title(title, fontsize=8.5)
            if self._panel_cbar is not None:
                try:
                    self._panel_cbar.remove()
                except Exception:
                    pass
            self._panel_cbar = self.fig.colorbar(im, ax=self.ax_st, fraction=0.046, pad=0.03)
            self._panel_cbar.ax.tick_params(labelsize=7)
        self.ax_st.tick_params(labelsize=7)

        # ---- cuts with envelope ----
        self.ax_cut.clear()
        self.ax_cut.fill_between(x_um, c["cut_x_min"] / norm, c["cut_x_max"] / norm,
                                 color="tab:orange", alpha=0.22, lw=0,
                                 label="min/max over t")
        self.ax_cut.plot(x_um, c["I_avg"][cut_row, :] / norm, "k--", lw=1.0,
                         label="time average")
        (self._art["cut_now"],) = self.ax_cut.plot(
            x_um, c["cube"][k, cut_row, :] / norm, color="tab:orange", lw=1.3,
            label="instantaneous")
        self.ax_cut.axvline(x_um[cut_col], color="magenta", lw=0.8, alpha=0.7)
        self.ax_cut.set_xlabel("x (um)", fontsize=8)
        self.ax_cut.set_ylabel("I / I_max", fontsize=8)
        self.ax_cut.set_title("x cut", fontsize=9)
        self.ax_cut.tick_params(labelsize=7)
        self.ax_cut.legend(fontsize=6.5, loc="upper right", framealpha=0.5)

        # ---- I(t) at the crosshair ----
        self.ax_time.clear()
        tr = c["trace"] / norm
        self.ax_time.plot(t_us, tr, color="tab:blue", lw=1.0)
        pz = c.get("pulse")
        if pz is not None:
            self.ax_time.axvspan(pz["t0"] * 1e6,
                                 min((pz["t0"] + pz["t_p"]) * 1e6, t_us[-1]),
                                 color="#3b6ea5", alpha=0.15, lw=0)
        self._art["time_cur"] = self.ax_time.axvline(t_us[k], color="k", lw=0.9)
        (self._art["time_dot"],) = self.ax_time.plot([t_us[k]], [tr[k]], "o",
                                                     color="tab:red", ms=4)
        mean_v = float(tr.mean())
        self.ax_time.axhline(mean_v, color="k", ls="--", lw=0.8)
        if np.isfinite(T0_us) and T0_us > 0:
            for p in range(1, s["n_periods"]):
                self.ax_time.axvline(p * T0_us, color="gray", ls=":", lw=0.6)
        lo, hi = float(tr.min()), float(tr.max())
        depth = (hi - lo) / (hi + lo) if (hi + lo) > 0 else 0.0
        self.ax_time.set_xlabel("t (us)", fontsize=8)
        self.ax_time.set_ylabel("I / I_max", fontsize=8)
        extra = (f"\npi pulse {pz['t_p'] * 1e6:.2f} us from {pz['t0'] * 1e6:.2f} us: "
                 f"U(area) {pz['u_now'] * 100:.0f} %" if pz else "")
        self.ax_time.set_title(
            f"I(t) at the crosshair - modulation {depth * 100:.0f} %" + extra, fontsize=8.5)
        self.ax_time.tick_params(labelsize=7)

        prof = "Airy" if s["use_airy"] else "Gauss"
        self.fig.suptitle(
            f"N = {s['N_x']}x{s['N_y']} tones, {prof}, waist = {s['win'] * 1e6:.3f} um, "
            f"width = {s['width_x'] * 1e-6:.4f}/{s['width_y'] * 1e-6:.4f} MHz (x/y), "
            f"r_x = {s['r_x']:g}, r_y = {s['r_y']:g}, "
            f"lambda = {s['lambda_opt'] * 1e9:.1f} nm, f0 = {s['offset'] * 1e-6:.3f} MHz\n"
            + (f"Fundamental period T_0 = {T0_us:.3f} us  (f_0 = {c['f0'] * 1e-3:.3f} kHz), "
               f"window = {s['n_periods']} periods" if c["periodic"]
               else f"not periodic - scale T_ref = {T0_us:.3f} us, "
                    f"window = {s['n_periods']} x T_ref"),
            fontsize=10)
        self._art["panel"] = self.cmb_panel.currentIndex()
        self._art["shape"] = c["cube"].shape
        self._art["rowcol"] = (c["row"], c["col"])
        self.canvas.draw_idle()

    # --------------------------------------------------------
    def _on_save_clicked(self):
        self.draw_frame(full=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        s = self.state
        # Vector PDF instead of a raster image: the outputs end up in LaTeX
        # documents, where a PNG is needlessly blurry after downscaling.
        name = (f"Beating_N{s['N_x']}x{s['N_y']}_"
                f"{'Airy' if s['use_airy'] else 'Gauss'}_"
                f"w{s['win'] * 1e6:.3f}um_width{s['width_x'] * 1e-6:.4f}MHz_"
                f"frame{self.frame_idx:04d}_{stamp}.pdf")
        path = self.out_dir / name
        try:
            self.fig.savefig(path, format="pdf", bbox_inches="tight")
            self.lbl_status.setText(f"saved: {short_name(path)}")
            self.lbl_status.setToolTip(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "Saving failed", str(exc))


def main():
    app = QApplication(sys.argv)
    win = BeatingMultitoneWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
