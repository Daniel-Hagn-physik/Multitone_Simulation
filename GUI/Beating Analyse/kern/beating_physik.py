"""beating_physik.py - physics and numerics of the Beating Multitone GUI.

Everything the Beating GUI computes lives in this file: frequencies, geometry,
fields, beat orders, exact time statistics, pulse area, atom weighting. No Qt,
no matplotlib - only numpy and scipy, so it can be imported, tested and read
on its own.

Used by
    Beating_Multitone_GUI.py      the main window (orchestration and drawing)
    pulse_timing.py, power_budget.py, camera_series.py, one_lens_design.py
    kern/beating_profil.py        bridge to the Rabi calculation

Units are SI throughout (m, s, Hz, rad). The field is normalised to 1 at the
centre of a spot; intensities are therefore in arbitrary units until a sub-
window (power_budget, pulse_timing) fixes the scale.

Contents
--------
     0. Constants                     fLO, theta_max, f_band, AIRY_FACTOR
     1. AOD frequencies and geometry  multitone_frequencies ... compute_centers_and_freqs
     2. Amplitudes                    amps_from_ratio, amp_spots_from_ratios
     3. Field profile and grid        spot_field, build_field_stack, compute_grid
     4. Beat frequencies              unique_beat_frequencies, fundamental_beat_frequency,
                                      degenerate_groups, min_frames_per_period,
                                      resonance_check
     5. Tone phases                   schroeder_phases, kitayoshi_phases,
                                      spot_phases_from_tones, quadrature_penalty,
                                      crest_factor
     6. Time series on a time grid    intensity_cube, boxcar_in_time
     7. Exact time statistics         beat_orders, pair_lists, order_amplitudes,
                                      time_stats_exact, camera_frames_exact
     8. Figures of merit              uniformity_of, VariationObjective,
                                      UniformitySeries
     9. Power of the profile          single_spot_power, spot_overlap_integral,
                                      profile_total_power, spot_power_shares,
                                      tone_power_shares, rf_voltage_ratios
    10. The atom as a weight          sigma_thermal, atom_local_stack
    11. Pulse area                    beat_coeffs_mean, pulse_area_curve,
                                      pulse_area_map, sqrt_*,
                                      sqrt_law_excitation, PulseArea


Coherent superposition
----------------------
The incoherent GUIs (Multitone_Lens_GUI.py, Weighted_Multitone_Lens_GUI.py)
sum the INTENSITIES of the individual tones:

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
exactly why the incoherent picture is correct as a time average, but it says
nothing about the instantaneous values. The beating becomes visible wherever
two spots overlap spatially (u_s * u_s' != 0), that is, between the traps.
The average only equals the incoherent sum if EVERY difference frequency is
non-zero - frequency degenerate spots break it (degenerate_groups()).

Frequencies (section 1 and 4)
-----------------------------
As in the AWG, the tones sit at

    f_x(n) = offset + width_x * n/(N_x-1),   n = 0 .. N_x-1
    f_y(m) = offset + width_y * m/(N_y-1),   m = 0 .. N_y-1

Both AODs shift the light frequency, so a spot (n,m) carries
f_s = f_x(n) + f_y(m). Only the DIFFERENCE of two spot frequencies
appears in |E|^2. At equal width on both axes

    f_s - f_s' = width * ( dn/(N_x-1) + dm/(N_y-1) )

and all of these differences are integer multiples of

    f_0 = width / lcm(N_x-1, N_y-1)

so width/6 for 3x4. At the start value width = 0.37 MHz that is 61.67 kHz,
fundamental period T_0 = 16.22 us. The absolute light frequency and the
offset cancel completely - wavelength and offset change the geometry (waist
conversion and beam deflection, respectively), but not a single beat
frequency.

Geometry (section 1)
--------------------
    theta(f) = theta_max * (f - offset) / f_band           the AOD, always
    telescope:    r = (f1*fLO/f2) * tan(theta)     w_0 = (f1/f2) * lam*fLO/(pi*w_in)
    single lens:  r = f_lens * tan(theta)          w_0 = lam*f_lens/(pi*w_in)

Field amplitudes (section 2)
----------------------------
The other scripts carry `amps` as INTENSITY weights (I += a * profile).
The field therefore carries sqrt(a). The amplitudes come from the familiar
outer/inner parametrisation (amps_from_ratio):

    amp_x = [r_x, 1, ..., 1, r_x],   amp_y = [r_y, 1, ..., 1, r_y]
    a_spot(n,m) = amp_x[n] * amp_y[m]

What the weights mean at the AOD. In the linear regime the diffraction
efficiency of a tone is proportional to its RF POWER, so the diffracted
optical field follows the RF VOLTAGE (and takes over the RF phase). A spot is
diffracted twice:

    E_spot(n,m) ~ V_x(n) * V_y(m)          I_spot ~ P_RF,x(n) * P_RF,y(m)

amp_x is therefore an RF POWER ratio, sqrt(amp_x) an RF VOLTAGE ratio. It
follows that the power of a spot goes as a_spot (not a_spot^2), the optical
power fed by RF tone n of the x axis as amp_x[n], and the RF signal of the
crest factor carries sqrt(amp_x). An AWG that takes voltage amplitudes needs
sqrt(r): r = 1.16 is 1.077 in voltage, +0.64 dB.
Valid while the total diffraction efficiency is small - the saturation of
sin^2 costs roughly eta_diff/3 - and without intermodulation between tones.

Field profile of a single spot (section 3, normalised to 1 at the centre)
-------------------------------------------------------------------------
    Gauss:  u(r) = exp(-r^2 / w^2)          -> |u|^2 = exp(-2r^2/w^2)
    Airy:   u(r) = 2*J1(k r)/(k r)          -> |u|^2 = (2*J1/u)^2
            k = 3.83170597 / (AIRY_FACTOR * waist),  AIRY_FACTOR = 1.4830 fixed
For the Airy profile u is NEGATIVE in the rings - this sign matters for
the coherent sum and is deliberately kept here.

Excitation model (section 11), and what is NOT in it
----------------------------------------------------
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
extra spatial non-uniformity - the uniformity numbers are unaffected by it.
Compensating the mean shift by detuning the Raman recovers most of the loss
(at eta = 0.5: 0.77 -> 0.93 in the excitation). Such a fixed detuning is NOT
proportional to Omega any more; with beating the residual shift varies in
time and the closed form no longer applies.

For Omega ~ sqrt(I) the shift of the multitone leg still follows I while the
Rabi frequency follows sqrt(I), so delta/Omega is NOT constant and the closed
form does not apply at all. There eta means the part of delta/Omega that the
MULTITONE leg causes, at the calibration intensity, and the two-level problem
is propagated numerically (sqrt_law_excitation). Each leg contributes to the
differential shift in proportion to its power, so for equal legs this is
half of the total eta of kern/rb85_raman.py; the other half comes from the
clean leg, is constant and is assumed to be compensated.

Order of magnitude: D1 dominated, sigma+/sigma+, clock states, Delta >> the
ground hyperfine splitting: eta ~ omega_HF / (2 Delta sqrt(beta(1-beta))),
i.e. omega_HF/Delta ~ 0.06 at +50 GHz and beta = 0.5. The exact value comes
from kern/rb85_raman.py.

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
multitone path is exactly what the coupling law above encodes. The atomic
coefficients themselves (C_rabi, C_shift, C_scatter) come from kern/rb85_raman.py.
"""

import math

import numpy as np
from scipy.special import j1


# ============================================================================
# 0. Constants
# ============================================================================
# ---- AOD and telescope, identical to the other GUIs ----
fLO = 52.88e-3          # m   focusing lens behind the telescope
theta_max = 43e-3       # rad maximum deflection angle of the AOD
f_band = 36e6           # Hz  AOD bandwidth
pitch = 5.288e-6        # m   physical atom spacing (for information only)

# ---- beam profile ----
# Airy profile: first_zero_radius = AIRY_FACTOR * waist. 1.4830 gives the
# Airy spot the same 1/e^2 width as a Gauss of the same waist. FIXED - it is
# no longer read from airy_scale.py (that file lives in "Flattop GUI", the
# import failed silently and only the fallback value was ever used).
AIRY_FACTOR = 1.4830

# np.trapezoid only exists from numpy 2.0 on, before that it is np.trapz.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# ============================================================================
# 1. AOD frequencies and geometry
# ============================================================================
# Tone frequency -> deflection angle -> position in the focus, and the
# waist conversion. Only radius_from_angle() and conjugate_waist() know
# whether the telescope build or the single lens is used.


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


def radius_from_angle(theta, f1, f2, fLO_, one_lens=False, f_single=None):
    """Position in the focal plane for a deflection angle theta.

    Two builds are supported and they differ ONLY here and in
    conjugate_waist():

      telescope build (default)   r = (f1 * fLO / f2) * tan(theta)
      single lens                 r = f_lens * tan(theta)

    The angle itself comes from the AOD in both cases and is untouched."""
    if one_lens:
        return float(f_single) * np.tan(theta)
    return (f1 * fLO_ / f2) * np.tan(theta)


def conjugate_waist(w, f1, f2, lam, one_lens=False, f_single=None,
                    fLO_=None):
    """Input <-> output waist.

    Telescope build: w_out = (f1/f2) * (lam * fLO) / (pi * w_in)
    Single lens:     w_out = lam * f_lens / (pi * w_in)

    Both relations are symmetric, the same function computes both
    directions.

    fLO_ = None uses the module constant fLO."""
    if w <= 0:
        return float("nan")
    if one_lens:
        return float(f_single) * lam / np.pi / w
    fLO_ = fLO if fLO_ is None else fLO_
    return (f1 / f2) * (lam * fLO_) / np.pi / w


def compute_centers_and_freqs(N_x, N_y, width_x, width_y, f1, f2, offset,
                              one_lens=False, f_single=None,
                              fLO_=None, theta_max_=None, f_band_=None):
    """Spot centres AND the frequency of every spot.

    The spot ordering is identical to compute_centers() of the other
    GUIs (fx outer, fy inner), so that amp_spots = repeat(amp_x, N_y) *
    tile(amp_y, N_x) still fits unchanged.

    Here x and y get their OWN width. The other GUIs set both equal;
    that is precisely what produces frequency degeneracies (see
    degenerate_groups()), and separate widths are the only way to lift
    them - a constant frequency offset on one axis does NOT help, because
    it shifts all spot frequencies by the same amount and therefore leaves
    every difference unchanged.

    fLO_, theta_max_, f_band_ = None use the module constants. They are
    arguments so that kern/beating_profil.py can vary them without
    overwriting the module constants."""
    fLO_ = fLO if fLO_ is None else fLO_
    theta_max_ = theta_max if theta_max_ is None else theta_max_
    f_band_ = f_band if f_band_ is None else f_band_
    fx_freq = multitone_frequencies(N_x, offset, width_x)
    fy_freq = multitone_frequencies(N_y, offset, width_y)
    f_center_x = offset + width_x / 2.0
    f_center_y = offset + width_y / 2.0
    r_center_x = radius_from_angle(
        angle_from_frequency(f_center_x, offset, theta_max_, f_band_), f1, f2, fLO_,
        one_lens, f_single)
    r_center_y = radius_from_angle(
        angle_from_frequency(f_center_y, offset, theta_max_, f_band_), f1, f2, fLO_,
        one_lens, f_single)

    centers_x, centers_y, f_spots = [], [], []
    for fx in fx_freq:
        rx = radius_from_angle(angle_from_frequency(fx, offset, theta_max_, f_band_),
                               f1, f2, fLO_, one_lens, f_single)
        for fy in fy_freq:
            ry = radius_from_angle(angle_from_frequency(fy, offset, theta_max_, f_band_),
                                   f1, f2, fLO_, one_lens, f_single)
            centers_x.append(rx)
            centers_y.append(ry)
            # Both AODs shift the light frequency -> the spot carries the sum.
            f_spots.append(fx + fy)
    return (np.array(centers_x), np.array(centers_y), np.array(f_spots),
            r_center_x, r_center_y, fx_freq, fy_freq)


# ============================================================================
# 2. Amplitudes
# ============================================================================


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


# ============================================================================
# 3. Field profile and computation grid
# ============================================================================


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


def compute_grid(centers_x, centers_y, win_eff, resolution, pad_factor=2.5):
    """Square grid covering all spots plus a margin of pad_factor * win_eff
    (for Airy the radius of the first zero, for Gauss the waist).

    The fields are evaluated analytically at every pixel, so the margin does
    not change any local value. It only truncates INTEGRALS over the grid:
    for Airy about 5 % of the power lies outside at pad_factor = 2.5. Powers
    therefore come from profile_total_power(), never from a grid sum.

    No pitch copies of neighbouring sites are added. That is an ASSUMPTION
    about the experiment - one site at a time, or neighbouring patterns that
    are mutually incoherent. If several sites are driven simultaneously from
    the same laser through the same AODs, their tones sit pitch/(dr/df)
    apart (837 kHz at the start values) and interfere with this pattern,
    LINEARLY in the neighbour field: about 1 % intensity crosstalk becomes a
    beat of about 15 % rms. With crossed AODs the anti-diagonal copies even
    carry the same spot frequencies and add a static term. None of that is
    in this model."""
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


# ============================================================================
# 4. Beat frequencies, fundamental period, degeneracy
# ============================================================================
# Only DIFFERENCES of spot frequencies appear in |E|^2. Everything about
# the time structure follows from the set of these differences.


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


def resonance_check(beats, nu_r, tol_rel=0.06):
    """Does a beat line coincide with nu_r or 2*nu_r?

    The coherent beating produces a LINE spectrum at multiples of f_0. If
    no line lies near nu_r or 2*nu_r, the trap receives practically no
    power on its resonances despite 100 % modulation depth - that is the
    decisive point, not the size of the modulation.

    Only meaningful for illumination lasting MANY trap periods. A pulse of
    length T_p is spectrally about 1/T_p wide; a pi pulse of 0.5 us is
    shorter than one trap period at 60 kHz, the line picture does not apply
    and what matters is the kick of the single pulse (and, for a pulse
    train, the repetition rate).

    Returns: (distance_nu, distance_2nu, critical) with the distances of
    the nearest line in Hz."""
    if beats is None or len(beats) == 0 or nu_r <= 0:
        return float("nan"), float("nan"), False
    b = np.asarray(beats, dtype=float)
    d1 = float(np.min(np.abs(b - nu_r)))
    d2 = float(np.min(np.abs(b - 2 * nu_r)))
    crit = (d1 < tol_rel * nu_r) or (d2 < tol_rel * 2 * nu_r)
    return d1, d2, crit


# ============================================================================
# 5. Tone phases
# ============================================================================
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


def kitayoshi_phases(N):
    """Kitayoshi phases (H. Kitayoshi, S. Sumida, K. Shirakawa, S. Takeshita,
    "DSP synthesized signal source for analog testing stimulus and new test
    method", IEEE Int. Test Conf. 1985, pp. 825-834):

        phi_k = phi_0 - (2 pi / N) * sum_{j=1..k} j
              = pi/2 - pi * k(k+1) / N ,      k = 0 ... N-1

    Same quadratic family as Schroeder - a linear frequency chirp - only the
    index is shifted by one and there is a constant offset of pi/2. A constant
    offset is physically irrelevant (it is a global phase), so the two differ
    only in which tone gets which phase, and the crest factors come out within
    a few percent of each other. Kitayoshi is the version quoted in the
    mixed-signal test literature."""
    k = np.arange(N)
    return np.pi / 2.0 - np.pi * k * (k + 1) / max(N, 1)


def spot_phases_from_tones(phase_x, phase_y, N_x, N_y):
    """phi_spot(n,m) = phi_x(n) + phi_y(m), in the spot ordering of
    compute_centers_and_freqs() (fx outer, fy inner)."""
    return np.repeat(phase_x, N_y) + np.tile(phase_y, N_x)


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


def crest_factor(f_tones, phases, n_samples=20000, f_ref=None, amps=None):
    """Crest factor of the RF signal of one axis: peak amplitude / rms.
    Decisive for how strongly the AOD and its amplifier are driven for
    short times.

    Computed WITHOUT the carrier. With

        s(t) = Re[ e^{i 2 pi f_c t} A(t) ],
        A(t) = sum_n a_n e^{i(2 pi (f_n - f_c) t + phi_n)} ,

    the peak of s over the carrier is max|A| and its rms is rms|A|/sqrt(2),
    hence crest = sqrt(2) * max|A| / rms|A|. That removes the need to
    resolve 100 MHz in the sampling and makes the number exact.

    THE AVERAGING WINDOW IS ONE PERIOD OF THE ENVELOPE, 1/f_env with f_env
    the gcd of the tone spacings - NOT 1/span, which is what an earlier
    version used. At 13 tones spaced 130 kHz the span is twelve times the
    spacing; a window of 1/span sits entirely inside the rephasing peak,
    so the rms comes out far too large and the crest far too small: 2.19
    instead of the correct 5.10 = sqrt(2N) for all phases at zero. The
    error grows with the number of tones, which is exactly where the
    number matters."""
    f = np.asarray(f_tones, dtype=float)
    if f.size == 0:
        return float("nan")
    a = np.ones(f.size) if amps is None else np.asarray(amps, dtype=float)
    if f.size == 1:
        return float(np.sqrt(2.0))
    if f_ref is None or f_ref <= 0:
        f_ref = fundamental_beat_frequency(f)
        if f_ref <= 0:                     # incommensurable: take the closest pair
            d = np.diff(np.sort(f))
            d = d[d > 0]
            f_ref = float(d.min()) if d.size else 1.0
    t = np.linspace(0.0, 1.0 / f_ref, n_samples, endpoint=False)
    ph = (2 * np.pi * (f - f.mean())[:, None] * t[None, :]
          + np.asarray(phases, dtype=float)[:, None])
    A = (a[:, None] * np.exp(1j * ph)).sum(axis=0)
    mod = np.abs(A)
    rms = np.sqrt(np.mean(mod ** 2))
    return float(np.sqrt(2.0) * mod.max() / rms) if rms > 0 else float("nan")


# ============================================================================
# 6. Time series on a time grid
# ============================================================================
# The brute-force path: the intensity at every frame. Used for the
# animation, the envelopes and the space-time map. All NUMBERS that must
# be exact come from section 7 instead.


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


def boxcar_in_time(cube, n_win):
    """Cyclic running mean over n_win frames - the camera's exposure.

    The time window of the cube covers whole fundamental periods, so the
    average may wrap around; that is exactly what a camera does when the
    exposure straddles the end of a period."""
    n_win = int(n_win)
    if n_win <= 1 or n_win >= cube.shape[0]:
        return cube
    n_t = cube.shape[0]
    out = np.empty_like(cube)
    acc = np.zeros(cube.shape[1:], dtype=np.float64)
    for j in range(n_win):
        acc += cube[j % n_t]
    out[0] = (acc / n_win).astype(cube.dtype)
    for i in range(1, n_t):
        acc += cube[(i + n_win - 1) % n_t] - cube[(i - 1) % n_t]
        out[i] = (acc / n_win).astype(cube.dtype)
    return out


# ============================================================================
# 7. Exact time statistics without a time loop
# ============================================================================
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


def time_stats_exact_pairs(F, k, phases):
    """Reference implementation over all spot pairs - O(S^2) Python loop.

    Kept for verification; time_stats_exact() below gives the same numbers
    via an FFT over the beat orders and is the one that is used."""
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


def order_amplitudes(F, k, phases):
    """Complex amplitude per BEAT ORDER instead of per spot.

    Spots that share the same order u = (f_s - f_min)/f_0 can never beat
    against each other; only their sum matters. That sum

        H_u(r) = sum_{k_s = u} A_s u_s(r) e^{i phi_s}

    is all the time evolution depends on:  E(r,t) = sum_u H_u(r) z^u with
    z = exp(2 pi i f_0 t). The number of orders never exceeds the number of
    spots and the pair loop disappears - which is what makes grids of a
    hundred and more tones usable at all."""
    return _order_amplitudes_G(F.reshape(F.shape[0], -1), k, phases)


def _order_amplitudes_G(G, k, phases):
    """order_amplitudes() auf einer bereits flachen (S, P)-Matrix."""
    S = G.shape[0]
    k = np.asarray(k, dtype=int)
    kk = k - k.min() if k.size else k
    K = int(kk.max()) if kk.size else 0
    e = np.exp(1j * np.asarray(phases, dtype=float))
    H = np.zeros((K + 1, G.shape[1]), dtype=np.complex128)
    for sidx in range(S):
        H[kk[sidx]] += e[sidx] * G[sidx]
    return H, K


def time_stats_exact(F, k, phases, f0=0.0, t_exp=0.0, chunk=4000):
    """Time average and time variance of I(r,t), exact and without a pair
    loop.

    I(t) is a trigonometric polynomial of degree K in exp(2 pi i f_0 t), so
    n_t = 2K+1 equidistant samples reproduce it exactly - no aliasing, no
    approximation. The samples come from one FFT over the order axis.

    t_exp > 0 additionally applies a CAMERA EXPOSURE: a boxcar of length
    t_exp in time multiplies the Fourier coefficient of order d by
    sinc(d*f_0*t_exp). The mean is untouched (sinc(0) = 1), the variance
    becomes the variance of what a camera with that exposure actually
    records. The obtained Fourier coefficients are exact, so this costs one
    further FFT and nothing else.

    Returns (mean_map, var_map) in the shape of F[0]."""
    shape = F.shape[1:]
    H, K = order_amplitudes(F, k, phases)
    P = H.shape[1]
    if K == 0:
        mean = np.abs(H[0]) ** 2
        return mean.reshape(shape), np.zeros(P).reshape(shape)

    n_t = 2 * K + 1
    weight = None
    if t_exp > 0 and f0 > 0:
        d = np.fft.fftfreq(n_t, d=1.0 / n_t)          # 0,1,..,K,-K,..,-1
        weight = np.abs(np.sinc(d * f0 * t_exp))
        weight[0] = 1.0

    mean = np.empty(P)
    var = np.empty(P)
    for lo in range(0, P, chunk):
        hi = min(P, lo + chunk)
        A = np.zeros((n_t, hi - lo), dtype=np.complex128)
        A[:K + 1] = H[:, lo:hi]
        I = np.abs(np.fft.fft(A, axis=0)) ** 2         # I(r, t_j), exact
        mean[lo:hi] = I.mean(axis=0)
        if weight is None:
            var[lo:hi] = I.var(axis=0)
        else:
            D = np.fft.ifft(I, axis=0)                 # Fourier coeff. D_d
            D *= weight[:, None]
            var[lo:hi] = np.sum(np.abs(D[1:]) ** 2, axis=0)
    return mean.reshape(shape), var.reshape(shape)


def camera_frames_exact(F, k, phases, f0, t_exp, t0_list, chunk=4000):
    """Bilder einer Kamera mit Belichtung t_exp, Start bei jedem t_0.

    Exakt fuer BELIEBIGE t_0 und t_exp. I(t) ist ein trigonometrisches
    Polynom in exp(2 pi i f_0 t),

        I(r,t) = sum_d C_d(r) exp(2 pi i d f_0 t) ,

    und die Belichtung ist ein Boxcar darauf:

        I_cam(r,t_0) = sum_d C_d(r) sinc(d f_0 t_exp)
                                    exp(2 pi i d f_0 (t_0 + t_exp/2)) .

    Die C_d kommen aus einer FFT ueber die Beat-Ordnungen, nicht aus einer
    Schleife ueber Spotpaare - dieselbe Maschinerie wie in
    time_stats_exact(). Kein Zeitraster, also auch kein Aliasing und keine
    Rundung der Belichtung auf ganze Frames.

    Rueckgabe: Array (len(t0_list),) + F[0].shape, reell."""
    shape = F.shape[1:]
    t0_list = np.atleast_1d(np.asarray(t0_list, dtype=float))
    H, K = order_amplitudes(F, k, phases)
    P = H.shape[1]
    out = np.empty((t0_list.size, P))
    if K == 0 or f0 <= 0:
        out[:] = np.abs(H[0]) ** 2
        return out.reshape((t0_list.size,) + shape)

    n_t = 2 * K + 1
    d = np.fft.fftfreq(n_t, d=1.0 / n_t)                  # 0,1,..,K,-K,..,-1
    # Boxcar mal Zeitverschiebung, ein Faktor pro Ordnung und Bild
    W = (np.sinc(d * f0 * t_exp)[None, :]
         * np.exp(2j * np.pi * d[None, :] * f0
                  * (t0_list[:, None] + 0.5 * t_exp)))    # (n_frames, n_t)
    for lo in range(0, P, chunk):
        hi = min(P, lo + chunk)
        A = np.zeros((n_t, hi - lo), dtype=np.complex128)
        A[:K + 1] = H[:, lo:hi]
        # ifft*n_t: E_j = sum_u H_u exp(+2 pi i u j / n_t), also t_j = j/(n_t f_0)
        E = np.fft.ifft(A, axis=0) * n_t
        I = np.abs(E) ** 2
        C = np.fft.fft(I, axis=0) / n_t                   # I_j = sum_d C_d e^{+i..}
        out[:, lo:hi] = np.real(W @ C)
    return out.reshape((t0_list.size,) + shape)


# ============================================================================
# 8. Figures of merit: uniformity and temporal variation
# ============================================================================
# U = std/mean in a region is the project convention. VariationObjective
# measures the temporal fluctuation, UniformitySeries U at every instant.


def uniformity_of(I, mask):
    """U = std/mean in the region - the project convention."""
    v = I[mask]
    m = float(np.mean(v))
    return float(np.std(v) / m) if m > 0 else float("nan")


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


# ============================================================================
# 9. Power of the profile
# ============================================================================


def single_spot_power(waist, use_airy, airy_factor):
    """Integral von |u|^2 ueber die ganze Ebene fuer EINEN Spot mit
    Amplitude 1 im Zentrum, analytisch.

    Gauss  u = exp(-r^2/w^2):          int = pi*w^2/2
    Airy   u = 2 J_1(k r)/(k r):       int = 4*pi/k^2,  k = 3.8317/(factor*w)

    (die zweite folgt aus int_0^inf J_1(u)^2/u du = 1/2)

    Analytisch, weil das Rechengitter der GUI die Auslaeufer abschneidet -
    beim Airy-Profil fehlen dadurch mehrere Prozent der Leistung, und
    ausgerechnet die Leistung ist die Groesse, die man nicht um Prozente
    danebenhaben will."""
    if not use_airy:
        return np.pi * waist ** 2 / 2.0
    k = 3.83170597 / (airy_factor * waist)
    return 4.0 * np.pi / k ** 2


def spot_overlap_integral(d, waist, use_airy, airy_factor):
    """Integral of u(r - c1) * u(r - c2) over the plane, |c1 - c2| = d, for
    two spots with amplitude 1 at their centres. Analytic.

    Gauss  u = exp(-r^2/w^2):     (pi w^2 / 2) * exp(-d^2 / (2 w^2))
    Airy   u = 2 J_1(k r)/(k r):  (4 pi / k^2) * 2 J_1(k d)/(k d)

    The Airy field is the Fourier transform of a disc; the autocorrelation
    of a disc transforms back into the same disc, so the overlap is the
    single-spot power times u(d) itself (checked against an FFT to 0.4 %).
    For d = 0 both reduce to single_spot_power()."""
    d = np.asarray(d, dtype=float)
    if not use_airy:
        return np.pi * waist ** 2 / 2.0 * np.exp(-d ** 2 / (2.0 * waist ** 2))
    k = 3.83170597 / (airy_factor * waist)
    x = k * d
    out = np.ones_like(x)
    m = x > 1e-12
    out[m] = 2.0 * j1(x[m]) / x[m]
    return 4.0 * np.pi / k ** 2 * out


def profile_total_power(amp_spots, waist, use_airy, airy_factor,
                        centers_x=None, centers_y=None, phases=None,
                        groups=None):
    """Total power of the TIME AVERAGED profile in simulation units,
    integral of <I> over the whole plane, analytic.

    The field of spot s carries sqrt(a_s) (build_field_stack), so spot s
    alone carries a_s * single_spot_power - the sum runs over a, NOT a^2.

    The cross terms of non-degenerate spots run and average away. Those of
    FREQUENCY DEGENERATE spots do not: they add

        2 * sqrt(a_s a_s') * cos(phi_s - phi_s') * spot_overlap_integral(d)

    For 3x4 at the start values that is +1.1 % at all phases 0 (Airy, the
    corner pair 3.3 um apart), not negligible. Pass centers, phases and
    groups (degenerate_groups()) to include it; without them only the
    incoherent sum is returned."""
    a = np.clip(np.asarray(amp_spots, dtype=float), 0.0, None)
    P = float(np.sum(a)) * single_spot_power(waist, use_airy, airy_factor)
    if (groups and centers_x is not None and centers_y is not None
            and phases is not None):
        cx = np.asarray(centers_x, dtype=float)
        cy = np.asarray(centers_y, dtype=float)
        ph = np.asarray(phases, dtype=float)
        for grp in groups:
            grp = list(grp)
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    s, q = grp[i], grp[j]
                    d = float(np.hypot(cx[s] - cx[q], cy[s] - cy[q]))
                    P += (2.0 * np.sqrt(a[s] * a[q]) * np.cos(ph[s] - ph[q])
                          * float(spot_overlap_integral(d, waist, use_airy,
                                                        airy_factor)))
    return P


def spot_power_shares(amp_spots):
    """Share of the diffracted power that goes into each spot: a_s / sum a.

    a_s is an intensity weight, so the power goes as a_s (not a_s^2). The
    static interference of degenerate spots is not attributed to single
    spots; it is in profile_total_power()."""
    a = np.clip(np.asarray(amp_spots, dtype=float), 0.0, None)
    tot = float(a.sum())
    return a / tot if tot > 0 else np.full(a.size, np.nan)


def tone_power_shares(r, N):
    """Share of the optical power fed by each RF tone of one axis.

    Tone n feeds all spots of its column: sum_m amp_x[n] amp_y[m] ~ amp_x[n].
    In the linear AOD regime this is also its share of the RF power of the
    channel."""
    amp = amps_from_ratio(r, N)
    return amp / float(amp.sum())


def rf_voltage_ratios(r, N):
    """RF voltage amplitudes of the tones of one axis, inner tones = 1.

    Diffraction efficiency ~ RF power ~ V^2, so the voltage is sqrt(amp).
    This is what an AWG with voltage amplitudes needs, and what enters the
    crest factor."""
    return np.sqrt(np.clip(amps_from_ratio(r, N), 0.0, None))


# ============================================================================
# 10. The atom as a weight
# ============================================================================
# Die harte Maske ("alle Pixel im Plateau") beantwortet eine Frage, die kein
# Atom stellt. Ein Atom sitzt an EINEM Ort und ist dort um sigma_thermal
# unscharf; was es sieht, ist die mit seiner Aufenthaltswahrscheinlichkeit
# gewichtete Intensitaet. Genau so rechnet Weighted_Multitone_Lens_GUI die
# gewichtete Uniformity, und dieselben zwei Funktionen werden hier benutzt.
#
# Der Unterschied ist gewaltig: die Streuung ueber das ganze Plateau ist eine
# Kamera-Groesse, die Streuung ueber ein Atom-Gewicht von gut 100 nm eine
# Atom-Groesse. Fuer das 13x14-Feld liegen zwischen beiden zwei
# Groessenordnungen.
RB85_MASS = 84.911789738 * 1.66053906660e-27      # kg
HBAR = 1.054571817e-34
KB = 1.380649e-23


def _masked_columns(F, mask):
    """(S, M)-Matrix der Spalten innerhalb der Maske."""
    G = F.reshape(F.shape[0], -1)
    if mask is None:
        return G
    idx = np.flatnonzero(np.asarray(mask).ravel())
    return G[:, idx] if idx.size else G


def _masked_weights(F, mask, weights):
    """Gewichtsvektor passend zu _masked_columns(); ohne Angabe alles 1."""
    n_all = int(np.prod(F.shape[1:]))
    w = (np.ones(n_all) if weights is None
         else np.asarray(weights, dtype=float).ravel())
    if mask is None:
        return w
    idx = np.flatnonzero(np.asarray(mask).ravel())
    return w[idx] if idx.size else w


def sigma_thermal(nu_trap, T, mass=RB85_MASS):
    """Thermische 1-sigma-Ortsbreite im harmonischen Fallenpotential,
    inklusive Nullpunktsbewegung:

        sigma^2 = hbar/(2 m omega) * coth( hbar omega / (2 kB T) )

    Identisch zu sigma_thermal() in Weighted_Multitone_Lens_GUI.py."""
    omega = 2.0 * np.pi * float(nu_trap)
    if omega <= 0 or T <= 0:
        return float("nan")
    x = HBAR * omega / (2.0 * KB * T)
    return float(np.sqrt(HBAR / (2.0 * mass * omega) / np.tanh(x)))


def atom_local_stack(centers_x, centers_y, amp_spots, waist, use_airy,
                     airy_factor, sigma, n_sigma=4.0, n_grid=81,
                     center=None):
    """Feines lokales Gitter um den Atomort plus Gewicht W(r).

    Das globale Rechengitter hilft hier nicht: bei 90 um Feld und 140 Punkten
    ist eine Zelle 0.64 um breit, das Atom aber nur 0.1 um. Also wird um den
    Atomort ein eigenes Gitter von +-n_sigma*sigma aufgespannt.

    Rueckgabe (xs, ys, Xs, Ys, F_local, W)."""
    cx0, cy0 = (float(np.mean(centers_x)), float(np.mean(centers_y))) \
        if center is None else (float(center[0]), float(center[1]))
    half = n_sigma * sigma
    xs = np.linspace(cx0 - half, cx0 + half, n_grid)
    ys = np.linspace(cy0 - half, cy0 + half, n_grid)
    Xs, Ys = np.meshgrid(xs, ys)
    F = build_field_stack(Xs, Ys, centers_x, centers_y, amp_spots, waist,
                          use_airy, airy_factor)
    W = np.exp(-((Xs - cx0) ** 2 + (Ys - cy0) ** 2) / (2.0 * sigma ** 2))
    return xs, ys, Xs, Ys, F, W


# ============================================================================
# 11. Pulse area and trigger jitter
# ============================================================================
# Die akkumulierte Rabi-Flaeche eines Rechteckpulses der Laenge T_p ab t_0 ist
#
#     theta(r, t_0) = int_{t_0}^{t_0+T_p} Omega(r,t) dt .
#
# Fuer den Zwei-Photonen-Uebergang ist Omega ~ I, und I ist eine endliche
# Fourierreihe in f_0. Dann ist die Pulsflaeche BIS AUF DEN FAKTOR T_p genau
# das, was auch eine Kamera mit Belichtung T_p aufnimmt:
#
#     theta(r,t_0) = T_p * sum_d C_d(r) sinc(d f_0 T_p) e^{2 pi i d f_0 (t_0+T_p/2)}
#
# Ein Puls ist also ein Boxcar wie eine Belichtung - nur ein tausendmal
# kuerzeres. Bei T_p = 1 us und f_0 = 10 kHz ueberleben die schnellen Ordnungen
# fast ungedaempft (sinc(0.12) = 0.976 bei 120 kHz); der Puls mittelt die
# Schwebung NICHT weg, er tastet sie ab. Deshalb haengt die Flaeche empfindlich
# vom Trigger ab, und deshalb lohnt der Scan ueber den Delay.
#
# Fuer Omega ~ sqrt(I) gibt es keine geschlossene Form; dort wird |E| auf einem
# feinen Zeitraster ausgewertet und das Integral als laufende Summe gebildet.


def beat_coeffs_mean(F, k, phases, mask=None, weights=None, chunk=4000):
    """Fourierkoeffizienten C_d der Intensitaet, raeumlich ueber die Maske
    gemittelt.

    Der raeumliche Mittelwert vertauscht mit allem Weiteren, deshalb reicht
    dieser eine Vektor, um die mittlere Pulsflaeche fuer BELIEBIG viele
    Trigger-Delays auszurechnen - ein Matrix-Vektor-Produkt statt einer
    Rechnung pro Delay.

    `weights` gewichtet das Ortsmittel - fuer die atomgewichtete Auswertung
    die Aufenthaltswahrscheinlichkeit W(r) des Atoms. Ohne Gewicht wird
    gleichmaessig ueber die Maske gemittelt.

    Rueckgabe (c, d): c[j] = <C_{d[j]}>, Ordnungen d in FFT-Reihenfolge."""
    G = _masked_columns(F, mask)
    wts = _masked_weights(F, mask, weights)
    H, K = _order_amplitudes_G(G, k, phases)
    M = G.shape[1]
    if K == 0:
        return (np.array([np.average(np.abs(H[0]) ** 2, weights=wts)], dtype=complex),
                np.zeros(1, int))
    n_t = 2 * K + 1
    d = np.rint(np.fft.fftfreq(n_t, d=1.0 / n_t)).astype(int)
    c = np.zeros(n_t, dtype=np.complex128)
    for lo in range(0, M, chunk):
        hi = min(M, lo + chunk)
        A = np.zeros((n_t, hi - lo), dtype=np.complex128)
        A[:K + 1] = H[:, lo:hi]
        E = np.fft.ifft(A, axis=0) * n_t
        I = np.abs(E) ** 2
        c += np.fft.fft(I * wts[None, lo:hi], axis=0).sum(axis=1) / n_t
    return c / max(float(np.sum(wts)), 1e-300), d


def pulse_area_curve(c, d, f0, t_p, t0_list):
    """Mittlere Pulsflaeche (in Einheiten von Intensitaet mal Zeit) fuer jeden
    Trigger-Zeitpunkt in t0_list. Exakt, aus den Koeffizienten von
    beat_coeffs_mean()."""
    t0_list = np.atleast_1d(np.asarray(t0_list, dtype=float))
    if f0 <= 0 or d.size == 1:
        return np.full(t0_list.size, float(np.real(c[0])) * t_p)
    W = (np.sinc(d * f0 * t_p)[None, :]
         * np.exp(2j * np.pi * d[None, :] * f0 * (t0_list[:, None] + 0.5 * t_p)))
    return t_p * np.real(W @ c)


def pulse_area_map(F, k, phases, f0, t_p, t0):
    """theta(r) eines Pulses der Laenge t_p ab t0, Omega ~ I. Der Puls ist
    derselbe Boxcar wie eine Belichtung, daher genuegt camera_frames_exact."""
    return t_p * camera_frames_exact(F, k, phases, f0, t_p, [t0])[0]


def sqrt_mean_series(F, k, phases, f0, mask=None, weights=None, oversample=6,
                     chunk=2000):
    """<sqrt(I)>_Maske auf einem feinen Zeitraster ueber eine Grundperiode.

    Rueckgabe (w, dt): w[j] zur Zeit t_j = j*dt, zyklisch."""
    G = _masked_columns(F, mask)
    wts = _masked_weights(F, mask, weights)
    H, K = _order_amplitudes_G(G, k, phases)
    M = G.shape[1]
    if K == 0 or f0 <= 0:
        return np.array([float(np.average(np.abs(H[0]), weights=wts))]), 1.0
    n_f = int(oversample * (2 * K + 1))
    w = np.zeros(n_f)
    for lo in range(0, M, chunk):
        hi = min(M, lo + chunk)
        A = np.zeros((n_f, hi - lo), dtype=np.complex64)
        A[:K + 1] = H[:, lo:hi].astype(np.complex64)
        E = np.fft.ifft(A, axis=0) * n_f
        w += (np.abs(E) * wts[None, lo:hi]).sum(axis=1)
    return w / max(float(np.sum(wts)), 1e-300), 1.0 / (f0 * n_f)


def sqrt_area_curve(w, dt, t_p, t0_list):
    """Integral von w ueber [t0, t0+t_p], zyklisch, mit linearer Interpolation
    an den Raendern. Fuer Omega ~ sqrt(I)."""
    n = w.size
    T = n * dt
    cum = np.concatenate(([0.0], np.cumsum(w) * dt))      # cum[j] = int_0^{t_j}
    total = cum[-1]

    def integral(t):
        q, r = np.divmod(t, T)
        x = r / dt
        j = np.floor(x).astype(int)
        frac = x - j
        base = np.take(cum, j, mode="clip") + frac * np.take(w, j % n) * dt
        return q * total + base

    t0_list = np.atleast_1d(np.asarray(t0_list, dtype=float))
    return integral(t0_list + t_p) - integral(t0_list)


def sqrt_area_map(F, k, phases, f0, t_p, t0, n_sub=41):
    """theta(r) fuer Omega ~ sqrt(I), numerisch ueber das Pulsfenster.

    Direkt an den n_sub Stuetzstellen im Fenster ausgewertet - das ist
    billiger als eine FFT ueber die ganze Periode, weil der Puls nur einen
    winzigen Ausschnitt davon braucht."""
    shape = F.shape[1:]
    H, K = order_amplitudes(F, k, phases)
    t = np.linspace(t0, t0 + t_p, n_sub)
    u = np.arange(K + 1)
    Bt = np.exp(2j * np.pi * f0 * np.outer(t, u))          # (n_sub, K+1)
    acc = np.zeros(H.shape[1])
    step = max(1, int(4e7 // max(n_sub, 1)))
    for lo in range(0, H.shape[1], step):
        hi = min(H.shape[1], lo + step)
        E = Bt @ H[:, lo:hi]
        acc[lo:hi] = _trapezoid(np.abs(E), t, axis=0)
    return acc.reshape(shape)


def _two_level_step(cg, ce, omega, delta, dt):
    """One step of a resonant-frame two-level system with constant
    H = (omega sigma_x + delta sigma_z)/2 over dt, exact for that step:
    exp(-i H dt) = cos(a/2) - i sin(a/2) (n . sigma), a = W dt.
    Same convention as kern/rb85_raman.excitation_series()."""
    W = np.hypot(omega, delta)
    a = W * dt
    ca, sa = np.cos(0.5 * a), np.sin(0.5 * a)
    safe = np.where(W > 0, W, 1.0)
    nx = np.where(W > 0, omega / safe, 0.0)
    nz = np.where(W > 0, delta / safe, 0.0)
    return ((ca - 1j * sa * nz) * cg - 1j * sa * nx * ce,
            -1j * sa * nx * cg + (ca + 1j * sa * nz) * ce)


def sqrt_law_excitation(H, f0, t_start, durations, omega_per_field, eta, g_ref,
                        steps_per_cycle=24, steps_per_beat=16, n_min=41,
                        n_max=20000):
    """Excitation for Omega ~ sqrt(I) WITH a differential light shift.

    Model
        Omega(r,t) = omega_per_field * |E(r,t)|
        delta(r,t) = eta * omega_per_field * |E(r,t)|^2 / g_ref

    eta is the part of delta/Omega caused by the multitone leg at the
    calibration point (half the total eta of rb85_raman for equal legs; the
    constant part of the clean leg is assumed compensated). That shift follows
    the intensity of the multitone leg, the Rabi frequency its field. At
    |E| = g_ref delta/Omega is exactly eta; elsewhere and at other times it
    is not, which is why the closed
    form sin^2(sqrt(1+eta^2) theta/2)/(1+eta^2) is wrong here and the
    Schroedinger equation is propagated instead (midpoint rule, each step
    exact for constant H, second order in dt).

    H          order amplitudes (K+1, P) from order_amplitudes()
    t_start    pulse start
    durations  pulse lengths at which the excitation is returned
    Returns an array (len(durations), P). With eta = 0 it reproduces
    sin^2(omega_per_field * int|E|dt / 2)."""
    H = np.asarray(H)
    durations = np.atleast_1d(np.asarray(durations, dtype=float))
    n_pix = H.shape[1]
    out = np.zeros((durations.size, n_pix))
    T = float(np.max(durations)) if durations.size else 0.0
    if T <= 0 or g_ref <= 0:
        return out
    K = H.shape[0] - 1
    E_max = float(np.max(np.sum(np.abs(H), axis=0)))          # |E| <= sum |H_u|
    W_max = float(np.hypot(omega_per_field * E_max,
                           eta * omega_per_field * E_max ** 2 / g_ref))
    n = max(n_min, steps_per_beat * K * f0 * T, steps_per_cycle * W_max * T / (2 * np.pi))
    n = int(min(n_max, np.ceil(n)))
    d_cl = np.clip(durations, 0.0, T)
    grid = np.union1d(np.linspace(0.0, T, n + 1), d_cl)
    rec = np.searchsorted(grid, d_cl)
    u = np.arange(K + 1)
    cg = np.ones(n_pix, dtype=complex)
    ce = np.zeros(n_pix, dtype=complex)
    for j in range(1, grid.size):
        dt = grid[j] - grid[j - 1]
        if dt > 0:
            tm = t_start + 0.5 * (grid[j] + grid[j - 1])
            E = np.exp(2j * np.pi * f0 * u * tm) @ H
            I = E.real ** 2 + E.imag ** 2
            cg, ce = _two_level_step(cg, ce, omega_per_field * np.sqrt(I),
                                     eta * omega_per_field * I / g_ref, dt)
        hit = rec == j
        if hit.any():
            out[hit] = np.abs(ce) ** 2
    return out


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
        self.k = np.asarray(k, dtype=int)
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
        # Without beating the intensity is constant, delta = eta*Omega holds
        # for both laws and the closed form is exact.
        ideal = exc(np.asarray(t_p_values) / t_pi * theta_pi)
        if self.law != "I" and eta != 0.0:
            # Omega ~ sqrt(I), delta ~ I: the ratio is not constant, the closed
            # form is wrong - propagate. Omega = (pi/theta_pi)|E|, so a pulse
            # of t_pi at the mean field g_ref = theta_pi/t_pi is a pi pulse.
            H, _ = _order_amplitudes_G(self.G, self.k, phases)
            f0 = self.w0 / (2 * np.pi)
            om_pf, g_ref = np.pi / theta_pi, theta_pi / t_pi
            tpv = np.asarray(t_p_values, dtype=float)
            curve = lambda tt: np.mean(sqrt_law_excitation(
                H, f0, tt, tpv, om_pf, eta, g_ref), axis=1)
            trig = curve(t0)
            if T0 is None or not np.isfinite(T0) or T0 <= 0 or n_random < 2:
                return ideal, trig, trig
            t0s = np.linspace(0.0, T0, n_random, endpoint=False)
            none = np.mean([curve(tt) for tt in t0s], axis=0)
            return ideal, trig, none
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
