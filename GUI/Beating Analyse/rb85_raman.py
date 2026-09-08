"""rb85_raman.py - two-photon Raman on Rb-85, with real atomic data from ARC.

Effective two-level description of the ground-state hyperfine transition

    |5S_1/2, F=2, m> <-> |5S_1/2, F=3, m>

driven by a two-frequency laser field via the 5P manifold.  "Effective two
level" here means: the excited 5P states are adiabatically eliminated, but
the elimination is done with the FULL sum over both fine-structure lines and
all their hyperfine levels, using signed dipole matrix elements from ARC.
The relative signs matter - the F' contributions interfere, and dropping them
is wrong by tens of percent whenever the detuning is comparable to the
excited-state hyperfine splitting.

What comes out are three coefficients, all LINEAR in the optical intensity:

    Omega(I)   = C_rabi    * I      two-photon Rabi frequency   [rad/s]
    delta_LS(I)= C_shift   * I      differential light shift    [rad/s]
    Gamma(I)   = C_scatter * I      photon scattering rate      [1/s]

Because Omega and delta_LS share the same factor I, their ratio

    eta = delta_LS / Omega

is a pure number - independent of intensity, of position, and of time.  That
is what makes the two-level picture close: the light shift caps the contrast
and rescales the Rabi frequency, but it cannot introduce any extra spatial
structure of its own.

Geometry assumed
----------------
Co-propagating beams, sigma+ on both legs.  This is not cosmetic.  The scalar
part of the AC Stark operator is proportional to the identity and cannot
drive a hyperfine transition at all; only the vector part can, and for two
legs of polarisation eps_1, eps_2 it goes as eps_1* x eps_2.  For two
LINEARLY polarised legs that cross product vanishes and nothing happens.
sigma+/sigma+ gives eps* x eps = z, which drives Delta m_F = 0 - the clock
transition.

Co-propagating also means Delta k = w_hfs/c = 64 1/m, a recoil velocity of
5e-8 m/s.  The transition is Doppler-free and completely insensitive to the
atom's motion, unlike a counter-propagating Raman pair.

Sign convention: Delta < 0 is RED detuning, measured from the 5P_1/2
hyperfine centroid for the leg that addresses F = 2.

Intensity interface
-------------------
Everything downstream takes an INTENSITY in W/m^2.  For a simple beam use
gaussian_peak_intensity(P, w).  For a structured or time-dependent profile
(e.g. the multitone flat-top of Beating_Multitone_GUI.py) hand an array
I(t) to excitation_series() instead - nothing else changes.
"""

import numpy as np
import arc
from scipy.constants import hbar, epsilon_0, c, physical_constants

a0 = physical_constants["Bohr radius"][0]
e_charge = physical_constants["elementary charge"][0]
MU_B_H = physical_constants["Bohr magneton in Hz/T"][0] * 1e-4    # Hz/G

ATOM = arc.Rubidium85()

G_F = 1.0 / 3.0        # |g_F| of both Rb-85 5S_1/2 hyperfine levels
# Measured quadratic Zeeman shift of the Rb-85 clock transition,
# 1296.8(33) Hz/G^2  [arXiv:0812.0231].
K_QZ = 1296.8

_LINES = {"D1": dict(j=0.5, Fp=[2, 3]), "D2": dict(j=1.5, Fp=[1, 2, 3, 4])}


def _hfs(n, l, j, F):
    return ATOM.getHFSEnergyShift(j, F, *ATOM.getHFSCoefficients(n, l, j))


def _line_data():
    out = {}
    for name, d in _LINES.items():
        j = d["j"]
        out[name] = dict(
            j=j, Fp=d["Fp"],
            f=ATOM.getTransitionFrequency(5, 0, 0.5, 5, 1, j),   # Hz, centroid
            gamma=1.0 / ATOM.getStateLifetime(5, 1, j),          # 1/s
            hfs={F: _hfs(5, 1, j, F) for F in d["Fp"]})
    return out


LINES = _line_data()
W_HFS = _hfs(5, 0, 0.5, 3) - _hfs(5, 0, 0.5, 2)     # Hz, 3.0357 GHz


def _dip(Fg, mg, j_e, Fe, me, q):
    """Signed <5P j_e Fe me| d_q |5S1/2 Fg mg> in C m."""
    return ATOM.getDipoleMatrixElementHFS(
        5, 0, 0.5, Fg, mg, 5, 1, j_e, Fe, me, q) * a0 * e_charge


def zeeman_detuning(m, B_gauss):
    """Two-photon detuning [rad/s] of |F=2,m> <-> |F=3,m> in a bias field.

    First order: (g_3 - g_2) m mu_B B / hbar = 2 pi * 0.933 MHz/G * m.
    Only m = 0 escapes it; there the leading term is the measured quadratic
    shift, 1296.8 Hz/G^2, which at a few gauss is a few kHz - small against a
    Rabi frequency of some 100 kHz, and constant rather than noisy."""
    if m == 0:
        return 2 * np.pi * K_QZ * B_gauss ** 2
    return 2 * np.pi * 2 * G_F * MU_B_H * m * B_gauss


class RamanRb85:
    """Raman coefficients for one detuning / power ratio / polarisation."""

    def __init__(self, delta_Hz=-8e9, beta=0.5, q=1):
        self.delta = float(delta_Hz)
        self.beta = float(beta)
        self.q = int(q)
        self._cache = {}

    # -- detunings of the two legs from every intermediate level ----
    def _det(self, line, Fe):
        """Detuning [rad/s] of the two-photon-resonant pair from |line, Fe>.

        Both legs are two-photon resonant, so each is at the SAME detuning
        from a given excited level when addressing 'its own' ground state.
        They differ only in their distance from the other ground state's
        transitions - and that difference is the whole origin of the
        differential light shift."""
        d0 = self.delta + (LINES["D1"]["f"] - LINES[line]["f"])
        return 2 * np.pi * (d0 - LINES[line]["hfs"][Fe])

    def coeff(self, m=0):
        """C_rabi, C_shift, C_scatter (per W/m^2) and eta for sublevel m."""
        if m in self._cache:
            return self._cache[m]
        k = 2.0 / (epsilon_0 * c * hbar ** 2)       # Omega^2 = d^2 * k * I
        b, w = self.beta, 2 * np.pi * W_HFS
        rabi = 0.0
        shift = {2: 0.0, 3: 0.0}
        scat = {2: 0.0, 3: 0.0}
        parts = {}
        for line, L in LINES.items():
            for Fe in L["Fp"]:
                me = m + self.q
                d2 = _dip(2, m, L["j"], Fe, me, self.q)
                d3 = _dip(3, m, L["j"], Fe, me, self.q)
                D = self._det(line, Fe)
                term = d2 * d3 * k * np.sqrt(b * (1 - b)) / (2 * D)
                rabi += term
                parts[f"{line} F'={Fe}"] = term
                shift[2] += d2 ** 2 * k * (b / (4 * D) + (1 - b) / (4 * (D - w)))
                shift[3] += d3 ** 2 * k * (b / (4 * (D + w)) + (1 - b) / (4 * D))
                g = L["gamma"]
                scat[2] += g * d2 ** 2 * k * (b / (4 * D ** 2)
                                              + (1 - b) / (4 * (D - w) ** 2))
                scat[3] += g * d3 ** 2 * k * (b / (4 * (D + w) ** 2)
                                              + (1 - b) / (4 * D ** 2))
        out = dict(C_rabi=abs(rabi), C_shift=shift[3] - shift[2],
                   C_scatter_2=scat[2], C_scatter_3=scat[3],
                   C_shift_common=0.5 * (shift[2] + shift[3]),
                   parts=parts,
                   eta=(shift[3] - shift[2]) / abs(rabi) if rabi else np.nan)
        self._cache[m] = out
        return out

    # -- convenience ------------------------------------------------
    def omega(self, I, m=0):
        return self.coeff(m)["C_rabi"] * np.asarray(I)

    def eta(self, m=0):
        return self.coeff(m)["eta"]

    def contrast_cap(self, m=0):
        return 1.0 / (1.0 + self.eta(m) ** 2)

    def intensity_for_rabi(self, f_rabi_Hz, m=0):
        """Intensity [W/m^2] giving a two-photon Rabi frequency f_rabi_Hz."""
        return 2 * np.pi * f_rabi_Hz / self.coeff(m)["C_rabi"]

    def cg(self):
        """Relative coupling of the five m_F, normalised to the clock state."""
        ref = self.coeff(0)["C_rabi"]
        return {m: self.coeff(m)["C_rabi"] / ref for m in (-2, -1, 0, 1, 2)}


# ------------------------------------------------------------------ beams
def gaussian_peak_intensity(power_W, waist_m):
    """Peak (on-axis) intensity of a Gaussian beam, I0 = 2P/(pi w^2)."""
    return 2.0 * power_W / (np.pi * waist_m ** 2)


def power_for_intensity(I, waist_m):
    return I * np.pi * waist_m ** 2 / 2.0


# ------------------------------------------------------- the excitation
def excitation_constant(t_p, I, raman, m=0, B_gauss=0.0, compensate=False,
                        with_shift=True, with_scatter=True):
    """Excitation after a square pulse of length t_p at constant intensity.

    For constant I every term of H is constant, so the closed form is EXACT:

        P = Omega^2/(Omega^2+delta^2) * sin^2( sqrt(Omega^2+delta^2) t/2 )

    with delta the total two-photon detuning (light shift + Zeeman - any
    deliberate compensation).  With delta = eta*Omega this is the familiar
    P = 1/(1+eta^2) sin^2( sqrt(1+eta^2) Omega t/2 ).

    Scattering enters as a survival factor exp(-Gamma t); Gamma is the mean of
    the two ground-state rates, which differ by only a few percent.
    """
    cm = raman.coeff(m)
    Om = cm["C_rabi"] * I
    de = (cm["C_shift"] * I if with_shift else 0.0) + zeeman_detuning(m, B_gauss)
    if compensate:
        de = de - (cm["C_shift"] * I if with_shift else 0.0)
    t_p = np.asarray(t_p, dtype=float)
    W = np.hypot(Om, de)
    P = np.where(W > 0, (Om / np.where(W > 0, W, 1.0)) ** 2
                 * np.sin(W * t_p / 2) ** 2, 0.0)
    if with_scatter:
        G = 0.5 * (cm["C_scatter_2"] + cm["C_scatter_3"]) * I
        P = P * np.exp(-G * t_p)
    return P


def excitation_series(t, I_t, raman, m=0, B_gauss=0.0, compensate=False,
                      I_ref=None, with_shift=True, with_scatter=True):
    """Excitation for an ARBITRARY intensity history I(t).

    This is the entry point for a structured profile - hand it I(t) at one
    position (or loop it over positions) and it returns the excitation after
    every time step, i.e. the full Rabi curve in one pass.

    Over one step H is constant, so exp(-i H dt) is analytic:
        exp(-i H dt) = cos(a/2) - i sin(a/2) (n.sigma),
        a = sqrt(Omega^2+delta^2) dt,  n = (Omega, 0, delta)/sqrt(...)
    That is unitary to machine precision, unconditionally stable and second
    order in dt when H is sampled at the centre of each step.

    Note this is NOT just decoration around the closed form: as soon as a term
    appears that does not scale with I - a Zeeman detuning, a static
    two-photon detuning - H(t) at different times stops commuting, the pulse
    area theorem fails, and integration is the only correct route.
    """
    t = np.asarray(t, dtype=float)
    I_t = np.asarray(I_t, dtype=float)
    # t[0] is the pulse start, so out[0] = 0 and step j carries the system from
    # t[j-1] to t[j].  The intensity of that step is taken at its CENTRE, which
    # makes the scheme second order in dt instead of first.
    dt = np.diff(t)
    I_mid = 0.5 * (I_t[:-1] + I_t[1:])
    cm = raman.coeff(m)
    Om = cm["C_rabi"] * I_mid
    de = (cm["C_shift"] * I_mid if with_shift else np.zeros_like(I_mid))
    de = de + zeeman_detuning(m, B_gauss)
    if compensate:
        ref = I_t.mean() if I_ref is None else I_ref
        de = de - (cm["C_shift"] * ref if with_shift else 0.0)
    G = (0.5 * (cm["C_scatter_2"] + cm["C_scatter_3"]) * I_mid
         if with_scatter else np.zeros_like(I_mid))

    cg, ce, surv = 1.0 + 0j, 0.0 + 0j, 1.0
    out = np.empty(t.size)
    out[0] = 0.0
    for j in range(dt.size):
        W = np.hypot(Om[j], de[j])
        a = W * dt[j]
        ca, sa = np.cos(a / 2), np.sin(a / 2)
        nx, nz = (Om[j] / W, de[j] / W) if W > 0 else (0.0, 0.0)
        cg, ce = ((ca - 1j * sa * nz) * cg + (-1j * sa * nx) * ce,
                  (-1j * sa * nx) * cg + (ca + 1j * sa * nz) * ce)
        surv *= np.exp(-G[j] * dt[j])
        out[j + 1] = abs(ce) ** 2 * surv
    return out


def thermal_average(fn, raman, weights=None):
    """Average a per-m quantity over an unpolarised F = 2 population.

    fn(m) must return the excitation curve for sublevel m.  Without optical
    pumping the five m_F are equally populated and each flops at its own
    Clebsch-Gordan-reduced Rabi frequency, which damps the ensemble curve in a
    quasi-periodic way - with revivals, unlike the monotonic damping that an
    intensity spread produces.  Telling the two apart in a measurement is the
    point of having both in one tool.
    """
    ms = (-2, -1, 0, 1, 2)
    w = weights or {m: 0.2 for m in ms}
    return sum(w[m] * np.asarray(fn(m)) for m in ms)


if __name__ == "__main__":
    r = RamanRb85(-8e9)
    print(f"w_hfs = {W_HFS / 1e9:.7f} GHz")
    for n, L in LINES.items():
        print(f"{n}: {c / L['f'] * 1e9:.4f} nm, Gamma/2pi = "
              f"{L['gamma'] / 2 / np.pi / 1e6:.4f} MHz")
    print(f"\nDelta = {r.delta / 1e9:g} GHz:  eta = {r.eta():+.4f}, "
          f"cap = {r.contrast_cap() * 100:.2f} %")
    I = r.intensity_for_rabi(0.2e6)
    print(f"I for 200 kHz = {I / 1e4:.3f} W/cm^2 "
          f"= {power_for_intensity(I, 1.1e-6) * 1e9:.1f} nW at w = 1.1 um")
    print("CG ratios:", {k: round(v, 4) for k, v in r.cg().items()})
