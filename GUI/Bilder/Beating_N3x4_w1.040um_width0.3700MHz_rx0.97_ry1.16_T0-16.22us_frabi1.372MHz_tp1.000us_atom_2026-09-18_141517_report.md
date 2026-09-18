# Pulse area and trigger jitter

Generated 2026-09-18_141517 by `pulse_timing.py` (Beating_Multitone_GUI).

## Profile

| quantity | value |
|---|---|
| N_x x N_y | 3 x 4 |
| width_x / width_y | 0.370000 / 0.370000 MHz |
| r_x / r_y | 0.9700 / 1.1600 |
| waist | 1.0400 um |
| beam profile | Airy, airy factor 1.4830 |
| build | telescope f1 = 75.00 mm, f2 = 750.00 mm |
| lambda / RF offset | 795.00 nm / 100.0000 MHz |
| AOD | theta_max = 43.0 mrad, f_band = 36.0 MHz, v_ac = 665.6 m/s |
| f_0 / T_0 | 61.666667 kHz / 16.2162 us |
| grid / frames / periods | 200^2 / 60 / 3 |
| tone phases x (deg) | 0.00, 12.50, 25.00 |
| tone phases y (deg) | 0.00, 98.00, 16.00, 114.00 |

## Evaluation region

| quantity | value |
|---|---|
| region | atom, sigma = 107.6 nm |
| atom position (global) | x = 1.1685 um, y = 1.1685 um |
| atom T / nu_r | 17.00 uK / 60.40 kHz |
| sigma | 107.64 nm |
| weighting | Gaussian position probability density W(r), **not** a hard mask |

All curves are weighted **spatial means** `<x>_W = sum_r W(r) x(r) / sum_r W(r)`; only the map is point by point. The excitation is averaged after the sin^2, not before.

## Pulse

| quantity | value |
|---|---|
| Omega/2pi (reference) | 1.371524 MHz, on the time averaged intensity in the region |
| coupling law | Omega ~ I |
| T_p | 1.000000 us (pi pulse at 0.364558 us) |
| pulse start t_0 | 15.300000 us in the cycle (t_0 is the START) |
| pulse end t_0 + T_p | 16.300000 us |
| **mean area** <theta>_W(0) | **5.243143 pi** |
| area at the reference | 2.743048 pi |
| sigma_theta / <theta>_W | 0.3127 % (spatial spread in the region) |
| same for the time average | 0.3960 % |
| <sin^2(theta/2)>_W | 0.846908 (per point, then averaged) |
| eta (light shift) | -0.060557 |
| deviation at +-1.000 us | 68.9886 % |

## Power (in the profile, i.e. after both AODs)

| quantity | value |
|---|---|
| Delta | 50.000 GHz |
| C_rabi | 11.6573 rad/s per W/m^2 (kern/rb85_raman.py) |
| I_ref | 73.923722 W/cm^2 (mean over time AND region) |
| A_eff | 13.5275 um^2 (= integral I dA / <I>) |
| **P_profile** | **0.01000000 mW** |
| P per spot | 0.83333333 uW mean, 0.91332829 uW brightest, 0.76373142 uW weakest |
| P per RF tone x | 3.33333333 uW mean, 3.40136054 uW strongest |
| P per RF tone y | 2.50000000 uW mean, 2.68518519 uW strongest |
| scattering rate F=3 | 929.7 1/s |
| scattering per pulse | 0.0929 % |
| eta (from Delta) | -0.0606 -> contrast <= 0.9963 |
| P for a pi pulse of T_p | 3.645580 uW |
| not included | AOD diffraction efficiency, optics transmission, intermodulation |

## Figures

Neither PDF carries a title; this is what the panels show.

**`_curves.pdf`**, three panels top to bottom:

1. Profile intensity `I(t)` divided by its own time average, over ONE beat period. The red line marks the pulse **start** t_0, the shading the pulse duration T_p.
2. Mean pulse area `<theta>/pi` versus the pulse start t_0, over the same beat period.
3. The same curve zoomed around the chosen t_0. x is the trigger error, the right axis the deviation from the area at zero error.

Breiten: `_curves.pdf` ist 16 cm breit gespeichert und gehoert mit `width=\textwidth` ins Dokument, die Karte 8 cm und mit `width=0.5\textwidth`. Beide dann unskaliert, Beschriftung 10 pt.

**`_map.pdf`**: pulse area `theta(r)/pi` point by point, no averaging. The white rings are 1 and 2 sigma of the atomic position distribution (or the outline of the hard mask). In the atom weighted case the axes are RELATIVE to the atom, so the origin is the atom; the global origin would be the position of a tone at the bare RF offset, which is why the pattern as a whole sits at positive coordinates.

## Files

- `Beating_N3x4_w1.040um_width0.3700MHz_rx0.97_ry1.16_T0-16.22us_frabi1.372MHz_tp1.000us_atom_2026-09-18_141517_curves.pdf`
- `Beating_N3x4_w1.040um_width0.3700MHz_rx0.97_ry1.16_T0-16.22us_frabi1.372MHz_tp1.000us_atom_2026-09-18_141517_areamap.pdf`
- `Beating_N3x4_w1.040um_width0.3700MHz_rx0.97_ry1.16_T0-16.22us_frabi1.372MHz_tp1.000us_atom_2026-09-18_141517_period.txt`, `Beating_N3x4_w1.040um_width0.3700MHz_rx0.97_ry1.16_T0-16.22us_frabi1.372MHz_tp1.000us_atom_2026-09-18_141517_delayscan.txt`, `Beating_N3x4_w1.040um_width0.3700MHz_rx0.97_ry1.16_T0-16.22us_frabi1.372MHz_tp1.000us_atom_2026-09-18_141517_intensity.txt` (data columns)

## Summary line from the GUI

```
MEAN over atom, sigma = 107.6 nm:  <theta>_W(0) = 5.2431 pi  ->  excitation <sin^2(theta/2)>_W = 0.847, eta = -0.061   |   sigma_theta/<theta>_W = 0.31 %  (time average in the same region: 0.40 %)
the pulse STARTS at t_0 = 15.3000 us and ends at t_0 + T_p = 16.3000 us (t_0 is the start, never the centre);  at +-1.000 us trigger error the mean area deviates by 68.99 %, 20 ns until 1 % area error
POWER: P_profile = 10.0000 uW -> I_ref = 73.92 W/cm^2 (mean over time AND region) -> Omega/2pi = 1.3715 MHz  |  per spot 0.8333 uW mean, 0.9133 uW brightest  |  per RF tone x 3.4014 uW, y 2.6852 uW (strongest)  |  scattering 0.0929 % per pulse  |  a pi pulse of 1.000 us would need 3.6456 uW
```
