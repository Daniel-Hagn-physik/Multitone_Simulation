"""beispiel_3x4_nachrechnen.py - rechnet alle Zahlen des 3x4-Beispiels nach.

Gehoert zum Abschnitt "Time-dependent interference of multiple spots" der
Arbeit. Die Bezeichnungen folgen dem Text:

    W_x, W_y        Frequenzbreite der AODs          f_off   Offset
    N_x, N_y        Tonzahl                           a_x, a_y Intensitaetsgewichte
    r_x, r_y        amplitude ratios (aeussere Toene) A_s = sqrt(a_x,i a_y,j)
    u(r)            reelle Mode, u(0) = 1             g_s = A_s u(r - c_s)
    phi_s           = phi_x,i + phi_y,j               k_s     Frequenzordnung
    f_0, T_0        Grundfrequenz, -periode           S_m     Gruppe mit k_s = m
    H_m             Gruppenfeld                       D_d     Fourierkoeffizient
    p_x, p_y        Pitch in der Fokusebene           z_ij = g_ij e^{i phi_ij}

Das Skript ist absichtlich UNABHAENGIG von kern/beating_physik.py geschrieben,
damit es die GUI-Zahlen gegenprueft statt sie nur zu wiederholen. Am Ende wird
- falls kern/ gefunden wird - trotzdem mit dem GUI-Code verglichen.

Die Frequenzen werden mit Bruechen (fractions.Fraction) gerechnet: f_0 und die
Ordnungen k_s sind dann exakt, ohne Rundung.

Wo dieselben Groessen im GUI-Code stehen:

    <I>_t, Var_t         kern/beating_physik.py  ->  time_stats_exact()
    H_m (Ordnungen)      kern/beating_physik.py  ->  order_amplitudes(), beat_orders()
    sigma_d je Ordnung   kern/beating_physik.py  ->  VariationObjective.components()
    f_0, Beat-Linien     kern/beating_physik.py  ->  fundamental_beat_frequency(),
                                                     unique_beat_frequencies()
    entartete Gruppen    kern/beating_physik.py  ->  degenerate_groups()
    Leistung, Ueberlapp  kern/beating_physik.py  ->  profile_total_power(),
                                                     spot_overlap_integral()
    Auswertebereiche     Beating_Multitone_GUI.py -> recompute() (Plateau, Kreis,
                                                     Spotzentren), _target_mask()
    Atom-Gewicht W(r)    kern/beating_physik.py  ->  atom_local_stack(), sigma_thermal()

Start:  python beispiel_3x4_nachrechnen.py
Braucht numpy und scipy. Laeuft ab Python 3.9.
"""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.special import j1

# =============================================================================
# Arbeitspunkt - hier aendern, der Rest rechnet allgemein
# =============================================================================
N_x, N_y = 3, 4
W_x = W_y = Fraction(370_000)          # Hz, gleiche width
f_off = Fraction(100_000_000)          # Hz, Offset beider AODs
r_x, r_y = 0.97, 1.16                  # amplitude ratios der aeusseren Toene

waist = 1.04e-6                        # m, Waist nach den Linsen
use_airy = True
AIRY_FACTOR = 1.4830                   # erste Nullstelle = AIRY_FACTOR * waist

# Optik (fuer dr/df und die Pitches)
f1, f2, fLO = 75e-3, 750e-3, 52.88e-3  # m
theta_max, f_band = 43e-3, 36e6        # rad, Hz

# Rechengitter wie im Beating-GUI
grid_n = 200
pad_factor = 2.5                       # Rand in Einheiten von win_eff

# Falle und Puls
nu_r = 60.4e3                          # Hz, radiale Fallenfrequenz
T_atom = 17e-6                         # K, Atomtemperatur (fuer das Atom-Gewicht)
f_rabi = 1.0e6                         # Hz

# Auswertebereich fuer sigma_d und sigma_I.  Der Text der Arbeit benutzt
# "rechteck" - dieselbe Flaeche, ueber der auch die Uniformity U definiert ist.
#   "rechteck" - Rechteck zwischen den aeussersten Spotzentren
#   "plateau"  - <I>_t > 0.5 max   (frueherer Standard, phasenabhaengige Maske)
#   "kreis"    - Kreis um die Mitte, Radius kreis_radius
#   "atom"     - Gauss-Gewicht der Aufenthaltswahrscheinlichkeit, sigma aus nu_r und T
REGION = "rechteck"
kreis_radius = 1.0e-6                  # m, nur fuer REGION = "kreis"


sigma_atom = None                      # wird nach sigma_thermal() gesetzt


def titel(text):
    print("\n" + "=" * 78 + "\n" + text + "\n" + "=" * 78)


def frac_gcd(a, b):
    """ggT zweier Brueche: ggT(p/q, r/s) = ggT(p*s, r*q) / (q*s)."""
    a, b = Fraction(a), Fraction(b)
    return Fraction(math.gcd(a.numerator * b.denominator,
                             b.numerator * a.denominator),
                    a.denominator * b.denominator)


def amps_from_ratio(r, N):
    """a = [r, 1, ..., 1, r] - Intensitaetsgewichte einer Achse."""
    a = np.ones(N)
    a[0] = a[-1] = r
    return a


def sigma_thermal(nu, T, masse=84.911789738 * 1.66053906660e-27):
    """Thermische 1-sigma-Ortsbreite im harmonischen Topf, mit Nullpunktsbewegung."""
    hbar, kB = 1.054571817e-34, 1.380649e-23
    om = 2 * np.pi * nu
    return math.sqrt(hbar / (2 * masse * om) / math.tanh(hbar * om / (2 * kB * T)))


def schroeder(N):
    """Schroeder-Phasen phi_n = -pi n(n-1)/N."""
    n = np.arange(N)
    return -np.pi * n * (n - 1) / N


sigma_atom = sigma_thermal(nu_r, T_atom)

# =============================================================================
# 1. Toene und Grundfrequenz
# =============================================================================
titel("1. Toene und Grundfrequenz")

f_x = [f_off + W_x * Fraction(i, N_x - 1) for i in range(N_x)]
f_y = [f_off + W_y * Fraction(j, N_y - 1) for j in range(N_y)]
df_x = W_x / (N_x - 1)
df_y = W_y / (N_y - 1)

print("f_x,i [MHz]:", ", ".join(f"{float(f) / 1e6:.3f}" for f in f_x))
print("f_y,j [MHz]:", ", ".join(f"{float(f) / 1e6:.3f}" for f in f_y))
print(f"Delta f_x = W_x/{N_x - 1} = {float(df_x) / 1e3:.2f} kHz")
print(f"Delta f_y = W_y/{N_y - 1} = {float(df_y) / 1e3:.2f} kHz")

# f_0 als ggT aller Spot-Differenzen (allgemein, auch fuer ungleiche widths)
f_s_exakt = {(i, j): f_x[i] + f_y[j] for i in range(N_x) for j in range(N_y)}
f_min = min(f_s_exakt.values())
f_0 = Fraction(0)
for f in f_s_exakt.values():
    if f != f_min:
        f_0 = frac_gcd(f_0, f - f_min) if f_0 else f - f_min
T_0 = 1 / f_0

print(f"f_0 = ggT aller Differenzen = {float(f_0) / 1e3:.4f} kHz"
      f"   ->  T_0 = {float(T_0) * 1e6:.3f} us")
if W_x == W_y:
    L = math.lcm(N_x - 1, N_y - 1)
    print(f"Probe Formel: W / lcm({N_x - 1},{N_y - 1}) = W/{L} = "
          f"{float(W_x / L) / 1e3:.4f} kHz  ->  "
          f"{'stimmt' if W_x / L == f_0 else 'WEICHT AB'}")
step_x = df_x / f_0
step_y = df_y / f_0
print(f"Delta f_x = {step_x} f_0,   Delta f_y = {step_y} f_0")

# Geometrie: theta = theta_max (f - f_off)/f_band, r = (f1 fLO/f2) tan(theta)
def position(f):
    theta = theta_max * float(f - f_off) / f_band
    return (f1 * fLO / f2) * math.tan(theta)

drdf = (f1 * fLO / f2) * theta_max / f_band
p_x = position(f_x[1]) - position(f_x[0])
p_y = position(f_y[1]) - position(f_y[0])
print(f"dr/df = {drdf * 1e12:.3f} nm/kHz")
print(f"p_x = {p_x * 1e6:.4f} um,  p_y = {p_y * 1e6:.4f} um,  "
      f"Ausdehnung x/y = {(position(f_x[-1]) - position(f_x[0])) * 1e6:.4f} / "
      f"{(position(f_y[-1]) - position(f_y[0])) * 1e6:.4f} um")

# =============================================================================
# 2. Frequenzordnungen und Gruppen
# =============================================================================
titel("2. Frequenzordnungen k_s und Gruppen S_m")

spots = [(i, j) for i in range(N_x) for j in range(N_y)]
k_s = {s: int((f_s_exakt[s] - f_min) / f_0) for s in spots}
K = max(k_s.values())

print(f"k_ij = {step_x} i + {step_y} j,   f_ij = "
      f"{float(f_min) / 1e6:.0f} MHz + k_ij f_0\n")
print("       " + "".join(f"  j={j:<3d}" for j in range(N_y)))
for i in range(N_x):
    print(f"  i={i}  " + "".join(f"  {k_s[(i, j)]:<5d}" for j in range(N_y)))

S_m = {m: [s for s in spots if k_s[s] == m] for m in range(K + 1)}
besetzt = [m for m in S_m if S_m[m]]
leer = [m for m in S_m if not S_m[m]]
entartet = {m: S_m[m] for m in besetzt if len(S_m[m]) > 1}
print(f"\n{len(spots)} Spots in {len(besetzt)} Gruppen, "
      f"leere Ordnungen: {leer}")
for m in besetzt:
    terme = " + ".join(f"z_{i},{j}" for i, j in S_m[m])
    print(f"  H_{m:<2d} = {terme}")
for m, grp in entartet.items():
    (ia, ja), (ib, jb) = grp[0], grp[1]      # nicht j1 nennen - das ist die Besselfunktion
    d_um = math.hypot((ia - ib) * p_x, (ja - jb) * p_y) * 1e6
    print(f"entartet: Ordnung {m}: {grp}, Abstand {d_um:.3f} um")

# =============================================================================
# 3. Felder auf dem Rechengitter
# =============================================================================
a_x = amps_from_ratio(r_x, N_x)
a_y = amps_from_ratio(r_y, N_y)
A_s = {(i, j): math.sqrt(a_x[i] * a_y[j]) for (i, j) in spots}
c_s = {(i, j): (position(f_x[i]), position(f_y[j])) for (i, j) in spots}

k_airy = 3.83170597 / (AIRY_FACTOR * waist)


def u(r):
    """Reelle Mode, u(0) = 1. Airy behaelt das Vorzeichen der Ringe."""
    if not use_airy:
        return np.exp(-r ** 2 / waist ** 2)
    x = k_airy * np.asarray(r, dtype=float)
    out = np.ones_like(x)
    m = x > 1e-12
    out[m] = 2 * j1(x[m]) / x[m]
    return out


# quadratisches Gitter wie compute_grid() im GUI
win_eff = waist * (AIRY_FACTOR if use_airy else 1.0)
xs = [c[0] for c in c_s.values()]
ys = [c[1] for c in c_s.values()]
pad = pad_factor * win_eff
x_lo, x_hi, y_lo, y_hi = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
cx0, cy0 = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
half = max(x_hi - x_lo, y_hi - y_lo) / 2
X, Y = np.meshgrid(np.linspace(cx0 - half, cx0 + half, grid_n),
                   np.linspace(cy0 - half, cy0 + half, grid_n))

# Mitte und halbe Kantenlaenge des Spot-Rechtecks, Atom-Breite
x_mitte = 0.5 * (min(xs) + max(xs))
y_mitte = 0.5 * (min(ys) + max(ys))
h_x = 0.5 * (max(xs) - min(xs))
h_y = 0.5 * (max(ys) - min(ys))

g = {s: A_s[s] * u(np.hypot(X - c_s[s][0], Y - c_s[s][1])) for s in spots}
I_inkoh = sum(g[s] ** 2 for s in spots)


def statistik(phi_x, phi_y):
    """H_m, D_0 und alle D_d auf dem Gitter fuer gegebene Tonphasen."""
    phi_s = {(i, j): phi_x[i] + phi_y[j] for (i, j) in spots}
    H = np.zeros((K + 1,) + X.shape, dtype=complex)
    for s in spots:
        H[k_s[s]] += g[s] * np.exp(1j * phi_s[s])
    D0 = np.sum(np.abs(H) ** 2, axis=0)                    # = <I>_t
    D = {d: np.sum(H[d:] * np.conj(H[:K + 1 - d]), axis=0)  # D_d = sum H_{n+d} H_n^*
         for d in range(1, K + 1)}
    return phi_s, H, D0, D


phi_x_wp = np.radians([0.0, 12.5, 25.0])     # Arbeitspunkt dieser Arbeit
phi_y_wp = np.radians([0.0, 98.0, 16.0, 114.0])

phasen = {"phi = 0": (np.zeros(N_x), np.zeros(N_y)),
          "Schroeder": (schroeder(N_x), schroeder(N_y)),
          "Arbeitspkt": (phi_x_wp, phi_y_wp)}
erg = {name: statistik(*ph) for name, ph in phasen.items()}

# =============================================================================
# 4. Statische Interferenz des entarteten Paars
# =============================================================================
titel("3. Statische Interferenz  <I>_t - sum g_s^2")

P_spot = 4 * math.pi / k_airy ** 2 if use_airy else math.pi * waist ** 2 / 2


def overlap(d):
    """Integral u(r-c1) u(r-c2) dA, |c1-c2| = d, analytisch."""
    if use_airy:
        return P_spot * float(u(np.array([d]))[0])
    return P_spot * math.exp(-d ** 2 / (2 * waist ** 2))


P_inkoh = sum(a_x[i] * a_y[j] for i, j in spots) * P_spot
for name, (phi_x, phi_y) in phasen.items():
    phi_s, H, D0, D = erg[name]
    I_stat = D0 - I_inkoh
    idx = np.unravel_index(np.argmax(np.abs(I_stat)), I_stat.shape)
    print(f"\n{name}:")
    P_stat = 0.0
    for m, grp in entartet.items():
        for a in range(len(grp)):
            for b in range(a + 1, len(grp)):
                s1, s2 = grp[a], grp[b]
                dphi = phi_s[s1] - phi_s[s2]
                dphi_w = (dphi + np.pi) % (2 * np.pi) - np.pi
                dist = math.hypot(c_s[s1][0] - c_s[s2][0], c_s[s1][1] - c_s[s2][1])
                P_stat += 2 * A_s[s1] * A_s[s2] * math.cos(dphi) * overlap(dist)
                print(f"  Paar {s1}-{s2}: A^2 = {A_s[s1] * A_s[s2]:.4f}, "
                      f"Phasendifferenz = {dphi_w / np.pi:+.4f} pi "
                      f"({np.degrees(dphi_w):+.1f} deg), cos = {math.cos(dphi):+.4f}")
    print(f"  max |<I>_t - inkoh.| / max(inkoh.) = "
          f"{np.max(np.abs(I_stat)) / I_inkoh.max() * 100:.2f} %"
          f"   bei x = {X[idx] * 1e6:.2f} um, y = {Y[idx] * 1e6:.2f} um")
    print(f"  Beitrag zur Gesamtleistung: {P_stat / P_inkoh * 100:+.2f} %")
    # unabhaengige Probe: explizite Formel 2 g_s g_s' cos(phi_s - phi_s')
    I_formel = sum(2 * g[grp[a]] * g[grp[b]] * np.cos(phi_s[grp[a]] - phi_s[grp[b]])
                   for grp in entartet.values()
                   for a in range(len(grp)) for b in range(a + 1, len(grp)))
    print(f"  Probe sum|H_m|^2 - sum g_s^2 gegen 2 g g' cos(dphi): "
          f"max Abweichung {np.max(np.abs(I_stat - I_formel)) / I_inkoh.max():.1e}")

# =============================================================================
# 5. Beat-Ordnungen
# =============================================================================
titel("4. Beat-Ordnungen d = k_s - k_s'")

flaechen_namen = {
    "rechteck": f"Spot-Rechteck {2 * h_x * 1e6:.2f} x {2 * h_y * 1e6:.2f} um",
    "plateau": "Plateau <I>_t > 0.5 max",
    "kreis": f"Kreis r = {kreis_radius * 1e6:.2f} um",
    "atom": f"Atom-Gewicht sigma = {sigma_atom * 1e9:.0f} nm (T = {T_atom * 1e6:.0f} uK)",
}
flaeche = flaechen_namen[REGION]
print(f"Auswertebereich: {flaeche}\n")

paare = {}                              # d -> Liste (s, s') mit k_s - k_s' = d >= 0
for a in range(len(spots)):
    for b in range(a + 1, len(spots)):
        s1, s2 = spots[a], spots[b]
        if k_s[s1] < k_s[s2]:
            s1, s2 = s2, s1
        paare.setdefault(k_s[s1] - k_s[s2], []).append((s1, s2))

print(f"d = {step_x} Delta_i + {step_y} Delta_j\n")


def gewichte(art, D0):
    """Normiertes Gewicht w(r) des Auswertebereichs, sum w = 1.

    Maske und Gewicht sind dasselbe Konzept: eine harte Region ist ein Gewicht
    aus Nullen und Einsen. So laesst sich auch das Atom-Gewicht einsetzen."""
    if art == "plateau":                       # phasenabhaengige Maske!
        w = (D0 > 0.5 * D0.max()).astype(float)
    elif art == "rechteck":                    # zwischen den aeussersten Spotzentren
        w = ((np.abs(X - x_mitte) <= h_x + 1e-15) &
             (np.abs(Y - y_mitte) <= h_y + 1e-15)).astype(float)
    elif art == "kreis":
        w = (((X - x_mitte) ** 2 + (Y - y_mitte) ** 2) <= kreis_radius ** 2).astype(float)
    elif art == "atom":
        w = np.exp(-((X - x_mitte) ** 2 + (Y - y_mitte) ** 2) / (2 * sigma_atom ** 2))
    else:
        raise ValueError(art)
    return w / w.sum()


def sigma_d_rel(D0, Dd, w):
    """Gewichtetes rms von sqrt(2|D_d|^2)/D_0 - der Beitrag der Ordnung d."""
    return math.sqrt(np.sum(w * 2 * np.abs(Dd) ** 2 / D0 ** 2))


masken = {name: gewichte(REGION, erg[name][2]) for name in erg}
kopf = (f"{'d':>3} {'f_beat[kHz]':>12} {'Paare':>6}  {'(Di,Dj)':<18} "
        f"{'Abstand[um]':<13}" + "".join(f"{n:>11}" for n in erg))
print(kopf + "\n" + "-" * len(kopf))
latex = []
n_paare = 0
for d in range(K + 1):
    ps = paare.get(d, [])
    n_paare += len(ps)
    versatz = sorted({(s1[0] - s2[0], s1[1] - s2[1]) for s1, s2 in ps},
                     key=lambda v: math.hypot(v[0] * p_x, v[1] * p_y))
    abst = [math.hypot(di * p_x, dj * p_y) * 1e6 for di, dj in versatz]
    vtxt = ", ".join(f"({di},{dj})" for di, dj in versatz) or "--"
    atxt = ", ".join(f"{v:.2f}" for v in abst) or "--"
    sig = []
    for name in erg:
        _, _, D0, D = erg[name]
        sig.append(sigma_d_rel(D0, D[d], masken[name]) * 100 if (d > 0 and ps) else None)
    stxt = "".join(f"{v:>11.1f}" if v is not None else f"{'--':>11}" for v in sig)
    print(f"{d:>3} {float(d * f_0) / 1e3:>12.1f} {len(ps):>6}  {vtxt:<18} "
          f"{atxt:<13}" + stxt)
    ltx_v = ", ".join(f"$({di},{dj})$" for di, dj in versatz) or "--"
    ltx_s = " & ".join(f"{v:.1f}" if v is not None else "--" for v in sig)
    latex.append(f"{d} & {float(d * f_0) / 1e3:.1f} & {len(ps)} & {ltx_v} & "
                 f"{atxt} & {ltx_s}\\\\")
print(f"\nSumme der Paare: {n_paare}  (erwartet S(S-1)/2 = "
      f"{len(spots) * (len(spots) - 1) // 2})")

print(f"\nGesamtschwankung sigma_I/<I>_t (gewichtetes rms, {flaeche}):")
for name in erg:
    _, _, D0, D = erg[name]
    m = masken[name]
    var = sum(2 * np.abs(D[d]) ** 2 for d in D)
    total = math.sqrt(np.sum(m * var / D0 ** 2))
    summe = math.sqrt(sum(sigma_d_rel(D0, D[d], m) ** 2 for d in D))
    n_pix = int(np.count_nonzero(m))
    print(f"  {name:<10}: {total * 100:.1f} %   (Wurzel der Summe der Quadrate: "
          f"{summe * 100:.1f} %, {n_pix} Pixel im Bereich)")

# alle Bereiche nebeneinander - die Zahl haengt sichtbar davon ab
print("\nZum Vergleich, dieselbe Groesse ueber andere Bereiche:")
print(f"  {'Bereich':<42}" + "".join(f"{n:>13}" for n in erg))
for art in ("plateau", "rechteck", "kreis", "atom"):
    zeile = ""
    for name in erg:
        _, _, D0, D = erg[name]
        w = gewichte(art, D0)
        var = sum(2 * np.abs(D[d]) ** 2 for d in D)
        zeile += f"{math.sqrt(np.sum(w * var / D0 ** 2)) * 100:>13.1f}"
    print(f"  {flaechen_namen[art]:<42}" + zeile)

# =============================================================================
# 6. Probe: Parseval gegen eine echte Zeitreihe
# =============================================================================
titel("5. Probe: Fourier-Koeffizienten gegen Zeitreihe I(r,t)")

n_t = 4 * K + 7                          # mehr als die noetigen 2K+1
t = np.arange(n_t) / n_t * float(T_0)
for name in erg:
    _, H, D0, D = erg[name]
    E = np.tensordot(np.exp(2j * np.pi * float(f_0) * np.outer(t, np.arange(K + 1))),
                     H, axes=(1, 0))    # (n_t, ny, nx)
    I_t = np.abs(E) ** 2
    var = sum(2 * np.abs(D[d]) ** 2 for d in D)
    print(f"  {name:<10}: max|mean_t I - D_0|/max D_0 = "
          f"{np.max(np.abs(I_t.mean(0) - D0)) / D0.max():.1e},  "
          f"max|var_t I - sum 2|D_d|^2|/max var = "
          f"{np.max(np.abs(I_t.var(0) - var)) / var.max():.1e}")
print(f"  Grad des Polynoms K = {K}  ->  {2 * K + 1} Abtastpunkte pro Periode "
      f"reichen exakt")

# =============================================================================
# 7. Wo Phasen helfen und wo nicht
# =============================================================================
titel("6. D_1 und D_12, untere Schranke")

print(f"D_1 hat {len(paare.get(1, []))} Zeiger:")
print("   " + " + ".join(f"z_{a[0]},{a[1]} z_{b[0]},{b[1]}^*" for a, b in paare.get(1, [])))
print(f"D_{K} hat {len(paare.get(K, []))} Zeiger: "
      + ", ".join(f"z_{a[0]},{a[1]} z_{b[0]},{b[1]}^*" for a, b in paare.get(K, [])))

# Schranke fuer VOELLIG freie Paarphasen: je Ordnung bleibt mindestens
# max(0, 2 max|c_p| - sum|c_p|), c_p = g_s g_s'.  Bezug: Phasen 0.
_, _, D0_0, _ = erg["phi = 0"]
m0 = masken["phi = 0"]                      # Gewicht, siehe gewichte()
tot = np.zeros(X.shape)
for d, ps in paare.items():
    if d == 0:
        continue
    c = np.stack([np.abs(g[s1] * g[s2]) for s1, s2 in ps])
    lo = np.maximum(0.0, 2 * c.max(axis=0) - c.sum(axis=0))
    tot += 2 * lo ** 2 / D0_0 ** 2
print(f"untere Schranke (freie Paarphasen, {flaeche}): "
      f"{math.sqrt(np.sum(m0 * tot)) * 100:.1f} %")

# =============================================================================
# 8. Falle und Puls
# =============================================================================
titel("7. Linien neben den Fallenresonanzen, Puls")

linien = np.array([float(d * f_0) for d in sorted(paare) if d > 0])
for ziel, name in ((nu_r, "nu_r"), (2 * nu_r, "2 nu_r")):
    i = int(np.argmin(np.abs(linien - ziel)))
    d_near = int(round(linien[i] / float(f_0)))
    print(f"  {name:<6} = {ziel / 1e3:6.1f} kHz: naechste Linie d = {d_near} bei "
          f"{linien[i] / 1e3:.1f} kHz, Abstand {abs(linien[i] - ziel) / 1e3:.2f} kHz "
          f"({abs(linien[i] - ziel) / ziel * 100:.1f} %)")
t_pi = 1 / (2 * f_rabi)
print(f"  pi-Puls bei f_Rabi = {f_rabi / 1e6:g} MHz: t_pi = {t_pi * 1e6:.2f} us, "
      f"spektrale Breite ~ 1/t_pi = {1 / t_pi / 1e6:.1f} MHz, "
      f"Fallenperiode 1/nu_r = {1 / nu_r * 1e6:.1f} us")

# =============================================================================
# 9. LaTeX-Zeilen der Tabelle
# =============================================================================
titel("8. Tabellenzeilen fuer LaTeX")
print("% sigma_d als gewichtetes rms ueber: " + flaeche)
print("\n".join(latex))

# =============================================================================
# 10. Vergleich mit dem GUI-Code (optional)
# =============================================================================
kern = next((p for p in (Path(__file__).resolve().parent.parent / "kern",
                         Path(__file__).resolve().parent / "kern") if p.is_dir()), None)
if kern is not None:
    titel("9. Vergleich mit kern/beating_physik.py")
    sys.path.insert(0, str(kern))
    try:
        import beating_physik as phys
        cx_, cy_, fs_, *_ = phys.compute_centers_and_freqs(
            N_x, N_y, float(W_x), float(W_y), f1, f2, float(f_off))
        amp_ = phys.amp_spots_from_ratios(r_x, r_y, N_x, N_y)
        F_ = phys.build_field_stack(X, Y, cx_, cy_, amp_, waist, use_airy, AIRY_FACTOR)
        f0_ = phys.fundamental_beat_frequency(fs_)
        k_ = phys.beat_orders(fs_, f0_)
        for name, (phi_x, phi_y) in phasen.items():
            ph_ = phys.spot_phases_from_tones(phi_x, phi_y, N_x, N_y)
            mean_, var_ = phys.time_stats_exact(F_, k_, ph_)
            _, _, D0, D = erg[name]
            var = sum(2 * np.abs(D[d]) ** 2 for d in D)
            print(f"  {name:<10}: f_0 {f0_ / 1e3:.4f} kHz, "
                  f"max rel. Abweichung <I>: {np.max(np.abs(mean_ - D0)) / D0.max():.1e}, "
                  f"Var: {np.max(np.abs(var_ - var)) / var.max():.1e}")
    except Exception as exc:                      # kern fehlt oder aendert sich
        print("  Vergleich uebersprungen:", exc)


# =============================================================================
# 11. Crestfaktor, Leistungsbudget, Beugung, Zentrierungskriterium
#     (nachtraeglich ergaenzt - prueft die Zahlen des Anhangs)
# =============================================================================
titel("10. Zentrierungskriterium")

def paarsummen(phi, N):
    return [(phi[n] + phi[N - 1 - n]) % (2 * np.pi) for n in range(N)]

for name, (px, py) in phasen.items():
    sx, sy = paarsummen(px, N_x), paarsummen(py, N_y)
    okx = np.allclose(sx, sx[0], atol=1e-9) or np.allclose(np.abs(np.diff(sx)), 2*np.pi, atol=1e-9)
    oky = np.allclose(sy, sy[0], atol=1e-9)
    print(f"  {name:<10}: c_x = {np.degrees(sx)} -> {'erfuellt' if okx else 'VERLETZT'}"
          f" | c_y = {np.degrees(sy)} -> {'erfuellt' if oky else 'VERLETZT'}")

# Gradient der Momentanintensitaet am Atomort ueber eine volle Periode
xs_f = x_mitte + np.linspace(-4 * sigma_atom, 4 * sigma_atom, 41)
ys_f = y_mitte + np.linspace(-4 * sigma_atom, 4 * sigma_atom, 41)
Xf, Yf = np.meshgrid(xs_f, ys_f, indexing="ij")
gf = {s: A_s[s] * u(np.hypot(Xf - c_s[s][0], Yf - c_s[s][1])) for s in spots}
tt = np.linspace(0, float(T_0), 241, endpoint=False)
print("  max_t |dI/dx| / <I> am Atomort (nur x-Achse variiert, phi_y = Arbeitspunkt):")
for psi in (0.0, 30.0, 90.0, 180.0):
    px = np.radians([0.0, psi, 0.0])
    ph_s = {(i, j): px[i] + phi_y_wp[j] for (i, j) in spots}
    gx = []
    for t_ in tt:
        E = sum(gf[s] * np.exp(1j * (ph_s[s] - 2 * np.pi * float(f_s_exakt[s] - f_min) * t_))
                for s in spots)
        I = np.abs(E) ** 2
        gx.append(np.gradient(I, xs_f, axis=0)[20, 20] / I.mean())
    print(f"    psi = {psi:5.1f} deg -> {np.abs(gx).max() * 1e-9:9.2e} pro nm")

titel("11. Crestfaktor")

def crest(a_volt, f_tones, n=200001):
    f = np.asarray([float(x) for x in f_tones]); a = np.asarray(a_volt, complex)
    d = f[1] - f[0]
    t = np.linspace(0, 1 / d, n, endpoint=False)
    A = np.abs((a[:, None] * np.exp(1j * 2 * np.pi * (f - f.mean())[:, None] * t[None, :])).sum(0))
    return math.sqrt(2) * A.max() / math.sqrt(np.mean(A ** 2)), A.max(), math.sqrt(np.mean(A ** 2))

vx, vy = np.sqrt(a_x), np.sqrt(a_y)
print(f"  Spannungsamplituden x: {np.round(vx,7)}   y: {np.round(vy,7)}")
print(f"  sum a_n (x) = {vx.sum():.7f}   rms = {math.sqrt((vx**2).sum()):.7f}"
      f"   -> C_max = {math.sqrt(2)*vx.sum()/math.sqrt((vx**2).sum()):.7f}"
      f"   (sqrt(6) = {math.sqrt(6):.7f})")
print(f"  sum a_n (y) = {vy.sum():.7f}   rms = {math.sqrt((vy**2).sum()):.7f}"
      f"   -> C_max = {math.sqrt(2)*vy.sum()/math.sqrt((vy**2).sum()):.7f}")
print("\n  C_x fuer die x-Phasen (Zentrierung erzwingt C = C_max):")
for name, (px, _) in phasen.items():
    c, mx, rms = crest(vx * np.exp(1j * px), f_x)
    print(f"    {name:<10}: max|A| = {mx:.5f}  C_x = {c:.4f}")
print("\n  C_y als Funktion von b bei c_y = 114 deg:")
for b in (38.0, 60.0, 98.0, 114.0):
    py = np.radians([0.0, b, 114.0 - b, 114.0])
    c, mx, rms = crest(vy * np.exp(1j * py), f_y)
    print(f"    b = {b:6.1f} deg -> max|A| = {mx:.4f}   C_y = {c:.4f}")
print("\n  Rampen sind crestneutral (x-Achse):")
for delta in (0.0, 12.5, 25.0, 90.0, 180.0, 270.0):
    px = np.radians(delta) * np.arange(N_x)
    c, mx, _ = crest(vx * np.exp(1j * px), f_x)
    print(f"    delta = {delta:6.1f} deg/Ton -> C_x = {c:.6f}   (tau = "
          f"{np.radians(delta)/(2*np.pi*float(df_x))*1e6:.4f} us)")

titel("12. Leistungsbudget und Beugungseffizienz")
P1dB = 3.98                     # W, +36 dBm, Datenblatt ZHL-03-5WF+
alpha = 0.92                    # gemessen, Mittenbuehler
Ps_x, Ps_y = 1.70, 1.62         # W, Saettigungsleistung je Achse
C_x = math.sqrt(2) * vx.sum() / math.sqrt((vx ** 2).sum())
C_y = crest(vy * np.exp(1j * np.radians([0., 98., 16., 114.])), f_y)[0]
eta = lambda P, Ps: alpha * math.sin(math.pi / 2 * math.sqrt(P / Ps)) ** 2
print(f"  P_1dB = {P1dB} W = {10*math.log10(P1dB*1000):.1f} dBm")
for tag, C, Ps in (("x, zentriert", C_x, Ps_x), ("y, b=98 deg", C_y, Ps_y),
                   ("y, Rampe", 2.8265, Ps_y), ("C = sqrt(2)", math.sqrt(2), Ps_x)):
    P = P1dB / C ** 2
    print(f"    {tag:<14} C = {C:.4f}  P_avg = {P:.3f} W  eta = {eta(P, Ps):.3f}")
ex, ey = eta(P1dB / C_x ** 2, Ps_x), eta(P1dB / C_y ** 2, Ps_y)
e2 = eta(P1dB / 2, Ps_x) * eta(P1dB / 2, Ps_y)
print(f"  eta_profil = {ex:.3f} * {ey:.3f} = {ex*ey*100:.1f} %"
      f"   |  bei C = sqrt(2) beidseitig: {e2*100:.1f} %  (Faktor {e2/(ex*ey):.2f})")
print(f"  Laserleistung fuer 10 uW im Profil: {10/(ex*ey):.1f} uW ;"
      f" fuer 0.87 uW (pi-Puls): {0.87/(ex*ey):.2f} uW")

titel("13. Moden- und Ueberlappintegrale")
P_gauss = math.pi * waist ** 2 / 2
print(f"  P_1^A = 4 pi / k^2 = {P_spot:.6e} m^2 ;  P_1^G = pi w^2/2 = {P_gauss:.6e} m^2")
print(f"  Verhaeltnis Airy/Gauss = {P_spot/P_gauss:.4f}")
d_ecke = 3.305e-6
print(f"  u(3.305 um) = {float(u(np.array([d_ecke]))[0]):+.5f}"
      f"   -> O(d)/P_1 = derselbe Wert (Autokorrelation = Mode)")
print(f"  Beitrag des Eckenpaars bei phi = 0: "
      f"{2*1.1252*float(u(np.array([d_ecke]))[0])/sum(a_x[i]*a_y[j] for i,j in spots)*100:+.2f} %"
      f"   (sum a_s = {sum(a_x[i]*a_y[j] for i,j in spots):.4f})")

titel("14. U_t als Feld ueber dem Spot-Rechteck (Arbeitspunkt)")
_, _, D0w, Dw = erg["Arbeitspkt"]
mask = ((np.abs(X - x_mitte) <= h_x + 1e-15) & (np.abs(Y - y_mitte) <= h_y + 1e-15))
Ut = np.sqrt(sum(2 * np.abs(Dw[d]) ** 2 for d in Dw) / D0w ** 2)[mask]
print(f"  min {Ut.min()*100:.1f} %   Median {np.median(Ut)*100:.1f} %"
      f"   Mittel {Ut.mean()*100:.1f} %   rms {math.sqrt(np.mean(Ut**2))*100:.1f} %"
      f"   max {Ut.max()*100:.1f} %")
print(f"  lineares Mittel der Einzelordnungen, quadratisch summiert: "
      f"{math.sqrt(sum(np.mean(np.sqrt(2*np.abs(Dw[d])**2)/D0w[mask].reshape(-1) if False else (np.sqrt(2*np.abs(Dw[d])**2)/D0w)[mask])**2 for d in Dw))*100:.1f} %"
      f"   (gegen rms {math.sqrt(sum(np.mean(((np.sqrt(2*np.abs(Dw[d])**2)/D0w)[mask])**2) for d in Dw))*100:.1f} %)")
