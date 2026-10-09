"""c_rabi_aufschluesselung.py - C_rabi Schritt fuer Schritt, mit allen ARC-Eingaben.

Rechnet die Zwei-Photonen-Kopplung

    C_rabi = kappa * sqrt(beta (1-beta)) * | sum_e d_2e d_3e / (2 Delta_e) |,
    kappa  = 2 / (eps0 c hbar^2),
    Delta_e = 2 pi [ Delta + (nu_D1 - nu_J') - Delta_hfs(J', F') ]

explizit aus und zeigt fuer jeden Zwischenzustand e = |5P_J', F', m+1>, woher
die Zahlen kommen:

    Dipolmatrixelemente d_2e, d_3e  ARC  getDipoleMatrixElementHFS
    Hyperfeinverschiebung            ARC  getHFSCoefficients / getHFSEnergyShift
    Linienschwerpunkte nu_J'         ARC  getTransitionFrequency
    Linienbreiten Gamma              ARC  getStateLifetime

Nur die adiabatische Elimination selbst ist von Hand programmiert, in
RamanRb85.coeff() in kern/rb85_raman.py. Am Ende wird das Ergebnis gegen
coeff() geprueft und eine LaTeX-Tabelle ausgegeben.

    python c_rabi_aufschluesselung.py            # Delta = +50 GHz
    python c_rabi_aufschluesselung.py -8         # Delta = -8 GHz
"""

import sys
from pathlib import Path as _Pfad

# Dieses Beispiel liegt in beispiele/; die Module liegen eine Ebene darueber
# bzw. in kern/.
_WURZEL = _Pfad(__file__).resolve().parent.parent
sys.path[:0] = [str(_WURZEL), str(_WURZEL / "kern")]

import numpy as np
from scipy.constants import hbar, epsilon_0, c, physical_constants

import rb85_raman as R

# ----------------------------------------------------------------- Eingaben
DELTA_GHZ = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0   # vom D1-Schwerpunkt
BETA = 0.5            # Intensitaetsanteil des Beins an F=2
M = 0                 # Uhrenuebergang m_F = 0
Q = 1                 # sigma+  ->  m' = m + 1
F_RABI = 1.0e6        # Hz, nur fuer die Umrechnung in eine Intensitaet

ea0 = physical_constants["Bohr radius"][0] * physical_constants["elementary charge"][0]
atom = R.ATOM                                    # arc.Rubidium85()
kappa = 2.0 / (epsilon_0 * c * hbar ** 2)        # Omega^2 = d^2 * kappa * I

# ------------------------------------------------- 1) Atomdaten aus ARC
print("=" * 78)
print("1) Atomdaten aus ARC (Rb-85)")
print("=" * 78)
nu_D1 = atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 0.5)     # Hz, Schwerpunkt
nu_D2 = atom.getTransitionFrequency(5, 0, 0.5, 5, 1, 1.5)
nu_J = {0.5: nu_D1, 1.5: nu_D2}
print(f"  getTransitionFrequency : nu_D1 = {nu_D1 / 1e12:.6f} THz "
      f"({c / nu_D1 * 1e9:.4f} nm)")
print(f"                           nu_D2 = {nu_D2 / 1e12:.6f} THz "
      f"({c / nu_D2 * 1e9:.4f} nm)")
print(f"                           nu_D1 - nu_D2 = {(nu_D1 - nu_D2) / 1e9:.3f} GHz")
for j, name in ((0.5, "5P1/2"), (1.5, "5P3/2")):
    tau = atom.getStateLifetime(5, 1, j)
    print(f"  getStateLifetime       : {name}: tau = {tau * 1e9:.3f} ns, "
          f"Gamma/2pi = {1 / tau / 2 / np.pi / 1e6:.4f} MHz")
A, B = atom.getHFSCoefficients(5, 0, 0.5)[:2]
w_hfs = atom.getHFSEnergyShift(0.5, 3, A, B) - atom.getHFSEnergyShift(0.5, 2, A, B)
print(f"  getHFSEnergyShift      : 5S1/2 F=3 - F=2 = {w_hfs / 1e9:.6f} GHz")

# ---------------------------------------- 2) Summe ueber die Zwischenzustaende
print()
print("=" * 78)
print(f"2) Summe ueber e = |5P_J', F', m'={M + Q}>   (Delta = {DELTA_GHZ:+g} GHz, "
      f"beta = {BETA}, sigma+)")
print("=" * 78)
kopf = (f"  {'e':<12}{'Dhfs[MHz]':>10}{'Delta_e/2pi[GHz]':>18}"
        f"{'d_2e[ea0]':>11}{'d_3e[ea0]':>11}{'Beitrag':>12}")
print(kopf)
print("  " + "-" * (len(kopf) - 2))

zeilen = []
summe = 0.0
for J, Fps in ((0.5, (2, 3)), (1.5, (1, 2, 3, 4))):
    A, B = atom.getHFSCoefficients(5, 1, J)[:2]
    for Fp in Fps:
        d_hfs = atom.getHFSEnergyShift(J, Fp, A, B)                       # Hz
        Delta_e = 2 * np.pi * (DELTA_GHZ * 1e9 + (nu_D1 - nu_J[J]) - d_hfs)  # rad/s
        d2 = atom.getDipoleMatrixElementHFS(5, 0, 0.5, 2, M, 5, 1, J, Fp, M + Q, Q) * ea0
        d3 = atom.getDipoleMatrixElementHFS(5, 0, 0.5, 3, M, 5, 1, J, Fp, M + Q, Q) * ea0
        beitrag = kappa * np.sqrt(BETA * (1 - BETA)) * d2 * d3 / (2 * Delta_e)
        summe += beitrag
        name = f"{'D1' if J == 0.5 else 'D2'} F'={Fp}"
        zeilen.append((name, d_hfs, Delta_e, d2, d3, beitrag))
        print(f"  {name:<12}{d_hfs / 1e6:>10.3f}{Delta_e / 2 / np.pi / 1e9:>18.3f}"
              f"{d2 / ea0:>11.4f}{d3 / ea0:>11.4f}{beitrag:>12.4f}")

C_rabi = abs(summe)
d1 = sum(z[5] for z in zeilen if z[0].startswith("D1"))
d2_ = sum(z[5] for z in zeilen if z[0].startswith("D2"))
print("  " + "-" * (len(kopf) - 2))
print(f"  Summe D1 = {d1:+.4f},  Summe D2 = {d2_:+.4f}   [rad/s per W/m^2]")
print(f"\n  C_rabi = |Summe| = {C_rabi:.4f} rad/s per W/m^2")

# ------------------------------------------------------ 3) Kontrolle
print()
print("=" * 78)
print("3) Kontrolle gegen RamanRb85.coeff() und Umrechnung")
print("=" * 78)
co = R.RamanRb85(delta_Hz=DELTA_GHZ * 1e9, beta=BETA, q=Q).coeff(M)
print(f"  coeff()['C_rabi'] = {co['C_rabi']:.4f}   "
      f"(Abweichung {abs(co['C_rabi'] - C_rabi) / C_rabi:.1e})")
print(f"  eta = delta_LS / Omega = {co['eta']:+.4f}  ->  Kontrast <= "
      f"{100 / (1 + co['eta'] ** 2):.2f} %")
I_ref = 2 * np.pi * F_RABI / C_rabi
print(f"  f_Rabi = {F_RABI / 1e6:g} MHz braucht I_ref = {I_ref:.4g} W/m^2 "
      f"= {I_ref / 1e4:.2f} W/cm^2")

# ------------------------------------------------------ 4) LaTeX
print()
print("=" * 78)
print("4) Tabellenzeilen fuer LaTeX")
print("=" * 78)
for name, d_hfs, Delta_e, dd2, dd3, beitrag in zeilen:
    lbl = name.replace("D1", "D$_1$,").replace("D2", "D$_2$,").replace("F'", "$F'") + "$"
    print(f"{lbl} & ${dd2 / ea0:.3f}$ & ${dd3 / ea0:.3f}$ & "
          f"${Delta_e / 2 / np.pi / 1e9:.2f}$ & ${beitrag:.3f}$\\\\")
