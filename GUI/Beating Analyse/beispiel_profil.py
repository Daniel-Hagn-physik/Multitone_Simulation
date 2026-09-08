"""beispiel_profil.py - eigenes Intensitätsprofil statt Gaußstrahl.

Das GUI rechnet mit einem konstanten I. Sobald I zeitabhängig oder
ortsabhängig ist, ist der Einstiegspunkt rb85_raman.excitation_series():
sie nimmt eine beliebige Intensitätsgeschichte I(t) entgegen und gibt die
Anregung nach jedem Zeitschritt zurück - also die ganze Rabi-Kurve in einem
Durchgang.

Hier zwei Beispiele: ein moduliertes I(t), und eine Mittelung über mehrere
Orte mit unterschiedlicher Intensität (so würde man ein Ensemble in einem
strukturierten Profil rechnen).

    python beispiel_profil.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rb85_raman as R

raman = R.RamanRb85(delta_Hz=-8e9)
I0 = raman.intensity_for_rabi(0.2e6)          # Intensität für Omega/2pi = 200 kHz
t = np.linspace(0, 8e-6, 6001)

# --- 1) konstantes I: das, was das GUI zeigt --------------------------
P_const = R.excitation_series(t, np.full_like(t, I0), raman)

# --- 2) moduliertes I(t): z.B. eine Schwebung bei 75 kHz --------------
I_mod = I0 * (1 + 0.9 * np.cos(2 * np.pi * 75e3 * t))
P_mod = R.excitation_series(t, I_mod, raman)

# --- 3) Ensemble: 200 Orte mit 15 % Intensitätsstreuung ---------------
rng = np.random.default_rng(0)
scales = 1.0 + 0.15 * rng.standard_normal(200)
P_ens = np.mean([R.excitation_series(t, np.full_like(t, I0 * s), raman)
                 for s in scales], axis=0)

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(t * 1e6, P_const * 100, lw=1.8, color="#2f6f4e", label="konstantes $I$")
ax.plot(t * 1e6, P_mod * 100, lw=1.4, color="#b3452c",
        label="$I(t)$ moduliert, 75 kHz")
ax.plot(t * 1e6, P_ens * 100, lw=1.4, ls="--", color="#1b4f8a",
        label="Ensemble, 15 % Intensitätsstreuung")
ax.set_xlabel("Pulslänge $t_p$ ($\\mu$s)")
ax.set_ylabel("Anregung (%)")
ax.set_ylim(0, 100)
ax.legend(fontsize=7.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("beispiel_profil.pdf")
print("geschrieben: beispiel_profil.pdf")
print(f"  konstant : P_max = {P_const.max() * 100:.2f} %")
print(f"  moduliert: P_max = {P_mod.max() * 100:.2f} %")
print(f"  Ensemble : P_max = {P_ens.max() * 100:.2f} %")
