"""
Beating-Auswertung: Oszilloskop-Messung vs. Simulation (3x4 Toene)
==================================================================

A) MESSUNG
  1. Signal + Hintergrund einlesen (Rigol-CSV, 1 Spalte, t0/tInc im Kopf)
  2. Mittlere Spannung:      U_bar = <U> - <U_BG>
  3. Spektrum (Flattop):     U_hat(f_k) = 2 |FFT((U - <U>) w)| / sum(w)
  4. Linie je Ordnung d:     U_hat_d = max U_hat in f_d +- 4 kHz,  f_d = d f0
  5. Rel. RMS-Beitrag:       U_td = U_hat_d / (sqrt(2) U_bar)
  6. Gesamt:                 U_t  = sqrt(sum_d U_td^2)

B) SIMULATION  (gleiche Physik wie kern/beating_physik.py der Beating-GUI)
  Toene      f_x(n) = offset + width n/(N_x-1),  f_y(m) analog
  Spot (n,m) Frequenz f_s = f_x(n) + f_y(m),  Ordnung k_s = (f_s - f_min)/f0
             Ort  r = (f1 fLO / f2) tan(theta_max (f - offset)/f_band)
             Feld g_s(r) = sqrt(a_s) u(r - c_s),  u = Airy 2 J1(kr)/(kr),
                  k = 3.8317/(AIRY_FACTOR w)
             Phase phi_s = phi_x(n) + phi_y(m)
  Intensitaet I(r,t) = |sum_s g_s e^{i(2 pi f_s t + phi_s)}|^2
                     = D_0(r) + sum_{d>0} 2 Re[D_d(r) e^{2 pi i d f0 t}]
             D_d(r) = sum_{k_s - k_s' = d} g_s g_s' e^{i(phi_s - phi_s')}

  (i)  lokal, RMS ueber das Spot-Quadrat R  (= Tabelle der Arbeit):
           U_td^R   = sqrt( < 2 |D_d(r)|^2 / D_0(r)^2 >_R )
  (ii) Detektor, der die ganze Fokalebene aufsammelt (= Focal-Messung):
           U_td^det = sqrt(2) |int D_d dA| / int D_0 dA
       analytisch, weil int u(r-c1) u(r-c2) dA = P_1 u(|c1-c2|)  (Airy)
       bzw. P_1 exp(-s^2/2w^2) (Gauss)  ->  U_td^det = sqrt(2) |sum g g' e^{i dphi} u(s)| / sum a

  Geometrie "eine_linse": r = f_single tan(theta), Waist frei waehlbar (WAISTS),
  weil die Spotgroesse bei Clipping an der AOD-Apertur nicht aus w_in folgt.

C) Plot + Vergleichstabelle (.txt)
"""
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import get_window
from scipy.special import j1

# ============================ EINSTELLUNGEN ============================
# ---- Messung ----
SIGNAL = r"Focal phases 3x4.csv"
HINTERGRUND = r"Focal BG 3x4.csv"
FS = 2.5e9                    # Abtastrate (aus tInc = 4e-10 s)
SUCHFENSTER = 4e3             # +- Hz um d*f0
N_ORDNUNGEN = 12

# ---- Simulation (Arbeitspunkt) ----
N_X, N_Y = 3, 4
WIDTH = 0.37e6                # Hz, Frequenzspanne je Achse
OFFSET = 100e6                # Hz, niedrigster Ton
PHASEN_X = [0, 12, 24]        # Grad, wie in der AWG
PHASEN_Y = [0, 98, 16, 114]   # Grad
R_X, R_Y = 1.0, 1.0           # Intensitaetsgewicht der aeusseren Toene (AWG: alle gleich -> 1)
# Geometrie: "teleskop" (f1/f2 + Objektiv fLO, Tabelle der Arbeit) oder "eine_linse" (Labor, f = 45 mm)
GEOMETRIE = "eine_linse"
F_SINGLE = 45e-3              # m, die eine Linse hinter dem AOD
# Spot-DURCHMESSER (1/e^2) in der Fokalebene, die verglichen werden. Waist = Durchmesser/2.
# Bei Clipping an der AOD-Apertur unbekannt -> mehrere Werte und beide Grenzprofile.
#   teleskop:   [2.08e-6]          eine_linse: z.B. [11e-6, 13e-6]
SPOT_DURCHMESSER = [12.8e-6]   # gemessen (1/e^2-Durchmesser)
# Profile: "airy" (harte Apertur, voll ausgeleuchtet) und "gauss" (gar kein Clipping).
# Ein geclippter Gaussstrahl liegt dazwischen. Beim Airy-Profil ist der Durchmesser
# der 1/e^2-Durchmesser des HAUPTMAXIMUMS (die Ringe kommen dazu): r(1/e^2) = waist.
PROFILE = ["gauss", "airy"]
WAIST = SPOT_DURCHMESSER[0] / 2   # wird in der Schleife gesetzt
AIRY = True                       # wird in der Schleife gesetzt
AIRY_FACTOR = 1.4830          # erste Nullstelle = AIRY_FACTOR * waist
F1, F2, F_LO = 75e-3, 750e-3, 52.88e-3   # Teleskop + Objektiv
THETA_MAX, F_BAND = 43e-3, 36e6          # AOD: Scanwinkel bei Bandbreite
N_GRID = 301                  # Gitterpunkte je Achse im Spot-Quadrat R

# Werte aus der Tabelle der Arbeit (nur zur Kontrolle in der Konsole)
U_TABELLE = np.array([20.0, 40.4, 51.9, 20.4, 18.8, 16.1, 6.9, 3.2, 6.2, 5.4, 0.0, 1.8])

PLOT_PDF = "beating_3x4_measurement_vs_simulation.pdf"
TABELLE_TXT = "beating_3x4_measurement_vs_simulation.txt"
# =======================================================================


# ======================================================================
# A) Messung
# ======================================================================
def lade(pfad):
    return pd.read_csv(pfad, skiprows=1, header=None, usecols=[0]).to_numpy(float)[:, 0]


def amplitudenspektrum(u, fs):
    w = get_window("flattop", len(u))            # Fenster w_n
    X = np.fft.rfft((u - u.mean()) * w)          # X_k
    return np.fft.rfftfreq(len(u), 1 / fs), 2 * np.abs(X) / w.sum()   # f_k, U_hat(f_k) in V


def messung_auswerten(f0):
    u, u_bg = lade(SIGNAL), lade(HINTERGRUND)
    U_bar = u.mean() - u_bg.mean()
    f, U_hat = amplitudenspektrum(u, FS)
    _, U_hat_bg = amplitudenspektrum(u_bg, FS)
    df = f[1]
    rows = []
    for d in range(1, N_ORDNUNGEN + 1):
        k0, dk = int(round(d * f0 / df)), int(SUCHFENSTER / df)
        k = k0 - dk + np.argmax(U_hat[k0 - dk:k0 + dk + 1])
        rows.append(dict(f=f[k], U_hat=U_hat[k], U_bg=U_hat_bg[k],
                         U_td=U_hat[k] / (np.sqrt(2) * U_bar) * 100))
    info = dict(N=len(u), df=df, U_mean=u.mean(), U_bg=u_bg.mean(), U_bar=U_bar)
    return rows, info


# ======================================================================
# B) Simulation
# ======================================================================
def tonfrequenzen(N):
    return OFFSET + WIDTH * np.arange(N) / (N - 1)


def ort(f):
    """Position in der Fokalebene, theta = theta_max (f-offset)/f_band:
    teleskop: r = (f1 fLO/f2) tan(theta),  eine_linse: r = f_single tan(theta)."""
    brennweite = F_SINGLE if GEOMETRIE == "eine_linse" else F1 * F_LO / F2
    return brennweite * np.tan(THETA_MAX * (f - OFFSET) / F_BAND)


def profil(r):
    """Feldprofil u(r), 1 im Zentrum. Airy behaelt das Vorzeichen der Ringe."""
    if not AIRY:
        return np.exp(-r ** 2 / WAIST ** 2)
    x = 3.83170597 / (AIRY_FACTOR * WAIST) * np.asarray(r, float)
    out = np.ones_like(x)
    m = x > 1e-12
    out[m] = 2 * j1(x[m]) / x[m]
    return out


def ueberlapp(s):
    """int u(r-c1) u(r-c2) dA / int u^2 dA fuer Abstand s (analytisch)."""
    return profil(s) if AIRY else np.exp(-np.asarray(s) ** 2 / (2 * WAIST ** 2))


def spots():
    fx, fy = tonfrequenzen(N_X), tonfrequenzen(N_Y)
    amp_x = np.ones(N_X); amp_x[[0, -1]] = R_X
    amp_y = np.ones(N_Y); amp_y[[0, -1]] = R_Y
    px, py = np.radians(PHASEN_X), np.radians(PHASEN_Y)
    S = []
    for n in range(N_X):
        for m in range(N_Y):
            S.append(dict(cx=ort(fx[n]), cy=ort(fy[m]), f=fx[n] + fy[m],
                          a=amp_x[n] * amp_y[m], phi=px[n] + py[m]))
    f0 = WIDTH / np.lcm(N_X - 1, N_Y - 1)
    fmin = min(s["f"] for s in S)
    for s in S:
        s["k"] = int(round((s["f"] - fmin) / f0))
    return S, f0


def simulation(S):
    """U_td^R (lokal, RMS ueber R) und U_td^det (ganze Ebene) in %, d = 1..N_ORDNUNGEN."""
    cx = np.array([s["cx"] for s in S]); cy = np.array([s["cy"] for s in S])
    X, Y = np.meshgrid(np.linspace(cx.min(), cx.max(), N_GRID),
                       np.linspace(cy.min(), cy.max(), N_GRID))     # Spot-Quadrat R
    g = [np.sqrt(s["a"]) * profil(np.hypot(X - s["cx"], Y - s["cy"])) for s in S]

    D = {d: 0j for d in range(N_ORDNUNGEN + 1)}          # lokale Karten D_d(r)
    Ddet = {d: 0j for d in range(N_ORDNUNGEN + 1)}       # Integrale int D_d dA (/P_1)
    for i, si in enumerate(S):
        for j, sj in enumerate(S):
            d = si["k"] - sj["k"]
            if d < 0 or d > N_ORDNUNGEN or (d == 0 and i != j and i > j):
                continue
            w = np.exp(1j * (si["phi"] - sj["phi"]))
            fak = 1.0 if (d == 0 and i == j) else (2.0 if d == 0 else 1.0)  # d=0: Paar i<j doppelt
            s_ij = np.hypot(si["cx"] - sj["cx"], si["cy"] - sj["cy"])
            D[d] = D[d] + fak * w * g[i] * g[j]
            Ddet[d] += fak * w * np.sqrt(si["a"] * sj["a"]) * ueberlapp(s_ij)
    D0 = np.real(D[0])
    U_R = np.array([np.sqrt(np.mean(2 * np.abs(D[d]) ** 2 / D0 ** 2)) for d in range(1, N_ORDNUNGEN + 1)])
    U_det = np.array([np.sqrt(2) * abs(Ddet[d]) / np.real(Ddet[0]) for d in range(1, N_ORDNUNGEN + 1)])
    return U_R * 100, U_det * 100


# ======================================================================
# Ausfuehren
# ======================================================================
S, F0 = spots()
SIM = []                                        # pro Profil und Durchmesser: (label, U_R, U_det)
for prof in PROFILE:
    AIRY = (prof == "airy")
    for D_spot in SPOT_DURCHMESSER:
        WAIST = D_spot / 2
        SIM.append((f"{prof}_{D_spot*1e6:g}um",) + simulation(S))
rows, info = messung_auswerten(F0)
U_meas = np.array([r["U_td"] for r in rows])
tot = lambda v: float(np.sqrt(np.sum(np.asarray(v) ** 2)))
profilname = "/".join(PROFILE) + f", Durchmesser {[f'{x*1e6:g}' for x in SPOT_DURCHMESSER]} um"

print(f"f0 = {F0/1e3:.4f} kHz, N = {info['N']}, df = {info['df']:.1f} Hz")
print(f"<U> = {info['U_mean']*1e3:.4f} mV, <U_BG> = {info['U_bg']*1e3:.4f} mV, "
      f"U_bar = {info['U_bar']*1e3:.4f} mV")
print(f"Geometrie {GEOMETRIE}: Spot-Abstand x = {abs(S[N_Y]['cx']-S[0]['cx'])*1e6:.3f} um, "
      f"y = {abs(S[1]['cy']-S[0]['cy'])*1e6:.3f} um, Profil {profilname}\n")
kopf = " d  f_d/kHz  meas/%" + "".join(f" {'R_'+w+'/%':>16} {'det_'+w+'/%':>18}" for w, _, _ in SIM)
print(kopf)
for d, r in enumerate(rows, 1):
    print(f"{d:2d} {r['f']/1e3:8.3f} {r['U_td']:7.2f}" +
          "".join(f" {UR[d-1]:16.2f} {UD[d-1]:18.2f}" for _, UR, UD in SIM))
print(f"\nU_t Messung {tot(U_meas):.1f} %")
for w, UR, UD in SIM:
    print(f"U_t Simulation {w}: rms ueber R {tot(UR):.1f} %, ganze Fokalebene {tot(UD):.1f} %")

# ------------------------- Vergleichstabelle (.txt) -------------------------
with open(TABELLE_TXT, "w", encoding="utf-8") as fh:
    fh.write("# Beating 3x4: Messung vs. Simulation\n")
    fh.write(f"# erstellt:    {datetime.now():%Y-%m-%d %H:%M}\n")
    fh.write(f"# Signal:      {SIGNAL}\n# Hintergrund: {HINTERGRUND}\n")
    fh.write(f"# Messung: fs = {FS/1e9:.2f} GSa/s, N = {info['N']}, df = {info['df']:.1f} Hz, "
             f"Fenster flattop, U_bar = {info['U_bar']*1e3:.4f} mV\n")
    fh.write(f"# Simulation: {N_X}x{N_Y}, width = {WIDTH/1e6:.3f} MHz, f0 = {F0/1e3:.4f} kHz, "
             f"{profilname}, Geometrie {GEOMETRIE}, r_x = {R_X}, r_y = {R_Y}\n")
    fh.write(f"#   Spot-Abstand x = {abs(S[N_Y]['cx']-S[0]['cx'])*1e6:.3f} um, "
             f"y = {abs(S[1]['cy']-S[0]['cy'])*1e6:.3f} um\n")
    fh.write(f"#   phi_x = {PHASEN_X} deg, phi_y = {PHASEN_Y} deg\n")
    fh.write("# U_meas  = U_hat_d / (sqrt(2) U_bar)\n")
    fh.write("# R_<w>   = sqrt(< 2|D_d|^2 / D_0^2 >_R)     lokal, RMS ueber Spot-Quadrat R\n")
    fh.write("# det_<w> = sqrt(2)|int D_d| / int D_0       Detektor ueber ganze Fokalebene\n#\n")
    fh.write(f"# {'d':>2} {'f_d/kHz':>9} {'U_hat_d/mV':>11} {'U_hat_BG/uV':>11} {'U_meas/%':>9}" +
             "".join(f" {'R_'+w+'/%':>16} {'det_'+w+'/%':>18}" for w, _, _ in SIM) + "\n")
    for d, r in enumerate(rows, 1):
        fh.write(f"  {d:2d} {r['f']/1e3:9.3f} {r['U_hat']*1e3:11.4f} {r['U_bg']*1e6:11.3f} {r['U_td']:9.2f}" +
                 "".join(f" {UR[d-1]:16.2f} {UD[d-1]:18.2f}" for _, UR, UD in SIM) + "\n")
    fh.write(f"#\n# U_t = sqrt(sum U_td^2): Messung {tot(U_meas):.2f} %\n")
    for w, UR, UD in SIM:
        fh.write(f"#   Simulation {w}: R {tot(UR):.2f} %, ganze Fokalebene {tot(UD):.2f} %\n")
print(f"Tabelle gespeichert: {TABELLE_TXT}")

# ------------------------------- Plot -------------------------------
# Balken: Messung + Detektor-Modell (ganze Fokalebene) fuer jeden Waist
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
                     "axes.formatter.use_mathtext": True, "font.size": 10, "legend.fontsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.unicode_minus": False})
cm = 1 / 2.54
d = np.arange(1, N_ORDNUNGEN + 1)
fk = d * F0 / 1e3
def plotlabel(lab):
    prof, dm = lab.split("_")
    return rf"{'Airy' if prof == 'airy' else 'Gauss'}, $2w_0={dm[:-2]}\,\mu$m"
serien = [(UD, plotlabel(lab)) for lab, _, UD in SIM]
serien.append((U_meas, "measurement"))
farben = ["#3b6ea5", "#2a78d6", "#c3c2b7", "#6b6b66", "#1baf7a", "#e87ba4"][:len(serien) - 1] + ["#eb6834"]
n = len(serien)
bw = 52 / n
fig, ax = plt.subplots(figsize=(16 * cm, 7.5 * cm))
for i, ((v, lab), c) in enumerate(zip(serien, farben)):
    ax.bar(fk + (i - (n - 1) / 2) * (bw + 0.8), v, bw, color=c, label=lab)
ax.set_ylim(0, 1.15 * max(float(np.max(v)) for v, _ in serien))
ax.set_xlim(10, fk[-1] + 50)
ax.set_xlabel(r"beat frequency $f_\mathrm{beat}=d\,f_0$ (kHz)")
ax.set_ylabel(r"$U_{t,d}$ (%)")
top = ax.secondary_xaxis("top")
top.set_xticks(fk)
top.set_xticklabels([str(k) for k in d])
top.set_xlabel(r"beat order $d$", labelpad=4)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, loc="upper right", handlelength=1.2)
fig.tight_layout()
fig.savefig(PLOT_PDF)
plt.show()
