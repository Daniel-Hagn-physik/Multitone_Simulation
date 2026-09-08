"""rabi_pro_atom.py - Rabi-Oszillation EINES Atoms unter EINEM 3x4-Flattop.

Der Anwendungsfall: die 3x4 Toene sind das Adressier-Profil pro Atom, nicht
ein globaler Strahl. Ein Atom sitzt unter einem solchen Flattop, und gefragt
ist seine Anregung ueber der Pulslaenge.

Eingabe ist hier die LEISTUNG im Profil (nicht eine Ziel-Rabifrequenz), denn
die ist das, was aus dem Aufbau folgt:

    P_Profil = P_vor_AOD * Wirkungsgrad / N_gleichzeitig

Der Punkt, auf den es hinauslaeuft
----------------------------------
Ob das Beating stoert, entscheidet das Verhaeltnis t_pi / T_0. Die
Grundperiode des Beatings liegt bei 13.3 us; ein pi-Puls von wenigen ns sieht
davon eine eingefrorene Momentaufnahme und mittelt NICHTS weg - die Pulsflaeche
schwankt dann von Schuss zu Schuss um den vollen Modulationshub. Je schneller
der Puls, desto schlimmer das Beating. Das ist der Grund, warum viel Leistung
hier nicht hilft, sondern schadet.

Aufruf:
    python rabi_pro_atom.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import beating_profil as BP
import rb85_raman as R

# ---- Aufbau -------------------------------------------------------
P_VOR_AOD = 0.300          # W, vor den gekreuzten AODs
WIRKUNGSGRAD = 0.50        # zwei AODs plus Optik; das ist die Rate-Zahl
N_GLEICHZEITIG = 1681      # 41x41 gleichzeitig adressiert

SZENARIEN = [
    # (Leistung je Profil [W], Delta [Hz], Beschriftung)
    (P_VOR_AOD * WIRKUNGSGRAD / N_GLEICHZEITIG, -8e9,
     "alle 1681, $\\Delta=-8$ GHz"),
    (P_VOR_AOD * WIRKUNGSGRAD / N_GLEICHZEITIG, -640e9,
     "alle 1681, $\\Delta=-640$ GHz"),
    (222e-9, -8e9,
     "auf 222 nW gedämpft, $\\Delta=-8$ GHz"),
]


def bester_t0(prof, raman, t_p, skala):
    """Pulsstart mit der gleichmaessigsten Pulsflaeche - aber nur unter den
    Startzeiten, deren Flaeche nahe am Median liegt. Ohne diese Einschraenkung
    landet die Suche auf der Flanke des Rephasierungs-Kamms, wo die Flaeche ein
    Vielfaches betraegt und der 'pi-Puls' keiner mehr ist."""
    t0s, A, U = BP.flaeche_ueber_t0(prof, t_p, n=120, skala=skala, raman=raman)
    med = np.median(A)
    ok = np.flatnonzero(np.abs(A - med) < 0.25 * med)
    i = int(ok[np.argmin(U[ok])]) if ok.size else int(np.argmin(U))
    return float(t0s[i]), float(U[i]), float(A.max() / A.min())


def main():
    prof = BP.profil()
    T0 = prof["T0"]
    skala_1W = BP.kalibrieren_auf_leistung(prof, 1.0)
    t = np.linspace(0, T0, 2001)
    A_eff = 1.0 / (BP._I_roh(prof, t).mean() * skala_1W)

    print(f"Ein 3x4-Flattop: effektive Fläche {A_eff*1e12:.2f} um^2, "
          f"Grundperiode T_0 = {T0*1e6:.2f} us")
    print(f"{P_VOR_AOD*1e3:.0f} mW vor den AODs, {WIRKUNGSGRAD*100:.0f} % "
          f"Wirkungsgrad, {N_GLEICHZEITIG} gleichzeitig "
          f"-> {P_VOR_AOD*WIRKUNGSGRAD/N_GLEICHZEITIG*1e6:.1f} uW je Profil\n")

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.6))
    farben = ["#b3452c", "#2f6f4e", "#1b4f8a"]
    for (P_prof, delta, lab), col in zip(SZENARIEN, farben):
        raman = R.RamanRb85(delta)
        skala = skala_1W * P_prof
        I_mit = BP._I_roh(prof, t).mean() * skala
        Om = raman.coeff(0)["C_rabi"] * I_mit
        t_pi = np.pi / (Om * np.sqrt(1 + raman.eta() ** 2))
        t0, U, schwank = bester_t0(prof, raman, t_pi, skala)
        tp, P = BP.rabi_kurve(prof, t0, 2.6 * t_pi, n_t=6001,
                              skala=skala, raman=raman)
        ax[0].plot(tp / t_pi, P * 100, lw=1.7, color=col,
                   label=f"{lab}\n  $t_\\pi$ = {t_pi*1e9:.3g} ns, max {P.max()*100:.0f} %")
        print(f"{lab}")
        print(f"   P = {P_prof*1e6:9.4g} uW  ->  <I> = {I_mit/1e4:9.4g} W/cm^2")
        print(f"   Omega/2pi = {Om/2/np.pi/1e6:.4g} MHz,  t_pi = {t_pi*1e9:.4g} ns")
        print(f"   t_pi/T_0 = {t_pi/T0:.2e}  ({'Beating eingefroren' if t_pi/T0 < 0.02 else 'Puls mittelt teilweise'})")
        print(f"   eta = {raman.eta():+.4f}, Deckel {raman.contrast_cap()*100:.2f} %, "
              f"erreicht {P.max()*100:.2f} %")
        print(f"   bester t_0 = {t0*1e6:.3f} us, U(Fläche) = {U*100:.2f} %, "
              f"Schwankung über t_0 = x{schwank:.0f}\n")

    ax[0].set_xlabel("Pulslänge $t_p / t_\\pi$")
    ax[0].set_ylabel("Anregung (%)")
    ax[0].set_ylim(0, 100)
    ax[0].legend(fontsize=6.5, loc="lower right")
    ax[0].set_title("Ein Atom unter einem 3×4-Flattop", fontsize=9.5)

    # rechts: warum die Pulsdauer alles entscheidet
    tps = np.logspace(-9, -4.5, 60)
    ax[1].loglog(tps * 1e9, tps / T0, color="#333", lw=1.5)
    ax[1].axhline(1.0, color="#bbb", lw=0.9)
    ax[1].text(1.5, 1.3, "Puls = eine Beat-Periode", fontsize=6.5, color="#888")
    for (P_prof, delta, lab), col in zip(SZENARIEN, farben):
        raman = R.RamanRb85(delta)
        I_mit = BP._I_roh(prof, t).mean() * skala_1W * P_prof
        Om = raman.coeff(0)["C_rabi"] * I_mit
        t_pi = np.pi / (Om * np.sqrt(1 + raman.eta() ** 2))
        ax[1].plot(t_pi * 1e9, t_pi / T0, "o", color=col, ms=6)
    ax[1].set_xlabel("$t_\\pi$ (ns)")
    ax[1].set_ylabel("$t_\\pi / T_0$")
    ax[1].set_title("Mittelt der Puls das Beating weg?", fontsize=9.5)
    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("rabi_pro_atom.pdf")
    print("geschrieben: rabi_pro_atom.pdf")


if __name__ == "__main__":
    main()
