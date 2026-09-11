"""beating_profil.py - ein Profil aus dem Beating GUI in die Rabi-Rechnung.

Verbindet die Physik des Beating GUI (kern/beating_physik.py) mit
rb85_raman.py.

Es wird NICHTS kopiert: beating_physik.py ist dieselbe Datei, aus der auch
Beating_Multitone_GUI.py rechnet. Aenderst du dort etwas, aendert sich diese
Rechnung automatisch mit. Qt wird dafuer nicht gebraucht.

Der eine Punkt, den man verstehen muss
--------------------------------------
Das Beating GUI kennt KEINE absolute Intensitaet. Sein Feldprofil ist auf 1 im
Spot-Zentrum normiert, alle Zahlen darin sind willkuerliche Einheiten. Fuer
eine Rabi-Rechnung braucht man aber W/m^2. Also muss ein Skalenfaktor gesetzt
werden, und dafuer gibt es genau zwei sinnvolle Konventionen:

  kalibrieren_auf_rabi()    "die mittlere Rabi-Frequenz in der Region soll
                             f_rabi sein" - so kalibriert man im Labor, indem
                             man den pi-Puls auf das Ensemble-Signal stellt.
  kalibrieren_auf_leistung() "durch das Bild laeuft insgesamt P Watt".

Beides steht unten. Die erste ist der Normalfall.

Der zweite Punkt: t_0
---------------------
Ein gepulster Trieb sieht nicht das Zeitmittel, sondern das Integral ueber die
Pulsdauer AB DEM STARTZEITPUNKT t_0. Bei Tonphasen 0 rephasieren alle Toene
einmal pro Grundperiode zu einem Kamm - startet der Puls dort, ist die
Pulsflaeche ein Vielfaches des Normalwerts. Deshalb ist t_0 ein Parameter und
kein Detail; flaeche_ueber_t0() zeigt die Abhaengigkeit.

Aufruf
------
    python beating_profil.py

Als Modul:
    import beating_profil as BP
    prof = BP.profil()                       # Arbeitspunkt aus WP
    t, I  = BP.intensitaet(prof, t0=6e-6, t_p=8e-6)
    P     = BP.rabi_kurve(prof, t0=6e-6, t_p=8e-6)
"""

import numpy as np

import beating_physik as phys       # liegt hier in kern/, neben dieser Datei
import rb85_raman as R

# np.trapezoid heisst erst ab numpy 2.0 so; davor np.trapz. Euer PyCharm laeuft
# auf Python 3.9, wo beides vorkommen kann - deshalb hier einmal festlegen.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# ---- Arbeitspunkt: identisch mit den Startwerten des Beating GUI ----
WP = dict(
    N_x=3, N_y=4,
    use_airy=True, airy_factor=phys.AIRY_FACTOR,
    waist=1.04e-6,                 # m, Waist nach den Linsen
    width_x=0.37e6, width_y=0.37e6,  # Hz
    r_x=0.97, r_y=1.16,
    offset=100e6, f1=75e-3, f2=750e-3, fLO=52.88e-3,
    theta_max=43e-3, f_band=36e6,
    grid_n=200,
    phase_x=None, phase_y=None,    # None = alle Tonphasen 0
    radius=1.0e-6,                 # m, Auswertekreis um die Mitte
    max_pixel=800,                 # Pixel, auf die ausgeduennt wird
    # Raman
    delta=+50.0e9,                 # Hz von der D1-Zentroide, + = blau
    f_rabi=1.0e6,                  # Hz, Kalibrierung der mittleren Rabifrequenz
    m=0, B_gauss=0.0,
)


def profil(wp=WP):
    """Feldstack, Spot-Frequenzen, Grundperiode und Auswertemaske.

    fLO, theta_max und f_band gehen als Argumente an
    compute_centers_and_freqs(); die Modulkonstanten in beating_physik.py
    bleiben unberuehrt."""
    cx, cy, f_spots, rcx, rcy, fx, fy = phys.compute_centers_and_freqs(
        wp["N_x"], wp["N_y"], wp["width_x"], wp["width_y"],
        wp["f1"], wp["f2"], wp["offset"],
        fLO_=wp["fLO"], theta_max_=wp["theta_max"], f_band_=wp["f_band"])
    amp = phys.amp_spots_from_ratios(wp["r_x"], wp["r_y"], wp["N_x"], wp["N_y"])
    win_eff = wp["waist"] * (wp["airy_factor"] if wp["use_airy"] else 1.0)
    x, y, X, Y = phys.compute_grid(cx, cy, win_eff, wp["grid_n"])
    F = phys.build_field_stack(X, Y, cx, cy, amp, wp["waist"],
                               wp["use_airy"], wp["airy_factor"])
    px = np.zeros(wp["N_x"]) if wp["phase_x"] is None else np.asarray(wp["phase_x"])
    py = np.zeros(wp["N_y"]) if wp["phase_y"] is None else np.asarray(wp["phase_y"])
    phases = phys.spot_phases_from_tones(px, py, wp["N_x"], wp["N_y"])
    f0 = phys.fundamental_beat_frequency(f_spots)
    # Ein einziger Ton (1x1) hat keine Differenzfrequenz: f0 = 0, kein Beating,
    # keine Grundperiode. Damit alles Weitere trotzdem eine Zeitskala hat, wird
    # eine Referenzperiode gesetzt - sie beeinflusst nichts ausser der Laenge
    # der Mittelungsfenster, denn I(t) ist dann konstant.
    beats = phys.unique_beat_frequencies(f_spots)
    periodisch = f0 > 0 and beats.size > 0
    T0 = 1.0 / f0 if periodisch else 10e-6
    maske = ((X - rcx) ** 2 + (Y - rcy) ** 2) <= wp["radius"] ** 2
    idx = np.flatnonzero(maske.ravel())
    if idx.size > wp["max_pixel"]:
        idx = idx[np.linspace(0, idx.size - 1, wp["max_pixel"]).astype(int)]
    return dict(wp=wp, X=X, Y=Y, x=x, y=y, F=F, amp=amp, f_spots=f_spots, phases=phases,
                f0=f0, T0=T0, periodisch=periodisch, maske=maske, idx=idx,
                centers_x=cx, centers_y=cy, rcx=rcx, rcy=rcy, beats=beats,
                degen=phys.degenerate_groups(f_spots, f0) if periodisch else [])


def _I_roh(prof, t, idx=None):
    """I(t) je Pixel in WILLKUERLICHEN Einheiten - die kohaerente Feldsumme
    des GUI, formelgleich mit dessen intensity_cube()."""
    idx = prof["idx"] if idx is None else idx
    G = prof["F"].reshape(prof["F"].shape[0], -1)[:, idx]
    ph = 2 * np.pi * np.outer(t, prof["f_spots"]) + prof["phases"]
    return (np.cos(ph) @ G) ** 2 + (np.sin(ph) @ G) ** 2


def kalibrieren_auf_rabi(prof, raman, f_rabi=None):
    """Skalenfaktor [W/m^2 je Profileinheit], so dass die ueber Zeit UND Ort
    gemittelte Rabi-Frequenz f_rabi betraegt."""
    wp = prof["wp"]
    f_rabi = wp["f_rabi"] if f_rabi is None else f_rabi
    t = np.linspace(0, prof["T0"], 4001)
    return raman.intensity_for_rabi(f_rabi, wp["m"]) / _I_roh(prof, t).mean()


def kalibrieren_auf_leistung(prof, P_gesamt):
    """Skalenfaktor, so dass durch das ganze Bild P_gesamt Watt laufen.

    Das Flaechenintegral des Zeitmittels kommt ANALYTISCH aus
    profile_total_power(), inklusive des statischen Kreuzterms
    frequenzentarteter Spots. Frueher wurde ueber das Rechengitter summiert;
    das schneidet die Airy-Ringe ab, es fehlten rund 5 % der Leistung und die
    Intensitaet kam entsprechend zu hoch heraus."""
    wp = prof["wp"]
    P_sim = phys.profile_total_power(prof["amp"], wp["waist"], wp["use_airy"],
                                     wp["airy_factor"], prof["centers_x"],
                                     prof["centers_y"], prof["phases"],
                                     prof["degen"])
    return P_gesamt / P_sim


def intensitaet(prof, t0, t_p, n_t=4001, skala=None, raman=None):
    """t und I(t) [W/m^2] fuer jeden Pixel der Region, ab dem Pulsstart t0."""
    raman = raman or R.RamanRb85(prof["wp"]["delta"])
    skala = kalibrieren_auf_rabi(prof, raman) if skala is None else skala
    t = np.linspace(t0, t0 + t_p, n_t)
    return t, _I_roh(prof, t) * skala


def rabi_kurve(prof, t0, t_p, n_t=4001, skala=None, raman=None, **kw):
    """Anregung ueber der Pulslaenge, gemittelt ueber die Auswerteregion.

    Jeder Pixel wird einzeln propagiert - die Intensitaet ist dort verschieden,
    also auch die Rabi-Frequenz, und genau diese Streuung ist es, die den
    Kontrast frisst. Ein Mittelwert VOR der Propagation wuerde sie verschlucken."""
    wp = prof["wp"]
    raman = raman or R.RamanRb85(wp["delta"])
    t, I = intensitaet(prof, t0, t_p, n_t, skala, raman)
    opts = dict(m=wp["m"], B_gauss=wp["B_gauss"])
    opts.update(kw)
    return t - t0, R.excitation_series_multi(t, I, raman, **opts)


def phasen(art, N, seed=0):
    """Tonphasen-Vorgaben, wie im Beating GUI: 'null', 'schroeder', 'zufall'."""
    if art == "schroeder":
        return phys.schroeder_phases(N)
    if art == "zufall":
        return np.random.default_rng(seed).uniform(0, 2 * np.pi, N)
    return np.zeros(N)


def flaeche_ueber_t0(prof, t_p, n=120, skala=None, raman=None):
    """Mittlere Pulsflaeche und ihre Uniformity ueber dem Pulsstart t0.

    Das ist die Kurve, an der man sieht, ob ein Trigger noetig ist - und wo er
    hingehoert."""
    wp = prof["wp"]
    raman = raman or R.RamanRb85(wp["delta"])
    skala = kalibrieren_auf_rabi(prof, raman) if skala is None else skala
    C = raman.coeff(wp["m"])["C_rabi"]
    t0s = np.linspace(0.0, prof["T0"], n, endpoint=False)
    flaeche, unif = [], []
    for t0 in t0s:
        tt = np.linspace(t0, t0 + t_p, 1200)
        th = C * _trapz(_I_roh(prof, tt) * skala, tt, axis=0)
        flaeche.append(th.mean())
        unif.append(th.std() / th.mean())
    return t0s, np.array(flaeche), np.array(unif)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prof = profil()
    raman = R.RamanRb85(WP["delta"])
    t_pi = 1.0 / (2 * WP["f_rabi"])
    print(f"Grundperiode T_0 = {prof['T0'] * 1e6:.3f} us  "
          f"(f_0 = {prof['f0'] / 1e3:.2f} kHz)")
    print(f"Pixel in der Region: {prof['maske'].sum()}, benutzt {prof['idx'].size}")
    print(f"eta = {raman.eta():+.4f}, Kontrastdeckel "
          f"{raman.contrast_cap() * 100:.2f} %")
    skala = kalibrieren_auf_rabi(prof, raman)
    print(f"Kalibrierung: <I> = "
          f"{_I_roh(prof, np.linspace(0, prof['T0'], 2001)).mean() * skala / 1e4:.4f} W/cm2")

    # wo sitzt ein guter Trigger?
    t0s, A, U = flaeche_ueber_t0(prof, t_pi, skala=skala, raman=raman)
    med = np.median(A)
    ok = np.flatnonzero(np.abs(A - med) < 0.25 * med)
    i = int(ok[np.argmin(U[ok])])
    print(f"Pulsflaeche schwankt ueber t_0 um Faktor {A.max() / A.min():.1f}")
    print(f"bester Start t_0 = {t0s[i] * 1e6:.3f} us, U(Flaeche) = {U[i] * 100:.2f} %")

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for t0, lab, col in [(0.0, "$t_0=0$ (Kamm)", "#b3452c"),
                         (t0s[i], f"$t_0={t0s[i]*1e6:.1f}\\,\\mu$s", "#2f6f4e")]:
        tp, P = rabi_kurve(prof, t0, 2.8 * t_pi, skala=skala, raman=raman)
        ax[0].plot(tp * 1e6, P * 100, lw=1.7, color=col,
                   label=f"{lab}  (max {P.max()*100:.0f} %)")
    ax[0].axhline(raman.contrast_cap() * 100, color="#bbb", lw=0.9)
    ax[0].set_xlabel("Pulslänge $t_p$ ($\\mu$s)")
    ax[0].set_ylabel("Anregung (%)")
    ax[0].set_ylim(0, 100); ax[0].legend(fontsize=8)
    ax[0].set_title("Rabi im Multiton-Profil", fontsize=9.5)

    ax[1].plot(t0s * 1e6, A / A.mean(), color="#1b4f8a", lw=1.5)
    ax[1].set_yscale("log")
    ax[1].axvline(t0s[i] * 1e6, color="#2f6f4e", lw=1.0, ls="--")
    ax[1].set_xlabel("Pulsstart $t_0$ ($\\mu$s)")
    ax[1].set_ylabel(r"$\bar\theta(t_0)/\langle\bar\theta\rangle$")
    ax[1].set_title(f"Pulsfläche über dem Trigger (×{A.max()/A.min():.0f})",
                    fontsize=9.5)
    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig("beating_profil.pdf")
    print("geschrieben: beating_profil.pdf")
