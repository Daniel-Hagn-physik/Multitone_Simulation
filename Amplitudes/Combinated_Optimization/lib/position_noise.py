"""
lib/position_noise.py - Wie weit wandert das Atom gegen das Raman-Profil?

Abschaetzung der Positionsschwankung eines Atoms RELATIV zum Multitone-
Profil, wenn Laborgroessen schwanken. Ergebnis ist eine Laenge sigma_pos
(1 sigma je Achse), bis zu der run_atom_offset.py den zweiten Plot faehrt
und ueber die es die Statistik der Metriken bildet.

Zwei Arten von Beitraegen
-------------------------
1. THERMISCH - das Atom sitzt nicht ruhig im Fallenminimum. Im harmonischen
   Potential ist seine Ortsverteilung je Achse gaussfoermig mit

       sigma^2 = hbar/(2 m omega) * coth(hbar omega / (2 kB T))

   (dieselbe Formel wie sigma_thermal() im Optimierer, dort steht sie als
   Gewicht W in U_w und eta_w). Schwanken T oder nu_r, schwankt sigma; in
   die Abschaetzung geht der UNGUENSTIGE Rand ein (T + dT, nu_r - dnu).

2. TECHNISCH - die Falle und das Raman-Profil stehen nicht exakt
   uebereinander. Jeder Mechanismus wird ueber seinen geometrischen
   Hebelarm in eine Verschiebung in der Atomebene umgerechnet. Die
   Hebelarme folgen aus derselben Optik, mit der der Optimierer die
   Spot-Zentren legt:

       r = (f1 fLO / f2) * tan(theta),   theta = theta_max (f - offset) / f_band

   - Strahllage VOR dem AOD, Winkel dtheta:
         dx = (f1 fLO / f2) * dtheta           (Teleskop verkleinert 10x)
   - Strahllage HINTER dem Teleskop (vor dem Objektiv), Winkel dtheta:
         dx = fLO * dtheta
   - Schallgeschwindigkeit im AOD, relativ dv/v (Temperatur!):
         der Beugungswinkel ist theta_abs = lambda f / v, er skaliert mit
         1/v. Die Site-Mitte liegt bei f_c = offset + width/2, also
         dx = (f1 fLO / f2) * (theta_max / f_band) * f_c * dv/v
   - RF-Frequenzfehler df:
         dx = (f1 fLO / f2) * (theta_max / f_band) * df
   - Fallenposition selbst (Tweezer-/Mikrolinsen-Strahllage), direkt in nm
   - Pitch-Unsicherheit dp an der Site n Pitches vom Justierpunkt:
         dx = n * dp
   - eine GEMESSENE Relativposition (alles zusammen), direkt in nm

   Die Beitraege gelten als unabhaengig und werden quadratisch addiert.

Was dabei NICHT als Positionsschwankung zaehlt
----------------------------------------------
- Der Durchhang durch die Schwerkraft, g / omega^2, falls die Falle quer
  zur Schwerkraft steht: bei nu_r = 60 kHz sind das 0.07 nm, und eine
  Leistungsschwankung aendert daran nur einen Bruchteil davon. Er steht
  zur Einordnung im Bericht, geht aber nicht in die Summe ein.
- Eine Leistungsschwankung der Falle verschiebt das Atom nicht, sie aendert
  nur omega und damit sigma (sigma ~ P^(-1/2) im klassischen Grenzfall,
  bei fester Temperatur).
  Auch das steht nur zur Einordnung da.

Die Default-Werte
-----------------
Gemessen (Projekt-Defaults des Optimierers): T = 17(2) uK,
nu_r = 60.4(+2.4/-2.2) kHz, pitch = 5.16(3) um.

Fuer die technischen Mechanismen sind keine Labormesswerte bekannt. Sie
stehen deshalb auf 0 - bis auf EINEN Gesamtwert fuer die Relativposition:
47 nm, die Positionsstreuung je Site, die Seubert et al. (arXiv:2502.13560)
fuer einzelne Atome in AOD-positionierten Tweezern gemessen haben (inklusive
Detektionsfehler, also eine obere Schranke). Das ist eine Groessenordnung
aus einem anderen Aufbau, kein Wert dieses Experiments. Wer die
Einzelmechanismen eintraegt, setzt diesen Gesamtwert auf 0, sonst wird
doppelt gezaehlt.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import hbar, k as kB, g as G_ERDE

from . import paths  # noqa: F401  (setzt sys.path auf Weighted_Optimization)

from weighted_multitone_flattop_optimizer import (  # noqa: E402
    MultitoneFlatTopOptimizer, RB85_MASS, sigma_thermal,
)

_OPT_DEFAULTS = MultitoneFlatTopOptimizer.DEFAULTS

# Optik, fest wie in allen Combined-Skripten.
F1_FIXED = 75e-3
F2_FIXED = 750e-3


# ======================================================================
# Eingaben
# ======================================================================
# (Schluessel, Anzeigetext, Einheit, Default, Tooltip/Quelle)
EINGABEN_THERMISCH = [
    ("T_uK", "Atomtemperatur T", "µK", 17.0,
     "gemessen, Projekt-Default des Optimierers"),
    ("dT_uK", "Schwankung von T", "µK", 2.0,
     "Messunsicherheit 17(2) µK"),
    ("nu_kHz", "radiale Fallenfrequenz nu_r", "kHz", 60.4,
     "gemessen, Projekt-Default des Optimierers"),
    ("dnu_kHz", "Schwankung von nu_r (nach unten)", "kHz", 2.2,
     "Messunsicherheit 60.4(+2.4/-2.2) kHz; kleineres nu_r = breitere Verteilung"),
    ("dP_prozent", "Leistungsschwankung der Falle dP/P", "%", 1.0,
     "nur zur Einordnung (aendert sigma, verschiebt das Atom nicht)"),
]

EINGABEN_TECHNISCH = [
    ("relativ_nm", "Relativposition Falle - Raman, gemessen (gesamt)", "nm", 47.0,
     "Groessenordnung: 47(5) nm Positionsstreuung je Site fuer Atome in "
     "AOD-Tweezern, Seubert et al., arXiv:2502.13560 (inkl. Detektionsfehler, "
     "obere Schranke; anderer Aufbau). Auf 0 setzen, wenn die Einzelbeitraege "
     "darunter eingetragen werden."),
    ("theta_vor_aod_urad", "Raman-Strahllage VOR dem AOD", "µrad", 0.0,
     "Winkeljitter des Strahls, der in den AOD laeuft"),
    ("theta_vor_obj_urad", "Raman-Strahllage VOR dem Objektiv", "µrad", 0.0,
     "Winkeljitter hinter dem Teleskop - der groesste Hebel (fLO)"),
    ("dv_ppm", "AOD-Schallgeschwindigkeit dv/v", "ppm", 0.0,
     "= Temperaturkoeffizient der Schallgeschwindigkeit x Temperaturschwankung "
     "des Kristalls (Datenblatt/Messung)"),
    ("df_Hz", "RF-Frequenzfehler df", "Hz", 0.0,
     "Frequenzfehler des RF-Generators"),
    ("falle_nm", "Fallenposition (Tweezer-Strahllage)", "nm", 0.0,
     "Schwankung der Fallenposition in der Atomebene"),
    ("dp_nm", "Pitch-Unsicherheit dp", "nm", 30.0,
     "Messung pitch = 5.16(3) µm; wirkt n-fach an der Site n"),
    ("n_site", "Site-Abstand vom Justierpunkt n", "Pitches", 0.0,
     "0 = die Site, auf die justiert wurde"),
]

DEFAULTS = {k: d for k, _t, _e, d, _q in EINGABEN_THERMISCH + EINGABEN_TECHNISCH}

# Welche Groesse spannt den zweiten Plot und die Statistik auf.
UMFANG_CHOICES = [
    ("gesamt", "thermisch und technisch (quadratisch addiert)"),
    ("technisch", "nur technisch (thermisch steckt schon in U_w, eta_w)"),
    ("thermisch", "nur thermisch"),
]


@dataclass
class Beitrag:
    key: str
    name: str
    eingabe: str
    hebel: str
    wert_m: float
    art: str               # "thermisch" | "technisch" | "info"
    quelle: str = ""


@dataclass
class Abschaetzung:
    beitraege: list = field(default_factory=list)
    sigma_therm_nom_m: float = np.nan
    sigma_therm_max_m: float = np.nan
    sigma_tech_m: float = np.nan
    sigma_gesamt_m: float = np.nan
    nullpunkt_m: float = np.nan
    n_mittel: float = np.nan
    eingaben: dict = field(default_factory=dict)

    def sigma(self, umfang="gesamt"):
        return {"gesamt": self.sigma_gesamt_m,
                "technisch": self.sigma_tech_m,
                "thermisch": self.sigma_therm_max_m}[umfang]

    def als_dict(self):
        return dict(
            beitraege=[b.__dict__.copy() for b in self.beitraege],
            sigma_therm_nom_m=self.sigma_therm_nom_m,
            sigma_therm_max_m=self.sigma_therm_max_m,
            sigma_tech_m=self.sigma_tech_m,
            sigma_gesamt_m=self.sigma_gesamt_m,
            nullpunkt_m=self.nullpunkt_m,
            n_mittel=self.n_mittel,
            eingaben=dict(self.eingaben),
        )


# ======================================================================
# Hebelarme
# ======================================================================
def hebel(width=0.45e6, offset=None, f1=F1_FIXED, f2=F2_FIXED, fLO=None,
          theta_max=None, f_band=None):
    """Geometrische Hebelarme der Optik (alle in SI).

    Rueckgabe: dict mit
      m_pro_rad_vor_aod   Verschiebung pro Winkel vor dem AOD
      m_pro_rad_vor_obj   Verschiebung pro Winkel vor dem Objektiv
      m_pro_Hz            Verschiebung pro RF-Frequenzfehler
      m_pro_dv_rel        Verschiebung pro relativer Schallgeschwindigkeit
      f_c                 Mittenfrequenz der Site
    """
    offset = _OPT_DEFAULTS["offset"] if offset is None else offset
    fLO = _OPT_DEFAULTS["fLO"] if fLO is None else fLO
    theta_max = _OPT_DEFAULTS["theta_max"] if theta_max is None else theta_max
    f_band = _OPT_DEFAULTS["f_band"] if f_band is None else f_band

    relay = f1 * fLO / f2                       # r = relay * tan(theta)
    rad_pro_Hz = theta_max / f_band             # = lambda / v
    f_c = offset + 0.5 * width
    return dict(
        relay_m=relay,
        m_pro_rad_vor_aod=relay,
        m_pro_rad_vor_obj=fLO,
        m_pro_Hz=relay * rad_pro_Hz,
        m_pro_dv_rel=relay * np.tan(rad_pro_Hz * f_c),
        f_c=f_c,
        v_akustisch=_OPT_DEFAULTS["lambda_opt"] / rad_pro_Hz,
    )


# ======================================================================
# Die Abschaetzung
# ======================================================================
def abschaetzen(eingaben=None, width=0.45e6, atom_mass=RB85_MASS):
    """Alle Beitraege plus Summen. `eingaben` in den Einheiten der
    EINGABEN_*-Tabellen (µK, kHz, µrad, nm, ppm, Hz); fehlende Schluessel
    nehmen DEFAULTS."""
    e = dict(DEFAULTS)
    e.update(eingaben or {})
    h = hebel(width=width)

    # --- thermisch -----------------------------------------------------
    T = e["T_uK"] * 1e-6
    nu = e["nu_kHz"] * 1e3
    T_max = (e["T_uK"] + abs(e["dT_uK"])) * 1e-6
    nu_min = max(e["nu_kHz"] - abs(e["dnu_kHz"]), 1e-3) * 1e3
    omega = 2 * np.pi * nu
    s_nom = float(sigma_thermal(atom_mass, omega, T)) if T > 0 else np.nan
    s_max = float(sigma_thermal(atom_mass, 2 * np.pi * nu_min, T_max)) if T_max > 0 else np.nan
    nullpunkt = float(np.sqrt(hbar / (2 * atom_mass * omega)))
    n_mittel = float(1.0 / np.expm1(hbar * omega / (kB * T))) if T > 0 else np.nan

    beitraege = [
        Beitrag("thermisch_nom", "thermische Ortsbreite (nominal)",
                f"T = {e['T_uK']:.3g} µK, nu_r = {e['nu_kHz']:.4g} kHz",
                "sigma = sqrt(hbar/(2 m w) coth(hbar w/2kT))", s_nom, "info",
                "steckt als Gewicht W bereits in U_w und eta_w"),
        Beitrag("thermisch_max", "thermische Ortsbreite (unguenstiger Rand)",
                f"T = {e['T_uK'] + abs(e['dT_uK']):.3g} µK, "
                f"nu_r = {nu_min * 1e-3:.4g} kHz",
                "wie oben", s_max, "thermisch",
                "geht in die Summe ein"),
    ]

    # Leistung: die Fallentiefe ist proportional zu P, omega^2 zur Tiefe.
    # Im klassischen Grenzfall sigma^2 = kT/(m omega^2) ~ 1/P, also
    # sigma ~ P^(-1/2) bei fester Temperatur. Genommen wird die Richtung,
    # in der sigma waechst (weniger Leistung).
    dP = abs(e["dP_prozent"]) / 100.0
    ds_P = s_nom * (1.0 / np.sqrt(max(1.0 - dP, 1e-9)) - 1.0) if np.isfinite(s_nom) else np.nan
    beitraege.append(Beitrag(
        "leistung", "Aenderung von sigma durch Fallenleistung",
        f"dP/P = {e['dP_prozent']:.3g} %", "sigma ~ P^(-1/2) (bei festem T)",
        ds_P, "info", "keine Verschiebung, nur Breite"))
    sag = G_ERDE / omega ** 2
    beitraege.append(Beitrag(
        "durchhang", "Durchhang durch Schwerkraft g/omega^2",
        f"nu_r = {e['nu_kHz']:.4g} kHz", "g / omega^2", sag, "info",
        "nur falls die Falle quer zur Schwerkraft steht; vernachlaessigbar"))

    # --- technisch -----------------------------------------------------
    tech = [
        Beitrag("relativ", "Relativposition Falle - Raman (gemessen, gesamt)",
                f"{e['relativ_nm']:.4g} nm", "1", e["relativ_nm"] * 1e-9, "technisch",
                _quelle("relativ_nm")),
        Beitrag("vor_aod", "Raman-Strahllage vor dem AOD",
                f"{e['theta_vor_aod_urad']:.4g} µrad",
                f"f1 fLO/f2 = {h['m_pro_rad_vor_aod'] * 1e3:.4g} mm "
                f"({h['m_pro_rad_vor_aod'] * 1e9 * 1e-6:.4g} nm/µrad)",
                e["theta_vor_aod_urad"] * 1e-6 * h["m_pro_rad_vor_aod"], "technisch",
                _quelle("theta_vor_aod_urad")),
        Beitrag("vor_obj", "Raman-Strahllage vor dem Objektiv",
                f"{e['theta_vor_obj_urad']:.4g} µrad",
                f"fLO = {h['m_pro_rad_vor_obj'] * 1e3:.4g} mm "
                f"({h['m_pro_rad_vor_obj'] * 1e9 * 1e-6:.4g} nm/µrad)",
                e["theta_vor_obj_urad"] * 1e-6 * h["m_pro_rad_vor_obj"], "technisch",
                _quelle("theta_vor_obj_urad")),
        Beitrag("schall", "AOD-Schallgeschwindigkeit",
                f"{e['dv_ppm']:.4g} ppm",
                f"f1 fLO/f2 * tan(theta_c) = {h['m_pro_dv_rel'] * 1e3:.4g} mm "
                f"({h['m_pro_dv_rel'] * 1e9 * 1e-6:.4g} nm/ppm, f_c = {h['f_c'] * 1e-6:.4g} MHz)",
                e["dv_ppm"] * 1e-6 * h["m_pro_dv_rel"], "technisch",
                _quelle("dv_ppm")),
        Beitrag("rf", "RF-Frequenzfehler",
                f"{e['df_Hz']:.4g} Hz",
                f"{h['m_pro_Hz'] * 1e12:.4g} pm/Hz",
                e["df_Hz"] * h["m_pro_Hz"], "technisch", _quelle("df_Hz")),
        Beitrag("falle", "Fallenposition (Tweezer-Strahllage)",
                f"{e['falle_nm']:.4g} nm", "1", e["falle_nm"] * 1e-9, "technisch",
                _quelle("falle_nm")),
        Beitrag("pitch", "Pitch-Unsicherheit an Site n",
                f"dp = {e['dp_nm']:.4g} nm, n = {e['n_site']:.3g}", "n",
                abs(e["n_site"]) * e["dp_nm"] * 1e-9, "technisch", _quelle("dp_nm")),
    ]
    beitraege += tech

    s_tech = float(np.sqrt(sum(b.wert_m ** 2 for b in tech)))
    s_ges = float(np.sqrt(s_tech ** 2 + (s_max if np.isfinite(s_max) else 0.0) ** 2))

    return Abschaetzung(beitraege=beitraege, sigma_therm_nom_m=s_nom,
                        sigma_therm_max_m=s_max, sigma_tech_m=s_tech,
                        sigma_gesamt_m=s_ges, nullpunkt_m=nullpunkt,
                        n_mittel=n_mittel, eingaben=e)


def _quelle(key):
    for k, _t, _e, _d, q in EINGABEN_THERMISCH + EINGABEN_TECHNISCH:
        if k == key:
            return q
    return ""


def anteil_innerhalb(k):
    """Anteil der Atome mit Abstand r <= k*sigma bei einer isotropen
    2D-Gaussverteilung mit sigma je Achse (Rayleigh-Verteilung)."""
    return float(1.0 - np.exp(-0.5 * float(k) ** 2))


def kurztext(ab, umfang="gesamt"):
    """Einzeiler fuer den Dialog."""
    return (f"sigma_therm = {ab.sigma_therm_nom_m * 1e9:.1f} nm "
            f"(ungünstig {ab.sigma_therm_max_m * 1e9:.1f} nm),  "
            f"sigma_tech = {ab.sigma_tech_m * 1e9:.1f} nm,  "
            f"gesamt = {ab.sigma_gesamt_m * 1e9:.1f} nm  "
            f"->  benutzt ({umfang}): {ab.sigma(umfang) * 1e9:.1f} nm")
