"""
gaussbeam.py -- Gaussche Strahlpropagation mit ABCD-Matrizen.

Konventionen
------------
* Strahlvektor (y, theta) mit *echten* Winkeln (nicht reduziert).
  Damit gilt fuer eine Grenzflaeche det(M) = n1/n2.
* Der komplexe Strahlparameter q ist im *lokalen* Medium definiert:

      1/q = 1/R - i * lam_eff / (pi * n * w**2),     lam_eff = M^2 * lam_0

  Freiraum/Medium ueber die geometrische Laenge L:  q -> q + L
  Element:                                          q -> (A q + B) / (C q + D)

* Tangential (t) = Ebene des Einfalls (Kippung), Sagittal (s) = senkrecht dazu.
  Beide Ebenen werden komplett getrennt propagiert -> Astigmatismus wird
  korrekt erfasst.

Alle Groessen intern in SI (Meter, Radiant). Winkel in den Parametern der
Komponenten sind in Grad, weil das die GUI so anzeigt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

INF = float("inf")


# ----------------------------------------------------------------------------
# Parameter-Beschreibung (die GUI baut daraus automatisch ihre Eingabefelder)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Param:
    key: str
    label: str
    unit: str
    default: float          # in Anzeige-Einheit
    scale: float = 1.0      # Anzeige-Einheit * scale = SI
    vmin: float = -1e12
    vmax: float = 1e12
    decimals: int = 4
    tip: str = ""

    def to_si(self, display_value: float) -> float:
        return display_value * self.scale

    def to_display(self, si_value: float) -> float:
        return si_value / self.scale


# Haeufig gebrauchte Bausteine
def _P_len(key, label, default, tip=""):
    return Param(key, label, "mm", default, 1e-3, -1e7, 1e7, 4, tip)


def _P_ang(key, label, default=0.0, tip=""):
    return Param(key, label, "Grad", default, 1.0, -89.0, 89.0, 3, tip)


def _P_n(key, label, default=1.5, tip=""):
    return Param(key, label, "-", default, 1.0, 1.0, 5.0, 5, tip)


# ----------------------------------------------------------------------------
# Elementare Schritte, in die jede Komponente zerlegt wird
# ----------------------------------------------------------------------------
@dataclass
class Prop:
    """Ausbreitung ueber die geometrische Weglaenge L im Medium n."""
    length: float
    n: float
    label: str = ""


@dataclass
class Mat:
    """Duennes Element: 2x2-Matrix je Ebene, evtl. Brechungsindexwechsel."""
    Mt: np.ndarray
    Ms: np.ndarray
    n_out: float
    label: str = ""


class BeamError(Exception):
    pass


# ----------------------------------------------------------------------------
# Matrizen einzelner Flaechen
# ----------------------------------------------------------------------------
def snell(n1: float, n2: float, theta1: float) -> float:
    """Brechungswinkel (rad). Wirft bei Totalreflexion."""
    s = n1 * math.sin(theta1) / n2
    if abs(s) > 1.0:
        raise BeamError(
            f"Totalreflexion: n1={n1:.4f}, n2={n2:.4f}, "
            f"Einfallswinkel={math.degrees(theta1):.2f} Grad"
        )
    return math.asin(s)


def interface_matrices(n1: float, n2: float, R: float, theta1: float
                       ) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Brechung an einer (evtl. gekippten) sphaerischen Grenzflaeche.

    R > 0  : Kruemmungsmittelpunkt liegt hinter der Flaeche (konvex zur
             einfallenden Seite)  --  R = inf  ->  ebene Flaeche.
    theta1 : Einfallswinkel gegen die Flaechennormale (rad).

    Liefert (M_tangential, M_sagittal, theta2).

    Herleitung ueber die Coddington-Gleichungen:
        P = (n2*cos(t2) - n1*cos(t1)) / R
        tangential:  n2 cos^2(t2)/s' - n1 cos^2(t1)/s = P
        sagittal:    n2/s'           - n1/s           = P
    """
    t2 = snell(n1, n2, theta1)
    c1, c2 = math.cos(theta1), math.cos(t2)

    if math.isinf(R) or R == 0.0 and False:
        P = 0.0
    else:
        P = (n2 * c2 - n1 * c1) / R

    Mt = np.array([[c2 / c1, 0.0],
                   [-P / (n2 * c1 * c2), n1 * c1 / (n2 * c2)]], dtype=float)
    Ms = np.array([[1.0, 0.0],
                   [-P / n2, n1 / n2]], dtype=float)
    return Mt, Ms, t2


def lens_matrices(f: float, tilt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Duenne Linse, evtl. gekippt: f_t = f*cos(tilt), f_s = f/cos(tilt)."""
    c = math.cos(tilt)
    ft, fs = f * c, f / c
    Mt = np.array([[1.0, 0.0], [-1.0 / ft, 1.0]])
    Ms = np.array([[1.0, 0.0], [-1.0 / fs, 1.0]])
    return Mt, Ms


# ----------------------------------------------------------------------------
# Komponenten
# ----------------------------------------------------------------------------
class Component:
    kind: str = "base"
    label: str = "Komponente"
    params: List[Param] = []

    def __init__(self, **display_values):
        self.values: Dict[str, float] = {}
        for p in self.params:
            self.values[p.key] = p.to_si(display_values.get(p.key, p.default))
        self.enabled: bool = True

    # -- Zugriff -------------------------------------------------------------
    def get(self, key: str) -> float:
        return self.values[key]

    def set(self, key: str, si_value: float) -> None:
        self.values[key] = si_value

    def param(self, key: str) -> Param:
        for p in self.params:
            if p.key == key:
                return p
        raise KeyError(key)

    # -- Physik --------------------------------------------------------------
    def steps(self, n_in: float) -> List[Any]:
        raise NotImplementedError

    # -- Anzeige / Persistenz ------------------------------------------------
    def summary(self) -> str:
        parts = []
        for p in self.params:
            v = p.to_display(self.values[p.key])
            parts.append(f"{p.label}={v:g}{'' if p.unit == '-' else ' ' + p.unit}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "enabled": self.enabled,
                "values": dict(self.values)}

    @staticmethod
    def from_dict(d: dict) -> "Component":
        cls = COMPONENT_TYPES[d["kind"]]
        c = cls()
        c.values.update({k: float(v) for k, v in d["values"].items()})
        c.enabled = bool(d.get("enabled", True))
        return c


class Distance(Component):
    kind = "distance"
    label = "Abstand / Freiraum"
    params = [_P_len("L", "Laenge", 100.0, "Geometrische Weglaenge im aktuellen Medium")]

    def steps(self, n_in):
        return [Prop(self.get("L"), n_in, "Abstand")]


class ThinLens(Component):
    kind = "thin_lens"
    label = "Duenne Linse"
    params = [
        _P_len("f", "Brennweite", 100.0, "f > 0 sammelnd, f < 0 zerstreuend"),
        _P_ang("tilt", "Kippwinkel", 0.0,
               "Kippung erzeugt Astigmatismus: f_t = f*cos, f_s = f/cos"),
    ]

    def steps(self, n_in):
        f = self.get("f")
        if f == 0.0:
            raise BeamError("Brennweite darf nicht 0 sein.")
        Mt, Ms = lens_matrices(f, math.radians(self.get("tilt")))
        return [Mat(Mt, Ms, n_in, "Duenne Linse")]


class CurvedSurface(Component):
    kind = "surface"
    label = "Grenzflaeche (sphaerisch/plan)"
    params = [
        Param("R", "Kruemmungsradius", "mm", 50.0, 1e-3, -1e7, 1e7, 4,
              "R > 0: Mittelpunkt hinter der Flaeche. 0 oder leer = plan"),
        _P_n("n2", "Index danach", 1.5),
        _P_ang("theta", "Einfallswinkel", 0.0),
    ]

    def steps(self, n_in):
        R = self.get("R")
        R = INF if R == 0.0 else R
        n2 = self.get("n2")
        Mt, Ms, _ = interface_matrices(n_in, n2, R, math.radians(self.get("theta")))
        return [Mat(Mt, Ms, n2, "Grenzflaeche")]


class ThickLens(Component):
    kind = "thick_lens"
    label = "Dicke Linse"
    params = [
        Param("R1", "Radius R1", "mm", 50.0, 1e-3, -1e7, 1e7, 4,
              "R1 > 0: erste Flaeche konvex. 0 = plan"),
        Param("R2", "Radius R2", "mm", -50.0, 1e-3, -1e7, 1e7, 4,
              "R2 < 0: zweite Flaeche konvex. 0 = plan"),
        _P_len("d", "Mittendicke", 5.0),
        _P_n("n", "Brechzahl Glas", 1.5168, "z.B. N-BK7 = 1.5168 bei 587 nm"),
        _P_ang("tilt", "Kippwinkel", 0.0,
               "Einfallswinkel auf die erste Flaeche"),
    ]

    def steps(self, n_in):
        R1 = self.get("R1") or INF
        R2 = self.get("R2") or INF
        d = self.get("d")
        n = self.get("n")
        th1 = math.radians(self.get("tilt"))

        Mt1, Ms1, th2 = interface_matrices(n_in, n, R1, th1)
        # Innenweg entlang des gebrochenen Strahls
        path = d / math.cos(th2)
        Mt2, Ms2, _ = interface_matrices(n, n_in, R2, th2)
        return [Mat(Mt1, Ms1, n, "Linse Flaeche 1"),
                Prop(path, n, "Linse innen"),
                Mat(Mt2, Ms2, n_in, "Linse Flaeche 2")]


class Crystal(Component):
    kind = "crystal"
    label = "Kristall / planparallele Platte"
    params = [
        _P_len("t", "Dicke", 10.0, "Dicke senkrecht zu den Flaechen"),
        _P_n("n", "Brechzahl", 1.5),
        _P_ang("theta", "Einfallswinkel", 0.0,
               "0 Grad = senkrecht. Schraeg -> Astigmatismus"),
    ]

    def steps(self, n_in):
        t = self.get("t")
        n = self.get("n")
        th1 = math.radians(self.get("theta"))
        Mt1, Ms1, th2 = interface_matrices(n_in, n, INF, th1)
        Mt2, Ms2, _ = interface_matrices(n, n_in, INF, th2)
        return [Mat(Mt1, Ms1, n, "Eintrittsflaeche"),
                Prop(t / math.cos(th2), n, "im Kristall"),
                Mat(Mt2, Ms2, n_in, "Austrittsflaeche")]


class BrewsterCrystal(Component):
    kind = "brewster"
    label = "Kristall im Brewsterwinkel"
    params = [
        _P_len("t", "Dicke", 10.0),
        _P_n("n", "Brechzahl", 1.76, "z.B. Ti:Saphir 1.76, YAG 1.82"),
    ]

    def steps(self, n_in):
        n = self.get("n")
        th1 = math.atan(n / n_in)          # Brewsterwinkel
        t = self.get("t")
        Mt1, Ms1, th2 = interface_matrices(n_in, n, INF, th1)
        Mt2, Ms2, _ = interface_matrices(n, n_in, INF, th2)
        return [Mat(Mt1, Ms1, n, "Brewster ein"),
                Prop(t / math.cos(th2), n, "im Kristall"),
                Mat(Mt2, Ms2, n_in, "Brewster aus")]

    def summary(self):
        n = self.get("n")
        return (super().summary() +
                f", Brewsterwinkel={math.degrees(math.atan(n)):.2f} Grad")


class CurvedMirror(Component):
    kind = "mirror"
    label = "Gekruemmter Spiegel"
    params = [
        _P_len("R", "Kruemmungsradius", 100.0, "R > 0 = konkav (fokussierend)"),
        _P_ang("theta", "Einfallswinkel", 0.0,
               "Halber Faltwinkel. f_t = R*cos/2, f_s = R/(2*cos)"),
    ]

    def steps(self, n_in):
        R = self.get("R")
        if R == 0.0:
            raise BeamError("Spiegelradius darf nicht 0 sein.")
        th = math.radians(self.get("theta"))
        ft = R * math.cos(th) / 2.0
        fs = R / (2.0 * math.cos(th))
        Mt = np.array([[1.0, 0.0], [-1.0 / ft, 1.0]])
        Ms = np.array([[1.0, 0.0], [-1.0 / fs, 1.0]])
        return [Mat(Mt, Ms, n_in, "Spiegel")]


class GenericABCD(Component):
    kind = "abcd"
    label = "Freie ABCD-Matrix"
    params = [
        Param("A", "A", "-", 1.0, 1.0, -1e6, 1e6, 6),
        Param("B", "B", "mm", 0.0, 1e-3, -1e7, 1e7, 6),
        Param("C", "C", "1/m", 0.0, 1.0, -1e7, 1e7, 6),
        Param("D", "D", "-", 1.0, 1.0, -1e6, 1e6, 6),
    ]

    def steps(self, n_in):
        M = np.array([[self.get("A"), self.get("B")],
                      [self.get("C"), self.get("D")]], dtype=float)
        return [Mat(M, M.copy(), n_in, "ABCD")]


COMPONENT_TYPES: Dict[str, type] = {
    c.kind: c for c in [Distance, ThinLens, ThickLens, CurvedSurface,
                        Crystal, BrewsterCrystal, CurvedMirror, GenericABCD]
}


# ----------------------------------------------------------------------------
# Eingangsstrahl
# ----------------------------------------------------------------------------
@dataclass
class InputBeam:
    wavelength: float = 1064e-9   # Vakuumwellenlaenge [m]
    w0: float = 0.5e-3            # Waistradius (1/e^2 Feldradius) [m]
    z_waist: float = 0.0          # Abstand Waist -> erste Komponente [m]
    m2: float = 1.0               # Strahlqualitaet
    n0: float = 1.0               # Index vor der ersten Komponente
    w0_sag: Optional[float] = None      # optional getrennter sagittaler Waist
    z_waist_sag: Optional[float] = None

    @property
    def lam_eff(self) -> float:
        return self.m2 * self.wavelength

    def divergence(self) -> float:
        """Fernfeld-Halbwinkel im Eingangsmedium [rad]."""
        return self.lam_eff / (math.pi * self.n0 * self.w0)

    def rayleigh(self) -> float:
        return math.pi * self.w0 ** 2 * self.n0 / self.lam_eff

    @classmethod
    def from_divergence(cls, wavelength, w0, theta, **kw):
        """M^2 aus Waist und gemessener Fernfeld-Divergenz bestimmen."""
        n0 = kw.get("n0", 1.0)
        m2 = math.pi * n0 * w0 * theta / wavelength
        return cls(wavelength=wavelength, w0=w0, m2=m2, **kw)

    def q_start(self) -> Tuple[complex, complex]:
        """q in der Ebene der ersten Komponente, (tangential, sagittal)."""
        zr_t = math.pi * self.w0 ** 2 * self.n0 / self.lam_eff
        qt = complex(self.z_waist, zr_t)
        w0s = self.w0 if self.w0_sag is None else self.w0_sag
        zws = self.z_waist if self.z_waist_sag is None else self.z_waist_sag
        zr_s = math.pi * w0s ** 2 * self.n0 / self.lam_eff
        qs = complex(zws, zr_s)
        return qt, qs

    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("wavelength", "w0", "z_waist", "m2", "n0", "w0_sag", "z_waist_sag")}


# ----------------------------------------------------------------------------
# q-Auswertung
# ----------------------------------------------------------------------------
def w_of_q(q: complex, n: float, lam_eff: float) -> float:
    inv = 1.0 / q
    return math.sqrt(-lam_eff / (math.pi * n * inv.imag))


def R_of_q(q: complex) -> float:
    inv = 1.0 / q
    return INF if inv.real == 0.0 else 1.0 / inv.real


@dataclass
class PlaneResult:
    """Kennwerte einer Ebene (tangential oder sagittal) an einer Stelle."""
    w: float             # Strahlradius hier
    R: float             # Kruemmungsradius der Phasenfront
    w0: float            # Waistradius des zugehoerigen Strahls
    z_to_waist: float    # Abstand bis zum Waist (>0: liegt voraus)
    z_rayleigh: float
    theta: float         # Fernfeld-Halbwinkel
    n: float

    def as_row(self):
        return (self.w, self.w0, self.z_to_waist, self.z_rayleigh, self.theta, self.R)


def analyse(q: complex, n: float, lam_eff: float) -> PlaneResult:
    zr = q.imag
    w0 = math.sqrt(lam_eff * zr / (math.pi * n))
    return PlaneResult(
        w=w_of_q(q, n, lam_eff),
        R=R_of_q(q),
        w0=w0,
        z_to_waist=-q.real,
        z_rayleigh=zr,
        theta=lam_eff / (math.pi * n * w0),
        n=n,
    )


# ----------------------------------------------------------------------------
# System
# ----------------------------------------------------------------------------
@dataclass
class Sample:
    z: np.ndarray
    wt: np.ndarray
    ws: np.ndarray


@dataclass
class Marker:
    z: float
    label: str


@dataclass
class SystemResult:
    z_total: float
    n_out: float
    tangential: PlaneResult
    sagittal: PlaneResult
    samples: Sample
    markers: List[Marker]
    stages: List[Tuple[str, float, PlaneResult, PlaneResult]]  # (label, z, t, s)
    q_t: complex = 0j          # q am Ausgang (fuer Weiterrechnen/Plot-Nachlauf)
    q_s: complex = 0j
    lam_eff: float = 1.0

    @property
    def astigmatism(self) -> float:
        """Abstand der beiden Waists (tangential vs. sagittal)."""
        return self.tangential.z_to_waist - self.sagittal.z_to_waist


class OpticalSystem:
    def __init__(self, beam: Optional[InputBeam] = None,
                 components: Optional[List[Component]] = None):
        self.beam = beam or InputBeam()
        self.components: List[Component] = components or []

    # -- Aufbau --------------------------------------------------------------
    def add(self, comp: Component) -> Component:
        self.components.append(comp)
        return comp

    # -- Rechnung ------------------------------------------------------------
    def _flatten(self) -> List[Any]:
        steps: List[Any] = []
        n = self.beam.n0
        for c in self.components:
            if not c.enabled:
                continue
            for st in c.steps(n):
                if isinstance(st, Mat):
                    n = st.n_out
                steps.append(st)
        return steps

    def propagate(self, n_samples: int = 1200) -> SystemResult:
        lam = self.beam.lam_eff
        qt, qs = self.beam.q_start()
        n = self.beam.n0
        z = 0.0

        zs: List[float] = []
        wt: List[float] = []
        ws: List[float] = []
        markers: List[Marker] = []
        stages: List[Tuple[str, float, PlaneResult, PlaneResult]] = []

        steps = self._flatten()
        total_len = sum(s.length for s in steps if isinstance(s, Prop))
        if total_len <= 0:
            total_len = max(self.beam.rayleigh() * 4.0, 1e-3)

        zs.append(0.0)
        wt.append(w_of_q(qt, n, lam))
        ws.append(w_of_q(qs, n, lam))

        for st in steps:
            if isinstance(st, Prop):
                L, n = st.length, st.n
                if L < 0:
                    raise BeamError("Negative Laenge ist nicht erlaubt.")
                k = max(3, int(n_samples * L / total_len) + 2)
                for zz in np.linspace(0.0, L, k)[1:]:
                    zs.append(z + zz)
                    wt.append(w_of_q(qt + zz, n, lam))
                    ws.append(w_of_q(qs + zz, n, lam))
                qt = qt + L
                qs = qs + L
                z += L
            else:
                A, B = st.Mt[0]
                C, D = st.Mt[1]
                den = C * qt + D
                if den == 0:
                    raise BeamError(f"Singulaere Matrix bei '{st.label}'.")
                qt = (A * qt + B) / den
                A, B = st.Ms[0]
                C, D = st.Ms[1]
                den = C * qs + D
                if den == 0:
                    raise BeamError(f"Singulaere Matrix bei '{st.label}'.")
                qs = (A * qs + B) / den
                n = st.n_out
                markers.append(Marker(z, st.label))
                zs.append(z)
                wt.append(w_of_q(qt, n, lam))
                ws.append(w_of_q(qs, n, lam))
            stages.append((st.label, z, analyse(qt, n, lam), analyse(qs, n, lam)))

        return SystemResult(
            z_total=z,
            n_out=n,
            tangential=analyse(qt, n, lam),
            sagittal=analyse(qs, n, lam),
            samples=Sample(np.array(zs), np.array(wt), np.array(ws)),
            markers=markers,
            stages=stages,
            q_t=qt,
            q_s=qs,
            lam_eff=lam,
        )

    # -- Persistenz ----------------------------------------------------------
    def to_dict(self):
        return {"beam": self.beam.to_dict(),
                "components": [c.to_dict() for c in self.components]}

    @staticmethod
    def from_dict(d: dict) -> "OpticalSystem":
        beam = InputBeam(**d["beam"])
        comps = [Component.from_dict(c) for c in d["components"]]
        return OpticalSystem(beam, comps)


# ----------------------------------------------------------------------------
# Ziel-Optimierung (ohne scipy)
# ----------------------------------------------------------------------------
PLANES = ("tangential", "sagittal", "mittel")

GOALS = {
    "waist_size":  "Waist-Radius am Ausgang = Ziel",
    "waist_pos":   "Waist-Position hinter dem letzten Element = Ziel",
    "w_end":       "Strahlradius am Ausgang = Ziel",
    "min_w_end":   "Strahlradius am Ausgang minimieren",
    "collimate":   "Ausgang kollimieren (Waist am Ausgang)",
    "round":       "Astigmatismus beseitigen (w_t = w_s am Ausgang)",
}


def _plane_value(res: SystemResult, plane: str, attr: str) -> float:
    t = getattr(res.tangential, attr)
    s = getattr(res.sagittal, attr)
    if plane == "tangential":
        return t
    if plane == "sagittal":
        return s
    return 0.5 * (t + s)


def objective(res: SystemResult, goal: str, target: float, plane: str) -> float:
    if goal == "waist_size":
        return abs(_plane_value(res, plane, "w0") - target)
    if goal == "waist_pos":
        return abs(_plane_value(res, plane, "z_to_waist") - target)
    if goal == "w_end":
        return abs(_plane_value(res, plane, "w") - target)
    if goal == "min_w_end":
        return _plane_value(res, plane, "w")
    if goal == "collimate":
        return abs(_plane_value(res, plane, "z_to_waist"))
    if goal == "round":
        return abs(res.tangential.w - res.sagittal.w)
    raise ValueError(goal)


@dataclass
class OptimizeResult:
    value: float          # optimaler Parameterwert (SI)
    cost: float
    ok: bool
    message: str


def optimize(system: OpticalSystem,
             comp_index: int,
             param_key: str,
             lo: float, hi: float,
             goal: str,
             target: float = 0.0,
             plane: str = "tangential",
             coarse: int = 240,
             tol: float = 1e-12) -> OptimizeResult:
    """
    Sucht den Wert eines einzelnen Parameters (SI-Einheit), der das Ziel am
    besten trifft: grobes Raster + Golden-Section-Verfeinerung.
    """
    comp = system.components[comp_index]
    original = comp.get(param_key)

    def cost(x: float) -> float:
        comp.set(param_key, x)
        try:
            return objective(system.propagate(n_samples=8), goal, target, plane)
        except (BeamError, ValueError, ZeroDivisionError, FloatingPointError):
            return float("inf")

    try:
        xs = np.linspace(lo, hi, max(11, coarse))
        costs = np.array([cost(float(x)) for x in xs])
        if not np.isfinite(costs).any():
            return OptimizeResult(original, float("inf"), False,
                                  "Kein gueltiger Wert im Suchbereich gefunden.")
        i = int(np.nanargmin(np.where(np.isfinite(costs), costs, np.inf)))
        a = xs[max(i - 1, 0)]
        b = xs[min(i + 1, len(xs) - 1)]

        # Golden Section
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c_, d_ = b - gr * (b - a), a + gr * (b - a)
        fc, fd = cost(float(c_)), cost(float(d_))
        for _ in range(200):
            if abs(b - a) < tol + 1e-9 * max(abs(a), abs(b)):
                break
            if fc < fd:
                b, d_, fd = d_, c_, fc
                c_ = b - gr * (b - a)
                fc = cost(float(c_))
            else:
                a, c_, fc = c_, d_, fd
                d_ = a + gr * (b - a)
                fd = cost(float(d_))
        best = 0.5 * (a + b)
        fbest = cost(float(best))
        if not math.isfinite(fbest) or fbest > costs[i]:
            best, fbest = float(xs[i]), float(costs[i])
        return OptimizeResult(best, fbest, True, "OK")
    finally:
        comp.set(param_key, original)
