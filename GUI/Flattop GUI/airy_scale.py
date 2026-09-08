"""
airy_scale.py - Parametrisierung des Airy-Profils (Skalenfaktor)
================================================================

Eine Kopie dieser Datei liegt in `Hard_Optimization/` und in
`Weighted_Optimization/` (dasselbe Muster wie `perf_log.py` und
`scan_checkpoint.py`). `Combinated_Optimization/lib/combine.py` importiert
die Konstanten von hier, damit es im ganzen Projekt genau EINE Definition
gibt.

Worum es geht
-------------
Beim Airy-Profil rechnet der Optimierer

    first_zero_radius = airy_scale_factor * waist

Der Faktor legt fest, was die Zahl "waist" physikalisch BEDEUTET - und
damit jede Metrik eines Scans. Datensaetze mit verschiedenen Faktoren sind
NICHT vergleichbar.

    1.19      der historische Default des Projekts. Entspricht keiner
              gaengigen Konvention (nachgerechnet: gleicher 1/e^2-Radius
              1.4830, bester Gauss-Fit an die Hauptkeule 1.4499, gleiche
              FWHM 1.3956, gleicher Radius fuer 50 % Leistung 1.3425).
              Bei 1.19 liegt der tatsaechliche 1/e^2-Radius des
              Airy-Profils bei 0.8025 * waist, also 20 % unter der Zahl,
              die spaeter "waist" heisst.

    1.482951  so gewaehlt, dass der 1/e^2-Radius der Airy-HAUPTKEULE genau
              auf `waist` liegt - der Waist bedeutet dann bei Airy dasselbe
              wie bei einem Gauss-Strahl. Herleitung: (2*J1(u)/u)^2 faellt
              bei u = 2.583838989865 auf exp(-2), die erste Nullstelle
              liegt bei u = 3.831705970207512, also
              Faktor = 3.8317.../2.5838... = 1.482951.
              Gegengerechnet: damit ist der 1/e^2-Radius 1.000000168*waist.
"""

AIRY_SCALE_LEGACY = 1.19
AIRY_SCALE_GAUSS_E2 = 1.482951

# (key, Anzeigetext, Wert) - Wert None heisst "frei eingeben".
AIRY_SCALE_CHOICES = [
    ("legacy", "wie bisher (%.4f)" % AIRY_SCALE_LEGACY, AIRY_SCALE_LEGACY),
    ("gauss_e2", "1/e^2 der Airy-Hauptkeule = waist (%.4f)" % AIRY_SCALE_GAUSS_E2,
     AIRY_SCALE_GAUSS_E2),
    ("frei", "frei eingeben", None),
]

# u-Wert, bei dem die Airy-Hauptkeule auf 1/e^2 abgefallen ist, geteilt
# durch die erste Nullstelle - damit laesst sich zu JEDEM Faktor der
# zugehoerige 1/e^2-Radius ausrechnen (siehe airy_e2_radius_factor).
AIRY_E2_OVER_FIRST_ZERO = 2.583838989865 / 3.831705970207512

# Voreinstellung fuer NEUE Scans/GUIs: der physikalisch saubere Wert.
# Der Optimierer-Default (MultitoneFlatTopOptimizer.DEFAULTS) bleibt
# bewusst 1.19, damit aeltere Skripte und bereits gerechnete Datensaetze
# unveraendert weiterlaufen.
AIRY_SCALE_DIALOG_DEFAULT = AIRY_SCALE_GAUSS_E2


def airy_e2_radius_factor(scale_factor):
    """1/e^2-Radius der Airy-Hauptkeule in Einheiten von `waist`, zu einem
    gegebenen airy_scale_factor. Bei 1.19 sind das 0.8025, bei 1.482951
    genau 1.0."""
    return AIRY_E2_OVER_FIRST_ZERO * float(scale_factor)


def choice_index_for(value, tol=1e-9):
    """Index in AIRY_SCALE_CHOICES, dessen Wert `value` entspricht;
    sonst der Index von "frei eingeben"."""
    for i, (_key, _label, wert) in enumerate(AIRY_SCALE_CHOICES):
        if wert is not None and abs(float(value) - float(wert)) <= tol:
            return i
    return len(AIRY_SCALE_CHOICES) - 1


def scale_tag(value, tol=1e-9):
    """Kurzes Namenskuerzel fuer Dateinamen. Beim historischen Faktor 1.19
    ein LEERER String - dadurch bleiben die bisherigen Dateinamen exakt so,
    wie sie immer waren. Jeder andere Faktor bekommt ein eigenes Kuerzel,
    damit ein Lauf mit anderer Parametrisierung nie einen vorhandenen
    Datensatz ueberschreibt (die beiden waeren nicht vergleichbar)."""
    if abs(float(value) - AIRY_SCALE_LEGACY) <= tol:
        return ""
    return "_k%.4g" % float(value)


def describe(value):
    """Einzeiler, was der eingestellte Faktor konkret bedeutet."""
    f = float(value)
    return (
        "Bei Faktor %.4f: erste Nullstelle bei %.4f x waist, tatsaechlicher "
        "1/e^2-Radius bei %.4f x waist. Zwei Spots beruehren sich mit ihren "
        "Hauptkeulen bei einem Abstand von %.4f x waist (= k im verbotenen "
        "Bereich)." % (f, f, airy_e2_radius_factor(f), 2 * f)
    )


_TOOLTIP_MODE = (
    "Was die Zahl 'waist' physikalisch bedeuten soll.\n\n"
    "  wie bisher (1.19): der historische Default. Der tatsaechliche\n"
    "  1/e^2-Radius des Airy-Profils liegt dann bei 0.8025 * waist.\n\n"
    "  1/e^2 der Airy-Hauptkeule = waist (1.4830): der Waist bedeutet\n"
    "  dann bei Airy dasselbe wie bei einem Gauss-Strahl.\n\n"
    "  frei eingeben: eigener Wert im Feld darunter."
)

_TOOLTIP_FACTOR = (
    "first_zero_radius = Faktor * waist. Der Faktor setzt die physikalische\n"
    "Spotgroesse und damit JEDE Metrik dieses Laufs.\n"
    "\n"
    "Der bisherige Default 1.19 entspricht KEINER der gaengigen Konventionen\n"
    "(nachgerechnet):\n"
    "   gleicher 1/e^2-Radius wie ein Gauss  -> 1.4830\n"
    "   bester Gauss-Fit an die Hauptkeule   -> 1.4499\n"
    "   gleiche FWHM                         -> 1.3956\n"
    "   gleicher Radius fuer 50 % Leistung   -> 1.3425\n"
    "\n"
    "Bei 1.19 liegt der tatsaechliche 1/e^2-Radius des Airy-Profils bei\n"
    "0.8025 * waist, also 20 % unter der Zahl, die spaeter 'waist' heisst.\n"
    "\n"
    "ACHTUNG: Datensaetze mit verschiedenen Faktoren sind NICHT vergleichbar."
)


# ----------------------------------------------------------------------
# Optionale Qt-Gruppe - nur verfuegbar, wenn PyQt5 installiert ist.
# Der Rest dieses Moduls (Konstanten, Funktionen) haengt NICHT an Qt und
# laesst sich ueberall importieren, auch dort, wo kein Qt vorhanden ist.
# ----------------------------------------------------------------------
try:
    from PyQt5.QtWidgets import (
        QGroupBox, QFormLayout, QLabel, QComboBox, QDoubleSpinBox,
    )
except Exception:      # pragma: no cover - Qt fehlt (z.B. reiner Rechenlauf)
    QGroupBox = None


if QGroupBox is not None:

    class AiryScaleGroup(QGroupBox):
        """Fertige Dialog-Gruppe "Strahlprofil (Airy)": Dropdown mit den
        benannten Konventionen, Zahlenfeld (nur bei "frei eingeben"
        editierbar) und eine Zeile, die live sagt, was der Wert bedeutet.

        Benutzung im Dialog:

            self.airy_group = AiryScaleGroup()
            main_layout.addWidget(self.airy_group)
            ...
            factor = self.airy_group.value()
        """

        def __init__(self, default=AIRY_SCALE_DIALOG_DEFAULT, parent=None,
                     title="Strahlprofil (Airy)"):
            super().__init__(title, parent)

            layout = QFormLayout()

            info = QLabel(
                "Der Skalenfaktor uebersetzt den Parameter waist in die\n"
                "Airy-Skala:    first_zero_radius = Faktor * waist")
            info.setStyleSheet("font-style: italic;")
            layout.addRow(info)

            self.mode_combo = QComboBox()
            for _key, label, _wert in AIRY_SCALE_CHOICES:
                self.mode_combo.addItem(label)
            self.mode_combo.setToolTip(_TOOLTIP_MODE)
            layout.addRow("Parametrisierung:", self.mode_combo)

            self.factor_spin = QDoubleSpinBox()
            self.factor_spin.setRange(0.1, 5.0)
            self.factor_spin.setDecimals(6)
            self.factor_spin.setSingleStep(0.01)
            self.factor_spin.setToolTip(_TOOLTIP_FACTOR)
            layout.addRow("airy_scale_factor:", self.factor_spin)

            self.hint = QLabel("")
            self.hint.setWordWrap(True)
            self.hint.setStyleSheet("color: gray;")
            layout.addRow(self.hint)

            self.setLayout(layout)

            # Signale erst nach dem Aufbau verbinden, damit setValue() unten
            # nicht in einen halb aufgebauten Zustand hineinlaeuft.
            self.factor_spin.setValue(float(default))
            self.mode_combo.setCurrentIndex(choice_index_for(default))
            self.mode_combo.currentIndexChanged.connect(lambda _i: self._sync_mode())
            self.factor_spin.valueChanged.connect(lambda _v: self._sync_hint())
            self._sync_mode()

        # -- oeffentliche API ------------------------------------------
        def value(self):
            """Der eingestellte airy_scale_factor."""
            return float(self.factor_spin.value())

        def set_value(self, factor):
            self.mode_combo.setCurrentIndex(choice_index_for(factor))
            self.factor_spin.setValue(float(factor))
            self._sync_mode()

        def mode_key(self):
            return AIRY_SCALE_CHOICES[self.mode_combo.currentIndex()][0]

        # -- intern ----------------------------------------------------
        def _sync_mode(self):
            """Bei einer benannten Konvention steht der Wert fest - das Feld
            zeigt ihn dann nur an. Nur "frei eingeben" laesst ihn tippen."""
            _key, _label, wert = AIRY_SCALE_CHOICES[self.mode_combo.currentIndex()]
            if wert is None:
                self.factor_spin.setEnabled(True)
            else:
                self.factor_spin.setEnabled(False)
                self.factor_spin.setValue(float(wert))
            self._sync_hint()

        def _sync_hint(self):
            self.hint.setText(describe(self.factor_spin.value()))
