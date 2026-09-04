"""
coherence.py - Statische Interferenz frequenzentarteter Spots
=============================================================

Eine Kopie dieser Datei liegt in `Hard_Optimization/lib/` und in
`Weighted_Optimization/lib/` (dasselbe Muster wie `airy_scale.py`,
`perf_log.py` und `scan_checkpoint.py`). `Combinated_Optimization` benutzt
ueber seinen sys.path-Eintrag die Fassung aus `Weighted_Optimization/lib`,
damit es im ganzen Projekt genau EINE Definition gibt.

Worum es geht
-------------
Die Toene sind untereinander KOHAERENT. Ihre Kreuzterme laufen mit der
Differenzfrequenz um und mitteln sich in jeder Messung weg - AUSSER bei
Paaren mit identischer Gesamtfrequenz. Deren Kreuzterm liegt bei 0 Hz,
mittelt sich also nie weg und steht als STATISCHES Interferenzmuster im
Bild. Die reine Intensitaetssumme laesst ihn weg.

Beide AODs schieben die Lichtfrequenz, ein Spot (n, m) traegt also
f_s = f_x(n) + f_y(m). Mit f_x(n) = offset + width * n/(N_x-1) ist

    f_s = 2*offset + width * ( n/(N_x-1) + m/(N_y-1) )

Zwei Spots sind entartet, wenn die Klammer uebereinstimmt - das haengt NUR
an N_x und N_y, nicht an offset oder width. Bei gleicher width auf beiden
Achsen gibt es deshalb IMMER mindestens ein entartetes Paar (die diagonal
gegenueberliegenden Eckspots), und bei N_x = N_y ist jede Anti-Diagonale
vollstaendig entartet.

Zeitmittel der kohaerenten Ueberlagerung:

    <I> = sum_s g_s^2  +  sum ueber entartete Paare 2 g_s g_t cos(dphi)

Der erste Term ist die bisherige inkohaerente Summe, der zweite der
statische Anteil. Gerechnet wird mit dphi = 0, also VOLL KONSTRUKTIV: das
ist der unguenstigste Fall, und ein Scan soll den bewerten. (Bei 90 Grad
verschwindet der Term fuer ein einzelnes Paar exakt - das auszunutzen
erfordert eingestellte Tonphasen und ist hier bewusst nicht vorgesehen.)

Die Felder und der Kreuzterm selbst stehen im jeweiligen Optimierer-Modul
(`static_interference`, `FIELD_FUNCS`), weil sie an dessen
Profil-Konventionen haengen: die Intensitaet ist dort exp(-2 r^2/sigma^2)
bzw. (2*J1(u)/u)^2, das Feld also die Wurzel davon.

ACHTUNG: Datensaetze mit und ohne Kohaerenz sind NICHT vergleichbar. Der
Schluessel `coherent` im gespeicherten dict sagt, was gerechnet wurde;
fehlt er, stammt der Datensatz aus der Zeit vor dieser Option und wurde
inkohaerent gerechnet.
"""

import numpy as np

COHERENT_DEFAULT = True


def degenerate_spot_groups(N_x, N_y, rel_tol=1e-9):
    """Gruppen von Spot-Indizes mit identischer Gesamtfrequenz.

    Reihenfolge der Indizes wie bei den Spot-Zentren: x aussen, y innen,
    also Index = n * N_y + m. Zurueck kommen nur Gruppen mit mindestens
    zwei Mitgliedern; ohne Entartung eine leere Liste.
    """
    if N_x < 1 or N_y < 1:
        return []
    a = np.arange(N_x) / (N_x - 1) if N_x > 1 else np.zeros(1)
    b = np.arange(N_y) / (N_y - 1) if N_y > 1 else np.zeros(1)
    schluessel = np.round((a[:, None] + b[None, :]).ravel() / rel_tol).astype(np.int64)
    gruppen = []
    for wert in np.unique(schluessel):
        idx = np.flatnonzero(schluessel == wert)
        if idx.size > 1:
            gruppen.append(idx)
    return gruppen


def n_degenerate_pairs(N_x, N_y):
    """Anzahl entarteter PAARE (eine Gruppe aus k Spots liefert k*(k-1)/2)."""
    return sum(len(g) * (len(g) - 1) // 2 for g in degenerate_spot_groups(N_x, N_y))


def degeneracy_summary(N_x, N_y):
    """Ein Satz fuer Dialog und Bericht."""
    gruppen = degenerate_spot_groups(N_x, N_y)
    paare = sum(len(g) * (len(g) - 1) // 2 for g in gruppen)
    if not paare:
        return (f"{N_x}x{N_y}: keine frequenzentarteten Spots - die Kohaerenz "
                f"aendert hier nichts.")
    return (f"{N_x}x{N_y}: {paare} entartete(s) Paar(e) in {len(gruppen)} "
            f"Gruppe(n) - deren Kreuzterm steht fest im Bild.")


def model_label(coherent):
    """Kurzbezeichnung des Rechenmodells fuer Berichte und Dateinamen-Notizen."""
    return ("kohaerent (statische Interferenz mitgerechnet)" if coherent
            else "inkohaerent (reine Intensitaetssumme)")


COHERENCE_TOOLTIP = (
    "Die Toene sind untereinander kohaerent. Ihre Kreuzterme laufen mit der\n"
    "Differenzfrequenz um und mitteln sich weg - AUSSER bei Paaren mit\n"
    "identischer Gesamtfrequenz f_x(n) + f_y(m). Deren Kreuzterm liegt bei\n"
    "0 Hz und steht als statisches Muster im Bild.\n\n"
    "Bei gleicher width auf beiden Achsen gibt es immer mindestens ein solches\n"
    "Paar; bei N_x = N_y ist jede Anti-Diagonale entartet.\n\n"
    "Gerechnet wird mit Phasendifferenz 0, also voll konstruktiv - der\n"
    "unguenstigste Fall.\n\n"
    "Ohne Haken rechnet der Scan wie vor dem 2026-09-04, also inkohaerent\n"
    "(reine Intensitaetssumme). ACHTUNG: Datensaetze mit und ohne Kohaerenz\n"
    "sind NICHT vergleichbar."
)


# ----------------------------------------------------------------------
# Optionale Qt-Gruppe - nur verfuegbar, wenn PyQt5 installiert ist. Der Rest
# dieses Moduls haengt NICHT an Qt (siehe airy_scale.py, gleiches Muster).
# ----------------------------------------------------------------------
try:
    from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QCheckBox
except Exception:      # pragma: no cover - Qt fehlt (reiner Rechenlauf)
    QGroupBox = None


if QGroupBox is not None:

    class CoherenceGroup(QGroupBox):
        """Fertige Dialog-Gruppe "Kohaerenz (statische Interferenz)":
        ein Haken (per Default gesetzt) und eine Zeile, die sagt, wie viele
        entartete Paare die eingestellte Tonzahl hat.

        Benutzung im Dialog:

            self.coherence_group = CoherenceGroup(N_x, N_y)
            main_layout.addWidget(self.coherence_group)
            ...
            coherent = self.coherence_group.value()
        """

        def __init__(self, N_x=3, N_y=4, default=COHERENT_DEFAULT, parent=None,
                     title="Kohärenz (statische Interferenz)"):
            super().__init__(title, parent)

            layout = QVBoxLayout()

            self.check = QCheckBox("frequenzentartete Spots kohärent überlagern")
            self.check.setChecked(bool(default))
            self.check.setToolTip(COHERENCE_TOOLTIP)
            layout.addWidget(self.check)

            self.info = QLabel(degeneracy_summary(N_x, N_y))
            self.info.setWordWrap(True)
            self.info.setStyleSheet("color: #555; font-style: italic;")
            layout.addWidget(self.info)

            self.setLayout(layout)

        def set_tones(self, N_x, N_y):
            """Tonzahl nachziehen, wenn der Dialog sie einstellbar macht."""
            self.info.setText(degeneracy_summary(N_x, N_y))

        def value(self):
            return self.check.isChecked()
