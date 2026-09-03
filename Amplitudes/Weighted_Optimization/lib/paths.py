"""
lib/paths.py - Ordner-Konstanten, sys.path-Anbindung, Datensatz-Namen.

Wird von allen anderen lib-Modulen und von den Haupt-Skripten als ERSTES
importiert: der sys.path-Eintrag auf DIESEN Ordner muss stehen, bevor
`multitone_flattop_optimizer` & Co. importiert werden koennen.

Die eigentliche Rechen- und Plot-Bibliothek liegt seit dem Aufraeumen in
diesem lib/-Ordner; ausgefuehrt werden nur die run_*.py eine Ebene hoeher.
Die Ausgabeordner liegen weiterhin neben den run_*.py (also eine Ebene
UEBER lib/) und werden bei Bedarf automatisch angelegt:

    Results/                 gepickelte Scan-Rohdaten (.pkl)
    Bilder/<JJJJ-MM-TT>/     PNG-Ausgaben der Scan-Plotter, tageweise
    Fit_Plots/<JJJJ-MM-TT>/  Vektor-PDFs der Auswertung, tageweise
    Fit_Results/             Markdown-Berichte (flach, Datum im Dateinamen)

Diese Datei ist das EINZIGE, was sich zwischen Hard_Optimization/lib und
Weighted_Optimization/lib unterscheidet - scan_data.py, report.py und
run_plots.py sind in beiden Ordnern identisch und holen alles
Ordner-Spezifische von hier.
"""

import sys
from datetime import date
from pathlib import Path as FilePath

# lib/ selbst, und eine Ebene darueber der eigentliche Arbeitsordner.
PACKAGE_DIR = FilePath(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
AMPLITUDES_DIR = BASE_DIR.parent

# Die Module in diesem Ordner importieren sich gegenseitig unter ihrem
# blossen Namen (`import scan_checkpoint`, `from multitone_flattop_optimizer
# import ...`). Damit das nach dem Umzug nach lib/ weiter funktioniert -
# und zwar unabhaengig davon, von wo aus gestartet wurde -, liegt lib/
# selbst im sys.path.
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# HINWEIS ZU PYCHARM: die Module aus lib/ werden erst zur LAUFZEIT ueber den
# sys.path-Eintrag oben gefunden. PyCharms statische Analyse kennt diesen
# Eintrag nicht und markiert solche Importe rot ("unresolved reference") -
# ein reines Anzeigeproblem, der Code laeuft. Dauerhaft weg bekommt man das
# Rot mit einem einmaligen Handgriff:
#   Rechtsklick auf den Ordner "lib" -> "Mark Directory as" -> "Sources Root"


def _default_dir(name):
    """Ordner neben den run_*.py anlegen (bzw. verwenden). Faellt auf einen
    Ordner relativ zum Arbeitsverzeichnis zurueck, falls der Pfad nicht
    beschreibbar ist (z.B. anderer Rechner)."""
    candidate = BASE_DIR / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = FilePath(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


# ----------------------------------------------------------------------
# Ausgabeordner
# ----------------------------------------------------------------------
# BILDER werden tageweise abgelegt: Bilder/2026-09-02/, Fit_Plots/2026-09-02/.
# Grund ist schlicht die Menge - ein Auswertungslauf schreibt bis zu acht
# PDFs, und nach ein paar Laeufen liess sich im flachen Ordner nicht mehr
# sagen, was zusammengehoert. Der Dateiname traegt das Datum weiterhin, der
# Ordner ist also redundant und nicht die einzige Auskunft darueber.
#
# ROHDATEN (Results/) und BERICHTE (Fit_Results/) bleiben flach: die .pkl ist
# der Datensatz, nicht das Ergebnis eines Tages, und die Berichte will man
# nebeneinander lesen koennen. Beide tragen das Datum im Namen.
DEFAULT_RESULTS_DIR = _default_dir("Results")
FIT_RESULTS_DIR = _default_dir("Fit_Results")

# Die flachen Elternordner - zum Blaettern, und damit ein Aufrufer die
# Tagesordner nebeneinander findet.
IMAGES_ROOT = _default_dir("Bilder")
FIT_PLOTS_ROOT = _default_dir("Fit_Plots")


def tages_unterordner(name, tag=None):
    """`<name>/<JJJJ-MM-TT>`, angelegt falls noetig.

    `tag=None` heisst HEUTE, ausgewertet beim Aufruf - ein Lauf, der ueber
    Mitternacht geht, schreibt seine spaeteren Plots also in den neuen Tag.
    Das ist gewollt: der Ordner soll sagen, wann die Datei entstanden ist."""
    tag = date.today().isoformat() if tag is None else str(tag)
    return _default_dir(f"{name}/{tag}")


def bilder_dir(tag=None):
    """Tagesordner fuer die PNG-Ausgaben der Scan-Plotter."""
    return tages_unterordner("Bilder", tag)


def fit_plots_dir(tag=None):
    """Tagesordner fuer die Vektor-PDFs der Auswertung (run_plots.py)."""
    return tages_unterordner("Fit_Plots", tag)


# Modul-Konstanten fuer Aufrufer, die sie als Konstante importieren (die
# run_*.py der Scans). Sie zeigen auf den Tag, an dem importiert wurde;
# report.py ruft statt dessen fit_plots_dir() auf und trifft damit immer den
# aktuellen Tag.
DEFAULT_IMAGES_DIR = bilder_dir()
FIT_PLOTS_DIR = fit_plots_dir()


# ----------------------------------------------------------------------
# Welche Metrik-Familie dieser Ordner rechnet
# ----------------------------------------------------------------------
# "hard"     - globale Pitch-Box-Maske (Hard_Optimization)
# "weighted" - atom-gewichtet ueber ein lokales Sub-Gitter (Weighted_Optimization)
#
# scan_data.py und report.py lesen das hier aus und sind deshalb in beiden
# Ordnern buchstabengleich.
FLAVOR = "weighted"
FLAVOR_LABEL = "atom-gewichtet"

# Die vom Optimierer gespeicherten Gitter-Schluessel dieser Familie.
UNIFORMITY_KEY = "uniformity_weighted_grid"
CROSSTALK_KEY = "eta_weighted_grid"

# Dateinamens-Muster der beiden Datensatz-Arten dieses Ordners.
FIXED_PKL_GLOB = "scan_data_weighted_*.pkl"      # feste Amplituden  (run_scan.py)
AMP_PKL_GLOB = "scan_amp_data_weighted_*.pkl"  # r_x/r_y optimiert (run_amp_scan.py)

# Praefix der Auswertungs-Dateien (run_plots.py)
REPORT_STEM = "WeightedScan"


# ----------------------------------------------------------------------
# Die Plot-Bibliothek dieses Ordners
# ----------------------------------------------------------------------
# report.py und run_plots.py importieren AUSSCHLIESSLICH von hier, damit sie
# in beiden Ordnern identisch bleiben koennen (die Weighted-Fassung dieser
# Datei re-exportiert dieselben Namen aus dem gewichteten Plot-Modul).
from weighted_multitone_amplitude_dependence_plots import (   # noqa: E402
    AmplitudeScanPlotter,
    WeightedFixedScanPlotter as FixedScanPlotter,
    load_amp_scan_results,
    resolve_save_path,
    win_input_to_win,
    width_to_um,
    summarize_amp_bounds,
    detect_amp_outliers,
    clean_amp_scan_results,
)


def profile_tag_of(profile):
    return {"airy": "Airy", "gaussian": "Gauss"}.get(profile, "ProfilUnbekannt")


def newest_matching(pattern, directory=None):
    """Neueste zum Muster passende Datei in `directory` (Default: Results/)
    oder None."""
    directory = DEFAULT_RESULTS_DIR if directory is None else FilePath(directory)
    kandidaten = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return kandidaten[-1] if kandidaten else None
