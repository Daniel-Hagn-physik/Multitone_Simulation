"""
lib/paths.py - Ordner-Konstanten und sys.path-Anbindung.

Wird von allen anderen lib-Modulen und von den drei Hauptskripten als
ERSTES importiert, weil der sys.path-Eintrag auf ../../Weighted_Optimization
gesetzt sein muss, bevor `weighted_multitone_flattop_optimizer` & Co.
importiert werden koennen.

Alle Ausgabeordner liegen in Combinated_Optimization/ (also eine Ebene
UEBER diesem lib/-Ordner) und werden bei Bedarf automatisch angelegt:

    Results/       gepickelte Scan-Rohdaten (.pkl)
    Bilder/<JJJJ-MM-TT>/     interaktive/PNG-Ausgaben, tageweise
    Fit_Plots/<JJJJ-MM-TT>/  Vektor-PDFs der Auswertung, tageweise
    Fit_Results/   Markdown-Berichte der Auswertung
"""

import sys
from datetime import date
from pathlib import Path as FilePath

# Combinated_Optimization/ - eine Ebene ueber lib/
PACKAGE_DIR = FilePath(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
AMPLITUDES_DIR = BASE_DIR.parent

# Der eigentliche Optimierer (MultitoneFlatTopOptimizer), die Plot-Klassen,
# scan_checkpoint und perf_log liegen unveraendert in Weighted_Optimization
# und werden von hier aus mitbenutzt - NICHT dupliziert.
#
# SEIT DEM AUFRAEUMEN (2026-09-02) liegen sie dort im Unterordner lib/;
# frueher lagen sie direkt in Weighted_Optimization. Beide Faelle werden
# unterstuetzt, damit dieser Ordner auch mit einer aelteren Kopie von
# Weighted_Optimization laeuft.
WEIGHTED_BASE = AMPLITUDES_DIR / "Weighted_Optimization"
_KANDIDATEN = (WEIGHTED_BASE / "lib", WEIGHTED_BASE)
WEIGHTED_DIR = next(
    (d for d in _KANDIDATEN
     if (d / "weighted_multitone_flattop_optimizer.py").exists()),
    None)

if WEIGHTED_DIR is None:
    raise ImportError(
        f"weighted_multitone_flattop_optimizer.py wurde nicht gefunden (gesucht in "
        f"{_KANDIDATEN[0]} und {_KANDIDATEN[1]}).\n"
        f"Combinated_Optimization benutzt von dort den Optimierer, die Plot-Klassen, "
        f"scan_checkpoint und perf_log. Beide Ordner muessen nebeneinander unter "
        f"'Amplitudes/' liegen."
    )

if str(WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(WEIGHTED_DIR))

# HINWEIS ZU PYCHARM: die Module aus WEIGHTED_DIR werden erst zur LAUFZEIT
# ueber den sys.path-Eintrag oben gefunden. PyCharms statische Analyse kennt
# diesen Eintrag nicht und markiert solche Importe rot ("unresolved
# reference") - das ist ein reines Anzeigeproblem, der Code laeuft.
# Dauerhaft weg bekommt man das Rot mit einem einmaligen Handgriff:
#   Rechtsklick auf den Ordner "Weighted_Optimization/lib"
#   -> "Mark Directory as" -> "Sources Root"
# Danach kennt PyCharm die Module und faerbt sie normal ein.
# Damit das Rot gar nicht erst in den drei Haupt-Skripten auftaucht, sind
# alle Importe aus WEIGHTED_DIR hier in lib/ gebuendelt; run_penalty_scan.py,
# run_hard_check.py und run_plots.py importieren ausschliesslich aus lib.


def _default_dir(name):
    """Ordner neben Combinated_Optimization/ anlegen (bzw. verwenden).
    Faellt auf einen Ordner relativ zum Arbeitsverzeichnis zurueck, falls
    der Pfad nicht beschreibbar ist (z.B. anderer Rechner)."""
    candidate = BASE_DIR / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = FilePath("..") / name
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
# Dateinamens-Muster
# ----------------------------------------------------------------------
# Penalty-Scan (run_penalty_scan.py) - bewusst DASSELBE Muster wie bisher,
# damit bereits vorhandene Datensaetze (z.B.
# scan_amp_data_combined_N3x4_21x21pts_Airy.pkl) unveraendert dazupassen
# und direkt weiter ausgewertet werden koennen.
PENALTY_PKL_GLOB = "scan_amp_data_combined_*.pkl"
# Hard-Check (run_hard_check.py) - eigenes Muster, damit die beiden
# Datensatz-Arten nie verwechselt werden.
HARDCHECK_PKL_GLOB = "hard_check_*.pkl"


def penalty_pkl_name(N_x, N_y, n_win, n_width, profile):
    profile_tag = profile_tag_of(profile)
    return f"scan_amp_data_combined_N{N_x}x{N_y}_{n_win}x{n_width}pts_{profile_tag}.pkl"


def hardcheck_pkl_name(N_x, N_y, n_win, n_width, profile):
    profile_tag = profile_tag_of(profile)
    return f"hard_check_N{N_x}x{N_y}_{n_win}x{n_width}pts_{profile_tag}.pkl"


def profile_tag_of(profile):
    return {"airy": "Airy", "gaussian": "Gauss"}.get(profile, str(profile))


def newest_matching(pattern, directory=None):
    """Neueste zum Muster passende Datei in `directory` (Default:
    Results/) oder None."""
    directory = DEFAULT_RESULTS_DIR if directory is None else FilePath(directory)
    kandidaten = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return kandidaten[-1] if kandidaten else None
