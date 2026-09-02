"""
scan_checkpoint.py
=====================
Generische Zwischenspeicherung/Fortsetzung fuer lange 2D-Parameter-Scans
(scan_win_width_uniformity() und alle Verwandten im ganzen Projekt -
Hard_Optimization, Weighted_Optimization, Combinated_Optimization).
Bewusst OHNE Abhaengigkeiten zu den Optimizer-Modulen - identisches Muster
wie perf_log.py: eigenstaendige, identische Kopie in Hard_Optimization UND
Weighted_Optimization (Combinated_Optimization importiert die Kopie aus
Weighted_Optimization ueber den bereits vorhandenen sys.path-Eintrag).

Hintergrund (siehe Chat "Amplituden Abhängigkeit"): manche Scans (v.a. die
amplituden-optimierten mit vielen Gitterpunkten) koennen mehrere Tage
laufen. Ohne Zwischenspeicherung ist ein Absturz, Neustart oder Kill
(Stromausfall, Windows-Update, versehentlich geschlossenes Fenster) ein
Totalverlust der bis dahin geleisteten Rechenzeit.

Prinzip:
1. Der Speicherort (Pfad) wird VOR dem Scan-Start festgelegt (siehe
   GUI-Dialoge - Speicherort wird jetzt am Anfang abgefragt statt erst am
   Ende) und als checkpoint_path an die jeweilige scan_win_width_...()-
   Methode durchgereicht.
2. Alle CHECKPOINT_INTERVAL_S (Default 3600s = 1 Stunde) wird der
   bisherige Fortschritt unter GENAU diesem Pfad gespeichert - als
   dieselbe Art dict, die auch get_scan_results()/get_scan_amp_results()/
   etc. am Ende liefern wuerden (Metrik-Grids mit NaN fuer noch nicht
   berechnete Punkte), nur zusaetzlich mit '_checkpoint': True markiert.
   Kein Sonderformat - die Datei ist jederzeit (auch mitten im Scan) mit
   den bestehenden Plot-/Fit-Skripten ladbar (NaN-Punkte werden von diesen
   ohnehin schon als "nicht auswertbar" behandelt).
3. Bei jedem (Neu-)Start wird zuerst geprueft, ob unter diesem Pfad
   bereits eine ZU DEN AKTUELLEN SCAN-PARAMETERN passende (Teil-)Datei
   liegt (identisches Gitter, N_x/N_y, ...) - falls ja, wird sie geladen
   und nur die darin noch NaN/fehlenden Punkte werden neu berechnet
   ("Sollte der Prozess abbrechen, kann ich den Scan an der Stelle neu
   starten").

Schreibt IMMER atomar (erst .tmp-Datei, dann umbenennen) - ein Absturz
WAEHREND des Speicherns kann so nie eine halb geschriebene/korrupte
Checkpoint-Datei hinterlassen; die vorherige, vollstaendige Version bleibt
in diesem Fall einfach erhalten.
"""

import time
import pickle
from pathlib import Path

import numpy as np


CHECKPOINT_INTERVAL_S = 3600.0  # 1 Stunde, wie gewuenscht


# ======================================================================
# Schreiben
# ======================================================================
def atomic_pickle_dump(obj, path):
    """Schreibt obj als Pickle nach path - atomar (erst .tmp, dann
    os.replace()), damit ein Absturz mitten im Schreiben nicht die
    vorherige, gueltige Checkpoint-Datei zerstoert."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(path)


class CheckpointWriter:
    """Kapselt das stundenweise (oder anders konfigurierte) periodische
    Speichern waehrend eines laufenden Scans. Wallclock-Uhr startet bei
    Konstruktion (NICHT beim ersten maybe_save()-Aufruf), damit ein
    interval_s=3600 auch wirklich nach spaetestens einer Stunde den ersten
    Checkpoint schreibt."""

    def __init__(self, checkpoint_path, interval_s=CHECKPOINT_INTERVAL_S, verbose=True):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.interval_s = interval_s
        self.verbose = verbose
        self._last_save = time.monotonic()

    @property
    def active(self):
        return self.checkpoint_path is not None

    def due(self):
        return self.active and (time.monotonic() - self._last_save) >= self.interval_s

    def maybe_save(self, build_results_fn, done=None, total=None, force=False):
        """build_results_fn() -> dict (wird NUR bei Bedarf aufgerufen -
        billig genug fuer die hier vorkommenden Grid-Groessen, aber
        trotzdem nicht bei jedem einzelnen Gitterpunkt ausgewertet)."""
        if not self.active:
            return False
        if not force and not self.due():
            return False
        results = dict(build_results_fn())
        results["_checkpoint"] = True
        results["_checkpoint_saved_at"] = time.time()
        atomic_pickle_dump(results, self.checkpoint_path)
        self._last_save = time.monotonic()
        if self.verbose:
            progress = f" ({done}/{total} Punkte)" if done is not None and total is not None else ""
            print(f"[Checkpoint] Zwischenstand gespeichert: {self.checkpoint_path}{progress}")
        return True


# ======================================================================
# Laden / Fortsetzen
# ======================================================================
def _values_equal(a, b):
    """Vergleich, der auch mit None, Arrays/Listen und Tupeln (z.B.
    r_bounds) sinnvoll umgeht - fuer die Kompatibilitaetspruefung beim
    Fortsetzen."""
    if a is None or b is None:
        return a is b
    try:
        a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a_arr.shape != b_arr.shape:
            return False
        return bool(np.allclose(a_arr, b_arr, equal_nan=True))
    except (TypeError, ValueError):
        return a == b


# Airy-Skalenfaktor, der galt, bevor er einstellbar war (vor 2026-09-01).
# Zwischenstaende aus dieser Zeit fuehren das Feld nicht mit - fuer sie gilt
# dieser Wert, damit ein damals abgebrochener Scan weiterhin fortsetzbar ist.
AIRY_SCALE_LEGACY_DEFAULT = 1.19


def load_resumable(checkpoint_path, win_input_range, width_range, n_win_input, n_width,
                    N_x, N_y, extra_match=None, airy_scale_factor=None,
                    optics_match=None, verbose=True):
    """Prueft, ob unter checkpoint_path bereits eine zu den AKTUELLEN
    Scan-Parametern passende (Teil- oder fertige) Ergebnis-Datei liegt,
    und laedt sie in diesem Fall. Gibt das geladene dict zurueck, oder
    None (nichts gefunden / nicht lesbar / Parameter passen nicht - dann
    startet der Scan bei allen Gitterpunkten von vorne, wie bisher).

    extra_match: optionales dict zusaetzlicher Schluessel/Werte, die
    ebenfalls uebereinstimmen muessen (z.B. amps, alpha, r_bounds) - siehe
    _values_equal() fuer den Vergleich.

    airy_scale_factor: der Faktor des JETZT laufenden Scans. Ist er gesetzt,
    wird er gegen den im Zwischenstand gespeicherten geprueft und bei
    Abweichung NICHT fortgesetzt. Das ist wichtig: der Faktor legt die
    physikalische Spotgroesse fest (first_zero_radius = Faktor * waist), ein
    fortgesetzter Scan mit anderem Faktor haette also zwei verschiedene
    Optiken in EINEM Datensatz - ohne diese Pruefung waere das von aussen
    nicht mehr erkennbar. Zwischenstaende von vor dem 2026-09-01 fuehren das
    Feld nicht; dort wird der damalige Optimierer-Default 1.19 angenommen.

    optics_match: optionales dict {Schluessel: aktueller Wert} fuer Groessen,
    die WIE FEIN gerechnet wurde festlegen - n_grid, weighted_n_grid,
    atom_offset_x/y. Steht der Schluessel im Zwischenstand, muss er
    uebereinstimmen, sonst wird nicht fortgesetzt (sonst haette der Datensatz
    zwei verschiedene Aufloesungen). Fehlt er (Datei von vor dem 2026-09-01),
    ist er nicht pruefbar - dann wird fortgesetzt, aber deutlich gewarnt.
    """
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path is None or not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, "rb") as f:
            saved = pickle.load(f)
    except Exception as exc:
        if verbose:
            print(f"[Checkpoint] WARNUNG: '{checkpoint_path}' existiert, konnte aber nicht "
                  f"gelesen werden ({exc!r}) - Scan startet komplett neu.")
        return None

    try:
        expected_win = np.linspace(win_input_range[0], win_input_range[1], n_win_input)
        expected_width = np.linspace(width_range[0], width_range[1], n_width)
        win_ok = _values_equal(saved.get("win_input_vals"), expected_win)
        width_ok = _values_equal(saved.get("width_vals"), expected_width)
        n_ok = saved.get("N_x") == N_x and saved.get("N_y") == N_y
        scale_ok = True
        if airy_scale_factor is not None:
            saved_scale = saved.get("airy_scale_factor")
            if saved_scale is None:
                saved_scale = AIRY_SCALE_LEGACY_DEFAULT   # Datei von vor 2026-09-01
            scale_ok = abs(float(saved_scale) - float(airy_scale_factor)) <= 1e-9
        optics_ok = True
        optics_problem = None
        optics_unknown = []
        if optics_match:
            for key, value in optics_match.items():
                if key not in saved or saved.get(key) is None:
                    optics_unknown.append(key)      # Datei von vor 2026-09-01
                    continue
                if not _values_equal(saved.get(key), value):
                    optics_ok = False
                    optics_problem = (key, saved.get(key), value)
                    break
        extra_ok = True
        if extra_match:
            for key, value in extra_match.items():
                if not _values_equal(saved.get(key), value):
                    extra_ok = False
                    break
    except Exception:
        win_ok = width_ok = n_ok = extra_ok = scale_ok = optics_ok = False
        optics_problem = None
        optics_unknown = []

    if not scale_ok and verbose:
        saved_scale = saved.get("airy_scale_factor", AIRY_SCALE_LEGACY_DEFAULT)
        print(f"[Checkpoint] '{checkpoint_path}' wurde mit airy_scale_factor="
              f"{saved_scale} gerechnet, dieser Scan laeuft mit {airy_scale_factor} - "
              f"NICHT fortgesetzt (das waeren zwei verschiedene Optiken in einem "
              f"Datensatz). Bitte einen anderen Speicherpfad waehlen, sonst wird die "
              f"vorhandene Datei beim naechsten Zwischenspeichern ueberschrieben.")

    if not optics_ok and verbose and optics_problem is not None:
        key, was, now = optics_problem
        print(f"[Checkpoint] '{checkpoint_path}' wurde mit {key}={was} gerechnet, dieser "
              f"Scan laeuft mit {key}={now} - NICHT fortgesetzt (der Datensatz haette "
              f"sonst zwei verschiedene Aufloesungen). Bitte einen anderen Speicherpfad "
              f"waehlen, sonst wird die vorhandene Datei beim naechsten "
              f"Zwischenspeichern ueberschrieben.")

    if optics_unknown and optics_ok and verbose:
        print(f"[Checkpoint] WARNUNG: '{checkpoint_path}' fuehrt "
              f"{', '.join(optics_unknown)} nicht mit (Datei von vor dem 2026-09-01) - "
              f"es laesst sich also NICHT pruefen, ob dieser Scan mit derselben "
              f"Aufloesung weiterrechnet. Wird fortgesetzt; bitte selbst darauf achten, "
              f"dieselben Werte wie beim ersten Lauf einzustellen.")

    if not (win_ok and width_ok and n_ok and extra_ok and scale_ok and optics_ok):
        if verbose and scale_ok and optics_ok:
            print(f"[Checkpoint] '{checkpoint_path}' existiert bereits, passt aber NICHT zu den "
                  f"aktuellen Scan-Parametern (anderes Gitter/N_x/N_y/Einstellungen) - wird beim "
                  f"naechsten Speichern ueberschrieben, Scan startet komplett neu.")
        return None

    return saved


def count_done(grid):
    """Anzahl bereits berechneter (endlicher) Punkte in einem Metrik-Grid
    - fuer Log-/Fortschrittsausgaben beim Fortsetzen."""
    grid = np.asarray(grid, dtype=float)
    return int(np.sum(np.isfinite(grid)))


# ======================================================================
# "Bester Punkt" aus einem (moeglicherweise noch unvollstaendigen) Grid-
# Paar neu berechnen - gebraucht, um waehrend eines laufenden Scans
# jederzeit ein vollstaendiges, sofort ladbares Zwischenergebnis zu
# schreiben (siehe CheckpointWriter oben). Generisch ueber die Feldnamen,
# da 'uniformity'/'crosstalk' (hart) und 'uniformity_weighted'/
# 'eta_weighted' (atom-gewichtet) dasselbe Schema mit anderen Schluesseln
# nutzen (siehe scan_win_width_uniformity()/scan_win_width_weighted_
# uniformity() in weighted_multitone_flattop_optimizer.py/weighted_amp_
# scan_methods.py).
# ======================================================================
def best_point(U, C, win_input_vals, width_vals, alpha, u_key="uniformity", c_key="crosstalk"):
    combined_grid = alpha * U + (1.0 - alpha) * C
    best = {"win_input": None, "width": None, u_key: None, c_key: None, "combined": None}
    if np.any(np.isfinite(combined_grid)):
        idx_min = np.unravel_index(np.nanargmin(combined_grid), combined_grid.shape)
        best.update({
            "win_input": win_input_vals[idx_min[1]], "width": width_vals[idx_min[0]],
            u_key: U[idx_min], c_key: C[idx_min], "combined": combined_grid[idx_min],
        })
    return best
