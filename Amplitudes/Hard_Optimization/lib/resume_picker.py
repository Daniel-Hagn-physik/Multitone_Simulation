"""
resume_picker.py - unfertige Scans aus einer Liste auswaehlen und fortsetzen
===========================================================================

Eine Kopie dieser Datei liegt in `Hard_Optimization/` und in
`Weighted_Optimization/` (dasselbe Muster wie `perf_log.py`,
`scan_checkpoint.py` und `airy_scale.py`); `Combinated_Optimization`
importiert die Kopie aus `Weighted_Optimization` ueber den bereits
vorhandenen sys.path-Eintrag.

Wozu
----
Ein abgebrochener Scan liess sich bisher nur fortsetzen, indem man im
Dialog ALLE Parameter noch einmal genauso eintippte wie beim ersten Lauf -
sonst passte der Zwischenstand nicht und der Scan begann von vorne. Schlimmer:
`n_grid`, `weighted_n_grid` und der Atom-Offset wurden frueher gar nicht
mitgespeichert, ein Tippfehler dort fiel also nicht einmal auf.

Dieses Modul dreht die Richtung um: man waehlt die unfertige Datei aus, und
ALLE Parameter kommen aus ihr. Die Eingabefelder werden dabei gesperrt, und
`apply_to_params()` ueberschreibt zusaetzlich die von `get_values()`
gelieferten Werte direkt aus der Datei - die Anzeige ist also nur noch
Information, massgeblich ist der Datensatz selbst.
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Datensatz-Arten
# ----------------------------------------------------------------------
# Welche Art Scan liegt vor? Ein Dialog zeigt nur die Dateien an, die zu ihm
# passen - ein Fest-Amplituden-Zwischenstand hat im Amplituden-Scan-Dialog
# nichts verloren und liesse sich dort auch nicht fortsetzen.
KIND_HARD_FIXED = "hard_fixed"        # scan_win_width_uniformity
KIND_HARD_AMP = "hard_amp"            # scan_win_width_amplitude_dependence
KIND_WEIGHTED_FIXED = "weighted_fixed"    # ..._weighted_uniformity
KIND_WEIGHTED_AMP = "weighted_amp"        # ..._amplitude_dependence_weighted
KIND_PENALTY = "penalty"                  # Combinated_Optimization

KIND_LABELS = {
    KIND_HARD_FIXED: "harter Scan, feste Amplituden",
    KIND_HARD_AMP: "harter Scan, Amplituden optimiert",
    KIND_WEIGHTED_FIXED: "gewichteter Scan, feste Amplituden",
    KIND_WEIGHTED_AMP: "gewichteter Scan, Amplituden optimiert",
    KIND_PENALTY: "Penalty-Scan (hart + gewichtet gemeinsam)",
}

# Metrik-Gitter, an dem der Fortschritt abgelesen wird (erstes vorhandenes).
_PROGRESS_GRIDS = ("uniformity_grid", "uniformity_weighted_grid")


def kind_of(results):
    """Art des Datensatzes aus seinen Schluesseln bestimmen, oder None."""
    if not isinstance(results, dict):
        return None
    if results.get("dataset_kind") == "penalty" or results.get("joint_optimization"):
        return KIND_PENALTY
    has_amp = "r_x_grid" in results
    has_hard = "uniformity_grid" in results
    has_weighted = "uniformity_weighted_grid" in results
    if has_hard and has_weighted:
        return KIND_PENALTY
    if has_weighted:
        return KIND_WEIGHTED_AMP if has_amp else KIND_WEIGHTED_FIXED
    if has_hard:
        return KIND_HARD_AMP if has_amp else KIND_HARD_FIXED
    return None


def progress_of(results):
    """(fertige Punkte, Punkte gesamt) - oder (None, None), wenn sich das
    aus diesem Datensatz nicht ablesen laesst."""
    for key in _PROGRESS_GRIDS:
        if key in results:
            grid = np.asarray(results[key], dtype=float)
            return int(np.sum(np.isfinite(grid))), int(grid.size)
    return None, None


def is_unfinished(results):
    """Unfertig = ausdruecklich als Zwischenstand markiert ODER es fehlen
    noch Punkte im Metrik-Gitter (NaN)."""
    if results.get("_checkpoint"):
        return True
    done, total = progress_of(results)
    return done is not None and done < total


# ----------------------------------------------------------------------
# Dateien finden - OHNE sie zu lesen
# ----------------------------------------------------------------------
# WICHTIG: Beim Oeffnen des Dialogs wird KEINE .pkl eingelesen. Ein
# Results-Ordner kann Dutzende Dateien mit zig MB enthalten, und wenn er
# (wie hier) in OneDrive liegt, loest schon das Oeffnen einer ausgelagerten
# Datei einen Download aus - der Dialog haengt dann sekunden- bis
# minutenlang, bevor er ueberhaupt erscheint. Deshalb wird die Liste nur
# aus Dateinamen und os.stat() aufgebaut (beides beruehrt den Inhalt
# nicht); eingelesen wird ausschliesslich die Datei, die der Nutzer
# tatsaechlich auswaehlt.
#
# Die Art wird dafuer erst am Dateinamen geraten und beim Auswaehlen am
# tatsaechlichen Inhalt geprueft - passt sie nicht, sagt der Dialog das
# klar, statt stillschweigend etwas Falsches fortzusetzen.

# Reihenfolge: spezifischste Praefixe zuerst.
_NAME_PATTERNS = [
    ("scan_amp_data_combined", KIND_PENALTY),
    ("scan_amp_data_joint", KIND_PENALTY),
    ("scan_amp_data_weighted", KIND_WEIGHTED_AMP),
    ("scan_data_weighted", KIND_WEIGHTED_FIXED),
    ("scan_amp_data_", KIND_HARD_AMP),
    ("scan_data_", KIND_HARD_FIXED),
]


def guess_kind_from_name(name):
    """Art am Dateinamen raten (ohne die Datei zu lesen), oder None."""
    low = name.lower()
    for prefix, kind in _NAME_PATTERNS:
        if low.startswith(prefix):
            return kind
    return None


def list_candidates(directory, kind=None, extra_dirs=()):
    """Kandidaten fuer das Auswahlfeld - nur Name, Groesse, Zeitstempel.

    Es wird NICHTS eingelesen. Enthalten sind Dateien, deren Name auf die
    gesuchte Art passt, sowie Dateien mit unbekanntem Namensmuster (etwa
    selbst umbenannte) - ob sie wirklich passen, entscheidet erst
    load_entry() beim Auswaehlen.
    """
    out = []
    seen = set()
    for d in [directory] + list(extra_dirs):
        if not d:
            continue
        d = Path(d)
        if not d.is_dir():
            continue
        try:
            it = list(os.scandir(d))
        except OSError:
            continue
        for e in it:
            if not e.name.lower().endswith(".pkl") or not e.is_file():
                continue
            rp = os.path.normcase(os.path.abspath(e.path))
            if rp in seen:
                continue
            seen.add(rp)
            guessed = guess_kind_from_name(e.name)
            if kind is not None and guessed is not None and guessed != kind:
                continue
            try:
                st = e.stat()
            except OSError:
                continue
            out.append(dict(path=e.path, name=e.name, mtime=st.st_mtime,
                            size=st.st_size, guessed_kind=guessed, results=None))
    out.sort(key=lambda x: x["mtime"], reverse=True)
    for x in out:
        x["label"] = describe_candidate(x)
    return out


def describe_candidate(cand):
    """Einzeiler fuer das Auswahlfeld - ohne die Datei gelesen zu haben."""
    when = time.strftime("%d.%m.%Y %H:%M", time.localtime(cand["mtime"]))
    return "%s  -  %.1f MB, %s" % (cand["name"], cand["size"] / 1e6, when)



# ----------------------------------------------------------------------
# Dateien finden
# ----------------------------------------------------------------------
def find_unfinished(directory, kind=None, extra_dirs=()):
    """Alle unfertigen .pkl-Datensaetze in `directory` (und optional in
    `extra_dirs`), neueste zuerst.

    kind: nur Datensaetze dieser Art zurueckgeben (siehe KIND_*).
          None = alle.

    Rueckgabe: Liste von dicts mit path, name, results, kind, n_done,
    n_total, saved_at, label. Unlesbare oder fremde Dateien werden
    stillschweigend uebersprungen - der Dialog soll nicht an einer
    kaputten Datei im Ordner scheitern.
    """
    entries = []
    dirs = [directory] + list(extra_dirs)
    seen = set()
    for d in dirs:
        if not d:
            continue
        d = Path(d)
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.pkl")):
            rp = str(path.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            try:
                with open(path, "rb") as fh:
                    res = pickle.load(fh)
            except Exception:
                continue
            if not isinstance(res, dict):
                continue
            k = kind_of(res)
            if k is None or not is_unfinished(res):
                continue
            if kind is not None and k != kind:
                continue
            done, total = progress_of(res)
            entries.append(dict(
                path=str(path), name=path.name, results=res, kind=k,
                n_done=done, n_total=total,
                saved_at=res.get("_checkpoint_saved_at") or path.stat().st_mtime,
            ))
    entries.sort(key=lambda e: e["saved_at"], reverse=True)
    for e in entries:
        e["label"] = describe(e)
    return entries


def describe(entry):
    """Einzeiler fuer das Auswahlfeld."""
    res = entry["results"]
    when = time.strftime("%d.%m.%Y %H:%M", time.localtime(entry["saved_at"]))
    n = len(np.asarray(res.get("win_input_vals", [])))
    m = len(np.asarray(res.get("width_vals", [])))
    pct = (100.0 * entry["n_done"] / entry["n_total"]) if entry["n_total"] else 0.0
    return ("%s  -  %d/%d Punkte (%.0f %%), %dx%d Gitter, zuletzt %s"
            % (entry["name"], entry["n_done"], entry["n_total"], pct, n, m, when))


def load_entry(path):
    """Eine einzelne Datei einlesen und wie in find_unfinished() beschreiben.
    Gibt (entry, None) zurueck, oder (None, Fehlertext)."""
    p = Path(path)
    if not p.is_file():
        return None, "Datei nicht gefunden: %s" % p
    try:
        with open(p, "rb") as fh:
            res = pickle.load(fh)
    except Exception as exc:
        return None, "Datei konnte nicht gelesen werden (%r)." % (exc,)
    if not isinstance(res, dict) or kind_of(res) is None:
        return None, "Das sieht nicht wie ein Scan-Datensatz dieses Projekts aus."
    done, total = progress_of(res)
    entry = dict(path=str(p), name=p.name, results=res, kind=kind_of(res),
                 n_done=done, n_total=total,
                 saved_at=res.get("_checkpoint_saved_at") or p.stat().st_mtime)
    entry["label"] = describe(entry)
    return entry, None


# ----------------------------------------------------------------------
# Parameter aus dem Datensatz uebernehmen
# ----------------------------------------------------------------------
def apply_to_params(params, results):
    """Die von get_values() gelieferten Parameter aus dem Datensatz
    ueberschreiben - fuer JEDEN Schluessel, den beide kennen.

    Das ist die eigentliche Sicherung: die Dialogfelder sind beim
    Fortsetzen zwar gesperrt und zeigen die Werte aus der Datei, aber
    massgeblich ist diese Funktion. Selbst wenn ein Feld nicht sauber
    gesetzt werden konnte, rechnet der Scan mit den Werten des Datensatzes
    weiter - nie mit denen des Dialogs.

    `params` wird nicht veraendert; die angepasste Kopie wird zurueckgegeben.
    """
    p = dict(params)
    win = np.asarray(results.get("win_input_vals", []), dtype=float)
    wid = np.asarray(results.get("width_vals", []), dtype=float)
    if win.size >= 2 and "win_input_range" in p:
        p["win_input_range"] = (float(win[0]), float(win[-1]))
    if wid.size >= 2 and "width_range" in p:
        p["width_range"] = (float(wid[0]), float(wid[-1]))
    if win.size and "n_points" in p:
        p["n_points"] = int(win.size)
    if win.size and "n_win_input" in p:
        p["n_win_input"] = int(win.size)
    if wid.size and "n_width" in p:
        p["n_width"] = int(wid.size)

    for key in ("N_x", "N_y", "n_grid", "weighted_n_grid", "alpha", "r_bounds",
                "combo_lambda", "combo_percentile", "airy_scale_factor",
                "atom_temperature", "trap_freq_r", "atom_offset_x", "atom_offset_y"):
        if key in p and results.get(key) is not None:
            p[key] = results[key]

    # Fest-Amplituden-Dialoge geben amp_x/amp_y getrennt zurueck, der
    # Datensatz fuehrt sie zusammengehaengt als 'amps'.
    amps = results.get("amps")
    if amps is not None and ("amp_x" in p or "amp_y" in p):
        amps = np.asarray(amps, dtype=float)
        n_x = int(results.get("N_x", p.get("N_x", 0)) or 0)
        if n_x and amps.size >= n_x:
            if "amp_x" in p:
                p["amp_x"] = amps[:n_x]
            if "amp_y" in p:
                p["amp_y"] = amps[n_x:]
    if amps is not None and "amps" in p:
        p["amps"] = np.asarray(amps, dtype=float)
    return p


# ----------------------------------------------------------------------
# Qt-Teil - nur verfuegbar, wenn PyQt5 installiert ist
# ----------------------------------------------------------------------
try:
    from PyQt5.QtWidgets import (
        QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
        QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    )
except Exception:      # pragma: no cover - Qt fehlt
    QGroupBox = None


# Felder, die auch beim Fortsetzen aenderbar bleiben: sie aendern KEINE
# Zahl des Ergebnisses, nur wie schnell bzw. wie ausfuehrlich gerechnet wird.
KEEP_ENABLED = ("n_jobs", "force_cpu", "enable_perf_log", "auto_report")


if QGroupBox is not None:

    class ResumePickerGroup(QGroupBox):
        """Auswahlfeld "Neuer Scan / unfertigen Datensatz fortsetzen".

        Benutzung im Dialog:

            self.resume_group = ResumePickerGroup(
                DEFAULT_RESULTS_DIR, kind=resume_picker.KIND_WEIGHTED_AMP,
                on_change=self._on_resume_changed)
            main_layout.addWidget(self.resume_group)

        und in get_values() ganz am Ende:

            return self.resume_group.apply(params)
        """

        def __init__(self, results_dir, kind=None, on_change=None, parent=None,
                     extra_dirs=()):
            super().__init__("Fortsetzen (unfertige Datensaetze)", parent)
            self.results_dir = results_dir
            self.kind = kind
            self.extra_dirs = tuple(extra_dirs)
            self._on_change = on_change
            self._entries = []
            self._selected = None

            layout = QVBoxLayout()
            info = QLabel(
                "Ein abgebrochener Scan wird hier ausgewaehlt, nicht neu eingetippt.\n"
                "Alle Parameter kommen dann aus der Datei; die Felder darunter werden\n"
                "gesperrt und dienen nur noch der Anzeige.")
            info.setStyleSheet("font-style: italic;")
            layout.addWidget(info)

            row = QHBoxLayout()
            self.combo = QComboBox()
            self.combo.setToolTip(
                "Unfertige Datensaetze aus dem Results-Ordner, neueste zuerst.\n"
                "Unfertig heisst: als Zwischenstand gespeichert oder es fehlen\n"
                "noch Gitterpunkte.")
            self.combo.currentIndexChanged.connect(lambda _i: self._on_combo())
            row.addWidget(self.combo, 1)
            self.refresh_btn = QPushButton("Aktualisieren")
            self.refresh_btn.clicked.connect(lambda: self.refresh())
            row.addWidget(self.refresh_btn)
            self.other_btn = QPushButton("Andere Datei...")
            self.other_btn.clicked.connect(self._on_other)
            row.addWidget(self.other_btn)
            layout.addLayout(row)

            self.status = QLabel("")
            self.status.setWordWrap(True)
            self.status.setStyleSheet("color: gray;")
            layout.addWidget(self.status)

            self.setLayout(layout)
            self.refresh()

        # -- oeffentliche API ------------------------------------------
        def selected_entry(self):
            """Der gewaehlte Zwischenstand, oder None fuer "Neuer Scan"."""
            return self._selected

        def selected_path(self):
            return None if self._selected is None else self._selected["path"]

        def apply(self, params):
            """In get_values() aufrufen: ueberschreibt die Parameter aus dem
            gewaehlten Datensatz. Ohne Auswahl unveraendert durchgereicht."""
            if self._selected is None:
                return params
            p = apply_to_params(params, self._selected["results"])
            if "save_path" in p:
                p["save_path"] = self._selected["path"]
            p["resume_from"] = self._selected["path"]
            return p

        def refresh(self, keep_path=None):
            """Ordner neu auflisten (z.B. nach einem gerade beendeten Scan).

            Liest NUR Dateinamen und Zeitstempel - keine einzige .pkl wird
            dabei geoeffnet. Der Dialog erscheint dadurch sofort, auch bei
            vielen oder grossen Dateien und auch, wenn der Ordner in
            OneDrive liegt.
            """
            keep_path = keep_path or self.selected_path()
            self._entries = list_candidates(self.results_dir, kind=self.kind,
                                            extra_dirs=self.extra_dirs)
            self.combo.blockSignals(True)
            self.combo.clear()
            self.combo.addItem("— Neuer Scan —")
            for e in self._entries:
                self.combo.addItem(e["label"])
            self.combo.setCurrentIndex(0)
            if keep_path:
                for i, e in enumerate(self._entries):
                    if os.path.normcase(e["path"]) == os.path.normcase(keep_path):
                        self.combo.setCurrentIndex(i + 1)
                        break
            self.combo.blockSignals(False)
            self._on_combo()

        # -- intern ----------------------------------------------------
        def _on_combo(self):
            i = self.combo.currentIndex()
            if i <= 0 or i - 1 >= len(self._entries):
                self._selected = None
                n = len(self._entries)
                self.status.setText(
                    "Neuer Scan - alle Felder frei." if n else
                    "Neuer Scan - im Results-Ordner liegt gerade keine passende "
                    "Datei.")
                if self._on_change is not None:
                    self._on_change(None)
                return

            cand = self._entries[i - 1]
            # ERST JETZT wird die Datei gelesen - genau die eine, die der
            # Nutzer ausgewaehlt hat.
            if cand.get("results") is None:
                self.status.setText("Lese %s ..." % cand["name"])
                self.repaint()
                entry, err = load_entry(cand["path"])
                if entry is None:
                    self._reject(cand, "Datei passt nicht", err)
                    return
                if self.kind is not None and entry["kind"] != self.kind:
                    self._reject(cand, "Falsche Datensatz-Art",
                                 "Diese Datei ist ein %s.\nDieser Dialog kann nur %s "
                                 "fortsetzen."
                                 % (KIND_LABELS.get(entry["kind"], entry["kind"]),
                                    KIND_LABELS.get(self.kind, self.kind)))
                    return
                if not is_unfinished(entry["results"]):
                    self._reject(cand, "Datensatz ist fertig",
                                 "Dieser Datensatz ist vollstaendig - es gibt nichts "
                                 "fortzusetzen.")
                    return
                cand.update(entry)

            self._selected = cand
            res = cand["results"]
            self.status.setText(
                "Wird fortgesetzt: %d von %d Punkten fehlen noch. "
                "Gespeichert wird weiter in dieselbe Datei. "
                "(Art: %s, airy_scale_factor %s)"
                % (cand["n_total"] - cand["n_done"], cand["n_total"],
                   KIND_LABELS.get(cand["kind"], cand["kind"]),
                   res.get("airy_scale_factor", "1.19 (nicht gespeichert)")))
            if self._on_change is not None:
                self._on_change(self._selected)

        def _reject(self, cand, title, text):
            """Ausgewaehlte Datei taugt nicht: Meldung zeigen, Eintrag aus der
            Liste nehmen und zurueck auf "Neuer Scan"."""
            QMessageBox.warning(self, title, text)
            try:
                idx = self._entries.index(cand)
            except ValueError:
                idx = None
            self.combo.blockSignals(True)
            if idx is not None:
                self._entries.pop(idx)
                self.combo.removeItem(idx + 1)
            self.combo.setCurrentIndex(0)
            self.combo.blockSignals(False)
            self._selected = None
            self.status.setText("Neuer Scan - alle Felder frei.")
            if self._on_change is not None:
                self._on_change(None)

        def _on_other(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Unfertigen Datensatz waehlen", str(self.results_dir),
                "Pickle files (*.pkl)")
            if not path:
                return
            entry, err = load_entry(path)
            if entry is None:
                QMessageBox.warning(self, "Datei passt nicht", err)
                return
            if self.kind is not None and entry["kind"] != self.kind:
                QMessageBox.warning(
                    self, "Falsche Datensatz-Art",
                    "Diese Datei ist ein %s.\nDieser Dialog kann nur %s fortsetzen."
                    % (KIND_LABELS.get(entry["kind"], entry["kind"]),
                       KIND_LABELS.get(self.kind, self.kind)))
                return
            if not is_unfinished(entry["results"]):
                QMessageBox.warning(
                    self, "Datensatz ist fertig",
                    "Dieser Datensatz ist vollstaendig - es gibt nichts fortzusetzen.")
                return
            # in die Liste aufnehmen und auswaehlen
            self._entries.insert(0, entry)
            self.combo.blockSignals(True)
            self.combo.insertItem(1, entry["label"])
            self.combo.setCurrentIndex(1)
            self.combo.blockSignals(False)
            self._on_combo()


    def apply_display(dialog, results):
        """Die Felder des Dialogs mit den Werten aus dem Datensatz fuellen -
        damit man sieht, womit weitergerechnet wird.

        Rein kosmetisch: massgeblich ist apply_to_params() in get_values().
        Fehlt ein Feld in diesem Dialog, wird es einfach uebersprungen.
        Signale werden dabei geblockt, damit das Setzen keine
        Auto-Dateinamen o.ae. ausloest.
        """
        def _set(name, value, scale=1.0):
            w = getattr(dialog, name, None)
            if w is None or value is None:
                return
            try:
                w.blockSignals(True)
                w.setValue(type(w.value())(float(value) * scale))
            except Exception:
                pass
            finally:
                w.blockSignals(False)

        # Waist-Eingabe zuerst auf "vor der Linse (mm)" stellen - die Werte
        # im Datensatz sind immer win_input in Metern.
        combo = getattr(dialog, "waist_mode_combo", None)
        if combo is not None:
            try:
                combo.blockSignals(True)
                combo.setCurrentIndex(0)
                dialog._current_waist_mode = "win_input"
            except Exception:
                pass
            finally:
                combo.blockSignals(False)

        win = np.asarray(results.get("win_input_vals", []), dtype=float)
        wid = np.asarray(results.get("width_vals", []), dtype=float)
        if win.size >= 2:
            _set("win_input_min", win[0], 1e3)
            _set("win_input_max", win[-1], 1e3)
            _set("n_points", win.size)
        if wid.size >= 2:
            _set("width_min", wid[0], 1e-6)
            _set("width_max", wid[-1], 1e-6)

        _set("nx_spin", results.get("N_x"))
        _set("ny_spin", results.get("N_y"))
        _set("n_grid", results.get("n_grid"))
        _set("weighted_n_grid", results.get("weighted_n_grid"))
        _set("alpha", results.get("alpha"))
        _set("combo_lambda", results.get("combo_lambda"))
        _set("combo_percentile", results.get("combo_percentile"))
        r_bounds = results.get("r_bounds")
        if r_bounds is not None and len(r_bounds) == 2:
            _set("r_min", r_bounds[0])
            _set("r_max", r_bounds[1])
        _set("atom_temperature", results.get("atom_temperature"), 1e6)   # K -> µK
        _set("trap_freq_r", results.get("trap_freq_r"), 1e-3)            # Hz -> kHz

        # Airy-Skalenfaktor
        grp = getattr(dialog, "airy_group", None)
        factor = results.get("airy_scale_factor")
        if grp is not None and factor is not None:
            try:
                grp.set_value(float(factor))
            except Exception:
                pass
        spin = getattr(dialog, "airy_scale_factor", None)
        if grp is None and spin is not None and factor is not None:
            _set("airy_scale_factor", factor)

        # Amplituden (nur Fest-Amplituden-Dialoge)
        amps = results.get("amps")
        boxes_x = getattr(dialog, "amp_x_boxes", None)
        boxes_y = getattr(dialog, "amp_y_boxes", None)
        if amps is not None and boxes_x and boxes_y:
            amps = np.asarray(amps, dtype=float)
            n_x = len(boxes_x)
            for i, b in enumerate(boxes_x):
                if i < amps.size:
                    b.blockSignals(True); b.setValue(float(amps[i])); b.blockSignals(False)
            for i, b in enumerate(boxes_y):
                if n_x + i < amps.size:
                    b.blockSignals(True); b.setValue(float(amps[n_x + i])); b.blockSignals(False)


    def set_inputs_locked(dialog, locked, resume_group=None, keep=KEEP_ENABLED):
        """Alle Parameterfelder des Dialogs sperren bzw. wieder freigeben.

        Ausgenommen sind die Auswahlgruppe selbst und die Felder aus `keep`
        (n_jobs, GPU, Perf-Log, Auto-Report) - die aendern keine Zahl des
        Ergebnisses. Buttons bleiben ebenfalls bedienbar, damit Start/Cancel
        und "Browse..." erreichbar sind.
        """
        keep_widgets = set()
        for name in keep:
            w = getattr(dialog, name, None)
            if w is not None:
                keep_widgets.add(id(w))

        targets = []
        for w in dialog.findChildren((QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox)):
            if resume_group is not None and resume_group.isAncestorOf(w):
                continue
            if id(w) in keep_widgets:
                continue
            targets.append(w)

        if locked:
            # Vorherigen Zustand merken - manche Felder sind vom Dialog selbst
            # absichtlich deaktiviert (z.B. das airy_scale_factor-Feld, solange
            # eine benannte Konvention gewaehlt ist). Ohne dieses Merken waeren
            # sie nach einmal Fortsetzen-und-zurueck faelschlich bedienbar.
            dialog._resume_prev_enabled = {id(w): w.isEnabled() for w in targets}
            for w in targets:
                w.setEnabled(False)
        else:
            prev = getattr(dialog, "_resume_prev_enabled", None)
            for w in targets:
                w.setEnabled(True if prev is None else prev.get(id(w), True))
            dialog._resume_prev_enabled = None
