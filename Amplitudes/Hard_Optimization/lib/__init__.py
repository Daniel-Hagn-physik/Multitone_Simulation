"""
lib - Hilfsmodule fuer die run_*.py eine Ebene hoeher.

Diese Dateien werden NICHT direkt ausgefuehrt, sondern von den
Haupt-Skripten importiert:

    ../run_scan.py       ->  lib.paths, Optimierer, FixedScanPlotter, scan_checkpoint
    ../run_amp_scan.py   ->  lib.paths, Optimierer, AmplitudeScanPlotter, scan_checkpoint
    ../run_plots.py      ->  lib.paths, lib.scan_data, lib.report

Neu geschrieben beim Aufraeumen (2026-09-02) - vorher lagen alle Module
zusammen mit den ausfuehrbaren Skripten in einem flachen Ordner:

- paths.py       Ordner-Konstanten (Results/Bilder/Fit_Plots/Fit_Results),
                 der sys.path-Eintrag auf lib/ und die Angaben, die
                 Hard_Optimization von Weighted_Optimization unterscheiden
                 (Metrik-Familie, Dateinamens-Muster, Plot-Modul).
- scan_data.py   Laden/Speichern, Datensatz-Art erkennen, Score/Bestpunkt/
                 Region, verbotener Bereich (ueberlappende Eck-Spots).
- report.py      Plots (Vektor-PDF) und Markdown-Bericht, inkl. Talschnitt
                 und Gerade durch den Talpfad.

scan_data.py und report.py sind in Hard_Optimization/lib und
Weighted_Optimization/lib buchstabengleich; nur paths.py unterscheidet sich.

Unveraendert uebernommene Module (nur verschoben):

- multitone_flattop_optimizer.py         der eigentliche Optimierer
- multitone_amplitude_dependence_plots.py Scan-Plotter (+ neu: FixedScanPlotter)
- scan_checkpoint.py                     stuendliche Zwischenspeicherung
- resume_picker.py                       Fortsetzen abgebrochener Scans
- airy_scale.py                          Airy-Skalenfaktor-Definitionen
- perf_log.py                            Laufzeit-Protokoll
- use_torch.py                           optionale GPU-Beschleunigung
"""

from . import paths  # noqa: F401  - setzt den sys.path-Eintrag auf lib/
