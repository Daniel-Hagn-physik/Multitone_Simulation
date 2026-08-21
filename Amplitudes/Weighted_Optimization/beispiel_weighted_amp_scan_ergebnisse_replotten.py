"""
Beispiel: gespeicherte GEWICHTETE Amplituden-Scan-Rohdaten (aus "Results")
erneut plotten
================================================================================

Port von `beispiel_amp_scan_ergebnisse_replotten.py` (Original-Ordner) fuer
den ATOM-GEWICHTETEN Amplituden-Scan. Zeigt, wie man ein von
`MultitoneFlatTopOptimizer.save_scan_amp_results_weighted()`
(weighted_amp_scan_methods.py) gespeichertes Pickle (liegt im Ordner
"Results", direkt neben diesem Skript) laedt und daraus - OHNE den
(potenziell langsamen) Scan neu zu berechnen - neue Bilder erzeugt (landen
automatisch im Ordner "Bilder"). Alle Optionen/Funktionen sind WORTWOERTLICH
identisch zum Original uebernommen (nur der Import-Pfad zeigt jetzt auf
`weighted_multitone_amplitude_dependence_plots` statt
`multitone_amplitude_dependence_plots"), die Bereinigungs-/Achsen-/
Markierungs-Logik selbst kennt gar keinen Unterschied zwischen hart maskiert
und atom-gewichtet - sie arbeitet immer nur auf r_x_grid/r_y_grid (die
Amplituden-Parametrisierung ist unveraendert).

Das ist genau der Punkt der Trennung von Rechnen (scan_win_width_amplitude_
dependence_weighted(), teuer) und Plotten (dieses Skript, billig): Achsen-
beschriftung, Farbschema, Legendengroesse etc. lassen sich beliebig oft
anpassen, ohne den Scan jemals neu laufen zu lassen.

Fuenf Optionen werden hier konkret demonstriert (alle direkt als Parameter
von AmplitudeScanPlotter-Methoden, siehe deren Docstrings in
weighted_multitone_amplitude_dependence_plots.py fuer Details):

1. win_axis: Wechsel der win_input-Achse zwischen
     - "before_lens" (Default): Eingangs-Waist VOR der ersten Linse, in mm
       (so, wie tatsaechlich gescannt wurde)
     - "after_lens": effektiver Waist an der Fokus-/Trap-Ebene NACH allen
       Linsen, in µm (ueber win_input_to_win() umgerechnet)
   Betrifft nur DIESE eine Achse (win_input) - width bleibt immer in MHz.

2. legend_fontsize: Schriftgroesse der Legende in plot_dependence_cuts()
   (die einzige Plot-Methode hier mit einer echten Legende). None = Default
   aus SCAN2D_RC (12pt); groessere Werte z.B. fuer Poster/Praesentationen.

3. MARK_BEST_POINT: True/False - markiert den markierten Punkt (siehe
   Option 4) in allen Plots mit einem Stern-Symbol, in plot_dependence_cuts()
   zusaetzlich mit duennen roten gestrichelten Linien (axvline) durch den
   Schnittpunkt in beiden Panels.

4. CUT_WIN_INPUT / CUT_WIDTH_MHZ: eigener Punkt fuer den Schnitt, statt
   des automatisch gefundenen besten Punkts (globales Minimum von
   alpha*uniformity_weighted+(1-alpha)*eta_weighted). Die Regel ist einfach:
   ist ein Wert gesetzt, wird ER verwendet - sonst automatisch der beste
   Punkt (auf None lassen = Default-Verhalten, automatisch bester Punkt).
   Wird einer oder beide gesetzt, legt plot_dependence_cuts() seine Schnitte
   durch den naechstgelegenen Gitterpunkt zu diesen Werten (statt durch das
   globale Optimum) - UND plot_scan2d_combined() markiert in den Heatmaps
   denselben Punkt, sodass Schnitt und Markierung immer zusammenpassen. Ist
   nur einer der beiden Werte gesetzt, wird fuer die andere Achse weiterhin
   automatisch das globale Optimum verwendet. CUT_WIN_INPUT_AXIS legt fest,
   in welcher Einheit/Achse CUT_WIN_INPUT zu verstehen ist - "before_lens":
   mm VOR der ersten Linse (wie WIN_AXIS="before_lens"), oder "after_lens":
   µm effektiver Waist AN DER FOKUSEBENE nach allen Linsen (wie
   WIN_AXIS="after_lens") - unabhaengig davon, welche Achse gerade fuer die
   Anzeige (WIN_AXIS) gewaehlt ist.

5. Ausreisser-Bereinigung (CLEAN_OUTLIERS/CLEAN_STRATEGY/CLEAN_BOUNDS):
   findet und entfernt/ersetzt automatisch Gitterpunkte, an denen r_x oder
   r_y exakt an einer r_bounds-Schranke liegt (siehe detect_amp_outliers()/
   clean_amp_scan_results() in weighted_multitone_amplitude_dependence_plots.py).
   Wirkt VOR dem Plotten auf die geladenen Rohdaten, betrifft also
   automatisch sowohl die Heatmaps als auch plot_dependence_cuts(). WICHTIG:
   die UNTERE (r=0) und die OBERE (r=2) Schranke sind NICHT automatisch
   gleich zu behandeln - siehe Erklaerung im Original-Skript/Projekt-Doc.
   Das Skript druckt daher vor der Bereinigung ueber summarize_amp_bounds()
   aus, wie oft jede Schranke tatsaechlich getroffen wird.

6. ASK_BEFORE_SAVE: fragt vor dem Speichern interaktiv nach (per input(),
   uebersprungen nur bei EOFError), damit man beim Ausprobieren nicht
   versehentlich den "Bilder"-Ordner mit Plots zumuellt, die man eigentlich
   nur mal kurz ansehen wollte.

7. REFINE_DISCONTINUITIES (NEU, 2026-08-21, auf User-Wunsch - Post-hoc-
   Nachoptimierung auffaelliger r_x/r_y-Sprungstellen): findet Gitterpunkte,
   an denen (r_x, r_y) sprunghaft von den direkten Nachbarn abweicht
   (detect_amp_discontinuities() in weighted_multitone_amplitude_dependence_
   plots.py - anders als CLEAN_OUTLIERS oben, das NUR exakte r_bounds-Treffer
   wie r=0/r=2 findet, erkennt das auch Spruenge INNERHALB des erlaubten
   Bereichs) und rechnet NUR diese Punkte mit MEHREREN festen Startpunkten
   neu durch (refine_scan_amp_results_weighted() in weighted_amp_scan_
   methods.py), statt sich auf die sequentielle Warm-Start-Kette des
   urspruenglichen Scans zu verlassen. Ein Punkt wird nur ersetzt, wenn die
   Nachrechnung tatsaechlich einen messbar besseren kombinierten Zielwert
   findet - HINTERGRUND siehe Diagnose vom 2026-08-21 im Projekt-Status-Doc:
   ein Teil der beobachteten Spruenge war ein echtes Warm-Start-Nelder-Mead-
   Artefakt (die Kette lief in ein schlechteres lokales statt das globale
   Optimum), ein anderer Teil war KEIN Artefakt, sondern eine reale
   Eigenschaft der gewichteten Metrik (mehrere fast gleich gute, benachbarte
   lokale Optima, die abwechselnd global gewinnen) - Punkte dieser zweiten
   Art bleiben unveraendert, weil keine Nachrechnung dort etwas Besseres
   findet. Standardmaessig AUS (kostet zusaetzliche Rechenzeit: pro
   auffaelligem Punkt werden 6 statt 1 Nelder-Mead-Laeufe durchgefuehrt) -
   bei Aktivierung wird dafuer eine neue MultitoneFlatTopOptimizer-Instanz
   aus den in der .pkl-Datei gespeicherten physikalischen Parametern
   rekonstruiert (siehe Hinweis im Skript unten zu weighted_n_grid/
   weighted_n_sigma, die NICHT mitgespeichert werden).

WICHTIG - richtige Datei auswaehlen: Im Ordner "Results" (neben diesem
Skript, also Weighted_Optimization\\Results) liegen ZWEI Arten von
Pickle-Dateien, die NICHT austauschbar sind:
  - "scan_amp_data_weighted_...pkl"  <- DIESE hier verwenden (gewichteter
    Amplituden-Scan, mit r_x_grid/r_y_grid - fuer jeden Gitterpunkt wurden
    die Amplituden unter der gewichteten Metrik extra optimiert)
  - "scan_data_weighted_...pkl"      <- NICHT fuer dieses Skript (einfacher
    uniformity_weighted/eta_weighted-Scan bei FESTEN Amplituden, ohne
    r_x_grid/r_y_grid - dafuer WeightedFixedScanPlotter verwenden, siehe
    weighted_multitone_amplitude_dependence_plots.py)
  - Un-gewichtete Dateien OHNE "_weighted" im Namen gehoeren gar nicht in
    diesen Ordner-Kontext (das sind die Original-Scans aus dem Elternordner).
Eine Datei ohne "amp" im Namen hier zu laden, gibt einen klaren Fehler (kein
kryptisches KeyError mehr) - siehe AmplitudeScanPlotter.__init__().

Nutzung:
    python beispiel_weighted_amp_scan_ergebnisse_replotten.py
Ggf. vorher PKL_DATEI unten anpassen (Name einer
"scan_amp_data_weighted_...pkl"-Datei aus "Results" - der Dateiname verraet
bereits Tonanzahl, Gitterausloesung und Strahlprofil, z.B.
"scan_amp_data_weighted_N3x4_15x15pts_Airy.pkl").
"""

from weighted_multitone_amplitude_dependence_plots import (
    AmplitudeScanPlotter,
    load_amp_scan_results,
    summarize_amp_bounds,
    detect_amp_outliers,
    detect_amp_discontinuities,
    clean_amp_scan_results,
    DEFAULT_RESULTS_DIR,
    DEFAULT_IMAGES_DIR,
)
from weighted_multitone_flattop_optimizer import MultitoneFlatTopOptimizer
import weighted_amp_scan_methods  # nur Import noetig - patcht refine_scan_amp_results_weighted() auf die Klasse

# ======================================================================
# Konfiguration - hier anpassen
# ======================================================================

# Name (oder vollstaendiger Pfad) einer "scan_amp_data_weighted_...pkl"-Datei
# (NICHT "scan_data_weighted_...pkl", siehe Hinweis oben im Modul-Docstring)
# aus dem "Results"-Ordner NEBEN diesem Skript (Weighted_Optimization\Results).
# Nur ein Dateiname (kein Pfad) wird automatisch in DEFAULT_RESULTS_DIR
# gesucht, siehe load_amp_scan_results(). Falls die Datei nicht existiert,
# listet das Skript unten alle vorhandenen gewichteten Amplituden-Scan-
# Dateien auf.
PKL_DATEI = r"C:\Users\Legion\PycharmProjects\Lern-repo\Optimierung Niklas+Claude\Weighted_Optimization\Results\scan_amp_data_weighted_N3x4_15x15pts_Airy.pkl"

# "before_lens" (mm vor der ersten Linse, Default) oder "after_lens" (µm,
# effektiver Waist an der Fokus-/Trap-Ebene nach allen Linsen).
WIN_AXIS = "before_lens"

# Legendengroesse in plot_dependence_cuts(). None = Standardgroesse (12pt).
LEGEND_FONTSIZE = 14

# Markierten Punkt (Stern-Symbol + in plot_dependence_cuts() duenne rote
# gestrichelte Linien durch den Schnittpunkt) in allen Plots anzeigen? Der
# markierte Punkt selbst ist CUT_WIN_INPUT/CUT_WIDTH_MHZ (falls gesetzt),
# sonst automatisch der beste Punkt (siehe Option 4 unten).
MARK_BEST_POINT = False

# Eigener Schnittpunkt statt automatisch gefundenem Optimum. CUT_WIN_INPUT
# auf None lassen -> automatisches globales Optimum fuer diese Achse
# (bisheriges Default-Verhalten); genauso fuer CUT_WIDTH_MHZ.
# CUT_WIN_INPUT_AXIS bestimmt die EINHEIT/ACHSE von CUT_WIN_INPUT:
#   "before_lens" -> CUT_WIN_INPUT ist in mm VOR der ersten Linse
#   "after_lens"  -> CUT_WIN_INPUT ist in µm effektiver Waist AN DER FOKUSEBENE
# (unabhaengig von WIN_AXIS oben, das nur die Anzeige betrifft).
CUT_WIN_INPUT_AXIS = "before_lens"   # "before_lens" oder "after_lens"
CUT_WIN_INPUT = None      # mm (before_lens) oder µm (after_lens), oder None fuer "automatisch"
CUT_WIDTH_MHZ = None      # MHz, oder None fuer "automatisch"

# Ausreisser-Bereinigung: findet Gitterpunkte, an denen r_x/r_y exakt an
# einer r_bounds-Schranke liegt und behandelt sie VOR dem Plotten (wirkt
# dann auf Heatmaps UND Cuts gleichermassen).
CLEAN_OUTLIERS = True
# "interpolate": Ausreisser durch Median der direkten Gitternachbarn ersetzen
# "drop_columns": komplette win_input-Spalte(n) mit Ausreissern weglassen
# "drop_rows":    komplette width-Zeile(n) mit Ausreissern weglassen
# "nan":          Ausreisser als Luecke (NaN) stehen lassen
CLEAN_STRATEGY = "interpolate"

# Welche r_bounds-Schranke(n) ueberhaupt als Ausreisser-Kandidat zaehlen -
# ("lower",), ("upper",) oder ("lower", "upper"). Default hier (wie im
# Original): NUR die untere Schranke (r=0).
CLEAN_BOUNDS = ("lower",)

# Post-hoc-Nachoptimierung auffaelliger r_x/r_y-Spruenge INNERHALB der
# r_bounds (nicht nur exakte Schranken-Treffer wie CLEAN_OUTLIERS oben) -
# siehe Option 7 im Modul-Docstring. Laeuft VOR CLEAN_OUTLIERS, da sie die
# echte Zielfunktion neu auswertet statt nur Nachbarwerte zu mitteln.
REFINE_DISCONTINUITIES = False
REFINE_Z_THRESH = 3.5   # Schwelle fuer detect_amp_discontinuities() - hoeher = weniger/nur staerkere Kandidaten
REFINE_TOL_REL = 1e-3   # Mindest-relative Verbesserung des kombinierten Zielwerts, um einen Punkt zu uebernehmen

# Bilder anzeigen (blockierendes Matplotlib-Fenster)?
SHOW = True

# Bilder speichern? Falls ASK_BEFORE_SAVE True ist (Default), wird SAVE nur
# als Vorschlagswert fuer die interaktive Abfrage verwendet.
SAVE = True
ASK_BEFORE_SAVE = True


def main():
    try:
        results = load_amp_scan_results(PKL_DATEI)
    except FileNotFoundError:
        vorhandene = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob("scan_amp_data_weighted_*.pkl"))
        andere = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob("*.pkl")
                         if not p.name.startswith("scan_amp_data_weighted_"))
        print(f"'{PKL_DATEI}' wurde weder im aktuellen Ordner noch in "
              f"'{DEFAULT_RESULTS_DIR}' gefunden.")
        if vorhandene:
            print("Vorhandene gewichtete Amplituden-Scan-Dateien ('scan_amp_data_weighted_...pkl') in Results:")
            for name in vorhandene:
                print(f"  - {name}")
            print("-> PKL_DATEI oben im Skript entsprechend anpassen.")
        else:
            print(f"Der Ordner '{DEFAULT_RESULTS_DIR}' enthaelt aktuell keine "
                  f"'scan_amp_data_weighted_...pkl'-Datei - zuerst einen gewichteten Amplituden-"
                  f"Scan (scan_win_width_amplitude_dependence_weighted(), siehe "
                  f"weighted_amp_scan_methods.py) laufen lassen und mit "
                  f"save_scan_amp_results_weighted() speichern.")
        if andere:
            print("(Andere .pkl-Dateien in Results, NICHT fuer dieses Skript geeignet: "
                  + ", ".join(andere) + ")")
        return

    n_win = len(results["win_input_vals"])
    n_width = len(results["width_vals"])
    profile = results.get("profile", "unbekannt")
    sigma_atom = results.get("sigma_atom")
    sigma_info = f", sigma_atom={sigma_atom * 1e9:.1f} nm" if sigma_atom else ""
    print(f"Geladen: N_x={results['N_x']}, N_y={results['N_y']}, "
          f"{n_win}x{n_width} Gitterpunkte, Profil={profile}{sigma_info}")

    # Erst mal unabhaengig von CLEAN_OUTLIERS anzeigen, wie oft jede Schranke
    # (unten r=0, oben r=2) tatsaechlich getroffen wird.
    summarize_amp_bounds(results)

    if REFINE_DISCONTINUITIES:
        disc_mask, _jump_z = detect_amp_discontinuities(results, z_thresh=REFINE_Z_THRESH)
        n_flagged = int(disc_mask.sum())
        print(f"Sprung-Erkennung (detect_amp_discontinuities, z_thresh={REFINE_Z_THRESH}): "
              f"{n_flagged} auffaellige Punkt(e).")
        if n_flagged:
            print("HINWEIS: weighted_n_grid/weighted_n_sigma stehen NICHT in der .pkl-Datei "
                  "und werden hier mit den Klassen-Defaults rekonstruiert (siehe DEFAULTS in "
                  "weighted_multitone_flattop_optimizer.py). Fuer eine exakte Reproduktion des "
                  "Original-Scans ggf. hier unten explizit denselben Wert wie beim Scan angeben.")
            opt = MultitoneFlatTopOptimizer(
                out_dir=".", N_x=results["N_x"], N_y=results["N_y"],
                f1=results["f1"], f2=results["f2"], fLO=results["fLO"],
                lambda_opt=results["lambda_opt"], theta_max=results["theta_max"],
                f_band=results["f_band"], profile=results.get("profile", "airy"),
                atom_mass=results["atom_mass"], atom_temperature=results["atom_temperature"],
                trap_freq_r=results["trap_freq_r"],
                # weighted_n_grid=241, weighted_n_sigma=6,  # <- bei Bedarf hier explizit setzen
            )
            results, refine_report = opt.refine_scan_amp_results_weighted(
                results, mask=disc_mask, z_thresh=REFINE_Z_THRESH, tol_rel=REFINE_TOL_REL,
            )
        else:
            print("-> keine Nachoptimierung noetig.")

    if CLEAN_OUTLIERS:
        outlier_mask = detect_amp_outliers(results, bounds=CLEAN_BOUNDS)
        n_outliers = int(outlier_mask.sum())
        if n_outliers:
            print(f"{n_outliers} Ausreisser-Punkt(e) gefunden (r_x/r_y an Schranke(n) "
                  f"{CLEAN_BOUNDS}) - bereinige mit CLEAN_STRATEGY='{CLEAN_STRATEGY}'.")
            results = clean_amp_scan_results(results, mask=outlier_mask, strategy=CLEAN_STRATEGY, verbose=False)
        else:
            print(f"Keine Ausreisser gefunden (kein r_x/r_y exakt an Schranke(n) {CLEAN_BOUNDS}).")

    try:
        # out_dir nicht angegeben -> automatisch DEFAULT_IMAGES_DIR ("Bilder")
        plotter = AmplitudeScanPlotter(results)
    except ValueError as exc:
        print(f"'{PKL_DATEI}' konnte nicht als Amplituden-Scan-Ergebnis geladen werden:\n{exc}")
        return

    print(f"Bilder werden gespeichert in: {DEFAULT_IMAGES_DIR}")

    win_input_fixed = None if CUT_WIN_INPUT is None else CUT_WIN_INPUT * (1e-3 if CUT_WIN_INPUT_AXIS == "before_lens" else 1e-6)
    width_fixed = None if CUT_WIDTH_MHZ is None else CUT_WIDTH_MHZ * 1e6
    if win_input_fixed is not None or width_fixed is not None:
        unit = "mm vor der Linse" if CUT_WIN_INPUT_AXIS == "before_lens" else "µm an der Fokusebene"
        print(f"Eigener Schnittpunkt: CUT_WIN_INPUT={CUT_WIN_INPUT} ({unit}), "
              f"width_fixed={CUT_WIDTH_MHZ} MHz (wird auf den naechstgelegenen "
              f"Gitterpunkt gerundet; nicht gesetzte Achse bleibt automatisch).")

    save = SAVE
    if SAVE and ASK_BEFORE_SAVE:
        try:
            antwort = input("Bilder in 'Bilder' speichern? [y/N]: ").strip().lower()
            save = antwort in ("y", "yes", "j", "ja")
            if not save:
                print("-> Bilder werden NICHT gespeichert (nur angezeigt, falls SHOW=True).")
        except EOFError:
            print("(ASK_BEFORE_SAVE=True, aber keine Eingabe moeglich (kein Terminal) - "
                  "verwende SAVE={} wie konfiguriert.)".format(SAVE))

    # Bis zu 4 Heatmaps (Uniformity_w, Crosstalk_w, r_x, r_y) nebeneinander,
    # mit gewaehlter win_input-Achse und markiertem Punkt:
    plotter.plot_scan2d_combined(
        show=SHOW, save=save,
        win_axis=WIN_AXIS,
        mark_best_point=MARK_BEST_POINT,
        win_input_fixed=win_input_fixed,
        width_fixed=width_fixed,
        win_input_fixed_axis=CUT_WIN_INPUT_AXIS,
    )

    # Abhaengigkeits-Schnitte (r_x/r_y als Kurve ueber win_input bzw. width):
    plotter.plot_dependence_cuts(
        show=SHOW, save=save,
        win_axis=WIN_AXIS,
        legend_fontsize=LEGEND_FONTSIZE,
        mark_best_point=MARK_BEST_POINT,
        win_input_fixed=win_input_fixed,
        width_fixed=width_fixed,
        win_input_fixed_axis=CUT_WIN_INPUT_AXIS,
    )

    # Einzelne Heatmaps sind genauso ansteuerbar, z.B. nur die gewichtete
    # Uniformity mit anderem Farbschema und ohne Sternmarkierung:
    # plotter.plot_scan2d_uniformity_weighted(show=SHOW, save=save, cmap="viridis_r",
    #                                          win_axis=WIN_AXIS, mark_best_point=False)


if __name__ == "__main__":
    main()
