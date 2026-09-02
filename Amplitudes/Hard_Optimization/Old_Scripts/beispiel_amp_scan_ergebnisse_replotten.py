"""
Beispiel: gespeicherte Amplituden-Scan-Rohdaten (aus "Results") erneut plotten
================================================================================

Zeigt, wie man ein von MultitoneFlatTopOptimizer.save_scan_amp_results()
gespeichertes Pickle (liegt im Ordner "Results", direkt neben diesem Skript)
lädt und daraus - OHNE den (potenziell langsamen) Scan neu zu berechnen -
neue Bilder erzeugt (landen automatisch im Ordner "Bilder").

Das ist genau der Punkt der Trennung von Rechnen (scan_win_width_amplitude_
dependence(), teuer) und Plotten (dieses Skript, billig): Achsenbeschriftung,
Farbschema, Legendengröße etc. lassen sich beliebig oft anpassen, ohne den
Scan jemals neu laufen zu lassen.

Vier Optionen werden hier konkret demonstriert (alle direkt als Parameter
von AmplitudeScanPlotter-Methoden, siehe deren Docstrings in
multitone_amplitude_dependence_plots.py für Details):

1. win_axis: Wechsel der win_input-Achse zwischen
     - "before_lens" (Default): Eingangs-Waist VOR der ersten Linse, in mm
       (so, wie tatsächlich gescannt wurde)
     - "after_lens": effektiver Waist an der Fokus-/Trap-Ebene NACH allen
       Linsen, in µm (über win_input_to_win() umgerechnet)
   Betrifft nur DIESE eine Achse (win_input) - width bleibt immer in MHz.

2. legend_fontsize: Schriftgröße der Legende in plot_dependence_cuts()
   (die einzige Plot-Methode hier mit einer echten Legende). None = Default
   aus SCAN2D_RC (12pt); größere Werte z.B. für Poster/Präsentationen.

3. MARK_BEST_POINT: True/False - markiert den markierten Punkt (siehe
   Option 4) in allen Plots mit einem Stern-Symbol, in plot_dependence_cuts()
   zusätzlich mit dünnen roten gestrichelten Linien (axvline) durch den
   Schnittpunkt in beiden Panels.

4. CUT_WIN_INPUT / CUT_WIDTH_MHZ: eigener Punkt für den Schnitt, statt
   des automatisch gefundenen besten Punkts (globales Minimum von
   alpha*Uniformity+(1-alpha)*Crosstalk). Die Regel ist einfach: ist ein
   Wert gesetzt, wird ER verwendet - sonst automatisch der beste Punkt (auf
   None lassen = Default-Verhalten, automatisch bester Punkt). Wird einer
   oder beide gesetzt, legt plot_dependence_cuts() seine Schnitte durch den
   nächstgelegenen Gitterpunkt zu diesen Werten (statt durch das globale
   Optimum) - UND plot_scan2d_combined() markiert in den Heatmaps denselben
   Punkt, sodass Schnitt und Markierung immer zusammenpassen. Ist nur einer
   der beiden Werte gesetzt, wird für die andere Achse weiterhin automatisch
   das globale Optimum verwendet. CUT_WIN_INPUT_AXIS legt fest, in welcher
   Einheit/Achse CUT_WIN_INPUT zu verstehen ist - "before_lens": mm VOR der
   ersten Linse (wie WIN_AXIS="before_lens"), oder "after_lens": µm effektiver
   Waist AN DER FOKUSEBENE nach allen Linsen (wie WIN_AXIS="after_lens") -
   unabhängig davon, welche Achse gerade für die Anzeige (WIN_AXIS) gewählt
   ist. Praktisch, wenn man in µm an der Fokusebene misst/denkt, statt den
   Wert erst von Hand in mm vor der Linse umrechnen zu müssen.

5. Ausreißer-Bereinigung (CLEAN_OUTLIERS/CLEAN_STRATEGY/CLEAN_BOUNDS): findet
   und entfernt/ersetzt automatisch Gitterpunkte, an denen r_x oder r_y exakt
   an einer r_bounds-Schranke liegt (siehe detect_amp_outliers()/
   clean_amp_scan_results() in multitone_amplitude_dependence_plots.py).
   Wirkt VOR dem Plotten auf die geladenen Rohdaten, betrifft also
   automatisch sowohl die Heatmaps als auch plot_dependence_cuts().
   WICHTIG: die UNTERE (r=0) und die OBERE (r=2) Schranke sind NICHT
   automatisch gleich zu behandeln! Ein einzelner/vereinzelter Treffer an
   einer Schranke ist meist ein fehlgeschlagener Optimierungslauf an genau
   diesem Gitterpunkt (physikalisch unplausibel -> guter Bereinigungs-
   Kandidat). Wird eine Schranke dagegen breit/häufig erreicht (viele
   zusammenhängende Punkte), ist das typischerweise KEIN Ausreißer, sondern
   eine echte Randsättigung (das unbeschränkte Optimum liegt dort außerhalb
   von r_bounds) - Bereinigen würde dann echte Information verfälschen. Das
   Skript druckt daher vor der Bereinigung über summarize_amp_bounds() aus,
   wie oft jede Schranke tatsächlich getroffen wird, und CLEAN_BOUNDS legt
   fest, welche Schranke(n) überhaupt als Ausreißer-Kandidat zählen (Default
   unten: nur die untere - siehe Kommentar bei CLEAN_BOUNDS).

6. ASK_BEFORE_SAVE: fragt vor dem Speichern interaktiv nach (per input(),
   übersprungen nur bei EOFError - z.B. bei automatisiertem Aufruf ganz ohne
   Eingabemöglichkeit; funktioniert auch in der PyCharm-"Run"-Konsole, wo
   sys.stdin.isatty() fälschlich False melden würde), damit man beim
   Ausprobieren nicht versehentlich den "Bilder"-Ordner mit Plots zumüllt,
   die man eigentlich nur mal kurz ansehen wollte.

WICHTIG - richtige Datei auswählen: Im Ordner "Results" liegen zwei ARTEN von
Pickle-Dateien, die NICHT austauschbar sind:
  - "scan_amp_data_...pkl"  <- DIESE hier verwenden (Amplituden-Scan, mit
    r_x_grid/r_y_grid - für jeden Gitterpunkt wurden die Amplituden extra
    optimiert)
  - "scan_data_...pkl"      <- NICHT für dieses Skript (einfacher Uniformity/
    Crosstalk-Scan bei FESTEN Amplituden, ohne r_x_grid/r_y_grid - dafür
    bräuchte man ScanPlotter aus multitone_flattop_scan_plots.py)
Eine Datei ohne "amp" im Namen hier zu laden, gibt einen klaren Fehler
(kein kryptisches KeyError mehr) - siehe AmplitudeScanPlotter.__init__().

Nutzung:
    python beispiel_amp_scan_ergebnisse_replotten.py
Ggf. vorher PKL_DATEI unten anpassen (Name einer "scan_amp_data_...pkl"-Datei
aus "Results" - der Dateiname verrät bereits Tonanzahl, Gitterauflösung und
Strahlprofil, z.B. "scan_amp_data_N3x4_15x15pts_Airy.pkl").
"""

from multitone_amplitude_dependence_plots import (
    AmplitudeScanPlotter,
    load_amp_scan_results,
    summarize_amp_bounds,
    detect_amp_outliers,
    clean_amp_scan_results,
    DEFAULT_RESULTS_DIR,
    DEFAULT_IMAGES_DIR,
)

# ======================================================================
# Konfiguration - hier anpassen
# ======================================================================

# Name (oder vollständiger Pfad) einer "scan_amp_data_...pkl"-Datei (NICHT
# "scan_data_...pkl", siehe Hinweis oben im Modul-Docstring) aus dem
# "Results"-Ordner. Nur ein Dateiname (kein Pfad) wird automatisch in
# DEFAULT_RESULTS_DIR gesucht, siehe load_amp_scan_results(). Falls die
# Datei nicht existiert, listet das Skript unten alle vorhandenen
# Amplituden-Scan-Dateien auf.
PKL_DATEI = r"C:\Users\Legion\PycharmProjects\Lern-repo\Optimierung Niklas+Claude\Results\scan_amp_data_N3x4_26x26pts_Airy.pkl"

# "before_lens" (mm vor der ersten Linse, Default) oder "after_lens" (µm,
# effektiver Waist an der Fokus-/Trap-Ebene nach allen Linsen).
WIN_AXIS = "before_lens"

# Legendengröße in plot_dependence_cuts(). None = Standardgröße (12pt).
LEGEND_FONTSIZE = 14

# Markierten Punkt (Stern-Symbol + in plot_dependence_cuts() dünne rote
# gestrichelte Linien durch den Schnittpunkt) in allen Plots anzeigen? Der
# markierte Punkt selbst ist CUT_WIN_INPUT/CUT_WIDTH_MHZ (falls gesetzt),
# sonst automatisch der beste Punkt (siehe Option 4 unten).
MARK_BEST_POINT = False

# Eigener Schnittpunkt statt automatisch gefundenem Optimum. CUT_WIN_INPUT
# auf None lassen -> automatisches globales Optimum für diese Achse
# (bisheriges Default-Verhalten); genauso für CUT_WIDTH_MHZ.
# CUT_WIN_INPUT_AXIS bestimmt die EINHEIT/ACHSE von CUT_WIN_INPUT:
#   "before_lens" -> CUT_WIN_INPUT ist in mm VOR der ersten Linse
#   "after_lens"  -> CUT_WIN_INPUT ist in µm effektiver Waist AN DER FOKUSEBENE
# (unabhängig von WIN_AXIS oben, das nur die Anzeige betrifft).
# Beispiel 1 (mm vor der Linse): CUT_WIN_INPUT_AXIS = "before_lens", CUT_WIN_INPUT = 1.1
# Beispiel 2 (µm an der Fokusebene): CUT_WIN_INPUT_AXIS = "after_lens", CUT_WIN_INPUT = 4.5
CUT_WIN_INPUT_AXIS = "before_lens"   # "before_lens" oder "after_lens"
CUT_WIN_INPUT = None      # mm (before_lens) oder µm (after_lens), oder None für "automatisch"
CUT_WIDTH_MHZ = None      # MHz, oder None für "automatisch"

# Ausreißer-Bereinigung: findet Gitterpunkte, an denen r_x/r_y exakt an einer
# r_bounds-Schranke liegt (typisch r=0 - physikalisch unplausibel, meist eine
# fehlgeschlagene Punkt-Optimierung an genau diesem Gitterpunkt) und
# behandelt sie VOR dem Plotten (wirkt dann auf Heatmaps UND Cuts gleichermaßen).
CLEAN_OUTLIERS = True
# "interpolate": Ausreißer durch Median der direkten Gitternachbarn ersetzen
#                ("auf die Umgebungswerte setzen")
# "drop_columns": komplette win_input-Spalte(n) mit Ausreißern weglassen
# "drop_rows":    komplette width-Zeile(n) mit Ausreißern weglassen
# "nan":          Ausreißer als Lücke (NaN) stehen lassen, nichts ersetzen/entfernen
CLEAN_STRATEGY = "interpolate"

# Welche r_bounds-Schranke(n) überhaupt als Ausreißer-Kandidat zählen -
# ("lower",), ("upper",) oder ("lower", "upper"). Default hier: NUR die
# untere Schranke (r=0), da r=0 für ein einzelnes Gitterfeld physikalisch
# unplausibel ist. Die obere Schranke (r=2) wird in der Praxis oft an einem
# ganzen Bereich erreicht (echte Randsättigung, siehe Hinweis 5 oben) - das
# Skript druckt vor der Bereinigung über summarize_amp_bounds() aus, wie oft
# jede Schranke tatsächlich getroffen wird; zeigt sich dort, dass auch die
# untere Schranke breit/zusammenhängend auftritt (nicht nur vereinzelt),
# sollte man CLEAN_BOUNDS entsprechend anpassen (z.B. auf () setzen, um gar
# nichts automatisch zu bereinigen).
CLEAN_BOUNDS = ("lower",)

# Bilder anzeigen (blockierendes Matplotlib-Fenster)?
SHOW = True

# Bilder speichern? Falls ASK_BEFORE_SAVE True ist (Default), wird SAVE nur
# als Vorschlagswert für die interaktive Abfrage verwendet - die tatsächliche
# Entscheidung fällt dann bei jedem Lauf per Eingabeaufforderung (praktisch
# beim Ausprobieren/Herumspielen mit den Optionen oben, um den "Bilder"-
# Ordner nicht mit Testplots zuzumüllen). Auf False setzen, um nie zu
# speichern, unabhängig von der Abfrage.
SAVE = True
ASK_BEFORE_SAVE = True


def main():
    try:
        results = load_amp_scan_results(PKL_DATEI)
    except FileNotFoundError:
        vorhandene = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob("scan_amp_data_*.pkl"))
        andere = sorted(p.name for p in DEFAULT_RESULTS_DIR.glob("*.pkl")
                         if not p.name.startswith("scan_amp_data_"))
        print(f"'{PKL_DATEI}' wurde weder im aktuellen Ordner noch in "
              f"'{DEFAULT_RESULTS_DIR}' gefunden.")
        if vorhandene:
            print("Vorhandene Amplituden-Scan-Dateien ('scan_amp_data_...pkl') in Results:")
            for name in vorhandene:
                print(f"  - {name}")
            print("-> PKL_DATEI oben im Skript entsprechend anpassen.")
        else:
            print(f"Der Ordner '{DEFAULT_RESULTS_DIR}' enthält aktuell keine "
                  f"'scan_amp_data_...pkl'-Datei - zuerst einen Amplituden-Scan "
                  f"(scan_win_width_amplitude_dependence()) laufen lassen und mit "
                  f"save_scan_amp_results() speichern.")
        if andere:
            print("(Andere .pkl-Dateien in Results, NICHT für dieses Skript geeignet: "
                  + ", ".join(andere) + ")")
        return

    n_win = len(results["win_input_vals"])
    n_width = len(results["width_vals"])
    profile = results.get("profile", "unbekannt")
    print(f"Geladen: N_x={results['N_x']}, N_y={results['N_y']}, "
          f"{n_win}x{n_width} Gitterpunkte, Profil={profile}")

    # Erst mal unabhängig von CLEAN_OUTLIERS anzeigen, wie oft jede Schranke
    # (unten r=0, oben r=2) tatsächlich getroffen wird - Grundlage, um zu
    # beurteilen, ob ein Treffer ein isolierter Ausreißer oder eine echte
    # Randsättigung ist (siehe Hinweis 5 im Modul-Docstring oben).
    summarize_amp_bounds(results)

    # Ausreißer (Default: nur an der unteren Schranke r=0, siehe CLEAN_BOUNDS
    # oben) VOR dem Plotten bereinigen - wirkt auf ALLE nachfolgenden Plots
    # (Heatmaps UND Dependence-Cuts), da beide nur mit dem hier ersetzten
    # "results" arbeiten.
    if CLEAN_OUTLIERS:
        outlier_mask = detect_amp_outliers(results, bounds=CLEAN_BOUNDS)
        n_outliers = int(outlier_mask.sum())
        if n_outliers:
            print(f"{n_outliers} Ausreißer-Punkt(e) gefunden (r_x/r_y an Schranke(n) "
                  f"{CLEAN_BOUNDS}) - bereinige mit CLEAN_STRATEGY='{CLEAN_STRATEGY}'.")
            results = clean_amp_scan_results(results, mask=outlier_mask, strategy=CLEAN_STRATEGY, verbose=False)
        else:
            print(f"Keine Ausreißer gefunden (kein r_x/r_y exakt an Schranke(n) {CLEAN_BOUNDS}).")

    try:
        # out_dir nicht angegeben -> automatisch DEFAULT_IMAGES_DIR ("Bilder")
        plotter = AmplitudeScanPlotter(results)
    except ValueError as exc:
        print(f"'{PKL_DATEI}' konnte nicht als Amplituden-Scan-Ergebnis geladen werden:\n{exc}")
        return

    print(f"Bilder werden gespeichert in: {DEFAULT_IMAGES_DIR}")

    # CUT_WIN_INPUT (mm oder µm, je nach CUT_WIN_INPUT_AXIS) und CUT_WIDTH_MHZ
    # (MHz) in SI-Einheiten (Meter/Hz) umrechnen, wie es die Plot-Methoden
    # erwarten. None bleibt None -> automatisches globales Optimum.
    win_input_fixed = None if CUT_WIN_INPUT is None else CUT_WIN_INPUT * (1e-3 if CUT_WIN_INPUT_AXIS == "before_lens" else 1e-6)
    width_fixed = None if CUT_WIDTH_MHZ is None else CUT_WIDTH_MHZ * 1e6
    if win_input_fixed is not None or width_fixed is not None:
        unit = "mm vor der Linse" if CUT_WIN_INPUT_AXIS == "before_lens" else "µm an der Fokusebene"
        print(f"Eigener Schnittpunkt: CUT_WIN_INPUT={CUT_WIN_INPUT} ({unit}), "
              f"width_fixed={CUT_WIDTH_MHZ} MHz (wird auf den nächstgelegenen "
              f"Gitterpunkt gerundet; nicht gesetzte Achse bleibt automatisch).")

    # Vor dem Speichern nachfragen, ob überhaupt gespeichert werden soll -
    # praktisch beim Ausprobieren/Herumspielen mit den Optionen oben, damit
    # der "Bilder"-Ordner nicht mit Testplots zugemüllt wird. Gleiches Muster
    # wie oben: input() mit EOFError-Fallback statt sys.stdin.isatty()-Check.
    save = SAVE
    if SAVE and ASK_BEFORE_SAVE:
        try:
            antwort = input("Bilder in 'Bilder' speichern? [y/N]: ").strip().lower()
            save = antwort in ("y", "yes", "j", "ja")
            if not save:
                print("-> Bilder werden NICHT gespeichert (nur angezeigt, falls SHOW=True).")
        except EOFError:
            print("(ASK_BEFORE_SAVE=True, aber keine Eingabe möglich (kein Terminal) - "
                  "verwende SAVE={} wie konfiguriert.)".format(SAVE))

    # 4 Heatmaps (Uniformity, Crosstalk, r_x, r_y) nebeneinander, mit
    # gewählter win_input-Achse und markiertem Punkt (automatisches Optimum
    # oder CUT_WIN_INPUT/CUT_WIDTH_MHZ):
    plotter.plot_scan2d_combined(
        show=SHOW, save=save,
        win_axis=WIN_AXIS,
        mark_best_point=MARK_BEST_POINT,
        win_input_fixed=win_input_fixed,
        width_fixed=width_fixed,
        win_input_fixed_axis=CUT_WIN_INPUT_AXIS,
    )

    # Abhängigkeits-Schnitte (r_x/r_y als Kurve über win_input bzw. width),
    # mit gewählter win_input-Achse, Legendengröße, markiertem Punkt UND dem
    # eigenen Schnittpunkt (falls gesetzt) statt des automatischen Optimums:
    plotter.plot_dependence_cuts(
        show=SHOW, save=save,
        win_axis=WIN_AXIS,
        legend_fontsize=LEGEND_FONTSIZE,
        mark_best_point=MARK_BEST_POINT,
        win_input_fixed=win_input_fixed,
        width_fixed=width_fixed,
        win_input_fixed_axis=CUT_WIN_INPUT_AXIS,
    )

    # Einzelne Heatmaps sind genauso ansteuerbar, z.B. nur die Uniformity
    # mit anderem Farbschema und ohne Sternmarkierung:
    # plotter.plot_scan2d_uniformity(show=SHOW, save=save, cmap="viridis_r",
    #                                 win_axis=WIN_AXIS, mark_best_point=False)


if __name__ == "__main__":
    main()
