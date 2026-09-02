"""
run_all_fits.py
================
EIN Klick-and-go-Master-Skript: liest BEIDE Scan-pkl-Dateien ein (den
AMPLITUDEN-optimierten Scan fuer fit_central_amplitudes.py UND den
FEST-Amplitude-Scan fuer fit_waist_width_relation.py), ruft beide
Fit-Pipelines PLUS die Uniformity-Valley-Uebersicht
(fit_uniformity_valley_overview.py) nacheinander auf und sammelt am Ende
alle erzeugten Dateien (LaTeX-taugliche PDF-Plots in Fit_Plots/mm_waist
bzw. Fit_Plots/um_waist, Formel-Dokumente + stripe_domain_mask.npz in
Fit_Results) in einer Zusammenfassung.

Wieso ein eigenes Skript statt `python fit_central_amplitudes.py && python
fit_waist_width_relation.py`? Weil beide Skripte bisher interaktiv nach
"Speichern? [y/N]" fragen (ASK_BEFORE_SAVE=True) - fuer "ein Klick and go"
laeuft dieses Skript beide NICHT-interaktiv (ask_before_save=False, direkt
speichern) und buendelt zusaetzlich die beiden grafischen Konfigurations-
Optionen, die auf User-Wunsch zentral einstellbar sein sollen:

- LEGEND_FONTSIZE: Schriftgroesse der Legenden in allen Plots beider Skripte.
- DRAW_BEST_POINT: zeichnet in allen Plots beider Skripte automatisch den
  besten gefundenen Punkt ein (roter Stern) - OHNE dass man selbst
  Koordinaten angeben muss:
    - fit_central_amplitudes.py (Amplituden-optimierter Scan): globales
      Minimum von alpha*Uniformity + (1-alpha)*Crosstalk UEBER DEM GANZEN
      Gitter (nicht nur im Streifen) - siehe _find_best_point() dort.
    - fit_waist_width_relation.py (Fest-Amplitude-Scan): der beim Scan
      selbst bereits gefundene und im pkl unter results['best'] gespeicherte
      Optimalpunkt - siehe _find_best_point() dort.
- SHOW_CROSSTALK: nur fuer fit_waist_width_relation.py - zeigt im rechten
  Schnitt-Panel zusaetzlich zu Uniformity_w auch Crosstalk_w (eta_weighted_grid)
  auf einer zweiten y-Achse.
- VALLEY_AXIS: nur fuer fit_uniformity_valley_overview.py (NEU) - "waist"
  oder "width": entlang welcher Achse das Uniformity-Minimum (+ Crosstalk/
  r_x/r_y an denselben Punkten) aufgetragen wird. Wird hier FEST vorgegeben
  statt interaktiv erfragt, damit "ein Klick and go" nicht durch einen
  Terminal-Prompt blockiert wird (beim direkten, manuellen Aufruf von
  fit_uniformity_valley_overview.py wird stattdessen gefragt).

Alle drei Skripte bleiben eigenstaendig direkt ausfuehrbar (python
fit_central_amplitudes.py / python fit_waist_width_relation.py / python
fit_uniformity_valley_overview.py) - dieses Skript importiert sie nur als
Module und ruft ihre main()-Funktion mit Parametern auf, ohne deren Code zu
duplizieren.

NEU (2026-08-26): RUN_AMPLITUDE_FIT/RUN_WAIST_WIDTH_FIT/DRAW_VALLEY_OVERVIEW
erlauben jetzt, jeden der drei Schritte EINZELN ein-/auszuschalten (vorher
liefen Schritt 1 und 2 immer beide zwangsweise, nur Schritt 3 war schon
abschaltbar) - z.B. um NUR den Fest-Amplitude-Scan (Schritt 2,
fit_waist_width_relation.py) auszuwerten, ohne dass dafuer zwingend auch
eine scan_amp_data_weighted_...pkl (Amplituden-optimierter Scan, fuer
Schritt 1 UND 3) vorhanden sein oder neu berechnet werden muss.

NEU (2026-08-27, auf User-Wunsch): "ich möchte lineare Funktionen an meine
Datensätze. Bei einem scan über konstante Amplituden sollen das lineare
Funktionen sein, keine Quadratischen" - Schritt 2 (fit_waist_width_relation.py,
Fest-Amplitude-Scan) waehlt das Modell fuer width(waist) jetzt per Default
NICHT mehr automatisch per Block-Kreuzvalidierung, sondern erzwingt "linear"
(siehe FORCE_WAIST_WIDTH_MODEL unten und FORCE_MODEL in
fit_waist_width_relation.py) - die automatische Modellwahl lief zuvor bei
manchen (v.a. verrauschten) Datensaetzen auch mal auf ein quadratisches/
kubisches Modell hinaus, obwohl der physikalisch erwartete Zusammenhang
linear ist.

NEU (2026-08-26): automatisch abgeleiteter output_prefix statt der in
fit_central_amplitudes.py/fit_waist_width_relation.py fest einprogrammierten
OUTPUT_PREFIX-Konstanten. GRUND: beide Konstanten tragen ein STEHENGEBLIEBENES
Datum ("...2026-08-24") UND einen fest einprogrammierten Datensatz-Namen
("...151x151pts...") - wird dieses Skript (ask_before_save=False, "ein Klick
and go") mit einer ANDEREN pkl-Datei (anderes N/andere Aufloesung/z.B. mit
Atom-Offset) oder an einem anderen Tag ausgefuehrt, wuerde OHNE diesen Fix
trotzdem IMMER derselbe Dateiname verwendet und damit ohne Rueckfrage der
komplett andere, aeltere (aber gueltige!) Fit-Ergebnis dieses Namens
ueberschrieben. _derive_output_prefix() baut den Prefix stattdessen aus dem
tatsaechlich verwendeten pkl-Dateinamen UND dem heutigen echten Datum -
siehe dortiger Docstring.

Nutzung:
    python run_all_fits.py
(vorher ggf. die Konfiguration unten anpassen - insbesondere DRAW_BEST_POINT
und LEGEND_FONTSIZE, sowie AMP_SCAN_PKL/FIXED_AMP_SCAN_PKL, falls die pkl-
Dateinamen nicht automatisch gefunden werden sollen). Findet beide pkl-
Dateien standardmaessig automatisch (neueste passende Datei in Results/),
falls die unten eingetragenen Namen nicht existieren.
"""
from datetime import date
from pathlib import Path


# Die Bibliothek dieses Ordners liegt seit dem Aufraeumen (2026-09-02) in
# lib/. Der Import setzt den dafuer noetigen sys.path-Eintrag - erst danach
# lassen sich die Module darunter unter ihrem blossen Namen importieren.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import paths  # noqa: E402,F401

import fit_central_amplitudes as fca
import fit_waist_width_relation as fww
import fit_uniformity_valley_overview as fuv
from weighted_multitone_amplitude_dependence_plots import DEFAULT_RESULTS_DIR


# ======================================================================
# Konfiguration - hier anpassen
# ======================================================================

# HIER die gewuenschten Dateinamen eintragen (muessen in Results/ liegen).
# Diese Eintraege haben IMMER Vorrang: existiert die eingetragene Datei,
# wird garantiert genau sie verwendet - unabhaengig davon, ob es neuere
# Dateien gibt. Die Auto-Erkennung (_resolve_pkl() unten) greift NUR als
# Fallback, wenn die eingetragene Datei nicht (mehr) existiert.
AMP_SCAN_PKL = "scan_amp_data_weighted_N3x4_151x151pts_Airy_2500res.pkl"  # Amplituden-optimierter Scan (r_x_grid/r_y_grid)
FIXED_AMP_SCAN_PKL = "scan_data_weighted_N3x4_40x40pts_Airy_1000res_+0.25x_+0.25y.pkl"  # Fest-Amplitude-Scan (uniformity_weighted_grid)

# Welche Schritte sollen laufen? Schritt 1 und 3 brauchen AMP_SCAN_PKL (den
# Amplituden-optimierten Scan mit r_x_grid/r_y_grid), Schritt 2 braucht
# FIXED_AMP_SCAN_PKL (den Fest-Amplitude-Scan). Um NUR den Fest-Amplitude-
# Scan auszuwerten (aktuell hier so eingestellt): RUN_AMPLITUDE_FIT=False,
# RUN_WAIST_WIDTH_FIT=True, DRAW_VALLEY_OVERVIEW=False.
RUN_AMPLITUDE_FIT = False       # Schritt 1/3: fit_central_amplitudes.py (r_x/r_y)
RUN_WAIST_WIDTH_FIT = True      # Schritt 2/3: fit_waist_width_relation.py (Fest-Amplitude-Scan)
DRAW_VALLEY_OVERVIEW = False    # Schritt 3/3: fit_uniformity_valley_overview.py (siehe unten)

# Grafische Optionen - gelten fuer ALLE Plots BEIDER Fit-Skripte:
LEGEND_FONTSIZE = 10
DRAW_BEST_POINT = True

# Nur fuer fit_waist_width_relation.py (Schnitt-Panel rechts): zusaetzlich
# zu Uniformity_w auch Crosstalk_w (eta_weighted_grid) einzeichnen, auf
# einer zweiten y-Achse - auf User-Wunsch. fit_central_amplitudes.py kennt
# diesen Parameter nicht (kein Schnitt-Panel dort).
SHOW_CROSSTALK = True

# NEU (2026-08-27, auf User-Wunsch), nur fuer fit_waist_width_relation.py
# (Schritt 2, Fest-Amplitude-Scan): erzwingt das Modell fuer width(waist)
# statt es automatisch per Block-Kreuzvalidierung waehlen zu lassen -
# "linear" (Default, wie gewuenscht: "Bei einem scan über konstante
# Amplituden sollen das lineare Funktionen sein, keine Quadratischen").
# None oder "auto" reaktiviert die automatische Modellwahl dort. Siehe
# FORCE_MODEL in fit_waist_width_relation.py fuer Details (die
# Block-CV laeuft bei "linear" trotzdem weiter, nur informativ im Log).
FORCE_WAIST_WIDTH_MODEL = "linear"

# Nur fuer fit_central_amplitudes.py (Amplituden-optimierter Scan): VOR den
# eigentlichen Fits eine 2x2-Uebersicht des GESAMTEN Scan-Gitters erzeugen
# (Uniformity, Crosstalk, r_x, r_y) - auf User-Wunsch. fit_waist_width_relation.py
# kennt diesen Parameter nicht (dort gibt es kein r_x/r_y). Ohne Wirkung,
# solange RUN_AMPLITUDE_FIT=False.
DRAW_DATASET_OVERVIEW = True

# NEU, nur fuer fit_uniformity_valley_overview.py (Amplituden-optimierter
# Scan, wie fit_central_amplitudes.py - dort gibt es sowohl Uniformity/
# Crosstalk als auch r_x/r_y in einer Datei): zeigt das Minimum von
# VALLEY_METRIC (+ Crosstalk/r_x/r_y an denselben Punkten) entlang
# VALLEY_AXIS ("waist" oder "width"), plus die zugehoerige Heatmap mit den
# Minimalpunkten in rot daneben. Ohne Wirkung, solange DRAW_VALLEY_OVERVIEW
# =False (siehe oben).
VALLEY_AXIS = "waist"  # "waist" oder "width"
# "uniformity" (Default, auf User-Wunsch, 2026-08-25): pro Spalte/Zeile wird
# der Gitterpunkt mit minimaler Uniformity_w gesucht. "combined": stattdessen
# der Gitterpunkt, an dem alpha*Uniformity_w + (1-alpha)*Crosstalk_w minimal
# ist (dieselbe Metrik wie fca._find_best_point(), hier spalten-/zeilenweise
# statt global) - beide Varianten wurden im Projektverlauf explizit
# gewuenscht, daher hier konfigurierbar statt fest verdrahtet.
VALLEY_METRIC = "uniformity"  # "uniformity" oder "combined"

# "Ein Klick and go": keine Terminal-Rueckfragen, direkt speichern.
ASK_BEFORE_SAVE = False
SAVE = True
SHOW = False


def _resolve_pkl(configured_name, glob_pattern, kind_label):
    """Falls configured_name (relativ zum Skript ODER zu Results/) nicht
    existiert, wird automatisch die zuletzt veraenderte Datei gefunden, die
    zu glob_pattern in Results/ passt - damit man nach einem neuen Scan
    nicht jedes Mal von Hand den Dateinamen in beiden Fit-Skripten
    nachpflegen muss. Gibt den zu verwendenden Dateinamen/Pfad zurueck."""
    here = Path(__file__).resolve().parent / configured_name
    in_results = DEFAULT_RESULTS_DIR / configured_name
    if here.exists() or in_results.exists():
        return configured_name

    kandidaten = sorted(DEFAULT_RESULTS_DIR.glob(glob_pattern), key=lambda p: p.stat().st_mtime)
    if not kandidaten:
        print(f"WARNUNG: weder '{configured_name}' noch irgendeine '{glob_pattern}'-Datei "
              f"in '{DEFAULT_RESULTS_DIR}' gefunden ({kind_label}). Verwende trotzdem "
              f"'{configured_name}' - die main()-Funktion gibt beim Aufruf eine genaue "
              f"Fehlermeldung mit Liste vorhandener Dateien aus.")
        return configured_name

    neueste = kandidaten[-1]
    if neueste.name != configured_name:
        print(f"Hinweis ({kind_label}): '{configured_name}' nicht gefunden - verwende "
              f"stattdessen die neueste passende Datei: '{neueste.name}'.")
    return neueste.name


def _derive_output_prefix(pkl_name, label):
    """
    Baut aus dem Dateinamen einer scan_data_weighted_...pkl bzw.
    scan_amp_data_weighted_...pkl AUTOMATISCH einen zum tatsaechlich
    verwendeten Datensatz UND zum heutigen echten Datum passenden
    Output-Prefix - statt der in fit_central_amplitudes.py/
    fit_waist_width_relation.py fest einprogrammierten OUTPUT_PREFIX-
    Konstanten (die beide noch ein STEHENGEBLIEBENES Datum "2026-08-24" UND
    einen fest einprogrammierten Datensatz-Namen tragen).

    Ohne diesen Fix wuerde ein non-interaktiver Lauf (ask_before_save=False,
    wie hier) mit einer ANDEREN pkl-Datei (anderes N/andere Aufloesung/z.B.
    mit Atom-Offset) oder an einem anderen Tag trotzdem IMMER denselben
    Dateinamen verwenden und damit OHNE Rueckfrage das aeltere, aber
    gueltige Fit-Ergebnis dieses Namens ueberschreiben.

    label: "AmpFit" oder "WaistWidthFit" (wie im Rest des Projekts, siehe
    claude/namenskonvention_fit_outputs.md).
    """
    stem = pkl_name
    for prefix in ("scan_amp_data_weighted_", "scan_data_weighted_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    if stem.endswith(".pkl"):
        stem = stem[:-len(".pkl")]
    return f"{label}_{stem}_weighted_{date.today().isoformat()}"


def main():
    # amp_pkl nur aufloesen (und dabei ggf. verlangen, dass die Datei
    # existiert), wenn tatsaechlich ein Schritt sie braucht - sonst muss fuer
    # "nur den Fest-Amplitude-Scan auswerten" (RUN_AMPLITUDE_FIT=False,
    # DRAW_VALLEY_OVERVIEW=False) gar keine scan_amp_data_weighted_...pkl
    # vorhanden sein.
    need_amp_pkl = RUN_AMPLITUDE_FIT or DRAW_VALLEY_OVERVIEW
    amp_pkl = (
        _resolve_pkl(AMP_SCAN_PKL, "scan_amp_data_weighted_*.pkl", "Amplituden-optimierter Scan")
        if need_amp_pkl else None
    )
    fixed_pkl = (
        _resolve_pkl(FIXED_AMP_SCAN_PKL, "scan_data_weighted_*.pkl", "Fest-Amplitude-Scan")
        if RUN_WAIST_WIDTH_FIT else None
    )

    amp_output_prefix = _derive_output_prefix(amp_pkl, "AmpFit") if amp_pkl is not None else None
    fixed_output_prefix = _derive_output_prefix(fixed_pkl, "WaistWidthFit") if fixed_pkl is not None else None

    common_kwargs = dict(
        draw_best_point=DRAW_BEST_POINT,
        legend_fontsize=LEGEND_FONTSIZE,
        ask_before_save=ASK_BEFORE_SAVE,
        save=SAVE,
        show=SHOW,
    )

    amp_result = None
    if RUN_AMPLITUDE_FIT:
        print("=" * 70)
        print(f"1/3: Amplituden-Fit (r_x/r_y) aus '{amp_pkl}' ...")
        print("=" * 70)
        try:
            amp_result = fca.main(pkl_datei=amp_pkl, output_prefix=amp_output_prefix,
                                   draw_dataset_overview=DRAW_DATASET_OVERVIEW, **common_kwargs)
        except Exception as exc:
            print(f"FEHLER beim Amplituden-Fit: {exc!r}")
            amp_result = None
    else:
        print("=" * 70)
        print("1/3: Amplituden-Fit - UEBERSPRUNGEN (RUN_AMPLITUDE_FIT=False)")
        print("=" * 70)

    width_result = None
    if RUN_WAIST_WIDTH_FIT:
        print()
        print("=" * 70)
        print(f"2/3: Waist-Width-Fit aus '{fixed_pkl}' ...")
        print("=" * 70)
        try:
            width_result = fww.main(pkl_datei=fixed_pkl, output_prefix=fixed_output_prefix,
                                     show_crosstalk=SHOW_CROSSTALK, force_model=FORCE_WAIST_WIDTH_MODEL,
                                     **common_kwargs)
        except Exception as exc:
            print(f"FEHLER beim Waist-Width-Fit: {exc!r}")
            width_result = None
    else:
        print()
        print("=" * 70)
        print("2/3: Waist-Width-Fit - UEBERSPRUNGEN (RUN_WAIST_WIDTH_FIT=False)")
        print("=" * 70)

    valley_result = None
    if DRAW_VALLEY_OVERVIEW:
        if not RUN_AMPLITUDE_FIT:
            print()
            print("=" * 70)
            print("3/3: Valley-Uebersicht - UEBERSPRUNGEN (braucht denselben "
                  "Amplituden-optimierten Scan wie Schritt 1, aber RUN_AMPLITUDE_FIT=False)")
            print("=" * 70)
        else:
            print()
            print("=" * 70)
            print(f"3/3: Valley-Uebersicht (Metrik: '{VALLEY_METRIC}', Achse: '{VALLEY_AXIS}') aus '{amp_pkl}' ...")
            print("=" * 70)
            try:
                valley_result = fuv.main(pkl_datei=amp_pkl, output_prefix=amp_output_prefix,
                                          axis=VALLEY_AXIS, metric=VALLEY_METRIC, **common_kwargs)
            except Exception as exc:
                print(f"FEHLER bei der Uniformity-Valley-Uebersicht: {exc!r}")
                valley_result = None
    else:
        print()
        print("=" * 70)
        print("3/3: Valley-Uebersicht - UEBERSPRUNGEN (DRAW_VALLEY_OVERVIEW=False)")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Zusammenfassung: alle erzeugten Dateien an einer Stelle auflisten.
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Zusammenfassung")
    print("=" * 70)

    if not RUN_AMPLITUDE_FIT:
        print("\nAmplituden-Fit (r_x/r_y) - UEBERSPRUNGEN (RUN_AMPLITUDE_FIT=False).")
    elif amp_result is not None:
        print(f"\nAmplituden-Fit (r_x/r_y, mm-Waist vor der Linse) - OK")
        print(f"  Formel-Dokument: {amp_result['formula_doc']}")
        print(f"  Plots: {fca.FIT_PLOTS_DIR}/{amp_output_prefix}_r_x_smoothformula.pdf")
        print(f"         {fca.FIT_PLOTS_DIR}/{amp_output_prefix}_r_y_smoothformula.pdf")
        print(f"         {fca.FIT_PLOTS_DIR}/{amp_output_prefix}_stripe_overview.pdf")
        if DRAW_DATASET_OVERVIEW:
            print(f"         {fca.FIT_PLOTS_DIR}/{amp_output_prefix}_dataset_overview.pdf")
        if amp_result["best_point"] is not None:
            bp = amp_result["best_point"]
            print(f"  Bester Punkt: waist={bp['waist_mm']:.4f} mm, width={bp['width_MHz']:.4f} MHz")
    else:
        print("\nAmplituden-Fit (r_x/r_y) - FEHLGESCHLAGEN, siehe Fehlermeldung oben.")

    if not RUN_WAIST_WIDTH_FIT:
        print("\nWaist-Width-Fit - UEBERSPRUNGEN (RUN_WAIST_WIDTH_FIT=False).")
    elif width_result is not None:
        print(f"\nWaist-Width-Fit (um-Waist nach der Linse) - OK")
        print(f"  Datensatz: {fixed_pkl}")
        modell_suffix = (f" (erzwungen, FORCE_WAIST_WIDTH_MODEL='{width_result['forced_model']}')"
                          if width_result.get("forced_model") else " (automatisch per Block-CV gewaehlt)")
        print(f"  Modell: {width_result['model']}{modell_suffix}")
        print(f"  Formel-Dokument: {width_result['formula_doc']}")
        print(f"  Plot: {fww.FIT_PLOTS_DIR}/{fixed_output_prefix}_waist_width_fit.pdf")
        if width_result["best_point"] is not None:
            bp = width_result["best_point"]
            print(f"  Bester Punkt: waist={bp['waist_um']:.4f} µm, width={bp['width_mhz']:.4f} MHz")
    else:
        print("\nWaist-Width-Fit - FEHLGESCHLAGEN, siehe Fehlermeldung oben.")

    if DRAW_VALLEY_OVERVIEW and RUN_AMPLITUDE_FIT:
        if valley_result is not None:
            axis_suffix = "over_waist" if valley_result["axis"] == "waist" else "over_width"
            print(f"\nValley-Uebersicht (Metrik: {valley_result['metric']}, Achse: {valley_result['axis']}) - OK")
            print(f"  Plot: {fca.FIT_PLOTS_DIR}/{amp_output_prefix}_valley_{axis_suffix}.pdf")
        else:
            print("\nValley-Uebersicht - FEHLGESCHLAGEN, siehe Fehlermeldung oben.")
    elif DRAW_VALLEY_OVERVIEW:
        print("\nValley-Uebersicht - UEBERSPRUNGEN (RUN_AMPLITUDE_FIT=False).")
    else:
        print("\nValley-Uebersicht - UEBERSPRUNGEN (DRAW_VALLEY_OVERVIEW=False).")

    print()
    return dict(amp_result=amp_result, width_result=width_result, valley_result=valley_result)


if __name__ == "__main__":
    main()
