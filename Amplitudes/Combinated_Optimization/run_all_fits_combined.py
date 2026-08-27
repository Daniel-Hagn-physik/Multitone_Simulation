"""
run_all_fits_combined.py
==========================
"Ein Klick and go"-Master-Skript fuer Combinated_Optimization, analog zu
Weighted_Optimization/run_all_fits.py: findet die (neuesten) kombinierten
Scan-pkl-Dateien automatisch, ruft NACHEINANDER

    1. fit_combined_region.py       (Fest-Amplituden-Scan)
    2. fit_combined_amp_region.py   (Amplituden-optimierter Scan)

nicht-interaktiv auf und fasst beide Ergebnisse am Ende zusammen. Jeder
Schritt ist unabhaengig (fehlt eine der beiden pkl-Dateien, wird nur der
jeweils andere Teil uebersprungen/als FEHLGESCHLAGEN markiert, ohne den
anderen Schritt zu verhindern).

Nutzung:
    python run_all_fits_combined.py
(vorher ggf. COMBINED_SCAN_PKL / COMBINED_AMP_SCAN_PKL unten anpassen, oder
die Auto-Erkennung nutzen lassen - siehe _resolve_pkl() in
fit_combined_region.py / fit_combined_amp_region.py).
"""
from pathlib import Path

import fit_combined_region as fcr
import fit_combined_amp_region as fcar
from combined_scan_methods import DEFAULT_RESULTS_DIR
from fit_combined_region import FIT_PLOTS_DIR


# ======================================================================
# Konfiguration - hier anpassen
# ======================================================================
COMBINED_SCAN_PKL = "scan_data_combined_N3x4_40x40pts_Airy.pkl"
COMBINED_AMP_SCAN_PKL = "scan_amp_data_combined_N3x4_15x15pts_Airy.pkl"

# None = aus dem pkl uebernehmen (guenstiges Nachjustieren ohne Re-Scan,
# siehe recombine_from_grids() in combined_scan_methods.py). Gilt fuer
# BEIDE Schritte.
ALPHA = None
COMBO_LAMBDA = None
COMBO_PERCENTILE = None
WIN_AXIS = "before_lens"

LEGEND_FONTSIZE = 9
DRAW_BEST_POINT = True

# Nur fuer Schritt 2 (Amplituden-optimiert): zusaetzlich die Standard-
# Amplituden-Scan-Uebersichtsplots (6-Panel + Dependence-Cuts) erzeugen?
PLOT_AMP_OVERVIEW = True

ASK_BEFORE_SAVE = False
SAVE = True
SHOW = False


def _run_fixed_step():
    pkl = fcr._resolve_pkl(COMBINED_SCAN_PKL)
    print("=" * 70)
    print(f"1/2: Combined-Region-Fit (Fest-Amplituden) aus '{pkl}' ...")
    print("=" * 70)
    try:
        return fcr.main(
            pkl_datei=pkl, alpha=ALPHA, combo_lambda=COMBO_LAMBDA,
            combo_percentile=COMBO_PERCENTILE, win_axis=WIN_AXIS,
            draw_best_point=DRAW_BEST_POINT, legend_fontsize=LEGEND_FONTSIZE,
            ask_before_save=ASK_BEFORE_SAVE, save=SAVE, show=SHOW,
        )
    except Exception as exc:
        print(f"FEHLER beim Combined-Region-Fit (Fest-Amplituden): {exc!r}")
        return None


def _run_amp_step():
    pkl = fcar._resolve_pkl(COMBINED_AMP_SCAN_PKL)
    print("=" * 70)
    print(f"2/2: Combined-Region-Fit (Amplituden-optimiert) aus '{pkl}' ...")
    print("=" * 70)
    try:
        return fcar.main(
            pkl_datei=pkl, alpha=ALPHA, combo_lambda=COMBO_LAMBDA,
            combo_percentile=COMBO_PERCENTILE, win_axis=WIN_AXIS,
            draw_best_point=DRAW_BEST_POINT, legend_fontsize=LEGEND_FONTSIZE,
            ask_before_save=ASK_BEFORE_SAVE, save=SAVE, show=SHOW,
            plot_amp_overview=PLOT_AMP_OVERVIEW,
        )
    except Exception as exc:
        print(f"FEHLER beim Combined-Region-Fit (Amplituden-optimiert): {exc!r}")
        return None


def _print_result(label, result):
    if result is None:
        print(f"\n{label} - FEHLGESCHLAGEN, siehe Fehlermeldung oben.")
        return
    print(f"\n{label} - OK")
    print(f"  Formel-/Region-Dokument: {result['formula_doc']}")
    print(f"  Plots: {result['plots']['metric_comparison']}")
    print(f"         {result['plots']['region']}")
    amp_overview = result['plots'].get('amp_overview')
    if amp_overview:
        print(f"         + Standard-Amplituden-Scan-Uebersichtsplots (6-Panel + "
              f"Dependence-Cuts) in {FIT_PLOTS_DIR} (Dateinamen s.o., "
              f"'FlatMultiTone_AmpScan_...')")
    if result["best_point"] is not None:
        bp = result["best_point"]
        print(f"  Bester Punkt: waist={bp['waist_um']:.4f} µm "
              f"(win_input={bp['win_input_mm']:.4f} mm), width={bp['width_MHz']:.4f} MHz")
        if bp.get('r_at_best'):
            print(f"    r_x/r_y an diesem Punkt: {bp['r_at_best']}")
    if result["region"] is not None:
        reg = result["region"]
        print(f"  Region: waist={reg['waist_um'][0]:.4f}..{reg['waist_um'][1]:.4f} µm "
              f"(win_input={reg['win_input_mm'][0]:.4f}..{reg['win_input_mm'][1]:.4f} mm), "
              f"width={reg['width_MHz'][0]:.4f}..{reg['width_MHz'][1]:.4f} MHz "
              f"({reg['n_points_region']}/{reg['n_points_total']} Gitterpunkte)")


def main():
    result_fixed = _run_fixed_step()
    print()
    result_amp = _run_amp_step()

    print()
    print("=" * 70)
    print("Zusammenfassung")
    print("=" * 70)
    _print_result("Combined-Region-Fit (Fest-Amplituden)", result_fixed)
    _print_result("Combined-Region-Fit (Amplituden-optimiert)", result_amp)

    print()
    return dict(result_fixed=result_fixed, result_amp=result_amp)


if __name__ == "__main__":
    main()
