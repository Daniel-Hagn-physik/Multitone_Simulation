"""
run_all_fits_combined.py
==========================
"Ein Klick and go"-Master-Skript fuer Combinated_Optimization, analog zu
Weighted_Optimization/run_all_fits.py: findet die (neueste) kombinierte
Scan-pkl-Datei automatisch, ruft fit_combined_region.py nicht-interaktiv
auf und fasst das Ergebnis am Ende zusammen.

Nutzung:
    python run_all_fits_combined.py
(vorher ggf. COMBINED_SCAN_PKL unten anpassen, oder die Auto-Erkennung
nutzen lassen - siehe _resolve_pkl() in fit_combined_region.py).
"""
from pathlib import Path

import fit_combined_region as fcr
from combined_scan_methods import DEFAULT_RESULTS_DIR


# ======================================================================
# Konfiguration - hier anpassen
# ======================================================================
COMBINED_SCAN_PKL = "scan_data_combined_N3x4_40x40pts_Airy.pkl"

# None = aus dem pkl uebernehmen (guenstiges Nachjustieren ohne Re-Scan,
# siehe recombine_from_grids() in combined_scan_methods.py).
ALPHA = None
COMBO_LAMBDA = None
COMBO_PERCENTILE = None
WIN_AXIS = "before_lens"

LEGEND_FONTSIZE = 9
DRAW_BEST_POINT = True

ASK_BEFORE_SAVE = False
SAVE = True
SHOW = False


def main():
    pkl = fcr._resolve_pkl(COMBINED_SCAN_PKL)

    print("=" * 70)
    print(f"Combined-Region-Fit aus '{pkl}' ...")
    print("=" * 70)
    try:
        result = fcr.main(
            pkl_datei=pkl, alpha=ALPHA, combo_lambda=COMBO_LAMBDA,
            combo_percentile=COMBO_PERCENTILE, win_axis=WIN_AXIS,
            draw_best_point=DRAW_BEST_POINT, legend_fontsize=LEGEND_FONTSIZE,
            ask_before_save=ASK_BEFORE_SAVE, save=SAVE, show=SHOW,
        )
    except Exception as exc:
        print(f"FEHLER beim Combined-Region-Fit: {exc!r}")
        result = None

    print()
    print("=" * 70)
    print("Zusammenfassung")
    print("=" * 70)
    if result is not None:
        print(f"\nCombined-Region-Fit - OK")
        print(f"  Formel-/Region-Dokument: {result['formula_doc']}")
        print(f"  Plots: {result['plots']['metric_comparison']}")
        print(f"         {result['plots']['region']}")
        if result["best_point"] is not None:
            bp = result["best_point"]
            print(f"  Bester Punkt: waist={bp['waist_um']:.4f} µm "
                  f"(win_input={bp['win_input_mm']:.4f} mm), width={bp['width_MHz']:.4f} MHz")
        if result["region"] is not None:
            reg = result["region"]
            print(f"  Region: waist={reg['waist_um'][0]:.4f}..{reg['waist_um'][1]:.4f} µm "
                  f"(win_input={reg['win_input_mm'][0]:.4f}..{reg['win_input_mm'][1]:.4f} mm), "
                  f"width={reg['width_MHz'][0]:.4f}..{reg['width_MHz'][1]:.4f} MHz "
                  f"({reg['n_points_region']}/{reg['n_points_total']} Gitterpunkte)")
    else:
        print("\nCombined-Region-Fit - FEHLGESCHLAGEN, siehe Fehlermeldung oben.")

    print()
    return dict(result=result)


if __name__ == "__main__":
    main()
