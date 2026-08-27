"""
Combinated_Optimization/combine_existing_datasets.py
=======================================================

Kombiniert ZWEI BEREITS VORHANDENE, SEPARAT berechnete Datensaetze (einen
"hart", einen "gewichtet") OHNE erneut zu scannen - fuer den Fall, dass die
Rohdaten schon vorliegen (z.B. aus frueheren Laeufen von
Weighted_Optimization/weighted_winwidthscan_startdialog.py bzw.
weighted_winwidthampscan_startdialog.py, oder aus dem urspruenglichen
Hard_Optimization) und nur noch die Kombination (siehe
combined_scan_methods.py/combined_amp_scan_methods.py fuer das Prinzip:
Normierung + Mittelwert + Uneinigkeits-Strafterm -> Score -> Region)
ausgerechnet werden soll - das spart die (teure) Wiederholung BEIDER Scans,
die scan_win_width_combined_uniformity()/
scan_win_width_amplitude_dependence_combined() sonst durchfuehren wuerden.

Deckt BEIDE Scan-Arten ab:

1. FEST-AMPLITUDEN-Scan (uniformity_grid/crosstalk_grid vs.
   uniformity_weighted_grid/eta_weighted_grid) - typische Dateinamen:
       hart:      scan_data_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl
       gewichtet: scan_data_weighted_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl
   -> combine_existing_fixed_scans()

2. AMPLITUDEN-OPTIMIERTER Scan (zusaetzlich r_x_grid/r_y_grid pro Punkt) -
   typische Dateinamen:
       hart:      scan_amp_data_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl
       gewichtet: scan_amp_data_weighted_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl
   -> combine_existing_amp_scans()

combine_existing_scans() waehlt automatisch anhand der im hart-pkl
enthaltenen Schluessel zwischen beiden (r_x_grid vorhanden -> Amplituden-
optimiert, sonst Fest-Amplituden).

WICHTIGE VORAUSSETZUNG (siehe Modul-Docstring von combined_scan_methods.py):
beide Datensaetze muessen auf EXAKT demselben (win_input, width)-Gitter
liegen (gleiche win_input_range/width_range/n_win_input/n_width beim
jeweiligen Scan) - nur dann ist eine punktweise Kombination ohne
Interpolation sinnvoll. Wird hier vor der Kombination explizit geprueft
(Gitter UND N_x/N_y/Profil), mit einer klaren Fehlermeldung, falls nicht.

Das Ergebnis-dict ist BYTE-IDENTISCH zu dem, was
scan_win_width_combined_uniformity()/
scan_win_width_amplitude_dependence_combined() liefern wuerden - es kann
also direkt und unveraendert in CombinedFixedScanPlotter/
AmplitudeScanPlotter, fit_combined_region.py/fit_combined_amp_region.py und
run_all_fits_combined.py weiterverwendet werden. Wird per
save_scan_combined_results()/save_scan_amp_results_combined() (siehe
combined_scan_methods.py/combined_amp_scan_methods.py) unter demselben
Dateinamensschema wie ein "echter" kombinierter Scan gespeichert, damit
diese Weiterverarbeitung nichts von der Abkuerzung merkt.

WICHTIG: von sich aus rechnet dieses Modul nur die Kombination - es plottet
NICHTS. Per Default (run_fit=True, siehe unten) wird direkt im Anschluss
ans Speichern aber automatisch der passende Fit-/Plot-Schritt mit
aufgerufen: fit_combined_region.py (Fest-Amplituden) bzw.
fit_combined_amp_region.py (Amplituden-optimiert), auf genau der eben
gespeicherten pkl-Datei. Ein einziger Aufruf hier liefert damit bereits die
PDF-Plots (Metrik-Vergleich + Region) und das Formel-/Region-Dokument -
kein separater Aufruf von fit_combined_region.py/fit_combined_amp_region.py/
run_all_fits_combined.py noetig. run_fit=False (oder save=False)
ueberspringt diesen Schritt und gibt nur das kombinierte dict zurueck.

Nutzung (Skript, siehe Konfiguration unten):
    python combine_existing_datasets.py

Oder direkt aus eigenem Code:
    from combine_existing_datasets import combine_existing_scans
    combined, filepath, fit_result = combine_existing_scans(
        "scan_data_N3x4_40x40pts_Airy.pkl",
        "scan_data_weighted_N3x4_40x40pts_Airy.pkl",
        alpha=0.9, combo_lambda=0.75, combo_percentile=25.0,
    )
    # fit_result['plots']['metric_comparison']/['region'], fit_result['formula_doc'], ...
    # (siehe fit_combined_region.py/fit_combined_amp_region.py main() fuer die genaue Struktur;
    # fit_result ist None, falls run_fit=False oder save=False)
"""
import sys
from pathlib import Path as FilePath

import numpy as np

_WEIGHTED_DIR = FilePath(__file__).resolve().parent.parent / "Weighted_Optimization"
if str(_WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(_WEIGHTED_DIR))

# load_amp_scan_results() ist format-agnostisch (laedt jedes gepickelte
# results-dict) und sucht bei einem reinen Dateinamen zuerst im aktuellen
# Arbeitsverzeichnis, danach in Weighted_Optimization/Results - also genau
# dort, wo bereits vorhandene hart-/gewichtet-Datensaetze typischerweise
# liegen (siehe Modul-Docstring dort).
from weighted_multitone_amplitude_dependence_plots import load_amp_scan_results  # noqa: E402

from combined_scan_methods import (  # noqa: E402
    DEFAULT_RESULTS_DIR, build_combined_scan_results,
    MultitoneFlatTopOptimizer,
)
import combined_amp_scan_methods as _camp  # noqa: E402  # fuer build_combined_amp_scan_results

# Fuer den automatischen Plot-/Fit-Schritt nach dem Kombinieren (run_fit=True,
# siehe Modul-Docstring) - dieselben Fit-Skripte, die auch ein "echter"
# kombinierter Scan am Ende durchlaeuft.
import fit_combined_region as _fcr  # noqa: E402
import fit_combined_amp_region as _fcar  # noqa: E402


# ======================================================================
# Konfiguration - hier anpassen (nur relevant, wenn dieses Skript direkt
# ausgefuehrt wird; combine_existing_scans() etc. lassen sich auch direkt
# importieren und mit eigenen Pfaden/Parametern aufrufen, siehe Docstring).
# ======================================================================
HARD_PKL = r"scan_data_N3x4_40x40pts_Airy.pkl"
WEIGHTED_PKL = r"scan_data_weighted_N3x4_40x40pts_Airy.pkl"

# "auto" (anhand r_x_grid im hart-pkl erkannt), "fixed" oder "amp".
SCAN_TYP = "auto"

ALPHA = 0.9  # Fest-Amplituden-Default; fuer Amplituden-optimiert eher 0.7
# verwenden (siehe combined_amp_scan_methods.py) - bei SCAN_TYP="auto"/"amp"
# unten ggf. anpassen.
COMBO_LAMBDA = 0.75
COMBO_PERCENTILE = 25.0

# Nur fuer den Amplituden-optimierten Fall relevant (siehe
# build_combined_amp_scan_results() in combined_amp_scan_methods.py):
R_GRID_SOURCE = "weighted"

SAVE = True
VERBOSE = True

# Nach dem Speichern automatisch den passenden Fit-/Plot-Schritt aufrufen
# (fit_combined_region.py bzw. fit_combined_amp_region.py) - erzeugt PDF-
# Plots + Formel-/Region-Dokument, ohne dass man diese Skripte danach noch
# separat aufrufen muesste. Braucht SAVE=True (das Fit-Skript laedt von der
# gerade gespeicherten pkl-Datei).
RUN_FIT = True

# Zusaetzliche Keyword-Argumente fuer den Fit-Schritt, z.B.
#   dict(win_axis="after_lens", show=True, plot_amp_overview=False)
# siehe main() in fit_combined_region.py/fit_combined_amp_region.py fuer
# alle moeglichen Schluessel. None = Defaults (ask_before_save=False,
# save=True, show=False, siehe _run_fixed_fit()/_run_amp_fit() unten).
FIT_KWARGS = None


# ======================================================================
# Validierung - klare Fehlermeldung statt kryptischem Numpy-Fehler, falls
# die beiden Datensaetze nicht zusammenpassen.
# ======================================================================
def _check_compatible(hard, weighted):
    problems = []
    for key in ("N_x", "N_y", "profile"):
        if hard.get(key) != weighted.get(key):
            problems.append(f"{key}: hart={hard.get(key)!r} != gewichtet={weighted.get(key)!r}")

    win_h = np.asarray(hard.get("win_input_vals"))
    win_w = np.asarray(weighted.get("win_input_vals"))
    width_h = np.asarray(hard.get("width_vals"))
    width_w = np.asarray(weighted.get("width_vals"))
    if win_h.shape != win_w.shape or not np.allclose(win_h, win_w):
        problems.append(
            f"win_input-Gitter unterschiedlich (hart: {win_h.shape}, gewichtet: {win_w.shape}, "
            f"oder Werte weichen ab) - beide Datensaetze muessen mit identischem win_input_range/"
            f"n_win_input erzeugt worden sein."
        )
    if width_h.shape != width_w.shape or not np.allclose(width_h, width_w):
        problems.append(
            f"width-Gitter unterschiedlich (hart: {width_h.shape}, gewichtet: {width_w.shape}, "
            f"oder Werte weichen ab) - beide Datensaetze muessen mit identischem width_range/"
            f"n_width erzeugt worden sein."
        )

    if problems:
        raise ValueError(
            "Die beiden Datensaetze passen nicht zusammen - jeder Punkt braeuchte hart UND "
            "gewichtet auf demselben Gitter, mit derselben Tonanzahl/Profil:\n  - "
            + "\n  - ".join(problems)
            + "\n(hart-Datei sollte mit derselben win_input_range/width_range/n_win_input/n_width/"
              "N_x/N_y wie die gewichtete Datei gescannt worden sein - im Zweifel beide neu "
              "scannen mit identischen Parametern, siehe combined_scan_methods.py-Docstring.)"
        )


def _is_amp_dataset(results):
    """Amplituden-optimierte Scans haben r_x_grid/r_y_grid (pro Punkt
    optimierte Amplituden-Verhaeltnisse); Fest-Amplituden-Scans nicht
    (dort ist amps ein fester, gleicher Wert fuer alle Gitterpunkte)."""
    return "r_x_grid" in results and "r_y_grid" in results


# ======================================================================
# Fest-Amplituden-Fall
# ======================================================================
def combine_existing_fixed_scans(hard_pkl, weighted_pkl, alpha=0.9, combo_lambda=0.75,
                                  combo_percentile=25.0, save=True, filepath=None, verbose=True,
                                  run_fit=True, fit_kwargs=None):
    """Laedt zwei bereits vorhandene Fest-Amplituden-Datensaetze (siehe
    Modul-Docstring fuer die typischen Dateinamen/Speicherorte), prueft
    Kompatibilitaet und kombiniert sie OHNE Re-Scan (siehe
    build_combined_scan_results() in combined_scan_methods.py fuer das
    Kombinationsprinzip). Gibt (combined, filepath, fit_result) zurueck -
    combined ist identisch zu dem, was scan_win_width_combined_uniformity()
    liefern wuerde, direkt verwendbar mit CombinedFixedScanPlotter/
    fit_combined_region.py. filepath ist None, falls save=False.

    run_fit=True (Default, braucht save=True): ruft im Anschluss automatisch
    fit_combined_region.main() auf der gerade gespeicherten pkl-Datei auf
    (PDF-Plots + Formel-/Region-Dokument) - siehe Modul-Docstring. Das
    main()-Rueckgabedict landet in fit_result (sonst None)."""
    hard = load_amp_scan_results(hard_pkl)
    weighted = load_amp_scan_results(weighted_pkl)
    if _is_amp_dataset(hard) or _is_amp_dataset(weighted):
        raise ValueError(
            "Mindestens einer der beiden Datensaetze sieht nach einem AMPLITUDEN-optimierten "
            "Scan aus (enthaelt r_x_grid/r_y_grid) - fuer diesen Fall combine_existing_amp_scans() "
            "verwenden, nicht combine_existing_fixed_scans()."
        )
    _check_compatible(hard, weighted)

    if verbose:
        print(f"Kombiniere bestehende Fest-Amplituden-Datensaetze OHNE Re-Scan:\n"
              f"  hart:      {hard_pkl}\n  gewichtet: {weighted_pkl}")
    combined = build_combined_scan_results(hard, weighted, alpha=alpha,
                                            combo_lambda=combo_lambda,
                                            combo_percentile=combo_percentile)

    if verbose:
        _print_summary(combined)

    if save:
        filepath = _save_combined(combined, filepath, amp=False)
    else:
        filepath = None

    fit_result = None
    if run_fit:
        if not save:
            if verbose:
                print("Hinweis: run_fit=True braucht save=True (das Fit-Skript laedt von einer "
                      "gespeicherten pkl-Datei) - Plot-/Fit-Schritt wird uebersprungen.")
        else:
            if verbose:
                print("Erzeuge Plots + Formel-/Region-Dokument (fit_combined_region.py) ...")
            fit_result = _run_fixed_fit(filepath, fit_kwargs)

    return combined, filepath, fit_result


# ======================================================================
# Amplituden-optimierter Fall
# ======================================================================
def combine_existing_amp_scans(hard_pkl, weighted_pkl, alpha=0.7, combo_lambda=0.75,
                                combo_percentile=25.0, r_grid_source="weighted",
                                save=True, filepath=None, verbose=True,
                                run_fit=True, fit_kwargs=None):
    """Wie combine_existing_fixed_scans(), aber fuer bereits vorhandene
    AMPLITUDEN-optimierte Datensaetze (mit r_x_grid/r_y_grid pro Punkt) -
    siehe build_combined_amp_scan_results() in combined_amp_scan_methods.py
    fuer das Kombinationsprinzip (identisch zum Fest-Amplituden-Fall,
    zusaetzlich werden beide r_x/r_y-Saetze mitgefuehrt). Gibt
    (combined, filepath, fit_result) zurueck - combined direkt verwendbar
    mit AmplitudeScanPlotter/CombinedFixedScanPlotter/
    fit_combined_amp_region.py.

    run_fit=True (Default, braucht save=True): ruft im Anschluss automatisch
    fit_combined_amp_region.main() auf der gerade gespeicherten pkl-Datei
    auf (PDF-Plots, Formel-/Region-Dokument, PLUS die Standard-Amplituden-
    Scan-Uebersichtsplots ueber AmplitudeScanPlotter) - siehe Modul-
    Docstring. Das main()-Rueckgabedict landet in fit_result (sonst None)."""
    hard = load_amp_scan_results(hard_pkl)
    weighted = load_amp_scan_results(weighted_pkl)
    if not (_is_amp_dataset(hard) and _is_amp_dataset(weighted)):
        raise ValueError(
            "Mindestens einer der beiden Datensaetze hat kein r_x_grid/r_y_grid - sieht nach "
            "einem FEST-Amplituden-Scan aus. Fuer diesen Fall combine_existing_fixed_scans() "
            "verwenden, nicht combine_existing_amp_scans()."
        )
    _check_compatible(hard, weighted)

    if verbose:
        print(f"Kombiniere bestehende Amplituden-optimierte Datensaetze OHNE Re-Scan:\n"
              f"  hart:      {hard_pkl}\n  gewichtet: {weighted_pkl}")
    combined = _camp.build_combined_amp_scan_results(
        hard, weighted, alpha=alpha, combo_lambda=combo_lambda,
        combo_percentile=combo_percentile, r_grid_source=r_grid_source,
    )

    if verbose:
        _print_summary(combined)

    if save:
        filepath = _save_combined(combined, filepath, amp=True)
    else:
        filepath = None

    fit_result = None
    if run_fit:
        if not save:
            if verbose:
                print("Hinweis: run_fit=True braucht save=True (das Fit-Skript laedt von einer "
                      "gespeicherten pkl-Datei) - Plot-/Fit-Schritt wird uebersprungen.")
        else:
            if verbose:
                print("Erzeuge Plots + Formel-/Region-Dokument (fit_combined_amp_region.py) ...")
            fit_result = _run_amp_fit(filepath, fit_kwargs)

    return combined, filepath, fit_result


# ======================================================================
# Auto-Erkennung
# ======================================================================
def combine_existing_scans(hard_pkl, weighted_pkl, **kwargs):
    """Laedt nur den hart-Datensatz vorab an, um automatisch zwischen
    Fest-Amplituden- und Amplituden-optimiertem Fall zu unterscheiden
    (r_x_grid vorhanden -> Amplituden-optimiert), und delegiert dann an
    combine_existing_fixed_scans()/combine_existing_amp_scans(). Alle
    zusaetzlichen Keyword-Argumente (alpha, combo_lambda, combo_percentile,
    r_grid_source [nur amp], save, filepath, verbose) werden durchgereicht."""
    hard = load_amp_scan_results(hard_pkl)
    if _is_amp_dataset(hard):
        return combine_existing_amp_scans(hard_pkl, weighted_pkl, **kwargs)
    kwargs.pop("r_grid_source", None)
    return combine_existing_fixed_scans(hard_pkl, weighted_pkl, **kwargs)


# ======================================================================
# Hilfsfunktionen
# ======================================================================
def _print_summary(combined):
    b, r = combined["best"], combined["region"]
    print("-" * 70)
    if b["win_input"] is not None:
        print(f"Kombiniert bester Punkt: win_input={b['win_input']*1e3:.4f} mm, "
              f"width={b['width']*1e-6:.3f} MHz -> "
              f"Uniformity_hart={b['uniformity_hart']*100:.2f}%, "
              f"Crosstalk_hart={b['crosstalk_hart']*100:.3f}%, "
              f"Uniformity_w={b['uniformity_weighted']*100:.2f}%, "
              f"Crosstalk_w={b['crosstalk_weighted']*100:.3f}%")
    if r["win_input_min"] is not None:
        print(f"Kombinierte Region (beste {combined['combo_percentile']:.0f}% des Scores, "
              f"groesstes eingeschriebenes Rechteck aus {r['n_points_region']}/"
              f"{r['n_points_total']} Punkten): "
              f"win_input in [{r['win_input_min']*1e3:.4f}, {r['win_input_max']*1e3:.4f}] mm, "
              f"width in [{r['width_min']*1e-6:.4f}, {r['width_max']*1e-6:.4f}] MHz")
    else:
        print("Kombinierte Region: kein gueltiges Rechteck gefunden (zu wenige valide Punkte).")
    print("-" * 70)


def _run_fixed_fit(filepath, fit_kwargs):
    """Ruft fit_combined_region.main() auf der gerade gespeicherten
    Fest-Amplituden-combined-pkl auf - Defaults hier bewusst nicht-
    interaktiv (ask_before_save=False), damit combine_existing_fixed_scans()
    ohne Rueckfrage in einem Rutsch durchlaeuft; ueber fit_kwargs
    ueberschreibbar (siehe main() in fit_combined_region.py fuer alle
    moeglichen Schluessel: alpha/combo_lambda/combo_percentile [hier i.d.R.
    unnoetig, da bereits kombiniert], win_axis, draw_best_point,
    legend_fontsize, ask_before_save, save, show)."""
    kwargs = dict(ask_before_save=False, save=True, show=False)
    if fit_kwargs:
        kwargs.update(fit_kwargs)
    return _fcr.main(pkl_datei=str(filepath), **kwargs)


def _run_amp_fit(filepath, fit_kwargs):
    """Wie _run_fixed_fit(), aber fit_combined_amp_region.main() fuer die
    Amplituden-optimierte combined-pkl (erzeugt zusaetzlich die Standard-
    Amplituden-Scan-Uebersichtsplots, siehe PLOT_AMP_OVERVIEW dort -
    ebenfalls per fit_kwargs=dict(plot_amp_overview=False) abschaltbar)."""
    kwargs = dict(ask_before_save=False, save=True, show=False)
    if fit_kwargs:
        kwargs.update(fit_kwargs)
    return _fcar.main(pkl_datei=str(filepath), **kwargs)


def _save_combined(combined, filepath, amp):
    """Speichert ueber dieselben save_scan_..._combined()-Methoden wie ein
    echter kombinierter Scan (identisches Dateinamensschema/Ordner), damit
    das Ergebnis von aussen nicht von einem 'echten' kombinierten Scan zu
    unterscheiden ist. Nutzt dafuer eine temporaere Optimizer-Instanz nur
    als duennen Traeger fuer self.results/self.N_x/self.profile - es wird
    NICHTS neu berechnet oder simuliert."""
    dummy = MultitoneFlatTopOptimizer(
        out_dir=DEFAULT_RESULTS_DIR, N_x=combined["N_x"], N_y=combined["N_y"],
    )
    dummy.profile = combined["profile"]
    if amp:
        dummy.results["scan2d_amp_combined"] = combined
        return dummy.save_scan_amp_results_combined(filepath)
    else:
        dummy.results["scan2d_combined"] = combined
        return dummy.save_scan_combined_results(filepath)


def main():
    kwargs = dict(
        combo_lambda=COMBO_LAMBDA, combo_percentile=COMBO_PERCENTILE,
        save=SAVE, verbose=VERBOSE, run_fit=RUN_FIT, fit_kwargs=FIT_KWARGS,
    )
    if SCAN_TYP == "fixed":
        _, filepath, fit_result = combine_existing_fixed_scans(HARD_PKL, WEIGHTED_PKL, alpha=ALPHA, **kwargs)
    elif SCAN_TYP == "amp":
        _, filepath, fit_result = combine_existing_amp_scans(HARD_PKL, WEIGHTED_PKL, alpha=ALPHA,
                                                               r_grid_source=R_GRID_SOURCE, **kwargs)
    elif SCAN_TYP == "auto":
        _, filepath, fit_result = combine_existing_scans(HARD_PKL, WEIGHTED_PKL, alpha=ALPHA,
                                                           r_grid_source=R_GRID_SOURCE, **kwargs)
    else:
        raise ValueError(f"SCAN_TYP muss 'auto', 'fixed' oder 'amp' sein, nicht {SCAN_TYP!r}.")

    if filepath is not None:
        print(f"\nKombiniertes pkl gespeichert: {filepath}")
    if fit_result is not None:
        print(f"Formel-/Region-Dokument: {fit_result['formula_doc']}")
        print(f"Plots: {fit_result['plots']['metric_comparison']}")
        print(f"       {fit_result['plots']['region']}")


if __name__ == "__main__":
    main()
