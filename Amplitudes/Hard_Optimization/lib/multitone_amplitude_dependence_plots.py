"""
Multitone FlatTop - Amplitude-Dependence Plots
================================================

Plotting-only companion to `MultitoneFlatTopOptimizer.scan_win_width_amplitude_dependence()`
(in `multitone_flattop_optimizer.py`). Same split as
`multitone_flattop_scan_plots.py`/`scan_win_width_uniformity()`: this module has
NO dependency on the optimizer (or on scipy) - it only needs the small dict
produced by `MultitoneFlatTopOptimizer.get_scan_amp_results()` /
`save_scan_amp_results()`, either handed over directly in the same session or
reloaded from a pickle file via `load_amp_scan_results()`.

Was hier geplottet wird - die eigentliche Antwort auf die Projektfrage
"Finde die Abhängigkeit der Amplituden von Waist und Width":
Für jeden (win_input, width)-Punkt hat scan_win_width_amplitude_dependence()
die Amplituden-Verhältnisse (r_x, r_y) gefunden, die dort das kombinierte
Ziel alpha*uniformity + (1-alpha)*eta minimieren (Default alpha=0.7 -
Uniformity UND Crosstalk gemeinsam; siehe amps_from_ratio() im Optimizer:
äußere Töne = r, innere Töne = 1).
r_x_opt(win_input, width) und r_y_opt(win_input, width) SIND diese
Abhängigkeit - als Heatmap (plot_scan2d_combined) und als Kurven entlang
eines festen Schnitts (plot_dependence_cuts).

Ordnerstruktur (Stand 2026-08-18): Rohdaten (Pickle-Dateien von
save_scan_amp_results()/save_scan_results()) landen im Ordner "Results",
Bilder (PNGs von diesem Modul hier) im Ordner "Bilder" - beide direkt neben
diesem Skript, siehe DEFAULT_RESULTS_DIR/DEFAULT_IMAGES_DIR unten. Beide
Ordner werden bei Bedarf automatisch angelegt (Path.mkdir(exist_ok=True)),
es muss also nichts von Hand vorbereitet werden.

Dateinamen kodieren jetzt (siehe _filetag()): Tonanzahl (N_x x N_y), Anzahl
der Gitterpunkte im Bild (n_win_input x n_width) UND das verwendete
Strahlprofil (Airy/Gauss) - z.B. "FlatMultiTone_AmpScan_Combined_N3x4_15x15pts_Airy.png".
So lässt sich aus dem Dateinamen allein ablesen, wie viele Datenpunkte ein
Bild zeigt und ob es sich um das Airy- oder Gauß-Profil handelt (wichtig,
weil dieselben (win_input, width)-Bereiche für beide Profile gescannt werden
können und sonst leicht verwechselbar wären).

Typische Verwendung:

    # nachdem der Scan gelaufen ist:
    #   opt.scan_win_width_amplitude_dependence(...)
    #   opt.save_scan_amp_results()   # landet automatisch in Results/

    from multitone_amplitude_dependence_plots import load_amp_scan_results, AmplitudeScanPlotter, DEFAULT_RESULTS_DIR

    results = load_amp_scan_results(DEFAULT_RESULTS_DIR / "scan_amp_data_N3x4_15x15pts_Airy.pkl")
    plotter = AmplitudeScanPlotter(results)               # out_dir=None -> speichert automatisch in Bilder/
    plotter.plot_scan2d_combined(show=True, save=True)    # 4 Heatmaps nebeneinander
    plotter.plot_dependence_cuts(show=True, save=True)    # r_x/r_y als Kurven entlang fester Schnitte

    # einzelne Heatmaps:
    plotter.plot_scan2d_rx(show=True, save=True)
    plotter.plot_scan2d_ry(show=True, save=True)

    # Achse vor/nach Linse umschalten, Legendengröße, bester Punkt markieren:
    plotter.plot_scan2d_uniformity(win_axis="after_lens", mark_best_point=True)
    plotter.plot_dependence_cuts(legend_fontsize=16, mark_best_point=True)

    # statt des automatisch gefundenen besten Punkts einen SELBST GEWÄHLTEN
    # Punkt für die Schnitte verwenden (win_input_fixed in Metern, width_fixed
    # in Hz - siehe plot_dependence_cuts()-Docstring für Details/Einheiten):
    plotter.plot_dependence_cuts(win_input_fixed=1.1e-3, width_fixed=0.32e6)
    # dasselbe geht auch in den Heatmaps (markiert dort denselben Punkt statt
    # des globalen Optimums):
    plotter.plot_scan2d_combined(win_input_fixed=1.1e-3, width_fixed=0.32e6)

    # ...oder den Punkt lieber in µm AN DER FOKUSEBENE (nach der Linse) statt
    # in mm vor der Linse angeben (win_input_fixed_axis umschalten - der Wert
    # selbst bleibt in Metern):
    plotter.plot_dependence_cuts(win_input_fixed=4.5e-6, win_input_fixed_axis="after_lens")

    # Ausreißer VOR dem Plotten bereinigen (z.B. r_x/r_y=0, an r_bounds
    # gelaufen - physikalisch unplausibel für ein einzelnes Gitterfeld).
    # WICHTIG: erst summarize_amp_bounds() ansehen - eine Schranke, die breit/
    # häufig erreicht wird (z.B. viele Punkte bei r=2), ist meist KEINE
    # Ausreißer, sondern eine echte Randsättigung und sollte NICHT bereinigt
    # werden. Über bounds=("lower",)/("upper",) lässt sich gezielt nur die
    # tatsächlich fehlerhafte Schranke bereinigen:
    from multitone_amplitude_dependence_plots import (
        summarize_amp_bounds, clean_amp_scan_results,
    )
    summarize_amp_bounds(results)  # druckt Häufigkeiten pro Schranke aus
    results_clean = clean_amp_scan_results(
        results, strategy="interpolate", bounds=("lower",),  # nur untere Schranke (r=0) bereinigen
    )  # strategy: "interpolate"/"drop_columns"/"drop_rows"/"nan"
    plotter = AmplitudeScanPlotter(results_clean)   # wirkt dann auf ALLE Plots (Heatmaps + Cuts)

    Siehe auch das eigenständige Beispielskript
    "beispiel_amp_scan_ergebnisse_replotten.py" (im selben Ordner), das genau
    diese Optionen (Achse vor/nach Linse, Legendengröße, bester/eigener
    Schnittpunkt inkl. µm-an-der-Fokusebene, Ausreißer-Bereinigung) an einem
    konkreten, geladenen Ergebnis demonstriert.

Speichern - Kollisionsschutz: wie multitone_flattop_scan_plots.py, siehe
resolve_save_path().
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ======================================================================
# Default-Ordner für Rohdaten (Results) und Bilder (Bilder) - beide direkt
# neben diesem Skript (d.h. im Projektordner "Optimierung Niklas+Claude").
# Werden automatisch angelegt, falls sie noch nicht existieren. Fällt (z.B.
# auf einem anderen Rechner / anderem Pfad) auf einen Ordner relativ zum
# aktuellen Arbeitsverzeichnis zurück, falls der Pfad neben dem Skript aus
# irgendeinem Grund nicht beschreibbar ist.
# ======================================================================
def _default_dir(name):
    candidate = Path(__file__).resolve().parent.parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = Path(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DEFAULT_RESULTS_DIR = _default_dir("Results")
DEFAULT_IMAGES_DIR = _default_dir("Bilder")


# ======================================================================
# Laden der von MultitoneFlatTopOptimizer.save_scan_amp_results()
# gesicherten Rohdaten
# ======================================================================
def load_amp_scan_results(filepath):
    """Lädt ein von MultitoneFlatTopOptimizer.save_scan_amp_results() erzeugtes
    Pickle und gibt das enthaltene dict zurück (siehe get_scan_amp_results()).

    filepath: vollständiger Pfad, ODER nur ein Dateiname - in diesem Fall
    wird zuerst im aktuellen Arbeitsverzeichnis und danach in
    DEFAULT_RESULTS_DIR (dem Standard-"Results"-Ordner) gesucht."""
    path = Path(filepath)
    if not path.is_absolute() and not path.exists() and len(path.parts) == 1:
        candidate = DEFAULT_RESULTS_DIR / path.name
        if candidate.exists():
            path = candidate
    with open(path, 'rb') as f:
        return pickle.load(f)


# ======================================================================
# Leichte, freistehende Physik-Hilfsfunktionen (Duplikate, siehe
# multitone_flattop_scan_plots.py - identisches Muster, damit dieses
# Modul komplett unabhängig bleibt)
# ======================================================================
def radius_from_angle(theta, f1, f2, fLO):
    return (f1 * fLO / f2) * np.tan(theta)


def win_input_to_win(win_input, f1, f2, lambda_opt, fLO):
    """Siehe MultitoneFlatTopOptimizer.win_input_to_win() für die Herleitung
    (3-Linsen-Kaskade f1 -> f2 Teleskop + fLO Fokussierlinse)."""
    if win_input <= 0:
        raise ValueError("win_input muss > 0 sein.")
    return (f1 / f2) * (lambda_opt * fLO) / (np.pi * win_input)


def width_to_um(width_hz, f1, f2, fLO, theta_max, f_band):
    """Siehe MultitoneFlatTopOptimizer.width_to_um()."""
    theta_width = theta_max * width_hz / f_band
    return radius_from_angle(theta_width, f1, f2, fLO) * 1e6


def amps_from_ratio(r, N):
    """Duplikat von multitone_flattop_optimizer.amps_from_ratio() - siehe dort."""
    amp = np.ones(N, dtype=float)
    if N >= 2:
        amp[0] = r
        amp[-1] = r
    elif N == 1:
        amp[0] = r
    return amp


# ======================================================================
# Ausreißer-Erkennung/-Bereinigung VOR dem Plotten
# ======================================================================
# scan_win_width_amplitude_dependence() optimiert r_x/r_y pro Gitterpunkt
# mit Nelder-Mead innerhalb r_bounds (Default (0.0, 2.0)). Landet die innere
# Optimierung an einem Punkt exakt auf einer der beiden Schranken, gibt es
# ZWEI grundsätzlich verschiedene Erklärungen, die man auseinanderhalten
# muss (siehe summarize_amp_bounds()):
#   - VEREINZELT/ISOLIERT (z.B. ein einzelner Punkt oder eine einzelne Spalte
#     bei r=0, umgeben von ganz anderen Werten): meist ein fehlgeschlagener/
#     entarteter Optimierungslauf an genau diesem Punkt - r_x/r_y sollte
#     glatt von (win_input, width) abhängen, ein isolierter Ausreißer ist
#     physikalisch nicht plausibel. Kandidat für Bereinigung.
#   - BREIT/HÄUFIG (die Schranke wird über einen ganzen Bereich hinweg
#     reproduzierbar erreicht, z.B. viele benachbarte Punkte bei r=2): das
#     ist typischerweise KEIN Ausreißer, sondern eine ECHTE Randlösung - das
#     unbeschränkte Optimum liegt dort außerhalb von r_bounds, die
#     Optimierung sättigt konsistent an der Schranke. Das WEGZUBEREINIGEN
#     (interpolieren oder entfernen) würde echte Information verlieren/
#     verfälschen.
# Deshalb sind beide Schranken hier UNABHÄNGIG über den `bounds`-Parameter
# ansteuerbar (Default: beide, wie zuvor) - summarize_amp_bounds() liefert
# die Häufigkeiten pro Schranke, um die Entscheidung "isoliert vs. breit"
# zu treffen, BEVOR man sich für eine Bereinigung entscheidet.
def summarize_amp_bounds(results, r_bounds=None, atol=1e-9, verbose=True):
    """
    Zählt, wie oft r_x bzw. r_y an der UNTEREN bzw. OBEREN r_bounds-Schranke
    liegen - Entscheidungsgrundlage für detect_amp_outliers()/
    clean_amp_scan_results() (siehe Modul-Kommentar oben): viele/
    zusammenhängende Treffer sprechen für eine ECHTE Randsättigung (nicht
    bereinigen), einzelne/verstreute Treffer eher für einen Ausreißer.

    r_bounds: (min, max). Default (None): results['r_bounds'], falls
    vorhanden, sonst (0.0, 2.0).

    Rückgabe: dict {'r_x': {'lower': n, 'upper': n, 'total': n_gültig},
    'r_y': {...}}. Bei verbose=True (Default) wird zusätzlich eine
    lesbare Zusammenfassung ausgedruckt.
    """
    r_bounds = r_bounds if r_bounds is not None else results.get('r_bounds', (0.0, 2.0))
    lo, hi = r_bounds
    counts = {}
    for name, G in (("r_x", results['r_x_grid']), ("r_y", results['r_y_grid'])):
        valid = np.isfinite(G)
        counts[name] = dict(
            lower=int(np.sum(valid & np.isclose(G, lo, atol=atol))),
            upper=int(np.sum(valid & np.isclose(G, hi, atol=atol))),
            total=int(valid.sum()),
        )
    if verbose:
        print(f"Schranken-Analyse (r_bounds=({lo}, {hi})):")
        for name, c in counts.items():
            print(f"  {name}: {c['lower']}/{c['total']} Punkt(e) an unterer Schranke ({lo}), "
                  f"{c['upper']}/{c['total']} an oberer Schranke ({hi}).")
        print("  -> viele/zusammenhängende Treffer an einer Schranke deuten eher auf eine "
              "ECHTE Randsättigung hin (NICHT automatisch bereinigen); einzelne/verstreute "
              "Treffer eher auf einen Ausreißer/fehlgeschlagenen Optimierungslauf (guter "
              "Kandidat für detect_amp_outliers()/clean_amp_scan_results(), ggf. mit "
              "bounds=('lower',) bzw. ('upper',), um nur die betroffene Schranke zu bereinigen).")
    return counts


def detect_amp_outliers(results, r_bounds=None, atol=1e-9, bounds=("lower", "upper")):
    """
    Findet Gitterpunkte, an denen r_x_grid ODER r_y_grid exakt (innerhalb
    atol) an einer der r_bounds-Schranken liegt - siehe Modul-Kommentar oben
    für die Begründung UND WARUM das nicht blind für beide Schranken
    passieren sollte.

    r_bounds: (min, max). Default (None): results['r_bounds'], falls
    vorhanden, sonst (0.0, 2.0) (Optimizer-Default).

    bounds: welche Schranke(n) als Ausreißer-Kandidat zählen -
    ("lower",), ("upper",) oder ("lower", "upper") (Default: beide, wie
    bisher). Vor der Wahl lohnt sich ein Blick auf
    summarize_amp_bounds(results) - wird eine Schranke breit/häufig
    erreicht, ist das meist eine echte Randlösung und sollte NICHT über
    diese Funktion "wegbereinigt" werden (bounds entsprechend einschränken).

    Rückgabe: boolsche Maske der Form (n_width, n_win_input), True an
    Ausreißer-Punkten.
    """
    r_bounds = r_bounds if r_bounds is not None else results.get('r_bounds', (0.0, 2.0))
    lo, hi = r_bounds
    targets = []
    if "lower" in bounds:
        targets.append(lo)
    if "upper" in bounds:
        targets.append(hi)
    RX = results['r_x_grid']
    RY = results['r_y_grid']
    mask = np.zeros(RX.shape, dtype=bool)
    for G in (RX, RY):
        valid = np.isfinite(G)
        for target in targets:
            mask |= valid & np.isclose(G, target, atol=atol)
    return mask


def _fill_from_neighbors(grid, bad_mask):
    """Ersetzt jeden mit bad_mask markierten Gitterpunkt durch den Median
    seiner gültigen (nicht selbst markierten, nicht-NaN) direkten
    Nachbarn (oben/unten/links/rechts). Punkte ohne einen einzigen
    gültigen Nachbarn (z.B. Ecke, komplett von Ausreißern umgeben) bleiben
    unverändert - kommt bei vereinzelten Ausreißern praktisch nie vor."""
    out = grid.copy()
    ny, nx = grid.shape
    bad = bad_mask | ~np.isfinite(grid)
    for i in range(ny):
        for j in range(nx):
            if not bad[i, j]:
                continue
            neighbor_vals = [
                grid[ii, jj]
                for ii, jj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                if 0 <= ii < ny and 0 <= jj < nx and not bad[ii, jj]
            ]
            if neighbor_vals:
                out[i, j] = float(np.median(neighbor_vals))
    return out


def clean_amp_scan_results(results, mask=None, strategy="interpolate", r_bounds=None,
                            atol=1e-9, bounds=("lower", "upper"), verbose=True):
    """
    Bereinigt Ausreißer-Punkte in einer KOPIE von results (das Original wird
    NICHT verändert) - vor dem Plotten aufrufen, damit sowohl Heatmaps als
    auch plot_dependence_cuts() automatisch die bereinigten Werte sehen
    (beide arbeiten nur auf dem übergebenen dict, unabhängig davon, ob es
    "roh" oder bereinigt ist).

    mask: optionale eigene boolsche Maske der Form (n_width, n_win_input),
    True an zu bereinigenden Punkten. Default (None):
    detect_amp_outliers(results, r_bounds, atol, bounds) - automatische
    Erkennung von Punkten, an denen r_x/r_y exakt an einer r_bounds-Schranke
    liegt.

    bounds: nur relevant, wenn mask=None (sonst ignoriert) - wird direkt an
    detect_amp_outliers() durchgereicht: ("lower",), ("upper",) oder
    ("lower", "upper") (Default: beide). Siehe Modul-Kommentar oben bzw.
    summarize_amp_bounds() - eine breit/häufig erreichte Schranke ist meist
    KEIN Ausreißer, sondern eine echte Randsättigung und sollte über diesen
    Parameter von der Bereinigung ausgenommen werden.

    strategy:
      - "interpolate" (Default): jeder Ausreißer-Punkt wird durch den
        Median seiner gültigen direkten Gitternachbarn ersetzt (in r_x_grid,
        r_y_grid, uniformity_grid UND crosstalk_grid an denselben Stellen,
        damit alle vier Größen konsistent zueinander bleiben - "auf die
        Umgebungswerte setzen").
      - "drop_columns": entfernt jede win_input-SPALTE, die mindestens
        einen Ausreißer enthält, komplett aus dem Gitter (win_input_vals
        UND alle vier *_grid entsprechend gekürzt) - "die Spalte weglassen".
      - "drop_rows": wie "drop_columns", aber für width-ZEILEN.
      - "nan": setzt Ausreißer-Punkte in allen vier Grids auf NaN (erscheint
        als Lücke/weißer Fleck in Heatmaps bzw. Lücke in den Dependence-
        Cuts) - ohne Interpolation und ohne ganze Zeilen/Spalten zu
        entfernen.

    verbose: gibt eine kurze Zusammenfassung aus (Anzahl gefundener/
    bereinigter Punkte bzw. entfernter Zeilen/Spalten).
    """
    if mask is None:
        mask = detect_amp_outliers(results, r_bounds=r_bounds, atol=atol, bounds=bounds)

    n_found = int(mask.sum())
    if verbose:
        print(f"clean_amp_scan_results: {n_found} Ausreißer-Punkt(e) gefunden "
              f"(r_x/r_y an r_bounds-Schranke), strategy='{strategy}'.")
    if n_found == 0:
        return dict(results)

    cleaned = dict(results)  # shallow copy - Original bleibt unangetastet

    if strategy == "nan":
        for key in ("r_x_grid", "r_y_grid", "uniformity_grid", "crosstalk_grid"):
            G = results[key].copy()
            G[mask] = np.nan
            cleaned[key] = G

    elif strategy == "interpolate":
        for key in ("r_x_grid", "r_y_grid", "uniformity_grid", "crosstalk_grid"):
            cleaned[key] = _fill_from_neighbors(results[key], mask)

    elif strategy in ("drop_columns", "drop_rows"):
        if strategy == "drop_columns":
            bad_axis = np.any(mask, axis=0)  # je win_input-Spalte
            keep = ~bad_axis
            if verbose:
                print(f"  -> entferne {int(bad_axis.sum())} von {len(keep)} win_input-Spalte(n).")
            cleaned['win_input_vals'] = results['win_input_vals'][keep]
            for key in ("r_x_grid", "r_y_grid", "uniformity_grid", "crosstalk_grid"):
                cleaned[key] = results[key][:, keep]
        else:
            bad_axis = np.any(mask, axis=1)  # je width-Zeile
            keep = ~bad_axis
            if verbose:
                print(f"  -> entferne {int(bad_axis.sum())} von {len(keep)} width-Zeile(n).")
            cleaned['width_vals'] = results['width_vals'][keep]
            for key in ("r_x_grid", "r_y_grid", "uniformity_grid", "crosstalk_grid"):
                cleaned[key] = results[key][keep, :]
    else:
        raise ValueError(
            f"strategy muss 'interpolate', 'drop_columns', 'drop_rows' oder 'nan' sein, nicht {strategy!r}."
        )

    return cleaned


# ======================================================================
# Kollisionsschutz beim Speichern (identisch zu multitone_flattop_scan_plots.py)
# ======================================================================
def resolve_save_path(out_dir, filename, confirm_overwrite=None):
    """
    Gibt den Pfad zurück, unter dem eine Grafik gespeichert werden soll.
    Siehe multitone_flattop_scan_plots.resolve_save_path() für Details.
    """
    path = Path(out_dir) / filename
    if not path.exists():
        return path

    if confirm_overwrite is None:
        def confirm_overwrite(existing_path):
            answer = input(
                f"'{existing_path.name}' existiert bereits. Überschreiben? [y/N]: "
            ).strip().lower()
            return answer in ("y", "yes", "j", "ja")

    if confirm_overwrite(path):
        return path

    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            print(f"Bestehende Datei bleibt erhalten, speichere stattdessen als: {candidate.name}")
            return candidate
        counter += 1


# ======================================================================
# Plotter
# ======================================================================
class AmplitudeScanPlotter:
    """
    Erzeugt alle Plots für den Amplituden-Abhängigkeits-Scan (Heatmaps von
    r_x_opt/r_y_opt über (win_input, width), plus Abhängigkeits-Schnitte)
    aus einem von get_scan_amp_results()/load_amp_scan_results() gelieferten
    dict - ohne den (potenziell langsamen) Scan selbst erneut zu berechnen.
    """

    SCAN2D_RC = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
    }
    SCAN2D_SAVE_DPI = 300

    # Diese beiden Grids sind dimensionslose Verhältnisse (0-1) und werden auf
    # der Colorbar (und NUR dort - Rohdaten in results bleiben unverändert als
    # Fraktion) als Prozent (0-100) dargestellt, siehe _plot_scan2d_metric()/
    # _build_combined_figure(). r_x_grid/r_y_grid (Amplitudenverhältnisse)
    # sind KEINE Prozentgrößen und bleiben unskaliert.
    PERCENT_METRIC_KEYS = ("uniformity_grid", "crosstalk_grid")

    # Marker-Stil für den markierten Punkt (automatisch gefundenes globales
    # Optimum von alpha*Uniformity+(1-alpha)*Crosstalk, ODER ein selbst
    # gewählter Punkt via win_input_fixed/width_fixed - siehe
    # _resolve_mark_point()).
    BEST_POINT_STYLE = dict(
        marker="*", markersize=20, markeredgecolor="white",
        markeredgewidth=1.3, color="red", zorder=8,
    )

    def __init__(self, results, out_dir=None, confirm_overwrite=None):
        """
        results: dict von get_scan_amp_results()/load_amp_scan_results(),
        enthält u.a. win_input_vals, width_vals, uniformity_grid,
        crosstalk_grid, r_x_grid, r_y_grid, alpha, r_bounds, N_x, N_y, f1,
        f2, fLO, lambda_opt, theta_max, f_band, profile ("airy"/"gaussian").

        out_dir: Ordner, in dem Bilder gespeichert werden. Default (None):
        DEFAULT_IMAGES_DIR, d.h. der "Bilder"-Ordner neben diesem Skript -
        wird bei Bedarf automatisch angelegt.

        confirm_overwrite: optionale Callable(Path) -> bool, wird an
        resolve_save_path() weitergereicht (z.B. für einen Qt-Dialog statt
        der Konsolen-Eingabe).
        """
        missing = [k for k in ("r_x_grid", "r_y_grid", "uniformity_grid", "crosstalk_grid",
                                "win_input_vals", "width_vals") if k not in results]
        if missing:
            raise ValueError(
                "results fehlen die Schlüssel " + ", ".join(missing) + " - das sieht nicht "
                "nach den Rohdaten des AMPLITUDEN-Scans aus. AmplitudeScanPlotter braucht "
                "das Ergebnis von get_scan_amp_results()/save_scan_amp_results() bzw. eine "
                "damit gespeicherte Pickle-Datei (typischerweise 'scan_amp_data_...pkl').\n"
                "Häufigste Ursache: eine 'scan_data_...pkl'-Datei geladen (OHNE 'amp' im "
                "Namen) - das sind die Rohdaten des EINFACHEN Uniformity/Crosstalk-Scans "
                "(scan_win_width_uniformity()/get_scan_results()), OHNE r_x_grid/r_y_grid, "
                "weil dort die Amplituden fest vorgegeben waren statt pro Punkt optimiert zu "
                "werden. Für diese Datei braucht es stattdessen ScanPlotter aus "
                "multitone_flattop_scan_plots.py, nicht AmplitudeScanPlotter."
            )
        self.results = results
        self.out_dir = Path(out_dir) if out_dir is not None else DEFAULT_IMAGES_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_overwrite = confirm_overwrite

    # ------------------------------------------------------------------
    def _win_input_to_win(self, win_input):
        r = self.results
        return win_input_to_win(win_input, r['f1'], r['f2'], r['lambda_opt'], r['fLO'])

    def _width_to_um(self, width_hz):
        r = self.results
        return width_to_um(width_hz, r['f1'], r['f2'], r['fLO'], r['theta_max'], r['f_band'])

    def _profile_tag(self):
        """'Airy' / 'Gauss', oder 'ProfilUnbekannt' für ältere Pickles ohne
        gespeichertes 'profile' (vor 2026-08-18)."""
        profile = self.results.get('profile')
        if profile == 'airy':
            return 'Airy'
        if profile == 'gaussian':
            return 'Gauss'
        return 'ProfilUnbekannt'

    def _filetag(self):
        """Dateinamens-Tag: Tonanzahl, Anzahl der Gitterpunkte im Bild
        (win_input x width) und Strahlprofil - z.B. 'N3x4_15x15pts_Airy'."""
        r = self.results
        n_win = len(r.get('win_input_vals', []))
        n_width = len(r.get('width_vals', []))
        return f"N{r['N_x']}x{r['N_y']}_{n_win}x{n_width}pts_{self._profile_tag()}"

    def _finish_figure(self, fig, filename, show, save, dpi=150):
        if save:
            out_file = resolve_save_path(self.out_dir, filename, confirm_overwrite=self.confirm_overwrite)
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {out_file}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _point_info_lines(self, win_input_val, width_val, win_eff, uniformity, crosstalk, r_x, r_y):
        r = self.results
        amp_x = amps_from_ratio(r_x, r['N_x'])
        amp_y = amps_from_ratio(r_y, r['N_y'])
        lines = [
            rf"$\omega_{{\mathrm{{in}}}}$ (before lenses):  {win_input_val*1e3:.4f} mm",
            rf"$\omega'$ (at focus):            {win_eff*1e6:.4f} µm" if win_eff is not None
            else r"$\omega'$ (at focus):            invalid ($\omega_{\mathrm{in}} \leq 0$)",
            rf"width:                       {width_val*1e-6:.4f} MHz",
        ]
        if uniformity is not None and np.isfinite(uniformity):
            lines.append(rf"Uniformity ($\sigma/\mu$, minimiert):  {uniformity*100:.3f} %")
            lines.append(rf"Crosstalk ($\eta$, bei diesem Optimum):  {crosstalk*100:.3f} %")
            lines.append(rf"$r_x$ (outer/inner, $N_x$={r['N_x']}):  {r_x:.4f}")
            lines.append(rf"$r_y$ (outer/inner, $N_y$={r['N_y']}):  {r_y:.4f}")
            lines.append(rf"$a_x$:  {np.array2string(amp_x, precision=3)}")
            lines.append(rf"$a_y$:  {np.array2string(amp_y, precision=3)}")
        else:
            lines.append("Invalid point (no optimum found).")
        return lines

    # ------------------------------------------------------------------
    # win_input-Achse: "mm vor der Linse" (Default, wie gescannt) oder
    # "µm nach der Linse" (effektiver Waist an der Fokus-/Trap-Ebene).
    # ------------------------------------------------------------------
    def _win_axis_values(self, win_input_vals, win_axis):
        """
        Rechnet win_input_vals (Meter, wie im Scan-Gitter gespeichert,
        aufsteigend sortiert) in die gewünschte Anzeige-Einheit um.

        win_axis="before_lens" (Default): unverändert, in mm (identisch zum
        bisherigen Verhalten).
        win_axis="after_lens": effektiver Waist an der Fokus-/Trap-Ebene
        NACH allen Linsen, in µm, über win_input_to_win() umgerechnet.
        Diese Umrechnung ist eine STRENG FALLENDE Funktion von win_input
        (größerer Eingangs-Waist -> kleinerer Fokus-Waist), das Ergebnis ist
        also absteigend sortiert.

        Rückgabe: (x_vals, x_label, reversed_) - x_vals ist IMMER aufsteigend
        sortiert (für pcolormesh); reversed_=True zeigt an, dass dafür die
        Reihenfolge gegenüber win_input_vals umgedreht wurde (der Aufrufer
        muss dann auch alle entlang dieser Achse aufgetragenen Daten mit
        [::-1] umdrehen, siehe _plot_scan2d_metric()/_build_combined_figure()).
        """
        if win_axis == "before_lens":
            return win_input_vals * 1e3, r"Input waist $\omega_{\mathrm{in}}$ (before lenses, mm)", False
        elif win_axis == "after_lens":
            x = np.array([self._win_input_to_win(w) for w in win_input_vals]) * 1e6
            if len(x) > 1 and x[0] > x[-1]:
                return x[::-1], r"Waist at focus $\omega'$ (after lenses, µm)", True
            return x, r"Waist at focus $\omega'$ (after lenses, µm)", False
        else:
            raise ValueError(f"win_axis muss 'before_lens' oder 'after_lens' sein, nicht {win_axis!r}.")

    def _win_axis_single_value(self, win_input_val, win_axis):
        """Wie _win_axis_values(), aber für einen einzelnen win_input-Wert
        (z.B. um den markierten Punkt auf der gerade gewählten Achse zu
        platzieren) - gibt nur den transformierten Wert zurück."""
        return self._win_axis_values(np.array([win_input_val]), win_axis)[0][0]

    # ------------------------------------------------------------------
    # Markierter Punkt: entweder (Default) das globale Minimum von
    # alpha*Uniformity + (1-alpha)*Crosstalk - derselbe Punkt, den
    # plot_dependence_cuts() standardmäßig als Ursprung seiner beiden
    # Schnitte verwendet - ODER ein selbst gewählter Punkt via
    # win_input_fixed/width_fixed (auf den nächstgelegenen Gitterpunkt
    # gerundet). Ein einziger Helper für Heatmaps UND Dependence-Cuts, damit
    # beide immer denselben Punkt zeigen.
    # ------------------------------------------------------------------
    def _resolve_mark_point(self, win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """
        win_input_fixed / width_fixed (Hz): optional selbst gewählte Werte -
        werden auf den jeweils nächstgelegenen Gitterwert gerundet. Ist nur
        einer von beiden gesetzt, wird für die jeweils andere Achse
        weiterhin der Wert am globalen Optimum verwendet (identisches
        Verhalten wie win_input_fixed/width_fixed in plot_dependence_cuts()).

        win_input_fixed_axis: legt fest, WIE win_input_fixed zu interpretieren
        ist (win_input_fixed selbst bleibt IMMER in Metern - SI):
          - "before_lens" (Default): win_input_fixed ist der Eingangs-Waist
            VOR der ersten Linse (identisch zur win_input_vals-Achse des
            Gitters, unverändertes bisheriges Verhalten).
          - "after_lens": win_input_fixed ist stattdessen der EFFEKTIVE
            Waist an der Fokus-/Trap-Ebene NACH allen Linsen (dieselbe
            Größe, die win_axis="after_lens" anzeigt) - nützlich, wenn man
            in dieser Einheit denkt/misst. Die Suche nach dem nächstgelegenen
            Gitterpunkt vergleicht dann win_input_to_win(win_input_vals)
            statt win_input_vals direkt gegen win_input_fixed.

        Rückgabe: (i, j, label) - Gitterindizes (i=width, j=win_input) plus
        Beschriftung für die Legende ("best point (global optimum)" bzw.
        "selected point (cut origin)", je nachdem ob ein Override gesetzt
        wurde). None, falls WEDER ein Override gesetzt ist NOCH ein
        automatisches Optimum bestimmbar ist (z.B. komplett leeres/
        ungültiges Scan-Gitter).
        """
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        U = r['uniformity_grid']
        C = r['crosstalk_grid']
        alpha = r.get('alpha', 1.0)

        manual = win_input_fixed is not None or width_fixed is not None
        has_valid = np.any(np.isfinite(U))
        if not manual and not has_valid:
            return None

        idx_min = (0, 0)
        if has_valid:
            combined = alpha * U + (1 - alpha) * C
            idx_min = np.unravel_index(np.nanargmin(combined), combined.shape)

        i = idx_min[0] if width_fixed is None else int(np.argmin(np.abs(width_vals - width_fixed)))
        if win_input_fixed is None:
            j = idx_min[1]
        elif win_input_fixed_axis == "before_lens":
            j = int(np.argmin(np.abs(win_input_vals - win_input_fixed)))
        elif win_input_fixed_axis == "after_lens":
            win_eff_vals = np.array([self._win_input_to_win(w) for w in win_input_vals])
            j = int(np.argmin(np.abs(win_eff_vals - win_input_fixed)))
        else:
            raise ValueError(
                f"win_input_fixed_axis muss 'before_lens' oder 'after_lens' sein, "
                f"nicht {win_input_fixed_axis!r}."
            )
        label = "selected point (cut origin)" if manual else "best point (global optimum)"
        return i, j, label

    # ------------------------------------------------------------------
    # Einzelne Heatmap für eine beliebige der vier Größen
    # ------------------------------------------------------------------
    def plot_scan2d_rx(self, show=True, save=True, cmap="viridis", vmin=None, vmax=None,
                        win_axis="before_lens", mark_best_point=True,
                        win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """Heatmap von r_x_opt(win_input, width) - Außen/Innen-Amplitudenverhältnis
        für die N_x=3-Achse, bei minimierter Uniformity. win_input_fixed/
        width_fixed: siehe plot_dependence_cuts()."""
        self._plot_scan2d_metric(
            grid_key="r_x_grid", colorbar_label=r"$r_x$ (outer/inner)",
            title_metric=r"Optimal $r_x$", filename_suffix="rx",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def plot_scan2d_ry(self, show=True, save=True, cmap="viridis", vmin=None, vmax=None,
                        win_axis="before_lens", mark_best_point=True,
                        win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """Heatmap von r_y_opt(win_input, width) - Außen/Innen-Amplitudenverhältnis
        für die N_y=4-Achse, bei minimierter Uniformity. win_input_fixed/
        width_fixed: siehe plot_dependence_cuts()."""
        self._plot_scan2d_metric(
            grid_key="r_y_grid", colorbar_label=r"$r_y$ (outer/inner)",
            title_metric=r"Optimal $r_y$", filename_suffix="ry",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def plot_scan2d_uniformity(self, show=True, save=True, cmap="viridis_r", vmin=None, vmax=None,
                                win_axis="before_lens", mark_best_point=True,
                                win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """Heatmap der Uniformity am (r_x,r_y)-Optimum (alpha*uniformity+(1-alpha)*eta
        minimiert, siehe self.results['alpha']).

        Farbschema: "viridis_r" (Default, statt vormals "magma") - perzeptuell
        gleichmäßig, farbenblind-sicher, UND niedrige Werte (0 % Uniformity =
        das eigentliche Ziel der Optimierung) werden hell/gelb statt fast
        schwarz dargestellt. Bei "magma" lag 0 nahe Schwarz, was gerade die
        besten (niedrigsten) Punkte am schwersten lesbar machte - "viridis_r"
        behebt das, indem die Helligkeits-Richtung umgedreht wird: niedrig =
        hell, hoch = dunkel.

        win_input_fixed/width_fixed: siehe plot_dependence_cuts()."""
        alpha = self.results.get('alpha')
        self._plot_scan2d_metric(
            grid_key="uniformity_grid", colorbar_label="Uniformity (σ/μ) at r-optimum (%)",
            title_metric=f"Uniformity (r-optimum, α={alpha:.2f})" if alpha is not None else "Uniformity (r-optimum)",
            filename_suffix="Uniformity",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def plot_scan2d_crosstalk(self, show=True, save=True, cmap="Oranges", vmin=None, vmax=None,
                               win_axis="before_lens", mark_best_point=True,
                               win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """Heatmap des Crosstalk am (r_x,r_y)-Optimum (alpha*uniformity+(1-alpha)*eta
        minimiert, siehe self.results['alpha']). win_input_fixed/width_fixed:
        siehe plot_dependence_cuts()."""
        alpha = self.results.get('alpha')
        self._plot_scan2d_metric(
            grid_key="crosstalk_grid", colorbar_label="Crosstalk (η) at r-optimum (%)",
            title_metric=f"Crosstalk (r-optimum, α={alpha:.2f})" if alpha is not None else "Crosstalk (r-optimum)",
            filename_suffix="Crosstalk",
            cmap=cmap, vmin=vmin, vmax=vmax, show=show, save=save,
            win_axis=win_axis, mark_best_point=mark_best_point,
            win_input_fixed=win_input_fixed, width_fixed=width_fixed,
            win_input_fixed_axis=win_input_fixed_axis,
        )

    def _plot_scan2d_metric(self, grid_key, colorbar_label, title_metric, filename_suffix,
                             cmap, vmin, vmax, show, save, win_axis="before_lens",
                             mark_best_point=True, win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        Z = r[grid_key]

        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        win_order = win_input_vals[::-1] if reversed_ else win_input_vals
        Z_plot = Z[:, ::-1] if reversed_ else Z

        # Uniformity/Crosstalk sind dimensionslose Verhältnisse (0-1) - auf der
        # Colorbar besser lesbar als Prozent (0-100), siehe PERCENT_METRIC_KEYS.
        # r_x/r_y bleiben unverändert (Amplitudenverhältnis, keine Prozentgröße).
        if grid_key in self.PERCENT_METRIC_KEYS:
            Z_plot = Z_plot * 100.0
            plot_vmin = None if vmin is None else vmin * 100.0
            plot_vmax = None if vmax is None else vmax * 100.0
        else:
            plot_vmin, plot_vmax = vmin, vmax

        with plt.rc_context(self.SCAN2D_RC):
            fig = plt.figure(figsize=(9, 8.9))
            ax = fig.add_axes([0.12, 0.36, 0.75, 0.56])
            info_ax = fig.add_axes([0.06, 0.03, 0.88, 0.24])
            info_ax.axis('off')

            im = ax.pcolormesh(
                x_vals, width_vals * 1e-6, Z_plot,
                shading='auto', cmap=cmap, vmin=plot_vmin, vmax=plot_vmax,
            )
            fig.colorbar(im, ax=ax, label=colorbar_label)

            dx = (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 0.1
            dy = (width_vals[1] - width_vals[0]) * 1e-6 if len(width_vals) > 1 else 0.05
            x_pad, y_pad = dx / 2, dy / 2
            ax.set_xlim(x_vals[0] - x_pad, x_vals[-1] + x_pad)
            ax.set_ylim(width_vals[0] * 1e-6 - y_pad, width_vals[-1] * 1e-6 + y_pad)

            ax.set_xlabel(x_label)
            ax.set_ylabel("width (MHz)")
            ax.set_title(title_metric)

            selection_rect = Rectangle(
                (0, 0), dx, dy, edgecolor="red", facecolor="none",
                linewidth=2.5, zorder=6,
            )
            selection_rect.set_visible(False)
            ax.add_patch(selection_rect)

            U = r['uniformity_grid']
            C = r['crosstalk_grid']
            RX = r['r_x_grid']
            RY = r['r_y_grid']

            if mark_best_point:
                mark = self._resolve_mark_point(win_input_fixed, width_fixed, win_input_fixed_axis)
                if mark is not None:
                    i_mark, j_mark, mark_label = mark
                    x_mark = self._win_axis_single_value(win_input_vals[j_mark], win_axis)
                    y_mark = width_vals[i_mark] * 1e-6
                    ax.plot(x_mark, y_mark, linestyle="none",
                            label=mark_label, **self.BEST_POINT_STYLE)
                    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

            info_text = info_ax.text(
                0.0, 1.0, "Click on a point in the heatmap to show all parameters "
                           "(incl. amplitude ratios r_x, r_y).",
                va='top', ha='left', fontsize=13, family='monospace',
                transform=info_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9),
            )

            def _on_click(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return
                j_plot = int(np.argmin(np.abs(x_vals - event.xdata)))
                win_input_actual = win_order[j_plot]
                width_val = event.ydata * 1e6
                j = int(np.argmin(np.abs(win_input_vals - win_input_actual)))
                i = int(np.argmin(np.abs(width_vals - width_val)))
                win_input_actual = win_input_vals[j]
                width_actual = width_vals[i]

                selection_rect.set_xy((x_vals[j_plot] - dx / 2, width_actual * 1e-6 - dy / 2))
                selection_rect.set_visible(True)

                try:
                    win_eff = self._win_input_to_win(win_input_actual)
                except ValueError:
                    win_eff = None

                lines = self._point_info_lines(
                    win_input_actual, width_actual, win_eff,
                    U[i, j], C[i, j], RX[i, j], RY[i, j],
                )
                info_text.set_text("\n".join(lines))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', _on_click)

            tag = self._filetag()
            self._finish_figure(
                fig, f"FlatMultiTone_AmpScan_{filename_suffix}_{tag}.png",
                show, save, dpi=self.SCAN2D_SAVE_DPI,
            )

    # ------------------------------------------------------------------
    # Kombinierte Ansicht: Uniformity, Crosstalk, r_x, r_y - 4 Heatmaps
    # ------------------------------------------------------------------
    def plot_scan2d_combined(self, show=True, save=True,
                              cmap_uniformity="viridis_r", cmap_crosstalk="Oranges",
                              cmap_rx="viridis", cmap_ry="viridis",
                              win_axis="before_lens", mark_best_point=True,
                              win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        """
        Plottet alle vier Größen nebeneinander (2x2): minimierte Uniformity,
        Crosstalk am jeweiligen Optimum, sowie r_x_opt und r_y_opt - die
        direkte Antwort auf "wie hängen die optimalen Amplituden von Waist
        und Width ab". Jede Heatmap hat ihre eigene Colorbar.

        cmap_uniformity: Default jetzt "viridis_r" statt vormals "magma" -
        siehe plot_scan2d_uniformity() für die Begründung (0 % darf nicht
        fast schwarz sein).

        win_axis: "before_lens" (Default, mm vor der ersten Linse) oder
        "after_lens" (µm, effektiver Waist an der Fokus-/Trap-Ebene) - gilt
        für die win_input-Achse (x) aller vier Panels.

        mark_best_point: markiert in jedem Panel einen einzelnen Punkt
        (Stern-Symbol) - siehe win_input_fixed/width_fixed.

        win_input_fixed (Meter) / width_fixed (Hz): optional - werden
        NICHT zum Schneiden verwendet (das macht nur plot_dependence_cuts()),
        bestimmen hier NUR, welcher Punkt markiert wird. Default (beide
        None): automatisch der Gitterpunkt mit global minimalem
        alpha*Uniformity+(1-alpha)*Crosstalk - derselbe Punkt, durch den
        plot_dependence_cuts() standardmäßig seine Schnitte legt. Wird
        einer oder beide gesetzt, wird stattdessen der nächstgelegene
        Gitterpunkt zu diesen Werten markiert (identische Rundung wie in
        plot_dependence_cuts()) - so lässt sich in den Heatmaps genau der
        Punkt sichtbar machen, für den man sich (z.B. in
        plot_dependence_cuts()) einen eigenen Schnitt gebaut hat.

        win_input_fixed_axis: "before_lens" (Default) oder "after_lens" -
        legt fest, in welcher Größe win_input_fixed zu verstehen ist (immer
        in Metern - SI): Eingangs-Waist vor der ersten Linse, oder
        effektiver Waist an der Fokus-/Trap-Ebene nach allen Linsen. Siehe
        plot_dependence_cuts() für ein Beispiel mit µm-Werten.

        Zwei unabhängige Figuren werden je nach Bedarf gebaut - GESPEICHERT
        (sauber, ohne Infofeld/Auswahlrahmen) und ANGEZEIGT (zusätzlich
        interaktiv: ein Klick in irgendeine der vier Heatmaps markiert die
        Zelle in ALLEN VIEREN und zeigt r_x, r_y, Uniformity, Crosstalk und
        die vollen Amplituden-Arrays im Infofeld darunter).
        """
        if save:
            fig_save = self._build_combined_figure(interactive=False,
                                                     cmap_uniformity=cmap_uniformity, cmap_crosstalk=cmap_crosstalk,
                                                     cmap_rx=cmap_rx, cmap_ry=cmap_ry,
                                                     win_axis=win_axis, mark_best_point=mark_best_point,
                                                     win_input_fixed=win_input_fixed, width_fixed=width_fixed,
                                                     win_input_fixed_axis=win_input_fixed_axis)
            tag = self._filetag()
            out_file = resolve_save_path(
                self.out_dir, f"FlatMultiTone_AmpScan_Combined_{tag}.png",
                confirm_overwrite=self.confirm_overwrite,
            )
            fig_save.savefig(out_file, dpi=self.SCAN2D_SAVE_DPI, bbox_inches='tight')
            print(f"Figure saved: {out_file}")
            plt.close(fig_save)

        if show:
            self._build_combined_figure(interactive=True,
                                         cmap_uniformity=cmap_uniformity, cmap_crosstalk=cmap_crosstalk,
                                         cmap_rx=cmap_rx, cmap_ry=cmap_ry,
                                         win_axis=win_axis, mark_best_point=mark_best_point,
                                         win_input_fixed=win_input_fixed, width_fixed=width_fixed,
                                         win_input_fixed_axis=win_input_fixed_axis)
            plt.show()

    def _build_combined_figure(self, interactive, cmap_uniformity, cmap_crosstalk, cmap_rx, cmap_ry,
                                win_axis="before_lens", mark_best_point=True,
                                win_input_fixed=None, width_fixed=None, win_input_fixed_axis="before_lens"):
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        U = r['uniformity_grid']
        C = r['crosstalk_grid']
        RX = r['r_x_grid']
        RY = r['r_y_grid']

        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        win_order = win_input_vals[::-1] if reversed_ else win_input_vals

        alpha = r.get('alpha')
        alpha_tag = f", α={alpha:.2f}" if alpha is not None else ""
        panels = [
            ("uniformity_grid", "Uniformity (σ/μ) at r-optimum (%)", f"Uniformity (r-optimum{alpha_tag})", cmap_uniformity),
            ("crosstalk_grid", "Crosstalk (η) at r-optimum (%)", f"Crosstalk (r-optimum{alpha_tag})", cmap_crosstalk),
            ("r_x_grid", r"$r_x$ (outer/inner)", r"Optimal $r_x$ ($N_x$=%d)" % r['N_x'], cmap_rx),
            ("r_y_grid", r"$r_y$ (outer/inner)", r"Optimal $r_y$ ($N_y$=%d)" % r['N_y'], cmap_ry),
        ]

        dx = (x_vals[1] - x_vals[0]) if len(x_vals) > 1 else 0.1
        dy = (width_vals[1] - width_vals[0]) * 1e-6 if len(width_vals) > 1 else 0.05

        mark = self._resolve_mark_point(win_input_fixed, width_fixed, win_input_fixed_axis) if mark_best_point else None
        if mark is not None:
            i_mark, j_mark, _ = mark
            x_mark = self._win_axis_single_value(win_input_vals[j_mark], win_axis)
            y_mark = width_vals[i_mark] * 1e-6

        with plt.rc_context(self.SCAN2D_RC):
            if interactive:
                fig = plt.figure(figsize=(15.5, 12.5))
                gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.3], hspace=0.4, wspace=0.3)
                axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
                info_ax = fig.add_subplot(gs[2, :])
                info_ax.axis('off')
            else:
                fig, axes2d = plt.subplots(2, 2, figsize=(14.5, 11.5))
                axes = list(axes2d.ravel())

            selection_rects = []
            for ax, (grid_key, cbar_label, title, cmap) in zip(axes, panels):
                Z = r[grid_key]
                Z_plot = Z[:, ::-1] if reversed_ else Z
                if grid_key in self.PERCENT_METRIC_KEYS:
                    Z_plot = Z_plot * 100.0
                im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading='auto', cmap=cmap)
                x_pad, y_pad = dx / 2, dy / 2
                ax.set_xlim(x_vals[0] - x_pad, x_vals[-1] + x_pad)
                ax.set_ylim(width_vals[0] * 1e-6 - y_pad, width_vals[-1] * 1e-6 + y_pad)
                fig.colorbar(im, ax=ax, label=cbar_label)
                ax.set_xlabel(x_label)
                ax.set_ylabel("width (MHz)")
                ax.set_title(title)

                if mark is not None:
                    ax.plot(x_mark, y_mark, linestyle="none", **self.BEST_POINT_STYLE)

                if interactive:
                    rect = Rectangle((0, 0), dx, dy, edgecolor="red", facecolor="none",
                                      linewidth=2.2, zorder=6)
                    rect.set_visible(False)
                    ax.add_patch(rect)
                    selection_rects.append(rect)

            if not interactive:
                fig.tight_layout()
                return fig

            fig.subplots_adjust(left=0.06, right=0.96, top=0.95, bottom=0.04)

            info_text = info_ax.text(
                0.0, 1.0, "Click on a point in any heatmap to show all parameters "
                           "(incl. amplitude ratios r_x, r_y and the full amplitude arrays).",
                va='top', ha='left', fontsize=13, family='monospace',
                transform=info_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9),
            )

            def _on_click(event):
                if event.inaxes not in axes or event.xdata is None or event.ydata is None:
                    return
                j_plot = int(np.argmin(np.abs(x_vals - event.xdata)))
                win_input_actual = win_order[j_plot]
                width_val = event.ydata * 1e6
                j = int(np.argmin(np.abs(win_input_vals - win_input_actual)))
                i = int(np.argmin(np.abs(width_vals - width_val)))
                win_input_actual = win_input_vals[j]
                width_actual = width_vals[i]

                xy = (x_vals[j_plot] - dx / 2, width_actual * 1e-6 - dy / 2)
                for rect in selection_rects:
                    rect.set_xy(xy)
                    rect.set_visible(True)

                try:
                    win_eff = self._win_input_to_win(win_input_actual)
                except ValueError:
                    win_eff = None

                lines = self._point_info_lines(
                    win_input_actual, width_actual, win_eff,
                    U[i, j], C[i, j], RX[i, j], RY[i, j],
                )
                info_text.set_text("\n".join(lines))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', _on_click)
            return fig

    # ------------------------------------------------------------------
    # Abhängigkeits-Schnitte: r_x/r_y als Kurven entlang eines festen
    # win_input- bzw. width-Werts - die "klassische" Abhängigkeitsdarstellung
    # (y = f(x)) statt Heatmap.
    # ------------------------------------------------------------------
    def plot_dependence_cuts(self, win_input_fixed=None, width_fixed=None, show=True, save=True,
                              win_axis="before_lens", legend_fontsize=None, mark_best_point=True,
                              win_input_fixed_axis="before_lens"):
        """
        Zeigt r_x_opt und r_y_opt als Kurven entlang zweier fester Schnitte
        durch das Gitter:

        - links: r_x_opt(win_input) und r_y_opt(win_input) bei festem width
        - rechts: r_x_opt(width) und r_y_opt(width) bei festem win_input

        win_input_fixed / width_fixed (Hz): legen den Schnittpunkt SELBST
        fest, statt ihn automatisch zu suchen - jeweils auf den
        nächstgelegenen Gitterwert gerundet. Default (beide None):
        automatisch der insgesamt niedrigste kombinierte Wert
        alpha*uniformity+(1-alpha)*eta (dasselbe Ziel, das pro Punkt beim
        Scan minimiert wurde) - width_fixed bestimmt die Zeile für den
        LINKEN Schnitt, win_input_fixed die Spalte für den RECHTEN Schnitt.
        Ist nur einer von beiden gesetzt, wird für die andere Achse weiter
        automatisch das globale Optimum verwendet. Beispiel: um den Schnitt
        durch einen selbst gewählten Punkt bei win_input=1.1 mm (vor der
        Linse) und width=0.32 MHz zu legen (win_input_fixed in Metern -
        SI - width_fixed immer in Hz):

            plotter.plot_dependence_cuts(win_input_fixed=1.1e-3, width_fixed=0.32e6)

        win_input_fixed_axis: "before_lens" (Default) oder "after_lens" -
        legt fest, WELCHE Größe win_input_fixed angibt (win_input_fixed
        bleibt dabei immer in METERN, nur die win_axis-Größe ändert sich):
        Eingangs-Waist vor der ersten Linse (wie oben), oder effektiver
        Waist an der Fokus-/Trap-Ebene NACH allen Linsen (dieselbe Größe,
        die win_axis="after_lens" anzeigt). Praktisch, wenn man lieber in
        µm an der Fokusebene denkt statt in mm vor der Linse - Beispiel für
        einen Schnitt durch 4.5 µm effektiven Fokus-Waist:

            plotter.plot_dependence_cuts(win_input_fixed=4.5e-6, win_input_fixed_axis="after_lens")

        Das ist die unmittelbarste Antwort auf "wie hängen die Amplituden
        von Waist bzw. Width ab" - als Kurve statt als Farbwert in der
        Heatmap. Liegt der automatische Default-Schnitt am Rand des
        gescannten Bereichs (z.B. weil dort r_x/r_y an r_bounds sättigen),
        lohnt es sich, win_input_fixed/width_fixed explizit auf einen Punkt
        weiter im Inneren des Gitters zu setzen, um eine repräsentativere
        Kurve zu sehen.

        win_axis: "before_lens" (Default, mm vor der ersten Linse) oder
        "after_lens" (µm, effektiver Waist an der Fokus-/Trap-Ebene) - gilt
        NUR für die ANZEIGE der x-Achse des LINKEN Panels (win_input), UNABHÄNGIG
        von win_input_fixed_axis (das nur festlegt, wie win_input_fixed zu
        interpretieren ist). Das rechte Panel (width, in MHz) bleibt
        unverändert.

        legend_fontsize: Schriftgröße der Legenden in beiden Panels. Default
        (None): SCAN2D_RC["legend.fontsize"] (12). Größerer Wert -> größere
        Legende, z.B. legend_fontsize=16 für Präsentationen/Poster.

        mark_best_point: markiert in beiden Panels zusätzlich den Punkt, an
        dem sich die beiden Schnitte treffen (Stern-Symbol + gestrichelte
        Linie) - im Default-Fall (win_input_fixed/width_fixed beide None)
        das globale Optimum ("best point"), bei gesetztem win_input_fixed/
        width_fixed der selbst gewählte Schnittpunkt ("selected point").
        """
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        RX = r['r_x_grid']
        RY = r['r_y_grid']
        alpha = r.get('alpha', 1.0)

        mark = self._resolve_mark_point(win_input_fixed, width_fixed, win_input_fixed_axis)
        if mark is None:
            raise RuntimeError(
                "Keine gültigen Scan-Punkte vorhanden, und kein eigener Schnittpunkt "
                "(win_input_fixed/width_fixed) angegeben."
            )
        i_fixed, j_fixed, mark_label = mark

        x_win, x_win_label, _ = self._win_axis_values(win_input_vals, win_axis)

        legend_kwargs = {} if legend_fontsize is None else {"fontsize": legend_fontsize}

        with plt.rc_context(self.SCAN2D_RC):
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))

            ax_left.plot(x_win, RX[i_fixed, :], 'o-', label=r"$r_x$ ($N_x$=%d)" % r['N_x'])
            ax_left.plot(x_win, RY[i_fixed, :], 's-', label=r"$r_y$ ($N_y$=%d)" % r['N_y'])
            ax_left.set_xlabel(x_win_label)
            ax_left.set_ylabel("outer/inner ratio")
            ax_left.set_title(f"width fixed = {width_vals[i_fixed]*1e-6:.4f} MHz")
            ax_left.grid(True, alpha=0.3)

            ax_right.plot(width_vals * 1e-6, RX[:, j_fixed], 'o-', label=r"$r_x$ ($N_x$=%d)" % r['N_x'])
            ax_right.plot(width_vals * 1e-6, RY[:, j_fixed], 's-', label=r"$r_y$ ($N_y$=%d)" % r['N_y'])
            ax_right.set_xlabel("width (MHz)")
            ax_right.set_ylabel("outer/inner ratio")
            ax_right.set_title(f"win_input fixed = {win_input_vals[j_fixed]*1e3:.4f} mm")
            ax_right.grid(True, alpha=0.3)

            if mark_best_point:
                x_mark_win = self._win_axis_single_value(win_input_vals[j_fixed], win_axis)
                ax_left.axvline(x_mark_win, color="red", linestyle="--", linewidth=1.3, alpha=0.7)
                ax_left.plot(x_mark_win, RX[i_fixed, j_fixed], linestyle="none",
                             label=mark_label, **self.BEST_POINT_STYLE)
                ax_left.plot(x_mark_win, RY[i_fixed, j_fixed], linestyle="none", **self.BEST_POINT_STYLE)

                width_mark = width_vals[i_fixed] * 1e-6
                ax_right.axvline(width_mark, color="red", linestyle="--", linewidth=1.3, alpha=0.7)
                ax_right.plot(width_mark, RX[i_fixed, j_fixed], linestyle="none",
                              label=mark_label, **self.BEST_POINT_STYLE)
                ax_right.plot(width_mark, RY[i_fixed, j_fixed], linestyle="none", **self.BEST_POINT_STYLE)

            ax_left.legend(**legend_kwargs)
            ax_right.legend(**legend_kwargs)

            fig.suptitle(f"Amplitude-ratio dependence on waist and width (α={alpha:.2f})",
                         fontsize=15, fontweight='bold')
            fig.tight_layout()

            tag = self._filetag()
            self._finish_figure(fig, f"FlatMultiTone_AmpScan_DependenceCuts_{tag}.png", show, save)


# ======================================================================
# NEU (Aufraeumen 2026-09-02): Plotter fuer den FEST-AMPLITUDEN-Scan
# (scan_win_width_uniformity() / get_scan_results() aus
# multitone_flattop_optimizer.py).
#
# Bis dahin lag diese Klasse als `ScanPlotter` in einer Datei
# `multitone_flattop_scan_plots.py`, die in Hard_Optimization gar nicht
# vorhanden war (nur in Simulation_old/) - der Fest-Amplituden-Dialog
# stuerzte deshalb schon beim Import ab. Sie ist jetzt hier, als
# wortwoertliches Gegenstueck zu WeightedFixedScanPlotter aus
# weighted_multitone_amplitude_dependence_plots.py, nur mit
# uniformity_grid/crosstalk_grid statt uniformity_weighted_grid/
# eta_weighted_grid.
# ======================================================================
class FixedScanPlotter:
    """
    Heatmap-Plots fuer den Fest-Amplitude-Scan der HARTEN Uniformity/
    Crosstalk (uniformity_grid/crosstalk_grid ueber win_input x width, bei
    FESTEN Amplituden - keine r_x/r_y-Optimierung pro Punkt) aus einem von
    get_scan_results()/load_amp_scan_results() gelieferten dict.
    """

    SCAN2D_RC = AmplitudeScanPlotter.SCAN2D_RC
    SCAN2D_SAVE_DPI = AmplitudeScanPlotter.SCAN2D_SAVE_DPI
    BEST_POINT_STYLE = AmplitudeScanPlotter.BEST_POINT_STYLE

    def __init__(self, results, out_dir=None, confirm_overwrite=None):
        missing = [k for k in ("uniformity_grid", "crosstalk_grid",
                               "win_input_vals", "width_vals") if k not in results]
        if missing:
            raise ValueError(
                "results fehlen die Schluessel " + ", ".join(missing) + " - das sieht "
                "nicht nach den Rohdaten von scan_win_width_uniformity() "
                "(multitone_flattop_optimizer.py) aus."
            )
        if "r_x_grid" in results:
            raise ValueError(
                "Dieser Datensatz enthaelt r_x_grid/r_y_grid, stammt also aus dem "
                "AMPLITUDEN-Scan - dafuer ist AmplitudeScanPlotter zustaendig, nicht "
                "FixedScanPlotter."
            )
        self.results = results
        self.out_dir = Path(out_dir) if out_dir is not None else DEFAULT_IMAGES_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_overwrite = confirm_overwrite

    def _win_input_to_win(self, win_input):
        r = self.results
        return win_input_to_win(win_input, r['f1'], r['f2'], r['lambda_opt'], r['fLO'])

    def _profile_tag(self):
        profile = self.results.get('profile')
        if profile == 'airy':
            return 'Airy'
        if profile == 'gaussian':
            return 'Gauss'
        return 'ProfilUnbekannt'

    def _filetag(self):
        r = self.results
        n_win = len(r.get('win_input_vals', []))
        n_width = len(r.get('width_vals', []))
        return f"N{r['N_x']}x{r['N_y']}_{n_win}x{n_width}pts_{self._profile_tag()}_Hard"

    def _finish_figure(self, fig, filename, show, save, dpi=150):
        if save:
            out_file = resolve_save_path(self.out_dir, filename,
                                         confirm_overwrite=self.confirm_overwrite)
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {out_file}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _win_axis_values(self, win_input_vals, win_axis):
        if win_axis == "before_lens":
            return (win_input_vals * 1e3,
                    r"Input waist $\omega_{\mathrm{in}}$ (before lenses, mm)", False)
        elif win_axis == "after_lens":
            x = np.array([self._win_input_to_win(w) for w in win_input_vals]) * 1e6
            label = r"Waist at focus $\omega'$ (after lenses, µm)"
            if len(x) > 1 and x[0] > x[-1]:
                return x[::-1], label, True
            return x, label, False
        else:
            raise ValueError(
                f"win_axis muss 'before_lens' oder 'after_lens' sein, nicht {win_axis!r}.")

    def _mark_point(self):
        r = self.results
        best = r.get('best', {})
        if not best or best.get('win_input') is None:
            return None
        j = int(np.argmin(np.abs(np.asarray(r['win_input_vals']) - best['win_input'])))
        i = int(np.argmin(np.abs(np.asarray(r['width_vals']) - best['width'])))
        return i, j

    def _plot_one(self, grid_key, colorbar_label, title, cmap, filename_suffix,
                  win_axis, show, save):
        r = self.results
        win_input_vals = np.asarray(r['win_input_vals'], dtype=float)
        width_vals = np.asarray(r['width_vals'], dtype=float)
        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        grid = np.asarray(r[grid_key], dtype=float)
        Z_plot = (grid[:, ::-1] if reversed_ else grid) * 100.0

        with plt.rc_context(self.SCAN2D_RC):
            fig, ax = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
            im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
            fig.colorbar(im, ax=ax, label=colorbar_label)

            mark = self._mark_point()
            if mark is not None:
                i_mark, j_mark = mark
                x_mark = self._win_axis_values(np.array([win_input_vals[j_mark]]), win_axis)[0][0]
                y_mark = width_vals[i_mark] * 1e-6
                ax.plot(x_mark, y_mark, linestyle="none",
                        label="best point (global optimum)", **self.BEST_POINT_STYLE)
                ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

            ax.set_xlabel(x_label)
            ax.set_ylabel("width (MHz)")
            ax.set_title(title)

        tag = self._filetag()
        self._finish_figure(fig, f"FlatMultiTone_Scan_{filename_suffix}_{tag}.png",
                            show, save, dpi=self.SCAN2D_SAVE_DPI)
        return fig

    def plot_scan2d_uniformity(self, show=True, save=True, cmap="viridis_r",
                               win_axis="before_lens"):
        """Heatmap von uniformity (%) ueber (win_input, width) bei festen Amplituden."""
        return self._plot_one(
            "uniformity_grid", r"Uniformity ($\sigma/\mu$) (%)",
            "Uniformity (hard mask, fixed-amplitude scan)", cmap, "Uniformity",
            win_axis, show, save,
        )

    def plot_scan2d_crosstalk(self, show=True, save=True, cmap="Oranges",
                              win_axis="before_lens"):
        """Heatmap von crosstalk (%) ueber (win_input, width) bei festen Amplituden."""
        return self._plot_one(
            "crosstalk_grid", r"Crosstalk ($\eta$) (%)",
            "Crosstalk (hard mask, fixed-amplitude scan)", cmap, "Crosstalk",
            win_axis, show, save,
        )

    def plot_scan2d_combined(self, show=True, save=True,
                             cmap_uniformity="viridis_r", cmap_crosstalk="Oranges",
                             win_axis="before_lens"):
        """Beide Heatmaps (Uniformity, Crosstalk) nebeneinander in einer Figure."""
        r = self.results
        win_input_vals = np.asarray(r['win_input_vals'], dtype=float)
        width_vals = np.asarray(r['width_vals'], dtype=float)
        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        mark = self._mark_point()

        panels = [
            ("uniformity_grid", r"Uniformity ($\sigma/\mu$) (%)",
             "Uniformity (hard mask)", cmap_uniformity),
            ("crosstalk_grid", r"Crosstalk ($\eta$) (%)",
             "Crosstalk (hard mask)", cmap_crosstalk),
        ]

        with plt.rc_context(self.SCAN2D_RC):
            fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
            for ax, (grid_key, cbar_label, title, cmap) in zip(axes, panels):
                grid = np.asarray(r[grid_key], dtype=float)
                Z_plot = (grid[:, ::-1] if reversed_ else grid) * 100.0
                im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
                fig.colorbar(im, ax=ax, label=cbar_label)
                ax.set_xlabel(x_label)
                ax.set_ylabel("width (MHz)")
                ax.set_title(title)
                if mark is not None:
                    i_mark, j_mark = mark
                    x_mark = self._win_axis_values(np.array([win_input_vals[j_mark]]), win_axis)[0][0]
                    y_mark = width_vals[i_mark] * 1e-6
                    ax.plot(x_mark, y_mark, linestyle="none",
                            label="best point (global optimum)", **self.BEST_POINT_STYLE)
                    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

            amps = r.get('amps')
            suptitle = "Waist-width scan: hard-mask metrics (fixed-amplitude)"
            if amps is not None and len(np.asarray(amps).ravel()):
                a = np.asarray(amps, dtype=float).ravel()
                suptitle += f" (amps: {a.min():.3g} .. {a.max():.3g})"
            fig.suptitle(suptitle)

        tag = self._filetag()
        self._finish_figure(fig, f"FlatMultiTone_Scan_HardCombined_{tag}.png",
                            show, save, dpi=self.SCAN2D_SAVE_DPI)
        return fig
