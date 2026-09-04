"""
lib/combine.py - die Penalty-Kombination aus hart + atom-gewichtet.

Hier steht der rechnerische Kern, den BEIDE Datensatz-Arten teilen:

1. penalty_objective() - die ROHE Penalty-Zielfunktion, die waehrend der
   Optimierung an jedem einzelnen Gitterpunkt minimiert wird (siehe
   penalty_scan.py). "Roh" = ohne gitterweite Normierung, weil die
   waehrend der Optimierung eines EINZELNEN Punktes noch gar nicht zur
   Verfuegung steht (sie braucht alle Gitterpunkte).

       U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
       C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
       J       = alpha*U_kombi + (1-alpha)*C_kombi

   Der Term combo_lambda*|Differenz| ist der Penalty-Term: er bestraft
   Amplituden, bei denen hartes und atom-gewichtetes Kriterium
   auseinanderlaufen. Ohne ihn (combo_lambda=0) waere es der reine
   Mittelwert beider Kriterien.

2. combine_grids() - dieselbe Formel als NACHBEARBEITUNG ueber das
   fertige Gitter, diesmal auf gitterweit Min-Max-normierten Groessen
   (damit Uniformity und Crosstalk vergleichbar skaliert in den Score
   eingehen), plus Bestpunkt und Region (groesstes achsenparalleles
   Rechteck innerhalb der besten combo_percentile% aller Punkte).

Punkt 2 ist reine Auswertung/Visualisierung ueber bereits fertige Grids
und hat keinen Einfluss mehr auf die Wahl von r_x/r_y aus Punkt 1.

Verifiziert: combine_grids() reproduziert die Grids, den Schwellwert,
die Maske, das Region-Rechteck und den Bestpunkt des vorhandenen
Datensatzes scan_amp_data_combined_N3x4_21x21pts_Airy.pkl bit-exakt.
"""

import pickle
from pathlib import Path as FilePath

import numpy as np

from . import paths

from weighted_multitone_amplitude_dependence_plots import (  # noqa: E402
    width_to_um, win_input_to_win,
)


# ======================================================================
# 1) Die rohe Penalty-Zielfunktion (waehrend der Optimierung)
# ======================================================================
def penalty_pair(x_hard, x_weighted, combo_lambda):
    """Mittelwert + Penalty-Term fuer EINE Groesse (Uniformity oder
    Crosstalk). Arbeitet elementweise, also auch auf ganzen Grids."""
    return 0.5 * (x_hard + x_weighted) + combo_lambda * np.abs(x_hard - x_weighted)


def penalty_objective(U_hard, C_hard, U_weighted, C_weighted, alpha, combo_lambda):
    """Die Zielgroesse J, die pro Gitterpunkt ueber (r_x, r_y) minimiert
    wird. Auf ROHEN (unnormierten) Metriken - siehe Modul-Docstring."""
    U_kombi = penalty_pair(U_hard, U_weighted, combo_lambda)
    C_kombi = penalty_pair(C_hard, C_weighted, combo_lambda)
    return alpha * U_kombi + (1.0 - alpha) * C_kombi


# ======================================================================
# 2) Kombination/Region als Nachbearbeitung ueber das fertige Gitter
# ======================================================================
# HINWEIS ZUR AENDERUNG (2026-09-01): hier stand frueher normalize01(), eine
# gitterweite Min-Max-Normierung, und combine_grids() bildete daraus einen
# NORMIERTEN combined_score. Der ist ersatzlos entfallen. Gruende, alle
# gemessen und nicht vermutet:
#
#   - Der Optimierer hat diese Groesse nie gesehen; er minimiert an jedem
#     Gitterpunkt das ROHE J (siehe penalty_objective oben).
#   - Die Normierung haengt am gescannten Fenster: dieselbe Physik ergibt
#     bei anderem Scan-Bereich andere Zahlen.
#   - Sie hebt die atom-gewichteten Groessen um das Fuenf- bis Zehnfache an
#     (ihre rohen Spannen sind 4-7 pp gegen 41 pp bei U_hart) und verschiebt
#     damit Bestpunkt, Region und Talpfad.
#
# `combined_score` heisst weiterhin so (Dateiformat), IST aber jetzt das
# rohe J. Datensaetze aus der Zeit davor tragen dort noch den normierten
# Wert; deshalb setzt combine_grids() das Kennzeichen score_is_raw=True,
# und run_plots.py rechnet beim Laden grundsaetzlich neu.


def largest_rectangle(mask):
    """Groesstes achsenparalleles Rechteck, das vollstaendig innerhalb der
    True-Punkte von `mask` liegt.

    Gibt (flaeche, (row0, row1, col0, col1)) zurueck, Grenzen INKLUSIVE.
    Bei leerer Maske: (0, None).

    Standard-Verfahren: zeilenweise Histogramm der Saeulenhoehen, darin
    per Stack das groesste Rechteck.
    """
    mask = np.asarray(mask, dtype=bool)
    n_rows, n_cols = mask.shape
    best_area = 0
    best_bounds = None
    heights = np.zeros(n_cols, dtype=int)

    for i in range(n_rows):
        heights = np.where(mask[i], heights + 1, 0)
        stack = []  # (start_index, height)
        for j in range(n_cols + 1):
            current = int(heights[j]) if j < n_cols else 0
            start = j
            while stack and stack[-1][1] >= current:
                idx, ht = stack.pop()
                area = ht * (j - idx)
                if area > best_area:
                    best_area = area
                    best_bounds = (i - ht + 1, i, idx, j - 1)
                start = idx
            stack.append((start, current))

    return best_area, best_bounds


def combine_grids(U_hard, C_hard, U_weighted, C_weighted,
                  win_input_vals, width_vals,
                  alpha=0.7, combo_lambda=0.75, combo_percentile=25.0):
    """Penalty-Kombination auf den ROHEN Gittern -> Score -> Bestpunkt +
    Region. Gibt das Teil-dict zurueck, das in jedes Ergebnis-dict
    einfliesst (siehe build_penalty_results()/hard_check.py).

    Identisch zu penalty_objective(), nur ueber ganze Gitter statt ueber
    einen Punkt - der Score ist also genau die Groesse, die der Scan an
    jedem Gitterpunkt minimiert hat."""
    U_hard = np.asarray(U_hard, dtype=float)
    C_hard = np.asarray(C_hard, dtype=float)
    U_weighted = np.asarray(U_weighted, dtype=float)
    C_weighted = np.asarray(C_weighted, dtype=float)
    win_input_vals = np.asarray(win_input_vals, dtype=float)
    width_vals = np.asarray(width_vals, dtype=float)

    U_kombi = penalty_pair(U_hard, U_weighted, combo_lambda)
    C_kombi = penalty_pair(C_hard, C_weighted, combo_lambda)
    combined_score = alpha * U_kombi + (1.0 - alpha) * C_kombi

    finite = np.isfinite(combined_score)

    best = dict(win_input=None, width=None,
                uniformity_hart=None, crosstalk_hart=None,
                uniformity_weighted=None, crosstalk_weighted=None,
                uniformity_kombi=None, crosstalk_kombi=None, combined_score=None,
                row=None, col=None, at_edge=False)
    if np.any(finite):
        i, j = np.unravel_index(
            int(np.argmin(np.where(finite, combined_score, np.inf))), combined_score.shape)
        best = dict(
            win_input=float(win_input_vals[j]), width=float(width_vals[i]),
            uniformity_hart=float(U_hard[i, j]), crosstalk_hart=float(C_hard[i, j]),
            uniformity_weighted=float(U_weighted[i, j]), crosstalk_weighted=float(C_weighted[i, j]),
            uniformity_kombi=float(U_kombi[i, j]), crosstalk_kombi=float(C_kombi[i, j]),
            combined_score=float(combined_score[i, j]),
            row=int(i), col=int(j),
            # Liegt der beste Punkt auf dem Rand des gescannten Fensters,
            # ist er vermutlich gar kein Optimum, sondern nur der Rand -
            # beim rohen J ist das der Regelfall (gemessen: 41x41 und
            # 21x21 landen beide auf width = 0.200 MHz, dem unteren
            # Fensterrand). Der Plot zeichnet ihn dann als OFFENEN Stern.
            at_edge=bool(i in (0, combined_score.shape[0] - 1)
                         or j in (0, combined_score.shape[1] - 1)),
        )

    region = dict(win_input_min=None, win_input_max=None, width_min=None, width_max=None,
                  mask=np.zeros_like(combined_score, dtype=bool), threshold=None,
                  n_points_total=int(combined_score.size), n_points_region=0,
                  row_bounds=None, col_bounds=None)
    if np.any(finite):
        threshold = float(np.nanpercentile(combined_score[finite], combo_percentile))
        mask = finite & (combined_score <= threshold)
        area, bounds = largest_rectangle(mask)
        region = dict(
            mask=mask, threshold=threshold,
            n_points_total=int(combined_score.size), n_points_region=int(mask.sum()),
            win_input_min=None, win_input_max=None, width_min=None, width_max=None,
            row_bounds=None, col_bounds=None,
        )
        if bounds is not None and area > 0:
            r0, r1, c0, c1 = bounds
            region.update(
                win_input_min=float(win_input_vals[c0]), win_input_max=float(win_input_vals[c1]),
                width_min=float(width_vals[r0]), width_max=float(width_vals[r1]),
                row_bounds=(int(r0), int(r1)), col_bounds=(int(c0), int(c1)),
            )

    return dict(
        uniformity_kombi=U_kombi, crosstalk_kombi=C_kombi,
        combined_score=combined_score, score_is_raw=True,
        best=best, region=region,
        alpha=alpha, combo_lambda=combo_lambda, combo_percentile=combo_percentile,
    )


def score_is_raw(results):
    """Ist der gespeicherte combined_score schon der ROHE J-Wert?

    Datensaetze, die vor dem 2026-09-01 erzeugt wurden, tragen dort den
    frueheren normierten Score. run_plots.py rechnet deshalb beim Laden
    grundsaetzlich neu; diese Funktion sagt nur, ob das noetig war.
    """
    return bool(results.get('score_is_raw', False))


# ======================================================================
# Ergebnis-dict des Penalty-Scans
# ======================================================================
def build_penalty_results(scan, alpha=0.7, combo_lambda=0.75, combo_percentile=25.0):
    """Baut aus den rohen Scan-Grids das vollstaendige Ergebnis-dict.

    `scan` enthaelt die vier Metrik-Grids, r_x_grid/r_y_grid und die
    Geometrie-/Atom-Parameter (siehe penalty_scan._current_results()).

    r_x_grid_hart/_weighted (bzw. r_y_...) werden identisch zu
    r_x_grid/r_y_grid gesetzt: bei der Penalty-Methode gibt es pro Punkt
    nur EIN gemeinsames Optimum, hart und gewichtet wurden am SELBEN
    (r_x, r_y) ausgewertet. Die beiden Zusatz-Schluessel existieren
    ausschliesslich, damit Auswertungs-Code, der sie erwartet,
    unveraendert funktioniert.
    """
    win_input_vals = np.asarray(scan['win_input_vals'], dtype=float)
    width_vals = np.asarray(scan['width_vals'], dtype=float)

    U_hard = np.asarray(scan['uniformity_grid'], dtype=float)
    C_hard = np.asarray(scan['crosstalk_grid'], dtype=float)
    U_weighted = np.asarray(scan['uniformity_weighted_grid'], dtype=float)
    C_weighted = np.asarray(scan['eta_weighted_grid'], dtype=float)

    combo = combine_grids(U_hard, C_hard, U_weighted, C_weighted,
                          win_input_vals, width_vals,
                          alpha=alpha, combo_lambda=combo_lambda,
                          combo_percentile=combo_percentile)

    r_x_grid = np.asarray(scan['r_x_grid'], dtype=float)
    r_y_grid = np.asarray(scan['r_y_grid'], dtype=float)

    out = dict(
        win_input_vals=win_input_vals, width_vals=width_vals,
        uniformity_grid=U_hard, crosstalk_grid=C_hard,
        uniformity_weighted_grid=U_weighted, eta_weighted_grid=C_weighted,
        r_x_grid=r_x_grid, r_y_grid=r_y_grid, r_grid_source="joint",
        r_x_grid_hart=r_x_grid, r_y_grid_hart=r_y_grid,
        r_x_grid_weighted=r_x_grid, r_y_grid_weighted=r_y_grid,
        r_bounds=scan.get('r_bounds'),
        joint_optimization=True,
        dataset_kind="penalty",
        N_x=scan['N_x'], N_y=scan['N_y'],
        f1=scan['f1'], f2=scan['f2'], fLO=scan['fLO'],
        lambda_opt=scan['lambda_opt'], theta_max=scan['theta_max'], f_band=scan['f_band'],
        profile=scan['profile'],
        airy_scale_factor=scan.get('airy_scale_factor'),
        sigma_atom=scan.get('sigma_atom'),
        atom_mass=scan.get('atom_mass'), atom_temperature=scan.get('atom_temperature'),
        trap_freq_r=scan.get('trap_freq_r'),
    )
    out.update(combo)
    return out


def recombine_from_grids(results, alpha=None, combo_lambda=None, combo_percentile=None):
    """Score/Region eines bereits vorhandenen Datensatzes mit ANDEREN
    Kombinationsparametern neu berechnen - ohne teuren Re-Scan.

    Jeder Parameter, der None bleibt, behaelt den im Datensatz
    gespeicherten Wert. Die Metrik-Grids und r_x/r_y bleiben unangetastet
    (sie sind das Ergebnis der Optimierung und aendern sich nicht).
    """
    results = dict(results)
    alpha = results.get('alpha', 0.7) if alpha is None else alpha
    combo_lambda = results.get('combo_lambda', 0.75) if combo_lambda is None else combo_lambda
    combo_percentile = (results.get('combo_percentile', 25.0)
                        if combo_percentile is None else combo_percentile)

    combo = combine_grids(
        results['uniformity_grid'], results['crosstalk_grid'],
        results['uniformity_weighted_grid'], results['eta_weighted_grid'],
        results['win_input_vals'], results['width_vals'],
        alpha=alpha, combo_lambda=combo_lambda, combo_percentile=combo_percentile,
    )
    results.update(combo)
    return results


# ======================================================================
# Verbotener Bereich: ueberlappende Eck-Spots
# ======================================================================
# Die beiden diagonal gegenueberliegenden ECK-Spots des N_x x N_y-Arrays
# duerfen sich nicht ueberlappen.
#
# `width` ist die GESAMTSPANNWEITE des Tonarrays in Frequenz - und zwar
# dieselbe fuer x und y: multitone_frequencies(N, offset, width) laeuft in
# BEIDEN Achsen von offset bis offset+width, nur die Zwischenschritte
# unterscheiden sich (N_x bzw. N_y Toene). Nachgesehen in
# weighted_multitone_flattop_optimizer._compute_centers_for_width(), nicht
# angenommen. Raeumlich entspricht das der Kantenlaenge
#
#     S(width) = width_to_um(width, ...)                          [µm]
#
# Die beiden Ecken liegen damit bei (0, 0) und (S, S), ihr Abstand ist
#
#     d = sqrt(S^2 + S^2) = sqrt(2) * S            (Pythagoras)
#
# Nicht-Ueberlappung heisst d > k * waist, mit k = 2 (Default): die beiden
# Spot-Radien beruehren sich dann gerade. S ist linear in width
# (radius_from_angle geht zwar ueber tan, aber theta liegt bei 1.2e-3 rad -
# die Abweichung von der Geraden ist 5e-7 relativ, also weit unter jeder
# Gitterweite). Die Bedingung ist deshalb in der (waist, width)-Ebene
# exakt eine Ursprungsgerade:
#
#     width/MHz > a * waist/µm     mit     a = k / (sqrt(2) * u)
#
# u = S(1 MHz) in µm/MHz, aus der Optik des Datensatzes. Fuer f1=75mm,
# f2=750mm, fLO=52.88mm, theta_max=43mrad, f_band=36MHz ist
# u = 6.3162 µm/MHz und damit a = 0.22390 MHz/µm bei k = 2.
#
# Der verbotene Bereich liegt UNTERHALB dieser Geraden: grosser Waist bei
# kleiner width - dicke Spots, die eng beieinander sitzen.
#
# Zum Faktor k: der Nutzer hat "2 mal waist" vorgegeben, also den
# gaussaequivalenten Radius w_0 (1/e^2). Bei Airy-Profil ist die
# Hauptkeule bis zur ersten Nullstelle 1.19*w_0 breit - wer DAS als
# "den Spot" ansieht, setzt k = 2*1.19 = 2.38. Deshalb ist k frei
# einstellbar und nirgends fest verdrahtet.

# ----------------------------------------------------------------------
# Parametrisierung des Airy-Profils
# ----------------------------------------------------------------------
# `first_zero_radius = airy_scale_factor * waist`. Der Faktor legt fest, was
# die Zahl "waist" physikalisch bedeutet - und damit JEDE Metrik eines Scans.
#
#   1.19      der historische Default des Optimierers. Entspricht keiner
#             gaengigen Konvention (nachgerechnet: gleicher 1/e^2-Radius
#             1.4830, bester Gauss-Fit an die Hauptkeule 1.4499, gleiche
#             FWHM 1.3956). Bei 1.19 liegt der tatsaechliche 1/e^2-Radius
#             des Airy-Profils bei 0.8025 * waist.
#
#   1.482951  so gewaehlt, dass der 1/e^2-Radius der Airy-HAUPTKEULE genau
#             auf `waist` liegt - der Waist bedeutet dann bei Airy dasselbe
#             wie bei einem Gauss-Strahl. Herleitung: (2*J1(u)/u)^2 faellt
#             bei u = 2.583838989865 auf exp(-2), die erste Nullstelle liegt
#             bei u = 3.831705970207512, also Faktor = 3.8317.../2.5838... .
#             Gegengerechnet: mit diesem Wert ist der 1/e^2-Radius
#             1.000000168 * waist.
# Seit 2026-09-01 stehen diese Definitionen EINMAL im Projekt, naemlich in
# `Weighted_Optimization/airy_scale.py` (identische Kopie in
# `Hard_Optimization/`) - von dort benutzen sie auch die vier
# Scan-Startdialoge und die Lens-GUIs. Hier werden sie nur noch
# re-exportiert, damit bestehender Code (`combine.AIRY_SCALE_CHOICES`,
# `combine.airy_e2_radius_factor`, ...) unveraendert weiterlaeuft.
# Die Zahlenwerte sind dieselben wie vorher.
from airy_scale import (                                    # noqa: E402
    AIRY_SCALE_LEGACY,
    AIRY_SCALE_GAUSS_E2,
    AIRY_SCALE_CHOICES,
    AIRY_E2_OVER_FIRST_ZERO,
    AIRY_SCALE_DIALOG_DEFAULT,
    airy_e2_radius_factor,
    scale_tag as airy_scale_tag,
    describe as airy_scale_describe,
)


FORBIDDEN_FACTOR_DEFAULT = 2.0

# Fallbacks, falls ein aelterer Datensatz theta_max/f_band nicht
# mitgespeichert hat - dieselben Werte wie die Optimierer-Defaults.
THETA_MAX_DEFAULT = 43e-3
F_BAND_DEFAULT = 36e6


def um_per_MHz(results):
    """Wieviele µm an der Trap-Ebene entspricht 1 MHz width?

    Benutzt width_to_um() des Projekts, nicht eine nachgebaute Formel.
    """
    return float(width_to_um(
        1e6, results['f1'], results['f2'], results['fLO'],
        results.get('theta_max') or THETA_MAX_DEFAULT,
        results.get('f_band') or F_BAND_DEFAULT))


def waist_um_vals(results):
    """Effektiver Waist (µm nach der Linse) je win_input-Spalte."""
    return np.array([win_input_to_win(w, results['f1'], results['f2'],
                                      results['lambda_opt'], results['fLO'])
                     for w in np.asarray(results['win_input_vals'], dtype=float)]) * 1e6


def forbidden_boundary(results, factor=FORBIDDEN_FACTOR_DEFAULT):
    """Die Grenzgerade des verbotenen Bereichs, oder None.

    None nur, wenn es gar keine zwei Eck-Spots gibt (N_x < 2 UND N_y < 2)
    - dann kann nichts ueberlappen.
    """
    n_achsen = int(results.get('N_x', 1) > 1) + int(results.get('N_y', 1) > 1)
    if n_achsen == 0:
        return None
    u = um_per_MHz(results)
    diag = np.sqrt(float(n_achsen))          # sqrt(2) beim ueblichen 2D-Array
    slope = float(factor) / (diag * u)
    return dict(factor=float(factor), um_per_MHz=u, n_axes=n_achsen,
                diag_factor=float(diag), slope=slope)


def forbidden_mask(results, factor=FORBIDDEN_FACTOR_DEFAULT):
    """Boolesche Maske (Zeilen=width, Spalten=win_input): True, wo sich die
    Eck-Spots ueberlappen wuerden. None, wenn es keine Grenze gibt."""
    grenze = forbidden_boundary(results, factor)
    if grenze is None:
        return None
    waist = waist_um_vals(results)                              # µm
    width = np.asarray(results['width_vals'], dtype=float) * 1e-6   # MHz
    W, WD = np.meshgrid(waist, width)
    # Erlaubt ist d > k*waist, also width > slope*waist. Gleichheit heisst
    # "beruehren sich gerade" und zaehlt damit als verboten.
    return WD <= grenze['slope'] * W


GRID_KEYS_FORBIDDEN = ("uniformity_grid", "crosstalk_grid",
                       "uniformity_weighted_grid", "eta_weighted_grid",
                       "r_x_grid", "r_y_grid",
                       "r_x_grid_hart", "r_y_grid_hart",
                       "r_x_grid_weighted", "r_y_grid_weighted")


def mask_forbidden_grids(results, factor=FORBIDDEN_FACTOR_DEFAULT):
    """Kopie des Datensatzes, in der alle Gitter im verbotenen Bereich auf
    NaN stehen. Gibt (results, maske) zurueck; maske ist None, wenn es
    keine Grenze gibt.

    Score, Region und Bestpunkt werden hier NICHT neu gerechnet - das
    macht der Aufrufer mit recombine_from_grids() bzw.
    hard_check.recheck_from_grids(), weil nur er weiss, welche Art
    Datensatz vorliegt. Wichtig dabei: die Normierung in combine_grids()
    ist gitterweit, das Ausschliessen aendert den combined_score also
    UEBERALL, nicht nur im verbotenen Bereich.
    """
    maske = forbidden_mask(results, factor)
    if maske is None:
        return dict(results), None
    neu = dict(results)
    for key in GRID_KEYS_FORBIDDEN:
        if key in neu and neu[key] is not None:
            grid = np.array(neu[key], dtype=float)
            if grid.shape == maske.shape:
                grid[maske] = np.nan
                neu[key] = grid
    neu['forbidden_factor'] = float(factor)
    neu['forbidden_excluded'] = True
    return neu, maske


# ======================================================================
# Speichern / Laden
# ======================================================================
def resolve_pickle_path(out_dir, filename, overwrite=False):
    """Vollstaendiger Pfad in out_dir. Bei overwrite=False und bereits
    existierender Datei wird ein freier Name mit Zaehler (_2, _3, ...)
    gewaehlt."""
    out_dir = FilePath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / filename
    if overwrite or not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    counter = 2
    while (out_dir / f"{stem}_{counter}{suffix}").exists():
        counter += 1
    return out_dir / f"{stem}_{counter}{suffix}"


def save_results(results, filepath, overwrite=False):
    """Ergebnis-dict als .pkl speichern.

    overwrite=True ueberschreibt GENAU den angegebenen Pfad. Das ist der
    Normalfall am Ende eines Scans: derselbe Pfad wurde vorher schon als
    Checkpoint-Pfad benutzt, dort liegt also bereits ein Zwischenstand,
    der jetzt bewusst durch den sauberen Endstand ersetzt werden soll -
    statt daneben eine verwirrende "_2"-Doppeldatei anzulegen.
    """
    filepath = FilePath(filepath)
    if not overwrite:
        filepath = resolve_pickle_path(filepath.parent, filepath.name, overwrite=False)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as fh:
        pickle.dump(results, fh)
    print(f"Datensatz gespeichert: {filepath}")
    return filepath


def load_results(filepath):
    """Ergebnis-dict aus .pkl laden. `filepath` darf ein voller Pfad oder
    ein blosser Dateiname sein (dann wird in Results/ und neben
    Combinated_Optimization/ gesucht)."""
    candidate = FilePath(filepath)
    if not candidate.exists():
        for base in (paths.DEFAULT_RESULTS_DIR, paths.BASE_DIR):
            alt = base / candidate.name
            if alt.exists():
                candidate = alt
                break
    if not candidate.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {filepath}")
    with open(candidate, 'rb') as fh:
        results = pickle.load(fh)
    results['_source_path'] = str(candidate)
    return results


def describe(results):
    """Kurze, menschenlesbare Einordnung eines geladenen Datensatzes -
    genutzt von den GUIs, um sofort anzuzeigen, was da geladen wurde."""
    kind = dataset_kind(results)
    n_win = len(np.asarray(results.get('win_input_vals', [])))
    n_width = len(np.asarray(results.get('width_vals', [])))
    lines = [
        f"Art: {KIND_LABELS.get(kind, kind)}",
        f"Gitter: {n_win} x {n_width} Punkte, N_x={results.get('N_x')}, "
        f"N_y={results.get('N_y')}, Profil={results.get('profile')}",
    ]
    if results.get('checkpoint') or results.get('_checkpoint'):
        lines.append("ACHTUNG: das ist ein unvollstaendiger Zwischenstand (Checkpoint).")
    # Mit oder ohne die statische Interferenz frequenzentarteter Spots
    # gerechnet? Ohne den Schluessel: der Scan lief vor dieser Option, also
    # inkohaerent. Das gehoert sichtbar an die Oberflaeche - die Zahlen
    # beider Modelle stehen sonst unbemerkt nebeneinander.
    koh = results.get('coherent')
    if koh is None:
        lines.append(
            "ACHTUNG: ohne Kohaerenz gerechnet (Datensatz von vor dieser Option) - "
            "die statische Interferenz frequenzentarteter Spots fehlt. Nicht direkt "
            "mit kohaerent gerechneten Datensaetzen vergleichbar.")
    elif koh:
        lines.append("Kohaerenz: statische Interferenz mitgerechnet (Phasendifferenz 0)")
    else:
        lines.append("Kohaerenz: AUS - reine Intensitaetssumme")
    params = [f"{name}={results[name]}" for name in ('alpha', 'combo_lambda', 'combo_percentile')
              if results.get(name) is not None]
    if params:
        lines.append(", ".join(params))
    best = results.get('best') or {}
    if best.get('win_input') is not None:
        lines.append(f"Bester Punkt: win_input={best['win_input'] * 1e3:.4f} mm, "
                     f"width={best['width'] * 1e-6:.4f} MHz")
    return "\n".join(lines)


KIND_LABELS = {
    "penalty": "Penalty-Scan (gemeinsame Amplituden-Optimierung, hart + gewichtet)",
    "hard_check": "Hard-Check (harte Metriken bei den Amplituden eines gewichteten Scans)",
    "weighted_amp": "Gewichteter Amplituden-Scan (Eingangsdatei fuer den Hard-Check)",
    "unknown": "unbekannt",
}


def dataset_kind(results):
    """Erkennt, um welche Art Datensatz es sich handelt."""
    if results.get('dataset_kind') in KIND_LABELS:
        return results['dataset_kind']
    has_hard = 'uniformity_grid' in results and 'crosstalk_grid' in results
    has_weighted = 'uniformity_weighted_grid' in results and 'eta_weighted_grid' in results
    has_r = 'r_x_grid' in results and 'r_y_grid' in results
    if has_hard and has_weighted and has_r:
        # Sowohl die alten scan_amp_data_combined_*.pkl (joint_optimization=True)
        # als auch neue Penalty-Scans landen hier.
        return "penalty"
    if has_weighted and has_r and not has_hard:
        return "weighted_amp"
    return "unknown"


def looks_like_weighted_amp_scan(results):
    """Pruefung fuer den Hard-Check: enthaelt die Datei einen
    amplituden-optimierten GEWICHTETEN Scan (r_x/r_y je Gitterpunkt +
    gewichtete Metriken)?"""
    needed = ('win_input_vals', 'width_vals', 'r_x_grid', 'r_y_grid',
              'uniformity_weighted_grid', 'eta_weighted_grid')
    missing = [k for k in needed if k not in results]
    return (len(missing) == 0), missing
