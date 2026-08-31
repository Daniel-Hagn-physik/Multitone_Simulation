"""
lib/penalty_opt.py - Penalty-Optimierung OHNE Gitter.

run_penalty_scan.py legt ein (win_input, width)-Gitter an und optimiert an
JEDEM Gitterpunkt (r_x, r_y). Hier ist es umgekehrt: es gibt kein Gitter.
Der Nutzer gibt vor, WELCHE Groessen er festsetzt und in welchem Bereich
die uebrigen liegen duerfen - und die uebrigen werden GEMEINSAM so
gewaehlt, dass dieselbe Penalty-Zielfunktion minimal wird:

    U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
    C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
    J       = alpha*U_kombi + (1-alpha)*C_kombi        ->  min

Typische Frage: "Waist und Brennweiten habe ich - wie muessen width, r_x
und r_y sein?"

Identisch zum Scan ist dabei alles, was den Zahlenwert bestimmt: dieselbe
Zielfunktion aus lib/combine.py, auf ROHEN (unnormierten) Metriken, und
pro Auswertung EIN _evaluate(..., weighted=True), das die harten und die
atom-gewichteten Metriken am SELBEN Parametersatz liefert. Ein Ergebnis
dieses Moduls und ein Gitterpunkt des Scans sind deshalb direkt
vergleichbar (gegengerechnet, siehe Projektdoku).

Anders als der Scan schreibt dieses Modul KEINE .pkl und keine Plots -
nur einen Markdown-Bericht. Es gibt hier nichts zu plotten: das Ergebnis
ist ein einzelner Parametersatz, kein Feld.

Die sieben waehlbaren Groessen (siehe PARAM_SPECS):

    waist_um    effektiver Waist NACH der Linse, µm
    width_MHz   Frequenzabstand der Toene, MHz
    r_x, r_y    Amplituden-Verhaeltnisse (dimensionslos)
    f1_mm       Brennweite der ersten Teleskoplinse, mm
    f2_mm       Brennweite der zweiten Teleskoplinse, mm
    fLO_mm      Brennweite der Fokussierlinse, mm

Zu den Brennweiten: die drei sind KEINE reine Anzeigegroesse. Sie gehen
an zwei Stellen ein - ueber radius_from_angle() in die Spot-Positionen in
der Fallenebene (also in die Geometrie, die width in eine Laenge
uebersetzt) und ueber win_input_to_win() in die Umrechnung zwischen dem
Eingangswaist vor der Linse und dem effektiven Waist. Weil der Nutzer
hier den effektiven Waist DIREKT vorgibt, ist nur der erste Weg wirksam;
den zweiten benutzt der Bericht, um zu sagen, welchen Eingangswaist (mm
vor der Linse) man fuer den gefundenen effektiven Waist einstellen muss.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from . import paths
from .combine import penalty_pair, penalty_objective

from weighted_multitone_flattop_optimizer import (  # noqa: E402
    MultitoneFlatTopOptimizer, amps_from_ratios,
)


# Dieselben Werte wie in lib/penalty_scan.py, damit sich beide Verfahren
# bei gleicher Aufgabe auch gleich verhalten.
NELDER_MEAD_OPTIONS = {'xatol': 1e-6, 'fatol': 1e-9, 'maxiter': 300}
INVALID_PENALTY = 1e10


# ======================================================================
# Die waehlbaren Groessen
# ======================================================================
# key, Klartext, Einheit, Nachkommastellen im Dialog, erlaubte Spanne,
# Vorgabe als fester Wert, Vorgabe als Bereich.
#
# Die Reihenfolge ist die Reihenfolge im Dialog und im Bericht.
PARAM_SPECS = [
    ("waist_um", "Waist nach der Linse", "µm", 4, (0.05, 50.0), 0.9213, (0.75, 1.70)),
    ("width_MHz", "Width", "MHz", 4, (0.001, 50.0), 0.2700, (0.20, 0.40)),
    ("r_x", "Amplituden-Verhaeltnis r_x", "", 4, (0.05, 20.0), 1.0000, (0.80, 2.20)),
    ("r_y", "Amplituden-Verhaeltnis r_y", "", 4, (0.05, 20.0), 1.1791, (0.80, 2.20)),
    ("f1_mm", "Brennweite f1", "mm", 3, (1.0, 5000.0), 75.0, (50.0, 150.0)),
    ("f2_mm", "Brennweite f2", "mm", 3, (1.0, 5000.0), 750.0, (500.0, 1000.0)),
    ("fLO_mm", "Brennweite fLO", "mm", 4, (1.0, 5000.0), 52.88, (40.0, 70.0)),
]
PARAM_KEYS = [spec[0] for spec in PARAM_SPECS]
PARAM_LABEL = {spec[0]: spec[1] for spec in PARAM_SPECS}
PARAM_UNIT = {spec[0]: spec[2] for spec in PARAM_SPECS}

# Faktor key -> SI (nur fuer die Umrechnung in den Optimierer)
_TO_SI = {"waist_um": 1e-6, "width_MHz": 1e6, "r_x": 1.0, "r_y": 1.0,
          "f1_mm": 1e-3, "f2_mm": 1e-3, "fLO_mm": 1e-3}


def param_label(key, mit_einheit=True):
    """'Width (MHz)' bzw. 'Width'."""
    einheit = PARAM_UNIT[key]
    return f"{PARAM_LABEL[key]} ({einheit})" if (mit_einheit and einheit) else PARAM_LABEL[key]


# ======================================================================
# Duenne Huelle um den Optimierer (wie in lib/penalty_scan.py)
# ======================================================================
def make_optimizer(**params):
    """MultitoneFlatTopOptimizer mit Ausgabe in Combinated_Optimization/Bilder."""
    return MultitoneFlatTopOptimizer(out_dir=str(paths.DEFAULT_IMAGES_DIR), **params)


def win_input_for_waist(opt, waist_m):
    """Welcher Eingangswaist VOR der Linse gehoert zu einem effektiven
    Waist NACH der Linse?

    Die Kaskade ist win_eff = (f1/f2) * (lambda_opt * fLO) / (pi * win_input)
    (siehe MultitoneFlatTopOptimizer.win_input_to_win), also reziprok - die
    Umkehrung ist dieselbe Formel mit vertauschten Rollen. Das Ergebnis wird
    hier NICHT geglaubt, sondern durch Vorwaertsrechnen gegengeprueft; passt
    es nicht, gibt die Funktion None zurueck statt einer falschen Zahl.
    """
    if waist_m <= 0:
        return None
    try:
        win_input = (opt.f1 / opt.f2) * (opt.lambda_opt * opt.fLO) / (np.pi * waist_m)
        zurueck = opt.win_input_to_win(win_input)
    except (ValueError, ZeroDivisionError, AttributeError):
        return None
    if not np.isfinite(zurueck) or abs(zurueck - waist_m) > 1e-12 + 1e-9 * abs(waist_m):
        return None
    return float(win_input)


# ======================================================================
# Eine einzelne Auswertung
# ======================================================================
def evaluate_params(opt, werte, alpha, combo_lambda):
    """Wertet die Penalty-Zielfunktion an EINEM vollstaendigen
    Parametersatz aus.

    werte: dict mit allen Schluesseln aus PARAM_KEYS, in den
    Einheiten von PARAM_SPECS (µm, MHz, mm).

    Gibt (J, details) zurueck; details ist das dict von _evaluate() um die
    Kombinationsgroessen ergaenzt, oder None, wenn der Parametersatz
    ungueltig ist (dann ist J = INVALID_PENALTY).

    Zu den Brennweiten: sie werden hier DIREKT als Attribut gesetzt und
    nicht ueber set_parameters(). set_parameters() wuerde bei jedem Aufruf
    _setup_geometry() ausfuehren und dabei das feste n_grid x n_grid-Gitter
    self.X/self.Y neu aufbauen (bei n_grid=1000 zwei 8-MB-Arrays je
    Auswertung) - gebraucht wird es nicht, weil hier wie im Scan mit einem
    per _build_dynamic_grid() eigens gebauten Gitter gerechnet wird und
    _compute_centers_for_width() self.f1/self.f2/self.fLO ohnehin bei jedem
    Aufruf frisch liest. Dass beide Wege dasselbe liefern, ist gegengeprueft
    (siehe Projektdoku).
    """
    waist_m = werte["waist_um"] * _TO_SI["waist_um"]
    width_hz = werte["width_MHz"] * _TO_SI["width_MHz"]
    if not (waist_m > 0 and width_hz > 0):
        return INVALID_PENALTY, None

    opt.f1 = werte["f1_mm"] * _TO_SI["f1_mm"]
    opt.f2 = werte["f2_mm"] * _TO_SI["f2_mm"]
    opt.fLO = werte["fLO_mm"] * _TO_SI["fLO_mm"]

    try:
        punkt_grid = opt._build_dynamic_grid(waist_m, width_hz)
        amps = amps_from_ratios(werte["r_x"], werte["r_y"], opt.N_x, opt.N_y)
        val = opt._evaluate(waist_m, width_hz, amps=amps, grid=punkt_grid, weighted=True)
    except Exception:
        return INVALID_PENALTY, None
    if val is None:
        return INVALID_PENALTY, None

    U_h, C_h = val.get('uniformity'), val.get('eta')
    U_w, C_w = val.get('uniformity_weighted'), val.get('eta_weighted')
    if any(x is None or not np.isfinite(x) for x in (U_h, C_h, U_w, C_w)):
        return INVALID_PENALTY, None

    J = float(penalty_objective(U_h, C_h, U_w, C_w, alpha, combo_lambda))
    details = dict(
        uniformity_hard=float(U_h), crosstalk_hard=float(C_h),
        uniformity_weighted=float(U_w), crosstalk_weighted=float(C_w),
        uniformity_kombi=float(penalty_pair(U_h, U_w, combo_lambda)),
        crosstalk_kombi=float(penalty_pair(C_h, C_w, combo_lambda)),
        J=J,
    )
    return J, details


# ======================================================================
# Ein Optimierungslauf von EINEM Startpunkt
# ======================================================================
# Optimiert wird nicht in den physikalischen Einheiten, sondern in
# normierten Koordinaten u in [0, 1] pro freier Groesse. Grund: Nelder-Mead
# baut seinen Startsimplex aus festen relativen Schritten. Mit
# waist ~ 1 (µm), width ~ 0.3 (MHz) und f2 ~ 750 (mm) im selben Vektor
# waere der Simplex um drei Groessenordnungen verzerrt und der Lauf
# wuerde faktisch nur noch f2 variieren. In [0,1] sind alle Richtungen
# gleich gewichtet, und die Schranken sind automatisch eingehalten.
def _entpacken(u, freie, feste):
    """Normierter Vektor -> vollstaendiger Parametersatz in den Einheiten
    von PARAM_SPECS."""
    werte = dict(feste)
    for wert, (key, lo, hi) in zip(np.asarray(u, dtype=float), freie):
        werte[key] = float(lo + np.clip(wert, 0.0, 1.0) * (hi - lo))
    return werte


def _lauf(u0, freie, feste, alpha, combo_lambda, optimizer_kwargs, maxiter):
    """Ein Nelder-Mead-Lauf ab dem normierten Startpunkt u0."""
    opt = make_optimizer(**optimizer_kwargs)
    zaehler = {'n': 0}

    def ziel(u):
        zaehler['n'] += 1
        J, _ = evaluate_params(opt, _entpacken(u, freie, feste), alpha, combo_lambda)
        return J

    optionen = dict(NELDER_MEAD_OPTIONS)
    if maxiter is not None:
        optionen['maxiter'] = int(maxiter)
    ergebnis = minimize(ziel, x0=np.asarray(u0, dtype=float), method='Nelder-Mead',
                        bounds=[(0.0, 1.0)] * len(freie), options=optionen)
    werte = _entpacken(ergebnis.x, freie, feste)
    J, details = evaluate_params(opt, werte, alpha, combo_lambda)
    return dict(J=J, werte=werte, details=details, n_eval=zaehler['n'],
                start=_entpacken(u0, freie, feste))


def _worker(task):
    """Worker fuer n_jobs > 1 - muss auf Modulebene stehen, damit
    ProcessPoolExecutor ihn picklen kann."""
    (u0, freie, feste, alpha, combo_lambda, optimizer_kwargs, maxiter) = task
    return _lauf(u0, freie, feste, alpha, combo_lambda, optimizer_kwargs, maxiter)


def _startpunkte(n_dim, n_starts, seed=0):
    """Der Mittelpunkt plus (n_starts - 1) Latin-Hypercube-Punkte.

    Der Mittelpunkt ist immer dabei, damit n_starts=1 genau der
    naheliegende Lauf "aus der Mitte der Bereiche" ist und das Ergebnis
    reproduzierbar bleibt. Latin Hypercube statt Zufall, damit die
    Startpunkte die Bereiche gleichmaessig abdecken; fester seed, damit
    zwei Laeufe mit denselben Eingaben dasselbe liefern.
    """
    punkte = [np.full(n_dim, 0.5)]
    if n_starts > 1:
        sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
        punkte.extend(np.asarray(sampler.random(n=n_starts - 1), dtype=float))
    return punkte[:max(1, n_starts)]


# ======================================================================
# Die Optimierung
# ======================================================================
def optimize_penalty(feste, bereiche, alpha=0.7, combo_lambda=0.75,
                     n_starts=8, n_jobs=1, maxiter=None, optimizer_kwargs=None,
                     progress_callback=None, verbose=True):
    """Minimiert J ueber die freien Groessen.

    feste:     dict key -> Wert, fuer die vom Nutzer vorgegebenen Groessen.
    bereiche:  dict key -> (min, max), fuer die freien Groessen.
               Zusammen muessen beide dicts genau PARAM_KEYS abdecken.
    n_starts:  Zahl der Startpunkte (1 = nur der Mittelpunkt).
    n_jobs:    > 1 verteilt die Startpunkte auf Prozesse.
    maxiter:   Nelder-Mead-Iterationsdeckel je Lauf (None = Voreinstellung).

    Gibt ein dict mit dem besten Lauf, allen Laeufen (nach J sortiert) und
    den Eingaben zurueck.
    """
    optimizer_kwargs = dict(optimizer_kwargs or {})

    fehlend = [k for k in PARAM_KEYS if k not in feste and k not in bereiche]
    doppelt = [k for k in PARAM_KEYS if k in feste and k in bereiche]
    fremd = [k for k in list(feste) + list(bereiche) if k not in PARAM_KEYS]
    if fehlend:
        raise ValueError(f"Fuer diese Groessen fehlt die Angabe (fest ODER Bereich): {fehlend}")
    if doppelt:
        raise ValueError(f"Diese Groessen sind gleichzeitig fest UND variabel: {doppelt}")
    if fremd:
        raise ValueError(f"Unbekannte Groessen: {fremd}. Erlaubt: {PARAM_KEYS}")

    freie = []
    for key in PARAM_KEYS:                      # feste Reihenfolge, nicht dict-Reihenfolge
        if key in bereiche:
            lo, hi = (float(bereiche[key][0]), float(bereiche[key][1]))
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                raise ValueError(f"Bereich fuer {key} ist leer oder ungueltig: ({lo}, {hi})")
            freie.append((key, lo, hi))
    feste = {k: float(v) for k, v in feste.items()}

    t_start = time.perf_counter()

    # Sonderfall: nichts ist frei. Dann gibt es nichts zu optimieren - eine
    # einzige Auswertung ist die vollstaendige Antwort. Ohne diesen Zweig
    # liefe Nelder-Mead mit einem leeren Vektor ins Leere.
    if not freie:
        opt = make_optimizer(**optimizer_kwargs)
        J, details = evaluate_params(opt, feste, alpha, combo_lambda)
        laeufe = [dict(J=J, werte=dict(feste), details=details, n_eval=1, start=dict(feste))]
        if verbose:
            print("Keine freie Groesse - es wurde nur ausgewertet, nicht optimiert.")
    else:
        punkte = _startpunkte(len(freie), int(n_starts))
        aufgaben = [(u0, freie, feste, alpha, combo_lambda, optimizer_kwargs, maxiter)
                    for u0 in punkte]
        laeufe = []
        if verbose:
            print(f"{len(aufgaben)} Startpunkt(e), freie Groessen: "
                  f"{[k for k, _, _ in freie]}")
        if n_jobs and n_jobs > 1 and len(aufgaben) > 1:
            with ProcessPoolExecutor(max_workers=int(n_jobs)) as pool:
                for i, res in enumerate(pool.map(_worker, aufgaben), start=1):
                    laeufe.append(res)
                    if verbose:
                        print(f"  Lauf {i}/{len(aufgaben)}: J = {res['J']:.6g} "
                              f"({res['n_eval']} Auswertungen)")
                    if progress_callback is not None and not progress_callback(i, len(aufgaben)):
                        break
        else:
            for i, task in enumerate(aufgaben, start=1):
                res = _lauf(*task)
                laeufe.append(res)
                if verbose:
                    print(f"  Lauf {i}/{len(aufgaben)}: J = {res['J']:.6g} "
                          f"({res['n_eval']} Auswertungen)")
                if progress_callback is not None and not progress_callback(i, len(aufgaben)):
                    break

    gueltig = [r for r in laeufe if r['details'] is not None]
    laeufe_sortiert = sorted(laeufe, key=lambda r: r['J'])
    bester = min(gueltig, key=lambda r: r['J']) if gueltig else None

    dauer = time.perf_counter() - t_start
    if verbose:
        if bester is None:
            print(f"Kein gueltiger Parametersatz gefunden ({dauer:.1f}s).")
        else:
            print(f"Bestes J = {bester['J']:.6g} nach {dauer:.1f}s "
                  f"({sum(r['n_eval'] for r in laeufe)} Auswertungen insgesamt).")

    return dict(
        best=bester, runs=laeufe_sortiert,
        fixed=dict(feste), ranges={k: (lo, hi) for k, lo, hi in freie},
        free_keys=[k for k, _, _ in freie],
        alpha=float(alpha), combo_lambda=float(combo_lambda),
        n_starts=len(laeufe), n_valid=len(gueltig),
        n_eval_total=int(sum(r['n_eval'] for r in laeufe)),
        duration_s=float(dauer), optimizer_kwargs=optimizer_kwargs,
    )


# ======================================================================
# Bericht
# ======================================================================
def output_name(optimizer_kwargs):
    """PenaltyOpt_N3x4_Airy_2026-08-31.md"""
    N_x = optimizer_kwargs.get('N_x', 3)
    N_y = optimizer_kwargs.get('N_y', 4)
    tag = paths.profile_tag_of(optimizer_kwargs.get('profile', 'airy'))
    return f"PenaltyOpt_N{N_x}x{N_y}_{tag}_{date.today().isoformat()}.md"


def _zahl(key, wert):
    """Einheitliche Formatierung einer Groesse im Bericht."""
    stellen = dict((s[0], s[3]) for s in PARAM_SPECS)[key]
    einheit = PARAM_UNIT[key]
    return f"{wert:.{stellen}f}" + (f" {einheit}" if einheit else "")


def report_lines(ergebnis):
    """Der Bericht als Liste von Zeilen (ohne Zeilenumbrueche)."""
    kw = ergebnis['optimizer_kwargs']
    N_x, N_y = kw.get('N_x', 3), kw.get('N_y', 4)
    profil = kw.get('profile', 'airy')
    bester = ergebnis['best']

    lines = [
        f"# Penalty-Optimierung - N{N_x}x{N_y}, {profil}, {date.today().isoformat()}",
        "",
        "Kein Gitter, kein Scan: die unten als VORGEGEBEN gelisteten Groessen wurden "
        "festgehalten, die als FREI gelisteten wurden gemeinsam so gewaehlt, dass die "
        "Penalty-Zielfunktion minimal wird. Harte und atom-gewichtete Metriken werden "
        "dabei bei jedem Schritt am SELBEN Parametersatz ausgewertet - das gefundene "
        "Ergebnis ist daher fuer beide Kriterien zugleich gueltig.",
        "",
        "## Zielfunktion",
        "",
        "```",
        "U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|",
        "C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|",
        "J       = alpha*U_kombi + (1-alpha)*C_kombi   ->  min",
        "```",
        "",
        "Auf ROHEN (unnormierten) Metriken - dieselbe Zielfunktion, die auch "
        "run_penalty_scan.py an jedem Gitterpunkt minimiert. Eine gitterweite "
        "Normierung gibt es hier nicht und kann es nicht geben: es gibt kein Gitter, "
        "ueber das normiert werden koennte. Die Zahlen sind deshalb mit den ROHEN "
        "Spalten eines Scan-Berichts vergleichbar, nicht mit dem dortigen "
        "combined_score.",
        "",
        f"Parameter dieses Laufs: alpha = {ergebnis['alpha']:.3f}, "
        f"combo_lambda = {ergebnis['combo_lambda']:.3f}.",
        "",
        "## Vorgaben",
        "",
        "| Groesse | Rolle | Wert bzw. Bereich |",
        "|---|---|---|",
    ]
    for key in PARAM_KEYS:
        if key in ergebnis['fixed']:
            lines.append(f"| {param_label(key)} | vorgegeben | {_zahl(key, ergebnis['fixed'][key])} |")
        else:
            lo, hi = ergebnis['ranges'][key]
            lines.append(f"| {param_label(key)} | **frei** | {_zahl(key, lo)} .. {_zahl(key, hi)} |")
    lines += [
        "",
        f"- N_x = {N_x}, N_y = {N_y}, Profil = {profil}",
        f"- offset = {kw.get('offset', 100e6) * 1e-6:.4f} MHz, n_grid = {kw.get('n_grid', 1000)}",
        f"- lambda = {kw.get('lambda_opt', 795e-9) * 1e9:.2f} nm, "
        f"pitch = {kw.get('pitch', 5.288e-6) * 1e6:.4f} µm, "
        f"theta_max = {kw.get('theta_max', 43e-3) * 1e3:.3f} mrad, "
        f"f_band = {kw.get('f_band', 36e6) * 1e-6:.3f} MHz",
        "",
    ]

    if bester is None:
        lines += [
            "## Ergebnis",
            "",
            "**Kein gueltiger Parametersatz gefunden.** Alle Auswertungen sind auf "
            "einen ungueltigen Zustand gelaufen (leere Maske, kein auswertbares "
            "Atom-Sub-Grid oder unzulaessige Geometrie). Meist liegt das an einem "
            "Bereich, der physikalisch nicht erreichbar ist - die Bereiche oben "
            "pruefen und enger fassen.",
            "",
        ]
        return lines

    werte = bester['werte']
    d = bester['details']
    lines += [
        "## Ergebnis",
        "",
        "| Groesse | Wert | |",
        "|---|---|---|",
    ]
    for key in PARAM_KEYS:
        rolle = "vorgegeben" if key in ergebnis['fixed'] else "**optimiert**"
        lines.append(f"| {param_label(key)} | {_zahl(key, werte[key])} | {rolle} |")

    # Zusatzzeile: welchen Eingangswaist muss man fuer diesen effektiven
    # Waist einstellen? Haengt an f1/f2 und ist damit erst hier bekannt.
    opt = make_optimizer(**{**kw, 'f1': werte['f1_mm'] * 1e-3,
                            'f2': werte['f2_mm'] * 1e-3,
                            'fLO': werte['fLO_mm'] * 1e-3})
    win_input = win_input_for_waist(opt, werte['waist_um'] * 1e-6)
    lines.append("")
    if win_input is not None:
        lines.append(f"Dazu gehoerender Eingangswaist VOR der Linse: "
                     f"**win_input = {win_input * 1e3:.4f} mm** "
                     f"(aus f1/f2 oben; das ist die Groesse, die run_penalty_scan.py scannt).")
    else:
        lines.append("Der zugehoerige Eingangswaist vor der Linse liess sich nicht "
                     "zurueckrechnen (Brennweiten oder Waist ausserhalb des "
                     "physikalisch sinnvollen Bereichs).")

    lines += [
        "",
        "### Metriken an diesem Punkt",
        "",
        "| | hart (Pitch-Box) | atom-gewichtet | kombiniert |",
        "|---|---|---|---|",
        f"| Uniformity | {d['uniformity_hard'] * 100:.4f}% | "
        f"{d['uniformity_weighted'] * 100:.4f}% | {d['uniformity_kombi'] * 100:.4f}% |",
        f"| Crosstalk | {d['crosstalk_hard'] * 100:.4f}% | "
        f"{d['crosstalk_weighted'] * 100:.4f}% | {d['crosstalk_kombi'] * 100:.4f}% |",
        "",
        f"**J = {d['J']:.6f}** (dieselben Einheiten wie die rohen Metriken, "
        f"also Bruchteile - nicht Prozent).",
        "",
    ]

    # Die uebrigen Laeufe: sie sagen, ob man dem Optimum trauen kann.
    if len(ergebnis['runs']) > 1:
        lines += [
            "## Die einzelnen Startpunkte",
            "",
            "Mehrere Startpunkte, weil die Zielfunktion nicht glatt ist (siehe unten). "
            "Liegen die besten Laeufe dicht beieinander, ist das Optimum belastbar; "
            "streuen sie, ist es eher eine von mehreren gleichwertigen Loesungen.",
            "",
            "| Lauf | J | " + " | ".join(param_label(k, mit_einheit=False)
                                          for k in ergebnis['free_keys']) + " | Auswertungen |",
            "|---|---|" + "---|" * (len(ergebnis['free_keys']) + 1),
        ]
        for i, r in enumerate(ergebnis['runs'], start=1):
            if r['details'] is None:
                zellen = ["-"] * len(ergebnis['free_keys'])
                j_text = "ungueltig"
            else:
                zellen = [_zahl(k, r['werte'][k]) for k in ergebnis['free_keys']]
                j_text = f"{r['J']:.6f}"
            lines.append(f"| {i} | {j_text} | " + " | ".join(zellen) + f" | {r['n_eval']} |")
        lines.append("")

        gueltige_J = [r['J'] for r in ergebnis['runs'] if r['details'] is not None]
        if len(gueltige_J) > 1:
            spanne = max(gueltige_J) - min(gueltige_J)
            lines += [
                f"Streuung von J ueber die {len(gueltige_J)} gueltigen Laeufe: "
                f"{spanne:.6f} ({spanne / max(gueltige_J[0], 1e-12) * 100:.2f}% des besten Werts).",
                "",
            ]

    lines += [
        "## Wie genau ist das?",
        "",
        "Die harten Metriken rauschen. Ihr globales Intensitaetsgitter wird pro "
        "Auswertung neu aus Waist und width aufgebaut, wodurch die Abtastpunkte "
        "gegenueber den Fallen wandern und die Maskengrenzen um ganze Pixel springen "
        "(gemessen: Saegezahn von 0.05-0.09 Prozentpunkten im harten Crosstalk, "
        "waehrend der atom-gewichtete an derselben Stelle glatt ist). Die "
        "Zielfunktion hat dadurch feine lokale Minima, die nichts mit Physik zu tun "
        "haben. Deshalb mehrere Startpunkte - und deshalb sollte man die letzten "
        "Nachkommastellen des Optimums nicht ernst nehmen. Ein groesseres n_grid "
        "hilft dagegen nachweislich NICHT (gemessen ueber n_grid = 1000 .. 2400: "
        "Streuung ohne Konvergenztrend).",
        "",
        f"- Startpunkte: {ergebnis['n_starts']}, davon gueltig: {ergebnis['n_valid']}",
        f"- Auswertungen insgesamt: {ergebnis['n_eval_total']}",
        f"- Laufzeit: {ergebnis['duration_s']:.1f} s",
        "",
    ]
    return lines


def write_report(ergebnis, output_path):
    """Schreibt den Bericht und gibt den Pfad zurueck."""
    text = "\n".join(report_lines(ergebnis)) + "\n"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Bericht geschrieben: {output_path}")
    return output_path
