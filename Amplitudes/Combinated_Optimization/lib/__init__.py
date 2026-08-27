"""
Combinated_Optimization/lib
===========================

Hilfsmodule fuer die drei Hauptskripte eine Ebene hoeher. Diese Dateien
werden NICHT direkt ausgefuehrt - sie werden von den Hauptskripten
importiert:

    ../run_penalty_scan.py   ->  lib.paths, lib.combine, lib.penalty_scan, lib.report
    ../run_hard_check.py     ->  lib.paths, lib.combine, lib.hard_check,   lib.report
    ../run_plots.py          ->  lib.paths, lib.combine, lib.report

Inhalt:

- paths.py         Ordner-Konstanten (Results/Bilder/Fit_Plots/Fit_Results) und
                   der sys.path-Eintrag auf ../../Weighted_Optimization, ueber
                   den der eigentliche Optimierer erreichbar ist.
- combine.py       Die Penalty-Kombination (Normierung, Mittelwert +
                   combo_lambda*|Differenz|, combined_score), die Region-Suche
                   (groesstes Rechteck) sowie Laden/Speichern/Neuberechnen.
- penalty_scan.py  Die gemeinsame (jointe) Amplituden-Optimierung MIT
                   Penalty-Term: pro Gitterpunkt EINE Nelder-Mead-Optimierung
                   direkt gegen die Kombination aus hart + atom-gewichtet.
- hard_check.py    Nachrechnung der HARTEN Metriken bei den Amplituden, die
                   ein bereits vorhandener GEWICHTETER Scan gefunden hat -
                   ohne erneute Optimierung, plus Konsistenzanalyse.
- report.py        Plots (Vektor-PDF) und Markdown-Berichte fuer beide
                   Datensatz-Arten.
"""
