"""
Combinated_Optimization/lib
===========================

Hilfsmodule fuer die drei Hauptskripte eine Ebene hoeher. Diese Dateien
werden NICHT direkt ausgefuehrt - sie werden von den Hauptskripten
importiert:

    ../run_penalty_scan.py   ->  lib.paths, lib.combine, lib.penalty_scan, lib.report
    ../run_penalty_only.py   ->  lib.paths, lib.penalty_opt
    ../run_hard_check.py     ->  lib.paths, lib.combine, lib.hard_check,   lib.report
    ../run_plots.py          ->  lib.paths, lib.combine, lib.report

Inhalt:

- paths.py         Ordner-Konstanten (Results/Bilder/Fit_Plots/Fit_Results) und
                   der sys.path-Eintrag auf ../../Weighted_Optimization, ueber
                   den der eigentliche Optimierer erreichbar ist.
- combine.py       Die Penalty-Kombination (Normierung, Mittelwert +
                   combo_lambda*|Differenz|, combined_score), die Region-Suche
                   (groesstes Rechteck), der verbotene Bereich (Ueberlappung
                   der Eck-Spots) sowie Laden/Speichern/Neuberechnen.
- penalty_scan.py  Die gemeinsame (jointe) Amplituden-Optimierung MIT
                   Penalty-Term: pro Gitterpunkt EINE Nelder-Mead-Optimierung
                   direkt gegen die Kombination aus hart + atom-gewichtet.
- penalty_opt.py   Dieselbe Penalty-Zielfunktion, aber OHNE Gitter: der Nutzer
                   gibt einen Teil der Groessen vor, die uebrigen werden
                   gemeinsam gegen J optimiert. Schreibt nur einen Bericht.
- hard_check.py    Nachrechnung der HARTEN Metriken bei den Amplituden, die
                   ein bereits vorhandener GEWICHTETER Scan gefunden hat -
                   ohne erneute Optimierung, plus Konsistenzanalyse.
- report.py        Plots (Vektor-PDF) und Markdown-Berichte fuer beide
                   Datensatz-Arten.
"""
