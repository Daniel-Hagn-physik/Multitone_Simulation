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
    ../run_single_beam.py    ->  lib.paths, lib.single_beam, lib.single_beam_report

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
- single_beam.py   EIN Strahl statt eines Tonarrays: Sweep ueber den Waist
                   und Sweep ueber die Atomposition bei festem Waist,
                   harte Metriken ueber einer Kreisregion (Beam-Pointing),
                   atom-gewichtete Metriken und deren Penalty-Kombination.
                   Weder width noch r_x/r_y - beides ist bei einem Ton nicht
                   definiert.
- single_beam_report.py  Kurven ueber dem Waist als PDF (Stil, Farben und
                   Achsen-Buendelung aus report.py) und der zugehoerige
                   Markdown-Bericht.
"""
