# Old_Scripts – abgelöste Skripte

Archiv. **Nichts hier wird noch benutzt**, und lauffähig ist es an dieser
Stelle auch nicht mehr (die Module, die es importiert, liegen jetzt in
`../lib/`). Es steht hier nur als Referenz, falls jemand nachsehen will, wie
etwas früher gerechnet wurde.

| Datei | abgelöst durch | warum |
|---|---|---|
| `beispiel_amp_scan_ergebnisse_replotten.py` | `../run_plots.py` | dasselbe Ziel – vorhandene Datensätze neu plotten –, aber über Konfig-Konstanten oben im Skript statt über einen Dialog. `run_plots.py` kann alles davon und zusätzlich Talschnitt, Geradenfit, verbotenen Bereich und Markdown-Bericht. |

Die Ausreißer-Bereinigung an den `r_bounds`-Schranken
(`detect_amp_outliers()` / `clean_amp_scan_results()`), die dieses Skript
demonstrierte, steckt unverändert in
`../lib/multitone_amplitude_dependence_plots.py` und lässt sich von dort
weiterhin aufrufen. `run_plots.py` markiert geklemmte Punkte stattdessen
grau in den Amplituden-Karten und nennt ihren Anteil auf der Konsole, statt
sie stillschweigend zu ersetzen.
