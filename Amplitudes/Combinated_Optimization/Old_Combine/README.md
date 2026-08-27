# Old_Combine - archivierte "Kombinieren zweier Datensaetze"-Skripte

Archiviert am 2026-08-27 (siehe `status.md`, Nachtrag 30, im claude.ai-Projekt
"Amplituden Abhängigkeit"). Kurswechsel im Chat: "Vergiss erstmal das
Kombinieren der Zwei Datensätze, das wird rechnerisch nichts." - der Ansatz,
zwei UNABHAENGIG optimierte Datensaetze (ein "harter", ein atom-gewichteter)
nachtraeglich zu einer gemeinsamen Bewertung zu verrechnen, wurde vom Nutzer
verworfen: die dabei jeweils gefundenen Amplituden-Verhaeltnisse (r_x, r_y)
sind zwischen beiden Datensaetzen unterschiedlich, die Kombination bewertet
also de facto zwei nicht vergleichbare Konfigurationen.

Die vier Dateien in diesem Ordner sind der UNVERAENDERTE, funktionierende
Stand dieser Skripte zum Zeitpunkt der Archivierung (bit-identisch zur
zuletzt auf dem Geraet vorhandenen Version):

- `combine_existing_datasets.py`
- `combined_amp_scan_methods.py`
- `combined_winwidthampscan_startdialog.py`
- `fit_combined_amp_region.py`

An ihrer urspruenglichen Stelle (eine Ebene hoeher, direkt in
`Combinated_Optimization/`) liegen jetzt nur noch kurze Stub-Dateien, die auf
diesen Ordner sowie auf die neuen Ersatz-Skripte verweisen (kein
Schreibzugriff zum Loeschen von Dateien in der Sitzung, die diesen Umbau
durchgefuehrt hat - daher Stubs statt Entfernen).

**Ersetzt durch** (siehe deren Modul-Docstrings fuer den vollen Hintergrund):

| Alt (hier archiviert)                          | Neu                                          |
|-------------------------------------------------|-----------------------------------------------|
| `combined_amp_scan_methods.py`                  | `comparison_scan_methods.py`                  |
| `combine_existing_datasets.py`                  | `comparison_from_existing_weighted.py`        |
| `combined_winwidthampscan_startdialog.py`       | `comparison_winwidthampscan_startdialog.py`   |
| `fit_combined_amp_region.py`                    | `fit_comparison_region.py`                    |

Neue Idee: KEINE zweite (r_x,r_y)-Optimierung/Kombination mehr - stattdessen
werden bei einem bereits vorhandenen, atom-gewichteten Amplituden-Scan die
harten Metriken bei GENAU dessen bereits gefundenen r_x/r_y nachgerechnet,
um zu pruefen, ob unter dem gewichteten Ziel gute Punkte auch unter dem
harten Ziel gut bleiben (siehe `comparison_scan_methods.py`).

**Wichtig:** die FEST-Amplituden-Kombination (`combined_scan_methods.py`,
`combined_winwidthscan_startdialog.py`, `combined_scan_plots.py`,
`fit_combined_region.py`, `run_all_fits_combined.py`) ist von diesem Umbau
NICHT betroffen und bleibt unveraendert aktiv - dort wird an jedem Punkt mit
IDENTISCHEN, festen Amplituden gerechnet, das Amplituden-Mismatch-Problem
tritt dort gar nicht auf.

Diese Dateien koennen bei Gelegenheit vom Nutzer selbst geloescht werden,
falls sie nicht mehr gebraucht werden.
