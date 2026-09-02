# Aufräumen: diese Dateien können gelöscht werden

Beim Umbau am 2026-09-02 sind alle Hilfsmodule nach `lib/` gewandert und die
beiden Scan-Dialoge haben kürzere Namen bekommen. Die **alten Dateien konnten
von hier aus nicht gelöscht werden** (die Verbindung zu diesem Rechner darf
schreiben, aber nicht löschen) – sie liegen also noch daneben.

Sie stören nichts: alle `run_*.py` legen `lib/` an den Anfang des `sys.path`,
und `Combinated_Optimization/lib/paths.py` sucht ebenfalls zuerst in
`Weighted_Optimization/lib`. Es wird also garantiert die Fassung aus `lib/`
benutzt. Aber sie sind Ballast, und genau darum ging es. **Bitte im Explorer
löschen:**

```
Weighted_Optimization\weighted_winwidthscan_startdialog.py
Weighted_Optimization\weighted_winwidthampscan_startdialog.py
Weighted_Optimization\beispiel_weighted_amp_scan_ergebnisse_replotten.py
Weighted_Optimization\beispiel_weighted_amp_fit_abhaengigkeiten.py
Weighted_Optimization\weighted_multitone_flattop_optimizer.py
Weighted_Optimization\weighted_multitone_amplitude_dependence_plots.py
Weighted_Optimization\weighted_amp_scan_methods.py
Weighted_Optimization\weighted_use_torch.py
Weighted_Optimization\fit_central_amplitudes.py
Weighted_Optimization\fit_waist_width_relation.py
Weighted_Optimization\fit_uniformity_valley_overview.py
Weighted_Optimization\scan_checkpoint.py
Weighted_Optimization\resume_picker.py
Weighted_Optimization\airy_scale.py
Weighted_Optimization\perf_log.py
Weighted_Optimization\__pycache__\        (ganzer Ordner)
```

Danach diese Datei hier auch.

## Wo ist was hin?

| alt | neu |
|---|---|
| `weighted_winwidthscan_startdialog.py` | `run_scan.py` |
| `weighted_winwidthampscan_startdialog.py` | `run_amp_scan.py` |
| `beispiel_weighted_amp_scan_ergebnisse_replotten.py` | `Old_Scripts/` – abgelöst durch das neue `run_plots.py` |
| `beispiel_weighted_amp_fit_abhaengigkeiten.py` | `Old_Scripts/` – abgelöst durch `run_all_fits.py`, Schritt 1 |
| alle übrigen `.py` | `lib/` – inhaltlich unverändert |

`run_all_fits.py`, `plot_crosstalk_stripe_geometry.py` und
`plot_neighbour_airy_rings.py` behalten ihren Namen und bleiben oben liegen –
sie werden weiterhin direkt ausgeführt.

`Results/`, `Bilder/`, `Fit_Plots/` und `Fit_Results/` bleiben, wo sie sind.
