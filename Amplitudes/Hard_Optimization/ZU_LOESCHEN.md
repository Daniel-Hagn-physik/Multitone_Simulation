# Aufräumen: diese Dateien können gelöscht werden

Beim Umbau am 2026-09-02 sind alle Hilfsmodule nach `lib/` gewandert und die
beiden Scan-Dialoge haben brauchbare Namen bekommen. Die **alten Dateien
konnten von hier aus nicht gelöscht werden** (die Verbindung zu diesem Rechner
darf schreiben, aber nicht löschen) – sie liegen also noch daneben.

Sie stören nichts: alle `run_*.py` legen `lib/` an den Anfang des `sys.path`,
es wird also garantiert die Fassung aus `lib/` benutzt. Aber sie sind
Ballast, und genau darum ging es. **Bitte im Explorer löschen:**

```
Hard_Optimization\Winwidthscan startdialog.py
Hard_Optimization\Winwidthampscan startdialog.py
Hard_Optimization\Use torch.py
Hard_Optimization\beispiel_amp_scan_ergebnisse_replotten.py
Hard_Optimization\beispiel_amp_fit_abhaengigkeiten.py
Hard_Optimization\multitone_flattop_optimizer.py
Hard_Optimization\multitone_amplitude_dependence_plots.py
Hard_Optimization\scan_checkpoint.py
Hard_Optimization\resume_picker.py
Hard_Optimization\airy_scale.py
Hard_Optimization\perf_log.py
Hard_Optimization\__pycache__\          (ganzer Ordner)
```

Danach diese Datei hier auch.

## Wo ist was hin?

| alt | neu |
|---|---|
| `Winwidthscan startdialog.py` | `run_scan.py` |
| `Winwidthampscan startdialog.py` | `run_amp_scan.py` |
| `beispiel_amp_fit_abhaengigkeiten.py` | `run_amp_fits.py` |
| `beispiel_amp_scan_ergebnisse_replotten.py` | `Old_Scripts/` – abgelöst durch das neue `run_plots.py` |
| `Use torch.py` | `lib/use_torch.py` (mit Leerzeichen war es nie importierbar) |
| alle übrigen `.py` | `lib/` – inhaltlich unverändert |

`Results/`, `Bilder/`, `Fit_Plots/` und `Fit_Results/` bleiben, wo sie sind.
