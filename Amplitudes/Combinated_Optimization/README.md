# Combinated_Optimization

Drei Skripte zum Ausfuehren, ein Ordner `lib/` mit dem, was sie benutzen.

## Was will ich gerade?

| Ich moechte ... | ... dann dieses Skript ausfuehren |
|---|---|
| **neue Daten erzeugen**, bei denen die Amplituden gemeinsam auf hart + gewichtet optimiert werden (Penalty-Methode) | `run_penalty_scan.py` |
| pruefen, **ob mein vorhandener Weighted-Datensatz auch im Hard Case gut ist** | `run_hard_check.py` |
| **vorhandene Datensaetze plotten** und den Bericht (neu) erzeugen | `run_plots.py` |

Alle drei oeffnen einen Dialog, in dem die Parameter stehen. Nichts muss
im Code geaendert werden.

**Welchen Datensatz?** `run_plots.py` und `run_hard_check.py` zeigen im
Dialog ein Auswahlfeld mit allen passenden Dateien aus `Results/` (neueste
zuerst) - es wird bewusst **nichts** automatisch vorausgewaehlt, damit nie
versehentlich der falsche Datensatz ausgewertet wird. Wer immer denselben
nimmt, traegt ihn ganz oben im jeweiligen Skript ein:

```python
PKL_DATEI    = "scan_amp_data_combined_N3x4_21x21pts_Airy.pkl"   # run_plots.py
WEIGHTED_PKL = "scan_amp_data_weighted_N3x4_10x10pts_Airy.pkl"   # run_hard_check.py
```

Dateiname genuegt (wird in `Results/` gesucht), ein vollstaendiger Pfad geht
auch. "Andere Datei..." waehlt eine von ausserhalb, "Aktualisieren" liest die
Liste neu ein (z.B. nach einem gerade fertig gewordenen Scan).

---

## 1. `run_penalty_scan.py` - neue Daten mit der Penalty-Methode

An JEDEM (win_input, width)-Gitterpunkt laeuft GENAU EINE
(r_x, r_y)-Optimierung, die direkt gegen die Kombination aus hartem und
atom-gewichtetem Ziel minimiert:

```
U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
J       = alpha*U_kombi + (1-alpha)*C_kombi        ->  min ueber (r_x, r_y)
```

`combo_lambda*|Differenz|` ist der **Penalty-Term**: er bestraft
Amplituden, bei denen das harte und das atom-gewichtete Kriterium
auseinanderlaufen. Hart und gewichtet werden bei jedem Optimierungsschritt
am SELBEN (r_x, r_y) ausgewertet - die gefundene Amplitude gilt daher fuer
beide zugleich.

- Ergebnis: `Results/scan_amp_data_combined_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl`
  (unveraendertes Namensmuster - die bereits vorhandenen Datensaetze passen dazu)
- Speicherort wird VOR dem Start abgefragt, stuendliche Zwischenspeicherung,
  automatisches Fortsetzen nach einem Abbruch
- am Ende optional direkt Plots + Bericht

## 2. `run_hard_check.py` - passt der Hard Case zu meinem Weighted?

Eingabe ist ein bereits vorhandener gewichteter Amplituden-Scan
(`scan_amp_data_weighted_*.pkl` aus `Weighted_Optimization/Results/`).
An jedem Gitterpunkt werden win_input, width und die dort GEFUNDENEN
Amplituden r_x/r_y genommen und damit **genau einmal** die harten Metriken
ausgewertet.

> Keine erneute Optimierung. Keine Kombination zweier Datensaetze.
> Nur: dieselben Amplituden, einmal hart nachgerechnet.

Deshalb ist der Lauf billig (Sekunden bis wenige Minuten) und braucht
keine Zwischenspeicherung. Die Eingangsdatei wird nicht veraendert.

Ausgewertet wird zweifach:

- **Vierfeldertafel**: wie viele der unter dem gewichteten Ziel guten
  Punkte sind auch unter dem harten Ziel gut? Dazu Pearson-Korrelationen
  von Score, Uniformity und Crosstalk.
- **Consistency-Score/Region**: dieselbe Penalty-Kombination wie oben,
  hier als raeumlich zusammenhaengende Karte der Uebereinstimmung.

Ergebnis: `Results/hard_check_N{Nx}x{Ny}_{n}x{n}pts_{Profil}.pkl`
(eigenes Namensmuster, nie mit den Penalty-Datensaetzen zu verwechseln).

## 3. `run_plots.py` - vorhandene Datensaetze auswerten

Erkennt automatisch, welche Art Datensatz vorliegt, und erzeugt in
`Fit_Plots/` bzw. `Fit_Results/`:

| Datei | Penalty-Scan | Hard-Check |
|---|---|---|
| `..._metric_comparison.pdf` (hart vs. gewichtet, 2x2) | x | x |
| `..._region.pdf` (Score-Karte mit Region und bestem Punkt) | x | x |
| `..._agreement.pdf` (Uebereinstimmungs-Karte, 4 Kategorien) | - | x |
| `..._score_scatter.pdf` (gewichtet vs. hart je Gitterpunkt) | - | x |
| `..._Report.md` (alle Kennzahlen) | x | x |
| 6-Panel-Uebersicht + Schnitte (PNG, optional) | x | x |

`alpha`, `combo_lambda` und das Perzentil lassen sich hier neu setzen -
Score, Region und die Vierfeldertafel werden dann aus den vorhandenen
Grids neu berechnet, **ohne** den teuren Scan zu wiederholen.

---

## Ordner

```
Combinated_Optimization/
    run_penalty_scan.py     <- ausfuehren: neue Daten (Penalty)
    run_hard_check.py       <- ausfuehren: Hard Case zu vorhandenem Weighted
    run_plots.py            <- ausfuehren: plotten/auswerten
    lib/                    <- wird von den drei Skripten benutzt,
                               nicht direkt ausfuehren
        paths.py            Ordner-Konstanten, Anbindung an Weighted_Optimization
        combine.py          Penalty-Kombination, Region, Laden/Speichern
        penalty_scan.py     die gemeinsame Amplituden-Optimierung
        hard_check.py       harte Nachrechnung + Konsistenz-Analyse
        report.py           Plots und Markdown-Berichte
    Results/                gespeicherte Datensaetze (.pkl)
    Fit_Plots/              Vektor-PDFs der Auswertung
    Fit_Results/            Markdown-Berichte
    Bilder/                 PNG-Ausgaben
    Old_Combine/            Archiv des verworfenen GETRENNTEN Verfahrens
```

Der eigentliche Optimierer (`MultitoneFlatTopOptimizer`), die Plot-Klassen,
`scan_checkpoint` und `perf_log` liegen unverändert in
`../Weighted_Optimization/` und werden von hier aus mitbenutzt, nicht
dupliziert.

## PyCharm markiert Importe rot?

Die Module aus `../Weighted_Optimization/` werden erst zur **Laufzeit** über
einen `sys.path`-Eintrag gefunden. PyCharms statische Analyse kennt diesen
Eintrag nicht und färbt solche Importe rot ("unresolved reference") — **der
Code läuft trotzdem**, es ist ein reines Anzeigeproblem.

Zwei Dinge dagegen:

1. Alle Importe aus `Weighted_Optimization` sind in `lib/` gebündelt. Die
   drei Skripte, die du täglich öffnest, importieren nur noch aus `lib` —
   dort ist also kein Rot mehr.
2. Dauerhaft weg bekommst du es mit einem einmaligen Handgriff: Rechtsklick
   auf den Ordner **`Weighted_Optimization`** → **"Mark Directory as"** →
   **"Sources Root"**. Danach kennt PyCharm die Module und färbt sie normal
   ein — auch in `lib/`.

## Was hier bewusst NICHT mehr drin ist

Das **getrennte** Verfahren (hart und gewichtet je einzeln optimieren,
danach die an verschiedenen r_x/r_y erreichten Metriken verrechnen) liegt
unveraendert im Archivordner `Old_Combine/` und wird nicht mehr angeboten -
es war rechnerisch nicht schluessig, weil die beiden Optimierungen am
selben Gitterpunkt unterschiedliche Amplituden liefern.

## Verifikation

Die Auswertung wurde gegen den vorhandenen Datensatz
`scan_amp_data_combined_N3x4_21x21pts_Airy.pkl` geprueft: normierte Grids,
`combined_score`, Schwellwert, Region-Maske, Region-Rechteck und bester
Punkt stimmen bit-exakt mit dem frueher erzeugten Ergebnis ueberein
(win_input = 1.7000 mm, width = 0.2300 MHz, r_x/r_y = 0.9997/1.1869,
Region 111/441 Punkte, waist 0.7872..0.9986 µm, width 0.22..0.25 MHz).
