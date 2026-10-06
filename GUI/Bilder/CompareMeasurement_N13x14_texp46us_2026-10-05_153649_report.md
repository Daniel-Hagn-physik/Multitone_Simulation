# Predicted vs. measured camera frame

Generated 2026-10-05_153649 by `camera_series.py` (Beating_Multitone_GUI).

Saved at 13.60 cm width with 10 pt type - goes into LaTeX with `\includegraphics[width=0.85\linewidth]` unscaled, so the 10 pt in the figure are 10 pt on paper.

## Parameters

| quantity | value |
|---|---|
| measured file | C:\Users\Legion\OneDrive\Dokumente\Uni2\Buch\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Beating_Testbild\05.10\fokus2\13x14_1.56MHz_45us_0.12V_1SF_focus2.bmp |
| read as | PIL, mode L |
| orientation | rotation 90 deg, mirror x False, mirror y False |
| background removed | 5.0 % percentile |
| window | profile extent x 1.50, same rule for both panels |
| simulated panel | best match to the measurement -> frame 2 (t_0 = 45.6 us) |
| N_x x N_y | 13 x 14 |
| width_x / width_y | 1.560000 / 1.560000 MHz |
| build | single lens f = 45.000 mm, w_in = 1.7500 mm |
| waist (focus) | 6.5072 um |
| exposure | 45.6000 us |
| f_0 / T_0 | 10.000000 kHz / 100.0000 us |
| tone phases x [deg] | 0.00, 0.00, 60.00, 180.00, 0.00, 240.00, 180.00, 180.00, 240.00, 0.00, 180.00, 60.00, 360.00 |
| tone phases y [deg] | 0.00, 0.00, 55.40, 166.20, 332.30, 193.80, 110.80, 83.10, 110.80, 193.80, 332.30, 166.20, 55.40, 0.00 |

## Numbers

```
Correlation with the shown panel r = 0.885 (time average 0.910, best frame 0.885 at t_0 = 45.6 us).  shifted by (-3, +3) px for the alignment.
U = std/mean above half maximum: simulation 21.4 % over 8231 px, measurement 15.6 % over 6883 px.  0.00 % of the measured pixels sit at the top of the range.
```

## What is not in the figure

No position axes. Neither the magnification at the camera nor the pixel the profile sits on is known well enough for a micrometre scale, and a wrong scale is worse than none. Both panels are cut to the same fraction of the profile size and each is normalised to its own maximum: what is compared is the shape.

Left panel: simulation. Right panel: the measured frame. Without captions in the graphic that order is what names them - say it in the LaTeX caption.

## Files

- `CompareMeasurement_N13x14_texp46us_2026-10-05_153649.pdf`
