# Predicted vs. measured camera frame

Generated 2026-10-31_112606 by `camera_series.py` (Beating_Multitone_GUI).

Saved at 13.60 cm width with 10 pt type - goes into LaTeX with `\includegraphics[width=0.85\linewidth]` unscaled, so the 10 pt in the figure are 10 pt on paper.

## Parameters

| quantity | value |
|---|---|
| measured file | \\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Beating_Testbild\05.10\13x14_1.56MHz_45us_0.12V_1SF_focus2.bmp |
| read as | PIL, mode L |
| orientation | rotation 270 deg, mirror x False, mirror y False |
| background removed | 5.0 % percentile |
| window | profile extent x 1.50, same rule for both panels |
| simulated panel | best match to the measurement -> frame 1 (t_0 = 0.0 us) |
| peak-to-valley region | central 70 % (width and height) of the box above half maximum, peak/valley = p99.5/p0.5 |
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
Correlation with the shown panel r = 0.960 (time average 0.972, best frame 0.960 at t_0 = 0.0 us).  shifted by (+1, -1) px for the alignment.
U = std/mean above half maximum: simulation 21.5 % over 8337 px, measurement 15.7 % over 8076 px.  0.00 % of the measured pixels sit at the top of the range.
Peak-to-valley (I_p - I_v)/<I>, p99.5/p0.5 in the central 70 % of the plateau:
  simulation  68.5 % (peak 0.992, valley 0.490, mean 0.733 of max; contrast 33.9 %; 4356 px)
  measurement 59.6 % (peak 0.939, valley 0.511, mean 0.717 of max; contrast 29.5 %; 4096 px)
  measurement / simulation = 0.87
```

## Peak-to-valley

Region: central 70 % of the box above half maximum; I_p / I_v = p99.5 / p0.5; intensities in units of the panel maximum.

| panel | region [px] | pixels | I_v | <I> | I_p | PV | contrast |
|---|---|---|---|---|---|---|---|
| simulation | 66 x 66 | 4356 | 0.490 | 0.733 | 0.992 | 68.5 % | 33.9 % |
| measurement | 64 x 64 | 4096 | 0.511 | 0.717 | 0.939 | 59.6 % | 29.5 % |

measurement / simulation = 0.87

## What is not in the figure

No position axes. Neither the magnification at the camera nor the pixel the profile sits on is known well enough for a micrometre scale, and a wrong scale is worse than none. Both panels are cut to the same fraction of the profile size and each is normalised to its own maximum: what is compared is the shape.

Left panel: simulation. Right panel: the measured frame. Without captions in the graphic that order is what names them - say it in the LaTeX caption.

## Files

- `CompareMeasurement_N13x14_texp46us_2026-10-31_112606.pdf`
- `CompareMeasurement_N13x14_texp46us_2026-10-31_112606_peak_to_valley.pdf` (evaluation region, histograms and numbers of the peak-to-valley)
