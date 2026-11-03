# Predicted vs. measured camera frame

Generated 2026-10-29_115919 by `camera_series.py` (Beating_Multitone_GUI).

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
Correlation with the shown panel r = 0.898 (time average 0.910, best frame 0.898 at t_0 = 0.0 us).  shifted by (+2, -2) px for the alignment.
U = std/mean above half maximum: simulation 21.5 % over 8337 px, measurement 15.8 % over 6926 px.  0.00 % of the measured pixels sit at the top of the range.
Peak-to-valley (I_p - I_v)/<I>, p99.5/p0.5 in the central 70 % of the plateau:
  simulation  68.5 % (peak 0.992, valley 0.490, mean 0.733 of max; contrast 33.9 %; 4356 px)
  measurement 59.0 % (peak 0.948, valley 0.519, mean 0.726 of max; contrast 29.2 %; 3540 px)
  measurement / simulation = 0.86
```

## Peak-to-valley

Region: central 70 % of the box above half maximum; I_p / I_v = p99.5 / p0.5; intensities in units of the panel maximum.

| panel | region [px] | pixels | I_v | <I> | I_p | PV | contrast |
|---|---|---|---|---|---|---|---|
| simulation | 66 x 66 | 4356 | 0.490 | 0.733 | 0.992 | 68.5 % | 33.9 % |
| measurement | 60 x 59 | 3540 | 0.519 | 0.726 | 0.948 | 59.0 % | 29.2 % |

measurement / simulation = 0.86

## What is not in the figure

No position axes. Neither the magnification at the camera nor the pixel the profile sits on is known well enough for a micrometre scale, and a wrong scale is worse than none. Both panels are cut to the same fraction of the profile size and each is normalised to its own maximum: what is compared is the shape.

Left panel: simulation. Right panel: the measured frame. Without captions in the graphic that order is what names them - say it in the LaTeX caption.

## Files

- `CompareMeasurement_N13x14_texp46us_2026-10-29_115919.pdf`
