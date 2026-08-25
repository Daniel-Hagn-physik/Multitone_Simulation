"""
Combinated_Optimization/combined_scan_plots.py
================================================

Plot-Modul (bewusst ohne Optimizer-Abhaengigkeit, wie die anderen Plot-
Module im Projekt - laedt nur ein von save_scan_combined_results()
gepickeltes dict) fuer die kombinierten Fest-Amplituden-Scan-Ergebnisse aus
combined_scan_methods.py.

Zwei Plots:
- plot_metric_comparison(): 2x2-Uebersicht Uniformity_hart / Uniformity_w /
  Crosstalk_hart / Crosstalk_w nebeneinander, jeweils mit der Region
  (gestricheltes Rechteck) und dem besten Punkt ueberlagert - macht auf
  einen Blick sichtbar, WO hart und weighted sich einig bzw. uneinig sind.
- plot_combined_region(): EIN Panel, Gesamt-Score-Heatmap mit der Region
  (Rechteck) und dem besten Punkt - das ist "der tatsaechliche Bereich",
  nach dem man sich richten kann (siehe Chat).

Stil (RC-Parameter, BEST_POINT_STYLE, Speicher-Kollisionsschutz) wird 1:1
von AmplitudeScanPlotter (weighted_multitone_amplitude_dependence_plots.py)
uebernommen, damit Combinated_Optimization optisch nicht von den anderen
Plots im Projekt abweicht.
"""

import sys
from pathlib import Path as FilePath

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_WEIGHTED_DIR = FilePath(__file__).resolve().parent.parent / "Weighted_Optimization"
if str(_WEIGHTED_DIR) not in sys.path:
    sys.path.insert(0, str(_WEIGHTED_DIR))

from weighted_multitone_amplitude_dependence_plots import (  # noqa: E402
    AmplitudeScanPlotter, win_input_to_win, resolve_save_path,
)


def _default_dir(name):
    candidate = FilePath(__file__).resolve().parent / name
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except Exception:
        fallback = FilePath(".") / name
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DEFAULT_RESULTS_DIR = _default_dir("Results")
DEFAULT_IMAGES_DIR = _default_dir("Bilder")


class CombinedFixedScanPlotter:
    """
    Erzeugt Heatmap-Plots fuer den kombinierten Fest-Amplituden-Scan (hart +
    atom-gewichtet, siehe combined_scan_methods.py) aus einem von
    get_scan_combined_results()/load_combined_scan_results() gelieferten
    dict.
    """

    SCAN2D_RC = AmplitudeScanPlotter.SCAN2D_RC
    SCAN2D_SAVE_DPI = AmplitudeScanPlotter.SCAN2D_SAVE_DPI
    BEST_POINT_STYLE = AmplitudeScanPlotter.BEST_POINT_STYLE
    REGION_RECT_STYLE = dict(edgecolor="white", facecolor="none",
                              linewidth=2.3, linestyle="--", zorder=7)

    _REQUIRED_KEYS = (
        "uniformity_grid", "crosstalk_grid", "uniformity_weighted_grid", "eta_weighted_grid",
        "uniformity_kombi", "crosstalk_kombi", "combined_score",
        "win_input_vals", "width_vals", "region", "best",
    )

    def __init__(self, results, out_dir=None, confirm_overwrite=None):
        missing = [k for k in self._REQUIRED_KEYS if k not in results]
        if missing:
            raise ValueError(
                "results fehlen die Schluessel " + ", ".join(missing) + " - das sieht nicht "
                "nach den Rohdaten von scan_win_width_combined_uniformity() "
                "(combined_scan_methods.py) aus."
            )
        self.results = results
        self.out_dir = FilePath(out_dir) if out_dir is not None else DEFAULT_IMAGES_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_overwrite = confirm_overwrite

    # ------------------------------------------------------------------
    # kleine Hilfsfunktionen (analog WeightedFixedScanPlotter)
    # ------------------------------------------------------------------
    def _win_input_to_win(self, win_input):
        r = self.results
        return win_input_to_win(win_input, r['f1'], r['f2'], r['lambda_opt'], r['fLO'])

    def _profile_tag(self):
        profile = self.results.get('profile')
        if profile == 'airy':
            return 'Airy'
        if profile == 'gaussian':
            return 'Gauss'
        return 'ProfilUnbekannt'

    def _filetag(self):
        r = self.results
        n_win = len(r.get('win_input_vals', []))
        n_width = len(r.get('width_vals', []))
        return f"N{r['N_x']}x{r['N_y']}_{n_win}x{n_width}pts_{self._profile_tag()}_Combined"

    def _finish_figure(self, fig, filename, show, save, dpi=150):
        if save:
            out_file = resolve_save_path(self.out_dir, filename, confirm_overwrite=self.confirm_overwrite)
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {out_file}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return out_file if save else None

    def _win_axis_values(self, win_input_vals, win_axis):
        if win_axis == "before_lens":
            return win_input_vals * 1e3, r"Input waist $\omega_{\mathrm{in}}$ (before lenses, mm)", False
        elif win_axis == "after_lens":
            x = np.array([self._win_input_to_win(w) for w in win_input_vals]) * 1e6
            if len(x) > 1 and x[0] > x[-1]:
                return x[::-1], r"Waist at focus $\omega'$ (after lenses, µm)", True
            return x, r"Waist at focus $\omega'$ (after lenses, µm)", False
        else:
            raise ValueError(f"win_axis muss 'before_lens' oder 'after_lens' sein, nicht {win_axis!r}.")

    def _mark_point(self):
        r = self.results
        best = r.get('best', {})
        if best.get('win_input') is None:
            return None
        j = int(np.argmin(np.abs(r['win_input_vals'] - best['win_input'])))
        i = int(np.argmin(np.abs(r['width_vals'] - best['width'])))
        return i, j

    def _region_rect_xy(self, win_axis):
        """Rechteck-Eckkoordinaten (x0, x1, y0, y1) der Region in Plot-
        Einheiten (mm bzw. µm auf x, MHz auf y). None falls keine Region
        gefunden wurde. Behandelt die moegliche Achsenumkehr bei
        win_axis='after_lens' (win_input_to_win ist monoton fallend)."""
        region = self.results.get('region', {})
        if region.get('win_input_min') is None:
            return None
        x_edges, _, _ = self._win_axis_values(
            np.array([region['win_input_min'], region['win_input_max']]), win_axis,
        )
        x0, x1 = float(np.min(x_edges)), float(np.max(x_edges))
        y0, y1 = region['width_min'] * 1e-6, region['width_max'] * 1e-6
        return x0, x1, y0, y1

    def _draw_region_and_mark(self, ax, win_axis):
        rect_xy = self._region_rect_xy(win_axis)
        if rect_xy is not None:
            x0, x1, y0, y1 = rect_xy
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    label="combined region", **self.REGION_RECT_STYLE))
        mark = self._mark_point()
        if mark is not None:
            r = self.results
            i_mark, j_mark = mark
            x_mark = self._win_axis_values(np.array([r['win_input_vals'][j_mark]]), win_axis)[0][0]
            y_mark = r['width_vals'][i_mark] * 1e-6
            ax.plot(x_mark, y_mark, linestyle="none", label="best point (combined)",
                    **self.BEST_POINT_STYLE)
        if rect_xy is not None or mark is not None:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ------------------------------------------------------------------
    # Plot 1: 2x2-Vergleich hart vs. weighted (Uniformity + Crosstalk)
    # ------------------------------------------------------------------
    def plot_metric_comparison(self, show=True, save=True, win_axis="before_lens",
                                cmap_uniformity="viridis_r", cmap_crosstalk="Oranges"):
        """2x2: Uniformity_hart, Uniformity_w (oben), Crosstalk_hart,
        Crosstalk_w (unten) - jeweils mit Region-Rechteck + bestem Punkt.
        Zeigt direkt, wo harte und gewichtete Metrik uebereinstimmen bzw.
        auseinanderlaufen."""
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)

        panels = [
            ("uniformity_grid", r"Uniformity ($\sigma/\mu$) (%)", "Uniformity (hart, global)", cmap_uniformity),
            ("uniformity_weighted_grid", r"Uniformity$_w$ ($\sigma_w/\mu_w$) (%)",
             "Uniformity$_w$ (atom-gewichtet, lokal)", cmap_uniformity),
            ("crosstalk_grid", r"Crosstalk ($\eta$) (%)", "Crosstalk (hart, global)", cmap_crosstalk),
            ("eta_weighted_grid", r"Crosstalk$_w$ ($\eta_w$) (%)",
             "Crosstalk$_w$ (atom-gewichtet, lokal)", cmap_crosstalk),
        ]

        with plt.rc_context(self.SCAN2D_RC):
            fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), constrained_layout=True)
            for ax, (grid_key, cbar_label, title, cmap) in zip(axes.flat, panels):
                Z_plot = (r[grid_key][:, ::-1] if reversed_ else r[grid_key]) * 100.0
                im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
                fig.colorbar(im, ax=ax, label=cbar_label)
                ax.set_xlabel(x_label)
                ax.set_ylabel("width (MHz)")
                ax.set_title(title)
                self._draw_region_and_mark(ax, win_axis)

            fig.suptitle("Fixed-amplitude scan: hard vs. atom-weighted metrics "
                          "(same combined region/best point overlaid on all four)")

        tag = self._filetag()
        out = self._finish_figure(fig, f"FlatMultiTone_Scan_MetricComparison_{tag}.png",
                                   show, save, dpi=self.SCAN2D_SAVE_DPI)
        return out

    # ------------------------------------------------------------------
    # Plot 2: kombinierter Score + Region - "der tatsaechliche Bereich"
    # ------------------------------------------------------------------
    def plot_combined_region(self, show=True, save=True, win_axis="before_lens", cmap="magma_r"):
        """EIN Panel: combined_score-Heatmap mit Region-Rechteck + bestem
        Punkt - die Kernaussage dieses Skripts."""
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        x_vals, x_label, reversed_ = self._win_axis_values(win_input_vals, win_axis)
        Z_plot = r['combined_score'][:, ::-1] if reversed_ else r['combined_score']

        with plt.rc_context(self.SCAN2D_RC):
            fig, ax = plt.subplots(figsize=(8.5, 6.3), constrained_layout=True)
            im = ax.pcolormesh(x_vals, width_vals * 1e-6, Z_plot, shading="auto", cmap=cmap)
            fig.colorbar(im, ax=ax, label="combined score (normiert, kleiner = besser)")
            ax.set_xlabel(x_label)
            ax.set_ylabel("width (MHz)")

            region = r.get('region', {})
            pct = r.get('combo_percentile')
            n_region = region.get('n_points_region')
            n_total = region.get('n_points_total')
            title = "Combined region (hart + atom-gewichtet)"
            if pct is not None and n_region is not None:
                title += f"\nbeste {pct:.0f}% des Scores, groesstes Rechteck: {n_region}/{n_total} Punkte"
            ax.set_title(title)

            self._draw_region_and_mark(ax, win_axis)

        tag = self._filetag()
        out = self._finish_figure(fig, f"FlatMultiTone_Scan_CombinedRegion_{tag}.png",
                                   show, save, dpi=self.SCAN2D_SAVE_DPI)
        return out
