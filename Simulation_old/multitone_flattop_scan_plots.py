"""
Multitone FlatTop - 2D Scan Plots
===================================

Plotting-only companion to `multitone_flattop_optimizer.py`. Deliberately
has NO dependency on that module (or on scipy) - it only needs the small
dict produced by `MultitoneFlatTopOptimizer.get_scan_results()` /
`save_scan_results()`, either handed over directly in the same session or
reloaded from a pickle file via `load_scan_results()`.

Why this split exists: `scan_win_width_uniformity()` can be slow (its
runtime scales with n_win_input * n_width grid points, each a full
intensity-profile evaluation). Iterating on plot styling - colors, fonts,
labels, legend placement - previously meant re-running that whole scan
every time. Now the scan only needs to run ONCE; after that, tweaking a
plot is just re-running this file, which loads the saved numbers.

Typische Verwendung:

    # nachdem Skript 1 (multitone_flattop_optimizer.py) das hier lief:
    #   opt.scan_win_width_uniformity(...)
    #   opt.save_scan_results("scan_data.pkl")

    from multitone_flattop_scan_plots import load_scan_results, ScanPlotter

    results = load_scan_results("scan_data.pkl")
    plotter = ScanPlotter(results, out_dir=".")
    plotter.plot_scan2d_combined(show=True, save=True)

    # einzelne Heatmaps statt der Nebeneinander-Ansicht:
    plotter.plot_scan2d_uniformity(show=True, save=True)
    plotter.plot_scan2d_crosstalk(show=True, save=True)

Speichern - Kollisionsschutz:
Jede save=True-Methode nutzt resolve_save_path(), das VOR dem Überschreiben
einer bereits existierenden Datei nachfragt (Konsole: y/N-Eingabe). Bei
"nein" wird stattdessen automatisch ein neuer Dateiname mit angehängtem
Zähler (_2, _3, ...) verwendet - das alte Bild bleibt also erhalten, auch
wenn z.B. dieselben Parameter nochmal mit weniger Rasterpunkten geplottet
werden (der Dateiname hängt nur von offset/width/N_x/N_y ab, nicht von der
Auflösung des Scans, daher sonst ein stiller Überschreib-Fall). Für nicht-
interaktive/GUI-Nutzung kann eine eigene confirm_overwrite-Callback-Funktion
übergeben werden (siehe resolve_save_path()).
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ======================================================================
# Laden der von MultitoneFlatTopOptimizer.save_scan_results() gesicherten
# Rohdaten
# ======================================================================
def load_scan_results(filepath):
    """Lädt ein von MultitoneFlatTopOptimizer.save_scan_results() erzeugtes
    Pickle und gibt das enthaltene dict zurück (siehe get_scan_results())."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ======================================================================
# Leichte, freistehende Physik-Hilfsfunktionen (Duplikate der in
# multitone_flattop_optimizer.py verwendeten Formeln, absichtlich hier
# noch einmal definiert, damit dieses Modul komplett unabhängig bleibt)
# ======================================================================
def radius_from_angle(theta, f1, f2, fLO):
    return (f1 * fLO / f2) * np.tan(theta)


def win_input_to_win(win_input, f1, f2, lambda_opt, fLO):
    """Siehe MultitoneFlatTopOptimizer.win_input_to_win() für die Herleitung
    (3-Linsen-Kaskade f1 -> f2 Teleskop + fLO Fokussierlinse)."""
    if win_input <= 0:
        raise ValueError("win_input muss > 0 sein.")
    return (f1 / f2) * (lambda_opt * fLO) / (np.pi * win_input)


def width_to_um(width_hz, f1, f2, fLO, theta_max, f_band):
    """Siehe MultitoneFlatTopOptimizer.width_to_um()."""
    theta_width = theta_max * width_hz / f_band
    return radius_from_angle(theta_width, f1, f2, fLO) * 1e6


# ======================================================================
# Kollisionsschutz beim Speichern
# ======================================================================
def resolve_save_path(out_dir, filename, confirm_overwrite=None):
    """
    Gibt den Pfad zurück, unter dem eine Grafik gespeichert werden soll.

    Existiert `out_dir/filename` noch nicht, wird er direkt zurückgegeben.
    Existiert er bereits, wird `confirm_overwrite(path)` gefragt, ob
    überschrieben werden soll:
      - True  -> derselbe Pfad wird zurückgegeben (überschreiben)
      - False -> ein neuer Pfad mit angehängtem Zähler (_2, _3, ...) wird
                 gesucht und zurückgegeben (altes Bild bleibt erhalten)

    confirm_overwrite: Callable(Path) -> bool. Default: Konsolen-Eingabe
    (y/N). Für GUI-Nutzung kann hier z.B. eine Funktion übergeben werden,
    die stattdessen einen QMessageBox.question(...)-Dialog zeigt.
    """
    path = Path(out_dir) / filename
    if not path.exists():
        return path

    if confirm_overwrite is None:
        def confirm_overwrite(existing_path):
            answer = input(
                f"'{existing_path.name}' existiert bereits. Überschreiben? [y/N]: "
            ).strip().lower()
            return answer in ("y", "yes", "j", "ja")

    if confirm_overwrite(path):
        return path

    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            print(f"Bestehende Datei bleibt erhalten, speichere stattdessen als: {candidate.name}")
            return candidate
        counter += 1


# ======================================================================
# Plotter
# ======================================================================
class ScanPlotter:
    """
    Erzeugt alle 2D-Scan-Plots (Uniformity, Crosstalk, kombiniert) aus
    einem von get_scan_results()/load_scan_results() gelieferten dict -
    ohne den (potenziell langsamen) Scan selbst erneut zu berechnen.
    """

    # Schriftgrößen/dpi für die Scan-2D-Figuren, damit das gespeicherte PNG
    # sich direkt in ein LaTeX-Dokument einbinden lässt (auch verkleinert
    # noch lesbar, Serifenschrift wie üblicher LaTeX-Fließtext).
    SCAN2D_RC = {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
    SCAN2D_SAVE_DPI = 300

    def __init__(self, results, out_dir=".", confirm_overwrite=None):
        """
        results: dict von get_scan_results()/load_scan_results(), enthält
        u.a. win_input_vals, width_vals, uniformity_grid, crosstalk_grid,
        amps, alpha, best, N_x, N_y, f1, f2, fLO, lambda_opt, theta_max,
        f_band.

        confirm_overwrite: optionale Callable(Path) -> bool, wird an
        resolve_save_path() weitergereicht, z.B. um beim Überschreiben
        einen Qt-Dialog statt der Konsolen-Eingabe zu zeigen.
        """
        self.results = results
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.confirm_overwrite = confirm_overwrite

    # ------------------------------------------------------------------
    # Kleine Convenience-Wrapper um die freistehenden Physik-Funktionen,
    # mit den in self.results hinterlegten Parametern
    # ------------------------------------------------------------------
    def _win_input_to_win(self, win_input):
        r = self.results
        return win_input_to_win(win_input, r['f1'], r['f2'], r['lambda_opt'], r['fLO'])

    def _width_to_um(self, width_hz):
        r = self.results
        return width_to_um(width_hz, r['f1'], r['f2'], r['fLO'], r['theta_max'], r['f_band'])

    def _filetag(self):
        r = self.results
        return f"{r['N_x']}x{r['N_y']}"

    def _finish_figure(self, fig, filename, show, save, dpi=150):
        if save:
            out_file = resolve_save_path(self.out_dir, filename, confirm_overwrite=self.confirm_overwrite)
            fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved: {out_file.name}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _scan2d_info_lines(self, win_input_val, width_val, win_eff, uniformity, crosstalk, amps):
        """Baut die Infofeld-Textzeilen für einen einzelnen Scan-Punkt, mit
        LaTeX-Mathesymbolen statt Parameternamen mit Unterstrich."""
        r = self.results
        lines = [
            rf"$\omega_{{\mathrm{{in}}}}$ (before lenses):  {win_input_val*1e3:.4f} mm",
            rf"$\omega'$ (at focus):            {win_eff*1e6:.4f} µm" if win_eff is not None
            else r"$\omega'$ (at focus):            invalid ($\omega_{\mathrm{in}} \leq 0$)",
            rf"width:                       {width_val*1e-6:.4f} MHz",
            rf"$f_1$ / $f_2$ / $f_{{LO}}$:            {r['f1']*1e3:.1f} mm / {r['f2']*1e3:.1f} mm / {r['fLO']*1e3:.2f} mm",
        ]
        if uniformity is not None and np.isfinite(uniformity):
            lines.append(rf"Uniformity ($\sigma/\mu$):        {uniformity*100:.3f} %")
            lines.append(rf"Crosstalk ($\eta$):              {crosstalk*100:.3f} %")
        else:
            lines.append("Uniformity / Crosstalk:      invalid point")

        if amps is not None:
            amps_arr = np.asarray(amps)
            amp_x = amps_arr[:r['N_x']]
            amp_y = amps_arr[r['N_x']:r['N_x'] + r['N_y']]
            lines.append(rf"$a_x$:  {np.array2string(amp_x, precision=3)}")
            lines.append(rf"$a_y$:  {np.array2string(amp_y, precision=3)}")
        return lines

    # ------------------------------------------------------------------
    # Einzelne Heatmap (Uniformity ODER Crosstalk), interaktiv mit
    # Klick-Infofeld + rotem Auswahlrahmen
    # ------------------------------------------------------------------
    def plot_scan2d_uniformity(self, show=True, save=True, cmap="viridis", vmax=None):
        """Plottet die Uniformity-Heatmap. Siehe _plot_scan2d_metric() für
        Layout-/Interaktivitätsdetails."""
        self._plot_scan2d_metric(
            grid_key="uniformity_grid", colorbar_label="Uniformity (σ/μ)",
            title_metric="Uniformity", filename_suffix="Uniformity",
            cmap=cmap, vmax=vmax, show=show, save=save,
        )

    def plot_scan2d_crosstalk(self, show=True, save=True, cmap="Oranges", vmax=None):
        """Plottet die Crosstalk-Heatmap. Siehe _plot_scan2d_metric() für
        Layout-/Interaktivitätsdetails. Default-Colormap vermeidet
        Schwarztöne (anders als z.B. 'magma'), die bei einer Größe, wo
        Schwarz mit "keine Daten" verwechselt werden könnte, ungünstig
        wirken."""
        self._plot_scan2d_metric(
            grid_key="crosstalk_grid", colorbar_label="Crosstalk (η)",
            title_metric="Crosstalk", filename_suffix="Crosstalk",
            cmap=cmap, vmax=vmax, show=show, save=save,
        )

    def plot_scan2d(self, show=True, save=True, cmap="viridis", vmax=None):
        """Alias für plot_scan2d_uniformity(), aus Kompatibilitätsgründen."""
        self.plot_scan2d_uniformity(show=show, save=save, cmap=cmap, vmax=vmax)

    def _plot_scan2d_metric(self, grid_key, colorbar_label, title_metric, filename_suffix,
                             cmap, vmax, show, save):
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        Z = r[grid_key]
        amps = r.get('amps')
        best = r.get('best', {})

        with plt.rc_context(self.SCAN2D_RC):
            fig = plt.figure(figsize=(9, 8.9))
            ax = fig.add_axes([0.12, 0.36, 0.75, 0.56])
            info_ax = fig.add_axes([0.06, 0.03, 0.88, 0.24])
            info_ax.axis('off')

            im = ax.pcolormesh(
                win_input_vals * 1e3, width_vals * 1e-6, Z,
                shading='auto', cmap=cmap, vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label=colorbar_label)

            dx = (win_input_vals[1] - win_input_vals[0]) * 1e3 if len(win_input_vals) > 1 else 0.1
            dy = (width_vals[1] - width_vals[0]) * 1e-6 if len(width_vals) > 1 else 0.05
            x_pad, y_pad = dx / 2, dy / 2
            ax.set_xlim(win_input_vals[0] * 1e3 - x_pad, win_input_vals[-1] * 1e3 + x_pad)
            ax.set_ylim(width_vals[0] * 1e-6 - y_pad, width_vals[-1] * 1e-6 + y_pad)

            if best.get("win_input") is not None:
                ax.plot(
                    best["win_input"] * 1e3, best["width"] * 1e-6,
                    marker="+", color="red", markeredgecolor="red",
                    markersize=20, markeredgewidth=3, linestyle="none", zorder=7,
                    label="Optimal (combined)",
                )
                ax.legend(loc="upper right", framealpha=0.9)

            ax.set_xlabel(r"Input waist $\omega_{\mathrm{in}}$ (mm)")
            ax.set_ylabel("width (MHz)")
            ax.set_title(title_metric)

            selection_rect = Rectangle(
                (0, 0), dx, dy, edgecolor="red", facecolor="none",
                linewidth=2.5, zorder=6,
            )
            selection_rect.set_visible(False)
            ax.add_patch(selection_rect)

            U = r['uniformity_grid']
            C = r['crosstalk_grid']

            if best.get("win_input") is not None:
                win_eff_best = self._win_input_to_win(best["win_input"])
                initial_lines = ["Optimal (combined), see cross:"]
                initial_lines += self._scan2d_info_lines(
                    best["win_input"], best["width"], win_eff_best,
                    best["uniformity"], best["crosstalk"], amps,
                )
                initial_lines.append("")
                initial_lines.append("Click any point in the heatmap to inspect it instead.")
                initial_msg = "\n".join(initial_lines)
            else:
                initial_msg = "Click on a point in the heatmap to show all parameters (incl. waist before/after the lenses)."

            info_text = info_ax.text(
                0.0, 1.0, initial_msg, va='top', ha='left', fontsize=14,
                family='monospace', transform=info_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9),
            )

            def _on_click(event):
                if event.inaxes != ax or event.xdata is None or event.ydata is None:
                    return

                win_input_val = event.xdata * 1e-3
                width_val = event.ydata * 1e6

                j = int(np.argmin(np.abs(win_input_vals - win_input_val)))
                i = int(np.argmin(np.abs(width_vals - width_val)))
                win_input_actual = win_input_vals[j]
                width_actual = width_vals[i]

                selection_rect.set_xy((win_input_actual * 1e3 - dx / 2, width_actual * 1e-6 - dy / 2))
                selection_rect.set_visible(True)

                try:
                    win_eff = self._win_input_to_win(win_input_actual)
                except ValueError:
                    win_eff = None

                # direkt aus dem bereits berechneten Grid lesen - kein
                # erneutes Evaluieren nötig, da der Klick immer auf den
                # nächstgelegenen tatsächlich berechneten Punkt einrastet.
                lines = self._scan2d_info_lines(
                    win_input_actual, width_actual, win_eff,
                    U[i, j], C[i, j], amps,
                )
                info_text.set_text("\n".join(lines))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', _on_click)

            tag = self._filetag()
            self._finish_figure(
                fig, f"FlatMultiTone_Scan2D_{filename_suffix}_{tag}.png",
                show, save, dpi=self.SCAN2D_SAVE_DPI,
            )

    # ------------------------------------------------------------------
    # Kombinierte Ansicht: Uniformity + Crosstalk nebeneinander
    # ------------------------------------------------------------------
    def plot_scan2d_combined(self, show=True, save=True,
                              cmap_uniformity="viridis", cmap_crosstalk="Oranges",
                              vmax_uniformity=None, vmax_crosstalk=None):
        """
        Plottet Uniformity und Crosstalk nebeneinander (der "finale" Scan-
        Plot). Jede Heatmap hat ihre EIGENE Colorbar (eigene Farbskala,
        auf die jeweiligen Daten skaliert) - Uniformity und Crosstalk
        unterscheiden sich stark in typischer Größenordnung, eine
        gemeinsame Colorbar würde den Kontrast der kleineren Größe
        auswaschen. Eine einzelne, gemeinsame Legende (ein Eintrag, mit
        Zeilenumbruch) sitzt in der Ecke des Uniformity-Plots statt pro
        Subplot dupliziert zu werden.

        Zwei unabhängige Figuren werden je nach Bedarf gebaut:
        - Die GESPEICHERTE Figur (save=True) ist sauber: kein Infofeld,
          kein roter Auswahlrahmen - nur die zwei Heatmaps, ihre
          Colorbars, und die Legende mit den Achsenwerten des kombiniert-
          optimalen Punkts (omega_in in mm, width in MHz und die
          entsprechende räumliche Länge in µm über width_to_um()).
        - Die ANGEZEIGTE Figur (show=True) hat zusätzlich das interaktive
          Infofeld sowie den Klick-Auswahlrahmen (wie
          plot_scan2d_uniformity()/plot_scan2d_crosstalk()): ein Klick in
          eine der beiden Heatmaps markiert die Zelle in BEIDEN und zeigt
          alle Details im Feld darunter.
        """
        if save:
            fig_save = self._build_scan2d_combined_figure(
                interactive=False, cmap_uniformity=cmap_uniformity, cmap_crosstalk=cmap_crosstalk,
                vmax_uniformity=vmax_uniformity, vmax_crosstalk=vmax_crosstalk,
            )
            tag = self._filetag()
            out_file = resolve_save_path(
                self.out_dir, f"FlatMultiTone_Scan2D_Combined_{tag}.png",
                confirm_overwrite=self.confirm_overwrite,
            )
            fig_save.savefig(out_file, dpi=self.SCAN2D_SAVE_DPI, bbox_inches='tight')
            print(f"Figure saved: {out_file.name}")
            plt.close(fig_save)

        if show:
            self._build_scan2d_combined_figure(
                interactive=True, cmap_uniformity=cmap_uniformity, cmap_crosstalk=cmap_crosstalk,
                vmax_uniformity=vmax_uniformity, vmax_crosstalk=vmax_crosstalk,
            )
            plt.show()

    def _build_scan2d_combined_figure(self, interactive, cmap_uniformity, cmap_crosstalk,
                                       vmax_uniformity, vmax_crosstalk):
        r = self.results
        win_input_vals = r['win_input_vals']
        width_vals = r['width_vals']
        U = r['uniformity_grid']
        C = r['crosstalk_grid']
        amps = r.get('amps')
        best = r.get('best', {})

        optimal_label = None
        if best.get("win_input") is not None:
            width_um = self._width_to_um(best["width"])
            optimal_label = (
                rf"Optimal: $\omega_{{\mathrm{{in}}}}$={best['win_input']*1e3:.3f} mm, "
                rf"width={best['width']*1e-6:.3f} MHz"
                "\n"
                rf"($\approx${width_um:.3f} µm spot spacing)"
            )

        with plt.rc_context(self.SCAN2D_RC):
            if interactive:
                fig = plt.figure(figsize=(15.5, 7.4))
                gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.32, wspace=0.28)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])
                info_ax = fig.add_subplot(gs[1, :])
                info_ax.axis('off')
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.4))

            panels = [
                (ax1, U, "Uniformity (σ/μ)", "Uniformity", cmap_uniformity, vmax_uniformity),
                (ax2, C, "Crosstalk (η)", "Crosstalk", cmap_crosstalk, vmax_crosstalk),
            ]

            dx = (win_input_vals[1] - win_input_vals[0]) * 1e3 if len(win_input_vals) > 1 else 0.1
            dy = (width_vals[1] - width_vals[0]) * 1e-6 if len(width_vals) > 1 else 0.05
            selection_rects = []

            for ax, Z, colorbar_label, title_metric, cmap, vmax in panels:
                im = ax.pcolormesh(
                    win_input_vals * 1e3, width_vals * 1e-6, Z,
                    shading='auto', cmap=cmap, vmax=vmax,
                )
                x_pad, y_pad = dx / 2, dy / 2
                ax.set_xlim(win_input_vals[0] * 1e3 - x_pad, win_input_vals[-1] * 1e3 + x_pad)
                ax.set_ylim(width_vals[0] * 1e-6 - y_pad, width_vals[-1] * 1e-6 + y_pad)

                fig.colorbar(im, ax=ax, label=colorbar_label)

                if optimal_label is not None:
                    ax.plot(
                        best["win_input"] * 1e3, best["width"] * 1e-6,
                        marker="+", color="red", markeredgecolor="red",
                        markersize=18, markeredgewidth=3, linestyle="none", zorder=7,
                        label=(optimal_label if ax is ax1 else None),
                    )

                ax.set_xlabel(r"Input waist $\omega_{\mathrm{in}}$ (mm)")
                ax.set_ylabel("width (MHz)")
                ax.set_title(title_metric)

                if interactive:
                    rect = Rectangle((0, 0), dx, dy, edgecolor="red", facecolor="none",
                                      linewidth=2.5, zorder=6)
                    rect.set_visible(False)
                    ax.add_patch(rect)
                    selection_rects.append(rect)

            if optimal_label is not None:
                ax1.legend(loc="upper right", framealpha=0.9, fontsize=12 if interactive else None)

            if not interactive:
                fig.tight_layout()
                return fig

            fig.subplots_adjust(left=0.055, right=0.95, top=0.93, bottom=0.05)

            if best.get("win_input") is not None:
                win_eff_best = self._win_input_to_win(best["win_input"])
                initial_lines = ["Optimal (combined), see cross:"]
                initial_lines += self._scan2d_info_lines(
                    best["win_input"], best["width"], win_eff_best,
                    best["uniformity"], best["crosstalk"], amps,
                )
                initial_lines.append("")
                initial_lines.append("Click any point in either heatmap to inspect it instead.")
                initial_msg = "\n".join(initial_lines)
            else:
                initial_msg = "Click on a point in either heatmap to show all parameters (incl. waist before/after the lenses)."

            info_text = info_ax.text(
                0.0, 1.0, initial_msg, va='top', ha='left', fontsize=14,
                family='monospace', transform=info_ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.9),
            )

            def _on_click(event):
                if event.inaxes not in (ax1, ax2) or event.xdata is None or event.ydata is None:
                    return

                win_input_val = event.xdata * 1e-3
                width_val = event.ydata * 1e6

                j = int(np.argmin(np.abs(win_input_vals - win_input_val)))
                i = int(np.argmin(np.abs(width_vals - width_val)))
                win_input_actual = win_input_vals[j]
                width_actual = width_vals[i]

                xy = (win_input_actual * 1e3 - dx / 2, width_actual * 1e-6 - dy / 2)
                for rect in selection_rects:
                    rect.set_xy(xy)
                    rect.set_visible(True)

                try:
                    win_eff = self._win_input_to_win(win_input_actual)
                except ValueError:
                    win_eff = None

                lines = self._scan2d_info_lines(
                    win_input_actual, width_actual, win_eff,
                    U[i, j], C[i, j], amps,
                )
                info_text.set_text("\n".join(lines))
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', _on_click)
            return fig