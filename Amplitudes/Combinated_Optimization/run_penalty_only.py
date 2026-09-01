"""
run_penalty_only.py  -  WELCHE PARAMETER MACHEN DIE PENALTY MINIMAL?
====================================================================

    ==> Dieses Skript ausfuehren, wenn KEIN Scan und KEIN Plot gebraucht
    ==> wird, sondern eine Antwort auf die Frage:
    ==>
    ==>     "Waist und Brennweiten habe ich vorgegeben - wie muessen
    ==>      width, r_x und r_y sein?"

Im Dialog wird fuer jede der sieben Groessen

    Waist (µm nach der Linse), Width (MHz), r_x, r_y, f1, f2, fLO

einzeln eingestellt, ob sie VORGEGEBEN wird (fester Wert) oder FREI ist
(dann mit einem Bereich, in dem sie liegen darf). Die freien Groessen
werden anschliessend GEMEINSAM so gewaehlt, dass dieselbe
Penalty-Zielfunktion minimal wird, die auch run_penalty_scan.py an jedem
Gitterpunkt minimiert:

    U_kombi = 0.5*(U_hart + U_w) + combo_lambda*|U_hart - U_w|
    C_kombi = 0.5*(C_hart + C_w) + combo_lambda*|C_hart - C_w|
    J       = alpha*U_kombi + (1-alpha)*C_kombi   ->  min

Abgrenzung zu run_penalty_scan.py: dort wird ein (win_input, width)-Gitter
abgefahren und an JEDEM Punkt ueber (r_x, r_y) optimiert - das Ergebnis
ist eine Landkarte. Hier gibt es kein Gitter; das Ergebnis ist EIN
Parametersatz.

Ergebnis: Fit_Results/PenaltyOpt_N{Nx}x{Ny}_{Profil}_{Datum}.md
Keine .pkl, keine Plots - es gibt nichts zu plotten, das Ergebnis ist ein
einzelner Punkt.

Die anderen Hauptskripte:
    run_penalty_scan.py  -  neuen Datensatz mit der Penalty-Methode scannen
    run_hard_check.py    -  Hard Case zu einem vorhandenen Weighted-Scan
    run_plots.py         -  vorhandene Datensaetze plotten/auswerten
"""

import os
import sys
import time
from pathlib import Path as FilePath

from PyQt5.QtWidgets import (
    QApplication, QDialog, QFormLayout, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox, QMessageBox,
    QProgressDialog, QCheckBox, QLineEdit, QComboBox, QScrollArea, QWidget,
)
from PyQt5.QtCore import Qt

sys.path.insert(0, str(FilePath(__file__).resolve().parent))

# Alles kommt aus lib - auch der Optimierer, der eigentlich in
# ../Weighted_Optimization liegt (siehe Hinweis in lib/paths.py).
from lib import paths, combine, penalty_opt  # noqa: E402


ROLLE_FEST = "vorgeben"
ROLLE_FREI = "optimieren"


class PenaltyOnlyDialog(QDialog):
    """Pro Groesse: vorgeben oder optimieren lassen. Dazu die
    Penalty-Parameter, der Aufbau und die Suche."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Penalty-Optimierung - freie Parameter statt Gitter")
        self._save_name_auto = True

        outer_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        main_layout = QVBoxLayout(content)
        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)

        info = QLabel(
            "Jede Groesse wird entweder VORGEGEBEN oder OPTIMIERT. Die optimierten\n"
            "werden gemeinsam so gewaehlt, dass die Penalty-Zielfunktion minimal wird -\n"
            "innerhalb des Bereichs, der hier jeweils dahintersteht. Kein Gitter, kein\n"
            "Plot: das Ergebnis ist ein Parametersatz und ein Markdown-Bericht."
        )
        info.setStyleSheet("font-style: italic;")
        main_layout.addWidget(info)

        # ------------------------------------------------------------------
        # Die sieben Groessen
        # ------------------------------------------------------------------
        param_group = QGroupBox("Groessen - vorgeben oder optimieren lassen")
        grid = QGridLayout()
        for spalte, text in enumerate(("Groesse", "", "Wert", "von", "bis")):
            kopf = QLabel(f"<b>{text}</b>")
            grid.addWidget(kopf, 0, spalte)

        self.rows = {}
        for zeile, (key, klartext, einheit, stellen, spanne,
                    fest_default, bereich_default) in enumerate(penalty_opt.PARAM_SPECS, start=1):
            grid.addWidget(QLabel(penalty_opt.param_label(key)), zeile, 0)

            rolle = QComboBox()
            rolle.addItems([ROLLE_FEST, ROLLE_FREI])
            rolle.setToolTip(
                f"'{ROLLE_FEST}': {klartext} wird auf den Wert daneben festgehalten.\n"
                f"'{ROLLE_FREI}': {klartext} wird innerhalb von [von, bis] mitoptimiert."
            )
            grid.addWidget(rolle, zeile, 1)

            wert = self._spin(fest_default, spanne, stellen)
            von = self._spin(bereich_default[0], spanne, stellen)
            bis = self._spin(bereich_default[1], spanne, stellen)
            grid.addWidget(wert, zeile, 2)
            grid.addWidget(von, zeile, 3)
            grid.addWidget(bis, zeile, 4)

            self.rows[key] = dict(rolle=rolle, wert=wert, von=von, bis=bis)
            rolle.currentIndexChanged.connect(
                lambda _i, k=key: self._sync_row(k))

        param_group.setLayout(grid)
        main_layout.addWidget(param_group)

        hinweis = QLabel(
            "Der Waist ist der EFFEKTIVE Waist nach der Linse (µm) - genau die Groesse,\n"
            "die man vorgibt, wenn man sie kennt. Welcher Eingangswaist vor der Linse\n"
            "(mm, der Scan-Parameter win_input) dazugehoert, rechnet der Bericht aus\n"
            "den gefundenen Brennweiten zurueck."
        )
        hinweis.setStyleSheet("color: #555555;")
        main_layout.addWidget(hinweis)

        # ------------------------------------------------------------------
        # Penalty
        # ------------------------------------------------------------------
        penalty_group = QGroupBox("Penalty-Zielfunktion")
        penalty_form = QFormLayout()
        self.alpha = self._spin(0.70, (0.0, 1.0), 3, step=0.05)
        self.alpha.setToolTip("Gewicht der Uniformity. 1 - alpha ist das Gewicht des Crosstalks.")
        self.combo_lambda = self._spin(0.75, (0.0, 10.0), 3, step=0.05)
        self.combo_lambda.setToolTip(
            "Staerke des Penalty-Terms: bestraft Parameter, bei denen das harte und das\n"
            "atom-gewichtete Kriterium auseinanderlaufen. 0 = reiner Mittelwert.")
        penalty_form.addRow("alpha (Gewicht Uniformity):", self.alpha)
        penalty_form.addRow("combo_lambda (Penalty-Staerke):", self.combo_lambda)
        penalty_group.setLayout(penalty_form)
        main_layout.addWidget(penalty_group)

        # ------------------------------------------------------------------
        # Aufbau
        # ------------------------------------------------------------------
        setup_group = QGroupBox("Aufbau")
        setup_form = QFormLayout()
        self.N_x = QSpinBox(); self.N_x.setRange(1, 20); self.N_x.setValue(3)
        self.N_y = QSpinBox(); self.N_y.setRange(1, 20); self.N_y.setValue(4)
        self.profile = QComboBox(); self.profile.addItems(["airy", "gaussian"])
        self.offset = self._spin(100.0, (0.1, 1000.0), 4)
        self.n_grid = QSpinBox(); self.n_grid.setRange(100, 4000); self.n_grid.setSingleStep(100)
        self.n_grid.setValue(1000)
        self.n_grid.setToolTip(
            "Punkte pro Achse im globalen Intensitaetsgitter der HARTEN Metriken.\n"
            "Groesser ist langsamer (quadratisch) und macht das Ergebnis nachweislich\n"
            "NICHT genauer - siehe Abschnitt 'Wie genau ist das?' im Bericht.")
        for widget, name in ((self.N_x, "N_x:"), (self.N_y, "N_y:")):
            widget.valueChanged.connect(lambda _v: self._sync_save_name())
            setup_form.addRow(name, widget)
        self.profile.currentIndexChanged.connect(lambda _i: self._sync_save_name())
        setup_form.addRow("Profil:", self.profile)
        self.airy_scale_mode = QComboBox()
        for _key, label, _wert in combine.AIRY_SCALE_CHOICES:
            self.airy_scale_mode.addItem(label)
        self.airy_scale_mode.setToolTip(
            "Was die Zahl 'waist' physikalisch bedeuten soll - siehe den\n"
            "gleichnamigen Schalter in run_penalty_scan.py.\n\n"
            "Fuer einen Vergleich mit einem vorhandenen Scan dieselbe\n"
            "Parametrisierung waehlen, mit der jener gerechnet wurde (der\n"
            "Faktor steht in dessen Bericht unter Scan-Parameter).")
        self.airy_scale_mode.currentIndexChanged.connect(
            lambda _i: self._sync_airy_mode())
        setup_form.addRow("Parametrisierung:", self.airy_scale_mode)
        self.airy_scale_factor = self._spin(1.19, (0.1, 5.0), 6, 0.01)
        self.airy_scale_factor.setToolTip(
            "Nur beim Airy-Profil wirksam: first_zero_radius = Faktor * waist.\n"
            "Setzt die physikalische Spotgroesse und damit jede Metrik.\n\n"
            "Der bisherige Default 1.19 entspricht keiner gaengigen Konvention;\n"
            "gleicher 1/e^2-Radius wie ein Gauss waere 1.4830, bester Fit an die\n"
            "Hauptkeule 1.4499, gleiche FWHM 1.3956.\n\n"
            "Wichtig: fuer einen Vergleich mit einem vorhandenen Scan denselben\n"
            "Wert waehlen, mit dem jener gerechnet wurde (steht in dessen Bericht).")
        setup_form.addRow("airy_scale_factor:", self.airy_scale_factor)
        self.profile.currentIndexChanged.connect(lambda _i: self._sync_airy_mode())
        self._sync_airy_mode()
        setup_form.addRow("offset (MHz):", self.offset)
        setup_form.addRow("n_grid:", self.n_grid)
        setup_group.setLayout(setup_form)
        main_layout.addWidget(setup_group)

        # ------------------------------------------------------------------
        # Suche
        # ------------------------------------------------------------------
        search_group = QGroupBox("Suche")
        search_form = QFormLayout()
        self.n_starts = QSpinBox(); self.n_starts.setRange(1, 200); self.n_starts.setValue(8)
        self.n_starts.setToolTip(
            "Zahl der Startpunkte. Der erste ist immer die Mitte aller Bereiche, die\n"
            "uebrigen decken sie per Latin Hypercube ab. Mehr Startpunkte kosten linear\n"
            "mehr Zeit, sind aber der einzige Schutz gegen die feinen lokalen Minima,\n"
            "die das Rauschen der harten Metriken erzeugt. 1 = nur der Mittelpunkt.")
        self.n_jobs = QSpinBox(); self.n_jobs.setRange(1, max(1, os.cpu_count() or 1))
        self.n_jobs.setValue(min(4, max(1, (os.cpu_count() or 1))))
        self.n_jobs.setToolTip("Startpunkte parallel auf so viele Prozesse verteilen.")
        self.maxiter = QSpinBox(); self.maxiter.setRange(20, 20000); self.maxiter.setSingleStep(50)
        self.maxiter.setValue(300)
        self.maxiter.setToolTip("Nelder-Mead-Iterationen je Startpunkt (Obergrenze).")
        search_form.addRow("Startpunkte:", self.n_starts)
        search_form.addRow("Parallele Prozesse:", self.n_jobs)
        search_form.addRow("max. Iterationen je Start:", self.maxiter)
        search_group.setLayout(search_form)
        main_layout.addWidget(search_group)

        # ------------------------------------------------------------------
        # Ausgabe
        # ------------------------------------------------------------------
        out_group = QGroupBox("Ausgabe")
        out_form = QFormLayout()
        self.save_name = QLineEdit()
        self.save_name.setToolTip(f"Wird in {paths.FIT_RESULTS_DIR} abgelegt.")
        self.save_name.textEdited.connect(self._on_save_name_edited)
        out_form.addRow("Bericht:", self.save_name)
        self.ask_before_save = QCheckBox("Vor dem Ueberschreiben nachfragen")
        self.ask_before_save.setChecked(True)
        out_form.addRow(self.ask_before_save)
        out_group.setLayout(out_form)
        main_layout.addWidget(out_group)

        self.summary = QLabel("")
        self.summary.setStyleSheet("color: #444444;")
        main_layout.addWidget(self.summary)
        main_layout.addStretch(1)

        # -- Buttons AUSSERHALB der Scrollflaeche --
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Optimierung starten")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_accept)
        cancel_btn = QPushButton("Abbrechen")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        outer_layout.addLayout(btn_layout)

        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry().height() if screen is not None else 900
        self.resize(760, min(760, int(avail * 0.85)))

        # Vorbelegung: Waist und die drei Brennweiten vorgeben, width/r_x/r_y
        # optimieren - genau der Fall, fuer den dieses Skript gebaut wurde.
        for key in ("width_MHz", "r_x", "r_y"):
            self.rows[key]["rolle"].setCurrentText(ROLLE_FREI)
        for key in penalty_opt.PARAM_KEYS:
            self._sync_row(key)
        self._sync_save_name()

    # ------------------------------------------------------------------
    def _sync_airy_mode(self):
        """Feld nur bei Airy UND 'frei eingeben' bedienbar; bei einer
        benannten Konvention zeigt es deren Wert."""
        airy = self.profile.currentText() == "airy"
        _key, _label, wert = combine.AIRY_SCALE_CHOICES[
            self.airy_scale_mode.currentIndex()]
        self.airy_scale_mode.setEnabled(airy)
        if wert is not None:
            self.airy_scale_factor.setValue(float(wert))
        self.airy_scale_factor.setEnabled(airy and wert is None)

    @staticmethod
    def _spin(value, spanne, stellen, step=None):
        box = QDoubleSpinBox()
        box.setDecimals(stellen)
        box.setRange(spanne[0], spanne[1])
        box.setSingleStep(step if step is not None else 10 ** (-max(1, stellen - 2)))
        box.setValue(value)
        return box

    def _sync_row(self, key):
        """Nur die Felder aktiv lassen, die zur gewaehlten Rolle gehoeren -
        sonst steht im Dialog ein Wert, der gar nicht benutzt wird."""
        row = self.rows[key]
        frei = row["rolle"].currentText() == ROLLE_FREI
        row["wert"].setEnabled(not frei)
        row["von"].setEnabled(frei)
        row["bis"].setEnabled(frei)
        self._sync_summary()

    def _sync_summary(self):
        frei = [penalty_opt.PARAM_LABEL[k] for k in penalty_opt.PARAM_KEYS
                if self.rows[k]["rolle"].currentText() == ROLLE_FREI]
        if not frei:
            self.summary.setText("Nichts wird optimiert - es wird nur einmal ausgewertet.")
        else:
            self.summary.setText(f"Optimiert werden {len(frei)} Groesse(n): " + ", ".join(frei))

    def _on_save_name_edited(self, _text):
        self._save_name_auto = False

    def _sync_save_name(self):
        if not self._save_name_auto:
            return
        self.save_name.setText(penalty_opt.output_name(dict(
            N_x=self.N_x.value(), N_y=self.N_y.value(),
            profile=self.profile.currentText())))

    # ------------------------------------------------------------------
    def _on_accept(self):
        """Bereiche pruefen, BEVOR die Optimierung laeuft - sonst faellt ein
        leerer Bereich erst mitten im Lauf auf."""
        kaputt = []
        for key in penalty_opt.PARAM_KEYS:
            row = self.rows[key]
            if row["rolle"].currentText() != ROLLE_FREI:
                continue
            if row["bis"].value() <= row["von"].value():
                kaputt.append(penalty_opt.param_label(key))
        if kaputt:
            QMessageBox.warning(
                self, "Bereich leer",
                "Bei diesen Groessen ist 'bis' nicht groesser als 'von':\n\n  "
                + "\n  ".join(kaputt))
            return
        if not self.save_name.text().strip():
            QMessageBox.warning(self, "Kein Dateiname", "Bitte einen Dateinamen fuer den Bericht angeben.")
            return
        self.accept()

    def get_values(self):
        feste, bereiche = {}, {}
        for key in penalty_opt.PARAM_KEYS:
            row = self.rows[key]
            if row["rolle"].currentText() == ROLLE_FREI:
                bereiche[key] = (row["von"].value(), row["bis"].value())
            else:
                feste[key] = row["wert"].value()

        # Die Brennweiten stehen in PARAM_SPECS und werden pro Auswertung
        # gesetzt; hier gehen sie nur als Startwert in den Optimierer, damit
        # das Objekt von Anfang an konsistent ist.
        optimizer_kwargs = dict(
            N_x=self.N_x.value(), N_y=self.N_y.value(),
            profile=self.profile.currentText(),
            airy_scale_factor=self.airy_scale_factor.value(),
            offset=self.offset.value() * 1e6,
            n_grid=self.n_grid.value(),
            weighted_metrics_enabled=True,
        )
        for key, attr in (("f1_mm", "f1"), ("f2_mm", "f2"), ("fLO_mm", "fLO")):
            if key in feste:
                optimizer_kwargs[attr] = feste[key] * 1e-3

        name = self.save_name.text().strip()
        if not name.lower().endswith(".md"):
            name += ".md"
        return dict(
            fixed=feste, ranges=bereiche,
            alpha=self.alpha.value(), combo_lambda=self.combo_lambda.value(),
            n_starts=self.n_starts.value(), n_jobs=self.n_jobs.value(),
            maxiter=self.maxiter.value(),
            optimizer_kwargs=optimizer_kwargs,
            save_name=name,
            ask_before_save=self.ask_before_save.isChecked(),
        )


def _zielpfad(name, ask_before_save):
    """Pfad im Fit_Results-Ordner; bei vorhandener Datei ggf. nachfragen.
    Antwortet der Nutzer mit Nein, wird _2, _3, ... angehaengt statt still
    zu ueberschreiben."""
    pfad = FilePath(paths.FIT_RESULTS_DIR) / name
    if not pfad.exists() or not ask_before_save:
        return pfad
    antwort = QMessageBox.question(
        None, "Bericht existiert bereits",
        f"'{pfad.name}' gibt es schon.\n\nUeberschreiben?",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    if antwort == QMessageBox.Yes:
        return pfad
    zaehler = 2
    while True:
        kandidat = pfad.with_name(f"{pfad.stem}_{zaehler}{pfad.suffix}")
        if not kandidat.exists():
            return kandidat
        zaehler += 1


def main():
    app = QApplication(sys.argv)

    dialog = PenaltyOnlyDialog()
    if dialog.exec_() != QDialog.Accepted:
        sys.exit(0)
    params = dialog.get_values()

    n_frei = len(params["ranges"])
    n_starts = params["n_starts"] if n_frei else 1
    progress = QProgressDialog(
        f"{n_frei} Groesse(n) werden gegen die Penalty optimiert...",
        "Abbrechen", 0, n_starts)
    progress.setWindowTitle("Penalty-Optimierung")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    def on_progress(done, total):
        progress.setMaximum(total)
        progress.setValue(done)
        QApplication.processEvents()
        return not progress.wasCanceled()

    t_start = time.perf_counter()
    try:
        ergebnis = penalty_opt.optimize_penalty(
            params["fixed"], params["ranges"],
            alpha=params["alpha"], combo_lambda=params["combo_lambda"],
            n_starts=params["n_starts"], n_jobs=params["n_jobs"],
            maxiter=params["maxiter"], optimizer_kwargs=params["optimizer_kwargs"],
            progress_callback=on_progress, verbose=True,
        )
    except ValueError as exc:
        progress.close()
        QMessageBox.critical(None, "Optimierung nicht moeglich", str(exc))
        sys.exit(1)
    finally:
        progress.close()

    print(f"Gesamtdauer: {time.perf_counter() - t_start:.1f}s")

    pfad = _zielpfad(params["save_name"], params["ask_before_save"])
    penalty_opt.write_report(ergebnis, pfad)

    bester = ergebnis["best"]
    if bester is None:
        QMessageBox.warning(
            None, "Kein gueltiger Punkt",
            "Alle Auswertungen sind auf einen ungueltigen Zustand gelaufen - "
            "meist liegt das an einem physikalisch nicht erreichbaren Bereich.\n\n"
            f"Der Bericht mit den Eingaben liegt trotzdem unter:\n{pfad}")
        sys.exit(0)

    zeilen = ["Penalty-Optimierung abgeschlossen.", ""]
    for key in penalty_opt.PARAM_KEYS:
        rolle = "vorgegeben" if key in ergebnis["fixed"] else "optimiert"
        zeilen.append(f"{penalty_opt.param_label(key)}: "
                      f"{bester['werte'][key]:.4f}   ({rolle})")
    d = bester["details"]
    zeilen += [
        "",
        f"Uniformity hart / gewichtet: {d['uniformity_hard'] * 100:.4f}% / "
        f"{d['uniformity_weighted'] * 100:.4f}%",
        f"Crosstalk  hart / gewichtet: {d['crosstalk_hard'] * 100:.4f}% / "
        f"{d['crosstalk_weighted'] * 100:.4f}%",
        f"J = {d['J']:.6f}",
        "",
        f"Bericht: {pfad}",
    ]
    QMessageBox.information(None, "Fertig", "\n".join(zeilen))


if __name__ == "__main__":
    main()
