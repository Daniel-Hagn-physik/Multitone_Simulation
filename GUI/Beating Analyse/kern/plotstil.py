"""plotstil - eine Breite und eine Schriftgroesse fuer alle gespeicherten PDFs.

Die Abbildungen gehen in ein LaTeX-Dokument mit A4 und 2,5 cm Raendern, also
16 cm Textbreite. Wird eine Abbildung dort mit
``\\includegraphics[width=\\textwidth]`` eingebunden, ist sie GENAU dann nicht
skaliert, wenn sie schon 16 cm breit gespeichert wurde - und nur dann sind die
10 pt in der Abbildung auch 10 pt auf dem Papier. Jede andere Breite wird beim
Einbinden gestaucht oder gedehnt, und dieselbe Beschriftung erscheint in zwei
Abbildungen in zwei Groessen.

Deshalb: JEDE gespeicherte Abbildung bekommt ``FIG_WIDTH_IN`` als Breite und
``style_figure()`` als letzten Schritt vor ``savefig``. Die Hoehe ist frei -
sie wird nicht mitskaliert, solange die Breite stimmt.

Kein Qt, keine Physik - nur diese beiden Zahlen und die Funktion, die sie auf
eine fertige Figure anwendet.
"""

FIG_WIDTH_IN = 16.0 / 2.54        # 6.299 in = 16 cm  -> width=\textwidth
FIG_WIDTH_SMALL_IN = 10.0 / 2.54  # 3.937 in = 10 cm  -> width=0.625\textwidth

# Warum es eine zweite Breite gibt: nicht jede Abbildung soll ueber die ganze
# Textbreite laufen. Eine kleiner eingebundene Abbildung DARF aber nicht
# einfach herunterskaliert werden - dann schrumpft ihre Schrift mit und ist
# nicht mehr 10 pt. Sie muss stattdessen gleich in der Breite gespeichert
# werden, in der sie im Dokument steht. Daher: Breite passend waehlen,
# Schriftgroesse immer 10 pt lassen.
FONT_PT = 10.0                    # eine Groesse fuer alles = Fliesstextgroesse
# 10 pt ist dieselbe Zielgroesse, auf die die Exporte des Flattop-GUI ueber
# ihren Skalierungsfaktor kommen (dort 20 pt in einer 12,5-Zoll-Figur, im
# Dokument auf 16 cm gestaucht). Beide Satz Abbildungen stehen damit im
# Dokument mit derselben Beschriftungsgroesse nebeneinander.


def style_figure(fig, pt=FONT_PT):
    """Setzt in der fertigen Figure JEDE Schrift auf `pt`.

    Nachtraeglich statt ueber rcParams, weil die Panels dieselben Methoden
    zeichnen wie der Bildschirm und dort die kleineren Groessen gebraucht
    werden - das hier ist der letzte Schritt vor dem Speichern.

    Erfasst Titel, Achsenbeschriftungen, Tick-Labels, den Exponenten am
    Achsenende, freie Texte und Legenden - auch die der Colorbar, denn deren
    Achse steht mit in ``fig.get_axes()``.
    """
    # Inset axes (z. B. eine Colorbar per ax.inset_axes) stehen nicht in
    # fig.get_axes(), sondern in ax.child_axes - die gehoeren mit dazu.
    axes = []
    for a in fig.get_axes():
        axes += [a] + list(getattr(a, "child_axes", []))
    for ax in axes:
        ax.title.set_fontsize(pt)
        ax.xaxis.label.set_fontsize(pt)
        ax.yaxis.label.set_fontsize(pt)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontsize(pt)
        ax.xaxis.get_offset_text().set_fontsize(pt)
        ax.yaxis.get_offset_text().set_fontsize(pt)
        for txt in ax.texts:
            txt.set_fontsize(pt)
        leg = ax.get_legend()
        if leg is not None:
            for txt in leg.get_texts():
                txt.set_fontsize(pt)
    for txt in fig.texts:
        txt.set_fontsize(pt)
    return fig


_LM_FAMILY = "Latin Modern Roman"


def latex_rc():
    """rcParams fuer gespeicherte Abbildungen: dieselbe Schrift wie die
    Flattop-Exporte (Weighted_Multitone_Lens_GUI.py, _rows_export_rc) -
    Latin Modern Roman fuer Text und Zahlen, Computer Modern fuer Formeln.

    Gesucht wird lmroman10-*.otf in den ueblichen MiKTeX-/TeX-Live-Ordnern;
    registriert wird nur die 10-pt-Optikgroesse. Fehlt die Schrift, bleibt
    es bei serif + Computer-Modern-Mathematik.

    Benutzung: ``with matplotlib.rc_context(latex_rc()):`` um das Anlegen,
    Zeichnen UND Speichern der Figure - die Schrift wird beim Anlegen der
    Texte festgelegt, nicht erst beim Speichern.
    """
    import glob
    import os
    from matplotlib import font_manager as fm
    rc = {"font.family": "serif", "mathtext.fontset": "cm",
          "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False}
    try:
        has_10 = any(f.name == _LM_FAMILY
                     and "lmroman10" in os.path.basename(f.fname).lower()
                     for f in fm.fontManager.ttflist)
        if not has_10:
            local = os.environ.get("LOCALAPPDATA", "")
            patterns = [
                r"C:\Program Files\MiKTeX*\fonts\opentype\public\lm\lmroman10-*.otf",
                os.path.join(local, "Programs", "MiKTeX", "fonts", "opentype",
                             "public", "lm", "lmroman10-*.otf"),
                r"C:\texlive\*\texmf-dist\fonts\opentype\public\lm\lmroman10-*.otf",
                os.path.join(local, "Microsoft", "Windows", "Fonts", "lmroman10-*.otf"),
                r"C:\Windows\Fonts\lmroman10-*.otf",
                "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-*.otf",
                "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
                "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
                "/Library/TeX/Root/texmf-dist/fonts/opentype/public/lm/lmroman10-*.otf",
            ]
            for pat in patterns:
                for path in glob.glob(pat):
                    fm.fontManager.addfont(path)
        # andere Optikgroessen derselben Familie aus der Auswahl nehmen
        fm.fontManager.ttflist = [
            f for f in fm.fontManager.ttflist
            if not (f.name == _LM_FAMILY
                    and "lmroman10" not in os.path.basename(f.fname).lower())]
        if any(f.name == _LM_FAMILY for f in fm.fontManager.ttflist):
            rc["font.family"] = _LM_FAMILY
            rc["axes.unicode_minus"] = True      # echtes Minus, LM hat es
    except Exception:
        pass
    return rc


# Fuer Bild-Plots (Karten, Profile): Ticks nach aussen, nur unten/links,
# keine Nebenstriche - auch auf der Colorbar. Fest eingestellt, damit eine
# matplotlibrc mit Ticks nach innen/rundum das Bild nicht verdeckt.
IMAGE_RC = {
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "xtick.bottom": True, "ytick.left": True,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
}


def image_rc():
    """latex_rc() plus IMAGE_RC - fuer gespeicherte Bild-Plots."""
    rc = latex_rc()
    rc.update(IMAGE_RC)
    return rc
