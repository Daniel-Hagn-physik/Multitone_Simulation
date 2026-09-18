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
    for ax in fig.get_axes():
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
