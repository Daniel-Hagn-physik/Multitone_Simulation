"""kern - Hilfsmodule der Beating-Analyse.

    beating_physik.py   Physik und Numerik des Beating GUI (kein Qt)
    rb85_raman.py       Raman-Koeffizienten fuer Rb-85 (braucht ARC)
    beating_profil.py   Bruecke vom Beating-Profil zur Rabi-Rechnung

Diese Datei macht kern/ zu einem Paket, damit `from kern import beating_physik`
von Python UND von PyCharm aufgeloest wird. Sie importiert absichtlich nichts,
damit `import kern.beating_physik` nicht nebenbei ARC laedt.
"""
