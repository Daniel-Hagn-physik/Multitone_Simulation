"""
GPU-Beschleunigung ueber PyTorch/CUDA, per Monkey-Patch auf die
dyn_gaussian_2d_weighted_distance_from_centers /
dyn_airy_2d_weighted_distance_from_centers Funktionszeiger in
multitone_flattop_optimizer.py (siehe dort, Kommentar direkt vor den
beiden Zeigern).

Aufruf: `import use_torch; use_torch.patch()` - typischerweise einmal
beim Programmstart im Hauptprozess (siehe WinWidthAmpScan_StartDialog.py).
Bei n_jobs > 1 (ProcessPoolExecutor) muss patch() zusaetzlich in JEDEM
Worker-Prozess erneut aufgerufen werden, siehe der neue
pool_initializer-Parameter von scan_win_width_amplitude_dependence():

    from functools import partial
    opt.scan_win_width_amplitude_dependence(
        ..., n_jobs=4, pool_initializer=use_torch.patch,
    )

WICHTIG - in dieser Sandbox konnte torch NICHT installiert werden (kein
funktionierender pip-Zugriff), dieses Modul wurde daher NICHT tatsaechlich
ausgefuehrt/getestet. Die Formeln wurden manuell gegen die bereits
getestete NumPy-Implementierung
(numpy_gaussian_2d_weighted_distance_from_centers /
numpy_airy_2d_weighted_distance_from_centers) geprueft und sind
mathematisch aequivalent (Broadcast-und-Summe ueber die Spot-Dimension
statt einer Python-Schleife ueber Spots), aber bitte vor produktivem
Einsatz einmal an einem Einzelpunkt gegen die CPU-Variante verifizieren
(z.B. torch aktivieren, einen Punkt mit _evaluate() auswerten, dann
dyn_..._distance_from_centers wieder auf die numpy_...-Funktion
zuruecksetzen und denselben Punkt nochmal auswerten, Ergebnisse
vergleichen).

Fuer automatisches GPU-mit-CPU-Fallback (siehe WinWidthAmpScan_StartDialog.py):
cuda_available() prueft, ob torch installiert ist UND eine CUDA-GPU findet,
OHNE dabei selbst eine Exception zu werfen (z.B. wenn torch gar nicht
installiert ist) - Aufrufer koennen so unbesorgt `if cuda_available():
patch(); ... else: ... parallelisierte CPU ...` schreiben.
"""


def cuda_available():
    """True, wenn torch importierbar ist UND torch.cuda.is_available() True
    liefert - sonst False (inkl. wenn torch gar nicht installiert ist oder
    die CUDA-Erkennung selbst eine Exception wirft, z.B. bei einer kaputten
    Treiber-Installation). Wirft absichtlich NIE selbst, damit Aufrufer im
    Zweifel sicher auf die CPU-Variante zurueckfallen koennen."""
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def patch():
    import torch
    from scipy.special import j0, j1
    import multitone_flattop_optimizer as mfo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using cuda")
    else:
        print(f"Using cpu")

    def cuda_gaussian_2d_weighted_distance_from_centers(
        X, Y, centers_x, centers_y, sigma, amps
    ):
        X = (
            torch.from_numpy(X)
            .to(device)
            .unsqueeze(0)
            .expand((len(amps), X.shape[0], X.shape[1]))
        )
        Y = (
            torch.from_numpy(Y)
            .to(device)
            .unsqueeze(0)
            .expand((len(amps), Y.shape[0], Y.shape[1]))
        )
        centers_x = (
            torch.from_numpy(centers_x)
            .to(device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(X.shape)
        )
        centers_y = (
            torch.from_numpy(centers_y)
            .to(device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(Y.shape)
        )
        amps = (
            torch.from_numpy(amps)
            .to(device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(X.shape)
        )

        I = (
            amps
            * torch.exp(-2 * ((X - centers_x) ** 2 + (Y - centers_y) ** 2) / (sigma**2))
        ).sum(dim=0)

        return I.numpy(force=True)

    def cuda_airy_2d_weighted_distance_from_centers(
        X, Y, centers_x, centers_y, first_zero_radius, amps
    ):
        k = 3.83170597 / first_zero_radius
        X = (
            torch.from_numpy(X)
            .to(device)
            .unsqueeze(0)
            .expand((len(amps), X.shape[0], X.shape[1]))
        )
        Y = (
            torch.from_numpy(Y)
            .to(device)
            .unsqueeze(0)
            .expand((len(amps), Y.shape[0], Y.shape[1]))
        )
        centers_x = (
            torch.from_numpy(centers_x)
            .to(device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(X.shape)
        )
        centers_y = (
            torch.from_numpy(centers_y)
            .to(device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(Y.shape)
        )
        amps = (
            torch.from_numpy(amps)
            .to(device)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(X.shape)
        )
        r = torch.sqrt((X - centers_x) ** 2 + (Y - centers_y) ** 2)
        u = k * r
        airy = torch.ones_like(u)
        mask = u > 1e-12
        airy[mask] = (2 * j1(u[mask]) / u[mask]) ** 2
        I = torch.sum(amps * airy, dim=0)
        return I.numpy(force=True)


    mfo.dyn_gaussian_2d_weighted_distance_from_centers = (
        cuda_gaussian_2d_weighted_distance_from_centers
    )
    mfo.dyn_airy_2d_weighted_distance_from_centers = (
        cuda_airy_2d_weighted_distance_from_centers
    )