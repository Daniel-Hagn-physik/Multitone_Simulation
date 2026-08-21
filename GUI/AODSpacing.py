import os
from matplotlib import patches
#import WaistAnalysis
from pathlib import Path
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from imageio.v2 import imread
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from matplotlib.patches import Circle
from imageio import imread
import time
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter, label, find_objects

from scipy.ndimage import rotate
from scipy.ndimage import label, center_of_mass

dirname=r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Kalibration\5MHz"
pixelsize=1.45/43.75  #IDS Sony Kamera mit 40er Mikroskopobjektiv

def multitone_frequencies(N, offset, width):
    """Erzeugt diskrete Frequenzen wie im AWG: width * n/(N-1) + offset"""
    if N <= 1:
        return np.array([offset], dtype=float)
    return width * np.arange(N) / (N - 1) + offset


def angle_from_frequency(f, offset, theta_max, f_band, centerfq=102e6):
    """
    Winkel theta aus Frequenz f:
    theta = theta_max * (f - offset) / f_band
    wobei f_band = 40 MHz theta_max entspricht.
    theta_max = 43 mrad.
    """
    return theta_max * (f - offset) / ( f_band)


def radius_from_angle(theta, f1, f2, fLO):
    """
    Räumlicher Abstand r aus Winkel theta:
    r = (f1 * fLO / f2) * tan(theta)
    """
    return  (f1 * fLO / f2) * np.tan(theta)#*1.0175

def beam_radius_scaled(f1, f2, lambda_opt, fLO, win):
    """
    Strahlradius w im Ortsraum (korrekte Formel):
    w = (f1 / f2) * (lambda_opt * fLO) / (pi * win)
    """
    return (f1 / f2) * (lambda_opt * fLO) / (np.pi * win)

offset=100e6
width = 0.5e6        # Multitone-Span, Hz
N_x = 1              # Anzahl Spots in x
N_y = 1              # Anzahl Spots in y

# Parameter: Optik + Winkel
f1 = 75e-3           # f1 Achromat im Teleskop
f2 = 100e-3          # f2 Achromat im Teleskop
#fLO = 52.88e-3       # f_LO
fLO = 100e-1
theta_max = 43e-3    # 43 mrad AA opto
f_band = 36e6        # 36 MHz entspricht theta_max, Hz
lambda_opt = 795e-9  # optische Wellenlänge
win = 1e-3        # Eingangsstrahlradius (vor Linse f1), m

fx_centers_freq = multitone_frequencies(N_x, offset, width)
fy_centers_freq = multitone_frequencies(N_y, offset, width)

# --------------------------------------------------
# Winkel und Ortsraum-Positionen für jede Frequenz
# --------------------------------------------------
centers_r_x = []
centers_r_y = []

for fx in fx_centers_freq:
    theta_x = angle_from_frequency(fx, offset, theta_max, f_band)
    #print(theta_x)
    r_x = radius_from_angle(theta_x, f1, f2, fLO)
    print(r_x)
    for fy in fy_centers_freq:
        theta_y = angle_from_frequency(fy, offset, theta_max, f_band)
        r_y = radius_from_angle(theta_y, f1, f2, fLO)
        centers_r_x.append(r_x)
        centers_r_y.append(r_y)

centers_r_x = np.array(centers_r_x)
centers_r_y = np.array(centers_r_y)
sigma_spot = beam_radius_scaled(f1, f2, lambda_opt, fLO, win)  # Strahlradius, m
scanning_range = radius_from_angle(theta_max, f1, f2, fLO)/(5.288e-6)
print('sigma_spot',sigma_spot)
print('scanning range', scanning_range)
#print(f"Center_r_x: {centers_r_x} Center_r_y: {centers_r_y}")

def gaussian_2D_double(coords, A1, x1, y1, wx1, wy1,
                       A2, x2, y2, wx2, wy2,
                       offset):
    x, y = coords
    g1 = A1 * np.exp(-2*((x-x1)**2/wx1**2 + (y-y1)**2/wy1**2))
    g2 = A2 * np.exp(-2*((x-x2)**2/wx2**2 + (y-y2)**2/wy2**2))

    return (g1 + g2 + offset).ravel()
def fitGaussian_2D_two_peaks(img, pixelsize, filename, background=None, mikro_meter=True):

    start = time.time()

    # Background subtraction
    if background is not None:
        if img.shape != background.shape:
            raise ValueError("Background und Bild müssen gleiche Dimensionen haben")

        img = img.astype(float) - background.astype(float)
        img[img < 0] = 0

    img = img.astype(float)

    # Peak detection
    neighborhood = maximum_filter(img, size=20)
    peaks = (img == neighborhood)

    labeled, num = label(peaks)
    slices = find_objects(labeled)

    if num < 2:
        raise RuntimeError("Weniger als 2 Peaks gefunden!")

    # Top 2 Peaks nach Intensität
    peak_values = []
    peak_positions = []

    for sl in slices:
        region = img[sl]
        max_idx = np.unravel_index(np.argmax(region), region.shape)

        y0, x0 = sl[0].start + max_idx[0], sl[1].start + max_idx[1]
        peak_values.append(img[y0, x0])
        peak_positions.append((x0, y0))

    # sortiere nach Intensität
    sorted_idx = np.argsort(peak_values)[::-1][:2]

    (x1, y1), (x2, y2) = [peak_positions[i] for i in sorted_idx]

    # Initial guesses
    A1, A2 = img[y1, x1], img[y2, x2]
    wx0 = wy0 = 10

    offset0 = np.min(img)

    # p0 = [A1, x1, y1, wx0, wy0,
    #       A2, x2, y2, wx0, wy0,
    #       offset0]
    p0 = [150, 2000, 400, 50, 50,
          150, 2000, 1900, 50, 50,
          10]
    bound=bound = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [256, 1e4, 1e4, 1e4, 1e4, 256, 1e4, 1e4, 1e4, 1e4, 256]]
    # -------------------------
    # Fit
    # -------------------------
    y, x = np.indices(img.shape)

    popt, pcov = curve_fit(
        gaussian_2D_double,
        (x, y),
        img.ravel(),
        p0=p0,
        bounds=bound,
        maxfev=20000
    )

    # -------------------------
    # Unpack
    # -------------------------
    (A1, x1, y1, wx1, wy1,
     A2, x2, y2, wx2, wy2,
     offset) = popt

    errors = np.sqrt(np.diag(pcov))

    # -------------------------
    # Unit conversion
    # -------------------------
    scale = pixelsize / 1000 if not mikro_meter else pixelsize
    postfix = "mm" if not mikro_meter else "um"

    wx1 *= scale
    wy1 *= scale
    wx2 *= scale
    wy2 *= scale
    print(f"wx1: {wx1} um")
    print(f"wy1: {wy1} um")
    print(f"wx2: {wx2} um")
    print(f"wy2: {wy2} um")

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="jet")

    fitted = gaussian_2D_double((x, y), *popt).reshape(img.shape)
    ax.contour(fitted, levels=8, colors="white")

    ax.scatter([x1, x2], [y1, y2], c="red", s=50)
    x1 *= scale
    y1 *= scale
    x2 *= scale
    y2 *= scale
    textstr = (
        f"Peak 1: ({x1:.4f},{y1:.4f})  wx={wx1:.2f}{postfix}\n"
        f"Peak 2: ({x2:.4f},{y2:.4f})  wx={wx2:.2f}{postfix}"
    )

    ax.text(0.05, 0.95, textstr,
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

    plt.savefig(filename[:-4] + "_doubleGaussian.png", dpi=300)
    plt.show()

    print("Fit fertig in", (time.time() - start)/60, "min")

    return (wx1, wy1, wx2, wy2, x1, y1, x2, y2)

def Double_Gaussian_eval(dirname):
    os.chdir(dirname)
    BackgroundImage = None

    for name in os.listdir(dirname):
        if "background" in name.lower() and name.lower().endswith(".bmp"):
            background_path = os.path.join(dirname, name)
            print(f"Globaler Background gefunden: {background_path}")
            BackgroundImage = np.array(
                Image.open(background_path).convert('L')
            )
            break
    for name in os.listdir(dirname):
        filename = os.path.join(dirname, name)
        # Wir akzeptieren .bmp UND .png
        if filename.lower().endswith(('.bmp')):
            if "background" in name.lower():
                continue
            print(f"Aktuelle Datei: {filename}")
            # --- Bild einlesen (Graustufen) ---
            Img = np.array(Image.open(filename).convert('L'))

            # --- Background laden (falls vorhanden) ---
            if BackgroundImage is not None:
                print("→ Verwende globalen Background")
                if Img.shape != BackgroundImage.shape:
                    raise ValueError("Background und Bild müssen gleiche Dimensionen haben")
                Img = Img.astype(float) - BackgroundImage.astype(float)
                # Optional: negative Werte vermeiden (sehr empfehlenswert)
                Img[Img < 0] = 0
            else:
                print("→ Kein Background gefunden")
            wx1, wy1, wx2, wy2, x1, y1, x2, y2= fitGaussian_2D_two_peaks(Img,1.45/43.75,filename,None,True)
            print(f"dx={np.abs(x1-x2):.2f} um")
            print(f"dy={np.abs(y1-y2):.2f} um")


def find_pattern_center(img, threshold=0.5):
    """
    Findet den Mittelpunkt eines symmetrischen Quadratmusters.

    img:
        normiertes Bild [0..1]

    threshold:
        Schwelle für Quadrate/Spots

    return:
        xc, yc, centers
    """

    # Binärbild
    mask = img > threshold

    # zusammenhängende Regionen finden
    labeled, num = label(mask)

    centers = []

    for i in range(1, num+1):

        region = labeled == i

        # zu kleine Regionen ignorieren
        if np.sum(region) < 20:
            continue

        yc, xc = center_of_mass(img, region)

        centers.append((xc,yc))

    centers = np.array(centers)

    if len(centers) == 0:
        raise RuntimeError("Keine Quadrate gefunden")

    # Mittelpunkt des Musters
    xc = np.mean(centers[:,0])
    yc = np.mean(centers[:,1])

    return xc, yc, centers

def _get_exposure(filename):
    """Extrahiert die Belichtungszeit (z.B. 5.91 aus '...5.91ms...')."""
    match = re.search(r'(\d+(?:\.\d+)?)ms', filename)
    if match is None:
        raise ValueError(f"Keine Belichtungszeit in '{filename}' gefunden.")
    return float(match.group(1))

def Construct_Image_from_List(filename_list,background_name_list):
    images=[]
    for filename, background_name in zip(filename_list,background_name_list):
        BackgroundImage = np.array(Image.open(background_name).convert('L'))

        Img = np.array(Image.open(filename).convert('L'))

        if Img.shape != BackgroundImage.shape:
            raise ValueError("Background und Bild müssen gleiche Dimensionen haben")
        Img = Img.astype(float) - BackgroundImage.astype(float)
        # Negative Werte entfernen
        Img = np.clip(Img, 0, None)

        # Auf Maximum normieren
        max_val = np.max(Img)
        if max_val > 0:
            Img = Img / max_val  # Wertebereich: 0 ... 1
        else:
            print("Warnung: Bild enthält nach Background-Abzug nur Nullen.")
        images.append(Img)
    picture = np.zeros_like(images[0])
    for image in images:
        picture = picture+image.astype(float)
    return picture

def Construct_Image_from_File(image_dir, background_dir):
    image_dir = Path(image_dir)
    background_dir = Path(background_dir)

    # Alle Background-Bilder einlesen
    background_dict = {}

    for bg in background_dir.glob("*.bmp"):
        exposure = _get_exposure(bg.name)

        if exposure in background_dict:
            raise ValueError(f"Mehrere Backgrounds mit {exposure} ms gefunden.")

        background_dict[exposure] = bg

    images = []

    # Alle Messbilder einlesen
    for filename in sorted(image_dir.glob("*.bmp")):

        # Background-Dateien überspringen
        if "background" in filename.name.lower():
            continue

        exposure = _get_exposure(filename.name)

        if exposure not in background_dict:
            raise ValueError(
                f"Kein Background für {filename.name} ({exposure} ms) gefunden."
            )

        background_name = background_dict[exposure]

        BackgroundImage = np.array(Image.open(background_name).convert("L"))
        Img = np.array(Image.open(filename).convert("L"))

        if Img.shape != BackgroundImage.shape:
            raise ValueError(
                f"{filename.name}: Bild und Background haben unterschiedliche Größe."
            )

        Img = Img.astype(float) - BackgroundImage.astype(float)
        Img = np.clip(Img, 0, None)

        max_val = Img.max()
        if max_val > 0:
            Img /= max_val
        else:
            print(f"Warnung: {filename.name} enthält nach Backgroundabzug nur Nullen.")

        images.append(Img)

    if not images:
        raise ValueError("Keine Messbilder gefunden.")

    picture = np.sum(images, axis=0)

    return picture

def plot_image(Img,xcenter=None,ycenter=None):
    width_um=5.288
    pixel_size=1.45/43.75
    width_px=width_um/pixel_size
    half = width_px / 2

    if xcenter and ycenter is not None:
        xc=xcenter
        yc=ycenter
        print("Mittelpunkt:", xc, yc)

    else:
        xc, yc, centers = find_pattern_center(Img,threshold=0.2)
        print("Mittelpunkt:", xc, yc)
        print("Mittelpunkte:", centers)
    x_min = int(np.round(xc - half))
    y_min = int(np.round(yc - half))



    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(Img, origin="lower", cmap="viridis", vmin=0, vmax=np.max(Img))
    plt.colorbar(im, ax=ax, label="normierte Intensität")

    # Mittelpunkt einzeichnen
    ax.plot(xc,yc,marker="+",markersize=15,markeredgewidth=2,color="red",label=f"Mittelpunkt ({xc:.1f}, {yc:.1f})")
    rect = patches.Rectangle(
        (x_min, y_min),
        width_px,
        width_px,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label=f"{width_um} µm × {width_um} µm"
    )
    ax.add_patch(rect)
    #ax.scatter(centers[:,0],centers[:,1],color="red",marker="x",s=80,label="Quadrat Zentren")
    ax.set_xlabel("Pixel x")
    ax.set_ylabel("Pixel y")
    ax.set_title("Background entfernt und normiert")

    ax.legend(labelcolor="white")

    plt.show()
    return im


def sum_rectangle_around_center(img, xc, yc, width_um, pixel_size):
    """
    Summiert die Intensität innerhalb eines Quadrats um (xc,yc).
    """
    # Breite in Pixel
    width_px = width_um / pixel_size

    half = width_px / 2

    # Grenzen
    x_min = int(np.round(xc - half))
    x_max = int(np.round(xc + half))

    y_min = int(np.round(yc - half))
    y_max = int(np.round(yc + half))

    # Bildgrenzen beachten
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)

    x_max = min(x_max, img.shape[1])
    y_max = min(y_max, img.shape[0])

    roi = img[y_min:y_max, x_min:x_max]

    intensity_sum = np.sum(roi)

    return intensity_sum

def Crosstalk(Img_center,Img_neighbor,CenterImage=None,width=5.288,pixel_size=1.45/43.75):
    xc1, yc1, centers1 = find_pattern_center(Img_center,threshold=0.2)
    if CenterImage is None:
        xc2, yc2, centers2 = find_pattern_center(Img_neighbor,threshold=0.2)
    else:
        xc2, yc2, centers2 = find_pattern_center(CenterImage,threshold=0.2)
        plot_image(CenterImage)
    P_center = sum_rectangle_around_center(Img_center,xc1,yc1,width,pixel_size)
    print(P_center)
    P_neighbor = sum_rectangle_around_center(Img_neighbor, xc2, yc2, width, pixel_size)
    print(P_neighbor)
    eta=P_neighbor / (P_center)
    print(f"eta={eta:.2}")


    plot_image(Img_neighbor, xc2, yc2)
    plot_image(Img_center)
    return eta

pixel_size = 1.45 / 43.75   # µm/Pixel
width_um = 5.288

#1x2 Eckpunkte 0.3MHz
# Imagelist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\2_Nachbarn\3,4_0.3MHz_3.01ms_1x2_Eckpunkte_UntenLinks_23.bmp"]
# Backgroundlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\2_Nachbarn\Background_3.01ms_25.bmp"]
# RelativePositionlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\2_Nachbarn\3,4_0.3MHz_25.00ms_2x3-Referenz_22.bmp"]
# RelativePositionBackground=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\2_Nachbarn\Background_25.00ms_26.bmp"]
# centerpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\2_Nachbarn\3,4_0.3MHz_5.91ms_1x1_5.bmp"]
# backgroundpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\2_Nachbarn\Background_5.91ms_6.bmp"]

#1x2 Eckpunkte 0.35MHz
# Imagelist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\2_Nachbarn\3,4_0.35MHz_4.00ms_1x2_Eckpunkte_UntenLinks_29.bmp"]
# Backgroundlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\2_Nachbarn\Background_4.00ms_31.bmp"]
# RelativePositionlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\2_Nachbarn\3,4_0.35MHz_35.00ms_2x3_Referenz_27.bmp"]
# RelativePositionBackground=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\2_Nachbarn\Background_35.00ms_30.bmp"]
# centerpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\3x3_Grid\3,4_100MHz_8.32ms_1x1_8.bmp"]
# backgroundpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\3x3_Grid\Background_8.32ms_7.bmp"]


#3x3 Raster 0.3MHz
# Neighbor_dirname=r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\3x3_Grid\Nachbarn"
# Neighbor_Background_dirname=r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\3x3_Grid\Nachbarn_Background"
# centerpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\2_Nachbarn\3,4_0.3MHz_5.91ms_1x1_5.bmp"]
# backgroundpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\2_Nachbarn\Background_5.91ms_6.bmp"]

#3x3 Raster 0.35MHz
# Neighbor_dirname=r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\3x3_Grid\Nachbarn"
# Neighbor_Background_dirname=r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\3x3_Grid\Nachbarn_Background"
# centerpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\3x3_Grid\3,4_100MHz_8.32ms_1x1_8.bmp"]
# backgroundpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\3x3_Grid\Background_8.32ms_7.bmp"]

# Diagonaler Eckpunkt 0.3MHz
# Imagelist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\1_Nachbar\Diagonal\3,4_0.3MHz_5.49ms_(1,1)_34.bmp"]
# Backgroundlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\1_Nachbar\Diagonal\Background_5.49ms_36.bmp"]
# RelativePositionlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\1_Nachbar\Diagonal\3,4_0.3MHz_151ms_(-1,0,1)(-1,1)_37.bmp"]
# RelativePositionBackground=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\1_Nachbar\Diagonal\Background_151ms_38.bmp"]
# centerpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\1_Nachbar\Vertikal\3,4_0.3MHz_5.49ms_(0,0)_39.bmp"]
# backgroundpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\1_Nachbar\Diagonal\Background_5.49ms_36.bmp"]

# Vertikaler Eckpunkt 0.3MHz
# Imagelist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\1_Nachbar\Vertikal\3,4_0.3MHz_5.49ms_(0,1)_35.bmp"]
# Backgroundlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.3MHz\1_Nachbar\Diagonal\Background_5.49ms_36.bmp"]


# Diagonaler Eckpunkt 0.35MHz
#Imagelist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\1_Nachbar\Diagonal\3,4_0.35MHz_7.00ms_(1,1)_42.bmp"]
Backgroundlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\1_Nachbar\Diagonal\Background_7.00ms_43.bmp"]
RelativePositionlist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\1_Nachbar\Diagonal\3,4_0.35MHz_182.01ms_(-1,0,1)(-1,1)_44.bmp"]
RelativePositionBackground=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\1_Nachbar\Diagonal\Background_182.01ms_45.bmp"]
centerpath=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\1_Nachbar\Diagonal\3,4_0.35MHz_7.00ms_(0,0)_41.bmp"]
backgroundpath=Backgroundlist

# Vertikaler Eckpunkt 0.35MHz
Imagelist=[r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\Kameraaufnahmen\Testaufbau_lokalerRamann\Multitone\Überlapp\100MHz_0.35MHz\1_Nachbar\Vertikal\3,4_0.35MHz_7.00ms_(0,1)_46.bmp"]

#
# Img_norm_center = Construct_Image_from_List(centerpath,backgroundpath)
# Img_norm_neighbor = Construct_Image_from_List(Imagelist,Backgroundlist)
# Img_norm_position = Construct_Image_from_List(RelativePositionlist,RelativePositionBackground)
#
# Crosstalk(Img_norm_center,Img_norm_neighbor,Img_norm_position,width=width_um)
# def spacing_angle(Delta_f):
#     return 795*1e-9/(2*650)*Delta_f
# def spacing(Delta_Theta,f1=0.045,f2=0.3,L=0.565):
#     return (L-f2)*f1/f2*Delta_Theta
# print(spacing(spacing_angle(3*1e6)))
#Double_Gaussian_eval(dirname)


