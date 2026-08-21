import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from matplotlib.path import Path
from pathlib import Path as FilePath
from scipy.special import j0
from scipy.special import j1
from matplotlib.patches import Rectangle

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


def gaussian_2d_distance_from_centers(X, Y, centers_x, centers_y, sigma):
    """
    2D Intensität als Summe von Gauß-Profilen mit gleicher Breite sigma:
    I = sum_i exp( -((X-xi)^2 + (Y-yi)^2) / (2*sigma^2) )
    """
    I = np.zeros_like(X, dtype=float)

    for cx, cy in zip(centers_x, centers_y):
        I += np.exp(-2*((X - cx)**2 + (Y - cy)**2) / (sigma**2))

    return I


def gaussian_2d_weighted_distance_from_centers(X, Y, centers_x, centers_y, sigma, amps):
    """Wie gaussian_2d_distance_from_centers, aber mit individueller Amplitude pro Spot"""
    I = np.zeros_like(X, dtype=float)
    for cx, cy, a in zip(centers_x, centers_y, amps):
        I += a * np.exp(-2 * ((X - cx)**2 + (Y - cy)**2) / (sigma**2))
    return I


def bessel_2d_distance_from_centers(X, Y, centers_x, centers_y, width):
    """
    Summe von radialsymmetrischen Bessel-Profilen:

    I = sum_i J0(r_i / width)

    width bestimmt die radiale Skala der Oszillationen.
    """
    I = np.zeros_like(X, dtype=float)

    for cx, cy in zip(centers_x, centers_y):
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        I += j0(r / width)
    return I
def airy_2d_distance_from_centers(X, Y, centers_x, centers_y, first_zero_radius):

    I = np.zeros_like(X, dtype=float)

    k = 3.83170597 / first_zero_radius

    for cx, cy in zip(centers_x, centers_y):

        r = np.sqrt((X-cx)**2 + (Y-cy)**2)

        u = k * r

        airy = np.ones_like(u)
        mask = u > 1e-12
        airy[mask] = (2*j1(u[mask])/u[mask])**2

        I += airy

    return I
def overlap_mask_circular_3sigma(X, Y, center_x, center_y, radius):
    R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    return R <= radius

def overlap_mask_pitch(X, Y, center_x, center_y, side_length):
    half_side = side_length / 2
    return (
        (np.abs(X - center_x) <= half_side) &
        (np.abs(Y - center_y) <= half_side)
    )

def create_neighbourhood(X, Y, pitch, centers_x, centers_y, w_in, amps=None):
    if amps is None:
        amps = np.ones(len(centers_x))
    I_neighbor = np.zeros_like(X)
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            if ix == 0 and iy == 0:
                continue
            shifted_x = centers_x + ix * pitch
            shifted_y = centers_y + iy * pitch
            I_spot = gaussian_2d_weighted_distance_from_centers(X, Y, shifted_x, shifted_y, w_in, amps)
            I_spot = I_spot / np.max(I_spot)
            I_neighbor += I_spot
    return I_neighbor


# Verzeichnis zum Speichern aller Plots
out_dir = FilePath(r"\\brain43\public\__Transfer__\DHagn\LokalerRaman_Master\PythonCode\Multitone_FlatTop")
# --------------------------------------------------
# Parameter: Frequenzraum
# --------------------------------------------------
offset=100e6
width = 0.3e6        # Multitone-Span, Hz
N_x = 3             # Anzahl Spots in x
N_y = 4              # Anzahl Spots in y

# Parameter: Optik + Winkel
f1 = 45e-3           # f1 Achromat im Teleskop
f2 = 300e-3          # f2 Achromat im Teleskop
fLO = 52.88e-3       # f_LO
theta_max = 43e-3    # 43 mrad AA opto
f_band = 36e6        # 36 MHz entspricht theta_max, Hz
lambda_opt = 795e-9  # optische Wellenlänge
#win = 1.75e-3        # Eingangsstrahlradius (vor Linse f1),
win = 1.6e-6
pitch = 5.288e-6     # Atomabstand
uniformity_side_length = 2.6e-6  # Kantenlänge des Uniformity-Bereichs (kleiner als pitch), initial gesetzt
integration_radius = 0.45e-6 # 3Sigma Intervall für Aufenthaltsort des Atoms
threshindex = 2      # Setzt den Index für das Level, das verwendet werden soll
levels = [np.exp(-2), 0.5, 0.9 ]         # Entweder 90% Breite, 0.5 FWHM oder 1/e^2
useconvexhull = False# Statt Threshold Convex Hull benutzen
use_Levels = False


drawpoints=True #Bei vielen Gaußstrahlen wird der Plot nur aus roten Punkten bestehen
N=N_x+N_y
if N>=16:
    drawpoints=False
# --------------------------------------------------
# Frequenzpositionen der Spots im Frequenzraum
# --------------------------------------------------
f_center = offset + width/2
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
    #print(r_x)
    for fy in fy_centers_freq:
        theta_y = angle_from_frequency(fy, offset, theta_max, f_band)
        r_y = radius_from_angle(theta_y, f1, f2, fLO)
        centers_r_x.append(r_x)
        centers_r_y.append(r_y)

theta_center = angle_from_frequency(f_center,offset,theta_max, f_band)
r_center = radius_from_angle(theta_center,f1,f2,fLO)

centers_r_x = np.array(centers_r_x)
centers_r_y = np.array(centers_r_y)
print(f"Center_r_x: {centers_r_x[-1]} Center_r_y: {centers_r_y[-1]}")
# --------------------------------------------------
# Strahldurchmesser im Ortsraum (gleiche Breite für alle Spots)
# --------------------------------------------------
#sigma_spot = 1.1*np.sqrt(2)*beam_radius_scaled(f1, f2, lambda_opt, fLO, win)  # Strahlradius,
sigma_spot = win
first_zero_radius = 1.19 * sigma_spot
print('sigma_spot',sigma_spot)

# --------------------------------------------------
# Gitter im Ortsraum (meter)
# --------------------------------------------------
margin = 10 * sigma_spot
x_range = np.max(np.abs(centers_r_x)) + margin
y_range = np.max(np.abs(centers_r_y)) + margin

# x = np.linspace(np.min(np.abs(centers_r_x)) - margin, np.max(np.abs(centers_r_x)) + margin, 500)
# y = np.linspace(np.min(np.abs(centers_r_y)) - margin, np.max(np.abs(centers_r_y)) + margin, 500)


x = np.linspace( - margin/2, np.max(np.abs(centers_r_x)) + margin/2, 1000)
y = np.linspace( - margin/2, np.max(np.abs(centers_r_y)) +margin/2, 1000)

X, Y = np.meshgrid(x, y)

print(centers_r_x, centers_r_y)
# --------------------------------------------------
# Intensität im Ortsraum
# --------------------------------------------------
I_ort_airy = airy_2d_distance_from_centers(X, Y,centers_r_x,centers_r_y,sigma_spot)
I_ort = gaussian_2d_distance_from_centers(X, Y,centers_r_x, centers_r_y,sigma=sigma_spot)

I_neighbor = create_neighbourhood(X, Y, pitch, centers_r_x, centers_r_y, w_in=sigma_spot)# Teil-Profile bereits normiert in Funktionsdefinition

I_ort /= np.max(I_ort) # Normierung auf 1

I = I_ort  # kurz Alias
I_list = []
# --------------------------------------------------
# Maske erstellen
# --------------------------------------------------

# --------------------------------------------------
# Uniformity: Standardabweichung / Mittelwert
# Fläche: Konvexe Hülle der Spot-Zentren
# --------------------------------------------------
# Erstelle ConvexHull der Spot-Zentren

if useconvexhull:
    points = np.column_stack([centers_r_x, centers_r_y])
    hull = ConvexHull(points)
    hull_vertices = hull.points[hull.vertices]

    # Erstelle Path-Polygon aus der ConvexHull
    polygon_path = Path(hull_vertices)

    # Erstelle Maske: welche Pixel liegen innerhalb des Polygons?
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    inside_polygon = polygon_path.contains_points(grid_points).reshape(X.shape)
    mask_uniformity = inside_polygon
    mask_crosstalk = overlap_mask_pitch(X, Y, r_center, r_center, pitch)

elif use_Levels:
    mask_uniformity = I_ort >= levels[threshindex]
    mask_crosstalk = overlap_mask_pitch(X, Y, r_center, r_center, pitch)
else:
    mask_uniformity = overlap_mask_pitch(X, Y, r_center, r_center, uniformity_side_length)
    mask_crosstalk = overlap_mask_pitch(X, Y, r_center, r_center, pitch)

# Nur alles innerhalb der jeweiligen zentralen Integrationsstruktur ist relevant
I_inside_uni = I_ort[mask_uniformity]

I_inside_cross = I_ort[mask_crosstalk]
I_neighbor_calc = I_neighbor[mask_crosstalk]

# Berechne Uniformity
mean_intensity = np.mean(I_inside_uni)
std_intensity = np.std(I_inside_uni)
uniformity = std_intensity / mean_intensity if mean_intensity > 0 else 0

print(f"Mittlere Intensität in der Region: {mean_intensity:.4f}")
print(f"Standardabweichung: {std_intensity:.4f}")
print(f"Uniformity (σ/μ): {uniformity:.4f} ({uniformity*100:.2f}%)")

#Berechne Überlapp
P_ort = np.sum(I_inside_cross)
P_neighbor = np.sum(I_neighbor_calc)
print(P_ort, P_neighbor)
eta=P_neighbor/P_ort
print(f"Überlapp der Nachbarn in die Zentralstruktur: {eta*100:.3f}%")

# --------------------------------------------------
# Optimierung: Finde win, das Uniformity und Overlapp minimiert innerhalb vonprint Quadrat um Zentrum mit Pitch als Kantenlänge
# --------------------------------------------------
def calculate_uniformity(win_value, return_details=False):
    """Berechnet die Uniformity für einen gegebenen win-Wert"""
    # Berechne neuen Strahlradius
    win_0=win_value[0]
    #sigma = beam_radius_scaled(f1, f2, lambda_opt, fLO, win_0)
    sigma=win_0
    # Berechne neue Intensität im Ortsraum
    I_new = gaussian_2d_distance_from_centers(X, Y, centers_r_x, centers_r_y, sigma=sigma)
    I_new /= np.max(I_new) #Normierung auf 1

    # Intensität nur innerhalb des Polygons (ConvexHull)
    if useconvexhull:
        mask=inside_polygon
        I_inside_new = I_new[mask]
    if use_Levels:
        mask = I_new >= levels[threshindex]
        I_inside_new = I_new[mask]
    else:
        mask = overlap_mask_pitch(X, Y, r_center, r_center, uniformity_side_length)
        I_inside_new = I_new[mask]

    if len(I_inside_new) == 0 or np.mean(I_inside_new) == 0:
        return 1e10  # Penalty für ungültige Fälle
    # if win_value<0 or win_value>3.5e-3:
    #     return 1e10   # Verhindert in Optimierung mit Nelder-Mead ungültige Eingaben, da Bounds irgnoriert würden sonst
    # Berechne Uniformity
    uniformity_new = np.std(I_inside_new) / np.mean(I_inside_new)
    if return_details:
        return uniformity_new, I_new, mask
    return uniformity_new


def calculate_overlapp(win_value,return_details=False):
    """Berechnet die Uniformity für einen gegebenen win-Wert"""
    # Berechne neuen Strahlradius
    win_0=win_value[0]
    # sigma = beam_radius_scaled(f1, f2, lambda_opt, fLO, win_0)
    sigma = win_0

    # Berechne neue zentrale Intensität im Ortsraum
    I_new = gaussian_2d_distance_from_centers(X, Y, centers_r_x, centers_r_y, sigma=sigma)
    I_new /= np.max(I_new)  # Normierung auf 1

    #Berechne die 8 Nachbarn jeweils um den Pitch verschoben
    I_neighbor_new=create_neighbourhood(X,Y,pitch,centers_r_x,centers_r_y,w_in=sigma)

    #Nur alles innerhalb des zentralen Integrationsradius ist relevant
    mask=overlap_mask_pitch(X,Y,r_center,r_center,pitch)

    I_inside_new = I_new[mask]
    I_inside_neighbor_new = I_neighbor_new[mask]

    P_inside_new = np.sum(I_inside_new)
    P_neighbor_new = np.sum(I_inside_neighbor_new)

    if len(I_inside_new) == 0 or np.mean(I_inside_new) == 0:
        return 1e10  # Penalty für ungültige Fälle
    if win_0<0 or win_0>3.5e-3:
        return 1e10   # Verhindert in Optimierung mit Nelder-Mead ungültige Eingaben, da Bounds irgnoriert würden sonst

    # Berechne Überlapp
    eta = P_neighbor_new / P_inside_new
    if return_details:
        return eta, I_neighbor_new, mask
    return eta

def objective(win_value, alpha=0.7):
    u = calculate_uniformity(win_value)
    o = calculate_overlapp(win_value)

    return alpha * u + (1 - alpha) * o
# Starte Optimierung mit Startwert der aktuellen win
print("\n" + "="*60)
print(f"Starte Optimierung von win zur Minimierung der Uniformity und des Crosstalks...")
print("="*60)

result = minimize(
    objective,
    x0=[win],
    method='Nelder-Mead',
    options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 1000},
    bounds=[(0, 3.5e-6)]
)
# result = minimize(
#     calculate_uniformity,
#     x0=[win],
#     method='Nelder-Mead',
#     options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 1000},
#     bounds=[(0,3.5e-6)]
# )

win_optimal = result.x[0]
uniformity_optimal, I_ort_optimal,mask_optimal  = calculate_uniformity([win_optimal], return_details=True)
overlap_optimal,I_neighbor_optimal,_ = calculate_overlapp([win_optimal], return_details=True)

print(f"\nOptimales win: {win_optimal:.6e} m")
print(f"Optimale Uniformity: {uniformity_optimal:.4f} ({uniformity_optimal*100:.2f}%)")
print(f"Optimaler Overlapp: {overlap_optimal:.4f} ({overlap_optimal*100:.2f}%)")
print(f"Verbesserung: {((uniformity - uniformity_optimal) / uniformity * 100):.2f}%")
print("="*60)

# --------------------------------------------------
# Plot der Original-Ergebnisse + Speichern
# --------------------------------------------------
fig1 = plt.figure(figsize=(12, 7))
fig1.suptitle(f"Original: win = {win:.3e} m, Uniformity = {uniformity*100:.2f}%", fontsize=14, fontweight='bold')

# Colorplot links
ax_2d = plt.subplot(1, 2, 1)
im = ax_2d.imshow(
    I_ort,
    origin="lower",
    extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
    aspect="equal",
    cmap="viridis"
)
if drawpoints:
    ax_2d.scatter(
        centers_r_x * 1e6,
        centers_r_y * 1e6,
        c="red",
        s=20,
        label="Spot-Zentren"
    )

# Zeichne die ConvexHull-Region
if useconvexhull:
    hull_vertices_closed = np.vstack([hull_vertices, hull_vertices[0]])
    ax_2d.plot(
        hull_vertices_closed[:, 0] * 1e6,
        hull_vertices_closed[:, 1] * 1e6,
        'b--',
        linewidth=2,
        label="Uniformity-Region"
    )
elif use_Levels:
    cs = ax_2d.contour(
        X * 1e6,
        Y * 1e6,
        I_ort,
        levels=levels,
        colors=['blue', 'red', 'green'],
        linewidths=2
    )
    ax_2d.clabel(cs, inline=True, fontsize=8)
else:
    half_side = pitch / 2
    rect = Rectangle(
        ((r_center - half_side) * 1e6, (r_center - half_side) * 1e6),
        pitch * 1e6, pitch * 1e6,
        edgecolor="red", facecolor="none", linewidth=2, label="Crosstalk-Bereich"
    )
    ax_2d.add_patch(rect)

    half_side_uniform = uniformity_side_length / 2
    rect_uniform = Rectangle(
        ((r_center - half_side_uniform) * 1e6, (r_center - half_side_uniform) * 1e6),
        uniformity_side_length * 1e6, uniformity_side_length * 1e6,
        edgecolor="cyan", facecolor="none", linewidth=2, label="Uniformity-Bereich"
    )
    ax_2d.add_patch(rect_uniform)

    ax_2d.plot(
        r_center * 1e6, r_center * 1e6,
        "r+", markersize=12, markeredgewidth=2, label="Mittelpunkt"
    )

ax_2d.set_xlabel("Ort $x$ (µm)")
ax_2d.set_ylabel("Ort $y$ (µm)")
ax_2d.legend()
plt.colorbar(im, ax=ax_2d, label="Normierte Intensität")

# Schnitte rechts
mid_y_idx = len(y) // 2
mid_x_idx = len(x) // 2

ax_x = plt.subplot(2, 2, 2)
ax_x.plot(x * 1e6, I_ort[mid_y_idx, :], 'b-', linewidth=2)
ax_x.set_xlabel("Ort $x$ (µm)")
ax_x.set_ylabel("Intensität")
ax_x.set_title("Schnitt entlang x durch Zentrum")
ax_x.grid(True, alpha=0.3)

ax_y = plt.subplot(2, 2, 4)
ax_y.plot(y * 1e6, I_ort[:, mid_x_idx], 'g-', linewidth=2)
ax_y.set_xlabel("Ort $y$ (µm)")
ax_y.set_ylabel("Intensität")
ax_y.set_title("Schnitt entlang y durch Zentrum")
ax_y.grid(True, alpha=0.3)

plt.tight_layout()
out_file_original = out_dir / f"FlatMultiTone_Original_{offset*1e-6,width*1e-6,N_x,N_y}.png"
plt.savefig(out_file_original, dpi=150, bbox_inches='tight')
print("\nGrafik gespeichert: FlatMultiTone_Original.png")
plt.show()

# --------------------------------------------------
# Plot: Nachbarregionen (Crosstalk) für Original
# --------------------------------------------------
fig1b = plt.figure(figsize=(8, 7))
fig1b.suptitle(f"Nachbarregionen (Original): Crosstalk = {eta*100:.3f}%", fontsize=14, fontweight='bold')

ax_neighbor_orig = plt.subplot(1, 1, 1)
im_neighbor_orig = ax_neighbor_orig.imshow(
    I_neighbor,
    origin="lower",
    extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
    aspect="equal",
    cmap="viridis"
)
if drawpoints:
    ax_neighbor_orig.scatter(
        centers_r_x * 1e6,
        centers_r_y * 1e6,
        c="red",
        s=20,
        label="Spot-Zentren"
    )

half_side = pitch / 2
rect_orig = Rectangle(
    ((r_center - half_side) * 1e6, (r_center - half_side) * 1e6),
    pitch * 1e6,
    pitch * 1e6,
    edgecolor="red",
    facecolor="none",
    linewidth=2,
    label="Maske"
)
ax_neighbor_orig.add_patch(rect_orig)

ax_neighbor_orig.plot(
    r_center * 1e6,
    r_center * 1e6,
    "r+",
    markersize=12,
    markeredgewidth=2,
    label="Mittelpunkt"
)

plt.colorbar(im_neighbor_orig, ax=ax_neighbor_orig, label="Intensität (Nachbarn)")
ax_neighbor_orig.set_xlabel("x [µm]")
ax_neighbor_orig.set_ylabel("y [µm]")
ax_neighbor_orig.legend()
plt.tight_layout()

out_file_neighborOriginal = out_dir / f"FlatMultiTone_NeighborOriginal_{offset*1e-6,width*1e-6,N_x,N_y}.png"
plt.savefig(out_file_neighborOriginal, dpi=150, bbox_inches='tight')
print("Grafik gespeichert: FlatMultiTone_NeighborOriginal.png")
plt.show()

# --------------------------------------------------
# Plot der Optimierten-Ergebnisse + Speichern
# --------------------------------------------------
fig2 = plt.figure(figsize=(12, 7))
fig2.suptitle(f"Optimiert: win = {win_optimal:.3e} m, Uniformity = {uniformity_optimal*100:.2f}%", fontsize=14, fontweight='bold')

ax_2d_opt = plt.subplot(1, 2, 1)
im_opt = ax_2d_opt.imshow(
    I_ort_optimal,
    origin="lower",
    extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
    aspect="equal",
    cmap="viridis"
)
if drawpoints:
    ax_2d_opt.scatter(
        centers_r_x * 1e6,
        centers_r_y * 1e6,
        c="red",
        s=20,
        label="Spot-Zentren"
    )

if useconvexhull:
    ax_2d_opt.plot(
        hull_vertices_closed[:, 0] * 1e6,
        hull_vertices_closed[:, 1] * 1e6,
        'b--',
        linewidth=2,
        label="Uniformity-Region"
    )
elif use_Levels:
    cs = ax_2d_opt.contour(
        X * 1e6,
        Y * 1e6,
        I_ort_optimal,
        levels=[levels[0], levels[1], levels[2]],
        colors=['blue', 'red', 'green'],
        linewidths=2
    )
    ax_2d_opt.clabel(cs, inline=True, fontsize=8)
else:
    half_side_win = pitch / 2
    rect_opt = Rectangle(
        ((r_center - half_side_win) * 1e6, (r_center - half_side_win) * 1e6),
        pitch * 1e6, pitch * 1e6,
        edgecolor="red", facecolor="none", linewidth=2, label="Crosstalk-Bereich"
    )
    ax_2d_opt.add_patch(rect_opt)

    half_side_uniform_win = uniformity_side_length / 2
    rect_uniform_win = Rectangle(
        ((r_center - half_side_uniform_win) * 1e6, (r_center - half_side_uniform_win) * 1e6),
        uniformity_side_length * 1e6, uniformity_side_length * 1e6,
        edgecolor="cyan", facecolor="none", linewidth=2, label="Uniformity-Bereich"
    )
    ax_2d_opt.add_patch(rect_uniform_win)

    ax_2d_opt.plot(
        r_center * 1e6,
        r_center * 1e6,
        "r+",
        markersize=12,
        markeredgewidth=2,
        label="Mittelpunkt"
    )
ax_2d_opt.set_xlabel("Ort $x$ (µm)")
ax_2d_opt.set_ylabel("Ort $y$ (µm)")
ax_2d_opt.set_xlim(-5, 10)  # µm
ax_2d_opt.set_ylim(-5, 10)  # µm
ax_2d_opt.legend()
plt.colorbar(im_opt, ax=ax_2d_opt, label="Normierte Intensität")

ax_x_opt = plt.subplot(2, 2, 2)
ax_x_opt.plot(x * 1e6, I_ort_optimal[mid_y_idx, :], 'b-', linewidth=2)
ax_x_opt.set_xlabel("Ort $x$ (µm)")
ax_x_opt.set_ylabel("Intensität")
ax_x_opt.set_title("Schnitt entlang x durch Zentrum")
ax_x_opt.grid(True, alpha=0.3)

ax_y_opt = plt.subplot(2, 2, 4)
ax_y_opt.plot(y * 1e6, I_ort_optimal[:, mid_x_idx], 'g-', linewidth=2)
ax_y_opt.set_xlabel("Ort $y$ (µm)")
ax_y_opt.set_ylabel("Intensität")
ax_y_opt.set_title("Schnitt entlang y durch Zentrum")
ax_y_opt.grid(True, alpha=0.3)

plt.tight_layout()
out_file_optimized = out_dir / f"FlatMultiTone_Optimized_{offset*1e-6,width*1e-6,N_x,N_y}.png"
plt.savefig(out_file_optimized, dpi=150, bbox_inches='tight')
print("Grafik gespeichert: FlatMultiTone_Optimized.png")
plt.show()

# --------------------------------------------------
# Plot: Nachbarregionen (Crosstalk) für win-optimiert
# --------------------------------------------------
fig2b = plt.figure(figsize=(8, 7))
fig2b.suptitle(f"Nachbarregionen (win-optimiert): Crosstalk = {overlap_optimal*100:.3f}%", fontsize=14, fontweight='bold')

ax_neighbor_win = plt.subplot(1, 1, 1)
im_neighbor_win = ax_neighbor_win.imshow(
    I_neighbor_optimal,
    origin="lower",
    extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
    aspect="equal",
    cmap="viridis"
)
if drawpoints:
    ax_neighbor_win.scatter(
        centers_r_x * 1e6,
        centers_r_y * 1e6,
        c="red",
        s=20,
        label="Spot-Zentren"
    )

half_side_win = pitch / 2
rect_win = Rectangle(
    ((r_center - half_side_win) * 1e6, (r_center - half_side_win) * 1e6),
    pitch * 1e6,
    pitch * 1e6,
    edgecolor="red",
    facecolor="none",
    linewidth=2,
    label="Maske"
)
ax_neighbor_win.add_patch(rect_win)

ax_neighbor_win.plot(
    r_center * 1e6,
    r_center * 1e6,
    "r+",
    markersize=12,
    markeredgewidth=2,
    label="Mittelpunkt"
)

plt.colorbar(im_neighbor_win, ax=ax_neighbor_win, label="Intensität (Nachbarn)")
ax_neighbor_win.set_xlabel("x [µm]")
ax_neighbor_win.set_ylabel("y [µm]")
ax_neighbor_win.legend()
plt.tight_layout()

out_file_neighborWinOptimized = out_dir / f"FlatMultiTone_NeighborWinOptimized_{offset*1e-6,width*1e-6,N_x,N_y}.png"
plt.savefig(out_file_neighborWinOptimized, dpi=150, bbox_inches='tight')
print("\nGrafik gespeichert: FlatMultiTone_NeighborWinOptimized.png")
plt.show()

# # ==================================================
# # ZWEITE ITERATION: Optimiere Frequenzabstände (width)
# # ==================================================
print("\n" + "="*60)
print("ZWEITE ITERATION: Optimierung der Frequenzabstände (width)")
print("win bleibt auf optimiertem Wert: {:.3e}".format(win_optimal))
print("="*60)

def calculate_uniformity_width(width_value, return_details=False):
    """
    Berechnet die Uniformity für einen gegebenen width-Wert
    (Frequenzabstände der Spots)
    """
    width_0 = width_value[0] if hasattr(width_value, '__len__') else width_value
    # Berechne neue Frequenzpositionen sowie Zentrum
    fx_freq_new = multitone_frequencies(N_x, offset, width_0)
    fy_freq_new = multitone_frequencies(N_y, offset, width_0)

    f_center_new = offset + width_0 / 2
    theta_center_new = angle_from_frequency(f_center_new, offset, theta_max, f_band)
    r_center_new = radius_from_angle(theta_center_new, f1, f2, fLO)

    # Berechne neue Ortsraum-Positionen
    centers_r_x_new = []
    centers_r_y_new = []
    
    for fx in fx_freq_new:
        theta_x = angle_from_frequency(fx, offset, theta_max, f_band)
        r_x = radius_from_angle(theta_x, f1, f2, fLO)
        for fy in fy_freq_new:
            theta_y = angle_from_frequency(fy, offset, theta_max, f_band)
            r_y = radius_from_angle(theta_y, f1, f2, fLO)
            centers_r_x_new.append(r_x)
            centers_r_y_new.append(r_y)
    
    centers_r_x_new = np.array(centers_r_x_new)
    centers_r_y_new = np.array(centers_r_y_new)
    
    # Berechne neue Intensität mit originalem win und neuem sigma_spot
    #sigma_new = beam_radius_scaled(f1, f2, lambda_opt, fLO, win)
    sigma_new = win_optimal
    I_new = gaussian_2d_distance_from_centers(X, Y, centers_r_x_new, centers_r_y_new, sigma=sigma_new)
    I_new /= np.max(I_new)
    
    # Erstelle ConvexHull der neuen Spot-Zentren jedes Mal neu
    if useconvexhull:
        try:
            points_new = np.column_stack([centers_r_x_new, centers_r_y_new])
            hull_new = ConvexHull(points_new)
            hull_vertices_new = hull_new.points[hull_new.vertices]
            polygon_path_new = Path(hull_vertices_new)

            # Erstelle Maske: welche Pixel liegen innerhalb des neuen Polygons?
            grid_points = np.column_stack([X.ravel(), Y.ravel()])
            mask = polygon_path_new.contains_points(grid_points).reshape(X.shape)

        except Exception:
            if return_details:
                return 1e10, I_new, None, centers_r_x_new, centers_r_y_new, r_center_new
            return 1e10  # Penalty bei Fehler (z.B. weniger als 3 Punkte)

    elif use_Levels:
        mask = I_new >= levels[threshindex]

    else:
        mask = overlap_mask_pitch(X, Y, r_center_new, r_center_new, uniformity_side_length)

    I_inside_new = I_new[mask]
    if len(I_inside_new) == 0 or np.mean(I_inside_new) == 0:
        if return_details:
            return 1e10, I_new, mask, centers_r_x_new, centers_r_y_new, r_center_new
        return 1e10

    # Berechne Uniformity
    uniformity_new = np.std(I_inside_new) / np.mean(I_inside_new)
    if return_details:
        return uniformity_new, I_new, mask, centers_r_x_new, centers_r_y_new, r_center_new
    return uniformity_new

def calculate_overlapp_width(width_value,return_details=False):
    """Berechnet den Crosstalk (Nachbar-Überlapp) für einen gegebenen width-Wert"""
    width_0 = width_value[0] if hasattr(width_value, '__len__') else width_value

    fx_freq_new = multitone_frequencies(N_x, offset, width_0)
    fy_freq_new = multitone_frequencies(N_y, offset, width_0)

    f_center_new = offset + width_0 / 2
    theta_center_new = angle_from_frequency(f_center_new, offset, theta_max, f_band)
    r_center_new = radius_from_angle(theta_center_new, f1, f2, fLO)

    centers_r_x_new = []
    centers_r_y_new = []
    for fx in fx_freq_new:
        theta_x = angle_from_frequency(fx, offset, theta_max, f_band)
        r_x = radius_from_angle(theta_x, f1, f2, fLO)
        for fy in fy_freq_new:
            theta_y = angle_from_frequency(fy, offset, theta_max, f_band)
            r_y = radius_from_angle(theta_y, f1, f2, fLO)
            centers_r_x_new.append(r_x)
            centers_r_y_new.append(r_y)

    centers_r_x_new = np.array(centers_r_x_new)
    centers_r_y_new = np.array(centers_r_y_new)

    sigma_new = win_optimal
    I_new = gaussian_2d_distance_from_centers(X, Y, centers_r_x_new, centers_r_y_new, sigma=sigma_new)
    I_new /= np.max(I_new)

    I_neighbor_new = create_neighbourhood(X, Y, pitch, centers_r_x_new, centers_r_y_new, w_in=sigma_new)

    if width_0 < 0 or width_0 > 1e6:
        if return_details:
            return 1e10, I_new, I_neighbor_new, None, centers_r_x_new, centers_r_y_new, r_center_new
        return 1e10

    if useconvexhull:
        try:
            points_new = np.column_stack([centers_r_x_new, centers_r_y_new])
            hull_new = ConvexHull(points_new)
            hull_vertices_new = hull_new.points[hull_new.vertices]
            polygon_path_new = Path(hull_vertices_new)

            grid_points = np.column_stack([X.ravel(), Y.ravel()])
            mask = polygon_path_new.contains_points(grid_points).reshape(X.shape)

        except Exception:
            if return_details:
                return 1e10, I_new, I_neighbor_new, None, centers_r_x_new, centers_r_y_new, r_center_new
            return 1e10
    elif use_Levels:
        mask = I_new >= levels[threshindex]

    else:
        mask = overlap_mask_pitch(X, Y, r_center_new, r_center_new, pitch)

    I_inside_new = I_new[mask]
    I_inside_neighbor_new = I_neighbor_new[mask]

    if len(I_inside_new) == 0 or np.mean(I_inside_new) == 0:
        if return_details:
            return 1e10, I_new, I_neighbor_new, mask, centers_r_x_new, centers_r_y_new, r_center_new
        return 1e10

    P_inside_new = np.sum(I_inside_new)
    P_neighbor_new = np.sum(I_inside_neighbor_new)
    eta = P_neighbor_new / P_inside_new

    if return_details:
        return eta, I_new, I_neighbor_new, mask, centers_r_x_new, centers_r_y_new, r_center_new
    return eta

def objective_width(width_value, alpha=0.9):
    u = calculate_uniformity_width(width_value)
    o = calculate_overlapp_width(width_value)

    return alpha * u + (1 - alpha) * o

print("\nStarte Optimierung von width zur Minimierung von Uniformity + Crosstalk...")
result_width = minimize(
    objective_width,
    x0=[width],
    method='Nelder-Mead',
    options={'xatol': 1e-8, 'fatol': 1e-7, 'maxiter': 1000}
)

width_optimal = result_width.x[0]

# Einmalige Berechnung mit allen Details für Auswertung und Plots
uniformity_opt_width, I_ort_width_opt, mask_opt, centers_r_x_opt, centers_r_y_opt, r_center_opt = \
    calculate_uniformity_width([width_optimal], return_details=True)
overlap_opt_width, _, I_neighbor_width_opt, _, _, _, _ = \
    calculate_overlapp_width([width_optimal], return_details=True)

mean_intensity_opt_width = np.mean(I_ort_width_opt[mask_opt])
std_intensity_opt_width = np.std(I_ort_width_opt[mask_opt])

print(f"\nOptimale width: {width_optimal:.6e} Hz")
print(f"Mittlere Intensität (width-Opt.): {mean_intensity_opt_width:.4f}")
print(f"Standardabweichung (width-Opt.): {std_intensity_opt_width:.4f}")
print(f"Uniformity (width-Opt.): {uniformity_opt_width:.4f} ({uniformity_opt_width*100:.2f}%)")
print(f"Crosstalk (width-Opt.): {overlap_opt_width:.4f} ({overlap_opt_width*100:.3f}%)")

# --------------------------------------------------
# Plot: Frequenzabstände optimiert + Speichern
# --------------------------------------------------
fig3 = plt.figure(figsize=(12, 7))
fig3.suptitle(f"Frequenz-Optimiert: width = {width_optimal:.3e} Hz, Uniformity = {uniformity_opt_width*100:.2f}%", fontsize=14, fontweight='bold')

# Colorplot links
ax_2d_width = plt.subplot(1, 2, 1)
im_width = ax_2d_width.imshow(
    I_ort_width_opt,
    origin="lower",
    extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
    aspect="equal",
    cmap="viridis"
)
if drawpoints:

    ax_2d_width.scatter(
        centers_r_x_opt * 1e6,
        centers_r_y_opt * 1e6,
        c="red",
        s=20,
        label="Spot-Zentren"
    )

# Zeichne die ConvexHull-Region
if useconvexhull:
    points_opt = np.column_stack([centers_r_x_opt, centers_r_y_opt])
    hull_opt = ConvexHull(points_opt)
    hull_vertices_opt = hull_opt.points[hull_opt.vertices]
    hull_vertices_closed_opt = np.vstack([hull_vertices_opt, hull_vertices_opt[0]])
    ax_2d_width.plot(
        hull_vertices_closed_opt[:, 0] * 1e6,
        hull_vertices_closed_opt[:, 1] * 1e6,
        'b--',
        linewidth=2,
        label="Uniformity-Region"
    )
elif use_Levels:
    cs = ax_2d_width.contour(
        X * 1e6, Y * 1e6, I_ort_width_opt,
        levels=levels, colors=['blue', 'red', 'green'], linewidths=2
    )
    ax_2d_width.clabel(cs, inline=True, fontsize=8)

else:
    half_side_opt = pitch / 2
    rect_opt2d = Rectangle(
        ((r_center_opt - half_side_opt) * 1e6, (r_center_opt - half_side_opt) * 1e6),
        pitch * 1e6, pitch * 1e6,
        edgecolor="red", facecolor="none", linewidth=2, label="Crosstalk-Bereich"
    )
    ax_2d_width.add_patch(rect_opt2d)

    half_side_uniform_opt = uniformity_side_length / 2
    rect_uniform_opt = Rectangle(
        ((r_center_opt - half_side_uniform_opt) * 1e6, (r_center_opt - half_side_uniform_opt) * 1e6),
        uniformity_side_length * 1e6, uniformity_side_length * 1e6,
        edgecolor="cyan", facecolor="none", linewidth=2, label="Uniformity-Bereich"
    )
    ax_2d_width.add_patch(rect_uniform_opt)

    ax_2d_width.plot(
        r_center_opt * 1e6, r_center_opt * 1e6,
        "r+", markersize=12, markeredgewidth=2, label="Mittelpunkt"
    )

ax_2d_width.set_xlabel("Ort $x$ (µm)")
ax_2d_width.set_ylabel("Ort $y$ (µm)")
ax_2d_width.legend()
plt.colorbar(im_width, ax=ax_2d_width, label="Normierte Intensität")

# Schnitte rechts
# x-Schnitt
ax_x_width = plt.subplot(2, 2, 2)
ax_x_width.plot(x * 1e6, I_ort_width_opt[mid_y_idx, :], 'b-', linewidth=2)
ax_x_width.set_xlabel("Ort $x$ (µm)")
ax_x_width.set_ylabel("Intensität")
ax_x_width.set_title("Schnitt durch $y = 0$")
ax_x_width.grid(True, alpha=0.3)

# y-Schnitt
ax_y_width = plt.subplot(2, 2, 4)
ax_y_width.plot(y * 1e6, I_ort_width_opt[:, mid_x_idx], 'g-', linewidth=2)
ax_y_width.set_xlabel("Ort $y$ (µm)")
ax_y_width.set_ylabel("Intensität")
ax_y_width.set_title("Schnitt durch $x = 0$")
ax_y_width.grid(True, alpha=0.3)

plt.tight_layout()

out_file_frequencyOptimized = out_dir / f"FlatMultiTone_FrequencyOptimized_{offset*1e-6,width*1e-6,N_x,N_y}.png"
plt.savefig(out_file_frequencyOptimized, dpi=150, bbox_inches='tight')
print("\nGrafik gespeichert: FlatMultiTone_FrequencyOptimized.png")
plt.show()

# --------------------------------------------------
# Plot: Nachbarregionen (Crosstalk) für width-optimiert
# --------------------------------------------------
fig3b = plt.figure(figsize=(8, 7))
fig3b.suptitle(f"Nachbarregionen (width-optimiert): Crosstalk = {overlap_opt_width*100:.3f}%", fontsize=14, fontweight='bold')

ax_neighbor_width = plt.subplot(1, 1, 1)
im_neighbor_width = ax_neighbor_width.imshow(
    I_neighbor_width_opt,
    origin="lower",
    extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
    aspect="equal",
    cmap="viridis"
)

if drawpoints:
    ax_neighbor_width.scatter(
        centers_r_x_opt * 1e6,
        centers_r_y_opt * 1e6,
        c="red",
        s=20,
        label="Spot-Zentren"
    )

# Quadrat (Zielstruktur) einzeichnen
half_side_opt = pitch / 2
rect_width = Rectangle(
    ((r_center_opt - half_side_opt) * 1e6, (r_center_opt - half_side_opt) * 1e6),
    pitch * 1e6,
    pitch * 1e6,
    edgecolor="red",
    facecolor="none",
    linewidth=2,
    label="Maske"
)
ax_neighbor_width.add_patch(rect_width)

# Mittelpunkt einzeichnen
ax_neighbor_width.plot(
    r_center_opt * 1e6,
    r_center_opt * 1e6,
    "r+",
    markersize=12,
    markeredgewidth=2,
    label="Mittelpunkt"
)

plt.colorbar(im_neighbor_width, ax=ax_neighbor_width, label="Intensität (Nachbarn)")
ax_neighbor_width.set_xlabel("x [µm]")
ax_neighbor_width.set_ylabel("y [µm]")
ax_neighbor_width.legend()
plt.tight_layout()

out_file_neighborWidthOptimized = out_dir / f"FlatMultiTone_NeighborWidthOptimized_{offset*1e-6,width*1e-6,N_x,N_y}.png"
plt.savefig(out_file_neighborWidthOptimized, dpi=150, bbox_inches='tight')
print("\nGrafik gespeichert: FlatMultiTone_NeighborWidthOptimized.png")
plt.show()

print("\n" + "="*60)
print("ZUSAMMENFASSUNG ALLER OPTIMIERUNGEN")
print("="*60)
print(f"Original:           Uniformity = {uniformity*100:.2f}%, Crosstalk = {eta*100:.3f}%")
print(f"win-optimiert:      Uniformity = {uniformity_optimal*100:.2f}% (Verbesserung: {((uniformity - uniformity_optimal)/uniformity*100):.2f}%), "
      f"Crosstalk = {overlap_optimal*100:.3f}% (Verbesserung: {((eta - overlap_optimal)/eta*100):.2f}%)")
print(f"width-optimiert:    Uniformity = {uniformity_opt_width*100:.2f}% (Verbesserung: {((uniformity - uniformity_opt_width)/uniformity*100):.2f}%), "
      f"Crosstalk = {overlap_opt_width*100:.3f}% (Verbesserung: {((eta - overlap_opt_width)/eta*100):.2f}%)")

# ==================================================
# DRITTE ITERATION: Optimiere individuelle Frequenzverschiebungen (Phasen)
# ==================================================
# print("\n" + "="*60)
# print("DRITTE ITERATION: Optimierung individueller Frequenzverschiebungen (Phasen)")
# print("width bleibt bei: {:.3e} Hz".format(width_optimal))
# print("win bleibt bei: {:.3e} m".format(win))
# print("="*60)

# def calculate_uniformity_phases(params):
#     """params: array of length N_x + N_y, first N_x are delta fx, next N_y are delta fy"""
#     try:
#         delta_fx = params[:N_x]
#         delta_fy = params[N_x:]

#         fx_new = multitone_frequencies(N_x, offset, width_optimal) + delta_fx
#         fy_new = multitone_frequencies(N_y, offset, width_optimal) + delta_fy

#         centers_r_x_new = []
#         centers_r_y_new = []
#         for fx in fx_new:
#             theta_x = angle_from_frequency(fx, offset, theta_max, f_band)
#             r_x = radius_from_angle(theta_x, f1, f2, fLO)
#             for fy in fy_new:
#                 theta_y = angle_from_frequency(fy, offset, theta_max, f_band)
#                 r_y = radius_from_angle(theta_y, f1, f2, fLO)
#                 centers_r_x_new.append(r_x)
#                 centers_r_y_new.append(r_y)

#         centers_r_x_new = np.array(centers_r_x_new)
#         centers_r_y_new = np.array(centers_r_y_new)

#         sigma_new = beam_radius_scaled(f1, f2, lambda_opt, fLO, win)
#         I_new = gaussian_2d_distance_from_centers(X, Y, centers_r_x_new, centers_r_y_new, sigma=sigma_new)
#         I_new /= np.max(I_new)

#         points_new = np.column_stack([centers_r_x_new, centers_r_y_new])
#         hull_new = ConvexHull(points_new)
#         hull_vertices_new = hull_new.points[hull_new.vertices]
#         polygon_path_new = Path(hull_vertices_new)

#         grid_points = np.column_stack([X.ravel(), Y.ravel()])
#         inside_polygon_new = polygon_path_new.contains_points(grid_points).reshape(X.shape)

#         I_inside_new = I_new[inside_polygon_new]
#         if len(I_inside_new) == 0 or np.mean(I_inside_new) == 0:
#             return 1e10

#         return np.std(I_inside_new) / np.mean(I_inside_new)
#     except Exception:
#         return 1e10

# # Start with zero deviations
# init_params = np.zeros(N_x + N_y)
# print("\nStarte Optimierung der Phasen ({} Variablen)...".format(len(init_params)))
# res_phases = minimize(
#     calculate_uniformity_phases,
#     x0=init_params,
#     method='Nelder-Mead',
#     options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 2000}
# )

# params_opt = res_phases.x
# uniformity_phases_opt = res_phases.fun

# delta_fx_opt = params_opt[:N_x]
# delta_fy_opt = params_opt[N_x:]

# fx_centers_freq_phase = multitone_frequencies(N_x, offset, width_optimal) + delta_fx_opt
# fy_centers_freq_phase = multitone_frequencies(N_y, offset, width_optimal) + delta_fy_opt

# centers_r_x_phase = []
# centers_r_y_phase = []
# for fx in fx_centers_freq_phase:
#     theta_x = angle_from_frequency(fx, offset, theta_max, f_band)
#     r_x = radius_from_angle(theta_x, f1, f2, fLO)
#     for fy in fy_centers_freq_phase:
#         theta_y = angle_from_frequency(fy, offset, theta_max, f_band)
#         r_y = radius_from_angle(theta_y, f1, f2, fLO)
#         centers_r_x_phase.append(r_x)
#         centers_r_y_phase.append(r_y)

# centers_r_x_phase = np.array(centers_r_x_phase)
# centers_r_y_phase = np.array(centers_r_y_phase)

# # compute intensity and uniformity for phase-optimized
# I_ort_phase = gaussian_2d_distance_from_centers(X, Y, centers_r_x_phase, centers_r_y_phase, sigma=beam_radius_scaled(f1, f2, lambda_opt, fLO, win))
# I_ort_phase /= np.max(I_ort_phase)

# points_phase = np.column_stack([centers_r_x_phase, centers_r_y_phase])
# hull_phase = ConvexHull(points_phase)
# hull_vertices_phase = hull_phase.points[hull_phase.vertices]
# polygon_path_phase = Path(hull_vertices_phase)
# inside_polygon_phase = polygon_path_phase.contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
# I_inside_phase = I_ort_phase[inside_polygon_phase]
# mean_intensity_phase = np.mean(I_inside_phase)
# std_intensity_phase = np.std(I_inside_phase)
# uniformity_phase = std_intensity_phase / mean_intensity_phase

# print(f"\nOptimierte Phasen Uniformity: {uniformity_phases_opt:.4f} ({uniformity_phases_opt*100:.2f}%)")
# print(f"Delta fx (Hz): {delta_fx_opt}")
# print(f"Delta fy (Hz): {delta_fy_opt}")

# # Plot phase-optimized result
# fig4 = plt.figure(figsize=(12, 7))
# fig4.suptitle(f"Phasen-Optimiert: Uniformity = {uniformity_phase*100:.2f}%", fontsize=14, fontweight='bold')
# ax_2d_phase = plt.subplot(1, 2, 1)
# im_phase = ax_2d_phase.imshow(I_ort_phase, origin='lower', extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6], aspect='equal', cmap='viridis')
# ax_2d_phase.scatter(centers_r_x_phase*1e6, centers_r_y_phase*1e6, c='red', s=20, label='Spot-Zentren')
# hull_vertices_closed_phase = np.vstack([hull_vertices_phase, hull_vertices_phase[0]])
# ax_2d_phase.plot(hull_vertices_closed_phase[:,0]*1e6, hull_vertices_closed_phase[:,1]*1e6, 'b--', linewidth=2, label='Uniformity-Region')
# ax_2d_phase.set_xlabel('Ort $x$ (µm)')
# ax_2d_phase.set_ylabel('Ort $y$ (µm)')
# ax_2d_phase.legend()
# plt.colorbar(im_phase, ax=ax_2d_phase, label='Normierte Intensität')

# ax_x_phase = plt.subplot(2, 2, 2)
# ax_x_phase.plot(x*1e6, I_ort_phase[mid_y_idx,:], 'b-')
# ax_x_phase.set_title('Schnitt durch $y=0$')
# ax_y_phase = plt.subplot(2, 2, 4)
# ax_y_phase.plot(y*1e6, I_ort_phase[:,mid_x_idx], 'g-')
# ax_y_phase.set_title('Schnitt durch $x=0$')
# plt.tight_layout()
# out_file_phaseOptimized = out_dir / f"FlatMultiTone_PhaseOptimized_{offset*1e-6,width*1e-6,N_x,N_y}.png"
# plt.savefig(out_file_phaseOptimized, dpi=150, bbox_inches='tight')
# print('\nGrafik gespeichert: FlatMultiTone_PhaseOptimized.png')
# plt.show()

# print("\n" + "="*60)
# print("ENDGÜLTIGE ZUSAMMENFASSUNG")
# print("="*60)
# print(f"Original:           Uniformity = {uniformity*100:.2f}%")
# print(f"win-optimiert:      Uniformity = {uniformity_optimal_check*100:.2f}% (Verbesserung: {((uniformity - uniformity_optimal_check)/uniformity*100):.2f}%)")
# print(f"width-optimiert:    Uniformity = {uniformity_opt_width*100:.2f}% (Verbesserung: {((uniformity - uniformity_opt_width)/uniformity*100):.2f}%)")
# print(f"phasen-optimiert:   Uniformity = {uniformity_phase*100:.2f}% (Verbesserung: {((uniformity - uniformity_phase)/uniformity*100):.2f}%)")

print("\n" + "="*60)
print("VIERTE ITERATION: Optimierung von win, width und Amplituden pro Achse")
print("="*60)

def calculate_combined(params, return_details=False):
    """
    params = [win_val, width_val, amp_x_0..amp_x_{N_x-1}, amp_y_0..amp_y_{N_y-1}]
    Berechnet Uniformity und Crosstalk für gegebene win, width und Amplituden.
    """
    win_val = params[0]
    width_val = params[1]
    amp_x = np.array(params[2:2+N_x])
    amp_y = np.array(params[2+N_x:2+N_x+N_y])

    # Spot-Amplituden: Reihenfolge muss zur Ortsraum-Positionsschleife passen (x außen, y innen)
    amp_spots = np.repeat(amp_x, N_y) * np.tile(amp_y, N_x)

    # Neue Frequenzpositionen und Zentrum
    fx_freq_new = multitone_frequencies(N_x, offset, width_val)
    fy_freq_new = multitone_frequencies(N_y, offset, width_val)
    f_center_new = offset + width_val / 2
    theta_center_new = angle_from_frequency(f_center_new, offset, theta_max, f_band)
    r_center_new = radius_from_angle(theta_center_new, f1, f2, fLO)

    centers_r_x_new = []
    centers_r_y_new = []
    for fx in fx_freq_new:
        theta_x = angle_from_frequency(fx, offset, theta_max, f_band)
        r_x = radius_from_angle(theta_x, f1, f2, fLO)
        for fy in fy_freq_new:
            theta_y = angle_from_frequency(fy, offset, theta_max, f_band)
            r_y = radius_from_angle(theta_y, f1, f2, fLO)
            centers_r_x_new.append(r_x)
            centers_r_y_new.append(r_y)

    centers_r_x_new = np.array(centers_r_x_new)
    centers_r_y_new = np.array(centers_r_y_new)

    sigma = win_val
    I_new = gaussian_2d_weighted_distance_from_centers(X, Y, centers_r_x_new, centers_r_y_new, sigma, amp_spots)

    if np.max(I_new) == 0:
        if return_details:
            return 1e10, None, None, centers_r_x_new, centers_r_y_new, r_center_new
        return 1e10
    I_new /= np.max(I_new)

    I_neighbor_new = create_neighbourhood(X, Y, pitch, centers_r_x_new, centers_r_y_new, w_in=sigma, amps=amp_spots)

    # Masken: Uniformity im kleineren Bereich, Crosstalk weiterhin über pitch
    if useconvexhull:
        try:
            points_new = np.column_stack([centers_r_x_new, centers_r_y_new])
            hull_new = ConvexHull(points_new)
            hull_vertices_new = hull_new.points[hull_new.vertices]
            polygon_path_new = Path(hull_vertices_new)
            grid_points = np.column_stack([X.ravel(), Y.ravel()])
            mask_uniformity = polygon_path_new.contains_points(grid_points).reshape(X.shape)
        except Exception:
            if return_details:
                return 1e10, I_new, I_neighbor_new, centers_r_x_new, centers_r_y_new, r_center_new
            return 1e10
    elif use_Levels:
        mask_uniformity = I_new >= levels[threshindex]
    else:
        mask_uniformity = overlap_mask_pitch(X, Y, r_center_new, r_center_new, uniformity_side_length)

    mask_crosstalk = overlap_mask_pitch(X, Y, r_center_new, r_center_new, pitch)

    I_inside_uniform = I_new[mask_uniformity]
    I_inside_cross = I_new[mask_crosstalk]
    I_neighbor_inside = I_neighbor_new[mask_crosstalk]

    if len(I_inside_uniform) == 0 or np.mean(I_inside_uniform) == 0 or len(I_inside_cross) == 0 or np.sum(I_inside_cross) == 0:
        if return_details:
            return 1e10, I_new, I_neighbor_new, centers_r_x_new, centers_r_y_new, r_center_new
        return 1e10

    uniformity_new = np.std(I_inside_uniform) / np.mean(I_inside_uniform)
    eta_new = np.sum(I_neighbor_inside) / np.sum(I_inside_cross)

    if return_details:
        return uniformity_new, eta_new, I_new, I_neighbor_new, centers_r_x_new, centers_r_y_new, r_center_new
    return uniformity_new, eta_new


def objective_combined(params, alpha=0.9):
    u, o = calculate_combined(params)
    return alpha * u + (1 - alpha) * o


# Startwerte: bereits optimierte win/width, Amplituden = 1
init_win = win_optimal
init_width = width_optimal
init_params = np.concatenate(([init_win, init_width], np.ones(N_x), np.ones(N_y)))

# Bounds: win wie in der ersten Optimierung, width großzügig, Amplituden in [0,2]
bounds = [(0.1e-6, 3.5e-6), (1e3, 5e6)] + [(0.0, 2.0)] * (N_x + N_y)

print("\nStarte kombinierte Optimierung (win + width + Amplituden)...")
result_combined = minimize(
    objective_combined,
    x0=init_params,
    method='L-BFGS-B',
    bounds=bounds,
    options={'ftol': 1e-9, 'maxiter': 2000}
)

win_comb_opt = result_combined.x[0]
width_comb_opt = result_combined.x[1]
amps_x_comb_opt = result_combined.x[2:2+N_x]
amps_y_comb_opt = result_combined.x[2+N_x:2+N_x+N_y]

uniformity_comb_opt, overlap_comb_opt, I_ort_comb, I_neighbor_comb, centers_r_x_comb, centers_r_y_comb, r_center_comb = \
    calculate_combined(result_combined.x, return_details=True)

print(f"\nOptimales win: {win_comb_opt:.6e} m")
print(f"Optimale width: {width_comb_opt:.6e} Hz")
print(f"Optimierte amp_x: {amps_x_comb_opt}")
print(f"Optimierte amp_y: {amps_y_comb_opt}")
print(f"Uniformity (win+width+amps): {uniformity_comb_opt:.4f} ({uniformity_comb_opt*100:.2f}%)")
print(f"Crosstalk (win+width+amps): {overlap_comb_opt:.4f} ({overlap_comb_opt*100:.3f}%)")

# --------------------------------------------------
# Plot: Intensitätsverteilung (win+width+amps optimiert)
# --------------------------------------------------
fig_comb = plt.figure(figsize=(12, 7))
fig_comb.suptitle(
    f"win+width+Amp-Optimiert: win = {win_comb_opt:.3e} m, width = {width_comb_opt:.3e} Hz, "
    f"Uniformity = {uniformity_comb_opt*100:.2f}%, Crosstalk = {overlap_comb_opt*100:.2f}%",
    fontsize=13, fontweight='bold'
)

ax_2d_comb = plt.subplot(1, 2, 1)
im_comb = ax_2d_comb.imshow(
    I_ort_comb,
    origin="lower",
    extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
    aspect="equal",
    cmap="viridis"
)
if drawpoints:
    ax_2d_comb.scatter(
        centers_r_x_comb * 1e6,
        centers_r_y_comb * 1e6,
        c="red",
        s=20,
        label="Spot-Zentren"
    )

if useconvexhull:
    points_comb = np.column_stack([centers_r_x_comb, centers_r_y_comb])
    hull_comb = ConvexHull(points_comb)
    hull_vertices_comb = hull_comb.points[hull_comb.vertices]
    hull_vertices_closed_comb = np.vstack([hull_vertices_comb, hull_vertices_comb[0]])
    ax_2d_comb.plot(
        hull_vertices_closed_comb[:, 0] * 1e6,
        hull_vertices_closed_comb[:, 1] * 1e6,
        'b--', linewidth=2, label="Uniformity-Region"
    )
elif use_Levels:
    cs = ax_2d_comb.contour(
        X * 1e6, Y * 1e6, I_ort_comb,
        levels=levels, colors=['blue', 'red', 'green'], linewidths=2
    )
    ax_2d_comb.clabel(cs, inline=True, fontsize=8)
else:
    half_side_comb = pitch / 2
    rect_comb = Rectangle(
        ((r_center_comb - half_side_comb) * 1e6, (r_center_comb - half_side_comb) * 1e6),
        pitch * 1e6, pitch * 1e6,
        edgecolor="red", facecolor="none", linewidth=2, label="Crosstalk-Bereich"
    )
    ax_2d_comb.add_patch(rect_comb)

    half_side_uniform_comb = uniformity_side_length / 2
    rect_uniform_comb = Rectangle(
        ((r_center_comb - half_side_uniform_comb) * 1e6, (r_center_comb - half_side_uniform_comb) * 1e6),
        uniformity_side_length * 1e6, uniformity_side_length * 1e6,
        edgecolor="cyan", facecolor="none", linewidth=2, label="Uniformity-Bereich"
    )
    ax_2d_comb.add_patch(rect_uniform_comb)

    ax_2d_comb.plot(
        r_center_comb * 1e6, r_center_comb * 1e6,
        "r+", markersize=12, markeredgewidth=2, label="Mittelpunkt"
    )

ax_2d_comb.set_xlabel("Ort $x$ (µm)")
ax_2d_comb.set_ylabel("Ort $y$ (µm)")
ax_2d_comb.legend()
plt.colorbar(im_comb, ax=ax_2d_comb, label="Normierte Intensität")

mid_y_idx_comb = len(y) // 2
mid_x_idx_comb = len(x) // 2

ax_x_comb = plt.subplot(2, 2, 2)
ax_x_comb.plot(x * 1e6, I_ort_comb[mid_y_idx_comb, :], 'b-', linewidth=2)
ax_x_comb.set_xlabel("Ort $x$ (µm)")
ax_x_comb.set_ylabel("Intensität")
ax_x_comb.set_title("Schnitt entlang x durch Zentrum")
ax_x_comb.grid(True, alpha=0.3)

ax_y_comb = plt.subplot(2, 2, 4)
ax_y_comb.plot(y * 1e6, I_ort_comb[:, mid_x_idx_comb], 'g-', linewidth=2)
ax_y_comb.set_xlabel("Ort $y$ (µm)")
ax_y_comb.set_ylabel("Intensität")
ax_y_comb.set_title("Schnitt entlang y durch Zentrum")
ax_y_comb.grid(True, alpha=0.3)

plt.tight_layout()
out_file_combOptimized = out_dir / f"FlatMultiTone_CombinedOptimized_{offset*1e-6,width*1e-6,N_x,N_y}.png"
plt.savefig(out_file_combOptimized, dpi=150, bbox_inches='tight')
print("\nGrafik gespeichert: FlatMultiTone_CombinedOptimized.png")
plt.show()

# --------------------------------------------------
# Plot: Nachbarregionen (Crosstalk) für win+width+amps optimiert
# --------------------------------------------------
fig_comb_b = plt.figure(figsize=(8, 7))
fig_comb_b.suptitle(f"Nachbarregionen (win+width+Amp-optimiert): Crosstalk = {overlap_comb_opt*100:.3f}%", fontsize=14, fontweight='bold')

ax_neighbor_comb = plt.subplot(1, 1, 1)
im_neighbor_comb = ax_neighbor_comb.imshow(
    I_neighbor_comb,
    origin="lower",
    extent=[x[0]*1e6, x[-1]*1e6, y[0]*1e6, y[-1]*1e6],
    aspect="equal",
    cmap="viridis"
)
if drawpoints:
    ax_neighbor_comb.scatter(
        centers_r_x_comb * 1e6,
        centers_r_y_comb * 1e6,
        c="red",
        s=20,
        label="Spot-Zentren"
    )

half_side_comb2 = pitch / 2
rect_comb2 = Rectangle(
    ((r_center_comb - half_side_comb2) * 1e6, (r_center_comb - half_side_comb2) * 1e6),
    pitch * 1e6, pitch * 1e6,
    edgecolor="red", facecolor="none", linewidth=2, label="Maske"
)
ax_neighbor_comb.add_patch(rect_comb2)

ax_neighbor_comb.plot(
    r_center_comb * 1e6, r_center_comb * 1e6,
    "r+", markersize=12, markeredgewidth=2, label="Mittelpunkt"
)

plt.colorbar(im_neighbor_comb, ax=ax_neighbor_comb, label="Intensität (Nachbarn)")
ax_neighbor_comb.set_xlabel("x [µm]")
ax_neighbor_comb.set_ylabel("y [µm]")
ax_neighbor_comb.legend()
plt.tight_layout()

out_file_neighborCombOptimized = out_dir / f"FlatMultiTone_NeighborCombinedOptimized_{offset*1e-6,width*1e-6,N_x,N_y}.png"
plt.savefig(out_file_neighborCombOptimized, dpi=150, bbox_inches='tight')
print("Grafik gespeichert: FlatMultiTone_NeighborCombinedOptimized.png")
plt.show()

print("\n" + "="*60)
print("ENDGÜLTIGE ZUSAMMENFASSUNG")
print("="*60)
print(f"Original:              Uniformity = {uniformity*100:.2f}%, Crosstalk = {eta*100:.3f}%")
print(f"win-optimiert:         Uniformity = {uniformity_optimal*100:.2f}%, Crosstalk = {overlap_optimal*100:.3f}%")
print(f"width-optimiert:       Uniformity = {uniformity_opt_width*100:.2f}%, Crosstalk = {overlap_opt_width*100:.3f}%")
print(f"win+width+amp-optim.:  Uniformity = {uniformity_comb_opt*100:.2f}%, Crosstalk = {overlap_comb_opt*100:.3f}%")
print("="*60)