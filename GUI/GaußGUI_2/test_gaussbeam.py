"""Analytische Gegenrechnungen zur Engine. Aufruf:  python3 test_gaussbeam.py"""
import math
import numpy as np
from gaussbeam import (INF, InputBeam, OpticalSystem, Distance, ThinLens,
                       ThickLens, Crystal, BrewsterCrystal, CurvedMirror,
                       CurvedSurface, interface_matrices, optimize)

OK = [0]
def check(name, got, want, rtol=1e-9, atol=0.0):
    d = abs(got - want)
    tol = atol + rtol * abs(want)
    status = "PASS" if d <= tol else "FAIL"
    if status == "FAIL":
        OK[0] += 1
    print(f"[{status}] {name}: got={got:.12g} want={want:.12g} (dev={d:.3g})")

lam = 1064e-9

# ---------------------------------------------------------------- Freiraum
b = InputBeam(wavelength=lam, w0=0.5e-3, z_waist=0.0)
sys1 = OpticalSystem(b, [Distance(L=300.0)])
r = sys1.propagate()
zr = math.pi * (0.5e-3)**2 / lam
check("Freiraum w(z)", r.tangential.w, 0.5e-3*math.sqrt(1+(0.3/zr)**2))
check("Freiraum w0 erhalten", r.tangential.w0, 0.5e-3)
check("Freiraum Waist liegt 300mm zurueck", r.tangential.z_to_waist, -0.3, atol=1e-12)
check("Freiraum Divergenz", r.tangential.theta, lam/(math.pi*0.5e-3))

# ------------------------------------------------- Duenne Linse, Waist an Linse
f = 0.1
sys2 = OpticalSystem(InputBeam(wavelength=lam, w0=1e-3), [ThinLens(f=100.0)])
r = sys2.propagate()
w0p = 1e-3/math.sqrt(1+(math.pi*(1e-3)**2/lam/f)**2)
dp = f/(1+(f/(math.pi*(1e-3)**2/lam))**2)
check("Linse: neuer Waist", r.tangential.w0, w0p)
check("Linse: Waistabstand", r.tangential.z_to_waist, dp)
check("Linse: keine Asymmetrie", r.sagittal.w0, r.tangential.w0)

# ---------------------------------------------- Gekippte Platte: Ersatzlaenge
t, n, th1 = 10e-3, 1.5, math.radians(35.0)
th2 = math.asin(math.sin(th1)/n)
Ls = t/(n*math.cos(th2))
Lt = t*math.cos(th1)**2/(n*math.cos(th2)**3)
sysA = OpticalSystem(InputBeam(wavelength=lam, w0=0.4e-3),
                     [Crystal(t=10.0, n=1.5, theta=35.0)])
sysB_s = OpticalSystem(InputBeam(wavelength=lam, w0=0.4e-3), [Distance(L=Ls*1e3)])
sysB_t = OpticalSystem(InputBeam(wavelength=lam, w0=0.4e-3), [Distance(L=Lt*1e3)])
ra, rs, rt = sysA.propagate(), sysB_s.propagate(), sysB_t.propagate()
check("Platte sagittal == Freiraum t/(n cos t2)", ra.sagittal.w, rs.tangential.w)
check("Platte tangential == Freiraum t c1^2/(n c2^3)", ra.tangential.w, rt.tangential.w)
check("Platte erzeugt Astigmatismus", ra.astigmatism, -(Lt - Ls), rtol=1e-9)

# ------------------------------------------------------ Determinantenregel
Mt, Ms, th2b = interface_matrices(1.0, 1.7, 0.05, math.radians(28.0))
check("det(Mt) = n1/n2", np.linalg.det(Mt), 1.0/1.7)
check("det(Ms) = n1/n2", np.linalg.det(Ms), 1.0/1.7)
check("Snellius", 1.0*math.sin(math.radians(28.0)), 1.7*math.sin(th2b))

# ------------------------------------- Dicke Linse gegen Linsenschleiferformel
R1, R2, d, ng = 0.06, -0.06, 0.008, 1.5168
finv = (ng-1)*(1/R1 - 1/R2 + (ng-1)*d/(ng*R1*R2))
fL = 1/finv
bfd = fL*(1 - (ng-1)*d/(ng*R1))
# quasi-kollimierter Eingang: sehr grosser Waist an der Linse
big = 30e-3
sys3 = OpticalSystem(InputBeam(wavelength=lam, w0=big),
                     [ThickLens(R1=60.0, R2=-60.0, d=8.0, n=ng)])
r3 = sys3.propagate()
check("Dicke Linse: hintere Schnittweite", r3.tangential.z_to_waist, bfd, rtol=2e-4)

# ---------------------------------------------------------- Gekippter Spiegel
Rm, thm = 0.2, math.radians(15.0)
sys4 = OpticalSystem(InputBeam(wavelength=lam, w0=big),
                     [CurvedMirror(R=200.0, theta=15.0)])
r4 = sys4.propagate()
check("Spiegel tangential f = R cos/2", r4.tangential.z_to_waist,
      Rm*math.cos(thm)/2, rtol=2e-4)
check("Spiegel sagittal f = R/(2 cos)", r4.sagittal.z_to_waist,
      Rm/(2*math.cos(thm)), rtol=2e-4)

# --------------------------------------------------------------------- M^2
b5 = InputBeam(wavelength=lam, w0=0.5e-3, m2=2.0)
check("M^2=2 verdoppelt Divergenz", b5.divergence(), 2*lam/(math.pi*0.5e-3))
b6 = InputBeam.from_divergence(lam, 0.5e-3, 2*lam/(math.pi*0.5e-3))
check("from_divergence findet M^2", b6.m2, 2.0)

# ------------------------------------------------ Brechzahl im Medium: z_R * n
sysG = OpticalSystem(InputBeam(wavelength=lam, w0=0.3e-3),
                     [CurvedSurface(R=0.0, n2=1.5, theta=0.0)])
rg = sysG.propagate()
check("Planflaeche: Waist unveraendert", rg.tangential.w0, 0.3e-3)
check("Planflaeche: z_R skaliert mit n", rg.tangential.z_rayleigh,
      1.5*math.pi*(0.3e-3)**2/lam)

# ------------------------------------------------------------- Optimierung
# Kollimator: Waist 100 um, danach Abstand D, dann f=50mm -> D=f kollimiert
sysO = OpticalSystem(InputBeam(wavelength=lam, w0=100e-6),
                     [Distance(L=150.0), ThinLens(f=200.0)])
res = optimize(sysO, 0, "L", 0.10, 0.30, goal="collimate", plane="tangential")
zr0 = math.pi*(100e-6)**2/lam
fO = 0.2
d_exact = (fO + math.sqrt(fO**2 - 4*zr0**2))/2      # R(d) = f
check("Optimierung findet kollimierenden Abstand", res.value, d_exact, rtol=1e-5)
sysO.components[0].set("L", res.value)
ro = sysO.propagate()
check("nach Optimierung: Waist am Ausgang", ro.tangential.z_to_waist, 0.0, atol=2e-6)

# ------------------------------------------------ Brewster-Kristall Konsistenz
bc = BrewsterCrystal(t=10.0, n=1.76)
sysBc = OpticalSystem(InputBeam(wavelength=lam, w0=0.4e-3), [bc])
rbc = sysBc.propagate()
th1b = math.atan(1.76); th2b_ = math.asin(math.sin(th1b)/1.76)
Ls_ = 10e-3/(1.76*math.cos(th2b_))
Lt_ = 10e-3*math.cos(th1b)**2/(1.76*math.cos(th2b_)**3)
check("Brewster: Astigmatismus", rbc.astigmatism, -(Lt_-Ls_))

print()
print("FEHLER:" , OK[0])
