"""Single Spot, Airy, Waist 1.9 um: Werte und Unsicherheiten entlang des Shifts.

Rechnet die Metriken aus lib/single_beam.py unabhaengig nach (identische Formeln,
reproduziert den Report vom 18.09. auf alle Stellen) und variiert:
  Numerik (Gitter), Waist 1.83/1.97 um, pitch +-30 nm, T +-2 uK, nu_r +-2.3 kHz,
  Nachbarkranz 2, pitch 5.16 um, Shift +-10 nm, Richtung diagonal,
  Atomposition (Gauss-Hermite 7x7, sigma_pos 47 nm und 127 nm).
Ausgabe: single_spot_unsicherheiten.json und die Tabellen als Markdown
(single_spot_unsicherheiten_tabellen.md) neben diesem Skript. Laufzeit ~4 min.
"""
import json, math
from pathlib import Path
import numpy as np
from scipy.special import j1
from scipy.constants import hbar, k as kB, atomic_mass
M = 84.911789738 * atomic_mass
P = dict(waist=1.9e-6, k=1.482951, pitch=5.288e-6, ring=1, R=1e-6, n_hard=401,
         T=17e-6, nu=60.4e3, n_sig=6.0, n_w=241, alpha=0.7, lam=0.75)

def sigma_th(T, nu):
    om = 2*np.pi*nu; x = hbar*om/(2*kB*T)
    return np.sqrt(hbar/(2*M*om)/np.tanh(x))

def airy(X, Y, cx, cy, fz):
    I = np.zeros_like(X)
    kk = 3.83170597/fz
    for a, b in zip(np.atleast_1d(cx), np.atleast_1d(cy)):
        u = kk*np.hypot(X-a, Y-b); p = np.ones_like(u); m = u > 1e-12
        p[m] = (2*j1(u[m])/u[m])**2; I += p
    return I

def nb(pitch, ring):
    idx = np.arange(-ring, ring+1); ix, iy = np.meshgrid(idx, idx); keep = ~((ix == 0) & (iy == 0))
    return ix[keep]*pitch, iy[keep]*pitch

def cells(h, n): s = 2*h/n; return -h+(np.arange(n)+0.5)*s

def metrics(p, ox=0.0, oy=0.0):
    fz = p['k']*p['waist']; cx, cy = nb(p['pitch'], p['ring'])
    n = p['n_hard']; R = p['R']
    xs = cells(R, n); X, Y = np.meshgrid(xs+ox, xs+oy); ins = (X-ox)**2+(Y-oy)**2 <= R**2
    Io = airy(X, Y, 0, 0, fz)[ins]; In = airy(X, Y, cx, cy, fz)[ins]
    U_h = np.std(Io)/np.mean(Io); e_circ = In.sum()/Io.sum()
    xs = cells(p['pitch']/2, n); X, Y = np.meshgrid(xs+ox, xs+oy)
    e_box = airy(X, Y, cx, cy, fz).sum()/airy(X, Y, 0, 0, fz).sum()
    s = sigma_th(p['T'], p['nu']); h = p['n_sig']*s
    xs = np.linspace(ox-h, ox+h, p['n_w']); ys = np.linspace(oy-h, oy+h, p['n_w']); X, Y = np.meshgrid(xs, ys)
    W = np.exp(-((X-ox)**2+(Y-oy)**2)/(2*s*s)); Io = airy(X, Y, 0, 0, fz); In = airy(X, Y, cx, cy, fz)
    mu = (Io*W).sum()/W.sum(); U_w = np.sqrt(((Io-mu)**2*W).sum()/W.sum())/mu
    e_w = (In*W).sum()/(Io*W).sum()
    pen = lambda a, b: 0.5*(a+b)+p['lam']*abs(a-b)
    U_c = pen(U_h, U_w); e_c = pen(e_box, e_w); J = p['alpha']*U_c+(1-p['alpha'])*e_c
    return dict(U_h=U_h*100, eta_circ=e_circ*100, eta_box=e_box*100, U_w=U_w*100, eta_w=e_w*100,
                U_c=U_c*100, eta_c=e_c*100, J=J*100)


# ---------------- Rechnung ----------------
KEYS = ['U_h','eta_circ','eta_box','U_w','eta_w','U_c','eta_c','J']
R = np.round(np.arange(0, 11)*0.19e-6, 12)
def ev(p, r, d='v'):
    return metrics(p, *( (0.0, r) if d=='v' else (r/np.sqrt(2), r/np.sqrt(2)) ))
def var(**kw): q = dict(P); q.update(kw); return q
out = []
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(7); gh_w = gh_w/gh_w.sum()
for r in R:
    row = dict(r=r*1e6)
    base = ev(P, r); row['val'] = base
    fine = ev(var(n_hard=801, n_w=481, n_sig=8.0), r)
    ring2 = ev(var(ring=2), r)
    diag = ev(P, r, 'd')
    def cd(key, h, **kw):
        lo = ev(var(**{key: P[key]-h}), r); hi = ev(var(**{key: P[key]+h}), r)
        return {k: (hi[k]-lo[k])/2 for k in KEYS}   # Aenderung pro +h
    s_w = cd('waist', 10e-9); s_p = cd('pitch', 30e-9); s_T = cd('T', 2e-6); s_nu = cd('nu', 2.3e3)
    pitch_meas = ev(var(pitch=5.16e-6), r)
    lo = ev(P, max(r-10e-9, 0)); hi = ev(P, r+10e-9)
    s_r = {k: (hi[k]-lo[k])/((r+10e-9)-max(r-10e-9, 0))*10e-9 for k in KEYS}
    # Positionsstreuung: hart 127 nm (gesamt), gewichtet 47 nm (technisch)
    pos = {}
    for lab, sp in (('127', 127.1e-9), ('47', 47.0e-9)):
        vals = {k: [] for k in KEYS}; ws = []
        for xi, wi in zip(gh_x, gh_w):
            for yj, wj in zip(gh_x, gh_w):
                m = metrics(P, xi*sp, r+yj*sp)
                for k in KEYS: vals[k].append(m[k])
                ws.append(wi*wj)
        ws = np.array(ws)
        pos[lab] = {k: dict(mean=float((np.array(v)*ws).sum()), std=float(np.sqrt(((np.array(v)-(np.array(v)*ws).sum())**2*ws).sum()))) for k, v in vals.items()}
    row.update(num={k: abs(fine[k]-base[k]) for k in KEYS}, ring2={k: ring2[k]-base[k] for k in KEYS},
               diag={k: diag[k] for k in KEYS}, s_waist10=s_w, s_pitch30=s_p, s_T2=s_T, s_nu23=s_nu,
               pitch516={k: pitch_meas[k]-base[k] for k in KEYS}, s_r10=s_r, pos=pos)
    for lab, w in (('wm', 1.83e-6), ('wp', 1.97e-6)):
        row[lab] = ev(var(waist=w), r)
    out.append(row); print('r', r*1e6, 'done', flush=True)


R = json.loads(json.dumps(out, default=float))

# ---------------- Tabellen ----------------
K = ['U_h','eta_circ','eta_box','U_w','eta_w','U_c','eta_c','J']
H = ['U_h','eta_h^circ','eta_h^box','U_w','eta_w','U_c','eta_c','J']
SW = 7.0   # sigma_waist = 70 nm = 7 x 10 nm
def nd(s):
    if s <= 0: return 4
    return max(0, 1-math.floor(math.log10(s)))
def pm(v, s): d = nd(s); return f"{v:.{d}f} ± {s:.{d}f}"
def rest(row, k):
    return row['s_pitch30'][k]**2 + row['s_T2'][k]**2 + row['s_nu23'][k]**2
def updn(row, k):
    a = row['wm'][k]-row['val'][k]; b = row['wp'][k]-row['val'][k]
    up = max(0.0, a, b); dn = -min(0.0, a, b)
    return math.sqrt(up**2+rest(row, k)), math.sqrt(dn**2+rest(row, k))
def sig(row, k):
    return max(updn(row, k))
def pmA(v, up, dn):
    d = nd(max(up, dn))
    if abs(up-dn) <= 0.1*max(up, dn):
        return f"{v:.{d}f} ± {max(up,dn):.{d}f}"
    return f"{v:.{d}f} +{up:.{d}f}/−{dn:.{d}f}"
def tab(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|"*len(head)] + ["| " + " | ".join(r) + " |" for r in rows])
T = {}
T['main'] = tab(['r (µm)', 'r/w'] + [h+' (%)' for h in H],
    [[f"{r['r']:.2f}", f"{r['r']/1.9:.1f}"] + [pmA(r['val'][k], *updn(r, k)) for k in K] for r in R])
r0 = R[0]
T['budget0'] = tab(['Groesse', 'Wert', 'Numerik', 'Waist 1.83 / 1.97 µm', 'pitch ±30 nm', 'T ±2 µK', 'nu_r ±2.3 kHz', 'sigma_Parameter', 'Nachbarkranz 2 (syst.)', 'pitch 5.16 statt 5.288 µm (syst.)'],
    [[h, f"{r0['val'][k]:.4g} %", f"{r0['num'][k]:.1e}", f"{r0['wm'][k]-r0['val'][k]:+.3f} / {r0['wp'][k]-r0['val'][k]:+.3f}", f"{r0['s_pitch30'][k]:+.3f}", f"{r0['s_T2'][k]:+.3f}", f"{r0['s_nu23'][k]:+.3f}", f"+{updn(r0,k)[0]:.3f} / −{updn(r0,k)[1]:.3f}", f"{r0['ring2'][k]:+.3f}", f"{r0['pitch516'][k]:+.3f}"] for h, k in zip(H, K)])
def single(key, fmt="{:+.3f}", scale=1.0):
    return tab(['r (µm)'] + H, [[f"{r['r']:.2f}"] + [fmt.format(r[key][k]*scale) for k in K] for r in R])
T['dr'] = single('s_r10')
T['waist'] = tab(['r (µm)'] + H, [[f"{r['r']:.2f}"] + [f"{r['wm'][k]-r['val'][k]:+.3f} / {r['wp'][k]-r['val'][k]:+.3f}" for k in K] for r in R])
T['ring2'] = single('ring2')
T['p516'] = single('pitch516')
T['diag'] = tab(['r (µm)'] + ['eta_h^circ senkr.', 'eta_h^circ diag.', 'eta_h^box senkr.', 'eta_h^box diag.', 'eta_w senkr.', 'eta_w diag.', 'eta_c senkr.', 'eta_c diag.', 'J senkr.', 'J diag.'],
    [[f"{r['r']:.2f}"] + sum([[f"{r['val'][k]:.4g}", f"{r['diag'][k]:.4g}"] for k in ['eta_circ','eta_box','eta_w','eta_c','J']], []) for r in R])
for lab in ('47', '127'):
    T['pos'+lab] = tab(['r (µm)'] + [h+' Mittel ± Streuung (%)' for h in H],
        [[f"{r['r']:.2f}"] + [pm(r['pos'][lab][k]['mean'], r['pos'][lab][k]['std']) for k in K] for r in R])
T['nummax'] = f"{max(r['num'][k] for r in R for k in K):.4f}"

BASE = Path(__file__).resolve().parent
json.dump(R, open(BASE / "single_spot_unsicherheiten.json", "w"), indent=1)
with open(BASE / "single_spot_unsicherheiten_tabellen.md", "w", encoding="utf-8") as f:
    for k, v in T.items():
        f.write(f"## {k}\n\n{v}\n\n")
print(T["budget0"])
