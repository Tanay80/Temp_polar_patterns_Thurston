import numpy as np
import healpy as hp

GEOMETRIES = {
    "H2xR": ("results_maps/map_rh2.npz",  "negative"),
    "S2xR": ("results_maps/map_rs2.npz",  "positive"),
    "UH2":  ("results_maps/map_uh2.npz",  "negative"),
    "Nil":  ("results_maps/map_nil.npz",  "negative"),
    "Solv": ("results_maps/map_solv.npz", "negative"),
}
PLANCK_MAP  = "COM_CMB_IQU-SMICA_2048_R3.00_full.fits"
THEORY_FILE = "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
LMAX = 63                                                                                           #Conservative choise; max value is 3NSIDE - 1
BB_UPPER_LIMIT_C2 = 1.101135e-02                                    
DIPOLE_L, DIPOLE_B = 264.021, 48.253                                                                #Planck 2018 results. III. High Frequency Instrument data processing and frequency maps
_l, _b = np.radians(DIPOLE_L), np.radians(DIPOLE_B)
DIPOLE_VEC = np.array([np.cos(_b)*np.cos(_l), np.cos(_b)*np.sin(_l), np.sin(_b)])

def load_geometry(path):    
    d = np.load(path)
    return d["T"] * 1e6, d["Q"] * 1e6, d["U"] * 1e6, int(d["nside"])                                 #K -> uK

def fit_quadrupole_tensor(T_map, nside):
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix))
    A = np.column_stack([x*x - z*z, y*y - z*z, x*y, x*z, y*z])
    coeffs, *_ = np.linalg.lstsq(A, T_map, rcond=None)
    a, b, c, d, e = coeffs
    return np.array([[a, c/2, d/2], [c/2, b, e/2], [d/2, e/2, -(a+b)]])

def dominant_axis(Q):
    eigvals, eigvecs = np.linalg.eigh(Q)
    return eigvecs[:, np.argmax(np.abs(eigvals))]

def local_compactness(T_map, nside, pct=10, radius_deg=30):
    """Fraction of the coldest `pct`% of pixels lying within `radius_deg` of the
    single coldest pixel. High = one localized cold region; low = cold pixels
    scattered across multiple/antipodal regions (e.g. a smooth even quadrupole)."""
    npix = hp.nside2npix(nside)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix))).T
    coldest_vec = vecs[np.argmin(T_map)]
    thr = np.percentile(T_map, pct)
    cold_vecs = vecs[T_map <= thr]
    ang = np.degrees(np.arccos(np.clip(cold_vecs @ coldest_vec, -1, 1)))
    return np.mean(ang <= radius_deg)

def n_eff_l2(T_map, nside, lmax=LMAX):
    alm = hp.map2alm(T_map, lmax=lmax, iter=3)
    idx = hp.Alm.getidx(lmax, 2, np.arange(0, 3))
    p = np.abs(alm[idx])**2
    tot = p[0] + 2*np.sum(p[1:])
    sq = p[0]**2 + 2*np.sum(p[1:]**2)
    return tot**2 / sq if sq > 0 else 0.0

def check_duplicates(cache):
    names = list(cache.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            Ti = cache[names[i]][0]
            Tj = cache[names[j]][0]
            if np.allclose(Ti, Tj, rtol=1e-10):
                print(f"  *** WARNING: {names[i]} and {names[j]} T-maps are IDENTICAL "
                      f"-- check these were solved with distinct geometry parameters ***")

nside_ref = int(np.load(list(GEOMETRIES.values())[0][0])["nside"])
T_pl = hp.read_map(PLANCK_MAP, field=0) * 1e6                                                           #K_CMB -> uK
T_pl = hp.ud_grade(T_pl, nside_ref)
clTT_pl = hp.anafast(T_pl, lmax=LMAX, iter=3)

theory = np.loadtxt(THEORY_FILE, comments="#")
ell_th = theory[:, 0].astype(int)
idx2 = np.where(ell_th == 2)[0][0]
clTT_th2 = theory[idx2, 1] * 2 * np.pi / (2*3)                                                          #Dl -> Cl
clBB_th2 = theory[idx2, 4] * 2 * np.pi / (2*3)

cache = {name: load_geometry(path) for name, (path, _) in GEOMETRIES.items()}
print("Sanity check for duplicate/degenerate geometry outputs:")
check_duplicates(cache)
print()

results = {}
for name, (T, Q, U, nside) in cache.items():
    Qtens = fit_quadrupole_tensor(T, nside)
    axis = dominant_axis(Qtens)

    npix = hp.nside2npix(nside)
    vecs = np.array(hp.pix2vec(nside, np.arange(npix))).T
    dot = vecs @ axis
    rms_p = np.sqrt(np.nanmean(T[dot >= 0]**2))
    rms_n = np.sqrt(np.nanmean(T[dot < 0]**2))

    cl_full = hp.anafast([T, Q, U], lmax=LMAX, pol=True, iter=3)
    clTT, clEE, clBB, clTE, clEB, clTB = cl_full

    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(axis, DIPOLE_VEC)), -1, 1)))
    compact = local_compactness(T, nside)
    neff = n_eff_l2(T, nside)

    results[name] = dict(
        c2tt=clTT[2], c2bb=clBB[2], angle=angle,
        hemi_ratio=rms_p/rms_n, compact=compact, neff=neff,
        curvature=GEOMETRIES[name][1],
    )

print("=" * 70)
print("1. QUADRUPOLE AMPLITUDE ANOMALY")
print("=" * 70)
print(f"{'Geometry':<8}{'C2^TT (uK^2)':>16}{'Model/Theory':>15}{'Model/Planck':>15}")
for name, r in results.items():
    print(f"{name:<8}{r['c2tt']:>16.4e}{r['c2tt']/clTT_th2:>15.4f}{r['c2tt']/clTT_pl[2]:>15.4f}")
print(f"{'Planck':<8}{clTT_pl[2]:>16.4e}{clTT_pl[2]/clTT_th2:>15.4f}{'--':>15}")
print(f"{'Theory':<8}{clTT_th2:>16.4e}{'--':>15}{clTT_th2/clTT_pl[2]:>15.4f}")
print("NOTE: Planck's observed low-C2 anomaly (ratio << 1 vs theory) is the real")
print("      target; model values are seed-amplitude-dependent, not yet calibrated.\n")

print("=" * 70)
print("2 & 3. QUADRUPOLE AXIS ALIGNMENT + HEMISPHERICAL ASYMMETRY")
print("=" * 70)
print(f"{'Geometry':<8}{'Axis->Dipole(deg)':>18}{'Hemi Ratio':>14}")
for name, r in results.items():
    print(f"{name:<8}{r['angle']:>18.2f}{r['hemi_ratio']:>14.3f}")
print("CAVEAT: a pure l<=2 field is exactly antipodally even, so hemi-ratio ~1.000")
print("        is mathematically expected, not evidence of isotropy.\n")

print("=" * 70)
print("4. LOCALIZED FEATURES (fraction of cold pixels within 30 deg of coldest point)")
print("=" * 70)
for name, r in results.items():
    tag = "localized" if r['compact'] > 0.7 else "spread out" if r['compact'] < 0.3 else "intermediate"
    print(f"{name:<8}{r['compact']:>10.3f}   {tag}")
print("CAVEAT: model is l<=2 only (Thomson-truncated) -- this proxies focusing,")
print("        not a true multi-scale Cold Spot (which needs l~20-30 structure).\n")

print("=" * 70)
print("5. COHERENCE STATISTIC N_eff(l=2)  [isotropic-random max = 2l+1 = 5]")
print("=" * 70)
for name, r in results.items():
    print(f"{name:<8}{r['neff']:>10.2f}")
print("NOTE: N_eff=2 is mathematically forced by single-(l=2,m=2)-mode seeding --")
print("      this is a methodology/limitation result, not a per-geometry finding.\n")

print("=" * 70)
print("6. CROSS-GEOMETRY SUMMARY")
print("=" * 70)
print(f"{'Geom':<6}{'Curv':<10}{'C2':>12}{'Axis':>8}{'Hemi':>8}{'Compact':>9}{'N_eff':>7}")
for name, r in results.items():
    print(f"{name:<6}{r['curvature']:<10}{r['c2tt']:>12.3e}{r['angle']:>8.1f}"
          f"{r['hemi_ratio']:>8.3f}{r['compact']:>9.3f}{r['neff']:>7.2f}")
print()

print("=" * 70)
print("7. B-MODE / TENSOR CONTAMINATION CHECK")
print("=" * 70)
if BB_UPPER_LIMIT_C2 is None:
    print("BB_UPPER_LIMIT_C2 not set -- printing raw C2^BB only, no bound comparison:")
    for name, r in results.items():
        print(f"{name:<8}{r['c2bb']:>16.4e}")
    print("Set BB_UPPER_LIMIT_C2 to a real observational limit before drawing conclusions.")
else:
    for name, r in results.items():
        ratio = r['c2bb'] / BB_UPPER_LIMIT_C2
        flag = "EXCEEDS bound" if ratio > 1 else "within bound"
        print(f"{name:<8}{r['c2bb']:>16.4e}{ratio:>14.4f}   {flag}")
