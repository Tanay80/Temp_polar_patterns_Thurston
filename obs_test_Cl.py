import os
import glob
import numpy as np
import healpy as hp
import pandas as pd

fits_filename = "COM_CMB_IQU-SMICA_2048_R3.00_full.fits"
TARGET_NSIDE = 32
lmax = 3 * TARGET_NSIDE - 1

def compute_principal_axis(m, l_val):
    alm = hp.map2alm(m, lmax=l_val)
    alm_l = np.zeros_like(alm)
    for m_val in range(l_val + 1):
        idx_lm = hp.Alm.getidx(l_val, l_val, m_val)
        alm_l[idx_lm] = alm[idx_lm]
    map_l = hp.alm2map(alm_l, nside=TARGET_NSIDE, verbose=False)
    max_pix = np.argmax(np.abs(map_l))
    theta, phi = hp.pix2ang(TARGET_NSIDE, max_pix)
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def extract_metrics(map_data, name, target_var=None):
    T_map = map_data - np.mean(map_data)
    current_var = np.var(T_map)
    
    if target_var is not None and current_var > 0:
        T_map = T_map * np.sqrt(target_var / current_var)
        
    cl = hp.anafast(T_map, lmax=lmax)
    ell = np.arange(len(cl))
    
    C2, C3 = cl[2], cl[3]
    ratio_C2_C3 = C2 / C3 if C3 > 1e-40 else np.inf
    
    dl = ell * (ell + 1) * cl / (2 * np.pi)
    P_even = np.sum(dl[2:31:2])
    P_odd = np.sum(dl[3:31:2])
    parity_ratio = P_odd / P_even if P_even > 0 else 0.0
    
    v2 = compute_principal_axis(T_map, 2)
    v3 = compute_principal_axis(T_map, 3)
    cos_angle = np.abs(np.dot(v2, v3))
    align_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0.0, 1.0)))
    
    return {
        "Source": name,
        "C2 [K^2]": f"{C2:.2e}",
        "C3 [K^2]": f"{C3:.2e}",
        "C2 / C3": f"{ratio_C2_C3:.2e}" if np.isinf(ratio_C2_C3) else round(ratio_C2_C3, 3),
        "Parity (Odd/Even)": round(parity_ratio, 3),
        "Alignment (l=2,3) [deg]": round(align_angle_deg, 1)
    }, np.var(T_map)

#Data loading
planck_T_raw = hp.read_map(fits_filename, field=0, verbose=False)
planck_T = hp.ud_grade(planck_T_raw, TARGET_NSIDE)
planck_metrics, planck_var = extract_metrics(planck_T, "Planck 2018")

results = [planck_metrics]

export_dir = "results_maps"
map_files = sorted(glob.glob(os.path.join(export_dir, "map_*.npz")))

for fpath in map_files:
    data = np.load(fpath)
    geom_name = str(data['geometry'])
    geom_metrics, _ = extract_metrics(data['T'], f"Geom: {geom_name}", target_var=planck_var)
    results.append(geom_metrics)

df = pd.DataFrame(results)
print(df.to_string(index=False))
