import os
import glob
import numpy as np
import healpy as hp
import pandas as pd

fits_filename = "COM_CMB_IQU-SMICA_2048_R3.00_full.fits"
if not os.path.exists(fits_filename):
    raise FileNotFoundError(f"'{fits_filename}' not found.")

TARGET_NSIDE = 32
lmax = 3 * TARGET_NSIDE - 1
L_MAX_PARITY = 22

planck_T_raw = hp.read_map(fits_filename, field=0)
planck_T = hp.ud_grade(planck_T_raw, TARGET_NSIDE)
planck_T -= np.mean(planck_T)
planck_var = np.var(planck_T)

def compute_parity_metrics(T_map, target_var=None, l_max_p=L_MAX_PARITY):
    T_map = T_map - np.mean(T_map)
    current_var = np.var(T_map)
    
    if target_var is not None and current_var > 0:
        T_map = T_map * np.sqrt(target_var / current_var)
        
    cl = hp.anafast(T_map, lmax=lmax)
    l_arr = np.arange(len(cl))
    
    dl_uK2 = (l_arr * (l_arr + 1) * cl / (2 * np.pi)) * (1e6)**2
    
    even_mask = (l_arr >= 2) & (l_arr <= l_max_p) & (l_arr % 2 == 0)
    odd_mask = (l_arr >= 3) & (l_arr <= l_max_p) & (l_arr % 2 == 1)
    total_mask = (l_arr >= 2) & (l_arr <= l_max_p)
    
    P_even = np.sum(dl_uK2[even_mask])
    P_odd = np.sum(dl_uK2[odd_mask])
    
    ratio = P_odd / P_even if P_even > 0 else 0.0
    
    g_stat = np.sum(dl_uK2[total_mask] * ((-1.0)**l_arr[total_mask])) / np.sum(dl_uK2[total_mask]) if np.sum(dl_uK2[total_mask]) > 0 else 0.0
    
    return P_even, P_odd, ratio, g_stat

pe_p, po_p, ratio_p, g_p = compute_parity_metrics(planck_T)

results = [{
    "Source": "Planck 2018",
    "P_even [uK^2]": f"{pe_p:.2f}",
    "P_odd [uK^2]": f"{po_p:.2f}",
    "Parity Ratio (Odd/Even)": f"{ratio_p:.3f}",
    "g(l_max=22)": f"{g_p:.3f}"
}]

export_dir = "results_maps"
map_files = sorted(glob.glob(os.path.join(export_dir, "map_*.npz")))

for fpath in map_files:
    data = np.load(fpath)
    geom_name = str(data['geometry'])
    
    pe, po, r, g = compute_parity_metrics(data['T'], target_var=planck_var)
    
    results.append({
        "Source": f"Geom: {geom_name}",
        "P_even [uK^2]": f"{pe:.2f}",
        "P_odd [uK^2]": f"{po:.2f}",
        "Parity Ratio (Odd/Even)": f"{r:.3f}",
        "g(l_max=22)": f"{g:.3f}"
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))
