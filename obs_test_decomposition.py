import os
import glob
import numpy as np
import healpy as hp
import pandas as pd

TARGET_NSIDE = 32
lmax = 3 * TARGET_NSIDE - 1  # lmax = 95.
MAPS_IN_KELVIN = True

results = []
cl_file_path = "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"

if os.path.exists(cl_file_path):
    #Expects columns: ell, TT, TE, EE, BB, PP
    data_spec = np.loadtxt(cl_file_path, comments='#')
    ells = data_spec[:, 0].astype(int)
    
    mask = (ells >= 2) & (ells <= lmax)
    
    dl_ee_planck = data_spec[mask, 3]  								#EE column (D_l in uK^2)
    dl_bb_planck = data_spec[mask, 4]  								#BB column (D_l in uK^2)
    
    l_vals = ells[mask]
    
    #Conversion of D_l [uK^2] to C_l [uK^2]
    cl_factor = (2 * np.pi) / (l_vals * (l_vals + 1))
    cl_ee_planck = dl_ee_planck * cl_factor
    cl_bb_planck = dl_bb_planck * cl_factor
    
    #Weight by (2l + 1) / 4pi to compute total physical sky variance
    variance_weight_planck = (2 * l_vals + 1) / (4 * np.pi)
    ref_ee = np.sum(cl_ee_planck * variance_weight_planck)
    ref_bb = np.sum(cl_bb_planck * variance_weight_planck)
    
    ref_ratio_str = "> 10^8 (Pure E)" if ref_bb < 1e-6 else f"{ref_ee / ref_bb:.2f}"
    
    results.append({
        "Source": "Planck 2018 Theory (Ref)",
        "EE Power [uK^2]": f"{ref_ee:.4e}",
        "BB Power [uK^2]": f"{ref_bb:.4e}",
        "EE / BB Ratio": ref_ratio_str
    })
else:
    raise FileNotFoundError(
        f"Could not find '{cl_file_path}' in current directory."
    )

export_dir = "results_maps"
map_files = sorted(glob.glob(os.path.join(export_dir, "map_*.npz")))

if not map_files:
    raise FileNotFoundError(f"No .npz files found in '{export_dir}/'")

ell_vals = np.arange(2, lmax + 1)
variance_weight_geom = (2 * ell_vals + 1) / (4 * np.pi)
unit_scale = (1e6)**2 if MAPS_IN_KELVIN else 1.0

for fpath in map_files:
    data = np.load(fpath)
    geom_name = str(data['geometry'])
    
    Q_geom = data['Q']
    U_geom = data['U']
    T_geom = data['T']
    
    alms_geom = hp.map2alm([T_geom, Q_geom, U_geom], lmax=lmax)
    cl_ee = hp.alm2cl(alms_geom[1])
    cl_bb = hp.alm2cl(alms_geom[2])
    
    #Calculation of physical variance (total power) over 2 <= ell <= lmax
    sum_ee = np.sum(cl_ee[2:lmax+1] * variance_weight_geom) * unit_scale
    sum_bb = np.sum(cl_bb[2:lmax+1] * variance_weight_geom) * unit_scale
    
    ratio_str = "> 10^8 (Pure E)" if sum_bb < 1e-12 else f"{sum_ee / sum_bb:.2f}"
        
    results.append({
        "Source": f"Geom: {geom_name}",
        "EE Power [uK^2]": f"{sum_ee:.4e}",
        "BB Power [uK^2]": f"{sum_bb:.4e}",
        "EE / BB Ratio": ratio_str
    })

df = pd.DataFrame(results)
print("\n" + df.to_string(index=False))
