import os
import glob
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt

fits_filename = "COM_CMB_IQU-SMICA_2048_R3.00_full.fits"
if not os.path.exists(fits_filename):
    raise FileNotFoundError(f"'{fits_filename}' not found.")

TARGET_NSIDE = 32
lmax = 3 * TARGET_NSIDE - 1

planck_T_raw = hp.read_map(fits_filename, field=0)
planck_T = hp.ud_grade(planck_T_raw, TARGET_NSIDE)
planck_T -= np.mean(planck_T)
planck_var = np.var(planck_T)

def compute_C_theta_and_S12(T_map, target_var=None):
    T_map = T_map - np.mean(T_map)
    current_var = np.var(T_map)
    
    if target_var is not None and current_var > 0:
        T_map = T_map * np.sqrt(target_var / current_var)
        
    cl = hp.anafast(T_map, lmax=lmax)
    
    theta_deg = np.linspace(0, 180, 181)
    theta_rad = np.radians(theta_deg)
    cos_theta = np.cos(theta_rad)
    
    #Computing C(theta) using Legendre polynomials sum
    C_theta = np.zeros_like(theta_rad)
    for l_val in range(2, lmax + 1):
        P_l = np.polynomial.legendre.Legendre.basis(l_val)(cos_theta)
        C_theta += (2 * l_val + 1) * cl[l_val] * P_l
    C_theta /= (4 * np.pi)
    
    #Converting C(theta) to uK^2 for standard cosmological units
    C_theta_uK2 = C_theta * (1e6)**2 
    
    # Integration
    mask_60 = cos_theta <= 0.5
    S12 = np.trapezoid(C_theta_uK2[mask_60]**2, -cos_theta[mask_60])
    
    return theta_deg, C_theta_uK2, S12

theta_deg, C_planck, S12_planck = compute_C_theta_and_S12(planck_T)

results = [{
    "Source": "Planck 2018",
    "S_1/2 [uK^4]": f"{S12_planck:.2f}",
    "Ratio to Planck": "1.00"
}]

plt.figure(figsize=(10, 6))
plt.plot(theta_deg, C_planck, 'k-', lw=2.5, label=f'Planck 2018 ($S_{{1/2}}={S12_planck:.1f}$)')

export_dir = "results_maps"
map_files = sorted(glob.glob(os.path.join(export_dir, "map_*.npz")))

for fpath in map_files:
    data = np.load(fpath)
    geom_name = str(data['geometry'])
    
    _, C_geom, S12_geom = compute_C_theta_and_S12(data['T'], target_var=planck_var)
    ratio = S12_geom / S12_planck if S12_planck > 0 else np.nan
    
    results.append({
        "Source": f"Geom: {geom_name}",
        "S_1/2 [uK^4]": f"{S12_geom:.2f}",
        "Ratio to Planck": f"{ratio:.2f}"
    })
    
    plt.plot(theta_deg, C_geom, '--', lw=1.5, label=f'{geom_name} ($S_{{1/2}}={S12_geom:.1f}$)')

plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
plt.axvline(60, color='red', linestyle='--', alpha=0.5, label=r'$\theta = 60^\circ$ cutoff')
plt.xlabel(r'Separation Angle $\theta$ [degrees]', fontsize=12)
plt.ylabel(r'$C(\theta)$ $[\mu\mathrm{K}^2]$', fontsize=12)
plt.title(r'Angular Correlation Function $C(\theta)$ and $S_{1/2}$ Comparison', fontsize=13, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=9, loc='upper right')
plt.tight_layout()

os.makedirs("maps", exist_ok=True)
plt.savefig("maps/Objective_C_Correlation_Function_S12.png", dpi=250)
plt.close()

df = pd.DataFrame(results)
print(df.to_string(index=False))
