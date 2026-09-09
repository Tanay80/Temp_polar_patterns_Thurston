import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

GEOMETRIES = {
    r"$\mathbb{R} \times \mathbb{H}^2$": "results_maps/map_rh2.npz",
    r"$\mathbb{R} \times S^2$": "results_maps/map_rs2.npz",
    r"$\widetilde{U \left(\mathbb{H}^2 \right)}$":  "results_maps/map_uh2.npz",
    "Nil": "results_maps/map_nil.npz",
    "Solv": "results_maps/map_solv.npz",
}
PLANCK_MAP = "COM_CMB_IQU-SMICA_2048_R3.00_full.fits"
PCTL = 99.5
GRID_ROWS, GRID_COLS = 2, 3

models = {}
for name, path in GEOMETRIES.items():
    d = np.load(path)
    models[name] = (d["T"] * 1e6, int(d["nside"]))                                                                  #K -> uK

nside_ref = models["Nil"][1]                                                                                        #Reference, can be taken any geometry

#Loading + downgrading Planck T map to the same resolution
T_pl = hp.read_map(PLANCK_MAP, field=0) * 1e6   # K_CMB -> uK
T_pl = hp.ud_grade(T_pl, nside_ref)

fig = plt.figure(figsize=(4 * GRID_COLS, 3.5 * GRID_ROWS))

def plot_robust_mollview(T_map, title, sub, target_fig, cmap="turbo"):
    vmax = np.nanpercentile(np.abs(T_map), PCTL)
    hp.mollview(T_map, cmap=cmap, min=-vmax, max=vmax,
                title=title, sub=sub, cbar=True, unit="uK", fig=target_fig.number)

panel_order = ["Reference"] + list(GEOMETRIES.keys())
plot_robust_mollview(T_pl, "Reference (Planck SMICA)", (GRID_ROWS, GRID_COLS, 1), fig)

for i, name in enumerate(GEOMETRIES.keys(), start=2):
    T_map, nside = models[name]
    plot_robust_mollview(T_map, name, (GRID_ROWS, GRID_COLS, i), fig)

plt.savefig("model_vs_planck_Tmaps.png", dpi=200, bbox_inches="tight")

fig2 = plt.figure(figsize=(4 * GRID_COLS, 3.5 * GRID_ROWS))

for i, name in enumerate(GEOMETRIES.keys(), start=1):
    T_map, nside = models[name]
    residual = T_map - T_pl
    plot_robust_mollview(residual, f"{name}", (GRID_ROWS, GRID_COLS, i), fig2,
                          cmap="coolwarm")

plt.savefig("model_minus_planck_residuals.png", dpi=200, bbox_inches="tight")
