import os
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("TEB/RS2", exist_ok=True)

data = np.load("results_maps/map_rs2.npz")
nside = int(data["nside"])
lmax = 2

t_map = data["T"]
q_map = data["Q"]
u_map = data["U"]

alm_t, alm_e, alm_b = hp.map2alm((t_map, q_map, u_map), lmax=lmax, pol=True)
zero_alm = np.zeros_like(alm_t)

map_pure_t = hp.alm2map((alm_t, zero_alm, zero_alm), nside=nside, lmax=lmax, pol=True)
map_pure_e = hp.alm2map((zero_alm, alm_e, zero_alm), nside=nside, lmax=lmax, pol=True)
map_pure_b = hp.alm2map((zero_alm, zero_alm, alm_b), nside=nside, lmax=lmax, pol=True)

cl_t_out = hp.anafast(map_pure_t, lmax=lmax)
cl_e_out = hp.anafast(map_pure_e, lmax=lmax)
cl_b_out = hp.anafast(map_pure_b, lmax=lmax)

t_uK = map_pure_t[0] * 1e6

q_e_uK = map_pure_e[1] * 1e6
u_e_uK = map_pure_e[2] * 1e6
vmax_e = np.max(np.abs(np.concatenate([q_e_uK, u_e_uK])))

q_b_uK = map_pure_b[1] * 1e6
u_b_uK = map_pure_b[2] * 1e6
vmax_b = np.max(np.abs(np.concatenate([q_b_uK, u_b_uK])))

hp.mollview(t_uK, title="Pure T", unit="µK", cmap="turbo")
plt.gca().set_title("Pure T", fontsize=22)
plt.gcf().axes[1].tick_params(labelsize=20)
for t in plt.gcf().axes[1].texts:
    t.set_fontsize(20)
plt.savefig("TEB/RS2/pure_t.png", dpi=150, bbox_inches="tight")

hp.mollview(q_e_uK, title="Pure E (Q)", unit="µK", cmap="turbo", min=-vmax_e, max=vmax_e)
plt.gca().set_title("Pure E (Q)", fontsize=22)
plt.gcf().axes[1].tick_params(labelsize=20)
for t in plt.gcf().axes[1].texts:
    t.set_fontsize(20)
plt.savefig("TEB/RS2/pure_e(q).png", dpi=150, bbox_inches="tight")

hp.mollview(u_e_uK, title="Pure E (U)", unit="µK", cmap="turbo", min=-vmax_e, max=vmax_e)
plt.gca().set_title("Pure E (U)", fontsize=22)
plt.gcf().axes[1].tick_params(labelsize=20)
for t in plt.gcf().axes[1].texts:
    t.set_fontsize(20)
plt.savefig("TEB/RS2/pure_e(u).png", dpi=150, bbox_inches="tight")

hp.mollview(q_b_uK, title="Pure B (Q)", unit="µK", cmap="turbo", min=-vmax_b, max=vmax_b)
plt.gca().set_title("Pure B (Q)", fontsize=22)
plt.gcf().axes[1].tick_params(labelsize=20)
for t in plt.gcf().axes[1].texts:
    t.set_fontsize(20)
plt.savefig("TEB/RS2/pure_b(q).png", dpi=150, bbox_inches="tight")

hp.mollview(u_b_uK, title="Pure B (U)", unit="µK", cmap="turbo", min=-vmax_b, max=vmax_b)
plt.gca().set_title("Pure B (U)", fontsize=22)
plt.gcf().axes[1].tick_params(labelsize=20)
for t in plt.gcf().axes[1].texts:
    t.set_fontsize(20)
plt.savefig("TEB/RS2/pure_b(u).png", dpi=150, bbox_inches="tight")
