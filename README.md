# Temperature and Polarization Patterns in Exotic Anisotropic Geometries

This repository extends the Bianchi-model formalism of Sung & Coles
(2011, *JCAP* 06:036) to five homogeneous but anisotropic 3-geometries
drawn from Thurston's classification, and tests whether the coherent
CMB temperature/polarization patterns they produce can address any of
the well-known large-scale CMB anomalies.

## Geometries studied

| Symbol | Description | Curvature |
|---|---|---|
| $\mathbb{R}\times\mathbb{H}^2$ | `RH2.py` | Negative |
| $\mathbb{R}\times S^2$ | `RS2.py` | Positive |
| $\widetilde{U \left(\mathbb{H}^2 \right)}$ | `UH2.py` | Negative |
| Nil | `nil.py` | Negative |
| Solv | `solv.py` | Negative |

Each solver numerically integrates the Boltzmann/Thomson-scattering
radiative transfer hierarchy (truncated at $l\leq2$, following the
physical justification that Thomson scattering damps *and* re-radiates
only the monopole/dipole/quadrupole, while higher moments are only
damped) over a HEALPix pixel grid, seeded with a single deterministic
$(l=2,m=2)$ initial quadrupole.

## Repository structure

```
RH2.py, RS2.py, UH2.py, nil.py, solv.py   # per-geometry ODE solvers
helper_code.py                             # shared utility functions
run_geometries.py                          # runs all five solvers in sequence
maps_summary.py                            # per-geometry T/Q/U/P map grids (redshift evolution)
teb_nil.py, teb_rh2.py, teb_rs2.py,
teb_solv.py, teb_uh2.py                    # per-geometry T/E/B decomposition (development versions)
teb.py                                     # unified T/E/B decomposition across all five geometries
unified_cosmic_anomaly_analysis.py         # the seven-point anomaly diagnostic suite
skymap_comparison_vs_planck.py             # model-vs-Planck sky map grid + residuals
MASTER_BUTTON.py                           # runs the full pipeline end-to-end
```

### Data dependencies (not tracked in git -- download separately)

- `COM_CMB_IQU-SMICA_2048_R3.00_full.fits` -- Planck 2018 SMICA
  temperature map, $N_\mathrm{side}=2048$. [Planck Legacy Archive](https://pla.esac.esa.int)
- `COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt`
  -- Planck 2018 fiducial theory power spectrum. [IRSA mirror](https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/)

Both files are large; add them to `.gitignore` and document the exact
download path in your local setup rather than committing them.

## Pipeline

Run the entire analysis end-to-end with:

```bash
python MASTER_BUTTON.py
```

which executes, in order:
1. `run_geometries.py` -- solves all five geometries, saves `results_maps/map_*.npz`
2. `maps_summary.py` -- generates per-geometry redshift-evolution map grids
3. `unified_cosmic_anomaly_analysis.py` -- the seven-point diagnostic suite (see below)
4. `skymap_comparison_vs_planck.py` -- model-vs-Planck comparison figures
5. `teb.py` -- T/E/B harmonic decomposition and cross-correlation

### The seven-point anomaly diagnostic

`unified_cosmic_anomaly_analysis.py` tests:

1. Quadrupole amplitude anomaly ($C_2^{TT}$ vs. Planck/theory)
2. Quadrupole axis alignment (vs. the CMB kinematic dipole)
3. Hemispherical power asymmetry
4. Localized features (Cold-Spot-like compactness proxy)
5. Coherence of harmonic modes (participation ratio $N_\mathrm{eff}$)
6. Cross-geometry systematics (summary across all five geometries)
7. Large-scale $B$-mode consistency (vs. the Planck 2018 low-$l$ bound)

The observational $B$-mode bound used in point 7 is derived via CAMB
from Planck 2018 V's low-$l$-only constraint $r<0.41$ (95% CL) --
see `snippet_to_unified_anomaly_checker.txt` for the exact derivation,
which should be run once with `camb` installed to obtain
`BB_UPPER_LIMIT_C2` before trusting point 7's output.

## Requirements

```
numpy
scipy
healpy
astropy
matplotlib
camb        #Only needed to regenerate the B-mode bound
```

## References

- Sung, R. & Coles, P., *Temperature and polarization patterns in
  anisotropic cosmologies*, JCAP 06 (2011) 036.
- Planck Collaboration, *Planck 2018 results. III. High Frequency
  Instrument data processing*, arXiv:1807.06207.
- Planck Collaboration, *Planck 2018 results. V. CMB power spectra and
  likelihoods*, arXiv:1907.12875.
- BICEP/Keck Collaboration, *Improved Constraints on Primordial
  Gravitational Waves...*, Phys. Rev. Lett. 127, 151301 (2021).
