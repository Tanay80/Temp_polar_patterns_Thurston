# CMB Polarization Patterns in Thurston Geometries

This repository contains Python scripts to simulate and analyze CMB temperature and polarization patterns across different spatial Thurston geometries.

## Repository Architecture

* **Geometry Scripts:** Individual Python modules dedicated to specific spatial geometries.
* **`helper_code.py`:** A core utility script that takes the user-supplied vierbein matrix to calculate the Ricci rotation coefficients ($\Gamma$'s) and subsequently the Boltzmann coefficients ($A, B, C, \dots, K$) directly from them.
* **Observational Testing:** Dedicated scripts used to run numerical comparisons and tests against observational data.

---

## Observational Data Sources

To construct **Tables \ref{tab:cmb_observables}--\ref{tab:parity_stat}** and **Table \ref{tab:eb_decomposition}$**, this project uses full-sky astronomical data from the Planck 2018 data release. Due to their large size, these data files are omitted from the repository and must be downloaded manually:

### 1. CMB Map (For Anomaly Tests)
* **Description:** Planck 2018 SMICA component-separated CMB map with no SZ subtraction at an $N_{\text{side}} = 2048$ HEALPix resolution.
* **Download URL:** [Zenodo Record](https://zenodo.org/records/16283859/files/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits?download=1)

### 2. Reference Power Spectra (For $E$-$B$ Decomposition)
* **Description:** Planck 2018 best-fit theory power spectrum.
* **Download URL:** [IPAC Caltech](https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt)

---

## Quick Setup & Usage

1. Install the required Python dependencies:
   ```bash
   pip install numpy healpy pandas
