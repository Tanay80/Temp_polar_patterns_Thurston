# Temperature and polarization patterns in Thurston geometries

This repository extends the Bianchi-model formalism of Sung & Coles
(2011, *JCAP* 06:036) to the eight homogeneous Thurston geometries. For each
geometry, it computes the coherent CMB temperature and polarization patterns
produced by a deterministic initial quadrupole, by integrating the Thomson-scattering
radiative-transfer hierarchy along every line of sight.

The goal is to study how spatial geometry alone shapes the temperature maps and the
Stokes parameters Q and U, and how the patterns evolve with redshift. The code does
not compare against observational data.

## Geometries

| # | Geometry | Script | Spatial curvature |
|---|---|---|---|
| 1 | $\mathbb{R}^3$ | `1_R3.py` | Flat |
| 2 | $S^3$ | `2_S3.py` | Positive |
| 3 | $\mathbb{H}^3$ | `3_H3.py` | Negative |
| 4 | $\mathbb{R}\times S^2$ | `4_RS2.py` | Positive |
| 5 | $\mathbb{R}\times\mathbb{H}^2$ | `5_RH2.py` | Negative |
| 6 | $\widetilde{U \left(\mathbb{H}^2 \right)}$ | `6_UH2.py` | Negative |
| 7 | Nil | `7_nil.py` | Negative |
| 8 | Solv | `8_solv.py` | Negative |

## Method

### 1. Symbolic stage (`injector.py`, `generator.py`)

- **`injector.py`** takes a set of commutation functions $\gamma^a{} _{bc}$ for a
  chosen geometry and builds the Ricci rotation coefficients $\Gamma^a{} _{bc}$ in an
  orthonormal tetrad (Sung & Coles (eq. 2.15), being contracted with MInkowski metric), with torsion and metric-compatibility checks. From these, it
  evaluates the Boltzmann coefficients $\hat{A}^k _i$ to $\hat{K}^{kl} _{ij}$ of Sung & Coles
  (eq. 3.25). Coefficients with index pairs $\left(ij \right)$ or $\left(kl \right)$ are projected
  onto their symmetric-traceless (STF) part.
- **`generator.py`** builds on these coefficients to produce the photon geodesic
  equations $\left(\dot\theta, \dot\phi \right)$, the polarization-angle equations
  $\left(\dot\psi \right)$, and implicit expressions for the spectral term $\zeta$ from the
  monopole equation using Cramer's rule.

The output of this stage is pasted into the per-geometry solvers as explicit
expressions.

### 2. Numerical stage (`1_R3.py` ... `8_solv.py`)

Each solver integrates, for every HEALPix pixel ($N_\text{side}=32$, 12288 pixels),
a stiff system of coupled ODEs. The state vector contains:

- the 44 real and imaginary moments $R^0 _\mu, I^0 _\mu, R^0 _{ij}, I^0 _{ij}, R^2 _{ij}, I^2 _{ij}$, and
- the geodesic variables $\left(\Re\left\lbrace\psi\right\rbrace,\ \Im\left\lbrace\psi\right\rbrace,\ \theta,\ \phi,\ \chi\right)$ and,
  where needed, the Cartesian coordinates of the ray.

The total length depends on the geometry (for example 49 for $\mathbb{R}^3$,
51 for $\mathbb{R}\times\mathbb{H}^2$, 52 for Solv).

**Background expansion.** The same background is used in every geometry, so that
differences between maps come only from the spatial geometry:

$$H(a)/H_0 = \sqrt{\Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_\Lambda},$$

with $\Omega_m = 0.315$, $\Omega_k = 0.044$ (Awwad & Prokopec, pure CMB constraints), $\Omega_\Lambda = 1-\Omega_m-\Omega_k$
and $H_0 = 2.56 \times 10^{-18}\,\text{s}^{-1}$. Radiation is neglected. The scale
factor $a$ is the independent variable, with $d/dt = aH\,d/da$. The comoving
distance is $\chi(a) = \int_a^1 da'/[a'^2 H(a')]$.

**Thomson scattering rate.** $\tau = n_e\sigma_T c$ is computed from the opacity of
`CAMB` (full H and He recombination plus reionization with $\tau_\text{reio}=0.054$,
$\Omega_b h^2 = 0.023$). It is tabulated on a grid uniform in $\ln(1+z)$ for
$0\le z\le3000$ and interpolated in log-log during the integration.

**Initial conditions.** The integration starts at $z=1200$ with a deterministic
$\ell=2$ quadrupole. The three cases $m=0, 1, 2$ are integrated separately and their
contributions are summed. The seed amplitudes come from the Sachs-Wolfe integral
$C_2 = \tfrac{4\pi}{25}\int d\ln k\,\mathcal{P}_\mathcal{R}(k)\,j_2^2(k\Delta\eta)$
evaluated with a comoving mean free path $\Delta\eta$ at decoupling.

**Numerics.**
- Symmetry $\left(N^{0,2}_{ij} = N^{0,2}_{ji}\right)$ and tracelessness $\left(\delta^{ij}N^{0,2}_{ij} = 0\right)$ are enforced at every right-hand-side
  evaluation on a *copy* of the state vector. This avoids corrupting the solver's
  finite-difference Jacobian.
- Integration uses SciPy's implicit `Radau` method (`rtol=1e-6`) with
  component-wise absolute tolerances: $10^{-13}$ for the moments, $10^{-10}$ for the
  angles, and $10^{6}$ for the coordinates ($\sim10^{18}$).
- Pixels are distributed over cores with `multiprocessing`.

### 3. Outputs

Maps are saved at timepoints z = 1200, 550, 10 and 0, respectively.

- `maps/<Geometry>_maps.npz` contains `T` (temperature fluctuation, K), `Q_K`, `U_K`
  (K), `I`, `Q`, `U`, `P`, `z` and `nside`.
- `maps/<Geometry>.png` shows Mollweide grids of $\Delta T$, P, Q and U at each
  redshift, with colorbar limits set per redshift from the maximum absolute value.
- Polarization orientation is defined in the local basis
  $\hat e_\theta=(\cos\theta \cos\phi, \cos\theta \sin\phi, -\sin\theta)$,
  $\hat e_\phi=(-\sin\phi, \cos\phi,0)$. $Q>0$ is along $\hat e_\theta$ and $Q<0$
  along $\hat e_\phi$, after rotation by $2\psi$.

## Repository structure

```
injector.py                                             #Gamma^a_bc and STF-projected Boltzmann coefficients from commutation functions
generator.py                                            #Geodesic & polarization equations and zeta expressions
initial.py                                              #Initial conditions/ seed amplitudes
1_R3.py ... 8_solv.py                                   #Per-geometry ODE solvers and map generation
maps/                                                   #Output .png figures and .npz files
```

## Usage

```bash
python 1_R3.py ... 8_solv.py                            #Each for a single geometry
```

Output goes to `maps/`.

## Requirements

```
numpy
scipy
sympy
healpy
matplotlib
tqdm
camb
```
