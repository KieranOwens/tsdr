<h1 align="center"><em>tsdr</em>: Time-Series Dimension Reduction</h1>

<p align="center"><img src="img/tsdr_categories.png" height="300" alt="The seven categories of time-series dimension reduction methods"/></p>

<p align="center">
  <a href="https://doi.org/10.36227/techrxiv.176409662.29359239/v1"><img src="https://img.shields.io/badge/paper-TechRxiv%20(2025)-informational" alt="Paper"/></a>
  <a href="https://pypi.org/project/tsdr/"><img src="https://img.shields.io/pypi/v/tsdr?label=PyPI" alt="PyPI version"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"/></a>
</p>

_tsdr_ is a Python package of time-series dimension reduction (TSDR) methods: algorithms that reduce multivariate data to a few components by exploiting temporal structure.
In contrast to general dimension reduction (GDR) methods, whose outputs are invariant to temporal permutation, TSDR methods are sensitive to temporal structure.
The package implements 30 methods from the seven categories of TSDR methods defined in [Owens & Fulcher (2025)](https://doi.org/10.36227/techrxiv.176409662.29359239/v1), written in a simple, readable style to help time-series analysts understand the algorithms discussed in that review.

## Installation

```bash
pip install tsdr
```

The package is on [PyPI](https://pypi.org/project/tsdr/) and requires Python 3.8 or later.
It depends on `numpy`, `scipy`, `scikit-learn`, `pywavelets` and `kneed`.
The example below also uses `matplotlib`.

## Quick start

Every method is a class with a scikit-learn-style `fit`, `transform` and `fit_transform`.
Here we extract a slow sinusoid from a linear mixture of noise channels using slow feature analysis (SFA):

```python
from tsdr.slowness import SFA

import numpy as np
import scipy
import matplotlib.pyplot as plt

# create a dataset for analysis
# a sinusoid
z = np.sin(np.linspace(0, 4*np.pi, 1000))
# multichannel noise
D = 5
noise = np.random.normal(scale=1.0, size=((len(z), D-1)))
# linear mixing
A = scipy.stats.ortho_group.rvs(D, random_state=1)
X = np.concatenate([z.reshape(-1,1), noise], axis=1) @ A

# extract a slow component using SFA
slow_component = SFA(n_components=1).fit_transform(X)

# visualize the component
fig, axs = plt.subplots(6, 1, figsize=(8, 8))
axs[0].plot(slow_component, lw=1.0)
axs[0].set_ylabel('SFA1')
for i in range(D):
    axs[i].spines[['right', 'top', 'bottom']].set_visible(False)
    axs[i].set_xticks([])
    axs[i+1].plot(X[:, i], c='black', linewidth=0.5)
    axs[i+1].set_ylabel(f'X{i+1}')
axs[D].spines[['right', 'top']].set_visible(False)
axs[D].set_xlabel('Time')
plt.tight_layout()
```

<p align="center"><img src="img/sfa_example.png" height="600" alt="SFA recovers the slow sinusoid from five mixed channels"/></p>

## Available methods

The methods are grouped by the aspect of temporal structure that they exploit.
Each is imported from the module named in the first column, e.g., `from tsdr.slowness import SFA`.

| Category | Module | Methods |
|---|---|---|
| <sub>(a) Slowness</sub> | <sub>`tsdr.slowness`</sub> | <sub>`SFA` (slow feature analysis)</sub><br><sub>`BioSFA` (biologically plausible SFA)</sub><br><sub>`ISFA` (independent slow feature analysis)</sub><br><sub>`SICA` (slow independent component analysis)</sub> |
| <sub>(b) Autocorrelation</sub> | <sub>`tsdr.autocorrelation`</sub> | <sub>`TICA` (time-lagged independent component analysis)</sub><br><sub>`TMCA` (time-lagged maximum covariance analysis)</sub><br><sub>`CSA` (colored subspace analysis)</sub><br><sub>`TCCA_Stiefel` (time-lagged canonical component analysis by trace optimization on the Stiefel manifold)</sub><br><sub>`sPCA_dwt` (spectral PCA using the discrete wavelet transform)</sub> |
| <sub>(c) Predictability</sub> | <sub>`tsdr.predictability`</sub> | <sub>`TLPC` (temporally local predictive coding)</sub><br><sub>`PrCA` (predictable component analysis)</sub><br><sub>`DiCCA` (dynamic inner canonical correlation analysis)</sub><br><sub>`ForeCA` (forecastable component analysis)</sub><br><sub>`PFA` (predictable feature analysis)</sub><br><sub>`PTS` (principal time series)</sub><br><sub>`DCA` (dynamical components analysis)</sub> |
| <sub>(d) Determinism</sub> | <sub>`tsdr.determinism`</sub> | <sub>`DyCA` (dynamical component analysis)</sub><br><sub>`DMD` (dynamic mode decomposition)</sub> |
| <sub>(e) Non-stationarity</sub> | <sub>`tsdr.nonstationarity`</sub> | <sub>`ASSA` (analytic stationary subspace analysis)</sub><br><sub>`SSAsir` (stationary subspace analysis via sliced inverse regression)</sub><br><sub>`SSAsave` (stationary subspace analysis via sliced average variance estimation)</sub><br><sub>`SSAcor` (stationary subspace analysis via time-lagged autocorrelation)</sub><br><sub>`WaSSAf` (Wasserstein stationary subspace analysis using the Fréchet mean)</sub><br><sub>`WaSSAr` (Wasserstein stationary subspace analysis using the root-Fréchet mean)</sub><br><sub>`BSSnonstat` (blind source separation via non-stationarity of variances)</sub> |
| <sub>(f) Diffusion</sub> | <sub>`tsdr.diffusion`</sub> | <sub>`DMcov` (diffusion maps using covariance distance)</sub><br><sub>`DIG` (dynamical information geometry)</sub><br><sub>`NLSA` (nonlinear Laplacian spectral analysis)</sub> |
| <sub>(g) Latent variable</sub> | <sub>`tsdr.latentvariable`</sub> | <sub>`LDS` (latent linear dynamical system)</sub><br><sub>`DFA` (dynamic factor analysis)</sub> |

The figure at the top summarizes these categories.

A few methods differ from the common pattern:

- Methods with a random initialization or random order of updates (`BioSFA`, `ISFA`, `SICA`, `ForeCA`, `DCA`, `BSSnonstat`, `LDS`, `DFA`) accept a `random_state` argument for reproducible results; `TCCA_Stiefel` uses a `seed` argument.
- The diffusion methods are nonlinear, so they have `fit_transform` but no separate `transform`. `DMcov` and `DIG` embed sliding windows of the time series (of length `window`, every `step` samples), so they return one row per window; `NLSA` embeds the time series with `q` lags, so it returns `n_samples - q` rows (for time points `q` onwards).

## Citation

If you use this software, please read and cite the open-access article:

> Owens, K. S. & Fulcher, B. D. (2025). Time-series dimension reduction: a comprehensive review and conceptual unification of algorithms. _TechRxiv_. [doi:10.36227/techrxiv.176409662.29359239/v1](https://doi.org/10.36227/techrxiv.176409662.29359239/v1)

```bibtex
@article{owens2025tsdr,
  title   = {Time-series dimension reduction: a comprehensive review and conceptual unification of algorithms},
  author  = {Owens, Kieran S. and Fulcher, Ben D.},
  journal = {TechRxiv},
  year    = {2025},
  doi     = {10.36227/techrxiv.176409662.29359239/v1}
}
```

## License

_tsdr_ is released under the [MIT License](LICENSE.txt).
