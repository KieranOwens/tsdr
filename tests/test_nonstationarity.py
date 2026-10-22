"""Value tests for the non-stationarity methods."""

import numpy as np
import pytest
import scipy.linalg
from sklearn.exceptions import ConvergenceWarning

from tsdr._whitening import fit_whitening
from tsdr.nonstationarity import BSSnonstat, WaSSAf

from helpers import corr2, random_mix
from reference import wassa_S_ref


@pytest.fixture(scope="module")
def variance_sources():
    """Three sources with smoothly varying variance plus two stationary
    Gaussian sources, randomly mixed."""
    rng = np.random.default_rng(0)
    T = 3000

    def smooth_var_source(period):
        phase = rng.uniform(0, 2 * np.pi)
        env = np.exp(np.sin(2 * np.pi * np.arange(T) / period + phase))
        return env * rng.normal(size=T)

    S = np.column_stack([smooth_var_source(400), smooth_var_source(150),
                         smooth_var_source(900), rng.normal(size=(T, 2))])
    X, _ = random_mix(S, seed=9)
    return S, X


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bssnonstat_recovers_all_sources(variance_sources, seed):
    S, X = variance_sources
    Y = BSSnonstat(n_components=3, max_iters=30,
                   random_state=seed).fit_transform(X)
    R2 = np.array([[corr2(Y[:, i], S[:, k]) for k in range(5)]
                   for i in range(3)])
    # each component is a different non-stationary source
    assert sorted(R2.argmax(1)) == [0, 1, 2]
    assert np.all(R2.max(1) > 0.99)


@pytest.fixture(scope="module")
def strongly_nonstationary_data():
    """Two sources with strongly varying variance (whitened window
    covariances far from the identity) and three stationary sources."""
    rng = np.random.default_rng(1)
    T = 3000
    t = np.arange(T)
    env = lambda p: np.exp(2 * np.sin(2 * np.pi * t / p + rng.uniform(0, 2 * np.pi)))
    S = np.column_stack([env(1500) * rng.normal(size=T),
                         env(700) * rng.normal(size=T),
                         rng.normal(size=(T, 3))])
    X, _ = random_mix(S, seed=9)
    return fit_whitening(X).transform(X)


def test_wassaf_frechet_mean_is_wasserstein_barycenter(strongly_nonstationary_data):
    """The Frechet mean satisfies the fixed-point equation that characterises
    the Wasserstein-2 barycenter of the window covariances,
    Sigma = mean_e (Sigma^1/2 Sigma_e Sigma^1/2)^1/2 (Alvarez-Esteban et
    al., 2016), and S is eq. (21) of Kaltenstadler et al. (2018) for it."""
    Xw = strongly_nonstationary_data
    model = WaSSAf(n_windows=10, epsilon=1e-10)
    S = model._compute_S(Xw)
    sqrtm = lambda A: np.real(scipy.linalg.sqrtm(A))
    bary = model.frechet_mean_
    L = len(Xw) // 10
    Cs = [Xw[i * L:(i + 1) * L].T @ Xw[i * L:(i + 1) * L] / (L - 1)
          for i in range(10)]
    R = sqrtm(bary)
    fixed_point = np.mean([sqrtm(R @ C @ R) for C in Cs], axis=0)
    np.testing.assert_allclose(bary, fixed_point, atol=1e-8)
    np.testing.assert_allclose(S, wassa_S_ref(Xw, 10, bary), atol=1e-8)


def test_wassaf_warns_if_frechet_mean_not_converged(strongly_nonstationary_data):
    with pytest.warns(ConvergenceWarning):
        WaSSAf(epsilon=1e-12, max_iters=1).fit(strongly_nonstationary_data)
