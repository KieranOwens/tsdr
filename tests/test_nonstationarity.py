"""Value tests for the non-stationarity methods."""

import numpy as np
import pytest
import scipy.linalg
import scipy.stats
from sklearn.exceptions import ConvergenceWarning

from tsdr._whitening import fit_whitening
from tsdr.nonstationarity import (ASSA, SSAsave, SSAcor, BSSnonstat,
                                  WaSSAf, WaSSAr)

from helpers import corr2, random_mix
from reference import wassa_S_ref, wassar_S_ref


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


@pytest.mark.parametrize("seed, random_state", [(0, 2), (2, 4), (3, 9)])
def test_bssnonstat_random_starts_avoid_local_optima(seed, random_state):
    """A sinusoid (cross-cumulant -1.5) among nine channels of low-rank 
    noise. For these data and random states, a single random start converges
    to a noise direction whose cross-cumulant is nonzero only through 
    sampling error (a local optimum); with several starts BSSnonstat 
    recovers the sinusoid."""
    rng = np.random.default_rng(seed)
    T, D = 1000, 10
    z = np.sqrt(2) * np.sin(np.linspace(0, 4 * np.pi, T))
    A = scipy.stats.ortho_group.rvs(D, random_state=rng)
    Q = scipy.stats.ortho_group.rvs(D - 1, random_state=rng)
    noise = rng.normal(size=(T, D - 1)) * np.sqrt(np.r_[np.full(D - 2, 0.1), 0])
    X = np.column_stack([z, noise @ Q.T]) @ A[:, :D - 1]
    single = BSSnonstat(max_iters=10, n_init=1,
                        random_state=random_state).fit_transform(X)
    multiple = BSSnonstat(max_iters=10, random_state=random_state).fit_transform(X)
    assert corr2(single, z) < 0.1
    assert corr2(multiple, z) > 0.99


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
    Cs = [np.cov(Xw[i * L:(i + 1) * L], rowvar=False) for i in range(10)]
    R = sqrtm(bary)
    fixed_point = np.mean([sqrtm(R @ C @ R) for C in Cs], axis=0)
    np.testing.assert_allclose(bary, fixed_point, atol=1e-8)
    np.testing.assert_allclose(S, wassa_S_ref(Xw, 10, bary), atol=1e-8)


def test_wassaf_warns_if_frechet_mean_not_converged(strongly_nonstationary_data):
    with pytest.warns(ConvergenceWarning):
        WaSSAf(epsilon=1e-12, max_iters=1).fit(strongly_nonstationary_data)


def test_wassar_matches_eq23(strongly_nonstationary_data):
    Xw = strongly_nonstationary_data
    S = WaSSAr(n_windows=10)._compute_S(Xw)
    np.testing.assert_allclose(np.real(S), wassar_S_ref(Xw, 10), atol=1e-8)


@pytest.mark.parametrize("method", [WaSSAf, WaSSAr])
def test_wassa_window_mean_enters_only_through_mean_term(
        strongly_nonstationary_data, method):
    """Each window is modelled as N(mu_e, Sigma_e) (eq. 19), so shifting the
    mean of one window changes S only through mu_e mu_e^T, and the
    covariance terms (and the barycenter) are unaffected."""
    Xw = strongly_nonstationary_data
    L = len(Xw) // 10
    shift = np.array([3.0, -2.0, 0.0, 1.0, 0.5])
    Xs = Xw.copy()
    Xs[2 * L:3 * L] += shift
    mu = Xw[2 * L:3 * L].mean(0)
    kw = dict(n_windows=10, epsilon=1e-10) if method is WaSSAf else dict(n_windows=10)
    dS = method(**kw)._compute_S(Xs) - method(**kw)._compute_S(Xw)
    np.testing.assert_allclose(
        np.real(dS), np.outer(mu + shift, mu + shift) - np.outer(mu, mu),
        atol=1e-6)


@pytest.mark.parametrize("method, compute", [
    (ASSA, "_compute_S"), (SSAsave, "_compute_Mv"), (SSAcor, "_compute_Mtau")])
def test_ssa_window_covariances_are_centred(
        strongly_nonstationary_data, method, compute):
    """The window (auto)covariances are computed about the window means
    (Hara et al., 2012; Flumian et al., 2024, Sec. 2): shifting the mean of
    one window leaves SSAsave's Mv unchanged and changes ASSA's S only
    through its m_i m_i^T term, and adding a constant to the series leaves
    SSAcor's Mtau unchanged."""
    Xw = strongly_nonstationary_data
    n = 10
    L = len(Xw) // n
    shift = np.array([3.0, -2.0, 0.0, 1.0, 0.5])
    Xs = Xw.copy()
    if method is SSAcor:
        Xs += shift
    else:
        Xs[2 * L:3 * L] += shift
    mu = Xw[2 * L:3 * L].mean(0)
    model = method(n_windows=n)
    dM = getattr(model, compute)(Xs) - getattr(model, compute)(Xw)
    expected = np.zeros_like(dM)
    if method is ASSA:
        expected = (np.outer(mu + shift, mu + shift) - np.outer(mu, mu)) / n
    np.testing.assert_allclose(dM, expected, atol=1e-10)
