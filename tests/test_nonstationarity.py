"""Value tests for the non-stationarity methods."""

import numpy as np
import pytest
import scipy.linalg
import scipy.stats
from sklearn.exceptions import ConvergenceWarning

from tsdr._whitening import fit_whitening
from tsdr.nonstationarity import (ASSA, SSAsir, SSAsave, SSAcor, BSSnonstat,
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


@pytest.mark.parametrize("nonstationary", [True, False])
def test_assa_eigvals_in_component_order(variance_sources, nonstationary):
    """eigvals_ lists the eigenvalues of S in the order of the components:
    S restricted to the components (centred, since transform does not 
    centre) is diagonal with these eigenvalues."""
    _, X = variance_sources
    D = X.shape[1]
    model = ASSA(n_components=D, nonstationary=nonstationary).fit(X)
    Y = model.transform(X)
    S_components = model._compute_S(Y - Y.mean(axis=0))
    lam = np.diag(S_components)
    np.testing.assert_allclose(S_components, np.diag(lam), atol=1e-8)
    if nonstationary:
        # most nonstationary (largest eigenvalue) first
        assert np.all(np.diff(lam) < 0)
        np.testing.assert_allclose(model.eigvals_, 1 - lam / lam.max(), atol=1e-8)
    else:
        # most stationary (least eigenvalue) first
        assert np.all(np.diff(lam) > 0)
        np.testing.assert_allclose(model.eigvals_, lam, atol=1e-8)


def test_assa_truthy_nonstationary_sets_eigvals(variance_sources):
    """A truthy value other than True (e.g., 1) behaves like True."""
    _, X = variance_sources
    a = ASSA(n_components=2, nonstationary=1).fit(X)
    b = ASSA(n_components=2, nonstationary=True).fit(X)
    np.testing.assert_array_equal(a.eigvals_, b.eigvals_)
    np.testing.assert_array_equal(a.linear_transform_, b.linear_transform_)


@pytest.mark.parametrize("method", [WaSSAf, WaSSAr])
def test_wassa_is_real_with_singular_window_covariances(method):
    """A channel that is constant in half of the windows makes their
    covariances singular, for which scipy.linalg.sqrtm returns complex values
    (and WaSSAf and WaSSAr used to fail); the symmetric square root keeps the
    computation real."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 4)) * np.linspace(1, 3, 400)[:, None]
    X[:200, 3] = 0.0
    model = method(n_components=2, n_windows=10).fit(X)
    Y = model.transform(X)
    assert np.isrealobj(model.linear_transform_) and np.isrealobj(Y)
    assert np.all(np.isfinite(Y))


def test_wassar_is_real_with_short_windows():
    """Windows with fewer samples than channels have singular covariances."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6)) * np.linspace(1, 3, 40)[:, None]
    Y = WaSSAr(n_components=2, n_windows=8).fit_transform(X)
    assert np.isrealobj(Y) and np.all(np.isfinite(Y))


def test_wassaf_needs_windows_longer_than_n_features():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_windows"):
        WaSSAf(n_windows=8).fit(rng.normal(size=(40, 6)))


WINDOW_METHODS = [ASSA, SSAsir, SSAsave, SSAcor, WaSSAf, WaSSAr]


@pytest.mark.parametrize("method", WINDOW_METHODS)
def test_window_methods_warn_when_samples_are_dropped(method):
    """The windows have n_samples // n_windows samples each, so with 1003
    samples and 10 windows the last 3 samples are not used, with a warning
    (and with 1000 samples, no warning)."""
    import warnings
    X = np.random.default_rng(0).normal(size=(1003, 3))
    with pytest.warns(UserWarning, match="last 3 samples are not used"):
        method(n_windows=10).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        method(n_windows=10).fit(X[:1000])
