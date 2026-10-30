"""Value tests for the slowness methods."""

import numpy as np
import pytest
from sklearn.decomposition import FastICA

from tsdr.slowness import SFA, BioSFA, ISFA, SICA
from tsdr._whitening import fit_whitening

from helpers import corr2, subspace_cos, random_mix, ar1


@pytest.fixture(scope="module")
def slow_sources():
    """Three AR(1) sources of decreasing slowness and three white-noise
    sources, randomly mixed."""
    rng = np.random.default_rng(0)
    T = 5000
    S = np.column_stack([ar1(T, .999, rng), ar1(T, .99, rng),
                         ar1(T, .95, rng), rng.normal(size=(T, 3))])
    X, _ = random_mix(S, seed=1)
    return S, X


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_biosfa_matches_sfa_components(slow_sources, seed):
    """BioSFA learns the slow subspace, and ordering within it by slowness
    recovers each SFA component (not an arbitrary rotation of them),
    independently of the random initialisation."""
    _, X = slow_sources
    sfa = SFA(n_components=3).fit(X)
    bio = BioSFA(n_components=3, random_state=seed).fit(X)
    assert np.all(subspace_cos(bio.linear_transform_,
                               sfa.linear_transform_) > 0.9999)
    Ys, Yb = sfa.transform(X), bio.transform(X)
    for i in range(3):
        assert corr2(Yb[:, i], Ys[:, i]) > 0.999


def test_biosfa_unit_variance_and_ordered(slow_sources):
    """The outputs y = M^-1 W x satisfy the whitening constraint, and are
    ordered from slowest to fastest."""
    _, X = slow_sources
    bio = BioSFA(n_components=3, random_state=0).fit(X)
    Y = bio.transform(X)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), np.eye(3), atol=1e-3)
    assert np.all(np.diff(bio.eigvals_) > 0)
    slowness = np.var(np.diff(Y, axis=0), axis=0)
    assert np.all(np.diff(slowness) > 0)


@pytest.mark.parametrize("lr, tau", [(0.1, 0.1), (0.2, 0.1), (0.0, 1.0)])
def test_biosfa_requires_lr_below_tau(slow_sources, lr, tau):
    _, X = slow_sources
    with pytest.raises(ValueError, match="lr must satisfy"):
        BioSFA(lr=lr, tau=tau).fit(X)


def test_sfa_eigvals_match_component_slowness(slow_sources):
    """eigvals_ is the derivative variance of each (unit-variance) slow
    feature, in the same (ascending) order as the components."""
    _, X = slow_sources
    model = SFA(n_components=3).fit(X)
    Y = model.transform(X)
    assert model.eigvals_.shape == (X.shape[1],)
    assert np.all(np.diff(model.eigvals_) > 0)
    np.testing.assert_allclose(np.var(np.diff(Y, axis=0), axis=0, ddof=1),
                               model.eigvals_[:3])


@pytest.fixture(scope="module")
def equal_lag1_sources():
    """Two sources with equal lag-1 autocorrelation (0.8) but different
    autocorrelation at longer lags (AR(1) with phi = 0.8, and AR(2) with
    a1 = 1.04, a2 = -0.3), and four white-noise sources, randomly mixed.
    SFA cannot separate the two sources, as they are equally slow."""
    rng = np.random.default_rng(0)
    T = 20000
    s1 = ar1(T, 0.8, rng)
    s2 = np.zeros(T)
    e = rng.normal(size=T)
    for t in range(2, T):
        s2[t] = 1.04 * s2[t - 1] - 0.3 * s2[t - 2] + e[t]
    S = np.column_stack([s1 / s1.std(), s2 / s2.std(), rng.normal(size=(T, 4))])
    X, _ = random_mix(S, seed=3)
    return S, X


def lagged_cross_correlation(Y, lags):
    """Sum of the squared off-diagonal symmetrised time-lagged correlations
    of the standardised components Y (the ICA part of the ISFA objective)."""
    Y = (Y - Y.mean(0)) / Y.std(0)
    total = 0
    for lag in lags:
        C = Y[:-lag].T @ Y[lag:] / (len(Y) - lag)
        C = 0.5 * (C + C.T)
        total += np.sum(C ** 2) - np.sum(np.diag(C) ** 2)
    return total


def test_isfa_separates_equally_slow_sources(equal_lag1_sources):
    """ISFA separates two equally slow sources by their independence over
    several time lags, and its components have smaller time-lagged
    cross-correlations than those of SFA."""
    S, X = equal_lag1_sources
    model = ISFA(n_components=2, random_state=0).fit(X)
    Y = model.transform(X)
    R2 = np.array([[corr2(Y[:, i], S[:, k]) for k in range(2)] for i in range(2)])
    assert np.all(R2.max(axis=0) > 0.99)
    assert sorted(R2.argmax(axis=0)) == [0, 1]
    Y_sfa = SFA(n_components=2).fit_transform(X)
    lags = range(1, 11)
    assert lagged_cross_correlation(Y, lags) < lagged_cross_correlation(Y_sfa, lags)


def test_isfa_givens_rotation_objective(equal_lag1_sources):
    """As a function of the rotation angle in a plane, the ISFA objective is
    a trigonometric polynomial in 2 phi and 4 phi (Blaschke et al., 2007,
    eqs. 4.10-4.11), and the optimal angle minimises it."""
    _, X = equal_lag1_sources
    Xw = fit_whitening(X).transform(X)
    T = len(Xw)
    C = []
    for lag in [1, 1, 2, 3]:
        Ctau = Xw[:-lag].T @ Xw[lag:] / (T - lag - 1)
        C.append(0.5 * (Ctau + Ctau.T))
    C = np.array(C)
    model = ISFA(n_components=2, lags=3)
    f = lambda phi: model._objective(model._rotate(C, 0, 3, phi)[0])
    angles = np.linspace(-np.pi / 2, np.pi / 2, 9)
    basis = np.column_stack([np.ones(9), np.cos(2 * angles), np.sin(2 * angles),
                             np.cos(4 * angles), np.sin(4 * angles)])
    coef = np.linalg.lstsq(basis, [f(a) for a in angles], rcond=None)[0]
    np.testing.assert_allclose(basis @ coef, [f(a) for a in angles], atol=1e-12)
    phi = model._optimal_angle(C, 0, 3)
    assert f(phi) <= min(f(a) for a in np.linspace(-np.pi / 2, np.pi / 2, 1001)) + 1e-12


def test_isfa_objective_improves_on_sfa(equal_lag1_sources):
    """ISFA starts from the SFA solution, and the Givens rotations do not
    increase the objective."""
    _, X = equal_lag1_sources
    model = ISFA(n_components=2, random_state=0).fit(X)
    Xw = fit_whitening(X).transform(X)
    W_sfa = np.linalg.eigh(0.5 * (Xw[:-1].T @ Xw[1:] + Xw[1:].T @ Xw[:-1]))[1][:, ::-1]
    T = len(Xw)
    C = []
    for lag in range(0, 11):
        lag = max(lag, 1)
        Ctau = Xw[:-lag].T @ Xw[lag:] / (T - lag - 1)
        C.append(0.5 * (Ctau + Ctau.T))
    C = W_sfa.T @ np.array(C) @ W_sfa
    assert model.objective_ <= model._objective(C) + 1e-12


def test_isfa_all_components_recover_sources(equal_lag1_sources):
    """With as many components as features, ISFA is a second-order ICA that
    recovers each source with distinct temporal structure."""
    S, X = equal_lag1_sources
    Y = ISFA(n_components=X.shape[1], random_state=0).fit_transform(X)
    for k in range(2):
        assert max(corr2(Y[:, i], S[:, k]) for i in range(Y.shape[1])) > 0.99


def test_isfa_independent_of_pair_order(equal_lag1_sources):
    _, X = equal_lag1_sources
    a = ISFA(n_components=2, random_state=0).fit(X)
    b = ISFA(n_components=2, random_state=1).fit(X)
    assert np.all(subspace_cos(a.linear_transform_[:, :1],
                               b.linear_transform_[:, :1]) > 0.9999)
    np.testing.assert_allclose(a.objective_, b.objective_, rtol=1e-6)


@pytest.fixture(scope="module")
def gaussian_and_laplace_sources():
    """Two Gaussian sources of different slowness (AR(1) with phi = 0.99 and
    0.5), which ICA cannot separate, and two white Laplace (super-Gaussian)
    sources, which SFA cannot separate, randomly mixed."""
    rng = np.random.default_rng(0)
    T = 10000
    S = np.column_stack([ar1(T, .99, rng), ar1(T, .5, rng),
                         rng.laplace(size=T), rng.laplace(size=T)])
    S = S / S.std(0)
    X, _ = random_mix(S, seed=1)
    return S, X


def best_corr2(Y, S):
    """For each source, the largest squared correlation with a component."""
    return np.array([max(corr2(Y[:, i], S[:, k]) for i in range(Y.shape[1]))
                     for k in range(S.shape[1])])


def test_sica_separates_gaussian_and_non_gaussian_sources(gaussian_and_laplace_sources):
    """By combining non-Gaussianity and slowness, SICA separates the Gaussian
    sources by their slowness and the Laplace sources by their
    non-Gaussianity (Feng et al., 2019, Sec. 3.2)."""
    S, X = gaussian_and_laplace_sources
    model = SICA(n_components=4, a=1.0, random_state=0).fit(X)
    Y = model.transform(X)
    assert np.all(best_corr2(Y, S) > 0.99)
    assert np.all(np.diff(model.slowness_) > 0)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), np.eye(4), atol=1e-8)


def test_sica_selects_slowest_components(gaussian_and_laplace_sources):
    S, X = gaussian_and_laplace_sources
    Y = SICA(n_components=2, random_state=0).fit_transform(X)
    assert corr2(Y[:, 0], S[:, 0]) > 0.99
    assert corr2(Y[:, 1], S[:, 1]) > 0.99


def test_sica_without_slowness_is_fastica():
    """For a = 0, SICA is FastICA with G(u) = log cosh(u) and symmetric
    decorrelation (Feng et al., 2019, Sec. 3.3.2)."""
    rng = np.random.default_rng(5)
    S = rng.laplace(size=(10000, 4))
    X, _ = random_mix(S, seed=2)
    Y = SICA(n_components=4, a=0.0, random_state=0).fit_transform(X)
    Y_ica = FastICA(n_components=4, whiten='unit-variance',
                    random_state=0).fit_transform(X)
    assert np.all(best_corr2(Y, S) > 0.99)
    assert np.all(best_corr2(Y, Y_ica) > 0.9999)


def test_sica_large_slowness_weight_is_sfa(gaussian_and_laplace_sources):
    """As a grows, SICA approaches SFA (Feng et al., 2019, Sec. 3.3.2)."""
    _, X = gaussian_and_laplace_sources
    Y = SICA(n_components=4, a=1e4, random_state=0).fit_transform(X)
    Y_sfa = SFA(n_components=4).fit_transform(X)
    for i in range(4):
        assert corr2(Y[:, i], Y_sfa[:, i]) > 0.999


def test_sica_independent_of_initialisation(gaussian_and_laplace_sources):
    _, X = gaussian_and_laplace_sources
    a = SICA(n_components=4, random_state=0).fit_transform(X)
    b = SICA(n_components=4, random_state=1).fit_transform(X)
    assert np.all(best_corr2(a, b) > 0.9999)
