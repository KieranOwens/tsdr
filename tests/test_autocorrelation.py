"""Value tests for the autocorrelation methods."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from tsdr.autocorrelation import TICA, CSA, TCCA, TCCA_Stiefel, sPCA_dwt

from helpers import subspace_cos, random_mix, ar1, corr2


@pytest.fixture(scope="module")
def ar_sources():
    rng = np.random.default_rng(42)
    T = 5000
    S = np.column_stack([ar1(T, .995, rng), ar1(T, .9, rng), ar1(T, -.8, rng),
                         rng.normal(size=(T, 3))])
    X, _ = random_mix(S, seed=3)
    return S, X


def lag1_autocorr(y):
    return np.corrcoef(y[:-1], y[1:])[0, 1]


def canonical_correlations_ref(X, lag):
    """Canonical correlations of x_t and x_{t+lag}: the square roots of the
    eigenvalues of C00^-1 C01 C11^-1 C10."""
    X0 = X[:-lag] - X[:-lag].mean(0)
    X1 = X[lag:] - X[lag:].mean(0)
    C00, C11, C01 = X0.T @ X0, X1.T @ X1, X0.T @ X1
    M = np.linalg.solve(C00, C01) @ np.linalg.solve(C11, C01.T)
    return np.sqrt(np.sort(np.linalg.eigvals(M).real)[::-1])


@pytest.mark.parametrize("lag", [1, 3])
def test_tcca_canonical_correlations(ar_sources, lag):
    """The canonical correlations are those of x_t and x_{t+lag}, and each
    component has that correlation with its right canonical variate."""
    _, X = ar_sources
    model = TCCA(n_components=3, lag=lag).fit(X)
    np.testing.assert_allclose(model.canonical_correlations_,
                               canonical_correlations_ref(X, lag), rtol=1e-6)
    Y = model.transform(X)
    Y_right = (X - model.mean_) @ model.right_transform_
    for k in range(3):
        rho = np.corrcoef(Y[:-lag, k], Y_right[lag:, k])[0, 1]
        np.testing.assert_allclose(rho, model.canonical_correlations_[k], rtol=1e-3)


def test_tcca_components_are_uncorrelated_with_unit_variance(ar_sources):
    _, X = ar_sources
    lag = 2
    Y = TCCA(n_components=4, lag=lag).fit_transform(X)
    Y0 = Y[:-lag] - Y[:-lag].mean(0)
    np.testing.assert_allclose(Y0.T @ Y0 / (len(Y0) - 1), np.eye(4), atol=1e-10)


def test_tcca_is_affine_invariant(ar_sources):
    """CCA does not depend on an invertible linear map of the data."""
    _, X = ar_sources
    A = np.random.default_rng(0).normal(size=(X.shape[1], X.shape[1]))
    Y = TCCA(n_components=3).fit_transform(X)
    YA = TCCA(n_components=3).fit_transform(X @ A + 1.0)
    signs = np.sign(np.sum(Y * YA, axis=0))
    np.testing.assert_allclose(YA * signs, Y, atol=1e-8)


def test_tcca_ranks_by_absolute_autocorrelation(ar_sources):
    """For these (time-reversible) AR(1) sources with autocorrelations .995,
    .9 and -.8, TCCA recovers all three, ranked by |autocorrelation|, while
    TICA (ranked by signed autocorrelation) replaces the third by white
    noise."""
    S, X = ar_sources
    Y = TCCA(n_components=3).fit_transform(X)
    for k in range(3):
        assert corr2(Y[:, k], S[:, k]) > 0.95
    Y_tica = TICA(n_components=3).fit_transform(X)
    assert max(corr2(Y_tica[:, k], S[:, 2]) for k in range(3)) < 0.1


def test_tcca_stiefel_converged_and_ordered(ar_sources):
    """With whitened data the trace-ratio optimum is the leading TICA
    subspace (not the CCA of x_t and x_{t+lag}); components are ordered by
    autocorrelation."""
    _, X = ar_sources
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = TCCA_Stiefel(n_components=3, lag=1, max_iters=20000).fit(X)
    tica = TICA(n_components=3, lag=1).fit(X)
    assert np.all(subspace_cos(model.linear_transform_,
                               tica.linear_transform_) > 0.999)
    Y = model.transform(X)
    ac = [lag1_autocorr(Y[:, i]) for i in range(3)]
    assert ac[0] > ac[1] > ac[2]


def test_tcca_stiefel_warns_if_not_converged(ar_sources):
    _, X = ar_sources
    with pytest.warns(ConvergenceWarning):
        TCCA_Stiefel(n_components=3, lag=1, max_iters=100).fit(X)


def test_tcca_stiefel_keeps_global_random_state(ar_sources):
    _, X = ar_sources
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        TCCA_Stiefel(n_components=1, seed=0).fit(X)
    assert np.random.rand() == expected


@pytest.mark.parametrize("T", [1000, 1001])
@pytest.mark.parametrize("method", ["pca", "svd"])
@pytest.mark.parametrize("wavelet", ["db1", "db4"])
def test_spca_dwt_output_length(T, method, wavelet):
    rng = np.random.default_rng(0)
    z = np.sin(np.linspace(0, 4 * np.pi, T))
    X, _ = random_mix(np.column_stack([z, rng.normal(size=(T, 4))]), seed=0)
    Y = sPCA_dwt(n_components=2, method=method, wavelet=wavelet).fit_transform(X)
    assert Y.shape == (T, 2)
    assert np.corrcoef(Y[:, 0], z)[0, 1] ** 2 > 0.99

@pytest.mark.parametrize("T", [1000, 1001])
@pytest.mark.parametrize("wavelet", ["db1", "db4"])
@pytest.mark.parametrize("whiten", [True, False])
def test_spca_dwt_pca_and_svd_agree(T, wavelet, whiten):
    """method='pca' and method='svd' give the same components (up to sign)."""
    rng = np.random.default_rng(0)
    z = np.sin(np.linspace(0, 4 * np.pi, T))
    X, _ = random_mix(np.column_stack([z, rng.normal(size=(T, 4))]), seed=0)

    kwargs = dict(n_components=2, wavelet=wavelet, whiten=whiten)
    Y_pca = sPCA_dwt(method="pca", **kwargs).fit_transform(X)
    Y_svd = sPCA_dwt(method="svd", **kwargs).fit_transform(X)

    assert Y_pca.shape == Y_svd.shape == (T, 2)

    # the leading component is the same series (corr2 ignores sign and scale)
    assert corr2(Y_pca[:, 0], Y_svd[:, 0]) > 0.999

    # the two-component subspaces coincide (robust to rotation within the
    # subspace if the 2nd and 3rd singular values are close)
    assert np.all(subspace_cos(Y_pca, Y_svd) > 0.999)

@pytest.mark.parametrize("lag", [np.int64(3), (1, 2, 3), [1, 2, 3],
                                 np.array([1, 2, 3])])
def test_csa_accepts_integer_and_sequence_lags(ar_sources, lag):
    """An integer lag (including NumPy integers) means lags 1, ..., lag, and
    any sequence of integers gives those lags."""
    _, X = ar_sources
    ref = CSA(n_components=2, lag=3, max_iters=50).fit(X).linear_transform_
    W = CSA(n_components=2, lag=lag, max_iters=50).fit(X).linear_transform_
    np.testing.assert_allclose(W, ref)


def test_csa_default_lag_and_params(ar_sources):
    _, X = ar_sources
    model = CSA()
    assert model.lag == 2
    assert model._get_lags() == [1, 2]


@pytest.mark.parametrize("lag, error", [(0, ValueError), ([1, 0], ValueError),
                                        ([], ValueError), (1.5, TypeError),
                                        ([1, 2.5], TypeError)])
def test_csa_rejects_invalid_lags(ar_sources, lag, error):
    _, X = ar_sources
    with pytest.raises(error):
        CSA(lag=lag).fit(X)
