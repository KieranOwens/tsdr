"""Value tests for the autocorrelation methods."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from tsdr.autocorrelation import TICA, TCCA_Stiefel, sPCA_dwt

from helpers import subspace_cos, random_mix, ar1


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


def test_tcca_stiefel_converged_and_ordered(ar_sources):
    """With whitened data the trace-ratio optimum is the leading TICA
    subspace; components are ordered by autocorrelation."""
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
