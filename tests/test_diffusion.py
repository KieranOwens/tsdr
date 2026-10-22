"""Value tests for the diffusion methods."""

import numpy as np
import pytest

from tsdr.diffusion import DMcov, DIG

from helpers import corr2, random_mix


@pytest.fixture(scope="module")
def drifting_mean():
    """Five channels whose mean drifts with one slow latent coordinate."""
    rng = np.random.default_rng(0)
    T, D = 2000, 5
    latent = np.sin(np.linspace(0, 2 * np.pi, T))
    S = np.column_stack([3 * latent + rng.normal(size=T),
                         rng.normal(size=(T, D - 1))])
    X, _ = random_mix(S, seed=1)
    window, step = 100, 40
    latent_w = np.array([latent[i:i + window].mean()
                         for i in range(0, T - window + 1, step)])
    return X, latent_w, window, step


@pytest.mark.parametrize("normalise", [True, False])
def test_dmcov_recovers_drifting_latent(drifting_mean, normalise):
    X, latent_w, window, step = drifting_mean
    Y = DMcov(window=window, step=step, n_components=2,
              normalise=normalise).fit_transform(X)
    assert Y.shape == (len(latent_w), 2)
    assert corr2(Y[:, 0], latent_w) > 0.95


def test_dmcov_normalised_is_leading_nontrivial_eigenvector(drifting_mean):
    """normalise=True returns the eigenvectors of D^-1/2 W D^-1/2 with the
    largest non-trivial eigenvalues."""
    X, _, window, step = drifting_mean
    model = DMcov(window=window, step=step, n_components=2, normalise=True)
    Y = model.fit_transform(X)
    G = model._cov_distance_matrix(X - X.mean(0))
    K = np.exp(-G ** 2 / model.sigma)
    d = K.sum(1)
    lam, V = np.linalg.eigh(K / np.sqrt(np.outer(d, d)))
    trivial = np.argmax(lam)
    order = [i for i in np.argsort(-np.abs(lam)) if i != trivial]
    for i in range(2):
        assert corr2(Y[:, i], V[:, order[i]]) > 0.999


@pytest.mark.parametrize("T, window, step", [(1000, 100, 40), (1000, 100, 50),
                                             (1000, 100, 100)])
@pytest.mark.parametrize("method", [DMcov, DIG])
def test_windows_include_last_window(method, T, window, step):
    """One window starts at each of 0, step, 2*step, ..., up to and including
    the last start index that fits, T - window (when it is on the grid)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, 3))
    G = method(window=window, step=step)._cov_distance_matrix(X)
    n = (T - window) // step + 1
    assert G.shape == (n, n)
