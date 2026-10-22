"""Value tests for the diffusion methods."""

import numpy as np
import pytest
import scipy.stats

from tsdr.diffusion import DMcov, DIG, NLSA

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


def sinusoid_benchmark(T=1000, D=10, noise_var=1.0, seed=0):
    """The sinusoid-in-noise benchmark of Owens & Fulcher (2025): a sinusoid
    and D - 1 white-noise channels, randomly rotated."""
    rng = np.random.default_rng(seed)
    z = np.sin(np.linspace(0, 4 * np.pi, T))
    noise = rng.normal(scale=np.sqrt(noise_var), size=(T, D - 1))
    A = scipy.stats.ortho_group.rvs(D, random_state=seed)
    return z, np.column_stack([noise, z]) @ A


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nlsa_recovers_sinusoid(seed):
    """The leading NLSA temporal mode recovers the sinusoid; the modes are
    given for time points q, ..., n_samples - 1."""
    z, X = sinusoid_benchmark(seed=seed)
    q = 10
    Y = NLSA(n_components=1, q=q).fit_transform(X)
    assert Y.shape == (len(z) - q, 1)
    assert corr2(Y[:, 0], z[q:]) > 0.9


def test_nlsa_whitening_recovers_low_variance_sinusoid():
    """With high-variance noise, the kernel distances of the unwhitened data
    are dominated by the noise, but after whitening NLSA recovers the 
    sinusoid."""
    z, X = sinusoid_benchmark(noise_var=100.0)
    q = 10
    assert corr2(NLSA(q=q).fit_transform(X)[:, 0], z[q:]) > 0.9
    assert corr2(NLSA(q=q, whiten=False).fit_transform(X)[:, 0], z[q:]) < 0.1


def test_nlsa_depends_on_temporal_order():
    """NLSA uses the temporal order of the data (through the time-lagged 
    embedding): after a random permutation of the time points, it no longer
    recovers the sinusoid."""
    z, X = sinusoid_benchmark()
    q = 10
    perm = np.random.default_rng(0).permutation(len(z))
    assert corr2(NLSA(q=q).fit_transform(X[perm])[:, 0], z[perm][q:]) < 0.1


def test_nlsa_temporal_modes_are_orthonormal():
    """The temporal modes are orthonormal with respect to the invariant
    measure of the diffusion map, which is a probability distribution."""
    _, X = sinusoid_benchmark(T=500)
    model = NLSA(n_components=5).fit(X)
    V = model.components
    np.testing.assert_allclose(model.measure_.sum(), 1)
    assert np.all(model.measure_ > 0)
    np.testing.assert_allclose(V.T @ (model.measure_[:, None] * V), np.eye(5),
                               atol=1e-10)
    assert np.all(np.diff(model.singular_values_) <= 0)


def test_nlsa_with_complete_eigenfunction_basis():
    """With all eigenfunctions (l = s), the eigenfunctions form a complete
    basis, so A A^T = E^T diag(mu) E and the singular values are those of
    the measure-weighted lagged embedding E (Giannakis & Majda, 2012)."""
    _, X = sinusoid_benchmark(T=300)
    q = 10
    s = len(X) - q
    model = NLSA(n_components=5, q=q, n_eigenfunctions=s, whiten=False).fit(X)
    Xc = X - X.mean(0)
    E = np.column_stack([Xc[q - 1 - i:len(X) - i] for i in range(q)])[1:]
    eigvals = np.linalg.eigvalsh(E.T @ (model.measure_[:, None] * E))[::-1]
    np.testing.assert_allclose(model.singular_values_, np.sqrt(eigvals[:5]),
                               rtol=1e-8)
