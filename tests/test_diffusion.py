"""Value tests for the diffusion methods."""

import numpy as np
import pytest
import scipy.stats

from tsdr.diffusion import (DMcov, DIG, NLSA, WINDOW_DISTANCES,
                            _window_distance, _window_distance_matrix)

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


@pytest.mark.parametrize("distance", ['mean_mahalanobis', 'bhattacharyya'])
def test_dig_recovers_drifting_latent(drifting_mean, distance):
    X, latent_w, window, step = drifting_mean
    Y = DIG(window=window, step=step, n_components=2,
            distance=distance).fit_transform(X)
    assert Y.shape == (len(latent_w), 2)
    assert corr2(Y[:, 0], latent_w) > 0.9


def test_dmcov_normalised_is_leading_nontrivial_eigenvector(drifting_mean):
    """normalise=True returns the eigenvectors of D^-1/2 W D^-1/2 with the
    largest non-trivial eigenvalues."""
    X, _, window, step = drifting_mean
    model = DMcov(window=window, step=step, n_components=2, normalise=True)
    Y = model.fit_transform(X)
    G = _window_distance_matrix(X - X.mean(0), window, step, model.distance)
    K = np.exp(-G ** 2 / model.sigma)
    d = K.sum(1)
    lam, V = np.linalg.eigh(K / np.sqrt(np.outer(d, d)))
    trivial = np.argmax(lam)
    order = [i for i in np.argsort(-np.abs(lam)) if i != trivial]
    for i in range(2):
        assert corr2(Y[:, i], V[:, order[i]]) > 0.999


@pytest.mark.parametrize("T, window, step", [(1000, 100, 40), (1000, 100, 50),
                                             (1000, 100, 100)])
def test_windows_include_last_window(T, window, step):
    """One window starts at each of 0, step, 2*step, ..., up to and including
    the last start index that fits, T - window (when it is on the grid)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(T, 3))
    G = _window_distance_matrix(X, window, step, 'mean_mahalanobis')
    n = (T - window) // step + 1
    assert G.shape == (n, n)
    assert DMcov(window, step).fit_transform(X).shape[0] == n
    assert DIG(window, step).fit_transform(X).shape[0] == n


# --- distances between windows (DMcov and DIG) ------------------------------

def gaussian_window(mean, cov, T=200000, seed=0):
    """A window of T samples with exactly the given mean and covariance."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(T, len(mean)))
    Z = Z - Z.mean(0)
    Z = Z @ np.linalg.inv(np.linalg.cholesky(np.atleast_2d(np.cov(Z, rowvar=False)))).T
    return Z @ np.linalg.cholesky(cov).T + mean


def test_window_distances_1d_closed_forms():
    """For univariate Gaussians N(m1, s1^2) and N(m2, s2^2)."""
    m1, s1, m2, s2 = 0.5, 1.0, -1.0, 2.0
    A = gaussian_window([m1], [[s1 ** 2]], seed=0)
    B = gaussian_window([m2], [[s2 ** 2]], seed=1)
    expected = {
        'mean_mahalanobis': np.sqrt((m1 - m2) ** 2 / (s1 ** 2 + s2 ** 2)),
        'wasserstein': np.sqrt((m1 - m2) ** 2 + (s1 - s2) ** 2),
        # four times the Bhattacharyya distance
        'bhattacharyya': np.sqrt((m1 - m2) ** 2 / (s1 ** 2 + s2 ** 2)
                                 + 2 * np.log((s1 ** 2 + s2 ** 2) / (2 * s1 * s2))),
        'frobenius': abs(s1 ** 2 - s2 ** 2),
        'geodesic': abs(np.log(s2 ** 2 / s1 ** 2)),
    }
    for distance, d in expected.items():
        np.testing.assert_allclose(_window_distance(A, B, distance), d, rtol=1e-10)


def test_window_distances_commuting_covariances():
    """For diagonal covariances, the Wasserstein distance is
    |dmu|^2 + sum (sqrt(a) - sqrt(b))^2 and the Bhattacharyya distance
    separates over the coordinates."""
    mu1, mu2 = np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, -1.0])
    a, b = np.array([1.0, 4.0, 0.25]), np.array([2.0, 1.0, 0.25])
    A = gaussian_window(mu1, np.diag(a), seed=0)
    B = gaussian_window(mu2, np.diag(b), seed=1)
    np.testing.assert_allclose(
        _window_distance(A, B, 'wasserstein') ** 2,
        np.sum((mu1 - mu2) ** 2) + np.sum((np.sqrt(a) - np.sqrt(b)) ** 2), rtol=1e-8)
    np.testing.assert_allclose(
        _window_distance(A, B, 'bhattacharyya') ** 2,
        np.sum((mu1 - mu2) ** 2 / (a + b) + 2 * np.log((a + b) / 2) - np.log(a) - np.log(b)),
        rtol=1e-8)


def test_mean_only_distance_ignores_covariances():
    """Two windows with equal means and different covariances are at distance
    zero for 'mean_mahalanobis', but not for the covariance-aware distances;
    with equal covariances, 'bhattacharyya' equals 'mean_mahalanobis'."""
    A = gaussian_window([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], seed=0)
    B = gaussian_window([0.0, 0.0], [[4.0, 1.0], [1.0, 1.0]], seed=1)
    assert _window_distance(A, B, 'mean_mahalanobis') < 1e-12
    for distance in ['wasserstein', 'bhattacharyya', 'frobenius', 'geodesic']:
        assert _window_distance(A, B, distance) > 0.3
    C = gaussian_window([1.0, -2.0], [[1.0, 0.0], [0.0, 1.0]], seed=1)
    np.testing.assert_allclose(_window_distance(A, C, 'bhattacharyya'),
                               _window_distance(A, C, 'mean_mahalanobis'), rtol=1e-10)


@pytest.mark.parametrize("distance", WINDOW_DISTANCES)
def test_window_distance_matrix_is_a_dissimilarity(distance):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)) * np.linspace(1, 3, 300)[:, None]
    G = _window_distance_matrix(X, 50, 25, distance)
    np.testing.assert_array_equal(G, G.T)
    np.testing.assert_array_equal(np.diag(G), 0)
    assert np.all(G[np.triu_indices(len(G), 1)] > 0)


@pytest.mark.parametrize("distance", ['mean_mahalanobis', 'bhattacharyya', 'geodesic'])
def test_affine_invariant_distances(distance):
    """These distances do not change under an invertible linear map and a
    shift of the data."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)) * np.linspace(1, 3, 300)[:, None]
    A = rng.normal(size=(3, 3))
    np.testing.assert_allclose(_window_distance_matrix(X @ A + 5.0, 50, 25, distance),
                               _window_distance_matrix(X, 50, 25, distance),
                               rtol=1e-8, atol=1e-10)


@pytest.fixture(scope="module")
def changing_variance():
    """Four channels with a constant mean, of which one (mixed into all
    channels) has a variance that switches between 1 and 9 every 500
    samples."""
    rng = np.random.default_rng(0)
    T, D = 2000, 4
    regime = (np.arange(T) // 500) % 2
    S = rng.normal(size=(T, D))
    S[:, 0] *= 1 + 2 * regime
    X, _ = random_mix(S, seed=1)
    window, step = 100, 20
    regime_w = np.array([regime[i:i + window].mean()
                         for i in range(0, T - window + 1, step)])
    return X, regime_w, window, step


@pytest.mark.parametrize("distance", ['wasserstein', 'bhattacharyya'])
@pytest.mark.parametrize("method", ['DMcov', 'DIG'])
def test_covariance_aware_distances_recover_changing_variance(changing_variance,
                                                              method, distance):
    """Windows that differ only in their covariance are separated by the
    covariance-aware distances, but not by the mean-only distance."""
    X, regime_w, window, step = changing_variance
    if method == 'DMcov':
        # the Wasserstein distance depends on the scale of the data, so the
        # kernel width sigma is set to the scale of the squared distances
        G = _window_distance_matrix(X - X.mean(0), window, step, distance)
        make = lambda d: DMcov(window, step, distance=d, normalise=False,
                               sigma=np.median(G ** 2))
    else:
        make = lambda d: DIG(window, step, distance=d)
    assert corr2(make(distance).fit_transform(X)[:, 0], regime_w) > 0.8
    assert corr2(make('mean_mahalanobis').fit_transform(X)[:, 0], regime_w) < 0.3


def test_dmcov_and_dig_use_the_same_distances(drifting_mean):
    """DMcov centres the data first, which does not change the distances."""
    X, _, window, step = drifting_mean
    G = _window_distance_matrix(X, window, step, 'bhattacharyya')
    np.testing.assert_allclose(_window_distance_matrix(X - X.mean(0), window, step,
                                                       'bhattacharyya'), G, rtol=1e-8)


def test_old_mahalanobis_name_warns_and_gives_mean_mahalanobis(drifting_mean):
    X, _, window, step = drifting_mean
    with pytest.warns(DeprecationWarning, match="mean_mahalanobis"):
        Y_old = DMcov(window, step, distance='mahalanobis').fit_transform(X)
    Y = DMcov(window, step, distance='mean_mahalanobis').fit_transform(X)
    np.testing.assert_array_equal(Y_old, Y)
    with pytest.warns(DeprecationWarning, match="mean_mahalanobis"):
        DIG(window, step, distance='mahalanobis').fit(X)


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


def test_dig_mds_sets_negative_eigenvalues_to_zero_with_warning():
    """Non-Euclidean distances give negative MDS eigenvalues, whose
    components are set to zero (rather than NaN), with a warning."""
    D = np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 1.0], [3.0, 1.0, 0.0]])
    with pytest.warns(RuntimeWarning, match="set to zero"):
        Y = DIG(10, 1, n_components=3)._mds(D)
    assert np.all(np.isfinite(Y))
    np.testing.assert_allclose(Y[:, 1:], 0, atol=1e-8)
    # the first component is the classical MDS solution for the positive
    # eigenvalue, 1.5 * (-1, 0, 1) up to sign
    np.testing.assert_allclose(np.abs(Y[:, 0]), [1.5, 0, 1.5], atol=1e-8)


def test_dig_mds_of_euclidean_distances_recovers_the_points():
    """For Euclidean distances, MDS recovers the (centred) points up to a
    rotation, without warnings."""
    import warnings
    rng = np.random.default_rng(0)
    P = rng.normal(size=(20, 2))
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Y = DIG(10, 1, n_components=2)._mds(D)
    np.testing.assert_allclose(np.linalg.norm(Y[:, None] - Y[None], axis=2), D,
                               atol=1e-10)


def test_nlsa_repeated_samples_give_finite_modes_with_warning():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    X[100:110] = X[100]
    with pytest.warns(RuntimeWarning, match="zero local velocity"):
        Y = NLSA(n_components=2, q=3).fit_transform(X)
    assert np.all(np.isfinite(Y))


@pytest.mark.parametrize("kwargs", [dict(q=3, n_eigenfunctions=298),
                                    dict(q=0), dict(q=299), dict(q=2.5),
                                    dict(n_eigenfunctions=0)])
def test_nlsa_invalid_q_or_n_eigenfunctions_raises(kwargs):
    X = np.random.default_rng(0).normal(size=(300, 3))
    with pytest.raises(ValueError, match="q|n_eigenfunctions"):
        NLSA(**kwargs).fit(X)
