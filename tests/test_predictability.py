"""Value tests for the predictability methods (TLPC, PrCA, DiCCA, ForeCA,
PFA, PTS, DCA)."""

import numpy as np
import pytest
import scipy.stats
from sklearn.decomposition import PCA

from tsdr.predictability import TLPC, PrCA, DiCCA, ForeCA, PFA, PTS, DCA

from tsdr._whitening import fit_whitening
from tsdr.slowness import SFA

from helpers import corr2, subspace_cos, random_mix, ar1, unmixing, whiten_ref
import reference as ref


def ar2(T, a1, a2, rng):
    x = np.zeros(T)
    e = rng.normal(size=T)
    for t in range(2, T):
        x[t] = a1 * x[t - 1] + a2 * x[t - 2] + e[t]
    return x


@pytest.fixture(scope="module")
def three_ar_sources():
    """Three AR(2) sources of decreasing predictability plus three white-noise
    sources, randomly mixed."""
    rng = np.random.default_rng(3)
    T = 5000
    S = np.column_stack([ar2(T, 1.6, -0.9, rng), ar2(T, 0.9, -0.5, rng),
                         ar2(T, 0.3, 0.2, rng), rng.normal(size=(T, 3))])
    X, _ = random_mix(S, seed=7)
    return S, X


def test_dicca_multiple_components(three_ar_sources):
    S, X = three_ar_sources
    s = 3
    Y = DiCCA(n_components=3, s=s).fit_transform(X)

    # the three components span the dynamic subspace
    model = DiCCA(n_components=3, s=s).fit(X)
    assert np.all(subspace_cos(model.linear_transform_,
                               unmixing(X, S[:, :3])) > 0.99)

    # scores are mutually uncorrelated and ordered by AR predictability
    C = np.corrcoef(Y.T)
    assert np.all(np.abs(C[np.triu_indices(3, 1)]) < 0.01)
    r2 = [ref.ar_R2(Y[:, i], s) for i in range(3)]
    assert r2[0] > r2[1] > r2[2]

    # each component matches the reference implementation and a distinct source
    Yr = (X - X.mean(0)) @ ref.dicca_ref(X, 3, s)
    for i in range(3):
        assert corr2(Y[:, i], Yr[:, i]) > 0.99
        assert corr2(Y[:, i], S[:, i]) > 0.99


def rotation_latent(T, theta, r, rng):
    R = r * np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    z = np.zeros((T, 2))
    for t in range(1, T):
        z[t] = R @ z[t - 1] + 0.3 * rng.normal(size=2)
    return z


@pytest.mark.parametrize("theta", [np.pi / 2, np.pi / 3, np.pi / 12])
def test_tlpc_rotational_latent(theta):
    """A noisy 2-d rotation is the most predictable plane; TLPC must find
    both of its coordinates (non-reversible dynamics, non-symmetric C_tau)."""
    rng = np.random.default_rng(0)
    T = 20000
    z = rotation_latent(T, theta, 0.97, rng)
    X, _ = random_mix(np.column_stack([z, rng.normal(size=(T, 4))]), seed=1)
    model = TLPC(n_components=2).fit(X)
    Y = model.transform(X)
    assert np.all(subspace_cos(model.linear_transform_, unmixing(X, z)) > 0.99)
    assert abs(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]) < 0.01
    Wr, _ = ref.tlpc_ref(X, 2)
    assert np.all(subspace_cos(model.linear_transform_, Wr) > 0.999)


def test_tlpc_reversible_matches_reference():
    rng = np.random.default_rng(2)
    T = 5000
    S = np.column_stack([ar1(T, 0.95, rng), ar1(T, 0.5, rng),
                         rng.normal(size=(T, 3))])
    X, _ = random_mix(S, seed=2)
    Y = TLPC(n_components=2).fit_transform(X)
    Wr, _ = ref.tlpc_ref(X, 2)
    Yr = (X - X.mean(0)) @ Wr
    for i in range(2):
        assert corr2(Y[:, i], Yr[:, i]) > 0.999


@pytest.fixture(scope="module")
def opposite_ar_sources():
    """Two predictable AR(1) sources (phi = +0.9, -0.9) and four noise
    sources."""
    rng = np.random.default_rng(1)
    T = 20000
    return np.column_stack([ar1(T, .9, rng), ar1(T, -.9, rng),
                            rng.normal(size=(T, 4))])


@pytest.mark.parametrize("p", [1, 3])
def test_prca_linear_reg_matches_multivariate_reference(opposite_ar_sources, p):
    S = opposite_ar_sources
    X, _ = random_mix(S, seed=5)
    model = PrCA(n_components=2, model='linear_reg', p=p).fit(X)
    assert np.all(subspace_cos(model.linear_transform_,
                               unmixing(X, S[:, :2])) > 0.999)
    Wr, _ = ref.prca_ref(X, 2, p=p)
    assert np.all(subspace_cos(model.linear_transform_, Wr) > 0.999)


def test_prca_linear_reg_is_basis_independent(opposite_ar_sources):
    """Two different mixings of the same sources give the same components in
    source coordinates."""
    S = opposite_ar_sources
    X1, A1 = random_mix(S, seed=11)
    X2, A2 = random_mix(S, seed=12)
    w1 = PrCA(n_components=2, model='linear_reg').fit(X1).linear_transform_
    w2 = PrCA(n_components=2, model='linear_reg').fit(X2).linear_transform_
    assert np.all(subspace_cos(A1 @ w1, A2 @ w2) > 0.999)


def test_prca_kernel_ridge_linear_kernel(opposite_ar_sources):
    S = opposite_ar_sources[:2000]
    X, _ = random_mix(S, seed=5)
    model = PrCA(n_components=2, model='kernel_ridge', kernel='linear',
                 alpha=1e-6).fit(X)
    assert np.all(subspace_cos(model.linear_transform_,
                               unmixing(X, S[:, :2])) > 0.99)


def test_prca_unknown_model_raises(opposite_ar_sources):
    with pytest.raises(ValueError, match="Unknown model"):
        PrCA(model='ridge').fit(opposite_ar_sources[:500])


def test_foreca_maximises_forecastability_2d():
    """In two dimensions the most forecastable direction can be found by
    brute force over the angle of a unit vector in whitened coordinates."""
    rng = np.random.default_rng(0)
    T = 600
    S = np.column_stack([ar2(T, 1.6, -0.9, rng), rng.normal(size=T)])
    X, _ = random_mix(S, seed=1)
    model = ForeCA(n_components=1, random_state=0).fit(X)
    Xw = fit_whitening(X).transform(X)
    angles = np.linspace(0, np.pi, 181)
    best = max(ref.foreca_omega_ref(Xw @ np.array([np.cos(a), np.sin(a)]))
               for a in angles)
    assert model.omega_[0] >= best - 1e-6


def test_foreca_recovers_most_forecastable_sources(three_ar_sources):
    """The components are ordered by forecastability, omega_ is the
    forecastability of each component (Goerg, 2013, eq. 15), and the
    components are uncorrelated with unit variance."""
    S, X = three_ar_sources
    model = ForeCA(n_components=3, random_state=0).fit(X)
    Y = model.transform(X)
    assert corr2(Y[:, 0], S[:, 0]) > 0.999
    assert np.all(subspace_cos(model.linear_transform_,
                               unmixing(X, S[:, :3])) > 0.999)
    assert np.all(np.diff(model.omega_) < 0)
    np.testing.assert_allclose(model.omega_,
                               [ref.foreca_omega_ref(Y[:, i]) for i in range(3)],
                               rtol=1e-8)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), np.eye(3), atol=1e-8)


def test_foreca_em_decreases_spectral_entropy(three_ar_sources):
    """Each iteration of the EM-like algorithm does not increase the
    spectral entropy (Goerg, 2013, Theorem 4.2)."""
    _, X = three_ar_sources
    Xw = fit_whitening(X).transform(X)
    T, D = Xw.shape
    F = np.fft.fft(Xw, axis=0) / np.sqrt(T * (T - 1))
    model = ForeCA(max_iters=1)
    w = np.random.default_rng(0).normal(size=D)
    w = w / np.linalg.norm(w)
    entropies = []
    for _ in range(20):
        w, H, _ = model._em(F, w)
        entropies.append(H)
    assert np.all(np.diff(entropies) <= 1e-12)


def test_foreca_independent_of_random_starts(three_ar_sources):
    _, X = three_ar_sources
    a = ForeCA(n_components=2, random_state=0).fit(X)
    b = ForeCA(n_components=2, random_state=1).fit(X)
    np.testing.assert_allclose(a.omega_, b.omega_, rtol=1e-8)
    assert np.all(subspace_cos(a.linear_transform_, b.linear_transform_) > 0.9999)


@pytest.mark.parametrize("p, k", [(1, 0), (2, 4), (3, 2)])
def test_pfa_matches_reference(three_ar_sources, p, k):
    _, X = three_ar_sources
    X = X[:1000]
    model = PFA(n_components=3, p=p, k=k).fit(X)
    W, eigvals = ref.pfa_ref(X, 3, p, k)
    assert np.all(subspace_cos(model.linear_transform_, W) > 1 - 1e-8)
    np.testing.assert_allclose(model.eigvals_, eigvals, atol=1e-10)


@pytest.mark.parametrize("p", [1, 2])
def test_pfa_without_iterated_prediction_is_prca(three_ar_sources, p):
    """For k = 0, PFA is PCA on the residuals of a least squares
    autoregression (Richthofer & Wiskott, 2013, eq. 21), i.e., PrCA."""
    _, X = three_ar_sources
    W = PFA(n_components=3, p=p, k=0).fit(X).linear_transform_
    W_prca = PrCA(n_components=3, model='linear_reg', p=p).fit(X).linear_transform_
    assert np.all(subspace_cos(W, W_prca) > 1 - 1e-8)


def test_pfa_recovers_predictable_sources(three_ar_sources):
    S, X = three_ar_sources
    model = PFA(n_components=3).fit(X)
    Y = model.transform(X)
    assert np.all(subspace_cos(model.linear_transform_,
                               unmixing(X, S[:, :3])) > 0.99)
    assert corr2(Y[:, 0], S[:, 0]) > 0.99
    assert np.all(np.diff(model.eigvals_) >= 0)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), np.eye(3), atol=1e-8)


def test_pfa_iterated_prediction_reduces_overfitting():
    """The example of Richthofer & Wiskott (2013, Sec. 2.3): three noisy
    sinusoids mixed with many white-noise dimensions. With k = 0 the
    components overfit to the noise, so the autoregressive prediction error
    of the extracted components (eq. 19) is higher than with k = 4."""
    rng = np.random.default_rng(0)
    T, n_noise = 1000, 200
    t = np.arange(T)
    eta = rng.normal(size=(T, 3))
    S = np.column_stack([np.sin(0.1 * t) + 0.7 * eta[:, 0],
                         np.sin(0.2 * t) + eta[:, 1],
                         np.sin(0.4 * t) + 5.3 * eta[:, 2],
                         rng.normal(size=(T, n_noise))])
    X = S @ scipy.stats.ortho_group.rvs(3 + n_noise, random_state=0)

    def ar_error(M, p=2):
        Zeta = np.column_stack([M[p - j:len(M) - j] for j in range(1, p + 1)])
        E = M[p:] - Zeta @ np.linalg.lstsq(Zeta, M[p:], rcond=None)[0]
        return np.sum(np.mean(E ** 2, axis=0))

    err_k0 = ar_error(PFA(n_components=2, p=2, k=0).fit_transform(X))
    err_k4 = ar_error(PFA(n_components=2, p=2, k=4).fit_transform(X))
    assert err_k4 < err_k0 - 0.2


def var_error(Y, M):
    """Mean squared prediction error (summed over components) of a least
    squares VAR(M) model of the centred time series Y."""
    Y = Y - Y.mean(0)
    R = np.column_stack([Y[M - i:len(Y) - i] for i in range(1, M + 1)])
    E = Y[M:] - R @ np.linalg.lstsq(R, Y[M:], rcond=None)[0]
    return np.sum(E ** 2) / len(E)


def test_pts_exact_solution_for_one_lag_and_component(three_ar_sources):
    """For M = 1 and one component, the most predictable projection is the
    eigenvector of the symmetrised lag-1 autocovariance of the whitened data
    with the eigenvalue of largest magnitude (Dong et al., 2022, Sec. 2.3)."""
    _, X = three_ar_sources
    model = PTS(n_components=1, M=1).fit(X)
    Xw, W_white, _ = whiten_ref(X)
    C1 = Xw[:-1].T @ Xw[1:]
    lam, V = np.linalg.eigh(C1 + C1.T)
    w = W_white @ V[:, [np.argmax(np.abs(lam))]]
    assert subspace_cos(model.linear_transform_, w)[0] > 1 - 1e-6
    np.testing.assert_allclose(model.prediction_errors_[0],
                               var_error(X @ w / np.std(X @ w, ddof=1), 1),
                               rtol=1e-4)


def test_pts_recovers_predictable_sources(three_ar_sources):
    """The components span the predictable subspace, are uncorrelated with
    unit variance, ar_coefs_ is the VAR model of the components, and an
    unpredictable fourth component adds about 1 to the prediction error
    (Dong et al., 2022, Sec. 3.3)."""
    S, X = three_ar_sources
    model = PTS(n_components=4, M=2).fit(X)
    Y = model.transform(X)
    assert np.all(subspace_cos(model.linear_transform_[:, :3],
                               unmixing(X, S[:, :3])) > 0.99)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), np.eye(4), atol=1e-8)
    increments = np.diff(np.r_[0, model.prediction_errors_])
    assert np.all(increments[:3] < 0.9)
    assert increments[3] > 0.95
    assert model.ar_coefs_.shape == (2, 4, 4)
    np.testing.assert_allclose(model.prediction_errors_[-1], var_error(Y, 2),
                               rtol=1e-3)


def test_pts_paper_simulation():
    """The simulation of Dong et al. (2022, Sec. 4.1, eq. 8), with 30 rather
    than 1000 observed dimensions: a 3-dimensional VAR(2) process embedded
    in noisy, standardised, randomly rotated observations."""
    rng = np.random.default_rng(0)
    T, n = 10000, 30
    B1 = np.array([[1.1241, 0.3045, 0.3806], [0.3902, -0.8169, -0.3114],
                   [-0.7166, -0.8630, 1.0115]])
    B2 = np.array([[-0.2482, 0.3676, 0.0328], [-0.4240, 0.1101, 0.0267],
                   [0.6011, -0.5975, -0.3224]])
    x = np.zeros((T, 3))
    for t in range(2, T):
        x[t] = B1 @ x[t - 1] + B2 @ x[t - 2] + rng.normal(size=3)
    P = np.linalg.qr(rng.normal(size=(n, 3)))[0]
    y = x @ P.T + 0.02 * rng.normal(size=(T, n))
    Q = scipy.stats.ortho_group.rvs(n, random_state=1)
    z = y @ np.linalg.inv(np.linalg.cholesky(y.T @ y / T)).T @ Q.T
    model = PTS(n_components=4, M=2).fit(z)
    assert np.all(subspace_cos(model.linear_transform_[:, :3],
                               unmixing(z, x)) > 0.999)
    increments = np.diff(np.r_[0, model.prediction_errors_])
    assert np.all(increments[:3] < 0.2)
    assert increments[3] > 0.95


def test_pts_alternating_steps_decrease_prediction_error(three_ar_sources):
    """Each alternation of the VAR fit and the update of w does not increase
    the prediction error of the second component."""
    _, X = three_ar_sources
    Xw, _, _ = whiten_ref(X)
    model = PTS(n_components=2, M=2)
    W1 = PTS(n_components=1, M=2).fit(X).linear_transform_
    Wk = np.linalg.lstsq(Xw, X @ W1, rcond=None)[0]
    Wk = Wk / np.linalg.norm(Wk)
    w = np.linalg.qr(np.column_stack([Wk, np.ones(X.shape[1])]))[0][:, 1]
    errors = []
    for _ in range(10):
        A, _ = model._fit_var(Xw, np.column_stack([Wk, w]))
        w = model._update_w(Xw, Wk, A)
        errors.append(model._fit_var(Xw, np.column_stack([Wk, w]))[1])
    assert np.all(np.diff(errors) <= 1e-12)


def test_dca_gradient(three_ar_sources):
    """The analytic gradient of the predictive information agrees with
    finite differences."""
    _, X = three_ar_sources
    Xw = fit_whitening(X).transform(X)
    N, D = Xw.shape
    model = DCA(n_components=2, T=3)
    windows = np.column_stack([Xw[i:N - 5 + i] for i in range(6)])
    Sigma = np.cov(windows, rowvar=False)
    V = np.random.default_rng(0).normal(size=(D, 2))
    _, grad = model._predictive_information(V, Sigma)
    eps = 1e-6
    for i, j in [(0, 0), (2, 1), (5, 0)]:
        E = np.zeros_like(V)
        E[i, j] = eps
        numerical = (model._predictive_information(V + E, Sigma)[0]
                     - model._predictive_information(V - E, Sigma)[0]) / (2 * eps)
        np.testing.assert_allclose(grad[i, j], numerical, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("n_components", [1, 2])
def test_dca_with_one_time_step_is_sfa(n_components):
    """For T = 1 and time-reversible data (AR(1) sources with positive
    autocorrelation), DCA and SFA find the same subspace (Clark et al., 2019,
    Sec. 3)."""
    rng = np.random.default_rng(1)
    T = 5000
    S = np.column_stack([ar1(T, .95, rng), ar1(T, .8, rng), ar1(T, .5, rng),
                         rng.normal(size=(T, 3))])
    X, _ = random_mix(S, seed=2)
    W = DCA(n_components=n_components, T=1, random_state=0).fit(X).linear_transform_
    W_sfa = SFA(n_components=n_components).fit(X).linear_transform_
    assert np.all(subspace_cos(W, W_sfa) > 1 - 1e-6)


def test_dca_finds_dynamics_rather_than_variance():
    """As in Clark et al. (2019, Fig. 1), a low-variance two-dimensional
    rotational dynamical system embedded with high-variance, spatially
    structured white noise: DCA recovers the dynamics, PCA the noise."""
    rng = np.random.default_rng(3)
    T = 5000
    th = 0.1
    R = 0.98 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    z = np.zeros((T, 2))
    for t in range(1, T):
        z[t] = R @ z[t - 1] + rng.normal(size=2)
    z = z / z.std(0)
    noise = rng.normal(size=(T, 10)) * np.r_[5, 4, 3, 2, 1, 1, 1, 1, 1, 1]
    X = np.column_stack([z, noise]) @ np.linalg.qr(rng.normal(size=(12, 12)))[0]
    U = unmixing(X, z)
    W = DCA(n_components=2, T=5, random_state=0).fit(X).linear_transform_
    assert np.all(subspace_cos(W, U) > 0.999)
    assert np.all(subspace_cos(PCA(2).fit(X).components_.T, U) < 0.1)


def test_dca_predictive_information(three_ar_sources):
    """predictive_information_ is that of the components (eq. 3), at least
    that of the true dynamical subspace, and the components are an
    orthonormal basis (uncorrelated, unit variance)."""
    S, X = three_ar_sources
    model = DCA(n_components=3, T=5, random_state=0).fit(X)
    Y = model.transform(X)
    np.testing.assert_allclose(model.predictive_information_,
                               ref.dca_predictive_information_ref(Y, 5), rtol=1e-4)
    assert model.predictive_information_ >= ref.dca_predictive_information_ref(
        X @ unmixing(X, S[:, :3]), 5) - 1e-8
    assert np.all(subspace_cos(model.linear_transform_,
                               unmixing(X, S[:, :3])) > 0.99)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), np.eye(3), atol=1e-8)


def test_dca_independent_of_initialisation(three_ar_sources):
    _, X = three_ar_sources
    a = DCA(n_components=2, random_state=0).fit(X)
    b = DCA(n_components=2, random_state=1).fit(X)
    assert np.all(subspace_cos(a.linear_transform_, b.linear_transform_) > 0.9999)
    np.testing.assert_allclose(a.predictive_information_, b.predictive_information_,
                               rtol=1e-6)


def test_dca_noise_free_signal():
    """A sinusoid in low-rank noise has a noise-free direction, whose 
    predictive information is unbounded (its projected covariance is 
    singular): DCA fits without error and recovers the sinusoid."""
    rng = np.random.default_rng(0)
    T, D = 1000, 10
    z = np.sin(np.linspace(0, 4 * np.pi, T))
    noise = rng.normal(size=(T, D - 2)) @ rng.normal(size=(D - 2, D - 1))
    X = np.column_stack([z, noise]) @ scipy.stats.ortho_group.rvs(D, random_state=0)[:, :D - 1]
    Y = DCA(random_state=0).fit_transform(X)
    assert corr2(Y, z) > 0.99
