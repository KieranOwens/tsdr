"""Value tests for the latent variable methods."""

import numpy as np

from tsdr.latentvariable import LDS

from reference import lds_posterior_ref


def test_lds_e_step_matches_exact_posterior():
    """The Kalman filter/RTS smoother expectations used in the M-step equal
    the exact posterior moments, including the lag-one cross-moments
    E[x_{t+1} x_t^T | Y], for non-reversible (rotational) latent dynamics."""
    rng = np.random.default_rng(0)
    d, D, T = 2, 4, 8
    th = 0.5
    A = 0.9 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = np.diag([1.0, 0.3])
    C = rng.normal(size=(D, d))
    R = 0.5 * np.eye(D)
    mu0, V0 = np.array([0.5, -0.5]), np.eye(d)
    x = np.zeros((T, d))
    x[0] = rng.multivariate_normal(mu0, V0)
    for t in range(1, T):
        x[t] = A @ x[t - 1] + rng.multivariate_normal(np.zeros(d), Q)
    Y = x @ C.T + rng.multivariate_normal(np.zeros(D), R, size=T)

    Ex, Exx, Exxm1 = LDS(n_components=d)._E_step(Y, A, Q, C, R, mu0, V0)
    mean, S = lds_posterior_ref(Y, A, Q, C, R, mu0, V0)

    np.testing.assert_allclose(Ex, mean, atol=1e-10)
    for t in range(T):
        blk = S[t * d:(t + 1) * d, t * d:(t + 1) * d]
        np.testing.assert_allclose(Exx[t], blk + np.outer(mean[t], mean[t]),
                                   atol=1e-10)
    for t in range(T - 1):
        blk = S[(t + 1) * d:(t + 2) * d, t * d:(t + 1) * d]
        np.testing.assert_allclose(
            Exxm1[t], blk + np.outer(mean[t + 1], mean[t]), atol=1e-10)
