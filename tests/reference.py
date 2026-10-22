"""Independent reference implementations of TSDR methods, written from the
original papers, used to check the values returned by tsdr."""

import numpy as np
import scipy.linalg

from helpers import whiten_ref


def ar_R2(t, s):
    """In-sample R^2 of a least-squares AR(s) fit to t."""
    t = t - t.mean()
    Z = np.column_stack([t[s - k - 1: len(t) - k - 1] for k in range(s)])
    y = t[s:]
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    return 1 - np.var(y - Z @ b) / np.var(y)


def dicca_ref(X, n_components, s, iters=200):
    """DiCCA (Dong & Qin, 2018, IFAC ADCHEM, Sec. 2.2-2.4) on whitened data:
    beta from least squares on the plain lagged scores, w from the
    generalized eigenproblem, deflation X <- X - t p^T, and the PLS-style
    weights R = W (P^T W)^-1 that map the original data to each score."""
    Xw, Wwh, _ = whiten_ref(X)
    Xd = Xw.copy()
    T, D = Xd.shape
    Ws, Ps = [], []
    for _ in range(n_components):
        Xs1 = Xd[s:]
        Xlag = [Xd[s - i: T - i] for i in range(1, s + 1)]
        w = np.linalg.eigh(Xs1.T @ Xlag[0] + Xlag[0].T @ Xs1)[1][:, -1]
        for _ in range(iters):
            t = Xd @ w
            nt = np.linalg.norm(t)
            t, w = t / nt, w / nt
            Ts = np.column_stack([t[s - i: T - i] for i in range(1, s + 1)])
            ts1 = t[s:]
            beta = np.linalg.lstsq(Ts, ts1, rcond=None)[0]
            beta = beta / np.sqrt(ts1 @ Ts @ beta)
            Xb = sum(beta[i] * Xlag[i] for i in range(s))
            M1 = Xs1.T @ Xs1 + Xb.T @ Xb
            M2 = Xs1.T @ Xb + Xb.T @ Xs1
            ev, V = np.linalg.eig(np.linalg.pinv(M1) @ M2)
            w_new = np.real(V[:, np.argmax(np.real(ev))])
            w_new *= np.sign(w_new @ w) if w_new @ w != 0 else 1
            converged = np.linalg.norm(w_new / np.linalg.norm(w_new)
                                       - w / np.linalg.norm(w)) < 1e-10
            w = w_new
            if converged:
                break
        w = w / np.linalg.norm(Xd @ w)
        t = Xd @ w
        p = Xd.T @ t / (t @ t)
        Ws.append(w)
        Ps.append(p)
        Xd = Xd - np.outer(t, p)
    Wm, Pm = np.column_stack(Ws), np.column_stack(Ps)
    return Wwh @ Wm @ np.linalg.inv(Pm.T @ Wm)


def tlpc_ref(X, n_components, lag=1):
    """TLPC as the Gaussian information bottleneck between x_t and x_{t+lag}
    (Creutzig & Sprekeler, 2008; Sprekeler 2008 thesis, eq. 6.22):
    C_{x_t|x_{t+lag}} w = lambda C_{x_t} w, which in whitened coordinates is
    (I - C_tau C_tau^T) w = lambda w."""
    Xw, W, _ = whiten_ref(X)
    T = len(Xw)
    C = Xw[:-lag].T @ Xw[lag:] / (T - lag - 1)
    lam, V = np.linalg.eigh(np.eye(X.shape[1]) - C @ C.T)
    return W @ V[:, :n_components], lam


def prca_ref(X, n_components, p=1, lag=1):
    """Multivariate linear (VAR) PrCA (Schneider & Griffies, 1999; Owens &
    Fulcher, 2025, Eqs. 26-27): regress x_{t+lag} on [x_t, ..., x_{t-p+1}],
    then solve C_E w = lambda C_0 w on whitened data."""
    Xw, W, _ = whiten_ref(X)
    T, D = Xw.shape
    rows = T - (p - 1) - lag
    Z = np.column_stack([Xw[p - 1 - k: p - 1 - k + rows] for k in range(p)]
                        + [np.ones(rows)])
    Y = Xw[p - 1 + lag:]
    B = np.linalg.lstsq(Z, Y, rcond=None)[0]
    lam, V = np.linalg.eigh(np.cov(Y - Z @ B, rowvar=False))
    return W @ V[:, :n_components], lam


def dmd_ref(X):
    """Exact DMD (Schmid, 2010; Tu et al., 2014, Algorithm 2) without rank
    truncation, on mean-centred data with snapshots as columns: the reduced
    operator A~ = U^T X1 V S^-1 from the SVD X0 = U S V^T, its eigenvalues
    lam, and the exact modes Phi = X1 V S^-1 W / lam, ordered by |lam|."""
    Xc = (X - X.mean(0)).T
    X0, X1 = Xc[:, :-1], Xc[:, 1:]
    U, s, Vh = np.linalg.svd(X0, full_matrices=False)
    At = U.T @ X1 @ Vh.T / s
    lam, W = np.linalg.eig(At)
    Phi = X1 @ Vh.T / s @ W / lam
    order = np.argsort(-np.abs(lam), kind="stable")
    return lam[order], Phi[:, order]


def lds_posterior_ref(Y, A, Q, C, R, mu0, V0):
    """Exact posterior of the latent states of a linear dynamical system
    (Ghahramani & Hinton, 1996) by direct Gaussian conditioning of the joint
    prior of x_1, ..., x_T on Y (feasible for short series only). Returns the
    posterior means (T, d) and covariance (T*d, T*d), in which block (t, s) is
    Cov(x_t, x_s | Y)."""
    T, d = len(Y), A.shape[0]
    P = [V0]
    for _ in range(1, T):
        P.append(A @ P[-1] @ A.T + Q)
    mu = [mu0]
    for _ in range(1, T):
        mu.append(A @ mu[-1])
    mu = np.concatenate(mu)
    S = np.zeros((T * d, T * d))
    for s in range(T):
        for t in range(s, T):
            blk = np.linalg.matrix_power(A, t - s) @ P[s]
            S[t * d:(t + 1) * d, s * d:(s + 1) * d] = blk
            S[s * d:(s + 1) * d, t * d:(t + 1) * d] = blk.T
    Cb, Rb = np.kron(np.eye(T), C), np.kron(np.eye(T), R)
    G = S @ Cb.T @ np.linalg.inv(Cb @ S @ Cb.T + Rb)
    mean = mu + G @ (Y.ravel() - Cb @ mu)
    return mean.reshape(T, d), S - G @ Cb @ S


def wassa_S_ref(X, n_windows, barycenter):
    """The WaSSA matrix S of Kaltenstadler et al. (2018), eq. (21), for the
    given barycenter, with each window modelled as N(mu_e, Sigma_e) (sample
    mean and covariance of the window)."""
    sqrtm = lambda A: np.real(scipy.linalg.sqrtm(A))
    T, D = X.shape
    L = T // n_windows
    S = np.zeros((D, D))
    for i in range(n_windows):
        Xi = X[i * L:(i + 1) * L]
        C = np.cov(Xi, rowvar=False)
        Cr = sqrtm(C)
        S += (np.outer(Xi.mean(0), Xi.mean(0)) + C + barycenter
              - 2 * sqrtm(Cr @ barycenter @ Cr))
    return S


def wassar_S_ref(X, n_windows):
    """The approximate WaSSA matrix S of Kaltenstadler et al. (2018),
    eq. (23), with the root-Frechet mean of eq. (15), and each window
    modelled as N(mu_e, Sigma_e)."""
    sqrtm = lambda A: np.real(scipy.linalg.sqrtm(A))
    T, D = X.shape
    L = T // n_windows
    wins = [X[i * L:(i + 1) * L] for i in range(n_windows)]
    roots = [sqrtm(np.cov(Xi, rowvar=False)) for Xi in wins]
    R = np.mean(roots, axis=0)
    return sum(np.outer(Xi.mean(0), Xi.mean(0)) + (Cr - R) @ (Cr - R).T
               for Xi, Cr in zip(wins, roots))


def foreca_omega_ref(y, bandwidth=5, prior_weight=1e-3):
    """Forecastability Omega = 1 - H of Goerg (2013), eqs. (12) and (15):
    the discrete spectral entropy (base T) of the periodogram of y at the
    T Fourier frequencies, computed by an explicit DFT sum, smoothed by a
    circular Daniell kernel of width 2 * bandwidth + 1, normalised to sum to
    one, and mixed with a uniform prior of weight prior_weight."""
    y = np.asarray(y, dtype=float)
    T = len(y)
    t = np.arange(T)
    dft = np.exp(-2j * np.pi * np.outer(t, t) / T) @ (y - y.mean())
    I = np.abs(dft) ** 2
    I = I / I.sum()
    f = np.mean([np.roll(I, l) for l in range(-bandwidth, bandwidth + 1)], axis=0)
    f = (1 - prior_weight) * f + prior_weight / T
    return 1 + np.sum(f * np.log(f)) / np.log(T)


def pfa_ref(X, n_components, p=2, k=4):
    """PFA (Richthofer & Wiskott, 2013, eqs. 7, 17, 22-25) on whitened data,
    computed time step by time step: the history zeta(t) = [z(t-1), ...,
    z(t-p)], the least squares predictions z(t) ~ W zeta(t) and zeta(t+1) ~
    V zeta(t), and the eigenvectors with the least eigenvalues of the summed
    error covariance of the iterated predictions z(t) ~ W V^i zeta(t-i)."""
    Z, Wh, _ = whiten_ref(X)
    T, D = Z.shape
    hist = lambda t: np.concatenate([Z[t - j] for j in range(1, p + 1)])
    mean_outer = lambda pairs: np.mean([np.outer(a, b) for a, b in pairs], axis=0)
    W = (mean_outer((Z[t], hist(t)) for t in range(p, T))
         @ np.linalg.pinv(mean_outer((hist(t), hist(t)) for t in range(p, T))))
    V = (mean_outer((hist(t + 1), hist(t)) for t in range(p, T - 1))
         @ np.linalg.pinv(mean_outer((hist(t), hist(t)) for t in range(p, T - 1))))
    C = np.zeros((D, D))
    for i in range(k + 1):
        WVi = W @ np.linalg.matrix_power(V, i)
        errors = [Z[t] - WVi @ hist(t - i) for t in range(p + i, T)]
        C += mean_outer((e, e) for e in errors)
    eigvals, U = np.linalg.eigh(C)
    return Wh @ U[:, :n_components], eigvals
