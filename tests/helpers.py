"""Shared helpers for the tsdr tests: simulated sources, mixing, and
independent reference implementations used to check values."""

import numpy as np


def corr2(a, b):
    """Squared Pearson correlation between two series."""
    a = np.real(np.asarray(a)).ravel()
    b = np.real(np.asarray(b)).ravel()
    return np.corrcoef(a, b)[0, 1] ** 2


def subspace_cos(A, B):
    """Cosines of the principal angles between the column spaces of A and B
    (all ones = identical subspaces)."""
    qa, _ = np.linalg.qr(np.real(A))
    qb, _ = np.linalg.qr(np.real(B))
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def random_mix(S, seed=0):
    """Mix the columns of S with a random (non-orthogonal) matrix."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(S.shape[1], S.shape[1]))
    return S @ A, A


def ar1(T, phi, rng, sd=1.0):
    """AR(1) process x_t = phi x_{t-1} + e_t."""
    x = np.zeros(T)
    e = rng.normal(scale=sd, size=T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x


def unmixing(X, S):
    """Least-squares linear map from the centered data X to the centered
    sources S (spans the true source subspace in data coordinates)."""
    return np.linalg.lstsq(X - X.mean(0), S - S.mean(0), rcond=None)[0]


def whiten_ref(X):
    """Symmetric whitening: returns Xw, W, mu with Xw = (X - mu) @ W and
    cov(Xw) = I (ddof=1)."""
    mu = X.mean(0)
    ev, V = np.linalg.eigh(np.cov(X, rowvar=False))
    W = V @ np.diag(ev ** -0.5) @ V.T
    return (X - mu) @ W, W, mu
