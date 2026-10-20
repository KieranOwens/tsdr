"""Value tests for the predictability methods (TLPC, PrCA, DiCCA)."""

import numpy as np
import pytest

from tsdr.predictability import TLPC, PrCA, DiCCA

from helpers import corr2, subspace_cos, random_mix, ar1, unmixing
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
