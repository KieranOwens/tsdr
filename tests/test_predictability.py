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
