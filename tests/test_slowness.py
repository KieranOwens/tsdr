"""Value tests for the slowness methods."""

import numpy as np
import pytest

from tsdr.slowness import SFA, BioSFA

from helpers import corr2, subspace_cos, random_mix, ar1


@pytest.fixture(scope="module")
def slow_sources():
    """Three AR(1) sources of decreasing slowness and three white-noise
    sources, randomly mixed."""
    rng = np.random.default_rng(0)
    T = 5000
    S = np.column_stack([ar1(T, .999, rng), ar1(T, .99, rng),
                         ar1(T, .95, rng), rng.normal(size=(T, 3))])
    X, _ = random_mix(S, seed=1)
    return S, X


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_biosfa_matches_sfa_components(slow_sources, seed):
    """BioSFA learns the slow subspace, and ordering within it by slowness
    recovers each SFA component (not an arbitrary rotation of them),
    independently of the random initialisation."""
    _, X = slow_sources
    sfa = SFA(n_components=3).fit(X)
    bio = BioSFA(n_components=3, random_state=seed).fit(X)
    assert np.all(subspace_cos(bio.linear_transform_,
                               sfa.linear_transform_) > 0.9999)
    Ys, Yb = sfa.transform(X), bio.transform(X)
    for i in range(3):
        assert corr2(Yb[:, i], Ys[:, i]) > 0.999


def test_biosfa_unit_variance_and_ordered(slow_sources):
    """The outputs y = M^-1 W x satisfy the whitening constraint, and are
    ordered from slowest to fastest."""
    _, X = slow_sources
    bio = BioSFA(n_components=3, random_state=0).fit(X)
    Y = bio.transform(X)
    np.testing.assert_allclose(np.cov(Y, rowvar=False), np.eye(3), atol=1e-3)
    assert np.all(np.diff(bio.eigvals_) > 0)
    slowness = np.var(np.diff(Y, axis=0), axis=0)
    assert np.all(np.diff(slowness) > 0)


@pytest.mark.parametrize("lr, tau", [(0.1, 0.1), (0.2, 0.1), (0.0, 1.0)])
def test_biosfa_requires_lr_below_tau(slow_sources, lr, tau):
    _, X = slow_sources
    with pytest.raises(ValueError, match="lr must satisfy"):
        BioSFA(lr=lr, tau=tau).fit(X)
