"""Value tests for the non-stationarity methods."""

import numpy as np
import pytest

from tsdr.nonstationarity import BSSnonstat

from helpers import corr2, random_mix


@pytest.fixture(scope="module")
def variance_sources():
    """Three sources with smoothly varying variance plus two stationary
    Gaussian sources, randomly mixed."""
    rng = np.random.default_rng(0)
    T = 3000

    def smooth_var_source(period):
        phase = rng.uniform(0, 2 * np.pi)
        env = np.exp(np.sin(2 * np.pi * np.arange(T) / period + phase))
        return env * rng.normal(size=T)

    S = np.column_stack([smooth_var_source(400), smooth_var_source(150),
                         smooth_var_source(900), rng.normal(size=(T, 2))])
    X, _ = random_mix(S, seed=9)
    return S, X


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bssnonstat_recovers_all_sources(variance_sources, seed):
    S, X = variance_sources
    Y = BSSnonstat(n_components=3, max_iters=30,
                   random_state=seed).fit_transform(X)
    R2 = np.array([[corr2(Y[:, i], S[:, k]) for k in range(5)]
                   for i in range(3)])
    # each component is a different non-stationary source
    assert sorted(R2.argmax(1)) == [0, 1, 2]
    assert np.all(R2.max(1) > 0.99)
