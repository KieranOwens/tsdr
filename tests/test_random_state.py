"""The methods with a random initialisation accept a random_state argument:
an int gives reproducible results and leaves the global NumPy random state
unchanged, and None uses the global NumPy random state."""

import numpy as np
import pytest

from tsdr.slowness import BioSFA, ISFA, SICA
from tsdr.nonstationarity import BSSnonstat
from tsdr.latentvariable import LDS, DFA
from tsdr.autocorrelation import TCCA_Stiefel
from tsdr.predictability import ForeCA, DCA

from helpers import random_mix, ar1

METHODS = {
    "BioSFA": lambda rs: BioSFA(n_components=2, n_iters=50, random_state=rs),
    "ISFA": lambda rs: ISFA(n_components=2, lags=3, random_state=rs),
    "SICA": lambda rs: SICA(n_components=2, random_state=rs),
    "BSSnonstat": lambda rs: BSSnonstat(n_components=2, max_iters=5,
                                        random_state=rs),
    "LDS": lambda rs: LDS(n_components=2, max_iter=2, init='random', random_state=rs),
    "DFA": lambda rs: DFA(n_components=2, max_iter=2, init='random', random_state=rs),
    "ForeCA": lambda rs: ForeCA(n_components=2, n_starts=2, random_state=rs),
    "DCA": lambda rs: DCA(n_components=2, T=2, n_init=2, random_state=rs),
    "TCCA_Stiefel": lambda rs: TCCA_Stiefel(n_components=2, max_iters=20,
                                            random_state=rs),
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    T = 500
    S = np.column_stack([ar1(T, .95, rng), rng.normal(size=(T, 3))])
    X, _ = random_mix(S, seed=1)
    return X


@pytest.mark.parametrize("name", list(METHODS))
def test_int_random_state_is_reproducible(data, name):
    Y1 = np.asarray(METHODS[name](0).fit_transform(data))
    Y2 = np.asarray(METHODS[name](0).fit_transform(data))
    np.testing.assert_array_equal(Y1, Y2)


@pytest.mark.parametrize("name", list(METHODS))
def test_int_random_state_keeps_global_random_state(data, name):
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    METHODS[name](0).fit(data)
    assert np.random.rand() == expected


@pytest.mark.parametrize("name", list(METHODS))
def test_random_state_instance_and_none(data, name):
    """A RandomState instance is used as given, and None uses the global
    NumPy random state (so np.random.seed still controls the result)."""
    Y1 = np.asarray(METHODS[name](np.random.RandomState(7)).fit_transform(data))
    np.random.seed(7)
    Y2 = np.asarray(METHODS[name](None).fit_transform(data))
    np.testing.assert_array_equal(Y1, Y2)
