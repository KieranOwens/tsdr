"""The sign of each component is fixed: for the linear methods, the loading
(entry of linear_transform_) of largest magnitude of each component is
positive; for sPCA_dwt, DMcov, DIG and NLSA, which have no loadings, the
value of largest magnitude of each component is positive. (DMD has its own
convention, and LDS and DFA are not sign-fixed.)"""

import numpy as np
import pytest

from tsdr._linalg import largest_entry_signs
from test_validation import METHODS, TRANSFORM_METHODS

from helpers import random_mix, ar1

LINEAR = [n for n in TRANSFORM_METHODS if n not in ("DMD", "LDS", "DFA")]
NONLINEAR = ["sPCA_dwt", "DMcov", "DIG", "NLSA"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    T = 400
    S = np.column_stack([ar1(T, .95, rng), np.sin(np.arange(T) / 20),
                         rng.normal(size=(T, 2))])
    X, _ = random_mix(S, seed=1)
    return X


def test_largest_entry_signs():
    M = np.array([[1.0, -3.0, 0.0], [-2.0, 1.0, 0.0]])
    np.testing.assert_array_equal(largest_entry_signs(M), [-1, -1, 1])


@pytest.mark.parametrize("name", LINEAR)
def test_largest_loading_is_positive(data, name):
    W = METHODS[name](2).fit(data).linear_transform_
    rows = np.argmax(np.abs(W), axis=0)
    assert np.all(W[rows, np.arange(W.shape[1])] > 0)


@pytest.mark.parametrize("name", NONLINEAR)
def test_largest_component_value_is_positive(data, name):
    Y = METHODS[name](2).fit_transform(data)
    rows = np.argmax(np.abs(Y), axis=0)
    assert np.all(Y[rows, np.arange(Y.shape[1])] > 0)
