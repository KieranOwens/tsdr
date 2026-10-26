"""The iterative methods with a fixed number of iterations raise a
ConvergenceWarning when the last iteration still changed the solution (or,
for DCA, when the optimiser reports failure), without changing the result."""

import warnings

import numpy as np
import pytest
import scipy.optimize
from sklearn.exceptions import ConvergenceWarning

from tsdr.slowness import BioSFA
from tsdr.autocorrelation import CSA
from tsdr.predictability import DiCCA, DCA
from tsdr.nonstationarity import BSSnonstat
from tsdr.latentvariable import LDS, DFA

from helpers import random_mix, ar1


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    T = 1000
    S = np.column_stack([ar1(T, .95, rng), ar1(T, .5, rng),
                         rng.normal(size=(T, 2)) * (1 + (np.arange(T) > T // 2))[:, None]])
    X, _ = random_mix(S, seed=1)
    return X


# too few iterations to converge
UNCONVERGED = {
    "BioSFA": lambda: BioSFA(n_components=2, n_iters=5, random_state=0),
    "CSA": lambda: CSA(n_components=2, max_iters=1),
    "BSSnonstat": lambda: BSSnonstat(n_components=2, max_iters=1, n_init=1, random_state=0),
    "LDS": lambda: LDS(n_components=2, max_iter=1),
    "DFA": lambda: DFA(n_components=2, max_iter=1),
    # DiCCA always runs 100 iterations; a negative tol flags any change
    "DiCCA": lambda: DiCCA(n_components=2, tol=-1),
}

# default iterations, which converge for these data
CONVERGED = {
    "BioSFA": lambda: BioSFA(n_components=1, random_state=0),
    "CSA": lambda: CSA(n_components=2),
    "BSSnonstat": lambda: BSSnonstat(n_components=1, random_state=0),
    "LDS": lambda: LDS(n_components=1),
    "DFA": lambda: DFA(n_components=1),
    "DiCCA": lambda: DiCCA(n_components=2),
    "DCA": lambda: DCA(n_components=2, T=3, n_init=2, random_state=0),
}


@pytest.mark.parametrize("name", list(UNCONVERGED))
def test_unconverged_fit_warns(data, name):
    with pytest.warns(ConvergenceWarning, match="converge"):
        UNCONVERGED[name]().fit(data)


@pytest.mark.parametrize("name", list(CONVERGED))
def test_converged_fit_does_not_warn(data, name):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        CONVERGED[name]().fit(data)


def test_tol_only_affects_the_warning(data):
    """tol changes whether a warning is raised, not the result."""
    with pytest.warns(ConvergenceWarning):
        Y1 = DiCCA(n_components=2, tol=-1).fit_transform(data)
    Y2 = DiCCA(n_components=2).fit_transform(data)
    np.testing.assert_array_equal(Y1, Y2)


def test_dca_warns_if_optimisation_fails(data, monkeypatch):
    minimize = scipy.optimize.minimize

    def minimize_one_iteration(*args, **kwargs):
        return minimize(*args, options={"maxiter": 1}, **kwargs)

    monkeypatch.setattr(scipy.optimize, "minimize", minimize_one_iteration)
    with pytest.warns(ConvergenceWarning, match="DCA"):
        DCA(n_components=2, T=3, n_init=1, random_state=0).fit(data)
