"""The linear methods are fitted to the centred (whitened) data, so transform
subtracts the mean of the fitted data before applying linear_transform_:
the components of the fitted data have zero mean, fit_transform(X) equals
fit(X).transform(X), and the components do not change when a constant is
added to the data."""

import numpy as np
import pytest

from tsdr.slowness import SFA, BioSFA, ISFA, SICA
from tsdr.autocorrelation import TICA, TMCA, TCCA_Stiefel, CSA
from tsdr.predictability import TLPC, PrCA, DiCCA, ForeCA, PFA, PTS, DCA
from tsdr.determinism import DyCA, DMD
from tsdr.nonstationarity import (ASSA, SSAsir, SSAsave, SSAcor, WaSSAf,
                                  WaSSAr, BSSnonstat)
from tsdr.latentvariable import LDS, DFA

from helpers import random_mix, ar1, subspace_cos

LINEAR = {
    "SFA": lambda: SFA(n_components=2),
    "BioSFA": lambda: BioSFA(n_components=2, n_iters=50, random_state=0),
    "ISFA": lambda: ISFA(n_components=2, lags=2, max_iters=5, random_state=0),
    "SICA": lambda: SICA(n_components=2, max_iters=20, random_state=0),
    "TICA": lambda: TICA(n_components=2),
    "TMCA": lambda: TMCA(n_components=2),
    "TCCA_Stiefel": lambda: TCCA_Stiefel(n_components=2, max_iters=50),
    "TCCA_Stiefel_nowhiten": lambda: TCCA_Stiefel(n_components=2, max_iters=50,
                                                  whiten=False),
    "CSA": lambda: CSA(n_components=2, max_iters=20),
    "TLPC": lambda: TLPC(n_components=2),
    "PrCA": lambda: PrCA(n_components=2),
    "DiCCA": lambda: DiCCA(n_components=2, s=2),
    "ForeCA": lambda: ForeCA(n_components=2, n_starts=2, max_iters=20, random_state=0),
    "PFA": lambda: PFA(n_components=2),
    "PTS": lambda: PTS(n_components=2, max_iters=10),
    "DCA": lambda: DCA(n_components=2, T=2, n_init=1, random_state=0),
    "DyCA": lambda: DyCA(n_components=2),
    "DMD": lambda: DMD(n_components=2),
    "ASSA": lambda: ASSA(n_components=2),
    "SSAsir": lambda: SSAsir(n_components=2),
    "SSAsave": lambda: SSAsave(n_components=2),
    "SSAcor": lambda: SSAcor(n_components=2),
    "WaSSAf": lambda: WaSSAf(n_components=2),
    "WaSSAr": lambda: WaSSAr(n_components=2),
    "BSSnonstat": lambda: BSSnonstat(n_components=2, max_iters=5, random_state=0),
}

# LDS and DFA are not linear maps, but are also fitted to centred data
LATENT = {
    "LDS": lambda: LDS(n_components=2, max_iter=3),
    "LDS_nowhiten": lambda: LDS(n_components=2, max_iter=3, whiten=False),
    "DFA_nowhiten": lambda: DFA(n_components=2, max_iter=3, whiten=False),
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    T = 600
    S = np.column_stack([ar1(T, .95, rng), np.sin(np.arange(T) / 20),
                         rng.normal(size=(T, 2))])
    X, _ = random_mix(S, seed=1)
    # a large mean, so that an uncentred transform would be easy to detect
    return X + np.array([50.0, -20.0, 10.0, 5.0])


@pytest.mark.parametrize("name", list(LINEAR))
def test_components_of_fitted_data_have_zero_mean(data, name):
    Y = LINEAR[name]().fit_transform(data)
    np.testing.assert_allclose(Y.mean(axis=0), 0, atol=1e-8 * np.abs(Y).max())


@pytest.mark.parametrize("name", list(LINEAR) + list(LATENT))
def test_fit_transform_equals_fit_then_transform(data, name):
    make = {**LINEAR, **LATENT}[name]
    Y1 = make().fit_transform(data)
    Y2 = make().fit(data).transform(data)
    np.testing.assert_allclose(Y1, Y2, rtol=1e-10, atol=1e-10)


# DCA's objective depends only on the subspace of the components, so their
# basis within it is sensitive to rounding (see the DCA test below)
SHIFT_INVARIANT = [n for n in list(LINEAR) + list(LATENT) if n != "DCA"]


@pytest.mark.parametrize("name", SHIFT_INVARIANT)
def test_components_do_not_depend_on_the_mean(data, name):
    """Adding a constant to the data does not change the components (up to
    their arbitrary signs)."""
    make = {**LINEAR, **LATENT}[name]
    Y = make().fit_transform(data)
    Y_shifted = make().fit_transform(data + 1000.0)
    signs = np.sign(np.sum(Y * Y_shifted, axis=0))
    np.testing.assert_allclose(Y_shifted * signs, Y, rtol=1e-5,
                               atol=1e-5 * np.abs(Y).max())


def test_dca_subspace_does_not_depend_on_the_mean(data):
    W = LINEAR["DCA"]().fit(data).linear_transform_
    W_shifted = LINEAR["DCA"]().fit(data + 1000.0).linear_transform_
    # (to the tolerance of its optimiser)
    assert np.all(subspace_cos(W, W_shifted) > 0.9999)


def test_dmd_without_whitening_is_not_centred(data):
    """Exact DMD without whitening uses the data as given (mean_ is zero)."""
    model = DMD(n_components=2, whiten=False).fit(data)
    np.testing.assert_array_equal(model.mean_, 0)
    np.testing.assert_array_equal(model.transform(data), data @ model.linear_transform_)


def test_transform_of_new_data_uses_the_fitted_mean(data):
    """transform subtracts the mean of the fitted data, not of the new data."""
    model = SFA(n_components=2).fit(data)
    X_new = data[:100] + 3.0
    np.testing.assert_allclose(model.transform(X_new),
                               (X_new - data.mean(axis=0)) @ model.linear_transform_)
