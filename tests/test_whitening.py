"""Rank-deficient input to the whitening-based methods must raise a clear
error rather than return non-finite or arbitrary components."""

import numpy as np
import pytest

from tsdr.slowness import SFA, BioSFA, ISFA
from tsdr.autocorrelation import TICA, TMCA, CSA, TCCA_Stiefel, sPCA_dwt
from tsdr.predictability import TLPC, PrCA, DiCCA, ForeCA, PFA, PTS
from tsdr.determinism import DyCA, DMD
from tsdr.nonstationarity import (ASSA, SSAsir, SSAsave, SSAcor, WaSSAf,
                                  WaSSAr, BSSnonstat)
from tsdr.latentvariable import LDS

from helpers import corr2, random_mix, ar1

METHODS = {
    "SFA": SFA, "BioSFA": BioSFA, "ISFA": ISFA, "TICA": TICA, "TMCA": TMCA, "CSA": CSA,
    "TCCA_Stiefel": TCCA_Stiefel, "sPCA_dwt": sPCA_dwt, "TLPC": TLPC,
    "PrCA": PrCA, "DiCCA": DiCCA, "ForeCA": ForeCA, "PFA": PFA, "PTS": PTS,
    "DyCA": DyCA, "DMD": DMD, "ASSA": ASSA,
    "SSAsir": SSAsir, "SSAsave": SSAsave, "SSAcor": SSAcor,
    "WaSSAf": WaSSAf, "WaSSAr": WaSSAr,
    "BSSnonstat": lambda: BSSnonstat(max_iters=2),
    "LDS": lambda: LDS(max_iter=2),
}


@pytest.fixture(scope="module")
def slow_source_data():
    rng = np.random.default_rng(0)
    T = 2000
    s = ar1(T, .995, rng)
    X, _ = random_mix(np.column_stack([s, rng.normal(size=(T, 4))]), seed=1)
    return s, X


def rank_deficient_cases(X):
    rng = np.random.default_rng(1)
    return {
        "constant channel": np.column_stack([X, np.full(len(X), 3.0)]),
        "duplicated channel": np.column_stack([X, X[:, 0]]),
        "fewer samples than features": rng.normal(size=(8, 20)),
    }


@pytest.mark.parametrize("case", ["constant channel", "duplicated channel",
                                  "fewer samples than features"])
@pytest.mark.parametrize("name", list(METHODS))
def test_rank_deficient_input_raises(slow_source_data, name, case):
    _, X = slow_source_data
    Xbad = rank_deficient_cases(X)[case]
    with pytest.raises(ValueError, match="singular"):
        METHODS[name]().fit(Xbad)


def test_full_rank_with_mixed_scales_is_accepted(slow_source_data):
    """Channels on very different (but resolvable) scales are not rejected."""
    s, X = slow_source_data
    Xs = X * np.array([1e-4, 1e-2, 1, 1e2, 1e4])
    Y = SFA().fit_transform(Xs)
    assert corr2(Y[:, 0], s) > 0.99
