"""The sinusoid-in-noise benchmark of Owens & Fulcher (2025), Sec. IV: every
linear method should recover the sinusoid as its first component."""

import numpy as np
import pytest
import scipy

from tsdr.slowness import SFA, BioSFA
from tsdr.autocorrelation import TICA, TMCA, CSA, TCCA_Stiefel, sPCA_dwt
from tsdr.predictability import TLPC, PrCA, DiCCA, ForeCA, PFA
from tsdr.determinism import DyCA, DMD
from tsdr.nonstationarity import (ASSA, SSAsir, SSAsave, SSAcor, WaSSAf,
                                  WaSSAr, BSSnonstat)

from helpers import corr2


def make_benchmark(T=1000, D=10, sigma2=1.0, seed=0):
    rng = np.random.default_rng(seed)
    z = np.sin(np.linspace(0, 4 * np.pi, T))
    noise = rng.normal(scale=np.sqrt(sigma2), size=(T, D - 1))
    A = scipy.stats.ortho_group.rvs(D, random_state=seed)
    return z, np.concatenate([noise, z[:, None]], axis=1) @ A


METHODS = {
    "SFA": lambda: SFA(),
    "BioSFA": lambda: BioSFA(),
    "TICA": lambda: TICA(lag=10),
    "TMCA": lambda: TMCA(lag=10),
    "CSA": lambda: CSA(lag=5),
    "TCCA_Stiefel": lambda: TCCA_Stiefel(lag=10),
    "sPCA_dwt": lambda: sPCA_dwt(),
    "TLPC": lambda: TLPC(),
    "PrCA(diff)": lambda: PrCA(),
    "PrCA(linear_reg)": lambda: PrCA(model="linear_reg", p=2),
    "DiCCA": lambda: DiCCA(),
    "ForeCA": lambda: ForeCA(random_state=0),
    "PFA": lambda: PFA(),
    "DyCA": lambda: DyCA(),
    "DMD": lambda: DMD(),
    "ASSA": lambda: ASSA(),
    "SSAsir": lambda: SSAsir(),
    "SSAsave": lambda: SSAsave(),
    "SSAcor": lambda: SSAcor(),
    "WaSSAf": lambda: WaSSAf(),
    "WaSSAr": lambda: WaSSAr(),
    "BSSnonstat": lambda: BSSnonstat(max_iters=20),
}


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", list(METHODS))
def test_sinusoid_benchmark(name):
    z, X = make_benchmark()
    np.random.seed(1)
    Y = np.asarray(METHODS[name]().fit_transform(X))
    assert Y.shape == (len(z), 1)
    assert corr2(Y[:, 0], z) > 0.9
