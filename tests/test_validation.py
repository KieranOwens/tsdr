"""Checks of the input and parameters of every method: invalid input raises a
clear ValueError, a DataFrame is accepted (and gives the same result as the
equivalent ndarray), and no method changes its input."""

import numpy as np
import pytest

from tsdr.slowness import SFA, BioSFA, ISFA, SICA
from tsdr.autocorrelation import TICA, TMCA, CSA, TCCA_Stiefel, sPCA_dwt
from tsdr.predictability import TLPC, PrCA, DiCCA, ForeCA, PFA, PTS, DCA
from tsdr.determinism import DyCA, DMD
from tsdr.nonstationarity import (ASSA, SSAsir, SSAsave, SSAcor, WaSSAf,
                                  WaSSAr, BSSnonstat)
from tsdr.diffusion import DMcov, DIG, NLSA
from tsdr.latentvariable import LDS, DFA

from helpers import random_mix, ar1

# all 30 methods, as functions of n_components, with small settings so that
# the tests are fast
METHODS = {
    "SFA": lambda n: SFA(n_components=n),
    "BioSFA": lambda n: BioSFA(n_components=n, n_iters=50, random_state=0),
    "ISFA": lambda n: ISFA(n_components=n, lags=2, max_iters=5, random_state=0),
    "SICA": lambda n: SICA(n_components=n, max_iters=20, random_state=0),
    "TICA": lambda n: TICA(n_components=n),
    "TMCA": lambda n: TMCA(n_components=n),
    "TCCA_Stiefel": lambda n: TCCA_Stiefel(n_components=n, max_iters=50),
    "CSA": lambda n: CSA(n_components=n, max_iters=20),
    "sPCA_dwt": lambda n: sPCA_dwt(n_components=n),
    "TLPC": lambda n: TLPC(n_components=n),
    "PrCA": lambda n: PrCA(n_components=n),
    "DiCCA": lambda n: DiCCA(n_components=n, s=2),
    "ForeCA": lambda n: ForeCA(n_components=n, n_starts=2, max_iters=20, random_state=0),
    "PFA": lambda n: PFA(n_components=n),
    "PTS": lambda n: PTS(n_components=n, max_iters=10),
    "DCA": lambda n: DCA(n_components=n, T=2, n_init=1, random_state=0),
    "DyCA": lambda n: DyCA(n_components=n),
    "DMD": lambda n: DMD(n_components=n),
    "ASSA": lambda n: ASSA(n_components=n),
    "SSAsir": lambda n: SSAsir(n_components=n),
    "SSAsave": lambda n: SSAsave(n_components=n),
    "SSAcor": lambda n: SSAcor(n_components=n),
    "WaSSAf": lambda n: WaSSAf(n_components=n),
    "WaSSAr": lambda n: WaSSAr(n_components=n),
    "BSSnonstat": lambda n: BSSnonstat(n_components=n, max_iters=5, n_init=1, random_state=0),
    "DMcov": lambda n: DMcov(40, 20, n_components=n),
    "DIG": lambda n: DIG(40, 20, n_components=n),
    "NLSA": lambda n: NLSA(n_components=n, q=3, n_neighbours=20, n_eigenfunctions=5),
    "LDS": lambda n: LDS(n_components=n, max_iter=2),
    "DFA": lambda n: DFA(n_components=n, max_iter=2),
}

# methods with an out-of-sample transform
TRANSFORM_METHODS = [name for name in METHODS
                     if name not in ("sPCA_dwt", "DMcov", "DIG", "NLSA")]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    T = 300
    S = np.column_stack([ar1(T, .95, rng), ar1(T, .5, rng), rng.normal(size=T)])
    X, _ = random_mix(S, seed=1)
    return X


def too_many_components(name, D):
    """The smallest invalid n_components for D features: DyCA has up to two
    columns per eigenvector, and the diffusion methods have one component
    per window or eigenfunction rather than per feature."""
    return {"DyCA": 2 * D + 1, "DMcov": 14, "DIG": 15, "NLSA": 6}.get(name, D + 1)


@pytest.mark.parametrize("name", list(METHODS))
def test_n_components_out_of_range_raises(data, name):
    D = data.shape[1]
    for n in [0, too_many_components(name, D)]:
        with pytest.raises(ValueError, match="n_components"):
            METHODS[name](n).fit(data)


@pytest.mark.parametrize("name", list(METHODS))
def test_non_integer_n_components_raises(data, name):
    with pytest.raises(ValueError, match="n_components"):
        METHODS[name](1.5).fit(data)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("name", list(METHODS))
def test_non_finite_input_raises(data, name, bad):
    X = data.copy()
    X[10, 1] = bad
    with pytest.raises(ValueError, match="NaN|infinity"):
        METHODS[name](1).fit(X)


@pytest.mark.parametrize("name", list(METHODS))
def test_1d_input_raises(data, name):
    with pytest.raises(ValueError, match="2D"):
        METHODS[name](1).fit(data[:, 0])


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", TRANSFORM_METHODS)
def test_transform_checks_input(data, name):
    model = METHODS[name](1).fit(data)
    X = data.copy()
    X[10, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        model.transform(X)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", list(METHODS))
def test_dataframe_input_equals_ndarray_input(data, name):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(data, columns=["a", "b", "c"])

    # the ndarray of the DataFrame's values (in the DataFrame's column-major 
    # memory layout, for which matrix products can round differently than 
    # for the row-major `data`, and iterative methods can amplify this)
    X = df.to_numpy()
    Y_array = METHODS[name](2).fit_transform(X)

    Y_df = METHODS[name](2).fit_transform(df)
    assert isinstance(Y_df, np.ndarray)
    np.testing.assert_array_equal(Y_df, Y_array)
    if name in TRANSFORM_METHODS:
        model = METHODS[name](2).fit(df)
        np.testing.assert_array_equal(model.transform(df), Y_array)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("name", list(METHODS))
def test_input_is_not_modified(data, name):
    X = data.copy()
    METHODS[name](2).fit_transform(X)
    np.testing.assert_array_equal(X, data)
    if name in TRANSFORM_METHODS:
        model = METHODS[name](2).fit(X)
        model.transform(X)
        np.testing.assert_array_equal(X, data)


def test_larger_n_components_where_valid(data):
    """n_components may exceed n_features where the method allows it."""
    D = data.shape[1]
    assert DyCA(n_components=2 * D).fit_transform(data).shape[1] == 2 * D
    model = LDS(n_components=D + 1, max_iter=2, init='random', random_state=0)
    assert model.fit_transform(data).shape[1] == D + 1
    assert DMcov(40, 20, n_components=D + 2).fit_transform(data).shape[1] == D + 2
    assert NLSA(n_components=D + 2, q=3, n_neighbours=20).fit_transform(data).shape[1] == D + 2


def test_lds_autocorrelation_init_needs_n_components_at_most_n_features(data):
    with pytest.raises(ValueError, match="n_components"):
        LDS(n_components=data.shape[1] + 1).fit(data)


@pytest.mark.parametrize("make", [
    lambda: DMcov(1, 10),                        # window too short
    lambda: DMcov(301, 10),                      # window longer than the data
    lambda: DMcov(40, 0),                        # step not positive
    lambda: DMcov(3, 10, distance='geodesic'),   # singular window covariances
    lambda: DIG(1, 10),
    lambda: DIG(40, 2.5),
])
def test_invalid_sliding_window_raises(data, make):
    with pytest.raises(ValueError, match="window|step"):
        make().fit(data)


@pytest.mark.parametrize("make", [
    lambda: ASSA(n_windows=0),
    lambda: ASSA(n_windows=200),         # windows of 1 sample
    lambda: SSAsave(n_windows=200),
    lambda: SSAcor(lag=5, n_windows=50), # windows of 6 < lag + 2 samples
    lambda: WaSSAf(n_windows=301),
    lambda: WaSSAr(n_windows=2.0),
    lambda: SSAsir(n_windows=301),       # windows of 0 samples
])
def test_invalid_n_windows_raises(data, make):
    with pytest.raises(ValueError, match="n_windows"):
        make().fit(data)


@pytest.mark.parametrize("make", [
    lambda: SSAcor(lag=0),
    lambda: SSAcor(lag=300),
    lambda: BSSnonstat(lag=0),
    lambda: BSSnonstat(lag=300),
])
def test_invalid_lag_raises(data, make):
    with pytest.raises(ValueError, match="lag"):
        make().fit(data)
