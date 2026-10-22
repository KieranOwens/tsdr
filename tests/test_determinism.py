"""Value tests for the determinism methods."""

import numpy as np
import pytest

from tsdr.determinism import DMD

from helpers import subspace_cos, random_mix, ar1
from reference import dmd_ref


@pytest.fixture(scope="module")
def oscillator_sources():
    """A damped rotation (complex-conjugate eigenvalue pair), two AR(1)
    sources and white noise, randomly mixed, with a nonzero mean."""
    rng = np.random.default_rng(1)
    T = 4000
    r, th = 0.97, 0.3
    R = r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    osc = np.zeros((T, 2))
    for t in range(1, T):
        osc[t] = R @ osc[t - 1] + rng.normal(size=2)
    S = np.column_stack([osc, ar1(T, .9, rng), ar1(T, .5, rng),
                         rng.normal(size=(T, 2))])
    X, _ = random_mix(S, seed=2)
    return S, X + 5.0


def matched_ref(model, X):
    """Reference eigenvalues and modes in the order of model.eigvals_."""
    lam, Phi = dmd_ref(X)
    idx = [np.argmin(np.abs(lam - l)) for l in model.eigvals_]
    return lam[idx], Phi[:, idx]


def centred(X):
    return X - X.mean(0)


@pytest.mark.parametrize("whiten", [True, False])
def test_dmd_matches_exact_dmd(oscillator_sources, whiten):
    """Eigenvalues and modes equal exact DMD (Tu et al., 2014)."""
    _, X = oscillator_sources
    Xin = X if whiten else centred(X)
    model = DMD(n_components=X.shape[1], whiten=whiten).fit(Xin)
    lam, Phi = matched_ref(model, X)
    np.testing.assert_allclose(model.eigvals_, lam, atol=1e-10)
    for k in range(X.shape[1]):
        cos = abs(np.vdot(model.modes_[:, k], Phi[:, k])) / (
            np.linalg.norm(model.modes_[:, k]) * np.linalg.norm(Phi[:, k]))
        assert cos > 1 - 1e-10


def test_dmd_eigenvalue_ordering_and_pairs(oscillator_sources):
    _, X = oscillator_sources
    ev = DMD().fit(X).eigvals_
    assert np.all(np.diff(np.abs(ev)) <= 1e-12)
    # the leading pair is the damped rotation, positive imaginary part first
    np.testing.assert_allclose(abs(ev[0]), 0.97, atol=0.01)
    np.testing.assert_allclose(np.angle(ev[0]), 0.3, atol=0.01)
    assert ev[1] == np.conj(ev[0])


@pytest.mark.parametrize("whiten", [True, False])
def test_dmd_components_evolve_by_eigenvalue(oscillator_sources, whiten):
    """The complex components satisfy z_{t+1} ~ lambda z_t, and the modes
    reconstruct the (centred) data from all components."""
    _, X = oscillator_sources
    Xin = X if whiten else centred(X)
    D = X.shape[1]
    model = DMD(n_components=D, whiten=whiten, return_complex=True).fit(Xin)
    Z = centred(X) @ model.linear_transform_
    coef = np.sum(np.conj(Z[:-1]) * Z[1:], 0) / np.sum(np.abs(Z[:-1]) ** 2, 0)
    np.testing.assert_allclose(coef, model.eigvals_, atol=1e-10)
    np.testing.assert_allclose(Z @ model.modes_.T, centred(X), atol=1e-8)


def test_dmd_real_pair_spans_complex_component(oscillator_sources):
    """The real components of a conjugate pair are (Re z, Im z), which are
    orthogonal over time, and span the oscillator subspace."""
    S, X = oscillator_sources
    Wr = DMD(n_components=2).fit(X).linear_transform_
    wc = DMD(n_components=1, return_complex=True).fit(X).linear_transform_
    np.testing.assert_allclose(Wr, np.column_stack([wc.real, wc.imag]))
    Y = centred(X)[:-1] @ Wr
    assert abs(Y[:, 0] @ Y[:, 1]) < 1e-8 * np.linalg.norm(Y) ** 2
    assert np.linalg.norm(Y[:, 0]) >= np.linalg.norm(Y[:, 1])
    U = np.linalg.lstsq(centred(X), centred(S[:, :2]), rcond=None)[0]
    assert np.all(subspace_cos(Wr, U) > 0.99)


def test_dmd_real_components_independent_of_whitening(oscillator_sources):
    """The phase and sign conventions are defined on the components, so with
    centred data the whitened and unwhitened fits give the same real
    components up to a positive scale."""
    _, X = oscillator_sources
    D = X.shape[1]
    Yw = DMD(n_components=D, whiten=True).fit_transform(centred(X))
    Yu = DMD(n_components=D, whiten=False).fit_transform(centred(X))
    scale = np.sum(Yw * Yu, 0) / np.sum(Yu * Yu, 0)
    assert np.all(scale > 0)
    np.testing.assert_allclose(Yw, Yu * scale, atol=1e-8 * np.abs(Yw).max())


def test_dmd_real_output_shape_and_dtype(oscillator_sources):
    _, X = oscillator_sources
    for n in range(1, X.shape[1] + 1):
        Y = DMD(n_components=n).fit_transform(X)
        assert Y.shape == (len(X), n)
        assert np.isrealobj(Y)
