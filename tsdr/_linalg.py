# Small linear-algebra helpers shared by several tsdr methods

import numpy as np


def sqrtm_psd(C):
    """The square root of a symmetric positive semi-definite matrix C.

    The square root is computed from the eigendecomposition 
    C = V diag(lam) V^T as V diag(sqrt(lam)) V^T, after symmetrising C and 
    setting to zero any eigenvalues that rounding has made slightly negative.
    It is therefore always real and symmetric, unlike scipy.linalg.sqrtm, 
    which can return complex values (or a slightly nonsymmetric matrix) for a 
    singular or nearly singular C. For a positive definite C the two agree up 
    to rounding.

    Parameters
    ----------
    C : ndarray, shape (n, n)
        A symmetric positive semi-definite matrix (e.g., a covariance matrix).

    Returns
    -------
    C_root : ndarray, shape (n, n)
        The symmetric positive semi-definite matrix with C_root @ C_root = C.
    """

    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)
    lam = np.clip(lam, 0.0, None)

    return (V * np.sqrt(lam)) @ V.T
