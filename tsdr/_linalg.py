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


def largest_entry_signs(M):
    """The sign of the entry of largest magnitude in each column of M.

    The sign of each component of most methods (e.g., of an eigenvector) is 
    arbitrary. Multiplying the columns of a linear transformation (or of the
    components themselves) by these signs fixes it, so that the entry of 
    largest magnitude in each column is positive (+1 is returned for a 
    column of zeros).

    Parameters
    ----------
    M : ndarray, shape (n, n_columns)
        A real matrix, e.g., a linear transformation with one column per 
        component.

    Returns
    -------
    signs : ndarray, shape (n_columns,)
        The signs (+1 or -1) of the entries of largest magnitude.
    """

    rows = np.argmax(np.abs(M), axis=0)
    signs = np.sign(M[rows, np.arange(M.shape[1])])
    signs[signs == 0] = 1

    return signs
