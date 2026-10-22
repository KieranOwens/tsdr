# Kieran Owens 2025
# Shared PCA whitening step used by the linear tsdr methods

import numpy as np
from sklearn.decomposition import PCA

def fit_whitening(X):
    """Fit a PCA whitening model to X.

    Whitening divides by the square root of each PCA variance, so it is
    undefined when the covariance matrix of X is singular (constant or 
    linearly dependent channels, or fewer time points than features). Rather 
    than silently returning non-finite or arbitrary components, raise an 
    error in that case. The rank is computed for the standardized channels
    (with the tolerance of np.linalg.matrix_rank), so that channels on very
    different scales are not mistaken for linearly dependent ones.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        A multivariate time series of shape (n_samples, n_features).

    Returns
    -------
    pca_white : sklearn.decomposition.PCA
        The fitted PCA whitening model.
    """

    # fit a PCA whitening model (this also validates X, e.g., NaN values)
    pca_white = PCA(whiten=True).fit(X)

    # data dimensions
    X = np.asarray(X, dtype=float)
    T, D = X.shape

    # constant channels
    std = X.std(axis=0)
    constant = std <= max(T, D) * np.finfo(float).eps * np.abs(X).max(axis=0)

    # numerical rank of the standardized data
    if np.any(constant):
        rank = D - int(np.sum(constant))
    else:
        rank = np.linalg.matrix_rank((X - X.mean(axis=0)) / std)

    if rank < D:
        raise ValueError(
            f"The covariance matrix of X is singular (rank {rank} < {D} "
            "features), so X cannot be whitened. This happens when channels "
            "are constant or linearly dependent (e.g., duplicated), or when "
            "there are fewer time points than features. Remove such "
            "channels, or reduce the dimension (e.g., with PCA), before "
            "fitting.")

    return pca_white
