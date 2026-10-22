# Kieran Owens 2025
# tsdr based on autocorrelation

# Contains:
# TICA - Time-lagged Independent Component Analysis
# TMCA - Time-lagged Maximum Covariance Analysis
# TCCA_Stiefel - TCCA via trace optimisation on a Stiefel manifold
# CSA - Coloured Subspace Analysis
# sPCA_dwt - Spectral Principal Component Analysis using the
#            discrete wavelet transform

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
import pywt

from ._whitening import fit_whitening

###############################################################################
###############################################################################
# TICA - Time-lagged Independent Component Analysis
###############################################################################
###############################################################################

class TICA():
    """
    Time-lagged Independent Component Analysis (TICA).

    TICA is a TSDR method for  extracting the most autocorrelated components 
    from a multivariate time series at a given time lag.

    The steps involved in TICA are: (1) whiten the data, (2) compute the 
    time-lagged autocorrelation Ctau, (3) enforce symmetry by assigning 
    Ctau = 0.5 * (Ctau + Ctau.T), (4) apply eigendecomposition to Ctau and 
    select the eigenvectors corresponding to the largest eigenvalues, and 
    (5) apply the eigenvector linear transformation to the whitened data.

    Reference: Molgedey & Schuster (1994) Separation of a mixture of 
    independent signals using time delayed correlations, PR Letters.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int
        The time lag at which to compute autocorrelation.
        Default: 1.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the TICA components when applied 
        to the input time series X.
    """

    def __init__(self, n_components=1, lag=1):

        self.n_components = n_components
        self.lag = lag

    def fit(self, X, y=None):
        """Fit the model to X

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        self : object
            Returns self.
        """

        # data dimensions
        T, _ = X.shape

        # fit a PCA whitening model
        pca_white = fit_whitening(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # compute the time-lagged covariance matrix
        Ctau = (Xw[:-self.lag,:].T @ Xw[self.lag:,:])/(T - self.lag - 1)

        # enforce symmetry (i.e., time reversibility)
        Ctau = 0.5 * (Ctau + Ctau.T)

        # eigendecomposition of Ctau
        _, eigvecs = np.linalg.eigh(Ctau)

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = eigvecs[:,-1:-(self.n_components+1):-1]
        self.linear_transform_ = (transform1 @ transform2)

        return self

    def transform(self, X):
        """Apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            autocorrelation.
        """

        return X @ self.linear_transform_

    def fit_transform(self, X, y=None):
        """Fit the model to X and apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            autocorrelation.
        """

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# TMCA - Time-lagged Maximum Covariance Analysis
###############################################################################
###############################################################################

class TMCA():
    """
    Time-lagged Maximum Covariance Analysis (TMCA).

    TMCA is a TSDR method for extracting the most autocorrelated components 
    from a multivariate time series at some time lag.

    The steps involved in TMCA are: (1) whiten the data, (2) compute the 
    time-lagged autocorrelation Ctau, (3) apply singular value decomposition to  
    Ctau and select the left singular vectors corresponding to the largest 
    singular values, and (5) apply the eigenvector linear transformation to the 
    whitened data.

    Reference: Bretherton et al (1992) An intercomparison of methods for 
    finding coupled patterns in climate data, J of Climate.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int
        The time lag at which to compute autocorrelation.
        Default: 1.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the TMCA components when applied 
        to the input time series X.
    """

    def __init__(self, n_components=1, lag=1):

        self.n_components = n_components
        self.lag = lag

    def fit(self, X, y=None):
        """Fit the model to X

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        self : object
            Returns self.
        """

        # data dimensions
        T, _ = X.shape

        # fit a PCA whitening model
        pca_white = fit_whitening(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # compute the time-lagged covariance matrix
        Ctau = (Xw[:-self.lag,:].T @ Xw[self.lag:,:])/(T - self.lag - 1)

        # SVD eigendecomposition of the time-lagged covariance matrix
        U, _, _ = np.linalg.svd(Ctau)

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = U[:,:self.n_components]
        self.linear_transform_ = (transform1 @ transform2)

        return self

    def transform(self, X):
        """Apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            autocorrelation.
        """

        return X @ self.linear_transform_

    def fit_transform(self, X, y=None):
        """Fit the model to X and apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            autocorrelation.
        """

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# TCCA_Stiefel - Time-lagged Canonical Component Analysis via 
#                trace optimisation on a Stiefel manifold
###############################################################################
###############################################################################

class TCCA_Stiefel:
    """
    Time-lagged Canonical Component Analysis via trace optimisation on a 
    Stiefel manifold (TCCA_Stiefel).

    TCCA_Stiefel is a TSDR method for extracting the most autocorrelated 
    components from a multivariate time series at some time lag.

    The steps involved in TCCA are: (1) whiten the data, and (2) apply CCA
    at some time lag. The TCCA_Stiefel function performs CCA via trace optimisation 
    on a Stiefel manifold.

    Reference: Cunningham & Ghahramani (2015) Linear dimensionality reduction,
    JMLR.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int
        The time lag at which to compute autocorrelation.
        Default: 1.

    max_iters: int
        The maximum number of iterations during trace optimisation. A
        ConvergenceWarning is raised if the optimisation has not converged
        (see tol) after max_iters iterations.
        Default: 1000.

    lr: float
        The learning rate used for gradient descent during trace optimisation.
        Default: 0.1.

    tol: float
        The optimisation stops when the norm of the Riemannian gradient falls
        below tol.
        Default: 1e-6.

    seed: int
        The random seed used for initialisation of the linear transformation
        weights.
        Default: 0.

    whiten: bool
        Whether to whiten the data.
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the TCCA components when applied 
        to the input time series X.
    """
    def __init__(self, n_components=1, lag=1, max_iters=1000, lr=0.1, seed=0, whiten=True,
                 tol=1e-6):
        self.n_components = n_components
        self.lag = lag
        self.max_iters = max_iters
        self.lr = lr
        self.seed = seed
        self.whiten = whiten
        self.tol = tol

    # The analytic gradient of the loss function
    # L(W) = -tr(W^T C_tau W) / tr(W^T C_0 W)
    def _grad(self, W, C0, Ctau):
        num = np.trace(W.T @ Ctau @ W)
        den = np.trace(W.T @ C0 @ W)
        grad_num = (Ctau + Ctau.T) @ W # in case of a degenerate non-symmetric Ctau
        grad_den = (C0 + C0.T)@W # in case of a degenerate non-symmetric C0
        dL = -(grad_num * den - grad_den * num) / (den ** 2)
        return dL

    # Utility function to symmetrise a matrix
    def _symmetric(self, A):
        return 0.5 * (A + A.T)

    # Project the gradient onto the tangent space of the Stiefel manifold
    # This ensures that the update step does not leave the manifold
    # G is the gradient, W is the current point on the manifold
    # The projection is done by subtracting the component of G that is in the normal direction
    # to the manifold at W, which is given by W @ (W.T @ G)
    # The term W.T @ G is symmetrised to ensure that the projection is valid
    def _project_to_tangent(self, W, G):
        return G - W @ self._symmetric(W.T @ G)

    # Retraction operation to ensure W remains on the Stiefel manifold
    # The Q factor from QR decomposition is orthonormal, i.e., Q.T @ Q = I
    def _retraction(self, W):
        Q, _ = np.linalg.qr(W)
        return Q

    def fit(self, X, y=None):
        """Fit the model to X

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        self : object
            Returns self.
        """

        # data shape
        T, D = X.shape

        # Time-lag used for TCCA
        lag = self.lag

        # Whiten the data if required
        if self.whiten:
            pca_white = fit_whitening(X)
            X = pca_white.transform(X)

        # Covariance matrices
        X0 = X[:-lag, :] 
        Xtau = X[lag:, :]
        C0 = (X0.T @ X0) / (T - 1 - lag)
        Ctau = (X0.T @ Xtau) / (T - 1 - lag)

        # Random number generator for initialisation of W (without 
        # resetting the global NumPy random state)
        rng = np.random.RandomState(self.seed)

        # Initialise W on the Stiefel manifold
        W = rng.randn(D, self.n_components)
        # QR decomposition to ensure W is orthonormal
        W, _ = np.linalg.qr(W)

        # Trace optimisation loop
        converged = False
        for _ in range(self.max_iters):
            G = self._grad(W, C0, Ctau)
            Rgrad = self._project_to_tangent(W, G)
            if np.linalg.norm(Rgrad) < self.tol:
                converged = True
                break
            W = W - self.lr * Rgrad
            W = self._retraction(W)

        if not converged:
            warnings.warn(f"TCCA_Stiefel did not converge in max_iters={self.max_iters} "
                          f"iterations (Riemannian gradient norm {np.linalg.norm(Rgrad):.2e} "
                          f">= tol={self.tol}); consider increasing max_iters.",
                          ConvergenceWarning)

        # The objective is invariant to rotations of W within the subspace, so
        # rotate W to order the components by autocorrelation
        eigvals, eigvecs = np.linalg.eigh(W.T @ self._symmetric(Ctau) @ W)
        W = W @ eigvecs[:,::-1]

        # Get linear transformation
        if self.whiten:
            transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
            transform2 = W
            self.linear_transform_ = transform1 @ transform2
        else:
            self.linear_transform_ = W

        return self

    def transform(self, X):
        """Apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            autocorrelation.
        """

        return X @ self.linear_transform_

    def fit_transform(self, X, y=None):
        """Fit the model to X and apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            autocorrelation.
        """

        self.fit(X)
        
        return self.transform(X)
    
###############################################################################
###############################################################################
# CSA - Coloured Subspace Analysis
###############################################################################
###############################################################################

class CSA():
    """
    Coloured Subspace Analysis (CSA).

    CSA is a TSDR method for  extracting the most autocorrelated components 
    from a multivariate time series over a range of time lags.

    The steps involved in CSA are: (1) whiten the data, (2) compute the 
    time-lagged autocorrelation Ctau over one or more time lags, (3) enforce 
    symmetry by assigning Ctau = 0.5 * (Ctau + Ctau.T), (4) apply  joint 
    low rank approximation (JLA) to {Ctau} and select the eigenvectors 
    corresponding to the largest eigenvalues, and (5) apply the eigenvector 
    linear transformation to the whitened data.

    Reference: Theis (2010) Colored subspace analysis: dimenion reduction
    based on a signal's autocorrelation structure, IEEE Transactions on 
    Circuits and Systems.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int or list
        The maximum time lag at which to compute autocorrelation,
        or else a list of specified lags.
        Default: 1.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the CSA components when applied 
        to the input time series X.
    """

    def __init__(self, n_components=1, lag=2, max_iters=1000):

        self.n_components = n_components
        if isinstance(lag, int):
            self.lags = range(1, lag+1)
        elif isinstance(lag, list):
            self.lags = list(lag)
        self.max_iters = max_iters

    def fit(self, X, y=None):
        """Fit the model to X

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        self : object
            Returns self.
        """

        # data dimensions
        T, _ = X.shape

        # fit a PCA whitening model
        pca_white = fit_whitening(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # list of covariance matrices at different time lags
        Ctensor = []

        # iterate over spacified lags
        for lag in self.lags:

            # compute the time-lagged covariance matrix
            Ctau = (Xw[:-lag,:].T @ Xw[lag:,:])/(T - lag - 1)

            # enforce symmetry (i.e., time reversibility)
            Ctau = 0.5 * (Ctau + Ctau.T)

            # append covariance matrice
            Ctensor.append(Ctau)

        # initialise W via eigendecomposition
        _, eigvecs = np.linalg.eigh(sum(Ctensor))
        W = eigvecs[:,-1:-self.n_components-1:-1]

        # iteratively approximate W
        for _ in range(self.max_iters):
            C = Ctensor[0]
            if self.n_components==1:
                outer = np.outer(W, W).T
            else:
                outer = W @ W.T
            R = C @ outer @ C.T + C.T @ outer @ C
            for C in Ctensor[1:]:
                R += C @ outer @ C.T + C.T @ outer @ C
            _, eigvecs = np.linalg.eigh(R)
            W = eigvecs[:,-1:-self.n_components-1:-1]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        self.linear_transform_ = transform1 @ W

        return self

    def transform(self, X):
        """Apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            autocorrelation.
        """

        return X @ self.linear_transform_

    def fit_transform(self, X, y=None):
        """Fit the model to X and apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            autocorrelation.
        """

        self.fit(X)
        
        return self.transform(X)
    
    
###############################################################################
###############################################################################
# sPCA_dwt - Spectral Principal Component Analysis using the
#            discrete wavelet transform
###############################################################################
###############################################################################

class sPCA_dwt():
    """
    Spectral Principal Component Analysis using a single-level discrete 
    wavelet transform (sPCA_dwt).

    sPCA_dwt is a TSDR method that applies PCA (or SVD) to the low-pass 
    (approximation) coefficients of a real discrete wavelet transform (DWT), 
    emphasising slow, autocorrelated structure in the data.

    The steps involved in sPCA_dwt are: (1) optionally whiten the data, 
    (2) compute one level of the DWT along time and keep only the 
    approximation coefficients (about n_samples / 2 rows), (3) perform PCA or 
    SVD on these coefficients, (4) retain the leading n_components, and 
    (5) map the components back to the time domain with the inverse DWT.

    Note: this is a simplified variant inspired by Guilloteau et al. (2021). 
    It is NOT their Morlet-wavelet rotated spectral PCA (rsPCA), which uses a 
    continuous wavelet transform and a different construction.

    The method does not learn a linear map of the features, so there is no 
    ``transform`` method. Use ``fit_transform``.

    Reference: Guilloteau et al (2021) Rotated spectral PCA for identifying
    dynamical modes of variability in climate systems, J of Climate.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    wavelet: str
        Any PyWavelets discrete wavelet name; see
        ``pywt.wavelist(kind='discrete')``. Examples: 'db1' (Haar), 'db4', 
        'sym5', 'coif3'.
        Default: 'db1'.

    whiten: bool
        Whether to whiten the data.
        Default: True

    method: str
        Whether to use PCA or SVD. Options: 'pca' or 'svd'.
        Default: 'pca'.
    """

    def __init__(self, n_components=1, wavelet='db1', whiten=True, method='pca'):

        self.n_components = n_components
        self.wavelet = wavelet
        self.whiten = whiten
        self.method = method

    def fit(self, X, y=None):
        """Fit the model to X

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        self : object
            Returns self.
        """

        # data length
        T = X.shape[0]

        # optional data whitening
        if self.whiten:

            X = fit_whitening(X).transform(X)

        # discrete wavelet transform
        dwt = pywt.dwt(X, self.wavelet, axis=0)[0]

        # centre the approximation coefficients
        dwt = dwt - dwt.mean(axis=0)

        if self.method not in ('pca', 'svd'):
            raise ValueError(f"Unknown method '{self.method}'; use 'pca' or 'svd'.")

        # principal component scores via SVD (identical to sklearn PCA scores
        # up to sign, and valid for both methods)
        U, s, _ = np.linalg.svd(dwt, full_matrices=False)
        scores = U[:, :self.n_components] * s[:self.n_components]

        # inverse discrete wavelet transform
        self.components = pywt.idwt(scores, cD=None, wavelet=self.wavelet, axis=0)

        # the inverse DWT has an extra sample for odd T
        self.components = self.components[:T]

        return self

    def transform(self, X):
        """Not supported: sPCA_dwt is transductive. Use ``fit_transform``."""
        raise NotImplementedError(
            "sPCA_dwt returns component time series of the fitted data and "
            "has no out-of-sample transform; use fit_transform(X)."
        )

    def fit_transform(self, X, y=None):
        """Fit the model to X and apply dimension reduction to X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            A multivariate time series of shape (n_samples, n_features).

        y : None
            Ignored. Used to comply with the scikit-learn API.

        Returns
        -------
        X_new: ndarray, shape (n_samples, n_components)
            A (possibly multivariate) time series of components ordered by 
            variance they explain in the low-pass DWT coefficients.
        """

        self.fit(X)
        
        return self.components