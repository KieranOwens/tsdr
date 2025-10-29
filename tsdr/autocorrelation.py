# Kieran Owens 2025
# tsdr based on autocorrelation

# Contains:
# TICA - Time-lagged Independent Component Analysis
# TMCA - Time-lagged Maximum Covariance Analysis
# TCCA_Stiefel - TCCA via trace optimisation on a Stiefel manifold
# CSA - Coloured Subspace Analysis
# sPCA_dwt - Spectral Principal Component Analysis using the
#            discrete wavelet transform

import numpy as np
from sklearn.decomposition import PCA
import pywt

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
        pca_white = PCA(whiten=True).fit(X)

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
        pca_white = PCA(whiten=True).fit(X)

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
        The maximum number of iterations during trace optimisation.
        Default: 100.

    lr: float
        The learning rate used for gradient descent during trace optimisation.
        Default: 0.1.

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
    def __init__(self, n_components=1, lag=1, max_iters=100, lr=0.1, seed=0, whiten=True):
        self.n_components = n_components
        self.lag = lag
        self.max_iters = max_iters
        self.lr = lr
        self.seed = seed
        self.whiten = whiten

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
            pca_white = PCA(whiten=True).fit(X)
            X = pca_white.transform(X)

        # Covariance matrices
        X0 = X[:-lag, :] 
        Xtau = X[lag:, :]
        C0 = (X0.T @ X0) / (T - 1 - lag)
        Ctau = (X0.T @ Xtau) / (T - 1 - lag)

        # Random seed for initialisation of W
        np.random.seed(self.seed)

        # Initialise W on the Stiefel manifold
        W = np.random.randn(D, self.n_components)
        # QR decomposition to ensure W is orthonormal
        W, _ = np.linalg.qr(W)

        # Trace optimisation loop
        for _ in range(self.max_iters):
            G = self._grad(W, C0, Ctau)
            Rgrad = self._project_to_tangent(W, G)
            W = W - self.lr * Rgrad
            W = self._retraction(W)

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

    lag: int
        The maximum time lag at which to compute autocorrelation.
        Default: 1.

    method: str
        The method by which CSA is computed. Options include 'jd'
        (joint diagonalisations), 'scca' (subspace candidate component
        analysis), and 'jla' (joint low-rank approximation).
        Default: 'jd'.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the CSA components when applied 
        to the input time series X.
    """

    def __init__(self, n_components=1, lags=2, max_iters=1000):

        self.n_components = n_components
        if isinstance(lags, int):
            self.lags = range(1, lags+1)
        elif isinstance(lags, list):
            self.lags = lags
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
        pca_white = PCA(whiten=True).fit(X)

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
    Spectral Principal Component Analysis (sPCA) using the discrete wavelet
    transform.

    sPCA_dwt is a TSDR method for  extracting period (hence, autocorrelated) 
    components from a multivariate time series

    The steps involved in sPCA-dwt are: (1) optionally whiten the data, 
    (2) compute the discrete wavelet transform (DWT) of the data,
    (3) perform PCA or SVD on DWT(X), (4) retain the components corresponding
    to the largest eigenvalues/singular values, (5) project back into the
    time domain by applying the inverse DWT to the components.

    Reference: Guilloteau et al (2021) Rotated spectral PCA for identifying
    dynamical modes of variability in climate systems, J of Climate.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    wavelet: str
        Specify a PyWavelet discrete wavelet. Options include 'bior1.1', 
        'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 
        'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 
        'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 
        'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 
        'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 
        'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 
        'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 
        'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 
        'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 
        'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 
        'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 
        'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 
        'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 
        'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 
        'sym17', 'sym18', 'sym19', 'sym20'
        Default: 'db1'.

    whiten: bool
        Whether to whiten the data.
        Default: False.

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

        # optional data whitening
        if self.whiten:

            X = PCA(whiten=True).fit_transform(X)

        # discrete wavelet transform
        dwt = pywt.dwt(X, self.wavelet, axis=0)[0]

        # principal component analysis
        if self.method == 'pca':

            # PCA
            pc = PCA(n_components=self.n_components).fit_transform(dwt)

            # inverse discrete wavelet transform
            self.components = pywt.idwt(pc, 
                                        cD=None, wavelet=self.wavelet, axis=0)

        # singular value decomposition
        elif self.method == 'svd':

            # SVD
            U, _, _ = np.linalg.svd(dwt)

            # inverse discrete wavelet transform
            self.components = pywt.idwt(U[:,0:self.n_components], 
                                        cD=None, wavelet=self.wavelet, axis=0)

        return self

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
        
        return self.components