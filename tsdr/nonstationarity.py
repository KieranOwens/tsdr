# Kieran Owens 2025
# tsdr based on non-stationarity

# Contains:
# ASSA - Analytic Stationary Subspace Analysis
# SSAsir - SSA using mean
# SSAsave - SSA using variance
# SSAcor - SSA using autocorrelation
# WaSSAf - Wasserstein SSA using the Frechet mean
# WaSSAr - Wasserstein SSA using the Root-Frechet mean
# BSSnonstat - Blind Source Separation via nonstationarity of variances

import numpy as np
import scipy
from sklearn.decomposition import PCA

###############################################################################
###############################################################################
# ASSA - Analytic Stationary Subspace Analysis
###############################################################################
###############################################################################

class ASSA():
    """
    Analytic Stationary Subspace Analysis (ASSA).

    ASSA is a TSDR method for extracting the most nonstationary (or stationary) 
    components from a multivariate time series (i.e., for which the second 
    order Taylor approximation of Gaussian KL divergence is maximised or 
    minimised between time series segments).

    The steps involved in ASSA are: (1) whiten the data, (2) compute the second 
    order Taylor approximation of Gaussian KL divergence across time series 
    segments to form matrix S, (3) apply eigendecomposition to S and select the 
    eigenvectors corresponding to the largest (or least) eigenvalues, and (4) 
    apply the eigenvector linear transformation to the whitened data.

    Reference: Hara et al (2012). Separation of stationary and non-stationary 
    sources, Neural Networks

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.

    n_windows: int
        The number of contiguous windows used for the approximation of Gaussian
        KL divergence.
        Default: 10.

    nonstationary: bool
        Whether to return nonstationary (True) or stationary (False) 
        components.
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the ASSA components when applied 
        to the input time series X.

    eigvals_: ndarray, shape (n_features)
        The singular/eigen-values of the eigendecomposition step.
    """

    def __init__(self, n_components=1, n_windows=10, nonstationary=True):

        self.n_components = n_components
        self.n_windows = n_windows
        self.nonstationary = nonstationary

    def _compute_S(self, X):
        """Compute matrix S from Hara et al (2012) equation (10)."""

        # data dimensions
        T, D = np.shape(X)

        # construct matrix S
        S = np.zeros((D, D))
        for i in range(self.n_windows):
            X_window = X[i*(T//self.n_windows):(i+1)*(T//self.n_windows),:]
            mu_i = np.mean(X_window, axis=0)
            C_i = X_window.T @ X_window /(X_window.shape[0] - 1)
            S += (1/self.n_windows) * (np.outer(mu_i, mu_i) + 0.5*(C_i @ C_i))
        
        # subtract 1/2 * identity 
        # note: I is the covariance of X due to whitening
        S -= 0.5 * np.eye(D)

        # ignore np.outer(mu,mu), where mu=np.mean(X, axis=0) 
        # note: mu is the zero vector due to whitening
        return S

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

        # fit a PCA whitening model
        pca_white = PCA(whiten=True).fit(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # get matrix S
        S = self._compute_S(Xw)

        # eigendecomposition of S
        eigvals, eigvecs = np.linalg.eig(S)
        eigvecs_sorted = eigvecs[:,np.argsort(eigvals)]

        # eigenvalues
        if self.nonstationary == True:
            self.eigvals_ = 1 - eigvals[np.argsort(eigvals)]/np.max(eigvals)
        elif self.nonstationary == False:
            self.eigvals_ = eigvals[np.argsort(eigvals)][::-1]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        if self.nonstationary:
            transform2 = eigvecs_sorted[:,-1:-(self.n_components+1):-1]
        else:
            transform2 = eigvecs_sorted[:,:self.n_components]
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
            either non-stationarity or stationarity.
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
            either non-stationarity or stationarity.
        """

        self.fit(X)
        
        return self.transform(X)
    
###############################################################################
###############################################################################
# SSAsir - Stationary Subspace Analysis via sliced inverse regression
###############################################################################
###############################################################################

class SSAsir():
    """
    Stationary Subspace Analysis via sliced inverse regression (SSAsir).

    SSAsir is a TSDR method for extracting the most nonstationary (or 
    stationary) components from a multivariate time series (i.e., for which 
    variation in the mean across time series segments is maximised or 
    minimised).

    The steps involved in SSAsir are: (1) whiten the data, (2) compute Mm which 
    averages the covariance of the mean across time-series segments, (3) apply
    eigendecomposition to Mm and select the eigenvectors corresponding to the 
    largest (or least) eigenvalues, and (4) apply the eigenvector linear 
    transformation to the whitened data.

    Reference: Flumian et al. (2024) SSA based on second-order statistics,
    J Comp. & Appl. Math

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.

    n_windows: int
        The number of contiguous windows used for computing means.
        Default: 10.

    nonstationary: bool
        Whether to return nonstationary (True) or stationary (False) 
        components.
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the SSAsir components when 
        applied to the input time series X.
    """

    def __init__(self, n_components=1, n_windows=10, nonstationary=True):

        self.n_components = n_components
        self.n_windows = n_windows
        self.nonstationary = nonstationary

    def _compute_Mm(self, X):
        """Compute matrix Mm from Flumian et al (2024) equation (3)."""

        # data dimensions
        T, D = np.shape(X)

        # construct matrix Mm
        Mm = np.zeros((D, D))
        for i in range(self.n_windows):
            Xi = X[i*(T//self.n_windows):(i+1)*(T//self.n_windows),:]
            mu_i = np.mean(Xi, axis=0)
            Mm += np.outer(mu_i, mu_i) * (Xi.shape[0]/T)

        return Mm

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

        # fit a PCA whitening model
        pca_white = PCA(whiten=True).fit(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # get matrix S
        Mm = self._compute_Mm(Xw)

        # eigendecomposition of S
        eigvals, eigvecs = np.linalg.eig(Mm)
        eigvecs_sorted = eigvecs[:,np.argsort(eigvals)]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        if self.nonstationary:
            transform2 = eigvecs_sorted[:,-1:-(self.n_components+1):-1]
        else:
            transform2 = eigvecs_sorted[:,:self.n_components]
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
            either non-stationarity or stationarity.
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
            either non-stationarity or stationarity.
        """

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# SSAsave - Stationary Subspace Analysis via sliced average variance estimation
###############################################################################
###############################################################################

class SSAsave():
    """
    Stationary Subspace Analysis via sliced average variance estimation 
    (SSAsave).

    SSAsave is a TSDR method for extracting the most nonstationary (or 
    stationary) components from a multivariate time series (i.e., for which 
    variation in the variance across time series segments is maximised or 
    minimised).

    The steps involved in SSAsave are: (1) whiten the data, (2) compute Mv 
    which averages the squared deviation of covariance from the identity matrix 
    across time-series segments, (3) apply eigendecomposition to Mv and select 
    the eigenvectors corresponding to the largest (or least) eigenvalues, and 
    (4) apply the eigenvector linear transformation to the whitened data.

    Reference: Flumian et al. (2024) SSA based on second-order statistics,
    J Comp. & Appl. Math

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.

    n_windows: int
        The number of contiguous windows used for computing covariance.
        Default: 10.

    nonstationary: bool
        Whether to return nonstationary (True) or stationary (False) 
        components.
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the SSAsave components when 
        applied to the input time series X.
    """

    def __init__(self, n_components=1, n_windows=10, nonstationary=True):

        self.n_components = n_components
        self.n_windows = n_windows
        self.nonstationary = nonstationary

    def _compute_Mv(self, X):
        """Compute matrix Mv from Flumian et al (2024) equation (4)."""

        # data dimensions
        T, D = np.shape(X)

        # construct matrix Mv
        Mv = np.zeros((D, D))
        for i in range(self.n_windows):
            Xi = X[i*(T//self.n_windows):(i+1)*(T//self.n_windows),:]
            Ti = Xi.shape[0]
            C0i = (1/(Ti - 1)) * Xi.T @ Xi
            G = (np.eye(D) - C0i)
            Mv += (G @ G.T) * (Ti/T)

        return Mv

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

        # fit a PCA whitening model
        pca_white = PCA(whiten=True).fit(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # get matrix S
        Mv = self._compute_Mv(Xw)

        # eigendecomposition of S
        eigvals, eigvecs = np.linalg.eig(Mv)
        eigvecs_sorted = eigvecs[:,np.argsort(eigvals)]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        if self.nonstationary:
            transform2 = eigvecs_sorted[:,-1:-(self.n_components+1):-1]
        else:
            transform2 = eigvecs_sorted[:,:self.n_components]
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
            either non-stationarity or stationarity.
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
            either non-stationarity or stationarity.
        """

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# SSAcor - Stationary Subspace Analysis using time-lagged autocorrelation
###############################################################################
###############################################################################

class SSAcor():
    """
    Stationary Subspace Analysis using time-lagged autocorrelation (SSAcor) 

    SSAcoris a TSDR method for extracting the most nonstationary (or 
    stationary) components from a multivariate time series (i.e., for which 
    variation in the time-lagged autocorrelation across time series segments is 
    maximised or minimised).

    The steps involved in SSAcor are: (1) whiten the data, (2) compute Mtau 
    which averages the squared deviation of time-lagged autocorrelation for 
    time-series segments versus that of the entire time series, (3) apply 
    eigendecomposition to Mtau and select the eigenvectors corresponding to the 
    largest (or least) eigenvalues, and (4) apply the eigenvector linear 
    transformation to the whitened data.

    Reference: Flumian et al. (2024) SSA based on second-order statistics,
    J Comp. & Appl. Math

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.

    lag: int
        The time lag at which to compute time-lagged autocorrelation.
        Default: 1.

    n_windows: int
        The number of contiguous windows used for computing time-lagged 
        autocorrelation.
        Default: 10.

    nonstationary: bool
        Whether to return nonstationary (True) or stationary (False) 
        components.
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the SSAcor components when 
        applied to the input time series X.
    """

    def __init__(self, n_components=1, lag=1, n_windows=10, 
                 nonstationary=True):

        self.n_components = n_components
        self.lag = lag
        self.n_windows = n_windows
        self.nonstationary = nonstationary

    def _compute_Mtau(self, X):
        """Compute matrix Mtau from Flumian et al (2024) equation (5)."""

        # data dimensions
        T, D = np.shape(X)

        # time-lagged autocorrelation
        Ctau = (X[:-self.lag,:].T @ X[self.lag:,:])/(T - 1 - self.lag)

        # construct matrix Mtau
        Mtau = np.zeros((D, D))
        for i in range(self.n_windows):
            Xi = X[i*(T//self.n_windows):(i+1)*(T//self.n_windows),:]
            Ti = Xi.shape[0]
            Ctau_i = (1/(Ti - 1 - self.lag)) * Xi[:-self.lag,:].T @ Xi[self.lag:,:]
            G = Ctau - Ctau_i
            Mtau += (G @ G.T) * (Ti/T)

        return Mtau

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

        # fit a PCA whitening model
        pca_white = PCA(whiten=True).fit(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # get matrix S
        Mtau = self._compute_Mtau(Xw)

        # eigendecomposition of S
        eigvals, eigvecs = np.linalg.eig(Mtau)
        eigvecs_sorted = eigvecs[:,np.argsort(eigvals)]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        if self.nonstationary:
            transform2 = eigvecs_sorted[:,-1:-(self.n_components+1):-1]
        else:
            transform2 = eigvecs_sorted[:,:self.n_components]
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
            either non-stationarity or stationarity.
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
            either non-stationarity or stationarity.
        """

        self.fit(X)
        
        return self.transform(X)
    
###############################################################################
###############################################################################
# WaSSAf - Wasserstein SSA using the Frechet mean
###############################################################################
###############################################################################

class WaSSAf():
    """
    Wasserstein SSA using the Frechet mean (WaSSAf).

    WaSSAf is a TSDR method for extracting the most nonstationary (or 
    stationary) components from a multivariate time series (i.e., for which the 
    approximate Wasserstein distance is maximised or minimised between 
    time-series segments).

    The steps involved in WaSSAf are: (1) whiten the data, (2) compute matrix S 
    which reflects the average deviation in Wasserstein distance across 
    time-series segments, (3) apply eigendecomposition to S and select the 
    eigenvectors corresponding to the largest (or least) eigenvalues, and (4) 
    apply the eigenvector linear transformation to the whitened data.

    Reference: Kaltenstadler et al (2018) Wasserstein SSA, 
    IEEE Selected Topics in Signal Proc.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    n_windows: int
        The number of contiguous windows used for the approximation of 
        Wassersteindistance.
        Default: 10.

    epsilon: float
        The convergence parameter for iterative approximation of the Frechet 
        mean.
        Default: 1e-4.

    max_iters: int
        The maximum number of iterations for approximation of the Frechet 
        mean.
        Default: 1000.

    nonstationary: bool
        Whether to return nonstationary (True) or stationary (False) 
        components.
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the WaSSAf components when 
        applied to the input time series X.
    """

    def __init__(self, n_components=1, n_windows=10, 
                 epsilon=1e-4, max_iters=1000, nonstationary=True):

        self.n_components = n_components
        self.n_windows = n_windows
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.nonstationary = nonstationary

    def _compute_S_r_root(self, X):
        "Compute the matrix from Kaltenstadler et al (2018) equation (15)"
    
        # data dimensions
        T, D = np.shape(X)

        # Root-Frechet mean
        S_r_root = np.zeros((D, D))
        C0_list = []
        for i in range(self.n_windows):
            Xi = X[i*(T//self.n_windows):(i+1)*(T//self.n_windows),:]
            Ti = Xi.shape[0]
            C0i = (1/(Ti - 1)) * (Xi.T @ Xi)
            C0_list.append(C0i)
            S_r_root += (1/self.n_windows) * scipy.linalg.sqrtm(C0i)

        return S_r_root, C0_list

    def _compute_S(self, X):
        """Compute matrix S from Kaltenstadler et al (2018) equation (21)."""

        # data dimensions
        T, D = np.shape(X)

        # Root-Frechet mean
        S_r_root, C0_list = self._compute_S_r_root(X)

        # Frechet mean
        Sw = S_r_root
        delta = self.epsilon
        iter = 0
        while delta >= self.epsilon and iter < self.max_iters:
            iter += 1
            Ti = np.zeros((D, D))
            Sinv = np.linalg.pinv(Sw)
            for C0i in C0_list:
                Ti += Sinv @ scipy.linalg.sqrtm(Sw @ C0i @ Sw) @ Sinv
            Ti = Ti/self.n_windows
            Snew = Ti @ Sw @ Ti
            delta = np.linalg.norm(Sw - Snew)
            Sw = Snew

        # construct matrix S
        S = np.zeros((D, D))
        for i in range(self.n_windows):
            Xi = X[i*(T//self.n_windows):(i+1)*(T//self.n_windows),:]
            Ti = Xi.shape[0]

            mu_i = np.mean(Xi, axis=0)
            C0i = (1/(Ti - 1)) * (Xi.T @ Xi)
            C0i_root = scipy.linalg.sqrtm(C0i)

            inner_term = scipy.linalg.sqrtm(C0i_root @ Sw @ C0i_root)
            S += np.outer(mu_i, mu_i) + C0i + Sw - 2 * inner_term

        return S

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

        # fit a PCA whitening model
        pca_white = PCA(whiten=True).fit(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # get matrix S
        S = self._compute_S(Xw)

        # eigendecomposition of S
        eigvals, eigvecs = np.linalg.eig(S)
        eigvecs_sorted = eigvecs[:,np.argsort(eigvals)]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        if self.nonstationary:
            transform2 = eigvecs_sorted[:,-1:-(self.n_components+1):-1]
        else:
            transform2 = eigvecs_sorted[:,:self.n_components]
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
            either non-stationarity or stationarity.
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
            either non-stationarity or stationarity.
        """

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# WaSSAr - Wasserstein SSA using the Root-Frechet mean
###############################################################################
###############################################################################

class WaSSAr():
    """
    Wasserstein SSA using the Root-Frechet mean (WaSSAr).

    WaSSAr is a TSDR method for extracting the most nonstationary (or 
    stationary) components from a multivariate time series (i.e., for which 
    the approximate Wasserstein distance is maximised or minimised between time 
    series segments).

    The steps involved in WaSSAr are: (1) whiten the data, (2) compute matrix S 
    which reflects the average deviation in Wasserstein distance across 
    time-series segments, (3) apply eigendecomposition to S and select the 
    eigenvectors corresponding to the largest (or least) eigenvalues, and (4) 
    apply the eigenvector linear transformation to the whitened data.

    Reference: Kaltenstadler et al (2018) Wasserstein SSA, 
    IEEE Selected Topics in Signal Proc.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    n_windows: int
        The number of contiguous windows used for the approximation of 
        Wassersteindistance.
        Default: 10.

    nonstationary: bool
        Whether to return nonstationary (True) or stationary (False) 
        components.
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the WaSSAr components when 
        applied to the input time series X.
    """

    def __init__(self, n_components=1, n_windows=10, nonstationary=True):

        self.n_components = n_components
        self.n_windows = n_windows
        self.nonstationary = nonstationary

    def _compute_S(self, X):
        """Compute matrix S from Kaltenstadler et al (2018) equation (23)."""

        # data dimensions
        T, D = np.shape(X)

        # Root-Frechet mean
        S_r_root = np.zeros((D, D))
        for i in range(self.n_windows):
            Xi = X[i*(T//self.n_windows):(i+1)*(T//self.n_windows),:]
            Ti = Xi.shape[0]
            C0i = (1/(Ti - 1)) * (Xi.T @ Xi)
            S_r_root += (1/self.n_windows) * scipy.linalg.sqrtm(C0i)

        # Construct matrix S
        S = np.zeros((D, D))
        for i in range(self.n_windows):
            Xi = X[i*(T//self.n_windows):(i+1)*(T//self.n_windows),:]
            Ti = Xi.shape[0]

            mu_i = np.mean(Xi, axis=0)
            C0i = (1/(Ti - 1)) * (Xi.T @ Xi)

            inner_term = scipy.linalg.sqrtm(C0i) - S_r_root
            S += np.outer(mu_i, mu_i) + inner_term @ inner_term.T

        return S

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

        # fit a PCA whitening model
        pca_white = PCA(whiten=True).fit(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # get matrix S
        S = self._compute_S(Xw)

        # eigendecomposition of S
        eigvals, eigvecs = np.linalg.eig(S)
        eigvecs_sorted = eigvecs[:,np.argsort(eigvals)]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        if self.nonstationary:
            transform2 = eigvecs_sorted[:,-1:-(self.n_components+1):-1]
        else:
            transform2 = eigvecs_sorted[:,:self.n_components]
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
            either non-stationarity or stationarity.
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
            either non-stationarity or stationarity.
        """

        self.fit(X)
        
        return self.transform(X)
    
###############################################################################
###############################################################################
# BSSnonstat - Blind Source Separation via nonstationarity of variances
###############################################################################
###############################################################################

class BSSnonstat():
    """
    Blind Source Separation via nonstationarity of variances (BSSnonstat).

    BSSnonstat is a TSDR method for  extracting signal components based on the
    nonstationarity of their variances.

    The steps involved in BSSnonstat are: (1) whiten the data, 
    (2) initialise a unit norm projection vector w, (3) iteratively optimise
    w using a fixed point algorithm, (4) deflate the data and repeat the
    optimisation to find further projection vectors, and (5) apply the
    transformation vectors to the whitened data.

    Reference: Hyvarinen (2001) Blind source separation by nonstationarity
    of variance, IEEE Trans. on NN

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int
        The time lag used for optimisation.
        Default: 1.

    max_iters: int
        The number of iterations used during optimisation.
        Default: 100.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the BSSnonstat components when 
        applied to the input time series X.
    """

    def __init__(self, n_components=1, lag=1, max_iters=100):

        self.n_components = n_components
        self.lag = lag
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

        # fit a PCA whitening model
        pca_white = PCA(whiten=True).fit(X)

        # whiten the data
        Xw = pca_white.transform(X)
        X0 = Xw[:-self.lag,:]
        Xlag = Xw[self.lag:,:]

        # data dimensions
        T, D = X0.shape

        # matrix used during optimisation
        M = X0.T @ Xlag + Xlag.T @ X0

        # collect projection vectors
        W = []

        # compute each projection vector
        for _ in range(self.n_components):

            # initialise the linear transformation vector
            w = np.random.normal(size=(D, 1))
            w = w / np.linalg.norm(w)

            # Hyvarinen (2001) equation (7)
            for _ in range(self.max_iters):
                e1 = np.zeros(shape=w.shape)
                e2 = np.zeros(shape=w.shape)

                for i in range(T - self.lag):
                    ztau = Xlag[i,:].reshape(-1,1)
                    zt = X0[i,:].reshape(-1,1)

                    e1 += (zt @ w.T @ zt @ (w.T @ ztau)**2)/(T - self.lag)
                    e2 += (ztau @ w.T @ ztau @ (w.T @ zt)**2) / (T - self.lag)

                w = e1 + e2 - 2*w - M @ w @ (w.T @ M @ w)
                w = w / np.linalg.norm(w)

            # collect projection vectors
            W.append(w)

            # deflate Xw to obtain subsequent vectors
            Xw = Xw - Xw @ (w @ w.T)
            X0 = Xw[:-self.lag,:]
            Xlag = Xw[self.lag:,:]
            M = X0.T @ Xlag + Xlag.T @ X0

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = np.concatenate(W, axis=1)
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
            nonstationarity of the variance.
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
            nonstationarity of the variance.
        """

        self.fit(X)
        
        return self.transform(X)