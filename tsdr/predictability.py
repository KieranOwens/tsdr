# Kieran Owens 2025
# tsdr based on predictability

# Contains:
# TLPC - Temporally Local Predictive Coding
# PrCA - Predictable Component Analysis
# DiCCA - Dynamic inner Canonical Correlation Analysis

import numpy as np
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

###############################################################################
###############################################################################
# TLPC - Temporally Local Predictive Coding
###############################################################################
###############################################################################

class TLPC():
    """
    Temporally Local Predictive Coding (TLPC).

    TLPC is a TSDR method for extracting the most predictable components 
    from a multivariate time series (i.e., for which a Gaussian approximation 
    of predictive information is maximised). For details on predictive 
    information see Bialek & Tishby (1999).

    The steps involved in TLPC are: (1) whiten the data, (2) compute the
    covariance (C0) and time-lagged covariance (Clag) matrices,
    (3) form Sigma = Clag C0^-1, (4) apply eigendecomposition to 
    I - Sigma^2 and select the eigenvectors corresponding to the least
    eigenvalues, and (5) apply the eigenvector linear transformation
    to the whitened data.

    Reference: Creutzik & Sprekeler (2008) Predictive coding and the slowness
    principle, Neural Comp.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int
        The time lag at which predictive information is maximised.
        Default: 1.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the TLPC components when applied 
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

        # compute the zero-lag covariance matrix
        C0 = Xw[:-1,:].T @ Xw[:-1,:]/(T - 1)

        # compute the time-lagged covariance matrix
        Ctau = Xw[:-self.lag,:].T @ Xw[self.lag:,:]/(T - 1 - self.lag)

        # compute Sigma from Creutzig et al (2008)
        Sigma = Ctau @ np.linalg.pinv(C0)

        # eigendecomposition of I - Sigma^2
        eigvals, eigvecs = np.linalg.eig(np.eye(Sigma.shape[0]) - Sigma @ Sigma)
        eigvecs_sorted = eigvecs[:,np.argsort(eigvals)]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = eigvecs_sorted[:,:self.n_components]
        self.linear_transform_ = np.real(transform1 @ transform2)

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
            predictability.
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
            predictability.
        """

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# PrCA - Predictable Component Analysis
###############################################################################
###############################################################################

class PrCA():
    """
    Predictable Component Analysis (PrCA).

    PrCA is a TSDR method for extracting the most predictable components from a 
    multivariate time series (i.e., for which the residuals of a given 
    predictive model are minimised).

    The steps involved in PrCA are: (1) train a prediction model and compute
    time series of residuals for each variable, (2) apply eigendecomposition
    to the covariance of the residuals and select the eigenvectors 
    corresponding to the least eigenvalues, and (3) apply the eigenvector 
    linear transformation to the input data.

    Reference: Schneider & Griffies (1999) A conceptual framework for
    predictability studies, J of Climate

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int
        The time lag at which predictability is maximised.
        Default: 1.

    p: int
        The number of time steps (Xt, Xt-1, ...) used to predict the future
        time point Xt+lag.
        Default: 1.

    whiten: bool
        Whether to whiten the input data.
        Defaul: True.

    model: str
        Specify the predictive model with options 'diff', 'linear_reg', and
        'kernel_ridge'. 'diff' uses naive differencing and is equivalent to
        Slow Feature Analysis (SFA) for lag=1. 'linear_reg' uses multivariate
        linear regression. 'kernel_ridge' uses kernel ridge regression.
        Default: diff.

    **kwargs: dictionary
        Optional keyword arguments for use with linear regression or 
        kernel ridge regression, e.g., to specify the rbf kernel pass
        `kernel='rbf'`.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the PrCA components when applied 
        to the input time series X.
    """

    def __init__(self, n_components=1, lag=1, p=1, whiten=True, model='diff', 
                 **kwargs):

        self.n_components = n_components
        self.lag = lag
        self.p = p
        self.whiten = whiten
        self.model = model
        self.kwargs = kwargs

    def _tde(self, X, m, tau):
        """ Apply time delay embedding to X."""

        # data dimensions
        T, D = X.shape

        # TDE output matrix
        time_delay_embedding = np.zeros((T - (m - 1) * tau, m * D))

        # compute TDE
        for t in range(T - (m -1) * tau):
            indices = np.arange(t, t + m*tau, tau)
            time_delay_embedding[t,:] = X[indices,:].flatten()

        return time_delay_embedding
    
    def _get_prediction_errors(self, X):
        """ Compute prediction errors"""

        # data dimensions
        T, D = X.shape

        # simple differencing (equivalent to SFA)
        if self.model == 'diff':

            E = X[1:,:] - X[:-1,:]

        else:

            # (empty) time series of prediction errors (i.e., residuals)
            E = np.zeros((T - (self.p + self.lag), D))

            # for each feature/variable
            for i in range(D):

                # data to use for predictions
                x = X[:,i]
                x_p = self._tde(x[:-(self.lag+1)].reshape(-1,1), self.p, 1)

                # data to predict
                y = x[self.p + self.lag:]

                # fit a model using x_p and y
                if self.model == 'linear_reg':
                    
                    E[:,i] = y - LinearRegression(**self.kwargs).fit(x_p, y).predict(x_p)

                elif self.model == 'kernel_ridge':

                    E[:,i] = y - KernelRidge(**self.kwargs).fit(x_p, y).predict(x_p)

        return E

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
        if self.whiten:
            pca_white = PCA(whiten=True).fit(X)
            X = pca_white.transform(X)

        # whiten the data
        E = self._get_prediction_errors(X)

        # fit a PCA model to the derivatives
        pca_pred_error = PCA().fit(E)

        # get linear transformation
        if self.whiten:
            transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
            transform2 = pca_pred_error.components_.T
            self.linear_transform_ = (transform1 @ transform2)[:,-1:-(self.n_components+1):-1]
        else:
            self.linear_transform_ = (pca_pred_error.components_.T)[:,-1:-(self.n_components+1):-1]

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
            predictability.
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
            predictability.
        """

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# DiCCA - Dynamic inner Canonical Correlation Analysis
###############################################################################
###############################################################################

class DiCCA():
    """
    Dynamic inner Canonical Correlation Analysis (DiCCA).

    DiCCA is a TSDR method for extracting the most predictable components 
    from a multivariate time series (in terms of an s-step AR model).

    The steps involved in DiCCA are: (1) whiten the data, (2) perform SVD and 
    then iteratively estimate optimal AR coefficients beta and linear 
    transformation w (via eigendecomposition), then (3) perform deflation 
    and repeat the process to obtain further components.

    Reference: Dong et al. (2020) Efficient dynamic latent variable analysis
    for high-dimensional time series data, IEEE Trans. on Indust. Inform.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    s: int
        The time lag over which AR predictability is maximised.
        Default: 5.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the DiCCA components when applied 
        to the input time series X.
    """

    def __init__(self, n_components=1, s=5):

        self.n_components = n_components
        self.s = s

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

        # whitening model and transformation
        pca_white = PCA(whiten=True).fit(X)
        Xw = pca_white.transform(X)
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)

        # collection projection vectors
        W = []

        # iteratively compute each linear transformation
        for nb in range(self.n_components):

            # SVD
            U, S, V = np.linalg.svd(Xw, full_matrices=False)

            # time-lagged matrices used for calculations
            Us = U[self.s:,:]
            U_list = [U[i:-(self.s-i),:] for i in range(self.s)][::-1]

            # initialise AR weights beta
            beta = np.eye(self.s)[:,0]

            # prediction (Ubeta) and prediction error (Utilde)
            Ubeta = sum([U_list[i]*beta[i] for i in range(self.s)])
            Utilde = Us - Ubeta

            # iteratively optimize transformation w and AR weights beta
            for _ in range(100):
                w = PCA().fit(Utilde).components_.T[:,-1]
                t = U @ w
                Ts = [beta[0] * U_list[0] @ w]
                for i in range(1, self.s):
                    Ts.append(Ts[-1] + beta[i] * U_list[i] @ w)
                Ts = np.concatenate([t.reshape(-1,1) for t in Ts], axis=1)
                beta = np.linalg.pinv(Ts.T @ Ts) @ Ts.T @ Us @ w

            # obtain transformation via composition of PCA whitening, SVA, and w
            W.append((transform1 @ V.T @ w).reshape(-1,1))

            # deflation step
            if nb + 1 < self.n_components:
                p = Xw.T @ t / (t.T @ t)
                Xw = Xw - np.outer(t, p)

        self.linear_transform_ = np.concatenate(W, axis=1)

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
            predictability.
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
            predictability.
        """

        self.fit(X)
        
        return self.transform(X)