# Kieran Owens 2025
# tsdr based on determinism

# Contains:
# DyCA - Dynamical Component Analysis
# DMD - Dynamic Mode Decomposition

import numpy as np
from scipy.linalg import eig
from sklearn.decomposition import PCA

###############################################################################
###############################################################################
# DyCA - Dynamical Component Analysis
###############################################################################
###############################################################################

class DyCA():
    """
    Dynamical Component Analysis (DyCA).

    DyCA is a TSDR method for extracting the most deterministic components from 
    a multivariate time seriese (i.e., that can be represented as trajectories 
    of a linear ordinary differential equation).

    The steps involved in DyCA are: (1) compute the temporal derivative of the 
    data, (2) compute covariance of the data (C0), the covariance of the data 
    with the derivative (C1) and the covariance of the derivative (C2), 
    (3) solve the generalised eigenvalue problem C1 C0^-1 C1 W = Lambda C2 W, 
    (4) obtain a linear transformation that is applied to the input data to 
    obtain deterministic components.

    Reference: Seifert et al (2018) DyCA: dimensionality reduction for 
    high-dimensional deterministic time series, 28th Int. Workshop on MLSP

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        This argument is ignored if eig_threshold is used.
        Default: 1.

    eig_threshold: float in [0.0, 1.0]
        Select components based on eigenvalues that exceed eig_threshold,
        which indicate the degree of (linear ODE) determinism.
        Default: None.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the DyCA components when applied 
        to the input time series X.

    eigvals_: ndarray, shape (n_features)
        The singular/eigen-values of the eigendecomposition step. The array
        length (n_features) may be shortened in case of a singular covariance
        matrix.
    """

    def __init__(self, n_components=1, eig_threshold=None):

        self.n_components = n_components
        self.eig_threshold = eig_threshold

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
        X = pca_white.transform(X)

        # derivative
        Xdot = np.gradient(X, axis=0, edge_order=1)

        # autocorrelation matrices
        C0 = (X.T @ X) / T
        C1 = (Xdot.T @ X) / T
        C2 = (Xdot.T @ Xdot) / T

        # generalised eigenvalue problem
        LHS = C1 @ np.linalg.pinv(C0) @ C1.T
        RHS = C2
        eigvals, eigvecs = eig(LHS, RHS)
        abseigvals = np.real(np.abs(eigvals))
        eigvals, eigvecs = eigvals[np.argsort(-abseigvals)], eigvecs[:,np.argsort(-abseigvals)]
        self.eigvals_ = eigvals[np.array(eigvals <= 1)]

        # exclude eigvals > 1 due to singularity of C0

        if self.eig_threshold == None:
            eigvecs = eigvecs[:,np.array(eigvals <= 1)]
        else:
            eigvecs = eigvecs[:,np.array(eigvals > self.eig_threshold) &  np.array(eigvals <= 1)]

        # return transformed data or else generate an error
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        if eigvecs.shape[1] > 0:
            C3 = np.linalg.pinv(C1) @ C2
            W = np.concatenate((eigvecs, 
                                np.apply_along_axis(lambda x: np.matmul(C3,x), 0, eigvecs)),axis=1)
            self.linear_transform_ = (transform1 @ W)[:,:self.n_components]
        else:
            raise ValueError(f'No generalised eigenvalue fulfills threshold > {self.eig_threshold}.')

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
            (linear ODE) determinism.
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
            (linear ODE) determinism.
        """

        self.fit(X)
        
        return self.transform(X)
    
###############################################################################
###############################################################################
# DMD - Dynamic Mode Decomposition
###############################################################################
###############################################################################

class DMD():
    """
    Dynamic Mode Decomposition (DMD).

    DMD is a TSDR method for  extracting the most linearly deterministic 
    components from a multivariate time series at a given time lag. The
    associated linear transformation is obtained through eigendecomposition
    of a linear operator that (approximately) advances the system forward
    in time, and which is conceptualised as an approximation of the
    Koopman operator of the process.

    The steps involved in TICA are: (1) whiten the data (optional), 
    (2) separated the data into non-lagged (X) and lagged (Y) components,
    (3) compute linear regression matrix from X to Y via least squares,
    i.e., A = (X.T X)^-1 X.T Y (4) apply eigendecomposition to A and 
    select the eigenvectors corresponding to the largest eigenvalues, and 
    (5) apply the eigenvector linear transformation to the (whitened) data.

    Reference: Schmid & Sesterhenn (2008) Dynamic mode decomposition of
    numerical and experimental data, Bulletin of the APS

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    whiten: bool
        Whether to whiten the data.
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the DMD components when applied 
        to the input time series X.
    """

    def __init__(self, n_components=1, whiten=True):

        self.n_components = n_components
        self.whiten = whiten

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

        if self.whiten:

            # fit a PCA whitening model
            pca_white = PCA(whiten=True).fit(X)

            # whiten the data
            X = pca_white.transform(X)

        # data and time-lagged data
        X, Y = X[:-1,:], X[1:]

        # linear regression from X to Y
        # the linear regression coefficient is A = (X^T X)^-1 X^T Y
        # if data is whitened presume X^T X = I
        # cf. eigenvalue formulation of TICA, TMCA
        if self.whiten:
            A = X.T @ Y
        else:
            A = np.linalg.pinv(X.T @ X) @ X.T @ Y

        # svd of the linear regression transformation
        U, _, _ = np.linalg.svd(A)

        # get linear transformation
        if self.whiten:

            transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
            transform2 = U[:, :self.n_components]
            self.linear_transform_ = (transform1 @ transform2)

        else:

            self.linear_transform_ = U[:, :self.n_components]

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
            linear determinism.
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
            linear determinism.
        """

        self.fit(X)
        
        return self.transform(X)