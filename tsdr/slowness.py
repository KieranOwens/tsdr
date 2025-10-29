# Kieran Owens 2025
# tsdr based on slowness

# Contains:
# SFA - Slow Feature Analysis
# BioSFA - Biologically plausible SFA

import numpy as np
import scipy
from sklearn.decomposition import PCA

###############################################################################
###############################################################################
# SFA - Slow Feature Analysis
###############################################################################
###############################################################################

class SFA():
    """
    Slow Feature Analysis (SFA).

    SFA is a TSDR method for extracting the slowest components from a
    multivariate time series (i.e., for which the square of the first 
    temporal derivative is minimised).

    The steps involved in SFA are: (1) whiten the data, (2) compute the
    temporal derivatives through differencing, (3) perform PCA in the
    space of temporal derivatives and select the eigenvectors corresponding
    to the least eigenvalues, and (4) apply the eigenvector linear 
    transformation to the whitened data.

    Reference: Wiskott & Sejnowski (2002) SFA: unsupervised learning of 
    invariances, Neural Comp.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the SFA components when applied 
        to the input time series X.

    eigvals_: ndarray, shape (n_features)
        The singular/eigen-values of the eigendecomposition step.
    """

    def __init__(self, n_components=1):

        self.n_components = n_components

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

        # get the first temporal derivative
        Xdot = Xw[1:,:] - Xw[:-1,:]

        # fit a PCA model to the derivatives
        pca_diff = PCA().fit(Xdot)

        # singular values
        self.eigvals_ = pca_diff.singular_values_

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = pca_diff.components_.T
        self.linear_transform_ = (transform1 @ transform2)[:,-1:-(self.n_components+1):-1]

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
            slowness.
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
            slowness.
        """

        self.fit(X)
        
        return self.transform(X)
    
###############################################################################
###############################################################################
# BioSFA - Biologically plausible Slow Feature Analysis
###############################################################################
###############################################################################

class BioSFA():
    """
    Biologically plausible Slow Feature Analysis (BioSFA).

    BioSFA is a TSDR method for extracting the slowest components from a
    multivariate time series (i.e., for which the square of the first 
    temporal derivative is minimised).

    BioSFA uses an optimisation approach to SFA which is conceptualised as an 
    algorithm that could plausibly be implemented in a biological neural network 
    to learn slow invariant features from perceptual inputs. Here we use the
    offline (rather than incremental) version of the BioSFA algorithm.

    Reference: Lipshutz et L (2020) A biologically plausible neural network
    for slow feature analysis, NeurIPS

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the BioSFA components when 
        applied to the input time series X.
    """

    def __init__(self, n_components=1, n_iters=1000, lr=0.1):

        self.n_components = n_components
        self.n_iters = n_iters
        self.lr = lr

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

        pca_white = PCA(whiten=True).fit(X)
        Xw = pca_white.transform(X)

        T, D = Xw.shape

        # 2-step sum
        X_bar = Xw[0:T-1,:] + Xw[1:T,:]

        # covariance matrices
        C0 = Xw.T @ Xw / (T - 1)
        C0_bar = X_bar.T @ X_bar / (T - 1)
            
        # Initialise W and M
        W = np.random.normal(size=(D, self.n_components))
        W = W / np.linalg.norm(W, axis=0)
        M = np.eye(self.n_components)

        for _ in range(self.n_iters):
            dW = C0_bar @ W @ np.linalg.pinv(M) - C0 @ W
            dM = (W.T @ C0_bar @ W) @ np.linalg.pinv(M @ M) - M
            W += self.lr * dW
            M += self.lr * dM

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = W
        self.linear_transform_ = transform1 @ transform2

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
            slowness.
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
            slowness.
        """

        self.fit(X)
        
        return self.transform(X)
