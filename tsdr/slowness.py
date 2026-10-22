# Kieran Owens 2025
# tsdr based on slowness

# Contains:
# SFA - Slow Feature Analysis
# BioSFA - Biologically plausible SFA

import numpy as np
import scipy
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state

from ._whitening import fit_whitening

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
        pca_white = fit_whitening(X)

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

    The steps involved in BioSFA are: (1) whiten the data, (2) compute the 
    covariance C0 of the data and the covariance C0_bar of the delayed sums 
    x_t + x_{t-1}, (3) optimise the synaptic weights W and M by gradient 
    descent-ascent (Lipshutz et al., 2020, eqs. 13-15), (4) obtain the 
    projection V = W M^-1 corresponding to the neural output y = M^-1 W x, 
    and (5) apply the linear transformation to the whitened data.

    The BioSFA objective is invariant to rotations of the output, so the 
    algorithm learns the slow subspace but not the individual slow features. 
    The components are therefore ordered within the learned subspace by 
    slowness (via the generalised eigenproblem of the covariance of the 
    temporal derivative restricted to the subspace; Rayleigh-Ritz), which 
    yields the SFA components once the subspace has converged.

    Reference: Lipshutz et al (2020) A biologically plausible neural network
    for slow feature analysis, NeurIPS

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    n_iters: int
        The number of iterations of gradient descent-ascent.
        Default: 1000.

    lr: float
        The learning rate (eta) of the synaptic weights W.
        Default: 0.1.

    tau: float
        The ratio of the learning rates of W and M, so that the learning rate
        of M is lr / tau. Must exceed lr, so that M remains positive definite.
        Default: 1.0.

    random_state: int, RandomState instance or None
        Controls the random initialisation of the linear transformation
        weights W. Pass an int for reproducible results across calls (the
        global NumPy random state is then left unchanged). If None, the
        global NumPy random state is used.
        Default: None.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the BioSFA components when 
        applied to the input time series X.

    eigvals_: ndarray, shape (n_components,)
        The variance of the temporal derivative of each (unit-variance) 
        component, in ascending order (slowest first).
    """

    def __init__(self, n_components=1, n_iters=1000, lr=0.1, tau=1.0,
                 random_state=None):

        self.n_components = n_components
        self.n_iters = n_iters
        self.lr = lr
        self.tau = tau
        self.random_state = random_state

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

        if not 0 < self.lr < self.tau:
            raise ValueError(f"lr must satisfy 0 < lr < tau (got lr={self.lr}, "
                             f"tau={self.tau}), so that M remains positive "
                             "definite (Lipshutz et al., 2020).")

        pca_white = fit_whitening(X)
        Xw = pca_white.transform(X)

        T, D = Xw.shape

        # 2-step sum
        X_bar = Xw[0:T-1,:] + Xw[1:T,:]

        # covariance matrices
        C0 = Xw.T @ Xw / (T - 1)
        C0_bar = X_bar.T @ X_bar / (T - 1)
            
        # Initialise W and M
        rng = check_random_state(self.random_state)
        W = rng.normal(size=(D, self.n_components))
        W = W / np.linalg.norm(W, axis=0)
        M = np.eye(self.n_components)

        # offline gradient descent-ascent, Lipshutz et al. (2020) eqs. 13-15
        # (W here is the transpose of W in the paper; Ybar = M^-1 W Xbar)
        for _ in range(self.n_iters):
            Minv = np.linalg.inv(M)
            dW = 2 * (C0_bar @ W @ Minv - C0 @ W)                 # eq. 14
            dM = (Minv @ W.T @ C0_bar @ W @ Minv - M) / self.tau   # eq. 15
            W += self.lr * dW
            M += self.lr * dM

        # neural output y = M^-1 W x, i.e. the projection V = W M^-1, which
        # satisfies the whitening constraint V^T C0 V = I at the fixed point
        V = W @ np.linalg.inv(M)

        # the objective is invariant to rotations of the output, so BioSFA
        # learns the slow subspace only; order the components within it by
        # slowness (Rayleigh-Ritz: generalised eigenproblem of the derivative
        # covariance restricted to span(V))
        Xdot = np.diff(Xw, axis=0)
        Cdot = Xdot.T @ Xdot / (T - 2)
        eigvals, R = scipy.linalg.eigh(V.T @ Cdot @ V, V.T @ C0 @ V)
        self.eigvals_ = eigvals

        # get linear transformation (ascending derivative variance, i.e., 
        # slowest first)
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = V @ R
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
