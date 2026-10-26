# Kieran Owens 2025
# tsdr based on autocorrelation

# Contains:
# TICA - Time-lagged Independent Component Analysis
# TMCA - Time-lagged Maximum Covariance Analysis
# TCCA - Time-lagged Canonical Correlation Analysis
# TCCA_Stiefel - Trace-ratio optimisation on a Stiefel manifold (which, for
#                whitened data, gives the TICA subspace)
# CSA - Coloured Subspace Analysis
# sPCA_dwt - Spectral Principal Component Analysis using the
#            discrete wavelet transform

import numbers
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
import pywt

from ._linalg import sqrtm_psd
from ._validation import check_input, check_lag
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
        to the centred time series X - mean_.

    mean_: ndarray, shape (n_features,)
        The mean of the fitted time series, which transform subtracts
        from X before applying linear_transform_.
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

        X = check_input(X, self.n_components)

        # data dimensions
        T, _ = X.shape

        # fit a PCA whitening model
        pca_white = fit_whitening(X)

        # the mean of X, which transform subtracts (the components are
        # fitted to the centred, whitened data)
        self.mean_ = pca_white.mean_

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

        X = check_input(X)

        # centre X with the mean of the fitted data, as in fit
        return (X - self.mean_) @ self.linear_transform_

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

        X = check_input(X, self.n_components)

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
        to the centred time series X - mean_.

    mean_: ndarray, shape (n_features,)
        The mean of the fitted time series, which transform subtracts
        from X before applying linear_transform_.
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

        X = check_input(X, self.n_components)

        # data dimensions
        T, _ = X.shape

        # fit a PCA whitening model
        pca_white = fit_whitening(X)

        # the mean of X, which transform subtracts (the components are
        # fitted to the centred, whitened data)
        self.mean_ = pca_white.mean_

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

        X = check_input(X)

        # centre X with the mean of the fitted data, as in fit
        return (X - self.mean_) @ self.linear_transform_

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

        X = check_input(X, self.n_components)

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# TCCA - Time-lagged Canonical Correlation Analysis
###############################################################################
###############################################################################

class TCCA():
    """
    Time-lagged Canonical Correlation Analysis (TCCA).

    TCCA is a TSDR method for extracting the components of a multivariate 
    time series that are most correlated with the future of the time series 
    at some time lag. It applies canonical correlation analysis (CCA) to the 
    pairs (x_t, x_{t+lag}): the first pair of canonical weights (u_1, v_1) 
    maximises corr(u^T x_t, v^T x_{t+lag}), the second pair maximises it 
    among components uncorrelated with the first, and so on. The canonical 
    correlations are the singular values of the whitened time-lagged 
    covariance (the Koopman matrix of Wu et al., 2017).

    The steps involved in TCCA are: (1) whiten the data, (2) form the 
    instantaneous data X0 = x_t and the time-lagged data X1 = x_{t+lag}, 
    centre each, and compute their covariances C00 and C11 and their 
    cross-covariance C01, (3) compute the SVD of the whitened cross-covariance
    K = C00^-1/2 C01 C11^-1/2 = U diag(s) V^T, and (4) return the 
    projections of the data onto the left canonical weights C00^-1/2 U, 
    ordered by canonical correlation s.

    The components are the "instantaneous" (left) canonical variates u^T x_t.
    The right canonical weights C11^-1/2 V, whose variates v^T x_{t+lag} are
    the parts of the future that each component predicts best, are stored 
    in right_transform_. For a time-reversible process (C01 symmetric and 
    C00 = C11), the left and right weights coincide and the components are 
    those of TICA, except that TCCA orders them by the absolute value of 
    their lagged autocorrelation (so a strongly negatively autocorrelated 
    component ranks high), whereas TICA orders them by its signed value.

    Unlike TCCA_Stiefel, which optimises one shared matrix of weights for a 
    trace ratio, TCCA uses separate weights for x_t and x_{t+lag}, and is 
    computed in closed form.

    References: 
    Hotelling (1936) Relations between two sets of variates, Biometrika.
    Wu et al (2017) Variational Koopman models: slow collective variables 
    and molecular kinetics from short off-equilibrium simulations, J Chem 
    Phys.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int
        The time lag between x_t and x_{t+lag}.
        Default: 1.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The left canonical weights, the linear transformation that yields the
        TCCA components when applied to the centred time series X - mean_.

    right_transform_: ndarray, shape (n_features, n_components)
        The right canonical weights: (X - mean_) @ right_transform_ gives the
        canonical variates of the time-lagged data matched to the components.

    mean_: ndarray, shape (n_features,)
        The mean of the fitted time series, which transform subtracts
        from X before applying linear_transform_.

    canonical_correlations_: ndarray, shape (n_features,)
        The canonical correlations corr(u_k^T x_t, v_k^T x_{t+lag}) of all
        pairs of canonical weights, in decreasing order (the first 
        n_components are those of the components).
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

        X = check_input(X, self.n_components)

        # data dimensions
        T, D = X.shape

        # at least D + 1 pairs (x_t, x_{t+lag}), for nonsingular covariances
        check_lag(self.lag, T - D)

        # fit a PCA whitening model
        pca_white = fit_whitening(X)

        # the mean of X, which transform subtracts (the components are
        # fitted to the centred, whitened data)
        self.mean_ = pca_white.mean_

        # whiten the data
        Xw = pca_white.transform(X)

        # instantaneous and time-lagged data, each centred
        X0 = Xw[:-self.lag,:]
        X1 = Xw[self.lag:,:]
        X0 = X0 - np.mean(X0, axis=0)
        X1 = X1 - np.mean(X1, axis=0)

        # covariances and cross-covariance
        n = T - self.lag
        C00 = X0.T @ X0 / (n - 1)
        C11 = X1.T @ X1 / (n - 1)
        C01 = X0.T @ X1 / (n - 1)

        # whitening matrices of the instantaneous and time-lagged data
        C00_inv_root = np.linalg.inv(sqrtm_psd(C00))
        C11_inv_root = np.linalg.inv(sqrtm_psd(C11))

        # SVD of the whitened cross-covariance (Koopman matrix); the singular
        # values are the canonical correlations, in decreasing order
        K = C00_inv_root @ C01 @ C11_inv_root
        U, s, Vt = np.linalg.svd(K)
        self.canonical_correlations_ = s

        # get linear transformations (left and right canonical weights)
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        self.linear_transform_ = transform1 @ C00_inv_root @ U[:,:self.n_components]
        self.right_transform_ = transform1 @ C11_inv_root @ Vt.T[:,:self.n_components]

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
            canonical correlation with the time-lagged data.
        """

        X = check_input(X)

        # centre X with the mean of the fitted data, as in fit
        return (X - self.mean_) @ self.linear_transform_

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
            canonical correlation with the time-lagged data.
        """

        X = check_input(X, self.n_components)

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# TCCA_Stiefel - Trace-ratio optimisation of the time-lagged covariance on a
#                Stiefel manifold
###############################################################################
###############################################################################

class TCCA_Stiefel:
    """
    Time-lagged trace-ratio optimisation on a Stiefel manifold 
    (TCCA_Stiefel).

    TCCA_Stiefel is a TSDR method for extracting the most autocorrelated 
    components from a multivariate time series at some time lag. It finds 
    the matrix W with orthonormal columns (a point on the Stiefel manifold) 
    that maximises the trace ratio tr(W^T Ctau W) / tr(W^T C0 W) by 
    Riemannian gradient ascent, in the framework of Cunningham & Ghahramani 
    (2015), and then rotates W within its column space to order the 
    components by autocorrelation.

    The steps involved in TCCA_Stiefel are: (1) whiten the data (optional), 
    (2) compute the covariance C0 of x_t and the time-lagged covariance Ctau
    of x_t and x_{t+lag}, (3) maximise the trace ratio over W on the Stiefel
    manifold, and (4) apply W to the (whitened) data.

    What it computes: the same weights W are used for x_t and x_{t+lag}, so 
    this is not canonical correlation analysis between x_t and x_{t+lag} 
    (for which see TCCA, which uses separate weights and is computed in 
    closed form). Since only the symmetric part of Ctau enters the trace, 
    and C0 is (almost exactly) the identity for whitened data, the optimum 
    is the span of the leading eigenvectors of the symmetrised lagged 
    covariance, i.e., the TICA subspace: with whiten=True (the default) the 
    components coincide with those of TICA (up to sign, convergence and the
    small difference between C0 and the identity). With whiten=False the 
    trace ratio is maximised for the raw (centred) data, which is not 
    invariant to the scaling of the channels.

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

    random_state: int, RandomState instance or None
        Controls the random initialisation of the linear transformation 
        weights. Pass an int for reproducible results across calls (the 
        global NumPy random state is then left unchanged). If None, the 
        global NumPy random state is used.
        Default: 0.

    seed: int or None
        Deprecated alias of random_state (if not None, it is used in place 
        of random_state, with a DeprecationWarning).
        Default: None.

    whiten: bool
        Whether to whiten the data (if False, the data are mean-centred).
        Default: True.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the TCCA components when applied 
        to the centred time series X - mean_.

    mean_: ndarray, shape (n_features,)
        The mean of the fitted time series, which transform subtracts
        from X before applying linear_transform_.
    """
    def __init__(self, n_components=1, lag=1, max_iters=1000, lr=0.1, random_state=0, 
                 whiten=True, tol=1e-6, seed=None):
        self.n_components = n_components
        self.lag = lag
        self.max_iters = max_iters
        self.lr = lr
        self.random_state = random_state
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

        X = check_input(X, self.n_components)

        # data shape
        T, D = X.shape

        # Time-lag used for TCCA
        lag = self.lag

        # the mean of X, which transform subtracts
        self.mean_ = np.mean(X, axis=0)

        # Whiten the data if required (whitening also centres the data);
        # otherwise centre it, since C0 and Ctau are covariances
        if self.whiten:
            pca_white = fit_whitening(X)
            X = pca_white.transform(X)
        else:
            X = X - self.mean_

        # Covariance matrices
        X0 = X[:-lag, :] 
        Xtau = X[lag:, :]
        C0 = (X0.T @ X0) / (T - 1 - lag)
        Ctau = (X0.T @ Xtau) / (T - 1 - lag)

        # Random number generator for initialisation of W (an int 
        # random_state does not reset the global NumPy random state); seed is 
        # the deprecated name of random_state
        random_state = self.random_state
        if self.seed is not None:
            warnings.warn("The seed argument of TCCA_Stiefel is deprecated; use "
                          "random_state instead.", DeprecationWarning, stacklevel=2)
            random_state = self.seed
        rng = check_random_state(random_state)

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

        X = check_input(X)

        # centre X with the mean of the fitted data, as in fit
        return (X - self.mean_) @ self.linear_transform_

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

        X = check_input(X, self.n_components)

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

    Reference: Theis (2010) Colored subspace analysis: dimension reduction
    based on a signal's autocorrelation structure, IEEE Transactions on 
    Circuits and Systems.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lag: int or sequence of int
        The maximum time lag at which to compute autocorrelation (i.e., lags 
        1, ..., lag), or else a sequence (e.g., list, tuple or array) of 
        specified lags. All lags must be positive integers.
        Default: 2.

    max_iters: int
        The number of iterations of the joint low-rank approximation (the 
        algorithm always runs for max_iters iterations).
        Default: 1000.

    tol: float
        A ConvergenceWarning is raised if the change in the subspace spanned
        by the components in the last iteration, ||W W^T - W_old W_old^T|| 
        for the orthonormal matrix W, exceeds tol. It does not affect the 
        result.
        Default: 1e-6.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the CSA components when applied 
        to the centred time series X - mean_.

    mean_: ndarray, shape (n_features,)
        The mean of the fitted time series, which transform subtracts
        from X before applying linear_transform_.
    """

    def __init__(self, n_components=1, lag=2, max_iters=1000, tol=1e-6):

        self.n_components = n_components
        self.lag = lag
        self.max_iters = max_iters
        self.tol = tol

    def _get_lags(self):
        """The list of lags: 1, ..., lag for an integer lag (including NumPy 
        integers), or the given lags for a sequence."""

        if isinstance(self.lag, numbers.Integral):
            lags = list(range(1, int(self.lag) + 1))
        else:
            try:
                lags = [int(l) for l in self.lag]
            except TypeError:
                raise TypeError(f"lag must be an integer or a sequence of "
                                f"integers (got {self.lag!r}).") from None
            if not all(isinstance(l, numbers.Integral) for l in self.lag):
                raise TypeError(f"lag must be an integer or a sequence of "
                                f"integers (got {self.lag!r}).")

        if len(lags) == 0 or min(lags) < 1:
            raise ValueError(f"All lags must be positive integers (got "
                             f"{self.lag!r}).")

        return lags

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

        X = check_input(X, self.n_components)

        # data dimensions
        T, _ = X.shape

        # fit a PCA whitening model
        pca_white = fit_whitening(X)

        # the mean of X, which transform subtracts (the components are
        # fitted to the centred, whitened data)
        self.mean_ = pca_white.mean_

        # whiten the data
        Xw = pca_white.transform(X)

        # list of covariance matrices at different time lags
        Ctensor = []

        # iterate over specified lags
        for lag in self._get_lags():

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
        change = np.inf
        for _ in range(self.max_iters):
            W_previous = W
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
            # change in the subspace spanned by W (whatever the signs of its
            # columns) in this iteration
            change = np.linalg.norm(W @ W.T - W_previous @ W_previous.T)

        if change > self.tol:
            warnings.warn(f"CSA may not have converged in max_iters="
                          f"{self.max_iters} iterations (change in the subspace "
                          f"in the last iteration {change:.2e} > tol={self.tol}); "
                          "consider increasing max_iters.", ConvergenceWarning)

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

        X = check_input(X)

        # centre X with the mean of the fitted data, as in fit
        return (X - self.mean_) @ self.linear_transform_

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

        X = check_input(X, self.n_components)

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
        Whether to use PCA or SVD. Options: 'pca' or 'svd'. Note: the two 
        options currently give the same result, since the component scores 
        are always computed by SVD of the centred coefficients (which equals
        PCA up to the signs of the components).
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

        X = check_input(X, self.n_components)

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

        X = check_input(X, self.n_components)

        self.fit(X)
        
        return self.components