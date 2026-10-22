# Kieran Owens 2025
# tsdr based on predictability

# Contains:
# TLPC - Temporally Local Predictive Coding
# PrCA - Predictable Component Analysis
# DiCCA - Dynamic inner Canonical Correlation Analysis
# ForeCA - Forecastable Component Analysis
# PFA - Predictable Feature Analysis
# PTS - Principal Time Series

import warnings

import numpy as np
import scipy
import scipy.linalg
import scipy.ndimage
import scipy.optimize
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

from ._whitening import fit_whitening

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
    covariance (C0) and time-lagged covariance (Ctau) matrices,
    (3) form the covariance of the past conditioned on the future,
    C0 - Ctau C0^-1 Ctau^T (I - Ctau Ctau^T for whitened data), 
    (4) apply generalized eigendecomposition relative to C0 and select the
    eigenvectors corresponding to the least eigenvalues, and (5) apply the
    eigenvector linear transformation to the whitened data.

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
        pca_white = fit_whitening(X)

        # whiten the data
        Xw = pca_white.transform(X)

        # past (X0) and future (Xlag) samples
        X0 = Xw[:-self.lag,:]
        Xlag = Xw[self.lag:,:]

        # compute the zero-lag covariance matrices of the past and future
        C0 = X0.T @ X0/(T - 1 - self.lag)
        Clag0 = Xlag.T @ Xlag/(T - 1 - self.lag)

        # compute the time-lagged covariance matrix
        Ctau = X0.T @ Xlag/(T - 1 - self.lag)

        # covariance of the past conditioned on the future from 
        # Creutzig & Sprekeler (2008), C0 - Ctau Clag0^-1 Ctau^T, which is 
        # I - Ctau Ctau^T for whitened data
        Ccond = C0 - Ctau @ np.linalg.pinv(Clag0) @ Ctau.T
        Ccond = 0.5 * (Ccond + Ccond.T)

        # generalized eigendecomposition Ccond w = lambda C0 w 
        # (eigenvalues in ascending order)
        _, eigvecs = scipy.linalg.eigh(Ccond, C0)

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = eigvecs[:,:self.n_components]
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

    model: str
        Specify the predictive model with options 'diff', 'linear_reg', and
        'kernel_ridge'. 'diff' uses naive differencing and is equivalent to
        Slow Feature Analysis (SFA) for lag=1 (with whiten=True). 
        'linear_reg' uses multivariate linear regression. 'kernel_ridge' uses
        kernel ridge regression. Both models predict all variables at time
        t+lag from the previous p time steps of all variables.
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

    def __init__(self, n_components=1, lag=1, p=1, model='diff', **kwargs):

        self.n_components = n_components
        self.lag = lag
        self.p = p
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

        if self.model not in ('diff', 'linear_reg', 'kernel_ridge'):
            raise ValueError(f"Unknown model '{self.model}'. Options are "
                             "'diff', 'linear_reg', and 'kernel_ridge'.")

        # simple differencing (equivalent to SFA)
        if self.model == 'diff':

            E = X[1:,:] - X[:-1,:]

        else:

            # predict the full state X_{t+lag} from the p previous states of
            # all variables (Xt, Xt-1, ..., Xt-p+1), so that the prediction 
            # errors E are those of the multivariate time series and w^T E 
            # is the prediction error of the component w^T X
            # (prediction errors have T - (p + lag - 1) rows)

            # data to use for predictions
            X_p = self._tde(X[:-self.lag,:], self.p, 1)   # drop last `lag` points

            # data to predict
            Y = X[self.p - 1 + self.lag:,:]

            # fit a model using X_p and Y
            if self.model == 'linear_reg':

                E = Y - LinearRegression(**self.kwargs).fit(X_p, Y).predict(X_p)

            elif self.model == 'kernel_ridge':

                E = Y - KernelRidge(**self.kwargs).fit(X_p, Y).predict(X_p)

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
        pca_white = fit_whitening(X)
        X = pca_white.transform(X)

        # whiten the data
        E = self._get_prediction_errors(X)

        # fit a PCA model to the derivatives
        pca_pred_error = PCA().fit(E)

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = pca_pred_error.components_.T
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
        pca_white = fit_whitening(X)
        Xw = pca_white.transform(X)
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)

        # data dimensions
        T, D = Xw.shape

        # the deflated data are Xd = Xw @ deflation
        Xd = Xw
        deflation = np.eye(D)

        # collection projection vectors
        W = []

        # iteratively compute each linear transformation
        for nb in range(self.n_components):

            # SVD, Xd = U diag(S) V, dropping the null directions left by
            # previous deflation steps
            U, S, V = np.linalg.svd(Xd, full_matrices=False)
            keep = S > S[0] * max(T, D) * np.finfo(float).eps
            U, S, V = U[:,keep], S[keep], V[keep,:]

            # time-lagged matrices used for calculations
            Us = U[self.s:,:]
            U_list = [U[i:-(self.s-i),:] for i in range(self.s)][::-1]

            # initialise AR weights beta
            beta = np.eye(self.s)[:,0]

            # iteratively optimize transformation w and AR weights beta
            for _ in range(100):
                # prediction (Ubeta) and prediction error (Utilde)
                Ubeta = sum([U_list[i]*beta[i] for i in range(self.s)])
                Utilde = Us - Ubeta
                w = PCA().fit(Utilde).components_.T[:,-1]
                t = U @ w
                # least-squares AR coefficients of the latent score t on its
                # lagged values Ts = [t_{k-1}, ..., t_{k-s}] (Dong & Qin, 2018)
                Ts = np.concatenate([(U_list[i] @ w).reshape(-1,1) 
                                     for i in range(self.s)], axis=1)
                beta = np.linalg.pinv(Ts.T @ Ts) @ Ts.T @ Us @ w

            # weights r such that t = Xd @ r (since U = Xd V^T diag(1/S))
            r = V.T @ (w / S)

            # obtain transformation via composition of PCA whitening, 
            # deflation, and r
            W.append((transform1 @ deflation @ r).reshape(-1,1))

            # deflation step, Xd <- Xd - t p^T = Xd (I - r p^T)
            if nb + 1 < self.n_components:
                p = Xd.T @ t / (t.T @ t)
                Xd = Xd - np.outer(t, p)
                deflation = deflation @ (np.eye(D) - np.outer(r, p))

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

###############################################################################
###############################################################################
# ForeCA - Forecastable Component Analysis
###############################################################################
###############################################################################

class ForeCA():
    """
    Forecastable Component Analysis (ForeCA).

    ForeCA is a TSDR method for extracting the most forecastable components 
    from a multivariate time series (i.e., for which the spectral entropy is 
    minimised, so that the spectral density is concentrated on few 
    frequencies).

    The steps involved in ForeCA are: (1) whiten the data, (2) estimate the 
    multivariate spectral density of the whitened data by a smoothed 
    periodogram, (3) from a random unit vector w, iterate until convergence:
    (a) compute the spectral density f_y of y = w^T x and (b) update w to the 
    eigenvector corresponding to the least eigenvalue of the spectral density 
    matrix weighted by -log f_y (an EM-like algorithm that decreases the 
    spectral entropy of y at each iteration), keeping the best of several 
    random starts, (4) obtain further components by repeating (3) in the 
    orthogonal complement of the previous components, and (5) apply the 
    linear transformation to the whitened data.

    The forecastability of a component is Omega = 1 - H, where H is the 
    spectral entropy of its (discrete) spectral density with logarithm base 
    n_samples, so that Omega is 0 for white noise and 1 for a sinusoid.

    Reference: Goerg (2013) Forecastable component analysis, ICML. The 
    defaults (smoothed periodogram, prior weight, number of random starts, 
    and ordering by Omega) follow the R package ForeCA by G. M. Goerg 
    (https://github.com/gmgeorg/ForeCA).

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    bandwidth: int
        The number of neighbouring Fourier frequencies on each side used to 
        smooth the periodogram (Daniell kernel of width 2 * bandwidth + 1).
        Default: 5.

    prior_weight: float
        The weight of a uniform (white noise) prior mixed into each spectral 
        density, which avoids log(0) in the spectral entropy.
        Default: 1e-3.

    n_starts: int
        The number of random starts of the EM-like algorithm for each 
        component, of which the one with the least spectral entropy is kept.
        Default: 10.

    max_iters: int
        The maximum number of iterations of the EM-like algorithm. A 
        ConvergenceWarning is raised if the best start has not converged 
        (see tol) after max_iters iterations.
        Default: 100.

    tol: float
        The algorithm stops when the norm of the change in w falls below tol.
        Default: 1e-6.

    random_state: int, RandomState instance or None
        Controls the random starts. Pass an int for reproducible results 
        across calls (the global NumPy random state is then left unchanged). 
        If None, the global NumPy random state is used.
        Default: None.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the ForeCA components when 
        applied to the input time series X.

    omega_: ndarray, shape (n_components,)
        The forecastability Omega of each component, in descending order.
    """

    def __init__(self, n_components=1, bandwidth=5, prior_weight=1e-3, 
                 n_starts=10, max_iters=100, tol=1e-6, random_state=None):

        self.n_components = n_components
        self.bandwidth = bandwidth
        self.prior_weight = prior_weight
        self.n_starts = n_starts
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

    def _smooth(self, a):
        """Circular moving average (Daniell kernel) over the frequencies."""

        return scipy.ndimage.uniform_filter1d(a, 2 * self.bandwidth + 1, 
                                              axis=0, mode='wrap')

    def _density(self, F, w):
        """Spectral density of y = w^T x at the Fourier frequencies, which
        sums to one for unit w, mixed with a uniform prior."""

        T = F.shape[0]
        f = self._smooth(np.abs(F @ w)**2)

        return (1 - self.prior_weight) * f + self.prior_weight / T

    def _em(self, F, w):
        """The EM-like algorithm of Goerg (2013, Sec. 4.2.1) from w."""

        T, D = F.shape
        delta = np.inf
        for _ in range(self.max_iters):

            # E-step: spectral density of y = w^T x and its -log (base T)
            c = -np.log(self._density(F, w)) / np.log(T)

            # M-step: w is the eigenvector of the least eigenvalue of the 
            # -log f_y weighted spectral density matrix, Goerg (2013) eq. (24)
            # (sum_j c_j f_j = F^H diag(smooth(c)) F for the smoothed 
            # periodogram f_j, plus the uniform prior)
            S = np.real(F.conj().T @ (self._smooth(c)[:,None] * F))
            S = (1 - self.prior_weight) * S + self.prior_weight / T * np.sum(c) * np.eye(D)
            _, eigvecs = np.linalg.eigh(S)
            w_new = eigvecs[:,0] * np.sign(eigvecs[:,0] @ w)

            delta = np.linalg.norm(w_new - w)
            w = w_new
            if delta < self.tol:
                break

        # spectral entropy of y = w^T x (base T)
        f = self._density(F, w)
        H = -np.sum(f * np.log(f)) / np.log(T)

        return w, H, delta

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

        # data dimensions
        T, D = Xw.shape

        # discrete Fourier transform, scaled so that the periodogram of 
        # y = w^T x, |F w|^2, sums over the T Fourier frequencies to the 
        # variance of y (Parseval), i.e., to one for unit w
        F = np.fft.fft(Xw, axis=0) / np.sqrt(T * (T - 1))

        rng = check_random_state(self.random_state)

        # compute each projection vector in the orthogonal complement of the
        # previous ones, Goerg (2013) Sec. 4.3
        W = np.zeros((D, 0))
        H = []
        for _ in range(self.n_components):

            # orthonormal basis of the orthogonal complement of W
            N = scipy.linalg.null_space(W.T) if W.shape[1] > 0 else np.eye(D)
            FN = F @ N

            # keep the best of several random starts
            best = None
            for _ in range(self.n_starts):
                w0 = rng.normal(size=N.shape[1])
                w, H_w, delta = self._em(FN, w0 / np.linalg.norm(w0))
                if best is None or H_w < best[1]:
                    best = (w, H_w, delta)

            if best[2] >= self.tol:
                warnings.warn(f"ForeCA did not converge in max_iters="
                              f"{self.max_iters} iterations (change in w "
                              f"{best[2]:.2e} >= tol={self.tol}); consider "
                              "increasing max_iters.", ConvergenceWarning)

            W = np.column_stack([W, N @ best[0]])
            H.append(best[1])

        # order the components by forecastability
        omega = 1 - np.array(H)
        order = np.argsort(-omega)
        self.omega_ = omega[order]

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = W[:,order]
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
            forecastability.
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
            forecastability.
        """

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# PFA - Predictable Feature Analysis
###############################################################################
###############################################################################

class PFA():
    """
    Predictable Feature Analysis (PFA).

    PFA is a TSDR method for extracting the most predictable components from a 
    multivariate time series (i.e., for which the error of a linear 
    autoregressive prediction, iterated over several time steps, is 
    minimised).

    The steps involved in PFA are: (1) whiten the data, (2) fit a linear
    autoregressive model by least squares that predicts x(t) from its history
    zeta(t) = [x(t-1), ..., x(t-p)], x(t) ~ W zeta(t), and a model that 
    predicts the history one time step ahead, zeta(t+1) ~ V zeta(t), (3) 
    compute the covariance of the prediction errors of x(t) from the 
    histories i = 0, ..., k time steps earlier, x(t) ~ W V^i zeta(t-i), summed 
    over i, (4) apply eigendecomposition to this covariance and select the 
    eigenvectors corresponding to the least eigenvalues, and (5) apply the 
    eigenvector linear transformation to the whitened data.

    Iterating the prediction over k time steps reduces overfitting of the 
    components to noise (Richthofer & Wiskott, 2013, Sec. 2.3). For k = 0, 
    PFA is equivalent to PrCA with model='linear_reg'.

    Reference: Richthofer & Wiskott (2013) Predictable feature analysis, 
    arXiv:1311.2503. A reference implementation by the authors is GNUPFA 
    (https://github.com/Stewori/GNUPFA).

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    p: int
        The order of the autoregressive model, i.e., the number of time steps 
        (x(t-1), ..., x(t-p)) used to predict x(t).
        Default: 2.

    k: int
        The number of iterated prediction steps used to reduce overfitting.
        Default: 4.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the PFA components when applied 
        to the input time series X.

    eigvals_: ndarray, shape (n_features,)
        The summed prediction error variance of each (unit-variance) 
        component, in ascending order (most predictable first).
    """

    def __init__(self, n_components=1, p=2, k=4):

        self.n_components = n_components
        self.p = p
        self.k = k

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

        # data dimensions
        T, D = Xw.shape

        # histories zeta(t) = [x(t-1), ..., x(t-p)] and x(t), for t = p, ..., T-1
        Zeta = np.column_stack([Xw[self.p - j:T - j,:] for j in range(1, self.p + 1)])
        Xt = Xw[self.p:,:]

        # least squares predictions x(t) ~ W zeta(t) and zeta(t+1) ~ V zeta(t),
        # Richthofer & Wiskott (2013) eqs. (17) and (22)
        W = np.linalg.lstsq(Zeta, Xt, rcond=None)[0].T
        V = np.linalg.lstsq(Zeta[:-1,:], Zeta[1:,:], rcond=None)[0].T

        # summed covariance of the errors of the iterated predictions 
        # x(t) ~ W V^i zeta(t-i), i = 0, ..., k, Richthofer & Wiskott (2013)
        # eq. (25)
        C = np.zeros((D, D))
        for i in range(self.k + 1):
            E = Xt[i:,:] - Zeta[:len(Zeta) - i,:] @ (W @ np.linalg.matrix_power(V, i)).T
            C += E.T @ E / E.shape[0]

        # eigendecomposition (eigenvalues in ascending order)
        eigvals, eigvecs = np.linalg.eigh(C)
        self.eigvals_ = eigvals

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = eigvecs[:,:self.n_components]
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
# PTS - Principal Time Series
###############################################################################
###############################################################################

class PTS():
    """
    Principal Time Series (PTS).

    PTS is a TSDR method for extracting the most predictable components from a 
    multivariate time series (i.e., for which the mean squared error of a 
    vector autoregressive (VAR) model of the components is minimised).

    The steps involved in PTS are: (1) whiten the data, (2) construct the 
    projection W one column w at a time, orthogonal to the previous columns, 
    by alternating until convergence between (a) fitting the order-M VAR 
    model x_t ~ A_1 x_{t-1} + ... + A_M x_{t-M} of the components 
    x_t = W^T z_t by least squares and (b) updating w to minimise the 
    prediction error of the VAR model with A fixed, a quadratic function of 
    w on the unit sphere that is minimised exactly, and (3) apply the linear 
    transformation W to the whitened data.

    Dong et al. (2022) start each column from w = 0, for which the first VAR 
    model is trivial and the first update of w is arbitrary. Here each column 
    is instead initialised by the exact solution for M = 1 and a single 
    component (Dong et al., 2022, Sec. 2.3) in the orthogonal complement of 
    the previous columns, i.e., the eigenvector of the symmetrised lag-1 
    autocovariance with the eigenvalue of largest magnitude.

    Reference: Dong, Qin & Boyd (2022) Extracting a low-dimensional 
    predictable time series, Optimization and Engineering. A MATLAB 
    implementation by the authors is available at 
    https://github.com/cvxgrp/PrincipalTimeSeries.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    M: int
        The memory (order) of the VAR model, i.e., the number of time steps 
        (t-1, ..., t-M) used to predict the components at time t.
        Default: 1.

    max_iters: int
        The maximum number of iterations of steps (a)-(b) for each component.
        A ConvergenceWarning is raised if the iteration has not converged 
        (see tol) after max_iters iterations.
        Default: 100.

    tol: float
        The iteration for each component stops when the decrease in the 
        prediction error falls below tol.
        Default: 1e-8.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the PTS components when applied 
        to the input time series X.

    ar_coefs_: ndarray, shape (M, n_components, n_components)
        The coefficients A_1, ..., A_M of the VAR model of the components.

    prediction_errors_: ndarray, shape (n_components,)
        The mean squared VAR prediction error (summed over components) of the 
        first k = 1, ..., n_components components. Each component adds at 
        most 1 (for an unpredictable component), so increments close to 1 
        indicate that no further predictable components remain (Dong et al., 
        2022, Sec. 3.3).
    """

    def __init__(self, n_components=1, M=1, max_iters=100, tol=1e-8):

        self.n_components = n_components
        self.M = M
        self.max_iters = max_iters
        self.tol = tol

    def _fit_var(self, Z, W):
        """Least squares VAR coefficients and prediction error of the 
        components Z W (Dong et al., 2022, Sec. 3.1)."""

        T = Z.shape[0]
        Xc = Z @ W
        R = np.column_stack([Xc[self.M - i:T - i,:] for i in range(1, self.M + 1)])
        coef = np.linalg.lstsq(R, Xc[self.M:,:], rcond=None)[0]
        error = np.sum((Xc[self.M:,:] - R @ coef)**2) / (T - self.M)

        # A_i such that x_t ~ sum_i A_i x_{t-i}, shape (M, k, k)
        A = np.stack(np.split(coef, self.M)).transpose(0, 2, 1)

        return A, error

    def _update_w(self, Z, Wk, A):
        """Minimise the VAR prediction error over the new column w of 
        W = [Wk w] with A fixed, subject to Wk^T w = 0 and w^T w = 1 (Dong et 
        al., 2022, Sec. 3.2)."""

        T, D = Z.shape
        k = Wk.shape[1]

        # lagged data z_{t-i}, i = 0, ..., M
        Zi = [Z[self.M - i:T - i,:] for i in range(self.M + 1)]

        # blocks of A_i = [[A_11, a_12], [a_21^T, a_22]] for the existing 
        # components (k) and the new component
        A11, a12, a21, a22 = A[:,:k,:k], A[:,:k,k], A[:,k,:k], A[:,k,k]

        # prediction errors of the existing components, c_top - sum_j P_j w,
        # and of the new component, Q w + c_bot
        c_top = Zi[0] @ Wk - sum(Zi[i] @ Wk @ A11[i-1].T for i in range(1, self.M + 1))
        P = [sum(a12[i-1, j] * Zi[i] for i in range(1, self.M + 1)) for j in range(k)]
        Q = Zi[0] - sum(a22[i-1] * Zi[i] for i in range(1, self.M + 1))
        c_bot = -sum(Zi[i] @ Wk @ a21[i-1] for i in range(1, self.M + 1))

        # prediction error w^T B w - 2 c^T w + const, Dong et al. (2022) eq. (6)
        B = (sum(Pj.T @ Pj for Pj in P) + Q.T @ Q) / (T - self.M)
        c = (sum(Pj.T @ c_top[:,j] for j, Pj in enumerate(P)) - Q.T @ c_bot) / (T - self.M)

        # restrict to the orthogonal complement U of Wk and diagonalise,
        # U^T B U = V diag(lam) V^T, Dong et al. (2022) eq. (7)
        U = scipy.linalg.null_space(Wk.T) if k > 0 else np.eye(D)
        lam, V = np.linalg.eigh(U.T @ B @ U)
        beta = V.T @ U.T @ c

        # y = (diag(lam) + mu I)^-1 beta with mu > -lam_1 the root of 
        # sum_i beta_i^2 / (lam_i + mu)^2 = 1
        g = lambda mu: np.sum(beta**2 / (lam + mu)**2) - 1
        lower = -lam[0] + 1e-12 * max(1, abs(lam[0]))
        if g(lower) <= 0:
            # (near) hard case: the eigenvector of the least eigenvalue
            y = np.zeros_like(lam)
            y[0] = 1
        else:
            mu = scipy.optimize.brentq(g, lower, -lam[0] + np.linalg.norm(beta))
            y = beta / (lam + mu)

        w = U @ V @ y

        return w / np.linalg.norm(w)

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
        Z = pca_white.transform(X)

        # data dimensions
        T, D = Z.shape

        # symmetrised lag-1 autocovariance
        C1 = Z[:-1,:].T @ Z[1:,:] / (T - 1)
        C1 = C1 + C1.T

        # construct the columns of W sequentially, Dong et al. (2022) Alg. 1
        W = np.zeros((D, 0))
        errors = []
        for _ in range(self.n_components):

            # initialise w by the exact solution for M = 1 and one component, 
            # in the orthogonal complement U of W (Dong et al., 2022, Sec. 2.3)
            U = scipy.linalg.null_space(W.T) if W.shape[1] > 0 else np.eye(D)
            lam, V = np.linalg.eigh(U.T @ C1 @ U)
            w = U @ V[:,np.argmax(np.abs(lam))]
            error = np.inf
            converged = False
            for _ in range(self.max_iters):

                # (a) VAR coefficients with W = [Wk w] fixed
                A, _ = self._fit_var(Z, np.column_stack([W, w]))

                # (b) w with A fixed
                w = self._update_w(Z, W, A)

                # prediction error of the VAR model refitted with the new w
                A, error_new = self._fit_var(Z, np.column_stack([W, w]))
                if error - error_new < self.tol:
                    converged = True
                    break
                error = error_new

            if not converged:
                warnings.warn(f"PTS did not converge in max_iters={self.max_iters} "
                              f"iterations (decrease in prediction error "
                              f"{error - error_new:.2e} >= tol={self.tol}); "
                              "consider increasing max_iters.", ConvergenceWarning)

            W = np.column_stack([W, w])
            errors.append(error_new)

        self.ar_coefs_ = A
        self.prediction_errors_ = np.array(errors)

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
