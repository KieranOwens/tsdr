# Kieran Owens 2025
# tsdr based on latent variables

# Contains:
# LDS - Latent linear dynamical system
# DFA - Dynamic Factor Analysis

import warnings

import numpy as np
import scipy.linalg
from scipy.linalg import pinv
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state

from ._validation import check_input, check_n_components
from ._whitening import fit_whitening

###############################################################################
###############################################################################
# LDS - Latent linear dynamical system
###############################################################################
###############################################################################

class LDS():
    """
    Latent linear dynamical system (LDS).

    LDS is a TSDR method for extracting the latent dynamics of a
    multivariate time series. It models the time series as a linear
    dynamical system with Gaussian noise, where the latent state evolves
    according to a linear transition model and the observations are
    generated from the latent state through a linear observation model.

    This implementaiton is based on the Expectation-Maximization (EM)
    algorithm for fitting the parameters of the linear dynamical system.
    This involves iteratively estimating the latent state and its
    covariance using the Kalman filter, and then updating the model
    parameters based on the expectations of the latent state and its
    covariance.

    Reference: Ghahramani and Hinton (1996) Parameter estimation for
    linear dynamical systems, Technical Report

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        This is the dimensionality of the latent state space.
        Default: 1.

    max_iter: int
        The number of iterations of the EM algorithm (EM always runs for 
        max_iter iterations).
        Default: 50.

    tol: float
        A ConvergenceWarning is raised if the relative change in the 
        output/observation matrix C in the last EM iteration, 
        ||C_new - C|| / ||C||, exceeds tol. It does not affect the result.
        Default: 1e-3.

    constrain_A_identity: bool
        If True, the latent state transition dynamics matrix A is constrained
        to be the identity matrix, so that the latent state follows a random 
        walk (see DFA). If False, the latent dynamics matrix A is estimated 
        from the data.
        Default: False.

    whiten: bool
        If True, the input time series is whitened before fitting the model.
        If False, the input time series is used unchanged.
        Whitened data can improve the convergence of the EM algorithm and
        the stability of the model parameters.
        Default: True.

    random_state: int, RandomState instance or None
        Controls the random initialisation of the output/observation
        matrix C if init='random'. Pass an int for reproducible results 
        across calls (the global NumPy random state is then left unchanged).
        If None, the global NumPy random state is used.
        Default: None.

    init: str
        The initialisation of the output/observation matrix C. If 
        'autocorrelation', C is initialised (deterministically) with the 
        n_components most autocorrelated directions of the data, i.e., the
        generalised eigenvectors of the symmetrised lag-1 autocovariance 
        matrix with the largest eigenvalues (as in TICA), which helps EM 
        avoid local optima in which the latent state models noise. If 
        'random', C is initialised with standard normal entries.
        Default: 'autocorrelation'.
    """

    def __init__(self, n_components=1, max_iter=50, constrain_A_identity=False, whiten=True,
                 random_state=None, init='autocorrelation', tol=1e-3):

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.constrain_A_identity = constrain_A_identity
        self.whiten = whiten
        self.random_state = random_state
        self.init = init

    # E-step: Kalman filtering and smoothing
    def _E_step(self, Y, A, Q, C, R, mu0, V0):

        # Time-series and latent space dimensions
        T, _ = Y.shape
        d = A.shape[0]

        x_pred = np.zeros((T, d))       # \hat{x}_{t|t-1}
        V_pred = np.zeros((T, d, d))    # P_{t|t-1}
        x_filt = np.zeros((T, d))       # \hat{x}_{t|t}
        V_filt = np.zeros((T, d, d))    # P_{t|t}

        # ----- Kalman filtering -----
        for t in range(T):

            # Initial predictions
            if t == 0: 
                x_pred[t] = mu0
                V_pred[t] = V0

            # Compute predictions as per Eqs. (26), (27)
            else:
                x_pred[t] = A @ x_filt[t - 1]
                V_pred[t] = A @ V_filt[t - 1] @ A.T + Q

            # Eq. (28), compute the Kalman gain matrix
            K = V_pred[t] @ C.T @ pinv(C @ V_pred[t] @ C.T + R)

            # Eq. (29)
            x_filt[t] = x_pred[t] + K @ (Y[t] - C @ x_pred[t])

            # Eq. (30)
            V_filt[t] = V_pred[t] - K @ C @ V_pred[t]

        # ----- Kalman smoothing -----
        x_smooth = np.zeros((T, d))     # smoothed mean, x_t^T
        V_smooth = np.zeros((T, d, d))  # smothed covariance, V_t^T
        Cov = np.zeros((T - 1, d, d))   # V_{t, t-1}^T

        # initialise the smoothed estimates
        x_smooth[-1] = x_filt[-1]
        V_smooth[-1] = V_filt[-1]

        for t in reversed(range(T - 1)):
            # Eq. (31), smoothing gain
            J = V_filt[t] @ A.T @ pinv(V_pred[t + 1])
            # Eq. (32), smoothed mean
            x_smooth[t] = x_filt[t] + J @ (x_smooth[t + 1] - x_pred[t + 1])
            # Eq. (33), smoothed covariance
            V_smooth[t] = V_filt[t] + J @ (V_smooth[t + 1] - V_pred[t + 1]) @ J.T
            # Eq. (34), lag-one covariance Cov(x_{t+1}, x_t | Y) = V_{t+1}^T J_t^T
            # (J_t V_{t+1}^T is its transpose, Cov(x_t, x_{t+1} | Y))
            Cov[t] = V_smooth[t + 1] @ J.T

        # ----- Update expectations to use in the M-Step -----
        # Eq. (10), E[x_t]
        Ex = x_smooth
        # Eq. (11), P_t = V_smooth + E[x_t]E[x_t]^T
        Exx = V_smooth + np.einsum('ti,tj->tij', Ex, Ex)
        # Eq. (12), P_{t,t-1} = Cov + + E[x_t]E[x_{t-1}]^T
        Exxm1 = Cov + np.einsum('ti,tj->tij', Ex[1:], Ex[:-1])

        return Ex, Exx, Exxm1
    
    # M-step: Update model parameters based on expectations
    def _M_step(self, Y, Ex, Exx, Exxm1):

        # Time-series and latent space dimensions
        T, _ = Y.shape
        d = Ex.shape[1]

        # Sum expectations over time
        sum_Exx = Exx.sum(axis=0)           # \sum_t E[x_t x_t^T]
        sum_Exx0 = Exx[:-1].sum(axis=0)      # sum_{t=1}^{T-1}
        sum_Exx1 = Exx[1:].sum(axis=0)      # \sum_{t=2}^T E[x_t x_t^T]
        sum_Exxm1 = Exxm1.sum(axis=0)       # \sum_{t=2}^T E[x_t x_{t-1}^T]

        # Update the latent state transition dynamics matrix A, Eq. (18)
        if self.constrain_A_identity:
            A = np.eye(d)
        else:
            A = sum_Exxm1 @ pinv(sum_Exx0)

        # Update the latent state noise covariance matrix Q, Eq. (20)
        # and enforce symmetry
        #Q = (sum_Exx1 - A @ sum_Exxm1.T) / (T - 1)
        Q = (sum_Exx1 - A @ sum_Exxm1.T - sum_Exxm1 @ A.T + A @ sum_Exx0 @ A.T) / (T - 1)
        Q = (Q + Q.T) / 2  

        # Update the output/Observation matrix C, Eq. (14)
        C = (Y.T @ Ex) @ pinv(sum_Exx)

        # Update the output/Observation noise covariance matrix R, Eq. (16)
        # and enforce symmetry
        R = (Y.T @ Y - C @ (Ex.T @ Y)) / T
        R = (R + R.T) / 2

        # Update the initial latent state mean, Eq. (21)
        mu0 = Ex[0]
        
        # Update the initial latent state covariance, Eq. (24)
        V0 = Exx[0] - np.outer(Ex[0], Ex[0])

        return A, Q, C, R, mu0, V0

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

        X = check_input(X)

        # new variable for the input time series to maintain 
        # the naming conventions in the Ghahramani and Hinton paper
        Y = X.copy()

        # Input time-series dimensions
        _, D = Y.shape
        d = self.n_components

        # the autocorrelation initialisation of C needs d <= D directions of 
        # the data, whereas a random initialisation allows a latent state of
        # any dimension (e.g., d = 2 for an oscillation in univariate data)
        if self.init == 'autocorrelation':
            check_n_components(d, max_components=D)
        else:
            check_n_components(d, max_components=np.inf)

        if self.whiten:
            # fit a PCA whitening model
            self.pca_white = fit_whitening(Y)

            # whiten the data
            Y = self.pca_white.transform(Y)

        # Initialization
        A = np.eye(d)               # Latent state transition dynamics matrix
        Q = np.eye(d)               # Latent state noise covariance matrix

        # Output/Observation matrix
        if self.init == 'autocorrelation':

            # the n_components most autocorrelated directions w of the data 
            # (generalised eigenvectors of the symmetrised lag-1 
            # autocovariance and the covariance, C1 w = lambda C0 w), as 
            # observation patterns C = C0 w of unit-variance latent states
            T = Y.shape[0]
            Yc = Y - np.mean(Y, axis=0)
            C0 = Yc.T @ Yc / T
            C1 = Yc[:-1,:].T @ Yc[1:,:] / (T - 1)
            _, W = scipy.linalg.eigh((C1 + C1.T) / 2, C0)
            C = C0 @ W[:,::-1][:,:d]

        elif self.init == 'random':

            rng = check_random_state(self.random_state)
            C = rng.randn(D, d)

        else:

            raise ValueError(f"Unknown init '{self.init}'; use 'autocorrelation' or 'random'.")

        R = np.eye(D) * np.var(Y)   # Output/Observation noise covariance matrix

        mu0 = np.zeros(d)           # Initial latent state mean
        V0 = np.eye(d)              # Initial latent state covariance matrix

        # EM iterations
        change = np.inf
        for _ in range(self.max_iter):
            C_previous = C
            # E step
            Ex, Exx, Exxm1 = self._E_step(Y, A, Q, C, R, mu0, V0)
            # M step
            A, Q, C, R, mu0, V0 = self._M_step(Y, Ex, Exx, Exxm1)
            # relative change in C in this iteration
            change = np.linalg.norm(C - C_previous) / np.linalg.norm(C_previous)

        if change > self.tol:
            warnings.warn(f"{type(self).__name__} may not have converged in "
                          f"max_iter={self.max_iter} EM iterations (relative "
                          f"change in C in the last iteration {change:.2e} > "
                          f"tol={self.tol}); consider increasing max_iter.",
                          ConvergenceWarning)

        # Latent linear dynamical system parameters
        self.A = A
        self.Q = Q
        self.C = C
        self.R = R
        self.mu0 = mu0
        self.V0 = V0

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
            A (possibly multivariate) time series of components.
        """

        X = check_input(X)
        # Whiten the input data if required
        if self.whiten:

            # whiten the data
            X = self.pca_white.transform(X)

        # E-step: Kalman filtering and smoothing
        Ex, _, _ = self._E_step(X, self.A, self.Q, self.C, self.R, self.mu0, self.V0)

        return Ex

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
            A (possibly multivariate) time series of components.
        """

        X = check_input(X)

        self.fit(X)
        
        return self.transform(X)

###############################################################################
###############################################################################
# DFA - Dynamic Factor Analysis
###############################################################################
###############################################################################

class DFA(LDS):
    """
    Dynamic Factor Analysis (DFA).

    DFA is a TSDR method for extracting a small number of common trends from 
    a multivariate time series. It models the time series as a linear 
    combination of latent common trends plus noise, where each trend follows 
    a random walk: x_t = x_{t-1} + w_t and y_t = C x_t + v_t, with Gaussian 
    noise w_t and v_t.

    DFA is the latent linear dynamical system (LDS) with the transition 
    matrix A constrained to be the identity matrix, and is fitted by the same 
    EM algorithm (with Kalman filtering and smoothing). The components are 
    the smoothed estimates of the common trends. Here the covariance 
    matrices of w_t and v_t are estimated freely, rather than fixing the 
    trend noise covariance to the identity and the observation noise 
    covariance to be diagonal as in Zuur et al. (2003); this changes only the 
    scaling and rotation of the trends, not the subspace that they span.

    Reference: Zuur et al. (2003) Dynamic factor analysis to estimate common 
    trends in fisheries time series, Canadian J of Fisheries and Aquatic 
    Sciences

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        This is the number of common trends.
        Default: 1.

    max_iter: int
        The number of iterations of the EM algorithm (EM always runs for 
        max_iter iterations).
        Default: 50.

    tol: float
        A ConvergenceWarning is raised if the relative change in the 
        output/observation matrix C in the last EM iteration exceeds tol 
        (see LDS). It does not affect the result.
        Default: 1e-3.

    whiten: bool
        If True, the input time series is whitened before fitting the model.
        If False, the input time series is used unchanged.
        Default: True.

    random_state: int, RandomState instance or None
        Controls the random initialisation of the output/observation
        matrix C if init='random'. Pass an int for reproducible results 
        across calls (the global NumPy random state is then left unchanged).
        If None, the global NumPy random state is used.
        Default: None.

    init: str
        The initialisation of the output/observation matrix C, 
        'autocorrelation' (deterministic) or 'random' (see LDS).
        Default: 'autocorrelation'.
    """

    def __init__(self, n_components=1, max_iter=50, whiten=True, 
                 random_state=None, init='autocorrelation', tol=1e-3):

        # an LDS with the transition matrix constrained to the identity
        super().__init__(n_components=n_components, max_iter=max_iter, 
                         constrain_A_identity=True, whiten=whiten, 
                         random_state=random_state, init=init, tol=tol)
