# Kieran Owens 2025
# tsdr based on slowness

# Contains:
# SFA - Slow Feature Analysis
# BioSFA - Biologically plausible SFA
# ISFA - Independent Slow Feature Analysis

import numbers
import warnings

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
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

    eigvals_: ndarray, shape (n_features,)
        The variance of the temporal derivative (difference) of each 
        (unit-variance) slow feature, in ascending order (slowest first), 
        i.e., in the same order as the components.
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

        # variance of the derivative of each slow feature, in ascending 
        # order to match the order of the components (slowest first)
        self.eigvals_ = pca_diff.explained_variance_[::-1]

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

###############################################################################
###############################################################################
# ISFA - Independent Slow Feature Analysis
###############################################################################
###############################################################################

class ISFA():
    """
    Independent Slow Feature Analysis (ISFA).

    ISFA is a TSDR method for extracting components from a multivariate time 
    series that are both slow (i.e., for which the square of the first 
    temporal derivative is minimised) and statistically independent (i.e., 
    for which the time-lagged cross-correlations are minimised), combining 
    SFA with second-order independent component analysis (ICA).

    The steps involved in ISFA are: (1) whiten the data, (2) compute the 
    symmetrised time-lagged correlation matrices C(tau) for the ICA time lags 
    and for tau = 1, (3) find an orthogonal transformation Q that minimises 
    b_ica * sum_tau sum_{i != j} C_ij(tau)^2 - b_sfa * sum_i C_ii(1)^2 over 
    the first n_components output components, starting from the SFA solution 
    and applying successive Givens rotations at their optimal angles, and 
    (4) apply the linear transformation to the whitened data.

    Blaschke et al. (2007) apply ISFA to a polynomial expansion of the data 
    for nonlinear blind source separation; this is achieved here by expanding
    X (e.g., with sklearn.preprocessing.PolynomialFeatures) before fitting.

    Reference: Blaschke, Zito & Wiskott (2007) Independent slow feature 
    analysis and nonlinear blind source separation, Neural Computation.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    lags: int or sequence of int
        The time lags of the ICA part of the objective: lags 1, ..., lags for 
        an integer, or else the given lags.
        Default: 10.

    b_ica: float
        The weight of the ICA (independence) part of the objective.
        Default: 1.0.

    b_sfa: float
        The weight of the SFA (slowness) part of the objective.
        Default: 1.0.

    max_iters: int
        The maximum number of sweeps of Givens rotations over all pairs of 
        axes. A ConvergenceWarning is raised if the optimisation has not 
        converged (see tol) after max_iters sweeps.
        Default: 100.

    tol: float
        The optimisation stops when the relative decrease of the objective in 
        a sweep falls below tol.
        Default: 1e-8.

    random_state: int, RandomState instance or None
        Controls the random order of the pairs of axes in each sweep. Pass an 
        int for reproducible results across calls (the global NumPy random 
        state is then left unchanged). If None, the global NumPy random state 
        is used.
        Default: None.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the ISFA components when applied 
        to the input time series X.

    objective_: float
        The value of the ISFA objective at the solution.
    """

    def __init__(self, n_components=1, lags=10, b_ica=1.0, b_sfa=1.0, 
                 max_iters=100, tol=1e-8, random_state=None):

        self.n_components = n_components
        self.lags = lags
        self.b_ica = b_ica
        self.b_sfa = b_sfa
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

    def _objective(self, C):
        """The ISFA objective, Blaschke et al. (2007) eq. (4.3), for the 
        correlation matrices C (C[0] at lag 1, C[1:] at the ICA lags) of the 
        current output components."""

        R = self.n_components
        Cr = C[:,:R,:R]

        # ICA part: squared off-diagonal elements at the ICA lags
        ica = np.sum(Cr[1:]**2) - np.sum(np.diagonal(Cr[1:], axis1=1, axis2=2)**2)

        # SFA part: squared diagonal elements at lag 1
        sfa = np.sum(np.diag(Cr[0])**2)

        return self.b_ica * ica - self.b_sfa * sfa

    def _rotate(self, C, mu, nu, phi):
        """Apply the Givens rotation in the (mu, nu)-plane by angle phi, 
        Blaschke et al. (2007) eqs. (4.4) and (4.6), to the matrices C."""

        G = np.eye(C.shape[1])
        G[mu,mu] = G[nu,nu] = np.cos(phi)
        G[mu,nu] = -np.sin(phi)
        G[nu,mu] = np.sin(phi)

        return G.T @ C @ G, G

    def _optimal_angle(self, C, mu, nu):
        """The rotation angle in the (mu, nu)-plane that minimises the 
        objective. As a function of the angle, the objective is 
        a0 + a2 cos(2 phi) + b2 sin(2 phi) + a4 cos(4 phi) + b4 sin(4 phi), 
        Blaschke et al. (2007) eqs. (4.10)-(4.11), whose coefficients are 
        found exactly from its values at five angles."""

        angles = np.arange(5) * np.pi / 5
        values = [self._objective(self._rotate(C, mu, nu, phi)[0]) for phi in angles]
        basis = lambda phi: np.column_stack([np.ones_like(phi), np.cos(2*phi), np.sin(2*phi),
                                             np.cos(4*phi), np.sin(4*phi)])
        coef = np.linalg.solve(basis(angles), values)

        # minimise the trigonometric polynomial on a grid, refined locally
        grid = np.linspace(-np.pi/2, np.pi/2, 721)
        phi0 = grid[np.argmin(basis(grid) @ coef)]
        result = scipy.optimize.minimize_scalar(lambda phi: (basis(np.array([phi])) @ coef)[0],
                                                bounds=(phi0 - np.pi/720, phi0 + np.pi/720),
                                                method='bounded')

        return result.x

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

        # ICA time lags
        if isinstance(self.lags, numbers.Integral):
            lags = list(range(1, int(self.lags) + 1))
        else:
            lags = [int(l) for l in self.lags]

        # symmetrised time-lagged correlation matrices, Blaschke et al. (2007)
        # eq. (2.4), for the SFA lag (1) followed by the ICA lags
        C = []
        for lag in [1] + lags:
            Ctau = Xw[:-lag,:].T @ Xw[lag:,:] / (T - lag - 1)
            C.append(0.5 * (Ctau + Ctau.T))
        C = np.array(C)

        # initialise with the SFA solution (the eigenvectors of C(1) in 
        # descending order of eigenvalue), Blaschke et al. (2007) Sec. 5
        _, eigvecs = np.linalg.eigh(C[0])
        Q = eigvecs[:,::-1]
        C = Q.T @ C @ Q

        # pairs of axes (mu, nu) with mu < nu and mu within the first 
        # n_components axes, Blaschke et al. (2007) Sec. 4.2
        pairs = [(mu, nu) for mu in range(self.n_components) for nu in range(mu + 1, D)]

        rng = check_random_state(self.random_state)
        objective = self._objective(C)
        converged = False
        for _ in range(self.max_iters):

            # sweep over the pairs of axes in a random order, rotating each 
            # pair by its optimal angle
            for i in rng.permutation(len(pairs)):
                mu, nu = pairs[i]
                C, G = self._rotate(C, mu, nu, self._optimal_angle(C, mu, nu))
                Q = Q @ G

            # relative decrease of the objective in this sweep
            objective_new = self._objective(C)
            decrease = (objective - objective_new) / max(abs(objective), 1e-12)
            objective = objective_new
            if decrease < self.tol:
                converged = True
                break

        if not converged:
            warnings.warn(f"ISFA did not converge in max_iters={self.max_iters} "
                          f"sweeps (relative decrease {decrease:.2e} >= tol="
                          f"{self.tol}); consider increasing max_iters.",
                          ConvergenceWarning)

        self.objective_ = objective

        # order the components by slowness (descending lag-1 correlation)
        order = np.argsort(-np.diag(C[0])[:self.n_components])

        # get linear transformation
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        transform2 = Q[:,order]
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
            A (possibly multivariate) time series of independent components 
            ordered by slowness.
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
            A (possibly multivariate) time series of independent components 
            ordered by slowness.
        """

        self.fit(X)
        
        return self.transform(X)
