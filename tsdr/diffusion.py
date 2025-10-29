# Kieran Owens 2025
# tsdr based on graph diffusion

# Contains:
# DMcov - Temporal Laplacian Eigenmaps using covariance distance
# DIG - Dynamical Information Geometry

import numpy as np
import scipy
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS, SpectralEmbedding
from sklearn.decomposition import PCA

###############################################################################
###############################################################################
# DMcov - Diffusion Maps using covariance distance
###############################################################################
###############################################################################

class DMcov():
    """
    Diffusion maps using covariance distance (DMcov).

    DMcov is a nonlinear TSDR method for extracting components from a 
    multivariate time series based on the distance between the covariance 
    matrices of time-series segments.

    The steps involved in DMcov are: (1) compute a square matrix of covariance
    distances between time-series segments, (2) apply a Gaussian (i.e., RBF) 
    kernel to the distance matrix to obtain W, (3) compute the diagonal matrix 
    D by summing across rows of W, (4) compute the normalised matrix D^(-1/2) W D^(-1/2)
    or the graph Laplacian L = D - W, (5) apply eigendecomposition to 
    the matrix from (4) and return the leading eigenvectors (after discarding 
    the first eigenvector).

    Warning: The behaviour of this function has not been thoroughly tested.

    Reference: Coelho Rodrigues et al (2018) Multivariate time-series analysis 
    via manifold learning, IEEE Workshop on SSP

    Parameters
    ----------
    window: int
        The temporal length of the time-series segments used for covariance
        distance approximation.

    step: int
        The step size between time-series segments. Decreasing step size
        increases temporal resolution but increases computing time.

    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    sigma: float
        The sigma parameter in the Gaussian (radial basis function - RBF) 
        kernel.
        Default: 2.0.

    distance: str
        Determines the covariance distance metric. Options include 
        'mahalanobis', 'frobenius', and 'geodesic'.
        Default: 'mahalanobis' (from Duque et al, 2019).

    normalise: bool
        Whether to use the standard graph Laplacian L = D - W (False) or the
        normalized matrix D^(-1/2) W D^(-1/2).
        Default: True.
    """

    def __init__(self, window, step, n_components=1, 
                 sigma=2.0, distance='mahalanobis', normalise=True):

        self.window = window
        self.step = step
        self.n_components = n_components
        self.sigma = sigma
        self.distance = distance
        self.normalise = normalise

    def _spd_geodesic(self, A, B):

        A_neg_root = np.linalg.pinv(scipy.linalg.sqrtm(A))
        log_term = scipy.linalg.logm(A_neg_root @ B @ A_neg_root)

        return np.linalg.norm(log_term)
    
    def _cov_distance(self, X, Y):

        Cx = X.T @ X / (X.shape[0] - 1)
        Cy = Y.T @ Y / (Y.shape[0] - 1)
        mu_diff = (np.mean(X, axis=0) - np.mean(Y, axis=0)).reshape(-1,1)
        
        if self.distance == 'frobenius':

            return np.linalg.norm(Cx - Cy)
        
        elif self.distance == 'geodesic':

            return self._spd_geodesic(Cx, Cy)
        
        elif self.distance == 'mahalanobis': # From Duque et al, 2019

            return (mu_diff.T @ np.linalg.pinv(Cx + Cy) @ mu_diff)[0,0]
        
    def _cov_distance_matrix(self, X):

        T, _ = X.shape
        indices = range(0, T-  self.window, self.step)
        G = np.zeros((len(indices), len(indices)))
        for i2, i in enumerate(indices):
            for j2, j in enumerate(indices):
                if j2 > i2:
                    G[i2,j2] = self._cov_distance(X[i:i+self.window,:], 
                                                  X[j:j+self.window,:])
        return G + G.T

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

        # mean centering
        X = X - np.mean(X, axis=0)

        # compute covariance distances
        W = self._cov_distance_matrix(X)

        # Gaussian kernel weights matrix
        W = np.exp(-W**2/self.sigma)
        for i in range(W.shape[0]):
            W[i,i] = 0.0

        # diagonal weights matrix
        D = np.diag(np.sum(W, axis=0))

        if self.normalise:

            # D^(-1/2)
            D_neg_root = np.linalg.pinv(scipy.linalg.sqrtm(D))

            # normalise W
            Wnorm = D_neg_root @ W @ D_neg_root

            # eigendecomposition
            _, eigvecs = np.linalg.eig(Wnorm)

        else:

            # eigendecomposition
            _, eigvecs = np.linalg.eig(D - W)

        self.components = eigvecs[:,1:1+self.n_components]

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
            A (possibly multivariate) time series of components.
        """

        self.fit(X)
        
        return self.components

###############################################################################
###############################################################################
# DIG - Dynamical Information Geometry
###############################################################################
###############################################################################

class DIG():
    """
    Dynamical Information Geometry (DIG).

    DIG is a nonlinear TSDR method for extracting components from a 
    multivariate time series based on the distance between the covariance 
    matrices of time-series segments.

    The steps involved in DIG are: (1) compute a square matrix of covariance
    distances between time-series segments, (2) apply an adaptive Gaussian 
    (i.e., RBF) kernel to the distance matrix to obtain W, (3) normalise the 
    rows of M to yield a Markovian matrix of transition probabilities, 
    (4) compute a time scale tau using von Neumann Entropy (VNE), (5) compute 
    the tau-step diffusion probabilities M^tau, (6) compute information 
    distances D for M^tau, and (7) apply multidimensional scaling (MDS) to D.

    Warning: The behaviour of this function has not been thoroughly tested.

    Reference: Duque et al (2019) Visualizing high dimensional dynamical 
    processes, IEEE Workshop on MLSP

    Parameters
    ----------
    window: int
        The temporal length of the time-series segments used for covariance
        distance approximation.

    step: int
        The step size between time-series segments. Decreasing step size
        increases temporal resolution but increases computing time.

    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    knn: int
        The k-th nearest neighbour distance to use in the denominator of
        the adaptive Gaussian kernel.
        Default: 5.

    alpha: float
        The exponent to use in the adaptive Gaussian kernel.
        Default: 1.0.

    tmax: int
        The maximum number of diffusion steps used for the estimation of
        time scale via von Neumann entropy.
    """

    def __init__(self, window, step, n_components=1, 
                 knn=5, alpha=1.0, tmax=50):

        self.window = window
        self.step = step
        self.n_components = n_components
        self.k = knn
        self.alpha = alpha
        self.tmax = tmax

    def _spd_geodesic(self, A, B):

        A_neg_root = np.linalg.pinv(scipy.linalg.sqrtm(A))
        log_term = scipy.linalg.logm(A_neg_root @ B @ A_neg_root)

        return np.linalg.norm(log_term)
    
    def _cov_distance(self, X, Y):

        Cx = X.T @ X / (X.shape[0] - 1)
        Cy = Y.T @ Y / (Y.shape[0] - 1)
        mu_diff = (np.mean(X, axis=0) - np.mean(Y, axis=0)).reshape(-1,1)

        return (mu_diff.T @ np.linalg.pinv(Cx + Cy) @ mu_diff)[0,0]
        
    def _cov_distance_matrix(self, X):

        T, _ = X.shape
        indices = range(0, T-  self.window, self.step)
        G = np.zeros((len(indices), len(indices)))
        for i2, i in enumerate(indices):
            for j2, j in enumerate(indices):
                if j2 > i2:
                    G[i2,j2] = self._cov_distance(X[i:i+self.window,:], 
                                                  X[j:j+self.window,:])
        return G + G.T
    
    # adaptive Gaussian kernel
    def _adaptive_gaussian_kernel(self, G):

        knn = NearestNeighbors(n_neighbors=self.k, metric='precomputed').fit(G)
        knn_k = knn.kneighbors()[0][:,self.k-1]

        W = np.zeros(G.shape)
        D = G.shape[0]

        for i in range(D):
            for j in range(D):
                if i != j:
                    W[i,j] += 0.5*(np.exp(-G[i,j]/knn_k[i])**self.alpha)
                    W[i,j] += 0.5*(np.exp(-G[i,j]/knn_k[j])**self.alpha)

        return W
    
    # compute timescale using the von Neumann entropy knee
    def _timescale_vne_knee(self, W):

        eigvals, _ = np.linalg.eig(W)
        eigvals = eigvals[np.argwhere(eigvals > 0)]
        vne = []
        eigvals_t = np.copy(eigvals)
        for _ in range(self.tmax):
            prob = eigvals_t / np.sum(eigvals_t)
            vne.append(-np.sum(prob * np.log(prob)))
            eigvals_t = eigvals_t * eigvals

        kneedle = KneeLocator(range(1, self.tmax + 1), vne, 
                            S=1.0, curve='convex', direction='decreasing')
        return int(kneedle.elbow)
    
    # computation of information distance
    def _information_distance(self, W):

        D = np.zeros(W.shape)
        d = D.shape[0]
        for i in range(d):
            for j in range(d):
                argument = np.sum(np.sqrt(W[:,i]*W[:,j]))
                if argument > 1.0: # catch numerical errors
                    D[i,j] = 0.0
                else:
                    D[i,j] = 2 * np.arccos(np.sum(np.sqrt(W[:,i]*W[:,j])))
        return D
    
    # multidimensional scaling with precomputed distances
    def _mds(self, D):

        N = D.shape[0]
        H = np.eye(N) - np.ones((N, N)) / N
        K = -0.5 * (H @ D @ H)
        eigenvals, eigenvecs = np.linalg.eigh(K)
        eigenvals_sorted = eigenvals.argsort()[::-1]
        idxs = eigenvals_sorted[:self.n_components]
        Lambda = np.diag(eigenvals[idxs])
        W = eigenvecs[:, idxs]
        return (np.sqrt(Lambda) @ W.T).T

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

        # copute covariance distances
        W = self._cov_distance_matrix(X)

        # Gaussian kernel weights matrix
        W = self._adaptive_gaussian_kernel(W)

        # normalise rows
        M = W / W.sum(axis=1)

        # timescale via Von Neumann entropy
        tau = self._timescale_vne_knee(M)

        # diffusion step
        Mtau = np.linalg.matrix_power(M, tau)

        # compute information distances
        D = self._information_distance(Mtau)

        self.components = self._mds(D)

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
            A (possibly multivariate) time series of components.
        """

        self.fit(X)
        
        return self.components