# Kieran Owens 2025
# tsdr based on graph diffusion

# Contains:
# DMcov - Temporal Laplacian Eigenmaps using covariance distance
# DIG - Dynamical Information Geometry
# NLSA - Nonlinear Laplacian Spectral Analysis

import numpy as np
import scipy.linalg
import scipy.spatial.distance
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from ._validation import check_input, check_n_components, check_window
from ._whitening import fit_whitening


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
    the matrix from (4) and return the leading eigenvectors ranked by absolute
    eigenvalue (after discarding the first eigenvector).

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
        The scale parameter of the Gaussian (radial basis function - RBF) 
        kernel exp(-d^2 / sigma), where d is the covariance distance between 
        two windows (sigma thus plays the role of a variance).
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
        # ||log(A^-1/2 B A^-1/2)||_F via generalised eigenvalues of (B, A)
        lam = scipy.linalg.eigvalsh(B, A)
        return np.sqrt(np.sum(np.log(lam) ** 2))
    
    def _cov_distance(self, X, Y):

        Cx = np.cov(X, rowvar=False)
        Cy = np.cov(Y, rowvar=False)
        mu_diff = (np.mean(X, axis=0) - np.mean(Y, axis=0)).reshape(-1,1)
        
        if self.distance == 'frobenius':

            return np.linalg.norm(Cx - Cy)
        
        elif self.distance == 'geodesic':

            return self._spd_geodesic(Cx, Cy)

        elif self.distance == 'mahalanobis':
            # Duque et al (2019) Eq. 6 form, using window means and
            # within-window covariances. The quadratic form is d^2 already,
            # so return its root since the kernel squares the distance.
            # This is a simplification of Duque et al (2019) which works in
            # histogram space
            q = (mu_diff.T @ np.linalg.pinv(Cx + Cy) @ mu_diff)[0, 0]
            return np.sqrt(max(q, 0.0))
        
    def _cov_distance_matrix(self, X):

        T, _ = X.shape
        # start indices of all windows that fit in the time series
        indices = range(0, T - self.window + 1, self.step)
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

        X = check_input(X)
        T, D = X.shape

        if self.distance not in ('mahalanobis', 'frobenius', 'geodesic'):
            raise ValueError(f"Unknown distance '{self.distance}'. Options are "
                             "'mahalanobis', 'frobenius', and 'geodesic'.")

        # windows of at least 2 samples for the window covariances, and of 
        # at least D + 1 samples for the geodesic distance, which needs 
        # positive definite (nonsingular) window covariances
        if self.distance == 'geodesic':
            check_window(self.window, self.step, T, min_length=D + 1)
        else:
            check_window(self.window, self.step, T, min_length=2)

        # the components are the nontrivial eigenvectors of the 
        # n_windows x n_windows kernel matrix
        n_windows = (T - self.window) // self.step + 1
        check_n_components(self.n_components, max_components=n_windows - 1)

        # mean centering
        X = X - np.mean(X, axis=0)

        # compute covariance distances
        W = self._cov_distance_matrix(X)

        # Gaussian kernel weights matrix
        W = np.exp(-W**2/self.sigma)

        # diagonal weights matrix
        D = np.diag(np.sum(W, axis=0))

        if self.normalise:

            # D^(-1/2)
            D_neg_root = np.linalg.pinv(scipy.linalg.sqrtm(D))

            # normalise W
            Wnorm = D_neg_root @ W @ D_neg_root

            # eigendecomposition (eigenvalues in ascending order)
            eigvals, eigvecs = np.linalg.eigh(Wnorm)

            # the trivial eigenvector is the one with eigenvalue 1
            trivial = np.argmax(eigvals)

            # order the rest by |lambda| (|l1| >= |l2| >= ...), as in the paper
            order = [i for i in np.argsort(-np.abs(eigvals)) if i != trivial]

            self.components = eigvecs[:, order[:self.n_components]]

        else:

            # eigendecomposition (eigenvalues in ascending order)
            _, eigvecs = np.linalg.eigh(D - W)

            # the leading eigenvectors have the smallest eigenvalues, and the
            # first eigenvector (eigenvalue 0) is the trivial one
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
        X_new: ndarray, shape (n_windows, n_components)
            A (possibly multivariate) time series of components, with one 
            row per window (starting at samples 0, step, 2*step, ..., so 
            n_windows = (n_samples - window) // step + 1).
        """

        X = check_input(X)

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
    columns of W to yield a Markovian matrix M of transition probabilities, 
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
        Default: 10.0.

    tmax: int
        The maximum number of diffusion steps used for the estimation of
        time scale via von Neumann entropy.
    """

    def __init__(self, window, step, n_components=1, 
                 knn=5, alpha=10.0, tmax=50):

        self.window = window
        self.step = step
        self.n_components = n_components
        self.k = knn
        self.alpha = alpha
        self.tmax = tmax

    def _cov_distance(self, X, Y):

        Cx = np.cov(X, rowvar=False)
        Cy = np.cov(Y, rowvar=False)
        mu_diff = (X.mean(axis=0) - Y.mean(axis=0)).reshape(-1, 1)

        return (mu_diff.T @ np.linalg.pinv(Cx + Cy) @ mu_diff)[0, 0] 
        
    def _cov_distance_matrix(self, X):

        T, _ = X.shape
        # start indices of all windows that fit in the time series
        indices = range(0, T - self.window + 1, self.step)
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
        knn_k = knn.kneighbors()[0][:, self.k - 1]
        W = (0.5 * np.exp(-(G / knn_k[:, None]) ** self.alpha)
             + 0.5 * np.exp(-(G / knn_k[None, :]) ** self.alpha))
        np.fill_diagonal(W, 0.0)

        return W

    
    # compute timescale using the von Neumann entropy knee
    def _timescale_vne_knee(self, W):

        d = W.sum(axis=0)
        eigvals = np.linalg.eigvalsh(W / np.sqrt(np.outer(d, d)))
        eigvals = eigvals[eigvals > 0]
        vne = []
        eigvals_t = np.copy(eigvals)
        for _ in range(self.tmax):
            prob = eigvals_t / np.sum(eigvals_t)
            vne.append(-np.sum(prob * np.log(prob)))
            eigvals_t = eigvals_t * eigvals

        kneedle = KneeLocator(range(1, self.tmax + 1), vne, 
                            S=1.0, curve='convex', direction='decreasing')
        return int(kneedle.elbow) if kneedle.elbow is not None else self.tmax//2
    
    # computation of information distance
    def _information_distance(self, M):

        R = np.sqrt(M)
        D = 2 * np.arccos(np.clip(R.T @ R, -1.0, 1.0))
        np.fill_diagonal(D, 0.0)

        return D
    
    # multidimensional scaling with precomputed distances
    def _mds(self, D):

        N = D.shape[0]
        H = np.eye(N) - np.ones((N, N)) / N
        K = -0.5 * (H @ (D **2) @ H)
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

        X = check_input(X)
        T, _ = X.shape

        # windows of at least 2 samples for the window covariances
        check_window(self.window, self.step, T, min_length=2)

        # multidimensional scaling of the n_windows x n_windows information
        # distances gives at most n_windows components
        n_windows = (T - self.window) // self.step + 1
        check_n_components(self.n_components, max_components=n_windows)

        # compute covariance distances
        W = self._cov_distance_matrix(X)

        # Gaussian kernel weights matrix
        W = self._adaptive_gaussian_kernel(W)

        # normalise the columns: since W is symmetric, W.sum(axis=1) are also
        # the column sums, so each column of M sums to one (column j holds 
        # the transition probabilities from window j, as compared by 
        # _information_distance)
        M = W / W.sum(axis=1)

        # timescale via Von Neumann entropy
        tau = self._timescale_vne_knee(W)

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
        X_new: ndarray, shape (n_windows, n_components)
            A (possibly multivariate) time series of components, with one 
            row per window (starting at samples 0, step, 2*step, ..., so 
            n_windows = (n_samples - window) // step + 1).
        """

        X = check_input(X)

        self.fit(X)
        
        return self.components

###############################################################################
###############################################################################
# NLSA - Nonlinear Laplacian Spectral Analysis
###############################################################################
###############################################################################

class NLSA():
    """
    Nonlinear Laplacian Spectral Analysis (NLSA).

    NLSA is a nonlinear TSDR method for extracting temporal modes from a 
    multivariate time series. Like singular spectrum analysis (SSA), it 
    applies a singular value decomposition (SVD) to the time-lagged embedding
    of the data, but with the temporal patterns restricted to the leading 
    eigenfunctions of a Laplacian on the data manifold in embedding space, 
    which capture its nonlinear geometry and so can reveal intermittent and 
    low-variance patterns that SSA misses.

    The steps involved in NLSA are: (1) whiten the data (optional), (2) form
    the time-lagged embedding of the data with q lags, (3) build a Gaussian 
    kernel on the b nearest neighbours of each embedded sample, scaled by the
    local velocity (the distance between consecutive embedded samples), and 
    normalise it to a diffusion map transition matrix P with invariant 
    measure mu (Coifman & Lafon, 2006), (4) compute the l leading 
    eigenvectors phi of P (the Laplacian eigenfunctions), (5) compute the 
    components of the linear operator from the eigenfunctions to embedding 
    space, A = X diag(mu) phi, and its SVD A = U S V^T, and (6) return the 
    temporal modes phi V, ordered by decreasing singular value.

    Giannakis & Majda (2012) apply NLSA to the data without whitening, so 
    that the kernel distances, like the variance in PCA and SSA, are 
    dominated by the highest-variance directions of the data. Whitening (the 
    default here) instead gives all directions equal weight, so that 
    low-variance components with temporal structure can be recovered.

    The temporal modes are computed for the embedded samples with a 
    preceding sample (needed for the local velocity), i.e., for time points 
    q, ..., n_samples - 1, so the output has n_samples - q rows.

    Reference: Giannakis & Majda (2012) Nonlinear Laplacian spectral analysis
    for time series with intermittency and low-frequency variability, PNAS 
    (SI Appendix, Algorithms S1 and S2).

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        Default: 1.

    q: int
        The lag window, i.e., the number of time points in each embedded 
        sample.
        Default: 10.

    epsilon: float
        The width of the Gaussian kernel (relative to the local velocities).
        Default: 1.0.

    n_neighbours: int
        The number of nearest neighbours of each embedded sample in the 
        kernel (b).
        Default: 50.

    n_eigenfunctions: int
        The number of Laplacian eigenfunctions (l), which must be at least 
        n_components.
        Default: 20.

    whiten: bool
        Whether to whiten the data (if False, the data are mean-centred, as 
        in Giannakis & Majda, 2012).
        Default: True.

    Attributes
    ----------
    components: ndarray, shape (n_samples - q, n_components)
        The temporal modes.

    singular_values_: ndarray, shape (n_components,)
        The singular values of the temporal modes, in decreasing order.

    measure_: ndarray, shape (n_samples - q,)
        The invariant measure mu of the diffusion map (Riemannian measure), 
        with respect to which the temporal modes are orthonormal.
    """

    def __init__(self, n_components=1, q=10, epsilon=1.0, n_neighbours=50, 
                 n_eigenfunctions=20, whiten=True):

        self.n_components = n_components
        self.q = q
        self.epsilon = epsilon
        self.n_neighbours = n_neighbours
        self.n_eigenfunctions = n_eigenfunctions
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

        X = check_input(X)
        _, D = X.shape

        # the temporal modes are the right singular vectors of the 
        # (q * D) x n_eigenfunctions linear operator A
        check_n_components(self.n_components, 
                           max_components=min(self.q * D, self.n_eigenfunctions))

        if self.whiten:

            # whiten the data
            X = fit_whitening(X).transform(X)

        else:

            # mean centering
            X = X - np.mean(X, axis=0)

        # time-lagged embedding: row j is [x_{j+q-1}, x_{j+q-2}, ..., x_j], 
        # Algorithm S1 lines 1-9
        T, _ = X.shape
        E = np.column_stack([X[self.q - 1 - i:T - i,:] for i in range(self.q)])

        # local velocity in embedding space, Algorithm S2 lines 1-5, for the
        # embedded samples with a preceding sample
        xi = np.linalg.norm(E[1:,:] - E[:-1,:], axis=1)
        E = E[1:,:]
        s = E.shape[0]

        # Gaussian kernel on the b nearest neighbours of each sample, scaled 
        # by the local velocities, and symmetrised, Algorithm S2 lines 6-22
        sq_dist = scipy.spatial.distance.cdist(E, E, 'sqeuclidean')
        b = min(self.n_neighbours, s)
        neighbours = np.argpartition(sq_dist, b - 1, axis=1)[:,:b]
        rows = np.repeat(np.arange(s), b)
        cols = neighbours.ravel()
        W = np.zeros((s, s))
        W[rows, cols] = np.exp(-sq_dist[rows, cols] / (self.epsilon * xi[rows] * xi[cols]))
        W = np.maximum(W, W.T)

        # diffusion map normalisation W_ij / (Q_i Q_j) and degrees Q, 
        # giving the transition matrix P = diag(Q)^-1 W and the invariant 
        # measure mu = Q / sum(Q), Algorithm S2 lines 24-42
        Q = W.sum(axis=0)
        W = W / np.outer(Q, Q)
        Q = W.sum(axis=1)
        mu = Q / Q.sum()
        self.measure_ = mu

        # leading l eigenvectors phi of P, computed from the symmetric matrix
        # diag(Q)^-1/2 W diag(Q)^-1/2 with the same eigenvalues, and 
        # normalised so that sum_t mu_t phi_t^2 = 1 (eigenvalues in ascending 
        # order)
        l = self.n_eigenfunctions
        Wsym = W / np.sqrt(np.outer(Q, Q))
        _, eigvecs = scipy.linalg.eigh(Wsym, subset_by_index=[s - l, s - 1])
        phi = eigvecs[:,::-1] / np.sqrt(Q[:,None])
        phi = phi / np.sqrt(mu @ phi**2)

        # linear operator components A = X diag(mu) phi and its SVD, and the 
        # temporal modes phi V, Algorithm S1 lines 15-28
        A = E.T @ (mu[:,None] * phi)
        _, sigma, Vt = np.linalg.svd(A, full_matrices=False)
        self.singular_values_ = sigma[:self.n_components]
        self.components = phi @ Vt.T[:,:self.n_components]

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
        X_new: ndarray, shape (n_samples - q, n_components)
            A (possibly multivariate) time series of temporal modes, for time 
            points q, ..., n_samples - 1.
        """

        X = check_input(X)

        self.fit(X)
        
        return self.components