# Kieran Owens 2025
# tsdr based on graph diffusion

# Contains:
# DMcov - Temporal Laplacian Eigenmaps using covariance distance
# DIG - Dynamical Information Geometry
# NLSA - Nonlinear Laplacian Spectral Analysis

import warnings

import numpy as np
import scipy.linalg
import scipy.spatial.distance
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors

from ._linalg import sqrtm_psd
from ._validation import check_input, check_n_components, check_window
from ._whitening import fit_whitening


###############################################################################
###############################################################################
# Distances between time-series windows (used by DMcov and DIG)
###############################################################################
###############################################################################

# the distances between windows, and those that need positive definite 
# (nonsingular) window covariances, i.e., windows of at least n_features + 1
# samples
WINDOW_DISTANCES = ('mean_mahalanobis', 'wasserstein', 'bhattacharyya', 
                    'frobenius', 'geodesic')
NEEDS_POSITIVE_DEFINITE = ('bhattacharyya', 'geodesic')


def _check_distance(distance):
    """Check the name of a window distance, and return it with the old name
    'mahalanobis' replaced by 'mean_mahalanobis' (with a warning)."""

    if distance == 'mahalanobis':
        warnings.warn("distance='mahalanobis' compares only the means of the "
                      "windows, and has been renamed 'mean_mahalanobis'; use "
                      "that name (or a distance that also compares the window "
                      "covariances, such as 'bhattacharyya').", 
                      DeprecationWarning, stacklevel=3)
        return 'mean_mahalanobis'

    if distance not in WINDOW_DISTANCES:
        raise ValueError(f"Unknown distance '{distance}'. Options are "
                         + ", ".join(f"'{d}'" for d in WINDOW_DISTANCES) + ".")

    return distance


def _window_distance(Xi, Xj, distance):
    """The distance d between two windows Xi and Xj of a time series (rows 
    are time points), each summarised by its mean mu and covariance C.

    'mean_mahalanobis': the Mahalanobis distance between the window means,
        d^2 = (mu_i - mu_j)^T (C_i + C_j)^+ (mu_i - mu_j), as in Duque et al.
        (2019), Eq. 6. The covariances only set the scale, so two windows 
        with the same mean are at distance zero, whatever their covariances.
    'wasserstein': the 2-Wasserstein distance between the Gaussians 
        N(mu_i, C_i) and N(mu_j, C_j), 
        d^2 = |mu_i - mu_j|^2 + tr(C_i + C_j - 2 (C_i^1/2 C_j C_i^1/2)^1/2).
        It depends on the scale of the data (d^2 has the units of a variance).
    'bhattacharyya': four times the Bhattacharyya distance between the 
        Gaussians N(mu_i, C_i) and N(mu_j, C_j), 
        d^2 = (mu_i - mu_j)^T (C_i + C_j)^-1 (mu_i - mu_j) 
              + 2 log det((C_i + C_j) / 2) - log det C_i - log det C_j,
        scaled so that it equals 'mean_mahalanobis' when C_i = C_j. Like 
        'mean_mahalanobis', it does not change under an invertible linear 
        transformation of the data.
    'frobenius': the Frobenius norm of the difference of the covariances, 
        d = ||C_i - C_j||_F (the means are ignored).
    'geodesic': the affine-invariant Riemannian (geodesic) distance between 
        the covariances, d = ||log(C_i^-1/2 C_j C_i^-1/2)||_F (the means are 
        ignored).
    """

    mu_diff = Xi.mean(axis=0) - Xj.mean(axis=0)
    Ci = np.atleast_2d(np.cov(Xi, rowvar=False))
    Cj = np.atleast_2d(np.cov(Xj, rowvar=False))

    if distance == 'mean_mahalanobis':

        d2 = mu_diff @ np.linalg.pinv(Ci + Cj) @ mu_diff

    elif distance == 'wasserstein':

        Ci_root = sqrtm_psd(Ci)
        d2 = mu_diff @ mu_diff + np.trace(Ci + Cj - 2 * sqrtm_psd(Ci_root @ Cj @ Ci_root))

    elif distance == 'bhattacharyya':

        _, logdet_mean = np.linalg.slogdet((Ci + Cj) / 2)
        _, logdet_i = np.linalg.slogdet(Ci)
        _, logdet_j = np.linalg.slogdet(Cj)
        d2 = (mu_diff @ np.linalg.pinv(Ci + Cj) @ mu_diff
              + 2 * logdet_mean - logdet_i - logdet_j)

    elif distance == 'frobenius':

        d2 = np.sum((Ci - Cj) ** 2)

    elif distance == 'geodesic':

        # the eigenvalues of C_i^-1/2 C_j C_i^-1/2 are the generalised 
        # eigenvalues of (C_j, C_i)
        lam = scipy.linalg.eigvalsh(Cj, Ci)
        d2 = np.sum(np.log(lam) ** 2)

    # rounding can make d2 slightly negative when the windows are (nearly) 
    # identical
    return np.sqrt(max(d2, 0.0))


def _window_distance_matrix(X, window, step, distance):
    """The matrix of distances between the windows X[i:i+window] of X that 
    start at i = 0, step, 2*step, ... (all the windows that fit in X)."""

    T, _ = X.shape
    starts = range(0, T - window + 1, step)
    G = np.zeros((len(starts), len(starts)))
    for a, i in enumerate(starts):
        for b, j in enumerate(starts):
            if b > a:
                G[a, b] = _window_distance(X[i:i + window,:], X[j:j + window,:], 
                                           distance)

    return G + G.T


###############################################################################
###############################################################################
# DMcov - Diffusion Maps using covariance distance
###############################################################################
###############################################################################

class DMcov():
    """
    Diffusion maps using covariance distance (DMcov).

    DMcov is a nonlinear TSDR method for extracting components from a 
    multivariate time series based on the distances between time-series 
    segments (windows), each summarised by its mean and covariance. With the
    default distance, 'mean_mahalanobis', only the window means are compared
    (the covariances set the scale), so changes in covariance alone are not 
    detected; the 'wasserstein' and 'bhattacharyya' distances compare both.

    The steps involved in DMcov are: (1) compute a square matrix of distances
    between time-series segments (windows), each summarised by its mean and 
    covariance, (2) apply a Gaussian (i.e., RBF) kernel to the distance 
    matrix to obtain W, (3) compute the diagonal matrix 
    D by summing across rows of W, (4) compute the normalised matrix D^(-1/2) W D^(-1/2)
    or the graph Laplacian L = D - W, (5) apply eigendecomposition to 
    the matrix from (4) and return the leading eigenvectors ranked by absolute
    eigenvalue (after discarding the first eigenvector).

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
        kernel exp(-d^2 / sigma), where d is the distance between two windows
        (sigma thus plays the role of a variance).
        Default: 2.0.

    distance: str
        The distance between two windows, each summarised by its mean and 
        covariance: 'mean_mahalanobis' (the Mahalanobis distance between the 
        window means, Duque et al., 2019, which ignores differences in 
        covariance), 'wasserstein' (2-Wasserstein distance between Gaussians),
        'bhattacharyya' (Bhattacharyya distance between Gaussians), 
        'frobenius' or 'geodesic' (distances between the covariances only, 
        which ignore the means). 'bhattacharyya' and 'geodesic' need windows
        of at least n_features + 1 samples. See _window_distance for the 
        formulas. The old name 'mahalanobis' is accepted for 
        'mean_mahalanobis', with a DeprecationWarning.
        Default: 'mean_mahalanobis'.

    normalise: bool
        Whether to use the standard graph Laplacian L = D - W (False) or the
        normalized matrix D^(-1/2) W D^(-1/2).
        Default: True.
    """

    def __init__(self, window, step, n_components=1, 
                 sigma=2.0, distance='mean_mahalanobis', normalise=True):

        self.window = window
        self.step = step
        self.n_components = n_components
        self.sigma = sigma
        self.distance = distance
        self.normalise = normalise

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

        distance = _check_distance(self.distance)

        # windows of at least 2 samples for the window covariances, and of 
        # at least D + 1 samples for the distances that need positive 
        # definite (nonsingular) window covariances
        if distance in NEEDS_POSITIVE_DEFINITE:
            check_window(self.window, self.step, T, min_length=D + 1)
        else:
            check_window(self.window, self.step, T, min_length=2)

        # the components are the nontrivial eigenvectors of the 
        # n_windows x n_windows kernel matrix
        n_windows = (T - self.window) // self.step + 1
        check_n_components(self.n_components, max_components=n_windows - 1)

        # mean centering
        X = X - np.mean(X, axis=0)

        # distances between the windows
        G = _window_distance_matrix(X, self.window, self.step, distance)

        # Gaussian kernel weights matrix
        W = np.exp(-G**2/self.sigma)

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
    multivariate time series based on the distances between time-series 
    segments (windows), each summarised by its mean and covariance. With the
    default distance, 'mean_mahalanobis', only the window means are compared
    (the covariances set the scale), so changes in covariance alone are not 
    detected; the 'wasserstein' and 'bhattacharyya' distances compare both.

    The steps involved in DIG are: (1) compute a square matrix of distances 
    between time-series segments (windows), each summarised by its mean and 
    covariance, (2) apply an adaptive Gaussian (i.e., RBF) kernel to the 
    squared distances to obtain W, (3) normalise the 
    columns of W to yield a Markovian matrix M of transition probabilities, 
    (4) compute a time scale tau using von Neumann Entropy (VNE), (5) compute 
    the tau-step diffusion probabilities M^tau, (6) compute information 
    distances D for M^tau, and (7) apply multidimensional scaling (MDS) to D.

    Note: the distances and the recovery of known latent structure are 
    tested, but the von Neumann entropy timescale and the information 
    distances are tested only end to end, not against a reference 
    implementation.

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
        The exponent to use in the adaptive Gaussian kernel 
        exp(-(d^2 / d_k^2)^alpha), where d is the distance between two 
        windows and d_k the distance to the knn-th nearest neighbour. The 
        kernel is applied to the squared distance d^2, the quadratic form of
        Duque et al. (2019), Eq. 6 (for distance='mean_mahalanobis').
        Default: 10.0.

    tmax: int
        The maximum number of diffusion steps used for the estimation of
        time scale via von Neumann entropy.
        Default: 50.

    distance: str
        The distance between two windows, as in DMcov: 'mean_mahalanobis' 
        (the Mahalanobis distance between the window means, Duque et al., 
        2019, which ignores differences in covariance), 'wasserstein', 
        'bhattacharyya', 'frobenius' or 'geodesic' (see DMcov and 
        _window_distance).
        Default: 'mean_mahalanobis'.
    """

    def __init__(self, window, step, n_components=1, 
                 knn=5, alpha=10.0, tmax=50, distance='mean_mahalanobis'):

        self.window = window
        self.step = step
        self.n_components = n_components
        self.k = knn
        self.alpha = alpha
        self.tmax = tmax
        self.distance = distance

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
        T, D = X.shape

        distance = _check_distance(self.distance)

        # windows of at least 2 samples for the window covariances, and of 
        # at least D + 1 samples for the distances that need positive 
        # definite (nonsingular) window covariances
        if distance in NEEDS_POSITIVE_DEFINITE:
            check_window(self.window, self.step, T, min_length=D + 1)
        else:
            check_window(self.window, self.step, T, min_length=2)

        # multidimensional scaling of the n_windows x n_windows information
        # distances gives at most n_windows components
        n_windows = (T - self.window) // self.step + 1
        check_n_components(self.n_components, max_components=n_windows)

        # distances between the windows
        G = _window_distance_matrix(X, self.window, self.step, distance)

        # adaptive Gaussian kernel weights matrix, applied to the squared 
        # distances (see alpha)
        W = self._adaptive_gaussian_kernel(G**2)

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