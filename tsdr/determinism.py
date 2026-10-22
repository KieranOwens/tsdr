# Kieran Owens 2025
# tsdr based on determinism

# Contains:
# DyCA - Dynamical Component Analysis
# DMD - Dynamic Mode Decomposition

import numpy as np
from scipy.linalg import eig
from sklearn.decomposition import PCA

from ._whitening import fit_whitening

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
        pca_white = fit_whitening(X)

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

    DMD is a TSDR method for extracting the most persistent linearly
    deterministic components from a multivariate time series. A linear
    operator that advances the (whitened) data one time step forward is
    fitted by least squares, and the components are the projections of the
    data onto its eigenvectors, so that each component evolves
    (approximately) as z_{t+1} = lambda z_t. Components are ordered by
    |lambda| (the most persistent first). The operator is conceptualised as
    a finite-dimensional approximation of the Koopman operator of the process.

    The steps involved in DMD are: (1) whiten the data (optional),
    (2) separate the data into non-lagged (X) and lagged (Y) components,
    (3) compute the linear regression matrix from X to Y via least squares,
    i.e., A = (X^T X)^-1 X^T Y, so that x_{t+1}^T ~ x_t^T A, (4) apply
    eigendecomposition to A, A V = V Lambda, and order the eigenvectors by
    the modulus of their eigenvalues, and (5) apply the eigenvector linear
    transformation to the (whitened) data.

    With data stored as rows, A is the transpose of the DMD operator
    K = Y^T (X^T)^+ of Schmid (2010) and Tu et al. (2014). The eigenvectors
    of A are therefore the left eigenvectors (adjoint modes) of K, which
    project the data onto the DMD mode amplitudes; the DMD modes themselves
    are stored in modes_. Whitening is a full-rank change of basis, under
    which the eigenvalues of A are invariant, so the eigenvalues and
    components do not depend on whitening (other than through the
    mean-centering it applies). This is exact DMD without rank truncation.

    A is real, so complex eigenvalues come in conjugate pairs. By default
    each conjugate pair contributes two real components, the real and
    imaginary parts of the complex component, which span the same
    two-dimensional (rotating) invariant subspace. The arbitrary complex
    phase of each eigenvector is fixed so that these two real components are
    orthogonal over time (the real part having the larger norm), and the sign
    so that the largest absolute value of the real part is positive; the
    components are thus reproducible and independent of the coordinates in
    which the eigendecomposition is computed.

    References:
    Schmid (2010) Dynamic mode decomposition of numerical and experimental
    data, J Fluid Mech.
    Tu et al (2014) On dynamic mode decomposition: theory and applications,
    J Comput Dyn.

    Parameters
    ----------
    n_components: int
        The dimension (number of variables) of the time-series output.
        With return_complex=False this counts real columns, so a conjugate
        pair uses two.
        Default: 1.

    whiten: bool
        Whether to whiten the data.
        Default: True.

    return_complex: bool
        If True, return complex components (one per eigenvalue, with both
        members of a conjugate pair included). If False, return real
        components (the real and imaginary parts of the member of each
        conjugate pair with positive imaginary part).
        Default: False.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the DMD components when applied
        to the input time series X. Complex if return_complex=True.

    eigvals_: ndarray (complex), shape (n_features,)
        The DMD eigenvalues, ordered by decreasing modulus.

    modes_: ndarray (complex), shape (n_features, n_features)
        The DMD modes in the coordinates of the input data, in the same order
        as eigvals_. With Z = (X - mean) @ W the complex components of all
        modes (W being the full complex linear transformation, of which the
        first columns are linear_transform_ when return_complex=True), the
        data are reconstructed as X - mean = Z @ modes_.T. Here mean is the
        mean of X if whiten=True and zero otherwise.
    """

    def __init__(self, n_components=1, whiten=True, return_complex=False):

        self.n_components = n_components
        self.whiten = whiten
        self.return_complex = return_complex

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
            pca_white = fit_whitening(X)

            # whiten the data
            X = pca_white.transform(X)

        # data and time-lagged data
        X, Y = X[:-1,:], X[1:]

        # linear regression from X to Y
        # the linear regression coefficient is A = (X^T X)^-1 X^T Y
        # (X^T X = I is not presumed for whitened data: the eigenvalues
        # depend on its scale, and the lag-truncated X is only approximately
        # white)
        A = np.linalg.pinv(X.T @ X) @ X.T @ Y

        # eigendecomposition of the linear regression transformation
        # A V = V Lambda, so that (X V)_{t+1} ~ (X V)_t Lambda
        eigvals, V = np.linalg.eig(A)

        # order by decreasing |lambda|, keeping each conjugate pair adjacent
        # with the positive imaginary part first
        order = np.lexsort((-eigvals.imag, -np.abs(eigvals)))
        eigvals, V = eigvals[order], V[:, order]

        # fix the arbitrary complex phase of each eigenvector so that the
        # real and imaginary parts of its component Z = X V are orthogonal
        # over time (sum_t z_t^2 real and positive, so that Re z has the
        # larger norm); this is unaffected by a change of basis of X
        Z = X @ V
        V = V * np.exp(-0.5j * np.angle(np.sum(Z * Z, axis=0)))

        # fix the remaining sign so that the largest absolute value of
        # the real part of each component is positive
        Z = X @ V
        idx = np.argmax(np.abs(Z.real), axis=0)
        V = V * np.sign(Z.real[idx, np.arange(Z.shape[1])])

        # DMD modes (in the coordinates of X) are the columns of V^-T
        modes = np.linalg.inv(V).T

        # select components
        if self.return_complex:

            transform2 = V[:, :self.n_components]

        else:

            # one real column for a real eigenvalue, and two (Re, Im) for
            # the member of a conjugate pair with positive imaginary part
            # (np.linalg.eig returns exactly real eigenvalues of a real
            # matrix with zero imaginary part)
            cols = []
            for k in range(len(eigvals)):
                if eigvals[k].imag == 0:
                    cols.append(V[:, k].real)
                elif eigvals[k].imag > 0:
                    cols.extend([V[:, k].real, V[:, k].imag])
            transform2 = np.column_stack(cols)[:, :self.n_components]

        # get linear transformation
        if self.whiten:

            transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
            self.linear_transform_ = (transform1 @ transform2)

            # modes in data coordinates: x - mean = transform1^-T x_white
            self.modes_ = np.linalg.inv(transform1).T @ modes

        else:

            self.linear_transform_ = transform2
            self.modes_ = modes

        self.eigvals_ = eigvals

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
            the modulus of their DMD eigenvalue.
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
            the modulus of their DMD eigenvalue.
        """

        self.fit(X)

        return self.transform(X)
