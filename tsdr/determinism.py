# Kieran Owens 2025
# tsdr based on determinism

# Contains:
# DyCA - Dynamical Component Analysis
# DMD - Dynamic Mode Decomposition

import numpy as np
import scipy.linalg

from ._validation import check_input, check_n_components
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
    a multivariate time series (i.e., that can be represented as trajectories 
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
        The dimension (number of variables) of the time-series output. If 
        eig_threshold is used, the output is truncated to n_components 
        columns after the eigenvectors are selected by the threshold.
        Default: 1.

    eig_threshold: float in [0.0, 1.0]
        Select components based on eigenvalues that exceed eig_threshold,
        which indicate the degree of (linear ODE) determinism. If None, all 
        eigenvectors with eigenvalues <= 1 are used.
        Default: None.

    Attributes
    ----------
    linear_transform_: ndarray, shape (n_features, n_components)
        The linear transformation that yields the DyCA components when applied 
        to the centred time series X - mean_.

    mean_: ndarray, shape (n_features,)
        The mean of the fitted time series, which transform subtracts
        from X before applying linear_transform_.

    eigvals_: ndarray, shape (n_eigvals,)
        The (real) generalised eigenvalues that are <= 1, in decreasing 
        order (eigenvalues > 1 arise from the singularity of C0 and are 
        excluded, so n_eigvals <= n_features).
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

        X = check_input(X)

        # data dimensions
        T, D = X.shape

        # the linear transformation has two columns (eigenvector and C3 @
        # eigenvector) for each selected eigenvalue, so up to 2 * D columns
        check_n_components(self.n_components, max_components=2 * D)

        # fit a PCA whitening model
        pca_white = fit_whitening(X)

        # the mean of X, which transform subtracts (the components are
        # fitted to the centred, whitened data)
        self.mean_ = pca_white.mean_

        # whiten the data
        X = pca_white.transform(X)

        # derivative
        Xdot = np.gradient(X, axis=0, edge_order=1)

        # autocorrelation matrices
        C0 = (X.T @ X) / T
        C1 = (Xdot.T @ X) / T
        C2 = (Xdot.T @ Xdot) / T

        # generalised eigenvalue problem LHS w = lambda RHS w. LHS is 
        # symmetric positive semi-definite (symmetrised here, as it is only 
        # symmetric up to rounding) and RHS is the positive definite 
        # covariance of the derivative, so this is a symmetric-definite 
        # problem with real eigenvalues, which scipy.linalg.eigh returns in 
        # ascending order
        LHS = C1 @ np.linalg.pinv(C0) @ C1.T
        LHS = 0.5 * (LHS + LHS.T)
        RHS = C2
        eigvals, eigvecs = scipy.linalg.eigh(LHS, RHS)

        # order by decreasing eigenvalue, and scale the eigenvectors to unit
        # length (eigh scales them so that w^T RHS w = 1)
        eigvals, eigvecs = eigvals[::-1], eigvecs[:,::-1]
        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

        # exclude eigvals > 1 due to singularity of C0
        valid = eigvals <= 1
        self.eigvals_ = eigvals[valid]

        # select the eigenvectors (all valid ones, or those whose eigenvalues
        # exceed the threshold)
        if self.eig_threshold is None:
            selected = valid
        else:
            selected = valid & (eigvals > self.eig_threshold)
        eigvecs = eigvecs[:,selected]

        # return transformed data or else generate an error
        transform1 = pca_white.components_.T / np.sqrt(pca_white.explained_variance_)
        if eigvecs.shape[1] > 0:
            C3 = np.linalg.pinv(C1) @ C2
            W = np.concatenate((eigvecs, C3 @ eigvecs), axis=1)
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
            (linear ODE) determinism.
        """

        X = check_input(X)

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
        to the centred time series X - mean_. Complex if return_complex=True.

    mean_: ndarray, shape (n_features,)
        The mean of the fitted time series if whiten=True (whitening centres
        the data), and zero if whiten=False (exact DMD of the data as given).
        transform subtracts it from X before applying linear_transform_.

    eigvals_: ndarray (complex), shape (n_features,)
        The DMD eigenvalues, ordered by decreasing modulus.

    modes_: ndarray (complex), shape (n_features, n_features)
        The DMD modes in the coordinates of the input data, in the same order
        as eigvals_. With Z = (X - mean_) @ W the complex components of all
        modes (W being the full complex linear transformation, of which the
        first columns are linear_transform_ when return_complex=True), the
        data are reconstructed as X - mean_ = Z @ modes_.T.
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

        X = check_input(X, self.n_components)

        if self.whiten:

            # fit a PCA whitening model
            pca_white = fit_whitening(X)

            # the mean of X, which transform subtracts (whitening centres
            # the data)
            self.mean_ = pca_white.mean_

            # whiten the data
            X = pca_white.transform(X)

        else:

            # exact DMD of the data as given, without centring
            self.mean_ = np.zeros(X.shape[1], dtype=X.dtype)

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
            the modulus of their DMD eigenvalue.
        """

        X = check_input(X, self.n_components)

        self.fit(X)

        return self.transform(X)
