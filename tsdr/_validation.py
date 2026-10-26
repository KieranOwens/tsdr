# Shared checks of the inputs and parameters of the tsdr methods

import numbers

import numpy as np
from sklearn.utils import check_array


def check_input(X, n_components=None):
    """Check that X is a 2-D array of finite numbers, and optionally that
    n_components is between 1 and the number of features of X.

    X may be any array-like, e.g., a NumPy array or a pandas DataFrame, and
    is returned as a NumPy array. float64 and float32 arrays keep their
    dtype (and are not copied); other numeric arrays are converted to
    float64. A ValueError is raised if X is not 2-D or contains NaN or
    infinite values.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        A multivariate time series of shape (n_samples, n_features).

    n_components : int or None
        If not None, the number of components, which must satisfy
        1 <= n_components <= n_features.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        The validated time series.
    """

    X = check_array(X, ensure_2d=True, dtype=[np.float64, np.float32])

    if n_components is not None:
        check_n_components(n_components, max_components=X.shape[1])

    return X


def check_n_components(n_components, max_components):
    """Check that n_components is an integer with
    1 <= n_components <= max_components."""

    if not isinstance(n_components, numbers.Integral) or isinstance(n_components, bool):
        raise ValueError(f"n_components must be an integer (got {n_components!r}).")

    if not 1 <= n_components <= max_components:
        raise ValueError(f"n_components must be between 1 and {max_components} "
                         f"for this input (got {n_components}).")


def check_lag(lag, n_samples):
    """Check that the time lag is an integer with 1 <= lag < n_samples."""

    if not isinstance(lag, numbers.Integral) or isinstance(lag, bool):
        raise ValueError(f"lag must be an integer (got {lag!r}).")

    if not 1 <= lag < n_samples:
        raise ValueError(f"lag must be between 1 and n_samples - 1 = "
                         f"{n_samples - 1} (got {lag}).")


def check_n_windows(n_windows, n_samples, min_length=2):
    """Check that the time series can be split into n_windows contiguous
    windows of n_samples // n_windows >= min_length samples each."""

    if not isinstance(n_windows, numbers.Integral) or isinstance(n_windows, bool):
        raise ValueError(f"n_windows must be an integer (got {n_windows!r}).")

    if n_windows < 1 or n_samples // n_windows < min_length:
        raise ValueError(f"n_windows must be between 1 and n_samples // "
                         f"{min_length} = {n_samples // min_length}, so that each "
                         f"window has at least {min_length} samples (got "
                         f"n_windows={n_windows} for n_samples={n_samples}).")


def check_window(window, step, n_samples, min_length=2):
    """Check the sliding windows of length window, taken every step samples:
    min_length <= window <= n_samples and step >= 1."""

    if not isinstance(window, numbers.Integral) or isinstance(window, bool):
        raise ValueError(f"window must be an integer (got {window!r}).")

    if not min_length <= window <= n_samples:
        raise ValueError(f"window must be between {min_length} and n_samples = "
                         f"{n_samples} (got {window}).")

    if not isinstance(step, numbers.Integral) or isinstance(step, bool) or step < 1:
        raise ValueError(f"step must be a positive integer (got {step!r}).")
