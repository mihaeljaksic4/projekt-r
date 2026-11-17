import numpy as np


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable log(sum(exp(a))). Returns -inf when all terms are -inf."""
    a = np.asarray(a, dtype=float)
    maximum = np.max(a, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(maximum), maximum, 0.0)
    with np.errstate(under="ignore", divide="ignore"):
        result = safe_max + np.log(
            np.sum(np.exp(a - safe_max), axis=axis, keepdims=True)
        )
    if axis is None:
        return np.asarray(result).reshape(())
    return np.squeeze(result, axis=axis)


def gaussian_logpdf(
    x: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Per-state diagonal Gaussian log likelihood, marginalizing missing (NaN) features."""
    arr = np.asarray(x, dtype=float)
    one = arr.ndim == 1
    if one:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != means.shape[1]:
        raise ValueError(
            f"Expected observations with {means.shape[1]} features; got {arr.shape}"
        )
    known = np.isfinite(arr)
    safe_diff = np.where(known[:, None, :], arr[:, None, :] - means[None, :, :], 0.0)
    log_terms = np.log(2.0 * np.pi * variances)[None, :, :]
    cost = np.where(
        known[:, None, :], log_terms + safe_diff**2 / variances[None, :, :], 0.0
    )
    out = -0.5 * np.sum(cost, axis=2)
    return out[0] if one else out
