import numpy as np
from state_model.filtering import ModelArtifact
from state_model.filtering.numerics import gaussian_logpdf


def viterbi_path(x: np.ndarray, model: ModelArtifact) -> np.ndarray:
    x = (x - model.scaler_mean) / model.scaler_std
    emissions = gaussian_logpdf(x, model.means, model.variances)
    with np.errstate(divide="ignore"):
        log_a = np.log(model.transitions)
        log_pi = np.log(model.pi)
    n, k = emissions.shape
    delta = log_pi + emissions[0]
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        candidates = delta[:, None] + log_a
        back[t] = candidates.argmax(axis=0)
        delta = candidates.max(axis=0) + emissions[t]
    path = np.empty(n, dtype=int)
    path[-1] = delta.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path
