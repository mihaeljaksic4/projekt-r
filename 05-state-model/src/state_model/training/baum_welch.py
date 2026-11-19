from dataclasses import dataclass
import numpy as np

from state_model.filtering import ModelArtifact
from state_model.filtering.numerics import gaussian_logpdf, logsumexp


@dataclass
class TrainingResult:
    artifact: ModelArtifact
    log_likelihood: list[float]
    occupancy: dict[str, float]


def forward_backward(
    x: np.ndarray, model: ModelArtifact
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Return log P(X), posterior gamma (T,K), expected transitions (K,K).
    """
    if not len(x):
        raise ValueError("An HMM session cannot be empty")
    k = len(model.states)
    emission = gaussian_logpdf(x, model.means, model.variances)
    log_pi = np.where(model.pi > 0, np.log(np.maximum(model.pi, 1e-300)), -np.inf)
    log_a = np.where(
        model.transitions > 0, np.log(np.maximum(model.transitions, 1e-300)), -np.inf
    )

    alpha = np.empty((len(x), k))
    alpha[0] = log_pi + emission[0]
    for t in range(1, len(x)):
        alpha[t] = emission[t] + logsumexp(alpha[t - 1, :, None] + log_a, axis=0)
    log_likelihood = float(logsumexp(alpha[-1]))

    beta = np.zeros((len(x), k))
    for t in range(len(x) - 2, -1, -1):
        beta[t] = logsumexp(
            log_a + emission[t + 1][None, :] + beta[t + 1][None, :], axis=1
        )

    gamma = np.exp(alpha + beta - log_likelihood)
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
    transitions = np.zeros((k, k))
    for t in range(len(x) - 1):
        log_joint = (
            alpha[t, :, None]
            + log_a
            + emission[t + 1][None, :]
            + beta[t + 1][None, :]
            - log_likelihood
        )
        transitions += np.exp(log_joint)
    return log_likelihood, gamma, transitions


def fit(
    artifact: ModelArtifact,
    raw_sequences: list[np.ndarray],
    *,
    iterations: int = 15,
    min_variance: float = 0.025,
    transition_prior_strength: float = 3.0,
    learn_pi: bool = False,
    tolerance: float = 1e-4,
) -> TrainingResult:
    """
    Learn means/variances and allowed transitions.
    """
    artifact.validate()
    if not raw_sequences:
        raise ValueError("Need at least one training sequence")
    sequences = [
        (np.asarray(x) - artifact.scaler_mean) / artifact.scaler_std
        for x in raw_sequences
    ]
    if any(x.ndim != 2 or x.shape[1] != len(artifact.feature_names) for x in sequences):
        raise ValueError("Training sequence has wrong feature dimensionality")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    k, d = artifact.means.shape
    prior_a = artifact.transitions.copy()
    history = []
    occupancy = np.zeros(k)

    for iteration in range(iterations):
        total_ll = 0.0
        transition_counts = np.zeros((k, k))
        occupancy = np.zeros(k)
        first_gamma = np.zeros(k)
        weighted_sum = np.zeros((k, d))
        weighted_sum_sq = np.zeros((k, d))
        weighted_count = np.zeros((k, d))

        for x in sequences:
            ll, gamma, transitions = forward_backward(x, artifact)
            total_ll += ll
            transition_counts += transitions
            occupancy += gamma.sum(axis=0)
            first_gamma += gamma[0]
            observed = np.isfinite(x)
            observed_x = np.where(observed, x, 0.0)
            weighted_count += gamma.T @ observed.astype(float)
            weighted_sum += gamma.T @ observed_x
            weighted_sum_sq += gamma.T @ (observed_x**2)

        history.append(float(total_ll))
        new_means = artifact.means.copy()
        new_vars = artifact.variances.copy()
        enough = weighted_count > 1e-8
        np.divide(weighted_sum, weighted_count, out=new_means, where=enough)
        moment = np.zeros_like(new_vars)
        np.divide(weighted_sum_sq, weighted_count, out=moment, where=enough)
        new_vars[enough] = np.maximum(
            moment[enough] - new_means[enough] ** 2, min_variance
        )

        counts = np.where(artifact.mask, transition_counts, 0.0)
        numerator = counts + transition_prior_strength * prior_a
        numerator[~artifact.mask] = 0.0
        denominators = numerator.sum(axis=1, keepdims=True)
        new_a = np.divide(
            numerator, denominators, out=prior_a.copy(), where=denominators > 0
        )

        if learn_pi:
            new_pi = (first_gamma + 0.01) / (len(sequences) + 0.01 * k)
        else:
            new_pi = artifact.pi.copy()

        artifact = ModelArtifact(
            states=artifact.states,
            feature_names=artifact.feature_names,
            pi=new_pi,
            transitions=new_a,
            mask=artifact.mask.copy(),
            means=new_means,
            variances=new_vars,
            scaler_mean=artifact.scaler_mean,
            scaler_std=artifact.scaler_std,
            cadence_seconds=artifact.cadence_seconds,
            metadata=dict(artifact.metadata),
        )
        artifact.validate()
        if iteration >= 2:
            improvement = history[-1] - history[-2]
            if improvement >= 0 and improvement < tolerance * max(1, abs(history[-2])):
                break

    # E step under FINAL parameters, for consistent final occupancy and LL.
    occupancy = np.zeros(k)
    final_ll = 0.0
    for x in sequences:
        ll, gamma, _ = forward_backward(x, artifact)
        final_ll += ll
        occupancy += gamma.sum(axis=0)
    history.append(float(final_ll))
    occupancy /= occupancy.sum()
    return TrainingResult(
        artifact=artifact,
        log_likelihood=history,
        occupancy=dict(zip(artifact.states, occupancy.tolist())),
    )
