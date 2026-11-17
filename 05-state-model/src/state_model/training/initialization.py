import numpy as np
from state_model.filtering import ModelArtifact, FEATURE_NAMES
from .spec import STATES, TRANSITION_MASK, default_transition_matrix


def heuristic_labels(x: np.ndarray) -> np.ndarray:
    cpu, gpu, temp, _, _, load_slope, temp_slope, _, _ = x.T
    high = np.fmax(cpu, gpu)
    load_slope = np.nan_to_num(load_slope, nan=0.0)
    temp_slope = np.nan_to_num(temp_slope, nan=0.0)

    # 0 Idle, 1 Rising, 2 Sustained, 3 Response, 4 Steady, 5 Cooling.
    labels = np.full(len(x), 2, dtype=int)
    labels[(high < 20)] = 0
    labels[(high < 20) & (temp_slope < -0.035)] = 5
    labels[(high >= 20) & (load_slope > 0.45)] = 1
    labels[(high >= 20) & (temp >= 67) & (temp_slope > 0.035)] = 3
    labels[(high >= 20) & (temp >= 77) & (np.abs(temp_slope) <= 0.035)] = 4
    return labels


def initialize_model(
    raw_sequences: list[np.ndarray], *, model_version: str
) -> ModelArtifact:
    raw = np.concatenate(raw_sequences, axis=0)
    if raw.ndim != 2 or raw.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Observation shape does not match FEATURE_NAMES")
    with np.errstate(invalid="ignore"):
        means = np.nanmean(raw, axis=0)
        std = np.nanstd(raw, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-3), std, 1.0)
    standardized = (raw - means) / std
    raw_labels = heuristic_labels(raw)

    hand_means = np.array(
        [
            [5, 1, 53, 40, 2200, 0.00, -0.01, -0.01, 0],
            [62, 15, 65, 43, 2900, 1.00, 0.06, 0.03, 20],
            [78, 25, 73, 45, 3500, 0.00, 0.00, 0.00, 0],
            [78, 25, 83, 50, 4600, 0.00, 0.12, 0.08, 40],
            [78, 30, 91, 53, 5500, 0.00, 0.00, 0.00, 0],
            [7, 1, 70, 45, 3400, -0.15, -0.08, -0.04, -30],
        ]
    )
    fitted_means = (hand_means - means) / std
    fitted_vars = np.full_like(fitted_means, 0.9**2)
    counts = {}
    for state_idx, state in enumerate(STATES):
        eligible = standardized[raw_labels == state_idx]
        counts[state] = len(eligible)
        if len(eligible) < 15:
            continue
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(eligible, axis=0)
            var = np.nanvar(eligible, axis=0)
        fitted_means[state_idx] = np.where(np.isfinite(mu), mu, fitted_means[state_idx])
        fitted_vars[state_idx] = np.clip(
            np.where(np.isfinite(var), var, fitted_vars[state_idx]), 0.07**2, 2.5**2
        )

    pi = np.array([0.90, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=float)
    artifact = ModelArtifact(
        states=STATES,
        feature_names=FEATURE_NAMES,
        pi=pi,
        transitions=default_transition_matrix(),
        mask=TRANSITION_MASK.copy(),
        means=fitted_means,
        variances=fitted_vars,
        scaler_mean=means,
        scaler_std=std,
        metadata={
            "model_version": model_version,
            "seeded_state_counts_not_ground_truth": counts,
        },
    )
    artifact.validate()
    return artifact
