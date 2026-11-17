import numpy as np

STATES = (
    "IDLE",
    "LOAD_RISING",
    "SUSTAINED_LOAD",
    "THERMAL_RESPONSE",
    "THERMAL_STEADY",
    "COOLING",
)


TRANSITION_MASK = np.array(
    [
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 1],
    ],
    dtype=bool,
)


def default_transition_matrix() -> np.ndarray:
    stay = [0.995, 0.94, 0.995, 0.97, 0.995, 0.98]
    a = np.zeros(TRANSITION_MASK.shape, dtype=float)
    for i, p in enumerate(stay):
        other = np.flatnonzero(TRANSITION_MASK[i] & (np.arange(len(STATES)) != i))
        a[i, i] = p
        a[i, other] = (1.0 - p) / len(other)
    return a
