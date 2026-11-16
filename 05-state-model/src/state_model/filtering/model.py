from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from .artifact import ModelArtifact
from .numerics import gaussian_logpdf, logsumexp


MEASUREMENTS = ("cpu_usage", "gpu_usage", "cpu_temp", "gpu_temp", "fan_rpm")
SLOPES = ("cpu_load_slope", "cpu_temp_slope", "gpu_temp_slope", "fan_rpm_slope")
FEATURE_NAMES = MEASUREMENTS + SLOPES


def _time_seconds(timestamp: str | int | float) -> float:
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()


class StateModel:
    def __init__(self, artifact: ModelArtifact):
        artifact.validate()
        self.artifact = artifact
        self.reset()

    @classmethod
    def load(cls, path: str | Path) -> StateModel:
        return cls(ModelArtifact.load(path))

    def reset(self) -> None:
        self._posterior = self.artifact.pi.copy()
        self._initialized = False
        self._last_timestamp = None
        self._state = None
        self._previous_state = None
        self._seconds_in_state = 0.0
        self._history = deque(maxlen=32)

    @property
    def posterior(self) -> np.ndarray:
        return self._posterior.copy()

    def _step(self, x: np.ndarray, *, steps: int, timestamp: float) -> None:
        if self._initialized:
            prior = self._posterior @ np.linalg.matrix_power(
                self.artifact.transitions, steps
            )
        else:
            prior = self.artifact.pi
            self._initialized = True

        z = (x - self.artifact.scaler_mean) / self.artifact.scaler_std
        log_likelihood = gaussian_logpdf(
            z, self.artifact.means, self.artifact.variances
        )
        log_prior = np.full(len(prior), -np.inf)
        np.log(prior, where=prior > 0, out=log_prior)
        log_posterior = log_prior + log_likelihood
        log_posterior -= logsumexp(log_posterior)
        self._posterior = np.exp(log_posterior)

        state = self.artifact.states[int(self._posterior.argmax())]
        if state != self._state:
            if self._state is not None:
                self._previous_state = self._state
                self._history.append(
                    {"timestamp": timestamp, "from": self._state, "to": state}
                )
            self._state = state
            self._seconds_in_state = 0.0
        else:
            self._seconds_in_state += steps * self.artifact.cadence_seconds

    def update(self, packet: dict) -> dict:
        now = _time_seconds(packet["timestamp"])
        if self._last_timestamp is not None and now <= self._last_timestamp:
            raise ValueError("Packet timestamps must be strictly increasing")

        columns = packet["columns"]
        sequence = np.asarray(packet["sequence"], dtype=float)
        if sequence.shape != (30, len(columns)):
            raise ValueError("sequence must contain 30 rows matching columns")
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate column names")
        missing = set(MEASUREMENTS) - set(columns)
        if missing:
            raise ValueError(f"Missing pipeline columns: {sorted(missing)}")
        if tuple(self.artifact.feature_names) != MEASUREMENTS + SLOPES:
            raise ValueError("Model was trained for a different observation schema")

        indices = [columns.index(name) for name in MEASUREMENTS]
        derived = packet["derived"]
        latest_slopes = np.array(
            [
                np.nan if derived[name] is None else float(derived[name])
                for name in SLOPES
            ]
        )

        if not self._initialized:
            for i in range(29):
                x = np.r_[sequence[i, indices], np.full(len(SLOPES), np.nan)]
                self._step(
                    x, steps=1, timestamp=now - (29 - i) * self.artifact.cadence_seconds
                )
            steps = 1
        else:
            if self._last_timestamp is not None:
                steps = max(
                    1,
                    round((now - self._last_timestamp) / self.artifact.cadence_seconds),
                )

        latest = np.r_[sequence[-1, indices], latest_slopes]
        self._step(latest, steps=steps, timestamp=now)
        self._last_timestamp = now
        return {
            "timestamp": packet["timestamp"],
            "current_state": self._state,
            "previous_state": self._previous_state,
            "time_in_state": round(self._seconds_in_state, 3),
            "state_probabilities": dict(
                zip(self.artifact.states, self._posterior.tolist())
            ),
            "transition_history": list(self._history),
            "observed_features": int(np.isfinite(latest).sum()),
        }

    def forecast_state_probabilities(self, seconds_ahead: int) -> dict[str, float]:
        n = max(0, round(seconds_ahead / self.artifact.cadence_seconds))
        result = self._posterior @ np.linalg.matrix_power(self.artifact.transitions, n)
        return dict(zip(self.artifact.states, result.tolist()))
