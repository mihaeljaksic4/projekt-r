from collections import deque

import numpy as np


MIN_DWELL = {
    "IDLE": 5,
    "LOAD_RISING": 5,
    "SUSTAINED_LOAD": 15,
    "THERMAL_RESPONSE": 10,
    "THERMAL_STEADY": 20,
    "COOLING": 10,
}

GUARDS = {
    ("IDLE", "LOAD_RISING"): lambda f: f["load_onset"],
    ("LOAD_RISING", "IDLE"): lambda f: f["low_load"] and not f["cooling"],
    ("LOAD_RISING", "SUSTAINED_LOAD"): lambda f: f["sustained_load"],
    ("LOAD_RISING", "COOLING"): lambda f: f["low_load"] and f["cooling"],
    ("SUSTAINED_LOAD", "THERMAL_RESPONSE"): lambda f: f["warm"] and f["heating"],
    ("SUSTAINED_LOAD", "COOLING"): lambda f: f["low_load"] and f["cooling"],
    ("THERMAL_RESPONSE", "SUSTAINED_LOAD"): lambda f: (
        f["sustained_load"] and not f["warm"] and not f["heating"]
    ),
    ("THERMAL_RESPONSE", "THERMAL_STEADY"): lambda f: f["warm"] and f["thermal_stable"],
    ("THERMAL_RESPONSE", "COOLING"): lambda f: f["low_load"] and f["cooling"],
    ("THERMAL_STEADY", "SUSTAINED_LOAD"): lambda f: (
        f["sustained_load"] and not f["warm"]
    ),
    ("THERMAL_STEADY", "THERMAL_RESPONSE"): lambda f: f["heating"],
    ("THERMAL_STEADY", "COOLING"): lambda f: f["low_load"] and f["cooling"],
    ("COOLING", "IDLE"): lambda f: (
        f["low_load"] and not f["warm"] and f["thermal_stable"]
    ),
    ("COOLING", "LOAD_RISING"): lambda f: f["load_onset"],
}


def guard_features(columns: list[str], sequence: np.ndarray, derived: dict) -> dict:
    """Derive FSM predicates from the same 30-second packet the HMM receives."""
    col = {
        name: sequence[:, columns.index(name)]
        for name in ("cpu_usage", "gpu_usage", "cpu_temp", "gpu_temp")
    }
    load = np.fmax(col["cpu_usage"], col["gpu_usage"])
    latest = float(load[-1])
    recent = float(np.nanmean(load[-10:]))
    earlier = float(np.nanmean(load[-20:-10]))
    cpu_slope = float(derived.get("cpu_temp_slope") or 0)
    gpu_slope = float(derived.get("gpu_temp_slope") or 0)
    load_slope = float(derived.get("cpu_load_slope") or 0)
    warm = col["cpu_temp"][-1] >= 75 or col["gpu_temp"][-1] >= 50
    warm_recent = np.any(col["cpu_temp"][-30:] >= 75) or np.any(
        col["gpu_temp"][-30:] >= 50
    )
    heating = cpu_slope > 0.03 or gpu_slope > 0.02
    cooling_slope = cpu_slope < -0.03 or gpu_slope < -0.02
    return {
        "low_load": bool(latest < 15),
        "load_onset": bool(
            latest > 30 and (recent - earlier > 12 or load_slope > 0.45)
        ),
        "sustained_load": bool(np.count_nonzero(load[-10:] >= 30) >= 7),
        "warm": bool(warm),
        "heating": bool(heating),
        "thermal_stable": bool(abs(cpu_slope) <= 0.03 and abs(gpu_slope) <= 0.02),
        "cooling": bool(cooling_slope or (latest < 15 and warm_recent)),
    }


class GuardedFSM:
    def __init__(self, states: tuple[str, ...], mask: np.ndarray):
        self.states = states
        self.mask = mask
        self.reset()

    def reset(self) -> None:
        self.state = None
        self.previous_state = None
        self.entered_at = None
        self.history = deque(maxlen=32)

    def update(self, candidate: str, features: dict, timestamp: float) -> str:
        if self.state is None:
            self.state = candidate
            self.entered_at = timestamp
        elif candidate != self.state:
            i = self.states.index(self.state)
            j = self.states.index(candidate)
            assert self.entered_at is not None
            if (
                self.mask[i, j]
                and timestamp - self.entered_at >= MIN_DWELL[self.state]
                and GUARDS.get((self.state, candidate), lambda f: False)(features)
            ):
                self.previous_state = self.state
                self.history.append(
                    {"timestamp": timestamp, "from": self.state, "to": candidate}
                )
                self.state = candidate
                self.entered_at = timestamp
        return self.state

    def time_in_state(self, timestamp: float) -> float:
        return timestamp - self.entered_at if self.entered_at is not None else 0.0
