import numpy as np

from state_model.training.data import TelemetryData


MEASURED = (
    "cpu_load_pct",
    "gpu_load_pct",
    "cpu_temp_c",
    "gpu_temp_c",
    "cpu_fan_rpm",
    "gpu_fan_rpm",
)


def _categorical_matrix(
    categories: list[str], labels: np.ndarray, valid: np.ndarray, count: int
) -> dict:
    values = sorted(set(np.asarray(categories)[valid]))
    matrix = np.zeros((count, len(values)), dtype=int)
    for j, category in enumerate(values):
        matrix[:, j] = np.bincount(
            labels[valid & (np.asarray(categories) == category)], minlength=count
        )
    by_state = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    by_category = np.divide(
        matrix,
        matrix.sum(axis=0, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=0, keepdims=True) > 0,
    )
    return {
        "categories": values,
        "counts": matrix.tolist(),
        "category_given_state": by_state.tolist(),
        "state_given_category": by_category.tolist(),
    }


def describe_states(
    data: TelemetryData, labels: np.ndarray, states: tuple[str, ...]
) -> dict:
    labels = np.asarray(labels, dtype=int)
    valid = np.asarray(data.present, dtype=bool) & np.isfinite(data.features).all(
        axis=1
    )
    valid &= (labels >= 0) & (labels < len(states))
    counts = np.bincount(labels[valid], minlength=len(states))
    total = int(counts.sum())
    measurements = {}
    for key, series in {
        **data.columns,
        **{name: data.features[:, i] for i, name in enumerate(data.feature_names)},
    }.items():
        if key not in (*MEASURED, *data.feature_names):
            continue
        summary = {}
        for index, name in enumerate(states):
            values = np.asarray(series)[valid & (labels == index)]
            if values.size:
                summary[name] = {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                }
            else:
                summary[name] = None
        measurements[key] = summary
    return {
        "valid_samples": total,
        "excluded_samples": int(len(labels) - total),
        "states": {
            s: {
                "samples": int(counts[i]),
                "fraction": float(counts[i] / total) if total else 0.0,
            }
            for i, s in enumerate(states)
        },
        "workload": _categorical_matrix(data.workload, labels, valid, len(states)),
        "performance_mode": _categorical_matrix(data.mode, labels, valid, len(states)),
        "measurements": measurements,
    }


def describe_dynamics(
    data: TelemetryData,
    labels: np.ndarray,
    states: tuple[str, ...],
    transitions: np.ndarray,
    mask: np.ndarray,
    cadence_seconds: float = 1.0,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    count = len(states)
    valid = np.asarray(data.present, dtype=bool) & np.isfinite(data.features).all(
        axis=1
    )
    valid &= (labels >= 0) & (labels < count)
    times = data.seconds
    adjacent = valid[1:] & valid[:-1] & np.isclose(np.diff(times), cadence_seconds)
    from_states = labels[:-1][adjacent]
    to_states = labels[1:][adjacent]
    counts = np.zeros((count, count), dtype=int)
    np.add.at(counts, (from_states, to_states), 1)

    runs = []
    i = 0
    while i < len(labels):
        if not valid[i]:
            i += 1
            continue
        start = i
        state = int(labels[i])
        i += 1
        while (
            i < len(labels)
            and valid[i]
            and labels[i] == state
            and np.isclose(times[i] - times[i - 1], cadence_seconds)
        ):
            i += 1
        end = i - 1
        left = (
            start == 0
            or not valid[start - 1]
            or not np.isclose(times[start] - times[start - 1], cadence_seconds)
        )
        right = (
            end == len(labels) - 1
            or not valid[end + 1]
            or not np.isclose(times[end + 1] - times[end], cadence_seconds)
        )
        runs.append(
            {
                "state": states[state],
                "start": data.timestamps[start].isoformat(),
                "end": data.timestamps[end].isoformat(),
                "seconds": float(times[end] - times[start] + cadence_seconds),
                "completed": not (left or right),
                "left_censored": bool(left),
                "right_censored": bool(right),
            }
        )

    durations = {}
    for state, name in enumerate(states):
        series = np.array(
            [r["seconds"] for r in runs if r["state"] == name], dtype=float
        )
        complete = np.array(
            [r["seconds"] for r in runs if r["state"] == name and r["completed"]],
            dtype=float,
        )
        stay = float(transitions[state, state])
        expected = (cadence_seconds / (1.0 - stay)) if stay < 1.0 else None
        durations[name] = {
            "runs": int(series.size),
            "completed_runs": int(complete.size),
            "censored_runs": int(series.size - complete.size),
            "observed_mean_seconds": float(complete.mean()) if complete.size else None,
            "observed_median_seconds": float(np.median(complete))
            if complete.size
            else None,
            "observed_p25_seconds": float(np.percentile(complete, 25))
            if complete.size
            else None,
            "observed_p75_seconds": float(np.percentile(complete, 75))
            if complete.size
            else None,
            "observed_max_seconds": float(complete.max()) if complete.size else None,
            "hmm_expected_seconds": expected,
        }

    switches = counts.copy()
    np.fill_diagonal(switches, 0)
    forbidden = switches * (~mask)
    return {
        "valid_samples": int(valid.sum()),
        "excluded_samples": int((~valid).sum()),
        "adjacent_pairs": int(adjacent.sum()),
        "observed_transitions": counts.tolist(),
        "observed_state_changes": switches.tolist(),
        "state_change_count": int(switches.sum()),
        "forbidden_map_changes": int(forbidden.sum()),
        "runs": runs,
        "dwell": durations,
    }
