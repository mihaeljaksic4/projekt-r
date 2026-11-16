# Operational HMM

A single `uv` project for six-state operational state estimation:

- `filtering`: online Gaussian HMM inference (production).
- `training`: offline masked Baum–Welch parameter estimation and fixed transition topology.
- `analysis`: data quality, state interpretation, Viterbi decoding, Matplotlib plots, and SageMath diagnostics.

The model's six labels are `IDLE`, `LOAD_RISING`, `SUSTAINED_LOAD`, `THERMAL_RESPONSE`, `THERMAL_STEADY`, and `COOLING`.

## Setup

Requires Python 3.11+ , uv and SageMath. From the project directory:

```bash
uv sync
```

The `analysis` dependency group (Matplotlib and pytest) is installed by default. For a minimal production environment, use `uv sync --no-default-groups`.

## Train and inspect

```bash
uv run hmm-train --csv data/*.csv --out artifacts/omen_v1.npz

uv run hmm-analyze data \
  --csv data/*.csv --out outputs/data

uv run hmm-analyze model \
  --csv data/*.csv \
  --model artifacts/omen_v1.npz --out outputs/model
```

Training reads one or more independent OMEN CSV recordings. The `.npz` artifact contains the feature scaler, Gaussian emissions, initial probabilities, and masked transition matrix. The model analysis command also runs SageMath, writes JSON reports, and generates state timelines, Gaussian surfaces, transition plots, and observed dwell statistics.

## Use in another Python project

From the **other** uv project, add this repository as a local dependency:

```bash
uv add --editable ../05-state-model
```

Copy the trained `.npz` artifact into your application (or point to its absolute path). No training, Matplotlib, or SageMath imports are needed for filtering:

```python
from state_model.filtering import StateModel

model = StateModel.load("models/omen_v1.npz")


def on_pipeline_packet(packet: dict) -> dict:
    # Called once per second by the Time-Series Pipeline.
    result = model.update(packet)
    return {
        "current_state": result["current_state"],
        "previous_state": result["previous_state"],
        "time_in_state": result["time_in_state"],
        "state_probabilities": result["state_probabilities"],
        "transition_history": result["transition_history"],
    }
```

`packet` follows the existing Time-Series Pipeline contract:

```python
packet = {
    "timestamp": "2026-11-13T13:30:00+01:00",
    "columns": [
        "cpu_usage", "gpu_usage", "ram usage", "disk_io",
        "cpu_temp", "gpu_temp", "fan_rpm",
    ],
    "sequence": window,  # 30 rows, oldest to newest; each row matches columns
    "derived": {
        "cpu_load_slope": 0.8,
        "cpu_temp_slope": 0.31,
        "gpu_temp_slope": 0.2,
        "fan_rpm_slope": 12.0,
    },
    "context": {
        "workload": "CPU_STRESS",
        "fan_profile": "MAX",
        "performance_profile": "",
        "power_plan": "",
    },
}
```

Here `window` is the actual 30×7 sequence delivered by the pipeline. The filter uses `cpu_usage`, `gpu_usage`, `cpu_temp`, `gpu_temp`, `fan_rpm`, and the four derived slopes. 

The output also supports an unconditional state forecast:

```python
probabilities_in_30_seconds = model.forecast_state_probabilities(30)
```

This is a Markov-chain forecast, **not** a predicted future temperature or a complete controlled-system forecast.
