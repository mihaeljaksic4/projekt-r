import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from state_model.filtering import ModelArtifact, StateModel, MEASUREMENTS, SLOPES
from state_model.training.data import load_omen_csv
from .plots import data_plots, model_plots
from .diagnostics import describe_states, describe_dynamics
from .diagnostic_plots import interpretation_plots, dynamics_plots
from .sage_math import analyze_with_sage
from .viterbi import viterbi_path

RAW_COLUMNS = (
    "cpu_load_pct",
    "gpu_load_pct",
    "cpu_temp_c",
    "gpu_temp_c",
    "cpu_fan_rpm",
)


def filtered_posteriors(data, artifact):
    model = StateModel(artifact)
    raw = np.column_stack([data.columns[k] for k in RAW_COLUMNS])
    probabilities = []
    for t in range(29, len(raw)):
        packet = {
            "timestamp": data.timestamps[t].isoformat(),
            "columns": list(MEASUREMENTS),
            "sequence": raw[t - 29 : t + 1].tolist(),
            "derived": dict(zip(SLOPES, data.features[t, 5:].tolist())),
            "context": {"workload": data.workload[t], "fan_profile": data.mode[t]},
        }
        result = model.update(packet)
        probabilities.append(list(result["state_probabilities"].values()))
    return np.asarray(probabilities)


def analyze_model(data, artifact, sage, history, out, sage_path):
    """Generate one recording's plots and state diagnostics."""
    if artifact.feature_names != data.feature_names:
        raise ValueError("Model and data feature schemas differ; retrain the model")
    posteriors = filtered_posteriors(data, artifact)
    path = viterbi_path(data.features, artifact)[29:]
    sliced = replace(
        data,
        timestamps=data.timestamps[29:],
        columns={k: v[29:] for k, v in data.columns.items()},
        workload=data.workload[29:],
        mode=data.mode[29:],
        present=data.present[29:],
        features=data.features[29:],
    )
    plots = model_plots(sliced, artifact, posteriors, path, sage, history, out)
    map_path = posteriors.argmax(axis=1)
    interpretation = describe_states(sliced, map_path, artifact.states)
    map_dynamics = describe_dynamics(
        sliced,
        map_path,
        artifact.states,
        artifact.transitions,
        artifact.mask,
        artifact.cadence_seconds,
    )
    viterbi_dynamics = describe_dynamics(
        sliced,
        path,
        artifact.states,
        artifact.transitions,
        artifact.mask,
        artifact.cadence_seconds,
    )
    dynamics = {"online_map": map_dynamics, "viterbi": viterbi_dynamics}
    (out / "state_interpretation.json").write_text(json.dumps(interpretation, indent=2))
    (out / "state_dynamics.json").write_text(json.dumps(dynamics, indent=2))
    plots += interpretation_plots(
        sliced, map_path, artifact.states, interpretation, out
    )
    plots += dynamics_plots(
        artifact.states, artifact.mask, map_dynamics, viterbi_dynamics, out
    )
    report = {
        "observations_filtered": len(posteriors),
        "map_state_counts": dict(
            zip(
                artifact.states,
                np.bincount(map_path, minlength=len(artifact.states)).tolist(),
            )
        ),
        "map_vs_viterbi_agreement": float(np.mean(map_path == path)),
        "posterior_entropy_mean_nats": float(
            np.mean(
                -np.sum(posteriors * np.log(np.maximum(posteriors, 1e-300)), axis=1)
            )
        ),
        "sage_report": sage_path,
        "interpretation_report": "state_interpretation.json",
        "dynamics_report": "state_dynamics.json",
        "map_observed_switches": map_dynamics["state_change_count"],
        "viterbi_observed_switches": viterbi_dynamics["state_change_count"],
        "map_forbidden_direct_switches": map_dynamics["forbidden_map_changes"],
    }
    return plots, report


def main() -> None:
    parser = argparse.ArgumentParser(description="OMEN HMM diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "model"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--csv", nargs="+", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        if name == "model":
            cmd.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    multiple = len(args.csv) > 1
    if multiple and len({path.stem for path in args.csv}) != len(args.csv):
        parser.error(
            "CSV files must have distinct basenames when analyzing multiple recordings"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    if args.command == "model":
        artifact = ModelArtifact.load(args.model)
        sage = analyze_with_sage(artifact)  # Only once: independent of the CSVs.
        training_report = args.model.with_suffix(".training.json")
        history = (
            json.loads(training_report.read_text()).get("log_likelihood")
            if training_report.exists()
            else None
        )
        sage_out = args.out / "sage_diagnostics.json"
        sage_out.write_text(json.dumps(sage, indent=2))

    index = []
    for path in args.csv:
        data = load_omen_csv(path)
        out = args.out / path.stem if multiple else args.out
        out.mkdir(parents=True, exist_ok=True)
        if args.command == "data":
            plots, report = data_plots(data, out), data.quality_report()
        else:
            sage_ref = (
                "../sage_diagnostics.json" if multiple else "sage_diagnostics.json"
            )
            plots, report = analyze_model(data, artifact, sage, history, out, sage_ref)
        (out / "summary.json").write_text(json.dumps(report, indent=2))
        index.append(
            {
                "csv": str(path),
                "report": str((out / "summary.json").relative_to(args.out)),
            }
        )
        print(f"Analyzed {path} -> {out} ({len(plots)} plots)")

    if multiple:
        (args.out / "summary.json").write_text(
            json.dumps({"recordings": index}, indent=2)
        )
    print(f"Saved {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
