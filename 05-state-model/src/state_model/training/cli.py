import argparse
import json
from pathlib import Path

from .baum_welch import fit
from .data import load_omen_csv
from .initialization import initialize_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked Gaussian HMM training")
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()

    recordings = [load_omen_csv(path) for path in args.csv]
    sequences = [recording.features for recording in recordings]
    initial = initialize_model(sequences, model_version="exploratory-v1")
    trained = fit(initial, sequences, iterations=args.iterations)
    trained.artifact.save(args.out)

    report = {
        "log_likelihood": trained.log_likelihood,
        "state_occupancy": trained.occupancy,
        "recordings": [
            {"csv": str(path), **recording.quality_report()}
            for path, recording in zip(args.csv, recordings)
        ],
    }
    args.out.with_suffix(".training.json").write_text(json.dumps(report, indent=2))
    print(f"Model saved to {args.out} ({len(recordings)} independent recordings)")


if __name__ == "__main__":
    main()
