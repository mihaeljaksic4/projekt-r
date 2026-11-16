import json
from pathlib import Path
import subprocess

from state_model.filtering import ModelArtifact


def analyze_with_sage(model: ModelArtifact) -> dict:
    payload = {
        "states": model.states,
        "mask": model.mask.astype(int).tolist(),
        "transitions": model.transitions.tolist(),
        "means": model.means.tolist(),
        "variances": model.variances.tolist(),
        "feature_names": model.feature_names,
        "cadence_seconds": model.cadence_seconds,
    }
    worker = Path(__file__).with_name("sage_worker.py")
    try:
        proc = subprocess.run(
            ["sage", str(worker)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "SageMath is required for model analysis. Install SageMath and put `sage` on PATH."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"SageMath analysis failed:\n{error.stderr}") from error
    return json.loads(proc.stdout)
