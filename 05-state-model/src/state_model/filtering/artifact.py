from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass
class ModelArtifact:
    states: tuple[str, ...]
    feature_names: tuple[str, ...]
    pi: np.ndarray
    transitions: np.ndarray
    mask: np.ndarray
    means: np.ndarray  # standardized Gaussian means (K, D)
    variances: np.ndarray  # standardized diagonal variances (K, D)
    scaler_mean: np.ndarray  # original feature units (D,)
    scaler_std: np.ndarray  # original feature units (D,)
    cadence_seconds: float = 1.0
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        k, d = len(self.states), len(self.feature_names)
        shapes = {
            "pi": (self.pi, (k,)),
            "transitions": (self.transitions, (k, k)),
            "mask": (self.mask, (k, k)),
            "means": (self.means, (k, d)),
            "variances": (self.variances, (k, d)),
            "scaler_mean": (self.scaler_mean, (d,)),
            "scaler_std": (self.scaler_std, (d,)),
        }
        for name, (value, shape) in shapes.items():
            if np.shape(value) != shape:
                raise ValueError(
                    f"{name} has shape {np.shape(value)}, expected {shape}"
                )
        if len(set(self.feature_names)) != d or len(set(self.states)) != k:
            raise ValueError("State and feature names must be unique")
        for name in (
            "pi",
            "transitions",
            "means",
            "variances",
            "scaler_mean",
            "scaler_std",
        ):
            if not np.all(np.isfinite(getattr(self, name))):
                raise ValueError(f"Nonfinite parameters in {name}")
        if np.any(self.pi < 0) or not np.isclose(self.pi.sum(), 1, atol=1e-7):
            raise ValueError("Initial probabilities must sum to 1")
        if np.any(self.transitions < -1e-12) or not np.allclose(
            self.transitions.sum(axis=1), 1, atol=1e-7
        ):
            raise ValueError("Each transition row must sum to 1")
        if np.any(self.transitions[~self.mask] != 0):
            raise ValueError("Transition matrix violates the binary mask")
        if np.any(self.variances <= 0) or np.any(self.scaler_std <= 0):
            raise ValueError("Variances and scaling factors must be positive")
        if self.cadence_seconds <= 0:
            raise ValueError("Cadence must be positive")

    def save(self, path: str | Path) -> None:
        self.validate()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "states": list(self.states),
            "feature_names": list(self.feature_names),
            "cadence_seconds": self.cadence_seconds,
            "metadata": self.metadata,
        }
        np.savez_compressed(
            output,
            pi=self.pi,
            transitions=self.transitions,
            mask=self.mask.astype(np.uint8),
            means=self.means,
            variances=self.variances,
            scaler_mean=self.scaler_mean,
            scaler_std=self.scaler_std,
            manifest=np.array(json.dumps(manifest)),
        )

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        with np.load(path, allow_pickle=False) as archive:
            manifest = json.loads(str(archive["manifest"].item()))
            if manifest.get("schema_version") != 1:
                raise ValueError("Unsupported model artifact format")
            params = cls(
                states=tuple(manifest["states"]),
                feature_names=tuple(manifest["feature_names"]),
                pi=np.asarray(archive["pi"], dtype=float),
                transitions=np.asarray(archive["transitions"], dtype=float),
                mask=np.asarray(archive["mask"], dtype=bool),
                means=np.asarray(archive["means"], dtype=float),
                variances=np.asarray(archive["variances"], dtype=float),
                scaler_mean=np.asarray(archive["scaler_mean"], dtype=float),
                scaler_std=np.asarray(archive["scaler_std"], dtype=float),
                cadence_seconds=float(manifest["cadence_seconds"]),
                metadata=manifest.get("metadata", {}),
            )
        params.validate()
        return params
