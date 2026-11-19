import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter
import numpy as np

from state_model.filtering import FEATURE_NAMES

RAW_COLUMNS = (
    "cpu_load_pct",
    "gpu_load_pct",
    "cpu_temp_c",
    "gpu_temp_c",
    "cpu_fan_rpm",
    "gpu_fan_rpm",
)


@dataclass
class TelemetryData:
    timestamps: list[datetime]
    columns: dict[str, np.ndarray]
    workload: list[str]
    mode: list[str]
    present: np.ndarray
    duplicates: int
    source_rows: int
    features: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def seconds(self) -> np.ndarray:
        if not self.timestamps:
            return np.array([])
        first = self.timestamps[0]
        return np.array([(t - first).total_seconds() for t in self.timestamps])

    def quality_report(self) -> dict:
        report = {
            "source_rows": self.source_rows,
            "resampled_seconds": len(self.timestamps),
            "missing_seconds": int((~self.present).sum()),
            "duplicate_timestamps": self.duplicates,
            "duration_seconds": float(self.seconds[-1]) if self.timestamps else 0.0,
            "start": self.timestamps[0].isoformat() if self.timestamps else None,
            "end": self.timestamps[-1].isoformat() if self.timestamps else None,
            "workload_rows": dict(
                Counter(w for w, present in zip(self.workload, self.present) if present)
            ),
            "mode_rows": dict(
                Counter(m for m, present in zip(self.mode, self.present) if present)
            ),
            "feature_missing_percent": {},
        }
        for i, name in enumerate(self.feature_names):
            report["feature_missing_percent"][name] = round(
                100.0 * float((~np.isfinite(self.features[:, i])).mean()), 2
            )
        for key in ("cpu_temp_c", "gpu_temp_c"):
            x = self.columns[key]
            valid_adjacent = np.isfinite(x[1:]) & np.isfinite(x[:-1])
            unchanged = np.abs(x[1:] - x[:-1]) < 1e-12
            report[f"{key}_unchanged_adjacent_pct"] = (
                round(100.0 * float(unchanged[valid_adjacent].mean()), 2)
                if valid_adjacent.any()
                else None
            )
        return report


def _rolling_slope(y: np.ndarray, window: int, min_points: int = 10) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for t in range(len(y)):
        start = max(0, t - window + 1)
        segment = y[start : t + 1]
        valid = np.isfinite(segment)
        if np.sum(valid) < min_points:
            continue
        x = np.flatnonzero(valid).astype(float)
        values = segment[valid]
        x -= x.mean()
        denom = x @ x
        if denom > 0:
            out[t] = (x @ (values - values.mean())) / denom
    return out


def load_omen_csv(path: str | Path, *, slope_window_seconds: int = 30) -> TelemetryData:
    """
    Read the supplied OMEN logger schema, de-duplicate, and resample to 1 Hz.
    """
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Empty CSV file")
        required = {"timestamp", *RAW_COLUMNS}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in OMEN CSV: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("OMEN CSV has no data rows")

    # Prefer the last observation if a second was logged more than once.
    unique = {}
    for row in rows:
        stamp = datetime.fromisoformat(row["timestamp"].strip())
        unique[stamp] = row
    duplicates = len(rows) - len(unique)
    beginning, ending = min(unique), max(unique)
    n = int((ending - beginning).total_seconds()) + 1
    timestamps = [beginning + timedelta(seconds=t) for t in range(n)]
    columns = {name: np.full(n, np.nan) for name in RAW_COLUMNS}
    workload = ["UNKNOWN"] * n
    mode = ["UNKNOWN"] * n
    present = np.zeros(n, dtype=bool)
    previous_workload = "UNKNOWN"
    previous_mode = "UNKNOWN"
    for idx, timestamp in enumerate(timestamps):
        row = unique.get(timestamp)
        if row is not None:
            present[idx] = True
            for name in RAW_COLUMNS:
                val = row[name].strip()
                columns[name][idx] = float(val) if val else np.nan
            previous_workload = row.get("stress_type", "UNKNOWN").strip() or "UNKNOWN"
            previous_mode = row.get("mode", "UNKNOWN").strip() or "UNKNOWN"
        workload[idx] = previous_workload
        mode[idx] = previous_mode

    load_slope = _rolling_slope(columns["cpu_load_pct"], slope_window_seconds)
    temp_slope = _rolling_slope(columns["cpu_temp_c"], slope_window_seconds)
    gpu_temp_slope = _rolling_slope(columns["gpu_temp_c"], slope_window_seconds)
    fan_slope = _rolling_slope(columns["cpu_fan_rpm"], slope_window_seconds)
    feature_map = {
        "cpu_usage": columns["cpu_load_pct"],
        "gpu_usage": columns["gpu_load_pct"],
        "cpu_temp": columns["cpu_temp_c"],
        "gpu_temp": columns["gpu_temp_c"],
        "fan_rpm": columns["cpu_fan_rpm"],
        "cpu_load_slope": load_slope,
        "cpu_temp_slope": temp_slope,
        "gpu_temp_slope": gpu_temp_slope,
        "fan_rpm_slope": fan_slope,
    }
    features = np.column_stack([feature_map[k] for k in FEATURE_NAMES])
    features[~present, :] = np.nan
    return TelemetryData(
        timestamps=timestamps,
        columns=columns,
        workload=workload,
        mode=mode,
        present=present,
        duplicates=duplicates,
        source_rows=len(rows),
        features=features,
    )
