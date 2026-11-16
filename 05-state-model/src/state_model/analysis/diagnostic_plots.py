from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from state_model.training.data import TelemetryData
from .plots import PURPLES, INK, theme, save


FEATURE_NAMES = (
    ("cpu_load_pct", "CPU load (%)"),
    ("gpu_load_pct", "GPU load (%)"),
    ("cpu_temp_c", "CPU temp (°C)"),
    ("gpu_temp_c", "GPU temp (°C)"),
    ("cpu_fan_rpm", "CPU fan (RPM)"),
    ("gpu_fan_rpm", "GPU fan (RPM)"),
)


def _heatmap(
    ax,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: tuple[str, ...],
    title: str,
    *,
    fmt: str = ".0%",
    max_value: float | None = None,
) -> None:
    arr = np.asarray(matrix, dtype=float)
    image = ax.imshow(
        arr,
        aspect="auto",
        cmap="Purples",
        vmin=0,
        vmax=max_value if max_value is not None else max(1e-8, arr.max()),
    )
    ax.set(
        xticks=range(len(xlabels)),
        xticklabels=xlabels,
        yticks=range(len(ylabels)),
        yticklabels=[v.replace("_", " ") for v in ylabels],
    )
    ax.set_title(title, loc="left", fontsize=12)
    ax.tick_params(axis="x", rotation=25, labelsize=9)
    for row in range(len(ylabels)):
        for col in range(len(xlabels)):
            number = arr[row, col]
            ax.text(
                col,
                row,
                format(number, fmt),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if number > image.norm.vmax * 0.68 else INK,
            )
    ax.grid(False)


def interpretation_plots(
    data: TelemetryData,
    labels: np.ndarray,
    states: tuple[str, ...],
    report: dict,
    out: Path,
) -> list[Path]:
    theme()
    t = data.seconds / 60
    written = []

    fig, axes = plt.subplots(
        6,
        1,
        sharex=True,
        figsize=(14, 11.2),
        height_ratios=[1.15, 1.15, 0.85, 0.34, 0.34, 0.9],
        layout="constrained",
    )
    for ax, fields, name, unit in (
        (axes[0], ("cpu_load_pct", "gpu_load_pct"), "Utilization", "%"),
        (axes[1], ("cpu_temp_c", "gpu_temp_c"), "Temperature", "°C"),
        (axes[2], ("cpu_fan_rpm", "gpu_fan_rpm"), "Fan speed", "RPM"),
    ):
        for j, field in enumerate(fields):
            ax.plot(
                t,
                data.columns[field],
                color=PURPLES[1 + 3 * j],
                lw=1.3,
                label="CPU" if j == 0 else "GPU",
            )
        ax.set(ylabel=unit)
        ax.set_title(name, loc="left", fontsize=11)
        ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper right")
        ax.grid(alpha=0.35)

    for ax, values, title in (
        (axes[3], data.workload, "Workload"),
        (axes[4], data.mode, "Performance mode"),
    ):
        categories = list(dict.fromkeys(values))
        codes = np.array([categories.index(value) for value in values])
        colors = plt.cm.Purples(np.linspace(0.18, 0.90, max(len(categories), 2)))[
            : len(categories)
        ]
        ax.imshow(
            codes[None, :],
            aspect="auto",
            interpolation="nearest",
            extent=(t[0], t[-1] + 1 / 60, 0, 1),
            cmap=ListedColormap(colors),
            vmin=-0.5,
            vmax=len(categories) - 0.5,
        )
        ax.set(yticks=[], ylabel=title if title == "Workload" else "Mode")
        ax.legend(
            handles=[
                Patch(color=colors[i], label=category)
                for i, category in enumerate(categories)
            ],
            ncol=len(categories),
            loc="lower left",
            fontsize=7,
            framealpha=0.85,
            bbox_to_anchor=(0, 0.88),
        )
        ax.grid(False)

    axes[5].step(t, labels, where="post", lw=1.25, color=PURPLES[2])
    axes[5].set(
        yticks=range(len(states)),
        yticklabels=[name.replace("_", " ") for name in states],
        xlabel="Elapsed time (minutes)",
        ylabel="Online MAP",
    )
    axes[5].grid(alpha=0.35)
    written.append(save(fig, out, "11_state_interpretation_timeline.png"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(
        "State labels vs recorded context  |  associations, not ground truth",
        fontsize=16,
        weight="bold",
    )
    for row, (key, title) in enumerate(
        (("workload", "Workload"), ("performance_mode", "Performance mode"))
    ):
        info = report[key]
        _heatmap(
            axes[row, 0],
            info["category_given_state"],
            info["categories"],
            states,
            f"P({title.lower()} | inferred state)",
            max_value=1.0,
        )
        _heatmap(
            axes[row, 1],
            info["state_given_category"],
            info["categories"],
            states,
            f"P(inferred state | {title.lower()})",
            max_value=1.0,
        )
    written.append(save(fig, out, "12_state_context.png"))

    # Raw-scale medians, plus relative feature deviations (recording-wide z-scores).
    summaries = report["measurements"]
    raw = np.array(
        [
            [
                summaries[key][state]["median"] if summaries[key][state] else np.nan
                for key, _ in FEATURE_NAMES
            ]
            for state in states
        ]
    )
    valid = data.present & np.isfinite(data.features).all(axis=1)
    mean = np.mean(data.features[valid], axis=0)
    std = np.std(data.features[valid], axis=0)
    std = np.where(std > 0, std, 1.0)
    relative = np.array(
        [
            [
                (summaries[name][state]["mean"] - mean[j]) / std[j]
                if summaries[name][state]
                else np.nan
                for j, name in enumerate(data.feature_names)
            ]
            for state in states
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.4), layout="constrained")
    lo, hi = np.nanmin(raw, axis=0), np.nanmax(raw, axis=0)
    normalized = (raw - lo) / np.where(hi > lo, hi - lo, 1.0)
    image = axes[0].imshow(normalized, cmap="Purples", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title(
        "Median observations | color scaled within each column", loc="left"
    )
    axes[0].set(
        yticks=range(len(states)),
        yticklabels=[v.replace("_", " ") for v in states],
        xticks=range(len(FEATURE_NAMES)),
        xticklabels=[title for _, title in FEATURE_NAMES],
    )
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    for i in range(len(states)):
        for j in range(len(FEATURE_NAMES)):
            if np.isfinite(raw[i, j]):
                axes[0].text(
                    j,
                    i,
                    f"{raw[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] > 0.7 else INK,
                    fontsize=9,
                )
    axes[0].grid(False)

    cmap = LinearSegmentedColormap.from_list(
        "violet_diverging", ["#5A3194", "#FDFBFF", "#AA52BA"]
    )
    radius = max(1.0, float(np.nanmax(np.abs(relative))))
    image = axes[1].imshow(
        relative, cmap=cmap, aspect="auto", vmin=-radius, vmax=radius
    )
    axes[1].set_title("Mean difference from recording average (SD units)", loc="left")
    axes[1].set(
        yticks=range(len(states)),
        yticklabels=[v.replace("_", " ") for v in states],
        xticks=range(len(data.feature_names)),
        xticklabels=[name.replace("_", "\n") for name in data.feature_names],
    )
    axes[1].tick_params(axis="x", labelsize=8)
    for i in range(len(states)):
        for j in range(len(data.feature_names)):
            if np.isfinite(relative[i, j]):
                axes[1].text(
                    j,
                    i,
                    f"{relative[i, j]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(relative[i, j]) > radius * 0.65 else INK,
                )
    axes[1].grid(False)
    fig.colorbar(image, ax=axes[1], shrink=0.72, label="Standard deviations from mean")
    written.append(save(fig, out, "13_state_feature_profiles.png"))
    return written


def dynamics_plots(
    states: tuple[str, ...],
    mask: np.ndarray,
    map_report: dict,
    viterbi_report: dict,
    out: Path,
) -> list[Path]:
    theme()
    written = []
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), layout="constrained")
    fig.text(
        0.5,
        0.015,
        "! marks a reported MAP switch forbidden by the latent-state transition mask",
        ha="center",
        fontsize=9,
        color="#78618A",
    )
    for ax, report, title in (
        (axes[0], map_report, "Online MAP"),
        (axes[1], viterbi_report, "Offline Viterbi"),
    ):
        counts = np.array(report["observed_state_changes"], dtype=int)
        image = ax.imshow(
            np.ma.masked_where(np.eye(len(states), dtype=bool), counts),
            cmap="Purples",
            vmin=0,
            vmax=max(1, int(counts.max())),
        )
        ax.set(
            xticks=range(len(states)),
            yticks=range(len(states)),
            xticklabels=[s.replace("_", "\n") for s in states],
            yticklabels=[s.replace("_", " ") for s in states],
            xlabel="To",
            ylabel="From",
        )
        ax.set_title(
            f"{title}: {report['state_change_count']} observed switches", loc="left"
        )
        ax.tick_params(labelsize=8)
        for i in range(len(states)):
            for j in range(len(states)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="#91849C")
                else:
                    count = int(counts[i, j])
                    warning = "!" if count and not mask[i, j] else ""
                    ax.text(
                        j,
                        i,
                        f"{count}{warning}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if count > image.norm.vmax * 0.67 else INK,
                    )
        ax.grid(False)
    written.append(save(fig, out, "14_observed_transitions.png"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.85, bottom=0.19, wspace=0.045)
    for ax, report, title in (
        (axes[0], map_report, "Online MAP"),
        (axes[1], viterbi_report, "Offline Viterbi"),
    ):
        for i, state in enumerate(states):
            complete = [
                float(r["seconds"])
                for r in report["runs"]
                if r["state"] == state and r["completed"]
            ]
            expected = report["dwell"][state]["hmm_expected_seconds"]
            if complete:
                offsets = np.linspace(-0.15, 0.15, len(complete))
                ax.scatter(
                    complete,
                    i + offsets,
                    s=27,
                    alpha=0.55,
                    color=PURPLES[i],
                    edgecolors="none",
                    zorder=3,
                )
                ax.scatter(
                    [np.mean(complete)],
                    [i],
                    s=75,
                    marker="D",
                    color=PURPLES[i],
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=4,
                )
            if expected is not None:
                ax.scatter(
                    [expected], [i], s=120, marker="|", lw=2.3, color=INK, zorder=5
                )
            ax.text(
                0.98,
                i,
                f"n={len(complete)}",
                ha="right",
                va="center",
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                color="#685477",
            )
        ax.set(
            xscale="log",
            xlabel="Run duration (seconds, log scale)",
            yticks=range(len(states)),
            yticklabels=[state.replace("_", " ") for state in states],
        )
        ax.set_title(title, loc="left")
        ax.grid(axis="x")
    axes[0].invert_yaxis()
    fig.text(
        0.5,
        0.035,
        "Dots: completed runs     ◆: observed mean     |: HMM expected dwell     n: completed-run count",
        ha="center",
        fontsize=9,
        color="#78618A",
    )
    written.append(save(fig, out, "15_observed_dwell.png"))
    return written
