from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from state_model.training.data import TelemetryData
from .plots import COLORS, PALETTE, POSTER_CMAP, PURPLES, INK, theme, save, workloads


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
        workloads(ax, data)
        for j, field in enumerate(fields):
            draw = ax.step if "temp" in field else ax.plot
            draw(
                t,
                data.columns[field],
                color=COLORS[j],
                lw=1.3,
                label="CPU" if j == 0 else "GPU",
                **({"where": "post"} if "temp" in field else {}),
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
        colors = [COLORS[i % len(COLORS)] for i in range(len(categories))]
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

    changes = np.r_[0, np.flatnonzero(np.diff(labels) != 0) + 1, len(labels)]
    for start, stop in zip(changes[:-1], changes[1:]):
        index = int(labels[start])
        right = t[stop] if stop < len(t) else t[-1] + 1 / 60
        axes[5].hlines(index, t[start], right, color=COLORS[index], lw=3.0)
        if stop < len(t):
            axes[5].plot(
                [t[stop], t[stop]],
                [index, int(labels[stop])],
                color=PALETTE["muted"],
                lw=0.8,
                alpha=0.55,
            )
    axes[5].set(
        yticks=range(len(states)),
        yticklabels=[name.replace("_", " ") for name in states],
        xlabel="Elapsed time (minutes)",
        ylabel="Guarded state",
    )
    axes[5].set_title(
        "Reported operational state (HMM + FSM guards)", loc="left", fontsize=11
    )
    axes[5].grid(axis="y", alpha=0.35)

    written.append(save(fig, out, "11_state_interpretation_timeline.png"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle(
        "Guarded state vs recorded context  |  associations, not ground truth",
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
            f"P({title.lower()} | guarded state)",
            max_value=1.0,
        )
        _heatmap(
            axes[row, 1],
            info["state_given_category"],
            info["categories"],
            states,
            f"P(guarded state | {title.lower()})",
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
    guarded_report: dict,
    map_report: dict,
    viterbi_report: dict,
    out: Path,
) -> list[Path]:
    theme()
    reports = (
        (guarded_report, "Guarded HMM"),
        (map_report, "Raw MAP (comparison)"),
        (viterbi_report, "Viterbi (comparison)"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), layout="constrained")
    maximum = max(1, *(int(np.max(r["observed_state_changes"])) for r, _ in reports))
    for ax, (report, title) in zip(axes, reports):
        counts = np.asarray(report["observed_state_changes"], dtype=int)
        image = ax.imshow(
            np.ma.masked_where(np.eye(len(states), dtype=bool), counts),
            cmap=POSTER_CMAP,
            vmin=0,
            vmax=maximum,
        )
        ax.set(
            xticks=range(len(states)),
            yticks=range(len(states)),
            xticklabels=[name.replace("_", "\n") for name in states],
            yticklabels=[name.replace("_", " ") for name in states],
            xlabel="To",
            ylabel="From",
        )
        ax.set_title(
            f"{title}: {report['state_change_count']} switches", loc="left", fontsize=12
        )
        ax.tick_params(labelsize=8)
        for i in range(len(states)):
            for j in range(len(states)):
                if i == j:
                    value = "—"
                elif counts[i, j] and not mask[i, j]:
                    value = f"{counts[i, j]}!"
                elif not mask[i, j]:
                    value = "×"
                else:
                    value = str(counts[i, j])
                ax.text(
                    j,
                    i,
                    value,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if counts[i, j] > maximum * 0.67 else INK,
                )
        ax.grid(False)
    fig.colorbar(image, ax=axes, shrink=0.72, label="Observed direct switches")
    fig.suptitle(
        "State transitions | guarded output and decoding comparisons", fontsize=16
    )
    fig.text(
        0.5,
        0.01,
        "× forbidden edge    ! observed switch forbidden by HMM mask (raw MAP only)",
        ha="center",
        fontsize=9,
        color=PALETTE["muted"],
    )
    first = save(fig, out, "14_observed_transitions.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.87, bottom=0.19, wspace=0.08)
    durations = [
        float(run["seconds"])
        for report, _ in reports
        for run in report["runs"]
        if run["completed"]
    ]
    durations += [
        float(report["dwell"][state]["hmm_expected_seconds"])
        for report, _ in reports
        for state in states
        if report["dwell"][state]["hmm_expected_seconds"] is not None
    ]
    duration_limits = (max(0.1, min(durations) / 1.5), max(durations) * 1.5)
    for ax, (report, title) in zip(axes, reports):
        for i, state in enumerate(states):
            complete = [
                float(r["seconds"])
                for r in report["runs"]
                if r["state"] == state and r["completed"]
            ]
            expected = report["dwell"][state]["hmm_expected_seconds"]
            if complete:
                offsets = np.linspace(-0.14, 0.14, len(complete))
                ax.scatter(
                    complete,
                    i + offsets,
                    s=27,
                    alpha=0.65,
                    color=COLORS[i],
                    edgecolors="none",
                    zorder=3,
                )
                ax.scatter(
                    [np.mean(complete)],
                    [i],
                    s=75,
                    marker="D",
                    color=COLORS[i],
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
                i + 0.28,
                f"n={len(complete)}",
                ha="right",
                va="center",
                transform=ax.get_yaxis_transform(),
                fontsize=8,
                color=PALETTE["muted"],
            )
        ax.set(
            xscale="log",
            xlim=duration_limits,
            xlabel="Duration (s, log scale)",
            yticks=range(len(states)),
            yticklabels=[state.replace("_", " ") for state in states],
        )
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="x")
    axes[0].invert_yaxis()
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.suptitle("Observed state residence times | guarded output first", fontsize=16)
    fig.text(
        0.5,
        0.04,
        "Dots: complete runs    ◆: empirical mean    |: HMM expected dwell    n: complete-run count",
        ha="center",
        fontsize=9,
        color=PALETTE["muted"],
    )
    second = save(fig, out, "15_observed_dwell.png")
    return [first, second]


def guarded_state_plot(
    data: TelemetryData,
    posterior: np.ndarray,
    viterbi: np.ndarray,
    guarded: np.ndarray,
    states: tuple[str, ...],
    out: Path,
) -> Path:
    """Compare hard labels from the same telemetry without conflating the posterior."""
    theme()
    t = data.seconds / 60
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True, height_ratios=[3, 1], layout="constrained"
    )
    for values, label, color, offset in (
        (posterior.argmax(axis=1), "HMM MAP", PURPLES[0], -0.10),
        (guarded, "HMM + FSM guards", PURPLES[1], 0),
        (viterbi, "Offline Viterbi", PURPLES[5], 0.10),
    ):
        axes[0].step(
            t,
            values + offset,
            where="post",
            lw=1.25,
            color=color,
            label=label,
            alpha=0.90,
        )
    axes[0].set(
        yticks=range(len(states)),
        yticklabels=[s.replace("_", " ") for s in states],
        ylabel="Reported state",
    )
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[0].grid(axis="y", alpha=0.35)
    different = guarded != posterior.argmax(axis=1)
    axes[1].fill_between(
        t, different.astype(float), step="post", color=PURPLES[2], alpha=0.55
    )
    axes[1].set(
        ylim=(0, 1.1),
        yticks=[0, 1],
        yticklabels=["same", "different"],
        xlabel="Elapsed time (minutes)",
        ylabel="FSM vs MAP",
    )
    axes[1].grid(axis="y")
    fig.suptitle(
        "HMM posterior → guarded operational state", fontsize=17, weight="bold"
    )
    return save(fig, out, "20_guarded_fsm_timeline.png")
