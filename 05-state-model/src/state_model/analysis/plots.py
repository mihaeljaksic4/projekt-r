from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
import numpy as np

from state_model.filtering import ModelArtifact
from state_model.training.data import TelemetryData

PALETTE = {
    "ink": "#302A43",
    "muted": "#70677F",
    "paper": "#FCF9FF",
    "panel": "#FFFFFF",
    "grid": "#EAE3F2",
    "lavender": "#8D79C6",
    "violet": "#B095D8",
    "lilac": "#D2BDEA",
    "orchid": "#DFA6CF",
    "rose": "#E9B7C9",
    "mint": "#83C8BD",
    "sky": "#94BFE4",
    "amber": "#E6C681",
    "reset": "#C9689C",
}

COLORS = (
    PALETTE["lavender"],
    PALETTE["mint"],
    PALETTE["orchid"],
    PALETTE["sky"],
    PALETTE["amber"],
    PALETTE["violet"],
)
PURPLES = COLORS
INK, ACCENT, PALE = PALETTE["ink"], PALETTE["lavender"], PALETTE["lilac"]
POSTER_CMAP = LinearSegmentedColormap.from_list(
    "state_model_pastel",
    [PALETTE["paper"], PALETTE["lilac"], PALETTE["lavender"], INK],
)


def theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 17,
            "axes.titlelocation": "left",
            "axes.labelsize": 12.5,
            "axes.prop_cycle": cycler(color=COLORS),
            "text.color": INK,
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Avenir Next", "Inter", "DejaVu Sans"],
            "font.size": 11.5,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "patch.edgecolor": "none",
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.fancybox": True,
            "legend.framealpha": 0.94,
            "legend.facecolor": PALETTE["panel"],
            "legend.edgecolor": PALETTE["grid"],
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.9,
            "grid.alpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.16,
        }
    )


def save(fig, out: Path, name: str) -> Path:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def workloads(ax, data: TelemetryData) -> None:
    change = (
        [0]
        + [
            i
            for i in range(1, len(data.workload))
            if data.workload[i] != data.workload[i - 1]
        ]
        + [len(data.workload)]
    )
    for lo, hi in zip(change[:-1], change[1:]):
        label = data.workload[lo]
        if label in ("Cinebench", "FurMark"):
            ax.axvspan(
                data.seconds[lo] / 60,
                data.seconds[hi - 1] / 60,
                color=ACCENT if label == "Cinebench" else PALETTE["mint"],
                alpha=0.10,
            )


def data_plots(data: TelemetryData, out: Path) -> list[Path]:
    theme()
    t = data.seconds / 60
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True, layout="constrained")
    fig.suptitle("Recorded telemetry", fontsize=19, weight="bold")
    panels = [
        ("CPU / GPU load", ("cpu_load_pct", "gpu_load_pct"), "Utilization (%)"),
        ("CPU / GPU temperature", ("cpu_temp_c", "gpu_temp_c"), "Temperature (°C)"),
        ("Fan speeds", ("cpu_fan_rpm", "gpu_fan_rpm"), "RPM"),
    ]
    for ax, (heading, fields, unit) in zip(axes, panels):
        workloads(ax, data)
        for i, name in enumerate(fields):
            if "temp" in name:
                ax.step(
                    t,
                    data.columns[name],
                    where="post",
                    lw=1.25,
                    color=COLORS[i],
                    label="CPU" if i == 0 else "GPU",
                )
            else:
                ax.plot(
                    t,
                    data.columns[name],
                    lw=1.25,
                    color=COLORS[i],
                    label="CPU" if i == 0 else "GPU",
                )
        ax.set_title(heading, loc="left")
        ax.set_ylabel(unit)
        ax.grid()
        ax.legend(loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel("Elapsed time (minutes)")
    first = save(fig, out, "01_telemetry.png")

    fig, ax = plt.subplots(figsize=(11, 3.6), layout="constrained")
    for j, name in enumerate(("cpu_temp_c", "gpu_temp_c")):
        x = data.columns[name]
        changes = (
            np.flatnonzero(np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (x[1:] != x[:-1]))
            + 1
        )
        ax.eventplot(
            t[changes], lineoffsets=j, linelengths=0.5, linewidths=1.2, colors=COLORS[j]
        )
    ax.set_yticks([0, 1], ["CPU", "GPU"])
    ax.set_xlabel("Elapsed time (minutes)")
    ax.set_title(
        "Temperature value changes | not necessarily sensor polling times", loc="left"
    )
    ax.grid(axis="x")
    second = save(fig, out, "02_sensor_changes.png")
    return [first, second]


def _gaussian_surfaces(model: ModelArtifact, size: int = 55):
    ix, iy = (
        model.feature_names.index("cpu_usage"),
        model.feature_names.index("cpu_temp"),
    )
    means = model.means * model.scaler_std + model.scaler_mean
    stds = np.sqrt(model.variances) * model.scaler_std
    surfaces = []
    for mu, sd in zip(means, stds):
        xx = np.linspace(
            max(0, mu[ix] - 3 * sd[ix]), min(105, mu[ix] + 3 * sd[ix]), size
        )
        yy = np.linspace(
            max(25, mu[iy] - 3 * sd[iy]), min(110, mu[iy] + 3 * sd[iy]), size
        )
        x, y = np.meshgrid(xx, yy)
        z = np.exp(
            -0.5 * ((x - mu[ix]) / sd[ix]) ** 2 - 0.5 * ((y - mu[iy]) / sd[iy]) ** 2
        ) / (2 * np.pi * sd[ix] * sd[iy])
        surfaces.append((x, y, z))
    return surfaces


def _surface_axes(ax, height: float) -> None:
    ax.set(xlabel="CPU load (%)", ylabel="CPU temp (°C)", zlabel="Density")
    ax.set(xlim=(0, 105), ylim=(25, 110), zlim=(0, height * 1.12))
    ax.view_init(elev=28, azim=-62)
    ax.tick_params(labelsize=8, pad=1)
    ax.zaxis.set_major_locator(plt.MaxNLocator(3))
    ax.set_facecolor("white")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.line.set_color(PALETTE["grid"])


def model_plots(
    data: TelemetryData,
    model: ModelArtifact,
    posterior: np.ndarray,
    path: np.ndarray,
    sage: dict,
    history: list[float] | None,
    out: Path,
) -> list[Path]:
    theme()
    written = []
    names = model.states
    t = data.seconds / 60
    map_path = posterior.argmax(axis=1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12.5, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
        layout="constrained",
    )
    fig.suptitle(
        "Filtered beliefs and globally decoded path", fontsize=18, weight="bold"
    )
    workloads(axes[0], data)
    axes[0].stackplot(t, *posterior.T, colors=PURPLES, labels=names, alpha=0.9)
    axes[0].set(ylabel="Filtered probability", ylim=(0, 1))
    axes[0].legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        fontsize=8,
        frameon=False,
    )
    axes[1].step(
        t, map_path, where="post", color=PALETTE["lavender"], lw=1.5, label="Online MAP"
    )
    axes[1].step(
        t,
        path,
        where="post",
        color=PALETTE["reset"],
        lw=1.25,
        alpha=0.9,
        label="Offline Viterbi",
    )
    axes[1].set(
        yticks=range(len(names)),
        yticklabels=[n.replace("_", " ").title() for n in names],
        xlabel="Elapsed time (minutes)",
        ylabel="State index",
    )
    axes[1].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.grid()
    written.append(save(fig, out, "03_states_and_viterbi.png"))

    fig, ax = plt.subplots(figsize=(8.5, 7), layout="constrained")
    transition = np.where(model.mask, model.transitions, np.nan)
    cmap = POSTER_CMAP.copy()
    cmap.set_bad(PALETTE["grid"])
    image = ax.imshow(transition, cmap=cmap, vmin=0, vmax=1)
    ax.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=[n.replace("_", "\n") for n in names],
        yticklabels=[n.replace("_", " ") for n in names],
        xlabel="Destination state",
        ylabel="Origin state",
    )
    ax.set_title("Transition probabilities | × = forbidden", pad=17)
    for i in range(len(names)):
        for j in range(len(names)):
            v = model.transitions[i, j]
            ax.text(
                j,
                i,
                (f"{v:.4f}" if v >= 0.0001 else "<.0001") if model.mask[i, j] else "×",
                ha="center",
                va="center",
                color="white" if v > 0.55 else INK,
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, shrink=0.75, label="Probability")
    written.append(save(fig, out, "04_transitions.png"))

    surfaces = _gaussian_surfaces(model)
    height = max(float(np.max(z)) for _, _, z in surfaces)
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    for i, (x, y, z) in enumerate(surfaces):
        ax.plot_surface(
            x,
            y,
            z,
            color=PURPLES[i],
            alpha=0.6,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        ax.contour(
            x, y, z, zdir="z", offset=0, levels=4, colors=[PURPLES[i]], linewidths=0.7
        )
    _surface_axes(ax, height)
    ax.set_title("Six Gaussian emission marginals | CPU load × CPU temperature", pad=22)
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], color=PURPLES[i], lw=4, label=names[i])
            for i in range(len(names))
        ],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    written.append(save(fig, out, "05_gaussians_3d.png"))

    fig = plt.figure(figsize=(15, 9), layout="constrained")
    fig.suptitle(
        "Individual state emissions | 2D marginals; independent density scales",
        fontsize=17,
        weight="bold",
    )
    for i, (x, y, z) in enumerate(surfaces):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        ax.plot_surface(
            x,
            y,
            z,
            color=PURPLES[i],
            alpha=0.92,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        ax.contour(
            x, y, z, zdir="z", offset=0, levels=5, colors=[PURPLES[i]], linewidths=0.9
        )
        _surface_axes(ax, float(np.max(z)))
        ax.set_title(names[i].replace("_", " ").title(), pad=7, fontsize=12)
    written.append(save(fig, out, "06_gaussians_per_state_3d.png"))

    distances = np.array(sage["emissions"]["bhattacharyya_distance"])
    fig, ax = plt.subplots(figsize=(8.6, 7), layout="constrained")
    image = ax.imshow(distances, cmap=POSTER_CMAP, vmin=0)
    ax.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=[n.replace("_", "\n") for n in names],
        yticklabels=[n.replace("_", " ") for n in names],
    )
    ax.set_title("Pairwise Gaussian separation | Bhattacharyya distance", pad=15)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{distances[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if distances[i, j] > np.max(distances) * 0.65 else INK,
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Distance (larger = more separated)")
    written.append(save(fig, out, "07_emission_separation.png"))

    chain = sage["markov_chain"]
    probs = [chain["stationary_probabilities"][n] for n in names]
    dwell = [chain["expected_dwell_seconds"][n] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    positions = np.arange(len(names))
    axes[0].barh(positions, probs, color=PURPLES)
    axes[0].set(
        xlabel="Stationary probability",
        yticks=positions,
        yticklabels=[n.replace("_", " ") for n in names],
    )
    axes[1].barh(positions, [v or 0 for v in dwell], color=PURPLES)
    axes[1].set(
        xlabel="Expected dwell (seconds)",
        yticks=positions,
        yticklabels=[n.replace("_", " ") for n in names],
    )
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x")
    fig.suptitle("Long-run behavior inferred from the transition matrix", fontsize=16)
    written.append(save(fig, out, "08_stationary_and_dwell.png"))

    eig = [complex(z["real"], z["imag"]) for z in chain["eigenvalues"]]
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    full.add_patch(plt.Circle((0, 0), 1.0, color=PALE, fill=False, ls="--", lw=1.5))
    for ax in (full, zoom):
        for index, value in enumerate(eig):
            ax.scatter(
                value.real,
                value.imag,
                s=85,
                color=PURPLES[index],
                edgecolors="white",
                linewidths=1.0,
                zorder=4,
            )
        ax.axhline(0, color=PALETTE["grid"], lw=1)
        ax.axvline(0, color=PALETTE["grid"], lw=1)
        ax.set(xlabel="Real part", ylabel="Imaginary part")
        ax.grid()
    full.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08))
    full.set_aspect("equal")
    full.set_title("Unit circle and spectrum")
    real = np.array([v.real for v in eig])
    imaginary = np.array([v.imag for v in eig])
    margin_x = max(0.003, float(np.ptp(real)) * 0.10)
    margin_y = max(0.001, float(np.max(np.abs(imaginary))) * 0.55)
    zoom.set(
        xlim=(float(real.min()) - margin_x, 1 + margin_x),
        ylim=(float(imaginary.min()) - margin_y, float(imaginary.max()) + margin_y),
    )
    zoom.set_title(f"Close-up | spectral gap {chain['spectral_gap']:.4f}")
    for index, value in enumerate(eig):
        zoom.annotate(
            f"λ{index + 1}",
            (value.real, value.imag),
            xytext=(4, 8 if index % 2 == 0 else -15),
            textcoords="offset points",
            fontsize=9,
        )
    written.append(save(fig, out, "09_transition_spectrum.png"))

    if history:
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        ax.plot(history, lw=2.0, marker="o", markersize=3.5, color=ACCENT)
        ax.set(xlabel="EM iteration", ylabel="Log likelihood")
        ax.set_title("Baum–Welch observed-data log likelihood", loc="left")
        ax.grid()
        written.append(save(fig, out, "10_em_convergence.png"))
    written.extend(sage_plots(model, sage, out))
    return written


def sage_plots(model: ModelArtifact, report: dict, out: Path) -> list[Path]:
    """Interpretable plots for quantities calculated by SageMath."""
    names = model.states
    chain = report["markov_chain"]
    written = []
    k = len(names)
    labels = [name.replace("_", "\n") for name in names]

    # Long-run off-diagonal events per hour, under the transition matrix alone.
    flow = np.asarray(chain["stationary_offdiagonal_flow_per_hour"])
    fig, ax = plt.subplots(figsize=(9, 7.3), layout="constrained")
    masked = np.eye(k, dtype=bool) | ~model.mask
    display = np.ma.masked_where(masked, flow)
    cmap = POSTER_CMAP.copy()
    cmap.set_bad(PALETTE["grid"])
    image = ax.imshow(display, cmap=cmap, vmin=0)
    ax.set(
        xticks=range(k),
        xticklabels=labels,
        yticks=range(k),
        yticklabels=[n.replace("_", " ") for n in names],
        xlabel="Destination",
        ylabel="Origin",
    )
    ax.set_title("Stationary off-diagonal transitions | expected per hour")
    for i in range(k):
        for j in range(k):
            text = (
                "—"
                if i == j
                else (
                    "×"
                    if not model.mask[i, j]
                    else (f"{flow[i, j]:.2f}" if flow[i, j] >= 0.005 else "<.01")
                )
            )
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if flow[i, j] > float(np.max(flow)) * 0.55 else INK,
            )
    fig.colorbar(
        image, ax=ax, shrink=0.8, label="Transitions/hour under stationary chain"
    )
    written.append(save(fig, out, "16_stationary_flow.png"))

    convergence = chain["worst_case_convergence"]
    fig, ax = plt.subplots(figsize=(9.7, 4.6), layout="constrained")
    ax.plot(
        [v["seconds"] / 60 for v in convergence],
        [v["worst_total_variation"] for v in convergence],
        marker="o",
        lw=2,
        color=ACCENT,
    )
    ax.set(
        xlabel="Elapsed time without new observations (minutes)",
        ylabel="Worst-case total-variation distance",
        ylim=(0, 1),
    )
    ax.set_title("Convergence to stationarity from each initial state")
    ax.grid()
    written.append(save(fig, out, "17_convergence_to_stationarity.png"))

    explanations = report["emissions"]["adjacent_pair_feature_contributions"]
    pair_names = [
        " / ".join(item["states"]).replace("THERMAL_", "T_") for item in explanations
    ]
    feature_names = list(model.feature_names)
    proportions = np.array(
        [
            [
                item["contributions"][f] / item["distance"] if item["distance"] else 0
                for f in feature_names
            ]
            for item in explanations
        ]
    )
    fig, ax = plt.subplots(
        figsize=(12, max(6, 0.48 * len(pair_names))), layout="constrained"
    )
    image = ax.imshow(proportions, aspect="auto", cmap=POSTER_CMAP, vmin=0, vmax=1)
    ax.set(
        yticks=range(len(pair_names)),
        yticklabels=pair_names,
        xticks=range(len(feature_names)),
        xticklabels=[f.replace("_", "\n") for f in feature_names],
    )
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Features driving emission separation | permitted state neighbors")
    for i in range(len(pair_names)):
        for j in range(len(feature_names)):
            v = proportions[i, j]
            if v >= 0.08:
                ax.text(
                    j,
                    i,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v >= 0.58 else INK,
                )
    fig.colorbar(image, ax=ax, shrink=0.7, label="Share of Bhattacharyya distance")
    written.append(save(fig, out, "18_emission_separation_drivers.png"))

    hits = np.asarray(chain["expected_hitting_seconds"]) / 60
    fig, ax = plt.subplots(figsize=(9, 7.3), layout="constrained")
    image = ax.imshow(hits, cmap=POSTER_CMAP, vmin=0)
    ax.set(
        xticks=range(k),
        xticklabels=labels,
        yticks=range(k),
        yticklabels=[n.replace("_", " ") for n in names],
        xlabel="Target state",
        ylabel="Starting state",
    )
    ax.set_title("Mean first-passage times | transition-only (minutes)")
    for i in range(k):
        for j in range(k):
            ax.text(
                j,
                i,
                f"{hits[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if hits[i, j] > float(np.max(hits)) * 0.58 else INK,
            )
    fig.colorbar(image, ax=ax, shrink=0.8, label="Expected minutes")
    written.append(save(fig, out, "19_first_passage_times.png"))

    modes = chain["slow_modes"]
    if modes:
        amplitudes = np.array(
            [[mode["amplitude"][name] for name in names] for mode in modes]
        )
        fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
        image = ax.imshow(amplitudes, cmap=POSTER_CMAP, vmin=0, vmax=1, aspect="auto")
        mode_labels = [
            f"λ = {v['eigenvalue']['real']:.4f} "
            f"{'+' if v['eigenvalue']['imag'] >= 0 else '-'} "
            f"{abs(v['eigenvalue']['imag']):.4f}i"
            for v in modes
        ]
        ax.set(
            xticks=range(k),
            xticklabels=labels,
            yticks=range(len(modes)),
            yticklabels=mode_labels,
        )
        ax.set_title("Slow probability modes | left eigenvector magnitudes")
        ax.grid(False)
        for i in range(len(modes)):
            for j in range(k):
                value = amplitudes[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.63 else INK,
                    fontsize=9,
                )
        fig.colorbar(
            image, ax=ax, shrink=0.76, label="Magnitude relative to largest component"
        )
        written.append(save(fig, out, "21_slow_modes.png"))

    passage = chain["first_passage"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.1), layout="constrained", sharey=True)
    for ax, key, title, from_states in (
        (
            axes[0],
            "thermal",
            "First thermal state",
            ["IDLE", "LOAD_RISING", "SUSTAINED_LOAD", "COOLING"],
        ),
        (
            axes[1],
            "idle",
            "Return to IDLE",
            [
                "LOAD_RISING",
                "SUSTAINED_LOAD",
                "THERMAL_RESPONSE",
                "THERMAL_STEADY",
                "COOLING",
            ],
        ),
    ):
        item = passage[key]
        x = np.asarray(item["horizons_seconds"]) / 60
        for state in from_states:
            if state not in item["probability_by_start"]:
                continue
            i = names.index(state)
            ax.plot(
                x,
                item["probability_by_start"][state],
                marker="o",
                markersize=3.2,
                label=state.replace("_", " "),
                color=PURPLES[i],
            )
        ax.set(title=title, xlabel="Time horizon (minutes)", ylim=(-0.025, 1.025))
        ax.grid()
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("Probability of having reached target")
    fig.suptitle(
        "First-passage probabilities | transitions only", fontsize=16, weight="bold"
    )
    written.append(save(fig, out, "22_first_passage_probabilities.png"))

    committor = chain["thermal_before_idle_committor"]
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    values = [committor[name] for name in names]
    ax.bar(range(k), values, color=PURPLES)
    ax.set(
        xticks=range(k),
        xticklabels=labels,
        ylim=(0, 1.05),
        ylabel="P(reach thermal states before IDLE)",
    )
    ax.set_title("Committor | competing first-passage destinations")
    for i, value in enumerate(values):
        ax.text(i, min(1.01, value + 0.025), f"{value:.2f}", ha="center", fontsize=9)
    ax.grid(axis="y")
    written.append(save(fig, out, "23_thermal_committor.png"))

    from matplotlib.patches import FancyArrowPatch, Circle

    currents = np.asarray(chain["stationary_currents_per_hour"])
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    angles = np.pi / 2 - 2 * np.pi * np.arange(k) / k
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    edges = sorted(
        ((abs(currents[i, j]), i, j) for i in range(k) for j in range(i + 1, k)),
        reverse=True,
    )
    largest = edges[0][0] if edges else 1.0
    for strength, i, j in edges:
        if strength < max(0.001, largest * 0.015):
            continue
        src, dst = (i, j) if currents[i, j] > 0 else (j, i)
        arrow = FancyArrowPatch(
            points[src],
            points[dst],
            connectionstyle="arc3,rad=.17",
            arrowstyle="-|>",
            mutation_scale=12,
            shrinkA=27,
            shrinkB=27,
            linewidth=1.2 + 5 * strength / largest,
            color=PALETTE["lavender"],
            alpha=0.78,
        )
        ax.add_patch(arrow)
    for i, (x, y) in enumerate(points):
        ax.add_patch(Circle((x, y), 0.19, color=PURPLES[i], alpha=0.93, zorder=5))
        ax.text(
            x,
            y,
            names[i].replace("_", "\n"),
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            color=INK,
            zorder=6,
        )
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.35, 1.35))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Net stationary currents | direction and relative magnitude")
    fig.text(
        0.5,
        0.05,
        "Arrow width = |πᵢ Aᵢⱼ − πⱼ Aⱼᵢ| × 3600 / cadence",
        ha="center",
        fontsize=10,
        color=PALETTE["muted"],
    )
    written.append(save(fig, out, "24_stationary_currents.png"))

    sensitivity = chain["stationary_sensitivity"]
    if sensitivity:
        effects = np.array(
            [
                [item["delta_stationary_probability"][name] * 100 for name in names]
                for item in sensitivity
            ]
        )
        magnitude = max(0.01, float(np.max(np.abs(effects))))
        diverging = LinearSegmentedColormap.from_list(
            "mint_purple", [PALETTE["mint"], PALETTE["paper"], PALETTE["lavender"]]
        )
        fig, ax = plt.subplots(
            figsize=(12, max(6, 0.39 * len(sensitivity))), layout="constrained"
        )
        image = ax.imshow(
            effects, cmap=diverging, vmin=-magnitude, vmax=magnitude, aspect="auto"
        )
        ax.set(
            xticks=range(k),
            xticklabels=labels,
            yticks=range(len(sensitivity)),
            yticklabels=[f"{row['edge'][0]} → {row['edge'][1]}" for row in sensitivity],
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(False)
        ax.set_title(
            "Stationary sensitivity | increase edge by 0.0001 at self-loop expense"
        )
        for i in range(len(sensitivity)):
            for j in range(k):
                value = effects[i, j]
                if abs(value) >= 0.005:
                    ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(
            image,
            ax=ax,
            shrink=0.8,
            label="Estimated stationary change (percentage points)",
        )
        written.append(save(fig, out, "25_stationary_sensitivity.png"))

    return written
