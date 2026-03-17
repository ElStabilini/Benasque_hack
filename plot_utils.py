import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

try:
    plt.style.use(["science", "notebook"])
except Exception:
    plt.style.use("default")

optimizers = ["COBYLA", "SLSQP", "NFT"]

start_color = "#2596be"
end_color = "#d80780"
cmap = mcolors.LinearSegmentedColormap.from_list(
    "optimizer_palette", [start_color, end_color]
)

palette = [mcolors.to_hex(cmap(x)) for x in np.linspace(0, 1, len(optimizers))]
colors = dict(zip(optimizers, palette))

hw_styles = {
    "Ion Trap": {"ls": "-"},
    "Superconducting": {"ls": "-"},
}

TITLE_PAD = 16
LABEL_PAD = 10


def extract_raw_zne(data, optimizer="COBYLA"):
    return data[optimizer]["hist"][-1], data[optimizer]["zne"]


def build_bar_data(ion_data, sc_data, optimizer="COBYLA"):
    ion_raw, ion_zne = extract_raw_zne(ion_data, optimizer)
    sc_raw, sc_zne = extract_raw_zne(sc_data, optimizer)

    labels = ["Ion Raw", "Ion ZNE", "SC Raw", "SC ZNE"]
    values = [ion_raw, ion_zne, sc_raw, sc_zne]

    return labels, values


def plot_vqe_convergence(history_data, ideal_hist, fci_energy, hardware_label):
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(
        ideal_hist,
        color="black",
        lw=3,
        label="Noise-Free Baseline",
    )

    style = hw_styles.get(hardware_label, {"ls": "-"})

    for opt in optimizers:
        if opt not in history_data:
            continue

        hist = history_data[opt]["hist"]
        zne = history_data[opt]["zne"]

        ax.plot(
            hist,
            color=colors[opt],
            lw=2.2,
            label=f"{opt} Raw",
            **style,
        )

        ax.scatter(
            len(hist) - 1,
            zne,
            marker="*",
            s=220,
            color=colors[opt],
            edgecolor="black",
            zorder=10,
        )

    ax.axhline(
        fci_energy,
        color=mcolors.to_hex(cmap(1.0)),
        linestyle="-",
        lw=2,
        label="Exact FCI",
    )

    ax.set_title(
        f"VQE Convergence - {hardware_label}",
        fontsize=16,
        pad=TITLE_PAD,
    )
    ax.set_xlabel("Total Quantum Evaluations", fontsize=13, labelpad=LABEL_PAD)
    ax.set_ylabel("Energy (Ha)", fontsize=13, labelpad=LABEL_PAD)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            markersize=12,
            label="ZNE Value",
        )
    )
    legend_labels.append("ZNE Value")

    ax.legend(legend_handles, legend_labels, fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_residual_error(ion_data, sc_data, fci_energy, optimizer="COBYLA"):
    labels, energies = build_bar_data(ion_data, sc_data, optimizer)

    errors = [abs(E - fci_energy) for E in energies]

    fig, ax = plt.subplots(figsize=(9, 6))

    bar_colors = [mcolors.to_hex(cmap(x)) for x in np.linspace(0.2, 0.8, len(labels))]
    ax.bar(labels, errors, linewidth=1.3, color=bar_colors)

    ax.set_ylabel(r"$|E_{VQE}-E_{FCI}|$ (Ha)", fontsize=13, labelpad=LABEL_PAD)
    ax.set_title("Residual Error Comparison", fontsize=15, pad=TITLE_PAD)

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_energy_expectations(ion_data, sc_data, fci_energy, optimizer="COBYLA"):
    labels, energies = build_bar_data(ion_data, sc_data, optimizer)

    fig, ax = plt.subplots(figsize=(9, 6))

    bar_colors = [mcolors.to_hex(cmap(x)) for x in np.linspace(0.2, 0.8, len(labels))]
    ax.bar(labels, energies, color=bar_colors)

    ax.axhline(
        fci_energy,
        color=mcolors.to_hex(cmap(1.0)),
        linestyle="-",
        lw=2,
        label="Exact FCI",
    )

    ax.set_ylabel("Energy (Ha)", fontsize=13, labelpad=LABEL_PAD)
    ax.set_title("Energy Expectations vs Exact", fontsize=15, pad=TITLE_PAD)

    ax.invert_yaxis()

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_depth_comparison(depths):
    labels = ["Ion Trap\n(Unrestricted)", "Superconducting\n(Topological Constraint)"]
    values = [depths["Ion"], depths["SC"]]
    bar_colors = [mcolors.to_hex(cmap(0.25)), mcolors.to_hex(cmap(0.75))]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color=bar_colors, edgecolor="black", linewidth=1.5)

    ax.set_ylabel(
        "Transpiled Instruction Depth ($N_{gates}$)",
        fontsize=12,
        labelpad=LABEL_PAD,
    )
    ax.set_title(
        "Complexity Penalty: Topological Mapping Overhead",
        fontsize=14,
        fontweight="bold",
        pad=TITLE_PAD,
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2,
            f"{int(bar.get_height())}",
            ha="center",
            fontweight="bold",
            fontsize=12,
        )

    ax.grid(axis="y", alpha=0.2, linestyle="-")
    plt.tight_layout()
    plt.show()
