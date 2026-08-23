"""Render Endpoint--L--P agreement and transfer-threshold sensitivity."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from analysis_pipeline import OUTPUT_ROOT

ALGORITHM_ORDER = ("Infomap", "LeidenRBC", "DCSBM", "GeoLP", "Spinglass", "DirLou")


COMPARISON_LABELS = {
    "endpoint__l_space": "Endpoint--L",
    "l_space__p_space": "L--P",
    "endpoint__p_space": "Endpoint--P",
}


def _agreement_heatmap(data, metric, out_dir):
    unlimited = data[
        data["comparison"].eq("l_space__p_space")
        | data["transfer_threshold_m"].isna()
    ].copy()
    unlimited["column"] = unlimited.apply(
        lambda row: f"{COMPARISON_LABELS[row['comparison']]}\n{row['algorithm']}", axis=1
    )
    present_algorithms = [a for a in ALGORITHM_ORDER if a in set(unlimited["algorithm"])]
    column_order = [
        f"{comparison}\n{algorithm}"
        for comparison in ("Endpoint--L", "L--P", "Endpoint--P")
        for algorithm in present_algorithms
    ]
    pivot = unlimited.pivot(index="city", columns="column", values=metric).reindex(columns=column_order)
    fig, ax = plt.subplots(figsize=(19, 5.8), constrained_layout=True)
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=11)
    ax.set_xlabel("Representation comparison and algorithm", fontsize=11)
    ax.set_ylabel("City", fontsize=11)
    ax.grid(False)
    ax.tick_params(which="both", length=0)
    for row in range(len(pivot.index)):
        for col in range(len(pivot.columns)):
            value = pivot.iat[row, col]
            if pd.notna(value):
                ax.text(
                    col, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value < 0.72 else "black", fontsize=8, clip_on=True,
                )
    colorbar = fig.colorbar(image, ax=ax, label=metric.upper(), shrink=0.88)
    colorbar.ax.tick_params(labelsize=10)
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"representation_{metric}.{suffix}", dpi=300)
    plt.close(fig)


def _threshold_plot(data, out_dir):
    subset = data[
        data["comparison"].isin(["endpoint__l_space", "endpoint__p_space"])
        & data["transfer_threshold_m"].notna()
    ]
    cities = list(dict.fromkeys(subset["city"]))
    palette = plt.get_cmap("tab10")
    city_colors = {city: palette(index) for index, city in enumerate(cities)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for axis, comparison in zip(axes, ("endpoint__l_space", "endpoint__p_space")):
        comparison_data = subset[subset["comparison"].eq(comparison)]
        for (city, algorithm), group in comparison_data.groupby(["city", "algorithm"]):
            group = group.sort_values("transfer_threshold_m")
            axis.plot(
                group["transfer_threshold_m"], group["nmi"], marker="o",
                color=city_colors[city], linewidth=1.35, markersize=3.8,
                linestyle="-" if algorithm == "Infomap" else "--",
            )
        axis.set_title(COMPARISON_LABELS[comparison], fontsize=12)
        axis.set_xlabel("Maximum transfer distance (m)", fontsize=11)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2, linewidth=0.5)
    axes[0].set_ylabel("NMI", fontsize=11)
    city_handles = [
        Line2D([0], [0], color=city_colors[city], marker="o", linewidth=1.5,
               markersize=4, label=city)
        for city in cities
    ]
    algorithm_handles = [
        Line2D([0], [0], color="0.25", linestyle=(0, (1 + i % 3, 1 + (i // 3))),
               linewidth=1.5, label=algorithm)
        for i, algorithm in enumerate(ALGORITHM_ORDER)
        if algorithm in set(subset["algorithm"])
    ]
    city_legend = fig.legend(
        handles=city_handles, loc="lower center", bbox_to_anchor=(0.5, 0.075),
        ncol=7, fontsize=8, frameon=False, handlelength=2.1, columnspacing=1.1,
    )
    fig.add_artist(city_legend)
    fig.legend(
        handles=algorithm_handles, loc="lower center", bbox_to_anchor=(0.5, 0.018),
        ncol=2, fontsize=9, frameon=False, handlelength=3.0,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(out_dir / f"representation_transfer_thresholds.{suffix}", dpi=300)
    plt.close(fig)


def main():
    out_dir = OUTPUT_ROOT / "representation_sensitivity_all_algorithms_output"
    data = pd.read_csv(out_dir / "representation_pairwise_summary.csv")
    _agreement_heatmap(data, "nmi", out_dir)
    _agreement_heatmap(data, "ari", out_dir)
    _threshold_plot(data, out_dir)


if __name__ == "__main__":
    main()
