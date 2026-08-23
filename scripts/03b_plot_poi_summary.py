"""Render POI heatmaps from the already-computed summary CSV.

Matplotlib is used deliberately here: static export must not depend on a
headless browser, and the numerical POI analysis is never rerun for plotting.
"""
import matplotlib.pyplot as plt
import pandas as pd

from analysis_pipeline import SUMMARY_DIR


def main():
    data = pd.read_csv(SUMMARY_DIR / "poi_master.csv")
    for metric in ["poi_entropy_w", "poi_spec_w", "poi_simpson_w", "poi_entropy_z"]:
        pivot = data.pivot(index="city", columns="algorithm", values=metric)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        image = ax.imshow(
            pivot.values,
            aspect="auto",
            cmap="RdYlGn_r" if metric == "poi_entropy_w" else "RdYlGn",
        )
        ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right", fontsize=12)
        ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=12)
        ax.set_xlabel("Algorithm", fontsize=12)
        ax.set_ylabel("City", fontsize=12)
        ax.grid(False)
        ax.tick_params(which="both", length=0)
        for row in range(len(pivot.index)):
            for col in range(len(pivot.columns)):
                value = pivot.iat[row, col]
                ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                        fontsize=16, clip_on=True)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
        colorbar.ax.tick_params(labelsize=11)
        for suffix in ("pdf", "png"):
            fig.savefig(SUMMARY_DIR / f"poi_heatmap_{metric}.{suffix}", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
