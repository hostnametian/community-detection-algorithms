"""Export result summaries and diagnostic figures from archived outputs."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import analysis_pipeline as pipeline


def _saved_community_loader(original_get_community):
    def load(city_cfg, algorithm, **algorithm_kwargs):
        if algorithm_kwargs:
            return original_get_community(city_cfg, algorithm, **algorithm_kwargs)

        run_root = pipeline.RESULTS_DIR / city_cfg["city"] / algorithm
        candidates = []
        for config_path in run_root.glob("*/config.json"):
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            if config.get("script_version") == pipeline.PIPELINE_VERSION:
                candidates.append((config_path.parent.name, config_path.parent, config))
        if not candidates:
            return original_get_community(city_cfg, algorithm)

        _, run_dir, config = max(candidates, key=lambda item: item[0])
        community = {}
        with open(run_dir / "communities.csv", "r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                node_text = row["node_id"]
                try:
                    node_id = int(node_text)
                except ValueError:
                    node_id = node_text
                community[node_id] = int(row["community_id"])

        metadata_keys = [
            "algorithm_parameters", "directed_used", "weight_used",
            "spatial_prior", "algorithm_group", "note",
        ]
        metadata = {key: config[key] for key in metadata_keys if key in config}
        return community, metadata

    return load


def _original_figures():
    """Export static spatial and community-level panels from saved partitions."""
    original_save_result = pipeline.save_result
    pipeline.save_result = lambda *args, **kwargs: None
    try:
        pipeline.run_batch(
            cities=[city["city"] for city in pipeline.CITIES],
            algorithms=pipeline.ALGORITHMS,
            show_base=False,
            show_comm_map=False,
            show_comm_plot=False,
            show_table=False,
            show_poi=False,
            save_static_map=True,
        )
    finally:
        pipeline.save_result = original_save_result


def _static_community_structure(city_cfg, algorithm):
    graph, _ = pipeline.get_graph(city_cfg)
    community, _ = pipeline.get_community(city_cfg, algorithm)
    info, old_to_new = pipeline._comm_info(graph, community)
    aggregate = nx.DiGraph()
    aggregate.add_nodes_from(info)
    for u, v, edge in graph.edges(data=True):
        cu, cv = community.get(u, -1), community.get(v, -1)
        if cu == cv or cu == -1 or cv == -1:
            continue
        source, target = old_to_new[cu], old_to_new[cv]
        weight = aggregate.get_edge_data(source, target, {}).get("weight", 0.0)
        aggregate.add_edge(source, target, weight=weight + edge.get("weight", 1.0))
    position = nx.spring_layout(aggregate, seed=30, iterations=120)
    sizes = [120 + 45 * np.log1p(info[node]["size"]) for node in aggregate]
    weights = [edge["weight"] for _, _, edge in aggregate.edges(data=True)]
    maximum = max(weights, default=1.0)
    widths = [0.5 + 4.5 * weight / maximum for weight in weights]
    colors = [pipeline._static_get_color(node) for node in aggregate]
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    nx.draw_networkx_edges(aggregate, position, ax=ax, width=widths,
                           edge_color="#777777", alpha=0.65, arrows=True,
                           arrowsize=10, connectionstyle="arc3,rad=0.04")
    nx.draw_networkx_nodes(aggregate, position, ax=ax, node_size=sizes,
                           node_color=colors, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(aggregate, position, ax=ax, font_color="white",
                            font_weight="bold", font_size=8)
    ax.set_axis_off()
    out_dir = pipeline.RESULTS_DIR / city_cfg["city"] / algorithm / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "community_structure.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "community_structure.pdf", bbox_inches="tight")
    plt.close(fig)


def _all_static_community_structures():
    for city_cfg in pipeline.CITIES:
        for algorithm in pipeline.ALGORITHMS:
            _static_community_structure(city_cfg, algorithm)


def main():
    cities = [city["city"] for city in pipeline.CITIES]
    original_get_community = pipeline.get_community
    original_summary_saver = pipeline._save_plotly_summary
    pipeline.get_community = _saved_community_loader(original_get_community)
    pipeline._save_plotly_summary = lambda fig, name, **kwargs: fig.write_html(
        str(pipeline.SUMMARY_DIR / f"{name}.html")
    )
    try:
        _original_figures()
        _all_static_community_structures()
        pipeline.create_algorithm_fairness_table()
        pipeline.show_summary()
        revision_scripts = Path(__file__).resolve().parent / "revision"
        revision_output = pipeline.OUTPUT_ROOT / "revision/additional_analyses"
        for script, source, output in [
            ("metric_associations.py", pipeline.SUMMARY_DIR / "master_results.csv", revision_output),
            ("plot_metric_correlations.py", revision_output / "metric_associations.csv", revision_output / "figures"),
            ("plot_within_method_scatter.py", pipeline.SUMMARY_DIR / "master_results.csv", revision_output / "figures"),
        ]:
            subprocess.run(
                [sys.executable, str(revision_scripts / script), "--input", str(source), "--output-dir", str(output)],
                check=True,
            )
        pipeline.analyze_metric_independence(cities=cities, algorithms=pipeline.ALGORITHMS)
        pipeline.run_spatial_compactness_comparison(cities=cities, algorithms=pipeline.ALGORITHMS)
        pipeline.analyze_scale_sensitivity(cities=cities, algorithms=pipeline.ALGORITHMS)
        pipeline.document_city_boundaries(cities=cities)
    finally:
        pipeline.get_community = original_get_community
        pipeline._save_plotly_summary = original_summary_saver


if __name__ == "__main__":
    main()
