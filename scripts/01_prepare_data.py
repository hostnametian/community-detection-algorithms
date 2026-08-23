"""01 — Load raw OSM data, filter city extents, and build the seven PTN graphs."""
from __future__ import annotations

import json
import platform
from importlib.metadata import PackageNotFoundError, version

import networkx as nx
import pandas as pd

from analysis_pipeline import (
    ALGORITHMS,
    CITIES,
    GLOBAL_CFG,
    OUTPUT_ROOT,
    PIPELINE_VERSION,
    get_graph,
    graph_fingerprint,
)


def _package_versions():
    packages = [
        "networkx", "networkit", "numpy", "pandas", "scipy", "scikit-learn",
        "osmnx", "geopandas", "python-igraph", "leidenalg", "infomap", "graph-tool",
    ]
    found = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def main():
    stage_dir = OUTPUT_ROOT / "01_data_and_graphs"
    stage_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    route_quality_rows = []
    stop_sequence_rows = []
    for city_cfg in CITIES:
        graph, builder = get_graph(city_cfg)
        stop_graph = builder.build_stop_graph()
        p_space_graph = builder.build_p_space_graph()
        l_audit = stop_graph.graph.get("sequence_audit", {})
        route_audit = builder.stats.get("route_quality", {})
        rows.append({
            "city": city_cfg["city"],
            "country": city_cfg["country"],
            "source_file": city_cfg["json_file"],
            "nodes": graph.number_of_nodes(),
            "directed_edges": graph.number_of_edges(),
            "stop_groups": len(builder.ga),
            "l_space_nodes": stop_graph.number_of_nodes(),
            "l_space_directed_edges": stop_graph.number_of_edges(),
            "l_space_density": nx.density(stop_graph),
            "p_space_nodes": p_space_graph.number_of_nodes(),
            "p_space_directed_edges": p_space_graph.number_of_edges(),
            "p_space_density": nx.density(p_space_graph),
            "ordered_route_relations": stop_graph.graph.get("ordered_route_relations", 0),
            "raw_route_relations": route_audit.get("raw_route_relations", 0),
            "filtered_route_relations": route_audit.get("filtered_route_relations", 0),
            "continuous_route_relations": route_audit.get("continuous_route_relations", 0),
            "discontinuous_route_relations": route_audit.get("discontinuous_route_relations", 0),
            "ambiguous_direction_ways": route_audit.get("ways_with_ambiguous_direction", 0),
            "endpoint_nodes_before_gwcc": builder.stats.get("endpoint_nodes_before_gwcc", 0),
            "endpoint_edges_before_gwcc": builder.stats.get("endpoint_edges_before_gwcc", 0),
            "endpoint_nodes_removed_by_gwcc": builder.stats.get("gwcc", {}).get("nodes_removed", 0),
            "graph_fingerprint": graph_fingerprint(graph),
        })
        route_quality_rows.append({
            "city": city_cfg["city"],
            "country": city_cfg["country"],
            **route_audit,
            **{f"gwcc_{key}": value for key, value in builder.stats.get("gwcc", {}).items()},
        })
        stop_sequence_rows.append({
            "city": city_cfg["city"],
            "country": city_cfg["country"],
            **l_audit,
            "l_space_nodes_before_gwcc": stop_graph.graph.get("nodes_before_gwcc", 0),
            "l_space_edges_before_gwcc": stop_graph.graph.get("edges_before_gwcc", 0),
            "l_space_components_before_gwcc": stop_graph.graph.get("components_before_gwcc", 0),
            "l_space_nodes_removed_by_gwcc": stop_graph.graph.get("nodes_removed_by_gwcc", 0),
            "p_space_nodes_before_gwcc": p_space_graph.graph.get("nodes_before_gwcc", 0),
            "p_space_edges_before_gwcc": p_space_graph.graph.get("edges_before_gwcc", 0),
            "p_space_components_before_gwcc": p_space_graph.graph.get("components_before_gwcc", 0),
            "p_space_nodes_removed_by_gwcc": p_space_graph.graph.get("nodes_removed_by_gwcc", 0),
        })

    pd.DataFrame(rows).to_csv(
        stage_dir / "graph_inventory.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(route_quality_rows).to_csv(
        stage_dir / "endpoint_route_quality_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(stop_sequence_rows).to_csv(
        stage_dir / "stop_sequence_quality_audit.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "python": platform.python_version(),
        "global_config": GLOBAL_CFG,
        "algorithms": ALGORITHMS,
        "default_algorithm_parameters": {
            "Infomap": {"markov_time": 1.0, "flow_model": "directed"},
            "LeidenRBC": {"resolution": 0.1, "partition": "RBConfigurationVertexPartition"},
            "DCSBM": {"model": "non-nested directed weighted degree-corrected SBM", "deg_corr": True, "edge_covariate": "weight", "rec_type": "real-normal", "n_restarts": GLOBAL_CFG["dcsbm_restarts"], "selection": "minimum description length"},
            "GeoLP": {"geographic_weight": 0.7, "bandwidth_m": 5000, "initial_communities": 25, "min_cluster_size": 0.0, "reciprocal_strength": "w_ij + w_ji", "normalization": "global maximum reciprocal sum"},
            "Spinglass": {"spins": 25, "projection": "reciprocal-weight-summed", "interpretation": "explicit granularity cap"},
            "DirLou": {"resolution": 0.1, "max_levels": 5, "max_iter": 50},
        },
        "fixed_seeds": {
            "global": 42,
            "Infomap": 42,
            "LeidenRBC": 35,
            "DCSBM": 42,
            "GeoLP": 42,
            "Spinglass": 42,
            "DirLou": 40,
            "stability_increment": 17,
        },
        "package_versions": _package_versions(),
    }
    with open(stage_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
