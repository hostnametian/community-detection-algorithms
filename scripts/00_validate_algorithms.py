"""Deterministic smoke tests for the custom GeoLP and DirLou baselines."""
from __future__ import annotations

import math

import networkx as nx

from analysis_pipeline import directed_modularity, run_dirlou, run_geolp


def _add_coordinates(graph):
    for index, node in enumerate(graph.nodes()):
        graph.nodes[node]["lat"] = 55.0 + index * 0.001
        graph.nodes[node]["lon"] = 37.0 + (index % 4) * 0.001


def _two_directed_cliques(include_self_loops=False):
    graph = nx.DiGraph()
    for base in (0, 5):
        nodes = range(base, base + 5)
        for source in nodes:
            for target in nodes:
                if source != target:
                    graph.add_edge(source, target, weight=5.0)
                elif include_self_loops:
                    graph.add_edge(source, target, weight=2.0)
    graph.add_edge(4, 5, weight=0.1)
    graph.add_edge(5, 4, weight=0.05)
    _add_coordinates(graph)
    return graph


def validate_dirlou():
    for include_self_loops in (False, True):
        graph = _two_directed_cliques(include_self_loops=include_self_loops)
        partition_a, metadata_a = run_dirlou(graph, resolution=1.0, seed=40)
        partition_b, metadata_b = run_dirlou(graph, resolution=1.0, seed=40)
        assert partition_a == partition_b, "DirLou must reproduce the same partition for a fixed seed"
        assert set(partition_a) == set(graph), "DirLou must retain every original node"
        assert len(set(partition_a.values())) == 2, "The two-clique test should recover two groups"
        params = metadata_a["algorithm_parameters"]
        assert params["objective_monotonic_non_decreasing"], "DirLou objective decreased across levels"
        final_objective = directed_modularity(graph, partition_a, resolution=1.0)
        assert math.isclose(final_objective, params["objective_trace"][-1], rel_tol=1e-12)
        assert metadata_a["algorithm_parameters"] == metadata_b["algorithm_parameters"]


def validate_geolp():
    graph = _two_directed_cliques()
    partition, metadata = run_geolp(
        graph,
        geographic_weight=0.5,
        bandwidth_m=5000,
        initial_communities=2,
        seed=42,
    )
    assert set(partition) == set(graph), "GeoLP must retain every original node"
    params = metadata["algorithm_parameters"]
    assert params["edge_strength_normalization"] == (
        "reciprocal_sum_divided_by_global_max_reciprocal_sum"
    )
    assert params["converged"], "GeoLP did not converge on the deterministic smoke graph"


def main():
    validate_dirlou()
    validate_geolp()
    print("Custom algorithm validation passed.")


if __name__ == "__main__":
    main()
