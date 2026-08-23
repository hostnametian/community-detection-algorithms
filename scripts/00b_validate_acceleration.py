"""Fixed-seed equivalence and micro-benchmark checks for safe accelerations."""
from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


HERE = Path(__file__).resolve().parent


def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_assign_pois(graph, community, pois, max_distance_m=None):
    node_ids = []
    coordinates = []
    for node in graph.nodes():
        data = graph.nodes[node]
        if "lat" in data and "lon" in data:
            node_ids.append(node)
            coordinates.append([data["lat"], data["lon"]])
    tree = BallTree(np.radians(np.asarray(coordinates)), metric="haversine")
    distances, indices = tree.query(np.radians(pois[["lat", "lon"]].values), k=1)
    indices = indices[:, 0]
    result = pois.copy()
    result["nearest_node"] = [node_ids[index] for index in indices]
    result["nearest_distance_m"] = distances[:, 0] * 6371000.0
    result["community"] = [community.get(node_ids[index], -1) for index in indices]
    if max_distance_m is not None:
        result.loc[result["nearest_distance_m"] > float(max_distance_m), "community"] = -1
    result["matched"] = result["community"] != -1
    return result


def _legacy_permutation_test(
    assigned,
    community,
    n_permutations,
    seed,
    graph,
    spatial_block_m,
):
    """Pre-acceleration scalar implementation retained as a test oracle."""
    matched = assigned[
        (assigned["community"] != -1) & assigned["nearest_node"].notna()
    ].copy()
    node_ids = [node for node, label in community.items() if label != -1]
    node_index = {node: index for index, node in enumerate(node_ids)}
    matched = matched[matched["nearest_node"].isin(node_index)]
    label_values = sorted({community[node] for node in node_ids})
    label_index = {label: index for index, label in enumerate(label_values)}
    base_labels = np.array([label_index[community[node]] for node in node_ids], dtype=int)
    poi_node_indices = np.array(
        [node_index[node] for node in matched["nearest_node"]], dtype=int
    )
    _, category_codes = np.unique(matched["category"].to_numpy(), return_inverse=True)
    n_categories = int(category_codes.max()) + 1
    node_category_counts = np.zeros((len(node_ids), n_categories), dtype=np.int64)
    np.add.at(node_category_counts, (poi_node_indices, category_codes), 1)
    active_nodes = np.flatnonzero(node_category_counts.sum(axis=1) > 0)

    def weighted_entropy(node_labels):
        counts = np.zeros((len(label_values), n_categories), dtype=float)
        np.add.at(counts, node_labels[active_nodes], node_category_counts[active_nodes])
        totals = counts.sum(axis=1)
        present = totals > 0
        probabilities = np.divide(
            counts[present],
            totals[present, None],
            out=np.zeros_like(counts[present]),
            where=totals[present, None] > 0,
        )
        terms = np.zeros_like(probabilities)
        positive = probabilities > 0
        terms[positive] = probabilities[positive] * np.log2(probabilities[positive])
        entropies = (
            -terms.sum(axis=1) / np.log2(n_categories)
            if n_categories > 1
            else np.zeros(present.sum())
        )
        return float(np.average(entropies, weights=totals[present]))

    observed = weighted_entropy(base_labels)
    rng = np.random.default_rng(seed)
    block_indices = None
    spatial_block_count = None
    active_spatial_block_count = None
    if spatial_block_m is not None:
        coordinates = [
            (float(graph.nodes[node]["lat"]), float(graph.nodes[node]["lon"]))
            for node in node_ids
        ]
        lat0 = np.radians(np.mean([latitude for latitude, _ in coordinates]))
        radius_m = 6371000.0
        xy = np.array(
            [
                [
                    radius_m * np.radians(longitude) * np.cos(lat0),
                    radius_m * np.radians(latitude),
                ]
                for latitude, longitude in coordinates
            ]
        )
        cell_ids = np.floor(xy / float(spatial_block_m)).astype(np.int64)
        grouped_indices = defaultdict(list)
        for index, cell in enumerate(map(tuple, cell_ids)):
            grouped_indices[cell].append(index)
        spatial_block_count = len(grouped_indices)
        block_indices = []
        for indices in grouped_indices.values():
            indices = np.asarray(indices, dtype=int)
            if len(indices) > 1 and len(np.unique(base_labels[indices])) > 1:
                block_indices.append(indices)
        active_spatial_block_count = len(block_indices)

    def permuted_labels():
        if block_indices is None:
            return rng.permutation(base_labels)
        permuted = base_labels.copy()
        for indices in block_indices:
            permuted[indices] = rng.permutation(base_labels[indices])
        return permuted

    null = np.array([weighted_entropy(permuted_labels()) for _ in range(n_permutations)])
    null_mean = float(null.mean())
    null_sd = float(null.std(ddof=1)) if len(null) > 1 else 0.0
    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_ci_low": float(np.quantile(null, 0.025)),
        "null_ci_high": float(np.quantile(null, 0.975)),
        "z_score": (observed - null_mean) / null_sd if null_sd > 0 else float("nan"),
        "p_value": float(
            (1 + np.sum(np.abs(null - null_mean) >= abs(observed - null_mean)))
            / (len(null) + 1)
        ),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "null_type": "spatial_block" if spatial_block_m is not None else "global",
        "spatial_block_m": spatial_block_m,
        "spatial_block_count": spatial_block_count,
        "active_spatial_block_count": active_spatial_block_count,
    }


def _fixture():
    rng = np.random.default_rng(20260817)
    graph = nx.DiGraph()
    side = 20
    for node in range(side * side):
        row, column = divmod(node, side)
        graph.add_node(
            node,
            lat=39.80 + row * 0.001,
            lon=116.30 + column * 0.001,
        )
        if column:
            graph.add_edge(node - 1, node, weight=1.0)
        if row:
            graph.add_edge(node - side, node, weight=1.0)
    community = {node: (node // 8) % 7 for node in graph}
    pois = pd.DataFrame(
        {
            "lat": rng.uniform(39.798, 39.822, 6000),
            "lon": rng.uniform(116.298, 116.322, 6000),
            "category": rng.choice(list("ABCDEFGH"), 6000),
        }
    )
    return graph, community, pois


def _assert_frame_equal(left, right):
    columns = ["nearest_node", "nearest_distance_m", "community", "matched"]
    pd.testing.assert_frame_equal(
        left[columns].reset_index(drop=True),
        right[columns].reset_index(drop=True),
        check_exact=True,
    )


def _assert_result_equal(legacy, accelerated):
    exact_keys = [
        "n_permutations",
        "seed",
        "null_type",
        "spatial_block_m",
        "spatial_block_count",
        "active_spatial_block_count",
    ]
    for key in exact_keys:
        assert legacy[key] == accelerated[key], (key, legacy[key], accelerated[key])
    numeric_keys = [
        "observed",
        "null_mean",
        "null_sd",
        "null_ci_low",
        "null_ci_high",
        "z_score",
        "p_value",
    ]
    for key in numeric_keys:
        assert math.isclose(legacy[key], accelerated[key], rel_tol=1e-13, abs_tol=1e-13), (
            key,
            legacy[key],
            accelerated[key],
        )


def main():
    accelerated = _load_module("accelerated_pipeline", HERE / "analysis_pipeline.py")
    graph, community, pois = _fixture()

    nearest = accelerated.match_pois_to_nodes(graph, pois)
    for threshold in (250, 500, 1000):
        old_assignment = _legacy_assign_pois(graph, community, pois, max_distance_m=threshold)
        new_assignment = accelerated.assign_pois(
            graph,
            community,
            pois,
            max_distance_m=threshold,
            nearest_matches=nearest,
        )
        _assert_frame_equal(old_assignment, new_assignment)

    assigned = _legacy_assign_pois(graph, community, pois, max_distance_m=1000)
    timings = {}
    for null_name, block_m in (("global", None), ("spatial", 1000)):
        start = time.perf_counter()
        old_result = _legacy_permutation_test(
            assigned,
            community,
            999,
            731,
            graph,
            block_m,
        )
        timings[f"legacy_{null_name}"] = time.perf_counter() - start
        start = time.perf_counter()
        new_result = accelerated.poi_entropy_permutation_test(
            assigned,
            community,
            n_permutations=999,
            seed=731,
            graph=graph,
            spatial_block_m=block_m,
            permutation_batch_size=64,
        )
        timings[f"accelerated_{null_name}"] = time.perf_counter() - start
        _assert_result_equal(old_result, new_result)

    print("Acceleration equivalence checks passed.")
    for null_name in ("global", "spatial"):
        old_time = timings[f"legacy_{null_name}"]
        new_time = timings[f"accelerated_{null_name}"]
        print(
            f"{null_name}: legacy={old_time:.3f}s accelerated={new_time:.3f}s "
            f"speedup={old_time / new_time:.2f}x"
        )


if __name__ == "__main__":
    main()
