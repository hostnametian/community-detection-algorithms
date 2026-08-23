"""Scan stop-group distance thresholds for the L--P representation pair.

This is deliberately separate from the POI/annotation threshold experiment:
each threshold rebuilds the stop graph, reruns all six algorithms, and reports
L--P agreement, community counts, and graph sizes.  No outcome is selected.
"""
import argparse
from pathlib import Path
import pandas as pd

from analysis_pipeline import CITIES, PTNGraphBuilder, ALGORITHMS, ALGORITHM_REGISTRY, OUTPUT_ROOT, _partition_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs="+", type=int, default=[50, 75, 100, 125, 150])
    ap.add_argument("--city", action="append", default=None)
    ap.add_argument("--algorithm", action="append", choices=ALGORITHMS, default=None)
    ap.add_argument("--output-subdir", default="lp_stop_threshold_scan")
    args = ap.parse_args()
    selected = [c for c in CITIES if args.city is None or c["city"] in args.city]
    algorithms = ALGORITHMS if args.algorithm is None else args.algorithm
    out = OUTPUT_ROOT / args.output_subdir
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / "lp_stop_threshold_scan.csv"
    key_columns = ["city", "threshold_m", "algorithm"]
    if output_file.exists():
        existing = pd.read_csv(output_file)
        rows = existing.to_dict("records")
        completed = set(map(tuple, existing[key_columns].itertuples(index=False, name=None)))
    else:
        rows, completed = [], set()

    def save():
        data = pd.DataFrame(rows).drop_duplicates(key_columns, keep="last")
        data = data.sort_values(key_columns)
        temporary = output_file.with_suffix(".csv.tmp")
        data.to_csv(temporary, index=False)
        temporary.replace(output_file)

    for cfg in selected:
        for threshold in args.thresholds:
            builder = PTNGraphBuilder(cfg["json_file"], cfg["city"], cfg["country"], threshold,
                                      city_query=cfg.get("city_query", cfg["city"]))
            builder.build()
            l_graph, p_graph = builder.build_stop_graph(), builder.build_p_space_graph()
            common = sorted(set(l_graph) & set(p_graph))
            if len(common) < 2:
                continue
            for algorithm in algorithms:
                key = (cfg["city"], threshold, algorithm)
                if key in completed:
                    print(f"[skip] {key}")
                    continue

                def run(graph, representation):
                    kwargs = {}
                    if algorithm == "DCSBM":
                        kwargs["n_restarts"] = 50
                        kwargs["checkpoint_dir"] = str(
                            OUTPUT_ROOT / "dcsbm_checkpoints" / "lp_threshold"
                            / cfg["city"] / str(threshold) / representation
                        )
                        # The original seed 688 (restart slot 38) did not return
                        # for the Shanghai 150 m P-space graph.  Use the next
                        # unused deterministic seed to retain 50 valid fits.
                        if (cfg["city"], threshold, representation) == ("ShangHai", 150, "p_space"):
                            kwargs["restart_seed_overrides"] = {38: 892}
                    return ALGORITHM_REGISTRY[algorithm](graph, **kwargs)

                lp, _ = run(l_graph, "l_space")
                pp, _ = run(p_graph, "p_space")
                ll = [lp.get(n, -1) for n in common]; pl = [pp.get(n, -1) for n in common]
                valid = [(a, b) for a, b in zip(ll, pl) if a != -1 and b != -1]
                a, b = zip(*valid)
                agreement = _partition_agreement(a, b)
                rows.append({"city": cfg["city"], "threshold_m": threshold, "algorithm": algorithm,
                             "l_nodes": l_graph.number_of_nodes(), "p_nodes": p_graph.number_of_nodes(),
                             "l_edges": l_graph.number_of_edges(), "p_edges": p_graph.number_of_edges(),
                             "l_communities": len(set(lp.values())),
                             "p_communities": len(set(pp.values())),
                             **agreement, "common_nodes": len(valid),
                             "common_support_rate": len(valid) / len(set(l_graph) | set(p_graph))})
                completed.add(key)
                save()
                print(f"[saved] {key}: {len(completed)} completed rows")
    save()
    print(f"wrote {len(completed)} rows to {output_file}")


if __name__ == "__main__":
    main()
