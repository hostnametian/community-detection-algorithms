"""Precompute a disjoint range of DCSBM checkpoints for one representation.

The global restart index determines the unchanged seed ``42 + 17 * index``.
Multiple workers may therefore compute disjoint ranges safely; a subsequent
normal 50-restart call loads all checkpoints and applies the original global
minimum-description-length selection rule.
"""
import argparse

from analysis_pipeline import CITIES, OUTPUT_ROOT, PTNGraphBuilder, run_dcsbm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--threshold", required=True, type=int)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--stop", required=True, type=int)
    parser.add_argument("--representation", choices=("l_space", "p_space"), default="p_space")
    args = parser.parse_args()
    cfg = next((item for item in CITIES if item["city"] == args.city), None)
    if cfg is None:
        raise ValueError(f"Unknown city: {args.city}")
    indices = list(range(args.start, args.stop))
    if not indices or args.start < 0 or args.stop > 50:
        raise ValueError("Range must be a non-empty subset of [0, 50)")
    builder = PTNGraphBuilder(
        cfg["json_file"], cfg["city"], cfg["country"], args.threshold,
        city_query=cfg.get("city_query", cfg["city"]),
    )
    builder.build()
    graph = builder.build_p_space_graph() if args.representation == "p_space" else builder.build_stop_graph()
    checkpoint = (
        OUTPUT_ROOT / "dcsbm_checkpoints" / "lp_threshold" / args.city
        / str(args.threshold) / args.representation
    )
    run_dcsbm(
        graph, n_restarts=50, restart_indices=indices,
        checkpoint_dir=str(checkpoint),
    )
    print(f"[DONE] {args.city} {args.threshold} {args.representation}: {args.start}:{args.stop}")


if __name__ == "__main__":
    main()
