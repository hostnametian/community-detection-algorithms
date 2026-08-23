"""Run one DCSBM restart in an isolated, monitorable process.

Each invocation handles exactly one global restart index.  Existing checkpoints
are resumed by ``run_dcsbm``; a shell-level timeout can safely terminate a
stalled graph-tool call without affecting other restarts.
"""
import argparse
import datetime as dt
import os
import sys
from pathlib import Path

from analysis_pipeline import CITIES, OUTPUT_ROOT, PTNGraphBuilder, run_dcsbm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True)
    p.add_argument("--threshold", required=True, type=int)
    p.add_argument("--representation", choices=("l_space", "p_space"), required=True)
    p.add_argument("--restart", required=True, type=int)
    p.add_argument("--seed", type=int, default=None,
                   help="Optional replacement seed for this restart slot")
    args = p.parse_args()
    if not 0 <= args.restart < 50:
        raise SystemExit("--restart must be in [0,50)")
    cfg = next((x for x in CITIES if x["city"] == args.city), None)
    if cfg is None:
        raise SystemExit(f"Unknown city: {args.city}")
    checkpoint = (OUTPUT_ROOT / "dcsbm_checkpoints" / "lp_threshold" /
                  args.city / str(args.threshold) / args.representation)
    checkpoint.mkdir(parents=True, exist_ok=True)
    log = checkpoint / f"restart_{args.restart:03d}.monitor.log"
    def note(msg):
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    actual_seed = args.seed if args.seed is not None else 42 + 17 * args.restart
    note(f"START pid={os.getpid()} restart={args.restart} seed={actual_seed}")
    builder = PTNGraphBuilder(cfg["json_file"], cfg["city"], cfg["country"],
                              args.threshold, city_query=cfg.get("city_query", cfg["city"]))
    builder.build()
    graph = (builder.build_p_space_graph() if args.representation == "p_space"
             else builder.build_stop_graph())
    note(f"GRAPH_READY nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    run_dcsbm(graph, n_restarts=50, restart_indices=[args.restart],
              restart_seed_overrides={args.restart: actual_seed},
              checkpoint_dir=str(checkpoint))
    note("DONE checkpoint_written=" + str(checkpoint / f"restart_{args.restart:03d}.npz"))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
