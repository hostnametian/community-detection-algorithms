"""06 — Run/resume repeated-seed stability checks for the six methods."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from analysis_pipeline import ALGORITHMS, CITIES, RESULTS_DIR, run_stability


_THREADPOOL_LIMITER = None


def _init_worker(native_threads):
    global _THREADPOOL_LIMITER
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(native_threads)
    try:
        from threadpoolctl import threadpool_limits
        _THREADPOOL_LIMITER = threadpool_limits(limits=native_threads)
    except ImportError:
        _THREADPOOL_LIMITER = None


def _task(city_name, algorithm, runs, force=False):
    output = RESULTS_DIR / city_name / algorithm / "tables" / "stability_nmi_summary.csv"
    if output.is_file() and not force:
        existing = pd.read_csv(output)
        expected_pairs = runs * (runs - 1) // 2
        if not existing.empty and int(existing.iloc[0]["n_pairs"]) == expected_pairs:
            return f"[skip] completed stability: {city_name}×{algorithm}"
    city_cfg = next(city for city in CITIES if city["city"] == city_name)
    run_stability(city_cfg, algorithm, n_runs=runs)
    return f"[done] stability: {city_name}×{algorithm}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", action="append", help="Internal city name; repeatable")
    parser.add_argument("--algorithm", action="append", choices=ALGORITHMS)
    parser.add_argument("--exclude", action="append", choices=ALGORITHMS, default=[])
    parser.add_argument("--force", action="store_true", help="Recompute completed tables")
    parser.add_argument("--runs", type=int, help="Override the algorithm-specific run count")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--native-threads", type=int, default=1)
    args = parser.parse_args()
    cities = [city for city in CITIES if not args.city or city["city"] in args.city]
    algorithms = args.algorithm or ALGORITHMS
    algorithms = [algorithm for algorithm in algorithms if algorithm not in args.exclude]
    tasks = [
        (city_cfg["city"], algorithm, args.runs or (50 if algorithm == "DCSBM" else 20), args.force)
        for city_cfg in cities
        for algorithm in algorithms
    ]
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(max(1, args.native_threads),),
    ) as executor:
        futures = [executor.submit(_task, *task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
