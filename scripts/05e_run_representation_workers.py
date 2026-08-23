"""Run representation sensitivity by city with bounded process-level parallelism."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context


def _run_city(city: str, worker_root: str):
    # Import inside the spawned process: graph-tool is not fork-safe after import.
    from analysis_pipeline import run_representation_sensitivity
    return run_representation_sensitivity(
        cities=[city],
        algorithms=None,
        output_subdir=f"{worker_root}/{city}",
    )


def main():
    from analysis_pipeline import CITIES

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--worker-root",
        default="representation_sensitivity_all_algorithms_workers",
    )
    parser.add_argument("--city", action="append", default=None)
    args = parser.parse_args()
    cities = args.city or [city["city"] for city in CITIES]
    if not 1 <= args.workers <= len(cities):
        raise ValueError("--workers must be between 1 and the number of selected cities")

    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(_run_city, city, args.worker_root): city
            for city in cities
        }
        for future in as_completed(futures):
            city = futures[future]
            future.result()
            print(f"[OK] {city} complete", flush=True)


if __name__ == "__main__":
    main()
