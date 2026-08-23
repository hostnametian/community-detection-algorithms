"""04 — Run/resume Infomap and LeidenRBC parameter grids."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from analysis_pipeline import CITIES, RESULTS_DIR, run_sensitivity


PARAMETER_GRIDS = {
    "LeidenRBC": ("resolution", [0.025, 0.05, 0.1, 0.2, 0.5, 1.0]),
    "Infomap": ("markov_time", [0.5, 0.75, 1.0, 1.5, 2.0]),
}


def _task(city_name, algorithm, parameter, values, force=False):
    city_cfg = next(city for city in CITIES if city["city"] == city_name)
    output = RESULTS_DIR / city_name / algorithm / "tables" / f"sensitivity_{parameter}.csv"
    if output.exists() and not force:
        return f"[skip] {city_name}×{algorithm}×{parameter}"
    run_sensitivity(city_cfg, algorithm, parameter, values)
    return f"[done] {city_name}×{algorithm}×{parameter}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tasks = [
        (city["city"], algorithm, parameter, values, args.force)
        for city in CITIES
        for algorithm, (parameter, values) in PARAMETER_GRIDS.items()
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(_task, *task) for task in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
