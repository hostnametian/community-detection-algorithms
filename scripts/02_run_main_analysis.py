"""02 — Run/resume the six authoritative main-setting partitions."""
from __future__ import annotations

import argparse
import json

from analysis_pipeline import ALGORITHMS, CITIES, PIPELINE_VERSION, RESULTS_DIR, run_one


def _completed(city, algorithm):
    for config_path in (RESULTS_DIR / city / algorithm).glob("*/config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("script_version") == PIPELINE_VERSION:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", action="append", help="Internal city name; repeatable")
    parser.add_argument("--algorithm", action="append", choices=ALGORITHMS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cities = [city for city in CITIES if not args.city or city["city"] in args.city]
    algorithms = args.algorithm or ALGORITHMS

    for city_cfg in cities:
        for algorithm in algorithms:
            if not args.force and _completed(city_cfg["city"], algorithm):
                print(f"  [skip] completed main result: {city_cfg['city']}×{algorithm}")
                continue
            run_one(
                city_cfg,
                algorithm,
                show_base=False,
                show_comm_map=False,
                show_comm_plot=False,
                show_table=False,
                show_poi=False,
                save_static_map=False,
            )


if __name__ == "__main__":
    main()
