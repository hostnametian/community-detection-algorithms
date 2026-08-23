"""Merge independently computed city-level representation-sensitivity outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analysis_pipeline import ALGORITHMS, CITIES, OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker-root",
        default="representation_sensitivity_all_algorithms_workers",
    )
    parser.add_argument(
        "--output-subdir",
        default="representation_sensitivity_all_algorithms_output",
    )
    args = parser.parse_args()
    worker_root = OUTPUT_ROOT / args.worker_root
    files = [worker_root / city["city"] / "representation_pairwise_summary.csv" for city in CITIES]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing city worker outputs:\\n" + "\\n".join(missing))

    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    # Direct cardinality is more robust than matching NaN keys: one L--P row and
    # five rows for each Endpoint target (unrestricted + four distance limits).
    expected_rows = len(CITIES) * len(ALGORITHMS) * (1 + 5 + 5)
    if len(data) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(data)}")
    observed = data.groupby(["city", "algorithm", "comparison"], dropna=False).size().unstack("comparison")
    expected_counts = {"l_space__p_space": 1, "endpoint__l_space": 5, "endpoint__p_space": 5}
    if observed.to_dict("records") != [expected_counts] * len(observed):
        raise ValueError("Each city-algorithm comparison must have 1/5/5 rows")

    out_dir = OUTPUT_ROOT / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    data.sort_values(["city", "algorithm", "comparison", "transfer_threshold_m"], na_position="first").to_csv(
        out_dir / "representation_pairwise_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"[OK] merged {len(data)} rows into {out_dir}")


if __name__ == "__main__":
    main()
