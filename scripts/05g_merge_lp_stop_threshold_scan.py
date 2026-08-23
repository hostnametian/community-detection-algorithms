"""Merge and validate independently written L/P threshold-scan shards."""
from pathlib import Path
import pandas as pd

from analysis_pipeline import ALGORITHMS, CITIES, OUTPUT_ROOT


def main():
    files = sorted(
        path for path in OUTPUT_ROOT.glob("lp_stop_threshold_fast*/lp_stop_threshold_scan.csv")
        if path.parent.name != "lp_stop_threshold_fast4"  # superseded by 4a/4b/4c
    )
    if not files:
        raise FileNotFoundError("No accelerated L/P threshold-scan shard outputs found")
    data = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    keys = ["city", "algorithm", "threshold_m"]
    if data.duplicated(keys).any():
        duplicate = data.loc[data.duplicated(keys, keep=False), keys]
        raise ValueError(f"Duplicate conditions:\n{duplicate.to_string(index=False)}")
    expected = {
        (city["city"], algorithm, threshold)
        for city in CITIES for algorithm in ALGORITHMS
        for threshold in (50, 75, 100, 125, 150)
    }
    observed = set(map(tuple, data[keys].itertuples(index=False, name=None)))
    missing, extra = expected - observed, observed - expected
    if missing or extra:
        raise ValueError(
            f"Incomplete scan: {len(observed)}/{len(expected)} conditions; "
            f"missing={sorted(missing)}; extra={sorted(extra)}"
        )
    out = OUTPUT_ROOT / "lp_stop_threshold_scan"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "lp_stop_threshold_scan.csv"
    data.sort_values(keys).to_csv(target, index=False)
    print(f"[DONE] validated {len(data)} unique rows from {len(files)} shards: {target}")


if __name__ == "__main__":
    main()
