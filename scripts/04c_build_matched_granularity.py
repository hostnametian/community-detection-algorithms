"""Select parameter-grid partitions close to predeclared community counts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis_pipeline import ALGORITHMS, CITIES, RESULTS_DIR, SUMMARY_DIR


TARGET_COMMUNITY_COUNTS = (5, 10, 15, 20)


def main():
    frames = []
    for path in sorted(RESULTS_DIR.glob("*/*/tables/sensitivity_*.csv")):
        frame = pd.read_csv(path)
        if frame.empty or "community_count" not in frame:
            continue
        if "city" not in frame:
            frame["city"] = path.parts[-4]
        if "algorithm" not in frame:
            frame["algorithm"] = path.parts[-3]
        if "parameter" not in frame:
            frame["parameter"] = path.stem.removeprefix("sensitivity_")
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise RuntimeError("No parameter-sensitivity tables were found")

    candidates = pd.concat(frames, ignore_index=True)
    candidates["community_count"] = pd.to_numeric(candidates["community_count"], errors="coerce")
    candidates = candidates.dropna(subset=["community_count"])
    selected_rows = []
    grouped = {
        (city, algorithm): group
        for (city, algorithm), group in candidates.groupby(["city", "algorithm"], sort=True)
    }
    for city in [config["city"] for config in CITIES]:
      for algorithm in ALGORITHMS:
        group = grouped.get((city, algorithm))
        if group is None:
            for target in TARGET_COMMUNITY_COUNTS:
                selected_rows.append({
                    "city": city,
                    "algorithm": algorithm,
                    "target_community_count": target,
                    "matched": False,
                    "unmatched_reason": (
                        "block_count_inferred_by_MDL_not_tuned_to_target_K"
                        if algorithm == "DCSBM"
                        else "no_parameter_grid_candidates"
                    ),
                })
            continue
        deterministic_group = group.sort_values(
            ["parameter", "param_value", "source_file"], kind="stable"
        )
        for target in TARGET_COMMUNITY_COUNTS:
            tolerance = max(1, int(round(target * 0.10)))
            eligible = deterministic_group.assign(
                community_count_distance=(deterministic_group["community_count"] - target).abs()
            )
            eligible = eligible[eligible["community_count_distance"] <= tolerance]
            if eligible.empty:
                selected_rows.append({
                    "city": city,
                    "algorithm": algorithm,
                    "target_community_count": target,
                    "matched": False,
                    "unmatched_reason": "no_candidate_within_predeclared_K_tolerance",
                })
                continue
            # Tie-breaking is based only on K distance and a deterministic
            # parameter order, never on a performance metric.
            chosen = eligible.sort_values(
                ["community_count_distance", "parameter", "param_value", "source_file"],
                kind="stable",
            ).iloc[0].to_dict()
            chosen.update({
                "target_community_count": target,
                "matched": True,
                "unmatched_reason": "",
            })
            selected_rows.append(chosen)

    output = pd.DataFrame(selected_rows)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(
        SUMMARY_DIR / "matched_granularity_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(output[[
        "city", "algorithm", "target_community_count", "matched",
        "community_count", "parameter", "param_value"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
