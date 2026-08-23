"""03 — Run POI analysis using the authoritative saved main partitions."""
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import analysis_pipeline as pipeline


def _load_saved(city_cfg, algorithm, **algorithm_kwargs):
    if algorithm_kwargs:
        return _ORIGINAL(city_cfg, algorithm, **algorithm_kwargs)
    root = pipeline.RESULTS_DIR / city_cfg["city"] / algorithm
    candidates = []
    for config_path in root.glob("*/config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("script_version") == pipeline.PIPELINE_VERSION:
            candidates.append((config.get("date_run", ""), config_path.parent, config))
    if not candidates:
        return _ORIGINAL(city_cfg, algorithm)
    _, run_dir, config = max(candidates, key=lambda item: item[0])
    community = {}
    with open(run_dir / "communities.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node = int(row["node_id"])
            community[node] = int(row["community_id"])
    metadata_keys = ["algorithm_parameters", "directed_used", "weight_used", "spatial_prior", "algorithm_group", "note"]
    return community, {key: config[key] for key in metadata_keys if key in config}


_ORIGINAL = pipeline.get_community
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


def _run_task(city, algorithm, n_permutations, force=False):
    tag = f"{city}__{algorithm}"
    null_path = pipeline.SUMMARY_DIR / f"poi_null_model_tests__{tag}.csv"
    if null_path.exists() and not force:
        existing = pd.read_csv(null_path)
        if (
            len(existing) == 16
            and "n_permutations" in existing
            and existing["n_permutations"].eq(n_permutations).all()
        ):
            return f"[skip] POI complete: {city}×{algorithm}"
    pipeline.get_community = _load_saved
    try:
        pipeline.run_poi_all(
            cities=[city],
            algorithms=[algorithm],
            n_permutations=n_permutations,
            result_tag=tag,
            render_outputs=False,
        )
    finally:
        pipeline.get_community = _ORIGINAL
    return f"[done] POI: {city}×{algorithm}"


def _run_city_task(city, algorithms, n_permutations, force=False):
    """Keep a city's graph, POIs, and nearest matches warm across algorithms."""
    return [
        _run_task(city, algorithm, n_permutations, force=force)
        for algorithm in algorithms
    ]


def _aggregate_task_outputs():
    sensitivity_paths = sorted(
        pipeline.SUMMARY_DIR.glob("poi_matching_distance_sensitivity__*.csv")
    )
    null_paths = sorted(pipeline.SUMMARY_DIR.glob("poi_null_model_tests__*.csv"))
    if not sensitivity_paths or not null_paths:
        raise RuntimeError("No completed POI task outputs found")
    sensitivity = pd.concat([pd.read_csv(path) for path in sensitivity_paths], ignore_index=True)
    sensitivity = sensitivity.drop_duplicates(
        ["city", "algorithm", "poi_max_distance_m"], keep="last"
    )
    nulls = pd.concat([pd.read_csv(path) for path in null_paths], ignore_index=True)
    nulls = nulls.drop_duplicates(
        ["city", "algorithm", "poi_max_distance_m", "null_type", "spatial_block_m"],
        keep="last",
    )
    nulls["p_value_bh"] = np.nan
    family_columns = ["poi_max_distance_m", "null_type", "spatial_block_m"]
    for _, indices in nulls.groupby(family_columns, dropna=False).groups.items():
        nulls.loc[indices, "p_value_bh"] = pipeline._benjamini_hochberg(
            nulls.loc[indices, "p_value"]
        )
    nulls["significant_bh_0_05"] = nulls["p_value_bh"] <= 0.05
    # Report interpretable entropy contrasts alongside significance.  The global
    # and spatial-block nulls use the same observed statistic, so the residual
    # contrast is directly comparable across null specifications.
    nulls["entropy_effect"] = nulls["observed"] - nulls["null_mean"]
    global_effects = nulls[nulls["null_type"].eq("global")][
        ["city", "algorithm", "poi_max_distance_m", "entropy_effect"]
    ].rename(columns={"entropy_effect": "global_entropy_effect"})
    nulls = nulls.merge(global_effects, on=["city", "algorithm", "poi_max_distance_m"], how="left")
    nulls["effect_retained_after_spatial_block"] = np.where(
        nulls["null_type"].eq("spatial_block"), nulls["entropy_effect"], np.nan
    )
    sensitivity.to_csv(
        pipeline.SUMMARY_DIR / "poi_matching_distance_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nulls.to_csv(
        pipeline.SUMMARY_DIR / "poi_null_model_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_distance = int(pipeline.GLOBAL_CFG["poi_max_distance_m"])
    master = sensitivity[sensitivity["poi_max_distance_m"].eq(primary_distance)].copy()
    primary_null = nulls[
        nulls["poi_max_distance_m"].eq(primary_distance)
        & nulls["null_type"].eq("global")
    ][["city", "algorithm", "null_mean", "null_sd", "null_ci_low", "null_ci_high", "entropy_effect", "z_score", "p_value", "p_value_bh"]]
    primary_null = primary_null.rename(columns={
        "null_mean": "poi_entropy_null_mean",
        "null_sd": "poi_entropy_null_sd",
        "null_ci_low": "poi_entropy_null_ci_low",
        "null_ci_high": "poi_entropy_null_ci_high",
        "entropy_effect": "poi_entropy_effect",
        "z_score": "poi_entropy_z",
        "p_value": "poi_entropy_p",
        "p_value_bh": "poi_entropy_p_bh",
    })
    master = master.merge(primary_null, on=["city", "algorithm"], how="left")
    master.to_csv(
        pipeline.SUMMARY_DIR / "poi_master.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"[POI aggregate] {len(master)} main rows, {len(sensitivity)} sensitivity rows, "
        f"{len(nulls)} null tests",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", action="append")
    parser.add_argument("--algorithm", action="append", choices=pipeline.ALGORITHMS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--native-threads", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=pipeline.GLOBAL_CFG["poi_permutations"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cities = [city["city"] for city in pipeline.CITIES if not args.city or city["city"] in args.city]
    algorithms = args.algorithm or pipeline.ALGORITHMS
    tasks = [(city, algorithms, args.permutations, args.force) for city in cities]
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(max(1, args.native_threads),),
    ) as executor:
        futures = [executor.submit(_run_city_task, *task) for task in tasks]
        for future in as_completed(futures):
            for message in future.result():
                print(message, flush=True)
    _aggregate_task_outputs()


if __name__ == "__main__":
    main()
