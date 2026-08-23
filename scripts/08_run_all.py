"""Run the complete analysis pipeline in numbered order."""
from __future__ import annotations

import runpy
from pathlib import Path


STAGES = [
    "00_validate_algorithms.py",
    "01_prepare_data.py",
    "02_run_main_analysis.py",
    "03_run_poi_analysis.py",
    "03b_plot_poi_summary.py",
    "04_run_parameter_sensitivity.py",
    "04b_run_comparator_sensitivity.py",
    "04c_build_matched_granularity.py",
    "05_run_threshold_sensitivity.py",
    "05b_run_representation_sensitivity.py",
    "05c_plot_representation_sensitivity.py",
    "05f_run_lp_stop_threshold_scan.py",
    "06_run_stability.py",
    "07_summarize_and_plot.py",
]


def main():
    code_dir = Path(__file__).resolve().parent
    for stage in STAGES:
        print(f"\n{'=' * 72}\nRUNNING {stage}\n{'=' * 72}")
        runpy.run_path(str(code_dir / stage), run_name="__main__")


if __name__ == "__main__":
    main()
