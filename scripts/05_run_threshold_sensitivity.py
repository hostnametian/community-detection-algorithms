"""05 — Check sensitivity of stop annotation/grouping (not graph topology)."""
from analysis_pipeline import run_threshold_sensitivity


def main():
    run_threshold_sensitivity(
        thresholds=(50, 75, 100, 125, 150),
        algorithms=("LeidenRBC", "Infomap"),
    )


if __name__ == "__main__":
    main()
