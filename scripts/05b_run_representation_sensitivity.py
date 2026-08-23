"""05b — Compare all six algorithms across Endpoint, L-space and P-space.

Use ``--city`` to run independent city workers in parallel. Each worker writes
to its own directory; `05d_merge_representation_sensitivity.py` validates and
merges the seven worker outputs after completion.
"""
import argparse

from analysis_pipeline import run_representation_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", action="append", default=None)
    parser.add_argument(
        "--output-subdir",
        default="representation_sensitivity_all_algorithms_output",
    )
    args = parser.parse_args()
    run_representation_sensitivity(
        cities=args.city,
        algorithms=None,  # None explicitly expands to the six registered algorithms
        output_subdir=args.output_subdir,
    )


if __name__ == "__main__":
    main()
