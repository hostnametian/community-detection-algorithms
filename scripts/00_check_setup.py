"""Check repository configuration and expected input files without heavy imports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def check_python_syntax() -> int:
    """Compile source text in memory so repository checks leave no cache files."""
    scripts = sorted((REPOSITORY_ROOT / "scripts").glob("*.py"))
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    return len(scripts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-missing-data",
        action="store_true",
        help="Report missing raw inputs without returning an error.",
    )
    args = parser.parse_args()

    script_count = check_python_syntax()

    data_root = Path(os.environ.get("PTN_DATA_ROOT", REPOSITORY_ROOT / "data")).expanduser()
    config_path = Path(
        os.environ.get("PTN_CITY_CONFIG", REPOSITORY_ROOT / "config" / "cities.json")
    ).expanduser()
    cities = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(cities, list) or not cities:
        raise SystemExit(f"City configuration must be a non-empty JSON array: {config_path}")

    missing = []
    for city in cities:
        for field in ("json_file", "poi_file"):
            path = Path(city[field]).expanduser()
            if not path.is_absolute():
                path = data_root / path
            if not path.is_file():
                missing.append(path)

    print(f"Repository: {REPOSITORY_ROOT}")
    print(f"Python syntax: {script_count} scripts passed")
    print(f"City configuration: {config_path}")
    print(f"Data root: {data_root}")
    print(f"Configured cities: {len(cities)}")
    if missing:
        print(f"Missing input files: {len(missing)}")
        for path in missing:
            print(f"  - {path}")
        if not args.allow_missing_data:
            raise SystemExit(1)
    else:
        print("All expected input files are present.")


if __name__ == "__main__":
    main()
