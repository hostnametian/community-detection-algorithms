"""Portable locations for revision analyses; generated files stay under outputs/."""
import os
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_ROOT.parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / 'scripts'
BASELINE_ROOT = Path(os.environ.get('PTN_BASELINE_ROOT', REPOSITORY_ROOT / 'outputs')).expanduser().resolve()
ROOT = Path(os.environ.get('PTN_REVISION_ROOT', REPOSITORY_ROOT / 'outputs/revision')).expanduser().resolve()
ANALYSIS_ROOT = ROOT / 'additional_analyses'
BOUNDARY_ROOT = Path(os.environ.get('PTN_BOUNDARY_ROOT', BASELINE_ROOT / 'threshold_sensitivity_output/boundary_cache')).expanduser().resolve()


def table_path(relative):
    """Prefer locally generated tables, falling back to the published summaries."""
    path = BASELINE_ROOT / relative
    if path.is_file():
        return path
    published = REPOSITORY_ROOT / 'outputs/published' / Path(relative).name
    if published.is_file():
        return published
    raise FileNotFoundError(f'Expected analysis table: {path}')


def run_path(record):
    """Relocate archived run references beneath the selected baseline outputs."""
    path = Path(record)
    if 'results' in path.parts:
        index = path.parts.index('results')
        candidate = BASELINE_ROOT.joinpath(*path.parts[index:])
        if candidate.is_dir():
            return candidate
    if path.is_absolute() and path.is_dir():
        return path
    candidate = REPOSITORY_ROOT / path
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f'Archived partition is required: {path}; set PTN_BASELINE_ROOT to its output directory')
