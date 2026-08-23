# Bus-network regionalization across seven cities

Reproducible analysis code and compact published outputs for the manuscript
**“Bus-network regionalization across seven cities: effects of graph
representation and community-detection method”** by Tian Gao and Ivan
Blekanov.

The workflow compares six community-detection methods (Infomap, LeidenRBC,
DCSBM, GeoLP, Spinglass, and DirLou) on three directed weighted public-
transport network representations (Endpoint, L-space, and P-space) in seven
cities. The code version used for the manuscript is `ceus_v3`.

## Repository layout

```text
.
├── config/                 City definitions and expected input filenames
├── data/                   Local raw OSM/POI JSON files (not tracked)
├── outputs/
│   └── published/          Compact CSV results reported in the manuscript
├── scripts/                Numbered analysis workflow and shared pipeline
├── CITATION.cff            Software citation metadata
├── environment.yml         Reproducible Conda environment
└── Makefile                Common validation and analysis commands
```

Large raw OSM/POI extracts, intermediate graph objects, checkpoints, HTML
maps, and raster figures are intentionally excluded from Git. The committed
CSV files in `outputs/published/` are sufficient to inspect the principal
reported numerical results.

## Installation

The recommended installation uses Conda or Mamba because `graph-tool` is not
distributed through PyPI.

```bash
conda env create -f environment.yml
conda activate ptn-regionalization
```

Verify the source tree without loading the scientific stack:

```bash
make check
```

After installing the environment, run the deterministic GeoLP and DirLou
smoke tests:

```bash
make validate
```

## Data setup

Raw inputs are derived from OpenStreetMap and are not committed because the
seven bus and POI extracts exceed GitHub's file limits. Place the files listed
in [`data/README.md`](data/README.md) under `data/`, or point the workflow to a
different directory:

```bash
export PTN_DATA_ROOT=/path/to/osm-json
```

To use different filenames or add cities, copy and edit `config/cities.json`,
then set:

```bash
export PTN_CITY_CONFIG=/path/to/cities.json
```

Generated files are written to `outputs/` by default. Use
`PTN_OUTPUT_ROOT=/path/to/output` to keep large results elsewhere.

Map-producing stages request administrative boundaries from the OpenStreetMap
Nominatim service and therefore require internet access. If that request is
unavailable, graph construction and community analysis can still run, but
boundary-dependent maps may be omitted. Please follow the Nominatim usage
policy when reproducing figures.

## Reproducing the analysis

The numbered scripts correspond to the analysis stages. Useful entry points
include:

```bash
# Inventory and graph-construction audit
python scripts/01_prepare_data.py

# Main Endpoint partitions; filters are repeatable
python scripts/02_run_main_analysis.py --city Kazan --algorithm Infomap

# Full three-representation comparison
python scripts/05b_run_representation_sensitivity.py --city Kazan

# L/P sensitivity to 50–150 m stop grouping
python scripts/05f_run_lp_stop_threshold_scan.py --city Kazan

# Repeated-run stability
python scripts/06_run_stability.py --city Kazan --algorithm Infomap
```

Run the complete workflow with:

```bash
make reproduce
```

The complete workflow is computationally expensive. DCSBM uses 50 independent
starts for every main or representation condition, and POI null models use
9,999 permutations. The individual stage commands are preferable for testing
an installation.

## Workflow map

| Stage | Purpose |
|---|---|
| `00`, `00b` | Custom-algorithm and acceleration checks |
| `01` | Endpoint/L/P inventories and construction audits |
| `02` | Six authoritative Endpoint partitions |
| `03`, `03b` | POI matching, null models, and summaries |
| `04`, `04b`, `04c` | Parameter and matched-granularity analyses |
| `05` | Endpoint stop-annotation audit |
| `05b`–`05e` | Three-representation analysis and parallel helpers |
| `05f`–`05i` | L/P threshold scan and DCSBM checkpoint helpers |
| `06` | Repeated-run stability |
| `07` | Result summaries and analysis figures |

## Important implementation choices

- Endpoint edge weight is the number of distinct directed route relations,
  not service frequency or passenger demand.
- L-space joins consecutive ordered stop groups. P-space joins every ordered
  same-route pair reachable without a transfer.
- L-space and P-space share node units, so their comparison isolates adjacency
  sensitivity; Endpoint comparisons change both node units and adjacency.
- GeoLP combines reciprocal route-overlap strength with a distance kernel.
- DirLou is an experimental directed-modularity comparator.
- DCSBM is directed, weighted, degree corrected, and selected by minimum
  description length across 50 starts.
- POI composition is used for functional characterization, not as ground truth
  for community boundaries.

## Published outputs

[`outputs/published/README.md`](outputs/published/README.md) maps the committed
CSV files to the manuscript analyses. Absolute workstation paths have been
removed from these public copies; numerical values are unchanged.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) to cite this software. A reuse license has
not yet been selected; the authors should add one before making the repository
public.
