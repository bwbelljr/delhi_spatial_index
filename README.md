# Delhi Public Services Index

## Description

This repository includes code to generate a spatial index of services for Delhi, based on the forthcoming paper "Towards an Urban Public Services Index" from Georgetown's [Urban Spatial Observatory](https://www.urbanspatialobservatory.org/). Although the data is not provided in this repository, the scripts can be used to generate an urban public services index in another city.

## Repository layout

* `spatial_index_utils.py` - library with all spatial index functions
* `scripts/preprocess.py` - pre-processing pipeline (validation, deduplication, reprojection, barriers, neighbors)
* `scripts/compute_psi.py` - PSI calculation pipeline
* `scripts/verify_against_baseline.py` - compares a fresh run to the July 2025 baseline outputs
* `archive/master-2021/` - snapshot of the original 2020-2021 code, including variant analyses (ward-level index, buffer-based PSI, exclusions). See `archive/master-2021/ARCHIVE_README.md`.

## Setup

* Install [uv](https://docs.astral.sh/uv/)
* Clone: `git clone https://github.com/bwbelljr/delhi_spatial_index.git`
* `cd delhi_spatial_index && uv sync`
* Point the pipeline at your data directory (defaults to `~/delhi_data`;
  override with `--data-dir` or the `DELHI_DATA_DIR` environment variable)

## Running the pipeline

```bash
uv run python scripts/preprocess.py --data-dir ~/delhi_data --out-dir ~/delhi_data/phase1_verify
uv run python scripts/compute_psi.py --data-dir ~/delhi_data \
    --neighbors-file ~/delhi_data/phase1_verify/colonies_bbox_nbrs_aug2026.joblib \
    --out-dir ~/delhi_data/phase1_verify
uv run python scripts/verify_against_baseline.py --data-dir ~/delhi_data
```

Tests: `uv run pytest`
