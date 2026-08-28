# Delhi Public Services Index

## Description

This repository includes code to generate a spatial index of services for Delhi, based on the forthcoming paper "Towards an Urban Public Services Index" from Georgetown's [Urban Spatial Observatory](https://www.urbanspatialobservatory.org/). Although the data is not provided in this repository, the scripts can be used to generate an urban public services index in another city.

## Repository layout

* `delhi_psi/` - the installable package: `config` (profiles and validation),
  `geometry`, `neighbors`, `index` (the math, as pure functions with keyword
  knobs), `io`, `validate`, `pipeline` (the stages), `cli`, `verify`
* `delhi_psi/profiles/` - one YAML per methodology profile (`code-2025` is
  today's production behaviour; `manuscript` is the paper's rule-set)
* `scripts/verify_against_baseline.py` - compares a fresh run to the July 2025 baseline outputs
* `scripts/generate_oraculum_fixtures.py`, `scripts/generate_production_fixtures.py` - regenerate the committed test fixtures
* `scripts/render_oracle_maps.py`, `scripts/check_oraculum_invariants.py` - oracle dev tools
* `archive/master-2021/` - snapshot of the original 2020-2021 code, including variant analyses (ward-level index, buffer-based PSI, exclusions). See `archive/master-2021/ARCHIVE_README.md`.

## Setup

* Install [uv](https://docs.astral.sh/uv/)
* Clone: `git clone https://github.com/bwbelljr/delhi_spatial_index.git`
* `cd delhi_spatial_index && uv sync`
* Point the pipeline at your data directory (defaults to `~/delhi_data`;
  override with `--data-dir` or the `DELHI_DATA_DIR` environment variable)

## Running the pipeline

```bash
uv run delhi-psi preprocess --config code-2025 --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_verify
uv run delhi-psi compute    --config code-2025 --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_verify
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3_verify
```

`--config` takes a shipped profile name or a path to a YAML file. Every
methodology choice — adjacency rule, barrier rule, decay, roads formula,
denominator, second normalization, exclusion semantics — is a config value.
**To change one (or turn a methodology decision into a new profile), follow
[`docs/methodology-config.md`](docs/methodology-config.md).** The schema
rationale is in `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` § 3.

Tests: `uv run pytest`
