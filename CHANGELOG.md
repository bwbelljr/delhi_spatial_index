# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) conventions. Each WORKPLAN
phase lands as one entry set when its branch merges; the `[Unreleased]`
section accumulates changes on in-flight branches.

## [Unreleased]

## [2026-08-17] Phase 1 — runnable pipeline on modern dependencies

Verification: fresh run reproduced the July 2025 baseline outputs with zero
numeric deviation (all columns, both PSI variants, all neighbor sets and
distances).

### Added
- Repo-scoped `/ship` build-and-ship pipeline command
  (`.claude/commands/ship.md`)
- Phase 1 design spec (`docs/superpowers/specs/2026-08-16-phase1-runnable-pipeline-design.md`)
- `scripts/preprocess.py`, `scripts/compute_psi.py` — pipeline as plain
  scripts with configurable `--data-dir`/`--out-dir` (flag > `DELHI_DATA_DIR`
  env var > `~/delhi_data`)
- `scripts/verify_against_baseline.py` — proves a fresh run matches the
  July 2025 baseline outputs
- pytest suite for path resolution and baseline comparison

### Changed
- Dependency management consolidated to uv + `pyproject.toml` (all packages
  at latest stable; Python 3.13). One documented exception: `pandas>=2.3,<3`
  — the pandas 3.0 major-version jump is deferred until the Phase 2 oracle
  can validate it
- Output filenames now dated `aug2026` (previously mislabeled `12Sep2021`)

### Removed
- Both Jupyter notebooks (logic now in `scripts/`; history preserved in git
  and `archive/master-2021/`)
- `requirements.txt`, `environment.yml`, `poetry.lock`, `Dockerfile`,
  `install_conda_environment.sh`

## [2026-08-16] Repository restructure (pre-phase)

### Changed
- `main` became the default branch (content from `bb_update`); 2020–2021
  code archived under `archive/master-2021/`; `master` and `bb_update`
  branches removed with history preserved in `main`
- README updated for the new layout

### Added
- `WORKPLAN.md` — sequenced plan toward HAS submission, with meta-planning
  decisions and open Raj/group questions
