# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) conventions. Each WORKPLAN
phase lands as one entry set when its branch merges; the `[Unreleased]`
section accumulates changes on in-flight branches.

## [Unreleased]

_Nothing yet — Phase 3 (refactor & bug audit) is next._

## [2026-08-17] Phase 2 — the mythical-city oracle (PR #6)

Verification: 65 tests green; production == independent reference
implementation == hand-derived anchors at 1e-12 across all scenarios and
both denominators; mutation testing (17+ sabotages of the index) confirms
the suite catches a broken index. Hand ratification of the derivation
worksheet is pending by design.

### Added
- Mythical-city oracle ("Oraculum"): hand-verifiable fixtures, an
  independent Eq. 1–4 reference implementation, and pytest suites
  establishing production == reference == hand-arithmetic agreement
  (`tests/fixtures/oraculum/`, `tests/reference_impl.py`,
  `tests/test_oracle*.py`, `tests/test_fixture_invariants.py`,
  `tests/test_divergence_exhibit.py`)
- Oracle maps, derivation worksheet (ratification pending), and the
  exclusion-semantics memo for Raj (`docs/oracle/`)
- Empirically pinned manuscript-vs-code divergences: directed bbox
  adjacency, global asymmetric barrier rule, neighbor-decayed roads,
  code-only `norm_psi` and popdensity denominator, and exclusion
  semantics (a) degenerating to (b) via silent exception swallowing
  (flagged for Phase 3 bug audit)

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
