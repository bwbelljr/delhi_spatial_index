# Phase 1: Runnable Pipeline — Design Spec

Date: 2026-08-16
Status: **approved by owner (2026-08-16)** — /ship pipeline authorized
Branch: `phase1-runnable-pipeline` (off `origin/main`)
Parent plan: `WORKPLAN.md` Phase 1

## Decision log (autonomous-run authorizations, per /ship)

- **Autonomy scope**: fix forward, commit, push, and **merge the PR** — all
  authorized without mid-run check-ins.
- **Plan-vs-reviewer conflicts**: a CONFIRMED Critical finding governs over
  the implementation plan; every deviation recorded in CHANGELOG.md and the
  PR description. Non-critical conflicts follow the plan and are surfaced
  afterward.
- **Failure policy**: fix and retry to done. The spec's stop-and-ask red
  lines still apply: no methodology changes (index equations, exclusion
  semantics, oracle-relevant behavior) and no writes to baseline data —
  those halt the run for the owner.
- **Changelog**: CHANGELOG.md added to the repo; this run updates its
  `[Unreleased]` section as part of the PR.

## Purpose

Make the Delhi PSI pipeline runnable end-to-end on current (Aug 2026)
dependencies, on any machine, with no hardcoded paths — and prove the new
environment reproduces the July 2025 outputs. This is a **consistency**
milestone, not a correctness one: correctness against the manuscript's
equations arrives with the Phase 2 oracle. Phase 1 therefore changes the
*packaging and drivers* of the code, never its logic.

## Background

- The pipeline is two Jupyter notebooks driving `spatial_index_utils.py`:
  pre-processing (validate/dedupe/reproject shapefiles, barrier flags,
  bbox neighbors → joblib) and PSI calculation (merge population, drop RV,
  `calc_all_services` in pop-size and pop-density variants → CSV/shapefile/
  joblib outputs).
- The July 2025 run already modernized the stack substantially (pyproject
  targets Python ≥3.13, geopandas 1.1, shapely 2.1, pandas 2.3), so the
  dependency delta is mid-2025 → Aug-2026 latest, not 2021 → 2026.
- The July 2025 outputs exist in `~/delhi_data` (`colonies_bbox_nbrs2025.joblib`,
  `psi_2020_results/*`) and serve as the frozen baseline.
- Meta-decisions already made (see `WORKPLAN.md` "Decisions" section):
  uv + `pyproject.toml` as the single dependency source; zero notebooks in
  the end state; package restructure deferred to Phase 3.

## Goals

1. `uv sync` produces a working environment from `pyproject.toml` alone.
2. The pipeline runs top-to-bottom as two plain Python scripts on any
   machine pointed at a copy of the dataset.
3. No absolute paths anywhere in the code.
4. A verification script demonstrates the new environment's outputs are
   numerically equivalent to the July 2025 baseline.

## Non-goals (explicitly out of scope)

- No refactoring of `spatial_index_utils.py` (Phase 3).
- No behavior changes, including known oddities:
  - the dedup steps guarded by `if not os.path.exists(...)` keep their
    skip-when-present behavior;
  - bbox-based adjacency stays as-is (adjudicated in Phase 2/3);
  - the RV-only exclusion stays (industrial areas dropped in Phase 4).
- No package layout, CLI entry points, or config files (Phase 3).
- No new tests beyond the verification script (the oracle is Phase 2).

## Design

### 1. Tooling

- Install `uv`; target Python 3.13 via `uv python install` (system has 3.12;
  uv manages its own interpreters).
- Rewrite `pyproject.toml` for uv as a **non-package project**
  (`[tool.uv] package = false` or equivalent current convention — confirm
  against uv docs via context7): Phase 1 ships scripts + a root module, not
  an installable package.
- Dependencies, all latest stable at implementation time: `geopandas`,
  `pandas`, `shapely`, `matplotlib`, `pyproj`, `tqdm`, `joblib`.
  **Removed**: `jupyterlab` (no notebooks remain). Dev group: `pytest`
  (scaffold for Phase 2).
- `uv lock` committed.
- **Deleted files**: `requirements.txt`, `environment.yml`, `poetry.lock`,
  `Dockerfile`, `install_conda_environment.sh`.
- README setup section rewritten: clone → `uv sync` → run scripts.

### 2. Notebooks → scripts (mechanical translation)

- `scripts/preprocess.py` — line-for-line port of
  `Colonies Dataset Pre-Processing (2025).ipynb`.
- `scripts/compute_psi.py` — line-for-line port of
  `Colonies Public Services Index Calculations Updated (no RV) 2025.ipynb`.
- Translation rules: identical call sequence and arguments; cell-boundary
  echo expressions (`.head()`, `len()`, `.crs`) become `print()`/logging
  lines so the scripts narrate the same sanity information; tqdm progress
  preserved; no reordering, no "improvements."
- Both notebooks are deleted on this branch (history + `archive/master-2021/`
  retain the lineage).

### 3. Path handling

- Data root resolved in priority order: `--data-dir` flag →
  `DELHI_DATA_DIR` env var → `~/delhi_data` (user-expanded).
- The relative layout under the data root is unchanged (matches both the
  local copy and the synced shared-drive copy `Spatial_Index_GIS/delhi_data/`).
- Outputs go through `--out-dir` (default: the data root, matching current
  behavior for normal runs). Output directories are created with
  `os.makedirs(..., exist_ok=True)`.
- New output filenames use honest dates (`*_aug2026.*`), replacing the
  stale `12Sep2021` names. The verifier maps old → new names explicitly.
- Stage coupling: `compute_psi.py` takes `--neighbors-file` for the joblib
  produced by `preprocess.py` (default:
  `<data-dir>/colonies_bbox_nbrs2025.joblib`, today's behavior). The
  verification run passes the freshly generated file from
  `phase1_verify/` so the whole chain — not just the last stage — is
  exercised on the new environment.
- Acceptance grep: `grep -rn "/home/" *.py scripts/` returns nothing.

### 4. Baseline-safe verification

- Verification runs write to `<data-dir>/phase1_verify/`; the July 2025
  baseline files are never opened for writing.
- `scripts/verify_against_baseline.py` compares:
  - **Neighbors joblib**: same set of colony IDs, identical neighbor lists
    (as sets), neighbor distances within tolerance.
  - **PSI outputs** (both pop-size and pop-density variants): CSVs joined on
    colony ID; numeric columns compared with `numpy.allclose`
    (rtol=1e-9, atol=1e-12 as the starting tolerance); per-column max
    absolute/relative deviation reported.
- Exit code 0 with a summary table when equivalent; nonzero with a diff
  report when not. Deviations beyond tolerance are investigated before
  merge — a legitimate library-behavior change (e.g. geometry engine
  updates) may justify loosening the tolerance, but only with a written
  note in the PR explaining the cause.

### 5. Known risks

- **Dependency drift breaks an API** (mid-2025 → Aug-2026): expected to be
  minor; consult context7 for current geopandas/shapely APIs rather than
  guessing; fix call sites minimally.
- **Numeric drift from geometry engines**: shapely/GEOS updates can move
  results in the last decimals; the tolerance + investigation rule covers
  this.
- **Translation slips**: the output comparison is precisely the net that
  catches them.
- **Long runtime**: the neighbors computation is the slow step (tqdm loops
  over 4,352 colonies). Verification needs one full run; budget accordingly
  in the implementation plan.

## Acceptance criteria

1. Fresh clone + `uv sync` + `uv run scripts/preprocess.py --data-dir ...`
   + `uv run scripts/compute_psi.py --data-dir ...` complete without error
   on this machine against `~/delhi_data`.
2. `verify_against_baseline.py` passes: neighbors and both PSI variants
   equivalent to the July 2025 baseline within tolerance.
3. No absolute paths in code; no notebook files in the repo root; exactly
   one dependency declaration file (`pyproject.toml` + `uv.lock`).
4. README documents the new setup and run commands.
5. PR from `phase1-runnable-pipeline` reviewed (`/code-review`) and merged
   to `main`.
