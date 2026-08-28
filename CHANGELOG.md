# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) conventions. Each WORKPLAN
phase lands as one entry set when its branch merges; the `[Unreleased]`
section accumulates changes on in-flight branches.

## [Unreleased]

- Phase 3B settlement-category mapping layer: a profile now declares
  `categories: {scheme, mapping}` (source type → category, 1:1 or X:1) and
  writes `methodology.exclusion.types` in **category** names; every output —
  CSV, shapefile, joblib — and `missing_population.csv` carry a `category`
  column beside the raw `USO_FINAL`, the joblib also carries the
  scheme/mapping in `attrs`, and each run logs
  `categories: scheme=… n_categories=…`. The mapping is applied in the one
  population/exclusion prelude both entry points share, so `compute_frames`
  and the CLI cannot diverge; `compute_frames` gains `mapping=`/`scheme=`
  (`None` = the identity over the city's own types). An unmapped source type
  is an **error**, never a warning and never a fallback: `compute` exits 1
  naming every offending type with its row count, and `categories.default` is
  rejected at load as reserved. `categories` is required in every profile,
  duplicate YAML keys are now rejected naming key and line (PyYAML kept the
  last one silently), and `exclusion.types ⊆ categories` is checked both at
  load and at run time (in-memory callers never pass through `load_config`).
  **No numbers changed**: both shipped profiles use the identity `uso-10`
  scheme, `tests/fixtures/oraculum/production/code-2025.csv` and
  `manuscript.csv` are byte-identical, and
  `scripts/verify_against_baseline.py --config code-2025` still reports
  `0.000e+00` on all 23 columns against the July 2025 baseline (real-data
  proof, 27 Aug 2026: `categories: scheme=uso-10 n_categories=10`, 4,131
  reported, `PASS — new run equivalent to July 2025 baseline within
  tolerance`). Proved by a CLI end-to-end that collapses the oracle city's
  six types into five, excludes the category `non-urban`, and reproduces
  both today's raw `[RV, IND]` exclusion and the independent reference
  implementation's `code/excl_contributing` and `code/excl_removed` blocks
  for both stages and both denominators. Raj's Phase 4 decision (DEL-31) is
  now one YAML file. Tests 246 → 278 (`test_categories`, the config and
  pipeline cases, the collapse e2e, the unmapped-type guard, the fixture
  id == type pin). Docs: `docs/methodology-config.md` § 2,
  `docs/data/uso_final_vocabulary.md`. Spec:
  `docs/superpowers/specs/2026-08-27-phase3b-categories-design.md`.
  [DEL-17, DEL-18 (further partial)]
- Phase 3A refactor: `spatial_index_utils.py` (822 lines) and the two driver
  scripts are gone; the pipeline is the installable `delhi_psi` package
  (`config`, `io`, `validate`, `geometry`, `neighbors`, `index`, `pipeline`,
  `cli`, `verify`) built with hatchling, with a `delhi-psi
  {preprocess,compute} --config <profile>` CLI. Every methodology choice —
  adjacency rule, barrier rule and layer combination, decay, roads formula,
  denominator, second normalization, exclusion `stage` × `absent_neighbor` —
  is a validated config value; two profiles ship (`code-2025` = today's
  behaviour, `manuscript` = the paper's rule-set). **No numbers changed**:
  `tests/fixtures/oraculum/production/code-2025.csv` was snapshotted from the
  pre-refactor code before any module moved and is reproduced byte-for-byte by
  the refactored pipeline, and
  `scripts/verify_against_baseline.py --config code-2025` reports zero
  deviation from the July 2025 baseline (real-data proof, 27 Aug 2026:
  `delhi-psi preprocess` — 4,357 settlements, 595 barrier-flagged, all five
  layers pass the validation battery; `delhi-psi compute` — 4,131 reported,
  15 missing-population rows; `verify_against_baseline.py` — max abs
  deviation `0.000e+00` on all 23 output columns, `PASS — new run equivalent
  to July 2025 baseline within tolerance`). The `compute` stage now drops
  exact-duplicate service rows before validation, generalising
  `compute_psi.py`'s bank-only `drop_duplicates` to every service layer
  (only `bank` has duplicate rows on the real layers, 1,240 of 10,637). The
  silent `except: pass` in `calc_pcen_mobile` is now an explicit lookup
  miss, which is what makes `absent_neighbor: contributes` implementable
  (DEL-21). Root `conftest.py` and every `sys.path.insert` are gone
  (`tests/__init__.py` plus the editable install do that job); `pyyaml`
  moved to the runtime dependencies and `uv.lock` was regenerated. Notebook
  eyeball checks became raising assertions in `delhi_psi.validate` (DEL-25).
  Tests 77 → 230, including `test_config`, `test_profiles_match_reference`
  (both profiles × every scenario × both reference denominators),
  `test_manuscript_anchors` (the hand-ratified worksheet values),
  `test_production_fixtures`, `test_cli`, `test_validate`. Spec:
  `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md`.
  [DEL-15, DEL-16, DEL-18 (partial), DEL-21, DEL-22, DEL-25]
- Dead code: removed 17 functions with no callers (transitively) from
  `spatial_index_utils.py` — 684 lines, 44% of the module — including every
  `*_wards` / `*_buffer` variant, the unused `generate_colonies_with_exclusions`
  exclusion helpers, two superseded neighbor builders and two plotting helpers;
  pruned the `pickle`, `importlib.reload` and `matplotlib.pyplot` imports they
  alone used. CI now runs `uv run pytest -W error` (the suite has been
  warning-free since DEL-26); `tests/test_ci_workflow.py` pins it. [DEL-23]
- pandas 3: lifted the `pandas<3` cap (`pyproject.toml`), lock moved
  2.3.3 → 3.0.5. Five integer-sentinel column initializations
  (`spatial_index_utils.py` L812/L1090/L1161/L1203, `scripts/preprocess.py`
  L160) now start as float — pandas 3 raises `TypeError` where 2.x emitted the
  "incompatible dtype" `FutureWarning` and silently upcast. Oracle suite 77/77,
  fixtures byte-identical, warning count 364 → 0. Dependabot: 280 open alerts
  (all on `archive/master-2021/requirements.txt` or the deleted `poetry.lock`)
  dismissed as not-used. [DEL-26]
- CI: `.github/workflows/ci.yml` runs `uv sync --locked`, the oracle suite
  and a fixture-drift guard (regenerate `scripts/generate_*_fixtures.py`,
  `git diff --exit-code tests/fixtures/`) on every push to `main` and every
  PR. Drift guard uses `git status --porcelain` so untracked generator output also fails. Structural contract pinned by `tests/test_ci_workflow.py`; PyYAML added
  to the dev group. Spec: `docs/superpowers/specs/2026-08-24-ci-workflow-design.md`.
- Oracle worksheet hand-ratified by Bob (24 Aug 2026) against the April
  2026 manuscript's Eq. 1–4; Phase 2 fully closed. Added
  `docs/oracle/suggested-fixes-memo.md` (proposed fix per divergence,
  incl. new #7: distance unit unstated in the manuscript) and paper
  evidence for the roads and barrier items.

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
