# DEL-36: Distance-band stability — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-10-27-del-36-distance-bands-design.md`.

### Task 1: `baseline_paths` (TDD)

- [ ] Write `test_baseline_paths_uses_the_profiles_own_artifact_name` in
  `tests/test_summarize_sweep.py`: on a tmp dir, `baseline_paths("code-2025",
  d)` ends in `delhi_psi_code-2025_popdensity_2020.csv` +
  `colonies_neighbors.joblib`; `baseline_paths("ratified-2026", d)` ends in
  `delhi_psi_ratified-2026_popdensity_2020.csv` +
  `colonies_neighbors_ratified-2026.joblib`. Watch it fail.
- [ ] Implement `baseline_paths(profile, baseline_dir)` using
  `load_config(profile, data_dir=baseline_dir).paths.neighbors_artifact`;
  have `gather_points(..., baseline_profile="code-2025")` call it. Thread
  the argument through the three render functions and add
  `--baseline-profile` to the CLI (default `code-2025`). Green.

### Task 2: The three profiles + fixtures

- [ ] Rewrite `band-1km.yaml`, `band-5km.yaml`, `band-10km.yaml` as the
  full `ratified-2026` methodology with only the adjacency pair changed
  (`within_distance`, `max_distance_km`), outputs `[popdensity]`/`[csv]`,
  and the header comment updated (ratified base, ratified rules, not a dry
  run).
- [ ] `uv run python scripts/generate_production_fixtures.py`; confirm only
  the three `band-*` production fixtures changed (`git status --porcelain --
  tests/fixtures/`), then the three `band-*` regenerability tests pass.

### Task 3: The sweep

- [ ] `uv run python scripts/run_sweep.py --group bands --work-dir
  /tmp/del36-sweep` (background, ~45–75 min). Expect three manifests with
  `status: OK`.
- [ ] Render: `uv run python scripts/summarize_sweep.py --work-dir
  /tmp/del36-sweep --baseline-dir ~/delhi_data/phase4_ratified
  --baseline-profile ratified-2026 --block points` and `--block ordering`
  (stdout first; check the numbers).

### Task 4: The document

- [ ] Write `docs/data/distance_bands.md` with a preamble (governing
  principle, baseline = the ratified run, run date/commit), the two fenced
  blocks spliced from the summariser (`--splice`), and a short Finding.
- [ ] Register `"distance_bands.md": "scripts/summarize_sweep.py"` in
  `tests/test_measure_common.DOCUMENT_GENERATORS`.
- [ ] CHANGELOG `[Unreleased]` entry.

### Task 5: Ship

- [ ] `uv run pytest -q -W error` (background), then PR (body: spec/plan,
  the stability table, the fixture blast radius, smoke tail), CI green,
  squash-merge, sync `main`, Jira DEL-36 In Progress → In Review → Done
  with a plain one-paragraph comment.

## Review record (plan phase, round 1 — 27 Oct 2026)

- **Model:** `opencode-go/glm-5.3-flash`, one finder + judge round.
- **Findings (2 Important, 0 Critical), both fixed in this revision:**
  1. `gather_points` hard-wires the baseline's IDENTITY and manifest
     profile string in addition to the two paths (summarize_sweep.py
     ~1004–1006); threading only the paths would render a row labelled
     `code-2025` that actually holds ratified numbers. Fixed: the profile
     argument feeds `baseline_paths`, `_identity_fields(load_config(...))`
     and the manifest profile string.
  2. `tests/test_sweep_profiles.py` pins every sweep profile to
     `BASE = "code-2025"` with exactly the adjacency pair moved — the
     ratified rewrite moves ~9 keys. Fixed: `RATIFIED_BASE` makes the three
     band profiles compare against `ratified-2026`, and the categories test
     uses the same base.
- **Verified sound:** the loader accepts every ratified value in the band
  profiles; `--baseline-profile` is additive; `--out`/`--splice`/`--block`
  behave as the plan assumes; `BANDS` is exactly 1/5/10 km; a new
  block-bearing doc only needs the `DOCUMENT_GENERATORS` entry.
- **Verdict:** not clean on round 1 (2 Important), both fixed; the fixes
  are mechanical and the review budget rule stops the loop here (one
  scaled round).
