# Phase 3A Refactor (DEL-15) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 822-line `spatial_index_utils.py` monolith plus two driver scripts into the installable `delhi_psi` package with every methodology choice a validated config value — changing no numbers.

**Architecture:** A snapshot of today's production numbers is committed first (`tests/fixtures/oraculum/production/code-2025.csv`), before any code moves; it is the target the refactored pipeline must reproduce string-for-string. Then the package is built bottom-up — `config` (frozen dataclasses, enums from one table) → `geometry`/`neighbors`/`index` (pure functions with keyword knobs, math copied verbatim, `spatial_index_utils.py` delegating so the suite stays green after each move) → `io`/`validate` → `pipeline` (`compute_frames` in-memory seam, then the path stages) → `cli`. The old files are deleted only once every caller is rewired, and the last correctness step swaps the snapshot generator's backend to `compute_frames` and proves a no-op diff.

**Tech Stack:** Python 3.13 / uv, hatchling, geopandas 1.1, shapely 2.1, pandas 3.0, pyproj, joblib, tqdm, PyYAML (runtime), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` (read it in full first — sections 1–10 are the authority; § 5's migration order fixes the task order and § 4 fixes the fixture format)

## Global Constraints

- **The refactor changes no numbers.** `code-2025` must reproduce today's oracle output string-equal at `%.17g` and the July 2025 real-data baseline at zero deviation.
- **Copy the math verbatim.** Every function moved out of `spatial_index_utils.py` keeps its arithmetic, its iteration order, its sentinel values (`-1.0` initialisation) and its lack of guards (`minmax` has no `hi == lo` branch — do not add one). Renaming a parameter is fine; changing an expression is not.
- Branch: `del-15-phase3-refactor` (off `origin/main` at c9bce27). Never `git add -A`, never `git commit -a` — every commit names its files (review agents may be running: memory note "Review agents: isolate worktree").
- After **every** task: `uv run pytest -q -W error` must be green. The carried-over count is **77** at the start; it only ever grows.
- `~/delhi_data` is read-only. All test IO under pytest `tmp_path` or the repo.
- Config reaches the math as **explicit keyword arguments** to pure functions. `geometry.py`, `neighbors.py` and `index.py` never import `delhi_psi.config`.
- Every enum is validated at config load **and** the math functions still raise `ValueError` on an unknown value (no silent `UnboundLocalError` path).
- No bare `except`. The `except: pass` in `calc_pcen_mobile` becomes an explicit lookup miss (DEL-21).
- Numeric tolerance in oracle assertions: `pytest.approx(..., abs=1e-12)` unless stated.
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Stop and report (spec § 10) — do not work around — if: the `code-2025` production fixture or the real-data baseline deviate at any step; a reference-pinned key has no clean production counterpart; or a test in the carried-over 77 needs its expected value changed.

## File Structure

```
delhi_psi/
  __init__.py      version; public re-exports                     (Task 2)
  config.py        frozen dataclasses, one enum table, YAML load  (Task 2)
  profiles/code-2025.yaml, manuscript.yaml                        (Tasks 2, 12)
  geometry.py      reproject, dedup, bbox frame, barrier flags,
                   distance_to_point_km                           (Task 3)
  neighbors.py     adjacency(rule), barrier(rule, combine),
                   centroid distances                             (Task 4)
  index.py         point_counts, road_lengths, pcen, minmax,
                   service_index, overall_psi                     (Task 5)
  io.py            path resolution, layer/CSV reads, writes       (Task 6)
  validate.py      layer battery + post-compute checks, Report    (Task 6)
  pipeline.py      compute_frames (Task 7); preprocess/compute    (Task 8)
  cli.py           delhi-psi {preprocess,compute}                 (Task 8)
  verify.py        the two baseline comparison functions          (Task 9)
scripts/
  generate_production_fixtures.py   NEW — step-0 snapshot         (Task 1, 11)
  verify_against_baseline.py        thin wrapper, gains --config  (Task 9)
  generate_oraculum_fixtures.py     unchanged
  check_oraculum_invariants.py      unchanged
  render_oracle_maps.py             rewired (Task 10)
  preprocess.py, compute_psi.py, common.py   DELETED              (Task 10)
spatial_index_utils.py              delegating shim, then DELETED (Tasks 3-5, 10)
conftest.py                         DELETED                       (Task 10)
tests/
  __init__.py                       NEW                           (Task 2)
  test_config.py                    NEW                           (Task 2)
  test_validate.py                  NEW                           (Task 6)
  test_profiles_match_reference.py  NEW                           (Tasks 7, 12)
  test_cli.py                       NEW                           (Task 8)
  test_production_fixtures.py       NEW                           (Task 1)
  test_manuscript_anchors.py        NEW                           (Task 12)
  fixtures/oraculum/production/code-2025.csv, manuscript.csv       (Tasks 1, 12)
```

## Canonical facts (verified against the repo on 2026-08-27 — do not re-derive)

- The suite is **77 tests** green under `uv run pytest -q -W error` today.
- `tests/fixtures/oraculum/expected_values.csv` has 2,610 rows; its metric names are the **reference's** (`road_length_km`, `psi_eq1`). The production fixture uses the **production** names (`road_length`, `unnorm_psi`).
- The step-0 fixture is **1,450 rows** (5 scenarios × 2 denominators × 5–7 settlements × 25 metrics) and is byte-stable across reruns.
- Shapefile writing emits exactly two warnings, captured verbatim from this repo's environment:
  - `UserWarning: Column names longer than 10 characters will be truncated when saved to ESRI Shapefile.`
  - `RuntimeWarning: Normalized/laundered field name: 'a_very_long_column_name' to 'a_very_lon'` (one per long column)
- Hatchling's editable install writes a plain `.pth` containing the **project root**, so after Task 2's `uv sync` the modules `delhi_psi`, `tests` and `scripts` all import from any working directory with **no** `sys.path` manipulation. This is what makes Task 10's sweep safe.
- Deleting root `conftest.py` and adding `tests/__init__.py` keeps `from tests.oraculum_fixtures import …`, `from scripts.verify_against_baseline import …` working under pytest (verified on a throwaway checkout: 35/35 passed).

## Column names are frozen by the baseline contract

`nbrs_bbox`, `nbrs_dist_bbox`, `centroid`, `area_km2`, `ndmc_dist_km`, `road_length`, `unnorm_psi`, `norm_psi`, `<service>_count/_pcen/_idx` are the July 2025 baseline's names (spec § 5 output-column contract). They keep those names even when `adjacency.rule` is `touch`. `compare_numeric_frames` treats a missing baseline column as a deviation, so **renaming any of them fails the real-data proof.**

---

### Task 1: Step 0 — snapshot today's production numbers (spec § 5 step 0)

**This task must land before any production code moves.** The committed CSV is the refactor's correctness proof precisely because it predates the refactor.

**Files:**
- Create: `scripts/generate_production_fixtures.py`
- Create: `tests/fixtures/oraculum/production/code-2025.csv` (generator output, committed)
- Create: `tests/test_production_fixtures.py`

**Interfaces:**
- Consumes: `tests.test_oracle.SCENARIO_WIRING` and `tests.test_oracle._production_frame(denom, drop_ids_post=frozenset(), drop_ids_pre=frozenset())` — today's wiring, untouched.
- Produces:
  - `scripts.generate_production_fixtures.PRODUCTION_DIR: Path`
  - `scripts.generate_production_fixtures.POINT_SERVICES: tuple[str, ...]` = `("clinic", "school", "bank", "police", "ration", "transport")`
  - `scripts.generate_production_fixtures.SERVICES: tuple[str, ...]` = `POINT_SERVICES + ("road",)`
  - `scripts.generate_production_fixtures.DENOMS: tuple[str, ...]` = `("pop", "popdensity")`
  - `metric_columns(*, second_normalization: bool) -> list[str]`
  - `frame_records(profile: str, frame, scenario: str, denom: str, columns: list[str]) -> list[tuple]`
  - `write_fixture(path: Path, records: list[tuple]) -> None`
  - `emit_profile(profile: str, out_path: Path) -> Path`
  - `main() -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_production_fixtures.py`:

```python
"""The committed production fixtures must be exactly what the generator emits.

Same contract as test_expected_values_csv_is_regenerable: without this a red
build could be 'fixed' by hand-editing the fixture, turning the refactor's
correctness proof into a record of whatever the code now does.
"""
from pathlib import Path

import pytest

from scripts.generate_production_fixtures import (
    PRODUCTION_DIR, SERVICES, emit_profile, metric_columns,
)

PROFILES = ["code-2025"]


@pytest.mark.parametrize("profile", PROFILES)
def test_fixture_is_regenerable(profile, tmp_path):
    committed = PRODUCTION_DIR / f"{profile}.csv"
    assert committed.exists(), f"missing committed fixture {committed}"
    regen = emit_profile(profile, tmp_path / f"{profile}.csv")
    assert regen.read_text() == committed.read_text()


@pytest.mark.parametrize("profile", PROFILES)
def test_fixture_has_the_spec_shape(profile):
    text = (PRODUCTION_DIR / f"{profile}.csv").read_text()
    assert "\r" not in text, "fixtures are LF-only"
    lines = text.splitlines()
    assert lines[0] == "profile,scenario,denom,settlement,metric,value"
    rows = [line.split(",") for line in lines[1:]]
    assert all(r[0] == profile for r in rows)
    # sorted by (scenario, denom, settlement, metric)
    keys = [(r[1], r[2], r[3], r[4]) for r in rows]
    assert keys == sorted(keys)


def test_metric_set_is_explicit():
    cols = metric_columns(second_normalization=True)
    assert cols == [
        "clinic_count", "school_count", "bank_count", "police_count",
        "ration_count", "transport_count", "road_length",
        "clinic_pcen", "clinic_idx", "school_pcen", "school_idx",
        "bank_pcen", "bank_idx", "police_pcen", "police_idx",
        "ration_pcen", "ration_idx", "transport_pcen", "transport_idx",
        "road_pcen", "road_idx",
        "unnorm_psi", "norm_psi", "population", "area_km2",
    ]
    assert "norm_psi" not in metric_columns(second_normalization=False)
    # geometry / centroid / neighbor-list columns are never serialized
    for banned in ("geometry", "centroid", "nbrs_bbox", "nbrs_dist_bbox"):
        assert banned not in cols
    assert set(SERVICES) == {
        "clinic", "school", "bank", "police", "ration", "transport", "road"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_production_fixtures.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'scripts.generate_production_fixtures'`.

- [ ] **Step 3: Write the generator**

Create `scripts/generate_production_fixtures.py`:

```python
"""Emit the per-profile production fixtures (spec § 4).

Long format, one row per (profile, scenario, denom, settlement, metric):
columns `profile,scenario,denom,settlement,metric,value`, sorted by
(scenario, denom, settlement, metric), `value` at %.17g, LF line endings.
Geometry, centroid and neighbor-list columns are never serialized — their
reprs are not stable.

STEP-0 BACKEND (spec § 5 step 0): the numbers come from
`tests.test_oracle._production_frame`, i.e. today's pre-refactor wiring
through `spatial_index_utils`. The committed output is the target the
refactored pipeline must reproduce string-for-string. Migration step 5 swaps
the backend to `delhi_psi.pipeline.compute_frames` and proves a no-op diff;
nothing else about this file changes.

Regenerate with:
    uv run python scripts/generate_production_fixtures.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
# Bootstrap: the package is not installed yet at migration step 0, so the
# repo root is not on sys.path when this script is run by path (the CI drift
# guard does exactly that). Removed in migration step 1, once `uv sync`
# installs the project editable and puts the root on sys.path for good.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.test_oracle import SCENARIO_WIRING, _production_frame  # noqa: E402

PRODUCTION_DIR = REPO / "tests" / "fixtures" / "oraculum" / "production"

POINT_SERVICES = ("clinic", "school", "bank", "police", "ration", "transport")
SERVICES = POINT_SERVICES + ("road",)
DENOMS = ("pop", "popdensity")

HEADER = ["profile", "scenario", "denom", "settlement", "metric", "value"]


def metric_columns(*, second_normalization):
    """The spec § 4 metric set, in a fixed order (the CSV is sorted anyway)."""
    columns = [f"{svc}_count" for svc in POINT_SERVICES]
    columns.append("road_length")
    for svc in SERVICES:
        columns.append(f"{svc}_pcen")
        columns.append(f"{svc}_idx")
    columns.append("unnorm_psi")
    if second_normalization:
        columns.append("norm_psi")
    columns.append("population")
    columns.append("area_km2")
    return columns


def frame_records(profile, frame, scenario, denom, columns):
    """One record per (settlement, metric); `frame` is indexed by settlement."""
    return [(profile, scenario, denom, sid, metric, row[metric])
            for sid, row in frame.iterrows()
            for metric in columns]


def write_fixture(path, records):
    ordered = sorted(records, key=lambda r: (r[1], r[2], r[3], r[4]))
    df = pd.DataFrame(ordered, columns=HEADER)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def emit_profile(profile, out_path):
    """Write `profile`'s production fixture to out_path; return out_path."""
    if profile != "code-2025":
        raise ValueError(
            f"unknown profile {profile!r}: the step-0 backend only knows "
            "'code-2025' (migration step 5 generalises this)")
    columns = metric_columns(second_normalization=True)
    records = []
    for scenario, drop_pre, drop_post in SCENARIO_WIRING:
        for denom in DENOMS:
            frame = _production_frame(denom, drop_ids_post=drop_post,
                                      drop_ids_pre=drop_pre)
            records.extend(frame_records(profile, frame, scenario, denom,
                                         columns))
    write_fixture(out_path, records)
    return out_path


def main():
    out_path = emit_profile("code-2025", PRODUCTION_DIR / "code-2025.csv")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the snapshot**

Run: `uv run python scripts/generate_production_fixtures.py`
Expected: `wrote …/tests/fixtures/oraculum/production/code-2025.csv`.

Then sanity-check the shape:

```bash
wc -l tests/fixtures/oraculum/production/code-2025.csv
head -3 tests/fixtures/oraculum/production/code-2025.csv
```

Expected: **1451** lines (1,450 records + header); first data row
`code-2025,baseline,pop,A,area_km2,1`.

- [ ] **Step 5: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_production_fixtures.py -q` — Expected: 3 passed.
Run: `uv run pytest -q -W error` — Expected: **80 passed** (77 + 3).

- [ ] **Step 6: Rehearse the CI drift guard**

The guard globs `scripts/generate_*_fixtures.py`, so it now covers this file too.

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done \
  && test -z "$(git status --porcelain -- tests/fixtures/)" || echo DRIFT
```

Expected: no `DRIFT` line — but note the new CSV is **untracked** until Step 7, so run this check again after committing. (Order: commit first, then re-run to see a clean tree.)

- [ ] **Step 7: Commit**

```bash
git add scripts/generate_production_fixtures.py \
        tests/fixtures/oraculum/production/code-2025.csv \
        tests/test_production_fixtures.py
git commit -m "test: snapshot today's production numbers before the refactor (DEL-15 step 0)"
```

Then re-run the drift rehearsal from Step 6; expected: clean tree, no `DRIFT`.

---

### Task 2: Package skeleton, packaging, and `config.py` (spec § 5 step 1)

**Files:**
- Modify: `pyproject.toml` (build system, wheel target, scripts, pytest, `pyyaml` → runtime)
- Modify: `uv.lock` (regenerated, **same commit**)
- Modify: `scripts/generate_production_fixtures.py` (drop the step-0 `sys.path` bootstrap)
- Create: `tests/__init__.py` (empty)
- Create: `delhi_psi/__init__.py`, `delhi_psi/config.py`
- Create: `delhi_psi/profiles/code-2025.yaml`, `delhi_psi/profiles/manuscript.yaml`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (every later task uses these exact names):
  - `delhi_psi.config.ConfigError(ValueError)`
  - `delhi_psi.config.REFERENCE_KNOBS: dict[str, dict]` — the single enum table; keys are dotted paths, values map config value → reference knob value
  - `delhi_psi.config.ENUM_KEYS: tuple[str, ...]`, `delhi_psi.config.ENUMS: dict[str, type[Enum]]`
  - `delhi_psi.config.RESERVED_VALUES: dict[str, dict[str, str]]`, `delhi_psi.config.RESERVED_KEYS: dict[str, str]`
  - `StrEnum`s `AdjacencyRule`, `BarrierRule`, `RoadsFormula`, `ExclusionStage`, `AbsentNeighbor`, `Denominator` — members compare equal to their string values (so they pass straight into the math functions) and `str(member)` / `f"{member}"` give the bare value (so `name_template.format(...)` yields `delhi_psi_code-2025_pop_2020`, not `…_Denominator.POP_…`)
  - frozen dataclasses `CrsConfig`, `PathsConfig`, `LayerSpec`, `PopulationSpec`, `LayersConfig`, `ServicesConfig`, `AdjacencyConfig`, `BarrierConfig`, `DecayConfig`, `ExclusionConfig`, `MethodologyConfig`, `ValidateConfig`, `OutputsConfig`, `Config`
  - `delhi_psi.config.PROFILES_DIR: Path`, `shipped_profiles() -> list[str]`
  - `delhi_psi.config.load_config(profile_or_path: str | Path, *, data_dir: str | None = None, out_dir: str | None = None) -> Config`

- [ ] **Step 1: Rewrite `pyproject.toml`**

Replace the whole file with:

```toml
[project]
name = "delhi-spatial-index"
version = "0.3.0"
description = "Delhi Public Services Index (PSI) pipeline"
authors = [
    { name = "Bob Bell", email = "bwbelljr@gmail.com" },
]
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "geopandas>=1.1",
    "pandas>=2.3",
    "shapely>=2.1",
    "matplotlib>=3.10",
    "pyproj>=3.7",
    "tqdm>=4.67",
    "joblib>=1.5",
    "pyyaml>=6.0",
]

[project.scripts]
delhi-psi = "delhi_psi.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
# Required: the project name (delhi-spatial-index) is not the package name.
# This also ships delhi_psi/profiles/*.yaml inside the wheel.
packages = ["delhi_psi"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[dependency-groups]
dev = [
    "pytest>=8.4",
]
```

Notes for the implementer:
- `pyyaml` moves out of the dev group into runtime dependencies (the loader needs it at run time).
- `[project.scripts]` is declared now, per spec § 2, but `delhi_psi/cli.py` does not exist until Task 8. Console-script entry points are resolved lazily, so `uv sync` succeeds; **do not run `delhi-psi` before Task 8.**

- [ ] **Step 2: Create the package skeleton and regenerate the lock**

```bash
mkdir -p delhi_psi/profiles
touch tests/__init__.py
```

Create `delhi_psi/__init__.py`:

```python
"""Delhi Public Services Index (PSI) pipeline.

Public entry points live in `delhi_psi.pipeline` (`preprocess`, `compute`,
`compute_frames`) and `delhi_psi.config` (`load_config`). The math modules
(`geometry`, `neighbors`, `index`) are pure functions with keyword knobs and
never import `config`.
"""

__version__ = "0.3.0"
```

Then:

```bash
uv lock
uv sync
```

Expected: `uv.lock`'s root entry changes from `source = { virtual = "." }` to
`source = { editable = "." }`; `pyyaml` moves into the project's dependency
list. Verify:

```bash
grep -n 'editable = "."' uv.lock
uv run python -c "import delhi_psi, yaml; print(delhi_psi.__version__)"
```
Expected: a match for the editable line, then `0.3.0`.

Confirm the editable install put the repo root on `sys.path`:

```bash
cd /tmp && uv run --project ~/delhi_spatial_index python -c "import tests, scripts, delhi_psi; print('root importable')"
```
Expected: `root importable`.

- [ ] **Step 3: Drop the now-dead bootstrap from the fixture generator**

In `scripts/generate_production_fixtures.py`, delete these lines:

```python
import sys                       # (the import line, at the top)

# ... and this whole block, further down:
# Bootstrap: the package is not installed yet at migration step 0, so the
# repo root is not on sys.path when this script is run by path (the CI drift
# guard does exactly that). Removed in migration step 1, once `uv sync`
# installs the project editable and puts the root on sys.path for good.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
```

and drop the `# noqa: E402` from the `from tests.test_oracle import …` line, which is now an ordinary top-level import. Keep `REPO` — `PRODUCTION_DIR` is derived from it.

Run: `uv run python scripts/generate_production_fixtures.py && git diff --stat -- tests/fixtures/`
Expected: `wrote …/code-2025.csv` and an **empty** diff (the snapshot is unchanged).

- [ ] **Step 4: Write the failing config test**

Create `tests/test_config.py`:

```python
"""Config schema: defaults, enum validation, reserved values, precedence.

Spec § 3. The reference-pinned enums are generated from ONE table
(config.REFERENCE_KNOBS); tests/test_profiles_match_reference.py reads the
same table, so a value without a reference knob cannot be added silently.
"""
from pathlib import Path

import pytest

from delhi_psi.config import (
    ENUMS, ENUM_KEYS, REFERENCE_KNOBS, RESERVED_KEYS, RESERVED_VALUES,
    Config, ConfigError, load_config, shipped_profiles,
)

MINIMAL = """
profile: minimal
methodology:
  adjacency: {rule: bbox}
  barrier: {rule: global_asymmetric, combine: any}
  decay: {form: inverse_linear, distance_unit: km}
  roads: decayed
  second_normalization: true
  exclusion: {types: [RV], stage: post_neighbors, absent_neighbor: swallowed}
"""


def write(tmp_path, text, name="p.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_both_profiles_ship():
    assert sorted(shipped_profiles()) == ["code-2025", "manuscript"]


def test_profile_loads_by_name_and_by_path(tmp_path):
    by_name = load_config("code-2025", data_dir=str(tmp_path))
    from delhi_psi.config import PROFILES_DIR
    by_path = load_config(PROFILES_DIR / "code-2025.yaml",
                          data_dir=str(tmp_path))
    assert isinstance(by_name, Config)
    assert by_name == by_path


def test_defaults_equal_code_2025(tmp_path):
    """Every non-methodology key defaults to the code-2025 value (§ 3)."""
    minimal = load_config(write(tmp_path, MINIMAL), data_dir=str(tmp_path))
    full = load_config("code-2025", data_dir=str(tmp_path))
    assert minimal.profile == "minimal" and full.profile == "code-2025"
    assert minimal.crs == full.crs
    assert minimal.layers == full.layers
    assert minimal.services == full.services
    assert minimal.validate == full.validate
    assert minimal.outputs == full.outputs
    assert minimal.paths.neighbors_artifact == full.paths.neighbors_artifact
    # methodology was written out in full, so it matches too
    assert minimal.methodology == full.methodology


def test_methodology_is_required_in_full(tmp_path):
    partial = MINIMAL.replace("  roads: decayed\n", "")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, partial))
    assert "methodology.roads" in str(exc.value)


def test_profile_key_is_required(tmp_path):
    without = MINIMAL.replace("profile: minimal\n", "")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, without))
    assert "profile" in str(exc.value)


def test_unknown_key_is_rejected(tmp_path):
    text = MINIMAL + "\nnonsense: 1\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert "nonsense" in str(exc.value)


def test_unknown_nested_key_is_rejected(tmp_path):
    text = MINIMAL.replace("  roads: decayed",
                           "  roads: decayed\n  bogus_knob: 1")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert "methodology.bogus_knob" in str(exc.value)


@pytest.mark.parametrize("key,bad", [
    ("methodology.adjacency.rule", "  adjacency: {rule: diagonal}"),
    ("methodology.barrier.rule", "  barrier: {rule: sideways, combine: any}"),
    ("methodology.roads", "  roads: sideways"),
    ("methodology.exclusion.stage",
     "  exclusion: {types: [RV], stage: midway, absent_neighbor: swallowed}"),
    ("methodology.exclusion.absent_neighbor",
     "  exclusion: {types: [RV], stage: post_neighbors, absent_neighbor: maybe}"),
])
def test_out_of_enum_names_key_and_allowed_values(tmp_path, key, bad):
    line_start = bad.split(":")[0]
    text = "\n".join(bad if line.startswith(line_start) else line
                     for line in MINIMAL.splitlines())
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert key in message
    for allowed in REFERENCE_KNOBS[key]:
        assert str(allowed) in message


def test_bad_denominator_names_key_and_allowed_values(tmp_path):
    text = MINIMAL + "\noutputs: {denominators: [households]}\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert "outputs.denominators" in str(exc.value)
    assert "pop" in str(exc.value) and "popdensity" in str(exc.value)


def test_reserved_partial_weighted(tmp_path):
    text = MINIMAL.replace("  barrier: {rule: global_asymmetric, combine: any}",
                           "  barrier: {rule: partial_weighted, combine: any}")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert str(exc.value).endswith(
        RESERVED_VALUES["methodology.barrier.rule"]["partial_weighted"])


def test_reserved_denominator_one(tmp_path):
    text = MINIMAL + "\noutputs: {denominators: [one]}\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert str(exc.value).endswith(
        RESERVED_VALUES["outputs.denominators[]"]["one"])


@pytest.mark.parametrize("value", ["reported", "all", "true"])
def test_reserved_key_minmax_universe_rejects_every_value(tmp_path, value):
    """A KNOWN optional key: any value takes the reserved path, never the
    unknown-key path (spec § 3)."""
    text = MINIMAL.replace(
        "  exclusion: {types: [RV], stage: post_neighbors, "
        "absent_neighbor: swallowed}",
        "  exclusion: {types: [RV], stage: post_neighbors, "
        f"absent_neighbor: swallowed, minmax_universe: {value}}}")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "unknown key" not in message
    assert message.endswith(RESERVED_KEYS["methodology.exclusion.minmax_universe"])
    assert "A.2" in message


def test_enums_are_generated_from_the_reference_table():
    assert set(ENUMS) == set(ENUM_KEYS)
    for key in ENUM_KEYS:
        assert {member.value for member in ENUMS[key]} == set(REFERENCE_KNOBS[key])


def test_enum_members_compare_equal_to_their_strings():
    from delhi_psi.config import AdjacencyRule, Denominator
    assert AdjacencyRule.BBOX == "bbox"
    # str()/format() must give the BARE value: outputs.name_template is
    # rendered with .format(denominator=...), and a plain `Enum` with a str
    # mixin would render "Denominator.POP" there.
    assert str(Denominator.POP) == "pop"
    assert f"{Denominator.POPDENSITY}" == "popdensity"
    assert "delhi_psi_{profile}_{denominator}_2020".format(
        profile="code-2025", denominator=Denominator.POP) == \
        "delhi_psi_code-2025_pop_2020"


def test_data_dir_precedence_flag_beats_env_beats_yaml(tmp_path, monkeypatch):
    yaml_dir = tmp_path / "from_yaml"
    env_dir = tmp_path / "from_env"
    flag_dir = tmp_path / "from_flag"
    text = MINIMAL + f"\npaths: {{data_dir: {yaml_dir}}}\n"
    path = write(tmp_path, text)

    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    assert load_config(path).paths.data_dir == yaml_dir

    monkeypatch.setenv("DELHI_DATA_DIR", str(env_dir))
    assert load_config(path).paths.data_dir == env_dir

    assert load_config(path, data_dir=str(flag_dir)).paths.data_dir == flag_dir


def test_data_dir_falls_back_to_home_delhi_data(tmp_path, monkeypatch):
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.paths.data_dir == Path("~/delhi_data").expanduser()


def test_out_dir_defaults_to_data_dir_and_flag_wins(tmp_path, monkeypatch):
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    data = tmp_path / "data"
    out = tmp_path / "out"
    cfg = load_config(write(tmp_path, MINIMAL), data_dir=str(data))
    assert cfg.paths.out_dir == data
    cfg = load_config(write(tmp_path, MINIMAL), data_dir=str(data),
                      out_dir=str(out))
    assert cfg.paths.out_dir == out
    # load_config resolves paths only; it never creates directories
    assert not out.exists()


def test_unknown_profile_name_lists_the_shipped_ones():
    with pytest.raises(ConfigError) as exc:
        load_config("no-such-profile")
    assert "code-2025" in str(exc.value) and "manuscript" in str(exc.value)
```

- [ ] **Step 5: Run the test to verify it fails**

Run: `uv run pytest tests/test_config.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'delhi_psi.config'`.

- [ ] **Step 6: Write the two profile YAMLs**

Create `delhi_psi/profiles/code-2025.yaml` (spec § 3, verbatim):

```yaml
profile: code-2025
crs: {epsg: 7760}
paths:
  data_dir: ~/delhi_data            # overridable: --data-dir, DELHI_DATA_DIR
  out_dir: ~/delhi_data             # defaults to data_dir, as today; overridable: --out-dir
  neighbors_artifact: colonies_neighbors.joblib
layers:
  settlements: {path: uso_update_sep2021/uso_update_sep2021.shp,
                id_col: USO_AREA_U, type_col: USO_FINAL}
  population:  {path: pop_colony_wp_2020_jjc_adjusted.csv,
                id_col: uso_area_u, value_col: population,
                missing: drop}      # drop | error  (see below)
  bounds: delhi_bounds_buffer/delhi_bounds_buffer.shp
  ndmc_center: ndmc_center7760/ndmc_center7760.shp   # -> ndmc_dist_km column
  barriers: {canal: Barrier_Clip/Canal/Canal.shp,
             railway: Barrier_Clip/Railway/Railway_Line.shp,
             drain: Barrier_Clip/Drain/Major_Drain.shp}
services:
  point: {bank: Public Services/Banking/Banking.shp,
          health: Public Services/Health/Health.shp,
          police: Public Services/Police/Police Station.shp,
          ration: Public Services/Ration/Ration.shp,
          school: Public Services/School/schools7760.shp,
          transport: Public Services/Transport/Transport.shp}
  line:  {road: Public Services/Major Road/Road.shp}   # amount column: road_length
methodology:
  adjacency: {rule: bbox}           # rule: bbox | touch
  barrier:
    rule: global_asymmetric         # global_asymmetric | pairwise
                                    # partial_weighted: reserved (spec 4)
    combine: any                    # any | [layer names]; which flags OR into `barrier`
                                    # (every configured layer's flag column is always computed)
  decay: {form: inverse_linear, distance_unit: km}   # 1/(1+d); only value in 3A
  roads: decayed                    # decayed | eq4_own_only
  second_normalization: true        # norm_psi = minmax(unnorm_psi); column absent when false
  exclusion:
    types: [RV]                     # raw USO_FINAL strings until 3B adds `categories:`
    stage: post_neighbors           # post_neighbors | pre_neighbors
    absent_neighbor: swallowed      # swallowed | contributes
                                    # minmax_universe: reserved (spec 9)
validate:
  max_missing_population: 15        # today's real-data count; compute raises above it
outputs:
  denominators: [pop, popdensity]   # one compute run per entry; each in {pop, popdensity}
                                    # one: reserved (no reference rule yet)
  formats: [csv, shp, joblib]
  name_template: "delhi_psi_{profile}_{denominator}_2020"
```

Create `delhi_psi/profiles/manuscript.yaml` (spec § 4, verbatim):

```yaml
profile: manuscript
methodology:
  adjacency: {rule: touch}
  barrier: {rule: pairwise, combine: any}
  decay: {form: inverse_linear, distance_unit: km}   # manuscript is silent on the unit (spec 8)
  roads: eq4_own_only
  second_normalization: false                        # no norm_psi column
  exclusion: {types: [], stage: post_neighbors, absent_neighbor: contributes}
outputs: {denominators: [pop], formats: [csv], name_template: "delhi_psi_{profile}_{denominator}_2020"}
```

Both parse under `yaml.safe_load` as written (verified). `manuscript.yaml` omits `crs`/`paths`/`layers`/`services`/`validate`, so they take the `code-2025` defaults per § 3.

- [ ] **Step 7: Write `config.py`**

Create `delhi_psi/config.py`:

```python
"""Config schema (spec § 3): one YAML per profile, frozen dataclasses.

Required keys: `profile` and the whole `methodology` block — a profile is a
complete statement of method, never inherited. Everything else defaults to
the `code-2025` values. Unknown keys, missing required keys and out-of-enum
values raise ConfigError naming the key and the allowed values.

The reference-pinned enums are generated from ONE table, REFERENCE_KNOBS,
which maps each config value to its `tests.reference_impl.compute_city` knob.
tests/test_profiles_match_reference.py reads the same table, so a value with
no reference knob cannot be added to a reference-pinned key.
"""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from delhi_psi.io import resolve_data_dir, out_dir_path

PROFILES_DIR = Path(__file__).resolve().parent / "profiles"


class ConfigError(ValueError):
    """Unknown key, missing required key, out-of-enum or reserved value."""


# --- the single enum table (spec § 3) ----------------------------------
# dotted key -> {config value: reference knob value}
REFERENCE_KNOBS = {
    "methodology.adjacency.rule": {"bbox": "bbox", "touch": "border"},
    "methodology.barrier.rule": {"global_asymmetric": "global",
                                 "pairwise": "pair"},
    "methodology.roads": {"decayed": "decayed", "eq4_own_only": "eq4"},
    "methodology.second_normalization": {True: True, False: False},
    "methodology.exclusion.stage": {"post_neighbors": False,
                                    "pre_neighbors": True},
    "methodology.exclusion.absent_neighbor": {"swallowed": "swallowed",
                                              "contributes": "contributes"},
    "outputs.denominators[]": {"pop": "pop", "popdensity": "popdensity"},
}

# `second_normalization` is a bool, not an enum; every other reference-pinned
# key gets a str-valued Enum generated from the table above.
ENUM_KEYS = (
    "methodology.adjacency.rule",
    "methodology.barrier.rule",
    "methodology.roads",
    "methodology.exclusion.stage",
    "methodology.exclusion.absent_neighbor",
    "outputs.denominators[]",
)


def _make_enum(name, key):
    # StrEnum (not Enum + str mixin): its members compare equal to their
    # string values AND str()/format() return the bare value, which
    # outputs.name_template.format(denominator=...) depends on.
    return StrEnum(name, {value.upper(): value
                          for value in REFERENCE_KNOBS[key]})


AdjacencyRule = _make_enum("AdjacencyRule", "methodology.adjacency.rule")
BarrierRule = _make_enum("BarrierRule", "methodology.barrier.rule")
RoadsFormula = _make_enum("RoadsFormula", "methodology.roads")
ExclusionStage = _make_enum("ExclusionStage", "methodology.exclusion.stage")
AbsentNeighbor = _make_enum("AbsentNeighbor",
                            "methodology.exclusion.absent_neighbor")
Denominator = _make_enum("Denominator", "outputs.denominators[]")

ENUMS = {
    "methodology.adjacency.rule": AdjacencyRule,
    "methodology.barrier.rule": BarrierRule,
    "methodology.roads": RoadsFormula,
    "methodology.exclusion.stage": ExclusionStage,
    "methodology.exclusion.absent_neighbor": AbsentNeighbor,
    "outputs.denominators[]": Denominator,
}

# --- reserved values and keys (spec §§ 3, 4, 9) ------------------------
RESERVED_VALUES = {
    "methodology.barrier.rule": {
        "partial_weighted":
            "reserved: w_ij = 1 - L_blocked/L_shared "
            "(docs/oracle/suggested-fixes-memo.md § 2) is config-ready but "
            "reference-pending. Unblock it by adding the reference rule to "
            "tests/reference_impl.py, a hand anchor in "
            "docs/oracle/derivation-worksheet.md, and regenerating "
            "tests/fixtures/oraculum/expected_values.csv (cycle 3C).",
    },
    "outputs.denominators[]": {
        "one":
            "reserved: production supports denom='one' but the reference does "
            "not. Unblock it by adding `denom == \"one\"` to "
            "tests.reference_impl.compute_city and regenerating "
            "tests/fixtures/oraculum/expected_values.csv first.",
    },
}

RESERVED_KEYS = {
    "methodology.exclusion.minmax_universe":
        "reserved: Open Decision A.2 — whether Eq. 2's min/max spans reported "
        "settlements only or all settlements. There is no knob for it in the "
        "reference implementation or in production. Unblock it by putting the "
        "question to Raj (DEL-13) and adding the reference rule.",
}


# --- dataclasses -------------------------------------------------------
@dataclass(frozen=True)
class CrsConfig:
    epsg: int = 7760


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    out_dir: Path
    neighbors_artifact: str = "colonies_neighbors.joblib"


@dataclass(frozen=True)
class LayerSpec:
    path: str
    id_col: str | None = None
    type_col: str | None = None


@dataclass(frozen=True)
class PopulationSpec:
    path: str
    id_col: str
    value_col: str
    missing: str = "drop"          # drop | error


@dataclass(frozen=True)
class LayersConfig:
    settlements: LayerSpec
    population: PopulationSpec
    bounds: str
    ndmc_center: str | None
    barriers: dict


@dataclass(frozen=True)
class ServicesConfig:
    point: dict
    line: dict


@dataclass(frozen=True)
class AdjacencyConfig:
    rule: AdjacencyRule


@dataclass(frozen=True)
class BarrierConfig:
    rule: BarrierRule
    combine: object                # "any" or a tuple of layer names


@dataclass(frozen=True)
class DecayConfig:
    form: str
    distance_unit: str


@dataclass(frozen=True)
class ExclusionConfig:
    types: tuple
    stage: ExclusionStage
    absent_neighbor: AbsentNeighbor


@dataclass(frozen=True)
class MethodologyConfig:
    adjacency: AdjacencyConfig
    barrier: BarrierConfig
    decay: DecayConfig
    roads: RoadsFormula
    second_normalization: bool
    exclusion: ExclusionConfig


@dataclass(frozen=True)
class ValidateConfig:
    max_missing_population: int = 15


@dataclass(frozen=True)
class OutputsConfig:
    denominators: tuple = (Denominator.POP, Denominator.POPDENSITY)
    formats: tuple = ("csv", "shp", "joblib")
    name_template: str = "delhi_psi_{profile}_{denominator}_2020"


@dataclass(frozen=True)
class Config:
    profile: str
    methodology: MethodologyConfig
    crs: CrsConfig
    paths: PathsConfig
    layers: LayersConfig
    services: ServicesConfig
    validate: ValidateConfig
    outputs: OutputsConfig


# --- defaults for every non-methodology block (spec § 3) ---------------
DEFAULT_LAYERS = {
    "settlements": {"path": "uso_update_sep2021/uso_update_sep2021.shp",
                    "id_col": "USO_AREA_U", "type_col": "USO_FINAL"},
    "population": {"path": "pop_colony_wp_2020_jjc_adjusted.csv",
                   "id_col": "uso_area_u", "value_col": "population",
                   "missing": "drop"},
    "bounds": "delhi_bounds_buffer/delhi_bounds_buffer.shp",
    "ndmc_center": "ndmc_center7760/ndmc_center7760.shp",
    "barriers": {"canal": "Barrier_Clip/Canal/Canal.shp",
                 "railway": "Barrier_Clip/Railway/Railway_Line.shp",
                 "drain": "Barrier_Clip/Drain/Major_Drain.shp"},
}
DEFAULT_SERVICES = {
    "point": {"bank": "Public Services/Banking/Banking.shp",
              "health": "Public Services/Health/Health.shp",
              "police": "Public Services/Police/Police Station.shp",
              "ration": "Public Services/Ration/Ration.shp",
              "school": "Public Services/School/schools7760.shp",
              "transport": "Public Services/Transport/Transport.shp"},
    "line": {"road": "Public Services/Major Road/Road.shp"},
}
DEFAULT_CRS = {"epsg": 7760}
DEFAULT_PATHS = {"data_dir": "~/delhi_data", "out_dir": None,
                 "neighbors_artifact": "colonies_neighbors.joblib"}
DEFAULT_VALIDATE = {"max_missing_population": 15}
DEFAULT_OUTPUTS = {"denominators": ["pop", "popdensity"],
                   "formats": ["csv", "shp", "joblib"],
                   "name_template": "delhi_psi_{profile}_{denominator}_2020"}

TOP_LEVEL_KEYS = ("profile", "crs", "paths", "layers", "services",
                  "methodology", "validate", "outputs")


# --- validation helpers ------------------------------------------------
def _reject_unknown(mapping, allowed, prefix):
    for key in mapping:
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if dotted in RESERVED_KEYS:
            raise ConfigError(f"{dotted}: {RESERVED_KEYS[dotted]}")
        if key not in allowed:
            raise ConfigError(
                f"unknown key {dotted!r}; allowed keys here: "
                f"{sorted(allowed)}")


def _require(mapping, key, prefix):
    dotted = f"{prefix}.{key}" if prefix else str(key)
    if key not in mapping:
        raise ConfigError(f"missing required key {dotted!r}")
    return mapping[key]


def _coerce_enum(key, value):
    reserved = RESERVED_VALUES.get(key, {})
    if value in reserved:
        raise ConfigError(f"{key}: {reserved[value]}")
    try:
        return ENUMS[key](value)
    except ValueError:
        allowed = sorted(str(v) for v in REFERENCE_KNOBS[key])
        extra = sorted(reserved)
        note = f" (reserved: {extra})" if extra else ""
        raise ConfigError(
            f"{key}: {value!r} is not allowed; allowed values: "
            f"{allowed}{note}") from None


def _bool(key, value):
    if not isinstance(value, bool):
        raise ConfigError(f"{key}: {value!r} is not allowed; "
                          "allowed values: [True, False]")
    return value


# --- loader ------------------------------------------------------------
def shipped_profiles():
    return sorted(p.stem for p in PROFILES_DIR.glob("*.yaml"))


def _profile_path(profile_or_path):
    candidate = Path(profile_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        if not candidate.exists():
            raise ConfigError(f"config file not found: {candidate}")
        return candidate
    shipped = PROFILES_DIR / f"{profile_or_path}.yaml"
    if not shipped.exists():
        raise ConfigError(
            f"unknown profile {str(profile_or_path)!r}; shipped profiles: "
            f"{shipped_profiles()} (or pass a path to a .yaml file)")
    return shipped


def _methodology(raw):
    _reject_unknown(raw, {"adjacency", "barrier", "decay", "roads",
                          "second_normalization", "exclusion"}, "methodology")

    adjacency_raw = _require(raw, "adjacency", "methodology")
    _reject_unknown(adjacency_raw, {"rule"}, "methodology.adjacency")
    adjacency = AdjacencyConfig(rule=_coerce_enum(
        "methodology.adjacency.rule",
        _require(adjacency_raw, "rule", "methodology.adjacency")))

    barrier_raw = _require(raw, "barrier", "methodology")
    _reject_unknown(barrier_raw, {"rule", "combine"}, "methodology.barrier")
    combine = _require(barrier_raw, "combine", "methodology.barrier")
    if combine != "any":
        if not isinstance(combine, list) or not all(
                isinstance(item, str) for item in combine):
            raise ConfigError(
                "methodology.barrier.combine: expected 'any' or a list of "
                f"layer names, got {combine!r}")
        combine = tuple(combine)
    barrier = BarrierConfig(
        rule=_coerce_enum("methodology.barrier.rule",
                          _require(barrier_raw, "rule", "methodology.barrier")),
        combine=combine)

    decay_raw = _require(raw, "decay", "methodology")
    _reject_unknown(decay_raw, {"form", "distance_unit"}, "methodology.decay")
    form = _require(decay_raw, "form", "methodology.decay")
    unit = _require(decay_raw, "distance_unit", "methodology.decay")
    if form != "inverse_linear":
        raise ConfigError(f"methodology.decay.form: {form!r} is not allowed; "
                          "allowed values: ['inverse_linear']")
    if unit != "km":
        raise ConfigError(
            f"methodology.decay.distance_unit: {unit!r} is not allowed; "
            "allowed values: ['km']")
    decay = DecayConfig(form=form, distance_unit=unit)

    exclusion_raw = _require(raw, "exclusion", "methodology")
    _reject_unknown(exclusion_raw, {"types", "stage", "absent_neighbor"},
                    "methodology.exclusion")
    types = _require(exclusion_raw, "types", "methodology.exclusion")
    if not isinstance(types, list) or not all(
            isinstance(item, str) for item in types):
        raise ConfigError("methodology.exclusion.types: expected a list of "
                          f"settlement-type strings, got {types!r}")
    exclusion = ExclusionConfig(
        types=tuple(types),
        stage=_coerce_enum("methodology.exclusion.stage",
                           _require(exclusion_raw, "stage",
                                    "methodology.exclusion")),
        absent_neighbor=_coerce_enum(
            "methodology.exclusion.absent_neighbor",
            _require(exclusion_raw, "absent_neighbor",
                     "methodology.exclusion")))

    return MethodologyConfig(
        adjacency=adjacency,
        barrier=barrier,
        decay=decay,
        roads=_coerce_enum("methodology.roads",
                           _require(raw, "roads", "methodology")),
        second_normalization=_bool(
            "methodology.second_normalization",
            _require(raw, "second_normalization", "methodology")),
        exclusion=exclusion)


def _layers(raw):
    merged = {**DEFAULT_LAYERS, **raw}
    _reject_unknown(merged, set(DEFAULT_LAYERS), "layers")
    settlements = {**DEFAULT_LAYERS["settlements"], **merged["settlements"]}
    _reject_unknown(settlements, {"path", "id_col", "type_col"},
                    "layers.settlements")
    population = {**DEFAULT_LAYERS["population"], **merged["population"]}
    _reject_unknown(population, {"path", "id_col", "value_col", "missing"},
                    "layers.population")
    if population["missing"] not in ("drop", "error"):
        raise ConfigError(
            f"layers.population.missing: {population['missing']!r} is not "
            "allowed; allowed values: ['drop', 'error']")
    return LayersConfig(
        settlements=LayerSpec(**settlements),
        population=PopulationSpec(**population),
        bounds=merged["bounds"],
        ndmc_center=merged["ndmc_center"],
        barriers=dict(merged["barriers"]))


def _services(raw):
    merged = {**DEFAULT_SERVICES, **raw}
    _reject_unknown(merged, set(DEFAULT_SERVICES), "services")
    return ServicesConfig(point=dict(merged["point"]),
                          line=dict(merged["line"]))


def _outputs(raw):
    merged = {**DEFAULT_OUTPUTS, **raw}
    _reject_unknown(merged, set(DEFAULT_OUTPUTS), "outputs")
    denominators = merged["denominators"]
    if not isinstance(denominators, list) or not denominators:
        raise ConfigError("outputs.denominators: expected a non-empty list, "
                          f"got {denominators!r}")
    formats = merged["formats"]
    for fmt in formats:
        if fmt not in ("csv", "shp", "joblib"):
            raise ConfigError(f"outputs.formats: {fmt!r} is not allowed; "
                              "allowed values: ['csv', 'shp', 'joblib']")
    return OutputsConfig(
        denominators=tuple(_coerce_enum("outputs.denominators[]", d)
                           for d in denominators),
        formats=tuple(formats),
        name_template=merged["name_template"])


def _paths(raw, cli_data_dir, cli_out_dir):
    merged = {**DEFAULT_PATHS, **raw}
    _reject_unknown(merged, set(DEFAULT_PATHS), "paths")
    yaml_data_dir = merged["data_dir"]
    if cli_data_dir:
        data_dir = Path(cli_data_dir).expanduser()
    elif os.environ.get("DELHI_DATA_DIR"):
        data_dir = Path(os.environ["DELHI_DATA_DIR"]).expanduser()
    else:
        data_dir = resolve_data_dir(yaml_data_dir)
    out_dir = out_dir_path(cli_out_dir or merged["out_dir"], data_dir)
    return PathsConfig(data_dir=data_dir, out_dir=out_dir,
                       neighbors_artifact=merged["neighbors_artifact"])


def load_config(profile_or_path, *, data_dir=None, out_dir=None):
    """Load a shipped profile by name, or a YAML file by path.

    Path precedence: --data-dir/--out-dir argument > DELHI_DATA_DIR env var
    > the YAML value > ~/delhi_data. Resolution only — no directory is
    created here (the pipeline stages do that).
    """
    path = _profile_path(profile_or_path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    _reject_unknown(raw, set(TOP_LEVEL_KEYS), "")

    return Config(
        profile=_require(raw, "profile", ""),
        methodology=_methodology(_require(raw, "methodology", "")),
        crs=CrsConfig(**{**DEFAULT_CRS, **raw.get("crs", {})}),
        paths=_paths(raw.get("paths", {}), data_dir, out_dir),
        layers=_layers(raw.get("layers", {})),
        services=_services(raw.get("services", {})),
        validate=ValidateConfig(
            **{**DEFAULT_VALIDATE, **raw.get("validate", {})}),
        outputs=_outputs(raw.get("outputs", {})))
```

> `config.py` imports `resolve_data_dir` and `out_dir_path` from `delhi_psi.io`, which Task 6 creates. **This task therefore also creates the two path helpers in `delhi_psi/io.py`** — see the next step. `io.py` never imports `config`, so there is no cycle.

- [ ] **Step 8: Create the path half of `delhi_psi/io.py`**

Create `delhi_psi/io.py` with only the path helpers for now (Task 6 adds the read/write functions to this same file):

```python
"""Filesystem seam: path resolution, layer/CSV reads, output writes.

Absorbs scripts/common.py. Never imports delhi_psi.config — the pipeline
passes explicit values.
"""

import os
from pathlib import Path

DEFAULT_DATA_DIR = "~/delhi_data"


def resolve_data_dir(cli_value=None):
    """Resolve the data directory from flag, env var, or default."""
    if cli_value:
        return Path(cli_value).expanduser()
    env_value = os.environ.get("DELHI_DATA_DIR")
    if env_value:
        return Path(env_value).expanduser()
    return Path(DEFAULT_DATA_DIR).expanduser()


def out_dir_path(cli_value, data_dir):
    """Resolve the output directory (default: the data directory). No mkdir."""
    return Path(cli_value).expanduser() if cli_value else Path(data_dir)


def resolve_out_dir(cli_value, data_dir):
    """Resolve the output directory and create it."""
    out_dir = out_dir_path(cli_value, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
```

This is `scripts/common.py`'s behaviour verbatim (`resolve_out_dir` still defaults to `data_dir` and still creates the directory), split so that `load_config` can resolve without creating.

- [ ] **Step 9: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_config.py -q` — Expected: 24 passed.
Run: `uv run pytest -q -W error` — Expected: **104 passed** (80 + 24).

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py \
        delhi_psi/__init__.py delhi_psi/config.py delhi_psi/io.py \
        delhi_psi/profiles/code-2025.yaml delhi_psi/profiles/manuscript.yaml \
        tests/test_config.py scripts/generate_production_fixtures.py
git commit -m "feat: delhi_psi package skeleton, hatchling packaging, config schema (DEL-15 step 1)"
```

---

### Task 3: `delhi_psi/geometry.py` (spec § 5 step 2, part 1)

Move the geometry functions out of `spatial_index_utils.py`, leaving that module in place and **delegating**, so the whole suite stays green with no test edits.

**Files:**
- Create: `delhi_psi/geometry.py`
- Modify: `spatial_index_utils.py` (the moved functions become one-line delegations)
- Test: `tests/test_geometry.py` (new)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `delhi_psi.geometry.row_index(gdf, id_col: str, id_num) -> int`
  - `delhi_psi.geometry.reproject(gdf, epsg_code: int) -> GeoDataFrame`
  - `delhi_psi.geometry.remove_duplicate_geom(gdf, geom_col: str = "geometry") -> GeoDataFrame`
  - `delhi_psi.geometry.bbox_frame(polygon_gdf) -> GeoDataFrame`
  - `delhi_psi.geometry.barrier_flags(polygon_gdf, barriers: dict[str, GeoDataFrame], *, id_col: str = "USO_AREA_U") -> GeoDataFrame`
  - `delhi_psi.geometry.distance_to_point_km(polygon_gdf, point, *, centroid_col: str = "centroid", out_col: str = "ndmc_dist_km") -> GeoDataFrame`

- [ ] **Step 1: Write the failing test**

Create `tests/test_geometry.py`:

```python
"""delhi_psi.geometry — the moved geometry primitives, pinned on fixtures."""
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box

from delhi_psi import geometry
from tests.oraculum_fixtures import load_barriers, load_settlements


def test_row_index_finds_the_row():
    city = load_settlements()
    idx = geometry.row_index(city, "USO_AREA_U", "C")
    assert city.loc[idx, "USO_AREA_U"] == "C"


def test_reproject_changes_crs_and_moves_geometry():
    city = load_settlements()
    out = geometry.reproject(city, 4326)
    assert out.crs.to_epsg() == 4326
    assert out.geometry.iloc[0].bounds != city.geometry.iloc[0].bounds


def test_remove_duplicate_geom_keeps_first_occurrence():
    geom = box(0, 0, 1, 1)
    gdf = gpd.GeoDataFrame({"name": ["a", "b", "c"]},
                           geometry=[geom, box(2, 2, 3, 3), box(0, 0, 1, 1)],
                           crs="EPSG:7760")
    out = geometry.remove_duplicate_geom(gdf)
    assert list(out["name"]) == ["a", "b"]


def test_bbox_frame_is_the_exact_envelope():
    city = load_settlements()
    boxes = geometry.bbox_frame(city)
    assert list(boxes["USO_AREA_U"]) == list(city["USO_AREA_U"])
    for original, produced in zip(city.geometry, boxes.geometry):
        assert produced.equals(original.envelope)
    assert boxes.crs == city.crs


def test_barrier_flags_one_column_per_layer():
    city = load_settlements()
    out = geometry.barrier_flags(city, {"canal": load_barriers()})
    flagged = set(out.loc[out["canal"], "USO_AREA_U"])
    # the fixture canal is a strict interior sub-segment of the A|D edge
    assert flagged == {"A", "D"}


def test_barrier_flags_missing_layer_is_all_false():
    city = load_settlements()
    empty = gpd.GeoDataFrame(
        {"name": ["far"]},
        geometry=[LineString([(9_000_000, 9_000_000), (9_000_001, 9_000_001)])],
        crs=city.crs)
    out = geometry.barrier_flags(city, {"railway": empty})
    assert not out["railway"].any()


def test_distance_to_point_km_is_metres_over_1000():
    city = load_settlements()
    city = city.copy()
    city["centroid"] = city.centroid
    centre = Point(1_001_500, 1_001_500)
    out = geometry.distance_to_point_km(city, centre)
    row = out[out["USO_AREA_U"] == "A"].iloc[0]
    assert out["ndmc_dist_km"].dtype.kind == "f"
    assert row["ndmc_dist_km"] == pytest.approx(
        centre.distance(row["centroid"]) / 1000, abs=1e-12)


def test_shim_still_exposes_the_old_names():
    """spatial_index_utils keeps working until the deletion task."""
    import spatial_index_utils

    city = load_settlements()
    assert spatial_index_utils.get_row_index(city, "USO_AREA_U", "C") == \
        geometry.row_index(city, "USO_AREA_U", "C")
    for original, produced in zip(
            spatial_index_utils.create_bbox_gdf(city.copy()).geometry,
            geometry.bbox_frame(city).geometry):
        assert produced.equals(original)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_geometry.py -q`
Expected: collection error — `ImportError: cannot import name 'geometry' from 'delhi_psi'`.

- [ ] **Step 3: Write `delhi_psi/geometry.py`**

Every body below is copied verbatim from `spatial_index_utils.py`; only `print` becomes `logging` and parameter names change.

```python
"""Geometry primitives: reprojection, deduplication, bounding boxes,
barrier flags, point distances.

Pure functions with explicit keyword arguments — never imports
delhi_psi.config. The math is copied verbatim from spatial_index_utils.py;
the O(n^2) remove_duplicate_geom algorithm is deliberately unchanged
(spec § 6).
"""

import logging
from itertools import islice

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import box
from tqdm import tqdm

log = logging.getLogger(__name__)


def row_index(gdf, id_col, id_num):
    """Row index of a GeoDataFrame given a unique id."""
    return gdf[gdf[id_col] == id_num].index.values[0]


def reproject(gdf, epsg_code):
    """Reproject to the CRS with this EPSG code (WKT, as production does)."""
    target_projection = CRS.from_epsg(epsg_code).to_wkt()
    reprojected_gdf = gdf.to_crs(target_projection)
    log.debug("GeoDataFrame now has CRS %s", reprojected_gdf.crs)
    return reprojected_gdf


def remove_duplicate_geom(gdf, geom_col="geometry"):
    """Remove rows with duplicate geometries (Shapely `equals`), O(n^2).

    Keeps the first occurrence. Returns a frame with a NEW index.
    """
    old_size = len(gdf)
    gdf["not_duplicate"] = True

    for idx, row in tqdm(gdf.iterrows()):
        row_geom = row[geom_col]
        for idx2, row2 in islice(gdf.iterrows(), idx + 1, None):
            other_geom = row2[geom_col]
            if row_geom.equals(other_geom):
                gdf.loc[idx2, "not_duplicate"] = False

    gdf = gdf[gdf["not_duplicate"]]
    gdf = gdf.drop(columns=["not_duplicate"])
    gdf = gdf.reset_index()
    log.info("deduplicated %d rows to %d", old_size, len(gdf))
    return gdf


def bbox_frame(polygon_gdf):
    """GeoDataFrame whose geometry is each row's bounding box."""
    gdf_bbox = gpd.GeoDataFrame(
        pd.concat([polygon_gdf, polygon_gdf.bounds], axis=1))
    gdf_bbox["bbox"] = None
    for idx, row in gdf_bbox.iterrows():
        row_bbox = box(row["minx"], row["miny"], row["maxx"], row["maxy"])
        gdf_bbox.loc[idx, "bbox"] = row_bbox
    gdf_bbox = gdf_bbox.drop(
        columns=["geometry", "minx", "miny", "maxx", "maxy"])
    gdf_bbox = gdf_bbox.rename(columns={"bbox": "geometry"})
    gdf_bbox = gdf_bbox.set_geometry("geometry")
    gdf_bbox.crs = polygon_gdf.crs
    return gdf_bbox


def _flag_one(polygon_gdf, barrier_gdf, flag_col, id_col):
    polygon_gdf[flag_col] = False
    joined = gpd.sjoin(polygon_gdf, barrier_gdf, how="inner")
    ids_with_intersection = list(joined[id_col].unique())
    for polygon_id in ids_with_intersection:
        idx = polygon_gdf[polygon_gdf[id_col] == polygon_id].index.values[0]
        polygon_gdf.loc[idx, flag_col] = True
    return polygon_gdf


def barrier_flags(polygon_gdf, barriers, *, id_col="USO_AREA_U"):
    """One boolean column per barrier layer, named after the layer.

    Every configured layer's flag column is always computed (spec § 3);
    which of them OR into `barrier` is neighbors.combine_barrier_flags'
    job.
    """
    out = polygon_gdf.copy()
    for name, barrier_gdf in barriers.items():
        out = _flag_one(out, barrier_gdf, name, id_col)
    return out


def distance_to_point_km(polygon_gdf, point, *, centroid_col="centroid",
                         out_col="ndmc_dist_km"):
    """Distance in km from `point` to each row's centroid (the NDMC column)."""
    out = polygon_gdf.copy()
    out[out_col] = 0.0
    for idx, row in out.iterrows():
        out.loc[idx, out_col] = point.distance(row[centroid_col]) / 1000
    return out
```

- [ ] **Step 4: Make `spatial_index_utils.py` delegate**

In `spatial_index_utils.py`, add at the top of the file, after the existing imports:

```python
from delhi_psi import geometry as _geometry
```

Then replace the **bodies** of these five functions (keep the names, signatures and docstrings so every existing caller and test is unaffected):

```python
def get_row_index(polygon_gdf, id_colname, id_num):
    """Get row index of GeoDataFrame given a unique id number"""
    return _geometry.row_index(polygon_gdf, id_colname, id_num)


def reproject_gdf(gdf, epsg_code):
    """Reprojects GeoDataFrame to CRS with EPSG code"""
    return _geometry.reproject(gdf, epsg_code)


def remove_duplicate_geom(gdf, geom_colname='geometry'):
    """Removes rows with duplicate geometries"""
    return _geometry.remove_duplicate_geom(gdf, geom_colname)


def create_bbox_gdf(polygon_gdf):
    """Create GeoDataFrame with bounding box as geometry"""
    return _geometry.bbox_frame(polygon_gdf)


def barrier_intersection(colonies_gdf, barrier_gdf, barrier_colname,
    id_colname="USO_AREA_U"):
    """Add new column indicating intersection with barrier"""
    return _geometry._flag_one(colonies_gdf, barrier_gdf, barrier_colname,
                               id_colname)
```

`barrier_intersection` delegates to `_flag_one` (not `barrier_flags`) because
the old signature takes a single layer and **mutates in place**, which
`scripts/preprocess.py` and `tests/oraculum_fixtures.run_production_chain`
both rely on. That contract is preserved exactly until Task 10 deletes them.

- [ ] **Step 5: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_geometry.py -q` — Expected: 8 passed.
Run: `uv run pytest -q -W error` — Expected: **112 passed** (104 + 8).

The 77 carried-over tests must still pass **unchanged**. If any of them fails,
the move was not verbatim — stop and diff the function against git history.

- [ ] **Step 6: Prove the fixtures did not move**

```bash
uv run python scripts/generate_production_fixtures.py
git diff --exit-code -- tests/fixtures/ && echo NO-DRIFT
```
Expected: `NO-DRIFT`.

- [ ] **Step 7: Commit**

```bash
git add delhi_psi/geometry.py spatial_index_utils.py tests/test_geometry.py
git commit -m "refactor: move geometry primitives to delhi_psi.geometry (DEL-16)"
```

---

### Task 4: `delhi_psi/neighbors.py` (spec § 5 step 2, part 2)

**Files:**
- Create: `delhi_psi/neighbors.py`
- Modify: `spatial_index_utils.py` (delegate the two neighbor functions)
- Test: `tests/test_neighbors.py` (new)

**Interfaces:**
- Consumes: `delhi_psi.geometry.row_index`, `delhi_psi.geometry.bbox_frame`.
- Produces:
  - `delhi_psi.neighbors.combine_barrier_flags(polygon_gdf, *, layers: tuple[str, ...], combine, out_col: str = "barrier") -> GeoDataFrame`
  - `delhi_psi.neighbors.adjacency(polygon_gdf, *, id_col: str = "USO_AREA_U", neighbor_col: str = "nbrs_bbox", rule: str = "bbox") -> GeoDataFrame`
  - `delhi_psi.neighbors.apply_barrier(polygon_gdf, barrier_geoms: list, *, id_col: str = "USO_AREA_U", neighbor_col: str = "nbrs_bbox", rule: str = "global_asymmetric", flag_col: str = "barrier") -> GeoDataFrame`
  - `delhi_psi.neighbors.centroid_distances(polygon_gdf, *, neighbor_col: str = "nbrs_bbox", nbr_dist_col: str = "nbrs_dist_bbox", centroid_col: str = "centroid", id_col: str = "USO_AREA_U") -> GeoDataFrame`

> **Why `adjacency` and `apply_barrier` are split.** Production's
> `add_polygon_neighbors_column_fast` does both in one pass: it builds the
> sjoin neighbour list, then drops every id whose `barrier` flag is True.
> Dropping flagged ids after the list is built is exactly the same set as
> dropping them during the loop, so the split changes nothing — and it is
> what lets `barrier.rule: pairwise` exist at all.

- [ ] **Step 1: Write the failing test**

Create `tests/test_neighbors.py`:

```python
"""delhi_psi.neighbors — adjacency rules, barrier rules, centroid distances.

The `bbox` + `global_asymmetric` combination must reproduce production's
directed lists exactly (the empirical pin from Phase 2); `touch` + `pairwise`
must reproduce the manuscript's symmetric lists from the worksheet.
"""
import pytest

from delhi_psi import geometry, neighbors
from tests.oraculum_fixtures import load_barriers, load_settlements

# docs/oracle/derivation-worksheet.md, "Ideal neighbor lists"
IDEAL_DIRECTED = {"A": {"B", "E"}, "B": {"A", "C", "RV", "E"},
                  "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"E"},
                  "E": {"A", "B", "C", "D", "IND"}, "IND": {"C", "E"}}
# plan 2026-08-17 "Canonical numbers": flagged {A, D} stripped from every list
CODE_DIRECTED = {"A": {"B", "E"}, "B": {"C", "RV", "E"},
                 "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"E"},
                 "E": {"B", "C", "IND"}, "IND": {"C", "E"}}


def prepared():
    city = geometry.barrier_flags(load_settlements(), {"canal": load_barriers()})
    city = neighbors.combine_barrier_flags(city, layers=("canal",),
                                           combine="any")
    city["centroid"] = city.centroid
    return city


def lists_of(frame, col="nbrs_bbox"):
    return {row["USO_AREA_U"]: set(row[col]) for _, row in frame.iterrows()}


def test_combine_any_ors_every_layer():
    city = load_settlements().copy()
    city["canal"] = [True, False, False, False, False, False, False]
    city["railway"] = [False, True, False, False, False, False, False]
    out = neighbors.combine_barrier_flags(city, layers=("canal", "railway"),
                                          combine="any")
    assert list(out["barrier"]) == [True, True, False, False, False, False,
                                    False]


def test_combine_selects_named_layers_only():
    city = load_settlements().copy()
    city["canal"] = [True, False, False, False, False, False, False]
    city["railway"] = [False, True, False, False, False, False, False]
    out = neighbors.combine_barrier_flags(city, layers=("canal", "railway"),
                                          combine=("railway",))
    assert list(out["barrier"]) == [False, True, False, False, False, False,
                                    False]


def test_bbox_adjacency_then_global_barrier_matches_production():
    city = prepared()
    nbrs = neighbors.adjacency(city, rule="bbox")
    nbrs = neighbors.apply_barrier(nbrs, list(load_barriers().geometry),
                                   rule="global_asymmetric")
    assert lists_of(nbrs) == CODE_DIRECTED


def test_touch_adjacency_then_pairwise_barrier_matches_the_manuscript():
    city = prepared()
    nbrs = neighbors.adjacency(city, rule="touch")
    nbrs = neighbors.apply_barrier(nbrs, list(load_barriers().geometry),
                                   rule="pairwise")
    assert lists_of(nbrs) == IDEAL_DIRECTED


def test_touch_adjacency_excludes_bbox_only_neighbours():
    """C and A share no boundary, but A's bbox reaches C under `bbox`."""
    city = prepared()
    touch = lists_of(neighbors.adjacency(city, rule="touch"))
    assert "A" not in touch["C"] and "C" not in touch["A"]


def test_unknown_adjacency_rule_raises_value_error():
    with pytest.raises(ValueError, match="diagonal"):
        neighbors.adjacency(prepared(), rule="diagonal")


def test_unknown_barrier_rule_raises_value_error():
    city = neighbors.adjacency(prepared(), rule="bbox")
    with pytest.raises(ValueError, match="sideways"):
        neighbors.apply_barrier(city, list(load_barriers().geometry),
                                rule="sideways")


def test_centroid_distances_are_km_tuples():
    city = prepared()
    nbrs = neighbors.adjacency(city, rule="bbox")
    nbrs = neighbors.apply_barrier(nbrs, list(load_barriers().geometry),
                                   rule="global_asymmetric")
    nbrs = neighbors.centroid_distances(nbrs)
    row = nbrs[nbrs["USO_AREA_U"] == "B"].iloc[0]
    dist = dict(row["nbrs_dist_bbox"])
    assert dist["E"] == pytest.approx(1.0, abs=1e-9)
    assert dist["RV"] == pytest.approx(1.0, abs=1e-9)


def test_shim_still_matches_the_new_path():
    import spatial_index_utils

    city = prepared()
    old = spatial_index_utils.add_polygon_neighbors_column_fast(
        polygon_gdf=city.copy(),
        right_gdf=geometry.bbox_frame(city.copy()),
        id_colname="USO_AREA_U", neighbor_colname="nbrs_bbox",
        barrier_colname="barrier")
    new = neighbors.apply_barrier(
        neighbors.adjacency(city, rule="bbox"),
        list(load_barriers().geometry), rule="global_asymmetric")
    assert lists_of(old) == lists_of(new)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_neighbors.py -q`
Expected: collection error — `ImportError: cannot import name 'neighbors' from 'delhi_psi'`.

- [ ] **Step 3: Write `delhi_psi/neighbors.py`**

```python
"""Neighbour construction: adjacency rule, barrier rule, centroid distances.

Pure functions with explicit keyword arguments — never imports
delhi_psi.config. The `bbox` adjacency path and the `global_asymmetric`
barrier path are copied verbatim from spatial_index_utils'
add_polygon_neighbors_column_fast (split into two passes, which selects the
same ids); `touch` and `pairwise` implement the manuscript's border-sharing
and pair-severing rules.
"""

import logging

import geopandas as gpd
import numpy as np
from tqdm import tqdm

from delhi_psi.geometry import bbox_frame, row_index

log = logging.getLogger(__name__)


def combine_barrier_flags(polygon_gdf, *, layers, combine, out_col="barrier"):
    """OR the selected per-layer flag columns into `out_col`.

    combine == "any" uses every configured layer; otherwise it is a sequence
    of layer names. Every configured layer's own flag column is left intact.
    """
    out = polygon_gdf.copy()
    selected = tuple(layers) if combine == "any" else tuple(combine)
    unknown = [name for name in selected if name not in layers]
    if unknown:
        raise ValueError(
            f"barrier.combine names layers that are not configured: {unknown}; "
            f"configured layers: {sorted(layers)}")
    flag = None
    for name in selected:
        column = out[name].fillna(False).astype(bool)
        flag = column if flag is None else (flag | column)
    out[out_col] = False if flag is None else flag
    return out


def _adjacency_bbox(polygon_gdf, id_col, neighbor_col):
    """Production's spatial join of polygons against bounding boxes."""
    right_gdf = gpd.GeoDataFrame(bbox_frame(polygon_gdf),
                                 crs=polygon_gdf.crs)
    joined_gdf = gpd.sjoin(polygon_gdf, right_gdf, how="left")

    id_col_left = id_col + "_left"
    id_col_right = id_col + "_right"
    joined_grouped = joined_gdf.groupby(id_col_left)

    out = polygon_gdf.copy()
    out[neighbor_col] = np.empty((len(out), 0)).tolist()

    for group in tqdm(joined_grouped.groups):
        group_list = list(joined_grouped.get_group(group)[id_col_right])
        # a polygon intersects itself
        group_list.remove(group)
        group_idx = row_index(out, id_col, group)
        out.loc[group_idx, neighbor_col].extend(group_list)
    return out


def _adjacency_touch(polygon_gdf, id_col, neighbor_col):
    """Border sharing: the intersection must be a line of positive length."""
    out = polygon_gdf.copy()
    out[neighbor_col] = np.empty((len(out), 0)).tolist()
    geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
    for idx, row in tqdm(out.iterrows(), total=len(out)):
        i = row[id_col]
        for j, other in geoms.items():
            if i == j:
                continue
            shared = geoms[i].intersection(other)
            if not shared.is_empty and shared.length > 0:
                out.loc[idx, neighbor_col].append(j)
    return out


def adjacency(polygon_gdf, *, id_col="USO_AREA_U", neighbor_col="nbrs_bbox",
              rule="bbox"):
    """Directed neighbour lists under `rule` ("bbox" or "touch").

    The column keeps its historical name `nbrs_bbox` under both rules — it is
    part of the July 2025 baseline's column contract (spec § 5).
    """
    if rule == "bbox":
        return _adjacency_bbox(polygon_gdf, id_col, neighbor_col)
    if rule == "touch":
        return _adjacency_touch(polygon_gdf, id_col, neighbor_col)
    raise ValueError(
        f"unknown adjacency rule {rule!r}; allowed values: ['bbox', 'touch']")


def apply_barrier(polygon_gdf, barrier_geoms, *, id_col="USO_AREA_U",
                  neighbor_col="nbrs_bbox", rule="global_asymmetric",
                  flag_col="barrier"):
    """Sever neighbour links across barriers.

    global_asymmetric: drop every neighbour whose `flag_col` is True — the
        production rule (a per-polygon flag, so severing is one-directional).
    pairwise: drop j from i's list when a barrier geometry intersects the
        boundary i and j share — the manuscript rule.
    """
    if rule not in ("global_asymmetric", "pairwise"):
        raise ValueError(
            f"unknown barrier rule {rule!r}; allowed values: "
            "['global_asymmetric', 'pairwise']")
    out = polygon_gdf.copy()
    if not barrier_geoms:
        return out
    geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
    flags = {row[id_col]: bool(row[flag_col]) for _, row in out.iterrows()} \
        if rule == "global_asymmetric" else {}

    for idx, row in out.iterrows():
        i = row[id_col]
        kept = []
        for j in row[neighbor_col]:
            if rule == "global_asymmetric":
                if not flags[j]:
                    kept.append(j)
            else:
                shared = geoms[i].intersection(geoms[j])
                if not any(b.intersects(shared) for b in barrier_geoms):
                    kept.append(j)
        out.at[idx, neighbor_col] = kept
    return out


def centroid_distances(polygon_gdf, *, neighbor_col="nbrs_bbox",
                       nbr_dist_col="nbrs_dist_bbox",
                       centroid_col="centroid", id_col="USO_AREA_U"):
    """Add [(neighbor_id, distance_km), ...] per row (verbatim calc_nbr_dist)."""
    gdf_copy = polygon_gdf.copy()
    gdf_copy[nbr_dist_col] = np.empty((len(gdf_copy), 0)).tolist()

    with tqdm(total=len(gdf_copy)) as pbar:
        for idx, row in gdf_copy.iterrows():
            row_centroid = row[centroid_col]
            neighbor_ids = row[neighbor_col]

            for neighbor_id in neighbor_ids:
                neighbor_row = gdf_copy[gdf_copy[id_col] == neighbor_id]
                neighbor_centroid = neighbor_row[centroid_col].array[0]
                neighbor_distance = row_centroid.distance(neighbor_centroid)
                neighbor_distance = neighbor_distance / 1000
                gdf_copy.loc[idx, nbr_dist_col].append(
                    (neighbor_id, neighbor_distance))

            pbar.update(1)

    return gdf_copy
```

- [ ] **Step 4: Make `spatial_index_utils.py` delegate**

Add near the other delegation import:

```python
from delhi_psi import neighbors as _neighbors
```

Replace these three bodies (`remove_ids_with_barrier` has no new-module
counterpart — `apply_barrier` absorbed it — so keep its verbatim body):

```python
def add_polygon_neighbors_column_fast(polygon_gdf, right_gdf, id_colname,
    neighbor_colname, barrier_colname):
    """Add polygon neighbors based on spatial join"""
    built = _neighbors._adjacency_bbox(polygon_gdf, id_colname,
                                       neighbor_colname)
    out = built.copy()
    for idx, row in out.iterrows():
        out.at[idx, neighbor_colname] = remove_ids_with_barrier(
            id_list=row[neighbor_colname], polygon_gdf=out,
            id_colname=id_colname, barrier_colname=barrier_colname)
    return out


def calc_nbr_dist(polygon_gdf, nbr_dist_colname='nbr_dist',
                    centroid_colname='centroid',
                    neighbor_colname = "polygon_neighbors",
                    neighbor_id_col='USO_AREA_U'):
    """Add column with distances to neighbors (in kilometers)"""
    return _neighbors.centroid_distances(
        polygon_gdf, neighbor_col=neighbor_colname,
        nbr_dist_col=nbr_dist_colname, centroid_col=centroid_colname,
        id_col=neighbor_id_col)
```

Note the shim ignores `right_gdf` — `_adjacency_bbox` builds the bbox frame
itself. Both callers (`scripts/preprocess.py`,
`tests/oraculum_fixtures.run_production_chain`) pass exactly
`create_bbox_gdf(polygon_gdf)`, so nothing changes; Task 10 deletes the shim.

- [ ] **Step 5: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_neighbors.py -q` — Expected: 9 passed.
Run: `uv run pytest -q -W error` — Expected: **121 passed** (112 + 9).

`tests/test_fixture_invariants.py::test_empirical_pin_*` is the hard red line
(spec's Phase 2 owner gate) — if it fails, **stop and report**.

- [ ] **Step 6: Prove the fixtures did not move**

```bash
uv run python scripts/generate_production_fixtures.py
git diff --exit-code -- tests/fixtures/ && echo NO-DRIFT
```
Expected: `NO-DRIFT`.

- [ ] **Step 7: Commit**

```bash
git add delhi_psi/neighbors.py spatial_index_utils.py tests/test_neighbors.py
git commit -m "refactor: move neighbour construction to delhi_psi.neighbors, add touch/pairwise rules (DEL-16, DEL-22)"
```

---

### Task 5: `delhi_psi/index.py` (spec § 5 step 2, part 3)

This is where DEL-21 lands: the silent `except: pass` in `calc_pcen_mobile`
becomes an explicit lookup miss, and `absent_neighbor: contributes` looks up
a **pre-exclusion frame passed explicitly**.

**Files:**
- Create: `delhi_psi/index.py`
- Modify: `spatial_index_utils.py` (delegate the six index functions)
- Test: `tests/test_index.py` (new)

**Interfaces:**
- Consumes: nothing from earlier tasks (pure pandas/geopandas).
- Produces:
  - `delhi_psi.index.point_counts(polygon_gdf, point_gdf, *, count_col: str, id_col: str = "USO_AREA_U") -> GeoDataFrame`
  - `delhi_psi.index.road_lengths(polygon_gdf, line_gdf, *, length_col: str, id_col: str = "USO_AREA_U") -> GeoDataFrame`
  - `delhi_psi.index.service_amount_column(service: str, kind: str) -> str` — `f"{service}_count"` for `"point"`, `f"{service}_length"` for `"line"` (so the line service `road` yields `road_length`, matching today, and the `road_count → road_length` special case disappears)
  - `delhi_psi.index.pcen(polygon_gdf, *, amount_col: str, pcen_col: str, denominator: str, nbr_dist_col: str = "nbrs_dist_bbox", lookup_frame=None, absent_neighbor: str = "swallowed", include_neighbors: bool = True, decay_form: str = "inverse_linear", distance_unit: str = "km", pop_col: str = "population", area_col: str = "area_km2", id_col: str = "USO_AREA_U") -> GeoDataFrame`
  - `delhi_psi.index.minmax(polygon_gdf, *, source_col: str, target_col: str) -> GeoDataFrame`
  - `delhi_psi.index.service_index(polygon_gdf, amount_col: str, *, service: str, denominator: str, nbr_dist_col: str = "nbrs_dist_bbox", lookup_frame=None, absent_neighbor: str = "swallowed", include_neighbors: bool = True, decay_form: str = "inverse_linear", distance_unit: str = "km", pop_col: str = "population", area_col: str = "area_km2", id_col: str = "USO_AREA_U") -> GeoDataFrame` — adds `f"{service}_pcen"` and `f"{service}_idx"`
  - `delhi_psi.index.overall_psi(polygon_gdf, *, second_normalization: bool) -> GeoDataFrame` — adds `unnorm_psi`, and `norm_psi` only when `second_normalization`

- [ ] **Step 1: Write the failing test**

Create `tests/test_index.py`:

```python
"""delhi_psi.index — counts, lengths, PCEN (Eq. 3), min-max (Eq. 2), PSI (Eq. 1).

The exclusion axes are tested here directly, because this is where DEL-21's
`except: pass` becomes an explicit lookup.
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

from delhi_psi import index
from tests.oraculum_fixtures import load_services, load_settlements


def city_with_neighbours():
    """Two settlements, one clinic each in X, distance 1 km -> decay 1/2."""
    gdf = gpd.GeoDataFrame(
        {"USO_AREA_U": ["X", "Y"], "population": [100.0, 200.0],
         "area_km2": [1.0, 2.0],
         "nbrs_dist_bbox": [[("Y", 1.0)], [("X", 1.0)]],
         "clinic_count": [2.0, 0.0]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000)],
        crs="EPSG:7760")
    return gdf


def test_point_counts_uses_intersects_and_fills_zero():
    city = load_settlements()
    counted = index.point_counts(city, load_services()["clinic"],
                                 count_col="clinic_count")
    counts = counted.set_index("USO_AREA_U")["clinic_count"]
    assert counts["A"] == 2 and counts["B"] == 1 and counts["C"] == 0
    assert counts.dtype.kind == "i"


def test_road_lengths_are_kilometres():
    city = load_settlements()
    lengths = index.road_lengths(city.copy(), load_services()["road"],
                                 length_col="road_length")
    values = lengths.set_index("USO_AREA_U")["road_length"]
    assert values["A"] == pytest.approx(0.75, abs=1e-12)
    assert values["E"] == pytest.approx(0.75, abs=1e-12)
    assert values["C"] == 0.0


def test_service_amount_column_names():
    assert index.service_amount_column("clinic", "point") == "clinic_count"
    assert index.service_amount_column("road", "line") == "road_length"
    with pytest.raises(ValueError, match="polygon"):
        index.service_amount_column("road", "polygon")


def test_pcen_pop_denominator_matches_eq3_by_hand():
    got = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop")
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["X"] == pytest.approx(2 / 100, abs=1e-12)
    assert values["Y"] == pytest.approx((0 + 2 * 0.5) / 200, abs=1e-12)


def test_pcen_popdensity_divides_by_population_over_area():
    got = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="popdensity")
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((0 + 2 * 0.5) / (200 / 2), abs=1e-12)


def test_pcen_include_neighbors_false_is_eq4():
    got = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     include_neighbors=False)
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == 0.0


def test_swallowed_skips_a_neighbour_with_no_row():
    """Today's behaviour: an absent neighbour contributes nothing."""
    frame = city_with_neighbours()
    reported = frame[frame["USO_AREA_U"] == "Y"]
    got = index.pcen(reported, amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     absent_neighbor="swallowed")
    assert got.set_index("USO_AREA_U").loc["Y", "clinic_pcen"] == 0.0


def test_contributes_uses_the_pre_exclusion_frame():
    """DEL-21: excluded settlements still lend their services (Eq. 3)."""
    frame = city_with_neighbours()
    reported = frame[frame["USO_AREA_U"] == "Y"]
    got = index.pcen(reported, amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     absent_neighbor="contributes", lookup_frame=frame)
    assert got.set_index("USO_AREA_U").loc["Y", "clinic_pcen"] == \
        pytest.approx((0 + 2 * 0.5) / 200, abs=1e-12)


def test_contributes_without_a_lookup_frame_is_a_value_error():
    with pytest.raises(ValueError, match="lookup_frame"):
        index.pcen(city_with_neighbours(), amount_col="clinic_count",
                   pcen_col="clinic_pcen", denominator="pop",
                   absent_neighbor="contributes")


def test_contributes_with_an_id_absent_from_the_lookup_frame_raises():
    frame = city_with_neighbours()
    reported = frame[frame["USO_AREA_U"] == "Y"]
    with pytest.raises(KeyError, match="X"):
        index.pcen(reported, amount_col="clinic_count",
                   pcen_col="clinic_pcen", denominator="pop",
                   absent_neighbor="contributes", lookup_frame=reported)


@pytest.mark.parametrize("kwargs,match", [
    (dict(denominator="households"), "households"),
    (dict(absent_neighbor="maybe"), "maybe"),
    (dict(decay_form="exponential"), "exponential"),
    (dict(distance_unit="m"), "'m'"),
])
def test_pcen_rejects_unknown_values(kwargs, match):
    call = dict(amount_col="clinic_count", pcen_col="clinic_pcen",
                denominator="pop")
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        index.pcen(city_with_neighbours(), **call)


def test_minmax_is_eq2():
    frame = pd.DataFrame({"pcen": [1.0, 2.0, 5.0]})
    got = index.minmax(frame, source_col="pcen", target_col="idx")
    assert list(got["idx"]) == pytest.approx([0.0, 0.25, 1.0], abs=1e-12)


def test_service_index_adds_pcen_and_idx():
    got = index.service_index(city_with_neighbours(), "clinic_count",
                              service="clinic", denominator="pop")
    assert list(got["clinic_idx"]) == pytest.approx([1.0, 0.0], abs=1e-12)


def test_overall_psi_averages_idx_columns():
    frame = pd.DataFrame({"a_idx": [0.0, 1.0], "b_idx": [1.0, 1.0],
                          "other": [9.0, 9.0]})
    got = index.overall_psi(frame, second_normalization=True)
    assert list(got["unnorm_psi"]) == pytest.approx([0.5, 1.0], abs=1e-12)
    assert list(got["norm_psi"]) == pytest.approx([0.0, 1.0], abs=1e-12)


def test_overall_psi_omits_norm_psi_when_second_normalization_is_false():
    frame = pd.DataFrame({"a_idx": [0.0, 1.0]})
    got = index.overall_psi(frame, second_normalization=False)
    assert "unnorm_psi" in got.columns
    assert "norm_psi" not in got.columns


def test_shim_still_matches_the_new_path():
    import spatial_index_utils

    city = load_settlements()
    old = spatial_index_utils.add_point_count_column(
        polygon_gdf=city.copy(), point_gdf=load_services()["clinic"],
        count_colname="clinic_count")
    new = index.point_counts(city, load_services()["clinic"],
                             count_col="clinic_count")
    assert list(old["clinic_count"]) == list(new["clinic_count"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_index.py -q`
Expected: collection error — `ImportError: cannot import name 'index' from 'delhi_psi'`.

- [ ] **Step 3: Write `delhi_psi/index.py`**

```python
"""The index math: counts, lengths, PCEN (Eq. 3), min-max (Eq. 2), PSI (Eq. 1).

Pure functions with explicit keyword arguments — never imports
delhi_psi.config. Every expression is copied verbatim from
spatial_index_utils.py. Two deliberate non-changes:
  * `minmax` has NO hi == lo guard, exactly as `calc_service_index` had none.
  * the -1.0 sentinel initialisations stay.
The one behavioural change is DEL-21: `calc_pcen_mobile`'s bare
`except: pass` becomes an explicit lookup miss, which is what makes
`absent_neighbor: contributes` implementable at all.
"""

import logging

import geopandas as gpd

log = logging.getLogger(__name__)

DENOMINATORS = ("pop", "popdensity", "one")
ABSENT_NEIGHBOR = ("swallowed", "contributes")


def point_counts(polygon_gdf, point_gdf, *, count_col, id_col="USO_AREA_U"):
    """Count points inside each polygon (gpd.sjoin's default `intersects`).

    NOTE: boundary-inclusive, so a point exactly on a shared border counts
    for both neighbours (rule-set gap #6, latent on the real layers — pinned
    by tests/test_oracle.py::test_gap6_border_point_is_double_counted...).
    Unchanged here on purpose.
    """
    point_cnt = gpd.sjoin(polygon_gdf, point_gdf).groupby(id_col).size().\
        reset_index()
    point_cnt = point_cnt.rename(columns={0: count_col})
    out = polygon_gdf.merge(point_cnt, how="left", on=id_col)
    out[count_col] = out[count_col].fillna(0)
    out[count_col] = out[count_col].astype(int)
    return out


def _length_in_polygon(small_gdf, poly_geom_col, line_geom_col):
    """Total length (km) of the line pieces inside one polygon."""
    total_length = 0
    for i, row in small_gdf.iterrows():
        polygon = row[poly_geom_col]
        line = row[line_geom_col]
        intersection = polygon.intersection(line)
        length = intersection.length / 1000
        total_length += length
    return total_length


def road_lengths(polygon_gdf, line_gdf, *, length_col, id_col="USO_AREA_U"):
    """Total (poly)line length in km inside each polygon."""
    polygon_gdf[length_col] = 0.0

    line_geom_col = "line_geometry"
    line_gdf[line_geom_col] = line_gdf["geometry"]

    joined = gpd.sjoin(polygon_gdf, line_gdf)
    joined_grouped = joined.groupby(id_col)

    for name, group in joined_grouped:
        name_index = polygon_gdf[polygon_gdf[id_col] == name].index.values[0]
        total_road_length = _length_in_polygon(
            small_gdf=group, poly_geom_col="geometry",
            line_geom_col=line_geom_col)
        polygon_gdf.loc[name_index, length_col] = total_road_length

    return polygon_gdf


def service_amount_column(service, kind):
    """The column a service's own amount lands in.

    Point services count (`clinic_count`); line services measure length
    (`road_length`) — which is what production ended up with after its
    `road_count -> road_length` rename, so the special case disappears.
    """
    if kind == "point":
        return f"{service}_count"
    if kind == "line":
        return f"{service}_length"
    raise ValueError(
        f"unknown service kind {kind!r}; allowed values: ['point', 'line']")


def _decay(distance_km, decay_form, distance_unit):
    if decay_form != "inverse_linear":
        raise ValueError(
            f"unknown decay form {decay_form!r}; allowed values: "
            "['inverse_linear']")
    if distance_unit != "km":
        raise ValueError(
            f"unknown decay distance unit {distance_unit!r}; allowed values: "
            "['km']")
    return 1 / (1 + distance_km)


def pcen(polygon_gdf, *, amount_col, pcen_col, denominator,
         nbr_dist_col="nbrs_dist_bbox", lookup_frame=None,
         absent_neighbor="swallowed", include_neighbors=True,
         decay_form="inverse_linear", distance_unit="km",
         pop_col="population", area_col="area_km2", id_col="USO_AREA_U"):
    """Eq. 3: effective service count per denominator, with distance decay.

    absent_neighbor="swallowed": a neighbour id with no row in the compute
        frame contributes nothing (today's behaviour, as an explicit lookup
        miss — never a bare `except`).
    absent_neighbor="contributes": amounts are looked up in `lookup_frame`,
        the PRE-EXCLUSION frame, so excluded settlements still contribute
        their services; an id absent from that frame too is an error.
    include_neighbors=False: Eq. 4 as written — own amount only, no
        neighbour term (`roads: eq4_own_only`).
    """
    if denominator not in DENOMINATORS:
        raise ValueError(
            f"unknown denominator {denominator!r}; allowed values: "
            f"{list(DENOMINATORS)}")
    if absent_neighbor not in ABSENT_NEIGHBOR:
        raise ValueError(
            f"unknown absent_neighbor {absent_neighbor!r}; allowed values: "
            f"{list(ABSENT_NEIGHBOR)}")

    gdf_copy = polygon_gdf.copy()

    if absent_neighbor == "contributes":
        if lookup_frame is None:
            raise ValueError(
                "absent_neighbor='contributes' requires lookup_frame — the "
                "pre-exclusion frame the neighbour amounts are read from")
        lookup = lookup_frame
    else:
        lookup = gdf_copy

    # probe the decay knobs once, so a bad value fails even on a city with
    # no neighbour links at all
    _decay(0.0, decay_form, distance_unit)

    gdf_copy[pcen_col] = -1.0

    for idx, row in gdf_copy.iterrows():
        if denominator == "popdensity":
            denom = row[pop_col] / row[area_col]
        elif denominator == "pop":
            denom = row[pop_col]
        else:
            denom = 1

        poly_count = row[amount_col]

        if include_neighbors:
            for nbr_id, nbr_dist in row[nbr_dist_col]:
                match = lookup[lookup[id_col] == nbr_id]
                if len(match) == 0:
                    if absent_neighbor == "contributes":
                        raise KeyError(
                            f"neighbour {nbr_id!r} of {row[id_col]!r} has no "
                            "row in the pre-exclusion lookup frame")
                    continue
                nbr_count = match[amount_col].array[0]
                poly_count += nbr_count * _decay(nbr_dist, decay_form,
                                                 distance_unit)

        gdf_copy.loc[idx, pcen_col] = poly_count / denom

    return gdf_copy


def minmax(polygon_gdf, *, source_col, target_col):
    """Eq. 2: rescale a column to [0, 1] across the frame.

    Verbatim `calc_service_index` — deliberately WITHOUT a hi == lo guard.
    """
    gdf_copy = polygon_gdf.copy()

    pcen_min = gdf_copy[source_col].min()
    pcen_max = gdf_copy[source_col].max()

    gdf_copy[target_col] = -1.0

    for idx, row in gdf_copy.iterrows():
        result = (row[source_col] - pcen_min) / (pcen_max - pcen_min)
        gdf_copy.loc[idx, target_col] = result

    return gdf_copy


def service_index(polygon_gdf, amount_col, *, service, denominator,
                  nbr_dist_col="nbrs_dist_bbox", lookup_frame=None,
                  absent_neighbor="swallowed", include_neighbors=True,
                  decay_form="inverse_linear", distance_unit="km",
                  pop_col="population", area_col="area_km2",
                  id_col="USO_AREA_U"):
    """pcen then minmax for one service — replaces BOTH create_service_index
    variants (DEL-16). Fed by point_counts() or road_lengths()."""
    pcen_col = f"{service}_pcen"
    idx_col = f"{service}_idx"
    out = pcen(polygon_gdf, amount_col=amount_col, pcen_col=pcen_col,
               denominator=denominator, nbr_dist_col=nbr_dist_col,
               lookup_frame=lookup_frame, absent_neighbor=absent_neighbor,
               include_neighbors=include_neighbors, decay_form=decay_form,
               distance_unit=distance_unit, pop_col=pop_col,
               area_col=area_col, id_col=id_col)
    return minmax(out, source_col=pcen_col, target_col=idx_col)


def overall_psi(polygon_gdf, *, second_normalization):
    """Eq. 1: the mean of every `*_idx` column, plus the optional second
    normalization (`norm_psi`); the column is absent when it is off."""
    out = polygon_gdf.copy()
    idx_columns = [column for column in out.columns
                   if column.endswith("_idx")]
    out["unnorm_psi"] = out[idx_columns].mean(axis=1)
    if second_normalization:
        out = minmax(out, source_col="unnorm_psi", target_col="norm_psi")
    return out
```

- [ ] **Step 4: Make `spatial_index_utils.py` delegate**

Add near the other delegation imports:

```python
from delhi_psi import index as _index
```

Replace these bodies (keeping names and signatures):

```python
def add_point_count_column(polygon_gdf, point_gdf, count_colname,
                           join_col='USO_AREA_U'):
    """Add count of services for each polygon to polygon_gdf"""
    return _index.point_counts(polygon_gdf, point_gdf,
                               count_col=count_colname, id_col=join_col)


def calc_service_length(small_gdf, poly_geom_colname, line_geom_colname):
    """Calculate length of all (poly)line services in a colony"""
    return _index._length_in_polygon(small_gdf, poly_geom_colname,
                                     line_geom_colname)


def add_service_length_column(polygon_gdf, line_gdf, length_colname,
    id_colname='USO_AREA_U'):
    """Add distance of (poly)line services for each polygon"""
    return _index.road_lengths(polygon_gdf, line_gdf,
                               length_col=length_colname, id_col=id_colname)


def calc_pcen_mobile(polygon_gdf, count_colname,
                     pcen_mobile_colname,
                     pcen_denom,
                     nbr_dist_colname='nbr_dist',
                     pop_colname='population',
                     area_colname='area_km2',
                     id_col='USO_AREA_U'):
    """Calculates and adds column for PCEN_mobile"""
    return _index.pcen(polygon_gdf, amount_col=count_colname,
                       pcen_col=pcen_mobile_colname, denominator=pcen_denom,
                       nbr_dist_col=nbr_dist_colname,
                       absent_neighbor="swallowed", pop_col=pop_colname,
                       area_col=area_colname, id_col=id_col)


def calc_service_index(polygon_gdf, pcen_mobile_colname, service_idx_colname):
    """Calculates service index [0, 1] based on PCEN_MOBILE"""
    return _index.minmax(polygon_gdf, source_col=pcen_mobile_colname,
                         target_col=service_idx_colname)


def create_overall_psi(colonies_gdf):
    """Create Overall PSI across all indices (unnormalized and normalized)"""
    return _index.overall_psi(colonies_gdf, second_normalization=True)
```

Leave `create_service_index`, `create_service_length_index`,
`calc_point_services` and `calc_all_services` **as they are** — they are
compositions that now call the delegating primitives, and Task 7's
`compute_frames` supersedes them. They die in Task 10.

- [ ] **Step 5: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_index.py -q` — Expected: 19 passed.
Run: `uv run pytest -q -W error` — Expected: **140 passed** (121 + 19).

If `tests/test_oracle.py` moves by so much as 1e-12, the `except: pass`
replacement was not equivalent — **stop and report** (spec § 10).

- [ ] **Step 6: Prove the fixtures did not move**

```bash
uv run python scripts/generate_production_fixtures.py
git diff --exit-code -- tests/fixtures/ && echo NO-DRIFT
```
Expected: `NO-DRIFT`. This is the load-bearing check for this task: the
production fixture is generated through `_production_frame`, which now runs
on `delhi_psi.index`, and it must still be **byte-identical** to the step-0
snapshot committed in Task 1.

- [ ] **Step 7: Commit**

```bash
git add delhi_psi/index.py spatial_index_utils.py tests/test_index.py
git commit -m "refactor: move index math to delhi_psi.index; explicit absent-neighbour lookup (DEL-16, DEL-21)"
```

---

### Task 6: `delhi_psi/io.py` (reads and writes) and `delhi_psi/validate.py`

**Files:**
- Modify: `delhi_psi/io.py` (add reads/writes to the path helpers from Task 2)
- Create: `delhi_psi/validate.py`
- Create: `tests/test_validate.py`
- Modify: `tests/test_common.py` (rewire `scripts.common` → `delhi_psi.io`; expectations unchanged)

**Interfaces:**
- Consumes: `delhi_psi.io.resolve_data_dir`, `out_dir_path`, `resolve_out_dir` (Task 2).
- Produces:
  - `delhi_psi.io.read_layer(path, *, epsg: int | None = None) -> GeoDataFrame`
  - `delhi_psi.io.read_population(path) -> DataFrame`
  - `delhi_psi.io.read_neighbors(path) -> GeoDataFrame`
  - `delhi_psi.io.write_neighbors(frame, path) -> Path`
  - `delhi_psi.io.SHAPEFILE_DROP_COLUMNS: tuple[str, ...]` = `("nbrs_bbox", "nbrs_dist_bbox", "centroid")`
  - `delhi_psi.io.write_outputs(frame, out_dir, *, basename: str, formats) -> list[Path]`
  - `delhi_psi.validate.ValidationError(RuntimeError)`
  - `delhi_psi.validate.LayerReport` (frozen dataclass: `name`, `geom_type`, `n_rows`, `has_duplicate_rows`, `invalid_geometries`, `none_geometries`, `all_geom_type`, `within_bounds`; property `ok`)
  - `delhi_psi.validate.has_duplicate_rows(gdf) -> bool`
  - `delhi_psi.validate.invalid_geometries(gdf) -> tuple`
  - `delhi_psi.validate.geometries_are(gdf, geom_type: str) -> bool`
  - `delhi_psi.validate.within_bounds(gdf, bounds_gdf) -> bool`
  - `delhi_psi.validate.check_layer(gdf, *, name: str, geom_type: str, bounds_gdf) -> LayerReport`
  - `delhi_psi.validate.require_layer(gdf, *, name: str, geom_type: str, bounds_gdf) -> LayerReport` (raises `ValidationError`)
  - `delhi_psi.validate.check_missing_population(missing_count: int, *, maximum: int) -> None` (raises)
  - `delhi_psi.validate.check_no_negative(frame, *, suffixes=("_count", "_pcen", "_idx")) -> None` (raises)
  - `delhi_psi.validate.check_crs_match(frames: dict[str, GeoDataFrame]) -> None` (raises)
  - `delhi_psi.validate.check_crs_defined(frames: dict[str, GeoDataFrame]) -> None` (raises when any frame has `crs is None`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validate.py`:

```python
"""delhi_psi.validate — every check on synthetic frames, pass and fail.

The notebooks' eyeball checks become assertions that RAISE (DEL-25).
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from delhi_psi import validate

BOUNDS = gpd.GeoDataFrame({"name": ["bounds"]},
                          geometry=[box(0, 0, 10_000, 10_000)],
                          crs="EPSG:7760")


def polygons(geoms, crs="EPSG:7760"):
    return gpd.GeoDataFrame({"id": list(range(len(geoms)))},
                            geometry=list(geoms), crs=crs)


def test_has_duplicate_rows_both_ways():
    gdf = polygons([box(0, 0, 1, 1), box(2, 2, 3, 3)])
    assert validate.has_duplicate_rows(gdf) is False
    doubled = pd.concat([gdf, gdf.iloc[[0]]], ignore_index=True)
    doubled["id"] = [0, 1, 0]
    assert validate.has_duplicate_rows(doubled) is True


def test_invalid_geometries_lists_offending_rows():
    bowtie = LineString([(0, 0), (1, 1)]).buffer(0).union(
        box(0, 0, 1, 1))          # valid
    good = polygons([box(0, 0, 1, 1), bowtie])
    assert validate.invalid_geometries(good) == ()
    from shapely.geometry import Polygon
    bad = polygons([Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])])
    assert validate.invalid_geometries(bad) == (0,)


def test_geometries_are_accepts_multipolygon_for_polygon():
    from shapely.geometry import MultiPolygon
    gdf = polygons([box(0, 0, 1, 1),
                    MultiPolygon([box(2, 2, 3, 3), box(4, 4, 5, 5)])])
    assert validate.geometries_are(gdf, "Polygon") is True
    assert validate.geometries_are(gdf, "Point") is False


def test_geometries_are_accepts_linestring_for_line():
    gdf = gpd.GeoDataFrame({"id": [0]},
                           geometry=[LineString([(0, 0), (1, 1)])],
                           crs="EPSG:7760")
    assert validate.geometries_are(gdf, "Line") is True


def test_geometries_are_rejects_a_bad_geom_type_argument():
    with pytest.raises(ValueError, match="Curve"):
        validate.geometries_are(polygons([box(0, 0, 1, 1)]), "Curve")


def test_geometries_are_does_not_mutate_the_frame():
    gdf = polygons([box(0, 0, 1, 1)])
    before = list(gdf.columns)
    validate.geometries_are(gdf, "Polygon")
    assert list(gdf.columns) == before


def test_within_bounds_both_ways():
    inside = polygons([box(10, 10, 20, 20)])
    outside = polygons([box(10, 10, 20_000, 20_000)])
    assert validate.within_bounds(inside, BOUNDS) is True
    assert validate.within_bounds(outside, BOUNDS) is False


def test_check_layer_reports_ok():
    report = validate.check_layer(polygons([box(10, 10, 20, 20)]),
                                  name="settlements", geom_type="Polygon",
                                  bounds_gdf=BOUNDS)
    assert report.ok is True
    assert report.name == "settlements" and report.n_rows == 1


def test_check_layer_reports_not_ok_without_raising():
    report = validate.check_layer(polygons([box(10, 10, 20_000, 20_000)]),
                                  name="settlements", geom_type="Polygon",
                                  bounds_gdf=BOUNDS)
    assert report.ok is False
    assert report.within_bounds is False


def test_require_layer_raises_on_a_bad_layer():
    with pytest.raises(validate.ValidationError) as exc:
        validate.require_layer(polygons([box(10, 10, 20_000, 20_000)]),
                               name="settlements", geom_type="Polygon",
                               bounds_gdf=BOUNDS)
    assert "settlements" in str(exc.value)
    assert "within_bounds" in str(exc.value)


def test_require_layer_returns_the_report_when_ok():
    report = validate.require_layer(polygons([box(10, 10, 20, 20)]),
                                    name="settlements", geom_type="Polygon",
                                    bounds_gdf=BOUNDS)
    assert report.ok is True


def test_check_missing_population_passes_at_the_limit_and_raises_above():
    assert validate.check_missing_population(15, maximum=15) is None
    with pytest.raises(validate.ValidationError) as exc:
        validate.check_missing_population(16, maximum=15)
    assert "16" in str(exc.value) and "15" in str(exc.value)


def test_check_no_negative_passes_and_raises():
    good = pd.DataFrame({"bank_count": [0, 1], "bank_pcen": [0.0, 0.5],
                         "bank_idx": [0.0, 1.0], "ignored": [-9, -9]})
    assert validate.check_no_negative(good) is None
    bad = good.copy()
    bad["bank_pcen"] = [-1.0, 0.5]
    with pytest.raises(validate.ValidationError) as exc:
        validate.check_no_negative(bad)
    assert "bank_pcen" in str(exc.value)


def test_check_crs_match_passes_and_raises():
    a = polygons([box(0, 0, 1, 1)])
    b = polygons([box(2, 2, 3, 3)])
    assert validate.check_crs_match({"a": a, "b": b}) is None
    c = b.to_crs(epsg=4326)
    with pytest.raises(validate.ValidationError) as exc:
        validate.check_crs_match({"a": a, "c": c})
    assert "c" in str(exc.value)


def test_check_crs_defined_raises_when_a_frame_has_no_crs():
    a = polygons([box(0, 0, 1, 1)])
    naked = polygons([box(0, 0, 1, 1)]).set_crs(None, allow_override=True)
    assert validate.check_crs_defined({"a": a}) is None
    with pytest.raises(validate.ValidationError) as exc:
        validate.check_crs_defined({"a": a, "naked": naked})
    assert "naked" in str(exc.value)


def test_read_layer_missing_file_raises_file_not_found(tmp_path):
    # pyogrio raises DataSourceError; io must translate it so the CLI's
    # exit-code mapping (FileNotFoundError/OSError -> 1) holds.
    from delhi_psi import io
    with pytest.raises(FileNotFoundError):
        io.read_layer(tmp_path / "nope" / "missing.shp")
```

Rewrite `tests/test_common.py` — same expectations, new import (spec § 7):

```python
"""Tests for delhi_psi.io path resolution (was scripts/common.py)."""
from pathlib import Path

from delhi_psi.io import resolve_data_dir, resolve_out_dir


def test_flag_beats_env_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("DELHI_DATA_DIR", "/env/ignored")
    assert resolve_data_dir(str(tmp_path)) == tmp_path


def test_env_beats_default(monkeypatch, tmp_path):
    monkeypatch.setenv("DELHI_DATA_DIR", str(tmp_path))
    assert resolve_data_dir(None) == tmp_path


def test_default_is_home_delhi_data(monkeypatch):
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    assert resolve_data_dir(None) == Path("~/delhi_data").expanduser()


def test_flag_expands_user(monkeypatch):
    monkeypatch.delenv("DELHI_DATA_DIR", raising=False)
    assert resolve_data_dir("~/somewhere") == Path("~/somewhere").expanduser()


def test_out_dir_defaults_to_data_dir(tmp_path):
    out = resolve_out_dir(None, tmp_path)
    assert out == tmp_path


def test_out_dir_flag_wins_and_is_created(tmp_path):
    target = tmp_path / "sub" / "verify"
    out = resolve_out_dir(str(target), tmp_path)
    assert out == target
    assert target.is_dir()
```

Also add an IO test to the same new file `tests/test_validate.py`? **No** —
put the write test in `tests/test_cli.py` (Task 8), where the shp path runs
in-process and the warning filter is the thing under test.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_validate.py tests/test_common.py -q`
Expected: `tests/test_validate.py` fails to collect with
`ImportError: cannot import name 'validate' from 'delhi_psi'`;
`tests/test_common.py` passes already (Task 2 created the two helpers).

- [ ] **Step 3: Add reads and writes to `delhi_psi/io.py`**

Append to `delhi_psi/io.py` (keep the Task 2 path helpers above):

```python
import logging
import warnings

import geopandas as gpd
from pyogrio.errors import DataSourceError
import joblib
import pandas as pd

log = logging.getLogger(__name__)

# Shapefiles cannot hold list or geometry-valued columns; production drops
# exactly these three before to_file (spec § 5).
SHAPEFILE_DROP_COLUMNS = ("nbrs_bbox", "nbrs_dist_bbox", "centroid")


def read_layer(path, *, epsg=None):
    """Read a vector layer; optionally force a CRS (fixtures do this).

    A missing or unreadable source raises FileNotFoundError. pyogrio raises
    its own DataSourceError for that case, which the CLI's exit-code mapping
    (FileNotFoundError/OSError -> exit 1) would not catch (plan review R1,
    Critical).
    """
    try:
        gdf = gpd.read_file(path)
    except DataSourceError as exc:
        raise FileNotFoundError(f"cannot read vector layer {path}: {exc}") from exc
    if epsg is not None:
        gdf = gdf.set_crs(epsg=epsg, allow_override=True)
    return gdf


def read_population(path):
    return pd.read_csv(path)


def read_neighbors(path):
    with open(path, "rb") as handle:
        return joblib.load(handle)


def write_neighbors(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        joblib.dump(frame, handle)
    log.info("wrote %s", path)
    return path


def _write_shapefile(frame, path):
    """Write a shapefile, muting the two warnings this repo accepts.

    geopandas truncates column names over 10 characters and pyogrio emits one
    'Normalized/laundered field name' RuntimeWarning per truncated column from
    its C error handler (under -W error it can surface as
    PytestUnraisableExceptionWarning). Both are accepted behaviour — they were
    invisible only because the old e2e test ran the scripts in a subprocess.
    The filter is scoped to THIS write and nothing else.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Column names longer than 10 characters",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Normalized/laundered field name",
            category=RuntimeWarning,
        )
        frame.to_file(path)
    return path


def write_outputs(frame, out_dir, *, basename, formats):
    """Write one PSI result set. Returns the paths written, in order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt == "csv":
            path = out_dir / f"{basename}.csv"
            frame.to_csv(path, index=False)
        elif fmt == "shp":
            path = out_dir / f"{basename}.shp"
            droppable = [c for c in SHAPEFILE_DROP_COLUMNS
                         if c in frame.columns]
            _write_shapefile(frame.drop(columns=droppable), path)
        elif fmt == "joblib":
            path = out_dir / f"{basename}.joblib"
            with open(path, "wb") as handle:
                joblib.dump(frame, handle)
        else:
            raise ValueError(
                f"unknown output format {fmt!r}; allowed values: "
                "['csv', 'shp', 'joblib']")
        written.append(path)
        log.info("wrote %s", path)
    return written
```

- [ ] **Step 4: Write `delhi_psi/validate.py`**

```python
"""Validation as assertions (DEL-25): the layer battery and the post-compute
sanity checks. Every check returns a Report or None; `require_*` raises.

The notebook printed these; here they raise, so a bad layer or a negative
index stops the run instead of scrolling past.
"""

import logging
from dataclasses import dataclass

from shapely.geometry import box

log = logging.getLogger(__name__)

GEOM_TYPES = ("Point", "Line", "Polygon")


class ValidationError(RuntimeError):
    """A validation check failed; the pipeline stage must not continue."""


@dataclass(frozen=True)
class LayerReport:
    name: str
    geom_type: str
    n_rows: int
    has_duplicate_rows: bool
    invalid_geometries: tuple
    none_geometries: tuple
    all_geom_type: bool
    within_bounds: bool

    @property
    def ok(self):
        return (not self.has_duplicate_rows
                and not self.invalid_geometries
                and not self.none_geometries
                and self.all_geom_type
                and self.within_bounds)

    def failures(self):
        problems = []
        if self.has_duplicate_rows:
            problems.append("has_duplicate_rows")
        if self.invalid_geometries:
            problems.append(f"invalid_geometries={list(self.invalid_geometries)}")
        if self.none_geometries:
            problems.append(f"none_geometries={list(self.none_geometries)}")
        if not self.all_geom_type:
            problems.append(f"all_geom_type is False (expected {self.geom_type})")
        if not self.within_bounds:
            problems.append("within_bounds is False")
        return problems


def has_duplicate_rows(gdf):
    return bool(len(gdf[gdf.duplicated()]) > 0)


def invalid_geometries(gdf):
    return tuple(i for i, row in gdf.iterrows()
                 if row["geometry"] is not None and not row["geometry"].is_valid)


def none_geometries(gdf):
    return tuple(gdf[gdf["geometry"].isna()].index)


def geometries_are(gdf, geom_type):
    """True if every geometry is of geom_type.

    Verbatim `check_geometries`, minus its vestigial `geom_type` column: the
    original assigned `gdf['geom_type'] = type(gdf['geometry'])` and then read
    `gdf.geom_type`, which resolves to the GeoDataFrame PROPERTY, not the
    column — so the assignment never affected the result. Dropping it means
    this function no longer mutates its argument.
    """
    if geom_type not in GEOM_TYPES:
        raise ValueError(f"unknown geom_type {geom_type!r}; allowed values: "
                         f"{list(GEOM_TYPES)}")
    geom_type_list = gdf.geom_type.unique()
    geom_is_geom_type = [geom_type in geom for geom in geom_type_list]
    return False not in geom_is_geom_type


def within_bounds(gdf, bounds_gdf):
    """True if the layer's total extent sits inside the bounds polygon."""
    reprojected = gdf.to_crs(bounds_gdf.crs)
    extent = box(reprojected.total_bounds[0], reprojected.total_bounds[1],
                 reprojected.total_bounds[2], reprojected.total_bounds[3])
    return bool(bounds_gdf.contains(extent).iloc[0])


def check_layer(gdf, *, name, geom_type, bounds_gdf):
    """Run the whole battery; never raises."""
    assert "geometry" in gdf.columns, 'there is no "geometry" column'
    report = LayerReport(
        name=name,
        geom_type=geom_type,
        n_rows=len(gdf),
        has_duplicate_rows=has_duplicate_rows(gdf),
        invalid_geometries=invalid_geometries(gdf),
        none_geometries=none_geometries(gdf),
        all_geom_type=geometries_are(gdf, geom_type),
        within_bounds=within_bounds(gdf, bounds_gdf),
    )
    log.info("layer %s: %d rows, ok=%s", name, report.n_rows, report.ok)
    return report


def require_layer(gdf, *, name, geom_type, bounds_gdf):
    """check_layer, but a failure raises ValidationError."""
    report = check_layer(gdf, name=name, geom_type=geom_type,
                         bounds_gdf=bounds_gdf)
    if not report.ok:
        raise ValidationError(
            f"layer {name!r} failed validation: {'; '.join(report.failures())}")
    return report


def check_missing_population(missing_count, *, maximum):
    if missing_count > maximum:
        raise ValidationError(
            f"{missing_count} settlements have no population row, above the "
            f"configured maximum of {maximum} "
            "(validate.max_missing_population)")
    log.info("%d settlements missing population (max %d)", missing_count,
             maximum)


def check_no_negative(frame, *, suffixes=("_count", "_pcen", "_idx")):
    offenders = []
    for suffix in suffixes:
        for column in [c for c in frame.columns if str(c).endswith(suffix)]:
            n_negative = int((frame[column] < 0).sum())
            if n_negative:
                offenders.append(f"{column}: {n_negative} negative value(s)")
    if offenders:
        raise ValidationError("negative values in derived columns: "
                              + "; ".join(offenders))


def check_crs_match(frames):
    """Every frame must share one CRS (the reprojection target)."""
    seen = {name: gdf.crs for name, gdf in frames.items()}
    distinct = {str(crs) for crs in seen.values()}
    if len(distinct) > 1:
        detail = ", ".join(f"{name}={crs}" for name, crs in seen.items())
        raise ValidationError(f"CRS mismatch across layers: {detail}")


def check_crs_defined(frames):
    """Every frame must carry a CRS; a CRS-less layer would be silently
    reprojected from nothing (spec § 6, compute-stage CRS check)."""
    missing = [name for name, gdf in frames.items() if gdf.crs is None]
    if missing:
        raise ValidationError(f"layers without a CRS: {', '.join(missing)}")
```

Note the `from pathlib import Path` needed by `write_outputs` — make sure
`io.py`'s import block (already imports `Path` for the Task 2 helpers) still
has it after the append.

- [ ] **Step 5: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_validate.py tests/test_common.py -q` — Expected: 20 passed (14 new + the 6 rewired).
Run: `uv run pytest -q -W error` — Expected: **154 passed** (140 + 14).

- [ ] **Step 6: Commit**

```bash
git add delhi_psi/io.py delhi_psi/validate.py tests/test_validate.py tests/test_common.py
git commit -m "feat: delhi_psi.io reads/writes and delhi_psi.validate assertions (DEL-25)"
```

---

### Task 7: `pipeline.compute_frames` — the in-memory seam (spec § 5 step 3, part 1)

**Files:**
- Create: `delhi_psi/pipeline.py` (`compute_frames` and its helpers only; the path stages arrive in Task 8)
- Modify: `tests/oraculum_fixtures.py` (add the profile helpers; **keep `run_production_chain` unchanged** — it is still the step-0 snapshot's backend until Task 11)
- Modify: `tests/test_oracle.py` (re-expressed as profile + `types`/`stage` overrides)
- Create: `tests/test_profiles_match_reference.py` (`code-2025` half + the enum-table mapping test)

**Interfaces:**
- Consumes: `delhi_psi.geometry.barrier_flags`, `reproject`; `delhi_psi.neighbors.{combine_barrier_flags, adjacency, apply_barrier, centroid_distances}`; `delhi_psi.index.{point_counts, road_lengths, service_amount_column, service_index, overall_psi}`; `delhi_psi.config.MethodologyConfig`; `delhi_psi.validate.check_missing_population`.
- Produces:
  - `delhi_psi.pipeline.ID_COL = "USO_AREA_U"`, `TYPE_COL = "USO_FINAL"`, `NBRS_COL = "nbrs_bbox"`, `NBRS_DIST_COL = "nbrs_dist_bbox"`, `CENTROID_COL = "centroid"`
  - `delhi_psi.pipeline.service_kind(name: str, gdf) -> str` (`"point"` | `"line"`)
  - `delhi_psi.pipeline.attach_population(settlements, population, *, id_col=ID_COL, population_id_col="uso_area_u", population_value_col="population") -> tuple[DataFrame, frozenset[str]]`
  - `delhi_psi.pipeline.excluded_ids(frame, *, types, id_col=ID_COL, type_col=TYPE_COL) -> frozenset[str]`
  - `delhi_psi.pipeline.build_neighbors(settlements, barriers, methodology, *, epsg_code=7760, id_col=ID_COL) -> GeoDataFrame`
  - `delhi_psi.pipeline.apply_exclusion(neighbor_frame, *, dropped, stage, id_col=ID_COL) -> tuple[GeoDataFrame, GeoDataFrame]`
  - `delhi_psi.pipeline.index_frames(neighbor_frame, services, methodology, denominator, *, dropped=frozenset(), epsg_code=7760, id_col=ID_COL) -> GeoDataFrame`
  - `delhi_psi.pipeline.compute_frames(settlements, barriers, services, population, methodology, denominator, *, epsg_code=7760, id_col=ID_COL, type_col=TYPE_COL, population_id_col="uso_area_u", population_value_col="population", missing_population="drop", max_missing_population=None) -> GeoDataFrame`
  - `tests.oraculum_fixtures.ORACLE_SCENARIOS: list[tuple[str, tuple[str, ...], str]]`
  - `tests.oraculum_fixtures.methodology_with(profile, *, types=None, stage=None) -> MethodologyConfig`
  - `tests.oraculum_fixtures.compute_oracle_frame(profile, *, types, stage, denom) -> GeoDataFrame` (indexed by settlement id)

> **The spec's signature is honoured exactly** — `compute_frames(settlements, barriers, services, population, methodology, denominator)` is the documented positional call. Everything after `*` is keyword-only with the `code-2025` value as its default, so the documented call is unchanged. See "Spec ambiguities resolved" at the end of this plan for why each was needed.

- [ ] **Step 1: Write the failing tests**

Add to `tests/oraculum_fixtures.py` (leave the existing loaders and
`run_production_chain` exactly as they are):

```python
# --- profile-driven helpers (Phase 3A) --------------------------------
# The § 7 scenario table: a profile plus `types`/`stage` overrides ONLY.
# `absent_neighbor` always comes from the profile, because in the reference
# it is a rule-set property, not a scenario property.
ORACLE_SCENARIOS = [
    # (reference scenario, exclusion.types, exclusion.stage)
    ("baseline", (), "post_neighbors"),
    ("excl_rv_only", ("RV",), "post_neighbors"),
    ("excl_contributing", ("RV", "IND"), "post_neighbors"),
    ("excl_removed", ("RV", "IND"), "pre_neighbors"),
    ("excl_ind_removed", ("IND",), "pre_neighbors"),
]


def methodology_with(profile, *, types=None, stage=None):
    """The shipped profile's methodology with the two allowed overrides."""
    from dataclasses import replace

    from delhi_psi.config import ExclusionStage, load_config

    methodology = load_config(profile).methodology
    exclusion = methodology.exclusion
    if types is not None:
        exclusion = replace(exclusion, types=tuple(types))
    if stage is not None:
        exclusion = replace(exclusion, stage=ExclusionStage(stage))
    return replace(methodology, exclusion=exclusion)


def compute_oracle_frame(profile, *, types, stage, denom):
    """compute_frames on the Oraculum city, indexed by settlement id.

    The fixture city carries its own `population` column, so population=None.
    """
    from delhi_psi.pipeline import compute_frames

    return compute_frames(
        load_settlements(), {"canal": load_barriers()}, load_services(),
        None, methodology_with(profile, types=types, stage=stage), denom,
    ).set_index("USO_AREA_U")
```

Rewrite `tests/test_oracle.py`. Keep the module docstring's oracle-contract
warning, keep every non-scenario test, and replace the wiring:

```python
"""Production code vs the oracle's expected values (rule=code rows).

Every comparison here is production-vs-hand-anchored-reference. A failure
means production behavior changed (or the oracle is wrong) — investigate,
never blindly update the CSV (oracle contract: docs/superpowers/specs/
2026-08-17-phase2-oracle-design.md).

Scenarios are expressed as the `code-2025` profile plus `exclusion.types` and
`exclusion.stage` overrides ONLY; `absent_neighbor` comes from the profile
(`swallowed` for code-2025), because in the reference it is a rule-set
property, not a scenario property (spec § 7).
"""

from pathlib import Path

import pandas as pd
import pytest

from delhi_psi.config import load_config
from tests.oraculum_fixtures import (
    ORACLE_SCENARIOS, compute_oracle_frame, load_barriers, load_services,
    load_settlements, run_production_chain,
)

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"
PROFILE = "code-2025"

# production column name -> expected_values metric name
METRICS = {
    "clinic_pcen": "clinic_pcen", "clinic_idx": "clinic_idx",
    "school_pcen": "school_pcen", "school_idx": "school_idx",
    "bank_pcen": "bank_pcen", "police_pcen": "police_pcen",
    "ration_pcen": "ration_pcen", "transport_pcen": "transport_pcen",
    "road_pcen": "road_pcen", "road_idx": "road_idx",
    "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
}


@pytest.fixture(scope="module")
def expected():
    return pd.read_csv(CSV)


def _expected_frame(expected, scenario, denom):
    sub = expected[(expected["rule"] == "code")
                   & (expected["scenario"] == scenario)
                   & (expected["denom"] == denom)]
    return sub.pivot(index="settlement", columns="metric", values="value")


def _frame(denom, *, types=(), stage="post_neighbors"):
    return compute_oracle_frame(PROFILE, types=types, stage=stage, denom=denom)


def _compared_metrics():
    """The metric set is derived from the profile: norm_psi only when the
    profile turns the second normalization on."""
    cfg = load_config(PROFILE)
    if cfg.methodology.second_normalization:
        return METRICS
    return {k: v for k, v in METRICS.items() if k != "norm_psi"}


@pytest.mark.parametrize("denom", ["pop", "popdensity"])
@pytest.mark.parametrize("scenario,types,stage", ORACLE_SCENARIOS)
def test_production_matches_code_rows(expected, scenario, types, stage, denom):
    exp = _expected_frame(expected, scenario, denom)
    got = _frame(denom, types=types, stage=stage)
    assert set(got.index) == set(exp.index)
    for prod_col, metric in _compared_metrics().items():
        for sid in exp.index:
            assert got.loc[sid, prod_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-12), (scenario, denom, sid, prod_col)


def test_zero_service_settlement(expected):
    got = _frame("pop")
    assert got.loc["C", "clinic_count"] == 0
    assert got.loc["C", "clinic_pcen"] > 0  # entirely from decayed neighbors


def test_second_order_neighbor_excluded(expected):
    got = _frame("pop")
    assert "A" not in set(got.loc["C", "nbrs_bbox"])


def test_barrier_rule_is_global_and_directed(expected):
    got = _frame("pop")
    assert "A" not in set(got.loc["B", "nbrs_bbox"])   # A stripped from B
    assert set(got.loc["A", "nbrs_bbox"]) == {"B", "E"}  # A keeps its own


def test_popdensity_differs_from_popsize(expected):
    pop = _frame("pop")
    dens = _frame("popdensity")
    assert pop.loc["E", "clinic_pcen"] != pytest.approx(
        dens.loc["E", "clinic_pcen"], abs=1e-15)


def test_road_decay_divergence(expected):
    """Code roads are decayed; Eq. 4 has no neighbor term (rule-set gap #3)."""
    got = _frame("pop")
    ideal = expected[(expected["rule"] == "ideal")
                     & (expected["scenario"] == "baseline")
                     & (expected["denom"] == "pop")
                     & (expected["metric"] == "road_pcen")] \
        .set_index("settlement")["value"]
    assert got.loc["D", "road_pcen"] == pytest.approx(0.003, abs=1e-12)
    assert ideal["D"] == 0.0
    assert got.loc["A", "road_pcen"] == pytest.approx(0.010606601717798213,
                                                      abs=1e-12)
    assert ideal["A"] == pytest.approx(0.0075, abs=1e-12)


def test_second_normalization_divergence(expected):
    got = _frame("pop")
    assert got["norm_psi"].min() == pytest.approx(0.0, abs=1e-12)
    assert got["norm_psi"].max() == pytest.approx(1.0, abs=1e-12)
    assert not got["unnorm_psi"].equals(got["norm_psi"])


def test_minmax_anchors_unique(expected):
    got = _frame("pop")
    for svc in ("clinic", "school"):
        pcen = got[f"{svc}_pcen"]
        assert (pcen == pcen.max()).sum() == 1, svc
        assert (pcen == pcen.min()).sum() == 1, svc


@pytest.mark.parametrize("denom", ["pop", "popdensity"])
def test_production_collapse_gap5(expected, denom):
    """Rule-set gap #5, pinned against PRODUCTION: dropping rows after
    neighbor computation (absent_neighbor=swallowed drops the missing
    contributions) equals dropping them before — semantics (a) degenerates to
    (b) in the real code, not just in the reference impl's model of it."""
    post = _frame(denom, types=("RV", "IND"), stage="post_neighbors")
    pre = _frame(denom, types=("RV", "IND"), stage="pre_neighbors")
    assert set(post.index) == set(pre.index)
    for col in [c for c in post.columns
                if c.endswith(("_pcen", "_idx")) or c in ("unnorm_psi",
                                                          "norm_psi")]:
        for sid in post.index:
            assert post.loc[sid, col] == pytest.approx(
                pre.loc[sid, col], abs=1e-12), (denom, sid, col)


def test_gap6_border_point_is_double_counted_by_production():
    """Rule-set gap #6 (found by code-review round 2 mutation testing).

    Production's point counting uses gpd.sjoin's default `intersects`
    predicate, so a service point lying exactly on a shared settlement border
    is counted for BOTH neighbors. The manuscript's per-settlement counts say
    only "within an administrative unit" and are silent on the boundary case;
    the reference impl resolves that as strict containment, counting it for
    neither.

    Measured against the real Delhi layers (Aug 2026), this gap is LATENT:
    zero service points lie exactly on a colony boundary in any of the six
    point layers (closest approach 1.3 mm). The real double-counting today
    comes from a different mechanism — 4,050 overlapping colony polygon
    pairs put ~450 service points inside two or more colonies, which
    `within` would not fix. Both are routed to the Phase 3 bug audit.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    from delhi_psi import index
    from tests.reference_impl import _service_amounts

    city = load_settlements()
    # (1_001_000, 1_001_500) lies exactly on the A|B shared edge
    border_point = gpd.GeoDataFrame(
        {"service": ["clinic"]},
        geometry=[Point(1_001_000, 1_001_500)], crs=city.crs)

    counted = index.point_counts(city.copy(), border_point,
                                 count_col="probe_count")
    counts = counted.set_index("USO_AREA_U")["probe_count"]
    assert counts["A"] == 1 and counts["B"] == 1, "production double-counts"

    ref = _service_amounts(city, {"clinic": border_point,
                                  "road": load_services()["road"]})
    assert ref["clinic"]["A"] == 0 and ref["clinic"]["B"] == 0, \
        "reference impl (manuscript-literal `within`) counts it for neither"


def test_reprojection_is_load_bearing():
    """Feed a service layer in a different CRS and require the same answers.

    Code review round 2: every fixture is already EPSG:7760, so reprojection
    could be replaced by the identity function with a green suite — yet every
    real service layer depends on it.
    """
    from delhi_psi.pipeline import compute_frames

    services = load_services()
    services_wgs84 = dict(services)
    services_wgs84["clinic"] = services["clinic"].to_crs(epsg=4326)
    assert services_wgs84["clinic"].crs.to_epsg() == 4326

    from tests.oraculum_fixtures import methodology_with
    methodology = methodology_with(PROFILE, types=(), stage="post_neighbors")
    baseline = compute_frames(load_settlements(), {"canal": load_barriers()},
                              services, None, methodology, "pop")
    reprojected = compute_frames(load_settlements(),
                                 {"canal": load_barriers()},
                                 services_wgs84, None, methodology, "pop")
    for sid in baseline["USO_AREA_U"]:
        got = reprojected[reprojected["USO_AREA_U"] == sid]["clinic_pcen"].iloc[0]
        exp = baseline[baseline["USO_AREA_U"] == sid]["clinic_pcen"].iloc[0]
        assert got == pytest.approx(exp, abs=1e-12), sid


# --- step-0 snapshot backend (retired in migration step 5) -------------
# `scripts/generate_production_fixtures.py` still generates
# tests/fixtures/oraculum/production/code-2025.csv through these two, so the
# snapshot keeps being produced by the SAME wiring it was created with. They
# are deleted in the task that swaps the generator to compute_frames and
# proves a no-op diff. Do not delete them earlier.
SCENARIO_WIRING = [
    # (scenario, drop_pre, drop_post)
    ("baseline", frozenset(), frozenset()),
    ("excl_rv_only", frozenset(), frozenset({"RV"})),
    ("excl_contributing", frozenset(), frozenset({"RV", "IND"})),
    ("excl_removed", frozenset({"RV", "IND"}), frozenset()),
    ("excl_ind_removed", frozenset({"IND"}), frozenset()),
]


def _production_frame(denom, drop_ids_post=frozenset(),
                      drop_ids_pre=frozenset()):
    city = load_settlements()
    if drop_ids_pre:
        city = city[~city["USO_AREA_U"].isin(drop_ids_pre)]
    result = run_production_chain(
        city, load_barriers(), load_services(), denom,
        drop_ids_post=drop_ids_post)
    return result.set_index("USO_AREA_U")
```

Create `tests/test_profiles_match_reference.py`:

```python
"""Every shipped profile reproduces the reference at its mapped knobs.

Called through compute_frames with an explicit `denominator=`, for BOTH
reference denominators, independent of the profile's outputs.denominators
(spec § 7). The mapping test reads the SAME table the config enums are
generated from, so a value with no reference knob cannot be added silently.
"""
from pathlib import Path

import pandas as pd
import pytest

from delhi_psi.config import ENUMS, ENUM_KEYS, REFERENCE_KNOBS, load_config
from tests.oraculum_fixtures import (
    ORACLE_SCENARIOS, compute_oracle_frame, load_barriers, load_services,
    load_settlements,
)
from tests.reference_impl import RULESETS, compute_city

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"
REFERENCE_DENOMS = ("pop", "popdensity")

# profile -> the reference rule-set whose rows it must reproduce (spec § 4)
PROFILE_RULES = {"code-2025": "code"}

# production column -> reference metric
METRIC_MAP = {
    "clinic_pcen": "clinic_pcen", "clinic_idx": "clinic_idx",
    "school_pcen": "school_pcen", "school_idx": "school_idx",
    "bank_pcen": "bank_pcen", "bank_idx": "bank_idx",
    "police_pcen": "police_pcen", "police_idx": "police_idx",
    "ration_pcen": "ration_pcen", "ration_idx": "ration_idx",
    "transport_pcen": "transport_pcen", "transport_idx": "transport_idx",
    "road_pcen": "road_pcen", "road_idx": "road_idx",
    "road_length": "road_length_km",
    "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
}


@pytest.fixture(scope="module")
def expected():
    return pd.read_csv(CSV)


def reference_block(expected, rule, scenario, denom):
    sub = expected[(expected["rule"] == rule)
                   & (expected["scenario"] == scenario)
                   & (expected["denom"] == denom)]
    return sub.pivot(index="settlement", columns="metric", values="value")


def metrics_for(profile):
    cfg = load_config(profile)
    skip = set() if cfg.methodology.second_normalization else {"norm_psi"}
    return {k: v for k, v in METRIC_MAP.items() if k not in skip}


@pytest.mark.parametrize("denom", REFERENCE_DENOMS)
@pytest.mark.parametrize("scenario,types,stage", ORACLE_SCENARIOS)
@pytest.mark.parametrize("profile", sorted(PROFILE_RULES))
def test_profile_matches_reference(expected, profile, scenario, types, stage,
                                   denom):
    exp = reference_block(expected, PROFILE_RULES[profile], scenario, denom)
    got = compute_oracle_frame(profile, types=types, stage=stage, denom=denom)
    assert set(got.index) == set(exp.index)
    for prod_col, metric in metrics_for(profile).items():
        for sid in exp.index:
            assert got.loc[sid, prod_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-12), (profile, scenario, denom,
                                                   sid, prod_col)


def test_enums_cover_exactly_the_reference_table():
    assert set(ENUMS) == set(ENUM_KEYS)
    for key in ENUM_KEYS:
        assert {m.value for m in ENUMS[key]} == set(REFERENCE_KNOBS[key]), key


def test_every_mapped_knob_is_one_the_reference_actually_implements():
    """Drive compute_city once per mapped knob value; an unimplemented knob
    raises ValueError inside the reference, so this fails loudly."""
    city, barriers, services = (load_settlements(), load_barriers(),
                                load_services())
    base = dict(RULESETS["code"], scenario="baseline", denom="pop")
    knob_for_key = {
        "methodology.adjacency.rule": "adjacency_rule",
        "methodology.barrier.rule": "barrier_rule",
        "methodology.roads": "roads_formula",
        "methodology.second_normalization": "second_norm",
        "methodology.exclusion.absent_neighbor": "absent_neighbor_contribution",
        "outputs.denominators[]": "denom",
    }
    for key, knob in knob_for_key.items():
        for config_value, reference_value in REFERENCE_KNOBS[key].items():
            kwargs = dict(base)
            kwargs[knob] = reference_value
            frame = compute_city(city, services, barriers, **kwargs)
            assert len(frame) == 7, (key, config_value)


def test_exclusion_stage_maps_onto_dropped_before_neighbors():
    """`stage` has no compute_city keyword — it selects the SCENARIO, whose
    second element is `dropped_before_neighbors` (spec § 3 table)."""
    from tests.reference_impl import SCENARIOS

    stage_of = REFERENCE_KNOBS["methodology.exclusion.stage"]
    for scenario, types, stage in ORACLE_SCENARIOS:
        _, drop_before = SCENARIOS[scenario]
        assert drop_before is stage_of[stage], scenario
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_oracle.py tests/test_profiles_match_reference.py -q`
Expected: both fail to collect with
`ModuleNotFoundError: No module named 'delhi_psi.pipeline'`.

- [ ] **Step 3: Write `delhi_psi/pipeline.py`**

```python
"""Pipeline stages. `compute_frames` is the in-memory seam the oracle uses;
`preprocess` / `compute` (added next) are the path-based stages.

This is the ONLY module that sees a Config. The math modules take explicit
keyword arguments.
"""

import logging

from delhi_psi import geometry, index, neighbors, validate

log = logging.getLogger(__name__)

ID_COL = "USO_AREA_U"
TYPE_COL = "USO_FINAL"
NBRS_COL = "nbrs_bbox"
NBRS_DIST_COL = "nbrs_dist_bbox"
CENTROID_COL = "centroid"

POINT_GEOMS = frozenset({"Point", "MultiPoint"})
LINE_GEOMS = frozenset({"LineString", "MultiLineString", "LinearRing"})


def service_kind(name, gdf):
    """Classify a service layer as "point" or "line" from its geometries."""
    kinds = set(gdf.geom_type.dropna().unique())
    if kinds and kinds <= POINT_GEOMS:
        return "point"
    if kinds and kinds <= LINE_GEOMS:
        return "line"
    raise ValueError(
        f"service layer {name!r}: cannot classify geometry types "
        f"{sorted(kinds)}; expected every geometry to be a point or a line")


def service_layout(services):
    """[(service, kind, amount_col)], point services before line services —
    the order production used, so output columns keep their familiar order."""
    layout = [(name, service_kind(name, gdf)) for name, gdf in services.items()]
    layout.sort(key=lambda item: item[1] == "line")
    return [(name, kind, index.service_amount_column(name, kind))
            for name, kind in layout]


def attach_population(settlements, population, *, id_col=ID_COL,
                      population_id_col="uso_area_u",
                      population_value_col="population"):
    """Attach a `population` column; return (frame, ids with no population).

    population=None means the settlements frame already carries the column
    (the oracle city does); the missing rule then applies to that column.
    Otherwise this is compute_psi.py's merge, verbatim: rename to
    population_new, keep two columns, left-merge, drop the join key, rename
    back. Nothing is dropped here — the exclusion `stage` decides when.
    """
    if population is None:
        out = settlements.copy()
    else:
        updated = population.rename(
            columns={population_value_col: "population_new"})
        updated = updated[["population_new", population_id_col]]
        out = settlements.merge(updated, how="left", left_on=id_col,
                                right_on=population_id_col)
        out = out.drop(columns=[population_id_col])
        out = out.rename(columns={"population_new": "population"})
    missing = frozenset(out.loc[out["population"].isna(), id_col])
    return out, missing


def excluded_ids(frame, *, types, id_col=ID_COL, type_col=TYPE_COL):
    """Ids whose settlement type is in `types` (raw USO_FINAL strings)."""
    if not types:
        return frozenset()
    return frozenset(frame.loc[frame[type_col].isin(list(types)), id_col])


def build_neighbors(settlements, barriers, methodology, *, epsg_code=7760,
                    id_col=ID_COL):
    """Barrier flags, combined flag, centroids, neighbour lists, distances.

    Always built on the FULL settlement universe — preprocess never excludes
    (spec § 3).
    """
    frame = geometry.barrier_flags(settlements, barriers, id_col=id_col)
    frame = neighbors.combine_barrier_flags(
        frame, layers=tuple(barriers), combine=methodology.barrier.combine)
    frame[CENTROID_COL] = frame.centroid

    frame = neighbors.adjacency(frame, id_col=id_col, neighbor_col=NBRS_COL,
                                rule=methodology.adjacency.rule)
    barrier_geoms = [geom for gdf in barriers.values() for geom in gdf.geometry]
    frame = neighbors.apply_barrier(frame, barrier_geoms, id_col=id_col,
                                    neighbor_col=NBRS_COL,
                                    rule=methodology.barrier.rule)
    frame = neighbors.centroid_distances(
        frame, neighbor_col=NBRS_COL, nbr_dist_col=NBRS_DIST_COL,
        centroid_col=CENTROID_COL, id_col=id_col)
    frame["index"] = frame.index
    return frame


def apply_exclusion(neighbor_frame, *, dropped, stage, id_col=ID_COL):
    """Return (universe, reported).

    universe — the frame neighbour AMOUNTS are read from.
    reported — the rows that get PCEN and index values.

    post_neighbors: excluded rows leave the reported frame; their ids stay in
        other settlements' neighbour lists (today's production).
    pre_neighbors: excluded ids are ALSO stripped from every neighbour list.
        Stripping is exactly what re-running adjacency on the reduced universe
        would give — adjacency and the barrier rules are pairwise, so removing
        a row removes precisely that id from every other list — and it is what
        the universe-wide stored artifact allows (spec § 3).
    """
    if stage not in ("post_neighbors", "pre_neighbors"):
        raise ValueError(
            f"unknown exclusion stage {stage!r}; allowed values: "
            "['post_neighbors', 'pre_neighbors']")
    universe = neighbor_frame.copy()
    if stage == "pre_neighbors" and dropped:
        for idx, row in universe.iterrows():
            universe.at[idx, NBRS_COL] = [
                j for j in row[NBRS_COL] if j not in dropped]
            universe.at[idx, NBRS_DIST_COL] = [
                (j, d) for j, d in row[NBRS_DIST_COL] if j not in dropped]
        universe = universe[~universe[id_col].isin(dropped)]
    reported = universe[~universe[id_col].isin(dropped)] if dropped else universe
    return universe, reported


def index_frames(neighbor_frame, services, methodology, denominator, *,
                 dropped=frozenset(), epsg_code=7760, id_col=ID_COL):
    """Amounts, PCEN, min-max and the overall PSI for one denominator."""
    exclusion = methodology.exclusion
    universe, _ = apply_exclusion(neighbor_frame, dropped=dropped,
                                  stage=exclusion.stage, id_col=id_col)

    # Own amounts are computed over the WHOLE universe, so excluded
    # settlements still have something to lend under absent_neighbor
    # "contributes". They are per-row independent, so computing them for rows
    # that are dropped a moment later cannot change a kept row's value.
    amounts = universe
    layout = service_layout(services)
    for service, kind, amount_col in layout:
        projected = geometry.reproject(services[service], epsg_code)
        if kind == "point":
            amounts = index.point_counts(amounts, projected,
                                         count_col=amount_col, id_col=id_col)
        else:
            amounts = index.road_lengths(amounts, projected,
                                         length_col=amount_col, id_col=id_col)

    out = amounts[~amounts[id_col].isin(dropped)] if dropped else amounts

    for service, kind, amount_col in layout:
        include_neighbors = not (kind == "line"
                                 and methodology.roads == "eq4_own_only")
        out = index.service_index(
            out, amount_col, service=service, denominator=denominator,
            nbr_dist_col=NBRS_DIST_COL, lookup_frame=amounts,
            absent_neighbor=exclusion.absent_neighbor,
            include_neighbors=include_neighbors,
            decay_form=methodology.decay.form,
            distance_unit=methodology.decay.distance_unit,
            id_col=id_col)

    return index.overall_psi(
        out, second_normalization=methodology.second_normalization)


def compute_frames(settlements, barriers, services, population, methodology,
                   denominator, *, epsg_code=7760, id_col=ID_COL,
                   type_col=TYPE_COL, population_id_col="uso_area_u",
                   population_value_col="population",
                   missing_population="drop", max_missing_population=None):
    """The documented in-memory entry point (spec § 2).

    settlements: settlement polygons with `area_km2` (and `population` when
        `population` is None).
    barriers: {layer name: GeoDataFrame} — every layer gets its own flag
        column; `methodology.barrier.combine` decides which OR into `barrier`.
    services: {service name: GeoDataFrame}; point/line is read off the
        geometries. Output columns are named after these keys — the oracle
        fixture's `clinic` maps to config `health` in the path-based stages'
        test wiring, exactly as tests/test_oracle_e2e.SERVICE_LAYOUT does.
    population: the population table, or None when settlements already carry
        the column.
    methodology: a MethodologyConfig. Exclusion overrides are applied by
        constructing a modified MethodologyConfig, never by mutating frames.
    denominator: "pop" | "popdensity" | "one".
    """
    frame, missing = attach_population(
        settlements, population, id_col=id_col,
        population_id_col=population_id_col,
        population_value_col=population_value_col)
    if missing and missing_population == "error":
        raise validate.ValidationError(
            f"{len(missing)} settlements have no population row and "
            "layers.population.missing is 'error': "
            f"{sorted(missing)[:10]}")
    if max_missing_population is not None:
        validate.check_missing_population(
            len(missing), maximum=max_missing_population)

    neighbor_frame = build_neighbors(frame, barriers, methodology,
                                     epsg_code=epsg_code, id_col=id_col)
    dropped = excluded_ids(neighbor_frame, types=methodology.exclusion.types,
                           id_col=id_col, type_col=type_col) | set(missing)
    return index_frames(neighbor_frame, services, methodology, denominator,
                        dropped=dropped, epsg_code=epsg_code, id_col=id_col)
```

- [ ] **Step 4: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_oracle.py tests/test_profiles_match_reference.py -q`
Expected: 34 passed (`test_oracle.py` 21 — the same count as before the rewrite — plus `test_profiles_match_reference.py` 13).

Run: `uv run pytest -q -W error` — Expected: **167 passed** (154 + 13).

If a re-expressed `test_oracle.py` case needs a changed expected value,
**stop and report** (spec § 10: a carried-over test may not have its expected
value edited).

- [ ] **Step 5: Prove the snapshot did not move**

```bash
uv run python scripts/generate_production_fixtures.py
git diff --exit-code -- tests/fixtures/ && echo NO-DRIFT
```
Expected: `NO-DRIFT`. The generator still runs the legacy backend, which is
the point — the snapshot is untouched while `compute_frames` is proven
against the reference independently.

- [ ] **Step 6: Commit**

```bash
git add delhi_psi/pipeline.py tests/oraculum_fixtures.py tests/test_oracle.py \
        tests/test_profiles_match_reference.py
git commit -m "feat: pipeline.compute_frames — config-driven in-memory seam (DEL-18, DEL-22)"
```

---

### Task 8: path stages + `cli.py` (spec §§ 5 step 3, 6)

**Files:**
- Modify: `delhi_psi/pipeline.py` (add `preprocess`, `compute` and their result dataclasses)
- Create: `delhi_psi/cli.py`
- Create: `tests/test_cli.py`
- Modify: `tests/test_oracle_e2e.py` (rewired from `scripts/*.py` subprocesses to the CLI)

**Interfaces:**
- Consumes: everything from Tasks 2–7.
- Produces:
  - `delhi_psi.pipeline.PreprocessResult` (frozen dataclass: `neighbors_path: Path`, `n_settlements: int`, `n_barrier_flagged: int`, `reports: tuple[LayerReport, ...]`)
  - `delhi_psi.pipeline.ComputeResult` (frozen dataclass: `outputs: tuple[Path, ...]`, `missing_population_path: Path`, `n_missing_population: int`, `n_reported: int`)
  - `delhi_psi.pipeline.preprocess(cfg) -> PreprocessResult`
  - `delhi_psi.pipeline.compute(cfg) -> ComputeResult`
  - `delhi_psi.pipeline.output_basename(cfg, denominator) -> str` — `cfg.outputs.name_template.format(profile=cfg.profile, denominator=denominator)`
  - `delhi_psi.cli.main(argv=None) -> int` — exit 0 ok, 1 validation/IO failure, 2 bad configuration or usage

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli.py`:

```python
"""Both CLI stages on the Oraculum temp dir: csv and shp, exit codes,
--config by name and by path.

The shp case runs IN-PROCESS on purpose: under `-W error` it is the only
thing that exercises io._write_shapefile's warning filter. The old e2e test
ran the scripts in a subprocess, which is why the two shapefile warnings were
invisible (spec § 6).
"""
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from delhi_psi import cli
from delhi_psi.config import PROFILES_DIR
from tests.oraculum_fixtures import (
    EPSG, load_barriers, load_services, load_settlements,
)

SERVICE_LAYOUT = {
    "clinic": ("Health", "Health.shp"),
    "school": ("School", "schools7760.shp"),
    "bank": ("Banking", "Banking.shp"),
    "police": ("Police", "Police Station.shp"),
    "ration": ("Ration", "Ration.shp"),
    "transport": ("Transport", "Transport.shp"),
    "road": ("Major Road", "Road.shp"),
}


def _empty_line_layer(path):
    gdf = gpd.GeoDataFrame(
        {"name": ["placeholder"]},
        geometry=[LineString([(0, 0), (1, 1)])], crs=f"EPSG:{EPSG}")
    gdf.geometry = gdf.translate(xoff=2_000_000, yoff=2_000_000)
    gdf.to_file(path)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """The spec § 3 manifest, laid out at the code-2025 default paths."""
    root = tmp_path_factory.mktemp("oraculum_data")
    city = load_settlements()

    (root / "uso_update_sep2021").mkdir()
    # Drop `population`: compute merges the population CSV and renames it to
    # `population`; a second column of that name collides. The real dataset's
    # shapefile has no population field either.
    city.drop(columns=["population"]).to_file(
        root / "uso_update_sep2021" / "uso_update_sep2021.shp")

    barrier_dir = root / "Barrier_Clip"
    (barrier_dir / "Canal").mkdir(parents=True)
    load_barriers().to_file(barrier_dir / "Canal" / "Canal.shp")
    for sub, fname in [("Drain", "Major_Drain.shp"),
                       ("Railway", "Railway_Line.shp")]:
        (barrier_dir / sub).mkdir()
        _empty_line_layer(barrier_dir / sub / fname)

    (root / "ndmc_center7760").mkdir()
    gpd.GeoDataFrame({"name": ["ndmc"]},
                     geometry=[Point(1_001_500, 1_001_500)],
                     crs=f"EPSG:{EPSG}").to_file(
        root / "ndmc_center7760" / "ndmc_center7760.shp")

    (root / "delhi_bounds_buffer").mkdir()
    gpd.GeoDataFrame({"name": ["bounds"]},
                     geometry=[box(1_000_000 - 10_000, 1_000_000 - 10_000,
                                   1_000_000 + 10_000, 1_000_000 + 10_000)],
                     crs=f"EPSG:{EPSG}").to_file(
        root / "delhi_bounds_buffer" / "delhi_bounds_buffer.shp")

    services = load_services()
    for svc, (folder, fname) in SERVICE_LAYOUT.items():
        d = root / "Public Services" / folder
        d.mkdir(parents=True)
        services[svc].to_file(d / fname)

    pop = load_settlements()[["USO_AREA_U", "population"]].rename(
        columns={"USO_AREA_U": "uso_area_u"})
    pop.to_csv(root / "pop_colony_wp_2020_jjc_adjusted.csv", index=False)
    return root


def run(*args):
    return cli.main(list(args))


def test_preprocess_then_compute_by_profile_name(data_dir, tmp_path):
    out = tmp_path / "by_name"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    assert (out / "colonies_neighbors.joblib").exists()

    assert run("compute", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    for denom in ("pop", "popdensity"):
        base = f"delhi_psi_code-2025_{denom}_2020"
        # formats: [csv, shp, joblib] — the shp write happened IN-PROCESS
        # under -W error, so the warning filter is exercised here
        for suffix in (".csv", ".shp", ".joblib"):
            assert (out / f"{base}{suffix}").exists(), base + suffix
    assert (out / "missing_population.csv").exists()


def test_config_by_path_is_equivalent(data_dir, tmp_path):
    out = tmp_path / "by_path"
    assert run("preprocess", "--config", str(PROFILES_DIR / "code-2025.yaml"),
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    assert (out / "colonies_neighbors.joblib").exists()


def test_shapefile_columns_drop_the_unserializable_ones(data_dir, tmp_path):
    out = tmp_path / "shp"
    run("preprocess", "--config", "code-2025", "--data-dir", str(data_dir),
        "--out-dir", str(out))
    run("compute", "--config", "code-2025", "--data-dir", str(data_dir),
        "--out-dir", str(out))
    shp = gpd.read_file(out / "delhi_psi_code-2025_pop_2020.shp")
    for dropped in ("nbrs_bbox", "nbrs_dist_bbox", "centroid"):
        assert dropped not in shp.columns


def test_csv_output_carries_the_baseline_columns(data_dir, tmp_path):
    out = tmp_path / "csv"
    run("preprocess", "--config", "code-2025", "--data-dir", str(data_dir),
        "--out-dir", str(out))
    run("compute", "--config", "code-2025", "--data-dir", str(data_dir),
        "--out-dir", str(out))
    got = pd.read_csv(out / "delhi_psi_code-2025_pop_2020.csv")
    for column in ("USO_AREA_U", "population", "area_km2", "ndmc_dist_km",
                   "road_length", "unnorm_psi", "norm_psi", "health_idx"):
        assert column in got.columns, column


def test_unknown_profile_exits_2(data_dir, tmp_path):
    assert run("compute", "--config", "no-such-profile",
               "--data-dir", str(data_dir),
               "--out-dir", str(tmp_path / "x")) == 2


def test_missing_input_layer_exits_1(tmp_path):
    empty = tmp_path / "empty_data"
    empty.mkdir()
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(empty),
               "--out-dir", str(tmp_path / "y")) == 1


def test_unknown_stage_exits_2(data_dir, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run("frobnicate", "--config", "code-2025")
    assert exc.value.code == 2


def test_ndmc_center_outside_bounds_exits_1(data_dir, tmp_path):
    # preprocess runs the layer battery on the NDMC point too (spec § 6).
    import shutil
    d = tmp_path / "d"
    shutil.copytree(data_dir, d)
    gpd.GeoDataFrame({"name": ["far"]},
                     geometry=[Point(9_000_000, 9_000_000)],
                     crs=f"EPSG:{EPSG}").to_file(
        d / "ndmc_center7760" / "ndmc_center7760.shp")
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(d), "--out-dir", str(tmp_path / "o")) == 1


def test_service_layer_without_crs_exits_1(data_dir, tmp_path):
    # compute refuses a service layer that has no CRS (spec § 6 CRS check).
    import shutil
    d = tmp_path / "d"
    shutil.copytree(data_dir, d)
    out = tmp_path / "o"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(d), "--out-dir", str(out)) == 0
    bank = d / "Public Services" / "Banking" / "Banking.shp"
    gpd.read_file(bank).set_crs(None, allow_override=True).to_file(bank)
    assert run("compute", "--config", "code-2025",
               "--data-dir", str(d), "--out-dir", str(out)) == 1
```

Rewrite `tests/test_oracle_e2e.py` to drive the CLI as a subprocess (the
console script is the thing under test), reusing the fixture layout:

```python
"""Real-CLI end-to-end: temp data dir -> delhi-psi preprocess -> compute.

The code-2025 profile excludes RV, so the output is compared against
rule=code / scenario=excl_rv_only. This leg stays a SUBPROCESS run: it proves
the installed console script works, which the in-process tests in
tests/test_cli.py cannot.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.test_cli import data_dir  # noqa: F401  (module-scoped fixture)

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"


def _run(*args):
    proc = subprocess.run([sys.executable, "-m", "delhi_psi.cli", *args],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


def test_full_cli_chain_matches_excl_rv_only(data_dir, tmp_path):  # noqa: F811
    out_dir = tmp_path / "out"
    _run("preprocess", "--config", "code-2025", "--data-dir", str(data_dir),
         "--out-dir", str(out_dir))
    assert (out_dir / "colonies_neighbors.joblib").exists()
    _run("compute", "--config", "code-2025", "--data-dir", str(data_dir),
         "--out-dir", str(out_dir))

    got = pd.read_csv(out_dir / "delhi_psi_code-2025_pop_2020.csv")
    got = got.set_index("USO_AREA_U")

    exp = pd.read_csv(CSV)
    exp = exp[(exp["rule"] == "code") & (exp["scenario"] == "excl_rv_only")
              & (exp["denom"] == "pop")] \
        .pivot(index="settlement", columns="metric", values="value")

    assert set(got.index) == set(exp.index) == {"A", "B", "C", "D", "E", "IND"}
    # real pipeline service naming: the fixture's clinic layer is written to
    # Public Services/Health/Health.shp, so the config service is `health`
    mapping = {
        "health_pcen": "clinic_pcen", "health_idx": "clinic_idx",
        "school_pcen": "school_pcen", "school_idx": "school_idx",
        "bank_pcen": "bank_pcen", "police_pcen": "police_pcen",
        "ration_pcen": "ration_pcen", "transport_pcen": "transport_pcen",
        "road_pcen": "road_pcen", "road_idx": "road_idx",
        "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
    }
    for got_col, metric in mapping.items():
        for sid in exp.index:
            assert got.loc[sid, got_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-9), (got_col, sid)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py tests/test_oracle_e2e.py -q`
Expected: collection error — `ImportError: cannot import name 'cli' from 'delhi_psi'`.

- [ ] **Step 3: Add the path stages to `delhi_psi/pipeline.py`**

Append to `delhi_psi/pipeline.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from delhi_psi import io


@dataclass(frozen=True)
class PreprocessResult:
    neighbors_path: Path
    n_settlements: int
    n_barrier_flagged: int
    reports: tuple


@dataclass(frozen=True)
class ComputeResult:
    outputs: tuple
    missing_population_path: Path
    n_missing_population: int
    n_reported: int


def output_basename(cfg, denominator):
    return cfg.outputs.name_template.format(profile=cfg.profile,
                                            denominator=str(denominator))


def _dedup_cached(gdf, cache_dir, name, source_path):
    """Deduplicate once, caching under out_dir keyed on source mtime+size.

    The O(n^2) algorithm is unchanged (spec § 6); only the cache location and
    the staleness rule move here from scripts/preprocess.py, which keyed the
    cache on existence alone.
    """
    stat = Path(source_path).stat()
    stamp = cache_dir / f"{name}.dedup.stamp"
    cached = cache_dir / f"{name}.dedup"
    signature = f"{stat.st_mtime_ns}:{stat.st_size}\n"
    if cached.exists() and stamp.exists() and stamp.read_text() == signature:
        log.info("reusing dedup cache %s", cached)
        return io.read_layer(cached)
    deduped = geometry.remove_duplicate_geom(gdf)
    cache_dir.mkdir(parents=True, exist_ok=True)
    deduped.to_file(cached, index=False)
    stamp.write_text(signature)
    return deduped


def preprocess(cfg):
    """Settlements + barriers -> the universe-wide neighbours artifact."""
    data_dir = cfg.paths.data_dir
    out_dir = io.resolve_out_dir(cfg.paths.out_dir, data_dir)

    bounds = io.read_layer(data_dir / cfg.layers.bounds)
    settlements = io.read_layer(data_dir / cfg.layers.settlements.path)
    barriers = {name: io.read_layer(data_dir / path)
                for name, path in cfg.layers.barriers.items()}

    reports = [validate.require_layer(settlements, name="settlements",
                                      geom_type="Polygon", bounds_gdf=bounds)]
    for name, gdf in barriers.items():
        reports.append(validate.require_layer(gdf, name=name,
                                              geom_type="Line",
                                              bounds_gdf=bounds))

    settlements = _dedup_cached(settlements, out_dir, "settlements",
                                data_dir / cfg.layers.settlements.path)
    barriers = {
        name: _dedup_cached(gdf, out_dir, name,
                            data_dir / cfg.layers.barriers[name])
        for name, gdf in barriers.items()}

    epsg = cfg.crs.epsg
    settlements = geometry.reproject(settlements, epsg)
    barriers = {name: geometry.reproject(gdf, epsg)
                for name, gdf in barriers.items()}
    validate.check_crs_match({"settlements": settlements, **barriers})

    settlements["area_km2"] = settlements.area / 1000000
    drop = {"index", "level_0"}.intersection(settlements.columns)
    settlements = settlements.drop(columns=drop)

    frame = build_neighbors(settlements, barriers, cfg.methodology,
                            epsg_code=epsg,
                            id_col=cfg.layers.settlements.id_col)

    if cfg.layers.ndmc_center:
        centre = io.read_layer(data_dir / cfg.layers.ndmc_center)
        reports.append(validate.require_layer(centre, name="ndmc_center",
                                              geom_type="Point",
                                              bounds_gdf=bounds))
        centre = geometry.reproject(centre, epsg)
        frame = geometry.distance_to_point_km(
            frame, centre["geometry"].values[0], centroid_col=CENTROID_COL,
            out_col="ndmc_dist_km")

    path = io.write_neighbors(frame, out_dir / cfg.paths.neighbors_artifact)
    return PreprocessResult(
        neighbors_path=path,
        n_settlements=len(frame),
        n_barrier_flagged=int(frame["barrier"].sum()),
        reports=tuple(reports))


def compute(cfg):
    """Neighbours artifact + population + services -> one PSI set per
    outputs.denominators entry."""
    data_dir = cfg.paths.data_dir
    out_dir = io.resolve_out_dir(cfg.paths.out_dir, data_dir)
    id_col = cfg.layers.settlements.id_col

    neighbor_frame = io.read_neighbors(out_dir / cfg.paths.neighbors_artifact)
    bounds = io.read_layer(data_dir / cfg.layers.bounds)
    population = io.read_population(data_dir / cfg.layers.population.path)

    services = {}
    for name, path in {**cfg.services.point, **cfg.services.line}.items():
        gdf = io.read_layer(data_dir / path)
        geom_type = "Point" if name in cfg.services.point else "Line"
        validate.require_layer(gdf, name=name, geom_type=geom_type,
                               bounds_gdf=bounds)
        services[name] = gdf
    # Spec § 6: the compute stage's CRS check. Services are reprojected
    # per-service inside index_frames, so here we assert every input has a
    # CRS to reproject FROM and that the artifact is in the target CRS.
    validate.check_crs_defined({**services, "neighbors": neighbor_frame})
    if neighbor_frame.crs.to_epsg() != cfg.crs.epsg:
        raise validate.ValidationError(
            f"neighbors artifact is in {neighbor_frame.crs}, "
            f"config crs.epsg is {cfg.crs.epsg}")

    frame, missing = attach_population(
        neighbor_frame, population, id_col=id_col,
        population_id_col=cfg.layers.population.id_col,
        population_value_col=cfg.layers.population.value_col)
    if missing and cfg.layers.population.missing == "error":
        raise validate.ValidationError(
            f"{len(missing)} settlements have no population row and "
            f"layers.population.missing is 'error': {sorted(missing)[:10]}")
    validate.check_missing_population(
        len(missing), maximum=cfg.validate.max_missing_population)

    missing_path = out_dir / "missing_population.csv"
    frame[frame[id_col].isin(missing)].drop(
        columns=[c for c in io.SHAPEFILE_DROP_COLUMNS if c in frame.columns]
    ).to_csv(missing_path, index=False)

    dropped = excluded_ids(frame, types=cfg.methodology.exclusion.types,
                           id_col=id_col,
                           type_col=cfg.layers.settlements.type_col) \
        | set(missing)

    outputs = []
    n_reported = 0
    for denominator in cfg.outputs.denominators:
        result = index_frames(frame, services, cfg.methodology,
                              str(denominator), dropped=dropped,
                              epsg_code=cfg.crs.epsg, id_col=id_col)
        validate.check_no_negative(result)
        n_reported = len(result)
        outputs.extend(io.write_outputs(
            result, out_dir, basename=output_basename(cfg, denominator),
            formats=cfg.outputs.formats))

    return ComputeResult(outputs=tuple(outputs),
                         missing_population_path=missing_path,
                         n_missing_population=len(missing),
                         n_reported=n_reported)
```

- [ ] **Step 4: Write `delhi_psi/cli.py`**

```python
"""delhi-psi <stage> --config <profile-or-path> [--data-dir D] [--out-dir O].

Exit codes: 0 success, 1 a validation or IO failure, 2 a bad configuration or
a usage error (argparse's own code).
"""

import argparse
import logging
import sys

from delhi_psi import pipeline
from delhi_psi.config import ConfigError, load_config
from delhi_psi.validate import ValidationError

log = logging.getLogger("delhi_psi")

STAGES = {"preprocess": pipeline.preprocess, "compute": pipeline.compute}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="delhi-psi", description="Delhi Public Services Index pipeline")
    parser.add_argument("stage", choices=sorted(STAGES),
                        help="pipeline stage to run")
    parser.add_argument("--config", default="code-2025",
                        help="shipped profile name or path to a YAML file")
    parser.add_argument("--data-dir", default=None, help="input data root")
    parser.add_argument("--out-dir", default=None,
                        help="output directory (default: the data dir)")
    parser.add_argument("--log-level", default="INFO",
                        help="logging level (default: INFO)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config, data_dir=args.data_dir,
                          out_dir=args.out_dir)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        result = STAGES[args.stage](cfg)
    except ValidationError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, OSError) as exc:
        print(f"input/output error: {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_cli.py tests/test_oracle_e2e.py -q` — Expected: 8 passed (7 CLI + 1 e2e).
Run: `uv run pytest -q -W error` — Expected: **174 passed** (167 + 7).

If the shp case raises `PytestUnraisableExceptionWarning` about a
"Normalized/laundered field name", the filter's `message=` prefix does not
match — fix the filter, not the test (the message is quoted verbatim in
"Canonical facts").

- [ ] **Step 6: Check the console script is wired**

```bash
uv run delhi-psi --help
```
Expected: the usage line naming `{compute,preprocess}`.

- [ ] **Step 7: Prove the snapshot did not move, then commit**

```bash
uv run python scripts/generate_production_fixtures.py
git diff --exit-code -- tests/fixtures/ && echo NO-DRIFT
git add delhi_psi/pipeline.py delhi_psi/cli.py tests/test_cli.py tests/test_oracle_e2e.py
git commit -m "feat: preprocess/compute stages and the delhi-psi CLI (DEL-25)"
```

---

### Task 9: `delhi_psi/verify.py` and `--config` for the baseline script

**Files:**
- Create: `delhi_psi/verify.py` (the two comparison functions, moved verbatim)
- Modify: `scripts/verify_against_baseline.py` (thin wrapper; gains `--config`)
- Modify: `tests/test_verify.py` (import from `delhi_psi.verify`)

**Interfaces:**
- Consumes: `delhi_psi.config.load_config`, `delhi_psi.io.read_neighbors`, `delhi_psi.pipeline.output_basename`.
- Produces:
  - `delhi_psi.verify.RTOL = 1e-9`, `delhi_psi.verify.ATOL = 1e-12`
  - `delhi_psi.verify.compare_neighbor_frames(new_df, base_df) -> list[str]`
  - `delhi_psi.verify.compare_numeric_frames(new_df, base_df, id_col, rtol, atol) -> tuple[list[str], list[str]]`

- [ ] **Step 1: Write the failing test**

Change the import at the top of `tests/test_verify.py` (nothing else in that file changes — same functions, same expectations):

```python
from delhi_psi.verify import (
    compare_neighbor_frames,
    compare_numeric_frames,
)
```

Then add one new test to the bottom of `tests/test_verify.py`:

```python
def test_fresh_paths_come_from_the_config(tmp_path):
    """--config locates the fresh files; the baseline paths stay the
    script's own arguments (they exist only for code-2025)."""
    from scripts.verify_against_baseline import fresh_paths

    paths = fresh_paths("code-2025", verify_dir=tmp_path)
    assert paths["neighbors"] == tmp_path / "colonies_neighbors.joblib"
    assert paths["psi"]["pop"] == tmp_path / "delhi_psi_code-2025_pop_2020.csv"
    assert paths["psi"]["popdensity"] == \
        tmp_path / "delhi_psi_code-2025_popdensity_2020.csv"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_verify.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'delhi_psi.verify'`.

- [ ] **Step 3: Create `delhi_psi/verify.py`**

Move `compare_neighbor_frames` and `compare_numeric_frames` out of
`scripts/verify_against_baseline.py` **verbatim** — body, comments and
docstrings unchanged — into a new module:

```python
"""Baseline comparison functions (moved from scripts/verify_against_baseline).

A baseline column absent from the new run is itself a deviation — never
silently skipped.
"""

import numpy as np
import pandas as pd

RTOL = 1e-9
ATOL = 1e-12


def compare_neighbor_frames(new_df, base_df):
    ...   # verbatim from scripts/verify_against_baseline.py lines 22-61


def compare_numeric_frames(new_df, base_df, id_col, rtol, atol):
    ...   # verbatim from scripts/verify_against_baseline.py lines 64-117
```

**Copy the two bodies byte-for-byte from git** rather than retyping:

```bash
sed -n '18,117p' scripts/verify_against_baseline.py
```
gives `RTOL`, `ATOL` and both functions exactly as they must appear.

- [ ] **Step 4: Rewrite `scripts/verify_against_baseline.py` as a thin wrapper**

```python
"""Compare a fresh pipeline run against the read-only July 2025 baseline.

Baseline files are opened read-only; this script never writes to the data
directory. Exit code 0 = equivalent within tolerance; 1 = deviations found.

--config locates the FRESH files (they follow the profile's
paths.neighbors_artifact and outputs.name_template). The baseline paths stay
this script's own arguments, because they exist only for code-2025 — which is
why there is no `verify` CLI stage and no baseline key in the config schema.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from delhi_psi.config import load_config
from delhi_psi.io import read_neighbors, resolve_data_dir
from delhi_psi.pipeline import output_basename
from delhi_psi.verify import (
    ATOL, RTOL, compare_neighbor_frames, compare_numeric_frames,
)

BASELINE_NEIGHBORS = "colonies_bbox_nbrs2025.joblib"
BASELINE_PSI = {
    "pop": "psi_2020_results/delhi_psi_bbox_popsize2020_norv_12Sep2021.csv",
    "popdensity":
        "psi_2020_results/delhi_psi_bbox_popdensity2020_norv_12Sep2021.csv",
}


def fresh_paths(config, *, verify_dir, data_dir=None, out_dir=None):
    """Where the fresh run's files live, per the profile."""
    cfg = load_config(config, data_dir=data_dir, out_dir=out_dir)
    verify_dir = Path(verify_dir)
    return {
        "neighbors": verify_dir / cfg.paths.neighbors_artifact,
        "psi": {str(denominator):
                verify_dir / f"{output_basename(cfg, denominator)}.csv"
                for denominator in cfg.outputs.denominators},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that produced the fresh run")
    parser.add_argument("--data-dir", default=None,
                        help="data root holding the baseline")
    parser.add_argument(
        "--verify-dir", default=None,
        help="directory holding the fresh run (default: <data-dir>/phase3_verify)")
    parser.add_argument("--baseline-neighbors", default=None,
                        help=f"default: <data-dir>/{BASELINE_NEIGHBORS}")
    parser.add_argument("--baseline-psi-pop", default=None,
                        help=f"default: <data-dir>/{BASELINE_PSI['pop']}")
    parser.add_argument("--baseline-psi-popdensity", default=None,
                        help=f"default: <data-dir>/{BASELINE_PSI['popdensity']}")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    verify_dir = (Path(args.verify_dir).expanduser() if args.verify_dir
                  else data_dir / "phase3_verify")
    fresh = fresh_paths(args.config, verify_dir=verify_dir,
                        data_dir=args.data_dir)

    baseline_psi = {
        "pop": Path(args.baseline_psi_pop).expanduser()
        if args.baseline_psi_pop else data_dir / BASELINE_PSI["pop"],
        "popdensity": Path(args.baseline_psi_popdensity).expanduser()
        if args.baseline_psi_popdensity
        else data_dir / BASELINE_PSI["popdensity"],
    }
    baseline_neighbors = (Path(args.baseline_neighbors).expanduser()
                          if args.baseline_neighbors
                          else data_dir / BASELINE_NEIGHBORS)

    all_issues = []

    print("== Neighbors artifact ==")
    base_nbrs = read_neighbors(baseline_neighbors)
    new_nbrs = read_neighbors(fresh["neighbors"])
    issues = compare_neighbor_frames(new_nbrs, base_nbrs)
    print(f"  {len(base_nbrs)} baseline colonies; {len(issues)} issue(s)")
    all_issues.extend(issues)

    for denominator, new_path in fresh["psi"].items():
        if denominator not in baseline_psi:
            print(f"== psi {denominator} == (no baseline; skipped)")
            continue
        print(f"== psi {denominator} ==")
        base_df = pd.read_csv(baseline_psi[denominator])
        new_df = pd.read_csv(new_path)
        issues, report = compare_numeric_frames(new_df, base_df, "USO_AREA_U",
                                                RTOL, ATOL)
        print("\n".join(report))
        all_issues.extend(f"psi {denominator}: {i}" for i in issues)

    if all_issues:
        print(f"\nFAIL — {len(all_issues)} deviation(s) from baseline:")
        for issue in all_issues[:50]:
            print(f"  - {issue}")
        sys.exit(1)
    print("\nPASS — new run equivalent to July 2025 baseline within tolerance")


if __name__ == "__main__":
    main()
```

Note: the `sys.path.insert` and the `scripts.common` import are gone — the
editable install puts the repo root on `sys.path`.

- [ ] **Step 5: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_verify.py -q` — Expected: 9 passed (8 existing + 1 new).
Run: `uv run pytest -q -W error` — Expected: **175 passed** (174 + 1).

- [ ] **Step 6: Commit**

```bash
git add delhi_psi/verify.py scripts/verify_against_baseline.py tests/test_verify.py
git commit -m "refactor: move baseline comparison to delhi_psi.verify; --config for the fresh paths"
```

---

### Task 10: delete the monolith and the old scripts (spec § 5 step 4)

**Files:**
- Delete: `spatial_index_utils.py`, `scripts/preprocess.py`, `scripts/compute_psi.py`, `scripts/common.py`, `conftest.py`
- Modify: `tests/oraculum_fixtures.py` (`run_production_chain` rewired to `delhi_psi` — it stays alive as the step-0 snapshot backend until Task 11)
- Modify: `tests/test_divergence_exhibit.py`, `tests/test_fixture_invariants.py`, `tests/test_reference_impl.py` (drop the `spatial_index_utils` imports and the `sys.path` hack)
- Modify: `scripts/render_oracle_maps.py` (drop its `sys.path.insert`)
- Modify: `README.md` (it documents the deleted scripts)

**Interfaces:**
- Consumes: `delhi_psi.geometry`, `delhi_psi.neighbors`, `delhi_psi.index`, `delhi_psi.pipeline`.
- Produces: `tests.oraculum_fixtures.run_production_chain(settlements, barriers, services, pcen_denom, drop_ids_post=frozenset())` — unchanged signature and unchanged numbers, now implemented on `delhi_psi`.

> **Why `run_production_chain` survives this task.** Spec § 2 says
> `compute_frames` replaces it, and spec § 5 step 5 says the generator's
> backend swaps there — not here. The step-0 snapshot must keep being
> produced by its original wiring until the swap is proven a no-op, so this
> task only removes `run_production_chain`'s dependency on the deleted
> monolith. `test_production_fixtures.py` re-runs the generator on every
> commit, so any drift introduced by this rewiring fails immediately.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_production_fixtures.py`:

```python
def test_no_sys_path_hacks_and_no_monolith():
    """The package is installed; nothing may reach for the repo root."""
    import subprocess

    repo = PRODUCTION_DIR.parents[3]
    assert not (repo / "spatial_index_utils.py").exists()
    assert not (repo / "conftest.py").exists()
    for script in ("preprocess.py", "compute_psi.py", "common.py"):
        assert not (repo / "scripts" / script).exists(), script

    hits = subprocess.run(
        ["git", "grep", "-n", "sys.path.insert", "--",
         "*.py", ":!archive/"],
        cwd=repo, capture_output=True, text=True)
    assert hits.stdout == "", f"sys.path.insert still present:\n{hits.stdout}"

    imports = subprocess.run(
        ["git", "grep", "-n", "spatial_index_utils", "--",
         "*.py", ":!archive/"],
        cwd=repo, capture_output=True, text=True)
    assert imports.stdout == "", \
        f"spatial_index_utils still referenced:\n{imports.stdout}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_production_fixtures.py::test_no_sys_path_hacks_and_no_monolith -q`
Expected: FAIL — `AssertionError: assert not True` on `spatial_index_utils.py` existing.

- [ ] **Step 3: Rewire `tests/oraculum_fixtures.py`**

Replace the module's `import spatial_index_utils` with the package imports and
rewrite `run_production_chain`'s body to the same call sequence:

```python
from delhi_psi import geometry, index, neighbors
```

```python
def run_production_chain(settlements, barriers, services, pcen_denom,
                         drop_ids_post=frozenset()):
    """Preprocess-style neighbor computation + compute_psi-style indexing.

    The step-0 production snapshot's backend. It mirrors, call for call, the
    wiring the snapshot was generated with — now through delhi_psi rather than
    the deleted spatial_index_utils. Retired once the generator swaps to
    pipeline.compute_frames and the diff is proven a no-op.

    drop_ids_post: ids removed AFTER neighbor computation (the scripts'
    post-drop semantics — e.g. {'RV'} replicates compute_psi's RV filter).
    """
    colonies = geometry.barrier_flags(settlements.copy(),
                                      {"canal": barriers})
    colonies["barrier"] = colonies["canal"]
    colonies["centroid"] = colonies.centroid

    nbrs = neighbors.adjacency(colonies, id_col="USO_AREA_U",
                               neighbor_col="nbrs_bbox", rule="bbox")
    nbrs = neighbors.apply_barrier(nbrs, list(barriers.geometry),
                                   id_col="USO_AREA_U",
                                   neighbor_col="nbrs_bbox",
                                   rule="global_asymmetric",
                                   flag_col="barrier")
    nbrs = neighbors.centroid_distances(
        nbrs, neighbor_col="nbrs_bbox", nbr_dist_col="nbrs_dist_bbox",
        centroid_col="centroid", id_col="USO_AREA_U")
    nbrs["index"] = nbrs.index

    if drop_ids_post:
        nbrs = nbrs[~nbrs["USO_AREA_U"].isin(drop_ids_post)]

    layout = [(name, "line" if name == "road" else "point")
              for name in services]
    layout.sort(key=lambda item: item[1] == "line")

    out = nbrs
    for service, kind in layout:
        amount_col = index.service_amount_column(service, kind)
        projected = geometry.reproject(services[service], EPSG)
        if kind == "point":
            out = index.point_counts(out, projected, count_col=amount_col)
        else:
            out = index.road_lengths(out, projected, length_col=amount_col)
        out = index.service_index(out, amount_col, service=service,
                                  denominator=pcen_denom,
                                  nbr_dist_col="nbrs_dist_bbox",
                                  absent_neighbor="swallowed")
    return index.overall_psi(out, second_normalization=True)
```

- [ ] **Step 4: Rewire the three remaining test modules**

In `tests/test_divergence_exhibit.py`, replace both
`import spatial_index_utils` blocks with `from delhi_psi import geometry,
neighbors` and the two call sites:

```python
    bbox_gdf = geometry.bbox_frame(gdf)
    result = neighbors.apply_barrier(
        neighbors.adjacency(gdf, id_col="USO_AREA_U",
                            neighbor_col="nbrs_bbox", rule="bbox"),
        [], id_col="USO_AREA_U", neighbor_col="nbrs_bbox",
        rule="global_asymmetric", flag_col="barrier")
```

(the exhibit sets `gdf["barrier"] = False` and has no barrier layer, so
`apply_barrier` with an empty geometry list returns the lists unchanged —
which is exactly what the old call with an all-False flag column did).
`test_production_bbox_geometry_is_exact_envelope` becomes
`geometry.bbox_frame(gdf)`.

In `tests/test_fixture_invariants.py`, nothing changes: it uses
`run_production_chain`, which still exists.

In `tests/test_reference_impl.py`, replace the `sys.path` hack in
`test_invariants_guard_csv_wide`:

```python
def test_invariants_guard_csv_wide():
    from scripts.check_oraculum_invariants import check
    assert check() == []
```

and drop the now-unused `import sys` if nothing else in the module uses it.

- [ ] **Step 5: Rewire `scripts/render_oracle_maps.py`**

Delete these three lines:

```python
import sys                       # (line 14 — the import)

# ... and line 26, immediately after REPO is defined. KEEP the REPO line:
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))    # <- delete only this line
```

Keep `REPO` (the `OUT` and `CSV` paths use it) but drop the `sys` import and
the insert, and drop the `# noqa: E402` markers from the two `tests.` imports,
which are now ordinary top-level imports.

- [ ] **Step 6: Delete the old files**

```bash
git rm spatial_index_utils.py scripts/preprocess.py scripts/compute_psi.py \
       scripts/common.py conftest.py
```

- [ ] **Step 7: Update `README.md`**

Replace the "Repository layout" bullets and the "Running the pipeline" block:

```markdown
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

## Running the pipeline

```bash
uv run delhi-psi preprocess --config code-2025 --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_verify
uv run delhi-psi compute    --config code-2025 --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_verify
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3_verify
```

`--config` takes a shipped profile name or a path to a YAML file. Every
methodology choice — adjacency rule, barrier rule, decay, roads formula,
denominator, second normalization, exclusion semantics — is a config value;
see `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` § 3.
```

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest -q -W error` — Expected: **176 passed** (175 + 1).

`tests/test_ci_workflow.py` must still pass unchanged. Re-read what it asserts
before touching anything: it requires `uv python install 3.13`, no
`.python-version` file, `uv sync --locked`, `-W error` on every
`uv run pytest` line, exactly one drift step matching
`scripts/generate_*_fixtures.py`, and that the workflow text contains none of
`delhi_data`, `DELHI_DATA_DIR`, `verify_against_baseline`. Nothing in this
task changes `.github/workflows/ci.yml`, so all 12 stay green — but
`uv sync --locked` now depends on the `uv.lock` regenerated in Task 2, which
is why that lock had to be committed in the same commit as `pyproject.toml`.

- [ ] **Step 9: Prove the snapshot did not move**

```bash
uv run python scripts/generate_production_fixtures.py
git diff --exit-code -- tests/fixtures/ && echo NO-DRIFT
```
Expected: `NO-DRIFT`. This is the check that the `run_production_chain`
rewiring was numerically exact. If it drifts, **stop and report** — do not
regenerate the fixture.

- [ ] **Step 10: Verify the maps still render**

```bash
uv run python scripts/render_oracle_maps.py
git diff --stat -- docs/oracle/
```
Expected: the script completes; the PNGs may differ in bytes (matplotlib is
not byte-deterministic across runs) — `git checkout -- docs/oracle/` to
discard any such churn, since nothing about the maps' content changed.

- [ ] **Step 11: Commit**

```bash
git add -u spatial_index_utils.py scripts/preprocess.py scripts/compute_psi.py \
           scripts/common.py conftest.py
git add tests/oraculum_fixtures.py tests/test_divergence_exhibit.py \
        tests/test_reference_impl.py tests/test_production_fixtures.py \
        scripts/render_oracle_maps.py README.md
git commit -m "refactor: delete spatial_index_utils, the driver scripts and every sys.path hack (DEL-16)"
```

---

### Task 11: swap the generator backend and prove a no-op diff (spec § 5 step 5)

The refactor's correctness proof. The generator now produces the fixture
through `pipeline.compute_frames`, and the committed CSV — written in Task 1
from the pre-refactor code — must not move by a single byte.

**Files:**
- Modify: `scripts/generate_production_fixtures.py` (backend → `compute_frames`; generalised to any shipped profile)
- Modify: `tests/test_oracle.py` (delete `SCENARIO_WIRING` and `_production_frame`)
- Modify: `tests/oraculum_fixtures.py` (delete `run_production_chain`)
- Modify: `tests/test_fixture_invariants.py` (its two `run_production_chain` calls → `compute_oracle_frame`)

**Interfaces:**
- Consumes: `tests.oraculum_fixtures.{ORACLE_SCENARIOS, compute_oracle_frame}`, `delhi_psi.config.load_config`.
- Produces: `scripts.generate_production_fixtures.emit_profile(profile, out_path)` now accepts **any shipped profile**; `PROFILES` in `tests/test_production_fixtures.py` grows in Task 12.

- [ ] **Step 1: Swap the backend**

Replace the import block and `emit_profile` in
`scripts/generate_production_fixtures.py`:

```python
from delhi_psi.config import load_config
from tests.oraculum_fixtures import ORACLE_SCENARIOS, compute_oracle_frame
```

```python
def emit_profile(profile, out_path):
    """Write `profile`'s production fixture to out_path; return out_path."""
    methodology = load_config(profile).methodology
    columns = metric_columns(
        second_normalization=methodology.second_normalization)
    records = []
    for scenario, types, stage in ORACLE_SCENARIOS:
        for denom in DENOMS:
            frame = compute_oracle_frame(profile, types=types, stage=stage,
                                         denom=denom)
            records.extend(frame_records(profile, frame, scenario, denom,
                                         columns))
    write_fixture(out_path, records)
    return out_path


def main():
    for profile in PROFILES:
        out_path = emit_profile(profile, PRODUCTION_DIR / f"{profile}.csv")
        print(f"wrote {out_path}")
```

and add, next to the other module constants:

```python
# Every profile with a committed production fixture. Adding a profile is one
# YAML plus one entry here, then a regeneration commit (spec § 4).
PROFILES = ("code-2025",)
```

Update the module docstring: replace the "STEP-0 BACKEND" paragraph with

```
The numbers come from delhi_psi.pipeline.compute_frames, driven by the
profile's own methodology plus the § 7 scenario overrides. Migration step 0
generated the code-2025 fixture from the pre-refactor wiring; that committed
file is the refactor's correctness proof, so this generator must reproduce it
byte for byte.
```

- [ ] **Step 2: Run the generator and prove the no-op diff**

```bash
uv run python scripts/generate_production_fixtures.py
git diff --exit-code -- tests/fixtures/oraculum/production/code-2025.csv \
  && echo "NO-OP DIFF — the refactor reproduces the pre-refactor numbers"
```

Expected: `NO-OP DIFF — …`.

**If this diff is non-empty, STOP and report** (spec § 10). Do not regenerate,
do not adjust a tolerance. Diff the offending rows to find the first metric
that moved:

```bash
git diff -- tests/fixtures/oraculum/production/code-2025.csv | head -40
```

- [ ] **Step 3: Retire the legacy backend**

Delete from `tests/test_oracle.py` the whole trailing section:
`SCENARIO_WIRING`, `_production_frame`, and the `run_production_chain` name
from its import list.

Delete `run_production_chain` from `tests/oraculum_fixtures.py`, and the
`from delhi_psi import geometry, index, neighbors` import if nothing else in
that module uses it.

In `tests/test_fixture_invariants.py`, replace the two call sites:

```python
def test_empirical_pin_directed_neighbor_lists():
    result = compute_oracle_frame("code-2025", types=(), stage="post_neighbors",
                                  denom="pop")
    got = {sid: set(row["nbrs_bbox"]) for sid, row in result.iterrows()}
    assert got == CODE_DIRECTED


def test_empirical_pin_neighbor_distances():
    result = compute_oracle_frame("code-2025", types=(), stage="post_neighbors",
                                  denom="pop")
    dist = dict(result.loc["B", "nbrs_dist_bbox"])
    assert dist["E"] == pytest.approx(1.0, abs=1e-9)
    assert dist["RV"] == pytest.approx(1.0, abs=1e-9)
```

and change its import line to

```python
from tests.oraculum_fixtures import (
    compute_oracle_frame, load_barriers, load_services, load_settlements,
)
```

(`compute_oracle_frame` returns a frame indexed by settlement id, which is
why the two bodies index by `sid` / `.loc["B"]` rather than filtering.)

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q -W error` — Expected: **176 passed** (unchanged count;
`test_no_sys_path_hacks_and_no_monolith` from Task 10 still passes and no test
was added or removed).

`test_empirical_pin_*` is the hard red line — if it fails, **stop and report**.

- [ ] **Step 5: Re-prove the drift guard end to end**

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
test -z "$(git status --porcelain -- tests/fixtures/)" && echo DRIFT-OK
```
Expected: `DRIFT-OK`.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_production_fixtures.py tests/test_oracle.py \
        tests/oraculum_fixtures.py tests/test_fixture_invariants.py
git commit -m "test: production fixtures now generated by compute_frames — no-op diff vs the step-0 snapshot (DEL-15)"
```

---

### Task 12: the `manuscript` profile — fixture, reference match, hand anchors

**Files:**
- Create: `tests/fixtures/oraculum/production/manuscript.csv` (generator output, committed)
- Modify: `scripts/generate_production_fixtures.py` (`PROFILES` gains `"manuscript"`)
- Modify: `tests/test_production_fixtures.py` (`PROFILES` gains `"manuscript"`)
- Modify: `tests/test_profiles_match_reference.py` (`PROFILE_RULES` gains `manuscript → ideal`)
- Create: `tests/test_manuscript_anchors.py`

`delhi_psi/profiles/manuscript.yaml` already exists — it shipped in Task 2, so
`test_both_profiles_ship` has been green since then. This task is where it
gets its numbers.

**Interfaces:**
- Consumes: `tests.oraculum_fixtures.compute_oracle_frame`, `scripts.generate_production_fixtures.{PROFILES, emit_profile, metric_columns}`.
- Produces: no new API — a second committed fixture and three test modules' worth of coverage.

- [ ] **Step 1: Write the failing tests**

In `scripts/generate_production_fixtures.py` and
`tests/test_production_fixtures.py`, change both lists to:

```python
PROFILES = ("code-2025", "manuscript")     # generator (a tuple)
```
```python
PROFILES = ["code-2025", "manuscript"]     # test module (a list)
```

In `tests/test_profiles_match_reference.py`:

```python
PROFILE_RULES = {"code-2025": "code", "manuscript": "ideal"}
```

Create `tests/test_manuscript_anchors.py`:

```python
"""The manuscript profile against Bob's hand-ratified anchors.

Every number below is quoted from docs/oracle/derivation-worksheet.md
(RATIFIED 2026-08-24). No generator can rewrite these — they are the reason
the reference implementation has authority at all.

The worksheet prints irrational PCENs to 8 decimals, so each of those is
asserted twice: against the printed value at 5e-9 (half a unit in the last
printed place) and against the worksheet's own closed-form arithmetic at
1e-12. Terminating decimals are asserted at 1e-12 directly.

The manuscript profile maps to reference rule-set `ideal`; scenario
`baseline` is `exclusion.types: []` with the profile's own
`absent_neighbor: contributes` (spec §§ 4, 7).
"""
import math

import pytest

from tests.oraculum_fixtures import compute_oracle_frame

PROFILE = "manuscript"

# Worksheet, "Map: Decays": 1 km -> 1/2; 1.5 km -> 0.4; sqrt(2) km -> sqrt(2)-1
DECAY_1KM = 0.5
DECAY_1_5KM = 0.4
DECAY_SQRT2 = 1 / (1 + math.sqrt(2))


@pytest.fixture(scope="module")
def baseline_pop():
    return compute_oracle_frame(PROFILE, types=(), stage="post_neighbors",
                                denom="pop")


@pytest.fixture(scope="module")
def baseline_popdensity():
    return compute_oracle_frame(PROFILE, types=(), stage="post_neighbors",
                                denom="popdensity")


def test_the_decay_constants_are_the_worksheets():
    assert DECAY_SQRT2 == pytest.approx(math.sqrt(2) - 1, abs=1e-15)
    assert DECAY_SQRT2 == pytest.approx(0.414214, abs=5e-7)
    assert 1 / (1 + 1.5) == DECAY_1_5KM


# --- "## Clinics (counts: A 2, B 1, E 1, RV 2) — Eq. 3, popsize" -------
CLINIC_EXACT = {"B": 3.5 / 200, "RV": 2.5 / 100, "D": 0.4 / 100,
                "IND": 0.4 / 10}
CLINIC_PRINTED = {"A": 0.02914214, "C": 0.00228553, "E": 0.00776142}
CLINIC_CLOSED_FORM = {
    "A": (2 + 1 * DECAY_1KM + 1 * DECAY_SQRT2) / 100,
    "C": (0 + 1 * DECAY_1KM + 1 * DECAY_SQRT2) / 400,
    "E": (1 + 2 * DECAY_SQRT2 + 1 * DECAY_1KM) / 300,
}


@pytest.mark.parametrize("sid,value", sorted(CLINIC_EXACT.items()))
def test_clinic_pcen_exact_anchors(baseline_pop, sid, value):
    assert baseline_pop.loc[sid, "clinic_pcen"] == pytest.approx(value,
                                                                 abs=1e-12)


@pytest.mark.parametrize("sid", sorted(CLINIC_PRINTED))
def test_clinic_pcen_irrational_anchors(baseline_pop, sid):
    closed = CLINIC_CLOSED_FORM[sid]
    assert closed == pytest.approx(CLINIC_PRINTED[sid], abs=5e-9), \
        "the worksheet's printed value disagrees with its own arithmetic"
    assert baseline_pop.loc[sid, "clinic_pcen"] == pytest.approx(closed,
                                                                 abs=1e-12)


def test_clinic_minmax_anchors_are_c_and_ind_and_unique(baseline_pop):
    """Worksheet: 'Eq. 2 anchors: min = C (0.00228553), max = IND (0.04) —
    both unique.'"""
    pcen = baseline_pop["clinic_pcen"]
    assert pcen.idxmin() == "C" and pcen.idxmax() == "IND"
    assert (pcen == pcen.min()).sum() == 1
    assert (pcen == pcen.max()).sum() == 1


def test_clinic_index_for_a(baseline_pop):
    """Worksheet: 'A_idx = 0.02685661/0.03771447 = 0.71210346
    (CSV: 0.7121034578830464 ... check to ~6 decimals)'."""
    got = baseline_pop.loc["A", "clinic_idx"]
    assert got == pytest.approx(0.71210346, abs=5e-7)
    assert got == pytest.approx(0.7121034578830464, abs=1e-12)


# --- "## Schools (A 1, D 1, E 1) — Eq. 3, popsize" ---------------------
SCHOOL_EXACT = {"B": 1.0 / 200, "RV": 0.0, "D": 1.4 / 100, "IND": 0.4 / 10}
SCHOOL_PRINTED = {"A": 0.01414214, "C": 0.00103553, "E": 0.00604738}
SCHOOL_CLOSED_FORM = {
    "A": (1 + 1 * DECAY_SQRT2) / 100,
    "C": (0 + 1 * DECAY_SQRT2) / 400,
    "E": (1 + 1 * DECAY_SQRT2 + 1 * DECAY_1_5KM) / 300,
}


@pytest.mark.parametrize("sid,value", sorted(SCHOOL_EXACT.items()))
def test_school_pcen_exact_anchors(baseline_pop, sid, value):
    assert baseline_pop.loc[sid, "school_pcen"] == pytest.approx(value,
                                                                 abs=1e-12)


@pytest.mark.parametrize("sid", sorted(SCHOOL_PRINTED))
def test_school_pcen_irrational_anchors(baseline_pop, sid):
    closed = SCHOOL_CLOSED_FORM[sid]
    assert closed == pytest.approx(SCHOOL_PRINTED[sid], abs=5e-9)
    assert baseline_pop.loc[sid, "school_pcen"] == pytest.approx(closed,
                                                                 abs=1e-12)


def test_school_near_tie_between_a_and_d_survives(baseline_pop):
    """Worksheet: 'the deliberate near-tie A vs D (0.014142 vs 0.014)'."""
    assert baseline_pop.loc["A", "school_pcen"] > \
        baseline_pop.loc["D", "school_pcen"]
    assert baseline_pop.loc["A", "school_pcen"] - \
        baseline_pop.loc["D", "school_pcen"] == pytest.approx(0.000142136,
                                                              abs=5e-9)


# --- "## Roads — Eq. 4 literally (NO neighbor term)" -------------------
def test_road_lengths_are_075_km_for_a_and_e(baseline_pop):
    assert baseline_pop.loc["A", "road_length"] == pytest.approx(0.75,
                                                                 abs=1e-12)
    assert baseline_pop.loc["E", "road_length"] == pytest.approx(0.75,
                                                                 abs=1e-12)


def test_road_pcen_pop_is_eq4_with_a_tied_zero_minimum(baseline_pop):
    """'pop: A = 0.75/100 = 0.0075; E = 0.75/300 = 0.0025;
    B = C = RV = D = IND = 0 exactly (tied minimum)'."""
    assert baseline_pop.loc["A", "road_pcen"] == pytest.approx(0.0075,
                                                               abs=1e-12)
    assert baseline_pop.loc["E", "road_pcen"] == pytest.approx(0.0025,
                                                               abs=1e-12)
    for sid in ("B", "C", "RV", "D", "IND"):
        assert baseline_pop.loc[sid, "road_pcen"] == 0.0, sid


def test_road_pcen_popdensity(baseline_popdensity):
    """'popdensity: A = 0.0075; E = 0.75/150 = 0.005'."""
    assert baseline_popdensity.loc["A", "road_pcen"] == pytest.approx(
        0.0075, abs=1e-12)
    assert baseline_popdensity.loc["E", "road_pcen"] == pytest.approx(
        0.005, abs=1e-12)


# --- "## Singleton services (bank@A, police@B, ration@D, transport@E)" -
def test_police_singleton_table(baseline_pop):
    """'B = 1/200 = 0.005; A = 1*1/2/100 = 0.005 (tied argmax);
    C = 0.00125; RV = 0.005 (three-way tie A/B/RV); E = 0.00166667;
    D = 0; IND = 0.'"""
    police = baseline_pop["police_pcen"]
    assert police["B"] == pytest.approx(0.005, abs=1e-12)
    assert police["A"] == pytest.approx(0.005, abs=1e-12)
    assert police["RV"] == pytest.approx(0.005, abs=1e-12)
    assert police["C"] == pytest.approx(0.00125, abs=1e-12)
    assert police["E"] == pytest.approx(0.00166667, abs=5e-9)
    assert police["E"] == pytest.approx(1 * DECAY_1KM / 300, abs=1e-12)
    assert police["D"] == 0.0 and police["IND"] == 0.0


# --- "## Worked extras (complete the anchor subset)" -------------------
def test_extra_1_exclusion_delta_for_b():
    """'B, ideal, excl_removed, pop: (1 + 2*1/2 + 0 + 1*1/2)/200
    = 2.5/200 = 0.0125 (vs 0.0175 baseline - the RV contribution effect,
    -0.005)'."""
    removed = compute_oracle_frame(PROFILE, types=("RV", "IND"),
                                   stage="pre_neighbors", denom="pop")
    assert removed.loc["B", "clinic_pcen"] == pytest.approx(0.0125, abs=1e-12)
    baseline = compute_oracle_frame(PROFILE, types=(),
                                    stage="post_neighbors", denom="pop")
    assert baseline.loc["B", "clinic_pcen"] - removed.loc["B", "clinic_pcen"] \
        == pytest.approx(0.005, abs=1e-12)


def test_extra_2_renormalization_delta_for_a():
    """'A clinic_idx, ideal, excl_ind_removed, pop = 1.0 exactly
    (was 0.71210346) - anchor movement with zero numerator change. This
    delta is denominator-INVARIANT because A, C, IND all have area
    1.0 km^2.'"""
    for denom in ("pop", "popdensity"):
        frame = compute_oracle_frame(PROFILE, types=("IND",),
                                     stage="pre_neighbors", denom=denom)
        assert frame.loc["A", "clinic_idx"] == pytest.approx(1.0, abs=1e-12), \
            denom


def test_extra_3_popdensity_coverage_for_e(baseline_pop, baseline_popdensity):
    """'E clinic, ideal, baseline: popsize 2.328427/300 = 0.00776142;
    popdensity divides by pop/area = 300/2 = 150 -> 2.328427/150
    = 0.01552285'."""
    assert baseline_pop.loc["E", "clinic_pcen"] == pytest.approx(0.00776142,
                                                                 abs=5e-9)
    assert baseline_popdensity.loc["E", "clinic_pcen"] == pytest.approx(
        0.01552285, abs=5e-9)
    assert baseline_popdensity.loc["E", "clinic_pcen"] == pytest.approx(
        (1 + 2 * DECAY_SQRT2 + 1 * DECAY_1KM) / (300 / 2), abs=1e-12)


def test_extra_4_road_eq4_value_for_a(baseline_pop):
    """'Road Eq. 4 value (A, pop): 0.75/100 = 0.0075'."""
    assert baseline_pop.loc["A", "road_pcen"] == pytest.approx(0.0075,
                                                               abs=1e-12)


def test_manuscript_profile_has_no_second_normalization(baseline_pop):
    """second_normalization: false, so the column is absent (spec § 4)."""
    assert "unnorm_psi" in baseline_pop.columns
    assert "norm_psi" not in baseline_pop.columns
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_manuscript_anchors.py tests/test_production_fixtures.py -q`
Expected: the anchor tests fail on the missing fixture only if `manuscript.yaml`
were absent (it is not) — so they should mostly **pass immediately**, while
`tests/test_production_fixtures.py::test_fixture_is_regenerable[manuscript]`
FAILS with `assert False` on `missing committed fixture …/manuscript.csv`.

That is the correct RED for this task: the anchors are a check on the profile
that already exists; the fixture is what is missing. If an *anchor* test fails,
the `manuscript` profile does not reproduce the manuscript — **stop and
report**.

- [ ] **Step 3: Generate and commit the manuscript fixture**

```bash
uv run python scripts/generate_production_fixtures.py
wc -l tests/fixtures/oraculum/production/manuscript.csv
git diff --exit-code -- tests/fixtures/oraculum/production/code-2025.csv \
  && echo "code-2025 UNCHANGED"
```

Expected: `manuscript.csv` has **1393** lines (1,392 records + header — 24
metrics, because `second_normalization` is false, × 29 settlement-rows × 2
denominators), and `code-2025 UNCHANGED`.

- [ ] **Step 4: Run the tests to verify they pass, then the whole suite**

Run: `uv run pytest tests/test_manuscript_anchors.py tests/test_production_fixtures.py tests/test_profiles_match_reference.py -q`
Expected: 56 passed (anchors 27, production fixtures 6, profiles 23).

Run: `uv run pytest -q -W error` — Expected: **215 passed** (176 + 27 anchors + 2 fixture params + 10 manuscript reference params).

- [ ] **Step 5: Commit**

```bash
git add delhi_psi/profiles/manuscript.yaml \
        tests/fixtures/oraculum/production/manuscript.csv \
        scripts/generate_production_fixtures.py \
        tests/test_production_fixtures.py \
        tests/test_profiles_match_reference.py \
        tests/test_manuscript_anchors.py
git commit -m "feat: manuscript profile with its production fixture and hand anchors (DEL-22)"
```

---

### Task 13: real-data proof, changelog, WORKPLAN

**Files:**
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `WORKPLAN.md` (ticks + the spec § 9 item 7 housekeeping)

**Interfaces:** none.

- [ ] **Step 1: Run the real-data proof by hand**

This is a hand-run step, **not a CI test** — CI never touches `~/delhi_data`
(`tests/test_ci_workflow.py::test_no_data_dependency` enforces that). The data
directory is read-only; everything lands under `--out-dir`.

```bash
uv run delhi-psi preprocess --config code-2025 \
    --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_verify
uv run delhi-psi compute --config code-2025 \
    --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_verify
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3_verify
```

Expected, verbatim:

```
PASS — new run equivalent to July 2025 baseline within tolerance
```

with every reported `max abs deviation` line reading `0.000e+00`. Paste the
full output into the PR body, as for DEL-26/23.

**Any non-zero deviation is a stop condition** (spec § 10) — do not tune a
tolerance, do not exclude a column. Note in particular that
`compare_numeric_frames` treats a baseline column missing from the new run as
a deviation, so a "columns missing from new run" line means the § 5
output-column contract was broken (most likely a renamed column).

- [ ] **Step 2: Add the changelog entry**

Under `## [Unreleased]` in `CHANGELOG.md`, add:

```markdown
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
  deviation from the July 2025 baseline. The silent `except: pass` in
  `calc_pcen_mobile` is now an explicit lookup miss, which is what makes
  `absent_neighbor: contributes` implementable (DEL-21). Root `conftest.py`
  and every `sys.path.insert` are gone (`tests/__init__.py` plus the editable
  install do that job); `pyyaml` moved to the runtime dependencies and
  `uv.lock` was regenerated. Notebook eyeball checks became raising
  assertions in `delhi_psi.validate` (DEL-25). Tests 77 → 215, including
  `test_config`, `test_profiles_match_reference` (both profiles × every
  scenario × both reference denominators), `test_manuscript_anchors` (the
  hand-ratified worksheet values), `test_production_fixtures`, `test_cli`,
  `test_validate`. Spec:
  `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md`.
  [DEL-15, DEL-16, DEL-18 (partial), DEL-21, DEL-22, DEL-25]
```

- [ ] **Step 3: Tick the WORKPLAN items**

In `WORKPLAN.md`, Phase 3:

1. DEL-15 → done. Change `- [ ] Brainstorm → owner-approved spec →` to
   `- [x] Brainstorm → owner-approved spec →` and append to that bullet:
   ```
         — done 27 Aug 2026: spec
         `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md`, plan
         `docs/superpowers/plans/2026-08-27-phase3a-refactor.md`; Phase 3
         split into cycles 3A (this one), 3B (DEL-17) and 3C (DEL-24/19/20)
   ```
2. DEL-16 → done. Change `- [ ] One canonical implementation:` to `- [x]` and
   append:
   ```
         — done 27 Aug 2026 (3A): `create_service_index` /
         `create_service_length_index` collapsed into `index.service_index`
         fed by `point_counts`/`road_lengths`; the `road_count → road_length`
         special case is gone (line services name their amount column
         `<service>_length`)
   ```
3. DEL-18 → partial. Change `- [ ] Modular & extensible structure:` to
   `- [~]` and append:
   ```
         — PARTIAL (3A, 27 Aug 2026): adjacency/barrier rules, service sets,
         denominators, exclusion and units are config. Distance thresholds
         (DEL-36) and alternative decay weights (DEL-37) are NOT — the schema
         leaves room (`adjacency` and `decay` are mappings, not bare strings)
         and Phase 6 adds the values with their reference rules
   ```
4. DEL-21 → done. In the bug-audit list, strike item 3 the way DEL-23 was
   struck:
   ```
         3. ~~**Silent `except: pass` in `calc_pcen_mobile`** — swallows
            missing neighbors, making exclusion semantics (a) unimplementable
            (WORKPLAN Open Decision A is half-answered by this).~~ [DEL-21] —
            done 27 Aug 2026 (3A): replaced by an explicit lookup;
            `exclusion.absent_neighbor: contributes` reads amounts from the
            pre-exclusion frame, so semantics (a) is now implementable.
   ```
5. DEL-22 → done. Strike item 4 similarly:
   ```
         4. ~~Barrier rule is global + asymmetric vs. the manuscript's pair
            severing; roads carry neighbor decay Eq. 4 does not have;
            `norm_psi` is a second normalization absent from Eq. 1;
            popdensity has no manuscript equation.~~ [DEL-22] — done 27 Aug
            2026 (3A): all four are config switches with both values
            implemented and reference-pinned (`barrier.rule`, `roads`,
            `second_normalization`, `outputs.denominators`), and the
            `manuscript` profile runs the paper's rule-set end to end. The
            fix-or-ratify CALL is still Raj's (DEL-13); whichever he picks is
            a profile edit, not a code change.
   ```
6. DEL-25 → done. Change `- [~] Retire the notebooks entirely` to `- [x]` and
   append:
   ```
         — done 27 Aug 2026 (3A): `delhi-psi {preprocess,compute}` are the
         pipeline stages, `delhi_psi.validate` turns the eyeball checks into
         raising assertions, `logging` replaced `print`. The figures command
         is Phase 4 (DEL-33).
   ```

- [ ] **Step 4: Do the spec § 9 item 7 WORKPLAN housekeeping**

Three separate defects, all pre-existing:

1. **The DEL-27 note is on the wrong heading.** In the Phase 4 heading
   paragraph, delete the appended sentence
   `— **done 25 Aug 2026, PR #7**: .github/workflows/ci.yml (locked sync, 77
   tests, fixture-drift guard); spec
   docs/superpowers/specs/2026-08-24-ci-workflow-design.md. Owner follow-up:
   make test a required check in branch protection.`
   so the Phase 4 heading ends at `Epic DEL-5.*`.
2. **The DEL-27 bullet still says the workflow does not exist.** Rewrite the
   Phase 3 bullet as:
   ```
   - [x] Add GitHub Actions CI running `uv run pytest` on every push/PR
         (decided in meta-planning "once the suite exists") [DEL-27] — done
         25 Aug 2026, PR #7: `.github/workflows/ci.yml` (locked sync, the
         oracle suite under `-W error`, fixture-drift guard); spec
         `docs/superpowers/specs/2026-08-24-ci-workflow-design.md`. Owner
         follow-up: make `test` a required check in branch protection.
   ```
3. **The owner list cites the wrong decision letter for roads.** In "Open
   items by owner → **Raj:**", change
   `the two blocking ones are exclusion semantics (Open Decision A) and roads
   Eq. 4. Open Decision B; Phase 4 categorization.`
   to
   `the two blocking ones are exclusion semantics (Open Decision A) and roads
   Eq. 4 (Open Decision C — B is the data-release posture); Phase 4
   categorization.`

Also refresh the two places that describe the repo as three scripts:

- The "Repo state" paragraph: replace
  `The pipeline is three plain scripts on uv + pyproject.toml
  (scripts/preprocess.py → scripts/compute_psi.py →
  scripts/verify_against_baseline.py) around spatial_index_utils.py;`
  with
  `The pipeline is the installable delhi_psi package on uv + pyproject.toml
  (delhi-psi preprocess → delhi-psi compute, plus
  scripts/verify_against_baseline.py), config-driven via
  delhi_psi/profiles/*.yaml;`
  and change `(65 tests green)` to `(215 tests green)`.
- The "Status at a glance" table: change the Phase 3 row to
  `| 3 Refactor & bug audit | **in progress** — cycle 3A done (PR pending) | delhi_psi package; code-2025 reproduces the step-0 snapshot byte-for-byte and the July 2025 baseline at zero deviation |`

- [ ] **Step 5: Run the whole suite one last time**

Run: `uv run pytest -q -W error` — Expected: **215 passed**.

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
test -z "$(git status --porcelain -- tests/fixtures/)" && echo DRIFT-OK
```
Expected: `DRIFT-OK`.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md WORKPLAN.md
git commit -m "docs: changelog and WORKPLAN for Phase 3A (DEL-15, DEL-16, DEL-18, DEL-21, DEL-22, DEL-25)"
```

- [ ] **Step 7: Sync the Jira board**

Per the standing memory note ("Jira Delhi board — DEL project mirrors
WORKPLAN"): transition DEL-15, DEL-16, DEL-21, DEL-22 and DEL-25 to Done with
the PR link in a comment; leave DEL-18 in progress with a comment recording
that adjacency/barrier/services/denominators/exclusion/units landed in 3A and
that DEL-36/DEL-37 carry the rest. Record on DEL-17 that 3B is next and on
DEL-19/20/24 that they are 3C, gated on Raj (DEL-13).

---

## Spec ambiguities resolved (for the owner)

These are decisions the plan had to make where the spec was silent or where
two of its statements had to be sequenced. None changes a number; all are
recorded here for adjudication.

1. **`compute_frames`' signature has six positional parameters in spec § 2,
   but it needs the CRS, the population join keys, the id/type column names
   and the `max_missing_population` limit** — none of which live in
   `MethodologyConfig`. Resolved by adding them **keyword-only, each
   defaulting to its `code-2025` value**, so the spec's documented positional
   call is unchanged and a caller that passes nothing extra gets today's
   behaviour.
2. **Point vs line services.** Spec § 2 types `services` as
   `dict[str, GeoDataFrame]`, which carries no point/line marker, yet the
   roads formula and the amount-column name both depend on it. Resolved by
   classifying each layer from its own geometry types
   (`pipeline.service_kind`), raising `ValueError` on an empty or mixed layer.
   No extra parameter, and the partition matches the config's
   `services.point` / `services.line` split on both the oracle and the real
   data.
3. **`run_production_chain`'s lifetime.** Spec § 2 says `compute_frames`
   replaces it; spec § 5 says the generator's backend swaps at step 5. Since
   the step-0 snapshot must keep being produced by its original wiring until
   the swap is proven a no-op, the deletion task (step 4) only re-points
   `run_production_chain` at `delhi_psi` and the swap task (step 5) deletes
   it. `test_production_fixtures` catches any drift the re-pointing might
   introduce, on the same commit.
4. **`exclusion.stage: pre_neighbors` on a universe-wide artifact.** Spec § 3
   says excluded ids are "removed from every neighbor list before PCEN", not
   that adjacency is re-run. The plan implements the strip, and argues (in
   `apply_exclusion`'s docstring) that it is identical to re-running adjacency
   because both adjacency rules and both barrier rules are pairwise. Today's
   `_production_frame(drop_ids_pre=…)` *does* re-run adjacency, so the step-0
   snapshot is the empirical check on that argument — if the equivalence is
   wrong, Task 11's diff is non-empty and the run stops.
5. **The step-0 generator's `sys.path` bootstrap.** The package is not
   installed at step 0, so the generator cannot import `tests.test_oracle`
   when the CI drift guard runs it by path. The plan adds one bootstrap in
   Task 1 and deletes it in Task 2 (the editable install puts the repo root on
   `sys.path`), so the "no `sys.path.insert` anywhere" invariant holds from
   Task 2 onward rather than only from the deletion task.
6. **Anchor tolerance.** Spec § 4 asks for the worksheet anchors at 1e-12, but
   the worksheet prints irrational PCENs to 8 decimals, which cannot be met at
   1e-12. Resolved by asserting each irrational anchor twice: against the
   printed value at 5e-9 (half a unit in the last printed place — the
   worksheet itself says "check to ~6 decimals" for the derived index) and
   against the worksheet's own closed-form arithmetic at 1e-12. Terminating
   decimals are asserted at 1e-12 directly.
7. **`[project.scripts]` before `cli.py` exists.** Spec § 2 puts every
   `pyproject.toml` change in migration step 1, but the CLI arrives in step 3.
   Console-script entry points resolve lazily, so `uv sync` succeeds; the plan
   declares the script in Task 2 and notes that `delhi-psi` must not be run
   until Task 8.
8. **`validate.geometries_are` no longer mutates its argument.** The original
   `check_geometries` assigned a vestigial `geom_type` column and then read
   `gdf.geom_type`, which resolves to the GeoDataFrame *property*, not the
   column — so the assignment never affected the result, and
   `scripts/preprocess.py` had to drop the column afterwards. The plan removes
   the assignment (and the drop). Value-identical; the column is non-numeric
   so it is outside `compare_numeric_frames`' scope.
9. **`validate.within_bounds` uses `.iloc[0]`** where `gdf_within_delhi` used
   `delhi_contains_gdf[0]`. Identical on the RangeIndex every `read_file`
   produces, and unambiguous now that the battery runs **in-process** under
   `-W error` (the old e2e test ran it in a subprocess, where warnings were
   invisible).
10. **Dedup cache location and staleness.** Spec § 6 says the cache location
    and staleness rule move; it does not say where. The plan puts the cache
    under `out_dir` (`<name>.dedup` plus a `<name>.dedup.stamp` holding
    `mtime_ns:size` of the source), which keeps `~/delhi_data` read-only —
    `scripts/preprocess.py` wrote its `*.data` caches into the data directory.
11. **`missing_colonies_aug2026.csv` → `missing_population.csv`.** Spec § 6
    names the artifact `missing_population.csv` and § 2 says outputs land
    directly in `out_dir`. The plan uses that name and location. It is not a
    baseline-compared file, so the rename cannot affect the real-data proof.

## Self-review

**Spec coverage.** § 1 decomposition → the plan is 3A only, and the WORKPLAN
ticks in Task 13 record DEL-18 as partial. § 2 package layout → Tasks 2–9
create every listed module; the deletion list is Task 10; the `pyproject`
block is Task 2 verbatim. § 3 config schema → Task 2 (`config.py`, both
profiles, `test_config.py`), including the two reserved values, the reserved
key with the A.2 message, required `profile` + full `methodology`, and every
other block defaulting to `code-2025`; the two exclusion axes are Tasks 5
(`pcen`) and 7 (`apply_exclusion`); the missing-population rule follows the
same two keys (Task 7 `compute_frames`, Task 8 `compute`). § 4 fixtures →
Tasks 1, 11, 12; the reference CSV is untouched throughout; both pins per
profile are covered (string-equal in `test_production_fixtures`, 1e-12 in
`test_profiles_match_reference`, worksheet anchors in
`test_manuscript_anchors`). § 5 migration order → Tasks 1, 2, 3–5, 7–8, 10,
11 in exactly the spec's 0–5 order; the output-column contract is enforced by
the real-data proof and named in "Column names are frozen"; all three proofs
appear (drift guard every task, e2e in Task 8, real data in Task 13). § 6
stages/validation/CLI → Tasks 6 and 8, including the two-warning filter scoped
to the shapefile write and the in-process `test_cli` shp case. § 7 testing →
every listed new module plus the re-expressed `test_oracle.py` and the
rewired `test_common.py`. § 8 is Raj's meeting, not code — the profiles are
what he edits. § 9 item 7 housekeeping → Task 13 Step 4; items 1–6 are
out-of-scope follow-ups and are not planned here. § 10 stop conditions → the
Global Constraints and the explicit "stop and report" lines in Tasks 5, 7, 11,
12 and 13.

**Placeholder scan.** The only ellipses in the plan are in Task 9 Step 3,
where two function bodies are deliberately **not** retyped — the step gives
the exact `sed` range to copy them from, which is stricter than reproducing
them. Every other code step contains runnable code. No "TBD", no "similar to
Task N", no "add appropriate error handling".

**Type consistency.** Names used by later tasks and defined earlier:
`geometry.{row_index, reproject, remove_duplicate_geom, bbox_frame,
barrier_flags, _flag_one, distance_to_point_km}` (Task 3) are called in Tasks
4, 7, 8, 10; `neighbors.{combine_barrier_flags, adjacency, apply_barrier,
centroid_distances}` (Task 4) in Tasks 7, 10; `index.{point_counts,
road_lengths, service_amount_column, pcen, minmax, service_index,
overall_psi}` (Task 5) in Tasks 7, 10; `io.{resolve_data_dir, out_dir_path,
resolve_out_dir}` (Task 2) in Tasks 6, 8, 9 and `io.{read_layer,
read_population, read_neighbors, write_neighbors, write_outputs,
SHAPEFILE_DROP_COLUMNS}` (Task 6) in Task 8; `validate.{require_layer,
check_missing_population, check_no_negative, check_crs_match, check_crs_defined,
ValidationError}` (Task 6) in Tasks 7, 8; `config.{load_config, ConfigError,
ExclusionStage, REFERENCE_KNOBS, ENUMS, ENUM_KEYS, RESERVED_VALUES,
RESERVED_KEYS, PROFILES_DIR, MethodologyConfig}` (Task 2) in Tasks 7, 8, 9,
11; `pipeline.{compute_frames, index_frames, attach_population,
excluded_ids, build_neighbors, apply_exclusion, output_basename, preprocess,
compute}` (Tasks 7–8) in Tasks 8, 9; `oraculum_fixtures.{ORACLE_SCENARIOS,
methodology_with, compute_oracle_frame}` (Task 7) in Tasks 11, 12;
`generate_production_fixtures.{PRODUCTION_DIR, PROFILES, SERVICES,
metric_columns, frame_records, write_fixture, emit_profile}` (Task 1) in
Tasks 11, 12. The neighbour columns are `nbrs_bbox` / `nbrs_dist_bbox`
everywhere; the amount columns are `<service>_count` / `road_length`
everywhere; the PSI columns are `unnorm_psi` / `norm_psi` everywhere, and the
reference's `psi_eq1` / `road_length_km` appear only inside the two metric
maps that translate between them.

**Running test count.** 77 → 80 (T1) → 104 (T2) → 112 (T3) → 121 (T4) → 140
(T5) → 154 (T6) → 167 (T7) → 174 (T8) → 175 (T9) → 176 (T10) → 176 (T11) →
215 (T12) → 215 (T13). If a task's actual count differs, the plan's test
bodies are the authority — recount, do not delete a test to hit a number.
