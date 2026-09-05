# Pre-recalculation Measurements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer DEL-49/50/51/52 with four re-runnable measurement scripts whose numbers are carried verbatim in `docs/data/` and pinned by drift tests, so the ratified profile (DEL-31), the recalculation (DEL-32) and the batched reply to Raj rest on measured facts.

**Architecture:** Four scripts under `scripts/` share one internal module (`scripts/_measure_common.py`: the work-dir guard, the settlement loader, the fenced-block renderer/parser). Each script is READ-ONLY over `--data-dir`, writes only under `--work-dir`, and prints provenance lines plus one or more ```` ```text ```` blocks that a `docs/data/*.md` file carries verbatim. Every counting function is unit-tested on the fixture cities (Oraculum, Messy) before it ever meets 4,357 real polygons; the real-data runs happen once, in the final task, by the controller.

**Tech Stack:** Python 3.13, uv, geopandas ≥ 1.1, pandas ≥ 2.3, shapely ≥ 2.1, PyYAML ≥ 6.0, pytest ≥ 8.4. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-05-pre-recalculation-measurements-design.md` — read it in full before starting; its § 8 records decisions already made and must not be reopened.

## Global Constraints

- **No `delhi_psi/` behaviour change.** All new code is `scripts/` + `tests/` + `docs/`. No new shipped profile, no fixture change, no expected-value change.
- **READ-ONLY over the data directory.** `~/delhi_data` is bisynced to the shared drive; nothing is ever written there. Scratch goes under `--work-dir`, which every script refuses to place inside the data directory.
- **CLI shape, the THREE NEW scripts:** `--config` (default `code-2025`), `--data-dir`, `--work-dir`, plus the script-specific `--verify-dir` / `--baseline-dir` / `--all-candidates`. `measure_layer_pathologies.py` keeps its historic `--cache-dir` flag name and its documented command line (spec § 3); the guard behind both names is the same `resolve_work_dir`.
- **A warm dedup cache upcasts Polygon → MultiPolygon.** `pipeline._dedup_cached` returns the in-memory frame on a COLD cache but re-reads its own GeoPackage on a WARM one, and a GeoPackage layer has a single geometry type: the raw layer's 3,801 Polygon + 556 MultiPolygon all come back as MultiPolygon after a round trip. So any count that inspects `geom_type` — today only `multipolygons` in the pathology block — is valid ONLY on a cold cache. Rule (spec § 3): `measure_layer_pathologies.py` always runs cold (its default fresh temp `--cache-dir`; the run step never points it at a staged cache), and `_measure_common.load_settlements` says so in its docstring. The other scripts' predicates (`intersects`, intersection length, `touch` adjacency, barrier flags) are type-agnostic and may share a warm cache.
- **Every script exposes** `measure(...)` (or `inventory(...)`) returning an ordered dict, a `render`, and `main(argv=None)`. A script with ONE block returns a flat ordered dict and renders it unlabeled (the pathology script's shape). A script with MORE THAN ONE block returns an ordered dict of *block name → block dict* and renders each with `name=`. "Ordered dict" means a plain `dict` (insertion-ordered since 3.7) — never `collections.OrderedDict`.
- **Tests call the functions, not `main`** — with exactly one argparse smoke test per script. The one other place a script is invoked as a whole is the data-gated doc-drift test, which runs it as a subprocess and compares its stdout with the committed block: that comparison is spec § 4's requirement ("it equals the script's output on this machine"), and it is the shape `tests/test_layer_pathologies.py` already uses.
- **Imports:** the repo root is on `sys.path` (editable install), so `from scripts._measure_common import ...` works both under pytest and when a script is run as `uv run python scripts/<name>.py`. Tests import scripts as `from scripts.<name> import ...` — the mechanism `tests/test_layer_pathologies.py` already uses.
- **Block format:** a fence line ` ```text `, an optional first line `block: <name>`, then `key: value` lines, then ` ``` `. Integers are bare; floats are pre-formatted strings (`%.6g` for means/lengths, `%.4f` for gaps) so the drift comparison is exact.
- **Every test step names the exact pytest command and the expected RED reason, then GREEN.** Run tests with `uv run pytest -q -W error`.
- **One cache per machine, not one per test: `DELHI_PSI_MEASURE_CACHE`.** A cold settlement dedup on the real layer costs ≈ 4.5 minutes (measured 5 Sep 2026: 267 s), and the `touch` adjacency behind it is a second O(n²) pass. The two real-data doc-drift tests that need the settlement universe — `tests/test_inventory_barriers.py` and `tests/test_measure_roads_access.py` — therefore take their `--work-dir` from the environment variable `DELHI_PSI_MEASURE_CACHE` and **skip unless it is set**, with the reason `"set DELHI_PSI_MEASURE_CACHE to run the real-data drift check"`. `needs_data` alone is NOT enough for those two: this machine has the data, so a plain `needs_data` gate would make every implementer's per-task full-suite run pay for two cold dedups. `tests/test_measure_psi_columns.py`'s drift test reads CSVs only (seconds, no settlement layer) and keeps the plain `needs_data` gate; `tests/test_layer_pathologies.py` keeps its own COLD `fresh` fixture (previous bullet — its `multipolygons` key demands cold). Only Task 6 exports the variable.
- **Before the final commit of EVERY task, run the full suite in the FOREGROUND:** `uv run pytest -q -W error` (about 6.5 minutes). Never background it; never commit on an unseen result. **Tasks 1–5 only** — Task 6 exports `DELHI_PSI_MEASURE_CACHE`, which wakes the two real-data drift tests and pushes the suite to 15–20 minutes; that one run is backgrounded to a log file and the log is read (Task 6 Steps 8–9), because 15–20 minutes exceeds a foreground command's timeout.
- **One commit per task**, message prefix `feat(measure):` / `test(measure):` / `docs(measure):`, ending with exactly these two trailer lines:

  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
  ```

- **Real-data commands are NOT run by task implementers.** Tasks 1–5 run only fixture-scale tests. Task 6 is the controller's run step and is the ONLY task in this plan permitted to contain bracketed placeholders — because the numbers do not exist until the commands run.
- **Vocabulary (28 Aug 2026 decisions):** reported types `Planned, UAC, RUAC, JJC, JJR, UV, SDA`; dropped types `RV, Industrial, Other`. Type column `USO_FINAL`, id column `USO_AREA_U`.

---

## File Structure

```
scripts/
  _measure_common.py            NEW  work-dir guard, settlement loader,
                                     render/parse_block, FENCE            (Task 1)
  measure_layer_pathologies.py  MOD  imports the four names back;
                                     + count_corner_only_pairs        (Tasks 1, 2)
  inventory_barriers.py         NEW  DEL-51                                (Task 3)
  measure_psi_columns.py        NEW  DEL-52                                (Task 4)
  measure_roads_access.py       NEW  DEL-49                                (Task 5)
docs/data/
  layer_pathologies.md          MOD  + corner-only definition and keys  (Tasks 2, 6)
  barriers.md                   NEW  skeleton + provenance prose       (Tasks 3, 6)
  psi_columns.md                NEW  skeleton + figure table           (Tasks 4, 6)
  roads_access.md               NEW  skeleton + reopen threshold       (Tasks 5, 6)
tests/
  test_measure_common.py        NEW  guard, parser, prose helper          (Task 1)
  test_layer_pathologies.py     MOD  imports; corner-only tests        (Tasks 1, 2)
  test_inventory_barriers.py    NEW                                       (Task 3)
  test_measure_psi_columns.py   NEW                                       (Task 4)
  test_measure_roads_access.py  NEW                                       (Task 5)
docs/decisions/2026-08-28-raj-methodology-decisions.md   MOD              (Task 6)
WORKPLAN.md, CHANGELOG.md                                MOD              (Task 6)
```

`README.md` is **not** touched: spec § 5 says add a `docs/data/` pointer "if the README lists `docs/data/`". It does not (checked 5 Sep 2026 — the only `docs/` links in README are `docs/methodology-config.md` and a spec path).

---

### Task 1: `scripts/_measure_common.py` — the shared plumbing

**Files:**
- Create: `scripts/_measure_common.py`
- Create: `tests/test_measure_common.py`
- Modify: `scripts/measure_layer_pathologies.py:20-53` (imports, `resolve_cache_dir`, `load_settlements`), `:156-201` (`render`, `parse_block`, `main`)
- Modify: `tests/test_layer_pathologies.py:19-21` (imports), `:111-123` (the cache-dir test)

**Interfaces:**
- Consumes: `delhi_psi.geometry.reproject`, `delhi_psi.io.read_layer`, `delhi_psi.pipeline._dedup_cached` (all existing).
- Produces — every later task imports from here:
  - `scripts._measure_common.FENCE: str` — the literal `` "```text" ``
  - `scripts._measure_common.resolve_work_dir(cli_value=None, *, data_dir=None, prefix="delhi_psi_measure_") -> Path`
  - `scripts._measure_common.load_settlements(cfg, cache_dir) -> GeoDataFrame`
  - `scripts._measure_common.render(report, *, name=None) -> str`
  - `scripts._measure_common.parse_block(text, *, name=None) -> dict[str, str]`
  - `tests.test_measure_common.DATA_DIR: Path`, `tests.test_measure_common.needs_data` (pytest marker), `tests.test_measure_common.MEASURE_CACHE: str | None`, `tests.test_measure_common.needs_measure_cache` (pytest marker), `tests.test_measure_common.prose_numbers(text) -> set[str]`, `tests.test_measure_common.assert_prose_numbers_come_from_the_blocks(text, blocks) -> None`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_measure_common.py`:

```python
"""The shared plumbing every measurement script imports (spec § 1).

Nothing here needs the real data: the guard, the renderer and the parser are
pure. `DATA_DIR`/`needs_data`/`MEASURE_CACHE`/`needs_measure_cache`/
`prose_numbers` live here because the four doc-drift test modules all need
them and this is the module they all already import from.
"""
import os
import re
import shutil
from pathlib import Path

import pytest

from scripts._measure_common import (FENCE, parse_block, render,
                                     resolve_work_dir)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DELHI_DATA_DIR", "~/delhi_data")).expanduser()

needs_data = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"real Delhi data not present at {DATA_DIR}")

# One settlement dedup per MACHINE, not one per test. A cold dedup of the real
# 4,357-polygon layer costs ~4.5 min (267 s, measured 5 Sep 2026) and the
# `touch` adjacency behind it is a second O(n^2) pass, so the two real-data
# drift tests that need the settlement universe (barriers, roads) share ONE
# staged work dir named by this variable and skip when it is unset. They are
# not gated on `needs_data` alone: this machine HAS the data, and a
# data-only gate would make every per-task suite run pay for two cold dedups.
# The run step (task 6) is the only place that exports it. The psi_columns
# drift test reads CSVs only and keeps the plain `needs_data` gate.
MEASURE_CACHE = os.environ.get("DELHI_PSI_MEASURE_CACHE")

needs_measure_cache = pytest.mark.skipif(
    not MEASURE_CACHE,
    reason="set DELHI_PSI_MEASURE_CACHE to run the real-data drift check")


def prose_numbers(text):
    """Backticked numeric literals in PROSE — outside every fenced block.

    The guard behind them: a number a document quotes in prose must be a
    value the block actually carries, or the prose has drifted from the
    measurement. Derived quantities (shares) are written with a `%` sign or
    without backticks, so they are deliberately not matched here.
    """
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return {match.replace(",", "")
            for match in re.findall(r"`(-?\d[\d,]*(?:\.\d+)?)`", prose)}


def assert_prose_numbers_come_from_the_blocks(text, blocks):
    values = {str(value) for block in blocks for value in block.values()}
    for number in sorted(prose_numbers(text)):
        assert number in values, (number, sorted(values))


# --- the work-dir guard ------------------------------------------------
def test_the_default_work_dir_is_a_fresh_directory_each_time():
    """~/delhi_data is bisynced to the shared drive: a scratch directory
    derived from it propagates to everyone. The default never is."""
    made = [resolve_work_dir(), resolve_work_dir()]
    try:
        assert made[0] != made[1], "each run must get its own work dir"
        for path in made:
            assert path.is_dir()
            assert path != DATA_DIR and DATA_DIR not in path.parents
    finally:
        for path in made:
            shutil.rmtree(path, ignore_errors=True)


def test_an_explicit_work_dir_is_used_as_given():
    assert resolve_work_dir("/somewhere/else") == Path("/somewhere/else")


@pytest.mark.parametrize("inside", ["", "nested", "nested/deeper"])
def test_a_work_dir_inside_the_data_dir_is_refused(tmp_path, inside):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    candidate = data_dir / inside if inside else data_dir
    with pytest.raises(SystemExit) as exc:
        resolve_work_dir(str(candidate), data_dir=data_dir)
    assert "is inside the data directory" in str(exc.value)


def test_a_work_dir_outside_the_data_dir_is_created(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    work_dir = resolve_work_dir(str(tmp_path / "work" / "roads"),
                                data_dir=data_dir)
    assert work_dir.is_dir()


# --- render / parse_block ----------------------------------------------
def test_render_and_parse_block_round_trip():
    report = {"settlements": 4357, "area_km2_min": "2.30282e-09"}
    assert parse_block(render(report)) == {"settlements": "4357",
                                           "area_km2_min": "2.30282e-09"}


def test_render_labels_a_named_block_and_parse_block_selects_it():
    text = "\n".join([render({"a": 1}, name="access"),
                      render({"b": 2}, name="one_factor")])
    assert render({"a": 1}, name="access").splitlines()[1] == "block: access"
    assert parse_block(text, name="access") == {"a": "1"}
    assert parse_block(text, name="one_factor") == {"b": "2"}


def test_parse_block_without_a_name_returns_the_first_block():
    text = "\n".join([render({"a": 1}, name="access"),
                      render({"b": 2}, name="one_factor")])
    assert parse_block(text) == {"a": "1"}


def test_parse_block_raises_for_an_unknown_name():
    with pytest.raises(ValueError, match="no_such_block"):
        parse_block(render({"a": 1}, name="access"), name="no_such_block")


def test_parse_block_raises_on_an_unterminated_block():
    with pytest.raises(ValueError, match="unterminated"):
        parse_block(f"{FENCE}\na: 1\n")


def test_the_committed_pathology_block_survives_the_move_byte_for_byte():
    """Task 1 moves render/parse_block out of the pathology script; the
    document it has produced since 28 Aug must still render byte for byte."""
    doc = (REPO / "docs" / "data" / "layer_pathologies.md").read_text()
    assert render(parse_block(doc)) in doc


# --- the prose guard ---------------------------------------------------
def test_prose_numbers_finds_only_backticked_numbers_outside_the_blocks():
    text = "\n".join(["prose says `4,357` and `0.05` and 4069 and `touch`",
                      render({"settlements": 4357})])
    assert prose_numbers(text) == {"4357", "0.05"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest -q -W error tests/test_measure_common.py`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'scripts._measure_common'`.

- [ ] **Step 3: Write `scripts/_measure_common.py`**

```python
"""Shared plumbing for the `scripts/measure_*.py` measurement scripts.

Exactly five public names (spec § 1): the work-dir guard, the settlement
loader, the fenced-block renderer, its parser, and the fence itself. NOT a
`delhi_psi` module — these are measurement utilities, not pipeline API, and
they are imported by path-sibling scripts the way `tests/cities.py` is
imported by tests.
"""

import tempfile
from pathlib import Path

from delhi_psi import geometry, io, pipeline

FENCE = "```text"


def resolve_work_dir(cli_value=None, *, data_dir=None,
                     prefix="delhi_psi_measure_"):
    """Where a script's scratch output goes — NEVER the data directory.

    ~/delhi_data is bisynced to the shared drive, so a stray file there
    propagates to everyone. With `data_dir` given the guard fires and the
    directory is created; with `data_dir` None this only RESOLVES a path (no
    guard, no mkdir), which is the shape the default test uses.
    """
    work_dir = (Path(cli_value).expanduser() if cli_value
                else Path(tempfile.mkdtemp(prefix=prefix)))
    if data_dir is None:
        return work_dir
    data_dir = Path(data_dir).expanduser().resolve()
    resolved = work_dir.resolve()
    if resolved == data_dir or data_dir in resolved.parents:
        raise SystemExit(
            f"work directory {work_dir} is inside the data directory "
            f"{data_dir}, which these scripts never write to (it is bisynced "
            "to the shared drive)")
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir


def load_settlements(cfg, cache_dir):
    """Read, deduplicate and reproject exactly as `pipeline.preprocess` does,
    so every count below describes the universe the pipeline actually scores.

    WARM CACHE UPCASTS Polygon -> MultiPolygon. `pipeline._dedup_cached`
    returns the in-memory frame on a COLD cache but re-reads its own
    GeoPackage on a WARM one, and a GeoPackage layer carries a single
    geometry type: the raw layer's 3,801 Polygon + 556 MultiPolygon all come
    back as MultiPolygon after that round trip. So a caller that inspects
    `geom_type` — today only `count_multipolygons` in
    measure_layer_pathologies.py — MUST pass a cold `cache_dir` (a fresh
    directory), or its answer is 4,357 instead of 556. Every other predicate
    these scripts use (`intersects`, intersection length, `touch` adjacency,
    barrier flags) is type-agnostic and may share a warm cache.
    """
    source = cfg.paths.data_dir / cfg.layers.settlements.path
    gdf = io.read_layer(source)
    gdf = pipeline._dedup_cached(gdf, cache_dir, "settlements", source)
    # `remove_duplicate_geom` reset_index()es, which leaves an `index`
    # column; preprocess drops exactly these two, and bbox_frame's
    # pd.concat needs the same shape.
    gdf = gdf.drop(columns={"index", "level_0"}.intersection(gdf.columns))
    gdf = geometry.reproject(gdf, cfg.crs.epsg)
    gdf["area_km2"] = gdf.area / 1_000_000
    return gdf


def render(report, *, name=None):
    """The fenced block a `docs/data/*.md` carries verbatim.

    `name` labels the block for a multi-block script; without it the output
    is byte-identical to what the pathology script has always printed.
    """
    lines = [FENCE]
    if name is not None:
        lines.append(f"block: {name}")
    lines.extend(f"{key}: {value}" for key, value in report.items())
    lines.append("```")
    return "\n".join(lines)


def _blocks(text):
    """[(label or None, {key: value})] for every fenced block, in order."""
    out = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != FENCE:
            index += 1
            continue
        index += 1
        label, body, closed = None, {}, False
        while index < len(lines):
            line = lines[index]
            index += 1
            if line.strip() == "```":
                closed = True
                break
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "block" and label is None and not body:
                label = value
            else:
                body[key] = value
        if not closed:
            raise ValueError(f"unterminated {FENCE} block")
        out.append((label, body))
    return out


def parse_block(text, *, name=None):
    """The inverse of `render`. The SAME parser reads the committed document
    and a script's stdout, so the drift test compares like with like.

    `name` selects a labelled block; without it the FIRST block is returned,
    which is the single-block scripts' shape.
    """
    blocks = _blocks(text)
    if not blocks:
        raise ValueError(f"no {FENCE} block found")
    if name is None:
        return blocks[0][1]
    for label, body in blocks:
        if label == name:
            return body
    raise ValueError(f"no {FENCE} block labelled {name!r}; found "
                     f"{[label for label, _ in blocks]}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest -q -W error tests/test_measure_common.py`
Expected: PASS (13 tests, counting the three parametrized guard cases).

- [ ] **Step 5: Make `measure_layer_pathologies.py` import the moved names**

In `scripts/measure_layer_pathologies.py`: delete `FENCE`, `resolve_cache_dir`, `load_settlements`, `render` and `parse_block`, and replace the import block. The module docstring, every counting function and `measure()` are untouched.

Replace lines 20–53 (imports through `load_settlements`) with:

```python
import argparse
import sys

import geopandas as gpd

from delhi_psi import geometry, io, neighbors, pipeline
from delhi_psi.config import load_config
from scripts._measure_common import (load_settlements, parse_block, render,
                                     resolve_work_dir)
```

Delete the `render` and `parse_block` definitions (old lines 156–175) entirely, and replace `main`'s first half so the guard comes from `_measure_common`:

```python
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--cache-dir", default=None,
                        help="where the dedup cache goes; default a fresh "
                             "temporary directory. Never under --data-dir.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    cache_dir = resolve_work_dir(args.cache_dir, data_dir=cfg.paths.data_dir,
                                 prefix="delhi_psi_pathologies_")

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
    print(f"cache: {cache_dir}")
    print(render(measure(cfg, cache_dir)))
    return 0
```

Note: `render`, `parse_block` and `load_settlements` stay importable from `scripts.measure_layer_pathologies` (they are module-level names after the import), so nothing that imports them from there breaks. `tempfile` and `Path` are no longer used by this module — remove those two imports if the editor leaves them behind.

- [ ] **Step 6: Point the pathology tests at the moved names**

In `tests/test_layer_pathologies.py`, replace the import block (lines 19–21):

```python
from scripts._measure_common import parse_block, resolve_work_dir
from scripts.measure_layer_pathologies import (count_isolated_bbox,
                                               count_isolated_touch)
```

and replace the cache-dir test (lines 111–123) with:

```python
def test_the_cache_dir_default_is_a_fresh_directory_outside_the_data_dir():
    """~/delhi_data is bisynced to the shared drive: a cache written there
    propagates to everyone. The default must never be derived from it. The
    guard now lives in scripts/_measure_common.resolve_work_dir; this script
    keeps its historic --cache-dir flag name."""
    made = [resolve_work_dir(), resolve_work_dir()]
    try:
        assert made[0] != made[1], "each run must get its own cache"
        for path in made:
            assert path.is_dir()
            assert path != DATA_DIR and DATA_DIR not in path.parents
    finally:
        for path in made:
            shutil.rmtree(path, ignore_errors=True)
    assert resolve_work_dir("/somewhere/else") == Path("/somewhere/else")
```

- [ ] **Step 7: Run both test files**

Run: `uv run pytest -q -W error tests/test_measure_common.py tests/test_layer_pathologies.py`
Expected: PASS. The two `needs_data` tests skip on a machine without `~/delhi_data`; on this machine they run and must still reproduce the committed counts — proof the move changed no behaviour.

- [ ] **Step 8: Run the full suite in the FOREGROUND**

Run: `uv run pytest -q -W error` (about 6.5 minutes — wait for it; do not background it)
Expected: PASS, no new failures.

- [ ] **Step 9: Commit**

```bash
git add scripts/_measure_common.py scripts/measure_layer_pathologies.py \
        tests/test_measure_common.py tests/test_layer_pathologies.py
git commit -m "$(cat <<'EOF'
feat(measure): shared _measure_common for the measurement scripts (DEL-49/50/51/52)

The work-dir guard, the settlement loader, the fenced-block renderer and its
parser move out of measure_layer_pathologies.py, which imports them back; the
block it prints is byte-identical. render/parse_block gain an optional block
`name=` so a multi-block script can label its output.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
EOF
)"
```

---

### Task 2: DEL-50 — corner-only contact pairs

**Files:**
- Modify: `scripts/measure_layer_pathologies.py` (add `corner_only_pairs` / `count_corner_only_pairs` next to `count_overlapping_pairs`; two new keys in `measure`)
- Modify: `tests/test_layer_pathologies.py` (four new tests; the key-set test tolerates the two pending keys)
- Modify: `docs/data/layer_pathologies.md` (definition paragraph only — the VALUES arrive in Task 6)

**Interfaces:**
- Consumes: `scripts._measure_common.parse_block` (Task 1).
- Produces:
  - `scripts.measure_layer_pathologies.corner_only_pairs(gdf, *, id_col) -> list[tuple[str, str]]`
  - `scripts.measure_layer_pathologies.count_corner_only_pairs(gdf, *, id_col) -> int`
  - two new block keys: `corner_only_pairs` (int), `corner_only_settlements` (int)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layer_pathologies.py` (and add `from tests.cities import MESSY, ORACULUM` plus `count_corner_only_pairs`, `corner_only_pairs` to the script import):

```python
# --- DEL-50: corner-only contact pairs ---------------------------------
def test_the_messy_city_has_exactly_one_corner_only_pair():
    """`T`'s contact with `L` is a single Point (docs/oracle/messy-city.md):
    zero length, zero area. It is a `bbox` neighbour and never a `touch`
    one, and a 0 km distance band would make it a neighbour again — which is
    exactly the question DEL-50 asks of the real layer."""
    city = MESSY.load_settlements()
    assert corner_only_pairs(city, id_col="USO_AREA_U") == [("L", "T")]
    assert count_corner_only_pairs(city, id_col="USO_AREA_U") == 1


def test_oraculum_has_no_corner_only_pairs():
    """Every Oraculum settlement is an axis-aligned rectangle in a grid; the
    pairs that meet, meet along an edge."""
    assert count_corner_only_pairs(ORACULUM.load_settlements(),
                                   id_col="USO_AREA_U") == 0


def test_a_shared_edge_and_a_shared_corner_are_told_apart():
    """A-B share an edge (positive length), B-C share one point, A-C are
    disjoint: exactly one corner-only pair."""
    gdf = gpd.GeoDataFrame(
        {"id": ["A", "B", "C"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 1, 3, 2)],
        crs="EPSG:7760")
    assert corner_only_pairs(gdf, id_col="id") == [("B", "C")]


def test_an_overlapping_pair_is_not_corner_only():
    """The intersection of an overlap has AREA, so the measure test — not a
    geom_type test — excludes it."""
    gdf = gpd.GeoDataFrame(
        {"id": ["O1", "O2"]},
        geometry=[box(0, 0, 2, 1), box(1, 0, 3, 1)], crs="EPSG:7760")
    assert count_corner_only_pairs(gdf, id_col="id") == 0
```

Also change the key-set test so the two new keys are accepted but not yet required (the values come from the run step):

```python
COUNT_KEYS = ("settlements", "rectangles", "multipolygons", "isolated_bbox",
              "isolated_touch", "no_population", "overlapping_pairs")
# DEL-50: the script emits these two from this commit on; the committed
# document gains them when the run step (task 6) pastes a fresh block, which
# also moves them into COUNT_KEYS above.
PENDING_KEYS = ("corner_only_pairs", "corner_only_settlements")
```

and in `test_the_doc_has_the_fenced_block_with_every_required_key`, replace the final equality with:

```python
    for key in PENDING_KEYS:
        if key in committed:
            assert committed[key].isdigit(), (key, committed[key])
    expected = (set(COUNT_KEYS) | set(AREA_KEYS)
                | {f"multi_settlement_points_{s}" for s in POINT_SERVICES})
    assert expected <= set(committed)
    assert set(committed) <= expected | set(PENDING_KEYS)
```

and in `test_a_fresh_run_reproduces_the_committed_counts`, replace `assert set(measured) == set(committed)` with:

```python
    # `<=`, not `==`, until the run step pastes the two DEL-50 keys: a fresh
    # run emits them, the committed block does not carry them yet.
    assert set(committed) <= set(measured), sorted(set(committed)
                                                   - set(measured))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -W error tests/test_layer_pathologies.py`
Expected: FAIL at collection — `ImportError: cannot import name 'corner_only_pairs' from 'scripts.measure_layer_pathologies'`.

- [ ] **Step 3: Implement the counter**

In `scripts/measure_layer_pathologies.py`, immediately after `count_overlapping_pairs`:

```python
def corner_only_pairs(gdf, *, id_col):
    """Pairs whose intersection is NON-EMPTY but has zero length and zero
    area — they meet at one or more isolated points. `touch` (positive shared
    length) does NOT make them neighbours; a 0 km distance band does.

    Same sjoin-then-test shape as `count_overlapping_pairs`: the join narrows
    the candidates so the measure test never runs on all n^2 pairs, and
    `left < right` keeps one of each unordered pair. Shapely returns a Point
    or MultiPoint for such an intersection, but the test is on MEASURES, not
    on `geom_type`, so a GeometryCollection of points also counts and one
    containing a line does not.
    """
    frame = gdf[[id_col, "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(frame, frame, how="inner", predicate="intersects")
    geoms = frame.geometry
    ids = frame[id_col]
    out = []
    for left, right in zip(joined.index, joined["index_right"]):
        if left >= right:
            continue
        shared = geoms.iloc[left].intersection(geoms.iloc[right])
        if not shared.is_empty and shared.length == 0 and shared.area == 0:
            out.append((ids.iloc[left], ids.iloc[right]))
    return out


def count_corner_only_pairs(gdf, *, id_col):
    """How many pairs meet at a point and nowhere else."""
    return len(corner_only_pairs(gdf, id_col=id_col))
```

and in `measure`, directly after the `"overlapping_pairs"` entry (so the two keys sit with the pathology they belong to), replace the dict tail:

```python
        "overlapping_pairs": count_overlapping_pairs(gdf, id_col=id_col),
    }
    pairs = corner_only_pairs(gdf, id_col=id_col)
    report["corner_only_pairs"] = len(pairs)
    report["corner_only_settlements"] = len(
        {settlement for pair in pairs for settlement in pair})
    for service, path in sorted(cfg.services.point.items()):
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/test_layer_pathologies.py`
Expected: PASS (the four new tests plus the existing ones; `needs_data` tests skip or pass).

- [ ] **Step 5: Document the definition (no numbers yet)**

In `docs/data/layer_pathologies.md`, under `## Reading the numbers`, insert after the `overlapping_pairs` bullet:

```markdown
- `corner_only_pairs` / `corner_only_settlements` — polygon pairs whose
  intersection is non-empty but has **zero length and zero area**: they meet
  at one or more isolated points and nowhere else, and the settlements
  involved. Raj ratified shared-border adjacency on 28 Aug 2026 (decision log
  § 3), and a corner is not a border: such a pair is a neighbour under `bbox`
  and under a 0 km distance band, and NOT under `touch`. The messy city's
  `T`/`L` is the hand-checkable case (`docs/oracle/messy-city.md`); Oraculum
  has none.
```

- [ ] **Step 6: Run the full suite in the FOREGROUND**

Run: `uv run pytest -q -W error` (about 6.5 minutes)
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/measure_layer_pathologies.py tests/test_layer_pathologies.py \
        docs/data/layer_pathologies.md
git commit -m "$(cat <<'EOF'
feat(measure): count corner-only contact pairs on the settlement layer (DEL-50)

Pairs whose intersection is non-empty with zero length and zero area — a
corner, not a border. Two new keys in the pathology block; the real-layer
values are pasted by the run step. Pinned on the messy city (T/L = 1) and
Oraculum (0), plus a synthetic edge/corner/disjoint frame and an overlap.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
EOF
)"
```

---

### Task 3: DEL-51 — `scripts/inventory_barriers.py` + `docs/data/barriers.md`

**Files:**
- Create: `scripts/inventory_barriers.py`
- Create: `tests/test_inventory_barriers.py`
- Create: `docs/data/barriers.md`

**Interfaces:**
- Consumes: `scripts._measure_common.{load_settlements, render, resolve_work_dir}` and, in the test, `parse_block` (Task 1); `tests.test_measure_common.{DATA_DIR, MEASURE_CACHE, needs_measure_cache, assert_prose_numbers_come_from_the_blocks}` (Task 1); `delhi_psi.geometry.{barrier_flags, reproject}`; `delhi_psi.neighbors.combine_barrier_flags`; `delhi_psi.io.read_layer`; `delhi_psi.pipeline.ID_COL`.
- Produces:
  - `scripts.inventory_barriers.metadata_dates(text) -> dict[str, str]`
  - `scripts.inventory_barriers.sidecar_facts(path) -> dict[str, str]`
  - `scripts.inventory_barriers.layer_facts(gdf, projected, *, settlements) -> dict[str, str | int]`
  - `scripts.inventory_barriers.attributes(layers, *, columns=NAME_COLUMNS, cap=VALUE_CAP) -> dict[str, str]`
  - `scripts.inventory_barriers.barrier_flagged(settlements, layers, *, combine="any", configured=None, id_col=ID_COL) -> GeoDataFrame`
  - `scripts.inventory_barriers.inventory(layers, settlements, *, paths=None, epsg=7760, id_col=ID_COL, combine="any", configured=None) -> {"layers": dict, "attributes": dict}`
  - `scripts.inventory_barriers.main(argv=None) -> int`
  - block names: `layers`, `attributes`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_inventory_barriers.py`:

```python
"""The barrier-layer inventory (DEL-51, spec § 2.3).

Fixture-level tests run everywhere. The doc-drift tests wake up when the run
step pastes the measured blocks into docs/data/barriers.md; until then they
skip with a reason that says so. The REAL-DATA drift test needs one more
thing: DELHI_PSI_MEASURE_CACHE, the shared warm work dir the run step
exports, without which it skips rather than paying ~4.5 min for a cold
settlement dedup in every implementer's suite run.
"""
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from scripts._measure_common import FENCE, parse_block
from scripts.inventory_barriers import (attributes, barrier_flagged,
                                        inventory, layer_facts, main,
                                        metadata_dates)
from tests.cities import ORACULUM
from tests.test_measure_common import (DATA_DIR, MEASURE_CACHE,
                                       assert_prose_numbers_come_from_the_blocks,
                                       needs_measure_cache)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "barriers.md"

LAYER_KEYS = ("features", "geom_types", "crs", "length_km",
              "within_settlement_bbox", "crea_date", "crea_time", "mod_date",
              "has_qpj", "flagged_settlements")
CONFIGURED = ("canal", "railway", "drain")

ESRI_SIDECAR = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<metadata xml:lang=\"en\"><Esri><CreaDate>20200802</CreaDate>\n"
    "<CreaTime>17595100</CreaTime></Esri></metadata>\n")


def committed_blocks():
    """The two blocks the document carries, or a skip while it carries none."""
    if not DOC.exists() or FENCE not in DOC.read_text():
        pytest.skip(f"{DOC} carries no measured block yet — the run step "
                    "pastes it")
    text = DOC.read_text()
    return parse_block(text, name="layers"), parse_block(text,
                                                         name="attributes")


# --- the fixture city --------------------------------------------------
def test_the_oraculum_canal_is_inventoried_correctly():
    """One 450 m LineString along the A/D edge, in the fixture CRS, inside
    the settlement layer's bounding box."""
    city = ORACULUM.load_settlements()
    got = inventory({"canal": ORACULUM.load_barriers()}, city,
                    epsg=ORACULUM.epsg)["layers"]
    assert got["settlements"] == 7
    assert got["canal_features"] == 1
    assert got["canal_geom_types"] == "LineString"
    assert got["canal_crs"] == "EPSG:7760"
    assert got["canal_length_km"] == "0.45"
    assert got["canal_within_settlement_bbox"] == "yes"
    assert got["canal_flagged_settlements"] == 2
    assert got["flagged_any"] == 2
    # no sidecar next to an in-memory fixture layer
    assert got["canal_crea_date"] == "none"


def test_the_oraculum_canal_flags_a_and_d():
    """The same code path production uses — geometry.barrier_flags plus
    neighbors.combine_barrier_flags — so the doc's flagged count is the
    number the pipeline itself would produce."""
    city = ORACULUM.load_settlements()
    flagged = barrier_flagged(city, {"canal": ORACULUM.load_barriers()},
                              combine="any", configured=("canal",))
    assert set(flagged.loc[flagged["canal"], "USO_AREA_U"]) == {"A", "D"}
    assert set(flagged.loc[flagged["barrier"], "USO_AREA_U"]) == {"A", "D"}


def test_layer_facts_reports_the_source_crs_and_the_projected_length():
    line = gpd.GeoDataFrame(
        {"name": ["x"]}, geometry=[LineString([(0, 0), (0, 1000)])],
        crs="EPSG:7760")
    settlements = ORACULUM.load_settlements()
    got = layer_facts(line, line, settlements=settlements)
    assert got["length_km"] == "1"
    assert got["within_settlement_bbox"] == "no"


# --- the ESRI sidecar --------------------------------------------------
def test_metadata_dates_reads_an_esri_sidecar():
    assert metadata_dates(ESRI_SIDECAR) == {"crea_date": "20200802",
                                            "crea_time": "17595100",
                                            "mod_date": "none"}


def test_metadata_dates_reports_none_for_an_absent_tag():
    assert metadata_dates("<metadata><Esri/></metadata>") == {
        "crea_date": "none", "crea_time": "none", "mod_date": "none"}


# --- the attributes block ----------------------------------------------
def test_name_values_are_capped_at_twenty():
    gdf = gpd.GeoDataFrame(
        {"CAN_NM": [f"canal {n:02d}" for n in range(25)]},
        geometry=[LineString([(n, 0), (n, 1)]) for n in range(25)],
        crs="EPSG:7760")
    got = attributes({"canal": gdf})
    assert got["canal_columns"] == "CAN_NM"
    values = got["canal_values_CAN_NM"]
    assert values.startswith("canal 00;canal 01;")
    assert values.endswith("... (+5 more)")
    assert len(values.split(";")) == 21          # 20 names plus the tail


# --- the committed document --------------------------------------------
def test_the_doc_block_has_every_required_key():
    layers, attributes_block = committed_blocks()
    assert layers["settlements"].isdigit()
    assert layers["flagged_any"].isdigit()
    for name in CONFIGURED:
        for key in LAYER_KEYS:
            assert f"{name}_{key}" in layers, f"{name}_{key}"
        assert layers[f"{name}_features"].isdigit()
        float(layers[f"{name}_length_km"])
        assert f"{name}_columns" in attributes_block, name


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, committed_blocks())


@needs_measure_cache
def test_a_fresh_run_reproduces_the_committed_blocks():
    """The real-data drift check. It runs the script over the 4,357-polygon
    layer, so it needs a settlement dedup — ~4.5 min cold. It therefore takes
    its work dir from DELHI_PSI_MEASURE_CACHE (one warm cache per machine,
    shared with the roads drift test) and SKIPS when that is unset, rather
    than gating on `needs_data` alone and charging every per-task suite run
    for a cold dedup on a machine that has the data. The run step exports it.
    A warm cache is safe here: nothing in this script inspects `geom_type`.
    """
    layers, attributes_block = committed_blocks()
    proc = subprocess.run(
        [sys.executable, "scripts/inventory_barriers.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--all-candidates", "--work-dir", MEASURE_CACHE],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="layers") == layers
    assert parse_block(proc.stdout, name="attributes") == attributes_block


def test_main_prints_its_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--all-candidates" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -W error tests/test_inventory_barriers.py`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'scripts.inventory_barriers'`.

- [ ] **Step 3: Write the script**

Create `scripts/inventory_barriers.py`:

```python
"""Inventory the barrier layers the pipeline uses (DEL-51, spec § 2.3).

Raj asked where the barrier layers came from. What the repo can establish
MECHANICALLY is an inventory: how many features, what geometry, which CRS,
how long, what attribute schema, which ESRI/QGIS sidecars sit beside the
files, and how many settlements each layer flags under today's rule. The
provenance sentence itself is prose in docs/data/barriers.md, written from
this evidence — this script makes no claim about an agency.

READ-ONLY over --data-dir. Scratch (the settlement dedup cache) goes under
--work-dir, which is never inside the data directory.

    uv run python scripts/inventory_barriers.py --config code-2025 \
        --all-candidates --work-dir ~/measure_work/cache

Prints provenance lines, then two fenced blocks: `layers` (the drift-tested
counts, lengths, dates and flagged settlements) and `attributes` (each
layer's schema and the distinct values of its name-like columns).
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from delhi_psi import geometry, io, neighbors
from delhi_psi.config import load_config
from delhi_psi.pipeline import ID_COL
from scripts._measure_common import load_settlements, render, resolve_work_dir

DEFAULT_EPSG = 7760
# The columns that say WHOSE layer this is, per the 5 Sep 2026 survey.
NAME_COLUMNS = ("CAN_NM", "RL_ZONE", "Drain_Name", "DISTRICT", "AC_NAME")
VALUE_CAP = 20
METADATA_TAGS = {"CreaDate": "crea_date", "CreaTime": "crea_time",
                 "ModDate": "mod_date"}
# The other copies found under the data root on 5 Sep 2026 (--all-candidates).
CANDIDATES = {
    "canal_checked": "Barrier_Clip/Canal/new/checked_Canal.shp",
    "canal_root": "canal.data/canal.shp",
    "railway_root": "railway.data/railway.shp",
    "drain_root": "drain.data/drain.shp",
}


def metadata_dates(text):
    """CreaDate / CreaTime / ModDate from an ESRI `.shp.xml` sidecar."""
    root = ET.fromstring(text)
    out = {}
    for tag, key in METADATA_TAGS.items():
        node = next((found for found in root.iter(tag)
                     if (found.text or "").strip()), None)
        out[key] = "none" if node is None else node.text.strip()
    return out


def sidecar_facts(path):
    """The ESRI dates beside a shapefile, and whether QGIS wrote a `.qpj`.

    `path` is the `.shp` itself; ESRI names its sidecar `<file>.shp.xml`, so
    this appends rather than replacing the suffix. No path (an in-memory
    fixture layer) means every fact is unknown, which prints as "none".
    """
    unknown = {key: "none" for key in METADATA_TAGS.values()}
    if path is None:
        return {**unknown, "has_qpj": "none"}
    path = Path(path)
    xml = path.with_name(path.name + ".xml")
    dates = metadata_dates(xml.read_text()) if xml.exists() else unknown
    return {**dates,
            "has_qpj": "yes" if path.with_suffix(".qpj").exists() else "no"}


def _within(inner, outer):
    """Bounds containment, done on numbers: a barrier layer's bounding box
    can be degenerate (a straight canal has zero height), and shapely's
    `within` is false for a zero-area polygon."""
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def layer_facts(gdf, projected, *, settlements):
    """`gdf` as read (its own CRS); `projected` the same layer in the target
    CRS, where `.length` is metres and the bounding box is comparable with
    the settlement layer's."""
    return {
        "features": len(gdf),
        "geom_types": ";".join(sorted(set(gdf.geom_type.dropna()))),
        "crs": gdf.crs.to_string() if gdf.crs is not None else "none",
        "length_km": f"{projected.length.sum() / 1000:.6g}",
        "within_settlement_bbox": (
            "yes" if _within(projected.total_bounds, settlements.total_bounds)
            else "no"),
    }


def attributes(layers, *, columns=NAME_COLUMNS, cap=VALUE_CAP):
    """The `attributes` block: each layer's schema, and the distinct values
    of its name-like columns capped at `cap` — the point is to see whose
    layer this is, not to dump 616 drain names."""
    report = {}
    for name, gdf in layers.items():
        report[f"{name}_columns"] = ";".join(
            str(column) for column in gdf.columns if column != "geometry")
        for column in columns:
            if column not in gdf.columns:
                continue
            values = sorted({str(value) for value in gdf[column].dropna()})
            tail = f";... (+{len(values) - cap} more)" if len(values) > cap \
                else ""
            report[f"{name}_values_{column}"] = ";".join(values[:cap]) + tail
    return report


def barrier_flagged(settlements, layers, *, combine="any", configured=None,
                    id_col=ID_COL):
    """Production's own flag columns: one per layer, plus the combined
    `barrier` over the CONFIGURED layers only, so a --all-candidates copy
    cannot inflate the number production would produce. Every frame must
    already be in the same CRS."""
    configured = tuple(configured) if configured is not None else tuple(layers)
    frame = geometry.barrier_flags(settlements, layers, id_col=id_col)
    return neighbors.combine_barrier_flags(frame, layers=configured,
                                           combine=combine)


def inventory(layers, settlements, *, paths=None, epsg=DEFAULT_EPSG,
              id_col=ID_COL, combine="any", configured=None):
    """Both blocks: {"layers": {...}, "attributes": {...}}."""
    paths = paths or {}
    configured = tuple(configured) if configured is not None else tuple(layers)
    projected = {name: geometry.reproject(gdf, epsg)
                 for name, gdf in layers.items()}
    flagged = barrier_flagged(settlements, projected, combine=combine,
                              configured=configured, id_col=id_col)

    report = {"settlements": len(settlements)}
    for name, gdf in layers.items():
        facts = {**layer_facts(gdf, projected[name], settlements=settlements),
                 **sidecar_facts(paths.get(name))}
        report.update({f"{name}_{key}": value
                       for key, value in facts.items()})
        report[f"{name}_flagged_settlements"] = int(flagged[name].sum())
    report["flagged_any"] = int(flagged["barrier"].sum())
    return {"layers": report, "attributes": attributes(layers)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch (the settlement dedup cache); default a "
                             "fresh temporary directory. Never under --data-dir.")
    parser.add_argument("--all-candidates", action="store_true",
                        help="also inventory the other copies of these layers "
                             "found under the data root on 5 Sep 2026")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_barriers_")

    paths = {name: cfg.paths.data_dir / path
             for name, path in cfg.layers.barriers.items()}
    if args.all_candidates:
        for name, relative in CANDIDATES.items():
            candidate = cfg.paths.data_dir / relative
            if candidate.exists():
                paths[name] = candidate
            else:
                print(f"candidate missing: {candidate}")
    layers = {name: io.read_layer(path) for name, path in paths.items()}
    settlements = load_settlements(cfg, work_dir)

    for name, path in paths.items():
        print(f"layer {name}: {path}")
    print(f"work-dir: {work_dir}")
    blocks = inventory(layers, settlements, paths=paths, epsg=cfg.crs.epsg,
                       id_col=cfg.layers.settlements.id_col,
                       combine=cfg.methodology.barrier.combine,
                       configured=tuple(cfg.layers.barriers))
    for name, report in blocks.items():
        print(render(report, name=name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/test_inventory_barriers.py -rs`
Expected: PASS — the fixture and sidecar tests green; the two committed-document tests SKIP with "carries no measured block yet" (the document has no block until Task 6), and the real-data drift test SKIPS with "set DELHI_PSI_MEASURE_CACHE to run the real-data drift check". Confirm both skip reasons in the `-rs` summary. **No real-data run happens in this task** — that skip is the point of the env-var gate.

- [ ] **Step 5: Write the document skeleton with the provenance prose**

Create `docs/data/barriers.md`:

```markdown
# The barrier layers: an inventory

Raj asked on 28 Aug 2026 where the barrier layers came from — city data, or
drawn by hand (decision log
`docs/decisions/2026-08-28-raj-methodology-decisions.md` § 4, DEL-51). What
this repository can establish mechanically is an INVENTORY; it is below,
produced by `scripts/inventory_barriers.py`, which reads the layers named by
the `code-2025` profile, flags settlements through the pipeline's own
`geometry.barrier_flags` + `neighbors.combine_barrier_flags`, and writes
nothing under the data directory. `tests/test_inventory_barriers.py` re-runs
it and compares the blocks (it skips when the data is not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

## What this tells us about provenance

**No claim about an agency is made here.** The evidence, and what it
suggests:

- **The attribute schemas are an official GIS office's, not a hand
  digitiser's.** The canal layer carries `CAN_NM` (canal name), `CAN_CLSF`
  (classification), `EL_GND` (ground elevation) and `DIST_NM`; the railway
  layer carries `RL_ZONE`, whose values name a railway zone; the drain layer
  carries `Drain_type`, `Drain_Name`, `MAINTAINED`, `AC_NAME` (assembly
  constituency) and `DISTRICT`. Ground elevations, maintenance
  responsibility and assembly constituencies are fields a utility or survey
  department keeps; nobody tracing lines over a basemap invents them. The
  `attributes` block below carries the schemas and the distinct values, so
  this paragraph can be checked rather than believed.
- **The clipped files were written by ArcGIS on 2 Aug 2020.** `Canal.shp.xml`
  and `Railway_Line.shp.xml` carry ESRI `<CreaDate>`/`<CreaTime>` elements
  and a `lineage` of `RepairGeometry` runs from ArcGIS Pro; the `layers`
  block reports the dates. A `.qpj` sidecar beside each layer says QGIS also
  opened them. `Major_Drain.shp` has no `.shp.xml`.
- **The paper's "manually marked" sentence describes the CLIPPING, not the
  digitising.** The April 2026 draft (pp. 15–16) says the team "manually
  marked areas that had river or railroad tracks". The directory name is
  `Barrier_Clip/`, and the extra copies at the data root
  (`canal.data/`, `railway.data/`, `drain.data/`) are the unclipped
  originals — so what was done by hand is plausibly the selection and
  clipping of an existing layer to the study area.

**The question that remains, for Bijoy (batched reply, item 7):** which
agency's layer are these — and what exactly did "manually marked" cover:
selecting features, clipping to the study area, or drawing lines?
```

*(The provenance bullet list — Run date / Inputs / Commit / Command — and the
two fenced blocks are inserted by the run step, which is when those facts
exist.)*

- [ ] **Step 6: Run the test file again**

Run: `uv run pytest -q -W error tests/test_inventory_barriers.py -rs`
Expected: PASS, with the two committed-document tests still SKIPPED for the stated reason (the doc now exists but carries no block) and the real-data drift test still SKIPPED for the missing `DELHI_PSI_MEASURE_CACHE`.

- [ ] **Step 7: Run the full suite in the FOREGROUND**

Run: `uv run pytest -q -W error` (about 6.5 minutes)
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/inventory_barriers.py tests/test_inventory_barriers.py \
        docs/data/barriers.md
git commit -m "$(cat <<'EOF'
feat(measure): inventory the barrier layers (DEL-51)

Per configured layer (and, with --all-candidates, the other copies found on
5 Sep 2026): feature count, geometry types, source CRS, length, bounding box
against the settlement layer's, attribute schema and capped distinct name
values, the ESRI .shp.xml dates and the .qpj sidecar, and how many
settlements each flags through production's own flag code path. The
provenance prose is in docs/data/barriers.md; the measured blocks are pasted
by the run step.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
EOF
)"
```

---

### Task 4: DEL-52 — `scripts/measure_psi_columns.py` + `docs/data/psi_columns.md`

**Files:**
- Create: `scripts/measure_psi_columns.py`
- Create: `tests/test_measure_psi_columns.py`
- Create: `docs/data/psi_columns.md`

**Interfaces:**
- Consumes: `scripts._measure_common.{render, resolve_work_dir}` and, in the test, `parse_block` (Task 1); `tests.test_measure_common.{DATA_DIR, needs_data, assert_prose_numbers_come_from_the_blocks}` (Task 1); `delhi_psi.config.load_config`; `delhi_psi.pipeline.TYPE_COL`.
- Produces:
  - `scripts.measure_psi_columns.FIGURE_4_BARS: dict[str, float]` (eight types), `FIGURE_TOLERANCE = 0.002`, `MATCH_FLOOR = 6`
  - `scripts.measure_psi_columns.type_means(frame, *, column, type_col="USO_FINAL") -> dict[str, float]`
  - `scripts.measure_psi_columns.score_candidate(means, *, bars=FIGURE_4_BARS, tolerance=FIGURE_TOLERANCE) -> tuple[int, float]`
  - `scripts.measure_psi_columns.score_candidates(frames, *, columns=CANDIDATE_COLUMNS, type_col="USO_FINAL", bars=FIGURE_4_BARS, tolerance=FIGURE_TOLERANCE) -> dict`
  - `scripts.measure_psi_columns.cross_check(baseline, verify, *, columns=CANDIDATE_COLUMNS, type_col="USO_FINAL", atol=1e-9) -> dict`
  - `scripts.measure_psi_columns.measure(baseline_dir, *, verify_dir=None) -> dict` (ONE unlabeled block)
  - `scripts.measure_psi_columns.main(argv=None) -> int`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_measure_psi_columns.py`:

```python
"""Which PSI column and which denominator the paper's figures report
(DEL-52, spec § 2.4).

The scoring functions are proven on hand-built frames whose means are known
by construction; the real comparison against the July 2025 baseline files is
the run step's.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts._measure_common import FENCE, parse_block
from scripts.measure_psi_columns import (FIGURE_4_BARS, FIGURE_TOLERANCE,
                                         cross_check, main, score_candidate,
                                         score_candidates, type_means)
from tests.test_measure_common import (DATA_DIR,
                                       assert_prose_numbers_come_from_the_blocks,
                                       needs_data)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "psi_columns.md"
BASELINE_DIR = DATA_DIR / "psi_2020_results"
VERIFY_DIR = DATA_DIR / "phase3_verify"
CANDIDATES = ("unnorm_psi_popsize", "unnorm_psi_popdensity",
              "norm_psi_popsize", "norm_psi_popdensity")


def frame(**columns):
    """A tiny PSI-output-shaped frame: one row per (type, value) pair."""
    return pd.DataFrame(columns)


def committed_block():
    if not DOC.exists() or FENCE not in DOC.read_text():
        pytest.skip(f"{DOC} carries no measured block yet — the run step "
                    "pastes it")
    return parse_block(DOC.read_text())


def test_figure_4_bars_has_the_eight_types_from_the_april_2026_draft():
    """Figure 4 has eight bars — no RV, no Other (spec § 2.4)."""
    assert set(FIGURE_4_BARS) == {"JJR", "JJC", "SDA", "Planned", "RUAC",
                                  "UAC", "UV", "Industrial"}
    assert all(0 < value < 0.05 for value in FIGURE_4_BARS.values())
    assert FIGURE_TOLERANCE == 0.002


def test_type_means_averages_per_settlement_type():
    got = type_means(frame(USO_FINAL=["JJC", "JJC", "Planned"],
                           unnorm_psi=[0.0, 0.004, 0.044]),
                     column="unnorm_psi")
    assert got == {"JJC": 0.002, "Planned": 0.044}


def test_score_candidate_counts_matches_and_the_max_gap():
    """Two bars hit exactly, one misses by 0.01 — one match short of the
    figure's eight, and the gap is the miss."""
    means = dict(FIGURE_4_BARS)
    means["JJC"] = FIGURE_4_BARS["JJC"] + 0.01
    matched, maxgap = score_candidate(means)
    assert matched == 7
    assert maxgap == pytest.approx(0.01)


def test_a_missing_figure_type_can_never_match():
    means = {name: value for name, value in FIGURE_4_BARS.items()
             if name != "UV"}
    matched, maxgap = score_candidate(means)
    assert matched == 7
    assert maxgap == float("inf")


def test_score_candidates_picks_the_column_that_reproduces_the_figure():
    """`unnorm_psi` under popdensity carries the figure values exactly;
    every other candidate is stretched to [0, 1] or shifted."""
    types = list(FIGURE_4_BARS)
    exact = [FIGURE_4_BARS[name] for name in types]
    stretched = [value * 20 for value in exact]
    frames = {
        "popsize": frame(USO_FINAL=types, unnorm_psi=stretched,
                         norm_psi=stretched),
        "popdensity": frame(USO_FINAL=types, unnorm_psi=exact,
                            norm_psi=stretched),
    }
    got = score_candidates(frames)
    assert got["best_candidate"] == "unnorm_psi_popdensity"
    assert got["matched_unnorm_psi_popdensity"] == 8
    assert got["maxgap_unnorm_psi_popdensity"] == "0.0000"
    assert got["mean_unnorm_psi_popdensity_JJC"] == "0.0015"
    assert got["matched_unnorm_psi_popsize"] == 0


def test_the_best_candidate_tie_break_prefers_the_smaller_max_gap():
    """Both candidates match the same number of bars; the one that is closer
    on the bar it misses wins."""
    types = list(FIGURE_4_BARS)
    near = [FIGURE_4_BARS[name] for name in types]
    near[0] += 0.003            # just outside the tolerance
    far = [FIGURE_4_BARS[name] for name in types]
    far[0] += 0.009
    frames = {"popsize": frame(USO_FINAL=types, unnorm_psi=far, norm_psi=far),
              "popdensity": frame(USO_FINAL=types, unnorm_psi=near,
                                  norm_psi=far)}
    got = score_candidates(frames)
    assert got["matched_unnorm_psi_popdensity"] == 7
    assert got["matched_unnorm_psi_popsize"] == 7
    assert got["best_candidate"] == "unnorm_psi_popdensity"


def test_cross_check_raises_when_the_refactored_run_disagrees():
    types = list(FIGURE_4_BARS)
    values = [FIGURE_4_BARS[name] for name in types]
    baseline = {"popsize": frame(USO_FINAL=types, unnorm_psi=values,
                                 norm_psi=values)}
    moved = [value + 1e-6 for value in values]
    verify = {"popsize": frame(USO_FINAL=types, unnorm_psi=moved,
                               norm_psi=values)}
    with pytest.raises(ValueError, match="differs from the July 2025 baseline"):
        cross_check(baseline, verify)


def test_cross_check_reports_the_max_difference_when_the_runs_agree():
    types = list(FIGURE_4_BARS)
    values = [FIGURE_4_BARS[name] for name in types]
    frames = {"popsize": frame(USO_FINAL=types, unnorm_psi=values,
                               norm_psi=values)}
    got = cross_check(frames, frames)
    assert got["verify_maxdiff_unnorm_psi_popsize"] == "0.0e+00"


def test_the_doc_block_has_every_required_key():
    committed = committed_block()
    for candidate in CANDIDATES:            # <column>_<denom>
        assert committed[f"matched_{candidate}"].isdigit()
        float(committed[f"maxgap_{candidate}"])
        for name in FIGURE_4_BARS:
            key = f"mean_{candidate}_{name}"
            assert key in committed, key
            float(committed[key])
    assert committed["best_candidate"] in CANDIDATES


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, [committed_block()])


@needs_data
def test_a_fresh_run_reproduces_the_committed_block(tmp_path):
    """This one keeps the PLAIN `needs_data` gate — no DELHI_PSI_MEASURE_CACHE
    marker. The script reads two CSVs and never touches the settlement layer,
    so there is no dedup and no O(n^2) pass: it costs seconds, and it is
    honest for it to run in every implementer's suite run on this machine."""
    committed = committed_block()
    proc = subprocess.run(
        [sys.executable, "scripts/measure_psi_columns.py",
         "--baseline-dir", str(BASELINE_DIR), "--verify-dir", str(VERIFY_DIR),
         "--data-dir", str(DATA_DIR), "--work-dir", str(tmp_path / "work")],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout) == committed


def test_main_prints_its_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--baseline-dir" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q -W error tests/test_measure_psi_columns.py`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'scripts.measure_psi_columns'`.

- [ ] **Step 3: Write the script**

Create `scripts/measure_psi_columns.py`:

```python
"""Which PSI column and which denominator do the paper's figures report?
(DEL-52, spec § 2.4.)

The April 2026 draft's Figure 4 is a bar chart of the mean PSI per settlement
type, y-axis "Mean Public Services Index (per person per square kilometer)",
eight bars, footnote 12 saying the average "rarely exceeds 0.05". This script
turns the read-off bar values into a measured match: for each of the four
candidates {unnorm_psi, norm_psi} x {popsize, popdensity} it computes the
mean per USO_FINAL type in the July 2025 baseline outputs and scores it
against the figure.

The baseline files are the ones that PRODUCED the figures, so they are the
comparison; --verify-dir is a cross-check that the refactored `code-2025` run
would give the same answer. Both are opened READ-ONLY.

    uv run python scripts/measure_psi_columns.py \
        --baseline-dir ~/delhi_data/psi_2020_results \
        --verify-dir ~/delhi_data/phase3_verify

Prints provenance lines, then one fenced block.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from delhi_psi.config import load_config
from delhi_psi.pipeline import TYPE_COL
from scripts._measure_common import render, resolve_work_dir

# Read off Figure 4 ("Mean public service index by settlement", p. 40 of the
# April 2026 draft PDF) on 5 Sep 2026. Eight bars: no RV, no Other. These are
# chart read-offs, not data — hence FIGURE_TOLERANCE (spec § 8 item 6).
FIGURE_4_BARS = {
    "JJR": 0.037,
    "JJC": 0.0015,
    "SDA": 0.028,
    "Planned": 0.044,
    "RUAC": 0.021,
    "UAC": 0.017,
    "UV": 0.038,
    "Industrial": 0.038,
}
FIGURE_TOLERANCE = 0.002
# Fewer than this many matched bars for every candidate means the figure was
# not produced from these columns as-is — an escalation, not a guess.
MATCH_FLOOR = 6

CANDIDATE_COLUMNS = ("unnorm_psi", "norm_psi")
BASELINE_FILES = {
    "popsize": "delhi_psi_bbox_popsize2020_norv_12Sep2021.csv",
    "popdensity": "delhi_psi_bbox_popdensity2020_norv_12Sep2021.csv",
}
# The same two denominators from the refactored code-2025 run (--verify-dir);
# `popsize` is what that profile calls `pop`.
VERIFY_FILES = {
    "popsize": "delhi_psi_code-2025_pop_2020.csv",
    "popdensity": "delhi_psi_code-2025_popdensity_2020.csv",
}
BASELINE_SUBDIR = "psi_2020_results"


def type_means(frame, *, column, type_col=TYPE_COL):
    """Mean of `column` per settlement type, ordered by type name."""
    if column not in frame.columns:
        raise KeyError(f"{column!r} is not a column of this file; it has "
                       f"{sorted(frame.columns)[:20]}")
    grouped = frame.groupby(type_col)[column].mean().sort_index()
    return {str(name): float(value) for name, value in grouped.items()}


def score_candidate(means, *, bars=FIGURE_4_BARS, tolerance=FIGURE_TOLERANCE):
    """(bars matched within `tolerance`, max absolute gap over all eight).

    A figure type absent from the file is unmatched with an INFINITE gap: a
    column that does not even carry the type cannot be what the figure was
    drawn from.
    """
    matched, maxgap = 0, 0.0
    for name, bar in bars.items():
        if name not in means:
            maxgap = float("inf")
            continue
        gap = abs(means[name] - bar)
        maxgap = max(maxgap, gap)
        if gap <= tolerance:
            matched += 1
    return matched, maxgap


def score_candidates(frames, *, columns=CANDIDATE_COLUMNS, type_col=TYPE_COL,
                     bars=FIGURE_4_BARS, tolerance=FIGURE_TOLERANCE):
    """The block: every candidate's per-type means, its score, and the best
    candidate — most bars matched, ties broken by the smaller max gap and
    then by name, so the answer is deterministic."""
    report, scores = {}, {}
    for column in columns:
        for denom, frame in frames.items():
            means = type_means(frame, column=column, type_col=type_col)
            for name, value in means.items():
                report[f"mean_{column}_{denom}_{name}"] = f"{value:.6g}"
            matched, maxgap = score_candidate(means, bars=bars,
                                              tolerance=tolerance)
            report[f"matched_{column}_{denom}"] = matched
            report[f"maxgap_{column}_{denom}"] = f"{maxgap:.4f}"
            scores[f"{column}_{denom}"] = (matched, maxgap)
    report["best_candidate"] = min(
        scores, key=lambda name: (-scores[name][0], scores[name][1], name))
    return report


def cross_check(baseline, verify, *, columns=CANDIDATE_COLUMNS,
                type_col=TYPE_COL, atol=1e-9):
    """The refactored `code-2025` run must give the same per-type means as
    the July 2025 baseline — the cheap proof that this comparison would come
    out the same on the pipeline as it stands today."""
    out = {}
    for column in columns:
        for denom, frame in baseline.items():
            left = type_means(frame, column=column, type_col=type_col)
            right = type_means(verify[denom], column=column, type_col=type_col)
            if set(left) != set(right):
                raise ValueError(
                    f"{column}/{denom}: baseline types {sorted(left)} != "
                    f"verify types {sorted(right)}")
            worst = max(abs(left[name] - right[name]) for name in left)
            if worst > atol:
                raise ValueError(
                    f"{column}/{denom}: the code-2025 run differs from the "
                    f"July 2025 baseline by {worst:.3e} (> {atol:.0e})")
            out[f"verify_maxdiff_{column}_{denom}"] = f"{worst:.1e}"
    return out


def measure(baseline_dir, *, verify_dir=None):
    """The whole report, as an ordered {key: value} mapping (one block)."""
    baseline = {denom: pd.read_csv(Path(baseline_dir) / name)
                for denom, name in BASELINE_FILES.items()}
    report = score_candidates(baseline)
    if verify_dir:
        verify = {denom: pd.read_csv(Path(verify_dir) / name)
                  for denom, name in VERIFY_FILES.items()}
        report.update(cross_check(baseline, verify))
        # keep the answer last, whatever else was appended
        report["best_candidate"] = report.pop("best_candidate")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that locates the data root "
                             "(default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch; this script writes nothing, but the "
                             "flag is refused inside --data-dir like every "
                             "other measurement script's")
    parser.add_argument("--baseline-dir", default=None,
                        help=f"the July 2025 outputs; default "
                             f"<data-dir>/{BASELINE_SUBDIR}")
    parser.add_argument("--verify-dir", default=None,
                        help="a complete code-2025 run, for the cross-check")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_psi_columns_")
    baseline_dir = (Path(args.baseline_dir).expanduser() if args.baseline_dir
                    else cfg.paths.data_dir / BASELINE_SUBDIR)
    verify_dir = Path(args.verify_dir).expanduser() if args.verify_dir else None

    print(f"baseline-dir: {baseline_dir}")
    print(f"verify-dir: {verify_dir}")
    print(f"work-dir: {work_dir}")
    report = measure(baseline_dir, verify_dir=verify_dir)
    print(render(report))
    best = report["best_candidate"]
    if report[f"matched_{best}"] < MATCH_FLOOR:
        print(f"WARNING: the best candidate ({best}) matches only "
              f"{report[f'matched_{best}']} of {len(FIGURE_4_BARS)} bars — "
              "the figure was not produced from these columns as-is "
              "(spec § 2.4: escalate, do not guess)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/test_measure_psi_columns.py -rs`
Expected: PASS, with the three doc tests SKIPPED for "carries no measured block yet".

- [ ] **Step 5: Write the document skeleton with the figure table**

Create `docs/data/psi_columns.md`:

```markdown
# Which PSI column, and which denominator, do the figures report?

Raj does not know which column the April 2026 draft's figures were drawn
from (decision log
`docs/decisions/2026-08-28-raj-methodology-decisions.md` §§ 7–8, DEL-52); the
memo's fallback was that Bob measures it and Raj confirms. This document is
that measurement. `scripts/measure_psi_columns.py` reads the two July 2025
baseline outputs — the files that produced the figures — computes the mean
per `USO_FINAL` type for each of the four candidates and scores each against
the figure's bars. `tests/test_measure_psi_columns.py` re-runs it and
compares the block (it skips when the data is not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

## What the figure shows

Figure 4, "Mean public service index by settlement" (p. 40 of the April 2026
draft PDF), is a bar chart of the mean PSI per settlement type with 95 %
whiskers. Its y-axis is labelled **"Mean Public Services Index (per person
per square kilometer)"** — a population-DENSITY denominator — and footnote 12
says the average "rarely exceeds 0.05". It has eight bars: no RV, no Other.

Bar values read off the chart on 5 Sep 2026. **They are read-offs, not data**
(no figure data file exists in the repo or the data folder — checked the same
day), so the script scores a match within ±0.002:

| type | bar (≈) | type | bar (≈) |
|---|---|---|---|
| JJR | 0.037 | RUAC | 0.021 |
| JJC | 0.0015 | UAC | 0.017 |
| SDA | 0.028 | UV | 0.038 |
| Planned | 0.044 | Industrial | 0.038 |

## The four candidates

`unnorm_psi` is Eq. 1 as the methods write it: the mean of the min-maxed
per-service indices. `norm_psi` is that column min-maxed a second time, which
stretches it to [0, 1] — a step the methods never mention. Each is available
under two denominators: `popsize` (population) and `popdensity`
(population / area). The block below names them `<column>_<denom>`; the
`matched_*` keys count bars hit within ±0.002 and the `maxgap_*` keys give
the worst absolute miss.

`verify_maxdiff_*` keys are the cross-check: the same per-type means computed
from the refactored `code-2025` run must equal the baseline's to 1e-9, or the
script fails rather than reporting.

## What follows from the answer

**Two independent axes.** The finding names a COLUMN and a DENOMINATOR, and
they are answered separately: `best_candidate` is a pair, and either half can
land where Bob's proposed default did not. One paragraph each, below. Neither
paragraph decides anything — the decision is Raj's; this document supplies the
numbers he asked for (decision log §§ 7–8).

### The column axis — `unnorm_psi` or `norm_psi`

- **If `unnorm_psi` matches:** the paper already reports Eq. 1 as the methods
  write it. `second_normalization: false` costs nothing and removes a column
  the methods never mention. Bob's recommendation stands.
- **If `norm_psi` matches and `unnorm_psi` does not: Bob's proposed default of
  `second_normalization: false` is withdrawn.** The paper's headline figure
  reports the SECOND-normalised column, so switching it off would silently
  move every bar in Figure 4. The real choice for Raj, then, is: keep
  `norm_psi` as the reported PSI and add the second min-max to the methods —
  one sentence after Eq. 1, "the mean is then min-max scaled across
  settlements" — or switch the figures to Eq. 1 as written and let every bar
  move. Both columns stay in the config either way; this is a question about
  what the paper reports, not about what the pipeline can compute.

  *Expected outcome, stated before the run so the write-up cannot be
  back-fitted:* the plan-review round of 5 Sep 2026 already ran this
  comparison against the same baseline files and found norm_psi under
  popdensity matching 8 of the 8 bars, max gap 0.0006, against 1 of 8 for
  unnorm_psi. The script's job is to make that reproducible and drift-tested,
  not to discover it.

### The denominator axis — `popsize` or `popdensity`

- **If the popdensity denominator matches** (the y-axis label says it will):
  **Bob's proposed default of dropping popdensity from the reported results is
  withdrawn** — the paper's headline figure is the popdensity variant. The
  real choice for Raj: keep popdensity as the reported denominator and add its
  equation to the methods (Eq. 3 with Population_i / Area_i), or switch the
  figures to the per-population Eq. 3 the manuscript prints. Both denominators
  stay in the config either way.
- **If popsize matches instead:** the manuscript's Eq. 3 and its figures agree
  and Bob's proposed default stands unchanged.

### If nothing matches

If no candidate matches at least 6 of the 8 bars, this document prints all
four candidate tables in full and the finding is "the figure was not produced
from these columns as-is". That is an escalation to the owner (spec § 7), not
a guess.
```

*(The provenance bullet list, the fenced block and the finding sentence are
written by the run step — they are the numbers. The consequence paragraphs
above are written HERE, before the run, so the write-up cannot be force-fitted
to whichever branch the data lands in; the run step keeps the branch that
fired, deletes the others, and fills in the measured N and G.)*

- [ ] **Step 6: Run the test file again**

Run: `uv run pytest -q -W error tests/test_measure_psi_columns.py -rs`
Expected: PASS with the same three skips.

- [ ] **Step 7: Run the full suite in the FOREGROUND**

Run: `uv run pytest -q -W error` (about 6.5 minutes)
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add scripts/measure_psi_columns.py tests/test_measure_psi_columns.py \
        docs/data/psi_columns.md
git commit -m "$(cat <<'EOF'
feat(measure): score the PSI column/denominator candidates against Figure 4 (DEL-52)

Mean per USO_FINAL type for {unnorm_psi, norm_psi} x {popsize, popdensity} in
the two July 2025 baseline outputs, scored against the eight Figure 4 bars
read off the April 2026 draft at +/-0.002, with a deterministic best
candidate and an optional 1e-9 cross-check against the refactored code-2025
run. The measured block and the finding are written by the run step.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
EOF
)"
```

---

### Task 5: DEL-49 — `scripts/measure_roads_access.py` + `docs/data/roads_access.md`

**Files:**
- Create: `scripts/measure_roads_access.py`
- Create: `tests/test_measure_roads_access.py`
- Create: `docs/data/roads_access.md`

**Interfaces:**
- Consumes: `scripts._measure_common.{load_settlements, render, resolve_work_dir}` and, in the test, `parse_block` (Task 1); `tests.test_measure_common.{DATA_DIR, MEASURE_CACHE, needs_measure_cache, assert_prose_numbers_come_from_the_blocks}` (Task 1); `delhi_psi.neighbors.adjacency`; `delhi_psi.index.road_lengths` semantics; `delhi_psi.pipeline.{compute, methodology_stamp, output_basename, ID_COL, TYPE_COL}`; `delhi_psi.config.{PROFILES_DIR, load_config}`; and in the test `tests.test_cli.data_dir` (the Oraculum data-dir fixture) with `tests.oraculum_fixtures.oracle_profile_path` — the existing end-to-end machinery, reused rather than rebuilt.
- Produces:
  - `scripts.measure_roads_access.REPORTED_TYPES / DROPPED_TYPES / ALL_TYPES / DENOMINATORS / OWN_ONLY_PROFILE / ROAD_LENGTH_COL / MISMATCH_SAMPLE`
  - `scripts.measure_roads_access.road_inside_ids(gdf, roads, *, id_col=ID_COL) -> set[str]`
  - `scripts.measure_roads_access.assert_road_inside_matches_output(inside_ids, verify_csv_path, id_col) -> None` (raises `ValueError` on any mismatch)
  - `scripts.measure_roads_access.measure_access(gdf, roads, *, id_col=ID_COL, type_col=TYPE_COL, types=ALL_TYPES, inside=None) -> dict`
  - `scripts.measure_roads_access.base_profile_path(base) -> Path`
  - `scripts.measure_roads_access.derived_profile(base, work_dir, *, profile_name=OWN_ONLY_PROFILE) -> Path`
  - `scripts.measure_roads_access.stage_artifacts(verify_dir, run_dir, *, source_name, artifact_name) -> Path`
  - `scripts.measure_roads_access.measure_effect(decayed, own, *, denom, id_col=ID_COL, type_col=TYPE_COL, types=REPORTED_TYPES) -> dict`
  - `scripts.measure_roads_access.measure(cfg, work_dir, *, base, verify_dir) -> {"access": dict, "one_factor": dict}`
  - `scripts.measure_roads_access.main(argv=None) -> int`
  - block names: `access`, `one_factor`

- [ ] **Step 1: Write the failing test file — part 1, the pure functions**

Create `tests/test_measure_roads_access.py`:

```python
"""JJC road access and the roads one-factor effect (DEL-49, spec § 2.1).

The counting is proven on four squares in a row; the one-factor machinery is
proven END TO END on the Oraculum city, where the reference implementation
already says what `roads: eq4_own_only` must produce. The real run is the run
step's, and the real-data drift test needs DELHI_PSI_MEASURE_CACHE — the
shared warm work dir the run step exports — or it skips, rather than paying
~4.5 min for a cold settlement dedup in every implementer's suite run.
"""
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import yaml
from shapely.geometry import LineString, box

from delhi_psi import cli, pipeline
from delhi_psi.config import PROFILES_DIR
from scripts._measure_common import FENCE, parse_block
from scripts.measure_roads_access import (DENOMINATORS, OWN_ONLY_PROFILE,
                                          REPORTED_TYPES,
                                          assert_road_inside_matches_output,
                                          derived_profile, main,
                                          measure_access, measure_effect,
                                          road_inside_ids, stage_artifacts)
from tests.oraculum_fixtures import oracle_profile_path
from tests.test_cli import data_dir  # noqa: F401 — the Oraculum data dir
from tests.test_measure_common import (DATA_DIR, MEASURE_CACHE,
                                       assert_prose_numbers_come_from_the_blocks,
                                       needs_measure_cache)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "roads_access.md"
VERIFY_DIR = DATA_DIR / "phase3_verify"
REFERENCE_CSV = (REPO / "tests" / "fixtures" / "oraculum"
                 / "expected_values.csv")

# Confirmed from tests/fixtures/oraculum/expected_values.csv on 5 Sep 2026
# (rule=ideal — whose roads knob IS eq4 — scenario=baseline, denom=pop).
# NOTE: 0.0075 and 0.0025 are `road_pcen` values; the memo calls them "the
# ideal roads column". The min-maxed `road_idx` for the same rows is 1.0 and
# 1/3, because under eq4 only A and E own any road at all.
IDEAL_ROAD_PCEN = {"A": 0.0075, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0025,
                   "IND": 0.0}
IDEAL_ROAD_IDX = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 1 / 3,
                  "IND": 0.0}


def four_squares():
    """S1..S4 in a row, each 1 km wide, consecutive squares sharing an edge.
    A road runs INSIDE S1; another meets S3's top edge at exactly one point.
    Types are chosen so both a reported and a dropped type are exercised."""
    settlements = gpd.GeoDataFrame(
        {"USO_AREA_U": ["S1", "S2", "S3", "S4"],
         "USO_FINAL": ["JJC", "JJC", "Planned", "RV"]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
                  box(2000, 0, 3000, 1000), box(3000, 0, 4000, 1000)],
        crs="EPSG:7760")
    roads = gpd.GeoDataFrame(
        {"name": ["inside S1", "corner of S3"]},
        geometry=[LineString([(200, 500), (800, 500)]),
                  LineString([(2500, 1000), (2500, 1500)])],
        crs="EPSG:7760")
    return settlements, roads


# --- block 1: road access ----------------------------------------------
def test_a_road_that_only_touches_the_boundary_is_not_inside():
    """`index.road_lengths` clips the road to the polygon and sums the
    pieces, so a point contact contributes zero length — road_inside must
    mean the same thing, or the block would not describe today's outputs."""
    settlements, roads = four_squares()
    assert road_inside_ids(settlements, roads) == {"S1"}


def test_road_access_on_four_squares_in_a_row():
    settlements, roads = four_squares()
    got = measure_access(settlements, roads)
    assert got["road_inside_JJC"] == 1          # S1
    assert got["road_via_neighbor_JJC"] == 1    # S2 touches S1
    assert got["no_road_JJC"] == 0
    assert got["road_inside_Planned"] == 0
    assert got["no_road_Planned"] == 1          # S3: point contact only
    assert got["no_road_RV"] == 1               # S4
    assert got["road_inside_total"] == 1
    assert got["road_via_neighbor_total"] == 1
    assert got["no_road_total"] == 2
    # every reported and dropped type gets a key, even at zero
    assert got["road_inside_SDA"] == 0
    assert got["no_road_Other"] == 0


def test_measure_access_refuses_a_settlement_type_it_has_no_key_for():
    settlements, roads = four_squares()
    settlements.loc[0, "USO_FINAL"] = "Nonesuch"
    with pytest.raises(ValueError, match="Nonesuch"):
        measure_access(settlements, roads)


# --- road_inside <=> road_length > 0 (spec § 2.1 (a), § 3) --------------
def output_csv(tmp_path, ids, lengths, name="verify.csv"):
    """A code-2025-shaped output CSV: one row per REPORTED settlement, with
    the `road_length` column production writes."""
    path = tmp_path / name
    pd.DataFrame({"USO_AREA_U": ids, "road_length": lengths}).to_csv(
        path, index=False)
    return path


def test_the_road_inside_set_matches_the_outputs_road_length_column(tmp_path):
    """The equivalence spec § 2.1 (a) asserts: this script's own geometry
    (`road_inside_ids`) and production's `index.road_lengths` must classify
    the same settlements. S4 is left OUT of the CSV on purpose — the run
    excludes some types (RV), and the comparison is over the ids the CSV
    reports, not over the whole layer."""
    settlements, roads = four_squares()
    inside = road_inside_ids(settlements, roads)
    csv = output_csv(tmp_path, ["S1", "S2", "S3"], [600.0, 0.0, 0.0])
    assert assert_road_inside_matches_output(
        inside, csv, id_col="USO_AREA_U") is None


def test_a_road_inside_mismatch_against_the_output_raises_both_ways(tmp_path):
    """Not a warning. If the two disagree, the access block would misdescribe
    today's outputs, so the script stops. The message names the offenders in
    both directions — inside-but-zero-length and positive-length-but-not-inside
    — because which way it fell tells you which side is wrong."""
    settlements, roads = four_squares()
    inside = road_inside_ids(settlements, roads)          # {"S1"}
    csv = output_csv(tmp_path, ["S1", "S2", "S3"], [0.0, 0.0, 5.0])
    with pytest.raises(ValueError) as exc:
        assert_road_inside_matches_output(inside, csv, id_col="USO_AREA_U")
    message = str(exc.value)
    assert "road_inside" in message and "road_length" in message
    assert "S1" in message and "S3" in message
```

- [ ] **Step 2: Write the failing test file — part 2, the one-factor machinery and the docs**

Append to `tests/test_measure_roads_access.py`:

```python
# --- block 2: the derived profile and the diff -------------------------
def test_the_derived_profile_changes_exactly_one_value(tmp_path):
    """One methodology value changed and two path keys dropped (so the
    per-profile artifact name and --out-dir apply); everything else byte-equal
    after the YAML round trip."""
    path = derived_profile("code-2025", tmp_path)
    got = yaml.safe_load(path.read_text())
    assert path.name == f"{OWN_ONLY_PROFILE}.yaml"
    assert got["methodology"]["roads"] == "eq4_own_only"
    assert got["profile"] == OWN_ONLY_PROFILE
    assert "neighbors_artifact" not in got["paths"]
    assert "out_dir" not in got["paths"]

    expected = yaml.safe_load((PROFILES_DIR / "code-2025.yaml").read_text())
    expected["methodology"]["roads"] = "eq4_own_only"
    expected["profile"] = OWN_ONLY_PROFILE
    expected["paths"].pop("neighbors_artifact")
    assert got == expected


def test_the_methodology_stamp_does_not_carry_roads():
    """The whole one-factor design rests on this: `roads` is applied
    downstream in `compute`, so the proven neighbours artifact stays valid
    and no 11-minute preprocess is needed. If a future stamp change adds it,
    this fails loudly instead of the script silently re-preprocessing."""
    from delhi_psi.config import load_config

    stamp = pipeline.methodology_stamp(load_config("code-2025").methodology)
    assert set(stamp) == {"adjacency", "barrier"}
    assert not any("roads" in block for block in stamp.values())


def test_stage_artifacts_copies_the_artifact_under_the_derived_name(tmp_path):
    verify = tmp_path / "verify"
    verify.mkdir()
    (verify / "colonies_neighbors.joblib").write_bytes(b"artifact")
    target = stage_artifacts(
        verify, tmp_path / "run", source_name="colonies_neighbors.joblib",
        artifact_name=f"colonies_neighbors_{OWN_ONLY_PROFILE}.joblib")
    assert target.name == f"colonies_neighbors_{OWN_ONLY_PROFILE}.joblib"
    assert target.read_bytes() == b"artifact"


def effect_frames():
    """Two four-row outputs: JJC J1 keeps a road of its own, JJC J2 borrowed
    everything and falls to zero, Planned P1 is unchanged, Planned P2 was
    already zero (so it did not FALL to zero)."""
    decayed = pd.DataFrame({
        "USO_AREA_U": ["J1", "J2", "P1", "P2"],
        "USO_FINAL": ["JJC", "JJC", "Planned", "Planned"],
        "road_idx": [0.4, 0.2, 0.8, 0.0],
        "unnorm_psi": [0.10, 0.20, 0.30, 0.40]})
    own = pd.DataFrame({
        "USO_AREA_U": ["J1", "J2", "P1", "P2"],
        "USO_FINAL": ["JJC", "JJC", "Planned", "Planned"],
        "road_idx": [0.4, 0.0, 0.8, 0.0],
        "unnorm_psi": [0.10, 0.15, 0.30, 0.40]})
    return decayed, own


def test_measure_effect_reports_means_and_the_settlements_zeroed():
    decayed, own = effect_frames()
    got = measure_effect(decayed, own, denom="pop")
    assert got["n_pop_JJC"] == 2
    assert got["n_pop_total"] == 4
    assert got["road_idx_decayed_pop_JJC"] == "0.3"
    assert got["road_idx_own_pop_JJC"] == "0.2"
    assert got["psi_decayed_pop_JJC"] == "0.15"
    assert got["psi_own_pop_JJC"] == "0.125"
    assert got["road_idx_zeroed_pop_JJC"] == 1       # J2 only
    assert got["road_idx_zeroed_pop_Planned"] == 0   # P2 was already zero
    assert got["road_idx_zeroed_pop_total"] == 1


def test_measure_effect_refuses_a_partial_join():
    decayed, own = effect_frames()
    with pytest.raises(ValueError, match="different settlements"):
        measure_effect(decayed, own.iloc[:3], denom="pop")


# --- the Oraculum one-factor proof -------------------------------------
def test_the_pinned_ideal_road_values_are_the_reference_implementations():
    """Confirmed against the fixture rather than against the memo: the memo's
    "A 0.0075, E 0.0025, others 0" are `road_pcen`, not `road_idx`."""
    expected = pd.read_csv(REFERENCE_CSV)
    expected = expected[(expected["rule"] == "ideal")
                        & (expected["scenario"] == "baseline")
                        & (expected["denom"] == "pop")].pivot(
        index="settlement", columns="metric", values="value")
    for sid, value in IDEAL_ROAD_PCEN.items():
        assert expected.loc[sid, "road_pcen"] == pytest.approx(value, abs=1e-12)
    for sid, value in IDEAL_ROAD_IDX.items():
        assert expected.loc[sid, "road_idx"] == pytest.approx(value, abs=1e-12)


def test_own_only_reproduces_the_reference_ideal_roads_column(data_dir,
                                                              tmp_path):
    """The one-factor machinery — this script's derived profile driven through
    the real CLI — on the city where the answer is known by hand.

    BOTH roads columns are pinned, against
    `tests/fixtures/oraculum/expected_values.csv` rows with **rule=`ideal`,
    scenario=`baseline`, denom=`pop`** (spec § 4): `road_pcen` A 0.0075,
    E 0.0025, others 0; `road_idx` A 1.0, E 1/3, others 0. The memo § 3 quoted
    the PCEN values as if they were the index — pinning both is what stops
    that confusion recurring.

    The base is the DERIVED oracle profile (the shipped `uso-10` mapping does
    not cover this city's vocabulary). The reference's `ideal` rule differs
    from `code-2025` in more than roads, but under `eq4_own_only` road_pcen is
    own length / own denominator — per-row and independent of adjacency,
    barriers and who else is in the frame — and road_idx is its min-max, whose
    min (0) and max (A's 0.0075) are the same in both frames. So these are the
    right values to expect.
    """
    base = oracle_profile_path("code-2025", tmp_path)
    run_dir = tmp_path / "own_only"
    profile = derived_profile(base, run_dir)
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0
    assert (run_dir / f"colonies_neighbors_{OWN_ONLY_PROFILE}.joblib").exists()
    assert cli.main(["compute", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0

    got = pd.read_csv(
        run_dir / f"delhi_psi_{OWN_ONLY_PROFILE}_pop_2020.csv"
    ).set_index("USO_AREA_U")
    assert set(got.index) == set(IDEAL_ROAD_PCEN)
    for sid, value in IDEAL_ROAD_PCEN.items():
        assert got.loc[sid, "road_pcen"] == pytest.approx(value, abs=1e-9), sid
    for sid, value in IDEAL_ROAD_IDX.items():
        assert got.loc[sid, "road_idx"] == pytest.approx(value, abs=1e-9), sid


# --- the committed document --------------------------------------------
def committed_blocks():
    if not DOC.exists() or FENCE not in DOC.read_text():
        pytest.skip(f"{DOC} carries no measured block yet — the run step "
                    "pastes it")
    text = DOC.read_text()
    return (parse_block(text, name="access"),
            parse_block(text, name="one_factor"))


def test_the_doc_blocks_have_every_required_key():
    access, one_factor = committed_blocks()
    for kind in ("road_inside", "road_via_neighbor", "no_road"):
        for name in (*REPORTED_TYPES, "RV", "Industrial", "Other", "total"):
            assert access[f"{kind}_{name}"].isdigit(), f"{kind}_{name}"
    for denom in DENOMINATORS:
        for name in (*REPORTED_TYPES, "total"):
            assert one_factor[f"n_{denom}_{name}"].isdigit()
            for metric in ("road_idx_decayed", "road_idx_own", "psi_decayed",
                           "psi_own"):
                float(one_factor[f"{metric}_{denom}_{name}"])
            assert one_factor[f"road_idx_zeroed_{denom}_{name}"].isdigit()


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, committed_blocks())


@needs_measure_cache
def test_a_fresh_run_reproduces_the_committed_blocks():
    """The real-data drift check, and with it the real-layer exercise of
    `assert_road_inside_matches_output`: the script raises before printing if
    `road_inside` and `road_length > 0` disagree, so a zero exit code IS that
    assertion passing on all 4,131 reported settlements.

    It costs a settlement dedup, a `touch` adjacency over 4,357 polygons and a
    full `compute`, so it takes its work dir from DELHI_PSI_MEASURE_CACHE (one
    warm cache per machine, shared with the barriers drift test) and SKIPS
    when that is unset — `needs_data` alone would charge every per-task suite
    run on this machine ~4.5 min for a cold dedup. The run step exports it. A
    warm cache is safe here: nothing in this script inspects `geom_type`.
    """
    access, one_factor = committed_blocks()
    proc = subprocess.run(
        [sys.executable, "scripts/measure_roads_access.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR),
         "--work-dir", MEASURE_CACHE],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="access") == access
    assert parse_block(proc.stdout, name="one_factor") == one_factor


def test_main_requires_a_verify_dir(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--config", "code-2025"])
    assert exc.value.code == 2
    assert "--verify-dir" in capsys.readouterr().err
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest -q -W error tests/test_measure_roads_access.py`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'scripts.measure_roads_access'`.

- [ ] **Step 4: Write the script — the access half**

Create `scripts/measure_roads_access.py`:

```python
"""JJC road access, and the one-factor effect of `roads: eq4_own_only`
(DEL-49, spec § 2.1).

Two questions, one script, two blocks:

  access      For every settlement in the deduplicated, reprojected universe:
              is there a major road INSIDE its own polygon, only in a `touch`
              neighbour's, or neither — counted by USO_FINAL type. "Inside"
              is positive intersection length, the same membership
              `delhi_psi.index.road_lengths` uses, so `road_inside` is
              exactly `road_length > 0` in today's outputs. Neighbours use
              `touch` (Raj's ratified rule, DEL-19), not today's bbox: the
              question is about the world after the decision.

  one_factor  `code-2025` with ONE value changed — `methodology.roads:
              eq4_own_only` — run against the SAME neighbours artifact, and
              diffed by type against the proven `code-2025` outputs, which
              are READ from --verify-dir and never recomputed here.

READ-ONLY over --data-dir and --verify-dir. Everything this script writes
goes under --work-dir, which is never inside the data directory.

    uv run python scripts/measure_roads_access.py --config code-2025 \
        --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache
"""

import argparse
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from delhi_psi import geometry, io, neighbors, pipeline
from delhi_psi.config import PROFILES_DIR, load_config
from delhi_psi.pipeline import ID_COL, NBRS_COL, TYPE_COL
from scripts._measure_common import load_settlements, render, resolve_work_dir

# The 28 Aug 2026 decisions: seven reported types, three dropped.
REPORTED_TYPES = ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV", "SDA")
DROPPED_TYPES = ("RV", "Industrial", "Other")
ALL_TYPES = REPORTED_TYPES + DROPPED_TYPES
DENOMINATORS = ("pop", "popdensity")
OWN_ONLY_PROFILE = "roads-own-only"
ROAD_SERVICE = "road"
ROAD_IDX_COL = "road_idx"
ROAD_LENGTH_COL = "road_length"
PSI_COL = "unnorm_psi"
ACCESS_KINDS = ("road_inside", "road_via_neighbor", "no_road")
MISMATCH_SAMPLE = 10


def road_inside_ids(gdf, roads, *, id_col=ID_COL):
    """Ids whose polygon contains a POSITIVE LENGTH of road.

    `index.road_lengths` clips the road layer to the polygon and sums the
    pieces, so a road that merely touches the boundary at a point contributes
    nothing and is not "inside". The sjoin narrows the candidates; the length
    test decides.
    """
    frame = gdf[[id_col, "geometry"]].reset_index(drop=True)
    lines = roads[["geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(frame, lines, how="inner", predicate="intersects")
    ids, geoms, line_geoms = frame[id_col], frame.geometry, lines.geometry
    inside = set()
    for left, right in zip(joined.index, joined["index_right"]):
        settlement = ids.iloc[left]
        if settlement in inside:
            continue
        if geoms.iloc[left].intersection(line_geoms.iloc[right]).length > 0:
            inside.add(settlement)
    return inside


def assert_road_inside_matches_output(inside_ids, verify_csv_path, id_col):
    """`road_inside` must BE `road_length > 0` in today's outputs — spec
    § 2.1 (a) requires this script to assert it, not to claim it.

    `road_inside_ids` reads and reprojects the road layer itself and tests
    intersection length; production's `index.road_lengths` clips the same
    layer inside `compute`. They are two implementations of one membership,
    and if they ever disagree — a different duplicate-dropping rule, a
    different sjoin predicate, a MultiLineString with mixed zero/positive
    parts — the `access` block would quietly stop describing the outputs it
    claims to describe. So this RAISES; it never warns.

    The comparison is over the ids the CSV reports, not over the whole layer:
    the code-2025 run excludes some settlements (RV and friends — 4,131 rows
    out of 4,357), and their absence from the output is not a mismatch. Every
    id the CSV DOES carry must fall on the same side of the line in both.

    Returns None; raises ValueError naming up to MISMATCH_SAMPLE offenders in
    each direction, because which direction it fell tells you which side to
    look at.
    """
    reported = pd.read_csv(verify_csv_path, usecols=[id_col, ROAD_LENGTH_COL])
    reported_ids = set(reported[id_col])
    positive = set(reported.loc[reported[ROAD_LENGTH_COL] > 0, id_col])
    classified = set(inside_ids) & reported_ids

    inside_but_zero = sorted(classified - positive)
    positive_but_outside = sorted(positive - classified)
    if not inside_but_zero and not positive_but_outside:
        return None
    raise ValueError(
        f"road_inside does not match {ROAD_LENGTH_COL} > 0 in "
        f"{verify_csv_path}: {len(inside_but_zero)} classified road_inside "
        f"with {ROAD_LENGTH_COL} == 0 "
        f"{inside_but_zero[:MISMATCH_SAMPLE]}, and "
        f"{len(positive_but_outside)} with {ROAD_LENGTH_COL} > 0 not "
        f"classified road_inside {positive_but_outside[:MISMATCH_SAMPLE]} "
        f"(showing at most {MISMATCH_SAMPLE} of each). The access block "
        "would misdescribe today's outputs; spec § 2.1 (a).")


def measure_access(gdf, roads, *, id_col=ID_COL, type_col=TYPE_COL,
                   types=ALL_TYPES, inside=None):
    """Block `access`: road inside / only via a `touch` neighbour / neither,
    per settlement type and in total.

    A `touch` neighbour may be a DROPPED type; that is correct — dropped
    settlements still lend (28 Aug decision, semantics (a)), and a road in the
    rural village next door is exactly what Raj asked about.

    `inside` lets a caller pass a `road_inside_ids` result it has ALREADY
    computed — `measure` does, because it must run the
    `assert_road_inside_matches_output` check on that same set before using
    it — so the sjoin happens once. Default None recomputes it.
    """
    unknown = sorted(set(gdf[type_col].dropna()) - set(types))
    if unknown:
        raise ValueError(
            f"the layer carries settlement types this report has no key for: "
            f"{unknown}; known types: {list(types)}")
    if inside is None:
        inside = road_inside_ids(gdf, roads, id_col=id_col)
    frame = neighbors.adjacency(gdf, id_col=id_col, neighbor_col=NBRS_COL,
                                rule="touch")
    kinds = {}
    for _, row in frame.iterrows():
        if row[id_col] in inside:
            kind = "road_inside"
        elif any(neighbor in inside for neighbor in row[NBRS_COL]):
            kind = "road_via_neighbor"
        else:
            kind = "no_road"
        kinds[row[id_col]] = (kind, row[type_col])

    report = {}
    for kind in ACCESS_KINDS:
        for name in (*types, "total"):
            report[f"{kind}_{name}"] = sum(
                1 for found, found_type in kinds.values()
                if found == kind and (name == "total" or found_type == name))
    return report
```

- [ ] **Step 5: Write the script — the one-factor half**

Append to `scripts/measure_roads_access.py`:

```python
def base_profile_path(base):
    """A shipped profile NAME or a path to a YAML file — `load_config`'s own
    rule, so `--config code-2025` and `--config some/derived.yaml` both work
    (the Oraculum proof passes a derived path)."""
    candidate = Path(base)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate
    return PROFILES_DIR / f"{base}.yaml"


def derived_profile(base, work_dir, *, profile_name=OWN_ONLY_PROFILE):
    """`base` with ONE methodology value changed: `roads: eq4_own_only`.

    `paths.neighbors_artifact` and `paths.out_dir` are dropped, so the
    per-profile default name applies (colonies_neighbors_<profile>.joblib) and
    --out-dir decides where the run writes. Every other key is byte-equal
    after the YAML round trip. It is written to DISK, not held in memory, so
    the run is reproducible by hand with

        delhi-psi compute
            --config <work-dir>/roads-own-only/roads-own-only.yaml
            --data-dir <data-dir>
            --out-dir <work-dir>/roads-own-only

    — the --out-dir matters: with `paths.out_dir` dropped, a hand run without
    it would not write beside the staged artifact and would not find it.
    """
    raw = yaml.safe_load(base_profile_path(base).read_text())
    raw["profile"] = profile_name
    raw["methodology"]["roads"] = "eq4_own_only"
    paths = dict(raw.get("paths", {}))
    paths.pop("neighbors_artifact", None)
    paths.pop("out_dir", None)
    raw["paths"] = paths
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{profile_name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def stage_artifacts(verify_dir, run_dir, *, source_name, artifact_name):
    """Copy the PROVEN neighbours artifact into the run directory under the
    name the derived profile looks for.

    THE ARTIFACT ALONE (spec § 2.1 (b) item 2). `compute` reads the neighbours
    joblib, the population CSV and the service layers — never a
    `*.dedup.gpkg`. Those caches belong to `preprocess`, which this script
    never runs: the roads switch is applied downstream in `index_frames`, so
    the stored neighbour lists stay valid (spec § 2.1 (c)). Do not add
    dedup-cache copying here, and do not stage dedup caches into the run
    directory from outside either — the only dedup cache this script benefits
    from is the one behind `load_settlements` for the `access` block, which
    lives in --work-dir itself, not in <work-dir>/roads-own-only.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / artifact_name
    shutil.copy2(Path(verify_dir) / source_name, target)
    return target


def _mean(values):
    """Means are pre-formatted, like the pathology block's areas, so the
    drift comparison is exact."""
    return "nan" if values.empty else f"{values.mean():.6g}"


def measure_effect(decayed, own, *, denom, id_col=ID_COL, type_col=TYPE_COL,
                   types=REPORTED_TYPES):
    """Block `one_factor`, one denominator.

    `decayed` is the proven `code-2025` output read from --verify-dir; `own`
    is this script's run with only `roads` changed. Both must report exactly
    the same settlements, or the comparison is not one-factor.
    """
    left = decayed.set_index(id_col)
    right = own.set_index(id_col)
    if set(left.index) != set(right.index):
        raise ValueError(
            f"{denom}: the two runs report different settlements "
            f"(decayed-only {len(set(left.index) - set(right.index))}, "
            f"own-only {len(set(right.index) - set(left.index))})")
    right = right.reindex(left.index)

    selectors = {name: (left[type_col] == name) for name in types}
    selectors["total"] = pd.Series(True, index=left.index)

    report = {}
    for name, rows in selectors.items():
        report[f"n_{denom}_{name}"] = int(rows.sum())
    for name, rows in selectors.items():
        report[f"road_idx_decayed_{denom}_{name}"] = _mean(
            left.loc[rows, ROAD_IDX_COL])
    for name, rows in selectors.items():
        report[f"road_idx_own_{denom}_{name}"] = _mean(
            right.loc[rows, ROAD_IDX_COL])
    for name, rows in selectors.items():
        report[f"psi_decayed_{denom}_{name}"] = _mean(left.loc[rows, PSI_COL])
    for name, rows in selectors.items():
        report[f"psi_own_{denom}_{name}"] = _mean(right.loc[rows, PSI_COL])
    for name, rows in selectors.items():
        # FELL to zero: it had something to lose and lost all of it.
        report[f"road_idx_zeroed_{denom}_{name}"] = int(
            ((right.loc[rows, ROAD_IDX_COL] == 0)
             & (left.loc[rows, ROAD_IDX_COL] > 0)).sum())
    return report


def measure(cfg, work_dir, *, base, verify_dir):
    """Both blocks: {"access": {...}, "one_factor": {...}}."""
    id_col = cfg.layers.settlements.id_col
    type_col = cfg.layers.settlements.type_col
    settlements = load_settlements(cfg, work_dir)
    roads = io.read_layer(cfg.paths.data_dir / cfg.services.line[ROAD_SERVICE])
    # `compute` drops exact-duplicate service rows before counting; do the
    # same here so a duplicated road row cannot change the membership.
    roads = roads.drop_duplicates().reset_index(drop=True)
    roads = geometry.reproject(roads, cfg.crs.epsg)

    inside = road_inside_ids(settlements, roads, id_col=id_col)
    # --verify-dir is required by main(), so on the real run this ALWAYS
    # fires: the block's `road_inside` claim is checked against the column it
    # claims to equal before a single count is written (spec § 2.1 (a)).
    # `road_length` is the same in both denominators' outputs; the `pop` one
    # is read.
    assert_road_inside_matches_output(
        inside,
        Path(verify_dir) / f"{pipeline.output_basename(cfg, 'pop')}.csv",
        id_col)
    access = measure_access(settlements, roads, id_col=id_col,
                            type_col=type_col, inside=inside)

    stamp = pipeline.methodology_stamp(cfg.methodology)
    if any("roads" in block for block in stamp.values()):
        raise SystemExit(
            "pipeline.methodology_stamp now carries `roads`: the neighbours "
            "artifact would have to be rebuilt and this script's one-factor "
            "run cannot reuse --verify-dir's. Re-read spec § 2.1 (b) before "
            "changing anything.")

    run_dir = Path(work_dir) / OWN_ONLY_PROFILE
    profile_path = derived_profile(base, run_dir)
    own_cfg = load_config(profile_path, data_dir=str(cfg.paths.data_dir),
                          out_dir=str(run_dir))
    stage_artifacts(verify_dir, run_dir,
                    source_name=cfg.paths.neighbors_artifact,
                    artifact_name=own_cfg.paths.neighbors_artifact)
    pipeline.compute(own_cfg)

    one_factor = {}
    for denom in DENOMINATORS:
        decayed = pd.read_csv(
            Path(verify_dir) / f"{pipeline.output_basename(cfg, denom)}.csv")
        own = pd.read_csv(
            run_dir / f"{pipeline.output_basename(own_cfg, denom)}.csv")
        one_factor.update(measure_effect(decayed, own, denom=denom,
                                         id_col=id_col, type_col=type_col))
    return {"access": access, "one_factor": one_factor}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch (dedup cache, the derived profile, its "
                             "outputs); default a fresh temporary directory. "
                             "Never under --data-dir.")
    parser.add_argument("--verify-dir", required=True,
                        help="an existing, complete code-2025 run "
                             "(colonies_neighbors.joblib + both output CSVs), "
                             "opened READ-ONLY")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_roads_")
    verify_dir = Path(args.verify_dir).expanduser()

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
    print(f"roads: {cfg.paths.data_dir / cfg.services.line[ROAD_SERVICE]}")
    print(f"verify-dir: {verify_dir}")
    print(f"work-dir: {work_dir}")
    blocks = measure(cfg, work_dir, base=args.config, verify_dir=verify_dir)
    for name, report in blocks.items():
        print(render(report, name=name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q -W error tests/test_measure_roads_access.py -rs`
Expected: PASS — including the Oraculum end-to-end proof and the two `assert_road_inside_matches_output` unit tests; the two committed-document tests SKIP with "carries no measured block yet" and the real-data drift test SKIPS with "set DELHI_PSI_MEASURE_CACHE to run the real-data drift check". **No real-data run happens in this task.**

- [ ] **Step 7: Write the document skeleton, including the reopen threshold**

Create `docs/data/roads_access.md`:

```markdown
# Road access, and what `roads: eq4_own_only` does to today's numbers

Raj ratified Eq. 4 as the manuscript writes it — each colony counts only the
roads inside its own boundary (decision log
`docs/decisions/2026-08-28-raj-methodology-decisions.md` § 2, DEL-22/DEL-49).
The premise on the call was inverted: the July 2025 code DECAYS roads like
clinics, so this is a change from the published numbers, and Raj is owed the
size of it. Two measurements answer that, both produced by
`scripts/measure_roads_access.py`, which reads the layers named by the
`code-2025` profile and writes nothing under the data directory.
`tests/test_measure_roads_access.py` re-runs it and compares both blocks (it
skips when the data is not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

## Block `access` — road access on the layer, no PSI

For every settlement in the deduplicated, reprojected universe (the same
loader the pathology measurement uses, so these counts describe exactly what
the pipeline scores):

- `road_inside_<TYPE>` — the settlement's polygon contains a positive length
  of the major-road layer. This is the membership `delhi_psi.index.road_lengths`
  uses, so it is `road_length > 0` in today's outputs; a road that only
  touches the boundary at a point does not count. That equivalence is not
  asserted here in prose and hoped for: with `--verify-dir` given the script
  compares its own `road_inside` set against the `road_length` column of the
  `code-2025` output CSV and RAISES if they differ, so the block below is
  either a true description of today's outputs or it was never printed.
- `road_via_neighbor_<TYPE>` — no road of its own, but at least one
  **`touch`** neighbour (positive shared border — Raj's ratified rule, DEL-19,
  not today's bbox) has one. A neighbour may be a dropped type: dropped
  settlements still lend (semantics (a)), and a road in the rural village next
  door is exactly what Raj asked about.
- `no_road_<TYPE>` — neither.

Counts are integers; shares are derived in the prose below, so the drift test
stays exact.

## Block `one_factor` — the effect on today's numbers

`code-2025` with ONE value changed, `methodology.roads: eq4_own_only`, run
against the SAME neighbours artifact (the roads formula is applied downstream
in `pipeline.index_frames`, and `pipeline.methodology_stamp` carries adjacency
and barrier only, so no re-`preprocess` is needed — the script asserts that
before it runs). The `decayed` side is READ from the proven `code-2025` run,
never recomputed. Per denominator and reported type: `n`, the mean `road_idx`
and mean `unnorm_psi` under each formula, and `road_idx_zeroed` — how many
settlements had a decayed road index above zero and fall to exactly zero
because everything they had was borrowed.

## When this reopens the decision

Stated before the numbers, so it is a rule and not a reaction. **The roads
decision (own-only) stands unless one of these is true:**

1. the JJC mean `unnorm_psi` falls by more than 20 % relative to its
   `code-2025` value under either denominator; or
2. the ordering of mean `unnorm_psi` between JJC and Planned flips.

Either would mean the switch changes the paper's headline comparison rather
than refining it, and it goes back to Raj as a question instead of a
correction.
```

*(The provenance bullet list, the two fenced blocks and the three-sentence
findings per block are written by the run step.)*

- [ ] **Step 8: Run the test file again**

Run: `uv run pytest -q -W error tests/test_measure_roads_access.py -rs`
Expected: PASS with the same three skips (two "carries no measured block yet", one "set DELHI_PSI_MEASURE_CACHE …").

- [ ] **Step 9: Run the full suite in the FOREGROUND**

Run: `uv run pytest -q -W error` (about 6.5 minutes)
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add scripts/measure_roads_access.py tests/test_measure_roads_access.py \
        docs/data/roads_access.md
git commit -m "$(cat <<'EOF'
feat(measure): JJC road access and the roads one-factor effect (DEL-49)

Block `access`: road inside / only in a touch neighbour / neither, by
settlement type, with "inside" defined as index.road_lengths defines it —
and asserted to be it: with --verify-dir the script raises unless its
road_inside set is exactly the code-2025 output's road_length > 0.
Block `one_factor`: code-2025 with only methodology.roads changed, run in
process against the proven neighbours artifact through a derived YAML in the
work dir, diffed by type against the code-2025 outputs read from --verify-dir.
The machinery is proven end to end on Oraculum against the reference
implementation's ideal roads column; the real numbers come from the run step.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
EOF
)"
```

---

### Task 6: The run step — measure, paste, write the findings

> **This is the controller's task, not an implementer's, and it is the ONLY
> task in this plan that contains placeholders.** Every `[finding from block]`
> below is a sentence whose numbers do not exist until the four commands have
> run. Run the command, read the block, write the sentence from what it says.
> Nothing here is a guess: if a number is missing, re-run, do not invent.

**Files:**
- Modify: `docs/data/layer_pathologies.md` (fresh block with the two DEL-50 keys)
- Modify: `docs/data/barriers.md`, `docs/data/psi_columns.md`, `docs/data/roads_access.md` (provenance list + blocks + findings)
- Modify: `tests/test_layer_pathologies.py` (move `PENDING_KEYS` into `COUNT_KEYS`; restore the `==` set comparison)
- Modify: `docs/decisions/2026-08-28-raj-methodology-decisions.md` (summary table rows 2/7/8; §§ 2, 3, 4, 7, 8; "What goes in the batched reply")
- Modify: `WORKPLAN.md:348-351` (Phase 3 DEL-50/51 item) and `WORKPLAN.md:382-387` (Phase 4 DEL-49/52 item)
- Modify: `CHANGELOG.md:8` (`[Unreleased]`)

**Interfaces:**
- Consumes: everything Tasks 1–5 produced. Produces: no code.

- [ ] **Step 1: Prepare ONE shared warm cache outside the data directory — and leave the pathology run COLD**

The guard refuses any `--work-dir` inside `~/delhi_data`, so the existing
`~/delhi_data/phase3_verify` caches cannot be used in place. Stage the
settlement dedup cache out ONCE, into a single shared directory — it is keyed
on the source shapefile's mtime+size, which has not changed, so everything
that reads the settlement universe reuses it instead of spending ~4.5 minutes
each on the O(n²) dedup:

```bash
mkdir -p ~/measure_work/cache ~/measure_work/psi-columns
cp ~/delhi_data/phase3_verify/settlements.dedup.gpkg \
   ~/delhi_data/phase3_verify/settlements.dedup.stamp ~/measure_work/cache/
export DELHI_PSI_MEASURE_CACHE=~/measure_work/cache
```

Three things about that, all load-bearing:

1. **`measure_layer_pathologies.py` gets NO staged cache — ever.** It is the
   one script that inspects `geom_type`, and a warm GeoPackage cache upcasts
   every Polygon to MultiPolygon (Global Constraints; spec § 3), which turns
   `multipolygons: 556` into `4357` with nothing else moving. It runs with no
   `--cache-dir` at all, on its own fresh temp directory, and pays the cold
   dedup. Do not "optimise" this back.
2. **`inventory_barriers.py` and `measure_roads_access.py` DO share
   `~/measure_work/cache` as their `--work-dir`.** Their predicates —
   `intersects`, intersection length, `touch` adjacency, barrier flags — are
   type-agnostic, so the upcast cannot change any number they print, and the
   shared warm cache saves two cold dedups here and two more in Step 8.
3. **The `roads-own-only` compute is staged by the SCRIPT, not by this step,
   and it stages the neighbours artifact ALONE** (spec § 2.1 (b) item 2).
   `pipeline.compute` never reads a `*.dedup.gpkg` — only `preprocess` does.
   Do not copy dedup caches into `<work-dir>/roads-own-only`; the cache the
   roads script benefits from is the one in `--work-dir` itself, for the
   `access` block's `load_settlements`.

`DELHI_PSI_MEASURE_CACHE` must stay exported for Steps 2, 8 and 9 — Step 8's
two real-data drift tests skip without it. Nothing is written under
`~/delhi_data` at any point in this task.

- [ ] **Step 2: Run the four measurements (spec § 4)**

Run each in the foreground, keep the whole stdout:

```bash
# COLD, deliberately: no --cache-dir. ~4.5 min of dedup, and the only way
# `multipolygons` comes back as 556 rather than 4357 (Step 1, note 1).
uv run python scripts/measure_layer_pathologies.py --config code-2025

uv run python scripts/inventory_barriers.py \
    --config code-2025 --all-candidates --work-dir "$DELHI_PSI_MEASURE_CACHE"

uv run python scripts/measure_psi_columns.py \
    --baseline-dir ~/delhi_data/psi_2020_results \
    --verify-dir ~/delhi_data/phase3_verify \
    --work-dir ~/measure_work/psi-columns

uv run python scripts/measure_roads_access.py \
    --config code-2025 --verify-dir ~/delhi_data/phase3_verify \
    --work-dir "$DELHI_PSI_MEASURE_CACHE"
```

Checks before pasting anything:
- the pathology run's existing key VALUES are unchanged from the committed
  block (`settlements: 4357`, `rectangles: 0`, `multipolygons: 556`,
  `isolated_bbox: 6`, `isolated_touch: 20`, `no_population: 15`,
  `overlapping_pairs: 4069`, the three areas, the six point services). Only
  the two DEL-50 keys are new. If any existing value moved, STOP: something
  upstream changed and this is not a measurement question.
  **`multipolygons: 556` only reproduces on a COLD run.** If it comes back as
  `4357` with every other key unchanged, that is not an upstream change and
  not a regression — it is the warm-cache GeoPackage upcast (Global
  Constraints; spec § 3), which means a `--cache-dir` was passed or a stale
  one was inherited. Re-run with no `--cache-dir` before treating any
  `multipolygons` difference as real.
- `measure_roads_access.py` did not raise `ValueError: road_inside does not
  match road_length > 0`. Its exit 0 IS the spec § 2.1 (a) equivalence
  holding on the real layer — the assertion runs before anything is printed,
  so a printed `access` block is a passed check. If it DID raise, the message
  names the offending ids in both directions: that is a real divergence
  between this script's geometry and `index.road_lengths`, and it is a stop,
  not a number to paste around.
- `measure_psi_columns.py` printed no `WARNING:` line. If it did, no candidate
  matched 6 of 8 bars — that is spec § 7's stop-and-ask outcome: record the
  four candidate tables, write the finding as "the figure was not produced
  from these columns as-is", and escalate to the owner instead of continuing.
- `measure_roads_access.py` exited 0 (a `SystemExit` about the methodology
  stamp means the stamp gained `roads` and the whole one-factor design needs
  re-reading).

- [ ] **Step 3: Paste the blocks into the four documents**

Each document gets, immediately above its block, the provenance list in the
`layer_pathologies.md` shape (the doc tests require all four labels):

```markdown
- **Run date:** 2026-09-05
- **Inputs:** `<the layer / directory paths the script printed>`
- **Commit:** `<git rev-parse --short HEAD>`
- **Command:** `<the exact command from step 2>`
```

- `docs/data/layer_pathologies.md`: replace the whole fenced block with the
  fresh one (existing key values identical, `corner_only_pairs` and
  `corner_only_settlements` now present after `overlapping_pairs`), and update
  the `**Run date:**`/`**Commit:**` lines. Keep `**Layer:**` — that document's
  historic label.
- `docs/data/barriers.md`: the provenance list plus BOTH blocks (`layers`,
  then `attributes`), above the "What this tells us about provenance"
  section.
- `docs/data/psi_columns.md`: the provenance list plus the single block, after
  "The four candidates".
- `docs/data/roads_access.md`: the provenance list plus BOTH blocks
  (`access`, then `one_factor`), each directly under its own section heading.

- [ ] **Step 4: Write the findings prose**

Rule for every sentence below: a number quoted in backticks must be a block
value verbatim (the prose-drift test enforces it); derived shares carry a `%`
sign or no backticks.

`docs/data/layer_pathologies.md`, in the `corner_only_pairs` bullet, append
one sentence: **[finding from block]** — if the count is zero, "the `touch`
rule and the intersection rule pick the same neighbours on this layer, so the
corner question is moot"; otherwise the count, the settlements involved, and
the note that these pairs are neighbours under a 0 km band and not under
`touch`, for Bob to rule on and Raj to hear.

`docs/data/roads_access.md`, three sentences under each block:
- access: **[finding from block]** how many JJCs have a road inside, how many
  only next door, how many neither, with the shares, and the same for Planned
  as the comparison.
- one_factor: **[finding from block]** how far the JJC and Planned mean
  `unnorm_psi` move under own-only, under both denominators; how many
  settlements fall to a zero road index; whether the JJC-vs-Planned ordering
  holds.
- then, against the threshold already written in "When this reopens the
  decision": **[finding from block]** state explicitly that the decision
  stands, or that criterion 1 or 2 fired and the question goes back to Raj.

`docs/data/barriers.md`: one paragraph under the blocks — **[finding from
block]** the feature counts and lengths per layer, how many settlements each
flags and how many under any layer, whether the `--all-candidates` copies
differ from the clipped ones (and if so, how), and whether the sidecar dates
agree. The provenance prose above it is already written and does not change
unless the run contradicts it.

`docs/data/psi_columns.md`: the finding section — **[finding from block]**
"matched N of 8 bars within 0.002, max gap G" for the best candidate, named
as `<column>` under `<denominator>`, plus the `verify_maxdiff_*` line as the
cross-check.

Then the consequences. **Task 4 already wrote the consequence paragraphs into
the skeleton, both axes, all branches** — this step does not invent them. It
keeps the branch that fired, deletes the branches that did not, and fills in
the measured N and G. Read `best_candidate` and the `matched_*` keys, then:

**The two axes are independent, and the doc carries ONE PARAGRAPH FOR EACH.**
`best_candidate` is a `<column>_<denom>` pair; the column question
(`unnorm_psi` vs `norm_psi`) and the denominator question (`popsize` vs
`popdensity`) are answered separately and can land on opposite sides of Bob's
proposed defaults. Do not collapse them into one sentence, and do not let the
answer on one axis decide the other.

*Column axis:*
1. if `unnorm_psi` matches: the paper already reports Eq. 1 as written, so
   `second_normalization: false` costs nothing and removes a column the
   methods never mention — Bob's recommendation stands;
2. **if `norm_psi` matches and `unnorm_psi` does not: Bob's proposed default
   of `second_normalization: false` is WITHDRAWN** (spec § 2.4 item 2) — the
   paper's headline figure reports the second-normalised column, so switching
   it off would silently move every bar in Figure 4. State the real choice for
   Raj, symmetric to the popdensity reversal: keep `norm_psi` as the reported
   PSI and add the second min-max to the methods (one sentence after Eq. 1,
   "the mean is then min-max scaled across settlements"), or switch the
   figures to Eq. 1 as written and let every bar move. Both columns stay in
   the config either way. **This is not a stop-and-ask**: spec § 8 item 8's
   reasoning — a recommendation the paper's own output contradicts is not a
   recommendation — is what authorises writing the reversal, exactly as it
   authorises the popdensity one. The only stop-and-ask outcome here is
   branch 5.

   *Expect this branch.* The plan-review round of 5 Sep 2026 already ran this
   comparison against the same baseline files and found **norm_psi ×
   popdensity matching 8 of 8 bars, max gap 0.0006**, against 1 of 8 for
   unnorm_psi × popdensity and 0 of 8 for both popsize candidates. So
   `best_candidate` is expected to be `norm_psi_popdensity` with
   `matched_norm_psi_popdensity: 8` — not an escalation, and not branch 1.
   If the run says something else, the disagreement itself is the finding:
   re-run before writing anything, and if it persists, stop and ask.

*Denominator axis:*
3. if the popdensity denominator matches: **Bob's proposed default of dropping
   popdensity from the reported results is withdrawn**, because the paper's
   headline figure is the popdensity variant; state the real choice for Raj —
   keep popdensity as the reported denominator and add its equation to the
   methods (Eq. 3 with Population_i/Area_i), or switch the figures to the
   per-population Eq. 3 the manuscript prints. Both denominators stay in the
   config either way;
4. if popsize matches instead: the manuscript's Eq. 3 and its figures agree,
   and Bob's proposed default stands unchanged.

*Neither axis:*
5. if no candidate matched at least 6 of 8: print all four candidate tables,
   state "the figure was not produced from these columns as-is", and escalate
   to the owner (spec § 7). This is the one DEL-52 outcome that stops the run.

- [ ] **Step 5: Retire the pending-key tolerance**

In `tests/test_layer_pathologies.py`: move `"corner_only_pairs"` and
`"corner_only_settlements"` from `PENDING_KEYS` into `COUNT_KEYS`, delete
`PENDING_KEYS` and the `if key in committed` loop, restore the exact key-set
assertion, and restore `assert set(measured) == set(committed)` in
`test_a_fresh_run_reproduces_the_committed_counts`. The doc now carries the
keys, so the relaxations Task 2 introduced are no longer honest.

- [ ] **Step 6: Update the decision log**

In `docs/decisions/2026-08-28-raj-methodology-decisions.md`:
- **Summary table:** row 2 (roads) gains the measured effect in one clause;
  rows 7 (`norm_psi`) and 8 (popdensity) change from *pending* to the measured
  answer with the DEL-52 ticket marked ✓.
- **§ 2 Roads:** a new paragraph "**Measured (5 Sep 2026, DEL-49):**"
  — **[finding from block]** the access split for JJC and Planned, the
  one-factor movement, and the explicit sentence that the decision stands (or
  the criterion that fired), pointing at `docs/data/roads_access.md`.
- **§ 3 Adjacency:** replace "Open empirical sub-question: pairs touching only
  at a corner point — Bob to check the real layer" with the measured answer —
  **[finding from block]** the count, and what it means for `touch` versus a
  0 km band, pointing at `docs/data/layer_pathologies.md`.
- **§ 4 Barriers:** under "Raj's question (14:30–14:31)", a paragraph
  answering it as far as the evidence allows — **[finding from block]** the
  inventory summary, the "suggests an official source, no claim about the
  agency" sentence, and the open question for Bijoy, pointing at
  `docs/data/barriers.md`.
- **§ 7 `norm_psi`:** replace "pending" with the measured finding and what it
  means for `second_normalization`. If the finding is `norm_psi` — the
  outcome Step 4 says to expect — this section records the **withdrawal** of
  Bob's proposed `second_normalization: false` and the two options for Raj,
  in the same shape as § 8's popdensity reversal below. Withdrawing a
  proposed default the measurement contradicts is what spec § 8 item 8
  commits to; it is not a methodology change and not a stop-and-ask.
- **§ 8 Popdensity:** replace Bob's proposed default with either its
  confirmation or its **withdrawal** and the two options for Raj (spec § 8
  item 8 commits to writing the reversal if the data says so).
- **"What goes in the batched reply":** rewrite items 1, 4, 7 and 8 with the
  numbers in, and drop the "once DEL-49 has run" / "when DEL-52 has run"
  hedges. Item 8's corner-only clause takes the measured count.

- [ ] **Step 7: Update WORKPLAN.md and CHANGELOG.md**

- `WORKPLAN.md:348-351` (Phase 3, "Pre-recalculation measurements … DEL-50;
  DEL-51"): tick `- [x]` and append two one-line findings — **[finding from
  block]** corner-only pairs, barrier provenance — each naming its
  `docs/data/` file.
- `WORKPLAN.md:382-387` (Phase 4, "Bob: pre-recalculation measurements …
  DEL-49 … DEL-52"): tick `- [x]` and append two one-line findings —
  **[finding from block]** the roads effect and the PSI column/denominator —
  each naming its `docs/data/` file. If DEL-52 withdrew the popdensity
  default, the DEL-31 bullet's "`outputs.denominators` per DEL-52" line gains
  the measured answer; if it withdrew the `second_normalization: false`
  default (the expected `norm_psi` outcome), that bullet's
  `second_normalization` line gains it too — both axes, separately.
- `CHANGELOG.md`, top of `[Unreleased]`: one entry for this cycle — the four
  scripts, the shared `_measure_common`, the four `docs/data/` documents, the
  new pathology keys, the decision-log updates, and the four findings in one
  clause each. Say explicitly: no `delhi_psi/` behaviour change, no profile
  change, no fixture change.

- [ ] **Step 8: Prove the doc-drift tests are live — IN THE BACKGROUND, to a log**

This is the one run in the whole plan that must NOT be a foreground call.
With `DELHI_PSI_MEASURE_CACHE` exported the two real-data drift tests wake up:
`test_inventory_barriers` and `test_measure_roads_access` each drive a full
script over the real layer, and the roads one adds a `touch` adjacency over
4,357 polygons and a whole `pipeline.compute`, on top of
`test_layer_pathologies`'s own COLD dedup fixture (~4.5 min, unavoidable —
its `multipolygons` key demands cold). **Budget 15–20 minutes.** That is past
a foreground command's timeout, so run it detached and read the log:

```bash
mkdir -p ~/measure_work/logs
export DELHI_PSI_MEASURE_CACHE=~/measure_work/cache   # still, if the shell is new
uv run pytest -q -W error -rs \
    tests/test_measure_common.py tests/test_layer_pathologies.py \
    tests/test_inventory_barriers.py tests/test_measure_psi_columns.py \
    tests/test_measure_roads_access.py \
    > ~/measure_work/logs/drift.log 2>&1
```

Run that with the tool's background/detached mode (never `&` in a foreground
call, and never a 10-minute timeout), then wait for it to exit and READ
`~/measure_work/logs/drift.log`. "Never background it; never commit on an
unseen result" from Global Constraints still binds in the sense that matters:
the result is seen — in the log, in full, before anything is committed. What
is forbidden is committing on a run whose output nobody read, not detaching a
20-minute command from a 10-minute timeout.

Expected in the log:
- PASS, and **no** skip whose reason contains "carries no measured block yet"
  — every document now carries its block, so the committed-document tests in
  all four modules run for real.
- **no** skip whose reason contains "set DELHI_PSI_MEASURE_CACHE" — if one
  appears, the variable did not reach pytest (a new shell, or `uv run`
  started before the `export`), the two real-data drift checks did NOT run,
  and the step is not done. Fix the environment and re-run; do not proceed.
- the roads drift test passing is also the real-layer proof of
  `assert_road_inside_matches_output` (Step 2's second check).

- [ ] **Step 9: Run the full suite — also in the BACKGROUND, to a log**

```bash
uv run pytest -q -W error > ~/measure_work/logs/full.log 2>&1
```

Same rule as Step 8, same reason: with `DELHI_PSI_MEASURE_CACHE` exported the
full suite is **15–20 minutes**, not the 6.5 that Tasks 1–5 budget (they run
with the variable unset, so the two real-data drift tests skip). Run it
detached, wait for the exit, read `~/measure_work/logs/full.log`, and commit
only on a green result you have read. If the log shows the
"set DELHI_PSI_MEASURE_CACHE" skips, the suite did not prove what this step
claims — re-export and re-run.

- [ ] **Step 10: Commit**

```bash
git add docs/data/ docs/decisions/2026-08-28-raj-methodology-decisions.md \
        tests/test_layer_pathologies.py WORKPLAN.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(measure): the measured answers to DEL-49, DEL-50, DEL-51 and DEL-52

The four measurements run against the real layers and the July 2025 baseline;
their blocks are carried verbatim in docs/data/ with a finding per ticket.
The decision log's §§ 2, 3, 4, 7 and 8 and the batched-reply checklist now
carry numbers instead of open questions; WORKPLAN's two measurement items are
ticked. No code change: delhi_psi/, the shipped profiles and every fixture are
untouched.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
EOF
)"
```

---

Execution: subagent-driven-development
