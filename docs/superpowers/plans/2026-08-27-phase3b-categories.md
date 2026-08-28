# Phase 3B — Settlement-Category Mapping Layer (DEL-17) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the collapse of settlement source types into analysis categories a config choice — a profile declares a `categories.mapping`, exclusion is written in category names, and every output carries a `category` column — without changing a single number under the shipped profiles.

**Architecture:** One new pure module, `delhi_psi/categories.py` (`apply_mapping`, `categories_of`), which never imports `config`. The mapping is declared inline in each profile (`categories:` block, required from this cycle on) and validated at load — including a duplicate-key YAML loader, because PyYAML silently keeps the last of two identical keys, which is exactly the silence this layer exists to prevent. The mapping is applied in the one shared population/exclusion prelude (`pipeline._population_and_exclusion`), immediately before `excluded_ids`, so both entry points (`compute_frames` in memory, `compute` on paths) get the column and the exclusion semantics from one place. The mapping **value** is threaded through every path, not only the CLI, so a collapsing profile's fixtures record the numbers the CLI actually produces. Both shipped Delhi profiles get the identity `uso-10` mapping, so every fixture and the real-data baseline stay byte-identical.

**Tech Stack:** Python 3.13 / uv, hatchling, geopandas 1.1, shapely 2.1, pandas 3.0, pyproj, joblib, tqdm, PyYAML (runtime), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-phase3b-categories-design.md` (rev 3, approved by owner 2026-08-27 — read it in full first; §§ 1–8 are the authority, § 2 fixes the schema rules, § 4 fixes the proofs and § 5 enumerates the tests). Parent: `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` §§ 3–5.

## Global Constraints

- **Both production fixtures must stay byte-identical.** `tests/fixtures/oraculum/production/code-2025.csv` and `tests/fixtures/oraculum/production/manuscript.csv` must be unchanged at the end of this cycle: `category` is a label, not a metric, and both shipped profiles use the identity scheme. Prove it in every task that touches `delhi_psi/` or `tests/oraculum_fixtures.py` by running

  ```bash
  for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
  git status --porcelain -- tests/fixtures/
  ```

  and requiring the `git status` output to be **empty**. Any modified, deleted **or untracked** file under `tests/fixtures/` is a failure — that is exactly what the CI drift guard checks.
- **The real-data baseline must stay at zero deviation.** `scripts/verify_against_baseline.py --config code-2025` must still report `PASS — new run equivalent to July 2025 baseline within tolerance` with every `max abs deviation` line reading `0.000e+00` on all 23 output columns. (Hand-run in Task 5; `compare_numeric_frames` only ever iterates the *baseline's* numeric columns, so the new string column `category` is invisible to it — an extra column is not a deviation, a missing one is.)
- **No carried-over test may have its EXPECTED VALUE changed.** The suite is **246 tests** green under `uv run pytest -q -W error` today; it only ever grows. The only permitted edits to carried-over tests are the *scaffold wiring* changes the spec § 5 lists, enumerated once here:
  1. `tests/test_config.py`'s `MINIMAL` gains the identity `uso-10` `categories:` block (Task 2).
  2. `tests/test_cli.py`'s `run` helper swaps a shipped profile **name** in `--config` for the derived oracle profile path, and `test_config_by_path_is_equivalent` passes that derived path (Task 3).
  3. `tests/test_oracle_e2e.py` passes `oracle_profile_path("code-2025", tmp_path)` instead of the shipped name (Task 3).
  4. `tests/oraculum_fixtures.py`'s `methodology_with` and `compute_oracle_frame` resolve the profile through `oracle_config` and thread the mapping (Task 3).
  Every assertion and every tolerance in those files stays exactly as it is. Anything beyond this list is a **stop condition** (spec § 8).
- **The oracle city is not Delhi.** Its vocabulary is `Planned, UC, JJC, RV, RUAC, IND`; the shipped Delhi profiles are deliberately **not** padded with `UC`/`IND` — padding them would blunt the unmapped-type guard on real data. Tests that drive the CLI or the oracle helpers on the fixture city use the **derived** test-only profile (`oracle_config` / `oracle_profile_path`). The one exception is the test that proves the guard fires.
- **Unmapped source types are an error**, never a warning and never a fallback category. `categories.default` is rejected at load as reserved.
- `delhi_psi/categories.py` is **pure**: it may import `delhi_psi.validate` (for `ValidationError`) and nothing else from the package. It never imports `delhi_psi.config`.
- Branch: `del-17-categories` (off `origin/main` at 2b5d8ae; HEAD f0e4c85). Never `git add -A`, never `git commit -a` — every commit names its files (review agents may be running: memory note "Review agents: isolate worktree").
- After **every** task: `uv run pytest -q -W error` must be green.
- `~/delhi_data` is read-only. All test IO under pytest `tmp_path`/`tmp_path_factory` or the repo.
- Numeric tolerance: `abs=1e-12` for in-memory comparisons; `abs=1e-9` for anything that has round-tripped through a CSV (the existing e2e's tolerance).
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Stop and report (spec § 8) — do not work around — if: either production fixture changes; the real-data baseline deviates; or a carried-over test needs an expected value changed.

## File Structure

```
delhi_psi/
  categories.py     NEW — apply_mapping, categories_of, CATEGORY_COLUMN   (Task 1)
  __init__.py       docstring mentions the new pure module                (Task 1)
  config.py         CategoriesConfig, _categories, _UniqueKeyLoader,
                    exclusion.types ⊆ categories at load                  (Task 2)
  profiles/code-2025.yaml, manuscript.yaml   gain the identity uso-10 block (Task 2)
  pipeline.py       prelude applies the mapping; compute_frames(mapping=,
                    scheme=); compute stamps attrs + logs                 (Task 3)
tests/
  test_categories.py            NEW                                       (Task 1)
  test_config.py                MINIMAL + the § 5 config cases            (Task 2)
  test_pipeline.py              the in-memory mapping cases               (Task 3)
  oraculum_fixtures.py          oracle_config, oracle_profile_path        (Task 3)
  test_cli.py                   run() rewiring (Task 3); collapse e2e,
                                unmapped exit-1, category columns/attrs   (Task 4)
  test_oracle_e2e.py            derived profile path                      (Task 3)
  test_fixture_invariants.py    id == type pin, vocabulary pin            (Task 4)
docs/
  methodology-config.md         new "Categories" section, both examples   (Task 5)
  data/uso_final_vocabulary.md  NEW                                       (Task 5)
CHANGELOG.md, WORKPLAN.md                                                 (Task 5)
```

## Canonical facts (verified against the repo on 2026-08-27 — do not re-derive)

- `uv run pytest -q -W error` reports **246 passed** at HEAD f0e4c85.
- The Oraculum fixture city, from `tests/fixtures/oraculum/settlements.geojson`, in file order:

  | `USO_AREA_U` | `USO_FINAL` | `area_km2` | `population` |
  |---|---|---|---|
  | A | Planned | 1.0 | 100 |
  | B | UC | 1.0 | 200 |
  | C | JJC | 1.0 | 400 |
  | RV | RV | 0.8 | 100 |
  | D | Planned | 1.0 | 100 |
  | E | RUAC | 2.0 | 300 |
  | IND | IND | 1.0 | 10 |

  Six distinct source types; exactly two settlements (`RV`, `IND`) have an id equal to their type — which is *why* the reference implementation's id-based scenario drops and production's category-based exclusion agree.
- `tests/fixtures/oraculum/expected_values.csv` carries rules `code` and `ideal` × scenarios `baseline`, `excl_rv_only`, `excl_contributing`, `excl_removed`, `excl_ind_removed` × denominators `pop`, `popdensity`. Its metric names are the **reference's**: `clinic_*`, `road_length_km`, `psi_eq1`, `norm_psi`.
- The CLI's config service for the fixture's clinic layer is `health` (the layer is written to `Public Services/Health/Health.shp`), so CLI output columns are `health_count/_pcen/_idx` where the oracle frame and the reference CSV say `clinic_*`. Both existing e2e tests already do this rename; so does Task 4's.
- `delhi_psi.verify.compare_numeric_frames` iterates the **baseline's** numeric columns only. A new string column in the fresh run cannot make it fail; a *missing* baseline column can.
- `category` is 8 characters — under the 10-character ESRI shapefile limit, so it triggers neither the truncation `UserWarning` nor the `Normalized/laundered field name` `RuntimeWarning`.
- PyYAML's default `SafeLoader` keeps the **last** of two identical mapping keys, silently. Verified in this repo's environment; a `construct_mapping` override that raises before delegating fires correctly, and when the YAML is loaded from an **open file handle** the mark carries the file's path (`duplicate key 'c' at /tmp/dupe.yaml:4`).
- The parent 3A spec's required-keys erratum (spec 3B § 6) is **already applied** on this branch — `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` § 3 already reads "**Required keys:** `profile`, the whole `methodology` block and (from cycle 3B, spec `2026-08-27-phase3b-categories-design.md`) the whole `categories` block". Task 5 verifies this with a `grep` and makes no edit.

---

### Task 1: `delhi_psi/categories.py` — the pure mapping module (spec §§ 3, 5)

**Files:**
- Create: `delhi_psi/categories.py`
- Create: `tests/test_categories.py`
- Modify: `delhi_psi/__init__.py` (docstring only)

**Interfaces:**
- Consumes: `delhi_psi.validate.ValidationError` (existing).
- Produces:
  - `delhi_psi.categories.CATEGORY_COLUMN: str` = `"category"`
  - `delhi_psi.categories.categories_of(mapping: Mapping[str, str]) -> frozenset[str]`
  - `delhi_psi.categories.apply_mapping(frame, *, type_col: str, mapping: Mapping[str, str], out_col: str = CATEGORY_COLUMN) -> DataFrame` — returns a **copy** with `out_col` added; raises `delhi_psi.validate.ValidationError` listing every unmapped type with its row count.

- [ ] **Step 1: Write the failing test**

Create `tests/test_categories.py`:

```python
"""The settlement-category mapping layer (spec 3B §§ 3, 5).

Frame in, frame out: no config, no pipeline, no geo fixtures. That the whole
module needs nothing but a mapping is the point — a different city is a
different mapping, not different code.
"""
import pandas as pd
import pytest

from delhi_psi.categories import CATEGORY_COLUMN, apply_mapping, categories_of
from delhi_psi.validate import ValidationError

# The identity over Delhi's 10 USO_FINAL source types (spec 3B § 1).
USO_10 = {"Planned": "Planned", "UAC": "UAC", "JJC": "JJC", "RUAC": "RUAC",
          "RV": "RV", "UV": "UV", "SDA": "SDA", "JJR": "JJR",
          "Industrial": "Industrial", "Other": "Other"}

# The Phase 4 working candidate: five urban categories plus the `non-urban`
# bucket a run then excludes. UV/SDA/Other are Raj's call (DEL-29) and are
# deliberately absent here — under this mapping a layer carrying them errors.
URBAN_5 = {"Planned": "planned", "UAC": "unauthorized",
           "RUAC": "regularized", "JJR": "resettlement", "JJC": "jjc",
           "RV": "non-urban", "Industrial": "non-urban"}


def frame(types):
    return pd.DataFrame({"USO_AREA_U": [f"s{i}" for i in range(len(types))],
                         "USO_FINAL": list(types)})


def test_identity_mapping_copies_the_type_into_the_category_column():
    got = apply_mapping(frame(["Planned", "JJC", "RV"]),
                        type_col="USO_FINAL", mapping=USO_10)
    assert list(got[CATEGORY_COLUMN]) == ["Planned", "JJC", "RV"]
    assert list(got["USO_FINAL"]) == ["Planned", "JJC", "RV"], \
        "the raw source column is kept as-is alongside the category"


def test_many_to_one_collapse_is_the_point():
    got = apply_mapping(frame(["RV", "Industrial", "UAC", "Planned"]),
                        type_col="USO_FINAL", mapping=URBAN_5)
    assert list(got[CATEGORY_COLUMN]) == [
        "non-urban", "non-urban", "unauthorized", "planned"]


def test_a_mapping_broader_than_the_data_is_fine():
    """A scheme may be broader than one city (spec 3B § 2): an entry for a
    type absent from the data is not an error."""
    got = apply_mapping(frame(["Planned"]), type_col="USO_FINAL",
                        mapping=USO_10)
    assert list(got[CATEGORY_COLUMN]) == ["Planned"]


def test_unmapped_types_raise_naming_every_offender_with_counts():
    """One run diagnoses the WHOLE layer: every unmapped type with its row
    count, not just the first one hit."""
    with pytest.raises(ValidationError) as excinfo:
        apply_mapping(frame(["Planned", "UC", "IND", "UC"]),
                      type_col="USO_FINAL", mapping=USO_10)
    message = str(excinfo.value)
    assert "'IND' (1 row)" in message
    assert "'UC' (2 rows)" in message
    assert "categories.mapping" in message
    assert "Planned" in message, "the message lists what the mapping covers"


def test_a_custom_out_column_is_honoured():
    got = apply_mapping(frame(["Planned"]), type_col="USO_FINAL",
                        mapping=USO_10, out_col="cat")
    assert list(got["cat"]) == ["Planned"]
    assert CATEGORY_COLUMN not in got.columns


def test_the_input_frame_is_not_mutated():
    original = frame(["Planned", "JJC"])
    before = original.copy()
    apply_mapping(original, type_col="USO_FINAL", mapping=USO_10)
    pd.testing.assert_frame_equal(original, before)
    assert CATEGORY_COLUMN not in original.columns


def test_a_missing_type_column_is_named():
    with pytest.raises(ValidationError) as excinfo:
        apply_mapping(pd.DataFrame({"USO_AREA_U": ["s0"]}),
                      type_col="USO_FINAL", mapping=USO_10)
    assert "USO_FINAL" in str(excinfo.value)


def test_categories_of_is_the_set_of_category_names():
    assert categories_of(USO_10) == frozenset(USO_10)
    assert categories_of(URBAN_5) == frozenset(
        {"planned", "unauthorized", "regularized", "resettlement", "jjc",
         "non-urban"})
    assert len(URBAN_5) == 7 and len(categories_of(URBAN_5)) == 6, \
        "X:1 collapse: 7 source types produce 6 categories"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_categories.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'delhi_psi.categories'`.

- [ ] **Step 3: Write the module**

Create `delhi_psi/categories.py`:

```python
"""Settlement source type -> analysis category (spec 3B § 3).

Pure: this module never imports `delhi_psi.config`. The pipeline hands it a
plain `{source type: category}` mapping, so running a different city, or a
different collapse of the same city, is a different mapping — not different
code.

An unmapped source type is an ERROR: never a warning, never a fallback
category. Silence is the failure mode this layer exists to prevent.
"""

from delhi_psi.validate import ValidationError

CATEGORY_COLUMN = "category"


def categories_of(mapping):
    """The set of category names `mapping` produces (its values).

    Used by the config loader (`methodology.exclusion.types` must be a
    subset of it), by the run-time repeat of that check in the pipeline
    prelude, and by tests.
    """
    return frozenset(mapping.values())


def apply_mapping(frame, *, type_col, mapping, out_col=CATEGORY_COLUMN):
    """Return a COPY of `frame` with `out_col` = mapping[frame[type_col]].

    The raw source column is kept as-is; the category is an added label.
    Raises ValidationError naming EVERY source type with no mapping entry,
    with its row count, so one run diagnoses the whole layer. A mapping entry
    for a type that is absent from the data is fine — a scheme may be
    broader than one city.
    """
    if type_col not in frame.columns:
        raise ValidationError(
            f"settlement frame has no type column {type_col!r}; columns: "
            f"{sorted(str(column) for column in frame.columns)}")
    # dropna=False: a null type must be reported as unmapped, not dropped
    # from the diagnosis and then silently mapped to NaN.
    counts = frame[type_col].value_counts(dropna=False)
    unmapped = sorted(((value, int(n)) for value, n in counts.items()
                       if value not in mapping),
                      key=lambda item: str(item[0]))
    if unmapped:
        detail = ", ".join(f"{value!r} ({n} row{'' if n == 1 else 's'})"
                           for value, n in unmapped)
        raise ValidationError(
            f"settlement types with no categories.mapping entry: {detail}; "
            f"the mapping covers: {sorted(mapping)}")
    out = frame.copy()
    out[out_col] = frame[type_col].map(dict(mapping))
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_categories.py -q`
Expected: **8 passed**.

- [ ] **Step 5: Mention the module in the package docstring**

In `delhi_psi/__init__.py`, replace:

```python
The math modules
(`geometry`, `neighbors`, `index`) are pure functions with keyword knobs and
never import `config`.
```

with:

```python
The math modules
(`geometry`, `neighbors`, `index`) and the category mapping (`categories`)
are pure functions with keyword knobs and never import `config`.
```

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **254 passed** (246 carried over + 8 new).

- [ ] **Step 7: Commit**

```bash
git add delhi_psi/categories.py delhi_psi/__init__.py tests/test_categories.py
git commit -m "feat(categories): pure source-type -> category mapping (DEL-17)

apply_mapping adds the `category` column and refuses an unmapped source
type, naming every offender with its row count; categories_of gives the
set of category names the config and the pipeline validate against.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: config — the required `categories` block (spec § 2)

**Files:**
- Modify: `delhi_psi/config.py`
- Modify: `delhi_psi/profiles/code-2025.yaml`
- Modify: `delhi_psi/profiles/manuscript.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `delhi_psi.categories.categories_of` (Task 1).
- Produces:
  - `delhi_psi.config.CategoriesConfig` — frozen dataclass with fields `scheme: str`, `mapping: dict`
  - `delhi_psi.config.Config.categories: CategoriesConfig` (new field, declared after `methodology`)
  - `delhi_psi.config._categories(raw: dict) -> CategoriesConfig`
  - `delhi_psi.config._methodology(raw: dict, *, allowed_categories: frozenset[str]) -> MethodologyConfig` (the existing function, gaining one **required keyword-only** parameter)
  - `delhi_psi.config._UniqueKeyLoader` — a `yaml.SafeLoader` subclass that raises `ConfigError` on a repeated mapping key
  - `delhi_psi.config.RESERVED_KEYS["categories.default"]` and `"categories"` in `TOP_LEVEL_KEYS`
- Unchanged: `load_config(profile_or_path, *, data_dir=None, out_dir=None) -> Config`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, replace the `MINIMAL` constant (lines 16–25) with the block below — `MINIMAL` gains the identity `uso-10` `categories` block, which is scaffold-change (1) from the Global Constraints. Nothing else about the constant changes, so the existing `.replace(...)` surgery in the carried-over tests keeps working.

```python
CATEGORIES_BLOCK = """categories:
  scheme: uso-10
  mapping:
    Planned: Planned
    UAC: UAC
    JJC: JJC
    RUAC: RUAC
    RV: RV
    UV: UV
    SDA: SDA
    JJR: JJR
    Industrial: Industrial
    Other: Other
"""

MINIMAL = """profile: minimal
""" + CATEGORIES_BLOCK + """methodology:
  adjacency: {rule: bbox}
  barrier: {rule: global_asymmetric, combine: any}
  decay: {form: inverse_linear, distance_unit: km}
  roads: decayed
  second_normalization: true
  exclusion: {types: [RV], stage: post_neighbors, absent_neighbor: swallowed}
"""

# The 10 Delhi source types, measured 27 Aug 2026 (spec 3B § 1).
DELHI_TYPES = {"Planned", "UAC", "JJC", "RUAC", "RV", "UV", "SDA", "JJR",
               "Industrial", "Other"}
```

Extend the import at the top of the file (line 11) to pull in `categories_of`:

```python
from delhi_psi.categories import categories_of
from delhi_psi.config import (
    ENUMS, ENUM_KEYS, REFERENCE_KNOBS, RESERVED_KEYS, RESERVED_VALUES,
    Config, ConfigError, load_config, shipped_profiles,
)
```

Append to the end of `tests/test_config.py`:

```python
# --- 3B: the categories block (spec 3B §§ 2, 5) ------------------------
def test_categories_block_is_required(tmp_path):
    """A profile states its method completely — `categories` is required
    from cycle 3B on, exactly like `methodology`."""
    without = MINIMAL.replace(CATEGORIES_BLOCK, "")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, without))
    assert "categories" in str(exc.value)


def test_exclusion_type_that_is_not_a_category_is_rejected(tmp_path):
    """`methodology.exclusion.types` holds CATEGORY names — the values of
    the mapping — and the error lists the categories the mapping produces."""
    text = MINIMAL.replace("types: [RV]", "types: [non-urban]")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "methodology.exclusion.types" in message
    assert "non-urban" in message
    for category in sorted(DELHI_TYPES):
        assert category in message, category


def test_duplicate_mapping_key_is_rejected_naming_the_key(tmp_path):
    """PyYAML's default loader keeps the LAST occurrence silently — half
    the layer would be mapped by a rule nobody can see."""
    text = MINIMAL.replace("    JJC: JJC\n", "    JJC: JJC\n    JJC: Other\n")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "duplicate key" in message
    assert "JJC" in message


def test_duplicate_key_anywhere_in_the_profile_is_rejected(tmp_path):
    """The checking loader is not scoped to `categories` — a repeated
    `profile:` is the same silent overwrite."""
    text = MINIMAL + "\nprofile: second\n"
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "duplicate key" in message
    assert "profile" in message


def test_categories_default_is_reserved(tmp_path):
    """A catch-all category is deliberately not offered: it is a KNOWN
    optional key, so it takes the reserved path, never the unknown-key
    path."""
    text = MINIMAL.replace("  scheme: uso-10\n",
                           "  scheme: uso-10\n  default: Other\n")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "unknown key" not in message
    assert message.endswith(RESERVED_KEYS["categories.default"])


@pytest.mark.parametrize("bad,fragment", [
    ("    JJC:\n", "JJC"),          # empty value -> None
    ("    JJC: 7\n", "7"),          # non-string value
])
def test_non_string_mapping_values_are_rejected(tmp_path, bad, fragment):
    text = MINIMAL.replace("    JJC: JJC\n", bad)
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "categories.mapping" in message
    assert fragment in message


@pytest.mark.parametrize("bad", ["  scheme: ''\n", "  scheme:\n"])
def test_scheme_must_be_a_non_empty_string(tmp_path, bad):
    text = MINIMAL.replace("  scheme: uso-10\n", bad)
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    assert "categories.scheme" in str(exc.value)


@pytest.mark.parametrize("profile", ["code-2025", "manuscript"])
def test_shipped_profiles_map_the_ten_delhi_types_to_themselves(profile,
                                                                tmp_path):
    """Both shipped profiles use the identity scheme — which is what makes
    'this cycle changes no numbers' true."""
    cfg = load_config(profile, data_dir=str(tmp_path))
    assert cfg.categories.scheme == "uso-10"
    assert set(cfg.categories.mapping) == DELHI_TYPES
    assert categories_of(cfg.categories.mapping) == frozenset(DELHI_TYPES)
    assert all(source == category
               for source, category in cfg.categories.mapping.items())


def test_shipped_exclusion_types_are_categories(tmp_path):
    """`code-2025` keeps [RV], `manuscript` keeps [] — both valid category
    names under the identity mapping."""
    code = load_config("code-2025", data_dir=str(tmp_path))
    assert code.methodology.exclusion.types == ("RV",)
    assert set(code.methodology.exclusion.types) <= categories_of(
        code.categories.mapping)
    manuscript = load_config("manuscript", data_dir=str(tmp_path))
    assert manuscript.methodology.exclusion.types == ()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL. `test_categories_block_is_required` fails with `Failed: DID NOT RAISE <class 'delhi_psi.config.ConfigError'>` (nothing requires the block yet), and `test_shipped_profiles_map_the_ten_delhi_types_to_themselves` fails with `AttributeError: 'Config' object has no attribute 'categories'`. The carried-over cases in the file stay green — `MINIMAL` with an unknown top-level `categories` key would be rejected, which is *itself* a failure you will see until Step 3 adds the key to `TOP_LEVEL_KEYS`; that is expected.

- [ ] **Step 3: Add the schema to `delhi_psi/config.py`**

Add the import beneath the existing `from delhi_psi.io import ...` line (line 21):

```python
from delhi_psi.categories import categories_of
```

Add to `RESERVED_KEYS` (after the `methodology.exclusion.minmax_universe` entry):

```python
    "categories.default":
        "reserved: a catch-all category is deliberately NOT offered — an "
        "unmapped source type must fail the run, because silence is the "
        "failure mode this layer exists to prevent (spec 3B § 2). Map every "
        "source type explicitly instead.",
```

Add the dataclass next to the other block dataclasses (after `PopulationSpec`, before `LayersConfig`):

```python
@dataclass(frozen=True)
class CategoriesConfig:
    scheme: str
    mapping: dict                  # source type -> category name
```

Add the field to `Config`, immediately after `methodology`:

```python
@dataclass(frozen=True)
class Config:
    profile: str
    methodology: MethodologyConfig
    categories: CategoriesConfig
    crs: CrsConfig
    paths: PathsConfig
    layers: LayersConfig
    services: ServicesConfig
    validate: ValidateConfig
    outputs: OutputsConfig
```

Add `"categories"` to `TOP_LEVEL_KEYS`:

```python
TOP_LEVEL_KEYS = ("profile", "categories", "crs", "paths", "layers",
                  "services", "methodology", "validate", "outputs")
```

Add the checking loader just above `def shipped_profiles():`:

```python
class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated mapping key.

    PyYAML's default keeps the LAST occurrence and says nothing, so a
    profile with two `Planned:` lines under `categories.mapping` would map
    half its layer by a rule nobody can see. That is the exact silence this
    whole layer exists to prevent, so it is an error here — everywhere in
    the profile, not only inside `categories`.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                mark = key_node.start_mark
                raise ConfigError(
                    f"duplicate key {key!r} at {mark.name}:{mark.line + 1}; "
                    "PyYAML would silently keep the last occurrence")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)
```

Add the block parser next to the other `_block` parsers (after `_methodology`):

```python
def _categories(raw):
    """The `categories` block: a scheme name and a source type -> category
    map. Several sources mapping to one category (X:1) is the point; a
    category name equal to a source name is fine (identity)."""
    _reject_unknown(raw, {"scheme", "mapping"}, "categories")

    scheme = _require(raw, "scheme", "categories")
    if not isinstance(scheme, str) or not scheme.strip():
        raise ConfigError(
            f"categories.scheme: {scheme!r} is not allowed; expected a "
            "non-empty string naming the scheme (e.g. 'uso-10')")

    mapping = _require(raw, "mapping", "categories")
    if not isinstance(mapping, dict) or not mapping:
        raise ConfigError(
            "categories.mapping: expected a non-empty map of source type -> "
            f"category, got {mapping!r}")
    for source, category in mapping.items():
        if not isinstance(source, str) or not source:
            raise ConfigError(
                f"categories.mapping: source type {source!r} is not "
                "allowed; expected a non-empty string")
        if not isinstance(category, str) or not category:
            raise ConfigError(
                f"categories.mapping[{source!r}]: {category!r} is not "
                "allowed; expected a non-empty category name")
    return CategoriesConfig(scheme=scheme, mapping=dict(mapping))
```

Give `_methodology` the cross-block check. Change its signature (line 311) to:

```python
def _methodology(raw, *, allowed_categories):
```

and, immediately after the `types` list validation inside the `exclusion` parsing (right after the `if not isinstance(types, list) or not all(...)` block, before `exclusion = ExclusionConfig(`), insert:

```python
    unknown = [item for item in types if item not in allowed_categories]
    if unknown:
        raise ConfigError(
            f"methodology.exclusion.types: {unknown} are not categories "
            "produced by categories.mapping — exclusion is written in "
            "CATEGORY names, not source types; allowed values: "
            f"{sorted(allowed_categories)}")
```

Finally rewrite the body of `load_config` (lines 461–476) so the YAML goes through the checking loader and `categories` is parsed *before* `methodology`:

```python
    path = _profile_path(profile_or_path)
    # An open file handle, not read_text(): PyYAML then puts the file's path
    # into every mark, so _UniqueKeyLoader's message names the file and line.
    with path.open() as handle:
        raw = yaml.load(handle, Loader=_UniqueKeyLoader) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    _reject_unknown(raw, set(TOP_LEVEL_KEYS), "")

    profile = _require(raw, "profile", "")
    # categories first: _methodology's exclusion check needs the category set.
    categories = _categories(_require(raw, "categories", ""))
    return Config(
        profile=profile,
        methodology=_methodology(
            _require(raw, "methodology", ""),
            allowed_categories=categories_of(categories.mapping)),
        categories=categories,
        crs=_crs(raw.get("crs", {})),
        paths=_paths(raw.get("paths", {}), data_dir, out_dir, profile),
        layers=_layers(raw.get("layers", {})),
        services=_services(raw.get("services", {})),
        validate=_validate(raw.get("validate", {})),
        outputs=_outputs(raw.get("outputs", {})))
```

Also update the module docstring's first paragraph (lines 3–4) to match the erratum:

```python
Required keys: `profile`, the whole `methodology` block and (from cycle 3B)
the whole `categories` block — a profile is a complete statement of method,
never inherited. Everything else defaults to the `code-2025` values.
```

- [ ] **Step 4: Add the identity block to both shipped profiles**

In `delhi_psi/profiles/code-2025.yaml`, insert between the `services:` block and `methodology:`:

```yaml
categories:                         # source type -> category (spec 3B 2)
  scheme: uso-10                    # free-form name; stamped into the joblib output
  mapping:                          # IDENTITY: the 10 USO_FINAL types, uncollapsed
    Planned: Planned
    UAC: UAC
    JJC: JJC
    RUAC: RUAC
    RV: RV
    UV: UV
    SDA: SDA
    JJR: JJR
    Industrial: Industrial
    Other: Other
```

and change the `types:` comment in the same file (line 42) from

```yaml
    types: [RV]                     # raw USO_FINAL strings until 3B adds `categories:`
```

to

```yaml
    types: [RV]                     # CATEGORY names (values of categories.mapping)
```

In `delhi_psi/profiles/manuscript.yaml`, insert the same block between `profile: manuscript` and `methodology:`:

```yaml
categories:
  scheme: uso-10
  mapping:                          # identity, as code-2025 (spec 3B 2)
    Planned: Planned
    UAC: UAC
    JJC: JJC
    RUAC: RUAC
    RV: RV
    UV: UV
    SDA: SDA
    JJR: JJR
    Industrial: Industrial
    Other: Other
```

- [ ] **Step 5: Run the config tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS — 12 new cases plus every carried-over case in the file.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **266 passed** (254 + 12).

- [ ] **Step 7: Prove the production fixtures did not move**

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
```

Expected: the `git status` output is **empty**. (Nothing consumes the mapping yet, so this is a plain no-op — but it is also the first point at which the shipped YAMLs changed, so it is worth the ten seconds.)

- [ ] **Step 8: Commit**

```bash
git add delhi_psi/config.py delhi_psi/profiles/code-2025.yaml \
        delhi_psi/profiles/manuscript.yaml tests/test_config.py
git commit -m "feat(config): required categories block, checked YAML loader (DEL-17)

categories.{scheme,mapping} joins the schema as CategoriesConfig; both
shipped profiles gain the identity uso-10 mapping. exclusion.types must be
a category the mapping produces (checked at load). Duplicate YAML keys are
now rejected naming key and line — PyYAML kept the last one silently.
categories.default is reserved: an unmapped type must fail the run.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: pipeline — the mapping reaches every path (spec § 3)

The prelude both entry points share applies the mapping; the mapping value is threaded through the oracle helpers too, so the fixture path runs under the profile's own mapping. **This task also carries scaffold-change items (2), (3) and (4) from the Global Constraints**: once `compute` consumes `cfg.categories.mapping`, the shipped `code-2025` profile (correctly) refuses to run on the oracle city, so the tests that drive the CLI there must move to the derived profile in this same task or the suite cannot be green at its end.

**Files:**
- Modify: `delhi_psi/pipeline.py`
- Modify: `tests/oraculum_fixtures.py`
- Modify: `tests/test_cli.py` (the `run` helper and `test_config_by_path_is_equivalent` only)
- Modify: `tests/test_oracle_e2e.py` (the profile argument only)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `delhi_psi.categories.{CATEGORY_COLUMN, apply_mapping, categories_of}` (Task 1); `delhi_psi.config.{CategoriesConfig, load_config, PROFILES_DIR}` (Task 2).
- Produces:
  - `delhi_psi.pipeline.CATEGORY_COL: str` = `categories.CATEGORY_COLUMN`
  - `delhi_psi.pipeline.excluded_ids(frame, *, types, id_col=ID_COL, category_col=CATEGORY_COL) -> frozenset[str]` — **the keyword `type_col` is renamed to `category_col`**; the only caller is the prelude
  - `delhi_psi.pipeline._population_and_exclusion(frame, population, *, id_col, type_col, population_id_col, population_value_col, missing_population, max_missing_population, exclusion_types, mapping, category_col=CATEGORY_COL) -> (frame, dropped, missing)` — gains the required keyword `mapping`
  - `delhi_psi.pipeline.compute_frames(..., *, ..., mapping: Mapping[str, str] | None = None, scheme: str = "identity")` — returns a frame carrying `result.attrs["categories"] == {"scheme": scheme, "mapping": dict(mapping)}`
  - `tests.oraculum_fixtures.ORACLE_SCHEME: str` = `"oracle-6"`
  - `tests.oraculum_fixtures.ORACLE_VOCABULARY: tuple[str, ...]` = `("Planned", "UC", "JJC", "RV", "RUAC", "IND")`
  - `tests.oraculum_fixtures.oracle_mapping() -> dict[str, str]`
  - `tests.oraculum_fixtures.oracle_config(base: str) -> delhi_psi.config.Config`
  - `tests.oraculum_fixtures.oracle_profile_path(base: str, directory: Path) -> Path`
  - `tests.test_cli.SHIPPED_PROFILES: tuple[str, ...]` = `("code-2025", "manuscript")`
- Unchanged signatures: `methodology_with(profile, *, types=None, stage=None)`, `compute_oracle_frame(profile, *, types, stage, denom)`, `tests.test_cli.run(*args)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# --- 3B: the category mapping on the in-memory path (spec 3B §§ 3, 5) ---
def test_mapping_none_equals_an_explicit_identity_mapping():
    """`mapping=None` builds the identity over the types the city actually
    carries, so every existing in-memory caller keeps its call shape AND
    its numbers."""
    city = load_settlements()
    identity = {t: t for t in city["USO_FINAL"].unique()}
    implicit = _compute(city, None).set_index("USO_AREA_U")
    explicit = compute_frames(
        city, {"canal": load_barriers()}, load_services(), None,
        _methodology(), "pop", mapping=identity,
        scheme="explicit-identity").set_index("USO_AREA_U")

    assert set(implicit.index) == set(explicit.index)
    metrics = [c for c in implicit.columns
               if str(c).endswith(("_count", "_pcen", "_idx", "_length"))
               or c in ("unnorm_psi", "norm_psi", "population", "area_km2")]
    assert "unnorm_psi" in metrics and "clinic_pcen" in metrics
    for column in metrics:
        for sid in implicit.index:
            assert implicit.loc[sid, column] == pytest.approx(
                explicit.loc[sid, column], abs=1e-12), (column, sid)


def test_compute_frames_stamps_the_scheme_and_mapping_on_its_result():
    """pandas drops `attrs` across the merges inside index_frames, so the
    stamp has to go on the frame compute_frames actually returns."""
    mapping = {"Planned": "Planned", "UC": "UC", "JJC": "JJC", "RV": "RV",
               "RUAC": "RUAC", "IND": "IND"}
    result = compute_frames(
        load_settlements(), {"canal": load_barriers()}, load_services(),
        None, _methodology(), "pop", mapping=mapping, scheme="oracle-6")
    assert result.attrs["categories"] == {"scheme": "oracle-6",
                                          "mapping": mapping}


def test_compute_frames_defaults_the_scheme_to_identity():
    result = _compute(load_settlements(), None)
    assert result.attrs["categories"]["scheme"] == "identity"
    assert result.attrs["categories"]["mapping"] == {
        "Planned": "Planned", "UC": "UC", "JJC": "JJC", "RV": "RV",
        "RUAC": "RUAC", "IND": "IND"}


def test_exclusion_type_outside_the_mapping_raises_on_the_in_memory_path():
    """The LOAD-time check cannot see this call: compute_frames and the
    oracle helpers build a MethodologyConfig directly and never pass through
    load_config. Without the run-time repeat, a category the mapping no
    longer produces would exclude nothing — silently."""
    city = load_settlements()
    methodology = methodology_with(PROFILE, types=("non-urban",),
                                   stage="post_neighbors")
    with pytest.raises(validate.ValidationError) as excinfo:
        compute_frames(city, {"canal": load_barriers()}, load_services(),
                       None, methodology, "pop",
                       mapping={t: t for t in city["USO_FINAL"].unique()})
    message = str(excinfo.value)
    assert "non-urban" in message
    assert "RV" in message, "the message lists the categories the mapping produces"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: FAIL — `TypeError: compute_frames() got an unexpected keyword argument 'mapping'` for three of them, and `Failed: DID NOT RAISE` for `test_exclusion_type_outside_the_mapping_raises_on_the_in_memory_path`.

- [ ] **Step 3: Apply the mapping in the shared prelude**

In `delhi_psi/pipeline.py`, extend the package import (line 12):

```python
from delhi_psi import categories, geometry, index, io, neighbors, validate
```

and add a constant next to the other column names (after `CENTROID_COL`, line 20):

```python
CATEGORY_COL = categories.CATEGORY_COLUMN
```

Replace `excluded_ids` (lines 75–79) with:

```python
def excluded_ids(frame, *, types, id_col=ID_COL, category_col=CATEGORY_COL):
    """Ids whose CATEGORY is in `types`.

    From cycle 3B `methodology.exclusion.types` holds CATEGORY names — the
    values of `categories.mapping` — and matches the mapped column the
    prelude has just added, not the raw source-type column.
    """
    if not types:
        return frozenset()
    return frozenset(frame.loc[frame[category_col].isin(list(types)), id_col])
```

Replace the body of `_population_and_exclusion` (lines 176–200) with:

```python
def _population_and_exclusion(frame, population, *, id_col, type_col,
                              population_id_col, population_value_col,
                              missing_population, max_missing_population,
                              exclusion_types, mapping,
                              category_col=CATEGORY_COL):
    """The prelude both entry points share: attach population, apply the
    missing rule, map source types to categories, and work out which ids are
    dropped.

    Returns (frame with population and `category`, dropped ids, ids with no
    population). `dropped` is the CATEGORY exclusion UNION the unpriced
    rows — the same set in `compute_frames` and `compute`, so the rule (and
    its message) lives once. The mapping is applied here, immediately before
    `excluded_ids`, which is why both entry points get the `category` column
    and identical exclusion semantics from one place.
    """
    out, missing = attach_population(
        frame, population, id_col=id_col,
        population_id_col=population_id_col,
        population_value_col=population_value_col)
    if missing and missing_population == "error":
        raise validate.ValidationError(
            f"{len(missing)} settlements have no population row and "
            f"layers.population.missing is 'error': {sorted(missing)[:10]}")
    if max_missing_population is not None:
        validate.check_missing_population(
            len(missing), maximum=max_missing_population)
    out = categories.apply_mapping(out, type_col=type_col, mapping=mapping,
                                   out_col=category_col)
    # The same subset rule load_config enforces, repeated at run time:
    # in-memory callers build a MethodologyConfig directly and never pass
    # through the loader, and a category the mapping does not produce would
    # exclude nothing at all, silently.
    allowed = categories.categories_of(mapping)
    unknown = sorted(item for item in exclusion_types if item not in allowed)
    if unknown:
        raise validate.ValidationError(
            f"methodology.exclusion.types {unknown} are not categories "
            "produced by categories.mapping; it produces: "
            f"{sorted(allowed)}")
    dropped = excluded_ids(out, types=exclusion_types, id_col=id_col,
                           category_col=category_col) | set(missing)
    return out, dropped, missing
```

- [ ] **Step 4: Thread the mapping through `compute_frames`**

In `delhi_psi/pipeline.py`, change `compute_frames`' signature (lines 203–207) to:

```python
def compute_frames(settlements, barriers, services, population, methodology,
                   denominator, *, epsg_code=7760, id_col=ID_COL,
                   type_col=TYPE_COL, population_id_col="uso_area_u",
                   population_value_col="population",
                   missing_population="drop", max_missing_population=None,
                   mapping=None, scheme="identity"):
```

Add to its docstring, after the `missing_population:` paragraph:

```
    mapping: {source type: category}, or None to build the identity over the
        types this city carries. `scheme` names the mapping in the result's
        `attrs` (and, through `compute`, in the joblib output).
```

and replace its body (lines 226–244) with:

```python
    if missing_population not in MISSING_POPULATION:
        raise ValueError(
            f"unknown missing_population {missing_population!r}; allowed "
            f"values: {list(MISSING_POPULATION)}")
    if mapping is None:
        # The identity over the types this city actually carries: existing
        # in-memory callers keep their call shape and their numbers.
        mapping = {t: t for t in settlements[type_col].unique()}
    frame, dropped, _ = _population_and_exclusion(
        settlements, population, id_col=id_col, type_col=type_col,
        population_id_col=population_id_col,
        population_value_col=population_value_col,
        missing_population=missing_population,
        max_missing_population=max_missing_population,
        exclusion_types=methodology.exclusion.types,
        mapping=mapping)

    # `dropped` is read off `frame`, not the neighbours frame: build_neighbors
    # adds columns and never adds or removes a row, so the two give the same
    # ids.
    neighbor_frame = build_neighbors(frame, barriers, methodology,
                                     id_col=id_col)
    result = index_frames(neighbor_frame, services, methodology, denominator,
                          dropped=dropped, epsg_code=epsg_code, id_col=id_col)
    # After the last index_frames call, never before: pandas drops `attrs`
    # across the merges inside it (a caller that merges this result further
    # loses the stamp too, as pandas documents).
    result.attrs["categories"] = {"scheme": scheme, "mapping": dict(mapping)}
    return result
```

- [ ] **Step 5: Thread it through the `compute` stage**

In `delhi_psi/pipeline.py`, in `compute`, add `mapping=cfg.categories.mapping` to the `_population_and_exclusion` call (lines 438–445), so it reads:

```python
    frame, dropped, missing = _population_and_exclusion(
        neighbor_frame, population, id_col=id_col,
        type_col=cfg.layers.settlements.type_col,
        population_id_col=cfg.layers.population.id_col,
        population_value_col=cfg.layers.population.value_col,
        missing_population=cfg.layers.population.missing,
        max_missing_population=cfg.validate.max_missing_population,
        exclusion_types=cfg.methodology.exclusion.types,
        mapping=cfg.categories.mapping)
```

and replace the output loop (lines 452–462) with:

```python
    # CSV and shapefile cannot carry `attrs`, so for those formats the
    # record of which scheme produced these rows is this line plus the
    # `category` column itself. The scheme is never a column.
    stamp = {"scheme": cfg.categories.scheme,
             "mapping": dict(cfg.categories.mapping)}
    log.info("categories: scheme=%s n_categories=%d", cfg.categories.scheme,
             len(categories.categories_of(cfg.categories.mapping)))

    outputs = []
    n_reported = 0
    for denominator in cfg.outputs.denominators:
        result = index_frames(frame, services, cfg.methodology,
                              str(denominator), dropped=dropped,
                              epsg_code=cfg.crs.epsg, id_col=id_col)
        validate.check_no_negative(result)
        n_reported = len(result)
        # Immediately before the write, mirroring the neighbours stamp:
        # index_frames' merges drop `attrs`, so stamping earlier vanishes.
        result.attrs["categories"] = stamp
        outputs.extend(io.write_outputs(
            result, out_dir, basename=output_basename(cfg, denominator),
            formats=cfg.outputs.formats))
```

`preprocess` is deliberately **untouched**: the neighbours artifact is built
on the full universe and stamped with adjacency/barrier only, so it stays
category-free and a mapping change never forces an 11-minute re-`preprocess`
(spec § 3). If you find yourself adding `categories` to `preprocess`, stop.

- [ ] **Step 6: Run the pipeline tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py tests/test_categories.py -q`
Expected: PASS.

Now run the whole suite: `uv run pytest -q -W error`
Expected: **RED, and only in these two files** — `tests/test_cli.py` and `tests/test_oracle_e2e.py` fail with `validation failed: settlement types with no categories.mapping entry: 'IND' (1 row), 'UC' (1 row); …`, because they drive the CLI with the *shipped* `code-2025`/`manuscript` profiles on the oracle city, whose vocabulary `uso-10` deliberately does not cover. That is the guard working. Steps 7–9 rewire them; if anything **outside** those two files is red, stop and report.

- [ ] **Step 7: Add the derived-profile helpers**

In `tests/oraculum_fixtures.py`, extend the `ORACLE_SCENARIOS` comment block (lines 33–36) so fact (1) of spec § 4 is written down where the scenarios are defined (a module-level list cannot carry a docstring):

```python
# --- profile-driven helpers (Phase 3A) --------------------------------
# The § 7 scenario table: a profile plus `types`/`stage` overrides ONLY.
# `absent_neighbor` always comes from the profile, because in the reference
# it is a rule-set property, not a scenario property.
#
# 3B: `types` entries are CATEGORY names. They select the right rows here
# only because the derived oracle profile's mapping is the IDENTITY over the
# fixture vocabulary — and the reference implementation drops settlements by
# ID, which agrees only because the fixture gives RV and IND ids equal to
# their types. Both facts are pinned in tests/test_fixture_invariants.py.
```

and append the helpers below `ORACLE_SCENARIOS`, before `methodology_with`:

```python
# --- the derived test-only profile (spec 3B § 2) ----------------------
# The oracle city is NOT Delhi: its vocabulary is not covered by the shipped
# profiles' `uso-10` mapping, and padding those with `UC`/`IND` would blunt
# the unmapped-type guard on real data. Every test that runs the CLI or the
# oracle helpers on this city therefore uses a DERIVED profile whose mapping
# is the identity over the fixture vocabulary. The single exception is the
# test that proves the guard fires (tests/test_cli.py).
ORACLE_SCHEME = "oracle-6"
ORACLE_VOCABULARY = ("Planned", "UC", "JJC", "RV", "RUAC", "IND")


def oracle_mapping():
    """The identity over the fixture city's six source types."""
    return {source: source for source in ORACLE_VOCABULARY}


def oracle_config(base):
    """`base`'s shipped Config with the oracle-6 identity categories block.

    Purely in memory — no file, no pytest fixture — because
    scripts/generate_production_fixtures.py reaches it through
    `compute_oracle_frame` and runs as a plain script outside pytest, where
    there is no `tmp_path`.
    """
    from dataclasses import replace

    from delhi_psi.config import CategoriesConfig, load_config

    return replace(load_config(base),
                   categories=CategoriesConfig(scheme=ORACLE_SCHEME,
                                               mapping=oracle_mapping()))


def oracle_profile_path(base, directory):
    """Write the same derived profile as YAML into `directory`; return the
    path, for the tests that drive the real CLI with `--config <path>`."""
    import yaml

    from delhi_psi.config import PROFILES_DIR

    raw = yaml.safe_load((PROFILES_DIR / f"{base}.yaml").read_text())
    raw["categories"] = {"scheme": ORACLE_SCHEME, "mapping": oracle_mapping()}
    path = Path(directory) / f"{base}.oracle.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path
```

Then rewire the two existing helpers. Replace `methodology_with`'s body (lines 47–59) with:

```python
def methodology_with(profile, *, types=None, stage=None):
    """The DERIVED profile's methodology with the two allowed overrides.

    It reads only `.methodology`, so the derived categories block leaves the
    result identical to the shipped profile's.
    """
    from dataclasses import replace

    from delhi_psi.config import ExclusionStage

    methodology = oracle_config(profile).methodology
    exclusion = methodology.exclusion
    if types is not None:
        exclusion = replace(exclusion, types=tuple(types))
    if stage is not None:
        exclusion = replace(exclusion, stage=ExclusionStage(stage))
    return replace(methodology, exclusion=exclusion)
```

and `compute_oracle_frame` (lines 62–72) with:

```python
def compute_oracle_frame(profile, *, types, stage, denom):
    """compute_frames on the Oraculum city under the DERIVED profile's own
    category mapping, indexed by settlement id.

    The fixture city carries its own `population` column, so population=None.
    Passing the profile's mapping (not letting compute_frames default to the
    identity) is what makes a future COLLAPSING profile's fixture record the
    numbers the CLI actually produces; under today's identity profiles it is
    a no-op.
    """
    from delhi_psi.pipeline import compute_frames

    cfg = oracle_config(profile)
    return compute_frames(
        load_settlements(), {"canal": load_barriers()}, load_services(),
        None, methodology_with(profile, types=types, stage=stage), denom,
        mapping=cfg.categories.mapping, scheme=cfg.categories.scheme,
    ).set_index("USO_AREA_U")
```

- [ ] **Step 8: Rewire `tests/test_cli.py`'s helpers**

In `tests/test_cli.py`, add `from pathlib import Path` to the imports and replace the fixtures import (lines 17–20) with:

```python
from tests.oraculum_fixtures import (
    EPSG, load_barriers, load_services, load_settlements, oracle_profile_path,
)
```

(the `from delhi_psi.config import PROFILES_DIR` line goes away — its only user is rewritten below).

Replace the `run` helper (lines 93–94) with:

```python
SHIPPED_PROFILES = ("code-2025", "manuscript")


def run(*args):
    """`cli.main`, with a shipped profile NAME in `--config` swapped for the
    DERIVED oracle profile (spec 3B § 2).

    The oracle city's `UC`/`IND` are deliberately absent from the shipped
    Delhi mappings, so the shipped profiles correctly refuse to run here.
    The derived YAML is written into the run's own `--data-dir`. A test that
    WANTS the shipped profile — the unmapped-type guard — calls `cli.main`
    directly.
    """
    args = list(args)
    if "--config" in args and "--data-dir" in args:
        at = args.index("--config") + 1
        if args[at] in SHIPPED_PROFILES:
            directory = Path(args[args.index("--data-dir") + 1])
            args[at] = str(oracle_profile_path(args[at], directory))
    return cli.main(args)
```

Replace `test_config_by_path_is_equivalent` (lines 114–118) with:

```python
def test_config_by_path_is_equivalent(data_dir, tmp_path):
    """`--config <path>` still works — exercised with the derived profile,
    which is the only complete profile this city can run (spec 3B § 2)."""
    out = tmp_path / "by_path"
    profile = oracle_profile_path("code-2025", tmp_path)
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(out)]) == 0
    assert (out / "colonies_neighbors.joblib").exists()
```

- [ ] **Step 9: Rewire `tests/test_oracle_e2e.py`**

Add the import and use the derived profile for both stages. Replace lines 16 and 28–34 with:

```python
from tests.oraculum_fixtures import oracle_profile_path
from tests.test_cli import data_dir  # noqa: F401  (module-scoped fixture)
```

```python
def test_full_cli_chain_matches_excl_rv_only(data_dir, tmp_path):  # noqa: F811
    out_dir = tmp_path / "out"
    # The derived profile (spec 3B § 2): code-2025's method, with the
    # identity mapping over THIS city's vocabulary. Every methodology value
    # and every assertion below is code-2025's, unchanged.
    profile = str(oracle_profile_path("code-2025", tmp_path))
    _run("preprocess", "--config", profile, "--data-dir", str(data_dir),
         "--out-dir", str(out_dir))
    assert (out_dir / "colonies_neighbors.joblib").exists()
    _run("compute", "--config", profile, "--data-dir", str(data_dir),
         "--out-dir", str(out_dir))
```

- [ ] **Step 10: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **270 passed** (266 + 4).

- [ ] **Step 11: Prove the production fixtures did not move**

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
```

Expected: **empty**. This is the load-bearing check of the whole cycle: `compute_oracle_frame` now runs under the derived profile's mapping, and `scripts/generate_production_fixtures.emit_profile` goes through it — byte-identity here is what proves the identity scheme changed nothing.

- [ ] **Step 12: Commit**

```bash
git add delhi_psi/pipeline.py tests/oraculum_fixtures.py tests/test_pipeline.py \
        tests/test_cli.py tests/test_oracle_e2e.py
git commit -m "feat(pipeline): map types to categories in the shared prelude (DEL-17)

The prelude applies categories.mapping immediately before excluded_ids and
repeats the exclusion.types subset check at run time (in-memory callers never
pass through load_config). compute_frames gains mapping=/scheme= and stamps
attrs on its return; compute stamps immediately before write_outputs and logs
the scheme. Oracle helpers resolve a derived test-only profile whose mapping
is the identity over the fixture vocabulary, so the shipped Delhi profiles
stay pure. Fixtures byte-identical.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: the proofs — collapse equivalence, the guard, the column and the stamp (spec §§ 4, 5)

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_fixture_invariants.py`

**Interfaces:**
- Consumes: `tests.oraculum_fixtures.{ORACLE_VOCABULARY, compute_oracle_frame, oracle_mapping}` and `tests.test_cli.run` (Task 3); `delhi_psi.categories.categories_of` (Task 1); `delhi_psi.io.read_neighbors`.
- Produces:
  - `tests.test_cli.ORACLE_5: dict[str, str]` — the 5-way collapse of the fixture vocabulary
  - `tests.test_cli.collapse_profile_path(directory: Path, *, stage: str) -> Path`
  - No production API.

- [ ] **Step 1: Write the proof tests**

Append to `tests/test_cli.py`:

```python
# --- 3B: the vocabulary-change equivalence proof (spec 3B § 4) --------
REFERENCE_CSV = (Path(__file__).resolve().parent / "fixtures" / "oraculum"
                 / "expected_values.csv")

# The fixture city's six source types collapsed into five categories, of
# which one — `non-urban` — is what the run then excludes. RV and IND are
# the two settlements today's raw `exclusion.types: [RV, IND]` drops.
ORACLE_5 = {"Planned": "planned", "UC": "unauthorized",
            "RUAC": "regularized", "JJC": "jjc", "RV": "non-urban",
            "IND": "non-urban"}

# exclusion.stage -> the reference scenario with the same dropped set.
REFERENCE_SCENARIO = {"post_neighbors": "excl_contributing",
                      "pre_neighbors": "excl_removed"}

# CLI output column -> the same quantity in compute_oracle_frame's frame.
# The fixture's clinic layer is written to Public Services/Health/Health.shp,
# so the config service is `health` where the oracle frame says `clinic`.
COLLAPSE_TO_ORACLE = {
    "health_count": "clinic_count", "health_pcen": "clinic_pcen",
    "health_idx": "clinic_idx",
    "school_count": "school_count", "school_pcen": "school_pcen",
    "school_idx": "school_idx",
    "bank_count": "bank_count", "bank_pcen": "bank_pcen",
    "bank_idx": "bank_idx",
    "police_count": "police_count", "police_pcen": "police_pcen",
    "police_idx": "police_idx",
    "ration_count": "ration_count", "ration_pcen": "ration_pcen",
    "ration_idx": "ration_idx",
    "transport_count": "transport_count",
    "transport_pcen": "transport_pcen", "transport_idx": "transport_idx",
    "road_length": "road_length", "road_pcen": "road_pcen",
    "road_idx": "road_idx",
    "unnorm_psi": "unnorm_psi", "norm_psi": "norm_psi",
    "population": "population", "area_km2": "area_km2",
}

# CLI output column -> expected_values.csv metric name (the REFERENCE's
# names: psi_eq1 for unnorm_psi, road_length_km for road_length).
COLLAPSE_TO_REFERENCE = {
    "health_count": "clinic_count", "health_pcen": "clinic_pcen",
    "health_idx": "clinic_idx",
    "school_count": "school_count", "school_pcen": "school_pcen",
    "school_idx": "school_idx",
    "bank_count": "bank_count", "bank_pcen": "bank_pcen",
    "bank_idx": "bank_idx",
    "police_count": "police_count", "police_pcen": "police_pcen",
    "police_idx": "police_idx",
    "ration_count": "ration_count", "ration_pcen": "ration_pcen",
    "ration_idx": "ration_idx",
    "transport_count": "transport_count",
    "transport_pcen": "transport_pcen", "transport_idx": "transport_idx",
    "road_length": "road_length_km", "road_pcen": "road_pcen",
    "road_idx": "road_idx",
    "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
}


def collapse_profile_path(directory, *, stage):
    """A profile derived from `code-2025` that collapses the fixture's six
    source types into five and excludes the CATEGORY `non-urban`.

    Everything else is code-2025's: reference rule `code`, `swallowed`, the
    second normalization on. Only the vocabulary changes — which is the
    claim under test.
    """
    import yaml

    from delhi_psi.config import PROFILES_DIR

    raw = yaml.safe_load((PROFILES_DIR / "code-2025.yaml").read_text())
    raw["profile"] = "oracle-5"
    raw["categories"] = {"scheme": "oracle-5", "mapping": dict(ORACLE_5)}
    raw["methodology"]["exclusion"]["types"] = ["non-urban"]
    raw["methodology"]["exclusion"]["stage"] = stage
    path = Path(directory) / f"oracle-5-{stage}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


@pytest.mark.parametrize("denom", ["pop", "popdensity"])
@pytest.mark.parametrize("stage", ["post_neighbors", "pre_neighbors"])
def test_five_way_collapse_reproduces_raw_type_exclusion(data_dir, tmp_path,
                                                         stage, denom):
    """Spec 3B § 4, the vocabulary-change equivalence proof.

    A profile that collapses six source types into five and excludes the
    CATEGORY `non-urban` must produce (a) exactly the numbers today's raw
    `exclusion.types: [RV, IND]` produces, and (b) the independent reference
    implementation's own `code` rows for the scenario with the same dropped
    set. Together they are the proof that this layer changed the vocabulary
    and nothing else.

    Tolerance is the CSV round-trip's 1e-9 (the existing e2e's); 1e-12
    applies only to in-memory comparisons.
    """
    from tests.oraculum_fixtures import compute_oracle_frame

    profile = collapse_profile_path(tmp_path, stage=stage)
    out = tmp_path / "collapse"
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    assert cli.main(["compute", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0

    got = pd.read_csv(
        out / f"delhi_psi_oracle-5_{denom}_2020.csv").set_index("USO_AREA_U")
    assert set(got.index) == {"A", "B", "C", "D", "E"}
    assert got["category"].to_dict() == {
        "A": "planned", "B": "unauthorized", "C": "jjc", "D": "planned",
        "E": "regularized"}

    # (a) the same numbers as today's raw-string exclusion
    direct = compute_oracle_frame("code-2025", types=("RV", "IND"),
                                  stage=stage, denom=denom)
    assert set(direct.index) == set(got.index)
    for column, oracle_column in COLLAPSE_TO_ORACLE.items():
        for sid in got.index:
            assert got.loc[sid, column] == pytest.approx(
                direct.loc[sid, oracle_column], abs=1e-9), (column, sid)

    # (b) the independent reference implementation, rule `code`
    expected = pd.read_csv(REFERENCE_CSV)
    expected = expected[
        (expected["rule"] == "code")
        & (expected["scenario"] == REFERENCE_SCENARIO[stage])
        & (expected["denom"] == denom)
    ].pivot(index="settlement", columns="metric", values="value")
    assert set(expected.index) == set(got.index)
    for column, metric in COLLAPSE_TO_REFERENCE.items():
        for sid in got.index:
            assert got.loc[sid, column] == pytest.approx(
                expected.loc[sid, metric], abs=1e-9), (column, sid)


def test_unmapped_settlement_type_exits_1(data_dir, tmp_path, capsys):
    """The shipped `code-2025` mapping is Delhi's `uso-10`; this city
    carries `UC` and `IND`, which are deliberately NOT in it. Running the
    SHIPPED profile straight at this city is therefore the proof that an
    unmapped source type fails the run, naming every offender with its row
    count (spec 3B §§ 2, 5).

    `cli.main`, not `run`: the whole point is the shipped profile.
    """
    out = tmp_path / "unmapped"
    assert cli.main(["preprocess", "--config", "code-2025",
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    assert cli.main(["compute", "--config", "code-2025",
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 1
    err = capsys.readouterr().err
    assert "validation failed" in err
    assert "'IND' (1 row)" in err
    assert "'UC' (1 row)" in err
    assert "categories.mapping" in err


def test_outputs_carry_the_category_column_and_the_scheme_stamp(data_dir,
                                                                tmp_path,
                                                                caplog):
    """`category` on the CSV, the shapefile, the joblib and
    missing_population.csv; the scheme/mapping stamp on the joblib, which is
    the only format that can hold `attrs`. For CSV and shapefile the record
    is the INFO line plus the column itself — the scheme is never a column.
    """
    import logging as _logging

    from delhi_psi import io

    out = tmp_path / "categories"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    caplog.clear()
    with caplog.at_level(_logging.INFO, logger="delhi_psi.pipeline"):
        assert run("compute", "--config", "code-2025",
                   "--data-dir", str(data_dir), "--out-dir", str(out)) == 0

    base = out / "delhi_psi_code-2025_pop_2020"
    csv = pd.read_csv(base.with_suffix(".csv")).set_index("USO_AREA_U")
    # RV is excluded (code-2025's exclusion.types), so it is not reported.
    assert csv["category"].to_dict() == {
        "A": "Planned", "B": "UC", "C": "JJC", "D": "Planned", "E": "RUAC",
        "IND": "IND"}
    assert "scheme" not in csv.columns, "the scheme is metadata, not a column"
    assert "category" in gpd.read_file(base.with_suffix(".shp")).columns
    assert "category" in pd.read_csv(out / "missing_population.csv").columns

    frame = io.read_neighbors(base.with_suffix(".joblib"))
    assert frame.attrs["categories"]["scheme"] == "oracle-6"
    assert frame.attrs["categories"]["mapping"] == {
        "Planned": "Planned", "UC": "UC", "JJC": "JJC", "RV": "RV",
        "RUAC": "RUAC", "IND": "IND"}

    # The NEIGHBOURS artifact stays category-free: it is built on the full
    # universe and stamped with adjacency/barrier only, so a mapping change
    # must never force an 11-minute re-`preprocess` (spec 3B § 3).
    nbrs = io.read_neighbors(out / "colonies_neighbors.joblib")
    assert "category" not in nbrs.columns
    assert "categories" not in nbrs.attrs

    messages = [record.getMessage() for record in caplog.records]
    assert "categories: scheme=oracle-6 n_categories=6" in messages, messages
```

Append to `tests/test_fixture_invariants.py`:

```python
# --- 3B: what makes category exclusion and the reference agree ---------
def test_ids_equal_their_types_for_exactly_rv_and_ind(city):
    """The reference implementation drops settlements by ID; production
    drops them by CATEGORY. `types=("RV", "IND")` selects the same rows in
    both only because these two settlements have an id equal to their type.
    Pin it: change the fixture and every exclusion scenario silently means
    something else (spec 3B § 4).
    """
    same = {sid for sid, row in city.iterrows() if sid == row["USO_FINAL"]}
    assert same == {"RV", "IND"}


def test_the_oracle_identity_mapping_is_the_fixture_vocabulary(city):
    """The derived test profile's mapping is the identity over exactly the
    types this city carries — no more (it would hide an unmapped type), no
    fewer (the run would error)."""
    from delhi_psi.categories import categories_of
    from tests.oraculum_fixtures import ORACLE_VOCABULARY, oracle_mapping

    assert set(city["USO_FINAL"]) == set(ORACLE_VOCABULARY)
    assert set(oracle_mapping()) == set(ORACLE_VOCABULARY)
    assert categories_of(oracle_mapping()) == frozenset(ORACLE_VOCABULARY)
```

- [ ] **Step 2: Run the new tests**

Run: `uv run pytest tests/test_cli.py tests/test_fixture_invariants.py -q`
Expected: **PASS**.

This task is the one that inverts the usual order, deliberately: these are
*proofs* of behaviour Task 3 already built, not drivers of new code — there is
no production change in this task at all. Written before Task 3 they would
have failed with `TypeError: compute_frames() got an unexpected keyword
argument 'mapping'` and `KeyError: 'category'`; written after it, they pass on
the first run. So Step 3 does the job the RED run normally does: it proves
they are load-bearing rather than vacuous.

> **If any of them fails here** — a number off beyond 1e-9, a missing
> `category` column, an absent stamp — that is a defect in Task 3, not in the
> test. Fix `delhi_psi/pipeline.py`; never weaken an assertion or a tolerance.

- [ ] **Step 3: Prove the new tests are load-bearing (two mutations)**

**Mutation A — the equivalence is real, not vacuous.** In
`tests/test_cli.py`, temporarily change `collapse_profile_path`'s exclusion
line from

```python
    raw["methodology"]["exclusion"]["types"] = ["non-urban"]
```

to

```python
    raw["methodology"]["exclusion"]["types"] = ["jjc"]
```

Run: `uv run pytest tests/test_cli.py -q -k five_way_collapse`
Expected: **4 failed** at the reported-index assertion —
`assert {'A', 'B', 'D', 'E', 'IND', 'RV'} == {'A', 'B', 'C', 'D', 'E'}`:
excluding `jjc` drops C and keeps RV and IND, so a different category
excluded means a different set of settlements. Revert:

```bash
git checkout -- tests/test_cli.py
```

**Mutation B — the stamp really comes from `compute`.** In
`delhi_psi/pipeline.py`, temporarily comment out the line
`result.attrs["categories"] = stamp` inside `compute`'s denominator loop.

Run: `uv run pytest tests/test_cli.py -q -k category_column_and_the_scheme_stamp`
Expected: **1 failed** — `KeyError: 'categories'` reading
`frame.attrs["categories"]["scheme"]`. Revert:

```bash
git checkout -- delhi_psi/pipeline.py
```

**Mutation C — the numeric comparisons (a) and (b) are load-bearing.**
Mutation A trips the settlement-set assertion first and never reaches the
value loops (plan review R1). So, keeping the settlement set and the
`category` dict untouched, mis-pair one column: in `tests/test_cli.py`
temporarily change `COLLAPSE_TO_ORACLE`'s entry
`"health_pcen": "clinic_pcen"` to `"health_pcen": "school_pcen"`.

Run: `uv run pytest tests/test_cli.py -q -k five_way_collapse`
Expected: **4 failed**, each at the comparison-(a) loop with an
`assert got.loc[sid, column] == pytest.approx(direct.loc[sid, oracle_column], abs=1e-9)`
mismatch for `column == "health_pcen"` (clinic and school PCENs differ on
every settlement — e.g. A: clinic 0.029142… vs school 0.014142…), NOT at the
index or category assertions. Then do the same to `COLLAPSE_TO_REFERENCE`
(`"health_pcen": "clinic_pcen"` → `"health_pcen": "school_pcen"`) with
`COLLAPSE_TO_ORACLE` restored, and confirm the failure moves to the
comparison-(b) loop. Revert:

```bash
git checkout -- tests/test_cli.py
```

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **278 passed** (270 + 8: four collapse parametrizations, the exit-1 guard, the column/stamp/log case, and the two fixture invariants).

- [ ] **Step 5: Prove the production fixtures did not move**

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
```

Expected: **empty**.

- [ ] **Step 6: Commit**

```bash
git add tests/test_cli.py tests/test_fixture_invariants.py
git commit -m "test: vocabulary-change equivalence and the unmapped-type guard (DEL-17)

A 5-way collapse profile on the oracle city reproduces, for both stages and
both denominators, the numbers today's raw exclusion.types [RV, IND]
produces AND the reference implementation's code/excl_contributing and
code/excl_removed blocks at 1e-9. The shipped code-2025 profile run straight
at this city exits 1 naming UC and IND with their row counts. Plus the
category column on every output format, the joblib scheme stamp, and the
id == type / vocabulary pins the equivalence rests on.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: documentation, changelog, WORKPLAN, real-data proof (spec § 6)

**Files:**
- Modify: `docs/methodology-config.md`
- Create: `docs/data/uso_final_vocabulary.md`
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `WORKPLAN.md` (DEL-17 tick, DEL-31 note)

**Interfaces:** none.

- [ ] **Step 1: Run the real-data proof by hand**

This is a **hand-run step, not a CI test** — CI never touches `~/delhi_data`
(`tests/test_ci_workflow.py::test_no_data_dependency` enforces that). The data
directory is read-only; everything lands under `--out-dir`. The first
`preprocess` in a fresh out-dir recomputes the O(n²) dedup over 4,357 colonies
— budget several minutes and run it in the background.

```bash
uv run delhi-psi preprocess --config code-2025 \
    --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3b_verify
uv run delhi-psi compute --config code-2025 \
    --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3b_verify
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3b_verify
```

Expected from `compute`, among the INFO lines (this is the cycle's new one,
and the proof that all 10 Delhi types are mapped — an unmapped type would
have exited 1 instead):

```
INFO delhi_psi.pipeline: categories: scheme=uso-10 n_categories=10
```

Expected from the verifier, verbatim:

```
PASS — new run equivalent to July 2025 baseline within tolerance
```

with every reported `max abs deviation` line reading `0.000e+00` on all 23
columns. Paste the full output into the PR body, as for 3A/DEL-26/DEL-23, and
quote the `categories:` line above in the changelog entry in Step 3.

**Any non-zero deviation is a stop condition** (spec § 8) — do not tune a
tolerance, do not exclude a column. A `columns missing from new run` line
would mean an output column was renamed or dropped; the new `category` column
is a string and cannot itself appear there.

- [ ] **Step 2: Write the "Categories" section of `docs/methodology-config.md`**

First, correct the sentence in § 1 that still describes exclusion in raw-type
terms. Replace:

```markdown
Also in the block: `exclusion.types` (which `USO_FINAL` types are dropped;
`[RV]` today), `exclusion.stage` (`post_neighbors` = today: neighbours are
built on the full universe, exclusion happens at compute), `barrier.combine`.
```

with:

```markdown
Also in the block: `exclusion.types` (which **categories** are dropped —
category names, not raw `USO_FINAL` types; `[RV]` today, which is a category
only because the shipped mapping is the identity — see § 2),
`exclusion.stage` (`post_neighbors` = today: neighbours are built on the full
universe, exclusion happens at compute), `barrier.combine`.
```

Then insert a whole new section between § 1 and today's § 2, and **renumber
the three sections that follow** (`## 2. Procedure for a decision` → `## 3.`,
`## 3. What each proof guards` → `## 4.`, `## 4. Things that are code, not
config` → `## 5.`). Finally delete the two closing lines of the file
(`Category mappings (`categories:` block, 10/8/5/4 settlement types) are /
cycle 3B (DEL-17).`) — the new section replaces them.

The new section:

````markdown
## 2. Categories — the settlement-type mapping

Since cycle 3B (27 Aug 2026) a profile also states, in full, how the
layer's **source types** collapse into the **categories** the analysis
uses. `categories:` is required in every profile, like `methodology:`.

Two knobs, and they do different jobs:

| knob | job |
|---|---|
| `categories.mapping` | source type → category. 1:1 (identity) or X:1 (several sources into one category). This is the vocabulary. |
| `methodology.exclusion.types` | which **categories** are dropped from the reported frame. Written in the mapping's category names, never in raw source types. |

`categories.scheme` is a free-form name for the mapping. It is recorded in
the joblib output's `attrs` and in one INFO line per run
(`categories: scheme=… n_categories=…`); it is never a column. Every output
— CSV, shapefile, joblib — and `missing_population.csv` carry a `category`
column next to the raw `USO_FINAL`, which is kept as-is.

**An unmapped source type fails the run.** If the layer carries a type with
no entry in `mapping`, `compute` exits 1 and names every offender with its
row count, so one run diagnoses the whole layer. There is deliberately no
catch-all: `categories.default` is rejected at load. A mapping entry for a
type that is *absent* from the data is fine — a scheme may be broader than
one city. Duplicate keys in the YAML are also rejected (PyYAML keeps the
last one silently, which is the same failure wearing a different hat).

### Worked example 1 — the oracle city, six types into five

This is what the test suite exercises
(`tests/test_cli.py::test_five_way_collapse_reproduces_raw_type_exclusion`).
The 7-settlement fixture city carries `Planned, UC, JJC, RV, RUAC, IND`
(`UC` and `IND` are the fixture's shorthand for `UAC` and `Industrial`):

```yaml
categories:
  scheme: oracle-5
  mapping:
    Planned: planned
    UC: unauthorized
    RUAC: regularized
    JJC: jjc
    RV: non-urban
    IND: non-urban
methodology:
  exclusion:
    types: [non-urban]
```

Six source types, five categories, and the run drops `non-urban` — which is
exactly the two settlements the old raw `exclusion.types: [RV, IND]`
dropped. The test proves those two runs are numerically identical, and that
both match the independent reference implementation. That equivalence is
the whole claim of this layer: **it changes the vocabulary, not the
numbers.**

### Worked example 2 — Delhi, ten types into the Phase 4 candidate

The workshop's working candidate (WORKPLAN DEL-29): planned /
unauthorized / regularized-unauthorized / resettlement / JJC, with the
non-urban types dropped. In YAML, `regularized` is the token for WORKPLAN's
"regularized-unauthorized colonies":

```yaml
categories:
  scheme: urban-5
  mapping:
    Planned: planned
    UAC: unauthorized
    RUAC: regularized
    JJR: resettlement
    JJC: jjc
    RV: non-urban
    Industrial: non-urban
    # UV: ?            # Raj to decide — must be mapped or the run errors
    # SDA: ?           # Raj to decide — must be mapped or the run errors
    # Other: ?         # Raj to decide — must be mapped or the run errors
methodology:
  exclusion:
    types: [non-urban]
```

**This profile does not ship, and as written it would not run:** `UV` (138
rows), `SDA` (86) and `Other` (33) are real types on the real layer with no
entry, so `compute` would exit 1 naming all three. That is the design —
they are open questions (DEL-29 explicitly flags SDA), and the pipeline
refuses to guess. Counts and provenance for all ten types:
`docs/data/uso_final_vocabulary.md`.

### Procedure for Raj's decision (DEL-31)

1. Copy `delhi_psi/profiles/code-2025.yaml` to
   `delhi_psi/profiles/urban-5.yaml`, set `profile: urban-5`, and write the
   agreed `categories:` block — every one of the 10 source types mapped.
2. Set `methodology.exclusion.types: [non-urban]` (category names).
3. Register the profile and regenerate the fixtures exactly as in § 3
   below. `code-2025.csv` and `manuscript.csv` must not change; the new
   `urban-5.csv` diff *is* the categorization decision on the oracle city.
4. Run the suite, then the real data (§ 3 step 5). This is the DEL-32
   recalculation; no code changes.
````

- [ ] **Step 3: Write `docs/data/uso_final_vocabulary.md`**

```bash
mkdir -p docs/data
```

Create `docs/data/uso_final_vocabulary.md`:

```markdown
# `USO_FINAL` — the settlement-type vocabulary

Measured 27 Aug 2026 on `~/delhi_data/uso_update_sep2021/uso_update_sep2021.shp`.
Recorded here because the mapping layer (`categories:`, cycle 3B / DEL-17) and
Raj's categorization decision (Phase 4 / DEL-29) both argue from these counts.

## The 10 types

4,357 rows, no nulls, unchanged by deduplication:

| `USO_FINAL` | rows |
|---|---:|
| UAC | 1,684 |
| Planned | 964 |
| JJC | 764 |
| RUAC | 393 |
| RV | 211 |
| UV | 138 |
| SDA | 86 |
| JJR | 48 |
| Industrial | 36 |
| Other | 33 |
| **total** | **4,357** |

`code-2025` today excludes `RV` (211 rows) and nothing else. DEL-28 proposes
dropping every non-urban type, which adds `Industrial`.

## Provenance: these 10 are already a merge

The 10 are themselves an undocumented **16 → 10** merge performed in the 2021
notebooks (`archive/master-2021/`): `UAC1 → UAC`, `JJC1`/`JJC2 → JJC`, and
`Institutional`, `Commercial`, `DCB`, `NDMC` folded in or dropped. This page
records that; it does not re-derive it (spec 3B § 7 — out of scope). Anyone
re-deriving the merge should start from
`archive/master-2021/Colonies Dataset Pre-Processing (29 Aug 2021).ipynb`.

## The oracle fixture's vocabulary is different on purpose

`tests/fixtures/oraculum/settlements.geojson` carries six types —
`Planned, UC, JJC, RV, RUAC, IND` — where `UC` and `IND` are shorthand for
`UAC` and `Industrial`. The shipped Delhi profiles are deliberately **not**
padded with `UC`/`IND`: padding them would blunt the unmapped-type guard on
real data. Tests that run the oracle city therefore use a derived, test-only
profile (`tests/oraculum_fixtures.oracle_config`), and one test runs the
shipped profile at the fixture city precisely to prove the guard fires.
```

- [ ] **Step 4: Add the changelog entry**

Under `## [Unreleased]` in `CHANGELOG.md`, add as the **first** bullet
(newest first, matching the existing ordering), substituting the actual
figures from Step 1's run if they differ:

```markdown
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
```

- [ ] **Step 5: Tick the WORKPLAN items**

In `WORKPLAN.md`, Phase 3, change the DEL-17 bullet from `- [ ]` to `- [x]`
and append the note, so it reads:

```markdown
- [x] Make settlement types configurable via a mapping layer: run with 10, 8,
      5, or 4 categories from a config (1:1 or X:1 mapping of the 10
      `USO_FINAL` source types), so Raj's categorization decision (Phase 4)
      plugs in without code changes — and so the method ports to other cities
      [DEL-17]
      — done 27 Aug 2026 (3B): profiles carry a required `categories:` block
      (`scheme` + source type → category mapping); `exclusion.types` is
      written in category names; every output carries a `category` column and
      the joblib the scheme stamp. An unmapped source type errors (no
      catch-all, `categories.default` reserved). Both shipped profiles use
      the identity `uso-10` scheme, so no number moved: fixtures
      byte-identical, real-data baseline at 0.000e+00. Spec
      `docs/superpowers/specs/2026-08-27-phase3b-categories-design.md`, plan
      `docs/superpowers/plans/2026-08-27-phase3b-categories.md`
```

and in Phase 4, append to the DEL-31 bullet:

```markdown
- [ ] **Bob:** encode the agreed mapping in the Phase 3 mapping layer [DEL-31]
      — unblocked 27 Aug 2026 (3B): this is now **one YAML profile** —
      copy `code-2025.yaml`, write the `categories.mapping` block, set
      `exclusion.types: [non-urban]`, regenerate the fixtures. No code
      change. Worked example and procedure: `docs/methodology-config.md`
      § 2. Every one of the 10 source types must be mapped — `UV`, `SDA` and
      `Other` are the open ones (DEL-29)
```

- [ ] **Step 6: Confirm the 3A spec erratum needs no edit**

Spec 3B § 6 asks for a one-line erratum in the parent spec. It is **already
applied on this branch** — verify, do not re-edit:

```bash
grep -n "from cycle 3B" docs/superpowers/specs/2026-08-27-phase3-refactor-design.md
```

Expected: exactly one hit, in § 3's "**Required keys:**" sentence. If it is
missing, apply it: the sentence must read "**Required keys:** `profile`, the
whole `methodology` block and (from cycle 3B, spec
`2026-08-27-phase3b-categories-design.md`) the whole `categories` block"; the
defaulted list is unchanged.

- [ ] **Step 7: Run the whole suite and the fixture guard one last time**

```bash
uv run pytest -q -W error
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
```

Expected: **278 passed**, and an empty `git status` output.

- [ ] **Step 8: Commit**

```bash
git add docs/methodology-config.md docs/data/uso_final_vocabulary.md \
        CHANGELOG.md WORKPLAN.md
git commit -m "docs: categories procedure, USO_FINAL vocabulary, DEL-17 done

docs/methodology-config.md gains a Categories section with both worked
examples (the oracle city's oracle-5 collapse and Delhi's urban-5 candidate,
with UV/SDA/Other marked for Raj); docs/data/uso_final_vocabulary.md records
the 10 types, their counts and the undocumented 16->10 merge behind them.
CHANGELOG + WORKPLAN: DEL-17 done, DEL-31 is now one YAML.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Test counts** (`uv run pytest -q -W error`): 246 at the start → 254 (T1) →
  266 (T2) → 270 (T3) → 278 (T4, T5). If your count is higher because you
  split a parametrization differently, that is fine. A count that *drops* is
  not — it means a carried-over test disappeared.
- **Task 3 is the only task with a deliberate red window** (between Steps 6
  and 10) and it is bounded: `tests/test_cli.py` and `tests/test_oracle_e2e.py`
  only, all with the unmapped-type message. Anything else red there is a
  defect.
- **The two comparison tolerances are not interchangeable.** In-memory: 1e-12.
  Anything read back from a CSV: 1e-9. Do not "harmonise" them.
- If a review finding conflicts with this plan, spec § 8 governs: a confirmed
  Critical finding wins over the plan.
