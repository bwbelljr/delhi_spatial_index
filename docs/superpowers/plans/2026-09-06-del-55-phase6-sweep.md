# Phase 6 sweep harness + dry run — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship eleven Phase 6 sweep profiles, a runner and a summariser, then
produce provisional sensitivity tables from a real-data dry run against
`code-2025`.

**Architecture:** Each sweep point is one short YAML profile that inherits
everything from `code-2025` except the one or two methodology keys under test
(the loader supplies layer, service, CRS and validation defaults for any block
a profile omits). `scripts/run_sweep.py` drives `delhi-psi preprocess` /
`compute` per point as subprocesses and records cost in a manifest;
`scripts/summarize_sweep.py` reads the resulting CSVs plus the already-proven
`code-2025` baseline and emits rank-based comparison blocks into
`docs/data/phase6_sweep.md`.

**Tech Stack:** uv, Python 3.13, pandas/geopandas, pytest, PyYAML. **No new
dependency.** scipy is not a dependency of this project — not direct, not
transitive, absent from `uv.lock` — so Spearman ρ and Kendall τ-b are
implemented on pandas + numpy in Task 5. Adding scipy would also mean
regenerating and committing `uv.lock`, which CI's `uv sync --locked` requires
and which is easy to forget; and it would promote a heavyweight runtime
dependency for code that lives in `scripts/`, not in the shipped package.

**Spec:** `docs/superpowers/specs/2026-09-06-phase6-sweep-dry-run-design.md`

## Global Constraints

- **Nothing writes under `~/delhi_data`** except through an explicit
  `--out-dir`. It is bisynced hourly to a shared drive. The sweep's work dir
  is `~/psi_sweep`. `~/delhi_data/phase3_verify` is **read-only** here.
- **No existing expected value may move.** `expected_values.csv`,
  `production/code-2025.csv` and `production/manuscript.csv` must be
  byte-identical to `main`. `variants_expected_values.csv` must be
  **addition-only**, verified by `diff`, not by a generator's own report.
- **No new methodology values, no reference-implementation changes**, beyond
  the four variant rows in Task 3 (which are new *constants* of already-proven
  forms, not new code paths).
- Every sweep profile carries `outputs.denominators: [popdensity]` and
  `outputs.formats: [csv]`.
- Sweep profiles are **not** added to `PROFILE_RULES` in
  `tests/test_profiles_match_reference.py` — they match neither ruleset
  wholesale and are pinned by their production fixture alone.
- Python style: match the file you are editing. Comments explain *why*, in the
  register the surrounding code uses.
- **Implementers run only their own test files.** The controller runs
  `uv run pytest -q -W error`. A backgrounded full-suite run notifies only the
  controller session, and sub-agents have stalled on exactly this.
- Commit trailers on every commit:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
  ```

---

### Task 1: The five adjacency and band profiles

**Files:**
- Create: `delhi_psi/profiles/adj-touch.yaml`, `band-0km.yaml`,
  `band-1km.yaml`, `band-5km.yaml`, `band-10km.yaml`
- Create: `tests/test_sweep_profiles.py`
- Modify: `scripts/generate_production_fixtures.py:36` (`PROFILES`)
- Modify: `tests/test_production_fixtures.py:17` (`PROFILES`)
- Modify: `tests/test_config.py:55-56` (`test_both_profiles_ship` asserts the
  EXACT set of shipped YAMLs and goes red the moment a profile is added)
- Modify: `docs/methodology-config.md` § 3 step 2 (the registration list omits
  `tests/test_config.py`, which is why the surprise exists)
- Create (generated): `tests/fixtures/oraculum/production/<profile>.csv` and
  `tests/fixtures/messy/production/<profile>.csv`, ten files

**Interfaces:**
- Consumes: `delhi_psi.config.load_config`, which accepts a profile *name* and
  supplies defaults for every omitted top-level block except `profile`,
  `categories` and `methodology`.
- Produces: the profile names above, and `SWEEP_PROFILES` in
  `tests/test_sweep_profiles.py` — a `{profile: {dotted key: value}}` table
  that Task 2 extends.

- [ ] **Step 1: Write the failing test**

`tests/test_sweep_profiles.py`:

```python
"""Every sweep profile is `code-2025` with exactly the keys it claims moved.

A Phase 6 sweep point is only interpretable if its diff against the baseline
is ONE factor. A profile that quietly also changed, say, `roads` would produce
a number nobody could attribute, and no other test in this repo would notice:
the production fixtures pin what the profile DOES, not what it differs from.
"""
import dataclasses

import pytest

from delhi_psi.config import load_config

BASE = "code-2025"

# profile -> the dotted methodology keys it moves, and their values.
# Task 2 appends the six decay profiles here.
SWEEP_PROFILES = {
    "adj-touch": {"adjacency.rule": "touch"},
    "band-0km": {"adjacency.rule": "within_distance",
                 "adjacency.max_distance_km": 0.0},
    "band-1km": {"adjacency.rule": "within_distance",
                 "adjacency.max_distance_km": 1.0},
    "band-5km": {"adjacency.rule": "within_distance",
                 "adjacency.max_distance_km": 5.0},
    "band-10km": {"adjacency.rule": "within_distance",
                  "adjacency.max_distance_km": 10.0},
}

# The profiles whose neighbourhood differs from `code-2025`, so each needs its
# own artifact and must NOT pin a name (the per-profile default keeps two
# points from overwriting each other).
OWN_ARTIFACT = set(SWEEP_PROFILES)


def flatten(obj, prefix=""):
    """{dotted key: comparable value} over a nested dataclass."""
    if dataclasses.is_dataclass(obj):
        out = {}
        for field in dataclasses.fields(obj):
            out.update(flatten(getattr(obj, field.name), f"{prefix}{field.name}."))
        return out
    key = prefix.rstrip(".")
    if isinstance(obj, (list, tuple)):
        return {key: tuple(str(v) for v in obj)}
    # Enum members are str-valued; compare on the string so a test table can
    # be written in plain YAML vocabulary.
    return {key: str(obj) if not isinstance(obj, bool) else obj}


@pytest.fixture(scope="module")
def base_methodology():
    return flatten(load_config(BASE).methodology)


@pytest.mark.parametrize("profile", sorted(SWEEP_PROFILES))
def test_the_profile_moves_exactly_the_keys_it_claims(profile, base_methodology):
    got = flatten(load_config(profile).methodology)
    moved = {k: v for k, v in got.items() if base_methodology.get(k) != v}
    expected = {k: (str(v) if not isinstance(v, bool) else v)
                for k, v in SWEEP_PROFILES[profile].items()}
    assert moved == expected


@pytest.mark.parametrize("profile", sorted(SWEEP_PROFILES))
def test_the_profile_copies_the_baseline_categories_verbatim(profile):
    """Each sweep profile hand-copies the ten-category identity mapping, and
    the one-factor guard above only inspects `methodology`. A typo in a
    category name would change which settlements are excluded — a second
    factor, invisible to every other test in the repo."""
    base = load_config(BASE).categories
    got = load_config(profile).categories
    assert got.mapping == base.mapping
    assert got.scheme == base.scheme


@pytest.mark.parametrize("profile", sorted(SWEEP_PROFILES))
def test_the_profile_reports_one_denominator_as_csv(profile):
    cfg = load_config(profile)
    assert [str(d) for d in cfg.outputs.denominators] == ["popdensity"]
    assert [str(f) for f in cfg.outputs.formats] == ["csv"]


@pytest.mark.parametrize("profile", sorted(SWEEP_PROFILES))
def test_a_profile_with_its_own_neighbourhood_does_not_pin_an_artifact(profile):
    if profile not in OWN_ARTIFACT:
        pytest.skip("shares the bbox artifact; Task 2 pins that case")
    cfg = load_config(profile)
    assert str(cfg.paths.neighbors_artifact) == f"colonies_neighbors_{profile}.joblib"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest -q -W error tests/test_sweep_profiles.py`
Expected: FAIL — `ConfigError`/`FileNotFoundError`, no such profile.

- [ ] **Step 3: Write the five profiles**

`delhi_psi/profiles/band-1km.yaml` in full. The other four are this file with
the header comment, `profile:` and the `adjacency:` block changed; every other
line is identical.

```yaml
profile: band-1km
# A Phase 6 sweep point (DEL-36 / DEL-55): `code-2025` with the ADJACENCY RULE
# and nothing else moved, so the diff against the baseline's outputs is
# attributable to the distance band alone. Every block this file omits —
# layers, services, crs, validate, paths.data_dir — takes the code-2025
# default from the loader. `paths.neighbors_artifact` is deliberately absent:
# the per-profile default `colonies_neighbors_band-1km.joblib` is what keeps
# this point from overwriting another's artifact.
#
# DRY RUN ONLY. Its numbers describe the frozen July 2025 rule set, which
# Raj's 28 Aug decisions supersede. See
# docs/superpowers/specs/2026-09-06-phase6-sweep-dry-run-design.md.
categories:
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
methodology:
  adjacency:
    rule: within_distance           # THE CHANGE (1 of 2)
    max_distance_km: 1.0            # THE CHANGE (2 of 2) — km, polygon-to-polygon
  barrier:
    rule: global_asymmetric
    combine: any
  overlap:
    lending: whole
  decay:
    form: inverse_linear
    distance: centroid
    distance_unit: km
  roads: decayed
  second_normalization: true
  exclusion:
    types: [RV]
    stage: post_neighbors
    absent_neighbor: swallowed
outputs:
  denominators: [popdensity]        # ONE denominator: `compute` runs the whole
                                    # per-link pcen loop once per entry, and on
                                    # the 10 km band that loop walks 4.37M
                                    # links. Figure 4's denominator is
                                    # popdensity; the baseline run carries both
                                    # and the doc's denominator check compares
                                    # them once. Spec § 4.3.
  formats: [csv]                    # the summariser reads CSV; shp/joblib
                                    # would add ~18 MB per point for nothing
  name_template: "delhi_psi_{profile}_{denominator}_2020"
```

`band-0km.yaml`, `band-5km.yaml`, `band-10km.yaml`: identical but for
`profile:` and `max_distance_km:` (`0.0`, `5.0`, `10.0`). `band-0km.yaml`'s
header comment adds:

```
# X = 0 is NOT `touch`: it is intersection-inclusive, so it is `touch` plus
# corner-only contacts. It is DEL-39's third comparison point.
```

`adj-touch.yaml` has the same file with the adjacency block replaced by:

```yaml
  adjacency:
    rule: touch                     # THE CHANGE — shared border of positive
                                    # length. `manuscript` also uses `touch`,
                                    # but it moves seven other switches too;
                                    # this profile is the one-factor comparison
                                    # DEL-39 actually asks for.
```

- [ ] **Step 4: Run the profile tests and watch them pass**

Run: `uv run pytest -q -W error tests/test_sweep_profiles.py`
Expected: PASS.

- [ ] **Step 5: Register the profiles**

In `scripts/generate_production_fixtures.py`, extend `PROFILES` (keep
`code-2025` and `manuscript` first, then the sweep profiles in the § 3 order):

```python
PROFILES = ("code-2025", "manuscript",
            "adj-touch", "band-0km", "band-1km", "band-5km", "band-10km")
```

Make the same edit to `PROFILES` in `tests/test_production_fixtures.py` (it is
a list there, not a tuple).

Then `tests/test_config.py:55-56`, which no registration document mentions and
which goes red the moment an eighth line lands in `delhi_psi/profiles/`:

```python
def test_both_profiles_ship():
    assert sorted(shipped_profiles()) == ["code-2025", "manuscript"]
```

Keep its intent — the shipped set is *exactly* what we think it is, so a
forgotten or stray YAML is caught — by widening the literal rather than
loosening the assertion:

```python
# Every YAML in delhi_psi/profiles/, in one place. A stray or forgotten file
# is exactly what this test exists to catch, so it stays an equality: adding a
# profile means adding it here, and `docs/methodology-config.md` § 3 step 2
# says so.
SHIPPED = [
    "adj-touch", "band-0km", "band-10km", "band-1km", "band-5km", "code-2025",
    "decay-boundary", "decay-exp2km", "decay-exp5km", "decay-none",
    "decay-power05", "decay-power2", "manuscript",
]


def test_the_shipped_profiles_are_exactly_these():
    assert sorted(shipped_profiles()) == sorted(SHIPPED)


def test_the_two_production_profiles_still_ship():
    """The pair everything else defaults to; named separately so the intent
    survives the sweep profiles being deleted one day."""
    assert {"code-2025", "manuscript"} <= set(shipped_profiles())
```

Task 1 adds the five band/adjacency names; Task 2 appends its six.

Finally, `docs/methodology-config.md` § 3 step 2 ("**Register it.**") gains the
missing line, so the next person adding a profile is not ambushed:

```
   Add the name to `SHIPPED` in `tests/test_config.py` as well — it asserts
   the exact set of shipped YAMLs, and a new profile turns it red.
```

- [ ] **Step 6: Generate the fixtures and verify nothing else moved**

```bash
uv run python scripts/generate_production_fixtures.py
git status --porcelain tests/fixtures
```
Expected: exactly ten new `production/*.csv` files, and **no modification** to
`production/code-2025.csv` or `production/manuscript.csv`. If either is
modified, STOP and report — that is a hard stop under spec § 8.

- [ ] **Step 7: Run this task's tests**

Run: `uv run pytest -q -W error tests/test_sweep_profiles.py tests/test_production_fixtures.py tests/test_config.py`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add delhi_psi/profiles tests/test_sweep_profiles.py tests/test_config.py \
        scripts/generate_production_fixtures.py tests/test_production_fixtures.py \
        docs/methodology-config.md tests/fixtures
git commit -m "feat(profiles): the five adjacency and band sweep points (DEL-55)"
```

---

### Task 2: The six decay profiles, sharing one neighbours artifact

**Files:**
- Create: `delhi_psi/profiles/decay-none.yaml`, `decay-power05.yaml`,
  `decay-power2.yaml`, `decay-exp2km.yaml`, `decay-exp5km.yaml`,
  `decay-boundary.yaml`
- Modify: `tests/test_sweep_profiles.py` (extend `SWEEP_PROFILES`, add the
  shared-artifact test)
- Modify: `scripts/generate_production_fixtures.py`,
  `tests/test_production_fixtures.py` (`PROFILES`), `tests/test_config.py`
  (`SHIPPED`, the six decay names)
- Create (generated): twelve more `production/*.csv`

**Interfaces:**
- Consumes: `SWEEP_PROFILES` and `OWN_ARTIFACT` from Task 1.
- Produces: `SHARED_ARTIFACT = "colonies_neighbors_sweep-bbox.joblib"` in
  `tests/test_sweep_profiles.py`, which Task 4's runner test imports.

- [ ] **Step 1: Extend the test table and add the shared-artifact test**

In `tests/test_sweep_profiles.py`, add to `SWEEP_PROFILES`:

```python
    "decay-none": {"decay.form": "none"},
    "decay-power05": {"decay.form": "inverse_power", "decay.exponent": 0.5},
    "decay-power2": {"decay.form": "inverse_power", "decay.exponent": 2.0},
    "decay-exp2km": {"decay.form": "exponential", "decay.scale_km": 2.0},
    "decay-exp5km": {"decay.form": "exponential", "decay.scale_km": 5.0},
    "decay-boundary": {"decay.distance": "boundary"},
```

`OWN_ARTIFACT` stays exactly the five Task 1 names. Add:

```python
SHARED_ARTIFACT = "colonies_neighbors_sweep-bbox.joblib"
DECAY_PROFILES = sorted(set(SWEEP_PROFILES) - OWN_ARTIFACT)


@pytest.mark.parametrize("profile", DECAY_PROFILES)
def test_every_decay_point_pins_the_one_shared_artifact(profile):
    """`methodology_stamp` covers the adjacency and barrier blocks only, so a
    decay change leaves a neighbours artifact valid. Six identical
    preprocesses of an identical neighbourhood would be six identical answers
    at six times the cost, so all six read one file — and
    `check_methodology_stamp` is what makes that safe rather than merely
    conventional: it refuses an artifact whose adjacency or barrier differs.
    """
    assert str(load_config(profile).paths.neighbors_artifact) == SHARED_ARTIFACT


def test_the_shared_artifact_is_not_the_baseline_artifact():
    """A typo here would point the decay sweep at the PROVEN code-2025
    artifact and let a sweep run overwrite the July 2025 correctness proof."""
    assert SHARED_ARTIFACT != str(load_config(BASE).paths.neighbors_artifact)
```

Note that `decay.exponent` and `decay.scale_km` are `None` in `code-2025`, so
they appear in the moved-key diff exactly as the table states.

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest -q -W error tests/test_sweep_profiles.py`
Expected: FAIL — six profiles do not exist.

- [ ] **Step 3: Write the six profiles**

Each is Task 1's `band-1km.yaml` with `profile:` changed, the `adjacency:`
block restored to `code-2025`'s (`rule: bbox`, no `max_distance_km`), the
`decay:` block carrying the change, a `paths:` block, and a header comment
naming DEL-37 rather than DEL-36. The `paths:` block, identical in all six:

```yaml
paths:
  neighbors_artifact: colonies_neighbors_sweep-bbox.joblib
                                    # PINNED, and shared by all six decay
                                    # points: the methodology stamp covers
                                    # adjacency and barrier only, so one bbox
                                    # preprocess serves every decay point, and
                                    # check_methodology_stamp refuses the file
                                    # if that ever stops being true. It is a
                                    # DIFFERENT name from code-2025's
                                    # colonies_neighbors.joblib, so a sweep run
                                    # can never overwrite the proven artifact.
```

The six `decay:` blocks:

```yaml
  decay: {form: none, distance: centroid, distance_unit: km}
  decay: {form: inverse_power, exponent: 0.5, distance: centroid, distance_unit: km}
  decay: {form: inverse_power, exponent: 2, distance: centroid, distance_unit: km}
  decay: {form: exponential, scale_km: 2.0, distance: centroid, distance_unit: km}
  decay: {form: exponential, scale_km: 5.0, distance: centroid, distance_unit: km}
  decay: {form: inverse_linear, distance: boundary, distance_unit: km}
```

Write each expanded over multiple lines in the style of `code-2025.yaml`, with
a one-line comment on the changed key. `decay-none` and `decay-power2` carry
this comment, because their presence is otherwise unmotivated:

```
# Raj's 28 Aug steer is for forms that spread the neighbour weights UP FROM
# ZERO. `none` is that limit (every neighbour undecayed) and exponent 2 the
# steep contrast; without both ends the steer has nothing to be measured
# against. Decision log § 9.
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `uv run pytest -q -W error tests/test_sweep_profiles.py`
Expected: PASS.

- [ ] **Step 5: Register and generate**

Append the six names to **three** lists — `PROFILES` in
`scripts/generate_production_fixtures.py`, `PROFILES` in
`tests/test_production_fixtures.py`, and `SHIPPED` in `tests/test_config.py`
(Task 1 left it holding seven names and a comment saying the decay profiles
append here). Then:

```bash
uv run python scripts/generate_production_fixtures.py
git status --porcelain tests/fixtures
```
Expected: twelve new files, nothing modified. A modification to `code-2025.csv`
or `manuscript.csv` is a hard stop.

- [ ] **Step 6: Run this task's tests**

Run: `uv run pytest -q -W error tests/test_sweep_profiles.py tests/test_production_fixtures.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add delhi_psi/profiles tests/test_sweep_profiles.py \
        scripts/generate_production_fixtures.py tests/test_production_fixtures.py \
        tests/fixtures
git commit -m "feat(profiles): the six decay sweep points, one shared artifact (DEL-55)"
```

---

### Task 3: Reference cross-check for the seven unpinned constants

**Files:**
- Modify: `tests/variants.py` (`VARIANTS`)
- Regenerate: `tests/fixtures/oraculum/variants_expected_values.csv`,
  `tests/fixtures/messy/variants_expected_values.csv`

**Interfaces:**
- Consumes: the `VARIANTS` contract — a variant states each block it overrides
  IN FULL; a block it does not name keeps the `code` base's value.
- Produces: four new variant names, which
  `tests/test_variants_match_reference.py` picks up automatically.

- [ ] **Step 1: Add the four rows**

In `tests/variants.py`, after `"exp1"`, add:

```python
    # DEL-55: the decay CONSTANTS the Phase 6 sweep uses that 3D never pinned.
    # Same code paths as pow1/pow2/exp1 at different numbers, so the risk is
    # low — but the rule since 3D is that a methodology value ships as a row
    # here, scored by both implementations and compared at 1e-12.
    "decay_none": {
        "decay": {"form": "none", "distance": "centroid",
                  "distance_unit": "km"},
    },
    "pow05": {
        "decay": {"form": "inverse_power", "exponent": 0.5,
                  "distance": "centroid", "distance_unit": "km"},
    },
    "exp2": {
        "decay": {"form": "exponential", "scale_km": 2.0,
                  "distance": "centroid", "distance_unit": "km"},
    },
    "exp5": {
        "decay": {"form": "exponential", "scale_km": 5.0,
                  "distance": "centroid", "distance_unit": "km"},
    },
    # DEL-55: the sweep's three real radii. Unlike 0.25 and 0.75 km, these were
    # NOT chosen to avoid the `<=` boundary — they are the methodological
    # points DEL-36 asks for, and the Delhi run uses them as they are. Both
    # cities distinguish all three (Oraculum spans 4.0 x 3.0 km, messy
    # 21.0 x 2.4 km; undirected pairs 16/21/21 and 13/27/48), and 1 km lands
    # EXACTLY on the boundary: two pairs at 1000.000 m in Oraculum, one in
    # messy, one more at 10 km in messy. That is the point of pinning them —
    # if the two implementations ever disagree about a distance sitting
    # exactly on the radius, it surfaces here rather than on 4,357 settlements.
    "band_1km": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 1.0},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    "band_5km": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 5.0},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    "band_10km": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 10.0},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
```

Then extend the band constants below the table. `BAND_RADII_KM` and
`EXPECTED_BAND_PAIRS` / `ADDED_BAND_PAIRS` are consumed by
`scripts/check_oraculum_invariants.check_bands` and
`tests/test_variant_rules.py` — **read both before editing**, and add the three
radii with the measured pair counts:

```python
BAND_RADII_KM = (0.0, 0.25, 0.75, 1.0, 5.0, 10.0)

EXPECTED_BAND_PAIRS = {
    "oraculum": {0.0: 10, 0.25: 12, 0.75: 14, 1.0: 16, 5.0: 21, 10.0: 21},
    "messy": {0.0: 5, 0.25: 8, 0.75: 10, 1.0: 13, 5.0: 27, 10.0: 48},
}
```

`ADDED_BAND_PAIRS` needs the pairs each new radius adds over the one below it.
These were derived from the committed fixtures before this plan was revised —
use them, and re-derive only to check:

```python
    "oraculum": {
        0.25: {("A", "RV"), ("C", "RV")},          # 0.100 km each
        0.75: {("B", "D"), ("B", "IND")},          # 0.500 km each
        # BOTH pairs are at EXACTLY 1000.000 m — the `<=` boundary. See
        # test_a_pair_exactly_at_the_radius_is_a_neighbour.
        1.0: {("A", "C"), ("E", "RV")},            # 1.000 km each, ON the edge
        5.0: {("A", "IND"), ("C", "D"),            # 1.500 km each
              ("D", "IND"),                        # 2.000 km
              ("D", "RV"), ("IND", "RV")},         # 1.166190379 km each
        # Nothing: 21 pairs IS the complete graph on 7 settlements, so the
        # city cannot distinguish 5 km from 10 km. The empty set is the pin.
        10.0: set(),
    },
    "messy": {
        0.25: {("H", "L"), ("H", "T"), ("L", "S")},   # 0.131519/0.223607/0.199
        0.75: {("G", "M"), ("S", "T")},               # 0.450 / 0.630242
        # M-U is at EXACTLY 1000.000 m — the `<=` boundary.
        1.0: {("M", "U"),                             # 1.000 km, ON the edge
              ("N", "O1"), ("O2", "U")},              # 0.800 km each
        5.0: {("G", "H"), ("G", "L"), ("G", "O1"), ("G", "O2"), ("G", "T"),
              ("G", "U"), ("H", "M"), ("L", "M"), ("M", "N"), ("M", "O1"),
              ("M", "O2"), ("M", "S"), ("M", "T"), ("N", "U")},
        # I-U is at EXACTLY 10000.000 m — the `<=` boundary again.
        10.0: {("G", "N"), ("G", "S"), ("H", "N"), ("H", "O1"), ("H", "O2"),
               ("H", "U"), ("I", "N"), ("I", "O1"), ("I", "O2"),
               ("I", "U"),                            # 10.000 km, ON the edge
               ("L", "N"), ("L", "O1"), ("L", "O2"), ("L", "U"), ("N", "T"),
               ("O1", "S"), ("O1", "T"), ("O2", "S"), ("O2", "T"), ("S", "U"),
               ("T", "U")},
    },
```

Check the arithmetic before you rely on it: each city's added-pair counts must
sum with the row below to the `EXPECTED_BAND_PAIRS` total — Oraculum
10+2+2+2+5+0 = 21, messy 5+3+2+3+14+21 = 48.

- [ ] **Step 1b: Pin the `<=` boundary explicitly**

The three radii sit on ties, so the inclusive comparison stops being an
implementation detail and becomes a pinned rule. Add to
`tests/test_variant_rules.py` (or the file where `check_bands` is exercised —
follow what is already there):

```python
# Measured on the committed fixtures: these pairs are at EXACTLY the radius.
# Oraculum A-C and E-RV, messy M-U — all three at 1000.000000 m.
BOUNDARY_PAIRS_1KM = {"oraculum": 2, "messy": 1}


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_a_pair_exactly_at_the_radius_is_a_neighbour(city):
    """`within_distance` is `<=`, not `<`. 3D never had to say so — 0.25 and
    0.75 km were chosen to sit in a gap of both cities' distance lists. The
    sweep's 1 km radius cannot: it lands exactly on the boundary. Pin the
    inclusive reading, and pin how many pairs depend on it.
    """
    gdf = city.load_settlements()
    exact = [(i, j) for i, j in itertools.combinations(range(len(gdf)), 2)
             if gdf.geometry.iloc[i].distance(gdf.geometry.iloc[j]) == 1000.0]
    assert len(exact) == BOUNDARY_PAIRS_1KM[city.name]
    pairs = adjacency_pairs(gdf, rule="within_distance", max_distance_km=1.0)
    for i, j in exact:
        assert pair_of(gdf, i, j) in pairs, "a pair AT the radius must be in"
```

Adapt `adjacency_pairs` / `pair_of` to whatever the existing band tests call —
read `tests/test_variant_rules.py` first and reuse its helpers rather than
inventing new ones.

- [ ] **Step 2: Run the variant tests and watch them fail**

Run: `uv run pytest -q -W error tests/test_variants_match_reference.py tests/test_reference_impl.py -k variant`
Expected: FAIL — the committed expected-values CSVs have no rows for the four
new variants, and `test_variants_expected_values_csv_is_regenerable` reports a
byte mismatch.

- [ ] **Step 3: Record the pre-change line counts**

```bash
wc -l tests/fixtures/oraculum/variants_expected_values.csv \
      tests/fixtures/messy/variants_expected_values.csv > /tmp/variants_before.txt
cp tests/fixtures/oraculum/variants_expected_values.csv /tmp/oraculum_before.csv
cp tests/fixtures/messy/variants_expected_values.csv /tmp/messy_before.csv
```

- [ ] **Step 4: Regenerate**

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
```

- [ ] **Step 5: Prove the change is addition-only**

Do not trust the generators' own reports; diff.

```bash
diff <(sort /tmp/oraculum_before.csv) <(sort tests/fixtures/oraculum/variants_expected_values.csv) | grep '^<' | head
diff <(sort /tmp/messy_before.csv) <(sort tests/fixtures/messy/variants_expected_values.csv) | grep '^<' | head
```
Expected: **no output from either** — every line of the old file survives.
Lines beginning `<` are deletions or modifications and are a hard stop. Also
confirm `expected_values.csv` (the non-variant file) is untouched:
`git status --porcelain tests/fixtures | grep -v variants_expected` should show
no `M`.

- [ ] **Step 6: Run this task's tests**

Run: `uv run pytest -q -W error tests/test_variants_match_reference.py tests/test_reference_impl.py tests/test_variant_rules.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/variants.py tests/fixtures
git commit -m "test(variants): pin the four decay constants the sweep uses (DEL-55)"
```

---

### Task 4: The sweep runner

**Files:**
- Create: `scripts/run_sweep.py`
- Create: `tests/test_run_sweep.py`

**Interfaces:**
- Consumes: `scripts._measure_common.resolve_work_dir`;
  `delhi_psi.config.load_config`; `delhi_psi.pipeline.methodology_stamp`;
  `delhi_psi.io.read_neighbors`.
- Produces, all imported by Task 6's summariser and Task 7's run:
  - `GROUPS: dict[str, tuple[str, ...]]` — `"adjacency"`, `"bands"`,
    `"decay"`, `"all"`, in the cheapest-first order of spec § 7.
  - `plan_point(profile, work_dir) -> Point` where
    `Point = namedtuple("Point", "profile artifact stages reason")` and
    `stages` is a subset of `("preprocess", "compute")`.
  - `artifact_matches(path, cfg) -> bool`
  - `degree_report(frame, id_col) -> dict` with keys `n_links`, `deg_mean`,
    `deg_p50`, `deg_max`, `n_isolates`
  - `manifest_path(work_dir, profile) -> Path`

- [ ] **Step 1: Write the failing tests**

`tests/test_run_sweep.py`:

```python
"""The runner's decisions, without running anything.

Every expensive thing this script does is a subprocess; what is worth testing
is what it decides to run and what it refuses to touch.
"""
import json

import geopandas as gpd
import pytest
from shapely.geometry import Point as Pt

from delhi_psi.config import load_config
from scripts import run_sweep
from tests.test_sweep_profiles import SHARED_ARTIFACT


def test_a_missing_artifact_means_both_stages(tmp_path):
    point = run_sweep.plan_point("band-1km", tmp_path)
    assert point.stages == ("preprocess", "compute")
    assert "missing" in point.reason


def test_a_matching_artifact_skips_preprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(run_sweep, "artifact_matches", lambda path, cfg: True)
    (tmp_path / "colonies_neighbors_band-1km.joblib").write_bytes(b"")
    point = run_sweep.plan_point("band-1km", tmp_path)
    assert point.stages == ("compute",)


def test_the_six_decay_points_plan_one_preprocess_between_them(tmp_path,
                                                               monkeypatch):
    """The saving that § 4.2 of the spec exists for: the first decay point
    builds the shared artifact, and the other five must not rebuild it."""
    built = set()
    monkeypatch.setattr(run_sweep, "artifact_matches",
                        lambda path, cfg: path.name in built)

    stages = []
    for profile in run_sweep.GROUPS["decay"]:
        point = run_sweep.plan_point(profile, tmp_path)
        stages.append(point.stages)
        built.add(point.artifact.name)

    assert stages[0] == ("preprocess", "compute")
    assert all(s == ("compute",) for s in stages[1:])
    assert {run_sweep.plan_point(p, tmp_path).artifact.name
            for p in run_sweep.GROUPS["decay"]} == {SHARED_ARTIFACT}


def test_a_stamp_mismatch_forces_a_rebuild(tmp_path):
    """An artifact built at another radius must never be silently reused: the
    numbers would describe a neighbourhood nobody configured."""
    frame = gpd.GeoDataFrame({"USO_AREA_U": ["A"]},
                             geometry=[Pt(0, 0)], crs="EPSG:7760")
    frame.attrs["methodology"] = {
        "adjacency": {"rule": "within_distance", "max_distance_km": 5.0},
        "barrier": {"rule": "global_asymmetric", "combine": "any",
                    "buffer_m": None}}
    assert not run_sweep.artifact_matches(frame, load_config("band-1km"))
    assert run_sweep.artifact_matches(frame, load_config("band-5km"))


def test_the_work_dir_may_not_be_the_data_dir(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_sweep.resolve_work_dir(str(tmp_path / "inside"),
                                   data_dir=str(tmp_path))
    assert "bisynced" in str(exc.value)


def test_degree_report_counts_links_and_isolates():
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B", "C"],
         "nbrs_bbox": [["B", "C"], ["A"], []]},
        geometry=[Pt(0, 0), Pt(1, 1), Pt(2, 2)], crs="EPSG:7760")
    got = run_sweep.degree_report(frame, "USO_AREA_U", nbr_col="nbrs_bbox")
    assert got["n_links"] == 3
    assert got["n_isolates"] == 1
    assert got["deg_max"] == 2
    assert got["deg_mean"] == pytest.approx(1.0)


def test_a_failed_point_is_recorded_and_the_run_continues(tmp_path, monkeypatch):
    """One expensive point falling over must not cost the other ten."""
    calls = []

    def fake_stage(profile, stage, **kwargs):
        calls.append((profile, stage))
        if profile == "decay-power2":
            raise run_sweep.StageFailed(stage, 1, "boom")
        return {"seconds": 0.0}

    monkeypatch.setattr(run_sweep, "run_stage", fake_stage)
    monkeypatch.setattr(run_sweep, "artifact_matches", lambda path, cfg: True)
    run_sweep.run_group("decay", work_dir=tmp_path, data_dir=tmp_path / "data",
                        run_date="2026-09-06", commit="deadbee")

    failed = json.loads(run_sweep.manifest_path(tmp_path, "decay-power2")
                        .read_text())
    assert failed["status"] == "FAILED"
    assert failed["failed_stage"] == "compute"
    assert "boom" in failed["stderr_tail"]
    later = json.loads(run_sweep.manifest_path(tmp_path, "decay-exp2km")
                       .read_text())
    assert later["status"] == "OK"


def test_dry_run_writes_nothing(tmp_path, capsys):
    run_sweep.main(["--group", "decay", "--work-dir", str(tmp_path),
                    "--data-dir", str(tmp_path / "data"), "--dry-run"])
    assert not list(tmp_path.rglob("*.json"))
    out = capsys.readouterr().out
    assert "decay-none" in out and "preprocess" in out
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_run_sweep.py`
Expected: FAIL — `ModuleNotFoundError: scripts.run_sweep`.

- [ ] **Step 3: Write the runner**

`scripts/run_sweep.py`. Requirements, all of them load-bearing:

1. Module docstring: what a sweep point is, why the work dir may never be the
   data directory, and that this is a DRY RUN whose numbers are provisional.
2. `GROUPS` in the cheapest-first order of spec § 7:
   ```python
   DECAY = ("decay-none", "decay-power05", "decay-power2", "decay-exp2km",
            "decay-exp5km", "decay-boundary")
   ADJACENCY = ("adj-touch", "band-0km")
   BANDS = ("band-1km", "band-5km", "band-10km")
   GROUPS = {"decay": DECAY, "adjacency": ADJACENCY, "bands": BANDS,
             "all": DECAY + ADJACENCY + BANDS}
   ```
   `decay-none` leads `all` on purpose: its `bbox` preprocess warms the
   settlement dedup cache (about a quarter of every later preprocess) and
   establishes the per-link compute rate at the baseline's link count.
3. `artifact_matches(frame_or_path, cfg)` — accepts either a loaded frame or a
   path (load it with `io.read_neighbors`); returns `True` iff every key of
   `pipeline.methodology_stamp(cfg.methodology)` equals the stored value.
   Reuse the comparison shape of `pipeline.check_methodology_stamp`; do not
   duplicate its logic in a way that could drift — call it inside a
   `try/except validate.ValidationError` and return the boolean.
4. `plan_point(profile, work_dir)` → `Point(profile, artifact, stages, reason)`.
   `artifact = work_dir / load_config(profile).paths.neighbors_artifact`.
   Stages are `("preprocess", "compute")` when the artifact is missing or its
   stamp mismatches, `("compute",)` otherwise; `reason` is a short human
   string (`"artifact missing"`, `"stamp mismatch: adjacency.max_distance_km"`,
   `"artifact matches — preprocess skipped"`).
5. `run_stage(profile, stage, *, data_dir, work_dir)` — `subprocess.run` of
   `[sys.executable, "-m", "delhi_psi.cli", stage, "--config", profile,
   "--data-dir", str(data_dir), "--out-dir", str(work_dir)]`, capturing stderr,
   timing with `time.monotonic()`. On a non-zero exit raise
   `StageFailed(stage, returncode, stderr_tail)` where the tail is the last 40
   lines. Subprocesses, not in-process calls: a segfault or an OOM in one point
   must not take the runner with it, and the CLI is the interface the docs tell
   a human to use.
6. `degree_report(frame, id_col, *, nbr_col=None)` — when `nbr_col` is None,
   pick the neighbour-list column the frame actually carries (`nbrs_bbox` is
   `code-2025`'s; confirm the real column names by reading
   `delhi_psi/neighbors.py` before writing this, and fall back to the single
   column whose name starts with `nbrs_` and holds lists). Returns `n_links`
   (sum of list lengths, i.e. DIRECTED links), `deg_mean` (float),
   `deg_p50`, `deg_max`, `n_isolates`.
7. `run_group(group, *, work_dir, data_dir, run_date, commit, only=None)` —
   for each point: plan, run its stages, then, **only when this point actually
   ran `preprocess`**, load the artifact once with `io.read_neighbors` and
   compute `degree_report`. The stages are subprocesses, so the artifact is
   not in the runner's memory when `preprocess` returns — this one load is
   deliberate, it is the peak-memory moment of the cycle for `band-10km`
   (4.37 M links), and it happens exactly once per artifact.
   A point that skipped `preprocess` copies the degree summary from the
   manifest of the point that built its artifact and records
   `"degree_from": "<that profile>"`; if no such manifest exists (a re-run
   against a pre-existing artifact), the degree keys are `null` and
   `degree_from` says `"artifact predates this run"` — never silently zero,
   because a zero would read as "no links" and trip the `isolates` flag.
   Then write `manifest/<profile>.json`. On `StageFailed`, write a `FAILED`
   manifest and continue.
8. Manifest keys: `profile`, `status` (`OK`/`FAILED`), `stages_run`,
   `skip_reason`, `stamp`, `preprocess_s`, `compute_s`, `n_links`, `deg_mean`,
   `deg_p50`, `deg_max`, `n_isolates`, `degree_from`, `n_settlements`,
   `n_barrier_flagged`, `n_reported`, `n_missing_population`, `outputs`,
   `commit`, `run_date`, and on failure `failed_stage`, `returncode`,
   `stderr_tail`.

   **Where each count comes from, because three of them are not obvious.**
   `n_settlements` and `n_barrier_flagged` are fields of `PreprocessResult`;
   `n_reported` and `n_missing_population` are fields of `ComputeResult` — but
   the runner invokes the CLI as a subprocess and therefore never sees those
   objects. Before writing this, read `delhi_psi/cli.py` and find out what each
   stage prints or logs on success. If a count is not recoverable from the
   subprocess's output, recover it from the artifacts instead — `n_settlements`
   from the neighbours frame, `n_reported` from the output CSV's row count,
   `n_missing_population` from `missing_population.csv` — and say in a comment
   which route each took. **Do not invent a value and do not silently drop a
   key**: a manifest is this multi-hour run's only durable record.

9. A schema test (`test_the_manifest_carries_every_documented_key`) asserts the
   exact key set of both an `OK` and a `FAILED` manifest against a literal
   list, so a key that quietly stops being written fails a test rather than
   showing up as a blank column months later.
10. `main(argv=None)` — `--group` (required, one of `GROUPS`), `--data-dir`
   (default `~/delhi_data`), `--work-dir` (default `~/psi_sweep`), `--only`
   (repeatable, restricts to named profiles), `--dry-run`, `--run-date`
   (default: today, so a re-run is reproducible), `--log-level`. The work dir
   goes through `resolve_work_dir(..., data_dir=data_dir)`, which refuses the
   data directory and any child of it — **except under `--dry-run`, which must
   still refuse but must not create anything**. Re-export `resolve_work_dir`
   at module level so the test above can call it.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `uv run pytest -q -W error tests/test_run_sweep.py`
Expected: PASS.

- [ ] **Step 5: Prove `--dry-run` on the real profiles**

Run: `uv run python scripts/run_sweep.py --group all --work-dir /tmp/sweep_probe --dry-run`
Expected: eleven points listed, `decay-none` first with both stages, the other
five decay points with `compute` only, and nothing created under
`/tmp/sweep_probe`.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_sweep.py tests/test_run_sweep.py
git commit -m "feat(sweep): the runner — plan, cost manifest, failure isolation (DEL-55)"
```

---

### Task 5: The summariser's statistics core

**Files:**
- Create: `scripts/summarize_sweep.py` (statistics only; rendering is Task 6)
- Create: `tests/test_summarize_sweep.py`

**Interfaces:**
- Consumes: the output CSV schema — `USO_AREA_U`, `USO_FINAL`, `category`,
  `population`, `area_km2`, `<service>_count` for the six point services,
  `road_length`, `<service>_pcen`, `<service>_idx`, `unnorm_psi`, `norm_psi`.
- Produces, all imported by Task 6:
  - `SERVICES`, `AMOUNT_COLUMNS`
  - `denominator_values(frame, denominator) -> Series`
  - `own_share(frame, denominator) -> Series` (per row, pooled over services)
  - `own_only_psi(frame, denominator, *, second_normalization=True) -> Series`
  - `percentile_rank(series) -> Series`
  - `category_order(frame, psi_col) -> list[str]`
  - `kendall_tau_order(a, b) -> float`
  - `cliffs_delta(x, y) -> float`
  - `cohens_d(x, y) -> float`
  - `bootstrap_rank_intervals(frame, *, seed=0, n=1000) -> dict`
  - `bootstrap_p_greater(frame, a, b, *, seed=0, n=1000) -> float`
  - `decile_set(series, *, top, fraction=0.10) -> tuple[set, bool]` — the
    tie-inclusive set and whether it is gated
  - `decile_jaccard(a, b, *, top=True) -> float | None` — `None` when either
    side is gated; the renderer turns that into `—`
  - `spearman_rho(a, b) -> float`, `kendall_tau_b(a, b) -> float`
  - `flags(row) -> tuple[str, ...]`
  - `OUTPUT_USECOLS` — the explicit column list every CSV read passes, so a
    `band-10km` output's neighbour lists are never parsed

- [ ] **Step 1: Write the failing tests**

`tests/test_summarize_sweep.py` — every statistic against a hand-computed
answer on a small frame. Write these first, in this order:

```python
"""The sweep statistics, each against an answer computed by hand.

Nothing here reads a real output file. The point of these tests is that the
arithmetic is right; the point of the drift test in Task 6 is that the
document's numbers came from this arithmetic.
"""
import numpy as np
import pandas as pd
import pytest

from scripts import summarize_sweep as S


def frame(**cols):
    return pd.DataFrame(cols)


def test_the_popdensity_denominator_is_population_over_area():
    f = frame(population=[100.0, 50.0], area_km2=[2.0, 0.5])
    assert list(S.denominator_values(f, "popdensity")) == [50.0, 100.0]
    assert list(S.denominator_values(f, "pop")) == [100.0, 50.0]


def test_own_share_is_own_over_own_plus_neighbour():
    # one service; own = 2 over a denominator of 50 -> own_pcen 0.04.
    # pcen 0.10 means the neighbour term contributed 0.06.
    f = frame(population=[100.0], area_km2=[2.0],
              bank_count=[2.0], bank_pcen=[0.10])
    got = S.own_share(f, "popdensity", columns=[("bank_count", "bank_pcen")])
    assert got.iloc[0] == pytest.approx(0.4)


def test_own_share_of_one_when_there_is_no_neighbour_term():
    f = frame(population=[100.0], area_km2=[2.0],
              bank_count=[2.0], bank_pcen=[0.04])
    got = S.own_share(f, "popdensity", columns=[("bank_count", "bank_pcen")])
    assert got.iloc[0] == pytest.approx(1.0)


def test_an_own_share_above_one_is_refused_not_reported():
    """The load-bearing self-check of spec § 6.2: the neighbour term cannot be
    negative, so a share above 1 means the denominator reconstruction is wrong
    and every statistic built on it is wrong too."""
    f = frame(population=[100.0], area_km2=[2.0],
              bank_count=[2.0], bank_pcen=[0.01])
    with pytest.raises(ValueError, match="own_share"):
        S.own_share(f, "popdensity", columns=[("bank_count", "bank_pcen")])


def test_own_only_psi_is_minmax_of_own_counts_summed():
    # two settlements, two services, denominator 1 -> own pcens are the counts.
    # bank: [0, 4] -> idx [0, 1]; school: [3, 3] -> min == max.
    ...  # assert the constant-column case raises the same ValueError Eq. 2 does


def test_percentile_rank_is_zero_to_hundred_and_average_ties():
    got = S.percentile_rank(pd.Series([1.0, 2.0, 2.0, 4.0]))
    assert list(got) == [0.0, pytest.approx(50.0), pytest.approx(50.0), 100.0]


def test_category_order_sorts_by_mean_percentile_rank_descending():
    ...


def test_cliffs_delta_is_one_when_every_x_beats_every_y():
    assert S.cliffs_delta([3, 4, 5], [1, 2]) == 1.0
    assert S.cliffs_delta([1, 2], [3, 4, 5]) == -1.0
    assert S.cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_cliffs_delta_reads_as_a_probability():
    # (delta + 1) / 2 == P(a random x outranks a random y), ties at half
    x, y = [1.0, 3.0], [2.0, 4.0]
    assert (S.cliffs_delta(x, y) + 1) / 2 == pytest.approx(0.25)


def test_cohens_d_on_a_known_pair():
    ...


def test_kendall_tau_of_a_reversed_ordering_is_minus_one():
    order = ["Planned", "UAC", "JJC"]
    assert S.kendall_tau_order(order, order) == 1.0
    assert S.kendall_tau_order(order, list(reversed(order))) == -1.0


def test_the_bootstrap_is_seeded_and_reproducible():
    ...  # same seed twice -> identical intervals


def test_decile_jaccard_of_a_frame_with_itself_is_one():
    ...


def test_a_tie_block_at_the_cut_is_taken_whole():
    """Membership must be a property of the numbers. With values
    [0,0,0,0,0,0,0,0,1,2] and k=1, the bottom decile is all EIGHT zeros, not
    whichever one pandas happened to sort first."""
    got, gated = S.decile_set(pd.Series([0]*8 + [1, 2]), top=False)
    assert len(got) == 8


def test_an_oversized_tie_block_gates_the_cell():
    """Eight rows is 8x the decile of 1 — past 1.5x, so every statistic built
    on this set renders an em dash rather than a number measuring sort order."""
    _, gated = S.decile_set(pd.Series([0]*8 + [1, 2]), top=False)
    assert gated
    assert S.decile_jaccard(pd.Series([0]*8 + [1, 2]),
                            pd.Series([0]*8 + [1, 2]), top=False) is None


def test_the_real_baseline_shape_ties_but_does_not_gate():
    """The two real shapes, and they land on opposite sides of the gate.

    Baseline: 452 rows at norm_psi == 0 against a decile of 413. Tie-inclusive
    gives a well-defined set of 452 — 9 % oversized, under the 1.5x gate, so it
    is REPORTED. The rule that matters here is tie-inclusion, not gating: it is
    what stops nsmallest(413) from picking 413 of the 452 by sort order.
    """
    baseline = pd.Series([0.0]*452 + list(np.linspace(0.1, 1.0, 4131 - 452)))
    got, gated = S.decile_set(baseline, top=False)
    assert len(got) == 452 and not gated
    assert not S.decile_set(baseline, top=True)[1]


def test_the_own_only_anchor_shape_gates():
    """Own-only: 1,834 of 4,131 rows own nothing, so 44 % of the universe sits
    at exactly 0. A "decile" of 1,834 against k=413 is 4.4x — past the gate,
    and the cell renders an em dash."""
    anchor = pd.Series([0.0]*1834 + list(np.linspace(0.1, 1.0, 4131 - 1834)))
    got, gated = S.decile_set(anchor, top=False)
    assert len(got) == 1834 and gated


def test_kendall_tau_b_differs_from_tau_a_when_there_are_ties():
    """An implementation that silently computes tau-a passes every tie-free
    case. Hand-computed: x = [1,1,2,3], y = [1,2,2,3].
    Pairs: 6 total; concordant 4, discordant 0, 1 tied in x only,
    1 tied in y only -> tau_b = 4 / sqrt(5 * 5) = 0.8, while tau_a = 4/6.
    """
    got = S.kendall_tau_b(pd.Series([1, 1, 2, 3]), pd.Series([1, 2, 2, 3]))
    assert got == pytest.approx(0.8)
    assert got != pytest.approx(4 / 6)


def test_spearman_rho_averages_tied_ranks():
    ...


def test_flags_fire_on_the_documented_conditions():
    assert "smoothed" in S.flags({"own_share_p50": 0.05, ...})
    assert "isolates" in S.flags({"n_isolates": 3, ...})
```

Fill in every `...` with real arithmetic and a real assertion — a test that
asserts nothing is a defect, not a placeholder. Where a case needs a
hand-computed constant, compute it and write it as a literal with a comment
showing the arithmetic, in the style of `tests/test_index.py`.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_summarize_sweep.py`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the statistics**

Rules that the tests above do not fully pin, and that matter:

- `own_share` pools over the **seven** amount columns (six point services plus
  `road_length`) by summing own_pcen and summing pcen across services, then
  dividing — not by averaging per-service shares, which would weight a
  settlement's rarest service equally with its commonest.
- **0/0 is `NaN`, not 0.** A settlement that owns nothing and receives nothing
  has no share; 1,834 of the baseline's 4,131 reported settlements own zero of
  all seven services, so this is the common case. Return `NaN` for those rows,
  exclude them from the median, and return the count alongside as
  `n_own_share_undef`. Reading them as 0 would drag `own_share_p50` toward the
  `smoothed` flag on every point.
- The guard is `own_share <= 1 + 1e-9` on the non-NaN rows; raise `ValueError`
  naming the worst offending row and its share. Note in the docstring what the
  guard does *not* prove: the denominator cancels in the ratio, so it catches a
  mis-shaped reconstruction, not a wrong denominator. What pins the denominator
  is `test_own_share_of_one_when_there_is_no_neighbour_term`.
- **Decile sets are tie-inclusive and gated.** `decile_set(series, *, top,
  fraction=0.10)` returns every row tied with the k-th value, so membership is
  a property of the numbers and not of pandas' sort order — the bottom-decile
  cut on this data falls inside a 452-row tie block at the baseline and an
  1,834-row block at the own-only anchor, against a decile of 413. When the
  returned set exceeds `1.5 * k`, the set is still returned but
  `decile_is_gated(...)` is True and every statistic built on it renders `—`.
  This governs `jaccard_top10`, `jaccard_bottom10`, and § 6.5's three
  `*_decile_share_*` cells alike. A test must pin BOTH halves: that a tie block
  is taken whole, and that an oversized one gates.
- **No scipy.** `spearman_rho(a, b)` is Pearson on average-tied ranks
  (`Series.rank()` then `numpy.corrcoef`). `kendall_tau_b(a, b)` uses the
  standard formula `(C - D) / sqrt((n0 - n1) * (n0 - n2))`, computed by
  broadcasting in chunks (4,131 rows is a 17 M-pair comparison; chunk it at
  ~1,000 rows and accumulate, keeping the intermediate as `int8`). Test τ-b
  against at least one hand-computed case WITH ties — an implementation that
  silently uses τ-a passes every tie-free case.
- `own_only_psi` reuses Eq. 2's min-max and must fail the same way
  `delhi_psi.index` does on a constant column, with the same explanation —
  import and call the production helper rather than re-deriving it if one is
  exposed; if it is not, replicate it and say so in a comment.
- `bootstrap_*` use `numpy.random.default_rng(seed)`, resample settlement rows
  **stratified by category** (resample within each category to its own size),
  and are vectorised over the 1,000 draws. Seed 0, n=1000, both stated in the
  rendered caption.
- `kendall_tau_order` compares two orderings of the same category set; if the
  sets differ (a category absent from a run), restrict to the intersection and
  record how many were dropped.
- No statistic on the § 6.9 forbidden list is implemented. Not "implemented but
  unused" — absent.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `uv run pytest -q -W error tests/test_summarize_sweep.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/summarize_sweep.py tests/test_summarize_sweep.py
git commit -m "feat(sweep): rank-based sweep statistics (DEL-55)"
```

No `pyproject.toml`, no `uv.lock`: this task adds no dependency.

---

### Task 6: Rendering, the document skeleton, and the drift test

**Files:**
- Modify: `scripts/summarize_sweep.py` (the `main`, the three blocks)
- Modify: `tests/test_summarize_sweep.py` (round-trip + drift tests)
- Create: `docs/data/phase6_sweep.md`

**Interfaces:**
- Consumes: `scripts._measure_common.render`, `parse_block`, `FENCE`;
  `tests/test_measure_common.assert_prose_numbers_come_from_the_blocks` and
  `needs_measure_cache` — read `tests/test_measure_rule_effects.py` first and
  follow its shape exactly; this document is the fourth of its kind.
- Produces: `BLOCKS = ("points", "ordering", "gap", "denominator_check")` and
  `main(argv=None)`.

- [ ] **Step 1: Write the failing round-trip test**

```python
def test_every_block_round_trips_through_the_parser():
    report = {"point": "band-1km", "n_reported": 4131, "own_share_p50": 0.412}
    text = render(report, name="points")
    assert parse_block(text, name="points") == {k: str(v)
                                                for k, v in report.items()}


def test_the_document_carries_every_block(...):
    doc = DOC.read_text()
    for name in summarize_sweep.BLOCKS:
        parse_block(doc, name=name)


def test_every_caption_says_the_run_is_provisional():
    """Spec § 6.9: no number from this run may reach the manuscript, and the
    only defence against that is that the document says so at every table."""
    doc = DOC.read_text()
    for heading in re.findall(r"^## .*$", doc, re.M):
        ...  # the section that follows each table heading contains the label
    assert doc.count("DRY RUN") >= len(summarize_sweep.BLOCKS)
```

- [ ] **Step 2: Run and watch it fail**

Run: `uv run pytest -q -W error tests/test_summarize_sweep.py`
Expected: FAIL — no `BLOCKS`, no document.

- [ ] **Step 3: Implement rendering and `main`**

`main(argv)` takes `--work-dir`, `--baseline-dir` (default
`~/delhi_data/phase3_verify`), `--out` (default
`docs/data/phase6_sweep.md`; when absent, print to stdout), `--block` (render
one block only). It reads each manifest and its output CSV, adds the baseline
point (reading `delhi_psi_code-2025_popdensity_2020.csv` and, for the
denominator check only, the `pop` one) and the derived own-only anchor, and
prints the four blocks in order.

Multi-row blocks render as one `render(...)` call per row with the row's
`point` as the first key, all under one block label — follow whatever
`measure_psi_columns.py` does for its multi-row case; if it has none, use one
labelled block per row named `points:<profile>` and say why in a comment.

The baseline point reads its structure from
`<baseline-dir>/colonies_neighbors.joblib` (9.5 MB, read-only) and writes `—`
for `preprocess_s` / `compute_s`. The own-only anchor writes `—` for every
structure column and `1.000` for `own_share_p50`.

- [ ] **Step 4: Write the document skeleton**

`docs/data/phase6_sweep.md`, in the shape of `docs/data/rule_effects.md`:
title, the provenance paragraph (run date, inputs, commit, exact command), the
"what the run must show, stated before it runs" paragraph, then one `##`
section per block with its caption and its fenced block. Until Task 7 runs, the
blocks carry the values a `--dry-run`-scale invocation produces or are marked
`pending` — but the file must parse, and every caption must already carry:

```
**DRY RUN on `code-2025` — superseded by the ratified profile.** The frozen
July 2025 rule set still carries `bbox` adjacency, `global_asymmetric`
barriers and decayed roads, all three of which Raj's 28 Aug 2026 decisions
change. No number in this document is quotable, and none of it belongs in the
manuscript. Phase 6's reported variants (DEL-36/37/39) are the same sweep
re-run against the ratified profile (DEL-31).
```

State in the prose, once: PSI levels are not comparable across points, which
is why every cross-point statistic here is rank-based; the bootstrap treats a
census as a sample and is therefore a composition-sensitivity device; the
formal/informal pooling is `{Planned, SDA}` vs `{JJC, JJR}`, an assumption of
this document and not a ruling from the paper.

- [ ] **Step 5: Run this task's tests**

Run: `uv run pytest -q -W error tests/test_summarize_sweep.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/summarize_sweep.py tests/test_summarize_sweep.py docs/data/phase6_sweep.md
git commit -m "feat(sweep): rendering and the provisional sweep document (DEL-55)"
```

---

### Task 7: The real run (controller-only)

**This task is NOT dispatched to an implementer.** It runs for hours, it is
the thing background-completion notifications reach only the controller for,
and its rulings (spec § 7) are the controller's.

- [ ] **Step 1: Dry run the plan**

```bash
uv run python scripts/run_sweep.py --group all --dry-run
```

- [ ] **Step 2: Run the decay group** (background)

```bash
uv run python scripts/run_sweep.py --group decay --work-dir ~/psi_sweep
```
Record the per-link compute rate from `decay-none`'s manifest.

- [ ] **Step 3: Run the adjacency group** (background)

- [ ] **Step 4: Run `band-1km`, then `band-5km`** (background, in that order)

After `band-5km`, project the 10 km compute from its link count. Apply spec
§ 7's ruling: run it last and alone; kill only past 24 hours, and record the
extrapolation either way.

- [ ] **Step 5: Run `band-10km`** (background, alone)

- [ ] **Step 6: Summarise**

```bash
uv run python scripts/summarize_sweep.py --work-dir ~/psi_sweep \
    --out docs/data/phase6_sweep.md
```
Then write the prose findings around the blocks: what the ordering does across
the sweep, whether Planned > JJC survives every point, which points flagged and
why, and the measured cost table for DEL-31's budget.

- [ ] **Step 7: Verify the baseline did not move**

```bash
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3_verify
```
Expected: PASS, 60 comparisons at `0.000e+00`. The sweep must not have touched
it; this is the proof that it did not.

- [ ] **Step 8: Full suite, CHANGELOG, PR**

```bash
uv run pytest -q -W error
```
Update `CHANGELOG.md` `[Unreleased]`, including any deviation from this plan.
Then the final whole-branch review, the PR, and the merge.

---

## Self-review

**Spec coverage.** D1→T1/T2, D2→T1, D3→T1/T2, D4→T3, D5→T4, D6→T5/T6,
D7→T4/T5/T6, D8→T6/T7, D9→T7. Spec § 6.3–6.7 statistics → T5's interface list;
§ 6.8's dropped statistics are absent from that list by construction; § 6.9's
forbidden list is enforced by T5 step 3's last rule and T6's caption test.
§ 7's cost ordering → `GROUPS` order in T4 and the step order in T7. § 8's hard
stops → T1 step 6, T2 step 5, T3 step 5, T5's `own_share` guard, T7 step 7.

**Placeholders.** T5 step 1 contains `...` markers *by design*, with an
explicit instruction to fill each with real arithmetic; every other step
carries its content. T6 step 1's third test is a sketch for the same reason.
These are the two places where the exact constant depends on a frame the
implementer builds, and both say so.

**Type consistency.** `plan_point` returns `Point(profile, artifact, stages,
reason)` in T4's interface block and is used with those four fields in T4's
tests and T6's rendering. `SHARED_ARTIFACT` is defined in T2 and imported by
T4's test. `own_share(frame, denominator, columns=...)` takes the same three
arguments in every test and in T6's use. `degree_report(frame, id_col,
nbr_col=None)` — T4's test passes `nbr_col`, the runner does not.
`decile_set` returns `(set, gated)` everywhere it appears, and
`decile_jaccard` returns `float | None` in T5's interface, its tests, and T6's
renderer.

**Revised after the plan review** (spec § 12): Task 1 gained
`tests/test_config.py` and the `docs/methodology-config.md` § 3 fix; Task 1's
guard gained the `categories` comparison; Task 3 grew from four variant rows to
seven and gained the `<=` boundary pin; Task 4's `degree_report` no longer
claims the artifact is in memory and gained `degree_from`, the manifest key
provenance and a schema test; Task 5 dropped scipy, made `own_share` NaN-aware,
and gained the tie-inclusive gated decile rule; the CSV reads gained
`OUTPUT_USECOLS`.
