# Messy-City Fixture Tier (DEL-24) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second fixture city that carries every real-layer pathology Oraculum omits by construction — non-rectangular polygons, a MultiPolygon whose centroid falls outside it, an overlapping pair, an isolated settlement, a settlement with no population row, an area-extreme sliver — score it with the independent reference implementation, and pin what production does on each one today, so the DEL-19/DEL-20 fixes are proven by a test that flips rather than by argument.

**Architecture:** A `City` abstraction (`tests/cities.py`) with two instances, `ORACULUM` and `MESSY`. The reference implementation is **generalised only** — parameterised over a city, taking an explicit scenario table, summing every road row; not one rule changes. A new deterministic generator (`scripts/generate_messy_fixtures.py`) writes the messy GeoJSON layers, has the reference emit their `expected_values.csv`, runs the invariants guard on the result and refuses to write a fixture that fails it. Every existing proof — reference match, production fixtures, invariants guard — is parametrised over both cities. A data-gated measurement script gives the tier's real-data premises a reproducible source. **No production code changes at all in this cycle.**

**Tech Stack:** Python 3.13 / uv, hatchling, geopandas 1.1, shapely 2.1, pandas 3.0, pyproj, joblib, tqdm, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-messy-city-tier-design.md` (rev 3, approved by owner 2026-08-28 — read it in full first; §§ 1–8 are the authority, § 2 fixes the city, § 3 fixes the abstraction and the compat views, § 4 fixes the proofs, § 5 the measurement script, § 6 the tests and docs). Parents: `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` § 4, `docs/superpowers/specs/2026-08-27-phase3b-categories-design.md`, `docs/superpowers/specs/2026-08-24-ci-workflow-design.md` § "Robustness to oracle changes".

## Global Constraints

State each of these to yourself before every task; they are every task's requirements, implicitly.

- **Oraculum's `expected_values.csv` and both Oraculum production CSVs must be byte-identical to today.** `tests/fixtures/oraculum/expected_values.csv`, `tests/fixtures/oraculum/production/code-2025.csv` and `tests/fixtures/oraculum/production/manuscript.csv` must be unchanged at the end of every task. The reference change is a no-op there and the messy city is a new directory. Prove it in every task that touches `tests/`, `scripts/` or `delhi_psi/` by running

  ```bash
  for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
  git status --porcelain -- tests/fixtures/
  ```

  and requiring the `git status` output to be **empty** (once the messy fixtures are committed in Task 3; before that, the only acceptable output is the untracked messy files that task itself creates and commits). Any modified, deleted **or untracked** file under `tests/fixtures/` is a failure — that is exactly what the CI drift guard checks.
- **The real-data baseline must stay unchanged, and there are NO production code changes at all in this cycle.** Nothing under `delhi_psi/` is edited by any task in this plan. `scripts/verify_against_baseline.py --config code-2025` must still report `PASS — new run equivalent to July 2025 baseline within tolerance` with every `max abs deviation` line reading `0.000e+00` (hand-run in Task 7; it can only be a no-op, because no production module is touched). If a task finds itself needing to edit `delhi_psi/`, that is a **stop condition** (spec § 8) — report it, do not work around it.
- **No carried-over test may have its EXPECTED VALUE changed.** The suite is **281 tests** green under `uv run pytest -q -W error` today; it only ever grows. The only permitted edits to carried-over tests and their helpers are the *scaffold wiring* changes the spec §§ 3, 6 list, enumerated once here:
  1. `tests/reference_impl.py` — `compute_city` gains keyword-only `scenarios=None`; `SCENARIOS` becomes the literal view over `ORACULUM.scenarios`; `_service_amounts` sums every road row; `emit_expected_values` gains `city=ORACULUM`; `__main__` writes both cities (Task 2).
  2. `tests/oraculum_fixtures.py` — every public name becomes a thin wrapper taking `city=ORACULUM`; `ORACLE_SCENARIOS` / `ORACLE_VOCABULARY` / `ORACLE_SCHEME` become views over `ORACULUM` (Task 2).
  3. `scripts/render_oracle_maps.py` — the `SCENARIOS.setdefault(...)` mutation becomes a module-level `MAP_SCENARIOS` table passed explicitly (Task 2).
  4. `scripts/check_oraculum_invariants.py` — `expected_values_path(city)`, `check(df=None, *, city=ORACULUM)`, `emit_checked_expected_values(city, out_path)`, `__main__` over `CITIES` (Task 3).
  5. `scripts/generate_oraculum_fixtures.py` — a final "emit and check the expected-values CSV" step (Task 3).
  6. `scripts/generate_production_fixtures.py` — `production_dir(city)`, `emit_profile(profile, out_path, city=ORACULUM)`, `main()` over `CITIES × PROFILES` (Task 4).
  7. `tests/test_production_fixtures.py` — stacked `parametrize("city", CITIES)` × `parametrize("profile", PROFILES)`; `PRODUCTION_DIR` becomes `production_dir(city)` and the repo root comes from the script's `REPO` (Task 4).
  8. `tests/test_profiles_match_reference.py` — **only** `test_profile_matches_reference` is parametrised over `CITIES`, taking its scenario list from `city.scenarios`; the enum-coverage, every-knob and stage-mapping tests stay Oraculum-only as written (Task 4).
  9. `tests/test_reference_impl.py` — `test_expected_values_csv_is_regenerable` and `test_invariants_guard_csv_wide` are parametrised over `CITIES`; new tests are appended for the three generalisations (Tasks 2, 4).
  Every assertion and every tolerance in those files stays exactly as it is. Anything beyond this list is a **stop condition**.
- **The reference INDEPENDENCE RULE holds.** `tests/reference_impl.py` must never import, call, or mirror the production spatial-index library. `tests/cities.py` is fixture plumbing, not index math: it imports `geopandas` and the standard library and **nothing else from this repo** — never `delhi_psi`, never `tests.reference_impl`. A test pins this by reading the module source.
- **Fixture generators write only under `tests/fixtures/`** and are named `scripts/generate_*_fixtures.py` (CI spec § Robustness — the drift step globs that pattern). `scripts/generate_messy_fixtures.py` writes `tests/fixtures/messy/{settlements,services,barriers}.geojson` and `tests/fixtures/messy/expected_values.csv`; `scripts/generate_production_fixtures.py` writes `tests/fixtures/<city>/production/<profile>.csv`. Nothing else, anywhere else.
- **The measurement script never writes under `data_dir`.** `~/delhi_data` is bisynced to the shared drive (memory note "delhi_data sync setup"), so a stray file there propagates. `scripts/measure_layer_pathologies.py` puts its dedup cache in `--cache-dir` (default: a fresh temporary directory) and opens everything under `--data-dir` read-only.
- **Tests that need the real Delhi data must skip when it is absent** (`pytest.mark.skipif` on `~/delhi_data` not resolving). CI runs on a bare runner.
- Fixture GeoJSON is written with `json.dumps(..., indent=1, sort_keys=True)` (never a GDAL driver) so files stay human-readable and diff-stable; loaders re-apply the CRS on read. Running a generator twice produces byte-identical files.
- Branch: `del-24-messy-city` (off `origin/main` at b2e8e51; HEAD 2e612d8). Never `git add -A`, never `git commit -a` — every commit names its files (review agents may be running: memory note "Review agents: isolate worktree").
- After **every** task: `uv run pytest -q -W error` must be green.
- `~/delhi_data` is read-only. All test IO under pytest `tmp_path` / `tmp_path_factory` or the repo.
- Numeric tolerance: `abs=1e-12` for in-memory comparisons; exact `==` only where this plan says a value is exact (it has been verified — see "Canonical facts").
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Stop and report (spec § 8) — do not work around, do not tune away — if: any Oraculum fixture changes; the real-data baseline deviates; a carried-over test needs an expected value changed; or the messy city's reference and production disagree on any scenario (that would be a real divergence, to be reported).

## File Structure

```
tests/
  cities.py                     NEW — Scenario, City, ORACULUM, MESSY, CITIES   (Task 1)
  test_cities.py                NEW — the abstraction + the compat views        (Tasks 1-3)
  reference_impl.py             scenarios=, SCENARIOS view, city=, road sum     (Task 2)
  oraculum_fixtures.py          thin city= wrappers over tests.cities           (Task 2)
  test_reference_impl.py        3 new generalisation tests; 2 parametrised      (Tasks 2, 4)
  test_production_fixtures.py   stacked city × profile parametrisation          (Task 4)
  test_profiles_match_reference.py  test_profile_matches_reference over CITIES  (Task 4)
  test_messy_fixtures.py        NEW — every spec § 4.3 pin                      (Task 5)
  test_layer_pathologies.py     NEW — data-gated provenance check               (Task 6)
  fixtures/messy/               NEW — settlements/services/barriers.geojson,
                                expected_values.csv, production/*.csv       (Tasks 3, 4)
scripts/
  generate_messy_fixtures.py    NEW — the eleven settlements + self-checks      (Task 3)
  generate_oraculum_fixtures.py final expected-values step                      (Task 3)
  check_oraculum_invariants.py  per-city paths + emit_checked_expected_values   (Task 3)
  generate_production_fixtures.py  production_dir(city), CITIES × PROFILES      (Task 4)
  render_oracle_maps.py         MAP_SCENARIOS instead of mutating the global    (Task 2)
  measure_layer_pathologies.py  NEW — real-layer provenance                     (Task 6)
docs/
  oracle/messy-city.md          NEW                                             (Task 7)
  data/layer_pathologies.md     NEW — hand-run output                           (Task 6)
  methodology-config.md         § 4 one line                                    (Task 7)
CHANGELOG.md, WORKPLAN.md                                                       (Task 7)
```

`delhi_psi/` does not appear in this list. That is deliberate and load-bearing (Global Constraints).

## Canonical facts (verified against the repo on 2026-08-28 — do not re-derive)

Everything in this section was checked by running it. Do not re-derive it; if
something here disagrees with what you observe, **stop and report**.

- `uv run pytest -q -W error` reports **281 passed** at HEAD 2e612d8.
- **Both `bbox` implementations are directed the same way:** `j ∈ nbrs(i)` iff `geom_i ∩ envelope_j ≠ ∅`. Production `neighbors._adjacency_bbox` sjoins the polygons (left) against the bounding boxes (right) with the default `intersects` predicate; the reference tests `idx[i].intersects(box(*idx[j].bounds))`. Every directed pin below is written in that direction.
- **The messy city as verified.** Coordinates are EPSG:7760 metre **offsets from `BASE_X = BASE_Y = 1_000_000`**, exactly as Oraculum's generator writes them. Polygon rings are listed counter-/clockwise as given; all eleven geometries are `is_valid`.

  | id | type | pop | `area_km2` | geometry (offsets from BASE) |
  |---|---|---|---|---|
  | `H` | Planned | 110 | 1.33 | hexagon `(400,2400) (1600,2300) (2200,1600) (1900,1000) (1100,1000) (900,2200)` |
  | `L` | Planned | 200 | 2.56 | L-shape `(0,0) (2000,0) (2000,800) (800,800) (800,2000) (0,2000)` |
  | `T` | Planned | 300 | 0.25 | triangle `(2000,800) (3000,200) (3000,700)` |
  | `M` | Planned | 400 | 2.0 | MultiPolygon: boxes `(5000,0,6000,1000)` and `(7000,0,8000,1000)` |
  | `G` | Planned | 50 | 0.01 | box `(6450,450,6550,550)` |
  | `O1` | Planned | 600 | 1.0 | box `(10000,0,11000,1000)` |
  | `O2` | Planned | 700 | 1.0 | box `(10800,0,11800,1000)` |
  | `I` | Planned | 800 | 1.0 | box `(20000,0,21000,1000)` |
  | `N` | **RV** | 900 | 1.0 | box `(11800,0,12800,1000)` |
  | `U` | Planned | **null** | 1.0 | box `(9000,0,10000,1000)` |
  | `S` | Planned | 100 | **2e-06** | box `(1400,999,1402,1000)` |

  Shapely's areas match the declared `area_km2` exactly (`H` 1 330 000 m², `L` 2 560 000 m², `T` 250 000 m², `M` 2 × 1 000 000 m², `G` 10 000 m², `S` **2.0 m²**). All ten non-null populations are distinct.
- **Verified centroids** (offsets from BASE): `H (1463.659148…, 1667.418546…)`, `L (775, 775)`, `T (2666.666…, 566.666…)`, `M (6500, 500)`, `G (6500, 500)`, `O1 (10500, 500)`, `O2 (11300, 500)`, `I (20500, 500)`, `N (12300, 500)`, `U (9500, 500)`, `S (1401, 999.5)`.
  `M.centroid.distance(G.centroid)` is **exactly `0.0`** — production's `nbrs_dist_bbox` for `G` is exactly `[("M", 0.0)]`, so the decay weight is exactly `1`.
- **Verified directed neighbour tables** (production, both rules, scenario `nopop_only`):

  | id | `bbox` (`code-2025`) | `touch` (`manuscript`) |
  |---|---|---|
  | `H` | `L, S` | `S` |
  | `L` | `H, T` | — |
  | `T` | `L` | — |
  | `M` | — | — |
  | `G` | `M` | — |
  | `O1` | `O2, U` | `O2, U` |
  | `O2` | `N, O1` | `N, O1` |
  | `I` | — | — |
  | `N` | `O2` | `O2` |
  | `U` | `O1` | `O1` |
  | `S` | `H, L` | `H` |

  `H ∩ L` is empty; `geom_H ∩ envelope_L ≠ ∅` **and** `geom_L ∩ envelope_H ≠ ∅` (`L`'s vertical arm reaches back under `H`'s envelope at `(400, 1500)`). `T ∩ L` is the single **Point** `(2000, 800)`, length `0`. `O1 ∩ O2` is a polygon of area 200 000 m² and perimeter 2 400 m. `I` is disjoint from every other settlement and no envelope reaches it.
  **`L` has no `touch` neighbours either** — see "Spec ambiguities resolved" below.
- **Services** (all strictly `within` exactly the hosts named, so `within` and `intersects` agree except on the deliberate overlap point):
  - clinic ×7: `H (1500,1500)`, `L (400,400)`, `T (2800,550)`, `M (5500,500)`, `G (6500,500)`, **`O1∩O2` `(10900,500)`**, `I (20500,500)`
  - school ×7: `L (600,200)`, `M (7500,500)`, `O2 (11400,500)`, `N (12300,500)`, `S (1401.5,999.5)`, `T (2900,450)`, `G (6480,480)`
  - bank ×2: `H (1600,1400)`, `I (20600,600)`
  - police ×2: `L (200,600)`, `O1 (10400,500)`
  - ration ×2: `M (5600,600)`, `S (1400.5,999.5)`
  - transport ×2: `H (1700,1300)`, `N (12500,400)`
  - road ×**2** LineStrings: `[(1500,200),(1500,2200)]` and `[(4800,800),(8200,800)]`
- **Verified road lengths (km), summing every road row:** `H 1.2`, `L 0.6`, `M 2.0`, all others `0.0`. Using only the **first** road row (today's `_service_amounts`) gives `M 0.0` — so the messy city is what makes the "sum all road rows" generalisation load-bearing.
- **Verified agreement.** With the generalised reference (`scenarios=` table, summed road rows) and the production `compute_frames` driven by a derived config over vocabulary `("Planned", "RV")`, scheme `messy-2`: for both profiles (`code-2025`→`code`, `manuscript`→`ideal`) × all three scenarios × both denominators × every mapped metric, the **worst absolute deviation is 0.0** when both are fed the in-memory frames, and **2.22e-16** when both are fed the committed GeoJSON files. Both are inside `abs=1e-12`.
- **Verified invariants.** `scripts.check_oraculum_invariants.check(df)` on the messy reference output returns `[]` — no degenerate min-max group and no tied clinic/school argmin/argmax, in any of the 2 rules × 3 scenarios × 2 denominators × 7 `_pcen` metrics = 84 groups. The messy `expected_values.csv` has **2520** rows.
- **Verified reported id sets:** `nopop_only` → `G H I L M N O1 O2 S T` (10 rows); `excl_rv_post` and `excl_rv_pre` → `G H I L M O1 O2 S T` (9 rows). `U` is absent under every profile, scenario and denominator.
- **Verified exact values** (production, `code-2025` unless stated):
  - `G.clinic_pcen` under `pop` is exactly `(1 + 1) / 50 == 0.04`; under `popdensity` exactly `(1 + 1) / (50 / 0.01) == 0.0004`. Exact `==` holds because the decay weight is exactly 1.
  - `I.clinic_pcen` is exactly `1 / 800 == 0.00125` under both rules (no neighbours).
  - `O1.clinic_count == 1` and `O2.clinic_count == 1` — the same physical clinic, counted for both (DEL-20).
  - Under `popdensity`, the ration owners are exactly `{M, S}` with `M 0.005` and `S 2e-08`; the ratio is `250000` (5.40 orders of magnitude). Rows with **exactly** `0.0` ration PCEN under `code-2025`: `L T O1 O2 I N` (under `manuscript` also `G`). Under `pop` the order flips (`M 0.0025 < S 0.01`) — which is why the spec claims no ordering there.
  - `compute_frames(..., missing_population="error")` raises `ValidationError` with the message `1 settlements have no population row and layers.population.missing is 'error': ['U']`.
- **Both reference generalisations are byte no-ops for Oraculum**, verified by regenerating with them applied: `expected_values.csv` is byte-identical with the summed-road `_service_amounts`, and `production/code-2025.csv` and `production/manuscript.csv` are byte-identical when the scenario list is reordered into `SCENARIOS` order (`write_fixture` sorts by `(scenario, denom, settlement, metric)` before writing, so list order cannot reach the bytes).
- **An empty GeoJSON `FeatureCollection` round-trips.** `{"type": "FeatureCollection", "features": []}` reads back as a 0-row GeoDataFrame with a `geometry` column and CRS `EPSG:4326`; `set_crs(epsg=7760, allow_override=True)` then works. `compute_frames(..., {"canal": <that frame>}, ...)` runs: `geometry.barrier_flags` adds an all-False `canal` column, `neighbors.combine_barrier_flags` ORs it into `barrier`, and `neighbors.apply_barrier` returns early on the empty geometry list. The reference's `apply_barrier` returns `nbrs` unchanged for `len(barriers) == 0`. So the messy city needs no special-casing anywhere.
- `~/delhi_data` exists on this machine. `delhi_psi.pipeline._dedup_cached` on the real settlement layer (4 357 rows, O(n²)) takes **≈2.5–3 minutes** on a cold cache — which is why `tests/test_layer_pathologies.py` runs the measurement script **once**, from a module-scoped fixture.
- `PROFILE_RULES` is `{"code-2025": "code", "manuscript": "ideal"}`; `manuscript` has `second_normalization: false` (no `norm_psi`), `exclusion.types: []`, `adjacency.rule: touch`, `roads: eq4_own_only`, `absent_neighbor: contributes`. `code-2025` has `adjacency.rule: bbox`, `roads: decayed`, `second_normalization: true`, `absent_neighbor: swallowed`. Both are compatible with the messy vocabulary (`manuscript` excludes nothing; `code-2025` excludes the category `RV`, which the messy vocabulary produces).

## Spec ambiguities resolved

Resolved while verifying the coordinates; recorded here so a reviewer can
re-open them deliberately.

1. **`L` also has no `touch` neighbours.** Spec § 2 says the schools in `T` and `G` exist "so no three-way tied zero under `touch`, where `T`, `G` and `I` have no neighbours". In the verified layout `L` has no `touch` neighbours either — `H ∩ L = ∅` is *required* by the spec, `T ∩ L` is a point, and `S` is required to sit against `H`. That is harmless and does **not** re-open the tie the sentence is about: a settlement's school PCEN is `0` only when it owns no school *and* has no serving neighbour, and `L` owns a school. Verified: under `touch`, `I` is the unique school argmin and `G` the unique argmax in every scenario and denominator, and the invariants guard returns `[]`.
2. **Who writes the messy production CSVs.** § 2 lists `production/<profile>.csv` among the messy fixture directory's files "(generator-emitted)", while § 3 gives `generate_production_fixtures.py` a `main()` over `CITIES × PROFILES`. Resolved in favour of § 3: `generate_messy_fixtures.py` writes the three GeoJSONs and `expected_values.csv`; `generate_production_fixtures.py` writes **both** cities' production CSVs. Both scripts match the CI drift glob, and the glob's alphabetical order (`generate_messy_…`, `generate_oraculum_…`, `generate_production_…`) already runs the geometry generators before the production one.
3. **Where the "emit, check, then move into place" helper lives.** The spec requires both generators to run `check_oraculum_invariants.check` before writing, but does not say where the shared step lives. Resolved: `emit_checked_expected_values(city, out_path)` in `scripts/check_oraculum_invariants.py` — that module already owns "this CSV is valid"; it now also owns "only write a valid one". No new module, and the reference import stays lazy inside the function.
4. **`City.fixtures` is a real field, not a derived property.** Spec § 3 lists `fixtures: Path` as a dataclass field, so it is one, constructed explicitly per instance; `tests/test_cities.py` pins `city.fixtures == FIXTURES_ROOT / city.name` so the two cannot drift.
5. **What "compares counts only" means for the pathologies doc.** Spec § 5's key list includes three float-valued keys (`area_km2_min/median/max`). The data-gated test compares every **integer** key exactly and asserts the three area keys parse as floats, rather than comparing float text. The prose header (date, layer file, commit) is never compared, as the spec requires.

---

### Task 1: `tests/cities.py` — the `City` abstraction (spec § 3)

Declares both cities. Only Oraculum's fixture files exist at this point, so
the file-backed tests are parametrised over a `FIXTURED` tuple that Task 3
widens to `CITIES` in one line — the messy declaration is written now
because Task 3's generator needs `MESSY` to emit its CSV.

**Files:**
- Create: `tests/cities.py`
- Create: `tests/test_cities.py`

**Interfaces:**
- Consumes: `geopandas` only (INDEPENDENCE RULE — no `delhi_psi`, no `tests.reference_impl`); in the *test*, `delhi_psi.pipeline.attach_population` / `excluded_ids` and `delhi_psi.categories.apply_mapping` (all existing).
- Produces:
  - `tests.cities.FIXTURES_ROOT: Path` = `tests/fixtures`
  - `tests.cities.DEFAULT_EPSG: int` = `7760`
  - `tests.cities.Scenario` — frozen dataclass, fields in order: `name: str`, `dropped: frozenset[str]`, `dropped_before_neighbors: bool`, `exclusion_types: tuple[str, ...]`, `stage: str`
  - `tests.cities.City` — frozen dataclass, fields in order: `name: str`, `fixtures: Path`, `scheme: str`, `vocabulary: tuple[str, ...]`, `scenarios: tuple[Scenario, ...]`, `epsg: int = DEFAULT_EPSG`; methods `mapping() -> dict[str, str]`, `load_settlements()`, `load_barriers()`, `load_services() -> dict[str, GeoDataFrame]`
  - `tests.cities.ORACULUM: City`, `tests.cities.MESSY: City`, `tests.cities.CITIES: tuple[City, City]` = `(ORACULUM, MESSY)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cities.py`:

```python
"""The two fixture cities and the compat views over them (spec 3C § 3).

`tests/cities.py` is fixture plumbing: it says where a city's files are, what
its vocabulary is, and which scenarios it is scored under. It must never
reach into `delhi_psi` (the reference implementation imports it, and the
INDEPENDENCE RULE forbids the reference from seeing production code) nor into
`tests.reference_impl` (which imports it).

The scenario pin below is the one that keeps the two sides honest: the
reference drops settlements by ID, production excludes them by CATEGORY and
then unions the rows with no population. This asserts the two agree, per city
and per scenario, instead of assuming it.
"""
from pathlib import Path

import pytest

from delhi_psi.categories import apply_mapping
from delhi_psi.pipeline import attach_population, excluded_ids
from tests.cities import CITIES, FIXTURES_ROOT, MESSY, ORACULUM

# Task 3 (scripts/generate_messy_fixtures.py) widens this to CITIES. Until the
# messy GeoJSON files exist, only Oraculum can be loaded from disk.
FIXTURED = (ORACULUM,)
SCENARIO_CASES = [(city, scenario) for city in FIXTURED
                  for scenario in city.scenarios]


def case_id(case):
    city, scenario = case
    return f"{city.name}-{scenario.name}"


DECLARED = {
    "oraculum": ("oracle-6", ("Planned", "UC", "JJC", "RV", "RUAC", "IND")),
    "messy": ("messy-2", ("Planned", "RV")),
}


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_scheme_vocabulary_and_epsg(city):
    scheme, vocabulary = DECLARED[city.name]
    assert city.scheme == scheme
    assert city.vocabulary == vocabulary
    assert city.epsg == 7760
    assert len(city.vocabulary) == len(set(city.vocabulary)), "duplicate type"


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_mapping_is_the_identity_over_the_vocabulary(city):
    assert city.mapping() == {t: t for t in city.vocabulary}


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_fixtures_path_is_tests_fixtures_slash_name(city):
    """`fixtures` is a real field (spec § 3), so pin it against `name` — the
    two must not drift."""
    assert city.fixtures == FIXTURES_ROOT / city.name
    assert FIXTURES_ROOT == Path(__file__).resolve().parent / "fixtures"


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_scenario_names_are_unique(city):
    names = [scenario.name for scenario in city.scenarios]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_stage_agrees_with_dropped_before_neighbors(city):
    """One set, one flag: production applies ONE `stage` to `excluded ∪
    missing`, so the reference's `dropped_before_neighbors` is exactly
    `stage == pre_neighbors` (spec § 3, rev 3)."""
    for scenario in city.scenarios:
        assert scenario.stage in ("post_neighbors", "pre_neighbors"), scenario
        assert scenario.dropped_before_neighbors is (
            scenario.stage == "pre_neighbors"), scenario.name


def test_oraculum_scenarios_are_todays_reference_table_in_order():
    """Order wins: it fixes expected_values.csv's row order, which is
    round-trip tested byte for byte."""
    assert [(s.name, set(s.dropped), s.dropped_before_neighbors,
             s.exclusion_types, s.stage) for s in ORACULUM.scenarios] == [
        ("baseline", set(), False, (), "post_neighbors"),
        ("excl_contributing", {"RV", "IND"}, False, ("RV", "IND"),
         "post_neighbors"),
        ("excl_removed", {"RV", "IND"}, True, ("RV", "IND"), "pre_neighbors"),
        ("excl_ind_removed", {"IND"}, True, ("IND",), "pre_neighbors"),
        ("excl_rv_only", {"RV"}, False, ("RV",), "post_neighbors"),
    ]


def test_messy_scenarios_are_the_spec_table():
    """Every messy scenario drops `U` with the scenario's OWN flag, because
    production drops a no-population id unconditionally and applies its single
    `stage` to the whole drop set (spec § 3, rev 3)."""
    assert [(s.name, set(s.dropped), s.dropped_before_neighbors,
             s.exclusion_types, s.stage) for s in MESSY.scenarios] == [
        ("nopop_only", {"U"}, False, (), "post_neighbors"),
        ("excl_rv_post", {"U", "N"}, False, ("RV",), "post_neighbors"),
        ("excl_rv_pre", {"U", "N"}, True, ("RV",), "pre_neighbors"),
    ]
    assert MESSY.scenarios[0].dropped != MESSY.scenarios[1].dropped, \
        "nopop_only and excl_rv_post must differ, or category exclusion is " \
        "never exercised on this city"


def test_cities_module_imports_no_production_code():
    """INDEPENDENCE RULE: tests/reference_impl.py imports this module, so this
    module must not reach production code — nor back into the reference."""
    source = (Path(__file__).resolve().parent / "cities.py").read_text()
    assert "delhi_psi" not in source
    assert "reference_impl" not in source


@pytest.mark.parametrize("city", FIXTURED, ids=lambda c: c.name)
def test_every_layer_loads(city):
    settlements = city.load_settlements()
    assert len(settlements) > 0
    assert set(settlements.columns) >= {"USO_AREA_U", "USO_FINAL",
                                        "population", "area_km2", "geometry"}
    assert settlements.crs.to_epsg() == city.epsg
    assert city.load_barriers().crs.to_epsg() == city.epsg
    services = city.load_services()
    assert set(services) >= {"clinic", "school", "bank", "police", "ration",
                             "transport", "road"}


@pytest.mark.parametrize("city", FIXTURED, ids=lambda c: c.name)
def test_vocabulary_is_exactly_the_types_the_layer_carries(city):
    """No more (it would hide an unmapped type), no fewer (the run errors)."""
    assert set(city.load_settlements()["USO_FINAL"]) == set(city.vocabulary)


@pytest.mark.parametrize("city", FIXTURED, ids=lambda c: c.name)
def test_there_is_at_least_one_road_row(city):
    assert len(city.load_services()["road"]) >= 1


@pytest.mark.parametrize("case", SCENARIO_CASES, ids=case_id)
def test_dropped_is_excluded_ids_union_missing(case):
    """THE agreement pin: the reference's id-based `dropped` is exactly
    production's `excluded_ids(types) ∪ missing` for this city."""
    city, scenario = case
    frame, missing = attach_population(city.load_settlements(), None)
    frame = apply_mapping(frame, type_col="USO_FINAL", mapping=city.mapping())
    excluded = excluded_ids(frame, types=scenario.exclusion_types)
    assert excluded | missing == scenario.dropped, (
        city.name, scenario.name, sorted(excluded), sorted(missing))
    # ... and every id the scenario names is really a settlement of this city.
    assert scenario.dropped <= set(frame["USO_AREA_U"]), scenario.name
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cities.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'tests.cities'`.

- [ ] **Step 3: Write the module**

Create `tests/cities.py`:

```python
"""The fixture cities: Oraculum (hand-ratifiable) and Messy (pathologies).

Fixture PLUMBING only — where a city's files live, what its vocabulary is,
which scenarios it is scored under. Deliberately importing nothing from this
repo: `tests/reference_impl.py` imports this module, and the reference's
INDEPENDENCE RULE forbids it from seeing the production library; importing
`tests.reference_impl` from here would also be a cycle. The index math lives
in `tests/reference_impl.py` and `delhi_psi/`, never here.

A `Scenario` carries ONE drop set and ONE flag, because that is exactly what
production does: `dropped = excluded_ids(types) ∪ missing`, with a single
`exclusion.stage` applied to all of it. `exclusion_types` and `stage` are the
production-side spelling of the same scenario; `tests/test_cities.py` pins
that the two spellings select the same rows.
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd

FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
DEFAULT_EPSG = 7760


@dataclass(frozen=True)
class Scenario:
    """One row of a city's scenario table, in both spellings.

    name: the reference implementation's scenario name (also the value in
        `expected_values.csv`'s `scenario` column).
    dropped: settlement IDS the reference drops.
    dropped_before_neighbors: True iff the drop happens before neighbour
        construction.
    exclusion_types: CATEGORY names for production's
        `methodology.exclusion.types`.
    stage: production's `methodology.exclusion.stage`.
    """

    name: str
    dropped: frozenset
    dropped_before_neighbors: bool
    exclusion_types: tuple
    stage: str


@dataclass(frozen=True)
class City:
    """One fixture city: its files, its vocabulary, its scenario table."""

    name: str
    fixtures: Path
    scheme: str
    vocabulary: tuple
    scenarios: tuple
    epsg: int = DEFAULT_EPSG

    def mapping(self):
        """The identity over this city's source types.

        The fixture cities are not Delhi, so the shipped profiles' `uso-10`
        mapping does not cover them; every test that runs production on a
        fixture city swaps in this identity (spec 3B § 2).
        """
        return {source: source for source in self.vocabulary}

    def _read(self, filename):
        gdf = gpd.read_file(self.fixtures / filename)
        return gdf.set_crs(epsg=self.epsg, allow_override=True)

    def load_settlements(self):
        return self._read("settlements.geojson")

    def load_barriers(self):
        """May be an EMPTY collection (the messy city has no barriers): an
        empty GeoJSON FeatureCollection reads back as a 0-row frame with a
        geometry column, which both implementations short-circuit on."""
        return self._read("barriers.geojson")

    def load_services(self):
        gdf = self._read("services.geojson")
        return {name: grp.reset_index(drop=True)
                for name, grp in gdf.groupby("service")}


ORACULUM = City(
    name="oraculum",
    fixtures=FIXTURES_ROOT / "oraculum",
    scheme="oracle-6",
    vocabulary=("Planned", "UC", "JJC", "RV", "RUAC", "IND"),
    # ORDER IS LOAD-BEARING: it is today's `reference_impl.SCENARIOS` order,
    # which fixes `expected_values.csv`'s row order — round-trip tested byte
    # for byte.
    scenarios=(
        Scenario(name="baseline", dropped=frozenset(),
                 dropped_before_neighbors=False, exclusion_types=(),
                 stage="post_neighbors"),
        Scenario(name="excl_contributing", dropped=frozenset({"RV", "IND"}),
                 dropped_before_neighbors=False,
                 exclusion_types=("RV", "IND"), stage="post_neighbors"),
        Scenario(name="excl_removed", dropped=frozenset({"RV", "IND"}),
                 dropped_before_neighbors=True,
                 exclusion_types=("RV", "IND"), stage="pre_neighbors"),
        Scenario(name="excl_ind_removed", dropped=frozenset({"IND"}),
                 dropped_before_neighbors=True, exclusion_types=("IND",),
                 stage="pre_neighbors"),
        Scenario(name="excl_rv_only", dropped=frozenset({"RV"}),
                 dropped_before_neighbors=False, exclusion_types=("RV",),
                 stage="post_neighbors"),
    ),
)

# Every messy scenario drops `U` with the scenario's own flag: production
# drops a no-population id unconditionally and applies its single `stage` to
# the whole drop set, so under `pre_neighbors` `U` leaves the neighbour lists
# too (spec § 3, rev 3). That is why the no-population pathology lives in its
# own settlement and not in the RV one.
MESSY = City(
    name="messy",
    fixtures=FIXTURES_ROOT / "messy",
    scheme="messy-2",
    vocabulary=("Planned", "RV"),
    scenarios=(
        Scenario(name="nopop_only", dropped=frozenset({"U"}),
                 dropped_before_neighbors=False, exclusion_types=(),
                 stage="post_neighbors"),
        Scenario(name="excl_rv_post", dropped=frozenset({"U", "N"}),
                 dropped_before_neighbors=False, exclusion_types=("RV",),
                 stage="post_neighbors"),
        Scenario(name="excl_rv_pre", dropped=frozenset({"U", "N"}),
                 dropped_before_neighbors=True, exclusion_types=("RV",),
                 stage="pre_neighbors"),
    ),
)

CITIES = (ORACULUM, MESSY)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_cities.py -q`
Expected: **21 passed**.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **302 passed** (281 carried over + 21 new).

- [ ] **Step 6: Prove no fixture moved**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
```

Expected: the `git status` output is **empty**.

- [ ] **Step 7: Commit**

```bash
git add tests/cities.py tests/test_cities.py
git commit -m "feat(fixtures): City abstraction with Oraculum and Messy (DEL-24)

tests/cities.py declares both fixture cities: files, scheme, vocabulary and
a scenario table carrying one drop set and one flag in both spellings (the
reference's ids, production's categories + stage). It imports geopandas and
nothing else from this repo, so the reference implementation may import it
without breaching the INDEPENDENCE RULE.

test_cities.py pins, per city and scenario, that production's
excluded_ids(types) union missing equals the reference's dropped set, and
that the stage agrees with dropped_before_neighbors. Messy's file-backed
cases arrive with its fixtures.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: reference generalisation — city, scenario table, all road rows (spec § 3)

Three surgical generalisations to `tests/reference_impl.py`, the matching
`city=` wrappers in `tests/oraculum_fixtures.py`, and the removal of
`render_oracle_maps.py`'s global mutation. **Not one rule changes**, and
Oraculum's `expected_values.csv` stays byte-identical — proved in Step 8.

**Files:**
- Modify: `tests/reference_impl.py` (`SCENARIOS`, `_service_amounts`, `compute_city`, `emit_expected_values`, `__main__`)
- Modify: `tests/oraculum_fixtures.py`
- Modify: `scripts/render_oracle_maps.py:33` (import) and `:345-349` (the `setdefault` site), `:381-385` (`_ideal_frame`)
- Modify: `tests/test_reference_impl.py` (append three tests)
- Modify: `tests/test_cities.py` (append four tests)

**Interfaces:**
- Consumes: `tests.cities.ORACULUM`, `tests.cities.CITIES`, `tests.cities.City` (Task 1).
- Produces:
  - `tests.reference_impl.SCENARIOS: dict[str, tuple[frozenset, bool]]` — the literal view `{s.name: (s.dropped, s.dropped_before_neighbors) for s in ORACULUM.scenarios}`, same content and same order as today
  - `tests.reference_impl.compute_city(settlements, services, barriers, *, adjacency_rule, barrier_rule, roads_formula, scenario, denom, second_norm, absent_neighbor_contribution, scenarios=None)` — `scenarios=None` means the module-level `SCENARIOS`
  - `tests.reference_impl.emit_expected_values(out_path, city=ORACULUM) -> DataFrame` — `out_path` stays first
  - `tests.oraculum_fixtures.oracle_mapping(city=ORACULUM) -> dict[str, str]`
  - `tests.oraculum_fixtures.oracle_config(base, city=ORACULUM) -> Config`
  - `tests.oraculum_fixtures.oracle_profile_path(base, directory, city=ORACULUM) -> Path`
  - `tests.oraculum_fixtures.methodology_with(profile, *, types=None, stage=None, city=ORACULUM) -> MethodologyConfig`
  - `tests.oraculum_fixtures.compute_oracle_frame(profile, *, types, stage, denom, city=ORACULUM) -> DataFrame` (indexed by `USO_AREA_U`)
  - `tests.oraculum_fixtures.ORACLE_SCENARIOS: list[tuple[str, tuple[str, ...], str]]` = `[(s.name, s.exclusion_types, s.stage) for s in ORACULUM.scenarios]`
  - `tests.oraculum_fixtures.ORACLE_SCHEME`, `ORACLE_VOCABULARY` — aliases of `ORACULUM.scheme` / `ORACULUM.vocabulary`
  - `scripts.render_oracle_maps.MAP_SCENARIOS: dict[str, tuple[frozenset, bool]]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reference_impl.py`:

```python
# --- 3C: the reference generalisations (spec § 3) ----------------------
def test_compute_city_accepts_an_explicit_scenario_table(city, services,
                                                         barriers):
    """The drop mechanics are untouched; only where the table comes from
    moves. A caller-supplied table must NOT leak into the module global —
    scripts/render_oracle_maps.py used to mutate it, which would have
    widened the round-trip-tested fixture CSV."""
    from tests.reference_impl import SCENARIOS

    before = dict(SCENARIOS)
    table = {"nothing_dropped": (frozenset(), False)}
    got = compute_city(city, services, barriers,
                       scenario="nothing_dropped", denom="pop",
                       scenarios=table, **RULESETS["ideal"])
    expected = _city_df(city, services, barriers, "ideal")
    assert list(got.index) == list(expected.index)
    for sid in expected.index:
        assert got.loc[sid, "clinic_pcen"] == pytest.approx(
            expected.loc[sid, "clinic_pcen"], abs=1e-15), sid
    assert dict(SCENARIOS) == before, "the module table was mutated"


def test_service_amounts_sums_every_road_row(city, services):
    """`_service_amounts` used the FIRST road row only. The messy city has
    two, so the sum is load-bearing; pinned here on Oraculum with a second
    row bolted on, so the pin does not depend on the messy fixtures."""
    import geopandas as gpd
    from shapely.geometry import LineString

    from tests.reference_impl import _service_amounts

    base = 1_000_000
    # 500 m of road strictly inside D (x in [-500, 500], y in [0, 1000]),
    # touching no other settlement.
    extra = LineString([(base - 250, base + 500), (base + 250, base + 500)])
    two_rows = gpd.GeoDataFrame(
        {"service": ["road", "road"]},
        geometry=[services["road"].geometry.iloc[0], extra], crs=city.crs)

    amounts = _service_amounts(city, {**services, "road": two_rows})["road"]
    assert amounts["A"] == pytest.approx(0.75, abs=1e-12)
    assert amounts["E"] == pytest.approx(0.75, abs=1e-12)
    assert amounts["D"] == pytest.approx(0.5, abs=1e-12), \
        "the second road row was ignored"
    for sid in ("B", "C", "RV", "IND"):
        assert amounts[sid] == 0.0, sid


def test_emit_expected_values_takes_a_city_and_defaults_to_oraculum(tmp_path):
    from tests.cities import ORACULUM

    implicit = tmp_path / "implicit.csv"
    explicit = tmp_path / "explicit.csv"
    emit_expected_values(implicit)
    emit_expected_values(explicit, ORACULUM)
    assert implicit.read_bytes() == explicit.read_bytes() == CSV.read_bytes()
```

Append to `tests/test_cities.py`:

```python
# --- 3C: the backward-compatible module views (spec § 3) ---------------
def test_reference_scenarios_view_is_oraculums_table():
    """`reference_impl.SCENARIOS` keeps the 2-tuple shape consumed today,
    with Oraculum's order — which fixes expected_values.csv's row order."""
    from tests.reference_impl import SCENARIOS

    assert SCENARIOS == {s.name: (s.dropped, s.dropped_before_neighbors)
                         for s in ORACULUM.scenarios}
    assert list(SCENARIOS) == [s.name for s in ORACULUM.scenarios]


def test_oracle_scenarios_view_is_the_three_tuple_of_oraculums_table():
    from tests.oraculum_fixtures import ORACLE_SCENARIOS

    assert ORACLE_SCENARIOS == [(s.name, s.exclusion_types, s.stage)
                                for s in ORACULUM.scenarios]


def test_oracle_scheme_and_vocabulary_are_oraculum_aliases():
    from tests.oraculum_fixtures import (
        ORACLE_SCHEME, ORACLE_VOCABULARY, oracle_mapping,
    )

    assert ORACLE_SCHEME == ORACULUM.scheme == "oracle-6"
    assert ORACLE_VOCABULARY == ORACULUM.vocabulary
    assert oracle_mapping() == ORACULUM.mapping()


def test_render_oracle_maps_does_not_mutate_the_reference_scenario_table():
    """Importing the map script must not widen reference_impl.SCENARIOS: the
    fixture CSV is round-trip tested at its current row count, and a
    setdefault at import time would add a sixth scenario to every later
    emit in the same process."""
    from tests.reference_impl import SCENARIOS

    before = dict(SCENARIOS)
    from scripts.render_oracle_maps import MAP_SCENARIOS

    assert dict(SCENARIOS) == before
    assert "rv_removed" not in SCENARIOS
    assert MAP_SCENARIOS["rv_removed"] == (frozenset({"RV"}), True)
    assert all(MAP_SCENARIOS[name] == value
               for name, value in SCENARIOS.items())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_reference_impl.py tests/test_cities.py -q`
Expected: FAIL —
`TypeError: compute_city() got an unexpected keyword argument 'scenarios'`;
`assert amounts["D"] == 0.5` fails with `0.0`;
`TypeError: emit_expected_values() takes 1 positional argument but 2 were given`;
`ImportError: cannot import name 'MAP_SCENARIOS'`.

- [ ] **Step 3: Generalise `tests/reference_impl.py`**

Replace the `SCENARIOS` literal (lines 32-39) with:

```python
from tests.cities import ORACULUM

# Backward-compatible view of Oraculum's table in the 2-tuple shape this
# module has always consumed: {name: (dropped ids, dropped_before_neighbors)}.
# ORACULUM.scenarios' ORDER is today's order, which fixes expected_values.csv.
# (tests/cities.py imports geopandas and nothing from this repo, so the
# INDEPENDENCE RULE is intact: it is fixture plumbing, not index math.)
SCENARIOS = {s.name: (s.dropped, s.dropped_before_neighbors)
             for s in ORACULUM.scenarios}
```

(The `import pandas as pd` / `from shapely.geometry import box` lines stay
where they are; put the `tests.cities` import beneath them.)

In `_service_amounts`, replace:

```python
    road = services["road"].geometry.iloc[0]
    amounts["road"] = {i: road.intersection(idx[i]).length / 1000
                       for i in idx.index}
```

with:

```python
    # EVERY road row, not just the first: a city may carry the road network
    # as several LineStrings (the messy city does), and production's
    # `road_lengths` already sums all of them per settlement.
    road_geoms = list(services["road"].geometry)
    amounts["road"] = {
        i: sum(road.intersection(idx[i]).length for road in road_geoms) / 1000
        for i in idx.index}
```

In `compute_city`, change the signature line and the first statement:

```python
def compute_city(settlements, services, barriers, *, adjacency_rule,
                 barrier_rule, roads_formula, scenario, denom, second_norm,
                 absent_neighbor_contribution, scenarios=None):
    # `scenarios` defaults to the module table, so every existing call keeps
    # working; a caller may pass its own WITHOUT mutating the global (which
    # is what scripts/render_oracle_maps.py used to do).
    table = SCENARIOS if scenarios is None else scenarios
    dropped, drop_before = table[scenario]
```

Replace `emit_expected_values` and `__main__` with:

```python
def emit_expected_values(out_path, city=ORACULUM):
    """Score `city` under every rule-set x scenario x denominator and write
    the long-format CSV. `out_path` stays FIRST so existing callers (the
    round-trip test, the generators) are unchanged.
    """
    settlements, barriers, services = (city.load_settlements(),
                                       city.load_barriers(),
                                       city.load_services())
    scenarios = {s.name: (s.dropped, s.dropped_before_neighbors)
                 for s in city.scenarios}
    records = []
    for rule, kwargs in RULESETS.items():
        for scenario in scenarios:
            for denom in ("pop", "popdensity"):
                df = compute_city(settlements, services, barriers,
                                  scenario=scenario, denom=denom,
                                  scenarios=scenarios, **kwargs)
                for sid, row in df.iterrows():
                    for metric, value in row.items():
                        records.append((rule, scenario, denom, sid,
                                        metric, value))
    out = pd.DataFrame(records, columns=["rule", "scenario", "denom",
                                         "settlement", "metric", "value"])
    out.to_csv(out_path, index=False, float_format="%.17g")
    return out


if __name__ == "__main__":
    from tests.cities import CITIES

    for target_city in CITIES:
        target = target_city.fixtures / "expected_values.csv"
        emit_expected_values(target, target_city)
        print(f"wrote {target}")
```

Nothing else in the file changes: `RULESETS`, `POINT_SERVICES`, `adjacency`,
`apply_barrier`, `_centroid_km`, the point-service counting, every rule
inside `compute_city`, the min-max guard and the `%.17g` CSV format are
untouched.

- [ ] **Step 4: Make `tests/oraculum_fixtures.py` thin wrappers**

Replace lines 1-30 (the header and loaders) with:

```python
"""Loaders for the Oraculum fixtures.

Every public name here is now a thin wrapper over `tests/cities.py`'s
city-taking functions with `city=ORACULUM` bound, so no existing test
changes its call shape or its expected value (spec 3C § 3).
"""

from pathlib import Path

import geopandas as gpd

from tests.cities import ORACULUM

FIXTURES = ORACULUM.fixtures
EPSG = ORACULUM.epsg


def _read(path):
    gdf = gpd.read_file(path)
    return gdf.set_crs(epsg=EPSG, allow_override=True)


def load_settlements(city=ORACULUM):
    return city.load_settlements()


def load_barriers(city=ORACULUM):
    return city.load_barriers()


def load_services(city=ORACULUM):
    return city.load_services()


def load_exhibit():
    """The divergence exhibit is Oraculum-only (spec § 4.5: unchanged)."""
    return _read(ORACULUM.fixtures / "divergence" / "exhibit.geojson")
```

Replace the `ORACLE_SCENARIOS` literal (lines 43-50) with — keeping the
comment block above it exactly as it is:

```python
# 3C: the 3-tuple view of ORACULUM.scenarios. Its LIST ORDER now follows
# reference_impl.SCENARIOS rather than today's hand-written order; only
# pytest collection order is affected, because
# generate_production_fixtures.write_fixture sorts by
# (scenario, denom, settlement, metric) before writing.
ORACLE_SCENARIOS = [(s.name, s.exclusion_types, s.stage)
                    for s in ORACULUM.scenarios]
```

Replace the `ORACLE_SCHEME` / `ORACLE_VOCABULARY` literals (lines 60-61) with:

```python
ORACLE_SCHEME = ORACULUM.scheme
ORACLE_VOCABULARY = ORACULUM.vocabulary
```

Give the four profile helpers a `city` parameter — the bodies are otherwise
verbatim, and every docstring above them stays:

```python
def oracle_mapping(city=ORACULUM):
    """The identity over the fixture city's source types."""
    return city.mapping()


def oracle_config(base, city=ORACULUM):
    from dataclasses import replace

    from delhi_psi.config import CategoriesConfig, load_config

    return replace(load_config(base),
                   categories=CategoriesConfig(scheme=city.scheme,
                                               mapping=city.mapping()))


def oracle_profile_path(base, directory, city=ORACULUM):
    import yaml

    from delhi_psi.config import PROFILES_DIR

    raw = yaml.safe_load((PROFILES_DIR / f"{base}.yaml").read_text())
    raw["categories"] = {"scheme": city.scheme, "mapping": city.mapping()}
    path = Path(directory) / f"{base}.oracle.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def methodology_with(profile, *, types=None, stage=None, city=ORACULUM):
    from dataclasses import replace

    from delhi_psi.config import ExclusionStage

    methodology = oracle_config(profile, city).methodology
    exclusion = methodology.exclusion
    if types is not None:
        exclusion = replace(exclusion, types=tuple(types))
    if stage is not None:
        exclusion = replace(exclusion, stage=ExclusionStage(stage))
    return replace(methodology, exclusion=exclusion)


def compute_oracle_frame(profile, *, types, stage, denom, city=ORACULUM):
    from delhi_psi.pipeline import compute_frames

    cfg = oracle_config(profile, city)
    return compute_frames(
        city.load_settlements(), {"canal": city.load_barriers()},
        city.load_services(), None,
        methodology_with(profile, types=types, stage=stage, city=city),
        denom, mapping=cfg.categories.mapping, scheme=cfg.categories.scheme,
    ).set_index("USO_AREA_U")
```

`oracle_profile_path`'s filename stays `{base}.oracle.yaml` on purpose: no
test writes two cities' derived profiles into one directory, and changing it
would churn `tests/test_cli.py` and `tests/test_oracle_e2e.py` for nothing.

- [ ] **Step 5: Stop `render_oracle_maps.py` mutating the global**

Replace lines 345-349:

```python
# The reference impl's SCENARIOS has "RV excluded but contributing"
# (excl_rv_only) but no "RV alone, fully removed"; the decision memo needs
# both sides of the same coin, so register it here rather than widening the
# fixture CSV (which is round-trip tested at its current row count).
SCENARIOS.setdefault("rv_removed", (frozenset({"RV"}), True))
```

with:

```python
# The reference impl's SCENARIOS has "RV excluded but contributing"
# (excl_rv_only) but no "RV alone, fully removed"; the decision memo needs
# both sides of the same coin. Pass the extra row EXPLICITLY instead of
# mutating the module global: the fixture CSV is round-trip tested at its
# current row count, so a setdefault at import time would widen every later
# emit_expected_values in the same process. Same figures.
MAP_SCENARIOS = {**SCENARIOS, "rv_removed": (frozenset({"RV"}), True)}
```

and, in `_ideal_frame`, pass it:

```python
def _ideal_frame(scenario, denom="pop"):
    city, barriers, services = (load_settlements(), load_barriers(),
                                load_services())
    return compute_city(city, services, barriers, scenario=scenario,
                        denom=denom, scenarios=MAP_SCENARIOS,
                        **RULESETS["ideal"])
```

The import on line 33 is unchanged (`SCENARIOS` is still imported — it is
what `MAP_SCENARIOS` is built from).

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `uv run pytest tests/test_reference_impl.py tests/test_cities.py -q`
Expected: **43 passed** — `tests/test_reference_impl.py` 15 carried over + 3 new = 18, `tests/test_cities.py` 21 from Task 1 + 4 new = 25.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **309 passed** (302 + 7 new).

- [ ] **Step 8: Prove Oraculum is byte-identical, and the maps still render**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
uv run python tests/reference_impl.py || true
uv run python scripts/render_oracle_maps.py
git status --porcelain -- tests/fixtures/ docs/oracle/
```

Expected: the `git status` output is **empty**. (`tests/reference_impl.py`'s
`__main__` now loops both cities and will fail on the missing messy fixtures
until Task 3 — hence the `|| true`; it must still have rewritten Oraculum's
CSV identically before failing, which the empty `git status` proves. Task 3
removes the need for `|| true`.)

- [ ] **Step 9: Commit**

```bash
git add tests/reference_impl.py tests/oraculum_fixtures.py \
        scripts/render_oracle_maps.py tests/test_reference_impl.py \
        tests/test_cities.py
git commit -m "refactor(oracle): generalise the reference over a City (DEL-24)

compute_city takes an explicit scenario table (default: the module one),
_service_amounts sums EVERY road row, and emit_expected_values takes a city.
oraculum_fixtures' public names become thin city= wrappers and
ORACLE_SCENARIOS/SCHEME/VOCABULARY become views over ORACULUM.
render_oracle_maps passes its extra rv_removed row explicitly instead of
mutating reference_impl.SCENARIOS at import time.

No rule changed: Oraculum's expected_values.csv and both production CSVs are
byte-identical, and the three figures re-render unchanged.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `scripts/generate_messy_fixtures.py` — the eleven settlements (spec § 2)

The generator asserts **every** § 2 / § 4.3 relation on the geometries it is
about to write, then writes the three GeoJSON layers, then has the reference
score them and refuses to install a CSV that violates the invariants guard.
`generate_oraculum_fixtures.py` gains the same final step.

**Files:**
- Create: `scripts/generate_messy_fixtures.py`
- Modify: `scripts/check_oraculum_invariants.py` (per-city paths + `emit_checked_expected_values`)
- Modify: `scripts/generate_oraculum_fixtures.py` (final expected-values step)
- Modify: `tests/test_cities.py` (`FIXTURED = CITIES`; two new tests)
- Create (generated, committed): `tests/fixtures/messy/settlements.geojson`, `services.geojson`, `barriers.geojson`, `expected_values.csv`

**Interfaces:**
- Consumes: `tests.cities.MESSY`, `tests.cities.ORACULUM`, `tests.cities.CITIES` (Task 1); `tests.reference_impl.emit_expected_values(out_path, city)` (Task 2).
- Produces:
  - `scripts.check_oraculum_invariants.expected_values_path(city=ORACULUM) -> Path` = `city.fixtures / "expected_values.csv"`
  - `scripts.check_oraculum_invariants.check(df=None, *, city=ORACULUM) -> list[str]` — unchanged semantics; `df=None` reads that city's CSV
  - `scripts.check_oraculum_invariants.emit_checked_expected_values(city, out_path) -> Path` — emits to a temp file, runs `check`, raises `SystemExit(1)` writing nothing on any violation, otherwise moves the file into place
  - `scripts.generate_messy_fixtures.BASE_X`, `BASE_Y` = `1_000_000`; `OUT: Path`; `SETTLEMENTS`, `POINT_SERVICES`, `ROADS` tables; `main()`

- [ ] **Step 1: Write the failing test**

In `tests/test_cities.py`, change the one line

```python
FIXTURED = (ORACULUM,)
```

to

```python
FIXTURED = CITIES
```

and delete the two-line comment above it (`# Task 3 ... from disk.`).
Then append:

```python
# --- 3C: the messy fixtures exist and are what the generator emits ------
def test_messy_expected_values_csv_covers_the_scenario_grid():
    import pandas as pd

    df = pd.read_csv(MESSY.fixtures / "expected_values.csv")
    assert set(df.columns) == {"rule", "scenario", "denom", "settlement",
                               "metric", "value"}
    assert set(df["rule"]) == {"ideal", "code"}
    assert set(df["scenario"]) == {s.name for s in MESSY.scenarios}
    assert set(df["denom"]) == {"pop", "popdensity"}
    # `U` has no population row, so production drops it unconditionally and
    # every messy scenario drops it on the reference side too.
    assert "U" not in set(df["settlement"])
    assert set(df[df["scenario"] == "nopop_only"]["settlement"]) == {
        "H", "L", "T", "M", "G", "O1", "O2", "I", "N", "S"}
    for scenario in ("excl_rv_post", "excl_rv_pre"):
        assert set(df[df["scenario"] == scenario]["settlement"]) == {
            "H", "L", "T", "M", "G", "O1", "O2", "I", "S"}, scenario
    # norm_psi is the `code` rule-set's second normalization only.
    assert ("norm_psi" in set(df[df["rule"] == "code"]["metric"]))
    assert ("norm_psi" not in set(df[df["rule"] == "ideal"]["metric"]))


def test_generators_write_only_under_tests_fixtures():
    """CI spec § Robustness: the drift step globs generate_*_fixtures.py and
    diffs tests/fixtures/. A generator that writes anywhere else escapes the
    guard entirely."""
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    generators = sorted(scripts_dir.glob("generate_*_fixtures.py"))
    assert {p.name for p in generators} == {
        "generate_messy_fixtures.py", "generate_oraculum_fixtures.py",
        "generate_production_fixtures.py"}
    for path in generators:
        source = path.read_text()
        # Every output path is anchored on a fixtures directory — either the
        # repo-relative literal or a City's own `fixtures` attribute.
        assert "fixtures" in source, path.name
        # ... and none of them may reach for the real data directory.
        for banned in ("delhi_data", "DELHI_DATA_DIR", "data_dir"):
            assert banned not in source, (path.name, banned)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_cities.py -q`
Expected: FAIL — `pyogrio.errors.DataSourceError` (`City._read` calls
`gpd.read_file` directly, so it is pyogrio's error, not `io.read_layer`'s
`FileNotFoundError`) for `tests/fixtures/messy/settlements.geojson` in
`test_every_layer_loads[messy]`,
`test_vocabulary_is_exactly_the_types_the_layer_carries[messy]`,
`test_there_is_at_least_one_road_row[messy]` and the three
`test_dropped_is_excluded_ids_union_missing[messy-*]` cases;
`FileNotFoundError` in `test_messy_expected_values_csv_covers_the_scenario_grid`;
`AssertionError` on the generator name set in
`test_generators_write_only_under_tests_fixtures`.

- [ ] **Step 3: Add the checked-emit helper to `scripts/check_oraculum_invariants.py`**

Replace the module docstring and the `CSV` constant with:

```python
"""Spec §7 consistency guard over a city's expected_values.csv (CSV-wide
scope; geometry-scope checks live in tests/test_fixture_invariants.py).

From cycle 3C this module also owns the "only write a VALID fixture" step:
`emit_checked_expected_values` is what both geometry generators call, so a
city whose numbers would violate the guard is never committed.

Run standalone over every city (exit 1 on violation) or via its pytest
wrapper:
    uv run python scripts/check_oraculum_invariants.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

from tests.cities import CITIES, ORACULUM

SERVICES = ("clinic", "school", "bank", "police", "ration", "transport",
            "road")
UNIQUE_ANCHOR_SERVICES = ("clinic", "school")


def expected_values_path(city=ORACULUM):
    return city.fixtures / "expected_values.csv"


CSV = expected_values_path(ORACULUM)
```

Change `check`'s signature and first line only:

```python
def check(df=None, *, city=ORACULUM):
    df = pd.read_csv(expected_values_path(city)) if df is None else df
```

Everything below that line — the grouping, the degenerate min-max check and
the tied argmin/argmax check — is untouched.

Append the new helper and replace `__main__`:

```python
def emit_checked_expected_values(city, out_path):
    """Emit `city`'s expected_values.csv, but ONLY if it passes `check`.

    The reference scores the city into a temporary file, the guard runs on
    exactly the bytes that would be committed, and the file is moved into
    place only when there are no violations. On any violation NOTHING is
    written and the process exits 1 — a fixture that ties a clinic/school
    anchor or flattens a min-max group is not a fixture, it is a silently
    degenerate oracle.

    The reference import is local so `check` stays importable without
    dragging in geopandas.
    """
    from tests.reference_impl import emit_expected_values

    out_path = Path(out_path)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "expected_values.csv"
        emit_expected_values(staged, city)
        violations = check(pd.read_csv(staged))
        if violations:
            for violation in violations:
                print(f"VIOLATION [{city.name}]:", violation)
            raise SystemExit(
                f"{len(violations)} invariant violation(s) for city "
                f"{city.name!r}; refusing to write {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # The staging directory is OUTSIDE the repo, so a failed run can
        # never leave an untracked file under tests/fixtures/ for the CI
        # drift guard to trip over.
        shutil.move(str(staged), str(out_path))
    return out_path


if __name__ == "__main__":
    problems = []
    for target_city in CITIES:
        problems.extend(f"{target_city.name}: {problem}"
                        for problem in check(city=target_city))
    for p in problems:
        print("VIOLATION:", p)
    print("OK" if not problems else f"{len(problems)} violation(s)")
    sys.exit(1 if problems else 0)
```

- [ ] **Step 4: Write the messy generator**

Create `scripts/generate_messy_fixtures.py`:

```python
"""Generate the messy-city fixture (spec: 2026-08-28-messy-city-tier-design.md).

Eleven settlements, each carrying one real-layer pathology Oraculum omits by
construction. Deterministic: running twice produces byte-identical files.
Coordinates are EPSG:7760 metre offsets from BASE, written with json.dump
(not a GDAL driver) so the files stay human-readable and diff-stable;
loaders re-apply the CRS on read. Same conventions as
generate_oraculum_fixtures.py.

`_assert_relations` re-derives EVERY relation the tier exists to pin, from
the geometries themselves, before a byte is written: move a vertex so that H
starts touching L, or so that some third envelope reaches G, and this script
fails loudly instead of quietly emitting a city that pins nothing. Then the
reference implementation scores the city and the invariants guard runs on
the result, so a fixture with a tied clinic/school anchor is never written.

    uv run python scripts/generate_messy_fixtures.py
"""

import json
from pathlib import Path

from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from scripts.check_oraculum_invariants import emit_checked_expected_values
from tests.cities import MESSY

BASE_X, BASE_Y = 1_000_000, 1_000_000
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "messy"


def _rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# id: (USO_FINAL, population, [ring, ...])  -- more than one ring is a
# MultiPolygon. Rings are open (the writer closes them). Populations are all
# distinct, so no two settlements can tie by construction; `U` alone has
# none, which is the no-population pathology.
SETTLEMENTS = {
    # irregular hexagon: H cap L is EMPTY, yet each geometry reaches into the
    # other's envelope, so they are bbox neighbours BOTH ways and touch
    # neighbours neither way.
    "H":  ("Planned", 110, [[(400, 2400), (1600, 2300), (2200, 1600),
                             (1900, 1000), (1100, 1000), (900, 2200)]]),
    # concave L: its envelope [0,2000]x[0,2000] swallows H's lower half, and
    # its vertical arm (x<=800) reaches back under H's envelope at (400,1500).
    "L":  ("Planned", 200, [[(0, 0), (2000, 0), (2000, 800), (800, 800),
                             (800, 2000), (0, 2000)]]),
    # triangle meeting L at the single point (2000, 800): bbox neighbour,
    # never a touch neighbour.
    "T":  ("Planned", 300, [[(2000, 800), (3000, 200), (3000, 700)]]),
    # two equal parts with a 1 km gap -> centroid (6500, 500) lies OUTSIDE M.
    "M":  ("Planned", 400, [_rect(5000, 0, 6000, 1000),
                            _rect(7000, 0, 8000, 1000)]),
    # 100 m square centred EXACTLY on M's centroid, inside M's envelope and
    # disjoint from both parts: M is in G's bbox list, G is not in M's, and
    # the centroid distance is exactly 0 (weight exactly 1).
    "G":  ("Planned", 50,  [_rect(6450, 450, 6550, 550)]),
    # overlapping pair: a 200 m x 1000 m strip in common.
    "O1": ("Planned", 600, [_rect(10000, 0, 11000, 1000)]),
    "O2": ("Planned", 700, [_rect(10800, 0, 11800, 1000)]),
    # 17 km away: no neighbours under any rule.
    "I":  ("Planned", 800, [_rect(20000, 0, 21000, 1000)]),
    # the one RV settlement: what `code-2025` excludes by CATEGORY.
    "N":  ("RV",      900, [_rect(11800, 0, 12800, 1000)]),
    # no population row -> production drops it unconditionally.
    "U":  ("Planned", None, [_rect(9000, 0, 10000, 1000)]),
    # 2 m x 1 m sliver on H's southern edge: area 2e-6 km2, popdensity
    # denominator 5e7.
    "S":  ("Planned", 100, [_rect(1400, 999, 1402, 1000)]),
}

# All seven services are placed: the reference scores every service in
# POINT_SERVICES regardless, and production's PSI averages over the services
# present, so both sides must carry the same seven. `T` and `G` get a school
# so that `I` is the UNIQUE school minimum under the touch rule.
POINT_SERVICES = {
    "clinic": [("H", 1500, 1500), ("L", 400, 400), ("T", 2800, 550),
               ("M", 5500, 500), ("G", 6500, 500),
               ("O1+O2", 10900, 500),          # strictly inside the overlap
               ("I", 20500, 500)],
    "school": [("L", 600, 200), ("M", 7500, 500), ("O2", 11400, 500),
               ("N", 12300, 500), ("S", 1401.5, 999.5), ("T", 2900, 450),
               ("G", 6480, 480)],
    "bank": [("H", 1600, 1400), ("I", 20600, 600)],
    "police": [("L", 200, 600), ("O1", 10400, 500)],
    "ration": [("M", 5600, 600), ("S", 1400.5, 999.5)],
    "transport": [("H", 1700, 1300), ("N", 12500, 400)],
}

# TWO LineString rows, so "sum every road row" is load-bearing: the first row
# alone gives M nothing.
ROADS = [
    ("H+L", [(1500, 200), (1500, 2200)]),   # 0.6 km in L, 1.2 km in H
    ("M",   [(4800, 800), (8200, 800)]),    # 1.0 km in each part of M
]

ROAD_KM = {"H": 1.2, "L": 0.6, "M": 2.0}
SLIVER_AREA_KM2 = 2e-06


def _pt(x, y):
    return [BASE_X + x, BASE_Y + y]


def _ring(coords):
    """A closed GeoJSON linear ring from an open coordinate list."""
    return [[_pt(x, y) for x, y in list(coords) + [coords[0]]]]


def _shapely(rings):
    parts = [Polygon([(BASE_X + x, BASE_Y + y) for x, y in ring])
             for ring in rings]
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _ring_area_m2(ring):
    """Shoelace area in m^2 — analytic, as Oraculum's generator does it."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


def _area_km2(rings):
    return sum(_ring_area_m2(ring) for ring in rings) / 1_000_000


def _feature(geom, props):
    return {"type": "Feature", "properties": props, "geometry": geom}


def _dump(path, features):
    path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection",
          "crs_note": "coordinates are EPSG:7760 meters; loaders apply set_crs(7760)",
          "features": features}
    path.write_text(json.dumps(fc, indent=1, sort_keys=True) + "\n")


def _bbox_nbrs(geoms):
    """Directed bbox adjacency, exactly as BOTH implementations define it:
    j is in i's list iff geom_i intersects envelope_j."""
    return {i: {j for j in geoms
                if j != i and geoms[i].intersects(geoms[j].envelope)}
            for i in geoms}


def _touch_nbrs(geoms):
    """Border sharing: the intersection must have positive length."""
    out = {}
    for i in geoms:
        out[i] = set()
        for j in geoms:
            if i == j:
                continue
            shared = geoms[i].intersection(geoms[j])
            if not shared.is_empty and shared.length > 0:
                out[i].add(j)
    return out


def _assert_relations(geoms, points, roads):
    """Every spec § 2 / § 4.3 relation, re-derived from the geometries."""
    bbox, touch = _bbox_nbrs(geoms), _touch_nbrs(geoms)

    for sid, geom in geoms.items():
        assert geom.is_valid, f"{sid} is not a valid geometry"

    # H / L: disjoint, but each reaches into the other's envelope.
    assert geoms["H"].intersection(geoms["L"]).is_empty, "H cap L must be empty"
    assert not geoms["H"].intersection(geoms["L"].envelope).is_empty, \
        "geom_H must reach into envelope_L"
    assert not geoms["L"].intersection(geoms["H"].envelope).is_empty, \
        "geom_L must reach into envelope_H (the L's arm under H's envelope)"
    assert "L" in bbox["H"] and "H" in bbox["L"], bbox
    assert "L" not in touch["H"] and "H" not in touch["L"], touch

    # T / L: a single point of contact.
    shared = geoms["T"].intersection(geoms["L"])
    assert shared.geom_type == "Point" and shared.length == 0, shared.geom_type
    assert "T" in bbox["L"] and "L" in bbox["T"], bbox
    assert "T" not in touch["L"] and "L" not in touch["T"], touch

    # M: two parts, centroid in the gap.
    assert len(geoms["M"].geoms) == 2, "M must be a two-part MultiPolygon"
    assert not geoms["M"].centroid.within(geoms["M"]), \
        "M's centroid must lie OUTSIDE M, in the gap"

    # G: centred exactly on M's centroid, the directed-bbox exhibit.
    assert geoms["G"].disjoint(geoms["M"]), "G must be disjoint from M"
    assert geoms["G"].centroid.equals(geoms["M"].centroid), \
        "G must be centred exactly on M's centroid"
    assert geoms["G"].centroid.distance(geoms["M"].centroid) == 0.0, \
        "the decay weight must be exactly 1"
    assert bbox["G"] == {"M"}, f"nbrs_bbox(G) must be exactly {{M}}: {bbox['G']}"
    assert "G" not in bbox["M"], "G must NOT be in M's bbox list"
    assert touch["G"] == set() and touch["M"] == set(), touch

    # O1 / O2: positive-area overlap, neighbours under BOTH rules.
    overlap = geoms["O1"].intersection(geoms["O2"])
    assert overlap.area > 0, "O1 cap O2 must have positive area"
    assert "O2" in bbox["O1"] and "O1" in bbox["O2"], bbox
    assert "O2" in touch["O1"] and "O1" in touch["O2"], \
        "overlapping polygons are touch neighbours (the DEL-19 finding)"
    shared_clinic = Point(*_pt(10900, 500))
    assert shared_clinic.within(geoms["O1"]) and shared_clinic.within(geoms["O2"]), \
        "the shared clinic must be STRICTLY inside the overlap"

    # I: isolated under both rules, disjoint from everything.
    assert bbox["I"] == set() and touch["I"] == set(), (bbox["I"], touch["I"])
    for sid, geom in geoms.items():
        if sid != "I":
            assert geoms["I"].disjoint(geom), f"I must be disjoint from {sid}"

    # U / N: the no-population settlement is NOT the excluded one.
    assert SETTLEMENTS["U"][1] is None, "U must have no population"
    assert SETTLEMENTS["N"][1] is not None, "N must HAVE a population"
    rv = [sid for sid, (uso, _, _) in SETTLEMENTS.items() if uso == "RV"]
    assert rv == ["N"], f"exactly one RV settlement, got {rv}"
    assert "U" in touch["O1"] and "N" in touch["O2"], touch

    # S: the area-extreme sliver, against H.
    assert _area_km2(SETTLEMENTS["S"][2]) == SLIVER_AREA_KM2, "S must be 2 m^2"
    assert geoms["S"].area > 0
    assert "H" in touch["S"] and "S" in touch["H"], touch

    # populations: all present ones distinct.
    pops = [pop for _, pop, _ in SETTLEMENTS.values() if pop is not None]
    assert len(pops) == len(set(pops)), f"tied populations: {pops}"

    # vocabulary matches the City declaration.
    assert {uso for uso, _, _ in SETTLEMENTS.values()} == set(MESSY.vocabulary)

    # every service point strictly inside EXACTLY its declared hosts, and
    # `within` (reference) agrees with `intersects` (production) on all of
    # them — the only multi-host point is the deliberate overlap clinic.
    for service, hosts, geom in points:
        expected = set(hosts)
        assert {sid for sid in geoms if geom.within(geoms[sid])} == expected, \
            (service, hosts)
        assert {sid for sid in geoms if geom.intersects(geoms[sid])} == expected, \
            (service, hosts)
    assert set(POINT_SERVICES) == {"clinic", "school", "bank", "police",
                                   "ration", "transport"}
    placed = {service: {host for host, _, _ in pts}
              for service, pts in POINT_SERVICES.items()}
    assert placed["clinic"] == {"H", "L", "T", "M", "G", "O1+O2", "I"}
    assert placed["school"] == {"L", "M", "O2", "N", "S", "T", "G"}
    assert placed["bank"] == {"H", "I"}
    assert placed["police"] == {"L", "O1"}
    assert placed["ration"] == {"M", "S"}
    assert placed["transport"] == {"H", "N"}

    # roads: two rows, and the SUM is load-bearing.
    assert len(roads) == 2, "the road layer must have two rows"
    summed = {sid: sum(road.intersection(geom).length for road in roads) / 1000
              for sid, geom in geoms.items()}
    for sid, km in ROAD_KM.items():
        assert abs(summed[sid] - km) < 1e-9, (sid, summed[sid], km)
    for sid in geoms:
        if sid not in ROAD_KM:
            assert summed[sid] == 0.0, sid
    first_only = roads[0].intersection(geoms["M"]).length / 1000
    assert first_only == 0.0, \
        "M must get its road length from the SECOND row only"


def main():
    geoms = {sid: _shapely(rings)
             for sid, (_, _, rings) in SETTLEMENTS.items()}
    points = [(service, host.split("+"), Point(*_pt(x, y)))
              for service, pts in POINT_SERVICES.items()
              for host, x, y in pts]
    roads = [LineString([_pt(*p) for p in coords]) for _, coords in ROADS]
    _assert_relations(geoms, points, roads)

    settlement_feats = []
    for sid, (uso, pop, rings) in SETTLEMENTS.items():
        area_km2 = _area_km2(rings)
        assert abs(geoms[sid].area / 1_000_000 - area_km2) <= 1e-12 * area_km2, sid
        geom = ({"type": "Polygon", "coordinates": _ring(rings[0])}
                if len(rings) == 1 else
                {"type": "MultiPolygon",
                 "coordinates": [_ring(ring) for ring in rings]})
        settlement_feats.append(_feature(
            geom, {"USO_AREA_U": sid, "USO_FINAL": uso, "population": pop,
                   "area_km2": area_km2}))
    _dump(OUT / "settlements.geojson", settlement_feats)

    service_feats = []
    for service, pts in POINT_SERVICES.items():
        for host, x, y in pts:
            service_feats.append(_feature(
                {"type": "Point", "coordinates": _pt(x, y)},
                {"service": service, "host": host}))
    for host, coords in ROADS:
        service_feats.append(_feature(
            {"type": "LineString", "coordinates": [_pt(*p) for p in coords]},
            {"service": "road", "host": host}))
    _dump(OUT / "services.geojson", service_feats)

    # No barrier coverage in this tier (spec § 2): an EMPTY collection, which
    # both implementations short-circuit on.
    _dump(OUT / "barriers.geojson", [])

    path = emit_checked_expected_values(MESSY, OUT / "expected_values.csv")
    print(f"wrote fixtures to {OUT}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Give the Oraculum generator the same final step**

In `scripts/generate_oraculum_fixtures.py`, add to the imports:

```python
from scripts.check_oraculum_invariants import emit_checked_expected_values
from tests.cities import ORACULUM
```

and replace the last two lines of `main()`:

```python
    _dump(OUT / "divergence" / "exhibit.geojson", exhibit_feats)
    print(f"wrote fixtures to {OUT}")
```

with:

```python
    _dump(OUT / "divergence" / "exhibit.geojson", exhibit_feats)
    print(f"wrote fixtures to {OUT}")

    # The CSV is part of the fixture, so the generator owns it too: the CI
    # drift step globs generate_*_fixtures.py, which is what makes a
    # reference change without a regenerated CSV fail the build. The output
    # is byte-identical to what tests/reference_impl.py's __main__ wrote.
    print(f"wrote {emit_checked_expected_values(ORACULUM, OUT / 'expected_values.csv')}")
```

- [ ] **Step 6: Run the generator**

Run: `uv run python scripts/generate_messy_fixtures.py`
Expected: no assertion error, and two lines —
`wrote fixtures to .../tests/fixtures/messy` and
`wrote .../tests/fixtures/messy/expected_values.csv`.

Then confirm the guard has teeth and the write is deterministic:

```bash
uv run python scripts/generate_messy_fixtures.py
uv run python scripts/generate_messy_fixtures.py
git status --porcelain -- tests/fixtures/messy/
uv run python scripts/check_oraculum_invariants.py
wc -l tests/fixtures/messy/expected_values.csv
```

Expected: four untracked files under `tests/fixtures/messy/` (identical
across the two runs — `git status` shows `??` and nothing else changes),
`OK` from the invariants script, and **2521** lines (2520 rows + header).

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cities.py -q`
Expected: **33 passed** (25 after Task 2 + 6 new parametrisations + 2 new tests).

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **317 passed** (309 + 8 new).

- [ ] **Step 9: Prove Oraculum is byte-identical**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
```

Expected: only the four new **untracked** (`??`) messy files. Nothing under
`tests/fixtures/oraculum/` is modified.

- [ ] **Step 10: Commit**

```bash
git add scripts/generate_messy_fixtures.py \
        scripts/check_oraculum_invariants.py \
        scripts/generate_oraculum_fixtures.py \
        tests/test_cities.py \
        tests/fixtures/messy/settlements.geojson \
        tests/fixtures/messy/services.geojson \
        tests/fixtures/messy/barriers.geojson \
        tests/fixtures/messy/expected_values.csv
git commit -m "feat(fixtures): the messy city — eleven settlements (DEL-24)

Non-rectangular H/L/T with an empty intersection but crossing envelopes, a
two-part M whose centroid falls in its own gap, G centred exactly on that
centroid (weight exactly 1, and the directed-bbox exhibit), an overlapping
O1/O2 pair sharing one clinic, an isolated I, an RV settlement N, a
no-population U, and a 2 m^2 sliver S. The generator re-derives every spec
2/4.3 relation before writing a byte, and installs expected_values.csv only
after check_oraculum_invariants passes on exactly the bytes to be committed.
generate_oraculum_fixtures.py gains the same final step; its output is
byte-identical.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: production fixtures and every proof, per city (spec § 4.1, § 4.2, § 6)

**Files:**
- Modify: `scripts/generate_production_fixtures.py`
- Modify: `tests/test_production_fixtures.py`
- Modify: `tests/test_profiles_match_reference.py` (only `test_profile_matches_reference` and the `expected` fixture)
- Modify: `tests/test_reference_impl.py` (`test_expected_values_csv_is_regenerable`, `test_invariants_guard_csv_wide`)
- Create (generated, committed): `tests/fixtures/messy/production/code-2025.csv`, `tests/fixtures/messy/production/manuscript.csv`

**Interfaces:**
- Consumes: `tests.cities.CITIES`, `tests.cities.ORACULUM` (Task 1); `tests.oraculum_fixtures.compute_oracle_frame(..., city=)` (Task 2); `scripts.check_oraculum_invariants.check(df=None, *, city=)` (Task 3).
- Produces:
  - `scripts.generate_production_fixtures.production_dir(city=ORACULUM) -> Path` = `city.fixtures / "production"`
  - `scripts.generate_production_fixtures.emit_profile(profile, out_path, city=ORACULUM) -> Path`
  - `scripts.generate_production_fixtures.REPO: Path` (already exists; now the test's repo root)
  - `PRODUCTION_DIR` is **removed** — `production_dir(city)` replaces it.

- [ ] **Step 1: Write the failing tests**

Rewrite the top of `tests/test_production_fixtures.py` (imports and the two
parametrised tests); `test_metric_set_is_explicit` and
`test_no_sys_path_hacks_and_no_monolith` keep every assertion, with only the
repo-root line changed:

```python
"""The committed production fixtures must be exactly what the generator emits.

Same contract as test_expected_values_csv_is_regenerable: without this a red
build could be 'fixed' by hand-editing the fixture, turning the refactor's
correctness proof into a record of whatever the code now does. From cycle 3C
this runs for BOTH cities (spec § 4.2).
"""
from pathlib import Path

import pytest

from scripts.generate_production_fixtures import (
    REPO, SERVICES, emit_profile, metric_columns, production_dir,
)
from tests.cities import CITIES, ORACULUM

PROFILES = ["code-2025", "manuscript"]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_fixture_is_regenerable(city, profile, tmp_path):
    committed = production_dir(city) / f"{profile}.csv"
    assert committed.exists(), f"missing committed fixture {committed}"
    regen = emit_profile(profile, tmp_path / f"{profile}.csv", city)
    # Read as bytes: .read_text() performs universal-newline translation,
    # which would silently hide a line-ending regression (e.g. CRLF).
    assert regen.read_bytes() == committed.read_bytes()


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_fixture_has_the_spec_shape(city, profile):
    path = production_dir(city) / f"{profile}.csv"
    data = path.read_bytes()
    assert b"\r" not in data, "fixtures are LF-only"
    text = data.decode()
    lines = text.splitlines()
    assert lines[0] == "profile,scenario,denom,settlement,metric,value"
    rows = [line.split(",") for line in lines[1:]]
    assert all(r[0] == profile for r in rows)
    # sorted by (scenario, denom, settlement, metric)
    keys = [(r[1], r[2], r[3], r[4]) for r in rows]
    assert keys == sorted(keys)
    assert {r[1] for r in rows} == {s.name for s in city.scenarios}


def test_production_dir_is_per_city():
    assert production_dir() == production_dir(ORACULUM)
    for city in CITIES:
        assert production_dir(city) == city.fixtures / "production"
    assert production_dir(ORACULUM) == (
        REPO / "tests" / "fixtures" / "oraculum" / "production")
```

In `test_no_sys_path_hacks_and_no_monolith`, replace the one line

```python
    repo = PRODUCTION_DIR.parents[3]
```

with

```python
    repo = REPO
```

Every other line of that test — the three `assert not ... exists()` blocks,
both `git grep` invocations and their pathspecs — is unchanged.

In `tests/test_profiles_match_reference.py`, replace the imports, the `CSV`
constant, the `expected` fixture and `test_profile_matches_reference`:

```python
from tests.cities import CITIES
from tests.oraculum_fixtures import (
    ORACLE_SCENARIOS, compute_oracle_frame, load_barriers, load_services,
    load_settlements,
)
from tests.reference_impl import RULESETS, compute_city

REFERENCE_DENOMS = ("pop", "popdensity")

# (city, scenario) — each city brings its OWN scenario table (spec § 6).
CASES = [(city, scenario) for city in CITIES for scenario in city.scenarios]


def case_id(case):
    city, scenario = case
    return f"{city.name}-{scenario.name}"


@pytest.fixture(scope="module")
def expected():
    return {city.name: pd.read_csv(city.fixtures / "expected_values.csv")
            for city in CITIES}


@pytest.mark.parametrize("denom", REFERENCE_DENOMS)
@pytest.mark.parametrize("case", CASES, ids=case_id)
@pytest.mark.parametrize("profile", sorted(PROFILE_RULES))
def test_profile_matches_reference(expected, profile, case, denom):
    city, scenario = case
    exp = reference_block(expected[city.name], PROFILE_RULES[profile],
                          scenario.name, denom)
    got = compute_oracle_frame(profile, types=scenario.exclusion_types,
                               stage=scenario.stage, denom=denom, city=city)
    assert set(got.index) == set(exp.index)
    for prod_col, metric in metrics_for(profile).items():
        for sid in exp.index:
            assert got.loc[sid, prod_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-12), (city.name, profile,
                                                   scenario.name, denom, sid,
                                                   prod_col)
```

Delete the now-unused `from pathlib import Path` at the top of that file
(the `CSV` constant it served is gone).

`PROFILE_RULES`, `METRIC_MAP`, `reference_block`, `metrics_for`,
`test_enums_cover_exactly_the_reference_table`,
`test_every_mapped_knob_is_one_the_reference_actually_implements` and
`test_exclusion_stage_maps_onto_dropped_before_neighbors` are **unchanged** —
the last three stay Oraculum-only as the spec requires (`ORACLE_SCENARIOS`
and the Oraculum loaders are still imported for them).

In `tests/test_reference_impl.py`, replace the last two tests:

```python
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_invariants_guard_csv_wide(city):
    from scripts.check_oraculum_invariants import check
    assert check(city=city) == []


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_expected_values_csv_is_regenerable(city, tmp_path):
    """The committed CSV must be exactly what reference_impl produces.

    Without this, a red build could be 'fixed' by hand-editing the CSV,
    silently turning the oracle into a record of production behavior
    instead of the equations (code review round 1, Critical).
    """
    regen = tmp_path / "regen.csv"
    emit_expected_values(regen, city)
    # bytes, not text: read_text() would normalise nothing here but would
    # hide a line-ending or encoding change in the committed CSV.
    assert regen.read_bytes() == (
        city.fixtures / "expected_values.csv").read_bytes()
```

and add `from tests.cities import CITIES` to that file's imports. The
module-level `CSV` constant stays — the hand-anchor tests still read it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_production_fixtures.py tests/test_profiles_match_reference.py tests/test_reference_impl.py -q`
Expected: FAIL — `ImportError: cannot import name 'production_dir'` (and
`'REPO'` is importable but `PRODUCTION_DIR` no longer exists), then, once the
generator exists, `AssertionError: missing committed fixture
tests/fixtures/messy/production/code-2025.csv`.

- [ ] **Step 3: Generalise `scripts/generate_production_fixtures.py`**

Replace the imports and `PRODUCTION_DIR`:

```python
from delhi_psi.config import load_config
from tests.cities import CITIES, ORACULUM
from tests.oraculum_fixtures import compute_oracle_frame


def production_dir(city=ORACULUM):
    """Where `city`'s per-profile production fixtures live."""
    return city.fixtures / "production"
```

(`ORACLE_SCENARIOS` is no longer imported here — `emit_profile` iterates the
city's own scenario table, which carries the same `(name, types, stage)`
triple and works for either city.)

Replace `emit_profile` and `main`:

```python
def emit_profile(profile, out_path, city=ORACULUM):
    """Write `profile`'s production fixture for `city` to out_path; return it."""
    methodology = load_config(profile).methodology
    columns = metric_columns(
        second_normalization=methodology.second_normalization)
    records = []
    for scenario in city.scenarios:
        for denom in DENOMS:
            frame = compute_oracle_frame(profile,
                                         types=scenario.exclusion_types,
                                         stage=scenario.stage, denom=denom,
                                         city=city)
            records.extend(frame_records(profile, frame, scenario.name, denom,
                                         columns))
    write_fixture(out_path, records)
    return out_path


def main():
    for city in CITIES:
        for profile in PROFILES:
            out_path = emit_profile(profile,
                                    production_dir(city) / f"{profile}.csv",
                                    city)
            print(f"wrote {out_path}")
```

`metric_columns`, `frame_records`, `write_fixture`, `PROFILES`,
`POINT_SERVICES`, `SERVICES`, `DENOMS`, `HEADER` and `REPO` are unchanged.
`write_fixture` still sorts by `(scenario, denom, settlement, metric)` before
writing, which is why the scenario list's ORDER cannot reach the bytes — the
Oraculum CSVs stay byte-identical.

- [ ] **Step 4: Generate the messy production fixtures**

Run: `uv run python scripts/generate_production_fixtures.py`
Expected: four `wrote …` lines — Oraculum's two (unchanged bytes) and the two
new messy files. Then:

```bash
wc -l tests/fixtures/messy/production/*.csv
git status --porcelain -- tests/fixtures/
```

Expected: `1401` lines for `code-2025.csv` (1400 rows: 3 scenarios × 2
denominators × {10, 9, 9} settlements × 25 metrics) and `1345` for
`manuscript.csv` (1344 rows, 24 metrics — no `norm_psi`); `git status` shows
only the two new **untracked** messy files.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_production_fixtures.py tests/test_profiles_match_reference.py tests/test_reference_impl.py -q`
Expected: **PASS** — `test_production_fixtures.py` 11 (4 + 4 + 1 + 2),
`test_profiles_match_reference.py` 35 (32 + 3),
`test_reference_impl.py` 20 (15 carried + 3 from Task 2, with two of the
carried tests now doubled).

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **336 passed** (317 + 19 new).

- [ ] **Step 7: Prove Oraculum is byte-identical**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
uv run python scripts/check_oraculum_invariants.py
git status --porcelain -- tests/fixtures/
```

Expected: `OK` from the invariants script, and `git status` showing only the
two new untracked messy production files.

- [ ] **Step 8: Commit**

```bash
git add scripts/generate_production_fixtures.py \
        tests/test_production_fixtures.py \
        tests/test_profiles_match_reference.py \
        tests/test_reference_impl.py \
        tests/fixtures/messy/production/code-2025.csv \
        tests/fixtures/messy/production/manuscript.csv
git commit -m "test(oracle): run every proof on both cities (DEL-24)

generate_production_fixtures emits tests/fixtures/<city>/production/<profile>.csv
for CITIES x PROFILES, iterating each city's own scenario table.
test_profile_matches_reference, the production-fixture regenerability and
shape tests, the expected-values round trip and the invariants guard are all
parametrised over both cities. Oraculum's three CSVs are byte-identical:
write_fixture sorts before writing, so the scenario list's order cannot
reach the bytes.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `tests/test_messy_fixtures.py` — every pathology pin (spec § 4.3)

Direct assertions against **production**, not comparisons against the
reference (§ 4.1 already proves those agree). Each pin records what
production does **today**; when DEL-19 or DEL-20 lands, the pin it flips is
the proof.

**Files:**
- Create: `tests/test_messy_fixtures.py`

**Interfaces:**
- Consumes: `tests.cities.MESSY` (Task 1); `tests.oraculum_fixtures.compute_oracle_frame(..., city=)` and `methodology_with(..., city=)` (Task 2); `tests.reference_impl.POINT_SERVICES` (existing); `delhi_psi.pipeline.compute_frames` and `delhi_psi.validate.ValidationError` (existing).
- Produces: nothing importable — this file is a leaf.

- [ ] **Step 1: Write the failing test**

Create `tests/test_messy_fixtures.py`:

```python
"""What production does on each messy-city pathology, TODAY (spec 3C § 4.3).

These are direct assertions against production, not comparisons against the
reference — `tests/test_profiles_match_reference.py` already proves the two
agree on this city to 1e-12. Each pin here RECORDS today's behaviour on a
pathology the real layer has and Oraculum cannot express:

  bbox adjacency invents neighbours (DEL-19)  -> the H/L, T/L and G/M pins
  overlapping polygons double-count a point (DEL-20) -> the O1/O2 clinic pin

When a fix lands, the pin it flips is the proof that it landed. A pin that
starts failing for any OTHER reason means adjacency, exclusion or counting
changed silently — investigate, never re-record.
"""
from functools import lru_cache

import pandas as pd
import pytest

from tests.cities import MESSY

PROFILES = ("code-2025", "manuscript")
DENOMS = ("pop", "popdensity")
SCENARIOS = {scenario.name: scenario for scenario in MESSY.scenarios}

# The two shipped profiles ARE the two adjacency rules on this city.
BBOX_PROFILE = "code-2025"     # methodology.adjacency.rule: bbox
TOUCH_PROFILE = "manuscript"   # methodology.adjacency.rule: touch


@lru_cache(maxsize=None)
def frame(profile, scenario_name, denom):
    """One production run, memoised. Treat the result as READ-ONLY: every
    test in this file shares the same object."""
    from tests.oraculum_fixtures import compute_oracle_frame

    scenario = SCENARIOS[scenario_name]
    return compute_oracle_frame(profile, types=scenario.exclusion_types,
                                stage=scenario.stage, denom=denom, city=MESSY)


def nbrs(result, sid):
    return set(result.loc[sid, "nbrs_bbox"])


def denominator(result, sid, denom):
    population = result.loc[sid, "population"]
    if denom == "pop":
        return population
    return population / result.loc[sid, "area_km2"]


@pytest.fixture(scope="module")
def geoms():
    city = MESSY.load_settlements().set_index("USO_AREA_U")
    return {sid: row.geometry for sid, row in city.iterrows()}


# --- bbox adjacency invents neighbours (DEL-19) ------------------------
def test_h_and_l_are_disjoint_but_each_reaches_the_others_envelope(geoms):
    """The bbox rule is DIRECTED, so each shape has to reach into the other's
    envelope for the pair to be neighbours both ways."""
    assert geoms["H"].intersection(geoms["L"]).is_empty
    assert not geoms["H"].intersection(geoms["L"].envelope).is_empty
    assert not geoms["L"].intersection(geoms["H"].envelope).is_empty


def test_bbox_pairs_are_directed_as_the_spec_says():
    got = frame(BBOX_PROFILE, "nopop_only", "pop")
    assert "L" in nbrs(got, "H") and "H" in nbrs(got, "L")
    assert "T" in nbrs(got, "L") and "L" in nbrs(got, "T")
    # the asymmetric one: an axis-aligned square IS its own envelope, so M
    # never reaches G even though G sits inside M's envelope.
    assert "M" in nbrs(got, "G")
    assert "G" not in nbrs(got, "M")


def test_touch_has_none_of_the_bbox_only_pairs():
    got = frame(TOUCH_PROFILE, "nopop_only", "pop")
    assert "L" not in nbrs(got, "H") and "H" not in nbrs(got, "L")
    assert "T" not in nbrs(got, "L") and "L" not in nbrs(got, "T")
    assert "M" not in nbrs(got, "G") and "G" not in nbrs(got, "M")


def test_corner_contact_between_t_and_l_is_a_single_point(geoms):
    shared = geoms["T"].intersection(geoms["L"])
    assert shared.geom_type == "Point"
    assert shared.length == 0


def test_overlapping_pair_are_neighbours_under_both_rules():
    """THE DEL-19 finding: the `touch` test asks for `.length > 0`, and an
    overlap polygon's `.length` is its PERIMETER — so two overlapping
    polygons are 'border-sharing' neighbours."""
    for profile in PROFILES:
        got = frame(profile, "nopop_only", "pop")
        assert "O2" in nbrs(got, "O1"), profile
        assert "O1" in nbrs(got, "O2"), profile


# --- the MultiPolygon and its gap --------------------------------------
def test_multipolygon_has_two_parts_and_its_centroid_lies_outside_it(geoms):
    assert geoms["M"].geom_type == "MultiPolygon"
    assert len(geoms["M"].geoms) == 2
    assert not geoms["M"].centroid.within(geoms["M"])


def test_gap_settlement_sits_exactly_on_the_multipolygons_centroid(geoms):
    assert geoms["G"].disjoint(geoms["M"])
    assert geoms["G"].centroid.equals(geoms["M"].centroid)
    assert geoms["G"].centroid.distance(geoms["M"].centroid) == 0.0


def test_bbox_neighbours_of_the_gap_settlement_are_exactly_the_multipolygon():
    assert nbrs(frame(BBOX_PROFILE, "nopop_only", "pop"), "G") == {"M"}


@pytest.mark.parametrize("denom", DENOMS)
def test_gap_settlement_clinic_pcen_is_the_undecayed_weight_one_case(denom):
    """d = 0 -> 1/(1+d) is exactly 1, so G's PCEN is a plain SUM of its own
    and M's clinic counts. Asserted with `==`, not approx: any decay at all
    would move it."""
    got = frame(BBOX_PROFILE, "nopop_only", denom)
    assert got.loc["G", "nbrs_dist_bbox"] == [("M", 0.0)]
    expected = ((got.loc["G", "clinic_count"] + got.loc["M", "clinic_count"])
                / denominator(got, "G", denom))
    assert got.loc["G", "clinic_pcen"] == expected


# --- overlapping polygons double-count a point (DEL-20) ----------------
def test_overlap_has_positive_area(geoms):
    assert geoms["O1"].intersection(geoms["O2"]).area > 0


def test_the_overlap_clinic_is_counted_for_both_owners():
    """One physical clinic, strictly inside the overlap, counted for BOTH.
    Agreed behaviour: production's `intersects` and the reference's `within`
    both do it, so it is pinned directly, never by comparison."""
    services = MESSY.load_services()
    assert (services["clinic"]["host"] == "O1+O2").sum() == 1
    for profile in PROFILES:
        got = frame(profile, "nopop_only", "pop")
        assert got.loc["O1", "clinic_count"] == 1, profile
        assert got.loc["O2", "clinic_count"] == 1, profile


# --- the isolated settlement -------------------------------------------
def test_isolated_settlement_has_no_neighbours_under_either_rule():
    for profile in PROFILES:
        got = frame(profile, "nopop_only", "pop")
        assert list(got.loc["I", "nbrs_bbox"]) == [], profile
        assert list(got.loc["I", "nbrs_dist_bbox"]) == [], profile


def test_isolated_settlement_is_disjoint_from_every_other(geoms):
    for sid, geom in geoms.items():
        if sid != "I":
            assert geoms["I"].disjoint(geom), sid


@pytest.mark.parametrize("denom", DENOMS)
@pytest.mark.parametrize("profile", PROFILES)
def test_isolated_settlement_clinic_pcen_is_own_over_denominator(profile,
                                                                 denom):
    got = frame(profile, "nopop_only", denom)
    assert got.loc["I", "clinic_pcen"] == (
        got.loc["I", "clinic_count"] / denominator(got, "I", denom))


# --- category exclusion (N) --------------------------------------------
@pytest.mark.parametrize("profile", PROFILES)
def test_rv_settlement_is_reported_only_when_it_is_not_excluded(profile):
    """`nopop_only` and `excl_rv_post` differ exactly by N, so it is the
    CATEGORY exclusion that removed it — not the missing-population rule."""
    assert "N" in frame(profile, "nopop_only", "pop").index
    assert "N" not in frame(profile, "excl_rv_post", "pop").index
    assert "N" not in frame(profile, "excl_rv_pre", "pop").index


def test_rv_settlement_leaves_o2s_neighbour_list_only_under_pre_neighbors():
    assert "N" in nbrs(frame(BBOX_PROFILE, "excl_rv_post", "pop"), "O2")
    assert "N" not in nbrs(frame(BBOX_PROFILE, "excl_rv_pre", "pop"), "O2")


# --- the settlement with no population row (U) -------------------------
@pytest.mark.parametrize("denom", DENOMS)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("profile", PROFILES)
def test_no_population_settlement_is_never_reported(profile, scenario, denom):
    """Production drops a no-population row UNCONDITIONALLY — every profile,
    every scenario, every denominator — because `dropped` is
    `excluded_ids ∪ missing`."""
    assert "U" not in frame(profile, scenario, denom).index


def test_no_population_settlement_leaves_o1s_neighbour_list_only_under_pre_neighbors():
    """...but it stays in other settlements' neighbour lists, unless the
    scenario's single `stage` is `pre_neighbors` — which strips the whole
    drop set, `U` included."""
    assert "U" in nbrs(frame(BBOX_PROFILE, "nopop_only", "pop"), "O1")
    assert "U" in nbrs(frame(BBOX_PROFILE, "excl_rv_post", "pop"), "O1")
    assert "U" not in nbrs(frame(BBOX_PROFILE, "excl_rv_pre", "pop"), "O1")


def test_missing_population_error_names_the_settlement_with_no_row():
    from delhi_psi.pipeline import compute_frames
    from delhi_psi.validate import ValidationError
    from tests.oraculum_fixtures import methodology_with

    methodology = methodology_with(BBOX_PROFILE, types=(),
                                   stage="post_neighbors", city=MESSY)
    with pytest.raises(ValidationError) as excinfo:
        compute_frames(MESSY.load_settlements(),
                       {"canal": MESSY.load_barriers()},
                       MESSY.load_services(), None, methodology, "pop",
                       mapping=MESSY.mapping(), scheme=MESSY.scheme,
                       missing_population="error")
    assert "'U'" in str(excinfo.value)
    assert "no population row" in str(excinfo.value)


def test_populations_are_distinct_and_only_u_has_none():
    city = MESSY.load_settlements().set_index("USO_AREA_U")
    assert pd.isna(city.loc["U", "population"])
    assert not pd.isna(city.loc["N", "population"])
    present = city["population"].dropna()
    assert len(present) == len(city) - 1 == 10
    assert len(set(present)) == len(present), "populations must not tie"


# --- the area-extreme sliver (S) ---------------------------------------
def test_sliver_area_is_exactly_two_square_metres(geoms):
    city = MESSY.load_settlements().set_index("USO_AREA_U")
    assert city.loc["S", "area_km2"] == 2e-06
    assert geoms["S"].area > 0
    assert geoms["S"].area == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize("profile", PROFILES)
def test_sliver_ration_pcen_is_the_popdensity_minimum_among_owners(profile):
    """Scoped to the settlements that OWN a ration point: a settlement with
    no ration point but a serving neighbour can sit above M (H does), and
    one with neither sits at exactly 0 (I always does)."""
    got = frame(profile, "nopop_only", "popdensity")
    owners = sorted(sid for sid in got.index if got.loc[sid, "ration_count"] > 0)
    assert owners == ["M", "S"]
    assert got.loc["S", "ration_pcen"] == min(
        got.loc[owner, "ration_pcen"] for owner in owners)
    assert got.loc["M", "ration_pcen"] / got.loc["S", "ration_pcen"] >= 1e4
    assert got.loc["I", "ration_pcen"] == 0.0


def test_no_ration_ordering_is_claimed_under_pop():
    """Under `pop` the area is irrelevant, and the order flips — which is
    exactly why the spec scopes the sliver claim to `popdensity`."""
    got = frame(BBOX_PROFILE, "nopop_only", "pop")
    assert got.loc["S", "ration_pcen"] > got.loc["M", "ration_pcen"]


# --- the two-row road layer and the full service set -------------------
def test_the_road_layer_has_two_rows_and_lengths_are_summed():
    roads = MESSY.load_services()["road"]
    assert len(roads) == 2
    got = frame(BBOX_PROFILE, "nopop_only", "pop")
    for sid, km in (("H", 1.2), ("L", 0.6), ("M", 2.0)):
        assert got.loc[sid, "road_length"] == pytest.approx(km, abs=1e-12), sid
    for sid in got.index:
        if sid not in ("H", "L", "M"):
            assert got.loc[sid, "road_length"] == 0.0, sid
    # M's whole length comes from the SECOND row, so "first row only" is
    # observably wrong here.
    multipolygon = MESSY.load_settlements().set_index(
        "USO_AREA_U").loc["M"].geometry
    assert roads.geometry.iloc[0].intersection(multipolygon).length == 0.0


def test_all_seven_services_are_present_on_both_sides():
    """The reference scores every service in POINT_SERVICES regardless, and
    production's PSI averages over the services present — so a service
    missing from this city would make the two average different things."""
    from tests.reference_impl import POINT_SERVICES

    services = MESSY.load_services()
    assert set(services) == set(POINT_SERVICES) | {"road"}
    got = frame(BBOX_PROFILE, "nopop_only", "pop")
    for service in POINT_SERVICES:
        assert f"{service}_pcen" in got.columns, service
        assert got[f"{service}_count"].sum() > 0, service
```

- [ ] **Step 2: Run the test to verify it fails**

Before writing anything, prove the file is a real gate rather than a
restatement: temporarily edit `tests/fixtures/messy/settlements.geojson` so
that `G`'s square is `_rect(6450, 450, 6550, 551)` (one metre taller, so its
centroid no longer coincides with `M`'s), then run

Run: `uv run pytest tests/test_messy_fixtures.py -q`
Expected: FAIL in
`test_gap_settlement_sits_exactly_on_the_multipolygons_centroid` and both
`test_gap_settlement_clinic_pcen_is_the_undecayed_weight_one_case` cases.
Then restore the file with `git checkout -- tests/fixtures/messy/settlements.geojson`
and re-run: **42 passed**.

(The pins are assertions about a fixture that already exists, so there is no
"module not found" red. This deliberate-sabotage red is what proves they
have teeth — the same technique the CI spec's verification plan uses.)

- [ ] **Step 3: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **378 passed** (336 + 42 new).

- [ ] **Step 4: Prove no fixture moved**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
```

Expected: **empty**.

- [ ] **Step 5: Commit**

```bash
git add tests/test_messy_fixtures.py
git commit -m "test(oracle): pin every messy-city pathology (DEL-24)

Directed bbox neighbours where the geometries are disjoint (H/L), where the
contact is a single point (T/L) and where the relation is one-way (M in G's
list, G not in M's); the overlap clinic counted for both O1 and O2; the
MultiPolygon centroid outside its own geometry with G on it at distance
exactly 0 (weight exactly 1); I isolated under both rules; N reported only
when the RV category is not excluded, and leaving O2's list only under
pre_neighbors; U never reported under any profile, scenario or denominator,
and leaving O1's list only under pre_neighbors; the 2 m^2 sliver's ration
PCEN as the popdensity minimum among ration owners, five orders below M's.

Each pin records TODAY. DEL-19 and DEL-20 are proven by the pin that flips.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `scripts/measure_layer_pathologies.py` — the tier's provenance (spec § 5)

The tier's premises ("0 of 4,357 are rectangles, 556 MultiPolygons, 4,050
overlapping pairs, 360 isolated, 15 with no population row, areas 2.3e-9 →
29 km²") came from an ad-hoc session. This makes them reproducible.

**Files:**
- Create: `scripts/measure_layer_pathologies.py`
- Create: `tests/test_layer_pathologies.py`
- Create: `docs/data/layer_pathologies.md` (hand-run output — Step 5)

**Interfaces:**
- Consumes: `delhi_psi.config.load_config`, `delhi_psi.io.read_layer` / `read_population`, `delhi_psi.geometry.reproject`, `delhi_psi.neighbors.adjacency`, `delhi_psi.pipeline._dedup_cached` / `attach_population` (all existing, all read-only usage).
- Produces:
  - `scripts.measure_layer_pathologies.resolve_cache_dir(cli_value=None) -> Path` — a fresh `tempfile.mkdtemp()` when `cli_value` is falsy
  - `scripts.measure_layer_pathologies.measure(cfg, cache_dir) -> dict[str, str | int]`
  - `scripts.measure_layer_pathologies.render(report) -> str` — the fenced ```` ```text ```` block
  - `scripts.measure_layer_pathologies.parse_block(text) -> dict[str, str]` — the inverse; the SAME parser reads the doc and the script's stdout
  - `scripts.measure_layer_pathologies.main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_layer_pathologies.py`:

```python
"""The messy tier's real-data premises, from a reproducible source (§ 5).

Data-gated: CI runs on a bare runner, so the two tests that actually run the
measurement skip without ~/delhi_data. The three that do not need data — the
committed document's shape, its provenance header, and where the dedup cache
is allowed to go — always run.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.measure_layer_pathologies import parse_block, resolve_cache_dir

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "layer_pathologies.md"
DATA_DIR = Path(os.environ.get("DELHI_DATA_DIR", "~/delhi_data")).expanduser()

needs_data = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"real Delhi data not present at {DATA_DIR}")

COUNT_KEYS = ("settlements", "rectangles", "multipolygons", "isolated_bbox",
              "no_population", "overlapping_pairs")
AREA_KEYS = ("area_km2_min", "area_km2_median", "area_km2_max")
POINT_SERVICES = ("bank", "health", "police", "ration", "school", "transport")


@pytest.fixture(scope="module")
def committed():
    assert DOC.exists(), f"missing {DOC} — run the script and commit its block"
    return parse_block(DOC.read_text())


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """ONE run of the script, shared by every data-gated test in this file:
    the pipeline's O(n^2) dedup takes ~3 minutes on a cold cache."""
    if not DATA_DIR.exists():
        pytest.skip(f"real Delhi data not present at {DATA_DIR}")
    cache = tmp_path_factory.mktemp("pathologies_cache")
    before = set(DATA_DIR.rglob("*"))
    proc = subprocess.run(
        [sys.executable, "scripts/measure_layer_pathologies.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--cache-dir", str(cache)],
        cwd=REPO, capture_output=True, text=True)
    after = set(DATA_DIR.rglob("*"))
    assert proc.returncode == 0, proc.stderr[-4000:]
    return parse_block(proc.stdout), before, after


def test_the_doc_has_the_fenced_block_with_every_required_key(committed):
    for key in COUNT_KEYS:
        assert key in committed, key
        assert committed[key].isdigit(), (key, committed[key])
    for key in AREA_KEYS:
        assert key in committed, key
        float(committed[key])          # parses, whatever the formatting
    services = sorted(key[len("multi_settlement_points_"):]
                      for key in committed
                      if key.startswith("multi_settlement_points_"))
    assert services == sorted(POINT_SERVICES), services
    assert set(committed) == (set(COUNT_KEYS) | set(AREA_KEYS)
                              | {f"multi_settlement_points_{s}"
                                 for s in POINT_SERVICES})


def test_the_doc_records_its_provenance():
    """Counts without a date, a layer and a commit are not evidence."""
    head = DOC.read_text().split("```text")[0]
    for label in ("**Run date:**", "**Layer:**", "**Commit:**", "**Command:**"):
        assert label in head, label


def test_the_cache_dir_default_is_a_fresh_directory_outside_the_data_dir():
    """~/delhi_data is bisynced to the shared drive: a cache written there
    propagates to everyone. The default must never be derived from it."""
    made = [resolve_cache_dir(), resolve_cache_dir()]
    try:
        assert made[0] != made[1], "each run must get its own cache"
        for path in made:
            assert path.is_dir()
            assert path != DATA_DIR and DATA_DIR not in path.parents
    finally:
        for path in made:
            shutil.rmtree(path, ignore_errors=True)
    assert resolve_cache_dir("/somewhere/else") == Path("/somewhere/else")


@needs_data
def test_a_fresh_run_reproduces_the_committed_counts(committed, fresh):
    """Counts only: the prose header (date, layer, commit) is never compared,
    and the three float area keys are checked for presence and parseability
    by the shape test instead of by text equality."""
    measured, _, _ = fresh
    assert set(measured) == set(committed)
    for key, value in committed.items():
        if key.startswith("area_km2_"):
            continue
        assert measured[key] == value, (key, measured[key], value)


@needs_data
def test_the_script_writes_nothing_under_the_data_directory(fresh):
    """~/delhi_data is bisynced hourly, so an unrelated file can appear
    mid-run; assert specifically that no dedup artifact — the only thing this
    script could ever write there — was created."""
    _, before, after = fresh
    created = after - before
    leaked = sorted(str(path) for path in created if ".dedup." in path.name)
    assert leaked == [], leaked
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_layer_pathologies.py -q`
Expected: collection error —
`ModuleNotFoundError: No module named 'scripts.measure_layer_pathologies'`.

- [ ] **Step 3: Write the measurement script**

Create `scripts/measure_layer_pathologies.py`:

```python
"""Measure the real settlement layer's pathologies (spec 3C § 5).

The messy-city tier's premises came from an ad-hoc session; this script is
their reproducible source — the same numbers, from the same layers, through
the same pipeline functions, on demand.

READ-ONLY over --data-dir. The deduplication cache goes under --cache-dir
(default: a fresh temporary directory) and NEVER under the data directory:
~/delhi_data is bisynced to the shared drive, so a stray file there
propagates to everyone.

    uv run python scripts/measure_layer_pathologies.py --config code-2025

Prints a provenance line, then the fenced block that
docs/data/layer_pathologies.md carries verbatim. The first run is slow — the
pipeline's O(n^2) dedup takes about three minutes on 4,357 rows — so pass
--cache-dir <a persistent directory OUTSIDE the data directory> to reuse it.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import geopandas as gpd

from delhi_psi import geometry, io, neighbors, pipeline
from delhi_psi.config import load_config

FENCE = "```text"


def resolve_cache_dir(cli_value=None):
    """Where the dedup cache goes. NEVER derived from the data directory."""
    if cli_value:
        return Path(cli_value).expanduser()
    return Path(tempfile.mkdtemp(prefix="delhi_psi_pathologies_"))


def load_settlements(cfg, cache_dir):
    """Read, deduplicate and reproject exactly as `pipeline.preprocess` does,
    so every count below describes the universe the pipeline actually scores.
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


def count_rectangles(gdf, *, rtol=1e-9):
    """A polygon is a rectangle iff it fills its own bounding box."""
    return int(sum(1 for geom in gdf.geometry
                   if abs(geom.area - geom.envelope.area)
                   <= rtol * geom.envelope.area))


def count_multipolygons(gdf):
    return int((gdf.geom_type == "MultiPolygon").sum())


def count_isolated_bbox(gdf, *, id_col):
    """Settlements with an EMPTY neighbour list under the production rule."""
    frame = neighbors.adjacency(gdf, id_col=id_col, neighbor_col="nbrs_bbox",
                                rule="bbox")
    return int(sum(1 for nbrs in frame["nbrs_bbox"] if len(nbrs) == 0))


def count_no_population(gdf, cfg):
    """The pipeline's own join, so the key and the rule are not re-invented."""
    population = io.read_population(
        cfg.paths.data_dir / cfg.layers.population.path)
    _, missing = pipeline.attach_population(
        gdf, population, id_col=cfg.layers.settlements.id_col,
        population_id_col=cfg.layers.population.id_col,
        population_value_col=cfg.layers.population.value_col)
    return len(missing)


def count_overlapping_pairs(gdf, *, id_col):
    """Pairs whose intersection has POSITIVE AREA — a shared border is not an
    overlap. The sjoin narrows the candidates so the area test never runs on
    all n^2 pairs; a self-join yields each unordered pair twice, and
    `left < right` keeps exactly one.
    """
    frame = gdf[[id_col, "geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(frame, frame, how="inner", predicate="intersects")
    geoms = frame.geometry
    return int(sum(
        1 for left, right in zip(joined.index, joined["index_right"])
        if left < right
        and geoms.iloc[left].intersection(geoms.iloc[right]).area > 0))


def count_multi_settlement_points(gdf, points, *, id_col):
    """Service points that fall inside MORE THAN ONE settlement (production
    counts such a point for every one of them)."""
    frame = gdf[[id_col, "geometry"]]
    pts = points[["geometry"]].reset_index(drop=True)
    joined = gpd.sjoin(pts, frame, how="inner", predicate="intersects")
    per_point = joined.groupby(joined.index).size()
    return int((per_point > 1).sum())


def measure(cfg, cache_dir):
    """The whole report, as an ordered {key: value} mapping."""
    id_col = cfg.layers.settlements.id_col
    gdf = load_settlements(cfg, cache_dir)
    areas = gdf["area_km2"]
    report = {
        "settlements": len(gdf),
        "rectangles": count_rectangles(gdf),
        "multipolygons": count_multipolygons(gdf),
        "isolated_bbox": count_isolated_bbox(gdf, id_col=id_col),
        "no_population": count_no_population(gdf, cfg),
        "area_km2_min": f"{areas.min():.6g}",
        "area_km2_median": f"{areas.median():.6g}",
        "area_km2_max": f"{areas.max():.6g}",
        "overlapping_pairs": count_overlapping_pairs(gdf, id_col=id_col),
    }
    for service, path in sorted(cfg.services.point.items()):
        points = io.read_layer(cfg.paths.data_dir / path)
        # `compute` drops exact-duplicate service rows before counting; do the
        # same here so a duplicated point is not reported as a pathology.
        points = points.drop_duplicates().reset_index(drop=True)
        points = geometry.reproject(points, cfg.crs.epsg)
        report[f"multi_settlement_points_{service}"] = \
            count_multi_settlement_points(gdf, points, id_col=id_col)
    return report


def render(report):
    """The fenced block docs/data/layer_pathologies.md carries verbatim."""
    return "\n".join([FENCE,
                      *(f"{key}: {value}" for key, value in report.items()),
                      "```"])


def parse_block(text):
    """The inverse of `render`. The SAME parser reads the committed document
    and this script's stdout, so the test compares like with like."""
    lines = text.splitlines()
    if FENCE not in lines:
        raise ValueError(f"no {FENCE} block found")
    out = {}
    for line in lines[lines.index(FENCE) + 1:]:
        if line.strip() == "```":
            return out
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    raise ValueError(f"unterminated {FENCE} block")


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
    cache_dir = resolve_cache_dir(args.cache_dir)
    data_dir = cfg.paths.data_dir.resolve()
    if cache_dir.resolve() == data_dir or data_dir in cache_dir.resolve().parents:
        raise SystemExit(
            f"--cache-dir {cache_dir} is inside the data directory "
            f"{data_dir}, which this script never writes to (it is bisynced "
            "to the shared drive)")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
    print(f"cache: {cache_dir}")
    print(render(measure(cfg, cache_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_layer_pathologies.py -q`
Expected: FAIL on the two `committed`-fixture tests with
`missing …/docs/data/layer_pathologies.md — run the script and commit its
block`, and PASS on
`test_the_cache_dir_default_is_a_fresh_directory_outside_the_data_dir`.
That is the correct red for Step 5.

- [ ] **Step 5: HAND-RUN — produce `docs/data/layer_pathologies.md`**

**The controller performs this step and supplies the numbers.** The dedup
cache in `~/delhi_data/phase3_verify/` is *inside* the data directory, so
copy it out rather than pointing `--cache-dir` at it (that would make a
stale-cache run write there):

```bash
mkdir -p /tmp/psi_pathologies_cache
cp ~/delhi_data/phase3_verify/settlements.dedup.gpkg \
   ~/delhi_data/phase3_verify/settlements.dedup.stamp \
   /tmp/psi_pathologies_cache/ 2>/dev/null || true
uv run python scripts/measure_layer_pathologies.py \
    --config code-2025 --cache-dir /tmp/psi_pathologies_cache
git rev-parse --short HEAD
```

(Without the cache copy the run takes about three minutes; with it, seconds.
Either way the numbers are identical — the stamp check re-runs the dedup if
the source layer has changed.)

Then create `docs/data/layer_pathologies.md`, pasting the fenced block the
script printed **verbatim** and filling the four header fields from the run:

```markdown
# Real-layer pathologies

Where the messy-city fixture tier's premises come from
(`docs/superpowers/specs/2026-08-28-messy-city-tier-design.md` § 5). Every
number below is produced by `scripts/measure_layer_pathologies.py`, which
reads the layers named by the `code-2025` profile, applies the pipeline's own
deduplication and population join, and writes nothing under the data
directory. `tests/test_layer_pathologies.py` re-runs it and compares the
counts (it skips when the data is not present).

- **Run date:** <YYYY-MM-DD>
- **Layer:** `uso_update_sep2021/uso_update_sep2021.shp`
- **Commit:** `<short sha of the commit this was measured at>`
- **Command:** `uv run python scripts/measure_layer_pathologies.py --config code-2025`

<PASTE THE FENCED ```text BLOCK THE SCRIPT PRINTED, VERBATIM>

## Reading the numbers

- `rectangles` — polygons that fill their own bounding box. Every one of the
  Oraculum city's seven settlements is one; this is what makes Oraculum
  unable to tell `bbox` adjacency apart from polygon intersection, and the
  messy city's `H`/`L`/`T` the fix.
- `isolated_bbox` — settlements with an EMPTY neighbour list under the
  production rule; the messy city's `I`.
- `no_population` — settlements the population join leaves without a value.
  Production drops them from the reported frame unconditionally; the messy
  city's `U`.
- `overlapping_pairs` — polygon pairs whose intersection has positive area.
  They are `touch` neighbours today (DEL-19) and they double-count any
  service point inside the overlap (DEL-20); the messy city's `O1`/`O2`.
- `multi_settlement_points_<service>` — points inside more than one
  settlement, counted for each. The `<service>` names are the `code-2025`
  profile's service layer names, so `health` here is the messy city's
  `clinic`.
```

Then run: `uv run pytest tests/test_layer_pathologies.py -q`
Expected: **5 passed** (nothing skipped — this machine has the data).

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **383 passed** (378 + 5 new). On a machine without `~/delhi_data`:
**381 passed, 2 skipped**.

- [ ] **Step 7: Confirm the data directory is untouched**

Run: `git status --porcelain -- tests/fixtures/` and, separately,
`ls -la ~/delhi_data | head` before and after the pytest run above.
Expected: empty `git status`; no new entries in `~/delhi_data`. (The
data-gated test already asserts this inside the suite; this is the eyeball
confirmation.)

- [ ] **Step 8: Commit**

```bash
git add scripts/measure_layer_pathologies.py \
        tests/test_layer_pathologies.py \
        docs/data/layer_pathologies.md
git commit -m "feat(provenance): measure the real layer's pathologies (DEL-24)

scripts/measure_layer_pathologies.py reports rectangles, MultiPolygons,
bbox-isolated settlements, settlements with no population row, the area
range, overlapping polygon pairs and per-layer multi-settlement points,
through the pipeline's own dedup and population join. It opens --data-dir
read-only and puts its cache in --cache-dir (default: a fresh temporary
directory), because ~/delhi_data is bisynced to the shared drive.

docs/data/layer_pathologies.md is the committed run; the data-gated test
re-runs the script and compares the counts, and asserts nothing appeared
under the data directory.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: docs, WORKPLAN, CHANGELOG and the real-data confirmation (spec § 6)

**Files:**
- Create: `docs/oracle/messy-city.md`
- Modify: `docs/methodology-config.md` § 4
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `WORKPLAN.md` (DEL-24 tick, DEL-19/20 pointers, bug-audit item 6 correction)
- Modify: `tests/test_messy_fixtures.py` (append two documentation tests)

**Interfaces:**
- Consumes: `tests.cities.MESSY` (Task 1).
- Produces: nothing importable.

- [ ] **Step 1: HAND-RUN — confirm the real-data baseline is untouched**

**The controller performs this step.** No production module was edited in
this cycle, so this can only be a no-op — run it anyway, because "can only
be" is an argument and this is evidence:

```bash
uv run delhi-psi preprocess --config code-2025 \
    --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3c_verify
uv run delhi-psi compute --config code-2025 \
    --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3c_verify
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3c_verify
```

Expected from the verifier, verbatim:

```
PASS — new run equivalent to July 2025 baseline within tolerance
```

with every reported `max abs deviation` line reading `0.000e+00` on all 23
columns. Paste the full output into the PR body, as for 3A/3B.

**Any non-zero deviation is a stop condition** (spec § 8) — do not tune a
tolerance, do not exclude a column. It would mean this cycle changed
production behaviour, which it has no business doing.

- [ ] **Step 2: Write the failing documentation tests**

Append to `tests/test_messy_fixtures.py`:

```python
# --- the tier's documentation (spec § 6) -------------------------------
DOC = Path(__file__).resolve().parent.parent / "docs" / "oracle" / "messy-city.md"


def test_the_messy_city_doc_documents_every_settlement():
    """Eleven settlements, each with a stated pathology and a stated pin. A
    settlement the doc does not name is a case nobody can maintain."""
    text = DOC.read_text()
    city = MESSY.load_settlements()
    for sid in city["USO_AREA_U"]:
        assert f"`{sid}`" in text, sid
    for pathology in ("MultiPolygon", "overlap", "isolated", "population",
                      "sliver", "envelope"):
        assert pathology.lower() in text.lower(), pathology
    assert "## How to add a case" in text


def test_methodology_config_section_4_says_the_proofs_run_on_both_cities():
    config_doc = (Path(__file__).resolve().parent.parent / "docs"
                  / "methodology-config.md").read_text()
    section = config_doc.split("## 4. What each proof guards")[1].split("## 5.")[0]
    assert "oraculum" in section.lower()
    assert "messy" in section.lower()
```

Add `from pathlib import Path` to that file's imports.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_messy_fixtures.py -q`
Expected: FAIL — `FileNotFoundError: …/docs/oracle/messy-city.md` and, in the
second test, `IndexError: list index out of range` (§ 4 does not yet mention
either city by name).

- [ ] **Step 4: Write `docs/oracle/messy-city.md`**

Create it with exactly this content (the coordinates are the verified ones
from this plan's "Canonical facts"; the pins are the ones
`tests/test_messy_fixtures.py` asserts):

```markdown
# The messy city

Oraculum is small, rectangular and hand-ratifiable by design; the real layer
is none of those. The messy city carries each real-layer pathology Oraculum
omits **once**, is scored by the independent reference implementation
(`tests/reference_impl.py`) rather than by hand arithmetic, and pins what
production does on each pathology **today** — so that the adjacency and
overlap fixes (DEL-19, DEL-20) are proven by a test that flips, not by
argument.

- Spec: `docs/superpowers/specs/2026-08-28-messy-city-tier-design.md`
- Generator: `scripts/generate_messy_fixtures.py` (asserts every relation
  below before writing a byte, and installs `expected_values.csv` only after
  `scripts/check_oraculum_invariants.py` passes on it)
- Fixtures: `tests/fixtures/messy/`
- Pins: `tests/test_messy_fixtures.py`
- Real-layer premises: `docs/data/layer_pathologies.md`

Oraculum stays the hand-ratifiable ground truth for the *math*
(`docs/oracle/derivation-worksheet.md`). This city has **no hand anchors** by
design: its numbers are whatever the reference says, and its job is the
*geometry*.

## The eleven settlements

Coordinates are EPSG:7760 metre offsets from `BASE_X = BASE_Y = 1_000_000`.
Vocabulary `messy-2` = `(Planned, RV)`. There are no barriers in this tier.

| id | type | pop | area km² | shape | what it pins |
|---|---|---|---|---|---|
| `H` | Planned | 110 | 1.33 | irregular hexagon | `H ∩ L = ∅`, yet each geometry reaches into the other's **envelope** → bbox neighbours both ways, `touch` neighbours neither way |
| `L` | Planned | 200 | 2.56 | concave L | the envelope-only relation with `H`; corner-only contact with `T` |
| `T` | Planned | 300 | 0.25 | triangle | contact with `L` is a single **Point** (length 0): a `bbox` neighbour, never a `touch` one |
| `M` | Planned | 400 | 2.0 | two-part MultiPolygon | its centroid `(6500, 500)` lies **outside** it, in the gap; its envelope spans the gap |
| `G` | Planned | 50 | 0.01 | 100 m square in `M`'s gap | centred **exactly** on `M`'s centroid: `M ∈ nbrs_bbox(G)` but `G ∉ nbrs_bbox(M)` (an axis-aligned square *is* its own envelope), and `d = 0` → decay weight exactly 1, the undecayed maximum |
| `O1` | Planned | 600 | 1.0 | rectangle | overlaps `O2` in a 200 m × 1000 m strip |
| `O2` | Planned | 700 | 1.0 | rectangle | one clinic strictly inside the overlap is counted for **both** (DEL-20); the pair are `touch` neighbours because an overlap polygon's `.length` is its perimeter (DEL-19) |
| `I` | Planned | 800 | 1.0 | far-away square | **isolated**: an empty neighbour list under both rules |
| `N` | **RV** | 900 | 1.0 | square beside `O2` | the settlement `code-2025` excludes by **category** — and it *has* a population, so exclusion is what removes it |
| `U` | Planned | **none** | 1.0 | square beside `O1` | **no population row**: production drops it unconditionally, under every profile and scenario |
| `S` | Planned | 100 | **2e-06** | 2 m × 1 m sliver on `H`'s edge | the area extreme: a `popdensity` denominator of 5e7, so its ration PCEN is the minimum among ration owners and five orders of magnitude below `M`'s |

All ten present populations are distinct, so no two settlements can tie by
construction. All seven services are placed (clinic in `H L T M G O1∩O2 I`;
school in `L M O2 N S T G`; bank in `H I`; police in `L O1`; ration in
`M S`; transport in `H N`), because the reference scores every service in
`POINT_SERVICES` regardless and production's PSI averages over the services
present — both sides must carry the same seven. The road is **two**
LineString rows (`H` 1.2 km, `L` 0.6 km, `M` 2.0 km), which is what makes
"sum every road row" load-bearing: `M`'s whole length comes from the second.

## The three scenarios

Every scenario drops `U`, with the scenario's own flag, because production
drops a no-population id unconditionally and applies its single `stage` to
the whole drop set (`dropped = excluded_ids ∪ missing`).

| name | reference `dropped` | before neighbours? | production side |
|---|---|---|---|
| `nopop_only` | `{U}` | no | `types: []`, `post_neighbors` |
| `excl_rv_post` | `{U, N}` | no | `types: [RV]`, `post_neighbors` |
| `excl_rv_pre` | `{U, N}` | yes | `types: [RV]`, `pre_neighbors` |

`nopop_only` and `excl_rv_post` differ exactly by `N`, so category exclusion
is genuinely exercised; `excl_rv_post` and `excl_rv_pre` differ exactly by
whether `N` and `U` stay in other settlements' neighbour lists.

## What is deliberately NOT here

- **Barriers.** `barriers.geojson` is an empty collection; multi-layer
  `combine` coverage needs a second barrier layer and is its own follow-up.
- **Hand anchors.** By design (see above).
- **Any rule change.** Edge-only adjacency, single-assignment overlap and
  `partial_weighted` are DEL-19/20/22, after Raj. This tier records today.
- **`L` has no `touch` neighbours** either, which is fine: it owns a school,
  so it is not part of the zero-tie that the schools in `T` and `G` exist to
  break (only `I` sits at exactly 0).

## How to add a case

1. Add the settlement (and any service point) to `SETTLEMENTS` /
   `POINT_SERVICES` / `ROADS` in `scripts/generate_messy_fixtures.py`, with a
   **distinct** population and an `area_km2` the shoelace helper computes.
2. Add the relation you are pinning to `_assert_relations` in that script —
   the generator must fail loudly if a later coordinate edit breaks it.
3. Run `uv run python scripts/generate_messy_fixtures.py`. If the invariants
   guard rejects the result (a degenerate min-max group, or a tied
   clinic/school argmin/argmax), nothing is written: give some settlement the
   service it needs and try again.
4. Run `uv run python scripts/generate_production_fixtures.py` to refresh
   `tests/fixtures/messy/production/*.csv`.
5. Add the production-side pin to `tests/test_messy_fixtures.py`.
6. Run `uv run pytest -q -W error`. `test_profile_matches_reference` proves
   the two implementations still agree on the new city at 1e-12; if it fails,
   you have found a real divergence — **report it, do not tune it away**.
```

- [ ] **Step 5: Add the one-line § 4 note to `docs/methodology-config.md`**

Replace this line in § 4:

```markdown
- `tests/test_production_fixtures.py` — every profile's numbers on the
  oracle city, byte-for-byte; the CI drift guard regenerates and diffs them
  on every push, so an accidental edit cannot pass.
```

with:

```markdown
- Every proof below runs on **both** fixture cities (`tests/cities.py`):
  **oraculum**, the small hand-ratified one, and **messy**, which carries the
  real layer's pathologies (`docs/oracle/messy-city.md`).
- `tests/test_production_fixtures.py` — every profile's numbers on each
  fixture city, byte-for-byte; the CI drift guard regenerates and diffs them
  on every push, so an accidental edit cannot pass.
```

and, in the same section, replace:

```markdown
- `tests/test_profiles_match_reference.py` — production == the independent
  reference implementation at 1e-12 for the profiles in `PROFILE_RULES`.
```

with:

```markdown
- `tests/test_profiles_match_reference.py` — production == the independent
  reference implementation at 1e-12 for the profiles in `PROFILE_RULES`, on
  every city × scenario × denominator.
- `tests/test_messy_fixtures.py` — what production does on each real-layer
  pathology today (bbox-invented neighbours, the overlap double count, the
  no-population drop). A pin here flips when DEL-19/DEL-20 land; that is the
  point.
```

- [ ] **Step 6: Run the documentation tests**

Run: `uv run pytest tests/test_messy_fixtures.py -q`
Expected: **44 passed** (42 from Task 5 + 2 new).

- [ ] **Step 7: Update `WORKPLAN.md`**

Four edits, all in the Phase 3 section.

(a) **Bug-audit item 1 (DEL-19)** — append to the end of the item, after
`[DEL-19]`:

```markdown
         Pinned today's behaviour on a purpose-built city: `H`/`L` are
         disjoint yet bbox neighbours both ways, `T`/`L` touch at a single
         point, and `M` is in `G`'s list while `G` is not in `M`'s
         (`tests/test_messy_fixtures.py`, `docs/oracle/messy-city.md`).
         Whichever rule Raj chooses, the fix flips those pins.
```

(b) **Bug-audit item 2 (DEL-20)** — append to the end of the item, after
`[DEL-20]`:

```markdown
         Pinned today's behaviour: one clinic strictly inside `O1 ∩ O2`
         is counted for both (`tests/test_messy_fixtures.py::
         test_the_overlap_clinic_is_counted_for_both_owners`). Agreed
         behaviour on both sides — production's `intersects` and the
         reference's `within` both do it — so it is asserted directly, never
         by comparison. The measured real-layer counts now have a
         reproducible source: `docs/data/layer_pathologies.md`.
```

(c) **Bug-audit item 6** — replace the last two sentences:

```markdown
         The resulting NaN is not caught by `check_no_negative` and
         `overall_psi`'s mean skips NaN, so `unnorm_psi` would silently
         average fewer services instead of failing. Routed to the 3C bug
         audit (not guarded in 3A).
```

with:

```markdown
         CORRECTED 28 Aug 2026 (3C): under `-W error` — which is how CI and
         every local run invoke pytest — the 0/0 does **not** produce a
         silent NaN; numpy emits `RuntimeWarning: invalid value encountered
         in scalar divide` and the warning filter turns it into a raised
         error. The silent-NaN path (uncaught by `check_no_negative`, skipped
         by `overall_psi`'s mean) exists only OUTSIDE a `-W error` run. Both
         fixture cities are therefore built so that no PCEN column is
         constant: `scripts/check_oraculum_invariants.py` refuses to write a
         fixture with a degenerate min-max group, and the generators call it
         before writing. Still routed to the bug audit: the guard belongs in
         `index.minmax`, and the decision (raise, or 0.0 as the reference
         does) is Raj's with DEL-13.
```

(d) **The DEL-24 item** — change `- [ ]` to `- [x]` and append after
`[DEL-24]`:

```markdown
      — done 28 Aug 2026 (3C): `tests/fixtures/messy/` carries eleven
      settlements covering all six pathologies, scored by the reference
      implementation and byte-stable through `scripts/generate_messy_fixtures.py`
      + `scripts/generate_production_fixtures.py`. A `City` abstraction
      (`tests/cities.py`) with two instances runs every proof on both cities;
      the reference was **generalised only** (a city, an explicit scenario
      table, all road rows summed) — not one rule changed, no production code
      was touched, Oraculum's three CSVs are byte-identical and the real-data
      baseline is at `0.000e+00`. Spec
      `docs/superpowers/specs/2026-08-28-messy-city-tier-design.md`, plan
      `docs/superpowers/plans/2026-08-28-messy-city-tier.md`, docs
      `docs/oracle/messy-city.md` and `docs/data/layer_pathologies.md`
```

Also, in bug-audit item 1, append `(reproducible source:
`docs/data/layer_pathologies.md`)` after the sentence beginning "Measured on
the real layer:", and leave every existing number in place — the pointer is
added, the prose is not deleted.

- [ ] **Step 8: Add the CHANGELOG entry**

Insert at the top of the `[Unreleased]` list in `CHANGELOG.md`, above the
Phase 3B bullet:

```markdown
- Phase 3C messy-city fixture tier (DEL-24): a **second** fixture city,
  `tests/fixtures/messy/`, carrying every real-layer pathology Oraculum omits
  by construction — eleven settlements with an irregular hexagon and a
  concave L that are **disjoint** yet bbox neighbours both ways, a triangle
  meeting the L at a single point, a two-part MultiPolygon whose centroid
  falls in its own gap with a square sitting exactly on that centroid
  (distance 0, decay weight exactly 1, and the directed-bbox exhibit: `M` is
  in `G`'s neighbour list, `G` is not in `M`'s), an overlapping pair sharing
  one clinic, an isolated settlement, a settlement with **no population
  row**, and a 2 m² sliver. It is scored by the independent reference
  implementation, never by hand arithmetic, and pins what production does on
  each pathology **today**, so the DEL-19 (bbox adjacency) and DEL-20
  (overlap double count) fixes will be proven by a test that flips.
  A `City` abstraction (`tests/cities.py`) with two instances now drives
  every proof: the reference match, the production fixtures, the
  expected-values round trip and the invariants guard all run on both cities.
  The reference implementation was **generalised only** — `compute_city`
  takes an explicit scenario table, `emit_expected_values` takes a city, and
  `_service_amounts` sums **every** road row (the messy city has two, and the
  second is where `M`'s whole road length comes from); not one rule changed.
  `scripts/generate_messy_fixtures.py` re-derives every geometric relation
  before writing a byte and installs `expected_values.csv` only after the
  invariants guard passes on exactly the bytes to be committed;
  `scripts/generate_oraculum_fixtures.py` gained the same step.
  `scripts/measure_layer_pathologies.py` + `docs/data/layer_pathologies.md`
  give the tier's real-data premises a reproducible source, read-only over
  `~/delhi_data`. Bug-audit item 6 corrected: `index.minmax`'s missing
  `hi == lo` guard **raises** under `-W error` (numpy's
  `invalid value encountered in scalar divide`), it does not silently NaN —
  the NaN path exists only outside a `-W error` run.
  **No production code changed at all**: nothing under `delhi_psi/` was
  touched, Oraculum's `expected_values.csv` and both Oraculum production CSVs
  are byte-identical, and `scripts/verify_against_baseline.py --config
  code-2025` still reports `PASS — new run equivalent to July 2025 baseline
  within tolerance` with `0.000e+00` on all 23 columns (real-data proof,
  28 Aug 2026). Tests 281 → 385. Docs: `docs/oracle/messy-city.md`,
  `docs/data/layer_pathologies.md`, `docs/methodology-config.md` § 4.
```

- [ ] **Step 9: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: **385 passed** (383 + 2 new).

- [ ] **Step 10: Final proof — fixtures, invariants, figures**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
uv run python scripts/check_oraculum_invariants.py
uv run python tests/reference_impl.py
uv run python scripts/render_oracle_maps.py
git status --porcelain -- tests/fixtures/ docs/oracle/
```

Expected: `OK` from the invariants script, `wrote …` for both cities from
`reference_impl.py`, and an **empty** `git status`.

- [ ] **Step 11: Commit**

```bash
git add docs/oracle/messy-city.md docs/methodology-config.md \
        CHANGELOG.md WORKPLAN.md tests/test_messy_fixtures.py
git commit -m "docs: the messy city, and the 3C record (DEL-24)

docs/oracle/messy-city.md documents all eleven settlements, what each pins,
the three scenarios and how to add a case. methodology-config 4 records that
every proof now runs on both cities. WORKPLAN ticks DEL-24, points DEL-19 and
DEL-20 at the pins that will flip when they land, and corrects bug-audit item
6: index.minmax RAISES under -W error, it does not silently NaN.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Done when

- `uv run pytest -q -W error` reports **385 passed** (281 carried over, none
  with a changed expected value).
- `tests/fixtures/oraculum/expected_values.csv`,
  `tests/fixtures/oraculum/production/code-2025.csv` and
  `tests/fixtures/oraculum/production/manuscript.csv` are **byte-identical**
  to their state at HEAD 2e612d8, and `git status --porcelain --
  tests/fixtures/` is empty after re-running every generator.
- `git diff --stat b2e8e51 -- delhi_psi/` is **empty**: no production code
  changed.
- `scripts/verify_against_baseline.py --config code-2025` reports
  `PASS — new run equivalent to July 2025 baseline within tolerance` with
  `0.000e+00` on all 23 columns.
- The messy city's reference and production agree on every scenario ×
  denominator × metric at `abs=1e-12`, and
  `scripts/check_oraculum_invariants.py` exits 0 for both cities.
