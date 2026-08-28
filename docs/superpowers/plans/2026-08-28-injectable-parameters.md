# Injectable Parameters — Distance Band and Decay Forms (DEL-18) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the last two methodological choices that are still code — the distance-based neighbourhood and the `1/(1+D)` decay weight — opt-in config *values* with reference-implementation rules and oracle pins on both fixture cities, so a Phase 6 sweep point (DEL-36/37/39) is one YAML file and nothing else.

**Architecture:** `methodology.adjacency.rule` gains a third value, `within_distance`, with a new conditional parameter `adjacency.max_distance_km` (polygon-to-polygon band, metres in EPSG:7760, `>= 0`); `methodology.decay` gains `form` (`inverse_linear` | `none` | `inverse_power` + `exponent` | `exponential` + `scale_km`) and a required `distance` (`centroid` | `boundary`). One new plumbing table, `tests/variants.py`, is read by BOTH sides — the reference implementation builds `VARIANT_RULESETS` from it by renaming block keys only, and the production tests build a `MethodologyConfig` from it — so the two cannot drift. Eight derived variants are scored by the independent reference into two new committed fixtures (`tests/fixtures/{oraculum,messy}/variants_expected_values.csv`) and production is required to reproduce them at 1e-12. Nothing about the existing profiles' behaviour moves: they gain exactly one key, `decay.distance: centroid`, which names the definition they have always used.

**Tech Stack:** Python 3.13 / uv, hatchling, geopandas 1.1.4, shapely 2.1.2, pandas 3.0.5, pyproj, joblib, tqdm, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-injectable-parameters-design.md` (rev 3, approved 2026-08-28 — read it in full first; § 0 fixes the non-goals, § 1 the config surface, § 2 the production code, § 3 the reference, § 4 the tests and pins, § 5 the docs, § 6 the task shape, § 8 the definition of done, § 9 the process). Parents: `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` (§ 3 schema, § 4 fixtures), `docs/superpowers/specs/2026-08-28-messy-city-tier-design.md` (the second city), `docs/superpowers/specs/2026-08-24-ci-workflow-design.md` (the drift guard).

## Global Constraints

State each of these to yourself before every task; they are every task's requirements, implicitly. The first five are copied verbatim from the spec (§ 0, § 8, § 9).

- **No change to the existing profiles' behaviour.** `code-2025.yaml` and `manuscript.yaml` do not use any new value. Their outputs, both cities' `expected_values.csv`, both cities' `production/*.csv`, and the real-data verify are byte-identical / 0.000e+00 at the end of the cycle.
- **`tests/fixtures/{oraculum,messy}/expected_values.csv` and `production/*.csv` must be byte-identical to `main`.** Prove it in EVERY task that touches `tests/`, `scripts/` or `delhi_psi/` by running

  ```bash
  for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
  git status --porcelain -- tests/fixtures/
  ```

  and requiring the `git status` output to be **empty** — with one exception, Task 3, where the only acceptable output is the two new untracked `variants_expected_values.csv` files that task itself creates and commits. Any modified, deleted **or untracked** file under `tests/fixtures/` is a failure; that is exactly what the CI drift guard checks.
- **No new shipped profile.** Phase 6 picks the sweep points and adds the profiles (and their production fixtures) with the reference rules already in place. This cycle proves the rules on *derived* profiles only. Nothing new appears in `delhi_psi/profiles/`, in `scripts/generate_production_fixtures.PROFILES`, or in `tests/test_production_fixtures.PROFILES`.
- **The reference INDEPENDENCE RULE holds.** `tests/reference_impl.py` must never import, call, or mirror the production spatial-index library — no `delhi_psi` import, directly or transitively.
- **`tests/variants.py` imports nothing from the repo.** Like `tests/cities.py`, it is a plumbing table: the standard library only — never `delhi_psi`, never `tests.reference_impl`, never `tests.cities`. A test pins this by reading the module source.
- **`decay.distance` is REQUIRED — no default.** The repo's rule is "methodology has no defaults, never inherited" (`config.py` module docstring) and this cycle does not add an exception. The key is added as `distance: centroid` in exactly three places, and there is no fourth: `delhi_psi/profiles/code-2025.yaml`, `delhi_psi/profiles/manuscript.yaml`, and the `MINIMAL` profile string in `tests/test_config.py` (`test_defaults_equal_code_2025` compares `minimal.methodology == full.methodology`, so it requires `centroid` specifically).
- **All new tests run under `uv run pytest -q -W error`,** and after **every** task the whole suite must be green under that exact command. The suite is **386 tests** at HEAD `6afc2c6`; it only ever grows.
- **Named-file commits only.** Never `git add -A`, never `git commit -a` — every commit names its files (review agents may be running: memory note "Review agents: isolate worktree"). Branch: `del-18-knobs`, worktree `/home/bwbelljr2/delhi_spatial_index/.claude/worktrees/del-18`.
- **Never write under `~/delhi_data`** — with ONE established exception: the verify output directory `~/delhi_data/phase3_verify` used by 3A–3C (controller ruling; it already exists on the bisynced share, holds the warm GPKG cache, and is not a baseline file). Task 5's real-data verify writes there and nowhere else under the data dir; the 10 km timing run writes to a tempdir.
- Numeric tolerance: `abs=1e-12` for in-memory comparisons, `abs=1e-9` across a CSV round trip (the house precedent, `tests/test_oracle_e2e.py`), exact `==` only where this plan says a value is exact — every such value was verified by running it (see "Canonical facts").
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- **Stop and report** (spec § 9) — do not work around, do not tune away — if: any existing profile's behaviour changes; any committed `expected_values.csv` or `production/*.csv` changes; the real-data verify deviates from 0.000e+00; or a write under `~/delhi_data` would be needed.

## File Structure

```
tests/
  variants.py                       NEW — VARIANTS + the band constants        (Task 1)
  reference_impl.py                 within_distance, 4 decay forms, boundary
                                    distance, VARIANT_RULESETS,
                                    emit_variant_expected_values               (Task 1)
  test_variant_rules.py             NEW — spec § 4.1 items 1-6 (reference)     (Task 1)
                                         spec § 4.1 item 7 (config)            (Task 2)
  test_config.py                    MINIMAL gains `distance: centroid`         (Task 2)
  test_profiles_match_reference.py  knob_for_key + EXTRA_PARAMS                (Task 2)
  test_reference_impl.py            variants-CSV regenerability + shape         (Task 3)
  test_neighbors.py                 within_distance + boundary_distances       (Task 4)
  test_index.py                     the four decay forms                       (Task 4)
  test_variants_match_reference.py  NEW — spec § 4.2 (in-memory)               (Task 4)
                                         spec § 4.2 (the CLI leg)              (Task 5)
  test_cli.py                       stamp literal + spec § 4.5                 (Task 4)
  oraculum_fixtures.py              variant_methodology                        (Task 4)
                                    oracle_profile_path(methodology_overrides) (Task 5)
  fixtures/oraculum/variants_expected_values.csv   NEW                         (Task 3)
  fixtures/messy/variants_expected_values.csv      NEW                         (Task 3)
delhi_psi/
  config.py                         enums, dataclasses, conditional validation (Task 2)
  profiles/code-2025.yaml           `decay.distance: centroid` + comments      (Task 2)
  profiles/manuscript.yaml          `decay.distance: centroid` + comments      (Task 2)
  neighbors.py                      _adjacency_within_distance,
                                    boundary_distances                          (Task 4)
  index.py                          _decay dispatch, exponent/scale_km wiring  (Task 4)
  pipeline.py                       band forwarding + log, boundary column,
                                    stamp                                       (Task 4)
scripts/
  check_oraculum_invariants.py      check_bands,
                                    emit_checked_variant_expected_values       (Task 3)
  generate_oraculum_fixtures.py     one call                                    (Task 3)
  generate_messy_fixtures.py        one call                                    (Task 3)
docs/
  methodology-config.md             § 1 rows, § 4 proofs, NEW § 6              (Task 6)
CHANGELOG.md, WORKPLAN.md                                                       (Task 6)
```

`.github/workflows/ci.yml` does **not** appear: the drift guard already globs `scripts/generate_*_fixtures.py` and already fails on any modified **or untracked** file under `tests/fixtures/`, so the two new CSVs are covered with no workflow edit. `delhi_psi/cli.py` and `delhi_psi/io.py` do not appear either — there is no CLI change and the output column set is unchanged (spec § 2.4).

## Canonical facts (verified against the repo on 2026-08-28 — do not re-derive)

Everything in this section was checked by running it in this worktree. Do not re-derive it; if something here disagrees with what you observe, **stop and report**.

- HEAD is `6afc2c6`, branch `del-18-knobs`, tree clean. `uv run pytest -q -W error` collects **386 tests**.
- Versions: geopandas **1.1.4**, shapely **2.1.2**, pandas **3.0.5** — `predicate="dwithin"` is supported.
- **`sjoin(predicate="dwithin", distance=X)` equals brute force.** For both cities, at X = 0 / 250 / 750 / 1000 m, the DIRECTED pair sets from `gpd.sjoin(g, g, how="left", predicate="dwithin", distance=X)` (self pair removed) are identical to `geom_i.distance(geom_j) <= X` computed pairwise. Directed counts: Oraculum 20 / 24 / 28 / 32; messy 10 / 16 / 20 / 26. Every polygon matches ITSELF under `dwithin` at every radius including 0, so the left join never produces a NaN right-hand id and no neighbour list can pick one up.
- **Band radii are B₁ = 0.25 km and B₂ = 0.75 km** (spec § 3). Never use 1.0 km: three fixture pairs sit at exactly 1.000000 km.
- **UNDIRECTED pair counts on `adjacency(...)`'s own output, BEFORE any barrier rule:**

  | city | 0 km | 0.25 km | 0.75 km |
  |---|---|---|---|
  | oraculum | **10** | **12** | **14** |
  | messy | **5** | **8** | **10** |

  The added pairs are exactly:

  | city | added at 0.25 km | added at 0.75 km |
  |---|---|---|
  | oraculum | `A–RV` (0.100 km), `C–RV` (0.100 km) | `B–D` (0.500 km), `B–IND` (0.500 km) |
  | messy | `H–L` (0.131519 km), `L–S` (0.199 km), `H–T` (0.223607 km) | `G–M` (0.450 km), `S–T` (0.630242 km) |

  The bands are strictly nested (`nbrs(0) ⊂ nbrs(0.25) ⊂ nbrs(0.75)`) and symmetric on both cities at every radius.
- **Band 0 vs the other rules.** On BOTH cities `within_distance` at 0 equals the reference's `intersects`. On Oraculum it also equals `touch` and equals undirected `bbox` (all three are the same 10 pairs — no corner-only contacts, no overlaps). On messy, `within_distance` 0 = `touch` ∪ {`L–T`} exactly: `L ∩ T` is a **Point** (`length 0`), which `touch` rejects, while `O1 ∩ O2` is a **Polygon** whose `.length` is its perimeter, **2400.0 m**, which `touch` accepts. Both `L–T` and `O1–O2` are also `bbox` pairs. Messy `touch` is 4 pairs: `H–S`, `N–O2`, `O1–O2`, `O1–U`.
- **Directed band tables** (used verbatim in Task 4's `neighbors` test):

  Oraculum, 0 km — `A: {B,D,E}`, `B: {A,C,E,RV}`, `C: {B,E,IND}`, `RV: {B}`, `D: {A,E}`, `E: {A,B,C,D,IND}`, `IND: {C,E}`
  Oraculum, 0.25 km — `A: {B,D,E,RV}`, `B: {A,C,E,RV}`, `C: {B,E,IND,RV}`, `RV: {A,B,C}`, `D: {A,E}`, `E: {A,B,C,D,IND}`, `IND: {C,E}`
  Oraculum, 0.75 km — `A: {B,D,E,RV}`, `B: {A,C,D,E,IND,RV}`, `C: {B,E,IND,RV}`, `RV: {A,B,C}`, `D: {A,B,E}`, `E: {A,B,C,D,IND}`, `IND: {B,C,E}`
- **Boundary vs centroid distances (km), verified to full precision:**

  | pair | boundary (polygon-to-polygon) | centroid |
  |---|---|---|
  | `A–RV` (oraculum) | `0.1` | `1.4142135623730951` |
  | `C–RV` (oraculum) | `0.1` | `1.4142135623730951` |
  | `A–B` (oraculum) | `0.0` | `1.0` |
  | `A–E` (oraculum) | `0.0` | `1.4142135623730951` |
  | `RV–B` (oraculum) | `0.0` | `1.0` |
  | `D–E` (oraculum) | `0.0` | `1.5` |
  | `B–D`, `B–IND` (oraculum) | `0.5` | `1.8027756377319948` |
  | `H–L` (messy) | `0.13151918984428584` | `1.1272365695995052` |
  | `H–S` (messy) | `0.0` | `0.6708512155398708` |
  | `H–T` (messy) | `0.22360679774997896` | `1.6306077980138336` |
  | `L–S` (messy) | `0.199` | `0.6650385327182178` |
  | `S–T` (messy) | `0.6302420170061656` | `1.3376310423863118` |
  | `G–M` (messy) | `0.45` | **`0.0`** — the one pair where boundary > centroid |
- **`RULESETS["code"]` neighbourhoods after the barrier rule** (`bbox` + `global_asymmetric`; the canal flags `A` and `D` on Oraculum, messy has no barriers): Oraculum `A: {B,E}`, `B: {C,E,RV}`, `C: {B,E,IND}`, `RV: {B}`, `D: {E}`, `E: {B,C,IND}`, `IND: {C,E}` — 9 undirected pairs. **`RV` and `D` are the only settlements with exactly one neighbour**, at centroid distances **1.0 km** (`RV→B`) and **1.5 km** (`D→E`). Messy `H: {L,S}`, `L: {H,T}`, `T: {L}`, `M: {}`, `G: {M}`, `O1: {O2,U}`, `O2: {N,O1}`, `I: {}`, `N: {O2}`, `U: {O1}`, `S: {H,L}`.
- **Own amounts, Oraculum** — clinic `A 2, B 1, C 0, RV 2, D 0, E 1, IND 0`; school `A 1, D 1, E 1`, others 0; populations `A 100, B 200, C 400, RV 100, D 100, E 300, IND 10`.
  **Own amounts, messy** — clinic `H 1, L 1, T 1, M 1, G 1, O1 1, O2 1, I 1, N 0, U 0, S 0`; school `L 1, T 1, M 1, G 1, O2 1, N 1, S 1`, `H 0, O1 0, I 0, U 0`; populations `H 110, L 200, T 300, M 400, G 50, O1 600, O2 700, I 800, N 900, S 100`, `U` **null**.
- **Verified closed-form pins** (reference, `city.scenarios[0]`, `denom="pop"`), each computed twice — by the reference and by hand:

  | variant | pin | value |
  |---|---|---|
  | `band0_none` | oraculum `B.clinic_pcen` = (1 + 0 + 1 + 2)/200 | `0.02` |
  | `pow2` | oraculum `RV.clinic_pcen` = (2 + 1·1/(1+1.0)²)/100 | `0.0225` |
  | `pow2` | oraculum `D.clinic_pcen` = (0 + 1·1/(1+1.5)²)/100 | `0.0016` |
  | `exp1` | oraculum `RV.clinic_pcen` = (2 + 1·e^−1)/100 | `0.023678794411714423` |
  | `exp1` | oraculum `D.clinic_pcen` = (0 + 1·e^−1.5)/100 | `0.0022313016014842983` |
  | `boundary` | oraculum `A.clinic_pcen` = (2 + 1 + 1)/100 | `0.04` |
  | `boundary` | messy `O1.clinic_pcen` = (1 + 1)/600 | `0.0033333333333333335` |
  | `boundary` | messy `G.school_pcen` = (1 + 1/1.45)/50 | `0.033793103448275866` |
  | `code` | messy `G.school_pcen` = (1 + 1)/50 | `0.04` |
  | `band_small` | oraculum `A.clinic_pcen` = (2 + 1/2 + 3/(1+√2))/100 | `0.037426406871192856` |
  | `band_small_boundary` | oraculum `A.clinic_pcen` = (2 + 1 + 1 + 2/1.1)/100 | `0.05818181818181818` |
  | `band_small` | messy `H.school_pcen` | `0.013170282557128916` |
  | `band_small_boundary` | messy `H.school_pcen` = (1/1.13151918984428584 + 1 + 1/1.22360679774997896)/110 | `0.024554760031472653` |

  `pow1` reproduces `code` at **exactly 0.0** worst deviation on every column, both cities, both denominators (`x**1.0 == x` in IEEE).
- **The whole § 4.2 comparison was rehearsed** with prototypes of the production changes in this plan: production == reference for 8 variants × 2 cities × 2 denominators × every `METRIC_MAP` column, worst absolute deviation **2.220446e-16**, well inside `abs=1e-12`. The reported id sets match too (messy drops `U`, which has no population row).
- **The invariants guard passes on the variants output.** `scripts.check_oraculum_invariants.check(df, city=...)` returns `[]` for both cities' variant frames — no degenerate min-max group and no tied clinic/school argmin/argmax in any of the 8 rules × 1 scenario × 2 denominators × 7 `_pcen` metrics.
- **Variants CSV shape:** oraculum **2576** rows (7 settlements × 8 variants × 2 denominators × 23 metrics), messy **3680** rows (10 reported settlements × 8 × 2 × 23). Scenario column is `baseline` (oraculum) / `nopop_only` (messy) — `city.scenarios[0]`.
- `tests/test_index.py`'s existing case `(dict(decay_form="exponential"), "exponential")` keeps passing after Task 4 (the new message is `decay form 'exponential' requires scale_km`, which still matches), but it no longer tests what it says. Task 4 replaces it (see that task).
- `tests/test_cli.py` already carries `COLLAPSE_TO_REFERENCE` (CLI output column → reference metric name, including `health_* → clinic_*` and `road_length → road_length_km`); Task 5 reuses it rather than writing a third copy.
- `delhi_psi/io.py:23` — `SHAPEFILE_DROP_COLUMNS = ("nbrs_bbox", "nbrs_dist_bbox", "centroid")`. The compute-local boundary column must never reach `io`, which is why `index_frames` drops it before returning.
- Only `delhi_psi/config.py` constructs `AdjacencyConfig` / `DecayConfig` / `MethodologyConfig` today; every test reaches a methodology through `load_config` or `tests.oraculum_fixtures.methodology_with`. Adding required fields therefore touches no other call site.

## Spec ambiguities and interpretations (recorded so a reviewer can re-open them)

1. **Task order 1 ↔ 2 is swapped relative to spec § 6.** Spec § 6 puts config first and the reference second, but the config change adds `within_distance`, `inverse_power`, `exponential`, `none`, `boundary` to `REFERENCE_KNOBS`, and `test_every_mapped_knob_is_one_the_reference_actually_implements` drives `compute_city` once per mapped value — so a config-first task would leave the suite RED until the reference lands (which is exactly what `docs/methodology-config.md` § 5 documents as the procedure, but it violates this plan's "green after every task"). The plan therefore lands the reference (+ `tests/variants.py`) as Task 1 and the config as Task 2, and moves the `knob_for_key`/`EXTRA_PARAMS` update from spec task 5 into Task 2, where it is what makes the config change green. No content is dropped or added.
2. **`tests/variants.py` also carries the band expectations.** Spec § 4.3 fixes `VARIANTS`; the pre-barrier pair counts and the named added pairs (spec § 3) are needed by BOTH the generator guard (Task 3) and the reference pins (Task 1), so they live in the same import-nothing table as `BAND_RADII_KM`, `EXPECTED_BAND_PAIRS`, `ADDED_BAND_PAIRS`. One source of truth; the property "imports nothing from the repo" is preserved.
3. **Where the band guard lives.** Spec § 3 says "the generator asserts"; it does not say where the shared step lives. Resolved: `check_bands(city)` in `scripts/check_oraculum_invariants.py`, called by `emit_checked_variant_expected_values(city, out_path)` in the same module — that module already owns "only write a valid fixture", and both generators get the assertion by calling the one helper.
4. **Which variants take the CLI leg.** Spec § 4.2 says "the CLI path once". Resolved as two variants, `band_small_boundary` and `exp1`: one round trip cannot exercise both the band + boundary path and the `scale_km` path, and each run is a 7-settlement city (seconds).
5. **Spec § 4.5's three stamp pins are written as unit tests over `methodology_stamp` / `check_methodology_stamp`,** in `tests/test_cli.py` beside the existing stamp tests, not as extra CLI runs — `check_methodology_stamp` reads only `frame.attrs`, and the end-to-end "config → artifact → compute" claim is carried by Task 5's CLI round trip.
6. **The real-data verify uses the established verify dir `~/delhi_data/phase3_verify`** (3A–3C; it holds the warm GPKG cache and is NOT a baseline file — the constraint protects the July 2025 baseline outputs, not sibling output dirs). Task 5 passes it as `--out-dir`/`--verify-dir`. Controller ruling.

---

### Task 1: `tests/variants.py` + the reference implementation (spec § 3, § 4.1 items 1–6)

The independent side first: the variant table both sides read, the reference's new rules, and every hand-derivable pin. No production code and no config change in this task, so `expected_values.csv` cannot move — every new keyword defaults to today's behaviour.

**Files:**
- Create: `tests/variants.py`
- Create: `tests/test_variant_rules.py`
- Modify: `tests/reference_impl.py`

**Interfaces:**
- Consumes: `tests.cities.ORACULUM` / `MESSY` / `CITIES` (existing); `tests.reference_impl.RULESETS`, `apply_barrier`, `_service_amounts`, `_centroid_km`, `POINT_SERVICES` (existing).
- Produces:
  - `tests.variants.VARIANTS: dict[str, dict[str, dict]]` — 8 keys: `band0_none`, `band_small`, `band_large`, `band_small_boundary`, `pow1`, `pow2`, `exp1`, `boundary`; each value has an optional `"adjacency"` block and a required `"decay"` block, in CONFIG vocabulary.
  - `tests.variants.BAND_RADII_KM: tuple[float, float, float]` = `(0.0, 0.25, 0.75)`
  - `tests.variants.EXPECTED_BAND_PAIRS: dict[str, dict[float, int]]`
  - `tests.variants.ADDED_BAND_PAIRS: dict[str, dict[float, set[tuple[str, str]]]]`
  - `tests.reference_impl.adjacency(settlements, rule, max_distance_km=None) -> dict[str, set[str]]` — new rule `"within_distance"`.
  - `tests.reference_impl.compute_city(..., max_distance_km=None, decay_form="inverse_linear", exponent=None, scale_km=None, decay_distance="centroid") -> DataFrame` — five new keyword-only parameters, all defaulting to today's behaviour.
  - `tests.reference_impl.VARIANT_KNOBS: dict[tuple[str, str], str]` — block key → reference keyword.
  - `tests.reference_impl.VARIANT_RULESETS: dict[str, dict]` — `RULESETS["code"]` plus each variant's overrides.
  - `tests.reference_impl.emit_variant_expected_values(out_path, city) -> DataFrame`

- [ ] **Step 1: Create the variant table**

Create `tests/variants.py`:

```python
"""The variant table: one row per Phase-6-shaped configuration.

Imports NOTHING from this repo (like tests/cities.py). BOTH sides read it:
`tests/reference_impl.py` builds VARIANT_RULESETS from it, and the
production-side tests build a MethodologyConfig from it. Every value it
names — `within_distance`, the four decay forms, `centroid`/`boundary` —
has the SAME spelling in the config vocabulary and in the reference's
keyword vocabulary, so there is no translation layer to get wrong; a
variant that one day needs `touch`/`bbox` will add one then.

A variant states each block it overrides IN FULL, because the CLI round trip
replaces `methodology.<block>` wholesale: no key is ever inherited. A
variant that does not override `adjacency` at all keeps the `code` base's
`bbox`.

The band constants live here for the same reason the table does: the fixture
generators' guard (scripts/check_oraculum_invariants.check_bands) and the
reference pins (tests/test_variant_rules.py) must not hold two copies of
them. All distances are kilometres; pairs are sorted 2-tuples of settlement
ids and are UNDIRECTED and PRE-BARRIER — they are counted on the adjacency
function's own output, never on anything downstream of a barrier rule.
"""

VARIANTS = {
    # X = 0 is intersection-inclusive: every polygon that intersects i,
    # corner-only touches and overlaps included. With `none` the weights are
    # all 1, so the pcen is a plain sum of counts.
    "band0_none": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 0.0},
        "decay": {"form": "none", "distance": "centroid",
                  "distance_unit": "km"},
    },
    # B1 and B2 sit in a gap of BOTH cities' polygon-to-polygon distance
    # lists (>= 26 m clearance), so no pair lands on the `<=` boundary.
    "band_small": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 0.25},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    "band_large": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 0.75},
        "decay": {"form": "inverse_linear", "distance": "centroid",
                  "distance_unit": "km"},
    },
    # The only variant where Oraculum can tell boundary from centroid:
    # A-RV and C-RV are 0.100 km apart at the boundary and 1.414214 km apart
    # at the centroids.
    "band_small_boundary": {
        "adjacency": {"rule": "within_distance", "max_distance_km": 0.25},
        "decay": {"form": "inverse_linear", "distance": "boundary",
                  "distance_unit": "km"},
    },
    "pow1": {
        "decay": {"form": "inverse_power", "exponent": 1,
                  "distance": "centroid", "distance_unit": "km"},
    },
    "pow2": {
        "decay": {"form": "inverse_power", "exponent": 2,
                  "distance": "centroid", "distance_unit": "km"},
    },
    "exp1": {
        "decay": {"form": "exponential", "scale_km": 1.0,
                  "distance": "centroid", "distance_unit": "km"},
    },
    # Degenerate on Oraculum (every `code` neighbour there is in contact, so
    # every weight is 1); messy's G-M pair carries the proof.
    "boundary": {
        "decay": {"form": "inverse_linear", "distance": "boundary",
                  "distance_unit": "km"},
    },
}

BAND_RADII_KM = (0.0, 0.25, 0.75)

# Undirected pair counts on adjacency()'s own output, BEFORE any barrier.
EXPECTED_BAND_PAIRS = {
    "oraculum": {0.0: 10, 0.25: 12, 0.75: 14},
    "messy": {0.0: 5, 0.25: 8, 0.75: 10},
}

# Exactly which pairs each radius adds to the one below it.
ADDED_BAND_PAIRS = {
    "oraculum": {
        0.25: {("A", "RV"), ("C", "RV")},          # 0.100 km each
        0.75: {("B", "D"), ("B", "IND")},          # 0.500 km each
    },
    "messy": {
        0.25: {("H", "L"), ("H", "T"), ("L", "S")},   # 0.131519/0.223607/0.199
        0.75: {("G", "M"), ("S", "T")},               # 0.450 / 0.630242
    },
}
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_variant_rules.py`:

```python
"""Hand-derivable pins for the injectable parameters (spec § 4.1, items 1-6).

Every number here was derived from the fixture geometry and checked against
the fixtures; the arithmetic is spelled out in each test so a reader can
re-derive it on paper. `within_distance` is POLYGON-TO-POLYGON distance, so
the radii are judged against boundary distances, never the worksheet's
centroid distances.

Reference side only: nothing in this file imports delhi_psi.
"""
import math

import pytest

from tests.cities import CITIES, MESSY, ORACULUM
from tests.reference_impl import (
    RULESETS, VARIANT_KNOBS, VARIANT_RULESETS, adjacency, apply_barrier,
    compute_city,
)
from tests.variants import (
    ADDED_BAND_PAIRS, BAND_RADII_KM, EXPECTED_BAND_PAIRS, VARIANTS,
)

B0, B1, B2 = BAND_RADII_KM          # 0.0, 0.25, 0.75 km


def undirected(nbrs):
    return {tuple(sorted((i, j))) for i, js in nbrs.items() for j in js}


def band(city, km):
    return adjacency(city.load_settlements(), "within_distance", km)


def scored(city, ruleset, denom="pop"):
    """The city's FIRST scenario under `ruleset` (a VARIANT_RULESETS entry or
    a RULESETS entry), indexed by settlement id."""
    scenarios = {s.name: (s.dropped, s.dropped_before_neighbors)
                 for s in city.scenarios}
    return compute_city(city.load_settlements(), city.load_services(),
                        city.load_barriers(), scenario=city.scenarios[0].name,
                        denom=denom, scenarios=scenarios, **ruleset)


def variant(city, name, denom="pop"):
    return scored(city, VARIANT_RULESETS[name], denom)


# --- item 1: what a 0 km band IS ---------------------------------------
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_band_zero_is_the_intersects_neighbourhood(city):
    assert band(city, B0) == adjacency(city.load_settlements(), "intersects")


def test_on_oraculum_band_zero_is_also_touch_and_undirected_bbox():
    """Ten pairs, three ways: that city has no corner-only contact and no
    overlap, which is exactly what the messy city adds."""
    settlements = ORACULUM.load_settlements()
    zero = undirected(band(ORACULUM, B0))
    assert zero == undirected(adjacency(settlements, "border"))
    assert zero == undirected(adjacency(settlements, "bbox"))
    assert len(zero) == 10


def test_on_messy_band_zero_is_touch_plus_the_corner_only_pair():
    """L and T meet at the single Point (2000, 800): `touch` wants positive
    LENGTH, so it misses them; a 0 km band does not. O1 and O2 OVERLAP, and
    an overlap's intersection is a Polygon whose `.length` is its perimeter
    (2400 m), so `touch` already accepts that one."""
    settlements = MESSY.load_settlements()
    geoms = settlements.set_index("USO_AREA_U").geometry
    assert geoms["L"].intersection(geoms["T"]).geom_type == "Point"
    assert geoms["L"].intersection(geoms["T"]).length == 0.0
    assert geoms["O1"].intersection(geoms["O2"]).geom_type == "Polygon"
    assert geoms["O1"].intersection(geoms["O2"]).length == 2400.0

    zero = undirected(band(MESSY, B0))
    touch = undirected(adjacency(settlements, "border"))
    assert zero - touch == {("L", "T")}
    assert touch - zero == set()
    assert {("L", "T"), ("O1", "O2")} <= undirected(
        adjacency(settlements, "bbox"))


@pytest.mark.parametrize("km", BAND_RADII_KM)
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_the_band_is_symmetric(city, km):
    nbrs = band(city, km)
    for i, js in nbrs.items():
        for j in js:
            assert i in nbrs[j], (i, j, km)


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_pre_barrier_pair_counts_and_the_pairs_each_radius_adds(city):
    """Counted on adjacency()'s OWN output — never downstream of a barrier
    rule, which would fold the canal's severing into the band's numbers."""
    pairs = {km: undirected(band(city, km)) for km in BAND_RADII_KM}
    assert {km: len(p) for km, p in pairs.items()} == \
        EXPECTED_BAND_PAIRS[city.name]
    assert pairs[B1] - pairs[B0] == ADDED_BAND_PAIRS[city.name][B1]
    assert pairs[B2] - pairs[B1] == ADDED_BAND_PAIRS[city.name][B2]


# --- item 2: monotonicity ----------------------------------------------
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_bands_are_nested_and_strictly_growing(city):
    small, large = band(city, B1), band(city, B2)
    for i, js in small.items():
        assert js <= large[i], i
    assert any(large[i] > small[i] for i in small), "no pair added at B2"


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_every_touching_pair_is_in_every_band(city):
    """The large-neighbour property that motivated the boundary definition:
    a shared border is distance 0, so no radius can drop it."""
    touch = adjacency(city.load_settlements(), "border")
    for km in BAND_RADII_KM:
        nbrs = band(city, km)
        for i, js in touch.items():
            assert js <= nbrs[i], (i, km)


# --- item 3: inverse_power 1 == inverse_linear -------------------------
@pytest.mark.parametrize("denom", ["pop", "popdensity"])
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_inverse_power_one_reproduces_inverse_linear(city, denom):
    """`x ** 1.0 == x` exactly in IEEE, so this holds at 0 tolerance; the
    pin uses 1e-12 anyway."""
    base = scored(city, RULESETS["code"], denom)
    got = variant(city, "pow1", denom)
    assert list(got.columns) == list(base.columns)
    for column in base.columns:
        assert list(got[column]) == pytest.approx(list(base[column]),
                                                  abs=1e-12), column


# --- item 4: none ------------------------------------------------------
def test_none_lets_every_neighbour_count_in_full():
    """B's band-0 list is {A, C, E, RV} (identical to its `bbox` list on this
    city) and the canal severs A, leaving {C, E, RV}. Clinics: B owns 1,
    C 0, E 1, RV 2; pop 200. With `none` every weight is 1."""
    got = variant(ORACULUM, "band0_none")
    assert got.loc["B", "clinic_pcen"] == pytest.approx((1 + 0 + 1 + 2) / 200,
                                                        abs=1e-12)


# --- item 5: pow2 and exp1 at a closed form ----------------------------
def test_rv_and_d_are_the_single_neighbour_settlements():
    settlements = ORACULUM.load_settlements()
    nbrs = apply_barrier(adjacency(settlements, "bbox"), settlements,
                         ORACULUM.load_barriers(), "global")
    assert nbrs["RV"] == {"B"} and nbrs["D"] == {"E"}
    assert [i for i, js in nbrs.items() if len(js) == 1] == ["RV", "D"]
    cent = settlements.set_index("USO_AREA_U").geometry.centroid
    assert cent["RV"].distance(cent["B"]) / 1000 == pytest.approx(1.0,
                                                                  abs=1e-12)
    assert cent["D"].distance(cent["E"]) / 1000 == pytest.approx(1.5,
                                                                  abs=1e-12)


def test_pow2_and_exp1_on_the_two_single_neighbour_settlements():
    """RV's only `code` neighbour is B at 1.0 km and D's is E at 1.5 km, so
    each pcen is one weight. Clinics: RV owns 2 and B owns 1; D owns 0 and
    E owns 1. Both populations are 100."""
    pow2, exp1 = variant(ORACULUM, "pow2"), variant(ORACULUM, "exp1")
    assert pow2.loc["RV", "clinic_pcen"] == pytest.approx(
        (2 + 1 * 1 / (1 + 1.0) ** 2) / 100, abs=1e-12)        # 0.0225
    assert pow2.loc["D", "clinic_pcen"] == pytest.approx(
        (0 + 1 * 1 / (1 + 1.5) ** 2) / 100, abs=1e-12)        # 0.0016
    assert exp1.loc["RV", "clinic_pcen"] == pytest.approx(
        (2 + 1 * math.exp(-1.0)) / 100, abs=1e-12)
    assert exp1.loc["D", "clinic_pcen"] == pytest.approx(
        (0 + 1 * math.exp(-1.5)) / 100, abs=1e-12)


# --- item 6: boundary vs centroid --------------------------------------
def test_a_contact_neighbour_is_undecayed_under_boundary():
    """(a) Oraculum: A's `code` neighbours B and E both share a border with
    it, so each lends its whole clinic count (A owns 2, B 1, E 1; pop 100).
    Messy: O1 and O2 OVERLAP, so their boundary distance is 0 too — O1's
    list is {O2, U}, and U (no population row) is dropped by the
    `nopop_only` scenario and swallowed."""
    assert variant(ORACULUM, "boundary").loc["A", "clinic_pcen"] == \
        pytest.approx((2 + 1 + 1) / 100, abs=1e-12)
    assert variant(MESSY, "boundary").loc["O1", "clinic_pcen"] == \
        pytest.approx((1 + 1) / 600, abs=1e-12)


def test_boundary_beats_centroid_for_an_interlocked_neighbour_on_messy():
    """(b) H and L are DISJOINT but interlocked: 0.131519 km apart at the
    boundary, 1.127237 km apart at the centroids. In `band_small` H's list is
    {L, S, T} (schools 1, 1, 1; H owns none; pop 110), so the whole row is
    weights. Written on H's row: the pin is directional."""
    centroid = variant(MESSY, "band_small").loc["H", "school_pcen"]
    boundary = variant(MESSY, "band_small_boundary").loc["H", "school_pcen"]
    assert boundary == pytest.approx(
        (1 / (1 + 0.13151918984428584) + 1 / (1 + 0.0)
         + 1 / (1 + 0.22360679774997896)) / 110, abs=1e-12)
    assert centroid == pytest.approx(0.013170282557128916, abs=1e-12)
    assert boundary > centroid


def test_boundary_beats_centroid_for_a_large_neighbour_on_oraculum():
    """(b) A's `band_small` list is {B, D, E, RV}; the canal severs D. RV sits
    0.100 km away at the boundary and 1.414214 km away at the centroids,
    while B and E are in contact. Clinics A 2, B 1, E 1, RV 2; pop 100.
    A is canal-flagged, so RV's own row drops A and is not pinned here."""
    centroid = variant(ORACULUM, "band_small").loc["A", "clinic_pcen"]
    boundary = variant(ORACULUM, "band_small_boundary").loc["A",
                                                            "clinic_pcen"]
    assert centroid == pytest.approx(
        (2 + 1 * 1 / (1 + 1.0) + 3 * 1 / (1 + math.sqrt(2))) / 100, abs=1e-12)
    assert boundary == pytest.approx(
        (2 + 1 + 1 + 2 * 1 / (1 + 0.1)) / 100, abs=1e-12)
    assert boundary > centroid


def test_centroid_distance_can_understate_a_gap_too():
    """(c) The opposite pathology, on G's row. M is a two-part MultiPolygon
    whose centroid falls in its own gap, exactly on G's centroid: centroid
    distance 0 (weight 1) but boundary distance 0.45 km (weight 1/1.45).
    G owns a school and M owns a school; pop 50. M's list never contains G
    (M's envelope holds G, but the two do not meet), so the pin is on G."""
    code = scored(MESSY, RULESETS["code"])
    boundary = variant(MESSY, "boundary")
    assert code.loc["G", "school_pcen"] == pytest.approx((1 + 1) / 50,
                                                          abs=1e-12)
    assert boundary.loc["G", "school_pcen"] == pytest.approx(
        (1 + 1 / (1 + 0.45)) / 50, abs=1e-12)
    assert boundary.loc["G", "school_pcen"] < code.loc["G", "school_pcen"]


def test_g_m_enters_the_band_at_the_large_radius():
    assert ("G", "M") in ADDED_BAND_PAIRS["messy"][B2]


# --- the table and the reference agree ---------------------------------
def test_variant_rulesets_are_the_code_base_plus_the_table():
    assert set(VARIANT_RULESETS) == set(VARIANTS)
    for name, spec in VARIANTS.items():
        got = VARIANT_RULESETS[name]
        overridden = set()
        for block, mapping in spec.items():
            for key, value in mapping.items():
                if (block, key) not in VARIANT_KNOBS:
                    continue                      # decay.distance_unit
                knob = VARIANT_KNOBS[(block, key)]
                assert got[knob] == value, (name, knob)
                overridden.add(knob)
        for knob, value in RULESETS["code"].items():
            if knob not in overridden:
                assert got[knob] == value, (name, knob)


# --- the reference rejects the same combinations the config will -------
def test_within_distance_requires_a_radius():
    with pytest.raises(ValueError, match="max_distance_km"):
        adjacency(ORACULUM.load_settlements(), "within_distance")


@pytest.mark.parametrize("rule", ["bbox", "border", "intersects"])
def test_a_radius_without_within_distance_is_rejected(rule):
    with pytest.raises(ValueError, match="max_distance_km"):
        adjacency(ORACULUM.load_settlements(), rule, 0.25)


@pytest.mark.parametrize("kwargs,match", [
    (dict(decay_form="sideways"), "sideways"),
    (dict(decay_form="inverse_power"), "exponent"),
    (dict(decay_form="exponential"), "scale_km"),
    (dict(decay_form="inverse_linear", exponent=2), "exponent"),
    (dict(decay_form="none", scale_km=1.0), "scale_km"),
    (dict(decay_distance="as_the_crow_flies"), "as_the_crow_flies"),
])
def test_compute_city_rejects_missing_or_unused_decay_parameters(kwargs,
                                                                 match):
    call = dict(RULESETS["code"], scenario="baseline", denom="pop")
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        compute_city(ORACULUM.load_settlements(), ORACULUM.load_services(),
                     ORACULUM.load_barriers(), **call)


def test_variants_module_imports_nothing_at_all():
    """Both sides read this table, so it must reach neither of them — not
    `delhi_psi` (the reference's INDEPENDENCE RULE) and not
    `tests.reference_impl` (which imports it). It is data, so the check is
    simply that it has no import statement whatsoever."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent / "variants.py").read_text()
    tree = ast.parse(source)
    assert not [node for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))], \
        "tests/variants.py is a data table: it imports nothing"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_variant_rules.py -q -W error`
Expected: collection error — `ImportError: cannot import name 'VARIANT_KNOBS' from 'tests.reference_impl'`.

- [ ] **Step 4: Add the new rules to the reference implementation**

In `tests/reference_impl.py`, extend the module docstring's knob list with
`max_distance_km, decay_form, exponent, scale_km, decay_distance`, add
`import math` at the top of the imports, and make these four edits.

(a) `adjacency` gains the rule and the parameter check:

```python
def adjacency(settlements, rule, max_distance_km=None):
    """Directed neighbour lists under `rule`.

    within_distance: j is a neighbour of i iff the POLYGON-TO-POLYGON
        shortest distance is <= max_distance_km * 1000 metres. At 0 km that
        is `intersects` — corner-only touches and overlaps included — which
        is what the § 4.1 pins compare it against.
    """
    if rule == "within_distance":
        if max_distance_km is None:
            raise ValueError(
                "adjacency rule 'within_distance' requires max_distance_km")
    elif max_distance_km is not None:
        raise ValueError(
            f"max_distance_km is only used by rule 'within_distance', not "
            f"{rule!r}")
    idx = settlements.set_index("USO_AREA_U").geometry
    out = {}
    for i in idx.index:
        nbrs = set()
        for j in idx.index:
            if i == j:
                continue
            if rule == "border":
                inter = idx[i].intersection(idx[j])
                if not inter.is_empty and inter.length > 0:
                    nbrs.add(j)
            elif rule == "bbox":
                if idx[i].intersects(box(*idx[j].bounds)):
                    nbrs.add(j)
            elif rule == "intersects":
                if idx[i].intersects(idx[j]):
                    nbrs.add(j)
            elif rule == "within_distance":
                if idx[i].distance(idx[j]) <= max_distance_km * 1000:
                    nbrs.add(j)
            else:
                raise ValueError(rule)
        out[i] = nbrs
    return out
```

(b) `compute_city` gains the five keyword-only parameters, validates them,
and computes the weight by form and distance definition. Replace its
signature, insert the validation block first thing in the body, add the
geometry lookup beside `cent`, and replace `contribution_weight`:

```python
DECAY_FORMS = ("inverse_linear", "none", "inverse_power", "exponential")
DECAY_DISTANCES = ("centroid", "boundary")


def compute_city(settlements, services, barriers, *, adjacency_rule,
                 barrier_rule, roads_formula, scenario, denom, second_norm,
                 absent_neighbor_contribution, scenarios=None,
                 max_distance_km=None, decay_form="inverse_linear",
                 exponent=None, scale_km=None, decay_distance="centroid"):
    # Every parameter a form does not use is REJECTED, not ignored — the
    # mapped-knob test relies on an unimplemented combination raising.
    if decay_form not in DECAY_FORMS:
        raise ValueError(f"unknown decay form {decay_form!r}; allowed "
                         f"values: {list(DECAY_FORMS)}")
    if decay_distance not in DECAY_DISTANCES:
        raise ValueError(f"unknown decay distance {decay_distance!r}; "
                         f"allowed values: {list(DECAY_DISTANCES)}")
    if decay_form == "inverse_power":
        if exponent is None:
            raise ValueError("decay form 'inverse_power' requires exponent")
    elif exponent is not None:
        raise ValueError(f"exponent is not used by decay form "
                         f"{decay_form!r}; it is used by 'inverse_power'")
    if decay_form == "exponential":
        if scale_km is None:
            raise ValueError("decay form 'exponential' requires scale_km")
    elif scale_km is not None:
        raise ValueError(f"scale_km is not used by decay form "
                         f"{decay_form!r}; it is used by 'exponential'")

    table = SCENARIOS if scenarios is None else scenarios
    dropped, drop_before = table[scenario]
    universe = settlements[~settlements["USO_AREA_U"].isin(dropped)] \
        if drop_before else settlements

    nbrs = apply_barrier(adjacency(universe, adjacency_rule, max_distance_km),
                         universe, barriers, barrier_rule)
    cent = _centroid_km(universe)
    geom = universe.set_index("USO_AREA_U").geometry
    amounts = _service_amounts(universe, services)
```

and, in place of today's two-line `contribution_weight`:

```python
    def contribution_weight(i, j):
        # boundary: polygon-to-polygon, so every touching or overlapping
        # neighbour is at distance 0 and lends its amount undecayed.
        if decay_distance == "boundary":
            d_km = geom[i].distance(geom[j]) / 1000
        else:
            d_km = cent[i].distance(cent[j]) / 1000
        if decay_form == "inverse_linear":
            return 1 / (1 + d_km)
        if decay_form == "none":
            return 1.0
        if decay_form == "inverse_power":
            return 1 / (1 + d_km) ** exponent
        return math.exp(-d_km / scale_km)
```

(b′) **Deterministic summation order — REQUIRED (plan review R1, Critical).**
`nbrs[i]` is a Python `set`; its iteration order depends on the per-process
hash seed, and floating-point addition is not associative. With the band
variants some settlements have 3–4 neighbours, and the `%.17g` CSV then
differs by 1 ULP between processes (measured: three distinct byte outputs
across 15 `PYTHONHASHSEED` values before the fix; identical after). Change
the summation loop in `compute_city` from `for j in nbrs[i]:` to:

```python
            for j in sorted(nbrs[i]):
```

This is the ONLY change to the loop. Verified before planning: with
`sorted()` in place, `python -m tests.reference_impl` leaves BOTH cities'
committed `expected_values.csv` byte-identical under `PYTHONHASHSEED`
0/4/7/11 (`git status --porcelain -- tests/fixtures/` empty each time), so
the Global Constraint holds. Consequently the "Canonical facts" band
values are the SORTED-order values; a value there may differ from an
unsorted run in the last digit — every test in this plan compares band
values with `pytest.approx(..., abs=1e-12)`, never `==`.

(c) the variant table, immediately after `RULESETS`:

```python
# tests/variants.py speaks CONFIG vocabulary; the only difference is the
# KEY names, so this map is a rename and never a translation of values.
# `decay.distance_unit` has no reference knob (the reference is km-only,
# as the manuscript is), so it is deliberately absent.
VARIANT_KNOBS = {
    ("adjacency", "rule"): "adjacency_rule",
    ("adjacency", "max_distance_km"): "max_distance_km",
    ("decay", "form"): "decay_form",
    ("decay", "distance"): "decay_distance",
    ("decay", "exponent"): "exponent",
    ("decay", "scale_km"): "scale_km",
}
IGNORED_VARIANT_KEYS = frozenset({("decay", "distance_unit")})


def _variant_overrides(spec):
    out = {}
    for block, mapping in spec.items():
        for key, value in mapping.items():
            if (block, key) in IGNORED_VARIANT_KEYS:
                continue
            if (block, key) not in VARIANT_KNOBS:
                raise ValueError(
                    f"tests/variants.py: {block}.{key} has no reference "
                    f"knob; add one to VARIANT_KNOBS or to "
                    f"IGNORED_VARIANT_KEYS")
            out[VARIANT_KNOBS[(block, key)]] = value
    return out


# `code` base + the table's overrides: a variant is today's empirical
# rule-set with one or two values changed, so a difference in the output is
# attributable to those values alone.
VARIANT_RULESETS = {name: dict(RULESETS["code"], **_variant_overrides(spec))
                    for name, spec in VARIANTS.items()}
```

with `from tests.variants import VARIANTS` added beside `from tests.cities import ORACULUM`.

(d) the variants emitter, beside `emit_expected_values`:

```python
def emit_variant_expected_values(out_path, city):
    """Score `city` under every VARIANT_RULESETS entry and write the
    long-format CSV.

    ONE scenario — `city.scenarios[0]` (Oraculum `baseline`, messy
    `nopop_only`; the messy city has no scenario literally named `baseline`,
    because `U` is dropped by every one of them). The exclusion machinery is
    proven elsewhere and is orthogonal to these two knobs. Both denominators,
    `%.17g`, same columns as emit_expected_values.
    """
    settlements, barriers, services = (city.load_settlements(),
                                       city.load_barriers(),
                                       city.load_services())
    scenario = city.scenarios[0]
    scenarios = {s.name: (s.dropped, s.dropped_before_neighbors)
                 for s in city.scenarios}
    records = []
    for rule, kwargs in VARIANT_RULESETS.items():
        for denom in ("pop", "popdensity"):
            df = compute_city(settlements, services, barriers,
                              scenario=scenario.name, denom=denom,
                              scenarios=scenarios, **kwargs)
            for sid, row in df.iterrows():
                for metric, value in row.items():
                    records.append((rule, scenario.name, denom, sid, metric,
                                    value))
    out = pd.DataFrame(records, columns=["rule", "scenario", "denom",
                                         "settlement", "metric", "value"])
    out.to_csv(out_path, index=False, float_format="%.17g")
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_variant_rules.py -q -W error`
Expected: PASS (about 40 cases, most of them parametrised over both cities).

- [ ] **Step 6: Prove the reference change is a byte no-op**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
```

Expected: the generators print their `wrote …` lines and `git status` prints
**nothing**. Every new keyword defaults to today's behaviour, so both cities'
`expected_values.csv` and all four `production/*.csv` are unchanged.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q -W error`
Expected: PASS. The count is 386 plus the new cases; the number that matters is that **nothing fails** and nothing was skipped.

- [ ] **Step 8: Commit**

```bash
git add tests/variants.py tests/reference_impl.py tests/test_variant_rules.py
git commit -m "test(oracle): reference rules for the distance band and the decay forms (DEL-18)"
```

---

### Task 2: config — enums, conditional parameters, `decay.distance` (spec § 1, § 4.1 item 7, § 4.4)

The loader learns the new values. Because `REFERENCE_KNOBS` is the single
table the enums AND the mapped-knob test are generated from, this task also
updates that test — which is only green because Task 1 landed the reference
rules it now drives.

**Files:**
- Modify: `delhi_psi/config.py`
- Modify: `delhi_psi/profiles/code-2025.yaml`
- Modify: `delhi_psi/profiles/manuscript.yaml`
- Modify: `tests/test_config.py` (the `MINIMAL` string + the § 4.1 item 7 cases)
- Modify: `tests/test_profiles_match_reference.py` (`knob_for_key`, `EXTRA_PARAMS`)

**Interfaces:**
- Consumes: `tests.variants.VARIANTS` (Task 1); `tests.reference_impl.compute_city`'s new keywords (Task 1).
- Produces:
  - `delhi_psi.config.DecayForm` — StrEnum with members `INVERSE_LINEAR`, `NONE`, `INVERSE_POWER`, `EXPONENTIAL`.
  - `delhi_psi.config.DecayDistance` — StrEnum with members `CENTROID`, `BOUNDARY`.
  - `delhi_psi.config.AdjacencyRule` gains `WITHIN_DISTANCE` (`"within_distance"`).
  - `delhi_psi.config.AdjacencyConfig(rule: AdjacencyRule, max_distance_km: float | None = None)` — frozen.
  - `delhi_psi.config.DecayConfig(form: DecayForm, distance_unit: str, distance: DecayDistance, exponent: float | None = None, scale_km: float | None = None)` — frozen.
  - `delhi_psi.config.ENUM_KEYS` / `ENUMS` / `REFERENCE_KNOBS` gain `"methodology.decay.form"` and `"methodology.decay.distance"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py` (and add `VARIANTS` to its imports:
`from tests.variants import VARIANTS`):

```python
# --- 3D: the distance band and the decay forms (spec § 1, § 4.1 item 7) ---
BAND = "  adjacency: {rule: within_distance, max_distance_km: 0.25}"


def swap(line_start, replacement):
    """MINIMAL with the one line that starts with `line_start` replaced."""
    return "\n".join(replacement if line.startswith(line_start) else line
                     for line in MINIMAL.splitlines()) + "\n"


def test_within_distance_loads_with_its_radius(tmp_path):
    cfg = load_config(write(tmp_path, swap("  adjacency:", BAND)),
                      data_dir=str(tmp_path))
    assert cfg.methodology.adjacency.rule == "within_distance"
    assert cfg.methodology.adjacency.max_distance_km == 0.25


@pytest.mark.parametrize("block,expected", [
    ("  decay: {form: none, distance: centroid, distance_unit: km}",
     (None, None)),
    ("  decay: {form: inverse_power, exponent: 2, distance: centroid, "
     "distance_unit: km}", (2, None)),
    ("  decay: {form: exponential, scale_km: 1.0, distance: boundary, "
     "distance_unit: km}", (None, 1.0)),
])
def test_every_decay_form_loads_with_exactly_its_own_parameter(tmp_path,
                                                               block,
                                                               expected):
    cfg = load_config(write(tmp_path, swap("  decay:", block)),
                      data_dir=str(tmp_path))
    assert (cfg.methodology.decay.exponent,
            cfg.methodology.decay.scale_km) == expected


@pytest.mark.parametrize("key,line_start,bad", [
    # a parameter the rule/form does not use is REJECTED, not ignored
    ("methodology.adjacency.max_distance_km", "  adjacency:",
     "  adjacency: {rule: bbox, max_distance_km: 1.0}"),
    ("methodology.adjacency.max_distance_km", "  adjacency:",
     "  adjacency: {rule: touch, max_distance_km: 1.0}"),
    ("methodology.decay.exponent", "  decay:",
     "  decay: {form: inverse_linear, exponent: 2, distance: centroid, "
     "distance_unit: km}"),
    ("methodology.decay.exponent", "  decay:",
     "  decay: {form: exponential, scale_km: 1.0, exponent: 2, "
     "distance: centroid, distance_unit: km}"),
    ("methodology.decay.scale_km", "  decay:",
     "  decay: {form: none, scale_km: 1.0, distance: centroid, "
     "distance_unit: km}"),
    # required and missing
    ("methodology.adjacency.max_distance_km", "  adjacency:",
     "  adjacency: {rule: within_distance}"),
    ("methodology.decay.exponent", "  decay:",
     "  decay: {form: inverse_power, distance: centroid, distance_unit: km}"),
    ("methodology.decay.scale_km", "  decay:",
     "  decay: {form: exponential, distance: centroid, distance_unit: km}"),
    ("methodology.decay.distance", "  decay:",
     "  decay: {form: inverse_linear, distance_unit: km}"),
    # out of range, and booleans are not numbers
    ("methodology.adjacency.max_distance_km", "  adjacency:",
     "  adjacency: {rule: within_distance, max_distance_km: -1}"),
    ("methodology.adjacency.max_distance_km", "  adjacency:",
     "  adjacency: {rule: within_distance, max_distance_km: true}"),
    ("methodology.decay.exponent", "  decay:",
     "  decay: {form: inverse_power, exponent: 0, distance: centroid, "
     "distance_unit: km}"),
    ("methodology.decay.scale_km", "  decay:",
     "  decay: {form: exponential, scale_km: 0, distance: centroid, "
     "distance_unit: km}"),
    ("methodology.decay.scale_km", "  decay:",
     "  decay: {form: exponential, scale_km: true, distance: centroid, "
     "distance_unit: km}"),
])
def test_conditional_parameters_are_rejected_naming_the_key(tmp_path, key,
                                                            line_start, bad):
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, swap(line_start, bad)))
    assert key in str(exc.value)


@pytest.mark.parametrize("key,line_start,bad", [
    ("methodology.decay.form", "  decay:",
     "  decay: {form: sideways, distance: centroid, distance_unit: km}"),
    ("methodology.decay.distance", "  decay:",
     "  decay: {form: inverse_linear, distance: as_the_crow_flies, "
     "distance_unit: km}"),
])
def test_new_enums_name_the_key_and_the_allowed_values(tmp_path, key,
                                                       line_start, bad):
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, swap(line_start, bad)))
    message = str(exc.value)
    assert key in message
    for allowed in REFERENCE_KNOBS[key]:
        assert str(allowed) in message


@pytest.mark.parametrize("profile", ["code-2025", "manuscript"])
def test_shipped_profiles_name_the_centroid_distance_explicitly(profile,
                                                                tmp_path):
    """The one key both profiles gain. It names the definition they have
    always used — the value is a record, not a change."""
    cfg = load_config(profile, data_dir=str(tmp_path))
    assert cfg.methodology.decay.distance == "centroid"
    assert cfg.methodology.decay.form == "inverse_linear"
    assert cfg.methodology.decay.exponent is None
    assert cfg.methodology.decay.scale_km is None
    assert cfg.methodology.adjacency.max_distance_km is None


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_block_is_one_the_loader_accepts(tmp_path, variant):
    """tests/variants.py is written in CONFIG vocabulary, so every block in
    it must load — and every enum value it names must be a member of the
    matching REFERENCE_KNOBS entry. Without this the reference could be
    pinned against values the loader would refuse."""
    import yaml

    enum_key = {("adjacency", "rule"): "methodology.adjacency.rule",
                ("decay", "form"): "methodology.decay.form",
                ("decay", "distance"): "methodology.decay.distance"}
    for block, values in VARIANTS[variant].items():
        for key, value in values.items():
            if (block, key) in enum_key:
                assert value in REFERENCE_KNOBS[enum_key[(block, key)]], \
                    (variant, block, key)

    raw = yaml.safe_load(MINIMAL)
    for block, values in VARIANTS[variant].items():
        raw["methodology"][block] = dict(values)
    path = write(tmp_path, yaml.safe_dump(raw, sort_keys=False))
    cfg = load_config(path, data_dir=str(tmp_path))
    for block, values in VARIANTS[variant].items():
        loaded = getattr(cfg.methodology, block)
        for key, value in values.items():
            assert getattr(loaded, key) == value, (variant, block, key)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q -W error`
Expected: FAIL — the first new test errors with
`ConfigError: unknown key 'methodology.adjacency.max_distance_km'`, and
`test_shipped_profiles_name_the_centroid_distance_explicitly` with
`AttributeError: 'DecayConfig' object has no attribute 'distance'`.

- [ ] **Step 3: Extend the enum table and the dataclasses**

In `delhi_psi/config.py`:

```python
REFERENCE_KNOBS = {
    "methodology.adjacency.rule": {"bbox": "bbox", "touch": "border",
                                   "within_distance": "within_distance"},
    "methodology.barrier.rule": {"global_asymmetric": "global",
                                 "pairwise": "pair"},
    "methodology.decay.form": {"inverse_linear": "inverse_linear",
                               "none": "none",
                               "inverse_power": "inverse_power",
                               "exponential": "exponential"},
    "methodology.decay.distance": {"centroid": "centroid",
                                   "boundary": "boundary"},
    "methodology.roads": {"decayed": "decayed", "eq4_own_only": "eq4"},
    "methodology.second_normalization": {True: True, False: False},
    "methodology.exclusion.stage": {"post_neighbors": False,
                                    "pre_neighbors": True},
    "methodology.exclusion.absent_neighbor": {"swallowed": "swallowed",
                                              "contributes": "contributes"},
    "outputs.denominators[]": {"pop": "pop", "popdensity": "popdensity"},
}

ENUM_KEYS = (
    "methodology.adjacency.rule",
    "methodology.barrier.rule",
    "methodology.decay.form",
    "methodology.decay.distance",
    "methodology.roads",
    "methodology.exclusion.stage",
    "methodology.exclusion.absent_neighbor",
    "outputs.denominators[]",
)
```

then, beside the other generated enums:

```python
DecayForm = _make_enum("DecayForm", "methodology.decay.form")
DecayDistance = _make_enum("DecayDistance", "methodology.decay.distance")
```

and two entries in `ENUMS`:

```python
    "methodology.decay.form": DecayForm,
    "methodology.decay.distance": DecayDistance,
```

Replace the two dataclasses:

```python
@dataclass(frozen=True)
class AdjacencyConfig:
    rule: AdjacencyRule
    # None is "not applicable", never a default for the YAML key: the key is
    # required by `within_distance` and rejected for every other rule.
    max_distance_km: float | None = None


@dataclass(frozen=True)
class DecayConfig:
    form: DecayForm
    distance_unit: str
    distance: DecayDistance
    exponent: float | None = None
    scale_km: float | None = None
```

- [ ] **Step 4: Validate the conditional parameters at load**

Add the helper beside `_bool` in `delhi_psi/config.py`:

```python
def _conditional_number(mapping, key, prefix, *, used_by, applies, minimum,
                        strict):
    """A parameter exactly one rule/form uses: required by that one, refused
    by every other, and never silently ignored.

    `used_by` is the dotted key and value that DO use it, so the message
    tells the reader which line to change.
    """
    dotted = f"{prefix}.{key}"
    if not applies:
        if key in mapping:
            raise ConfigError(
                f"{dotted}: not allowed here — it is only used by {used_by}")
        return None
    value = _require(mapping, key, prefix)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{dotted}: {value!r} is not allowed; expected a "
                          "number")
    if strict and not value > minimum:
        raise ConfigError(f"{dotted}: {value!r} is not allowed; expected a "
                          f"number > {minimum}")
    if not strict and value < minimum:
        raise ConfigError(f"{dotted}: {value!r} is not allowed; expected a "
                          f"number >= {minimum}")
    return float(value)
```

Replace the adjacency and decay blocks of `_methodology`:

```python
    adjacency_raw = _require(raw, "adjacency", "methodology")
    _reject_unknown(adjacency_raw, {"rule", "max_distance_km"},
                    "methodology.adjacency")
    adjacency_rule = _coerce_enum(
        "methodology.adjacency.rule",
        _require(adjacency_raw, "rule", "methodology.adjacency"))
    adjacency = AdjacencyConfig(
        rule=adjacency_rule,
        max_distance_km=_conditional_number(
            adjacency_raw, "max_distance_km", "methodology.adjacency",
            used_by="methodology.adjacency.rule: within_distance",
            applies=adjacency_rule == AdjacencyRule.WITHIN_DISTANCE,
            minimum=0, strict=False))
```

```python
    decay_raw = _require(raw, "decay", "methodology")
    _reject_unknown(decay_raw, {"form", "distance_unit", "distance",
                                "exponent", "scale_km"}, "methodology.decay")
    form = _coerce_enum("methodology.decay.form",
                        _require(decay_raw, "form", "methodology.decay"))
    unit = _require(decay_raw, "distance_unit", "methodology.decay")
    if unit != "km":
        raise ConfigError(
            f"methodology.decay.distance_unit: {unit!r} is not allowed; "
            "allowed values: ['km']")
    decay = DecayConfig(
        form=form,
        distance_unit=unit,
        # required like every methodology key: no default, never inherited
        distance=_coerce_enum(
            "methodology.decay.distance",
            _require(decay_raw, "distance", "methodology.decay")),
        exponent=_conditional_number(
            decay_raw, "exponent", "methodology.decay",
            used_by="methodology.decay.form: inverse_power",
            applies=form == DecayForm.INVERSE_POWER,
            minimum=0, strict=True),
        scale_km=_conditional_number(
            decay_raw, "scale_km", "methodology.decay",
            used_by="methodology.decay.form: exponential",
            applies=form == DecayForm.EXPONENTIAL,
            minimum=0, strict=True))
```

The ad-hoc `if form != "inverse_linear": raise ...` block is deleted —
`_coerce_enum` replaces it, so the message now lists all four forms.

- [ ] **Step 5: Add the key and the comments to both shipped profiles**

In `delhi_psi/profiles/code-2025.yaml`, replace the `adjacency` and `decay`
lines of the `methodology:` block with:

```yaml
  adjacency:
    rule: bbox                      # bbox | touch | within_distance
                                    # max_distance_km: required iff rule is
                                    # within_distance (>= 0 km, polygon-to-polygon)
```

```yaml
  decay:
    form: inverse_linear            # inverse_linear | none | inverse_power | exponential
                                    # exponent: required iff form is inverse_power (> 0)
                                    # scale_km:  required iff form is exponential (> 0)
    distance: centroid              # centroid | boundary — `centroid` names the
                                    # centroid-to-centroid distance this profile
                                    # has always used; nothing changes
    distance_unit: km               # km
```

In `delhi_psi/profiles/manuscript.yaml`, the same two blocks:

```yaml
  adjacency:
    rule: touch                     # bbox | touch | within_distance
                                    # max_distance_km: required iff rule is
                                    # within_distance (>= 0 km, polygon-to-polygon)
```

```yaml
  decay:
    form: inverse_linear            # inverse_linear | none | inverse_power | exponential
                                    # exponent: required iff form is inverse_power (> 0)
                                    # scale_km:  required iff form is exponential (> 0)
    distance: centroid              # centroid | boundary — the manuscript's d_ij,
                                    # centroid-to-centroid as always
    distance_unit: km               # manuscript is silent on the unit (spec 8)
```

- [ ] **Step 6: Add the key to the `MINIMAL` profile string**

In `tests/test_config.py`, the third and last place the key appears:

```python
MINIMAL = """profile: minimal
""" + CATEGORIES_BLOCK + """methodology:
  adjacency: {rule: bbox}
  barrier: {rule: global_asymmetric, combine: any}
  decay: {form: inverse_linear, distance: centroid, distance_unit: km}
  roads: decayed
  second_normalization: true
  exclusion: {types: [RV], stage: post_neighbors, absent_neighbor: swallowed}
"""
```

`test_defaults_equal_code_2025` compares `minimal.methodology ==
full.methodology`, so `centroid` here is what keeps that test honest.

- [ ] **Step 7: Teach the mapped-knob test the new values**

In `tests/test_profiles_match_reference.py`, add the table above
`test_every_mapped_knob_is_one_the_reference_actually_implements` and the
two `knob_for_key` entries plus the merge line inside it:

```python
# A conditional parameter the reference REQUIRES for one value of a knob.
# The same constants the § 4.1 pins use — never fresh numbers.
EXTRA_PARAMS = {
    ("methodology.adjacency.rule", "within_distance"):
        {"max_distance_km": 0.25},
    ("methodology.decay.form", "inverse_power"): {"exponent": 2},
    ("methodology.decay.form", "exponential"): {"scale_km": 1.0},
}
```

```python
    knob_for_key = {
        "methodology.adjacency.rule": "adjacency_rule",
        "methodology.barrier.rule": "barrier_rule",
        "methodology.decay.form": "decay_form",
        "methodology.decay.distance": "decay_distance",
        "methodology.roads": "roads_formula",
        "methodology.second_normalization": "second_norm",
        "methodology.exclusion.absent_neighbor": "absent_neighbor_contribution",
        "outputs.denominators[]": "denom",
    }
    for key, knob in knob_for_key.items():
        for config_value, reference_value in REFERENCE_KNOBS[key].items():
            kwargs = dict(base)
            kwargs[knob] = reference_value
            kwargs.update(EXTRA_PARAMS.get((key, config_value), {}))
            frame = compute_city(city, services, barriers, **kwargs)
            assert len(frame) == 7, (key, config_value)
```

`decay.distance: boundary` needs no extra keyword, which is why it has no
`EXTRA_PARAMS` row.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_profiles_match_reference.py -q -W error`
Expected: PASS. `test_enums_cover_exactly_the_reference_table` and
`test_enums_are_generated_from_the_reference_table` now cover eight keys.

- [ ] **Step 9: Prove no number moved**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
uv run pytest -q -W error
```

Expected: `git status` prints **nothing** (the profiles gained a key that
selects the behaviour they already had), and the suite is green — 386 plus the cases added in Tasks 1 and 2.

- [ ] **Step 10: Commit**

```bash
git add delhi_psi/config.py delhi_psi/profiles/code-2025.yaml \
        delhi_psi/profiles/manuscript.yaml tests/test_config.py \
        tests/test_profiles_match_reference.py
git commit -m "feat(config): distance band and decay forms as config values (DEL-18)"
```

---

### Task 3: the generators emit the variants CSVs (spec § 3, § 4.4)

Two new committed fixtures, written by the two geometry generators, guarded
before a byte is written — by the CSV-wide invariants guard (as
`expected_values.csv` already is) and by a new band guard that re-derives the
pre-barrier pair counts from the geometry.

**Files:**
- Modify: `scripts/check_oraculum_invariants.py`
- Modify: `scripts/generate_oraculum_fixtures.py`
- Modify: `scripts/generate_messy_fixtures.py`
- Modify: `tests/test_reference_impl.py`
- Create: `tests/fixtures/oraculum/variants_expected_values.csv` (generated)
- Create: `tests/fixtures/messy/variants_expected_values.csv` (generated)

**Interfaces:**
- Consumes: `tests.reference_impl.emit_variant_expected_values(out_path, city)` and `adjacency(settlements, rule, max_distance_km=None)` (Task 1); `tests.variants.BAND_RADII_KM` / `EXPECTED_BAND_PAIRS` / `ADDED_BAND_PAIRS` (Task 1); `scripts.check_oraculum_invariants.check(df, *, city)` (existing).
- Produces:
  - `scripts.check_oraculum_invariants.variant_expected_values_path(city=ORACULUM) -> Path`
  - `scripts.check_oraculum_invariants.check_bands(city, *, expected=None, added=None) -> list[str]`
  - `scripts.check_oraculum_invariants.emit_checked_variant_expected_values(city, out_path) -> Path`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reference_impl.py`. Add `emit_variant_expected_values`
to the existing `from tests.reference_impl import (...)` block and
`from tests.variants import BAND_RADII_KM, VARIANTS` beside the other
top-level imports; `check` and `check_bands` are imported INSIDE each test,
matching how `test_invariants_guard_csv_wide` already imports `check` (the
module reaches into `scripts/`, so the file keeps that import local):

```python
# --- 3D: the variants fixture (spec § 3, § 4.4) ------------------------
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_variants_expected_values_csv_is_regenerable(city, tmp_path):
    """Same contract as expected_values.csv: the committed file must be
    exactly what the reference produces, or a red build could be 'fixed' by
    editing the fixture."""
    regen = tmp_path / "regen.csv"
    emit_variant_expected_values(regen, city)
    assert regen.read_bytes() == (
        city.fixtures / "variants_expected_values.csv").read_bytes()


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_variants_csv_passes_the_csv_wide_invariants_guard(city):
    """`check` groups by (rule, scenario, denom, metric), so it is CSV-shape
    agnostic: the variants file gets the same degenerate-min-max and
    tied-anchor guarantees as expected_values.csv."""
    from scripts.check_oraculum_invariants import check

    frame = pd.read_csv(city.fixtures / "variants_expected_values.csv")
    assert check(frame, city=city) == []


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_variants_csv_has_one_scenario_and_every_variant(city):
    path = city.fixtures / "variants_expected_values.csv"
    frame = pd.read_csv(path)
    assert set(frame["rule"]) == set(VARIANTS)
    assert set(frame["scenario"]) == {city.scenarios[0].name}
    assert set(frame["denom"]) == {"pop", "popdensity"}
    assert list(frame.columns) == ["rule", "scenario", "denom", "settlement",
                                   "metric", "value"]
    assert b"\r" not in path.read_bytes(), "fixtures are LF-only"


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_band_guard_passes_for_both_cities(city):
    from scripts.check_oraculum_invariants import check_bands

    assert check_bands(city) == []


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_band_guard_reports_a_wrong_count(city):
    """The guard must be able to FAIL: move a vertex so a band gains or
    loses a pair and the generator has to refuse to write."""
    from scripts.check_oraculum_invariants import check_bands

    violations = check_bands(city, expected={km: 0 for km in BAND_RADII_KM})
    assert len(violations) == len(BAND_RADII_KM)
    assert all("pair count" in violation for violation in violations)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_reference_impl.py -q -W error`
Expected: collection error —
`ImportError: cannot import name 'check_bands' from 'scripts.check_oraculum_invariants'`.

- [ ] **Step 3: Add the band guard and the checked emitter**

In `scripts/check_oraculum_invariants.py`, add
`from tests.variants import ADDED_BAND_PAIRS, BAND_RADII_KM, EXPECTED_BAND_PAIRS`
to the imports and append these three functions (keeping the
`reference_impl` import lazy, for the same import-cycle reason the module
already documents):

```python
def variant_expected_values_path(city=ORACULUM):
    return city.fixtures / "variants_expected_values.csv"


def check_bands(city, *, expected=None, added=None):
    """Re-derive the band neighbourhoods from the geometry (spec § 3).

    Against `adjacency(...)` DIRECTLY — never anything downstream of a
    barrier rule, which would fold the canal's severing into the band's
    numbers. Move a vertex so a radius gains or loses a pair, or so two of
    the three bands coincide, and this returns violations instead of quietly
    emitting a fixture that pins nothing.
    """
    from tests.reference_impl import adjacency

    expected = EXPECTED_BAND_PAIRS[city.name] if expected is None else expected
    added = ADDED_BAND_PAIRS[city.name] if added is None else added
    settlements = city.load_settlements()
    pairs = {}
    for km in BAND_RADII_KM:
        nbrs = adjacency(settlements, "within_distance", km)
        pairs[km] = {tuple(sorted((i, j)))
                     for i, js in nbrs.items() for j in js}

    violations = []
    for km in BAND_RADII_KM:
        if len(pairs[km]) != expected[km]:
            violations.append(
                f"band {km} km: pair count {len(pairs[km])}, expected "
                f"{expected[km]}")
    zero, small, large = BAND_RADII_KM
    for lower, upper in ((zero, small), (small, large)):
        if not pairs[lower] < pairs[upper]:
            violations.append(
                f"band {lower} km is not a STRICT subset of band {upper} km "
                "— the three neighbourhoods must be pairwise distinct")
        got = pairs[upper] - pairs[lower]
        if upper in added and got != added[upper]:
            violations.append(
                f"band {upper} km adds {sorted(got)}, expected "
                f"{sorted(added[upper])}")
    return violations


def emit_checked_variant_expected_values(city, out_path):
    """Emit `city`'s variants_expected_values.csv, but ONLY if the band
    neighbourhoods are the ones the spec fixed AND the emitted numbers pass
    `check`.

    Same staging discipline as emit_checked_expected_values: the temporary
    file lives OUTSIDE the repo, so a failed run can never leave an untracked
    file under tests/fixtures/ for the CI drift guard to trip over.
    """
    from tests.reference_impl import emit_variant_expected_values

    out_path = Path(out_path)
    band_violations = check_bands(city)
    if band_violations:
        for violation in band_violations:
            print(f"VIOLATION [{city.name}]:", violation)
        raise SystemExit(
            f"{len(band_violations)} band violation(s) for city "
            f"{city.name!r}; refusing to write {out_path}")
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "variants_expected_values.csv"
        emit_variant_expected_values(staged, city)
        violations = check(pd.read_csv(staged))
        if violations:
            for violation in violations:
                print(f"VIOLATION [{city.name}]:", violation)
            raise SystemExit(
                f"{len(violations)} invariant violation(s) for city "
                f"{city.name!r}; refusing to write {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(out_path))
    return out_path
```

and extend the module's `__main__` so the standalone run covers the bands
and the variants file too:

```python
if __name__ == "__main__":
    problems = []
    for target_city in CITIES:
        problems.extend(f"{target_city.name}: {problem}"
                        for problem in check(city=target_city))
        problems.extend(f"{target_city.name}: {problem}"
                        for problem in check_bands(target_city))
        problems.extend(
            f"{target_city.name} (variants): {problem}"
            for problem in check(
                pd.read_csv(variant_expected_values_path(target_city)),
                city=target_city))
    for p in problems:
        print("VIOLATION:", p)
    print("OK" if not problems else f"{len(problems)} violation(s)")
    sys.exit(1 if problems else 0)
```

- [ ] **Step 4: Call it from both geometry generators**

In `scripts/generate_oraculum_fixtures.py`, change the import to

```python
from scripts.check_oraculum_invariants import (
    emit_checked_expected_values, emit_checked_variant_expected_values,
)
```

and append one statement at the end of `main()`, after the existing
`emit_checked_expected_values` line:

```python
    # The variants fixture (spec 3D § 3): same city, ONE scenario, eight
    # derived rule-sets. Guarded by the band check AND the CSV-wide
    # invariants guard before anything is written.
    print(f"wrote {emit_checked_variant_expected_values(ORACULUM, OUT / 'variants_expected_values.csv')}")
```

In `scripts/generate_messy_fixtures.py`, the same import change and, at the
end of `main()` after the existing `path = emit_checked_expected_values(...)`
and its `print`:

```python
    variants = emit_checked_variant_expected_values(
        MESSY, OUT / "variants_expected_values.csv")
    print(f"wrote {variants}")
```

- [ ] **Step 5: Generate the fixtures**

Run:

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
git status --porcelain -- tests/fixtures/
```

Expected: two `wrote …/variants_expected_values.csv` lines, and `git status`
showing **exactly two untracked files** and nothing else:

```
?? tests/fixtures/messy/variants_expected_values.csv
?? tests/fixtures/oraculum/variants_expected_values.csv
```

Then check their size:

```bash
wc -l tests/fixtures/oraculum/variants_expected_values.csv \
      tests/fixtures/messy/variants_expected_values.csv
```

Expected: **2577** and **3681** lines (2576 and 3680 data rows plus the
header). If either differs, **stop and report** — the scenario or the variant
list has moved.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_reference_impl.py -q -W error`
Expected: PASS.

- [ ] **Step 7: Run the drift guard exactly as CI does, then the suite**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
uv run pytest -q -W error
```

Expected: the same two untracked files and nothing modified (regenerating is
idempotent), and the suite green.

- [ ] **Step 8: Commit**

```bash
git add tests/fixtures/oraculum/variants_expected_values.csv
git add tests/fixtures/messy/variants_expected_values.csv
git add scripts/check_oraculum_invariants.py
git add scripts/generate_oraculum_fixtures.py
git add scripts/generate_messy_fixtures.py
git add tests/test_reference_impl.py
git commit -m "test(fixtures): variant expected values for both cities (DEL-18)"
```

After this commit `git status --porcelain -- tests/fixtures/` must be
**empty** — the state every later task's byte-identity step requires.

---

### Task 4: production — `neighbors`, `index`, `pipeline` (spec § 2, § 4.2, § 4.5)

Production learns the same two rules and is required to reproduce the
reference on all eight variants. The stored artifact's shape does not change:
`nbrs_dist_bbox` stays centroid distances under every configuration, and the
boundary distances live in a compute-local column dropped before
`index_frames` returns.

**Files:**
- Modify: `delhi_psi/neighbors.py`
- Modify: `delhi_psi/index.py`
- Modify: `delhi_psi/pipeline.py`
- Modify: `tests/oraculum_fixtures.py`
- Modify: `tests/test_neighbors.py`
- Modify: `tests/test_index.py`
- Modify: `tests/test_cli.py`
- Create: `tests/test_variants_match_reference.py`

**Interfaces:**
- Consumes: `tests.variants.VARIANTS` (Task 1); `delhi_psi.config.AdjacencyConfig` / `AdjacencyRule` / `DecayConfig` / `DecayForm` / `DecayDistance` (Task 2); `tests/fixtures/<city>/variants_expected_values.csv` (Task 3); `tests.oraculum_fixtures.methodology_with` / `oracle_config` and `tests.test_profiles_match_reference.METRIC_MAP` (existing).
- Produces:
  - `delhi_psi.neighbors.adjacency(polygon_gdf, *, id_col="USO_AREA_U", neighbor_col="nbrs_bbox", rule="bbox", max_distance_km=None) -> GeoDataFrame`
  - `delhi_psi.neighbors.boundary_distances(polygon_gdf, *, neighbor_col="nbrs_bbox", nbr_dist_col="nbrs_dist_boundary", id_col="USO_AREA_U") -> GeoDataFrame` — `[(neighbor_id, km), ...]` per row, the same OUTPUT shape as `centroid_distances`.
  - `delhi_psi.index._decay(distance_km, decay_form, distance_unit, *, exponent=None, scale_km=None) -> float`
  - `delhi_psi.index.pcen(..., exponent=None, scale_km=None)` and `delhi_psi.index.service_index(..., exponent=None, scale_km=None)`
  - `delhi_psi.pipeline.NBRS_DIST_BOUNDARY_COL = "nbrs_dist_boundary"`
  - `delhi_psi.pipeline.methodology_stamp(methodology)["adjacency"]` gains `"max_distance_km"` (None for `bbox`/`touch`).
  - `tests.oraculum_fixtures.variant_methodology(base, variant, *, city=ORACULUM, types=None, stage=None) -> MethodologyConfig`

- [ ] **Step 1: Write the failing neighbour and decay tests**

Append to `tests/test_neighbors.py`:

```python
# --- 3D: the distance band and boundary distances (spec § 2.1) ---------
# Verified against the fixture geometry. Polygon-to-polygon, so `A` reaches
# `RV` (0.100 km) at 0.25 km, and `B` reaches `D` and `IND` (0.500 km) at
# 0.75 km.
BAND_DIRECTED = {
    0.0: {"A": {"B", "D", "E"}, "B": {"A", "C", "E", "RV"},
          "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"A", "E"},
          "E": {"A", "B", "C", "D", "IND"}, "IND": {"C", "E"}},
    0.25: {"A": {"B", "D", "E", "RV"}, "B": {"A", "C", "E", "RV"},
           "C": {"B", "E", "IND", "RV"}, "RV": {"A", "B", "C"},
           "D": {"A", "E"}, "E": {"A", "B", "C", "D", "IND"},
           "IND": {"C", "E"}},
    0.75: {"A": {"B", "D", "E", "RV"},
           "B": {"A", "C", "D", "E", "IND", "RV"},
           "C": {"B", "E", "IND", "RV"}, "RV": {"A", "B", "C"},
           "D": {"A", "B", "E"}, "E": {"A", "B", "C", "D", "IND"},
           "IND": {"B", "C", "E"}},
}


@pytest.mark.parametrize("km", [0.0, 0.25, 0.75])
def test_within_distance_lists_match_the_hand_table(km):
    got = lists_of(neighbors.adjacency(prepared(), rule="within_distance",
                                       max_distance_km=km))
    assert got == BAND_DIRECTED[km]


@pytest.mark.parametrize("km", [0.0, 0.25, 0.75, 1.0])
def test_the_dwithin_join_selects_what_brute_force_selects(km):
    """The sjoin is an optimisation, not a definition: it must agree with
    `geom_i.distance(geom_j) <= X` pair for pair (spec § 7)."""
    from tests.reference_impl import adjacency as reference_adjacency

    got = lists_of(neighbors.adjacency(prepared(), rule="within_distance",
                                       max_distance_km=km))
    assert got == reference_adjacency(load_settlements(), "within_distance",
                                      km)


def test_no_neighbour_list_picks_up_a_missing_join_partner():
    """A left join with no match yields NaN. Every polygon is within 0 m of
    itself, so that cannot happen here — pinned, because a NaN id would be
    silently swallowed by pcen's lookup miss instead of failing."""
    frame = neighbors.adjacency(prepared(), rule="within_distance",
                                max_distance_km=0.0)
    for ids in frame["nbrs_bbox"]:
        assert all(isinstance(i, str) for i in ids), ids


def test_within_distance_requires_a_radius():
    with pytest.raises(ValueError, match="max_distance_km"):
        neighbors.adjacency(prepared(), rule="within_distance")


@pytest.mark.parametrize("rule", ["bbox", "touch"])
def test_a_radius_with_another_rule_is_a_value_error(rule):
    """Mirrors the config rule: `build_neighbors` forwards the configured
    value unconditionally, and it is None for every non-band rule."""
    with pytest.raises(ValueError, match="max_distance_km"):
        neighbors.adjacency(prepared(), rule=rule, max_distance_km=1.0)


def test_boundary_distances_have_the_centroid_shape_and_the_gap_values():
    """Same [(id, km), ...] shape as centroid_distances, different numbers:
    A's band-0.25 neighbours B, D and E all touch it (0 km), while RV is
    0.100 km away — where the CENTROID distance is 1.414214 km."""
    frame = neighbors.adjacency(prepared(), rule="within_distance",
                                max_distance_km=0.25)
    boundary = neighbors.boundary_distances(frame)
    centroid = neighbors.centroid_distances(frame)
    row = boundary[boundary["USO_AREA_U"] == "A"].iloc[0]
    assert dict(row["nbrs_dist_boundary"]) == pytest.approx(
        {"B": 0.0, "D": 0.0, "E": 0.0, "RV": 0.1}, abs=1e-12)
    assert [i for i, _ in row["nbrs_dist_boundary"]] == list(row["nbrs_bbox"])
    centroid_row = centroid[centroid["USO_AREA_U"] == "A"].iloc[0]
    assert dict(centroid_row["nbrs_dist_bbox"])["RV"] == pytest.approx(
        1.4142135623730951, abs=1e-12)
```

Append to `tests/test_index.py` (adding `import math` if it is not already
imported):

```python
# --- 3D: the four decay forms (spec § 2.2) -----------------------------
@pytest.mark.parametrize("form,kwargs,expected", [
    ("inverse_linear", {}, 1 / 1.5),
    ("none", {}, 1.0),
    ("inverse_power", {"exponent": 1}, 1 / 1.5),
    ("inverse_power", {"exponent": 2}, 1 / 1.5 ** 2),
    ("exponential", {"scale_km": 1.0}, math.exp(-0.5)),
    ("exponential", {"scale_km": 2.0}, math.exp(-0.25)),
])
def test_decay_forms_at_half_a_kilometre(form, kwargs, expected):
    assert index._decay(0.5, form, "km", **kwargs) == pytest.approx(
        expected, abs=1e-15)


@pytest.mark.parametrize("form,kwargs", [
    ("inverse_linear", {}), ("none", {}),
    ("inverse_power", {"exponent": 2}), ("exponential", {"scale_km": 1.0}),
])
def test_every_form_gives_weight_one_at_zero_distance(form, kwargs):
    """Why `decay.distance: boundary` leaves every touching or overlapping
    neighbour undecayed, under all four forms."""
    assert index._decay(0.0, form, "km", **kwargs) == 1.0


@pytest.mark.parametrize("args,kwargs,match", [
    (("sideways", "km"), {}, "sideways"),
    (("inverse_power", "km"), {}, "exponent"),
    (("exponential", "km"), {}, "scale_km"),
    (("inverse_linear", "km"), {"exponent": 2}, "exponent"),
    (("none", "km"), {"scale_km": 1.0}, "scale_km"),
    (("inverse_linear", "m"), {}, "'m'"),
])
def test_decay_rejects_unknown_forms_and_misplaced_parameters(args, kwargs,
                                                              match):
    with pytest.raises(ValueError, match=match):
        index._decay(0.5, *args, **kwargs)
```

Also add two pcen-level tests that the parameters reach the weight. The
helper city is the file's existing `city_with_neighbours()`: `X` owns 2
clinics with population 100, `Y` owns 0 with population 200, and each is the
other's only neighbour at 1.0 km — so `Y`'s pcen is one weight times 2, over
200 (today, with `inverse_linear`, that is `(0 + 2 * 0.5) / 200`, which
`test_pcen_pop_denominator_matches_eq3_by_hand` already pins):

```python
def test_pcen_uses_the_form_and_its_parameter():
    """Same two-settlement city as test_pcen_pop_denominator_matches_eq3_by_hand
    (X owns 2 clinics, Y owns 0, each is the other's only neighbour at
    1.0 km, Y's population is 200) with the weight changed: under
    `inverse_power` 2 the neighbour lends 2 * 1/(1+1)**2 instead of 2 * 1/2.
    """
    got = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     decay_form="inverse_power", exponent=2)
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((0 + 2 * 1 / (1 + 1.0) ** 2) / 200,
                                        abs=1e-12)   # 0.0025


def test_service_index_forwards_the_decay_parameters():
    """`service_index` is what `index_frames` actually calls, so the
    parameters have to survive that hop too: `exponential` with scale_km 1
    gives Y (0 + 2 * e^-1) / 200."""
    import math

    got = index.service_index(city_with_neighbours(), "clinic_count",
                              service="clinic", denominator="pop",
                              decay_form="exponential", scale_km=1.0)
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((0 + 2 * math.exp(-1.0)) / 200,
                                        abs=1e-12)
```

Finally REPLACE the `decay_form="exponential"` row of the existing
`test_pcen_rejects_unknown_values` — after this task `exponential` is an
implemented form, so that row no longer tests what its name says:

```python
@pytest.mark.parametrize("kwargs,match", [
    (dict(denominator="households"), "households"),
    (dict(absent_neighbor="maybe"), "maybe"),
    (dict(decay_form="sideways"), "sideways"),
    (dict(decay_form="exponential"), "scale_km"),
    (dict(distance_unit="m"), "'m'"),
])
def test_pcen_rejects_unknown_values(kwargs, match):
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_neighbors.py tests/test_index.py -q -W error`
Expected: FAIL — `TypeError: adjacency() got an unexpected keyword argument
'max_distance_km'` and `TypeError: _decay() got an unexpected keyword
argument 'exponent'`.

- [ ] **Step 3: Implement the neighbour rules**

In `delhi_psi/neighbors.py`:

```python
def _adjacency_within_distance(polygon_gdf, id_col, neighbor_col,
                               max_distance_km):
    """Polygon-to-polygon band: j is a neighbour of i iff their shortest
    distance is <= max_distance_km * 1000 metres (EPSG:7760 is metric).

    `dwithin` is symmetric and matches every polygon with ITSELF at every
    radius (distance 0), so the left join never yields a missing partner and
    the self pair is the only one that has to be removed. Lists are written
    in the frame's row order, like the other two rules.
    """
    if max_distance_km is None:
        raise ValueError(
            "adjacency rule 'within_distance' requires max_distance_km")
    joined_gdf = gpd.sjoin(polygon_gdf, polygon_gdf, how="left",
                           predicate="dwithin",
                           distance=max_distance_km * 1000)
    id_col_left = id_col + "_left"
    id_col_right = id_col + "_right"
    joined_grouped = joined_gdf.groupby(id_col_left)

    out = polygon_gdf.copy()
    out[neighbor_col] = np.empty((len(out), 0)).tolist()

    for group in tqdm(joined_grouped.groups):
        group_list = list(joined_grouped.get_group(group)[id_col_right])
        # a polygon is within any distance of itself
        group_list.remove(group)
        group_idx = row_index(out, id_col, group)
        out.loc[group_idx, neighbor_col].extend(group_list)
    return out


def adjacency(polygon_gdf, *, id_col="USO_AREA_U", neighbor_col="nbrs_bbox",
              rule="bbox", max_distance_km=None):
    """Directed neighbour lists under `rule` ("bbox", "touch" or
    "within_distance").

    The column keeps its historical name `nbrs_bbox` under EVERY rule — it is
    part of the July 2025 baseline's column contract (spec § 5).

    max_distance_km is used by `within_distance` alone; passing it with any
    other rule is an error, mirroring the config rule (`build_neighbors`
    forwards the configured value unconditionally, and it is None there).
    """
    if rule != "within_distance" and max_distance_km is not None:
        raise ValueError(
            "max_distance_km is only used by adjacency rule "
            f"'within_distance', not {rule!r}")
    if rule == "bbox":
        return _adjacency_bbox(polygon_gdf, id_col, neighbor_col)
    if rule == "touch":
        return _adjacency_touch(polygon_gdf, id_col, neighbor_col)
    if rule == "within_distance":
        return _adjacency_within_distance(polygon_gdf, id_col, neighbor_col,
                                          max_distance_km)
    raise ValueError(
        f"unknown adjacency rule {rule!r}; allowed values: "
        "['bbox', 'touch', 'within_distance']")


def boundary_distances(polygon_gdf, *, neighbor_col="nbrs_bbox",
                       nbr_dist_col="nbrs_dist_boundary",
                       id_col="USO_AREA_U"):
    """Add [(neighbor_id, distance_km), ...] per row, measured POLYGON TO
    POLYGON — 0 for every touching or overlapping neighbour.

    Same OUTPUT shape as `centroid_distances`, but built over an
    id -> geometry dict made ONCE (the `_adjacency_touch` pattern), never the
    per-neighbour boolean-mask lookup `centroid_distances` inherited from the
    2025 script. On a MultiPolygon shapely's `distance` is the minimum over
    the parts, which is the intended meaning.
    """
    out = polygon_gdf.copy()
    out[nbr_dist_col] = np.empty((len(out), 0)).tolist()
    geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
    for idx, row in tqdm(out.iterrows(), total=len(out)):
        geom = geoms[row[id_col]]
        out.at[idx, nbr_dist_col] = [
            (neighbor_id, geom.distance(geoms[neighbor_id]) / 1000)
            for neighbor_id in row[neighbor_col]]
    return out
```

Also update the module docstring's first line to say the adjacency rules are
`bbox`, `touch` and `within_distance`, and that distances come in two
definitions (centroid and boundary).

- [ ] **Step 4: Implement the decay dispatch**

In `delhi_psi/index.py`, add `import math` at the top and replace `_decay`:

```python
DECAY_FORMS = ("inverse_linear", "none", "inverse_power", "exponential")


def _decay(distance_km, decay_form, distance_unit, *, exponent=None,
           scale_km=None):
    """The distance-decay weight w(D).

    inverse_linear: 1/(1+D) — the July 2025 rule.
    none:           1 — every neighbour counts in full.
    inverse_power:  1/(1+D)^exponent; exponent 1 reproduces inverse_linear.
    exponential:    exp(-D/scale_km).

    A parameter the form does not use is an error, never an ignored value.
    """
    if distance_unit != "km":
        raise ValueError(
            f"unknown decay distance unit {distance_unit!r}; allowed values: "
            "['km']")
    if decay_form not in DECAY_FORMS:
        raise ValueError(
            f"unknown decay form {decay_form!r}; allowed values: "
            f"{list(DECAY_FORMS)}")
    if decay_form == "inverse_power":
        if exponent is None:
            raise ValueError("decay form 'inverse_power' requires exponent")
    elif exponent is not None:
        raise ValueError(
            f"exponent is not used by decay form {decay_form!r}; it is used "
            "by 'inverse_power'")
    if decay_form == "exponential":
        if scale_km is None:
            raise ValueError("decay form 'exponential' requires scale_km")
    elif scale_km is not None:
        raise ValueError(
            f"scale_km is not used by decay form {decay_form!r}; it is used "
            "by 'exponential'")

    if decay_form == "inverse_linear":
        return 1 / (1 + distance_km)
    if decay_form == "none":
        return 1.0
    if decay_form == "inverse_power":
        return 1 / (1 + distance_km) ** exponent
    return math.exp(-distance_km / scale_km)
```

Then thread the two parameters through:

- `pcen(...)`: add `exponent=None, scale_km=None` to the signature right
  after `distance_unit="km"`; change the probe call to
  `_decay(0.0, decay_form, distance_unit, exponent=exponent, scale_km=scale_km)`
  (it still fails early on a city with no links); change the in-loop call to
  `_decay(nbr_dist, decay_form, distance_unit, exponent=exponent, scale_km=scale_km)`.
- `service_index(...)`: add the same two parameters and forward them to
  `pcen`.

Nothing else in either function changes — `pcen` still reads `nbr_dist_col`
and does not know which distance definition filled it.

- [ ] **Step 5: Wire the pipeline**

In `delhi_psi/pipeline.py`:

(a) beside the other column constants:

```python
NBRS_DIST_COL = "nbrs_dist_bbox"
# Compute-local: `index_frames` builds it, hands it to pcen and drops it
# before returning, so io.SHAPEFILE_DROP_COLUMNS and the CSV/shapefile column
# contract are untouched and one stored artifact serves every decay.* value.
NBRS_DIST_BOUNDARY_COL = "nbrs_dist_boundary"
```

(b) in `build_neighbors`, log the band and forward it:

```python
    log.info("adjacency: rule=%s band_km=%s", methodology.adjacency.rule,
             methodology.adjacency.max_distance_km)
    frame = neighbors.adjacency(
        frame, id_col=id_col, neighbor_col=NBRS_COL,
        rule=methodology.adjacency.rule,
        max_distance_km=methodology.adjacency.max_distance_km)
```

(c) in `index_frames`, choose the distance column, pass the decay
parameters, and drop the column again:

```python
    exclusion = methodology.exclusion
    universe = apply_exclusion(neighbor_frame, dropped=dropped,
                               stage=exclusion.stage, id_col=id_col)

    # `boundary` needs polygon-to-polygon distances, which the stored
    # artifact deliberately does not carry. Build them HERE, on the
    # already exclusion-stripped lists (so nothing needs re-stripping), and
    # drop the column before returning.
    nbr_dist_col = NBRS_DIST_COL
    if methodology.decay.distance == "boundary":
        universe = neighbors.boundary_distances(
            universe, neighbor_col=NBRS_COL,
            nbr_dist_col=NBRS_DIST_BOUNDARY_COL, id_col=id_col)
        nbr_dist_col = NBRS_DIST_BOUNDARY_COL
```

```python
        out = index.service_index(
            out, amount_col, service=service, denominator=denominator,
            nbr_dist_col=nbr_dist_col, lookup_frame=amounts,
            absent_neighbor=exclusion.absent_neighbor,
            include_neighbors=include_neighbors,
            decay_form=methodology.decay.form,
            distance_unit=methodology.decay.distance_unit,
            exponent=methodology.decay.exponent,
            scale_km=methodology.decay.scale_km,
            id_col=id_col)

    result = index.overall_psi(
        out, second_normalization=methodology.second_normalization)
    if NBRS_DIST_BOUNDARY_COL in result.columns:
        result = result.drop(columns=[NBRS_DIST_BOUNDARY_COL])
    return result
```

(d) `methodology_stamp` records the band:

```python
    return {
        "adjacency": {
            "rule": str(methodology.adjacency.rule),
            "max_distance_km": methodology.adjacency.max_distance_km,
        },
        "barrier": {
            "rule": str(methodology.barrier.rule),
            "combine": combine if isinstance(combine, str)
            else [str(layer) for layer in combine],
        },
    }
```

`check_methodology_stamp` needs no change: it iterates the stamp, and for an
artifact built before this key existed `stored.get(block, {}).get(key)`
yields None, which equals the configured None for `bbox`/`touch`.
`apply_exclusion` needs no change either — it strips `nbrs_bbox` and
`nbrs_dist_bbox` only, and the boundary column is built after it.

- [ ] **Step 6: Run the unit tests to verify they pass**

Run: `uv run pytest tests/test_neighbors.py tests/test_index.py -q -W error`
Expected: PASS.

- [ ] **Step 7: Add the variant methodology helper**

Append to `tests/oraculum_fixtures.py`:

```python
def variant_methodology(base, variant, *, city=ORACULUM, types=None,
                        stage=None):
    """`base`'s methodology with `tests/variants.py`'s `variant` applied.

    Layered on `methodology_with`, so the SCENARIO travels with it: pass the
    scenario's `types`/`stage`. Without that, `code-2025`'s own
    `exclusion.types: [RV]` would drop RV (and messy's N) from the production
    frame while the variants CSV keeps them — and RV is the settlement the
    § 4.1 pins name.

    A block the variant does not mention keeps `base`'s: today only the band
    variants override `adjacency`, and every variant states each block it
    does override IN FULL.
    """
    from dataclasses import replace

    from delhi_psi.config import (
        AdjacencyConfig, AdjacencyRule, DecayConfig, DecayDistance, DecayForm,
    )
    from tests.variants import VARIANTS

    methodology = methodology_with(base, types=types, stage=stage, city=city)
    spec = VARIANTS[variant]
    if "adjacency" in spec:
        block = spec["adjacency"]
        methodology = replace(methodology, adjacency=AdjacencyConfig(
            rule=AdjacencyRule(block["rule"]),
            max_distance_km=block.get("max_distance_km")))
    if "decay" in spec:
        block = spec["decay"]
        methodology = replace(methodology, decay=DecayConfig(
            form=DecayForm(block["form"]),
            distance_unit=block["distance_unit"],
            distance=DecayDistance(block["distance"]),
            exponent=block.get("exponent"),
            scale_km=block.get("scale_km")))
    return methodology
```

- [ ] **Step 8: Write the § 4.2 comparison**

Create `tests/test_variants_match_reference.py`:

```python
"""Production reproduces the independent reference on every variant (§ 4.2).

ONE scenario per city — `city.scenarios[0]`, the one the variants CSV is
written for — because the exclusion machinery is proven elsewhere and is
orthogonal to the two knobs this cycle adds. The scenario travels with the
methodology (`variant_methodology`), so the production frame reports exactly
the settlements the CSV holds.
"""
import pandas as pd
import pytest

from delhi_psi.pipeline import compute_frames
from tests.cities import CITIES, MESSY
from tests.oraculum_fixtures import variant_methodology
from tests.test_profiles_match_reference import METRIC_MAP
from tests.variants import VARIANTS

BASE_PROFILE = "code-2025"
DENOMS = ("pop", "popdensity")
CASES = [(city, variant) for city in CITIES for variant in sorted(VARIANTS)]


def case_id(case):
    city, variant = case
    return f"{city.name}-{variant}"


@pytest.fixture(scope="module")
def expected():
    return {city.name: pd.read_csv(
        city.fixtures / "variants_expected_values.csv") for city in CITIES}


def produced(city, variant, denom):
    scenario = city.scenarios[0]
    methodology = variant_methodology(
        BASE_PROFILE, variant, city=city,
        types=scenario.exclusion_types, stage=scenario.stage)
    return compute_frames(
        city.load_settlements(), {"canal": city.load_barriers()},
        city.load_services(), None, methodology, denom,
        mapping=city.mapping(), scheme=city.scheme).set_index("USO_AREA_U")


@pytest.mark.parametrize("denom", DENOMS)
@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_production_matches_the_reference_on_each_variant(expected, case,
                                                          denom):
    city, variant = case
    got = produced(city, variant, denom)
    block = expected[city.name]
    block = block[(block["rule"] == variant) & (block["denom"] == denom)]
    exp = block.pivot(index="settlement", columns="metric", values="value")
    assert set(got.index) == set(exp.index)
    for prod_col, metric in METRIC_MAP.items():
        for sid in exp.index:
            assert got.loc[sid, prod_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-12), (city.name, variant, denom,
                                                    sid, prod_col)


def test_the_boundary_column_never_leaves_index_frames():
    """The compute-local column is dropped before returning, and the stored
    `nbrs_dist_bbox` holds CENTROID distances under EVERY configuration —
    which is what lets one artifact serve every decay.* value. Messy `G`'s
    only neighbour is `M`, at centroid distance exactly 0 and boundary
    distance 0.45 km, so the column's content is unambiguous."""
    got = produced(MESSY, "boundary", "pop")
    assert "nbrs_dist_boundary" not in got.columns
    assert dict(got.loc["G", "nbrs_dist_bbox"]) == {"M": 0.0}
```

- [ ] **Step 9: Add the § 4.5 stamp pins**

In `tests/test_cli.py`, update the literal expected dict in
`test_neighbors_artifact_carries_the_methodology_stamp`:

```python
    assert frame.attrs["methodology"] == {
        "adjacency": {"rule": "bbox", "max_distance_km": None},
        "barrier": {"rule": "global_asymmetric", "combine": "any"},
    }
```

and append these pins beside the other stamp tests:

```python
# --- 3D: the band is part of the stamp (spec § 4.5) --------------------
def _stamped(methodology, profile="code-2025"):
    """A frame carrying nothing but a stamp: check_methodology_stamp reads
    `attrs` only, so this is the whole input it needs."""
    from delhi_psi import pipeline

    frame = pd.DataFrame({"USO_AREA_U": ["A"]})
    frame.attrs["profile"] = profile
    frame.attrs["methodology"] = pipeline.methodology_stamp(methodology)
    return frame


def _band_config(km):
    from dataclasses import replace

    from delhi_psi.config import AdjacencyConfig, AdjacencyRule
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    return replace(cfg, methodology=replace(
        cfg.methodology,
        adjacency=AdjacencyConfig(rule=AdjacencyRule.WITHIN_DISTANCE,
                                  max_distance_km=km)))


def test_the_stamp_records_the_band():
    from delhi_psi import pipeline
    from tests.oraculum_fixtures import oracle_config

    assert pipeline.methodology_stamp(
        oracle_config("code-2025").methodology)["adjacency"] == {
            "rule": "bbox", "max_distance_km": None}
    assert pipeline.methodology_stamp(
        _band_config(1.0).methodology)["adjacency"] == {
            "rule": "within_distance", "max_distance_km": 1.0}


@pytest.mark.parametrize("configured,fragment", [
    ("bbox", "within_distance"),      # a different RULE
    (1.5, "max_distance_km"),         # the same rule, a different radius
])
def test_an_artifact_built_at_another_band_is_refused(configured, fragment):
    from delhi_psi import pipeline, validate
    from tests.oraculum_fixtures import oracle_config

    frame = _stamped(_band_config(1.0).methodology)
    cfg = oracle_config("code-2025") if configured == "bbox" \
        else _band_config(configured)
    with pytest.raises(validate.ValidationError, match=fragment):
        pipeline.check_methodology_stamp(frame, cfg)


def test_a_pre_3d_artifact_still_loads_for_a_bbox_config():
    """3A-3C artifacts have no `max_distance_km` key: `stored.get(...)`
    yields None, which equals the configured None — so `code-2025`'s pinned
    colonies_neighbors.joblib keeps loading without a re-preprocess."""
    from delhi_psi import pipeline
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    frame = pd.DataFrame({"USO_AREA_U": ["A"]})
    frame.attrs["profile"] = "code-2025"
    frame.attrs["methodology"] = {
        "adjacency": {"rule": "bbox"},
        "barrier": {"rule": "global_asymmetric", "combine": "any"},
    }
    pipeline.check_methodology_stamp(frame, cfg)      # must not raise


def test_changing_only_the_decay_does_not_invalidate_an_artifact():
    """Decay is applied downstream in `compute`, so an artifact stays valid
    across every decay.* value — `boundary` included, whose distances are
    computed at compute time and never stored."""
    from dataclasses import replace

    from delhi_psi import pipeline
    from delhi_psi.config import DecayConfig, DecayDistance, DecayForm
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    frame = _stamped(cfg.methodology)
    other = replace(cfg, methodology=replace(
        cfg.methodology,
        decay=DecayConfig(form=DecayForm.EXPONENTIAL, distance_unit="km",
                          distance=DecayDistance.BOUNDARY, scale_km=1.0)))
    pipeline.check_methodology_stamp(frame, other)    # must not raise
```

- [ ] **Step 10: Run the new production tests**

Run: `uv run pytest tests/test_variants_match_reference.py tests/test_cli.py -q -W error`
Expected: PASS — 33 variant cases (2 cities × 8 variants × 2 denominators,
plus the boundary-column test) and the whole CLI file.

- [ ] **Step 11: Prove no committed number moved, then run everything**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
uv run pytest -q -W error
```

Expected: `git status` prints **nothing** — the shipped profiles use
`bbox` + `inverse_linear` + `centroid`, so every branch added here is
unreachable for them — and the suite is green (the full run takes about 12 minutes).

- [ ] **Step 12: Commit**

```bash
git add delhi_psi/neighbors.py delhi_psi/index.py delhi_psi/pipeline.py
git add tests/oraculum_fixtures.py tests/test_neighbors.py tests/test_index.py
git add tests/test_cli.py tests/test_variants_match_reference.py
git commit -m "feat(pipeline): within_distance adjacency, decay forms, boundary distance (DEL-18)"
```

---

### Task 5: the CLI round trip and the real-data proofs (spec § 4.2, § 4.6)

The last leg of § 4.2 — config file → artifact → compute → CSV — plus the
two data-gated runs. Everything up to Step 5 is ordinary work; Steps 6 and 7
are **run by the controller**, because they need `~/delhi_data`.

**Files:**
- Modify: `tests/oraculum_fixtures.py`
- Modify: `tests/test_variants_match_reference.py`

**Interfaces:**
- Consumes: `tests.variants.VARIANTS` (Task 1); `tests.oraculum_fixtures.oracle_profile_path` (existing); `tests.test_cli.data_dir` (module-scoped fixture) and `tests.test_cli.COLLAPSE_TO_REFERENCE` (existing); the variants CSVs (Task 3); the whole production path (Task 4).
- Produces: `tests.oraculum_fixtures.oracle_profile_path(base, directory, city=ORACULUM, *, methodology_overrides=None, name=None) -> Path` — two new keyword-only parameters, both defaulting to today's behaviour.

- [ ] **Step 1: Write the failing CLI test**

Append to `tests/test_variants_match_reference.py` (extending its imports
with `from delhi_psi import cli`, `from tests.oraculum_fixtures import
oracle_profile_path`, `from tests.test_cli import COLLAPSE_TO_REFERENCE,
data_dir  # noqa: F401  (module-scoped fixture)` and
`from tests.cities import ORACULUM`):

```python
# --- the CLI leg: config file -> artifact -> compute (spec § 4.2) ------
# A variant profile must state each block it overrides IN FULL, and it must
# also state the SCENARIO the variants CSV was written for — Oraculum's
# `baseline`, i.e. no exclusion at all, where the shipped profile excludes
# the category RV.
BASELINE_EXCLUSION = {"types": [], "stage": "post_neighbors",
                      "absent_neighbor": "swallowed"}


@pytest.mark.parametrize("variant", ["band_small_boundary", "exp1"])
def test_a_derived_variant_profile_runs_end_to_end(expected, data_dir,  # noqa: F811
                                                   tmp_path, variant):
    """Proves the whole chain the in-memory test skips: YAML -> load_config
    -> preprocess -> the stamped artifact -> compute -> CSV. `exp1` is here
    for `scale_km`; `band_small_boundary` for the band, the boundary
    distance and the stamped `max_distance_km` together.
    """
    overrides = dict(VARIANTS[variant])
    overrides["exclusion"] = BASELINE_EXCLUSION
    profile = oracle_profile_path(BASE_PROFILE, tmp_path,
                                  methodology_overrides=overrides,
                                  name=variant)
    out = tmp_path / variant
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    assert cli.main(["compute", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0

    block = expected["oraculum"]
    block = block[(block["rule"] == variant) & (block["denom"] == "pop")]
    exp = block.pivot(index="settlement", columns="metric", values="value")
    got = pd.read_csv(
        out / "delhi_psi_code-2025_pop_2020.csv").set_index("USO_AREA_U")
    assert set(got.index) == set(exp.index)
    for got_col, metric in COLLAPSE_TO_REFERENCE.items():
        for sid in exp.index:
            assert got.loc[sid, got_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-9), (variant, sid, got_col)


def test_the_stored_artifact_records_the_bands_radius(data_dir, tmp_path):  # noqa: F811
    """The stamp is what stops a `compute` reading one band's neighbour
    lists under another band's config."""
    from delhi_psi import io

    overrides = dict(VARIANTS["band_small"])
    overrides["exclusion"] = BASELINE_EXCLUSION
    profile = oracle_profile_path(BASE_PROFILE, tmp_path,
                                  methodology_overrides=overrides,
                                  name="band_small")
    out = tmp_path / "stamped_band"
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    frame = io.read_neighbors(out / "colonies_neighbors.joblib")
    assert frame.attrs["methodology"]["adjacency"] == {
        "rule": "within_distance", "max_distance_km": 0.25}


def test_compute_refuses_an_artifact_built_at_another_band(data_dir,  # noqa: F811
                                                           tmp_path, capsys):
    """Build at 0.25 km, compute at 0.75 km: every number compute would
    produce describes a neighbourhood nobody built."""
    import shutil

    small = dict(VARIANTS["band_small"], exclusion=BASELINE_EXCLUSION)
    large = dict(VARIANTS["band_large"], exclusion=BASELINE_EXCLUSION)
    out = tmp_path / "band_mismatch"
    built = oracle_profile_path(BASE_PROFILE, tmp_path,
                                methodology_overrides=small, name="small")
    assert cli.main(["preprocess", "--config", str(built),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    other = oracle_profile_path(BASE_PROFILE, tmp_path,
                                methodology_overrides=large, name="large")
    assert cli.main(["compute", "--config", str(other),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 1
    err = capsys.readouterr().err
    assert "max_distance_km" in err and "0.25" in err and "0.75" in err
```

Note the artifact name: both derived profiles keep `profile: code-2025`, so
they share `colonies_neighbors.joblib` — which is exactly what makes the
mismatch test possible without copying files. Each other test uses its own
`--out-dir`.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_variants_match_reference.py -q -W error`
Expected: FAIL — `TypeError: oracle_profile_path() got an unexpected keyword
argument 'methodology_overrides'`.

- [ ] **Step 3: Let the derived profile carry methodology overrides**

In `tests/oraculum_fixtures.py`, replace `oracle_profile_path` with:

```python
def oracle_profile_path(base, directory, city=ORACULUM, *,
                        methodology_overrides=None, name=None):
    """Write the derived profile as YAML into `directory`; return the path,
    for the tests that drive the real CLI with `--config <path>`.

    `methodology_overrides` is a mapping of TOP-LEVEL methodology sub-blocks
    (`adjacency`, `decay`, `exclusion`) that REPLACE `raw["methodology"]
    [<block>]` wholesale — never a deep merge. A variant therefore always
    states its full block and no key is ever inherited from the base
    profile, which is what `tests/variants.py` is written for.

    `name` distinguishes two derived profiles written into ONE directory;
    without it the filename is the historic `<base>.oracle.yaml`.

    Precondition: `base`'s `exclusion.types` (or the override's) must be
    category names present in the oracle vocabulary (`ORACLE_VOCABULARY`,
    since the swapped-in mapping is its identity) — the shipped profiles
    satisfy this. A profile whose `exclusion.types` names a category the
    oracle-6 identity does not produce (e.g. a collapsing profile's
    `non-urban`) fails to load through this helper with the "not categories
    produced by categories.mapping" error.
    """
    import yaml

    from delhi_psi.config import PROFILES_DIR

    raw = yaml.safe_load((PROFILES_DIR / f"{base}.yaml").read_text())
    raw["categories"] = {"scheme": city.scheme, "mapping": city.mapping()}
    for block, values in (methodology_overrides or {}).items():
        raw["methodology"][block] = dict(values)
    stem = base if name is None else f"{base}.{name}"
    path = Path(directory) / f"{stem}.oracle.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path
```

Every existing caller passes two positional arguments and no keywords, so
none of them changes.

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run: `uv run pytest tests/test_variants_match_reference.py -q -W error`
Expected: PASS.

- [ ] **Step 5: Full suite, byte-identity, commit**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
uv run pytest -q -W error
```

Expected: empty `git status`, suite green.

```bash
git add tests/oraculum_fixtures.py tests/test_variants_match_reference.py
git commit -m "test(cli): derived variant profile round trip (DEL-18)"
```

- [ ] **Step 6: DATA-GATED — the `code-2025` real-data verify (controller runs this)**

**The controller performs this step and supplies the output.** It is
data-gated (`~/delhi_data`) and takes about 10 minutes. Nothing on the
default path changed, so this can only be a no-op — and that is the claim
being proven. Note the out-dir is **outside** `~/delhi_data`, which is
bisynced to the shared drive:

```bash
OUT="$HOME/delhi_data/phase3_verify"
mkdir -p "$OUT"
# Optional, saves ~3 minutes: reuse an existing dedup cache by COPYING it
# out of the data directory (never point --out-dir inside it).
cp ~/delhi_data/phase3_verify/*.dedup.gpkg ~/delhi_data/phase3_verify/*.dedup.stamp "$OUT"/ 2>/dev/null || true
uv run delhi-psi preprocess --config code-2025 --data-dir ~/delhi_data --out-dir "$OUT"
uv run delhi-psi compute    --config code-2025 --data-dir ~/delhi_data --out-dir "$OUT"
uv run python scripts/verify_against_baseline.py --config code-2025 --verify-dir "$OUT"
```

Expected: `preprocess` reports **4,357 settlements, 595 barrier-flagged** and
logs `adjacency: rule=bbox band_km=None`; `compute` reports
`categories: scheme=uso-10 n_categories=10` and 4,131 reported rows; the
verify prints
`PASS — new run equivalent to July 2025 baseline within tolerance` with every
`max abs deviation` line reading `0.000e+00` on all 30 compared numeric
columns. **Any other result is a stop condition** (spec § 9).

Record the date, the two row counts and the PASS line — Task 6's CHANGELOG
entry quotes them.

- [ ] **Step 7: DATA-GATED — the 10 km band timing note (controller runs this)**

**The controller performs this step and supplies the numbers.** Budget tens
of minutes: the `dwithin` join is fast, but `apply_barrier` and
`centroid_distances` are per-link Python loops and a 10 km band has roughly
two orders of magnitude more links than adjacency. `preprocess` ONLY — there
is no `compute` at 10 km, and nothing here is committed as a fixture.

Write the derived profile into a scratch directory (never under
`~/delhi_data`), give it its own `profile:` name so it gets its own artifact
filename, and time the run:

```bash
SCRATCH="$(mktemp -d)"
uv run python - "$SCRATCH" <<'PY'
import sys, yaml
from pathlib import Path
from delhi_psi.config import PROFILES_DIR

raw = yaml.safe_load((PROFILES_DIR / "code-2025.yaml").read_text())
raw["profile"] = "band-10km"
raw["methodology"]["adjacency"] = {"rule": "within_distance",
                                   "max_distance_km": 10.0}
raw["paths"].pop("neighbors_artifact", None)   # per-profile default name
out = Path(sys.argv[1]) / "band-10km.yaml"
out.write_text(yaml.safe_dump(raw, sort_keys=False))
print(out)
PY

/usr/bin/time -v uv run delhi-psi preprocess --config "$SCRATCH/band-10km.yaml" \
    --data-dir ~/delhi_data --out-dir "$SCRATCH" 2>&1 | tail -30
```

Then the neighbour-count summary from the artifact it wrote:

```bash
uv run python - "$SCRATCH" <<'PY'
import sys
from pathlib import Path
import numpy as np
from delhi_psi.io import read_neighbors

frame = read_neighbors(Path(sys.argv[1]) / "colonies_neighbors_band-10km.joblib")
degree = np.array([len(ids) for ids in frame["nbrs_bbox"]])
print(f"settlements: {len(frame)}")
print(f"links (directed, post-barrier): {int(degree.sum())}")
print(f"degree mean {degree.mean():.1f} median {np.median(degree):.0f} "
      f"min {degree.min()} max {degree.max()}")
print(f"stamp: {frame.attrs['methodology']['adjacency']}")
PY
rm -rf "$SCRATCH"
```

Expected: the stamp reads
`{'rule': 'within_distance', 'max_distance_km': 10.0}`, the settlement count
is 4,357, and the wall-clock plus the degree summary are the numbers Task 6
writes into `docs/methodology-config.md` § 6. Nothing is committed from this
run except those sentences.

---

### Task 6: docs, CHANGELOG, WORKPLAN (spec § 5)

**Files:**
- Modify: `docs/methodology-config.md`
- Modify: `CHANGELOG.md`
- Modify: `WORKPLAN.md`

**Interfaces:**
- Consumes: the controller's numbers from Task 5 Steps 6 and 7.
- Produces: no code; `docs/methodology-config.md` § 6 is what a Phase 6 ticket follows.

- [ ] **Step 1: Extend the § 1 switch table**

In `docs/methodology-config.md` § 1, change the `adjacency.rule` row's last
cell to read

```
memo § 1 (DEL-19) — bbox adjacency invents neighbours. A third value, `within_distance`, is the Phase 6 distance band (§ 6, DEL-36/39)
```

and add three rows immediately after the `decay.distance_unit` row:

```markdown
| `adjacency.max_distance_km` | — (unused) | — (unused) | DEL-36 — the band's radius in km, polygon-to-polygon. **Required** iff `adjacency.rule: within_distance`, and **rejected** otherwise; `>= 0`, where 0 means "every polygon that intersects i" (§ 6) |
| `decay.form` | `inverse_linear` | `inverse_linear` | DEL-37 — the decay weight w(D): `inverse_linear` = 1/(1+D), `none` = 1, `inverse_power` = 1/(1+D)^`exponent`, `exponential` = e^(−D/`scale_km`). `exponent` / `scale_km` are required by, and only by, their own form |
| `decay.distance` | `centroid` | `centroid` | DEL-37 — what D means: `centroid` (centroid-to-centroid, as every run so far) or `boundary` (polygon-to-polygon, so every touching or overlapping neighbour is at 0 and lends its services undecayed) |
```

- [ ] **Step 2: Add the § 4 proofs**

In `docs/methodology-config.md` § 4, append two bullets after the
`tests/test_profiles_match_reference.py` one:

```markdown
- `tests/test_variant_rules.py` — **both** cities: the hand-derivable pins for
  the injectable parameters on the REFERENCE side — a 0 km band is exactly
  the `intersects` neighbourhood (and on the messy city that is `touch` plus
  the corner-only `L`/`T` pair), the bands are strictly nested,
  `inverse_power` 1 reproduces `inverse_linear` on every PCEN, and `pow2` /
  `exp1` / `none` / `boundary` are pinned at closed-form arithmetic written
  out in the test. Plus the loader's rejection table: a parameter its form or
  rule does not use is an error, never an ignored value.
- `tests/test_variants_match_reference.py` — **both** cities: production ==
  the independent reference at 1e-12 on all eight derived variants × both
  denominators (`tests/fixtures/<city>/variants_expected_values.csv`,
  generator-emitted and covered by the CI drift guard), plus a CLI round trip
  through a derived variant profile YAML — config file → stamped artifact →
  `compute` → CSV — and the stamp refusing an artifact built at another band.
```

- [ ] **Step 3: Write the new § 6**

Append to `docs/methodology-config.md`, after § 5. `<…>` placeholders are
filled from Task 5 Step 7's output; nothing else is left open:

````markdown
## 6. Phase 6 sweeps: one profile per point

Since cycle 3D (28 Aug 2026) the two remaining methodological choices —
which settlements count as neighbours at a DISTANCE, and how distance
discounts their services — are config values with reference rules and oracle
pins. A sweep point (DEL-36 thresholds, DEL-37 decay weights, DEL-39
adjacency comparison) is therefore one YAML file plus the § 3 registration
steps: no code fork, no branch.

### The whole profile for one point

`delhi_psi/profiles/band-1km.yaml` — a copy of `code-2025` with **two**
values changed (`adjacency.rule`, `adjacency.max_distance_km`) and the
profile renamed. Nothing else moves, which is what makes the diff against
`code-2025`'s outputs attributable to the band alone:

```yaml
profile: band-1km
crs: {epsg: 7760}
paths:
  data_dir: ~/delhi_data            # overridable: --data-dir, DELHI_DATA_DIR
                                    # neighbors_artifact: omitted on purpose —
                                    # it defaults to
                                    # colonies_neighbors_band-1km.joblib, so
                                    # this profile cannot overwrite code-2025's
layers:
  settlements: {path: uso_update_sep2021/uso_update_sep2021.shp,
                id_col: USO_AREA_U, type_col: USO_FINAL}
  population:  {path: pop_colony_wp_2020_jjc_adjusted.csv,
                id_col: uso_area_u, value_col: population,
                missing: drop}
  bounds: delhi_bounds_buffer/delhi_bounds_buffer.shp
  ndmc_center: ndmc_center7760/ndmc_center7760.shp
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
  line:  {road: Public Services/Major Road/Road.shp}
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
validate:
  max_missing_population: 15
outputs:
  denominators: [pop, popdensity]
  formats: [csv, shp, joblib]
  name_template: "delhi_psi_{profile}_{denominator}_2020"
```

For 5 km and 10 km, change the one number. For a decay sweep, leave
`adjacency` at `code-2025`'s `{rule: bbox}` and change the `decay` block
instead — e.g. `{form: inverse_power, exponent: 2, distance: centroid,
distance_unit: km}` or `{form: exponential, scale_km: 1.0, distance:
centroid, distance_unit: km}`. `exponent` and `scale_km` are required by, and
rejected outside of, their own form: a profile that carries `exponent` under
`inverse_linear` fails to load naming the key, rather than silently ignoring
it.

### X = 0 is not `touch`

`max_distance_km: 0` means "every polygon whose distance to i is 0" —
intersection-inclusive. That is **not** the `touch` rule: `touch` requires the
shared intersection to have positive LENGTH, so it misses a pair meeting at a
single corner point. On the messy fixture city the 0 km band is exactly the
`touch` neighbourhood plus the corner-only pair `L`/`T`, whose intersection is
a Point. Overlapping polygons are in both (an overlap's intersection is a
polygon, whose `.length` is its perimeter). If you want "shares a border",
write `rule: touch`; the band at 0 is the intersection rule.

### Centroid vs boundary distance

`decay.distance` names what D is in the weight, and the two definitions
disagree in BOTH directions:

- A big neighbour is far by centroid and near by boundary. On the messy city
  `H` and `L` are disjoint but interlocked: 0.131519 km apart at the
  boundary, 1.127237 km apart at the centroids. Under `inverse_linear` that
  is a weight of 0.88 against 0.47 for the same physical adjacency.
- A neighbour with a hole in it can be near by centroid and far by boundary.
  `M` is a two-part settlement whose centroid falls in its own gap, exactly
  on `G`'s centroid: centroid distance 0 (weight **1**, as if the two were
  the same place) while the polygons are 0.45 km apart (weight 1/1.45).

Under `boundary` every touching or overlapping neighbour is at distance 0 and
therefore lends its services **undecayed**, under all four forms. That is a
real methodological choice, not a bug fix — hence the config value.

**The band radius is judged against BOUNDARY distances even when
`decay.distance: centroid`,** because `within_distance` is polygon-to-polygon
by definition. Do not size a radius from the derivation worksheet's
centroid-to-centroid numbers.

### What a band costs on the real layer

Measured once, on the real settlement layer (4,357 settlements), by
`delhi-psi preprocess` with `adjacency: {rule: within_distance,
max_distance_km: 10.0}` into a scratch `--out-dir` — no `compute`, and
nothing committed as a fixture:

- **wall clock:** `<MM:SS from the timed run>`
- **directed links after the barrier rule:** `<links>` (degree mean
  `<mean>`, median `<median>`, max `<max>`), against a mean degree of about 7
  under `bbox`
- **measured:** `<YYYY-MM-DD>`, commit `<short sha>`

The `dwithin` join itself is fast; the time goes into `apply_barrier` and
`centroid_distances`, which are per-link Python loops. Budget tens of minutes
per wide-band `preprocess`, and remember each band needs its own
`preprocess`: the neighbours artifact is stamped with
`adjacency.max_distance_km`, and `compute` refuses an artifact built at a
different radius.
````

- [ ] **Step 4: Write the CHANGELOG entry**

At the TOP of the `[Unreleased]` list in `CHANGELOG.md` (newest first, as the
3C entry sits above 3B), insert the entry below. Three fields are filled in
by hand: `<date>` and `<n>` (the reported-row count) come from Task 5
Step 6's real-data run, and `<final count>` is the number `uv run pytest -q
-W error` prints in this task's Step 6.

```markdown
- Phase 3D injectable parameters (DEL-18): the last two methodological
  choices that were still code are config values. `methodology.adjacency.rule`
  gains **`within_distance`** with a required `max_distance_km` — a
  polygon-to-polygon band in km, `>= 0`, where 0 is the intersection rule
  (corner-only touches and overlaps included), implemented with geopandas'
  `dwithin` spatial join and pinned pair-for-pair against brute force.
  `methodology.decay` gains **`form`** (`inverse_linear` | `none` |
  `inverse_power` + `exponent` | `exponential` + `scale_km`) and a required
  **`distance`** (`centroid` | `boundary`); under `boundary` every touching or
  overlapping neighbour is at distance 0 and lends its services undecayed,
  computed in a compute-local column so the stored neighbours artifact still
  carries centroid distances and stays valid across every `decay.*` value. A
  parameter its rule or form does not use is **rejected at load naming the
  key**, never ignored. The artifact stamp now records
  `adjacency.max_distance_km`, so a `compute` against another band's
  neighbour lists is refused; artifacts built by 3A–3C, which lack the key,
  keep loading. Proved on BOTH fixture cities by eight derived variants
  (`tests/fixtures/{oraculum,messy}/variants_expected_values.csv`,
  generator-emitted and drift-guarded, scored by the independent reference):
  production reproduces the reference at 1e-12 on every variant × denominator,
  plus a CLI round trip through a derived variant profile YAML. One table,
  `tests/variants.py`, feeds both sides — it imports nothing from the repo, and
  the reference builds its rule-sets from it by renaming block keys only, so
  the two cannot drift. Hand pins: a 0 km band is the `intersects`
  neighbourhood on both cities (`touch` ∪ {`L`↔`T`} on messy — the
  corner-only pair `touch` cannot see); the bands are strictly nested
  (10/12/14 undirected pairs on Oraculum, 5/8/10 on messy at 0 / 0.25 /
  0.75 km); `inverse_power` 1 reproduces `inverse_linear` exactly; `RV` and
  `D`, Oraculum's only single-neighbour settlements, pin `pow2` and `exp1` at
  closed form; and `H`/`L` (boundary 0.131519 km vs centroid 1.127237 km) and
  `G`/`M` (boundary 0.45 km vs centroid **0**) pin that centroid distance
  misstates proximity in BOTH directions. **No number moved**: both shipped
  profiles gained exactly one key, `decay.distance: centroid`, which names the
  definition they have always used; both cities' `expected_values.csv` and all
  four `production/*.csv` are byte-identical, and
  `scripts/verify_against_baseline.py --config code-2025` still reports
  `PASS — new run equivalent to July 2025 baseline within tolerance` with max
  abs deviation `0.000e+00` on all 30 compared numeric columns (real-data
  proof, `<date>`: `delhi-psi preprocess` — 4,357 settlements,
  595 barrier-flagged; `delhi-psi compute` — `<n>` reported,
  `categories: scheme=uso-10 n_categories=10`). Phase 6's sweeps (DEL-36
  thresholds, DEL-37 decay weights, DEL-39 adjacency comparison) are now
  loops over YAML profiles. Tests 386 → `<final count>`. Docs:
  `docs/methodology-config.md` §§ 1, 4, 6 (a complete `band-1km.yaml`, the
  X = 0 ≠ `touch` note, the centroid-vs-boundary note and the real-layer
  timing note).
```

- [ ] **Step 5: Close the WORKPLAN item**

In `WORKPLAN.md`, replace the Phase 3 item (currently `- [~] Modular &
extensible structure…` with its "PARTIAL (3A…)" note) with:

```markdown
- [x] Modular & extensible structure: distance thresholds, decay weights,
      service sets, adjacency/barrier rules, and category mappings injectable
      as parameters (feeds the Phase 6 sweeps) [DEL-18]
      — done 28 Aug 2026 (3D): every item in that list is now a config value.
      3A made adjacency/barrier rules, service sets, denominators, exclusion
      and units config; 3B the category mappings; 3D adds the last two —
      `adjacency.rule: within_distance` with `adjacency.max_distance_km`
      (polygon-to-polygon band, `>= 0` km) and `decay.form`
      (`inverse_linear` | `none` | `inverse_power` + `exponent` |
      `exponential` + `scale_km`) with `decay.distance`
      (`centroid` | `boundary`). Each new value has a reference-implementation
      rule and oracle pins on both fixture cities (eight derived variants in
      `tests/fixtures/*/variants_expected_values.csv`), so a Phase 6 sweep
      point is one YAML file. No behaviour changed: both shipped profiles
      gained only `decay.distance: centroid`, naming what they always did;
      every committed fixture is byte-identical and the real-data baseline is
      0.000e+00. Spec
      `docs/superpowers/specs/2026-08-28-injectable-parameters-design.md`,
      plan `docs/superpowers/plans/2026-08-28-injectable-parameters.md`,
      procedure `docs/methodology-config.md` § 6
```

and append a note line to each of the three Phase 6 items:

```markdown
- [ ] Distance-threshold sweep: 1 km / 5 km / 10 km; show index stability
      [DEL-36]
      — profile only since 3D: `adjacency: {rule: within_distance,
      max_distance_km: X}`, one YAML per point; see
      `docs/methodology-config.md` § 6 (a complete `band-1km.yaml`, and the
      measured cost of a 10 km `preprocess`)
- [ ] Parameterize and vary the decay weight 1/(1+D) (currently arbitrary);
      revisit centroid-to-centroid vs. other distance definitions [DEL-37]
      — profile only since 3D: `decay.form` (`none` | `inverse_power` +
      `exponent` | `exponential` + `scale_km`) and `decay.distance`
      (`centroid` | `boundary`); changing only `decay.*` does not invalidate
      the neighbours artifact, so a decay sweep needs no re-`preprocess`
- [ ] Adjacency-method comparison (bbox vs. touch) as a reported variant —
      whichever rule Raj ratifies in bug-audit item 1 is the main text, the
      other is this variant [DEL-39]
      — profile only since 3D: `adjacency.rule` is `bbox` | `touch` |
      `within_distance`, and a 0 km band is the third comparison point (the
      intersection rule, which is `touch` plus corner-only contacts)
```

- [ ] **Step 6: Run the suite one last time and commit**

Run:

```bash
for g in scripts/generate_*_fixtures.py; do uv run python "$g"; done
git status --porcelain -- tests/fixtures/
uv run pytest -q -W error
```

Expected: empty `git status`, suite green. (`tests/test_ci_workflow.py` reads
the workflow file, which this cycle does not touch.)

```bash
git add docs/methodology-config.md CHANGELOG.md WORKPLAN.md
git commit -m "docs: injectable parameters — sweep procedure, changelog, workplan (DEL-18)"
```

- [ ] **Step 7: Jira (controller performs this step)**

Not a repo change; recorded here so spec § 5 is complete:

- **DEL-18 → Done**, with an evidence comment: what is config now, the two
  new fixtures, the byte-identity and real-data results from Step 6 of
  Task 5, and links to the spec, this plan and `docs/methodology-config.md`
  § 6.
- **DEL-36 / DEL-37 / DEL-39** — one comment each naming the exact keys and
  values to set (as in the WORKPLAN notes above) and pointing at
  `docs/methodology-config.md` § 6.
- Remove the `Blocks` links **DEL-18 → DEL-40 / DEL-41 / DEL-42**: service
  sets have been config since 3A, so those were never really blocked.
