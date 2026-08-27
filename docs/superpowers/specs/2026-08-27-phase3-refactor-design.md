# Phase 3 Refactor — Design Spec (DEL-15; cycle 3A in full)

Date: 2026-08-27
Status: **approved by owner (2026-08-27)** — approval conditional on the
ultracode review fixes below being applied; they are (rev 2).
Branch: `del-15-phase3-refactor` (off `origin/main` at c9bce27)
Parent plan: WORKPLAN.md Phase 3 (Epic DEL-4); § Decisions made → "End-state
architecture — Option B"; Jira DEL-15. Restates DEL-16 after DEL-23.

Decision log (brainstorm, 27 Aug 2026):
- Every Raj-gated methodology choice is a config switch; today's production
  behavior is the default. Bob meets Raj 28 Aug; § 8 lists the defaults to confirm.
- Phase 3 splits into three `/ship` cycles (§ 1). This spec designs 3A fully;
  3B and 3C get their own short specs when their external inputs exist.
- Config reaches the math as explicit keyword arguments to pure functions
  (approach 1 of 3), never as a config object threaded through the math.
- Reference-pinned config keys map 1:1 onto `tests/reference_impl.compute_city`
  knobs (§ 3); production behavior per profile is pinned by its own fixture (§ 4).
- Rev 2 (ultracode review, 31 confirmed findings → 6 root causes): exclusion
  modelled as the reference's two axes, not one; the reference CSV stays
  single and untouched; production fixtures added per profile; schema gaps
  (NDMC layer, validate block, missing population, denominators) closed;
  packaging traps spelled out; `render`/`verify` removed from the CLI.

## Purpose

Turn the working-but-monolithic pipeline (one 822-line module + two scripts,
importable only through `sys.path` hacks) into the installable `delhi_psi`
package the WORKPLAN decided on, with every methodology choice — adjacency,
barrier rule, decay, roads formula, denominator, second normalization,
exclusion semantics — a validated config value. The refactor changes **no
numbers**: the `code-2025` profile must reproduce today's production output
on the oracle city exactly (string-equal at `%.17g`) and the July 2025
real-data baseline at zero deviation. Raj's answers then land as profile
edits plus a fixture regeneration, and Phase 6 sweeps become loops over
profiles.

## 1. Phase 3 decomposition

| Cycle | Scope | Tickets | Gate |
|---|---|---|---|
| **3A** (this spec) | `delhi_psi` package, config schema, profiles + per-profile production fixtures, all methodology switches the reference already models, pipeline stages + CLI, validation-as-assertions | DEL-16, DEL-18 (partial — see below), DEL-21, DEL-22, DEL-25 | none — starts now |
| 3B | settlement-category mapping layer (`categories:` config block; 10/8/5/4-category runs) | DEL-17 | none remaining — the vocabulary is measured (§ 9 item 1) |
| 3C | messy-city fixture tier; bbox-adjacency and overlap-sharing fixes as new switch values; multi-barrier oracle coverage; reserved switches that need new reference rules | DEL-24, DEL-19, DEL-20 | Raj's decisions (DEL-13) |

DEL-16 restated: the `*_wards`/`*_buffer` variants it named were deleted under
DEL-23. The remaining duplication is the `create_service_index` /
`create_service_length_index` pair (identical apart from count-vs-length and a
swapped positional argument order) and the `road_count → road_length` special
case inside `calc_all_services`. 3A collapses both (§ 2).

DEL-18 is **partially** delivered by 3A: adjacency/barrier rules, service
sets, denominators, exclusion and units become parameters. Distance
thresholds (DEL-36 sweeps 1/5/10 km) and alternative decay weights (DEL-37)
are not — the schema leaves room for them (`adjacency` and `decay` are
mappings, not bare strings) and Phase 6 adds the values with their reference
rules. Record this on the ticket.

## 2. Package layout

```
delhi_psi/
  __init__.py      version; public re-exports
  config.py        Config dataclasses, YAML load + validation, profile lookup
  io.py            read layers/CSVs, write outputs, path resolution
                   (absorbs scripts/common.py; the neighbors artifact name is
                   config, ending the aug2026/2025 filename mismatch)
  validate.py      the check_shapefile battery and the post-compute sanity
                   checks, returning a Report and raising on failure
  geometry.py      reproject, remove_duplicate_geom, bbox frame, barrier flags,
                   distance_to_point_km (the NDMC distance)
  neighbors.py     adjacency(rule) + barrier(rule, combine) + centroid distances
  index.py         point_counts, road_lengths, pcen (Eq. 3), minmax (Eq. 2),
                   overall_psi (Eq. 1 + second_norm), exclusion handling —
                   pure functions with keyword knobs
  pipeline.py      preprocess(cfg) / compute(cfg) path-based stages, plus
                   compute_frames(...) — the in-memory seam the oracle uses
  cli.py           delhi-psi {preprocess,compute} --config X.yaml
  profiles/
    code-2025.yaml today's behavior (the default profile)
    manuscript.yaml the ideal rule-set
```

Rules:
- Math modules (`geometry`, `neighbors`, `index`) never import `config`.
  They take explicit parameters. Only `pipeline.py` sees a `Config`.
- One function per concept. `service_index(gdf, amount_col, *, decay,
  denominator, …)` replaces both `create_service_index` variants; it is fed by
  `point_counts()` or `road_lengths()`, whose output column is named by the
  service config (`road_length` for the line service, matching today), so the
  rename special case disappears.
- Every enum is validated at config load; the math functions still raise
  `ValueError` on an unknown value (no silent `UnboundLocalError` path).
- `pipeline.compute_frames(settlements, barriers: dict[str, GeoDataFrame],
  services: dict[str, GeoDataFrame], population: DataFrame | None,
  methodology: MethodologyConfig, denominator: str) -> GeoDataFrame` is the
  documented entry point for tests that hold frames rather than paths
  (replaces `tests/oraculum_fixtures.run_production_chain`). Exclusion
  overrides are applied by constructing a modified `MethodologyConfig`, never
  by mutating frames. The oracle fixture's `clinic` service maps to config
  `health` inside the test wiring, as `test_oracle_e2e.SERVICE_LAYOUT` does.

Files removed: `spatial_index_utils.py` (23 functions move; no shim — every
caller is in-repo), `scripts/preprocess.py`, `scripts/compute_psi.py`,
`scripts/common.py`, root `conftest.py`, every `sys.path.insert`.
Files kept as scripts importing from `delhi_psi`, each a thin wrapper:
- `scripts/verify_against_baseline.py` — its two comparison functions move
  to `delhi_psi/verify.py`; the script gains `--config` to derive the fresh
  output paths from `outputs.name_template`. Baseline paths stay the script's
  own arguments (they exist only for `code-2025`), so there is no `verify`
  CLI stage and no baseline key in the schema.
- `scripts/render_oracle_maps.py` — unchanged in role (dev tool that renders
  `docs/oracle/*.png` from fixtures); not a CLI stage. A `figures` stage for
  paper figures from `out_dir` is Phase 4 (DEL-33), § 9.
- `scripts/generate_oraculum_fixtures.py` (geometry, unchanged) and the new
  `scripts/generate_production_fixtures.py` (§ 4) — both matched by the CI
  drift-guard glob.
- `scripts/check_oraculum_invariants.py` — unchanged; it reads the reference
  CSV, which does not move.

`pyproject.toml` changes, all in migration step 1 (§ 5) with the regenerated
`uv.lock` in the same commit so `uv sync --locked` in CI keeps passing:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["delhi_psi"]          # required: project name ≠ package name;
                                  # also ships delhi_psi/profiles/*.yaml
[project.scripts]
delhi-psi = "delhi_psi.cli:main"
[tool.pytest.ini_options]
testpaths = ["tests"]
```
`pyyaml` moves from the dev group to runtime dependencies. `uv.lock`'s root
entry changes from `virtual` to `editable`. `tests/` gains `__init__.py` so
`tests.oraculum_fixtures` and `tests.reference_impl` import without rootdir
tricks.

## 3. Config schema

One YAML per profile. Loaded into frozen dataclasses; unknown keys, missing
required keys and out-of-enum values raise `ConfigError` naming the key and
the allowed values.

```yaml
profile: code-2025
crs: {epsg: 7760}
paths:
  data_dir: ~/delhi_data            # overridable: --data-dir, DELHI_DATA_DIR
  out_dir: ~/delhi_data/phase3      # overridable: --out-dir
  neighbors_artifact: colonies_neighbors.joblib
layers:
  settlements: {path: uso_update_sep2021/uso_update_sep2021.shp,
                id_col: USO_AREA_U, type_col: USO_FINAL}
  population:  {path: pop_colony_wp_2020_jjc_adjusted.csv,
                id_col: uso_area_u, value_col: population,
                missing: drop}      # drop | error  (see below)
  bounds: delhi_bounds_buffer/delhi_bounds_buffer.shp
  ndmc_center: ndmc_center7760/ndmc_center7760.shp   # → ndmc_dist_km column
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
                                    # partial_weighted: reserved (§ 4)
    combine: any                    # any | [layer names]; which barrier flags OR together
  decay: {form: inverse_linear, distance_unit: km}   # 1/(1+d); only value in 3A
  roads: decayed                    # decayed | eq4_own_only
  second_normalization: true        # norm_psi = minmax(unnorm_psi); column absent when false
  exclusion:
    types: [RV]                     # raw USO_FINAL strings until 3B adds `categories:`
    stage: post_neighbors           # post_neighbors | pre_neighbors
    absent_neighbor: swallowed      # swallowed | contributes
                                    # minmax_universe: reserved (§ 9)
validate:
  max_missing_population: 15        # today's real-data count; compute raises above it
outputs:
  denominators: [pop, popdensity]   # one compute run per entry; each in {pop, popdensity}
                                    # one: reserved (no reference rule yet)
  formats: [csv, shp, joblib]
  name_template: "delhi_psi_{profile}_{denominator}_2020"
```

**Exclusion, precisely.** The neighbors artifact is always built on the full
settlement universe — `preprocess` never excludes. Exclusion happens in
`compute`, and the reference models it on two independent axes, so the config
has two keys:
- `stage: post_neighbors` — excluded rows are dropped from the compute frame
  after the artifact is loaded; their ids remain in other settlements'
  neighbor lists. `pre_neighbors` — excluded ids are also removed from every
  neighbor list before PCEN (reference `dropped_before_neighbors=True`).
- `absent_neighbor: swallowed` — a neighbor id with no row in the compute
  frame contributes nothing (today's behavior, implemented as an explicit
  lookup miss, never a bare `except`). `contributes` — DEL-21: amounts are
  looked up in the **pre-exclusion** frame passed explicitly, so excluded
  settlements still contribute their services (Eq. 3 as written); an id
  absent from that frame too is an error.
- Today's production is `post_neighbors` + `swallowed` (reference scenario
  `excl_rv_only`): `compute_psi.py` filters `USO_FINAL != "RV"` *after*
  loading the neighbors joblib and relies on `except: pass`. The rev-1 draft
  described this as a pre-neighbor drop; that was wrong.
- Rows dropped for missing population (`layers.population.missing: drop`)
  are treated exactly like `exclusion.types` rows under both keys. With
  `contributes` their service amounts still count (counts need no
  population). `missing: error` makes any missing row fatal.

**Reference-pinned keys** (1:1 with `tests/reference_impl.compute_city`):

| config key | value | reference knob |
|---|---|---|
| `adjacency.rule` | `bbox` / `touch` | `adjacency_rule="bbox"` / `"border"` |
| `barrier.rule` | `global_asymmetric` / `pairwise` | `barrier_rule="global"` / `"pair"` |
| `roads` | `decayed` / `eq4_own_only` | `roads_formula="decayed"` / `"eq4"` |
| `second_normalization` | `true` / `false` | `second_norm` |
| `exclusion.stage` | `post_neighbors` / `pre_neighbors` | `scenario` with `dropped_before_neighbors` False / True |
| `exclusion.absent_neighbor` | `swallowed` / `contributes` | `absent_neighbor_contribution` |
| `exclusion.types` | set of type strings | `scenario` dropped-id set (fixture types ↔ ids) |
| `outputs.denominators[i]` | `pop` / `popdensity` | `denom` — one reference call per entry |

The loader's enums for these keys come from one table in `config.py`; the
mapping test (§ 7) reads the same table, so a value without a reference knob
cannot be added to a reference-pinned key without failing that test.

**Production-only keys** — no reference counterpart; pinned by the per-profile
production fixture (§ 4) and by `code-2025`'s real-data baseline only:
`barrier.combine` (the oracle city has one barrier layer, so `any` and
`[canal]` coincide there), `decay` (single value; the reference hard-codes
1/(1+d) km), `layers.population.missing`, `validate.*`, `outputs.formats`,
`outputs.name_template`.

**Reserved values** — documented in the enum help text, rejected at load with
a message naming what unblocks them: `barrier.rule: partial_weighted`,
`outputs.denominators: one` (production supports it; the reference does not
— add `denom == "one"` to `compute_city` and regenerate the CSV first),
`exclusion.minmax_universe` (Open Decision A.2 — whether Eq. 2's min/max
spans reported settlements only or all; no knob in reference or production).

## 4. Config profiles and fixtures

Two fixture families, with different owners:

1. **Reference CSV — unchanged in 3A.** `tests/fixtures/oraculum/expected_values.csv`
   (rule × scenario × denom × settlement × metric, `%.17g`) stays where it is,
   stays emitted by `tests/reference_impl.py`, and stays pinned by
   `test_expected_values_csv_is_regenerable`. Every current consumer
   (`test_oracle.py`, `test_reference_impl.py`, `check_oraculum_invariants.py`,
   `render_oracle_maps.py`) keeps working untouched. A profile selects rows
   from it: `code-2025` ↔ `rule=code`, `manuscript` ↔ `rule=ideal`; the
   scenario is chosen by the `exclusion` keys (table in § 7).
2. **Production fixtures — new, one per profile.**
   `tests/fixtures/oraculum/production/<profile>.csv`: the output of
   `pipeline.compute_frames` on the oracle city for that profile, every
   scenario in § 7's table × every denominator, written at `%.17g` in a fixed
   row order by `scripts/generate_production_fixtures.py`. The CI drift guard
   already globs `scripts/generate_*_fixtures.py`, so any change to production
   numbers — refactor slip or config edit — fails CI with a per-profile diff.

Pins per profile:
- `code-2025`: production fixture **string-equal** to a snapshot taken on
  `main` before the refactor (migration step 0, § 5) — the refactor's
  correctness proof, not a tautology because the snapshot predates the
  refactor. Plus production == reference `rule=code` rows at 1e-12
  (today's `test_oracle.py`, re-expressed per § 7).
- `manuscript`: production == reference `rule=ideal` rows at 1e-12, **and**
  the `baseline`/`pop` block equals the hand-ratified anchors in
  `docs/oracle/derivation-worksheet.md` (the clinics, schools, roads and
  singleton tables) at 1e-12. No generator can rewrite the anchors.

`manuscript.yaml`, in full (`paths`/`layers`/`services`/`validate` identical
to `code-2025`):
```yaml
profile: manuscript
methodology:
  adjacency: {rule: touch}
  barrier: {rule: pairwise, combine: any}
  decay: {form: inverse_linear, distance_unit: km}   # manuscript is silent on the unit (§ 8)
  roads: eq4_own_only
  second_normalization: false                        # no norm_psi column
  exclusion: {types: [], stage: post_neighbors, absent_neighbor: contributes}
outputs: {denominators: [pop], formats: [csv], name_template: "delhi_psi_{profile}_{denominator}_2020"}
```

Adding a profile = one YAML + `generate_production_fixtures.py` + commit; the
mapping test proves the reference-pinned knobs are honored. After Raj's
decisions, flipping a default is a profile edit; the regenerated production
fixture diff *is* the methodology change, reviewable line by line.

`barrier.rule: partial_weighted` (w_ij = 1 − L_blocked/L_shared, from
`docs/oracle/suggested-fixes-memo.md` § 2) is **config-ready, reference
pending**: it needs a reference rule, a hand anchor and a worksheet update,
then it becomes a legal value. It cannot be enabled by editing YAML alone.

## 5. Migration and byte-identity proof

Order inside the `/ship`, suite green at every commit:
0. **Snapshot on `main`**: run today's `run_production_chain` for the five
   `SCENARIO_WIRING` wirings × two denominators, write
   `tests/fixtures/oraculum/production/code-2025.csv` (`%.17g`, fixed order)
   and commit it *before any production code moves*. This file is the target
   the refactored pipeline must reproduce string-for-string.
1. Package skeleton, `pyproject` build config + regenerated `uv.lock`,
   `config.py` with tests (TDD).
2. Move functions module by module (`geometry` → `neighbors` → `index`),
   leaving `spatial_index_utils.py` in place and delegating, suite green after
   each move.
3. `pipeline.py` (`compute_frames` first, then the path stages) + `cli.py`;
   `test_oracle_e2e.py` rewired to the CLI.
4. Delete `spatial_index_utils.py`, the two scripts, `common.py`,
   `conftest.py`, all `sys.path` hacks; rewire remaining scripts and tests.
5. `generate_production_fixtures.py` regenerates `code-2025.csv` — must be a
   no-op diff against the step-0 snapshot; add `manuscript.yaml` and its
   fixture; mapping and anchor tests.

**Output-column contract for `code-2025`** (what "no numbers change" means
for the real-data proof — `compare_numeric_frames` treats a missing baseline
column as a deviation): the neighbors artifact carries `nbrs_bbox`,
`nbrs_dist_bbox` (list of `(id, km)` tuples), `centroid`, `canal`, `railway`,
`drain`, `barrier`, `area_km2`, `ndmc_dist_km`; the PSI frames carry every
column the July 2025 baseline has, including `ndmc_dist_km`, `road_length`,
`unnorm_psi`, `norm_psi`. Shapefile output drops `nbrs_bbox`,
`nbrs_dist_bbox`, `centroid` as today.

Proofs, in increasing cost:
1. Oracle: `production/code-2025.csv` string-equal to the step-0 snapshot
   (drift guard, every push) and production == reference `rule=code` at
   1e-12 (`test_oracle.py`).
2. e2e: `delhi-psi preprocess && delhi-psi compute --config code-2025` on the
   Oraculum temp dir matches the `excl_rv_only` block (CI, every push).
3. Real data: `scripts/verify_against_baseline.py --config code-2025`
   reports zero deviation from the July 2025 baseline (data-gated `skipif`;
   run by hand before merge and pasted into the PR, as for DEL-26/23).

## 6. Pipeline stages, validation, CLI (DEL-25)

`delhi-psi <stage> --config <profile-or-path> [--data-dir D] [--out-dir O]`.
`--config` accepts a shipped profile name or a YAML path. Two stages:

| stage | reads | writes | validation |
|---|---|---|---|
| `preprocess` | settlements, barriers, bounds, ndmc_center | neighbors artifact (universe-wide) + dedup cache under `out_dir`, keyed on source mtime+size | layer battery: geometry type, validity, duplicates, within bounds — **raises** |
| `compute` | neighbors artifact, population CSV, service layers | one PSI output set per `outputs.denominators` entry, `missing_population.csv` | layer battery; population join: missing rows ≤ `validate.max_missing_population` else raise; post-compute: no negative `*_count/_pcen/_idx`, CRS match — **raises** |

Each stage is a function returning a small result dataclass; `cli.py` prints
it and maps exceptions to exit codes. `logging` replaces `print`; `tqdm` stays.
`io.write_outputs` wraps the shapefile write in `warnings.catch_warnings()`
filtering only geopandas' "Column names longer than 10 characters will be
truncated" `UserWarning` — accepted today, and otherwise fatal under
`pytest -W error` once stages run in-process. The O(n²)
`remove_duplicate_geom` algorithm is unchanged (not on a ticket; the oracle
cannot distinguish it) — only its cache location and staleness rule move.

## 7. Testing

- The 77 existing tests carry over. `test_oracle.py`'s five scenarios become
  `code-2025` plus `exclusion` overrides:

  | scenario | `types` | `stage` | `absent_neighbor` | reference block |
  |---|---|---|---|---|
  | `baseline` | `[]` | post | swallowed | `code/baseline` |
  | `excl_rv_only` (production default) | `[RV]` | post | swallowed | `code/excl_rv_only` |
  | `excl_contributing` | `[RV, IND]` | post | swallowed | `code/excl_contributing` |
  | `excl_removed` | `[RV, IND]` | pre | swallowed | `code/excl_removed` |
  | `excl_ind_removed` | `[IND]` | pre | swallowed | `code/excl_ind_removed` |

  `test_production_collapse_gap5` (post+swallowed collapses to pre) keeps its
  meaning: it compares the `excl_contributing` and `excl_removed` rows above.
  The metric set compared is derived from the profile — `norm_psi` only when
  `second_normalization` is true; `compute` omits the column otherwise.
- New: `test_config.py` (defaults equal `code-2025`; every enum rejects bad
  values with the key and allowed values in the message; unknown key rejected;
  each reserved value rejected with its unblock message; CLI/env/YAML
  precedence for paths), `test_profiles_match_reference.py` (for each shipped
  profile, `compute_frames` == reference at the mapped knobs, every scenario
  in the table × every denominator, 1e-12), `test_manuscript_anchors.py`
  (worksheet anchors), `test_production_fixtures.py` (regenerable,
  string-equal), `test_cli.py` (both stages on the Oraculum temp dir, csv and
  shp formats, exit codes, `--config` by name and by path), `test_validate.py`
  (each check on synthetic frames, pass and fail).
- `pytest -W error` remains the CI gate; the drift guard covers both fixture
  generators. Data-gated tests use the `skipif` convention from the CI spec.

## 8. Defaults for Raj to confirm (28 Aug 2026)

| switch | default (today) | manuscript | memo item |
|---|---|---|---|
| `adjacency.rule` | `bbox` | `touch` | suggested-fixes § 1 (DEL-19) |
| `barrier.rule` | `global_asymmetric` | `pairwise` | § 2 (DEL-22) |
| `roads` | `decayed` | `eq4_own_only` | § 3 (DEL-22) |
| `second_normalization` | `true` | `false` | § 4 (DEL-22) |
| `outputs.denominators` | `[pop, popdensity]` | `[pop]` (paper states Population only) | "Popdensity denominator" (unnumbered; DEL-22) |
| `exclusion.absent_neighbor` | `swallowed` | `contributes` | § 5 (DEL-13, DEL-21) |
| `exclusion.minmax_universe` | reported only (implicit) | — | Open Decision A.2 — ask whether to defer |
| `decay.distance_unit` | `km` | unstated in manuscript | § 7 |

Whatever he decides, the production default stays `code-2025` until Phase 4
recalculation (DEL-32) explicitly switches it; the decision is recorded as a
new or edited profile, not a code change.

## 9. Out of scope for 3A (follow-ups)

1. **`USO_FINAL` vocabulary** — measured 27 Aug 2026 on
   `uso_update_sep2021.shp` (4,357 rows, no nulls, unchanged by dedup):
   UAC 1,684 · Planned 964 · JJC 764 · RUAC 393 · RV 211 · UV 138 · SDA 86 ·
   JJR 48 · Industrial 36 · Other 33. These are the 10 types DEL-17 refers
   to; the 2021 notebooks show they came from an undocumented 16→10 merge
   (`UAC1→UAC`, `JJC1/JJC2→JJC`; `Institutional/Commercial/DCB/NDMC` folded
   or dropped). 3B's spec records this as `docs/data/uso_final_vocabulary.md`.
2. 3B: `categories:` mapping block; 10/8/5/4-category profiles.
3. 3C: messy-city tier; `adjacency.rule: touch` and overlap-sharing rules
   proven on it; oracle coverage for `barrier.combine` with several layers
   (today only canal is exercised); `partial_weighted` reference rule + hand
   anchor; `denominators: one` reference rule; `minmax_universe` if Raj asks.
4. Phase 4 `figures` stage (DEL-33) rendering paper figures from `out_dir`.
5. Phase 6: distance thresholds and decay-weight values with reference rules
   (rest of DEL-18).
6. Algorithmic replacement of O(n²) `remove_duplicate_geom`.
7. WORKPLAN housekeeping: the DEL-27 note misplaced on the Phase 4 heading;
   DEL-27 bullet still says "no `.github/workflows/` yet"; owner list cites
   "Open Decision B" for roads where the file has it under C.

## 10. Autonomy and stopping rules

Same terms as the Phase 1/2 and CI specs. Stop and report — do not work
around — if: the `code-2025` production fixture or the real-data baseline
deviate at any step; a reference-pinned key has no clean production
counterpart; a test in the carried-over 77 needs its expected value changed;
or a review loop exceeds the usual budget. Real data is read-only throughout;
outputs go under `out_dir` only.
