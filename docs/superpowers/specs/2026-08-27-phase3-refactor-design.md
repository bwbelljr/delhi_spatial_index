# Phase 3 Refactor — Design Spec (DEL-15; cycle 3A in full)

Date: 2026-08-27
Status: **draft for owner review**
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
- Config profiles map 1:1 onto `tests/reference_impl.compute_city` knobs; a
  profile gets its own fixture directory (§ 4).

## Purpose

Turn the working-but-monolithic pipeline (one 822-line module + two scripts,
importable only through `sys.path` hacks) into the installable `delhi_psi`
package the WORKPLAN decided on, with every methodology choice — adjacency,
barrier rule, decay, roads formula, denominator, second normalization,
exclusion semantics — a validated config value. The refactor changes **no
numbers**: the `code-2025` profile must reproduce today's oracle fixtures
byte-for-byte and the July 2025 real-data baseline at zero deviation.
Raj's answers then land as profile edits plus a fixture regeneration, and
Phase 6 sweeps become loops over profiles.

## 1. Phase 3 decomposition

| Cycle | Scope | Tickets | Gate |
|---|---|---|---|
| **3A** (this spec) | `delhi_psi` package, config schema, profiles → fixtures, all methodology switches, pipeline stages + CLI, validation-as-assertions | DEL-16, DEL-18, DEL-21, DEL-22, DEL-25 | none — starts now |
| 3B | settlement-category mapping layer (`categories:` config block; 10/8/5/4-category runs) | DEL-17 | the `USO_FINAL` vocabulary artifact (§ 9, follow-up 1) |
| 3C | messy-city fixture tier; bbox-adjacency and overlap-sharing fixes as new switch values; multi-barrier oracle coverage | DEL-24, DEL-19, DEL-20 | Raj's decisions (DEL-13) |

DEL-16 restated: the `*_wards`/`*_buffer` variants it named were deleted under
DEL-23. The remaining duplication is the `create_service_index` /
`create_service_length_index` pair (identical apart from count-vs-length and a
swapped positional argument order) and the `road_count → road_length` special
case inside `calc_all_services`. 3A collapses both (§ 2).

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
  geometry.py      reproject, remove_duplicate_geom, bbox frame, barrier flags
  neighbors.py     adjacency(rule) + barrier(rule, combine) + centroid distances
  index.py         point_counts, road_lengths, pcen (Eq. 3), minmax (Eq. 2),
                   overall_psi (Eq. 1 + second_norm), exclusion handling —
                   pure functions with keyword knobs
  pipeline.py      preprocess(cfg) / compute(cfg) / verify(cfg) / render(cfg):
                   unpacks Config into the pure functions in today's order
  cli.py           delhi-psi {preprocess,compute,verify,render} --config X.yaml
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
  service config, so the rename special case disappears.
- `pcen_denom` and every other enum is validated at config load; the math
  functions still raise `ValueError` on an unknown value (no silent
  `UnboundLocalError` path).

Files removed: `spatial_index_utils.py` (23 functions move; no shim — every
caller is in-repo), `scripts/preprocess.py`, `scripts/compute_psi.py`,
`scripts/common.py`, root `conftest.py`, every `sys.path.insert`.
Files kept as scripts importing from `delhi_psi`: `verify_against_baseline.py`
(its comparison functions move to `pipeline.verify`; the script becomes a thin
wrapper), `generate_oraculum_fixtures.py`, `check_oraculum_invariants.py`,
`render_oracle_maps.py`.

`pyproject.toml` gains `[build-system]` (hatchling), `[project.scripts]
delhi-psi = "delhi_psi.cli:main"`, `[tool.pytest.ini_options] testpaths =
["tests"]`. `uv sync` installs the package editable; `uv.lock` records the
project as non-virtual. `tests/` gains `__init__.py` so `tests.oraculum_fixtures`
imports without rootdir tricks. `pyyaml` moves from the dev group to runtime
dependencies.

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
                id_col: uso_area_u, value_col: population}
  bounds: delhi_bounds_buffer/delhi_bounds_buffer.shp
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
methodology:
  adjacency: bbox                   # bbox | touch
  barrier:
    rule: global_asymmetric         # global_asymmetric | pairwise
                                    # (partial_weighted: reserved — rejected at load, see § 4)
    combine: any                    # any = canal | railway | drain
  decay: {form: inverse_linear, distance_unit: km}   # 1/(1+d)
  roads: decayed                    # decayed | eq4_own_only
  denominator: [pop, popdensity]    # each in {pop, popdensity, one}; one run per entry
  second_normalization: true        # norm_psi = minmax(unnorm_psi)
  exclusion:
    types: [RV]                     # raw USO_FINAL strings until 3B adds `categories:`
    semantics: removed              # removed | contributing
outputs:
  formats: [csv, shp, joblib]
  name_template: "delhi_psi_{profile}_{denominator}_2020"
```

`methodology.*` ↔ `tests/reference_impl.compute_city` knobs:

| config key | value | reference knob |
|---|---|---|
| `adjacency` | `bbox` / `touch` | `adjacency_rule="bbox"` / `"border"` |
| `barrier.rule` | `global_asymmetric` / `pairwise` | `barrier_rule="global"` / `"pair"` |
| `roads` | `decayed` / `eq4_own_only` | `roads_formula="decayed"` / `"eq4"` |
| `denominator[i]` | `pop` / `popdensity` / `one` | `denom` |
| `second_normalization` | bool | `second_norm` |
| `exclusion.semantics` | `removed` / `contributing` | `absent_neighbor_contribution="swallowed"` / `"contributes"`, with `scenario` = the `drop_pre` / `drop_post` form |
| `decay` | `inverse_linear`, km | fixed in the reference (`1/(1+d)`, km); the enum exists for Phase 6 and accepts only this value in 3A |

A `methodology` value with no reference knob cannot exist: the loader's enums
are generated from a single table in `config.py` that the mapping test (§ 6)
also reads.

`exclusion.semantics: contributing` is what DEL-21 delivers: `pcen` looks up
neighbor amounts in the **pre-exclusion** frame passed explicitly; an id that
is genuinely absent from that frame is an error, never a swallowed exception.
`removed` drops the excluded settlements before neighbors are computed, as
today's `!= "RV"` filter does.

## 4. Config profiles and fixtures

- A profile is a YAML in `delhi_psi/profiles/`. 3A ships `code-2025` and
  `manuscript`. Fixtures live per profile:
  `tests/fixtures/oraculum/<profile>/expected.csv`; the city geometry
  (`settlements/barriers/services/exhibit.geojson`) stays shared one level up
  because the city does not change with rules.
- `scripts/generate_oraculum_fixtures.py` regenerates every profile. The CI
  drift guard already globs `scripts/generate_*_fixtures.py`, so a config
  edit that changes numbers fails CI with a per-profile diff. Nothing changes
  silently.
- `manuscript` carries two independent pins: it must equal the reference
  implementation's `ideal` rule-set, **and** the hand-ratified anchors in
  `docs/oracle/derivation-worksheet.md` at 1e-12. No generator can rewrite the
  anchors.
- `code-2025` must be byte-identical to the fixtures on `main` today. That is
  the refactor's correctness proof (§ 5).
- Adding a profile = one YAML + regenerate + commit; the mapping test proves
  the knobs are honored. After Raj's decisions, flipping a default is a
  profile edit; the regenerated fixture diff *is* the methodology change,
  reviewable line by line.
- `barrier.rule: partial_weighted` (w_ij = 1 − L_blocked/L_shared, from
  `docs/oracle/suggested-fixes-memo.md` § 2) is **config-ready, reference
  pending**: the enum documents it as reserved and the loader rejects it with
  a message naming what unblocks it — a reference-impl rule, a hand anchor,
  and a worksheet update. It cannot be enabled by editing YAML alone.

## 5. Migration and byte-identity proof

Order inside the `/ship`, suite green at every commit:
1. Package skeleton, `pyproject` build config, `config.py` with tests (TDD).
2. Move functions module by module (`geometry` → `neighbors` → `index`),
   leaving `spatial_index_utils.py` in place and delegating, suite green after
   each move.
3. `pipeline.py` + `cli.py`; `test_oracle_e2e.py` rewired to the CLI.
4. Delete `spatial_index_utils.py`, the two scripts, `common.py`,
   `conftest.py`, all `sys.path` hacks; rewire remaining scripts and tests.
5. Per-profile fixtures; `manuscript` profile; mapping test.

Proofs, in increasing cost:
1. Oracle: `code-2025` fixtures byte-identical (drift guard, every push).
2. e2e: `delhi-psi preprocess && delhi-psi compute` on the Oraculum temp dir
   matches `code-2025` (CI, every push).
3. Real data: `delhi-psi verify --config code-2025` reports zero deviation
   from the July 2025 baseline (data-gated `skipif`; run by hand before merge
   and pasted into the PR, as for DEL-26/23).

## 6. Pipeline stages, validation, CLI (DEL-25)

`delhi-psi <stage> --config <profile-or-path> [--data-dir D] [--out-dir O]`.
`--config` accepts a shipped profile name or a YAML path. Stages:

| stage | reads | writes | validation |
|---|---|---|---|
| `preprocess` | settlement, barrier, bounds layers | neighbors artifact (+ dedup cache under `out_dir`, keyed on source mtime+size) | layer battery: geometry type, validity, duplicates, within bounds — **raises** |
| `compute` | neighbors artifact, population CSV, service layers | PSI outputs per denominator, `missing_population.csv` | layer battery; post-compute: no negative `*_count/_pcen/_idx`, missing-population rows ≤ `validate.max_missing_population` (default 15, today's count), CRS match — **raises** |
| `verify` | fresh outputs, baseline | report | exit 1 on any deviation (today's script) |
| `render` | fixtures | `docs/oracle/*.png` | — |

Each stage is a function returning a small result dataclass; `cli.py` prints
it and maps exceptions to exit codes. `logging` replaces `print`; `tqdm` stays.
The O(n²) `remove_duplicate_geom` algorithm is unchanged (not on a ticket; the
oracle cannot distinguish it) — only its cache location and staleness rule move.

## 7. Testing

- The 77 existing tests carry over. `test_oracle.py`'s five scenarios become
  `code-2025` plus `exclusion` overrides, so "scenario" and "profile" are one
  mechanism.
- New: `test_config.py` (defaults equal `code-2025`; every enum rejects bad
  values; unknown key rejected; `partial_weighted` rejected with the reserved
  message; CLI/env/YAML precedence for paths), `test_profiles_match_reference.py`
  (for each shipped profile, production == reference at the mapped knobs on
  Oraculum, both denominators, 1e-12), `test_cli.py` (each stage on the
  Oraculum temp dir; exit codes; `--config` by name and by path),
  `test_validate.py` (each check on synthetic frames, pass and fail).
- `pytest -W error` remains the CI gate; the drift guard covers every profile
  directory. Data-gated tests use the `skipif` convention from the CI spec.

## 8. Defaults for Raj to confirm (28 Aug 2026)

| switch | default (today) | manuscript | memo item |
|---|---|---|---|
| `adjacency` | `bbox` | `touch` | suggested-fixes § 1 (DEL-19) |
| `barrier.rule` | `global_asymmetric` | `pairwise` | § 2 (DEL-22) |
| `roads` | `decayed` | `eq4_own_only` | § 3 (DEL-22) |
| `second_normalization` | `true` | `false` | § 4 (DEL-22) |
| `exclusion.semantics` | `removed` | `contributing` | § 5 (DEL-13, DEL-21) |
| `decay.distance_unit` | `km` | unstated in manuscript | § 7 |

Whatever he decides, the production default stays `code-2025` until Phase 4
recalculation (DEL-32) explicitly switches it; the decision is recorded as a
new or edited profile, not a code change.

## 9. Out of scope for 3A (follow-ups)

1. **`USO_FINAL` vocabulary artifact** — `docs/data/uso_final_vocabulary.md`
   listing the category strings and counts in the real layer, plus the
   undocumented 16→10 merge recovered from the 2021 notebooks. Produced by a
   small data-gated script; prerequisite for 3B. (Owner follow-up, this week.)
2. 3B: `categories:` mapping block; 10/8/5/4-category profiles.
3. 3C: messy-city tier; `adjacency: touch` and overlap-sharing rules proven on
   it; oracle coverage for the canal|railway|drain combination (today only
   canal is exercised).
4. `partial_weighted` barrier reference rule + hand anchor.
5. Algorithmic replacement of O(n²) `remove_duplicate_geom`.
6. WORKPLAN housekeeping: the DEL-27 note misplaced on the Phase 4 heading;
   DEL-27 bullet still says "no `.github/workflows/` yet"; owner list cites
   "Open Decision B" for roads where the file has it under C.

## 10. Autonomy and stopping rules

Same terms as the Phase 1/2 and CI specs. Stop and report — do not work
around — if: `code-2025` fixtures or the real-data baseline deviate at any
step; a reference-impl knob has no clean production counterpart; a test in
the carried-over 77 needs its expected value changed; or a review loop
exceeds the usual budget. Real data is read-only throughout; outputs go under
`out_dir` only.
