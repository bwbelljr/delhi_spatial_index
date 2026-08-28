# Injectable parameters: distance band and decay forms — design (DEL-18)

**Status:** draft for ultracode review, 28 Aug 2026. Completes WORKPLAN
Phase 3 item "Modular & extensible structure" [DEL-18], PARTIAL since 3A.

**Cycle:** Phase 3D. Same process and autonomy terms as 3A–3C (§ 9).

## 0. Goal and non-goals

Phase 6's robustness sweeps (DEL-36 distance thresholds, DEL-37 decay
weights, DEL-39 adjacency comparison) must be "loops over YAML profiles",
not code forks. After 3A/3B every methodological choice is a config value
EXCEPT two: there is no distance-based neighbourhood, and `1/(1+D)` is the
only decay. This cycle adds both as opt-in config *values*, each with a
reference-implementation rule and oracle pins, so a Phase 6 profile is one
YAML file and nothing else.

Non-goals, stated so reviewers can hold the line:

- **No change to the existing profiles' behaviour.** `code-2025.yaml` and
  `manuscript.yaml` do not use any new value. Their outputs, both cities'
  `expected_values.csv`, both cities' `production/*.csv`, and the real-data
  verify are byte-identical / 0.000e+00 at the end of the cycle.
- **No new shipped profile.** Phase 6 picks the sweep points and adds the
  profiles (and their production fixtures) with the reference rules already
  in place. This cycle proves the rules on *derived* profiles only.
- **No methodology decision for the paper.** Which variants are reported is
  DEL-36/37/39 + Raj.
- **No graph-distance ("neighbours of neighbours") rule.** Not what a
  "1/5/10 km" sweep describes; add later if asked.

## 1. Config surface

```yaml
methodology:
  adjacency:
    rule: within_distance        # bbox | touch | within_distance
    max_distance_km: 1.0         # required iff rule == within_distance; >= 0
                                 # (0 = strict contact: every polygon that
                                 # intersects i, incl. corner-only touches)
  decay:
    form: inverse_power          # inverse_linear | none | inverse_power | exponential
    exponent: 2                  # required iff form == inverse_power; > 0
    scale_km: 1.0                # required iff form == exponential; > 0
    distance: centroid           # centroid | boundary   (optional; default centroid)
    distance_unit: km            # km (unchanged)
```

Definitions (D = the selected distance in km between i and j):

| key | value | meaning |
|---|---|---|
| `adjacency.rule` | `within_distance` | j ∈ nbrs(i) iff `geom_i.distance(geom_j) <= max_distance_km * 1000` (polygon-to-polygon shortest distance, metres in EPSG:7760). Symmetric. Self excluded. |
| `decay.form` | `inverse_linear` | w = 1/(1+D) — today; unchanged |
| | `none` | w = 1 |
| | `inverse_power` | w = 1/(1+D)^exponent; exponent = 1 reproduces `inverse_linear` |
| | `exponential` | w = exp(−D/scale_km) |
| `decay.distance` | `centroid` | D = centroid-to-centroid, as today (the stored `nbrs_dist_bbox` column) |
| | `boundary` | D = polygon-to-polygon shortest distance; 0 for every touching/overlapping neighbour → w = 1 under every form |

Validation, all at load in `config._methodology` with `ConfigError`:

- A parameter its rule/form does not use is **rejected**, not ignored:
  `max_distance_km` without `within_distance`, `exponent` without
  `inverse_power`, `scale_km` without `exponential` → error naming the key
  and the rule that would use it. Missing when required → the existing
  `_require` error.
- `max_distance_km`: number, `>= 0`. `exponent`: number, `> 0`.
  `scale_km`: number, `> 0`. Booleans are not numbers.
- `decay.distance` defaults to `centroid` when absent. It is the only new
  key with a default, so every existing profile parses to the same
  `MethodologyConfig` values it does today (plus `distance="centroid"`).
- `decay.form` and `decay.distance` join `REFERENCE_KNOBS` / `ENUM_KEYS` as
  enums (`DecayForm`, `DecayDistance`); `adjacency.rule` gains
  `within_distance`. `test_enums_cover_exactly_the_reference_table` keeps
  the table and the enums in lock-step. The ad-hoc `form != "inverse_linear"`
  check in `_methodology` is replaced by `_coerce_enum`.

Dataclasses: `AdjacencyConfig(rule, max_distance_km: float | None = None)`;
`DecayConfig(form, distance_unit, exponent: float | None = None,
scale_km: float | None = None, distance: DecayDistance = "centroid")`.
Frozen, as today.

Shipped profiles: `code-2025.yaml` and `manuscript.yaml` gain **comments
only** — the allowed values for `adjacency.rule` and `decay.form`, and one
comment line per optional parameter saying when it is required. No key is
added to either file (a `distance: centroid` key would be harmless but is
left out so the file stays a record of what 3A wrote).

## 2. Production code

All functions stay pure and keyword-driven; nothing imports
`delhi_psi.config` below `pipeline`.

### 2.1 `delhi_psi/neighbors.py`

- `adjacency(polygon_gdf, *, id_col, neighbor_col, rule, max_distance_km=None)`.
  New branch `rule == "within_distance"` → `_adjacency_within_distance`,
  which requires `max_distance_km is not None` (ValueError otherwise) and
  uses `gpd.sjoin(polygon_gdf, polygon_gdf, how="left",
  predicate="dwithin", distance=max_distance_km * 1000)` (geopandas 1.1.4 /
  shapely 2.1.2 support `dwithin`), removes the self pair, and writes lists
  in the frame's row order like the other two rules. `max_distance_km`
  passed with `bbox`/`touch` is a ValueError (mirrors the config rule).
  The column keeps the historical name `nbrs_bbox` (docstring updated to
  "under every rule").
- `boundary_distances(polygon_gdf, *, neighbor_col, nbr_dist_col, id_col)`:
  same shape as `centroid_distances` — `[(neighbor_id, km), ...]` per row —
  with `row.geometry.distance(neighbor.geometry) / 1000`. Used only at
  compute time (§ 2.3).

### 2.2 `delhi_psi/index.py`

- `_decay(distance_km, decay_form, distance_unit, *, exponent=None,
  scale_km=None)`: dispatch over the four forms; unknown form / unit, a
  missing required parameter, or a parameter the form does not use →
  ValueError (same wording style as today). The probe call in `pcen`
  (`_decay(0.0, ...)`) keeps failing early on a city with no links.
- `pcen(...)` and `service_index(...)` gain `exponent=None, scale_km=None`
  and pass them through. No other change; `pcen` still reads
  `nbr_dist_col` and does not know which distance definition filled it.

### 2.3 `delhi_psi/pipeline.py`

- `build_neighbors` passes `max_distance_km=methodology.adjacency.max_distance_km`
  to `neighbors.adjacency`. The stored artifact is otherwise unchanged:
  `nbrs_dist_bbox` stays centroid distances under every configuration.
- `index_frames`: when `methodology.decay.distance == "boundary"`, compute
  a compute-local column `nbrs_dist_boundary` via
  `neighbors.boundary_distances` on the (post-exclusion) universe and hand
  THAT to `pcen` as `nbr_dist_col`; otherwise `nbrs_dist_bbox` as today.
  The artifact therefore stays valid across `decay.*`, preserving the stamp's
  "decay is downstream" contract. Cost: one `distance` call per stored link
  (≈ 4,357 × mean degree; seconds).
- `methodology_stamp` adds `adjacency.max_distance_km` (None for
  `bbox`/`touch`). `check_methodology_stamp` needs no change — it iterates
  the stamp. A `compute` against an artifact built with a different band is
  refused with the existing message.
- Existing artifacts (built by 3A–3C) lack the `max_distance_km` key:
  `stored.get(block, {}).get(key)` yields None, which equals the configured
  None for `bbox`/`touch`, so `code-2025`'s pinned `colonies_neighbors.joblib`
  keeps loading. Pinned by a test (§ 4.5).
- Exclusion: `apply_exclusion` strips ids from `nbrs_bbox` and
  `nbrs_dist_bbox` only; `nbrs_dist_boundary` is built after exclusion from
  the already-stripped lists, so it needs no stripping.

### 2.4 CLI / outputs

No CLI change. Output naming is unchanged (`name_template`). `preprocess`
under `within_distance` logs the band in km alongside the rule.

## 3. Reference implementation (`tests/reference_impl.py`)

Independence rule unchanged: no `delhi_psi` import.

- `adjacency(settlements, rule, max_distance_km=None)`: new rule
  `"within_distance"`: `idx[i].distance(idx[j]) <= max_distance_km * 1000`.
  (`"intersects"` stays; it is what `within_distance` at 0 equals, and the
  pin in § 4.1 uses that.)
- `compute_city(..., adjacency_rule, ..., max_distance_km=None,
  decay_form="inverse_linear", exponent=None, scale_km=None,
  decay_distance="centroid")`. `contribution_weight(i, j)` becomes:
  D = centroid km (today's expression) or `idx[i].distance(idx[j])/1000`;
  w by form as in § 1. Missing/extra parameters → ValueError (the mapped-knob
  test relies on it). Existing `RULESETS` entries are unchanged and, since
  every new keyword defaults to today's behaviour, `expected_values.csv`
  is byte-identical (checked by the drift guard and § 4.4).
- `VARIANT_RULESETS` (module constant, `code` base + overrides), each also
  a row of `tests/variants.py` (§ 4.3):

  | name | overrides | pins |
  |---|---|---|
  | `band0_none` | `within_distance` 0 km, `none` | = `intersects` neighbourhood, undecayed counts |
  | `band_small` | `within_distance` B₁ km | adds ≥ 1 non-adjacent pair on each city, ⊂ `band_large` |
  | `band_large` | `within_distance` B₂ km (B₂ > B₁) | ⊇ `band_small` |
  | `pow1` | `inverse_power` 1 | ≡ `code` exactly |
  | `pow2` | `inverse_power` 2 | 1/(1+D)² at a hand distance |
  | `exp1` | `exponential` 1 km | e^{−D} at a hand distance |
  | `boundary` | `inverse_linear`, `distance: boundary` | adjacent → w = 1 |

  B₁/B₂ are chosen by the implementer from the fixture coordinates so that
  the three band neighbourhoods are pairwise distinct on BOTH cities
  (asserted in the generator, § 4.3); the spec fixes the names, not the km.
- `emit_variant_expected_values(out_path, city)`: long-format CSV like
  `emit_expected_values` with `rule` ∈ VARIANT_RULESETS, scenario
  `baseline` only, both denominators, `%.17g`. Written to
  `tests/fixtures/<city>/variants_expected_values.csv` by the two geometry
  generators after `emit_checked_expected_values` (the invariants guard
  runs on the variants file too: `check(df, city=...)` is CSV-shape
  agnostic — the generator passes the variants frame through it).

## 4. Tests and pins

### 4.1 Hand-derivable pins (`tests/test_variant_rules.py`, reference side)

1. `within_distance` 0 ≡ `intersects` on both cities; on Oraculum ≡ `touch`
   (no corner contacts, no overlaps); on messy = `touch` ∪ {T's corner
   pairs} ∪ {O1↔O2} — T's pairs are the only difference from `touch` that
   `bbox` does not also have. Symmetry of the band on both cities.
2. Band monotonicity: nbrs(B₁) ⊆ nbrs(B₂) for every settlement; strict
   somewhere. Every adjacent (`touch`) pair is in every band (the
   large-neighbour property that motivated the boundary definition).
3. `inverse_power` 1 ≡ `inverse_linear` on every pcen (1e-12).
4. `none`: for one Oraculum settlement under the `code` neighbourhood,
   pcen = (own + Σ neighbour counts)/pop, arithmetic written in the test.
5. `pow2`, `exp1`: one Oraculum pair at the worksheet's known 1 km / 1.5 km
   centroid distances: weights 1/4, 1/6.25 and e^{−1}, e^{−1.5}; the
   contribution difference vs `code` equals (w − 1/(1+D)) × neighbour count.
6. `boundary`: one contact pair per city (any `touch` pair on Oraculum;
   `O1↔O2` on messy, where overlap gives distance 0) — boundary distance is
   0 and the weight is 1 under all four forms; plus one non-adjacent band
   pair whose boundary distance is strictly less than its centroid distance
   (on messy, `G`→`M`: inside M's envelope, not in contact).
7. Validation: each rejected combination in § 1 raises `ConfigError` with
   the offending key in the message (parametrised).

### 4.2 Production == reference on the variants (`tests/test_variants_match_reference.py`)

For each city × variant × denominator: build the variant's
`MethodologyConfig` from `tests/variants.py` (§ 4.3) via a new
`tests/oraculum_fixtures.variant_methodology(base="code-2025", variant,
city)` (derived in memory, like `methodology_with`), run
`compute_frames`, and compare every `METRIC_MAP` column to
`variants_expected_values.csv` at 1e-12. Baseline scenario only (the
exclusion machinery is proven elsewhere and is orthogonal).

Also the CLI path once: write a derived variant profile YAML
(`oracle_profile_path` extended with a `methodology_overrides` argument),
run `preprocess` + `compute` on Oraculum, and compare to the same CSV —
proves config → artifact → compute, including the stamp.

### 4.3 `tests/variants.py`

A single plumbing table, importable by BOTH sides (imports nothing from the
repo, like `tests/cities.py`): `VARIANTS = {name: {"adjacency": {...},
"decay": {...}}}` in config vocabulary. The reference builds
`VARIANT_RULESETS` from it via a tiny vocabulary map (`touch`→`border`,
etc., the same map `REFERENCE_KNOBS` encodes); production builds
`MethodologyConfig` from it. One table, so the two sides cannot drift.

### 4.4 Byte-identity and drift

- `tests/fixtures/{oraculum,messy}/expected_values.csv` and
  `production/*.csv`: byte-identical at every task (checked as in 3C).
- The two new `variants_expected_values.csv` files are generator-emitted
  and covered by the CI drift guard (same glob, same `tests/fixtures/`
  porcelain check).
- `test_every_mapped_knob_is_one_the_reference_actually_implements` gains
  the per-value required parameters (`within_distance` needs
  `max_distance_km`; `inverse_power` needs `exponent`; `exponential` needs
  `scale_km`) from a small table so the loop still drives every value.

### 4.5 Artifact stamp

- Building with `within_distance` 1 km and computing with `bbox` (or a
  different km) is refused with the existing stamp message.
- An artifact whose stamp lacks `max_distance_km` (3A–3C shape) still
  passes `check_methodology_stamp` for `bbox`/`touch` configs.
- Changing only `decay.*` does NOT invalidate an artifact (positive test).

### 4.6 Real data

`code-2025` verify PASS 0.000e+00 (warm cache; nothing on the default path
changes). Additionally, one timing note for the doc: `preprocess` on the
real layer with `within_distance` 10 km (derived profile, `--out-dir` in a
tempdir, never under the baseline) — reported as wall-clock and neighbour
count summary in `docs/methodology-config.md` § 6, **not** committed as a
fixture. Data-gated; skipped without `~/delhi_data`.

## 5. Docs

- `docs/methodology-config.md`: § 1 table gains rows for
  `adjacency.max_distance_km`, `decay.form` (+ `exponent`, `scale_km`),
  `decay.distance`; new § 6 "Phase 6 sweeps: one profile per point" with a
  complete `band-1km.yaml` example (copy of `code-2025` with the two keys
  changed), the X = 0 ≠ `touch` note, the centroid-vs-boundary note, and
  the timing note from § 4.6. § 4 lists the two new proofs.
- `CHANGELOG.md` `[Unreleased]`: the 3D entry.
- `WORKPLAN.md`: item 3 → `[x]`, DEL-18 note rewritten (what is config now:
  everything in the item's list); Phase 6 items DEL-36/37/39 note "profile
  only".
- Jira: DEL-18 → Done with the evidence comment; DEL-36/37/39 comments
  naming the keys; `Blocks` links from DEL-18 to DEL-40/41/42 removed
  (service sets were config since 3A; they were never really blocked).

## 6. Task shape (for the plan)

1. `tests/variants.py` + config: enums, dataclasses, validation, profile
   comments (tests first: § 4.1 item 7).
2. Reference: `within_distance`, decay forms, boundary distance,
   `VARIANT_RULESETS`, `emit_variant_expected_values`; hand pins § 4.1
   items 1–6 on the reference; byte-identity of `expected_values.csv`.
3. Generators emit the variants CSVs (both cities) with the distinctness
   assertions; commit fixtures.
4. Production: `neighbors`, `index`, `pipeline` (stamp, boundary column);
   § 4.2 comparison + § 4.5 stamp tests; byte-identity of production CSVs.
5. CLI derived-profile round trip; mapped-knob table; real-data verify +
   timing note (controller runs the data-gated part).
6. Docs, CHANGELOG, WORKPLAN.

## 7. Risks and open points (for reviewers)

- `sjoin(predicate="dwithin")` semantics for `distance=0` must equal
  `intersects` (shapely `dwithin(a, b, 0)` is true iff distance == 0, which
  is true iff they intersect). Pinned in § 4.1 item 1; if the join's
  handling of 0 differs, the implementation falls back to
  `predicate="intersects"` for X = 0 and the pin still decides.
- Boundary distances on MultiPolygons: shapely `distance` is min over
  parts — that is the intended meaning.
- Oraculum's centroid distances are 1 km / √2 km / 1.5 km-ish; B₁/B₂ must
  be chosen so the bands are distinct AND the non-adjacent pairs they add
  give `pcen` differences well above 1e-12 (any pair does).
- Floating-point: `w = 1/(1+D)**1` vs `1/(1+D)` are the same double for
  exponent 1.0 (`x**1.0 == x` exactly in IEEE), so `pow1 ≡ code` holds at
  0 tolerance; the pin uses 1e-12 anyway.

## 8. Definition of done

- All § 4 tests green under `uv run pytest -q -W error`; CI green.
- Both cities' `expected_values.csv` and `production/*.csv` byte-identical
  to `main` (3a61069); real-data `code-2025` verify PASS 0.000e+00.
- Variants CSVs committed and regenerable (drift guard).
- Docs/CHANGELOG/WORKPLAN/Jira per § 5.

## 9. Process and autonomy (decision log)

Same as 3A–3C: spec → ultracode review (mixed models, adversarial refuters)
→ /ship (plan → plan review ≤ 3 rounds → SDD with per-task review → final
whole-branch review → PR → merge). A CONFIRMED Critical governs over the
plan. Fix-forward, commit, push, merge: yes once CI and proofs are green.
Stop only for: a change to any existing profile's behaviour, an
expected-value change in `expected_values.csv` or `production/*.csv`, or a
write under the baseline data.
