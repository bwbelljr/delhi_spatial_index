# Injectable parameters: distance band and decay forms — design (DEL-18)

**Status:** rev 2 after ultracode review (16 confirmed → 8 root causes
fixed), 28 Aug 2026. Approved for /ship by the owner's standing instruction. Completes WORKPLAN
Phase 3 item "Modular & extensible structure" [DEL-18], PARTIAL since 3A.

**Cycle:** Phase 3D. Same process and autonomy terms as 3A–3C (§ 9).

## 0. Goal and non-goals

Phase 6's robustness sweeps (DEL-36 distance thresholds, DEL-37 decay
weights, DEL-39 adjacency comparison) must be "loops over YAML profiles",
not code forks. After 3A/3B every methodological choice is a config value
EXCEPT two: there is no distance-based neighbourhood, and `1/(1+D)` is the
only decay. This cycle adds both — distance thresholds and decay weights,
the two items WORKPLAN marks open — as opt-in config *values*, each with a
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
                                 # (0 = intersection-inclusive: every polygon that
                                 # intersects i, incl. corner-only touches and overlaps)
  decay:
    form: inverse_power          # inverse_linear | none | inverse_power | exponential
    exponent: 2                  # required iff form == inverse_power; > 0
    scale_km: 1.0                # required iff form == exponential; > 0
    distance: centroid           # centroid | boundary   (required, like every methodology key)
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
- `decay.distance` is **required** — the repo's rule is "methodology has
  no defaults, never inherited" (`config.py` module docstring), and this
  cycle does not add an exception. Consequently `code-2025.yaml` and
  `manuscript.yaml` each gain ONE key, `distance: centroid`, under `decay`
  (with a comment saying it names today's definition). Behaviour is
  unchanged: the value selects the centroid column that has always been
  used. The 3A "record of what was written" is preserved by the comment,
  not by omitting the key.
- `decay.form` and `decay.distance` join `REFERENCE_KNOBS` / `ENUM_KEYS` as
  enums (`DecayForm`, `DecayDistance`); `adjacency.rule` gains
  `within_distance`. `test_enums_cover_exactly_the_reference_table` keeps
  the table and the enums in lock-step. The ad-hoc `form != "inverse_linear"`
  check in `_methodology` is replaced by `_coerce_enum`.

Dataclasses: `AdjacencyConfig(rule: AdjacencyRule, max_distance_km: float |
None = None)`; `DecayConfig(form: DecayForm, distance_unit: str,
distance: DecayDistance, exponent: float | None = None, scale_km: float |
None = None)`. Frozen, as today; the `None` defaults exist only because
the dataclass must hold "not applicable" for the parameters a form/rule
does not use — the YAML key itself is never defaulted.

Shipped profiles: `code-2025.yaml` and `manuscript.yaml` gain the
`decay.distance: centroid` key (above) and **comments** — the allowed
values for `adjacency.rule` and `decay.form`, and one comment line per
conditional parameter saying when it is required. Nothing else changes in
either file; both cities' production CSVs and the real-data verify prove
the outputs are unchanged.

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
  in the frame's row order like the other two rules. `max_distance_km is
  not None` together with `bbox`/`touch` is a ValueError (mirrors the
  config rule; `build_neighbors` forwards the config value unconditionally,
  which is None for those rules).
  The column keeps the historical name `nbrs_bbox` (docstring updated to
  "under every rule").
- `boundary_distances(polygon_gdf, *, neighbor_col, nbr_dist_col, id_col)`:
  same OUTPUT shape as `centroid_distances` — `[(neighbor_id, km), ...]`
  per row — computed as `geoms[i].distance(geoms[j]) / 1000` over an
  id→geometry dict built ONCE (the `_adjacency_touch` pattern,
  `neighbors.py:69`), never the per-neighbour boolean-mask lookup
  `centroid_distances` inherited from the 2025 script. Used only at
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
- `index_frames` passes `exponent=methodology.decay.exponent` and
  `scale_km=methodology.decay.scale_km` to `index.service_index` alongside
  `decay_form` / `distance_unit` (today only the latter two are wired).
- `index_frames`: when `methodology.decay.distance == "boundary"`, compute
  a compute-local column `nbrs_dist_boundary` via
  `neighbors.boundary_distances` on the (post-exclusion) universe, hand
  THAT to `pcen` as `nbr_dist_col`, and **drop the column before
  returning** (so `io.SHAPEFILE_DROP_COLUMNS` and the CSV/shapefile column
  contract are untouched); otherwise `nbrs_dist_bbox` as today. The
  artifact therefore stays valid across `decay.*`, preserving the stamp's
  "decay is downstream" contract. Cost with the dict-based implementation:
  one shapely `distance` per stored link (≈ 4,357 × mean degree ≈ 3×10⁴
  calls; seconds).
- `build_neighbors` logs one INFO line `adjacency: rule=%s band_km=%s`
  before calling `neighbors.adjacency` (the "logs the band" of § 2.4).
- `methodology_stamp` adds `adjacency.max_distance_km` (None for
  `bbox`/`touch`). `check_methodology_stamp` needs no change — it iterates
  the stamp. A `compute` against an artifact built with a different band is
  refused with the existing message. The literal expected dict in
  `tests/test_cli.py::test_neighbors_artifact_carries_the_methodology_stamp`
  gains `"max_distance_km": None` (task 4).
- Existing artifacts (built by 3A–3C) lack the `max_distance_km` key:
  `stored.get(block, {}).get(key)` yields None, which equals the configured
  None for `bbox`/`touch`, so `code-2025`'s pinned `colonies_neighbors.joblib`
  keeps loading. Pinned by a test (§ 4.5).
- Exclusion: `apply_exclusion` strips ids from `nbrs_bbox` and
  `nbrs_dist_bbox` only; `nbrs_dist_boundary` is built after exclusion from
  the already-stripped lists, so it needs no stripping.

### 2.4 CLI / outputs

No CLI change. Output naming and the output column set are unchanged
(`name_template`; the compute-local boundary column never reaches
`io`). `preprocess` under `within_distance` logs the band (§ 2.3).

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
- `VARIANT_RULESETS` (module constant, `code` base + overrides), built
  from the rows of `tests/variants.py` (§ 4.3). B₁ = **0.25 km**, B₂ =
  **0.75 km** — fixed by the spec, chosen from the fixtures' polygon-to-
  polygon distances (Oraculum non-zero set {0.1, 0.5, 1.0, 1.16619, 1.5,
  2.0} km; messy {0.131519, 0.199, 0.223607, 0.45, 0.630242, 0.8, 1.0, 1.8,
  …} km) so each radius sits in a gap of both lists (≥ 26 m clearance; no
  `<=` tie). Undirected pair counts: Oraculum 10 / 12 / 14 at 0 / B₁ / B₂
  (B₁ adds A–RV, C–RV at 0.100 km; B₂ adds B–D, B–IND at 0.500 km); messy
  5 / 8 / 10 (B₁ adds H–L 0.131519, L–S 0.199, H–T 0.223607 km; B₂ adds
  M–G 0.450, T–S 0.630242 km). Never use 1.0 km as a radius: three fixture
  pairs sit at exactly 1.000000 km.

  | name | overrides | pins |
  |---|---|---|
  | `band0_none` | `within_distance` 0 km, `none` | = `intersects` neighbourhood, undecayed counts |
  | `band_small` | `within_distance` 0.25 km | adds the pairs listed above; ⊂ `band_large` |
  | `band_large` | `within_distance` 0.75 km | ⊇ `band_small`, strictly |
  | `band_small_boundary` | `within_distance` 0.25 km, `inverse_linear`, `distance: boundary` | the only variant where Oraculum discriminates boundary from centroid: A↔RV, C↔RV at boundary 0.100 km vs centroid 1.414214 km (w = 1/1.1 vs 1/2.414214) |
  | `pow1` | `inverse_power` 1 | ≡ `code` exactly |
  | `pow2` | `inverse_power` 2 | 1/(1+D)² at a hand distance |
  | `exp1` | `exponential` 1 km | e^{−D} at a hand distance |
  | `boundary` | `inverse_linear`, `distance: boundary` | adjacent → w = 1. On Oraculum every `code` neighbour is in contact, so this variant is degenerate there (every weight 1); messy (`G↔M`, 0.45 km) carries its proof, `band_small_boundary` carries Oraculum's |

  The generator asserts (§ 6 step 3) that the three band neighbourhoods
  are pairwise distinct on BOTH cities and that the pair counts above hold.
- `emit_variant_expected_values(out_path, city)`: long-format CSV like
  `emit_expected_values` with `rule` ∈ VARIANT_RULESETS, ONE scenario per
  city — `city.scenarios[0]` (Oraculum `baseline`; messy `nopop_only` —
  the messy city has no scenario literally named `baseline`, because `U`
  is dropped by every scenario) — both denominators, `%.17g`. The scenario
  name is written into the CSV's `scenario` column as today. Written to
  `tests/fixtures/<city>/variants_expected_values.csv` by the two geometry
  generators after `emit_checked_expected_values` (the invariants guard
  runs on the variants file too: `check(df, city=...)` is CSV-shape
  agnostic — the generator passes the variants frame through it).

## 4. Tests and pins

### 4.1 Hand-derivable pins (`tests/test_variant_rules.py`, reference side)

1. `within_distance` 0 ≡ `intersects` on both cities; on Oraculum ≡ `touch`
   ≡ undirected `bbox` (10 pairs; no corner contacts, no overlaps). On
   messy `within_distance` 0 = `touch` ∪ {L↔T} exactly — the corner-only
   contact is the ONLY pair `touch` lacks (its intersection is a Point,
   length 0). O1↔O2 is already a `touch` pair: an overlap's intersection is
   a Polygon whose `.length` is its perimeter (2,400 m), so
   `shared.length > 0` accepts it. Both L↔T and O1↔O2 are also in `bbox`.
   Symmetry of the band on both cities.
2. Band monotonicity: nbrs(B₁) ⊆ nbrs(B₂) for every settlement; strict
   somewhere. Every adjacent (`touch`) pair is in every band (the
   large-neighbour property that motivated the boundary definition).
3. `inverse_power` 1 ≡ `inverse_linear` on every pcen (1e-12).
4. `none`: for one Oraculum settlement under the `code` neighbourhood,
   pcen = (own + Σ neighbour counts)/pop, arithmetic written in the test.
5. `pow2`, `exp1`: the two Oraculum settlements with exactly one `code`
   neighbour — `RV` (D = 1 km) and `D` (D = 1.5 km) — so the pin is a
   closed form: weights 1/4 and 1/6.25 (`pow2`), e^{−1} and e^{−1.5}
   (`exp1`); pcen = (own + w × neighbour count)/pop, written out in the test.
6. `boundary`: (a) one contact pair per city (any `touch` pair on
   Oraculum; `O1↔O2` on messy, where overlap gives distance 0) — boundary
   distance 0, weight 1 under all four forms; (b) the large-neighbour case:
   `H↔L` on messy (boundary 0.131519 km < centroid 1.127237 km; in
   `band_small`) and `A↔RV` on Oraculum (0.100 km < 1.414214 km) — boundary
   weight strictly greater than centroid weight; (c) the OPPOSITE
   pathology, `G↔M` on messy: M's centroid falls in the gap between its
   parts and coincides with G's, so centroid distance is exactly 0 (w = 1
   under `inverse_linear`) while boundary distance is 0.45 km (w = 1/1.45)
   — the one messy pair where boundary > centroid, and the clearest
   argument that centroid distance can misstate proximity in both
   directions. `G↔M` enters the band at B₂.
7. Validation: each rejected combination in § 1 raises `ConfigError` with
   the offending key in the message (parametrised).

Where an item says "on messy" and the variant needs a scenario, it is
`nopop_only` (§ 3); `U` never appears in a messy pin.

### 4.2 Production == reference on the variants (`tests/test_variants_match_reference.py`)

For each city × variant × denominator: build the variant's
`MethodologyConfig` from `tests/variants.py` (§ 4.3) via a new
`tests/oraculum_fixtures.variant_methodology(base="code-2025", variant,
city)` (derived in memory, like `methodology_with`), run
`compute_frames`, and compare every `METRIC_MAP` column to
`variants_expected_values.csv` at 1e-12. One scenario per city
(`city.scenarios[0]`, § 3) — the exclusion machinery is proven elsewhere
and is orthogonal.

Also the CLI path once: write a derived variant profile YAML
(`oracle_profile_path` extended with a `methodology_overrides` mapping
whose top-level sub-blocks — `adjacency`, `decay` — REPLACE
`raw["methodology"][<block>]` wholesale, so a variant always states its
full block and no key is inherited),
run `preprocess` + `compute` on Oraculum, and compare to the same CSV —
proves config → artifact → compute, including the stamp.

### 4.3 `tests/variants.py`

A single plumbing table, importable by BOTH sides (imports nothing from the
repo, like `tests/cities.py`): `VARIANTS = {name: {"adjacency": {...},
"decay": {...}}}` in config vocabulary. Every value the table uses
(`within_distance`, the four decay forms, `centroid`/`boundary`) has the
SAME name in the reference's keyword vocabulary, so `reference_impl`
builds `VARIANT_RULESETS` from it with NO translation layer — a variant
that one day needs `touch`/`bbox` will add a translation then, guarded
like `REFERENCE_KNOBS`. A pin asserts every table value is a member of the
matching `REFERENCE_KNOBS` entry, so the table cannot name a value the
config would reject. One table, so the two sides cannot drift.

### 4.4 Byte-identity and drift

- `tests/fixtures/{oraculum,messy}/expected_values.csv` and
  `production/*.csv`: byte-identical at every task (checked as in 3C).
- The two new `variants_expected_values.csv` files are generator-emitted
  and covered by the CI drift guard (same glob, same `tests/fixtures/`
  porcelain check).
- `test_every_mapped_knob_is_one_the_reference_actually_implements`:
  `knob_for_key` gains `"methodology.decay.form": "decay_form"` and
  `"methodology.decay.distance": "decay_distance"` (no extra kwarg for
  `boundary`), and a table `EXTRA_PARAMS = {("methodology.adjacency.rule",
  "within_distance"): {"max_distance_km": 0.25}, ("methodology.decay.form",
  "inverse_power"): {"exponent": 2}, ("methodology.decay.form",
  "exponential"): {"scale_km": 1.0}}` is merged via
  `kwargs.update(EXTRA_PARAMS.get((key, config_value), {}))` before each
  `compute_city` call — the same constants the § 4.1 pins use, never fresh
  numbers.

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
tempdir, never under the baseline; NO `compute` at 10 km) — reported as
wall-clock and neighbour-count summary in `docs/methodology-config.md`
§ 6, **not** committed as a fixture. Budget tens of minutes: the `dwithin`
join is fast, but `apply_barrier` and `centroid_distances` are per-link
Python loops and a 10 km band has ~two orders of magnitude more links than
adjacency. Run in the background by the controller; data-gated.

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

- `sjoin(predicate="dwithin", distance=0)` equals `intersects`: measured
  at the pinned versions (geopandas 1.1.4 / shapely 2.1.2) on both cities,
  directed pair counts identical to brute force at 0 / 0.25 / 0.75 / 1.0
  km. Pinned in § 4.1 item 1.
- Boundary distances on MultiPolygons: shapely `distance` is min over
  parts — that is the intended meaning.
- The band is polygon-to-polygon distance, so band radii must be judged
  against boundary distances, NOT the worksheet's centroid distances
  (1 / √2 / 1.5 km): at 1.0 / 1.5 km the two bands would be identical on
  messy. § 3 fixes 0.25 / 0.75 km for that reason.
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
