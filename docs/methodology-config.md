# Changing the methodology — the config procedure

Since Phase 3A (27 Aug 2026) every methodology choice in the PSI pipeline
is a value in a YAML **profile**, not code. This page is the operating
procedure for turning a decision (typically one of Raj's) into a change.
The design rationale lives in
`docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` (§ 3 schema,
§ 4 fixtures, § 8 decisions); this page only tells you what to do.

## 1. The switches

Profiles live in `delhi_psi/profiles/`. Two ship:

| profile | meaning |
|---|---|
| `code-2025` | **Today's behaviour** — the rules that produced the July 2025 numbers. The production default. Treat it as a frozen record. |
| `manuscript` | The paper's ideal rules (Eq. 1–4 as written). Proven against the independent reference implementation and the hand-ratified worksheet. |

The `methodology:` block of `code-2025.yaml` lists every switch with its
allowed values as an inline comment. The ones that were on the table with
Raj, and what he decided on 28 Aug 2026
(`docs/decisions/2026-08-28-raj-methodology-decisions.md`; the "ratified"
column is what the Phase 4 profile, DEL-31, will carry):

| switch | today (`code-2025`) | paper (`manuscript`) | **ratified (28 Aug 2026)** | context |
|---|---|---|---|---|
| `adjacency.rule` | `bbox` | `touch` | **`touch`** — shared border; corner-only pairs being counted (DEL-50) | memo § 1 (DEL-19) — bbox adjacency invents neighbours. A third value, `within_distance`, is the Phase 6 distance band (§ 6, DEL-36/39) |
| `barrier.rule` | `global_asymmetric` | `pairwise` | **`partial_weighted`** — a barrier covering part of a shared boundary discounts that neighbour by the covered share, w_ij = 1 − L_blocked/L_shared, instead of severing it. Landed in cycle 3E (DEL-48) | memo § 2 (DEL-22) — `pairwise` severs the crossing pair only; `partial_weighted` generalises it |
| `barrier.buffer_m` | — (unused) | — (unused) | `5` m, pending ratification | DEL-48 — how close a barrier has to be to block a boundary point, in metres (EPSG:7760 is metric). **Required** iff `barrier.rule: partial_weighted`, **rejected** otherwise, and strictly **> 0**: `LineString.buffer(0)` is EMPTY in shapely, so a 0 m buffer would silently make every weight 1. It is a DISTANCE with round caps, so the blocked span extends `buffer_m` past each end of the barrier *where the shared boundary continues past it* — on the oracle city that makes w_AD 0.08, not the memo's buffer-free 0.1 |
| `overlap.lending` | `whole` | `whole` | **Raj to answer** — Bob's addition of 5 Sep 2026; `outside_receiver` is implemented and proven, `whole` is what both profiles carry until he confirms | DEL-20 — what a NEIGHBOUR lends. `whole` is today's rule, the neighbour's whole amount S_j. `outside_receiver` lends \|S_j \ S_i\|: the amount minus whatever of the same service already lies inside the receiver, so a service in the overlap of two colonies is not counted twice for one of them. Clean pairs are unaffected — for every pair whose polygons do not overlap, \|S_j \ S_i\| == \|S_j\| exactly |
| `roads` | `decayed` | `eq4_own_only` | **`eq4_own_only`** — each colony counts only its own roads. NB a change from the published numbers; effect measured first (DEL-49) | memo § 3 (DEL-22) — Eq. 4 has no neighbour term |
| `second_normalization` | `true` | `false` | **Raj to choose** — measured 5 Sep 2026 (DEL-52, `docs/data/psi_columns.md`): the paper's Figure 4 reports `norm_psi`, so `true` unless Raj switches the figures to Eq. 1 as written; Bob's earlier `false` recommendation withdrawn | memo § 4 (DEL-22) — `norm_psi` is not in Eq. 1 |
| `outputs.denominators` | `[pop, popdensity]` | `[pop]` | **Raj to choose** — measured 5 Sep 2026 (DEL-52): Figure 4 uses the popdensity denominator, so it stays reported unless Raj switches to per-population Eq. 3; Bob's earlier `[pop]` default withdrawn | memo "Popdensity denominator" (DEL-22) |
| `exclusion.absent_neighbor` | `swallowed` | `contributes` | **`contributes`** — dropped settlements still lend services; they get no PSI and no output row | memo § 5 / Open Decision A (DEL-13, DEL-21) |
| `exclusion.types` | `[RV]` | `[RV]` | **`[RV, Industrial, Other]`** — identity mapping kept; no category collapse (DEL-28/29) | § 2 below |
| `decay.distance_unit` | `km` | (manuscript silent) | `km` — FYI for the methods text | memo § 7 |
| `adjacency.max_distance_km` | — (unused) | — (unused) | — | DEL-36 — the band's radius in km, polygon-to-polygon. **Required** iff `adjacency.rule: within_distance`, and **rejected** otherwise; `>= 0`, where 0 means "every polygon that intersects i" (§ 6) |
| `decay.form` | `inverse_linear` | `inverse_linear` | `inverse_linear` — sweep later (DEL-37; Raj wants weights spread away from zero) | the decay weight w(D): `inverse_linear` = 1/(1+D), `none` = 1, `inverse_power` = 1/(1+D)^`exponent`, `exponential` = e^(−D/`scale_km`). `exponent` / `scale_km` are required by, and only by, their own form |
| `decay.distance` | `centroid` | `centroid` | `centroid` | DEL-37 — what D means: `centroid` (centroid-to-centroid, as every run so far) or `boundary` (polygon-to-polygon, so every touching or overlapping neighbour is at 0 and lends its services undecayed) |
| `transform.form` | `none` | `none` | **Raj to choose** — measurable since DEL-34, adopted by neither profile | DEL-34 — a function applied to a value, to spread the compressed 0–1 scores: `none`, `log1p` = log(1+x), `cbrt` = x^(1/3). Yeo-Johnson and sigmoid were excluded with reasons (a scikit-learn dependency and a fitted λ; an arbitrary scale parameter) |
| `transform.stage` | — (unused) | — (unused) | — | DEL-34 — WHERE the transform applies: `pcen` (each service's PCEN, before Eq. 2 — attacks the skew at its source) or `psi` (the composite, before the second normalisation — what the 2021 notebook did). **Required** iff `transform.form` is not `none`, **rejected** otherwise |
| `aggregation.rule` | `mean_minmax` | `mean_minmax` | **Raj to choose, with `transform.form`** — measurable since DEL-57, adopted by neither profile | DEL-57 — what Eq. 2 IS, as opposed to `transform`, which chooses a function applied to a value. `mean_minmax` min-maxes each service's PCEN then averages; `mean_rank` replaces the min-max with a percentile rank (ties averaged, rescaled so min → 0 and max → 1). See § 1.1 |

Also in the block: `exclusion.stage` (`post_neighbors` = today and
ratified: neighbours are built on the full universe, exclusion happens at
compute — this is what lets dropped settlements lend services) and
`barrier.combine`. `exclusion.types` is written in **category** names, not
raw `USO_FINAL` types; with the identity mapping the two coincide (§ 2).

Two things Raj ratified need **no switch**: the min-max in Eq. 2 runs over
the reported settlements only (today's behaviour), and a service inside
overlapping colony polygons counts directly for each of them (today's
behaviour). The other half of that second ruling — what such a service is
worth when it is *lent* — is now the switch `overlap.lending`, landed in
cycle 3E (DEL-20), and it stays at `whole` until Raj confirms: he agreed
that an overlap service counts for each owner, but the lending half is
Bob's proposal of 5 Sep 2026 and is still out for his answer. DEL-31
decides what the ratified profile carries.

**Reserved — the loader refuses these and tells you why:**
`outputs.denominators: one` (reference does not model it), and the keys
`exclusion.minmax_universe` (Open Decision A.2 was decided as today's
behaviour, so no knob is needed) and `overlap.counting` (Raj ratified on
28 Aug 2026 that a service inside k overlapping colonies counts for each of
the k — today's behaviour on both sides — so there is nothing to choose;
the key exists to tell the next reader why only ONE half of memo § 6 is a
switch). A reserved value is a cycle-3x ticket, not a YAML edit.
`barrier.rule: partial_weighted` was reserved until cycle 3E (DEL-48)
supplied the reference rule, the anchors and the production implementation;
it is now a loadable value, and it is what Raj chose.

### 1.1 What changes under `aggregation.rule: mean_rank`

Two consequences a reader cannot infer from the enum, both of which the
oracle pins.

**A constant column stops being an error.** `mean_minmax` *raises* when a
service's PCEN is the same for every reported settlement — Eq. 2 divides
0/0, and the DEL-54 guard refuses to invent a value. Under `mean_rank`
every settlement ties, every average rank is equal, and the result is
**0.5 for all of them**: defined, and a uniform shift that changes no
ordering. So a service nobody owns, or one everybody owns equally, halts a
`mean_minmax` run and passes silently under `mean_rank`. That is a real
difference in what the pipeline accepts, not a detail. (`mean_rank` has its
own refusal in the same spirit: `n == 1`, where `(rank − 1) / (n − 1)` is
the same 0/0 in different clothing.)

**A `pcen`-stage transform becomes an exact no-op.** `log1p` and `cbrt` are
strictly monotone and injective, so they preserve the ordering *and* the tie
structure, so the average ranks cannot move. Not approximately — the variant
pair `aggregation_mean_rank` and `aggregation_mean_rank_log1p_pcen` is
asserted **equal** on every `*_idx` column, on both fixture cities and both
denominators. (The `*_pcen` columns do differ, and must: a `pcen`-stage
transform replaces the reported PCEN by design.)

A `psi`-stage transform still bites under `mean_rank`, because it acts on
the composite — a mean of ranks, not a rank.

The practical reading: **`transform` and `aggregation` are alternative
answers to the same complaint**, not companions. Both exist because the
compressed 0–1 index puts 452 of 4,131 settlements at exactly zero; adopting
a rank aggregation would make the transform knob moot for the composite.
They should be decided together.

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

### Worked example 2 — Delhi, ten types into the workshop's five (parked)

**Raj decided on 28 Aug 2026 not to collapse the types for now** (DEL-29
parked; revisit if reviewers push back). The Phase 4 profile keeps the
identity `uso-10` mapping and drops three types:

```yaml
categories:
  scheme: uso-10            # identity, as in code-2025
  mapping: {Planned: Planned, UAC: UAC, JJC: JJC, RUAC: RUAC, RV: RV,
            UV: UV, SDA: SDA, JJR: JJR, Industrial: Industrial, Other: Other}
methodology:
  exclusion:
    types: [RV, Industrial, Other]   # the 28 Aug 2026 decision; UV and SDA stay in
```

The example below is the recipe **if** a reviewer asks for the collapse.
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

### Procedure for a category collapse (if it is ever asked for)

DEL-31 no longer needs this — with the identity mapping the ratified
profile's category change is the single `exclusion.types` line above, made
alongside the other 28 Aug decisions by the § 3 procedure. Kept for the
day a reviewer asks for the collapse:

1. Copy `delhi_psi/profiles/code-2025.yaml` to
   `delhi_psi/profiles/urban-5.yaml`, set `profile: urban-5`, and write the
   agreed `categories:` block — every one of the 10 source types mapped.
2. Set `methodology.exclusion.types: [non-urban]` (category names).
3. **Do not add `urban-5` to `tests/test_cli.SHIPPED_PROFILES` or
   `scripts/generate_production_fixtures.PROFILES`.** `compute_oracle_frame`
   always resolves `oracle_config(profile)` — the derived helpers swap in
   the oracle-6 identity mapping but keep `base`'s shipped
   `methodology.exclusion.types` verbatim — and `emit_profile` supplies
   `types` from `ORACLE_SCENARIOS`; `urban-5`'s `exclusion.types:
   [non-urban]` is not a category the oracle-6 identity produces (its
   vocabulary is `Planned, UC, JJC, RV, RUAC, IND`), so `oracle_config`/
   `oracle_profile_path` fail to load it. Even for a profile whose
   exclusion categories *did* exist in that vocabulary, the regenerated
   `urban-5.csv` would still be byte-identical to `code-2025.csv` — the
   production fixture pins METHODOLOGY only and is EXPECTED to be
   unchanged by a mapping change, never the categorization decision.
   Instead, prove the vocabulary claim with a `collapse_profile_path`-style
   test in `tests/test_cli.py`: copy the pattern of
   `test_five_way_collapse_reproduces_raw_type_exclusion`'s hand-written
   `ORACLE_5` block, adapted to `urban-5`'s mapping and dropped category —
   a new profile's mapping gets a sibling test, not a fixture diff.
4. Run the suite, then the real data (§ 3 step 5) — `--config urban-5`
   resolves the shipped YAML directly and needs none of the § 3 step 2
   test-list registrations above. This is the DEL-32 recalculation; no
   code changes.

## 3. Procedure for a decision

Work on a branch; `main` requires the CI check.

1. **Create a new profile rather than editing `code-2025`.** Copy
   `delhi_psi/profiles/code-2025.yaml` to e.g.
   `delhi_psi/profiles/ratified-2026.yaml`, set `profile: ratified-2026`,
   and change the decided values. `methodology:` must be complete (every
   key written out); everything else may be omitted and takes the
   `code-2025` defaults — with one exception: `paths.neighbors_artifact`
   defaults to `colonies_neighbors_<profile>.joblib`, whereas `code-2025`
   pins the legacy name `colonies_neighbors.joblib` (so the real-data proof's
   filename never changed). Leave the key out of a new profile and you get
   the per-profile name, which is what you want. Do **not** add a literal
   `paths.out_dir` — it would ignore `--data-dir`.
2. **Register it.** Add the name to `PROFILES` in
   `scripts/generate_production_fixtures.py` and
   `tests/test_production_fixtures.py`. If it should be checked against the
   reference implementation, add it to `PROFILE_RULES` in
   `tests/test_profiles_match_reference.py` mapped to `"code"` or `"ideal"`
   — only if *all* its reference-pinned switches match that rule-set;
   a mixed profile is pinned by its production fixture alone.
   Add the name to `SHIPPED` in `tests/test_config.py` as well — it asserts
   the exact set of shipped YAMLs, and a new profile turns it red.
3. **Regenerate the fixtures:** `uv run python scripts/generate_production_fixtures.py`.
   A new file `tests/fixtures/oraculum/production/<profile>.csv` appears.
   The diff against `code-2025.csv` *is* the methodology change on the
   7-settlement oracle city — read it; it is the discussion artefact.
   `code-2025.csv` must not change.
4. **Run the suite:** `uv run pytest -q -W error`. Commit the YAML, the
   fixture and the test edits together.
5. **Real data** (long; run in the background; outputs only under `--out-dir`):
   ```
   uv run delhi-psi preprocess --config ratified-2026 --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_ratified
   uv run delhi-psi compute    --config ratified-2026 --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_ratified
   ```
   Both stages must use the same profile: the neighbours artifact is
   stamped with the methodology that built it and `compute` refuses a
   mismatch ("re-run preprocess"). A new profile gets its own artifact
   name automatically (`colonies_neighbors_<profile>.joblib`; `code-2025`
   alone keeps the legacy `colonies_neighbors.joblib`).
   **Outputs** land directly in `--out-dir`: one set per entry of
   `outputs.denominators`, named by `outputs.name_template` (e.g.
   `delhi_psi_ratified-2026_pop_2020.csv`, plus `.shp`/`.joblib` if listed
   in `outputs.formats`), a `missing_population.csv`, the neighbours
   artifact, and the `*.dedup.gpkg` cache files.
   `scripts/verify_against_baseline.py` compares to July 2025 and is only
   meaningful for `code-2025`; for a new profile the comparison of interest
   is against the `code-2025` outputs in a separate directory.
6. **Record it:** WORKPLAN Open Decisions + the DEL-13 ticket; the
   production default stays `code-2025` until Phase 4 recalculation
   (DEL-32) switches it deliberately.

## 4. What each proof guards

- The first two proofs below run on **both** fixture cities
  (`tests/cities.py`): **oraculum**, the small hand-ratified one, and
  **messy**, which carries the real layer's pathologies
  (`docs/oracle/messy-city.md`). The rest are scoped to a single city, or to
  real data, as noted.
- `tests/test_production_fixtures.py` — every profile's numbers on each
  fixture city, byte-for-byte; the CI drift guard regenerates and diffs them
  on every push, so an accidental edit cannot pass.
- `tests/test_profiles_match_reference.py` — production == the independent
  reference implementation at 1e-12 for the profiles in `PROFILE_RULES`, on
  every city × scenario × denominator.
- `tests/test_variant_rules.py` — **both** cities: the hand-derivable pins for
  the injectable parameters on the REFERENCE side — a 0 km band is exactly
  the `intersects` neighbourhood (and on the messy city that is `touch` plus
  the corner-only `L`/`T` pair), the bands are strictly nested,
  `inverse_power` 1 reproduces `inverse_linear` on every PCEN, and `pow2` /
  `exp1` / `none` / `boundary` are pinned at closed-form arithmetic written
  out in the test, and **`partial_5m`** pins the fractional barrier weight —
  w_AD = 0.08 on the only partly-blocked edge either fixture city has, its
  symmetry (w_AD == w_DA bit for bit), and the comparison against a 1 m
  buffer (0.096) that shows the buffer is a distance and not a flag, and
  **`overlap_outside`** pins the lending rule on the messy city's one
  overlapping pair (O1's clinic PCEN falls from (1 + 1/1.8)/600 to 1/600
  while its school PCEN does not move at all, because O2's school is outside
  the overlap), plus the statement that the shared structure is EMPTY on
  Oraculum, which is why the rule is degenerate there. Plus the
  loader's rejection table: a parameter its form or rule does not use is an
  error, never an ignored value.
- `tests/test_variants_match_reference.py` — **both** cities: production ==
  the independent reference at 1e-12 on **eleven** derived variants × both
  denominators (`tests/fixtures/<city>/variants_expected_values.csv`,
  generator-emitted and covered by the CI drift guard), plus a CLI round trip
  through a derived variant profile YAML — config file → stamped artifact →
  `compute` → CSV — for a band, for `partial_5m`, and for `overlap_outside`,
  and the stamp refusing an artifact built at another band OR at another
  `buffer_m`.
- `tests/test_reference_impl.py` — the synthetic production-vs-reference
  city: a fractional barrier weight on an overlapping pair with a
  MultiPolygon neighbour, and a second variant of the same city with a
  service point inside the overlap, so both 3E multipliers act on one pair —
  scored by BOTH implementations at 1e-12. Neither can live in a fixture
  city without moving an existing expected value, so they live in in-test
  geometry.
- `tests/test_messy_fixtures.py` — **messy**-only: what production does on
  each real-layer pathology today (bbox-invented neighbours, the overlap
  double count, the no-population drop). A pin here flips when DEL-19 lands;
  that is the point. DEL-20's two halves now sit side by side there — the
  ratified counting pin, which does not move, and the lending pin under the
  `overlap_outside` methodology.
- `tests/test_manuscript_anchors.py` — **oraculum**-only by design (the messy
  city has no hand anchors): `manuscript` == the hand-ratified worksheet
  (`docs/oracle/derivation-worksheet.md`).
- `scripts/verify_against_baseline.py --config code-2025` — real data ==
  July 2025 baseline, zero deviation.

## 5. Things that are code, not config

New *values* for a reference-pinned switch (anything not listed in the YAML
comments) require, in order — this is the 3C cycle:
1. add `{new_value: reference_knob}` to `REFERENCE_KNOBS` in
   `delhi_psi/config.py` — the enums the YAML loader accepts are generated
   from that one table, so nothing else makes the value loadable;
2. implement the matching branch in `tests/reference_impl.py::compute_city`
   (`test_every_mapped_knob_is_one_the_reference_actually_implements`
   fails until you do);
3. a hand anchor in `docs/oracle/derivation-worksheet.md`;
4. regenerate `tests/fixtures/oraculum/expected_values.csv`
   (`uv run python tests/reference_impl.py`);
5. the production implementation in `delhi_psi/` if the branch does not
   already exist.

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
profile renamed (and `paths.neighbors_artifact` left out so the artifact
takes its default per-profile name instead of overwriting `code-2025`'s).
No methodology value moves, which is what makes the diff against
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

A wide band multiplies the link count roughly with the swept area, so a
10 km band has on the order of two orders of magnitude more directed links
than `code-2025`'s `bbox` rule, whose mean degree on the real settlement
layer (4,357 settlements) is about 7 pre-barrier (4.9 after the barrier
rule severs links). The `dwithin` spatial join itself is a
single vectorized query and stays fast at any radius; the cost is
downstream, in `apply_barrier` and `centroid_distances`, which are per-link
Python loops. Budget tens of minutes for a `preprocess` at a 10 km band —
run it into a scratch `--out-dir`, never under the baseline data, and skip
`compute` for a timing check. Each band needs its own `preprocess`: the
neighbours artifact is stamped with `adjacency.max_distance_km`, and
`compute` refuses an artifact built at a different radius.

Measured on 28 Aug 2026 at commit `afec696` (`code-2025` with
`adjacency: {rule: within_distance, max_distance_km: 10.0}`, scratch
`--out-dir`, cold dedup cache, no `compute`): `preprocess` took
**2,491 s (41.5 min)** wall-clock and produced **4,366,055 directed links**
over 4,357 settlements — mean degree 1,002, maximum 1,779, and **no**
isolated settlement (the `bbox` rule leaves 6, `touch` 20; see
`docs/data/layer_pathologies.md`). The stamp on the artifact read
`adjacency: {rule: within_distance, max_distance_km: 10.0}`. About a
quarter of that time is the one-off settlement dedup that a warm cache
skips; the rest is the per-link loops, so expect the cost to scale with
the link count, i.e. roughly with the square of the radius.
