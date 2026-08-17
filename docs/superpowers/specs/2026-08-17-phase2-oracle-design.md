# Phase 2: The Mythical-City Oracle — Design Spec

Date: 2026-08-17 (rev 4 — three ultracode review rounds, 29 findings applied)
Status: **approved by owner (2026-08-17)** — /ship authorized

## Decision log (autonomous-run authorizations, per /ship — same terms as Phase 1)

- **Autonomy scope**: fix forward, commit, push, and merge the PR — all
  authorized without mid-run check-ins.
- **Plan-vs-reviewer conflicts**: a CONFIRMED Critical finding governs over
  the implementation plan; deviations recorded in CHANGELOG.md and the PR.
- **Failure policy**: fix and retry to done. Red lines that halt for the
  owner: methodology changes (index equations, exclusion semantics,
  oracle expected values beyond what this spec derives) and writes to
  baseline data (~/delhi_data outside temp/scratch paths). Per the spec's
  own rule: if empirical pinning contradicts the directed neighbor table,
  STOP and update the spec before proceeding.
- **Smoke**: `uv run pytest`.
Branch: `phase2-oracle` (off `origin/main`)
Parent plan: `WORKPLAN.md` Phase 2

## Purpose

Build a hand-verifiable ground truth for the PSI: a tiny fictional city
("Oraculum") whose index can be computed with a calculator from the
manuscript's equations (Eq. 1–4 of "Making the City Unequal"), encoded as a
permanent pytest suite. After this phase, any code change that alters the
index math fails loudly, and formula/semantics experiments (Phases 3–6) have
a safe sandbox.

**Oracle contract (decided in meta-planning, restated):** the manuscript's
equations are truth. Expected values are hand-derived from them; production
code must reproduce those values. On mismatch, default is to fix the code; a
deliberate deviation may instead be ratified with Raj, updating the methods
text and the derived values. End invariant: manuscript, hand calculation,
and code agree — no silent deviations.

**Three-way agreement architecture (Approach C, approved):**
production code == independent reference implementation == human calculator.

## The two rule-sets (central concept)

The ultracode review of this spec, running the production code against the
draft geometry, established that the code deviates from the manuscript in
FOUR distinct places. The oracle therefore computes everything under two
named rule-sets, and every artifact (expected values, worksheet, memo)
labels which one it is using:

| # | Aspect | `ideal` rule-set (manuscript-literal) | `code` rule-set (empirically pinned) |
|---|--------|---------------------------------------|--------------------------------------|
| 1 | Adjacency | polygons sharing a border (edge); symmetric | **directed**: nbrs(i) = { j : geom_i intersects bbox_j } via sjoin `intersects` — asymmetric for non-rectangular shapes (verified: exhibit's P gets no neighbors while Q gets one). Directed and symmetric readings coincide for all-rectangle universes, so the main city is unaffected |
| 2 | Barrier | severs only the crossing pair (A–D), both directions | **global and asymmetric**: `remove_ids_with_barrier` deletes every barrier-flagged settlement from OTHER settlements' neighbor lists, while flagged settlements keep their own lists (verified against `barrier_intersection` + `add_polygon_neighbors_column_fast`) |
| 3 | Roads | Eq. 4 literally: own length / population, min-maxed, NO neighbor term | `create_service_length_index` applies Eq. 3-style neighbor decay to road lengths |
| 4 | PSI | Eq. 1: PSI = mean of service indices (the code's `unnorm_psi`) | adds a second min-max pass producing `norm_psi`, absent from Eq. 1 |
| 5 | Exclusion semantics | dropped settlements may still contribute services as neighbors (semantics a is realizable) | a bare `except: pass` in `calc_pcen_mobile` silently drops contributions from any neighbor id absent from the frame — so semantics (a) **degenerates to (b)**: `code`/`excl_contributing` equals `code`/`excl_removed` cell-for-cell (pinned by `test_excl_contributing_collapses_to_removed`; e.g. B clinic_pcen 0.0075 under both, vs ideal 0.0175). This is the concrete answer to "what does the current no-RV pipeline actually do" for the memo — and the silent exception swallowing itself is flagged for the Phase 3 bug audit |

Additionally the **popdensity denominator has no manuscript equation** — it
is a code-only extension of Eq. 3 (Population_i → Population_i / Area_i).
The worksheet labels its popdensity arithmetic "derived by analogy to Eq. 3,
ratified as a code extension", never implying manuscript authority.

**Primary oracle-ideal** = `ideal` rule-set, popsize denominator, Eq. 1 PSI.
The `code` rule-set columns are pinned as regression values, and every
ideal-vs-code gap is a documented finding routed to the memo/owner — none is
silently reconciled. An implementation task determines which column
(`unnorm_psi` or `norm_psi`) the paper's Figures actually report, since that
defines what "reproducing the paper's numbers" means.

## Decisions already made (brainstorm, owner-approved)

1. **Test surface**: both — library-first plus one end-to-end CLI run.
2. **Adjacency divergence**: main city geometry makes bbox == border-sharing
   provably identical (green suite); a divergence exhibit documents the
   disagreement cases; exhibit tests assert the documented divergence itself.
3. **Ratification**: ship-then-ratify, with a designated hand-verified
   anchor subset (see Worksheet section) so ratification is completable.
4. **Scope**: exclusion-semantics memo IN; Johannesburg reproduction OUT.
5. **Visuals are load-bearing**, including for missing-settlement variants.

## The city: Oraculum

Seven axis-aligned rectangles, EPSG:7760 (meters), coordinates as offsets
from base O = (1,000,000, 1,000,000). Rectangles only → bbox ≡ geometry →
adjacency rules 1-ideal and 1-code provably coincide in the main city. The
bottom row is offset 500 m so no settlement pair touches only at a corner.

| id | type | rectangle x×y (m) | centroid | pop | area km² | services |
|----|------|-------------------|----------|-----|----------|----------|
| A  | Planned | [0,1000]×[1000,2000] | (500,1500) | 100 | 1.0 | 2 clinics, 1 school, road segment |
| B  | Unauthorized | [1000,2000]×[1000,2000] | (1500,1500) | 200 | 1.0 | 1 clinic |
| C  | JJC | [2000,3000]×[1000,2000] | (2500,1500) | 400 | 1.0 | none |
| RV | Rural Village (removable) | [1100,1900]×[2000,3000] | (1500,2500) | **100** | 0.8 | 2 clinics |
| D  | Planned | [−500,500]×[0,1000] | (0,500) | 100 | 1.0 | 1 school |
| E  | Regularized-Unauthorized | [500,2500]×[0,1000] | (1500,500) | 300 | **2.0** | 1 clinic, **1 school**, road segment |
| IND| Industrial (removable) | [2500,3500]×[0,1000] | (3000,500) | 10 | 1.0 | none |

(RV population is 100, not 50 — deliberate: with RV at pop 100, IND —
though serviceless — is the strict argmax of clinic PCEN under BOTH
denominators, purely via E's decayed clinic over IND's tiny population
(1 × 1/2.5 = 0.4 → 0.4/10 = 0.04, vs RV's 2.5/100 = 0.025 popsize and
2.5/125 = 0.02 popdensity). Removing IND therefore moves a min–max anchor
with ZERO numerator change anywhere — a pure renormalization delta — while
RV removal produces a neighbor-contribution delta. The invariants script
asserts: (i) max > min for every service in every config — the only property
Eq. 2 needs; (ii) argmin/argmax uniqueness for CLINICS AND SCHOOLS under
both denominators in every scenario. Roads under literal Eq. 4 are
structurally exempt from (ii): with no neighbor term, every road-less
settlement is pinned at exactly 0, so the tied minimum is recorded as
expected ground truth in expected_values.csv rather than asserted away —
review-computed ideal road PCENs: A=0.0075, E=0.0025, all others exactly 0;
the code rule's decayed roads do have unique anchors.)

**Barrier**: canal linestring y=1000, x∈[25,475] — a strict interior
sub-segment of the A–D shared edge ([0,500] at y=1000). It must NOT reach
x=500: the draft's full-edge canal ended exactly on E's corner (500,1000),
which flags E as barrier-crossed and (under the code rule) guts the
adjacency graph — the review's headline Critical. Barrier flagging is
per-settlement, so a partial edge segment severs A–D exactly as well as the
full edge. Invariants: (a) canal ⊂ A–D shared edge; (b) the set of
settlements intersecting any barrier is exactly {A, D}; (c) no barrier
intersects any settlement at only a point.

### Geometric pair table (rule-independent; verified by shapely in review)

All ten touching pairs share an edge segment (never just a point):

| pair | centroid distance | decay 1/(1+D km) |
|------|------------------|------------------|
| A–B | 1000 m | 1/2 |
| A–D | 500·√5 m | (severed under both rule-sets) |
| A–E | 1000·√2 m | 1/(1+√2) |
| B–C | 1000 m | 1/2 |
| B–RV | 1000 m | 1/2 |
| B–E | 1000 m | 1/2 |
| C–E | 1000·√2 m | 1/(1+√2) |
| C–IND | 500·√5 m | 1/(1+√5/2) |
| D–E | 1500 m | 1/2.5 |
| E–IND | 1500 m | 1/2.5 |

### Directed post-barrier neighbor lists (rule-dependent)

The barrier rules make neighborhood DIRECTED (a settlement's list is whose
services it counts). Both columns below are part of expected ground truth;
the library test pins the `code` column empirically:

| settlement | `ideal` rule (A–D severed, both ways) | `code` rule (A, D flagged → removed from others' lists; keep their own) |
|------------|----------------------------------------|--------------------------------------------------------------------------|
| A  | B, E | B, E |
| B  | A, C, RV, E | C, RV, E |
| C  | B, E, IND | B, E, IND |
| RV | B | B |
| D  | E | E |
| E  | A, B, C, D, IND | B, C, IND |
| IND| C, E | C, E |

The asymmetry (A counts E's clinic; E does not count A's) is itself prime
memo material for Raj — it is the barrier-semantics gap made concrete.

**Coverage matrix (why each element exists):**
- C: zero services; PCEN entirely from decayed neighbors; A's clinics must
  not reach C (A, C not adjacent — second-order exclusion).
- Canal on A–D interior: touching settlements that are not neighbors, under
  both rule-sets; plus the ideal-vs-code asymmetry above.
- E double area (and RV 0.8 km²): popsize vs popdensity differ.
- RV service-rich next to B: exclusion scenario removes a *contributor*.
- IND serviceless with tiny population: its clinic PCEN (entirely E's
  decayed clinic) is the unique max, so the exclusion scenario moves a
  *normalization anchor* with no numerator change — the isolated
  renormalization effect.
- Road polyline x=750, y∈[250,1750]: 0.75 km in E, 0.75 km in A, nothing
  elsewhere (it crosses y=1000 at (750,1000), which lies on the A–E shared
  edge east of the canal's end at x=475 — it does not touch the canal;
  invariants assert these lengths and the non-intersection).
- Schools in A, D, and E: E's school breaks the otherwise-exact A/D
  symmetry (A receives E's school at decay 1/(1+√2) ≈ 0.4142, D at
  1/2.5 = 0.4), giving unique school argmin/argmax in all 16 configs
  (review-computed: ideal/baseline/pop A=0.014142, B=0.005, C=0.001036,
  RV=0, D=0.014, E=0.006047, IND=0.04). Without E's school, A and D tie
  at the max in every config — round-2 review's headline catch.

### Scenarios (exclusion semantics)

| scenario | meaning |
|----------|---------|
| `baseline` | all seven settlements indexed |
| `excl_contributing` | RV and IND get no index rows, but still contribute services as neighbors (semantics a) |
| `excl_removed` | RV and IND fully removed before neighbor computation (semantics b) |
| `excl_ind_removed` | IND alone fully removed — isolates the renormalization effect from RV's contribution effect |
| `excl_rv_only` | RV rows dropped AFTER neighbor computation, IND retained — the exact configuration `scripts/compute_psi.py` actually produces (its only filter is `USO_FINAL != "RV"`). This — not `baseline`, not `excl_removed` — is what `test_oracle_e2e.py` asserts against. Its PCENs coincide with `excl_removed` on shared settlements (IND is serviceless), but every min-max-derived `_idx`/`psi` column differs materially because IND is a max anchor — the e2e test must not be wired to `excl_removed` rows |

### Divergence exhibit (separate fixture, `divergence/`)

Explicit four-settlement universe, all in one frame, with pinned attributes
so deltas are hand-derivable. PCEN-level deltas are the primary recorded
quantity (PSI deltas can be annihilated by min-max in tiny universes; the
review proved a two-settlement universe forces them to zero):

| id | geometry (m) | pop | services |
|----|--------------|-----|----------|
| P | L-shape: [0,2000]×[0,1000] ∪ [0,1000]×[1000,2000] | 100 | 1 clinic |
| Q | [1200,1800]×[1200,1800] (inside P's notch; 200 m gap to P) | 100 | 1 clinic |
| R | [4000,5000]×[0,1000] | 100 | 2 clinics |
| S | [5000,6000]×[1000,2000] | 50 | none |

- P–Q: polygons disjoint, but Q lies inside P's bbox ([0,2000]×[0,2000]) →
  the bbox rule invents the directed link Q→P; border-sharing denies it,
  and so does `intersects` (the polygons never touch).
- R–S: touch at exactly the point (5000,1000). R and S are rectangles, so
  bbox ≡ geometry for both — **the bbox rule AND `intersects` both count
  this pair** (production-verified: `add_polygon_neighbors_column_fast`
  yields R:[S], S:[R]); "shares a border" (edge) says no.
- Net narrative (corrected in plan-review round 1, computationally, from
  earlier revisions that wrongly assigned R–S exclusively to `intersects`):
  the code's bbox rule exhibits BOTH divergence flavors — containment
  phantom (Q→P only, showing directedness: P's delta 0, Q's +0.005147186)
  and corner touch (R↔S, symmetric: S's delta +0.016568542, R's 0 since S
  is serviceless) — while `intersects` exhibits only the corner touch.
- The exhibit records, under each adjacency rule, every settlement's
  DIRECTED neighbor list, clinic PCEN, and (secondarily) service index —
  asserting the documented differences exist exactly as recorded.
  Figure 3 draws the P–Q phantom as a single arrow Q → P and the R–S
  corner-touch link as symmetric.

## Components (build order is normative — see Build Order)

### 1. Fixtures (`tests/fixtures/oraculum/`)
`settlements.geojson`, `services.geojson` (clinic/school points at pinned
interior coordinates + road linestring), `barriers.geojson`,
`divergence/*.geojson`, `expected_values.csv`. CRS EPSG:7760, tiny,
human-readable.

**`expected_values.csv` schema (long format, pinned now):** columns
`rule` ∈ {ideal, code}, `scenario` ∈ {baseline, excl_contributing,
excl_removed, excl_ind_removed, excl_rv_only}, `denom` ∈ {pop, popdensity},
`settlement`,
`metric` (e.g. clinic_count, clinic_pcen, clinic_idx, school_pcen, …,
road_length_km, road_pcen, road_idx, psi_eq1, norm_psi), `value`.
Example row: `ideal,baseline,pop,B,clinic_pcen,0.0175`.
`norm_psi` rows exist only under `rule=code`; road rows under `rule=ideal`
use Eq. 4 literally and under `rule=code` use the decayed formula.

### 2. Independent reference implementation (`tests/reference_impl.py`)
Pure numpy/pandas (+shapely for geometry predicates), written FROM THE
MANUSCRIPT'S EQUATIONS; must not import, call, or textually mirror
`spatial_index_utils.py` (review-enforced). Parameterized by: adjacency
rule (border | bbox | intersects), barrier rule (pair-severed | global
directed), roads formula (eq4 | decayed), exclusion scenario, denominator,
second normalization on/off, and `absent_neighbor_contribution` ∈
{contributes | swallowed} — bound to `contributes` under `rule=ideal` and
`swallowed` under `rule=code`; this seventh knob is the only axis on which
the two rule-sets differ for a shared scenario name, and it is what
`test_excl_contributing_collapses_to_removed` pins — so it can compute BOTH
rule-sets, all scenarios, and the exhibit.

### 3. Test suites
- `tests/test_reference_impl.py` — reference impl == expected_values.csv.
- `tests/test_oracle.py` — library-first: production functions on fixture
  GeoDataFrames == the `rule=code` rows (atol 1e-12), including the directed
  neighbor lists, and == the `rule=ideal` rows wherever the rule-sets agree
  (e.g. non-barrier-adjacent point services). Named cases:
  `test_zero_service_settlement`, `test_second_order_neighbor_excluded`,
  `test_barrier_rule_is_global_and_directed`,
  `test_popdensity_differs_from_popsize`, `test_minmax_anchors_unique`,
  `test_road_decay_divergence` (pins that code roads ≠ Eq. 4 by the
  recorded amount), `test_second_normalization_divergence`,
  `test_excl_contributing_collapses_to_removed` (pins that the code's bare
  `except: pass` makes semantics (a) degenerate to (b)).
- `tests/test_oracle_e2e.py` — temp data dir + real CLIs. **Manifest
  (complete, from reading the scripts):** colonies shapefile in
  `uso_update_sep2021/` layout (fields `USO_AREA_U`, `USO_FINAL`,
  geometry); `Barrier_Clip/Canal/Canal.shp` (the canal),
  `Barrier_Clip/Drain/Major_Drain.shp` and
  `Barrier_Clip/Railway/Railway_Line.shp` (empty-but-valid);
  `ndmc_center7760/ndmc_center7760.shp` (one point);
  `delhi_bounds_buffer/delhi_bounds_buffer.shp` (polygon containing all of
  Oraculum); `Public Services/{Banking,Health,Major Road,Police,Ration,
  School,Transport}/…` shapefiles with the real dataset's filenames —
  Health carries the clinics, School the schools, Major Road the road.
  The other four services are NOT empty (review finding: an all-empty
  service yields PCEN 0 everywhere → NaN index → Eq. 1's denominator
  silently shrinks): each gets exactly one pinned point in a settlement
  that survives every e2e scenario — Banking→A, Police→B, Ration→D,
  Transport→E — treated as ordinary point services by the reference impl,
  with their metrics in expected_values.csv (machine-checked, outside the
  hand-ratified anchor subset); `pop_colony_wp_2020_jjc_adjusted.csv`
  with exact lowercase columns `uso_area_u,population`; `--neighbors-file`
  wired from the preprocess output to compute_psi, outputs to a temp
  out-dir. **Scope note:** the CLI path hardcodes RV-only exclusion, so the
  e2e leg validates only the baseline/RV-drop path; the full scenario
  matrix is three-way-verified via the library and reference legs only —
  stated honestly in the worksheet.
- `tests/test_divergence_exhibit.py` — asserts the documented divergences
  and their PCEN deltas exactly as recorded.

### 4. Visuals (`scripts/render_oracle_maps.py` → `docs/oracle/*.png`)
Deterministic matplotlib from fixture files; implementation loads the
`dataviz` skill first. Three figures: (1) `oraculum_city.png` — types,
labels, service markers, canal glyph, DIRECTED neighbor graph (arrowheads
showing the code rule's asymmetry; ideal-only links dashed); (2)
`oraculum_exclusion_variants.png` — panels: baseline / excl_contributing
(RV+IND ghosted, dashed contribution arrows) / excl_removed /
excl_ind_removed, affected settlements annotated with PCEN/PSI per panel;
(3) `oraculum_divergence.png` — P's polygon vs dashed bbox, phantom P–Q
link, R–S corner touch, annotated PCEN deltas.

### 5. Derivation worksheet (`docs/oracle/derivation-worksheet.md`)
Banner: `STATUS: RATIFICATION PENDING …`. **Hand-verified anchor subset
(the ~15-minute calculator pass, per review):** the `ideal`/`baseline`/
`pop` config for all seven settlements end-to-end, plus one worked
exclusion delta (B under excl_removed), one worked road Eq. 4 value, the
`excl_ind_removed` clinic renormalization delta for settlement A (noting it
is denominator-INVARIANT by construction, since A, C, and IND all have area
1.0 km² — itself a worksheet observation worth stating), and — so the
calculator pass genuinely covers the popdensity extension — E's clinic PCEN
under both denominators (2.328427/300 = 0.00776142 popsize vs
2.328427/150 = 0.01552285 popdensity).
All other configs are machine-cross-checked by the (independently reviewed)
reference implementation; the worksheet says so explicitly. Radicals kept
exact until final decimals.

### 6. Exclusion-semantics memo (`docs/oracle/exclusion-semantics-memo.md`)
For Raj: the 4-panel figure, per-settlement delta tables (RV-contribution
effect vs IND-renormalization effect, both denominators), the directed
barrier asymmetry, the roads/norm_psi/popdensity manuscript gaps, and what
the current code actually does. Explicitly no recommendation.

### 7. Consistency guard (`scripts/check_oraculum_invariants.py`)
Asserts from fixtures: all main-city geometries are rectangles
(bbox == geometry); every touching settlement pair shares an edge (never
only a point); pair distances match the geometric table; canal ⊂ A–D edge,
touches exactly {A, D}, never at only a point; road lengths 0.75 km in each
of A and E and zero elsewhere; road does not intersect the canal;
max > min for every service PCEN in every config; argmin/argmax uniqueness
asserted ONLY for clinics and schools (positive list) under both
denominators in every scenario. All other services have tied anchors by
construction — road/bank/ration tied zero argmins, police tied argmax
between A and B — and those ties are themselves recorded expected ground
truth in expected_values.csv, so a fixture edit cannot silently change
them. Runs in the test suite.

## Build order (normative — review finding G18)

1. **Fixtures + invariants + empirical pin**: author geometry, run the real
   `barrier_intersection`/`add_polygon_neighbors_column_fast`/
   `remove_ids_with_barrier` chain against it, confirm the directed
   neighbor table above verbatim BEFORE anything downstream is written.
2. **Schema pin**: expected_values.csv columns + config enumeration frozen.
3. **Reference implementation** + `test_reference_impl.py`.
4. **Library-first oracle suite**, then **e2e suite**.
5. **Worksheet, visuals, memo** (they consume finalized numbers).

## Non-goals

- No changes to `spatial_index_utils.py` or the pipeline scripts. The four
  ideal-vs-code gaps are FINDINGS (documented, routed to owner/Raj), not
  things this phase fixes.
- No resolution of adjacency/barrier/roads/norm_psi questions (Phase 3+,
  with Raj, informed by the oracle's concrete numbers).
- No Johannesburg reproduction; no pandas-3 uncap.

## Risks / implementation notes

- Unit convention: `calc_nbr_dist` stores km (review-confirmed structure:
  list of (neighbor_id, distance_km) tuples); fixture arithmetic uses km.
- `calc_all_services` input contract: columns `USO_AREA_U`, `USO_FINAL`,
  `population`, `nbrs_bbox`, `nbrs_dist_bbox`, `centroid`, `area_km2`,
  geometry; the library test builds exactly what `compute_psi.py` feeds it.
- Shapefile 10-char field truncation applies to e2e fixture writing; use
  the real dataset's field names.
- If empirical pinning (build-order step 1) contradicts the directed
  neighbor table in this spec, STOP and update the spec before proceeding —
  the table is this spec's most load-bearing claim.

## Acceptance criteria

1. `uv run pytest` green: existing 14 tests + all new suites, including the
   invariants guard.
2. Reference implementation independent (review-verified), equal to
   expected_values.csv across every (rule, scenario, denom) config.
3. Production code equals the `rule=code` rows; every ideal-vs-code gap is
   asserted at its recorded magnitude (roads, norm_psi, barrier asymmetry)
   — nothing silently reconciled.
4. Three maps render deterministically from fixtures; worksheet (with
   anchor-subset scope statement) and memo committed; invariants green.
5. CHANGELOG updated; PR reviewed and merged; ratification banner PENDING
   until Bob's calculator pass (post-merge by design).
