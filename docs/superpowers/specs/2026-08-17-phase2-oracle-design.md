# Phase 2: The Mythical-City Oracle — Design Spec

Date: 2026-08-17
Status: awaiting owner approval
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

## Decisions already made (brainstorm, owner-approved)

1. **Test surface**: both — library-first (functions on fixture
   GeoDataFrames) plus one end-to-end CLI run of the real scripts.
2. **Adjacency divergence**: main city uses geometries where bbox and
   border-sharing provably agree (green suite vs. current code); a separate
   "divergence exhibit" documents where the two rules disagree and by how
   much, feeding the Phase 3 decision with Raj. Exhibit tests assert the
   documented divergence itself (they pass; no xfail).
3. **Ratification**: ship-then-ratify. The derivation worksheet carries a
   RATIFICATION PENDING banner until Bob (and ideally Raj) verify by hand;
   a one-line commit flips it.
4. **Scope**: exclusion-semantics decision-support memo for Raj is IN;
   South African (Johannesburg) reproduction is OUT (future stretch).
5. **Visuals are load-bearing**: maps must make the oracle interpretable,
   including the "missing settlements" (exclusion) variants.

## The city: Oraculum

Seven axis-aligned rectangular settlements in EPSG:7760 (meters). All
coordinates are offsets from base O = (1,000,000, 1,000,000); the table
gives offsets in meters. Rectangles only, in the main city — for a
rectangle, bbox ≡ geometry, so bbox-overlap and polygon-intersection
adjacency provably coincide. The bottom row is offset 500 m so **no pair of
settlements touches only at a corner point** (every touching pair shares an
edge segment).

| id | type (USO_FINAL-style) | rectangle x×y (m) | centroid | pop | area km² | services |
|----|------|-------------------|----------|-----|----------|----------|
| A  | Planned | [0,1000]×[1000,2000] | (500,1500) | 100 | 1.0 | 2 clinics, 1 school, road segment |
| B  | Unauthorized | [1000,2000]×[1000,2000] | (1500,1500) | 200 | 1.0 | 1 clinic |
| C  | JJC | [2000,3000]×[1000,2000] | (2500,1500) | 400 | 1.0 | none |
| RV | Rural Village (removable) | [1100,1900]×[2000,3000] | (1500,2500) | 50 | 0.8 | 2 clinics |
| D  | Planned | [−500,500]×[0,1000] | (0,500) | 100 | 1.0 | 1 school |
| E  | Regularized-Unauthorized | [500,2500]×[0,1000] | (1500,500) | 300 | **2.0** | 1 clinic, road segment |
| IND| Industrial (removable) | [2500,3500]×[0,1000] | (3000,500) | 10 | 1.0 | none |

**Barrier**: one canal linestring along y=1000 for x∈[0,500] — exactly the
A–D shared edge — severing A–D adjacency.

**Adjacency (border-sharing == bbox, by construction), after barrier:**

| pair | centroid distance | decay 1/(1+D km) |
|------|------------------|------------------|
| A–B | 1000 m | 1/2 |
| B–C | 1000 m | 1/2 |
| B–RV | 1000 m | 1/2 |
| B–E | 1000 m | 1/2 |
| D–E | 1500 m | 1/2.5 |
| E–IND | 1500 m | 1/2.5 |
| A–E | 1000·√2 m | 1/(1+√2) |
| C–E | 1000·√2 m | 1/(1+√2) |
| C–IND | 500·√5 m | 1/(1+√5/2) |
| A–D | 500·√5 m (unused) | **severed by canal — no link** |

Two exact irrational distances (√2, √5/2) are deliberate: they test
floating-point honesty while remaining exactly derivable on paper.

**Why each element exists (coverage matrix):**
- C has zero services and neighbors B/E/IND → zero-own-service PCEN comes
  entirely from decayed neighbors; A's clinics must NOT reach C
  (second-order exclusion: A and C are not adjacent).
- Canal on A–D: touching settlements that are NOT neighbors.
- E's double area: pop-size and pop-density PCEN variants produce different
  numbers (all-unit-squares would make them identical). RV's 0.8 km² adds a
  second area contrast.
- RV is service-rich and adjacent to B → removing RV visibly drops B's
  clinic PCEN (the dramatic case for the exclusion memo).
- IND has NO services → removing IND changes E/C only via the min–max
  normalization universe, isolating the renormalization effect (the subtle
  case for the memo).
- Road polyline: vertical segment x=750 from y=250 to y=1750 → exactly
  0.75 km inside E and 0.75 km inside A → exercises the length-based road
  index (Eq. 4) with hand-computable lengths.
- Service points sit at pinned interior coordinates (recorded in the
  fixture files; exact placement is arbitrary but fixed).

**Barrier-semantics determination (explicit sub-goal).** The manuscript says
barrier-crossed adjacencies are not counted ("we manually marked areas that
had river or railroad tracks and then ensured that we don't count these
services"). The code's mechanism (`barrier_intersection` flags settlements
touching any barrier; `remove_ids_with_barrier` filters neighbor lists) may
implement something broader — e.g., removing barrier-flagged settlements
from ALL neighbor lists, not just severing the crossing pair. The oracle
treats this as an empirical determination, not an assumption: the library
test pins the code's actual rule (with the canal flagging both A and D, do
A–B, A–E, D–E links survive?); the reference implementation computes the
manuscript-ideal (sever only the crossing pair); any gap becomes a
documented finding in the memo (same treatment as bbox-vs-border) and
`expected_values.csv` carries columns for both interpretations so the suite
stays green against current behavior while the ideal is on record.

**Divergence exhibit** (separate fixture, `divergence/`):
- P: L-shaped settlement (three 1 km cells: [0,2000]×[0,1000] ∪
  [0,1000]×[1000,2000]); its bbox is [0,2000]×[0,2000].
- Q: square [1200,1800]×[1200,1800] sitting in P's notch — polygons
  disjoint (200 m gap), but Q lies wholly inside P's bbox →
  **bbox adjacency invents a P–Q neighbor link that border-sharing denies**.
- R, S: unit squares touching only at the point (1000,1000) →
  intersects-predicate says neighbors; "shares a border" (edge) says no.
- The exhibit computes PCEN/PSI under both rules and records the deltas.

## Components

### 1. Fixtures (`tests/fixtures/oraculum/`)
`settlements.geojson`, `services.geojson` (points + road linestring),
`barriers.geojson`, `divergence/*.geojson`, `expected_values.csv`. All tiny,
human-readable, CRS EPSG:7760. `expected_values.csv` columns: settlement id;
per-service count/length, PCEN, service index; unnorm_psi; norm_psi — for
each of the 2×2 combinations {popsize, popdensity} × {exclusion semantics a
(excluded-but-contributing), b (fully removed)} plus the baseline
all-seven-settlements city.

### 2. Independent reference implementation (`tests/reference_impl.py`)
Pure numpy/pandas (+shapely for geometry predicates only), ~80–120 lines,
written FROM THE MANUSCRIPT'S EQUATIONS. Hard independence rule: it must not
import, call, or textually mirror `spatial_index_utils.py`; its docstring
cites Eq. 1–4; the code-review loop explicitly checks independence.
Parameterized by: adjacency rule (border-sharing | bbox-overlap |
intersects), exclusion semantics (a | b | none), denominator (popsize |
popdensity). This is what computes both sides of the exclusion memo and the
divergence exhibit numbers.

### 3. Test suites
- `tests/test_reference_impl.py` — reference impl == expected_values.csv
  (human ↔ reference).
- `tests/test_oracle.py` — library-first: `spatial_index_utils` neighbor
  computation (with barriers) and `calc_all_services` on fixture
  GeoDataFrames == expected_values.csv, atol 1e-12 (production ↔ human).
  Named cases: `test_zero_service_settlement`,
  `test_second_order_neighbor_excluded`, `test_barrier_severs_adjacency`,
  `test_popdensity_differs_from_popsize`, `test_minmax_anchors`,
  `test_road_length_index`.
- `tests/test_oracle_e2e.py` — materializes a temp data directory (writing
  the fixture layers as shapefiles in the real dataset's layout, plus the
  population CSV), runs the actual `scripts/preprocess.py` and
  `scripts/compute_psi.py` CLIs against it, and compares final CSVs to
  expected values. This test also PINS DOWN the current scripts' de-facto
  exclusion semantics (neighbors are computed before RV rows are dropped —
  whether a dropped neighbor still contributes services is currently
  undocumented behavior; the e2e test records what actually happens, and
  the memo reports how current behavior maps onto ideal semantics a/b).
- `tests/test_divergence_exhibit.py` — asserts the documented divergences
  exist exactly as recorded (bbox invents P–Q; intersects invents R–S;
  PSI deltas equal recorded values). Passing tests that document reality.

### 4. Visuals (`scripts/render_oracle_maps.py` → `docs/oracle/*.png`)
Deterministic matplotlib rendering FROM the fixture files (no hand-drawn
elements; map and data cannot drift). Implementation must load the `dataviz`
skill before writing chart code. Three figures:
1. `oraculum_city.png` — settlements colored by type, labels
   (name/pop/area), service markers, canal glyph, neighbor graph with
   distance labels, severed A–D link visibly absent.
2. `oraculum_exclusion_variants.png` — three panels: full city; semantics a
   (RV/IND ghosted, dashed contribution links); semantics b (RV/IND gone).
   Affected settlements (B, E, C) annotated with their PCEN/PSI under each
   panel so the delta reads directly off the map.
3. `oraculum_divergence.png` — P's polygon vs. dashed bbox overlay, the
   phantom P–Q link, the R–S corner touch, annotated deltas.

### 5. Derivation worksheet (`docs/oracle/derivation-worksheet.md`)
Opens with `STATUS: RATIFICATION PENDING — derived by Claude from Eq. 1–4;
awaiting hand verification by Bob (and Raj).` Then, per settlement: inputs →
arithmetic (shown step by step, fractions and radicals kept exact until the
final decimal) → result, cross-referenced to the map. Sized for a
~15-minute calculator pass. Ratification = one-line banner edit, committed.

### 6. Exclusion-semantics memo (`docs/oracle/exclusion-semantics-memo.md`)
For Raj: the 3-panel figure, a per-settlement delta table (both PSI
variants), a plain-language paragraph on what semantics a vs. b mean, what
the current code actually does (from the e2e findings), and what changes for
Delhi. Explicitly no recommendation — it grounds his decision (WORKPLAN
"Open decisions A").

### 7. Consistency guard (`scripts/check_oraculum_invariants.py`)
Asserts from the fixture files: all main-city geometries are axis-aligned
rectangles (bbox == geometry); no pair touches only at a point; every
documented adjacency/distance in this spec matches the geometry; barrier
coincides with the A–D shared edge; road lengths per settlement are
0.75 km each. Run in the test suite so fixture edits cannot silently
invalidate the spec's tables.

## Non-goals

- No changes to `spatial_index_utils.py` or the pipeline scripts (if the
  oracle exposes a production-code mismatch with the manuscript, that is a
  FINDING — documented and taken to the owner/Raj per the oracle contract —
  not something this phase silently fixes; exception per Phase 1 precedent:
  none anticipated here since no dependencies change).
- No resolution of the bbox-vs-border question (Phase 3, with Raj, informed
  by the exhibit).
- No Johannesburg reproduction.
- No pandas 3 uncap (Phase 2 may later use the oracle to validate it, but
  not in this phase).

## Risks / implementation notes

- **Unit convention**: Eq. 3's D is in km (decay 1/(1+D)); the code's
  `calc_nbr_dist` stores distances in some unit — implementation must
  confirm against the source (history says km) and the fixture arithmetic
  must match the code's actual convention; any mismatch is a finding.
- **`calc_all_services` input expectations**: the oracle constructs
  GeoDataFrames matching what `compute_psi.py` feeds it (column names
  `USO_AREA_U`, `USO_FINAL`, `population`, `nbrs_bbox`, `nbrs_dist_bbox`,
  `centroid`, `area_km2`); the e2e test guards against drift in those
  assumptions.
- **Min–max degeneracy**: with 7 settlements, ensure no service's PCEN is
  constant across all settlements (min == max would make Eq. 2 divide by
  zero); the chosen service placement avoids this — the invariants script
  checks it.
- **Shapefile column-name truncation** (10-char ESRI limit) affects the e2e
  fixture writing; use the same names the real dataset uses.

## Acceptance criteria

1. `uv run pytest` green: existing 14 tests + all new oracle suites.
2. Reference implementation demonstrably independent (review-verified) and
   equal to expected_values.csv, which equals the worksheet's arithmetic.
3. `spatial_index_utils` path and the real-CLI e2e path both reproduce the
   expected values (or any mismatch is surfaced as a documented finding and
   halts for the owner per the oracle contract — not patched silently).
4. Three maps render deterministically from fixtures; worksheet and memo
   committed; invariants script green.
5. CHANGELOG updated; PR reviewed and merged; ratification banner remains
   PENDING until Bob's calculator pass (post-merge by design).
