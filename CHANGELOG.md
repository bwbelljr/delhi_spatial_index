# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/) conventions. Each WORKPLAN
phase lands as one entry set when its branch merges; the `[Unreleased]`
section accumulates changes on in-flight branches.

## [Unreleased]

- **Phase 6 sweep harness + a dry run on `code-2025`** (DEL-55, serving
  DEL-36 distance bands / DEL-37 decay weights / DEL-39 the adjacency
  comparison). Eleven sweep profiles, a runner, a summariser, and
  `docs/data/phase6_sweep.md`. **Every number it produces is provisional by
  construction** — `code-2025` still carries `bbox` adjacency,
  `global_asymmetric` barriers and decayed roads, all of which Raj's 28 Aug
  decisions supersede — and every table in the document says so. The point
  was to prove the machinery and MEASURE THE COST before the numbers matter,
  so that Phase 6 against the ratified profile (DEL-31) is a re-run rather
  than a design exercise.
  - **Eleven profiles**, each `code-2025` with exactly one factor moved:
    `adj-touch`, `band-0km/1km/5km/10km`, and `decay-none / -power05 /
    -power2 / -exp2km / -exp5km / -boundary`. A one-factor guard test
    (`tests/test_sweep_profiles.py`) proves mechanically that no profile
    moves a key it does not claim, comparing `methodology` AND `categories`.
  - **Nothing existing moved.** Both cities' `expected_values.csv` and the
    `code-2025` / `manuscript` production fixtures are byte-identical;
    `variants_expected_values.csv` changed by addition only (3,220 lines on
    messy, 2,254 on oraculum, **zero deletions** by `git diff --numstat`);
    and the real-data `code-2025` verify still passes at `0.000e+00` on all
    60 comparisons.
  - **Seven new `tests/variants.py` rows** put the sweep's own constants
    through the two-implementation oracle: `decay.form: none`,
    `inverse_power` at 0.5, `exponential` at 2 and 5 km, and the three real
    radii. The radii earned their rows against an earlier claim in the spec
    that they would be redundant: the fixture cities are 4.0 x 3.0 km and
    21.0 x 2.4 km, not 200 m, so all three radii differ on both — and 1 km
    lands EXACTLY on the `<=` boundary in both cities (10 km does once on
    messy). Both implementations agree on every boundary pair, which is now
    pinned rather than assumed.
  - `scripts/run_sweep.py` — plans each point against the artifact actually
    on disk, runs the CLI as a subprocess so one point cannot take the
    others down, and records cost in a per-point manifest. **The six decay
    points share one neighbours artifact**, which is safe by construction
    rather than convention: `check_methodology_stamp` covers the adjacency
    and barrier blocks, and refuses a mismatch.
  - `scripts/summarize_sweep.py` — the comparison statistics. **Every
    cross-point statistic is rank-based**, because PSI levels are NOT
    comparable across points: Eq. 2's min-max runs per run and widening the
    neighbourhood inflates everyone at once. A "mean PSI by category" table
    across points would look rigorous and mean nothing. Decile sets are
    tie-inclusive and gated, because PSI has a mass point at exactly zero
    larger than a decile (452 rows at the baseline, 1,834 under the own-only
    anchor, against a decile of 413) — computed naively, "the bottom 10 %"
    is decided by sort order, and the same two series gave a Jaccard of
    0.070 in file order and 0.157 under a random permutation.
  - **`docs/methodology-config.md` § 3 gains the registration step it was
    missing**: `tests/test_config.py` asserts the exact set of shipped
    profiles, so adding one turns the suite red, and the documented
    procedure never said so.
  - **The measured cost, which is what DEL-31 needs.** All eleven points ran
    against the real layer (4,131 reported settlements), total 7.15 h:

    | point | links | preprocess | compute |
    |---|---|---|---|
    | `decay-*` (bbox baseline) | 21,211 | 694.6 s cold, then shared | ~70 s each |
    | `adj-touch` | 14,641 | 91.6 s | 50.4 s |
    | `band-0km` | 15,462 | 11.7 s | 52.7 s |
    | `band-1km` | 165,119 | 75.5 s | 569.6 s |
    | `band-5km` | 1,525,802 | 752.5 s | 5,600.3 s |
    | `band-10km` | 4,366,055 | 2,576.4 s † | **14,833.1 s (4.12 h)** |

    † measured under concurrent load (an 18-minute test-suite run overlapped
    it), so it is an upper bound. Every other figure is clean.

    The 10 km link count reproduces August's independent measurement of
    4,366,055 exactly. Compute is linear in links with a fixed floor, and the
    per-link rate drifts up with scale (0.002796 → 0.003480 → 0.003697 s/link
    across successive fits), so extrapolate conservatively. The settlement
    dedup is ~620 s of a cold preprocess and is paid **once per work dir**,
    not per point. `band-0km`'s preprocess is 8× cheaper than `adj-touch`'s
    despite more links: `dwithin` is one vectorised query, `touch` computes
    intersection lengths pair by pair — the intersection rule is not the
    cheap rule.
  - **What the sweep found.** `JJC` is the lowest-scoring category at all
    thirteen points, anchors included, and the bottom three are
    `RUAC` > `UAC` > `JJC` at every one: the adjacency and decay choices move
    the top of the ranking substantially and the bottom not at all. Widening
    the band erodes settlement-level agreement monotonically
    (`taub_vs_bbox` `0.677` → `0.540` → `0.507`) and collapses
    `own_share_p50` to `0.001` at 10 km — at which point the index is 99.9 %
    other settlements' services. The published baseline is itself already
    flagged `smoothed`, at `own_share_p50: 0.058`.
  - **A finding for DEL-52, which is Raj's open decision.** The two
    denominators disagree sharply: Kendall τ between their category orderings
    is `0.17`, and Cliff's δ for Planned-vs-JJC is `0.22` under `pop` against
    `0.90` under `popdensity` — P(a random Planned settlement outranks a
    random JJC) moves from 0.61 to 0.95. The mechanism is that `popdensity`'s
    denominator is population/area, so the index rewards large-area
    settlements (Spearman `0.929` between a category's median area and its
    percentile swing); JJC's median area is 0.003 km². The paper's central
    formal/informal claim is substantially stronger under the `popdensity`
    denominator Figure 4 already uses.

- **`methodology.overlap.lending`** — a new required switch: what a
  NEIGHBOUR lends. `whole` (today's rule) lends the neighbour's whole
  amount of a service; `outside_receiver` lends `|S_j \ S_i|` — the amount
  minus whatever of the same service already lies inside the receiver — so
  a service sitting in the overlap of two colony polygons is not counted
  twice for the same settlement (DEL-20, Bob's proposal of 5 Sep 2026;
  cycle 3E, the third and last of three per-ticket PRs after DEL-54 and
  DEL-48). Raj ratified only the COUNTING half of this on 28 Aug 2026 — a
  service inside k overlapping colonies counts for each of the k, which is
  today's behaviour on both sides and does not move — so `overlap.counting`
  is a reserved key with no knob, and the lending half awaits his answer.
  - Both shipped profiles gain the key with today's value `whole`, and
    **nothing either profile computes moves**: both cities'
    `expected_values.csv` and every `production/*.csv` are byte-identical,
    and the two `variants_expected_values.csv` files changed by **addition
    only** (322 new rows per rule on oraculum, 460 on messy, zero
    deletions).
  - The shared structure `{(i, j): amount}` is built compute-locally from
    service containment and is sparse — it costs nothing on a clean,
    non-overlapping pair.
  - The rule is **not** in the methodology stamp, so one stored neighbours
    artifact serves both values of the switch.
- **`methodology.barrier.rule: partial_weighted`** — a barrier that covers
  only part of a shared boundary now discounts that neighbour's contribution
  by the covered share, `w_ij = 1 − L_blocked/L_shared`, instead of severing
  the link outright (DEL-48, Raj's 28 Aug 2026 decision; cycle 3E, the second
  of three per-ticket PRs after DEL-54). Implemented independently on both
  sides of the oracle, and pinned by the new `partial_5m` variant on both
  fixture cities.
  - **`barrier.buffer_m`** is how close a barrier must be to block a boundary
    point, in metres. Required by, and rejected outside of, this rule, and
    strictly **> 0**: `LineString.buffer(0)` is empty in shapely, so a zero
    buffer would silently make every weight 1. It is a DISTANCE with round
    caps, so the blocked span extends past each end of the barrier *where the
    shared boundary continues past it* — which is why the oracle city's canal
    gives w_AD = 0.08 rather than the memo's buffer-free 0.1.
  - The weight travels in a new neighbours-frame column that exists **only**
    under this rule and never leaves `index_frames`, so the output column set
    is identical under every barrier rule and the pruned-neighbour-list
    contract every other consumer depends on is untouched. Links are pruned
    only at w == 0 exactly. `buffer_m` joins the methodology stamp, because it
    shapes the stored lists; artifacts written before this change still load.
  - A corner-only pair has no boundary to block, so its weight is 1 — which
    differs from `pairwise`, deliberately.
  - **Behaviour change worth noting:** `barrier.combine` now selects the
    layers the geometry-based rules see, not just the flag column. Previously
    `pairwise` severed across every configured layer whatever `combine` said.
    No profile, fixture or output uses a non-`any` combine, so nothing moved.
  - **No existing expected value moved.** Both cities' `expected_values.csv`
    and every `production/*.csv` are byte-identical; the two
    `variants_expected_values.csv` files changed by **addition only** (322 new
    rows on oraculum, 460 on messy, zero deletions). The shipped profiles are
    untouched — `code-2025` still carries `global_asymmetric`.
- `index.minmax` raises instead of dividing 0/0 on a degenerate group
  (DEL-54, WORKPLAN bug-audit item 6; cycle 3E, the first of three
  per-ticket PRs). Eq. 2 is undefined when every reported settlement scores
  the same on a service; the original computed `(v − lo) / (hi − lo)`
  regardless, which is a silent NaN column outside a `-W error` run and an
  unattributed numpy `RuntimeWarning` inside one. The guard precedes the
  division and names the column, the row count and the value, so both
  callers — every service's `service_index` and `overall_psi`'s second
  normalisation — surface the same `ValueError`. **The independent
  reference implementation now raises at both of its min-max sites too**,
  instead of returning `0.0`: the equations do not define a value at
  hi == lo, and a reference that invents one is a rule-set divergence
  waiting to be relied on. Deliberately out of scope, and stated in the
  docstring: an all-NaN column does not trigger it, because `NaN == NaN` is
  False and an all-NaN PCEN column is an upstream NaN that belongs to the
  population join and `validate`. **No config value, no profile change, no
  fixture change** — all three generators re-run byte-identical, because
  `scripts/check_oraculum_invariants.py` already refuses to write a city
  with a degenerate min-max group, and the real-data `code-2025` verify is
  unchanged at `0.000e+00`.
- Pre-recalculation measurements (DEL-49, DEL-50, DEL-51, DEL-52): four
  re-runnable scripts under `scripts/` — `measure_roads_access.py`,
  `inventory_barriers.py`, `measure_psi_columns.py`, and
  `count_corner_only_pairs` added to `measure_layer_pathologies.py` — with
  a shared `scripts/_measure_common.py` (the read-only work-dir guard, the
  settlement loader, and a labelled fenced-block `render`/`parse_block`),
  55 new tests (52 fixture-level, 3 real-data-gated), and four `docs/data/` documents that carry each
  script's block verbatim under a drift test (`roads_access.md`,
  `barriers.md`, `psi_columns.md`, and two new keys in
  `layer_pathologies.md`). Findings, one clause each: **DEL-49** 17 of 764
  JJCs contain a major road and 645 reach one only through a touching
  neighbour, so `roads: eq4_own_only` zeroes the road index of 422 of the
  749 reported JJCs and moves the JJC mean PSI by −13 % (pop) / −10 %
  (density) with no JJC/Planned ordering flip — the decision stands;
  **DEL-50** corner-only contact pairs on the real layer counted (value in
  `layer_pathologies.md`); **DEL-51** the barrier layers' schemas are an
  official GIS source's, created in ArcGIS Pro on 2 Aug 2020, the root
  copies are re-exports, the agency is still an open question for Bijoy;
  **DEL-52** the paper's Figure 4 reports `norm_psi` under the
  population-density denominator (8 of 8 bars, max gap 0.0006), so both of
  Bob's proposed defaults (`second_normalization: false`, drop popdensity)
  are withdrawn and the two choices go to Raj. The decision log's §§ 2, 3,
  4, 7 and 8 and its batched-reply checklist now carry numbers. Lesson
  recorded in the docs: a warm GeoPackage dedup cache upcasts Polygon to
  MultiPolygon, so any `geom_type`-based count must run cold. **No
  `delhi_psi/` behaviour change, no profile change, no fixture change.**
- Raj's methodology decisions from the 28 Aug 2026 call recorded (docs
  only, no code or profile change): new decision log
  `docs/decisions/2026-08-28-raj-methodology-decisions.md` with transcript
  timestamps, the config value each decision lands as, and Bob's rulings on
  the sub-questions; WORKPLAN Open Decisions A and C closed (A → semantics
  (a), min-max over reported types; adjacency → `touch`; roads →
  `eq4_own_only`; barrier → partial weighting, which is the reserved
  `partial_weighted` value and becomes cycle 3E, DEL-48; overlap services
  count for each owner, plus Bob's neighbour-lending rule on DEL-20);
  Phase 4 rewritten — no category collapse (DEL-29 parked), drop `RV,
  Industrial, Other`, DEL-31 shrinks to the ratified profile; new items
  DEL-48–53 (partial barriers, JJC roads measurement, corner-only pairs,
  barrier provenance, `norm_psi`/popdensity, media reclassification run);
  DEL-46 reproducibility appendix planned rather than optional.
  `docs/methodology-config.md` § 1 gains a "ratified" column and § 2 marks
  the collapse recipe as parked; `docs/oracle/suggested-fixes-memo.md`
  carries Raj's answer under each item. Recorded caveat: the call's roads
  premise was inverted (the code decays roads, the paper does not), so the
  roads decision changes the published numbers; its effect is measured
  first (DEL-49) and Raj is told.
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
  proof, 28 Aug 2026: `delhi-psi preprocess` — 4,357 settlements,
  595 barrier-flagged; `delhi-psi compute` — 4,131 reported,
  `categories: scheme=uso-10 n_categories=10`). Phase 6's sweeps (DEL-36
  thresholds, DEL-37 decay weights, DEL-39 adjacency comparison) are now
  loops over YAML profiles. Tests 386 → 539. Docs:
  `docs/methodology-config.md` §§ 1, 4, 6 (a complete `band-1km.yaml`, the
  X = 0 ≠ `touch` note, the centroid-vs-boundary note and the real-layer
  timing note).
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
  within tolerance` with max abs deviation `0.000e+00` on all 30 compared numeric
  columns (real-data proof, 28 Aug 2026 04:27–04:31 CDT: `delhi-psi
  preprocess` — 4,357 settlements, 595 barrier-flagged; `delhi-psi compute`
  — category column identity True on 4,131 rows, `categories: scheme=uso-10
  n_categories=10`). Tests 281 → 386. Docs: `docs/oracle/messy-city.md`,
  `docs/data/layer_pathologies.md`, `docs/methodology-config.md` § 4.
- Phase 3B settlement-category mapping layer: a profile now declares
  `categories: {scheme, mapping}` (source type → category, 1:1 or X:1) and
  writes `methodology.exclusion.types` in **category** names; every output —
  CSV, shapefile, joblib — and `missing_population.csv` carry a `category`
  column beside the raw `USO_FINAL`, the joblib also carries the
  scheme/mapping in `attrs`, and each run logs
  `categories: scheme=… n_categories=…`. The mapping is applied in the one
  population/exclusion prelude both entry points share, so `compute_frames`
  and the CLI cannot diverge; `compute_frames` gains `mapping=`/`scheme=`
  (`None` = the identity over the city's own types). An unmapped source type
  is an **error**, never a warning and never a fallback: `compute` exits 1
  naming every offending type with its row count, and `categories.default` is
  rejected at load as reserved. `categories` is required in every profile,
  duplicate YAML keys are now rejected naming key and line (PyYAML kept the
  last one silently), and `exclusion.types ⊆ categories` is checked both at
  load and at run time (in-memory callers never pass through `load_config`).
  **No numbers changed**: both shipped profiles use the identity `uso-10`
  scheme, `tests/fixtures/oraculum/production/code-2025.csv` and
  `manuscript.csv` are byte-identical, and
  `scripts/verify_against_baseline.py --config code-2025` still reports
  `0.000e+00` on all 30 numeric columns (both output sets) against the July 2025 baseline (real-data
  proof, 28 Aug 2026: `categories: scheme=uso-10 n_categories=10`, 4,131
  reported, `PASS — new run equivalent to July 2025 baseline within
  tolerance`). Proved by a CLI end-to-end that collapses the oracle city's
  six types into five, excludes the category `non-urban`, and reproduces
  both today's raw `[RV, IND]` exclusion and the independent reference
  implementation's `code/excl_contributing` and `code/excl_removed` blocks
  for both stages and both denominators. Raj's Phase 4 decision (DEL-31) is
  now one YAML file. Tests 246 → 279 (`test_categories`, the config and
  pipeline cases, the collapse e2e, the unmapped-type guard, the fixture
  id == type pin). Docs: `docs/methodology-config.md` § 2,
  `docs/data/uso_final_vocabulary.md`. Spec:
  `docs/superpowers/specs/2026-08-27-phase3b-categories-design.md`.
  [DEL-17, DEL-18 (further partial)]
- Phase 3A refactor: `spatial_index_utils.py` (822 lines) and the two driver
  scripts are gone; the pipeline is the installable `delhi_psi` package
  (`config`, `io`, `validate`, `geometry`, `neighbors`, `index`, `pipeline`,
  `cli`, `verify`) built with hatchling, with a `delhi-psi
  {preprocess,compute} --config <profile>` CLI. Every methodology choice —
  adjacency rule, barrier rule and layer combination, decay, roads formula,
  denominator, second normalization, exclusion `stage` × `absent_neighbor` —
  is a validated config value; two profiles ship (`code-2025` = today's
  behaviour, `manuscript` = the paper's rule-set). **No numbers changed**:
  `tests/fixtures/oraculum/production/code-2025.csv` was snapshotted from the
  pre-refactor code before any module moved and is reproduced byte-for-byte by
  the refactored pipeline, and
  `scripts/verify_against_baseline.py --config code-2025` reports zero
  deviation from the July 2025 baseline (real-data proof, 27 Aug 2026:
  `delhi-psi preprocess` — 4,357 settlements, 595 barrier-flagged, all five
  layers pass the validation battery; `delhi-psi compute` — 4,131 reported,
  15 missing-population rows; `verify_against_baseline.py` — max abs
  deviation `0.000e+00` on all 30 numeric output columns, `PASS — new run equivalent
  to July 2025 baseline within tolerance`). The `compute` stage now drops
  exact-duplicate service rows before validation, generalising
  `compute_psi.py`'s bank-only `drop_duplicates` to every service layer
  (only `bank` has duplicate rows on the real layers, 1,240 of 10,637). The
  silent `except: pass` in `calc_pcen_mobile` is now an explicit lookup
  miss, which is what makes `absent_neighbor: contributes` implementable
  (DEL-21). Root `conftest.py` and every `sys.path.insert` are gone
  (`tests/__init__.py` plus the editable install do that job); `pyyaml`
  moved to the runtime dependencies and `uv.lock` was regenerated. Notebook
  eyeball checks became raising assertions in `delhi_psi.validate` (DEL-25).
  Tests 77 → 230, including `test_config`, `test_profiles_match_reference`
  (both profiles × every scenario × both reference denominators),
  `test_manuscript_anchors` (the hand-ratified worksheet values),
  `test_production_fixtures`, `test_cli`, `test_validate`. Spec:
  `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md`.
  [DEL-15, DEL-16, DEL-18 (partial), DEL-21, DEL-22, DEL-25]
- Dead code: removed 17 functions with no callers (transitively) from
  `spatial_index_utils.py` — 684 lines, 44% of the module — including every
  `*_wards` / `*_buffer` variant, the unused `generate_colonies_with_exclusions`
  exclusion helpers, two superseded neighbor builders and two plotting helpers;
  pruned the `pickle`, `importlib.reload` and `matplotlib.pyplot` imports they
  alone used. CI now runs `uv run pytest -W error` (the suite has been
  warning-free since DEL-26); `tests/test_ci_workflow.py` pins it. [DEL-23]
- pandas 3: lifted the `pandas<3` cap (`pyproject.toml`), lock moved
  2.3.3 → 3.0.5. Five integer-sentinel column initializations
  (`spatial_index_utils.py` L812/L1090/L1161/L1203, `scripts/preprocess.py`
  L160) now start as float — pandas 3 raises `TypeError` where 2.x emitted the
  "incompatible dtype" `FutureWarning` and silently upcast. Oracle suite 77/77,
  fixtures byte-identical, warning count 364 → 0. Dependabot: 280 open alerts
  (all on `archive/master-2021/requirements.txt` or the deleted `poetry.lock`)
  dismissed as not-used. [DEL-26]
- CI: `.github/workflows/ci.yml` runs `uv sync --locked`, the oracle suite
  and a fixture-drift guard (regenerate `scripts/generate_*_fixtures.py`,
  `git diff --exit-code tests/fixtures/`) on every push to `main` and every
  PR. Drift guard uses `git status --porcelain` so untracked generator output also fails. Structural contract pinned by `tests/test_ci_workflow.py`; PyYAML added
  to the dev group. Spec: `docs/superpowers/specs/2026-08-24-ci-workflow-design.md`.
- Oracle worksheet hand-ratified by Bob (24 Aug 2026) against the April
  2026 manuscript's Eq. 1–4; Phase 2 fully closed. Added
  `docs/oracle/suggested-fixes-memo.md` (proposed fix per divergence,
  incl. new #7: distance unit unstated in the manuscript) and paper
  evidence for the roads and barrier items.

## [2026-08-17] Phase 2 — the mythical-city oracle (PR #6)

Verification: 65 tests green; production == independent reference
implementation == hand-derived anchors at 1e-12 across all scenarios and
both denominators; mutation testing (17+ sabotages of the index) confirms
the suite catches a broken index. Hand ratification of the derivation
worksheet is pending by design.

### Added
- Mythical-city oracle ("Oraculum"): hand-verifiable fixtures, an
  independent Eq. 1–4 reference implementation, and pytest suites
  establishing production == reference == hand-arithmetic agreement
  (`tests/fixtures/oraculum/`, `tests/reference_impl.py`,
  `tests/test_oracle*.py`, `tests/test_fixture_invariants.py`,
  `tests/test_divergence_exhibit.py`)
- Oracle maps, derivation worksheet (ratification pending), and the
  exclusion-semantics memo for Raj (`docs/oracle/`)
- Empirically pinned manuscript-vs-code divergences: directed bbox
  adjacency, global asymmetric barrier rule, neighbor-decayed roads,
  code-only `norm_psi` and popdensity denominator, and exclusion
  semantics (a) degenerating to (b) via silent exception swallowing
  (flagged for Phase 3 bug audit)

## [2026-08-17] Phase 1 — runnable pipeline on modern dependencies

Verification: fresh run reproduced the July 2025 baseline outputs with zero
numeric deviation (all columns, both PSI variants, all neighbor sets and
distances).

### Added
- Repo-scoped `/ship` build-and-ship pipeline command
  (`.claude/commands/ship.md`)
- Phase 1 design spec (`docs/superpowers/specs/2026-08-16-phase1-runnable-pipeline-design.md`)
- `scripts/preprocess.py`, `scripts/compute_psi.py` — pipeline as plain
  scripts with configurable `--data-dir`/`--out-dir` (flag > `DELHI_DATA_DIR`
  env var > `~/delhi_data`)
- `scripts/verify_against_baseline.py` — proves a fresh run matches the
  July 2025 baseline outputs
- pytest suite for path resolution and baseline comparison

### Changed
- Dependency management consolidated to uv + `pyproject.toml` (all packages
  at latest stable; Python 3.13). One documented exception: `pandas>=2.3,<3`
  — the pandas 3.0 major-version jump is deferred until the Phase 2 oracle
  can validate it
- Output filenames now dated `aug2026` (previously mislabeled `12Sep2021`)

### Removed
- Both Jupyter notebooks (logic now in `scripts/`; history preserved in git
  and `archive/master-2021/`)
- `requirements.txt`, `environment.yml`, `poetry.lock`, `Dockerfile`,
  `install_conda_environment.sh`

## [2026-08-16] Repository restructure (pre-phase)

### Changed
- `main` became the default branch (content from `bb_update`); 2020–2021
  code archived under `archive/master-2021/`; `master` and `bb_update`
  branches removed with history preserved in `main`
- README updated for the new layout

### Added
- `WORKPLAN.md` — sequenced plan toward HAS submission, with meta-planning
  decisions and open Raj/group questions
