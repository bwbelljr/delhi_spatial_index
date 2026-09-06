# Cycle 3E: partial-barrier weighting, overlap lending, min-max guard — design (DEL-48, DEL-20, bug-audit 6)

**Status:** draft for ultracode review, 5 Sep 2026. Written non-interactively
against the owner's accepted defaults of 5 Sep 2026 (recorded verbatim in
§ 0.2); every choice the brainstorm would have put to the owner is in the
decision log (§ 12) with its reasoning. Base: `main` at `3bf6341`.

**Cycle:** Phase 3E — the last code Phase 4 needs before the ratified profile
(DEL-31) and the recalculation (DEL-32). Same process and autonomy terms as
3A–3D (§ 11), with one change the owner made on 5 Sep 2026: **each ticket
ships on its own branch off `main` as its own PR** (§ 8).

**Why this cycle exists.** Raj's 28 Aug 2026 decisions
(`docs/decisions/2026-08-28-raj-methodology-decisions.md` §§ 4, 5, 12) leave
three things that are code, not config:

| item | ticket | what | how it lands |
|---|---|---|---|
| (a) | DEL-48 | `methodology.barrier.rule: partial_weighted` — a barrier covering part of a shared boundary discounts the neighbour's contribution by the covered share instead of severing it: w_ij = 1 − L_blocked/L_shared, linear, symmetric | a switch VALUE (the reserved one), with a reference rule, hand anchor, fixture rows, production implementation |
| (b) | DEL-20 | the overlap neighbour rule — a neighbour lends only the services NOT already inside the receiving settlement, so an overlap service is never counted twice for the same settlement | a switch VALUE; today's behaviour kept as the other value, because Raj has confirmed only the counting half, not the lending half |
| (c) | bug-audit 6 | `delhi_psi.index.minmax` divides 0/0 on a constant column | a guard that RAISES, naming the column |

---

## 0. Goals, non-goals, and the owner's accepted defaults

### 0.1 Goals

1. `partial_weighted` loads, is implemented on both sides, and is proven:
   a hand anchor on Oraculum with a genuinely fractional weight, production
   == reference at 1e-12, and unit tests on synthetic geometry for every
   limiting case (w = 0, w = 1, a point crossing, a half-covered edge, an
   overlapping pair, a MultiPolygon).
2. The overlap lending rule loads as a switch value with today's behaviour
   as the other value; both halves of memo § 6 are pinned on the messy
   city's `O1`/`O2` pair (Raj's counting half is already pinned and must
   not move; the lending half gains its pin).
3. `index.minmax` refuses a constant column with a message naming it.
4. **Every existing expected value stays byte-identical**: both cities'
   `expected_values.csv`, both cities' `production/*.csv`, and the real-data
   `code-2025` verify at 0.000e+00. This is the owner's hard condition. Both
   new switch values therefore ship as *variants* (new rows in
   `variants_expected_values.csv`), never as a change to the `ideal`/`code`
   rule-sets (§ 12 item 1).

### 0.2 The owner's accepted defaults (Bob, 5 Sep 2026 — decided, not reopened)

- Barrier buffer **5 m**, exposed as `methodology.barrier.buffer_m`.
- The "shared boundary" of an OVERLAPPING pair is the boundary of the
  intersection polygon.
- A barrier crossing a shared boundary at a point only (zero blocked
  length) severs nothing.
- The weight enters **linearly**.
- The overlap rule ships as a SWITCH VALUE with today's behaviour kept as
  the other value (Raj confirmed only that an overlap service counts for each
  owner; the lending half is Bob's proposal, still out for his answer).
- The min-max guard **raises** with a clear message naming the column.
- Fixture authority granted with a hard condition: the Oraculum canal may be
  redrawn and new geometry added to the messy city, but every existing
  expected value in `tests/fixtures/*/expected_values.csv` and the production
  fixtures stays byte-identical. Any change to an existing expected value is
  a hard stop.

Two of these turned out to need refinement rather than reopening; both are
recorded in § 12 and summarised for the owner at the end of § 12:

- The 5 m buffer is a **distance** definition (every boundary point within
  5 m of a barrier is blocked), which shapely's default round-capped buffer
  implements. On Oraculum the canal `[25, 475]` on the 500 m A–D edge
  therefore blocks `[20, 480]` — **w_AD = 0.08, not the memo's 0.1** (the
  memo ignored the buffer). Hand-verified in § 6.1.
- `buffer_m` must be **strictly > 0**: `LineString.buffer(0)` is EMPTY in
  shapely, so a 0 m buffer would make every weight 1 silently. The
  no-buffer meaning is not offered (§ 12 item 5).

### 0.3 Non-goals

- **No change to the shipped profiles' numbers.** `code-2025.yaml` and
  `manuscript.yaml` each gain ONE required key with today's value
  (`overlap: {lending: whole}`, § 1) and comments; nothing they compute
  moves.
- **No ratified profile.** DEL-31 writes it after this cycle and after Raj's
  two DEL-52 answers; whether it carries `overlap.lending: outside_receiver`
  depends on Raj's answer to the lending half.
- **No canal redraw, no new messy barrier, no third fixture city** (§ 12
  items 1–3). The owner's fixture authority is kept as a safety margin, not
  used.
- **No change to `pairwise`'s definition** beyond honouring
  `barrier.combine` (§ 2.6), which changes nothing on any fixture or any
  committed output.
- **No `absent_neighbor`, `exclusion`, `decay` or `adjacency` change.**

---

## 1. Config surface

```yaml
methodology:
  barrier:
    rule: partial_weighted      # global_asymmetric | pairwise | partial_weighted
    buffer_m: 5                 # required iff rule == partial_weighted; > 0 (metres,
                                # EPSG:7760): a boundary point within buffer_m of a
                                # barrier is blocked
    combine: any                # unchanged: any | [layer names]
  overlap:
    lending: whole              # whole | outside_receiver  (required, like every
                                # methodology key; `whole` names today's behaviour)
```

| key | value | meaning |
|---|---|---|
| `barrier.rule` | `partial_weighted` | for each stored link i→j, w_ij = 1 − L_blocked/L_shared (§ 2.1). j stays in i's list iff w_ij > 0; the contribution of j to i is w_ij · lent_ij · decay(d_ij). Symmetric by construction. |
| `barrier.buffer_m` | number > 0 | the blocked part of a shared boundary is the part within `buffer_m` metres of any selected barrier feature. Required by `partial_weighted`, rejected by the other two rules (they have no buffer: `global_asymmetric` uses the per-polygon flag, `pairwise` uses `intersects`). |
| `overlap.lending` | `whole` | a neighbour lends its whole amount S_j — today, unchanged |
| | `outside_receiver` | a neighbour lends \|S_j \ S_i\|: its amount minus the amount of the same service lying inside the receiver too (§ 3.1). Clean pairs are unaffected. |

Validation, in `config._methodology`, all `ConfigError`:

- `buffer_m` through `_conditional_number(barrier_raw, "buffer_m",
  "methodology.barrier", used_by="methodology.barrier.rule: partial_weighted",
  applies=rule == BarrierRule.PARTIAL_WEIGHTED, minimum=0, strict=True)` —
  verbatim the `exponent` / `scale_km` pattern. `_reject_unknown(barrier_raw,
  {"rule", "combine", "buffer_m"}, ...)`.
- `RESERVED_VALUES["methodology.barrier.rule"]["partial_weighted"]` is
  DELETED (the message ends "(cycle 3C)" and is stale);
  `REFERENCE_KNOBS["methodology.barrier.rule"]` gains
  `"partial_weighted": "partial_weighted"` (same spelling on both sides, like
  every 3D value, so `tests/variants.py` needs no translation layer).
- New enum key `methodology.overlap.lending` → `REFERENCE_KNOBS[...] =
  {"whole": "whole", "outside_receiver": "outside_receiver"}`, `ENUM_KEYS`,
  `ENUMS`, enum `OverlapLending`; `_reject_unknown(raw, {..., "overlap"},
  "methodology")`, `_reject_unknown(overlap_raw, {"lending"},
  "methodology.overlap")`. The key is **required** — "methodology has no
  defaults, never inherited" — so both shipped profiles AND
  `tests/test_config.py`'s `MINIMAL` string gain `overlap: {lending: whole}`
  (the three places the 3D `decay.distance` key had to appear;
  `test_defaults_equal_code_2025` requires the value `whole`).
- New reserved KEY `methodology.overlap.counting`: "reserved: Raj ratified on
  28 Aug 2026 that a service inside k overlapping colonies counts for each of
  the k — today's behaviour on both sides — so there is no knob (decision log
  § 5)". Same mechanism and same reason as `exclusion.minmax_universe`; it
  tells the next reader why only one half of memo § 6 is a switch.

Dataclasses: `BarrierConfig(rule, combine, buffer_m: float | None = None)`;
new `OverlapConfig(lending: OverlapLending)`; `MethodologyConfig` gains
`overlap: OverlapConfig`. Frozen, as today. The `None` default exists only
because the dataclass must hold "not applicable" for the two rules that
have no buffer; the YAML key itself is never defaulted.

Shipped profiles: `code-2025.yaml` gains the `overlap:` block, the comment
line `partial_weighted: reserved (spec 4)` becomes the allowed-values list
plus a `buffer_m` comment; `manuscript.yaml` gains the same block. Nothing
else in either file changes; both cities' production CSVs and the real-data
verify prove the outputs are unchanged.

---

## 2. Item (a): `partial_weighted` — the design

### 2.1 Definition

For a directed link i→j that survived adjacency:

1. `shared = geom_i ∩ geom_j` (shapely; handles overlaps, MultiPolygons,
   corner contacts, and the empty case for envelope-only `bbox` pairs).
2. The **shared boundary** `SB_ij` is the union of (i) the boundary of every
   polygonal component of `shared` (the owner's overlap rule: the boundary
   of the intersection polygon) and (ii) every linear component of `shared`.
   Point components contribute nothing. A GeometryCollection is decomposed
   into its parts first; this is the one place a naive `shared.boundary`
   would be wrong (shapely does not define `.boundary` for a collection),
   and it is a real-layer case: a polygon that overlaps its neighbour on one
   side and shares an edge on another.
3. `L_shared = SB_ij.length`. If `L_shared == 0` (empty intersection, or a
   corner-only contact), **w_ij = 1**: there is no boundary to block. This
   is exactly the owner's "a point crossing severs nothing", and it is also
   how `pairwise` treats an EMPTY intersection today (`intersects(empty)` is
   False). It differs from `pairwise` for a corner-only pair whose corner a
   barrier passes through (`pairwise` severs; `partial_weighted` keeps at
   w = 1) — recorded, and pinned by a unit test (§ 6.4).
4. `L_blocked = length( ∪_k (SB_ij ∩ buffer(b_k, buffer_m)) )` over the
   selected barrier features b_k — the union of the blocked PIECES, so two
   overlapping barrier buffers never count the same metre twice.
   `buffer(...)` is shapely's default (round caps): a boundary point is
   blocked iff its distance to a barrier is ≤ buffer_m.
5. `w_ij = 0.0 if L_blocked >= L_shared else 1 − L_blocked / L_shared`.
   The `>=` guards the float case where the intersection returns the whole
   boundary plus a rounding hair; w ∈ [0, 1] by construction.

Symmetry: `SB_ij` and the barriers are the same objects from either side,
so w_ij = w_ji exactly (same GEOS calls, same operands, same order of
union). A unit test asserts it on the synthetic cases, and the Oraculum
variant asserts `w(A,D) == w(D,A)` bit for bit.

### 2.2 Production: `delhi_psi/neighbors.py`

- `apply_barrier(polygon_gdf, barrier_geoms, *, id_col, neighbor_col, rule,
  flag_col, buffer_m=None, weight_col="nbrs_barrier_weight")`. The literal
  allowed-rule tuple gains `"partial_weighted"` (the message in
  `test_unknown_barrier_rule_raises_value_error` lists it). `buffer_m` is
  required by `partial_weighted` (ValueError otherwise) and rejected by the
  other two rules (ValueError, mirrors the config rule; `build_neighbors`
  forwards the configured value unconditionally and it is None there).
- Under `partial_weighted` the frame returned keeps the **existing
  contract**: `neighbor_col` is the pruned list of ids (j dropped iff
  w_ij == 0.0), so `centroid_distances`, `boundary_distances`,
  `verify.compare_neighbor_frames`, `apply_exclusion` and every
  `set(row[col])` test keep working unchanged. The weights travel in a NEW
  column `weight_col` = `[(neighbor_id, w), ...]` in the SAME order as the
  pruned list — the `nbrs_dist_bbox` 2-tuple shape. The column exists ONLY
  under `partial_weighted`; under the other two rules the frame is exactly
  what it is today (so `code-2025`'s artifact and outputs are byte-identical,
  and an artifact built before 3E loads).
- Helpers, public so the unit tests can hit them directly:
  `shared_boundary(geom_i, geom_j)` (§ 2.1 steps 1–2) and
  `partial_weight(shared_boundary, buffered_barriers)` (steps 3–5).
- Cost: the buffered barrier features are built ONCE (`[b.buffer(buffer_m)
  for b in barrier_geoms]`) and indexed with a `shapely.STRtree`; per link
  the intersection is taken only against the candidates the tree returns.
  On the real layer that is ~30k directed links (mean `bbox` degree ≈ 7)
  against 6,015 barrier features; a few minutes, comparable to
  `centroid_distances`' own per-link loop. Wall-clock is reported in the run
  step (§ 6.6). The naive alternative — one `unary_union` of all 6,015
  buffers and 30k overlays against that multipolygon — is refused in the
  design because each overlay would scale with the union's vertex count.

### 2.3 Production: `delhi_psi/pipeline.py`

- `NBRS_WEIGHT_COL = "nbrs_barrier_weight"` next to the other column names.
- `build_neighbors` passes `buffer_m=methodology.barrier.buffer_m` and
  `weight_col=NBRS_WEIGHT_COL`; logs one INFO line
  `barrier: rule=%s buffer_m=%s` before `apply_barrier`.
- `barrier_geoms` are taken from the layers `barrier.combine` selects (all
  when `any`), for BOTH geometry-based rules (§ 2.6).
- `methodology_stamp` adds `barrier.buffer_m` (None for the other rules):
  the buffer shapes the stored lists (a barrier 3 m off an edge blocks at
  5 m and not at 2 m), so an artifact built at another buffer must be
  refused with the existing message. `check_methodology_stamp` needs no
  change. Artifacts from 3A–3D lack the key: `stored.get(block,
  {}).get(key)` yields None, equal to the configured None for the two old
  rules, so `code-2025`'s pinned `colonies_neighbors.joblib` keeps loading
  (the 3D `max_distance_km` precedent; pinned by a test, § 6.5).
- `apply_exclusion` under `pre_neighbors` strips dropped ids from
  `NBRS_WEIGHT_COL` too, when the column is present.
- `index_frames` passes `nbr_weight_col=NBRS_WEIGHT_COL if
  methodology.barrier.rule == "partial_weighted" else None` to
  `index.service_index`, and **drops the column before returning**, exactly
  like `NBRS_DIST_BOUNDARY_COL`, so the CSV/shapefile column set is identical
  under every barrier rule. `io.SHAPEFILE_DROP_COLUMNS` gains the name anyway
  (it is also the list `compute` uses to strip list-valued columns from
  `missing_population.csv`, which is cut from the neighbours frame, where
  the column IS present).

### 2.4 Production: `delhi_psi/index.py`

`pcen(...)` and `service_index(...)` gain `nbr_weight_col=None`. In the
neighbour loop the contribution becomes

    poly_count += w * lent * _decay(nbr_dist, ...)

with `w = 1.0` when `nbr_weight_col is None`, else `weights[nbr_id]` from
`weights = dict(row[nbr_weight_col])` built once per row; an id in the
distance list with no weight is a `KeyError` (never a silent 1.0). `lent`
is `nbr_count` in this cycle's Group B and becomes § 3's adjusted amount in
Group C — **this line is the seam the two items share** (§ 8).

### 2.5 Reference: `tests/reference_impl.py`

- `partial_weights(nbrs, settlements, barriers, buffer_m) -> {(i, j): w}`
  over every directed link, implementing § 2.1 independently (its own
  `shared boundary` decomposition, its own union of blocked pieces; the
  naive all-buffers form is fine here — fixture cities have one barrier or
  none).
- `apply_barrier(nbrs, settlements, barriers, rule, buffer_m=None)`: new
  rule `"partial_weighted"` prunes `w == 0.0`; returns the same `{i: set}`
  shape as today, so no caller changes. `buffer_m` required by that rule,
  rejected by `"global"` / `"pair"` (ValueError — the mapped-knob test relies
  on an unimplemented combination raising).
- `compute_city(..., barrier_buffer_m=None, ...)`: when `barrier_rule ==
  "partial_weighted"`, `barrier_w = partial_weights(...)` and the neighbour
  sum multiplies by `barrier_w[(i, j)]`; otherwise the factor is 1 and the
  existing rows are byte-identical (checked by the drift guard and § 6.5).
- `RULESETS` unchanged: `ideal` keeps `barrier_rule="pair"`, `code` keeps
  `"global"`.

### 2.6 `barrier.combine` and the geometry-based rules

`build_neighbors` today hands `apply_barrier` the geometries of EVERY
configured layer, whatever `combine` says; `combine` only shapes the flag
column that `global_asymmetric` reads. For `pairwise` that means `combine:
[railway]` still severs across canals — a latent inconsistency nobody has
hit because every fixture has one layer and `code-2025` uses the flag rule.
This cycle makes `combine` select the LAYERS whose geometries the two
geometry-based rules see (all when `any`), which is what the stamp's
"`combine` decides who is severed" already claims. Changes nothing on any
fixture (one layer or none) or any committed output (`code-2025` is
`global_asymmetric`); pinned by a unit test with two layers and `combine`
naming one (§ 6.4). Recorded in § 12 item 8.

---

## 3. Item (b): `overlap.lending: outside_receiver` — the design

### 3.1 Definition

    PCEN_i = ( S_i + Σ_j  w_ij · decay(d_ij) · |S_j \ S_i| ) / denom_i

where, for a point service, `|S_j \ S_i| = amount_j − shared_ij` and
`shared_ij` is the number of that service's points that lie inside BOTH i
and j (under each side's own membership predicate: production's
boundary-inclusive `intersects`, the reference's strict `within` — they
agree on every point that is not exactly on a boundary, which is every point
in both fixture cities and, measured, every point on the real layer:
`docs/data/layer_pathologies.md`, rule-set gap #6). For the line service,
`shared_ij = Σ_roads length(road ∩ geom_i ∩ geom_j) / 1000` — the road
length inside the overlap. `shared_ij` is symmetric.

Under `whole` the term is `amount_j`, today's arithmetic, unchanged.

Why it is affordable: `shared_ij` is non-zero only for pairs that actually
share a service point (or road metres) — a subset of the 4,069 overlapping
pairs on the real layer (429 multi-settlement points across six services; 1
pair in the messy city; 0 in Oraculum). For every other pair `|S_j \ S_i| ==
|S_j|` and nothing is computed or stored. The structure is built from the
service points' containment, not from pair geometry, so it never touches the
~30k links.

### 3.2 The data structure and where it is built

`index.shared_amounts(polygon_gdf, service_gdf, *, kind, neighbor_col,
id_col) -> dict[(i, j), amount]`, sparse and symmetric (both orders
inserted, so `pcen` does one `shared.get((row_id, nbr_id), 0)` — the 0 is
the sparse representation, not a swallowed miss):

- `kind == "point"`: one `gpd.sjoin(polygon_gdf, service_gdf)` (the same
  join `point_counts` does); group by point; for every point with ≥ 2
  containing settlements, `+1` for every ordered pair among them. Neighbour
  status is irrelevant here (cheaper to record every sharing pair than to
  filter; `pcen` only looks up neighbour pairs).
- `kind == "line"`: for each i and each j in `row[neighbor_col]` with
  `amount_i > 0 and amount_j > 0`, `shared = geom_i ∩ geom_j`; if not empty,
  the summed clipped length of the road rows that intersect `shared`
  (STRtree prefilter). Pairs with an empty intersection are exactly 0 and
  skipped.

Built in `pipeline.index_frames`, compute-locally, only when
`methodology.overlap.lending == "outside_receiver"`, on the `amounts` frame
(the FULL universe, pre-exclusion — so under `absent_neighbor: contributes`
an excluded overlapping neighbour's lending is adjusted too), one dict per
service, passed as `shared_amounts=` to `index.service_index` → `pcen`.
Nothing is stored in the artifact and no column is added: the overlap rule
is downstream of the neighbour structure, like `decay` and `roads`, so it is
NOT in the methodology stamp and one artifact serves both values (pinned,
§ 6.5). Under `whole`, `shared_amounts=None` and the loop is byte-identical
to today.

### 3.3 `pcen`

    lent = nbr_count if shared_amounts is None else nbr_count - shared_amounts.get((row_id, nbr_id), 0)
    poly_count += w * lent * _decay(...)

`lent` cannot go negative: `shared_ij ≤ amount_j` by construction on both
sides. Asserted anyway (a negative `lent` is a bug, and
`validate.check_no_negative` would otherwise report it as a data problem).

### 3.4 Reference

`compute_city(..., overlap_lending="whole")`; `"outside_receiver"` builds
`shared[svc][(i, j)]` from `within` containment (points) and from
`road.intersection(idx[i]).intersection(idx[j]).length` (roads) and
subtracts inside the neighbour sum. Unknown value → ValueError.

### 3.5 What is deliberately NOT changed

- Own counts. Raj's half — a service in k overlapping colonies counts for
  each of the k — is today's behaviour on both sides and stays; the pin
  `test_the_overlap_clinic_is_counted_for_both_owners` must not move.
- Boundary points (gap #6): production and reference already disagree on
  whether a point exactly on a border is owned; this cycle does not touch
  that, and the definition above inherits each side's predicate. Zero such
  points exist on the real layer.

---

## 4. Item (c): the `hi == lo` guard

`index.minmax(polygon_gdf, *, source_col, target_col)`: after computing
`pcen_min` / `pcen_max`, if `pcen_max == pcen_min` raise

    ValueError(f"min-max of {source_col!r} is undefined: all {n} values equal {lo!r} "
               f"(max == min), so Eq. 2 divides 0/0. A constant column means every "
               f"reported settlement scores the same on this service — check the "
               f"service layer and the exclusion set upstream.")

- `ValueError`, like every other refusal in `index`; the CLI maps it to
  `pipeline error: ...`, exit 1. Under `-W error` today the same condition
  surfaces as numpy's `RuntimeWarning: invalid value encountered in scalar
  divide` turned into an error, with no column name; outside `-W error` it
  is a silent NaN column. The guard precedes the division in both cases.
- Covers every caller: each service's `service_index`, and `overall_psi`'s
  second normalisation (a one-settlement frame, or one where every
  `unnorm_psi` is equal, now raises naming `unnorm_psi`).
- `index.py`'s module docstring ("Two deliberate non-changes") and the
  `minmax` docstring are rewritten; the comment in `tests/test_pipeline.py`
  (`MISSING_ID` rationale) is updated to say the guard raises.
- **Reference ruling:** `tests/reference_impl.py`'s `0.0 if hi == lo` (two
  places) becomes the same `ValueError`. Reason: the reference is "the
  equations", Eq. 2 is undefined at hi == lo, and 0.0 is an invention that
  would be a rule-set divergence the moment anyone relied on it. It is
  unreachable through the committed fixtures either way
  (`scripts/check_oraculum_invariants.check` refuses a degenerate group, and
  the generators run it before writing). Consequence: a generator run on a
  degenerate city now fails inside `emit_expected_values` naming the column,
  instead of reaching `check`'s multi-violation report — acceptable, and
  `check` still guards the tied-anchor conditions it alone knows about.
  Recorded in § 12 item 9.

---

## 5. The variants table and the two proof cities

`tests/variants.py` gains three rows (config vocabulary, each block stated
IN FULL because `oracle_profile_path` replaces blocks wholesale):

| name | overrides | Oraculum | messy |
|---|---|---|---|
| `partial_5m` | `barrier: {rule: partial_weighted, combine: any, buffer_m: 5.0}` | **the fractional pin**: w_AD = 0.08; every other link w = 1 (§ 6.1) | degenerate — no barriers, so identical to the `code` base (stated, like `boundary` on Oraculum) |
| `overlap_outside` | `overlap: {lending: outside_receiver}` | degenerate — no overlaps, identical to `code` | **the lending pin**: O1/O2 clinic (§ 6.2) |
| `partial_5m_outside` | both of the above | = `partial_5m` | = `overlap_outside` |

The third row proves the two kwargs are accepted together and both paths run
in one `compute_frames`/`compute_city` call; the two multipliers are only
simultaneously non-trivial on the same pair in the synthetic tests (§ 6.4),
because no fixture city has a barrier across an overlap.

Plumbing that must grow (all pattern-following, nothing new in kind):

- `reference_impl.VARIANT_KNOBS` gains `("barrier", "rule"):
  "barrier_rule"`, `("barrier", "buffer_m"): "barrier_buffer_m"`,
  `("overlap", "lending"): "overlap_lending"`; `IGNORED_VARIANT_KEYS` gains
  `("barrier", "combine")` — the reference has no `combine` knob (it uses
  every barrier row, which is `any` on a one-layer city), with a comment
  saying so. Without these `_variant_overrides` raises "has no reference
  knob" — checked: nothing else in `_variant_overrides`, `VARIANT_RULESETS`
  or `check_bands` rejects a barrier/overlap variant; `check_bands` counts on
  `adjacency(...)`'s own output and never sees a barrier rule.
- `tests/oraculum_fixtures.variant_methodology` gains `barrier` and
  `overlap` branches (`BarrierConfig(rule=..., combine=... , buffer_m=
  block.get("buffer_m"))`; `OverlapConfig(lending=...)`).
- `tests/test_config.py::test_every_variant_block_is_one_the_loader_accepts`'s
  `enum_key` gains `("barrier", "rule")` and `("overlap", "lending")`.
- `tests/test_profiles_match_reference.py`: `knob_for_key` gains
  `"methodology.overlap.lending": "overlap_lending"`; **`EXTRA_PARAMS`**
  (it exists — `tests/test_profiles_match_reference.py:100`; the survey's
  "there is no EXTRA_PARAMS table" was wrong) gains
  `("methodology.barrier.rule", "partial_weighted"): {"barrier_buffer_m": 5.0}`
  — the same constant the pins use.
- The invariants guard (`check`) runs on the regenerated variants CSV: verified
  by hand-model in advance that none of the three rows creates a tie or a
  degenerate group on either city (§ 6.1–6.2).

---

## 6. Proof

Tolerances as in every cycle: hand anchors at 1e-12 against closed forms;
production == reference at 1e-12; CLI leg at 1e-9.

### 6.1 Item (a): the Oraculum `partial_5m` anchor (reference side, `tests/test_variant_rules.py`; then the CSV)

Geometry (worksheet coordinates): A–D share the edge x ∈ [0, 500] at
y = 1000, L_shared = 500 m. The canal is `[25, 475]` at y = 1000. Its 5 m
round-capped buffer covers the edge for x ∈ [20, 480]: L_blocked = 460,
**w_AD = w_DA = 1 − 460/500 = 0.08** (float: 0.07999999999999996; the pin is
`pytest.approx(0.08, abs=1e-12)` plus `w(A,D) == w(D,A)` exactly). Every
other link is uncovered, w = 1, and — the barrier rule now being pairwise in
spirit — the `global` severing of A and D from everyone else is gone:

    partial_5m lists (bbox, pruned at w == 0 only — nothing is pruned):
    A:[B,D,E] B:[A,C,E,RV] C:[B,E,IND] RV:[B] D:[A,E] E:[A,B,C,D,IND] IND:[C,E]

Decays: 1 km → ½; 1.5 km → 0.4; √2 km → √2−1; A–D centroids are √5/2 km
apart → decay 1/(1+√5/2) = 0.4721359549995794.

Hand anchors, `pop`, the `code` base (decayed roads, `swallowed`, nothing
dropped):

| row | arithmetic | value |
|---|---|---|
| D clinic | (0 + 0.08·2·0.4721359550 [A] + 1·0.4 [E]) / 100 | 0.0047554175279993 |
| D school | (1 + 0.08·1·0.4721359550 [A] + 1·0.4 [E]) / 100 | 0.0143777087639997 |
| A school | (1 + 0.08·1·0.4721359550 [D] + 1·(√2−1) [E]) / 100 | 0.0145198443877306 |
| A clinic | unchanged — D owns no clinic: (2 + ½ + (√2−1))/100 | 0.0291421356237309 |
| D road (decayed) | (0 + 0.08·0.75·0.4721359550 [A] + 0.75·0.4 [E]) / 100 | 0.0032832815729997 |
| B clinic | A is back: (1 + 2·½ + 0 + 2·½ + 1·½)/200 — the `ideal` 0.0175, not `code`'s 0.0125 | 0.0175 |
| E clinic | (1 + 2·(√2−1) + ½)/300 — the `ideal` value | 0.0077614237491540 |

`popdensity` changes only the denominators (D and A have area 1.0, so their
rows are identical; E divides by 150). Min-max anchors under `partial_5m`:
clinic min C (0.00228553) and max IND (0.04), school min RV (0) and max IND
(0.04) — all unique, both denominators; no service column is constant.
Verified against a hand-model that first reproduced every committed
`code`-rule PCEN on both cities at 1e-12.

A second, cheaper pin makes the buffer visible: `partial_weights` on
Oraculum with `buffer_m = 1.0` gives w_AD = 0.096 (blocked `[24, 476]`),
and the point at which a hypothetical `buffer_m → 0` would give the memo's
0.1 is stated in the test's docstring — but 0 is refused (§ 0.2), so the
0.1 is never a pin.

These anchors go into `docs/oracle/derivation-worksheet.md` as a new section
"partial_weighted (variant `partial_5m`, 5 Sep 2026 — machine-derived,
ratification is the owner's)", below the RATIFIED content and clearly
labelled, so `tests/test_manuscript_anchors.py` (which quotes only the
ratified section) is untouched. Config doc § 5 step 3 is thereby satisfied.

### 6.2 Item (b): the messy `overlap_outside` pin (reference side; then the CSV; then production directly)

O1 `_rect(10000,0,11000,1000)` and O2 `_rect(10800,0,11800,1000)` overlap in
x ∈ [10800, 11000]; the one clinic at (10900, 500) is strictly inside both;
centroids (10500, 500) and (11300, 500) are 0.8 km apart → decay 1/1.8.
Scenario `nopop_only` drops `U` (swallowed). `bbox` lists: O1:[O2, U],
O2:[N, O1]; N owns no clinic.

| row | `whole` (today) | `outside_receiver` |
|---|---|---|
| O1 clinic_pcen (pop) | (1 + 1·1/1.8)/600 = 0.0025925925925926 | (1 + 0·1/1.8)/600 = 1/600 = 0.0016666666666667 |
| O2 clinic_pcen (pop) | (1 + 1·1/1.8)/700 = 0.0022222222222222 | 1/700 = 0.0014285714285714 |
| O1 school_pcen (pop) | (0 + 1·1/1.8)/600 | **unchanged** — O2's school at (11400, 500) is outside the overlap |
| O1 police_pcen (pop) | 1/600 | unchanged — O1's own police point is not in O2 |

The clinic moves and the school does not: that is the whole rule in one
pair. `popdensity` scales by area (both 1.0 km², identical values). Ties:
the clinic min stays `N` (pop) / `S` (popdensity), unique; no group is
constant. Production-side direct pin (`tests/test_messy_fixtures.py`, new
section "the overlap lending rule (DEL-20, second half)"): under the
`overlap_outside` variant methodology `O1.clinic_pcen == 1/600` and
`O1.clinic_count == 1` — and, in the SAME test, under `code-2025`
`O1.clinic_pcen == (1 + 1/1.8)/600`, so the today-pin and the switch-pin
sit side by side and the counting pin
`test_the_overlap_clinic_is_counted_for_both_owners` is unchanged.

### 6.3 Item (c): the guard

`tests/test_index.py`: `minmax` raises `ValueError` naming `source_col` on a
two-equal-value frame and on a one-row frame; `service_index` propagates it;
`overall_psi(second_normalization=True)` raises naming `unnorm_psi` on a
frame whose `*_idx` means are equal; the happy path `test_minmax_is_eq2` is
unchanged. `tests/test_reference_impl.py`: `compute_city` on a synthetic
two-settlement city with a constant column raises the same way (the
reference no longer emits 0.0). `-W error` is the suite's mode, so the test
that the guard fires BEFORE numpy's warning is simply that the exception
type is `ValueError` with the column name in it.

### 6.4 Unit tests on synthetic geometry (production side, `tests/test_neighbors.py`; reference side mirrors the numeric ones)

Hand-built frames, EPSG:7760, kilometre-scale rectangles like
`test_index.city_with_neighbours`:

1. Two squares sharing a 1000 m edge, barrier collinear over the whole edge
   → w = 0 exactly; the pair is pruned; the lists equal `pairwise`'s.
   ("`partial_weighted` on a fully covered edge is `pairwise`" — the second
   "full-coverage" variant the memo imagined is this unit test, not a
   fixture.)
2. Same pair, barrier collinear over the middle 500 m, buffer 5 →
   L_blocked = 510 (caps), w = 0.49; with the barrier over `[0, 500]` from
   the corner, L_blocked = 505, w = 0.495. The half-covered edge with a
   buffer is NOT 0.5 — pinned so nobody "fixes" it.
3. A barrier crossing the shared edge perpendicularly at its midpoint,
   buffer 5 → L_blocked = 10, w = 0.99, kept; `pairwise` severs the same
   pair. The owner's "a point crossing severs nothing" pin, and the one
   documented `pairwise` ≠ `partial_weighted` case.
4. No barrier within 5 m → w = 1; a barrier 4 m off the edge, parallel, over
   the whole edge → w = 0 (the buffer's purpose: a barrier drawn just off a
   sliver gap still blocks).
5. Overlapping pair (the messy O1/O2 shape) with a barrier along x = 10900
   from y = 0 to 1000: `shared` is the 200 × 1000 m strip, SB is its 2400 m
   perimeter, the barrier crosses it twice → L_blocked = 20, w = 1 − 20/2400
   = 0.991666…; and with the barrier along the strip's own long edge x =
   11000, L_blocked = 1010, w = 1 − 1010/2400. The owner's overlap rule,
   made numeric.
6. A two-part MultiPolygon neighbour touching a square along both parts'
   edges (a MultiLineString shared boundary), barrier over one part's edge
   → w = the uncovered share; and a mixed intersection (overlap on one side,
   shared edge on another — a GeometryCollection) → SB = polygon boundary ∪
   line, length summed.
7. Corner-only contact with the barrier through the corner → w = 1 (kept),
   `pairwise` severs (§ 2.1 step 3).
8. Symmetry: `w(i, j) == w(j, i)` exactly on every case above.
9. `combine` selects layers for `pairwise` and `partial_weighted`: two
   layers, `combine=("railway",)`, a canal over the edge → not severed /
   w = 1; `combine="any"` → severed / w = 0 (§ 2.6).
10. `buffer_m` rules: required by `partial_weighted`, rejected by the other
    two (ValueError naming `buffer_m`); config level: `buffer_m: 0`, `-1`,
    `true`, missing, and present under `pairwise` all `ConfigError` naming
    `methodology.barrier.buffer_m` (parametrised, joins
    `test_conditional_parameters_are_rejected_naming_the_key`).

Item (b) synthetic tests (`tests/test_index.py`): `shared_amounts` on three
squares where P and Q overlap and one clinic sits in P ∩ Q, another in P
only, a road running through P ∩ Q: `{(P,Q): 1, (Q,P): 1}` for the clinic,
the overlap's road length for the road, nothing for the disjoint third; then
`pcen` with `shared_amounts` gives Q `(own + (2 − 1)·decay)/pop`. And the
composition test: the same P/Q pair with `nbr_weight_col` carrying
w = 0.5 → `(own + 0.5·(2 − 1)·decay)/pop` — the one place both multipliers
act on one pair.

**Production == reference on synthetic geometry.** A test-built city (three
settlements: the overlapping P/Q pair plus a MultiPolygon R touching P; one
clinic layer with points in P∩Q, P, R; one road row through P∩Q and into R
so no service column is constant; a canal partially covering the P–R
boundary) is scored by BOTH `compute_frames` (`partial_5m_outside`-shaped
methodology, built with `MethodologyConfig` directly) and
`compute_city(barrier_rule="partial_weighted", barrier_buffer_m=5,
overlap_lending="outside_receiver", ...)`, and every `METRIC_MAP` column
present on both sides is compared at 1e-12. This is the fractional-weight ×
overlap × MultiPolygon case the fixture cities cannot carry without
changing an existing expected value (§ 12 item 3), and it costs no fixture
file.

### 6.5 Production == reference, byte-identity, and the artifact

- `tests/test_variants_match_reference.py`: the three new variants × both
  cities × both denominators, automatically (it iterates `VARIANTS`).
- CLI leg: `test_a_derived_variant_profile_runs_end_to_end` is parametrised
  with `partial_5m` added — YAML → `load_config` → `preprocess` (weights
  computed, stored, stamped with `buffer_m: 5.0`) → `compute` → CSV at 1e-9
  against the variants CSV; and the output CSV's column set equals
  `code-2025`'s (the weight column never leaves `index_frames`).
- Stamp tests (`tests/test_cli.py` 3D section): an artifact built at
  `buffer_m: 5.0` is refused by a config at `2.0` with a message naming
  `buffer_m`, `5.0` and `2.0`; an artifact built under `pairwise` is refused
  by `partial_weighted` (existing message, `rule`); a 3A–3D-shaped stamp
  without `buffer_m` still passes for the two old rules; changing ONLY
  `overlap.lending` does NOT invalidate an artifact (positive test, the
  decay precedent). The literal expected dict in
  `test_neighbors_artifact_carries_the_methodology_stamp` gains
  `"buffer_m": None`. `tests/test_measure_roads_access.py::
  test_the_methodology_stamp_does_not_carry_roads` keeps passing as written
  (it asserts the top-level key SET, still `{"adjacency", "barrier"}`);
  `stamp_forms()` there is a hand-built shape for a hypothetical `roads`
  entry, not a comparison against `methodology_stamp`, so it needs no edit.
- Byte-identity at EVERY task of every group: both cities'
  `expected_values.csv` and `production/*.csv` unchanged (the 3C/3D check),
  and the two `variants_expected_values.csv` files change ONLY by the
  addition of the new variants' rows (a test asserts the pre-existing
  `rule` blocks are byte-identical to `main`'s — implemented as "drop the
  new rules from the regenerated frame and compare to the committed file at
  the base commit" in the plan's verification step, not as a permanent
  test).
- Existing tests that flip, per item: **(a)** `test_reserved_partial_weighted`
  is deleted (the value is no longer reserved) and replaced by the loading
  tests above; the allowed-values message in
  `test_unknown_barrier_rule_raises_value_error` and the two stamp literals
  above are updated; nothing else moves. **(b)** none flips; the profiles and
  `MINIMAL` gain a key. **(c)** none flips (the fixtures were built
  non-degenerate for exactly this reason); comments are updated.
- Tests that MUST NOT move: `tests/test_manuscript_anchors.py` (all),
  `tests/test_fixture_invariants.py::test_canal_inside_ad_edge_touches_exactly_a_and_d`
  and `::test_road_lengths_and_canal_clearance` (the canal is not redrawn),
  `tests/test_messy_fixtures.py::test_the_overlap_clinic_is_counted_for_both_owners`,
  `tests/test_neighbors.py::test_bbox_adjacency_then_global_barrier_matches_production`
  and `::test_touch_adjacency_then_pairwise_barrier_matches_the_manuscript`,
  every `test_profile_matches_reference` case, every
  `test_fixture_is_regenerable` case.

### 6.6 Real data (data-gated; the run step, never CI)

1. `scripts/verify_against_baseline.py --config code-2025` → PASS
   0.000e+00 on the warm cache. Nothing on the default path changes: no
   weight column, `overlap.lending: whole`, and the guard cannot fire on a
   run that passed before.
2. The rule-effect measurement, `scripts/measure_rule_effects.py` (§ 7),
   run against `~/delhi_data/phase3_verify` into a scratch `--work-dir`:
   - block `partial_barriers`: a derived profile `code-2025` + `barrier:
     {rule: partial_weighted, combine: any, buffer_m: 5}` (everything else
     `code-2025`, `bbox` included, so the diff is attributable to the barrier
     rule alone), `preprocess` into the work dir (the stamp changes, so the
     proven artifact cannot be reused — unlike the roads run) and `compute`.
     Reports: directed links by weight class (`w == 1`, `0 < w < 1`,
     `w == 0`), links severed under `code-2025` (`global_asymmetric`) vs
     under the derived profile, settlements whose list changed, the number
     of fractional links and the median fractional w, `preprocess`
     wall-clock, and the one-factor per-type PSI shift (mean `unnorm_psi`
     and `norm_psi`, both denominators, the DEL-49 `one_factor` shape).
   - block `overlap_lending`: `code-2025` + `overlap: {lending:
     outside_receiver}`, `compute` only on the proven artifact (the stamp is
     unchanged — asserted before running, as the roads script does), reports
     the number of (pair, service) entries in `shared_amounts` per service
     (must be ≤ the 429 multi-settlement points'-worth of pairs
     `layer_pathologies.md` counts), the settlements whose any-service PCEN
     changed, and the one-factor per-type PSI shift.
   - What the run must show, stated before it runs: no NaN and no negative
     anywhere (`check_no_negative` passes; the guard does not fire); under
     `partial_barriers` the number of severed links FALLS relative to
     `code-2025` (the global rule severs every link into a flagged
     settlement; the partial rule severs only fully covered boundaries) and
     the fractional class is non-empty; under `overlap_lending` every changed
     PCEN falls or stays (lending is only ever reduced), and the changed set
     is contained in the settlements with an overlapping neighbour. A result
     outside these bounds is a stop, not a number to write down.
3. The numbers go into `docs/data/rule_effects.md` (fenced blocks, drift
   test) and, as one sentence each, into the decision log §§ 4 and 5 for
   the batched reply to Raj.

---

## 7. Docs and the measurement script

- `docs/methodology-config.md`: § 1 table — `barrier.rule` row loses
  "reserved", gains the `partial_weighted` definition and a `buffer_m` row;
  new `overlap.lending` row; the paragraph "One thing Bob added needs code,
  not config" becomes "…is the switch `overlap.lending`, `whole` until Raj
  confirms"; the "Reserved" paragraph drops `partial_weighted` and gains
  `overlap.counting`; § 4 lists the new proofs (§ 6.4 synthetic
  production-vs-reference, the messy lending pin); § 5 unchanged; a new § 7
  "Partial barriers and overlap lending: what the real layer showed" carries
  the run-step numbers and the buffer semantics (distance, round caps, > 0).
- `docs/oracle/derivation-worksheet.md`: the § 6.1 section, labelled as
  machine-derived pending ratification.
- `docs/oracle/messy-city.md`: "What is deliberately NOT here — Any rule
  change" is rewritten: the overlap lending pin now lives here as a variant;
  barriers are still absent, with the reason (§ 12 item 3).
- `docs/data/rule_effects.md` (new) + `scripts/measure_rule_effects.py`
  (new): reuses `_measure_common` (guard, `render`/`parse_block`,
  `load_settlements`) and `measure_roads_access`'s `derived_profile` shape
  (generalised to "change these methodology values"), `stage_artifacts`,
  and `measure_effect`'s per-type diff. Two labelled blocks, one per rule.
  Fixture-level tests on the messy city (the `overlap_lending` block's
  `shared_amounts` count is exactly 1 pair × 1 service there) and Oraculum
  (the `partial_barriers` weight classes are 18 / 2 / 0 directed links —
  `w == 1` / fractional / severed — under `partial_5m`: 10 `bbox` pairs,
  of which only A–D is fractional, in both directions),
  plus the data-gated drift test in the `test_layer_pathologies.py` pattern.
  Created in Group B with the barrier block; Group C adds the overlap block.
- `CHANGELOG.md` `[Unreleased]`: one entry per group.
- `WORKPLAN.md`: cycle 3E item ticked per group; bug-audit 6 → `[x]`;
  DEL-31's blocker list updated.
- Jira: DEL-48 → Done with the evidence comment (anchor, real-data counts);
  DEL-20 → Done for the code, with the comment that the ratified profile's
  value awaits Raj's answer; bug-audit 6 noted on DEL-48's parent.

---

## 8. Tasks — three independently shippable groups

The owner requires one branch and one PR per ticket. The spec stays one
document because the three items share `pcen`'s neighbour loop and the
variants plumbing, but each group below is self-contained, leaves the suite
green at its own merge point, and states exactly what it assumes from the
group before it. Order: **A → B → C**. A different split was considered — a
preparatory "per-pair machinery" group with no consumer — and rejected: it
would be testable only by unit tests with no fixture or reference proof, and
Group B is already the smallest group that both introduces and proves the
machinery (§ 12 item 12).

### Group A — the min-max guard (DEL-54, bug-audit 6)

**Branch** `del-54-minmax-guard` off `main`. **Ships first**; assumes
nothing. Ticket **DEL-54** was created 5 Sep 2026 — bug-audit item 6 never
had one, and the owner's rule is one branch and one PR per ticket.

Files: `delhi_psi/index.py` (guard + docstrings), `tests/test_index.py`
(§ 6.3), `tests/reference_impl.py` (two `0.0 if hi == lo` → raise),
`tests/test_reference_impl.py` (§ 6.3), `tests/test_pipeline.py` (comment),
`CHANGELOG.md`, `WORKPLAN.md` (bug-audit 6). No config, no profile, no
fixture, no doc beyond the changelog.

Tasks:
1. Tests first (§ 6.3), then the guard in `index.minmax`; docstrings.
2. Reference: the same guard; its test.
3. Full suite green; both cities' `expected_values.csv`,
   `variants_expected_values.csv`, `production/*.csv` byte-identical
   (nothing regenerates); real-data `code-2025` verify PASS (nothing on the
   path changes — run it anyway, it is the cycle's standing proof).

Green at merge: the whole suite under `-W error`; CI drift guard.

### Group B — DEL-48 partial-barrier weighting

**Branch** `del-48-partial-barriers` off `main` **after A merges**. Assumes
from A: the guard (its synthetic city must have no constant column — § 6.4
says how). Introduces the per-pair multiplier machinery.

Files: `delhi_psi/config.py` (`buffer_m`, enum member, reserved entry
deleted, `BarrierConfig`), `delhi_psi/neighbors.py` (§ 2.2),
`delhi_psi/index.py` (`nbr_weight_col`, § 2.4), `delhi_psi/pipeline.py`
(§ 2.3), `delhi_psi/io.py` (`SHAPEFILE_DROP_COLUMNS`),
`delhi_psi/profiles/code-2025.yaml` (comment only), `tests/reference_impl.py`
(§ 2.5, `VARIANT_KNOBS`, `IGNORED_VARIANT_KEYS`), `tests/variants.py`
(`partial_5m`), `tests/oraculum_fixtures.py` (`variant_methodology` barrier
branch), `tests/test_config.py` (delete `test_reserved_partial_weighted`;
`buffer_m` rejection rows; `enum_key`), `tests/test_neighbors.py` (§ 6.4
items 1–10), `tests/test_index.py` (weight path), `tests/test_variant_rules.py`
(§ 6.1 anchors), `tests/test_variants_match_reference.py` (CLI leg
parametrised with `partial_5m`), `tests/test_cli.py` (stamp literal +
§ 6.5 stamp tests), `tests/test_profiles_match_reference.py`
(`EXTRA_PARAMS`), `tests/test_reference_impl.py` (synthetic
production-vs-reference city, barrier part), both
`tests/fixtures/*/variants_expected_values.csv` (regenerated: new rows
only), `docs/oracle/derivation-worksheet.md`, `docs/methodology-config.md`,
`scripts/measure_rule_effects.py` + `docs/data/rule_effects.md` +
`tests/test_measure_rule_effects.py` (barrier block), `CHANGELOG.md`,
`WORKPLAN.md`.

Tasks:
1. Config: enum member, `buffer_m`, dataclass, reserved entry deleted; the
   `test_config` rows first (§ 6.4 item 10), `MINIMAL` untouched (no new
   required key in this group).
2. Reference: `partial_weights`, `apply_barrier` rule, `compute_city`
   kwarg; `VARIANT_KNOBS`/`IGNORED_VARIANT_KEYS`; `tests/variants.py`
   `partial_5m`; § 6.1 anchors on the reference; regenerate both variants
   CSVs (new rows only — verify the pre-existing blocks byte-identical);
   `check` passes.
3. Production: `neighbors` (§ 2.2 + § 2.6), `index.pcen` weight path,
   `pipeline` (column, stamp, exclusion strip, drop before return), `io`;
   § 6.4 unit tests; `variant_methodology`; `EXTRA_PARAMS`;
   `test_variants_match_reference` green on `partial_5m`; the synthetic
   production-vs-reference city (barrier part).
4. CLI leg and stamp tests (§ 6.5); `code-2025.yaml` comment; worksheet
   section; config doc.
5. `measure_rule_effects.py` (barrier block) + doc skeleton + fixture tests;
   the run step (§ 6.6 items 1–2 barrier part) by the controller, numbers
   into the doc and the decision log § 4; CHANGELOG/WORKPLAN/Jira DEL-48.

Green at merge: whole suite; drift guard; `expected_values.csv` and
`production/*.csv` byte-identical; variants CSVs changed by addition only;
real-data verify PASS; the `partial_barriers` block within its stated
bounds.

**Names Group C relies on** (exact): `index.pcen(..., nbr_weight_col=None)`
and the loop line `poly_count += w * lent * _decay(...)` with `lent =
nbr_count`; `pipeline.NBRS_WEIGHT_COL`; `reference_impl.compute_city(...,
barrier_buffer_m=None)` with the neighbour-sum factor `barrier_w[(i, j)]`;
`VARIANT_KNOBS` accepting a non-adjacency/decay block;
`variant_methodology`'s per-block branch pattern; `EXTRA_PARAMS`.

### Group C — DEL-20 overlap lending

**Branch** `del-20-overlap-lending` off `main` **after B merges**. Assumes
from B: the names above — specifically that `pcen`'s neighbour term is
already `w * lent * decay` so C only redefines `lent`, and that a
non-decay variant block already flows through `VARIANT_KNOBS` /
`variant_methodology` / `enum_key`. If B is not merged when C starts, C
stands alone with `lent * decay` and adds the three plumbing entries itself;
the composition test (§ 6.4, both multipliers on one pair) is the only test
that genuinely needs B.

Files: `delhi_psi/config.py` (`overlap` block, `OverlapLending`,
`OverlapConfig`, reserved key `overlap.counting`), `delhi_psi/index.py`
(`shared_amounts`, `pcen(..., shared_amounts=None)`, § 3.2–3.3),
`delhi_psi/pipeline.py` (`index_frames` builds and passes the dicts), both
shipped profiles (`overlap: {lending: whole}` + comment),
`tests/test_config.py` (`MINIMAL`, reserved key test, enum_key),
`tests/reference_impl.py` (§ 3.4, `VARIANT_KNOBS`), `tests/variants.py`
(`overlap_outside`, `partial_5m_outside`), `tests/oraculum_fixtures.py`
(`variant_methodology` overlap branch), `tests/test_index.py` (§ 6.4 item (b)
tests + composition), `tests/test_variant_rules.py` (§ 6.2 pins),
`tests/test_messy_fixtures.py` (the direct production pin, § 6.2),
`tests/test_cli.py` (changing only `overlap.lending` keeps the artifact
valid), `tests/test_profiles_match_reference.py` (`knob_for_key`),
`tests/test_reference_impl.py` (synthetic city gains the overlap part), both
`variants_expected_values.csv` (new rows only), `docs/methodology-config.md`,
`docs/oracle/messy-city.md`, `scripts/measure_rule_effects.py` +
`docs/data/rule_effects.md` (overlap block), `CHANGELOG.md`, `WORKPLAN.md`.

Tasks:
1. Config + profiles + `MINIMAL` + reserved key; tests first.
2. Reference: `overlap_lending`, shared amounts (points via `within`, roads
   via clipped length); the two variants; § 6.2 reference pins; regenerate
   variants CSVs (addition only); `check` passes.
3. Production: `index.shared_amounts`, `pcen` `lent`, `index_frames`
   wiring; § 6.4 (b) tests + composition; `variant_methodology`;
   `knob_for_key`; `test_variants_match_reference` green on both new
   variants; the synthetic city (overlap part); the messy direct pin; the
   artifact-stays-valid test.
4. Docs (config doc, messy-city doc); `measure_rule_effects.py` overlap
   block + tests; run step (§ 6.6 item 2 overlap part); decision log § 5
   sentence; CHANGELOG/WORKPLAN/Jira DEL-20.

Green at merge: whole suite; drift guard; `expected_values.csv` and
`production/*.csv` byte-identical (the profiles gained a key whose value is
today's behaviour — the production CSVs prove it); real-data verify PASS;
the `overlap_lending` block within its stated bounds.

---

## 9. Risks and open points (for reviewers)

- **Float exactness of w = 0 and w = 1.** Pruning uses `L_blocked >=
  L_shared`; full coverage on the synthetic cases gives `L_blocked ==
  L_shared` exactly (the intersection of a segment lying inside a polygon
  returns the segment), verified for the buffer-5 Oraculum-like case
  (`edge ∩ canal.buffer(5)` → exactly 460.0). A near-covered real edge may
  yield w ≈ 1e-9 rather than 0: the link is kept with a negligible weight,
  which is the honest value. Nothing rounds.
- **GEOS cap vertices.** The round cap's vertex at 180° has a sin(π)-sized
  y offset (≈ 1e-13 m), so blocked endpoints are exact to ~1e-16 in w;
  every anchor is asserted at 1e-12 and the CSV carries whatever GEOS gives,
  identically on both sides (same library, same calls).
- **Reference vs production union order.** Production unions the blocked
  PIECES from STRtree candidates; the reference may union all buffers first.
  Mathematically equal; numerically equal on every fixture (one barrier) and
  within 1e-12 on the synthetic cities (two barriers at most). If a
  synthetic case ever disagrees beyond 1e-12 the reference adopts the
  piece-union form — never the other way round.
- **Performance of the partial rule on the real layer** (§ 2.2): budgeted at
  minutes with the STRtree; measured in the run step. A `within_distance`
  10 km profile with `partial_weighted` (4.4 M links) is Phase 6's problem
  and is noted in the config doc, not solved here.
- **`combine` for geometry rules** (§ 2.6) is a behaviour change for
  `pairwise` with a non-`any` combine on real data — no profile, fixture or
  output uses that combination; called out in the CHANGELOG.
- **Raj has not confirmed the lending half.** `outside_receiver` is proven
  and available; `whole` is what both shipped profiles carry; DEL-31 chooses.

---

## 10. Definition of done (the cycle)

- Groups A, B, C merged in order, each with CI green and its own PR.
- Both cities' `expected_values.csv` and `production/*.csv` byte-identical
  to `3bf6341`; both `variants_expected_values.csv` files changed by the
  addition of exactly the rows for `partial_5m`, `overlap_outside`,
  `partial_5m_outside`.
- Real-data `code-2025` verify PASS 0.000e+00 after each group; the two
  rule-effect blocks measured, within their stated bounds, and in the docs.
- Worksheet section, config doc, messy-city doc, CHANGELOG, WORKPLAN, Jira
  per § 7.

---

## 11. Process and autonomy terms

Same as 3A–3D per group: spec → ultracode review (mixed models, adversarial
refuters) → plan → plan review ≤ 3 rounds → SDD with per-task review → final
whole-branch review → PR → merge, once CI and the proofs are green. A
CONFIRMED Critical governs over the plan. Fix forward, commit, push, merge:
yes. **Stop and ask** only for: a change to any existing expected value in
`expected_values.csv`, `production/*.csv`, or a pre-existing block of
`variants_expected_values.csv` (the owner's hard condition); a change to
either shipped profile's numbers; any write under `~/delhi_data` outside
`--work-dir`/`--out-dir`; a real-data block outside its § 6.6 bounds; and
any need to actually USE the fixture authority (canal redraw or new messy
geometry) — the design does not need it, so needing it means the design
broke. The batched reply to Raj is drafted after Group C from the final docs
and is never sent.

---

## 12. Decision log (choices made without the owner, with reasons)

1. **Both new values ship as variants, not as changes to the `ideal`/`code`
   rule-sets.** The memo's "redraw the canal" advice assumed `partial_weighted`
   would REPLACE `pair` as the ideal barrier rule. It need not: 3D shipped
   its knobs as `tests/variants.py` rows whose numbers land in
   `variants_expected_values.csv` as new rows, and `RULESETS` never moved.
   Doing the same here keeps every existing expected value byte-identical
   BY CONSTRUCTION — the owner's hard condition satisfied without touching a
   fixture — and gives a fractional pin on the hand-ratifiable city. Checked
   that nothing in `_variant_overrides`, `VARIANT_KNOBS`, `check_bands`,
   `variant_methodology` or `enum_key` rejects a barrier/overlap variant once
   the listed entries are added (§ 5). Adopted from the coordinator's
   correction of 5 Sep 2026.
2. **No canal redraw.** Left at `[25, 475]`, the canal gives w_AD = 0.08 —
   the only genuinely fractional weight either fixture city can produce, on
   the city whose arithmetic is hand-checkable. Redrawing it to the full
   edge would make `partial_5m` on Oraculum identical to `pairwise` and
   push the fractional case onto the messy city, where it cannot go (next
   item). The worksheet's ratified section is untouched; the manuscript
   anchors test is untouched; `test_canal_inside_ad_edge_touches_exactly_a_and_d`
   and `test_road_lengths_and_canal_clearance` hold as written. Recommended
   to the owner in place of the decision log's "redraw" ruling.
3. **No new messy barrier.** Any barrier that touches a settlement with a
   `bbox` neighbour changes that settlement's rows under the `code`
   (`global`) and `ideal` (`pair`) rule-sets — i.e. existing expected values
   — because the fixture's barrier file is shared by every rule-set. The only
   barrier that changes nothing is one touching an isolated settlement
   (`I`), which produces no fractional case. Adding a new settlement PAIR
   plus a barrier also changes existing rows (every min-max group gains a
   row, and `bbox` lists of nearby settlements may gain an id). So the
   owner's condition rules out option (i) and (ii) of the brief's point 9
   alike; the fractional-overlap-MultiPolygon proof lives in synthetic
   in-test geometry scored by both implementations (§ 6.4), which costs no
   fixture file and is the pattern `test_index.city_with_neighbours` already
   uses. Recommended to the owner in place of "add a partial-coverage pair
   to the messy city".
4. **The buffer is a distance, round caps.** "Within `buffer_m` of a
   barrier" is the definition a methods sentence can state; shapely's default
   buffer implements it, and it makes the caps extend the blocked span
   longitudinally (a canal ending 3 m short of a corner blocks the corner).
   Flat caps would give the memo's 0.1 on Oraculum but a different, harder
   sentence. Consequence recorded: w_AD = 0.08 under the owner's 5 m, and
   the memo's 0.1 is the buffer-free limit that is deliberately not offered.
5. **`buffer_m` strictly > 0.** `LineString.buffer(0)` is empty in shapely
   (verified), so 0 would silently make every weight 1; a special-cased
   line-on-line intersection for 0 would be a second code path with its own
   collinearity pitfalls. `strict=True` in `_conditional_number`, message
   names the key.
6. **Weights in a new artifact column, lists pruned only at w == 0.** The
   `nbrs_bbox` / `nbrs_dist_bbox` contract has a large blast radius
   (`verify`, `io`, exclusion stripping, every `set(row[col])` test); a
   separate `[(id, w)]` column that only `pcen` consumes, present only under
   `partial_weighted`, leaves `code-2025` byte-identical at the artifact
   AND output level. Stored in the artifact rather than computed at
   `compute` time because `compute` does not read barrier layers and should
   not start to (dedup cache, reprojection, and an 11-minute-class cost
   moving to the wrong stage). Dropped before `index_frames` returns so the
   output column set is rule-independent (the 3D boundary-column
   precedent).
7. **`buffer_m` joins the stamp; `overlap.lending` does not.** The buffer
   shapes the stored lists; lending is downstream. Both facts are pinned.
8. **`combine` selects layers for the geometry rules** (§ 2.6). The
   alternative — leave `pairwise` ignoring `combine` — would make
   `partial_weighted` the only rule that honours a key the stamp claims
   shapes severing for every rule. Zero effect on anything committed.
9. **The reference raises on hi == lo too.** A reference that emits 0.0
   where production raises is a rule-set gap waiting to be relied on; the
   equations do not define the value. Unreachable through the fixtures
   (the invariants guard), so the cost is nil.
10. **Switch names.** `overlap.lending: whole | outside_receiver` — a block,
    not a scalar, so the ratified counting half has a home as a RESERVED key
    (`overlap.counting`) explaining why it is not a switch; `whole` names
    today's arithmetic in one word; `outside_receiver` says exactly what is
    lent. Considered `all`/`unshared` and `full`/`exclusive`; rejected as
    less literal.
11. **The line service is covered by the overlap rule** (road length inside
    the overlap is not lent twice), not only points, so the rule is one
    sentence for every service; under the ratified `roads: eq4_own_only`
    it is moot for roads, under `decayed` it is consistent.
12. **Three groups, A → B → C, no preparatory machinery group.** Per the
    owner's per-ticket-PR requirement. A is trivially independent. B is
    the smallest set that introduces AND proves the per-pair multiplier.
    C's true dependency on B is one line of `pcen` and three plumbing
    entries, stated in § 8 so C can be rebased onto A alone if B stalls.
13. **A rule-effects script rather than a doc-only note.** The decision log
    (§§ 4, 5) asks for the one-factor quantification "as for roads"; the
    roads pattern is a re-runnable script with a drift-tested doc, and its
    helpers exist. Kept to two blocks and reuse of `_measure_common` /
    `measure_roads_access` helpers; the owner may cut it to a config-doc
    note if PR size matters — everything else in the groups is independent
    of it.
14. **`EXTRA_PARAMS` exists.** The brief said it does not; it is at
    `tests/test_profiles_match_reference.py:100` and is the right place for
    `partial_weighted`'s buffer, so it is used.

**For the owner — where the accepted defaults are refined, not reopened:**
(i) w_AD on Oraculum is **0.08** under the 5 m buffer, not the memo's 0.1;
(ii) `buffer_m` must be **> 0**; (iii) "redraw the canal" and "add a
partial pair to the messy city" are **not needed and not done** — the
variant design satisfies the byte-identity condition without them, and the
messy-city option is actually incompatible with that condition (item 3);
(iv) the brief's "no `EXTRA_PARAMS` table" was wrong.
