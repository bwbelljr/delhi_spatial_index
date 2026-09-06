# DEL-48 — partial-barrier weighting (`barrier.rule: partial_weighted`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** A barrier that covers part of the boundary two settlements share
discounts that neighbour's contribution by the covered share
(`w_ij = 1 − L_blocked/L_shared`, linear, symmetric) instead of severing the
link, as a new value of `methodology.barrier.rule` with a reference rule, a
hand anchor on Oraculum, fixture rows, and a production implementation that
reproduces the reference at 1e-12.

**Architecture:** `partial_weighted` ships as a *variant*, never as a change
to the `ideal`/`code` rule-sets, so every committed expected value stays
byte-identical by construction. Production computes the per-link weights once
in `preprocess` (`neighbors.apply_barrier`), stores them in a new artifact
column `nbrs_barrier_weight` that exists ONLY under this rule, and
`index.pcen` multiplies each neighbour's contribution by its weight. The
column is dropped before `index_frames` returns, so the CSV/shapefile column
set is rule-independent. The independent reference implements § 2.1 from the
definition, with its own decomposition and its own union of blocked pieces.

**Tech Stack:** Python 3.13, shapely 2.1.2 (`buffer`, `unary_union`,
`STRtree`), geopandas/pandas, pytest under `-W error`, uv.

**Spec:** `docs/superpowers/specs/2026-09-05-cycle-3e-partial-barriers-design.md`
— § 1 (config surface), § 2 in full (2.1 definition, 2.2 neighbors, 2.3
pipeline, 2.4 index, 2.5 reference, 2.6 combine), § 5 (variants table), § 6.1
(the Oraculum anchors), § 6.4 (unit tests), § 6.5 (byte-identity, the stamp),
§ 6.6 (real data), § 8 Group B (scope), § 12 (decisions already made — do not
reopen them). **Read § 2 and § 6.1 before starting.**

## Global Constraints

- **This is Group B only.** Group A (the min-max guard) shipped as DEL-54 and
  is already on `main` at `bcc557c`; Group C (DEL-20 overlap lending) is a
  later branch. Do not implement `overlap.lending`, `OverlapConfig`,
  `index.shared_amounts`, or the `overlap_outside` / `partial_5m_outside`
  variants here.
- **Branch `del-48-partial-barriers` off `main` at `bcc557c`.** One branch,
  one PR, one ticket.
- **No existing expected value may move.** Both cities'
  `tests/fixtures/*/expected_values.csv` and `tests/fixtures/*/production/*.csv`
  must be byte-identical to `bcc557c` at every commit. The two
  `variants_expected_values.csv` files may change ONLY by the ADDITION of
  `partial_5m` rows (Task 6 gives the exact check). Any other difference is
  the spec's stop-and-ask condition (§ 11): STOP and report, never regenerate
  around it.
- **No canal redraw, no new messy geometry, no third fixture city** (spec
  § 12 items 2–3). The design does not need the owner's fixture authority;
  needing it means the design broke.
- **`buffer_m` is strictly > 0** (spec § 12 item 5): `LineString.buffer(0)`
  is EMPTY in shapely, so 0 would silently make every weight 1. The
  no-buffer meaning is not offered.
- **The buffer is a DISTANCE with round caps** (spec § 12 item 4): a boundary
  point is blocked iff its distance to a barrier is ≤ `buffer_m`. On Oraculum
  that gives **w_AD = 0.08**, not the memo's 0.1. Do not "fix" this.
- **`w = 1` on a zero-length shared boundary** (empty intersection, or a
  corner-only contact). This is the one documented case where
  `partial_weighted` differs from `pairwise`, which severs a corner-only pair
  whose corner a barrier passes through. Pinned, not accidental.
- **Weights travel in a separate column, lists are pruned only at `w == 0`**
  (spec § 12 item 6). `neighbor_col` keeps its existing contract, so
  `centroid_distances`, `boundary_distances`, `verify.compare_neighbor_frames`,
  `apply_exclusion` and every `set(row[col])` test keep working unchanged.
- **`buffer_m` joins the methodology stamp; nothing else does** (spec § 12
  item 7). An artifact built at another buffer must be refused.
- The suite runs `uv run pytest -q -W error`. Pass an **explicit `timeout` of
  600000 ms** on that Bash call and do NOT set `run_in_background`: a call
  without an explicit timeout is auto-backgrounded after 120 s and its
  completion never reaches you. The suite is ~10–15 minutes on this machine
  (real data present). Read the summary line before committing.
- Commit trailers, both lines, on every commit:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6`
- Commit message prefixes: `feat(barrier)` for production/config changes,
  `test(barrier)` for reference/test-only changes, `docs(barrier)` for docs.
- **Placeholders are forbidden everywhere except Task 11**, the controller's
  real-data run, where bracketed `[from the run]` marks numbers that do not
  exist until the run happens.

## Task order and why it is this one

The spec's § 8 Group B lists config first and the reference second. **That
order breaks the suite**, so this plan swaps them. The constraint:

`tests/test_profiles_match_reference.py::test_every_mapped_knob_is_one_the_reference_actually_implements`
iterates `REFERENCE_KNOBS[key].items()` and calls
`compute_city(**{knob: reference_value})` once per mapped value. The moment
`REFERENCE_KNOBS["methodology.barrier.rule"]` gains
`"partial_weighted": "partial_weighted"` (which is also what generates the
`BarrierRule.PARTIAL_WEIGHTED` enum member the loader needs), that test calls
`compute_city(barrier_rule="partial_weighted", ...)`. If the reference has no
such rule it raises `ValueError(rule)` and the test fails. So:

1. **The reference rule must land BEFORE the `REFERENCE_KNOBS` entry.**
2. **`EXTRA_PARAMS[("methodology.barrier.rule", "partial_weighted")] =
   {"barrier_buffer_m": 5.0}` must land in the SAME commit as the
   `REFERENCE_KNOBS` entry** — the reference *requires* `barrier_buffer_m`
   for that rule, so without it the same test fails with "requires
   barrier_buffer_m".

A second coupling fixes Task 6: adding a row to `tests/variants.py` is the
LAST link, because four separate tests read that table and each one fails on
a row that is not yet everywhere —
`test_config.py::test_every_variant_block_is_one_the_loader_accepts` (loader),
`test_reference_impl.py::test_variants_csv_has_one_scenario_and_every_variant`
and `::test_variants_expected_values_csv_is_regenerable` (the committed CSVs),
and `test_variants_match_reference.py::test_production_matches_the_reference_on_each_variant`
(the whole production path). So the variants row, `variant_methodology`'s
barrier branch and the two regenerated CSVs are one commit, and it comes
after config, neighbors, index and pipeline are all in place.

Resulting order:

| # | task | prefix |
|---|---|---|
| 1 | Reference: `partial_weights`, the rule, the Oraculum anchors | `test(barrier)` |
| 2 | Config: the `partial_weighted` value and `buffer_m` | `feat(barrier)` |
| 3 | Production geometry: `shared_boundary`, `partial_weight`, `apply_barrier`, `combine` | `feat(barrier)` |
| 4 | `index.pcen` / `service_index`: the weight path | `feat(barrier)` |
| 5 | Pipeline wiring: the column, the stamp, the exclusion strip, the drop | `feat(barrier)` |
| 6 | The `partial_5m` variant: table, methodology branch, regenerated CSVs | `test(barrier)` |
| 7 | The CLI leg and the stamp's `buffer_m` | `test(barrier)` |
| 8 | Production == reference on synthetic geometry | `test(barrier)` |
| 9 | Docs: worksheet, config doc, CHANGELOG, WORKPLAN | `docs(barrier)` |
| 10 | `scripts/measure_rule_effects.py` (barrier block) + doc + tests | `feat(barrier)` |
| 11 | **Controller:** the real-data run and its numbers | `docs(barrier)` |

---

### Task 1: The reference rule — `partial_weights`, `apply_barrier`, `compute_city`

**Files:**
- Modify: `tests/reference_impl.py` (`apply_barrier` at :127-148, `compute_city`
  at :180-291, `VARIANT_KNOBS` at :42-50)
- Test: `tests/test_reference_impl.py` (synthetic geometry unit pins)
- Test: `tests/test_variant_rules.py` (the § 6.1 Oraculum anchors)

**Interfaces:**
- Consumes: nothing. This task touches only the reference side, which never
  imports `delhi_psi` (the INDEPENDENCE RULE at the top of
  `tests/reference_impl.py`).
- Produces, and **Group C relies on these exact names**:
  - `reference_impl.partial_weights(nbrs, settlements, barriers, buffer_m)
    -> dict[(str, str), float]` — one entry for EVERY directed link in
    `nbrs`, both orders, `w ∈ [0, 1]`.
  - `reference_impl.apply_barrier(nbrs, settlements, barriers, rule,
    buffer_m=None) -> dict[str, set]` — unchanged return shape.
  - `reference_impl.compute_city(..., barrier_buffer_m=None, ...)` with the
    neighbour-sum factor `barrier_w[(i, j)]`.
  - `VARIANT_KNOBS` accepting a non-adjacency/decay block:
    `("barrier", "rule"): "barrier_rule"`,
    `("barrier", "buffer_m"): "barrier_buffer_m"`;
    `IGNORED_VARIANT_KEYS` gains `("barrier", "combine")`.

- [ ] **Step 1: Write the failing unit pins for the shared boundary and the weight**

Append to `tests/test_reference_impl.py` (the file imports `pytest` and
`math`; `gpd`, `box`, `LineString`, `MultiPolygon` are imported locally in
tests here — follow that local-import style):

```python
# --- DEL-48: partial_weighted, reference side (spec § 2.1, § 6.4) ------
def _pair_city(geom_a, geom_b):
    """Two settlements A and B with the given geometries, no services."""
    import geopandas as gpd

    return gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "population": [100.0, 200.0],
         "area_km2": [1.0, 1.0]},
        geometry=[geom_a, geom_b], crs="EPSG:7760")


def _barrier_frame(*geoms):
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=list(geoms), crs="EPSG:7760")


def _w(geom_a, geom_b, *barrier_geoms, buffer_m=5.0):
    """w(A, B) under partial_weighted, asserted symmetric."""
    from tests.reference_impl import partial_weights

    weights = partial_weights({"A": {"B"}, "B": {"A"}},
                              _pair_city(geom_a, geom_b),
                              _barrier_frame(*barrier_geoms), buffer_m)
    assert weights[("A", "B")] == weights[("B", "A")], "w must be symmetric"
    return weights[("A", "B")]


def test_reference_partial_weight_on_a_fully_covered_edge_is_zero():
    """A barrier lying along the whole shared edge blocks all 1000 m, so
    partial_weighted agrees with pairwise: the pair is severed."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(1000, 0), (1000, 1000)])) == 0.0


def test_reference_partial_weight_on_a_half_covered_edge_is_not_one_half():
    """The 5 m round caps extend the blocked span 5 m past each end, so the
    middle 500 m of a 1000 m edge blocks 510 m, not 500. Pinned so nobody
    'fixes' 0.49 into 0.5."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(1000, 250), (1000, 750)])) == pytest.approx(
                  0.49, abs=1e-12)
    # from the corner: one cap falls off the end of the edge, so 505 m
    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(1000, 0), (1000, 500)])) == pytest.approx(
                  0.495, abs=1e-12)


def test_reference_a_perpendicular_crossing_blocks_only_the_buffer():
    """The owner's 'a point crossing severs nothing': a barrier crossing the
    shared edge at right angles blocks 2 x 5 m, so w = 0.99 and the link is
    KEPT — where pairwise severs it outright."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(900, 500), (1100, 500)])) == pytest.approx(
                  0.99, abs=1e-12)


def test_reference_a_barrier_just_off_the_edge_still_blocks_it():
    """The buffer's purpose: a canal drawn 4 m off a sliver gap is within
    5 m of every boundary point, so w = 0. A barrier 200 m away is not."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(996, 0), (996, 1000)])) == 0.0
    assert _w(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
              LineString([(1200, 0), (1200, 1000)])) == 1.0


def test_reference_an_overlapping_pair_uses_the_intersection_boundary():
    """The owner's overlap rule made numeric: O1 and O2 overlap in a
    200 x 1000 m strip whose BOUNDARY is its 2400 m perimeter. A barrier
    crossing the strip end to end cuts that perimeter twice (20 m); a barrier
    lying along the strip's own long edge blocks 1000 m + 2 caps."""
    from shapely.geometry import LineString, box

    o1, o2 = box(10000, 0, 11000, 1000), box(10800, 0, 11800, 1000)
    assert _w(o1, o2, LineString([(10900, 0), (10900, 1000)])) == \
        pytest.approx(1 - 20 / 2400, abs=1e-12)
    assert _w(o1, o2, LineString([(11000, 0), (11000, 1000)])) == \
        pytest.approx(1 - 1010 / 2400, abs=1e-12)


def test_reference_a_multipolygon_neighbour_sums_both_shared_edges():
    """A two-part neighbour shares 400 m along each part, so L_shared is
    800 m; a barrier over one part's edge blocks 400 of them."""
    from shapely.geometry import LineString, MultiPolygon, box

    multi = MultiPolygon([box(1000, 0, 2000, 400), box(1000, 600, 2000, 1000)])
    assert _w(box(0, 0, 1000, 1000), multi,
              LineString([(1000, 0), (1000, 400)])) == pytest.approx(
                  0.5, abs=1e-12)


def test_reference_a_mixed_intersection_is_decomposed_part_by_part():
    """The one place a naive `shared.boundary` is WRONG: a neighbour that
    overlaps on one side and shares an edge on another intersects in a
    GeometryCollection, whose `.boundary` is None in shapely 2.1. SB is the
    overlap polygon's 1000 m perimeter plus the 400 m shared line."""
    from shapely.geometry import MultiPolygon, box

    mixed = MultiPolygon([box(900, 0, 1900, 400), box(1000, 600, 1900, 1000)])
    square = box(0, 0, 1000, 1000)
    assert square.intersection(mixed).geom_type == "GeometryCollection"
    assert square.intersection(mixed).boundary is None
    assert _w(square, mixed) == 1.0          # no barrier: SB length 1400, w 1


def test_reference_a_corner_only_contact_is_never_severed():
    """L_shared == 0, so there is no boundary to block and w = 1 even with a
    barrier straight through the corner (spec § 2.1 step 3). pairwise severs
    this pair; this is the documented difference between the two rules."""
    from shapely.geometry import LineString, box

    assert _w(box(0, 0, 1000, 1000), box(1000, 1000, 2000, 2000),
              LineString([(900, 1100), (1100, 900)])) == 1.0


def test_reference_partial_weighted_requires_a_positive_buffer():
    from shapely.geometry import box

    from tests.reference_impl import apply_barrier

    city = _pair_city(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000))
    nbrs = {"A": {"B"}, "B": {"A"}}
    with pytest.raises(ValueError, match="barrier_buffer_m"):
        apply_barrier(nbrs, city, _barrier_frame(), "partial_weighted")
    with pytest.raises(ValueError, match="barrier_buffer_m"):
        apply_barrier(nbrs, city, _barrier_frame(), "partial_weighted",
                      buffer_m=0)


@pytest.mark.parametrize("rule", ["global", "pair"])
def test_reference_the_other_rules_reject_a_buffer(rule):
    """An unimplemented combination must RAISE — the mapped-knob test relies
    on it."""
    from shapely.geometry import box

    from tests.reference_impl import apply_barrier

    city = _pair_city(box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000))
    with pytest.raises(ValueError, match="barrier_buffer_m"):
        apply_barrier({"A": {"B"}, "B": {"A"}}, city, _barrier_frame(), rule,
                      buffer_m=5.0)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_reference_impl.py -k reference_partial or reference_a_ or reference_the_other`

Expected: **FAIL with `ImportError: cannot import name 'partial_weights' from
'tests.reference_impl'`** on every test that imports it, and
`TypeError: apply_barrier() got an unexpected keyword argument 'buffer_m'`
on the two rejection tests. That is the RED reason: neither the function nor
the parameter exists yet. Record the exact text.

- [ ] **Step 3: Implement the reference rule**

In `tests/reference_impl.py`, add `from shapely.ops import unary_union` to
the imports (the module already imports `from shapely.geometry import box`),
then insert these two functions immediately above `apply_barrier`:

```python
def _shared_boundary(geom_i, geom_j):
    """SB_ij: the boundary of every POLYGONAL component of the intersection
    (the owner's overlap rule) plus every LINEAL component. Points contribute
    nothing, and a GeometryCollection is decomposed part by part — shapely
    does not define `.boundary` for a collection, and a polygon that overlaps
    its neighbour on one side and shares an edge on another produces exactly
    that.
    """
    shared = geom_i.intersection(geom_j)
    if shared.is_empty:
        return None
    parts = list(shared.geoms) if hasattr(shared, "geoms") else [shared]
    pieces = []
    for part in parts:
        if part.geom_type in ("Polygon", "MultiPolygon"):
            pieces.append(part.boundary)
        elif part.geom_type in ("LineString", "LinearRing", "MultiLineString"):
            pieces.append(part)
    return unary_union(pieces) if pieces else None


def partial_weights(nbrs, settlements, barriers, buffer_m):
    """{(i, j): w_ij} for every DIRECTED link in `nbrs` (spec § 2.1).

    w_ij = 1 - L_blocked / L_shared, where L_blocked is the length of the
    shared boundary within `buffer_m` metres of any barrier feature. A
    zero-length shared boundary (empty intersection, or a corner-only
    contact) has nothing to block, so w = 1.

    Independent of production by construction: its own decomposition, and it
    unions ALL the barrier buffers ONCE rather than intersecting piece by
    piece against STRtree candidates. Both fixture cities carry one barrier
    row or none, so the union is trivial here (spec § 9).
    """
    if buffer_m is None or not buffer_m > 0:
        raise ValueError(
            "barrier rule 'partial_weighted' requires barrier_buffer_m > 0, "
            f"got {buffer_m!r}")
    idx = settlements.set_index("USO_AREA_U").geometry
    geoms = ([] if barriers is None or len(barriers) == 0
             else list(barriers.geometry))
    blocked_area = (unary_union([g.buffer(buffer_m) for g in geoms])
                    if geoms else None)
    out = {}
    for i, js in nbrs.items():
        for j in js:
            shared = _shared_boundary(idx[i], idx[j])
            length = 0.0 if shared is None else shared.length
            if length == 0 or blocked_area is None:
                out[(i, j)] = 1.0
                continue
            blocked = shared.intersection(blocked_area).length
            out[(i, j)] = (0.0 if blocked >= length
                           else 1 - blocked / length)
    return out
```

Then replace `apply_barrier` (:127-148) with:

```python
def apply_barrier(nbrs, settlements, barriers, rule, buffer_m=None):
    """Sever (or, under partial_weighted, prune at w == 0) neighbour links.

    Returns the same {i: set} shape under every rule, so no caller changes.
    The rule and buffer_m are validated BEFORE the empty-barriers
    short-circuit — a city with no barriers must still refuse a bad
    combination, which is what production does and what the messy city (no
    barriers at all) exercises.
    """
    if rule not in ("global", "pair", "partial_weighted"):
        raise ValueError(rule)
    if rule == "partial_weighted":
        weights = partial_weights(nbrs, settlements, barriers, buffer_m)
        return {i: {j for j in js if weights[(i, j)] > 0.0}
                for i, js in nbrs.items()}
    if buffer_m is not None:
        raise ValueError(
            "barrier_buffer_m is only used by barrier rule "
            f"'partial_weighted', not {rule!r}")
    if barriers is None or len(barriers) == 0:
        return nbrs
    idx = settlements.set_index("USO_AREA_U").geometry
    barrier_geoms = list(barriers.geometry)
    flagged = {i for i in idx.index
               if any(idx[i].intersects(b) for b in barrier_geoms)}
    out = {}
    for i, js in nbrs.items():
        if rule == "global":
            out[i] = js - flagged
        else:
            kept = set()
            for j in js:
                shared = idx[i].intersection(idx[j])
                crossed = any(b.intersects(shared) for b in barrier_geoms)
                if not crossed:
                    kept.add(j)
            out[i] = kept
    return out
```

- [ ] **Step 4: Run the unit pins and see them pass**

Run: `uv run pytest -q -W error tests/test_reference_impl.py`
Expected: PASS, including the pre-existing anchors and both
`*_is_regenerable` tests — `compute_city` has not changed yet, so the CSVs
cannot have moved.

- [ ] **Step 5: Wire the multiplier into `compute_city`**

In `tests/reference_impl.py::compute_city`, add `barrier_buffer_m=None` to
the signature (immediately after `max_distance_km=None`, so the decay
parameters keep their positions), and replace the two lines

```python
    nbrs = apply_barrier(adjacency(universe, adjacency_rule, max_distance_km),
                         universe, barriers, barrier_rule)
```

with

```python
    adjacent = adjacency(universe, adjacency_rule, max_distance_km)
    nbrs = apply_barrier(adjacent, universe, barriers, barrier_rule,
                         barrier_buffer_m)
    # The weights are recomputed here rather than threaded out of
    # apply_barrier, which keeps its {i: set} contract. Both fixture cities
    # are seven and eleven settlements, so the second pass is free.
    barrier_w = (partial_weights(adjacent, universe, barriers,
                                 barrier_buffer_m)
                 if barrier_rule == "partial_weighted" else None)
```

and, in the neighbour sum, replace

```python
                decayed_sum += amounts[svc][j] * contribution_weight(i, j)
```

with

```python
                w = 1.0 if barrier_w is None else barrier_w[(i, j)]
                decayed_sum += w * amounts[svc][j] * contribution_weight(i, j)
```

`1.0 * x` is exact in IEEE 754 and multiplication is left-associative, so
`(1.0 * a) * b == a * b` bit for bit: the `ideal` and `code` rows cannot
move. Step 7 proves it rather than asserting it.

- [ ] **Step 6: Write the Oraculum anchors (spec § 6.1)**

Append to `tests/test_variant_rules.py`. Add `partial_weights` to the
existing `from tests.reference_impl import (...)` block. This file imports
nothing from `delhi_psi` — keep it that way.

```python
# --- DEL-48: partial_weighted on Oraculum (spec § 6.1) -----------------
# The canal is the segment [25, 475] at y = 1000, lying inside the 500 m
# A-D edge x in [0, 500]. Its 5 m round-capped buffer covers [20, 480], so
# L_blocked = 460 and w_AD = 1 - 460/500 = 0.08. (The memo's 0.1 ignored the
# buffer; buffer_m -> 0 would give it, and 0 is refused. Spec § 12 item 4.)
PARTIAL_5M = dict(RULESETS["code"], barrier_rule="partial_weighted",
                  barrier_buffer_m=5.0)
W_AD = 0.08
# A and D centroids are (500, 1500) and (0, 500): sqrt(5)/2 km apart.
D_AD_KM = math.sqrt(5) / 2
W_DECAY_AD = 1 / (1 + D_AD_KM)          # 0.4721359549995794
W_15 = 1 / 2.5                          # decay at 1.5 km (D-E, A-E via E)


def partial_5m_weights(city=ORACULUM, buffer_m=5.0):
    settlements = city.load_settlements()
    return partial_weights(adjacency(settlements, "bbox"), settlements,
                           city.load_barriers(), buffer_m)


def test_only_the_ad_edge_is_partially_blocked_on_oraculum():
    """Every other bbox pair's shared boundary is at least 20 m from the
    canal's ends (A-E starts at x = 500, the buffer stops at 480), so the
    canal produces exactly one fractional weight — in both directions."""
    weights = partial_5m_weights()
    assert weights[("A", "D")] == pytest.approx(W_AD, abs=1e-12)
    assert weights[("A", "D")] == weights[("D", "A")]
    fractional = {pair for pair, w in weights.items() if w != 1.0}
    assert fractional == {("A", "D"), ("D", "A")}


def test_a_smaller_buffer_blocks_less_of_the_same_edge():
    """The buffer made visible: at 1 m the canal blocks [24, 476] = 452 m,
    so w = 0.096. The limit as buffer_m -> 0 is the memo's 0.1, which is
    never a pin because buffer_m must be > 0 (spec § 12 item 5)."""
    assert partial_5m_weights(buffer_m=1.0)[("A", "D")] == pytest.approx(
        0.096, abs=1e-12)


def test_partial_weighted_prunes_nothing_on_oraculum():
    """No weight is 0, so the lists are the bbox lists — and A and D are back
    in everyone's list, because the global rule's flag-based severing is
    gone."""
    settlements = ORACULUM.load_settlements()
    got = apply_barrier(adjacency(settlements, "bbox"), settlements,
                        ORACULUM.load_barriers(), "partial_weighted", 5.0)
    assert got == {"A": {"B", "D", "E"}, "B": {"A", "C", "E", "RV"},
                   "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"A", "E"},
                   "E": {"A", "B", "C", "D", "IND"}, "IND": {"C", "E"}}


def test_partial_5m_pcen_anchors_on_oraculum():
    """Every row derived on paper from the geometry (spec § 6.1). D's clinic
    row is the one that shows the rule: A lends 2 clinics at 8% weight over
    a sqrt(5)/2 km centroid gap, and E lends 1 at 1.5 km, undiscounted."""
    got = scored(ORACULUM, PARTIAL_5M)
    assert got.loc["D", "clinic_pcen"] == pytest.approx(
        (0 + W_AD * 2 * W_DECAY_AD + 1 * W_15) / 100, abs=1e-12)
    assert got.loc["D", "school_pcen"] == pytest.approx(
        (1 + W_AD * 1 * W_DECAY_AD + 1 * W_15) / 100, abs=1e-12)
    assert got.loc["A", "school_pcen"] == pytest.approx(
        (1 + W_AD * 1 * W_DECAY_AD + 1 * W_SQRT2) / 100, abs=1e-12)
    # A's clinic row is UNCHANGED by the weight: D owns no clinic.
    assert got.loc["A", "clinic_pcen"] == pytest.approx(
        (2 + 1 * W_HALF + 1 * W_SQRT2) / 100, abs=1e-12)
    # roads are decayed under the `code` base: A owns 0.75 km
    assert got.loc["D", "road_pcen"] == pytest.approx(
        (0 + W_AD * 0.75 * W_DECAY_AD + 0.75 * W_15) / 100, abs=1e-12)


def test_partial_5m_restores_the_links_the_global_rule_severed():
    """B and E get A back — under `code` the global rule dropped every link
    INTO a flagged settlement, so B's clinic row was 0.0125 and E's was the
    code value. Under partial_weighted they are the `ideal` values, because
    no barrier touches those boundaries at all."""
    got = scored(ORACULUM, PARTIAL_5M)
    assert got.loc["B", "clinic_pcen"] == pytest.approx(0.0175, abs=1e-12)
    assert got.loc["E", "clinic_pcen"] == pytest.approx(
        (1 + 2 * W_SQRT2 + 1 * W_HALF) / 300, abs=1e-12)


def test_partial_5m_leaves_no_constant_column_on_oraculum():
    """The invariants guard refuses a degenerate min-max group, and DEL-54's
    guard raises on one. Both denominators, every service: checked here
    BEFORE the fixture regeneration step depends on it."""
    for denom in ("pop", "popdensity"):
        got = scored(ORACULUM, PARTIAL_5M, denom)
        for column in [c for c in got.columns if c.endswith("_pcen")]:
            assert got[column].max() > got[column].min(), (denom, column)
```

`math`, `pytest`, `RULESETS`, `adjacency`, `apply_barrier`, `scored`,
`variant` and `ORACULUM` are already imported or defined in
`tests/test_variant_rules.py`. `W_SQRT2` and `W_HALF` are NOT — those live in
`tests/test_reference_impl.py`, and this file must not import from it. Define
them next to `PARTIAL_5M`:

```python
W_SQRT2 = 1 / (1 + math.sqrt(2))        # decay at 1000*sqrt(2) m
W_HALF = 0.5                            # decay at 1000 m
```

- [ ] **Step 7: Run the anchors, then prove nothing regenerated**

Run: `uv run pytest -q -W error tests/test_variant_rules.py tests/test_reference_impl.py`
Expected: PASS. If `test_partial_5m_pcen_anchors_on_oraculum` fails, the
arithmetic in the plan is right and the implementation is wrong — re-derive
from the test's own docstring before touching the expected value.

Then:

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
uv run python scripts/generate_production_fixtures.py
git status --porcelain tests/fixtures/
```

Expected: **empty output**. The `w * ` factor is `1.0 *` for every committed
rule-set, which is exact in IEEE, so every CSV must come back byte-identical.
A modified fixture here is a STOP, not something to commit.

- [ ] **Step 8: Add the variant-table entries (inert until Task 6)**

In `tests/reference_impl.py`, extend the two tables. They are consulted only
for blocks a variant actually names, so these lines change nothing until
`tests/variants.py` gains its row:

```python
VARIANT_KNOBS = {
    ("adjacency", "rule"): "adjacency_rule",
    ("adjacency", "max_distance_km"): "max_distance_km",
    ("barrier", "rule"): "barrier_rule",
    ("barrier", "buffer_m"): "barrier_buffer_m",
    ("decay", "form"): "decay_form",
    ("decay", "distance"): "decay_distance",
    ("decay", "exponent"): "exponent",
    ("decay", "scale_km"): "scale_km",
}
# `barrier.combine` has no reference knob: the reference uses EVERY barrier
# row, which is what `any` means on a one-layer city, and both fixture cities
# have one layer or none. `decay.distance_unit` has none either (the
# reference is km-only, as the manuscript is).
IGNORED_VARIANT_KEYS = frozenset({("decay", "distance_unit"),
                                  ("barrier", "combine")})
```

- [ ] **Step 9: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call with an explicit `timeout`
of **600000 ms** and `run_in_background` NOT set — a call without an explicit
timeout is auto-backgrounded at 120 s and its completion never reaches you.
Expected: all pass (598 tests at `bcc557c` plus the ~19 added here). Read the
summary line and quote it in your report.

```bash
git add tests/reference_impl.py tests/test_reference_impl.py tests/test_variant_rules.py
git commit -m "$(cat <<'MSG'
test(barrier): the reference implements partial_weighted (DEL-48)

w_ij = 1 - L_blocked/L_shared over the shared boundary, where the blocked
part is what lies within buffer_m metres of a barrier. The shared boundary
is the boundary of every polygonal component of the intersection plus every
lineal one, decomposed part by part — a GeometryCollection has no .boundary
in shapely, and an overlap-plus-shared-edge pair produces exactly that.
L_shared == 0 means nothing to block, so a corner-only contact keeps w = 1
where pairwise severs.

Anchored on Oraculum: the canal's 5 m buffer covers [20, 480] of the 500 m
A-D edge, so w_AD = 0.08 (not the memo's buffer-free 0.1), and it is the
only fractional weight either fixture city can produce. Every committed
expected value is byte-identical: the new factor is 1.0 for both rule-sets
and 1.0 * x is exact.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 2: Config — the `partial_weighted` value and `buffer_m`

**Files:**
- Modify: `delhi_psi/config.py` (`REFERENCE_KNOBS` :34-52, `RESERVED_VALUES`
  :98-115, `BarrierConfig` :189-192, `_methodology`'s barrier block :412-425)
- Modify: `delhi_psi/profiles/code-2025.yaml` (the `barrier:` comment block)
- Test: `tests/test_config.py` (delete `test_reserved_partial_weighted`; new
  rejection rows; a loading test)
- Test: `tests/test_profiles_match_reference.py` (`EXTRA_PARAMS` :100-105)

**Interfaces:**
- Consumes from Task 1: `reference_impl.compute_city(barrier_rule=
  "partial_weighted", barrier_buffer_m=5.0, ...)` must already work — this
  task's `REFERENCE_KNOBS` entry is what makes
  `test_every_mapped_knob_is_one_the_reference_actually_implements` call it.
- Produces: `config.BarrierRule.PARTIAL_WEIGHTED` (value
  `"partial_weighted"`), `config.BarrierConfig(rule, combine,
  buffer_m: float | None = None)`, and
  `EXTRA_PARAMS[("methodology.barrier.rule", "partial_weighted")] =
  {"barrier_buffer_m": 5.0}`.

- [ ] **Step 1: Write the failing config tests**

In `tests/test_config.py`, **delete** `test_reserved_partial_weighted`
(:145-151) — the value is no longer reserved, and the test asserts the
message it no longer produces. Then add, next to the 3D band tests:

```python
# --- 3E: partial_weighted and its buffer (spec § 1) --------------------
BARRIER_PARTIAL = ("  barrier: {rule: partial_weighted, combine: any, "
                   "buffer_m: 5}")


def test_partial_weighted_loads_with_its_buffer(tmp_path):
    cfg = load_config(write(tmp_path, swap("  barrier:", BARRIER_PARTIAL)),
                      data_dir=str(tmp_path))
    assert cfg.methodology.barrier.rule == "partial_weighted"
    assert cfg.methodology.barrier.buffer_m == 5.0
    assert isinstance(cfg.methodology.barrier.buffer_m, float)


@pytest.mark.parametrize("profile", ["code-2025", "manuscript"])
def test_shipped_profiles_carry_no_buffer(profile, tmp_path):
    """Neither shipped profile uses partial_weighted, so `buffer_m` is
    'not applicable' — None in the dataclass, absent from the YAML."""
    cfg = load_config(profile, data_dir=str(tmp_path))
    assert cfg.methodology.barrier.buffer_m is None
```

and add these six rows to the existing
`test_conditional_parameters_are_rejected_naming_the_key` parametrization —
the same shape the `exponent` / `scale_km` / `max_distance_km` rows use:

```python
    # a parameter the rule does not use is REJECTED, not ignored
    ("methodology.barrier.buffer_m", "  barrier:",
     "  barrier: {rule: global_asymmetric, combine: any, buffer_m: 5}"),
    ("methodology.barrier.buffer_m", "  barrier:",
     "  barrier: {rule: pairwise, combine: any, buffer_m: 5}"),
    # required and missing
    ("methodology.barrier.buffer_m", "  barrier:",
     "  barrier: {rule: partial_weighted, combine: any}"),
    # out of range (strictly > 0: buffer(0) is EMPTY in shapely), and
    # booleans are not numbers
    ("methodology.barrier.buffer_m", "  barrier:",
     "  barrier: {rule: partial_weighted, combine: any, buffer_m: 0}"),
    ("methodology.barrier.buffer_m", "  barrier:",
     "  barrier: {rule: partial_weighted, combine: any, buffer_m: -1}"),
    ("methodology.barrier.buffer_m", "  barrier:",
     "  barrier: {rule: partial_weighted, combine: any, buffer_m: true}"),
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_config.py -k "partial_weighted or buffer_m or shipped_profiles_carry_no_buffer"`

Expected: **FAIL.** `test_partial_weighted_loads_with_its_buffer` fails with
`ConfigError: methodology.barrier.rule: 'partial_weighted' is not allowed`
(the value is still reserved / not in the enum);
`test_shipped_profiles_carry_no_buffer` fails with
`AttributeError: 'BarrierConfig' object has no attribute 'buffer_m'`; the six
rejection rows fail because `unknown key 'methodology.barrier.buffer_m'` is
raised for the wrong reason on some and nothing is raised on others. Record
the text.

- [ ] **Step 3: Make the value loadable**

In `delhi_psi/config.py`:

```python
REFERENCE_KNOBS = {
    ...
    "methodology.barrier.rule": {"global_asymmetric": "global",
                                 "pairwise": "pair",
                                 "partial_weighted": "partial_weighted"},
    ...
}
```

The `partial_weighted` spelling is deliberately the SAME on both sides, like
every 3D value, so `tests/variants.py` needs no translation layer.

Delete the whole `"methodology.barrier.rule"` entry from `RESERVED_VALUES`
(:99-107) — its message ends "(cycle 3C)" and is stale, and an empty dict
would leave `_coerce_enum` appending an empty "(reserved: [])" note. The
table keeps only its `outputs.denominators[]` entry:

```python
RESERVED_VALUES = {
    "outputs.denominators[]": {
        "one":
            "reserved: production supports denom='one' but the reference does "
            "not. Unblock it by adding `denom == \"one\"` to "
            "tests.reference_impl.compute_city and regenerating "
            "tests/fixtures/oraculum/expected_values.csv first.",
    },
}
```

- [ ] **Step 4: Add `buffer_m` to the dataclass and the loader**

```python
@dataclass(frozen=True)
class BarrierConfig:
    rule: BarrierRule
    combine: object                # "any" or a tuple of layer names
    # None is "not applicable", never a default for the YAML key: it is
    # required by partial_weighted and rejected by the other two rules,
    # which have no buffer at all (global_asymmetric reads the per-polygon
    # flag, pairwise uses `intersects`).
    buffer_m: float | None = None
```

and, in `_methodology`, replace the barrier block (:412-425) with:

```python
    barrier_raw = _require(raw, "barrier", "methodology")
    _reject_unknown(barrier_raw, {"rule", "combine", "buffer_m"},
                    "methodology.barrier")
    barrier_rule = _coerce_enum(
        "methodology.barrier.rule",
        _require(barrier_raw, "rule", "methodology.barrier"))
    combine = _require(barrier_raw, "combine", "methodology.barrier")
    if combine != "any":
        if not isinstance(combine, list) or not all(
                isinstance(item, str) for item in combine):
            raise ConfigError(
                "methodology.barrier.combine: expected 'any' or a list of "
                f"layer names, got {combine!r}")
        combine = tuple(combine)
    barrier = BarrierConfig(
        rule=barrier_rule,
        combine=combine,
        buffer_m=_conditional_number(
            barrier_raw, "buffer_m", "methodology.barrier",
            used_by="methodology.barrier.rule: partial_weighted",
            applies=barrier_rule == BarrierRule.PARTIAL_WEIGHTED,
            minimum=0, strict=True))
```

- [ ] **Step 5: Give the reference its buffer in `EXTRA_PARAMS`**

In `tests/test_profiles_match_reference.py`, extend the table at :100:

```python
EXTRA_PARAMS = {
    ("methodology.adjacency.rule", "within_distance"):
        {"max_distance_km": 0.25},
    ("methodology.barrier.rule", "partial_weighted"):
        {"barrier_buffer_m": 5.0},
    ("methodology.decay.form", "inverse_power"): {"exponent": 2},
    ("methodology.decay.form", "exponential"): {"scale_km": 1.0},
}
```

This is the same 5.0 the § 6.1 pins use — never a fresh number.
**This must be in this commit:** without it,
`test_every_mapped_knob_is_one_the_reference_actually_implements` calls
`compute_city(barrier_rule="partial_weighted")` with no buffer and the
reference raises.

- [ ] **Step 6: Update the shipped profile's comment**

In `delhi_psi/profiles/code-2025.yaml`, the barrier block currently says
`partial_weighted: reserved (spec 4)`, which is false the moment Step 3
lands. Replace those three lines with:

```yaml
  barrier:
    rule: global_asymmetric         # global_asymmetric | pairwise | partial_weighted
                                    # buffer_m: required iff rule is
                                    # partial_weighted (> 0 metres); a boundary
                                    # point within buffer_m of a barrier is blocked
    combine: any                    # any | [layer names]; which flags OR into `barrier`
                                    # (every configured layer's flag column is always computed)
```

Only the comments change. `manuscript.yaml` is not touched in this group —
it carries `pairwise` and gains nothing.

- [ ] **Step 7: Run the config and profile tests**

Run: `uv run pytest -q -W error tests/test_config.py tests/test_profiles_match_reference.py`
Expected: PASS. Note in particular that
`test_out_of_enum_names_key_and_allowed_values` and
`test_enums_are_generated_from_the_reference_table` now cover
`partial_weighted` automatically, because both read `REFERENCE_KNOBS`.

- [ ] **Step 8: Prove the shipped profiles' numbers did not move**

Run: `uv run pytest -q -W error tests/test_production_fixtures.py tests/test_manuscript_anchors.py`
Expected: PASS — both shipped profiles produce byte-identical fixture
outputs, which is what "this cycle changes no numbers" means at the config
level.

- [ ] **Step 9: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add delhi_psi/config.py delhi_psi/profiles/code-2025.yaml \
        tests/test_config.py tests/test_profiles_match_reference.py
git commit -m "$(cat <<'MSG'
feat(barrier): partial_weighted is a loadable value with a required buffer_m (DEL-48)

REFERENCE_KNOBS gains the value (same spelling on both sides, so the variant
table needs no translation layer) and the stale RESERVED_VALUES entry —
whose message still pointed at cycle 3C — is deleted. buffer_m goes through
_conditional_number verbatim like exponent and scale_km: required by
partial_weighted, rejected by the other two rules, strictly > 0 because
LineString.buffer(0) is EMPTY in shapely and 0 would silently make every
weight 1.

EXTRA_PARAMS lands in the same commit: the mapped-knob test drives the
reference once per value, and partial_weighted requires its buffer there.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 3: Production geometry — `shared_boundary`, `partial_weight`, `apply_barrier`, `combine`

**Files:**
- Modify: `delhi_psi/neighbors.py` (imports; `combine_barrier_flags` :24-42;
  `apply_barrier` :143-176; two new public helpers and a layer selector)
- Test: `tests/test_neighbors.py`

**Interfaces:**
- Consumes from Task 2: nothing at runtime — `neighbors.py` never imports
  `delhi_psi.config`. It takes `rule` as a plain string (a `BarrierRule`
  StrEnum compares equal to it).
- Produces:
  - `neighbors.shared_boundary(geom_i, geom_j) -> shapely geometry` (possibly
    EMPTY; `.length == 0` for an empty intersection or a corner contact)
  - `neighbors.partial_weight(shared, buffered_barriers) -> float`
  - `neighbors.selected_layers(layers, combine) -> tuple[str, ...]`
  - `neighbors.selected_barrier_geoms(barriers, *, combine) -> list`
  - `neighbors.apply_barrier(polygon_gdf, barrier_geoms, *, id_col,
    neighbor_col, rule, flag_col, buffer_m=None,
    weight_col="nbrs_barrier_weight")` — under `partial_weighted` the frame
    gains `weight_col` = `[(neighbor_id, w), ...]` in the SAME order as the
    pruned `neighbor_col` list.

- [ ] **Step 1: Write the failing unit tests (spec § 6.4 items 1-9)**

Append to `tests/test_neighbors.py`:

```python
# --- 3E: partial_weighted (spec § 2.1, § 2.6, § 6.4) -------------------
def two_squares(geom_a=None, geom_b=None):
    """A and B, 1 km squares sharing the 1000 m edge at x = 1000 unless the
    caller supplies its own geometries. The `test_index.city_with_neighbours`
    shape: hand-built, EPSG:7760, metre coordinates."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "nbrs_bbox": [["B"], ["A"]],
         "barrier": [False, False]},
        geometry=[geom_a if geom_a is not None else box(0, 0, 1000, 1000),
                  geom_b if geom_b is not None else box(1000, 0, 2000, 1000)],
        crs="EPSG:7760")


def weights_of(frame, col="nbrs_barrier_weight"):
    return {row["USO_AREA_U"]: dict(row[col]) for _, row in frame.iterrows()}


def partial(frame, *barrier_geoms, buffer_m=5.0):
    return neighbors.apply_barrier(frame, list(barrier_geoms),
                                   rule="partial_weighted",
                                   buffer_m=buffer_m)


def test_partial_weighted_on_a_fully_covered_edge_prunes_like_pairwise():
    """w = 0 exactly, so the pair leaves the list and the frame equals what
    `pairwise` produces. The 'full-coverage variant' the memo imagined is
    this unit test, not a fixture."""
    from shapely.geometry import LineString

    canal = LineString([(1000, 0), (1000, 1000)])
    got = partial(two_squares(), canal)
    assert lists_of(got) == {"A": set(), "B": set()}
    assert weights_of(got) == {"A": {}, "B": {}}
    severed = neighbors.apply_barrier(two_squares(), [canal], rule="pairwise")
    assert lists_of(got) == lists_of(severed)


def test_partial_weighted_on_a_half_covered_edge_is_not_one_half():
    """5 m round caps extend the blocked span past each end: the middle
    500 m blocks 510, and 500 m from the corner blocks 505. Pinned so nobody
    'fixes' 0.49 into 0.5."""
    from shapely.geometry import LineString

    middle = partial(two_squares(), LineString([(1000, 250), (1000, 750)]))
    assert weights_of(middle)["A"]["B"] == pytest.approx(0.49, abs=1e-12)
    corner = partial(two_squares(), LineString([(1000, 0), (1000, 500)]))
    assert weights_of(corner)["A"]["B"] == pytest.approx(0.495, abs=1e-12)


def test_a_perpendicular_crossing_is_kept_where_pairwise_severs():
    """The owner's 'a point crossing severs nothing'. This is the one
    documented case where partial_weighted and pairwise disagree."""
    from shapely.geometry import LineString

    crossing = LineString([(900, 500), (1100, 500)])
    got = partial(two_squares(), crossing)
    assert weights_of(got)["A"]["B"] == pytest.approx(0.99, abs=1e-12)
    assert lists_of(got) == {"A": {"B"}, "B": {"A"}}
    severed = neighbors.apply_barrier(two_squares(), [crossing],
                                      rule="pairwise")
    assert lists_of(severed) == {"A": set(), "B": set()}


def test_a_barrier_just_off_the_edge_still_blocks_it():
    """The buffer's purpose: a barrier drawn 4 m off a sliver gap is within
    5 m of every boundary point, so w = 0. One 200 m away is not."""
    from shapely.geometry import LineString

    close = partial(two_squares(), LineString([(996, 0), (996, 1000)]))
    assert lists_of(close) == {"A": set(), "B": set()}
    far = partial(two_squares(), LineString([(1200, 0), (1200, 1000)]))
    assert weights_of(far)["A"]["B"] == 1.0


def test_an_overlapping_pair_weighs_the_intersection_boundary():
    """The owner's overlap rule, made numeric on the messy O1/O2 shape: the
    shared boundary is the 200 x 1000 m strip's 2400 m PERIMETER."""
    from shapely.geometry import LineString, box

    frame = two_squares(box(10000, 0, 11000, 1000),
                        box(10800, 0, 11800, 1000))
    crossing = partial(frame, LineString([(10900, 0), (10900, 1000)]))
    assert weights_of(crossing)["A"]["B"] == pytest.approx(
        1 - 20 / 2400, abs=1e-12)
    along = partial(frame, LineString([(11000, 0), (11000, 1000)]))
    assert weights_of(along)["A"]["B"] == pytest.approx(
        1 - 1010 / 2400, abs=1e-12)


def test_a_multipolygon_neighbour_sums_both_shared_edges():
    from shapely.geometry import LineString, MultiPolygon, box

    multi = MultiPolygon([box(1000, 0, 2000, 400), box(1000, 600, 2000, 1000)])
    got = partial(two_squares(geom_b=multi),
                  LineString([(1000, 0), (1000, 400)]))
    assert weights_of(got)["A"]["B"] == pytest.approx(0.5, abs=1e-12)


def test_a_mixed_intersection_is_decomposed_part_by_part():
    """Overlap on one side, shared edge on another: shapely returns a
    GeometryCollection, whose `.boundary` is None. SB is the overlap
    polygon's 1000 m perimeter plus the 400 m line, so a barrier over the
    line alone blocks 410 of 1400."""
    from shapely.geometry import LineString, MultiPolygon, box

    mixed = MultiPolygon([box(900, 0, 1900, 400), box(1000, 600, 1900, 1000)])
    square = box(0, 0, 1000, 1000)
    assert square.intersection(mixed).boundary is None
    shared = neighbors.shared_boundary(square, mixed)
    assert shared.length == pytest.approx(1400.0, abs=1e-9)
    got = partial(two_squares(square, mixed),
                  LineString([(1000, 600), (1000, 1000)]))
    assert weights_of(got)["A"]["B"] == pytest.approx(
        1 - 410 / 1400, abs=1e-12)


def test_a_corner_only_contact_is_never_severed():
    """L_shared == 0: there is no boundary to block, so w = 1 even with a
    barrier through the corner (spec § 2.1 step 3)."""
    from shapely.geometry import LineString, box

    frame = two_squares(geom_b=box(1000, 1000, 2000, 2000))
    got = partial(frame, LineString([(900, 1100), (1100, 900)]))
    assert weights_of(got)["A"]["B"] == 1.0
    assert lists_of(got) == {"A": {"B"}, "B": {"A"}}


@pytest.mark.parametrize("barrier_geom", [
    "collinear", "middle", "perpendicular", "overlap"])
def test_the_weight_is_symmetric(barrier_geom):
    """Same GEOS calls on the same operands from either side, so w(i, j) and
    w(j, i) must agree BIT for bit, not to a tolerance."""
    from shapely.geometry import LineString, box

    cases = {
        "collinear": (two_squares(), LineString([(1000, 0), (1000, 1000)])),
        "middle": (two_squares(), LineString([(1000, 250), (1000, 750)])),
        "perpendicular": (two_squares(), LineString([(900, 500), (1100, 500)])),
        "overlap": (two_squares(box(10000, 0, 11000, 1000),
                                box(10800, 0, 11800, 1000)),
                    LineString([(10900, 0), (10900, 1000)])),
    }
    frame, barrier = cases[barrier_geom]
    got = neighbors.apply_barrier(frame, [barrier], rule="partial_weighted",
                                  buffer_m=5.0)
    forward = dict(got.iloc[0]["nbrs_barrier_weight"])
    backward = dict(got.iloc[1]["nbrs_barrier_weight"])
    assert forward.get("B") == backward.get("A")


def test_partial_weighted_with_no_barriers_writes_weights_of_one():
    """The messy city has an EMPTY barrier layer and is scored under this
    rule, so the column must still exist and hold 1.0 — otherwise `pcen`
    raises KeyError on a city with nothing to block."""
    got = partial(two_squares())
    assert weights_of(got) == {"A": {"B": 1.0}, "B": {"A": 1.0}}


def test_the_weight_column_exists_only_under_partial_weighted():
    """code-2025's artifact must be byte-identical, and an artifact built
    before 3E must still load."""
    for rule in ("global_asymmetric", "pairwise"):
        got = neighbors.apply_barrier(two_squares(), [], rule=rule)
        assert "nbrs_barrier_weight" not in got.columns


def test_buffer_m_is_required_by_partial_weighted_and_rejected_otherwise():
    with pytest.raises(ValueError, match="buffer_m"):
        neighbors.apply_barrier(two_squares(), [], rule="partial_weighted")
    with pytest.raises(ValueError, match="buffer_m"):
        neighbors.apply_barrier(two_squares(), [], rule="partial_weighted",
                                buffer_m=0)
    for rule in ("global_asymmetric", "pairwise"):
        with pytest.raises(ValueError, match="buffer_m"):
            neighbors.apply_barrier(two_squares(), [], rule=rule,
                                    buffer_m=5.0)


def test_combine_selects_the_layers_the_geometry_rules_see():
    """spec § 2.6: `combine` chooses the LAYERS whose geometries pairwise and
    partial_weighted read — which is what the stamp's '`combine` decides who
    is severed' already claims. With combine=('railway',) a canal over the
    shared edge severs nothing; with 'any' it severs."""
    import geopandas as gpd
    from shapely.geometry import LineString

    canal = LineString([(1000, 0), (1000, 1000)])
    barriers = {
        "canal": gpd.GeoDataFrame(geometry=[canal], crs="EPSG:7760"),
        "railway": gpd.GeoDataFrame(
            geometry=[LineString([(5000, 0), (5000, 1000)])],
            crs="EPSG:7760"),
    }
    railway_only = neighbors.selected_barrier_geoms(barriers,
                                                    combine=("railway",))
    every = neighbors.selected_barrier_geoms(barriers, combine="any")
    assert len(railway_only) == 1 and len(every) == 2

    kept = neighbors.apply_barrier(two_squares(), railway_only,
                                   rule="pairwise")
    assert lists_of(kept) == {"A": {"B"}, "B": {"A"}}
    cut = neighbors.apply_barrier(two_squares(), every, rule="pairwise")
    assert lists_of(cut) == {"A": set(), "B": set()}
    weighted = neighbors.apply_barrier(two_squares(), railway_only,
                                       rule="partial_weighted", buffer_m=5.0)
    assert weights_of(weighted)["A"]["B"] == 1.0


def test_selected_barrier_geoms_rejects_an_unconfigured_layer():
    """The same message `combine_barrier_flags` gives, from one helper."""
    import geopandas as gpd

    barriers = {"canal": gpd.GeoDataFrame(geometry=[], crs="EPSG:7760")}
    with pytest.raises(ValueError, match="drain"):
        neighbors.selected_barrier_geoms(barriers, combine=("drain",))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_neighbors.py -k "partial or weight or combine_selects or corner_only or multipolygon or mixed_intersection or perpendicular or symmetric"`

Expected: **FAIL** — `ValueError: unknown barrier rule 'partial_weighted';
allowed values: ['global_asymmetric', 'pairwise']` from `apply_barrier`, and
`AttributeError: module 'delhi_psi.neighbors' has no attribute
'shared_boundary' / 'selected_barrier_geoms'`. Record the text.

- [ ] **Step 3: Add the geometry helpers and the layer selector**

In `delhi_psi/neighbors.py`, extend the imports:

```python
from shapely import STRtree
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
```

then add, above `combine_barrier_flags`:

```python
_POLYGONAL = frozenset({"Polygon", "MultiPolygon"})
_LINEAL = frozenset({"LineString", "LinearRing", "MultiLineString"})


def selected_layers(layers, combine):
    """The layer names `combine` selects: every configured layer for "any"."""
    selected = tuple(layers) if combine == "any" else tuple(combine)
    unknown = [name for name in selected if name not in layers]
    if unknown:
        raise ValueError(
            f"barrier.combine names layers that are not configured: {unknown}; "
            f"configured layers: {sorted(layers)}")
    return selected


def selected_barrier_geoms(barriers, *, combine):
    """The geometries of the layers `combine` selects, in layer order.

    `pairwise` and `partial_weighted` read GEOMETRIES, so `combine` has to
    choose which layers they see — which is what the methodology stamp's
    "`combine` decides who is severed" already claims for every rule (spec
    § 2.6). `global_asymmetric` reads the flag column `combine_barrier_flags`
    builds and never comes here.
    """
    return [geom for name in selected_layers(barriers, combine)
            for geom in barriers[name].geometry]


def shared_boundary(geom_i, geom_j):
    """The boundary i and j share: SB_ij (spec § 2.1 steps 1-2).

    The boundary of every POLYGONAL component of the intersection — the
    owner's overlap rule, "the shared boundary of an overlapping pair is the
    boundary of the intersection polygon" — plus every LINEAL component.
    Point components contribute nothing.

    A GeometryCollection is decomposed into its parts FIRST. This is the one
    place a naive `shared.boundary` would be wrong (shapely returns None for
    a collection), and it is a real-layer case: a polygon that overlaps its
    neighbour on one side and shares an edge on another.

    Returns a possibly EMPTY geometry, so callers test `.length`, never
    `is None`.
    """
    shared = geom_i.intersection(geom_j)
    if shared.is_empty:
        return shared
    parts = list(shared.geoms) if hasattr(shared, "geoms") else [shared]
    pieces = [part.boundary if part.geom_type in _POLYGONAL else part
              for part in parts
              if part.geom_type in _POLYGONAL or part.geom_type in _LINEAL]
    return unary_union(pieces) if pieces else GeometryCollection()


def partial_weight(shared, buffered_barriers):
    """w = 1 - L_blocked / L_shared (spec § 2.1 steps 3-5).

    `buffered_barriers` are ALREADY buffered: the caller builds them once.
    The blocked PIECES are unioned before their length is taken, so two
    overlapping barrier buffers never count the same metre twice.

    L_shared == 0 (an empty intersection, or a corner-only contact) means
    there is nothing to block, so w = 1 — the owner's "a point crossing
    severs nothing". The `>=` guards the float case where the intersection
    returns the whole boundary plus a rounding hair; nothing is rounded.
    """
    length = shared.length
    if length == 0:
        return 1.0
    pieces = [piece for piece in
              (shared.intersection(b) for b in buffered_barriers)
              if not piece.is_empty]
    if not pieces:
        return 1.0
    covered = unary_union(pieces).length
    return 0.0 if covered >= length else 1 - covered / length
```

Then make `combine_barrier_flags` use the shared selector, so the two paths
cannot drift:

```python
def combine_barrier_flags(polygon_gdf, *, layers, combine, out_col="barrier"):
    """OR the selected per-layer flag columns into `out_col`.

    combine == "any" uses every configured layer; otherwise it is a sequence
    of layer names. Every configured layer's own flag column is left intact.
    """
    out = polygon_gdf.copy()
    flag = None
    for name in selected_layers(layers, combine):
        column = out[name].fillna(False).astype(bool)
        flag = column if flag is None else (flag | column)
    out[out_col] = False if flag is None else flag
    return out
```

- [ ] **Step 4: Rewrite `apply_barrier`**

Replace `apply_barrier` (:143-176) with:

```python
def apply_barrier(polygon_gdf, barrier_geoms, *, id_col="USO_AREA_U",
                  neighbor_col="nbrs_bbox", rule="global_asymmetric",
                  flag_col="barrier", buffer_m=None,
                  weight_col="nbrs_barrier_weight"):
    """Sever, or discount, neighbour links across barriers.

    global_asymmetric: drop every neighbour whose `flag_col` is True — the
        production rule (a per-polygon flag, so severing is one-directional).
    pairwise: drop j from i's list when a barrier geometry intersects the
        boundary i and j share — the manuscript rule.
    partial_weighted: keep j with weight w_ij = 1 - L_blocked/L_shared, and
        drop it only when w_ij == 0 (DEL-48, spec § 2.1). `neighbor_col`
        keeps its existing contract — the pruned list of ids — and the
        weights travel in `weight_col` as [(neighbor_id, w), ...] in the SAME
        order, the `nbrs_dist_bbox` 2-tuple shape. The column is written ONLY
        under this rule, so an artifact built under either other rule (and
        every artifact built before cycle 3E) is unchanged.

    buffer_m is used by `partial_weighted` alone: required by it, rejected by
    the other two, which have no buffer at all. It mirrors the config rule
    (`build_neighbors` forwards the configured value unconditionally, and it
    is None there).
    """
    if rule not in ("global_asymmetric", "pairwise", "partial_weighted"):
        raise ValueError(
            f"unknown barrier rule {rule!r}; allowed values: "
            "['global_asymmetric', 'pairwise', 'partial_weighted']")
    if rule == "partial_weighted":
        if buffer_m is None:
            raise ValueError(
                "barrier rule 'partial_weighted' requires buffer_m — the "
                "distance in metres within which a barrier blocks a boundary")
        if not buffer_m > 0:
            raise ValueError(
                f"buffer_m must be > 0, got {buffer_m!r}: a zero buffer is "
                "EMPTY in shapely and would make every weight 1 silently")
    elif buffer_m is not None:
        raise ValueError(
            "buffer_m is only used by barrier rule 'partial_weighted', not "
            f"{rule!r}")

    out = polygon_gdf.copy()
    if rule == "partial_weighted":
        # Buffer ONCE and index the buffers: per link only the candidates the
        # tree returns are intersected. The naive alternative — one
        # unary_union of every buffer, overlaid per link — scales each
        # overlay with the union's vertex count (spec § 2.2).
        buffered = [geom.buffer(buffer_m) for geom in barrier_geoms]
        tree = STRtree(buffered) if buffered else None
        geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
        out[weight_col] = np.empty((len(out), 0)).tolist()
        for idx, row in tqdm(out.iterrows(), total=len(out)):
            kept, weights = [], []
            for j in row[neighbor_col]:
                shared = shared_boundary(geoms[row[id_col]], geoms[j])
                if tree is None or shared.length == 0:
                    weight = 1.0
                else:
                    weight = partial_weight(
                        shared, [buffered[k] for k in tree.query(shared)])
                if weight > 0.0:
                    kept.append(j)
                    weights.append((j, weight))
            out.at[idx, neighbor_col] = kept
            out.at[idx, weight_col] = weights
        return out

    if not barrier_geoms:
        return out
    geoms = {row[id_col]: row["geometry"] for _, row in out.iterrows()}
    flags = {row[id_col]: bool(row[flag_col]) for _, row in out.iterrows()} \
        if rule == "global_asymmetric" else {}

    for idx, row in out.iterrows():
        i = row[id_col]
        kept = []
        for j in row[neighbor_col]:
            if rule == "global_asymmetric":
                if not flags[j]:
                    kept.append(j)
            else:
                shared = geoms[i].intersection(geoms[j])
                if not any(b.intersects(shared) for b in barrier_geoms):
                    kept.append(j)
        out.at[idx, neighbor_col] = kept
    return out
```

Note the `if not barrier_geoms: return out` short-circuit now sits AFTER the
`partial_weighted` branch: a city with no barriers still gets its weight
column, all 1.0. The messy city is exactly that case.

- [ ] **Step 5: Run the neighbours tests and see them pass**

Run: `uv run pytest -q -W error tests/test_neighbors.py`
Expected: PASS, including the two tests that must not move —
`test_bbox_adjacency_then_global_barrier_matches_production` and
`test_touch_adjacency_then_pairwise_barrier_matches_the_manuscript`.

- [ ] **Step 6: Update the allowed-values message the docstring promised**

`test_unknown_barrier_rule_raises_value_error` matches `"sideways"` and
therefore still passes, but the module docstring at the top of
`neighbors.py` lists only the old rules. Add one sentence to it naming
`partial_weighted` and where the definition lives (spec § 2.1). Do not
restate the arithmetic — it is in `partial_weight`'s docstring.

- [ ] **Step 7: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add delhi_psi/neighbors.py tests/test_neighbors.py
git commit -m "$(cat <<'MSG'
feat(barrier): partial_weighted in neighbors — shared boundary, weight, combine (DEL-48)

apply_barrier gains the rule, buffer_m and a weight column written ONLY
under it: neighbor_col keeps its pruned-list contract, so centroid_distances,
boundary_distances, verify and every set(row[col]) test are untouched, and
code-2025's artifact stays byte-identical.

The barrier buffers are built once and indexed with an STRtree; per link only
the tree's candidates are intersected, and the blocked PIECES are unioned so
two overlapping buffers never count the same metre twice. A city with no
barriers still gets the column, all 1.0 — the messy city is exactly that.

combine now selects the LAYERS the geometry rules see (spec § 2.6), through
one selector combine_barrier_flags shares. No profile, fixture or output uses
a non-`any` combine, so nothing committed moves.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 4: `index.pcen` and `service_index` — the weight path

**Files:**
- Modify: `delhi_psi/index.py` (`pcen` :138-209, `service_index` :250-268)
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes from Task 3: the column shape `[(neighbor_id, w), ...]`.
- Produces, and **Group C relies on this exactly**:
  `index.pcen(..., nbr_weight_col=None)` and `index.service_index(...,
  nbr_weight_col=None)`, with the neighbour-loop line
  `poly_count += w * lent * _decay(nbr_dist, ...)` where `lent = nbr_count`
  in this cycle and `w = 1.0` when `nbr_weight_col is None`. Group C
  redefines `lent` on that same line and nothing else.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_index.py`:

```python
# --- 3E: the partial-barrier weight (spec § 2.4) -----------------------
def city_with_weights(weight):
    """city_with_neighbours plus the weight column apply_barrier writes."""
    gdf = city_with_neighbours()
    gdf["nbrs_barrier_weight"] = [[("Y", weight)], [("X", weight)]]
    return gdf


def test_pcen_multiplies_the_neighbour_term_by_its_barrier_weight():
    """Y owns nothing and borrows X's 2 clinics at 1 km (decay 1/2); a
    half-blocked shared boundary halves what it borrows."""
    got = index.pcen(city_with_weights(0.5), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     nbr_weight_col="nbrs_barrier_weight")
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((0 + 0.5 * 2 * 0.5) / 200, abs=1e-12)
    assert values["X"] == pytest.approx(2 / 100, abs=1e-12)


def test_a_weight_of_one_is_bit_identical_to_no_weight_column():
    """1.0 * x is exact in IEEE and multiplication is left-associative, so
    the weighted loop cannot move a number when every weight is 1. This is
    what keeps code-2025 byte-identical."""
    weighted = index.pcen(city_with_weights(1.0), amount_col="clinic_count",
                          pcen_col="clinic_pcen", denominator="pop",
                          nbr_weight_col="nbrs_barrier_weight")
    plain = index.pcen(city_with_neighbours(), amount_col="clinic_count",
                       pcen_col="clinic_pcen", denominator="pop")
    assert list(weighted["clinic_pcen"]) == list(plain["clinic_pcen"])


def test_a_neighbour_with_no_weight_is_a_loud_key_error():
    """Never a silent 1.0: a distance list and a weight list that disagree
    mean the artifact and the frame came from different runs."""
    frame = city_with_weights(0.5)
    frame.at[1, "nbrs_barrier_weight"] = []
    with pytest.raises(KeyError, match="X"):
        index.pcen(frame, amount_col="clinic_count", pcen_col="clinic_pcen",
                   denominator="pop", nbr_weight_col="nbrs_barrier_weight")


def test_service_index_forwards_the_weight_column():
    got = index.service_index(city_with_weights(0.5), "clinic_count",
                              service="clinic", denominator="pop",
                              nbr_weight_col="nbrs_barrier_weight")
    values = got.set_index("USO_AREA_U")
    assert values.loc["Y", "clinic_pcen"] == pytest.approx(
        (0 + 0.5 * 2 * 0.5) / 200, abs=1e-12)
    # min-max still runs: X is the max, Y the min
    assert values.loc["X", "clinic_idx"] == 1.0
    assert values.loc["Y", "clinic_idx"] == 0.0
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_index.py -k "weight"`
Expected: **FAIL with `TypeError: pcen() got an unexpected keyword argument
'nbr_weight_col'`** (and the same for `service_index`). Record the text.

- [ ] **Step 3: Add the weight path to `pcen`**

In `delhi_psi/index.py`, add `nbr_weight_col=None` to `pcen`'s signature
immediately after `nbr_dist_col="nbrs_dist_bbox"`, document it in the
docstring:

```python
    nbr_weight_col: the [(neighbor_id, w), ...] column `apply_barrier` writes
        under `barrier.rule: partial_weighted`. The neighbour's contribution
        is multiplied by w; None means every weight is 1, which is bit
        identical because 1.0 * x is exact.
```

and replace the neighbour loop body (:193-205) with:

```python
        if include_neighbors:
            weights = (dict(row[nbr_weight_col])
                       if nbr_weight_col is not None else None)
            for nbr_id, nbr_dist in row[nbr_dist_col]:
                # An id in the distance list with no weight is a KeyError,
                # never a silent 1.0: it means the two lists came from
                # different runs.
                w = 1.0 if weights is None else weights[nbr_id]
                match = lookup[lookup[id_col] == nbr_id]
                if len(match) == 0:
                    if absent_neighbor == "contributes":
                        raise KeyError(
                            f"neighbour {nbr_id!r} of {row[id_col]!r} has no "
                            "row in the pre-exclusion lookup frame")
                    continue
                lent = match[amount_col].array[0]
                poly_count += w * lent * _decay(nbr_dist, decay_form,
                                                distance_unit,
                                                exponent=exponent,
                                                scale_km=scale_km)
```

- [ ] **Step 4: Forward it from `service_index`**

Add `nbr_weight_col=None` to `service_index`'s signature (after
`nbr_dist_col="nbrs_dist_bbox"`) and pass `nbr_weight_col=nbr_weight_col`
through to `pcen`.

- [ ] **Step 5: Run the index tests and see them pass**

Run: `uv run pytest -q -W error tests/test_index.py`
Expected: PASS, including every pre-existing PCEN anchor — they pass
`nbr_weight_col=None` implicitly and must not have moved.

- [ ] **Step 6: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add delhi_psi/index.py tests/test_index.py
git commit -m "$(cat <<'MSG'
feat(barrier): pcen multiplies each neighbour by its barrier weight (DEL-48)

The neighbour term becomes w * lent * decay, with w = 1.0 when no weight
column is passed — bit identical to today, because 1.0 * x is exact in IEEE
and multiplication is left-associative. An id in the distance list with no
weight is a KeyError, never a silent 1.0.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 5: Pipeline wiring — the column, the stamp, the exclusion strip, the drop

**Files:**
- Modify: `delhi_psi/pipeline.py` (`NBRS_WEIGHT_COL`; `build_neighbors`
  :92-118; `apply_exclusion` :121-147; `index_frames` :150-203;
  `methodology_stamp` :330-349)
- Modify: `delhi_psi/io.py` (`SHAPEFILE_DROP_COLUMNS` :23)
- Test: `tests/test_pipeline.py`
- Test: `tests/test_cli.py` (the stamp literal at :297-300)
- Test: `tests/test_measure_roads_access.py` (`stamp_forms`'s `real` dict
  at :260-261)

**Interfaces:**
- Consumes from Tasks 2, 3, 4: `config.BarrierConfig.buffer_m`,
  `neighbors.selected_barrier_geoms`, `neighbors.apply_barrier(...,
  buffer_m=, weight_col=)`, `index.service_index(..., nbr_weight_col=)`.
- Produces, and **Group C relies on this name**: `pipeline.NBRS_WEIGHT_COL
  == "nbrs_barrier_weight"`. Also: the stamp's
  `barrier.buffer_m` entry, and `index_frames` dropping the column before
  returning.

**Why `tests/test_measure_roads_access.py` is in this list.** The spec's
§ 8 Group B file list omits it. Its `stamp_forms()` helper (:252-267)
hand-builds `real = {"adjacency": {...}, "barrier": {"rule":
"global_asymmetric", "combine": "any"}}` and its docstring calls that "the
real stamp". It is only ever fed to `monkeypatch.setattr(...,
"methodology_stamp", ...)`, so **it does not break** when the real stamp
gains `buffer_m` — verified by reading the two tests that use it. But it
becomes a stale copy of a shape that has moved, and the next reader will
trust the docstring. Update it and re-run the file both before and after, so
"it did not break" is a measured fact rather than an assumption.
`test_the_methodology_stamp_does_not_carry_roads` (:226-235) asserts
`set(stamp) == {"adjacency", "barrier"}` and that no block key contains
"roads" — both still hold with `buffer_m` inside `barrier`, so it must NOT
be edited.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# --- 3E: the compute-local weight column (spec § 2.3) ------------------
def test_the_stamp_records_the_buffer():
    """The buffer SHAPES the stored lists — a barrier 3 m off an edge blocks
    at 5 m and not at 2 m — so an artifact built at another buffer must be
    refused. It is None for the two rules that have no buffer."""
    from dataclasses import replace

    from delhi_psi.config import BarrierConfig, BarrierRule
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    assert pipeline.methodology_stamp(cfg.methodology)["barrier"] == {
        "rule": "global_asymmetric", "combine": "any", "buffer_m": None}
    partial = replace(cfg.methodology, barrier=BarrierConfig(
        rule=BarrierRule.PARTIAL_WEIGHTED, combine="any", buffer_m=5.0))
    assert pipeline.methodology_stamp(partial)["barrier"] == {
        "rule": "partial_weighted", "combine": "any", "buffer_m": 5.0}


def test_the_weight_column_is_dropped_before_index_frames_returns():
    """Like NBRS_DIST_BOUNDARY_COL: compute-local, so the CSV/shapefile
    column set is identical under every barrier rule."""
    from dataclasses import replace

    from delhi_psi.config import BarrierConfig, BarrierRule
    from delhi_psi.pipeline import compute_frames
    from tests.cities import ORACULUM
    from tests.oraculum_fixtures import methodology_with

    methodology = methodology_with("code-2025", types=(), stage=None)
    methodology = replace(methodology, barrier=BarrierConfig(
        rule=BarrierRule.PARTIAL_WEIGHTED, combine="any", buffer_m=5.0))
    got = compute_frames(ORACULUM.load_settlements(),
                         {"canal": ORACULUM.load_barriers()},
                         ORACULUM.load_services(), None, methodology, "pop",
                         mapping=ORACULUM.mapping(), scheme=ORACULUM.scheme)
    assert pipeline.NBRS_WEIGHT_COL not in got.columns

    baseline = compute_frames(ORACULUM.load_settlements(),
                              {"canal": ORACULUM.load_barriers()},
                              ORACULUM.load_services(), None,
                              methodology_with("code-2025", types=(),
                                               stage=None),
                              "pop", mapping=ORACULUM.mapping(),
                              scheme=ORACULUM.scheme)
    assert list(got.columns) == list(baseline.columns)


def test_the_weight_column_is_in_the_shapefile_drop_list():
    """`compute` cuts missing_population.csv from the NEIGHBOURS frame,
    where the column IS present, using this list."""
    from delhi_psi import io

    assert pipeline.NBRS_WEIGHT_COL in io.SHAPEFILE_DROP_COLUMNS


def test_pre_neighbours_exclusion_strips_the_weight_column_too():
    """An id removed from nbrs_bbox must leave the weight list as well, or
    pcen's KeyError guard fires on a legitimate run."""
    import geopandas as gpd
    from shapely.geometry import box

    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B", "C"],
         "nbrs_bbox": [["B", "C"], ["A"], ["A"]],
         "nbrs_dist_bbox": [[("B", 1.0), ("C", 2.0)], [("A", 1.0)],
                            [("A", 2.0)]],
         pipeline.NBRS_WEIGHT_COL: [[("B", 0.5), ("C", 1.0)], [("A", 0.5)],
                                    [("A", 1.0)]]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:7760")
    got = pipeline.apply_exclusion(frame, dropped={"C"},
                                   stage="pre_neighbors")
    row = got[got["USO_AREA_U"] == "A"].iloc[0]
    assert row["nbrs_bbox"] == ["B"]
    assert row[pipeline.NBRS_WEIGHT_COL] == [("B", 0.5)]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_pipeline.py -k "buffer or weight_column or weight"`
Expected: **FAIL** — `AttributeError: module 'delhi_psi.pipeline' has no
attribute 'NBRS_WEIGHT_COL'`, and `test_the_stamp_records_the_buffer` fails
on the dict comparison because the stamp has no `buffer_m` key. Record the
text.

- [ ] **Step 3: Add the constant, the stamp entry and the exclusion strip**

In `delhi_psi/pipeline.py`, next to `NBRS_DIST_BOUNDARY_COL`:

```python
# Written by apply_barrier ONLY under barrier.rule: partial_weighted, stored
# in the artifact (the weights are geometry, and `compute` never reads
# barrier layers), and dropped by `index_frames` before it returns — so the
# CSV/shapefile column set is rule-independent, like NBRS_DIST_BOUNDARY_COL.
NBRS_WEIGHT_COL = "nbrs_barrier_weight"
```

In `methodology_stamp`, the barrier block becomes:

```python
        "barrier": {
            "rule": str(methodology.barrier.rule),
            "combine": combine if isinstance(combine, str)
            else [str(layer) for layer in combine],
            # The buffer shapes the stored lists, so an artifact built at
            # another buffer must be refused. Artifacts from 3A-3D have no
            # such key: `stored.get(block, {}).get(key)` yields None, equal
            # to the configured None for both older rules, so code-2025's
            # pinned colonies_neighbors.joblib keeps loading.
            "buffer_m": methodology.barrier.buffer_m,
        },
```

In `apply_exclusion`, inside the `pre_neighbors` loop, after the
`NBRS_DIST_COL` line:

```python
            if NBRS_WEIGHT_COL in universe.columns:
                universe.at[idx, NBRS_WEIGHT_COL] = [
                    (j, w) for j, w in row[NBRS_WEIGHT_COL]
                    if j not in dropped]
```

- [ ] **Step 4: Wire `build_neighbors` and `index_frames`**

In `build_neighbors`, replace the `barrier_geoms` line and the
`apply_barrier` call:

```python
    log.info("barrier: rule=%s buffer_m=%s", methodology.barrier.rule,
             methodology.barrier.buffer_m)
    barrier_geoms = neighbors.selected_barrier_geoms(
        barriers, combine=methodology.barrier.combine)
    frame = neighbors.apply_barrier(frame, barrier_geoms, id_col=id_col,
                                    neighbor_col=NBRS_COL,
                                    rule=methodology.barrier.rule,
                                    buffer_m=methodology.barrier.buffer_m,
                                    weight_col=NBRS_WEIGHT_COL)
```

In `index_frames`, immediately after the boundary-distance block:

```python
    nbr_weight_col = (NBRS_WEIGHT_COL
                      if methodology.barrier.rule == "partial_weighted"
                      else None)
```

add `nbr_weight_col=nbr_weight_col,` to the `index.service_index(...)` call,
and replace the single-column drop at the end with:

```python
    for column in (NBRS_DIST_BOUNDARY_COL, NBRS_WEIGHT_COL):
        if column in result.columns:
            result = result.drop(columns=[column])
    return result
```

In `delhi_psi/io.py`:

```python
SHAPEFILE_DROP_COLUMNS = ("nbrs_bbox", "nbrs_dist_bbox", "centroid",
                          "nbrs_barrier_weight")
```

- [ ] **Step 5: Update the two stamp literals**

In `tests/test_cli.py::test_neighbors_artifact_carries_the_methodology_stamp`
(:297-300):

```python
    assert frame.attrs["methodology"] == {
        "adjacency": {"rule": "bbox", "max_distance_km": None},
        "barrier": {"rule": "global_asymmetric", "combine": "any",
                    "buffer_m": None},
    }
```

In `tests/test_measure_roads_access.py::stamp_forms` (:260-261), keep `real`
a faithful copy of the shape `methodology_stamp` returns:

```python
    real = {"adjacency": {"rule": "bbox", "max_distance_km": None},
            "barrier": {"rule": "global_asymmetric", "combine": "any",
                        "buffer_m": None}}
```

Do **not** touch `test_the_methodology_stamp_does_not_carry_roads` — it
asserts the top-level key SET and that no block key contains "roads", both
of which still hold.

- [ ] **Step 6: Run the affected files, then prove nothing regenerated**

Run: `uv run pytest -q -W error tests/test_pipeline.py tests/test_cli.py tests/test_measure_roads_access.py tests/test_production_fixtures.py`
Expected: PASS. `test_a_pre_3d_artifact_still_loads_for_a_bbox_config`
(test_cli.py:394) is the 3A-3D compatibility pin and must pass **unchanged**:
its stored stamp has no `buffer_m`, `stored.get(...)` yields None, and the
configured value is None for `global_asymmetric`.

Then:

```bash
uv run python scripts/generate_production_fixtures.py
git status --porcelain tests/fixtures/
```

Expected: **empty output**.

- [ ] **Step 7: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add delhi_psi/pipeline.py delhi_psi/io.py tests/test_pipeline.py \
        tests/test_cli.py tests/test_measure_roads_access.py
git commit -m "$(cat <<'MSG'
feat(barrier): pipeline carries the weights, stamps the buffer, drops the column (DEL-48)

build_neighbors forwards buffer_m and the combine-selected barrier
geometries; the weight column is stored in the artifact, stripped alongside
the neighbour lists by a pre_neighbors exclusion, handed to pcen, and dropped
before index_frames returns — the NBRS_DIST_BOUNDARY_COL precedent — so the
output column set is identical under every barrier rule. io's drop list gains
it too, because missing_population.csv is cut from the neighbours frame,
where the column IS present.

buffer_m joins the methodology stamp: it shapes the stored lists, so an
artifact built at another buffer has to be refused. Pre-3E artifacts have no
such key, which reads back as None and equals the configured None for both
older rules — code-2025's pinned artifact keeps loading.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 6: The `partial_5m` variant — table, methodology branch, regenerated CSVs

**Files:**
- Modify: `tests/variants.py` (the `VARIANTS` table — append LAST)
- Modify: `tests/oraculum_fixtures.py` (`variant_methodology` :155-191)
- Modify: `tests/test_config.py` (`enum_key` in
  `test_every_variant_block_is_one_the_loader_accepts` :495-497)
- Modify: `tests/test_variant_rules.py` (one wiring assertion)
- Regenerate: `tests/fixtures/oraculum/variants_expected_values.csv`,
  `tests/fixtures/messy/variants_expected_values.csv`

**Interfaces:**
- Consumes from Tasks 1-5: everything. This is the integration commit.
- Produces, and **Group C relies on this**: `variant_methodology`'s
  per-block branch pattern extended to a non-adjacency/decay block, and a
  `VARIANTS` row whose block is stated IN FULL.

**Why this is one commit.** Four tests read `tests/variants.py` and each
fails on a row that is not yet everywhere: the loader check in
`test_config.py`, the two CSV checks in `test_reference_impl.py`, and the
production comparison in `test_variants_match_reference.py`. Splitting this
leaves the suite red.

- [ ] **Step 1: Add the variant row — LAST in the table**

In `tests/variants.py`, append to `VARIANTS` (position matters: Step 5's
byte-identity check relies on the new rows being appended at the END of both
CSVs, and `emit_variant_expected_values` iterates `VARIANT_RULESETS`, which
is a dict comprehension over `VARIANTS` and so preserves insertion order):

```python
    # DEL-48: the only genuinely fractional weight either fixture city can
    # produce. Oraculum's canal is [25, 475] on the 500 m A-D edge; its 5 m
    # round-capped buffer covers [20, 480], so w_AD = 1 - 460/500 = 0.08 and
    # every other link is untouched (spec § 6.1). Degenerate on the messy
    # city — it has no barriers, so every weight is 1 and its rows equal the
    # `code` base, exactly as `boundary` is degenerate on Oraculum.
    "partial_5m": {
        "barrier": {"rule": "partial_weighted", "combine": "any",
                    "buffer_m": 5.0},
    },
```

- [ ] **Step 2: Teach `variant_methodology` the barrier block**

In `tests/oraculum_fixtures.py::variant_methodology`, extend the local
import and add the branch before the `decay` one:

```python
    from delhi_psi.config import (
        AdjacencyConfig, AdjacencyRule, BarrierConfig, BarrierRule,
        DecayConfig, DecayDistance, DecayForm,
    )
    ...
    if "barrier" in spec:
        block = spec["barrier"]
        methodology = replace(methodology, barrier=BarrierConfig(
            rule=BarrierRule(block["rule"]),
            combine=block["combine"],
            buffer_m=block.get("buffer_m")))
```

and update the docstring's last paragraph — it says "today only the band
variants override `adjacency`"; make it name the barrier block too.

- [ ] **Step 3: Extend the loader's enum check**

In `tests/test_config.py::test_every_variant_block_is_one_the_loader_accepts`,
the `enum_key` map gains one line:

```python
    enum_key = {("adjacency", "rule"): "methodology.adjacency.rule",
                ("barrier", "rule"): "methodology.barrier.rule",
                ("decay", "form"): "methodology.decay.form",
                ("decay", "distance"): "methodology.decay.distance"}
```

- [ ] **Step 4: Pin the wiring on the reference side**

Append to `tests/test_variant_rules.py`:

```python
def test_the_partial_5m_variant_is_the_code_base_plus_the_barrier_rule():
    """The table, the knob map and the hand anchors are one thing: the
    variant's rule-set must BE the dict the § 6.1 anchors were derived
    under."""
    assert VARIANT_RULESETS["partial_5m"] == PARTIAL_5M


def test_partial_5m_is_degenerate_on_the_messy_city():
    """No barriers, so every weight is 1 and the rows are the `code` base's
    — stated, like `boundary` on Oraculum, so the CSV rows are not mistaken
    for a proof they are not."""
    base = scored(MESSY, RULESETS["code"])
    got = variant(MESSY, "partial_5m")
    for column in base.columns:
        assert list(got[column]) == pytest.approx(list(base[column]),
                                                  abs=1e-12), column
```

- [ ] **Step 5: Regenerate both CSVs and prove ADDITION ONLY**

Save the committed files first, regenerate, then compare. The check is
exact: the regenerated file's first N lines must be **byte-identical** to
the committed file, and every remaining line must belong to `partial_5m`.

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
uv run python - <<'PY'
import subprocess, sys

ok = True
for city in ("oraculum", "messy"):
    path = f"tests/fixtures/{city}/variants_expected_values.csv"
    old = subprocess.run(["git", "show", f"HEAD:{path}"], check=True,
                         capture_output=True).stdout.splitlines(keepends=True)
    with open(path, "rb") as handle:
        new = handle.readlines()
    same = new[:len(old)] == old
    added = new[len(old):]
    only_new = all(line.startswith(b"partial_5m,") for line in added)
    print(f"{city}: pre-existing block byte-identical={same} "
          f"added_rows={len(added)} all_partial_5m={only_new}")
    ok = ok and same and only_new and added
sys.exit(0 if ok else 1)
PY
git diff --stat tests/fixtures/
```

Expected: `pre-existing block byte-identical=True` and
`all_partial_5m=True` for **both** cities, a non-zero `added_rows` for each,
and `git diff --stat` showing ONLY the two `variants_expected_values.csv`
files, with insertions and **zero deletions**. Anything else — a changed
`expected_values.csv`, a changed `production/*.csv`, a deletion in a variants
CSV — is the owner's hard condition (spec § 11): **STOP and report**, do not
commit.

- [ ] **Step 6: Run the invariants guard**

Run: `uv run python scripts/check_oraculum_invariants.py`
Expected: `OK`, exit 0. The generators already ran it before writing (that is
what `emit_checked_variant_expected_values` does), so this is the standalone
confirmation that `partial_5m` creates no degenerate min-max group and no
tied clinic/school anchor on either city.

- [ ] **Step 7: Run the variant suites**

Run: `uv run pytest -q -W error tests/test_variant_rules.py tests/test_variants_match_reference.py tests/test_reference_impl.py tests/test_config.py`
Expected: PASS. `test_production_matches_the_reference_on_each_variant` now
runs `partial_5m` on both cities × both denominators — that is production ==
reference at 1e-12 on the fractional weight, and it is the core proof of
this ticket. The messy leg is the empty-barrier path from Task 3 Step 4.

- [ ] **Step 8: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add tests/variants.py tests/oraculum_fixtures.py tests/test_config.py \
        tests/test_variant_rules.py \
        tests/fixtures/oraculum/variants_expected_values.csv \
        tests/fixtures/messy/variants_expected_values.csv
git commit -m "$(cat <<'MSG'
test(barrier): the partial_5m variant, on both cities (DEL-48)

partial_weighted ships as a VARIANT, never as a change to the ideal/code
rule-sets, so every existing expected value stays byte-identical by
construction: both variants CSVs changed by the ADDITION of partial_5m rows
alone, verified line by line against the committed files.

Oraculum carries the fractional pin (w_AD = 0.08, the only fractional weight
either fixture city can produce); the messy city is degenerate under this
rule — no barriers, so every weight is 1 — and its rows equal the code base,
which is what exercises the empty-barrier path.

test_variants_match_reference now proves production == the reference at 1e-12
on the new variant, both cities, both denominators.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 7: The CLI leg and the stamp's `buffer_m`

**Files:**
- Modify: `tests/test_variants_match_reference.py` (the parametrization at
  :84, plus one column-set assertion)
- Modify: `tests/test_cli.py` (the 3D stamp section, after :427)

**Interfaces:**
- Consumes from Tasks 5 and 6: the stamp's `buffer_m` entry and the
  `partial_5m` variant row.
- Produces: no new production names.

- [ ] **Step 1: Add `partial_5m` to the CLI round trip**

In `tests/test_variants_match_reference.py`, change the parametrization of
`test_a_derived_variant_profile_runs_end_to_end` and extend its docstring:

```python
@pytest.mark.parametrize("variant", ["band_small_boundary", "exp1",
                                     "partial_5m"])
def test_a_derived_variant_profile_runs_end_to_end(expected, data_dir,  # noqa: F811
                                                   tmp_path, variant):
    """Proves the whole chain the in-memory test skips: YAML -> load_config
    -> preprocess -> the stamped artifact -> compute -> CSV. `exp1` is here
    for `scale_km`; `band_small_boundary` for the band, the boundary
    distance and the stamped `max_distance_km` together; `partial_5m` for
    the barrier weights, which are computed in `preprocess`, stored in the
    artifact, stamped with `buffer_m: 5.0`, and consumed by `compute`.
    """
```

and append to that test, after the existing value comparison:

```python
    # The weight column never leaves index_frames: the output column set is
    # the same under every barrier rule.
    plain = oracle_profile_path(BASE_PROFILE, tmp_path,
                                methodology_overrides={
                                    "exclusion": BASELINE_EXCLUSION},
                                name=f"{variant}_plain")
    plain_out = tmp_path / f"{variant}_plain"
    assert cli.main(["preprocess", "--config", str(plain),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(plain_out)]) == 0
    assert cli.main(["compute", "--config", str(plain),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(plain_out)]) == 0
    plain_csv = pd.read_csv(
        plain_out / "delhi_psi_code-2025_pop_2020.csv")
    assert list(got.reset_index().columns) == list(plain_csv.columns)
```

- [ ] **Step 2: Write the failing stamp tests**

Append to `tests/test_cli.py`, in the stamp section after
`test_changing_only_the_decay_does_not_invalidate_an_artifact` (:427):

```python
# --- 3E: the buffer is part of the stamp (spec § 6.5) ------------------
def _partial_config(buffer_m):
    from dataclasses import replace

    from delhi_psi.config import BarrierConfig, BarrierRule
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    return replace(cfg, methodology=replace(
        cfg.methodology,
        barrier=BarrierConfig(rule=BarrierRule.PARTIAL_WEIGHTED,
                              combine="any", buffer_m=buffer_m)))


def test_an_artifact_built_at_another_buffer_is_refused():
    """The buffer shapes the stored lists — a barrier 3 m off an edge blocks
    at 5 m and not at 2 m — so every number a mismatched compute produced
    would describe a neighbourhood nobody built."""
    from delhi_psi import pipeline, validate

    frame = _stamped(_partial_config(5.0).methodology)
    with pytest.raises(validate.ValidationError) as exc:
        pipeline.check_methodology_stamp(frame, _partial_config(2.0))
    message = str(exc.value)
    assert "buffer_m" in message and "5.0" in message and "2.0" in message


def test_an_artifact_built_under_pairwise_is_refused_by_partial_weighted():
    """The existing rule check, on the new value."""
    from dataclasses import replace

    from delhi_psi import pipeline, validate
    from delhi_psi.config import BarrierConfig, BarrierRule
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    pairwise = replace(cfg.methodology, barrier=BarrierConfig(
        rule=BarrierRule.PAIRWISE, combine="any"))
    frame = _stamped(pairwise)
    with pytest.raises(validate.ValidationError, match="rule"):
        pipeline.check_methodology_stamp(frame, _partial_config(5.0))


def test_a_pre_3e_artifact_still_loads_for_a_bbox_config():
    """3A-3D artifacts have no `buffer_m` key: `stored.get(...)` yields None,
    which equals the configured None for both rules that have no buffer — so
    code-2025's pinned colonies_neighbors.joblib keeps loading without a
    re-preprocess (the 3D max_distance_km precedent)."""
    from delhi_psi import pipeline
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    frame = pd.DataFrame({"USO_AREA_U": ["A"]})
    frame.attrs["profile"] = "code-2025"
    frame.attrs["methodology"] = {
        "adjacency": {"rule": "bbox", "max_distance_km": None},
        "barrier": {"rule": "global_asymmetric", "combine": "any"},
    }
    pipeline.check_methodology_stamp(frame, cfg)      # must not raise
```

- [ ] **Step 3: Run them**

Run: `uv run pytest -q -W error tests/test_cli.py -k "buffer or pre_3e or pairwise_is_refused"`
Expected: **PASS immediately** — the stamp entry landed in Task 5. These
tests are written after their implementation, so prove they are meaningful:
temporarily delete the `"buffer_m"` line from `pipeline.methodology_stamp`,
re-run, confirm `test_an_artifact_built_at_another_buffer_is_refused` FAILS
(no `ValidationError` raised at all), then restore the line. Report both
results.

- [ ] **Step 4: Run the CLI leg**

Run: `uv run pytest -q -W error tests/test_variants_match_reference.py`
Expected: PASS, including the new `partial_5m` CLI case at 1e-9 against the
variants CSV, and the column-set assertion.

- [ ] **Step 5: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add tests/test_variants_match_reference.py tests/test_cli.py
git commit -m "$(cat <<'MSG'
test(barrier): the CLI leg and the stamped buffer (DEL-48)

partial_5m joins the derived-variant round trip — YAML, load_config,
preprocess (weights computed and stored), the stamp, compute, CSV at 1e-9 —
and the output CSV's column set is asserted equal to a plain code-2025 run's,
so the weight column provably never leaves index_frames.

Stamp tests: an artifact at buffer_m 5.0 is refused by a config at 2.0
naming all three, a pairwise artifact is refused by partial_weighted, and a
3A-3D-shaped stamp with no buffer_m key still loads for the two rules that
have no buffer.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 8: Production == reference on synthetic geometry

**Files:**
- Test: `tests/test_reference_impl.py` (one new test, at the end)

**Interfaces:**
- Consumes: `pipeline.compute_frames`, `config.MethodologyConfig` and its
  sub-configs (built directly, no YAML), and
  `reference_impl.compute_city(barrier_rule="partial_weighted",
  barrier_buffer_m=5.0, ...)`.
- Produces: no new names. Group C extends this same test with the overlap
  part, so keep the city construction in a module-level helper it can reuse.

**Why this exists** (spec § 6.4, § 12 item 3): the fractional-weight ×
overlap × MultiPolygon case cannot go into either fixture city without
changing an existing expected value — any barrier touching a settlement with
a `bbox` neighbour moves that settlement's rows under both committed
rule-sets, because the fixture's barrier file is shared by every rule-set.
So it lives in synthetic in-test geometry scored by BOTH implementations, and
costs no fixture file.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reference_impl.py`:

```python
# --- 3E: production == reference on synthetic geometry (spec § 6.4) ----
def synthetic_partial_city():
    """Three settlements built for the case no fixture city can carry.

    P and Q OVERLAP (so their shared boundary is the intersection polygon's
    perimeter); R is a two-part MultiPolygon TOUCHING P along both parts
    (a MultiLineString shared boundary); a canal partially covers the P-R
    boundary, so at least one weight is strictly between 0 and 1.

    EVERY point service the reference scores gets a layer, because
    `compute_city` min-maxes all six of `POINT_SERVICES` plus road and
    DEL-54's guard raises on a constant column — a city with only a clinic
    layer would make school/bank/police/ration/transport all-zero and stop
    the comparison before it started. One point per settlement, reused for
    every service: the three denominators (100, 200, 400) are distinct, so
    no PCEN column can be constant. Each point is strictly interior to
    exactly one settlement and none lies in the P-Q overlap, so
    production's boundary-inclusive `intersects` and the reference's strict
    `within` agree on every one (rule-set gap #6 is not in scope here).
    """
    import geopandas as gpd
    from shapely.geometry import LineString, MultiPolygon, Point, box

    settlements = gpd.GeoDataFrame(
        {"USO_AREA_U": ["P", "Q", "R"], "USO_FINAL": ["Planned"] * 3,
         "population": [100.0, 200.0, 400.0],
         "area_km2": [1.2, 1.0, 0.8]},
        geometry=[box(0, 0, 1200, 1000),
                  box(1000, 0, 2000, 1000),
                  MultiPolygon([box(-400, 0, 0, 400),
                                box(-400, 600, 0, 1000)])],
        crs="EPSG:7760")
    # The canal covers y in [0, 400] of the x = 0 boundary P shares with R's
    # lower part, and reaches 5 m past each end: 410 of the 800 m shared
    # boundary, so w_PR is strictly fractional.
    barriers = gpd.GeoDataFrame(
        {"name": ["canal"]},
        geometry=[LineString([(0, 0), (0, 400)])], crs="EPSG:7760")
    # P only, Q only, R's upper part only — none in the P-Q overlap.
    hosts = [Point(600, 500), Point(1600, 500), Point(-200, 800)]
    services = {
        name: gpd.GeoDataFrame({"service": [name] * 3},
                               geometry=list(hosts), crs="EPSG:7760")
        for name in ("clinic", "school", "bank", "police", "ration",
                     "transport")
    }
    # 300 m inside R's lower part, 1100 m inside P, 100 m inside Q (the
    # overlap stretch counts for both owners, on both sides) — three
    # distinct lengths, so road_pcen is not constant either.
    services["road"] = gpd.GeoDataFrame(
        {"service": ["road"]},
        geometry=[LineString([(-300, 200), (1100, 200)])], crs="EPSG:7760")
    return settlements, barriers, services


def test_production_matches_the_reference_on_synthetic_partial_geometry():
    """The fractional-weight x overlap x MultiPolygon case, scored by BOTH
    implementations at 1e-12. It cannot live in a fixture city without
    moving an existing expected value (spec § 12 item 3), so it lives here
    and costs no fixture file.
    """
    from delhi_psi.config import (
        AbsentNeighbor, AdjacencyConfig, AdjacencyRule, BarrierConfig,
        BarrierRule, DecayConfig, DecayDistance, DecayForm, ExclusionConfig,
        ExclusionStage, MethodologyConfig, RoadsFormula,
    )
    from delhi_psi.pipeline import compute_frames
    from tests.test_profiles_match_reference import METRIC_MAP

    settlements, barriers, services = synthetic_partial_city()
    methodology = MethodologyConfig(
        adjacency=AdjacencyConfig(rule=AdjacencyRule.BBOX),
        barrier=BarrierConfig(rule=BarrierRule.PARTIAL_WEIGHTED,
                              combine="any", buffer_m=5.0),
        decay=DecayConfig(form=DecayForm.INVERSE_LINEAR, distance_unit="km",
                          distance=DecayDistance.CENTROID),
        roads=RoadsFormula.DECAYED,
        second_normalization=True,
        exclusion=ExclusionConfig(types=(), stage=ExclusionStage.POST_NEIGHBORS,
                                  absent_neighbor=AbsentNeighbor.SWALLOWED))

    for denom in ("pop", "popdensity"):
        got = compute_frames(settlements, {"canal": barriers}, services, None,
                             methodology, denom,
                             mapping={"Planned": "Planned"},
                             scheme="synthetic").set_index("USO_AREA_U")
        exp = compute_city(
            settlements, services, barriers, adjacency_rule="bbox",
            barrier_rule="partial_weighted", barrier_buffer_m=5.0,
            roads_formula="decayed", scenario="none", denom=denom,
            second_norm=True, absent_neighbor_contribution="swallowed",
            scenarios={"none": (frozenset(), False)})
        assert set(got.index) == set(exp.index)
        # Every METRIC_MAP column exists on both sides: the city carries all
        # six point services plus road, and second_norm is on, so nothing is
        # skipped and the comparison cannot pass by omission.
        for prod_col, metric in METRIC_MAP.items():
            for sid in exp.index:
                assert got.loc[sid, prod_col] == pytest.approx(
                    exp.loc[sid, metric], abs=1e-12), (denom, sid, prod_col)


def test_the_synthetic_city_really_carries_a_fractional_weight():
    """The test above would still pass if every weight were 1 or 0 — that is
    exactly the failure mode it exists to rule out."""
    from tests.reference_impl import adjacency, partial_weights

    settlements, barriers, _ = synthetic_partial_city()
    weights = partial_weights(adjacency(settlements, "bbox"), settlements,
                              barriers, 5.0)
    fractional = [w for w in weights.values() if 0.0 < w < 1.0]
    assert fractional, weights
    assert weights[("P", "R")] == weights[("R", "P")]
```

- [ ] **Step 2: Run them and record the result**

Run: `uv run pytest -q -W error tests/test_reference_impl.py -k synthetic`

Expected: **PASS** if both implementations agree, which is the point. This
test is written after both sides exist, so a RED here means a real
divergence, not a missing symbol. If it fails:
- read the failing `(denom, sid, prod_col)` tuple;
- if the gap is at the 1e-16 level and only on a weight, it is the
  union-order difference the spec anticipates (§ 9: production unions the
  blocked PIECES from STRtree candidates, the reference unions all buffers
  first). **The reference adopts the piece-union form — never the other way
  round.**
- if it is larger, one implementation has the definition wrong. Use
  `test_the_synthetic_city_really_carries_a_fractional_weight` and the § 6.4
  unit pins from Tasks 1 and 3 to find which.

If `test_the_synthetic_city_really_carries_a_fractional_weight` fails, the
geometry is wrong, not the arithmetic: adjust the canal's extent until
`w_PR` lands strictly inside (0, 1), and say in the docstring what it became.

- [ ] **Step 3: Confirm no service column is constant**

Run: `uv run pytest -q -W error tests/test_reference_impl.py -k synthetic -x`
and confirm no `ValueError` mentioning `min-max of ... is undefined` appears.
That guard is DEL-54's, on both sides; a constant column here would mean the
clinic/road placement needs another point, not that the guard is wrong.

- [ ] **Step 4: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add tests/test_reference_impl.py
git commit -m "$(cat <<'MSG'
test(barrier): production == reference on synthetic partial-barrier geometry (DEL-48)

The fractional-weight x overlap x MultiPolygon case, scored by both
implementations at 1e-12. It cannot go into either fixture city without
moving an existing expected value — the barrier file is shared by every
rule-set — so it lives in in-test geometry and costs no fixture file, the
pattern test_index.city_with_neighbours already uses.

A second test asserts the city really does carry a strictly fractional
weight, so the comparison cannot pass vacuously.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 9: Docs — worksheet, config doc, CHANGELOG, WORKPLAN

**Files:**
- Modify: `docs/oracle/derivation-worksheet.md` (a new section BELOW
  "Machine-checked remainder")
- Modify: `docs/methodology-config.md` (§ 1 table and the two paragraphs
  under it; § 4 proof list)
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `WORKPLAN.md` (cycle 3E, DEL-48)

**Interfaces:**
- Consumes: the numbers pinned in Tasks 1 and 6. Every number written here
  must already be asserted by a test — nothing new is derived in prose.

- [ ] **Step 1: The worksheet section**

Append to `docs/oracle/derivation-worksheet.md`, **below** the
"Machine-checked remainder" section, so the RATIFIED content above is
untouched and `tests/test_manuscript_anchors.py` — which quotes only the
ratified sections — cannot see it:

```markdown
## partial_weighted (variant `partial_5m`, 5 Sep 2026 — machine-derived, ratification is the owner's)

NOT RATIFIED. Everything above this heading was hand-checked and signed off
on 2026-08-24; this section was derived by the DEL-48 implementation and is
pinned by `tests/test_variant_rules.py`, which is what makes it re-derivable
rather than authoritative.

**The weight.** For a directed link i→j, w_ij = 1 − L_blocked / L_shared,
where L_shared is the length of the boundary i and j share and L_blocked is
the part of it within `buffer_m` metres of a barrier. The buffer is a
DISTANCE with round caps, so the blocked span extends `buffer_m` past each
end of the barrier.

**On this city.** The canal is the segment x ∈ [25, 475] at y = 1000, which
lies strictly inside the 500 m A–D edge x ∈ [0, 500]. At `buffer_m: 5` it
blocks x ∈ [20, 480]:

    L_shared  = 500 m
    L_blocked = 460 m
    w_AD = w_DA = 1 − 460/500 = 0.08     (float 0.07999999999999996)

At `buffer_m: 1` it blocks [24, 476] and w_AD = 0.096. The buffer-free limit
is the memo's 0.1, which is deliberately not offered: `LineString.buffer(0)`
is EMPTY in shapely, so a 0 m buffer would silently make every weight 1.

Every other shared boundary on this city is at least 20 m from the canal's
ends — A–E begins at x = 500, and the buffer stops at 480 — so **A–D is the
only fractional link, in both directions**, and nothing is pruned. The lists
are therefore the plain `bbox` lists, and A and D are back in everyone's,
because the `global_asymmetric` flag severing is gone:

    A:[B,D,E]  B:[A,C,E,RV]  C:[B,E,IND]  RV:[B]
    D:[A,E]    E:[A,B,C,D,IND]  IND:[C,E]

**Decays.** 1 km → ½; 1.5 km → 0.4; √2 km → √2−1; A and D centroids are
(500, 1500) and (0, 500), i.e. √5/2 km apart → 1/(1+√5/2) = 0.4721359549995794.

**Anchors** (`pop` denominator, the `code` base — decayed roads, `swallowed`,
nothing dropped):

| row | arithmetic | value |
|---|---|---|
| D clinic | (0 + 0.08·2·0.4721359550 [A] + 1·0.4 [E]) / 100 | 0.0047554175279993 |
| D school | (1 + 0.08·1·0.4721359550 [A] + 1·0.4 [E]) / 100 | 0.0143777087639997 |
| A school | (1 + 0.08·1·0.4721359550 [D] + 1·(√2−1) [E]) / 100 | 0.0145198443877306 |
| A clinic | unchanged — D owns no clinic: (2 + ½ + (√2−1))/100 | 0.0291421356237309 |
| D road (decayed) | (0 + 0.08·0.75·0.4721359550 [A] + 0.75·0.4 [E]) / 100 | 0.0032832815729997 |
| B clinic | A is back: (1 + 2·½ + 0 + 2·½ + 1·½)/200 — the `ideal` 0.0175, not `code`'s 0.0125 | 0.0175 |
| E clinic | (1 + 2·(√2−1) + ½)/300 — the `ideal` value | 0.0077614237491540 |

`popdensity` changes only the denominators (D and A have area 1.0 km², so
their rows are identical; E divides by 150). No service column is constant
under either denominator, which is what lets the invariants guard write the
fixture at all.
```

- [ ] **Step 2: The config doc, § 1**

In `docs/methodology-config.md`:

- In the switch table, the `barrier.rule` row loses "reserved":

```markdown
| `barrier.rule` | `global_asymmetric` | `pairwise` | **`partial_weighted`** — a barrier covering part of a shared boundary discounts that neighbour by the covered share, w_ij = 1 − L_blocked/L_shared, instead of severing it. Landed in cycle 3E (DEL-48) | memo § 2 (DEL-22) — `pairwise` severs the crossing pair only; `partial_weighted` generalises it |
```

- Add a `barrier.buffer_m` row immediately below it:

```markdown
| `barrier.buffer_m` | — (unused) | — (unused) | `5` m, pending ratification | DEL-48 — how close a barrier has to be to block a boundary point, in metres (EPSG:7760 is metric). **Required** iff `barrier.rule: partial_weighted`, **rejected** otherwise, and strictly **> 0**: `LineString.buffer(0)` is EMPTY in shapely, so a 0 m buffer would silently make every weight 1. It is a DISTANCE with round caps, so the blocked span extends `buffer_m` past each end of the barrier — on the oracle city that makes w_AD 0.08, not the memo's buffer-free 0.1 |
```

- In the "Reserved" paragraph, drop `barrier.rule: partial_weighted` so only
  `outputs.denominators: one` and the key `exclusion.minmax_universe`
  remain. Do NOT add `overlap.counting` — that is Group C's.
- Leave the "One thing Bob added needs code, not config" paragraph as it is:
  it describes DEL-20, which is Group C.

- [ ] **Step 3: The config doc, § 4**

In the `tests/test_variant_rules.py` bullet, add `partial_5m` to the list of
pinned variants and name what it pins (the fractional weight on the A–D
edge, the symmetry, the smaller-buffer comparison). In the
`tests/test_variants_match_reference.py` bullet, change "all eight derived
variants" to **nine** and add the `partial_5m` CLI case and the stamped
`buffer_m` to the sentence about the round trip. Add one bullet:

```markdown
- `tests/test_reference_impl.py` — the synthetic production-vs-reference
  city: a fractional barrier weight on an overlapping pair with a
  MultiPolygon neighbour, scored by BOTH implementations at 1e-12. It cannot
  live in a fixture city without moving an existing expected value, so it
  lives in in-test geometry.
```

§ 5 stays as written — its five numbered steps are exactly the order this
plan followed, and Task 1's ordering constraint is step 2 of that list.

- [ ] **Step 4: CHANGELOG**

At the top of `[Unreleased]`, one entry: `partial_weighted` on both sides;
the buffer as a distance with round caps and why > 0; the weight column that
exists only under this rule and never leaves `index_frames`; `buffer_m` in
the stamp; **and explicitly** that `combine` now selects the layers the
geometry rules see (spec § 9 calls this out as a behaviour change for
`pairwise` with a non-`any` combine — no profile, fixture or output uses that
combination). State that no existing expected value moved and that the two
variants CSVs changed by addition only. Mention this is the second of cycle
3E's three per-ticket PRs, after DEL-54.

- [ ] **Step 5: WORKPLAN**

In `WORKPLAN.md`'s cycle 3E item, tick DEL-48 in the style the neighbouring
completed items use, naming the branch and the anchor (w_AD = 0.08). Update
DEL-31's blocker list: DEL-48 is no longer a blocker; DEL-20 still is.

- [ ] **Step 6: Prove the ratified anchors did not move**

Run: `uv run pytest -q -W error tests/test_manuscript_anchors.py tests/test_fixture_invariants.py`
Expected: PASS. The worksheet's ratified sections are untouched and the canal
was not redrawn, so
`test_canal_inside_ad_edge_touches_exactly_a_and_d` and
`test_road_lengths_and_canal_clearance` hold as written.

- [ ] **Step 7: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add docs/oracle/derivation-worksheet.md docs/methodology-config.md \
        CHANGELOG.md WORKPLAN.md
git commit -m "$(cat <<'MSG'
docs(barrier): the partial_weighted anchor, the switch and the buffer (DEL-48)

The worksheet gains a section BELOW the ratified content, clearly labelled
machine-derived: the definition, the canal arithmetic (460 of 500 m blocked
at a 5 m buffer, so w_AD = 0.08), the unpruned lists, and the seven PCEN
anchors the reference pins. The ratified sections and the manuscript-anchors
test are untouched.

The config doc's barrier.rule row loses "reserved", gains a buffer_m row
stating the distance semantics and the > 0 rule, and the reserved paragraph
drops the value. Every number in prose is one a test already asserts.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 10: `scripts/measure_rule_effects.py` — the `partial_barriers` block

**Files:**
- Create: `scripts/measure_rule_effects.py`
- Create: `docs/data/rule_effects.md` (prose + provenance; the fenced block
  is pasted by Task 11)
- Test: `tests/test_measure_rule_effects.py`

**Interfaces:**
- Consumes: `scripts/_measure_common.{resolve_work_dir, load_settlements,
  render, parse_block, FENCE}`, and `measure_roads_access`'s
  `base_profile_path` / `stage_artifacts` shapes (generalised here).
- Produces: `measure_rule_effects.derived_profile(base, work_dir, *,
  profile_name, methodology)`, `weight_classes(frame, *, weight_col)`,
  `measure_partial_barriers(cfg, work_dir, *, base, verify_dir)`,
  `measure_effect(...)` (the per-type diff), `main(argv=None)`. Group C adds
  a second block to this same script.

**Note on the artifact.** Unlike the roads run, the barrier block **changes
the stamp**, so the proven artifact CANNOT be reused: this block runs
`preprocess` into its own `--work-dir`. That is the opposite of
`stage_artifacts`' contract, so do not reach for it here.

- [ ] **Step 1: Write the failing fixture-level tests**

Create `tests/test_measure_rule_effects.py`:

```python
"""The rule-effect measurement (DEL-48, spec § 7).

The weight classification is proven on Oraculum, where the answer is known
by hand: 10 undirected bbox pairs = 20 directed links, of which A-D and D-A
are fractional and none is severed. The real-data drift test needs
DELHI_PSI_MEASURE_CACHE — the shared warm work dir the run step exports — or
it skips.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from delhi_psi import cli
from scripts import measure_rule_effects
from scripts._measure_common import FENCE, parse_block
from scripts.measure_rule_effects import (
    WEIGHT_CLASSES, derived_profile, weight_classes,
)
from tests.oraculum_fixtures import oracle_profile_path
from tests.test_cli import data_dir  # noqa: F401 — the Oraculum data dir
from tests.test_measure_common import (
    DATA_DIR, MEASURE_CACHE, assert_prose_numbers_come_from_the_blocks,
    needs_measure_cache,
)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "rule_effects.md"
VERIFY_DIR = DATA_DIR / "phase3_verify"
PARTIAL_PROFILE = "partial-barriers-5m"


def test_weight_classes_counts_directed_links_by_class():
    frame = pd.DataFrame({
        "USO_AREA_U": ["A", "B", "C"],
        "nbrs_barrier_weight": [[("B", 1.0), ("C", 0.08)],
                                [("A", 1.0)],
                                [("A", 0.08)]]})
    got = weight_classes(frame, weight_col="nbrs_barrier_weight")
    assert got["links_w_one"] == 2
    assert got["links_fractional"] == 2
    assert got["links_severed"] == 0
    assert got["median_fractional_w"] == "0.08"


def test_the_derived_profile_changes_only_the_barrier_block(tmp_path):
    """One factor: the barrier rule and its buffer, nothing else. `bbox` is
    kept deliberately, so the diff is attributable to the barrier rule
    alone."""
    base = oracle_profile_path("code-2025", tmp_path)
    path = derived_profile(base, tmp_path / "run",
                           profile_name=PARTIAL_PROFILE,
                           methodology={"barrier": {
                               "rule": "partial_weighted", "combine": "any",
                               "buffer_m": 5}})
    got = yaml.safe_load(path.read_text())
    expected = yaml.safe_load(Path(base).read_text())
    expected["profile"] = PARTIAL_PROFILE
    expected["methodology"]["barrier"] = {"rule": "partial_weighted",
                                          "combine": "any", "buffer_m": 5}
    expected["paths"].pop("neighbors_artifact", None)
    expected["paths"].pop("out_dir", None)
    assert got == expected


def test_the_oraculum_weight_classes_are_the_hand_counted_ones(data_dir,  # noqa: F811
                                                               tmp_path):
    """10 undirected bbox pairs -> 20 directed links; only A-D is fractional,
    in both directions; nothing is severed (spec § 7)."""
    from delhi_psi import io

    base = oracle_profile_path("code-2025", tmp_path)
    run_dir = tmp_path / "partial"
    profile = derived_profile(base, run_dir, profile_name=PARTIAL_PROFILE,
                              methodology={"barrier": {
                                  "rule": "partial_weighted",
                                  "combine": "any", "buffer_m": 5}})
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0
    frame = io.read_neighbors(
        run_dir / f"colonies_neighbors_{PARTIAL_PROFILE}.joblib")
    got = weight_classes(frame, weight_col="nbrs_barrier_weight")
    assert got["links_w_one"] == 18
    assert got["links_fractional"] == 2
    assert got["links_severed"] == 0
    assert got["median_fractional_w"] == "0.08"


def test_the_stamp_records_the_buffer_on_the_derived_run(data_dir,  # noqa: F811
                                                         tmp_path):
    """The artifact CANNOT be reused from --verify-dir here: the barrier
    block is in the stamp, so this block re-runs preprocess. Pinned, so a
    future 'optimisation' that stages the proven artifact fails loudly."""
    from delhi_psi import io

    base = oracle_profile_path("code-2025", tmp_path)
    run_dir = tmp_path / "stamped"
    profile = derived_profile(base, run_dir, profile_name=PARTIAL_PROFILE,
                              methodology={"barrier": {
                                  "rule": "partial_weighted",
                                  "combine": "any", "buffer_m": 5}})
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0
    frame = io.read_neighbors(
        run_dir / f"colonies_neighbors_{PARTIAL_PROFILE}.joblib")
    assert frame.attrs["methodology"]["barrier"] == {
        "rule": "partial_weighted", "combine": "any", "buffer_m": 5.0}


def committed_block():
    if not DOC.exists() or FENCE not in DOC.read_text():
        pytest.skip(f"{DOC} carries no measured block yet — the run step "
                    "pastes it")
    return parse_block(DOC.read_text(), name="partial_barriers")


def test_the_doc_block_has_every_required_key():
    block = committed_block()
    for key in (*WEIGHT_CLASSES, "median_fractional_w",
                "settlements_list_changed", "links_kept_code_2025",
                "links_kept_partial", "preprocess_seconds"):
        assert key in block, key


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, (committed_block(),))


@needs_measure_cache
def test_a_fresh_run_reproduces_the_committed_block():
    """The real-data drift check, in the test_layer_pathologies pattern. It
    re-runs a full preprocess on 4,357 polygons, so it takes its work dir
    from DELHI_PSI_MEASURE_CACHE and SKIPS when that is unset."""
    block = committed_block()
    proc = subprocess.run(
        [sys.executable, "scripts/measure_rule_effects.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR), "--work-dir", MEASURE_CACHE],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="partial_barriers") == block


def test_main_requires_a_verify_dir(capsys):
    with pytest.raises(SystemExit) as exc:
        measure_rule_effects.main(["--config", "code-2025"])
    assert exc.value.code == 2
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_measure_rule_effects.py`
Expected: **collection error —
`ModuleNotFoundError: No module named 'scripts.measure_rule_effects'`.**
That is the RED reason. Record it.

- [ ] **Step 3: Write the script**

Create `scripts/measure_rule_effects.py`. It reuses `_measure_common`
(`resolve_work_dir`, `render`) and the `derived_profile` / `measure_effect`
shapes from `measure_roads_access`, generalised to "change these methodology
values":

```python
"""What the partial-barrier rule does to today's numbers (DEL-48, spec § 7).

One block for now:

  partial_barriers  `code-2025` with ONE thing changed —
                    `methodology.barrier`: {rule: partial_weighted,
                    combine: any, buffer_m: 5} — everything else including
                    `adjacency: bbox` left alone, so the diff is
                    attributable to the barrier rule alone. Reports the
                    directed links by weight class, how many links each rule
                    severs, how many settlements' lists changed, the
                    fractional count and median weight, the preprocess
                    wall-clock, and the per-type PSI shift against the proven
                    `code-2025` outputs read from --verify-dir.

Unlike the roads measurement this block CANNOT reuse --verify-dir's
neighbours artifact: `methodology.barrier` is in the methodology stamp, so
the artifact has to be rebuilt. It is rebuilt into --work-dir, never beside
the proven one.

READ-ONLY over --data-dir and --verify-dir. Everything this script writes
goes under --work-dir, which is never inside the data directory.

    uv run python scripts/measure_rule_effects.py --config code-2025 \
        --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from delhi_psi import io, pipeline
from delhi_psi.config import PROFILES_DIR, load_config
from delhi_psi.pipeline import ID_COL, NBRS_COL, NBRS_WEIGHT_COL, TYPE_COL
from scripts._measure_common import render, resolve_work_dir

REPORTED_TYPES = ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV", "SDA")
DENOMINATORS = ("pop", "popdensity")
PARTIAL_PROFILE = "partial-barriers-5m"
BUFFER_M = 5
PSI_COL = "unnorm_psi"
WEIGHT_CLASSES = ("links_w_one", "links_fractional", "links_severed")


def base_profile_path(base):
    """A shipped profile NAME or a path to a YAML file — `load_config`'s own
    rule, so `--config code-2025` and a derived path both work."""
    candidate = Path(base)
    if candidate.suffix in (".yaml", ".yml"):
        return candidate
    return PROFILES_DIR / f"{base}.yaml"


def derived_profile(base, work_dir, *, profile_name, methodology):
    """`base` with the named methodology BLOCKS replaced wholesale.

    Blocks are replaced, never deep-merged: `methodology.<block>` is a
    complete statement, exactly as tests/variants.py and
    oraculum_fixtures.oracle_profile_path treat it. `paths.neighbors_artifact`
    and `paths.out_dir` are dropped so the per-profile default name applies
    and --out-dir decides where the run writes. Written to DISK, so the run
    is reproducible by hand.
    """
    raw = yaml.safe_load(base_profile_path(base).read_text())
    raw["profile"] = profile_name
    for block, values in methodology.items():
        raw["methodology"][block] = dict(values)
    paths = dict(raw.get("paths", {}))
    paths.pop("neighbors_artifact", None)
    paths.pop("out_dir", None)
    raw["paths"] = paths
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / f"{profile_name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def weight_classes(frame, *, weight_col=NBRS_WEIGHT_COL):
    """Directed links by weight class, plus the median fractional weight.

    `links_severed` is 0 BY CONSTRUCTION on a stored artifact: apply_barrier
    prunes a w == 0 link out of both columns, so a severed link leaves no
    trace here. It is reported anyway, as the assertion that nothing
    survived at weight 0 — how many links each rule severs is
    `compare_link_sets`' job, from the two rules' kept counts. Medians are
    pre-formatted, like the roads block's means, so the drift comparison is
    exact.
    """
    weights = [w for row in frame[weight_col] for _, w in row]
    fractional = [w for w in weights if 0.0 < w < 1.0]
    return {
        "links_w_one": sum(1 for w in weights if w == 1.0),
        "links_fractional": len(fractional),
        "links_severed": sum(1 for w in weights if w == 0.0),
        "median_fractional_w": ("nan" if not fractional
                                else f"{statistics.median(fractional):.6g}"),
    }


def link_sets(frame, *, id_col=ID_COL, neighbor_col=NBRS_COL):
    """{i: frozenset(js)} — the directed neighbour lists, for diffing."""
    return {row[id_col]: frozenset(row[neighbor_col])
            for _, row in frame.iterrows()}


def compare_link_sets(before, after):
    """How the two rules' stored lists differ.

    KEPT links, not severed ones: a severed link is absent from both
    artifacts, so the only honest counts are what each rule left behind. The
    stated bound — the partial rule severs FEWER links, because the global
    rule drops every link INTO a flagged settlement while the partial rule
    drops only fully covered boundaries — reads as
    `links_kept_partial > links_kept_code_2025`.
    """
    changed = [i for i in before if before[i] != after.get(i, frozenset())]
    return {
        "links_kept_code_2025": sum(len(js) for js in before.values()),
        "links_kept_partial": sum(len(js) for js in after.values()),
        "settlements_list_changed": len(changed),
    }


def _mean(values):
    return "nan" if values.empty else f"{values.mean():.6g}"


def measure_effect(before, after, *, denom, id_col=ID_COL, type_col=TYPE_COL,
                   types=REPORTED_TYPES):
    """The one-factor per-type PSI shift, the DEL-49 `one_factor` shape.

    `before` is the proven `code-2025` output read from --verify-dir; `after`
    is this script's run with only the barrier block changed. Both must
    report exactly the same settlements, or the comparison is not
    one-factor.
    """
    left = before.set_index(id_col)
    right = after.set_index(id_col)
    if set(left.index) != set(right.index):
        raise ValueError(
            f"{denom}: the two runs report different settlements "
            f"(before-only {len(set(left.index) - set(right.index))}, "
            f"after-only {len(set(right.index) - set(left.index))})")
    right = right.reindex(left.index)

    selectors = {name: (left[type_col] == name) for name in types}
    selectors["total"] = pd.Series(True, index=left.index)

    report = {}
    for name, rows in selectors.items():
        report[f"n_{denom}_{name}"] = int(rows.sum())
    for name, rows in selectors.items():
        report[f"psi_code_{denom}_{name}"] = _mean(left.loc[rows, PSI_COL])
    for name, rows in selectors.items():
        report[f"psi_partial_{denom}_{name}"] = _mean(right.loc[rows, PSI_COL])
    if "norm_psi" in left.columns and "norm_psi" in right.columns:
        for name, rows in selectors.items():
            report[f"norm_code_{denom}_{name}"] = _mean(
                left.loc[rows, "norm_psi"])
        for name, rows in selectors.items():
            report[f"norm_partial_{denom}_{name}"] = _mean(
                right.loc[rows, "norm_psi"])
    return report


def measure_partial_barriers(cfg, work_dir, *, base, verify_dir):
    """Block `partial_barriers`: rebuild the neighbours under the partial
    rule, then diff against the proven code-2025 outputs."""
    run_dir = Path(work_dir) / PARTIAL_PROFILE
    profile_path = derived_profile(
        base, run_dir, profile_name=PARTIAL_PROFILE,
        methodology={"barrier": {"rule": "partial_weighted",
                                 "combine": "any", "buffer_m": BUFFER_M}})
    run_cfg = load_config(profile_path, data_dir=str(cfg.paths.data_dir),
                          out_dir=str(run_dir))

    started = time.monotonic()
    pipeline.preprocess(run_cfg)
    elapsed = time.monotonic() - started
    pipeline.compute(run_cfg)

    after_frame = io.read_neighbors(
        run_dir / run_cfg.paths.neighbors_artifact)
    before_frame = io.read_neighbors(
        Path(verify_dir) / cfg.paths.neighbors_artifact)

    report = weight_classes(after_frame)
    report.update(compare_link_sets(link_sets(before_frame),
                                    link_sets(after_frame)))
    report["preprocess_seconds"] = f"{elapsed:.6g}"
    for denom in DENOMINATORS:
        before = pd.read_csv(
            Path(verify_dir) / f"{pipeline.output_basename(cfg, denom)}.csv")
        after = pd.read_csv(
            run_dir / f"{pipeline.output_basename(run_cfg, denom)}.csv")
        report.update(measure_effect(before, after, denom=denom,
                                     id_col=cfg.layers.settlements.id_col,
                                     type_col=cfg.layers.settlements.type_col))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code-2025",
                        help="profile that names the layers (default code-2025)")
    parser.add_argument("--data-dir", default=None,
                        help="data root, opened READ-ONLY")
    parser.add_argument("--work-dir", default=None,
                        help="scratch (the derived profile, its artifact and "
                             "outputs); default a fresh temporary directory. "
                             "Never under --data-dir.")
    parser.add_argument("--verify-dir", required=True,
                        help="an existing, complete code-2025 run "
                             "(colonies_neighbors.joblib + both output CSVs), "
                             "opened READ-ONLY")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, data_dir=args.data_dir)
    work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                prefix="delhi_psi_rules_")
    verify_dir = Path(args.verify_dir).expanduser()

    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
    print(f"verify-dir: {verify_dir}")
    print(f"work-dir: {work_dir}")
    print(render(measure_partial_barriers(cfg, work_dir, base=args.config,
                                          verify_dir=verify_dir),
                 name="partial_barriers"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the fixture-level tests**

Run: `uv run pytest -q -W error tests/test_measure_rule_effects.py`
Expected: the four Oraculum/unit tests PASS; the three document tests SKIP
(`rule_effects.md` has no block yet); `test_a_fresh_run_reproduces_the_committed_block`
SKIPS unless `DELHI_PSI_MEASURE_CACHE` is set.

If `test_the_oraculum_weight_classes_are_the_hand_counted_ones` reports
18/2/0 — 18 links at w == 1, 2 fractional, 0 severed — the classification
agrees with the hand count from the worksheet section. Anything else means
either the classification or the weights are wrong; check
`test_only_the_ad_edge_is_partially_blocked_on_oraculum` first.

- [ ] **Step 5: Write the document, block-shaped but unmeasured**

Create `docs/data/rule_effects.md` in the `docs/data/roads_access.md`
pattern: title, why the measurement exists (Raj's 28 Aug decision log § 4
asks for the one-factor quantification "as for roads"), the sentence that
numbers in `backticks` are block values verbatim while percentages are
derived, the four provenance labels, and one section per block explaining
what each key means. **Leave the fenced block out entirely** — the three
document tests skip on its absence, and Task 11 pastes it. Do not invent
placeholder numbers inside a fence: `parse_block` would read them as real.

Say explicitly, in the block's section:

```markdown
Unlike the roads measurement, this block re-runs `preprocess`: the barrier
block is part of the methodology stamp, so the proven `code-2025` artifact
cannot be reused, and reusing it would be a silent lie about which
neighbourhood the numbers describe. The rebuild goes into `--work-dir`,
never beside the proven run.

**What the run must show, stated before it runs** (spec § 6.6): no NaN and no
negative anywhere; `links_kept_partial` > `links_kept_code_2025`, i.e. the
number of severed links FALLS (the global rule severs every link INTO a
flagged settlement, the partial rule severs only fully covered boundaries);
`links_severed` == 0 in the weight classes, because a w == 0 link is pruned
out of the artifact entirely; and the fractional class is non-empty. A result
outside these bounds is a stop, not a number to write down.
```

- [ ] **Step 6: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add scripts/measure_rule_effects.py docs/data/rule_effects.md \
        tests/test_measure_rule_effects.py
git commit -m "$(cat <<'MSG'
feat(barrier): measure_rule_effects.py — the partial-barrier one-factor block (DEL-48)

code-2025 with only methodology.barrier changed, reported as directed links
by weight class, the severed-link counts under each rule, the settlements
whose lists moved, the median fractional weight, the preprocess wall-clock,
and the per-type PSI shift against the proven outputs read from --verify-dir.

Unlike the roads run this block re-runs preprocess: the barrier block is in
the methodology stamp, so the proven artifact cannot be reused — pinned by a
test, so a future "optimisation" that stages it fails loudly.

Proven on Oraculum, where the classification is hand-counted: 18 links at
w == 1, 2 fractional (A-D both ways, w = 0.08), 0 severed. The document
carries its prose and its stated bounds; the run step pastes the block.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 11: **CONTROLLER ONLY** — the real-data run (spec § 6.6)

**This is the one task in this plan where placeholders are allowed.** Every
`[from the run]` below marks a number that does not exist until the run
happens. Nowhere else in this plan may contain one. Do not delegate this
task to a subagent: it writes outside the repo, it is data-gated, and its
stop conditions need the owner's judgement.

**Files:**
- Modify: `docs/data/rule_effects.md` (paste the measured block; fill the
  provenance labels and the prose)
- Modify: `docs/methodology-config.md` (a new § 7)
- Modify: `docs/decisions/2026-08-28-raj-methodology-decisions.md` (§ 4, one
  sentence)
- Modify: `CHANGELOG.md` (extend the DEL-48 entry with the measured effect)

**Read-only over `~/delhi_data` except `--work-dir` / `--out-dir`, which must
be scratch directories — never the baseline and never `phase3_verify`
itself.** Any write under `~/delhi_data` outside those is a stop-and-ask
(spec § 11); the directory is bisynced to the shared drive.

- [ ] **Step 1: The standing proof — `code-2025` verify at 0.000e+00**

Nothing on the default path changed: no weight column, `combine: any`
selecting every layer as before, and `buffer_m` None in the stamp where
pre-3E artifacts have no key at all. Which is exactly why this is worth
running. Run each command separately, backgrounded to a log (they exceed a
foreground timeout), and read each log before continuing:

```bash
mkdir -p ~/measure_work/logs ~/measure_work/del48-verify
uv run delhi-psi preprocess --config code-2025 --data-dir ~/delhi_data \
    --out-dir ~/measure_work/del48-verify > ~/measure_work/logs/del48-pre.log 2>&1
uv run delhi-psi compute    --config code-2025 --data-dir ~/delhi_data \
    --out-dir ~/measure_work/del48-verify > ~/measure_work/logs/del48-comp.log 2>&1
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/measure_work/del48-verify \
    > ~/measure_work/logs/del48-verify.log 2>&1
```

Expected in `del48-verify.log`: **PASS on every comparison at `0.000e+00`**
(30 numeric columns × 2 output sets = 60 comparisons). Keep the WHOLE log,
never a `tail` — a truncated log is how a previous cycle mis-reported this
count for three rounds. A single non-zero deviation is a STOP: it means the
default path moved, which this cycle promised it would not.

Record: PASS line, comparison count.

- [ ] **Step 2: The `partial_barriers` block**

```bash
export DELHI_PSI_MEASURE_CACHE=~/measure_work/cache
uv run python scripts/measure_rule_effects.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3_verify \
    --work-dir ~/measure_work/cache > ~/measure_work/logs/del48-rules.log 2>&1
```

This re-runs `preprocess` on 4,357 polygons against 6,015 barrier features
with the STRtree per-link intersection, so budget minutes, not seconds; the
wall-clock is one of the reported keys. Read the whole log.

- [ ] **Step 3: Check the run against its stated bounds BEFORE writing anything down**

From the printed block, confirm all of:

- `links_fractional` > 0 — the fractional class is non-empty. If it is zero,
  the rule is doing nothing on the real layer and that is a finding, not a
  number.
- `links_kept_partial` > `links_kept_code_2025` — the partial rule severs
  FEWER links than `global_asymmetric`, which drops every link INTO a
  flagged settlement while the partial rule drops only fully covered
  boundaries. State the two counts and their difference explicitly in the
  doc.
- `links_severed` == 0 in the weight classes — a w == 0 link is pruned out
  of the artifact, so a non-zero count here would mean the pruning rule and
  the column disagree.
- No NaN in any `psi_*` or `norm_*` value (`_mean` prints `nan` for an empty
  selection — an empty reported type is a finding).
- `compute` ran clean: `validate.check_no_negative` passed and DEL-54's
  min-max guard did not fire. Both surface as a non-zero exit; the exit was
  0, so say so.

**A result outside these bounds is a STOP** (spec § 6.6, § 11): report it to
the owner, do not write it into the doc as if it were expected.

- [ ] **Step 4: Paste the block and finish the document**

In `docs/data/rule_effects.md`, paste the script's fenced block **verbatim**
— the drift test parses the committed document and the script's stdout with
the same parser, so a hand-edited digit fails the build. Then fill the
provenance labels:

```markdown
- **Run date:** [from the run]
- **Inputs:** settlement layer `uso_update_sep2021`, the three barrier layers named by `code-2025`, and the proven `code-2025` run in `~/delhi_data/phase3_verify` (its neighbours artifact and its two output CSVs, read only)
- **Commit:** [from the run — the commit this branch is at]
- **Command:** `uv run python scripts/measure_rule_effects.py --config code-2025 --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache`
```

Write the prose around the block: how many directed links are fractional
(`[from the run]`), the median fractional weight (`[from the run]`), how many
settlements' neighbour lists changed (`[from the run]`), the preprocess
wall-clock (`[from the run]`), and the per-type PSI shift in one sentence.
Every number in `backticks` must be a block value verbatim — percentages and
other derived quantities are written with a `%` sign or without backticks,
because `assert_prose_numbers_come_from_the_blocks` enforces exactly that.

- [ ] **Step 5: Config doc § 7 and the decision log**

Add a new `## 7. Partial barriers: what the real layer showed` to
`docs/methodology-config.md`, after § 6, carrying: the run's headline numbers
(`[from the run]`), and the buffer semantics in one paragraph — a distance,
round caps, strictly > 0, and the consequence that a canal ending 3 m short
of a corner blocks the corner. Note that a `within_distance` 10 km profile
combined with `partial_weighted` (4.4 M links) is Phase 6's problem and is
not solved here.

In `docs/decisions/2026-08-28-raj-methodology-decisions.md` § 4, add ONE
sentence for the batched reply to Raj: `partial_weighted` is implemented,
proven and measured; on the real layer it leaves `[from the run]` links
fractional and severs `[from the run]` fewer than today's rule; the ratified
profile (DEL-31) decides whether to adopt it. **The reply to Raj is drafted,
never sent** (spec § 11).

- [ ] **Step 6: Run the drift test with the cache exported**

```bash
DELHI_PSI_MEASURE_CACHE=~/measure_work/cache \
    uv run pytest -q -W error tests/test_measure_rule_effects.py
```

Expected: PASS, with `test_a_fresh_run_reproduces_the_committed_block` now
RUNNING rather than skipping, and the three document tests running rather
than skipping. If the fresh run does not reproduce the committed block byte
for byte, the block was hand-edited or the run is not deterministic — find
out which before committing.

- [ ] **Step 7: Full suite in the FOREGROUND, then commit**

Run `uv run pytest -q -W error` as ONE Bash call, explicit `timeout` 600000,
NOT backgrounded. Read the summary line.

```bash
git add docs/data/rule_effects.md docs/methodology-config.md \
        docs/decisions/2026-08-28-raj-methodology-decisions.md CHANGELOG.md
git commit -m "$(cat <<'MSG'
docs(barrier): the real-layer effect of partial_weighted (DEL-48)

code-2025 verifies against the July 2025 baseline at 0.000e+00 on every
comparison after this branch: nothing on the default path moved. The
partial_barriers block measures the one-factor effect — directed links by
weight class, the severed-link counts under each rule, the settlements whose
lists changed, the median fractional weight, the preprocess wall-clock, and
the per-type PSI shift — inside the bounds stated before the run.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

- [ ] **Step 8: Jira**

Move DEL-48 to Done with an evidence comment: the anchor (w_AD = 0.08 on the
500 m A–D edge at a 5 m buffer), production == reference at 1e-12 on
`partial_5m` for both cities and both denominators, the real-data verify at
0.000e+00, and the `partial_barriers` block's headline numbers. Note that the
ratified profile (DEL-31) still has to choose the value, and that DEL-20
(Group C) is the remaining 3E blocker.

---

## Self-review

**1. Spec coverage.**

| spec | task |
|---|---|
| § 1 config surface: `buffer_m` via `_conditional_number`, `_reject_unknown`, `REFERENCE_KNOBS`, reserved entry deleted, `BarrierConfig` | Task 2 |
| § 1 shipped-profile comment | Task 2 Step 6 |
| § 2.1 definition (all five steps, incl. GeometryCollection decomposition and the `>=` guard) | Tasks 1 Step 3, 3 Step 3 |
| § 2.2 `apply_barrier` signature, pruned-list contract, weight column, public helpers, STRtree | Task 3 |
| § 2.3 `NBRS_WEIGHT_COL`, `build_neighbors`, the INFO line, the stamp, `apply_exclusion`, `index_frames` drop, `io.SHAPEFILE_DROP_COLUMNS` | Task 5 |
| § 2.4 `pcen`/`service_index` `nbr_weight_col`, the `w * lent * decay` line, the KeyError | Task 4 |
| § 2.5 reference `partial_weights`, the rule, `compute_city` kwarg, `RULESETS` unchanged | Task 1 |
| § 2.6 `combine` selects layers for the geometry rules | Task 3 (`selected_barrier_geoms`), Task 5 (`build_neighbors`) |
| § 5 variants table row, `VARIANT_KNOBS`, `IGNORED_VARIANT_KEYS`, `variant_methodology`, `enum_key`, `EXTRA_PARAMS`, the invariants guard | Tasks 1 Step 8, 2 Step 5, 6 |
| § 6.1 Oraculum anchors, the buffer-1 pin, the worksheet section | Tasks 1 Step 6, 9 Step 1 |
| § 6.4 items 1–10 (production), the reference mirrors, the synthetic production==reference city | Tasks 1 Step 1, 3 Step 1, 8 |
| § 6.5 variants CSV, CLI leg, stamp tests, byte-identity, the deleted `test_reserved_partial_weighted` | Tasks 2, 5, 6, 7 |
| § 6.6 real data | Task 11 |
| § 7 docs, `measure_rule_effects.py`, CHANGELOG, WORKPLAN, Jira | Tasks 9, 10, 11 |
| § 8 Group B file list | all tasks; `tests/test_measure_roads_access.py` added with its reason in Task 5 |

Group C items (`overlap.lending`, `OverlapConfig`, `index.shared_amounts`,
`overlap_outside`, `partial_5m_outside`, `docs/oracle/messy-city.md`) are
deliberately absent — see Global Constraints.

**2. Placeholder scan.** Every code step carries real code. The only
bracketed placeholders are the `[from the run]` markers in Task 11, which the
task itself declares as the one permitted place. No "TBD", no "similar to
Task N" (the two-squares helper, the stamp literal and the derived-profile
shape are each written out in full where they are used), no "add appropriate
error handling" — the two error paths (`buffer_m` rules, the weight KeyError)
have their messages spelled out.

**3. Name and type consistency, and the Group C contract.** The § 8 "Names
Group C relies on" block is reproduced exactly and each name is produced by
a named task: `index.pcen(..., nbr_weight_col=None)` and the loop line
`poly_count += w * lent * _decay(...)` with `lent = nbr_count` (Task 4);
`pipeline.NBRS_WEIGHT_COL` (Task 5); `reference_impl.compute_city(...,
barrier_buffer_m=None)` with the factor `barrier_w[(i, j)]` (Task 1);
`VARIANT_KNOBS` accepting a non-adjacency/decay block (Task 1 Step 8);
`variant_methodology`'s per-block branch pattern (Task 6 Step 2);
`EXTRA_PARAMS` (Task 2 Step 5). The column name is
`"nbrs_barrier_weight"` in `neighbors.apply_barrier`'s default, in
`pipeline.NBRS_WEIGHT_COL`, in `io.SHAPEFILE_DROP_COLUMNS` and in every test
— one spelling. `shared_boundary` returns a geometry (callers test
`.length`), while the reference's private `_shared_boundary` returns `None`
for an empty intersection; the two are independent by design and each
caller matches its own. Test frames are built with `gpd.GeoDataFrame` and
`box`/`LineString`/`MultiPolygon`/`Point`, matching each file's existing
import style (`tests/test_index.py` imports `gpd`, `Point` and `box` at
module level; `tests/test_neighbors.py` and `tests/test_reference_impl.py`
import geometry locally inside tests — both styles are followed where they
belong).

**4. Numbers.** Every geometric constant asserted in this plan (`w_AD = 0.08`
at buffer 5 and `0.096` at buffer 1; `0.49` / `0.495` / `0.99` / `0.5` on the
two-square cases; `1 − 20/2400` and `1 − 1010/2400` on the overlapping pair;
the 1400 m mixed-intersection boundary; `GeometryCollection.boundary is
None`; `LineString.buffer(0).is_empty`) was computed against shapely 2.1.2 in
this worktree before the plan was written, not copied from the spec. The
seven § 6.1 PCEN anchors were re-derived from the fixture geometry and agree
with the spec's table.

Execution: subagent-driven-development.
