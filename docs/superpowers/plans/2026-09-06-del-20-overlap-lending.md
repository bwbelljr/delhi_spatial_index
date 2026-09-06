# DEL-20 — the overlap neighbour rule (`overlap.lending: outside_receiver`) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** A neighbour lends only the services that are **not already inside
the receiving settlement** — `|S_j \ S_i|` instead of `S_j` — so a service
sitting in the overlap of two colony polygons is never counted twice for the
same settlement; shipped as the new value `outside_receiver` of a new
required switch `methodology.overlap.lending`, whose other value `whole`
names today's arithmetic and is what both shipped profiles carry.

**Architecture:** Raj ratified the *counting* half on 28 Aug 2026 (a service
inside k overlapping colonies counts for each of the k) — that is today's
behaviour on both sides and does not move. Bob noticed the other half: under
any adjacency rule those k colonies are also neighbours, so the same service
is lent back through the neighbour term. The fix is one redefinition of
`lent` in `index.pcen`, fed by a **sparse** per-service structure
`{(i, j): amount}` that only holds pairs which actually share something —
built compute-locally in `pipeline.index_frames`, never stored, never in the
methodology stamp. Like `partial_weighted` before it, the new value ships as
a *variant*, so every committed expected value stays byte-identical by
construction; the switch's `whole` value is added to both shipped profiles
as a required key naming today's rule.

**Tech Stack:** Python 3.13, shapely 2.1.2 (`intersection`, `STRtree`),
geopandas (`sjoin`) / pandas, pytest under `-W error`, uv.

**Spec:** `docs/superpowers/specs/2026-09-05-cycle-3e-partial-barriers-design.md`
— § 1 (config surface), § 3 in full (3.1 definition, 3.2 the data structure,
3.3 `pcen`, 3.4 reference, 3.5 what is deliberately NOT changed), § 5
(variants table), § 6.2 (the messy overlap pin), § 6.4 (unit tests), § 6.5
(byte-identity, the stamp), § 6.6 (real data), § 8 Group C (scope), § 12
(decisions already made — do not reopen them). **Read § 3 and § 6.2 before
starting.**

## Global Constraints

- **This is Group C only.** Group A (the min-max guard, DEL-54) and Group B
  (partial barriers, DEL-48) are already on `main` at `0dcca15`. Do not
  touch `barrier.rule`, `barrier.buffer_m`, `neighbors.apply_barrier`,
  `neighbors.partial_weight`, `neighbors.shared_boundary`,
  `reference_impl.partial_weights`, or `index.minmax`'s guard. Group B's
  names are CONSUMED here, never re-derived.
- **Branch `del-20-overlap-lending` off `main` at `0dcca15`.** One branch,
  one PR, one ticket. Work only inside the worktree
  `/home/bwbelljr2/delhi_spatial_index/.claude/worktrees/del-20-overlap-lending`.
- **No existing expected value may move.** Both cities'
  `tests/fixtures/*/expected_values.csv` and every
  `tests/fixtures/*/production/*.csv` must be byte-identical to `0dcca15` at
  every commit. The two `variants_expected_values.csv` files may change ONLY
  by the ADDITION of `overlap_outside` and `partial_5m_outside` rows (Task 5
  gives the exact check). Any other difference is the spec's stop-and-ask
  condition (§ 11): **STOP and report**, never regenerate around it.
- **Adding a REQUIRED key to both shipped profiles is the delicate part**
  (Task 2). `overlap: {lending: whole}` names today's behaviour, so nothing
  either profile computes may move: both cities' `expected_values.csv`,
  every `production/*.csv` and the PRE-EXISTING blocks of both
  `variants_expected_values.csv` stay byte-identical, and the real-data
  `code-2025` verify stays at `0.000e+00`. `tests/test_config.py`'s `MINIMAL`
  string needs the key too — it is the third of the three places the 3D
  `decay.distance` key had to appear.
- **Raj has NOT confirmed the lending half.** `outside_receiver` is
  implemented, proven and measured; `whole` is what ships. Do not change
  either profile's *value*, and do not write the ratified profile (that is
  DEL-31).
- **The counting half does not move** (spec § 3.5).
  `tests/test_messy_fixtures.py::test_the_overlap_clinic_is_counted_for_both_owners`
  must pass **unchanged**: one clinic strictly inside `O1 ∩ O2` still counts
  as 1 for O1 and 1 for O2 under every value of the new switch.
- **The structure is SPARSE, and that is the whole cost argument.**
  `shared_ij` is non-zero only for pairs that actually share a service point
  or road metres — a subset of the 4,069 positive-area overlapping pairs on
  the real layer (429 multi-settlement points across six services;
  `docs/data/layer_pathologies.md`), **1** pair in the messy city (the
  `O1`/`O2` clinic) and **0** in Oraculum. For every other pair
  `|S_j \ S_i| == |S_j|` exactly, nothing is computed and nothing is stored:
  `pcen` reads the structure with `.get((i, j), 0)` and the 0 IS the
  representation of "nothing shared", never a swallowed miss. The point
  structure is built from the service points' containment (one `sjoin`,
  grouped by POINT), so it never touches the ~30k directed links at all.
- **`overlap.lending` is NOT in the methodology stamp** (spec § 12 item 7):
  it is applied downstream in `compute`, like `decay` and `roads`, so ONE
  neighbours artifact serves both values. Pinned by a test (Task 6).
- **Under `whole` the arithmetic is bit-identical to today.**
  `shared_amounts=None` short-circuits the subtraction entirely — there is no
  `x - 0` on the default path. Prove it, never assert it.
- **Implementers must NOT run the full suite.** `uv run pytest -q -W error`
  exceeds a foreground timeout, is auto-backgrounded, and a sub-agent cannot
  receive the completion notification — the task stalls. Each task's final
  step runs ONLY the files that task touched, with the exact command given.
  **The controller runs the full suite** between tasks and before the PR.
- The suite's mode is `-W error`; every command in this plan already carries
  it. Use `uv run`, never a bare `python`.
- Commit trailers, both lines, on every commit:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6`
- Commit message prefixes: `feat(overlap)` for production/config changes,
  `test(overlap)` for reference/test-only changes, `docs(overlap)` for docs.
- **Placeholders are forbidden everywhere except Task 10**, the controller's
  real-data run, where bracketed `[from the run]` marks numbers that do not
  exist until the run happens.

## Task order and the ordering constraint

`tests/test_profiles_match_reference.py::test_every_mapped_knob_is_one_the_reference_actually_implements`
drives the reference once per mapped knob value:

```python
    for key, knob in knob_for_key.items():
        for config_value, reference_value in REFERENCE_KNOBS[key].items():
            kwargs = dict(base)
            kwargs[knob] = reference_value
            kwargs.update(EXTRA_PARAMS.get((key, config_value), {}))
            frame = compute_city(city, services, barriers, **kwargs)
```

The moment `knob_for_key` gains `"methodology.overlap.lending":
"overlap_lending"`, that loop calls `compute_city(overlap_lending="whole")`
and `compute_city(overlap_lending="outside_receiver")`. If the reference has
no such keyword the test fails with `TypeError: compute_city() got an
unexpected keyword argument 'overlap_lending'`. **So the reference rule must
land BEFORE — or at the latest in the same commit as — the `knob_for_key`
entry**, and since that entry belongs with the `REFERENCE_KNOBS` entry (they
are the same fact stated on two sides), the reference task ships first.

Two differences from DEL-48's version of this constraint, both worth stating
because they change what an implementer may safely do:

1. **`REFERENCE_KNOBS` alone does not drive `compute_city` here.** That test
   iterates its own `knob_for_key` map, which lives in the test file, not
   `REFERENCE_KNOBS`; nothing in the repo asserts the two are co-extensive
   (checked: `REFERENCE_KNOBS` is read by `test_config.py` lines 133, 173,
   477, 530 and by `test_profiles_match_reference.py`'s
   `test_enums_cover_exactly_the_reference_table` — none of them calls the
   reference). The coupling is via the `knob_for_key` line only. The order
   below respects it anyway, because splitting one fact across two commits
   to exploit a loophole is how the next reader gets misled.
2. **No `EXTRA_PARAMS` entry is needed.** Neither `whole` nor
   `outside_receiver` requires a conditional parameter, unlike
   `partial_weighted`'s `barrier_buffer_m`. Leave `EXTRA_PARAMS` alone.

A second coupling fixes Task 5, exactly as in DEL-48: adding rows to
`tests/variants.py` is the LAST link, because four separate tests read that
table and each fails on a row that is not yet everywhere —
`test_config.py::test_every_variant_block_is_one_the_loader_accepts`
(loader), `test_reference_impl.py::test_variants_csv_has_one_scenario_and_every_variant`
and `::test_variants_expected_values_csv_is_regenerable` (the committed
CSVs), and `test_variants_match_reference.py::test_production_matches_the_reference_on_each_variant`
(the whole production path). So the variant rows, `variant_methodology`'s
overlap branch, `enum_key` and the two regenerated CSVs are ONE commit, and
it comes after config, index and pipeline are all in place.

A third, new to this ticket: **`MethodologyConfig` gains a required field**,
so every direct construction of it must be fixed in the same commit as the
field. There is exactly one outside `config.py` —
`tests/test_reference_impl.py:597`, the synthetic production-vs-reference
city — and it uses keyword arguments, so the fix is one added line. Task 2
carries it.

Resulting order:

| # | task | prefix |
|---|---|---|
| 1 | Reference: `overlap_lending`, `shared_amounts`, the messy pins | `test(overlap)` |
| 2 | Config: the `overlap` block, both shipped profiles, `MINIMAL`, the reserved key | `feat(overlap)` |
| 3 | `index.shared_amounts` and `pcen`'s `lent` | `feat(overlap)` |
| 4 | Pipeline wiring: `index_frames` builds and passes the dicts | `feat(overlap)` |
| 5 | The `overlap_outside` / `partial_5m_outside` variants and the regenerated CSVs | `test(overlap)` |
| 6 | The messy production pin, the CLI leg, and the artifact that stays valid | `test(overlap)` |
| 7 | Production == reference on synthetic overlap geometry | `test(overlap)` |
| 8 | Docs: config doc, messy-city doc, CHANGELOG, WORKPLAN | `docs(overlap)` |
| 9 | `scripts/measure_rule_effects.py` — the `overlap_lending` block | `feat(overlap)` |
| 10 | **Controller:** the real-data run and its numbers | `docs(overlap)` |

---

### Task 1: The reference rule — `shared_amounts`, `compute_city(overlap_lending=…)`, the messy pins

**Files:**
- Modify: `tests/reference_impl.py` (`VARIANT_KNOBS` :43-52; a new function
  above `compute_city`; `compute_city` :261-381)
- Test: `tests/test_variant_rules.py` (append; the § 6.2 messy pins)

**Interfaces:**
- Consumes: nothing from this branch. This task touches only the reference
  side, which never imports `delhi_psi` (the INDEPENDENCE RULE at the top of
  `tests/reference_impl.py`). It does consume Group B's already-merged
  `compute_city(..., barrier_buffer_m=None)` and the neighbour-sum line
  `w * amounts[svc][j] * contribution_weight(i, j)` — that line is the seam
  this ticket redefines.
- Produces, and later tasks rely on these exact names:
  - `reference_impl.shared_amounts(nbrs, settlements, services) ->
    dict[str, dict[tuple[str, str], int | float]]` — one table per service
    name in `POINT_SERVICES + ("road",)`, sparse, symmetric (both orders
    inserted).
  - `reference_impl.OVERLAP_LENDINGS = ("whole", "outside_receiver")`.
  - `reference_impl.compute_city(..., overlap_lending="whole")`.
  - `VARIANT_KNOBS[("overlap", "lending")] = "overlap_lending"`.

- [ ] **Step 1: Write the failing reference pins**

Append to `tests/test_variant_rules.py`. Extend the existing
`from tests.reference_impl import (...)` block with `shared_amounts` (keep
the names alphabetical, as the block already is). This file imports nothing
from `delhi_psi` — keep it that way. `pytest`, `CITIES`, `MESSY`,
`ORACULUM`, `RULESETS`, `adjacency`, `apply_barrier`, `scored` and `variant`
are already imported or defined in it.

```python
# --- DEL-20: overlap lending on the messy city (spec § 3.1, § 6.2) -----
# O1 is _rect(10000, 0, 11000, 1000) and O2 is _rect(10800, 0, 11800, 1000),
# so they overlap in x in [10800, 11000]. The ONE clinic at (10900, 500) is
# strictly inside BOTH, so it is O1's own AND O2's own — Raj's ratified
# counting half, which does not move — and under `whole` it is ALSO lent
# from each to the other, which is the half this switch removes. Centroids
# (10500, 500) and (11300, 500) are 0.8 km apart, so the decay is 1/1.8.
OVERLAP_OUTSIDE = dict(RULESETS["code"], overlap_lending="outside_receiver")
W_O1O2 = 1 / 1.8


def test_the_overlap_clinic_is_lent_back_under_whole():
    """Today's arithmetic, stated so the switch has something to move: the
    single physical clinic reaches O1 twice — once as its own, once decayed
    from O2 — and reaches O2 twice as well."""
    got = scored(MESSY, RULESETS["code"])
    assert got.loc["O1", "clinic_pcen"] == pytest.approx(
        (1 + 1 * W_O1O2) / 600, abs=1e-12)
    assert got.loc["O2", "clinic_pcen"] == pytest.approx(
        (1 + 1 * W_O1O2) / 700, abs=1e-12)


def test_outside_receiver_lends_only_what_is_not_already_inside():
    """|S_j \\ S_i| is 0 for the clinic: O2's only clinic is already inside
    O1, so O1 gets it once. The OWN counts do not move — Raj's ratified half
    is untouched, and that is what makes this a lending rule and not a
    counting rule."""
    got = scored(MESSY, OVERLAP_OUTSIDE)
    assert got.loc["O1", "clinic_count"] == 1
    assert got.loc["O2", "clinic_count"] == 1
    assert got.loc["O1", "clinic_pcen"] == pytest.approx(1 / 600, abs=1e-12)
    assert got.loc["O2", "clinic_pcen"] == pytest.approx(1 / 700, abs=1e-12)


def test_a_neighbours_service_outside_the_overlap_is_lent_in_full():
    """The clinic moves and the school does not: O2's school at (11400, 500)
    lies outside O1, so |S_j \\ S_i| == |S_j| and O1's school row is exactly
    the `whole` value — asserted with `==`, because a pair with nothing
    shared must not go anywhere near the arithmetic. O1's own police point
    is not in O2 either."""
    whole = scored(MESSY, RULESETS["code"])
    got = scored(MESSY, OVERLAP_OUTSIDE)
    assert got.loc["O1", "school_pcen"] == whole.loc["O1", "school_pcen"]
    assert got.loc["O1", "school_pcen"] == pytest.approx(
        (0 + 1 * W_O1O2) / 600, abs=1e-12)
    assert got.loc["O1", "police_pcen"] == pytest.approx(1 / 600, abs=1e-12)


def test_overlap_outside_is_degenerate_on_oraculum():
    """No overlapping polygons and no point inside two settlements, so the
    shared structure is EMPTY and every row equals the `code` base — stated,
    like `boundary` on Oraculum and `partial_5m` on the messy city, so the
    CSV rows are never mistaken for a proof they are not."""
    base = scored(ORACULUM, RULESETS["code"])
    got = scored(ORACULUM, OVERLAP_OUTSIDE)
    for column in base.columns:
        assert list(got[column]) == pytest.approx(list(base[column]),
                                                  abs=1e-12), column


def test_the_shared_structure_is_sparse_and_symmetric():
    """One entry, both orders, on the one pair that shares anything; nothing
    at all on Oraculum, and nothing for the road (no road row crosses the
    O1/O2 overlap). Every other pair has |S_j \\ S_i| == |S_j| and is never
    computed or stored — that is the cost argument, made checkable."""
    for city, expected in ((ORACULUM, {}),
                           (MESSY, {("O1", "O2"): 1, ("O2", "O1"): 1})):
        settlements = city.load_settlements()
        nbrs = apply_barrier(adjacency(settlements, "bbox"), settlements,
                             city.load_barriers(), "global")
        got = shared_amounts(nbrs, settlements, city.load_services())
        assert got["clinic"] == expected, city.name
        assert all(not table for svc, table in got.items()
                   if svc != "clinic"), city.name


def test_no_service_column_is_constant_under_overlap_outside():
    """The invariants guard refuses a degenerate min-max group and DEL-54's
    guard raises on one. Both cities, both denominators, every service:
    checked HERE, before Task 5's fixture regeneration depends on it."""
    for city in CITIES:
        for denom in ("pop", "popdensity"):
            got = scored(city, OVERLAP_OUTSIDE, denom)
            for column in [c for c in got.columns if c.endswith("_pcen")]:
                assert got[column].max() > got[column].min(), (city.name,
                                                               denom, column)


def test_an_unknown_lending_value_raises():
    """An unimplemented value must RAISE — the mapped-knob test relies on
    it, and so does the loader's enum table being the only source of
    values."""
    with pytest.raises(ValueError, match="overlap lending"):
        scored(MESSY, dict(RULESETS["code"], overlap_lending="halves"))
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_variant_rules.py -k "overlap or lending or shared_structure"`

Expected: **FAIL.** `ImportError: cannot import name 'shared_amounts' from
'tests.reference_impl'` at collection — the whole file errors out on the
import, so every test in it reports as an error, not just the seven new
ones. That is the RED reason. Record the exact text; do not proceed until
you have seen it.

- [ ] **Step 3: Implement the shared-amount structure**

In `tests/reference_impl.py`, insert immediately above `DECAY_FORMS`
(:257) — i.e. after `_service_amounts`, whose predicates it must match:

```python
OVERLAP_LENDINGS = ("whole", "outside_receiver")


def shared_amounts(nbrs, settlements, services):
    """{svc: {(i, j): amount}} — how much of `svc` lies inside BOTH i and j.

    Points: the number of that service's points `within` both, the same
    strict predicate `_service_amounts` uses — so `shared_ij <= amount_j` by
    construction and the subtraction in `compute_city` can never go
    negative. Roads: the clipped length in km of every road row inside
    `geom_i n geom_j`, summed, which is `_service_amounts`' own road
    arithmetic restricted to the intersection.

    SPARSE and SYMMETRIC. Only pairs that actually share something get an
    entry, in both orders; `compute_city` reads it with `.get((i, j), 0)`
    and that 0 IS the representation of "nothing shared", not a swallowed
    miss. On a city with no overlapping polygons every table comes back
    empty (Oraculum), which is why `overlap_outside` is degenerate there.

    The point tables are built from the SERVICE POINTS' containment, never
    from the pair list: a point inside one settlement — every point on a
    clean layer — is looked at once and contributes nothing.
    """
    idx = settlements.set_index("USO_AREA_U").geometry
    out = {}
    for svc in POINT_SERVICES:
        gdf = services.get(svc)
        table = {}
        if gdf is not None:
            for point in gdf.geometry:
                inside = [i for i in idx.index if point.within(idx[i])]
                if len(inside) < 2:
                    continue
                for i in inside:
                    for j in inside:
                        if i != j:
                            table[(i, j)] = table.get((i, j), 0) + 1
        out[svc] = table
    road_geoms = list(services["road"].geometry)
    table = {}
    # Sorted UNDIRECTED pairs: each is measured once and written both ways,
    # so the table is symmetric by construction and the order a set would
    # have iterated in cannot reach the arithmetic.
    for i, j in sorted({tuple(sorted((i, j)))
                        for i, js in nbrs.items() for j in js}):
        overlap = idx[i].intersection(idx[j])
        if overlap.is_empty:
            continue
        length = sum(road.intersection(overlap).length
                     for road in road_geoms) / 1000
        if length > 0:
            table[(i, j)] = length
            table[(j, i)] = length
    out["road"] = table
    return out
```

- [ ] **Step 4: Wire it into `compute_city`**

In `tests/reference_impl.py::compute_city`, add `overlap_lending="whole"` to
the signature, immediately after `decay_distance="centroid"` so every
existing positional/keyword call is unaffected:

```python
def compute_city(settlements, services, barriers, *, adjacency_rule,
                 barrier_rule, roads_formula, scenario, denom, second_norm,
                 absent_neighbor_contribution, scenarios=None,
                 max_distance_km=None, barrier_buffer_m=None,
                 decay_form="inverse_linear", exponent=None, scale_km=None,
                 decay_distance="centroid", overlap_lending="whole"):
```

Add the value check next to the other "a value the reference does not
implement RAISES" checks, immediately after the `decay_distance` one
(:272-274):

```python
    if overlap_lending not in OVERLAP_LENDINGS:
        raise ValueError(f"unknown overlap lending {overlap_lending!r}; "
                         f"allowed values: {list(OVERLAP_LENDINGS)}")
```

Then, immediately after `amounts = _service_amounts(universe, services)`
(:307), add:

```python
    # Built on the post-barrier links, which are exactly the pairs the
    # neighbour sum below looks up. Nothing is built at all under `whole`.
    shared = (shared_amounts(nbrs, universe, services)
              if overlap_lending == "outside_receiver" else None)
```

and, in the neighbour sum, replace

```python
                w = 1.0 if barrier_w is None else barrier_w[(i, j)]
                decayed_sum += w * amounts[svc][j] * contribution_weight(i, j)
```

with

```python
                w = 1.0 if barrier_w is None else barrier_w[(i, j)]
                lent = amounts[svc][j]
                if shared is not None:
                    lent -= shared[svc].get((i, j), 0)
                decayed_sum += w * lent * contribution_weight(i, j)
```

Under `whole`, `shared is None`, so the expression is character-for-character
today's — no subtraction happens at all and the `ideal`/`code` rows cannot
move. Step 6 proves it rather than asserting it.

- [ ] **Step 5: Run the pins and see them pass**

Run: `uv run pytest -q -W error tests/test_variant_rules.py`

Expected: PASS, including every pre-existing pin in the file (the band
variants, `pow2`/`exp1`/`none`/`boundary`, and Group B's `partial_5m`
anchors). If `test_outside_receiver_lends_only_what_is_not_already_inside`
fails, the arithmetic in the test is right and the implementation is wrong —
re-derive from the test's own docstring before touching an expected value.

- [ ] **Step 6: Prove nothing regenerated**

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
uv run python scripts/generate_production_fixtures.py
git status --porcelain tests/fixtures/
```

Expected: **empty output.** `RULESETS` still binds `overlap_lending` to
nothing, so both rule-sets run the untouched `whole` branch and every CSV
must come back byte-identical. A modified fixture here is a **STOP**, not
something to commit (spec § 11).

- [ ] **Step 7: Add the variant-table entry (inert until Task 5)**

In `tests/reference_impl.py`, extend `VARIANT_KNOBS`. It is consulted only
for blocks a variant actually names, so this line changes nothing until
`tests/variants.py` gains its rows:

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
    ("overlap", "lending"): "overlap_lending",
}
```

`IGNORED_VARIANT_KEYS` is NOT touched: `overlap` has exactly one key and the
reference implements it.

- [ ] **Step 8: Run this task's files, then commit**

Run: `uv run pytest -q -W error tests/test_variant_rules.py tests/test_reference_impl.py`

Expected: PASS. `tests/test_reference_impl.py` is included because it holds
both `*_is_regenerable` tests, which re-derive the committed CSVs from
`compute_city` — the direct proof that Step 4's edit moved nothing. Do NOT
run the whole suite (Global Constraints); the controller does that.

```bash
git add tests/reference_impl.py tests/test_variant_rules.py
git commit -m "$(cat <<'MSG'
test(overlap): the reference lends only what is not already inside (DEL-20)

compute_city gains overlap_lending; under outside_receiver a neighbour lends
|S_j \ S_i| instead of S_j. The shared amount is the count of that service's
points `within` both settlements (points) or the clipped road length inside
the intersection (roads) — each side's own predicate, so shared_ij <=
amount_j by construction and the subtraction cannot go negative.

The structure is sparse and symmetric: only pairs that actually share
something get an entry, so it is EMPTY on Oraculum and holds exactly the one
O1/O2 clinic pair on the messy city. Pinned there: O1's clinic PCEN falls
from (1 + 1/1.8)/600 to 1/600 while its school PCEN does not move at all,
because O2's school is outside the overlap. The own counts are unchanged —
Raj's ratified counting half is not what this switch touches.

Under `whole` no subtraction happens at all, so both rule-sets are
byte-identical: all three generators re-emit every fixture unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 2: Config — the `overlap` block, both shipped profiles, `MINIMAL`, the reserved key

**Files:**
- Modify: `delhi_psi/config.py` (`REFERENCE_KNOBS` :34-53, `ENUM_KEYS`
  :57-66, the enum constructors :77-96, `RESERVED_KEYS` :109-120, a new
  `OverlapConfig` dataclass next to `BarrierConfig` :181-189,
  `MethodologyConfig` :208-215, `_methodology` :391-494)
- Modify: `delhi_psi/profiles/code-2025.yaml` (a new `overlap:` block)
- Modify: `delhi_psi/profiles/manuscript.yaml` (a new `overlap:` line)
- Test: `tests/test_config.py` (`MINIMAL` :33-41; new loading, rejection and
  reserved-key tests; one row in the enum-message parametrization)
- Test: `tests/test_profiles_match_reference.py` (`knob_for_key` :116-125)
- Test: `tests/test_reference_impl.py` (:590-605 — the synthetic city's
  `MethodologyConfig(...)` gains the new required field)

**Interfaces:**
- Consumes from Task 1: `compute_city(overlap_lending=…)` must already work
  — this task's `knob_for_key` entry is what makes
  `test_every_mapped_knob_is_one_the_reference_actually_implements` call it,
  once per mapped value.
- Produces:
  - `config.OverlapLending` (members `WHOLE` = `"whole"`,
    `OUTSIDE_RECEIVER` = `"outside_receiver"`)
  - `config.OverlapConfig(lending: OverlapLending)` — frozen, one field, no
    default
  - `config.MethodologyConfig.overlap: OverlapConfig`
  - `RESERVED_KEYS["methodology.overlap.counting"]`
  - both shipped profiles carrying `overlap: {lending: whole}`

- [ ] **Step 1: Write the failing config tests**

In `tests/test_config.py`, first give `MINIMAL` the new required key — the
`overlap:` line goes between `barrier:` and `decay:`, matching the block
order in `code-2025.yaml`:

```python
MINIMAL = """profile: minimal
""" + CATEGORIES_BLOCK + """methodology:
  adjacency: {rule: bbox}
  barrier: {rule: global_asymmetric, combine: any}
  overlap: {lending: whole}
  decay: {form: inverse_linear, distance: centroid, distance_unit: km}
  roads: decayed
  second_normalization: true
  exclusion: {types: [RV], stage: post_neighbors, absent_neighbor: swallowed}
"""
```

Then append, next to the 3E `partial_weighted` tests at the end of the file:

```python
# --- 3E: the overlap lending switch (spec § 1, § 3) --------------------
OVERLAP_OUTSIDE_BLOCK = "  overlap: {lending: outside_receiver}"


def test_overlap_lending_loads_both_values(tmp_path):
    outside = load_config(write(tmp_path, swap("  overlap:",
                                               OVERLAP_OUTSIDE_BLOCK)),
                          data_dir=str(tmp_path))
    assert outside.methodology.overlap.lending == "outside_receiver"
    whole = load_config(write(tmp_path, MINIMAL, name="whole.yaml"),
                        data_dir=str(tmp_path))
    assert whole.methodology.overlap.lending == "whole"


@pytest.mark.parametrize("profile", ["code-2025", "manuscript"])
def test_shipped_profiles_name_todays_lending_rule_explicitly(profile,
                                                              tmp_path):
    """The one key both profiles gain this cycle. `whole` names the
    arithmetic they have always run — a record, not a change. Raj has
    confirmed only the COUNTING half of memo § 6; the lending half is Bob's
    proposal and DEL-31 decides whether the ratified profile carries it."""
    cfg = load_config(profile, data_dir=str(tmp_path))
    assert cfg.methodology.overlap.lending == "whole"


def test_overlap_is_required_like_every_methodology_block(tmp_path):
    without = MINIMAL.replace("  overlap: {lending: whole}\n", "")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, without))
    assert "methodology.overlap" in str(exc.value)


def test_lending_is_required_inside_the_overlap_block(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, swap("  overlap:", "  overlap: {}")))
    assert "methodology.overlap.lending" in str(exc.value)


def test_an_unknown_overlap_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, swap(
            "  overlap:", "  overlap: {lending: whole, share: half}")))
    assert "methodology.overlap.share" in str(exc.value)


@pytest.mark.parametrize("value", ["each", "single", "true"])
def test_reserved_key_overlap_counting_rejects_every_value(tmp_path, value):
    """A KNOWN optional key: any value takes the reserved path, never the
    unknown-key path. It is where the reader learns why only ONE half of
    memo § 6 is a switch — the counting half was ratified as today's
    behaviour, so there is nothing to choose."""
    text = swap("  overlap:",
                f"  overlap: {{lending: whole, counting: {value}}}")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, text))
    message = str(exc.value)
    assert "unknown key" not in message
    assert message.endswith(RESERVED_KEYS["methodology.overlap.counting"])
    assert "28 Aug 2026" in message
```

and add ONE row to the existing
`test_new_enums_name_the_key_and_the_allowed_values` parametrization (the
same shape its `decay.form` and `decay.distance` rows use):

```python
    ("methodology.overlap.lending", "  overlap:",
     "  overlap: {lending: sideways}"),
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_config.py -k "overlap or lending"`

Expected: **FAIL.** `test_overlap_lending_loads_both_values` fails with
`ConfigError: unknown key 'methodology.overlap'; allowed keys here: [...]`
— `MINIMAL` now carries a block the loader does not know. The reserved-key
test fails with `KeyError: 'methodology.overlap.counting'` while building
its own assertion. Note that with `MINIMAL` changed, **most of the file is
now red**, which is expected and is the point of doing `MINIMAL` first.
Record the text.

- [ ] **Step 3: Add the switch to the enum table**

In `delhi_psi/config.py`, extend the three tables that make a value
loadable. The `overlap` entry goes directly after the `barrier` one, in
every list, so the file reads in the same order as the YAML:

```python
REFERENCE_KNOBS = {
    "methodology.adjacency.rule": {"bbox": "bbox", "touch": "border",
                                   "within_distance": "within_distance"},
    "methodology.barrier.rule": {"global_asymmetric": "global",
                                 "pairwise": "pair",
                                 "partial_weighted": "partial_weighted"},
    "methodology.overlap.lending": {"whole": "whole",
                                    "outside_receiver": "outside_receiver"},
    "methodology.decay.form": {"inverse_linear": "inverse_linear",
                               "none": "none",
                               "inverse_power": "inverse_power",
                               "exponential": "exponential"},
    "methodology.decay.distance": {"centroid": "centroid",
                                   "boundary": "boundary"},
    "methodology.roads": {"decayed": "decayed", "eq4_own_only": "eq4"},
    "methodology.second_normalization": {True: True, False: False},
    "methodology.exclusion.stage": {"post_neighbors": False,
                                    "pre_neighbors": True},
    "methodology.exclusion.absent_neighbor": {"swallowed": "swallowed",
                                              "contributes": "contributes"},
    "outputs.denominators[]": {"pop": "pop", "popdensity": "popdensity"},
}
```

```python
ENUM_KEYS = (
    "methodology.adjacency.rule",
    "methodology.barrier.rule",
    "methodology.overlap.lending",
    "methodology.decay.form",
    "methodology.decay.distance",
    "methodology.roads",
    "methodology.exclusion.stage",
    "methodology.exclusion.absent_neighbor",
    "outputs.denominators[]",
)
```

```python
BarrierRule = _make_enum("BarrierRule", "methodology.barrier.rule")
OverlapLending = _make_enum("OverlapLending", "methodology.overlap.lending")
DecayForm = _make_enum("DecayForm", "methodology.decay.form")
```

```python
ENUMS = {
    "methodology.adjacency.rule": AdjacencyRule,
    "methodology.barrier.rule": BarrierRule,
    "methodology.overlap.lending": OverlapLending,
    "methodology.decay.form": DecayForm,
    "methodology.decay.distance": DecayDistance,
    "methodology.roads": RoadsFormula,
    "methodology.exclusion.stage": ExclusionStage,
    "methodology.exclusion.absent_neighbor": AbsentNeighbor,
    "outputs.denominators[]": Denominator,
}
```

The spelling is deliberately the SAME on both sides, like every 3D and 3E
value, so `tests/variants.py` needs no translation layer.

- [ ] **Step 4: Reserve the counting half**

In `RESERVED_KEYS`, add the entry — same mechanism and same reason as
`exclusion.minmax_universe`:

```python
    "methodology.overlap.counting":
        "reserved: Raj ratified on 28 Aug 2026 that a service inside k "
        "overlapping colonies counts for each of the k — today's behaviour "
        "on both sides (production's `intersects`, the reference's "
        "`within`) — so there is no knob and nothing to choose (decision "
        "log § 5). The OTHER half of that memo section, what a neighbour "
        "LENDS, is the switch `methodology.overlap.lending`.",
```

- [ ] **Step 5: Add the dataclass and the loader branch**

Immediately after `BarrierConfig` in `delhi_psi/config.py`:

```python
@dataclass(frozen=True)
class OverlapConfig:
    # `whole` names today's arithmetic: a neighbour lends its whole amount
    # S_j. `outside_receiver` lends |S_j \ S_i| — the amount minus whatever
    # of the same service already lies inside the receiver, so a service in
    # the overlap of two colonies is never counted twice for one of them.
    # Required like every methodology key: no default, never inherited.
    lending: OverlapLending
```

and give `MethodologyConfig` the field, after `barrier`:

```python
@dataclass(frozen=True)
class MethodologyConfig:
    adjacency: AdjacencyConfig
    barrier: BarrierConfig
    overlap: OverlapConfig
    decay: DecayConfig
    roads: RoadsFormula
    second_normalization: bool
    exclusion: ExclusionConfig
```

In `_methodology`, extend the top-level key set:

```python
    _reject_unknown(raw, {"adjacency", "barrier", "overlap", "decay", "roads",
                          "second_normalization", "exclusion"}, "methodology")
```

and add the block itself immediately after the `barrier = BarrierConfig(...)`
statement (:423-430) and before the `decay_raw` block:

```python
    overlap_raw = _require(raw, "overlap", "methodology")
    _reject_unknown(overlap_raw, {"lending"}, "methodology.overlap")
    overlap = OverlapConfig(
        lending=_coerce_enum(
            "methodology.overlap.lending",
            _require(overlap_raw, "lending", "methodology.overlap")))
```

and pass it in the return:

```python
    return MethodologyConfig(
        adjacency=adjacency,
        barrier=barrier,
        overlap=overlap,
        decay=decay,
        roads=_coerce_enum("methodology.roads",
                           _require(raw, "roads", "methodology")),
        second_normalization=_bool(
            "methodology.second_normalization",
            _require(raw, "second_normalization", "methodology")),
        exclusion=exclusion)
```

- [ ] **Step 6: Add the key to both shipped profiles**

In `delhi_psi/profiles/code-2025.yaml`, insert between the `barrier:` block
and the `decay:` block. **Only this block is added; no existing line
changes:**

```yaml
  overlap:
    lending: whole                  # whole | outside_receiver — what a
                                    # NEIGHBOUR lends. `whole` is today's
                                    # rule: the neighbour's whole amount S_j.
                                    # `outside_receiver` lends |S_j \ S_i|,
                                    # so a service inside the overlap of two
                                    # colonies is not counted a second time
                                    # through the neighbour term (DEL-20;
                                    # Raj has confirmed the counting half
                                    # only, so `whole` ships)
                                    # counting: reserved — a service inside k
                                    # overlapping colonies counts for each of
                                    # the k (ratified 28 Aug 2026; no knob)
```

In `delhi_psi/profiles/manuscript.yaml`, insert one line directly after the
`barrier:` line:

```yaml
  barrier: {rule: pairwise, combine: any}
  overlap: {lending: whole}         # whole | outside_receiver (DEL-20); the
                                    # manuscript is silent, so today's rule
```

- [ ] **Step 7: Map the knob and fix the one direct dataclass construction**

In `tests/test_profiles_match_reference.py::test_every_mapped_knob_is_one_the_reference_actually_implements`,
add one line to `knob_for_key` (keeping the table in `REFERENCE_KNOBS`
order):

```python
    knob_for_key = {
        "methodology.adjacency.rule": "adjacency_rule",
        "methodology.barrier.rule": "barrier_rule",
        "methodology.overlap.lending": "overlap_lending",
        "methodology.decay.form": "decay_form",
        "methodology.decay.distance": "decay_distance",
        "methodology.roads": "roads_formula",
        "methodology.second_normalization": "second_norm",
        "methodology.exclusion.absent_neighbor": "absent_neighbor_contribution",
        "outputs.denominators[]": "denom",
    }
```

`EXTRA_PARAMS` is **not** touched: neither lending value needs a conditional
parameter.

In `tests/test_reference_impl.py::test_production_matches_the_reference_on_synthetic_partial_geometry`,
`MethodologyConfig` now has a required field, so extend the import and the
construction (nothing else in that test changes; Task 7 revisits it):

```python
    from delhi_psi.config import (
        AbsentNeighbor, AdjacencyConfig, AdjacencyRule, BarrierConfig,
        BarrierRule, DecayConfig, DecayDistance, DecayForm, ExclusionConfig,
        ExclusionStage, MethodologyConfig, OverlapConfig, OverlapLending,
        RoadsFormula,
    )
```

```python
    methodology = MethodologyConfig(
        adjacency=AdjacencyConfig(rule=AdjacencyRule.BBOX),
        barrier=BarrierConfig(rule=BarrierRule.PARTIAL_WEIGHTED,
                              combine="any", buffer_m=5.0),
        overlap=OverlapConfig(lending=OverlapLending.WHOLE),
        decay=DecayConfig(form=DecayForm.INVERSE_LINEAR, distance_unit="km",
                          distance=DecayDistance.CENTROID),
        roads=RoadsFormula.DECAYED,
        second_normalization=True,
        exclusion=ExclusionConfig(types=(), stage=ExclusionStage.POST_NEIGHBORS,
                                  absent_neighbor=AbsentNeighbor.SWALLOWED))
```

- [ ] **Step 8: Run the config and profile tests**

Run: `uv run pytest -q -W error tests/test_config.py tests/test_profiles_match_reference.py`

Expected: PASS. Note in particular that `test_defaults_equal_code_2025` now
also asserts the new key (it compares `minimal.methodology == full.methodology`
in full), and that `test_out_of_enum_names_key_and_allowed_values`,
`test_enums_are_generated_from_the_reference_table` and
`test_enums_cover_exactly_the_reference_table` pick the new value up
automatically because they all read `REFERENCE_KNOBS`.

- [ ] **Step 9: Prove the two shipped profiles' numbers did not move**

This is the delicate part of the whole ticket: a REQUIRED key was added to
both profiles, so every committed number they produce must be re-derived and
come back identical.

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
uv run python scripts/generate_production_fixtures.py
git status --porcelain tests/fixtures/
git diff --stat tests/fixtures/
```

Expected: **both commands print nothing.** `expected_values.csv` (both
cities), `production/code-2025.csv` and `production/manuscript.csv` (both
cities) and `variants_expected_values.csv` (both cities) are all
byte-identical. Any modified file here is the owner's hard condition (spec
§ 0.2, § 11): **STOP and report**, do not commit and do not regenerate
around it.

Then the same claim from the test side:

```bash
uv run pytest -q -W error tests/test_production_fixtures.py tests/test_manuscript_anchors.py
```

Expected: PASS — both shipped profiles produce byte-identical fixture
outputs, and the hand-ratified worksheet anchors are untouched.

- [ ] **Step 10: Run this task's files, then commit**

Run: `uv run pytest -q -W error tests/test_config.py tests/test_profiles_match_reference.py tests/test_reference_impl.py tests/test_production_fixtures.py tests/test_manuscript_anchors.py`

Expected: PASS. Do NOT run the whole suite.

```bash
git add delhi_psi/config.py delhi_psi/profiles/code-2025.yaml \
        delhi_psi/profiles/manuscript.yaml tests/test_config.py \
        tests/test_profiles_match_reference.py tests/test_reference_impl.py
git commit -m "$(cat <<'MSG'
feat(overlap): methodology.overlap.lending is a required switch (DEL-20)

whole | outside_receiver, generated from REFERENCE_KNOBS like every other
reference-pinned enum, with the same spelling on both sides so the variant
table needs no translation layer. The key is REQUIRED — methodology has no
defaults and never inherits — so both shipped profiles and test_config's
MINIMAL gain `overlap: {lending: whole}`, which names the arithmetic they
have always run.

methodology.overlap.counting is RESERVED with the reason: Raj ratified on
28 Aug 2026 that a service inside k overlapping colonies counts for each of
the k, so there is no knob. That is how the next reader learns why only one
half of memo § 6 is a switch.

Nothing either profile computes moves: all three generators re-emit both
cities' expected_values.csv, both production/*.csv and both
variants_expected_values.csv byte-identically.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 3: `index.shared_amounts` and `pcen`'s `lent`

**Files:**
- Modify: `delhi_psi/index.py` (imports :17-22; a new function after
  `service_amount_column` :77-89; `pcen` :138-220; `service_index` :261-281)
- Test: `tests/test_index.py` (append, after the 3E barrier-weight section
  at :276-329)

**Interfaces:**
- Consumes from Task 2: nothing at runtime — `index.py` never imports
  `delhi_psi.config` and takes plain keyword arguments.
- Produces:
  - `index.shared_amounts(polygon_gdf, service_gdf, *, kind, amount_col,
    neighbor_col="nbrs_bbox", id_col="USO_AREA_U") ->
    dict[tuple[str, str], int | float]` — sparse, symmetric, ordered pairs.
  - `index.pcen(..., shared_amounts=None)` and
    `index.service_index(..., shared_amounts=None)`; the neighbour term
    becomes `poly_count += w * lent * _decay(...)` with
    `lent = amount_j - shared_amounts.get((row_id, nbr_id), 0)`.

**A naming note the implementer will notice.** The module-level function and
`pcen`'s parameter have the same name, `shared_amounts` — the spec names
both (§ 3.2, § 3.3). Inside `pcen` the parameter shadows the function, which
is harmless because `pcen` never calls it. Keep both names as the spec has
them; do not "fix" one of them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_index.py`:

```python
# --- 3E: overlap lending (spec § 3.1-3.3, § 6.4) -----------------------
def overlap_city():
    """P and Q OVERLAP in x in [1000, 1200]; Z is 4 km away and disjoint.

    One clinic sits in the overlap (so it is P's own AND Q's own), one in P
    alone; one road runs from x = 500 to x = 1500 at y = 200, so 200 m of it
    lie inside BOTH P and Q. The amount columns are the ones
    `index_frames` would have computed: clinic P 2, Q 1, Z 0; road P 0.7 km,
    Q 0.5 km, Z 0.
    """
    return gpd.GeoDataFrame(
        {"USO_AREA_U": ["P", "Q", "Z"],
         "nbrs_bbox": [["Q"], ["P"], []],
         "clinic_count": [2, 1, 0],
         "road_length": [0.7, 0.5, 0.0]},
        geometry=[box(0, 0, 1200, 1000), box(1000, 0, 2000, 1000),
                  box(5000, 0, 6000, 1000)],
        crs="EPSG:7760")


def overlap_clinics():
    return gpd.GeoDataFrame(
        {"service": ["clinic", "clinic"]},
        geometry=[Point(1100, 500), Point(600, 500)], crs="EPSG:7760")


def overlap_roads():
    from shapely.geometry import LineString

    return gpd.GeoDataFrame(
        {"service": ["road"]},
        geometry=[LineString([(500, 200), (1500, 200)])], crs="EPSG:7760")


def test_shared_amounts_counts_a_point_inside_two_settlements():
    """One ENTRY per ordered pair that shares something, and nothing at all
    for the point inside P alone or for the disjoint third settlement — the
    sparse representation the cost argument rests on."""
    got = index.shared_amounts(overlap_city(), overlap_clinics(),
                               kind="point", amount_col="clinic_count")
    assert got == {("P", "Q"): 1, ("Q", "P"): 1}


def test_shared_amounts_measures_the_road_inside_the_overlap():
    """200 m of the road lie in P n Q, so 0.2 km is lent by neither side."""
    got = index.shared_amounts(overlap_city(), overlap_roads(),
                               kind="line", amount_col="road_length")
    assert got == {("P", "Q"): pytest.approx(0.2, abs=1e-12),
                   ("Q", "P"): pytest.approx(0.2, abs=1e-12)}


def test_shared_amounts_is_empty_when_nothing_is_shared():
    """A clean layer costs nothing: every point inside one settlement, every
    neighbour pair a plain border. The dict is EMPTY, not full of zeroes."""
    city = overlap_city()
    only_p = gpd.GeoDataFrame({"service": ["clinic"]},
                              geometry=[Point(600, 500)], crs="EPSG:7760")
    assert index.shared_amounts(city, only_p, kind="point",
                                amount_col="clinic_count") == {}


def test_shared_amounts_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="polygon"):
        index.shared_amounts(overlap_city(), overlap_clinics(),
                             kind="polygon", amount_col="clinic_count")


def city_with_a_shared_clinic():
    """X and Y, 1 km apart (decay 1/2), each owning the SAME one clinic —
    X owns a second of its own. This is the O1/O2 shape at the pcen level."""
    gdf = city_with_neighbours()
    gdf["clinic_count"] = [2.0, 1.0]
    return gdf


def test_pcen_subtracts_what_the_receiver_already_holds():
    """Y already holds the shared clinic, so X lends it (2 - 1) = 1; X holds
    Y's only clinic, so Y lends it nothing at all."""
    shared = {("X", "Y"): 1, ("Y", "X"): 1}
    got = index.pcen(city_with_a_shared_clinic(), amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     shared_amounts=shared)
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx((1 + (2 - 1) * 0.5) / 200, abs=1e-12)
    assert values["X"] == pytest.approx((2 + (1 - 1) * 0.5) / 100, abs=1e-12)


def test_an_empty_shared_structure_is_bit_identical_to_no_structure():
    """A pair with no entry is `|S_j \\ S_i| == |S_j|` EXACTLY: the sparse 0
    is the representation of 'nothing shared', not a swallowed miss."""
    sparse = index.pcen(city_with_a_shared_clinic(),
                        amount_col="clinic_count", pcen_col="clinic_pcen",
                        denominator="pop", shared_amounts={})
    plain = index.pcen(city_with_a_shared_clinic(),
                       amount_col="clinic_count", pcen_col="clinic_pcen",
                       denominator="pop")
    assert list(sparse["clinic_pcen"]) == list(plain["clinic_pcen"])


def test_the_barrier_weight_and_the_overlap_rule_compose():
    """The one place both multipliers act on one pair: a half-blocked shared
    boundary halves what is left after the overlap subtraction."""
    frame = city_with_a_shared_clinic()
    frame["nbrs_barrier_weight"] = [[("Y", 0.5)], [("X", 0.5)]]
    got = index.pcen(frame, amount_col="clinic_count",
                     pcen_col="clinic_pcen", denominator="pop",
                     nbr_weight_col="nbrs_barrier_weight",
                     shared_amounts={("X", "Y"): 1, ("Y", "X"): 1})
    values = got.set_index("USO_AREA_U")["clinic_pcen"]
    assert values["Y"] == pytest.approx(
        (1 + 0.5 * (2 - 1) * 0.5) / 200, abs=1e-12)


def test_a_shared_amount_larger_than_the_neighbours_own_raises():
    """S_j n S_i is part of S_j, so shared_ij <= amount_j on both sides by
    construction. A negative lent means the two frames came from different
    runs; validate.check_no_negative would report it much later as a data
    problem, so it is caught here instead."""
    with pytest.raises(ValueError, match="would lend"):
        index.pcen(city_with_a_shared_clinic(), amount_col="clinic_count",
                   pcen_col="clinic_pcen", denominator="pop",
                   shared_amounts={("Y", "X"): 5})


def test_service_index_forwards_the_shared_structure():
    got = index.service_index(city_with_a_shared_clinic(), "clinic_count",
                              service="clinic", denominator="pop",
                              shared_amounts={("X", "Y"): 1, ("Y", "X"): 1})
    values = got.set_index("USO_AREA_U")
    assert values.loc["Y", "clinic_pcen"] == pytest.approx(
        (1 + (2 - 1) * 0.5) / 200, abs=1e-12)
    # min-max still runs: X is the max, Y the min
    assert values.loc["X", "clinic_idx"] == 1.0
    assert values.loc["Y", "clinic_idx"] == 0.0
```

`gpd`, `pytest`, `Point`, `box` and `index` are already imported at the top
of `tests/test_index.py`; `LineString` is not, which is why
`overlap_roads()` imports it locally.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_index.py -k "shared or overlap or compose"`

Expected: **FAIL** with `AttributeError: module 'delhi_psi.index' has no
attribute 'shared_amounts'` on the four structure tests, and
`TypeError: pcen() got an unexpected keyword argument 'shared_amounts'` on
the rest. Record the text.

- [ ] **Step 3: Implement `shared_amounts`**

In `delhi_psi/index.py`, extend the imports:

```python
import logging
import math

import geopandas as gpd
from shapely import STRtree
```

and insert this function immediately after `service_amount_column`:

```python
def shared_amounts(polygon_gdf, service_gdf, *, kind, amount_col,
                   neighbor_col="nbrs_bbox", id_col="USO_AREA_U"):
    """{(i, j): amount} — how much of ONE service lies inside BOTH i and j.

    This is the `shared_ij` of `|S_j \\ S_i| = amount_j - shared_ij`
    (`overlap.lending: outside_receiver`). Sparse and symmetric: only pairs
    that actually share something get an entry, in both orders, and `pcen`
    reads it with `.get((i, j), 0)`. That 0 IS the representation of
    "nothing shared" — for every pair whose polygons do not overlap,
    `|S_j \\ S_i| == |S_j|` exactly and no arithmetic happens.

    point: ONE sjoin — the same boundary-inclusive `intersects` join
        `point_counts` does, so `shared_ij <= amount_j` by construction —
        grouped by POINT. A point inside k settlements contributes 1 to each
        of the k(k-1) ordered pairs among them; a point inside one
        settlement, which is every point on a clean layer, contributes
        nothing and is never looked at again. Neighbour status is irrelevant
        here: recording every sharing pair is cheaper than filtering it, and
        `pcen` only ever looks up neighbour pairs.
    line: the clipped length in km inside `geom_i n geom_j`, over the stored
        NEIGHBOUR lists only, skipping any pair where either side owns none
        of the service (it can then share none of it) and any pair whose
        intersection is empty. The road rows are indexed once in an STRtree,
        so each surviving pair clips only the candidates its own
        intersection returns.

    Never call this under `overlap.lending: whole`: there is nothing to
    subtract, and the caller passes `shared_amounts=None` so the neighbour
    loop stays bit-identical to today's.
    """
    if kind == "point":
        # Two columns only, so a service layer that happens to carry a
        # column of the same name cannot make sjoin add suffixes; and a
        # plain RangeIndex on the right, so the join column is always
        # `index_right` (geopandas names it after the right index when that
        # index HAS a name).
        left = polygon_gdf[[id_col, polygon_gdf.geometry.name]]
        joined = gpd.sjoin(left, service_gdf.reset_index(drop=True))
        table = {}
        for _, group in joined.groupby("index_right"):
            owners = list(group[id_col])
            if len(owners) < 2:
                continue
            for i in owners:
                for j in owners:
                    if i != j:
                        table[(i, j)] = table.get((i, j), 0) + 1
        return table
    if kind == "line":
        geoms = polygon_gdf.set_index(id_col).geometry
        amounts = polygon_gdf.set_index(id_col)[amount_col]
        lines = list(service_gdf.geometry)
        tree = STRtree(lines)
        pairs = set()
        for _, row in polygon_gdf.iterrows():
            i = row[id_col]
            for j in row[neighbor_col]:
                # An id with no row here lends nothing anyway — `pcen` skips
                # it under `swallowed` and reads it from the lookup frame
                # under `contributes`, where the amount is 0 for a row this
                # frame does not have.
                if j in geoms.index and amounts[i] > 0 and amounts[j] > 0:
                    pairs.add((i, j) if i < j else (j, i))
        table = {}
        # Sorted UNDIRECTED pairs: each is measured once and written both
        # ways, so the table is symmetric by construction.
        for i, j in sorted(pairs):
            overlap = geoms[i].intersection(geoms[j])
            if overlap.is_empty:
                continue
            length = sum(lines[k].intersection(overlap).length
                         for k in tree.query(overlap)) / 1000
            if length > 0:
                table[(i, j)] = length
                table[(j, i)] = length
        return table
    raise ValueError(
        f"unknown service kind {kind!r}; allowed values: ['point', 'line']")
```

- [ ] **Step 4: Redefine `lent` in `pcen`**

In `delhi_psi/index.py::pcen`, add the parameter after `nbr_weight_col` and
document it:

```python
def pcen(polygon_gdf, *, amount_col, pcen_col, denominator,
         nbr_dist_col="nbrs_dist_bbox", nbr_weight_col=None,
         shared_amounts=None,
         lookup_frame=None,
         absent_neighbor="swallowed", include_neighbors=True,
         decay_form="inverse_linear", distance_unit="km", exponent=None,
         scale_km=None,
         pop_col="population", area_col="area_km2", id_col="USO_AREA_U"):
```

and append to its docstring, after the `nbr_weight_col` paragraph:

```
    shared_amounts: the {(i, j): amount} structure `shared_amounts()` builds
        under `overlap.lending: outside_receiver`. A neighbour then lends
        |S_j \ S_i| — its amount minus whatever of the same service already
        lies inside the receiver — so a service in the overlap of two
        colonies is never counted twice for one of them. None means today's
        rule (`whole`), and then no subtraction happens at all, which is
        what keeps the default path bit-identical.
```

Then replace the two lines inside the neighbour loop

```python
                lent = match[amount_col].array[0]
                poly_count += w * lent * _decay(nbr_dist, decay_form,
```

with

```python
                lent = match[amount_col].array[0]
                if shared_amounts is not None:
                    lent = lent - shared_amounts.get((row[id_col], nbr_id), 0)
                    if lent < 0:
                        raise ValueError(
                            f"overlap lending: {nbr_id!r} would lend "
                            f"{lent} of {amount_col!r} to {row[id_col]!r}. "
                            "S_j n S_i is part of S_j, so the shared amount "
                            "can never exceed the neighbour's own — the "
                            "amounts frame and the shared structure came "
                            "from different runs.")
                poly_count += w * lent * _decay(nbr_dist, decay_form,
```

(the rest of the `_decay(...)` call is unchanged).

- [ ] **Step 5: Forward it from `service_index`**

```python
def service_index(polygon_gdf, amount_col, *, service, denominator,
                  nbr_dist_col="nbrs_dist_bbox", nbr_weight_col=None,
                  shared_amounts=None,
                  lookup_frame=None,
                  absent_neighbor="swallowed", include_neighbors=True,
                  decay_form="inverse_linear", distance_unit="km",
                  exponent=None, scale_km=None,
                  pop_col="population", area_col="area_km2",
                  id_col="USO_AREA_U"):
    """pcen then minmax for one service — replaces BOTH create_service_index
    variants (DEL-16). Fed by point_counts() or road_lengths()."""
    pcen_col = f"{service}_pcen"
    idx_col = f"{service}_idx"
    out = pcen(polygon_gdf, amount_col=amount_col, pcen_col=pcen_col,
               denominator=denominator, nbr_dist_col=nbr_dist_col,
               nbr_weight_col=nbr_weight_col, shared_amounts=shared_amounts,
               lookup_frame=lookup_frame, absent_neighbor=absent_neighbor,
               include_neighbors=include_neighbors, decay_form=decay_form,
               distance_unit=distance_unit, exponent=exponent,
               scale_km=scale_km, pop_col=pop_col,
               area_col=area_col, id_col=id_col)
    return minmax(out, source_col=pcen_col, target_col=idx_col)
```

- [ ] **Step 6: Run this task's file, then commit**

Run: `uv run pytest -q -W error tests/test_index.py`

Expected: PASS, including every pre-existing test in the file — the decay
pins, the exclusion pins, DEL-54's guard tests and Group B's barrier-weight
tests. `test_an_empty_shared_structure_is_bit_identical_to_no_structure`
passing is the local form of "nothing on the default path moves". Do NOT
run the whole suite.

```bash
git add delhi_psi/index.py tests/test_index.py
git commit -m "$(cat <<'MSG'
feat(overlap): index.shared_amounts and the lent term it subtracts (DEL-20)

pcen's neighbour term becomes w * lent * decay with lent = amount_j -
shared_ij, where shared_ij is how much of that service already lies inside
the receiver. shared_amounts is None under `whole`, so no subtraction
happens at all and the default path is bit-identical.

The structure is built from the SERVICE POINTS' containment — one sjoin,
grouped by point — so it never touches the neighbour links: a point inside
one settlement is looked at once and contributes nothing. The line branch
walks the stored neighbour lists, skips pairs where either side owns none of
the service or whose intersection is empty, and clips only the STRtree
candidates each surviving intersection returns. On a clean layer the dict
comes back empty and every pair has |S_j \ S_i| == |S_j| exactly.

A shared amount larger than the neighbour's own is impossible by
construction and now raises naming the pair, rather than reaching
validate.check_no_negative as a data problem.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 4: Pipeline wiring — `index_frames` builds and passes the dicts

**Files:**
- Modify: `delhi_psi/pipeline.py` (`index_frames` :164-223)
- Test: `tests/test_pipeline.py` (append, after the 3E barrier section
  :287-360)

**Interfaces:**
- Consumes from Task 2: `methodology.overlap.lending`. Consumes from
  Task 3: `index.shared_amounts(...)` and
  `index.service_index(..., shared_amounts=…)`.
- Produces: no new module-level name, no new column, no stamp entry. One
  INFO line per `index_frames` call under `outside_receiver`.

**Why there is no column and no stamp entry** (spec § 3.2, § 12 item 7): the
overlap rule is downstream of the neighbour structure, like `decay` and
`roads`, so ONE stored artifact serves both values. Nothing is added to
`methodology_stamp`, nothing to `io.SHAPEFILE_DROP_COLUMNS`. Task 6 pins
that.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py`:

```python
# --- 3E: overlap lending is wired per service (spec § 3.2) -------------
def _messy_overlap_frames(lending):
    """The messy city scored through compute_frames with ONE methodology
    value changed, both denominators' worth of plumbing in one call."""
    from dataclasses import replace

    from delhi_psi.config import OverlapConfig, OverlapLending
    from delhi_psi.pipeline import compute_frames
    from tests.cities import MESSY
    from tests.oraculum_fixtures import methodology_with

    methodology = methodology_with("code-2025", types=(), stage=None,
                                   city=MESSY)
    methodology = replace(methodology, overlap=OverlapConfig(
        lending=OverlapLending(lending)))
    return compute_frames(MESSY.load_settlements(),
                          {"canal": MESSY.load_barriers()},
                          MESSY.load_services(), None, methodology, "pop",
                          mapping=MESSY.mapping(),
                          scheme=MESSY.scheme).set_index("USO_AREA_U")


def test_index_frames_builds_one_shared_structure_per_service():
    """The clinic moves and the school does not, in ONE run: a single dict
    reused across services would move both, and a dict built for the wrong
    service would move the wrong one."""
    whole = _messy_overlap_frames("whole")
    outside = _messy_overlap_frames("outside_receiver")
    assert outside.loc["O1", "clinic_pcen"] == pytest.approx(
        1 / 600, abs=1e-12)
    assert whole.loc["O1", "clinic_pcen"] == pytest.approx(
        (1 + 1 / 1.8) / 600, abs=1e-12)
    assert outside.loc["O1", "school_pcen"] == whole.loc["O1", "school_pcen"]
    assert outside.loc["O1", "police_pcen"] == whole.loc["O1", "police_pcen"]


def test_the_overlap_rule_adds_no_column_and_no_stamp_entry():
    """It is applied downstream in `compute`, like decay and roads, so one
    stored artifact serves both values and the output column set is
    lending-independent."""
    from dataclasses import replace

    from delhi_psi.config import OverlapConfig, OverlapLending
    from tests.cities import MESSY
    from tests.oraculum_fixtures import methodology_with

    assert list(_messy_overlap_frames("outside_receiver").columns) == \
        list(_messy_overlap_frames("whole").columns)
    methodology = methodology_with("code-2025", types=(), stage=None,
                                   city=MESSY)
    outside = replace(methodology, overlap=OverlapConfig(
        lending=OverlapLending.OUTSIDE_RECEIVER))
    assert pipeline.methodology_stamp(outside) == \
        pipeline.methodology_stamp(methodology)


def test_no_shared_structure_is_built_under_whole(monkeypatch):
    """`whole` must not pay for a rule it does not use, and must not go near
    the arithmetic: the builder is never called AT ALL, which is the only
    way to state "bit-identical" as a test rather than as a comparison
    against a number that would be equal by construction."""
    from delhi_psi import index

    def refuse(*args, **kwargs):
        raise AssertionError("shared_amounts must not be called under whole")

    monkeypatch.setattr(index, "shared_amounts", refuse)
    got = _messy_overlap_frames("whole")
    assert got.loc["O1", "clinic_pcen"] == pytest.approx(
        (1 + 1 / 1.8) / 600, abs=1e-12)
```

`pytest` and `pipeline` are already imported at the top of
`tests/test_pipeline.py`.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_pipeline.py -k "overlap or shared_structure"`

Expected: **FAIL.** `test_index_frames_builds_one_shared_structure_per_service`
fails on the first assertion — `assert 0.0025925925925925925 ==
0.0016666666666666668 ± 1e-12` — because `index_frames` ignores the switch
and still lends the whole amount. The other two PASS already (nothing is
wired, so nothing differs, and nothing calls the builder); that is fine and
expected — they are the pins that must keep passing after Step 3. Record the
text.

- [ ] **Step 3: Build and pass the dicts**

In `delhi_psi/pipeline.py::index_frames`, keep the reprojected service
frames as they are built (they are needed twice now), then build one
structure per service and hand it to `service_index`. Replace the amounts
block and the service loop:

```python
    # Own amounts are computed over the WHOLE universe, so excluded
    # settlements still have something to lend under absent_neighbor
    # "contributes". They are per-row independent, so computing them for rows
    # that are dropped a moment later cannot change a kept row's value.
    amounts = universe
    layout = service_layout(services)
    projected = {}
    for service, kind, amount_col in layout:
        projected[service] = geometry.reproject(services[service], epsg_code)
        if kind == "point":
            amounts = index.point_counts(amounts, projected[service],
                                         count_col=amount_col, id_col=id_col)
        else:
            amounts = index.road_lengths(amounts, projected[service],
                                         length_col=amount_col, id_col=id_col)

    # `outside_receiver`: what a neighbour lends is |S_j \ S_i|, so each
    # service needs its own {(i, j): shared} table. Built HERE — compute
    # locally, on the FULL universe (pre-row-drop, so under
    # absent_neighbor: contributes an excluded overlapping neighbour's
    # lending is adjusted too), never stored and never in the methodology
    # stamp: the overlap rule is downstream of the neighbour structure, like
    # decay and roads, so one artifact serves both values. Under `whole`
    # this dict stays empty, `shared.get(service)` is None, and the
    # neighbour loop is bit-identical to today's.
    shared = {}
    if methodology.overlap.lending == "outside_receiver":
        for service, kind, amount_col in layout:
            shared[service] = index.shared_amounts(
                amounts, projected[service], kind=kind,
                amount_col=amount_col, neighbor_col=NBRS_COL, id_col=id_col)
        log.info("overlap: lending=%s shared_pairs=%s",
                 methodology.overlap.lending,
                 {name: len(table) for name, table in shared.items()
                  if table})

    out = amounts[~amounts[id_col].isin(dropped)] if dropped else amounts

    for service, kind, amount_col in layout:
        include_neighbors = not (kind == "line"
                                 and methodology.roads == "eq4_own_only")
        out = index.service_index(
            out, amount_col, service=service, denominator=denominator,
            nbr_dist_col=nbr_dist_col, nbr_weight_col=nbr_weight_col,
            shared_amounts=shared.get(service),
            lookup_frame=amounts,
            absent_neighbor=exclusion.absent_neighbor,
            include_neighbors=include_neighbors,
            decay_form=methodology.decay.form,
            distance_unit=methodology.decay.distance_unit,
            exponent=methodology.decay.exponent,
            scale_km=methodology.decay.scale_km,
            id_col=id_col)
```

Nothing else in `index_frames` changes: the `NBRS_DIST_BOUNDARY_COL` /
`NBRS_WEIGHT_COL` drop at the end is untouched, and no new column is
created to drop.

- [ ] **Step 4: Run this task's file, then commit**

Run: `uv run pytest -q -W error tests/test_pipeline.py`

Expected: PASS, all three new tests plus every pre-existing one — including
Group B's `test_the_weight_column_is_dropped_before_index_frames_returns`,
which runs the same code path with a different methodology. Do NOT run the
whole suite.

```bash
git add delhi_psi/pipeline.py tests/test_pipeline.py
git commit -m "$(cat <<'MSG'
feat(overlap): index_frames builds the shared-amount tables (DEL-20)

One table per service, compute-locally, only under outside_receiver, on the
FULL universe frame — so under absent_neighbor: contributes an excluded
overlapping neighbour's lending is adjusted too. Nothing is stored, no
column is added and the methodology stamp is untouched: the overlap rule is
downstream of the neighbour structure, like decay and roads, so ONE
artifact serves both values.

Pinned on the messy city in one run: O1's clinic PCEN falls to 1/600 while
its school and police rows do not move at all, which is what a per-service
table means and what a single reused dict would break. `whole` reproduces
the shipped profile's columns bit for bit.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 5: The `overlap_outside` and `partial_5m_outside` variants

**Files:**
- Modify: `tests/variants.py` (the `VARIANTS` table — append LAST)
- Modify: `tests/oraculum_fixtures.py` (`variant_methodology` :155-198)
- Modify: `tests/test_config.py` (`enum_key` in
  `test_every_variant_block_is_one_the_loader_accepts` :523-526)
- Modify: `tests/test_variant_rules.py` (two wiring assertions)
- Regenerate: `tests/fixtures/oraculum/variants_expected_values.csv`,
  `tests/fixtures/messy/variants_expected_values.csv`

**Interfaces:**
- Consumes from Tasks 1-4: everything. This is the integration commit.
- Produces: `VARIANTS["overlap_outside"]` and
  `VARIANTS["partial_5m_outside"]`, and `variant_methodology`'s `overlap`
  branch.

**Why this is one commit.** Four tests read `tests/variants.py` and each
fails on a row that is not yet everywhere: the loader check in
`test_config.py`, the two CSV checks in `test_reference_impl.py`, and the
production comparison in `test_variants_match_reference.py`. Splitting this
leaves the suite red.

- [ ] **Step 1: Add the two variant rows — LAST in the table**

In `tests/variants.py`, append to `VARIANTS`, after `partial_5m`. Position
matters: Step 5's byte-identity check relies on the new rows being appended
at the END of both CSVs, and `emit_variant_expected_values` iterates
`VARIANT_RULESETS`, a dict comprehension over `VARIANTS`, which preserves
insertion order.

```python
    # DEL-20: a neighbour lends only |S_j \ S_i|. Degenerate on Oraculum —
    # no overlapping polygons and no point inside two settlements, so the
    # shared structure is empty and the rows equal the `code` base, exactly
    # as `boundary` is degenerate there. The messy city carries the pin: its
    # O1/O2 clinic is inside both, so O1's clinic PCEN falls from
    # (1 + 1/1.8)/600 to 1/600 while its school PCEN does not move at all
    # (spec § 6.2).
    "overlap_outside": {
        "overlap": {"lending": "outside_receiver"},
    },
    # Both 3E switches at once: proves the two kwargs are accepted together
    # and both paths run in one compute_frames/compute_city call. It equals
    # `partial_5m` on Oraculum (no overlaps) and `overlap_outside` on the
    # messy city (no barriers) — no fixture city has a barrier across an
    # overlap, so the two multipliers are only simultaneously non-trivial in
    # the synthetic in-test geometry (spec § 5, § 6.4).
    "partial_5m_outside": {
        "barrier": {"rule": "partial_weighted", "combine": "any",
                    "buffer_m": 5.0},
        "overlap": {"lending": "outside_receiver"},
    },
```

- [ ] **Step 2: Teach `variant_methodology` the overlap block**

In `tests/oraculum_fixtures.py::variant_methodology`, extend the local
import and add the branch between the `barrier` one and the `decay` one, so
the branches read in block order:

```python
    from delhi_psi.config import (
        AdjacencyConfig, AdjacencyRule, BarrierConfig, BarrierRule,
        DecayConfig, DecayDistance, DecayForm, OverlapConfig, OverlapLending,
    )
```

```python
    if "overlap" in spec:
        block = spec["overlap"]
        methodology = replace(methodology, overlap=OverlapConfig(
            lending=OverlapLending(block["lending"])))
```

and update the docstring's last paragraph — it says "today the band variants
override `adjacency` and `partial_5m` overrides `barrier`"; make it name the
overlap block too:

```python
    A block the variant does not mention keeps `base`'s: today the band
    variants override `adjacency`, `partial_5m` overrides `barrier`,
    `overlap_outside` overrides `overlap` and `partial_5m_outside` overrides
    both, and every variant states each block it does override IN FULL.
```

- [ ] **Step 3: Extend the loader's enum check**

In `tests/test_config.py::test_every_variant_block_is_one_the_loader_accepts`,
the `enum_key` map gains one line:

```python
    enum_key = {("adjacency", "rule"): "methodology.adjacency.rule",
                ("barrier", "rule"): "methodology.barrier.rule",
                ("overlap", "lending"): "methodology.overlap.lending",
                ("decay", "form"): "methodology.decay.form",
                ("decay", "distance"): "methodology.decay.distance"}
```

- [ ] **Step 4: Pin the wiring on the reference side**

Append to `tests/test_variant_rules.py`:

```python
def test_the_overlap_outside_variant_is_the_code_base_plus_the_lending_rule():
    """The table, the knob map and the hand pins are one thing: the
    variant's rule-set must BE the dict the § 6.2 pins were derived under."""
    assert VARIANT_RULESETS["overlap_outside"] == OVERLAP_OUTSIDE


def test_partial_5m_outside_is_each_of_its_halves_on_the_city_that_shows_it():
    """No fixture city has a barrier across an overlap, so the combined
    variant is `partial_5m` on Oraculum (no overlaps) and `overlap_outside`
    on the messy city (no barriers). Its job in the table is to prove the
    two kwargs are accepted together, not to add a third number."""
    both = variant(ORACULUM, "partial_5m_outside")
    barrier_only = variant(ORACULUM, "partial_5m")
    for column in barrier_only.columns:
        assert list(both[column]) == pytest.approx(
            list(barrier_only[column]), abs=1e-12), ("oraculum", column)
    both = variant(MESSY, "partial_5m_outside")
    overlap_only = variant(MESSY, "overlap_outside")
    for column in overlap_only.columns:
        assert list(both[column]) == pytest.approx(
            list(overlap_only[column]), abs=1e-12), ("messy", column)
```

`VARIANT_RULESETS`, `variant`, `MESSY` and `ORACULUM` are already imported
in that file; `OVERLAP_OUTSIDE` was defined in Task 1.

- [ ] **Step 5: Regenerate both CSVs and prove ADDITION ONLY**

Save nothing by hand: the check compares the regenerated files against the
committed ones line by line. The regenerated file's first N lines must be
**byte-identical** to the committed file, and every remaining line must
belong to one of the two new rules.

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
uv run python - <<'PY'
import subprocess, sys

NEW = (b"overlap_outside,", b"partial_5m_outside,")
ok = True
for city in ("oraculum", "messy"):
    path = f"tests/fixtures/{city}/variants_expected_values.csv"
    old = subprocess.run(["git", "show", f"HEAD:{path}"], check=True,
                         capture_output=True).stdout.splitlines(keepends=True)
    with open(path, "rb") as handle:
        new = handle.readlines()
    same = new[:len(old)] == old
    added = new[len(old):]
    only_new = all(line.startswith(NEW) for line in added)
    per_rule = {name.decode().rstrip(","):
                sum(1 for line in added if line.startswith(name))
                for name in NEW}
    print(f"{city}: pre-existing block byte-identical={same} "
          f"added_rows={len(added)} all_new_rules={only_new} {per_rule}")
    ok = ok and same and only_new and added
sys.exit(0 if ok else 1)
PY
git status --porcelain tests/fixtures/
git diff --stat tests/fixtures/
```

Expected: `pre-existing block byte-identical=True` and `all_new_rules=True`
for **both** cities, `added_rows=644` on oraculum (322 per rule) and
`added_rows=920` on messy (460 per rule), `git status` listing ONLY the two
`variants_expected_values.csv` files, and `git diff --stat` showing
insertions with **zero deletions**. Anything else — a changed
`expected_values.csv`, a changed `production/*.csv`, a deletion in a variants
CSV — is the owner's hard condition (spec § 11): **STOP and report**, do not
commit.

- [ ] **Step 6: Run the invariants guard**

Run: `uv run python scripts/check_oraculum_invariants.py`

Expected: `OK`, exit 0. The generators already ran it before writing (that
is what `emit_checked_variant_expected_values` does), so this is the
standalone confirmation that neither new variant creates a degenerate
min-max group or a tied clinic/school anchor on either city — the same fact
Task 1's `test_no_service_column_is_constant_under_overlap_outside` asserts
from the other side.

- [ ] **Step 7: Run this task's files, then commit**

Run: `uv run pytest -q -W error tests/test_variant_rules.py tests/test_variants_match_reference.py tests/test_reference_impl.py tests/test_config.py`

Expected: PASS.
`test_production_matches_the_reference_on_each_variant` now runs
`overlap_outside` and `partial_5m_outside` on both cities × both
denominators — that is production == reference at 1e-12 on the new rule, and
it is the core proof of this ticket. Do NOT run the whole suite.

```bash
git add tests/variants.py tests/oraculum_fixtures.py tests/test_config.py \
        tests/test_variant_rules.py \
        tests/fixtures/oraculum/variants_expected_values.csv \
        tests/fixtures/messy/variants_expected_values.csv
git commit -m "$(cat <<'MSG'
test(overlap): the overlap_outside and partial_5m_outside variants (DEL-20)

outside_receiver ships as a VARIANT, never as a change to the ideal/code
rule-sets, so every existing expected value stays byte-identical by
construction: both variants CSVs changed by the ADDITION of the two new
rules' rows alone, verified line by line against the committed files (322
rows per rule on oraculum, 460 on messy, zero deletions).

The messy city carries the pin and Oraculum is degenerate under the rule —
no overlapping polygons and no point inside two settlements, so the shared
structure is empty there, which is itself worth pinning. partial_5m_outside
proves the two 3E switches are accepted together in one call; no fixture
city has a barrier across an overlap, so it equals partial_5m on Oraculum
and overlap_outside on the messy city, and the genuinely combined case lives
in synthetic geometry.

test_variants_match_reference now proves production == the reference at
1e-12 on both new variants, both cities, both denominators.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 6: The messy production pin, the CLI leg, and the artifact that stays valid

**Files:**
- Modify: `tests/test_messy_fixtures.py` (append a new section after the
  DEL-20 counting pin at :134-149)
- Modify: `tests/test_variants_match_reference.py` (the parametrization at
  :84-85)
- Modify: `tests/test_cli.py` (the stamp section, after
  `test_changing_only_the_decay_does_not_invalidate_an_artifact` :412-428)

**Interfaces:**
- Consumes from Task 5: the `overlap_outside` variant row and
  `variant_methodology`'s overlap branch. Consumes from Task 4: the fact
  that `methodology_stamp` does not carry `overlap`.
- Produces: no new production names.

- [ ] **Step 1: Write the direct production pin, beside the counting pin**

Append to `tests/test_messy_fixtures.py`, immediately after
`test_the_overlap_clinic_is_counted_for_both_owners` — the two pins belong
side by side, because they are the two halves of memo § 6 and one must not
move while the other does:

```python
# --- the overlap lending rule (DEL-20, second half) ---------------------
def overlap_outside_frame(denom="pop"):
    """The messy city under the `overlap_outside` variant methodology.

    Not memoised through `frame`: that helper keys on a PROFILE name, and
    this is a derived methodology, not a shipped profile.
    """
    from delhi_psi.pipeline import compute_frames
    from tests.oraculum_fixtures import variant_methodology

    scenario = SCENARIOS["nopop_only"]
    methodology = variant_methodology(
        BBOX_PROFILE, "overlap_outside", city=MESSY,
        types=scenario.exclusion_types, stage=scenario.stage)
    return compute_frames(MESSY.load_settlements(),
                          {"canal": MESSY.load_barriers()},
                          MESSY.load_services(), None, methodology, denom,
                          mapping=MESSY.mapping(),
                          scheme=MESSY.scheme).set_index("USO_AREA_U")


def test_the_overlap_clinic_is_lent_back_today_and_not_under_the_switch():
    """The two halves of memo § 6, side by side on one pair.

    Raj RATIFIED the counting half: the clinic strictly inside O1 n O2 is
    O1's own AND O2's own, so `clinic_count` is 1 for each under BOTH
    values — that is `test_the_overlap_clinic_is_counted_for_both_owners`
    above, and it does not move.

    Bob ADDED the lending half, still out for Raj's answer: under
    `code-2025` the same clinic also arrives at O1 as O2's, decayed over the
    0.8 km between the centroids, so it reaches O1 twice. Under
    `overlap.lending: outside_receiver` O2 lends |S_O2 \\ S_O1| = 0 of it and
    O1 keeps only its own.
    """
    today = frame(BBOX_PROFILE, "nopop_only", "pop")
    assert today.loc["O1", "clinic_pcen"] == pytest.approx(
        (1 + 1 / 1.8) / 600, abs=1e-12)
    switched = overlap_outside_frame()
    assert switched.loc["O1", "clinic_count"] == 1
    assert switched.loc["O2", "clinic_count"] == 1
    assert switched.loc["O1", "clinic_pcen"] == pytest.approx(
        1 / 600, abs=1e-12)
    assert switched.loc["O2", "clinic_pcen"] == pytest.approx(
        1 / 700, abs=1e-12)


def test_only_the_shared_service_moves_on_the_overlapping_pair():
    """O2's school at (11400, 500) is OUTSIDE O1, so it is lent in full and
    O1's school row does not move by a single bit. The rule is about what is
    shared, not about who is an overlapping neighbour."""
    today = frame(BBOX_PROFILE, "nopop_only", "pop")
    switched = overlap_outside_frame()
    assert switched.loc["O1", "school_pcen"] == today.loc["O1", "school_pcen"]
    assert switched.loc["O1", "police_pcen"] == today.loc["O1", "police_pcen"]
    # and no settlement without an overlapping neighbour is touched at all
    for sid in ("H", "L", "T", "M", "G", "I", "S"):
        for column in [c for c in today.columns if c.endswith("_pcen")]:
            assert switched.loc[sid, column] == today.loc[sid, column], (sid,
                                                                        column)
```

- [ ] **Step 2: Add `overlap_outside` to the CLI round trip**

In `tests/test_variants_match_reference.py`, extend the parametrization and
the docstring of `test_a_derived_variant_profile_runs_end_to_end`:

```python
@pytest.mark.parametrize("variant", ["band_small_boundary", "exp1",
                                     "partial_5m", "overlap_outside"])
def test_a_derived_variant_profile_runs_end_to_end(expected, data_dir,  # noqa: F811
                                                   tmp_path, variant):
    """Proves the whole chain the in-memory test skips: YAML -> load_config
    -> preprocess -> the stamped artifact -> compute -> CSV. `exp1` is here
    for `scale_km`; `band_small_boundary` for the band, the boundary
    distance and the stamped `max_distance_km` together; `partial_5m` for
    the barrier weights, which are computed in `preprocess`, stored in the
    artifact, stamped with `buffer_m: 5.0`, and consumed by `compute`; and
    `overlap_outside` for the new REQUIRED key, which has to survive the
    YAML round trip and reach `compute` even though it shapes nothing the
    artifact holds.
    """
```

Nothing else in that test changes.

- [ ] **Step 3: Write the stamp test**

Append to `tests/test_cli.py`, in the stamp section after
`test_changing_only_the_decay_does_not_invalidate_an_artifact`:

```python
def test_changing_only_the_overlap_rule_does_not_invalidate_an_artifact():
    """The overlap rule is applied downstream in `compute` — it changes what
    a neighbour LENDS, never who a neighbour IS — so one stored artifact
    serves both values and nobody has to re-preprocess 4,357 polygons to try
    the switch. The decay precedent, pinned (spec § 3.2, § 12 item 7)."""
    from dataclasses import replace

    from delhi_psi import pipeline
    from delhi_psi.config import OverlapConfig, OverlapLending
    from tests.oraculum_fixtures import oracle_config

    cfg = oracle_config("code-2025")
    frame = _stamped(cfg.methodology)
    other = replace(cfg, methodology=replace(
        cfg.methodology,
        overlap=OverlapConfig(lending=OverlapLending.OUTSIDE_RECEIVER)))
    pipeline.check_methodology_stamp(frame, other)    # must not raise
    assert "overlap" not in pipeline.methodology_stamp(other.methodology)
```

- [ ] **Step 4: Run them**

Run: `uv run pytest -q -W error tests/test_messy_fixtures.py tests/test_cli.py -k "overlap or shared_service or stamp"`

Expected: the messy pins and the stamp test PASS. They are written after
their implementation, so prove they are meaningful: temporarily change
`index_frames`' condition from `== "outside_receiver"` to
`== "whole"`, re-run, confirm
`test_the_overlap_clinic_is_lent_back_today_and_not_under_the_switch` FAILS
(O1's clinic PCEN stays at `(1 + 1/1.8)/600`), then restore the line. Report
both results.

- [ ] **Step 5: Run this task's files, then commit**

Run: `uv run pytest -q -W error tests/test_messy_fixtures.py tests/test_variants_match_reference.py tests/test_cli.py`

Expected: PASS, including the new `overlap_outside` CLI case at 1e-9 against
the variants CSV and the untouched counting pin. Do NOT run the whole suite.

```bash
git add tests/test_messy_fixtures.py tests/test_variants_match_reference.py \
        tests/test_cli.py
git commit -m "$(cat <<'MSG'
test(overlap): the messy pin, the CLI leg and the artifact that stays valid (DEL-20)

The two halves of memo § 6 now sit side by side in test_messy_fixtures: the
ratified counting pin is unchanged (the clinic inside O1 n O2 is 1 for each
owner under every value), and the new lending pin shows the same clinic
reaching O1 twice under code-2025 and once under outside_receiver. Nothing
else on the city moves — O2's school is outside the overlap and is lent in
full, and every settlement without an overlapping neighbour is bit-identical.

overlap_outside joins the derived-variant round trip, so the new REQUIRED
key is proven through YAML -> load_config -> preprocess -> compute -> CSV;
and changing only overlap.lending does NOT invalidate a stored artifact,
because the rule changes what a neighbour lends, not who a neighbour is.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 7: Production == reference on synthetic overlap geometry

**Files:**
- Test: `tests/test_reference_impl.py` (two new tests at the end, reusing
  `synthetic_partial_city()` at :525-580)

**Interfaces:**
- Consumes: `tests/test_reference_impl.py::synthetic_partial_city()` (already in that file,
  from Group B), `pipeline.compute_frames`, `config.MethodologyConfig` and
  its sub-configs built directly (no YAML), `reference_impl.compute_city(
  barrier_rule="partial_weighted", barrier_buffer_m=5.0,
  overlap_lending="outside_receiver", ...)`, and `METRIC_MAP` from
  `tests/test_profiles_match_reference.py`.
- Produces: `reference_impl` test-module helper
  `synthetic_overlap_city()`. No production names.

**Why a second city helper rather than an edit.** Group B's
`synthetic_partial_city()` deliberately places every service point outside
the P/Q overlap, and its two tests pin that shape. This task adds a clinic
INSIDE the overlap in a derived helper, so the existing tests keep testing
what they were written to test and the new one gets the case it needs: a
fractional barrier weight AND a shared service on the same city, scored by
both implementations.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reference_impl.py`:

```python
# --- 3E: production == reference with the overlap rule ON (spec § 6.4) --
def synthetic_overlap_city():
    """`synthetic_partial_city()` with ONE clinic added inside the P/Q
    overlap, at (1100, 500) — strictly interior to both, so production's
    boundary-inclusive `intersects` and the reference's strict `within`
    agree on it (rule-set gap #6 stays out of scope).

    That one point is what makes the overlap rule bite: P and Q each own it,
    and each must stop lending it to the other. The road already crosses the
    overlap (100 m of it lies in Q as well as P), so the LINE branch of the
    shared structure is exercised in the same run. Clinic counts become
    P 2, Q 2, R 1 — two distinct values, so no column is constant
    and DEL-54's guard cannot fire.
    """
    from shapely.geometry import Point

    settlements, barriers, services = synthetic_partial_city()
    clinics = services["clinic"]
    services = dict(services)
    services["clinic"] = gpd.GeoDataFrame(
        {"service": ["clinic"] * (len(clinics) + 1)},
        geometry=[*clinics.geometry, Point(1100, 500)], crs="EPSG:7760")
    return settlements, barriers, services


def test_production_matches_the_reference_with_both_3e_rules_on():
    """The fractional weight x overlap-lending x MultiPolygon case, scored
    by BOTH implementations at 1e-12. It is the only place the two 3E
    multipliers are simultaneously non-trivial: no fixture city has a
    barrier across an overlap, and adding one would move an existing
    expected value (spec § 12 item 3).
    """
    from delhi_psi.config import (
        AbsentNeighbor, AdjacencyConfig, AdjacencyRule, BarrierConfig,
        BarrierRule, DecayConfig, DecayDistance, DecayForm, ExclusionConfig,
        ExclusionStage, MethodologyConfig, OverlapConfig, OverlapLending,
        RoadsFormula,
    )
    from delhi_psi.pipeline import compute_frames
    from tests.test_profiles_match_reference import METRIC_MAP

    settlements, barriers, services = synthetic_overlap_city()
    methodology = MethodologyConfig(
        adjacency=AdjacencyConfig(rule=AdjacencyRule.BBOX),
        barrier=BarrierConfig(rule=BarrierRule.PARTIAL_WEIGHTED,
                              combine="any", buffer_m=5.0),
        overlap=OverlapConfig(lending=OverlapLending.OUTSIDE_RECEIVER),
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
            overlap_lending="outside_receiver",
            roads_formula="decayed", scenario="none", denom=denom,
            second_norm=True, absent_neighbor_contribution="swallowed",
            scenarios={"none": (frozenset(), False)})
        assert set(got.index) == set(exp.index)
        for prod_col, metric in METRIC_MAP.items():
            for sid in exp.index:
                assert got.loc[sid, prod_col] == pytest.approx(
                    exp.loc[sid, metric], abs=1e-12), (denom, sid, prod_col)


def test_the_synthetic_city_really_shares_a_point_and_a_road():
    """The test above would still pass if the shared structure were empty —
    that is exactly the failure mode it exists to rule out. Both branches:
    one clinic inside P n Q, and 100 m of the road inside it too."""
    from tests.reference_impl import adjacency, apply_barrier, shared_amounts

    settlements, barriers, services = synthetic_overlap_city()
    nbrs = apply_barrier(adjacency(settlements, "bbox"), settlements,
                         barriers, "partial_weighted", 5.0)
    got = shared_amounts(nbrs, settlements, services)
    assert got["clinic"] == {("P", "Q"): 1, ("Q", "P"): 1}
    assert got["road"][("P", "Q")] == pytest.approx(0.1, abs=1e-12)
    assert got["road"][("Q", "P")] == got["road"][("P", "Q")]
    assert got["school"] == {}          # every other service is clean
```

`gpd` is imported locally inside the tests in this file, so
`synthetic_overlap_city` needs `import geopandas as gpd` at its top —
follow the file's local-import style:

```python
def synthetic_overlap_city():
    """..."""
    import geopandas as gpd
    from shapely.geometry import Point
```

- [ ] **Step 2: Run them and record the result**

Run: `uv run pytest -q -W error tests/test_reference_impl.py -k "both_3e_rules or really_shares"`

Expected: **PASS immediately** — every piece was built in Tasks 1-4, and
this test's job is to catch a disagreement between them, not to drive new
code. Because it is written after its implementation, prove it is
meaningful: temporarily change the `overlap_lending=` argument in the
`compute_city` call to `"whole"`, re-run, confirm
`test_production_matches_the_reference_with_both_3e_rules_on` FAILS on
`clinic_pcen` for P and Q, then restore it. Report both results.

If it fails as written, the two implementations disagree on the rule. That
is a real divergence: **report it, do not tune either side** until you have
found which one departs from spec § 3.1.

- [ ] **Step 3: Confirm no service column is constant**

Run: `uv run pytest -q -W error tests/test_reference_impl.py`

Expected: PASS. DEL-54's guard raises on a constant column on both sides, so
a green run of the whole file is also the proof that adding the fourth
clinic did not flatten anything.

- [ ] **Step 4: Commit**

```bash
git add tests/test_reference_impl.py
git commit -m "$(cat <<'MSG'
test(overlap): production == reference with both 3E rules on (DEL-20)

The synthetic city gains one clinic inside the P/Q overlap, so the overlap
rule bites on the same pair whose shared boundary a canal partly covers —
the only place the two 3E multipliers are simultaneously non-trivial, since
no fixture city has a barrier across an overlap and adding one would move an
existing expected value. Both branches of the shared structure are
exercised: the added point, and the 100 m of road already inside P n Q.

Every METRIC_MAP column, both denominators, both implementations, 1e-12.
Group B's synthetic_partial_city is left exactly as it was, so its own pins
keep testing the shape they were written for.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 8: Docs — config doc, messy-city doc, CHANGELOG, WORKPLAN

**Files:**
- Modify: `docs/methodology-config.md` (§ 1 table and the two paragraphs
  under it; § 4 proof list; the § 6 example profile)
- Modify: `docs/oracle/messy-city.md` ("What is deliberately NOT here", and
  the `O2` row)
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `WORKPLAN.md` (cycle 3E item (b); DEL-31's blocker list)

**Interfaces:**
- Consumes: the numbers pinned in Tasks 1, 5 and 6. Every number written
  here must already be asserted by a test — nothing new is derived in prose.
  The real-layer numbers are Task 10's; do not anticipate them.

- [ ] **Step 1: The config doc, § 1**

In `docs/methodology-config.md`, add a row to the switch table immediately
after the `barrier.buffer_m` row:

```markdown
| `overlap.lending` | `whole` | `whole` | **Raj to answer** — Bob's addition of 5 Sep 2026; `outside_receiver` is implemented and proven, `whole` is what both profiles carry until he confirms | DEL-20 — what a NEIGHBOUR lends. `whole` is today's rule, the neighbour's whole amount S_j. `outside_receiver` lends \|S_j \ S_i\|: the amount minus whatever of the same service already lies inside the receiver, so a service in the overlap of two colonies is not counted twice for one of them. Clean pairs are unaffected — for every pair whose polygons do not overlap, \|S_j \ S_i\| == \|S_j\| exactly |
```

Then rewrite the paragraph under the table that currently reads "One thing
Bob added needs **code, not config**: …":

```markdown
Two things Raj ratified need **no switch**: the min-max in Eq. 2 runs over
the reported settlements only (today's behaviour), and a service inside
overlapping colony polygons counts directly for each of them (today's
behaviour). The other half of that second ruling — what such a service is
worth when it is *lent* — is now the switch `overlap.lending`, landed in
cycle 3E (DEL-20), and it stays at `whole` until Raj confirms: he agreed
that an overlap service counts for each owner, but the lending half is
Bob's proposal of 5 Sep 2026 and is still out for his answer. DEL-31
decides what the ratified profile carries.
```

And in the **Reserved** paragraph, add the new key alongside
`exclusion.minmax_universe`:

```markdown
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
```

- [ ] **Step 2: The config doc, § 4 and § 6**

In § 4, extend the `tests/test_variant_rules.py` bullet — after the
`partial_5m` sentence, add:

```markdown
  and **`overlap_outside`** pins the lending rule on the messy city's one
  overlapping pair (O1's clinic PCEN falls from (1 + 1/1.8)/600 to 1/600
  while its school PCEN does not move at all, because O2's school is outside
  the overlap), plus the statement that the shared structure is EMPTY on
  Oraculum, which is why the rule is degenerate there.
```

In the `tests/test_variants_match_reference.py` bullet, change "all nine
derived variants" to **eleven** and add `overlap_outside` to the list of
CLI round-trip cases.

In the `tests/test_messy_fixtures.py` bullet, change "A pin here flips when
DEL-19/DEL-20 land; that is the point" to:

```markdown
  A pin here flips when DEL-19 lands; that is the point. DEL-20's two halves
  now sit side by side there — the ratified counting pin, which does not
  move, and the lending pin under the `overlap_outside` methodology.
```

In the `tests/test_reference_impl.py` bullet, extend the synthetic-city
sentence to name the overlap case:

```markdown
- `tests/test_reference_impl.py` — the synthetic production-vs-reference
  city: a fractional barrier weight on an overlapping pair with a
  MultiPolygon neighbour, and a second variant of the same city with a
  service point inside the overlap, so both 3E multipliers act on one pair —
  scored by BOTH implementations at 1e-12. Neither can live in a fixture
  city without moving an existing expected value, so they live in in-test
  geometry.
```

In § 6's `band-1km.yaml` example profile, add the block a reader copying it
now needs, between `barrier:` and `decay:`:

```yaml
  overlap:
    lending: whole
```

- [ ] **Step 3: The messy-city doc**

In `docs/oracle/messy-city.md`, rewrite the "Any rule change" bullet under
"What is deliberately NOT here":

```markdown
- **Any rule change to the base profiles.** Edge-only adjacency is DEL-19,
  after Raj. `partial_weighted` (DEL-48) and `overlap.lending:
  outside_receiver` (DEL-20) landed in cycle 3E as VARIANTS, so this city's
  `expected_values.csv` and `production/*.csv` still record TODAY: the two
  new rules appear only as extra rule blocks in
  `variants_expected_values.csv`, and the lending pin lives in
  `tests/test_messy_fixtures.py` beside the counting pin it must not move.
  Barriers are still absent — the fixture's barrier file is shared by every
  rule-set, so any barrier touching a settlement with a `bbox` neighbour
  would change an existing expected value (spec 3E § 12 item 3).
```

and extend the `O2` row of the settlements table:

```markdown
| `O2` | Planned | 700 | 1.0 | rectangle | one clinic strictly inside the overlap is counted for **both** (DEL-20, ratified — that pin does not move); under the `overlap_outside` variant O2 lends O1 none of it, because it is already inside O1, while O2's own school at (11400, 500) is outside the overlap and is lent in full; the pair are `touch` neighbours because an overlap polygon's `.length` is its perimeter (DEL-19) |
```

- [ ] **Step 4: CHANGELOG**

At the TOP of `[Unreleased]` in `CHANGELOG.md`, above the DEL-48 entry, add
one entry covering: the new required switch `methodology.overlap.lending`
(`whole` | `outside_receiver`) and what `outside_receiver` means
(`|S_j \ S_i|`); that both shipped profiles gain the key with today's value
`whole` and **nothing either profile computes moves** — both cities'
`expected_values.csv` and every `production/*.csv` byte-identical, both
variants CSVs changed by addition only (322 new rows per rule on oraculum,
460 on messy, zero deletions); that Raj ratified the COUNTING half only, so
`overlap.counting` is a reserved key and the lending half awaits his answer;
that the structure is sparse — built from service containment, so it costs
nothing on a clean pair; that the rule is NOT in the methodology stamp, so
one artifact serves both values; and that this is the third and last of
cycle 3E's per-ticket PRs, after DEL-54 and DEL-48.

- [ ] **Step 5: WORKPLAN**

In `WORKPLAN.md`'s cycle 3E item, tick item (b) in the style the two
completed items above it use, naming the branch (`del-20-overlap-lending`)
and the pin (O1's clinic PCEN 1/600 under `outside_receiver` against
(1 + 1/1.8)/600 today), and stating that it ships as the `overlap_outside`
VARIANT with `whole` in both shipped profiles, so no existing expected value
moved. Update DEL-31's blocker list: DEL-20 is no longer a blocker; Raj's
two DEL-52 answers and his answer on the lending half remain. Update the
phase-3 status line if the cycle is now complete on all three tickets.

- [ ] **Step 6: Prove the docs' own tests still hold**

Run: `uv run pytest -q -W error tests/test_messy_fixtures.py tests/test_manuscript_anchors.py`

Expected: PASS.
`test_the_messy_city_doc_documents_every_settlement` and
`test_methodology_config_section_4_says_the_proofs_run_on_both_cities` read
the two docs edited here; the manuscript anchors prove the ratified
worksheet was not touched (this ticket adds no worksheet section — Oraculum
has no overlaps and therefore no overlap anchor to derive).

- [ ] **Step 7: Commit**

```bash
git add docs/methodology-config.md docs/oracle/messy-city.md \
        CHANGELOG.md WORKPLAN.md
git commit -m "$(cat <<'MSG'
docs(overlap): the lending switch, and which half of memo § 6 is Raj's (DEL-20)

The config doc gains an overlap.lending row, rewrites "one thing Bob added
needs code, not config" into the switch that now exists, and adds
overlap.counting to the reserved paragraph with its reason — that is where
the next reader learns why only one half of the memo section is a knob. The
§ 6 sweep profile gains the block a reader copying it now needs.

The messy-city doc records that DEL-48 and DEL-20 landed as VARIANTS, so its
committed numbers still record today, and that the lending pin sits beside
the counting pin it must not move.

Every number in prose is one a test already asserts; the real-layer numbers
are the run step's.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 9: `scripts/measure_rule_effects.py` — the `overlap_lending` block

**Files:**
- Modify: `scripts/measure_rule_effects.py` (module docstring; `measure_effect`
  :134-169; new helpers; `main` :208-238)
- Modify: `docs/data/rule_effects.md` (a new section; the fenced block is
  pasted by Task 10)
- Modify: `tests/test_measure_rule_effects.py` (`committed_block` :114-119
  and the three document tests; new fixture-level tests)

**Interfaces:**
- Consumes: `scripts/_measure_common.{resolve_work_dir, render, parse_block,
  FENCE}`, `scripts.measure_roads_access.stage_artifacts`, and this script's
  own `derived_profile` / `measure_effect` (both already there from Group B).
  Consumes from Task 3: `index.shared_amounts`.
- Produces: `measure_rule_effects.{OVERLAP_PROFILE, BLOCKS, service_layers,
  shared_pair_counts, overlapping_neighbours, pcen_changes,
  measure_overlap_lending}`, and `main(["--only", "<block>"])`.

**Note on the artifact.** Unlike the barrier block, this one **does not
change the stamp**, so it stages the PROVEN artifact and runs `compute`
alone — the roads-measurement pattern. The script asserts that before
running, so a future change that puts `overlap` in the stamp fails loudly
instead of silently describing a neighbourhood nobody built.

- [ ] **Step 1: Write the failing fixture-level tests**

In `tests/test_measure_rule_effects.py`, first generalise the document
helper — there are two blocks now, and the overlap one does not exist until
Task 10 pastes it:

```python
def committed_block(name="partial_barriers"):
    text = DOC.read_text() if DOC.exists() else ""
    if FENCE not in text or f"block: {name}" not in text:
        pytest.skip(f"{DOC} carries no measured `{name}` block yet — the run "
                    "step pastes it")
    return parse_block(text, name=name)


def committed_blocks():
    """Every block the document actually carries, for the prose-number
    guard: it must see all of them, or a number quoted from the second
    block reads as an invention."""
    text = DOC.read_text() if DOC.exists() else ""
    return [parse_block(text, name=name) for name in BLOCKS
            if f"block: {name}" in text]
```

and point the provenance test at it:

```python
def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, committed_blocks())
```

Extend the import at the top of the file:

```python
from scripts.measure_rule_effects import (
    BLOCKS, WEIGHT_CLASSES, derived_profile, overlapping_neighbours,
    pcen_changes, shared_pair_counts, weight_classes,
)
```

Then append the new tests:

```python
# --- the overlap block (DEL-20) ----------------------------------------
def _overlap_frame():
    """Two overlapping settlements and a disjoint third, in the shape a
    stored neighbours artifact has: ids, lists, geometry and NO amount
    columns — `shared_pair_counts` computes those itself, and a frame that
    already carried them would make `point_counts`' merge add suffixes."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"USO_AREA_U": ["P", "Q", "Z"],
         "nbrs_bbox": [["Q"], ["P"], []]},
        geometry=[box(0, 0, 1200, 1000), box(1000, 0, 2000, 1000),
                  box(5000, 0, 6000, 1000)], crs="EPSG:7760")


def test_overlapping_neighbours_finds_only_the_positive_area_pair():
    """The overlap rule cannot move any other settlement's PCEN, so this set
    is the containment bound the run is checked against."""
    assert overlapping_neighbours(_overlap_frame()) == {"P", "Q"}


def test_shared_pair_counts_counts_ordered_entries_per_service():
    """The size of the sparse structure, per service — one physical clinic
    inside both P and Q makes TWO ordered entries."""
    import geopandas as gpd
    from shapely.geometry import Point

    clinics = gpd.GeoDataFrame(
        {"service": ["clinic", "clinic"]},
        geometry=[Point(1100, 500), Point(600, 500)], crs="EPSG:7760")
    got = shared_pair_counts(_overlap_frame(), {"clinic": clinics},
                             point_names=("clinic",))
    assert got == {"shared_pairs_clinic": 2, "shared_pairs_total": 2}


def test_pcen_changes_counts_the_fall_and_reports_a_rise():
    """Lending is only ever REDUCED, so a risen PCEN is a bug, not a
    finding: `settlements_pcen_rose` must be 0 in the run."""
    before = pd.DataFrame({"USO_AREA_U": ["P", "Q", "Z"],
                           "clinic_pcen": [1.0, 2.0, 3.0]})
    after = pd.DataFrame({"USO_AREA_U": ["P", "Q", "Z"],
                          "clinic_pcen": [0.5, 2.0, 3.5]})
    report, changed = pcen_changes(before, after, id_col="USO_AREA_U")
    assert report == {"settlements_pcen_changed": 2,
                      "settlements_pcen_rose": 1}
    assert changed == {"P", "Z"}


def test_the_oraculum_shared_structure_is_empty(data_dir, tmp_path):  # noqa: F811
    """Oraculum has no overlapping polygons and no point inside two
    settlements, so every count is 0 — which is exactly why overlap_outside
    is degenerate there. The fixture proves the plumbing; the number that
    matters is the real layer's, and it comes from the run step."""
    from delhi_psi.config import load_config

    from scripts.measure_rule_effects import service_layers

    profile = oracle_profile_path("code-2025", tmp_path)
    cfg = load_config(profile, data_dir=str(data_dir))
    run_dir = tmp_path / "overlap"
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0
    from delhi_psi import io
    frame = io.read_neighbors(run_dir / cfg.paths.neighbors_artifact)
    got = shared_pair_counts(frame, service_layers(cfg),
                             point_names=tuple(cfg.services.point))
    assert got["shared_pairs_total"] == 0, got
    assert overlapping_neighbours(frame) == set()


@needs_measure_cache
def test_a_fresh_overlap_run_reproduces_the_committed_block():
    """The real-data drift check for the second block. It stages the proven
    artifact and runs `compute` only, so it is far cheaper than the barrier
    block's — but it still needs the layers, hence the cache gate."""
    block = committed_block("overlap_lending")
    proc = subprocess.run(
        [sys.executable, "scripts/measure_rule_effects.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR), "--work-dir", MEASURE_CACHE,
         "--only", "overlap_lending"],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="overlap_lending") == block
```

and add `--only partial_barriers` to the existing barrier drift test's
argument list, so it does not start running the overlap block too:

```python
        [sys.executable, "scripts/measure_rule_effects.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR), "--work-dir", MEASURE_CACHE,
         "--only", "partial_barriers"],
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_measure_rule_effects.py`

Expected: **collection error —
`ImportError: cannot import name 'BLOCKS' from 'scripts.measure_rule_effects'`.**
That is the RED reason. Record it.

- [ ] **Step 3: Add the overlap block to the script**

In `scripts/measure_rule_effects.py`, extend the module docstring's block
list:

```
  overlap_lending   `code-2025` with ONE thing changed —
                    `methodology.overlap`: {lending: outside_receiver} — so
                    the diff is attributable to the lending rule alone.
                    Unlike the barrier block this one REUSES the proven
                    neighbours artifact: the overlap rule is downstream of
                    the neighbour structure and is not in the methodology
                    stamp, which the script asserts before staging it.
                    Reports how many ordered (pair, service) entries the
                    sparse shared structure holds per service, how many
                    settlements have an overlapping neighbour at all, how
                    many settlements' PCEN moved and in which direction, and
                    the per-type PSI shift against the proven `code-2025`
                    outputs read from --verify-dir.
```

Extend the imports and constants:

```python
from delhi_psi import geometry, index, io, pipeline
from delhi_psi.config import PROFILES_DIR, load_config
from delhi_psi.pipeline import ID_COL, NBRS_COL, NBRS_WEIGHT_COL, TYPE_COL
from scripts._measure_common import render, resolve_work_dir
from scripts.measure_roads_access import stage_artifacts
```

```python
PARTIAL_PROFILE = "partial-barriers-5m"
OVERLAP_PROFILE = "overlap-outside-receiver"
BLOCKS = ("partial_barriers", "overlap_lending")
BUFFER_M = 5
PSI_COL = "unnorm_psi"
PCEN_SUFFIX = "_pcen"
WEIGHT_CLASSES = ("links_w_one", "links_fractional", "links_severed")
```

Generalise `measure_effect`'s "after" label so the second block does not
call its own run "partial" — the default keeps the barrier block's committed
key names byte-identical:

```python
def measure_effect(before, after, *, denom, label="partial", id_col=ID_COL,
                   type_col=TYPE_COL, types=REPORTED_TYPES):
```

and inside it, replace the two `psi_partial_` / `norm_partial_` literals:

```python
        report[f"psi_{label}_{denom}_{name}"] = _mean(right.loc[rows, PSI_COL])
```
```python
            report[f"norm_{label}_{denom}_{name}"] = _mean(
                right.loc[rows, "norm_psi"])
```

Then add the block's own helpers, after `measure_partial_barriers`:

```python
def service_layers(cfg):
    """The service layers `compute` loads, deduplicated and reprojected the
    way `index_frames` reprojects them — so the counts below describe the
    frames the run actually scored, not the files on disk."""
    data_dir = cfg.paths.data_dir
    out = {}
    for name, path in {**cfg.services.point, **cfg.services.line}.items():
        gdf = io.read_layer(data_dir / path).drop_duplicates().reset_index(
            drop=True)
        out[name] = geometry.reproject(gdf, cfg.crs.epsg)
    return out


def shared_pair_counts(frame, layers, *, point_names, id_col=ID_COL,
                       neighbor_col=NBRS_COL):
    """{'shared_pairs_<svc>': n} — the number of ORDERED (i, j) entries
    `index.shared_amounts` builds per service, plus the total.

    Built with the SAME function `index_frames` calls, on the same frames,
    so these are the real structure's sizes and not an estimate. It costs
    one extra pass of the amount helpers, because the line branch skips
    pairs where either side owns none of the service and those amounts are
    not in the stored artifact.
    """
    amounts = frame
    kinds = {name: ("point" if name in point_names else "line")
             for name in layers}
    for name, gdf in layers.items():
        column = index.service_amount_column(name, kinds[name])
        if kinds[name] == "point":
            amounts = index.point_counts(amounts, gdf, count_col=column,
                                         id_col=id_col)
        else:
            amounts = index.road_lengths(amounts, gdf, length_col=column,
                                         id_col=id_col)
    report = {}
    for name, gdf in layers.items():
        table = index.shared_amounts(
            amounts, gdf, kind=kinds[name],
            amount_col=index.service_amount_column(name, kinds[name]),
            neighbor_col=neighbor_col, id_col=id_col)
        report[f"shared_pairs_{name}"] = len(table)
    report["shared_pairs_total"] = sum(report.values())
    return report


def overlapping_neighbours(frame, *, id_col=ID_COL, neighbor_col=NBRS_COL):
    """Ids with at least one STORED neighbour whose polygon overlaps theirs
    (positive-area intersection). The lending rule cannot move any other
    settlement's PCEN, so the changed set must be a subset of this one —
    which is the containment bound stated before the run."""
    geoms = frame.set_index(id_col).geometry
    out = set()
    for _, row in frame.iterrows():
        i = row[id_col]
        for j in row[neighbor_col]:
            if j in geoms.index and geoms[i].intersection(geoms[j]).area > 0:
                out.add(i)
                break
    return out


def pcen_changes(before, after, *, id_col=ID_COL):
    """(report, changed ids) — how the PCEN columns moved, and which way.

    Lending is only ever REDUCED (|S_j \\ S_i| <= |S_j|), so a RISEN PCEN is
    a bug and not a finding: `settlements_pcen_rose` must be 0.
    """
    left = before.set_index(id_col)
    right = after.set_index(id_col).reindex(left.index)
    columns = [c for c in left.columns if c.endswith(PCEN_SUFFIX)]
    changed, rose = set(), set()
    for column in columns:
        diff = right[column] - left[column]
        changed |= set(left.index[diff != 0])
        rose |= set(left.index[diff > 0])
    return ({"settlements_pcen_changed": len(changed),
             "settlements_pcen_rose": len(rose)}, changed)


def measure_overlap_lending(cfg, work_dir, *, base, verify_dir):
    """Block `overlap_lending`: the SAME neighbours, one methodology value
    changed."""
    stamp = pipeline.methodology_stamp(cfg.methodology)
    # BOTH shapes, like measure_roads_access's guard: a new concern is most
    # likely to arrive as a new TOP-LEVEL block, which a values() scan alone
    # would walk straight past.
    if "overlap" in stamp or any("overlap" in block for block in stamp.values()):
        raise SystemExit(
            "pipeline.methodology_stamp now carries `overlap`: the neighbours "
            "artifact would have to be rebuilt and this block's one-factor "
            "run cannot reuse --verify-dir's. Re-read spec § 3.2 before "
            "changing anything.")

    run_dir = Path(work_dir) / OVERLAP_PROFILE
    profile_path = derived_profile(
        base, run_dir, profile_name=OVERLAP_PROFILE,
        methodology={"overlap": {"lending": "outside_receiver"}})
    run_cfg = load_config(profile_path, data_dir=str(cfg.paths.data_dir),
                          out_dir=str(run_dir))
    stage_artifacts(verify_dir, run_dir,
                    source_name=cfg.paths.neighbors_artifact,
                    artifact_name=run_cfg.paths.neighbors_artifact)
    pipeline.compute(run_cfg)

    frame = io.read_neighbors(run_dir / run_cfg.paths.neighbors_artifact)
    report = shared_pair_counts(frame, service_layers(cfg),
                                point_names=tuple(cfg.services.point),
                                id_col=cfg.layers.settlements.id_col)
    overlapping = overlapping_neighbours(
        frame, id_col=cfg.layers.settlements.id_col)
    report["settlements_with_an_overlapping_neighbour"] = len(overlapping)

    for denom in DENOMINATORS:
        before = pd.read_csv(
            Path(verify_dir) / f"{pipeline.output_basename(cfg, denom)}.csv")
        after = pd.read_csv(
            run_dir / f"{pipeline.output_basename(run_cfg, denom)}.csv")
        moved, changed = pcen_changes(
            before, after, id_col=cfg.layers.settlements.id_col)
        report.update({f"{key}_{denom}": value
                       for key, value in moved.items()})
        report[f"pcen_changed_outside_the_overlap_set_{denom}"] = len(
            changed - overlapping)
        report.update(measure_effect(before, after, denom=denom,
                                     label="outside",
                                     id_col=cfg.layers.settlements.id_col,
                                     type_col=cfg.layers.settlements.type_col))
    return report
```

Finally, teach `main` to select blocks:

```python
    parser.add_argument("--only", choices=BLOCKS, default=None,
                        help="run ONE block instead of both (the barrier "
                             "block re-runs preprocess and costs minutes; "
                             "the overlap block stages the proven artifact "
                             "and runs compute alone)")
```

```python
    print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
    print(f"verify-dir: {verify_dir}")
    print(f"work-dir: {work_dir}")
    wanted = BLOCKS if args.only is None else (args.only,)
    measures = {"partial_barriers": measure_partial_barriers,
                "overlap_lending": measure_overlap_lending}
    for name in wanted:
        print(render(measures[name](cfg, work_dir, base=args.config,
                                    verify_dir=verify_dir), name=name))
    return 0
```

- [ ] **Step 4: Run the fixture-level tests**

Run: `uv run pytest -q -W error tests/test_measure_rule_effects.py`

Expected: the unit and Oraculum tests PASS; `committed_block("overlap_lending")`
SKIPS the overlap document tests (the doc carries no such block yet); the two
drift tests SKIP unless `DELHI_PSI_MEASURE_CACHE` is set. The barrier block's
own document tests keep PASSING against the block already committed — if one
of them fails, the `--only` addition or the `measure_effect` `label` default
changed a key name, which it must not.

- [ ] **Step 5: Write the document section, unmeasured**

In `docs/data/rule_effects.md`, retitle the document so it covers both rules
(`# What the cycle-3E rule changes do to today's numbers`) and add a new
section after the barrier one. **Leave the fenced block out entirely** —
the document tests skip on its absence and Task 10 pastes it. Do not invent
placeholder numbers inside a fence: `parse_block` would read them as real.

The section must carry its own provenance line (`**Command:** uv run python
scripts/measure_rule_effects.py --config code-2025 --verify-dir
~/delhi_data/phase3_verify --work-dir ~/measure_work/cache --only
overlap_lending`), one bullet per reported key explaining what it means, and:

```markdown
Unlike the barrier block, this one REUSES the proven `code-2025` neighbours
artifact: `overlap.lending` is applied downstream in `compute` and is not in
the methodology stamp, so the stored lists are still the right ones. The
script asserts that before staging, so a future change that puts `overlap`
in the stamp fails loudly instead of quietly describing a neighbourhood
nobody built.

**What the run must show, stated before it runs** (spec § 6.6): no NaN and
no negative anywhere; `settlements_pcen_rose_pop` and
`settlements_pcen_rose_popdensity` both **0**, because lending is only ever
reduced — a risen PCEN is a bug, not a finding;
`pcen_changed_outside_the_overlap_set_*` both **0**, because the rule cannot
move a settlement with no overlapping neighbour;
`shared_pairs_total` greater than 0 and no larger than twice the 429
multi-settlement points `docs/data/layer_pathologies.md` counts (each such
point contributes at most k(k-1) ordered entries, and k is 2 for essentially
all of them), summed over the six point services; and `shared_pairs_road`
consistent with a road network that crosses the 4,069 overlapping pairs. A
result outside these bounds is a stop, not a number to write down.
```

- [ ] **Step 6: Run this task's files, then commit**

Run: `uv run pytest -q -W error tests/test_measure_rule_effects.py tests/test_measure_roads_access.py`

Expected: PASS or SKIP as described in Step 4. `tests/test_measure_roads_access.py`
is included because this task imports `stage_artifacts` from that script for
the first time — a green run proves the import introduced no cycle and no
side effect at import time. Do NOT run the whole suite.

```bash
git add scripts/measure_rule_effects.py docs/data/rule_effects.md \
        tests/test_measure_rule_effects.py
git commit -m "$(cat <<'MSG'
feat(overlap): measure_rule_effects.py gains the overlap_lending block (DEL-20)

code-2025 with only methodology.overlap changed, reported as the size of the
sparse shared structure per service, how many settlements have an
overlapping neighbour at all, how many settlements' PCEN moved and in which
direction, and the per-type PSI shift against the proven outputs read from
--verify-dir.

Unlike the barrier block this one REUSES the proven artifact — the overlap
rule is downstream of the neighbour structure and is not in the stamp — and
the script asserts that before staging, so a future change that stamps
`overlap` fails loudly. `--only` selects one block, so the two drift tests
do not pay for each other's run.

Proven on Oraculum, where the structure is provably empty: no overlapping
polygons and no point inside two settlements, which is exactly why
overlap_outside is degenerate there. The document carries its prose and its
stated bounds; the run step pastes the block.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 10: **CONTROLLER ONLY** — the real-data run (spec § 6.6)

**This is the one task in this plan where placeholders are allowed.** Every
`[from the run]` below marks a number that does not exist until the run
happens. Nowhere else in this plan may contain one. Do not delegate this
task to a subagent: it writes outside the repo, it is data-gated, and its
stop conditions need the owner's judgement.

**Files:**
- Modify: `docs/data/rule_effects.md` (paste the measured block; fill the
  provenance and the prose)
- Modify: `docs/methodology-config.md` (extend § 7 with the overlap result)
- Modify: `docs/decisions/2026-08-28-raj-methodology-decisions.md` (§ 5, one
  sentence)
- Modify: `CHANGELOG.md` (extend the DEL-20 entry with the measured effect)

**Read-only over `~/delhi_data` except `--work-dir` / `--out-dir`, which
must be scratch directories — never the baseline and never `phase3_verify`
itself.** Any write under `~/delhi_data` outside those is a stop-and-ask
(spec § 11); the directory is bisynced to the shared drive.

- [ ] **Step 1: The standing proof — `code-2025` verify at 0.000e+00**

Both shipped profiles gained a REQUIRED key this cycle. `whole` names
today's arithmetic, so nothing on the default path should move — which is
exactly why this is worth running. Run each command separately, backgrounded
to a log (they exceed a foreground timeout), and read each log in full
before continuing:

```bash
mkdir -p ~/measure_work/logs ~/measure_work/del20-verify
uv run delhi-psi preprocess --config code-2025 --data-dir ~/delhi_data \
    --out-dir ~/measure_work/del20-verify > ~/measure_work/logs/del20-pre.log 2>&1
uv run delhi-psi compute    --config code-2025 --data-dir ~/delhi_data \
    --out-dir ~/measure_work/del20-verify > ~/measure_work/logs/del20-comp.log 2>&1
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/measure_work/del20-verify \
    > ~/measure_work/logs/del20-verify.log 2>&1
```

Expected in `del20-verify.log`: **PASS on every comparison at `0.000e+00`**
(30 numeric columns × 2 output sets = 60 comparisons). Keep the WHOLE log,
never a `tail` — a truncated log is how a previous cycle mis-reported this
count for three rounds. A single non-zero deviation is a STOP: it means
adding the required key moved the default path, which this cycle promised it
would not.

Record: PASS line, comparison count.

- [ ] **Step 2: The `overlap_lending` block**

```bash
export DELHI_PSI_MEASURE_CACHE=~/measure_work/cache
uv run python scripts/measure_rule_effects.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3_verify \
    --work-dir ~/measure_work/cache --only overlap_lending \
    > ~/measure_work/logs/del20-overlap.log 2>&1
```

This stages the proven artifact and runs `compute` alone, then recomputes
the amounts once to size the shared structure — minutes, not the barrier
block's quarter of an hour. Read the whole log.

- [ ] **Step 3: Check the run against its stated bounds BEFORE writing anything down**

From the printed block, confirm all of:

- `settlements_pcen_rose_pop` == 0 **and**
  `settlements_pcen_rose_popdensity` == 0. Lending is only ever reduced, so
  a single risen PCEN is a bug in the subtraction, not a finding.
- `pcen_changed_outside_the_overlap_set_pop` == 0 **and**
  `..._popdensity` == 0. A settlement with no overlapping neighbour cannot
  be touched by this rule.
- `shared_pairs_total` > 0, and the six point services' counts summed are
  ≤ 2 × 429 (`docs/data/layer_pathologies.md`'s multi-settlement points; each
  contributes k(k-1) ordered entries, k = 2 for essentially all). A count
  above that means a point is inside three or more settlements, which is a
  finding about the layer worth stating, not an error — check
  `layer_pathologies.md` before deciding which it is.
- `settlements_pcen_changed_*` > 0 and ≤
  `settlements_with_an_overlapping_neighbour`.
- No NaN in any `psi_*` or `norm_*` value (`_mean` prints `nan` for an empty
  selection — an empty reported type is a finding).
- `compute` ran clean: `validate.check_no_negative` passed, DEL-54's min-max
  guard did not fire, and `pcen`'s "would lend" guard did not fire. All
  three surface as a non-zero exit; the exit was 0, so say so.

**A result outside these bounds is a STOP** (spec § 6.6, § 11): report it to
the owner, do not write it into the doc as if it were expected.

- [ ] **Step 4: Paste the block and finish the document**

In `docs/data/rule_effects.md`, paste the script's fenced block **verbatim**
— the drift test parses the committed document and the script's stdout with
the same parser, so a hand-edited digit fails the build. Then fill the
overlap section's provenance:

```markdown
- **Run date:** [from the run]
- **Inputs:** settlement layer `uso_update_sep2021`, the seven service layers named by `code-2025`, and the proven `code-2025` run in `~/delhi_data/phase3_verify` (its neighbours artifact, staged unchanged, and its two output CSVs, read only)
- **Commit:** [from the run — the commit this branch is at]
- **Command:** `uv run python scripts/measure_rule_effects.py --config code-2025 --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache --only overlap_lending`
```

Write the prose around the block: how many ordered (pair, service) entries
the structure holds and for which services (`[from the run]`), how many
settlements have an overlapping neighbour at all (`[from the run]`), how many
settlements' PCEN fell (`[from the run]`), and the per-type PSI shift in one
sentence — naming, if the run shows one, whether the shift is large enough
to matter to a figure. Every number in `backticks` must be a block value
verbatim; percentages and other derived quantities are written with a `%`
sign or without backticks, because
`assert_prose_numbers_come_from_the_blocks` enforces exactly that.

Say plainly what the number means for Raj's open question: the lending rule
removes a double count that exists **today, under `bbox`** and would exist
under `touch` as well; the measured effect is `[from the run]`; the decision
is his.

- [ ] **Step 5: Config doc § 7 and the decision log**

Extend `docs/methodology-config.md` § 7 (added by DEL-48) so it covers both
3E rules — retitle it "Partial barriers and overlap lending: what the real
layer showed" and add a paragraph with the overlap run's headline numbers
(`[from the run]`) and the sentence that the structure is sparse: only pairs
that actually share a service point or road metres are ever computed, which
is why the rule costs nothing on the 99%+ of pairs that do not overlap.

In `docs/decisions/2026-08-28-raj-methodology-decisions.md` § 5, add ONE
sentence to the "Added (Bob, 5 Sep 2026; to confirm with Raj)" paragraph for
the batched reply: the lending rule is implemented, proven on both sides and
measured; on the real layer it changes `[from the run]` settlements' PCEN,
all of them downward, out of `[from the run]` with an overlapping neighbour;
the ratified profile (DEL-31) carries it only if Raj confirms. **The reply to
Raj is drafted, never sent** (spec § 11).

- [ ] **Step 6: Run the drift tests with the cache exported**

```bash
DELHI_PSI_MEASURE_CACHE=~/measure_work/cache \
    uv run pytest -q -W error tests/test_measure_rule_effects.py
```

Expected: PASS, with `test_a_fresh_overlap_run_reproduces_the_committed_block`
and every document test now RUNNING rather than skipping. If the fresh run
does not reproduce the committed block byte for byte, the block was
hand-edited or the run is not deterministic — find out which before
committing.

- [ ] **Step 7: The full suite, then commit**

The controller — not a subagent — runs the whole suite once here, as ONE
Bash call with an explicit `timeout` of **600000 ms** and `run_in_background`
NOT set. Read the summary line and quote it in the report.

Run: `uv run pytest -q -W error`

Expected: all pass (the count at `0dcca15` plus roughly 30 added across this
branch).

```bash
git add docs/data/rule_effects.md docs/methodology-config.md \
        docs/decisions/2026-08-28-raj-methodology-decisions.md CHANGELOG.md
git commit -m "$(cat <<'MSG'
docs(overlap): the real-layer effect of outside_receiver (DEL-20)

code-2025 verifies against the July 2025 baseline at 0.000e+00 on every
comparison after this branch: adding a REQUIRED key to both shipped profiles
moved nothing, which is what `whole` naming today's arithmetic has to mean.

The overlap_lending block measures the one-factor effect — the size of the
sparse shared structure per service, the settlements with an overlapping
neighbour, how many settlements' PCEN fell, and the per-type PSI shift —
inside the bounds stated before the run: nothing rose, and nothing outside
the overlap set moved.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

- [ ] **Step 8: Jira**

Move DEL-20 to Done **for the code**, with an evidence comment: the messy
pin (O1's clinic PCEN 1/600 under `outside_receiver` against
(1 + 1/1.8)/600 today, with the counting pin unmoved), production ==
reference at 1e-12 on `overlap_outside` and `partial_5m_outside` for both
cities and both denominators, the real-data verify at 0.000e+00, and the
`overlap_lending` block's headline numbers. State explicitly that **both
shipped profiles carry `whole`** and that the ratified profile's value
awaits Raj's answer on the lending half — DEL-20 is done as code, not as a
decision. Note that cycle 3E is now complete (DEL-54, DEL-48, DEL-20) and
that DEL-31 is unblocked except for Raj's two DEL-52 answers and this one.

---

## Self-review

**1. Spec coverage.**

| spec | task |
|---|---|
| § 1 config surface: `overlap.lending` enum, `OverlapLending`, `OverlapConfig`, `MethodologyConfig.overlap`, `_reject_unknown` at both levels, `RESERVED_KEYS["methodology.overlap.counting"]`, the key required in both shipped profiles AND `MINIMAL` | Task 2 |
| § 3.1 definition: `|S_j \ S_i|`, points via each side's own predicate, roads via clipped length in the intersection, symmetry | Tasks 1 Step 3, 3 Step 3 |
| § 3.2 the data structure: `index.shared_amounts` signature and both branches, built in `index_frames`, compute-locally, only under `outside_receiver`, on the full universe, one dict per service, nothing stored, NOT in the stamp | Tasks 3 Step 3, 4 Step 3; pinned in Tasks 4 and 6 |
| § 3.3 `pcen`: `lent = nbr_count - shared.get(...)`, the sparse 0, the non-negative assertion | Task 3 Steps 4-5 |
| § 3.4 reference: `compute_city(overlap_lending=…)`, `within` for points, clipped length for roads, unknown value → ValueError | Task 1 |
| § 3.5 what is NOT changed: own counts; `test_the_overlap_clinic_is_counted_for_both_owners` unmoved; boundary points (gap #6) untouched | Global Constraints; Tasks 1 Step 1, 6 Step 1 |
| § 5 variants table: `overlap_outside`, `partial_5m_outside`, `VARIANT_KNOBS`, `variant_methodology`, `enum_key`, `knob_for_key`; `EXTRA_PARAMS` deliberately untouched | Tasks 1 Step 7, 2 Step 7, 5 |
| § 6.2 the messy pin, reference side then CSV then production directly, with the today-pin beside the switch-pin | Tasks 1 Step 1, 5, 6 Step 1 |
| § 6.4 item (b): `shared_amounts` on three squares with a clinic in P∩Q, one in P only, a road through the overlap; `pcen` with the structure; the composition test with `nbr_weight_col`; production == reference on synthetic geometry | Tasks 3 Step 1, 7 |
| § 6.5 variants CSV addition-only, the CLI leg, the artifact-stays-valid test, byte-identity at every task | Tasks 2 Step 9, 5 Step 5, 6 |
| § 6.6 real data, overlap part | Task 10 |
| § 7 docs: config doc, messy-city doc, `measure_rule_effects.py` overlap block + `rule_effects.md`, CHANGELOG, WORKPLAN, Jira; decision log § 5 | Tasks 8, 9, 10 |
| § 8 Group C file list | all tasks. Three files the spec's list does not name are touched with their reasons stated: `tests/test_pipeline.py` (Task 4 — the wiring needs a pin somewhere, and that is where the 3E pipeline pins live), `tests/test_variants_match_reference.py` (Task 6 — the CLI leg), and `tests/test_measure_roads_access.py` is NOT modified, only run (Task 9 Step 6) because `stage_artifacts` is imported from its script for the first time |

Group B items (`barrier.rule`, `buffer_m`, `neighbors.*`,
`reference_impl.partial_weights`) are deliberately absent — see Global
Constraints. There is no `docs/oracle/derivation-worksheet.md` section in
this ticket, and that is deliberate: Oraculum has no overlapping polygons
and no multi-settlement point, so there is no overlap anchor to hand-derive
there (verified: `overlapping pairs: []`, no point inside two settlements).
The hand-checkable pin lives on the messy city instead, which is where the
geometry is, and `docs/oracle/messy-city.md` records it.

**2. Placeholder scan.** Every code step carries real code. The only
bracketed placeholders are the `[from the run]` markers in Task 10, which
that task declares as the one permitted place. No "TBD", no "similar to
Task N" (the `overlap_city` helper, the `MethodologyConfig` construction and
the messy `compute_frames` call are each written out in full where they are
used), no "add appropriate error handling" — the two error paths (the
unknown-`kind` `ValueError`, the "would lend" guard) have their messages
spelled out, as does the reserved-key text and the script's stamp
`SystemExit`.

**3. Name and type consistency.** The § 8 "Names Group C relies on" block is
honoured: `index.pcen(..., nbr_weight_col=None)` and the loop line
`poly_count += w * lent * _decay(...)` are Group B's, already on `main`, and
this plan only redefines `lent` (Task 3 Step 4 quotes the existing lines it
replaces); `pipeline.NBRS_WEIGHT_COL` is read, never redefined;
`reference_impl.compute_city(..., barrier_buffer_m=None)` keeps its
signature and gains one keyword at the end; `VARIANT_KNOBS` and
`variant_methodology` follow the per-block branch pattern DEL-48
established. One spelling each, everywhere: the config value is
`outside_receiver`, the reference knob is `overlap_lending`, the config
dataclass is `OverlapConfig.lending`, the enum is `OverlapLending`, the
production function and the `pcen` parameter are both `shared_amounts` (the
spec names both; the shadowing inside `pcen` is called out in Task 3 so
nobody "fixes" it), the variants are `overlap_outside` and
`partial_5m_outside`, the measurement block is `overlap_lending` and its
derived profile is `overlap-outside-receiver`. The reference's
`shared_amounts` returns `{svc: {(i, j): amount}}` (a table per service);
production's returns `{(i, j): amount}` for ONE service, because
`index_frames` builds one per service — the two are independent by design
and each caller matches its own, which Task 1 and Task 3 both state.

**4. Numbers.** Every constant asserted in this plan was computed against
this worktree before the plan was written, not copied from the spec: the
messy `O1`/`O2` values under both rules (`(1 + 1/1.8)/600` = 0.0025925925925925925
and `1/600` = 0.0016666666666666668; `(1 + 1/1.8)/700` and `1/700`), O1's
unchanged `school_pcen` 0.000925925925925926 and `police_pcen` 1/600, the
post-switch min-max argmin staying unique (`N` under `pop`, `S` under
`popdensity`) with no constant column on either denominator, the fact that
Oraculum has **zero** overlapping pairs and **zero** points inside more than
one settlement while the messy city has exactly one of each, that
production's `intersects` and the reference's `within` agree on that one
point, that no road row crosses the `O1`/`O2` overlap, the row counts the
two new variants add (322 per rule on oraculum, 460 on messy), and the
`shared_amounts` prototype's outputs (`{('P','Q'): 1, ('Q','P'): 1}` and
`{('P','Q'): 0.2, ('Q','P'): 0.2}`, with `index_right` confirmed as the
sjoin column name in this environment's geopandas).

Execution: subagent-driven-development.
