# The messy city

Oraculum is small, rectangular and hand-ratifiable by design; the real layer
is none of those. The messy city carries each real-layer pathology Oraculum
omits **once**, is scored by the independent reference implementation
(`tests/reference_impl.py`) rather than by hand arithmetic, and pins what
production does on each pathology **today** — so that the adjacency and
overlap fixes (DEL-19, DEL-20) are proven by a test that flips, not by
argument.

- Spec: `docs/superpowers/specs/2026-08-28-messy-city-tier-design.md`
- Generator: `scripts/generate_messy_fixtures.py` (asserts every relation
  below before writing a byte, and installs `expected_values.csv` only after
  `scripts/check_oraculum_invariants.py` passes on it)
- Fixtures: `tests/fixtures/messy/`
- Pins: `tests/test_messy_fixtures.py`
- Real-layer premises: `docs/data/layer_pathologies.md`

Oraculum stays the hand-ratifiable ground truth for the *math*
(`docs/oracle/derivation-worksheet.md`). This city has **no hand anchors** by
design: its numbers are whatever the reference says, and its job is the
*geometry*.

## The eleven settlements

Coordinates are EPSG:7760 metre offsets from `BASE_X = BASE_Y = 1_000_000`.
Vocabulary `messy-2` = `(Planned, RV)`. There are no barriers in this tier.

| id | type | pop | area km² | shape | what it pins |
|---|---|---|---|---|---|
| `H` | Planned | 110 | 1.33 | irregular hexagon | `H ∩ L = ∅`, yet each geometry reaches into the other's **envelope** → bbox neighbours both ways, `touch` neighbours neither way |
| `L` | Planned | 200 | 2.56 | concave L | the envelope-only relation with `H`; corner-only contact with `T` |
| `T` | Planned | 300 | 0.25 | triangle | contact with `L` is a single **Point** (length 0): a `bbox` neighbour, never a `touch` one |
| `M` | Planned | 400 | 2.0 | two-part MultiPolygon | its centroid `(6500, 500)` lies **outside** it, in the gap; its envelope spans the gap |
| `G` | Planned | 50 | 0.01 | 100 m square in `M`'s gap | centred **exactly** on `M`'s centroid: `M ∈ nbrs_bbox(G)` but `G ∉ nbrs_bbox(M)` (an axis-aligned square *is* its own envelope), and `d = 0` → decay weight exactly 1, the undecayed maximum |
| `O1` | Planned | 600 | 1.0 | rectangle | overlaps `O2` in a 200 m × 1000 m strip |
| `O2` | Planned | 700 | 1.0 | rectangle | one clinic strictly inside the overlap is counted for **both** (DEL-20); the pair are `touch` neighbours because an overlap polygon's `.length` is its perimeter (DEL-19) |
| `I` | Planned | 800 | 1.0 | far-away square | **isolated**: an empty neighbour list under both rules |
| `N` | **RV** | 900 | 1.0 | square beside `O2` | the settlement `code-2025` excludes by **category** — and it *has* a population, so exclusion is what removes it |
| `U` | Planned | **none** | 1.0 | square beside `O1` | **no population row**: production drops it unconditionally, under every profile and scenario |
| `S` | Planned | 100 | **2e-06** | 2 m × 1 m sliver on `H`'s edge | the area extreme: a `popdensity` denominator of 5e7, so its ration PCEN is the minimum among ration owners and five orders of magnitude below `M`'s |

All ten present populations are distinct, so no two settlements can tie by
construction. All seven services are placed (clinic in `H L T M G O1∩O2 I`;
school in `L M O2 N S T G`; bank in `H I`; police in `L O1`; ration in
`M S`; transport in `H N`), because the reference scores every service in
`POINT_SERVICES` regardless and production's PSI averages over the services
present — both sides must carry the same seven. The road is **two**
LineString rows (`H` 1.2 km, `L` 0.6 km, `M` 2.0 km), which is what makes
"sum every road row" load-bearing: `M`'s whole length comes from the second.

## The three scenarios

Every scenario drops `U`, with the scenario's own flag, because production
drops a no-population id unconditionally and applies its single `stage` to
the whole drop set (`dropped = excluded_ids ∪ missing`).

| name | reference `dropped` | before neighbours? | production side |
|---|---|---|---|
| `nopop_only` | `{U}` | no | `types: []`, `post_neighbors` |
| `excl_rv_post` | `{U, N}` | no | `types: [RV]`, `post_neighbors` |
| `excl_rv_pre` | `{U, N}` | yes | `types: [RV]`, `pre_neighbors` |

`nopop_only` and `excl_rv_post` differ exactly by `N`, so category exclusion
is genuinely exercised; `excl_rv_post` and `excl_rv_pre` differ exactly by
whether `N` and `U` stay in other settlements' neighbour lists.

## What is deliberately NOT here

- **Barriers.** `barriers.geojson` is an empty collection; multi-layer
  `combine` coverage needs a second barrier layer and is its own follow-up.
- **Hand anchors.** By design (see above).
- **Any rule change.** Edge-only adjacency, single-assignment overlap and
  `partial_weighted` are DEL-19/20/22, after Raj. This tier records today.
- **`L` has no `touch` neighbours** either, which is fine: it owns a school,
  so it is not part of the zero-tie that the schools in `T` and `G` exist to
  break (only `I` sits at exactly 0).

## How to add a case

1. Add the settlement (and any service point) to `SETTLEMENTS` /
   `POINT_SERVICES` / `ROADS` in `scripts/generate_messy_fixtures.py`, with a
   **distinct** population and an `area_km2` the shoelace helper computes.
2. Add the relation you are pinning to `_assert_relations` in that script —
   the generator must fail loudly if a later coordinate edit breaks it.
3. Run `uv run python scripts/generate_messy_fixtures.py`. If the invariants
   guard rejects the result (a degenerate min-max group, or a tied
   clinic/school argmin/argmax), nothing is written: give some settlement the
   service it needs and try again.
4. Run `uv run python scripts/generate_production_fixtures.py` to refresh
   `tests/fixtures/messy/production/*.csv`.
5. Add the production-side pin to `tests/test_messy_fixtures.py`.
6. Run `uv run pytest -q -W error`. `test_profile_matches_reference` proves
   the two implementations still agree on the new city at 1e-12; if it fails,
   you have found a real divergence — **report it, do not tune it away**.
