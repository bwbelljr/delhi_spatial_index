# The oracle — validating a port of the method

This directory and `tests/fixtures/` together are the part of this repository
most useful to someone outside the project. They are a **portable validation
suite** for the Public Services Index: two small cities whose correct answers
are known independently of any code, and against which a reimplementation can
be checked.

If you are porting the method to another city, this is the thing to use. You
do not need Delhi's data, and you do not need to trust this implementation.

## Why it exists

Research code is normally checked by running it and looking at the output.
That finds crashes; it does not find wrong answers, because there is nothing
to compare the answer to. The index here is a chain of four equations over
neighbour relationships, distance decay, barriers and per-capita
normalisation — a place where a plausible-looking number can be wrong for a
dozen quiet reasons.

So the correct answers were produced twice, by routes that share nothing:

1. **By hand.** *Oraculum* is seven settlements laid out on graph paper with
   coordinates chosen so every intermediate quantity is computable with a
   calculator. The arithmetic is written out in
   [`derivation-worksheet.md`](derivation-worksheet.md) — every neighbour
   pair, every distance, every decayed contribution, every min-max — and it
   was ratified before any code was compared against it.
2. **By a second implementation.** [`tests/reference_impl.py`](../../tests/reference_impl.py)
   implements the same equations independently and **imports nothing from
   `delhi_psi`**. It is deliberately naive: slow, literal, easy to read
   against the paper.

The production pipeline is then required to agree with both, at `1e-12`.
When they disagree, the hand arithmetic wins — that is what makes this an
oracle rather than a fixture.

## What is in it

| | |
|---|---|
| **Oraculum** | 7 settlements, hand-derived. Clean geometry: the reference case. |
| **Messy city** | 11 settlements carrying the pathologies a real layer has — two polygons that share no area yet each reach into the other's bounding box, contact that is a single point rather than an edge, a two-part settlement whose centroid falls outside it, an overlapping pair, an isolated settlement, one with no population row, one excluded by category, and an area extreme five orders of magnitude off the others. [`messy-city.md`](messy-city.md) gives each one and what it pins. |
| **Expected values** | 2,610 committed numbers for Oraculum, across 2 rule-sets, 5 exclusion scenarios, 2 denominators and 23 metrics per settlement. Not a full cross-product: a scenario that excludes a settlement emits no rows for it. |
| **Variant expectations** | 5,796 further rows covering 18 methodology variants — distance bands, four decay forms, boundary vs centroid distance, partial barriers, overlap lending. |
| **Maps** | The `.png` files here, rendered from the fixtures by `scripts/render_oracle_maps.py`. |

## The fixture format

Everything is plain GeoJSON and CSV, readable without this package.

`tests/fixtures/<city>/settlements.geojson` — polygons, with properties:

| property | meaning |
|---|---|
| `USO_AREA_U` | settlement id (`A`, `B`, … on Oraculum) |
| `USO_FINAL` | settlement type, the category the index compares across |
| `area_km2` | area, pre-computed so a port need not match our geometry library |
| `population` | denominator input |

**Coordinates are EPSG:7760, in metres** — a projected CRS, so distances are
metric and areas are directly comparable. Each fixture file also carries a
top-level `crs_note` saying so.

`services.geojson` — points and lines, with `service` (the service type) and
`host` (the settlement it sits in, for readability; the code derives
containment geometrically rather than trusting it).

`barriers.geojson` — lines that sever or discount neighbour relationships.
May be an empty feature collection, as on the messy city.

`expected_values.csv` and `variants_expected_values.csv` — long format:

```
rule,scenario,denom,settlement,metric,value
ideal,baseline,pop,A,clinic_count,2
ideal,baseline,pop,A,clinic_pcen,0.029142135623730948
```

- `rule` — `ideal` (the paper's equations as written) or `code` (the frozen
  2025 production behaviour), or a named variant in the variants file.
- `scenario` — which settlement types are excluded, and whether excluded
  settlements still lend services to their neighbours.
- `denom` — `pop` or `popdensity`.
- `metric` — a per-service count, a `_pcen` (population-corrected exposure:
  the settlement's own services plus its neighbours' decayed contributions,
  divided by the denominator), a `_idx` (that column's min-max normalisation
  across the reported settlements), or a summary: **`psi_eq1`** (the index as
  Eq. 1 defines it) and `norm_psi` (Eq. 1 min-maxed a second time). Note the
  CSV says `psi_eq1` where the production dataframe column is called
  `unnorm_psi` — the fixtures use the paper's name, not the code's.
- `value` — at full float precision. Compare at `1e-12`, not by string.

## Validating a port

```bash
uv sync
uv run pytest -q tests/test_oracle.py tests/test_oracle_e2e.py tests/test_reference_impl.py
```

68 tests in under 10 seconds, no external data. That checks *this*
implementation. To check **yours**:

1. Read `settlements.geojson`, `services.geojson` and `barriers.geojson` for
   a city.
2. Compute the index with your implementation, under one `rule`, `scenario`
   and `denom` combination.
3. Compare against the matching rows of `expected_values.csv` at `1e-12`.
4. Start with `rule=ideal, scenario=baseline, denom=pop` on Oraculum — the
   simplest case, and the one the derivation worksheet walks through by hand.
   Then the messy city, which is where most ports break.

If step 4 disagrees, [`derivation-worksheet.md`](derivation-worksheet.md)
shows where: it gives the intermediate quantities, not just the final index,
so you can find the equation you and it read differently.

## What it does and does not prove

**Proves:** that an implementation computes the documented method — the
neighbour relation, the decay, the barrier handling, the exclusion semantics
and the two normalisations — to twelve decimal places, on geometry that
includes the awkward cases.

**Does not prove:** that the method is the right method for a research
question, that Delhi's input layers are accurate, or that a port's own data
is clean. Those are different problems and this suite is silent on them.

Two real defects it caught, as evidence that it works: a min-max that divided
0/0 on a degenerate group, and an over-subtraction in the overlap-lending rule
that only appeared on real geometry.

A third case is worth naming separately, because it was *pinned* rather than
caught: whether a settlement pair sitting exactly on a distance band's radius
counts as neighbours. It is inclusive, both implementations agree, and a test
holds it there — an edge case that is easy to get wrong in a port and easy to
never notice, which is a different kind of value from finding a bug.

## Also here

- [`messy-city.md`](messy-city.md) — what each pathology in the second city
  is, and which real-layer problem it stands for.
- [`exclusion-semantics-memo.md`](exclusion-semantics-memo.md),
  [`rv-exclusion-decision-memo.md`](rv-exclusion-decision-memo.md),
  [`suggested-fixes-memo.md`](suggested-fixes-memo.md) — working memos
  written while the method's open questions were being settled. Kept because
  they record *why* choices were made, which the code cannot.
