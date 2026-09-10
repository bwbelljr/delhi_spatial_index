# Lane-weighted road length — DEL-43 (narrowed)

**Ticket:** DEL-43. **Branch:** `del-43-lane-weighted-roads` off `main` at `00ae40d`.
**Date:** 9 Sep 2026. Cycle 5, ticket 3 of 3.

---

## 1. The ticket asked for capacity; the data has exactly one usable field

DEL-43 reads "facility size / capacity (intensive margin) — data-permitting".
I measured every service layer before writing this, and the honest answer is
that **the data does not permit it for any point service**:

| layer | attributes | capacity measure? |
|---|---|---|
| bank | `bank_name, Latitude, Longitude, Type` | no |
| health | `Hospital_C, ADDRESS, X, Y` | **no bed count** |
| police | `NAME, POLICE_STA, DISTRICT, x, y` | no |
| ration | `S No., License No, FPS ID, Circle, FPS Shop N, …` | **no cardholder count** |
| school | 21 cols: `schcat, school_cat, schtype, schmgt, …` | **categorical type only** |
| transport | `stop_id, stop_name, stop_lat, stop_lon, Type` | no |
| road | `RD_NM, RD_CLS, RD_LANES, RD_TP_SRF, ONEWAY, Speed_kmph` | **`RD_LANES`** |

The extensive margin — a facility is present or it is not — is not a
simplification this project chose. It is the only thing six of the seven
layers support.

**`RD_LANES` is clean enough to build on**, measured across all 17,647 rows:

| attribute | nulls | values |
|---|---|---|
| **`RD_LANES`** | **0 (0.0 %)** | `4`×8,304 · `2`×5,886 · `6`×2,725 · `1`×732 |
| `RD_CLS` | 13,575 (**76.9 %**) | needs an imputation rule nobody can justify |
| `Speed_kmph` | 0 (0.0 %) | but **637 rows are `0`** — a missing value wearing an integer's clothes |

So `RD_LANES` needs **no missing-data policy at all**, which is what makes it
cheap to pin in the oracle: a settlement's road amount stays a deterministic
function of geometry and one complete integer column. `Speed_kmph` is the
trap — it reads as complete and would silently zero 637 roads.

## 2. This is a service-definition change, not a methodology change

Eq. 4 already treats roads specially: their amount is **length**, not a
count. Lane weighting changes that amount from `Σ length` to
`Σ length × lanes` — still an amount of road, measured differently.

That puts it with **DEL-40/41/42** (which service layers count — config
values, no new maths) rather than with **DEL-34/DEL-57** (which alter what
the index computes). No new `methodology` enum.

**The config shape.** `services.line` entries today are bare path strings:

```yaml
services:
  line:  {road: Public Services/Major Road/Road.shp}
```

They gain the option of a mapping, with `layers.population` as the existing
precedent for "a path plus the columns to read from it":

```yaml
services:
  line:  {road: {path: Public Services/Major Road/Road.shp,
                 weight_col: RD_LANES}}
```

**Both forms stay valid**, so all 17 profiles are untouched and a bare string
keeps meaning exactly what it means today. A new `roads-lane-weighted`
profile carries the mapping form.

**The parsed shape keeps `services.line` a plain `{name: path}` mapping**,
and that is not a detail — four existing consumers use it directly as a path:

- `delhi_psi/pipeline.py` merges `{**cfg.services.point, **cfg.services.line}`
  and iterates `name, path`;
- `scripts/measure_roads_access.py` does
  `io.read_layer(cfg.paths.data_dir / cfg.services.line[ROAD_SERVICE])`, twice;
- `scripts/measure_rule_effects.py` repeats the merge pattern;
- `scripts/generate_production_fixtures.py` derives its column set from it.

So the YAML mapping is **split during parsing**, not carried through:
`ServicesConfig.line` stays exactly what it is today, and a new sibling
`ServicesConfig.line_weights` — `{name: weight_col}`, empty by default —
holds the column. Every existing consumer is then untouched by construction
rather than by inspection, and the only code that needs to know about
weighting is the code that computes a road amount.

## 3. The fixture authority decision, and the measurement behind it

The fixture road features carry no attributes today, so they need a `lanes`
property. Before choosing values I measured what each city's roads can
actually distinguish:

| city | road features | which settlements |
|---|---|---|
| oraculum | **1** | A 750 m, E 750 m |
| messy | **2** | road0 → H 1200 m, L 600 m · road1 → M 2000 m |

**Oraculum cannot test this at all, and that is arithmetic, not opinion.**
One road feature means every settlement's road amount takes the *same*
weight, and Eq. 2's min-max is affine-invariant, so the weight cancels
exactly. Verified across lane counts 1, 2, 4 and 6:

```
lanes=1: road_idx = {A: 1.0, E: 1.0, B: 0.0, C: 0.0, D: 0.0, IND: 0.0, RV: 0.0}
lanes=2: road_idx = {A: 1.0, E: 1.0, …}   ← identical
lanes=4: road_idx = {A: 1.0, E: 1.0, …}   ← identical
lanes=6: road_idx = {A: 1.0, E: 1.0, …}   ← identical
```

**This is the same affine-invariance trap that produced two wrong claims in
cycle 4** (the `ration_idx` case, where a uniform rescale was mistaken for a
moving column). Measured first this time.

**The messy city carries the proof** — but the first draft of this section
got its numbers badly wrong, and the correction is worth stating plainly.

**What the draft did.** It took the raw clipped road lengths (H 1200 m,
L 600 m, M 2000 m), min-maxed them by hand, and published the result as
`road_idx`. That produced H=0.6, L=0.3, M=1.0 and the conclusion that M
leads at baseline.

**Every part of that is wrong.** `road_idx` is the min-max of *PCEN*, not of
length — PCEN divides by the settlement's population (or population density),
and under `methodology.roads: decayed` it also carries a neighbour term. The
committed fixture says so directly:

```
ideal,nopop_only,pop,H,road_idx,1
ideal,nopop_only,pop,L,road_idx,0.27500000000000002
ideal,nopop_only,pop,M,road_idx,0.45833333333333337
```

**H is already the maximum at baseline**, not M. The spec review reproduced
the real path — patching only the two amount functions and leaving adjacency,
decay, min-max and exclusion untouched — and recovered these committed values
exactly, 6/6, which is what makes its lane sweep trustworthy.

**Its sweep also inverts the draft's choice.** Across both `roads` formulas
and both denominators:

- **`road0=6, road1=2` — the draft's pick — never reorders anything.** H stays
  the maximum throughout (code/pop: H=1.0, L=0.432, M=0.124).
- **`road0=2, road1=6` reorders under all four combinations** — M overtakes H
  (code/pop: H=0.112, L=0.049, **M=0.125**).

**Chosen fixture values: road0 = 2 lanes, road1 = 6 lanes.** Oraculum's single
road gets `lanes: 2`, and the variant is **degenerate there by
construction** — recorded, not hidden.

**The lesson, stated once and kept.** This section previously claimed the
cycle-4 affine-invariance trap had been "measured first this time". It had
not: measuring the *input* to a pipeline stage and presenting it as the
*output* is the same error wearing different clothes. **A fixture value is
only measured when it comes out of the real path** — for this repo that means
`compute_city` or the committed `expected_values.csv`, never hand arithmetic
on an intermediate. The reviewer's method — patch the two amount functions,
leave everything else, and check the unweighted case still reproduces the
committed CSV — is the standard to copy for any future fixture-authority
decision.

Degeneracy on one city is established practice here: `boundary` and
`overlap_outside` are degenerate on Oraculum, `partial_5m` is degenerate on
messy. Each time, the other city carries the pin.

## 4. What must not move

- **Both shipped profiles keep the bare-string form**, so every existing
  expected value and production fixture is **byte-identical**. If one moves,
  the weighting has leaked into the default path — STOP.
- Adding a `lanes` property to the fixture road features changes
  `services.geojson`, and **must not change `expected_values.csv`**: the
  unweighted path sums length regardless of properties. Asserting exactly
  that is the guard this ticket needs, because it is the one place a silent
  break could hide.
- `variants_expected_values.csv` is **addition-only**, verified by
  `--numstat`.
- No adoption. Whether a lane-weighted road index is a claim the paper wants
  to make is Raj's decision, like DEL-34 and DEL-57 before it.

## 5. Variant coverage

**The variant harness has no channel for this, and that is the largest piece
of work in the ticket.** The first draft said "one new row in
`tests/variants.py`". That is not possible today, and the spec review proved
it by running the shape:

```
>>> _variant_overrides({'services': {'line': {'road': {'weight_col': 'lanes'}}}})
ValueError: tests/variants.py: services.line has no reference knob;
            add one to VARIANT_KNOBS or to IGNORED_VARIANT_KEYS
```

**`VARIANTS` is methodology-only by construction**, and three separate pieces
enforce it: `_variant_overrides` walks `(block, key)` pairs against
`VARIANT_KNOBS`; `oracle_profile_path`'s `methodology_overrides` only ever
writes `raw["methodology"][block]`; and `test_variants_match_reference`
calls `compute_frames(...)` directly with a `MethodologyConfig`, never
building a `Config`, so there is no `ServicesConfig` in that path to carry a
`line_weights` at all.

**So DEL-43 must build the channel before it can add the row:**

1. a `services` key in a `VARIANTS` entry, written verbatim into
   `raw["services"]` by `oracle_profile_path` — **beside**
   `methodology_overrides`, not through it, since `VARIANT_KNOBS` is a
   methodology vocabulary and forcing a services key through it would corrupt
   the one guard that keeps that table honest;
2. a `road_weight_col=None` keyword on `compute_city`, wired by name rather
   than through `VARIANT_KNOBS`;
3. whatever `test_variants_match_reference.produced()` needs to thread the
   weight from the variant spec into `compute_frames`.

This is the shape of the work, and it is why the ticket is bigger than "a
config value like DEL-40/41/42". § 2's framing of *what* is being changed
stands; the *cost* does not.

Then one row — `roads_lane_weighted` — scored by both implementations on both
fixture cities at 1e-12. Degenerate on Oraculum (equal to the `code` base
there, provably), load-bearing on messy.

The reference implementation multiplies each road's clipped length by that
road's `lanes` before summing, written independently of the production path.

**Both road paths need weighting, not just the obvious one.** A road's amount
is computed in two places, and they must agree or the overlap rule breaks:

1. the **own amount** — `index.road_lengths` / the reference's
   `_service_amounts`;
2. the **shared amount** — `index.shared_amounts` / the reference's shared
   table, which measures road length inside `i ∩ j` for the
   `overlap.lending: outside_receiver` rule.

If (1) is weighted and (2) is not, then `|S_j \ S_i| = amount_j - shared_ij`
mixes lane-km with plain km, and the existing guard — the one DEL-61 shows
is strict to a fault — will fire with a large negative residual rather than
float noise. That guard firing loudly is the good outcome; the bad one is a
profile where `overlap.lending` is `whole`, which never subtracts, so the
mismatch would pass silently. **Weight both, and pin the combination with a
variant that sets `overlap.lending: outside_receiver` alongside the
weighting.**

## 6. Out of scope

- **No real-data run.** Cycle 5's standing constraint.
- **No `RD_CLS` or `Speed_kmph` weighting** — § 1 gives the reasons; either
  would need its own argument and its own missing-data rule.
- **No point-service capacity.** The data does not support it, and the DEL-43
  comment thread records the measurement so nobody re-opens it on a hunch.
- No licence file (DEL-56, pending Raj).

## 6.1 Three details the spec review pinned down

**The arithmetic order is canonical, not incidental.** Multiply each road's
clipped length by its integer lane count **first**, accumulate, and divide the
total by 1000 **once** — mirroring today's structure with one inserted
per-row multiply. The two orders are not bit-identical:

```
messy H, road0, lanes=6:  (L/1000)*6 = 7.199999999999999
                          (L*6)/1000 = 7.2            diff ≈ -8.9e-16
```

This repo already treats that class of drift as consequential — the
reference's own note on non-associative float summation, and DEL-61's
overlap-lending guard, which is strict to a fault. All four sites
(`index.road_lengths`, `index.shared_amounts`, and the reference's two)
use the same order.

**A third call site exists and is deliberately left unweighted.**
`scripts/measure_rule_effects.py`'s `shared_pair_counts()` calls
`index.road_lengths` independently of `index_frames`, with layers built from
plain paths. It is harmless today — it reports only `len(table)`, a pair
*count*, and a positive lane multiplier cannot turn a nonzero length into
zero. It must carry a comment saying so, so nobody later "fixes" it on the
assumption it was missed, and nobody rediscovers it as a bug.

**The new mapping form needs its own validation**, following
`layers.population`'s precedent: require `path`, reject unknown keys (a
typo'd `weigth_col` must fail loudly), and reject a mapping value under
`services.point`, which this design does not support. `_services` today
validates only the top-level `point`/`line` keys, so
`{'point': {'health': {'path': ..., 'weight_col': 'bogus'}}}` parses
silently and fails much later with `Path / dict`.

## 7. Definition of done

- `services.line` accepts both the bare-string and mapping forms, with the
  mapping's `weight_col` optional; all 17 existing profiles unchanged and
  still loading.
- A `roads-lane-weighted` profile carrying the mapping form.
- The weighting implemented in the production path and, **independently**, in
  `tests/reference_impl.py`.
- Fixture road features gain `lanes`; `expected_values.csv` proven
  byte-identical by that change alone.
- One variant row agreeing at 1e-12 on both cities, with Oraculum's
  degeneracy asserted rather than assumed.
- Full suite green; no production fixture moved.
