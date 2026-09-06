# Real-layer pathologies

Where the messy-city fixture tier's premises come from
(`docs/superpowers/specs/2026-08-28-messy-city-tier-design.md` § 5). Every
number below is produced by `scripts/measure_layer_pathologies.py`, which
reads the layers named by the `code-2025` profile, applies the pipeline's own
deduplication and population join, and writes nothing under the data
directory. `tests/test_layer_pathologies.py` re-runs it and compares the
counts (it skips when the data is not present).

- **Run date:** 2026-09-05
- **Layer:** `uso_update_sep2021/uso_update_sep2021.shp`
- **Commit:** `205eb7c` (the commit whose code emits all thirteen pathology
  keys, including the two DEL-50 keys `corner_only_pairs` /
  `corner_only_settlements`; the eleven earlier keys are unchanged from the
  28 Aug 2026 run at `181e92b`)
- **Command:** `uv run python scripts/measure_layer_pathologies.py --config code-2025` (cold: no `--cache-dir`, because a warm GeoPackage cache upcasts Polygon to MultiPolygon and would report `multipolygons` as 4357)

```text
settlements: 4357
rectangles: 0
multipolygons: 556
isolated_bbox: 6
isolated_touch: 20
no_population: 15
area_km2_min: 2.30282e-09
area_km2_median: 0.0506134
area_km2_max: 29.1165
overlapping_pairs: 4069
corner_only_pairs: 656
corner_only_settlements: 955
multi_settlement_points_bank: 211
multi_settlement_points_health: 18
multi_settlement_points_police: 2
multi_settlement_points_ration: 104
multi_settlement_points_school: 53
multi_settlement_points_transport: 41
```

## Reading the numbers

- `rectangles` — polygons that fill their own bounding box. Every one of the
  Oraculum city's seven settlements is one; this is what makes Oraculum
  unable to tell `bbox` adjacency apart from polygon intersection, and the
  messy city's `H`/`L`/`T` the fix.
- `isolated_bbox` — settlements with an EMPTY neighbour list under the
  production `bbox` rule; the messy city's `I`.
- `isolated_touch` — settlements with an EMPTY neighbour list under the
  `touch` rule (border-sharing with positive length). bbox-neighbours are a
  superset of touch-neighbours, so `isolated_bbox <= isolated_touch` always;
  here 6 <= 20. Neither number matches WORKPLAN's earlier ad-hoc "~360 have
  zero neighbours" — that figure predates this reproducible measurement and
  should be read as superseded, not confirmed, by either count.
- `no_population` — settlements the population join leaves without a value.
  Production drops them from the reported frame unconditionally; the messy
  city's `U`.
- `overlapping_pairs` — polygon pairs whose intersection has positive area.
  They are `touch` neighbours today (DEL-19) and they double-count any
  service point inside the overlap (DEL-20); the messy city's `O1`/`O2`.
- `corner_only_pairs` / `corner_only_settlements` — polygon pairs whose
  intersection is non-empty but has **zero length and zero area**: they meet
  at one or more isolated points and nowhere else, and the settlements
  involved. Raj ratified shared-border adjacency on 28 Aug 2026 (decision log
  § 3), and a corner is not a border: such a pair is a neighbour under `bbox`
  and under a 0 km distance band, and NOT under `touch`. The messy city's
  `T`/`L` is the hand-checkable case (`docs/oracle/messy-city.md`); Oraculum
  has none. **Measured 5 Sep 2026: `656` corner-only pairs involving `955`
  settlements** — not the "zero or near zero" the 28 Aug call assumed. The
  likely reading, NOT verified by this measurement (which tests only that a
  pair's intersection has zero length and zero area, never how many
  polygons meet at the point): on a tessellated layer four-way corners
  produce exactly this, the two diagonal polygons touching only there;
  T-junction vertices and digitisation near-misses would count the same
  way. Under
  `touch` (shared border, Raj's rule) none of the 656 pairs are neighbours;
  under a 0 km distance band all of them would be, on top of the
  positive-length pairs. Bob's ruling for the ratified profile: a corner is
  not a border — `touch` stands — and the count goes to Raj as an FYI so the
  methods can say "sharing a border of positive length". The 20 isolated
  settlements under `touch` (`isolated_touch`) already reflect this rule.
- `multi_settlement_points_<service>` — points inside more than one
  settlement, counted for each. The `<service>` names are the `code-2025`
  profile's service layer names, so `health` here is the messy city's
  `clinic`.
