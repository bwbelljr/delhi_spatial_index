# Real-layer pathologies

Where the messy-city fixture tier's premises come from
(`docs/superpowers/specs/2026-08-28-messy-city-tier-design.md` § 5). Every
number below is produced by `scripts/measure_layer_pathologies.py`, which
reads the layers named by the `code-2025` profile, applies the pipeline's own
deduplication and population join, and writes nothing under the data
directory. `tests/test_layer_pathologies.py` re-runs it and compares the
counts (it skips when the data is not present).

- **Run date:** 2026-08-28
- **Layer:** `uso_update_sep2021/uso_update_sep2021.shp`
- **Commit:** `e14a87f`
- **Command:** `uv run python scripts/measure_layer_pathologies.py --config code-2025`

```text
settlements: 4357
rectangles: 0
multipolygons: 556
isolated_bbox: 6
no_population: 15
area_km2_min: 2.30282e-09
area_km2_median: 0.0506134
area_km2_max: 29.1165
overlapping_pairs: 4069
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
  production rule; the messy city's `I`.
- `no_population` — settlements the population join leaves without a value.
  Production drops them from the reported frame unconditionally; the messy
  city's `U`.
- `overlapping_pairs` — polygon pairs whose intersection has positive area.
  They are `touch` neighbours today (DEL-19) and they double-count any
  service point inside the overlap (DEL-20); the messy city's `O1`/`O2`.
- `multi_settlement_points_<service>` — points inside more than one
  settlement, counted for each. The `<service>` names are the `code-2025`
  profile's service layer names, so `health` here is the messy city's
  `clinic`.
