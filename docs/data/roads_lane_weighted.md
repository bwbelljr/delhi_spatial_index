# Lane-weighted roads — the intensive margin (DEL-43)

**Run date:** 2026-10-28. **Baseline:** the DEL-32 `ratified-2026` run
(`~/delhi_data/phase4_ratified`). **Variant:** `roads-lane-weighted` — the
ratified profile with `services.line.road` written as
`{path: …/Road.shp, weight_col: RD_LANES}`, so a road's amount is
length × lanes rather than length.

**Command:** `uv run delhi-psi compute --config roads-lane-weighted
--data-dir ~/delhi_data --out-dir ~/delhi_data/phase4_lane`. The neighbours
artifact is the ratified one: the methodology stamp covers adjacency and
barrier only, and weighting does not change the neighbourhood.

**Rows:** 4,062 in both runs — this is the same universe, scored differently.

## Planned-relative means, popdensity

| type | km | lane-km |
|---|---|---|
| UV | 1.2737 | 1.2512 |
| JJR | 1.1796 | 1.1731 |
| Planned | 1.0000 | 1.0000 |
| RUAC | 0.9229 | 0.9013 |
| UAC | 0.5287 | 0.5221 |
| SDA | 0.5169 | 0.5147 |
| JJC | 0.0602 | 0.0598 |

## Finding

**The intensive margin moves levels, not order.** The category ordering is
identical, and every mean moves by 0.03 or less (the largest is UV, −0.02);
the mean `road_idx` itself barely moves (0.0075 → 0.0069) because the
min-max inside the road column is affine-invariant to a *uniform* rescale —
only the lanes' variation across settlements can bite, and Delhi's road
network is close to uniform. So the appendix sentence is: counting lanes
instead of kilometres does not change which types look better or worse.

The real layer carries `RD_LANES`; the fixture cities now carry `lanes`
(oraculum 2; messy 2 and 6, chosen so the messy pin is load-bearing — the
road that serves `M` outweighs the one serving `H`/`L`). Adding the property
left every committed expected value byte-identical, which is the guard the
design asked for.
