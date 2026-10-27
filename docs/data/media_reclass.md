# Media reclassification — the appendix check (DEL-53)

**Run date:** 2026-10-27. **Baseline:** the DEL-32 `ratified-2026` run
(`~/delhi_data/phase4_ratified`, commit `8470955`).
**Variant:** `reclass-media` — the same ratified profile, pointed at the
students' reclassified type column.

**Input.** Raj's 19 Oct sheet (`~/reclass_sheet.csv`): 4,342 rows, every
`uso_area_u` matched to the layer (validated 20 Oct); 105 flagged changes,
81 of them RV → UV. **Handling rule (2026-10-19 decisions § 6):** a
settlement takes the sheet's `FINAL CLASSIFICATION` only when that is a
single known type; multi-type finals (the 69 ambiguous rows plus one) keep
the state classification. That yields 105 effective single-type changes —
three of the flagged ones are UV→UV no-ops, so **81 settlements change
category** (RV → UV) and enter the reported universe.

**Command:** `uv run delhi-psi preprocess|compute --config reclass-media
--data-dir ~/delhi_data --out-dir ~/delhi_data/phase4_reclass`. The
settlement layer is a GeoPackage copy carrying `USO_RECLAS`; no production
code changed.

**Rows:** 4,062 → 4,143 (+81), all of them the reclassified RV settlements.

## Planned-relative means, by denominator

| type | popdensity (main) | reclassified | pop | reclassified |
|---|---|---|---|---|
| UV | 1.2737 | **3.3524** | 0.8473 | 0.8062 |
| JJR | 1.1796 | 1.1500 | 0.4005 | 0.3930 |
| Planned | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RUAC | 0.9229 | 0.8745 | 0.7165 | 0.7217 |
| UAC | 0.5287 | 0.5241 | 1.0569 | 1.0566 |
| SDA | 0.5169 | 0.5083 | 1.2297 | 1.2282 |
| JJC | 0.0602 | 0.0612 | 1.7773 | 1.7752 |

## Finding — "nothing changes"

**The category ordering is identical on both denominators.** On population
every mean moves by less than 0.05; on density the only large move is UV
itself (1.27 → 3.35), because the 81 ex-RV settlements join that category
and they sit at the high end of the density scale. Planned stays at 1 by
construction, and no other type changes place.

So the appendix sentence stands: reclassifying settlements from media
reports does not change which types look better or worse, only the level of
the category that absorbs the movers.

**Reconciling the count.** Raj remembered "~50" changes. The sheet flags
105; 81 are RV → UV (the "~80 %" he described), 24 are swaps among reported
types or no-ops, and only the 81 add rows to the universe. The 105/81 split
is the number to quote, not 50.
