# `USO_FINAL` — the settlement-type vocabulary

Measured 27 Aug 2026 on `~/delhi_data/uso_update_sep2021/uso_update_sep2021.shp`.
Recorded here because the mapping layer (`categories:`, cycle 3B / DEL-17) and
Raj's categorization decision (Phase 4 / DEL-29) both argue from these counts.

## The 10 types

4,357 rows, no nulls, unchanged by deduplication:

| `USO_FINAL` | rows |
|---|---:|
| UAC | 1,684 |
| Planned | 964 |
| JJC | 764 |
| RUAC | 393 |
| RV | 211 |
| UV | 138 |
| SDA | 86 |
| JJR | 48 |
| Industrial | 36 |
| Other | 33 |
| **total** | **4,357** |

`code-2025` today excludes `RV` (211 rows) and nothing else. DEL-28 proposes
dropping every non-urban type, which adds `Industrial`.

## Provenance: these 10 are already a merge

The 10 are themselves an undocumented **16 → 10** merge performed in the 2021
notebooks (`archive/master-2021/`): `UAC1 → UAC`, `JJC1`/`JJC2 → JJC`, and
`Institutional`, `Commercial`, `DCB`, `NDMC` folded in or dropped. This page
records that; it does not re-derive it (spec 3B § 7 — out of scope). Anyone
re-deriving the merge should start from
`archive/master-2021/Colonies Dataset Pre-Processing (29 Aug 2021).ipynb`.

## The oracle fixture's vocabulary is different on purpose

`tests/fixtures/oraculum/settlements.geojson` carries six types —
`Planned, UC, JJC, RV, RUAC, IND` — where `UC` and `IND` are shorthand for
`UAC` and `Industrial`. The shipped Delhi profiles are deliberately **not**
padded with `UC`/`IND`: padding them would blunt the unmapped-type guard on
real data. Tests that run the oracle city therefore use a derived, test-only
profile (`tests/oraculum_fixtures.oracle_config`), and one test runs the
shipped profile at the fixture city precisely to prove the guard fires.
