# Settlement-Category Mapping Layer — Design Spec (DEL-17; cycle 3B)

Date: 2026-08-27
Status: **draft for owner review** (pending ultracode review)
Branch: `del-17-categories` (off `origin/main` at 2b5d8ae)
Parent: WORKPLAN.md Phase 3 item "Make settlement types configurable via a
mapping layer" [DEL-17]; Phase 4 DEL-28/29/31/32 consume it. Builds on the
Phase 3A spec (`2026-08-27-phase3-refactor-design.md`) § 3 schema.

Decision log (brainstorm, 27 Aug 2026):
- Scope: the mapped category drives **exclusion** (written in category
  terms) and appears as a **`category` column on every output**. No
  per-category summary tables (Phase 5), no `urban-5` profile (DEL-31,
  after Raj).
- Unmapped source types are an **error**, never a warning or a fallback.
- Mapping lives **inline in each profile** (`categories:` block), not in
  separate scheme files — a profile stays a complete statement of method.

## Purpose

The Delhi layer carries 10 `USO_FINAL` source types; the paper's analysis
will use a small theory-first set (Phase 4 working candidate: planned /
unauthorized / regularized-unauthorized / resettlement / JJC, non-urban
dropped). Today the only place types touch the pipeline is `excluded_ids`,
which matches raw strings. This cycle makes the collapse a config choice:
a profile declares a **mapping** (source type → category) and expresses
**exclusion** in category terms; outputs carry the category. Raj's decision
then becomes one YAML profile (DEL-31) and a recalculation (DEL-32), with
no code change — and a different city's vocabulary is a different mapping.

The layer changes **no numbers** under the shipped profiles: both use the
identity scheme, so every fixture and the real-data baseline stay
byte-identical.

## 1. Vocabulary (measured 27 Aug 2026, `uso_update_sep2021.shp`)

4,357 rows, no nulls, unchanged by dedup:
UAC 1,684 · Planned 964 · JJC 764 · RUAC 393 · RV 211 · UV 138 · SDA 86 ·
JJR 48 · Industrial 36 · Other 33.
These 10 are themselves an undocumented 16→10 merge from the 2021
notebooks (`UAC1→UAC`, `JJC1/JJC2→JJC`; `Institutional/Commercial/DCB/NDMC`
folded or dropped). This spec records that; it does not re-derive it.
The oracle city's fixture vocabulary is `Planned, UC, JJC, RV, RUAC, IND`.

## 2. Config schema

Added to the profile schema of the 3A spec § 3. `categories` is
**required** in every profile from this cycle on (like `methodology`:
a profile states its method completely).

```yaml
categories:
  scheme: uso-10                  # free-form name; recorded in outputs' metadata
  mapping:                        # source type (in layers.settlements.type_col) → category
    Planned: Planned
    UAC: UAC
    JJC: JJC
    RUAC: RUAC
    RV: RV
    UV: UV
    SDA: SDA
    JJR: JJR
    Industrial: Industrial
    Other: Other
methodology:
  exclusion:
    types: [RV]                   # CATEGORY names (values of the mapping), not source types
```

Load-time rules (`ConfigError` naming the key and the allowed values):
- `categories.scheme`: non-empty string. `categories.mapping`: non-empty
  map of string → string; duplicate source keys are impossible in YAML but
  an empty or non-string value is rejected. Several sources mapping to one
  category (X:1) is the point; a category name equal to a source name is
  fine (identity).
- `methodology.exclusion.types`: every entry must be a **category** (a
  value of `mapping`); the error lists the categories the mapping
  produces. `[]` is allowed.
- Both shipped profiles (`code-2025`, `manuscript`) gain the identity
  `uso-10` block above. `code-2025` keeps `exclusion.types: [RV]`,
  `manuscript` keeps `[]`; both are valid category names under identity.

Run-time rule (`ValidationError`, exit 1 from the CLI): a source type
present in the settlements frame with no entry in `mapping` fails the run,
naming **every** unmapped type with its row count (one run diagnoses the
whole layer). A mapping entry for a type absent from the data is fine —
a scheme may be broader than one city.

Reserved for later, rejected at load with a message: `categories.default`
(a catch-all category) — deliberately not offered; silence is the failure
mode this spec exists to prevent.

## 3. Data flow

New module `delhi_psi/categories.py`, pure, never imports `config`:

`apply_mapping(frame, *, type_col: str, mapping: Mapping[str, str],
out_col: str = "category") -> DataFrame` — returns a copy with `out_col`
added; raises `validate.ValidationError` listing unmapped types and counts.
`categories_of(mapping) -> frozenset[str]` — the set of category names
(used by config validation and tests).

Where it runs: in the shared population/exclusion prelude
(`pipeline._population_and_exclusion`), **before** `excluded_ids`, on the
frame loaded from the neighbours artifact. `excluded_ids` matches
`out_col` against `exclusion.types` instead of `type_col`. Because
`compute` and `compute_frames` share the prelude, the oracle path and the
CLI path cannot diverge. `compute_frames` gains a keyword-only
`mapping: Mapping[str, str] | None = None`; `None` means identity over the
types present (so existing oracle tests keep their call shape).

The neighbours artifact is unchanged and category-free — it is built on
the full universe and stamped with adjacency/barrier only — so a mapping
change never forces an 11-minute `preprocess`. Exclusion still happens at
compute, as in 3A (`stage` × `absent_neighbor` untouched).

Outputs: every PSI CSV / shapefile / joblib row carries `category`
(string) alongside the raw `USO_FINAL`, which is kept as-is;
`missing_population.csv` too. The column name is under the 10-character
shapefile limit. The scheme name is written into the frame's `attrs`
(joblib) and logged; it is not a column.

## 4. Proofs and fixtures

- **No numbers change under the shipped profiles.** `category` is a label,
  not a metric, so the production fixture format (§ 4 of the 3A spec) is
  untouched: `code-2025.csv` and `manuscript.csv` must be byte-identical
  after this cycle — the CI drift guard proves it on every push. The
  real-data run under `code-2025` must still show 0.000e+00 on all 23
  columns.
- **Reference implementation unchanged.** The reference has no category
  concept; its scenarios drop settlements by id. `test_profiles_match_reference`
  keeps passing because category exclusion resolves to the same ids.
- **Vocabulary-change equivalence.** A CLI e2e on the oracle city with a
  5-way collapse (`Planned→planned, UC→unauthorized, RUAC→regularized,
  JJC→jjc, RV→non-urban, IND→non-urban`) and `exclusion.types:
  [non-urban]` must produce exactly the numbers of today's raw
  `[RV, IND]` exclusion (the `excl_contributing` / `excl_removed` reference
  blocks at 1e-12 depending on `stage`). This is the proof that the layer
  only changed the vocabulary.

## 5. Tests

- `tests/test_categories.py`: identity is a no-op on values; X:1 collapse;
  unmapped type raises naming all offenders with counts; mapping broader
  than the data passes; input frame not mutated.
- `tests/test_config.py` additions: missing `categories:` rejected;
  `exclusion.types` entry that is not a category rejected with the allowed
  list; `categories.default` rejected as reserved; both shipped profiles
  load with the identity scheme.
- `tests/test_cli.py` additions: `category` column present in CSV, shp and
  joblib outputs and in `missing_population.csv`; the 5-way collapse e2e
  above; an unmapped type in the layer → exit 1 with the type named.
- `tests/test_pipeline.py` additions: `compute_frames(mapping=None)`
  equals `mapping=identity` at 1e-12 on every metric column.
- Existing 246 tests carry over unchanged in expectation.

## 6. Documentation

- `docs/methodology-config.md`: new section "Categories" — the two knobs
  (mapping vs exclusion) with the worked `urban-5` example, and the
  procedure for Raj's decision (copy profile, write the mapping, set
  `exclusion.types: [non-urban]`, regenerate fixtures, recalculate).
- `docs/data/uso_final_vocabulary.md`: § 1 above as a standalone artifact
  (counts, provenance of the 16→10 merge, the oracle fixture vocabulary).
- CHANGELOG `[Unreleased]`; WORKPLAN DEL-17 tick; DEL-31 note ("one YAML").

## 7. Out of scope

Per-category summary/rank tables (Phase 5); the `urban-5` profile itself
(DEL-31 — Raj); any reference-impl change; re-deriving the 16→10 merge;
scheme files shared across profiles (revisit when a second city exists).

## 8. Autonomy and stopping rules

Same as the 3A spec: fix-forward/commit/push/merge yes once CI and § 4
proofs are green; a confirmed Critical review finding governs over the
plan. Stop and report if either shipped fixture or the real-data baseline
deviates, or if a carried-over test needs its expected value changed.
