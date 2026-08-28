# Settlement-Category Mapping Layer — Design Spec (DEL-17; cycle 3B)

Date: 2026-08-27
Status: **approved by owner (2026-08-27)** — rev 2 after ultracode review
(24 findings → 6 root causes fixed; confirmation round: 5 fixed — rev 3)
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
- Rev 2 (ultracode review): the shipped Delhi profiles stay pure — tests on
  the oracle city use a **test-only derived profile** whose mapping is the
  identity over the fixture vocabulary; the profile's mapping is **threaded
  into the oracle/fixture path** (`compute_oracle_frame`, `emit_profile`) so
  fixtures are generated under each profile's own mapping; duplicate YAML
  keys are rejected by a checking loader (PyYAML keeps the last one
  silently); the scheme stamp is applied by `compute` immediately before
  `write_outputs` (pandas drops `attrs` across `merge`); `exclusion.types ⊆
  categories` is checked at run time too, not only at load.

## Purpose

The Delhi layer carries 10 `USO_FINAL` source types; the paper's analysis
will use a small theory-first set (Phase 4 working candidate: planned /
unauthorized / regularized (WORKPLAN: "regularized-unauthorized") /
resettlement / JJC, non-urban dropped). Today the only place types touch the pipeline is `excluded_ids`,
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
  map of string → string; an empty or non-string value is rejected.
  **Duplicate source keys are rejected**: `load_config` parses profile YAML
  with a `yaml.SafeLoader` subclass whose `construct_mapping` raises
  `ConfigError` naming the repeated key (PyYAML's default keeps the last
  occurrence silently — the exact silence this layer exists to prevent).
  Several sources mapping to one category (X:1) is the point; a category
  name equal to a source name is fine (identity).
- Plumbing: `categories` joins `TOP_LEVEL_KEYS` and the `Config` dataclass
  as `CategoriesConfig(scheme, mapping)`; `load_config` parses
  `_categories(raw)` **before** `_methodology(raw, allowed_categories=…)`
  so the cross-block check below has the category set in hand.
- `methodology.exclusion.types`: every entry must be a **category** (a
  value of `mapping`); the error lists the categories the mapping
  produces. `[]` is allowed. The same check is repeated at **run time** in
  the shared prelude (`ValidationError`), because in-memory callers
  (`compute_frames`, the oracle helpers) build `MethodologyConfig`
  directly and never pass through `load_config`; without it a category the
  mapping no longer produces would exclude nothing, silently.
- Both shipped profiles (`code-2025`, `manuscript`) gain the identity
  `uso-10` block above. `code-2025` keeps `exclusion.types: [RV]`,
  `manuscript` keeps `[]`; both are valid category names under identity.

Run-time rule (`ValidationError`, exit 1 from the CLI): a source type
present in the settlements frame with no entry in `mapping` fails the run,
naming **every** unmapped type with its row count (one run diagnoses the
whole layer). A mapping entry for a type absent from the data is fine —
a scheme may be broader than one city.

**The oracle city is not Delhi.** Its vocabulary (`Planned, UC, JJC, RV,
RUAC, IND`) is not covered by `uso-10`, and the shipped Delhi profiles are
deliberately **not** padded with `UC`/`IND` (that would blunt the
unmapped-type guard on real data). Tests that run the CLI or the oracle
helpers on the fixture city therefore use a **test-only derived profile**,
provided by two helpers in `tests/oraculum_fixtures.py`:
- `oracle_config(base: str) -> Config` — `dataclasses.replace(load_config(base),
  categories=CategoriesConfig(scheme="oracle-6", mapping=<identity over the
  fixture vocabulary>))`. Purely in memory; no file, no pytest fixture
  needed. Used by `methodology_with`, `compute_oracle_frame` and (through
  it) `scripts/generate_production_fixtures.emit_profile` — which runs as a
  plain script outside pytest, so it must not depend on `tmp_path`.
- `oracle_profile_path(base: str, directory: Path) -> Path` — writes the
  same derived profile as YAML into the caller's directory (pytest
  `tmp_path` in `test_cli.py` / `test_oracle_e2e.py`) and returns the path
  for `--config <path>`, which the CLI already accepts.
Test-scaffold changes this implies are listed in § 5; they change
*wiring*, never an expected value.

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
(`pipeline._population_and_exclusion`), immediately **before**
`excluded_ids` — on the settlements frame in `compute_frames`, on the
neighbours frame in `compute`; both carry `type_col`, and `build_neighbors`
neither adds nor removes rows, so the two agree. The prelude then checks
`exclusion.types ⊆ categories_of(mapping)` (run-time rule above) and
`excluded_ids` matches `out_col` against `exclusion.types` instead of
`type_col`.

The mapping **value** is threaded through every path, not only the CLI:
- `compute_frames` gains keyword-only `mapping: Mapping[str, str] | None =
  None` and `scheme: str = "identity"`. `None` means `compute_frames`
  builds `{t: t for t in frame[type_col].unique()}` itself before calling
  `apply_mapping` (existing oracle tests keep their call shape). After its
  final `index_frames` call, `compute_frames` sets
  `result.attrs["categories"] = {"scheme": scheme, "mapping": dict(mapping)}`
  on the frame it returns (a test asserts it; callers that merge the
  result further lose `attrs`, as pandas documents).
- `tests.oraculum_fixtures.compute_oracle_frame(profile, …)` resolves
  `cfg = oracle_config(profile)` first, then passes
  `mapping=cfg.categories.mapping, scheme=cfg.categories.scheme` — i.e. the
  fixture path runs under the (derived) profile's own mapping, so a
  collapsing profile's fixture records the numbers the CLI actually
  produces. Under today's identity profiles this is a no-op. (rev 4
  erratum, final review: the derived mapping is always the identity over
  the fixture vocabulary, so the production fixture is methodology-only and
  does not change with a mapping; vocabulary proofs live in `test_cli.py`'s
  collapse test.)
  `methodology_with(profile, …)` likewise starts from `oracle_config(profile)`
  (it only reads `.methodology`, so its result is unchanged).
- `scripts/generate_production_fixtures.emit_profile` goes through
  `compute_oracle_frame`, so it inherits this with no `tmp_path` and no
  threading changes; `tests/test_fixture_invariants.py`'s bare calls are
  likewise unaffected.
- The shipped YAMLs are never loaded *directly* against the oracle city
  except by the one test that proves the unmapped-type guard.

The neighbours artifact is unchanged and category-free — it is built on
the full universe and stamped with adjacency/barrier only — so a mapping
change never forces an 11-minute `preprocess`. Exclusion still happens at
compute, as in 3A (`stage` × `absent_neighbor` untouched).

Outputs: every PSI CSV / shapefile / joblib row carries `category`
(string) alongside the raw `USO_FINAL`, which is kept as-is;
`missing_population.csv` too. The column name is under the 10-character
shapefile limit. Provenance: `compute` sets
`result.attrs["categories"] = {"scheme": …, "mapping": …}` on each result
frame **immediately before `io.write_outputs`** (mirroring the neighbours
stamp; pandas drops `attrs` across the merges inside `index_frames`, so
stamping earlier would silently vanish). The joblib output carries it;
CSV and shapefile cannot carry `attrs`, so for those formats the record is
one INFO log line (`categories: scheme=… n_categories=…`) and the
`category` column itself. The scheme is not a column.

## 4. Proofs and fixtures

- **No numbers change under the shipped profiles.** `category` is a label,
  not a metric, so the production fixture format (§ 4 of the 3A spec) is
  untouched: `code-2025.csv` and `manuscript.csv` must be byte-identical
  after this cycle — the CI drift guard proves it on every push. The
  real-data run under `code-2025` must still show 0.000e+00 on all 23
  columns.
- **Reference implementation unchanged.** The reference has no category
  concept; its scenarios drop settlements by **id** (`{"RV","IND"}`).
  `test_profiles_match_reference` keeps passing because category exclusion
  resolves to the same ids — which rests on two facts that must be written
  down and pinned: (1) the oracle helpers pass `types=("RV","IND")`, which
  after 3B are *category* names and select the right rows only because the
  derived profile's mapping is the identity; (2) the fixture gives those
  two settlements ids equal to their types. `ORACLE_SCENARIOS`' docstring
  states (1); a fixture-invariant test pins `USO_AREA_U == USO_FINAL` for
  exactly `{RV, IND}` and `categories_of(identity) == fixture vocabulary`.
- **Fixtures regenerated under each profile's own mapping** (via
  `compute_oracle_frame`); today's identity profiles make that a no-op, so
  byte-identity holds — and a future collapsing profile's fixture is
  honest.
- **Vocabulary-change equivalence.** A CLI e2e on the oracle city with a
  5-way collapse (`Planned→planned, UC→unauthorized, RUAC→regularized,
  JJC→jjc, RV→non-urban, IND→non-urban`; scheme `oracle-5`) and
  `exclusion.types: [non-urban]`, derived from `code-2025` (rule `code`,
  `swallowed`, second normalization on), runs both stages and compares the
  written CSV (a) to `compute_oracle_frame("code-2025", types=("RV","IND"),
  stage=<stage>, denom=…)` — the direct "same numbers as today's raw
  exclusion" claim — and (b) to reference block `code/excl_contributing`
  (`post_neighbors`) or `code/excl_removed` (`pre_neighbors`), for both
  denominators, at the CSV round-trip tolerance the existing e2e uses
  (`abs=1e-9`), with the same `health→clinic` column rename. 1e-12 applies
  only to in-memory comparisons. This is the proof that the layer only
  changed the vocabulary.

## 5. Tests

- `tests/test_categories.py`: identity is a no-op on values; X:1 collapse;
  unmapped type raises naming all offenders with counts; mapping broader
  than the data passes; input frame not mutated; `categories_of`.
- `tests/test_config.py` additions: missing `categories:` rejected;
  `exclusion.types` entry that is not a category rejected with the allowed
  list; duplicate mapping key rejected naming the key; `categories.default`
  rejected as reserved; both shipped profiles load with the identity
  scheme and `categories_of(mapping)` == the 10 Delhi types.
- `tests/test_cli.py` additions: `category` column present in CSV, shp and
  joblib outputs and in `missing_population.csv`; the reloaded joblib's
  `attrs["categories"]["scheme"]` equals the profile's; the 5-way collapse
  e2e above; an unmapped type in the layer → exit 1 with the type named
  (run the shipped `code-2025` YAML directly on the oracle city — `UC`/`IND`
  unmapped — which doubles as the proof that the guard works).
- `tests/test_pipeline.py` additions: `compute_frames(mapping=None)` equals
  an explicit identity mapping at 1e-12 on every metric column; an
  `exclusion.types` entry outside `categories_of(mapping)` raises
  `ValidationError` on the in-memory path.
- `tests/test_fixture_invariants.py` addition: the id == type pin for
  `{RV, IND}` and the fixture-vocabulary identity (§ 4).
- **Carried-over tests whose *scaffolding* changes** (expectations do not;
  § 8's stopping rule exempts exactly these): `tests/test_config.py`'s
  `MINIMAL` gains the identity `uso-10` block; `tests/test_cli.py`'s
  `data_dir`/`run` helpers and `tests/test_oracle_e2e.py` pass the derived
  `oracle_profile("code-2025")` / `oracle_profile("manuscript")` path
  instead of the shipped name; `tests/oraculum_fixtures.compute_oracle_frame`
  loads the derived profile. Every assertion and tolerance in those files
  stays as it is. Count: 246 carried over, all green, plus the additions.

## 6. Documentation

- Parent spec `2026-08-27-phase3-refactor-design.md` § 3: the
  required-keys sentence becomes "**Required keys:** `profile`, the whole
  `methodology` block and (from cycle 3B) the whole `categories` block";
  the defaulted list is unchanged. One-line erratum, same PR.
- `docs/methodology-config.md`: new section "Categories" — the two knobs
  (mapping vs exclusion) with **two** worked examples side by side: the
  oracle city's 6-type `oracle-5` (what the tests exercise) and Delhi's
  10-type `urban-5` candidate (`Planned→planned, UAC→unauthorized,
  RUAC→regularized` — the YAML token for WORKPLAN's
  "regularized-unauthorized colonies" — `JJR→resettlement, JJC→jjc, RV→non-urban,
  Industrial→non-urban`, with `UV`, `SDA`, `Other` marked "Raj to decide —
  must be mapped or the run errors"), noting that `UC`/`IND` are the
  fixture's shorthand for `UAC`/`Industrial`; and the procedure for Raj's
  decision (copy profile, write the mapping, set `exclusion.types:
  [non-urban]`, regenerate fixtures, recalculate).
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
deviates, or if a carried-over test needs its **expected value** changed
(the § 5 scaffolding changes — `MINIMAL`, the derived-profile wiring — are
expected and exempt).
