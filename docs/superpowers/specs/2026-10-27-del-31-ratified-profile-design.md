# DEL-31: The Ratified Methodology Profile — Design Spec

**Status:** design approved by owner in chat, 2026-10-26 ("we can use a 5
meter buffer. But ensure that it is configurable … please continue with all
the tickets in the queue").

**Origin:** every methodology decision in the table below is recorded in
the decision logs: `docs/decisions/2026-08-28-raj-methodology-decisions.md`
(28 Aug batch), 5 Sep 2026 (exclusion types, Jira DEL-13/21),
`docs/decisions/2026-10-19-raj-methodology-decisions.md` (overlap lending,
19 Oct), `docs/decisions/2026-10-22-raj-call-decisions.md` (4a →
planned-relative, 22 Oct), and the 26 Oct 2026 Jira DEL-31 comment (5 m
buffer, kept configurable; formally Raj's to ratify alongside the profile).

**What this ticket is.** One new profile,
`delhi_psi/profiles/ratified-2026.yaml` — a copy of `code-2025.yaml` with
the decided values — plus its § 3 registration. **No new code:** every
switch it turns on or off has existed and been proven since cycles 3B–3E
and DEL-63. The profile is the vehicle that hands the paper's decided
rules to the DEL-32 recalculation; `code-2025` stays the frozen record of
the July 2025 numbers, untouched.

## The decided values

| switch | `code-2025` | **`ratified-2026`** | decided |
|---|---|---|---|
| `adjacency.rule` | `bbox` | **`touch`** | 28 Aug (DEL-19/50) — corner-only pairs stop being neighbours |
| `adjacency.max_distance_km` | — | — (absent; `touch` rejects it) | — |
| `barrier.rule` | `global_asymmetric` | **`partial_weighted`** | 28 Aug (DEL-48) — a partly-blocked boundary is discounted, not severed |
| `barrier.buffer_m` | — | **`5`** (required iff `partial_weighted`, > 0) | 26 Oct (owner) — the Oraculum anchor; stays a YAML value, a different buffer is one edit away |
| `barrier.combine` | `any` | `any` | unchanged (never on the table) |
| `overlap.lending` | `whole` | **`outside_receiver`** | 19 Oct (DEL-20) — a neighbour lends \|S_j \ S_i\| |
| `decay` | `inverse_linear` / `centroid` / `km` | same | 28 Aug memo § 7 — recorded, not changed |
| `transform.form` | `none` | `none` | open (DEL-34) — not a decision; the code-2025 value stands until Raj chooses |
| `aggregation.rule` | `mean_minmax` | `mean_minmax` | open (DEL-57) — ditto |
| `roads` | `decayed` | **`eq4_own_only`** | 28 Aug (DEL-49 measured the price first: JJC mean PSI −13 % pop / −10 % density, no ordering flip) |
| `second_normalization` | `minmax` | **`planned_relative`** | 22 Oct 4a (DEL-63) — `norm_psi` + `rel_psi`; 1 = the average planned colony; min-max stays for the appendix |
| `exclusion.types` | `[RV]` | **`[RV, Industrial, Other]`** | 5 Sep (DEL-21/13) — identity categories, no collapse (DEL-29 parked) |
| `exclusion.stage` | `post_neighbors` | `post_neighbors` | 28 Aug |
| `exclusion.absent_neighbor` | `swallowed` | **`contributes`** | 28 Aug, Open Decision A — dropped settlements still lend |
| `categories` | `uso-10` identity | `uso-10` identity | 28 Aug — no collapse; the exclusion line *is* the category decision |
| `outputs.denominators` | `[pop, popdensity]` | `[pop, popdensity]` | 19 Oct — popdensity ratified; both stay reported |
| `outputs.formats` | `[csv, shp, joblib]` | same | unchanged |
| `validate.max_missing_population` | `15` | `15` | carry over; if the DEL-32 run exceeds it the profile value is the knob (a new measurement then), not a code path |
| `paths.neighbors_artifact` | pinned `colonies_neighbors.joblib` | **omitted** → default `colonies_neighbors_ratified-2026.joblib` | § 3 step 1 — a new profile takes the per-profile name |
| `paths.out_dir` | absent | absent | § 3 step 1 — a literal value would ignore `--data-dir` |

Everything else in the file is copied verbatim from `code-2025.yaml`
(paths, layers, services, the inline enum comments — with the now-stale
provenance sentences on `overlap.lending` and `second_normalization`
updated, since they still say "whole ships" / "norm_psi unless Raj
switches").

## Design

### 1. The YAML

`delhi_psi/profiles/ratified-2026.yaml`: full `methodology:` block
(complete-statement contract — every key written, as in `code-2025`), the
`uso-10` identity `categories:` block, the same `paths:`/`layers:`/
`services:`/`outputs:`/`validate:` blocks as `code-2025`, with
`profile: ratified-2026` and the two exclusions above
(no `neighbors_artifact` key, no `out_dir` key).

### 2. Registration (the § 3 step-2 list, applied)

| place | change |
|---|---|
| `SHIPPED` in `tests/test_config.py` | add `"ratified-2026"` (the set-equality test is the red until this) |
| `PROFILES` in `scripts/generate_production_fixtures.py` | append (write order is loop order; appending keeps the diff additive) |
| `PROFILES` in `tests/test_production_fixtures.py` | add (its `test_fixture_is_regenerable` is the second red) |
| `PROFILE_RULES` in `tests/test_profiles_match_reference.py` | **do NOT add** — `ratified-2026` is a **mixed profile**: its reference-pinned switches span the `code` and `ideal` rule-sets (`touch` from `ideal`, `global_asymmetric`'s successor `partial_weighted` from neither, `eq4_own_only` from `ideal`, `contributes` from `ideal`, `planned_relative` from neither) — so no single `RULESETS` row matches all of them. Per § 3 step 2 it is pinned by its production fixtures alone. The plan records this determination explicitly. |

Also verified against the audit: the parametrized "shipped profiles"
tests in `tests/test_config.py` are hard-wired to the two-name list
`["code-2025", "manuscript"]` (historical-record assertions about those
two), so the new profile does not touch them; the two tests that DO walk
every profile file — `test_every_profile_declares_an_aggregation_rule`
(globs `*.yaml`) and `test_shipped_profiles_let_the_data_dir_drive_out_dir`
— pass by construction (full `methodology:` block; no literal `out_dir`).
`tests/test_cli.SHIPPED_PROFILES` stays at the two production profiles:
the ratified profile's `uso-10` mapping cannot run on the fixture cities'
vocabularies, which is exactly why it never enters that list (the
collapse-procedure precedent).

### 3. The absent-category design point (resolved)

The profile's own `exclusion.types: [RV, Industrial, Other]` names Delhi
categories; the fixture cities' vocabularies (oracle-6: `Planned/UC/JJC/
RV/RUAC/IND`, messy-2: `Planned/RV`) do not contain `Industrial` or
`Other`. Three facts make that a non-collision, and the spec pins all
three:

1. **The shipped YAML loads.** The loader cross-checks
   `exclusion.types` against the *profile's own* mapping's values
   (`delhi_psi/config.py`), and the `uso-10` identity mapping produces
   exactly those categories.
2. **Fixture generation never reaches the profile's own types.**
   `emit_profile` → `compute_oracle_frame(types=scenario.exclusion_types,
   stage=…)` → `methodology_with` REPLACES `exclusion.types` with the
   city scenario's tuple for every scenario — including the empty tuple of
   the baseline scenarios — while keeping the profile's
   `absent_neighbor: contributes` (the one exclusion property that is a
   rule-set property and therefore travels). `oracle_config`'s mapping
   swap is an in-memory `dataclasses.replace` with no re-validation, so
   the Delhi names stay dormant in the `Config` object. The production
   fixtures therefore pin **methodology only** — adjacency, barrier,
   lending, roads, normalization, `absent_neighbor`, denominators — over
   each city's existing exclusion scenarios, exactly the § 3 step-3
   contract.
3. **The invariant that keeps it true:** no path may hand a
   `ratified-2026` methodology with its OWN types to a fixture-city
   `compute_frames` — the pipeline re-cross-checks types against the
   mapping's categories (`delhi_psi/pipeline.py`, "not categories
   produced by categories.mapping") and would raise. The implementation
   audits every caller of `compute_frames` / `compute_oracle_frame` /
   `oracle_profile_path` that could receive `ratified-2026`
   un-overridden; the expected answer is "none" (fixture generation is
   scenario-driven; no test or script uses `ratified-2026` as a
   `methodology_overrides` base — variants use `code-2025`/`manuscript`
   bases and this ticket adds no variants). If the audit finds such a
   path, the fix is the scenario override at that call site, never a
   weakening of the cross-check.

### 4. Fixture regeneration

`uv run python scripts/generate_production_fixtures.py` adds
`tests/fixtures/oraculum/production/ratified-2026.csv` and
`tests/fixtures/messy/production/ratified-2026.csv`; every pre-existing
fixture file stays byte-identical (the CI drift guard re-runs the
generator and would fail the build otherwise).

- **`planned_relative` precondition (DEL-63 guards):** for EVERY city ×
  scenario × denominator, the reported frame must contain ≥ 1 row whose
  category is `Planned` with mean `norm_psi` > 0, or
  `index.overall_psi` raises. Both fixture cities keep `Planned`
  settlements in every scenario, and the min-max universes are non-
  degenerate — the generator run is the proof, and if any scenario did
  raise, that is a spec violation to resolve in review, not a silent
  fixture shape.
- **The diff is the artefact:** `ratified-2026.csv` against
  `code-2025.csv` on the 7-settlement oracle city IS the methodology
  change, settlement by settlement. It is read during implementation and
  summarised in the PR description (which settlements move, which way,
  and which columns the `planned_relative` addition adds —
  `rel_psi` rows appear where `code-2025`'s `minmax` has only `norm_psi`).

### 5. The value-pin test (the TDD anchor)

A new test in `tests/test_config.py` loads `ratified-2026` and asserts
the ENTIRE decided table of the previous section, value for value —
including `buffer_m == 5.0` (the configurable 5 m), the default
`neighbors_artifact` rendered at load time
(`config.py:344`/`718`: `"colonies_neighbors_ratified-2026.joblib"`),
no literal `out_dir` (so `paths.out_dir` follows `--data-dir`, like the
existing data-dir-drives-out_dir test), `second_normalization ==
"planned_relative"`, and the exclusion triple. Written red first (the
profile does not exist yet); the YAML is what turns it green.

### 6. Docs

- `docs/methodology-config.md`: the § 1 profile table grows to three
  rows (add `ratified-2026` — "Raj's ratified rules: the profile the
  Phase 4 recalculation (DEL-32) runs; the production default until Phase
  4 deliberately switches it"); the switch table's **ratified** column
  stops saying "Raj to answer / to choose" for the four now-decided rows
  (lending, second normalization, denominators, buffer) and points at the
  decision docs; a note that `ratified-2026` is a mixed profile (absent
  from `PROFILE_RULES`, § 3 step 2); § 2 worked example 2 already shows
  the profile's shape and gains a pointer to the shipped file.
- `code-2025.yaml` itself: untouched (frozen record).
- `WORKPLAN.md`: synced at the end of the whole cycle, not in this PR.

## Out of scope (explicit)

- DEL-32 (the recalculation: fresh `preprocess` — `touch` makes a new
  neighbours artifact that cannot be reused — plus `compute` against
  `~/delhi_data`) and DEL-33 (figures) run AFTER this merge.
- DEL-36/38/39 and the other Phase 6 sweeps: they derive their band/decay
  variants from this profile, later.
- Category collapse (DEL-29, parked), the `transform`/`aggregation`
  choices (still open; the profile carries the code-2025 values by
  default, and a later decision is a later profile edit).
- The real-data `preprocess`/`compute` of `ratified-2026` — this ticket
  ships the profile + fixtures; a smoke that `--config ratified-2026`
  LOADS and resolves its per-profile artifact name is in scope, a full
  run is not.

## Acceptance

1. TDD: the value-pin test (§ 5) and the two registration tests
   (`SHIPPED` set, fixture-regenerable) watch-fail before the YAML and
   fixtures exist, and are green after.
2. Full suite green: `uv run pytest -q -W error` — including the drift
   guard's append-only behaviour on every pre-existing fixture.
3. CI green on the PR.
4. The fixture diff digest (§ 4) is in the PR description.
5. The `/ship` record: plan + spec/plan review round (finder + judge,
   stopping rule zero-Critical/zero-Important), both test outputs in the
   PR.
