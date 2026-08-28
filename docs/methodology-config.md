# Changing the methodology — the config procedure

Since Phase 3A (27 Aug 2026) every methodology choice in the PSI pipeline
is a value in a YAML **profile**, not code. This page is the operating
procedure for turning a decision (typically one of Raj's) into a change.
The design rationale lives in
`docs/superpowers/specs/2026-08-27-phase3-refactor-design.md` (§ 3 schema,
§ 4 fixtures, § 8 decisions); this page only tells you what to do.

## 1. The switches

Profiles live in `delhi_psi/profiles/`. Two ship:

| profile | meaning |
|---|---|
| `code-2025` | **Today's behaviour** — the rules that produced the July 2025 numbers. The production default. Treat it as a frozen record. |
| `manuscript` | The paper's ideal rules (Eq. 1–4 as written). Proven against the independent reference implementation and the hand-ratified worksheet. |

The `methodology:` block of `code-2025.yaml` lists every switch with its
allowed values as an inline comment. The ones on the table with Raj:

| switch | today (`code-2025`) | paper (`manuscript`) | decides |
|---|---|---|---|
| `adjacency.rule` | `bbox` | `touch` | suggested-fixes memo § 1 (DEL-19) — bbox adjacency invents neighbours |
| `barrier.rule` | `global_asymmetric` | `pairwise` | memo § 2 (DEL-22) — sever the crossing pair only |
| `roads` | `decayed` | `eq4_own_only` | memo § 3 (DEL-22) — Eq. 4 has no neighbour term |
| `second_normalization` | `true` | `false` | memo § 4 (DEL-22) — `norm_psi` is not in Eq. 1 |
| `outputs.denominators` | `[pop, popdensity]` | `[pop]` | memo "Popdensity denominator" (DEL-22) |
| `exclusion.absent_neighbor` | `swallowed` | `contributes` | memo § 5 / Open Decision A (DEL-13, DEL-21) — do dropped settlements still lend services? |
| `decay.distance_unit` | `km` | (manuscript silent) | memo § 7 |

Also in the block: `exclusion.types` (which **categories** are dropped —
category names, not raw `USO_FINAL` types; `[RV]` today, which is a category
only because the shipped mapping is the identity — see § 2),
`exclusion.stage` (`post_neighbors` = today: neighbours are built on the full
universe, exclusion happens at compute), `barrier.combine`.

**Reserved — the loader refuses these and tells you why:**
`barrier.rule: partial_weighted` (needs a reference rule and a hand anchor
first — see memo § 2), `outputs.denominators: one` (reference does not model
it), and the key `exclusion.minmax_universe` (Open Decision A.2 — no knob
anywhere yet). If Raj chooses one of these, it is a 3C ticket, not a YAML edit.

## 2. Categories — the settlement-type mapping

Since cycle 3B (27 Aug 2026) a profile also states, in full, how the
layer's **source types** collapse into the **categories** the analysis
uses. `categories:` is required in every profile, like `methodology:`.

Two knobs, and they do different jobs:

| knob | job |
|---|---|
| `categories.mapping` | source type → category. 1:1 (identity) or X:1 (several sources into one category). This is the vocabulary. |
| `methodology.exclusion.types` | which **categories** are dropped from the reported frame. Written in the mapping's category names, never in raw source types. |

`categories.scheme` is a free-form name for the mapping. It is recorded in
the joblib output's `attrs` and in one INFO line per run
(`categories: scheme=… n_categories=…`); it is never a column. Every output
— CSV, shapefile, joblib — and `missing_population.csv` carry a `category`
column next to the raw `USO_FINAL`, which is kept as-is.

**An unmapped source type fails the run.** If the layer carries a type with
no entry in `mapping`, `compute` exits 1 and names every offender with its
row count, so one run diagnoses the whole layer. There is deliberately no
catch-all: `categories.default` is rejected at load. A mapping entry for a
type that is *absent* from the data is fine — a scheme may be broader than
one city. Duplicate keys in the YAML are also rejected (PyYAML keeps the
last one silently, which is the same failure wearing a different hat).

### Worked example 1 — the oracle city, six types into five

This is what the test suite exercises
(`tests/test_cli.py::test_five_way_collapse_reproduces_raw_type_exclusion`).
The 7-settlement fixture city carries `Planned, UC, JJC, RV, RUAC, IND`
(`UC` and `IND` are the fixture's shorthand for `UAC` and `Industrial`):

```yaml
categories:
  scheme: oracle-5
  mapping:
    Planned: planned
    UC: unauthorized
    RUAC: regularized
    JJC: jjc
    RV: non-urban
    IND: non-urban
methodology:
  exclusion:
    types: [non-urban]
```

Six source types, five categories, and the run drops `non-urban` — which is
exactly the two settlements the old raw `exclusion.types: [RV, IND]`
dropped. The test proves those two runs are numerically identical, and that
both match the independent reference implementation. That equivalence is
the whole claim of this layer: **it changes the vocabulary, not the
numbers.**

### Worked example 2 — Delhi, ten types into the Phase 4 candidate

The workshop's working candidate (WORKPLAN DEL-29): planned /
unauthorized / regularized-unauthorized / resettlement / JJC, with the
non-urban types dropped. In YAML, `regularized` is the token for WORKPLAN's
"regularized-unauthorized colonies":

```yaml
categories:
  scheme: urban-5
  mapping:
    Planned: planned
    UAC: unauthorized
    RUAC: regularized
    JJR: resettlement
    JJC: jjc
    RV: non-urban
    Industrial: non-urban
    # UV: ?            # Raj to decide — must be mapped or the run errors
    # SDA: ?           # Raj to decide — must be mapped or the run errors
    # Other: ?         # Raj to decide — must be mapped or the run errors
methodology:
  exclusion:
    types: [non-urban]
```

**This profile does not ship, and as written it would not run:** `UV` (138
rows), `SDA` (86) and `Other` (33) are real types on the real layer with no
entry, so `compute` would exit 1 naming all three. That is the design —
they are open questions (DEL-29 explicitly flags SDA), and the pipeline
refuses to guess. Counts and provenance for all ten types:
`docs/data/uso_final_vocabulary.md`.

### Procedure for Raj's decision (DEL-31)

1. Copy `delhi_psi/profiles/code-2025.yaml` to
   `delhi_psi/profiles/urban-5.yaml`, set `profile: urban-5`, and write the
   agreed `categories:` block — every one of the 10 source types mapped.
2. Set `methodology.exclusion.types: [non-urban]` (category names).
3. **Do not add `urban-5` to `tests/test_cli.SHIPPED_PROFILES` or
   `scripts/generate_production_fixtures.PROFILES`.** `compute_oracle_frame`
   always resolves `oracle_config(profile)` — the derived helpers swap in
   the oracle-6 identity mapping but keep `base`'s shipped
   `methodology.exclusion.types` verbatim — and `emit_profile` supplies
   `types` from `ORACLE_SCENARIOS`; `urban-5`'s `exclusion.types:
   [non-urban]` is not a category the oracle-6 identity produces (its
   vocabulary is `Planned, UC, JJC, RV, RUAC, IND`), so `oracle_config`/
   `oracle_profile_path` fail to load it. Even for a profile whose
   exclusion categories *did* exist in that vocabulary, the regenerated
   `urban-5.csv` would still be byte-identical to `code-2025.csv` — the
   production fixture pins METHODOLOGY only and is EXPECTED to be
   unchanged by a mapping change, never the categorization decision.
   Instead, prove the vocabulary claim with a `collapse_profile_path`-style
   test in `tests/test_cli.py`: copy the pattern of
   `test_five_way_collapse_reproduces_raw_type_exclusion`'s hand-written
   `ORACLE_5` block, adapted to `urban-5`'s mapping and dropped category —
   a new profile's mapping gets a sibling test, not a fixture diff.
4. Run the suite, then the real data (§ 3 step 5) — `--config urban-5`
   resolves the shipped YAML directly and needs none of the § 3 step 2
   test-list registrations above. This is the DEL-32 recalculation; no
   code changes.

## 3. Procedure for a decision

Work on a branch; `main` requires the CI check.

1. **Create a new profile rather than editing `code-2025`.** Copy
   `delhi_psi/profiles/code-2025.yaml` to e.g.
   `delhi_psi/profiles/ratified-2026.yaml`, set `profile: ratified-2026`,
   and change the decided values. `methodology:` must be complete (every
   key written out); everything else may be omitted and takes the
   `code-2025` defaults — with one exception: `paths.neighbors_artifact`
   defaults to `colonies_neighbors_<profile>.joblib`, whereas `code-2025`
   pins the legacy name `colonies_neighbors.joblib` (so the real-data proof's
   filename never changed). Leave the key out of a new profile and you get
   the per-profile name, which is what you want. Do **not** add a literal
   `paths.out_dir` — it would ignore `--data-dir`.
2. **Register it.** Add the name to `PROFILES` in
   `scripts/generate_production_fixtures.py` and
   `tests/test_production_fixtures.py`. If it should be checked against the
   reference implementation, add it to `PROFILE_RULES` in
   `tests/test_profiles_match_reference.py` mapped to `"code"` or `"ideal"`
   — only if *all* its reference-pinned switches match that rule-set;
   a mixed profile is pinned by its production fixture alone.
3. **Regenerate the fixtures:** `uv run python scripts/generate_production_fixtures.py`.
   A new file `tests/fixtures/oraculum/production/<profile>.csv` appears.
   The diff against `code-2025.csv` *is* the methodology change on the
   7-settlement oracle city — read it; it is the discussion artefact.
   `code-2025.csv` must not change.
4. **Run the suite:** `uv run pytest -q -W error`. Commit the YAML, the
   fixture and the test edits together.
5. **Real data** (long; run in the background; outputs only under `--out-dir`):
   ```
   uv run delhi-psi preprocess --config ratified-2026 --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_ratified
   uv run delhi-psi compute    --config ratified-2026 --data-dir ~/delhi_data --out-dir ~/delhi_data/phase3_ratified
   ```
   Both stages must use the same profile: the neighbours artifact is
   stamped with the methodology that built it and `compute` refuses a
   mismatch ("re-run preprocess"). A new profile gets its own artifact
   name automatically (`colonies_neighbors_<profile>.joblib`; `code-2025`
   alone keeps the legacy `colonies_neighbors.joblib`).
   **Outputs** land directly in `--out-dir`: one set per entry of
   `outputs.denominators`, named by `outputs.name_template` (e.g.
   `delhi_psi_ratified-2026_pop_2020.csv`, plus `.shp`/`.joblib` if listed
   in `outputs.formats`), a `missing_population.csv`, the neighbours
   artifact, and the `*.dedup.gpkg` cache files.
   `scripts/verify_against_baseline.py` compares to July 2025 and is only
   meaningful for `code-2025`; for a new profile the comparison of interest
   is against the `code-2025` outputs in a separate directory.
6. **Record it:** WORKPLAN Open Decisions + the DEL-13 ticket; the
   production default stays `code-2025` until Phase 4 recalculation
   (DEL-32) switches it deliberately.

## 4. What each proof guards

- `tests/test_production_fixtures.py` — every profile's numbers on the
  oracle city, byte-for-byte; the CI drift guard regenerates and diffs them
  on every push, so an accidental edit cannot pass.
- `tests/test_profiles_match_reference.py` — production == the independent
  reference implementation at 1e-12 for the profiles in `PROFILE_RULES`.
- `tests/test_manuscript_anchors.py` — `manuscript` == the hand-ratified
  worksheet (`docs/oracle/derivation-worksheet.md`).
- `scripts/verify_against_baseline.py --config code-2025` — real data ==
  July 2025 baseline, zero deviation.

## 5. Things that are code, not config

New *values* for a reference-pinned switch (anything not listed in the YAML
comments) require, in order — this is the 3C cycle:
1. add `{new_value: reference_knob}` to `REFERENCE_KNOBS` in
   `delhi_psi/config.py` — the enums the YAML loader accepts are generated
   from that one table, so nothing else makes the value loadable;
2. implement the matching branch in `tests/reference_impl.py::compute_city`
   (`test_every_mapped_knob_is_one_the_reference_actually_implements`
   fails until you do);
3. a hand anchor in `docs/oracle/derivation-worksheet.md`;
4. regenerate `tests/fixtures/oraculum/expected_values.csv`
   (`uv run python tests/reference_impl.py`);
5. the production implementation in `delhi_psi/` if the branch does not
   already exist.
