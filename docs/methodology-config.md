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

Also in the block: `exclusion.types` (which `USO_FINAL` types are dropped;
`[RV]` today), `exclusion.stage` (`post_neighbors` = today: neighbours are
built on the full universe, exclusion happens at compute), `barrier.combine`.

**Reserved — the loader refuses these and tells you why:**
`barrier.rule: partial_weighted` (needs a reference rule and a hand anchor
first — see memo § 2), `outputs.denominators: one` (reference does not model
it), and the key `exclusion.minmax_universe` (Open Decision A.2 — no knob
anywhere yet). If Raj chooses one of these, it is a 3C ticket, not a YAML edit.

## 2. Procedure for a decision

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

## 3. What each proof guards

- `tests/test_production_fixtures.py` — every profile's numbers on the
  oracle city, byte-for-byte; the CI drift guard regenerates and diffs them
  on every push, so an accidental edit cannot pass.
- `tests/test_profiles_match_reference.py` — production == the independent
  reference implementation at 1e-12 for the profiles in `PROFILE_RULES`.
- `tests/test_manuscript_anchors.py` — `manuscript` == the hand-ratified
  worksheet (`docs/oracle/derivation-worksheet.md`).
- `scripts/verify_against_baseline.py --config code-2025` — real data ==
  July 2025 baseline, zero deviation.

## 4. Things that are code, not config

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
   already exist. Category mappings (`categories:` block, 10/8/5/4
settlement types) are cycle 3B (DEL-17).
