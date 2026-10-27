# DEL-31: The ratified-2026 profile — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note (ship.md deviation rule):** no new code — one YAML, three
> one-line registrations, two regenerated fixtures, one doc page. The lead
> implements inline under strict TDD (review-budget rule). Plan and code
> each get one scaled review round.

**Goal:** Ship `ratified-2026` — Raj's ratified rules as one profile — so the
DEL-32 recalculation has something to run, without moving a single number in
`code-2025` or `manuscript`.

**Architecture:** `delhi_psi/profiles/ratified-2026.yaml` is a copy of
`code-2025.yaml` with the decided values swapped (adjacency `touch`; barrier
`partial_weighted` + `buffer_m: 5`; overlap `outside_receiver`; roads
`eq4_own_only`; `second_normalization: planned_relative`; exclusion
`[RV, Industrial, Other]` / `post_neighbors` / `contributes`), the pinned
`neighbors_artifact` dropped so the default renders
`colonies_neighbors_ratified-2026.joblib`, and the two now-stale inline
comments updated. Registration is three one-line additions: `SHIPPED`
(tests/test_config.py), `PROFILES` (scripts/generate_production_fixtures.py),
`PROFILES` (tests/test_production_fixtures.py). `PROFILE_RULES`
(tests/test_profiles_match_reference.py) and `tests/test_cli.SHIPPED_PROFILES`
deliberately stay at two entries; the profile is pinned by its two production
fixtures alone — it is a **mixed profile**.

**Tech Stack:** Python (uv), PyYAML, pytest, pandas; fixture generator
`scripts/generate_production_fixtures.py`; doc page `docs/methodology-config.md`.

**Spec:** `docs/superpowers/specs/2026-10-27-del-31-ratified-profile-design.md`
(owner-approved 26 Oct 2026). Origin: the four decision docs
(`docs/decisions/2026-08-28-…`, `2026-10-19-…`, `2026-10-22-…`, plus the 5 Sep
exclusion-types record) and the 26 Oct 2026 Jira DEL-31 comment (5 m buffer,
kept configurable).

## Global Constraints

- **`code-2025.yaml` and `manuscript.yaml` are byte-identical** — frozen
  records. Every pre-existing fixture file (both cities' `expected_values.csv`,
  all `production/*.csv`, the variants CSVs, the GeoJSONs) stays
  **byte-identical**: the CI drift guard re-runs the generator and fails the
  build on any non-appenditive change.
- **No new code, no new enum values.** Every value the new YAML carries is an
  existing reference-pinned member of `config.REFERENCE_KNOBS`, landed and
  proven in cycles 3B–3E and DEL-63. If a needed knob does not exist, that is
  a spec violation — stop and re-ask; do not add a knob in this ticket.
- **`PROFILE_RULES` stays two entries.** `ratified-2026` is a **mixed
  profile**: `touch`, `eq4_own_only`, `contributes` sit on the reference
  `ideal` row, but `partial_weighted` and `planned_relative` sit on neither
  `code` nor `ideal` — and neither row pins `overlap.lending` at all, so
  `outside_receiver` sits on neither row either, while the whole `decay`
  block is carried by defaults and pinned by neither. No single `RULESETS`
  row can match all of the profile's values. Per
  `docs/methodology-config.md` § 3 step 2 it is pinned by its production
  fixtures only. The determination is recorded in the review record below.
- **`tests/test_cli.SHIPPED_PROFILES` stays two.** Its `run()` helper swaps a
  shipped profile name for the derived oracle profile
  (`tests/oraculum_fixtures.oracle_profile_path`), whose mapping is the
  fixture city's identity; the derived profile's `exclusion.types` must be
  names in that city's mapping, and `Industrial`/`Other` are not
  (oraculum: `Planned/UC/JJC/RV/RUAC/IND`, messy: `Planned/RV`). The shipped
  `uso-10` mapping also does not cover `UC`/`IND`, so the shipped profile
  already refuses to run on the city directly — that refusal is itself the
  guard under test (`test_unmapped_settlement_type_exits_1`).
- **TDD:** the three watched-red tests (the `SHIPPED` set-equality, the
  value-pin, the fixture-regenerable check) are written and watched failing
  BEFORE the YAML exists; they go green only on the Task 2–3 implementation.
- **The fixture diff is the artefact.** The oraculum
  `ratified-2026.csv` vs `code-2025.csv` difference is read at implementation
  time and summarised in the PR description (Task 3, Step 6).

## The measured facts this plan argues from

- **The per-profile artifact default renders at load time.**
  `config.py:344` `DEFAULT_PATHS`
  `"neighbors_artifact": "colonies_neighbors_{profile}.joblib"`; `_paths`
  (config.py:718) formats the profile name in. So `load_config("ratified-2026")`
  yields `paths.neighbors_artifact == "colonies_neighbors_ratified-2026.joblib"`,
  and with no literal `out_dir` in the YAML, `paths.out_dir` follows
  `--data-dir` (covered by construction via
  `test_shipped_profiles_let_the_data_dir_drive_out_dir` — that test is
  parametrized over the two production profiles, and the value-pin asserts
  the same property for the new one).
- **Two independent cross-checks on exclusion types.** The loader
  cross-checks `exclusion.types` against the profile's OWN mapping's values
  (config.py:581); the pipeline re-cross-checks them at `compute_frames`
  against the mapping it is handed (pipeline.py:287–290, "not categories
  produced by categories.mapping"). Fixture generation always hands the city's
  identity mapping with the SCENARIO's types tuple in place of the profile's
  (`methodology_with`, tests/oraculum_fixtures.py:100+), so `Industrial` /
  `Other` never reach the pipeline check; the one exclusion property that
  travels is `absent_neighbor: contributes` (a rule-set property, part of the
  block). `oracle_config` swaps the mapping by in-memory
  `dataclasses.replace` with no re-validation, so the dormant Delhi names are
  inert in the `Config` object.
- **The `planned_relative` guard** (`delhi_psi/index.overall_psi`) needs ≥ 1
  row of category `Planned` with mean `norm_psi` > 0 per reported frame. Both
  fixture cities keep `Planned` settlements in every scenario (oraculum
  scenarios drop at most `{RV, IND}`; messy at most `{U, N}` — `U` is the
  no-population drop, not an exclusion type; certifiably measured by the
  finder: Planned rows survive all 16 city × scenario × denom frames, Planned
  mean `norm_psi` in `0.246…0.744`), so the generator run cannot trip it —
  the run itself is the proof.
- **The `planned_relative` fixture shape:** `metric_columns`
  (scripts/generate_production_fixtures.py:70) appends a `rel_psi` metric row
  for every (scenario, denom, settlement) under `planned_relative`, on top of
  `norm_psi` — that is the row-count difference vs `code-2025.csv`'s `minmax`.
- **The `SHIPPED` list** (tests/test_config.py:61) is a plain list; the
  set-equality test sorts, so insertion position is free. The parametrized
  "shipped profiles" tests in that file are hard-wired to the two-name list
  `["code-2025", "manuscript"]` (historical-record assertions about those two)
  and do not run on the new profile. The only every-profile tests —
  `test_every_profile_declares_an_aggregation_rule` (globs `*.yaml`) and the
  data-dir-drives-out-dir test — pass by construction (full `methodology:`
  block, no literal `out_dir`).
- **Dataclass shapes for the value-pin** (config.py): `ExclusionConfig.types`
  is a `tuple`; `AdjacencyConfig.max_distance_km: float | None`;
  `BarrierConfig.buffer_m: float | None`; `OutputsConfig.denominators` is a
  tuple of the `Denominator` StrEnum (compares equal to strings).

---

### Task 1: RED — the three watched failures

**Files:**
- Modify: `tests/test_config.py` (`SHIPPED` list, line 61; new value-pin test)
- Modify: `tests/test_production_fixtures.py` (`PROFILES`, line 19)

- [ ] **Step 1: Add the name to `SHIPPED`.** In `tests/test_config.py`, insert
  `"ratified-2026",` into the `SHIPPED` list (place it after
  `"manuscript",`; order is irrelevant to the test).

- [ ] **Step 2: Write the value-pin test** at the end of `tests/test_config.py`
  (append — keeps the diff additive):

  ```python
  def test_ratified_profile_pins_the_decided_values(tmp_path):
      """One assertion per decided value in the spec's table (the 26 Oct
      instruction, value for value). The two still-open rows —
      transform.form and aggregation.rule — carry the code-2025 values by
      default and stay so until Raj decides; a later decision is a one-line
      profile edit, not a new profile."""
      cfg = load_config("ratified-2026", data_dir=str(tmp_path))
      m = cfg.methodology
      assert m.adjacency.rule == "touch"
      assert m.adjacency.max_distance_km is None
      assert m.barrier.rule == "partial_weighted"
      assert m.barrier.buffer_m == 5.0
      assert m.barrier.combine == "any"
      assert m.overlap.lending == "outside_receiver"
      assert (m.decay.form, m.decay.distance, m.decay.distance_unit) == \
          ("inverse_linear", "centroid", "km")
      assert m.transform.form == "none"
      assert m.aggregation.rule == "mean_minmax"
      assert m.roads == "eq4_own_only"
      assert m.second_normalization == "planned_relative"
      assert m.exclusion.types == ("RV", "Industrial", "Other")
      assert m.exclusion.stage == "post_neighbors"
      assert m.exclusion.absent_neighbor == "contributes"
      assert cfg.categories.scheme == "uso-10"
      assert set(cfg.categories.mapping) == DELHI_TYPES
      assert all(source == category
                 for source, category in cfg.categories.mapping.items())
      assert cfg.outputs.denominators == ("pop", "popdensity")
      assert cfg.outputs.formats == ("csv", "shp", "joblib")
      assert cfg.validate.max_missing_population == 15
      assert cfg.paths.neighbors_artifact == \
          "colonies_neighbors_ratified-2026.joblib"
      assert cfg.paths.out_dir == tmp_path
  ```

- [ ] **Step 3: Add the profile to the fixture-regenerable list.** Append
  `"ratified-2026"` to `PROFILES` in `tests/test_production_fixtures.py`
  (after `"services-contested"` — loop order determines write order;
  appending keeps the diff additive).

- [ ] **Step 4: Watch all three failures.**

  ```bash
  uv run pytest tests/test_config.py::test_the_shipped_profiles_are_exactly_these -q
  # Expected: FAIL — shipped_profiles() globs 17 YAMLs; SHIPPED now lists 18
  uv run pytest tests/test_config.py::test_ratified_profile_pins_the_decided_values -q
  # Expected: FAIL — ConfigError: unknown profile 'ratified-2026';
  #   shipped profiles: [...] (config.py _profile_path)
  uv run pytest tests/test_production_fixtures.py -k "ratified and Regenerable" -q
  # Expected: 2 FAILs (one per city) — "missing committed fixture
  #   .../production/ratified-2026.csv"
  # NB: plain `-k ratified` matches FOUR tests — test_fixture_has_the_spec_shape
  # also parametrizes over PROFILES and fails here with FileNotFoundError on
  # read_bytes(). That is EXPECTED in RED; narrow the selector as above.
  ```

  Watch all three before moving on: TDD says the Task 2–3 implementation is
  what turns them green, and any failure that does not match these exact
  shapes means the plan misread the code — stop and re-derive.

---

### Task 2: The YAML

**Files:**
- Create: `delhi_psi/profiles/ratified-2026.yaml`

- [ ] **Step 1: Write the profile** (a copy of `code-2025.yaml` with the
  decided values; the now-stale provenance sentences on `overlap.lending` and
  `second_normalization` updated; the `neighbors_artifact` pin dropped):

  ```yaml
  profile: ratified-2026
  crs: {epsg: 7760}
  paths:
    data_dir: ~/delhi_data            # overridable: --data-dir, DELHI_DATA_DIR
                                      # out_dir: omitted on purpose — it defaults
                                      # to data_dir, as today; overridable with
                                      # --out-dir. A literal value here would
                                      # ignore --data-dir.
                                      # neighbors_artifact: omitted on purpose —
                                      # the default renders the profile name
                                      # (colonies_neighbors_ratified-2026.joblib),
                                      # so a fresh preprocess writes a fresh
                                      # artifact beside code-2025's, never
                                      # overwriting it
  layers:
    settlements: {path: uso_update_sep2021/uso_update_sep2021.shp,
                  id_col: USO_AREA_U, type_col: USO_FINAL}
    population:  {path: pop_colony_wp_2020_jjc_adjusted.csv,
                  id_col: uso_area_u, value_col: population,
                  missing: drop}      # drop | error  (see below)
    bounds: delhi_bounds_buffer/delhi_bounds_buffer.shp
    ndmc_center: ndmc_center7760/ndmc_center7760.shp   # -> ndmc_dist_km column
    barriers: {canal: Barrier_Clip/Canal/Canal.shp,
               railway: Barrier_Clip/Railway/Railway_Line.shp,
               drain: Barrier_Clip/Drain/Major_Drain.shp}
  services:
    point: {bank: Public Services/Banking/Banking.shp,
            health: Public Services/Health/Health.shp,
            police: Public Services/Police/Police Station.shp,
            ration: Public Services/Ration/Ration.shp,
            school: Public Services/School/schools7760.shp,
            transport: Public Services/Transport/Transport.shp}
    line:  {road: Public Services/Major Road/Road.shp}   # amount column: road_length
  categories:                         # source type -> category (spec 3B 2)
    scheme: uso-10                    # free-form name; stamped into the joblib output
    mapping:                          # IDENTITY: the 10 USO_FINAL types, uncollapsed
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
    adjacency:
      rule: touch                     # bbox | touch | within_distance — Raj,
                                      # 28 Aug (DEL-19/50): corner-only pairs
                                      # stop being neighbours. max_distance_km:
                                      # required iff rule is within_distance
                                      # (>= 0 km, polygon-to-polygon)
    barrier:
      rule: partial_weighted          # global_asymmetric | pairwise |
                                      # partial_weighted — Raj, 28 Aug (DEL-48):
                                      # a partly-blocked boundary is discounted,
                                      # w = 1 - L_blocked/L_shared, not severed
      buffer_m: 5                     # required iff partial_weighted (> 0 m).
                                      # Raj, 26 Oct: the Oraculum anchor, KEPT
                                      # CONFIGURABLE — a different buffer is a
                                      # one-line profile edit (DEL-48)
      combine: any                    # any | [layer names]; which flags OR into
                                      # `barrier`; unchanged
    overlap:
      lending: outside_receiver       # whole | outside_receiver — Raj,
                                      # 19 Oct 2026 (DEL-20): a neighbour lends
                                      # |S_j \ S_i|, so a service in the overlap
                                      # of two colonies is not counted a second
                                      # time through the neighbour term.
                                      # counting: reserved — a service inside k
                                      # overlapping colonies counts for each of
                                      # the k (ratified 28 Aug 2026; no knob)
    decay:
      form: inverse_linear            # inverse_linear | none | inverse_power
                                      # | exponential — 28 Aug memo § 7:
                                      # recorded, not changed
      distance: centroid              # centroid | boundary — unchanged
      distance_unit: km
    transform:
      form: none                      # none | log1p | cbrt (DEL-34) — STILL
                                      # OPEN; the code-2025 value stands
    aggregation:
      rule: mean_minmax               # mean_minmax | mean_rank (DEL-57) —
                                      # STILL OPEN; the code-2025 value stands
    roads: eq4_own_only               # decayed | eq4_own_only — Raj, 28 Aug
                                      # (DEL-49 measured the price first: JJC
                                      # mean PSI -13 % pop / -10 % density, no
                                      # ordering flip)
    second_normalization: planned_relative  # minmax | none | planned_relative —
                                      # Raj, 22 Oct 2026, 4a (DEL-63): the
                                      # reported scale, 1 = the average planned
                                      # colony; min-max stays in the appendix
    exclusion:
      types: [RV, Industrial, Other]  # CATEGORY names — Raj, 5 Sep (DEL-21/13);
                                      # identity mapping kept, no collapse
                                      # (DEL-29 parked)
      stage: post_neighbors           # post_neighbors | pre_neighbors — 28 Aug:
                                      # neighbours built on the full universe,
                                      # exclusion at compute
      absent_neighbor: contributes    # swallowed | contributes — 28 Aug, Open
                                      # Decision A: dropped settlements still
                                      # lend; minmax_universe: reserved (spec 9)
  validate:
    max_missing_population: 15        # today's real-data count; the KNOB (not a
                                      # code path) if a run exceeds it
  outputs:
    denominators: [pop, popdensity]   # 19 Oct: popdensity ratified; both
                                      # stay reported
    formats: [csv, shp, joblib]
    name_template: "delhi_psi_{profile}_{denominator}_2020"
  ```

- [ ] **Step 2: Watch green.**

  ```bash
  uv run pytest tests/test_config.py -q
  # Expected: all pass — the set-equality test (18 == 18), the value-pin
  # test, and every other config test (the all-YAML glob test now sees the
  # new file and passes by construction)
  ```

- [ ] **Step 3: Load smoke — the per-profile artifact name resolves.**

  ```bash
  mkdir -p /tmp/del31-scratch
  uv run python -c "
  from delhi_psi.config import load_config
  cfg = load_config('ratified-2026', data_dir='/tmp/del31-scratch/data')
  print(cfg.paths.neighbors_artifact)
  print(cfg.paths.out_dir)
  "
  # Expected:
  #   colonies_neighbors_ratified-2026.joblib
  #   /tmp/del31-scratch/data
  ```

  This is the in-scope "the profile LOADS and resolves its per-profile
  artifact name" from the spec; a full preprocess/compute is DEL-32.

---

### Task 3: The production fixtures + the invariant audit + the diff digest

**Files:**
- Modify: `scripts/generate_production_fixtures.py` (`PROFILES`, line 36)
- Create: `tests/fixtures/oraculum/production/ratified-2026.csv`
- Create: `tests/fixtures/messy/production/ratified-2026.csv`

- [ ] **Step 1: Register in the generator.** Append `"ratified-2026"` to
  `PROFILES` in `scripts/generate_production_fixtures.py` (after
  `"services-contested"` — write order is loop order).

- [ ] **Step 2: Regenerate.**

  ```bash
  uv run python scripts/generate_production_fixtures.py
  ```

  Expected: exit 0; the two new CSVs appear. A `planned_relative` guard raise
  here would be a spec violation (Global Constraints; the measured-facts
  section says it cannot happen) — stop and report, never hand-edit a
  fixture.

- [ ] **Step 3: Watch the second watched-red green.**

  ```bash
  uv run pytest tests/test_production_fixtures.py -k "ratified and Regenerable" -q
  # Expected: 2 pass (one byte-for-byte regenerability check per city)
  # (plain `-k ratified` now passes 4, the two spec-shape tests included)
  ```

- [ ] **Step 4: Blast radius.**

  ```bash
  git status --porcelain -- tests/fixtures/
  # Expected: exactly two untracked files, nothing else
  git diff -- tests/fixtures/ | grep '^-' | grep -v '^---' || true
  # Expected: empty — no existing fixture byte changes
  ```

- [ ] **Step 5: The invariant audit (spec § 3.3).**

  ```bash
  grep -rn "ratified-2026" tests/ scripts/ delhi_psi/
  ```

  Expected references: the two registration lists, the value-pin test, the
  YAML's own `profile:` line, this plan, the spec, the CHANGELOG (Task 5) —
  and NO test or script that hands `ratified-2026` un-overridden to
  `compute_frames` / `compute_oracle_frame` / `oracle_profile_path` (that
  path would trip the pipeline's cross-check, pipeline.py:287–290). Record
  the output in the review record.

- [ ] **Step 6: Read the fixture diff (the PR-digest material).**

  ```bash
  uv run python - <<'EOF'
  import pandas as pd
  a = pd.read_csv("tests/fixtures/oraculum/production/code-2025.csv")
  b = pd.read_csv("tests/fixtures/oraculum/production/ratified-2026.csv")
  ka = a.set_index(["scenario", "denom", "settlement", "metric"])
  kb = b.set_index(["scenario", "denom", "settlement", "metric"])
  assert ka.index.is_unique and kb.index.is_unique
  print("rows only in ratified-2026:", kb.index.difference(ka.index).size)
  print("rows only in code-2025:   ", ka.index.difference(kb.index).size)
  common = ka.index.intersection(kb.index)
  d = (kb.loc[common, "value"] - ka.loc[common, "value"]).sort_index()
  changed = d[d.abs() > 1e-12]
  print("changed values:", len(changed), "of", len(common))
  for (sc, dn, st, mt), v in changed.items():
      print(f"  {sc:22s} {dn:11s} {st}  {mt:14s} "
            f"{ka.loc[(sc, dn, st, mt), 'value']:.6g} -> "
            f"{ka.loc[(sc, dn, st, mt), 'value'] + v:.6g}")
  EOF
  ```

  Pinned expectations (measured on the committed `code-2025.csv`; a mismatch
  is a stop-and-report, not a fixture hand-edit): oraculum **1508** rows
  (29 frame keys × 26 metrics incl. `rel_psi`), **58** keys only in
  `ratified-2026` (all `rel_psi`), **0** keys only in `code-2025`, and the
  per-(scenario, denom) settlement counts `7/5/5/6/6` (baseline,
  excl_contributing, excl_ind_removed, excl_removed, excl_rv_only). Messy:
  **1456** rows, **56** `rel_psi` keys, **0** lost, per-scenario settlements
  `10/9/9` per denom.

  Read the output and write the digest: which (scenario, denom, settlement,
  metric) rows move, and in which direction — expected signatures: `touch`
  drops corner-only neighbour pairs (neighbour-term contributions fall),
  `partial_weighted` restores discount-weighted links that
  `global_asymmetric` severed, `eq4_own_only` drops borrowed road access,
  `outside_receiver` removes the double-counted overlap share, `contributes`
  adds lending from the excluded settlements. The `rel_psi` rows are **new
  data**, absent from `code-2025.csv` (`minmax` emits only `norm_psi`); within
  each (scenario, denom) frame `rel_psi == norm_psi / that frame's ratified
  Planned-mean norm_psi`, a constant NOT computable from the old fixture
  (the min-max universe itself moves under the ratified rules). Record the
  digest in the plan's execution notes and paste it into the PR description.
  Do the same (row-count only) for the messy city and note it is
  unremarkable or not.

---

### Task 4: `docs/methodology-config.md`

**Files:**
- Modify: `docs/methodology-config.md`

- [ ] **Step 1: The § 1 profile table — three rows.**
  "Profiles live in `delhi_psi/profiles/`. Two ship:" → "Three ship:"; add
  the row after `manuscript`:

  ```markdown
  | `ratified-2026` | **The ratified rules** — every decided switch from the table below, in one profile: the profile the Phase 4 recalculation (DEL-32) runs. Mixed profile: no single reference rule-set matches all of its values, so it is absent from `tests/test_profiles_match_reference.PROFILE_RULES` and pinned by its production fixtures alone. Its `neighbors_artifact` default is profile-named, so a fresh `preprocess` writes `colonies_neighbors_ratified-2026.joblib` beside `code-2025`'s. |
  ```

- [ ] **Step 2: Finalize the four now-decided cells** in the switch table
  (the "ratified (28 Aug 2026)" column):
  - `barrier.buffer_m`: "`5` m, pending ratification" → "**`5` m** — ratified
    26 Oct 2026; kept a profile value, so a different buffer is a one-line
    edit".
  - `overlap.lending`: the "**Raj to answer** —" sentence →
    "**`outside_receiver`** — ratified 19 Oct 2026 (DEL-20)".
  - `second_normalization`: the "**Raj to choose** —" sentence →
    "**`planned_relative`** — 22 Oct 2026 (4a, DEL-63): the reported scale;
    min-max stays in the appendix".
  - `outputs.denominators`: the "**Raj to choose** —" sentence →
    "**`[pop, popdensity]`** — 19 Oct 2026: popdensity ratified; both stay
    reported".

  In the same pass: the `transform.form` and `aggregation.rule` rows stay
  "Raj to choose" and gain a clause that `ratified-2026` carries the
  `code-2025` values until then; the paragraph at the end of § 1 ("…it stays
  at `whole` until Raj confirms … DEL-31 decides what the ratified profile
  carries") is updated to say the lending half was ratified 19 Oct and
  `ratified-2026` carries `outside_receiver`.

- [ ] **Step 3: Two pointers.** § 2 (worked example 2 — its opening
  identity-mapping + `[RV, Industrial, Other]` exclusion snippet is the
  shipped profile's shape; the surrounding text documents the parked DEL-29
  urban-5 collapse, which this profile is NOT) gains a pointer to the shipped
  `delhi_psi/profiles/ratified-2026.yaml`; § 3 step 2 (registration) gains the
  mixed-profile determination as a one-line note (a profile whose values
  span reference rule-sets is pinned by its fixtures only).

- [ ] **Step 4: README's profile count.** `README.md` (the `--config`
  paragraph, ~line 125) says "Two profiles ship: `code-2025` … and
  `manuscript` …". Update it to three, adding `ratified-2026` — Raj's
  ratified rules, the profile the Phase 4 recalculation runs — so the README
  does not contradict § 1's three-row table.

---

### Task 5: CHANGELOG, full smoke, ship

- [ ] **Step 1: CHANGELOG `[Unreleased]` entry** (top of the section):

  ```markdown
  - **The ratified rules now ship as one profile (DEL-31).** `ratified-2026`
    — a copy of `code-2025` with the decided values: `touch` adjacency,
    `partial_weighted` barrier at the configurable `5 m`, `outside_receiver`
    overlap lending, `eq4_own_only` roads, `planned_relative` second
    normalization (the reported scale, 1 = the average planned colony), and
    the `[RV, Industrial, Other]` exclusion at `post_neighbors` with
    `contributes`.
    - **No shipped number moves.** `code-2025.yaml`, `manuscript.yaml` and
      every existing fixture are byte-identical; the profile is pinned by its
      two new production fixtures, and its profile-named `neighbors_artifact`
      default means a fresh `preprocess` writes
      `colonies_neighbors_ratified-2026.joblib` beside the historic
      artifact.
    - **Still open — `transform.form` and `aggregation.rule`** — the profile
      carries the `code-2025` values by default; a later decision is a
      one-line profile edit.
  ```

- [ ] **Step 2: Full smoke** (background, ~10 min):

  ```bash
  mkdir -p /tmp/del31-scratch
  uv run pytest -q -W error > /tmp/del31-scratch/smoke.log 2>&1
  echo "EXIT_CODE: $?" >> /tmp/del31-scratch/smoke.log
  ```

  Expected: `EXIT_CODE: 0`.

- [ ] **Step 3: Blast-radius proof against the merge base.**

  ```bash
  git diff --name-only 52812c5 -- tests/fixtures/ delhi_psi/ tests/ scripts/
  # Expected: only the new YAML and the two new CSVs, plus the three
  # one-line registrations and the value-pin test
  git diff 52812c5 -- tests/fixtures/ | grep '^-' | grep -v '^---' || true
  # Expected: empty
  ```

- [ ] **Step 4: Push, PR, CI.**

  ```bash
  git push -u origin del-31-ratified-profile
  gh pr create --base main --title "feat: the ratified-2026 methodology profile (DEL-31)" \
      --body-file /tmp/del31-scratch/pr.md
  ```

  The PR body carries: the spec link, this plan, the plan-phase review
  record (with the model-substitution note), both watched-red/green outputs,
  the smoke log tail, the blast-radius proof, and the fixture-diff digest
  (Task 3, Step 6). Wait for CI — the drift guard reproduces every committed
  byte, so green is the expected outcome.

- [ ] **Step 5: Merge and close out.**

  Squash-merge; sync `main`. Jira DEL-31: 21 (In Progress) → 31 (In Review) →
  41 (Done), with one plain comment: the profile name and the decided values
  in one line each, the PR link, the fixture-diff one-liner, and "5 m buffer
  kept configurable — formally yours to ratify alongside the profile". DEL-32
  (the recalculation) is then unblocked.

## Review record (plan phase, round 1 — 27 Oct 2026)

- **Model substitution (recorded per ship.md):** no hosted models were
  available when the round started; per the owner's 26 Oct approval the first
  dispatch used the local `gpu-router/ornith-1.5-35b`, which the owner
  cancelled when the session appeared stuck (no findings produced). The
  owner then switched the review lineup to `opencode-go/glm-5.3-flash`, on
  which both finders and the judge actually ran.
- **Finders:** `glm-5.3-flash` ×2. Finder A verified every code claim against
  source (loader cross-checks, the pipeline re-check, the oracle helpers, the
  registration points, the reference `RULESETS`, the `planned_relative`
  guard, the YAML against every loader rule) — all CONFIRMED. Finder B
  measured the fixtures and ran the ratified methodology in memory through
  `compute_frames` for all 16 city × scenario × denom frames (guard never
  trips; `rel_psi` identity holds to < 1e-12), and audited the CI drift guard
  and every other profile enumerator.
- **Judge:** `glm-5.3-flash`. Dispositions: **KEEP** A1 (Important), A2/B1
  (Minor), A4/B4 (Minor), B3 (Minor), B5 (Minor); **KILL** A3 (the stale
  `oracle_config`/`oracle_profile_path` docstrings and methodology-config § 3
  collapse-procedure line are real but out of scope for this no-new-code
  ticket — noted, not edited); **AMEND** A5 (the messy drop is `{U, N}`, not
  "`U`/`N` + `RV`" — no RV in messy), A6 (§ 2's surrounding text is the
  parked DEL-29 collapse; scope the pointer to its identity+exclusion
  snippet), B2 (`rel_psi` is new data, not a rescale computable from
  `code-2025.csv`).
- **Surviving and fixed in this revision:**
  - **(Important) `-k ratified` matched four tests, not two.** Both
    `test_fixture_is_regenerable` and `test_fixture_has_the_spec_shape`
    parametrize over `PROFILES`; RED is two missing-fixture failures plus two
    `FileNotFoundError`s. The selectors are now
    `-k "ratified and Regenerable"` (2 RED / 2 GREEN), with the four-test
    behaviour noted explicitly so the plan's stop rule cannot misfire.
  - **(Minor) Counts were off by one.** 17 profiles today, 18 after
    registration — Task 1's RED comment and Task 2's set-equality note now
    say 17 → 18 and "(18 == 18)".
  - **(Minor) Mixed-profile enumeration completed.** `outside_receiver` is
    pinned by neither `RULESETS` row (no `overlap_lending` key exists) and
    the `decay` block is carried by defaults; the Global Constraints bullet
    now says so.
  - **(Minor) Scenario-drop wording corrected:** oraculum `{RV, IND}`, messy
    `{U, N}`.
  - **(Minor) Digest expectations pinned:** 1508/1456 rows, 58/56 new
    `rel_psi` keys, 0 lost, per-scenario settlement counts, plus an
    index-uniqueness assert and a sorted diff in the snippet.
  - **(Minor) README's "Two profiles ship" added to Task 4**, and the § 2
    pointer scoped away from the parked collapse.
- **Round 1 verdict: NOT clean — 1 Important, 0 Critical.** Per the stopping
  rule, one fix pass (above) is made and round 2 re-checks the Important
  selector change plus the touched sections; Minors are batched without
  re-review.

## Self-review

- **Spec coverage:** § 1 YAML → Task 2; § 2 registration table → Tasks 1 and
  3; § 3 absent-category resolution → Global Constraints + Task 3 Step 5;
  § 4 fixture regeneration, the planned_relative precondition, the diff as
  artefact → Task 3; § 5 value-pin → Task 1; § 6 docs → Task 4; the
  out-of-scope boundary (loads + artifact name only, no full run) → Task 2
  Step 3; acceptance 1–5 → Tasks 1–5.
- **Placeholders:** the fixture-digest result (Task 3 Step 6), the review
  record, and any execution fixes are filled in at execution time; everything
  else in the plan is literal.
- **Consistency:** the profile name `ratified-2026` is used everywhere (YAML
  `profile:`, rendered artifact name, fixture file names, tests, CHANGELOG);
  the value-pin asserts loaded dataclass fields, never YAML text; the two
  production profiles stay byte-identical by Global Constraint 1, which the
  CI drift guard re-verifies on every build.
