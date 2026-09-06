# Delhi Paper — Sequenced Work Plan

Goal: finish the PSI analysis and ship **"Making the City Unequal: Locating
Public Services in Planned and Informal Settlements in Delhi"** to HAS as a
fresh submission (Patrick also suggests posting to SSRN; AJS noted as a
candidate venue). This is a meta-plan: it sequences the work and records
decisions/context; each phase gets its own detailed plan when it starts.

Sources: Raj ↔ Bob working call of 25 Jul 2026 ("Delhi Paper — To-Do List"),
the "delhi feedback from Brown workshop" doc (with Raj's triage annotations),
and the April 2026 paper draft. Owners: **Bob** (code/index), **Raj**
(categorization/framing). Paper manuscript lives on Overleaf.

Repo state (updated 23 Aug 2026, after Phases 1–2): `main` is the default
branch. The pipeline is the installable delhi_psi package on uv +
pyproject.toml (delhi-psi preprocess → delhi-psi compute, plus
scripts/verify_against_baseline.py), config-driven via
delhi_psi/profiles/*.yaml; the
notebooks are gone (history in git and `archive/master-2021/`). The Oraculum
oracle lives in `tests/` (230 tests green) with its docs under `docs/oracle/`.
The full input dataset (276 MB) lives locally at `~/delhi_data` and is
two-way synced hourly with the shared drive (`Spatial_Index_GIS/delhi_data/`).
Each phase runs brainstorm → approved spec → `/ship` (`.claude/commands/ship.md`)
on its own branch; specs/plans are under `docs/superpowers/`; `CHANGELOG.md`
records what each merged phase changed. Work is tracked in Jira project
**DEL** (bob-bell.atlassian.net) — one epic per phase, `[DEL-nn]` tags below
name each item's ticket; keep the two in sync when either changes.

### Status at a glance (5 Sep 2026)

| Phase | State | Evidence |
|---|---|---|
| 0 Environment & data | done | data synced, `gh` working |
| 1 Runnable pipeline | done | PR #5 — zero deviation from July 2025 baseline |
| 2 Oracle | done | PR #6 — 65 tests; production == reference == hand anchors at 1e-12; mutation-proven; worksheet hand-ratified 24 Aug |
| 3 Refactor & bug audit | **in progress** — cycles 3A–3D merged (PRs #10, #13, #14, #15); 3E's three tickets (DEL-54, DEL-48, DEL-20) all code-complete on their own branches, pending merge and the real-data run step | delhi_psi package, 540 tests; every methodology choice is a profile value; code-2025 reproduces July 2025 at zero deviation |
| 4 Categorization | **decisions received 28 Aug 2026** — measurements, then the ratified profile | `docs/decisions/2026-08-28-raj-methodology-decisions.md` |
| 5–7 | not started | — |

Open items by owner:

- **Bob:** (1) cycle 3E — partial-barrier weighting [DEL-48] + the overlap
  neighbour rule [DEL-20] + the min-max guard (bug-audit 6), the last code
  Phase 4 needs; (2) the pre-recalculation measurements [DEL-49/50/51/52];
  (3) the ratified profile [DEL-31] and recalculation [DEL-32]; (4) the
  batched reply to Raj (decision log, "What goes in the batched reply").
- **Raj:** confirm UV/SDA stay in, the overlap neighbour rule, popdensity,
  `norm_psi` (after Bob's check), Decision B; send the students'
  reclassification list [DEL-53]; the methods footnotes on his own list
  (roads, partial barriers, min-max universe, d in km).
- **Deferred by decision:** Dependabot alerts (absorbed into Phase 3's
  dependency work; the four Dependabot PRs #1–#4 were closed as superseded
  by Phase 1's modernization); `pandas<3` uncap (Phase 3, now that the
  oracle can validate it).

---

## Phase 0 — Environment & data (DONE)

Epic DEL-1.

- [x] Repo restructured: `main` default, 2 canonical notebooks, old code archived
- [x] Data recovered from old machine account; complete input set verified
- [x] Durable local ↔ shared-drive sync (rclone service account + hourly systemd timer)
- [x] GitHub API access (`gh`) for automation
- [x] Claude account logistics resolved (Raj's gifted Max plan, redeemed 12 Aug)

## Phase 1 — Make the pipeline runnable end-to-end (Bob) — P1 (DONE)

*"Get the repo running / modernize dependencies", "remove hardcoded machine
paths". Everything downstream depends on this. Epic DEL-2.*

- [x] Remove hardcoded `data_dir = /home/bwbelljr/delhi_data/` — make the data
      directory configurable (env var `DELHI_DATA_DIR` or config file), so the
      pipeline runs on any machine pointed at a copy of the dataset
      (replicability matters: the repo will be released, and some journals
      require the data too)
- [x] Modernize dependencies: **all packages to latest stable versions**
      (geopandas/shapely/pyproj have breaking API changes since 2021, e.g.
      removed `cascaded_union`; expect more of the same across the stack);
      consolidate to uv + `pyproject.toml` (decided — see Decisions section),
      removing the conda/poetry/Docker files; resolves the deferred
      Dependabot backlog. Safety net for upgrades: the Phase 1 verification
      that outputs still match the July 2025 run acts as the interim
      regression check until the oracle (Phase 2) takes over that job
      permanently.
- [x] Fix small runtime hazards: create output dirs (`os.makedirs`), rename
      stale `12Sep2021` output filenames to dated 2025+ names
- [x] Verify: both notebooks run top-to-bottom on this machine against
      `~/delhi_data`, outputs match the July 2025 run

**Definition of done:** a fresh clone + the shared-drive data reproduces the
existing PSI outputs. **DONE (17 Aug 2026, PR #5): fresh run reproduced the
July 2025 baseline with zero numeric deviation.**

## Phase 2 — The Oracle: ground-truth test harness (Bob) — P1 (DONE)

*Top code priority. Do this BEFORE trusting any recalculation, because the
index is the paper's core contribution. Epic DEL-3.*

- [x] Build a toy "mythical city": 2–3 settlement types, a handful of services
      and boundaries, small enough that the PSI can be computed by hand
- [x] Hand-verify expected values (Bob + Raj do the back-of-envelope check;
      Claude generates the fixture, humans confirm the arithmetic) [DEL-10] —
      **ratified by Bob 24 Aug 2026**: calculator pass over
      `docs/oracle/derivation-worksheet.md` against the manuscript's
      Eq. 1–4 confirmed every anchor; this breaks the circularity between
      production and the Claude-derived reference. Raj's check optional.
- [x] Encode as a pytest suite: oracle fixtures + expected PSI values as a
      permanent regression check on `spatial_index_utils.py`
- [x] Use the oracle as the fix-loop target for any bug found in Phase 3, and
      as the safe sandbox for index-formula experiments in Phase 6 (the
      reference implementation's seven knobs already model the alternatives)
- [ ] Stretch: reproduce the South African source paper's published numbers
      (the index formulation was adapted from Patrick's paper) as a second
      validation case [DEL-11] — **deferred** (explicit non-goal of the Phase 2 spec);
      revisit in Phase 6/7 if time allows. Releasing the harness with the
      package is now a Phase 7 item.
- [x] Decision-support variant: compute the mythical city's PSI under both
      exclusion semantics — (a) dropped settlement types still contribute
      services as neighbors vs. (b) dropped types fully removed before
      neighbor computation — and show the side-by-side delta to Raj to ground
      the open decision below (more informative than asking in the abstract)

**Definition of done:** `pytest` passes with hand-verified expected values;
any future code change that alters the index fails the suite.
**DONE (17 Aug 2026, PR #6): 65 tests green; production == independent
reference == hand anchors at 1e-12; mutation testing confirms the suite
catches a broken index. Hand ratification of
`docs/oracle/derivation-worksheet.md` completed by Bob on 24 Aug 2026.**
Spec: `docs/superpowers/specs/2026-08-17-phase2-oracle-design.md`; plan:
`docs/superpowers/plans/2026-08-17-phase2-oracle.md`; key artifacts:
`tests/fixtures/oraculum/` (inputs + `expected_values.csv`, 2,610 rows,
round-trip tested — regenerate with `scripts/generate_oraculum_fixtures.py`,
never hand-edit), `tests/reference_impl.py` (independent Eq. 1–4, never
imports production code), `docs/oracle/` (worksheet, memo, three maps).

**Findings for Phase 3/4 (see
`docs/oracle/exclusion-semantics-memo.md`):** six documented
manuscript-vs-code divergences, including two that need Raj: exclusion
semantics (a) is unimplementable in current code (silent `except:
pass`), and 429 service points are double-counted via 4,069 overlapping
colony polygons (`docs/data/layer_pathologies.md`). A seventh, latent
item — point membership is boundary-inclusive, but zero real service
points lie on a boundary (closest 1.3 mm) — is pinned by
`test_gap6_border_point_is_double_counted_by_production` and needs no
action unless the layers are re-digitized.

## Phase 3 — Refactor & bug audit (Bob) — P2 — NEXT (unblocked 17 Aug 2026)

*Refactor with the oracle as a safety net. Starts with a brainstorming
session → spec → `/ship`, like Phases 1–2. Items marked "needs Raj" can be
specified and the messy-city tier built before his answers arrive; the
fixes themselves wait for the memo decisions. Epic DEL-4.*

- [x] Brainstorm → owner-approved spec → implementation plan for this phase
      (the spec decides package layout, config schema, and which bug-audit
      items are in scope before Raj answers; may split Phase 3 into more than
      one `/ship` cycle) [DEL-15]
      — done 27 Aug 2026: spec
      `docs/superpowers/specs/2026-08-27-phase3-refactor-design.md`, plan
      `docs/superpowers/plans/2026-08-27-phase3a-refactor.md`; Phase 3
      split into cycles 3A (this one), 3B (DEL-17) and 3C (DEL-24/19/20)
- [x] One canonical implementation: collapse duplicated/near-duplicate logic
      in `spatial_index_utils.py` (e.g. the `*_wards` / `*_buffer` variants of
      `calc_all_services` / `create_service_index`) into single configurable
      functions [DEL-16]
      — done 27 Aug 2026 (3A): `create_service_index` /
      `create_service_length_index` collapsed into `index.service_index`
      fed by `point_counts`/`road_lengths`; the `road_count → road_length`
      special case is gone (line services name their amount column
      `<service>_length`)
- [x] Make settlement types configurable via a mapping layer: run with 10, 8,
      5, or 4 categories from a config (1:1 or X:1 mapping of the 10
      `USO_FINAL` source types), so Raj's categorization decision (Phase 4)
      plugs in without code changes — and so the method ports to other cities
      [DEL-17]
      — done 28 Aug 2026 (3B): profiles carry a required `categories:` block
      (`scheme` + source type → category mapping); `exclusion.types` is
      written in category names; every output carries a `category` column and
      the joblib the scheme stamp. An unmapped source type errors (no
      catch-all, `categories.default` reserved). Both shipped profiles use
      the identity `uso-10` scheme, so no number moved: fixtures
      byte-identical, real-data baseline at 0.000e+00. Spec
      `docs/superpowers/specs/2026-08-27-phase3b-categories-design.md`, plan
      `docs/superpowers/plans/2026-08-27-phase3b-categories.md`
- [x] Modular & extensible structure: distance thresholds, decay weights,
      service sets, adjacency/barrier rules, and category mappings injectable
      as parameters (feeds the Phase 6 sweeps) [DEL-18]
      — done 28 Aug 2026 (3D): every item in that list is now a config value.
      3A made adjacency/barrier rules, service sets, denominators, exclusion
      and units config; 3B the category mappings; 3D adds the last two —
      `adjacency.rule: within_distance` with `adjacency.max_distance_km`
      (polygon-to-polygon band, `>= 0` km) and `decay.form`
      (`inverse_linear` | `none` | `inverse_power` + `exponent` |
      `exponential` + `scale_km`) with `decay.distance`
      (`centroid` | `boundary`). Each new value has a reference-implementation
      rule and oracle pins on both fixture cities (eight derived variants in
      `tests/fixtures/*/variants_expected_values.csv`), so a Phase 6 sweep
      point is one YAML file. No behaviour changed: both shipped profiles
      gained only `decay.distance: centroid`, naming what they always did;
      every committed fixture is byte-identical and the real-data baseline is
      0.000e+00. Spec
      `docs/superpowers/specs/2026-08-28-injectable-parameters-design.md`,
      plan `docs/superpowers/plans/2026-08-28-injectable-parameters.md`,
      procedure `docs/methodology-config.md` § 6
- [ ] Bug audit — the Phase 2 oracle turned this from a vague mandate into a
      prioritized, evidence-backed list (all six divergences are documented in
      `docs/oracle/exclusion-semantics-memo.md` and pinned by tests):
      1. **bbox adjacency — now known to be the DOMINANT regime, not an edge
         case.** Measured on the real layer (reproducible source:
         `docs/data/layer_pathologies.md`): ZERO of 4,357 colonies are
         rectangles, and a colony's bounding box is typically ~2× its polygon
         area (median ratio 1.95, p90 3.6, max 28,766). So bbox-adjacency
         invents neighbors citywide, constantly. Plausibly the largest
         paper-vs-code gap. [DEL-19]
         Pinned today's behaviour on a purpose-built city: `H`/`L` are
         disjoint yet bbox neighbours both ways, `T`/`L` touch at a single
         point, and `M` is in `G`'s list while `G` is not in `M`'s
         (`tests/test_messy_fixtures.py`, `docs/oracle/messy-city.md`).
         **Decided 28 Aug 2026 (Raj): shared border — a fix, not a
         ratification.** `adjacency.rule: touch` goes into the ratified
         profile (DEL-31) and the pins flip then; bbox becomes the Phase 6
         comparison variant (DEL-39). Open sub-question: corner-only
         contact pairs on the real layer [DEL-50]. Decision log § 3.
      2. **429 service points double-counted** (bank 211, health 18, police 2,
         ration 104, school 53, transport 41) across 4,069 overlapping colony
         polygon pairs. A containment rule does NOT fix this — it needs a
         decision on how overlapping colonies share a point. [DEL-20]
         Pinned today's behaviour: one clinic strictly inside `O1 ∩ O2`
         is counted for both (`tests/test_messy_fixtures.py::
         test_the_overlap_clinic_is_counted_for_both_owners`). Agreed
         behaviour on both sides — production's `intersects` and the
         reference's `within` both do it — so it is asserted directly, never
         by comparison. The measured real-layer counts now have a
         reproducible source: `docs/data/layer_pathologies.md`.
         **Decided 28 Aug 2026 (Raj): a service in the overlap counts for
         every colony containing it** — today's behaviour, ratified; the
         pin stays as intended behaviour. **Added 5 Sep 2026 (Bob, to
         confirm with Raj): a neighbour lends only the services not already
         inside the receiving settlement**, so an overlap service is not
         counted a second time through the neighbour term (under `touch`
         overlapping colonies are neighbours too). That part is code —
         per-pair adjusted counts — and ships in cycle 3E with DEL-48; the
         messy city already holds the fixture case. Decision log § 5.
      3. ~~**Silent `except: pass` in `calc_pcen_mobile`** — swallows
         missing neighbors, making exclusion semantics (a) unimplementable
         (WORKPLAN Open Decision A is half-answered by this).~~ [DEL-21] —
         done 27 Aug 2026 (3A): replaced by an explicit lookup;
         `exclusion.absent_neighbor: contributes` reads amounts from the
         pre-exclusion frame, so semantics (a) is now implementable.
      4. ~~Barrier rule is global + asymmetric vs. the manuscript's pair
         severing; roads carry neighbor decay Eq. 4 does not have;
         `norm_psi` is a second normalization absent from Eq. 1;
         popdensity has no manuscript equation.~~ [DEL-22] — done 27 Aug
         2026 (3A): all four are config switches with both values
         implemented and reference-pinned (`barrier.rule`, `roads`,
         `second_normalization`, `outputs.denominators`), and the
         `manuscript` profile runs the paper's rule-set end to end.
         **Calls made 28 Aug 2026:** roads → `eq4_own_only` (each colony
         counts only its own roads; NB the call's premise was inverted —
         the code decays roads, the paper does not — so this CHANGES the
         published numbers; effect measured first [DEL-49]); barrier →
         partial weighting by the share of shared boundary covered, which
         is the reserved `partial_weighted` value and therefore **code:
         cycle 3E [DEL-48]** (Oraculum's canal redrawn to cover the full
         A–D edge so the worksheet holds; partial case pinned on the messy
         city; provenance of the barrier layers to check [DEL-51]);
         `norm_psi` and popdensity still open — Bob determines, Raj
         confirms [DEL-52]. Decision log §§ 2, 4, 7, 8.
      5. ~~Dead code: function(s) defined but never called; also the pandas
         `FutureWarning`s (dtype-incompatible setitem in
         `spatial_index_utils.py` ~L835/L1212) so a `-W error` CI run becomes
         feasible.~~ [DEL-23] — done 27 Aug 2026: 17 dead functions (684
         lines, incl. all `*_wards`/`*_buffer` variants) removed; warnings
         fixed under DEL-26; CI runs `pytest -W error`.
      6. ~~**`index.minmax` has no `hi == lo` guard**, so a constant PCEN
         column divides 0/0 — latent on real layers, reachable via the
         population-drop path.~~ [DEL-54] — **done 5 Sep 2026 (cycle 3E,
         ticket 1 of 3):** `index.minmax` raises a `ValueError` naming the
         column, the row count and the value, before the division, so both
         callers (each service's `service_index` and `overall_psi`'s second
         normalisation) surface the same diagnosable error. The independent
         reference now raises too, at both of its min-max sites, instead of
         inventing `0.0` — the equations do not define a value at hi == lo,
         and a reference that invents one is a rule-set divergence waiting
         to be relied on. Deliberately NOT covered: an all-NaN column,
         since `NaN == NaN` is False and an all-NaN PCEN column is an
         upstream NaN belonging to the population join and `validate`; the
         limit is stated in the `minmax` docstring. No config value, no
         profile change, no fixture regenerated (all three generators
         re-run byte-identical), real-data `code-2025` verify unchanged.
         The history below is kept because it is why the item was never
         urgent:
         CORRECTED 28 Aug 2026 (3C): under `-W error` — which is how CI and
         every local run invoke pytest — the 0/0 does **not** produce a
         silent NaN; numpy emits `RuntimeWarning: invalid value encountered
         in scalar divide` and the warning filter turns it into a raised
         error. The silent-NaN path (uncaught by `check_no_negative`, skipped
         by `overall_psi`'s mean) exists only OUTSIDE a `-W error` run. Both
         fixture cities are therefore built so that no PCEN column is
         constant: `scripts/check_oraculum_invariants.py` refuses to write a
         fixture with a degenerate min-max group, and the generators call it
         before writing — which is also why the guard is unreachable through
         the committed fixtures and why nothing regenerated when it landed.
         Spec `docs/superpowers/specs/2026-09-05-cycle-3e-partial-barriers-design.md`
         §§ 4, 6.3 and 12 item 9; decision log § 12.
- [x] Add a second "messy city" fixture tier (verified against
      `tests/reference_impl.py`, NOT hand arithmetic — Oraculum stays the
      hand-ratifiable ground truth for the math, deliberately small). Must
      cover the real-data pathologies Oraculum omits by construction:
      irregular non-rectangular polygons (so bbox ≠ geometry, the real
      regime), a MultiPolygon (556 of 4,357 real settlements are multi-part),
      an overlapping polygon pair, an isolated settlement (6 under bbox / 20
      under touch (see `docs/data/layer_pathologies.md`); the earlier ~360
      was an unreproducible ad-hoc figure), a settlement with no population
      row (15 real ones), and an area-extreme sliver (real areas span
      2.3e-9 → 29 km²). This tier is what would PROVE any fix to items 1 and
      2 above. [DEL-24]
      — done 28 Aug 2026 (3C): `tests/fixtures/messy/` carries eleven
      settlements covering all six pathologies, scored by the reference
      implementation and byte-stable through `scripts/generate_messy_fixtures.py`
      + `scripts/generate_production_fixtures.py`. A `City` abstraction
      (`tests/cities.py`) with two instances runs every proof on both cities;
      the reference was **generalised only** (a city, an explicit scenario
      table, all road rows summed) — not one rule changed, no production code
      was touched, Oraculum's three CSVs are byte-identical and the real-data
      baseline is at `0.000e+00`. Spec
      `docs/superpowers/specs/2026-08-28-messy-city-tier-design.md`, plan
      `docs/superpowers/plans/2026-08-28-messy-city-tier.md`, docs
      `docs/oracle/messy-city.md` and `docs/data/layer_pathologies.md`
- [x] Retire the notebooks entirely (decided): notebooks already removed
      in Phase 1 (logic lives in `scripts/`); remaining work is the package
      pipeline stages with logged validation (the notebooks' eyeball checks
      become assertions) and a figures command that renders to files [DEL-25]
      — done 27 Aug 2026 (3A): `delhi-psi {preprocess,compute}` are the
      pipeline stages, `delhi_psi.validate` turns the eyeball checks into
      raising assertions, `logging` replaced `print`. The figures command
      is Phase 4 (DEL-33).
- [x] Lift the `pandas<3` cap in `pyproject.toml` now that the oracle can
      validate the major-version jump; sweep any remaining Dependabot alerts
      at the same time (deferred here by decision — do not fix piecemeal)
      [DEL-26] — done 27 Aug 2026: pandas 3.0.5, 5 dtype fixes, 280 legacy
      alerts dismissed
- [x] Add GitHub Actions CI running `uv run pytest` on every push/PR
      (decided in meta-planning "once the suite exists") [DEL-27] — done
      25 Aug 2026, PR #7: `.github/workflows/ci.yml` (locked sync, the
      oracle suite under `-W error`, fixture-drift guard); spec
      `docs/superpowers/specs/2026-08-24-ci-workflow-design.md`. Owner
      follow-up: make `test` a required check in branch protection.
- [ ] **Cycle 3E — the last code Phase 4 needs.** One spec
      (`docs/superpowers/specs/2026-09-05-cycle-3e-partial-barriers-design.md`),
      **three tickets on three branches off `main`**, merged in order:
      - [x] (c) the `hi == lo` guard in `index.minmax` (raise) — bug-audit 6
            [DEL-54] — done 5 Sep 2026, PR #18 → `bcc557c`. Both sides of the
            oracle raise; no fixture moved; real-data verify `0.000e+00`.
      - [x] (a) `barrier.rule: partial_weighted` — weight a neighbour's
            contribution by the unblocked share of the shared boundary,
            linear and symmetric [DEL-48] — done 6 Sep 2026, branch
            `del-48-partial-barriers`. Shipped as the `partial_5m` VARIANT,
            not as a change to the `ideal`/`code` rule-sets, so **the canal
            was NOT redrawn and no messy geometry was added** — the owner's
            fixture authority was granted but proved unnecessary, and the
            canal at x ∈ [25, 475] is itself the fractional anchor:
            **w_AD = 0.08** at the 5 m buffer (460 of the 500 m A–D edge
            blocked). `barrier.buffer_m` is required by that rule, rejected
            outside it, and strictly > 0. Both cities' `expected_values.csv`
            and every `production/*.csv` byte-identical; the two variants
            CSVs changed by addition only.
      - [x] (b) the overlap neighbour rule — a neighbour lends only services
            not already inside the receiving settlement [DEL-20] — done
            6 Sep 2026, branch `del-20-overlap-lending`. Shipped as the
            `overlap_outside` VARIANT with `whole` in both shipped profiles,
            so **no existing expected value moved**: both cities'
            `expected_values.csv` and every `production/*.csv` byte-identical,
            the two variants CSVs changed by addition only. The pin: on the
            messy city's one overlapping pair, O1's clinic PCEN falls from
            (1 + 1/1.8)/600 today to 1/600 under `outside_receiver`.
      Decision log §§ 4, 5, 12.
- [x] Pre-recalculation measurements (no code; feed the batched reply to
      Raj): corner-only contact pairs on the real layer [DEL-50]; barrier
      layer provenance [DEL-51]. (The roads and `norm_psi` measurements are
      Phase 4 items, DEL-49/52.)
      — done 5 Sep 2026. **DEL-50** (`docs/data/layer_pathologies.md`):
      656 pairs involving 955 settlements touch only at a corner (most
      plausibly four-way junctions on a tessellated layer — an
      interpretation, not measured); not neighbours under `touch`, all
      neighbours under a 0 km band — a corner is not a border, `touch`
      stands. **DEL-51** (`docs/data/barriers.md`): 43 canal / 5,356
      railway / 616 drain features flagging 28 / 240 / 390 settlements
      (595 under any); schemas of an official GIS source, ArcGIS Pro files
      of 2 Aug 2020, the root copies are re-exports of the same data; the
      agency and what "manually marked" covered remain a question for
      Bijoy.

**Definition of done:** oracle suite still passes; one code path per concept;
settlement categories, services, and distance parameters are config, not code.

## Phase 4 — Settlement categorization (Raj decides, Bob implements) — P1

*The big analytical piece. Workshop consensus was that ~10 Delhi-specific
types are too much detail. **Raj decided on 28 Aug 2026 not to collapse them
now**: keep every type, drop three, do the framing in the writing, and
revisit only if reviewers push back (Patrick prefers the types separate).
Epic DEL-5. All decisions: `docs/decisions/2026-08-28-raj-methodology-decisions.md`.*
- [x] **Raj:** drop all non-urban categories from the entire analysis —
      figures and calculations; move their mention to footnotes. It's an
      urban project. [DEL-28] — **decided 28 Aug 2026: drop RV, Industrial
      and Other** (three types, not two). UV (138) and SDA (86) stay in as
      reported types — implied, to be confirmed in the batched reply.
      Excluded settlements still lend services to neighbours (Open
      Decision A → semantics (a)); they get no PSI and no output row.
- [ ] **Raj:** decide the collapsed categories — working candidate from the
      workshop triage: **planned / unauthorized / regularized-unauthorized /
      resettlement colonies / JJCs** (5 categories). [DEL-29] — **parked
      28 Aug 2026** ("not now … wait for people to complain"). Not
      rejected: the recipe is `docs/methodology-config.md` § 2 worked
      example 2, one YAML block, when a reviewer asks.
- [ ] **Raj:** figure decisions from the triage — full map for spatial extent;
      breakdown charts show the **seven reported types** (was: 5
      categories); feature the **JJC vs. planned juxtaposition**; remove
      the per-type data table (footnotes instead) [DEL-30]. Maps that show
      all of Delhi left-join the PSI file onto the settlement layer and
      draw the unscored types grey.
- [x] **Bob:** pre-recalculation measurements, reported to Raj before
      DEL-32 runs: JJC road access (road inside vs only in a touching
      neighbour vs neither) and the one-factor effect of `roads:
      eq4_own_only` on `code-2025`, by type [DEL-49]; which PSI column the
      April 2026 figures report (`unnorm_psi` vs `norm_psi`), and the
      popdensity keep/drop default [DEL-52]
      — done 5 Sep 2026. **DEL-49** (`docs/data/roads_access.md`): 17 of
      764 JJCs contain a major road, 645 reach one only via a touching
      neighbour, 102 neither; under own-only roads 422 of 749 reported
      JJCs fall to a zero road index, the JJC mean PSI falls 13 % (pop) /
      10 % (density), no JJC/Planned ordering flip — the decision stands.
      **DEL-52** (`docs/data/psi_columns.md`): Figure 4 matches `norm_psi`
      × popdensity on 8 of 8 bars (max gap 0.0006); both of Bob's proposed
      defaults (`second_normalization: false`, drop popdensity) are
      withdrawn and the two choices go to Raj.
- [ ] **Bob:** write the **ratified profile** [DEL-31] — one YAML, a copy of
      `code-2025.yaml` with the 28 Aug decisions: `adjacency.rule: touch`,
      `barrier.rule: partial_weighted` with `barrier.buffer_m` (loadable
      since DEL-48; the buffer value itself is Raj's to ratify),
      `roads: eq4_own_only`,
      `exclusion: {types: [RV, Industrial, Other], stage: post_neighbors,
      absent_neighbor: contributes}`; `second_normalization` — the figures
      report `norm_psi` (DEL-52), so `true` unless Raj switches to Eq. 1 as
      written; `outputs.denominators` — the figures use popdensity (DEL-52),
      so it stays reported unless Raj switches to per-population Eq. 3;
      identity mapping. Procedure: `docs/methodology-config.md` § 3.
      DEL-48 and DEL-20 are no longer blockers (both merged); still blocked
      by Raj's two DEL-52 answers and his answer on `overlap.lending` — the
      counting half is ratified, but whether `outside_receiver` becomes the
      ratified profile's value awaits him.
- [ ] **Bob:** recalculate all indexes with the ratified profile
      (supersedes the current "no RV" run) — gated on cycle 3E, the
      measurements above, and the ratified profile [DEL-32]. Hand
      ratification (done), Decision A (done), the mapping (identity) and
      the fix-or-ratify calls (made) no longer gate it.
- [ ] Regenerate paper figures from the new run [DEL-33]

**Definition of done:** new PSI outputs under the ratified profile, synced to
the shared drive with clearly dated filenames; figures updated.

## Phase 5 — Shippable minimum

Epic DEL-6. Checkpoint, not a work phase: Phases 1 + 2 + 4 together produce the minimum
credible revision (correct code, verified index, non-urban dropped, new
categories). If the deadline bites, ship after Phase 5 and treat Phase 6 as
appendix material added in revision.

## Phase 6 — Robustness sweeps & measurement variants (Bob) — P2

*Mostly appendix "gravy": run the alternatives, report in footnotes/appendix,
keep the main tables unchanged if variants align. Not fishing — demonstrating
rigor. "Since Claude makes code easy to create, just do all the checks."
Epic DEL-7.*

Index formulation:
- [ ] Alternative formulations for the compressed 0–1 effect sizes ("make the
      values less small"): transformations (e.g. log), tested against the
      oracle first (the 2021 `Transforms for Skewed Data` exploration in
      `archive/master-2021/` is prior art — none were adopted then) [DEL-34]
- [ ] **Rank-based index** (new idea from the workshop): instead of averaging,
      explore ranking mechanisms — average rank per settlement category, and
      composition of the top/bottom deciles by category (as in
      intergenerational-mobility research) [DEL-35]

Distance / reachability:
- [ ] Distance-threshold sweep: 1 km / 5 km / 10 km; show index stability
      [DEL-36]
      — profile only since 3D: `adjacency: {rule: within_distance,
      max_distance_km: X}`, one YAML per point; see
      `docs/methodology-config.md` § 6 (a complete `band-1km.yaml`, and the
      measured cost of a 10 km `preprocess`)
- [ ] Parameterize and vary the decay weight 1/(1+D) (currently arbitrary);
      revisit centroid-to-centroid vs. other distance definitions [DEL-37]
      — profile only since 3D: `decay.form` (`none` | `inverse_power` +
      `exponent` | `exponential` + `scale_km`) and `decay.distance`
      (`centroid` | `boundary`); changing only `decay.*` does not invalidate
      the neighbours artifact, so a decay sweep needs no re-`preprocess`.
      **Raj's steer (28 Aug 2026):** keep 1/(1+d) in km for the main text;
      the sweep should favour forms that spread the neighbour weights
      "up from zero" (`inverse_power` with exponent < 1, `exponential`
      with a scale of a few km, `boundary` distance) and report the effect
      on the category ordering. A steeper decay for roads than clinics is
      punted (and moot — roads have no neighbour term). Decision log § 9
- [ ] Per-service distance expectations (a school may reasonably be farther
      than water); connects to the walkability/food-desert framing [DEL-38]
- [ ] Adjacency-method comparison (bbox vs. touch) as a reported variant —
      **`touch` is the main text (Raj, 28 Aug 2026); bbox is this
      variant** [DEL-39]
      — profile only since 3D: `adjacency.rule` is `bbox` | `touch` |
      `within_distance`, and a 0 km band is the third comparison point (the
      intersection rule, which is `touch` plus corner-only contacts)

Service-set / measurement variants (from the workshop triage):
- [ ] With/without **ration shops** sensitivity (demand-driven, targeted to
      the poor — check whether the result strengthens without them; either way
      it qualifies the argument) [DEL-40]
- [ ] Decide on **ATMs/banking** (workshop said drop as private/market-driven;
      Raj's note: "they are material assets" — a with/without variant settles
      it empirically) [DEL-41]
- [ ] Core-universal-services variant: schools, health, water only [DEL-42]
- [ ] Facility size / capacity (intensive margin, not just counts) — P2,
      data-permitting [DEL-43]
- [ ] **Media-based reclassification** as an appendix robustness check
      (Raj, 28 Aug 2026): two students reclassified settlements from media
      articles (~80% of changes are RV → UV); rerun with the reclassified
      types and report next to the state-category results. **Waiting on
      Raj's list** (IDs, old type, new type). One profile pointing at the
      reclassified type column; no code if it joins on `USO_AREA_U`.
      [DEL-53]
- Rejected in triage (do not pursue; no ticket): roads as area instead of
  length
- Punted 28 Aug 2026 (revisit at revision, no ticket): a steeper decay for
  roads than for point services

- [ ] Write up all variants in an appendix; keep main claims unchanged if
      variants align (or honestly flag if not) [DEL-44]

**Definition of done:** appendix section with variant tables; main results
demonstrated stable (or divergences surfaced and discussed).

## Phase 7 — Release & ship

Epic DEL-8.

- [ ] Final repo cleanup for public release (README quickstart, data-access
      instructions, license check) — people will run the repo (and point
      Claude at it) first thing, so find issues before they do [DEL-45]
- [ ] Release the oracle/test harness and fixtures with the package, and
      write the **reproducibility appendix** (the fixture city, the
      hand-derived worksheet, the test that ties them) [DEL-46] — **planned,
      no longer optional: agreed with Raj 28 Aug 2026** ("Beautiful").
      Whether the messy city is described alongside is a writing call.
      DEL-45's public-release pass should present the oracle as a
      front-door feature, not a test detail.
- [ ] Ship to HAS (and post to SSRN per Patrick's suggestion) [DEL-47]

## Decisions made (16 Aug 2026 meta-planning session)

- **Oracle contract — the manuscript is truth.** The mythical city's expected
  values are hand-computed from the paper's equations (Eq. 1–4). Code must
  reproduce them before it is trusted on Delhi. On mismatch, default is to
  fix the code; a deliberate deviation (e.g. bbox-adjacency) may instead be
  ratified with Raj, in which case the methods text and the hand-derived
  values are updated to match. End invariant: manuscript, hand calculation,
  and code all agree — no silent deviations.
- **Mythical city ships as test fixtures**: tiny GeoJSON inputs + a
  hand-derived expected-values table checked into the repo; pytest asserts
  the pipeline reproduces them. Fixtures should cover the tricky edges
  (barrier between adjacent settlements, zero-service settlement,
  second-order neighbor that must not count).
- **End-state architecture — Option B**: installable package (`delhi_psi/`
  with io / neighbors / index / config modules), config-file-driven runs via
  a CLI (`delhi-psi run config.yaml`), robustness sweeps = loops over
  configs.
- **No notebooks.** Both current notebooks are linear drivers with no real
  visual content; they become pipeline stages with logged validation (their
  eyeball checks become assertions — e.g. the negative-PSI check becomes a
  hard guard). Figures are generated to files by a figures command. The two
  plotting helpers in utils survive as ordinary library functions.
- **Tooling — uv + `pyproject.toml`** as the single source of dependency
  truth. Remove `requirements.txt`, `environment.yml`, `poetry.lock`,
  `Dockerfile`, `install_conda_environment.sh` (Docker served a colleague's
  one-off need that no longer exists). Add GitHub Actions CI running the
  test suite on every push once it exists.
- **Scope/timeline**: full email scope ("thorough"); no hard calendar
  deadline — sequencing and correctness over speed.

## Open decisions (need Raj / group)

Epic DEL-9. Bob's handoff step — sending Raj the oracle memo — was DEL-12
(24 Aug 2026). **Raj answered on the 28 Aug 2026 call**; the full record,
with transcript timestamps and Bob's rulings on the sub-questions, is
`docs/decisions/2026-08-28-raj-methodology-decisions.md`. Summary here.

**A. Exclusion semantics — DECIDED 28 Aug 2026.** [DEL-13 ✓]

1. **Neighbor treatment of dropped types → (a), they still contribute.**
   "Just because we made an analytical decision about a categorization, it
   doesn't make sense to remove the physical elements." Config:
   `exclusion.absent_neighbor: contributes`. Dropped settlements get no
   PSI and no output row (Bob's ruling; maps left-join onto the settlement
   layer).
2. **Min–max universe → reported settlements only** (today's behaviour):
   "the minimum has to be the ones which are in contention." Methods
   sentence on Raj's list.
3. **Descriptive tables** — Raj's writing; noted so the restatement is
   deliberate.

Plus: the dropped types are **RV, Industrial, Other** (DEL-28); UV and SDA
stay in (to confirm).

**C. Oracle-memo methodology calls — DECIDED, two residuals.**
Adjacency → shared border (`touch`; DEL-19, corner check DEL-50).
Overlapping colonies → a service in the overlap counts for each owner
(ratified), and — Bob's addition, to confirm — is not lent again through the
neighbour term (DEL-20, code). Barrier → partial weighting by the share of
shared boundary covered (DEL-48, code; provenance DEL-51). Roads → own
settlement only, Eq. 4 as written (`eq4_own_only`; NB a change from the
published numbers — the call's premise was inverted; effect measured first,
DEL-49). **Residuals:** `norm_psi` (Raj does not know; Bob determines) and
popdensity (not discussed; Bob proposes drop from reported results) —
[DEL-52]. `docs/oracle/suggested-fixes-memo.md` keeps Bob's original
proposals with Raj's answer under each.

**B. Data-release posture** (Raj/group decision — not Bob's call alone).
Options, in ascending openness: code-only (repo + fixtures, runnable but
Delhi numbers not reproducible by outsiders); code + derived outputs
(publish the per-settlement PSI as CSV/GeoPackage — the paper's headline
dataset — without raw inputs); full archive (inputs + outputs on
Zenodo/OSF with DOI — requires a redistribution-rights check on the
DUSIB/DDA/MCD-derived data; WorldPop is CC-BY). **Not discussed on 28 Aug;
the reproducibility-appendix decision (DEL-46) fixes the floor at code +
fixtures.** Bob's proposed default for the batched reply: code + derived
outputs. [DEL-14]

## Out of repo scope (paper-side, tracked for completeness)

Workshop items that live in the manuscript/analysis rather than this codebase:
theory of mechanisms (legal authority, property rights, political
representation); endogeneity/sorting framing; neglect vs. failure-to-keep-up
distinction; language fixes (illegality vs. unauthorized vs. informal;
resettlement as formalization; "increasing" inequality claims); SES /
consumption controls (check with Aashish); within-unauthorized variation;
regularization-effects analyses (scope decisions pending); Sam's EB/Economic
Census data collaborations; process-tracing / media-accounts supplements.

---

## Critical path (from the call, updated for repo state)

1. ~~Phase 0~~ → ~~Phase 1~~ (runnable pipeline) → ~~Phase 2~~ (oracle) — done
2. ~~Raj answers the oracle memo and settles categories~~ — done 28 Aug 2026
   (no collapse; decision log)
3. **Phase 3, cycle 3E** — partial barriers + overlap neighbour rule +
   min-max guard (DEL-48/20): the last code before the numbers; in
   parallel the measurements (DEL-49/50/51/52) and the batched reply to Raj
4. **Phase 4** ratified profile (DEL-31) → recalculation (DEL-32) → figures
   (DEL-33) → shippable minimum (Phase 5); **Phase 6** sweeps fill the
   appendix (DEL-53 once Raj's list arrives)
5. **Phase 7** ship, with the reproducibility appendix (DEL-46)

Standing discipline (from the call): interrogate the design before letting
Claude run; verify and inspect its output — the oracle exists precisely to
make that verification mechanical.
