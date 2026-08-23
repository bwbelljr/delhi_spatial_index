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
branch. The pipeline is three plain scripts on uv + `pyproject.toml`
(`scripts/preprocess.py` → `scripts/compute_psi.py` →
`scripts/verify_against_baseline.py`) around `spatial_index_utils.py`; the
notebooks are gone (history in git and `archive/master-2021/`). The Oraculum
oracle lives in `tests/` (65 tests green) with its docs under `docs/oracle/`.
The full input dataset (276 MB) lives locally at `~/delhi_data` and is
two-way synced hourly with the shared drive (`Spatial_Index_GIS/delhi_data/`).
Each phase runs brainstorm → approved spec → `/ship` (`.claude/commands/ship.md`)
on its own branch; specs/plans are under `docs/superpowers/`; `CHANGELOG.md`
records what each merged phase changed.

### Status at a glance (23 Aug 2026)

| Phase | State | Evidence |
|---|---|---|
| 0 Environment & data | done | data synced, `gh` working |
| 1 Runnable pipeline | done | PR #5 — zero deviation from July 2025 baseline |
| 2 Oracle | done (code) | PR #6 — 65 tests; production == reference == hand anchors at 1e-12; mutation-proven |
| 3 Refactor & bug audit | **next** — unblocked | evidence-backed bug list below |
| 4 Categorization | waiting on Raj | — |
| 5–7 | not started | — |

Open items by owner:

- **Bob:** (1) hand-ratify `docs/oracle/derivation-worksheet.md` (~15-min
  calculator pass; flip `STATUS: RATIFICATION PENDING` → `RATIFIED by Bob on
  <date>`) — must precede Phase 4's recalculation, does not block Phase 3;
  (2) send Raj `docs/oracle/exclusion-semantics-memo.md` — his answers to
  its six divergences decide what Phase 3 fixes vs. ratifies.
- **Raj:** memo decisions (above); Open Decisions A and B below; Phase 4
  categorization.
- **Deferred by decision:** Dependabot alerts (absorbed into Phase 3's
  dependency work; the four Dependabot PRs #1–#4 were closed as superseded
  by Phase 1's modernization); `pandas<3` uncap (Phase 3, now that the
  oracle can validate it).

---

## Phase 0 — Environment & data (DONE)

- [x] Repo restructured: `main` default, 2 canonical notebooks, old code archived
- [x] Data recovered from old machine account; complete input set verified
- [x] Durable local ↔ shared-drive sync (rclone service account + hourly systemd timer)
- [x] GitHub API access (`gh`) for automation
- [x] Claude account logistics resolved (Raj's gifted Max plan, redeemed 12 Aug)

## Phase 1 — Make the pipeline runnable end-to-end (Bob) — P1 (DONE)

*"Get the repo running / modernize dependencies", "remove hardcoded machine
paths". Everything downstream depends on this.*

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
index is the paper's core contribution.*

- [x] Build a toy "mythical city": 2–3 settlement types, a handful of services
      and boundaries, small enough that the PSI can be computed by hand
- [ ] Hand-verify expected values (Bob + Raj do the back-of-envelope check;
      Claude generates the fixture, humans confirm the arithmetic) —
      **machine-verified only so far**: production agrees with an
      independent reference implementation and with Claude-derived hand
      anchors; Bob's calculator pass over
      `docs/oracle/derivation-worksheet.md` is the outstanding step that
      breaks the circularity
- [x] Encode as a pytest suite: oracle fixtures + expected PSI values as a
      permanent regression check on `spatial_index_utils.py`
- [x] Use the oracle as the fix-loop target for any bug found in Phase 3, and
      as the safe sandbox for index-formula experiments in Phase 6 (the
      reference implementation's seven knobs already model the alternatives)
- [ ] Stretch: reproduce the South African source paper's published numbers
      (the index formulation was adapted from Patrick's paper) as a second
      validation case — **deferred** (explicit non-goal of the Phase 2 spec);
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
`docs/oracle/derivation-worksheet.md` remains PENDING by design (Bob's
~15-min calculator pass) — do it before Phase 4's recalculation.**
Spec: `docs/superpowers/specs/2026-08-17-phase2-oracle-design.md`; plan:
`docs/superpowers/plans/2026-08-17-phase2-oracle.md`; key artifacts:
`tests/fixtures/oraculum/` (inputs + `expected_values.csv`, 2,610 rows,
round-trip tested — regenerate with `scripts/generate_oraculum_fixtures.py`,
never hand-edit), `tests/reference_impl.py` (independent Eq. 1–4, never
imports production code), `docs/oracle/` (worksheet, memo, three maps).

**Findings for Phase 3/4 (see `docs/oracle/exclusion-semantics-memo.md`):**
six documented manuscript-vs-code divergences, including two that need
Raj: exclusion semantics (a) is unimplementable in current code (silent
`except: pass`), and ~450 service points are double-counted via 4,050
overlapping colony polygons.

## Phase 3 — Refactor & bug audit (Bob) — P2 — NEXT (unblocked 17 Aug 2026)

*Refactor with the oracle as a safety net. Starts with a brainstorming
session → spec → `/ship`, like Phases 1–2. Items marked "needs Raj" can be
specified and the messy-city tier built before his answers arrive; the
fixes themselves wait for the memo decisions.*

- [ ] One canonical implementation: collapse duplicated/near-duplicate logic
      in `spatial_index_utils.py` (e.g. the `*_wards` / `*_buffer` variants of
      `calc_all_services` / `create_service_index`) into single configurable
      functions
- [ ] Make settlement types configurable via a mapping layer: run with 10, 8,
      5, or 4 categories from a config (1:1 or X:1 mapping), so Raj's
      categorization decision (Phase 4) plugs in without code changes — and so
      the method ports to other cities
- [ ] Modular & extensible structure: distance thresholds, decay weights,
      service sets, and category mappings injectable as parameters (feeds the
      Phase 6 sweeps)
- [ ] Bug audit — the Phase 2 oracle turned this from a vague mandate into a
      prioritized, evidence-backed list (all six divergences are documented in
      `docs/oracle/exclusion-semantics-memo.md` and pinned by tests):
      1. **bbox adjacency — now known to be the DOMINANT regime, not an edge
         case.** Measured on the real layer: ZERO of 4,357 colonies are
         rectangles, and a colony's bounding box is typically ~2× its polygon
         area (median ratio 1.95, p90 3.6, max 28,766). So bbox-adjacency
         invents neighbors citywide, constantly. Plausibly the largest
         paper-vs-code gap; needs Raj (methodology).
      2. **~450 service points double-counted** across 4,050 overlapping
         colony polygon pairs (bank 232, ration 104, school 53, transport 41,
         health 18, police 2). A containment rule does NOT fix this — it needs
         a decision on how overlapping colonies share a point. Needs Raj.
      3. **Silent `except: pass` in `calc_pcen_mobile`** — swallows missing
         neighbors, making exclusion semantics (a) unimplementable
         (WORKPLAN Open Decision A is half-answered by this).
      4. Barrier rule is global + asymmetric vs. the manuscript's pair
         severing; roads carry neighbor decay Eq. 4 does not have; `norm_psi`
         is a second normalization absent from Eq. 1; popdensity has no
         manuscript equation. Each: fix to match the paper, or ratify and
         write into the methods (with Raj).
      5. Dead code: function(s) defined but never called.
- [ ] Add a second "messy city" fixture tier (verified against
      `tests/reference_impl.py`, NOT hand arithmetic — Oraculum stays the
      hand-ratifiable ground truth for the math, deliberately small). Must
      cover the real-data pathologies Oraculum omits by construction:
      irregular non-rectangular polygons (so bbox ≠ geometry, the real
      regime), a MultiPolygon (556 of 4,357 real settlements are multi-part),
      an overlapping polygon pair, an isolated settlement (360 real ones have
      zero neighbors), a settlement with no population row (15 real ones),
      and an area-extreme sliver (real areas span 2.3e-9 → 29 km²). This tier
      is what would PROVE any fix to items 1 and 2 above.
- [~] Retire the notebooks entirely (decided): notebooks already removed
      in Phase 1 (logic lives in `scripts/`); remaining work is the package
      pipeline stages with logged validation (the notebooks' eyeball checks
      become assertions) and a figures command that renders to files
- [ ] Lift the `pandas<3` cap in `pyproject.toml` now that the oracle can
      validate the major-version jump; sweep any remaining Dependabot alerts
      at the same time (deferred here by decision — do not fix piecemeal)
- [ ] Add GitHub Actions CI running `uv run pytest` on every push/PR
      (decided in meta-planning "once the suite exists" — it now does; no
      `.github/workflows/` yet)

**Definition of done:** oracle suite still passes; one code path per concept;
settlement categories, services, and distance parameters are config, not code.

## Phase 4 — Settlement categorization (Raj decides, Bob implements) — P1

*The big analytical piece. Workshop consensus: ~10 Delhi-specific types are
too much detail — collapse into a small set of portable, theory-first
categories. Raj's conceptual work proceeds in parallel with Phases 1–3;
implementation lands here.*

- [ ] **Raj:** drop all non-urban categories (rural villages, industrial
      areas) from the entire analysis — figures and calculations; move their
      mention to footnotes. It's an urban project.
- [ ] **Raj:** decide the collapsed categories — working candidate from the
      workshop triage: **planned / unauthorized / regularized-unauthorized /
      resettlement colonies / JJCs** (5 categories). Theory-first (organized
      around property-rights security and legal service entitlements), no
      data-fishing. Run past Patrick; resolve the SDA question (missing from
      the current list; adding it may help the story).
- [ ] **Raj:** figure decisions from the triage — full map for spatial extent;
      breakdown charts show the 5 categories; feature the **JJC vs. planned
      juxtaposition**; remove the per-type data table (footnotes instead)
- [ ] **Bob:** encode the agreed mapping in the Phase 3 mapping layer
- [ ] **Bob:** recalculate all indexes with non-urban categories dropped
      (supersedes the current "no RV" run — industrial areas go too)
- [ ] Regenerate paper figures from the new run

**Definition of done:** new PSI outputs under the agreed categories, synced to
the shared drive with clearly dated filenames; figures updated.

## Phase 5 — Shippable minimum

Checkpoint, not a work phase: Phases 1 + 2 + 4 together produce the minimum
credible revision (correct code, verified index, non-urban dropped, new
categories). If the deadline bites, ship after Phase 5 and treat Phase 6 as
appendix material added in revision.

## Phase 6 — Robustness sweeps & measurement variants (Bob) — P2

*Mostly appendix "gravy": run the alternatives, report in footnotes/appendix,
keep the main tables unchanged if variants align. Not fishing — demonstrating
rigor. "Since Claude makes code easy to create, just do all the checks."*

Index formulation:
- [ ] Alternative formulations for the compressed 0–1 effect sizes ("make the
      values less small"): transformations (e.g. log), tested against the
      oracle first (the 2021 `Transforms for Skewed Data` exploration in
      `archive/master-2021/` is prior art — none were adopted then)
- [ ] **Rank-based index** (new idea from the workshop): instead of averaging,
      explore ranking mechanisms — average rank per settlement category, and
      composition of the top/bottom deciles by category (as in
      intergenerational-mobility research)

Distance / reachability:
- [ ] Distance-threshold sweep: 1 km / 5 km / 10 km; show index stability
- [ ] Parameterize and vary the decay weight 1/(1+D) (currently arbitrary);
      revisit centroid-to-centroid vs. other distance definitions
- [ ] Per-service distance expectations (a school may reasonably be farther
      than water); connects to the walkability/food-desert framing
- [ ] Adjacency-method comparison (bbox vs. touch) as a reported variant

Service-set / measurement variants (from the workshop triage):
- [ ] With/without **ration shops** sensitivity (demand-driven, targeted to
      the poor — check whether the result strengthens without them; either way
      it qualifies the argument)
- [ ] Decide on **ATMs/banking** (workshop said drop as private/market-driven;
      Raj's note: "they are material assets" — a with/without variant settles
      it empirically)
- [ ] Core-universal-services variant: schools, health, water only
- [ ] Facility size / capacity (intensive margin, not just counts) — P2,
      data-permitting
- [ ] Rejected in triage (do not pursue): roads as area instead of length

- [ ] Write up all variants in an appendix; keep main claims unchanged if
      variants align (or honestly flag if not)

**Definition of done:** appendix section with variant tables; main results
demonstrated stable (or divergences surfaced and discussed).

## Phase 7 — Release & ship

- [ ] Final repo cleanup for public release (README quickstart, data-access
      instructions, license check) — people will run the repo (and point
      Claude at it) first thing, so find issues before they do
- [ ] Optional: release the oracle/test harness and fixtures with the package
- [ ] Ship to HAS (and post to SSRN per Patrick's suggestion)

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

**A. Exclusion semantics.** "Drop rural villages and industrial areas" hides
three sub-decisions that change the numbers; Bob to put these to Raj,
informed by the mythical-city side-by-side demo (Phase 2). *Status (Aug
2026): the demo exists — `docs/oracle/exclusion-semantics-memo.md` has the
per-settlement delta tables; the oracle also showed sub-decision 1 is
currently forced to (b) by a silent `except: pass`, so (a) needs a code
change before it is even an option. Awaiting Bob sending the memo and Raj's
reply.*

1. **Neighbor treatment of dropped types** — do excluded settlements still
   contribute services to adjacent urban settlements' PCEN (Eq. 3), or are
   they removed entirely before neighbor computation? (The current "no RV"
   run removes them entirely — inherited, not chosen.)
2. **Min–max renormalization** — shrinking the settlement universe changes
   the min/max in Eq. 2, shifting every service index slightly; needs a
   sentence in the methods text.
3. **Descriptive tables** — dropped types also vanish from population-share
   and count tables; confirm the paper's descriptive claims are restated
   accordingly.

**B. Data-release posture** (Raj/group decision — not Bob's call alone).
Options, in ascending openness: code-only (repo + fixtures, runnable but
Delhi numbers not reproducible by outsiders); code + derived outputs
(publish the per-settlement PSI as CSV/GeoPackage — the paper's headline
dataset — without raw inputs); full archive (inputs + outputs on
Zenodo/OSF with DOI — requires a redistribution-rights check on the
DUSIB/DDA/MCD-derived data; WorldPop is CC-BY). Planning floor is
code + fixtures; anything more awaits the group.

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
2. In parallel: Raj settles categories with Patrick (Phase 4 decisions) and
   answers the oracle memo (exclusion semantics, bbox adjacency, overlaps)
3. **Phase 3** refactor + mapping layer is now the active work — it builds
   the configurable category layer Phase 4 plugs into
4. **Phase 4** implementation + recalculation (after Bob's hand ratification)
   → shippable minimum (Phase 5); **Phase 6** sweeps fill the appendix
5. **Phase 7** ship

Standing discipline (from the call): interrogate the design before letting
Claude run; verify and inspect its output — the oracle exists precisely to
make that verification mechanical.
