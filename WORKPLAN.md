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

Repo state this plan builds on (Aug 2026): `main` is the default branch with
the two canonical 2025 notebooks + `spatial_index_utils.py`; the 2021 code is
archived under `archive/master-2021/`; the full input dataset (276 MB) lives
locally at `~/delhi_data` and is two-way synced hourly with the shared drive
(`Spatial_Index_GIS/delhi_data/`).

---

## Phase 0 — Environment & data (DONE)

- [x] Repo restructured: `main` default, 2 canonical notebooks, old code archived
- [x] Data recovered from old machine account; complete input set verified
- [x] Durable local ↔ shared-drive sync (rclone service account + hourly systemd timer)
- [x] GitHub API access (`gh`) for automation
- [x] Claude account logistics resolved (Raj's gifted Max plan, redeemed 12 Aug)

## Phase 1 — Make the pipeline runnable end-to-end (Bob) — P1

*"Get the repo running / modernize dependencies", "remove hardcoded machine
paths". Everything downstream depends on this.*

- [ ] Remove hardcoded `data_dir = /home/bwbelljr/delhi_data/` — make the data
      directory configurable (env var `DELHI_DATA_DIR` or config file), so the
      pipeline runs on any machine pointed at a copy of the dataset
      (replicability matters: the repo will be released, and some journals
      require the data too)
- [ ] Modernize dependencies: **all packages to latest stable versions**
      (geopandas/shapely/pyproj have breaking API changes since 2021, e.g.
      removed `cascaded_union`; expect more of the same across the stack);
      consolidate to uv + `pyproject.toml` (decided — see Decisions section),
      removing the conda/poetry/Docker files; resolves the deferred
      Dependabot backlog. Safety net for upgrades: the Phase 1 verification
      that outputs still match the July 2025 run acts as the interim
      regression check until the oracle (Phase 2) takes over that job
      permanently.
- [ ] Fix small runtime hazards: create output dirs (`os.makedirs`), rename
      stale `12Sep2021` output filenames to dated 2025+ names
- [ ] Verify: both notebooks run top-to-bottom on this machine against
      `~/delhi_data`, outputs match the July 2025 run

**Definition of done:** a fresh clone + the shared-drive data reproduces the
existing PSI outputs.

## Phase 2 — The Oracle: ground-truth test harness (Bob) — P1

*Top code priority. Do this BEFORE trusting any recalculation, because the
index is the paper's core contribution.*

- [ ] Build a toy "mythical city": 2–3 settlement types, a handful of services
      and boundaries, small enough that the PSI can be computed by hand
- [ ] Hand-verify expected values (Bob + Raj do the back-of-envelope check;
      Claude generates the fixture, humans confirm the arithmetic)
- [ ] Encode as a pytest suite: oracle fixtures + expected PSI values as a
      permanent regression check on `spatial_index_utils.py`
- [ ] Use the oracle as the fix-loop target for any bug found in Phase 3, and
      as the safe sandbox for index-formula experiments in Phase 6
- [ ] Stretch: reproduce the South African source paper's published numbers
      (the index formulation was adapted from Patrick's paper) as a second
      validation case; consider releasing the harness with the package
- [ ] Decision-support variant: compute the mythical city's PSI under both
      exclusion semantics — (a) dropped settlement types still contribute
      services as neighbors vs. (b) dropped types fully removed before
      neighbor computation — and show the side-by-side delta to Raj to ground
      the open decision below (more informative than asking in the abstract)

**Definition of done:** `pytest` passes with hand-verified expected values;
any future code change that alters the index fails the suite.

## Phase 3 — Refactor & bug audit (Bob) — P2, gated on Phase 2

*Refactor with the oracle as a safety net.*

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
- [ ] Bug audit prioritizing anything that affects the index itself;
      specifically investigate:
      - the polygon/settlement adjacency logic (bbox overlap vs. touch)
      - dead code: function(s) defined but never called
- [ ] Retire the notebooks entirely (decided): their logic becomes package
      pipeline stages with logged validation; figures render to files via a
      figures command

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
informed by the mythical-city side-by-side demo (Phase 2):

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

1. ~~Phase 0~~ done → **Phase 1** (runnable pipeline) → **Phase 2** (oracle)
2. In parallel: Raj settles categories with Patrick (Phase 4 decisions)
3. **Phase 4** implementation + recalculation → shippable minimum (Phase 5)
4. **Phase 3** refactor interleaves after the oracle exists; **Phase 6**
   sweeps fill the appendix
5. **Phase 7** ship

Standing discipline (from the call): interrogate the design before letting
Claude run; verify and inspect its output — the oracle exists precisely to
make that verification mechanical.
