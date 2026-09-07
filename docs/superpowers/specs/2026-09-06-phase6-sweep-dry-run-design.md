# Phase 6 sweep harness + dry run on `code-2025` — design

**Ticket:** DEL-55 (serves DEL-36 distance bands, DEL-37 decay weights,
DEL-39 adjacency comparison)
**Branch:** `del-55-phase6-dry-run` off `main` at `715f6c0`
**Date:** 6 Sep 2026
**Authorisation:** Bob's 5 Sep 2026 three-cycle autonomous run, cycle 2 of 3.
Approved scope, verbatim: "Profiles for 1, 5 and 10 km bands and for the
decay forms, a sweep runner, and summary tables under `docs/data/` labelled
'dry run on code-2025, superseded by the ratified profile'. Committed numbers
are provisional and the tables say so."

---

## 1. What this is, and what it is not

Phase 6 reports **sensitivity variants** in the paper's appendix: does the
formal/informal finding survive the two choices that were arbitrary?

- **DEL-36** — who counts as a neighbour at a *distance* (1, 5, 10 km bands).
- **DEL-37** — how distance discounts a neighbour (the decay form, and what
  the distance even means).
- **DEL-39** — the adjacency-rule comparison (`bbox` vs `touch` vs the 0 km
  band).

Those three tickets must be run against the **ratified profile** (DEL-31),
which does not exist yet: Raj still owes answers on overlap lending, UV/SDA,
the two DEL-52 choices and Decision B. Their numbers are the ones that go in
the appendix.

**This ticket is the harness plus a dry run against `code-2025`.** Its
numbers are provisional *by construction* — `code-2025` still carries `bbox`
adjacency, `global_asymmetric` barriers and decayed roads, all three of which
Raj's 28 Aug decisions change. Nothing produced here is quotable, and every
table says so in its own caption.

Three things it buys:

1. The sweep machinery, its reporting shape, and its failure modes are proven
   before the numbers matter. When DEL-31 lands, Phase 6 is a re-run, not a
   design exercise.
2. The real cost is measured rather than discovered. Cycle 3D measured a 10 km
   `preprocess` at **2,491 s / 4,366,055 directed links**; nobody has yet paid
   for the `compute` on top of that, and that is the number that decides
   whether the 10 km band is affordable in the recalculation at all.
3. Any defect in the sweep path surfaces against numbers nobody will cite.

**Non-goals.** No new methodology values (every value below was shipped and
proven against the reference implementation in cycle 3D). No change to
`code-2025`, `manuscript`, or any existing expected value. No claim about
Delhi.

---

## 2. Deliverables

| # | Artefact | Purpose |
|---|---|---|
| D1 | 11 sweep profiles in `delhi_psi/profiles/` | one YAML per sweep point |
| D2 | `tests/test_sweep_profiles.py` | the one-factor guard: each profile differs from `code-2025` in exactly the keys it claims |
| D3 | 22 production fixtures (`tests/fixtures/<city>/production/<profile>.csv`) | a fixture-scale pin per profile, per § 3 of `docs/methodology-config.md` |
| D4 | new rows in `tests/variants.py` + `variants_expected_values.csv` | reference cross-check for the three parameter values 3D did not pin |
| D5 | `scripts/run_sweep.py` | runs the points, records cost, writes a manifest |
| D6 | `scripts/summarize_sweep.py` | reads outputs + manifests, emits the fenced blocks |
| D7 | `tests/test_run_sweep.py`, `tests/test_summarize_sweep.py` | unit proof of both scripts' pure logic |
| D8 | `docs/data/phase6_sweep.md` | the tables, every one labelled provisional |
| D9 | `CHANGELOG.md` `[Unreleased]` entry | per the /ship contract |

---

## 3. The sweep points

Twelve points. The baseline is **not re-run**: the proven `code-2025` outputs
already sit in `~/delhi_data/phase3_verify` (verified against the July 2025
baseline at `0.000e+00` on 60 comparisons), carry **both** denominators, and
are read read-only.

| point | profile | ticket | changed vs `code-2025` | own `preprocess`? |
|---|---|---|---|---|
| baseline | `code-2025` (existing run) | — | — | no — already on disk |
| own-only | *derived by the summariser* | — | — | no — arithmetic, § 6.2 |
| touch | `adj-touch` | DEL-39 | `adjacency.rule: touch` | yes |
| band 0 km | `band-0km` | DEL-39 | `adjacency.rule: within_distance`, `max_distance_km: 0.0` | yes |
| band 1 km | `band-1km` | DEL-36 | same, `1.0` | yes |
| band 5 km | `band-5km` | DEL-36 | same, `5.0` | yes |
| band 10 km | `band-10km` | DEL-36 | same, `10.0` | yes |
| no decay | `decay-none` | DEL-37 | `decay.form: none` | **shared**, § 4.2 |
| flat power | `decay-power05` | DEL-37 | `decay.form: inverse_power`, `exponent: 0.5` | shared |
| steep power | `decay-power2` | DEL-37 | `decay.form: inverse_power`, `exponent: 2` | shared |
| exp 2 km | `decay-exp2km` | DEL-37 | `decay.form: exponential`, `scale_km: 2.0` | shared |
| exp 5 km | `decay-exp5km` | DEL-37 | `decay.form: exponential`, `scale_km: 5.0` | shared |
| boundary D | `decay-boundary` | DEL-37 | `decay.distance: boundary` | shared |

**Why these decay points.** Raj's 28 Aug steer (decision log § 9): keep
1/(1+d) in km for the main text, and let the sweep favour forms that spread
the neighbour weights **up from zero** — `inverse_power` with exponent < 1,
`exponential` with a scale of a few km, `boundary` distance. `decay-none` is
the upper bound (every neighbour undecayed) and `decay-power2` the steep
contrast; without both ends, "spread up from zero" has nothing to be measured
against.

**Why `adj-touch` rather than reading `manuscript`.** `manuscript` changes
seven other switches at once, so its diff is not attributable to adjacency.
`adj-touch` is `code-2025` with exactly one key moved. That is the whole point
of the one-factor guard (D2).

---

## 4. Profile shape

### 4.1 The template

Every sweep profile is `code-2025` copied verbatim with:

- `profile:` renamed;
- **`paths.neighbors_artifact` omitted** for the five adjacency/band profiles,
  so each takes its per-profile default name and cannot overwrite another
  point's artifact (`code-2025` is the one profile that pins the legacy
  `colonies_neighbors.joblib`);
- `outputs.denominators: [popdensity]` — see § 4.3;
- `outputs.formats: [csv]` — the summariser reads CSV; `.shp` and `.joblib`
  would add ~18 MB per point for nothing. **This does not make the outputs
  small.** The CSV carries `geometry`, `centroid`, `nbrs_bbox` and
  `nbrs_dist_bbox`, so its size scales with the LINK count: the baseline's
  ~30 k links give 15 MB, and `band-10km`'s 4.37 M links will give a file
  two orders larger. The saving is real but bounded, and the summariser must
  therefore read with an explicit `usecols` (§ 6) rather than loading a
  quarter-gigabyte of neighbour lists it has no use for;
- the one or two `methodology:` keys the table in § 3 names, and nothing else.
- **No `paths.out_dir`** — a literal there would ignore `--data-dir` (§ 3 of
  `docs/methodology-config.md`).

`methodology:` must be written out in full (the loader requires every key);
"changed vs `code-2025`" above means changed *in value*, and the one-factor
test (D2) is what proves it.

### 4.2 The six decay profiles share one neighbours artifact

`pipeline.methodology_stamp` covers the **adjacency and barrier blocks only** —
decay is applied downstream in `compute`, so an artifact stays valid across a
decay change. All six decay profiles therefore pin

```yaml
paths:
  neighbors_artifact: colonies_neighbors_sweep-bbox.joblib
```

and one `preprocess` (run under `decay-none`) serves all six. This is safe by
construction, not by convention: `check_methodology_stamp` compares the
artifact's adjacency+barrier stamp against each profile's own and refuses a
mismatch, naming both profiles in the error. Six preprocesses of an identical
neighbourhood would be six identical answers at six times the cost.

### 4.3 One denominator for the sweep points, both for the baseline

`compute` runs `index_frames` — and therefore the whole per-link `pcen` loop —
**once per entry of `outputs.denominators`**. On the 10 km band that loop
walks 4.37 M directed links per amount column; a second denominator doubles
the single most expensive thing in this cycle to answer a question that is not
being asked.

So every sweep point reports **`popdensity`**, the denominator the paper's
Figure 4 uses (measured 5 Sep 2026, `docs/data/psi_columns.md`). The baseline
already carries both, at no cost, so the doc additionally carries a **one-off
denominator check**: the baseline's category ordering and Planned-vs-JJC
effect size under `pop` beside the same under `popdensity`. If the two agree,
the sweep's single denominator is justified in the document that used it; if
they disagree, that disagreement is itself a finding for DEL-52 and gets
written down.

This is a scoping decision made under autonomous authorisation, not a
methodological one. It is recorded here and in the doc's own caption so the
final Phase 6 run can reverse it deliberately.

### 4.4 Registration

Per `docs/methodology-config.md` § 3, each profile is added to `PROFILES` in
`scripts/generate_production_fixtures.py` and `tests/test_production_fixtures.py`,
and the fixtures regenerated. **No profile is added to `PROFILE_RULES` in
`tests/test_profiles_match_reference.py`**: a band or decay profile is
`code-2025`'s rules with one reference-pinned switch moved, so it matches
neither `"code"` nor `"ideal"` wholesale. § 3 says such a profile "is pinned by
its production fixture alone". That is the correct and only registration.

**§ 3's registration list is incomplete, and this ticket fixes it.**
`tests/test_config.py::test_both_profiles_ship` asserts the *exact* set of
shipped YAMLs (`sorted(shipped_profiles()) == ["code-2025", "manuscript"]`),
so dropping eleven files into `delhi_psi/profiles/` turns the suite red
deterministically — and following the documented procedure would not have
caught it. That test is updated here, and the missing step is added to
`docs/methodology-config.md` § 3 step 2 so the next profile does not repeat
the surprise.

`code-2025.csv` and `manuscript.csv` must not change. If either moves, that is
a **STOP**, not a regeneration.

### 4.5 Variant rows for the seven unpinned parameter values

Cycle 3D pinned `within_distance` at 0/0.25/0.75 km, `inverse_power` at
exponent 1 and 2, `exponential` at scale 1.0 km, and `boundary`. Seven values
this sweep uses have never been compared against the independent reference
implementation:

- `decay.form: none`
- `inverse_power`, `exponent: 0.5`
- `exponential`, `scale_km: 2.0` and `5.0`
- `within_distance` at `1.0`, `5.0` and `10.0` km

The standing rule since 3D is that a methodology value ships as a **row in
`tests/variants.py`**, cross-checked at 1e-12 against `tests/reference_impl.py`
on both fixture cities. Seven new rows, appended; `RULESETS` and every existing
expected value untouched.

**The band radii earn their rows, and an earlier draft of this spec said
otherwise on a false premise.** That draft claimed the fixture cities are
~200 m across, so 1/5/10 km would all be the same complete graph 0.75 km
already pins. Measured, they are not: Oraculum spans 4.0 × 3.0 km and the
messy city 21.0 × 2.4 km, and the three radii give genuinely different
neighbourhoods on both —

| radius | Oraculum undirected pairs | messy undirected pairs |
|---|---|---|
| 0.75 km (pinned in 3D) | 14 | 10 |
| 1 km | 16 | 13 |
| 5 km | 21 | 27 |
| 10 km | 21 | 48 |

More importantly, **1.0 km lands exactly on the `<=` boundary in both cities**
(two pairs at exactly 1000.000 m in Oraculum, one in messy), and 10.0 km lands
on it once in messy. Cycle 3D deliberately chose 0.25 and 0.75 km because they
"sit in a gap of BOTH cities' polygon-to-polygon distance lists (≥ 26 m
clearance), so no pair lands on the `<=` boundary" — this sweep cannot make
that choice, because 1/5/10 km are the methodological points DEL-36 asks for
and the Delhi run must use them.

So the fragility is pinned rather than avoided: the variant rows put both
implementations on the tie, and a test asserts the boundary pairs ARE included
(the rule is `<=`, inclusive) with the exact distances written down. If the two
implementations ever disagree about a distance that lands exactly on the
radius, this is where it surfaces — on seven settlements, not on 4,357.

**Addition-only is a hard condition.** The regenerated
`variants_expected_values.csv` must be a strict superset of the committed one,
verified by diff, not by the generator's own report.

---

## 5. The runner — `scripts/run_sweep.py`

```
uv run python scripts/run_sweep.py --group bands|adjacency|decay|all \
    --data-dir ~/delhi_data --work-dir ~/psi_sweep [--dry-run] [--only PROFILE]
```

**Behaviour.** For each profile in the group, in the order listed in § 3:

1. Decide whether `preprocess` is needed: the artifact is absent, or present
   with a stamp that does not match this profile. Never re-preprocess an
   artifact that already matches — that is the entire saving in § 4.2.
2. Run `preprocess` and `compute` as `delhi-psi` subprocesses (not in-process:
   a crash in one point must not take the runner down, and the CLI is the
   interface the docs tell a human to use).
3. Record, per point, into `<work-dir>/manifest/<profile>.json`: the profile,
   the resolved stamp, wall-clock seconds per stage, the artifact's directed
   link count and degree distribution summary, `n_settlements`,
   `n_barrier_flagged`, `n_reported`, `n_missing_population`, the output paths,
   the repo commit, and the run date passed in by the caller.
4. On failure: record the stage, the exit status and the last 40 lines of
   stderr in the manifest, and **continue to the next point**. One expensive
   point falling over must not cost the other ten. The summariser renders a
   failed point as a row of `FAILED` rather than omitting it.

**Guards.**

- `--work-dir` goes through `_measure_common.resolve_work_dir`, which refuses
  the data directory and any child of it. `~/delhi_data` is bisynced hourly to
  a shared drive; the sweep writes ~1.5 GB and none of it belongs there. The
  default work dir is `~/psi_sweep`.
- `--dry-run` prints the plan — which points, which stages, which are skipped
  because an artifact already matches — and exits without running anything.
  This is how the plan is checked before an hours-long run, and it is what the
  unit tests exercise.
- The runner never writes into `~/delhi_data/phase3_verify` and never passes
  it as `--out-dir`.

**Degree statistics are read once, by the runner, and never again.** The
stages are subprocesses, so the artifact is *not* in the runner's memory when
`preprocess` returns — the runner loads it exactly once, immediately after,
and writes only the summary (`n_links`, `deg_mean`, `deg_p50`, `deg_max`,
`n_isolates`) into the manifest. It happens between two stages rather than
during one, and nothing downstream repeats it.

**Measured afterwards, and it corrects this section's own premise:** the
10 km artifact is **83 MB**, not the ~1.4 GB estimated here by scaling the
9.5 MB baseline artifact by its 206× link count. joblib stores the neighbour
lists far more compactly than that, so the "peak-memory moment" this
paragraph was written to be careful about is not a concern at all. The
single-load design is still right — reading a file twice to build a string
is wasteful at any size — but it was justified by a number that was wrong by
a factor of seventeen.
A point that skips `preprocess` (the five later decay points) inherits the
degree summary from the manifest of the point that built the shared artifact,
named explicitly in its own manifest as `degree_from`.

---

## 6. The summariser — `scripts/summarize_sweep.py`

```
uv run python scripts/summarize_sweep.py --work-dir ~/psi_sweep \
    --baseline-dir ~/delhi_data/phase3_verify [--out docs/data/phase6_sweep.md]
```

Reads the per-point output CSVs and manifests plus the baseline run, computes
the statistics below, and prints fenced `block:` sections in the established
`docs/data/` shape (`scripts/_measure_common.render` / `parse_block`), which
the doc carries verbatim and a test re-checks.

Every CSV read passes an explicit `usecols`: the id, the type, `category`,
`population`, `area_km2`, the seven amount columns, the seven `*_pcen`
columns, `unnorm_psi` and `norm_psi`. The neighbour lists, the geometry and
the centroid are never loaded — at `band-10km`'s link count they are most of
the file (§ 4.1) and the summariser has no use for them.

### 6.1 The governing principle

**PSI levels are not comparable across sweep points.** Eq. 2's min-max runs
per run, and widening the neighbourhood inflates everyone at once. A table of
"mean PSI by category" across points would look rigorous and mean nothing.

Therefore: **every cross-point statistic is rank-based or a within-run
standardised effect size.** Levels appear only as diagnostics, never as a
comparison.

### 6.2 The two anchors

- **`bbox` baseline** — today's rule set; the continuity anchor.
- **`own-only`** — the null: every neighbour term set to zero, so each
  settlement is scored on its own services alone. It is the only point free of
  *both* choices under test, which makes it the primary anchor: every other
  point is an interpolation between it and full spatial smoothing.

The own-only anchor is **derived arithmetically by the summariser**, not run.
The output CSV carries the own counts (`<service>_count`, `road_length`),
`population` and `area_km2`, and the denominators are exactly
`popdensity = population / area_km2` and `pop = population`
(`delhi_psi/index.py:300-305`). So own-only PCEN is `own_count / denom`,
min-max normalised over the same reported universe, summed, and re-normalised
under `second_normalization: true`. No pipeline run, no new profile.

**Self-check:** `own_share = own_pcen / pcen` must lie in `[0, 1 + 1e-9]` for
every row, because the neighbour term is non-negative. It is a real guard
against a mis-shaped reconstruction, and it is worth stating what it does
*not* prove: the denominator **cancels** in that ratio, so a wrong denominator
formula passes it. What actually pins the denominator is the own-only anchor's
own arithmetic, and the test that a settlement with an empty neighbour list
has `own_share == 1` exactly.

**`own_share` is undefined, not zero, where `pcen == 0`.** A settlement that
owns nothing and receives nothing gives 0/0 — and 1,834 of the baseline's
4,131 reported settlements own zero of all seven services, so this is the
common case rather than an edge case. Those rows are `NaN`, excluded from the
median, and counted as `n_own_share_undef` beside it. Silently reading them as
0 would drag `own_share_p50` toward the `smoothed` flag on every point.

### 6.3 Block `points` — one row per sweep point

Identity: `point`, `profile`, `adjacency`, `radius_km`, `decay_form`,
`decay_param`, `decay_distance`.

Structure and cost: `n_reported`, `n_isolates`, `n_links`, `deg_mean` (1 dp),
`deg_p50`, `deg_max`, `preprocess_s`, `compute_s`.

The baseline point has no manifest — it was run in August, not by this
runner — so the summariser reads its structure straight off the proven
artifact `~/delhi_data/phase3_verify/colonies_neighbors.joblib` (9.5 MB;
read-only, and the only artifact the summariser ever opens). Its
`preprocess_s` / `compute_s` are written as `—`, not invented. The own-only
anchor has no neighbourhood at all: its structure columns are `—` and its
`own_share_p50` is 1.000 by construction.

Composition: `own_share_p50` (3 dp) — the median over settlements of
own / (own + neighbour), pooled over the seven amount columns (six point
services and `road_length`). This is the
number that says how much of the index is still a property of the settlement.
Below 0.10 the index is a spatial smooth of the city and the row is flagged.

Outcome, all rank-based:

- `cat_order` — the ten categories sorted by **mean percentile rank** of PSI
  within the run (each settlement's percentile, 0–100), written
  `Planned>SDA>UV>…`
- `tau_vs_own`, `tau_vs_bbox` — Kendall τ between this ordering and each
  anchor's (2 dp)
- `rho_vs_own`, `rho_vs_bbox` — Spearman ρ of settlement-level PSI (3 dp)
- `taub_vs_own`, `taub_vs_bbox` — Kendall τ-b (3 dp; τ-b because ties at
  PSI = 0 are common under `touch` and universal under own-only)
- `jaccard_top10`, `jaccard_bottom10` — overlap of the top and bottom decile
  *sets* with the `bbox` baseline (3 dp). ρ alone hides a tail reshuffle
  behind a stable middle, and the paper's claim lives in the tails.

  **These cells are tie-gated, and that is not a detail.** PSI has a mass
  point at exactly zero that is *larger than a decile*: on the baseline,
  452 of 4,131 reported settlements sit at `norm_psi == 0` against a decile
  of 413, and on the own-only anchor 1,834 do. So the bottom-decile cut falls
  *inside* a tie block, and which settlements land in "the bottom 10 %" is
  then decided by pandas' sort order, not by the index. Measured: the same
  two series give `jaccard_bottom10` of 0.070 in file order and 0.121–0.157
  under twenty random row permutations — the number moves more than twofold
  on nothing at all, and prints to 3 dp either way.

  The rule, applied to both ends and stated in the caption:

  1. The decile set is **tie-inclusive** — every row whose value ties the
     k-th is in it, so the set is a property of the numbers alone.
  2. If that set exceeds **1.5 × k**, the cell renders `—` and the tie mass
     is printed beside it. A "decile" that is 44 % of the universe is not a
     decile, and an em dash is the honest rendering.

  On today's numbers `jaccard_top10` survives (exactly one row sits at
  `norm_psi == 1`, and 412 above the cut) and `jaccard_bottom10` renders `—`
  against the own-only anchor. That is the correct outcome, not a failure:
  the tail comparison the spec wanted is unavailable at the bottom, and the
  document says so rather than printing a number that measures sort order.
- `planned_gt_jjc` — TRUE/FALSE
- `flag` — see § 6.6

### 6.4 Block `ordering` — is the ordering stable?

Rows are sweep points, columns the ten categories, each cell the category's
rank with a bootstrap 95 % rank interval: `1 [1-2]`. 1,000 settlement-level
resamples, stratified by category, seed 0, reported in the caption.

**Honesty about ties is the interval itself.** A rank interval wider than one
position *is* the tie report; no separate tie test. Plus `n_fragile_pairs`:
the number of adjacent pairs in `cat_order` whose bootstrap flip probability
exceeds 0.05.

The bootstrap treats a complete census as a sample. It is therefore a
**composition-sensitivity** device — "would this ordering survive a different
draw of settlements" — and the caption says exactly that rather than implying
sampling inference.

### 6.5 Block `gap` — the formal/informal gap as an effect size

One row per point:

- `cliffs_delta_planned_jjc` (2 dp) with a 95 % bootstrap CI. **Primary.**
  Fully rank-based, invariant under any monotone transform (min-max included),
  and it reads in plain words: `P(a random Planned settlement outranks a
  random JJC) = (δ+1)/2`.
- `cohens_d_planned_jjc` (2 dp). Secondary, reported because reviewers expect
  it; min-max is affine so d survives it, but d is not invariant to the
  smoothing itself, and its divergence from δ is the tail-behaviour signal.
- `pct_gap_planned_jjc` (1 dp) — difference in mean percentile rank.
- `p_planned_gt_jjc` (3 dp) — bootstrap probability that the Planned mean
  exceeds the JJC mean. If it comes out 1.000 at every point — which at
  n = 4,131 is the likely outcome — the doc states that **in one sentence**
  and drops the column. A column of identical 1.000s carries no information
  and invites exactly the significance reading § 6.9 forbids.
- `top_decile_share_planned`, `top_decile_share_jjc`,
  `bottom_decile_share_jjc` (3 dp) — built on the **same tie-gated decile
  sets as § 6.3**, and rendered `—` under the same 1.5 × k rule. A "bottom
  decile" holding 44 % of the universe would make all three of these
  meaningless in exactly the way the Jaccard cells are.

Then the same row for the pooled blocks **formal = {Planned, SDA} vs informal
= {JJC, JJR}**. That grouping is a *stated assumption of this document*, not a
ruling from the paper; UV, UAC, RUAC, RV, Industrial and Other are
deliberately left out of the pooling because their placement is exactly what
DEL-53 and the UV/SDA question are still open on. The caption says so, and the
per-category table above carries all ten regardless.

### 6.6 Degenerate-run flags

Computed automatically and printed in the `flag` column; a flagged row stays in
the table, greyed, and the prose says why:

- `isolates` — **more isolates than the `bbox` baseline**, not `n_isolates > 0`.
  Corrected 6 Sep 2026 against the measured decay group: the baseline itself
  has **360** isolated settlements out of 4,131 reported. They are a property
  of `code-2025`'s `global_asymmetric` barrier rule, which severs every link
  *into* a flagged settlement — nothing to do with the factor under test. As
  originally written the flag would fire on every row including both anchors,
  which makes it a constant, and a constant is not a flag. (One of the plan
  review's killed findings predicted exactly this and was refuted 3–0; the
  real data says the refuters were wrong. `n_isolates` is reported as a number
  regardless — see § 6.7.)
- `smoothed` — `own_share_p50 < 0.10`
- `pinned` — exactly one settlement at norm_psi = 1 while the 99th percentile
  is below 0.5 (one outlier compressing everyone)
- `reshuffled` — `rho_vs_own < 0.5` together with `n_fragile_pairs >= 3`

`band-10km` with `decay-none` characteristics is expected to flag. That is the
finding, not a failure.

### 6.7 Diagnostics that must accompany every point

`n_pcen_undef` (denominator zero or missing — these are excluded from the
min-max and counted, and `n_reported` is stated after exclusion),
`n_pcen_zero_all` (every amount column zero), `n_psi_pinned0`, `n_psi_pinned1`.
Per-service `own_share_p50` goes in the CSV bundle, not the doc.

### 6.8 Deliberately dropped, with reasons

- **Moran's I under a fixed touch contiguity matrix.** The right statistic for
  "how much smoothing was introduced", and genuinely tempting. It needs a
  separate touch-contiguity matrix over 4,357 polygons built and held outside
  the sweep, and it answers a question `own_share_p50` already answers well
  enough for a dry run. Reconsider for the DEL-31 run.
- **Effective degree** `median_i Σ_j w_ij·decay(D_ij)`. The most direct reading
  of "spread up from zero", but it is a link-level quantity and the outputs are
  settlement-level; getting it means reloading a gigabyte artifact per point.
  `own_share_p50` is its settlement-level shadow. Reconsider for DEL-31, where
  it can be computed inside `compute` for free.
- **`n_single_neighbour_dominated`** — link-level, same reason.
- **The full 45-pair flip-probability matrix** — kept in the CSV bundle,
  never rendered.

### 6.9 What must never be reported

Stated here so a future run cannot quietly add it:

- Mean or median PSI by category **compared across points**. Within a point
  only, and even then in the CSV, not the doc.
- Differences or percentage changes in PSI between points.
- Pearson correlation of PSI between points.
- t-tests, ANOVA, Kruskal–Wallis, or significance stars. At n = 4,357 every
  gap is significant at every point; a p-value here would manufacture a
  robustness finding out of sample size.
- Gini of PSI — not shift-invariant, and min-max resets the zero.
- More than 2 dp on an effect size or 3 dp on a correlation.
- **Any number from this run in the manuscript.** Every block and every table
  caption carries `rule_set: code-2025 (frozen July 2025 behaviour) — DRY RUN,
  superseded by the ratified profile`.

---

## 7. Cost, and the order the real run happens in

Known: a 10 km `preprocess` is 2,491 s / 4,366,055 links (28 Aug, cold dedup
cache). Unknown, and the thing this run measures: `compute` on that link count.

The run therefore goes **cheapest first**, and each point's measured cost
informs the next:

1. `decay-none` — the shared `bbox` preprocess (warms the dedup cache for
   everything after it) plus one compute. Establishes the per-link compute
   rate at the baseline's ~30 k links.
2. The other five decay points — compute only, no preprocess.
3. `adj-touch`, `band-0km` — cheap preprocesses, comparable link counts.
4. `band-1km` — the first band; extrapolate from its link count.
5. `band-5km`.
6. `band-10km` — last, alone, in the background.

**Ruling, made in advance so nobody improvises it mid-run:** if the projected
10 km `compute` exceeds six hours, it still runs — that projection *is* the
deliverable for the recalculation budget — but it runs last and alone, and the
doc records the projection beside the measurement. If it exceeds 24 hours it is
killed, and the doc records the extrapolated cost and the kill, which is a more
useful answer for DEL-31 than a number nobody waited for.

Disk: ~1.5 GB under `~/psi_sweep`, dominated by the 5 km and 10 km artifacts.
96 GB free. Nothing under `~/delhi_data`.

---

## 8. Guards, and what stops the run

**Hard stops** — halt and report, do not work around:

1. Any existing expected value moves (`expected_values.csv`,
   `variants_expected_values.csv` deletions or modifications, any
   `production/*.csv` for `code-2025` or `manuscript`).
2. The baseline `code-2025` real-data numbers move.
3. `own_share > 1 + 1e-9` anywhere (§ 6.2) — the reconstruction is mis-shaped
   and every derived statistic with it. (`NaN` from `pcen == 0` is expected,
   counted, and is *not* a stop.)
4. Anything writing under `~/delhi_data` outside an explicit `--out-dir`.

**Not a stop, by design:** a sweep point that fails or is degenerate. It is
recorded, flagged, and the run continues (§ 5, § 6.6).

---

## 9. Task breakdown

| Task | Deliverable | Test |
|---|---|---|
| 1 | The five adjacency/band profiles + registration + fixtures + the `test_config.py` and `methodology-config.md` § 3 registration fix | `test_sweep_profiles.py` one-factor guard (methodology **and** categories); regenerated fixtures byte-match |
| 2 | The six decay profiles (shared artifact) + registration + fixtures | same guard extended; a test that all six pin the same `neighbors_artifact`, and that it is not the baseline's |
| 3 | Seven new `tests/variants.py` rows + regenerated variant expectations + the `<=` boundary pin | addition-only verified by diff; the 1 km tie pairs asserted present with their exact distances |
| 4 | `scripts/run_sweep.py` | `test_run_sweep.py` — plan construction, skip-if-stamp-matches, work-dir guard, failure isolation, `--dry-run` |
| 5 | `scripts/summarize_sweep.py` statistics core | `test_summarize_sweep.py` — every statistic on hand-built frames with hand-computed answers |
| 6 | Rendering + `docs/data/phase6_sweep.md` skeleton | block round-trip through `parse_block` |
| 7 | The real run; fill the doc; CHANGELOG | the doc's numbers match a re-run of the summariser |

Tasks 1–3 touch fixtures and must not run concurrently with each other.
Tasks 4–6 are independent of 1–3 and of each other.

---

## 10. Decision log for the unattended window

Per the /ship contract's "Unattended runs" section, decided **before** the
window rather than at 2 a.m.:

1. **Plan vs reviewer conflict.** A CONFIRMED Critical finding governs over
   the plan; deviate, fix, and record the deviation in the ledger, the
   CHANGELOG and the report. Every non-Critical conflict follows the plan and
   is surfaced for adjudication.
2. **Autonomy scope.** Fix forward on failure: yes. Commit: yes. Push: yes.
   Open and merge the PR: yes (Bob's 5 Sep authorisation). Send anything to
   anyone: **no** — drafts only, never sent.
3. **Failure policy.** Load-bearing (a failure halts dependents): the profile
   loader accepting every sweep profile; the fixture regeneration being
   addition-only; the own-share self-check. Best-effort (log, flag, continue):
   any individual sweep point's real-data run.
4. **Decisions that belong to Bob or Raj** are not invented here. If one
   surfaces, it is escalated to Fable first per Bob's 5 Sep rule; if Fable
   cannot answer it, it is written to the decision log and the ticket moves on
   without it. Nothing in this cycle currently depends on one — that is why
   this cycle was chosen while Raj's answers are outstanding.
5. **Implementers never run the full suite** (background completion
   notifications reach only the controller session, and sub-agents have
   stalled on this five times). Implementers run their own test files; the
   controller runs `uv run pytest -q -W error`.

---

## 11. Definition of done

- `uv run pytest -q -W error` green, with the new tests in it.
- `code-2025` and `manuscript` production fixtures byte-identical to `main`.
- `expected_values.csv` byte-identical; `variants_expected_values.csv`
  addition-only, verified by diff.
- Eleven sweep profiles load, and each differs from `code-2025` in exactly the
  keys § 3 names.
- `scripts/run_sweep.py --dry-run` prints the full plan.
- Every sweep point either has a manifest with timings or a recorded failure.
- `docs/data/phase6_sweep.md` carries the blocks, every caption labelled
  `DRY RUN on code-2025 — superseded by the ratified profile`.
- CHANGELOG updated; PR opened and merged; DEL-55 closed with evidence.

---

## 12. What the plan review changed

One ultracode round: five lenses (attributability, statistics, APIs,
operational safety, test quality), 30 gating findings, each put to three
independent skeptics instructed to refute it. Three survived, and one that did
*not* survive was right anyway — recorded here because "the skeptics killed it"
is not the same as "it was wrong", and the difference is the reason to read a
killed list rather than only a confirmed one.

**Survived and applied:**

1. **The bottom decile is decided by sort order, not by the index** (§ 6.3).
   452 baseline rows and 1,834 own-only rows sit at exactly `norm_psi == 0`,
   against a decile of 413 — so `jaccard_bottom10` moved from 0.070 to 0.157
   across twenty random row permutations of the same data. Now tie-inclusive
   and gated to `—` past 1.5 × k, and the same rule applies to § 6.5's three
   decile-share cells, which had the same defect.
2. **Eleven new YAMLs turn the suite red on a test the documented procedure
   never mentions** (§ 4.4). `test_both_profiles_ship` asserts the exact set of
   shipped profiles. Under this spec's own rule that implementers run only
   their own test files, that failure would have surfaced *after* the
   multi-hour real-data run. `docs/methodology-config.md` § 3 gains the missing
   step.
3. **scipy is not a dependency of this project** — not direct, not transitive,
   not in `uv.lock`. The plan assumed it. Spearman ρ and Kendall τ-b are now
   implemented on numpy, which also avoids promoting a heavyweight runtime
   dependency for code that lives in `scripts/`.

**Refuted 3–0 by the skeptics, and correct regardless** (§ 4.5): the claim that
the fixture cities are ~200 m across, so the 1/5/10 km radii would all be the
same complete graph. Measured, Oraculum spans 4 × 3 km and the messy city
21 × 2.4 km; the three radii give different neighbourhoods on both, and 1 km
lands *exactly* on the `<=` boundary in both cities. The band radii now get
variant rows — the fragility is pinned on seven settlements rather than
discovered on 4,357.

**Applied from the killed list on their merits**, without re-litigating their
severity: `outputs.formats: [csv]` does not make the outputs small (the CSV
carries the neighbour lists, so the summariser reads with explicit `usecols`,
§ 6); `degree_report` cannot run "while the artifact is already in memory"
because the stages are subprocesses (§ 5 now says the runner loads it once,
and what that costs); `own_share` is 0/0 for the 1,834 settlements that own
nothing (§ 6.2 makes those `NaN` and counts them); the one-factor guard
inspected only `methodology`, so it now compares `categories` too; and the
spec said "eight amount columns" where there are seven.
