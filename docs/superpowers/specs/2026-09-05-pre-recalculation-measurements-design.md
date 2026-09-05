# Pre-recalculation measurements — design (DEL-49, DEL-50, DEL-51, DEL-52)

**Status:** approved for /ship by the owner's standing instruction of
5 Sep 2026 ("create a spec based on your best defaults/recommendations …
run ultracode review on the plan. Then the plan is approved. Then follow
/ship"). The brainstorm was non-interactive; every choice the skill would
have put to the owner is recorded in § 8 with its reasoning.

**Why this cycle exists.** Raj's 28 Aug 2026 decisions
(`docs/decisions/2026-08-28-raj-methodology-decisions.md`) left four
questions that need NUMBERS before the ratified profile (DEL-31) is written
and the recalculation (DEL-32) runs, and before the batched reply to Raj can
be sent. All four are measurements of the real layers or the published
outputs. **No methodology changes, no new shipped profile, no fixture
changes, no writes under the baseline data.**

| Ticket | Question | Deliverable |
|---|---|---|
| DEL-49 | How do JJCs fare on roads, and what does `roads: eq4_own_only` do to today's numbers? | `scripts/measure_roads_access.py` → `docs/data/roads_access.md` |
| DEL-50 | Do any real settlement pairs touch only at a corner? | `count_corner_only_pairs` in `scripts/measure_layer_pathologies.py` → new key in `docs/data/layer_pathologies.md` |
| DEL-51 | Where did the barrier layers come from? | `scripts/inventory_barriers.py` → `docs/data/barriers.md` |
| DEL-52 | Which PSI column and which denominator do the paper's figures report? | `scripts/measure_psi_columns.py` → `docs/data/psi_columns.md` |

---

## 0. Goals and non-goals

**Goals**

1. Each question above answered by a script that is re-runnable on demand,
   with the number it prints carried verbatim in a document under
   `docs/data/`, and a test that the document matches the script's parser
   (the `layer_pathologies.md` pattern, § 4).
2. Every script READ-ONLY over `--data-dir`. Scratch output under a
   `--work-dir` that the script refuses to place inside the data directory
   (the `measure_layer_pathologies.py` guard, reused).
3. Unit tests on the fixture cities for every counting function, so the
   logic is proven on geometry we can reason about before it runs on 4,357
   polygons.
4. A written finding per ticket in the results document — not just numbers
   — phrased so the batched reply to Raj can quote it.

**Non-goals**

- No change to `delhi_psi/` behaviour. The one place production code is
  touched is a small, additive helper if the roads script needs one (§ 2.1);
  everything else is `scripts/` and `tests/`.
- No decision is taken here. DEL-52's finding is evidence for Raj (and
  changes Bob's proposed default — § 2.4); the decision log is updated by
  the cycle's final task, not the code.
- No figure. Tables in Markdown only.
- The JJC road-access question is answered on the SETTLEMENT layer and the
  MAJOR-ROAD layer the pipeline already uses; no new data.

## 1. Where things live

```
scripts/
  measure_layer_pathologies.py   # + count_corner_only_pairs (DEL-50)
  measure_roads_access.py        # DEL-49 (new)
  inventory_barriers.py          # DEL-51 (new)
  measure_psi_columns.py         # DEL-52 (new)
  _measure_common.py             # shared: work-dir guard, fenced-block
                                 #   render/parse, settlement loader (new)
docs/data/
  layer_pathologies.md           # + corner_only_pairs key
  roads_access.md                # new
  barriers.md                    # new
  psi_columns.md                 # new
tests/
  test_layer_pathologies.py      # + corner-only tests on the messy city
  test_measure_roads_access.py   # new
  test_inventory_barriers.py     # new
  test_measure_psi_columns.py    # new
  test_measure_common.py         # new
```

`_measure_common.py` exists because three scripts would otherwise copy the
same forty lines. It holds exactly: `resolve_work_dir` (the guard),
`load_settlements(cfg, cache_dir)` (moved out of
`measure_layer_pathologies.py`, which imports it back), `render(report)`,
`parse_block(text)`, and `FENCE`. Nothing else. Each script's `measure()`
returns an ordered mapping; `render` prints it; the doc carries the fenced
block; the test parses both with `parse_block` and compares (§ 4).

## 2. The four measurements

### 2.1 DEL-49 — JJC road access and the roads one-factor effect

**Two questions, one script, two fenced blocks.**

**(a) Road access on the layer — a description, no PSI.** For every
settlement in the deduplicated, reprojected universe (the same
`load_settlements` the pathology script uses, so the counts describe what
the pipeline scores):

- `road_inside`: the settlement's polygon intersects at least one feature of
  the major-road layer with positive length (`geom.intersection(roads).length
  > 0`; a road that merely touches the boundary at a point is not "inside").
  This is the same membership the pipeline uses for road length
  (`delhi_psi.index.road_lengths` clips the road layer to the polygon), so
  `road_inside` ⇔ `road_length > 0` in today's outputs. The script asserts that equivalence against the
  `code-2025` output CSV when `--verify-dir` is given (§ 2.1 (c)).
- `road_via_neighbor`: not `road_inside`, but at least one `touch`-neighbour
  (positive shared length — Raj's adjacency rule, DEL-19) has `road_inside`.
- `no_road`: neither.

Reported per reported settlement type (the seven that survive the 28 Aug
exclusion: Planned, UAC, RUAC, JJC, JJR, UV, SDA) plus the three dropped
types and a total, as counts and shares. Under `touch` a neighbour may be a
dropped type; that is correct — dropped settlements still lend (semantics a)
and a road in a rural village next door is exactly what Raj asked about.

Block 1 keys, one line per type:
`road_inside_<TYPE>`, `road_via_neighbor_<TYPE>`, `no_road_<TYPE>`, and the
same three with `_total`. Shares are derived in the doc's prose, not in the
block (the block carries integers only, so the drift test is exact).

**(b) The one-factor effect.** Run `compute` with `code-2025` changed in
ONE value, `roads: eq4_own_only`, against the SAME neighbours artifact, and
diff against the `code-2025` outputs by type. Mechanics:

1. `--verify-dir` names an existing, complete `code-2025` run
   (`~/delhi_data/phase3_verify`: `colonies_neighbors.joblib`, the four
   `*.dedup.gpkg` + `.stamp` caches, and both output CSVs). Read-only.
2. The script creates `<work-dir>/roads-own-only/`, copies the artifact in
   under the name the derived profile will look for
   (`colonies_neighbors_roads-own-only.joblib`) — the artifact ALONE:
   `pipeline.compute` never reads the `*.dedup.gpkg` caches, only
   `preprocess` does — and writes `roads-own-only.yaml`: `code-2025.yaml` loaded with PyYAML,
   `profile` set to `roads-own-only`, `methodology.roads` set to
   `eq4_own_only`, `paths.neighbors_artifact` deleted (so the per-profile
   default applies), `paths.out_dir` deleted. Every other key byte-equal.
3. It calls `delhi_psi.pipeline.compute(load_config(<yaml path>,
   data_dir=…, out_dir=<work-dir>/roads-own-only))` in-process. The stamp
   check passes because `roads` is not part of the neighbours stamp
   (`pipeline.methodology_stamp` — adjacency and barrier only); the script
   asserts that before running so a future stamp change fails loudly rather
   than silently re-preprocessing.
4. For each denominator (`pop`, `popdensity`) it joins the two output CSVs
   on `USO_AREA_U` and reports, per type: `n`, mean `road_idx` under
   `decayed` and under `eq4_own_only`, mean `unnorm_psi` under both, and the
   share of settlements whose `road_idx` fell to exactly 0 (they had no road
   of their own and lost everything they borrowed). Block 2 keys:
   `road_idx_decayed_<denom>_<TYPE>`, `road_idx_own_<denom>_<TYPE>`,
   `psi_decayed_<denom>_<TYPE>`, `psi_own_<denom>_<TYPE>`,
   `road_idx_zeroed_<denom>_<TYPE>` (integer count), for the reported
   types and `_total`. Means formatted `%.6g` like the areas in the
   pathology block.
5. Every row of `code-2025`'s own output must be reproduced by the script's
   `decayed` side; it reads that side FROM `--verify-dir`, never recomputes
   it, so the comparison is against the proven run.

**(c) Why the roads switch does not need a new `preprocess`.**
`index.road_lengths` is computed in `compute`, and the `decayed` /
`eq4_own_only` branch lives in `pipeline.index_frames` (`pipeline.py` ≈ 187). The
artifact carries neighbour lists and centroid distances only. One `compute`
on the warm cache is minutes, not the 11-minute preprocess.

**What the doc says.** `docs/data/roads_access.md`: the two blocks, then
three sentences of finding per block written from the numbers (how many JJCs
have a road inside vs only next door; how far JJC and planned mean PSI move
under own-only; whether the JJC-vs-planned ordering holds). It ends with the
recommendation as it stands after the numbers — the roads decision is final
unless the JJC shift is large enough to reopen it, and "large" is stated as
a number in the doc, not left to the reader.

### 2.2 DEL-50 — corner-only contact pairs

Added to `scripts/measure_layer_pathologies.py`, next to
`count_overlapping_pairs`, using the same sjoin-then-test shape:

```python
def count_corner_only_pairs(gdf, *, id_col):
    """Pairs whose intersection is NON-EMPTY but has zero length and zero
    area — they meet at one or more isolated points. `touch` (positive
    shared length) does NOT make them neighbours; the 0 km band does."""
```

Predicate on the sjoin candidates (`intersects`, `left < right`):
`inter = gi.intersection(gj)`; count iff `not inter.is_empty and inter.length
== 0 and inter.area == 0`. Shapely returns a Point / MultiPoint for such
pairs; the test is on measures, not on `geom_type`, so a GeometryCollection
of points also counts and a collection containing a line does not.

Two keys in the block: `corner_only_pairs` (the count) and
`corner_only_settlements` (distinct settlements involved). The doc gains a
definition paragraph and the finding sentence: if the count is zero, "the
`touch` rule and the intersection rule pick the same neighbours on this
layer; the corner question is moot"; otherwise the count and the note that
these pairs are neighbours under a 0 km band and not under `touch`, for Bob
to rule on and Raj to hear. **Fixture pin:** on the messy city
`count_corner_only_pairs` is exactly 1 (`L`/`T`, `docs/oracle/messy-city.md`)
and on Oraculum 0 — hand-checkable, so the function is proven before it
meets the real layer.

### 2.3 DEL-51 — barrier layer inventory

Raj asked where the barrier layers came from. What the repo can establish
mechanically is an INVENTORY; the provenance sentence itself is partly a
fact-finding question. `scripts/inventory_barriers.py` prints, per layer
named in `cfg.layers.barriers` (canal / railway / drain) and — with
`--all-candidates` — the two other copies found on 5 Sep 2026
(`Barrier_Clip/Canal/new/checked_Canal.shp`; the unclipped
`canal.data/`, `railway.data/`, `drain.data/` at the data root):

- feature count, geometry types, CRS, total length in km (after reprojecting
  to `cfg.crs.epsg`), and the bounding box against the settlement layer's;
- the attribute schema (column names) and the distinct values of the
  name-like columns (`CAN_NM`, `RL_ZONE`, `Drain_Name`, `DISTRICT`,
  `AC_NAME`), capped at 20 per column;
- the ESRI metadata sidecar's `CreaDate` / `CreaTime` / `ModDate` if a
  `.shp.xml` exists, and whether a `.qpj` (QGIS) sidecar exists;
- how many settlements each layer flags under today's rule, and how many
  under any layer (`neighbors.combine_barrier_flags` with the profile's
  `barrier.combine`), so the doc can say "595 flagged" from the same code
  path as production.

The doc, `docs/data/barriers.md`, carries the fenced block (counts, lengths,
dates, flagged settlements — the drift-tested part) and a prose section
"What this tells us about provenance", written from what the survey found on
5 Sep 2026 and re-confirmed by the run: the attribute schemas
(`CAN_NM` / `EL_GND` canal names with ground elevation, `RL_ZONE = NORTHERN
RAILWAY`, `Drain_Name` / `AC_NAME` / `DISTRICT` / `MAINTAINED`) are those of
an official Delhi GIS source, not of hand-digitised lines; the sidecars say
the clipped files were created in ArcGIS on 2 Aug 2020; the paper (p. 15–16)
says the team "manually marked areas that had river or railroad tracks",
which describes the CLIPPING/selection step, not the digitising of the
lines. The section ends with the question that remains for Bijoy, phrased
for the batched reply: which agency's layer, and what "manually marked"
covered. **No claim about the agency is made** — the schemas suggest it, and
the doc says "suggest".

### 2.4 DEL-52 — which PSI column, which denominator

**What the paper prints.** The April 2026 draft's Figure 4 ("Mean public
service index by settlement", p. 40 of the PDF) is a bar chart of the mean
PSI per settlement type with 95 % whiskers, y-axis labelled **"Mean Public
Services Index (per person per square kilometer)"**, eight bars (JJR, JJC,
SDA, Planned, RUAC, UAC, UV, Industrial — no RV, no Other), and footnote 12
says the average "rarely exceeds 0.05". Read off the figure on 5 Sep 2026:

| type | bar (≈) | type | bar (≈) |
|---|---|---|---|
| JJR | 0.037 | RUAC | 0.021 |
| JJC | 0.0015 | UAC | 0.017 |
| SDA | 0.028 | UV | 0.038 |
| Planned | 0.044 | Industrial | 0.038 |

Two things follow before any code runs: the axis label says the figure used
the **population-density denominator**, and means of order 0.02–0.04 are
what `unnorm_psi` (the Eq. 1 mean of min-maxed service indices) produces,
whereas a second min-max would stretch the column to [0, 1]. The script
turns both into a measured match.

**The script.** `scripts/measure_psi_columns.py --baseline-dir
~/delhi_data/psi_2020_results` reads the two July 2025 baseline CSVs
(`delhi_psi_bbox_popsize2020_norv_12Sep2021.csv`,
`delhi_psi_bbox_popdensity2020_norv_12Sep2021.csv` — the files
`verify_against_baseline.py` already treats as the read-only truth) and, for
each of the four candidates {`unnorm_psi`, `norm_psi`} × {popsize,
popdensity}, computes the mean per `USO_FINAL` type. It then scores each
candidate against the eight figure values above (carried in the script as
`FIGURE_4_BARS`, with the read-off tolerance `0.002`): the number of bars
matched within tolerance, and the maximum absolute gap. Block keys:
`mean_<column>_<denom>_<TYPE>` for all four candidates and every type in the
file, `matched_<column>_<denom>` (integer 0–8) and `maxgap_<column>_<denom>`
(`%.4f`), and `best_candidate` (the `<column>_<denom>` with the most matches;
ties broken by smaller max gap).

`--verify-dir ~/delhi_data/phase3_verify` (optional) additionally computes
the same means from the `code-2025` outputs and asserts they equal the
baseline's to 1e-9 per type — the cheap proof that the comparison would come
out the same on the refactored pipeline.

**The doc.** `docs/data/psi_columns.md`: the figure table (read-off values,
with the caveat that they are read off a bar chart at ±0.002), the block, and
the finding: which column and denominator the paper's figures report, stated
as "matched N of 8 bars within 0.002, max gap G". Then the consequences,
written for Raj:

1. If `unnorm_psi` matches: the paper already reports Eq. 1 as written;
   `second_normalization: false` costs nothing and removes a column the
   methods never mention. Bob's recommendation stands.
2. **If `norm_psi` matches (and `unnorm_psi` does not): Bob's proposed
   default of `second_normalization: false` is withdrawn** — the paper's
   headline figure reports the second-normalised column. The doc then
   states the real choice for Raj, symmetric to item 3: keep `norm_psi` as
   the reported PSI and add the second min-max to the methods (one
   sentence after Eq. 1: "the mean is then min-max scaled across
   settlements"), or switch the figures to Eq. 1 as written and let every
   bar move. Both columns stay in the config either way. *The plan-review
   round of 5 Sep 2026 already ran this comparison on the baseline files
   and found `norm_psi` × popdensity matching 8 of 8 bars (max gap 0.0006)
   against 1 of 8 for `unnorm_psi`; the script's job is to make that
   reproducible and drift-tested, not to discover it.*
3. If the popdensity denominator matches (the axis label says it will):
   **Bob's proposed default of dropping popdensity from the reported results
   is withdrawn** — the paper's headline figure is the popdensity variant.
   The doc then states the real choice for Raj: keep popdensity as the
   reported denominator and add its equation to the methods (Eq. 3 with
   Population_i/Area_i), or switch the figures to the per-population Eq. 3
   the manuscript prints. Both denominators stay in the config either way.
4. Whichever way it comes out, the decision log's § 7–8 and DEL-52 are
   updated by the final task with the measured answer. Items 2 and 3 are
   independent: the finding names the column AND the denominator, and the
   doc's consequences section has one paragraph for each.

If NO candidate matches at least 6 of 8 bars, the doc says so, prints the
four candidate tables in full, and the finding is "the figure was not
produced from these columns as-is" — an escalation to the owner, recorded,
not guessed around.

## 3. Shared contract for scripts and docs

- CLI shape, the three new scripts: `--config` (default `code-2025`),
  `--data-dir` (read-only), `--work-dir` (scratch; default a fresh temp
  dir; refused inside the data dir — the existing guard, now in
  `_measure_common`), plus the script-specific `--verify-dir` /
  `--baseline-dir` / `--all-candidates`. `measure_layer_pathologies.py`
  keeps its historic `--cache-dir` name and documented command line.
- **The settlement dedup cache and geometry types.** `_dedup_cached`
  returns the in-memory frame on a cold cache but re-reads its own
  GeoPackage on a warm one, and GeoPackage stores the layer as
  MultiPolygon, so a warm cache upcasts every Polygon (the raw layer has
  3,801 Polygon + 556 MultiPolygon; after a round trip all 4,357 read as
  MultiPolygon). Any count that inspects `geom_type` — today only
  `multipolygons` in the pathology script — is therefore only valid on a
  COLD cache. Rule: `measure_layer_pathologies.py` always runs cold (its
  default; the run step never points it at a staged cache), and
  `_measure_common.load_settlements` says so in its docstring. The other
  scripts' predicates (`intersects`, intersection length, `touch`
  adjacency, barrier flags) are type-agnostic and may share a warm cache.
- **One cache per machine, not one per test.** The three data-gated
  doc-drift tests for the new docs read the cache directory from the
  environment variable `DELHI_PSI_MEASURE_CACHE` (default: a fresh temp
  dir, i.e. cold, ~4.5 min of dedup each); the run step exports it to the
  staged work dir so the dedup and the O(n²) `touch` adjacency happen once.
  The pathology drift test keeps its own cold `fresh` fixture (previous
  bullet). The run step's full-suite invocation is therefore budgeted at
  15–20 minutes with the cache exported, and is run in the background with
  its output captured — never under a 10-minute foreground timeout.
- **`road_inside` ⇔ `road_length > 0`.** When `--verify-dir` is given the
  roads script asserts (raises, not warns) that the set of settlements it
  classifies `road_inside` equals the set with `road_length > 0` in the
  `code-2025` output CSV; a unit test exercises the assertion on a
  hand-built frame, and the real-data run exercises it on the layer.
- Output: provenance lines (`layer:`, `work-dir:`, and for DEL-49/52 the
  input directories) then one or more fenced ```` ```text ```` blocks.
  Multi-block scripts label them: the first line inside each fence is
  `block: <name>` and `parse_block(text, name=...)` selects by it (a small
  extension of the existing parser; the pathology script's single unlabeled
  block keeps working).
- The doc for each script carries its block(s) verbatim under a heading
  "Measured on <date> at commit <sha>", the exact command, and the prose
  finding. The prose may quote numbers from the block; a test checks every
  number quoted in prose that is wrapped in backticks appears as a value in
  the block (cheap guard against prose drifting from the block).

## 4. Tests

Fixture-level (fast, run in CI):

- `count_corner_only_pairs`: messy city = 1, Oraculum = 0; a synthetic
  three-polygon frame where A–B share an edge, B–C share a corner, A–C are
  disjoint → 1; an overlap pair → 0 (its intersection has area).
- DEL-49 (a): a synthetic layer built in the test — four squares in a row,
  a road segment through the first and a road that only touches the
  boundary of the third at a point → `road_inside` {1}, `road_via_neighbor`
  {2}, `no_road` {3, 4}; types assigned so the per-type keys are exercised.
- DEL-49 (b): the derived-profile writer is unit-tested (loads
  `code-2025.yaml`, changes one value, drops two keys, everything else
  byte-equal after a dump/load round trip), and `pipeline.methodology_stamp`
  is asserted not to contain `roads`. The compute itself is exercised on the
  ORACLE city: run (b) end-to-end on the Oraculum fixture through the same
  code path and assert the roads columns under own-only equal the
  reference implementation's `ideal` values from
  `tests/fixtures/oraculum/expected_values.csv` (rule `ideal`, scenario
  `baseline`, denominator `pop`: `road_pcen` A 0.0075, E 0.0025, others 0;
  `road_idx` A 1.0, E 1/3, others 0 — the memo § 3 quoted the PCEN values
  as if they were the index; the plan pins both columns) — the one-factor
  machinery proven on the city where the answer is known by hand.
- DEL-51: `inventory_barriers.inventory(layers)` on the Oraculum canal
  fixture returns count 1, the right length, the right flagged settlements
  (A and D). The `.shp.xml` reader is tested on a three-line XML string.
- DEL-52: `score_candidates` on a hand-built frame with known means matches
  the expected candidate; `FIGURE_4_BARS` has exactly the eight types in
  § 2.4; the `best_candidate` tie-break is tested.
- `_measure_common`: the work-dir guard refuses the data dir and any child
  of it; `parse_block` with and without `name=`; `render`/`parse_block`
  round-trip.
- Doc-drift tests, one per doc, in the `tests/test_layer_pathologies.py`
  pattern: the committed block parses and has the expected keys (always);
  and under that file's `needs_data` skip marker (skips when `~/delhi_data`
  is absent, so CI never touches real data) it equals the script's output
  on this machine.

Real-data (data-gated, run in this cycle's `run:` step, not in CI):

- `measure_layer_pathologies.py` full report regenerated; the existing keys
  must be unchanged (byte-identical to the committed block) and the two new
  keys added.
- `measure_roads_access.py --verify-dir ~/delhi_data/phase3_verify`.
- `inventory_barriers.py --all-candidates`.
- `measure_psi_columns.py --baseline-dir ~/delhi_data/psi_2020_results
  --verify-dir ~/delhi_data/phase3_verify`.

## 5. Docs to update in the same PR

- `docs/data/layer_pathologies.md` — new keys + definition + finding.
- New `docs/data/roads_access.md`, `barriers.md`, `psi_columns.md` (§ 2).
- `docs/decisions/2026-08-28-raj-methodology-decisions.md` — § 2 (roads:
  the measured JJC effect and whether the decision stands), § 3 (corner
  count), § 4 (provenance paragraph), §§ 7–8 (the DEL-52 finding; the
  popdensity default withdrawn or confirmed), and the batched-reply
  checklist rewritten with the numbers in.
- `WORKPLAN.md` — the four items ticked with one-line findings.
- `CHANGELOG.md` `[Unreleased]`.
- `README.md` — one line under the data docs pointing at the three new
  files, if the README lists `docs/data/` (check; add only if it does).

## 6. Tasks (for the plan)

1. `_measure_common.py` + tests; `measure_layer_pathologies.py` imports
   from it (no behaviour change; its block byte-identical).
2. DEL-50: `count_corner_only_pairs` + fixture tests + doc keys (real-data
   value filled in the run step).
3. DEL-51: `inventory_barriers.py` + tests + `docs/data/barriers.md`
   skeleton with the provenance prose (§ 2.3).
4. DEL-52: `measure_psi_columns.py` + tests + `docs/data/psi_columns.md`
   skeleton with the figure table.
5. DEL-49: `measure_roads_access.py` (both blocks) + tests incl. the
   Oraculum one-factor proof + `docs/data/roads_access.md` skeleton.
6. Run step: the four real-data commands (§ 4), paste the blocks, write the
   findings, update the decision log / WORKPLAN / CHANGELOG (§ 5).

Task 6 is the only one that touches the decision log; it is also where the
controller writes the numbers into prose, so it is dispatched with the blocks
in hand, not before.

## 7. Autonomy terms for this /ship run

Same as cycles 3A–3D. Fix forward, commit, push, PR, merge once CI and the
run step are green. A CONFIRMED Critical governs over the plan. **Stop and
ask** only for: anything that would change a shipped profile, a fixture
expected value, or `delhi_psi/` behaviour beyond an additive helper; any
write under `~/delhi_data` outside `--work-dir`; and the DEL-52 "no
candidate matches" outcome (§ 2.4, last paragraph). The batched email to Raj
is drafted AFTER the merge from the final docs and is never sent.

## 8. Decision log (choices made without the owner, with reasons)

1. **One cycle, four tickets, one PR.** They share the loader, the guard,
   the block format and the run step; four PRs would repeat the real-data
   setup four times. Cost if wrong: a larger review. Mitigation: six tasks,
   each independently testable.
2. **A shared `_measure_common.py`** rather than three copies of the guard
   and parser. It is `scripts/`-internal, imported by path-sibling scripts
   the way `tests/cities.py` is by tests. Not a `delhi_psi` module: these
   are measurement utilities, not pipeline API.
3. **DEL-49 runs `compute` in-process on a derived YAML in the work dir**,
   not a shipped profile and not a CLI subprocess. A shipped profile would
   need fixture registration (config doc § 3) for a one-off diagnostic;
   in-process gives the script the frames without re-reading CSVs. The
   YAML is still written to disk so the run is reproducible by hand with
   `delhi-psi compute --config <path>`.
4. **Road access is "positive intersection length"**, matching
   `road_lengths`, and neighbours for "via neighbour" use `touch`, Raj's
   rule, not today's bbox — the question is about the world after the
   decision. Both stated in the doc.
5. **DEL-52 compares against the July 2025 baseline files**, not the
   `phase3_verify` outputs, because the baseline is what produced the
   figures; `--verify-dir` is a cross-check only.
6. **Figure 4 bar values are read off the chart at ±0.002.** No figure data
   file is in the repo or the data folder (checked 5 Sep 2026); the axis
   label and footnote 12 make the read-off decisive enough for a
   column/denominator match. If the match is ambiguous the doc says so.
7. **DEL-51 makes no provenance claim**, only an inventory plus the
   evidence-based "suggests official source" sentence and the open question
   for Bijoy. The `manually marked` sentence in the paper is quoted so Raj
   sees what his own text says.
8. **The popdensity finding may reverse Bob's proposed default.** The spec
   commits to writing that reversal into the docs if the data says so
   (§ 2.4 item 2), because the owner's instruction was "best
   recommendations", and a recommendation contradicted by the paper's own
   axis label is not one.
9. **`FIGURE_4_BARS` lives in the script**, not a data file — eight numbers
   with a provenance comment, tested for shape.
10. **Plan review R1 (5 Sep 2026) rulings.** (a) The reviewers measured
    DEL-52 on the baseline: `norm_psi` × popdensity matches 8/8 bars —
    § 2.4 gained the `norm_psi` branch so the write-up cannot force-fit
    the `unnorm_psi` template. (b) A warm GeoPackage dedup cache upcasts
    Polygon → MultiPolygon: the pathology script runs cold, always (§ 3).
    (c) One shared cache via `DELHI_PSI_MEASURE_CACHE` for the new drift
    tests, and the run step's suite runs in the background (§ 3). (d) The
    `road_inside` ⇔ `road_length > 0` assertion is a stated requirement
    with a test (§ 3). (e) `compute` does not need the dedup caches; the
    roads script stages the artifact alone (§ 2.1 (b)). (f) The Oraculum
    roads anchors are `road_pcen` values; both columns pinned (§ 4).
