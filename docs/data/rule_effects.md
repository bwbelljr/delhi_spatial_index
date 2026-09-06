# What the cycle-3E rule changes do to today's numbers

Raj's 28 Aug 2026 decision log § 4 (`docs/decisions/2026-08-28-raj-methodology-decisions.md`)
ratifies partial-coverage weighting for barriers (`w_ij = 1 −
L_blocked(i,j) / L_shared(i,j)`, linear, symmetric) in place of the code's
global/asymmetric severing rule, and asks for the same one-factor
quantification against today's numbers "as for roads" (§ 4, § 5's "What
goes in the batched reply to Raj"). This document is that quantification
for the barrier rule, produced by `scripts/measure_rule_effects.py`, which
reads the layers named by the `code-2025` profile and writes nothing under
the data directory. A second section below covers the same quantification
for the overlap lending rule (DEL-20). `tests/test_measure_rule_effects.py`
re-runs the script and compares each block below (it skips when the data is
not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

- **Run date:** 2026-09-06
- **Inputs:** settlement layer `uso_update_sep2021`, the three `Barrier_Clip` layers, and the proven `code-2025` run in `~/delhi_data/phase3_verify` (its neighbours artifact, read for the `links_kept_code_2025` comparison only)
- **Commit:** `a4b51c6`
- **Command:** `uv run python scripts/measure_rule_effects.py --config code-2025 --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache`

## Block `partial_barriers` — the effect on today's numbers

`code-2025` with ONE thing changed, `methodology.barrier: {rule:
partial_weighted, combine: any, buffer_m: 5}` — everything else, `bbox`
adjacency included, left exactly as `code-2025` states it, so the diff is
attributable to the barrier rule alone.

Unlike the roads measurement, this block re-runs `preprocess`: the barrier
block is part of the methodology stamp, so the proven `code-2025` artifact
cannot be reused, and reusing it would be a silent lie about which
neighbourhood the numbers describe. The rebuild goes into `--work-dir`,
never beside the proven run.

**What the run must show, stated before it runs** (spec § 6.6): no NaN and no
negative anywhere; `links_kept_partial` > `links_kept_code_2025`, i.e. the
number of severed links FALLS (the global rule severs every link INTO a
flagged settlement, the partial rule severs only fully covered boundaries);
`links_severed` == 0 in the weight classes, because a w == 0 link is pruned
out of the artifact entirely — which is why that key proves nothing about
whether anything was fully covered, and why the finding below measures the
pre-barrier link count separately instead; and the fractional class is
non-empty. A result outside these bounds is a stop, not a number to write
down.

- `links_w_one` / `links_fractional` / `links_severed` — the directed
  neighbour links stored in the rebuilt artifact, by weight: unblocked
  (`w == 1`), partially blocked (`0 < w < 1`), and fully blocked (`w == 0`).
  `links_severed` is 0 by construction on a stored artifact — `apply_barrier`
  prunes a `w == 0` link out of both directions' lists before it is ever
  written, so a severed link leaves no trace in this count. It is reported
  anyway, as the assertion that nothing survived at weight 0.
- `median_fractional_w` — the median weight among the fractional links only
  (pre-formatted, like the roads block's means, so the drift comparison is
  exact); `nan` if there are none.
- `links_kept_code_2025` / `links_kept_partial` — how many directed links
  each rule KEEPS in its stored artifact: the total neighbour-list length
  under the proven `code-2025` run (read from `--verify-dir`) and under this
  rebuild. Neither of these is a count of links either rule "severs" — a
  severed link is, by construction, absent from both artifacts, so the only
  honest counts are what each rule leaves behind. The stated bound (spec
  § 6.6) is that the partial rule keeps MORE than the global rule does,
  because the global rule drops every link into a flagged settlement while
  the partial rule drops only fully covered shared boundaries.
- `settlements_list_changed` — how many settlements' stored neighbour list
  (the set of ids, not the weights) differs between the two rules.
- `preprocess_seconds` — the wall-clock cost of rebuilding the neighbours
  artifact under the partial rule, so the one-time cost of adopting it is on
  the record next to its effect.
- `psi_code_<denom>_<TYPE>` / `psi_partial_<denom>_<TYPE>` (and, where both
  runs carry `norm_psi`, `norm_code_<denom>_<TYPE>` /
  `norm_partial_<denom>_<TYPE>`) — the mean unnormalised (and normalised) PSI
  under each rule, per denominator and settlement type and in total, the
  same one-factor shape `measure_roads_access.py`'s `one_factor` block uses
  (DEL-49). `psi_code_*` is read from the proven `--verify-dir` output,
  never recomputed; `psi_partial_*` is this run's own output.
- `n_<denom>_<TYPE>` — the settlement count behind each mean, so a mean of
  zero settlements is never mistaken for a mean of zero PSI.

```text
block: partial_barriers
links_w_one: 27482
links_fractional: 1636
links_severed: 0
median_fractional_w: 0.938525
links_kept_code_2025: 21211
links_kept_partial: 29118
settlements_list_changed: 3155
preprocess_seconds: 864.935
n_pop_Planned: 964
n_pop_UAC: 1684
n_pop_RUAC: 393
n_pop_JJC: 749
n_pop_JJR: 48
n_pop_UV: 138
n_pop_SDA: 86
n_pop_total: 4131
psi_code_pop_Planned: 0.00901129
psi_code_pop_UAC: 0.0106064
psi_code_pop_RUAC: 0.00440786
psi_code_pop_JJC: 0.0176336
psi_code_pop_JJR: 0.00269464
psi_code_pop_UV: 0.0103199
psi_code_pop_SDA: 0.0190069
psi_code_pop_total: 0.0109748
psi_partial_pop_Planned: 0.00850924
psi_partial_pop_UAC: 0.0141033
psi_partial_pop_RUAC: 0.00707284
psi_partial_pop_JJC: 0.0251647
psi_partial_pop_JJR: 0.00258057
psi_partial_pop_UV: 0.00840339
psi_partial_pop_SDA: 0.0105075
psi_partial_pop_total: 0.0136806
norm_code_pop_Planned: 0.0109918
norm_code_pop_UAC: 0.0129375
norm_code_pop_RUAC: 0.0053766
norm_code_pop_JJC: 0.021509
norm_code_pop_JJR: 0.00328685
norm_code_pop_UV: 0.0125879
norm_code_pop_SDA: 0.0231841
norm_code_pop_total: 0.0133868
norm_partial_pop_Planned: 0.0107792
norm_partial_pop_UAC: 0.0178656
norm_partial_pop_RUAC: 0.00895963
norm_partial_pop_JJC: 0.0318778
norm_partial_pop_JJR: 0.00326898
norm_partial_pop_UV: 0.0106451
norm_partial_pop_SDA: 0.0133106
norm_partial_pop_total: 0.0173301
n_popdensity_Planned: 964
n_popdensity_UAC: 1684
n_popdensity_RUAC: 393
n_popdensity_JJC: 749
n_popdensity_JJR: 48
n_popdensity_UV: 138
n_popdensity_SDA: 86
n_popdensity_total: 4131
psi_code_popdensity_Planned: 0.0304103
psi_code_popdensity_UAC: 0.0118991
psi_code_popdensity_RUAC: 0.0148165
psi_code_popdensity_JJC: 0.000914278
psi_code_popdensity_JJR: 0.0257077
psi_code_popdensity_UV: 0.0261655
psi_code_popdensity_SDA: 0.0191333
psi_code_popdensity_total: 0.015886
psi_partial_popdensity_Planned: 0.0371327
psi_partial_popdensity_UAC: 0.0223262
psi_partial_popdensity_RUAC: 0.0266961
psi_partial_popdensity_JJC: 0.00306383
psi_partial_popdensity_JJR: 0.0366119
psi_partial_popdensity_UV: 0.0422909
psi_partial_popdensity_SDA: 0.0180126
psi_partial_popdensity_total: 0.0240971
norm_code_popdensity_Planned: 0.0443043
norm_code_popdensity_UAC: 0.0173356
norm_code_popdensity_RUAC: 0.021586
norm_code_popdensity_JJC: 0.001332
norm_code_popdensity_JJR: 0.0374532
norm_code_popdensity_UV: 0.0381201
norm_code_popdensity_SDA: 0.027875
norm_code_popdensity_total: 0.023144
norm_partial_popdensity_Planned: 0.0482092
norm_partial_popdensity_UAC: 0.028986
norm_partial_popdensity_RUAC: 0.0346595
norm_partial_popdensity_JJC: 0.00397775
norm_partial_popdensity_JJR: 0.047533
norm_partial_popdensity_UV: 0.0549061
norm_partial_popdensity_SDA: 0.0233856
norm_partial_popdensity_total: 0.0312851
```

## Finding

**Barriers in Delhi mostly clip shared boundaries rather than closing
them.** Of the `29118` directed links the partial rule keeps, `27482` are
untouched at weight 1 and `1636` are partially blocked, and the median
partly-blocked link still keeps `0.938525` of its contribution.

*A correction the final review forced, worth stating plainly:* `links_severed`
is `0`, but that number **cannot** tell you whether anything was fully
covered — a w == 0 link is pruned before the artifact is written, so it is
zero by construction whatever the layer looks like. The honest test is to
count the directed links that exist BEFORE any barrier rule runs and compare.
Measured separately on the same layer and cache (bbox adjacency, no barrier):
**29,258 pre-barrier directed links.** So

| rule | keeps | severs | share severed |
|---|---|---|---|
| `global_asymmetric` (today) | 21,211 | 8,047 | 27.5 % |
| `partial_weighted` | 29,118 | 140 | 0.5 % |

140 links ARE fully covered — the earlier claim that none were was wrong, and
rested on a metric that could not show it. The real finding is the ratio:
the partial rule severs **57× fewer** links than the rule in the July 2025
numbers.

**Most of the effect is not the partial weighting at all — it is retiring the
global flag.** `global_asymmetric` deletes every link INTO a barrier-flagged
settlement, on every side, whether or not a barrier lies between that pair;
that accounts for 8,047 of the 8,187-link difference, against 1,636 links
merely discounted. `3155` of 4,357 settlements (72 %) get a different
neighbour list.

**What it does to the index.** Under the population-density denominator —
the one the paper's figures use (`psi_columns.md`) — every reported type
gains, because settlements that were cut off from their neighbours get them
back:

| type | `norm_psi` today | under `partial_weighted` | change |
|---|---|---|---|
| Planned | `0.0443043` | `0.0482092` | +8.8 % |
| JJC | `0.001332` | `0.00397775` | +199 % |
| UAC | `0.0173356` | `0.028986` | +67 % |
| RUAC | `0.021586` | `0.0346595` | +61 % |
| JJR | `0.0374532` | `0.047533` | +26.9 % |
| UV | `0.0381201` | `0.0549061` | +44 % |
| SDA | `0.027875` | `0.0233856` | −16.1 % |

**Two consequences for the paper's headline, and the second one is a
reordering.**

*The gap narrows sharply.* JJC stays the lowest-scoring type before and
after, and its mean roughly triples while Planned's rises by under a tenth,
so Planned-over-JJC falls from about 33× to about 12×. Top-to-bottom across
all reported types it falls from 33× to about 14×.

*Planned is no longer the top-scoring type.* Under `partial_weighted`, urban
villages overtake planned colonies — `0.0549061` against `0.0482092`, where
today it is Planned `0.0443043` over UV `0.0381201`. The same reordering
holds on the unnormalised column. This is not a change of degree but of
rank, in the density variant the figures are drawn from, and it is the one
result here that could change a sentence in the paper rather than a number.

Both are Raj's to weigh: they are what adopting his own partial-barrier
decision costs, and neither is a reason on its own to abandon it.

**Cost.** Rebuilding the neighbours artifact under this rule took
`864.935` seconds (14.4 minutes) on 4,357 settlements — a one-off per
profile, since the barrier block is in the methodology stamp and `compute`
refuses a mismatched artifact.

## The same classification, hand-counted

Proven on Oraculum (spec § 6.1), where every pair can be counted by hand:
10 undirected `bbox` pairs make 20 directed links, of which only the A–D
edge is fractional in both directions (w = 0.08, the 5 m round-cap buffer
against a canal covering 90% of the shared edge) and none is severed — 18
at w == 1, 2 fractional, 0 severed. (These are the Oraculum hand counts, not
the real-layer block above; they are deliberately not in `backticks` so the
prose-number guard never mistakes one for the other.)

## Block `overlap_lending` — the effect on today's numbers

`code-2025` with ONE thing changed, `methodology.overlap: {lending:
outside_receiver}` — everything else, `bbox` adjacency and
`global_asymmetric` barriers included, left exactly as `code-2025` states
it, so the diff is attributable to the lending rule alone.

Unlike the barrier block, this one REUSES the proven `code-2025` neighbours
artifact: `overlap.lending` is applied downstream in `compute` and is not in
the methodology stamp, so the stored lists are still the right ones. The
script asserts that before staging, so a future change that puts `overlap`
in the stamp fails loudly instead of quietly describing a neighbourhood
nobody built.

- **Run date:** 2026-09-06
- **Inputs:** settlement layer `uso_update_sep2021`, the six point-service layers and the road layer, and the proven `code-2025` run in `~/delhi_data/phase3_verify` (its neighbours artifact, staged unchanged, and its output CSVs, read for the one-factor comparison)
- **Commit:** `60d95f5`
- **Command:** `uv run python scripts/measure_rule_effects.py --config code-2025 --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache --only overlap_lending`

**What the run must show, stated before it runs** (spec § 6.6): no NaN and
no negative anywhere; `settlements_pcen_rose_pop` and
`settlements_pcen_rose_popdensity` both **0**, because lending is only ever
reduced — a risen PCEN is a bug, not a finding;
`pcen_changed_outside_the_overlap_set_*` both **0**, because the rule cannot
move a settlement with no overlapping neighbour;
`shared_pairs_total` greater than 0 and no larger than twice the 429
multi-settlement points `docs/data/layer_pathologies.md` counts (each such
point contributes at most k(k-1) ordered entries, and k is 2 for essentially
all of them), summed over the six point services; and `shared_pairs_road`
consistent with a road network that crosses the 4,069 overlapping pairs. A
result outside these bounds is a stop, not a number to write down.

- `shared_pairs_<service>` — the number of ORDERED `(i, j)` entries
  `index.shared_amounts` builds for that service: how many directed pairs
  share at least one unit of it. `shared_pairs_total` sums these across
  every point and line service.
- `settlements_with_an_overlapping_neighbour` — settlements with at least
  one STORED neighbour whose polygon overlaps theirs (a positive-area
  intersection). This is the containment bound: the lending rule cannot move
  any other settlement's PCEN.
- `settlements_pcen_changed_<denom>` / `settlements_pcen_rose_<denom>` — how
  many settlements have at least one `*_pcen` column that moved between the
  two runs, and how many of those moved UP. `outside_receiver` only ever
  subtracts, so a risen PCEN would be a bug.
- `pcen_changed_outside_the_overlap_set_<denom>` — settlements whose PCEN
  changed despite having no overlapping neighbour: must be 0, or the rule
  reached further than the shared structure it is built from.
- `psi_code_<denom>_<TYPE>` / `psi_outside_<denom>_<TYPE>` (and, where both
  runs carry `norm_psi`, `norm_code_<denom>_<TYPE>` /
  `norm_outside_<denom>_<TYPE>`) — the mean unnormalised (and normalised) PSI
  under each rule, per denominator and settlement type and in total, the
  same one-factor shape `measure_roads_access.py`'s `one_factor` block and
  the barrier block above use. `psi_code_*` is read from the proven
  `--verify-dir` output, never recomputed; `psi_outside_*` is this run's own
  output.
- `n_<denom>_<TYPE>` — the settlement count behind each mean, so a mean of
  zero settlements is never mistaken for a mean of zero PSI.

```text
block: overlap_lending
shared_pairs_bank: 194
shared_pairs_health: 48
shared_pairs_police: 4
shared_pairs_ration: 156
shared_pairs_school: 44
shared_pairs_transport: 64
shared_pairs_road: 248
shared_pairs_total: 758
settlements_with_an_overlapping_neighbour: 2427
settlements_pcen_changed_pop: 385
settlements_pcen_rose_pop: 0
pcen_changed_outside_the_overlap_set_pop: 0
n_pop_Planned: 964
n_pop_UAC: 1684
n_pop_RUAC: 393
n_pop_JJC: 749
n_pop_JJR: 48
n_pop_UV: 138
n_pop_SDA: 86
n_pop_total: 4131
psi_code_pop_Planned: 0.00901129
psi_code_pop_UAC: 0.0106064
psi_code_pop_RUAC: 0.00440786
psi_code_pop_JJC: 0.0176336
psi_code_pop_JJR: 0.00269464
psi_code_pop_UV: 0.0103199
psi_code_pop_SDA: 0.0190069
psi_code_pop_total: 0.0109748
psi_outside_pop_Planned: 0.00901039
psi_outside_pop_UAC: 0.010473
psi_outside_pop_RUAC: 0.00440775
psi_outside_pop_JJC: 0.0176297
psi_outside_pop_JJR: 0.0026894
psi_outside_pop_UV: 0.0103197
psi_outside_pop_SDA: 0.0190069
psi_outside_pop_total: 0.0109195
norm_code_pop_Planned: 0.0109918
norm_code_pop_UAC: 0.0129375
norm_code_pop_RUAC: 0.0053766
norm_code_pop_JJC: 0.021509
norm_code_pop_JJR: 0.00328685
norm_code_pop_UV: 0.0125879
norm_code_pop_SDA: 0.0231841
norm_code_pop_total: 0.0133868
norm_outside_pop_Planned: 0.0109907
norm_outside_pop_UAC: 0.0127747
norm_outside_pop_RUAC: 0.00537646
norm_outside_pop_JJC: 0.0215042
norm_outside_pop_JJR: 0.00328046
norm_outside_pop_UV: 0.0125877
norm_outside_pop_SDA: 0.0231841
norm_outside_pop_total: 0.0133193
settlements_pcen_changed_popdensity: 382
settlements_pcen_rose_popdensity: 0
pcen_changed_outside_the_overlap_set_popdensity: 0
n_popdensity_Planned: 964
n_popdensity_UAC: 1684
n_popdensity_RUAC: 393
n_popdensity_JJC: 749
n_popdensity_JJR: 48
n_popdensity_UV: 138
n_popdensity_SDA: 86
n_popdensity_total: 4131
psi_code_popdensity_Planned: 0.0304103
psi_code_popdensity_UAC: 0.0118991
psi_code_popdensity_RUAC: 0.0148165
psi_code_popdensity_JJC: 0.000914278
psi_code_popdensity_JJR: 0.0257077
psi_code_popdensity_UV: 0.0261655
psi_code_popdensity_SDA: 0.0191333
psi_code_popdensity_total: 0.015886
psi_outside_popdensity_Planned: 0.0304087
psi_outside_popdensity_UAC: 0.0113169
psi_outside_popdensity_RUAC: 0.0148076
psi_outside_popdensity_JJC: 0.000911273
psi_outside_popdensity_JJR: 0.0256916
psi_outside_popdensity_UV: 0.0261655
psi_outside_popdensity_SDA: 0.0191333
psi_outside_popdensity_total: 0.0156467
norm_code_popdensity_Planned: 0.0443043
norm_code_popdensity_UAC: 0.0173356
norm_code_popdensity_RUAC: 0.021586
norm_code_popdensity_JJC: 0.001332
norm_code_popdensity_JJR: 0.0374532
norm_code_popdensity_UV: 0.0381201
norm_code_popdensity_SDA: 0.027875
norm_code_popdensity_total: 0.023144
norm_outside_popdensity_Planned: 0.0443019
norm_outside_popdensity_UAC: 0.0164874
norm_outside_popdensity_RUAC: 0.0215729
norm_outside_popdensity_JJC: 0.00132762
norm_outside_popdensity_JJR: 0.0374296
norm_outside_popdensity_UV: 0.0381201
norm_outside_popdensity_SDA: 0.027875
norm_outside_popdensity_total: 0.0227954
```

## Finding — overlap lending

**Every bound stated before the run held.** `settlements_pcen_rose_pop` and
`settlements_pcen_rose_popdensity` are both `0`: the rule only ever
subtracts, and nothing rose. `pcen_changed_outside_the_overlap_set_pop` and
its density twin are both `0`: nothing moved that had no overlapping
neighbour, so the rule reaches exactly as far as the shared structure it is
built from and no further.

**The double count is real but small.** `758` ordered pairs share at least
one unit of a service — `194` for banks, `156` for ration shops, `248` for
roads, the rest smaller — and `2427` settlements have at least one
overlapping neighbour. Of those, `385` have a PCEN that actually moves
under the population denominator and `382` under population density — that
is `385` of the `4131` reported settlements, roughly one in eleven, and
about one in six of the `2427` that have an overlapping neighbour at all.
Every one of them was previously counting some service twice: once as its
own, once decayed as a neighbour's.

**It does not move the headline.** Under the density denominator the paper's
figures use, the means barely shift: Planned `0.0304103` → `0.0304087`,
JJC `0.000914278` → `0.000911273`, and the largest mover is UAC at
`0.0118991` → `0.0113169`, about −4.9 %. Every type falls or holds; none
rises; no ordering changes. This is a correction that removes a real
arithmetic error without disturbing what the paper says — the opposite
profile from the barrier rule above, which is small in mechanism and large
in effect.

**Why it is `whole` in both shipped profiles anyway.** Raj ratified that a
service in an overlap counts for every colony containing it; he has not been
asked about the lending half, which is Bob's proposal. The measurement is
what the question should be asked with: it costs about 5 % of one settlement
type's mean and nothing of the argument.

## A caveat this run earned

The first attempt at this measurement **failed**, and the failure was a real
defect rather than a flaky run. A road's own clipped length and its shared
part are two independent GEOS clips of the same line, so the shared part can
exceed the whole by an ulp; one pair out of the 4,069 overlapping ones did,
by `1.1102230246251565e-16` km — a tenth of a picometre — and the
over-subtraction guard aborted the whole run. The guard was right to exist
and too strict by one bit: it now clamps below a tolerance and still raises
above it, because a genuine mismatch between the amounts frame and the
shared structure is off by a length, not by an ulp. Point services are
integer counts and were never exposed to this.

The final review then caught the test written to pin that fix: adding
1.1e-16 to 2.0 rounds straight back to 2.0 in double precision, so the
"one ulp" case was never negative and the clamp never ran — reviewers
proved it by reverting the fix and watching the test still pass. It now
uses `math.nextafter` for a genuine one-ulp excess, and two further tests
sit either side of the tolerance itself so the constant cannot be changed
unnoticed. The tolerance was also tightened from an arbitrary `1e-9` to a
derived multiple of machine epsilon, and the clamp now logs when it fires
rather than absorbing silently.
