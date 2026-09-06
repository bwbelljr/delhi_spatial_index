# Phase 6 sweep — cost and rank-based comparison of the methodology points

Raj's 28 Aug 2026 decision log ratifies partial-coverage barrier weighting,
`outside_receiver` overlap lending's counting half, and a distance-decay
steer toward forms that spread the neighbour weight up from zero (§ 4, § 9);
DEL-36/37/39 ask for the same one-factor treatment `docs/data/rule_effects.md`
gave the barrier and overlap rules, but for adjacency and decay, across the
eleven-point sweep `docs/superpowers/specs/2026-09-06-phase6-sweep-dry-run-design.md`
lays out. This document is that quantification, produced by
`scripts/summarize_sweep.py`, which reads each sweep point's manifest and
output CSV under `--work-dir` plus the proven `code-2025` run under
`--baseline-dir`, and writes nothing under either directory.
`tests/test_summarize_sweep.py` re-runs the script against the real (still
in-flight) sweep and pins several of the numbers below directly.

- **Run date:** 2026-09-06
- **Inputs:** `~/psi_sweep` (9 of the 11 real sweep-point profiles carry a
  manifest and output CSV as of this writing — DEL-55 spec § 7's
  cheapest-first real run is still in flight) and the proven `code-2025`
  run in `~/delhi_data/phase3_verify` (the bbox baseline, both
  denominators, read read-only; its structure is read off
  `colonies_neighbors.joblib`, never rebuilt)
- **Commit:** `c402904`
- **Command:** `uv run python -m scripts.summarize_sweep --work-dir ~/psi_sweep --baseline-dir ~/delhi_data/phase3_verify`

**DRY RUN on `code-2025` — superseded by the ratified profile.** The frozen
July 2025 rule set still carries `bbox` adjacency, `global_asymmetric`
barriers and decayed roads, all three of which Raj's 28 Aug 2026 decisions
change. No number in this document is quotable, and none of it belongs in
the manuscript. Phase 6's reported variants (DEL-36/37/39) are the same
sweep re-run against the ratified profile (DEL-31).

**What the run must show, stated before it runs** (spec § 6.2, § 6.6, § 8):
`own_share` never exceeds `1 + 1e-9` on any row — a `NaN` from a zero pooled
`pcen` is expected, counted as `n_own_share_undef`, and is *not* a stop; the
`isolates` flag fires only when a point's isolate count exceeds the bbox
baseline's own count (`360`), never on an absolute count, since the
baseline is itself not isolate-free; every tie-gated decile cell (a tie
block larger than 1.5× its decile) renders the em dash `—`, never a number
or a blank; and every cross-point comparison is rank-based or a
within-run standardised effect size (spec § 6.1) — no level, difference, or
percentage change in PSI ever appears between two points, and grep over
this file confirms none does. A result outside these bounds is a stop, not
a number to write down; none was hit producing the numbers below, and one
genuine surprise was — the `isolates` flag was originally written as
`n_isolates > 0`, which the real bbox baseline's own 360 isolates would
have fired on every single row, anchors included; § 6.6 was corrected
before this run, and `tests/test_summarize_sweep.py` pins the corrected,
baseline-relative rule.

**PSI levels are not comparable across sweep points** (spec § 6.1): Eq. 2's
min-max runs once per point, and widening the neighbourhood inflates every
settlement's index at once. So nothing below is a level, a level
difference, or a level ratio between two points — every cross-point number
here is either a rank correlation (Spearman ρ, Kendall τ / τ-b), a decile
*set* overlap (Jaccard), or a within-run standardised effect size (Cliff's
δ, Cohen's d). Levels appear only inside one point's own row, as
diagnostics.

**The bootstrap in block `ordering` treats a complete census as a sample.**
It is therefore a *composition-sensitivity* device, not a significance
test: "would this category ordering survive a different draw of
settlements from the same city," never "is this ordering statistically
significant." 1,000 resamples, stratified by category, seed 0, at every
point (spec § 6.4).

**The formal/informal pooling in block `gap`** is `{Planned, SDA}` = formal
vs `{JJC, JJR}` = informal — a *stated assumption of this document*, not a
ruling from the paper. `UV`, `UAC`, `RUAC`, `Industrial` and `Other` are
deliberately left out of the pooling because their placement is exactly
what DEL-53 and the UV/SDA question are still open on; the per-category
`points`/`ordering` blocks above carry all nine reported categories
regardless (`RV` is excluded upstream by every profile's own
`methodology.exclusion`, so it never reaches the output CSV at all).

## Block `points` — one row per sweep point

**DRY RUN on `code-2025` — superseded by the ratified profile.** The frozen
July 2025 rule set still carries `bbox` adjacency, `global_asymmetric`
barriers and decayed roads, all three of which Raj's 28 Aug 2026 decisions
change. No number in this document is quotable, and none of it belongs in
the manuscript. Phase 6's reported variants (DEL-36/37/39) are the same
sweep re-run against the ratified profile (DEL-31).

Identity, structure/cost, composition (`own_share_p50` / `n_own_share_undef`,
spec § 6.2) and the rank-based outcome columns (spec § 6.3), one fenced
block per point, all labelled `points` — `parse_block(doc, name="points")`
returns the first (the bbox baseline); the drift/smoke tests in
`tests/test_summarize_sweep.py` read every row via `scripts._measure_common._blocks`.
A gated `jaccard_top10`/`jaccard_bottom10` cell (resolution #2) renders `—`.

```text
block: points
point: baseline
profile: code-2025
adjacency: bbox
radius_km: —
decay_form: inverse_linear
decay_param: —
decay_distance: centroid
n_reported: 4131
n_isolates: 360
n_links: 21211
deg_mean: 4.9
deg_p50: 4
deg_max: 96
preprocess_s: —
compute_s: —
own_share_p50: 0.058
n_own_share_undef: 452
cat_order: Other>Industrial>JJR>SDA>UV>Planned>RUAC>UAC>JJC
tau_vs_own: 0.72
rho_vs_own: 0.660
taub_vs_own: 0.511
tau_vs_bbox: 1.00
rho_vs_bbox: 1.000
taub_vs_bbox: 1.000
jaccard_top10: 1.000
jaccard_bottom10: 1.000
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: own-only
profile: own-only
adjacency: —
radius_km: —
decay_form: —
decay_param: —
decay_distance: —
n_reported: 4131
n_isolates: —
n_links: —
deg_mean: —
deg_p50: —
deg_max: —
preprocess_s: —
compute_s: —
own_share_p50: 1.000
n_own_share_undef: 1834
cat_order: JJR>Industrial>Other>UV>Planned>SDA>RUAC>UAC>JJC
tau_vs_own: 1.00
rho_vs_own: 1.000
taub_vs_own: 1.000
tau_vs_bbox: 0.72
rho_vs_bbox: 0.660
taub_vs_bbox: 0.511
jaccard_top10: 0.356
jaccard_bottom10: —
planned_gt_jjc: True
flag: pinned
```
```text
block: points
point: adj-touch
profile: adj-touch
adjacency: touch
radius_km: —
decay_form: inverse_linear
decay_param: —
decay_distance: centroid
n_reported: 4131
n_isolates: 715
n_links: 14641
deg_mean: 3.4
deg_p50: 2
deg_max: 78
preprocess_s: 91.635
compute_s: 50.393
own_share_p50: 0.127
n_own_share_undef: 788
cat_order: JJR>Other>Industrial>UV>SDA>Planned>RUAC>UAC>JJC
tau_vs_own: 0.89
rho_vs_own: 0.725
taub_vs_own: 0.579
tau_vs_bbox: 0.83
rho_vs_bbox: 0.904
taub_vs_bbox: 0.793
jaccard_top10: 0.669
jaccard_bottom10: —
planned_gt_jjc: True
flag: isolates,pinned
```
```text
block: points
point: band-0km
profile: band-0km
adjacency: within_distance
radius_km: 0.0
decay_form: inverse_linear
decay_param: —
decay_distance: centroid
n_reported: 4131
n_isolates: 697
n_links: 15462
deg_mean: 3.5
deg_p50: 3
deg_max: 81
preprocess_s: 11.691
compute_s: 52.702
own_share_p50: 0.122
n_own_share_undef: 786
cat_order: JJR>Other>Industrial>UV>SDA>Planned>RUAC>UAC>JJC
tau_vs_own: 0.89
rho_vs_own: 0.727
taub_vs_own: 0.579
tau_vs_bbox: 0.83
rho_vs_bbox: 0.907
taub_vs_bbox: 0.802
jaccard_top10: 0.693
jaccard_bottom10: —
planned_gt_jjc: True
flag: isolates,pinned
```
```text
block: points
point: band-1km
profile: band-1km
adjacency: within_distance
radius_km: 1.0
decay_form: inverse_linear
decay_param: —
decay_distance: centroid
n_reported: 4131
n_isolates: 15
n_links: 165119
deg_mean: 37.9
deg_p50: 33
deg_max: 252
preprocess_s: 75.514
compute_s: 569.579
own_share_p50: 0.010
n_own_share_undef: 36
cat_order: SDA>Other>Industrial>Planned>UV>JJR>RUAC>UAC>JJC
tau_vs_own: 0.39
rho_vs_own: 0.585
taub_vs_own: 0.440
tau_vs_bbox: 0.67
rho_vs_bbox: 0.850
taub_vs_bbox: 0.677
jaccard_top10: 0.527
jaccard_bottom10: 0.283
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-none
profile: decay-none
adjacency: bbox
radius_km: —
decay_form: none
decay_param: —
decay_distance: centroid
n_reported: 4131
n_isolates: 360
n_links: 21211
deg_mean: 4.9
deg_p50: 4
deg_max: 96
preprocess_s: 694.582
compute_s: 68.766
own_share_p50: 0.034
n_own_share_undef: 452
cat_order: Other>Industrial>JJR>UV>SDA>Planned>RUAC>UAC>JJC
tau_vs_own: 0.78
rho_vs_own: 0.624
taub_vs_own: 0.478
tau_vs_bbox: 0.94
rho_vs_bbox: 0.995
taub_vs_bbox: 0.945
jaccard_top10: 0.743
jaccard_bottom10: 1.000
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-power05
profile: decay-power05
adjacency: bbox
radius_km: —
decay_form: inverse_power
decay_param: 0.50
decay_distance: centroid
n_reported: 4131
n_isolates: 360
n_links: 21211
deg_mean: 4.9
deg_p50: 4
deg_max: 96
preprocess_s: —
compute_s: 69.156
own_share_p50: 0.044
n_own_share_undef: 452
cat_order: Other>Industrial>JJR>UV>SDA>Planned>RUAC>UAC>JJC
tau_vs_own: 0.78
rho_vs_own: 0.641
taub_vs_own: 0.494
tau_vs_bbox: 0.94
rho_vs_bbox: 0.998
taub_vs_bbox: 0.970
jaccard_top10: 0.860
jaccard_bottom10: 1.000
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-power2
profile: decay-power2
adjacency: bbox
radius_km: —
decay_form: inverse_power
decay_param: 2.00
decay_distance: centroid
n_reported: 4131
n_isolates: 360
n_links: 21211
deg_mean: 4.9
deg_p50: 4
deg_max: 96
preprocess_s: —
compute_s: 70.468
own_share_p50: 0.090
n_own_share_undef: 452
cat_order: JJR>Other>Industrial>UV>SDA>Planned>RUAC>UAC>JJC
tau_vs_own: 0.89
rho_vs_own: 0.701
taub_vs_own: 0.553
tau_vs_bbox: 0.83
rho_vs_bbox: 0.992
taub_vs_bbox: 0.928
jaccard_top10: 0.750
jaccard_bottom10: 1.000
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-exp2km
profile: decay-exp2km
adjacency: bbox
radius_km: —
decay_form: exponential
decay_param: 2.0
decay_distance: centroid
n_reported: 4131
n_isolates: 360
n_links: 21211
deg_mean: 4.9
deg_p50: 4
deg_max: 96
preprocess_s: —
compute_s: 71.758
own_share_p50: 0.051
n_own_share_undef: 452
cat_order: Other>Industrial>JJR>SDA>UV>Planned>RUAC>UAC>JJC
tau_vs_own: 0.72
rho_vs_own: 0.650
taub_vs_own: 0.501
tau_vs_bbox: 1.00
rho_vs_bbox: 1.000
taub_vs_bbox: 0.985
jaccard_top10: 0.934
jaccard_bottom10: 1.000
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-exp5km
profile: decay-exp5km
adjacency: bbox
radius_km: —
decay_form: exponential
decay_param: 5.0
decay_distance: centroid
n_reported: 4131
n_isolates: 360
n_links: 21211
deg_mean: 4.9
deg_p50: 4
deg_max: 96
preprocess_s: —
compute_s: 70.104
own_share_p50: 0.039
n_own_share_undef: 452
cat_order: Other>Industrial>JJR>UV>SDA>Planned>RUAC>UAC>JJC
tau_vs_own: 0.78
rho_vs_own: 0.634
taub_vs_own: 0.487
tau_vs_bbox: 0.94
rho_vs_bbox: 0.997
taub_vs_bbox: 0.960
jaccard_top10: 0.831
jaccard_bottom10: 1.000
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-boundary
profile: decay-boundary
adjacency: bbox
radius_km: —
decay_form: inverse_linear
decay_param: —
decay_distance: boundary
n_reported: 4131
n_isolates: 360
n_links: 21211
deg_mean: 4.9
deg_p50: 4
deg_max: 96
preprocess_s: —
compute_s: 70.313
own_share_p50: 0.036
n_own_share_undef: 452
cat_order: Other>Industrial>JJR>UV>SDA>Planned>RUAC>UAC>JJC
tau_vs_own: 0.78
rho_vs_own: 0.631
taub_vs_own: 0.484
tau_vs_bbox: 0.94
rho_vs_bbox: 0.996
taub_vs_bbox: 0.951
jaccard_top10: 0.769
jaccard_bottom10: 1.000
planned_gt_jjc: True
flag: smoothed,pinned
```

### Finding

**The bbox baseline itself is already heavily smoothed and pinned.** Its
own `own_share_p50` is `0.058` — under the `smoothed` threshold of 0.10 —
and it carries the `pinned` flag alongside: exactly one settlement sits at
`norm_psi == 1` while the 99th percentile is well below it, i.e. one
outlier is compressing the rest of the distribution even before any factor
is swept. Both flags fire at nearly every point measured so far, which is
itself the finding this dry run set out to surface — not a defect in the
flag logic (`tests/test_summarize_sweep.py::test_isolates_flag_is_relative_to_the_baseline_not_absolute`
and its neighbours pin the corrected, baseline-relative `isolates` rule
separately).

**Widening the band away from the baseline erodes the ranking fast.**
`band-1km` — the first band point measured — drops `tau_vs_bbox` to `0.67`
and `taub_vs_bbox` to `0.677`, against `adj-touch`'s `0.83`/`0.793` and
`band-0km`'s identical `0.83`/`0.802`. `band-1km` also has the sweep's
lowest isolate count so far, `15`, against the baseline's `360` — a
36 km²-per-settlement neighbourhood leaves almost nobody stranded, at the
cost of an `own_share_p50` of `0.010`: at 1 km the index is overwhelmingly
a property of the neighbourhood, not the settlement.

**The own-only anchor's bottom decile is unusable, exactly as spec § 6.3
predicted before this run.** 1,834 of 4,131 reported settlements own zero
of all seven services (`n_own_share_undef: 1834`), which swamps the
413-row decile the moment the neighbour term is removed —
`jaccard_bottom10` renders `—` against the bbox baseline, and the same
gate fires on `adj-touch` and `band-0km`'s bottom deciles too (both narrow
enough to reproduce a large zero-PSI tie block). The top decile survives
everywhere measured (`jaccard_top10` never gates), because the sweep's
real numbers put exactly one settlement at `norm_psi == 1`.

**The six decay forms barely move the ranking against the bbox baseline**
(all share its neighbourhood — spec § 4.2): `tau_vs_bbox` ranges only
`0.83`–`1.00` and `taub_vs_bbox` only `0.928`–`0.985` across all six, far
tighter than the band points' spread. `decay-power2` (the steep contrast)
is the outlier of the six, at `tau_vs_bbox: 0.83`; `decay-exp2km` (fast
decay) is the closest to the baseline, at `tau_vs_bbox: 1.00`.

## Block `ordering` — bootstrap 95% rank interval per category

**DRY RUN on `code-2025` — superseded by the ratified profile.** The frozen
July 2025 rule set still carries `bbox` adjacency, `global_asymmetric`
barriers and decayed roads, all three of which Raj's 28 Aug 2026 decisions
change. No number in this document is quotable, and none of it belongs in
the manuscript. Phase 6's reported variants (DEL-36/37/39) are the same
sweep re-run against the ratified profile (DEL-31).

One row per point, one cell per reported category (`RV` is excluded
upstream and never appears): `rank [2.5th-97.5th percentile rank]` over
1,000 settlement-level resamples stratified by category, seed 0 (spec
§ 6.4). This is a composition-sensitivity statement, not a significance
test — see the introduction above. `n_fragile_pairs` counts the adjacent
pairs in the point estimate whose bootstrap flip probability exceeds 0.05.

```text
block: ordering
point: baseline
Other: 1 [1-6]
Industrial: 2 [1-6]
JJR: 3 [1-6]
SDA: 4 [2-6]
UV: 5 [2-6]
Planned: 6 [3-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```
```text
block: ordering
point: own-only
JJR: 1 [1-2]
Industrial: 2 [1-3]
Other: 3 [2-5]
UV: 4 [3-4]
Planned: 5 [4-6]
SDA: 6 [5-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 4
seed: 0
n: 1000
```
```text
block: ordering
point: adj-touch
JJR: 1 [1-5]
Other: 2 [1-6]
Industrial: 3 [1-6]
UV: 4 [2-6]
SDA: 5 [2-6]
Planned: 6 [4-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```
```text
block: ordering
point: band-0km
JJR: 1 [1-5]
Other: 2 [1-6]
Industrial: 3 [1-6]
UV: 4 [2-6]
SDA: 5 [2-6]
Planned: 6 [4-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```
```text
block: ordering
point: band-1km
SDA: 1 [1-2]
Other: 2 [1-5]
Industrial: 3 [2-5]
Planned: 4 [2-5]
UV: 5 [3-6]
JJR: 6 [4-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 4
seed: 0
n: 1000
```
```text
block: ordering
point: decay-none
Other: 1 [1-6]
Industrial: 2 [1-6]
JJR: 3 [1-6]
UV: 4 [2-6]
SDA: 5 [2-6]
Planned: 6 [3-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```
```text
block: ordering
point: decay-power05
Other: 1 [1-6]
Industrial: 2 [1-6]
JJR: 3 [1-6]
UV: 4 [2-6]
SDA: 5 [2-6]
Planned: 6 [3-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```
```text
block: ordering
point: decay-power2
JJR: 1 [1-4]
Other: 2 [1-6]
Industrial: 3 [1-6]
UV: 4 [2-6]
SDA: 5 [2-6]
Planned: 6 [4-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```
```text
block: ordering
point: decay-exp2km
Other: 1 [1-6]
Industrial: 2 [1-6]
JJR: 3 [1-6]
SDA: 4 [2-6]
UV: 5 [2-6]
Planned: 6 [3-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```
```text
block: ordering
point: decay-exp5km
Other: 1 [1-6]
Industrial: 2 [1-6]
JJR: 3 [1-6]
UV: 4 [2-6]
SDA: 5 [2-6]
Planned: 6 [3-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```
```text
block: ordering
point: decay-boundary
Other: 1 [1-6]
Industrial: 2 [1-6]
JJR: 3 [1-6]
UV: 4 [2-6]
SDA: 5 [2-6]
Planned: 6 [3-6]
RUAC: 7 [7-7]
UAC: 8 [8-8]
JJC: 9 [9-9]
n_fragile_pairs: 5
seed: 0
n: 1000
```

### Finding

**The bottom three categories are pinned everywhere measured.** `RUAC`,
`UAC` and `JJC` land at ranks 7, 8 and 9 with a zero-width interval
(`7 [7-7]`, `8 [8-8]`, `9 [9-9]`) at every single point in this table,
anchors included — resampling never once produces a different bottom
three, in that order. `JJC` is the lowest-scoring category at every point,
which is the ordering the gap block below quantifies.

**The top six categories are a genuine composition-sensitivity finding,
not noise.** At the bbox baseline five of the six top categories share an
interval overlapping `[1-6]`, and `n_fragile_pairs` is `5` out of the
8 adjacent pairs the 9-category ordering has — most of the *top* of the
ranking would plausibly reorder under a different settlement draw, while
the *bottom* would not. `band-1km` and `own-only` are the two narrowest
top intervals measured (`n_fragile_pairs: 4`), and are also the two most
smoothed points in the `points` block above (`own_share_p50` `1.000`
[by construction] and `0.010` respectively) — the anchors bracket the
sweep, and the real points sit between them exactly as spec § 6.2 expects.

## Block `gap` — the formal/informal gap as an effect size

**DRY RUN on `code-2025` — superseded by the ratified profile.** The frozen
July 2025 rule set still carries `bbox` adjacency, `global_asymmetric`
barriers and decayed roads, all three of which Raj's 28 Aug 2026 decisions
change. No number in this document is quotable, and none of it belongs in
the manuscript. Phase 6's reported variants (DEL-36/37/39) are the same
sweep re-run against the ratified profile (DEL-31).

Two rows per point (spec § 6.5): `Planned` vs `JJC`, then the pooled
`formal = {Planned, SDA}` vs `informal = {JJC, JJR}` grouping (a stated
assumption of this document, not a ruling from the paper — see the
introduction above). `p_a_gt_b` (the bootstrap probability that `a`'s mean
PSI exceeds `b`'s) is **`1.000` on every single row of this table, both
groupings, at every point measured** — expected at n = 4,131 (spec § 6.5)
and carrying no information once it is constant, so the column is dropped
here rather than printed 22 times unchanged; `scripts.summarize_sweep`
still computes it (`render_gap_block`'s underlying `_gap_row`), and
`tests/test_summarize_sweep.py::test_gap_block_does_not_crash_against_the_real_partial_sweep`
pins the constant. A gated `*_decile_share_*` cell renders `—` under the
same rule as the `points` block's Jaccard cells.

```text
block: gap
point: baseline
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.90
cliffs_delta_ci_lo: 0.88
cliffs_delta_ci_hi: 0.92
cohens_d: 0.67
pct_gap: 49.3
top_decile_share_a: 0.499
top_decile_share_b: 0.000
bottom_decile_share_b: 0.522
```
```text
block: gap
point: baseline
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.85
cliffs_delta_ci_lo: 0.82
cliffs_delta_ci_hi: 0.87
cohens_d: 0.63
pct_gap: 23.2
top_decile_share_a: 0.525
top_decile_share_b: 0.019
bottom_decile_share_b: 0.524
```
```text
block: gap
point: own-only
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.83
cliffs_delta_ci_lo: 0.81
cliffs_delta_ci_hi: 0.85
cohens_d: 0.44
pct_gap: 43.3
top_decile_share_a: 0.477
top_decile_share_b: 0.000
bottom_decile_share_b: —
```
```text
block: gap
point: own-only
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.75
cliffs_delta_ci_lo: 0.72
cliffs_delta_ci_hi: 0.79
cohens_d: 0.35
pct_gap: 10.9
top_decile_share_a: 0.499
top_decile_share_b: 0.063
bottom_decile_share_b: —
```
```text
block: gap
point: adj-touch
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.92
cliffs_delta_ci_lo: 0.90
cliffs_delta_ci_hi: 0.94
cohens_d: 0.63
pct_gap: 50.8
top_decile_share_a: 0.499
top_decile_share_b: 0.000
bottom_decile_share_b: —
```
```text
block: gap
point: adj-touch
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.86
cliffs_delta_ci_lo: 0.83
cliffs_delta_ci_hi: 0.89
cohens_d: 0.59
pct_gap: 23.0
top_decile_share_a: 0.538
top_decile_share_b: 0.031
bottom_decile_share_b: —
```
```text
block: gap
point: band-0km
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.93
cliffs_delta_ci_lo: 0.91
cliffs_delta_ci_hi: 0.94
cohens_d: 0.63
pct_gap: 51.2
top_decile_share_a: 0.508
top_decile_share_b: 0.000
bottom_decile_share_b: —
```
```text
block: gap
point: band-0km
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.87
cliffs_delta_ci_lo: 0.84
cliffs_delta_ci_hi: 0.89
cohens_d: 0.59
pct_gap: 23.7
top_decile_share_a: 0.550
top_decile_share_b: 0.031
bottom_decile_share_b: —
```
```text
block: gap
point: band-1km
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.96
cliffs_delta_ci_lo: 0.95
cliffs_delta_ci_hi: 0.97
cohens_d: 0.84
pct_gap: 57.0
top_decile_share_a: 0.535
top_decile_share_b: 0.000
bottom_decile_share_b: 0.746
```
```text
block: gap
point: band-1km
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.92
cliffs_delta_ci_lo: 0.90
cliffs_delta_ci_hi: 0.94
cohens_d: 0.84
pct_gap: 36.4
top_decile_share_a: 0.625
top_decile_share_b: 0.010
bottom_decile_share_b: 0.746
```
```text
block: gap
point: decay-none
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.89
cliffs_delta_ci_lo: 0.87
cliffs_delta_ci_hi: 0.91
cohens_d: 0.60
pct_gap: 48.3
top_decile_share_a: 0.479
top_decile_share_b: 0.000
bottom_decile_share_b: 0.522
```
```text
block: gap
point: decay-none
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.84
cliffs_delta_ci_lo: 0.81
cliffs_delta_ci_hi: 0.87
cohens_d: 0.57
pct_gap: 23.4
top_decile_share_a: 0.494
top_decile_share_b: 0.010
bottom_decile_share_b: 0.524
```
```text
block: gap
point: decay-power05
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.90
cliffs_delta_ci_lo: 0.87
cliffs_delta_ci_hi: 0.91
cohens_d: 0.64
pct_gap: 48.8
top_decile_share_a: 0.501
top_decile_share_b: 0.000
bottom_decile_share_b: 0.522
```
```text
block: gap
point: decay-power05
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.84
cliffs_delta_ci_lo: 0.82
cliffs_delta_ci_hi: 0.87
cohens_d: 0.61
pct_gap: 23.4
top_decile_share_a: 0.525
top_decile_share_b: 0.015
bottom_decile_share_b: 0.524
```
```text
block: gap
point: decay-power2
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.91
cliffs_delta_ci_lo: 0.89
cliffs_delta_ci_hi: 0.93
cohens_d: 0.69
pct_gap: 50.0
top_decile_share_a: 0.475
top_decile_share_b: 0.000
bottom_decile_share_b: 0.522
```
```text
block: gap
point: decay-power2
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.85
cliffs_delta_ci_lo: 0.82
cliffs_delta_ci_hi: 0.87
cohens_d: 0.63
pct_gap: 21.5
top_decile_share_a: 0.508
top_decile_share_b: 0.039
bottom_decile_share_b: 0.524
```
```text
block: gap
point: decay-exp2km
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.90
cliffs_delta_ci_lo: 0.88
cliffs_delta_ci_hi: 0.92
cohens_d: 0.66
pct_gap: 49.2
top_decile_share_a: 0.506
top_decile_share_b: 0.000
bottom_decile_share_b: 0.522
```
```text
block: gap
point: decay-exp2km
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.85
cliffs_delta_ci_lo: 0.82
cliffs_delta_ci_hi: 0.87
cohens_d: 0.63
pct_gap: 23.5
top_decile_share_a: 0.533
top_decile_share_b: 0.017
bottom_decile_share_b: 0.524
```
```text
block: gap
point: decay-exp5km
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.89
cliffs_delta_ci_lo: 0.87
cliffs_delta_ci_hi: 0.91
cohens_d: 0.64
pct_gap: 48.6
top_decile_share_a: 0.492
top_decile_share_b: 0.000
bottom_decile_share_b: 0.522
```
```text
block: gap
point: decay-exp5km
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.84
cliffs_delta_ci_lo: 0.82
cliffs_delta_ci_hi: 0.87
cohens_d: 0.60
pct_gap: 23.5
top_decile_share_a: 0.513
top_decile_share_b: 0.015
bottom_decile_share_b: 0.524
```
```text
block: gap
point: decay-boundary
group: Planned_vs_JJC
a: Planned
b: JJC
cliffs_delta: 0.89
cliffs_delta_ci_lo: 0.87
cliffs_delta_ci_hi: 0.91
cohens_d: 0.61
pct_gap: 48.6
top_decile_share_a: 0.499
top_decile_share_b: 0.000
bottom_decile_share_b: 0.522
```
```text
block: gap
point: decay-boundary
group: formal_vs_informal
a: formal
b: informal
cliffs_delta: 0.84
cliffs_delta_ci_lo: 0.82
cliffs_delta_ci_hi: 0.87
cohens_d: 0.58
pct_gap: 23.5
top_decile_share_a: 0.518
top_decile_share_b: 0.015
bottom_decile_share_b: 0.524
```

### Finding

**Planned-over-JJC is large and stable everywhere measured.** `cliffs_delta`
for `Planned` vs `JJC` ranges `0.83`–`0.96` across every point, anchors
included — read as a probability, a randomly chosen Planned settlement
outranks a randomly chosen JJC settlement roughly 92-98% of the time
regardless of adjacency rule, band width, or decay form. `top_decile_share_b`
(JJC's share of the top decile) is `0.000` at every single point in this
table: not one JJC settlement has ever landed in the top 10% under any
factor swept so far.

**The band points widen the gap rather than narrow it.** `band-1km` has
both the largest `cliffs_delta` (`0.96`) and the largest `cohens_d`
(`0.84`) of every point measured — the opposite of what a wider
neighbourhood narrowing the formal/informal gap would look like, and the
opposite of `docs/data/rule_effects.md`'s partial-barrier finding (which
narrows Planned-over-JJC from about 33× to about 12×). `own-only`, by
contrast, has the SMALLEST gap on both statistics (`cliffs_delta: 0.83`,
`cohens_d: 0.44`) — with no neighbour term at all, the two categories'
own-service compositions are simply less separated than the spatial
comparison makes them.

**Cliff's δ and Cohen's d diverge exactly where spec § 6.5 said they
would.** At `band-1km`, δ and d are close (`0.96`/`0.84` — both near their
sweep-wide maxima), but at `own-only` they diverge sharply (`0.83`/`0.44`):
d is not invariant to the smoothing the way δ is, and the two statistics
telling a different story at the un-smoothed anchor is the tail-behaviour
signal spec § 6.5 predicted this divergence would carry.

## Block `denominator_check` — one-off check of spec § 4.3's scoping decision

**DRY RUN on `code-2025` — superseded by the ratified profile.** The frozen
July 2025 rule set still carries `bbox` adjacency, `global_asymmetric`
barriers and decayed roads, all three of which Raj's 28 Aug 2026 decisions
change. No number in this document is quotable, and none of it belongs in
the manuscript. Phase 6's reported variants (DEL-36/37/39) are the same
sweep re-run against the ratified profile (DEL-31).

Spec § 4.3 scopes every sweep point to the `popdensity` denominator alone
(the one the paper's Figure 4 uses) to avoid doubling the single most
expensive loop in `compute` at `band-10km`'s link count, and asks for a
one-off check that this scoping decision is not silently hiding a
denominator-sensitive result: the bbox baseline's category ordering and
Planned-vs-JJC effect size under `pop` beside the same under `popdensity`.
Only the baseline can answer this — it is the one point that was ever run
under both denominators.

```text
block: denominator_check
point: baseline
cat_order_pop: SDA>Other>Planned>UV>UAC>JJC>Industrial>RUAC>JJR
cat_order_popdensity: Other>Industrial>JJR>SDA>UV>Planned>RUAC>UAC>JJC
tau_pop_vs_popdensity: 0.17
cliffs_delta_planned_jjc_pop: 0.22
cliffs_delta_planned_jjc_popdensity: 0.90
agreement: DISAGREE
```

### Finding

**The two denominators DISAGREE, and the disagreement is large, not a
rounding artifact.** The nine-category orderings under `pop` and
`popdensity` correlate at only `tau_pop_vs_popdensity: 0.17` — barely above
independence — and `JJR`, ranked lowest under `popdensity`, ranks highest
under `pop`. The Planned-vs-JJC effect size moves from a small-to-moderate
`cliffs_delta_planned_jjc_pop: 0.22` (a coin flip that leans one way) to a
dominant `cliffs_delta_planned_jjc_popdensity: 0.90` under density — the
paper's own denominator. **Spec § 4.3's scoping decision is therefore not
free**: this document's `points`/`ordering`/`gap` blocks above, all run
under `popdensity` alone, would very likely look substantively different
under `pop` — not just at the margins the way the decay forms differ from
each other, but potentially reordering categories the way `pop` reorders
`JJR`. This disagreement is itself a finding for DEL-52, written down
rather than assumed away, and is exactly the reason spec § 4.3 asked for
this check to be run and recorded rather than skipped.
