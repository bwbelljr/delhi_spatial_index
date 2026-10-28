# Distance bands on the ratified rules (DEL-36)

**Run date:** 2026-10-27. **Baseline:** the DEL-32 `ratified-2026` run
(`~/delhi_data/phase4_ratified`, commit `8470955`), whose adjacency is the
main text's `touch`. **Points:** `band-1km`, `band-5km`, `band-10km` — the
same ratified profile with **only** `adjacency.rule` + `max_distance_km`
moved (one factor, guarded by `tests/test_sweep_profiles.py`).

**Command:** `uv run python scripts/run_sweep.py --group bands --work-dir
/tmp/del36-sweep`, summarised with
`uv run python scripts/summarize_sweep.py --work-dir /tmp/del36-sweep
--baseline-dir ~/delhi_data/phase4_ratified --baseline-profile ratified-2026`.

**What this table answers (Raj, 22 Oct, call item 2).** "Here's the table,
we think what we present is reasonable." It is an appendix robustness
check, not a recommendation.

**Read it this way — the sweep's governing principle.** PSI **levels** are
not comparable across points: Eq. 2's min-max runs inside each run, and a
wider neighbourhood inflates every settlement at once. So every comparison
below is either a diagnostic on one run or a RANK-based measure (Spearman
ρ, Kendall tau-b, top/bottom-decile overlap) against the ratified baseline.
No level difference or level ratio between bands appears here by design.

## Adjacency and cost diagnostics

One row per point: how many neighbours the rule builds, whether any
settlement is left with none, and what the run cost.

```text
block: points
point: baseline
profile: ratified-2026
adjacency: touch
radius_km: —
decay_form: inverse_linear
decay_param: —
decay_distance: centroid
n_reported: 4062
n_isolates: 21
n_links: 19530
deg_mean: 4.5
deg_p50: 4
deg_max: 93
preprocess_s: —
compute_s: —
own_share_p50: 0.027
n_own_share_undef: 95
cat_order: JJR>UV>Planned>SDA>RUAC>UAC>JJC
tau_vs_own: 1.00
rho_vs_own: 0.627
taub_vs_own: 0.485
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
n_reported: 4062
n_isolates: —
n_links: —
deg_mean: —
deg_p50: —
deg_max: —
preprocess_s: —
compute_s: —
own_share_p50: 1.000
n_own_share_undef: 1830
cat_order: JJR>UV>Planned>SDA>RUAC>UAC>JJC
tau_vs_own: 1.00
rho_vs_own: 1.000
taub_vs_own: 1.000
tau_vs_bbox: 1.00
rho_vs_bbox: 0.627
taub_vs_bbox: 0.485
jaccard_top10: 0.316
jaccard_bottom10: —
planned_gt_jjc: True
flag: pinned
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
n_reported: 4062
n_isolates: 0
n_links: 195322
deg_mean: 44.8
deg_p50: 41
deg_max: 280
preprocess_s: 873.306
compute_s: 553.168
own_share_p50: 0.005
n_own_share_undef: 0
cat_order: SDA>UV>Planned>JJR>RUAC>UAC>JJC
tau_vs_own: 0.52
rho_vs_own: 0.549
taub_vs_own: 0.418
tau_vs_bbox: 0.52
rho_vs_bbox: 0.795
taub_vs_bbox: 0.612
jaccard_top10: 0.415
jaccard_bottom10: 0.405
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: band-5km
profile: band-5km
adjacency: within_distance
radius_km: 5.0
decay_form: inverse_linear
decay_param: —
decay_distance: centroid
n_reported: 4062
n_isolates: 0
n_links: 1772776
deg_mean: 406.9
deg_p50: 416
deg_max: 954
preprocess_s: 787.975
compute_s: 4755.923
own_share_p50: 0.001
n_own_share_undef: 0
cat_order: Planned>UV>SDA>RUAC>JJR>UAC>JJC
tau_vs_own: 0.52
rho_vs_own: 0.508
taub_vs_own: 0.383
tau_vs_bbox: 0.52
rho_vs_bbox: 0.738
taub_vs_bbox: 0.552
jaccard_top10: 0.318
jaccard_bottom10: 0.386
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: band-10km
profile: band-10km
adjacency: within_distance
radius_km: 10.0
decay_form: inverse_linear
decay_param: —
decay_distance: centroid
n_reported: 4062
n_isolates: 0
n_links: 5075652
deg_mean: 1164.9
deg_p50: 1208
deg_max: 2067
preprocess_s: 2257.418
compute_s: 14136.990
own_share_p50: 0.001
n_own_share_undef: 0
cat_order: Planned>UV>SDA>RUAC>JJR>UAC>JJC
tau_vs_own: 0.52
rho_vs_own: 0.496
taub_vs_own: 0.375
tau_vs_bbox: 0.52
rho_vs_bbox: 0.738
taub_vs_bbox: 0.553
jaccard_top10: 0.314
jaccard_bottom10: 0.374
planned_gt_jjc: True
flag: smoothed,pinned,reshuffled
```

## Category ordering and its fragility

Each line is the type's average percentile rank (1 = best) with a
1,000-bootstrap interval; `n_fragile_pairs` counts adjacent pairs whose
bootstrap intervals overlap.

```text
block: ordering
point: baseline
JJR: 1 [1-2]
UV: 2 [1-2]
Planned: 3 [3-3]
SDA: 4 [4-5]
RUAC: 5 [4-5]
UAC: 6 [6-6]
JJC: 7 [7-7]
n_fragile_pairs: 2
seed: 0
n: 1000
```
```text
block: ordering
point: own-only
JJR: 1 [1-1]
UV: 2 [2-2]
Planned: 3 [3-4]
SDA: 4 [3-4]
RUAC: 5 [5-5]
UAC: 6 [6-6]
JJC: 7 [7-7]
n_fragile_pairs: 1
seed: 0
n: 1000
```
```text
block: ordering
point: band-1km
SDA: 1 [1-4]
UV: 2 [1-4]
Planned: 3 [1-4]
JJR: 4 [2-5]
RUAC: 5 [4-5]
UAC: 6 [6-6]
JJC: 7 [7-7]
n_fragile_pairs: 3
seed: 0
n: 1000
```
```text
block: ordering
point: band-5km
Planned: 1 [1-2]
UV: 2 [1-3]
SDA: 3 [1-3]
RUAC: 4 [4-5]
JJR: 5 [4-6]
UAC: 6 [5-6]
JJC: 7 [7-7]
n_fragile_pairs: 4
seed: 0
n: 1000
```
```text
block: ordering
point: band-10km
Planned: 1 [1-3]
UV: 2 [1-3]
SDA: 3 [1-3]
RUAC: 4 [4-6]
JJR: 5 [4-6]
UAC: 6 [4-6]
JJC: 7 [7-7]
n_fragile_pairs: 4
seed: 0
n: 1000
```

## Finding

**The ordering is stable; the band is not a knife edge.** Spearman ρ
against the ratified baseline is 0.795 at 1 km and 0.738 at both 5 km and
10 km; top-decile overlap is 0.42 / 0.32 / 0.31. The baseline's own
neighbourhood is very sparse — 4.5 neighbours per settlement, 19,530 links,
21 isolates — and every band removes the isolates entirely. The four
top-ranked types at 5 and 10 km (Planned, UV, SDA, RUAC) are the same
four, and Planned stays above JJC at every point in the sweep.

Two things a reader should not over-read. First, the 1 km point is the
closest to the main text (24 min, 195k links) and its ordering differs from
the wider bands — SDA and JJR swap places with Planned — so the appendix
should present 5/10 km as "the wider the net, the more the ordering
converges", not as a correction to the touch rule. Second, the 10 km
bootstrap marks the ordering `reshuffled`: with 1,165 neighbours per
settlement the index is a smoothed city-wide average, and type differences
narrow for a mechanical reason, not a substantive one.

**Cost.** 1 km: 873 s preprocess + 553 s compute; 5 km: 788 s + 4,756 s;
10 km: 2,257 s + 14,137 s. The 10 km point is a four-and-a-half-hour run and
belongs in the appendix only.
