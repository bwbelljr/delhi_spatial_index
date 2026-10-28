# Adjacency comparison: touch vs bbox vs 0 km (DEL-39)

**Run date:** 2026-10-27. **Baseline:** the DEL-32 `ratified-2026` run
(`~/delhi_data/phase4_ratified`, commit `8470955`), whose adjacency is
`touch` — the main text's rule (Raj, 28 Aug). **Points:** `adj-touch` (bbox,
the method the July 2025 numbers used) and `band-0km` (within 0 km:
intersection-inclusive, i.e. touch plus corner-only contacts and overlaps).

**Command:** `uv run python scripts/run_sweep.py --group adjacency
--work-dir /tmp/del39-sweep`, summarised against the ratified baseline.
Both points need their own neighbours, so both pay a `preprocess`.

**Read it this way.** Levels are not comparable across points (Eq. 2's
min-max runs per run); the comparisons are rank-based.

## Diagnostics

One row per point: the neighbour structure its rule builds, its run cost,
and its rank agreement with the baseline.

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
point: adj-touch
profile: adj-touch
adjacency: bbox
radius_km: —
decay_form: inverse_linear
decay_param: —
decay_distance: centroid
n_reported: 4062
n_isolates: 6
n_links: 29118
deg_mean: 6.7
deg_p50: 6
deg_max: 116
preprocess_s: 826.364
compute_s: 95.382
own_share_p50: 0.012
n_own_share_undef: 11
cat_order: UV>JJR>Planned>RUAC>SDA>UAC>JJC
tau_vs_own: 0.81
rho_vs_own: 0.566
taub_vs_own: 0.435
tau_vs_bbox: 0.81
rho_vs_bbox: 0.865
taub_vs_bbox: 0.721
jaccard_top10: 0.605
jaccard_bottom10: 0.504
planned_gt_jjc: True
flag: smoothed,pinned
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
n_reported: 4062
n_isolates: 20
n_links: 20842
deg_mean: 4.8
deg_p50: 4
deg_max: 98
preprocess_s: 22.432
compute_s: 71.580
own_share_p50: 0.026
n_own_share_undef: 91
cat_order: JJR>UV>Planned>SDA>RUAC>UAC>JJC
tau_vs_own: 1.00
rho_vs_own: 0.633
taub_vs_own: 0.488
tau_vs_bbox: 1.00
rho_vs_bbox: 0.991
taub_vs_bbox: 0.967
jaccard_top10: 0.884
jaccard_bottom10: 0.971
planned_gt_jjc: True
flag: smoothed,pinned
```

## Finding

**The main text's rule is the strict one, and the alternative is not
equivalent.** bbox builds 29,118 links where touch builds 19,530 (degree
6.7 vs 4.5) and leaves 6 settlements with no neighbour where touch leaves
21: a bounding-box overlap is not a shared border, and the difference is
not cosmetic — rank correlation is 0.865 and only 60 % of the top decile
survives. bbox also swaps the top two types (`UV` above `JJR`).

The 0 km band is the nearest neighbour of the main rule (ρ 0.991, 88 % top
decile) — as it should be, since it differs only by the corner-only pairs
touch excludes. Its ordering is identical.

So the appendix statement is: the published numbers used bbox; the ratified
results use touch; the gap between them is real but modest, and the 0 km
point confirms touch sits where the authors intended — between the loose
bbox and the strict border rule.
