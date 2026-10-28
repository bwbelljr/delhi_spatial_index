# Decay weight and distance definition (DEL-37)

**Run date:** 2026-10-27. **Baseline:** the DEL-32 `ratified-2026` run
(`~/delhi_data/phase4_ratified`, commit `8470955`), whose decay is
`inverse_linear` at the centroid distance. **Points:** six ratified
profiles with only the decay block moved — `none`, `inverse_power` at
exponent 0.5 and 2, `exponential` at 2 km and 5 km scales, and the
`boundary` distance.

**Command:** `uv run python scripts/run_sweep.py --group decay --work-dir
/tmp/del37-sweep`, summarised against the ratified baseline. One shared
neighbours artifact: `methodology_stamp` covers adjacency and barrier only,
so a decay change cannot invalidate it.

**Read it this way.** As with the distance bands, PSI **levels** are not
comparable across points (Eq. 2's min-max runs per run); every figure below
is rank-based against the baseline.

## Diagnostics

One row per point: the neighbour structure it inherits, its run cost, and
its rank agreement with the baseline.

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
point: decay-none
profile: decay-none
adjacency: touch
radius_km: —
decay_form: none
decay_param: —
decay_distance: centroid
n_reported: 4062
n_isolates: —
n_links: —
deg_mean: —
deg_p50: —
deg_max: —
preprocess_s: —
compute_s: 85.826
own_share_p50: 0.014
n_own_share_undef: 95
cat_order: JJR>UV>Planned>RUAC>SDA>UAC>JJC
tau_vs_own: 0.90
rho_vs_own: 0.598
taub_vs_own: 0.457
tau_vs_bbox: 0.90
rho_vs_bbox: 0.991
taub_vs_bbox: 0.926
jaccard_top10: 0.754
jaccard_bottom10: 0.897
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-power05
profile: decay-power05
adjacency: touch
radius_km: —
decay_form: inverse_power
decay_param: 0.50
decay_distance: centroid
n_reported: 4062
n_isolates: —
n_links: —
deg_mean: —
deg_p50: —
deg_max: —
preprocess_s: —
compute_s: 66.725
own_share_p50: 0.020
n_own_share_undef: 95
cat_order: JJR>UV>Planned>RUAC>SDA>UAC>JJC
tau_vs_own: 0.90
rho_vs_own: 0.610
taub_vs_own: 0.469
tau_vs_bbox: 0.90
rho_vs_bbox: 0.998
taub_vs_bbox: 0.962
jaccard_top10: 0.858
jaccard_bottom10: 0.943
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-power2
profile: decay-power2
adjacency: touch
radius_km: —
decay_form: inverse_power
decay_param: 2.00
decay_distance: centroid
n_reported: 4062
n_isolates: —
n_links: —
deg_mean: —
deg_p50: —
deg_max: —
preprocess_s: —
compute_s: 72.118
own_share_p50: 0.048
n_own_share_undef: 95
cat_order: JJR>UV>Planned>SDA>RUAC>UAC>JJC
tau_vs_own: 1.00
rho_vs_own: 0.668
taub_vs_own: 0.525
tau_vs_bbox: 1.00
rho_vs_bbox: 0.989
taub_vs_bbox: 0.916
jaccard_top10: 0.692
jaccard_bottom10: 0.924
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-exp2km
profile: decay-exp2km
adjacency: touch
radius_km: —
decay_form: exponential
decay_param: 2.0
decay_distance: centroid
n_reported: 4062
n_isolates: —
n_links: —
deg_mean: —
deg_p50: —
deg_max: —
preprocess_s: —
compute_s: 72.942
own_share_p50: 0.024
n_own_share_undef: 95
cat_order: JJR>UV>Planned>RUAC>SDA>UAC>JJC
tau_vs_own: 0.90
rho_vs_own: 0.620
taub_vs_own: 0.478
tau_vs_bbox: 0.90
rho_vs_bbox: 0.999
taub_vs_bbox: 0.980
jaccard_top10: 0.906
jaccard_bottom10: 0.961
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-exp5km
profile: decay-exp5km
adjacency: touch
radius_km: —
decay_form: exponential
decay_param: 5.0
decay_distance: centroid
n_reported: 4062
n_isolates: —
n_links: —
deg_mean: —
deg_p50: —
deg_max: —
preprocess_s: —
compute_s: 65.824
own_share_p50: 0.017
n_own_share_undef: 95
cat_order: JJR>UV>Planned>RUAC>SDA>UAC>JJC
tau_vs_own: 0.90
rho_vs_own: 0.604
taub_vs_own: 0.463
tau_vs_bbox: 0.90
rho_vs_bbox: 0.996
taub_vs_bbox: 0.952
jaccard_top10: 0.854
jaccard_bottom10: 0.933
planned_gt_jjc: True
flag: smoothed,pinned
```
```text
block: points
point: decay-boundary
profile: decay-boundary
adjacency: touch
radius_km: —
decay_form: inverse_linear
decay_param: —
decay_distance: boundary
n_reported: 4062
n_isolates: —
n_links: —
deg_mean: —
deg_p50: —
deg_max: —
preprocess_s: —
compute_s: 67.925
own_share_p50: 0.014
n_own_share_undef: 95
cat_order: JJR>UV>Planned>RUAC>SDA>UAC>JJC
tau_vs_own: 0.90
rho_vs_own: 0.598
taub_vs_own: 0.457
tau_vs_bbox: 0.90
rho_vs_bbox: 0.991
taub_vs_bbox: 0.926
jaccard_top10: 0.754
jaccard_bottom10: 0.897
planned_gt_jjc: True
flag: smoothed,pinned
```

## Finding

**The decay form is not load-bearing.** Rank correlation against the
baseline is 0.989–0.999 at every point, and top-decile overlap is 0.69–0.91;
no point reorders the top three (`JJR`, `UV`, `Planned`), and `Planned` stays
above `JJC` everywhere. The only movement is the `SDA`/`RUAC` pair, which
swaps at four of the five non-baseline forms — the same pair the bootstrap
already marks as fragile, and the one whose two types sit within 0.01 of
each other on the main run.

So Raj's steer — keep 1/(1+d) km for the main text — is supported: none of
the forms that spread the weights up from zero changes the story, and the
`boundary` distance (the one that makes every touching neighbour undecayed)
is equally quiet (ρ 0.991).

**Cost.** Six compute-only points, 66–86 s each; no `preprocess`.
