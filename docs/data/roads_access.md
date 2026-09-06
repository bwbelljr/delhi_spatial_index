# Road access, and what `roads: eq4_own_only` does to today's numbers

Raj ratified Eq. 4 as the manuscript writes it — each colony counts only the
roads inside its own boundary (decision log
`docs/decisions/2026-08-28-raj-methodology-decisions.md` § 2, DEL-22/DEL-49).
The premise on the call was inverted: the July 2025 code DECAYS roads like
clinics, so this is a change from the published numbers, and Raj is owed the
size of it. Two measurements answer that, both produced by
`scripts/measure_roads_access.py`, which reads the layers named by the
`code-2025` profile and writes nothing under the data directory.
`tests/test_measure_roads_access.py` re-runs it and compares both blocks (it
skips when the data is not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

- **Run date:** 2026-09-05
- **Inputs:** settlement layer `uso_update_sep2021`, `Public Services/Major Road/Road.shp`, and the proven `code-2025` run in `~/delhi_data/phase3_verify` (its neighbours artifact, staged alone into the work dir, and its two output CSVs)
- **Commit:** `205eb7c`
- **Command:** `uv run python scripts/measure_roads_access.py --config code-2025 --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/cache`

## Block `access` — road access on the layer, no PSI

For every settlement in the deduplicated, reprojected universe (the same
loader the pathology measurement uses, so these counts describe exactly what
the pipeline scores):

- `road_inside_<TYPE>` — the settlement's polygon contains a positive length
  of the major-road layer. This is the membership `delhi_psi.index.road_lengths`
  uses, so it is `road_length > 0` in today's outputs; a road that only
  touches the boundary at a point does not count. That equivalence is not
  asserted here in prose and hoped for: with `--verify-dir` given the script
  compares its own `road_inside` set against the `road_length` column of the
  `code-2025` output CSV and RAISES if they differ, so the block below is
  either a true description of today's outputs or it was never printed.
- `road_via_neighbor_<TYPE>` — no road of its own, but at least one
  **`touch`** neighbour (positive shared border — Raj's ratified rule, DEL-19,
  not today's bbox) has one. A neighbour may be a dropped type: dropped
  settlements still lend (semantics (a)), and a road in the rural village next
  door is exactly what Raj asked about.
- `no_road_<TYPE>` — neither.

Counts are integers; shares are derived in the prose below, so the drift test
stays exact.

```text
block: access
road_inside_Planned: 551
road_inside_UAC: 184
road_inside_RUAC: 190
road_inside_JJC: 17
road_inside_JJR: 32
road_inside_UV: 69
road_inside_SDA: 44
road_inside_RV: 137
road_inside_Industrial: 33
road_inside_Other: 25
road_inside_total: 1282
road_via_neighbor_Planned: 323
road_via_neighbor_UAC: 1093
road_via_neighbor_RUAC: 150
road_via_neighbor_JJC: 645
road_via_neighbor_JJR: 13
road_via_neighbor_UV: 63
road_via_neighbor_SDA: 42
road_via_neighbor_RV: 65
road_via_neighbor_Industrial: 2
road_via_neighbor_Other: 6
road_via_neighbor_total: 2402
no_road_Planned: 90
no_road_UAC: 407
no_road_RUAC: 53
no_road_JJC: 102
no_road_JJR: 3
no_road_UV: 6
no_road_SDA: 0
no_road_RV: 9
no_road_Industrial: 1
no_road_Other: 2
no_road_total: 673
```

**Finding.** Of the 764 JJCs on the layer, `17` (2.2 %) contain a major
road, `645` (84.4 %) have none of their own but touch a neighbour that
does, and `102` (13.4 %) have neither. Planned colonies are the mirror
image: `551` of 964 (57.2 %) contain a road, `323` (33.5 %) reach one only
through a neighbour, `90` (9.3 %) have neither. Across all 4,357
settlements the split is `1282` inside, `2402` via a neighbour, `673`
neither — so under own-only roads the large majority of JJCs, and a third
of planned colonies, lose every kilometre of road they currently borrow.
Raj's intuition on the call (14:21–14:22) was right in both directions:
JJCs almost never have a major road inside, and they are almost always
next to a settlement that does.

## Block `one_factor` — the effect on today's numbers

`code-2025` with ONE value changed, `methodology.roads: eq4_own_only`, run
against the SAME neighbours artifact (the roads formula is applied downstream
in `pipeline.index_frames`, and `pipeline.methodology_stamp` carries adjacency
and barrier only, so no re-`preprocess` is needed — the script asserts that
before it runs). The `decayed` side is READ from the proven `code-2025` run,
never recomputed. Per denominator and reported type: `n`, the mean `road_idx`
and mean `unnorm_psi` under each formula, and `road_idx_zeroed` — how many
settlements had a decayed road index above zero and fall to exactly zero
because everything they had was borrowed.

```text
block: one_factor
n_pop_Planned: 964
n_pop_UAC: 1684
n_pop_RUAC: 393
n_pop_JJC: 749
n_pop_JJR: 48
n_pop_UV: 138
n_pop_SDA: 86
n_pop_total: 4131
road_idx_decayed_pop_Planned: 0.00968715
road_idx_decayed_pop_UAC: 0.0128205
road_idx_decayed_pop_RUAC: 0.00393954
road_idx_decayed_pop_JJC: 0.016615
road_idx_decayed_pop_JJR: 0.00203724
road_idx_decayed_pop_UV: 0.0104629
road_idx_decayed_pop_SDA: 0.0132625
road_idx_decayed_pop_total: 0.0117207
road_idx_own_pop_Planned: 0.0199045
road_idx_own_pop_UAC: 0.00371157
road_idx_own_pop_RUAC: 0.00976446
road_idx_own_pop_JJC: 0.000512098
road_idx_own_pop_JJR: 0.00903885
road_idx_own_pop_UV: 0.00951034
road_idx_own_pop_SDA: 0.0124024
road_idx_own_pop_total: 0.00848644
psi_decayed_pop_Planned: 0.00901129
psi_decayed_pop_UAC: 0.0106064
psi_decayed_pop_RUAC: 0.00440786
psi_decayed_pop_JJC: 0.0176336
psi_decayed_pop_JJR: 0.00269464
psi_decayed_pop_UV: 0.0103199
psi_decayed_pop_SDA: 0.0190069
psi_decayed_pop_total: 0.0109748
psi_own_pop_Planned: 0.0104709
psi_own_pop_UAC: 0.00930515
psi_own_pop_RUAC: 0.00524
psi_own_pop_JJC: 0.0153331
psi_own_pop_JJR: 0.00369487
psi_own_pop_UV: 0.0101838
psi_own_pop_SDA: 0.018884
psi_own_pop_total: 0.0105128
road_idx_zeroed_pop_Planned: 286
road_idx_zeroed_pop_UAC: 769
road_idx_zeroed_pop_RUAC: 100
road_idx_zeroed_pop_JJC: 422
road_idx_zeroed_pop_JJR: 8
road_idx_zeroed_pop_UV: 47
road_idx_zeroed_pop_SDA: 42
road_idx_zeroed_pop_total: 1682
n_popdensity_Planned: 964
n_popdensity_UAC: 1684
n_popdensity_RUAC: 393
n_popdensity_JJC: 749
n_popdensity_JJR: 48
n_popdensity_UV: 138
n_popdensity_SDA: 86
n_popdensity_total: 4131
road_idx_decayed_popdensity_Planned: 0.0342785
road_idx_decayed_popdensity_UAC: 0.00837623
road_idx_decayed_popdensity_RUAC: 0.0144938
road_idx_decayed_popdensity_JJC: 0.000691567
road_idx_decayed_popdensity_JJR: 0.00735623
road_idx_decayed_popdensity_UV: 0.0243973
road_idx_decayed_popdensity_SDA: 0.0110248
road_idx_decayed_popdensity_total: 0.0149261
road_idx_own_popdensity_Planned: 0.0195588
road_idx_own_popdensity_UAC: 0.00218583
road_idx_own_popdensity_RUAC: 0.0130458
road_idx_own_popdensity_JJC: 4.0452e-05
road_idx_own_popdensity_JJR: 0.0060766
road_idx_own_popdensity_UV: 0.0171764
road_idx_own_popdensity_SDA: 0.00288773
road_idx_own_popdensity_total: 0.00788152
psi_decayed_popdensity_Planned: 0.0304103
psi_decayed_popdensity_UAC: 0.0118991
psi_decayed_popdensity_RUAC: 0.0148165
psi_decayed_popdensity_JJC: 0.000914278
psi_decayed_popdensity_JJR: 0.0257077
psi_decayed_popdensity_UV: 0.0261655
psi_decayed_popdensity_SDA: 0.0191333
psi_decayed_popdensity_total: 0.015886
psi_own_popdensity_Planned: 0.0283075
psi_own_popdensity_UAC: 0.0110147
psi_own_popdensity_RUAC: 0.0146097
psi_own_popdensity_JJC: 0.000821261
psi_own_popdensity_JJR: 0.0255249
psi_own_popdensity_UV: 0.025134
psi_own_popdensity_SDA: 0.0179709
psi_own_popdensity_total: 0.0148796
road_idx_zeroed_popdensity_Planned: 286
road_idx_zeroed_popdensity_UAC: 769
road_idx_zeroed_popdensity_RUAC: 100
road_idx_zeroed_popdensity_JJC: 422
road_idx_zeroed_popdensity_JJR: 8
road_idx_zeroed_popdensity_UV: 47
road_idx_zeroed_popdensity_SDA: 42
road_idx_zeroed_popdensity_total: 1682
```

**Finding.** Under the population denominator the JJC mean `unnorm_psi`
moves from `0.0176336` to `0.0153331` (−13.0 %) and the Planned mean from
`0.00901129` to `0.0104709` (+16.2 %); under the population-density
denominator — the one the paper's figures use (`psi_columns.md`) — JJC moves
from `0.000914278` to `0.000821261` (−10.2 %) and Planned from `0.0304103`
to `0.0283075` (−6.9 %). The JJC mean road index itself collapses, from
`0.016615` to `0.000512098` under population, because `422` of the `749`
reported JJCs (56 %) fall to a road index of exactly zero; `1682` of the
`4131` reported settlements do. The JJC-versus-Planned ordering does not
flip under either denominator: under population density JJC stays lowest
and Planned highest, as the paper says; under population JJC's mean is
above Planned's both before and after, which is a property of that
denominator (JJC populations are small) and one more sign that the figures
were drawn from the density variant.

**Against the rule above: the decision stands.** Criterion 1 does not fire
(the largest JJC fall is 13.0 %, under the 20 % line) and criterion 2 does
not fire (no ordering flip). The effect is real and concentrated — more
than half of JJCs lose their road index entirely — and that is the sentence
for Raj's footnote: under the ratified rule the road index measures whether
a settlement has a major road of its own, and most JJCs do not.

## When this reopens the decision

Stated before the numbers, so it is a rule and not a reaction. **The roads
decision (own-only) stands unless one of these is true:**

1. the JJC mean `unnorm_psi` falls by more than 20 % relative to its
   `code-2025` value under either denominator; or
2. the ordering of mean `unnorm_psi` between JJC and Planned flips.

Either would mean the switch changes the paper's headline comparison rather
than refining it, and it goes back to Raj as a question instead of a
correction.

