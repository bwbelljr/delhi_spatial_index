# The reported relative scale: 1 = the average Planned colony

Raj's answer to the last open methodology question (22 Oct 2026; decision
log `docs/decisions/2026-10-22-raj-call-decisions.md`, DEL-63) is that the
paper should report a relative scale: divide every settlement's `norm_psi`
by the mean `norm_psi` of the Planned colonies, so the average planned
colony reads 1 and every other type reads as a multiple of it. This
document is that measurement. `scripts/measure_planned_relative.py` runs
`code-2025` with one methodology value changed — `second_normalization:
planned_relative` — and reports the resulting per-type means for both
denominators. `tests/test_measure_planned_relative.py` re-runs it and
compares the block.

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

- **Run date:** 2026-10-26
- **Inputs:** `~/delhi_data` (the layers `code-2025` names), `~/delhi_data/phase3_verify` (the proven neighbours artifact, reused unchanged)
- **Commit:** `4b8d3fe`
- **Command:** `uv run python scripts/measure_planned_relative.py --config code-2025 --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/planned-relative`

## What the scale is

   rel_psi_i = norm_psi_i / mean(norm_psi | category == "Planned")

One constant per run: the mean of the min-max column over the Planned rows
of the same computed frame — post-exclusion, post-everything. `rel_psi` is
a pure per-row rescale, so a type's mean `rel_psi` IS the ratio of its mean
`norm_psi` to the Planned mean; the block measures that identity
independently (`rel_vs_ratio_maxdiff_*`).

The relative scale **inherits the min-max denominator**: `norm_psi` is an
affine transform of `unnorm_psi`, so ratios of means are not invariant to
min-maxing. That belongs in the methods text beside the scale itself.

```text
block: planned_relative
planned_norm_mean_pop: 0.0109918
mean_rel_psi_pop_Planned: 1
mean_norm_psi_pop_Planned: 0.0109918
mean_rel_psi_pop_UAC: 1.17702
mean_norm_psi_pop_UAC: 0.0129375
mean_rel_psi_pop_RUAC: 0.489149
mean_norm_psi_pop_RUAC: 0.0053766
mean_rel_psi_pop_JJC: 1.95683
mean_norm_psi_pop_JJC: 0.021509
mean_rel_psi_pop_JJR: 0.299029
mean_norm_psi_pop_JJR: 0.00328685
mean_rel_psi_pop_UV: 1.14521
mean_norm_psi_pop_UV: 0.0125879
mean_rel_psi_pop_SDA: 2.10923
mean_norm_psi_pop_SDA: 0.0231841
mean_rel_psi_pop_Industrial: 0.506544
mean_norm_psi_pop_Industrial: 0.00556781
mean_rel_psi_pop_Other: 1.67012
mean_norm_psi_pop_Other: 0.0183576
rel_vs_ratio_maxdiff_pop: 5.10703e-15
planned_norm_mean_popdensity: 0.0443043
mean_rel_psi_popdensity_Planned: 1
mean_norm_psi_popdensity_Planned: 0.0443043
mean_rel_psi_popdensity_UAC: 0.391285
mean_norm_psi_popdensity_UAC: 0.0173356
mean_rel_psi_popdensity_RUAC: 0.487221
mean_norm_psi_popdensity_RUAC: 0.021586
mean_rel_psi_popdensity_JJC: 0.0300647
mean_norm_psi_popdensity_JJC: 0.001332
mean_rel_psi_popdensity_JJR: 0.845362
mean_norm_psi_popdensity_JJR: 0.0374532
mean_rel_psi_popdensity_UV: 0.860416
mean_norm_psi_popdensity_UV: 0.0381201
mean_rel_psi_popdensity_SDA: 0.629172
mean_norm_psi_popdensity_SDA: 0.027875
mean_rel_psi_popdensity_Industrial: 0.851319
mean_norm_psi_popdensity_Industrial: 0.0377171
mean_rel_psi_popdensity_Other: 2.33318
mean_norm_psi_popdensity_Other: 0.10337
rel_vs_ratio_maxdiff_popdensity: 1.33227e-15
```

## The numbers

Under the population-density denominator the average Planned colony reads
`1`, UV `0.860416`, RUAC `0.487221`, UAC `0.391285` and JJC `0.0300647`:
the gap the 0–1 scale buried is the reported quantity. The full per-type
means for both denominators are in the block above; each denominator's
Planned mean `norm_psi` is the divisor (`0.0443043` under popdensity).

Under the population-size denominator the ordering is different — UV still
reads above 1, JJC is second-highest at `1.95683`, and JJR is the lowest at
`0.299029` — which is why the paper must name the denominator beside the
scale.

## What this does not decide

Nothing here changes a shipped profile. `code-2025` still runs `minmax`;
the ratified profile (DEL-31) is where `planned_relative` is adopted, after
Raj ratifies these numbers. Main-paper placement is Raj's writing call
("i am thinking move the relative to the main paper"); the calculation is
decided. Recalculation (DEL-32) and the main figure (DEL-33) follow DEL-31.
