# Which PSI column, and which denominator, do the figures report?

Raj does not know which column the April 2026 draft's figures were drawn
from (decision log
`docs/decisions/2026-08-28-raj-methodology-decisions.md` §§ 7–8, DEL-52); the
memo's fallback was that Bob measures it and Raj confirms. This document is
that measurement. `scripts/measure_psi_columns.py` reads the two July 2025
baseline outputs — the files that produced the figures — computes the mean
per `USO_FINAL` type for each of the four candidates and scores each against
the figure's bars. `tests/test_measure_psi_columns.py` re-runs it and
compares the block (it skips when the data is not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

- **Run date:** 2026-09-05
- **Inputs:** `~/delhi_data/psi_2020_results` (baseline: `delhi_psi_bbox_popsize2020_norv_12Sep2021.csv`, `delhi_psi_bbox_popdensity2020_norv_12Sep2021.csv`), `~/delhi_data/phase3_verify` (the `code-2025` cross-check)
- **Commit:** `205eb7c`
- **Command:** `uv run python scripts/measure_psi_columns.py --baseline-dir ~/delhi_data/psi_2020_results --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/psi-columns`

## What the figure shows

Figure 4, "Mean public service index by settlement" (p. 40 of the April 2026
draft PDF), is a bar chart of the mean PSI per settlement type with 95 %
whiskers. Its y-axis is labelled **"Mean Public Services Index (per person
per square kilometer)"** — a population-DENSITY denominator — and footnote 12
says the average "rarely exceeds 0.05". It has eight bars: no RV, no Other.

Bar values read off the chart on 5 Sep 2026. **They are read-offs, not data**
(no figure data file exists in the repo or the data folder — checked the same
day), so the script scores a match within ±0.002:

| type | bar (≈) | type | bar (≈) |
|---|---|---|---|
| JJR | 0.037 | RUAC | 0.021 |
| JJC | 0.0015 | UAC | 0.017 |
| SDA | 0.028 | UV | 0.038 |
| Planned | 0.044 | Industrial | 0.038 |

## The four candidates

`unnorm_psi` is Eq. 1 as the methods write it: the mean of the min-maxed
per-service indices. `norm_psi` is that column min-maxed a second time, which
stretches it to [0, 1] — a step the methods never mention. Each is available
under two denominators: `popsize` (population) and `popdensity`
(population / area). The block below names them `<column>_<denom>`; the
`matched_*` keys count bars hit within ±0.002 and the `maxgap_*` keys give
the worst absolute miss.

`verify_maxdiff_*` keys are the cross-check: the same per-type means computed
from the refactored `code-2025` run must equal the baseline's to 1e-9, or the
script fails rather than reporting.

```text
mean_unnorm_psi_popsize_Industrial: 0.00456462
mean_unnorm_psi_popsize_JJC: 0.0176336
mean_unnorm_psi_popsize_JJR: 0.00269464
mean_unnorm_psi_popsize_Other: 0.01505
mean_unnorm_psi_popsize_Planned: 0.00901129
mean_unnorm_psi_popsize_RUAC: 0.00440786
mean_unnorm_psi_popsize_SDA: 0.0190069
mean_unnorm_psi_popsize_UAC: 0.0106064
mean_unnorm_psi_popsize_UV: 0.0103199
matched_unnorm_psi_popsize: 0
maxgap_unnorm_psi_popsize: 0.0350
mean_unnorm_psi_popdensity_Industrial: 0.0258889
mean_unnorm_psi_popdensity_JJC: 0.000914278
mean_unnorm_psi_popdensity_JJR: 0.0257077
mean_unnorm_psi_popdensity_Other: 0.0709526
mean_unnorm_psi_popdensity_Planned: 0.0304103
mean_unnorm_psi_popdensity_RUAC: 0.0148165
mean_unnorm_psi_popdensity_SDA: 0.0191333
mean_unnorm_psi_popdensity_UAC: 0.0118991
mean_unnorm_psi_popdensity_UV: 0.0261655
matched_unnorm_psi_popdensity: 1
maxgap_unnorm_psi_popdensity: 0.0136
mean_norm_psi_popsize_Industrial: 0.00556781
mean_norm_psi_popsize_JJC: 0.021509
mean_norm_psi_popsize_JJR: 0.00328685
mean_norm_psi_popsize_Other: 0.0183576
mean_norm_psi_popsize_Planned: 0.0109918
mean_norm_psi_popsize_RUAC: 0.0053766
mean_norm_psi_popsize_SDA: 0.0231841
mean_norm_psi_popsize_UAC: 0.0129375
mean_norm_psi_popsize_UV: 0.0125879
matched_norm_psi_popsize: 0
maxgap_norm_psi_popsize: 0.0337
mean_norm_psi_popdensity_Industrial: 0.0377171
mean_norm_psi_popdensity_JJC: 0.001332
mean_norm_psi_popdensity_JJR: 0.0374532
mean_norm_psi_popdensity_Other: 0.10337
mean_norm_psi_popdensity_Planned: 0.0443043
mean_norm_psi_popdensity_RUAC: 0.021586
mean_norm_psi_popdensity_SDA: 0.027875
mean_norm_psi_popdensity_UAC: 0.0173356
mean_norm_psi_popdensity_UV: 0.0381201
matched_norm_psi_popdensity: 8
maxgap_norm_psi_popdensity: 0.0006
verify_maxdiff_unnorm_psi_popsize: 0.0e+00
verify_maxdiff_unnorm_psi_popdensity: 0.0e+00
verify_maxdiff_norm_psi_popsize: 0.0e+00
verify_maxdiff_norm_psi_popdensity: 0.0e+00
best_candidate: norm_psi_popdensity
```

## Finding

**The figures report `norm_psi` under the population-density denominator.**
`norm_psi_popdensity` matched `8` of 8 bars within ±0.002 with a maximum gap
of `0.0006`; the next best candidate, `unnorm_psi_popdensity`, matched `1`
of 8 with a maximum gap of `0.0136`, and neither population-size candidate
matched a single bar. The cross-check held: the refactored `code-2025` run
reproduces every per-type mean of the baseline exactly (`0.0e+00`).

Both halves of the answer land on the opposite side of Bob's proposed
defaults (decision log §§ 7–8): the paper's headline figure is the
second-normalised column AND the population-density variant. The
consequences, one axis at a time, are below.

## What follows from the answer

**Two independent axes.** The finding names a COLUMN and a DENOMINATOR, and
they are answered separately: `best_candidate` is a pair, and either half can
land where Bob's proposed default did not. One paragraph each, below. Neither
paragraph decides anything — the decision is Raj's; this document supplies the
numbers he asked for (decision log §§ 7–8).

### The column axis — `unnorm_psi` or `norm_psi`

**`norm_psi` matches and `unnorm_psi` does not, so Bob's proposed default of
`second_normalization: false` is withdrawn.** The paper's headline figure
reports the SECOND-normalised column; switching it off would silently move
every bar in Figure 4 (under popdensity the Planned mean would fall from
`0.0443043` to `0.0304103`, the JJC mean from `0.001332` to `0.000914278`).
The real choice for Raj is: keep `norm_psi` as the reported PSI and add the
second min-max to the methods — one sentence after Eq. 1, "the mean is then
min-max scaled across settlements" — or switch the figures to Eq. 1 as written
and let every bar move. Both columns stay in the config either way; this is a
question about what the paper reports, not about what the pipeline can
compute.

*This outcome was stated before the run:* the plan-review round of 5 Sep 2026
had already run the comparison against the same baseline files and found the
same 8-of-8 match, so the write-up was not fitted to the result.

### The denominator axis — `popsize` or `popdensity`

**The popdensity denominator matches, as the y-axis label said it would, so
Bob's proposed default of dropping popdensity from the reported results is
withdrawn** — the paper's headline figure is the popdensity variant, and the
population-size candidates matched no bar at all. The real choice for Raj:
keep popdensity as the reported denominator and add its equation to the
methods (Eq. 3 with Population_i / Area_i), or switch the figures to the
per-population Eq. 3 the manuscript prints. Both denominators stay in the
config either way.

### What this does not decide

Nothing here changes a profile. `code-2025` keeps both columns and both
denominators; the ratified profile (DEL-31) sets `second_normalization` and
`outputs.denominators` from Raj's answer to the two choices above, and the
methods text gains a sentence for each. The escalation branch of the spec
("no candidate matches at least 6 of 8") did not fire.
