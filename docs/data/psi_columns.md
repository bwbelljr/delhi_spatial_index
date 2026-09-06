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

- **Run date:** _pending — the run step fills this in_
- **Inputs:** _pending_
- **Commit:** _pending_
- **Command:** `uv run python scripts/measure_psi_columns.py --baseline-dir <baseline-dir> --verify-dir <verify-dir> --work-dir <work-dir>`

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

## What follows from the answer

**Two independent axes.** The finding names a COLUMN and a DENOMINATOR, and
they are answered separately: `best_candidate` is a pair, and either half can
land where Bob's proposed default did not. One paragraph each, below. Neither
paragraph decides anything — the decision is Raj's; this document supplies the
numbers he asked for (decision log §§ 7–8).

### The column axis — `unnorm_psi` or `norm_psi`

- **If `unnorm_psi` matches:** the paper already reports Eq. 1 as the methods
  write it. `second_normalization: false` costs nothing and removes a column
  the methods never mention. Bob's recommendation stands.
- **If `norm_psi` matches and `unnorm_psi` does not: Bob's proposed default of
  `second_normalization: false` is withdrawn.** The paper's headline figure
  reports the SECOND-normalised column, so switching it off would silently
  move every bar in Figure 4. The real choice for Raj, then, is: keep
  `norm_psi` as the reported PSI and add the second min-max to the methods —
  one sentence after Eq. 1, "the mean is then min-max scaled across
  settlements" — or switch the figures to Eq. 1 as written and let every bar
  move. Both columns stay in the config either way; this is a question about
  what the paper reports, not about what the pipeline can compute.

  *Expected outcome, stated before the run so the write-up cannot be
  back-fitted:* the plan-review round of 5 Sep 2026 already ran this
  comparison against the same baseline files and found norm_psi under
  popdensity matching 8 of the 8 bars, max gap 0.0006, against 1 of 8 for
  unnorm_psi. The script's job is to make that reproducible and drift-tested,
  not to discover it.

### The denominator axis — `popsize` or `popdensity`

- **If the popdensity denominator matches** (the y-axis label says it will):
  **Bob's proposed default of dropping popdensity from the reported results is
  withdrawn** — the paper's headline figure is the popdensity variant. The
  real choice for Raj: keep popdensity as the reported denominator and add its
  equation to the methods (Eq. 3 with Population_i / Area_i), or switch the
  figures to the per-population Eq. 3 the manuscript prints. Both denominators
  stay in the config either way.
- **If popsize matches instead:** the manuscript's Eq. 3 and its figures agree
  and Bob's proposed default stands unchanged.

### If nothing matches

If no candidate matches at least 6 of the 8 bars, this document prints all
four candidate tables in full and the finding is "the figure was not produced
from these columns as-is". That is an escalation to the owner (spec § 7), not
a guess.

*(The fenced block and the finding sentence are written by the run step —
they are the numbers. The consequence paragraphs above are written HERE,
before the run, so the write-up cannot be force-fitted to whichever branch
the data lands in; the run step keeps the branch that fired, deletes the
others, and fills in the measured N and G.)*
