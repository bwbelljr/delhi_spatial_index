# What `methodology.barrier: partial_weighted` does to today's numbers

Raj's 28 Aug 2026 decision log § 4 (`docs/decisions/2026-08-28-raj-methodology-decisions.md`)
ratifies partial-coverage weighting for barriers (`w_ij = 1 −
L_blocked(i,j) / L_shared(i,j)`, linear, symmetric) in place of the code's
global/asymmetric severing rule, and asks for the same one-factor
quantification against today's numbers "as for roads" (§ 4, § 5's "What
goes in the batched reply to Raj"). This document is that quantification
for the barrier rule, produced by `scripts/measure_rule_effects.py`, which
reads the layers named by the `code-2025` profile and writes nothing under
the data directory. `tests/test_measure_rule_effects.py` re-runs it and
compares the block below (it skips when the data is not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

- **Run date:** _pending_
- **Inputs:** _pending_
- **Commit:** _pending_
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
out of the artifact entirely; and the fractional class is non-empty. A result
outside these bounds is a stop, not a number to write down.

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

Proven on Oraculum (spec § 6.1), where the classification is hand-counted:
10 undirected `bbox` pairs make 20 directed links, of which only the A–D
edge is fractional in both directions (w = 0.08, the 5 m round-cap buffer
against a canal covering 90% of the shared edge) and none is severed — 18
at w == 1, 2 fractional, 0 severed. (These are the Oraculum hand counts, not
the real-layer block below; they are deliberately not in `backticks` so the
prose-number guard never mistakes one for the other.)

The fenced block itself is pasted here once the run step (spec § 6.6) has
produced it against `~/delhi_data/phase3_verify`; until then this section
states what each key means and what the numbers must satisfy, not what they
are.
