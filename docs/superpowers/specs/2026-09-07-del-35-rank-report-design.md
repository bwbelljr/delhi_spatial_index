# Rank-based reporting — DEL-35

**Ticket:** DEL-35. **Branch:** `del-35-rank-based-index` off `main` at `8ef2c9c`.
**Date:** 7 Sep 2026. Cycle 4, ticket 4 of 5.

---

## 1. An ambiguity in the ticket, settled before any code

WORKPLAN reads:

> **Rank-based index** (new idea from the workshop): instead of averaging,
> explore ranking mechanisms — average rank per settlement category, and
> composition of the top/bottom deciles by category (as in
> intergenerational-mobility research) [DEL-35]

"Instead of averaging" admits two readings, and they are very different pieces
of work:

**(a) A different INDEX.** Replace Eq. 1's mean-of-normalised-columns with a
rank aggregation — rank settlements on each service, then combine ranks.
That is a new methodology value, needs a reference-implementation rule, oracle
expected values, and Raj's ratification.

**(b) A different REPORT.** Keep Eq. 1, and summarise it by rank rather than
by level — mean percentile rank per category, and each category's share of the
top and bottom deciles.

**Reading (b) is correct**, and the ticket's own two examples settle it: "average
rank per settlement category" and "composition of the top/bottom deciles by
category" are both ways of *describing* a computed index, not ways of computing
one. The intergenerational-mobility analogy points the same way — that
literature ranks outcomes and reports transition matrices; it does not build
the outcome out of ranks.

Reading (a) is a real idea and partly overlaps DEL-34 (alternative
transformations for compressed effect sizes). If it is wanted, it should be
its own ticket with its own reference rule. **This spec builds (b)** and says
so on the ticket, so nobody later assumes (a) was silently dropped.

## 2. Most of this already exists

DEL-55 built the rank machinery inside `scripts/summarize_sweep.py`, because
comparing eleven sweep points *required* rank-based statistics — PSI levels
are not comparable across runs. Already implemented and tested there:

| function | what DEL-35 asks for |
|---|---|
| `percentile_rank` | the rank transform, 0–100, average-tied |
| `category_order` | categories ordered by mean percentile rank |
| `decile_set` / `decile_is_gated` | tie-inclusive decile membership, gated when a tie block is oversized |
| `decile_share` | a category's share of a decile |
| `bootstrap_rank_intervals` | rank intervals with a seeded, genuinely random tie-break |
| `cliffs_delta`, `spearman_rho`, `kendall_tau_b` | the effect sizes and correlations |

So DEL-35 is **extraction and productisation**, not new statistics. The work
is making these usable for *one run reported on its own terms*, rather than
only for cross-point comparison.

## 3. Scope

**`scripts/rank_report.py`** — reads one output CSV (the same
`OUTPUT_USECOLS` shape) and emits the category-level rank table in the
established `docs/data/` fenced-block form:

Per category, in one block:
- `n` — settlements in the category
- `mean_pct_rank` (1 dp) — the headline, and the reason this ticket exists:
  a rank mean is robust to the min-max compression that makes Eq. 1's levels
  hard to read
- `median_pct_rank` (1 dp)
- `top_decile_share`, `bottom_decile_share` (3 dp) — each gated to `—` when
  the decile's tie block exceeds 1.5 × k, per the DEL-55 rule
- `rank_ci_lo`, `rank_ci_hi` — the bootstrap interval on the category's mean
  rank, seed 0, 1,000 draws

Plus one summary line: `n_reported`, `decile_k`, `n_psi_tied_at_zero`, and
whether either decile gated.

**Reuse, do not reimplement.** Import from `scripts.summarize_sweep`. If a
function needs generalising to serve both callers, generalise it there — two
copies of a tie-inclusive decile rule is exactly the drift this repo has spent
the week removing.

**Out of scope:** no real-data run (cycle 4's standing constraint — the
reportable numbers belong to the ratified profile, DEL-31); no new index; no
change to Eq. 1.

## 4. What the tests must pin

- **The rank transform on a hand-computed case**, including ties — the same
  discipline as DEL-55's statistics.
- **A category's mean percentile rank** against a frame whose answer is
  computable by hand.
- **Decile gating fires**, reusing DEL-55's proven threshold behaviour.
- **A fixture-scale end-to-end run**: the report generated from a committed
  production fixture, so the script is exercised against real committed data
  without needing Delhi's layers. Oraculum has 7 settlements across 6
  categories, which is small enough to check by hand and large enough that the
  decile logic has to gate — assert that it does, rather than pretending a
  7-row decile is meaningful.

That last point is the honest one: **on a 7-settlement city a decile is 0.7
settlements.** The report must gate rather than emit a number, and the test
must assert the gating. A rank report that silently reported "the top decile is
100% Planned" from one settlement would be exactly the kind of
authoritative-looking nonsense this repo keeps catching.

## 5. Definition of done

- `scripts/rank_report.py` + `tests/test_rank_report.py`, green.
- No statistic reimplemented that `summarize_sweep` already has.
- Decile gating asserted on fixture-scale data.
- Full suite green; no existing fixture moved; no real-data run.
- The (a)/(b) reading recorded on the Jira ticket.
