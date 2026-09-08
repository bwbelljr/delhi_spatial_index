# Rank aggregation as a methodology variant — DEL-57

**Ticket:** DEL-57. **Branch:** `del-57-rank-aggregation` off `main` at `1b85dc8`.
**Date:** 8 Sep 2026. Cycle 5, ticket 2 of 3.

---

## 1. The reading DEL-35 did not build

WORKPLAN's Phase 6 entry said *"instead of averaging, explore ranking
mechanisms"*, which admitted two readings. DEL-35 built **(b) a different
report** — `scripts/rank_report.py`, which keeps Eq. 1 and summarises a
computed run by rank. This ticket is **(a) a different index**: rank
settlements on each service, then combine the ranks.

It is a real and different idea, and it was split out onto its own ticket
rather than dropped in a PR body.

**Why it is worth having.** The compression that motivated DEL-34 is a
property of min-max on skewed distributions: `docs/data/phase6_sweep.md`
records **452 of 4,131 reported settlements at exactly `norm_psi == 0`** at
the published baseline — a mass point larger than a decile, and the reason
that document needed decile gating at all. **Ranks are uniform by
construction**, so no mass point can form. It answers DEL-34's complaint by
a different route.

**Why it is not free, and why this ticket does not adopt it.** Eq. 1's
average of normalised counts preserves *magnitude* — a settlement with ten
clinics scores above one with two. A rank aggregation preserves only
*order*, discarding exactly the intensity the paper's "how much service is
reachable" framing rests on. That is a methodological argument, not an
implementation detail. **Raj decides it**; this ticket makes the option
measurable, exactly as DEL-34 did for the transforms, and both shipped
profiles keep today's behaviour.

## 2. The switch

**`methodology.aggregation.rule`: `mean_minmax` | `mean_rank`**

- `mean_minmax` — today. Eq. 2 min-maxes each service's PCEN across
  settlements; Eq. 1 averages those. Both shipped profiles keep it, so
  nothing moves.
- `mean_rank` — each service's PCEN is replaced by its **percentile rank
  across settlements**, and Eq. 1 averages those instead.

It is a sibling of `transform`, not a member of it: `transform` chooses a
function applied to a value, `aggregation` chooses what Eq. 2 *is*. The
ticket says so and it is right.

**Everything downstream is unchanged.** `overall_psi` still averages the
`*_idx` columns into `unnorm_psi`, still applies a `psi`-stage transform if
one is configured, and still min-maxes into `norm_psi` when
`second_normalization` is on. The switch replaces one step.

## 3. The rank rule, stated exactly enough for the oracle

For a service's PCEN column across the `n` reported settlements:

1. rank ascending, **averaging ranks within a tie block** — the standard
   convention, and the one `scripts/summarize_sweep.percentile_rank`
   already uses;
2. rescale `(rank - 1) / (n - 1)`.

Step 2's scaling is chosen so that **the minimum maps to 0 and the maximum
to 1 — the same endpoints Eq. 2 produces**. That makes `mean_rank` a
drop-in for `mean_minmax` in range, so `unnorm_psi` stays on [0, 1] and no
downstream code learns a new range.

**`n == 1` raises**, naming the column, exactly as `minmax`'s `hi == lo`
guard does (DEL-54): `(rank - 1) / 0` is the same 0/0 in different
clothing, and inventing a value for a one-settlement city is the thing that
guard exists to refuse.

## 4. Two behavioural differences from `minmax`, both measured

**A constant column stops being an error.** `index.minmax` raises when
`max == min`, because Eq. 2 is genuinely undefined there — a service no
settlement owns, or one they all own equally. Under `mean_rank` every
settlement ties, every average rank is equal, and the result is **0.5 for
everyone**: defined, and a uniform additive shift that changes no ordering.
Verified: `pct_rank(np.full(7, 0.4))` → a single distinct value, `0.5`.

This is a real difference in what the pipeline will accept, not a detail.
It should be stated in the config docs rather than discovered.

**A `pcen`-stage transform becomes an exact no-op.** DEL-34's `transform`
applies `log1p` or `cbrt`; both are strictly monotone and injective, so
they preserve both the ordering *and* the tie structure, so the average
ranks are unchanged. Verified rather than asserted, on 100 values with a
30 % mass at zero — the shape that actually causes the compression:

| transform at `pcen` | max abs difference in rank |
|---|---|
| `log1p` | `0.000e+00` |
| `cbrt` | `0.000e+00` |

Exactly zero, not merely close. **This turns the ticket's prose claim — "if
a rank aggregation is adopted, the `transform` knob becomes largely moot" —
into a test**, and it is the sharpest available statement of how DEL-34 and
DEL-57 relate: they are alternative answers to one complaint, and combining
them at the `pcen` stage does nothing at all.

A `psi`-stage transform is **not** a no-op under `mean_rank` — it acts on
the composite before the second normalisation, and the composite is a mean
of ranks, not a rank. The combination stays legal and is covered by a
variant row.

## 5. What must not move

- Both shipped profiles carry `rule: mean_minmax`, so **every existing
  expected value and production fixture is byte-identical**. If one moves,
  the switch has leaked into the default path — STOP.
- `variants_expected_values.csv` is **addition-only**, verified by `diff`.
- The real-data `code-2025` baseline still verifies at `0.000e+00`.

## 6. Variant coverage

Three new rows in `tests/variants.py`, each scored by both implementations
on both fixture cities at `1e-12`:

| row | what it pins |
|---|---|
| `aggregation_mean_rank` | the rule itself, against a hand-checkable ranking |
| `aggregation_mean_rank_log1p_pcen` | the no-op property — must equal `aggregation_mean_rank` **exactly** |
| `aggregation_mean_rank_log1p_psi` | that a `psi`-stage transform still bites under `mean_rank` |

The second row is the interesting one: it is the only variant in this repo
whose expected values must be **identical to another variant's**, and
asserting that identity directly is a stronger test than the 1e-12
agreement it also gets.

**The fixture cities are the right size for this.** Oraculum's 7
settlements make every average rank hand-computable, and the messy city's
tie structure (`ration_idx` is 0 for all but one settlement) exercises the
tie-block averaging that a naive `argsort` would get wrong.

## 7. Out of scope

- **No adoption.** Neither shipped profile changes.
- **No real-data run.** Cycle 5's standing constraint — the reportable
  numbers belong to the ratified profile (DEL-31).
- **No rank version of the second normalisation.** The switch replaces
  Eq. 2 only. Ranking `unnorm_psi` as well is a further idea and would need
  its own argument.
- **No licence file** (DEL-56, pending Raj).

## 8. Definition of done

- `methodology.aggregation.rule` in config, reference-pinned, present in
  every profile with today's value.
- The rule implemented in `delhi_psi/index.py` and, **independently**, in
  `tests/reference_impl.py`, which imports nothing from `delhi_psi`.
- The `n == 1` guard, with its own test.
- Three variant rows agreeing at 1e-12 on both cities.
- The `log1p`-at-`pcen` no-op asserted as an exact equality.
- The constant-column difference documented in `docs/methodology-config.md`.
- Every existing fixture byte-identical; `variants_expected_values.csv`
  addition-only; full suite green.
