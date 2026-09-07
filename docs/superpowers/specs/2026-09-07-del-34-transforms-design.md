# Index transformations — DEL-34

**Ticket:** DEL-34. **Branch:** `del-34-transforms` off `main`.
**Date:** 7 Sep 2026. Cycle 4, ticket 5 of 5 — and the only one that changes
the index rather than the service set.

---

## 1. What the ticket is actually asking

> Alternative formulations for the compressed 0–1 effect sizes ("make the
> values less small"): transformations (e.g. log), tested against the oracle
> first (the 2021 `Transforms for Skewed Data` exploration in
> `archive/master-2021/` is prior art — none were adopted then).

The complaint behind it is real and visible in this repo's own numbers. Eq. 2
min-maxes each service's PCEN across settlements. Those distributions are
heavily right-skewed — a handful of settlements hold most of a service — so
min-max pushes the great majority toward zero. `docs/data/phase6_sweep.md`
measures the consequence at the published baseline: **452 of 4,131 reported
settlements sit at exactly `norm_psi == 0`**, a mass point larger than a
decile, which is why that document had to gate its decile statistics at all.

So this is not cosmetic. A compressed index makes real differences
unreadable, and it forced genuine methodology contortions two tickets ago.

## 2. Unlike the rest of cycle 4, this is a methodology change

DEL-40/41/42 shipped service subsets — config values with no new maths.
DEL-35 shipped a report. **This one alters what the index computes**, so it
takes the full 3D-style treatment:

- a new `methodology.transform` block with enum values,
- a matching rule in `tests/reference_impl.py` (which imports nothing from
  `delhi_psi`), so both implementations are scored independently,
- rows in `tests/variants.py` cross-checked at 1e-12 on both fixture cities,
- production fixtures,
- **`code-2025` and `manuscript` keep `form: none`** — today's behaviour —
  so nothing shipped moves.

Raj ratifies which, if any, is adopted. This ticket makes the options
measurable; it does not choose.

## 3. Two knobs, because placement matters as much as form

**`transform.form`: `none` | `log1p` | `cbrt`**

- `none` — today. Both shipped profiles keep it.
- `log1p` — `log(1 + x)`. The 2021 notebook's choice, and the standard remedy
  for right skew with a zero floor. Defined at 0, which matters: many
  settlements own nothing.
- `cbrt` — `x^(1/3)`. Also in the 2021 exploration. Gentler than log, and
  unlike a square root it is defined and monotone for any input the pipeline
  can produce.

Both are **pure, parameter-free functions of a single value**, which is what
makes them cheap to pin: the reference implementation and production must
agree pointwise, with no fitted state.

**Deliberately excluded, with reasons:**

- **Yeo-Johnson** (the 2021 notebook's first attempt, via
  `sklearn.preprocessing.PowerTransformer`). Two disqualifiers: it needs
  scikit-learn, which is not a dependency of this project and would be a
  heavyweight addition for one transform; and it *fits* its λ from the data,
  so the transform is not a function of a settlement's own value but of the
  whole distribution. That is not fatal — min-max is data-dependent too — but
  it makes the oracle's per-settlement hand-derivation impossible, which is
  the repo's strongest correctness tool. If Raj wants it, it should be its own
  ticket that argues for the dependency.
- **Sigmoid** (also in the 2021 notebook). Needs a scale parameter, and
  choosing one is exactly the arbitrary decision Phase 6 exists to remove.

**`transform.stage`: `pcen` | `psi`**

Where the transform applies, and the two choices answer different questions:

- **`pcen`** — transform each service's PCEN *before* Eq. 2's min-max. This
  attacks the compression at its source: the skew is in the per-service
  distributions, and spreading them out before normalising is what stops the
  mass point at zero forming.
- **`psi`** — transform the composite *after* averaging, before the second
  normalisation. This is what the 2021 notebook did (`log(unnorm_psi + 1)`).
  It re-spreads the final score but cannot undo per-service compression that
  has already happened.

`pcen` is the more principled placement and `psi` is the prior art, and the
project has no basis to pick between them yet. Both ship; Raj sees both.
Required iff `form` is not `none`, and rejected when it is — the same
conditional-parameter shape as `decay.exponent` and `barrier.buffer_m`
(`config.py::_conditional_number` is the existing pattern; this is an enum, so
it needs the enum equivalent).

## 4. What must not move

- `code-2025` and `manuscript` carry `form: none`, so **every existing
  expected value and production fixture is byte-identical**. If any moves,
  STOP — the transform has leaked into the default path.
- `variants_expected_values.csv` is **addition-only**, verified by `diff`.
- The real-data baseline still verifies at `0.000e+00`.

## 5. Variant coverage

Four new rows in `tests/variants.py` — the cross-product that matters:
`log1p`×`pcen`, `log1p`×`psi`, `cbrt`×`pcen`, `cbrt`×`psi`. Each scored by
both implementations on both fixture cities.

**One property worth asserting beyond agreement:** a monotone transform must
not change the *ordering* of settlements within a service, only the spacing.
So for any variant, the rank correlation of each `*_pcen` column against
`none` must be exactly 1. If it is not, the transform has been applied
somewhere it should not be — for instance after min-max rather than before.
That is a cheap test and it catches the most likely implementation error.

## 6. Out of scope

- **No real-data run.** Cycle 4's standing constraint: the reportable numbers
  belong to the ratified profile (DEL-31).
- **No adoption.** Neither shipped profile changes. This is measurement
  machinery for a decision Raj makes later.
- **No rank-aggregated index.** DEL-35 flagged that reading as overlapping
  this ticket; it remains a separate idea needing its own rule.

## 7. Definition of done

- `methodology.transform.{form,stage}` in config, with the conditional rule
  and its rejection message.
- The transform implemented in `delhi_psi/index.py` and, independently, in
  `tests/reference_impl.py`.
- Four variant rows agreeing at 1e-12 on both cities.
- The monotonicity property asserted.
- Every existing fixture byte-identical; `variants_expected_values.csv`
  addition-only; full suite green.
