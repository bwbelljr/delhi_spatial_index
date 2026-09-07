# Banking in/out — DEL-41

**Ticket:** DEL-41. **Branch:** `del-41-banking-sensitivity` off `main`.
**Date:** 7 Sep 2026. Cycle 4, ticket 2 of 5.
**Depends on:** DEL-40, which builds the service-subset rail. Do not start
this until that has merged — without it a subset profile's fixture is
silently identical to the full-service one
(`docs/superpowers/specs/2026-09-07-service-subset-variants-design.md` § 1).

---

## 1. Why this ticket exists, and why it is a measurement rather than an argument

Two people looked at the same service and disagreed:

- **The Brown workshop triage:** drop banking — it is private and
  market-driven, not public provision.
- **Raj's annotation on that triage:** *"they are material assets"* — a bank
  branch in a settlement is a real economic amenity, and its absence is a real
  deprivation, whoever owns it.

Both are reasonable and neither is checkable by further discussion. What
settles it is what the index does with and without banking, which is one
profile and one fixture once DEL-40's rail exists. That is the whole point of
the ticket: convert a standing disagreement into a number, the same way the
four pre-recalculation measurements converted Raj's open questions in
cycle 2.

## 2. Scope

One profile: `services-no-bank` — `code-2025` with `services.point` omitting
`bank`, and nothing else moved. Registration, production fixtures on both
cities, and the one-factor guard (which by then compares `services` too).

**Durable half only.** No real-data run: DEL-41's reportable numbers belong to
the ratified profile (DEL-31). Cycle 4's standing constraint, DEL-7 comment of
7 Sep.

## 3. What the fixture must prove

Identical in shape to DEL-40's, with `bank` in place of `ration`:

- **no `bank_*` metric rows at all** — absent, not zeroed;
- every settlement's `psi_eq1` and `norm_psi` **move**, because Eq. 1 now
  averages six terms rather than seven;
- the other raw counts are **unchanged** — dropping a service changes what is
  averaged, not what is counted.

If DEL-40's rail is right, this should be a near-mechanical repeat. If any of
the three conditions needs special handling for banking specifically, that is
a defect in the rail, not a property of banking — report it rather than
working around it.

## 4. One thing to check that DEL-40 does not face

Oraculum and the messy city both place a `bank`. Confirm before building that
dropping it leaves each city with at least one service in every scenario —
a fixture city reduced to zero services in some exclusion scenario would make
Eq. 1 an average over an empty set, and the right response is to say so, not
to emit a NaN fixture. `messy-city.md` records the service placement; check it
rather than assuming.

## 5. What the write-up must say when the numbers eventually come

Not part of this ticket's code, but stated here so it is not invented later:
the comparison to report is **whether the formal/informal ordering and the
Planned-vs-JJC effect size survive** the removal, not whether the index levels
move — they always move, because the average is over fewer terms. That is the
same discipline `docs/data/phase6_sweep.md` established: levels are not
comparable across variants, rank-based statements are.

## 6. Definition of done

- `services-no-bank` ships, registered and guarded, differing from
  `code-2025` in exactly `services.point`.
- The three § 3 conditions hold, each with a test that fails if it does not.
- `expected_values.csv`, `variants_expected_values.csv` and the `code-2025` /
  `manuscript` production fixtures byte-identical.
- `uv run pytest -q -W error` green. No real-data run, no new dependency.
