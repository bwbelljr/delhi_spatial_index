# Service-subset variants — DEL-40, and the rail they need

**Tickets:** DEL-40 (ration shops in/out) first, then DEL-41 (ATMs/banking),
then DEL-42 (uncontested vs contested panels).
**Branch:** `del-40-ration-sensitivity` off `main` at `477f64b`.
**Date:** 7 Sep 2026. Cycle 4, ticket 1 of 5.

---

## 1. The finding this spec exists because of

DEL-40 looks like a one-line ticket: ship a profile that drops ration shops,
generate its fixture, done. **That would have produced a fixture identical to
the full-service one, and a green test proving nothing.**

Neither existing mechanism can express a service subset:

- **`tests/variants.py` cannot.** `variant_methodology` (in
  `tests/oraculum_fixtures.py:155`) applies a variant's overrides to
  `methodology` blocks — `adjacency`, `barrier`, `overlap`, `decay`.
  `services` is a *separate top-level config block*, so no variant row can
  reach it.
- **The production-fixture path cannot, and does so silently.**
  `compute_oracle_frame` (`tests/oraculum_fixtures.py:206`) passes
  `city.load_services()`, which reads every feature in the fixture city's
  `services.geojson` and groups by the `service` property. It never consults
  `cfg.services`. Grep confirms: neither `tests/oraculum_fixtures.py` nor
  `scripts/generate_production_fixtures.py` references the profile's service
  configuration at all.

So a `services-no-ration` profile would today emit a production fixture with
ration shops still in it. The profile would look pinned and would not be.

This is dormant rather than broken: no shipped profile subsets services, so
nothing is currently wrong. DEL-40 is what wakes it, and the fix belongs here
rather than in three tickets' worth of workarounds.

## 2. Scope

**Task 1 — make the fixture rail service-aware.** `compute_oracle_frame`
filters the city's services to those the profile configures, and the
production-fixture generator emits only the metrics for services that profile
actually carries. A test must fail if a subset profile produces full-service
numbers.

**Task 2 — DEL-40's two profiles.** `services-no-ration` (the six others) and,
for symmetry in the write-up, nothing else: the comparison is against
`code-2025` itself, which already carries all seven.

**Task 3 — registration and fixtures**, per `docs/methodology-config.md` § 3
as extended by DEL-45: `PROFILES` in two places, `SHIPPED` in
`tests/test_config.py`, the one-factor guard in `tests/test_sweep_profiles.py`
extended to compare `services` as well as `methodology` and `categories`.

**Out of scope, deliberately:** any real-data run. Cycle 4's standing
constraint (DEL-7 comment, 7 Sep): these build the durable half only —
profiles, fixtures, guards — because their reportable numbers belong to the
ratified profile (DEL-31), which does not exist yet. DEL-55 already bought one
provisional sweep to prove machinery; a second buys nothing.

## 3. What "one factor" means when the factor is a service list

The existing guard compares `methodology` and `categories` between a sweep
profile and `code-2025`, and asserts the diff is exactly the claimed keys. A
service-subset profile moves neither — it moves `services.point`. The guard
must therefore learn a third comparison, or these profiles ship unguarded,
which is the same silent-pin failure in a different place.

`services-no-ration` must differ from `code-2025` in **exactly** one way:
`services.point` is missing the `ration` key, and every other key of every
other block is identical.

## 4. What the fixture must prove

A subset profile's fixture is only meaningful if it differs from the full
profile's in the right way. Specifically, for `services-no-ration` on both
fixture cities:

- **no `ration_*` metric rows at all** — not zeroed rows, absent ones;
- **every settlement's `psi_eq1` MOVES**, because Eq. 1 averages over the
  services present and the average is now over six terms rather than seven;
- **`clinic_count`, `school_count` and the other raw counts do NOT move** —
  dropping a service changes what is averaged, not what is counted.

That third condition is the one that catches a wrong implementation: a filter
applied in the wrong place would change counts too.

**Corrected after implementation — `norm_psi` does NOT move for every
settlement, and an earlier draft of this spec wrongly required it to.**
On Oraculum, `ration_idx` is `0` for every settlement except `D`, so dropping
ration rescales `unnorm_psi` by exactly 7/6 for all the others — and min-max
normalisation is invariant under a positive affine transform. Measured: `A`,
`C` and `IND` tie **bit-for-bit**, `B`, `E` and `RV` differ only in the last
ulp from summing in a different order, and only `D` genuinely moves.

So the `norm_psi` assertion is "differs somewhere", not "differs everywhere",
and `psi_eq1` carries the unconditional claim. The implementer found this and
was right to push back; a spec that demanded the stronger condition would have
forced either a false test or a contorted implementation.

## 5. Why DEL-40 is worth building at all

Ration shops are subsidised food distribution, targeted at poor households by
design. Counting them as "public service access" may **mechanically flatter
exactly the settlements the paper argues are underserved** — a JJC with a
ration shop scores for a service that exists *because* the settlement is poor.

So this variant is not housekeeping. If the formal/informal gap widens without
ration shops, the paper's claim was being suppressed by its own service
basket; if it narrows, the claim depends partly on a service whose presence is
a consequence of deprivation. Either result qualifies the argument, which is
why the WORKPLAN entry says "either way it qualifies the argument".

## 6. Definition of done

- A subset profile's fixture demonstrably differs from the full profile's, in
  the three specific ways of § 4, and a test fails if it does not.
- The one-factor guard covers `services`.
- `uv run pytest -q -W error` green; `expected_values.csv`,
  `variants_expected_values.csv` and the `code-2025` / `manuscript` production
  fixtures byte-identical.
- No real-data run, no new dependency, no licence file.
