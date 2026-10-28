# DEL-38 — per-service distance expectations: design note

**Status:** design note, not implemented. This is the one P2 variant that
cannot be built from the repo's current data.

## What the ticket asks

Raj's workshop note: a school may reasonably be farther than a water point
or a ration shop, so the single decay function `1/(1+d)` may be wrong *per
service* — the walkability / food-desert framing. The reported variant would
let each service carry its own distance expectation.

## Why it is not a YAML edit

Decay is a single methodology block (`decay.form`, `decay.distance`,
`decay.distance_unit`) applied uniformly by
`index.decay_weights`/`reference_impl.decay_weight`. A per-service version
needs:

1. a new config value (e.g. `decay.per_service: {school: {scale_km: 3}}`),
   validated against the configured service names;
2. the same rule in the reference implementation, written independently;
3. a production fixture pin on both cities (the fixture services are few, so
   the pin would be thin but real);
4. a report variant, since the whole point is the *comparison* with the
   uniform rule.

That is a DEL-34-shaped change (a new knob with a reference rule), not a
sweep point.

## What is missing before it can be scoped honestly

The variant needs an **empirical anchor for each service's expected
distance**. The repo holds the service locations and the settlements, but no
travel-distance or usage data — the workshop note is a prior, not a
measurement. Choosing per-service scales without that anchor would be
fabricating the number the appendix is supposed to test.

## Recommended next step

Ask Raj for the per-service expectations he has in mind (or a source), then
scope the knob against those values. Until then, the honest appendix
statement is the one the current tables support: the *uniform* decay form is
not load-bearing (DEL-37, ρ 0.989–0.999), so the risk this variant probes is
a refinement, not a threat to the main result.
