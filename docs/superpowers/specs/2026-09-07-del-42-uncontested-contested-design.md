# Uncontested vs contested services — DEL-42

**Ticket:** DEL-42. **Branch:** `del-42-service-panels` off `main`.
**Date:** 7 Sep 2026. Cycle 4, ticket 3 of 5.
**Depends on:** DEL-40's service-subset rail.

---

## 1. The ticket could not be built as written

It said: *"Core-universal-services variant: schools, health, water only."*

**There is no water layer.** `~/delhi_data/Public Services/` holds seven
directories — Banking, Health, Major Road, Police, Ration, School, Transport.
Water was aspirational when the variant was written down months ago; it was
never collected.

Escalated to Fable on 7 Sep per Bob's standing rule for decisions that block a
ticket. Its recommendation is adopted in full and recorded on the Jira ticket.
The four load-bearing parts:

**Build it as schools + health. Do not add road length.** Roads are a line
measure on a different scale, so a three-item index turns the aggregation
weights into a design choice a reviewer will probe — and roads are the service
where the formal/informal gap is most mechanically guaranteed, since planned
colonies have planned road grids *by definition*. Including them inflates the
gap and hands a critic exactly the objection this variant exists to remove.

**Two services is a harder test, not a weaker one.** The obvious worry is that
seven services down to two leaves scores resting on too few observations. It
cuts the other way: schools and health are the services most likely to be
placed *inside* poor settlements by state policy — Sarvodaya schools,
dispensaries, mohalla clinics — so a JJC deficit that survives on those two
alone is stronger evidence than the seven-service result. Sparsity of owned
services is also not the hazard it looks: a settlement owning neither still
scores through distance-decayed neighbours.

**Rename it.** "Core universal services" is not accurate without water. This is
the **uncontested-services** variant, where *uncontested* means "services no
reviewer has argued are private, targeted, or coercive".

**Widen it into a decomposition.** Run the complement as well. Three panels —
the full index, uncontested, and **contested-only** (banks, police, ration,
transport, roads). If the gap appears in *both halves*, it is a property of
settlement location rather than of any one service class, which is a more
persuasive result than a single stripped-down variant. Transport's placement
is genuinely ambiguous; it goes in the contested panel and the ambiguity is
stated in one line rather than hidden.

## 2. Scope

Two profiles:

| profile | `services.point` | `services.line` |
|---|---|---|
| `services-uncontested` | `health`, `school` | *(none)* |
| `services-contested` | `bank`, `police`, `ration`, `transport` | `road` |

Together they partition `code-2025`'s seven services exactly — a property
worth asserting in a test, because a partition that silently drops or
duplicates a service would make the decomposition meaningless.

Registration, fixtures on both cities, the one-factor guard extended to these
two, and the partition test. **Durable half only** — no real-data run.

## 3. `services.line` empty is a new case

Every profile so far carries `road` under `services.line`.
`services-uncontested` carries none, so the loader, the fixture generator and
`index_frames` all meet an empty line-service set for the first time. Check it
is handled rather than assumed:

- does `_services` accept an empty `line` mapping, or does a default merge put
  `road` back? (`{**DEFAULT_SERVICES, **raw}` merges at the top level, so
  `line: {}` should replace it — verify, do not assume.)
- does `metric_columns` emit `road_length` / `road_pcen` / `road_idx` rows for
  a profile with no road service? It must not.

If an empty `line` block turns out not to be expressible, that is a finding to
report, not to work around by leaving `road` in the uncontested panel — doing
that would reintroduce exactly the contamination § 1 rejects.

## 4. What the fixtures must prove

For each panel, the three conditions DEL-40 established (absent metric rows,
`psi_eq1`/`norm_psi` move, remaining raw counts unchanged), plus:

- **the partition holds**: the union of the two panels' service sets equals
  `code-2025`'s, and their intersection is empty. Assert on the loaded
  configs, not on the YAML text.

## 5. What the write-up must say

Recorded now so it is not invented later. In the methods, in the body and not
a footnote:

> The pre-registered variant specified schools, health and water. No
> settlement-level water-provision layer was available, so it is implemented
> with schools and health only. It is therefore a two-service test, and we do
> not claim it captures piped-water access, which is plausibly the dimension
> on which JJCs fare worst.

That final clause is the one that matters. Without it a reviewer infers water
was dropped because it was inconvenient.

## 6. Definition of done

- Both profiles ship, registered, guarded, and partitioning `code-2025`'s
  services exactly.
- An empty `services.line` is either demonstrably supported, or reported as a
  blocker — not worked around.
- All existing fixtures byte-identical; suite green; no real-data run.
