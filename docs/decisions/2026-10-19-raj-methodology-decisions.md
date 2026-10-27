# Methodology decisions — Raj's reply to the batched email, 19 Oct 2026

**Source.** Raj's email "Re: meeting notes — decisions recorded, plus four
things measured" (19 Oct 2026, in the thread of the 28 Aug notes and
Bob's batched reply of 5–6 Sep), plus two SMS nudges (19–20 Oct: "Bob I
sent you an email. I have a question about the index and the low number
even for planned", "Thoughts?"). Item numbers below follow the batched
email. Bob's answer — the scale explanation plus the never-sent 7 Sep
sweep draft — went out as a reply in the same thread on 20 Oct 2026.

**What this resolves.** Everything the batched reply asked except one:
the `norm_psi` choice (DEL-52a), which Raj wants to settle on a call
("happy to jump on a call"). Also on the record, his writing feedback:
"the AI writing is hard to parse (FYI)".

---

## Summary table

| # | Question (batched item) | Decision | Lands as | Ticket |
|---|---|---|---|---|
| 1 | Roads: own-boundary only | **Ratified** | config `roads: eq4_own_only` (already measured, DEL-49) | DEL-22/49 ✓ |
| 2 | Overlap lending (Bob's addition) | **Ratified** | config `overlap.lending: outside_receiver` (landed in 3E) | DEL-20 ✓ |
| 3 | UV and SDA stay in | **Confirmed** | no change (identity mapping) | DEL-28 ✓ |
| 4b | Denominator | **Keep popdensity; fix the methods text** | config `outputs.denominators: popdensity`; methods gains Eq. 3 with Population_i / Area_i | DEL-52 ✓ |
| 4a | Column / second normalization | **OPEN** — wants a call | pending; facts in § 4 | DEL-52a |
| 5 | Barrier-layer provenance | **Parked** — "we can ask bijoy or shubhada" | measurement stands; question delegated | DEL-51 ✓ (Done) |
| 8 | Students' reclassification list | **Delivered**; join validated 20 Oct | DEL-53 unblocked | DEL-53 |

## 1. Roads — ratified (batched item 1)

Raj: "each colony counts only the roads inside its own boundary. I think
this is how it should be. I think i agree what you say." The DEL-49
measurement stands as the price of the call: JJC mean PSI −13 % (pop) /
−10 % (density), no JJC-vs-Planned ordering flip, 422 of 749 reported
JJCs to a zero road index. `eq4_own_only` goes in the ratified profile;
the `decayed` value stays in `code-2025` as the July 2025 baseline.

## 2. Overlap lending — ratified (batched item 2)

Raj: "YES! thanks for catching this." The `outside_receiver` value —
implemented and proven in cycle 3E — becomes the ratified profile's
`overlap.lending`. Measured price (DEL-20, `docs/data/rule_effects.md`):
about one settlement in eleven had a PCEN that moved; nothing outside
the overlapping set moved; no ordering change.

## 3. Settlement types — confirmed (batched item 3)

Raj: "YES." UV (138) and SDA (86) stay in as their own reported types.
Seven reported types, identity mapping, exclusion `[RV, Industrial,
Other]` — the 28 Aug decision, now explicit.

## 4. The DEL-52 pair (batched item 4)

**(b) Denominator — decided.** Raj: "b) yes it is density. let us fix
the text." The figures keep the population-density denominator; the
methods gain its equation (Eq. 3 with Population_i / Area_i in place of
Population_i). Bob drafts the sentence, with the justification below.

**(a) Column — open.** Raj: "I don't understand the numbers at all. why
is the planned colony index 0.0044 … what is 1? is it just one
settlement?" Answered in the 20 Oct reply with the facts for the call
(measured on the July 2025 baseline, `norm_psi` × popdensity, 4,131
scored settlements):

- The Figure 4 Planned bar is `0.044`, not 0.0044 — fn. 12's "rarely
  exceeds 0.05" is the same phenomenon.
- **1.0 is exactly one settlement** — `USO_AREA_U` 2227, a Planned
  colony (0.35 km², ~406 residents). The second min-max maps the
  best-scoring settlement to 1 and the worst to 0; Figure 4 averages
  the type.
- The distribution is heavily right-skewed: median `norm_psi` 0.0075,
  p90 0.055, p99 0.22; 88.6 % of settlements score below 0.05 and only
  13 of 4,131 above 0.5. 8 of the top 12 are Planned, but the typical
  Planned colony sits at 0.0227, so the type mean lands at 0.044. The
  low value is the scale's skew, not bad service — Planned is the
  best-served reported type.
- Measured, not adopted: rank-based reporting (DEL-35 / DEL-57 — Bob's
  lean: add alongside the current figure, not replace it) and
  log1p/cbrt transforms (DEL-34).

**Informed-consent caveat (recorded before treating (b) as final).**
Raj answered (b) without having seen the 7 Sep sweep draft, which was
never sent (folded into the 20 Oct reply): the density denominator is
services × area / population — it rewards land (rank correlation 0.93
with category median area); JJC drops 26 percentile points under
density (45th → 19th); Planned-vs-JJC Cliff's delta 0.22 → 0.90
(P(Planned outscores JJC) 0.61 → 0.95). Density remains defensible —
service access per resident per km² is a real thing to measure — but
the methods sentence must own the choice explicitly; Raj can veto it on
the call.

## 5. Barrier-layer provenance — parked (batched item 5)

Raj: "we can ask bijoy or shubhada. for now, let us leave this out."
The measurement stands (`docs/data/barriers.md`); the question — which
agency's layer, and what "manually marked" covered — goes to
Bijoy/Shubhada; nothing in the pipeline blocks on it. DEL-51 closed
20 Oct 2026.

## 6. The reclassification list — delivered and validated (batched item 8)

Raj's link (19 Oct): the students' Google Sheet, tab "Sheet with New
Classifications" — Population Estimate, SETTLEMENT NAME, uso_area_u,
Initial Settlement Classification, FINAL CLASSIFICATION, Ambiguous,
Changed?.

**Validated 20 Oct 2026** against `colonies_no_duplicates_July2025.csv`
(4,357 settlements):

- 4,342 rows; **all 4,342 `uso_area_u` IDs match the layer** — zero
  duplicates, zero unmatched; 15 layer IDs have no sheet row.
- 105 flagged changed; 69 flagged ambiguous (disjoint). The 24 changed
  non-RV rows all agree with the state layer's initial types. Rohini
  (2797) UV → Planned — the flagged example checks out.
- **81 of the 105 changes are RV → UV** — Raj's "~80 % rural → urban
  village" note confirmed. Those settlements enter the reported
  universe in the variant run (RV is excluded in the main run).

**Data-quality notes (small; query the students):** 3 UV rows are
flagged changed with FINAL == initial (2411 Nawada, 1707 Hastsal, 1994
Khampur — no-ops either way); 4 rows differ without either flag (3
single-type: 1089, 1488, 1318; 1 multi-type: 3488); one FINAL uses
"RC", which is not in the sheet's own legend (2836, an ambiguous row).

**Proposed handling rule (Bob, for the variant profile):** a settlement
takes the sheet's FINAL CLASSIFICATION when that is a single type;
multi-type FINALs (the 69 ambiguous rows, plus 3488) keep the state
classification. That yields 105 effective single-type changes (102
flagged + 3 unflagged; the three flagged UV→UV rows are no-ops). No
production code — the variant profile points at the reclassified type
column and joins on `USO_AREA_U` (`docs/methodology-config.md` § 2).

## What this means for the plan

- **DEL-31 is one line from done.** Every knob is decided except
  `second_normalization`; the call settles it, and can ratify
  `barrier.buffer_m` at the same time (propose 5 m, the Oraculum
  anchor).
- Then DEL-31 → DEL-32 → DEL-33, and the Phase 6 re-runs on the
  ratified rule set (the 7 Sep dry-run numbers are superseded).
- **DEL-53 unblocked**; the variant run lands on the ratified profile's
  rules.
- The Jira board carries the same record (comments on
  DEL-52/51/53/31/20/28/49; DEL-51 → Done).
