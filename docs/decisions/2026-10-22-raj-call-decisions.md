# Methodology decisions — Bob/Raj call, 22 Oct 2026

**Source.** Zoom call "Bob / Rajesh", Thu 22 Oct 2026, 15:33–15:49 CT
(scheduled via Raj's SMS of 21–22 Oct). Raj's post-call email "Re:
meeting notes" (22 Oct, 20:52 UTC) carries his own summary — "three
things in my words" — and the transcript
`Bob _ Rajesh transcript_2026-10-22_16.49.28.txt` as an attachment. As
with 28 Aug, **the transcript is not committed to this repo** (verbatim
conversation; the repo is headed for public release); timestamps below
refer to it. Bob asked Raj to put the three things in his own words —
this log quotes that email as the authoritative record.

**What the call answered.** The last open methodology decision (4a,
DEL-52a) plus two confirmations of Phase 6 scope. Every question the
batched process asked is now closed.

---

## Summary table

| # | Question | Decision | Lands as | Ticket |
|---|---|---|---|---|
| 1 | Which PSI the figures report (4a, DEL-52a) | **Planned-relative scale** — every type mean ÷ Planned mean; min-max kept in the appendix | new variant value + reported-outputs generator, then ratified profile | DEL-63 |
| 2 | Distance-variant robustness (DEL-36/38) | **Confirmed — the appendix table is wanted** | band-sweep re-run on ratified rules | DEL-36 |
| 3 | Media reclassification run (DEL-53) | **Confirmed** — appendix table + "nothing changes" sentence | variant profile + run | DEL-53 |

## 1. The reported scale goes planned-relative (4a answered; DEL-52 ✓, DEL-63)

**Raj's words (post-call email):** "relative calculation: make planned
1 by dividing all the numbers with planned colony averages and compute
the index. (leave what we have in the appendix and i am thinking move
the relative to the main paper)"

**On the call (16:35–16:42):** the 0–1 scale "just doesn't tell our
story carefully" — Planned reads 0.04 and everything else less, which
reads as "already low" and raises "you guys have been super aggressive"
questions instead of showing the gap. His fix, checked beforehand with
Claude: one line — **divide every type's average by the planned
average**. His spot numbers against the July 2025 `norm_psi` ×
popdensity means: Planned 1.00, UV ≈ 0.86, JJR ≈ 0.85, Industrial
≈ 0.85, RUAC ≈ 0.49, UAC ≈ 0.39, JJC ≈ 0.03.

**Decision:** the main figure reports rel(type) = mean(psi_i | type) /
mean(psi_i | Planned). The exact current calculation stays in the
appendix ("for the Puritans"); Raj is thinking the relative version
moves to the main paper (placement is his writing call; the calculation
is decided). This replaces the 4a choice as posed (min-max + one
sentence vs Eq. 1 as written) — a third form, and it answers the
"what is 1?" question directly: **1 = the average planned colony.**

**Mechanics (DEL-63):** the DEL-34/57 variant pattern, this time
adopted — a new second-normalization variant value (reference-category
mean), reference rule + oracle pins, committed fixtures byte-identical;
the min-max columns stay for the appendix. Final numbers come from the
ratified run (DEL-32), not the July 2025 baseline.

## 2. Distance variants — the appendix table (DEL-36 confirmed)

**Raj's words:** "varying distances for services create a table by
varying the distances and recomputing the numbers (for appendix)"

On the call (16:43): "somebody raised this question … a year ago … we
played with it, but there's no table to point to. So now we can say,
hey, here's the table. We think what we present is reasonable. If you
are worried about being too aggressive or not, just go here."
Distances "threefold, four times" — not a thousand miles. Bob: the
sweep harness (DEL-55) is built and the band profiles (0/1/5/10 km)
exist; re-run on the ratified rules once DEL-31/32 land. Raj: "if it's a
trivial code change, we should do it" — it is; DEL-36 is a run, not a
project.

## 3. Media reclassification — confirmed (DEL-53)

**Raj's words:** "recompute numbers with the media analysis (new
dataset on settlement classification)"

On the call (16:44–16:47) Raj flagged checking the join key ("you might
want to look at it … is that unique enough?") — **answered 20 Oct**: all
4,342 `USO_AREA_U` IDs match the settlement layer, zero duplicates
(`docs/decisions/2026-10-19-raj-methodology-decisions.md` § 6). His
expectation for the run: "nothing changes … we're only talking about 50
settlements moving here and there" — an expectation the run now
verifies with numbers; the paper gets one sentence plus the appendix
table. NB his "50" differs from the sheet's 105 flagged changes (81
RV→UV) — the measured record stands; reconcile in the appendix
write-up.

## Not discussed (still open)

- `barrier.buffer_m` ratification — propose 5 m (the Oraculum anchor,
  the DEL-48 spec default); flag to Raj in the next update.
- Decision B (data-release posture, DEL-14) and the licence (DEL-56).
- The density-justification methods sentence — not re-raised and not
  vetoed; keep drafting, include in the next update for his explicit OK.

## Timeline

Bob on the call (16:49): "I think we can get it done by this weekend …
or most of it, if not all of it." — the three items, most of the way,
by 25 Oct 2026.

## What this means for the plan

1. **DEL-63** (planned-relative variant) — the one piece of new code on
   the critical path; then
2. **DEL-31** ratified profile (min-max stays on for the appendix; the
   relative scale is the reported headline) → **DEL-32** recalculation →
   **DEL-33** figures;
3. **DEL-61 → DEL-60** (rule_effects guard + splice) — small, land
   alongside so the docs/data regeneration is clean;
4. **DEL-36** distance table + **DEL-53** media run on the ratified
   rules;
5. DEL-52 closed (popdensity 19 Oct; column → planned-relative 22 Oct).
