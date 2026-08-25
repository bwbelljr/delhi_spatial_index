# Suggested fixes for the oracle's manuscript-vs-code divergences

*Companion to `exclusion-semantics-memo.md`. That memo was written for Raj
deliberately without recommendations. This one is Bob's working position on
what the code SHOULD do, item by item, so the methodology conversation with
Raj starts from a proposal rather than a blank page. Every item here is
tracked on the Jira DEL board; nothing is implemented until the owner of the
relevant decision agrees. Numbering follows the six divergences in the
oracle spec (`docs/superpowers/specs/2026-08-17-phase2-oracle-design.md`,
"The two rule-sets").*

**Status: DRAFT — Bob's position, not yet discussed with Raj.**

**How to read the `Raj:` line under each heading.**
- **DECISION** — Raj has to choose; Bob has no default or the default is
  weak. Nothing downstream moves until he answers.
- **CONFIRM** — Bob has a proposed answer and will proceed with it unless
  Raj objects. A one-word reply is enough.
- **FYI** — no input needed; recorded so the methods text can be updated.

## Guiding principle

The paper's accessibility model is *local*: a settlement's PCEN counts its
own services plus decayed services from settlements it shares a border with,
minus pairs a physical barrier actually separates. Wherever the code
deviates from that, the default proposal is to fix the code to match the
paper and re-run, rather than rewrite the methods to match the code. The
exceptions (roads, `norm_psi`, popdensity) are places where the code
implements an idea the paper never states, and those need Raj to say which
version the published figures used.

---

## 2. Barrier rule — global/asymmetric → pairwise/edge-based  **(lead item)**

> **Raj: DECISION ×2 + CONFIRM.** (i) Confirm the paper means pairwise
> severing (a barrier only separates the two settlements it runs between),
> not the code's "flagged settlement is invisible to all neighbors".
> (ii) *Decide* whether partial barriers should weight contributions by the
> unblocked share of the shared boundary (Bob's proposal) or stay binary.
> (iii) *Decide* whether a barrier that merely crosses a boundary at a
> point should sever nothing (consequence of the weighting).

**What the code does.** `barrier_intersection` sets a per-*polygon* flag
`barrier=True` on any colony whose geometry intersects a barrier line. The
neighbor loop (`spatial_index_utils.py:484–490`) then asks only "is the
*other* polygon flagged?": a flagged settlement is removed from every other
settlement's neighbor list, on every side, while its own list is built
normally. In Oraculum the canal on the A–D edge means B and E stop counting
A's services and E stops counting D's — even though no barrier lies between
E and D or between B and A — while A and D keep counting everyone.

**What the paper says** (p. 15–16): "there are a few special cases when
the spatial boundary sharing may not necessarily allow individuals to walk
over to the other settlement. Take for example, a river running through
the settlements, or a railroad track that runs between the settlements. In
cases like these, we manually marked areas that had river or railroad
tracks and then ensured that we don't count these services." The framing
is per-pair ("walk over to *the other* settlement") — a barrier between
two settlements stops *those two* from counting each other. Nothing in the
text suggests a barrier-touching settlement becomes invisible to
neighbors on its other sides. The sentence is loose enough, though, that
the global rule is not literally contradicted — which is why this needs
Raj's confirmation rather than a unilateral fix.

**Proposed rule.** A barrier removes j from i's neighbor list **only if the
barrier crosses the shared boundary of i and j**. Services in a neighboring
settlement that does *not* share a barrier with the current polygon are still
counted. The rule is symmetric by construction (the shared boundary is the
same object from either side), so the A→E / E→A asymmetry disappears.

Reference implementation already exists: `tests/reference_impl.py::apply_barrier`
with `rule="pair"` — for each candidate pair, `shared = geom_i ∩ geom_j`;
drop the pair iff any barrier intersects `shared`.

**Refinement — partial barriers (Bob, 24 Aug).** A barrier often blocks
only part of a shared boundary. Rather than a binary sever/keep, weight the
neighbor's contribution by the *unblocked fraction* of the shared boundary:

    w_ij = 1 − L_blocked(i,j) / L_shared(i,j)
    contribution of j to i's PCEN = w_ij · decay(d_ij) · services_j

where `L_shared` is the length of `boundary(geom_i) ∩ boundary(geom_j)` and
`L_blocked` is the length of that shared boundary lying within a small
buffer of any barrier line. Two rectangles whose shared edge is half
covered by a canal give w = ½: each counts half the decayed services it
would with an open border. w = 1 recovers the no-barrier case, w = 0 the
fully-severed case, so the binary pairwise rule above is the special case
where the barrier covers the whole edge. The weight is symmetric (w_ij =
w_ji) because the shared boundary and the barrier are the same objects
from either side.

Consequences to be aware of:
- A barrier that merely *crosses* the shared boundary at a point (e.g. a
  railway running perpendicular to it) blocks ~0 length and therefore
  ~nothing. That is arguably correct — it does not separate i from j — but
  it is a change from the current code, which would flag both polygons and
  sever them from everyone.
- **This changes the oracle's `ideal` rule-set.** Oraculum's canal covers
  [25, 475] m of the 500 m A–D edge, so under this rule w_AD = 0.1 rather
  than 0: A would count 0.1 × decay × D's school, and D 0.1 × decay × A's
  two clinics. The worksheet currently treats A–D as fully severed. If this
  rule is adopted, the fixture should be regenerated with the canal
  covering the full edge (keeps the hand-anchor arithmetic) and a second
  partial-canal case added to the messy-city tier (DEL-24) to pin the
  fractional weight.
- Whether the fraction should enter linearly (w = unblocked share) or via
  some other function is a modelling choice — linear is the proposal
  absent a reason otherwise.
- The buffer width used to decide "lying along the barrier" is the same
  parameter as in the binary rule and matters more here, since it sets
  L_blocked directly.

**Real-data cautions before adopting verbatim.**
- On the real layer the "shared boundary" of two colonies is often not a
  clean line: 4,050 pairs overlap (so `shared` is a polygon) and some pairs
  touch at a point or have a sliver gap. The pairwise test should use
  `geom_i.intersection(geom_j)` (handles overlap) and probably a small
  buffer (order 1–5 m) so a barrier drawn just off a sliver gap still counts
  as crossing it. The buffer size is a parameter to expose, not to hardcode.
- Barrier layers are drain / railway / canal `Barrier_Clip` files; the
  proposal applies to all three identically.
- 595 of 4,357 real settlements are currently barrier-flagged; under the
  global rule every one of them is invisible to all its neighbors. Expect
  the pairwise fix to *raise* PCEN for many settlements adjacent to flagged
  ones. Worth quantifying (count of pairs restored) as part of the fix.

**Oracle change.** None to the `ideal` rule-set — the pairwise rule is
already its definition. The `code` column's barrier values become
regression values for the *old* behavior and get retired once the fix
lands. Test: `test_border_adjacency_severed_pairwise`.

**Tickets.** DEL-22 (implement), DEL-13 (Raj's methodology sign-off).

---

## 1. Adjacency — bounding box → shared border

> **Raj: CONFIRM.** The paper says "sharing a border"; the code uses
> bounding boxes and invents neighbors citywide. Bob will fix to true
> shared-border adjacency unless told otherwise. One sub-question to
> *confirm*: settlements touching only at a corner point are **not**
> neighbors.

**Code.** `nbrs(i) = { j : geom_j intersects bbox(i) }`. Zero of 4,357
colonies are rectangles; median bbox/polygon area ratio 1.95, p90 3.6. This
invents neighbors citywide and is asymmetric for irregular shapes.

**Paper.** Settlements "sharing a border".

**Proposal.** `nbrs(i) = { j ≠ i : geom_i touches-or-intersects geom_j }`
(`intersects` on the geometries themselves, not the bbox), which is
symmetric. Use the bbox only as the spatial-index prefilter it was
presumably meant to be. Decide separately whether corner-only contact
(`touches` at a single point) counts; the proposal is *no* — require the
intersection to have non-zero length — since a point contact is not a
shared border. Expose as a parameter.

**Oracle change.** None (Oraculum is all rectangles so bbox ≡ border); the
divergence exhibit (`oraculum_divergence.png`, P/Q) becomes a regression
test for the fix.

**Tickets.** DEL-19 (implement), DEL-13, DEL-28 (Raj: whether to treat as a
fix or a methods change).

---

## 3. Roads — neighbor decay absent from Eq. 4

> **Raj: DECISION.** Eq. 4 has no neighbor term; the code decays roads
> like clinics. Which one is intended? Bob has **no default**. See the
> question below — the answer decides whether the code or the equation
> changes.

**Code.** `create_service_length_index` applies Eq. 3-style neighbor decay
to road length (it feeds km of road into `calc_pcen_mobile` exactly as if
it were a clinic count). **Paper** (April 2026 draft, p. 15, checked 24 Aug
2026): Eq. 4 is `RoadsIndex_i = (Length_i/Pop_i − min)/(max − min)` — own
length over own population, min-maxed, **no neighbor term**. The
surrounding text reinforces the distinction: "the roads formula is not
based on point service delivery but polylines of road networks" (p. 14,
fn. 8), and the border-sharing/decay paragraph that follows Eq. 3 speaks
only of "mobile services". So the manuscript deliberately treats roads
differently from point services, and the code does not.

**Question for Raj.** Why does Eq. 4 exclude roads in neighboring
settlements? A road in an adjacent colony is at least as usable as a
clinic there — arguably more so, since it is the means of getting to the
clinic. Was the omission a modelling decision (roads are a property of the
settlement, not a service you travel to), an artefact of adapting the
Johannesburg index (which had no polyline service), or an oversight? The
code's behaviour suggests the implementation assumed decay applies; the
paper's text suggests the authors assumed it doesn't. One of the two has
to move.

**Proposal.** Ask Raj the question above, and which version produced the
published figures. If Eq. 4 is the intended model, remove the decay term (a one-line change, and the
oracle's ideal roads column already encodes it: A 0.0075, E 0.0025, others
0). If the decayed version was intended, add the neighbor sum to Eq. 4 in
the methods text. No default — this is genuinely a modelling choice.

**Tickets.** DEL-22, DEL-13.

## 4. `norm_psi` — second min-max pass absent from Eq. 1

> **Raj: DECISION (small).** Do the paper's figures/tables report the
> Eq. 1 mean directly, or the code's second min-max of that mean
> (`norm_psi`)? If Raj knows, one line answers it; otherwise Bob
> determines it empirically and Raj confirms.

**Proposal.** Determine empirically which column the paper's figures report
(compare figure values to both `unnorm_psi` and `norm_psi` from a rerun of
the original pipeline). Report that one; keep the other as a diagnostic
column but stop calling it PSI. Whichever it is, write it into the methods
in one sentence. **Tickets.** DEL-22, DEL-13, DEL-32.

## Popdensity denominator — no manuscript equation

> **Raj: DECISION (small).** Keep the population-density variant (then
> add its equation to the paper) or drop it from reported results? Either
> is fine for the code.

**Proposal.** Keep as an explicitly labelled code extension
(Population_i / Area_i in Eq. 3) and either add the equation to the paper
or drop the variant from the reported results. Not a bug; needs a sentence,
not a fix. **Tickets.** DEL-22, DEL-13.

---

## 5. Silent `except: pass` in `calc_pcen_mobile`

> **Raj: DECISION — the main one.** When rural villages (and later
> industrial areas) are excluded, do their services still count for
> neighbors (a) or vanish (b)? And should min/max be taken over the
> reported settlements only (current) or over all? See
> `rv-exclusion-decision-memo.md` and its three maps. The code fix itself
> is Bob's regardless.

**Code.** Any neighbor id missing from the frame is silently skipped. This
is why "excluded-but-contributing" (semantics a) degenerates to "fully
removed" (semantics b): dropped settlements vanish from the frame before
their services can be lent.

**Proposal.** Two separate changes.
1. Replace `except: pass` with an explicit lookup that either raises on an
   unknown id or, if the semantics chosen require it, looks the neighbor up
   in the *pre-exclusion* frame. Never silently skip.
2. Make exclusion semantics an explicit parameter: `exclude=(ids,
   contributing: bool)`. Semantics (a) then means "compute PCEN over the
   full universe, drop excluded rows *after* the neighbor sums, before
   min-max"; semantics (b) means "drop before". Both become testable
   against the oracle's `excl_contributing` / `excl_removed` scenarios
   (B clinic PCEN 0.0175 vs 0.0125).

Which semantics the paper wants is Open Decision A (Raj); the code fix is
needed regardless.

**Tickets.** DEL-21 (implement), DEL-13 / DEL-14 (decision).

---

## 6. Service-point membership and overlapping colonies

> **Raj: CONFIRM + one question.** *Confirm* the preference order below
> (clean the overlapping colony polygons first; single-assignment rule as
> fallback). *Question:* does Raj know the provenance of the 4,050
> overlapping colony pairs — digitization artefacts, or genuinely
> contested boundaries? That determines whether option 1 is legitimate.

**Boundary points** (latent: none on the real layer today). **Proposal.**
Count a point for a colony iff `within` (strict), and add a validation
check that fails loudly if any point sits exactly on a boundary, so the
convention gets applied deliberately rather than by accident after a
re-digitization.

**Overlapping colony polygons** (live: 4,050 pairs, ~450 points inside two
or more colonies — bank 232, ration 104, school 53, transport 41, health
18, police 2). This is a data problem first. **Proposal**, in order of
preference:
1. Fix the layer: resolve overlaps in the colony shapefile (the overlaps are
   digitization artefacts, not real shared territory). Quantify the overlap
   areas first; if most are slivers, a topology-clean pass solves it.
2. If the layer can't be fixed in time, assign each point to exactly one
   colony by a deterministic rule (e.g. smallest containing polygon, or
   nearest centroid) and record the rule.
3. Splitting a point fractionally between overlapping colonies is possible
   but hard to explain in the methods; not recommended.

**Tickets.** DEL-20 (implement), DEL-13, DEL-24 (messy-city fixture tier
should include an overlapping pair and a sliver gap so the fix is testable).

---

## 7. Distance unit and decay form — possibly unstated in the manuscript

> **Raj: FYI.** The manuscript never states that d is in kilometres.
> No decision — just a sentence to add next to Eq. 3. Flagging so it
> isn't missed.

**Code.** Decay is 1/(1 + d) with d = centroid distance in **kilometres**
(production `calc_nbr_dist` converts metres → km; the reference
implementation does the same independently). **Paper.** Eq. 3 gives the
decay form `1/(1+d_ij)` and describes d_ij only as "the distance from the
centroid of the current administrative unit to the centroid of the
neighboring administrative unit" (p. 15). **Checked 24 Aug 2026: the unit
is not stated anywhere in the manuscript.**

**Why it matters.** 1/(1 + d) is not scale-free: a neighbour 1 km away
lends ½ of its services when d is in km, ~0.001 when d is in metres, and
~0.62 when d is in miles. The unit is therefore a modelling parameter, not
a formatting detail.

**Proposal.** Keep km (it is what the published figures were computed
with). State it explicitly in the methods next to Eq. 3 — one sentence.
No code change; add `distance_unit_km` as a named constant so the
assumption is visible in the refactor (DEL-16/18).

**Tickets.** DEL-13 (methods sentence), DEL-18 (surface the constant).

---

## What this memo asks of Raj

| # | Decision | Raj's input | Bob's default if no objection |
|---|----------|-------------|-------------------------------|
| 5 | Exclusion semantics (a) vs (b); min/max universe | **DECISION** | fix `except: pass` regardless; Bob leans (a) |
| 3 | Roads: Eq. 4 literal or decayed? | **DECISION** | none — Raj decides |
| 2 | Barrier: pairwise severing; partial-barrier weighting | CONFIRM + **DECISION** | fix code; linear weight |
| 4 | Which PSI column the figures report | DECISION (small) | determine empirically, then document |
| — | popdensity variant: keep or drop | DECISION (small) | keep, label as extension |
| 1 | Adjacency: shared border, corner contact excluded | CONFIRM | fix code |
| 6 | Overlapping colonies | CONFIRM + question on provenance | clean the layer; fallback single-assignment rule |
| 7 | Distance unit of d in Eq. 3 (km) | FYI | keep km; state it in the methods |

Ordered by how much the answer gates: 5 and 3 block Phase 4's
recalculation outright; 2 and 4 change numbers but have workable defaults;
1, 6, 7 are Bob's to execute.

## What changes in the oracle

Nothing in the `ideal` rule-set. Each fix retires one `code`-column
divergence; once all agreed fixes land, `code` should equal `ideal` on
Oraculum and the two-rule-set machinery becomes a regression guard rather
than a divergence catalogue. The messy-city tier (DEL-24) is where the
real-data cautions above (overlaps, slivers, corner contact, barrier
buffer) get their own fixture cases.
