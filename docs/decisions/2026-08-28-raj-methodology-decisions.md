# Methodology decisions — Bob/Raj call, 28 Aug 2026

**Source.** Zoom call "Bob/Raj delhi", Fri 28 Aug 2026, 14:04–14:44 ET.
Raj's email "meeting notes" (28 Aug 2026, 14:47 ET, to Bob)
carries his summary inline and the transcript as the attachment
`Bob_Raj delhi transcript_2026-08-28_14.44.26.txt`. The transcript is
**not** committed to this repo (verbatim conversation; the repo is headed
for public release) — timestamps below refer to it. Bob's rulings on the
open sub-questions were made on 5 Sep 2026 while writing this log.

**What the call answered.** The memo package sent to Raj on 24 Aug 2026
(`docs/oracle/suggested-fixes-memo.md`, `exclusion-semantics-memo.md`,
`rv-exclusion-decision-memo.md`). Every DECISION/CONFIRM item in that memo
now has an entry here. Item numbers below are this log's, not the memo's;
each entry names the memo section and the DEL ticket.

**How decisions land.** Since Phase 3 every methodology choice is a value in
a YAML profile (`docs/methodology-config.md`). Decisions marked *config* are
one line in the ratified profile that DEL-31 writes and DEL-32 recalculates
with. Decisions marked *code* need a reference rule, fixture pins and a
production implementation first (config doc § 5) — that is cycle 3E
(DEL-48, DEL-20).

---

## Summary table

| # | Question | Decision | Lands as | Ticket |
|---|---|---|---|---|
| 1 | Excluded types: lend services or vanish? (Open Decision A.1) | **Lend** — semantics (a); no PSI of their own | config `exclusion.absent_neighbor: contributes` | DEL-13 ✓ |
| 1 | Min-max universe (A.2) | **Reported types only** (today's behaviour) | no change; methods sentence | DEL-13 ✓ |
| 1 | Which types are dropped | **RV, Industrial, Other** | config `exclusion.types: [RV, Industrial, Other]` | DEL-28 ✓ |
| 2 | Roads: neighbour term or not? (memo § 3) | **Own settlement only, Eq. 4 as written** — measured 5 Sep: JJC mean PSI −13 % (pop) / −10 % (density), no ordering flip, decision stands | config `roads: eq4_own_only` | DEL-22 → DEL-31; DEL-49 ✓ |
| 3 | Adjacency: bbox or shared border? (memo § 1) | **Shared border** | config `adjacency.rule: touch` | DEL-19 |
| 4 | Barrier rule (memo § 2) | **Partial weighting** — discount by share of shared boundary covered | **code** | DEL-48 |
| 5 | Overlapping colonies: a service in the overlap (memo § 6) | **Counts for every colony containing it** (today) + **not lent again via the neighbour term** (Bob's rule, to confirm) | first: no change; second: **code** | DEL-20 |
| 6 | Settlement categories: collapse 10 → 4/5? (DEL-29) | **No collapse now.** Keep all types, drop three; revisit if reviewers push back | config (identity mapping) | DEL-29 parked |
| 7 | `norm_psi`: which column do the figures report? (memo § 4) | **Measured 5 Sep: the figures report `norm_psi`** (8/8 bars); Bob's `false` default withdrawn — Raj chooses keep-and-document vs Eq. 1 as written | config `second_normalization` per Raj | DEL-52 ✓ |
| 8 | Popdensity denominator (memo) | **Measured 5 Sep: the figures use popdensity** (popsize 0/8); Bob's drop default withdrawn — Raj chooses keep-and-add-equation vs per-population Eq. 3 | config `outputs.denominators` per Raj | DEL-52 ✓ |
| 9 | Decay form / distance unit (memo § 7, DEL-37) | **Keep 1/(1+d), km, centroid**; sweep later; Raj wants weights spread away from zero | no change | DEL-37 |
| 10 | Oraculum reproducibility appendix (DEL-46) | **Yes** — release the fixture city + hand verification as an appendix | plan | DEL-46 |
| 11 | Data-release posture (Open Decision B) | **Not discussed**; floor is now code + fixtures | pending | DEL-14 |

**Punt list (Raj's words: "do not lose"):** category collapse (#6); a
steeper decay for roads than clinics (moot — roads have no neighbour term);
alternative decay forms are Phase 6 (#9).

---

## 1. Exclusion semantics — Open Decision A (DEL-13, DEL-28)

**Transcript 14:11–14:15, restated 14:36.** Raj: "we still count the
clinics, because the clinics are accessible … we're not erasing the clinic's
access … just because we made an analytical decision about a categorization,
it just doesn't make sense to remove the physical elements." Bob's
restatement, confirmed by Raj: dropped types contribute to the settlements
that get a service index; they themselves get none.

**Sub-decisions.**

- **A.1 Neighbour treatment: semantics (a).** Excluded settlements still
  lend their services to adjacent reported settlements, with decay and the
  barrier rule applying as for any neighbour. Config:
  `exclusion.absent_neighbor: contributes` (today `swallowed`) with
  `exclusion.stage: post_neighbors` (already today's value — neighbours are
  built on the full universe). On Oraculum this is B's clinic PCEN 0.0175
  instead of 0.0125 (`exclusion-semantics-memo.md`). Note: this was
  unimplementable in the 2021 code (silent `except: pass`, fixed in 3A,
  DEL-21).
- **A.2 Min-max universe: reported types only.** Raj (14:34): "It can't be
  the minimum. The minimum has to be the ones which are in contention."
  This is today's behaviour, so the reserved `exclusion.minmax_universe`
  key stays unneeded. Raj has the methods wording on his own list
  ("normalization runs over reported settlement types only").
- **A.3 Descriptive tables.** Dropped types also leave the population-share
  and count tables. Raj's writing; noted so it is restated deliberately.
- **Which types.** Three: rural villages (RV), industrial areas
  (Industrial), and Other. **UV (urban villages, 138) and SDA (86) stay in**
  as reported types — implied by "keep all types, drop the 3", never named
  on the call; confirm with Raj (batched reply).

**Bob's ruling (5 Sep 2026) — output treatment.** Dropped settlements do
**not** appear in the PSI outputs, unchanged from today. Maps that show all
of Delhi start from the settlement layer (all 4,357 polygons) and left-join
the PSI file; unscored polygons draw grey with a "not scored" legend entry.
Rejected alternative: keep every row with blank PSI columns — one file with
two kinds of rows, every consumer must filter, and it contradicts "no PSI of
their own".

## 2. Roads — no neighbour term (memo § 3; DEL-22, DEL-49)

**Transcript 14:16–14:28.** Raj's argument: road access is about mobility —
an ambulance cannot use a road in the next colony; "if you don't have a road
to the settlement, you don't have a road." Bob pushed back (a household at
the border; why not clinics too); Raj: "you can argue very logically both
ways … you just have to explain it." Raj (14:19): "let's just keep it at
that" — **but the premise was inverted**: Bob had said the code has no
neighbour-discounted access to roads. In fact the **code decays roads like
clinics** (`roads: decayed`, the July 2025 numbers) and it is the **paper's
Eq. 4** that has no neighbour term (own length / own population, min-maxed;
p. 15 and fn. 8). What Raj endorsed on the merits is therefore the paper's
version, and it is a **change from the published numbers**.

**Decision (Bob, 5 Sep 2026, recorded as final):** each colony counts only
the roads inside its own boundary — `roads: eq4_own_only`. "Binary" in
Raj's notes means "you have roads in your settlement or you don't", not a
0/1 indicator; Eq. 4 stays continuous. The `decayed` value is **kept in the
code**: `code-2025` must keep reproducing July 2025 byte-for-byte, and the
diff between the two is the robustness variant reviewers may ask for.

**Before the recalculation (DEL-49):** (1) how many JJCs have a road inside
vs only in a touching neighbour vs neither, by type; (2) the one-factor
effect — `code-2025` with only `roads` changed, diffed by type. Both go to
Raj with the correction. The decision stands unless the numbers move him.

**Measured (5 Sep 2026, DEL-49; `docs/data/roads_access.md`).** Of 764
JJCs, 17 (2 %) contain a major road, 645 (84 %) reach one only through a
touching neighbour, 102 (13 %) have neither; planned colonies are the
mirror image (551 inside, 323 via a neighbour, 90 neither). Under
own-only roads, 422 of the 749 reported JJCs fall to a road index of
exactly zero (1,682 of 4,131 settlements overall). The JJC mean PSI falls
13 % under the population denominator and 10 % under population density;
the JJC-versus-Planned ordering does not flip under either. Against the
reopen rule written before the run (a JJC fall over 20 %, or an ordering
flip), **the decision stands**. The sentence for Raj's footnote: under
Eq. 4 as written the road index measures whether a settlement has a major
road of its own, and most JJCs do not.

**Raj's writing:** the roads footnote (mobility argument; settlement-level
average, not household-level) — on his list.

## 3. Adjacency — shared border (memo § 1; DEL-19, DEL-50)

**Transcript 14:33.** Bob: "I think the paper says your neighbors are those
where you share a border." Raj: "Yeah. Correct." The empirical
sub-question — pairs touching only at a corner point — was expected to be
"zero or near zero".

**Measured (5 Sep 2026, DEL-50; `docs/data/layer_pathologies.md`).** It is
not: 656 pairs, involving 955 settlements, meet only at a point. The
likely reading — an interpretation, not something the measurement verifies
— is that these are the diagonal polygons of four-way corners on a
tessellated layer; T-junction vertices and digitisation near-misses would
count the same way. Under `touch` none of them
are neighbours; under a 0 km distance band all of them would be. **Bob's
ruling: a corner is not a border — `touch` stands.** For Raj: one FYI
sentence, so the methods can say "sharing a border of positive length";
the 20 settlements isolated under `touch` already reflect the rule.

**Decision:** fix, not ratify. `adjacency.rule: touch` (positive shared
length; corner-only contact is not a border; overlapping polygons are
neighbours). The messy-city pins of bbox behaviour flip when the ratified
profile lands; bbox becomes the Phase 6 comparison variant (DEL-39).
Consequences to state in the methods: 20 settlements are isolated under
`touch` (6 under bbox); `preprocess` must be rerun (the artifact is stamped
with the rule). **DEL-50** counts corner-only pairs before the batched reply.

## 4. Barriers — partial weighting (memo § 2; DEL-48, DEL-51)

**Transcript 14:29–14:32.** Raj: "if there's a barrier that only covers
half … you just count it as half instead of completely binary … that makes
sense." On the proposal (weight by the share of the shared boundary NOT
covered): "if there's like 30% left, let's not make it a binary thing …
they can get around the canal. I think that makes complete sense, and we
should just do it." … "let's just do that the way you have it."

**Decision:** `w_ij = 1 − L_blocked(i,j) / L_shared(i,j)`, linear,
symmetric by construction; contribution of j to i = `w_ij · decay(d_ij) ·
services_j`. Full coverage recovers the paper's pairwise severing; the
code's global/asymmetric flag is retired from the ratified profile.

**Bob's rulings (5 Sep 2026):**
- **Sequence:** build before DEL-32 (595 of 4,357 settlements are
  barrier-flagged today; recalculating twice means two rounds of figures).
- **Oraculum:** ~~the canal covers 90% of the A–D edge, which would make
  w_AD = 0.1 and move the hand-ratified anchors. Redraw the canal to cover
  the full edge; add a partial-coverage pair to the messy city.~~
  **Superseded 5 Sep 2026:** the rule ships as a `tests/variants.py` VARIANT,
  not as a change to the `ideal`/`code` rule-sets, so no existing expected
  value moves and NOTHING was redrawn. The canal where it is *is* the
  fractional anchor: **w_AD = 0.08** at a 5 m buffer (460 of the 500 m edge
  blocked; the memo's 0.1 is the buffer-free limit). Adding a messy barrier
  would in fact have violated the byte-identity condition.
- **Deferred to the DEL-48 spec brainstorm — settled there:** the buffer is
  `barrier.buffer_m`, a DISTANCE with round caps, 5 m, required by the rule
  and strictly > 0 (a zero-width buffer is empty in shapely and would make
  every weight 1); the shared boundary of an overlapping pair is the
  boundary of the intersection polygon; a point crossing blocks zero length
  and therefore severs nothing (w = 1).

**Measured (6 Sep 2026, DEL-48; `docs/data/rule_effects.md`).** Two findings,
and the second is the one for Raj.

1. **The partial rule severs 57× fewer links than today's.** Of 29,258
   directed links that exist before any barrier rule runs, today's
   `global_asymmetric` keeps 21,211 (severing 8,047, 27.5 %) while
   `partial_weighted` keeps 29,118 (severing 140, 0.5 %). Of the links it
   keeps, 1,636 are partly blocked with a median retained weight of 0.94.
   Barriers in Delhi mostly clip shared boundaries rather than closing them
   — though 140 boundaries ARE fully covered, so "none" would be wrong.
2. **Most of the effect is not the partial weighting — it is retiring the
   global flag**, which deletes every link INTO a flagged settlement whether
   or not a barrier lies between that pair. 3,155 of 4,357 settlements
   (72 %) get a different neighbour list. Under the population-density
   denominator the paper's figures use, every reported type gains except
   SDA: Planned +8.8 %, JJC +199 %, UAC +67 %, RUAC +61 %, UV +44 %,
   JJR +27 %, SDA −16 %.
3. **Two consequences for the headline, and the second is a reordering.**
   The gap narrows: JJC stays lowest and Planned-over-JJC falls from about
   33× to about 12×. But **urban villages overtake planned colonies** under
   the new rule (0.0549 against 0.0482, where today Planned leads UV 0.0443
   to 0.0381) — a change of RANK, not just of degree, in the exact variant
   the figures are drawn from. That is the one result here that could change
   a sentence in the paper rather than a number, and it belongs in the
   batched reply. Neither consequence is a reason on its own to abandon the
   decision; both are Raj's to weigh. Rebuilding the artifact cost 14.4 min.

**Raj's question (14:30–14:31):** where did the barrier layers come from —
city data or drawn by Bijoy? Bob to check (**DEL-51**). Raj's writing: the
partial-barrier footnote — on his list.

**Measured (5 Sep 2026, DEL-51; `docs/data/barriers.md`).** The three
layers hold 43 canal features (33.9 km), 5,356 railway features (752 km)
and 616 major-drain features (403 km); they flag 28, 240 and 390
settlements respectively, 595 under any layer — the figure used since the
memo. The attribute schemas (canal names with ground elevation, railway
zone, drain name with maintaining agency, assembly constituency and
district) are those of an official GIS source, not of hand-digitised
lines; the ESRI sidecars show the canal and railway files were created in
ArcGIS Pro on 2 Aug 2020 with geometry-repair runs; the extra copies in the
data folder are re-exports of the same clipped data. The paper's "manually
marked" (pp. 15–16) therefore most plausibly describes selecting features
for the study area, not drawing them. **No claim about which agency** is
made. Open for Bijoy (batched reply): which agency's layer, and what
"manually marked" covered.

## 5. Overlapping colonies (memo § 6; DEL-20)

**Transcript 14:37–14:38.** Raj on a service inside the overlap of two
colonies: "if it's a boundary to you, it doesn't matter if it's a boundary
to somebody else. We're gonna count the overlap." Bob: a bank in the
overlap counts for A and for B. Raj: "Yes." He did not know the provenance
of the 4,069 overlapping pairs (digitisation vs contested boundaries). The
memo's preferred option — clean the layer — is off the table.

**Ratified (Raj):** a service inside k overlapping colonies counts directly
for each of the k. This is today's behaviour on both sides (production
`intersects`, reference `within`; pinned by
`test_the_overlap_clinic_is_counted_for_both_owners`).

**Added (Bob, 5 Sep 2026; to confirm with Raj):** a neighbour lends only
the services **not already inside the receiving settlement**:

    PCEN_i = ( S_i + Σ_j w(d_ij) · | S_j \ S_i | ) / pop_i

Reason: under `touch` two overlapping colonies are also neighbours, so an
overlap clinic would otherwise appear four times across the pair (twice at
full weight, twice decayed) against two for a clinic sitting cleanly on one
side. A service counts for a settlement at most once — as its own or as a
neighbour's, never both. Clean pairs are unaffected; Raj's ruling is kept.
This exists today under bbox too — it is not created by `touch`. **Code**:
per-pair adjusted counts; the messy city already holds the fixture case
(clinic strictly inside O1 ∩ O2); Oraculum has no overlaps. DEL-20 stays
open with this scope and ships in cycle 3E with DEL-48. Quantify against
`code-2025` one-factor, as for roads. Methods sentence (Raj): overlap
services are counted for each colony, with the count
(`docs/data/layer_pathologies.md`).

**Measured (6 Sep 2026, DEL-20; `docs/data/rule_effects.md`).** The double
count is real, and small. 758 ordered pairs share at least one unit of a
service (194 banks, 156 ration shops, 248 roads, the rest smaller); 2,427
settlements have an overlapping neighbour, and 385 of them have a PCEN that
actually moves — about one settlement in eleven, each of which was counting
some service twice, once as its own and once decayed as a neighbour's.
Nothing rose and nothing outside the overlapping set moved, which is the
containment the rule promises.

**It does not move the headline.** Under the density denominator the
figures use: Planned 0.0304103 → 0.0304087, JJC 0.000914278 → 0.000911273,
largest mover UAC at −4.9 %. No ordering changes. So this is the opposite
profile from the barrier rule — small in effect, and a correction of a real
arithmetic error rather than a change of method.

**Still Bob's proposal, not Raj's decision.** Both shipped profiles carry
`whole` (today's behaviour); `outside_receiver` is available and proven.
The question for Raj now comes with its price attached: about 5 % of one
settlement type's mean, and nothing of the argument.

## 6. Settlement categories — keep all, drop three (DEL-28/29/30/31)

**Transcript 14:40–14:42.** Raj: "I spent a lot of time actually thinking
about it … for now we should just stick with dropping the 3 … keeping the 7
or 8 levels … do it in the writing … if people push back, we can do that
calculation again." Patrick prefers keeping the types separate rather than
a 2×2.

**Decision:** identity mapping (`uso-10`, as today) with
`exclusion.types: [RV, Industrial, Other]`; seven reported types: Planned,
UAC, RUAC, JJC, JJR, UV, SDA. DEL-28 decided; **DEL-29 parked** (not
rejected — revisit at revision; the `urban-5` worked example in
`docs/methodology-config.md` § 2 is the recipe if reviewers ask); DEL-30's
breakdown charts show seven types, not five; **DEL-31 shrinks** to the
exclusion line inside the ratified profile.

**New (14:42–14:44), DEL-53:** two students' media-based reclassification
(~80% RV → UV) becomes an appendix robustness run. Raj to send the list
(IDs, old type, new type; it is in a share, not the Urban Spatial
Observatory folder). Not received as of 5 Sep 2026.

## 7. `norm_psi` — MEASURED: the figures report `norm_psi` (memo § 4; DEL-52)

**Transcript 14:35–14:36.** "No, this, I don't know the answer to that.
This, 4 and 5, I don't know." Fallback per the memo: Bob compares the April
2026 draft's figure values against `unnorm_psi` and `norm_psi` in the July
2025 outputs; Raj confirms. Bob's recommendation at the time: report Eq. 1
as written (`second_normalization: false`).

**Measured (5 Sep 2026, DEL-52; `docs/data/psi_columns.md`).** Figure 4's
eight bars match the per-type means of `norm_psi` under the
population-density denominator on 8 of 8 within ±0.002 (max gap 0.0006);
`unnorm_psi` matches 1 of 8. The paper's headline figure therefore reports
the SECOND-normalised column, the one the methods never mention. **Bob's
proposed default of `second_normalization: false` is withdrawn**: switching
it off would silently move every bar (the Planned mean falls from 0.044 to
0.030, the JJC mean from 0.0013 to 0.0009). The choice for Raj: keep
`norm_psi` as the reported PSI and add one sentence after Eq. 1 ("the mean
is then min-max scaled across settlements"), or switch to Eq. 1 as written
and let every bar move. Both columns stay in the config; the ratified
profile's `second_normalization` takes his answer.

## 8. Popdensity denominator — MEASURED: the figures use it (DEL-52)

Not discussed on the call. Bob's proposed default at the time: drop from
the reported results, keep the config value.

**Measured (5 Sep 2026, DEL-52; `docs/data/psi_columns.md`).** Figure 4's
y-axis reads "per person per square kilometer", and the measurement agrees:
both population-size candidates match 0 of 8 bars; the population-density
`norm_psi` matches 8 of 8. **The proposed default of dropping popdensity is
withdrawn** — the paper's headline figure IS the popdensity variant. The
choice for Raj: keep popdensity as the reported denominator and add its
equation to the methods (Eq. 3 with Population_i/Area_i in place of
Population_i), or switch the figures to the per-population Eq. 3 the
manuscript prints. Both denominators stay in the config; the ratified
profile's `outputs.denominators` takes his answer. Note the two withdrawals
are independent: the column and the denominator are separate choices.

## 9. Decay form and distance unit (memo § 7; DEL-37)

**Transcript 14:38–14:40.** Bob: the decay is configurable; run the
alternatives once the code is done. Raj: "I agree … we want to get the
weights a little bit more up from zero … anything to make it spread a
little bit more is good." A steeper decay for roads than clinics (14:27):
"not do extra work at this stage … ship it."

**Decision:** ratified profile keeps
`decay: {form: inverse_linear, distance: centroid, distance_unit: km}`.
DEL-37's sweep gets Raj's steer (forms that spread weights upward:
`inverse_power` with exponent < 1, `exponential` with a scale of a few km,
`boundary` distance). FYI for Raj's methods text: d is in **kilometres**,
stated next to Eq. 3 (the manuscript is silent).

## 10. Oraculum reproducibility appendix (DEL-46)

**Transcript 14:08.** Bob: release the fake city and the hand verification
in the appendix. Raj: "Beautiful." His notes: "Add to the paper's appendix
plan." **DEL-46 moves from optional to planned.** Whether the messy city is
mentioned alongside is a writing call.

## 11. Data-release posture — not discussed (DEL-14)

The appendix decision sets the floor at code + fixtures. Derived
per-settlement outputs or the full archive still needs the group; Bob's
proposed default for the batched reply: code + derived outputs.

## 12. Also open — bug-audit item 6 (not on the call)

`index.minmax` has no `hi == lo` guard; a constant per-service column
divides 0/0 (latent on real layers; raised as an error under `-W error`).
Choice: raise, or 0.0 as the reference does. Bob's default: **raise** with
a clear message — a constant column on real data means something upstream
is wrong. Lands with the 3E cycle.

---

## What goes in the batched reply to Raj

1. The roads correction (#2) with the DEL-49 numbers: 17 of 764 JJCs have
   a major road inside, 645 reach one only via a neighbour; own-only
   roads zeroes the road index of 422 of 749 reported JJCs and moves the
   JJC mean PSI −13 % (pop) / −10 % (density) without an ordering flip —
   the decision stands, and that is the footnote sentence.
2. The overlap neighbour rule (#5) as Bob's proposal.
3. Confirm UV and SDA stay in (#1, #6).
4. **DEL-52, two choices for Raj (#7, #8):** Figure 4 was drawn from
   `norm_psi` under population density (8/8 bars). Column: keep `norm_psi`
   and add the second min-max sentence after Eq. 1, or report Eq. 1 as
   written and let every bar move. Denominator: keep popdensity and add
   its equation, or switch to per-population Eq. 3. Bob's earlier defaults
   (drop both) are withdrawn.
5. Decision B default (#11).
6. Nudge: the reclassification list (#6, DEL-53).
7. Barrier provenance (#4): schemas of an official GIS source, ArcGIS Pro
   files of 2 Aug 2020, root copies are re-exports; the question for Bijoy
   is which agency, and what "manually marked" covered.
8. FYI methods sentences: d in km; min-max over reported types; overlap
   services counted for each colony (with the count); 20 settlements
   isolated under shared-border adjacency; 656 pairs touch only at a
   corner and are not neighbours under the rule (§ 3, DEL-50).

## What this means for the plan

- **Next code cycle (3E):** DEL-48 partial barriers + DEL-20 overlap
  neighbour rule + bug-audit item 6 — brainstorm → spec → /ship.
- **Measurements** (no code): DEL-49, DEL-50, DEL-51, DEL-52 — **done
  5 Sep 2026**, `docs/data/`.
- **Then the ratified profile** (DEL-31): `adjacency.rule: touch`;
  `barrier.rule: partial_weighted`; `roads: eq4_own_only`;
  `exclusion: {types: [RV, Industrial, Other], stage: post_neighbors,
  absent_neighbor: contributes}`; `second_normalization` and
  `outputs.denominators` per Raj's two DEL-52 answers (the figures used
  `true` and popdensity); everything else as `code-2025`.
- **Then DEL-32** recalculation and DEL-33 figures (left join onto the
  settlement layer).
- **Phase 6** gains DEL-53; DEL-37 carries Raj's steer; DEL-39's main text
  is `touch`, the variant is bbox.
