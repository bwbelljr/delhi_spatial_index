# Oraculum Derivation Worksheet

**STATUS: RATIFIED by Bob on 2026-08-24** — expected values derived by
Claude from Eq. 1–4 of "Making the City Unequal" (April 2026 draft,
pp. 14–16); the anchor subset below was hand-checked with a calculator by
Bob against the manuscript's equations. Eq. 1–2 (min-max and mean) were
checked via the worked index values; Eq. 3's form, the border-sharing
neighbor rule, and the 1/(1+d) decay were confirmed against the text.
Raj's check remains welcome but is not required for the oracle's authority.

**Scope of hand ratification (the ~15-minute pass):** the
`ideal`/`baseline`/`pop` configuration for all seven settlements
(clinic/school/road below; the four singleton services follow the same
two-line pattern), plus the four worked extras at the end. Everything else
in `tests/fixtures/oraculum/expected_values.csv` is machine-cross-checked by
the independent reference implementation (`tests/reference_impl.py`), whose
correctness these anchors ratify. The e2e CLI leg is three-way-verified only
for the `excl_rv_only` configuration (the real pipeline's hardcoded filter).

Map: `oraculum_city.png`. Decays: 1 km → 1/2; 1.5 km → 0.4;
√2 km → √2−1 ≈ 0.414214; (√5)/2 km → ≈ 0.472136.

**Decay convention (confirmed against Eq. 3):** decay = 1/(1 + d_ij) with
d_ij the centroid-to-centroid distance **in kilometres**. Both production
(`calc_nbr_dist`, divides metres by 1000) and the reference implementation
(`_centroid_km`) use km. The choice is not scale-free — in metres a 1 km
neighbour would lend ~0.001 of its services instead of ½.
Checked 24 Aug 2026: the manuscript does NOT state the unit (p. 15 says
only "the distance from the centroid ... to the centroid"). Recorded as
divergence #7 — a silence, not a contradiction (`suggested-fixes-memo.md`
§7); one sentence needed in the methods.

Ideal neighbor lists (A–D severed by canal, both directions):
A:[B,E] B:[A,C,RV,E] C:[B,E,IND] RV:[B] D:[E] E:[A,B,C,D,IND] IND:[C,E]

## Clinics (counts: A 2, B 1, E 1, RV 2) — Eq. 3, popsize

| i | arithmetic | PCEN |
|---|-----------|------|
| A | (2 + 1·½ [B] + 1·(√2−1) [E]) / 100 = 2.914214/100 | 0.02914214 |
| B | (1 + 2·½ [A] + 0 [C] + 2·½ [RV] + 1·½ [E]) / 200 = 3.5/200 | 0.01750000 |
| C | (0 + 1·½ [B] + 1·(√2−1) [E] + 0 [IND]) / 400 = 0.914214/400 | 0.00228553 |
| RV | (2 + 1·½ [B]) / 100 = 2.5/100 | 0.02500000 |
| D | (0 + 1·0.4 [E]) / 100 | 0.00400000 |
| E | (1 + 2·(√2−1) [A] + 1·½ [B] + 0 + 0 + 0) / 300 = 2.328427/300 | 0.00776142 |
| IND | (0 + 0 [C] + 1·0.4 [E]) / 10 | 0.04000000 |

Eq. 2 anchors: min = C (0.00228553), max = IND (0.04) — both unique.
Example index: A_idx = (0.02914214 − 0.00228553)/(0.04 − 0.00228553)
= 0.02685661/0.03771447 = **0.71210346** (CSV: 0.7121034578830464 — computed from unrounded PCENs; the displayed 8-digit inputs give 0.71210360, so check to ~6 decimals).

## Schools (A 1, D 1, E 1) — Eq. 3, popsize

| i | arithmetic | PCEN |
|---|-----------|------|
| A | (1 + 1·(√2−1) [E]) / 100 = √2/100 | 0.01414214 |
| B | (0 + 1·½ [A] + 1·½ [E]) / 200 | 0.00500000 |
| C | (0 + 1·(√2−1) [E]) / 400 | 0.00103553 |
| RV | 0 / 100 (B has no school) | 0 |
| D | (1 + 1·0.4 [E]) / 100 | 0.01400000 |
| E | (1 + 1·(√2−1) [A] + 1·0.4 [D]) / 300 = 1.814214/300 | 0.00604738 |
| IND | (0 + 1·0.4 [E]) / 10 | 0.04000000 |

min = RV (0), max = IND (0.04) — unique. Note the deliberate near-tie A vs
D (0.014142 vs 0.014): E's school at different decay is what separates them.

## Roads — Eq. 4 literally (NO neighbor term), lengths A 0.75 km, E 0.75 km

pop: A = 0.75/100 = **0.0075**; E = 0.75/300 = **0.0025**;
B = C = RV = D = IND = **0 exactly** (tied minimum — recorded ground truth;
Eq. 4 gives every road-less settlement zero).
popdensity: A = 0.0075; E = 0.75/150 = 0.005.
(The production code decays roads like Eq. 3 — a documented divergence, not
part of this ideal derivation; see the memo.)

## Singleton services (bank@A, police@B, ration@D, transport@E)

Pattern: PCEN_i = (own + [X ∈ nbrs(i)] · decay) / pop_i. E.g. police (X=B):
B = 1/200 = 0.005; A = 1·½/100 = 0.005 (**tied argmax — recorded**);
C = 1·½/400 = 0.00125; RV = 1·½/100 = 0.005 (RV's list is [B], so it ties
too — a three-way tie A/B/RV at 0.005, all recorded in the CSV; ties
outside clinics/schools are expected ground truth). E = 1·½/300 = 0.00166667; D = 0; IND = 0.

## Worked extras (complete the anchor subset)

1. **Exclusion delta (B, ideal, excl_removed, pop):** RV and IND removed
   before neighbor computation → B's list [A,C,E]:
   (1 + 2·½ + 0 + 1·½)/200 = 2.5/200 = **0.0125** (vs 0.0175 baseline —
   the RV contribution effect, −0.005).
2. **Renormalization delta (A clinic_idx, ideal, excl_ind_removed, pop):**
   PCENs unchanged (IND serviceless); max moves from IND (0.04) to A
   (0.02914214); min still C. A_idx = (0.02914214−0.00228553)/
   (0.02914214−0.00228553) = **1.0** exactly (was 0.71210346) — anchor
   movement with zero numerator change. This delta is denominator-INVARIANT
   because A, C, IND all have area 1.0 km².
3. **Popdensity coverage (E clinic, ideal, baseline):**
   popsize 2.328427/300 = **0.00776142**; popdensity divides by
   pop/area = 300/2 = 150 → 2.328427/150 = **0.01552285**.
4. **Road Eq. 4 value (A, pop):** 0.75/100 = **0.0075** (worked above).

## Machine-checked remainder

All other configurations (code rule-set incl. directed barrier asymmetry
and decayed roads; excl_contributing/excl_rv_only; norm_psi; popdensity
tables; the four singleton services' full tables; the divergence exhibit
deltas) are asserted equal, to 1e-12, between the reference implementation
and the production code by `uv run pytest` — their authority derives from
these hand anchors plus the reviewed independence of the reference
implementation.
