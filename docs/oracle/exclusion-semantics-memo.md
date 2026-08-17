# Memo to Raj: Exclusion semantics, and what the code actually does

*(Machine-verified against the Oraculum oracle; no recommendation is made —
this grounds WORKPLAN "Open decisions A". Figures:
`oraculum_exclusion_variants.png`, `oraculum_city.png`.)*

## The question

When we drop settlement types (RV now; RV + industrial after the
recategorization), do the dropped settlements still LEND their services to
adjacent settlements' accessibility (semantics **a**), or vanish entirely
(semantics **b**)?

## What the mythical city shows (ideal rule, popsize, clinic PCEN for B)

| configuration | B's clinic PCEN | why |
|---------------|-----------------|-----|
| baseline (all 7) | 0.0175 | B counts RV's 2 clinics at decay ½ |
| semantics (a) — RV/IND excluded but contributing | 0.0175 | index rows dropped; services still lend |
| semantics (b) — RV/IND fully removed | 0.0125 | RV's clinics vanish from B's numerator |

## Full per-settlement delta tables (**rule = code**, i.e. what the pipeline does today; both denominators)

Switching from the manuscript's ideal semantics to production behavior for the
same scenarios. Note B's baseline clinic PCEN reads 0.0125 here versus 0.0175
in the ideal-rule table above: the code's global barrier rule strips A from
B's neighbor list, so B loses A's two clinics. Cells are NaN where a scenario
drops that settlement.

### denom = pop

| settlement   |   ('clinic_idx', 'baseline') |   ('clinic_idx', 'excl_contributing') |   ('clinic_idx', 'excl_ind_removed') |   ('clinic_idx', 'excl_removed') |   ('clinic_idx', 'excl_rv_only') |   ('clinic_pcen', 'baseline') |   ('clinic_pcen', 'excl_contributing') |   ('clinic_pcen', 'excl_ind_removed') |   ('clinic_pcen', 'excl_removed') |   ('clinic_pcen', 'excl_rv_only') |
|:-------------|-----------------------------:|--------------------------------------:|-------------------------------------:|---------------------------------:|---------------------------------:|------------------------------:|---------------------------------------:|--------------------------------------:|----------------------------------:|----------------------------------:|
| A            |                     0.712103 |                              1        |                             1        |                         1        |                         0.712103 |                      0.029142 |                               0.029142 |                              0.029142 |                          0.029142 |                          0.029142 |
| B            |                     0.270837 |                              0.19416  |                             0.380334 |                         0.19416  |                         0.138262 |                      0.0125   |                               0.0075   |                              0.0125   |                          0.0075   |                          0.0075   |
| C            |                     0        |                              0        |                             0        |                         0        |                         0        |                      0.002286 |                               0.002286 |                              0.002286 |                          0.002286 |                          0.002286 |
| D            |                     0.045459 |                              0.063838 |                             0.063838 |                         0.063838 |                         0.045459 |                      0.004    |                               0.004    |                              0.004    |                          0.004    |                          0.004    |
| E            |                     0.071974 |                              0.101073 |                             0.101073 |                         0.101073 |                         0.071974 |                      0.005    |                               0.005    |                              0.005    |                          0.005    |                          0.005    |
| IND          |                     1        |                            nan        |                           nan        |                       nan        |                         1        |                      0.04     |                             nan        |                            nan        |                        nan        |                          0.04     |
| RV           |                     0.602275 |                            nan        |                             0.845768 |                       nan        |                       nan        |                      0.025    |                             nan        |                              0.025    |                        nan        |                        nan        |

### denom = popdensity

| settlement   |   ('clinic_idx', 'baseline') |   ('clinic_idx', 'excl_contributing') |   ('clinic_idx', 'excl_ind_removed') |   ('clinic_idx', 'excl_removed') |   ('clinic_idx', 'excl_rv_only') |   ('clinic_pcen', 'baseline') |   ('clinic_pcen', 'excl_contributing') |   ('clinic_pcen', 'excl_ind_removed') |   ('clinic_pcen', 'excl_removed') |   ('clinic_pcen', 'excl_rv_only') |
|:-------------|-----------------------------:|--------------------------------------:|-------------------------------------:|---------------------------------:|---------------------------------:|------------------------------:|---------------------------------------:|--------------------------------------:|----------------------------------:|----------------------------------:|
| A            |                     0.712103 |                              1        |                             1        |                         1        |                         0.712103 |                      0.029142 |                               0.029142 |                              0.029142 |                          0.029142 |                          0.029142 |
| B            |                     0.270837 |                              0.19416  |                             0.380334 |                         0.19416  |                         0.138262 |                      0.0125   |                               0.0075   |                              0.0125   |                          0.0075   |                          0.0075   |
| C            |                     0        |                              0        |                             0        |                         0        |                         0        |                      0.002286 |                               0.002286 |                              0.002286 |                          0.002286 |                          0.002286 |
| D            |                     0.045459 |                              0.063838 |                             0.063838 |                         0.063838 |                         0.045459 |                      0.004    |                               0.004    |                              0.004    |                          0.004    |                          0.004    |
| E            |                     0.204549 |                              0.287247 |                             0.287247 |                         0.287247 |                         0.204549 |                      0.01     |                               0.01     |                              0.01     |                          0.01     |                          0.01     |
| IND          |                     1        |                            nan        |                           nan        |                       nan        |                         1        |                      0.04     |                             nan        |                            nan        |                        nan        |                          0.04     |
| RV           |                     0.4697   |                            nan        |                             0.659594 |                       nan        |                       nan        |                      0.02     |                             nan        |                              0.02     |                        nan        |                        nan        |

Separately, removing serviceless IND alone changes NOBODY's numerator but
moves the clinic max-anchor from IND (0.04) to A (0.0291): every
settlement's clinic index rescales (A: 0.712 → 1.000 exactly). Dropping a
settlement type can change results through *renormalization alone*.

## What the current code actually does (empirically pinned)

1. **Semantics (a) is not implementable in the current code.** A bare
   `except: pass` in `calc_pcen_mobile` silently swallows contributions
   from any neighbor missing from the frame — so excluded-but-contributing
   degenerates, cell-for-cell, to fully-removed. The current no-RV pipeline
   therefore implements semantics (b) de facto
   (`tests/test_oracle.py::test_production_collapse_gap5` — the production-facing pin; the schema-level companion `tests/test_reference_impl.py::test_code_excl_contributing_collapses_to_removed` is true by construction). The silent exception
   swallowing is flagged for the Phase 3 bug audit.
2. **The barrier rule is global and asymmetric**, not pair-severing: a
   barrier-crossed settlement is deleted from everyone else's neighbor
   lists but keeps its own (in Oraculum: A counts E's services; E does not
   count A's). The manuscript describes severing the crossing only.
3. **Service points can be counted twice — but not for the reason we first
   thought.** Production counts point services with a boundary-inclusive
   spatial join, so a service digitized exactly on a colony border would
   count for BOTH colonies (the paper says only "within an administrative
   unit" and is silent on the boundary case). Measured against the current
   Delhi layers, this is **latent**: no service point sits exactly on a
   boundary (closest approach ~1 mm). However, the check surfaced a real
   one: the colony layer contains **4,050 overlapping polygon pairs**,
   which place about **450 service points inside two or more colonies**
   (bank 232, ration 104, school 53, transport 41, health 18, police 2).
   Those are genuinely double-counted today, and a containment rule would
   not fix them — it needs a decision about how overlapping colonies
   should share (or split) a service
   (`tests/test_oracle.py::test_gap6_border_point_is_double_counted_by_production`).
4. Also documented for completeness (full details in the spec): roads are
   neighbor-decayed in code though Eq. 4 has no neighbor term; `norm_psi`
   is a second normalization absent from Eq. 1; the popdensity denominator
   has no manuscript equation.

## Decisions this memo requests (none urgent; Phase 3/4 gates)

- Semantics (a) vs (b) for dropped settlement types — the code currently
  gives (b); choosing (a) requires a code fix.
- Whether the min–max universe should renormalize after drops (it does
  today) — the IND exhibit isolates exactly this effect.
- Whether the barrier asymmetry and the roads/norm_psi deviations should be
  fixed to match the manuscript, or ratified and written into the methods.
- How overlapping colony polygons should share a service point (4,050
  overlapping pairs currently double-count ~450 points), and what boundary
  convention to adopt for points lying exactly on a shared edge — the paper
  is silent on both.
