# Memo to Raj: what "dropping rural villages" should mean

*Discussion piece, with three maps of the Oraculum test city. Companion to
`exclusion-semantics-memo.md` (full delta tables) and
`suggested-fixes-memo.md` §5 (proposed code change). All numbers use the
manuscript's own rules (shared-border neighbors, canal severs A–D only,
popsize denominator) so the only thing that varies between maps is how RV
is treated. Verified by `tests/reference_impl.py`; worksheet hand-ratified
24 Aug 2026.*

## The decision

We exclude rural villages (and, after recategorization, industrial areas)
from the PSI. There are two things "exclude" can mean, and they give
different numbers for the settlements that *stay*:

- **(a) Excluded but contributing.** RV gets no PSI row, but its clinics
  still count toward neighbors' accessibility. The geography is unchanged;
  we simply don't report on RV.
- **(b) Fully removed.** RV is deleted before neighbor sums. Its clinics
  cease to exist for everyone.

The current code does **(b)** — not by design, but because a bare
`except: pass` silently drops any neighbor missing from the data frame.
The published figures were produced this way.

## The three maps

| map | file | what to look at |
|---|---|---|
| 1 Baseline | `oraculum_rv_baseline.png` | B (an unauthorized colony) borders RV, which has two clinics 1 km away. B's clinic PCEN = (1 + 2·½ [A] + 2·½ [RV] + 1·½ [E]) / 200 = **0.0175**, clinic index 0.403, PSI 0.256. |
| 2 Semantics (a) | `oraculum_rv_contributing.png` | RV ghosted, its link to B dashed. **Every number on the map is identical to the baseline.** RV just has no row. |
| 3 Semantics (b) | `oraculum_rv_removed.png` | RV gone. B's clinic PCEN falls to **0.0125** (−29%), clinic index 0.403 → 0.271, PSI 0.256 → 0.237. Red box marks the change. Nobody else moves. |

The substantive question: *B's residents can still walk to RV's clinics
whether or not we report on RV.* Under (b) we tell the model those clinics
don't exist. On the real Delhi layer this affects every colony bordering a
rural village — and rural villages are numerous on the periphery, exactly
where JJCs and unauthorized colonies concentrate.

## A second effect: re-normalization

Eq. 2 takes min and max over whichever settlements remain. In Oraculum the
anchors (C, IND) survive RV's removal, so no other index moves in map 3.
That is luck of the fixture. If the dropped type contains the citywide max
or min for a service, *every* settlement's index for that service rescales
with no change to anyone's numerator (the oracle's IND exhibit shows this:
dropping serviceless IND moves A's clinic index from 0.712 to exactly
1.000). Question: is the PSI "relative to the settlements we study" or
"relative to the whole city"? The code does the former.

## What we need from you

1. (a) or (b)? Bob's default is (a), on the grounds that an analytical
   exclusion should not alter the physical service landscape.
2. Should min/max be taken over the reported settlements only (current)
   or over all settlements including dropped types?
3. The same two answers for industrial areas once they are recategorized.

Whichever way you decide, the `except: pass` gets replaced with an explicit
`exclude=(ids, contributing=True/False)` parameter so both semantics are
available and tested (DEL-21); the choice becomes one line in the methods.
