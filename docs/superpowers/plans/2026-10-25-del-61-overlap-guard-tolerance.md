# DEL-61: Overlap-lending guard tolerance recalibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note (recorded per ship.md's deviation rule):** this is a
> ~15-line change to one constant, three comment blocks and three tests.
> Per ship.md's review-budget rule ("review must not cost more than
> building"), the lead (GLM-5.3) implements inline under strict TDD rather
> than dispatching coder subagents; the plan and the code each still get
> the ultracode review loop (finders: step-5-preview + deepseek-v4.1-flash,
> judge: minimax-m3.1-flash — nvidia provider, owner-confirmed 25 Oct).

**Goal:** Recalibrate `_SHARED_TOLERANCE` so the overlap-lending guard tolerates the real layer's measured GEOS-clip noise (−2.33e-13 km, the DEL-61 pair) instead of aborting the `rule_effects` regeneration, while still raising on real cross-run mismatches.

**Architecture:** One derived constant in `delhi_psi/index.py` changes value (64 → 16384 machine epsilons) with corrected derivation comments (three comment blocks tell the old one-ulp story and all three get the corrected two-ulp story); the guard/clamp machinery is untouched. The constant's test pins are re-derived from the new measurement IN THE SAME COMMIT (they are one unit — the pins document the derivation), and one new deterministic test reproduces the exact DEL-61 deficit through the real code path. No fixture, no oracle value, no config key, no reference-implementation change.

**Tech Stack:** Python (uv + `pyproject.toml`), pytest; production module `delhi_psi.index`; shapely/GEOS is the noise source and is not touched.

**Spec:** the DEL-61 ticket (Jira, quoted below) — found by the DEL-59 whole-branch review 9 Sep 2026; the owner approved proceeding 25 Oct 2026. This plan is ship.md phase 1.

> **Ticket, verbatim (abridged to its operative parts):** "THE FAILURE. With DELHI_PSI_MEASURE_CACHE set, two tests in tests/test_measure_rule_effects.py fail: test_a_fresh_overlap_run_reproduces_the_committed_block, test_stdout_carries_blocks_and_nothing_else. Both from one root cause, delhi_psi/index.py around line 381: ValueError: overlap lending: 4770 would lend -2.3284152383951096e-13 of 'road_length' to 5169. … The guard is correct in spirit … But -2.3e-13 is not that. It is floating-point noise on a road-length subtraction … THE FIX. Give the guard a tolerance proportional to the lender's own amount rather than an absolute epsilon … Clamp a negative residual within tolerance to zero and keep raising beyond it … Pin it with a variant or unit test that reproduces the noise deterministically, so the tolerance is measured rather than guessed. … Blocks DEL-60."

## Plan review record (ship.md phase 2)

Round 1 (finders: step-5-preview, deepseek-v4.1-flash; judge: minimax-m3.1-flash):
15 raw findings → 4 killed by adversarial verification (inverted-inequality claim about the RED mechanism; band-vs-compare-tolerance scenario unreachable on fixtures because the pre-change suite is green; indentation claim; typo claim), 2 confirmed Important and fixed (Task 2's regression step previously ran the not-yet-rewritten pin test — fixed by merging the pins rewrite into Task 2's commit; Task 4's byte-identical expectation previously asserted what the `!= 0` count mechanism cannot guarantee — fixed by restating the expectation and specifying the contingency), 9 Minors fixed in one pass (logging-evidence claim, two-ulps wording, 16×/15.6× consistency, the third stale comment block, pin-bound note, name-anchored edits, fixture-vs-doc clarification, docstring wording, PR checklist item). Zero surviving blockers after fixes → loop stopped at round 1 per the stopping rule.

## Global Constraints

- The July 2025 outputs under `~/delhi_data` are a read-only baseline; nothing this plan does writes there. Measure scripts take a `--work-dir` outside it.
- No committed TEST fixture may move: `tests/fixtures/` byte-identity is guarded by the smoke suite. This is distinct from `docs/data/` blocks, which are regenerable documents — re-splicing them via DEL-59's `--splice` is the sanctioned path, not a fixture violation.
- `tests/reference_impl.py` is NOT changed. It propagates clip noise unguarded (`lent -= shared[svc].get(...)`, no raise, no clamp); production clamps to 0.0, so the two differ by ≤ ~2.4e-13 on noisy pairs — far inside the 1e-12 production-vs-reference comparison tolerance, which runs only on the fixture cities. The pre-change green suite proves no fixture deficit exceeds the old band (it would have raised), so widening the band cannot alter any fixture computation. State this in the PR.
- The guard must keep raising on a real cross-run mismatch (proved on both sides by the unchanged boundary test).
- TDD, non-negotiable: Task 1's test is written, run, and WATCHED FAILING before the constant changes.
- Match the file's existing comment style (derived-not-picked rationale, measured numbers inline). CHANGELOG.md gains an `[Unreleased]` entry. One commit per task (Task 2's commit carries the constant, its comments and its pins together — they are one unit).

## The measured facts this plan argues from

- The exact failure: `overlap lending: 4770 would lend -2.3284152383951096e-13 of 'road_length' to 5169`.
- The current constant: `_SHARED_TOLERANCE = 64 * sys.float_info.epsilon` (≈1.42e-14), band = `-_SHARED_TOLERANCE * max(1.0, abs(own))`. The observed noise is ~16,400× the band floor.
- Why the old derivation was wrong: it priced "a few dozen rounding steps of the length" and its only measurement was the benign 1.1e-16 km clamp seen on 6 Sep. DEL-61 found the real tail: `own` and `shared` are two independent GEOS clips of the same road; clip-order differences round intermediate vertices at COORDINATE magnitude. **Measured on this machine, 25 Oct 2026:** the layer reprojects to EPSG:7760 at bounds 984,181–1,033,476 m; `ulp(1.033e6 m) = 1.164e-10 m`; the observed deficit 2.328e-10 m is **exactly 2.0 coordinate ulps** = 2.33e-13 km.
- The new constant: `_SHARED_TOLERANCE = 16384 * sys.float_info.epsilon` (≈3.64e-12) — 2^14 eps, ≈15.6× the measured noise (enough for a couple dozen vertex-ulps of accumulation), and still ~275,000× below a millimetre (1e-6 km), the smallest difference a real cross-run mismatch could plausibly show.
- Acceptance already staged: `~/measure_work/cache/` holds `settlements.dedup.gpkg` + the `overlap-outside-receiver` variant, so the two gated real-data tests run `compute` only — minutes, not the 4.5-min cold dedup.

---

### Task 1: RED — pin the measured DEL-61 noise with a deterministic test

**Files:**
- Modify: `tests/test_index.py` (new test inserted after `test_the_tolerance_boundary_is_pinned_on_both_sides`)

**Interfaces:**
- Consumes: existing helper `_lend_with_shared(shared_xy)` (tests/test_index.py) — returns Y's `clinic_pcen` from the shared-clinic city, where X's own `clinic_count` is 2 and Y's own is 1; `index._SHARED_TOLERANCE`.
- Produces: `test_the_measured_del61_noise_is_inside_the_band_and_clamped` — the DEL-61 acceptance unit test; later tasks must keep it green.

- [ ] **Step 1: Write the failing test**

```python
def test_the_measured_del61_noise_is_inside_the_band_and_clamped():
    """DEL-61: regenerating the overlap_lending block on this machine
    aborted with `4770 would lend -2.3284152383951096e-13 of 'road_length'
    to 5169`. The pre-DEL-61 band (64 eps ~ 1.4e-14) read that as a
    cross-run mismatch; it is GEOS clip noise — the deficit is 2.33e-10 m,
    exactly 2.0 ulps of EPSG:7760's ~1e6 m projected coordinates (measured
    on the real layer, 25 Oct 2026), which two independent clips of the
    same road round differently.

    Pinned deterministically: the exact ticket deficit, through the real
    guard path. The failing pair's own road_length is unknown, so the
    calibration contract is asserted both at the band floor (the
    standalone first assert, own <= 1.0) and through the real code path
    (own = 2.0).
    """
    observed = 2.3284152383951096e-13
    # Calibration contract: the band's FLOOR (own <= 1.0) covers the noise.
    assert observed < index._SHARED_TOLERANCE
    # The exact deficit through the real code path: clamp, not raise.
    own = 2.0  # X's clinic_count in the shared-clinic city
    shared = own + observed
    assert shared > own, "the perturbation must survive rounding"
    assert own - shared < 0.0, "must be a real over-subtraction"
    assert _lend_with_shared(shared) == pytest.approx(1 / 200, abs=1e-15)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest -q tests/test_index.py::test_the_measured_del61_noise_is_inside_the_band_and_clamped`
Expected: FAIL — the first assert fires: `observed` (2.328e-13) is NOT below the current `_SHARED_TOLERANCE` (64 eps ≈ 1.421e-14). (Had it passed, the final `_lend_with_shared` call would raise `ValueError: ... would lend` instead — either way RED, and the RED is the guard rejecting the measured noise.)

- [ ] **Step 3: Commit the red test**

```bash
git add tests/test_index.py
git commit -m "test(index): pin the measured DEL-61 clip noise — red (fails on the 64-eps band)"
```

### Task 2: GREEN — recalibrate the constant, correct all three comment blocks, re-derive the pins (one commit)

The constant and its pins move together: the pins document the derivation, so they are one unit.

**Files:**
- Modify: `delhi_psi/index.py` — the stale `outside_receiver` comment block above the constant (currently telling the 1.1e-16 one-ulp story), the derivation comment + constant, and the in-loop clamp comment.
- Modify: `tests/test_index.py` — rewrite `test_the_tolerance_is_a_derived_multiple_of_machine_epsilon` (anchor by NAME, not line number).

**Interfaces:**
- Consumes: Task 1's test (the new band must satisfy it).
- Produces: `_SHARED_TOLERANCE = 16384 * sys.float_info.epsilon` and its re-derived pins; Tasks 3–5 rely on this.

- [ ] **Step 1: Replace the stale comment block above the constant (the `outside_receiver` explanation)**

```python
# `overlap.lending: outside_receiver` subtracts the part of a neighbour's
# amount that lies inside the receiver. For a POINT service both sides are
# integer counts and the subtraction is exact. For a LINE service they are
# two independent GEOS clips of the same road — length(road n j) and
# length(road n j n i) — so the shared part can come back larger than the
# whole by a couple of coordinate ulps: measured at -2.33e-13 km (exactly
# 2 ulps of the ~1e6 m EPSG:7760 coordinates) on the real layer (DEL-61),
# after the earlier 1.1e-16 km pair proved to be one benign case, not the
# tail. Below this tolerance the difference is clamped to zero; above it,
# it is a real mismatch between the amounts frame and the shared structure
# and raises.
```

- [ ] **Step 2: Replace the derivation comment + constant**

```python
#
# The bound is DERIVED, not picked. Two GEOS clips of the same road (one
# against the settlement, one against the pair's intersection) round
# intermediate vertices at COORDINATE magnitude — EPSG:7760 runs to
# 1.033e6 m (bounds measured 25 Oct 2026), and 2 ulps of that is
# 2.33e-10 m = 2.33e-13 km — not at the length's magnitude, which is what
# the pre-DEL-61 64-eps story assumed. The real layer measured exactly
# that: -2.3284152383951096e-13 km on the 4770->5169 road_length pair
# (DEL-61), ~16,400x the old band. 16384 * eps is 3.6e-12: 16x the
# measured noise (measured 15.6x), enough for a couple dozen vertex-ulps
# of accumulation, and still ~275,000x TIGHTER than a millimetre — the
# smallest difference a real cross-run mismatch could show.
_SHARED_TOLERANCE = 16384 * sys.float_info.epsilon
```

- [ ] **Step 3: Replace the in-loop clamp comment (keep the 20-space loop indentation)**

```python
                    # A LINE service's own amount and its shared part are two
                    # independent GEOS clips of the same road, so the shared
                    # part can exceed the whole by up to ~2.3e-13 km — two
                    # ulps of EPSG:7760's ~1e6 m coordinates (DEL-61, measured
                    # on the 4770->5169 road_length pair; the earlier 1.1e-16
                    # estimate was one benign pair, not the tail). That is
                    # float noise, not a mismatch: clamp it. The guard above
                    # still catches a real mismatch, which is off by a length,
                    # not by an ulp.
```

- [ ] **Step 4: Rewrite the pins test (by name: `test_the_tolerance_is_a_derived_multiple_of_machine_epsilon`)**

```python
def test_the_tolerance_is_a_derived_multiple_of_machine_epsilon():
    """Not an arbitrary round number: DEL-61 measured the real layer's
    clip noise at -2.3284152383951096e-13 km (4770 lending road_length to
    5169) — exactly 2.0 ulps of EPSG:7760's ~1e6 m projected coordinates,
    not the 'few dozen rounding steps of the length' the 64-eps constant
    assumed (that priced the benign 1.1e-16 pair of 6 Sep, not the tail).
    16384 eps is 3.6e-12: 16x the measurement (measured 15.6x), and still
    five-plus orders of magnitude below a millimetre — the smallest
    difference a real cross-run mismatch could show. An arbitrary 1e-9 —
    the first value shipped — would have absorbed a nanometre-scale REAL
    error without a word."""
    assert index._SHARED_TOLERANCE == 16384 * sys.float_info.epsilon
    # comfortably above the 2.33e-13 the real layer actually produced
    assert index._SHARED_TOLERANCE > 2.3284152383951096e-13 * 8
    assert index._SHARED_TOLERANCE < 1e-11
```

(The `==` pin is the airtight one; the other two are narrative bounds — the ceiling sanity-checks that the band stays orders below any real length.)

- [ ] **Step 5: Run everything the regression selection covers — including the rewritten pins and Task 1's test**

Run: `uv run pytest -q tests/test_index.py -k "shared or tolerance or lend"`
Expected: PASS — the boundary test derives its band from the constant and moves with it; the ulp-clamp and raises tests are magnitude-agnostic; the rewritten pins assert the new derivation; Task 1's test is now inside the band.

- [ ] **Step 6: Commit (constant + comments + pins, one unit)**

```bash
git add delhi_psi/index.py tests/test_index.py
git commit -m "fix(index): recalibrate the overlap-lending band to the measured DEL-61 clip noise (64 -> 16384 eps); re-derive the pins"
```

### Task 3: Smoke

- [ ] **Step 1:** Run: `uv run pytest -q -W error`
Expected: full green (the suite's committed-fixture byte-identity guards included); no fixture regenerated anywhere in this change.

### Task 4: The real-data run — DEL-61's acceptance

- [ ] **Step 1:** Run the two previously-failing gated tests against the warm cache:

```bash
DELHI_PSI_MEASURE_CACHE=~/measure_work/cache uv run pytest -q \
  "tests/test_measure_rule_effects.py::test_a_fresh_overlap_run_reproduces_the_committed_block" \
  "tests/test_measure_rule_effects.py::test_stdout_carries_blocks_and_nothing_else"
```

Expected: both PASS. **Honest expectation, not a guarantee:** the first test asserts the regenerated block equals the committed one, and the block's counts are exact `!= 0` over full-precision CSV cells (`scripts/measure_rule_effects.py:303`) with means rendered at `.6g` — so a machine-level rounding difference in a knife-edge pair can flip a count or carry a digit. The committed block was produced by a run that did not trip the 4770→5169 noise; this machine's run does, so the two runs' arithmetic differs at the ~1e-13 level by construction. Most likely it still reproduces (both runs clamp their noise pairs to 0.0), but that is what the test is FOR.

**Evidence note:** the clamp report is `log.info` and the measure subprocess never configures logging, so the "clamped N" line will NOT appear in the captured streams — the two passing tests are the acceptance evidence. (Optional, for the PR narrative: surface the clamp count with a `logging.basicConfig(level=logging.INFO)` wrapper around `runpy.run_path('scripts/measure_rule_effects.py', run_name='__main__')`.)

- [ ] **Step 2 (contingency):** if the regenerated block differs from the committed one — diagnose before anything else:
  - **Noise-level** (last digits of a `.6g` mean, or ±1 on a knife-edge `!= 0` count): regenerate the block and re-splice `docs/data/rule_effects.md` via DEL-59's `--splice`, and record the diff in the PR. `docs/data` blocks are regenerable documents — this is the sanctioned path, not a fixture violation.
  - **Real-level** (a count or mean moves beyond noise): STOP — that means the widened band is masking a genuine discrepancy, and the guard's discriminating power claim is wrong. Report to the owner before merging.

### Task 5: CHANGELOG + ship

- [ ] **Step 1:** Add to CHANGELOG.md `[Unreleased]`: the recalibration, the measured basis (2.0 coordinate ulps, the 4770→5169 pair), the DEL-60 unblock.
- [ ] **Step 2:** Branch `del-61-overlap-guard-tolerance` (off origin/main), push, PR containing: the plan link, the plan-review record (above), both test outputs (Task 4), the reference-divergence statement (production clamps ≤ ~2.4e-13 vs the unguarded reference — inside the fixture-city 1e-12 compare; no fixture computation can change, per the Global Constraints), and the deviations note (inline implementation; nvidia model lineup).
- [ ] **Step 3:** CI green → squash-merge → delete branch → sync local main.
- [ ] **Step 4:** Jira: DEL-61 → Done with the run output as evidence; comment the DEL-60 unblock.

## Self-review

- **Spec coverage:** ticket's "tolerance proportional to the lender's own amount" — already the band's shape; the constant is the deliverable (Task 2). "Clamp within tolerance, keep raising beyond" — existing machinery, proven by the unchanged boundary test. "Pin with a deterministic test reproducing the noise" — Task 1. Both failing tests pass — Task 4. Blocks DEL-60 — Task 5 notes it.
- **Placeholders:** none; every step carries its code or command.
- **Type consistency:** `_SHARED_TOLERANCE` and `_lend_with_shared` names verified against tests/test_index.py and delhi_psi/index.py; Task 2's edits anchored by symbol name, not line number.
