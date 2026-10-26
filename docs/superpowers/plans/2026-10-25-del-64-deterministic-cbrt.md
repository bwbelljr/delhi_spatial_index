# DEL-64: Deterministic cbrt — pin the transform to correctly-rounded pow — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note (ship.md deviation rule):** ~8-line change to two
> `_apply_transform` branches plus two regenerated fixture CSVs; the lead
> (GLM-5.3) implements inline under strict TDD (review-budget rule). Plan and
> code each get one scaled review round.

**Goal:** Make the cbrt transform machine-independent so the variants CSV byte-identity contract holds on every environment — unblocking CI for every PR, DEL-61 included.

**Architecture:** Both `_apply_transform` implementations (production `delhi_psi/index.py`, reference `tests/reference_impl.py`) switch from `math.cbrt` (libm, not correctly rounded, environment-dependent) to `copysign(abs(x) ** (1.0/3.0), x)` (glibc `pow` ≥ 2.28 is one fixed, documented algorithm that has been byte-stable across glibc builds to date; the CI reproduction in Task 5 Step 3 is the actual guard, not a mathematical guarantee — a future change to glibc's pow would re-break the contract identically; the sign split preserves the negative-input contract). The two committed `variants_expected_values.csv` files regenerate — the only fixture bytes that change, on cbrt rows only.

**Tech Stack:** Python (uv + pyproject), pytest; fixture generators `scripts/generate_oraculum_fixtures.py` / `generate_messy_fixtures.py`.

**Spec:** the DEL-64 ticket — the 25 Oct incident record: GitHub's runner rotation between 23–25 Oct made `math.cbrt` differ by 1 ulp from the fixture-generating environment; PR #34's re-run (unchanged code) fails identically to PR #35's two runs; first differing bytes at oraculum offset 329486 (`transform_cbrt_pcen,baseline,pop,A,clinic_pcen`) and messy offset 493573 (`transform_cbrt_pcen,nopop_only,pop,H,bank_pcen`). Owner approved the fix 25 Oct.

## Global Constraints

- `expected_values.csv` (hand-ratified oracle anchors) must NOT change — no cbrt in the ideal/code rule-sets; verify in Task 3.
- `tests/fixtures/*/production/*.csv` must NOT change — no cbrt in shipped profiles; verify in Task 3.
- The negative-input contract stands: `test_apply_transform_matches_the_named_function[("cbrt", -8.0, -2.0)]` (tests/test_index.py:287) must keep passing — hence `copysign(abs(x)**(1/3), x)`, since Python's `(-8.0) ** (1/3)` returns complex.
- Both implementations change TOGETHER in one commit (production and reference must not diverge).
- TDD: the pin tests land first, watched RED.
- Existing transform tests use `pytest.approx` tolerances ≥ 1e-15 — a ≤2-ulp shift passes them; verify, don't assume.

## The measured facts this plan argues from

- `math.cbrt` calls libm `cbrt`, which is **not** correctly rounded and varies across glibc builds. glibc `pow` (≥ 2.28) is one fixed, documented algorithm — *nearly* correctly rounded (~0.5 ulp worst case), not CR — that has been byte-stable across glibc builds to date; that empirical stability is the machine-independence argument, and the Task 5 Step 3 CI reproduction is its guard (a future glibc pow change would re-break the contract identically).
- The two incident inputs (real fixture values, verified locally 25 Oct):
  - oraculum `A,clinic_pcen` = `0.02914213562373095`: local `cbrt` → `0.30773280230712902` (committed); CI → `...907`; pow form → `...913`. Three-way disagreement — only pow is stable.
  - messy `H,bank_pcen` = `0.00909090909090909`: local `cbrt` → `0.20870640224232317` (committed); CI → `...322`; pow form → `0.2087064022423232`.
  - Both differ locally between the formulas → they serve as RED-test inputs.
- The pow form is the true cube root to ~1–2 ulp (the double `1/3` is off by ~5.6e-17 relative) — immaterial to a "measurable, not adopted" appendix variant, and the production-vs-reference compare runs at 1e-12.

---

### Task 1: RED — pin the pow formula on both sides

**Files:**
- Modify: `tests/test_index.py` (new test adjacent to `test_apply_transform_matches_the_named_function`, ~line 290)
- Modify: `tests/test_reference_impl.py` (mirror test adjacent to the existing reference transform tests)

- [ ] **Step 1: Production-side pin**

```python
@pytest.mark.parametrize("value", [0.02914213562373095,
                                   0.00909090909090909])
def test_the_cbrt_transform_is_the_deterministic_pow_formula(value):
    """DEL-64: math.cbrt is libm and NOT correctly rounded — the CI runner
    rotation of 25 Oct 2026 made it return a 1-ulp-different cube root and
    broke the variants byte-identity on every PR (including a re-run of
    unchanged, previously-green main). The pow form x**(1/3) rides glibc's
    single fixed pow algorithm, byte-stable across glibc builds to date
    (CI reproduction is the ongoing guard, not a mathematical guarantee).
    Both inputs are the real fixture values whose rows were the first to
    differ. pow yields complex for negative float bases with non-integer
    exponents, so the real cube root is sign(x) * |x|**(1/3) — the negative
    contract lives in test_apply_transform_matches_the_named_function and
    is preserved."""
    assert index._apply_transform(value, "cbrt") == (
        math.copysign(abs(value) ** (1.0 / 3.0), value))
```

- [ ] **Step 2: Reference-side pin (same body, against the reference's `_apply_transform`)**

First ADD `_apply_transform` to the existing import list at the top of
`tests/test_reference_impl.py` (currently only
`RULESETS, adjacency, apply_barrier, compute_city, emit_expected_values,
emit_variant_expected_values` — the bare call below would NameError at
collection otherwise):**

```python
@pytest.mark.parametrize("value", [0.02914213562373095,
                                   0.00909090909090909])
def test_the_reference_cbrt_transform_is_the_pow_formula(value):
    """DEL-64, reference side: same incident, same formula — the reference
    feeds the variants byte-identity regen, so its cbrt must be the same
    byte-stable pow form (glibc's single fixed pow algorithm), not
    environment-dependent libm."""
    assert _apply_transform(value, "cbrt") == (
        math.copysign(abs(value) ** (1.0 / 3.0), value))
```

- [ ] **Step 3: Run both — watch them fail**

Run: `uv run pytest -q tests/test_index.py -k "deterministic_pow" tests/test_reference_impl.py -k "pow_formula"` (as two invocations if -k scoping requires)
Expected: all four parametrized cases FAIL — `math.cbrt(value) != copysign(|value|**(1/3), value)` on both inputs.

- [ ] **Step 4: Commit the red tests**

```bash
git add tests/test_index.py tests/test_reference_impl.py
git commit -m "test: pin the cbrt transform to the byte-stable pow form — red (libm cbrt differs)"
```

### Task 2: GREEN — swap the formula in both implementations

**Files:**
- Modify: `delhi_psi/index.py` (`_apply_transform`, the `cbrt` branch ~line 224)
- Modify: `tests/reference_impl.py` (`_apply_transform`, the `cbrt` branch)

- [ ] **Step 1: Production branch**

```python
    if transform_form == "cbrt":
        # DEL-64: NOT math.cbrt — libm's cbrt is not correctly rounded, and
        # the CI runner rotation of 25 Oct 2026 made it return a 1-ulp-
        # different cube root, breaking the variants byte-identity on every
        # PR (a re-run of unchanged main failed identically). x**(1/3) rides
        # glibc's single fixed pow algorithm, byte-stable across glibc
        # builds to date — CI reproduction is the ongoing guard. pow yields
        # complex for negative float bases with non-integer exponents, so
        # the real cube root is sign(x) * |x|**(1/3).
        return math.copysign(abs(value) ** (1.0 / 3.0), value)
```

(Also touch the `TRANSFORM_FORMS` docstring note at ~line 217 only if it names `math.cbrt` — it currently describes `x^(1/3)`, which stays true.)

- [ ] **Step 2: Reference branch**

The reference's cbrt is a **bare fallthrough** `return math.cbrt(value)`
(no `if` guard, no trailing raise — reference_impl.py ~line 336). Preserve
that structure exactly: replace the bare return line, keeping it unguarded:

```python
    # DEL-64: copysign(|x|**(1/3), x), not math.cbrt — libm cbrt is not
    # correctly rounded (it broke the byte-identity regen on the rotated
    # CI runners); x**(1/3) rides glibc's single fixed pow algorithm,
    # byte-stable across glibc builds to date — CI reproduction is the
    # ongoing guard. This module must never import delhi_psi.index, and
    # the two formulas must stay identical.
    return math.copysign(abs(value) ** (1.0 / 3.0), value)
```

(Do NOT wrap it in `if transform_form == "cbrt":` — an unguarded snippet
with nothing after the `if` would change unknown-form behavior from
"compute cbrt" to "return None".)

- [ ] **Step 3: Run the pin tests + the existing transform tests**

Run: `uv run pytest -q tests/test_index.py tests/test_reference_impl.py -k "transform or pow_formula"`
Expected: PASS — pins green; `test_apply_transform_matches_the_named_function` (incl. `-8.0 → -2.0` via approx) green.

- [ ] **Step 4: Commit**

```bash
git add delhi_psi/index.py tests/reference_impl.py
git commit -m "fix(index): cbrt transform via the byte-stable pow form, not libm cbrt (DEL-64) — both implementations"
```

### Task 3: Regenerate the variants fixtures

- [ ] **Step 1:** Run both city generators (the CI drift guard's own commands):

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
uv run python scripts/generate_production_fixtures.py
```

- [ ] **Step 2: Verify the blast radius**

Run: `git status --porcelain -- tests/fixtures/`
Expected: ONLY `tests/fixtures/oraculum/variants_expected_values.csv` and `tests/fixtures/messy/variants_expected_values.csv` modified — `expected_values.csv`, `production/*.csv`, GeoJSON all untouched. Then check the diff touches only rows whose rule name contains `cbrt` (`transform_cbrt_pcen`, `transform_cbrt_psi`, AND `aggregation_mean_rank_cbrt_pcen`/`aggregation_mean_rank_cbrt_idx` — the mean-rank rule also carries transformed values), each shifting by 1–2 ulp in the last rendered digit.

- [ ] **Step 3: Commit the regenerated fixtures**

```bash
git add tests/fixtures/oraculum/variants_expected_values.csv tests/fixtures/messy/variants_expected_values.csv
git commit -m "test(fixtures): regenerate the variants CSVs under the deterministic cbrt (DEL-64)"
```

### Task 4: Smoke

- [ ] **Step 1:** Run: `uv run pytest -q -W error`
Expected: full green — including both `test_variants_expected_values_csv_is_regenerable` parametrizations, now matching the regenerated committed bytes. (~10 min on this machine; run in the background.)

### Task 5: CHANGELOG + ship

- [ ] **Step 1:** CHANGELOG `[Unreleased]` entry: the incident, the fix, the residual note (`log1p`/`exp` remain libm-dependent — today only cbrt broke).
- [ ] **Step 2:** Branch `del-64-deterministic-cbrt` (off origin/main), push, PR: the incident record (PR #34 re-run proof), the plan-review record, both smoke outputs.
- [ ] **Step 3:** CI green — this PR's own CI run is the fix's acceptance (the new runner must reproduce the new committed bytes) → squash-merge → sync main.
- [ ] **Step 4:** Re-run DEL-61's CI → green → merge PR #35 → both tickets → Done; DEL-60 unblock note.

## Review record (plan phase, round 1 — 26 Oct 2026)

- **Finders:** step-5-preview (nvidia) — verified every existing test's tolerance survives a ≤2-ulp shift, confirmed the blast radius, raised F1/F2/F3 (below); glm-flash-latest (openrouter) — verified ±0.0/±inf/subnormal behavior numerically, confirmed the generator set matches the CI drift guard and the DEL-61 merge stack is conflict-free, raised F4–F7. (deepseek-v4.1-flash dropped after two endpoint failures; minimax-judge unavailable this round — glm-latest (openrouter) judged. Noted for the record since the judge and one finder now share a provider.)
- **Judge:** glm-latest — KILLED F1 (the plan's two test names both end in `pow_formula`, so pytest's last-`-k`-wins still selects all four cases; the finder's repro used a `-k` that only existed in one file), KILLED F2 (the pins catch a revert to `math.cbrt` — that is their purpose; determinism acceptance is already located in Task 5 Step 3's CI reproduction), KILLED F3 (verified: Python's `(-8.0) ** (1/3)` DOES return complex, so the plan's original wording was correct). KEPT F4.
- **Surviving and fixed in this revision:** F4 (Important) — the "glibc pow IS correctly rounded" claim reworded everywhere to the defensible one: *nearly*-CR, one fixed algorithm, empirically byte-stable, with CI reproduction as the guard. F5 (Minor) — blast-radius check now names `aggregation_mean_rank_cbrt_*` rows too. F6 (Minor) — reference snippet now prescribes the bare unguarded fallthrough. F7 (Minor) — explicit `_apply_transform` import instruction added.
- **Zero surviving blockers → stopped at round 1.**

## Self-review

- **Spec coverage:** ticket's fix (deterministic formula both sides) — Task 2; the two failing tests green — Tasks 3–4; unblocks DEL-61 — Task 5 Step 4.
- **Placeholders:** none.
- **Type consistency:** `_apply_transform(value, transform_form)` signature verified on both sides; `math.copysign`/`abs` stdlib, no new imports (`math` already imported in both).
