# DEL-54 — the `hi == lo` guard in `index.minmax` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** `delhi_psi.index.minmax` raises a diagnosable `ValueError` naming
the column when a min-max group is degenerate (`max == min`), instead of
dividing 0/0; the independent reference implementation does the same instead
of inventing `0.0`.

**Architecture:** One guard, placed before the division, in the one function
every service index and the second normalisation share. The reference gets
the same guard in its two min-max sites. Nothing else changes: no config
value, no profile, no fixture, no regeneration — a degenerate group is
unreachable through the committed fixtures because
`scripts/check_oraculum_invariants.check` refuses to write one.

**Tech Stack:** Python 3.13, geopandas/pandas, pytest under `-W error`, uv.

**Spec:** `docs/superpowers/specs/2026-09-05-cycle-3e-partial-barriers-design.md`
— § 4 (design), § 6.3 (proof), § 8 Group A (scope), § 12 item 9 (why the
reference changes too). Read § 4 and § 6.3 before starting.

## Global Constraints

- **Branch `del-54-minmax-guard` off `main`.** This is the first of cycle
  3E's three per-ticket branches (DEL-54 → DEL-48 → DEL-20).
- **No fixture may be regenerated and no expected value may move.** Both
  cities' `expected_values.csv`, `variants_expected_values.csv` and
  `production/*.csv` must be byte-identical to `3bf6341` when this branch
  merges. If any differs, STOP — it means the guard fired somewhere real,
  which is a finding, not a thing to regenerate around.
- **`ValueError`**, not a custom exception and not `validate.ValidationError`
  — every other refusal in `delhi_psi/index.py` is a `ValueError`, and the
  CLI already maps it to `pipeline error: …`, exit 1.
- The suite runs `uv run pytest -q -W error`. Pass an **explicit
  `timeout` of 600000 ms** on that Bash call and do NOT background it: a
  call without an explicit timeout is auto-backgrounded after 120 s and its
  completion never reaches you. The suite is ~13–15 minutes on this machine
  (real data present); if it exceeds the timeout and is backgrounded, wait
  for the log and read the summary line before committing.
- Commit trailers, both lines, on every commit:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6`

---

### Task 1: The production guard

**Files:**
- Modify: `delhi_psi/index.py:208-224` (`minmax`), and the module docstring's
  "Two deliberate non-changes" paragraph
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `index.minmax` raising `ValueError` on `max == min`; the message
  contains `repr(source_col)`, the row count, and `repr(lo)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_index.py` (match the file's existing fixture style; build
the frames the way its neighbouring tests do):

```python
def test_minmax_raises_on_a_constant_column():
    """Eq. 2 is undefined when every settlement scores the same: (v-lo)/(hi-lo)
    is 0/0. A constant column means something upstream is wrong — an empty
    service layer, or an exclusion set that removed every settlement that had
    the service — so the guard names the column instead of dividing."""
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "bank_pcen": [0.25, 0.25]},
        geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:7760")
    with pytest.raises(ValueError) as excinfo:
        index.minmax(frame, source_col="bank_pcen", target_col="bank_idx")
    message = str(excinfo.value)
    assert "'bank_pcen'" in message
    assert "0.25" in message
    assert "max == min" in message


def test_minmax_raises_on_a_single_row_frame():
    """One reported settlement is the degenerate case that reaches this in
    practice (an exclusion set that leaves one row)."""
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A"], "bank_pcen": [0.4]},
        geometry=[Point(0, 0)], crs="EPSG:7760")
    with pytest.raises(ValueError, match="'bank_pcen'"):
        index.minmax(frame, source_col="bank_pcen", target_col="bank_idx")
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest -q -W error tests/test_index.py -k minmax_raises`
Expected: FAIL — currently `minmax` does not raise; under `-W error` the
failure is numpy's `RuntimeWarning: invalid value encountered in scalar
divide` escalated to an error (NOT a `ValueError`), or on the one-row frame a
`ValueError` is not raised at all. Either way the tests are RED for the
stated reason: no guard exists. Record the exact failure text in your report.

- [ ] **Step 3: Add the guard**

In `delhi_psi/index.py`, replace `minmax` (lines 208–224) with:

```python
def minmax(polygon_gdf, *, source_col, target_col):
    """Eq. 2: rescale a column to [0, 1] across the frame.

    `calc_service_index`'s arithmetic, plus the hi == lo guard the original
    lacked (DEL-54, WORKPLAN bug-audit item 6). Eq. 2 is undefined when every
    value is equal; the original divided 0/0, which is a silent NaN outside a
    `-W error` run and an unattributed numpy RuntimeWarning inside one.

    The guard tests `max == min` and so does NOT fire on an all-NaN column
    (NaN == NaN is False). That is deliberate and out of this guard's scope:
    an all-NaN PCEN column means a NaN reached the arithmetic upstream, which
    `io.read_population`'s join and `validate` are the place to catch, and
    conflating the two would hide it behind a message about constant scores.
    """
    gdf_copy = polygon_gdf.copy()

    pcen_min = gdf_copy[source_col].min()
    pcen_max = gdf_copy[source_col].max()

    if pcen_max == pcen_min:
        raise ValueError(
            f"min-max of {source_col!r} is undefined: all "
            f"{len(gdf_copy)} values equal {pcen_min!r} (max == min), so "
            f"Eq. 2 divides 0/0. A constant column means every reported "
            f"settlement scores the same on this service — check the service "
            f"layer and the exclusion set upstream.")

    gdf_copy[target_col] = -1.0

    for idx, row in gdf_copy.iterrows():
        result = (row[source_col] - pcen_min) / (pcen_max - pcen_min)
        gdf_copy.loc[idx, target_col] = result

    return gdf_copy
```

Then update the module docstring: the "Two deliberate non-changes" paragraph
currently lists the missing guard as deliberate. It is no longer missing —
rewrite that item to say the guard was added by DEL-54 and why (a constant
column is an upstream fault, not a value to invent), leaving the other
non-change as written. Do not restate the message.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest -q -W error tests/test_index.py`
Expected: PASS, including the untouched `test_minmax_is_eq2` happy path.

- [ ] **Step 5: Prove the guard reaches both callers**

Add to `tests/test_index.py`:

```python
def test_service_index_propagates_the_guard():
    """service_index = pcen then minmax; a constant PCEN column must surface
    as the same ValueError, not as a NaN idx column."""
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "bank_count": [1, 1],
         "population": [100.0, 100.0], "area_km2": [1.0, 1.0],
         "nbrs_dist_bbox": [[], []]},
        geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:7760")
    with pytest.raises(ValueError, match="'bank_pcen'"):
        index.service_index(frame, "bank_count", service="bank",
                            denominator="pop")


def test_overall_psi_second_normalization_propagates_the_guard():
    """The second min-max is the other caller: a frame whose per-service
    indices average to the same value everywhere now names unnorm_psi."""
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B"], "bank_idx": [0.5, 0.5]},
        geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:7760")
    with pytest.raises(ValueError, match="'unnorm_psi'"):
        index.overall_psi(frame, second_normalization=True)
```

Run: `uv run pytest -q -W error tests/test_index.py -k propagates`
Expected: RED first only if you write them before Step 3 — they are written
after, so run them and confirm they PASS, then confirm they are meaningful by
temporarily reverting the guard and seeing them fail (report both results).
Note `test_overall_psi_second_normalization_propagates_the_guard` also proves
the guard does NOT fire when `second_normalization=False`: add that assertion
in the same test (`index.overall_psi(frame, second_normalization=False)`
returns a frame whose `unnorm_psi` is 0.5 for both rows).

- [ ] **Step 6: Commit**

```bash
git add delhi_psi/index.py tests/test_index.py
git commit -m "$(cat <<'MSG'
fix(index): minmax raises on a degenerate group instead of dividing 0/0 (DEL-54)

Eq. 2 is undefined when every reported settlement scores the same on a
service. The original divided 0/0 — a silent NaN outside a -W error run,
an unattributed numpy RuntimeWarning inside one. The guard names the
column, the row count and the value, and precedes the division, so both
callers (each service's service_index and overall_psi's second
normalization) surface the same ValueError.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 2: The reference implementation

**Files:**
- Modify: `tests/reference_impl.py` (the two `0.0 if hi == lo` sites)
- Test: `tests/test_reference_impl.py`

**Interfaces:**
- Consumes: nothing from Task 1 (the two implementations are independent by
  design — that is the point of the oracle).
- Produces: `reference_impl`'s min-max raising `ValueError` naming the column.

**Why this changes** (spec § 12 item 9): the reference is "the equations".
Eq. 2 is undefined at `hi == lo`; `0.0` is an invention, and a reference that
invents a value where production refuses is a rule-set divergence waiting to
be relied on. It is unreachable through the committed fixtures either way.

- [ ] **Step 1: Find both sites**

Run: `grep -n "hi == lo" tests/reference_impl.py`
Expected: two matches (the per-service index and the second normalisation).
Read the surrounding lines; the two are not textually identical, so write the
replacement for each in place rather than a blind sed.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_reference_impl.py`:

```python
def test_reference_minmax_raises_on_a_degenerate_group():
    """The reference is the equations, and Eq. 2 is undefined when hi == lo.
    It used to emit 0.0 — an invention production does not share (DEL-54).
    Unreachable through the committed fixtures: check_oraculum_invariants
    refuses to write a city with a degenerate min-max group."""
```

Build the smallest city the reference's own `compute_city` accepts whose
per-service PCEN column is constant — the module's existing test helpers show
how a city is constructed; copy that shape rather than inventing one. Assert
`pytest.raises(ValueError)` with the column name in the message.

- [ ] **Step 3: Run it and watch it fail**

Run: `uv run pytest -q -W error tests/test_reference_impl.py -k degenerate`
Expected: FAIL — the reference returns 0.0 today, so no exception is raised.

- [ ] **Step 4: Add the guard at both sites**

Replace each `0.0 if hi == lo else (pcen - lo) / (hi - lo)` with an explicit
guard raising the same shape of message as production's (name the column, the
count, the value, and say Eq. 2 divides 0/0). The two implementations are
independent, so the wording need not be identical — but both must name the
column, because the tests assert on it.

- [ ] **Step 5: Run it and see it pass**

Run: `uv run pytest -q -W error tests/test_reference_impl.py`
Expected: PASS.

- [ ] **Step 6: Prove nothing regenerated**

Run:
```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
uv run python scripts/generate_production_fixtures.py
git status --porcelain tests/fixtures/
```
Expected: **empty output** from `git status` — every fixture byte-identical.
The generators run `check_oraculum_invariants`, which refuses a degenerate
group, so the new guard is never reached. If any fixture file shows as
modified, STOP and report: that is the spec's stop-and-ask condition, not
something to commit.

- [ ] **Step 7: Full suite in the FOREGROUND, then commit**

Run: `uv run pytest -q -W error` as ONE Bash call with an explicit
`timeout` of 600000 ms and `run_in_background` NOT set. Expected: all pass
(the count was 598 at `3bf6341` with real data present; it grows by the
tests added here). Read the summary line, then:

```bash
git add tests/reference_impl.py tests/test_reference_impl.py
git commit -m "$(cat <<'MSG'
test(reference): the reference min-max raises on a degenerate group too (DEL-54)

The reference is the equations, and Eq. 2 is undefined at hi == lo; the
0.0 it used to emit was an invention production does not share, and a
divergence waiting to be relied on. Unreachable through the committed
fixtures — check_oraculum_invariants refuses to write a degenerate group —
so every fixture regenerates byte-identical.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

### Task 3: Docs and the comment the guard invalidates

**Files:**
- Modify: `tests/test_pipeline.py` (the `MISSING_ID` rationale comment)
- Modify: `CHANGELOG.md` (`[Unreleased]`)
- Modify: `WORKPLAN.md` (Phase 3 bug-audit item 6)

- [ ] **Step 1: Fix the stale comment**

Run: `grep -n -B3 -A6 "MISSING_ID" tests/test_pipeline.py`
The comment explains the missing-population path and refers to the 0/0
behaviour. Update it to say the guard now raises naming the column. Do not
change the test's assertions.

- [ ] **Step 2: WORKPLAN**

In `WORKPLAN.md`'s Phase 3 bug-audit list, item 6: strike it through or mark
it done in the style items 3, 4 and 5 already use, naming DEL-54, the commit
and the ruling (raise, not 0.0, on both sides). Keep the 28 Aug correction
paragraph — it is why the item was not urgent.

- [ ] **Step 3: CHANGELOG**

Top of `[Unreleased]`, one entry: the guard, both sides, why raising beats
0.0, the callers it covers, and explicitly that no fixture, profile or
output changed. Mention it is the first of cycle 3E's three per-ticket PRs.

- [ ] **Step 4: Full suite in the FOREGROUND**

Run: `uv run pytest -q -W error`, one Bash call, explicit `timeout` 600000,
not backgrounded. Read the summary line and include it in your report.

- [ ] **Step 5: The real-data standing proof (spec § 8 Group A task 3)**

Nothing on the real-data path changes here, which is exactly why this is
worth running: it is the cycle's standing proof that the guard never fires
on the live layers. The warm cache in `~/delhi_data/phase3_verify` makes it
minutes rather than the cold 11.

**Read-only over `~/delhi_data` except `--out-dir`, which must be a scratch
directory, never the baseline and never `phase3_verify` itself.** Run each
as its own command, backgrounded to a log (they exceed a foreground
timeout), and read the log before continuing:

```bash
mkdir -p ~/measure_work/del54-verify
uv run delhi-psi preprocess --config code-2025 --data-dir ~/delhi_data \
    --out-dir ~/measure_work/del54-verify > ~/measure_work/logs/del54-pre.log 2>&1
uv run delhi-psi compute    --config code-2025 --data-dir ~/delhi_data \
    --out-dir ~/measure_work/del54-verify > ~/measure_work/logs/del54-comp.log 2>&1
uv run python scripts/verify_against_baseline.py --config code-2025 \
    --data-dir ~/delhi_data --verify-dir ~/measure_work/del54-verify \
    > ~/measure_work/logs/del54-verify.log 2>&1
```

Expected in `del54-verify.log`: **PASS on every comparison at `0.000e+00`**
(the July 2025 baseline comparison covers 30 numeric columns × 2 output
sets = 60 comparisons — keep the WHOLE log, never a `tail`; a truncated log
is how a previous cycle mis-reported this count for three rounds). A single
non-zero deviation, or the guard raising during `compute`, is a STOP: it
would mean a real min-max group is degenerate on the live data, which is a
finding for the owner, not something to code around. Record the PASS line
and the comparison count in your report.

- [ ] **Step 6: Commit**

```bash
git add tests/test_pipeline.py CHANGELOG.md WORKPLAN.md
git commit -m "$(cat <<'MSG'
docs(index): record the DEL-54 min-max guard

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AyvMmN2HWTBxNFQ67HvcL6
MSG
)"
```

---

## Self-review

- **Spec coverage.** § 4's message shape, `ValueError` choice, both callers,
  the module and function docstrings, the `test_pipeline.py` comment, and the
  reference ruling of § 12 item 9 each map to a step above. § 6.3's four
  production assertions and one reference assertion are Tasks 1 and 2.
- **Placeholders.** None: every code step carries the code, and the two
  places that say "copy the file's existing fixture style" name the file and
  the neighbouring tests to copy, because inventing a different frame shape is
  the likelier error.
- **Names.** `minmax(polygon_gdf, *, source_col, target_col)`,
  `service_index(...)`, `overall_psi(polygon_gdf, *, second_normalization)`
  are used exactly as they are defined in `delhi_psi/index.py`. Test frames
  are built with `gpd.GeoDataFrame` — the alias `tests/test_index.py`
  imports; a bare `geopandas.` would be a `NameError`.

## Plan review R1 (5 Sep 2026, 2 lenses + adversarial verify)

Three confirmed findings, all applied above; zero refuted:
1. The four test snippets used `geopandas.GeoDataFrame` where the file
   imports `geopandas as gpd` — every new test would have raised
   `NameError` before reaching the guard, so Step 2's RED would have been
   for the wrong reason and Step 4's PASS unreachable. Fixed.
2. The plan omitted the real-data `code-2025` verify that spec § 8 Group A
   task 3 requires and every prior cycle's plan runs. Added as Task 3
   Step 5, with the whole-log rule and the stop condition.
3. (Same as 1, found independently by the second lens.)

One minor, ruled rather than fixed: the guard does not fire on an all-NaN
column, because `NaN == NaN` is False. Left out of scope deliberately — an
all-NaN PCEN column is an upstream NaN, which belongs to the population
join and `validate`, not to a message about constant scores — and now
stated in the `minmax` docstring so the limit is visible rather than
accidental.

Execution: subagent-driven-development.
