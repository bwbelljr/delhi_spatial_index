# Rank Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `methodology.aggregation.rule: mean_minmax | mean_rank` — an alternative to Eq. 2's min-max that replaces each service's PCEN with its percentile rank — implemented independently in the production package and the reference implementation, cross-checked at 1e-12 on both fixture cities, and adopted by neither shipped profile.

**Architecture:** A new required `methodology.aggregation` block, parsed like `methodology.transform`. `index.service_index` dispatches its last step between `minmax` and a new `percentile_rank_column`; `tests/reference_impl.py` implements the same rule from scratch. Three `tests/variants.py` rows put both implementations against each other.

**Tech Stack:** Python 3, pandas, geopandas, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-08-del-57-rank-aggregation-design.md`

## Global Constraints

- **Both shipped profiles (`code-2025`, `manuscript`) carry `rule: mean_minmax`.** Every existing expected value and production fixture must be **byte-identical**. If one moves, STOP and report — the switch has leaked into the default path.
- **`variants_expected_values.csv` is addition-only**, verified with `git diff --numstat` (zero deletions).
- **`methodology:` is a complete statement in every profile.** Every profile file in `delhi_psi/profiles/` — the two shipped ones AND all eleven sweep profiles AND the service-subset profiles — gains the `aggregation` block. `tests/test_config.py` asserts the exact set of shipped profiles.
- **The reference implementation imports nothing from `delhi_psi`.** That independence is the whole value of the oracle; a shared helper would void it.
- **No real-data run.** No adoption. No new dependency. No licence file (DEL-56, pending Raj).
- **TDD is mandatory.** Write the failing test, run it, watch it fail, then implement. Code written before its test is unverified — discard and redo.
- Run only your own task's test files. Do not background any pytest process.
- The rank rule, verbatim, used identically by both implementations: **rank ascending, averaging ranks within a tie block, then rescale `(rank - 1) / (n - 1)`.** `n == 1` raises.

## Facts verified before this plan was written (8 Sep 2026)

- Under this rank rule, a `pcen`-stage `log1p` or `cbrt` is an **exact** no-op: measured on 100 values with a 30 % mass at zero, `max abs difference = 0.000e+00` for both. This is why Task 4's equality assertion is exact, not approximate.
- A constant column ranks to **0.5 for every settlement**, where `minmax` raises.
- `index.minmax` raises on `hi == lo` (DEL-54); `reference_impl` raises the same way at its own `hi == lo` check. `mean_rank` replaces both of those code paths, so neither guard fires under it.

---

### Task 1: the config block

**Files:**
- Modify: `delhi_psi/config.py`
- Modify: every `*.yaml` in `delhi_psi/profiles/`
- Modify: `tests/test_config.py` — **including the `MINIMAL` constant**
- Modify: `tests/test_reference_impl.py` — **two direct `MethodologyConfig(...)` constructions**
- Modify: `tests/oraculum_fixtures.py` — `variant_methodology`
- Modify: `tests/test_profiles_match_reference.py` — the `knob_for_key` dict

**Making `aggregation` a required block breaks four things the first draft
of this plan missed.** The plan review found all four; each is a hard
failure, not a value mismatch, and each must be fixed IN THIS TASK or the
suite goes red for reasons unrelated to the feature:

1. **`tests/test_config.py`'s `MINIMAL` constant** (~line 33) is the YAML
   nearly every test in that file builds on, referenced 28 times. Without an
   `aggregation` key every one of them raises `ConfigError`. Add
   `  aggregation: {rule: mean_minmax}` immediately after its `transform:`
   line.
2. **`tests/test_reference_impl.py` constructs `MethodologyConfig(...)`
   directly** in two tests (~lines 598 and ~688) — the only two places in
   the repo outside `config.py` that do. A new field with no default makes
   both raise `TypeError`. Add
   `aggregation=AggregationConfig(rule=AggregationRule.MEAN_MINMAX)` to
   both, and import the two names locally as those tests already import
   their siblings.
3. **`tests/oraculum_fixtures.py::variant_methodology`** (~lines 155-207)
   applies a variant's override block by block — `if "adjacency" in spec`,
   `"barrier"`, `"overlap"`, `"decay"`, `"transform"`. With no
   `"aggregation"` branch it **silently drops the override**, so production
   would compute `mean_minmax` while the reference computes `mean_rank`,
   and `test_variants_match_reference` fails at 1e-12 on 12 cases with no
   hint as to why. Add:

   ```python
   if "aggregation" in spec:
       block = spec["aggregation"]
       methodology = replace(methodology, aggregation=AggregationConfig(
           rule=AggregationRule(block["rule"])))
   ```

   This one is doing real work — without it Task 4 fails and the cause is
   invisible from the failure message.
4. **`tests/test_profiles_match_reference.py`'s `knob_for_key` dict**
   (~lines 120-132) is the repo's one generic "every mapped knob actually
   reaches `compute_city`" cross-check. Add the
   `"methodology.aggregation.rule"` entry so the new knob is exercised
   rather than silently skipped.

**Interfaces:**
- Produces: `AggregationRule` enum (`MEAN_MINMAX = "mean_minmax"`, `MEAN_RANK = "mean_rank"`), `AggregationConfig(rule)`, and `MethodologyConfig.aggregation`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`, matching the file's existing style for loading a profile and for asserting a rejection:

```python
def test_every_profile_declares_an_aggregation_rule():
    """`methodology:` is a complete statement in every profile (the config
    contract), so a new methodology block is added to ALL of them, not just
    the two shipped ones."""
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        assert "aggregation" in raw["methodology"], path.name
        assert raw["methodology"]["aggregation"]["rule"] in (
            "mean_minmax", "mean_rank"), path.name


def test_both_shipped_profiles_keep_todays_aggregation():
    for name in ("code-2025", "manuscript"):
        cfg = load_config(name)
        assert cfg.methodology.aggregation.rule is AggregationRule.MEAN_MINMAX


def test_an_unknown_aggregation_rule_is_rejected(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, swap("  aggregation:",
                                         "  aggregation: {rule: median_rank}")))
    message = str(exc.value)
    assert "methodology.aggregation.rule" in message
    for allowed in REFERENCE_KNOBS["methodology.aggregation.rule"]:
        assert str(allowed) in message


def test_the_rule_is_required_inside_the_aggregation_block(tmp_path):
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, swap("  aggregation:",
                                         "  aggregation: {}")))
    assert "methodology.aggregation.rule" in str(exc.value)


def test_aggregation_is_required_like_every_methodology_block(tmp_path):
    without = MINIMAL.replace("  aggregation: {rule: mean_minmax}\n", "")
    with pytest.raises(ConfigError) as exc:
        load_config(write(tmp_path, without))
    assert "methodology.aggregation" in str(exc.value)
```

**These use the file's real idiom, which the first draft of this plan got
wrong.** `tests/test_config.py` has no raw-dict helper: every rejection test
builds a **complete YAML profile** from the `MINIMAL` constant via `write()`
and `swap()`/`.replace()`, calls `load_config()`, and catches
**`ConfigError`** — not `ValueError`. Read the existing
`test_form_is_required_inside_the_transform_block` and
`test_transform_enums_name_the_key_and_the_allowed_values` and match them
exactly. The `swap()` calls above assume `MINIMAL` gains its `aggregation`
line as a `{rule: mean_minmax}` one-liner, matching how `transform` appears
there.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q -W error`
Expected: FAIL — `AggregationRule` is not importable and no profile has the block.

Note the ordering: adding the `aggregation` line to `MINIMAL` before
`config.py` accepts the key will make the *existing* tests fail too, with
"unknown key". That is expected and transient — do Step 3's `config.py`
change and the `MINIMAL` change together, then Step 4 must show the whole
file green.

- [ ] **Step 3: Implement**

In `delhi_psi/config.py`, following the `transform` block's pattern exactly:

1. Add to the reference-pinned mapping beside `"methodology.transform.stage"`:

```python
    "methodology.aggregation.rule": {"mean_minmax": "mean_minmax",
                                     "mean_rank": "mean_rank"},
```

2. Add `"methodology.aggregation.rule"` to the reference-pinned key list
   (the tuple a few lines below that holds `"methodology.transform.form"`
   and `"methodology.transform.stage"`).

3. Add the enum beside `TransformStage`:

```python
AggregationRule = _make_enum("AggregationRule", "methodology.aggregation.rule")
```

   and register it in the enum lookup dict beside
   `"methodology.transform.stage": TransformStage`.

4. Add the dataclass after `TransformConfig`:

```python
@dataclass(frozen=True)
class AggregationConfig:
    # DEL-57: what Eq. 2 IS, as opposed to `transform`, which chooses a
    # function applied to a value. `mean_minmax` is today: min-max each
    # service's PCEN, then average. `mean_rank` replaces the min-max with a
    # percentile rank, which cannot form the mass point at zero that Eq. 2's
    # min-max produces on a right-skewed distribution (452 of 4,131
    # settlements at the published baseline). Neither shipped profile adopts
    # it; this is measurement machinery, and Raj decides adoption.
    rule: AggregationRule
```

5. Add `aggregation: AggregationConfig` to `MethodologyConfig`.

6. Add `"aggregation"` to the methodology block's allowed-key tuple (the one
   listing `"transform", "roads", "second_normalization"`).

7. Parse it beside the transform parsing:

```python
    aggregation_raw = _require(raw, "aggregation", "methodology")
    _reject_unknown(aggregation_raw, {"rule"}, "methodology.aggregation")
    aggregation = AggregationConfig(
        rule=_coerce_enum(
            "methodology.aggregation.rule",
            _require(aggregation_raw, "rule", "methodology.aggregation")))
```

   and pass `aggregation=aggregation` in the `MethodologyConfig(...)` call.

8. Add to **every** file in `delhi_psi/profiles/`, inside `methodology:`,
   immediately after the `transform:` block:

```yaml
  aggregation:
    rule: mean_minmax               # mean_minmax | mean_rank (DEL-57)
                                    # mean_minmax is today: Eq. 2 min-maxes
                                    # each service's PCEN, then Eq. 1
                                    # averages. mean_rank replaces the
                                    # min-max with a percentile rank —
                                    # uniform by construction, so no mass
                                    # point at zero can form. Not adopted.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_reference_impl.py tests/test_profiles_match_reference.py -q -W error`
Expected: PASS — all three, because this task touches all three.

- [ ] **Step 5: Verify no fixture moved, then commit**

```bash
git diff --stat tests/fixtures/
git add delhi_psi/config.py delhi_psi/profiles tests/test_config.py tests/test_reference_impl.py tests/oraculum_fixtures.py tests/test_profiles_match_reference.py
git commit -m "feat(config): methodology.aggregation.rule — mean_minmax or mean_rank (DEL-57)"
```

`git diff --stat tests/fixtures/` must print nothing.

---

### Task 2: the production rule

**Files:**
- Modify: `delhi_psi/index.py`
- Test: `tests/test_index.py`

**Interfaces:**
- Consumes from Task 1: `AggregationRule` (as a plain string value at the `index.py` boundary — `index.py` takes `aggregation_rule="mean_minmax"` as a string keyword, matching how `transform_form`/`transform_stage` are already passed).
- Produces: `percentile_rank_column(polygon_gdf, *, source_col, target_col)` and a new `aggregation_rule` keyword on `service_index`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_index.py`:

```python
def test_percentile_rank_column_scales_min_to_zero_and_max_to_one():
    frame = gpd.GeoDataFrame({"x": [3.0, 1.0, 2.0]}, geometry=[None] * 3)
    out = index.percentile_rank_column(frame, source_col="x", target_col="r")
    assert list(out["r"]) == [1.0, 0.0, 0.5]


def test_percentile_rank_column_averages_ranks_within_a_tie_block():
    """Ranks 1,2,3,4 over values 1,2,2,5 -> the tie block takes (2+3)/2 = 2.5,
    so the rescaled ranks are 0, 0.5, 0.5, 1."""
    frame = gpd.GeoDataFrame({"x": [1.0, 2.0, 2.0, 5.0]}, geometry=[None] * 4)
    out = index.percentile_rank_column(frame, source_col="x", target_col="r")
    assert list(out["r"]) == [0.0, 0.5, 0.5, 1.0]


def test_percentile_rank_column_is_defined_where_minmax_raises():
    """A constant column: `minmax` raises (Eq. 2 divides 0/0), a rank does
    not — every settlement ties, so every rescaled rank is 0.5. This is a
    real behavioural difference between the two rules, not a detail."""
    frame = gpd.GeoDataFrame({"x": [0.4] * 5}, geometry=[None] * 5)
    out = index.percentile_rank_column(frame, source_col="x", target_col="r")
    assert set(out["r"]) == {0.5}
    with pytest.raises(ValueError, match="undefined"):
        index.minmax(frame, source_col="x", target_col="r")


def test_percentile_rank_column_refuses_a_single_row():
    """`(rank - 1) / (n - 1)` is the same 0/0 `minmax`'s hi == lo guard
    refuses (DEL-54); a one-settlement city has no value to invent."""
    frame = gpd.GeoDataFrame({"x": [1.0]}, geometry=[None])
    with pytest.raises(ValueError, match="x"):
        index.percentile_rank_column(frame, source_col="x", target_col="r")


def test_an_unknown_aggregation_rule_is_rejected():
    frame = gpd.GeoDataFrame({"x": [1.0, 2.0]}, geometry=[None] * 2)
    with pytest.raises(ValueError, match="mean_minmax"):
        index._apply_aggregation(frame, source_col="x", target_col="r",
                                 aggregation_rule="median_rank")
```

Match the file's existing conventions for building a small GeoDataFrame —
read a nearby `minmax` test first and copy its construction rather than the
`geometry=[None] * n` shown here if the file does it differently.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_index.py -q -W error`
Expected: FAIL — `module 'delhi_psi.index' has no attribute 'percentile_rank_column'`.

- [ ] **Step 3: Implement**

In `delhi_psi/index.py`, after `minmax`:

```python
AGGREGATION_RULES = ("mean_minmax", "mean_rank")


def percentile_rank_column(polygon_gdf, *, source_col, target_col):
    """DEL-57's alternative to Eq. 2: the percentile rank of a column.

    Rank ascending, AVERAGING ranks within a tie block, then rescale
    `(rank - 1) / (n - 1)` so the minimum maps to 0 and the maximum to 1 —
    the same endpoints `minmax` produces, which is what lets `mean_rank`
    stand in for `mean_minmax` without anything downstream learning a new
    range.

    Two deliberate differences from `minmax`:

    * A CONSTANT column is fine here and raises there. Every value ties, so
      every average rank is equal and every settlement scores 0.5 — a
      uniform shift that changes no ordering. Eq. 2's min-max is genuinely
      undefined on that input; a rank is not.
    * `n == 1` raises, because `(rank - 1) / 0` is the same 0/0 that
      `minmax`'s hi == lo guard refuses (DEL-54). A one-settlement city has
      no value to invent.
    """
    gdf_copy = polygon_gdf.copy()
    n = len(gdf_copy)
    if n < 2:
        raise ValueError(
            f"percentile rank of {source_col!r} is undefined across {n} "
            "row(s): the rescaling divides by (n - 1), so a single "
            "settlement has no rank to report. Check the exclusion set "
            "upstream.")
    ranks = gdf_copy[source_col].rank(method="average", ascending=True)
    gdf_copy[target_col] = (ranks - 1.0) / (n - 1.0)
    return gdf_copy


def _apply_aggregation(polygon_gdf, *, source_col, target_col,
                       aggregation_rule):
    """Eq. 2, dispatched on `methodology.aggregation.rule` (DEL-57)."""
    if aggregation_rule == "mean_minmax":
        return minmax(polygon_gdf, source_col=source_col,
                      target_col=target_col)
    if aggregation_rule == "mean_rank":
        return percentile_rank_column(polygon_gdf, source_col=source_col,
                                      target_col=target_col)
    raise ValueError(
        f"unknown aggregation rule {aggregation_rule!r}; allowed values: "
        f"{list(AGGREGATION_RULES)}")
```

Then in `service_index`: add `aggregation_rule="mean_minmax"` to the
keyword list, and replace the final line

```python
    return minmax(out, source_col=pcen_col, target_col=idx_col)
```

with

```python
    return _apply_aggregation(out, source_col=pcen_col, target_col=idx_col,
                              aggregation_rule=aggregation_rule)
```

Extend `service_index`'s docstring with a sentence naming the new keyword
and pointing at `_apply_aggregation`.

**Then wire it through the caller.** `delhi_psi/pipeline.py`, in
`index_frames` (~lines 238-239), passes the enum members **directly, never
`.value`** — this works because every config enum is a `StrEnum` and so
compares equal to its bare string:

```python
transform_form=methodology.transform.form,
transform_stage=methodology.transform.stage,
```

Add, in the same call, in the same style:

```python
aggregation_rule=methodology.aggregation.rule,
```

Do **not** write `.value`. (The plan review resolved this against the real
call site; the first draft guessed wrong.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_index.py tests/test_pipeline.py -q -W error`
Expected: PASS.

- [ ] **Step 5: Verify nothing moved, then commit**

```bash
git diff --stat tests/fixtures/
git add delhi_psi/index.py delhi_psi/pipeline.py tests/test_index.py
git commit -m "feat(index): percentile-rank aggregation as an alternative to Eq. 2's min-max (DEL-57)"
```

`git diff --stat tests/fixtures/` must print nothing — both shipped profiles
still say `mean_minmax`, so no fixture may move.

---

### Task 3: the independent reference rule

**Files:**
- Modify: `tests/reference_impl.py`
- Test: `tests/test_reference_impl.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2. **`tests/reference_impl.py` imports nothing from `delhi_psi` and must not start now** — implement the rank rule from scratch there. That independence is the entire value of the two-implementation oracle.
- Produces: an `aggregation_rule="mean_minmax"` keyword on `compute_city`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reference_impl.py`:

```python
def test_reference_mean_rank_ranks_each_service_independently(
        settlements, services, barriers):
    """Oraculum has 7 settlements, so every average rank is hand-checkable:
    the ranks of 7 values rescale to 0, 1/6, 2/6, ... 1. Under `mean_rank`
    every `*_idx` column must be drawn from exactly that set, or from a tie
    average of two members of it, for every service."""
    frame = _city_df(settlements, services, barriers, "code",
                     aggregation_rule="mean_rank")
    allowed = {i / 6 for i in range(7)}
    allowed |= {(a + b) / 2 for a in allowed for b in allowed}
    for col in [c for c in frame.columns if c.endswith("_idx")]:
        assert set(frame[col]).issubset(allowed), col


def test_reference_mean_rank_is_unmoved_by_a_pcen_stage_transform(
        settlements, services, barriers):
    """log1p is strictly monotone and injective, so it preserves both the
    ordering AND the tie structure — the average ranks cannot change. This
    is the ticket's claim that `transform` goes moot under `mean_rank`,
    asserted as an EXACT equality rather than approximately."""
    plain = _city_df(settlements, services, barriers, "code",
                     aggregation_rule="mean_rank")
    transformed = _city_df(settlements, services, barriers, "code",
                           aggregation_rule="mean_rank",
                           transform_form="log1p", transform_stage="pcen")
    idx_cols = [c for c in plain.columns if c.endswith("_idx")]
    pd.testing.assert_frame_equal(plain[idx_cols], transformed[idx_cols])
```

**Use the module's real fixtures and helper**, resolved by the plan review:
`tests/test_reference_impl.py` has module-scoped `settlements()`,
`services()` and `barriers()` fixtures (~lines 33-45) and a helper

```python
def _city_df(settlements, services, barriers, rule, **overrides):
```

which fills in `RULESETS[rule]` plus `scenario="baseline"`, `denom="pop"`.
`compute_city` takes `adjacency_rule`, `barrier_rule`, `roads_formula`,
`scenario`, `denom`, `second_norm` and `absent_neighbor_contribution` as
required keyword arguments, so calling it with a bare positional triple
would raise `TypeError` — the first draft of this plan did exactly that.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_reference_impl.py -q -W error`
Expected: FAIL — `compute_city() got an unexpected keyword argument 'aggregation_rule'`.

- [ ] **Step 3: Implement**

In `tests/reference_impl.py`, add `aggregation_rule="mean_minmax"` to
`compute_city`'s keyword list, validate it beside the existing transform
validation:

```python
    if aggregation_rule not in ("mean_minmax", "mean_rank"):
        raise ValueError(
            f"unknown aggregation rule {aggregation_rule!r}; allowed "
            "values: ['mean_minmax', 'mean_rank']")
```

Then replace the per-service normalisation loop. Today it reads:

```python
    for svc in POINT_SERVICES + ("road",):
        col = f"{svc}_pcen"
        pcen = df[col]
        lo, hi = pcen.min(), pcen.max()
        if hi == lo:
            raise ValueError(...)
        df[f"{svc}_idx"] = (pcen - lo) / (hi - lo)
        idx_cols.append(f"{svc}_idx")
```

Make it dispatch, keeping the min-max path byte-for-byte as it is:

```python
    n = len(df)
    for svc in POINT_SERVICES + ("road",):
        col = f"{svc}_pcen"
        pcen = df[col]
        if aggregation_rule == "mean_rank":
            # DEL-57: rank ascending with tie blocks averaged, rescaled so
            # min -> 0 and max -> 1. Written out longhand rather than via
            # Series.rank, so this stays an INDEPENDENT statement of the
            # rule rather than a second call to the same library routine
            # the production side uses.
            if n < 2:
                raise ValueError(
                    f"percentile rank of {col!r} is undefined across {n} "
                    "row(s): the rescaling divides by (n - 1)")
            order = sorted(range(n), key=lambda k: pcen.iloc[k])
            ranks = [0.0] * n
            position = 0
            while position < n:
                stop = position
                while (stop + 1 < n
                       and pcen.iloc[order[stop + 1]] == pcen.iloc[order[position]]):
                    stop += 1
                average = (position + stop) / 2 + 1
                for k in range(position, stop + 1):
                    ranks[order[k]] = average
                position = stop + 1
            df[f"{svc}_idx"] = [(r - 1.0) / (n - 1.0) for r in ranks]
        else:
            lo, hi = pcen.min(), pcen.max()
            if hi == lo:
                raise ValueError(
                    f"min-max of {col!r} is undefined: all {len(pcen)} "
                    f"values equal {lo!r} (hi == lo), so Eq. 2 divides 0/0 "
                    "— every settlement scores the same on this service")
            df[f"{svc}_idx"] = (pcen - lo) / (hi - lo)
        idx_cols.append(f"{svc}_idx")
```

Keep the existing `hi == lo` message text exactly as it is today — a test
may match on it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_reference_impl.py -q -W error`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/reference_impl.py tests/test_reference_impl.py
git commit -m "test(reference): the rank rule, stated independently of the package it checks (DEL-57)"
```

---

### Task 4: the variant rows

**Files:**
- Modify: `tests/variants.py`
- Modify: `tests/fixtures/*/variants_expected_values.csv` (regenerated)
- Test: `tests/test_variant_rules.py`

**Interfaces:**
- Consumes: Task 1's config key, Task 2's production rule, Task 3's reference rule.
- Produces: nothing later tasks use.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_variant_rules.py`:

```python
@pytest.mark.parametrize("denom", ["pop", "popdensity"])
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_a_pcen_stage_transform_does_not_move_a_rank_aggregation(city, denom):
    """The sharpest statement of how DEL-34 and DEL-57 relate: log1p is
    injective, so it preserves the tie structure as well as the ordering,
    so the average ranks are IDENTICAL — not merely close. This is the only
    pair of variants in this repo whose values must be equal, and asserting
    that equality directly is stronger than the 1e-12 agreement each of
    them separately gets against the reference implementation.

    `_idx` only, deliberately: a `pcen`-stage transform REPLACES the
    reported `*_pcen` value by design (DEL-34), so those columns differ
    substantially and must. Verified during plan review on a skewed
    vector — `*_idx` max abs difference 0.0, `*_pcen` ≈ 4.9."""
    plain = variant(city, "aggregation_mean_rank", denom)
    transformed = variant(city, "aggregation_mean_rank_log1p_pcen", denom)
    idx_cols = [c for c in plain.columns if c.endswith("_idx")]
    pd.testing.assert_frame_equal(plain[idx_cols], transformed[idx_cols])
```

**Use the module's real helper and its parametrization convention**, both
resolved by the plan review: `variant(city, name, denom)` (~lines 51-62)
returns a settlement × metric DataFrame, and every other test in this file
is parametrized over `CITIES` and both denominators. The first draft of
this plan invented a `_variant_values(name)` helper that does not exist and
covered one city.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_variant_rules.py -q -W error`
Expected: FAIL — the variant names do not exist.

- [ ] **Step 3: Implement**

Add to `tests/variants.py`'s variant dict, following the `transform_*` rows'
shape exactly, and add `("aggregation", "rule"): "aggregation_rule"` to the
key-mapping dict beside the two `transform` entries:

```python
    # DEL-57: rank aggregation — Eq. 2 becomes a percentile rank rather than
    # a min-max. Alternative to DEL-34's transforms, not a companion: the
    # third row exists to prove they are alternatives.
    "aggregation_mean_rank": {
        "aggregation": {"rule": "mean_rank"},
    },
    # Must produce values IDENTICAL to the row above — log1p is injective,
    # so it moves no rank and breaks no tie.
    "aggregation_mean_rank_log1p_pcen": {
        "aggregation": {"rule": "mean_rank"},
        "transform": {"form": "log1p", "stage": "pcen"},
    },
    # A `psi`-stage transform DOES still bite under mean_rank: it acts on
    # the composite, which is a mean of ranks, not a rank.
    "aggregation_mean_rank_log1p_psi": {
        "aggregation": {"rule": "mean_rank"},
        "transform": {"form": "log1p", "stage": "psi"},
    },
```

Then regenerate the variant expected values. The plan review resolved the
exact commands — it is **not** `scripts/generate_production_fixtures.py`
(that writes `tests/fixtures/*/production/*.csv`); the variant CSVs are
written by `scripts/check_oraculum_invariants.emit_checked_variant_expected_values`,
which the two per-city geometry generators call:

```bash
uv run python scripts/generate_oraculum_fixtures.py
uv run python scripts/generate_messy_fixtures.py
```

Both also rewrite the cities' geometry and `expected_values.csv`. **No
geometry changes here, so only `variants_expected_values.csv` may differ** —
Step 5 checks exactly that.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_variant_rules.py tests/test_variants_match_reference.py -q -W error`
Expected: PASS, with the three new variants agreeing at 1e-12 on both cities.

- [ ] **Step 5: Verify addition-only, then commit**

```bash
git diff --numstat tests/fixtures/
git diff --stat tests/fixtures/*/expected_values.csv
```

The first must show **zero deletions** in the `variants_expected_values.csv`
rows. The second must print nothing — the non-variant expected values may
not move at all. If either fails, STOP and report.

```bash
git add tests/variants.py tests/fixtures tests/test_variant_rules.py
git commit -m "test(variants): three rank-aggregation rows, and the transform no-op proved (DEL-57)"
```

---

### Task 5: the config documentation

**Files:**
- Modify: `docs/methodology-config.md`
- Test: none new — `tests/test_config.py` from Task 1 covers the behaviour.

- [ ] **Step 1: Write the documentation**

Add `methodology.aggregation.rule` to `docs/methodology-config.md`'s
switch table (match the existing rows' shape), and add a short subsection
covering the two things a reader cannot infer from the enum:

1. **A constant column stops being an error.** `mean_minmax` raises when a
   service's PCEN is constant (Eq. 2 divides 0/0, DEL-54); `mean_rank`
   returns 0.5 for every settlement. Under `mean_rank` a service nobody
   owns will pass silently rather than halting the run — a real change in
   what the pipeline accepts.
2. **A `pcen`-stage transform becomes a no-op.** `log1p` and `cbrt` are
   injective, so they change no rank and break no tie; the values are
   exactly identical. A `psi`-stage transform still bites, because the
   composite is a mean of ranks rather than a rank. Cite the variant pair
   that pins it.

Do not restate the spec. Two short paragraphs.

- [ ] **Step 2: Commit**

```bash
git add docs/methodology-config.md
git commit -m "docs(config): what changes under mean_rank — a constant column, and a moot transform (DEL-57)"
```

---

## Self-review

**Spec coverage.** § 2 the switch → Task 1. § 3 the rank rule and the `n == 1`
guard → Task 2. § 4's two measured differences → Task 2 (constant column,
`n == 1`) and Tasks 3–4 (the transform no-op). § 5 what must not move →
every task's Step 5. § 6 three variant rows → Task 4. § 8's "documented in
`docs/methodology-config.md`" → Task 5.

**Placeholders.** Four places name a helper the plan cannot know
(`_methodology_from_raw`, `_oraculum_args`, `_variant_values`, and the
variant generator script). Each says explicitly to read the file and use its
existing helper rather than invent one — that is a deliberate instruction,
not a gap, because inventing a parallel helper is the more likely error.

**Type consistency.** `AggregationRule` is the config enum; `index.py` takes
the plain string (`"mean_minmax"` / `"mean_rank"`) exactly as it already
takes `transform_form`. Task 2's Step 3 tells the implementer to read the
`transform_form` call site and match it rather than assume, because getting
this wrong silently passes an enum where a string is compared.

**Plan review (one round, Sonnet, 8 Sep 2026).** It resolved every helper
this plan could not name and found **three Criticals**, all of the same
shape: making `aggregation` a REQUIRED block breaks consumers of
`MethodologyConfig` that the first draft never looked for. All three are
fixed above, in Task 1:

1. `tests/test_config.py`'s `MINIMAL` constant — 28 references, every one
   of which would raise `ConfigError`.
2. `tests/test_reference_impl.py`'s two direct `MethodologyConfig(...)`
   constructions — the only two outside `config.py` — would raise
   `TypeError`.
3. `tests/oraculum_fixtures.py::variant_methodology` applies overrides
   block by block and has no `aggregation` branch, so it would **silently
   drop** the override: production computing `mean_minmax` against a
   reference computing `mean_rank`, failing 12 cases at 1e-12 with nothing
   in the message to say why. This is the one that would have cost real
   debugging time.

It also corrected three things the plan had guessed: the enum is passed to
`service_index` **directly, not as `.value`**; `test_config.py` raises
`ConfigError` from full-YAML fixtures rather than `ValueError` from a
raw-dict helper that does not exist; and the variant regeneration runs
through the two per-city geometry generators.

**The risk this plan named was the wrong one.** The first draft warned that
the longhand tie-block loop in Task 3 was where an off-by-one would live.
The reviewer executed it against all-distinct, tie blocks at the start,
middle and end, two tie blocks, all-tied, and `n = 2` — it matches
`Series.rank(method="average")` exactly in every case, and
`average = (position + stop) / 2 + 1` is algebraically the mean of the
1-based ranks it stands for. The arithmetic was never the danger; the
integration surface was. Worth remembering: the part of a change that feels
delicate is not reliably the part that breaks.
