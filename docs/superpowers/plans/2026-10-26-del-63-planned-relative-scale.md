# DEL-63: Planned-relative reported scale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Execution note (ship.md deviation rule):** the change threads one new
> methodology value through two implementations, the variant machinery, every
> shipped profile and a new measurement generator. The lead implements inline
> under strict TDD (review-budget rule). Plan and code each get one scaled
> review round.

**Goal:** Report Raj's relative scale — `rel_psi = norm_psi / mean(norm_psi | Planned)`, so the average Planned colony reads 1 — as a third value of `methodology.second_normalization`, without moving a single number under the two existing values.

**Architecture:** `second_normalization` stops being a boolean exception and becomes a normal reference-pinned enum (`none` / `minmax` / `planned_relative`) generated from the same `REFERENCE_KNOBS` table as every other enum. Production (`delhi_psi/index.py::overall_psi`) and the independent reference (`tests/reference_impl.py::compute_city`) each gain the same three-way dispatch, with the identical formula and the identical two guards. `tests/variants.py` gains one variant, appended last so the committed variants CSVs grow only at the tail; its `second_normalization` override is a SCALAR under a block name, so the four places that expand a variant into config key/value pairs learn the scalar shape. A new `scripts/measure_planned_relative.py` runs `code-2025` with the one value changed, reusing the proven neighbours artifact (the value is downstream of the neighbour structure and is not in the methodology stamp), and writes the committed per-type means block in `docs/data/planned_relative.md`.

**Tech Stack:** Python (uv + pyproject), pandas, geopandas, pytest; fixture generators `scripts/generate_oraculum_fixtures.py` / `generate_messy_fixtures.py`; measurement plumbing `scripts/_measure_common.py`.

**Spec:** `docs/superpowers/specs/2026-10-26-del-63-planned-relative-scale-design.md` (owner-approved 26 Oct 2026, committed `7fa9d8a`). Origin: Rajesh's post-call email of 22 Oct 2026 (4a answered), recorded in `docs/decisions/2026-10-22-raj-call-decisions.md`; ticket DEL-63.

## Global Constraints

- **No shipped profile adopts `planned_relative`.** `tests/fixtures/*/expected_values.csv` (hand-ratified anchors) and `tests/fixtures/*/production/*.csv` must stay **byte-identical**; only the two `variants_expected_values.csv` files grow, **append-only at the tail**.
- **`minmax` is a pure rename of today's `true`.** Every existing number under `code-2025`, `manuscript` and the fifteen variant profiles must be reproduced exactly; the full smoke is the proof.
- **Both implementations change together, formulas byte-identical** (DEL-64 discipline). The reference must never import `delhi_psi.index`.
- **The reference kwarg keeps its name `second_norm`**; it accepts `False` (= `"none"`), `"minmax"`, `"planned_relative"`. Bare `True` and unknown strings raise `ValueError`. The config vocabulary and the reference vocabulary spell every accepted value the same.
- **StrEnum trap:** members are truthy *and* compare equal to their strings. Every dispatch is `== "none"` / `== "planned_relative"` / `!= "none"` — never truthiness. Both implementations validate the value explicitly, so a stray `False` cannot silently mean `minmax`.
- **YAML migration:** `false` (boolean) and the string `"none"` both mean `none`; bare `true` is a `ConfigError` pointing at `minmax`; all sixteen `second_normalization: true` profile YAMLs are rewritten to `minmax` in the same commit that changes the loader. `manuscript.yaml`'s `false` is untouched.
- **TDD:** every new behaviour gets a test written and watched failing first.
- **Record-scaled numbers are `f"{value:.6g}"`**, like every other measurement block; the doc's prose may quote only block values, in backticks (enforced by `assert_prose_numbers_come_from_the_blocks`).

## The measured facts this plan argues from

- **Raj's spot values** (July 2025 `norm_psi` × popdensity per-type means, divided by Planned): Planned `1.00`, UV ≈ `0.86`, JJR ≈ `0.85`, Industrial ≈ `0.85`, RUAC ≈ `0.49`, UAC ≈ `0.39`, JJC ≈ `0.03`. The base column decision (`norm_psi`, not `unnorm_psi`) is pinned to these.
- **`norm_psi` is an affine transform of `unnorm_psi`**, so ratios of means are NOT invariant to min-maxing; the relative scale therefore **inherits the min-max denominator** and the paper must say so.
- **`rel_psi` is a pure per-row rescale**, so `mean(rel_psi | type) = mean(norm_psi | type) / mean(norm_psi | Planned)` up to float64 rounding. The generator measures that identity independently (`rel_vs_ratio_maxdiff_*`) rather than assuming it.
- **The production frame carries `category` at `overall_psi`'s call site**: `pipeline.py:30` (`CATEGORY_COL = categories.CATEGORY_COLUMN`, value `"category"`) and line 280 (`categories.apply_mapping(... out_col=CATEGORY_COL)`); the frame built at 190 (`amounts = universe`) flows through `service_index` (224+) into `overall_psi` (243) with the column present. The reference's frame has no type column; it reads `meta.loc[df.index, "USO_FINAL"]` (`meta = universe.set_index("USO_AREA_U")`, line 421).
- **The oracle planned mean is not degenerate**: on Oraculum `baseline`/`pop` under the `code` rule-set the Planned `norm_psi` mean is **`0.6076369267499986`** (A `0.870046`, D `0.345228`), so the mapped-knob test can drive `second_norm="planned_relative"` without tripping the zero guard.
- **Real code-2025 values, phase3_verify** (popdensity, planned mean `0.044304292394916275`): rel means Planned `1`, UV `0.860416`, JJR `0.845362`, Industrial `0.851319`, RUAC `0.487221`, UAC `0.391285`, JJC `0.0300647`, SDA `0.629172`, Other `2.33318`. Pop has a different ordering (planned mean `0.0109918`; JJR is lowest at `0.299029` and JJC second-highest at `1.95683`); the `UV ≈ 0.86` pin is **popdensity-only**.
- **The oracle's own rel ordering, popdensity** (measured from the reference, planned mean `0.6076369267499986`): IND `1.64572` > Planned `1` > UC `0.506699` > RUAC `0.467657` > RV `0.422163` > JJC `0`. Oraculum's vocabulary is `("Planned", "UC", "JJC", "RV", "RUAC", "IND")` — no UV, no UAC, so the Delhi ordering cannot be pinned on it.
- **`code-2025`'s `paths.neighbors_artifact` is the literal `colonies_neighbors.joblib`** (not profile-suffixed) and `type_col` is `USO_FINAL`; `~/delhi_data/phase3_verify` holds that artifact plus both `code-2025` output CSVs. The derived run's rendered artifact name is `colonies_neighbors_planned-relative.joblib` (hyphen: `PROFILE = "planned-relative"` fills the template).
- **Pandas `to_csv` writes every frame column**, so `rel_psi` reaches the output CSV with no writer change (`delhi_psi/io.py::write_outputs`).
- **`check_oraculum_invariants.check` only inspects `*_pcen` groups**, so a `rel_psi` group with values above 1 (e.g. Other `2.33318`) trips no invariant.

---

### Task 1: `second_normalization` enum + production dispatch + every caller migrated

**Files:**
- Modify: `delhi_psi/config.py` (knob table 54, comment 62–63, `ENUM_KEYS` 64–77, enum defs 88–100, `ENUMS` 102–115, `RESERVED_VALUES` 118–126, dataclass field 277, `_coerce_enum` 363–375, delete `_bool` 378–383, `_methodology` return 591–603)
- Modify: `delhi_psi/index.py` (imports 17–24, constants beside `ABSENT_NEIGHBOR` at line 27, `overall_psi` 572–592)
- Modify: `scripts/generate_production_fixtures.py` (`metric_columns` 51–72)
- Modify: `tests/test_index.py` (172, 179, 197–208, 331, 373) + new tests after 209
- Modify: `tests/test_reference_impl.py` (620, 633, 712, 726; imports in both synthetic tests)
- Modify: `tests/test_config.py` (`MINIMAL` 42 + new tests)
- Modify: `tests/test_production_fixtures.py` (64, 74, 116 + new test)
- Modify: `tests/test_profiles_match_reference.py` (`metrics_for` 59–62)
- Modify: `docs/methodology-config.md` (table line 32; example line 467; migration note after line 58)
- Modify: the sixteen profile YAMLs that say `second_normalization: true` (listed in Step 6)

- [ ] **Step 0: Remove the recon scratch**

  ```bash
  rm -f .del63_regions.txt .del63_regions0.txt .del63_regions4.txt
  git status --short --branch
  ```

  Expected: branch `del-63-planned-relative-scale`; no modified files. The
  only untracked path is this plan file
  (`docs/superpowers/plans/2026-10-26-del-63-planned-relative-scale.md`).

- [ ] **Step 1: RED — write the new config tests**

  In `tests/test_config.py`, change `MINIMAL`'s line 42 to `  second_normalization: minmax`, then append beside the other methodology-loader tests:

  ```python
  # --- DEL-63: second_normalization is a three-value enum ---------------
  def test_second_normalization_loads_its_three_strings(tmp_path):
      for value in ("none", "minmax", "planned_relative"):
          text = MINIMAL.replace("second_normalization: minmax",
                                 f"second_normalization: {value}")
          cfg = load_config(write(tmp_path, text, name=f"{value}.yaml"),
                            data_dir=str(tmp_path))
          assert cfg.methodology.second_normalization == value, value


  def test_second_normalization_false_still_means_none(tmp_path):
      """The exclusion.stage precedent: a YAML boolean stays acceptable at
      the one value it can mean without ambiguity."""
      text = MINIMAL.replace("second_normalization: minmax",
                             "second_normalization: false")
      cfg = load_config(write(tmp_path, text, name="none.yaml"),
                        data_dir=str(tmp_path))
      assert cfg.methodology.second_normalization == "none"


  def test_bare_true_second_normalization_is_refused_with_the_migration(
          tmp_path):
      """An anonymous boolean is the ambiguity this change removes: `true`
      used to mean min-max, so it must name minmax loudly rather than drift
      into one of two new meanings."""
      text = MINIMAL.replace("second_normalization: minmax",
                             "second_normalization: true")
      with pytest.raises(ConfigError) as exc:
          load_config(write(tmp_path, text, name="bare-true.yaml"))
      message = str(exc.value)
      assert "minmax" in message and "planned_relative" in message


  def test_an_unknown_second_normalization_is_refused(tmp_path):
      text = MINIMAL.replace("second_normalization: minmax",
                             "second_normalization: min_max")
      with pytest.raises(ConfigError, match="not allowed"):
          load_config(write(tmp_path, text, name="typo.yaml"),
                      data_dir=str(tmp_path))
  ```

- [ ] **Step 2: RED — write the production dispatch tests**

  In `tests/test_index.py`, migrate the six existing call sites (`172`, `179`,
  `205`, `207`, `331`, `373`: `second_normalization=True` → `"minmax"`,
  `False` → `"none"`, rename the `..._is_false` test to `..._is_none`, and
  update the docstring at 197–200 to say "when second_normalization='none'").
  Then append after `test_overall_psi_second_normalization_propagates_the_guard`:

  ```python
  # --- DEL-63: planned_relative ------------------------------------------
  def test_overall_psi_rejects_a_bare_bool_second_normalization():
      """StrEnum members are truthy AND equal their strings, and `False`
      would compare equal to neither "none" nor "planned_relative" — a
      silent fall-through to min-max. Both booleans must fail loudly."""
      frame = pd.DataFrame({"a_idx": [0.0, 1.0]})
      for value in (True, False):
          with pytest.raises(ValueError, match="unknown second_normalization"):
              index.overall_psi(frame, second_normalization=value)


  def test_planned_relative_rescales_norm_psi_by_the_planned_mean():
      frame = gpd.GeoDataFrame(
          {"USO_AREA_U": ["A", "B", "C"],
           "category": ["Planned", "Planned", "UAC"],
           "a_idx": [0.0, 1.0, 0.5]},
          geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
          crs="EPSG:7760")
      got = index.overall_psi(frame, second_normalization="planned_relative")

      # norm_psi is unchanged from `minmax` — the rel scale divides it.
      minmax_only = index.overall_psi(frame, second_normalization="minmax")
      assert list(got["norm_psi"]) == list(minmax_only["norm_psi"])

      planned = got.loc[got["category"] == "Planned", "norm_psi"]
      assert planned.mean() == pytest.approx(0.5, abs=1e-12)
      assert list(got["rel_psi"]) == pytest.approx([0.0, 2.0, 1.0],
                                                   abs=1e-12)
      # The category mean is 1 by construction; individual Planned rows
      # are not (A is 0 and B is 2 here).
      assert got.loc[got["category"] == "Planned", "rel_psi"].mean() == (
          pytest.approx(1.0, abs=1e-12))


  def test_planned_relative_refuses_a_frame_with_no_planned_rows():
      frame = gpd.GeoDataFrame(
          {"USO_AREA_U": ["A", "B"], "category": ["UAC", "JJC"],
           "a_idx": [0.0, 1.0]},
          geometry=[Point(0, 0), Point(1, 0)], crs="EPSG:7760")
      with pytest.raises(ValueError, match="'Planned'"):
          index.overall_psi(frame, second_normalization="planned_relative")


  def test_planned_relative_refuses_a_zero_planned_mean():
      """A degenerate universe: both Planned rows sit at the global minimum
      of unnorm_psi, so the divisor is exactly 0.0."""
      frame = gpd.GeoDataFrame(
          {"USO_AREA_U": ["A", "B", "C"],
           "category": ["Planned", "Planned", "UAC"],
           "a_idx": [0.0, 0.0, 1.0]},
          geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
          crs="EPSG:7760")
      with pytest.raises(ValueError, match="mean is 0.0"):
          index.overall_psi(frame, second_normalization="planned_relative")
  ```

  Extend `test_overall_psi_omits_norm_psi_when_second_normalization_is_none`
  with `assert "rel_psi" not in got.columns`.

- [ ] **Step 3: Watch them fail**

  Run: `uv run pytest -q tests/test_config.py -k second_normalization`
  Expected: **three** of the four new config tests FAIL — the three-value
  loop dies on `minmax` with `ConfigError` (the table still holds
  `{True: True, False: False}`), the `false` test gets `False` back rather
  than the string `"none"`, and bare `true` loads successfully. The fourth
  (`..._unknown_...`) already passes, because today's `_bool` also refuses
  `"min_max"` with "not allowed"; it is the post-change guard.
  Changing `MINIMAL` to `minmax` also reddens every other `MINIMAL`-based
  test in the module until Step 4 lands — that is expected, not a finding.

  Run: `uv run pytest -q tests/test_index.py -k "overall_psi"`
  Expected: the new dispatch tests FAIL. `"minmax"` is truthy, so the old
  branch still runs min-max; `planned_relative` is also truthy, so the old
  branch emits `norm_psi` and the test then fails with
  `KeyError: 'rel_psi'` at its own assertion (the implementation never
  touches `category` yet). The bare-bool guard does not exist, so
  `pytest.raises` fails; neither guard test raises.

- [ ] **Step 4: Config — make the knob a normal enum**

  1. Replace line 54:

  ```python
      "methodology.second_normalization": {"none": "none",
                                           "minmax": "minmax",
                                           "planned_relative": "planned_relative"},
  ```

  2. Delete the bool-exception comment (lines 62–63), leaving `ENUM_KEYS`
  starting immediately after the blank line. Add the key between
  `"methodology.roads"` and `"methodology.exclusion.stage"`:

  ```python
      "methodology.roads",
      "methodology.second_normalization",
      "methodology.exclusion.stage",
  ```

  3. After `RoadsFormula = _make_enum("RoadsFormula", "methodology.roads")`:

  ```python
  SecondNormalization = _make_enum(
      "SecondNormalization", "methodology.second_normalization")
  ```

  4. Add to `ENUMS` after `"methodology.roads": RoadsFormula,`:

  ```python
      "methodology.second_normalization": SecondNormalization,
  ```

  5. Add to `RESERVED_VALUES` (before `"outputs.denominators[]"`, or after —
  order is irrelevant):

  ```python
      "methodology.second_normalization": {
          True:
              "no longer allowed — it used to mean min-max. Write `minmax` "
              "(or `planned_relative`) instead.",
      },
  ```

  6. Dataclass field 277:

  ```python
      second_normalization: SecondNormalization
  ```

  7. Delete `_bool` (lines 378–383). Check it has no other caller:
  `grep -n "_bool(" delhi_psi/config.py` must print nothing after the edit.

  8. In `_methodology`, immediately before `return MethodologyConfig(`:

  ```python
      second_norm = _require(raw, "second_normalization", "methodology")
      # DEL-63: YAML `false` keeps meaning "off" (the exclusion.stage
      # precedent). Every other value is matched as an exact string, and a
      # bare `true` is refused by RESERVED_VALUES so an anonymous boolean
      # can never silently pick one of the two new meanings.
      if second_norm is False:
          second_norm = "none"
  ```

  and replace the `_bool(...)` argument with:

  ```python
          second_normalization=_coerce_enum(
              "methodology.second_normalization", second_norm),
  ```

- [ ] **Step 5: Production — the three-way dispatch**

  1. `delhi_psi/index.py` imports: add after `from shapely import STRtree`:

  ```python
  from delhi_psi.categories import CATEGORY_COLUMN
  ```

  (`categories` is not `config`; the module's "never imports delhi_psi.config"
  rule stands. `categories.py` imports only `delhi_psi.validate`.)

  2. Beside `ABSENT_NEIGHBOR = ("swallowed", "contributes")`:

  ```python
  SECOND_NORMALIZATIONS = ("none", "minmax", "planned_relative")
  ```

  3. Replace `overall_psi` (572–592) with:

  ```python
  def overall_psi(polygon_gdf, *, second_normalization, transform_form="none",
                  transform_stage=None):
      """Eq. 1: the mean of every `*_idx` column, plus the optional second
      normalization (`norm_psi`) and, under `planned_relative`, the reported
      `rel_psi`; each column is absent when the value that names it is off.

      transform_stage="psi" (DEL-34): the composite `unnorm_psi` is
      transformed BEFORE the second normalization — the 2021 notebook's
      `log(unnorm_psi + 1)`. It re-spreads the final score but cannot undo
      per-service compression that has already happened (spec § 3).

      DEL-63: `planned_relative` emits BOTH columns. `rel_psi` is a pure
      per-row rescale of `norm_psi` by one constant — the mean of `norm_psi`
      over the `Planned` rows of THIS frame — so the reported per-category
      mean of `rel_psi` IS the type ratio Raj asked for (linearity; no
      second aggregation step to get wrong). Because `norm_psi` is an affine
      transform of `unnorm_psi`, the relative scale inherits the min-max
      denominator; that is a property of the reported number, not a bug.

      A StrEnum member is truthy AND == its literal string, so the dispatch
      compares values, never truthiness; a bare bool is rejected rather than
      falling through to `minmax`.
      """
      if second_normalization not in SECOND_NORMALIZATIONS:
          raise ValueError(
              f"unknown second_normalization {second_normalization!r}; "
              f"allowed values: {list(SECOND_NORMALIZATIONS)}")
      _check_transform(transform_form, transform_stage)
      out = polygon_gdf.copy()
      idx_columns = [column for column in out.columns
                     if column.endswith("_idx")]
      out["unnorm_psi"] = out[idx_columns].mean(axis=1)
      if transform_stage == "psi":
          out["unnorm_psi"] = out["unnorm_psi"].map(
              lambda value: _apply_transform(value, transform_form))
      if second_normalization == "none":
          return out
      out = minmax(out, source_col="unnorm_psi", target_col="norm_psi")
      if second_normalization == "planned_relative":
          planned = out.loc[out[CATEGORY_COLUMN] == "Planned", "norm_psi"]
          if planned.empty:
              raise ValueError(
                  f"planned_relative needs at least one row whose "
                  f"{CATEGORY_COLUMN!r} is 'Planned': this frame has "
                  f"{len(out)} rows and none of them is, so there is no "
                  "denominator to divide by")
          planned_mean = float(planned.mean())
          if planned_mean == 0.0:
              raise ValueError(
                  "planned_relative divides by the mean norm_psi of the "
                  f"{len(planned)} Planned rows, and that mean is 0.0 — "
                  "every Planned settlement has the minimum unnormalized "
                  "PSI, so the relative scale is undefined")
          out["rel_psi"] = out["norm_psi"] / planned_mean
      return out
  ```

- [ ] **Step 6: Migrate every caller and every profile**

  - `scripts/generate_production_fixtures.py::metric_columns`:

  ```python
      if second_normalization != "none":
          columns.append("norm_psi")
      if second_normalization == "planned_relative":
          columns.append("rel_psi")
      columns.append("population")
  ```

  - `tests/test_production_fixtures.py`: 64 → `metric_columns(second_normalization="minmax")`; 74 → `metric_columns(second_normalization="none")` and additionally assert `"rel_psi" not in` it; 116 → `second_normalization="minmax"`. Append:

  ```python
  def test_planned_relative_emits_both_second_columns():
      """DEL-63: one run produces both tables — the base column and the
      relative one — and no other value emits `rel_psi`."""
      cols = metric_columns(second_normalization="planned_relative")
      assert cols[-5:] == ["unnorm_psi", "norm_psi", "rel_psi",
                           "population", "area_km2"]
      assert "rel_psi" not in metric_columns(second_normalization="minmax")
      assert "rel_psi" not in metric_columns(second_normalization="none")
  ```

  - `tests/test_reference_impl.py`, both synthetic tests: add
  `SecondNormalization` to each local `from delhi_psi.config import (...)`
  block and replace `second_normalization=True` (620, 712) with
  `second_normalization=SecondNormalization.MINMAX`; replace
  `second_norm=True` (633, 726) with `second_norm="minmax"`.

  - `tests/test_profiles_match_reference.py::metrics_for` — **moved into this
  task**, not Task 3: after the enum change a `none` member is truthy, so
  the old line would compare `norm_psi` for `manuscript` against a frame
  that no longer carries it. Write the final logic now; `METRIC_MAP` gains
  `rel_psi` in Task 3, so the `rel_psi` skip is a no-op until then:

  ```python
  def metrics_for(profile):
      cfg = load_config(profile)
      second = cfg.methodology.second_normalization
      skip = set() if second != "none" else {"norm_psi", "rel_psi"}
      if second != "planned_relative":
          skip.add("rel_psi")
      return {k: v for k, v in METRIC_MAP.items() if k not in skip}
  ```

  - The sixteen `second_normalization: true` profile YAMLs → `minmax`:
  `adj-touch` (54), `band-0km` (54), `band-1km` (51), `band-5km` (51),
  `band-10km` (51), `code-2025` (89), `decay-boundary` (63),
  `decay-exp2km` (62), `decay-exp5km` (62), `decay-none` (68),
  `decay-power05` (62), `decay-power2` (67), `services-contested` (63),
  `services-no-bank` (60), `services-no-ration` (60),
  `services-uncontested` (70). `code-2025`'s comment becomes:

  ```yaml
    second_normalization: minmax      # minmax | none | planned_relative — norm_psi under the first two, plus rel_psi under planned_relative
  ```

  `manuscript.yaml:42` (`false`) is left untouched.

  - `docs/methodology-config.md`: line 467 example → `second_normalization: minmax`.
  Line 32's cell recommends `true` twice ("today" column and the ratified
  sentence); replace **every** boolean mention in that row — `true` →
  `minmax`, `false` → `none` — then append to the context column:
  `Values: `none`, `minmax` (today's `true`), `planned_relative` (emits `norm_psi` **and** `rel_psi`; Raj's 22 Oct answer, DEL-63).`
  (Leaving `true` anywhere in the cell would recommend a value that now
  raises `ConfigError`.)
  Then insert after line 58 (`...DECIDES what the ratified profile carries.`):

  ```markdown

  `second_normalization` is now a **three-valued enum, not a boolean**
  (DEL-63): `minmax` is today's `true` (emits `norm_psi`), `none` is today's
  `false` (no second column), and `planned_relative` emits `norm_psi` **and**
  `rel_psi` = `norm_psi` / the mean `norm_psi` over the Planned rows, so the
  average Planned colony reads 1 (`docs/data/planned_relative.md`). YAML
  `false` is still accepted and means `none`; a bare `true` is **refused**
  with a message pointing at `minmax`, because an anonymous boolean is the
  ambiguity this change removes.
  ```

- [ ] **Step 7: Run the targeted suite**

  ```bash
  uv run pytest -q tests/test_config.py tests/test_index.py \
      tests/test_production_fixtures.py tests/test_reference_impl.py \
      tests/test_profiles_match_reference.py tests/test_oracle.py
  ```

  Expected: PASS. The reference still uses truthiness, so `minmax` and
  `True` both take the min-max branch; no expected or production fixture
  bytes change. Confirm:

  ```bash
  git status --porcelain -- tests/fixtures/
  ```

  Expected: empty.

- [ ] **Step 8: Commit**

  ```bash
  git add delhi_psi tests scripts/generate_production_fixtures.py docs/methodology-config.md
  git commit -m "feat(config): second_normalization is a three-value enum; production emits rel_psi (DEL-63)"
  ```

### Task 2: Reference implementation — same three-way rule, same guards

**Files:**
- Modify: `tests/reference_impl.py` (constants, validation after 392, second-norm block 523–532, `RULESETS` line 35)
- Modify: `tests/test_reference_impl.py` (new tests beside the synthetic-city section)

- [ ] **Step 1: RED — reference pins and guards**

  Append to `tests/test_reference_impl.py` after the synthetic-city tests
  (the module fixtures `settlements`, `barriers`, `services` are already
  defined at the top):

  ```python
  # --- DEL-63: the reference's planned_relative ----------------------
  def _planned_ids(settlements, index):
      types = settlements.set_index("USO_AREA_U")["USO_FINAL"]
      return index[types.loc[index] == "Planned"]


  def test_reference_planned_relative_divides_norm_psi_by_the_planned_mean(
          settlements, barriers, services):
      df = compute_city(settlements, services, barriers,
                        **dict(RULESETS["code"], scenario="baseline",
                               denom="pop",
                               second_norm="planned_relative"))
      planned = df.loc[_planned_ids(settlements, df.index), "norm_psi"]
      # The measured oracle value this pin argues from (A and D are Planned).
      assert planned.mean() == pytest.approx(0.6076369267499986, abs=1e-12)
      assert list(df["rel_psi"]) == pytest.approx(
          list(df["norm_psi"] / planned.mean()), abs=1e-15)
      assert df.loc[planned.index, "rel_psi"].mean() == pytest.approx(
          1.0, abs=1e-12)


  def test_reference_none_emits_neither_second_column(settlements, barriers,
                                                      services):
      df = compute_city(settlements, services, barriers,
                        **dict(RULESETS["code"], scenario="baseline",
                               denom="pop", second_norm=False))
      assert "norm_psi" not in df.columns
      assert "rel_psi" not in df.columns


  def test_reference_rejects_a_bare_true_second_norm(settlements, barriers,
                                                     services):
      with pytest.raises(ValueError, match="minmax"):
          compute_city(settlements, services, barriers,
                       **dict(RULESETS["code"], scenario="baseline",
                              denom="pop", second_norm=True))


  def test_reference_rejects_an_unknown_second_norm(settlements, barriers,
                                                    services):
      with pytest.raises(ValueError, match="unknown second normalization"):
          compute_city(settlements, services, barriers,
                       **dict(RULESETS["code"], scenario="baseline",
                              denom="pop", second_norm="sideways"))


  def test_reference_planned_relative_refuses_a_city_with_no_planned_rows(
          settlements, barriers, services):
      """A and D are the only Planned settlements, and `False` means the
      POST-neighbour stage — they are scored out, but their links remain in
      other settlements' neighbour lists. Every Planned row is gone from the
      frame the normalisation divides, which is all the guard needs.

      `RULESETS["ideal"]` on purpose: the code rule-set's GLOBAL barrier
      severs every link of A (Oraculum's only bank owner), so a frame
      without A and D would be all-zero on `bank_pcen` and the min-max
      guard would fire first. The ideal rule-set's `pair` barrier keeps
      A's links and its `contributes` lets the excluded settlements lend,
      so every service stays non-degenerate and the frame reaches the
      division."""
      with pytest.raises(ValueError, match="'Planned'"):
          compute_city(settlements, services, barriers,
                       **dict(RULESETS["ideal"], scenario="no_planned",
                              denom="pop",
                              second_norm="planned_relative"),
                       scenarios={"no_planned":
                                  (frozenset({"A", "D"}), False)})


  def test_reference_planned_relative_refuses_a_zero_planned_mean():
      """Three ISOLATED settlements (disjoint boxes, so no neighbour term):
      the two Planned ones own nothing at all, the UC one owns one of every
      point service plus a road. Every Planned value is the global minimum,
      so the min-max makes their norm_psi exactly 0 and the divisor is 0.0.
      """
      import geopandas as gpd
      from shapely.geometry import LineString, Point, box

      from tests.reference_impl import POINT_SERVICES

      settlements = gpd.GeoDataFrame(
          {"USO_AREA_U": ["P1", "P2", "U"],
           "USO_FINAL": ["Planned", "Planned", "UC"],
           "population": [100.0, 100.0, 100.0],
           "area_km2": [1.0, 1.0, 1.0]},
          geometry=[box(0, 0, 100, 100), box(10_000, 0, 10_100, 100),
                    box(20_000, 0, 20_100, 100)],
          crs="EPSG:7760")
      services = {
          name: gpd.GeoDataFrame({"service": [name]},
                                 geometry=[Point(20_050, 50)],
                                 crs="EPSG:7760")
          for name in POINT_SERVICES}
      services["road"] = gpd.GeoDataFrame(
          {"service": ["road"]},
          geometry=[LineString([(20_000, 50), (20_100, 50)])],
          crs="EPSG:7760")

      with pytest.raises(ValueError, match="mean is 0.0"):
          compute_city(settlements, services, None,
                       adjacency_rule="bbox", barrier_rule="global",
                       roads_formula="decayed", scenario="none", denom="pop",
                       second_norm="planned_relative",
                       absent_neighbor_contribution="swallowed",
                       scenarios={"none": (frozenset(), False)})
  ```

  Update the two synthetic tests' omission comments: replace
  "Every METRIC_MAP column exists on both sides ... the comparison cannot
  pass by omission." with a note that `rel_psi` exists only under
  `planned_relative`, and that the loop below asserts the two frames agree
  about every column's PRESENCE as well as its values.

- [ ] **Step 2: Watch them fail**

  Run: `uv run pytest -q tests/test_reference_impl.py -k "planned_relative or second_norm or neither_second"`
  Expected: FAIL — `second_norm="planned_relative"` is truthy so the old
  branch emits `norm_psi` and no `rel_psi` (`KeyError`); bare `True` and
  `"sideways"` are accepted; the no-Planned city produces a frame instead of
  raising; the zero-mean city produces `NaN`/no raise.

- [ ] **Step 3: Implement**

  1. Add beside `AGGREGATION_RULES` (the constants block above
  `compute_city`):

  ```python
  SECOND_NORMALIZATIONS = ("none", "minmax", "planned_relative")
  ```

  Update the module docstring's knob list to say `second_norm`
  (`False`/`"none"`/`"minmax"`/`"planned_relative"`).

  2. Insert after the `aggregation_rule` check (line 392):

  ```python
      # DEL-63: `second_norm` is the reference's name for the config's
      # `methodology.second_normalization`, and every accepted value is
      # spelled the same in both vocabularies. `False` still means "off";
      # a bare `True` used to mean the min-max, and now names it instead.
      if second_norm is True:
          raise ValueError(
              "second_norm=True is no longer allowed — it used to mean the "
              "min-max second normalization. Pass 'minmax' (or "
              "'planned_relative'); False still means 'none'.")
      if second_norm is False:
          second_norm = "none"
      if second_norm not in SECOND_NORMALIZATIONS:
          raise ValueError(
              f"unknown second normalization {second_norm!r}; allowed values: "
              f"{list(SECOND_NORMALIZATIONS)} (or False for 'none')")
  ```

  3. Replace the second-normalization block (523–532) with:

  ```python
      if second_norm == "none":
          return df
      p = df["psi_eq1"]
      lo, hi = p.min(), p.max()
      if hi == lo:
          raise ValueError(
              f"min-max of 'psi_eq1' is undefined: all {len(p)} values "
              f"equal {lo!r} (hi == lo), so Eq. 2 divides 0/0 — every "
              f"settlement's unnormalized PSI is the same")
      df["norm_psi"] = (p - lo) / (hi - lo)
      if second_norm == "planned_relative":
          # The reference frame has no type column; `meta` is the universe
          # indexed by settlement id and carries `USO_FINAL`.
          planned = df.loc[meta.loc[df.index, "USO_FINAL"] == "Planned",
                           "norm_psi"]
          if planned.empty:
              raise ValueError(
                  "planned_relative needs at least one row whose USO_FINAL "
                  f"is 'Planned': this frame has {len(df)} rows and none of "
                  "them is, so there is no denominator to divide by")
          planned_mean = float(planned.mean())
          if planned_mean == 0.0:
              raise ValueError(
                  "planned_relative divides by the mean norm_psi of the "
                  f"{len(planned)} Planned rows, and that mean is 0.0 — "
                  "every Planned settlement has the minimum unnormalized "
                  "PSI, so the relative scale is undefined")
          df["rel_psi"] = df["norm_psi"] / planned_mean
      return df
  ```

  4. `RULESETS["code"]`: `second_norm=True` → `second_norm="minmax"`.
  `RULESETS["ideal"]` keeps `second_norm=False` (it is a caller of the
  accepted `False` spelling, and keeps `expected_values.csv` unchanged).

- [ ] **Step 4: Green + the reference halves agree**

  ```bash
  uv run pytest -q tests/test_reference_impl.py tests/test_profiles_match_reference.py \
      tests/test_oracle.py tests/test_variants_match_reference.py
  ```

  Expected: PASS. `test_every_mapped_knob_is_one_the_reference_actually_implements`
  now drives `second_norm="planned_relative"` once (oracle Planned mean
  `0.6076369267499986`, no zero-division), and `expected_values.csv` /
  `production/*.csv` are byte-unchanged.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/reference_impl.py tests/test_reference_impl.py
  git commit -m "feat(reference): planned_relative mirrors the production formula, with guards (DEL-63)"
  ```

### Task 3: Variant machinery — one scalar override, one appended row

**Files:**
- Modify: `tests/variants.py` (append last in `VARIANTS`)
- Modify: `tests/reference_impl.py` (`VARIANT_KNOBS`, `_variant_overrides`)
- Modify: `tests/oraculum_fixtures.py` (`oracle_profile_path`, `variant_methodology`)
- Modify: `tests/test_config.py` (variant-accepts test)
- Modify: `tests/test_variant_rules.py` (the table-vs-reference loop)
- Modify: `tests/test_profiles_match_reference.py` (`METRIC_MAP` only —
  `metrics_for` lands in Task 1)
- Modify: `tests/test_variants_match_reference.py` (presence agreement + new pins)
- Regenerate: `tests/fixtures/{oraculum,messy}/variants_expected_values.csv`

- [ ] **Step 1: RED — the variant pin**

  Append to `tests/test_variants_match_reference.py`:

  ```python
  # --- DEL-63: the reported relative scale -------------------------------
  def test_the_planned_relative_variant_reads_one_for_the_planned_type():
      """1 = the average Planned colony, by construction; and the oracle's
      ordering survives the rescale. The per-settlement values are compared
      against the reference by
      test_production_matches_the_reference_on_each_variant (METRIC_MAP now
      carries rel_psi). Oraculum's vocabulary is
      ("Planned", "UC", "JJC", "RV", "RUAC", "IND") — the Delhi UV≈0.86 pin
      lives in the real-data block test (Task 4), never here."""
      got = produced(ORACULUM, "planned_relative", "popdensity")
      assert {"norm_psi", "rel_psi"} <= set(got.columns)
      means = got.groupby("category")["rel_psi"].mean()
      assert means["Planned"] == pytest.approx(1.0, abs=1e-12)
      assert means["IND"] > means["Planned"] > means["UC"]
      assert means["UC"] > means["RUAC"] > means["RV"] > means["JJC"]
      # The min-max floor stays the floor: JJC's norm_psi is 0.
      assert means["JJC"] == pytest.approx(0.0, abs=1e-12)
  ```

  In `tests/test_reference_impl.py`, beside the variants-CSV shape tests:

  ```python
  @pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
  def test_the_planned_relative_variant_carries_both_columns(city):
      frame = pd.read_csv(city.fixtures / "variants_expected_values.csv")
      block = frame[frame["rule"] == "planned_relative"]
      assert {"norm_psi", "rel_psi"} <= set(block["metric"])
      assert "rel_psi" not in set(
          frame.loc[frame["rule"] != "planned_relative", "metric"])
  ```

- [ ] **Step 2: Watch them fail**

  Run: `uv run pytest -q tests/test_variants_match_reference.py -k planned_relative`
  Expected: FAIL — `KeyError: 'planned_relative'` (no such variant yet).

- [ ] **Step 3: Add the variant and teach the four expanders the scalar shape**

  1. `tests/variants.py`, appended LAST after
  `"aggregation_mean_rank_log1p_psi"` (so the committed CSV's existing rows
  stay a byte-prefix):

  ```python
      # DEL-63: the reported scale Raj asked for. `rel_psi` = `norm_psi` /
      # the mean over the Planned rows, so Planned's category mean is 1.
      # Unlike every row above, this states a SCALAR under its block name:
      # `second_normalization` is one methodology key, not a sub-block.
      "planned_relative": {"second_normalization": "planned_relative"},
  ```
  Update the module docstring: a variant's value is either a full block
  mapping or (DEL-63) a scalar for a top-level methodology key.

  2. `tests/reference_impl.py::VARIANT_KNOBS` gains:

  ```python
      ("second_normalization", "second_normalization"): "second_norm",
  ```

  and `_variant_overrides` expands the scalar shape:

  ```python
  def _variant_overrides(spec):
      out = {}
      for block, mapping in spec.items():
          pairs = ([(block, mapping)] if isinstance(mapping, str)
                   else mapping.items())
          for key, value in pairs:
              ...
  ```

  (Keep the existing body and error messages; only the iteration changes.)

  3. `tests/oraculum_fixtures.py::oracle_profile_path`:

  ```python
      for block, values in (methodology_overrides or {}).items():
          raw["methodology"][block] = (values if isinstance(values, str)
                                       else dict(values))
  ```

  Update its docstring: an override value is a full block mapping OR a
  scalar for a single top-level methodology key.

  4. `tests/oraculum_fixtures.py::variant_methodology`, after the
  `aggregation` branch and before `return methodology`:

  ```python
      if "second_normalization" in spec:
          methodology = replace(
              methodology,
              second_normalization=SecondNormalization(
                  spec["second_normalization"]))
  ```

  and add `SecondNormalization` to the local `from delhi_psi.config import
  (...)` list.

  5. `tests/test_config.py::test_every_variant_block_is_one_the_loader_accepts`:
  add `("second_normalization", "second_normalization"):
  "methodology.second_normalization"` to `enum_key`; expand both loops:

  ```python
      for block, values in VARIANTS[variant].items():
          pairs = ([(block, values)] if isinstance(values, str)
                   else values.items())
          for key, value in pairs:
              if (block, key) in enum_key:
                  assert value in REFERENCE_KNOBS[enum_key[(block, key)]], \
                      (variant, block, key)

      raw = yaml.safe_load(MINIMAL)
      for block, values in VARIANTS[variant].items():
          raw["methodology"][block] = (values if isinstance(values, str)
                                       else dict(values))
      path = write(tmp_path, yaml.safe_dump(raw, sort_keys=False))
      cfg = load_config(path, data_dir=str(tmp_path))
      for block, values in VARIANTS[variant].items():
          if isinstance(values, str):
              assert getattr(cfg.methodology, block) == values, (variant, block)
              continue
          loaded = getattr(cfg.methodology, block)
          for key, value in values.items():
              assert getattr(loaded, key) == value, (variant, block, key)
  ```

  6. `tests/test_variant_rules.py::test_variant_rulesets_are_the_code_base_plus_the_table`:
  same scalar expansion:

  ```python
          for block, mapping in spec.items():
              pairs = ([(block, mapping)] if isinstance(mapping, str)
                       else mapping.items())
              for key, value in pairs:
  ```

  7. `tests/test_profiles_match_reference.py` — **`METRIC_MAP` only**:
  `metrics_for` was already rewritten in Task 1 Step 6, and until this line
  lands its `rel_psi` skip is a harmless no-op.

  ```python
  METRIC_MAP = {
      ...
      "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
      "rel_psi": "rel_psi",
  }
  ```

  8. Presence agreement in the three METRIC_MAP loops — `tests/test_variants_match_reference.py`
  (the in-memory loop) and the two synthetic tests in
  `tests/test_reference_impl.py`:

  ```python
          for prod_col, metric in METRIC_MAP.items():
              assert (prod_col in got.columns) == (metric in exp.columns), (
                  prod_col, metric)
              if prod_col not in got.columns:
                  continue
              for sid in exp.index:
                  ...
  ```

  This is the "cannot pass by omission" guard: a missing `rel_psi` on one
  side and not the other fails at the presence assertion, while the
  min-max-only rows legitimately skip it on both.

- [ ] **Step 4: Regenerate the two variants CSVs**

  ```bash
  uv run python scripts/generate_oraculum_fixtures.py
  uv run python scripts/generate_messy_fixtures.py
  ```

  Then prove the blast radius:

  ```bash
  git status --porcelain -- tests/fixtures/
  ```

  Expected: ONLY `tests/fixtures/oraculum/variants_expected_values.csv` and
  `tests/fixtures/messy/variants_expected_values.csv` modified.

  Prove append-only growth (existing rows unmoved):

  ```bash
  mkdir -p /tmp/opencode
  for city in oraculum messy; do
    git show HEAD:tests/fixtures/$city/variants_expected_values.csv > /tmp/opencode/$city-old.csv
    uv run python - "$city" <<'EOF'
  import sys
  city = sys.argv[1]
  old = open(f"/tmp/opencode/{city}-old.csv", "rb").read()
  new = open(f"tests/fixtures/{city}/variants_expected_values.csv", "rb").read()
  assert new.startswith(old), f"{city}: existing rows moved"
  print(city, "appended", len(new) - len(old), "bytes")
  EOF
  done
  ```

  Also confirm the new block's metric set is exactly the old one plus
  `rel_psi`:

  ```bash
  uv run python - <<'EOF'
  import pandas as pd
  for city in ("oraculum", "messy"):
      old = pd.read_csv(f"/tmp/opencode/{city}-old.csv")
      new = pd.read_csv(f"tests/fixtures/{city}/variants_expected_values.csv")
      assert set(old["rule"]) < set(new["rule"]) == set(new["rule"])
      assert "planned_relative" in set(new["rule"]) - set(old["rule"])
      print(city, sorted(set(new.loc[new["rule"] == "planned_relative", "metric"])))
  EOF
  ```

  Expected: each prints the full metric set INCLUDING `norm_psi` and
  `rel_psi`.

- [ ] **Step 5: Green**

  ```bash
  uv run pytest -q tests/test_variants_match_reference.py tests/test_variant_rules.py \
      tests/test_config.py tests/test_reference_impl.py tests/test_profiles_match_reference.py
  ```

  Expected: PASS, including both new parametrizations over both cities.

- [ ] **Step 6: Commit**

  ```bash
  git add tests/variants.py tests/reference_impl.py tests/oraculum_fixtures.py \
      tests/test_config.py tests/test_variant_rules.py \
      tests/test_profiles_match_reference.py tests/test_variants_match_reference.py \
      tests/test_reference_impl.py tests/fixtures/oraculum/variants_expected_values.csv \
      tests/fixtures/messy/variants_expected_values.csv
  git commit -m "test(variants): the planned_relative variant — scalar override machinery + regenerated CSVs (DEL-63)"
  ```

### Task 4: The relative-means generator and its committed block

**Files:**
- Create: `scripts/measure_planned_relative.py`
- Create: `tests/test_measure_planned_relative.py`
- Create: `docs/data/planned_relative.md`
- Modify: `tests/test_measure_common.py` (`DOCUMENT_GENERATORS` 426–433 —
  the new document carries a fenced block, so the coverage test requires it)
- Modify: `docs/data/psi_columns.md` (pointer in "What this does not decide")

- [ ] **Step 1: RED — write the test module first**

  Create `tests/test_measure_planned_relative.py`:

  ```python
  """The relative-scale block must be exactly what the generator emits.

  `docs/data/planned_relative.md` states Raj's reported scale (DEL-63): each
  type's mean `rel_psi`, where 1 is the average Planned colony. The pure
  tests pin the arithmetic and the guards; the `needs_data` test proves a
  fresh run reproduces the committed block byte for byte.
  """
  import subprocess
  import sys
  from pathlib import Path

  import pandas as pd
  import pytest

  from scripts._measure_common import FENCE, parse_block
  from scripts.measure_planned_relative import (
      DENOMINATORS, REPORTED_TYPES, main, measure_denominator,
  )
  from tests.test_measure_common import (
      DATA_DIR, assert_prose_numbers_come_from_the_blocks, needs_data,
  )

  REPO = Path(__file__).resolve().parent.parent
  DOC = REPO / "docs" / "data" / "planned_relative.md"
  VERIFY_DIR = DATA_DIR / "phase3_verify"


  def committed_block():
      if not DOC.exists() or FENCE not in DOC.read_text():
          pytest.skip(f"{DOC} carries no measured block yet — the run step "
                      "pastes it")
      return parse_block(DOC.read_text(), name="planned_relative")


  def frame(**columns):
      return pd.DataFrame(columns)


  def test_the_block_covers_every_reported_type():
      assert REPORTED_TYPES == ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV",
                                "SDA", "Industrial", "Other")
      assert DENOMINATORS == ("pop", "popdensity")


  def test_measure_denominator_reports_both_columns_and_the_identity():
      norm = pd.Series([0.2, 0.4, 0.6])
      planned_mean = float(norm[:2].mean())
      got = measure_denominator(
          frame(USO_FINAL=["Planned", "Planned", "UAC"],
                norm_psi=norm, rel_psi=norm / planned_mean),
          denom="pop", types=("Planned", "UAC"))
      assert got["planned_norm_mean_pop"] == f"{planned_mean:.6g}"
      assert got["mean_norm_psi_pop_Planned"] == f"{norm[:2].mean():.6g}"
      assert got["mean_rel_psi_pop_Planned"] == \
          f"{(norm[:2] / planned_mean).mean():.6g}"
      assert got["mean_rel_psi_pop_UAC"] == f"{(norm[2] / planned_mean):.6g}"
      assert got["rel_vs_ratio_maxdiff_pop"] == "0"


  def test_measure_denominator_refuses_a_city_with_no_planned_rows():
      with pytest.raises(ValueError, match="'Planned'"):
          measure_denominator(
              frame(USO_FINAL=["UAC"], norm_psi=[1.0], rel_psi=[1.0]),
              denom="pop", types=("Planned", "UAC"))


  def test_measure_denominator_refuses_a_zero_planned_mean():
      with pytest.raises(ValueError, match="mean is 0.0"):
          measure_denominator(
              frame(USO_FINAL=["Planned", "UAC"], norm_psi=[0.0, 1.0],
                    rel_psi=[0.0, 0.0]),
              denom="pop", types=("Planned", "UAC"))


  def test_the_script_refuses_a_stamp_that_shapes_the_neighbours(
          tmp_path, monkeypatch):
      """If `methodology_stamp` ever gains this key, --verify-dir's artifact
      is no longer valid and the script must stop BEFORE staging it."""
      from delhi_psi import pipeline
      from scripts import measure_planned_relative as script

      monkeypatch.setattr(
          pipeline, "methodology_stamp",
          lambda methodology: {"second_normalization": {"rule": "minmax"}})
      with pytest.raises(SystemExit, match="second_normalization"):
          script.measure(load_config("code-2025"), tmp_path,
                         base="code-2025", verify_dir=tmp_path)


  def test_the_doc_block_has_every_required_key():
      block = committed_block()
      for denom in DENOMINATORS:
          assert float(block[f"planned_norm_mean_{denom}"]) > 0
          for name in REPORTED_TYPES:
              for column in ("rel_psi", "norm_psi"):
                  key = f"mean_{column}_{denom}_{name}"
                  assert key in block, key
                  float(block[key])
          assert float(block[f"rel_vs_ratio_maxdiff_{denom}"]) < 1e-12


  def test_the_planned_category_reads_one_on_the_relative_scale():
      block = committed_block()
      for denom in DENOMINATORS:
          assert float(block[f"mean_rel_psi_{denom}_Planned"]) == (
              pytest.approx(1.0, abs=1e-9))


  def test_popdensity_orders_the_types_as_the_spot_values_did():
      block = committed_block()
      means = {name: float(block[f"mean_rel_psi_popdensity_{name}"])
               for name in REPORTED_TYPES}
      assert means["UV"] > means["RUAC"] > means["UAC"] > means["JJC"]
      assert means["UV"] == pytest.approx(0.86, abs=0.05)


  def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
      text = DOC.read_text()
      for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                    "**Command:**"):
          assert label in text, label
      assert_prose_numbers_come_from_the_blocks(text, [committed_block()])


  @needs_data
  def test_a_fresh_run_reproduces_the_committed_block(tmp_path):
      """PLAIN `needs_data`, not the measure-cache gate: the script reads two
      CSVs and one staged joblib and runs `compute` alone — no settlement
      dedup and no O(n^2) adjacency pass."""
      committed = committed_block()
      proc = subprocess.run(
          [sys.executable, "scripts/measure_planned_relative.py",
           "--config", "code-2025", "--data-dir", str(DATA_DIR),
           "--verify-dir", str(VERIFY_DIR),
           "--work-dir", str(tmp_path / "work")],
          cwd=REPO, capture_output=True, text=True)
      assert proc.returncode == 0, proc.stderr[-4000:]
      assert parse_block(proc.stdout, name="planned_relative") == committed


  def test_main_prints_its_usage_and_exits_zero(capsys):
      with pytest.raises(SystemExit) as exc:
          main(["--help"])
      assert exc.value.code == 0
      assert "--verify-dir" in capsys.readouterr().out


  def test_out_refuses_to_overwrite_a_target_that_holds_prose(tmp_path):
      target = tmp_path / "prose.md"
      before = "# Planned relative\n\nA hand-written caption.\n"
      target.write_text(before)
      with pytest.raises(SystemExit):
          # --verify-dir is required by argparse; the prose guard fires in
          # emit_check() before the path is ever read.
          main(["--out", str(target), "--verify-dir", str(tmp_path)])
      assert target.read_text() == before
  ```

  Add the missing import to that module's header:
  `from delhi_psi.config import load_config`.

- [ ] **Step 2: Watch it fail**

  Run: `uv run pytest -q tests/test_measure_planned_relative.py`
  Expected: collection ERROR — `ModuleNotFoundError: scripts.measure_planned_relative`.

- [ ] **Step 3: Write the generator**

  Create `scripts/measure_planned_relative.py`:

  ```python
  """The reported relative scale: rel_psi = norm_psi / mean(norm_psi | Planned)
  (DEL-63, spec § 5).

  ONE block. `code-2025` with exactly one methodology value changed —
  `second_normalization: planned_relative` — run against the PROVEN neighbours
  artifact from --verify-dir: the normalisation is applied downstream in
  `compute` and is not in the methodology stamp, which this script asserts
  before staging the artifact. Reports, per denominator: the Planned mean
  that divides `norm_psi`, the per-category mean of both `rel_psi` and
  `norm_psi`, and the largest
  |mean(rel | type) − mean(norm | type) / planned_mean| — the identity the
  relative scale claims, measured independently rather than assumed.

  READ-ONLY over --data-dir and --verify-dir. Everything this script writes
  goes under --work-dir, which is never inside the data directory.

      uv run python scripts/measure_planned_relative.py --config code-2025 \
          --verify-dir ~/delhi_data/phase3_verify \
          --work-dir ~/measure_work/planned-relative
  """

  import argparse
  import sys
  from pathlib import Path

  import pandas as pd
  import yaml

  from delhi_psi import pipeline
  from delhi_psi.config import PROFILES_DIR, load_config
  from delhi_psi.pipeline import ID_COL, TYPE_COL
  from scripts._measure_common import emit, emit_check, render, resolve_work_dir
  from scripts.measure_roads_access import stage_artifacts

  PROFILE = "planned-relative"
  DENOMINATORS = ("pop", "popdensity")
  # The nine categories the code-2025 outputs carry: the ten Delhi source
  # types minus RV, the one `code-2025` excludes. A missing type is a
  # KeyError, not an empty row — the block names what the run reports.
  REPORTED_TYPES = ("Planned", "UAC", "RUAC", "JJC", "JJR", "UV", "SDA",
                    "Industrial", "Other")
  REL_COL = "rel_psi"
  BASE_COL = "norm_psi"


  def base_profile_path(base):
      """A shipped profile NAME or a path to a YAML file — `load_config`'s own
      rule, so `--config code-2025` and a derived path both work."""
      candidate = Path(base)
      if candidate.suffix in (".yaml", ".yml"):
          return candidate
      return PROFILES_DIR / f"{base}.yaml"


  def derived_profile(base, work_dir, *, profile_name):
      """`base` with `second_normalization: planned_relative`.

      A SCALAR assignment, not a block replacement: `second_normalization`
      is one methodology key, so the `dict(values)` used by the block-shaped
      helpers would explode on the string. `paths.neighbors_artifact` and
      `paths.out_dir` are dropped so the per-profile default name applies and
      --out-dir decides where the run writes; the profile is written to DISK
      so the run is reproducible by hand.
      """
      raw = yaml.safe_load(base_profile_path(base).read_text())
      raw["profile"] = profile_name
      raw["methodology"]["second_normalization"] = "planned_relative"
      paths = dict(raw.get("paths", {}))
      paths.pop("neighbors_artifact", None)
      paths.pop("out_dir", None)
      raw["paths"] = paths
      work_dir = Path(work_dir)
      work_dir.mkdir(parents=True, exist_ok=True)
      path = work_dir / f"{profile_name}.yaml"
      path.write_text(yaml.safe_dump(raw, sort_keys=False))
      return path


  def measure(cfg, work_dir, *, base, verify_dir):
      """The block: one factor changed, the proven artifact reused."""
      stamp = pipeline.methodology_stamp(cfg.methodology)
      # BOTH shapes, like measure_rule_effects' guard: a new concern is most
      # likely to arrive as a new TOP-LEVEL block, which a values() scan alone
      # would walk straight past.
      if ("second_normalization" in stamp
              or any("second_normalization" in block
                     for block in stamp.values())):
          raise SystemExit(
              "pipeline.methodology_stamp now carries `second_normalization`: "
              "the neighbours artifact is no longer reusable across this value "
              "and this run would silently report over a different method. "
              "Re-read spec § 2.1 before changing anything.")

      run_dir = Path(work_dir) / PROFILE
      profile_path = derived_profile(base, run_dir, profile_name=PROFILE)
      run_cfg = load_config(profile_path, data_dir=str(cfg.paths.data_dir),
                            out_dir=str(run_dir))
      stage_artifacts(verify_dir, run_dir,
                      source_name=cfg.paths.neighbors_artifact,
                      artifact_name=run_cfg.paths.neighbors_artifact)
      pipeline.compute(run_cfg)

      report = {}
      for denom in DENOMINATORS:
          frame = pd.read_csv(
              run_dir / f"{pipeline.output_basename(run_cfg, denom)}.csv")
          report.update(measure_denominator(
              frame, denom=denom,
              id_col=cfg.layers.settlements.id_col,
              type_col=cfg.layers.settlements.type_col))
      return report


  def measure_denominator(frame, *, denom, id_col=ID_COL, type_col=TYPE_COL,
                          types=REPORTED_TYPES):
      """One denominator's keys: the divisor, both columns' per-type means,
      and the largest departure from the ratio identity."""
      planned = frame.loc[frame[type_col] == "Planned", BASE_COL]
      if planned.empty:
          raise ValueError(
              f"planned_relative needs at least one row whose {type_col!r} "
              f"is 'Planned': {denom} has {len(frame)} rows and none of them "
              "is, so there is no denominator to divide by")
      planned_mean = float(planned.mean())
      if planned_mean == 0.0:
          raise ValueError(
              "planned_relative divides by the mean norm_psi of the "
              f"{len(planned)} Planned rows, and that mean is 0.0 — every "
              "Planned settlement has the minimum unnormalized PSI, so the "
              "relative scale is undefined")
      grouped = frame.groupby(type_col)
      means_rel = grouped[REL_COL].mean()
      means_norm = grouped[BASE_COL].mean()
      report = {f"planned_norm_mean_{denom}": f"{planned_mean:.6g}"}
      for name in types:
          report[f"mean_{REL_COL}_{denom}_{name}"] = (
              f"{means_rel[name]:.6g}")
          report[f"mean_{BASE_COL}_{denom}_{name}"] = (
              f"{means_norm[name]:.6g}")
      worst = max(abs(means_rel[name] - means_norm[name] / planned_mean)
                  for name in types)
      report[f"rel_vs_ratio_maxdiff_{denom}"] = f"{worst:.6g}"
      return report


  def main(argv=None):
      parser = argparse.ArgumentParser(description=__doc__)
      parser.add_argument("--config", default="code-2025",
                          help="profile that names the layers (default code-2025)")
      parser.add_argument("--data-dir", default=None,
                          help="data root, opened READ-ONLY")
      parser.add_argument("--work-dir", default=None,
                          help="scratch (the derived profile, its staged "
                               "artifact and outputs); default a fresh "
                               "temporary directory. Never under --data-dir.")
      parser.add_argument("--verify-dir", required=True,
                          help="an existing, complete code-2025 run "
                               "(colonies_neighbors.joblib + both output CSVs), "
                               "opened READ-ONLY")
      target = parser.add_mutually_exclusive_group()
      target.add_argument("--out", default=None,
                          help="write the block here instead of stdout — "
                               "BLOCKS ONLY; REFUSES (exit 1) to overwrite a "
                               "target that already holds hand-written prose, "
                               "since that would delete every caption and "
                               "Finding — use --splice to refresh such a "
                               "document in place instead")
      target.add_argument("--splice", default=None,
                          help="refresh the block INSIDE this committed "
                               "document in place, preserving every caption "
                               "and Finding (DEL-59)")
      args = parser.parse_args(argv)

      emit_check(out=args.out, splice=args.splice)

      cfg = load_config(args.config, data_dir=args.data_dir)
      work_dir = resolve_work_dir(args.work_dir, data_dir=cfg.paths.data_dir,
                                  prefix="delhi_psi_planned_relative_")
      verify_dir = Path(args.verify_dir).expanduser()

      print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}",
            file=sys.stderr)
      print(f"verify-dir: {verify_dir}", file=sys.stderr)
      print(f"work-dir: {work_dir}", file=sys.stderr)
      report = measure(cfg, work_dir, base=args.config, verify_dir=verify_dir)
      emit(render(report, name="planned_relative"),
           out=args.out, splice=args.splice)
      return 0


  if __name__ == "__main__":
      sys.exit(main())
  ```

  Note `measure_denominator` takes `id_col` for signature parity with the
  other blocks but does not use it (the frame is already one row per
  settlement and the grouping is by type); keep the parameter so the call
  site reads like its siblings, and do not add an unused-name lint failure —
  if the project's linter objects, drop `id_col` from both the definition and
  the call.

- [ ] **Step 4: Create the document with a placeholder block**

  Create `docs/data/planned_relative.md`. The placeholder fenced block is
  what `--splice` needs to find; Step 5 replaces it with the measured block.

  ````markdown
  # The reported relative scale: 1 = the average Planned colony

  Raj's answer to the last open methodology question (22 Oct 2026; decision
  log `docs/decisions/2026-10-22-raj-call-decisions.md`, DEL-63) is that the
  paper should report a relative scale: divide every settlement's `norm_psi`
  by the mean `norm_psi` of the Planned colonies, so the average planned
  colony reads 1 and every other type reads as a multiple of it. This
  document is that measurement. `scripts/measure_planned_relative.py` runs
  `code-2025` with one methodology value changed — `second_normalization:
  planned_relative` — and reports the resulting per-type means for both
  denominators. `tests/test_measure_planned_relative.py` re-runs it and
  compares the block.

  Numbers quoted in prose below in `backticks` are block values verbatim;
  percentages and other derived quantities are written with a `%` sign or
  without backticks.

  - **Run date:** 2026-10-26
  - **Inputs:** `~/delhi_data` (the layers `code-2025` names), `~/delhi_data/phase3_verify` (the proven neighbours artifact, reused unchanged)
  - **Commit:** `RUN_COMMIT`
  - **Command:** `uv run python scripts/measure_planned_relative.py --config code-2025 --verify-dir ~/delhi_data/phase3_verify --work-dir ~/measure_work/planned-relative`

  ## What the scale is

      rel_psi_i = norm_psi_i / mean(norm_psi | category == "Planned")

  One constant per run: the mean of the min-max column over the Planned rows
  of the same computed frame — post-exclusion, post-everything. `rel_psi` is
  a pure per-row rescale, so a type's mean `rel_psi` IS the ratio of its mean
  `norm_psi` to the Planned mean; the block measures that identity
  independently (`rel_vs_ratio_maxdiff_*`).

  The relative scale **inherits the min-max denominator**: `norm_psi` is an
  affine transform of `unnorm_psi`, so ratios of means are not invariant to
  min-maxing. That belongs in the methods text beside the scale itself.

  ```text
  block: planned_relative
  placeholder: the generator's first run replaces this block
  ```

  ## The numbers

  Under the population-density denominator the average Planned colony reads
  `1`, UV `0.860416`, RUAC `0.487221`, UAC `0.391285` and JJC `0.0300647`:
  the gap the 0–1 scale buried is the reported quantity. The full per-type
  means for both denominators are in the block above; each denominator's
  Planned mean `norm_psi` is the divisor (`0.0443043` under popdensity).

  Under the population-size denominator the ordering is different — UV still
  reads above 1, JJC is second-highest at `1.95683`, and JJR is the lowest at
  `0.299029` — which is why the paper must name the denominator beside the
  scale.

  ## What this does not decide

  Nothing here changes a shipped profile. `code-2025` still runs `minmax`;
  the ratified profile (DEL-31) is where `planned_relative` is adopted, after
  Raj ratifies these numbers. Main-paper placement is Raj's writing call
  ("i am thinking move the relative to the main paper"); the calculation is
  decided. Recalculation (DEL-32) and the main figure (DEL-33) follow DEL-31.
  ````

  `RUN_COMMIT` is replaced in Step 5 with `git rev-parse --short HEAD` (the
  Task 3 commit — the code the run was made from).

  Add the pointer to `docs/data/psi_columns.md`, at the end of "What this
  does not decide" (after line 164):

  ```markdown

  The reported scale itself is now measured separately:
  `docs/data/planned_relative.md` gives the per-type means under
  `second_normalization: planned_relative`, where 1 is the average Planned
  colony (DEL-63).
  ```

  Finally, register the new document with its generator in
  `tests/test_measure_common.py::DOCUMENT_GENERATORS` (beside the other
  entries — order is irrelevant; the map is literal, never a glob):

  ```python
      "planned_relative.md": "scripts/measure_planned_relative.py",
  ```

- [ ] **Step 5: Run the real generator and splice the block**

  ```bash
  uv run python scripts/measure_planned_relative.py --config code-2025 \
      --data-dir ~/delhi_data --verify-dir ~/delhi_data/phase3_verify \
      --work-dir ~/measure_work/planned-relative \
      --splice docs/data/planned_relative.md
  ```

  Then:
  - replace `RUN_COMMIT` with the short SHA printed by
    `git rev-parse --short HEAD`;
  - check every backticked number in the prose against the spliced block.
    The block is authoritative. Expected values from the planning run
    (popdensity): `planned_norm_mean_popdensity: 0.0443043`,
    `mean_rel_psi_popdensity_Planned: 1`, `..._UV: 0.860416`,
    `..._RUAC: 0.487221`, `..._UAC: 0.391285`, `..._JJC: 0.0300647`,
    `rel_vs_ratio_maxdiff_popdensity: 0` (any value `< 1e-12` is fine; the
    prose quotes none of the maxdiff values). If a prose number differs from
    the block, replace the prose with the block's string — the test
    `assert_prose_numbers_come_from_the_blocks` is the arbiter.

  Inspect the result: `git diff -- docs/data/planned_relative.md` must show
  the real block in place of the placeholder, every caption intact.

- [ ] **Step 6: Green, including the real-data regen**

  ```bash
  uv run pytest -q tests/test_measure_planned_relative.py
  ```

  Expected: PASS. `test_a_fresh_run_reproduces_the_committed_block` runs
  (this machine has the data) and reproduces the block byte-for-byte;
  `planned_norm_mean_*` is not `0`; the Planned category mean is `1` to
  1e-9; `rel_vs_ratio_maxdiff_*` is `< 1e-12`.

- [ ] **Step 7: Commit**

  ```bash
  git add scripts/measure_planned_relative.py tests/test_measure_planned_relative.py \
      tests/test_measure_common.py docs/data/planned_relative.md \
      docs/data/psi_columns.md
  git commit -m "feat(scripts): planned-relative means generator and its committed block (DEL-63)"
  ```

### Task 5: CHANGELOG, full smoke, ship

- [ ] **Step 1: CHANGELOG `[Unreleased]` entry** (top of the section):

  ```markdown
  - **The reported PSI can now be the planned-relative scale (DEL-63).**
    `methodology.second_normalization` is a three-value enum instead of a
    boolean: `minmax` (today's `true`, emits `norm_psi`), `none` (today's
    `false`), and `planned_relative`, which emits `norm_psi` **and**
    `rel_psi = norm_psi / mean(norm_psi | Planned)` — one constant per run,
    so the average Planned colony reads 1 and every other type reads as a
    multiple of it.
    - **No shipped number moves.** No profile adopts the new value;
      `expected_values.csv`, every `production/*.csv` and every GeoJSON are
      byte-identical. The two `variants_expected_values.csv` files grow
      append-only with one new variant, and all sixteen profiles that said
      `second_normalization: true` now say `minmax` (a rename, not a
      change).
    - **Bare `true` is now a load error** naming `minmax`; YAML `false` still
      means `none`. An anonymous boolean is exactly the ambiguity the enum
      removes.
    - **The relative scale inherits the min-max denominator:** `norm_psi` is
      an affine transform of `unnorm_psi`, so the ratios are not invariant to
      min-maxing. The methods text must say so beside the scale
      (`docs/data/planned_relative.md`, generated by
      `scripts/measure_planned_relative.py`).
  ```

- [ ] **Step 2: Full smoke**

  ```bash
  mkdir -p /tmp/opencode
  uv run pytest -q -W error > /tmp/opencode/del63-smoke.log 2>&1; echo "EXIT_CODE: $?" >> /tmp/opencode/del63-smoke.log
  ```

  Run it in the background (~10 min). Expected: `EXIT_CODE: 0`, including
  both variants-CSV regen parametrizations against the regenerated bytes and
  the new real-data block test.

- [ ] **Step 3: Blast-radius proof against the merge base**

  ```bash
  git diff --stat c6a7654 -- tests/fixtures/
  git diff --name-only c6a7654 -- tests/fixtures/
  ```

  Expected: only the two `variants_expected_values.csv` files.

  ```bash
  git diff c6a7654 -- tests/fixtures/ | grep '^-' | grep -v '^---' || true
  ```

  Expected: no removed lines (pure append).

- [ ] **Step 4: Push, PR, CI**

  ```bash
  git push -u origin del-63-planned-relative-scale
  gh pr create --base main --title "feat: planned-relative reported scale (DEL-63)" \
      --body-file /tmp/opencode/del63-pr.md
  ```

  The PR body carries: the spec link, this plan, the plan-phase review
  record, the smoke log tail, the append-only proof, and the measured block.

  Wait for CI. Expected green — the drift guard reproduces every committed
  byte, including the new variant's rows.

- [ ] **Step 5: Merge and close out**

  Squash-merge; sync `main`; Jira DEL-63 → Done with a comment linking the
  PR and the spec; unblock DEL-31 with the measured numbers and the note
  that adoption is a profile edit plus ratification.

## Review record (plan phase, round 1 — 26 Oct 2026)

- **Finders:** step-5-preview (nvidia) — walked every call site of the changed
  names, traced the StrEnum/`_coerce_enum`/`RESERVED_VALUES` interaction, and
  checked the oracle's vocabulary against the new variant pin; glm-flash-latest
  (openrouter) — recomputed the plan's numbers from the fixtures and the real
  `phase3_verify` CSVs, audited the measurement-block conventions and the
  `DOCUMENT_GENERATORS` coverage guard. **Model note:** `minimax-m3.1-flash`
  and `deepseek-v4.1-flash` both failed at the endpoint, so the round used
  step-5-preview + glm-flash-latest, with glm-latest judging.
- **Judge:** glm-latest — **KILLED** finding 6 (the truthiness read in
  `tests/test_oracle.py:59` passes today and the profile is `minmax`; latent
  style, not a failure), KILLED 14 (YAML integer `1` colliding with the `True`
  reserved key is refused either way, so no code change is warranted), KILLED
  15 (the `-k "… or neither_second"` claim is factually wrong — the test name
  contains that substring), KILLED 16 (`generate_oraculum_fixtures.py:114`'s
  "eight derived rule-sets" is pre-existing staleness, out of scope); KEPT
  1–5 and 7; AMENDED 13 (replace every boolean in the table cell, not just the
  first two).
- **Surviving and fixed in this revision:**
  - **(Blocker) Oracle-vocabulary pin.** Task 3's new variant test asserted
    Delhi's `UV`/`UAC` on a city whose vocabulary is
    `("Planned", "UC", "JJC", "RV", "RUAC", "IND")`. Rewritten to the measured
    oracle ordering (`IND > Planned > UC > RUAC > RV > JJC`, JJC exactly `0`);
    the `UV ≈ 0.86` pin stays in Task 4's real-data block test only.
  - **(Blocker) `metrics_for` migration moved into Task 1.** A `none` StrEnum
    member is truthy, so the old truthiness read would request `norm_psi` from
    a `manuscript` frame that no longer carries it. Task 1 now owns the
    rewrite; Task 3 only adds `rel_psi` to `METRIC_MAP`.
  - **(Blocker) `DOCUMENT_GENERATORS` entry.** The new
    `docs/data/planned_relative.md` carries a fenced block, and
    `test_the_mapping_covers_every_block_bearing_document` is a literal-map
    coverage test. Task 4 now registers it (and its file list/commit names
    `tests/test_measure_common.py`).
  - **(Important) `JJC` rendering.** `f"{0.0300647:.6g}"` is `0.0300647`, not
    `0.030065`; plan prose and Task 4's expected values now use the block's
    own string.
  - **(Important) Population-size ordering.** JJC is second-highest (`1.95683`)
    and JJR lowest (`0.299029`) under `pop`; the template sentence now says
    so instead of "JJC is the lowest".
  - **(Important) `--out` guard test was vacuous.** `--verify-dir` is
    required, so argparse exited 2 before `emit_check`. The test now passes
    `--verify-dir`; the prose guard still fires before any path is read.
  - **(Minor) RED expectations, line references, docstrings, scratch paths.**
    Task 1 Step 3 now states the exact failures; `ABSENT_NEIGHBOR` is
    `index.py:27`; there are six existing `test_index.py` call sites; the
    no-Planned docstring says post-neighbours (the scenario tuple's `False`);
    the verification heredocs use `uv run python` and `/tmp/opencode`; the
    derived artifact name is `colonies_neighbors_planned-relative.joblib`.
  - **(Minor) Methodology-config cell.** Every `true`/`false` in the
    `second_normalization` row is replaced, since `true` would now raise
    `ConfigError`.
- **Zero surviving blockers → stopped at round 1.**
- **Execution fix (26 Oct, Task 2 Step 2, verified against the fixtures):** the
  no-Planned guard test above used `RULESETS["code"]`, whose GLOBAL barrier
  severs every link of A (Oraculum's only bank owner); with A and D excluded
  the remaining five settlements are all-zero on `bank_pcen`, so the min-max
  guard fires before the planned-empty guard. The test now uses
  `RULESETS["ideal"]` (`pair` keeps A's links; `contributes` lets the
  excluded settlements lend), which reaches the division. The guard under
  test is unchanged.

## Self-review

- **Spec coverage:** § 1 config enum → Task 1 Steps 4/6; § 2 production
  dispatch + both guards → Task 1 Step 5; § 3 reference mirror, vocabulary
  contract, one appended variant, the four pins → Tasks 2–3; § 4 blast radius
  → Tasks 1 Step 7, 3 Step 4, 5 Step 3; § 5 generator + committed block +
  drift test → Task 4; § 6 docs (column contract, config reference, ticket)
  → Task 1 Step 6 and Task 4 Step 4; acceptance 1–5 → Tasks 1–5.
- **Placeholders:** the only literal placeholder is the fenced block in
  Task 4 Step 4, which Step 5 replaces and Step 6 verifies. `RUN_COMMIT` is
  likewise filled in Step 5.
- **Type consistency:** `overall_psi(..., second_normalization=...)` keeps its
  keyword; `compute_city(..., second_norm=...)` keeps its keyword;
  `metric_columns(second_normalization=...)` keeps its keyword. The enum is
  `SecondNormalization` in `config.py` and is imported only where a typed
  value is constructed (`variant_methodology`, the two synthetic tests).
  `SECOND_NORMALIZATIONS` exists in both implementations as a literal tuple —
  they are independent by design and the mapped-knob test is what keeps them
  in step.
