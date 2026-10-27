# DEL-33: Paper figures from the ratified run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> **Execution note (ship.md deviation rule):** one new script + one test
> module + five committed PNGs. The lead implements inline under strict TDD.
> Plan and code each get one scaled review round (review-budget rule).

**Goal:** Add the missing figures command — five paper figures rendered from
the DEL-32 run CSVs, deterministic and reproducible — and commit them under
`docs/figures/ratified-2026/`.

**Architecture:** `scripts/render_paper_figures.py` with four PURE data
helpers (`load_run`, `type_means`, `map_frame`, `ramp_bounds`) and five
renderers that consume them. Inputs are only the DEL-32 CSVs and the
settlement layer; no index math is re-implemented. Styling reuses the
documented palette/rc block in `scripts/render_oracle_maps.py`. Tests:
synthetic-frame unit tests for the helpers, and one `needs_data` integration
test that renders twice and compares bytes.

**Tech Stack:** matplotlib (`Agg`), pandas, geopandas; pytest;
`tests/test_measure_common.py`'s `DATA_DIR`/`needs_data`.

**Spec:** `docs/superpowers/specs/2026-10-27-del-33-paper-figures-design.md`.

## Global Constraints

- **Plot, don't recompute.** Every number in every figure comes from a
  column already in the run CSV; the script adds no PSI math.
- **Only the seven reported types appear as a bar/ramp category**
  (Planned, UAC, JJC, RUAC, UV, SDA, JJR). RV/Industrial/Other are absent
  from the CSV by construction; on the map their settlements are grey.
- **Determinism:** fixed dpi/figsize, sorted iteration, deterministic strip
  offsets, PNG metadata without timestamps. Two runs over the same inputs
  are byte-identical.
- **No tracebacks:** a bad `--run-dir` exits 1 with one line naming the glob
  that failed; argparse errors stay exit 2.
- **The run is the DEL-32 directory** (`~/delhi_data/phase4_ratified`,
  commit `8470955` profile `ratified-2026`); figures must not be rendered
  from the fixture cities or the July 2025 outputs.
- **TDD:** each helper's test is written and watched failing before the
  helper exists.

## Measured facts this plan argues from

- Run CSVs: `delhi_psi_ratified-2026_{pop,popdensity}_2020.csv`, 4,062 rows
  each; columns include `USO_AREA_U`, `category`, `rel_psi`, `norm_psi`,
  and the per-service `*_pcen`/`*_idx` columns.
- Settlement layer: 4,357 rows → **295 unscored** (15 missing population +
  36 Industrial + 33 Other + RV, which has no CSV row either), so the map's
  grey class is non-trivial and must be tested.
- Type means (popdensity, `rel_psi`): UV 1.2737, JJR 1.1796, Planned 1.0000,
  RUAC 0.9229, UAC 0.5287, SDA 0.5169, JJC 0.0602. On pop the order flips
  (JJC 1.7773 first; JJR 0.4005 last) — hence the spec's per-denominator
  sort and the separate JJC panel.
- `docs/oracle/` is the precedent for committed PNGs rendered by a script.

---

### Task 1: Pure helpers (TDD)

**Files:**
- Create: `scripts/render_paper_figures.py`
- Create: `tests/test_render_paper_figures.py`

- [ ] **Step 1: Write the failing tests** for the four helpers, using tiny
  synthetic frames (3 types × 2 denominators), before any helper exists:

  - `test_load_run_finds_both_denominators_and_names_the_glob_when_missing`
    — a tmp dir with the two CSVs loads; an empty dir raises `FigureError`
    with the glob pattern in the message.
  - `test_type_means_keeps_only_reported_types_and_sorts_by_value` — an
    input frame containing a `category` outside the seven is dropped; the
    order is descending `mean_rel`, ties by name.
  - `test_map_frame_keeps_every_settlement_and_marks_unscored` — 4
    settlements, 2 in the frame → all 4 rows survive, 2 `scored`, NaN where
    unscored.
  - `test_ramp_bounds_are_the_first_and_ninety_ninth_percentile` — known
    vector, exact expected floats.

  Watch them fail: `uv run pytest tests/test_render_paper_figures.py -q`
  (`ImportError`/`NameError` — the module does not define them yet).

- [ ] **Step 2: Implement the helpers** with these signatures:

  ```python
  class FigureError(Exception): ...

  def load_run(run_dir: Path) -> dict[str, pd.DataFrame]
  def type_means(frame: pd.DataFrame) -> pd.DataFrame   # mean_rel, mean_norm, n
  def map_frame(settlements: gpd.GeoDataFrame, frame: pd.DataFrame) -> gpd.GeoDataFrame
  def ramp_bounds(values: pd.Series) -> tuple[float, float]
  ```

  Constants: `REPORTED_TYPES = ("Planned", "UAC", "JJC", "RUAC", "UV", "SDA", "JJR")`,
  `RUN_GLOB = "delhi_psi_*_{denom}_2020.csv"`, `ID_COL = "USO_AREA_U"`.

- [ ] **Step 3: Green:** `uv run pytest tests/test_render_paper_figures.py -q` → 4 passed.

### Task 2: The five renderers

**Files:**
- Modify: `scripts/render_paper_figures.py`

- [ ] **Step 1: Reuse the shared style.** Copy the documented palette/rc
  block from `scripts/render_oracle_maps.py` (`PAGE_BG`, `SURFACE`,
  `INK_PRIMARY`, `INK_SECONDARY`, `INK_MUTED`, `BASELINE`, the seven
  categorical hues mapped to the seven reported types, the rcParams update,
  `PNG_METADATA`), adding a diverging ramp for `rel_psi` (two documented
  hues with a neutral at 1) and a sequential ramp for `norm_psi`.

- [ ] **Step 2: Write the five renderers**, each `render_*(data, out_dir) -> Path`:
  - `render_psi_by_type(run, out_dir, *, scale)` — grouped horizontal bars,
    seven types, both denominators, rule at 1.0 for `rel`, labels `%.2f`.
  - `render_psi_map(run, settlements, out_dir, *, scale)` — full Delhi,
    popdensity, grey unscored, colour bar, clip bounds annotated.
  - `render_jjc_vs_planned(run, out_dir)` — strip distributions (Planned vs
    JJC, popdensity) + PCEN dot plot by service + the "share of JJC above
    the Planned median" annotation; deterministic strip offsets from the
    sorted index.
  - `render_psi_by_type_minmax` / `render_psi_map_minmax` are the same two
    renderers with `scale="norm"` under the appendix filenames.

  Filenames exactly: `fig4_psi_by_type.png`, `fig5_psi_map.png`,
  `fig6_jjc_vs_planned.png`, `figA1_psi_by_type_minmax.png`,
  `figA2_psi_map_minmax.png`.

- [ ] **Step 3: Smoke-render** into `/tmp/del33-scratch/` against the real
  run dir and eyeball all five (the numbers must match the manifest below;
  the map's grey class must be visible):

  ```bash
  uv run python scripts/render_paper_figures.py \
      --run-dir ~/delhi_data/phase4_ratified --out-dir /tmp/del33-scratch
  ```

### Task 3: Manifest, CLI, integration test

**Files:**
- Modify: `scripts/render_paper_figures.py`
- Modify: `tests/test_render_paper_figures.py`

- [ ] **Step 1: Write `manifest.txt`** next to the PNGs: run dir, profile
  name, input filenames + row counts, reported/unscored counts, per-type
  `mean_rel`/`mean_norm` for both denominators (`%.6g`, sorted), ramp clip
  bounds, and the git commit (`git rev-parse --short HEAD`, `unknown` if
  unavailable). Written by the CLI, not by a renderer.

- [ ] **Step 2: CLI** (`argparse`): `--run-dir` required, `--data-dir`
  default `~/delhi_data`, `--out-dir` default `docs/figures/ratified-2026`;
  `main()` loads, renders all five, writes the manifest, prints the output
  dir. Exit 1 via `FigureError` with one line on stderr.

- [ ] **Step 3: Integration test** (`tests/test_render_paper_figures.py`):

  ```python
  from tests.test_measure_common import DATA_DIR, needs_data

  def _run_cli(run_dir, out_dir):
      return subprocess.run([sys.executable, "scripts/render_paper_figures.py",
                             "--run-dir", str(run_dir), "--data-dir", str(DATA_DIR),
                             "--out-dir", str(out_dir)],
                            cwd=REPO, capture_output=True, text=True)

  @needs_data
  @pytest.mark.skipif(not RUN_DIR.exists(), reason="DEL-32 run not present")
  def test_the_five_figures_render_and_are_deterministic(tmp_path):
      a, b = tmp_path / "a", tmp_path / "b"
      assert _run_cli(RUN_DIR, a).returncode == 0
      assert _run_cli(RUN_DIR, b).returncode == 0
      names = sorted(p.name for p in a.glob("*.png"))
      assert names == sorted(EXPECTED_FIGURES)
      for name in names:
          assert (a / name).stat().st_size > 20_000
          assert (a / name).read_bytes() == (b / name).read_bytes()
      assert "reported: 4062" in (a / "manifest.txt").read_text()
  ```

  Watch it fail first (`scripts/render_paper_figures.py` has no `main`).

- [ ] **Step 4: Green:** the 4 unit tests plus the integration test pass.

### Task 4: Ship

- [ ] **Step 1: Render the committed set.**

  ```bash
  uv run python scripts/render_paper_figures.py --run-dir ~/delhi_data/phase4_ratified
  ```

  `docs/figures/ratified-2026/` gains the five PNGs + `manifest.txt`.

- [ ] **Step 2: Full smoke** (`uv run pytest -q -W error`) in the
  background; expected green, integration test included on this machine.

- [ ] **Step 3: CHANGELOG `[Unreleased]`** — one entry: the figures command
  and the committed figure set, naming the scale split (relative headline,
  min-max appendix) and the seven-type rule.

- [ ] **Step 4: PR** (body: spec/plan links, review record, manifest
  summary, the five images embedded), CI green, squash-merge, sync `main`,
  Jira DEL-33 In Progress → In Review → Done with a plain comment naming the
  figure set and the two headline numbers (popdensity: Planned 1.00, JJC
  0.06; pop: JJC 1.78, Planned 1.00).

## Review record (plan phase, round 1 — 27 Oct 2026)

- **Model:** `opencode-go/glm-5.3-flash` (owner-directed lineup), one finder
  + one judge round — a scaled review for a self-contained new script.
- **Finder:** *(to be filled)*
- **Judge:** *(to be filled)*

## Self-review

- **Spec coverage:** § 1 command → Task 3 Step 2; § 2 five figures +
  manifest → Tasks 2–3; § 3 helpers → Task 1; § 4 determinism → Task 3
  Step 3; § 5 committed artefacts → Task 4 Step 1; acceptance 1–4 →
  Tasks 1–4.
- **Placeholders:** the review record is filled after the round; no other.
- **Consistency:** the seven reported types appear once, as
  `REPORTED_TYPES`; the scale names are `rel`/`norm` everywhere; filenames
  are fixed in one constant and asserted by the integration test.
