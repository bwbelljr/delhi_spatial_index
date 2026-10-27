# DEL-33: Paper figures from the ratified run — design spec

**Status:** design fixed by recorded decisions; no new owner question. The
owner's 26 Oct instruction continues the queue. Every content decision below
was taken earlier: Raj's figure triage (DEL-30), the denominator/column
measurement (DEL-52), and the 22 Oct call (4a: planned-relative headline,
min-max in the appendix).

**What this ticket is.** A reproducible figures command — the one WORKPLAN
Phase 3A records as the last missing piece ("a figures command that renders
to files [DEL-25] … The figures command is Phase 4 (DEL-33)"). It reads the
**DEL-32 run** (`~/delhi_data/phase4_ratified`, both denominator CSVs) plus
the settlement layer, and renders the paper's figures to files. No pipeline
code changes; no numbers are recomputed outside the CSVs.

## Decisions this design implements

| # | question | decision | source |
|---|---|---|---|
| 1 | headline scale | **planned-relative** (`rel_psi`, 1 = the average planned colony); min-max in the appendix | 22 Oct 2026, 4a (DEL-63) |
| 2 | headline denominator | **popdensity** for the map and the headline bars; pop shown beside it in the breakdown | DEL-52 (`docs/data/psi_columns.md`) |
| 3 | types shown | the **seven reported types** (Planned, UAC, JJC, RUAC, UV, SDA, JJR) — excluded types never appear as a bar | DEL-30 |
| 4 | spatial extent | **full-Delhi map**; unscored settlements drawn grey | DEL-30 |
| 5 | JJC juxtaposition | **featured** (its own panel) | DEL-30 |
| 6 | per-type data table | **removed** — footnotes instead | DEL-30 |

## Design

### 1. The command

`scripts/render_paper_figures.py`:

```
uv run python scripts/render_paper_figures.py --run-dir ~/delhi_data/phase4_ratified
```

- `--run-dir` (required): the DEL-32 output directory; must contain
  `delhi_psi_<profile>_pop_2020.csv` and `…_popdensity_2020.csv` (found by
  glob `delhi_psi_*_{pop,popdensity}_2020.csv`, exactly one match each).
- `--data-dir` (default `~/delhi_data`): the settlement layer, read with the
  same `layers.settlements` contract as the profile (`USO_AREA_U`,
  `USO_FINAL`).
- `--out-dir` (default: `docs/figures/ratified-2026`): where the PNGs and
  `manifest.txt` land. The default keeps the figures reviewable in the repo
  (the `docs/oracle/` precedent); the run directory gets a copy via the
  delivery folder, not by default.

Exit codes: 0 on success; 2 on a bad argument; 1 with a one-line message on
a missing/unreadable input (never a traceback).

### 2. The figure set (content contract)

All figures 150 dpi, `Agg`, the palette and rc style documented in
`scripts/render_oracle_maps.py` (categorical hues, `PAGE_BG`/`SURFACE`/`INK_*`,
`BASELINE` grey), PNG metadata stripped of timestamps (deterministic bytes).

1. **`fig4_psi_by_type.png`** — the paper's Figure 4 replacement. Grouped
   horizontal bars, seven reported types, two denominators (pop, popdensity),
   **relative scale**; a vertical rule at 1.0 labelled "average planned
   colony". Types ordered by popdensity mean, descending (stable by name on
   ties). Value labels at the bar ends (`%.2f`).
2. **`fig5_psi_map.png`** — full-Delhi settlement map, **popdensity**, shaded
   by `rel_psi` on a diverging ramp **centred at 1.0** (values clipped at the
   1st/99th percentiles for the ramp only; the clip bounds are printed in the
   manifest), unscored settlements (the 295 not in the CSV) filled
   `BASELINE` grey, boundaries in `INK_MUTED`. Colour bar labelled
   "PSI relative to the planned average".
3. **`fig6_jjc_vs_planned.png`** — the juxtaposition: two panels. Left, the
   `rel_psi` distributions (Planned, JJC) as strips with the median marked,
   popdensity; right, the same two types' **PCEN by service** (the mechanism:
   which services drive the gap), as a dot plot. A one-line annotation states
   the share of JJC settlements above the Planned median.
4. **`figA1_psi_by_type_minmax.png`** — appendix: figure 1 on the min-max
   (`norm_psi`) scale, same layout and order.
5. **`figA2_psi_map_minmax.png`** — appendix: figure 2 on `norm_psi`, a
   sequential ramp (no centre).

`manifest.txt` records: run dir, profile name, input file names and row
counts, reported/unscored counts, per-type mean `rel_psi` and `norm_psi` for
both denominators, the ramp clip bounds, and the renderer's git commit when
available (else `unknown`).

### 3. Data preparation (pure, unit-testable)

- `load_run(run_dir)` → `{"pop": df, "popdensity": df}`; raises
  `FigureError` with the glob pattern when a file is missing or ambiguous.
- `type_means(frame)` → DataFrame indexed by category with `mean_rel`,
  `mean_norm`, `n`; **only** the seven reported types, in value order.
- `map_frame(settlements, frame)` → GeoDataFrame with `rel_psi`/`norm_psi`
  and a boolean `scored`; the left join is on `USO_AREA_U`, and **every**
  settlement layer row survives (unscored rows get NaN and `scored=False`).
- `ramp_bounds(values)` → the 1st/99th percentile pair actually used.
  All pure functions take plain frames so the tests need no real data.

### 4. Determinism

Fixed figsize/dpi, `font.family` from the shared rc block, no random
jitter (the strip uses a deterministic offset by sorted index), sorted
iteration everywhere, PNG metadata dict without timestamps. Two runs over
the same inputs must be **byte-identical** — that is the integration test.

### 5. What is committed

The script, its tests, and the five PNGs + `manifest.txt` under
`docs/figures/ratified-2026/`. The figures are generated from real data that
CI does not have, so — like `docs/oracle/` — they are committed artefacts;
the `needs_data` test regenerates and compares bytes when the run directory
is present.

## Out of scope

- Journal typesetting, PDF/EPS conversion, LaTeX edits (external to this
  repo).
- Interactive maps, any web output.
- Re-deriving any number: the script plots the CSVs it is given.
- DEL-53 (media reclassification) and DEL-36 (band sweeps) figures.

## Acceptance

1. TDD: every pure helper has a test written and watched failing first; the
   integration test is `needs_data` and skips cleanly in CI.
2. The five PNGs and the manifest render from the DEL-32 run; two runs are
   byte-identical.
3. Full smoke green (`uv run pytest -q -W error` = the CI gate).
4. The PR body links the spec/plan, shows the manifest, and embeds the five
   figures.
