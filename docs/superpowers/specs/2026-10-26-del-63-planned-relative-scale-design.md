# DEL-63: Planned-Relative Reported Scale — Design Spec

**Status:** design approved by owner in chat, 2026-10-26.
**Origin:** Rajesh's post-call email, 22 Oct 2026 (4a answered — the last
open methodology decision), recorded in
`docs/decisions/2026-10-22-raj-call-decisions.md` and ticket DEL-63.

> "relative calculation: make planned 1 by dividing all the numbers with
> planned colony averages and compute the index. (leave what we have in
> the appendix and i am thinking move the relative to the main paper)"

The 0–1 min-max scale "doesn't tell our story carefully" — every type mean
reads as tiny (Planned 0.044, JJC 0.0013), which buries the gap. On the
relative scale, **1 = the average planned colony**, and the gap IS the
story. Main-paper placement is Raj's writing call ("i am thinking") — the
calculation is decided.

## Owner decisions taken in chat (2026-10-26)

| # | Question | Decision |
|---|---|---|
| 1 | Base column the relative divides | **`norm_psi`** — matches Raj's spot values (computed from the July 2025 `norm_psi` × popdensity means: Planned 1.00, UV ≈ 0.86, JJR ≈ 0.85, Industrial ≈ 0.85, RUAC ≈ 0.49, UAC ≈ 0.39, JJC ≈ 0.03) and keeps appendix + main text reading off the same reported quantity |
| 2 | What `planned_relative` emits | **Both columns** — `norm_psi` AND `rel_psi`; one ratified run produces both tables, and the ratio is auditable in the CSV (Planned's rows must average exactly 1.00) |
| 3 | Scope | **One ticket, one PR** — variant machinery + relative-means table generator together; DEL-31/32/33 adoption explicitly out |

## The definition

    rel_psi_i = norm_psi_i / mean(norm_psi | category == "Planned")

- One constant per run: the mean of the **min-max column over the Planned
  rows of the same computed frame** (post-exclusion, post-everything — the
  universe whose `norm_psi` values are already emitted).
- `rel_psi` is a pure per-row rescale by that constant, so the reported
  table's per-category mean of `rel_psi` IS Raj's ratio
  `mean(norm_psi | type) / mean(norm_psi | Planned)` — linearity, no
  separate aggregation step to get wrong.
- Plumbing note (why the base choice matters): `norm_psi` is an affine
  transform of `unnorm_psi`, so ratios of means are NOT invariant to
  min-maxing. Option 2 in the decision table pins it to `norm_psi`; this
  must be stated in the paper (the relative scale inherits the min-max
  denominator).

## Design

### 1. Config: `methodology.second_normalization` bool → enum

Accepted YAML inputs, in the `pinned`-table style of every other
methodology knob:

| YAML input | Internal value | Behaviour |
|---|---|---|
| `false` (boolean, as today) | `"none"` | no secondary column — unchanged |
| `minmax` | `"minmax"` | `norm_psi` — equal to today's `true` |
| `planned_relative` | `"planned_relative"` | `norm_psi` + `rel_psi` |

- Bare `true` becomes a **validation error** pointing at `minmax` — an
  anonymous boolean is exactly the ambiguity we are removing; the error
  text must say so loudly (this kills any silent-migration trap).
- `config.py`: `second_normalization` leaves the bool-exception comment
  and the bool exception branch in `_require`/`_bool`; the pinned table
  gains `{"minmax": "minmax", "planned_relative": "planned_relative"}` —
  YAML boolean `false` and the string `"none"` both map to `"none"`
  (the `exclusion.stage` precedent of bool-or-string YAML inputs), and
  every other input is mapped by exact string.
- In-repo profiles that say `second_normalization: true` are rewritten to
  `minmax` in the same commit (enumerated during implementation; each
  rewrite is byte-visible in the diff).

### 2. Production dispatch: `overall_psi`

`delhi_psi/index.py::overall_psi` swaps `second_normalization: bool` for
the internal value (keyword keeps its name so `pipeline.py`'s call shape
survives):

- `"none"`: no secondary column (the "absent when off" contract stands,
  now also for `rel_psi`).
- `"minmax"`: `norm_psi` exactly as today — this branch must be a pure
  rename of the constant comparison, reproducible by the existing suite.
- `"planned_relative"`: compute `norm_psi` FIRST (rel is its ratio), then
  `rel_psi = norm_psi / planned_mean`, emitting **both** columns.

Guards, raised before any division:

- no row whose `category` is `"Planned"` → `ValueError` naming the category
  and the frame;
- `planned_mean == 0.0` → `ValueError` naming the category and the mean
  (a degenerate universe, never seen in real data — guard anyway).

### 3. Reference mirror + oracle variant pins

- `tests/reference_impl.py` gains the same three-way rule in its
  `second_norm` handling. Vocabulary contract (variants.py header): the
  reference kwarg keeps the name `second_norm`, accepts `False` (=
  `"none"`), `"minmax"`, `"planned_relative"` — every value spelled the
  SAME in the config vocabulary and the reference's keyword vocabulary.
  This module must never import `delhi_psi.index`; the formulas stay
  identical (DEL-64 discipline).
- `tests/variants.py` gains ONE variant, `planned_relative`, overriding
  `second_normalization` only, **appended last** in `VARIANTS` so the
  existing rows of `variants_expected_values.csv` stay byte-identical and
  the new section appends after them (the DEL-57 `mean_rank` precedent).
- Variant pins (TDD, watched red): per-settlement `rel_psi` values against
  the reference; Planned rows exactly 1.0; the UV-≈-0.86-style ordering
  survives; both `norm_psi` and `rel_psi` row sections appear in the
  regen CSV for the new variant.

### 4. Blast radius (the DEL-48/20/34/57 constraint, refined by DEL-64)

- `tests/fixtures/*/production/*.csv` and `expected_values.csv`
  (hand-ratified anchors): **byte-untouched** — no shipped profile uses
  the new value.
- `variants_expected_values.csv`: **append-only growth** (new variant's
  rows at the end, existing bytes identical to HEAD).
- Both regenerate from the reference; the CI drift guard covers them.

### 5. The relative-means table generator

- One new `scripts/` generator (DEL-59 discipline: `--splice` mode +
  `--out` refusing to overwrite prose), writing the per-category mean of
  `rel_psi` as a committed Markdown block under `docs/data/`,
  watch-verified against its committed block (byte-identity test, same
  pattern as the other `measure_*` blocks).
- The table needs `rel_psi` in the frame, so it references the same
  computation entry points the other generators use; the gated real-data
  tests reuse the warm measure cache where applicable.

### 6. Docs

- `docs/data/psi_columns.md`: the emitted-column contract — `norm_psi`
  under `minmax` and `planned_relative`; `rel_psi` under
  `planned_relative`; absent otherwise.
- `config` reference text: the three values and the `true`-now-errors
  migration rule.
- Cross-reference the decision doc and this spec from the DEL-63 ticket.

## Out of scope (explicit)

- DEL-31: the ratified profile adopting `planned_relative` (after owner
  ratification of numbers).
- DEL-32: recalculation of the reported tables; DEL-33: the main figure.
- July-2025-baseline regeneration — final numbers come from the ratified
  run.

## Acceptance

1. TDD red→green on every pin listed in § 3, plus the guards in § 2.
2. Full smoke (`uv run pytest -q -W error`) green.
3. CI green — including the append-only fixture check (the drift guard
   proves every pre-existing row is byte-identical).
4. The relative-means block regenerates byte-identically and its pinned
   Planned-mean-1.00 property is exact.
5. The `/ship` record: plan + review round (finders/judge per the
   DEL-61/64 lineup), both test outputs in the PR.
