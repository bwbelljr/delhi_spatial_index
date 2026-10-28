# Phase 5 checkpoint — audit (DEL-6), 2026-10-28

Phase 5 is a checkpoint, not a work phase: "Phases 1 + 2 + 4 together produce
the minimum credible revision (correct code, verified index, non-urban
dropped, new categories)". This note checks each clause against the repo as
it stands, so the milestone rests on evidence rather than on prose.

| criterion | evidence | verdict |
|---|---|---|
| **correct code** | `uv run pytest -q -W error` → 1,251 passed, 13 skipped (28 Oct, `main` `7941fde`); production and the independent reference agree at 1e-12 on both fixture cities | met |
| **verified index** | the oracle suite pins the maths; the real-data recalculation ran under the ratified profile (4,062 settlements scored, both denominators) and its outputs are delivered with a manifest | met |
| **non-urban dropped** | ratified exclusion `[RV, Industrial, Other]` at `post_neighbors` with `contributes`; the recalculation's row count dropped by exactly those 69 settlements | met |
| **new categories** | the decision was to keep the `uso-10` identity mapping (no collapse; DEL-29 parked), so the "new categories" clause is the exclusion set above, not a remap | met, as decided |
| **minimum credible revision** | the paper's headline figure now renders from the recalculation (`docs/figures/ratified-2026/`), and the variant appendix shows the main claims are stable except for the published bbox adjacency | met |

**Conclusion:** the checkpoint is passed. What remains before a public
release is Phase 7's licence and shipping work (DEL-56 → DEL-47), not any
Phase 5 item.

**Scope note:** this audit confirms the state; it does not re-open the
category-collapse question. That lives, explicitly, on DEL-29 (parked by
Raj on 28 Aug).
