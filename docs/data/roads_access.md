# Road access, and what `roads: eq4_own_only` does to today's numbers

Raj ratified Eq. 4 as the manuscript writes it — each colony counts only the
roads inside its own boundary (decision log
`docs/decisions/2026-08-28-raj-methodology-decisions.md` § 2, DEL-22/DEL-49).
The premise on the call was inverted: the July 2025 code DECAYS roads like
clinics, so this is a change from the published numbers, and Raj is owed the
size of it. Two measurements answer that, both produced by
`scripts/measure_roads_access.py`, which reads the layers named by the
`code-2025` profile and writes nothing under the data directory.
`tests/test_measure_roads_access.py` re-runs it and compares both blocks (it
skips when the data is not present).

Numbers quoted in prose below in `backticks` are block values verbatim;
percentages and other derived quantities are written with a `%` sign or
without backticks.

- **Run date:** _pending — the run step fills this in_
- **Inputs:** _pending_
- **Commit:** _pending_
- **Command:** `uv run python scripts/measure_roads_access.py --config code-2025 --verify-dir <verify-dir> --work-dir <work-dir>`

## Block `access` — road access on the layer, no PSI

For every settlement in the deduplicated, reprojected universe (the same
loader the pathology measurement uses, so these counts describe exactly what
the pipeline scores):

- `road_inside_<TYPE>` — the settlement's polygon contains a positive length
  of the major-road layer. This is the membership `delhi_psi.index.road_lengths`
  uses, so it is `road_length > 0` in today's outputs; a road that only
  touches the boundary at a point does not count. That equivalence is not
  asserted here in prose and hoped for: with `--verify-dir` given the script
  compares its own `road_inside` set against the `road_length` column of the
  `code-2025` output CSV and RAISES if they differ, so the block below is
  either a true description of today's outputs or it was never printed.
- `road_via_neighbor_<TYPE>` — no road of its own, but at least one
  **`touch`** neighbour (positive shared border — Raj's ratified rule, DEL-19,
  not today's bbox) has one. A neighbour may be a dropped type: dropped
  settlements still lend (semantics (a)), and a road in the rural village next
  door is exactly what Raj asked about.
- `no_road_<TYPE>` — neither.

Counts are integers; shares are derived in the prose below, so the drift test
stays exact.

## Block `one_factor` — the effect on today's numbers

`code-2025` with ONE value changed, `methodology.roads: eq4_own_only`, run
against the SAME neighbours artifact (the roads formula is applied downstream
in `pipeline.index_frames`, and `pipeline.methodology_stamp` carries adjacency
and barrier only, so no re-`preprocess` is needed — the script asserts that
before it runs). The `decayed` side is READ from the proven `code-2025` run,
never recomputed. Per denominator and reported type: `n`, the mean `road_idx`
and mean `unnorm_psi` under each formula, and `road_idx_zeroed` — how many
settlements had a decayed road index above zero and fall to exactly zero
because everything they had was borrowed.

## When this reopens the decision

Stated before the numbers, so it is a rule and not a reaction. **The roads
decision (own-only) stands unless one of these is true:**

1. the JJC mean `unnorm_psi` falls by more than 20 % relative to its
   `code-2025` value under either denominator; or
2. the ordering of mean `unnorm_psi` between JJC and Planned flips.

Either would mean the switch changes the paper's headline comparison rather
than refining it, and it goes back to Raj as a question instead of a
correction.

*(The provenance values above, the two fenced blocks and the three-sentence
findings per block are written by the run step.)*
