# DEL-36: Distance-band stability on the ratified rules — design spec

**Status:** content fixed by the 22 Oct call (item 2: the appendix table is
the point) and WORKPLAN's Phase 6 note ("the reported variants … re-run
against the ratified profile once Phase 4 unblocks"). No new owner question.

**What this ticket is.** Re-point the 1/5/10 km distance sweep at the
ratified rules and produce the appendix stability table. The sweep's own
governing principle (DEL-55 spec § 6.1) still holds: **PSI levels are not
comparable across points** (Eq. 2's min-max runs per run), so the table
reports rank stability and within-run diagnostics only — never a level
difference between bands.

## Design

1. **Three profiles, rebuilt on `ratified-2026`.** `band-1km.yaml`,
   `band-5km.yaml`, `band-10km.yaml` become the ratified methodology with
   exactly two values changed — `adjacency.rule: within_distance` and
   `adjacency.max_distance_km: {1,5,10}` — so the diff against the
   ratified run is attributable to the band alone. Outputs stay
   `[popdensity]` / `[csv]` (the sweep reads CSV; DEL-55 § 4.3).
   `band-0km` (the intersection comparison, DEL-39) is untouched.
2. **The summariser learns its baseline.** `summarize_sweep` hard-wires
   `delhi_psi_code-2025_popdensity_2020.csv` and
   `colonies_neighbors.joblib`. It gains `--baseline-profile`
   (default `code-2025`), and one pure helper
   `baseline_paths(profile, baseline_dir)` derives both names from
   `load_config(profile)` — the profile name for the CSV, and
   `paths.neighbors_artifact` for the artifact (which is exactly how
   `code-2025` keeps its legacy name while every other profile is
   profile-named). No behaviour change at the default.
3. **The run.** `scripts/run_sweep.py --group bands --work-dir
   /tmp/del36-sweep` (never the data dir; the script enforces that),
   then `summarize_sweep --work-dir /tmp/del36-sweep --baseline-dir
   ~/delhi_data/phase4_ratified --baseline-profile ratified-2026`.
4. **The document.** A new `docs/data/distance_bands.md`: the existing
   `points` and `ordering` blocks (diagnostics and rank stability), the
   prose explaining the governing principle and naming the baseline as the
   ratified run, and the measured cost of each band. Registered in
   `tests/test_measure_common.DOCUMENT_GENERATORS` (`summarize_sweep`
   already supports `--splice`).

## Acceptance

1. TDD: `baseline_paths` is written red first; the default keeps every
   existing sweep test byte-identical.
2. The three band fixtures regenerate; only those three change.
3. The sweep completes; the five figures' rule (no level comparisons) is
   kept — the table has ranks, agreement flags and diagnostics only.
4. Smoke + CI green; Jira DEL-36 closed with the stability one-liner.

## Out of scope

- Re-running the decay/adjacency sweep groups (DEL-37/39).
- Any change to `code-2025` or the ratified outputs.
