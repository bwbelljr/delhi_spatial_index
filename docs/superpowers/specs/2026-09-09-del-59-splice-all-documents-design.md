# Splice for the remaining five documents — DEL-59

**Ticket:** DEL-59. **Branch:** `del-59-splice-all` off `main` at `cac2272`.
**Date:** 9 Sep 2026.

---

## 1. What DEL-58 left behind

DEL-58 gave `summarize_sweep.py` and `rank_report.py` a `--splice` mode, an
`--out` that refuses to eat prose, and a glob-parametrised guard asserting
every `docs/data/*.md` still has its prose. It fixed the two CLIs that had
`--out` — the ones where the accident had actually happened.

**The other five documents are still updated by hand-pasting stdout.** That
is the same failure mode performed by a human instead of a flag: each of
these documents pairs measured blocks with hand-written captions, and the
paste is what puts them at risk.

| document | generator |
|---|---|
| `docs/data/barriers.md` | `scripts/inventory_barriers.py` |
| `docs/data/roads_access.md` | `scripts/measure_roads_access.py` |
| `docs/data/rule_effects.md` | `scripts/measure_rule_effects.py` |
| `docs/data/psi_columns.md` | `scripts/measure_psi_columns.py` |
| `docs/data/layer_pathologies.md` | `scripts/measure_layer_pathologies.py` |

**Only four of the five are `measure_*`.** `inventory_barriers.py` predates
that naming convention, so any code or test that globs `scripts/measure_*.py`
silently misses `barriers.md`. The ticket's own description got this wrong
before the code was written. **The mapping is pinned as a literal table, never
derived from a filename pattern.**

## 2. The real problem, which is not "wire up `emit`"

`summarize_sweep` and `rank_report` print **blocks and nothing else** to
stdout. That is what makes `--out` a straight redirect.

These five do not. Every one of them interleaves provenance and progress
lines with the rendered blocks:

```python
print(f"layer: {cfg.paths.data_dir / cfg.layers.settlements.path}")
print(f"verify-dir: {verify_dir}")
print(f"work-dir: {work_dir}")
...
print(render(report, name=name))
```

`measure_psi_columns.py` also prints a `WARNING:` line *after* its block.

So `--out` cannot simply capture stdout here — it would write the diagnostic
lines into the document, and `--splice` would have nothing clean to splice.
**Separating the two streams is the work; the flags are the easy part.**

## 3. The fix: diagnostics to stderr, blocks to `emit`

Each script:

1. sends its provenance and progress lines to **stderr**;
2. accumulates its rendered blocks into one `text`, exactly as
   `summarize_sweep.main` already does;
3. calls `emit(text, out=args.out, splice=args.splice)`, with
   `emit_check(out=..., splice=...)` at the **top** of `main` so a wrong flag
   is refused before any measurement runs.

That last point matters more here than it did in DEL-58: these scripts do
real geospatial work against `~/delhi_data`, and several take minutes. The
pre-flight is what stops a mistyped flag costing a full run.

**This is safe, and it is verified rather than assumed.** All five drift
tests read their script's output with `parse_block(proc.stdout, name=...)`,
and `_blocks` skips every line that is not inside a `` ```text `` fence. The
diagnostic lines are already invisible to those tests, so moving them to
stderr changes nothing they assert. The two tests that *do* read a stream
directly assert on `capsys.readouterr().err` for a missing `--verify-dir`
and on `.out` for `--help`; argparse owns both, and neither moves.

## 4. `layer_pathologies.md` is the shape worth testing hardest

Its only fenced block sits in the document **preamble**, before the first
`## ` heading — the only one of the seven so shaped. DEL-58's fix round
generalised `_sections_missing_prose` to treat the preamble as a section, so
the guard already covers it. The splice must be exercised against that shape
explicitly rather than assumed to work, because every other document's blocks
sit inside a `## ` section.

`psi_columns.md` carries a single **unlabelled** block, which `_label_runs`
keys on the absence of a label. Also worth an explicit round trip.

## 5. Partial splice

`measure_rule_effects.py` has `--only`, the same shape as `summarize_sweep`'s
`--block`. Combining `--only` with `--splice` must refresh that one run and
leave the document's other run untouched — the behaviour DEL-58 built and
proved on the real `phase6_sweep.md`.

## 6. What must not move

- **Every committed `docs/data/*.md` byte-identical.** `git diff --stat
  docs/data/` empty. If one moves, the splice is rewriting something it
  should be preserving.
- No change to any measured value, statistic, or block field.
- No new dependency. No real-data run beyond what the existing gated drift
  tests already do.
- `scripts/_measure_common.py`'s splice machinery is **used, not modified** —
  if it needs a change to serve a sixth caller, that is a finding worth
  reporting, not a quiet edit.

## 7. Definition of done

- All five scripts take `--out` and `--splice`, mutually exclusive, routed
  through the shared `emit`/`emit_check`.
- Diagnostics on stderr; stdout carries blocks and nothing else.
- A test pinning the document↔generator mapping as a literal, so a future
  `measure_*` glob cannot silently drop `barriers.md`.
- A round-trip test for the preamble-block shape (`layer_pathologies.md`) and
  the unlabelled-block shape (`psi_columns.md`).
- `--only` plus `--splice` proved to refresh one run and leave the rest.
- Full suite green; every committed document byte-identical.
