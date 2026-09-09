# Splice for the Remaining Five Documents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the five `docs/data/` documents that still lack it an `--out`/`--splice` path, by separating each generator's diagnostics (stderr) from its rendered blocks (stdout), and routing the blocks through the shared `emit`/`emit_check` DEL-58 built.

**Architecture:** No new machinery. `scripts/_measure_common.py` already owns `splice_blocks`, `holds_prose`, `emit` and `emit_check`; every one of the five scripts already imports `render` from it. Each script's `main` moves its `print(...)` provenance lines to stderr, accumulates its blocks into one `text`, and calls `emit`.

**Tech Stack:** Python 3, stdlib argparse, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-09-del-59-splice-all-documents-design.md`

## Global Constraints

- **Every committed `docs/data/*.md` must be byte-identical.** `git diff --stat docs/data/` must print nothing. If one moves, STOP and report.
- **`scripts/_measure_common.py` is USED, NOT MODIFIED.** If it genuinely needs a change to serve a sixth caller, report that as a finding rather than editing it quietly.
- No change to any measured value, statistic, or block field. No new dependency.
- The five scripts are named literally below. **Never derive the set by globbing `scripts/measure_*.py`** — `inventory_barriers.py` has no `measure_` prefix and would be silently dropped.
- The repo runs pytest with `-W error`. Warnings are failures.
- TDD is mandatory: write the failing test, run it, watch it fail, then implement.
- Run only your own task's test files. Do not background a pytest process.

## The document ↔ generator mapping (literal, pinned by a test in Task 3)

| document | generator |
|---|---|
| `docs/data/barriers.md` | `scripts/inventory_barriers.py` |
| `docs/data/roads_access.md` | `scripts/measure_roads_access.py` |
| `docs/data/rule_effects.md` | `scripts/measure_rule_effects.py` |
| `docs/data/psi_columns.md` | `scripts/measure_psi_columns.py` |
| `docs/data/layer_pathologies.md` | `scripts/measure_layer_pathologies.py` |

## Facts verified before this plan was written (9 Sep 2026)

- All five scripts already use `argparse`, so this extends an existing parser in each rather than introducing one.
- All five interleave provenance lines with blocks on stdout (`print(f"work-dir: ...")` … `print(render(...))`). `measure_psi_columns.py` also prints a `WARNING:` line *after* its block.
- **Moving diagnostics to stderr breaks no existing test.** All five drift tests read output with `parse_block(proc.stdout, name=...)`, and `_blocks` skips every line outside a `` ```text `` fence. The only two tests that read a stream directly assert argparse's own behaviour: `.err` for a missing `--verify-dir`, `.out` for `--help`.
- `measure_rule_effects.py` has `--only`, the same shape as `summarize_sweep`'s `--block`, so partial splice applies there.
- `layer_pathologies.md`'s only block sits in the preamble, before the first `## `; `psi_columns.md` carries a single unlabelled block.

---

### Task 1: the three single-shape scripts

**Files:**
- Modify: `scripts/inventory_barriers.py`, `scripts/measure_psi_columns.py`, `scripts/measure_layer_pathologies.py`
- Test: `tests/test_inventory_barriers.py`, `tests/test_measure_psi_columns.py`, `tests/test_layer_pathologies.py`

**Interfaces:**
- Consumes: `emit`, `emit_check` from `scripts._measure_common` (already exported; add to each script's existing import of `render`).
- Produces: nothing later tasks depend on. Task 2 repeats the pattern on the two `--verify-dir` scripts.

These three are batched because they are the same edit three times: no `--only`-style flag, one or two blocks, and a straight `print(render(...))` at the end.

- [ ] **Step 1: Write the failing tests**

**Read `tests/test_summarize_sweep.py` and `tests/test_rank_report.py` first** — DEL-58 already settled the exact shape of the `--out`/`--splice` assertions there, and copying that shape is more reliable than re-deriving it. Each of the three test modules also already has its own `subprocess.run` helper with the flags that script requires; use it rather than writing a new one.

Add to each of the three modules, three flag assertions:

1. `--out` and `--splice` both appear in the parser's `--help` output;
2. passing both at once exits non-zero (argparse's mutually-exclusive group);
3. `--out` pointed at a file holding prose refuses, exits 1, and leaves the file byte-identical.

For (3), build the prose file in `tmp_path`. **Do not point any test at a committed document.**

Then one behavioural test per script, using the module's existing runner and keeping whatever skip marker that module already applies:

```python
def test_stdout_carries_blocks_and_nothing_else():
    """DEL-59: the provenance lines move to stderr so --out and --splice
    have a clean stream to work with. `parse_block` always ignored those
    lines, so this is the first test that would notice one coming back."""
    proc = <the module's existing runner>
    assert "work-dir:" not in proc.stdout
    assert "work-dir:" in proc.stderr
```

Adapt the diagnostic string to what each script actually prints — `work-dir:` for `inventory_barriers` and `measure_psi_columns`, `cache:` for `measure_layer_pathologies`. Read each `main` and match it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_inventory_barriers.py tests/test_measure_psi_columns.py tests/test_layer_pathologies.py -q -W error`

Expected: FAIL — `unrecognized arguments: --out`.

- [ ] **Step 3: Implement**

In each of the three scripts:

1. Extend the existing `_measure_common` import to include `emit, emit_check`.
2. Add to `main`'s parser, after the existing arguments:

```python
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--out", default=None,
                        help="write the blocks here instead of stdout — "
                             "BLOCKS ONLY; REFUSES (exit 1) to overwrite a "
                             "target that already holds hand-written prose, "
                             "since that would delete every caption and "
                             "Finding — use --splice to refresh such a "
                             "document in place instead")
    target.add_argument("--splice", default=None,
                        help="refresh the blocks INSIDE this committed "
                             "document in place, preserving every caption "
                             "and Finding (DEL-59)")
```

3. Immediately after `args = parser.parse_args(argv)`, before any loading:

```python
    emit_check(out=args.out, splice=args.splice)
```

   These scripts do real geospatial work against `~/delhi_data` and several take minutes; the pre-flight is what stops a mistyped flag costing a full run.

4. Change every diagnostic `print(...)` in `main` to write to stderr, adding `import sys` where absent:

```python
    print(f"work-dir: {work_dir}", file=sys.stderr)
```

   **`measure_psi_columns.py`'s `WARNING:` line goes to stderr too** — it is a diagnostic, and leaving it on stdout would put it inside a spliced document.

5. Replace the block printing with accumulation plus one `emit`.

   For `inventory_barriers.py`, whose loop renders two named blocks:

```python
    text = "\n".join(render(report, name=name)
                     for name, report in blocks.items())
    emit(text, out=args.out, splice=args.splice)
    return 0
```

   For `measure_psi_columns.py` and `measure_layer_pathologies.py`, which render exactly one unlabelled block:

```python
    emit(render(report), out=args.out, splice=args.splice)
    return 0
```

   In `measure_psi_columns.py`, keep the `WARNING` check after the `emit`, writing to stderr.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_inventory_barriers.py tests/test_measure_psi_columns.py tests/test_layer_pathologies.py -q -W error`

Expected: PASS. Data-gated tests may skip; that is fine.

- [ ] **Step 5: Verify no document moved, then commit**

Run `git diff --stat docs/data/` — it must print nothing. Then stage the three scripts and their three test modules and commit with:

`feat(scripts): --out and --splice for barriers, psi_columns and layer_pathologies (DEL-59)`

---

### Task 2: the two `--verify-dir` scripts, including partial splice

**Files:**
- Modify: `scripts/measure_roads_access.py`, `scripts/measure_rule_effects.py`
- Test: `tests/test_measure_roads_access.py`, `tests/test_measure_rule_effects.py`

**Interfaces:**
- Consumes: `emit`, `emit_check`, and Task 1's established pattern.
- Produces: nothing later tasks depend on.

Separate from Task 1 because `measure_rule_effects.py` carries `--only`, which makes partial splice meaningful and needs its own test.

- [ ] **Step 1: Write the failing tests**

The same four assertions as Task 1 for both scripts, plus, for `measure_rule_effects.py` only:

```python
def test_only_plus_splice_refreshes_one_run_and_leaves_the_other(tmp_path):
    """The behaviour DEL-58 proved on the real phase6_sweep.md, here on a
    two-run document: refreshing `partial_barriers` must replace that run
    and leave `overlap_lending` untouched, along with every caption."""
    from scripts._measure_common import splice_blocks
    doc = ("## Partial barriers\n\nA caption.\n\n"
           "```text\nblock: partial_barriers\nx: 1\n```\n\n"
           "### Finding\n\nProse.\n\n"
           "## Overlap lending\n\nAnother caption.\n\n"
           "```text\nblock: overlap_lending\ny: 2\n```\n")
    out = splice_blocks(doc, "```text\nblock: partial_barriers\nx: 9\n```")
    assert "x: 9" in out
    assert "y: 2" in out
    assert "### Finding" in out
    assert "Another caption." in out
```

This exercises the splice against this document's real two-run shape without needing the real data the script requires.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_measure_roads_access.py tests/test_measure_rule_effects.py -q -W error`

Expected: FAIL — `unrecognized arguments: --out`.

- [ ] **Step 3: Implement**

Apply Task 1 Step 3's five changes to both scripts. Both print `layer:`, `verify-dir:` and `work-dir:` — all to stderr.

`measure_roads_access.py`'s block loop becomes:

```python
    text = "\n".join(render(report, name=name)
                     for name, report in blocks.items())
    emit(text, out=args.out, splice=args.splice)
```

`measure_rule_effects.py` already computes `wanted` from `--only`; join the rendered blocks over `wanted` into one `text` and `emit` it once, instead of printing each. Splicing with `--only` set then refreshes exactly the run(s) named, because `splice_blocks` leaves labels the fresh text does not mention alone — that is DEL-58's rule, not new behaviour.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_measure_roads_access.py tests/test_measure_rule_effects.py -q -W error`

Expected: PASS.

- [ ] **Step 5: Verify no document moved, then commit**

Run `git diff --stat docs/data/` — it must print nothing. Then stage the two scripts and their two test modules and commit with:

`feat(scripts): --out and --splice for roads_access and rule_effects, with partial splice (DEL-59)`

---

### Task 3: the mapping guard and the two awkward document shapes

**Files:**
- Test: `tests/test_measure_common.py`

**Interfaces:**
- Consumes: `splice_blocks` and `_block_spans` from `scripts._measure_common`; the five scripts from Tasks 1-2.
- Produces: nothing. This is the guard.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_measure_common.py`:

```python
# --- DEL-59: every docs/data document has a generator that can splice it ---
DOCUMENT_GENERATORS = {
    "barriers.md": "scripts/inventory_barriers.py",
    "roads_access.md": "scripts/measure_roads_access.py",
    "rule_effects.md": "scripts/measure_rule_effects.py",
    "psi_columns.md": "scripts/measure_psi_columns.py",
    "layer_pathologies.md": "scripts/measure_layer_pathologies.py",
    "phase6_sweep.md": "scripts/summarize_sweep.py",
}


def test_every_generated_document_has_a_generator_that_can_splice():
    """Pinned as a literal mapping, never a `scripts/measure_*.py` glob:
    `inventory_barriers.py` has no `measure_` prefix, so a glob silently
    drops `barriers.md` — which is exactly the mistake DEL-59's own ticket
    made before the code was written."""
    for document, script in DOCUMENT_GENERATORS.items():
        assert (REPO / "docs" / "data" / document).exists(), document
        source = (REPO / script).read_text()
        assert '"--splice"' in source, f"{script} cannot splice {document}"
        assert '"--out"' in source, f"{script} has no --out"


def test_the_mapping_covers_every_block_bearing_document():
    """A new document must not be able to appear without a splice path.
    `uso_final_vocabulary.md` is hand-written and carries no blocks, so it
    is legitimately absent."""
    for path in sorted((REPO / "docs" / "data").glob("*.md")):
        if not _block_spans(path.read_text()):
            continue
        assert path.name in DOCUMENT_GENERATORS, path.name


def test_splice_round_trips_a_preamble_block_document():
    """`layer_pathologies.md` is the only document whose block sits before
    the first `## ` heading. Every other document's blocks are inside a
    section, so this shape is the one most likely to be broken by a change
    to the run rule."""
    path = REPO / "docs" / "data" / "layer_pathologies.md"
    text = path.read_text()
    assert splice_blocks(text, text) == text


def test_splice_round_trips_an_unlabelled_block_document():
    """`psi_columns.md` carries a single block with no `block:` label, which
    `_label_runs` keys on the absence of a label."""
    path = REPO / "docs" / "data" / "psi_columns.md"
    text = path.read_text()
    assert splice_blocks(text, text) == text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_measure_common.py -q -W error -k "generator or preamble or unlabelled"`

Expected: the two mapping tests FAIL before Tasks 1-2 land (no `--splice` in those scripts). The two round-trip tests should PASS immediately — DEL-58's splice already handles both shapes. **If a round-trip test fails, that is a real finding about `splice_blocks`: report it, do not edit `_measure_common.py`.**

- [ ] **Step 3: Implement**

No production code. The tests are the deliverable. If `_block_spans` is not already imported in this module, add it to the existing import.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_measure_common.py -q -W error`

Expected: PASS.

- [ ] **Step 5: Commit**

Run `git diff --stat docs/data/` — it must print nothing. Then stage `tests/test_measure_common.py` and commit with:

`test(docs): pin the document-to-generator mapping and the two awkward shapes (DEL-59)`

---

## Self-review

**Spec coverage.** § 3 diagnostics-to-stderr and `emit`/`emit_check` → Tasks 1 and 2. § 4 the preamble and unlabelled shapes → Task 3. § 5 partial splice → Task 2. § 6 what must not move → every task's Step 5. § 7's mapping guard → Task 3.

**Placeholders.** Task 1 Step 1 deliberately points at `tests/test_summarize_sweep.py` and `tests/test_rank_report.py` for the help-text assertion shape rather than restating it, because those files already contain the form DEL-58 settled and copying them is more reliable than re-deriving it.

**Type consistency.** `emit(text, *, out=None, splice=None)` and `emit_check(*, out=None, splice=None)` are used identically in all five scripts, matching their signatures in `_measure_common.py`.

**One risk worth naming, and it is not the splice.** The splice machinery is proven — DEL-58's review round-tripped every one of these documents byte-identically. The risk here is the **stream separation**: a diagnostic `print` left on stdout would land inside a spliced document, and the existing drift tests cannot see it, because `parse_block` ignores every line outside a fence. That is why Task 1's behavioural test asserts on stdout's *shape* rather than only on its blocks — it is the only test in this plan that would catch a missed `print`.
