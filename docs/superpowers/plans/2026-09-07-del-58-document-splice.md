# Regenerable `docs/data/` documents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `docs/data/*.md` regenerable — refresh a document's measured blocks without destroying the hand-written prose around them, refuse the command that destroyed it once already, and add the test that would have caught it.

**Architecture:** One implementation in `scripts/_measure_common.py` (which already owns `FENCE`, `render`, `_blocks`, `parse_block`), imported by both `scripts/summarize_sweep.py` and `scripts/rank_report.py`. The splice replaces contiguous same-label *runs* of fenced blocks and preserves every other byte, including the blank-line separator a document uses between blocks in a run.

**Tech Stack:** Python 3, stdlib only, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-07-del-58-document-splice-design.md`

## Global Constraints

- **No new dependency.** Stdlib only.
- **Every committed `docs/data/*.md` must be byte-identical when this branch merges.** Verify with `git diff --stat docs/data/` — it must be empty. If a document moves, the splice is rewriting something it should preserve.
- **No real-data run.** Nothing in this ticket reads `~/delhi_data`. Tests that need sweep data keep their existing `@needs_sweep_data` gates.
- **No licence file** (DEL-56, pending Raj).
- **No change to any statistic, block field, or rendered value.**
- **TDD is mandatory.** Write the failing test, run it, watch it fail, then implement. Code written before its test is unverified — discard and redo.
- Run only your own task's test files. Do not run the full suite in the background; the controller runs it at the end.
- The fence is `` ```text `` (`_measure_common.FENCE`); a block closes with a bare `` ``` ``.

## Facts measured against the seven committed documents (7 Sep 2026)

These are not assumptions. They were verified before this plan was written, and the implementation may rely on them:

| document | blocks | label runs | intra-run gap |
|---|---|---|---|
| `phase6_sweep.md` | 54 | `points`, `ordering`, `gap`, `denominator_check` | 1 blank line, 50× |
| `barriers.md` | 2 | `layers`, `attributes` | — |
| `roads_access.md` | 2 | `access`, `one_factor` | — |
| `rule_effects.md` | 2 | `partial_barriers`, `overlap_lending` | — |
| `layer_pathologies.md` | 1 | *(unlabelled)* | — |
| `psi_columns.md` | 1 | *(unlabelled)* | — |
| `uso_final_vocabulary.md` | 0 | — | — |

- **No label appears as two separate runs** in any document.
- **No prose is interleaved inside a run** in any document.
- **Every `## ` section carrying a block also carries prose** — 0 violations across all seven.

---

### Task 1: `splice_blocks` and its refusals

**Files:**
- Modify: `scripts/_measure_common.py`
- Test: `tests/test_measure_common.py`

**Interfaces:**
- Consumes: `FENCE`, and the existing `_blocks` scanner it refactors.
- Produces, for Tasks 2 and 3:
  - `splice_blocks(document: str, fresh: str) -> str`
  - `holds_prose(text: str) -> bool`
  - `_block_spans(text) -> list[tuple[str | None, int, int]]` — `(label, start_line, end_line_exclusive)`
  - `_label_runs(text) -> list[tuple[str | None, int, int, list[str]]]` — `(label, start, end, separator_lines)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_measure_common.py`:

```python
# --- splice (DEL-58) ---------------------------------------------------
from scripts._measure_common import holds_prose, splice_blocks

DOC_ONE_RUN = """\
# Title

## Section

A hand-written caption.

```text
block: points
point: a
n: 1
```

```text
block: points
point: b
n: 2
```

### Finding

The prose that `--out` destroys.
"""


def test_splice_replaces_a_run_and_keeps_every_other_byte():
    fresh = ("```text\nblock: points\npoint: a\nn: 9\n```\n"
             "```text\nblock: points\npoint: b\nn: 8\n```")
    out = splice_blocks(DOC_ONE_RUN, fresh)

    assert "A hand-written caption." in out
    assert "### Finding" in out
    assert "The prose that `--out` destroys." in out
    assert "n: 9" in out and "n: 8" in out
    assert "n: 1" not in out and "n: 2" not in out


def test_splice_preserves_the_documents_own_separator():
    """The committed `phase6_sweep.md` puts one blank line between blocks in
    a run (50 times); script output puts none. A splice that wrote the fresh
    text verbatim would silently reformat the document on every refresh."""
    fresh = ("```text\nblock: points\npoint: a\nn: 9\n```\n"
             "```text\nblock: points\npoint: b\nn: 8\n```")
    out = splice_blocks(DOC_ONE_RUN, fresh)
    assert "n: 9\n```\n\n```text" in out


def test_splice_accepts_a_different_block_count():
    """13 sweep points becoming 11 is the real use; a one-to-one rule could
    not express it."""
    fresh = "```text\nblock: points\npoint: only\nn: 5\n```"
    out = splice_blocks(DOC_ONE_RUN, fresh)
    assert out.count("block: points") == 1
    assert "### Finding" in out


def test_splice_leaves_labels_the_fresh_text_does_not_mention():
    doc = ("## S\n\ncaption\n\n```text\nblock: a\nx: 1\n```\n\n"
           "```text\nblock: b\ny: 2\n```\n")
    out = splice_blocks(doc, "```text\nblock: a\nx: 9\n```")
    assert "x: 9" in out
    assert "y: 2" in out


def test_splice_round_trips_its_own_blocks_byte_for_byte():
    assert splice_blocks(DOC_ONE_RUN, DOC_ONE_RUN) == DOC_ONE_RUN


def test_splice_refuses_a_label_the_document_does_not_have():
    with pytest.raises(ValueError, match="ordering"):
        splice_blocks(DOC_ONE_RUN, "```text\nblock: ordering\nx: 1\n```")


def test_splice_refuses_a_label_that_appears_as_two_runs():
    doc = ("```text\nblock: a\nx: 1\n```\n\nprose between\n\n"
           "```text\nblock: a\nx: 2\n```\n")
    with pytest.raises(ValueError, match="two separate runs|separate runs"):
        splice_blocks(doc, "```text\nblock: a\nx: 9\n```")


def test_splice_refuses_a_document_with_no_blocks():
    with pytest.raises(ValueError, match="no .* block"):
        splice_blocks("# Just prose\n", "```text\nblock: a\nx: 1\n```")


def test_holds_prose_sees_text_outside_blocks_only():
    assert holds_prose("## S\n\ncaption\n\n```text\nx: 1\n```\n")
    assert holds_prose("### Finding\n")
    assert not holds_prose("```text\nx: 1\n```\n")
    assert not holds_prose("```text\nx: 1\n```\n\n```text\ny: 2\n```\n")
    assert not holds_prose("")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_measure_common.py -q -W error`
Expected: FAIL — `ImportError: cannot import name 'holds_prose'`.

- [ ] **Step 3: Implement**

In `scripts/_measure_common.py`, refactor the existing scanner so there is
exactly one, then add the splice. Replace the body of `_blocks` and add
below it:

```python
def _block_spans(text):
    """[(label, start_line, end_line_exclusive)] for every fenced block.

    THE one scanner: `_blocks` reads bodies out of these spans, and the
    splice replaces the spans themselves. Two scanners that disagree about
    where a block ends is the drift this module exists to prevent.
    """
    lines = text.splitlines()
    out, index = [], 0
    while index < len(lines):
        if lines[index].strip() != FENCE:
            index += 1
            continue
        start = index
        index += 1
        label, body, closed = None, {}, False
        while index < len(lines):
            line = lines[index]
            index += 1
            if line.strip() == "```":
                closed = True
                break
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "block" and label is None and not body:
                label = value
            else:
                body[key] = value
        if not closed:
            raise ValueError(f"unterminated {FENCE} block")
        out.append((label, body, start, index))
    return out


def _blocks(text):
    """[(label or None, {key: value})] for every fenced block, in order."""
    return [(label, body) for label, body, _, _ in _block_spans(text)]


def _label_runs(text):
    """[(label, start, end, separator)] — maximal sequences of contiguous
    same-label blocks, plus the lines the document puts between them.

    A run BREAKS on intervening prose, so a label separated by a paragraph
    is two runs, not one. That is what makes the ambiguity refusal in
    `splice_blocks` meaningful: a label with two runs has no single place
    for the fresh blocks to go, and no committed document has one.

    `separator` is the lines between the run's first two blocks — one blank
    line in `docs/data/phase6_sweep.md`, nothing in the other documents.
    Preserving it is why a splice does not silently reformat a document.
    """
    lines = text.splitlines()
    runs = []
    for label, _, start, end in _block_spans(text):
        if runs:
            prev_label, prev_start, prev_end, separator = runs[-1]
            gap = lines[prev_end:start]
            if prev_label == label and not any(line.strip() for line in gap):
                runs[-1] = (label, prev_start, end,
                            gap if separator is None else separator)
                continue
        runs.append((label, start, end, None))
    return [(label, start, end, separator or [])
            for label, start, end, separator in runs]


def splice_blocks(document, fresh):
    """`document` with each of `fresh`'s block runs substituted in place.

    Everything outside those runs — headings, captions, `### Finding`
    sections, blank lines — is preserved byte for byte. Labels `fresh` does
    not mention are left alone, which is what makes `--block points` refresh
    one run and keep the rest.

    Refuses rather than guesses (spec § 4). Every refusal writes nothing.
    """
    doc_lines = document.splitlines(keepends=True)
    fresh_lines = fresh.splitlines()
    doc_runs = _label_runs(document)
    fresh_runs = _label_runs(fresh)

    if not doc_runs:
        raise ValueError(
            f"the document has no {FENCE} block to splice into; "
            "--out writes a fresh blocks-only file")

    def _one_run_per_label(runs, what):
        seen = {}
        for label, *_ in runs:
            if label in seen:
                raise ValueError(
                    f"{what}: block label {label!r} appears as two separate "
                    "runs, so there is no single place to splice it")
            seen[label] = True
        return seen

    doc_labels = _one_run_per_label(doc_runs, "document")
    _one_run_per_label(fresh_runs, "fresh output")

    for label, *_ in fresh_runs:
        if label not in doc_labels:
            raise ValueError(
                f"block label {label!r} is not in the document (it has "
                f"{sorted(str(name) for name in doc_labels)}); splicing into "
                "the wrong file?")

    replacement = {}
    for label, start, end, _ in fresh_runs:
        replacement[label] = fresh_lines[start:end]

    out, cursor = [], 0
    for label, start, end, separator in doc_runs:
        out.extend(doc_lines[cursor:start])
        cursor = end
        if label not in replacement:
            out.extend(doc_lines[start:end])
            continue
        text = "\n".join(_joined(replacement[label], separator))
        out.append(text + "\n")
    out.extend(doc_lines[cursor:])
    return "".join(out)


def _joined(fresh_run_lines, separator):
    """The fresh run's lines re-joined with the document's own separator."""
    if not separator:
        return list(fresh_run_lines)
    out, block = [], []
    for line in fresh_run_lines:
        block.append(line)
        if line.strip() == "```":
            if out:
                out.extend(separator)
            out.extend(block)
            block = []
    out.extend(block)
    return out


def holds_prose(text):
    """True when `text` has any non-blank line outside a fenced block.

    The `--out` guard (spec § 5). Not "contains a `### Finding`": only two
    of the seven documents have Findings, and a caption above a block is
    worth no less than a Finding below it.
    """
    lines = text.splitlines()
    covered = set()
    for _, _, start, end in _block_spans(text):
        covered |= set(range(start, end))
    return any(line.strip() for index, line in enumerate(lines)
               if index not in covered)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_measure_common.py -q -W error`
Expected: PASS, and every pre-existing test in that file still passes (the `_blocks` refactor must not change its behaviour).

- [ ] **Step 5: Commit**

```bash
git add scripts/_measure_common.py tests/test_measure_common.py
git commit -m "feat(docs): splice_blocks — refresh a document's blocks without destroying its prose (DEL-58)"
```

---

### Task 2: `--splice` on both CLIs, and the `--out` refusal

**Files:**
- Modify: `scripts/_measure_common.py` (add `emit`)
- Modify: `scripts/summarize_sweep.py:1260-1288`
- Modify: `scripts/rank_report.py:134-149`
- Test: `tests/test_measure_common.py`, `tests/test_summarize_sweep.py`, `tests/test_rank_report.py`

**Interfaces:**
- Consumes from Task 1: `splice_blocks`, `holds_prose`.
- Produces: `emit(text, *, out=None, splice=None)` in `_measure_common.py` — the shared `--out`/`--splice` behaviour both CLIs call, so the refusal has one implementation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_measure_common.py`:

```python
from scripts._measure_common import emit


def test_emit_refuses_to_overwrite_a_document_holding_prose(tmp_path, capsys):
    doc = tmp_path / "doc.md"
    doc.write_text(DOC_ONE_RUN)
    with pytest.raises(SystemExit, match="--splice"):
        emit("```text\nblock: points\npoint: a\nn: 9\n```", out=str(doc))
    assert doc.read_text() == DOC_ONE_RUN


def test_emit_overwrites_a_blocks_only_file(tmp_path):
    target = tmp_path / "blocks.md"
    target.write_text("```text\nblock: points\npoint: a\nn: 1\n```\n")
    emit("```text\nblock: points\npoint: a\nn: 9\n```", out=str(target))
    assert "n: 9" in target.read_text()


def test_emit_writes_a_new_file(tmp_path):
    target = tmp_path / "new.md"
    emit("```text\nblock: a\nx: 1\n```", out=str(target))
    assert target.read_text() == "```text\nblock: a\nx: 1\n```\n"


def test_emit_splices_in_place(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(DOC_ONE_RUN)
    emit("```text\nblock: points\npoint: a\nn: 9\n```", splice=str(doc))
    text = doc.read_text()
    assert "n: 9" in text
    assert "### Finding" in text


def test_emit_prints_when_neither_flag_is_given(capsys):
    emit("```text\nblock: a\nx: 1\n```")
    assert "x: 1" in capsys.readouterr().out
```

Append to `tests/test_summarize_sweep.py`:

```python
def test_the_splice_flag_is_offered_alongside_out():
    help_text = S.build_parser().format_help()
    assert "--splice" in help_text
```

Append to `tests/test_rank_report.py` (import the module as the file already
does; if it imports it under a different alias, match that):

```python
def test_the_rank_report_offers_out_and_splice():
    help_text = build_parser().format_help()
    assert "--out" in help_text
    assert "--splice" in help_text
    assert "OVERWRITES" in help_text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_measure_common.py tests/test_rank_report.py -q -W error`
Expected: FAIL — `cannot import name 'emit'`, and the `--splice` assertions fail.

- [ ] **Step 3: Implement**

Add to `scripts/_measure_common.py`:

```python
def emit(text, *, out=None, splice=None):
    """The `--out` / `--splice` behaviour every measurement CLI shares.

    ONE implementation, because the failure this guards against — blocks
    written over a document's prose — happened once already, and a second
    copy of the guard is a second chance to get it wrong.
    """
    if splice:
        target = Path(splice)
        target.write_text(splice_blocks(target.read_text(), text))
        return
    if out:
        target = Path(out)
        if target.exists() and holds_prose(target.read_text()):
            raise SystemExit(
                f"{target} holds hand-written prose, and --out writes blocks "
                "only — it would delete every caption and Finding. Use "
                f"--splice {target} to refresh its blocks in place.")
        target.write_text(text + "\n")
        return
    print(text)
```

In `scripts/summarize_sweep.py`, add the flag beside `--out` and route
through `emit`:

```python
    parser.add_argument("--splice", default=None,
                        help="refresh the blocks INSIDE this committed "
                             "document in place, preserving every caption "
                             "and Finding — the safe way to update "
                             "docs/data/phase6_sweep.md (DEL-58)")
```

and replace the tail of `main`:

```python
    emit(text, out=args.out, splice=args.splice)
```

Do the same in `scripts/rank_report.py`, whose `--out` help must gain the
same `OVERWRITES ... prose` warning `summarize_sweep`'s already carries:

```python
    parser.add_argument("--out", default=None,
                        help="write the blocks here instead of stdout — "
                             "BLOCKS ONLY, OVERWRITES any prose in the "
                             "target; use --splice for a committed document")
    parser.add_argument("--splice", default=None,
                        help="refresh the blocks INSIDE this committed "
                             "document in place, preserving every caption "
                             "and Finding (DEL-58)")
```

Import `emit` from `scripts._measure_common` in both scripts (`rank_report`
already imports `render` from it).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_measure_common.py tests/test_rank_report.py tests/test_summarize_sweep.py -q -W error`
Expected: PASS (sweep-data-gated tests may skip; that is fine).

- [ ] **Step 5: Commit**

```bash
git add scripts/_measure_common.py scripts/summarize_sweep.py scripts/rank_report.py tests/test_measure_common.py tests/test_summarize_sweep.py tests/test_rank_report.py
git commit -m "feat(scripts): --splice on both document CLIs; --out refuses to eat prose (DEL-58)"
```

---

### Task 3: the prose-aware drift test

**Files:**
- Test: `tests/test_measure_common.py`

**Interfaces:**
- Consumes from Task 1: `_block_spans`, `holds_prose`.
- Produces: nothing other tasks use. This is the guard, not machinery.

- [ ] **Step 1: Write the failing test**

The test must fail against a destroyed document, so it is written with a
deliberately-destroyed fixture first, then pointed at the real directory.

Append to `tests/test_measure_common.py`:

```python
DOCS_DATA = REPO / "docs" / "data"


def _sections_missing_prose(text):
    """`## ` sections that carry a fenced block but no prose of their own."""
    lines = text.splitlines()
    covered = set()
    for _, _, start, end in _block_spans(text):
        covered |= set(range(start, end))

    missing, heading, prose, blocks = [], None, 0, 0
    def close():
        if heading is not None and blocks and not prose:
            missing.append(heading)
    for index, line in enumerate(lines):
        if index in covered:
            blocks += 1
            continue
        if line.startswith("## "):
            close()
            heading, prose, blocks = line.strip(), 0, 0
            continue
        if line.strip():
            prose += 1
    close()
    return missing


@pytest.mark.parametrize(
    "path", sorted(DOCS_DATA.glob("*.md")), ids=lambda p: p.name)
def test_every_committed_document_still_has_its_prose(path):
    """DEL-58's regression guard. `--out` pointed at a committed document
    writes BLOCKS ONLY — and the block-level drift tests cannot see it,
    because a document stripped of all its prose has identical blocks.

    Two clauses. The first is what actually fails on the accident: a
    blocks-only dump has no `## ` heading at all. The second catches the
    narrower loss of one section's caption. Both were verified true for all
    seven committed documents before this test was written.
    """
    text = path.read_text()
    assert "## " in text, f"{path.name} has no section heading — overwritten?"
    assert _sections_missing_prose(text) == []


def test_the_prose_guard_fails_on_a_blocks_only_document():
    """The guard's own test: a document reduced to its blocks — exactly what
    `--out` produces — must fail both clauses."""
    destroyed = "```text\nblock: points\npoint: a\nn: 1\n```\n"
    assert "## " not in destroyed
    assert not holds_prose(destroyed)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_measure_common.py -q -W error -k prose`
Expected: FAIL — `NameError`/`ImportError` on `_block_spans` if Task 1's export is missing; otherwise the parametrized test collects seven documents and passes. **If it passes immediately, prove it can fail:** temporarily delete a caption from a scratch copy of `docs/data/barriers.md`, run `_sections_missing_prose` on it, confirm non-empty, and report that in the task report.

- [ ] **Step 3: Implement**

No production code. The test is the deliverable. If Step 2 shows
`_block_spans` is not importable from `tests/test_measure_common.py`, add it
to that module's existing import from `scripts._measure_common`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_measure_common.py -q -W error`
Expected: PASS, seven parametrized cases among them.

- [ ] **Step 5: Verify no document moved, then commit**

```bash
git diff --stat docs/data/
git add tests/test_measure_common.py
git commit -m "test(docs): prose-aware drift guard over every docs/data document (DEL-58)"
```

`git diff --stat docs/data/` must print nothing. If it prints anything, stop
and report — a document changed, and no task in this plan is allowed to
change one.

---

## Self-review

**Spec coverage.** § 4 splice rule → Task 1. § 4's three refusals → Task 1
Steps 1/3. § 5 `--out` refusal → Task 2. § 6 prose-aware test → Task 3. § 8's
round-trip no-op → Task 1 `test_splice_round_trips_its_own_blocks_byte_for_byte`.
§ 8's "both callers share one implementation" → Task 2's `emit`.

**Placeholders.** None: every step carries the code it needs.

**Type consistency.** `_block_spans` returns 4-tuples `(label, body, start,
end)`; `_blocks` unpacks four and yields two; `_label_runs` and `holds_prose`
unpack four and use `start`/`end`. Task 3's `_sections_missing_prose` unpacks
four. Consistent throughout.

**One risk worth naming.** The spec's separator rule keys on the gap between
a run's *first two* blocks. A document with non-uniform intra-run spacing
would be normalised to its first gap. No committed document has non-uniform
spacing (`phase6_sweep.md` is 1 blank line, 50 times; every other run is a
single block or zero gap), so this cannot bite today, and normalising is the
benign outcome if it ever does.
