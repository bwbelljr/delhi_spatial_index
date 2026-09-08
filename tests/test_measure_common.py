"""The shared plumbing every measurement script imports (spec § 1).

Nothing here needs the real data: the guard, the renderer and the parser are
pure. `DATA_DIR`/`needs_data`/`MEASURE_CACHE`/`needs_measure_cache`/
`prose_numbers` live here because the four doc-drift test modules all need
them and this is the module they all already import from.
"""
import os
import re
import shutil
from pathlib import Path

import pytest

from scripts._measure_common import (FENCE, _block_spans, parse_block,
                                     render, resolve_work_dir)

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DELHI_DATA_DIR", "~/delhi_data")).expanduser()

needs_data = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"real Delhi data not present at {DATA_DIR}")

# One settlement dedup per MACHINE, not one per test. A cold dedup of the real
# 4,357-polygon layer costs ~4.5 min (267 s, measured 5 Sep 2026) and the
# `touch` adjacency behind it is a second O(n^2) pass, so the two real-data
# drift tests that need the settlement universe (barriers, roads) share ONE
# staged work dir named by this variable and skip when it is unset. They are
# not gated on `needs_data` alone: this machine HAS the data, and a
# data-only gate would make every per-task suite run pay for two cold dedups.
# The run step (task 6) is the only place that exports it. The psi_columns
# drift test reads CSVs only and keeps the plain `needs_data` gate.
MEASURE_CACHE = os.environ.get("DELHI_PSI_MEASURE_CACHE")

needs_measure_cache = pytest.mark.skipif(
    not MEASURE_CACHE,
    reason="set DELHI_PSI_MEASURE_CACHE to run the real-data drift check")


def prose_numbers(text):
    """Backticked numeric literals in PROSE — outside every fenced block.

    The guard behind them: a number a document quotes in prose must be a
    value the block actually carries, or the prose has drifted from the
    measurement. Derived quantities (shares) are written with a `%` sign or
    without backticks, so they are deliberately not matched here.
    """
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return {match.replace(",", "")
            for match in re.findall(r"`(-?\d[\d,]*(?:\.\d+)?)`", prose)}


def assert_prose_numbers_come_from_the_blocks(text, blocks):
    values = {str(value) for block in blocks for value in block.values()}
    for number in sorted(prose_numbers(text)):
        assert number in values, (number, sorted(values))


# --- the work-dir guard ------------------------------------------------
def test_the_default_work_dir_is_a_fresh_directory_each_time():
    """~/delhi_data is bisynced to the shared drive: a scratch directory
    derived from it propagates to everyone. The default never is."""
    made = [resolve_work_dir(), resolve_work_dir()]
    try:
        assert made[0] != made[1], "each run must get its own work dir"
        for path in made:
            assert path.is_dir()
            assert path != DATA_DIR and DATA_DIR not in path.parents
    finally:
        for path in made:
            shutil.rmtree(path, ignore_errors=True)


def test_an_explicit_work_dir_is_used_as_given():
    assert resolve_work_dir("/somewhere/else") == Path("/somewhere/else")


@pytest.mark.parametrize("inside", ["", "nested", "nested/deeper"])
def test_a_work_dir_inside_the_data_dir_is_refused(tmp_path, inside):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    candidate = data_dir / inside if inside else data_dir
    with pytest.raises(SystemExit) as exc:
        resolve_work_dir(str(candidate), data_dir=data_dir)
    assert "is inside the data directory" in str(exc.value)


def test_a_work_dir_outside_the_data_dir_is_created(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    work_dir = resolve_work_dir(str(tmp_path / "work" / "roads"),
                                data_dir=data_dir)
    assert work_dir.is_dir()


# --- render / parse_block ----------------------------------------------
def test_render_and_parse_block_round_trip():
    report = {"settlements": 4357, "area_km2_min": "2.30282e-09"}
    assert parse_block(render(report)) == {"settlements": "4357",
                                           "area_km2_min": "2.30282e-09"}


def test_render_labels_a_named_block_and_parse_block_selects_it():
    text = "\n".join([render({"a": 1}, name="access"),
                      render({"b": 2}, name="one_factor")])
    assert render({"a": 1}, name="access").splitlines()[1] == "block: access"
    assert parse_block(text, name="access") == {"a": "1"}
    assert parse_block(text, name="one_factor") == {"b": "2"}


def test_parse_block_without_a_name_returns_the_first_block():
    text = "\n".join([render({"a": 1}, name="access"),
                      render({"b": 2}, name="one_factor")])
    assert parse_block(text) == {"a": "1"}


def test_parse_block_raises_for_an_unknown_name():
    with pytest.raises(ValueError, match="no_such_block"):
        parse_block(render({"a": 1}, name="access"), name="no_such_block")


def test_parse_block_raises_on_an_unterminated_block():
    with pytest.raises(ValueError, match="unterminated"):
        parse_block(f"{FENCE}\na: 1\n")


def test_the_committed_pathology_block_survives_the_move_byte_for_byte():
    """Task 1 moves render/parse_block out of the pathology script; the
    document it has produced since 28 Aug must still render byte for byte."""
    doc = (REPO / "docs" / "data" / "layer_pathologies.md").read_text()
    assert render(parse_block(doc)) in doc


# --- the prose guard ---------------------------------------------------
def test_prose_numbers_finds_only_backticked_numbers_outside_the_blocks():
    text = "\n".join(["prose says `4,357` and `0.05` and 4069 and `touch`",
                      render({"settlements": 4357})])
    assert prose_numbers(text) == {"4357", "0.05"}


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
    """The strongest statement available that the splice preserves what it
    is not replacing (spec § 8). It is also the test the plan review's
    first draft FAILED: splicing a document into itself feeds `_joined` a
    fresh run that already carries the document's blank separators."""
    assert splice_blocks(DOC_ONE_RUN, DOC_ONE_RUN) == DOC_ONE_RUN


def test_splice_does_not_fabricate_a_trailing_newline():
    """A document ending at its last block with no trailing newline must
    come back with no trailing newline (plan review, finding 2)."""
    doc = "## S\n\ncaption\n\n```text\nblock: a\nx: 1\n```"
    out = splice_blocks(doc, "```text\nblock: a\nx: 9\n```")
    assert out.endswith("```")
    assert "x: 9" in out
    assert out.startswith("## S\n\ncaption\n\n")


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


# --- emit (DEL-58) -------------------------------------------------------
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


# --- the prose-aware drift guard (DEL-58) --------------------------------
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
