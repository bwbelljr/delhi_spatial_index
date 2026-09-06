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

from scripts._measure_common import (FENCE, parse_block, render,
                                     resolve_work_dir)

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
