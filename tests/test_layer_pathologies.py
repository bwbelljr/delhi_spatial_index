"""The messy tier's real-data premises, from a reproducible source (§ 5).

Data-gated: CI runs on a bare runner, so the two tests that actually run the
measurement skip without ~/delhi_data. The three that do not need data — the
committed document's shape, its provenance header, and where the dedup cache
is allowed to go — always run.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.measure_layer_pathologies import parse_block, resolve_cache_dir

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "layer_pathologies.md"
DATA_DIR = Path(os.environ.get("DELHI_DATA_DIR", "~/delhi_data")).expanduser()

needs_data = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"real Delhi data not present at {DATA_DIR}")

COUNT_KEYS = ("settlements", "rectangles", "multipolygons", "isolated_bbox",
              "no_population", "overlapping_pairs")
AREA_KEYS = ("area_km2_min", "area_km2_median", "area_km2_max")
POINT_SERVICES = ("bank", "health", "police", "ration", "school", "transport")


@pytest.fixture(scope="module")
def committed():
    assert DOC.exists(), f"missing {DOC} — run the script and commit its block"
    return parse_block(DOC.read_text())


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """ONE run of the script, shared by every data-gated test in this file:
    the pipeline's O(n^2) dedup takes ~3 minutes on a cold cache."""
    if not DATA_DIR.exists():
        pytest.skip(f"real Delhi data not present at {DATA_DIR}")
    cache = tmp_path_factory.mktemp("pathologies_cache")
    before = set(DATA_DIR.rglob("*"))
    proc = subprocess.run(
        [sys.executable, "scripts/measure_layer_pathologies.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--cache-dir", str(cache)],
        cwd=REPO, capture_output=True, text=True)
    after = set(DATA_DIR.rglob("*"))
    assert proc.returncode == 0, proc.stderr[-4000:]
    return parse_block(proc.stdout), before, after


def test_the_doc_has_the_fenced_block_with_every_required_key(committed):
    for key in COUNT_KEYS:
        assert key in committed, key
        assert committed[key].isdigit(), (key, committed[key])
    for key in AREA_KEYS:
        assert key in committed, key
        float(committed[key])          # parses, whatever the formatting
    services = sorted(key[len("multi_settlement_points_"):]
                      for key in committed
                      if key.startswith("multi_settlement_points_"))
    assert services == sorted(POINT_SERVICES), services
    assert set(committed) == (set(COUNT_KEYS) | set(AREA_KEYS)
                              | {f"multi_settlement_points_{s}"
                                 for s in POINT_SERVICES})


def test_the_doc_records_its_provenance():
    """Counts without a date, a layer and a commit are not evidence."""
    head = DOC.read_text().split("```text")[0]
    for label in ("**Run date:**", "**Layer:**", "**Commit:**", "**Command:**"):
        assert label in head, label


def test_the_cache_dir_default_is_a_fresh_directory_outside_the_data_dir():
    """~/delhi_data is bisynced to the shared drive: a cache written there
    propagates to everyone. The default must never be derived from it."""
    made = [resolve_cache_dir(), resolve_cache_dir()]
    try:
        assert made[0] != made[1], "each run must get its own cache"
        for path in made:
            assert path.is_dir()
            assert path != DATA_DIR and DATA_DIR not in path.parents
    finally:
        for path in made:
            shutil.rmtree(path, ignore_errors=True)
    assert resolve_cache_dir("/somewhere/else") == Path("/somewhere/else")


@needs_data
def test_a_fresh_run_reproduces_the_committed_counts(committed, fresh):
    """Counts only: the prose header (date, layer, commit) is never compared,
    and the three float area keys are checked for presence and parseability
    by the shape test instead of by text equality."""
    measured, _, _ = fresh
    assert set(measured) == set(committed)
    for key, value in committed.items():
        if key.startswith("area_km2_"):
            continue
        assert measured[key] == value, (key, measured[key], value)


@needs_data
def test_the_script_writes_nothing_under_the_data_directory(fresh):
    """~/delhi_data is bisynced hourly, so an unrelated file can appear
    mid-run; assert specifically that no dedup artifact — the only thing this
    script could ever write there — was created."""
    _, before, after = fresh
    created = after - before
    leaked = sorted(str(path) for path in created if ".dedup." in path.name)
    assert leaked == [], leaked
