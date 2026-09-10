"""The rule-effect measurement (DEL-48, spec § 7).

The weight classification is proven on Oraculum, where the answer is known
by hand: 10 undirected bbox pairs = 20 directed links, of which A-D and D-A
are fractional and none is severed. The real-data drift test needs
DELHI_PSI_MEASURE_CACHE — the shared warm work dir the run step exports — or
it skips.
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from delhi_psi import cli
from scripts import measure_rule_effects
from scripts._measure_common import FENCE, holds_prose, parse_block, splice_blocks
from scripts.measure_rule_effects import (
    BLOCKS, WEIGHT_CLASSES, derived_profile, overlapping_neighbours,
    pcen_changes, shared_pair_counts, weight_classes,
)
from tests.oraculum_fixtures import oracle_profile_path
from tests.test_cli import data_dir  # noqa: F401 — the Oraculum data dir
from tests.test_measure_common import (
    DATA_DIR, MEASURE_CACHE, assert_prose_numbers_come_from_the_blocks,
    needs_measure_cache,
)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "rule_effects.md"
VERIFY_DIR = DATA_DIR / "phase3_verify"
PARTIAL_PROFILE = "partial-barriers-5m"


def test_weight_classes_counts_directed_links_by_class():
    frame = pd.DataFrame({
        "USO_AREA_U": ["A", "B", "C"],
        "nbrs_barrier_weight": [[("B", 1.0), ("C", 0.08)],
                                [("A", 1.0)],
                                [("A", 0.08)]]})
    got = weight_classes(frame, weight_col="nbrs_barrier_weight")
    assert got["links_w_one"] == 2
    assert got["links_fractional"] == 2
    assert got["links_severed"] == 0
    assert got["median_fractional_w"] == "0.08"


def test_the_derived_profile_changes_only_the_barrier_block(tmp_path):
    """One factor: the barrier rule and its buffer, nothing else. `bbox` is
    kept deliberately, so the diff is attributable to the barrier rule
    alone."""
    base = oracle_profile_path("code-2025", tmp_path)
    path = derived_profile(base, tmp_path / "run",
                           profile_name=PARTIAL_PROFILE,
                           methodology={"barrier": {
                               "rule": "partial_weighted", "combine": "any",
                               "buffer_m": 5}})
    got = yaml.safe_load(path.read_text())
    expected = yaml.safe_load(Path(base).read_text())
    expected["profile"] = PARTIAL_PROFILE
    expected["methodology"]["barrier"] = {"rule": "partial_weighted",
                                          "combine": "any", "buffer_m": 5}
    expected["paths"].pop("neighbors_artifact", None)
    expected["paths"].pop("out_dir", None)
    assert got == expected


def test_the_oraculum_weight_classes_are_the_hand_counted_ones(data_dir,  # noqa: F811
                                                               tmp_path):
    """10 undirected bbox pairs -> 20 directed links; only A-D is fractional,
    in both directions; nothing is severed (spec § 7)."""
    from delhi_psi import io

    base = oracle_profile_path("code-2025", tmp_path)
    run_dir = tmp_path / "partial"
    profile = derived_profile(base, run_dir, profile_name=PARTIAL_PROFILE,
                              methodology={"barrier": {
                                  "rule": "partial_weighted",
                                  "combine": "any", "buffer_m": 5}})
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0
    frame = io.read_neighbors(
        run_dir / f"colonies_neighbors_{PARTIAL_PROFILE}.joblib")
    got = weight_classes(frame, weight_col="nbrs_barrier_weight")
    assert got["links_w_one"] == 18
    assert got["links_fractional"] == 2
    assert got["links_severed"] == 0
    assert got["median_fractional_w"] == "0.08"


def test_the_stamp_records_the_buffer_on_the_derived_run(data_dir,  # noqa: F811
                                                         tmp_path):
    """The artifact CANNOT be reused from --verify-dir here: the barrier
    block is in the stamp, so this block re-runs preprocess. Pinned, so a
    future 'optimisation' that stages the proven artifact fails loudly."""
    from delhi_psi import io

    base = oracle_profile_path("code-2025", tmp_path)
    run_dir = tmp_path / "stamped"
    profile = derived_profile(base, run_dir, profile_name=PARTIAL_PROFILE,
                              methodology={"barrier": {
                                  "rule": "partial_weighted",
                                  "combine": "any", "buffer_m": 5}})
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0
    frame = io.read_neighbors(
        run_dir / f"colonies_neighbors_{PARTIAL_PROFILE}.joblib")
    assert frame.attrs["methodology"]["barrier"] == {
        "rule": "partial_weighted", "combine": "any", "buffer_m": 5.0}


def committed_block(name="partial_barriers"):
    text = DOC.read_text() if DOC.exists() else ""
    if FENCE not in text or f"block: {name}" not in text:
        pytest.skip(f"{DOC} carries no measured `{name}` block yet — the run "
                    "step pastes it")
    return parse_block(text, name=name)


def committed_blocks():
    """Every block the document actually carries, for the prose-number
    guard: it must see all of them, or a number quoted from the second
    block reads as an invention."""
    text = DOC.read_text() if DOC.exists() else ""
    return [parse_block(text, name=name) for name in BLOCKS
            if f"block: {name}" in text]


def test_the_doc_block_has_every_required_key():
    block = committed_block()
    for key in (*WEIGHT_CLASSES, "median_fractional_w",
                "settlements_list_changed", "links_kept_code_2025",
                "links_kept_partial", "preprocess_seconds"):
        assert key in block, key


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, committed_blocks())


@needs_measure_cache
def test_a_fresh_run_reproduces_the_committed_block():
    """The real-data drift check, in the test_layer_pathologies pattern. It
    re-runs a full preprocess on 4,357 polygons, so it takes its work dir
    from DELHI_PSI_MEASURE_CACHE and SKIPS when that is unset."""
    block = committed_block()
    proc = subprocess.run(
        [sys.executable, "scripts/measure_rule_effects.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR), "--work-dir", MEASURE_CACHE,
         "--only", "partial_barriers"],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="partial_barriers") == block


def test_main_requires_a_verify_dir(capsys):
    with pytest.raises(SystemExit) as exc:
        measure_rule_effects.main(["--config", "code-2025"])
    assert exc.value.code == 2


# --- the overlap block (DEL-20) ----------------------------------------
def _overlap_frame():
    """Two overlapping settlements and a disjoint third, in the shape a
    stored neighbours artifact has: ids, lists, geometry and NO amount
    columns — `shared_pair_counts` computes those itself, and a frame that
    already carried them would make `point_counts`' merge add suffixes."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        {"USO_AREA_U": ["P", "Q", "Z"],
         "nbrs_bbox": [["Q"], ["P"], []]},
        geometry=[box(0, 0, 1200, 1000), box(1000, 0, 2000, 1000),
                  box(5000, 0, 6000, 1000)], crs="EPSG:7760")


def test_overlapping_neighbours_finds_only_the_positive_area_pair():
    """The overlap rule cannot move any other settlement's PCEN, so this set
    is the containment bound the run is checked against."""
    assert overlapping_neighbours(_overlap_frame()) == {"P", "Q"}


def test_shared_pair_counts_counts_ordered_entries_per_service():
    """The size of the sparse structure, per service — one physical clinic
    inside both P and Q makes TWO ordered entries."""
    import geopandas as gpd
    from shapely.geometry import Point

    clinics = gpd.GeoDataFrame(
        {"service": ["clinic", "clinic"]},
        geometry=[Point(1100, 500), Point(600, 500)], crs="EPSG:7760")
    got = shared_pair_counts(_overlap_frame(), {"clinic": clinics},
                             point_names=("clinic",))
    assert got == {"shared_pairs_clinic": 2, "shared_pairs_total": 2}


def test_pcen_changes_counts_the_fall_and_reports_a_rise():
    """Lending is only ever REDUCED, so a risen PCEN is a bug, not a
    finding: `settlements_pcen_rose` must be 0 in the run."""
    before = pd.DataFrame({"USO_AREA_U": ["P", "Q", "Z"],
                           "clinic_pcen": [1.0, 2.0, 3.0]})
    after = pd.DataFrame({"USO_AREA_U": ["P", "Q", "Z"],
                          "clinic_pcen": [0.5, 2.0, 3.5]})
    report, changed = pcen_changes(before, after, id_col="USO_AREA_U")
    assert report == {"settlements_pcen_changed": 2,
                      "settlements_pcen_rose": 1}
    assert changed == {"P", "Z"}


def test_the_oraculum_shared_structure_is_empty(data_dir, tmp_path):  # noqa: F811
    """Oraculum has no overlapping polygons and no point inside two
    settlements, so every count is 0 — which is exactly why overlap_outside
    is degenerate there. The fixture proves the plumbing; the number that
    matters is the real layer's, and it comes from the run step."""
    from delhi_psi.config import load_config

    from scripts.measure_rule_effects import service_layers

    profile = oracle_profile_path("code-2025", tmp_path)
    cfg = load_config(profile, data_dir=str(data_dir))
    run_dir = tmp_path / "overlap"
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0
    from delhi_psi import io
    frame = io.read_neighbors(run_dir / cfg.paths.neighbors_artifact)
    got = shared_pair_counts(frame, service_layers(cfg),
                             point_names=tuple(cfg.services.point))
    assert got["shared_pairs_total"] == 0, got
    assert overlapping_neighbours(frame) == set()


@needs_measure_cache
def test_a_fresh_overlap_run_reproduces_the_committed_block():
    """The real-data drift check for the second block. It stages the proven
    artifact and runs `compute` only, so it is far cheaper than the barrier
    block's — but it still needs the layers, hence the cache gate."""
    block = committed_block("overlap_lending")
    proc = subprocess.run(
        [sys.executable, "scripts/measure_rule_effects.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR), "--work-dir", MEASURE_CACHE,
         "--only", "overlap_lending"],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="overlap_lending") == block


# --- DEL-59: --out / --splice ------------------------------------------
def test_out_and_splice_appear_in_help(capsys):
    with pytest.raises(SystemExit) as exc:
        measure_rule_effects.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--out" in help_text
    assert "--splice" in help_text


def test_out_and_splice_together_are_refused_by_argparse():
    with pytest.raises(SystemExit):
        measure_rule_effects.main(
            ["--verify-dir", "x", "--out", "dump.md", "--splice", "doc.md"])


def test_out_refuses_to_overwrite_a_target_that_holds_prose(tmp_path):
    """--out writes blocks only; a target already holding hand-written prose
    must be refused (exit 1) rather than clobbered, and left untouched."""
    target = tmp_path / "prose.md"
    before = "# Rule effects\n\nA hand-written caption.\n"
    target.write_text(before)
    with pytest.raises(SystemExit):
        measure_rule_effects.main(["--verify-dir", "x", "--out", str(target)])
    assert target.read_text() == before


@needs_measure_cache
def test_stdout_carries_blocks_and_nothing_else():
    """DEL-59: the provenance lines move to stderr so --out and --splice
    have a clean stream to work with. `parse_block` always ignored those
    lines, so this is the first test that would notice one coming back.

    `holds_prose` is the whole-stream check, and it is the assertion that
    matters: it is true of ANY non-blank line outside a fenced block, so it
    catches every missed diagnostic rather than one named one. `--only
    overlap_lending` is the cheap block (stages the proven artifact and runs
    `compute` alone), used here because this test only cares about the
    stream, not which block ran.
    """
    proc = subprocess.run(
        [sys.executable, "scripts/measure_rule_effects.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR), "--work-dir", MEASURE_CACHE,
         "--only", "overlap_lending"],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert not holds_prose(proc.stdout)
    assert "layer:" in proc.stderr
    assert "verify-dir:" in proc.stderr
    assert "work-dir:" in proc.stderr


def test_only_plus_splice_refreshes_one_run_and_leaves_the_other(tmp_path):
    """The behaviour DEL-58 proved on the real phase6_sweep.md, here on a
    two-run document: refreshing `partial_barriers` must replace that run
    and leave `overlap_lending` untouched, along with every caption."""
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
