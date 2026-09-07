"""The runner's decisions, without running anything.

Every expensive thing this script does is a subprocess; what is worth testing
is what it decides to run and what it refuses to touch.
"""
import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point as Pt

from delhi_psi import pipeline
from delhi_psi.config import load_config
from scripts import run_sweep
from tests.test_sweep_profiles import SHARED_ARTIFACT


def test_a_missing_artifact_means_both_stages(tmp_path):
    point = run_sweep.plan_point("band-1km", tmp_path)
    assert point.stages == ("preprocess", "compute")
    assert "missing" in point.reason


def test_a_matching_artifact_skips_preprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(run_sweep, "artifact_matches", lambda path, cfg: True)
    (tmp_path / "colonies_neighbors_band-1km.joblib").write_bytes(b"")
    point = run_sweep.plan_point("band-1km", tmp_path)
    assert point.stages == ("compute",)


def test_the_six_decay_points_plan_one_preprocess_between_them(tmp_path,
                                                               monkeypatch):
    """The saving that § 4.2 of the spec exists for: the first decay point
    builds the shared artifact, and the other five must not rebuild it."""
    built = set()
    monkeypatch.setattr(run_sweep, "artifact_matches",
                        lambda path, cfg: path.name in built)

    stages = []
    for profile in run_sweep.GROUPS["decay"]:
        point = run_sweep.plan_point(profile, tmp_path)
        stages.append(point.stages)
        built.add(point.artifact.name)

    assert stages[0] == ("preprocess", "compute")
    assert all(s == ("compute",) for s in stages[1:])
    assert {run_sweep.plan_point(p, tmp_path).artifact.name
            for p in run_sweep.GROUPS["decay"]} == {SHARED_ARTIFACT}


def test_a_stamp_mismatch_forces_a_rebuild(tmp_path):
    """An artifact built at another radius must never be silently reused: the
    numbers would describe a neighbourhood nobody configured."""
    frame = gpd.GeoDataFrame({"USO_AREA_U": ["A"]},
                             geometry=[Pt(0, 0)], crs="EPSG:7760")
    frame.attrs["methodology"] = {
        "adjacency": {"rule": "within_distance", "max_distance_km": 5.0},
        "barrier": {"rule": "global_asymmetric", "combine": "any",
                    "buffer_m": None}}
    assert not run_sweep.artifact_matches(frame, load_config("band-1km"))
    assert run_sweep.artifact_matches(frame, load_config("band-5km"))


def test_the_work_dir_may_not_be_the_data_dir(tmp_path):
    with pytest.raises(SystemExit) as exc:
        run_sweep.resolve_work_dir(str(tmp_path / "inside"),
                                   data_dir=str(tmp_path))
    assert "bisynced" in str(exc.value)


def test_degree_report_counts_links_and_isolates():
    frame = gpd.GeoDataFrame(
        {"USO_AREA_U": ["A", "B", "C"],
         "nbrs_bbox": [["B", "C"], ["A"], []]},
        geometry=[Pt(0, 0), Pt(1, 1), Pt(2, 2)], crs="EPSG:7760")
    got = run_sweep.degree_report(frame, "USO_AREA_U", nbr_col="nbrs_bbox")
    assert got["n_links"] == 3
    assert got["n_isolates"] == 1
    assert got["deg_max"] == 2
    assert got["deg_mean"] == pytest.approx(1.0)


def test_a_failed_point_is_recorded_and_the_run_continues(tmp_path, monkeypatch):
    """One expensive point falling over must not cost the other ten."""
    calls = []

    def fake_stage(profile, stage, **kwargs):
        calls.append((profile, stage))
        if profile == "decay-power2":
            raise run_sweep.StageFailed(stage, 1, "boom")
        return {"seconds": 0.0}

    monkeypatch.setattr(run_sweep, "run_stage", fake_stage)
    monkeypatch.setattr(run_sweep, "artifact_matches", lambda path, cfg: True)
    run_sweep.run_group("decay", work_dir=tmp_path, data_dir=tmp_path / "data",
                        run_date="2026-09-06", commit="deadbee")

    failed = json.loads(run_sweep.manifest_path(tmp_path, "decay-power2")
                        .read_text())
    assert failed["status"] == "FAILED"
    assert failed["failed_stage"] == "compute"
    assert "boom" in failed["stderr_tail"]
    later = json.loads(run_sweep.manifest_path(tmp_path, "decay-exp2km")
                       .read_text())
    assert later["status"] == "OK"


def test_dry_run_writes_nothing(tmp_path, capsys):
    run_sweep.main(["--group", "decay", "--work-dir", str(tmp_path),
                    "--data-dir", str(tmp_path / "data"), "--dry-run"])
    # Fix round item 6: `*.json` alone lets a leaked `.joblib` or
    # `.dedup.stamp` slip through unnoticed — a dry run must leave the work
    # directory entirely empty, not merely free of manifests.
    assert not list(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert "decay-none" in out and "preprocess" in out


# --- additions beyond the literal brief, covering its explicit resolutions
# and the "WHAT DONE REQUIRES" schema-pinning requirement ------------------

def test_a_dry_run_inside_the_data_dir_still_refuses(tmp_path):
    """Resolution #1: --dry-run must still refuse a work dir inside the data
    dir, with the SAME containment check as a real run — and it must create
    neither directory while refusing."""
    with pytest.raises(SystemExit) as exc:
        run_sweep.main(["--group", "decay",
                        "--work-dir", str(tmp_path / "data" / "inside"),
                        "--data-dir", str(tmp_path / "data"), "--dry-run"])
    assert "bisynced" in str(exc.value)
    assert not (tmp_path / "data").exists()


OK_MANIFEST_KEYS = {
    "profile", "status", "stages_run", "skip_reason", "stamp",
    "preprocess_s", "dedup_cache_warm", "compute_s", "n_links", "deg_mean",
    "deg_p50", "deg_max", "n_isolates", "degree_from", "n_settlements",
    "n_barrier_flagged", "n_reported", "n_missing_population",
    "outputs", "commit", "run_date",
}
FAILED_MANIFEST_KEYS = OK_MANIFEST_KEYS | {
    "failed_stage", "returncode", "stderr_tail"}


def test_the_manifest_carries_every_documented_key(tmp_path, monkeypatch):
    """Item 9: pin the exact key set of an OK and a FAILED manifest against a
    literal list, so a key that quietly stops being written fails a test
    instead of showing up as a blank column months later."""
    def fake_stage(profile, stage, **kwargs):
        if profile == "decay-power2" and stage == "compute":
            raise run_sweep.StageFailed(stage, 1, "boom")
        return {"seconds": 0.0}

    monkeypatch.setattr(run_sweep, "run_stage", fake_stage)
    monkeypatch.setattr(run_sweep, "artifact_matches", lambda path, cfg: True)
    run_sweep.run_group("decay", work_dir=tmp_path, data_dir=tmp_path / "data",
                        run_date="2026-09-06", commit="deadbee")

    ok = json.loads(run_sweep.manifest_path(tmp_path, "decay-none").read_text())
    failed = json.loads(run_sweep.manifest_path(tmp_path, "decay-power2")
                        .read_text())
    assert set(ok.keys()) == OK_MANIFEST_KEYS
    assert set(failed.keys()) == FAILED_MANIFEST_KEYS


def test_a_borrowed_point_with_no_source_manifest_gets_null_degree(
        tmp_path, monkeypatch):
    """Item 7: a point that skips preprocess but has no manifest recording
    who built its artifact (e.g. a --only re-run against a pre-existing
    artifact) must report null degree fields, not zero — zero would misread
    as "no links" and trip the isolates flag."""
    monkeypatch.setattr(run_sweep, "run_stage", lambda *a, **k: {"seconds": 0.0})
    monkeypatch.setattr(run_sweep, "artifact_matches", lambda path, cfg: True)
    run_sweep.run_group("decay", work_dir=tmp_path, data_dir=tmp_path / "data",
                        run_date="2026-09-06", commit="deadbee",
                        only=("decay-power05",))
    manifest = json.loads(run_sweep.manifest_path(tmp_path, "decay-power05")
                         .read_text())
    assert manifest["n_links"] is None
    assert manifest["degree_from"] == "artifact predates this run"


# --- fix round 1 (task review) --------------------------------------------

def test_a_corrupt_artifact_forces_a_rebuild_instead_of_raising(tmp_path):
    """Review item 1: `io.write_neighbors` isn't atomic — a preprocess
    killed mid-write (an OOM or a segfault, the case run_stage's subprocess
    isolation exists for) leaves a truncated .joblib. The OLD code called
    `io.read_neighbors` outside any try/except in `artifact_matches`, so an
    unpickling/EOF error on a corrupt file propagated out of `plan_point`
    and would abort the whole group — the exact failure this module exists
    to isolate against. This must plan a rebuild instead of raising."""
    cfg = load_config("band-1km")
    artifact = tmp_path / cfg.paths.neighbors_artifact
    artifact.write_bytes(b"not a joblib file -- truncated mid-write")
    point = run_sweep.plan_point("band-1km", tmp_path)
    assert point.stages == ("preprocess", "compute")
    # artifact_matches itself must not raise either -- it is a documented,
    # independently-used interface (Task 6/7), not just a plan_point detail.
    assert run_sweep.artifact_matches(artifact, cfg) is False


def _fake_stage_writes_artifact(profile, stage, *, data_dir, work_dir):
    """A `run_stage` stand-in for tests that force a real `preprocess`
    (`artifact_matches` mocked False): `run_group` reads the artifact right
    back after a preprocess it thinks succeeded, so the fake has to leave
    something readable there, exactly as the real subprocess would."""
    if stage == "preprocess":
        cfg = load_config(profile)
        frame = gpd.GeoDataFrame(
            {"USO_AREA_U": ["A", "B"], "nbrs_bbox": [["B"], []]},
            geometry=[Pt(0, 0), Pt(1, 1)], crs="EPSG:7760")
        run_sweep.io.write_neighbors(
            frame, Path(work_dir) / cfg.paths.neighbors_artifact)
        # A real preprocess WRITES the dedup stamps as a side effect
        # (`pipeline._dedup_cached`). The fake must too, or the ordering
        # test below cannot fail: with no stamp ever appearing, moving the
        # `dedup_cache_warm` read to AFTER the stage reads False either
        # way, and the test passes against the very regression it exists
        # to catch. Proven by the fix-round re-review, which moved the read
        # and watched all 17 tests stay green.
        (Path(work_dir) / "settlements.dedup.stamp").write_text("mtime:size")
    return {"seconds": 1.0}


def test_dedup_cache_state_is_recorded_before_preprocess_runs(tmp_path,
                                                              monkeypatch):
    """Review item 2: `decay-none` is ordered first because its preprocess
    absorbs the one-off, work-dir-wide settlement dedup cost. If it fails,
    the next point pays that cold-cache cost instead and its `preprocess_s`
    would be silently inflated. `dedup_cache_warm` makes that self-
    describing: recorded from the `*.dedup.stamp` files BEFORE this point's
    own preprocess runs (so it never reports its own point as having warmed
    the cache it just cold-started)."""
    monkeypatch.setattr(run_sweep, "run_stage", _fake_stage_writes_artifact)
    monkeypatch.setattr(run_sweep, "artifact_matches", lambda path, cfg: False)

    run_sweep.run_group("decay", work_dir=tmp_path, data_dir=tmp_path / "data",
                        run_date="2026-09-06", commit="deadbee",
                        only=("decay-none",))
    cold = json.loads(run_sweep.manifest_path(tmp_path, "decay-none")
                      .read_text())
    assert cold["dedup_cache_warm"] is False

    # No hand-written stamp here: the FIRST point's own preprocess left one,
    # which is exactly why the second point is warm. Move the read to after
    # the stage and the assertion above flips to True, so the ordering is
    # what this test actually pins.
    assert (tmp_path / "settlements.dedup.stamp").exists()
    run_sweep.run_group("decay", work_dir=tmp_path, data_dir=tmp_path / "data",
                        run_date="2026-09-06", commit="deadbee",
                        only=("decay-power05",))
    warm = json.loads(run_sweep.manifest_path(tmp_path, "decay-power05")
                      .read_text())
    assert warm["dedup_cache_warm"] is True


def test_dedup_cache_warm_is_null_when_preprocess_is_skipped(tmp_path,
                                                              monkeypatch):
    """A `compute`-only point never touches the dedup cache, so it must not
    claim an opinion about it -- `null`, not a guessed True/False."""
    monkeypatch.setattr(run_sweep, "run_stage",
                        lambda *a, **k: {"seconds": 1.0})
    monkeypatch.setattr(run_sweep, "artifact_matches", lambda path, cfg: True)
    run_sweep.run_group("decay", work_dir=tmp_path, data_dir=tmp_path / "data",
                        run_date="2026-09-06", commit="deadbee",
                        only=("decay-power05",))
    manifest = json.loads(run_sweep.manifest_path(tmp_path, "decay-power05")
                         .read_text())
    assert manifest["dedup_cache_warm"] is None


def test_extract_int_reads_the_real_dataclass_reprs():
    """Review item 3: every existing test replaces `run_stage` with a fake
    returning no `"stdout"`, so `_extract_int` never actually runs against
    anything repr-shaped -- a field rename in `delhi_psi/pipeline.py`, a
    switch from `print` to `log.info`, or a custom `__repr__` would make all
    four manifest counts silently `None` with no test failing. Built from
    the REAL dataclasses so a rename there is exactly what breaks this."""
    preprocess_repr = repr(pipeline.PreprocessResult(
        neighbors_path=Path("x"), n_settlements=4357, n_barrier_flagged=12,
        reports=()))
    compute_repr = repr(pipeline.ComputeResult(
        outputs=(), missing_population_path=Path("y"),
        n_missing_population=3, n_reported=4350))

    assert run_sweep._extract_int(preprocess_repr, "n_settlements") == 4357
    assert run_sweep._extract_int(preprocess_repr, "n_barrier_flagged") == 12
    assert run_sweep._extract_int(compute_repr, "n_reported") == 4350
    assert run_sweep._extract_int(compute_repr, "n_missing_population") == 3


def test_the_dry_run_containment_check_uses_resolve_work_dir_create_false(
        tmp_path):
    """Review item 4: `_refuse_if_inside` is gone -- the dry-run path must
    route through the SAME `resolve_work_dir` containment check a real run
    uses (via its new `create=False` keyword), so the two paths cannot
    silently diverge. Re-asserts the literal brief's two dry-run guarantees
    against that single implementation."""
    with pytest.raises(SystemExit) as exc:
        run_sweep.main(["--group", "decay",
                        "--work-dir", str(tmp_path / "data" / "inside"),
                        "--data-dir", str(tmp_path / "data"), "--dry-run"])
    assert "bisynced" in str(exc.value)
    assert not (tmp_path / "data").exists()
    assert not hasattr(run_sweep, "_refuse_if_inside")


def test_plan_point_reads_a_matching_artifact_at_most_once(tmp_path,
                                                            monkeypatch):
    """Review item 5: on the stamp-mismatch path, `plan_point` used to call
    `io.read_neighbors` twice -- once inside `artifact_matches`, again
    inside `_stamp_mismatch_reason` -- to deserialize the SAME file just to
    describe why it didn't match. On `band-10km` (a ~4.37M-link frame)
    that is a second multi-minute load spent building a string. This
    counts real `io.read_neighbors` calls for one mismatching artifact."""
    from delhi_psi import io as delhi_io

    calls = []
    real_read = delhi_io.read_neighbors

    def counting_read(path):
        calls.append(path)
        return real_read(path)

    monkeypatch.setattr(run_sweep.io, "read_neighbors", counting_read)

    cfg = load_config("band-1km")
    artifact = tmp_path / cfg.paths.neighbors_artifact
    frame = gpd.GeoDataFrame({"USO_AREA_U": ["A"]}, geometry=[Pt(0, 0)],
                             crs="EPSG:7760")
    frame.attrs["methodology"] = {
        "adjacency": {"rule": "within_distance", "max_distance_km": 5.0},
        "barrier": {"rule": "global_asymmetric", "combine": "any",
                    "buffer_m": None}}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    delhi_io.write_neighbors(frame, artifact)

    calls.clear()
    point = run_sweep.plan_point("band-1km", tmp_path)
    assert point.stages == ("preprocess", "compute")
    assert "stamp mismatch" in point.reason
    assert len(calls) == 1


# --- fix round (final whole-branch review) --------------------------------

def test_run_stage_passes_out_dir_so_output_never_lands_in_the_data_dir(
        tmp_path, monkeypatch):
    """The one line protecting the bisynced `~/delhi_data`: every profile's
    `out_dir` config field DEFAULTS TO `data_dir` (`delhi_psi/io.py`), so
    without `run_stage` passing `--out-dir work_dir` explicitly, every sweep
    point would write its output straight into the hourly-bisynced data
    directory. Deleting that argv pair fails no OTHER test in this suite:
    every one of them monkeypatches `run_stage` itself away, never
    `subprocess.run`, so none of them ever inspects the argv `run_stage`
    actually builds. This test monkeypatches `subprocess.run` instead — the
    one layer below `run_stage` — so it is the one test that can catch this
    argv pair silently going missing.
    """
    calls = []

    class FakeCompleted:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return FakeCompleted()

    monkeypatch.setattr(run_sweep.subprocess, "run", fake_run)
    work_dir = tmp_path / "work"
    data_dir = tmp_path / "data"

    run_sweep.run_stage("decay-none", "preprocess", data_dir=data_dir,
                        work_dir=work_dir)

    assert len(calls) == 1
    argv = calls[0]
    assert "--out-dir" in argv
    i = argv.index("--out-dir")
    assert argv[i + 1] == str(work_dir)
