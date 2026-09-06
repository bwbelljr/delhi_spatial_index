"""The runner's decisions, without running anything.

Every expensive thing this script does is a subprocess; what is worth testing
is what it decides to run and what it refuses to touch.
"""
import json

import geopandas as gpd
import pytest
from shapely.geometry import Point as Pt

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
    assert not list(tmp_path.rglob("*.json"))
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
    "preprocess_s", "compute_s", "n_links", "deg_mean", "deg_p50",
    "deg_max", "n_isolates", "degree_from", "n_settlements",
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
