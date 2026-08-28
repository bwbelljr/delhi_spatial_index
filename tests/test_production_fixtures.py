"""The committed production fixtures must be exactly what the generator emits.

Same contract as test_expected_values_csv_is_regenerable: without this a red
build could be 'fixed' by hand-editing the fixture, turning the refactor's
correctness proof into a record of whatever the code now does. From cycle 3C
this runs for BOTH cities (spec § 4.2).
"""
from pathlib import Path

import pytest

from scripts.generate_production_fixtures import (
    REPO, SERVICES, emit_profile, metric_columns, production_dir,
)
from tests.cities import CITIES, ORACULUM

PROFILES = ["code-2025", "manuscript"]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_fixture_is_regenerable(city, profile, tmp_path):
    committed = production_dir(city) / f"{profile}.csv"
    assert committed.exists(), f"missing committed fixture {committed}"
    regen = emit_profile(profile, tmp_path / f"{profile}.csv", city)
    # Read as bytes: .read_text() performs universal-newline translation,
    # which would silently hide a line-ending regression (e.g. CRLF).
    assert regen.read_bytes() == committed.read_bytes()


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_fixture_has_the_spec_shape(city, profile):
    path = production_dir(city) / f"{profile}.csv"
    data = path.read_bytes()
    assert b"\r" not in data, "fixtures are LF-only"
    text = data.decode()
    lines = text.splitlines()
    assert lines[0] == "profile,scenario,denom,settlement,metric,value"
    rows = [line.split(",") for line in lines[1:]]
    assert all(r[0] == profile for r in rows)
    # sorted by (scenario, denom, settlement, metric)
    keys = [(r[1], r[2], r[3], r[4]) for r in rows]
    assert keys == sorted(keys)
    assert {r[1] for r in rows} == {s.name for s in city.scenarios}


def test_production_dir_is_per_city():
    assert production_dir() == production_dir(ORACULUM)
    for city in CITIES:
        assert production_dir(city) == city.fixtures / "production"
    assert production_dir(ORACULUM) == (
        REPO / "tests" / "fixtures" / "oraculum" / "production")


def test_metric_set_is_explicit():
    cols = metric_columns(second_normalization=True)
    assert cols == [
        "clinic_count", "school_count", "bank_count", "police_count",
        "ration_count", "transport_count", "road_length",
        "clinic_pcen", "clinic_idx", "school_pcen", "school_idx",
        "bank_pcen", "bank_idx", "police_pcen", "police_idx",
        "ration_pcen", "ration_idx", "transport_pcen", "transport_idx",
        "road_pcen", "road_idx",
        "unnorm_psi", "norm_psi", "population", "area_km2",
    ]
    assert "norm_psi" not in metric_columns(second_normalization=False)
    # geometry / centroid / neighbor-list columns are never serialized
    for banned in ("geometry", "centroid", "nbrs_bbox", "nbrs_dist_bbox"):
        assert banned not in cols
    assert set(SERVICES) == {
        "clinic", "school", "bank", "police", "ration", "transport", "road"}


def test_no_sys_path_hacks_and_no_monolith():
    """The package is installed; nothing may reach for the repo root."""
    import subprocess

    repo = REPO
    assert not (repo / "spatial_index_utils.py").exists()
    assert not (repo / "conftest.py").exists()
    for script in ("preprocess.py", "compute_psi.py", "common.py"):
        assert not (repo / "scripts" / script).exists(), script

    # Exclude this file's own pathspec: it must quote both search terms as
    # string literals to run the check, which would otherwise self-match.
    hits = subprocess.run(
        ["git", "grep", "-n", "sys.path.insert", "--",
         "*.py", ":!archive/", ":!tests/test_production_fixtures.py"],
        cwd=repo, capture_output=True, text=True)
    assert hits.stdout == "", f"sys.path.insert still present:\n{hits.stdout}"

    # A live reference is an import statement, not a historical docstring
    # mention of where the code was copied from (self-review allows those,
    # same as README/CHANGELOG/WORKPLAN prose).
    imports = subprocess.run(
        ["git", "grep", "-n", "-E",
         r"^\s*(import|from) spatial_index_utils\b", "--",
         "*.py", ":!archive/"],
        cwd=repo, capture_output=True, text=True)
    assert imports.stdout == "", \
        f"spatial_index_utils still referenced:\n{imports.stdout}"
