"""The committed production fixtures must be exactly what the generator emits.

Same contract as test_expected_values_csv_is_regenerable: without this a red
build could be 'fixed' by hand-editing the fixture, turning the refactor's
correctness proof into a record of whatever the code now does. From cycle 3C
this runs for BOTH cities (spec § 4.2).
"""
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_production_fixtures import (
    REPO, SERVICES, emit_profile, metric_columns, production_dir,
)
from tests.cities import CITIES, ORACULUM

PROFILES = ["code-2025", "manuscript",
            "adj-touch", "band-0km", "band-1km", "band-5km", "band-10km",
            "decay-none", "decay-power05", "decay-power2", "decay-exp2km",
            "decay-exp5km", "decay-boundary",
            "services-no-ration"]


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


def test_metric_columns_can_take_an_explicit_service_subset():
    """DEL-40 Task 1/3: a profile that subsets services must get a metric
    set with NO columns at all for the dropped service — not a zeroed
    column — so the generator emits a fixture that proves it."""
    cols = metric_columns(second_normalization=True,
                          point_services=("clinic", "school"),
                          line_services=())
    assert cols == ["clinic_count", "school_count",
                    "clinic_pcen", "clinic_idx",
                    "school_pcen", "school_idx",
                    "unnorm_psi", "norm_psi", "population", "area_km2"]
    for banned in ("ration_count", "ration_pcen", "ration_idx",
                  "road_length", "road_pcen", "road_idx",
                  "bank_count", "police_count", "transport_count"):
        assert banned not in cols


# --- DEL-40 § 4: what the services-no-ration fixture must prove ----------
def _pivot(city, profile):
    df = pd.read_csv(production_dir(city) / f"{profile}.csv")
    return df.pivot(index=["scenario", "denom", "settlement"],
                    columns="metric", values="value")


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_services_no_ration_has_no_ration_metric_rows_at_all(city):
    """Absent, not zeroed (spec § 4)."""
    df = pd.read_csv(production_dir(city) / "services-no-ration.csv")
    assert not df["metric"].str.startswith("ration").any()


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_services_no_ration_moves_psi_eq1_for_every_settlement(city):
    """Eq. 1 averages over the services present, so dropping one of seven
    moves psi_eq1 (`unnorm_psi`) for every reported settlement (spec § 4),
    unconditionally: no rescaling sits between the per-service indices and
    this value."""
    base = _pivot(city, "code-2025")
    subset = _pivot(city, "services-no-ration")
    assert set(base.index) == set(subset.index)
    assert len(base.index) > 0
    for idx in base.index:
        assert base.loc[idx, "unnorm_psi"] != subset.loc[idx, "unnorm_psi"], idx


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_services_no_ration_moves_norm_psi_somewhere(city):
    """norm_psi is Eq. 2's min-max RESCALING of psi_eq1, and min-max
    normalization is exactly invariant under a uniform positive-affine
    transform. In these fixtures `ration_idx` is 0 for every settlement but
    one (D in Oraculum's baseline scenario), so dropping ration rescales
    every OTHER settlement's psi_eq1 by the same constant factor — and
    norm_psi legitimately ties for those, which is why this checks that
    norm_psi differs SOMEWHERE, not everywhere: the bug this guards against
    is a filter that never reaches the second normalization at all."""
    base = _pivot(city, "code-2025")
    subset = _pivot(city, "services-no-ration")
    assert "norm_psi" in base.columns and "norm_psi" in subset.columns
    assert (base["norm_psi"] != subset["norm_psi"]).any()


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_services_no_ration_leaves_raw_counts_unchanged(city):
    """Dropping a service changes what is averaged, not what is counted —
    the condition that catches a filter applied in the wrong place (spec
    § 4): a wrongly-placed filter would move counts too."""
    base = _pivot(city, "code-2025")
    subset = _pivot(city, "services-no-ration")
    kept_columns = [c for c in subset.columns
                    if c.endswith("_count") or c == "road_length"]
    assert kept_columns, "expected at least one surviving count column"
    for idx in base.index:
        for col in kept_columns:
            assert base.loc[idx, col] == subset.loc[idx, col], (idx, col)
