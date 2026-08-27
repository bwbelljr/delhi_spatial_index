"""The committed production fixtures must be exactly what the generator emits.

Same contract as test_expected_values_csv_is_regenerable: without this a red
build could be 'fixed' by hand-editing the fixture, turning the refactor's
correctness proof into a record of whatever the code now does.
"""
from pathlib import Path

import pytest

from scripts.generate_production_fixtures import (
    PRODUCTION_DIR, SERVICES, emit_profile, metric_columns,
)

PROFILES = ["code-2025"]


@pytest.mark.parametrize("profile", PROFILES)
def test_fixture_is_regenerable(profile, tmp_path):
    committed = PRODUCTION_DIR / f"{profile}.csv"
    assert committed.exists(), f"missing committed fixture {committed}"
    regen = emit_profile(profile, tmp_path / f"{profile}.csv")
    assert regen.read_text() == committed.read_text()


@pytest.mark.parametrize("profile", PROFILES)
def test_fixture_has_the_spec_shape(profile):
    text = (PRODUCTION_DIR / f"{profile}.csv").read_text()
    assert "\r" not in text, "fixtures are LF-only"
    lines = text.splitlines()
    assert lines[0] == "profile,scenario,denom,settlement,metric,value"
    rows = [line.split(",") for line in lines[1:]]
    assert all(r[0] == profile for r in rows)
    # sorted by (scenario, denom, settlement, metric)
    keys = [(r[1], r[2], r[3], r[4]) for r in rows]
    assert keys == sorted(keys)


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
