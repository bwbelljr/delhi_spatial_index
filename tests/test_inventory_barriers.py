"""The barrier-layer inventory (DEL-51, spec § 2.3).

Fixture-level tests run everywhere. The doc-drift tests wake up when the run
step pastes the measured blocks into docs/data/barriers.md; until then they
skip with a reason that says so. The REAL-DATA drift test needs one more
thing: DELHI_PSI_MEASURE_CACHE, the shared warm work dir the run step
exports, without which it skips rather than paying ~4.5 min for a cold
settlement dedup in every implementer's suite run.
"""
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from scripts._measure_common import FENCE, parse_block
from scripts.inventory_barriers import (attributes, barrier_flagged,
                                        inventory, layer_facts, main,
                                        metadata_dates)
from tests.cities import ORACULUM
from tests.test_measure_common import (DATA_DIR, MEASURE_CACHE,
                                       assert_prose_numbers_come_from_the_blocks,
                                       needs_measure_cache)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "barriers.md"

LAYER_KEYS = ("features", "geom_types", "crs", "length_km",
              "within_settlement_bbox", "crea_date", "crea_time", "mod_date",
              "has_qpj", "flagged_settlements")
CONFIGURED = ("canal", "railway", "drain")

ESRI_SIDECAR = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<metadata xml:lang=\"en\"><Esri><CreaDate>20200802</CreaDate>\n"
    "<CreaTime>17595100</CreaTime></Esri></metadata>\n")


def committed_blocks():
    """The two blocks the document carries, or a skip while it carries none."""
    if not DOC.exists() or FENCE not in DOC.read_text():
        pytest.skip(f"{DOC} carries no measured block yet — the run step "
                    "pastes it")
    text = DOC.read_text()
    return parse_block(text, name="layers"), parse_block(text,
                                                         name="attributes")


# --- the fixture city --------------------------------------------------
def test_the_oraculum_canal_is_inventoried_correctly():
    """One 450 m LineString along the A/D edge, in the fixture CRS, inside
    the settlement layer's bounding box."""
    city = ORACULUM.load_settlements()
    got = inventory({"canal": ORACULUM.load_barriers()}, city,
                    epsg=ORACULUM.epsg)["layers"]
    assert got["settlements"] == 7
    assert got["canal_features"] == 1
    assert got["canal_geom_types"] == "LineString"
    assert got["canal_crs"] == "EPSG:7760"
    assert got["canal_length_km"] == "0.45"
    assert got["canal_within_settlement_bbox"] == "yes"
    assert got["canal_flagged_settlements"] == 2
    assert got["flagged_any"] == 2
    # no sidecar next to an in-memory fixture layer
    assert got["canal_crea_date"] == "none"


def test_the_oraculum_canal_flags_a_and_d():
    """The same code path production uses — geometry.barrier_flags plus
    neighbors.combine_barrier_flags — so the doc's flagged count is the
    number the pipeline itself would produce."""
    city = ORACULUM.load_settlements()
    flagged = barrier_flagged(city, {"canal": ORACULUM.load_barriers()},
                              combine="any", configured=("canal",))
    assert set(flagged.loc[flagged["canal"], "USO_AREA_U"]) == {"A", "D"}
    assert set(flagged.loc[flagged["barrier"], "USO_AREA_U"]) == {"A", "D"}


def test_layer_facts_reports_the_source_crs_and_the_projected_length():
    line = gpd.GeoDataFrame(
        {"name": ["x"]}, geometry=[LineString([(0, 0), (0, 1000)])],
        crs="EPSG:7760")
    settlements = ORACULUM.load_settlements()
    got = layer_facts(line, line, settlements=settlements)
    assert got["length_km"] == "1"
    assert got["within_settlement_bbox"] == "no"


# --- the ESRI sidecar --------------------------------------------------
def test_metadata_dates_reads_an_esri_sidecar():
    assert metadata_dates(ESRI_SIDECAR) == {"crea_date": "20200802",
                                            "crea_time": "17595100",
                                            "mod_date": "none"}


def test_metadata_dates_reports_none_for_an_absent_tag():
    assert metadata_dates("<metadata><Esri/></metadata>") == {
        "crea_date": "none", "crea_time": "none", "mod_date": "none"}


# --- the attributes block ----------------------------------------------
def test_name_values_are_capped_at_twenty():
    gdf = gpd.GeoDataFrame(
        {"CAN_NM": [f"canal {n:02d}" for n in range(25)]},
        geometry=[LineString([(n, 0), (n, 1)]) for n in range(25)],
        crs="EPSG:7760")
    got = attributes({"canal": gdf})
    assert got["canal_columns"] == "CAN_NM"
    values = got["canal_values_CAN_NM"]
    assert values.startswith("canal 00;canal 01;")
    assert values.endswith("... (+5 more)")
    assert len(values.split(";")) == 21          # 20 names plus the tail


# --- the committed document --------------------------------------------
def test_the_doc_block_has_every_required_key():
    layers, attributes_block = committed_blocks()
    assert layers["settlements"].isdigit()
    assert layers["flagged_any"].isdigit()
    for name in CONFIGURED:
        for key in LAYER_KEYS:
            assert f"{name}_{key}" in layers, f"{name}_{key}"
        assert layers[f"{name}_features"].isdigit()
        float(layers[f"{name}_length_km"])
        assert f"{name}_columns" in attributes_block, name


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, committed_blocks())


@needs_measure_cache
def test_a_fresh_run_reproduces_the_committed_blocks():
    """The real-data drift check. It runs the script over the 4,357-polygon
    layer, so it needs a settlement dedup — ~4.5 min cold. It therefore takes
    its work dir from DELHI_PSI_MEASURE_CACHE (one warm cache per machine,
    shared with the roads drift test) and SKIPS when that is unset, rather
    than gating on `needs_data` alone and charging every per-task suite run
    for a cold dedup on a machine that has the data. The run step exports it.
    A warm cache is safe here: nothing in this script inspects `geom_type`.
    """
    layers, attributes_block = committed_blocks()
    proc = subprocess.run(
        [sys.executable, "scripts/inventory_barriers.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--all-candidates", "--work-dir", MEASURE_CACHE],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="layers") == layers
    assert parse_block(proc.stdout, name="attributes") == attributes_block


def test_main_prints_its_usage_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--all-candidates" in capsys.readouterr().out
