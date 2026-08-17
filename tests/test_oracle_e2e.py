"""Real-CLI end-to-end: temp data dir -> preprocess -> compute_psi -> compare.

Manifest per spec §3: colonies shapefile, three Barrier_Clip layers (drain/
railway empty-but-valid), ndmc_center, delhi_bounds_buffer, seven Public
Services layers (four singletons so no service is degenerate), population
CSV. The CLI path hardcodes RV-only exclusion -> compare against
rule=code / scenario=excl_rv_only.
"""

import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from tests.oraculum_fixtures import (
    EPSG, load_settlements, load_barriers, load_services,
)

CSV = Path(__file__).resolve().parent / "fixtures" / "oraculum" / "expected_values.csv"
REPO = Path(__file__).resolve().parent.parent

SERVICE_LAYOUT = {
    "clinic": ("Health", "Health.shp"),
    "school": ("School", "schools7760.shp"),
    "bank": ("Banking", "Banking.shp"),
    "police": ("Police", "Police Station.shp"),
    "ration": ("Ration", "Ration.shp"),
    "transport": ("Transport", "Transport.shp"),
    "road": ("Major Road", "Road.shp"),
}


def _empty_line_layer(path):
    gdf = gpd.GeoDataFrame(
        {"name": ["placeholder"]},
        geometry=[LineString([(0, 0), (1, 1)])], crs=f"EPSG:{EPSG}")
    # a real but far-away line so the layer is valid yet touches nothing
    gdf.geometry = gdf.translate(xoff=2_000_000, yoff=2_000_000)
    gdf.to_file(path)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("oraculum_data")
    city = load_settlements()

    (root / "uso_update_sep2021").mkdir()
    # Drop `population`: compute_psi merges the population CSV and renames
    # it to `population` — a second column of the same name crashes it
    # (verified in plan review; the real dataset's shapefile has no
    # population field either).
    city.drop(columns=["population"]).to_file(
        root / "uso_update_sep2021" / "uso_update_sep2021.shp")

    barrier_dir = root / "Barrier_Clip"
    (barrier_dir / "Canal").mkdir(parents=True)
    load_barriers().to_file(barrier_dir / "Canal" / "Canal.shp")
    for sub, fname in [("Drain", "Major_Drain.shp"),
                       ("Railway", "Railway_Line.shp")]:
        (barrier_dir / sub).mkdir()
        _empty_line_layer(barrier_dir / sub / fname)

    (root / "ndmc_center7760").mkdir()
    gpd.GeoDataFrame({"name": ["ndmc"]},
                     geometry=[Point(1_001_500, 1_001_500)],
                     crs=f"EPSG:{EPSG}").to_file(
        root / "ndmc_center7760" / "ndmc_center7760.shp")

    (root / "delhi_bounds_buffer").mkdir()
    gpd.GeoDataFrame({"name": ["bounds"]},
                     geometry=[box(1_000_000 - 10_000, 1_000_000 - 10_000,
                                   1_000_000 + 10_000, 1_000_000 + 10_000)],
                     crs=f"EPSG:{EPSG}").to_file(
        root / "delhi_bounds_buffer" / "delhi_bounds_buffer.shp")

    services = load_services()
    for svc, (folder, fname) in SERVICE_LAYOUT.items():
        d = root / "Public Services" / folder
        d.mkdir(parents=True)
        services[svc].to_file(d / fname)

    pop = load_settlements()[["USO_AREA_U", "population"]].rename(
        columns={"USO_AREA_U": "uso_area_u"})
    pop.to_csv(root / "pop_colony_wp_2020_jjc_adjusted.csv", index=False)
    return root


def _run(script, *args):
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc


def test_full_cli_chain_matches_excl_rv_only(data_dir, tmp_path):
    out_dir = tmp_path / "out"
    _run("preprocess.py", "--data-dir", str(data_dir),
         "--out-dir", str(out_dir))
    nbrs_file = out_dir / "colonies_bbox_nbrs_aug2026.joblib"
    assert nbrs_file.exists()
    _run("compute_psi.py", "--data-dir", str(data_dir),
         "--neighbors-file", str(nbrs_file), "--out-dir", str(out_dir))

    got = pd.read_csv(out_dir / "psi_2020_results"
                      / "delhi_psi_bbox_popsize2020_norv_aug2026.csv")
    got = got.set_index("USO_AREA_U")

    exp = pd.read_csv(CSV)
    exp = exp[(exp["rule"] == "code") & (exp["scenario"] == "excl_rv_only")
              & (exp["denom"] == "pop")] \
        .pivot(index="settlement", columns="metric", values="value")

    assert set(got.index) == set(exp.index) == {"A", "B", "C", "D", "E", "IND"}
    # real pipeline service naming: clinic->health, road count renamed length
    mapping = {
        "health_pcen": "clinic_pcen", "health_idx": "clinic_idx",
        "school_pcen": "school_pcen", "school_idx": "school_idx",
        "bank_pcen": "bank_pcen", "police_pcen": "police_pcen",
        "ration_pcen": "ration_pcen", "transport_pcen": "transport_pcen",
        "road_pcen": "road_pcen", "road_idx": "road_idx",
        "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
    }
    for got_col, metric in mapping.items():
        for sid in exp.index:
            assert got.loc[sid, got_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-9), (got_col, sid)
