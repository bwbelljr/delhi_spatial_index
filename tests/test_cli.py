"""Both CLI stages on the Oraculum temp dir: csv and shp, exit codes,
--config by name and by path.

The shp case runs IN-PROCESS on purpose: under `-W error` it is the only
thing that exercises io._write_shapefile's warning filter. The old e2e test
ran the scripts in a subprocess, which is why the two shapefile warnings were
invisible (spec § 6).
"""
import warnings

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from delhi_psi import cli
from delhi_psi.config import PROFILES_DIR
from tests.oraculum_fixtures import (
    EPSG, load_barriers, load_services, load_settlements,
)

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
    # Placed inside the delhi_bounds_buffer box (990_000-1_010_000 on both
    # axes) but ~10km from the settlement cluster (999_500-1_003_500,
    # 1_000_000-1_003_000): far enough that it intersects nothing (still
    # "valid yet touches nothing"), close enough to pass the within_bounds
    # check that preprocess()'s layer battery now runs on every barrier
    # layer. The original 2_000_000 offset predates that check and fails it.
    gdf = gpd.GeoDataFrame(
        {"name": ["placeholder"]},
        geometry=[LineString([(0, 0), (1, 1)])], crs=f"EPSG:{EPSG}")
    gdf.geometry = gdf.translate(xoff=992_000, yoff=992_000)
    gdf.to_file(path)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """The spec § 3 manifest, laid out at the code-2025 default paths."""
    root = tmp_path_factory.mktemp("oraculum_data")
    city = load_settlements()

    (root / "uso_update_sep2021").mkdir()
    # Drop `population`: compute merges the population CSV and renames it to
    # `population`; a second column of that name collides. The real dataset's
    # shapefile has no population field either.
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


def run(*args):
    return cli.main(list(args))


def test_preprocess_then_compute_by_profile_name(data_dir, tmp_path):
    out = tmp_path / "by_name"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    assert (out / "colonies_neighbors.joblib").exists()

    assert run("compute", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    for denom in ("pop", "popdensity"):
        base = f"delhi_psi_code-2025_{denom}_2020"
        # formats: [csv, shp, joblib] — the shp write happened IN-PROCESS
        # under -W error, so the warning filter is exercised here
        for suffix in (".csv", ".shp", ".joblib"):
            assert (out / f"{base}{suffix}").exists(), base + suffix
    assert (out / "missing_population.csv").exists()


def test_config_by_path_is_equivalent(data_dir, tmp_path):
    out = tmp_path / "by_path"
    assert run("preprocess", "--config", str(PROFILES_DIR / "code-2025.yaml"),
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    assert (out / "colonies_neighbors.joblib").exists()


def test_shapefile_columns_drop_the_unserializable_ones(data_dir, tmp_path):
    out = tmp_path / "shp"
    run("preprocess", "--config", "code-2025", "--data-dir", str(data_dir),
        "--out-dir", str(out))
    run("compute", "--config", "code-2025", "--data-dir", str(data_dir),
        "--out-dir", str(out))
    shp = gpd.read_file(out / "delhi_psi_code-2025_pop_2020.shp")
    for dropped in ("nbrs_bbox", "nbrs_dist_bbox", "centroid"):
        assert dropped not in shp.columns


def test_csv_output_carries_the_baseline_columns(data_dir, tmp_path):
    out = tmp_path / "csv"
    run("preprocess", "--config", "code-2025", "--data-dir", str(data_dir),
        "--out-dir", str(out))
    run("compute", "--config", "code-2025", "--data-dir", str(data_dir),
        "--out-dir", str(out))
    got = pd.read_csv(out / "delhi_psi_code-2025_pop_2020.csv")
    for column in ("USO_AREA_U", "population", "area_km2", "ndmc_dist_km",
                   "road_length", "unnorm_psi", "norm_psi", "health_idx"):
        assert column in got.columns, column


def test_unknown_profile_exits_2(data_dir, tmp_path):
    assert run("compute", "--config", "no-such-profile",
               "--data-dir", str(data_dir),
               "--out-dir", str(tmp_path / "x")) == 2


def test_missing_input_layer_exits_1(tmp_path):
    empty = tmp_path / "empty_data"
    empty.mkdir()
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(empty),
               "--out-dir", str(tmp_path / "y")) == 1


def test_unknown_stage_exits_2(data_dir, tmp_path):
    with pytest.raises(SystemExit) as exc:
        run("frobnicate", "--config", "code-2025")
    assert exc.value.code == 2


def test_ndmc_center_outside_bounds_exits_1(data_dir, tmp_path):
    # preprocess runs the layer battery on the NDMC point too (spec § 6).
    import shutil
    d = tmp_path / "d"
    shutil.copytree(data_dir, d)
    gpd.GeoDataFrame({"name": ["far"]},
                     geometry=[Point(9_000_000, 9_000_000)],
                     crs=f"EPSG:{EPSG}").to_file(
        d / "ndmc_center7760" / "ndmc_center7760.shp")
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(d), "--out-dir", str(tmp_path / "o")) == 1


def test_service_layer_without_crs_exits_1(data_dir, tmp_path):
    # compute refuses a service layer that has no CRS (spec § 6 CRS check).
    import shutil
    d = tmp_path / "d"
    shutil.copytree(data_dir, d)
    out = tmp_path / "o"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(d), "--out-dir", str(out)) == 0
    bank = d / "Public Services" / "Banking" / "Banking.shp"
    gdf = gpd.read_file(bank)
    # A plain overwrite leaves the existing .prj sidecar in place (pyogrio
    # does not delete stale sidecar files on write), so the stripped CRS
    # never actually reaches disk unless the old files are removed first.
    for sidecar in bank.parent.glob(bank.stem + ".*"):
        sidecar.unlink()
    with warnings.catch_warnings():
        # pyogrio warns on every CRS-less write; that is the point of this
        # fixture setup, not something under test (-W error would otherwise
        # fail the test on its own setup step).
        warnings.filterwarnings(
            "ignore", message="'crs' was not provided", category=UserWarning)
        gdf.set_crs(None, allow_override=True).to_file(bank)
    assert run("compute", "--config", "code-2025",
               "--data-dir", str(d), "--out-dir", str(out)) == 1
