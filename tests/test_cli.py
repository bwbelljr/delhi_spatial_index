"""Both CLI stages on the Oraculum temp dir: csv and shp, exit codes,
--config by name and by path.

The shp case runs IN-PROCESS on purpose: under `-W error` it is the only
thing that exercises io._write_shapefile's warning filter. The old e2e test
ran the scripts in a subprocess, which is why the two shapefile warnings were
invisible (spec § 6).
"""
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from delhi_psi import cli
from tests.oraculum_fixtures import (
    EPSG, load_barriers, load_services, load_settlements, oracle_profile_path,
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


SHIPPED_PROFILES = ("code-2025", "manuscript")


def run(*args):
    """`cli.main`, with a shipped profile NAME in `--config` swapped for the
    DERIVED oracle profile (spec 3B § 2).

    The oracle city's `UC`/`IND` are deliberately absent from the shipped
    Delhi mappings, so the shipped profiles correctly refuse to run here.
    The derived YAML is written into the run's own `--data-dir`. A test that
    WANTS the shipped profile — the unmapped-type guard — calls `cli.main`
    directly.
    """
    args = list(args)
    if "--config" in args and "--data-dir" in args:
        at = args.index("--config") + 1
        if args[at] in SHIPPED_PROFILES:
            directory = Path(args[args.index("--data-dir") + 1])
            args[at] = str(oracle_profile_path(args[at], directory))
    return cli.main(args)


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
    """`--config <path>` still works — exercised with the derived profile,
    which is the only complete profile this city can run (spec 3B § 2)."""
    out = tmp_path / "by_path"
    profile = oracle_profile_path("code-2025", tmp_path)
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(out)]) == 0
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


def test_duplicate_service_row_is_dropped_before_validation(data_dir, tmp_path):
    # The real bank layer has exact-duplicate rows; compute_psi.py used to
    # drop_duplicates() the bank layer before its (print-only) checks.
    # compute() must do the same before validate.require_layer, or this
    # fails with `validation failed: layer 'bank' failed validation:
    # has_duplicate_rows` — and the dedup must not double-count: the
    # resulting bank_count must match the run without the duplicate.
    import shutil

    baseline_out = tmp_path / "baseline"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(baseline_out)) == 0
    assert run("compute", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(baseline_out)) == 0
    baseline = pd.read_csv(
        baseline_out / "delhi_psi_code-2025_pop_2020.csv").set_index("USO_AREA_U")

    dup_dir = tmp_path / "dup_data"
    shutil.copytree(data_dir, dup_dir)
    bank_path = dup_dir / "Public Services" / "Banking" / "Banking.shp"
    bank = gpd.read_file(bank_path)
    dup = gpd.GeoDataFrame(pd.concat([bank, bank.iloc[[0]]], ignore_index=True),
                           crs=bank.crs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Column names longer than 10 characters",
            category=UserWarning)
        warnings.filterwarnings(
            "ignore", message="Normalized/laundered field name",
            category=RuntimeWarning)
        dup.to_file(bank_path)

    out = tmp_path / "dup"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(dup_dir), "--out-dir", str(out)) == 0
    assert run("compute", "--config", "code-2025",
               "--data-dir", str(dup_dir), "--out-dir", str(out)) == 0
    got = pd.read_csv(out / "delhi_psi_code-2025_pop_2020.csv").set_index("USO_AREA_U")

    pd.testing.assert_series_equal(got["bank_count"], baseline["bank_count"])


# --- C1: the neighbours artifact is bound to the methodology that built it ---
GEOM_COLS = ("geometry", "centroid")


def _assert_neighbor_frames_equal(got, expected):
    """Frame equality for the neighbours artifact: plain columns through
    assert_frame_equal, the two geometry columns through shapely equality
    (assert_frame_equal cannot compare geometries)."""
    pd.testing.assert_frame_equal(
        pd.DataFrame(got.drop(columns=list(GEOM_COLS))),
        pd.DataFrame(expected.drop(columns=list(GEOM_COLS))))
    for col in GEOM_COLS:
        assert len(got[col]) == len(expected[col]), col
        for left, right in zip(got[col], expected[col]):
            assert left.equals(right), col


def test_neighbors_artifact_carries_the_methodology_stamp(data_dir, tmp_path):
    """preprocess stamps the artifact and pandas `attrs` survive joblib.

    Without the round trip the stamp exists only in memory and compute's
    guard can never fire.
    """
    from delhi_psi import io

    out = tmp_path / "stamped"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    frame = io.read_neighbors(out / "colonies_neighbors.joblib")
    assert frame.attrs["profile"] == "code-2025"
    assert frame.attrs["methodology"] == {
        "adjacency": {"rule": "bbox"},
        "barrier": {"rule": "global_asymmetric", "combine": "any"},
    }


def test_compute_rejects_an_artifact_built_by_another_methodology(
        data_dir, tmp_path, capsys):
    """preprocess with code-2025 then compute with manuscript: the stored
    neighbour lists were built with `bbox`/`global_asymmetric`, so every
    number compute would produce is a lie about the configured method."""
    import shutil

    out = tmp_path / "mismatch"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    # Hand manuscript the code-2025 artifact under the name it looks for, so
    # the STAMP is what fails, not the file lookup.
    shutil.copy(out / "colonies_neighbors.joblib",
                out / "colonies_neighbors_manuscript.joblib")

    assert run("compute", "--config", "manuscript",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 1
    err = capsys.readouterr().err
    assert "adjacency" in err and "rule" in err
    assert "bbox" in err and "touch" in err


def test_compute_rejects_an_unstamped_artifact(data_dir, tmp_path, capsys):
    """An artifact from before the stamp existed cannot be trusted either."""
    from delhi_psi import io

    out = tmp_path / "unstamped"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    path = out / "colonies_neighbors.joblib"
    frame = io.read_neighbors(path)
    frame.attrs.clear()
    io.write_neighbors(frame, path)

    assert run("compute", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 1
    assert "no methodology stamp" in capsys.readouterr().err


# --- I3: the dedup cache's hit branch ---------------------------------
def test_second_preprocess_reuses_the_dedup_cache(data_dir, tmp_path, caplog):
    import logging as _logging

    from delhi_psi import io

    out = tmp_path / "dedup_hit"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    cached = out / "settlements.dedup.gpkg"
    first_mtime = cached.stat().st_mtime_ns
    first = io.read_neighbors(out / "colonies_neighbors.joblib")

    caplog.clear()
    with caplog.at_level(_logging.INFO, logger="delhi_psi.pipeline"):
        assert run("preprocess", "--config", "code-2025",
                   "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any("reusing dedup cache" in message
               for message in messages), messages
    assert cached.stat().st_mtime_ns == first_mtime
    _assert_neighbor_frames_equal(
        io.read_neighbors(out / "colonies_neighbors.joblib"), first)


def test_touching_the_source_invalidates_the_dedup_cache(data_dir, tmp_path,
                                                         caplog):
    """The cache is keyed on the source's mtime+size, not on existence."""
    import logging as _logging
    import os
    import shutil

    root = tmp_path / "dedup_src"
    shutil.copytree(data_dir, root)
    out = tmp_path / "dedup_miss"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(root), "--out-dir", str(out)) == 0
    cached = out / "settlements.dedup.gpkg"
    before = cached.stat().st_mtime_ns

    shp = root / "uso_update_sep2021" / "uso_update_sep2021.shp"
    stat = shp.stat()
    os.utime(shp, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    caplog.clear()
    with caplog.at_level(_logging.INFO, logger="delhi_psi.pipeline"):
        assert run("preprocess", "--config", "code-2025",
                   "--data-dir", str(root), "--out-dir", str(out)) == 0
    reused = [message for message in
              (record.getMessage() for record in caplog.records)
              if "reusing dedup cache" in message
              and "settlements.dedup.gpkg" in message]
    assert reused == []
    assert cached.stat().st_mtime_ns != before


# --- I5: the manuscript profile, end to end through the CLI -----------
def test_manuscript_profile_runs_end_to_end(data_dir, tmp_path):
    """The path-based stages and the in-memory oracle share one population /
    exclusion prelude, so the CLI's manuscript run reproduces the oracle."""
    from tests.oraculum_fixtures import compute_oracle_frame

    out = tmp_path / "manuscript"
    assert run("preprocess", "--config", "manuscript",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    assert (out / "colonies_neighbors_manuscript.joblib").exists()
    assert run("compute", "--config", "manuscript",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0

    got = pd.read_csv(
        out / "delhi_psi_manuscript_pop_2020.csv").set_index("USO_AREA_U")
    # second_normalization: false -> no norm_psi column
    assert "unnorm_psi" in got.columns
    assert "norm_psi" not in got.columns
    # outputs.denominators is [pop] only
    assert not (out / "delhi_psi_manuscript_popdensity_2020.csv").exists()

    expected = compute_oracle_frame("manuscript", types=(),
                                    stage="post_neighbors", denom="pop")
    assert set(got.index) == set(expected.index)
    for sid in expected.index:
        assert got.loc[sid, "bank_pcen"] == pytest.approx(
            expected.loc[sid, "bank_pcen"], abs=1e-12), sid


# --- minor (c): the math layer's own exceptions map to exit 1 ---------
@pytest.mark.parametrize("exc", [ValueError("bad amount column"),
                                 KeyError("road_length")])
def test_math_layer_errors_exit_1(data_dir, tmp_path, monkeypatch, capsys, exc):
    def boom(cfg):
        raise exc

    monkeypatch.setitem(cli.STAGES, "compute", boom)
    assert run("compute", "--config", "code-2025",
               "--data-dir", str(data_dir),
               "--out-dir", str(tmp_path / "boom")) == 1
    assert capsys.readouterr().err.strip() != ""


# --- 3B: the vocabulary-change equivalence proof (spec 3B § 4) --------
REFERENCE_CSV = (Path(__file__).resolve().parent / "fixtures" / "oraculum"
                 / "expected_values.csv")

# The fixture city's six source types collapsed into five categories, of
# which one — `non-urban` — is what the run then excludes. RV and IND are
# the two settlements today's raw `exclusion.types: [RV, IND]` drops.
ORACLE_5 = {"Planned": "planned", "UC": "unauthorized",
            "RUAC": "regularized", "JJC": "jjc", "RV": "non-urban",
            "IND": "non-urban"}

# exclusion.stage -> the reference scenario with the same dropped set.
REFERENCE_SCENARIO = {"post_neighbors": "excl_contributing",
                      "pre_neighbors": "excl_removed"}

# CLI output column -> the same quantity in compute_oracle_frame's frame.
# The fixture's clinic layer is written to Public Services/Health/Health.shp,
# so the config service is `health` where the oracle frame says `clinic`.
COLLAPSE_TO_ORACLE = {
    "health_count": "clinic_count", "health_pcen": "clinic_pcen",
    "health_idx": "clinic_idx",
    "school_count": "school_count", "school_pcen": "school_pcen",
    "school_idx": "school_idx",
    "bank_count": "bank_count", "bank_pcen": "bank_pcen",
    "bank_idx": "bank_idx",
    "police_count": "police_count", "police_pcen": "police_pcen",
    "police_idx": "police_idx",
    "ration_count": "ration_count", "ration_pcen": "ration_pcen",
    "ration_idx": "ration_idx",
    "transport_count": "transport_count",
    "transport_pcen": "transport_pcen", "transport_idx": "transport_idx",
    "road_length": "road_length", "road_pcen": "road_pcen",
    "road_idx": "road_idx",
    "unnorm_psi": "unnorm_psi", "norm_psi": "norm_psi",
    "population": "population", "area_km2": "area_km2",
}

# CLI output column -> expected_values.csv metric name (the REFERENCE's
# names: psi_eq1 for unnorm_psi, road_length_km for road_length).
COLLAPSE_TO_REFERENCE = {
    "health_count": "clinic_count", "health_pcen": "clinic_pcen",
    "health_idx": "clinic_idx",
    "school_count": "school_count", "school_pcen": "school_pcen",
    "school_idx": "school_idx",
    "bank_count": "bank_count", "bank_pcen": "bank_pcen",
    "bank_idx": "bank_idx",
    "police_count": "police_count", "police_pcen": "police_pcen",
    "police_idx": "police_idx",
    "ration_count": "ration_count", "ration_pcen": "ration_pcen",
    "ration_idx": "ration_idx",
    "transport_count": "transport_count",
    "transport_pcen": "transport_pcen", "transport_idx": "transport_idx",
    "road_length": "road_length_km", "road_pcen": "road_pcen",
    "road_idx": "road_idx",
    "unnorm_psi": "psi_eq1", "norm_psi": "norm_psi",
}


def collapse_profile_path(directory, *, stage):
    """A profile derived from `code-2025` that collapses the fixture's six
    source types into five and excludes the CATEGORY `non-urban`.

    Everything else is code-2025's: reference rule `code`, `swallowed`, the
    second normalization on. Only the vocabulary changes — which is the
    claim under test.
    """
    import yaml

    from delhi_psi.config import PROFILES_DIR

    raw = yaml.safe_load((PROFILES_DIR / "code-2025.yaml").read_text())
    raw["profile"] = "oracle-5"
    raw["categories"] = {"scheme": "oracle-5", "mapping": dict(ORACLE_5)}
    raw["methodology"]["exclusion"]["types"] = ["non-urban"]
    raw["methodology"]["exclusion"]["stage"] = stage
    path = Path(directory) / f"oracle-5-{stage}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


@pytest.mark.parametrize("denom", ["pop", "popdensity"])
@pytest.mark.parametrize("stage", ["post_neighbors", "pre_neighbors"])
def test_five_way_collapse_reproduces_raw_type_exclusion(data_dir, tmp_path,
                                                         stage, denom):
    """Spec 3B § 4, the vocabulary-change equivalence proof.

    A profile that collapses six source types into five and excludes the
    CATEGORY `non-urban` must produce (a) exactly the numbers today's raw
    `exclusion.types: [RV, IND]` produces, and (b) the independent reference
    implementation's own `code` rows for the scenario with the same dropped
    set. Together they are the proof that this layer changed the vocabulary
    and nothing else.

    Tolerance is the CSV round-trip's 1e-9 (the existing e2e's); 1e-12
    applies only to in-memory comparisons.
    """
    from tests.oraculum_fixtures import compute_oracle_frame

    profile = collapse_profile_path(tmp_path, stage=stage)
    out = tmp_path / "collapse"
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    assert cli.main(["compute", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0

    got = pd.read_csv(
        out / f"delhi_psi_oracle-5_{denom}_2020.csv").set_index("USO_AREA_U")
    assert set(got.index) == {"A", "B", "C", "D", "E"}
    assert got["category"].to_dict() == {
        "A": "planned", "B": "unauthorized", "C": "jjc", "D": "planned",
        "E": "regularized"}

    # (a) the same numbers as today's raw-string exclusion
    direct = compute_oracle_frame("code-2025", types=("RV", "IND"),
                                  stage=stage, denom=denom)
    assert set(direct.index) == set(got.index)
    for column, oracle_column in COLLAPSE_TO_ORACLE.items():
        for sid in got.index:
            assert got.loc[sid, column] == pytest.approx(
                direct.loc[sid, oracle_column], abs=1e-9), (column, sid)

    # (b) the independent reference implementation, rule `code`
    expected = pd.read_csv(REFERENCE_CSV)
    expected = expected[
        (expected["rule"] == "code")
        & (expected["scenario"] == REFERENCE_SCENARIO[stage])
        & (expected["denom"] == denom)
    ].pivot(index="settlement", columns="metric", values="value")
    assert set(expected.index) == set(got.index)
    for column, metric in COLLAPSE_TO_REFERENCE.items():
        for sid in got.index:
            assert got.loc[sid, column] == pytest.approx(
                expected.loc[sid, metric], abs=1e-9), (column, sid)


def test_unmapped_settlement_type_exits_1(data_dir, tmp_path, capsys):
    """The shipped `code-2025` mapping is Delhi's `uso-10`; this city
    carries `UC` and `IND`, which are deliberately NOT in it. Running the
    SHIPPED profile straight at this city is therefore the proof that an
    unmapped source type fails the run, naming every offender with its row
    count (spec 3B §§ 2, 5).

    `cli.main`, not `run`: the whole point is the shipped profile.
    """
    out = tmp_path / "unmapped"
    assert cli.main(["preprocess", "--config", "code-2025",
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    assert cli.main(["compute", "--config", "code-2025",
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 1
    err = capsys.readouterr().err
    assert "validation failed" in err
    assert "'IND' (1 row)" in err
    assert "'UC' (1 row)" in err
    assert "categories.mapping" in err


def test_outputs_carry_the_category_column_and_the_scheme_stamp(data_dir,
                                                                tmp_path,
                                                                caplog):
    """`category` on the CSV, the shapefile, the joblib and
    missing_population.csv; the scheme/mapping stamp on the joblib, which is
    the only format that can hold `attrs`. For CSV and shapefile the record
    is the INFO line plus the column itself — the scheme is never a column.
    """
    import logging as _logging

    from delhi_psi import io

    out = tmp_path / "categories"
    assert run("preprocess", "--config", "code-2025",
               "--data-dir", str(data_dir), "--out-dir", str(out)) == 0
    caplog.clear()
    with caplog.at_level(_logging.INFO, logger="delhi_psi.pipeline"):
        assert run("compute", "--config", "code-2025",
                   "--data-dir", str(data_dir), "--out-dir", str(out)) == 0

    base = out / "delhi_psi_code-2025_pop_2020"
    csv = pd.read_csv(base.with_suffix(".csv")).set_index("USO_AREA_U")
    # RV is excluded (code-2025's exclusion.types), so it is not reported.
    assert csv["category"].to_dict() == {
        "A": "Planned", "B": "UC", "C": "JJC", "D": "Planned", "E": "RUAC",
        "IND": "IND"}
    assert "scheme" not in csv.columns, "the scheme is metadata, not a column"
    assert "category" in gpd.read_file(base.with_suffix(".shp")).columns
    assert "category" in pd.read_csv(out / "missing_population.csv").columns

    frame = io.read_neighbors(base.with_suffix(".joblib"))
    assert frame.attrs["categories"]["scheme"] == "oracle-6"
    assert frame.attrs["categories"]["mapping"] == {
        "Planned": "Planned", "UC": "UC", "JJC": "JJC", "RV": "RV",
        "RUAC": "RUAC", "IND": "IND"}

    # The NEIGHBOURS artifact stays category-free: it is built on the full
    # universe and stamped with adjacency/barrier only, so a mapping change
    # must never force an 11-minute re-`preprocess` (spec 3B § 3).
    nbrs = io.read_neighbors(out / "colonies_neighbors.joblib")
    assert "category" not in nbrs.columns
    assert "categories" not in nbrs.attrs

    messages = [record.getMessage() for record in caplog.records]
    assert "categories: scheme=oracle-6 n_categories=6" in messages, messages
