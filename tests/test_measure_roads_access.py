"""JJC road access and the roads one-factor effect (DEL-49, spec § 2.1).

The counting is proven on four squares in a row; the one-factor machinery is
proven END TO END on the Oraculum city, where the reference implementation
already says what `roads: eq4_own_only` must produce. The real run is the run
step's, and the real-data drift test needs DELHI_PSI_MEASURE_CACHE — the
shared warm work dir the run step exports — or it skips, rather than paying
~4.5 min for a cold settlement dedup in every implementer's suite run.
"""
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import yaml
from shapely.geometry import LineString, box

from delhi_psi import cli, pipeline
from delhi_psi.config import PROFILES_DIR
from scripts._measure_common import FENCE, parse_block
from scripts.measure_roads_access import (DENOMINATORS, OWN_ONLY_PROFILE,
                                          REPORTED_TYPES,
                                          assert_road_inside_matches_output,
                                          derived_profile, main,
                                          measure_access, measure_effect,
                                          road_inside_ids, stage_artifacts)
from tests.oraculum_fixtures import oracle_profile_path
from tests.test_cli import data_dir  # noqa: F401 — the Oraculum data dir
from tests.test_measure_common import (DATA_DIR, MEASURE_CACHE,
                                       assert_prose_numbers_come_from_the_blocks,
                                       needs_measure_cache)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "data" / "roads_access.md"
VERIFY_DIR = DATA_DIR / "phase3_verify"
REFERENCE_CSV = (REPO / "tests" / "fixtures" / "oraculum"
                 / "expected_values.csv")

# Confirmed from tests/fixtures/oraculum/expected_values.csv on 5 Sep 2026
# (rule=ideal — whose roads knob IS eq4 — scenario=baseline, denom=pop).
# NOTE: 0.0075 and 0.0025 are `road_pcen` values; the memo calls them "the
# ideal roads column". The min-maxed `road_idx` for the same rows is 1.0 and
# 1/3, because under eq4 only A and E own any road at all.
IDEAL_ROAD_PCEN = {"A": 0.0075, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0025,
                   "IND": 0.0}
IDEAL_ROAD_IDX = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 1 / 3,
                  "IND": 0.0}


def four_squares():
    """S1..S4 in a row, each 1 km wide, consecutive squares sharing an edge.
    A road runs INSIDE S1; another meets S3's top edge at exactly one point.
    Types are chosen so both a reported and a dropped type are exercised."""
    settlements = gpd.GeoDataFrame(
        {"USO_AREA_U": ["S1", "S2", "S3", "S4"],
         "USO_FINAL": ["JJC", "JJC", "Planned", "RV"]},
        geometry=[box(0, 0, 1000, 1000), box(1000, 0, 2000, 1000),
                  box(2000, 0, 3000, 1000), box(3000, 0, 4000, 1000)],
        crs="EPSG:7760")
    roads = gpd.GeoDataFrame(
        {"name": ["inside S1", "corner of S3"]},
        geometry=[LineString([(200, 500), (800, 500)]),
                  LineString([(2500, 1000), (2500, 1500)])],
        crs="EPSG:7760")
    return settlements, roads


# --- block 1: road access ----------------------------------------------
def test_a_road_that_only_touches_the_boundary_is_not_inside():
    """`index.road_lengths` clips the road to the polygon and sums the
    pieces, so a point contact contributes zero length — road_inside must
    mean the same thing, or the block would not describe today's outputs."""
    settlements, roads = four_squares()
    assert road_inside_ids(settlements, roads) == {"S1"}


def test_road_access_on_four_squares_in_a_row():
    settlements, roads = four_squares()
    got = measure_access(settlements, roads)
    assert got["road_inside_JJC"] == 1          # S1
    assert got["road_via_neighbor_JJC"] == 1    # S2 touches S1
    assert got["no_road_JJC"] == 0
    assert got["road_inside_Planned"] == 0
    assert got["no_road_Planned"] == 1          # S3: point contact only
    assert got["no_road_RV"] == 1               # S4
    assert got["road_inside_total"] == 1
    assert got["road_via_neighbor_total"] == 1
    assert got["no_road_total"] == 2
    # every reported and dropped type gets a key, even at zero
    assert got["road_inside_SDA"] == 0
    assert got["no_road_Other"] == 0


def test_measure_access_refuses_a_settlement_type_it_has_no_key_for():
    settlements, roads = four_squares()
    settlements.loc[0, "USO_FINAL"] = "Nonesuch"
    with pytest.raises(ValueError, match="Nonesuch"):
        measure_access(settlements, roads)


# --- road_inside <=> road_length > 0 (spec § 2.1 (a), § 3) --------------
def output_csv(tmp_path, ids, lengths, name="verify.csv"):
    """A code-2025-shaped output CSV: one row per REPORTED settlement, with
    the `road_length` column production writes."""
    path = tmp_path / name
    pd.DataFrame({"USO_AREA_U": ids, "road_length": lengths}).to_csv(
        path, index=False)
    return path


def test_the_road_inside_set_matches_the_outputs_road_length_column(tmp_path):
    """The equivalence spec § 2.1 (a) asserts: this script's own geometry
    (`road_inside_ids`) and production's `index.road_lengths` must classify
    the same settlements. S4 is left OUT of the CSV on purpose — the run
    excludes some types (RV), and the comparison is over the ids the CSV
    reports, not over the whole layer."""
    settlements, roads = four_squares()
    inside = road_inside_ids(settlements, roads)
    csv = output_csv(tmp_path, ["S1", "S2", "S3"], [600.0, 0.0, 0.0])
    assert assert_road_inside_matches_output(
        inside, csv, id_col="USO_AREA_U") is None


def test_a_road_inside_mismatch_against_the_output_raises_both_ways(tmp_path):
    """Not a warning. If the two disagree, the access block would misdescribe
    today's outputs, so the script stops. The message names the offenders in
    both directions — inside-but-zero-length and positive-length-but-not-inside
    — because which way it fell tells you which side is wrong."""
    settlements, roads = four_squares()
    inside = road_inside_ids(settlements, roads)          # {"S1"}
    csv = output_csv(tmp_path, ["S1", "S2", "S3"], [0.0, 0.0, 5.0])
    with pytest.raises(ValueError) as exc:
        assert_road_inside_matches_output(inside, csv, id_col="USO_AREA_U")
    message = str(exc.value)
    assert "road_inside" in message and "road_length" in message
    assert "S1" in message and "S3" in message


# --- block 2: the derived profile and the diff -------------------------
def test_the_derived_profile_changes_exactly_one_value(tmp_path):
    """One methodology value changed and two path keys dropped (so the
    per-profile artifact name and --out-dir apply); everything else byte-equal
    after the YAML round trip."""
    path = derived_profile("code-2025", tmp_path)
    got = yaml.safe_load(path.read_text())
    assert path.name == f"{OWN_ONLY_PROFILE}.yaml"
    assert got["methodology"]["roads"] == "eq4_own_only"
    assert got["profile"] == OWN_ONLY_PROFILE
    assert "neighbors_artifact" not in got["paths"]
    assert "out_dir" not in got["paths"]

    expected = yaml.safe_load((PROFILES_DIR / "code-2025.yaml").read_text())
    expected["methodology"]["roads"] = "eq4_own_only"
    expected["profile"] = OWN_ONLY_PROFILE
    expected["paths"].pop("neighbors_artifact")
    assert got == expected


def test_the_methodology_stamp_does_not_carry_roads():
    """The whole one-factor design rests on this: `roads` is applied
    downstream in `compute`, so the proven neighbours artifact stays valid
    and no 11-minute preprocess is needed. If a future stamp change adds it,
    this fails loudly instead of the script silently re-preprocessing."""
    from delhi_psi.config import load_config

    stamp = pipeline.methodology_stamp(load_config("code-2025").methodology)
    assert set(stamp) == {"adjacency", "barrier"}
    assert not any("roads" in block for block in stamp.values())


def test_stage_artifacts_copies_the_artifact_under_the_derived_name(tmp_path):
    verify = tmp_path / "verify"
    verify.mkdir()
    (verify / "colonies_neighbors.joblib").write_bytes(b"artifact")
    target = stage_artifacts(
        verify, tmp_path / "run", source_name="colonies_neighbors.joblib",
        artifact_name=f"colonies_neighbors_{OWN_ONLY_PROFILE}.joblib")
    assert target.name == f"colonies_neighbors_{OWN_ONLY_PROFILE}.joblib"
    assert target.read_bytes() == b"artifact"


def effect_frames():
    """Two four-row outputs: JJC J1 keeps a road of its own, JJC J2 borrowed
    everything and falls to zero, Planned P1 is unchanged, Planned P2 was
    already zero (so it did not FALL to zero)."""
    decayed = pd.DataFrame({
        "USO_AREA_U": ["J1", "J2", "P1", "P2"],
        "USO_FINAL": ["JJC", "JJC", "Planned", "Planned"],
        "road_idx": [0.4, 0.2, 0.8, 0.0],
        "unnorm_psi": [0.10, 0.20, 0.30, 0.40]})
    own = pd.DataFrame({
        "USO_AREA_U": ["J1", "J2", "P1", "P2"],
        "USO_FINAL": ["JJC", "JJC", "Planned", "Planned"],
        "road_idx": [0.4, 0.0, 0.8, 0.0],
        "unnorm_psi": [0.10, 0.15, 0.30, 0.40]})
    return decayed, own


def test_measure_effect_reports_means_and_the_settlements_zeroed():
    decayed, own = effect_frames()
    got = measure_effect(decayed, own, denom="pop")
    assert got["n_pop_JJC"] == 2
    assert got["n_pop_total"] == 4
    assert got["road_idx_decayed_pop_JJC"] == "0.3"
    assert got["road_idx_own_pop_JJC"] == "0.2"
    assert got["psi_decayed_pop_JJC"] == "0.15"
    assert got["psi_own_pop_JJC"] == "0.125"
    assert got["road_idx_zeroed_pop_JJC"] == 1       # J2 only
    assert got["road_idx_zeroed_pop_Planned"] == 0   # P2 was already zero
    assert got["road_idx_zeroed_pop_total"] == 1


def test_measure_effect_refuses_a_partial_join():
    decayed, own = effect_frames()
    with pytest.raises(ValueError, match="different settlements"):
        measure_effect(decayed, own.iloc[:3], denom="pop")


# --- the Oraculum one-factor proof -------------------------------------
def test_the_pinned_ideal_road_values_are_the_reference_implementations():
    """Confirmed against the fixture rather than against the memo: the memo's
    "A 0.0075, E 0.0025, others 0" are `road_pcen`, not `road_idx`."""
    expected = pd.read_csv(REFERENCE_CSV)
    expected = expected[(expected["rule"] == "ideal")
                        & (expected["scenario"] == "baseline")
                        & (expected["denom"] == "pop")].pivot(
        index="settlement", columns="metric", values="value")
    for sid, value in IDEAL_ROAD_PCEN.items():
        assert expected.loc[sid, "road_pcen"] == pytest.approx(value, abs=1e-12)
    for sid, value in IDEAL_ROAD_IDX.items():
        assert expected.loc[sid, "road_idx"] == pytest.approx(value, abs=1e-12)


def test_own_only_reproduces_the_reference_ideal_roads_column(data_dir,
                                                              tmp_path):
    """The one-factor machinery — this script's derived profile driven through
    the real CLI — on the city where the answer is known by hand.

    BOTH roads columns are pinned, against
    `tests/fixtures/oraculum/expected_values.csv` rows with **rule=`ideal`,
    scenario=`baseline`, denom=`pop`** (spec § 4): `road_pcen` A 0.0075,
    E 0.0025, others 0; `road_idx` A 1.0, E 1/3, others 0. The memo § 3 quoted
    the PCEN values as if they were the index — pinning both is what stops
    that confusion recurring.

    The base is the DERIVED oracle profile (the shipped `uso-10` mapping does
    not cover this city's vocabulary). The reference's `ideal` rule differs
    from `code-2025` in more than roads, but under `eq4_own_only` road_pcen is
    own length / own denominator — per-row and independent of adjacency,
    barriers and who else is in the frame — and road_idx is its min-max, whose
    min (0) and max (A's 0.0075) are the same in both frames. So these are the
    right values to expect.
    """
    base = oracle_profile_path("code-2025", tmp_path)
    run_dir = tmp_path / "own_only"
    profile = derived_profile(base, run_dir)
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0
    assert (run_dir / f"colonies_neighbors_{OWN_ONLY_PROFILE}.joblib").exists()
    assert cli.main(["compute", "--config", str(profile),
                     "--data-dir", str(data_dir),
                     "--out-dir", str(run_dir)]) == 0

    got = pd.read_csv(
        run_dir / f"delhi_psi_{OWN_ONLY_PROFILE}_pop_2020.csv"
    ).set_index("USO_AREA_U")
    assert set(got.index) == set(IDEAL_ROAD_PCEN)
    for sid, value in IDEAL_ROAD_PCEN.items():
        assert got.loc[sid, "road_pcen"] == pytest.approx(value, abs=1e-9), sid
    for sid, value in IDEAL_ROAD_IDX.items():
        assert got.loc[sid, "road_idx"] == pytest.approx(value, abs=1e-9), sid


# --- the committed document --------------------------------------------
def committed_blocks():
    if not DOC.exists() or FENCE not in DOC.read_text():
        pytest.skip(f"{DOC} carries no measured block yet — the run step "
                    "pastes it")
    text = DOC.read_text()
    return (parse_block(text, name="access"),
            parse_block(text, name="one_factor"))


def test_the_doc_blocks_have_every_required_key():
    access, one_factor = committed_blocks()
    for kind in ("road_inside", "road_via_neighbor", "no_road"):
        for name in (*REPORTED_TYPES, "RV", "Industrial", "Other", "total"):
            assert access[f"{kind}_{name}"].isdigit(), f"{kind}_{name}"
    for denom in DENOMINATORS:
        for name in (*REPORTED_TYPES, "total"):
            assert one_factor[f"n_{denom}_{name}"].isdigit()
            for metric in ("road_idx_decayed", "road_idx_own", "psi_decayed",
                           "psi_own"):
                float(one_factor[f"{metric}_{denom}_{name}"])
            assert one_factor[f"road_idx_zeroed_{denom}_{name}"].isdigit()


def test_the_doc_records_its_provenance_and_quotes_only_block_numbers():
    text = DOC.read_text()
    for label in ("**Run date:**", "**Inputs:**", "**Commit:**",
                  "**Command:**"):
        assert label in text, label
    assert_prose_numbers_come_from_the_blocks(text, committed_blocks())


@needs_measure_cache
def test_a_fresh_run_reproduces_the_committed_blocks():
    """The real-data drift check, and with it the real-layer exercise of
    `assert_road_inside_matches_output`: the script raises before printing if
    `road_inside` and `road_length > 0` disagree, so a zero exit code IS that
    assertion passing on all 4,131 reported settlements.

    It costs a settlement dedup, a `touch` adjacency over 4,357 polygons and a
    full `compute`, so it takes its work dir from DELHI_PSI_MEASURE_CACHE (one
    warm cache per machine, shared with the barriers drift test) and SKIPS
    when that is unset — `needs_data` alone would charge every per-task suite
    run on this machine ~4.5 min for a cold dedup. The run step exports it. A
    warm cache is safe here: nothing in this script inspects `geom_type`.
    """
    access, one_factor = committed_blocks()
    proc = subprocess.run(
        [sys.executable, "scripts/measure_roads_access.py",
         "--config", "code-2025", "--data-dir", str(DATA_DIR),
         "--verify-dir", str(VERIFY_DIR),
         "--work-dir", MEASURE_CACHE],
        cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert parse_block(proc.stdout, name="access") == access
    assert parse_block(proc.stdout, name="one_factor") == one_factor


def test_main_requires_a_verify_dir(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--config", "code-2025"])
    assert exc.value.code == 2
    assert "--verify-dir" in capsys.readouterr().err
