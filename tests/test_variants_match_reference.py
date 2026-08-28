"""Production reproduces the independent reference on every variant (§ 4.2).

ONE scenario per city — `city.scenarios[0]`, the one the variants CSV is
written for — because the exclusion machinery is proven elsewhere and is
orthogonal to the two knobs this cycle adds. The scenario travels with the
methodology (`variant_methodology`), so the production frame reports exactly
the settlements the CSV holds.
"""
import pandas as pd
import pytest

from delhi_psi import cli
from delhi_psi.pipeline import compute_frames
from tests.cities import CITIES, MESSY, ORACULUM
from tests.oraculum_fixtures import oracle_profile_path, variant_methodology
from tests.test_cli import COLLAPSE_TO_REFERENCE, data_dir  # noqa: F401  (module-scoped fixture)
from tests.test_profiles_match_reference import METRIC_MAP
from tests.variants import VARIANTS

BASE_PROFILE = "code-2025"
DENOMS = ("pop", "popdensity")
CASES = [(city, variant) for city in CITIES for variant in sorted(VARIANTS)]


def case_id(case):
    city, variant = case
    return f"{city.name}-{variant}"


@pytest.fixture(scope="module")
def expected():
    return {city.name: pd.read_csv(
        city.fixtures / "variants_expected_values.csv") for city in CITIES}


def produced(city, variant, denom):
    scenario = city.scenarios[0]
    methodology = variant_methodology(
        BASE_PROFILE, variant, city=city,
        types=scenario.exclusion_types, stage=scenario.stage)
    return compute_frames(
        city.load_settlements(), {"canal": city.load_barriers()},
        city.load_services(), None, methodology, denom,
        mapping=city.mapping(), scheme=city.scheme).set_index("USO_AREA_U")


@pytest.mark.parametrize("denom", DENOMS)
@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_production_matches_the_reference_on_each_variant(expected, case,
                                                          denom):
    city, variant = case
    got = produced(city, variant, denom)
    block = expected[city.name]
    block = block[(block["rule"] == variant) & (block["denom"] == denom)]
    exp = block.pivot(index="settlement", columns="metric", values="value")
    assert set(got.index) == set(exp.index)
    for prod_col, metric in METRIC_MAP.items():
        for sid in exp.index:
            assert got.loc[sid, prod_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-12), (city.name, variant, denom,
                                                   sid, prod_col)


def test_the_boundary_column_never_leaves_index_frames():
    """The compute-local column is dropped before returning, and the stored
    `nbrs_dist_bbox` holds CENTROID distances under EVERY configuration —
    which is what lets one artifact serve every decay.* value. Messy `G`'s
    only neighbour is `M`, at centroid distance exactly 0 and boundary
    distance 0.45 km, so the column's content is unambiguous."""
    got = produced(MESSY, "boundary", "pop")
    assert "nbrs_dist_boundary" not in got.columns
    assert dict(got.loc["G", "nbrs_dist_bbox"]) == {"M": 0.0}


# --- the CLI leg: config file -> artifact -> compute (spec § 4.2) ------
# A variant profile must state each block it overrides IN FULL, and it must
# also state the SCENARIO the variants CSV was written for — Oraculum's
# `baseline`, i.e. no exclusion at all, where the shipped profile excludes
# the category RV.
BASELINE_EXCLUSION = {"types": [], "stage": "post_neighbors",
                      "absent_neighbor": "swallowed"}


@pytest.mark.parametrize("variant", ["band_small_boundary", "exp1"])
def test_a_derived_variant_profile_runs_end_to_end(expected, data_dir,  # noqa: F811
                                                   tmp_path, variant):
    """Proves the whole chain the in-memory test skips: YAML -> load_config
    -> preprocess -> the stamped artifact -> compute -> CSV. `exp1` is here
    for `scale_km`; `band_small_boundary` for the band, the boundary
    distance and the stamped `max_distance_km` together.
    """
    overrides = dict(VARIANTS[variant])
    overrides["exclusion"] = BASELINE_EXCLUSION
    profile = oracle_profile_path(BASE_PROFILE, tmp_path,
                                  methodology_overrides=overrides,
                                  name=variant)
    out = tmp_path / variant
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    assert cli.main(["compute", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0

    block = expected["oraculum"]
    block = block[(block["rule"] == variant) & (block["denom"] == "pop")]
    exp = block.pivot(index="settlement", columns="metric", values="value")
    got = pd.read_csv(
        out / "delhi_psi_code-2025_pop_2020.csv").set_index("USO_AREA_U")
    assert set(got.index) == set(exp.index)
    for got_col, metric in COLLAPSE_TO_REFERENCE.items():
        for sid in exp.index:
            assert got.loc[sid, got_col] == pytest.approx(
                exp.loc[sid, metric], abs=1e-9), (variant, sid, got_col)


def test_the_stored_artifact_records_the_bands_radius(data_dir, tmp_path):  # noqa: F811
    """The stamp is what stops a `compute` reading one band's neighbour
    lists under another band's config."""
    from delhi_psi import io

    overrides = dict(VARIANTS["band_small"])
    overrides["exclusion"] = BASELINE_EXCLUSION
    profile = oracle_profile_path(BASE_PROFILE, tmp_path,
                                  methodology_overrides=overrides,
                                  name="band_small")
    out = tmp_path / "stamped_band"
    assert cli.main(["preprocess", "--config", str(profile),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    frame = io.read_neighbors(out / "colonies_neighbors.joblib")
    assert frame.attrs["methodology"]["adjacency"] == {
        "rule": "within_distance", "max_distance_km": 0.25}


def test_compute_refuses_an_artifact_built_at_another_band(data_dir,  # noqa: F811
                                                           tmp_path, capsys):
    """Build at 0.25 km, compute at 0.75 km: every number compute would
    produce describes a neighbourhood nobody built."""
    import shutil

    small = dict(VARIANTS["band_small"], exclusion=BASELINE_EXCLUSION)
    large = dict(VARIANTS["band_large"], exclusion=BASELINE_EXCLUSION)
    out = tmp_path / "band_mismatch"
    built = oracle_profile_path(BASE_PROFILE, tmp_path,
                                methodology_overrides=small, name="small")
    assert cli.main(["preprocess", "--config", str(built),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 0
    other = oracle_profile_path(BASE_PROFILE, tmp_path,
                                methodology_overrides=large, name="large")
    assert cli.main(["compute", "--config", str(other),
                     "--data-dir", str(data_dir), "--out-dir", str(out)]) == 1
    err = capsys.readouterr().err
    assert "max_distance_km" in err and "0.25" in err and "0.75" in err
