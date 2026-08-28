"""Production reproduces the independent reference on every variant (§ 4.2).

ONE scenario per city — `city.scenarios[0]`, the one the variants CSV is
written for — because the exclusion machinery is proven elsewhere and is
orthogonal to the two knobs this cycle adds. The scenario travels with the
methodology (`variant_methodology`), so the production frame reports exactly
the settlements the CSV holds.
"""
import pandas as pd
import pytest

from delhi_psi.pipeline import compute_frames
from tests.cities import CITIES, MESSY
from tests.oraculum_fixtures import variant_methodology
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
