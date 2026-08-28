"""What production does on each messy-city pathology, TODAY (spec 3C § 4.3).

These are direct assertions against production, not comparisons against the
reference — `tests/test_profiles_match_reference.py` already proves the two
agree on this city to 1e-12. Each pin here RECORDS today's behaviour on a
pathology the real layer has and Oraculum cannot express:

  bbox adjacency invents neighbours (DEL-19)  -> the H/L, T/L and G/M pins
  overlapping polygons double-count a point (DEL-20) -> the O1/O2 clinic pin

When a fix lands, the pin it flips is the proof that it landed. A pin that
starts failing for any OTHER reason means adjacency, exclusion or counting
changed silently — investigate, never re-record.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from tests.cities import MESSY

PROFILES = ("code-2025", "manuscript")
DENOMS = ("pop", "popdensity")
SCENARIOS = {scenario.name: scenario for scenario in MESSY.scenarios}

# The two shipped profiles ARE the two adjacency rules on this city.
BBOX_PROFILE = "code-2025"     # methodology.adjacency.rule: bbox
TOUCH_PROFILE = "manuscript"   # methodology.adjacency.rule: touch


@lru_cache(maxsize=None)
def frame(profile, scenario_name, denom):
    """One production run, memoised. Treat the result as READ-ONLY: every
    test in this file shares the same object."""
    from tests.oraculum_fixtures import compute_oracle_frame

    scenario = SCENARIOS[scenario_name]
    return compute_oracle_frame(profile, types=scenario.exclusion_types,
                                stage=scenario.stage, denom=denom, city=MESSY)


def nbrs(result, sid):
    return set(result.loc[sid, "nbrs_bbox"])


def denominator(result, sid, denom):
    population = result.loc[sid, "population"]
    if denom == "pop":
        return population
    return population / result.loc[sid, "area_km2"]


@pytest.fixture(scope="module")
def geoms():
    city = MESSY.load_settlements().set_index("USO_AREA_U")
    return {sid: row.geometry for sid, row in city.iterrows()}


# --- bbox adjacency invents neighbours (DEL-19) ------------------------
def test_h_and_l_are_disjoint_but_each_reaches_the_others_envelope(geoms):
    """The bbox rule is DIRECTED, so each shape has to reach into the other's
    envelope for the pair to be neighbours both ways."""
    assert geoms["H"].intersection(geoms["L"]).is_empty
    assert not geoms["H"].intersection(geoms["L"].envelope).is_empty
    assert not geoms["L"].intersection(geoms["H"].envelope).is_empty


def test_bbox_pairs_are_directed_as_the_spec_says():
    got = frame(BBOX_PROFILE, "nopop_only", "pop")
    assert "L" in nbrs(got, "H") and "H" in nbrs(got, "L")
    assert "T" in nbrs(got, "L") and "L" in nbrs(got, "T")
    # the asymmetric one: an axis-aligned square IS its own envelope, so M
    # never reaches G even though G sits inside M's envelope.
    assert "M" in nbrs(got, "G")
    assert "G" not in nbrs(got, "M")


def test_touch_has_none_of_the_bbox_only_pairs():
    got = frame(TOUCH_PROFILE, "nopop_only", "pop")
    assert "L" not in nbrs(got, "H") and "H" not in nbrs(got, "L")
    assert "T" not in nbrs(got, "L") and "L" not in nbrs(got, "T")
    assert "M" not in nbrs(got, "G") and "G" not in nbrs(got, "M")


def test_corner_contact_between_t_and_l_is_a_single_point(geoms):
    shared = geoms["T"].intersection(geoms["L"])
    assert shared.geom_type == "Point"
    assert shared.length == 0


def test_overlapping_pair_are_neighbours_under_both_rules():
    """THE DEL-19 finding: the `touch` test asks for `.length > 0`, and an
    overlap polygon's `.length` is its PERIMETER — so two overlapping
    polygons are 'border-sharing' neighbours."""
    for profile in PROFILES:
        got = frame(profile, "nopop_only", "pop")
        assert "O2" in nbrs(got, "O1"), profile
        assert "O1" in nbrs(got, "O2"), profile


# --- the MultiPolygon and its gap --------------------------------------
def test_multipolygon_has_two_parts_and_its_centroid_lies_outside_it(geoms):
    assert geoms["M"].geom_type == "MultiPolygon"
    assert len(geoms["M"].geoms) == 2
    assert not geoms["M"].centroid.within(geoms["M"])


def test_gap_settlement_sits_exactly_on_the_multipolygons_centroid(geoms):
    assert geoms["G"].disjoint(geoms["M"])
    assert geoms["G"].centroid.equals(geoms["M"].centroid)
    assert geoms["G"].centroid.distance(geoms["M"].centroid) == 0.0


def test_bbox_neighbours_of_the_gap_settlement_are_exactly_the_multipolygon():
    assert nbrs(frame(BBOX_PROFILE, "nopop_only", "pop"), "G") == {"M"}


@pytest.mark.parametrize("denom", DENOMS)
def test_gap_settlement_clinic_pcen_is_the_undecayed_weight_one_case(denom):
    """d = 0 -> 1/(1+d) is exactly 1, so G's PCEN is a plain SUM of its own
    and M's clinic counts. Asserted with `==`, not approx: any decay at all
    would move it."""
    got = frame(BBOX_PROFILE, "nopop_only", denom)
    assert got.loc["G", "nbrs_dist_bbox"] == [("M", 0.0)]
    expected = ((got.loc["G", "clinic_count"] + got.loc["M", "clinic_count"])
                / denominator(got, "G", denom))
    assert got.loc["G", "clinic_pcen"] == expected


# --- overlapping polygons double-count a point (DEL-20) ----------------
def test_overlap_has_positive_area(geoms):
    assert geoms["O1"].intersection(geoms["O2"]).area > 0


def test_the_overlap_clinic_is_counted_for_both_owners():
    """One physical clinic, strictly inside the overlap, counted for BOTH.
    Agreed behaviour: production's `intersects` and the reference's `within`
    both do it, so it is pinned directly, never by comparison."""
    services = MESSY.load_services()
    assert (services["clinic"]["host"] == "O1+O2").sum() == 1
    for profile in PROFILES:
        got = frame(profile, "nopop_only", "pop")
        assert got.loc["O1", "clinic_count"] == 1, profile
        assert got.loc["O2", "clinic_count"] == 1, profile


# --- the isolated settlement -------------------------------------------
def test_isolated_settlement_has_no_neighbours_under_either_rule():
    for profile in PROFILES:
        got = frame(profile, "nopop_only", "pop")
        assert list(got.loc["I", "nbrs_bbox"]) == [], profile
        assert list(got.loc["I", "nbrs_dist_bbox"]) == [], profile


def test_isolated_settlement_is_disjoint_from_every_other(geoms):
    for sid, geom in geoms.items():
        if sid != "I":
            assert geoms["I"].disjoint(geom), sid


@pytest.mark.parametrize("denom", DENOMS)
@pytest.mark.parametrize("profile", PROFILES)
def test_isolated_settlement_clinic_pcen_is_own_over_denominator(profile,
                                                                 denom):
    got = frame(profile, "nopop_only", denom)
    assert got.loc["I", "clinic_pcen"] == (
        got.loc["I", "clinic_count"] / denominator(got, "I", denom))


# --- category exclusion (N) --------------------------------------------
@pytest.mark.parametrize("profile", PROFILES)
def test_rv_settlement_is_reported_only_when_it_is_not_excluded(profile):
    """`nopop_only` and `excl_rv_post` differ exactly by N, so it is the
    CATEGORY exclusion that removed it — not the missing-population rule."""
    assert "N" in frame(profile, "nopop_only", "pop").index
    assert "N" not in frame(profile, "excl_rv_post", "pop").index
    assert "N" not in frame(profile, "excl_rv_pre", "pop").index


def test_rv_settlement_leaves_o2s_neighbour_list_only_under_pre_neighbors():
    assert "N" in nbrs(frame(BBOX_PROFILE, "excl_rv_post", "pop"), "O2")
    assert "N" not in nbrs(frame(BBOX_PROFILE, "excl_rv_pre", "pop"), "O2")


# --- the settlement with no population row (U) -------------------------
@pytest.mark.parametrize("denom", DENOMS)
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("profile", PROFILES)
def test_no_population_settlement_is_never_reported(profile, scenario, denom):
    """Production drops a no-population row UNCONDITIONALLY — every profile,
    every scenario, every denominator — because `dropped` is
    `excluded_ids ∪ missing`."""
    assert "U" not in frame(profile, scenario, denom).index


def test_no_population_settlement_leaves_o1s_neighbour_list_only_under_pre_neighbors():
    """...but it stays in other settlements' neighbour lists, unless the
    scenario's single `stage` is `pre_neighbors` — which strips the whole
    drop set, `U` included."""
    assert "U" in nbrs(frame(BBOX_PROFILE, "nopop_only", "pop"), "O1")
    assert "U" in nbrs(frame(BBOX_PROFILE, "excl_rv_post", "pop"), "O1")
    assert "U" not in nbrs(frame(BBOX_PROFILE, "excl_rv_pre", "pop"), "O1")


def test_missing_population_error_names_the_settlement_with_no_row():
    from delhi_psi.pipeline import compute_frames
    from delhi_psi.validate import ValidationError
    from tests.oraculum_fixtures import methodology_with

    methodology = methodology_with(BBOX_PROFILE, types=(),
                                   stage="post_neighbors", city=MESSY)
    with pytest.raises(ValidationError) as excinfo:
        compute_frames(MESSY.load_settlements(),
                       {"canal": MESSY.load_barriers()},
                       MESSY.load_services(), None, methodology, "pop",
                       mapping=MESSY.mapping(), scheme=MESSY.scheme,
                       missing_population="error")
    assert "'U'" in str(excinfo.value)
    assert "no population row" in str(excinfo.value)


def test_populations_are_distinct_and_only_u_has_none():
    city = MESSY.load_settlements().set_index("USO_AREA_U")
    assert pd.isna(city.loc["U", "population"])
    assert not pd.isna(city.loc["N", "population"])
    present = city["population"].dropna()
    assert len(present) == len(city) - 1 == 10
    assert len(set(present)) == len(present), "populations must not tie"


# --- the area-extreme sliver (S) ---------------------------------------
def test_sliver_area_is_exactly_two_square_metres(geoms):
    city = MESSY.load_settlements().set_index("USO_AREA_U")
    assert city.loc["S", "area_km2"] == 2e-06
    assert geoms["S"].area > 0
    assert geoms["S"].area == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize("profile", PROFILES)
def test_sliver_ration_pcen_is_the_popdensity_minimum_among_owners(profile):
    """Scoped to the settlements that OWN a ration point: a settlement with
    no ration point but a serving neighbour can sit above M (H does), and
    one with neither sits at exactly 0 (I always does)."""
    got = frame(profile, "nopop_only", "popdensity")
    owners = sorted(sid for sid in got.index if got.loc[sid, "ration_count"] > 0)
    assert owners == ["M", "S"]
    assert got.loc["S", "ration_pcen"] == min(
        got.loc[owner, "ration_pcen"] for owner in owners)
    assert got.loc["M", "ration_pcen"] / got.loc["S", "ration_pcen"] >= 1e4
    assert got.loc["I", "ration_pcen"] == 0.0


def test_no_ration_ordering_is_claimed_under_pop():
    """Under `pop` the area is irrelevant, and the order flips — which is
    exactly why the spec scopes the sliver claim to `popdensity`."""
    got = frame(BBOX_PROFILE, "nopop_only", "pop")
    assert got.loc["S", "ration_pcen"] > got.loc["M", "ration_pcen"]


# --- the two-row road layer and the full service set -------------------
def test_the_road_layer_has_two_rows_and_lengths_are_summed():
    roads = MESSY.load_services()["road"]
    assert len(roads) == 2
    got = frame(BBOX_PROFILE, "nopop_only", "pop")
    for sid, km in (("H", 1.2), ("L", 0.6), ("M", 2.0)):
        assert got.loc[sid, "road_length"] == pytest.approx(km, abs=1e-12), sid
    for sid in got.index:
        if sid not in ("H", "L", "M"):
            assert got.loc[sid, "road_length"] == 0.0, sid
    # M's whole length comes from the SECOND row, so "first row only" is
    # observably wrong here.
    multipolygon = MESSY.load_settlements().set_index(
        "USO_AREA_U").loc["M"].geometry
    assert roads.geometry.iloc[0].intersection(multipolygon).length == 0.0


def test_all_seven_services_are_present_on_both_sides():
    """The reference scores every service in POINT_SERVICES regardless, and
    production's PSI averages over the services present — so a service
    missing from this city would make the two average different things."""
    from tests.reference_impl import POINT_SERVICES

    services = MESSY.load_services()
    assert set(services) == set(POINT_SERVICES) | {"road"}
    got = frame(BBOX_PROFILE, "nopop_only", "pop")
    for service in POINT_SERVICES:
        assert f"{service}_pcen" in got.columns, service
        assert got[f"{service}_count"].sum() > 0, service


# --- the tier's documentation (spec § 6) -------------------------------
DOC = Path(__file__).resolve().parent.parent / "docs" / "oracle" / "messy-city.md"


def test_the_messy_city_doc_documents_every_settlement():
    """Eleven settlements, each with a stated pathology and a stated pin. A
    settlement the doc does not name is a case nobody can maintain."""
    text = DOC.read_text()
    city = MESSY.load_settlements()
    for sid in city["USO_AREA_U"]:
        assert f"`{sid}`" in text, sid
    for pathology in ("MultiPolygon", "overlap", "isolated", "population",
                      "sliver", "envelope"):
        assert pathology.lower() in text.lower(), pathology
    assert "## How to add a case" in text


def test_methodology_config_section_4_says_the_proofs_run_on_both_cities():
    config_doc = (Path(__file__).resolve().parent.parent / "docs"
                  / "methodology-config.md").read_text()
    section = config_doc.split("## 4. What each proof guards")[1].split("## 5.")[0]
    assert "oraculum" in section.lower()
    assert "messy" in section.lower()
