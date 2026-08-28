"""Hand-derivable pins for the injectable parameters (spec § 4.1, items 1-6).

Every number here was derived from the fixture geometry and checked against
the fixtures; the arithmetic is spelled out in each test so a reader can
re-derive it on paper. `within_distance` is POLYGON-TO-POLYGON distance, so
the radii are judged against boundary distances, never the worksheet's
centroid distances.

Reference side only: nothing in this file imports delhi_psi.
"""
import math

import pytest

from tests.cities import CITIES, MESSY, ORACULUM
from tests.reference_impl import (
    RULESETS, VARIANT_KNOBS, VARIANT_RULESETS, adjacency, apply_barrier,
    compute_city,
)
from tests.variants import (
    ADDED_BAND_PAIRS, BAND_RADII_KM, EXPECTED_BAND_PAIRS, VARIANTS,
)

B0, B1, B2 = BAND_RADII_KM          # 0.0, 0.25, 0.75 km


def undirected(nbrs):
    return {tuple(sorted((i, j))) for i, js in nbrs.items() for j in js}


def band(city, km):
    return adjacency(city.load_settlements(), "within_distance", km)


def scored(city, ruleset, denom="pop"):
    """The city's FIRST scenario under `ruleset` (a VARIANT_RULESETS entry or
    a RULESETS entry), indexed by settlement id."""
    scenarios = {s.name: (s.dropped, s.dropped_before_neighbors)
                 for s in city.scenarios}
    return compute_city(city.load_settlements(), city.load_services(),
                        city.load_barriers(), scenario=city.scenarios[0].name,
                        denom=denom, scenarios=scenarios, **ruleset)


def variant(city, name, denom="pop"):
    return scored(city, VARIANT_RULESETS[name], denom)


# --- item 1: what a 0 km band IS ---------------------------------------
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_band_zero_is_the_intersects_neighbourhood(city):
    assert band(city, B0) == adjacency(city.load_settlements(), "intersects")


def test_on_oraculum_band_zero_is_also_touch_and_undirected_bbox():
    """Ten pairs, three ways: that city has no corner-only contact and no
    overlap, which is exactly what the messy city adds."""
    settlements = ORACULUM.load_settlements()
    zero = undirected(band(ORACULUM, B0))
    assert zero == undirected(adjacency(settlements, "border"))
    assert zero == undirected(adjacency(settlements, "bbox"))
    assert len(zero) == 10


def test_on_messy_band_zero_is_touch_plus_the_corner_only_pair():
    """L and T meet at the single Point (2000, 800): `touch` wants positive
    LENGTH, so it misses them; a 0 km band does not. O1 and O2 OVERLAP, and
    an overlap's intersection is a Polygon whose `.length` is its perimeter
    (2400 m), so `touch` already accepts that one."""
    settlements = MESSY.load_settlements()
    geoms = settlements.set_index("USO_AREA_U").geometry
    assert geoms["L"].intersection(geoms["T"]).geom_type == "Point"
    assert geoms["L"].intersection(geoms["T"]).length == 0.0
    assert geoms["O1"].intersection(geoms["O2"]).geom_type == "Polygon"
    assert geoms["O1"].intersection(geoms["O2"]).length == 2400.0

    zero = undirected(band(MESSY, B0))
    touch = undirected(adjacency(settlements, "border"))
    assert zero - touch == {("L", "T")}
    assert touch - zero == set()
    assert {("L", "T"), ("O1", "O2")} <= undirected(
        adjacency(settlements, "bbox"))


@pytest.mark.parametrize("km", BAND_RADII_KM)
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_the_band_is_symmetric(city, km):
    nbrs = band(city, km)
    for i, js in nbrs.items():
        for j in js:
            assert i in nbrs[j], (i, j, km)


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_pre_barrier_pair_counts_and_the_pairs_each_radius_adds(city):
    """Counted on adjacency()'s OWN output — never downstream of a barrier
    rule, which would fold the canal's severing into the band's numbers."""
    pairs = {km: undirected(band(city, km)) for km in BAND_RADII_KM}
    assert {km: len(p) for km, p in pairs.items()} == \
        EXPECTED_BAND_PAIRS[city.name]
    assert pairs[B1] - pairs[B0] == ADDED_BAND_PAIRS[city.name][B1]
    assert pairs[B2] - pairs[B1] == ADDED_BAND_PAIRS[city.name][B2]


# --- item 2: monotonicity ----------------------------------------------
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_bands_are_nested_and_strictly_growing(city):
    small, large = band(city, B1), band(city, B2)
    for i, js in small.items():
        assert js <= large[i], i
    assert any(large[i] > small[i] for i in small), "no pair added at B2"


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_every_touching_pair_is_in_every_band(city):
    """The large-neighbour property that motivated the boundary definition:
    a shared border is distance 0, so no radius can drop it."""
    touch = adjacency(city.load_settlements(), "border")
    for km in BAND_RADII_KM:
        nbrs = band(city, km)
        for i, js in touch.items():
            assert js <= nbrs[i], (i, km)


# --- item 3: inverse_power 1 == inverse_linear -------------------------
@pytest.mark.parametrize("denom", ["pop", "popdensity"])
@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_inverse_power_one_reproduces_inverse_linear(city, denom):
    """`x ** 1.0 == x` exactly in IEEE, so this holds at 0 tolerance; the
    pin uses 1e-12 anyway."""
    base = scored(city, RULESETS["code"], denom)
    got = variant(city, "pow1", denom)
    assert list(got.columns) == list(base.columns)
    for column in base.columns:
        assert list(got[column]) == pytest.approx(list(base[column]),
                                                  abs=1e-12), column


# --- item 4: none ------------------------------------------------------
def test_none_lets_every_neighbour_count_in_full():
    """B's band-0 list is {A, C, E, RV} (identical to its `bbox` list on this
    city) and the canal severs A, leaving {C, E, RV}. Clinics: B owns 1,
    C 0, E 1, RV 2; pop 200. With `none` every weight is 1."""
    got = variant(ORACULUM, "band0_none")
    assert got.loc["B", "clinic_pcen"] == pytest.approx((1 + 0 + 1 + 2) / 200,
                                                        abs=1e-12)


# --- item 5: pow2 and exp1 at a closed form ----------------------------
def test_rv_and_d_are_the_single_neighbour_settlements():
    settlements = ORACULUM.load_settlements()
    nbrs = apply_barrier(adjacency(settlements, "bbox"), settlements,
                         ORACULUM.load_barriers(), "global")
    assert nbrs["RV"] == {"B"} and nbrs["D"] == {"E"}
    assert [i for i, js in nbrs.items() if len(js) == 1] == ["RV", "D"]
    cent = settlements.set_index("USO_AREA_U").geometry.centroid
    assert cent["RV"].distance(cent["B"]) / 1000 == pytest.approx(1.0,
                                                                  abs=1e-12)
    assert cent["D"].distance(cent["E"]) / 1000 == pytest.approx(1.5,
                                                                 abs=1e-12)


def test_pow2_and_exp1_on_the_two_single_neighbour_settlements():
    """RV's only `code` neighbour is B at 1.0 km and D's is E at 1.5 km, so
    each pcen is one weight. Clinics: RV owns 2 and B owns 1; D owns 0 and
    E owns 1. Both populations are 100."""
    pow2, exp1 = variant(ORACULUM, "pow2"), variant(ORACULUM, "exp1")
    assert pow2.loc["RV", "clinic_pcen"] == pytest.approx(
        (2 + 1 * 1 / (1 + 1.0) ** 2) / 100, abs=1e-12)        # 0.0225
    assert pow2.loc["D", "clinic_pcen"] == pytest.approx(
        (0 + 1 * 1 / (1 + 1.5) ** 2) / 100, abs=1e-12)        # 0.0016
    assert exp1.loc["RV", "clinic_pcen"] == pytest.approx(
        (2 + 1 * math.exp(-1.0)) / 100, abs=1e-12)
    assert exp1.loc["D", "clinic_pcen"] == pytest.approx(
        (0 + 1 * math.exp(-1.5)) / 100, abs=1e-12)


# --- item 6: boundary vs centroid --------------------------------------
def test_a_contact_neighbour_is_undecayed_under_boundary():
    """(a) Oraculum: A's `code` neighbours B and E both share a border with
    it, so each lends its whole clinic count (A owns 2, B 1, E 1; pop 100).
    Messy: O1 and O2 OVERLAP, so their boundary distance is 0 too — O1's
    list is {O2, U}, and U (no population row) is dropped by the
    `nopop_only` scenario and swallowed."""
    assert variant(ORACULUM, "boundary").loc["A", "clinic_pcen"] == \
        pytest.approx((2 + 1 + 1) / 100, abs=1e-12)
    assert variant(MESSY, "boundary").loc["O1", "clinic_pcen"] == \
        pytest.approx((1 + 1) / 600, abs=1e-12)


def test_boundary_beats_centroid_for_an_interlocked_neighbour_on_messy():
    """(b) H and L are DISJOINT but interlocked: 0.131519 km apart at the
    boundary, 1.127237 km apart at the centroids. In `band_small` H's list is
    {L, S, T} (schools 1, 1, 1; H owns none; pop 110), so the whole row is
    weights. Written on H's row: the pin is directional."""
    centroid = variant(MESSY, "band_small").loc["H", "school_pcen"]
    boundary = variant(MESSY, "band_small_boundary").loc["H", "school_pcen"]
    assert boundary == pytest.approx(
        (1 / (1 + 0.13151918984428584) + 1 / (1 + 0.0)
         + 1 / (1 + 0.22360679774997896)) / 110, abs=1e-12)
    assert centroid == pytest.approx(0.013170282557128916, abs=1e-12)
    assert boundary > centroid


def test_boundary_beats_centroid_for_a_large_neighbour_on_oraculum():
    """(b) A's `band_small` list is {B, D, E, RV}; the canal severs D. RV sits
    0.100 km away at the boundary and 1.414214 km away at the centroids,
    while B and E are in contact. Clinics A 2, B 1, E 1, RV 2; pop 100.
    A is canal-flagged, so RV's own row drops A and is not pinned here."""
    centroid = variant(ORACULUM, "band_small").loc["A", "clinic_pcen"]
    boundary = variant(ORACULUM, "band_small_boundary").loc["A",
                                                            "clinic_pcen"]
    assert centroid == pytest.approx(
        (2 + 1 * 1 / (1 + 1.0) + 3 * 1 / (1 + math.sqrt(2))) / 100, abs=1e-12)
    assert boundary == pytest.approx(
        (2 + 1 + 1 + 2 * 1 / (1 + 0.1)) / 100, abs=1e-12)
    assert boundary > centroid


def test_centroid_distance_can_understate_a_gap_too():
    """(c) The opposite pathology, on G's row. M is a two-part MultiPolygon
    whose centroid falls in its own gap, exactly on G's centroid: centroid
    distance 0 (weight 1) but boundary distance 0.45 km (weight 1/1.45).
    G owns a school and M owns a school; pop 50. M's list never contains G
    (M's envelope holds G, but the two do not meet), so the pin is on G."""
    code = scored(MESSY, RULESETS["code"])
    boundary = variant(MESSY, "boundary")
    assert code.loc["G", "school_pcen"] == pytest.approx((1 + 1) / 50,
                                                         abs=1e-12)
    assert boundary.loc["G", "school_pcen"] == pytest.approx(
        (1 + 1 / (1 + 0.45)) / 50, abs=1e-12)
    assert boundary.loc["G", "school_pcen"] < code.loc["G", "school_pcen"]


def test_g_m_enters_the_band_at_the_large_radius():
    assert ("G", "M") in ADDED_BAND_PAIRS["messy"][B2]


# --- the table and the reference agree ---------------------------------
def test_variant_rulesets_are_the_code_base_plus_the_table():
    assert set(VARIANT_RULESETS) == set(VARIANTS)
    for name, spec in VARIANTS.items():
        got = VARIANT_RULESETS[name]
        overridden = set()
        for block, mapping in spec.items():
            for key, value in mapping.items():
                if (block, key) not in VARIANT_KNOBS:
                    continue                      # decay.distance_unit
                knob = VARIANT_KNOBS[(block, key)]
                assert got[knob] == value, (name, knob)
                overridden.add(knob)
        for knob, value in RULESETS["code"].items():
            if knob not in overridden:
                assert got[knob] == value, (name, knob)


# --- the reference rejects the same combinations the config will -------
def test_within_distance_requires_a_radius():
    with pytest.raises(ValueError, match="max_distance_km"):
        adjacency(ORACULUM.load_settlements(), "within_distance")


@pytest.mark.parametrize("rule", ["bbox", "border", "intersects"])
def test_a_radius_without_within_distance_is_rejected(rule):
    with pytest.raises(ValueError, match="max_distance_km"):
        adjacency(ORACULUM.load_settlements(), rule, 0.25)


@pytest.mark.parametrize("kwargs,match", [
    (dict(decay_form="sideways"), "sideways"),
    (dict(decay_form="inverse_power"), "exponent"),
    (dict(decay_form="exponential"), "scale_km"),
    (dict(decay_form="inverse_linear", exponent=2), "exponent"),
    (dict(decay_form="none", scale_km=1.0), "scale_km"),
    (dict(decay_distance="as_the_crow_flies"), "as_the_crow_flies"),
])
def test_compute_city_rejects_missing_or_unused_decay_parameters(kwargs,
                                                                 match):
    call = dict(RULESETS["code"], scenario="baseline", denom="pop")
    call.update(kwargs)
    with pytest.raises(ValueError, match=match):
        compute_city(ORACULUM.load_settlements(), ORACULUM.load_services(),
                     ORACULUM.load_barriers(), **call)


def test_variants_module_imports_nothing_at_all():
    """Both sides read this table, so it must reach neither of them — not
    `delhi_psi` (the reference's INDEPENDENCE RULE) and not
    `tests.reference_impl` (which imports it). It is data, so the check is
    simply that it has no import statement whatsoever."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent / "variants.py").read_text()
    tree = ast.parse(source)
    assert not [node for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))], \
        "tests/variants.py is a data table: it imports nothing"
