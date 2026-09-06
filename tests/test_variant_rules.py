"""Hand-derivable pins for the injectable parameters (spec § 4.1, items 1-6).

Every number here was derived from the fixture geometry and checked against
the fixtures; the arithmetic is spelled out in each test so a reader can
re-derive it on paper. `within_distance` is POLYGON-TO-POLYGON distance, so
the radii are judged against boundary distances, never the worksheet's
centroid distances.

Reference side only: nothing in this file imports delhi_psi.
"""
import itertools
import math

import pytest

from tests.cities import CITIES, MESSY, ORACULUM
from tests.reference_impl import (
    RULESETS, VARIANT_KNOBS, VARIANT_RULESETS, adjacency, apply_barrier,
    compute_city, partial_weights, shared_amounts,
)
from tests.variants import (
    ADDED_BAND_PAIRS, BAND_RADII_KM, EXPECTED_BAND_PAIRS, VARIANTS,
)

# B0, B1, B2 are the three 3D pinned in a distance GAP (no pair on the
# boundary); B3, B4, B5 are DEL-55's sweep radii, ON the `<=` boundary. Named
# explicitly rather than by unpacking BAND_RADII_KM, which now holds six
# values and several tests below still want the original three by name.
B0, B1, B2 = 0.0, 0.25, 0.75
B3, B4, B5 = 1.0, 5.0, 10.0
assert BAND_RADII_KM == (B0, B1, B2, B3, B4, B5)


def undirected(nbrs):
    return {tuple(sorted((i, j))) for i, js in nbrs.items() for j in js}


def band(city, km):
    return adjacency(city.load_settlements(), "within_distance", km)


def adjacency_pairs(gdf, rule, max_distance_km=None):
    return undirected(adjacency(gdf, rule, max_distance_km))


def pair_of(gdf, i, j):
    ids = gdf["USO_AREA_U"]
    return tuple(sorted((ids.iloc[i], ids.iloc[j])))


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
    for lower, upper in zip(BAND_RADII_KM, BAND_RADII_KM[1:]):
        assert pairs[upper] - pairs[lower] == \
            ADDED_BAND_PAIRS[city.name][upper], (lower, upper)


# Measured on the committed fixtures: these pairs are at EXACTLY the radius.
# Oraculum A-C and E-RV, messy M-U — all three at 1000.000000 m.
BOUNDARY_PAIRS_1KM = {"oraculum": 2, "messy": 1}


@pytest.mark.parametrize("city", CITIES, ids=lambda c: c.name)
def test_a_pair_exactly_at_the_radius_is_a_neighbour(city):
    """`within_distance` is `<=`, not `<`. 3D never had to say so — 0.25 and
    0.75 km were chosen to sit in a gap of both cities' distance lists. The
    sweep's 1 km radius cannot: it lands exactly on the boundary. Pin the
    inclusive reading, and pin how many pairs depend on it.
    """
    gdf = city.load_settlements()
    exact = [(i, j) for i, j in itertools.combinations(range(len(gdf)), 2)
             if gdf.geometry.iloc[i].distance(gdf.geometry.iloc[j]) == 1000.0]
    assert len(exact) == BOUNDARY_PAIRS_1KM[city.name]
    pairs = adjacency_pairs(gdf, rule="within_distance", max_distance_km=1.0)
    for i, j in exact:
        assert pair_of(gdf, i, j) in pairs, "a pair AT the radius must be in"


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


# --- DEL-48: partial_weighted on Oraculum (spec § 6.1) -----------------
# The canal is the segment [25, 475] at y = 1000, lying inside the 500 m
# A-D edge x in [0, 500]. Its 5 m round-capped buffer covers [20, 480], so
# L_blocked = 460 and w_AD = 1 - 460/500 = 0.08. (The memo's 0.1 ignored the
# buffer; buffer_m -> 0 would give it, and 0 is refused. Spec § 12 item 4.)
PARTIAL_5M = dict(RULESETS["code"], barrier_rule="partial_weighted",
                  barrier_buffer_m=5.0)
W_AD = 0.08
# A and D centroids are (500, 1500) and (0, 500): sqrt(5)/2 km apart.
D_AD_KM = math.sqrt(5) / 2
W_DECAY_AD = 1 / (1 + D_AD_KM)          # 0.4721359549995794
W_15 = 1 / 2.5                          # decay at 1.5 km (D-E, A-E via E)
W_SQRT2 = 1 / (1 + math.sqrt(2))        # decay at 1000*sqrt(2) m
W_HALF = 0.5                            # decay at 1000 m


def partial_5m_weights(city=ORACULUM, buffer_m=5.0):
    settlements = city.load_settlements()
    return partial_weights(adjacency(settlements, "bbox"), settlements,
                           city.load_barriers(), buffer_m)


def test_only_the_ad_edge_is_partially_blocked_on_oraculum():
    """Every other bbox pair's shared boundary is at least 20 m from the
    canal's ends (A-E starts at x = 500, the buffer stops at 480), so the
    canal produces exactly one fractional weight — in both directions."""
    weights = partial_5m_weights()
    assert weights[("A", "D")] == pytest.approx(W_AD, abs=1e-12)
    assert weights[("A", "D")] == weights[("D", "A")]
    fractional = {pair for pair, w in weights.items() if w != 1.0}
    assert fractional == {("A", "D"), ("D", "A")}


def test_a_smaller_buffer_blocks_less_of_the_same_edge():
    """The buffer made visible: at 1 m the canal blocks [24, 476] = 452 m,
    so w = 0.096. The limit as buffer_m -> 0 is the memo's 0.1, which is
    never a pin because buffer_m must be > 0 (spec § 12 item 5)."""
    assert partial_5m_weights(buffer_m=1.0)[("A", "D")] == pytest.approx(
        0.096, abs=1e-12)


def test_partial_weighted_prunes_nothing_on_oraculum():
    """No weight is 0, so the lists are the bbox lists — and A and D are back
    in everyone's list, because the global rule's flag-based severing is
    gone."""
    settlements = ORACULUM.load_settlements()
    got = apply_barrier(adjacency(settlements, "bbox"), settlements,
                        ORACULUM.load_barriers(), "partial_weighted", 5.0)
    assert got == {"A": {"B", "D", "E"}, "B": {"A", "C", "E", "RV"},
                   "C": {"B", "E", "IND"}, "RV": {"B"}, "D": {"A", "E"},
                   "E": {"A", "B", "C", "D", "IND"}, "IND": {"C", "E"}}


def test_partial_5m_pcen_anchors_on_oraculum():
    """Every row derived on paper from the geometry (spec § 6.1). D's clinic
    row is the one that shows the rule: A lends 2 clinics at 8% weight over
    a sqrt(5)/2 km centroid gap, and E lends 1 at 1.5 km, undiscounted."""
    got = scored(ORACULUM, PARTIAL_5M)
    assert got.loc["D", "clinic_pcen"] == pytest.approx(
        (0 + W_AD * 2 * W_DECAY_AD + 1 * W_15) / 100, abs=1e-12)
    assert got.loc["D", "school_pcen"] == pytest.approx(
        (1 + W_AD * 1 * W_DECAY_AD + 1 * W_15) / 100, abs=1e-12)
    assert got.loc["A", "school_pcen"] == pytest.approx(
        (1 + W_AD * 1 * W_DECAY_AD + 1 * W_SQRT2) / 100, abs=1e-12)
    # A's clinic row is UNCHANGED by the weight: D owns no clinic.
    assert got.loc["A", "clinic_pcen"] == pytest.approx(
        (2 + 1 * W_HALF + 1 * W_SQRT2) / 100, abs=1e-12)
    # roads are decayed under the `code` base: A owns 0.75 km
    assert got.loc["D", "road_pcen"] == pytest.approx(
        (0 + W_AD * 0.75 * W_DECAY_AD + 0.75 * W_15) / 100, abs=1e-12)


def test_partial_5m_restores_the_links_the_global_rule_severed():
    """B and E get A back — under `code` the global rule dropped every link
    INTO a flagged settlement, so B's clinic row was 0.0125 and E's was the
    code value. Under partial_weighted they are the `ideal` values, because
    no barrier touches those boundaries at all."""
    got = scored(ORACULUM, PARTIAL_5M)
    assert got.loc["B", "clinic_pcen"] == pytest.approx(0.0175, abs=1e-12)
    assert got.loc["E", "clinic_pcen"] == pytest.approx(
        (1 + 2 * W_SQRT2 + 1 * W_HALF) / 300, abs=1e-12)


def test_partial_5m_leaves_no_constant_column_on_oraculum():
    """The invariants guard refuses a degenerate min-max group, and DEL-54's
    guard raises on one. Both denominators, every service: checked here
    BEFORE the fixture regeneration step depends on it."""
    for denom in ("pop", "popdensity"):
        got = scored(ORACULUM, PARTIAL_5M, denom)
        for column in [c for c in got.columns if c.endswith("_pcen")]:
            assert got[column].max() > got[column].min(), (denom, column)


def test_the_partial_5m_variant_is_the_code_base_plus_the_barrier_rule():
    """The table, the knob map and the hand anchors are one thing: the
    variant's rule-set must BE the dict the § 6.1 anchors were derived
    under."""
    assert VARIANT_RULESETS["partial_5m"] == PARTIAL_5M


def test_partial_5m_is_degenerate_on_the_messy_city():
    """No barriers, so every weight is 1 and the rows are the `code` base's
    — stated, like `boundary` on Oraculum, so the CSV rows are not mistaken
    for a proof they are not."""
    base = scored(MESSY, RULESETS["code"])
    got = variant(MESSY, "partial_5m")
    for column in base.columns:
        assert list(got[column]) == pytest.approx(list(base[column]),
                                                  abs=1e-12), column


# --- DEL-20: overlap lending on the messy city (spec § 3.1, § 6.2) -----
# O1 is _rect(10000, 0, 11000, 1000) and O2 is _rect(10800, 0, 11800, 1000),
# so they overlap in x in [10800, 11000]. The ONE clinic at (10900, 500) is
# strictly inside BOTH, so it is O1's own AND O2's own — Raj's ratified
# counting half, which does not move — and under `whole` it is ALSO lent
# from each to the other, which is the half this switch removes. Centroids
# (10500, 500) and (11300, 500) are 0.8 km apart, so the decay is 1/1.8.
OVERLAP_OUTSIDE = dict(RULESETS["code"], overlap_lending="outside_receiver")
W_O1O2 = 1 / 1.8


def test_the_overlap_clinic_is_lent_back_under_whole():
    """Today's arithmetic, stated so the switch has something to move: the
    single physical clinic reaches O1 twice — once as its own, once decayed
    from O2 — and reaches O2 twice as well."""
    got = scored(MESSY, RULESETS["code"])
    assert got.loc["O1", "clinic_pcen"] == pytest.approx(
        (1 + 1 * W_O1O2) / 600, abs=1e-12)
    assert got.loc["O2", "clinic_pcen"] == pytest.approx(
        (1 + 1 * W_O1O2) / 700, abs=1e-12)


def test_outside_receiver_lends_only_what_is_not_already_inside():
    """|S_j \\ S_i| is 0 for the clinic: O2's only clinic is already inside
    O1, so O1 gets it once. The OWN counts do not move — Raj's ratified half
    is untouched, and that is what makes this a lending rule and not a
    counting rule."""
    got = scored(MESSY, OVERLAP_OUTSIDE)
    assert got.loc["O1", "clinic_count"] == 1
    assert got.loc["O2", "clinic_count"] == 1
    assert got.loc["O1", "clinic_pcen"] == pytest.approx(1 / 600, abs=1e-12)
    assert got.loc["O2", "clinic_pcen"] == pytest.approx(1 / 700, abs=1e-12)


def test_a_neighbours_service_outside_the_overlap_is_lent_in_full():
    """The clinic moves and the school does not: O2's school at (11400, 500)
    lies outside O1, so |S_j \\ S_i| == |S_j| and O1's school row is exactly
    the `whole` value — asserted with `==`, because a pair with nothing
    shared must not go anywhere near the arithmetic. O1's own police point
    is not in O2 either."""
    whole = scored(MESSY, RULESETS["code"])
    got = scored(MESSY, OVERLAP_OUTSIDE)
    assert got.loc["O1", "school_pcen"] == whole.loc["O1", "school_pcen"]
    assert got.loc["O1", "school_pcen"] == pytest.approx(
        (0 + 1 * W_O1O2) / 600, abs=1e-12)
    assert got.loc["O1", "police_pcen"] == pytest.approx(1 / 600, abs=1e-12)


def test_overlap_outside_is_degenerate_on_oraculum():
    """No overlapping polygons and no point inside two settlements, so the
    shared structure is EMPTY and every row equals the `code` base — stated,
    like `boundary` on Oraculum and `partial_5m` on the messy city, so the
    CSV rows are never mistaken for a proof they are not."""
    base = scored(ORACULUM, RULESETS["code"])
    got = scored(ORACULUM, OVERLAP_OUTSIDE)
    for column in base.columns:
        assert list(got[column]) == pytest.approx(list(base[column]),
                                                  abs=1e-12), column


def test_the_shared_structure_is_sparse_and_symmetric():
    """One entry, both orders, on the one pair that shares anything; nothing
    at all on Oraculum, and nothing for the road (no road row crosses the
    O1/O2 overlap). Every other pair has |S_j \\ S_i| == |S_j| and is never
    computed or stored — that is the cost argument, made checkable."""
    for city, expected in ((ORACULUM, {}),
                           (MESSY, {("O1", "O2"): 1, ("O2", "O1"): 1})):
        settlements = city.load_settlements()
        nbrs = apply_barrier(adjacency(settlements, "bbox"), settlements,
                             city.load_barriers(), "global")
        got = shared_amounts(nbrs, settlements, city.load_services())
        assert got["clinic"] == expected, city.name
        assert all(not table for svc, table in got.items()
                   if svc != "clinic"), city.name


def test_no_service_column_is_constant_under_overlap_outside():
    """The invariants guard refuses a degenerate min-max group and DEL-54's
    guard raises on one. Both cities, both denominators, every service:
    checked HERE, before Task 5's fixture regeneration depends on it."""
    for city in CITIES:
        for denom in ("pop", "popdensity"):
            got = scored(city, OVERLAP_OUTSIDE, denom)
            for column in [c for c in got.columns if c.endswith("_pcen")]:
                assert got[column].max() > got[column].min(), (city.name,
                                                               denom, column)


def test_an_unknown_lending_value_raises():
    """An unimplemented value must RAISE — the mapped-knob test relies on
    it, and so does the loader's enum table being the only source of
    values."""
    with pytest.raises(ValueError, match="overlap lending"):
        scored(MESSY, dict(RULESETS["code"], overlap_lending="halves"))


def test_the_overlap_outside_variant_is_the_code_base_plus_the_lending_rule():
    """The table, the knob map and the hand pins are one thing: the
    variant's rule-set must BE the dict the § 6.2 pins were derived under."""
    assert VARIANT_RULESETS["overlap_outside"] == OVERLAP_OUTSIDE


def test_partial_5m_outside_is_each_of_its_halves_on_the_city_that_shows_it():
    """No fixture city has a barrier across an overlap, so the combined
    variant is `partial_5m` on Oraculum (no overlaps) and `overlap_outside`
    on the messy city (no barriers). Its job in the table is to prove the
    two kwargs are accepted together, not to add a third number."""
    both = variant(ORACULUM, "partial_5m_outside")
    barrier_only = variant(ORACULUM, "partial_5m")
    for column in barrier_only.columns:
        assert list(both[column]) == pytest.approx(
            list(barrier_only[column]), abs=1e-12), ("oraculum", column)
    both = variant(MESSY, "partial_5m_outside")
    overlap_only = variant(MESSY, "overlap_outside")
    for column in overlap_only.columns:
        assert list(both[column]) == pytest.approx(
            list(overlap_only[column]), abs=1e-12), ("messy", column)
